/* deepseek_v4.c — el motor: carga el checkpoint de 284B y genera texto.
 *
 * Monta las primitivas ya validadas (dsv4_*.h) a escala real: 43 capas, 72.317
 * tensores, 156 GiB en disco.
 *
 * REPARTO DE MEMORIA, que es la decisión de diseño central:
 *
 *   residente en RAM   8,67 GiB   atención, normas, routers, expertos
 *                                 compartidos, embeddings y lm_head — EN SU
 *                                 FORMATO NATIVO, sin dequantizar
 *   por streaming    137,1 GiB   los 11.008 expertos rutados, leídos del NVMe
 *                                 cuando el router los pide, con caché LRU
 *
 * Dequantizar el denso a f32 serían 26,8 GiB y no cabría; con descriptores
 * (`dsv4_weight.h`) el matmul lee el formato original y nunca materializa la
 * matriz. Es la misma decisión que toma colibrì con su struct `QT`.
 *
 * Build:  make -C port engine
 * Uso:    ./deepseek_v4 <dir_modelo> "prompt" [n_tokens]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "compat.h"
#include "json.h"
#include "st.h"
#define DSV4_WITH_MXFP4
#include "quant.h"
#include "tok.h"

#include "dsv4_fp8.h"
#include "dsv4_weight.h"
#include "dsv4_attn.h"
#include "dsv4_moe.h"
#include "dsv4_block.h"
#include "dsv4_decode.h"

static double now_s(void) {
#ifdef _WIN32
    return (double)clock() / CLOCKS_PER_SEC;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}

/* ---------------------------------------------------------------------------
 * Tier de expertos por streaming.
 *
 * Un experto son 3 matrices MXFP4 + sus escalas = 13,4 MB. No caben los 11.008
 * (137 GiB), así que se leen bajo demanda de los shards y se guardan en una
 * caché LRU por capa. Los offsets se resuelven una vez al cargar: en decode no
 * se busca por nombre, se hace `pread` a un offset conocido.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint8_t *buf[6];        /* w1,w1s, w2,w2s, w3,w3s */
    int layer, expert;
    uint64_t used;
} ExpSlot;

typedef struct {
    shards *S;
    int n_layers, n_experts, cap, inter, dim;
    char (*names)[6][96];   /* [layer*n_experts] -> los 6 nombres de tensor */
    ExpSlot *slots;
    int nslot;
    uint64_t clock, hits, miss;
    uint64_t bytes;
} ExpertTier;

static void tier_init(ExpertTier *T, shards *S, int n_layers, int n_experts,
                      int inter, int dim, int cap)
{
    memset(T, 0, sizeof *T);
    T->S = S; T->n_layers = n_layers; T->n_experts = n_experts;
    T->inter = inter; T->dim = dim; T->cap = cap;
    T->slots = calloc((size_t)cap, sizeof(ExpSlot));
    for (int i = 0; i < cap; i++) { T->slots[i].layer = -1; T->slots[i].expert = -1; }
    T->names = malloc((size_t)n_layers * n_experts * sizeof(*T->names));
    static const char *mats[3] = { "w1", "w2", "w3" };
    for (int l = 0; l < n_layers; l++)
        for (int e = 0; e < n_experts; e++) {
            char (*row)[96] = T->names[(size_t)l * n_experts + e];
            for (int k = 0; k < 3; k++) {
                snprintf(row[k * 2],     96, "layers.%d.ffn.experts.%d.%s.weight", l, e, mats[k]);
                snprintf(row[k * 2 + 1], 96, "layers.%d.ffn.experts.%d.%s.scale",  l, e, mats[k]);
            }
        }
}

/* Devuelve los descriptores del experto, leyéndolo si no está en caché. */
static void tier_get(ExpertTier *T, int layer, int e, DsV4W *w1, DsV4W *w2, DsV4W *w3)
{
    int hit = -1;
    for (int i = 0; i < T->nslot; i++)
        if (T->slots[i].layer == layer && T->slots[i].expert == e) { hit = i; break; }

    if (hit < 0) {
        T->miss++;
        if (T->nslot < T->cap) hit = T->nslot++;
        else {
            hit = 0;
            for (int i = 1; i < T->nslot; i++)
                if (T->slots[i].used < T->slots[hit].used) hit = i;
        }
        ExpSlot *s = &T->slots[hit];
        char (*row)[96] = T->names[(size_t)layer * T->n_experts + e];
        for (int k = 0; k < 6; k++) {
            const int64_t nb = st_nbytes(T->S, row[k]);
            s->buf[k] = realloc(s->buf[k], (size_t)nb);
            st_read_raw(T->S, row[k], s->buf[k], 0);
            T->bytes += (uint64_t)nb;
        }
        s->layer = layer; s->expert = e;
    } else T->hits++;

    ExpSlot *s = &T->slots[hit];
    s->used = ++T->clock;
    *w1 = dsv4_w_mxfp4(s->buf[0], s->buf[1], T->inter, T->dim);
    *w2 = dsv4_w_mxfp4(s->buf[2], s->buf[3], T->dim,   T->inter);
    *w3 = dsv4_w_mxfp4(s->buf[4], s->buf[5], T->inter, T->dim);
}

/* Adaptador: el MoE pide expertos por callback y no sabe nada de shards. */
static void tier_fetch(void *ctx, int layer, int e,
                       DsV4W *w1, DsV4W *w2, DsV4W *w3) {
    tier_get((ExpertTier *)ctx, layer, e, w1, w2, w3);
}

/* ---------------------------------------------------------------------------
 * El modelo
 * ------------------------------------------------------------------------- */
typedef struct {
    shards S;
    Tok tok;
    ExpertTier tier;

    int n_layers, dim, vocab, hc, n_hash, sinkhorn_iters;
    float norm_eps, hc_eps;
    int *ratios;

    DsV4BlockCfg *cfg;      /* [n_layers] */
    DsV4BlockW   *w;        /* [n_layers] */
    int32_t *tid2eid;       /* [vocab * topk], sólo si hay capas hash */

    DsV4W embed, head;
    const float *final_norm, *hc_head_fn, *hc_head_base, *hc_head_scale;
    float *freqs;           /* [max_seq, rd/2, 2] */
    int max_seq;
} Model;

/* --- helpers de carga --------------------------------------------------- */
static void *raw_of(shards *S, const char *nm) {
    const int64_t n = st_nbytes(S, nm);
    void *p = malloc((size_t)n);
    st_read_raw(S, nm, p, 0);
    return p;
}
static float *vec_f32(shards *S, const char *nm, int n) {
    float *out = malloc((size_t)n * sizeof(float));
    st_tensor *t = st_find(S, nm);
    if (!t) { fprintf(stderr, "falta %s\n", nm); exit(1); }
    if (t->dtype == 2) { st_read_f32(S, nm, out, 0); return out; }   /* F32 */
    uint8_t *r = raw_of(S, nm);                                       /* BF16 */
    DsV4W v = dsv4_w_bf16(r, 1, n);
    dsv4_vec_f32(out, &v, n);
    free(r);
    return out;
}
/* matriz cuantizada: FP8 si tiene .scale, BF16 si no */
static DsV4W mat_of(shards *S, const char *base, int O, int I) {
    char nw[192], ns[192];
    snprintf(nw, sizeof nw, "%s.weight", base);
    snprintf(ns, sizeof ns, "%s.scale", base);
    uint8_t *wb = raw_of(S, nw);
    if (st_has(S, ns)) return dsv4_w_fp8b(wb, raw_of(S, ns), O, I);
    return dsv4_w_bf16(wb, O, I);
}

static void model_load(Model *M, const char *dir) {
    memset(M, 0, sizeof *M);
    double t0 = now_s();
    st_init(&M->S, dir);

    /* --- config.json ---------------------------------------------------- */
    char cfgp[1024];
    snprintf(cfgp, sizeof cfgp, "%s/config.json", dir);
    FILE *f = fopen(cfgp, "rb");
    if (!f) { fprintf(stderr, "no encuentro %s\n", cfgp); exit(1); }
    fseek(f, 0, SEEK_END); long cn = ftell(f); fseek(f, 0, SEEK_SET);
    char *cbuf = malloc((size_t)cn + 1);
    if (fread(cbuf, 1, (size_t)cn, f) != (size_t)cn) { fprintf(stderr, "config corto\n"); exit(1); }
    cbuf[cn] = 0; fclose(f);
    char *arena = NULL;
    jval *C = json_parse(cbuf, &arena);
#define GI(k) ((int)json_get(C,k)->num)
#define GF(k) ((float)json_get(C,k)->num)
    M->n_layers = GI("num_hidden_layers");
    M->dim      = GI("hidden_size");
    M->vocab    = GI("vocab_size");
    M->hc       = GI("hc_mult");
    M->n_hash   = GI("num_hash_layers");
    M->sinkhorn_iters = GI("hc_sinkhorn_iters");
    M->norm_eps = GF("rms_norm_eps");
    M->hc_eps   = GF("hc_eps");
    const int heads = GI("num_attention_heads"), hd = GI("head_dim");
    const int rd = GI("qk_rope_head_dim"), q_lora = GI("q_lora_rank");
    const int groups = GI("o_groups"), o_lora = GI("o_lora_rank");
    const int win = GI("sliding_window");
    const int n_exp = GI("n_routed_experts"), topk = GI("num_experts_per_tok");
    const int inter = GI("moe_intermediate_size");
    const int i_heads = GI("index_n_heads"), i_hd = GI("index_head_dim");
    const int i_topk = GI("index_topk");
    const float route_scale = GF("routed_scaling_factor");
    const float swiglu = GF("swiglu_limit");
    jval *rs = json_get(C, "rope_scaling");
    const double base = json_get(C, "rope_theta")->num;
    const double factor = json_get(rs, "factor")->num;
    const int orig = (int)json_get(rs, "original_max_position_embeddings")->num;
    const double bf = json_get(rs, "beta_fast")->num, bs = json_get(rs, "beta_slow")->num;
    jval *cr = json_get(C, "compress_ratios");
    M->ratios = malloc((size_t)M->n_layers * sizeof(int));
    for (int i = 0; i < M->n_layers; i++) M->ratios[i] = (int)cr->kids[i]->num;
#undef GI
#undef GF

    printf("config: %d capas, dim %d, %d cabezas x %d, %d expertos top-%d\n",
           M->n_layers, M->dim, heads, hd, n_exp, topk);

    /* --- freqs_cis con YaRN --------------------------------------------- */
    M->max_seq = 2048;                 /* suficiente para prompts de prueba */
    M->freqs = malloc((size_t)M->max_seq * (rd / 2) * 2 * sizeof(float));
    dsv4_precompute_freqs(M->freqs, rd, M->max_seq, orig, base, factor, bf, bs);

    /* --- globales -------------------------------------------------------- */
    M->embed = mat_of(&M->S, "embed", M->vocab, M->dim);
    M->head  = mat_of(&M->S, "head",  M->vocab, M->dim);
    M->final_norm    = vec_f32(&M->S, "norm.weight", M->dim);
    M->hc_head_fn    = (float *)raw_of(&M->S, "hc_head_fn");
    M->hc_head_base  = (float *)raw_of(&M->S, "hc_head_base");
    M->hc_head_scale = (float *)raw_of(&M->S, "hc_head_scale");

    /* --- por capa -------------------------------------------------------- */
    M->cfg = calloc((size_t)M->n_layers, sizeof(DsV4BlockCfg));
    M->w   = calloc((size_t)M->n_layers, sizeof(DsV4BlockW));
    char nm[192];
    for (int L = 0; L < M->n_layers; L++) {
        DsV4BlockCfg *c = &M->cfg[L];
        DsV4BlockW *w = &M->w[L];
        const int ratio = M->ratios[L];

        c->hc = M->hc; c->sinkhorn_iters = M->sinkhorn_iters;
        c->hc_eps = M->hc_eps; c->norm_eps = M->norm_eps;
        c->attn.dim = M->dim; c->attn.q_lora = q_lora; c->attn.heads = heads;
        c->attn.hd = hd; c->attn.rd = rd; c->attn.groups = groups;
        c->attn.o_lora = o_lora; c->attn.win = win; c->attn.eps = M->norm_eps;
        c->attn.scale = 1.0f / sqrtf((float)hd);
        c->attn.ratio = ratio;
        c->moe.dim = M->dim; c->moe.n_experts = n_exp; c->moe.topk = topk;
        c->moe.inter = inter; c->moe.route_scale = route_scale;
        c->moe.swiglu_limit = swiglu; c->moe.hash = (L < M->n_hash);

#define B(fmt, ...) (snprintf(nm, sizeof nm, fmt, __VA_ARGS__), nm)
        w->attn.wq_a = mat_of(&M->S, B("layers.%d.attn.wq_a", L), q_lora, M->dim);
        w->attn.wq_b = mat_of(&M->S, B("layers.%d.attn.wq_b", L), heads * hd, q_lora);
        w->attn.wkv  = mat_of(&M->S, B("layers.%d.attn.wkv",  L), hd, M->dim);
        w->attn.wo_a = mat_of(&M->S, B("layers.%d.attn.wo_a", L), groups * o_lora,
                              heads * hd / groups);
        w->attn.wo_b = mat_of(&M->S, B("layers.%d.attn.wo_b", L), M->dim, groups * o_lora);
        w->attn.q_norm    = vec_f32(&M->S, B("layers.%d.attn.q_norm.weight", L), q_lora);
        w->attn.kv_norm   = vec_f32(&M->S, B("layers.%d.attn.kv_norm.weight", L), hd);
        w->attn.attn_sink = vec_f32(&M->S, B("layers.%d.attn.attn_sink", L), heads);
        w->attn.freqs = M->freqs;

        if (ratio) {
            const int coff = (ratio == 4) ? 2 : 1;
            w->attn.c_wkv   = mat_of(&M->S, B("layers.%d.attn.compressor.wkv", L),
                                     coff * hd, M->dim);
            w->attn.c_wgate = mat_of(&M->S, B("layers.%d.attn.compressor.wgate", L),
                                     coff * hd, M->dim);
            w->attn.c_norm  = vec_f32(&M->S, B("layers.%d.attn.compressor.norm.weight", L), hd);
            w->attn.c_ape   = (float *)raw_of(&M->S, B("layers.%d.attn.compressor.ape", L));
            if (ratio == 4) {
                c->attn.i_heads = i_heads; c->attn.i_hd = i_hd; c->attn.i_topk = i_topk;
                c->attn.i_scale = 1.0f / sqrtf((float)i_hd);
                w->attn.i_wkv   = mat_of(&M->S, B("layers.%d.attn.indexer.compressor.wkv", L),
                                         coff * i_hd, M->dim);
                w->attn.i_wgate = mat_of(&M->S, B("layers.%d.attn.indexer.compressor.wgate", L),
                                         coff * i_hd, M->dim);
                w->attn.i_norm  = vec_f32(&M->S, B("layers.%d.attn.indexer.compressor.norm.weight", L), i_hd);
                w->attn.i_ape   = (float *)raw_of(&M->S, B("layers.%d.attn.indexer.compressor.ape", L));
                w->attn.i_wq_b  = mat_of(&M->S, B("layers.%d.attn.indexer.wq_b", L),
                                         i_heads * i_hd, q_lora);
                w->attn.i_wproj = mat_of(&M->S, B("layers.%d.attn.indexer.weights_proj", L),
                                         i_heads, M->dim);
            }
        }

        w->attn_norm = vec_f32(&M->S, B("layers.%d.attn_norm.weight", L), M->dim);
        w->ffn_norm  = vec_f32(&M->S, B("layers.%d.ffn_norm.weight", L), M->dim);
        w->hc_attn_fn    = (float *)raw_of(&M->S, B("layers.%d.hc_attn_fn", L));
        w->hc_attn_base  = (float *)raw_of(&M->S, B("layers.%d.hc_attn_base", L));
        w->hc_attn_scale = (float *)raw_of(&M->S, B("layers.%d.hc_attn_scale", L));
        w->hc_ffn_fn     = (float *)raw_of(&M->S, B("layers.%d.hc_ffn_fn", L));
        w->hc_ffn_base   = (float *)raw_of(&M->S, B("layers.%d.hc_ffn_base", L));
        w->hc_ffn_scale  = (float *)raw_of(&M->S, B("layers.%d.hc_ffn_scale", L));

        w->moe.gate_w = mat_of(&M->S, B("layers.%d.ffn.gate", L), n_exp, M->dim);
        if (c->moe.hash) {
            /* la tabla es la misma para todas las capas hash: se carga una vez */
            if (!M->tid2eid) {
                snprintf(nm, sizeof nm, "layers.%d.ffn.gate.tid2eid", L);
                const int64_t nb = st_nbytes(&M->S, nm);
                int64_t *r = malloc((size_t)nb);
                st_read_raw(&M->S, nm, r, 0);
                const int64_t n64 = nb / 8;
                M->tid2eid = malloc((size_t)n64 * sizeof(int32_t));
                for (int64_t i = 0; i < n64; i++) M->tid2eid[i] = (int32_t)r[i];
                free(r);
            }
            w->moe.tid2eid = M->tid2eid;
        } else {
            w->moe.gate_bias = vec_f32(&M->S, B("layers.%d.ffn.gate.bias", L), n_exp);
        }
        w->moe.s_w1 = mat_of(&M->S, B("layers.%d.ffn.shared_experts.w1", L), inter, M->dim);
        w->moe.s_w2 = mat_of(&M->S, B("layers.%d.ffn.shared_experts.w2", L), M->dim, inter);
        w->moe.s_w3 = mat_of(&M->S, B("layers.%d.ffn.shared_experts.w3", L), inter, M->dim);
#undef B
        if ((L + 1) % 8 == 0 || L == M->n_layers - 1)
            printf("\r  capas cargadas: %d/%d", L + 1, M->n_layers), fflush(stdout);
    }
    printf("\n");

    /* --- tier de expertos ------------------------------------------------ */
    const int cache = 384;             /* ~5,1 GB de caché */
    tier_init(&M->tier, &M->S, M->n_layers, n_exp, inter, M->dim, cache);
    /* Cada capa pide sus expertos al tier por callback: el MoE no sabe nada de
     * shards ni de política de caché. */
    for (int L = 0; L < M->n_layers; L++) {
        M->w[L].moe.fetch = tier_fetch;
        M->w[L].moe.fetch_ctx = &M->tier;
        M->w[L].moe.layer = L;
    }
    printf("expertos: streaming con cache de %d slots de %d totales\n",
           cache, M->n_layers * n_exp);
    printf("cargado en %.1f s\n\n", now_s() - t0);
}

/* ---------------------------------------------------------------------------
 * Forward de un token (decode) o de un prompt (prefill).
 * `h` es el stream residual [hc, dim] de UNA posición (batch 1).
 * ------------------------------------------------------------------------- */
static void model_layer_decode(Model *M, int L, DsV4AttnState *st,
                               const float *hin, int pos, int32_t tokid,
                               float *hout)
{
    DsV4BlockCfg *c = &M->cfg[L];
    DsV4BlockW *w = &M->w[L];
    const int dim = M->dim, hc = M->hc;

    float post[DSV4_MAX_HC], comb[DSV4_MAX_HC * DSV4_MAX_HC];
    float *coll = malloc((size_t)dim * sizeof(float));
    float *sub  = malloc((size_t)dim * sizeof(float));
    float *mid  = malloc((size_t)hc * dim * sizeof(float));
    float *tmp  = malloc((size_t)dim * sizeof(float));

    /* --- atención --- */
    dsv4_hc_pre(hin, w->hc_attn_fn, w->hc_attn_scale, w->hc_attn_base,
                hc, dim, c->sinkhorn_iters, c->hc_eps, c->norm_eps,
                coll, post, comb);
    dsv4_rmsnorm(tmp, coll, w->attn_norm, dim, c->norm_eps);
    for (int d = 0; d < dim; d++) coll[d] = dsv4_to_bf16(tmp[d]);
    dsv4_attention_decode(&c->attn, &w->attn, st, coll, pos, sub);
    dsv4_hc_expand(mid, sub, post, comb, hin, hc, dim);

    /* --- MoE --- */
    dsv4_hc_pre(mid, w->hc_ffn_fn, w->hc_ffn_scale, w->hc_ffn_base,
                hc, dim, c->sinkhorn_iters, c->hc_eps, c->norm_eps,
                coll, post, comb);
    dsv4_rmsnorm(tmp, coll, w->ffn_norm, dim, c->norm_eps);
    for (int d = 0; d < dim; d++) coll[d] = dsv4_to_bf16(tmp[d]);
    /* el id del token sólo lo usan las capas hash, pero se pasa siempre */
    dsv4_moe_forward(&c->moe, &w->moe, coll, &tokid, 1, sub);
    dsv4_hc_expand(hout, sub, post, comb, mid, hc, dim);

    free(coll); free(sub); free(mid); free(tmp);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *dir = (argc > 1) ? argv[1]
        : "C:\\Users\\Gus\\ai\\models\\DeepSeek-V4-Flash-0731";
    const char *prompt = (argc > 2) ? argv[2] : "hola";
    const int ngen = (argc > 3) ? atoi(argv[3]) : 8;

    Model M;
    model_load(&M, dir);

    char tokp[1024];
    snprintf(tokp, sizeof tokp, "%s/tokenizer.json", dir);
    tok_load(&M.tok, tokp);

    /* BOS delante: el checkpoint no trae plantilla de chat, pero sí
     * `<|begin_of_sentence|>` (id 0), y el modelo se entrenó viéndolo. */
    int ids[1024];
    ids[0] = 0;
    int n = 1 + tok_encode(&M.tok, prompt, (int)strlen(prompt), ids + 1, 1000);
    printf("prompt: \"%s\" -> %d tokens (con BOS)\n", prompt, n);

    /* estado por capa */
    DsV4AttnState *st = calloc((size_t)M.n_layers, sizeof(DsV4AttnState));
    for (int L = 0; L < M.n_layers; L++)
        dsv4_state_init_full(&st[L], 1, M.cfg[L].attn.hd, M.cfg[L].attn.win,
                             M.max_seq, M.cfg[L].attn.ratio, M.cfg[L].attn.i_hd);

    const int dim = M.dim, hc = M.hc;
    float *h  = malloc((size_t)hc * dim * sizeof(float));
    float *h2 = malloc((size_t)hc * dim * sizeof(float));
    float *logits = malloc((size_t)M.vocab * sizeof(float));
    float *emb = malloc((size_t)dim * sizeof(float));

    printf("generando %d tokens...\n\n%s", ngen, prompt);
    const double tgen = now_s();
    int pos = 0;
    for (int step = 0; step < n + ngen; step++) {
        /* `ids[step]` siempre: para step < n es el prompt, y a partir de ahí lo
         * escribió la iteración anterior. Realimentar `ids[step-1]` —que es lo
         * que hacía antes— repite el último token del prompt y descarrila la
         * generación desde el primer paso. */
        const int tokid = ids[step];

        /* embedding -> hc copias */
        {
            const DsV4W row = dsv4_w_rows(&M.embed, tokid, 1);
            const uint16_t *p = (const uint16_t *)row.w;
            for (int d = 0; d < dim; d++) emb[d] = dsv4_bf16_to_f32(p[d]);
            for (int m = 0; m < hc; m++)
                memcpy(h + (size_t)m * dim, emb, (size_t)dim * sizeof(float));
        }

        for (int L = 0; L < M.n_layers; L++) {
            model_layer_decode(&M, L, &st[L], h, pos, (int32_t)tokid, h2);
            memcpy(h, h2, (size_t)hc * dim * sizeof(float));
        }
        pos++;

        if (step >= n - 1) {
            float y[8192], nz[8192];
            dsv4_hc_head(h, M.hc_head_fn, M.hc_head_scale, M.hc_head_base,
                         hc, dim, M.norm_eps, M.hc_eps, y);
            dsv4_rmsnorm(nz, y, M.final_norm, dim, M.norm_eps);
            for (int d = 0; d < dim; d++) nz[d] = dsv4_to_bf16(nz[d]);
            dsv4_matmul_w(logits, nz, &M.head, 1, 0);
            int best = 0;
            for (int v = 1; v < M.vocab; v++) if (logits[v] > logits[best]) best = v;
            if (step < n + ngen - 1) ids[step + 1] = best;
            char piece[64];
            const int m = tok_decode(&M.tok, &best, 1, piece, sizeof piece - 1);
            piece[m] = 0;
            printf("%s", piece);
        }
    }
    const double dt = now_s() - tgen;
    printf("\n\n%d tokens en %.1f s (%.2f tok/s)\n", n + ngen, dt, (n + ngen) / dt);
    printf("expertos: %llu hits / %llu miss, %.2f GB leidos\n",
           (unsigned long long)M.tier.hits, (unsigned long long)M.tier.miss,
           (double)M.tier.bytes / 1e9);
    return 0;
}
