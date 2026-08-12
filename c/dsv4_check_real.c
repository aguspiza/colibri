/* check_real.c — read weights out of the real 284B checkpoint and apply them.
 *
 * This closes the last stretch of the load path. It opens the REAL checkpoint with
 * colibri's reader (st.h), locates a given expert among the 72,317 tensors spread
 * over 48 shards, reads its MXFP4 bytes as they are, and computes `y = x @ W^T`
 * with `matmul_mxfp4` — the kernel colibri already uses for Kimi K3.
 *
 * The engine does NOT dequantize: it consumes the e2m1 nibbles and the ue8m0
 * scales straight from the file. The reference (ref/make_expert_fixture.py) does
 * dequantize, in Python, and the two have to agree.
 *
 * Build:  make check-real
 * Uso:    ./check_real <dir_del_modelo> [dir_fixtures]
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
/* Enables dsv4_matmul_w's MXFP4 path. `matmul_mxfp4` is `static` in quant.h, so
 * quant.h has to be included in this same translation unit. */
#define DSV4_WITH_MXFP4
#include "quant.h"
#include "dsv4_fp8.h"
#include "dsv4_weight.h"
#include "dsv4_attn.h"
#include "dsv4_decode.h"
#include "dsv4_moe.h"

static double now_s(void) {
#ifdef _WIN32
    return (double)clock() / CLOCKS_PER_SEC;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *model = (argc > 1) ? argv[1]
        : "C:\\Users\\Gus\\ai\\models\\DeepSeek-V4-Flash-0731";
    const char *fixt = (argc > 2) ? argv[2] : "../ref/fixtures";

    printf("checkpoint: %s\n", model);
    double t0 = now_s();
    shards S;
    st_init(&S, model);
    printf("indexado en %.2f s\n\n", now_s() - t0);

    const char *WN = "layers.0.ffn.experts.0.w1.weight";
    const char *SN = "layers.0.ffn.experts.0.w1.scale";
    if (!st_has(&S, WN) || !st_has(&S, SN)) {
        fprintf(stderr, "cannot find the expert in the checkpoint\n");
        return 2;
    }

    st_tensor *tw = st_find(&S, WN);
    st_tensor *ts = st_find(&S, SN);
    const int64_t nw = st_nbytes(&S, WN);
    const int64_t ns = st_nbytes(&S, SN);

    /* Layout: the weight arrives as int8 [O, I/2] (two e2m1 nibbles per byte) and
     * the scale as e8m0 [O, I/32], one per 32 logical values. */
    const int O = 2048, I = 4096;
    printf("experto: %s\n", WN);
    printf("  weights %8lld bytes  (expected %d)\n",
           (long long)nw, O * I / 2);
    printf("  scales  %8lld bytes  (expected %d)\n",
           (long long)ns, O * I / 32);
    if (nw != (int64_t)O * I / 2 || ns != (int64_t)O * I / 32) {
        fprintf(stderr, "the sizes do not match the MXFP4 layout\n");
        return 1;
    }

    /* Two reads, not one: safetensors groups all the `.weight` tensors in one
     * region and all the `.scale` tensors in another, so an expert is 2 contiguous
     * spans. colibri's happy case — a single `pread` per expert — is two here. */
    uint8_t *q4 = malloc((size_t)nw);
    uint8_t *e8 = malloc((size_t)ns);
    if (!q4 || !e8) { fprintf(stderr, "OOM\n"); return 1; }
    t0 = now_s();
    st_read_raw(&S, WN, q4, 0);
    st_read_raw(&S, SN, e8, 0);
    const double t_read = now_s() - t0;
    printf("  read %.2f MB in %.1f ms\n\n",
           (double)(nw + ns) / 1e6, t_read * 1e3);

    /* --- the reference -------------------------------------------------- */
    shards F;
    st_init(&F, fixt);
    if (!st_has(&F, "x") || !st_has(&F, "y")) {
        fprintf(stderr, "fixtures missing; run ref/make_expert_fixture.py\n");
        return 2;
    }
    float *x = malloc((size_t)I * sizeof(float));
    float *ref = malloc((size_t)O * sizeof(float));
    st_read_f32(&F, "x", x, 0);
    st_read_f32(&F, "y", ref, 0);

    /* --- colibri's kernel, no dequantization ---------------------------- */
    float *y = malloc((size_t)O * sizeof(float));
    t0 = now_s();
    matmul_mxfp4(y, x, q4, e8, 1 /*S*/, I, O);
    const double t_mm = now_s() - t0;

    double ss = 0, rr = 0, mx = 0;
    int worst = 0;
    for (int o = 0; o < O; o++) {
        const double d = fabs((double)y[o] - (double)ref[o]);
        if (d > mx) { mx = d; worst = o; }
        ss += d * d;
        rr += (double)ref[o] * ref[o];
    }
    const double rel = sqrt(ss / rr);

    double m = 0, sd = 0;
    for (int o = 0; o < O; o++) m += y[o];
    m /= O;
    for (int o = 0; o < O; o++) sd += (y[o] - m) * (y[o] - m);
    sd = sqrt(sd / O);
    printf("matmul_mxfp4: %.1f ms for [1,%d] x [%d,%d]^T\n", t_mm * 1e3, I, O, I);
    printf("  y   : mean %+.6f  std %.6f\n", m, sd);
#if defined(__AVX512F__) && defined(__AVX512BW__)
    printf("  SIMD path: AVX-512\n");
#elif defined(__AVX2__)
    /* Zen 2 (Ryzen 5700U) has no AVX-512. quant.h pairs each path
     * AVX-512 path with an AVX2 one, so it builds and works the same — slower. */
    printf("  SIMD path: AVX2 (no AVX-512 on this CPU)\n");
#else
    printf("  SIMD path: scalar\n");
#endif

    /* f32 tolerance: the reference accumulates in f64 through numpy, the kernel in
     * f32 with SIMD, so ~1e-7 is expected, not exact zero. */
    const int ok = (rel < 1e-5);
    printf("\n  [%s] matches the reference   rel err %.2e   max abs %.2e",
           ok ? "ok  " : "FALLO", rel, mx);
    if (!ok) printf("  (o=%d: C=%.6f ref=%.6f)", worst, y[worst], ref[worst]);
    printf("\n");

    if (ok) {
        printf("\nThe engine reads MXFP4 experts out of the 284B checkpoint and\n"
               "applies them with colibri kernel, no intermediate conversion.\n");
    }

    /* --- the dense set: FP8-e4m3 with UE8M0 scales ---------------------- */
    printf("\n--- the dense set (the format colibri refuses) ---\n");
    int ok2 = 0;
    {
        const char *DW = "layers.0.attn.wq_a.weight";
        const char *DS = "layers.0.attn.wq_a.scale";
        if (st_has(&S, DW) && st_has(&S, DS) && st_has(&F, "xd")) {
            const int dO = 1024, dI = 4096;
            const int64_t dnw = st_nbytes(&S, DW), dns = st_nbytes(&S, DS);
            printf("%s\n", DW);
            printf("  weights %8lld bytes  (expected %d)\n",
                   (long long)dnw, dO * dI);
            printf("  scales  %8lld bytes  (expected %d = %dx%d blocks)\n",
                   (long long)dns, dsv4_nblk(dO) * dsv4_nblk(dI),
                   dsv4_nblk(dO), dsv4_nblk(dI));

            uint8_t *dq = malloc((size_t)dnw);
            uint8_t *de = malloc((size_t)dns);
            st_read_raw(&S, DW, dq, 0);
            st_read_raw(&S, DS, de, 0);

            float *xd = malloc((size_t)dI * sizeof(float));
            float *rd = malloc((size_t)dO * sizeof(float));
            float *yd = malloc((size_t)dO * sizeof(float));
            st_read_f32(&F, "xd", xd, 0);
            st_read_f32(&F, "yd", rd, 0);

            t0 = now_s();
            dsv4_matmul_fp8_ue8m0(yd, xd, dq, de, 1, dI, dO);
            const double t_d = now_s() - t0;

            double s2 = 0, r2 = 0, m2 = 0;
            int w2 = 0;
            for (int o = 0; o < dO; o++) {
                const double d = fabs((double)yd[o] - (double)rd[o]);
                if (d > m2) { m2 = d; w2 = o; }
                s2 += d * d; r2 += (double)rd[o] * rd[o];
            }
            const double rel2 = sqrt(s2 / r2);
            ok2 = (rel2 < 1e-5);
            printf("  matmul_fp8_ue8m0: %.1f ms\n", t_d * 1e3);
            printf("  [%s] matches the reference   rel err %.2e   max abs %.2e",
                   ok2 ? "ok  " : "FALLO", rel2, m2);
            if (!ok2) printf("  (o=%d: C=%.6f ref=%.6f)", w2, yd[w2], rd[w2]);
            printf("\n");

            free(dq); free(de); free(xd); free(rd); free(yd);
        } else {
            printf("  dense fixtures missing; run make_expert_fixture.py\n");
        }
    }

    /* --- all four paths, through the dispatcher ------------------------- */
    printf("\n--- the weight dispatcher (dsv4_matmul_w) ---\n");
    int ok3 = 1;
    {
        /* MXFP4 and FP8B were validated above against their direct kernels;
         * this checks the dispatcher gives the SAME answer, and adds BF16, which
         * is the fourth path and had not been exercised. */
        struct { const char *tag, *wn, *sn, *xn, *yn; DsV4WKind k; int O, I; } cases[] = {
            { "MXFP4 (expert)", "layers.0.ffn.experts.0.w1.weight",
              "layers.0.ffn.experts.0.w1.scale", "x", "y", DSV4_W_MXFP4, 2048, 4096 },
            { "FP8B  (dense)  ", "layers.0.attn.wq_a.weight",
              "layers.0.attn.wq_a.scale", "xd", "yd", DSV4_W_FP8B, 1024, 4096 },
            { "BF16  (router) ", "layers.0.ffn.gate.weight",
              NULL, "xb", "yb", DSV4_W_BF16, 256, 4096 },
        };
        for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            if (!st_has(&S, cases[c].wn) || !st_has(&F, cases[c].xn)) {
                printf("  [----] %s  no fixture\n", cases[c].tag);
                continue;
            }
            const int cO = cases[c].O, cI = cases[c].I;
            uint8_t *wb = malloc((size_t)st_nbytes(&S, cases[c].wn));
            st_read_raw(&S, cases[c].wn, wb, 0);
            uint8_t *sb = NULL;
            if (cases[c].sn) {
                sb = malloc((size_t)st_nbytes(&S, cases[c].sn));
                st_read_raw(&S, cases[c].sn, sb, 0);
            }
            float *cx = malloc((size_t)cI * sizeof(float));
            float *cr = malloc((size_t)cO * sizeof(float));
            float *cy = malloc((size_t)cO * sizeof(float));
            st_read_f32(&F, cases[c].xn, cx, 0);
            st_read_f32(&F, cases[c].yn, cr, 0);

            DsV4W W;
            switch (cases[c].k) {
            case DSV4_W_MXFP4: W = dsv4_w_mxfp4(wb, sb, cO, cI); break;
            case DSV4_W_FP8B:  W = dsv4_w_fp8b(wb, sb, cO, cI);  break;
            default:           W = dsv4_w_bf16(wb, cO, cI);      break;
            }
            dsv4_matmul_w(cy, cx, &W, 1, 0 /* unrounded, like the reference */);

            double s3 = 0, r3 = 0, m3 = 0;
            for (int o = 0; o < cO; o++) {
                const double d = fabs((double)cy[o] - (double)cr[o]);
                if (d > m3) m3 = d;
                s3 += d * d; r3 += (double)cr[o] * cr[o];
            }
            const double rel3 = sqrt(s3 / r3);
            const int good = (rel3 < 1e-5);
            ok3 &= good;
            printf("  [%s] %s  rel err %.2e   max abs %.2e\n",
                   good ? "ok  " : "FALLO", cases[c].tag, rel3, m3);

            free(wb); free(sb); free(cx); free(cr); free(cy);
        }
    }

    /* --- freqs_cis with YaRN, the config's real parameters -------------- */
    printf("\n--- freqs_cis with YaRN (1M context) ---\n");
    int ok4 = 1;
    {
        const int dim = 64, SEQ = 64, orig = 65536;
        const double factor = 16.0, bf = 32.0, bs = 1.0;
        struct { const char *key; double base; } tabs[] = {
            { "freqs_rope",  10000.0 },   /* the window KV              */
            { "freqs_crope", 160000.0 },  /* the compressed KV (compress_rope_theta) */
        };
        for (size_t t = 0; t < 2; t++) {
            if (!st_has(&F, tabs[t].key)) {
                printf("  [----] %s no fixture\n", tabs[t].key);
                ok4 = 0;
                continue;
            }
            const int64_t n = st_numel(&F, tabs[t].key);
            float *rf = malloc((size_t)n * sizeof(float));
            float *mine = malloc((size_t)n * sizeof(float));
            st_read_f32(&F, tabs[t].key, rf, 0);
            dsv4_precompute_freqs(mine, dim, SEQ, orig, tabs[t].base,
                                  factor, bf, bs);
            double s4 = 0, r4 = 0, m4 = 0;
            for (int64_t i = 0; i < n; i++) {
                const double d = fabs((double)mine[i] - (double)rf[i]);
                if (d > m4) m4 = d;
                s4 += d * d; r4 += (double)rf[i] * rf[i];
            }
            const double rel4 = sqrt(s4 / r4);
            const int good = (rel4 < 1e-6);
            ok4 &= good;
            printf("  [%s] %-12s base %8.0f   rel err %.2e   max abs %.2e\n",
                   good ? "ok  " : "FALLO", tabs[t].key, tabs[t].base, rel4, m4);
            free(rf); free(mine);
        }
    }

    /* --- ONE REAL LAYER: the complete attention block ------------------- */
    printf("\n--- a real layer of the 284B model: attention block ---\n");
    int ok5 = 1;
    {
        int L = (argc > 3) ? atoi(argv[3]) : 0;
        char kx[64], ko[64];
        snprintf(kx, sizeof kx, "lx");
        snprintf(ko, sizeof ko, "lout");

        if (!st_has(&F, kx)) {
            printf("  no layer fixture; run ref/make_layer_fixture.py %d\n", L);
            ok5 = 0;
        } else {
            /* the checkpoint's real dimensions */
            DsV4AttnCfg c;
            memset(&c, 0, sizeof c);
            c.dim = 4096;  c.q_lora = 1024; c.heads = 64;  c.hd = 512;
            c.rd  = 64;    c.groups = 8;    c.o_lora = 1024;
            c.win = 128;   c.eps = 1e-6f;
            c.scale = 1.0f / sqrtf((float)c.hd);
            /* ratio and seqlen come from the fixture: the length depends on the
             * layer type (at ratio 128 it takes 128 tokens to close a
             * compressed block). */
            /* The FIXTURE dictates the layer, not argv: if they disagree the
             * one layer's weights get paired with another's metadata, and the
             * symptom is an opaque segfault far from the cause. Happened twice. */
            float meta[4] = {0, 0, 8, 1};
            if (st_has(&F, "lmeta")) st_read_f32(&F, "lmeta", meta, 0);
            L = (int)meta[0];
            c.ratio = (int)meta[1];
            const int s = (int)meta[2], b = (int)meta[3];
            printf("  layer %d, ratio %d, %d tokens\n", L, c.ratio, s);
            printf("  layer %d, ratio %d, %d tokens, batch %d\n",
                   L, c.ratio, s, b);

            /* Pesos en su formato NATIVO: nada se dequantiza al cargar. */
            DsV4AttnW w;
            memset(&w, 0, sizeof w);
            /* 2 pointers per weight (bytes + scales) x 5 weights = 10. This was
             * dimensionado a 8 y desbordaba: heap corruption al liberar. */
            uint8_t *keep[40];
            int nk = 0;
#define RW(field, tname, OO, II) do { \
        snprintf(nm, sizeof nm, "layers.%d." tname, L); \
        uint8_t *_b = malloc((size_t)st_nbytes(&S, nm)); \
        st_read_raw(&S, nm, _b, 0); keep[nk++] = _b; \
        snprintf(nm, sizeof nm, "layers.%d." tname, L); \
        char _sn[192]; snprintf(_sn, sizeof _sn, "layers.%d.%.*s.scale", L, \
                 (int)(strlen(tname) - 7), tname); \
        uint8_t *_s = malloc((size_t)st_nbytes(&S, _sn)); \
        st_read_raw(&S, _sn, _s, 0); keep[nk++] = _s; \
        w.field = dsv4_w_fp8b(_b, _s, (OO), (II)); } while (0)
            char nm[256];
            RW(wq_a, "attn.wq_a.weight", c.q_lora, c.dim);
            RW(wq_b, "attn.wq_b.weight", c.heads * c.hd, c.q_lora);
            RW(wkv,  "attn.wkv.weight",  c.hd, c.dim);
            RW(wo_a, "attn.wo_a.weight", c.groups * c.o_lora,
                                         c.heads * c.hd / c.groups);
            RW(wo_b, "attn.wo_b.weight", c.dim, c.groups * c.o_lora);
#undef RW
            /* Vectors: BF16 in the checkpoint, converted to f32 at load. */
            float *qn = malloc((size_t)c.q_lora * sizeof(float));
            float *kn = malloc((size_t)c.hd * sizeof(float));
            float *sk = malloc((size_t)c.heads * sizeof(float));
            snprintf(nm, sizeof nm, "layers.%d.attn.q_norm.weight", L);
            { uint8_t *r = malloc((size_t)st_nbytes(&S, nm)); st_read_raw(&S, nm, r, 0);
              DsV4W v = dsv4_w_bf16(r, 1, c.q_lora); dsv4_vec_f32(qn, &v, c.q_lora);
              free(r); }
            snprintf(nm, sizeof nm, "layers.%d.attn.kv_norm.weight", L);
            { uint8_t *r = malloc((size_t)st_nbytes(&S, nm)); st_read_raw(&S, nm, r, 0);
              DsV4W v = dsv4_w_bf16(r, 1, c.hd); dsv4_vec_f32(kn, &v, c.hd);
              free(r); }
            snprintf(nm, sizeof nm, "layers.%d.attn.attn_sink", L);
            st_read_f32(&S, nm, sk, 0);
            w.q_norm = qn; w.kv_norm = kn; w.attn_sink = sk;

            /* --- Compressor and Indexer weights, if the layer compresses --- */
            float *cape = NULL, *cnorm = NULL, *iape = NULL, *inorm = NULL;
#define RWC(field, tname, OO, II) do { \
        snprintf(nm, sizeof nm, "layers.%d." tname ".weight", L); \
        uint8_t *_b = malloc((size_t)st_nbytes(&S, nm)); \
        st_read_raw(&S, nm, _b, 0); keep[nk++] = _b; \
        snprintf(nm, sizeof nm, "layers.%d." tname ".scale", L); \
        uint8_t *_s = NULL; \
        if (st_has(&S, nm)) { _s = malloc((size_t)st_nbytes(&S, nm)); \
                              st_read_raw(&S, nm, _s, 0); keep[nk++] = _s; } \
        w.field = _s ? dsv4_w_fp8b(_b, _s, (OO), (II)) \
                     : dsv4_w_bf16(_b, (OO), (II)); } while (0)
#define RVEC(dst, tname, N) do { \
        snprintf(nm, sizeof nm, "layers.%d." tname, L); \
        uint8_t *_r = malloc((size_t)st_nbytes(&S, nm)); \
        st_read_raw(&S, nm, _r, 0); \
        (dst) = malloc((size_t)(N) * sizeof(float)); \
        DsV4W _v = dsv4_w_bf16(_r, 1, (N)); dsv4_vec_f32((dst), &_v, (N)); \
        free(_r); } while (0)
            if (c.ratio) {
                const int coff = (c.ratio == 4) ? 2 : 1;
                RWC(c_wkv,   "attn.compressor.wkv",   coff * c.hd, c.dim);
                RWC(c_wgate, "attn.compressor.wgate", coff * c.hd, c.dim);
                RVEC(cnorm,  "attn.compressor.norm.weight", c.hd);
                /* `ape` is F32 in the checkpoint, not BF16 */
                snprintf(nm, sizeof nm, "layers.%d.attn.compressor.ape", L);
                cape = malloc((size_t)st_numel(&S, nm) * sizeof(float));
                st_read_f32(&S, nm, cape, 0);
                w.c_norm = cnorm; w.c_ape = cape;

                if (c.ratio == 4) {
                    c.i_hd = 128; c.i_heads = 64; c.i_topk = 512;
                    c.i_scale = 1.0f / sqrtf((float)c.i_hd);
                    RWC(i_wkv,   "attn.indexer.compressor.wkv",   coff * c.i_hd, c.dim);
                    RWC(i_wgate, "attn.indexer.compressor.wgate", coff * c.i_hd, c.dim);
                    RWC(i_wq_b,  "attn.indexer.wq_b",  c.i_heads * c.i_hd, c.q_lora);
                    RWC(i_wproj, "attn.indexer.weights_proj", c.i_heads, c.dim);
                    RVEC(inorm,  "attn.indexer.compressor.norm.weight", c.i_hd);
                    snprintf(nm, sizeof nm,
                             "layers.%d.attn.indexer.compressor.ape", L);
                    iape = malloc((size_t)st_numel(&S, nm) * sizeof(float));
                    st_read_f32(&S, nm, iape, 0);
                    w.i_norm = inorm; w.i_ape = iape;
                }
            }
#undef RWC
#undef RVEC

            float *freqs = malloc((size_t)st_numel(&F, "lfreqs") * sizeof(float));
            st_read_f32(&F, "lfreqs", freqs, 0);
            w.freqs = freqs;

            int64_t n_lx = 0, n_lo = 0;
            float *lx = malloc((size_t)st_numel(&F, kx) * sizeof(float));
            float *lo = malloc((size_t)st_numel(&F, ko) * sizeof(float));
            st_read_f32(&F, kx, lx, 0); n_lx = st_numel(&F, kx);
            st_read_f32(&F, ko, lo, 0); n_lo = st_numel(&F, ko);

            int *tk = malloc((size_t)b * s * c.win * sizeof(int));
            const int ntopk = dsv4_window_topk_prefill(tk, b, s, c.win);

            float *got = malloc((size_t)n_lo * sizeof(float));
            t0 = now_s();
            dsv4_attention_prefill(&c, &w, lx, tk, b, s, ntopk, got);
            const double t_l = now_s() - t0;

            double s5 = 0, r5 = 0, m5 = 0;
            for (int64_t i = 0; i < n_lo; i++) {
                const double d = fabs((double)got[i] - (double)lo[i]);
                if (d > m5) m5 = d;
                s5 += d * d; r5 += (double)lo[i] * lo[i];
            }
            const double rel5 = sqrt(s5 / r5);
            ok5 = (rel5 < 1e-2);
            printf("  layer %d, %d tokens, dim %d, %d heads x %d\n",
                   L, s, c.dim, c.heads, c.hd);
            printf("  forward in %.0f ms\n", t_l * 1e3);
            printf("  [%s] matches model.py   rel err %.2e   max abs %.2e\n",
                   ok5 ? "ok  " : "FALLO", rel5, m5);

            /* --- PREFILL vs SEQUENTIAL DECODE, same tokens -------------- */
            /* The two paths have each been checked against model.py, but never
             * against EACH OTHER, and never past 8 tokens. That gap matters now:
             * batched prefill is only useful if it can replace N decode steps,
             * which requires it to agree with them on the output AND to leave the
             * same state behind.
             *
             * This is the output half, and it is testable today. Sequential decode
             * over the same `s` tokens must reproduce what prefill produced -- not
             * bit for bit (the accumulation order differs) but inside the bf16
             * band the rest of the port lives in.
             *
             * The state half needs prefill to populate DsV4AttnState, which it
             * does not yet. The final decode state is dumped here so the batched
             * implementation has something to be compared against. */
            {
                DsV4AttnState sq;
                dsv4_state_init_full(&sq, b, c.hd, c.win, 4096, c.ratio, c.i_hd);
                float *dseq = malloc((size_t)b * c.dim * sizeof(float));
                float *acc  = malloc((size_t)n_lo * sizeof(float));
                const int64_t per = n_lo / s;      /* outputs per position */

                for (int t = 0; t < s; t++) {
                    /* position t of the same input, one token at a time */
                    dsv4_attention_decode(&c, &w, &sq, lx + (size_t)t * c.dim,
                                          t, dseq);
                    memcpy(acc + (size_t)t * per, dseq,
                           (size_t)per * sizeof(float));
                }

                double sd = 0, rd_ = 0, md = 0;
                for (int64_t i = 0; i < n_lo; i++) {
                    const double d = fabs((double)acc[i] - (double)got[i]);
                    if (d > md) md = d;
                    sd += d * d; rd_ += (double)got[i] * (double)got[i];
                }
                const double reld = sqrt(sd / rd_);
                /* Looser than the model.py comparison on purpose: this measures
                 * two DIFFERENT accumulation orders against each other, so it
                 * carries both paths' rounding, not one path's against a
                 * reference. */
                printf("  [%s] prefill == %d sequential decodes   rel err %.2e   max abs %.2e\n",
                       (reld < 2e-2) ? "ok  " : "FAIL ", s, reld, md);

                /* Dump the state the sequential path ended with, so a batched
                 * prefill can be diffed against it rather than eyeballed. */
                const char *dp = getenv("DSV4_DUMP_STATE");
                if (dp) {
                    FILE *df = fopen(dp, "wb");
                    if (df) {
                        const int hdr[6] = { b, c.hd, c.win, c.ratio, c.i_hd, sq.ncomp };
                        fwrite(hdr, sizeof hdr, 1, df);
                        fwrite(sq.kv, sizeof(float),
                               (size_t)b * (sq.win + sq.ncomp) * c.hd, df);
                        if (sq.kv_state) {
                            const size_t nst = (size_t)b * sq.coff * c.ratio
                                             * sq.coff * c.hd;
                            fwrite(sq.kv_state,    sizeof(float), nst, df);
                            fwrite(sq.score_state, sizeof(float), nst, df);
                        }
                        if (sq.i_kv) {
                            const size_t nim = (size_t)b * sq.coff * c.ratio
                                             * sq.coff * c.i_hd;
                            fwrite(sq.i_kv, sizeof(float),
                                   (size_t)b * sq.ncomp * c.i_hd, df);
                            fwrite(sq.i_kv_state,    sizeof(float), nim, df);
                            fwrite(sq.i_score_state, sizeof(float), nim, df);
                        }
                        fclose(df);
                        printf("         state dumped to %s\n", dp);
                    }
                }
                dsv4_state_free(&sq);
                free(dseq); free(acc);
            }

            /* --- the same comparison, at lengths that cross the window --
             *
             * The fixture is 8 tokens, which never reaches the 128-token window
             * boundary, never wraps the KV ring and -- on a ratio-128 layer --
             * never closes a compressor block. Those are exactly the paths a long
             * prompt walks, and a shared-RoPE-table bug hid there for hours
             * because nothing exercised them.
             *
             * No reference output is needed to compare prefill against sequential
             * decode: they must agree with each other. So the input can be
             * synthetic and the length free, which makes the interesting lengths
             * cheap to reach (one layer, not 43). */
            {
                const int lens[] = { 32, 127, 128, 129, 200, 300 };
                float *xs = malloc((size_t)300 * c.dim * sizeof(float));
                /* Deterministic pseudo-random input in the range activations
                 * actually take, so a failure is reproducible. */
                uint32_t rs = 0x12345678u;
                for (int64_t i = 0; i < (int64_t)300 * c.dim; i++) {
                    rs = rs * 1664525u + 1013904223u;
                    xs[i] = dsv4_to_bf16(((float)(rs >> 8) / 8388608.0f - 1.0f) * 0.5f);
                }
                for (unsigned li = 0; li < sizeof lens / sizeof *lens; li++) {
                    const int sl = lens[li];
                    int *tks = malloc((size_t)sl * c.win * sizeof(int));
                    const int nt = dsv4_window_topk_prefill(tks, 1, sl, c.win);
                    float *pre = malloc((size_t)sl * c.dim * sizeof(float));
                    dsv4_attention_prefill(&c, &w, xs, tks, 1, sl, nt, pre);

                    DsV4AttnState q2;
                    dsv4_state_init_full(&q2, 1, c.hd, c.win, 4096, c.ratio, c.i_hd);
                    float *one = malloc((size_t)c.dim * sizeof(float));
                    double sd = 0, rr = 0, md = 0;
                    for (int t = 0; t < sl; t++) {
                        dsv4_attention_decode(&c, &w, &q2, xs + (size_t)t * c.dim,
                                              t, one);
                        for (int d = 0; d < c.dim; d++) {
                            const double a = one[d];
                            const double bq = pre[(size_t)t * c.dim + d];
                            const double df = fabs(a - bq);
                            if (df > md) md = df;
                            sd += df * df; rr += bq * bq;
                        }
                    }
                    const double rel = (rr > 0) ? sqrt(sd / rr) : 0.0;
                    printf("  [%s] len %3d  prefill == decode   rel err %.2e   max abs %.2e\n",
                           (rel < 2e-2) ? "ok  " : "FAIL ", sl, rel, md);
                    dsv4_state_free(&q2);
                    free(tks); free(pre); free(one);
                }
                free(xs);
            }

            /* ---------------------------------------------------------------
             * PREFILL FROM A NON-ZERO POSITION: the target, and it fails today.
             *
             * This is the one limitation that forces every prompt following a
             * reused prefix -- and every chunk after the first -- onto the
             * token-at-a-time path. Measured on the real build, that path spends
             * 67 % of the wall clock in a single thread, so this is the ceiling
             * once the per-token loops are parallel.
             *
             * dsv4_attention_prefill_cap assumes the batch starts at position 0 in
             * three places (`r % s` for the query and kv RoPE, and `(r % s + 1) /
             * ratio` for the compressed top-k) and, more fundamentally, it receives
             * no prior state: the window cannot reach tokens before pos0 and the
             * already-compressed blocks are invisible to it.
             *
             * So the test is written against the behaviour we want -- prefill the
             * tail of a sequence and match what sequential decode produces for the
             * same tokens -- and reports the gap rather than failing the build. The
             * number is the thing to drive to zero; it is printed as `todo` so that
             * a real regression elsewhere is still visible.
             * ------------------------------------------------------------- */
            {
                /* The freqs table comes from the fixture, so the positions it covers
                 * are fixed. Reading past it produced ~1e20 garbage in BOTH the
                 * reference and the result, which made rel err come out as 0.00e+00
                 * next to a max abs of 5.9e20 -- a green verdict on nonsense. The
                 * bound has to come from the table, not from a number I picked. */
                const int fmax = (int)(st_numel(&F, "lfreqs") / (int64_t)(c.rd / 2 * 2));
                /* pos0 must be a multiple of ratio so the compressor has no partial
                 * block at the boundary; 128 satisfies both 4 and 128. */
                int pre = 128, n = 64;
                if (pre + n > fmax) n = (fmax > pre + 8) ? fmax - pre - 1 : 0;
                if (pre < 1 || n < 1) {
                    printf("  [skip] prefill from a non-zero position: the fixture's "
                           "freqs table only covers %d positions\n", fmax);
                } else {
                const int tot = pre + n;
                float *xs = malloc((size_t)tot * c.dim * sizeof(float));
                uint32_t rs = 0x9e3779b9u;
                for (int64_t i = 0; i < (int64_t)tot * c.dim; i++) {
                    rs = rs * 1664525u + 1013904223u;
                    xs[i] = dsv4_to_bf16(((float)(rs >> 8) / 8388608.0f - 1.0f) * 0.5f);
                }

                /* Reference: decode everything one token at a time, keep the tail. */
                float *ref = malloc((size_t)n * c.dim * sizeof(float));
                {
                    DsV4AttnState st;
                    dsv4_state_init_full(&st, 1, c.hd, c.win, 4096, c.ratio, c.i_hd);
                    float *one = malloc((size_t)c.dim * sizeof(float));
                    for (int t = 0; t < tot; t++) {
                        dsv4_attention_decode(&c, &w, &st, xs + (size_t)t * c.dim, t, one);
                        if (t >= pre)
                            memcpy(ref + (size_t)(t - pre) * c.dim, one,
                                   (size_t)c.dim * sizeof(float));
                    }
                    free(one);
                    dsv4_state_free(&st);
                }

                /* Under test: the tail as one batch. Today this is told nothing about
                 * the first `pre` tokens, so it answers as if the sequence began at
                 * the tail -- which is exactly the gap being measured. */
                DsV4AttnState pv;
                dsv4_state_init_full(&pv, 1, c.hd, c.win, 4096, c.ratio, c.i_hd);
                {   float *one = malloc((size_t)c.dim * sizeof(float));
                    for (int t = 0; t < pre; t++)
                        dsv4_attention_decode(&c, &w, &pv, xs + (size_t)t * c.dim, t, one);
                    free(one); }

                const int can = dsv4_prefill_can_resume(&c);
                const int prewin = can ? ((c.win - 1 < pre) ? c.win - 1 : pre) : 0;
                int *tks = malloc((size_t)n * c.win * sizeof(int));
                const int nt = dsv4_window_topk_prefill_at(tks, 1, n, c.win,
                                                           can ? pre : 0, prewin);
                float *got = malloc((size_t)n * c.dim * sizeof(float));
                dsv4_attention_prefill_cap(&c, &w, xs + (size_t)pre * c.dim, tks, 1, n,
                                           nt, can ? pre : 0,
                                           can ? pv.kv : NULL, pv.win, pv.ncomp,
                                           got, NULL);

                double sd = 0, rr = 0, md = 0;
                for (int64_t i = 0; i < (int64_t)n * c.dim; i++) {
                    const double d = fabs((double)got[i] - (double)ref[i]);
                    if (d > md) md = d;
                    sd += d * d; rr += (double)ref[i] * (double)ref[i];
                }
                const double rel = (rr > 0) ? sqrt(sd / rr) : 0.0;
                printf("  [%s] prefill from pos %d (%d tokens) vs decode   "
                       "rel err %.2e   max abs %.2e\n",
                       (rel < 2e-2) ? "ok  " : "todo", pre, n, rel, md);
                dsv4_state_free(&pv);
                if (rel >= 2e-2)
                    printf("         ^ expected until prefill_cap takes pos0 and the "
                           "prior state; this is the number to drive to zero\n");
                free(xs); free(ref); free(tks); free(got);
                }
            }

            /* ---------------------------------------------------------------
             * BENCH. Seconds instead of the ten-minute engine runs that every
             * measurement in this session cost, and on the two things that decide
             * the prefill: batched attention against the same tokens decoded one at
             * a time, and the per-token glue that used to run single-threaded.
             *
             * The engine measured 67 % of the wall clock outside both CPU (29 %) and
             * I/O wait (4 %), which is this serial work. A bench that isolates it is
             * what makes the next change measurable before it is deployed.
             *
             * Set DSV4_BENCH=1; skipped by default so `make check-real` stays a
             * correctness run.
             * ------------------------------------------------------------- */
            if (getenv("DSV4_BENCH")) {
                const int lens[] = { 256, 1024 };
                printf("\n--- bench (one layer, %d heads x %d) ---\n", c.heads, c.hd);
                for (unsigned bi = 0; bi < sizeof lens / sizeof *lens; bi++) {
                    const int nb = lens[bi];
                    float *xb = malloc((size_t)nb * c.dim * sizeof(float));
                    uint32_t rs = 0xdeadbeefu;
                    for (int64_t i = 0; i < (int64_t)nb * c.dim; i++) {
                        rs = rs * 1664525u + 1013904223u;
                        xb[i] = dsv4_to_bf16(((float)(rs >> 8) / 8388608.0f - 1.0f) * 0.5f);
                    }
                    float *ob = malloc((size_t)nb * c.dim * sizeof(float));
                    int *tkb = malloc((size_t)nb * c.win * sizeof(int));
                    const int ntb = dsv4_window_topk_prefill(tkb, 1, nb, c.win);

                    double t = now_s();
                    dsv4_attention_prefill(&c, &w, xb, tkb, 1, nb, ntb, ob);
                    const double t_pre = now_s() - t;

                    DsV4AttnState st;
                    dsv4_state_init_full(&st, 1, c.hd, c.win, 4096, c.ratio, c.i_hd);
                    float *one = malloc((size_t)c.dim * sizeof(float));
                    t = now_s();
                    for (int k = 0; k < nb; k++)
                        dsv4_attention_decode(&c, &w, &st, xb + (size_t)k * c.dim, k, one);
                    const double t_dec = now_s() - t;
                    dsv4_state_free(&st);

                    printf("  n=%4d  batched attn %7.1f ms (%.3f ms/tok)  |  "
                           "%d decodes %7.1f ms (%.3f ms/tok)  = %.2fx  |  "
                           "hc_pre %6.1f ms\n",
                           nb, t_pre * 1e3, t_pre * 1e3 / nb, nb,
                           t_dec * 1e3, t_dec * 1e3 / nb,
                           t_pre > 0 ? t_dec / t_pre : 0.0);
                    free(xb); free(ob); free(tkb); free(one);
                }
                printf("  (the last column is what batched attention would buy if it\n"
                       "   accepted a non-zero start position)\n");

                /* THE OPTIMAL CASE: everything already in physical RAM.
                 *
                 * Every other number in this project mixes compute with getting the
                 * bytes there. This one does not -- the expert read at the top of main
                 * is still resident -- so it is the pure MXFP4 ceiling at the shape the
                 * MoE uses. Whatever the engine achieves below this is serialization or
                 * I/O, and the gap IS the size of the defect. Without this line there
                 * is no way to tell "as fast as the machine allows" from "a third of
                 * it", which is how a morning went into optimizing a 4 % I/O wait. */
                printf("\n--- optimal case: expert resident in RAM, pure compute ---\n");
                {
                    const int rows[] = { 1, 256, 1024 };
                    for (unsigned ri = 0; ri < sizeof rows / sizeof *rows; ri++) {
                        const int R = rows[ri];
                        float *xin = malloc((size_t)R * I * sizeof(float));
                        float *yo  = malloc((size_t)R * O * sizeof(float));
                        for (int64_t i = 0; i < (int64_t)R * I; i++)
                            xin[i] = 0.01f * (float)((i % 97) - 48);
                        matmul_mxfp4(yo, xin, q4, e8, R, I, O);   /* fault it in */
                        const double t = now_s();
                        matmul_mxfp4(yo, xin, q4, e8, R, I, O);
                        const double dt = now_s() - t;
                        printf("  rows %4d  %8.1f ms  %6.1f GFLOP/s  %6.3f ms/row\n",
                               R, dt * 1e3,
                               2.0 * (double)R * I * O / dt / 1e9, dt * 1e3 / R);
                        free(xin); free(yo);
                    }
                    printf("  (3 matrices per expert, 6 experts per token: the model's\n"
                           "   compute floor per token follows from the best row)\n");
                }
            }

            /* --- SEEDED STATE: does prefill leave what decode would? ----
             *
             * The output halves already match bit for bit. This is the other
             * half, and the one that decides whether chunked prefill is usable:
             * after prefilling n tokens, the NEXT token must decode to the same
             * thing it would have if those n tokens had gone through decode one
             * at a time.
             *
             * Comparing the raw state arrays would be weaker than it looks -- a
             * wrong slot in a compressor's in-progress block can sit unread until
             * the block closes, so a byte-diff can pass while the next few tokens
             * are wrong. Decoding several tokens past the seam and comparing those
             * outputs exercises the state instead of inspecting it. */
            {
                const int lens[] = { 32, 127, 128, 129, 200 };
                const int follow = 8;           /* tokens decoded past the seam */
                const int NMAX = 300;
                float *xs = malloc((size_t)NMAX * c.dim * sizeof(float));
                uint32_t rs = 0xC0FFEEu;
                for (int64_t i = 0; i < (int64_t)NMAX * c.dim; i++) {
                    rs = rs * 1664525u + 1013904223u;
                    xs[i] = dsv4_to_bf16(((float)(rs >> 8) / 8388608.0f - 1.0f) * 0.5f);
                }
                const int coff = (c.ratio == 4) ? 2 : 1;

                for (unsigned li = 0; li < sizeof lens / sizeof *lens; li++) {
                    const int n = lens[li];

                    /* (a) reference: n decode steps, then `follow` more */
                    DsV4AttnState ref;
                    dsv4_state_init_full(&ref, 1, c.hd, c.win, 4096, c.ratio, c.i_hd);
                    float *tmp1 = malloc((size_t)c.dim * sizeof(float));
                    for (int t = 0; t < n; t++)
                        dsv4_attention_decode(&c, &w, &ref, xs + (size_t)t * c.dim, t, tmp1);
                    float *refo = malloc((size_t)follow * c.dim * sizeof(float));
                    for (int t = 0; t < follow; t++)
                        dsv4_attention_decode(&c, &w, &ref, xs + (size_t)(n + t) * c.dim,
                                              n + t, refo + (size_t)t * c.dim);

                    /* (b) prefill n tokens with capture, seed, then `follow` */
                    DsV4Capture cap; memset(&cap, 0, sizeof cap);
                    cap.kv = malloc((size_t)n * c.hd * sizeof(float));
                    if (c.ratio) {
                        cap.ckv = malloc((size_t)n * coff * c.hd * sizeof(float));
                        cap.csc = malloc((size_t)n * coff * c.hd * sizeof(float));
                        cap.kv_comp = malloc((size_t)(n / c.ratio + 1) * c.hd * sizeof(float));
                        if (c.ratio == 4 && c.i_hd) {
                            cap.ikv = malloc((size_t)n * coff * c.i_hd * sizeof(float));
                            cap.isc = malloc((size_t)n * coff * c.i_hd * sizeof(float));
                            cap.i_kv_comp = malloc((size_t)(n / c.ratio + 1) * c.i_hd * sizeof(float));
                        }
                    }
                    int *tks = malloc((size_t)n * c.win * sizeof(int));
                    const int nt = dsv4_window_topk_prefill(tks, 1, n, c.win);
                    float *pre = malloc((size_t)n * c.dim * sizeof(float));
                    dsv4_attention_prefill_cap(&c, &w, xs, tks, 1, n, nt, 0, NULL, 0, 0, pre, &cap);

                    DsV4AttnState sd;
                    dsv4_state_init_full(&sd, 1, c.hd, c.win, 4096, c.ratio, c.i_hd);
                    dsv4_state_seed_from_prefill(&sd, &c, &w, &cap, n);

                    float *seedo = malloc((size_t)follow * c.dim * sizeof(float));
                    for (int t = 0; t < follow; t++)
                        dsv4_attention_decode(&c, &w, &sd, xs + (size_t)(n + t) * c.dim,
                                              n + t, seedo + (size_t)t * c.dim);

                    /* per-token error, so a seam failure is visible as such */
                    double worst = 0; int worst_t = -1;
                    for (int t = 0; t < follow; t++) {
                        double sd2 = 0, rr = 0;
                        for (int d = 0; d < c.dim; d++) {
                            const double a = seedo[(size_t)t * c.dim + d];
                            const double bq = refo[(size_t)t * c.dim + d];
                            sd2 += (a - bq) * (a - bq); rr += bq * bq;
                        }
                        const double rel = (rr > 0) ? sqrt(sd2 / rr) : 0.0;
                        if (rel > worst) { worst = rel; worst_t = t; }
                    }
                    printf("  [%s] seed n=%3d + %d decodes   worst rel err %.2e (token +%d)\n",
                           (worst < 2e-2) ? "ok  " : "FAIL ", n, follow, worst, worst_t);

                    dsv4_state_free(&ref); dsv4_state_free(&sd);
                    free(tmp1); free(refo); free(seedo); free(pre); free(tks);
                    free(cap.kv); free(cap.ckv); free(cap.csc); free(cap.kv_comp);
                    free(cap.ikv); free(cap.isc); free(cap.i_kv_comp);
                }
                free(xs);
            }

            /* --- one DECODE step, seeded from the prefill state --------- */
            if (st_has(&F, "dx")) {
                DsV4AttnState state;
                dsv4_state_init_full(&state, b, c.hd, c.win, 256, c.ratio, c.i_hd);

                /* the kv_cache the reference's prefill left behind */
                const int64_t n_dkv = st_numel(&F, "dkv");
                float *dkv = malloc((size_t)n_dkv * sizeof(float));
                st_read_f32(&F, "dkv", dkv, 0);
                const int64_t cache_rows = n_dkv / c.hd;
                const int64_t have = (cache_rows < state.win + state.ncomp)
                                   ? cache_rows : state.win + state.ncomp;
                memcpy(state.kv, dkv, (size_t)have * c.hd * sizeof(float));

                /* Prefill leaves MORE state behind than attention's kv_cache: the
                 * Indexer's compressed KV and the in-progress block of both
                 * Compressors. With overlap that block holds the previous group so
                 * it can be mixed with the next one, so without seeding it the
                 * first block of decode comes out wrong (measured: 2.4e-1). */
#define SEED(key, dst, n) do { \
        if (st_has(&F, key) && (dst)) { \
            const int64_t _n = st_numel(&F, key); \
            const int64_t _c = (_n < (n)) ? _n : (n); \
            float *_t = malloc((size_t)_n * sizeof(float)); \
            st_read_f32(&F, key, _t, 0); \
            memcpy((dst), _t, (size_t)_c * sizeof(float)); \
            free(_t); } } while (0)
                const int coff = state.coff;
                SEED("dcks", state.kv_state,
                     (int64_t)b * coff * c.ratio * coff * c.hd);
                SEED("dcss", state.score_state,
                     (int64_t)b * coff * c.ratio * coff * c.hd);
                SEED("dikv", state.i_kv, (int64_t)b * state.ncomp * c.i_hd);
                SEED("diks", state.i_kv_state,
                     (int64_t)b * coff * c.ratio * coff * c.i_hd);
                SEED("diss", state.i_score_state,
                     (int64_t)b * coff * c.ratio * coff * c.i_hd);
#undef SEED

                float *dx = malloc((size_t)c.dim * sizeof(float));
                float *dref = malloc((size_t)c.dim * sizeof(float));
                float *dgot = malloc((size_t)c.dim * sizeof(float));
                st_read_f32(&F, "dx", dx, 0);
                st_read_f32(&F, "dout", dref, 0);

                t0 = now_s();
                dsv4_attention_decode(&c, &w, &state, dx, s, dgot);
                const double t_d2 = now_s() - t0;

                double s6 = 0, r6 = 0, m6 = 0;
                for (int i = 0; i < c.dim; i++) {
                    const double d = fabs((double)dgot[i] - (double)dref[i]);
                    if (d > m6) m6 = d;
                    s6 += d * d; r6 += (double)dref[i] * dref[i];
                }
                const double rel6 = sqrt(s6 / r6);
                const int okd = (rel6 < 1e-2);
                ok5 &= okd;
                printf("  decode at start_pos=%d: %.0f ms\n", s, t_d2 * 1e3);
                printf("  [%s] matches model.py   rel err %.2e   max abs %.2e\n",
                       okd ? "ok  " : "FALLO", rel6, m6);

                free(dkv); free(dx); free(dref); free(dgot);
                dsv4_state_free(&state);
            }

            for (int i = 0; i < nk; i++) free(keep[i]);
            free(qn); free(kn); free(sk); free(freqs);
            free(cape); free(cnorm); free(iape); free(inorm);
            free(lx); free(lo); free(tk); free(got);
            (void)n_lx;
        }
    }

    /* --- A REAL MoE BLOCK: 256 MXFP4 experts ---------------------------- */
    printf("\n--- MoE block of a real layer ---\n");
    int ok6 = 1;
    if (!st_has(&F, "mx")) {
        printf("  no fixture; run ref/make_moe_fixture.py <layer>\n");
    } else {
        char nm[256];
        float mm[4];
        st_read_f32(&F, "mmeta", mm, 0);
        const int L = (int)mm[0], s = (int)mm[1], b = (int)mm[2];
        const int hashed = (int)mm[3];

        DsV4MoeCfg mc;
        memset(&mc, 0, sizeof mc);
        mc.dim = 4096; mc.n_experts = 256; mc.topk = 6; mc.inter = 2048;
        mc.route_scale = 1.5f; mc.swiglu_limit = 10.0f; mc.hash = hashed;

        DsV4MoeW mw;
        memset(&mw, 0, sizeof mw);
        printf("  layer %d, %s routing, %d MXFP4 experts\n",
               L, hashed ? "hash" : "score-based", mc.n_experts);

        /* router: BF16, with the bias in F32 */
        snprintf(nm, sizeof nm, "layers.%d.ffn.gate.weight", L);
        uint8_t *gwb = malloc((size_t)st_nbytes(&S, nm));
        st_read_raw(&S, nm, gwb, 0);
        mw.gate_w = dsv4_w_bf16(gwb, mc.n_experts, mc.dim);

        float *gb = NULL;
        int32_t *t2e = NULL;
        if (hashed) {
            /* `tid2eid` ships as I64 even though model.py declares it int32:
             * it has to be narrowed at load time. */
            snprintf(nm, sizeof nm, "layers.%d.ffn.gate.tid2eid", L);
            const int64_t nb64 = st_nbytes(&S, nm);
            int64_t *raw = malloc((size_t)nb64);
            st_read_raw(&S, nm, raw, 0);
            const int64_t n64 = nb64 / 8;
            t2e = malloc((size_t)n64 * sizeof(int32_t));
            for (int64_t i = 0; i < n64; i++) t2e[i] = (int32_t)raw[i];
            free(raw);
            mw.tid2eid = t2e;
            printf("  tid2eid: %lld I64 entries -> int32\n", (long long)n64);
        } else {
            snprintf(nm, sizeof nm, "layers.%d.ffn.gate.bias", L);
            gb = malloc((size_t)mc.n_experts * sizeof(float));
            st_read_f32(&S, nm, gb, 0);
            mw.gate_bias = gb;
        }

        /* all 256 experts, in MXFP4, not dequantized */
        DsV4W *E1 = malloc(mc.n_experts * sizeof(DsV4W));
        DsV4W *E2 = malloc(mc.n_experts * sizeof(DsV4W));
        DsV4W *E3 = malloc(mc.n_experts * sizeof(DsV4W));
        uint8_t **eb = malloc((size_t)mc.n_experts * 6 * sizeof(uint8_t *));
        int nb = 0;
        t0 = now_s();
        for (int e = 0; e < mc.n_experts; e++) {
            struct { const char *m; DsV4W *dst; int O, I; } mats[3] = {
                { "w1", &E1[e], mc.inter, mc.dim },
                { "w2", &E2[e], mc.dim,   mc.inter },
                { "w3", &E3[e], mc.inter, mc.dim },
            };
            for (int k = 0; k < 3; k++) {
                snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.%s.weight",
                         L, e, mats[k].m);
                uint8_t *wb = malloc((size_t)st_nbytes(&S, nm));
                st_read_raw(&S, nm, wb, 0); eb[nb++] = wb;
                snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.%s.scale",
                         L, e, mats[k].m);
                uint8_t *sb = malloc((size_t)st_nbytes(&S, nm));
                st_read_raw(&S, nm, sb, 0); eb[nb++] = sb;
                *mats[k].dst = dsv4_w_mxfp4(wb, sb, mats[k].O, mats[k].I);
            }
        }
        const double t_load = now_s() - t0;
        printf("  loaded in %.1f s (%.2f GB)\n", t_load,
               (double)mc.n_experts * 3 * (12582912 / 3 + 262144) / 1e9);
        mw.e_w1 = E1; mw.e_w2 = E2; mw.e_w3 = E3;

        /* shared expert: FP8 with block scales, not MXFP4 */
        uint8_t *shb[6];
        int nsh = 0;
        struct { const char *m; DsV4W *dst; int O, I; } shm[3] = {
            { "w1", &mw.s_w1, mc.inter, mc.dim },
            { "w2", &mw.s_w2, mc.dim,   mc.inter },
            { "w3", &mw.s_w3, mc.inter, mc.dim },
        };
        for (int k = 0; k < 3; k++) {
            snprintf(nm, sizeof nm, "layers.%d.ffn.shared_experts.%s.weight",
                     L, shm[k].m);
            uint8_t *wb = malloc((size_t)st_nbytes(&S, nm));
            st_read_raw(&S, nm, wb, 0); shb[nsh++] = wb;
            snprintf(nm, sizeof nm, "layers.%d.ffn.shared_experts.%s.scale",
                     L, shm[k].m);
            uint8_t *sb = malloc((size_t)st_nbytes(&S, nm));
            st_read_raw(&S, nm, sb, 0); shb[nsh++] = sb;
            *shm[k].dst = dsv4_w_fp8b(wb, sb, shm[k].O, shm[k].I);
        }

        const int64_t rows = (int64_t)b * s;
        float *mx = malloc((size_t)rows * mc.dim * sizeof(float));
        float *mref = malloc((size_t)rows * mc.dim * sizeof(float));
        float *mgot = malloc((size_t)rows * mc.dim * sizeof(float));
        st_read_f32(&F, "mx", mx, 0);
        st_read_f32(&F, "mout", mref, 0);
        float *idf = malloc((size_t)rows * sizeof(float));
        st_read_f32(&F, "mids", idf, 0);
        int32_t *ids = malloc((size_t)rows * sizeof(int32_t));
        for (int64_t i = 0; i < rows; i++) ids[i] = (int32_t)idf[i];
        free(idf);

        /* 1) the routing: indices must be exact or other experts get applied */
        int *gi = malloc((size_t)rows * mc.topk * sizeof(int));
        float *gwt = malloc((size_t)rows * mc.topk * sizeof(float));
        dsv4_moe_route(&mc, &mw, mx, ids, rows, gi, gwt);
        float *ridx = malloc((size_t)rows * mc.topk * sizeof(float));
        st_read_f32(&F, "midx", ridx, 0);
        int64_t bad = 0;
        for (int64_t i = 0; i < rows * mc.topk; i++)
            if (gi[i] != (int)ridx[i]) bad++;
        if (bad) ok6 = 0;
        printf("  [%s] routing   %lld/%lld indices differ\n",
               bad ? "FALLO" : "ok  ", (long long)bad,
               (long long)(rows * mc.topk));

        /* 2) the whole block */
        t0 = now_s();
        dsv4_moe_forward(&mc, &mw, mx, ids, rows, mgot);
        const double t_moe = now_s() - t0;
        double s7 = 0, r7 = 0, m7 = 0;
        for (int64_t i = 0; i < rows * mc.dim; i++) {
            const double d = fabs((double)mgot[i] - (double)mref[i]);
            if (d > m7) m7 = d;
            s7 += d * d; r7 += (double)mref[i] * mref[i];
        }
        const double rel7 = sqrt(s7 / r7);
        const int okm = (rel7 < 1e-2);
        ok6 &= okm;
        printf("  forward in %.0f ms\n", t_moe * 1e3);
        printf("  [%s] output  rel err %.2e   max abs %.2e\n",
               okm ? "ok  " : "FALLO", rel7, m7);

        for (int i = 0; i < nb; i++) free(eb[i]);
        for (int i = 0; i < nsh; i++) free(shb[i]);
        free(eb); free(E1); free(E2); free(E3);
        free(gwb); free(gb); free(t2e);
        free(mx); free(mref); free(mgot); free(ids);
        free(gi); free(gwt); free(ridx);
    }

    if (ok && ok2 && ok3 && ok4 && ok5 && ok6) {
        printf("\nBOTH of the checkpoint formats are covered:\n"
               "  MXFP4 experts -> colibri matmul_mxfp4, verbatim\n"
               "  dense FP8/UE8M0 -> the decoder colibri was missing\n");
    }

    free(q4); free(e8); free(x); free(y); free(ref);
    (void)tw; (void)ts;
    return (ok && ok2) ? 0 : 1;
}

