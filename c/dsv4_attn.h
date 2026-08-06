/* dsv4_attn.h — el bloque de atención completo (prefill).
 *
 * Ensambla las primitivas ya validadas de `dsv4_math.h` siguiendo
 * `ref/model.py::Attention.forward` (líneas 497-548).
 *
 * Cubre el camino `compress_ratio == 0`: ventana deslizante pura, sin
 * Compressor ni Indexer. Las capas con compresión añaden la KV comprimida y los
 * índices del indexer al mismo esqueleto (ambos ya validados por separado).
 *
 * Precisión: `model.py` corre con dtype por defecto bf16, así que cada matmul
 * acumula en f32 y **redondea la salida a bf16**. Es la misma regla que hizo
 * falta en el Compressor y en el indexer.
 */

#ifndef DSV4_ATTN_H
#define DSV4_ATTN_H

#include "dsv4_math.h"
#include "dsv4_weight.h"

/* y[rows, out] = x[rows, in] @ w[out, in]^T
 *
 * `bf16out` decide si la salida se redondea. NO es un detalle de rendimiento:
 * las capas `Linear` del modelo son bf16 y redondean, pero las del Compressor
 * se declaran explícitamente en fp32 (`model.py:303-304`, con un comentario que
 * dice que en el checkpoint van en bf16 y aquí se suben a fp32 "por
 * comodidad"). Usar la variante equivocada mete un error que sólo se ve al
 * cuantizar después. */
static inline void dsv4_matmul_ex(float *y, const float *x, const float *w,
                                  int64_t rows, int in, int out, int bf16out) {
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * in;
        float *yr = y + r * out;
        for (int o = 0; o < out; o++) {
            const float *wo = w + (size_t)o * in;
            float acc = 0.0f;
            for (int i = 0; i < in; i++) acc += xr[i] * wo[i];
            yr[o] = bf16out ? dsv4_to_bf16(acc) : acc;
        }
    }
}

static inline void dsv4_matmul(float *y, const float *x, const float *w,
                               int64_t rows, int in, int out) {
    dsv4_matmul_ex(y, x, w, rows, in, out, 1);
}

/* Índices de la ventana deslizante en prefill (`get_window_topk_idxs`).
 *
 * Cada posición atiende a las `win` anteriores incluida ella misma; lo que caiga
 * antes del principio de la secuencia se marca -1 y `sparse_attn` lo ignora.
 * Con seqlen <= win queda una máscara causal pura.
 *
 *   out : [b, s, min(s, win)]
 */
static inline int dsv4_window_topk_prefill(int *out, int b, int s, int win) {
    const int ntopk = (s < win) ? s : win;
    for (int bi = 0; bi < b; bi++)
        for (int i = 0; i < s; i++) {
            int *row = out + ((size_t)bi * s + i) * ntopk;
            const int start = (i - win + 1 > 0) ? i - win + 1 : 0;
            for (int j = 0; j < ntopk; j++) {
                const int t = start + j;
                row[j] = (t > i) ? -1 : t;
            }
        }
    return ntopk;
}

typedef struct {
    int dim, q_lora, heads, hd, rd, groups, o_lora, win;
    float scale, eps;
    /* compresión (0 = capa de ventana pura) */
    int ratio, i_heads, i_hd, i_topk;
    float i_scale;
} DsV4AttnCfg;

/* Las MATRICES van como descriptor (`DsV4W`): en el checkpoint real están en
 * FP8/BF16/MXFP4 y no caben dequantizadas. Los VECTORES —normas, `attn_sink`,
 * `ape`— se quedan en f32 porque ocupan unos pocos MB en total y convertirlos al
 * cargar no cuesta nada. Distinguirlos así recorta el refactor a lo que de
 * verdad importa. */
typedef struct {
    DsV4W wq_a, wq_b, wkv, wo_a, wo_b;
    const float *q_norm, *kv_norm, *attn_sink;
    const float *freqs;          /* [max_seq, rd/2, 2] */

    /* Sólo si compress_ratio != 0. El Compressor principal comprime la KV que
     * se atiende; el del Indexer construye una KV propia, más pequeña, contra
     * la que puntuar. Son dos compresiones distintas del mismo `x`. */
    DsV4W c_wkv, c_wgate, i_wkv, i_wgate, i_wq_b, i_wproj;
    const float *c_ape, *c_norm, *i_ape, *i_norm;
} DsV4AttnW;

/* x   : [b, s, dim]     entrada del bloque (ya normalizada por attn_norm)
 * topk: [b, s, ntopk]   índices de ventana (+ los del indexer si los hubiera)
 * out : [b, s, dim]                                                          */
static inline void dsv4_attention_prefill(const DsV4AttnCfg *c,
                                          const DsV4AttnW *w,
                                          const float *x, const int *topk,
                                          int b, int s, int ntopk, float *out)
{
    const int64_t bs = (int64_t)b * s;
    const int qdim = c->heads * c->hd;

    float *qr = malloc((size_t)bs * c->q_lora * sizeof(float));
    float *q  = malloc((size_t)bs * qdim * sizeof(float));
    float *kv = malloc((size_t)bs * c->hd * sizeof(float));
    float *o  = malloc((size_t)bs * qdim * sizeof(float));

    /* --- q: wq_a -> q_norm -> wq_b ------------------------------------- */
    dsv4_matmul_w(qr, x, &w->wq_a, (int)bs, 1);
    for (int64_t r = 0; r < bs; r++) {
        float tmp[4096];
        dsv4_rmsnorm(tmp, qr + r * c->q_lora, w->q_norm, c->q_lora, c->eps);
        for (int i = 0; i < c->q_lora; i++)
            qr[r * c->q_lora + i] = dsv4_to_bf16(tmp[i]);
    }
    dsv4_matmul_w(q, qr, &w->wq_b, (int)bs, 1);

    /* SEGUNDA normalización de q: RMS **sin pesos**, ya por cabeza. No sale en
     * la config y es facilísima de saltarse — `model.py:504`. */
    for (int64_t r = 0; r < bs; r++) {
        for (int h = 0; h < c->heads; h++) {
            float *v = q + r * qdim + (size_t)h * c->hd;
            float ss = 0.0f;
            for (int i = 0; i < c->hd; i++) ss += v[i] * v[i];
            const float sc = 1.0f / sqrtf(ss / (float)c->hd + c->eps);
            for (int i = 0; i < c->hd; i++) v[i] = dsv4_to_bf16(v[i] * sc);
        }
    }

    /* RoPE sobre los últimos rd canales de cada cabeza */
    for (int64_t r = 0; r < bs; r++)
        for (int h = 0; h < c->heads; h++) {
            const int pos = (int)(r % s);
            dsv4_rope(q + r * qdim + (size_t)h * c->hd + (c->hd - c->rd),
                      w->freqs + (size_t)pos * (c->rd / 2) * 2,
                      1, 1, 1, c->rd, 0);
        }

    /* --- kv: wkv -> kv_norm -> rope -> act_quant ------------------------ */
    dsv4_matmul_w(kv, x, &w->wkv, (int)bs, 1);
    for (int64_t r = 0; r < bs; r++) {
        float tmp[4096];
        dsv4_rmsnorm(tmp, kv + r * c->hd, w->kv_norm, c->hd, c->eps);
        for (int i = 0; i < c->hd; i++) kv[r * c->hd + i] = dsv4_to_bf16(tmp[i]);
    }
    for (int64_t r = 0; r < bs; r++) {
        const int pos = (int)(r % s);
        dsv4_rope(kv + r * c->hd + (c->hd - c->rd),
                  w->freqs + (size_t)pos * (c->rd / 2) * 2, 1, 1, 1, c->rd, 0);
    }
    /* Sólo los canales NO-rope se cuantizan: en los de rope la precisión
     * posicional importa y se dejan en bf16. */
    {
        const int nrope = c->hd - c->rd;
        float *tmp = malloc((size_t)bs * nrope * sizeof(float));
        for (int64_t r = 0; r < bs; r++)
            memcpy(tmp + r * nrope, kv + r * c->hd, (size_t)nrope * sizeof(float));
        dsv4_act_quant_inplace(tmp, bs, nrope, 64, 1);
        for (int64_t r = 0; r < bs; r++)
            memcpy(kv + r * c->hd, tmp + r * nrope, (size_t)nrope * sizeof(float));
        free(tmp);
    }

    /* --- KV comprimida e índices del indexer ----------------------------- */
    /* En las capas con compresión la atención mira a DOS zonas: la ventana
     * deslizante reciente (los `s` tokens tal cual) y los bloques comprimidos
     * que vienen detrás. Los índices del indexer ya llegan desplazados por
     * `offset = s`, que es donde empieza la segunda zona. */
    float *kv_cat = kv;
    const int *tk_use = topk;
    int n_kv = s, ntk = ntopk;
    float *kv_comp = NULL, *tk_buf = NULL;
    int *ctopk = NULL;

    if (c->ratio) {
        const int ngrp = s / c->ratio;
        const int overlap = (c->ratio == 4);
        const int coff = overlap ? 2 : 1;

        /* Compressor principal: fp32, sin redondeo en los matmuls */
        float *ckv = malloc((size_t)bs * coff * c->hd * sizeof(float));
        float *csc = malloc((size_t)bs * coff * c->hd * sizeof(float));
        dsv4_matmul_w(ckv, x, &w->c_wkv, (int)bs, 0);
        dsv4_matmul_w(csc, x, &w->c_wgate, (int)bs, 0);
        kv_comp = malloc((size_t)b * ngrp * c->hd * sizeof(float));
        dsv4_compress_forward(ckv, csc, w->c_ape, w->c_norm, w->freqs,
                              b, s, c->ratio, c->hd, c->rd, overlap, 0,
                              c->eps, kv_comp);
        free(ckv); free(csc);

        /* SÓLO las capas con ratio == 4 tienen Indexer (`model.py:474`). Las de
         * ratio alto —las HCA, con 128 en el modelo real— no aprenden a
         * seleccionar: atienden a TODOS los bloques comprimidos causalmente
         * disponibles, en orden. Tiene sentido: con ratio 128 hay tan pocos
         * bloques que seleccionar no ahorra nada.
         *
         * Es una asimetría que no se ve en el config.json y que cambia el
         * trabajo del motor: en esas capas no hay que puntuar nada. */
        int keep;
        if (c->ratio != 4) {
            keep = ngrp;
            ctopk = malloc((size_t)bs * keep * sizeof(int));
            for (int64_t r = 0; r < bs; r++) {
                const int limit = ((int)(r % s) + 1) / c->ratio;
                for (int k = 0; k < keep; k++)
                    ctopk[r * keep + k] = (k >= limit) ? -1 : k + s;
            }
        } else {
            /* Compressor del Indexer: su propia KV, más pequeña, Hadamard+FP4 */
            float *ikv = malloc((size_t)bs * coff * c->i_hd * sizeof(float));
            float *isc = malloc((size_t)bs * coff * c->i_hd * sizeof(float));
            dsv4_matmul_w(ikv, x, &w->i_wkv, (int)bs, 0);
            dsv4_matmul_w(isc, x, &w->i_wgate, (int)bs, 0);
            float *ikvc = malloc((size_t)b * ngrp * c->i_hd * sizeof(float));
            dsv4_compress_forward(ikv, isc, w->i_ape, w->i_norm, w->freqs,
                                  b, s, c->ratio, c->i_hd, c->rd, overlap, 1,
                                  c->eps, ikvc);
            free(ikv); free(isc);

            /* q del indexer: wq_b -> RoPE -> Hadamard -> FP4 */
            const int iqdim = c->i_heads * c->i_hd;
            float *iq = malloc((size_t)bs * iqdim * sizeof(float));
            dsv4_matmul_w(iq, qr, &w->i_wq_b, (int)bs, 1);
            for (int64_t r = 0; r < bs; r++)
                for (int h = 0; h < c->i_heads; h++) {
                    const int pos = (int)(r % s);
                    dsv4_rope(iq + r * iqdim + (size_t)h * c->i_hd
                                 + (c->i_hd - c->rd),
                              w->freqs + (size_t)pos * (c->rd / 2) * 2,
                              1, 1, 1, c->rd, 0);
                }
            dsv4_hadamard(iq, bs * c->i_heads, c->i_hd);
            dsv4_blockwise_quant(iq, bs * c->i_heads, c->i_hd, 32, 1, DSV4_FP4);

            float *iw = malloc((size_t)bs * c->i_heads * sizeof(float));
            dsv4_matmul_w(iw, x, &w->i_wproj, (int)bs, 1);

            keep = (c->i_topk < ngrp) ? c->i_topk : ngrp;
            ctopk = malloc((size_t)bs * keep * sizeof(int));
            dsv4_indexer_topk(iq, ikvc, iw, b, s, c->i_heads, c->i_hd, ngrp,
                              ngrp, c->ratio, keep, s /* offset */,
                              c->i_scale, ctopk);
            free(iq); free(iw); free(ikvc);
        }

        /* concatenar: KV = [ventana | comprimido], topk = [ventana | indexer] */
        n_kv = s + ngrp;
        kv_cat = malloc((size_t)b * n_kv * c->hd * sizeof(float));
        for (int bi = 0; bi < b; bi++) {
            memcpy(kv_cat + (size_t)bi * n_kv * c->hd,
                   kv + (size_t)bi * s * c->hd,
                   (size_t)s * c->hd * sizeof(float));
            memcpy(kv_cat + ((size_t)bi * n_kv + s) * c->hd,
                   kv_comp + (size_t)bi * ngrp * c->hd,
                   (size_t)ngrp * c->hd * sizeof(float));
        }
        ntk = ntopk + keep;
        tk_buf = malloc((size_t)bs * ntk * sizeof(float));
        int *tkc = (int *)tk_buf;
        for (int64_t r = 0; r < bs; r++) {
            memcpy(tkc + r * ntk, topk + r * ntopk, (size_t)ntopk * sizeof(int));
            memcpy(tkc + r * ntk + ntopk, ctopk + r * keep,
                   (size_t)keep * sizeof(int));
        }
        tk_use = tkc;
    }

    /* --- atención dispersa ---------------------------------------------- */
    dsv4_sparse_attn(q, kv_cat, w->attn_sink, tk_use, b, s, c->heads, c->hd,
                     n_kv, ntk, c->scale, o);

    if (c->ratio) {
        free(kv_comp); free(ctopk); free(tk_buf);
        if (kv_cat != kv) free(kv_cat);
    }

    /* RoPE INVERSA sobre la salida antes de la proyección (`model.py:539`) */
    for (int64_t r = 0; r < bs; r++)
        for (int h = 0; h < c->heads; h++) {
            const int pos = (int)(r % s);
            dsv4_rope(o + r * qdim + (size_t)h * c->hd + (c->hd - c->rd),
                      w->freqs + (size_t)pos * (c->rd / 2) * 2,
                      1, 1, 1, c->rd, 1);
        }

    /* --- proyección de salida agrupada: o = wo_b(einsum(o, wo_a)) ------- */
    const int dpg = qdim / c->groups;              /* canales por grupo */
    float *mid = malloc((size_t)bs * c->groups * c->o_lora * sizeof(float));
    for (int g = 0; g < c->groups; g++) {
        /* `wo_a` visto como [groups, o_lora, dpg]: cada grupo es su bloque de
         * filas. Se rebana el descriptor en vez de indexar bytes a mano, así
         * funciona igual con f32, BF16 o FP8. */
        const DsV4W wg = dsv4_w_rows(&w->wo_a, g * c->o_lora, c->o_lora);
        for (int64_t r = 0; r < bs; r++)
            dsv4_matmul_w(mid + (r * c->groups + g) * c->o_lora,
                          o + r * qdim + (size_t)g * dpg, &wg, 1, 1);
    }
    dsv4_matmul_w(out, mid, &w->wo_b, (int)bs, 1);

    free(qr); free(q); free(kv); free(o); free(mid);
}

#endif /* DSV4_ATTN_H */
