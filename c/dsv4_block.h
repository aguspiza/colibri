/* dsv4_block.h — la capa completa: mHC + atención + mHC + MoE.
 *
 * Sigue `ref/model.py::Block.forward`. El stream residual NO es el clásico
 * `x + f(x)`: se mantienen `hc_mult` copias del estado y cada sub-capa las
 * colapsa a una, calcula, y vuelve a expandirlas con una matriz de mezcla
 * normalizada por Sinkhorn.
 *
 *     residual = x                      [b, s, hc, d]
 *     x, post, comb = hc_pre(x, hc_attn_*)
 *     x = attn(attn_norm(x))
 *     x = hc_post(x, residual, post, comb)
 *     residual = x
 *     x, post, comb = hc_pre(x, hc_ffn_*)
 *     x = ffn(ffn_norm(x))
 *     x = hc_post(x, residual, post, comb)
 */

#ifndef DSV4_BLOCK_H
#define DSV4_BLOCK_H

#include "dsv4_math.h"
#include "dsv4_attn.h"
#include "dsv4_moe.h"

/* ---------------------------------------------------------------------------
 * hc_pre: colapsar las `hc` copias en una.
 *
 * Detalle que se escapa: el matmul de `hc_fn` va sobre el estado SIN
 * normalizar, y la normalización entra después como un escalar
 * (`F.linear(x, hc_fn) * rsqrt`). Y el colapso `sum(pre * x)` también usa el
 * estado sin normalizar. Normalizar primero y luego multiplicar da lo mismo en
 * exacto, pero no en punto flotante — y sobre todo, si se normaliza el estado
 * que se colapsa, el resultado ya no es el mismo.
 *
 *   x    : [hc*d]  (una posición)
 *   out  : [d]     post: [hc]     comb: [hc*hc]
 * ------------------------------------------------------------------------- */
static inline void dsv4_hc_pre(const float *x, const float *hc_fn,
                               const float *hc_scale, const float *hc_base,
                               int hc, int dim, int iters, float eps,
                               float norm_eps,
                               float *out, float *post, float *comb)
{
    const int n = hc * dim;
    const int mix_hc = (2 + hc) * hc;

    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float rsqrt = 1.0f / sqrtf((float)(ss / n) + norm_eps);

    float mixes[64];                       /* mix_hc = 24 con hc_mult = 4 */
    for (int m = 0; m < mix_hc; m++) {
        const float *row = hc_fn + (size_t)m * n;
        float acc = 0.0f;
        for (int i = 0; i < n; i++) acc += row[i] * x[i];
        mixes[m] = acc * rsqrt;
    }

    float pre[DSV4_MAX_HC];
    dsv4_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps,
                           pre, post, comb);

    dsv4_hc_collapse(out, x, pre, hc, dim);
    for (int d = 0; d < dim; d++) out[d] = dsv4_to_bf16(out[d]);  /* .to(dtype) */
}

/* ---------------------------------------------------------------------------
 * hc_head: el colapso FINAL, tras la última capa.
 *
 * Comparte la forma con `hc_pre` (matmul sobre el estado sin normalizar, escala
 * por rsqrt) pero es más simple: no hay Sinkhorn ni `post`/`comb`, sólo una
 * sigmoide. Tiene sentido — ya no hay que reexpandir a `hc` copias, aquí se
 * cierra el stream.
 *
 *   hc_fn : [hc, hc*dim]   hc_base : [hc]   hc_scale : [1]
 * ------------------------------------------------------------------------- */
static inline void dsv4_hc_head(const float *x, const float *hc_fn,
                                const float *hc_scale, const float *hc_base,
                                int hc, int dim, float norm_eps, float hc_eps,
                                float *out)
{
    const int n = hc * dim;
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float rsqrt = 1.0f / sqrtf((float)(ss / n) + norm_eps);

    float pre[DSV4_MAX_HC];
    for (int m = 0; m < hc; m++) {
        const float *row = hc_fn + (size_t)m * n;
        float acc = 0.0f;
        for (int i = 0; i < n; i++) acc += row[i] * x[i];
        pre[m] = dsv4_sigmoid(acc * rsqrt * hc_scale[0] + hc_base[m]) + hc_eps;
    }
    dsv4_hc_collapse(out, x, pre, hc, dim);
    for (int d = 0; d < dim; d++) out[d] = dsv4_to_bf16(out[d]);
}

typedef struct {
    DsV4AttnCfg attn;
    DsV4MoeCfg  moe;
    int hc, sinkhorn_iters;
    float hc_eps, norm_eps;
} DsV4BlockCfg;

typedef struct {
    DsV4AttnW attn;
    DsV4MoeW  moe;
    const float *attn_norm, *ffn_norm;
    const float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    const float *hc_ffn_fn,  *hc_ffn_base,  *hc_ffn_scale;
} DsV4BlockW;

/* x, out : [b, s, hc, dim] — el stream residual con sus `hc` copias */
/* `dbg_route` (opcional, puede ser NULL) recibe los índices que el router del
 * MoE eligió. Sirve para diagnosticar: el top-k es discreto, así que una
 * desviación numérica mínima en su entrada puede cambiar de experto y provocar
 * un error localizado y grande, muy distinto del redondeo repartido. */
static inline void dsv4_block_forward(const DsV4BlockCfg *c,
                                      const DsV4BlockW *w,
                                      const float *x, const int *winidx,
                                      const int32_t *ids,
                                      int b, int s, int ntopk, float *out,
                                      int *dbg_route)
{
    const int dim = c->attn.dim, hc = c->hc;
    const int64_t bs = (int64_t)b * s;
    const size_t stride = (size_t)hc * dim;

    float *coll = malloc((size_t)bs * dim * sizeof(float));
    float *post = malloc((size_t)bs * hc * sizeof(float));
    float *comb = malloc((size_t)bs * hc * hc * sizeof(float));
    float *sub  = malloc((size_t)bs * dim * sizeof(float));
    float *mid  = malloc((size_t)bs * stride * sizeof(float));
    float *tmp  = malloc((size_t)dim * sizeof(float));

    /* --- sub-capa 1: atención ------------------------------------------- */
    for (int64_t r = 0; r < bs; r++)
        dsv4_hc_pre(x + r * stride, w->hc_attn_fn, w->hc_attn_scale,
                    w->hc_attn_base, hc, dim, c->sinkhorn_iters, c->hc_eps,
                    c->norm_eps, coll + r * dim, post + r * hc,
                    comb + r * hc * hc);

    for (int64_t r = 0; r < bs; r++) {
        dsv4_rmsnorm(tmp, coll + r * dim, w->attn_norm, dim, c->norm_eps);
        for (int d = 0; d < dim; d++) coll[r * dim + d] = dsv4_to_bf16(tmp[d]);
    }
    dsv4_attention_prefill(&c->attn, &w->attn, coll, winidx, b, s, ntopk, sub);

    for (int64_t r = 0; r < bs; r++)
        dsv4_hc_expand(mid + r * stride, sub + r * dim, post + r * hc,
                       comb + r * hc * hc, x + r * stride, hc, dim);

    /* --- sub-capa 2: MoE ------------------------------------------------ */
    for (int64_t r = 0; r < bs; r++)
        dsv4_hc_pre(mid + r * stride, w->hc_ffn_fn, w->hc_ffn_scale,
                    w->hc_ffn_base, hc, dim, c->sinkhorn_iters, c->hc_eps,
                    c->norm_eps, coll + r * dim, post + r * hc,
                    comb + r * hc * hc);

    for (int64_t r = 0; r < bs; r++) {
        dsv4_rmsnorm(tmp, coll + r * dim, w->ffn_norm, dim, c->norm_eps);
        for (int d = 0; d < dim; d++) coll[r * dim + d] = dsv4_to_bf16(tmp[d]);
    }
    if (dbg_route) {
        float *dw = malloc((size_t)bs * c->moe.topk * sizeof(float));
        dsv4_moe_route(&c->moe, &w->moe, coll, ids, bs, dbg_route, dw);
        free(dw);
    }
    dsv4_moe_forward(&c->moe, &w->moe, coll, ids, bs, sub);

    for (int64_t r = 0; r < bs; r++)
        dsv4_hc_expand(out + r * stride, sub + r * dim, post + r * hc,
                       comb + r * hc * hc, mid + r * stride, hc, dim);

    free(coll); free(post); free(comb); free(sub); free(mid); free(tmp);
}

#endif /* DSV4_BLOCK_H */
