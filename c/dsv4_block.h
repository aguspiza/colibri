/* dsv4_block.h — the whole layer: mHC + attention + mHC + MoE.
 *
 * Follows ref/model.py::Block.forward. The residual stream is NOT the usual
 * `x + f(x)`: `hc_mult` copies of the state are kept, and each sub-layer
 * collapses them to one, computes, and expands them back through a mixing matrix
 * normalized by Sinkhorn.
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
 * hc_pre: collapse the `hc` copies into one.
 *
 * The detail that is easy to miss: the `hc_fn` matmul runs on the UNNORMALIZED
 * state, and the normalization enters afterwards as a scalar
 * (`F.linear(x, hc_fn) * rsqrt`). The `sum(pre * x)` collapse also uses the
 * unnormalized state. Normalizing first and multiplying after is the same in
 * exact arithmetic but not in floating point — and, more importantly, if the
 * state being collapsed is normalized then the result is simply not the same.
 *
 *   x    : [hc*d]  (one position)
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

    float mixes[64];                       /* mix_hc = 24 when hc_mult = 4 */
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
 * hc_head: the FINAL collapse, after the last layer.
 *
 * Same shape as `hc_pre` (matmul on the unnormalized state, scaled by rsqrt) but
 * simpler: no Sinkhorn and no `post`/`comb`, just a sigmoid. Which makes sense —
 * there is nothing left to re-expand into `hc` copies, this is where the stream
 * closes.
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

/* x, out : [b, s, hc, dim] — the residual stream with its `hc` copies */
/* `dbg_route` (optional, may be NULL) receives the indices the MoE router chose.
 * It exists for diagnosis: the top-k is discrete, so a tiny numerical deviation
 * on its input can flip an expert and produce one large localized error, which
 * looks nothing like evenly spread rounding. */
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

    /* --- sub-layer 1: attention ----------------------------------------- */
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

    /* --- sub-layer 2: MoE ----------------------------------------------- */
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
