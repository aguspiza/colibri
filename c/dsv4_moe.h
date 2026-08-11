/* dsv4_moe.h — the MoE block: router + routed experts + shared expert.
 *
 * This is the piece that meets what makes colibri colibri. The routed experts are
 * 97.5 % of the model's parameters (277B of 284B); here they can be applied from
 * RAM, while in the real engine each one arrives by streaming from disk exactly
 * when the router proves it is needed. The rest of the block -- router, weights,
 * shared expert -- is identical either way.
 *
 * Follows ref/model.py::Gate.forward and MoE.forward.
 */

#ifndef DSV4_MOE_H
#define DSV4_MOE_H

#include "dsv4_math.h"
#include "dsv4_attn.h"     /* dsv4_matmul_ex */
#include "dsv4_stream.h"   /* streaming expert tier */

typedef struct {
    int dim, n_experts, topk, inter;
    float route_scale, swiglu_limit;
    int hash;                  /* early layers: route by table, not by score */
} DsV4MoeCfg;

typedef struct {
    DsV4W gate_w;              /* [n_experts, dim] */
    const float *gate_bias;    /* [n_experts] -- NULL on the hash layers */
    const int32_t *tid2eid;    /* [vocab, topk] -- hash layers only */
    /* Experts resident in RAM: one descriptor each. */
    const DsV4W *e_w1, *e_w2, *e_w3;
    DsV4W s_w1, s_w2, s_w3;             /* shared expert */

    /* STREAMING path. When `store` is non-NULL the routed experts are read from
     * disk through the per-layer cache instead of from `e_w*`. The shared expert
     * stays resident: it is used by every token, so streaming it would be
     * pointless -- which is exactly the distinction colibri draws between the
     * dense set and the routed tier. */
    DsV4Store *store;
    int layer;                          /* which layer to ask the store for */

    /* Third route: a callback that hands back the expert's descriptors. This is
     * what the real engine uses, where the experts live in the checkpoint shards
     * with an LRU cache in between. It is a function pointer so that dsv4_moe.h
     * need know nothing about st.h or about the cache policy. */
    void (*fetch)(void *ctx, int layer, int e, DsV4W *w1, DsV4W *w2, DsV4W *w3);
    void *fetch_ctx;

    /* Optional prefetch: the router knows all `topk` experts BEFORE any of them
     * is applied, so they can be requested in one go and the backend can overlap
     * the reads. Without it they are read one at a time at queue depth 1, which
     * is the worst way to use an NVMe. It changes nothing about the result: it
     * only brings forward work `fetch` would do anyway. */
    void (*prefetch)(void *ctx, int layer, const int *es, int n);
} DsV4MoeW;

/* ---------------------------------------------------------------------------
 * Router. Three details that have to be respected:
 *
 *   1. Scoring runs in fp32, unrounded (`linear(x.float(), w.float())`).
 *   2. THE BIAS ONLY DECIDES, IT DOES NOT WEIGHT. The top-k is chosen on
 *      `score+bias` but weighted with the RAW score. The bias is load balancing
 *      (noaux_tc), not an opinion about the expert. Identical to GLM-5.2's
 *      router in colibri; the only change is sqrtsoftplus instead of sigmoid.
 *   3. Hash layers have NO bias and take their indices from a table keyed by
 *      token id -- but the WEIGHTS are still computed by the router, so the
 *      router matmul runs regardless.
 *
 *   out_idx : [rows, topk]     out_w : [rows, topk]
 * ------------------------------------------------------------------------- */
static inline void dsv4_moe_route(const DsV4MoeCfg *c, const DsV4MoeW *w,
                                  const float *x, const int32_t *ids,
                                  int64_t rows, int *out_idx, float *out_w)
{
    const int E = c->n_experts;
    float *sc = (float *)malloc((size_t)E * sizeof(float));

    for (int64_t r = 0; r < rows; r++) {
        dsv4_matmul_w(sc, x + r * c->dim, &w->gate_w, 1, 0);
        for (int e = 0; e < E; e++) sc[e] = dsv4_sqrtsoftplus(sc[e]);

        int *idx = out_idx + r * c->topk;
        if (c->hash) {
            for (int k = 0; k < c->topk; k++)
                idx[k] = (int)w->tid2eid[(size_t)ids[r] * c->topk + k];
        } else {
            /* top-k by direct selection: with topk<=8 and E=256 this beats sorting */
            for (int k = 0; k < c->topk; k++) {
                int best = -1;
                float bv = -INFINITY;
                for (int e = 0; e < E; e++) {
                    int taken = 0;
                    for (int j = 0; j < k; j++) if (idx[j] == e) { taken = 1; break; }
                    if (taken) continue;
                    const float v = sc[e] + w->gate_bias[e];   /* the bias ONLY decides */
                    if (v > bv) { bv = v; best = e; }
                }
                idx[k] = best;
            }
        }

        float *wt = out_w + r * c->topk;
        float sum = 0.0f;
        for (int k = 0; k < c->topk; k++) { wt[k] = sc[idx[k]]; sum += wt[k]; }
        for (int k = 0; k < c->topk; k++) wt[k] = wt[k] / sum * c->route_scale;
    }
    free(sc);
}

/* ---------------------------------------------------------------------------
 * One expert: SwiGLU with clamping, computed in fp32.
 *
 * The routing weight multiplies the INTERMEDIATE STATE, before w2, not the
 * output (`x = weights * x; return self.w2(x)`). Mathematically equivalent since
 * w2 is linear, but not in bf16: it moves where the rounding lands.
 * ------------------------------------------------------------------------- */
static inline void dsv4_expert_apply(const DsV4MoeCfg *c,
                                     const DsV4W *w1, const DsV4W *w2,
                                     const DsV4W *w3,
                                     const float *x, float wk, float *acc)
{
    float *gate = (float *)malloc((size_t)c->inter * sizeof(float));
    float *up   = (float *)malloc((size_t)c->inter * sizeof(float));
    float *h    = (float *)malloc((size_t)c->inter * sizeof(float));

    dsv4_matmul_w(gate, x, w1, 1, 1);
    dsv4_matmul_w(up,   x, w3, 1, 1);

    for (int i = 0; i < c->inter; i++)
        h[i] = dsv4_to_bf16(wk * dsv4_swiglu(gate[i], up[i], c->swiglu_limit));

    float *outv = (float *)malloc((size_t)c->dim * sizeof(float));
    dsv4_matmul_w(outv, h, w2, 1, 1);
    for (int i = 0; i < c->dim; i++) acc[i] += outv[i];

    free(gate); free(up); free(h); free(outv);
}


/* ---------------------------------------------------------------------------
 * The same expert, applied to SEVERAL rows at once.
 *
 * `dsv4_expert_apply` runs one row, so the union loop below used to call it once
 * per (expert, row) pair -- every call a GEMV that re-walks the expert's weights.
 * colibri's matmul_mxfp4 is already nested the right way for this (output row
 * outside, batch row inside, so an output row's ~2 KB of weights stay in L1
 * across the batch), it was simply never given more than one row.
 *
 * `rows[]` lists which rows of `x` selected this expert and `wk[]` their routing
 * weights. Each row's accumulation order is unchanged -- experts still arrive in
 * union order and the weight still multiplies the intermediate before w2 -- so
 * this agrees bit for bit with the per-row version.
 * ------------------------------------------------------------------------- */
static inline void dsv4_expert_apply_rows(const DsV4MoeCfg *c,
                                          const DsV4W *w1, const DsV4W *w2,
                                          const DsV4W *w3,
                                          const float *x, int64_t dim_stride,
                                          const int *rows, const float *wk,
                                          int nr, float *acc, int64_t acc_stride)
{
    if (nr <= 0) return;
    const int inter = c->inter, dim = c->dim;
    float *xg   = (float *)malloc((size_t)nr * dim * sizeof(float));
    float *gate = (float *)malloc((size_t)nr * inter * sizeof(float));
    float *up   = (float *)malloc((size_t)nr * inter * sizeof(float));
    float *h    = (float *)malloc((size_t)nr * inter * sizeof(float));
    float *outv = (float *)malloc((size_t)nr * dim * sizeof(float));

    for (int i = 0; i < nr; i++)
        memcpy(xg + (size_t)i * dim, x + (size_t)rows[i] * dim_stride,
               (size_t)dim * sizeof(float));

    dsv4_matmul_w(gate, xg, w1, nr, 1);
    dsv4_matmul_w(up,   xg, w3, nr, 1);
    for (int i = 0; i < nr; i++)
        for (int k = 0; k < inter; k++) {
            const size_t j = (size_t)i * inter + k;
            h[j] = dsv4_to_bf16(wk[i] * dsv4_swiglu(gate[j], up[j], c->swiglu_limit));
        }
    dsv4_matmul_w(outv, h, w2, nr, 1);
    for (int i = 0; i < nr; i++) {
        float *dst = acc + (size_t)rows[i] * acc_stride;
        const float *src = outv + (size_t)i * dim;
        for (int d = 0; d < dim; d++) dst[d] += src[d];
    }

    free(xg); free(gate); free(up); free(h); free(outv);
}

/* ---------------------------------------------------------------------------
 * The complete MoE block.
 *
 * `y` accumulates in fp32: routed experts first, the shared one after. The
 * shared expert is ALWAYS applied, with weight 1 -- it is what guarantees every
 * token gets something even if routing goes lopsided.
 * ------------------------------------------------------------------------- */
static inline void dsv4_moe_forward(const DsV4MoeCfg *c, const DsV4MoeW *w,
                                    const float *x, const int32_t *ids,
                                    int64_t rows, float *out)
{
    int *idx = (int *)malloc((size_t)rows * c->topk * sizeof(int));
    float *wt = (float *)malloc((size_t)rows * c->topk * sizeof(float));
    dsv4_moe_route(c, w, x, ids, rows, idx, wt);

    /* WARNING about the hash layers: all `topk` contributions accumulate here,
     * even when two of them point at the same expert. The reference does NOT.
     *
     * `MoE.forward` writes `y[idx] += expert(...)` with `idx` taken from
     * `torch.where(indices == i)`. With repeated indices that expression does
     * not accumulate: it reads, adds and writes back, so the last write wins and
     * one of the two contributions is silently lost.
     *
     * It cannot happen with score top-k (the indices come out distinct by
     * construction), and a trained `tid2eid` table should not repeat either. But
     * if a real checkpoint ever carries duplicates, engine and reference diverge
     * -- and the engine would be the correct one. Worth checking the table at
     * load time.
     *
     * Found because the oracle generated `tid2eid` by sampling WITH replacement:
     * 11 of 64 tokens got the same expert twice and the block came out with an
     * error of 2.2e-1 while the routing itself was exact. */
    memset(out, 0, (size_t)rows * c->dim * sizeof(float));

    /* EXPERT UNION.
     *
     * With several rows, the same expert is usually picked by more than one:
     * measured on the real checkpoint, a batch of 5 tokens asks for 724
     * layer-experts instead of 5x258=1290, i.e. 1.78x fewer reads per token.
     * Iterating row by row, each row would request it separately and 13.4 MB
     * would be read twice.
     *
     * The traversal goes BY EXPERT and, inside, over the rows that selected it.
     * The order is first-appearance, not by id: with rows==1 that reproduces the
     * previous accumulation order exactly, so decode still yields the same
     * bits. */
    const int64_t nslots = rows * c->topk;
    int *uniq = (int *)malloc((size_t)nslots * sizeof(int));
    /* A row can select the same expert more than once on a hash layer, so the
     * per-expert row list is sized for the worst case, not for `rows`. */
    int *erow = (int *)malloc((size_t)nslots * sizeof(int));
    float *ewk = (float *)malloc((size_t)nslots * sizeof(float));
    int nu = 0;
    for (int64_t i = 0; i < nslots; i++) {
        int seen = 0;
        for (int j = 0; j < nu; j++) if (uniq[j] == idx[i]) { seen = 1; break; }
        if (!seen) uniq[nu++] = (int)idx[i];
    }
    if (w->prefetch) w->prefetch(w->fetch_ctx, w->layer, uniq, nu);

    for (int u = 0; u < nu; u++) {
        const int e = uniq[u];
        DsV4W w1, w2, w3;
        if (w->fetch) {
            w->fetch(w->fetch_ctx, w->layer, e, &w1, &w2, &w3);
        } else if (w->store) {
            /* From disk, with an LRU cache. The rest of the computation is
             * identical: where the bytes came from cannot change the result. */
            const int64_t per = (int64_t)c->inter * c->dim;
            const float *blk = dsv4_store_get(w->store, w->layer, e);
            w1 = dsv4_w_f32(blk,           c->inter, c->dim);
            w2 = dsv4_w_f32(blk + per,     c->dim,   c->inter);
            w3 = dsv4_w_f32(blk + 2 * per, c->inter, c->dim);
        } else {
            w1 = w->e_w1[e]; w2 = w->e_w2[e]; w3 = w->e_w3[e];
        }
        /* It is in RAM now: spend it on every row that asked for it before the
         * cache can evict it. That is the whole point of the union.
         *
         * The rows are collected first and applied as ONE batch: with rows == 1
         * this is the same single call as before, and with a prefill chunk it is
         * what lets the expert's weights be walked once instead of once per row. */
        int nr = 0;
        for (int64_t r = 0; r < rows; r++)
            for (int k = 0; k < c->topk; k++)
                if (idx[r * c->topk + k] == e) {
                    erow[nr] = (int)r;
                    ewk[nr] = wt[r * c->topk + k];
                    nr++;
                }
        dsv4_expert_apply_rows(c, &w1, &w2, &w3, x, c->dim, erow, ewk, nr,
                               out, c->dim);
    }
    free(uniq); free(erow); free(ewk);

    /* The shared expert ALWAYS runs, with weight 1, after the routed ones. */
    for (int64_t r = 0; r < rows; r++)
        dsv4_expert_apply(c, &w->s_w1, &w->s_w2, &w->s_w3,
                          x + r * c->dim, 1.0f, out + r * c->dim);
    free(idx); free(wt);
}

#endif /* DSV4_MOE_H */
