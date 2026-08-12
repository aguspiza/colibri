/* dsv4_attn.h — the complete attention block (prefill).
 *
 * Assembles the already-validated primitives from dsv4_math.h following
 * ref/model.py::Attention.forward.
 *
 * The `compress_ratio == 0` path is a pure sliding window, with no Compressor and
 * no Indexer. Compressed layers add the compressed KV and the indexer's indices
 * onto the same skeleton (both validated separately).
 *
 * Precision: model.py runs with bf16 as the default dtype, so every matmul
 * accumulates in f32 and ROUNDS ITS OUTPUT to bf16. Same rule that turned out to
 * be needed in the Compressor and in the indexer.
 */

#ifndef DSV4_ATTN_H
#define DSV4_ATTN_H

#include "dsv4_math.h"
#include "dsv4_weight.h"

/* y[rows, out] = x[rows, in] @ w[out, in]^T
 *
 * `bf16out` decides whether the output is rounded. This is NOT a performance
 * detail: the model's `Linear` layers are bf16 and do round, but the
 * Compressor's are explicitly declared fp32 (model.py, with a comment saying the
 * checkpoint stores them in bf16 and they are promoted to fp32 "for
 * convenience"). Using the wrong variant injects an error that only becomes
 * visible once the result is quantized further down. */
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

/* Sliding-window indices for prefill (`get_window_topk_idxs`).
 *
 * Each position attends to the `win` preceding ones including itself; anything
 * falling before the start of the sequence is marked -1 and `sparse_attn` skips
 * it. With seqlen <= win this reduces to a plain causal mask.
 *
 *   out : [b, s, min(s, win)]
 */
static inline int dsv4_window_topk_prefill_at(int *out, int b, int s, int win,
                                             int pos0) {
    const int ntopk = (s < win) ? s : win;
    for (int bi = 0; bi < b; bi++)
        for (int i = 0; i < s; i++) {
            int *row = out + ((size_t)bi * s + i) * ntopk;
            /* With pos0 > 0 the window reaches before the batch. Those entries are
             * marked -1 here, which is CORRECT ONLY once the caller has nothing there
             * to attend -- i.e. it is still the position-0 case. Prepending the prior
             * window is the piece that makes pos0 > 0 usable. */
            const int start = (i - win + 1 > 0) ? i - win + 1 : 0;
            for (int j = 0; j < ntopk; j++) {
                const int t = start + j;
                row[j] = (t > i) ? -1 : t;
            }
        }
    return ntopk;
}

static inline int dsv4_window_topk_prefill(int *out, int b, int s, int win) {
    return dsv4_window_topk_prefill_at(out, b, s, win, 0);
}

typedef struct {
    int dim, q_lora, heads, hd, rd, groups, o_lora, win;
    float scale, eps;
    /* compression (0 = pure window layer) */
    int ratio, i_heads, i_hd, i_topk;
    float i_scale;
} DsV4AttnCfg;

/* MATRICES are passed as descriptors (`DsV4W`): in the real checkpoint they are
 * FP8/BF16/MXFP4 and do not fit dequantized. VECTORS — norms, `attn_sink`, `ape`
 * — stay f32 because they are a few MB in total and converting them at load time
 * costs nothing. Drawing the line there keeps the refactor down to what actually
 * matters. */
typedef struct {
    DsV4W wq_a, wq_b, wkv, wo_a, wo_b;
    const float *q_norm, *kv_norm, *attn_sink;
    const float *freqs;          /* [max_seq, rd/2, 2] */

    /* Only when compress_ratio != 0. The main Compressor compresses the KV that
     * gets attended to; the Indexer's own Compressor builds a separate, smaller
     * KV to score against. They are two different compressions of the same `x`. */
    DsV4W c_wkv, c_wgate, i_wkv, i_wgate, i_wq_b, i_wproj;
    const float *c_ape, *c_norm, *i_ape, *i_norm;
} DsV4AttnW;

/* x   : [b, s, dim]     the block's input (already normalized by attn_norm)
 * topk: [b, s, ntopk]   window indices (+ the indexer's, when present)
 * out : [b, s, dim]                                                          */
/* ---------------------------------------------------------------------------
 * Prefill capture.
 *
 * Batched prefill is only a replacement for N decode steps if it also leaves the
 * same state behind -- the KV ring, the compressor's in-progress block, the
 * indexer's own compressed KV. Prefill already computes every one of those
 * internally and then frees them, so rather than recomputing (which is the whole
 * cost being avoided) it hands them out through this struct.
 *
 * Any field may be NULL: the caller allocates only what it intends to use, and a
 * layer with compress_ratio == 0 has no compressor buffers to give. Sizes are the
 * caller's responsibility and are documented per field.
 * ------------------------------------------------------------------------- */
typedef struct {
    float *kv;        /* [b*s, hd]         window KV, post-norm/RoPE/quant */
    float *ckv;       /* [b*s, coff*hd]    compressor projections, pre-pool */
    float *csc;       /* [b*s, coff*hd]    its gate scores */
    float *kv_comp;   /* [b, ngrp, hd]     the compressed entries */
    float *ikv;       /* [b*s, coff*i_hd]  indexer compressor projections */
    float *isc;       /* [b*s, coff*i_hd]  its gate scores */
    float *i_kv_comp; /* [b, ngrp, i_hd]   indexer compressed KV */
    int ngrp;         /* blocks actually produced (written by prefill) */
} DsV4Capture;

/* `pos0` is the ABSOLUTE position of the first token of the batch.
 *
 * Everything here used to derive a token's position as `r % s`, i.e. it assumed the
 * batch was the start of the sequence. That is what confines batched attention to the
 * first chunk and forces every later chunk -- and every turn of a conversation, which
 * always continues from a non-zero position -- onto the token-at-a-time path. Measured
 * on the engine: that path leaves 66 % of the wall clock in a single thread.
 *
 * This makes the RoPE and the compressed-block limit absolute, which is necessary but
 * NOT sufficient: with pos0 > 0 the window of the first `win - 1` tokens reaches back
 * before the batch, and the blocks compressed earlier are not in this KV at all. Those
 * need the prior state passed in, and until they are, pos0 > 0 is rejected rather than
 * silently wrong -- see the `todo` line in check_real. */
static inline void dsv4_attention_prefill_cap(const DsV4AttnCfg *c,
                                             const DsV4AttnW *w,
                                             const float *x, const int *topk,
                                             int b, int s, int ntopk, int pos0,
                                             float *out, DsV4Capture *cap)
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

    /* A SECOND normalization of q: RMS with NO weights, per head this time. It
     * does not appear in the config and is extremely easy to miss. */
    for (int64_t r = 0; r < bs; r++) {
        for (int h = 0; h < c->heads; h++) {
            float *v = q + r * qdim + (size_t)h * c->hd;
            float ss = 0.0f;
            for (int i = 0; i < c->hd; i++) ss += v[i] * v[i];
            const float sc = 1.0f / sqrtf(ss / (float)c->hd + c->eps);
            for (int i = 0; i < c->hd; i++) v[i] = dsv4_to_bf16(v[i] * sc);
        }
    }

    /* RoPE over the last rd channels of each head */
    for (int64_t r = 0; r < bs; r++)
        for (int h = 0; h < c->heads; h++) {
            const int pos = pos0 + (int)(r % s);
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
        const int pos = pos0 + (int)(r % s);
        dsv4_rope(kv + r * c->hd + (c->hd - c->rd),
                  w->freqs + (size_t)pos * (c->rd / 2) * 2, 1, 1, 1, c->rd, 0);
    }
    /* Only the NON-rope channels get quantized: positional precision matters in
     * the rope ones, so those stay bf16. */
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
    if (cap && cap->kv)
        memcpy(cap->kv, kv, (size_t)bs * c->hd * sizeof(float));

    /* --- compressed KV and the indexer's indices ------------------------- */
    /* In compressed layers attention looks at TWO regions: the recent sliding
     * window (the `s` tokens as they are) and the compressed blocks behind them.
     * The indexer's indices already arrive shifted by `offset = s`, which is
     * where the second region begins. */
    float *kv_cat = kv;
    const int *tk_use = topk;
    int n_kv = s, ntk = ntopk;
    float *kv_comp = NULL, *tk_buf = NULL;
    int *ctopk = NULL;

    if (c->ratio) {
        const int ngrp = s / c->ratio;
        const int overlap = (c->ratio == 4);
        const int coff = overlap ? 2 : 1;

        /* Main Compressor: fp32, no rounding in its matmuls */
        float *ckv = malloc((size_t)bs * coff * c->hd * sizeof(float));
        float *csc = malloc((size_t)bs * coff * c->hd * sizeof(float));
        dsv4_matmul_w(ckv, x, &w->c_wkv, (int)bs, 0);
        dsv4_matmul_w(csc, x, &w->c_wgate, (int)bs, 0);
        kv_comp = malloc((size_t)b * ngrp * c->hd * sizeof(float));
        if (cap) {
            const size_t nproj = (size_t)bs * coff * c->hd;
            if (cap->ckv) memcpy(cap->ckv, ckv, nproj * sizeof(float));
            if (cap->csc) memcpy(cap->csc, csc, nproj * sizeof(float));
            cap->ngrp = ngrp;
        }
        dsv4_compress_forward(ckv, csc, w->c_ape, w->c_norm, w->freqs,
                              b, s, c->ratio, c->hd, c->rd, overlap, 0,
                              c->eps, kv_comp);
        if (cap && cap->kv_comp)
            memcpy(cap->kv_comp, kv_comp,
                   (size_t)b * ngrp * c->hd * sizeof(float));
        free(ckv); free(csc);

        /* ONLY layers with ratio == 4 have an Indexer. The high-ratio ones —
         * the HCA layers, 128 in the real model — do not learn to select: they
         * attend to ALL causally available compressed blocks, in order. Which
         * makes sense: at ratio 128 there are so few blocks that selecting saves
         * nothing.
         *
         * It is an asymmetry that does not show up in config.json and it changes
         * what the engine has to do: in those layers there is nothing to score. */
        int keep;
        if (c->ratio != 4) {
            keep = ngrp;
            ctopk = malloc((size_t)bs * keep * sizeof(int));
            for (int64_t r = 0; r < bs; r++) {
                const int limit = (pos0 + (int)(r % s) + 1) / c->ratio;
                for (int k = 0; k < keep; k++)
                    ctopk[r * keep + k] = (k >= limit) ? -1 : k + s;
            }
        } else {
            /* The Indexer's Compressor: its own smaller KV, Hadamard+FP4 */
            float *ikv = malloc((size_t)bs * coff * c->i_hd * sizeof(float));
            float *isc = malloc((size_t)bs * coff * c->i_hd * sizeof(float));
            dsv4_matmul_w(ikv, x, &w->i_wkv, (int)bs, 0);
            dsv4_matmul_w(isc, x, &w->i_wgate, (int)bs, 0);
            float *ikvc = malloc((size_t)b * ngrp * c->i_hd * sizeof(float));
            if (cap) {
                const size_t niproj = (size_t)bs * coff * c->i_hd;
                if (cap->ikv) memcpy(cap->ikv, ikv, niproj * sizeof(float));
                if (cap->isc) memcpy(cap->isc, isc, niproj * sizeof(float));
            }
            dsv4_compress_forward(ikv, isc, w->i_ape, w->i_norm, w->freqs,
                                  b, s, c->ratio, c->i_hd, c->rd, overlap, 1,
                                  c->eps, ikvc);
            if (cap && cap->i_kv_comp)
                memcpy(cap->i_kv_comp, ikvc,
                       (size_t)b * ngrp * c->i_hd * sizeof(float));
            free(ikv); free(isc);

            /* the indexer's q: wq_b -> RoPE -> Hadamard -> FP4 */
            const int iqdim = c->i_heads * c->i_hd;
            float *iq = malloc((size_t)bs * iqdim * sizeof(float));
            dsv4_matmul_w(iq, qr, &w->i_wq_b, (int)bs, 1);
            for (int64_t r = 0; r < bs; r++)
                for (int h = 0; h < c->i_heads; h++) {
                    /* The indexer's q needs the absolute position too. This was the
                     * fourth site assuming the batch starts at 0, after the main q,
                     * the kv, and the compressed-block limit -- found by reading, not
                     * by the tests, because every existing test calls with pos0 = 0. */
                    const int pos = pos0 + (int)(r % s);
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

    /* --- sparse attention ----------------------------------------------- */
    dsv4_sparse_attn(q, kv_cat, w->attn_sink, tk_use, b, s, c->heads, c->hd,
                     n_kv, ntk, c->scale, o);

    if (c->ratio) {
        free(kv_comp); free(ctopk); free(tk_buf);
        if (kv_cat != kv) free(kv_cat);
    }

    /* INVERSE RoPE on the output, before the projection.
     *
     * The fifth and last site: it has to undo exactly the rotation applied to q, so
     * if that one is absolute this one must be too. Getting one of the pair wrong
     * would leave a residual rotation -- plausible output, subtly wrong. */
    for (int64_t r = 0; r < bs; r++)
        for (int h = 0; h < c->heads; h++) {
            const int pos = pos0 + (int)(r % s);
            dsv4_rope(o + r * qdim + (size_t)h * c->hd + (c->hd - c->rd),
                      w->freqs + (size_t)pos * (c->rd / 2) * 2,
                      1, 1, 1, c->rd, 1);
        }

    /* --- grouped output projection: o = wo_b(einsum(o, wo_a)) ----------- */
    const int dpg = qdim / c->groups;              /* channels per group */
    float *mid = malloc((size_t)bs * c->groups * c->o_lora * sizeof(float));
    for (int g = 0; g < c->groups; g++) {
        /* `wo_a` viewed as [groups, o_lora, dpg]: each group is its own block of
         * rows. The descriptor is sliced instead of indexing bytes by hand, so it
         * works the same for f32, BF16 or FP8. */
        const DsV4W wg = dsv4_w_rows(&w->wo_a, g * c->o_lora, c->o_lora);
        for (int64_t r = 0; r < bs; r++)
            dsv4_matmul_w(mid + (r * c->groups + g) * c->o_lora,
                          o + r * qdim + (size_t)g * dpg, &wg, 1, 1);
    }
    dsv4_matmul_w(out, mid, &w->wo_b, (int)bs, 1);

    free(qr); free(q); free(kv); free(o); free(mid);
}

/* The original entry point: prefill with nothing captured. */
static inline void dsv4_attention_prefill(const DsV4AttnCfg *c,
                                          const DsV4AttnW *w,
                                          const float *x, const int *topk,
                                          int b, int s, int ntopk, float *out)
{
    dsv4_attention_prefill_cap(c, w, x, topk, b, s, ntopk, 0, out, NULL);
}

#endif /* DSV4_ATTN_H */
