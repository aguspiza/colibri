/* dsv4_math.h — the primitives DeepSeek-V4-Flash adds.
 *
 * Kept as a separate header so each function can be validated against a
 * reference oracle one at a time, rather than writing 2,000 lines and comparing
 * only the final logits.
 *
 * What follows does NOT exist in colibri's GLM engine:
 *   - mHC: the [b,s,4,dim] residual stream with Sinkhorn (hc_split_sinkhorn)
 *   - the router's sqrtsoftplus scoring (GLM uses sigmoid)
 *   - SwiGLU with clamping (swiglu_limit)
 *
 * Reference: DeepSeek's own hc_split_sinkhorn_kernel (tilelang) and a verified
 * Python port of it (19/19 tests).
 */

#ifndef DSV4_MATH_H
#define DSV4_MATH_H

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define DSV4_MAX_HC 8   /* real hc_mult = 4; slack for experiments */

static inline float dsv4_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---------------------------------------------------------------------------
 * Rounding to bfloat16 (round-to-nearest-even), returned as a float.
 *
 * Not cosmetic: model.py computes the Compressor's pooling in fp32 and then does
 * `self.norm(kv.to(dtype))`, i.e. it casts to bf16 BEFORE normalizing. That cast
 * is part of the model's semantics, and skipping it injects a relative error of
 * ~1.6e-3 that then propagates through the whole stack of layers.
 *
 * The port's general rule: wherever the reference changes precision, the engine
 * has to change it in the same place.
 * ------------------------------------------------------------------------- */
static inline float dsv4_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    if (((u >> 23) & 0xFF) == 0xFF) return x;          /* NaN/Inf untouched */
    const uint32_t r = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    float y;
    memcpy(&y, &r, sizeof y);
    return y;
}

/* ---------------------------------------------------------------------------
 * Router: the score function.
 *
 * GLM-5.2 uses sigmoid; DeepSeek-V4-Flash uses sqrtsoftplus. That is literally
 * the only difference between the two families' routers — the biased top-k
 * (noaux_tc), the renormalization and routed_scaling_factor are identical.
 *
 * `ln(1+exp(x))` overflows to +inf for x > ~88 in f32, and the resulting NaNs
 * poison the top-k silently: no expert ever wins the comparison and the router
 * returns -1. For large x, softplus(x) -> x, so it is cut off there.
 * ------------------------------------------------------------------------- */
static inline float dsv4_sqrtsoftplus(float x) {
    return sqrtf(x > 20.0f ? x : log1pf(expf(x)));
}

/* ---------------------------------------------------------------------------
 * RMSNorm — same as everywhere else in colibri, here for completeness.
 * ------------------------------------------------------------------------- */
static inline void dsv4_rmsnorm(float *out, const float *x, const float *w,
                                int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float sc = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * sc * w[i];
}

/* ---------------------------------------------------------------------------
 * SwiGLU with clamping (`swiglu_limit: 10.0`).
 * kimi_k3.c already has a bounded variant (`situf_`) but with a different
 * formula: this is DeepSeek's, a hard clip before the gate.
 * ------------------------------------------------------------------------- */
static inline float dsv4_swiglu(float gate, float up, float limit) {
    if (limit > 0.0f) {
        /* ASYMMETRIC, and deliberately so (model.py::Expert.forward):
         *   up   = clamp(up, min=-limit, max=limit)   -> both sides
         *   gate = clamp(gate, max=limit)             -> UPPER bound ONLY
         * Clipping `gate` from below as well looks like the natural thing to do
         * and is wrong: SiLU already saturates towards -inf, so the lower bound
         * is unnecessary and adding it changes the function. */
        if (gate > limit) gate = limit;
        if (up   >  limit) up   =  limit;
        if (up   < -limit) up   = -limit;
    }
    return (gate / (1.0f + expf(-gate))) * up;   /* silu(gate) * up */
}

/* ---------------------------------------------------------------------------
 * mHC — hc_split_sinkhorn.
 *
 * Splits mixes[(2+hc)*hc] into (pre, post, comb):
 *
 *   [0    : hc  ] -> pre  = sigmoid(m*scale[0] + base) + eps
 *   [hc   : 2*hc] -> post = 2*sigmoid(m*scale[1] + base)
 *   [2*hc :     ] -> comb = (m*scale[2] + base) as an [hc,hc] matrix
 *
 * and comb then goes through: row softmax, +eps, column normalization, and then
 * (iters-1) rounds of (normalize rows, normalize columns). The result tends
 * towards doubly stochastic.
 *
 * MIND THE ITERATION COUNT: the first round is the softmax one, NOT a full
 * iteration. Running `iters` rounds instead of `iters-1` is a mistake that breaks
 * nothing visibly — comb stays very nearly doubly stochastic — but shifts the
 * numbers just enough that later validation stops matching.
 * ------------------------------------------------------------------------- */
static inline void dsv4_hc_split_sinkhorn(
        const float *mixes,      /* [(2+hc)*hc] */
        const float *hc_scale,   /* [3]         */
        const float *hc_base,    /* [(2+hc)*hc] */
        int hc, int iters, float eps,
        float *pre,              /* [hc]        */
        float *post,             /* [hc]        */
        float *comb)             /* [hc*hc]     */
{
    for (int j = 0; j < hc; j++)
        pre[j] = dsv4_sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + eps;

    for (int j = 0; j < hc; j++)
        post[j] = 2.0f * dsv4_sigmoid(mixes[j + hc] * hc_scale[1] + hc_base[j + hc]);

    const int off = 2 * hc;
    for (int i = 0; i < hc * hc; i++)
        comb[i] = mixes[off + i] * hc_scale[2] + hc_base[off + i];

    /* row softmax (stable: the max is subtracted), then +eps */
    for (int r = 0; r < hc; r++) {
        float *row = comb + (size_t)r * hc;
        float mx = row[0];
        for (int c = 1; c < hc; c++) if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < hc; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }
        const float inv = 1.0f / sum;
        for (int c = 0; c < hc; c++) row[c] = row[c] * inv + eps;
    }

    float acc[DSV4_MAX_HC];

    /* normalize columns */
    for (int c = 0; c < hc; c++) {
        float s = 0.0f;
        for (int r = 0; r < hc; r++) s += comb[(size_t)r * hc + c];
        acc[c] = 1.0f / (s + eps);
    }
    for (int r = 0; r < hc; r++)
        for (int c = 0; c < hc; c++) comb[(size_t)r * hc + c] *= acc[c];

    for (int it = 0; it < iters - 1; it++) {
        for (int r = 0; r < hc; r++) {
            float s = 0.0f;
            for (int c = 0; c < hc; c++) s += comb[(size_t)r * hc + c];
            const float inv = 1.0f / (s + eps);
            for (int c = 0; c < hc; c++) comb[(size_t)r * hc + c] *= inv;
        }
        for (int c = 0; c < hc; c++) {
            float s = 0.0f;
            for (int r = 0; r < hc; r++) s += comb[(size_t)r * hc + c];
            acc[c] = 1.0f / (s + eps);
        }
        for (int r = 0; r < hc; r++)
            for (int c = 0; c < hc; c++) comb[(size_t)r * hc + c] *= acc[c];
    }
}

/* ---------------------------------------------------------------------------
 * mHC — collapse the residual's `hc` copies into the sub-layer's input.
 *   x[d] = sum_m pre[m] * h[m][d]
 * ------------------------------------------------------------------------- */
static inline void dsv4_hc_collapse(float *x, const float *h, const float *pre,
                                    int hc, int dim) {
    memset(x, 0, (size_t)dim * sizeof(float));
    for (int m = 0; m < hc; m++) {
        const float w = pre[m];
        const float *src = h + (size_t)m * dim;
        for (int d = 0; d < dim; d++) x[d] += w * src[d];
    }
}

/* ---------------------------------------------------------------------------
 * mHC — re-expand into `hc` copies after the sub-layer.
 *   h[m][d] = post[m]*y[d] + sum_n comb[m][n]*hres[n][d]
 *
 * `hres` is the residual that went INTO the sub-layer; `h` may only alias it if
 * they do not overlap, so callers are required to keep them distinct.
 * ------------------------------------------------------------------------- */
static inline void dsv4_hc_expand(float *h, const float *y, const float *post,
                                  const float *comb, const float *hres,
                                  int hc, int dim) {
    for (int j = 0; j < hc; j++) {
        float *dst = h + (size_t)j * dim;
        const float pj = post[j];
        for (int d = 0; d < dim; d++) dst[d] = pj * y[d];
        /* comb is used TRANSPOSED: output j accumulates over the first index.
         *
         * model.py::hc_post computes
         *     sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
         * which expands to  y[j] = sum_i comb[i][j] * residual[i]  — not
         * `sum_j comb[i][j]`, which is the natural thing to write and what this
         * code had at first. Because Sinkhorn leaves `comb` doubly stochastic,
         * rows and columns both sum to 1 and the error is nearly invisible in
         * aggregate metrics: it only shows up once the whole block is chained. */
        for (int i = 0; i < hc; i++) {
            const float c = comb[(size_t)i * hc + j];
            const float *src = hres + (size_t)i * dim;
            for (int d = 0; d < dim; d++) dst[d] += c * src[d];
        }
    }
}

/* ---------------------------------------------------------------------------
 * CSA — the `Compressor`'s learned-gate pooling (prefill).
 *
 * This is the central operation of Compressed Sparse Attention: it fuses `ratio`
 * consecutive tokens into one, weighting them with a learned gate:
 *
 *     kv = (kv * (score + ape).softmax(dim=slot)).sum(dim=slot)
 *
 * Three things have to be understood or the numbers do not come out:
 *
 *   1. THE SOFTMAX IS PER CHANNEL, not scalar. `score` has the same shape as
 *      `kv`, so every channel `c` decides its own mixture over the slots. It is
 *      not "weighting tokens", it is "weighting tokens per dimension".
 *
 *   2. `ape` IS A POSITION EMBEDDING WITHIN THE BLOCK ([ratio, coff*d]), added to
 *      the score before the softmax. That is how the compressor tells a group's
 *      first token from its last.
 *
 *   3. OVERLAP MODE (`overlap`, which is when compress_ratio == 4): the
 *      projections emit 2*d channels and each group attends to 2*ratio slots —
 *      the upper half of the channels from its own group, the lower half from the
 *      PREVIOUS group. That way block boundaries do not cut abruptly. The first
 *      group has no predecessor: its low slots are zeroed with score -inf, which
 *      after the softmax weigh exactly 0.
 *
 * Only covers the prefill path (start_pos == 0) with no remainder (seqlen a
 * multiple of ratio). The incremental decode path uses `kv_state`/`score_state`
 * and lives in dsv4_decode.h.
 *
 *   kv, score : [b, s, coff*d]   with coff = overlap ? 2 : 1
 *   ape       : [ratio, coff*d]
 *   out       : [b, s/ratio, d]
 * ------------------------------------------------------------------------- */
static inline void dsv4_compress_prefill(
        const float *kv, const float *score, const float *ape,
        int b, int s, int ratio, int d, int overlap,
        float *out)
{
    const int coff  = overlap ? 2 : 1;
    const int chan  = coff * d;              /* kv/score/ape channels */
    const int nslot = overlap ? 2 * ratio : ratio;
    const int ngrp  = s / ratio;

    float *w = (float *)malloc((size_t)nslot * sizeof(float));
    float *v = (float *)malloc((size_t)nslot * sizeof(float));
    if (!w || !v) { fprintf(stderr, "OOM compress\n"); exit(1); }

    for (int bi = 0; bi < b; bi++) {
        for (int g = 0; g < ngrp; g++) {
            float *dst = out + ((size_t)bi * ngrp + g) * d;

            for (int c = 0; c < d; c++) {
                /* gather (weight, value) from each slot for this channel */
                for (int j = 0; j < nslot; j++) {
                    int t, ch;
                    if (!overlap) {
                        t = g * ratio + j;
                        ch = c;
                    } else if (j >= ratio) {
                        t = g * ratio + (j - ratio);
                        ch = d + c;              /* mitad alta: grupo propio */
                    } else if (g == 0) {
                        w[j] = -INFINITY;        /* no previous group */
                        v[j] = 0.0f;
                        continue;
                    } else {
                        t = (g - 1) * ratio + j;
                        ch = c;                  /* mitad baja: grupo anterior */
                    }
                    const size_t idx = ((size_t)bi * s + t) * chan + ch;
                    /* ape is indexed by the position WITHIN the block */
                    w[j] = score[idx] + ape[(size_t)(t % ratio) * chan + ch];
                    v[j] = kv[idx];
                }

                /* stable softmax over the slots + weighted sum */
                float mx = -INFINITY;
                for (int j = 0; j < nslot; j++) if (w[j] > mx) mx = w[j];
                float sum = 0.0f;
                for (int j = 0; j < nslot; j++) {
                    w[j] = (w[j] == -INFINITY) ? 0.0f : expf(w[j] - mx);
                    sum += w[j];
                }
                const float inv = 1.0f / sum;
                float acc = 0.0f;
                for (int j = 0; j < nslot; j++) acc += w[j] * inv * v[j];
                dst[c] = acc;
            }
        }
    }

    free(w);
    free(v);
}

/* ---------------------------------------------------------------------------
 * Indexer: score the compressed KV and keep the best `keep` entries.
 *
 * GOOD NEWS FOR THE PORT: function for function, this is the same scoring as the
 * DSA indexer colibri already has for GLM-5.2:
 *
 *     d0 = dot(q_h, k_t) * rs;
 *     if (d0 > 0) a += w32[h] * d0;      // ReLU and THEN the per-head weight
 *     isc[t] = a * wsc;                  // wsc = 1/sqrt(n_heads)
 *
 * and DeepSeek-V4-Flash does exactly the same:
 *
 *     index_score = (einsum("bshd,btd->bsht", q, kv).relu_()
 *                    * weights.unsqueeze(-1)).sum(dim=2)
 *     weights = weights_proj(x) * (softmax_scale * n_heads**-0.5)
 *
 * (Since rs > 0, relu(d0*rs) == rs*relu(d0): it does not matter where the scale
 * is applied.) Kimi K3 has none of this — it uses KDA, a delta-rule recurrence
 * with no selection.
 *
 * There are three real differences from GLM, and none of them touches this
 * function:
 *   1. V4 scores against the COMPRESSED KV (the one the indexer's own Compressor
 *      fills); GLM scores against the full latent KV.
 *   2. V4's `q` goes through a Hadamard rotation + FP4 simulation.
 *   3. GLM normalizes the keys with a `k_norm` (LayerNorm); in V4 that job is
 *      done by the `norm` of the indexer's compressor.
 *
 * PRECISION: the scoring runs in BFLOAT16, not f32, and here that is not a
 * cosmetic detail. Computing it in f32 yields a different order — and, when
 * `index_topk` is smaller than the number of compressed blocks, a different SET —
 * so attention would end up looking at different positions. Measured: in f32,
 * 2/512 of the tiny model's indices were wrong; in bf16, none.
 *
 * It is reproduced by accumulating in f32 and rounding to bf16 at every tensor
 * boundary (the einsum's output, the scaled weight, the sum over heads), which is
 * what torch does with bf16 tensors. Verified against the native reference.
 *
 * The selection here is O(keep * T) for clarity. Production uses colibri's
 * `partial_select_desc` (quickselect, O(T) average) with the same tie-break.
 *
 *   q       : [b, s, h, d]   already rotated and quantized
 *   kv      : [b, Tcap, d]   compressed KV; only the first `nvalid` are valid
 *   weights : [b, s, h]      the RAW weights_proj output (the scale goes here)
 *   out     : [b, s, keep]   indices, or -1 for non-causal padding
 * ------------------------------------------------------------------------- */
static inline void dsv4_indexer_topk_ex(
        const float *q, const float *kv, const float *weights,
        int b, int s, int h, int d, int Tcap, int nvalid,
        int ratio, int keep, int offset, float softmax_scale,
        int no_causal,   /* 1 = no mask: everything compressed is past */
        int *out)
{
    const float wsc = softmax_scale / sqrtf((float)h);
    float *sc = (float *)malloc((size_t)nvalid * sizeof(float));
    int *taken = (int *)malloc((size_t)nvalid * sizeof(int));
    if (!sc || !taken) { fprintf(stderr, "OOM indexer\n"); exit(1); }

    for (int bi = 0; bi < b; bi++) {
        for (int si = 0; si < s; si++) {
            /* Causal limit: a position may only look at compressed blocks that
             * are already closed. For the first `ratio-1` positions that is 0,
             * i.e. NOTHING is visible and the whole row comes out -1.
             *
             * In DECODE it does not apply: everything compressed is past by
             * construction. Trying to disable it by passing ratio=1 does not work
             * — that gives limit=1 and clips to a single block — hence the flag. */
            const int limit = no_causal ? nvalid : (si + 1) / ratio;

            for (int t = 0; t < nvalid; t++) {
                if (t >= limit) { sc[t] = -INFINITY; taken[t] = 0; continue; }
                float a = 0.0f;
                for (int hi = 0; hi < h; hi++) {
                    const float *qh = q + (((size_t)bi * s + si) * h + hi) * d;
                    const float *kt = kv + ((size_t)bi * Tcap + t) * d;
                    float dot = 0.0f;
                    for (int i = 0; i < d; i++) dot += qh[i] * kt[i];
                    dot = dsv4_to_bf16(dot);              /* the einsum's output */
                    if (dot > 0.0f) {                     /* ReLU */
                        const float wh = dsv4_to_bf16(
                                weights[((size_t)bi * s + si) * h + hi] * wsc);
                        /* the PRODUCT is rounded before accumulating: the sum
                         * over heads runs in f32 and is rounded at the end */
                        a += dsv4_to_bf16(wh * dot);
                    }
                }
                sc[t] = dsv4_to_bf16(a);
                taken[t] = 0;
            }

            /* selection: highest score first, ties broken by lower index */
            int *dst = out + ((size_t)bi * s + si) * keep;
            for (int k = 0; k < keep; k++) {
                int best = -1;
                for (int t = 0; t < nvalid; t++) {
                    if (taken[t]) continue;
                    if (best < 0 || sc[t] > sc[best]) best = t;
                }
                if (best < 0) { dst[k] = -1; continue; }
                taken[best] = 1;
                /* anything past the causal limit is padding, not a position:
                 * it is marked -1 and `sparse_attn` skips it. */
                dst[k] = (best >= limit) ? -1 : best + offset;
            }
        }
    }

    free(sc);
    free(taken);
}

/* Wrapper with the causal mask: the prefill path, already validated. */
static inline void dsv4_indexer_topk(
        const float *q, const float *kv, const float *weights,
        int b, int s, int h, int d, int Tcap, int nvalid,
        int ratio, int keep, int offset, float softmax_scale, int *out)
{
    dsv4_indexer_topk_ex(q, kv, weights, b, s, h, d, Tcap, nvalid, ratio, keep,
                         offset, softmax_scale, 0, out);
}

/* ---------------------------------------------------------------------------
 * sparse_attn — attention over the positions the indexer selected.
 *
 * A port of DeepSeek's sparse_attn_kernel. Three details have to be respected or
 * the numbers do not add up:
 *
 *   1. `kv` SERVES AS BOTH K AND V ([b,n,d], a single head: the config says
 *      `num_key_value_heads: 1`). This is absorbed MLA attention — there is no
 *      separate V projection.
 *   2. INDICES OF -1 ARE PADDING and are masked to -inf, not to zero.
 *   3. `attn_sink` GOES IN THE DENOMINATOR ONLY: `sum_exp += exp(sink - max)`.
 *      It is a per-head sink — a learned "null token" — that lets a head attend
 *      to nothing. Adding it to the numerator as well is the easy, silent
 *      mistake: the output would still look plausible.
 *
 *   q    : [b, m, h, d]
 *   kv   : [b, n, d]
 *   sink : [h]
 *   idxs : [b, m, topk]   with -1 as padding
 *   out  : [b, m, h, d]
 * ------------------------------------------------------------------------- */
static inline void dsv4_sparse_attn(
        const float *q, const float *kv, const float *sink, const int *idxs,
        int b, int m, int h, int d, int n, int topk, float softmax_scale,
        float *out)
{
    float *e = (float *)malloc((size_t)topk * sizeof(float));
    if (!e) { fprintf(stderr, "OOM sparse_attn\n"); exit(1); }

    for (int bi = 0; bi < b; bi++) {
        for (int mi = 0; mi < m; mi++) {
            const int *sel = idxs + ((size_t)bi * m + mi) * topk;
            for (int hi = 0; hi < h; hi++) {
                const float *qh = q + (((size_t)bi * m + mi) * h + hi) * d;

                float mx = -INFINITY;
                for (int k = 0; k < topk; k++) {
                    const int t = sel[k];
                    if (t < 0 || t >= n) { e[k] = -INFINITY; continue; }
                    const float *kt = kv + ((size_t)bi * n + t) * d;
                    float dot = 0.0f;
                    for (int i = 0; i < d; i++) dot += qh[i] * kt[i];
                    e[k] = dot * softmax_scale;
                    if (e[k] > mx) mx = e[k];
                }
                /* If no position is valid, the max stays 0 (as the kernel does)
                 * and only the sink carries any weight. */
                if (mx == -INFINITY) mx = 0.0f;

                float denom = 0.0f;
                for (int k = 0; k < topk; k++) {
                    e[k] = (e[k] == -INFINITY) ? 0.0f : expf(e[k] - mx);
                    denom += e[k];
                }
                denom += expf(sink[hi] - mx);     /* denominator only */

                float *o = out + (((size_t)bi * m + mi) * h + hi) * d;
                for (int i = 0; i < d; i++) o[i] = 0.0f;
                for (int k = 0; k < topk; k++) {
                    if (e[k] == 0.0f) continue;
                    const int t = sel[k];
                    if (t < 0 || t >= n) continue;
                    const float *kt = kv + ((size_t)bi * n + t) * d;
                    const float w = e[k];
                    for (int i = 0; i < d; i++) o[i] += w * kt[i];
                }
                const float inv = 1.0f / denom;
                for (int i = 0; i < d; i++) o[i] = dsv4_to_bf16(o[i] * inv);
            }
        }
    }
    free(e);
}

/* ---------------------------------------------------------------------------
 * RoPE — positional rotation over the last `rd` channels.
 *
 * model.py applies it IN-PLACE and INTERLEAVED: the channels are read as
 * consecutive pairs (x0,x1) treated as one complex number, not by splitting the
 * vector in half. Same convention as `rope_interleave` in colibri.c.
 *
 *   (x0, x1) * (c, s) = (x0*c - x1*s,  x0*s + x1*c)
 *
 * `inverse` uses the conjugate (un-rotate). It really is needed: the attention
 * output goes through an inverse RoPE before the output projection, which is easy
 * to overlook.
 *
 *   x     : [..., rows, (heads,) rd]  rotated in place
 *   freqs : [rows, rd/2, 2]           (real, imag) per position and pair
 * ------------------------------------------------------------------------- */
static inline void dsv4_rope(float *x, const float *freqs, int outer, int rows,
                             int heads, int rd, int inverse) {
    const int pairs = rd / 2;
    for (int o = 0; o < outer; o++) {
        for (int r = 0; r < rows; r++) {
            for (int h = 0; h < heads; h++) {
                float *v = x + (((size_t)o * rows + r) * heads + h) * rd;
                for (int p = 0; p < pairs; p++) {
                    const float c = freqs[((size_t)r * pairs + p) * 2 + 0];
                    const float s0 = freqs[((size_t)r * pairs + p) * 2 + 1];
                    const float s = inverse ? -s0 : s0;
                    const float a = v[2 * p], b = v[2 * p + 1];
                    /* `apply_rotary_emb` computes in complex f32 but then does
                     * `y.copy_(x)` onto the ORIGINAL tensor, which is bf16: the
                     * result ends up rounded. Without this the RoPE sits at
                     * 1.6e-3 instead of being exact, and the error is amplified
                     * by the later FP4 quantization (values cross to a different
                     * point on the e2m1 grid). */
                    v[2 * p]     = dsv4_to_bf16(a * c - b * s);
                    v[2 * p + 1] = dsv4_to_bf16(a * s + b * c);
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * Blockwise FP8 (e4m3) act_quant, with quant+dequant fused.
 *
 * model.py applies it in place over the KV's NON-rope channels to simulate QAT:
 * the values come back as bf16 but already carrying FP8's loss. The rope channels
 * are left untouched, since positional precision matters there.
 *
 * With `pow2` (scale_fmt="ue8m0") the scale is rounded up to a power of two —
 * which is what the MXFP format does and what the checkpoint ships.
 * ------------------------------------------------------------------------- */
#define DSV4_FP8_MAX 448.0f

static inline float dsv4_quant_e4m3(float v) {
    /* e4m3: 4 exponent bits, 3 mantissa bits, no infinities.
     * Rounded to the nearest representable via power-of-two scaling. */
    if (v == 0.0f || !isfinite(v)) return v;
    const float a = fabsf(v);
    if (a > DSV4_FP8_MAX) return v > 0 ? DSV4_FP8_MAX : -DSV4_FP8_MAX;
    int ex;
    frexpf(a, &ex);                       /* a = m * 2^ex, m en [0.5,1) */
    if (ex < -5) ex = -5;                 /* e4m3's subnormal range */
    const float step = ldexpf(1.0f, ex - 4);   /* 3 mantissa bits + implicit */
    /* nearbyintf, NOT roundf: FP8 hardware rounds to nearest even and `roundf`
     * rounds halves away from zero. It only shows up on exact ties, but there it
     * is wrong: a value landing on 8.5 steps comes out 2.25 instead of 2.0.
     * Measured against the reference. */
    float q = nearbyintf(v / step) * step;
    if (q > DSV4_FP8_MAX) q = DSV4_FP8_MAX;
    if (q < -DSV4_FP8_MAX) q = -DSV4_FP8_MAX;
    return q;
}

/* FP4 e2m1: only 8 representable magnitudes. The grid is scanned in full because
 * that is 8 comparisons; nothing cleverer is worth it. */
#define DSV4_FP4_MAX 6.0f

static inline float dsv4_quant_e2m1(float v) {
    static const float G[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float a = fabsf(v) > DSV4_FP4_MAX ? DSV4_FP4_MAX : fabsf(v);
    float best = G[0], bd = fabsf(a - G[0]);
    for (int k = 1; k < 8; k++) {
        const float d = fabsf(a - G[k]);
        if (d < bd) { bd = d; best = G[k]; }
    }
    return v < 0.0f ? -best : best;
}

typedef enum { DSV4_FP8 = 0, DSV4_FP4 = 1 } DsV4QuantFmt;

/* Blockwise quant+dequant, fused, in place.
 *
 * FP8 and FP4 share EVERYTHING except the rounding grid and the format's maximum:
 * same per-block amax, same scale rounded up to a power of two (ue8m0), same
 * rescale. Keeping them as separate functions invited one of them getting a fix
 * the other did not. */
static inline void dsv4_blockwise_quant(float *x, int64_t rows, int n,
                                        int block, int pow2, DsV4QuantFmt fmt) {
    const float vmax = (fmt == DSV4_FP4) ? DSV4_FP4_MAX : DSV4_FP8_MAX;
    const int nb = n / block;
    for (int64_t r = 0; r < rows; r++) {
        float *row = x + r * n;
        for (int b = 0; b < nb; b++) {
            float *blk = row + (size_t)b * block;
            float amax = 0.0f;
            for (int i = 0; i < block; i++) {
                const float a = fabsf(blk[i]);
                if (a > amax) amax = a;
            }
            float s;
            if (pow2) {
                const float t = amax / vmax;
                s = (t > 0.0f) ? ldexpf(1.0f, (int)ceilf(log2f(t))) : 1.0f;
            } else {
                s = (amax > 0.0f) ? amax / vmax : 1.0f;
            }
            const float inv = 1.0f / s;
            for (int i = 0; i < block; i++) {
                const float q = (fmt == DSV4_FP4) ? dsv4_quant_e2m1(blk[i] * inv)
                                                  : dsv4_quant_e4m3(blk[i] * inv);
                blk[i] = dsv4_to_bf16(q * s);
            }
        }
    }
}

static inline void dsv4_act_quant_inplace(float *x, int64_t rows, int n,
                                          int block, int pow2) {
    dsv4_blockwise_quant(x, rows, n, block, pow2, DSV4_FP8);
}

/* ---------------------------------------------------------------------------
 * Walsh-Hadamard transform over the last dimension, scaled by 1/sqrt(n).
 *
 * model.py::rotate_activation uses it to spread information across dimensions
 * before quantizing to FP4/FP8. It is ORTHOGONAL, so rotating q and k together
 * leaves their dot products unchanged: it exists only to make quantization behave
 * better, not to alter attention.
 *
 * Iterative in-place butterfly. `n` must be a power of two (it is: 128, 256).
 * ------------------------------------------------------------------------- */
static inline void dsv4_hadamard(float *x, int64_t rows, int n) {
    const float scale = 1.0f / sqrtf((float)n);
    for (int64_t r = 0; r < rows; r++) {
        float *v = x + r * n;
        for (int h = 1; h < n; h <<= 1) {
            for (int i = 0; i < n; i += (h << 1)) {
                for (int j = i; j < i + h; j++) {
                    const float a = v[j], b = v[j + h];
                    v[j]     = a + b;
                    v[j + h] = a - b;
                }
            }
        }
        for (int i = 0; i < n; i++) v[i] = dsv4_to_bf16(v[i] * scale);
    }
}

/* ---------------------------------------------------------------------------
 * The complete Compressor (prefill): gated pooling -> bf16 -> RMSNorm -> RoPE ->
 * simulated quantization.
 *
 * The detail that a quick read misses: the compressed KV's RoPE uses the
 * frequencies SUBSAMPLED EVERY `ratio` (`freqs_cis[:cutoff:ratio]`), because each
 * compressed entry stands for a block of `ratio` tokens and its position is that
 * of the block's first token. Using `freqs[g]` instead of
 * `freqs[g*ratio]` da un resultado plausible pero equivocado.
 *
 * `rotate` tells the two compressors apart: the Indexer's rotates with Hadamard
 * and quantizes the whole vector to FP4; the main one quantizes to FP8 only the
 * no-rope.
 *
 *   out : [b, s/ratio, d]
 * ------------------------------------------------------------------------- */
static inline void dsv4_compress_forward(
        const float *kv_in, const float *score, const float *ape,
        const float *norm_w, const float *freqs,
        int b, int s, int ratio, int d, int rd, int overlap, int rotate,
        float eps, float *out)
{
    const int ngrp = s / ratio;
    const int64_t rows = (int64_t)b * ngrp;

    dsv4_compress_prefill(kv_in, score, ape, b, s, ratio, d, overlap, out);

    /* model.py does `self.norm(kv.to(dtype))`: bf16 BEFORE normalizing */
    for (int64_t i = 0; i < rows * d; i++) out[i] = dsv4_to_bf16(out[i]);

    float *tmp = (float *)malloc((size_t)d * sizeof(float));
    for (int64_t r = 0; r < rows; r++) {
        dsv4_rmsnorm(tmp, out + r * d, norm_w, d, eps);
        for (int i = 0; i < d; i++) out[r * d + i] = dsv4_to_bf16(tmp[i]);
    }
    free(tmp);

    /* RoPE with subsampled frequencies: group g sits at position g*ratio of the
     * original sequence. */
    for (int64_t r = 0; r < rows; r++) {
        const int g = (int)(r % ngrp);
        dsv4_rope(out + r * d + (d - rd),
                  freqs + (size_t)(g * ratio) * (rd / 2) * 2,
                  1, 1, 1, rd, 0);
    }

    if (rotate) {
        /* The Indexer's Compressor: Hadamard and FP4 over the whole vector */
        dsv4_hadamard(out, rows, d);
        dsv4_blockwise_quant(out, rows, d, 32, 1, DSV4_FP4);
    } else {
        /* Main Compressor: FP8 on the non-rope channels only. They have to be
         * compacted first because they are not contiguous across rows. */
        const int nrope = d - rd;
        float *t2 = (float *)malloc((size_t)rows * nrope * sizeof(float));
        for (int64_t r = 0; r < rows; r++)
            memcpy(t2 + r * nrope, out + r * d, (size_t)nrope * sizeof(float));
        dsv4_blockwise_quant(t2, rows, nrope, 64, 1, DSV4_FP8);
        for (int64_t r = 0; r < rows; r++)
            memcpy(out + r * d, t2 + r * nrope, (size_t)nrope * sizeof(float));
        free(t2);
    }
}

/* ---------------------------------------------------------------------------
 * freqs_cis with YaRN scaling.
 *
 * A port of model.py::precompute_freqs_cis. YaRN extends the context without
 * retraining: instead of dividing ALL frequencies by `factor` — which would ruin
 * the short-period ones that encode local position — it interpolates only the
 * long-period ones, with a linear ramp between two "correction ranges" derived
 * from beta_fast and beta_slow.
 *
 * DeepSeek-V4-Flash: base 10000, factor 16, original 65536 -> 1M of context. And
 * there is a SECOND table with base 160000 (`compress_rope_theta`) for the
 * compressed KV, because its positions advance in steps of `ratio` and need a
 * different frequency scale.
 *
 *   out : [seqlen, dim/2, 2]  with (cos, sin) per position and pair
 * ------------------------------------------------------------------------- */
static inline void dsv4_precompute_freqs(float *out, int dim, int seqlen,
                                         int original_seq_len, double base,
                                         double factor, double beta_fast,
                                         double beta_slow)
{
    const int half = dim / 2;
    double *freqs = (double *)malloc((size_t)half * sizeof(double));
    if (!freqs) { fprintf(stderr, "OOM freqs\n"); exit(1); }

    for (int i = 0; i < half; i++)
        freqs[i] = 1.0 / pow(base, (double)(2 * i) / (double)dim);

    if (original_seq_len > 0) {
        /* find_correction_range(beta_fast, beta_slow, ...): qué dimensiones
         * necesitan interpolarse y cuáles se dejan intactas. */
        const double c = (double)dim / (2.0 * log(base));
        double lo = floor(c * log((double)original_seq_len /
                                  (beta_fast * 2.0 * M_PI)));
        double hi = ceil(c * log((double)original_seq_len /
                                 (beta_slow * 2.0 * M_PI)));
        if (lo < 0) lo = 0;
        if (hi > dim - 1) hi = dim - 1;
        if (lo == hi) hi += 0.001;

        for (int i = 0; i < half; i++) {
            double ramp = ((double)i - lo) / (hi - lo);
            if (ramp < 0) ramp = 0;
            if (ramp > 1) ramp = 1;
            const double smooth = 1.0 - ramp;
            /* short-period ones (smooth~1) are kept; long ones get divided */
            freqs[i] = freqs[i] / factor * (1.0 - smooth) + freqs[i] * smooth;
        }
    }

    for (int t = 0; t < seqlen; t++)
        for (int i = 0; i < half; i++) {
            const double a = (double)t * freqs[i];
            out[((size_t)t * half + i) * 2 + 0] = (float)cos(a);
            out[((size_t)t * half + i) * 2 + 1] = (float)sin(a);
        }

    free(freqs);
}

#endif /* DSV4_MATH_H */
