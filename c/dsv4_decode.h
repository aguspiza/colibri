/* dsv4_decode.h — token-by-token generation (start_pos > 0).
 *
 * Up to here the engine only knew how to *prefill*: process the whole prompt in
 * one go. That is enough to predict ONE token; generating text needs the
 * incremental path, which differs in three places:
 *
 *   1. The window's KV lives in a RING BUFFER of `win` entries. Prefill writes it
 *      in order; in decode each token overwrites position `start_pos % win`. The
 *      indices `sparse_attn` receives are RING positions, not sequence
 *      positions — confusing the two is the classic bug here.
 *
 *   2. The Compressor cannot compress a block until it has all of it, so it
 *      accumulates the individual tokens in `kv_state`/`score_state` and only
 *      emits a compressed entry when `(start_pos+1) % ratio == 0`.
 *
 *   3. The compressed KV's indices no longer carry a causal mask: anything
 *      compressed is by definition in the past.
 *
 * Follows ref/model.py: `Attention.forward` (else branch), `Compressor.forward`
 * (else branch) and `get_window_topk_idxs` / `get_compress_topk_idxs`.
 */

#ifndef DSV4_DECODE_H
#define DSV4_DECODE_H

#include "dsv4_attn.h"

/* State that persists between tokens. In colibri this is what gets written to
 * `.coli_kv` so a conversation reopens WARM, with no re-prefill. */
typedef struct {
    int b, hd, win, ncomp, ratio, coff;
    float *kv;            /* [b, win + ncomp, hd]  ventana (anillo) + comprimidos */
    float *kv_state;      /* [b, coff*ratio, coff*hd]  bloque en curso */
    float *score_state;
    /* The Indexer has its OWN Compressor, with a different dimension and its own
     * cache: it builds a smaller KV to score against. Two different compressions
     * of the same `x`, so two separate states. */
    int i_hd;
    float *i_kv;          /* [b, ncomp, i_hd] */
    float *i_kv_state, *i_score_state;
    int n_written;        /* tokens seen so far, i.e. where to write next */
} DsV4AttnState;

static inline void dsv4_state_init_full(DsV4AttnState *st, int b, int hd,
                                        int win, int max_seq, int ratio,
                                        int i_hd)
{
    memset(st, 0, sizeof *st);
    st->b = b; st->hd = hd; st->win = win; st->ratio = ratio;
    st->coff = (ratio == 4) ? 2 : 1;
    st->ncomp = ratio ? max_seq / ratio : 0;
    st->kv = (float *)calloc((size_t)b * (win + st->ncomp) * hd, sizeof(float));
    if (ratio) {
        const size_t n = (size_t)b * st->coff * ratio * st->coff * hd;
        st->kv_state = (float *)calloc(n, sizeof(float));
        /* score_state starts at -inf: slots with no token yet must not weigh in
         * the pooling softmax. model.py initializes it the same way. */
        st->score_state = (float *)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) st->score_state[i] = -INFINITY;

        if (i_hd > 0 && ratio == 4) {      /* only ratio 4 has an Indexer */
            st->i_hd = i_hd;
            st->i_kv = (float *)calloc((size_t)b * st->ncomp * i_hd, sizeof(float));
            const size_t m = (size_t)b * st->coff * ratio * st->coff * i_hd;
            st->i_kv_state = (float *)calloc(m, sizeof(float));
            st->i_score_state = (float *)malloc(m * sizeof(float));
            for (size_t i = 0; i < m; i++) st->i_score_state[i] = -INFINITY;
        }
    }
}

static inline void dsv4_state_init(DsV4AttnState *st, int b, int hd, int win,
                                   int max_seq, int ratio) {
    dsv4_state_init_full(st, b, hd, win, max_seq, ratio, 0);
}

static inline void dsv4_state_free(DsV4AttnState *st) {
    free(st->kv); free(st->kv_state); free(st->score_state);
    free(st->i_kv); free(st->i_kv_state); free(st->i_score_state);
    memset(st, 0, sizeof *st);
}

/* Window indices in decode: RING positions, oldest to newest. A port of
 * `get_window_topk_idxs` with start_pos > 0.
 *
 *   out : [b, 1, win]   (-1 for slots not written yet)
 */
static inline void dsv4_window_topk_decode(int *out, int b, int win,
                                           int start_pos)
{
    for (int bi = 0; bi < b; bi++) {
        int *row = out + (size_t)bi * win;
        if (start_pos >= win - 1) {
            /* ring is full: read starting just after the last one written */
            const int p = start_pos % win;
            int k = 0;
            for (int i = p + 1; i < win; i++) row[k++] = i;
            for (int i = 0; i <= p; i++) row[k++] = i;
        } else {
            /* has not wrapped yet: 0..start_pos, the rest empty */
            for (int i = 0; i <= start_pos; i++) row[i] = i;
            for (int i = start_pos + 1; i < win; i++) row[i] = -1;
        }
    }
}

/* Compressed-KV indices in decode. No causal mask: what is compressed is already
 * in the past by construction. `offset` is `win`, where the second region starts.
 * Returns how many there are. */
static inline int dsv4_compress_topk_decode(int *out, int b, int ratio,
                                            int start_pos, int offset)
{
    const int n = (start_pos + 1) / ratio;
    for (int bi = 0; bi < b; bi++)
        for (int i = 0; i < n; i++) out[(size_t)bi * n + i] = i + offset;
    return n;
}

/* ---------------------------------------------------------------------------
 * The incremental Compressor.
 *
 * Returns 1 if it emitted a compressed entry (left in `out`), 0 if it is still
 * accumulating. It only closes a block when `(start_pos+1) % ratio` reaches zero,
 * which is precisely why state has to survive between tokens.
 *
 *   kv_in, score : [coff*hd]  the current token's projections
 * ------------------------------------------------------------------------- */
static inline int dsv4_compress_decode_st(float *ks_all, float *ss_all,
                                          int hd, int ratio, int coff,
                                          const float *kv_in,
                                          const float *score, const float *ape,
                                          const float *norm_w, const float *freqs,
                                          int bi, int start_pos, int rd,
                                          int rotate, float eps, float *out)
{
    const int chan = coff * hd;
    const int slots = coff * ratio;
    const int r = start_pos % ratio;

    float *ks = ks_all + (size_t)bi * slots * chan;
    float *ss = ss_all + (size_t)bi * slots * chan;

    /* Which slot this token lands in. With overlap, the ring's upper half is the
     * block in progress and the lower half is the previous one. */
    const int slot = (coff == 2) ? ratio + r : r;
    for (int c = 0; c < chan; c++) {
        ks[(size_t)slot * chan + c] = kv_in[c];
        /* `ape` is indexed by the position WITHIN the block */
        ss[(size_t)slot * chan + c] = score[c] + ape[(size_t)r * chan + c];
    }

    if ((start_pos + 1) % ratio != 0) return 0;      /* bloque aún abierto */

    /* Gated pooling over the slots. With overlap, the LOW channels come from the
     * previous block and the HIGH ones from the current block — the same split
     * `overlap_transform` performs in prefill. */
    const int nslot = slots;
    for (int d = 0; d < hd; d++) {
        float mx = -INFINITY;
        for (int j = 0; j < nslot; j++) {
            const int c = (coff == 2 && j < ratio) ? d : (coff == 2 ? hd + d : d);
            const float v = ss[(size_t)j * chan + c];
            if (v > mx) mx = v;
        }
        float sum = 0.0f, acc = 0.0f;
        for (int j = 0; j < nslot; j++) {
            const int c = (coff == 2 && j < ratio) ? d : (coff == 2 ? hd + d : d);
            const float sv = ss[(size_t)j * chan + c];
            const float e = (sv == -INFINITY) ? 0.0f : expf(sv - mx);
            sum += e;
            acc += e * ks[(size_t)j * chan + c];
        }
        out[d] = acc / sum;
    }

    /* rotate the ring: the block just closed becomes "the previous one" */
    if (coff == 2)
        for (int j = 0; j < ratio; j++) {
            memcpy(ks + (size_t)j * chan, ks + (size_t)(ratio + j) * chan,
                   (size_t)chan * sizeof(float));
            memcpy(ss + (size_t)j * chan, ss + (size_t)(ratio + j) * chan,
                   (size_t)chan * sizeof(float));
        }

    /* norm -> RoPE -> quantization, same as prefill */
    for (int d = 0; d < hd; d++) out[d] = dsv4_to_bf16(out[d]);
    float *tmp = (float *)malloc((size_t)hd * sizeof(float));
    dsv4_rmsnorm(tmp, out, norm_w, hd, eps);
    for (int d = 0; d < hd; d++) out[d] = dsv4_to_bf16(tmp[d]);
    free(tmp);

    /* A block's position is that of its FIRST token: start_pos+1-ratio. */
    dsv4_rope(out + (hd - rd),
              freqs + (size_t)(start_pos + 1 - ratio) * (rd / 2) * 2,
              1, 1, 1, rd, 0);

    if (rotate) {
        dsv4_hadamard(out, 1, hd);
        dsv4_blockwise_quant(out, 1, hd, 32, 1, DSV4_FP4);
    } else {
        dsv4_blockwise_quant(out, 1, hd - rd, 64, 1, DSV4_FP8);
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * One token. `x` is [b, dim]; `out` is [b, dim].
 *
 * Covers the uncompressed path and the compressed-without-indexer one
 * (ratio != 4). The ratio-4 path additionally needs the indexer over the
 * compressed KV, which scores the same way in decode but without a causal mask.
 * ------------------------------------------------------------------------- */
static inline void dsv4_attention_decode(const DsV4AttnCfg *c,
                                         const DsV4AttnW *w,
                                         DsV4AttnState *st,
                                         const float *x, int start_pos,
                                         float *out)
{
    const int b = st->b, qdim = c->heads * c->hd;
    float *qr = (float *)malloc((size_t)b * c->q_lora * sizeof(float));
    float *q  = (float *)malloc((size_t)b * qdim * sizeof(float));
    float *kv = (float *)malloc((size_t)b * c->hd * sizeof(float));
    float *o  = (float *)malloc((size_t)b * qdim * sizeof(float));
    float tmp[8192];

    /* --- q --- */
    dsv4_matmul_w(qr, x, &w->wq_a, b, 1);
    for (int bi = 0; bi < b; bi++) {
        dsv4_rmsnorm(tmp, qr + (size_t)bi * c->q_lora, w->q_norm, c->q_lora, c->eps);
        for (int i = 0; i < c->q_lora; i++)
            qr[(size_t)bi * c->q_lora + i] = dsv4_to_bf16(tmp[i]);
    }
    dsv4_matmul_w(q, qr, &w->wq_b, b, 1);
    for (int bi = 0; bi < b; bi++)
        for (int h = 0; h < c->heads; h++) {
            float *v = q + (size_t)bi * qdim + (size_t)h * c->hd;
            float ss = 0.0f;
            for (int i = 0; i < c->hd; i++) ss += v[i] * v[i];
            const float sc = 1.0f / sqrtf(ss / (float)c->hd + c->eps);
            for (int i = 0; i < c->hd; i++) v[i] = dsv4_to_bf16(v[i] * sc);
            dsv4_rope(v + (c->hd - c->rd),
                      w->freqs + (size_t)start_pos * (c->rd / 2) * 2,
                      1, 1, 1, c->rd, 0);
        }

    /* --- the token's kv, into the ring --- */
    dsv4_matmul_w(kv, x, &w->wkv, b, 1);
    for (int bi = 0; bi < b; bi++) {
        dsv4_rmsnorm(tmp, kv + (size_t)bi * c->hd, w->kv_norm, c->hd, c->eps);
        for (int i = 0; i < c->hd; i++)
            kv[(size_t)bi * c->hd + i] = dsv4_to_bf16(tmp[i]);
        dsv4_rope(kv + (size_t)bi * c->hd + (c->hd - c->rd),
                  w->freqs + (size_t)start_pos * (c->rd / 2) * 2,
                  1, 1, 1, c->rd, 0);
        dsv4_blockwise_quant(kv + (size_t)bi * c->hd, 1, c->hd - c->rd, 64, 1,
                             DSV4_FP8);
        /* the ring: this token overwrites slot start_pos % win */
        memcpy(st->kv + ((size_t)bi * (st->win + st->ncomp)
                         + (start_pos % st->win)) * c->hd,
               kv + (size_t)bi * c->hd, (size_t)c->hd * sizeof(float));
    }

    /* --- compresión incremental --- */
    int ncomp_avail = 0;
    if (c->ratio) {
        const int coff = st->coff;
        float *ck = (float *)malloc((size_t)b * coff * c->hd * sizeof(float));
        float *cs = (float *)malloc((size_t)b * coff * c->hd * sizeof(float));
        dsv4_matmul_w(ck, x, &w->c_wkv, b, 0);
        dsv4_matmul_w(cs, x, &w->c_wgate, b, 0);
        float *emit = (float *)malloc((size_t)c->hd * sizeof(float));
        for (int bi = 0; bi < b; bi++)
            if (dsv4_compress_decode_st(st->kv_state, st->score_state, c->hd,
                                        c->ratio, coff,
                                        ck + (size_t)bi * coff * c->hd,
                                        cs + (size_t)bi * coff * c->hd,
                                        w->c_ape, w->c_norm, w->freqs, bi,
                                        start_pos, c->rd, 0, c->eps, emit))
                memcpy(st->kv + ((size_t)bi * (st->win + st->ncomp) + st->win
                                 + start_pos / c->ratio) * c->hd,
                       emit, (size_t)c->hd * sizeof(float));
        free(ck); free(cs); free(emit);
        ncomp_avail = (start_pos + 1) / c->ratio;

        /* The Indexer's Compressor, alongside: its own compressed KV. */
        if (c->ratio == 4 && st->i_kv) {
            float *ik = (float *)malloc((size_t)b * coff * c->i_hd * sizeof(float));
            float *is = (float *)malloc((size_t)b * coff * c->i_hd * sizeof(float));
            dsv4_matmul_w(ik, x, &w->i_wkv, b, 0);
            dsv4_matmul_w(is, x, &w->i_wgate, b, 0);
            float *ie = (float *)malloc((size_t)c->i_hd * sizeof(float));
            for (int bi = 0; bi < b; bi++)
                if (dsv4_compress_decode_st(st->i_kv_state, st->i_score_state,
                                            c->i_hd, c->ratio, coff,
                                            ik + (size_t)bi * coff * c->i_hd,
                                            is + (size_t)bi * coff * c->i_hd,
                                            w->i_ape, w->i_norm, w->freqs, bi,
                                            start_pos, c->rd, 1 /*Hadamard+FP4*/,
                                            c->eps, ie))
                    memcpy(st->i_kv + ((size_t)bi * st->ncomp
                                       + start_pos / c->ratio) * c->i_hd,
                           ie, (size_t)c->i_hd * sizeof(float));
            free(ik); free(is); free(ie);
        }
    }

    /* --- indices and attention --- */
    /* In layers with an indexer the compressed KV is not attended in full: the
     * indexer picks the best `index_topk`. In decode there is NO causal mask —
     * everything compressed is past — so it only scores and sorts. */
    int keep = ncomp_avail;
    int *ctopk = NULL;
    if (c->ratio == 4 && st->i_kv && ncomp_avail > 0) {
        const int iqdim = c->i_heads * c->i_hd;
        float *iq = (float *)malloc((size_t)b * iqdim * sizeof(float));
        dsv4_matmul_w(iq, qr, &w->i_wq_b, b, 1);
        for (int bi = 0; bi < b; bi++)
            for (int h = 0; h < c->i_heads; h++)
                dsv4_rope(iq + (size_t)bi * iqdim + (size_t)h * c->i_hd
                             + (c->i_hd - c->rd),
                          w->freqs + (size_t)start_pos * (c->rd / 2) * 2,
                          1, 1, 1, c->rd, 0);
        dsv4_hadamard(iq, (int64_t)b * c->i_heads, c->i_hd);
        dsv4_blockwise_quant(iq, (int64_t)b * c->i_heads, c->i_hd, 32, 1,
                             DSV4_FP4);
        float *iw = (float *)malloc((size_t)b * c->i_heads * sizeof(float));
        dsv4_matmul_w(iw, x, &w->i_wproj, b, 1);

        keep = (c->i_topk < ncomp_avail) ? c->i_topk : ncomp_avail;
        ctopk = (int *)malloc((size_t)b * keep * sizeof(int));
        /* `no_causal=1`: in decode everything compressed is past. Passing
         * ratio=1 to "disable" the mask does NOT work — it yields limit=1 and
         * clips to a single block (measured: 2.4e-1 error with two blocks). */
        dsv4_indexer_topk_ex(iq, st->i_kv, iw, b, 1, c->i_heads, c->i_hd,
                             st->ncomp, ncomp_avail, c->ratio, keep,
                             st->win /*offset*/, c->i_scale, 1, ctopk);
        free(iq); free(iw);
    }

    const int ntk = st->win + keep;
    int *tk = (int *)malloc((size_t)b * ntk * sizeof(int));
    for (int bi = 0; bi < b; bi++) {
        dsv4_window_topk_decode(tk + (size_t)bi * ntk, 1, st->win, start_pos);
        if (ctopk)
            memcpy(tk + (size_t)bi * ntk + st->win, ctopk + (size_t)bi * keep,
                   (size_t)keep * sizeof(int));
        else
            for (int i = 0; i < keep; i++)
                tk[(size_t)bi * ntk + st->win + i] = st->win + i;
    }
    free(ctopk);
    dsv4_sparse_attn(q, st->kv, w->attn_sink, tk, b, 1, c->heads, c->hd,
                     st->win + st->ncomp, ntk, c->scale, o);

    for (int bi = 0; bi < b; bi++)
        for (int h = 0; h < c->heads; h++)
            dsv4_rope(o + (size_t)bi * qdim + (size_t)h * c->hd + (c->hd - c->rd),
                      w->freqs + (size_t)start_pos * (c->rd / 2) * 2,
                      1, 1, 1, c->rd, 1);

    /* --- grouped output projection --- */
    const int dpg = qdim / c->groups;
    float *mid = (float *)malloc((size_t)b * c->groups * c->o_lora * sizeof(float));
    for (int g = 0; g < c->groups; g++) {
        const DsV4W wg = dsv4_w_rows(&w->wo_a, g * c->o_lora, c->o_lora);
        for (int bi = 0; bi < b; bi++)
            dsv4_matmul_w(mid + ((size_t)bi * c->groups + g) * c->o_lora,
                          o + (size_t)bi * qdim + (size_t)g * dpg, &wg, 1, 1);
    }
    dsv4_matmul_w(out, mid, &w->wo_b, b, 1);

    free(qr); free(q); free(kv); free(o); free(tk); free(mid);
}

#endif /* DSV4_DECODE_H */
