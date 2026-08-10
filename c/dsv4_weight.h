/* dsv4_weight.h — a weight, in whatever format it came out of the checkpoint.
 *
 * The engine CANNOT dequantize to f32 at load time: the dense set is 6.7 GiB in
 * FP8, which would be 26.8 GiB as f32, and this host has 31.4 GB of RAM in total.
 * Add the embeddings and it does not fit.
 *
 * So the weights stay exactly as they left the file and the matmul dequantizes on
 * the fly, never materializing the matrix. Same decision colibri makes with its
 * `QT` struct and its `fmt=N` values.
 *
 * Each `kind` maps to a format that actually occurs in the
 * DeepSeek-V4-Flash-0731 checkpoint:
 *
 *   F32     the tiny model's fixtures (so the 95 validations keep working)
 *   BF16    embeddings, lm_head, norms, router, the Compressors' wkv/wgate
 *   FP8B    the dense set: e4m3 + a UE8M0 scale per 128x128 block
 *   MXFP4   the routed experts: e2m1 nibbles + a ue8m0 scale per 32 values
 */

#ifndef DSV4_WEIGHT_H
#define DSV4_WEIGHT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsv4_math.h"
#include "dsv4_fp8.h"

typedef enum {
    DSV4_W_NONE = 0,
    DSV4_W_F32,
    DSV4_W_BF16,
    DSV4_W_FP8B,
    DSV4_W_MXFP4
} DsV4WKind;

typedef struct {
    DsV4WKind kind;
    const void *w;        /* weights: float*, uint16_t* or uint8_t*, per kind */
    const uint8_t *s;     /* scales; NULL for F32/BF16 */
    int O, I;             /* LOGICAL shape [O, I] */
} DsV4W;

/* Wrappers, so the descriptor cannot be built with the fields swapped. */
static inline DsV4W dsv4_w_f32(const float *p, int O, int I) {
    DsV4W w = { DSV4_W_F32, p, NULL, O, I };
    return w;
}
static inline DsV4W dsv4_w_bf16(const void *p, int O, int I) {
    DsV4W w = { DSV4_W_BF16, p, NULL, O, I };
    return w;
}
static inline DsV4W dsv4_w_fp8b(const void *p, const uint8_t *s, int O, int I) {
    DsV4W w = { DSV4_W_FP8B, p, s, O, I };
    return w;
}
static inline DsV4W dsv4_w_mxfp4(const void *p, const uint8_t *s, int O, int I) {
    DsV4W w = { DSV4_W_MXFP4, p, s, O, I };
    return w;
}

/* A sub-matrix of `nrows` rows starting at `row0`.
 *
 * The grouped output projection needs it: `wo_a` is [groups*o_lora, dpg] and each
 * group uses its own block of rows (model.py does it with a `.view()` and an
 * einsum). Slicing rows is valid in all four formats since all are row-major, but
 * FP8B carries a condition: the scales go in blocks of 128 rows, so `row0` must be
 * a multiple of 128. In the real model o_lora=1024 = 8x128, so it lines up. */
static inline DsV4W dsv4_w_rows(const DsV4W *W, int row0, int nrows) {
    DsV4W r = *W;
    r.O = nrows;
    switch (W->kind) {
    case DSV4_W_F32:
        r.w = (const float *)W->w + (size_t)row0 * W->I;
        break;
    case DSV4_W_BF16:
        r.w = (const uint16_t *)W->w + (size_t)row0 * W->I;
        break;
    case DSV4_W_FP8B:
        if (row0 % DSV4_FP8_BLOCK) {
            fprintf(stderr, "dsv4_w_rows: FP8B needs row0 to be a multiple of %d (row0=%d)\n",
                    DSV4_FP8_BLOCK, row0);
            exit(1);
        }
        r.w = (const uint8_t *)W->w + (size_t)row0 * W->I;
        r.s = W->s + (size_t)(row0 / DSV4_FP8_BLOCK) * dsv4_nblk(W->I);
        break;
    case DSV4_W_MXFP4:
        r.w = (const uint8_t *)W->w + (size_t)row0 * (W->I / 2);
        r.s = W->s + (size_t)row0 * (W->I / 32);
        break;
    default:
        fprintf(stderr, "dsv4_w_rows: kind %d\n", W->kind);
        exit(1);
    }
    return r;
}

static inline float dsv4_bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* y[S,O] = x[S,I] @ W^T
 *
 * `bf16out` rounds the output, the way the model's Linear layers do. The
 * Compressor's own layers run in fp32 and pass 0 (see dsv4_attn.h). */
static inline void dsv4_matmul_w(float *y, const float *x, const DsV4W *W,
                                 int S, int bf16out)
{
    const int I = W->I, O = W->O;
    switch (W->kind) {
    case DSV4_W_F32: {
        const float *p = (const float *)W->w;
        for (int s = 0; s < S; s++)
            for (int o = 0; o < O; o++) {
                const float *row = p + (size_t)o * I;
                float acc = 0.0f;
                for (int i = 0; i < I; i++) acc += x[(size_t)s * I + i] * row[i];
                y[(size_t)s * O + o] = bf16out ? dsv4_to_bf16(acc) : acc;
            }
        break;
    }
    case DSV4_W_BF16: {
        const uint16_t *p = (const uint16_t *)W->w;
        for (int s = 0; s < S; s++) {
            const float *xr = x + (size_t)s * I;
#ifdef _OPENMP
#           pragma omp parallel for schedule(static) if (O >= 256)
#endif
            for (int o = 0; o < O; o++) {
                const uint16_t *row = p + (size_t)o * I;
                float acc = 0.0f;
                int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
                /* bf16 -> f32 is exact and needs no table: they are the high 16
                 * bits of the f32, so zero-extend to 32 and shift. */
                __m256 va = _mm256_setzero_ps();
                for (; i + 8 <= I; i += 8) {
                    const __m256i w32 = _mm256_slli_epi32(
                        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(row + i))), 16);
                    va = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i),
                                         _mm256_castsi256_ps(w32), va);
                }
                {
                    __m128 lo = _mm256_castps256_ps128(va);
                    lo = _mm_add_ps(lo, _mm256_extractf128_ps(va, 1));
                    lo = _mm_hadd_ps(lo, lo);
                    lo = _mm_hadd_ps(lo, lo);
                    acc = _mm_cvtss_f32(lo);
                }
#endif
                for (; i < I; i++) acc += xr[i] * dsv4_bf16_to_f32(row[i]);
                y[(size_t)s * O + o] = bf16out ? dsv4_to_bf16(acc) : acc;
            }
        }
        break;
    }
    case DSV4_W_FP8B:
        dsv4_matmul_fp8_ue8m0(y, x, (const uint8_t *)W->w, W->s, S, I, O);
        if (bf16out)
            for (int64_t i = 0; i < (int64_t)S * O; i++) y[i] = dsv4_to_bf16(y[i]);
        break;
    case DSV4_W_MXFP4:
#ifdef DSV4_WITH_MXFP4
        /* `matmul_mxfp4` is colibri's and already has AVX-512/AVX2 paths. It is
         * `static` in quant.h, so it cannot be declared `extern`: quant.h has to
         * be included in the same translation unit with DSV4_WITH_MXFP4 defined.
         * The tiny-model tests do not — they have no MXFP4 weights — which keeps
         * 1,569 lines of kernels out of a binary that never calls them. */
        matmul_mxfp4(y, x, (const uint8_t *)W->w, W->s, S, I, O);
        if (bf16out)
            for (int64_t i = 0; i < (int64_t)S * O; i++)
                y[i] = dsv4_to_bf16(y[i]);
#else
        fprintf(stderr, "dsv4_matmul_w: MXFP4 needs quant.h "
                        "and -DDSV4_WITH_MXFP4\n");
        exit(1);
#endif
        break;
    default:
        fprintf(stderr, "dsv4_matmul_w: uninitialized weight (kind=%d)\n", W->kind);
        exit(1);
    }
}

/* An f32 vector from any 1-D format (norms, biases, attn_sink).
 * `out` must hold `n` floats. */
static inline void dsv4_vec_f32(float *out, const DsV4W *W, int n) {
    switch (W->kind) {
    case DSV4_W_F32:
        memcpy(out, W->w, (size_t)n * sizeof(float));
        break;
    case DSV4_W_BF16: {
        const uint16_t *p = (const uint16_t *)W->w;
        for (int i = 0; i < n; i++) out[i] = dsv4_bf16_to_f32(p[i]);
        break;
    }
    default:
        fprintf(stderr, "dsv4_vec_f32: kind %d no soportado\n", W->kind);
        exit(1);
    }
}

#endif /* DSV4_WEIGHT_H */
