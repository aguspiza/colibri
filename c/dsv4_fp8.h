/* dsv4_fp8.h — the dense set: FP8-e4m3 with UE8M0 scales over 128x128 blocks.
 *
 * This was the ONE DeepSeek-V4-Flash format colibri refused on purpose, and
 * colibri.c said so by name:
 *
 *     "DeepSeek-V4 ships the SAME weight layout (FP8 E4M3, 128x128 blocks)
 *      with UE8M0 [...] build does not implement UE8M0 decode: recognizing
 *      the signature and refusing is safer than misreading"
 *
 * So fmt=8 already read this geometry -- raw e4m3 bytes plus one scale per
 * 128x128 block -- but expected the scale as f32, the way Z.ai publishes it for
 * GLM-5.2-FP8. DeepSeek publishes it as UE8M0: one byte, a power of two.
 * `qt_resolve_fmt` spotted that signature and refused rather than misread it,
 * which was the right call -- but it left the dense set unloadable.
 *
 * This is the decoder that was missing. It is small: the scale goes from reading
 * an f32 to computing 2^(e-127). Upstream has since added UE8M0 to fmt=8 itself,
 * which is the better home for it; this stays as the port's own path.
 */

#ifndef DSV4_FP8_H
#define DSV4_FP8_H

#include <math.h>
#include <stdint.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* e4m3-fn decode table, built once.
 *
 * OCP E4M3-FN convention, which is NOT IEEE's: exp==0xF is not reserved for
 * infinity, and only mant==0x7 with exp==0xF is NaN. The largest finite value is
 * therefore exp=0xF, mant=0x6 -> 448. Treating it as IEEE would silently lose
 * the format's top magnitudes.
 */
static float g_dsv4_e4m3[256];
/* UE8M0 scales: the byte IS the exponent, so all 256 fit in a table and the
 * `ldexpf` leaves the hot loop. It used to be one libm call per 128-block per
 * output row -- eight million per token. */
static float g_dsv4_ue8m0[256];
static int g_dsv4_e4m3_ready = 0;

static inline void dsv4_e4m3_init(void) {
    if (g_dsv4_e4m3_ready) return;
    for (int b = 0; b < 256; b++) g_dsv4_ue8m0[b] = ldexpf(1.0f, b - 127);
    for (int b = 0; b < 256; b++) {
        const int sign = (b & 0x80) ? -1 : 1;
        const int exp = (b >> 3) & 0x0F;
        const int man = b & 0x07;
        float v;
        if (exp == 0) {
            v = (float)man / 8.0f * 0.015625f;          /* subnormal: 2^-6 */
        } else if (exp == 0x0F && man == 0x07) {
            v = NAN;
        } else {
            v = (1.0f + (float)man / 8.0f) * ldexpf(1.0f, exp - 7);
        }
        g_dsv4_e4m3[b] = (float)sign * v;
    }
    g_dsv4_e4m3_ready = 1;
}

#define DSV4_FP8_BLOCK 128

static inline int dsv4_nblk(int n) {
    return (n + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;
}

/* y[S,O] = x[S,I] @ W[O,I]^T
 *
 *   q8  : [O, I]                        raw e4m3 bytes
 *   e8s : [ceil(O/128), ceil(I/128)]    UE8M0 scales (1 byte, 2^(e-127))
 *
 * The scale MULTIPLIES (it does not divide), same as colibri's fmt=8 path and as
 * `dequant()` in tools/convert_fp8_to_int4.py, which is the authoritative
 * reference for how these checkpoints are read at the source.
 */
/* One block of e4m3 bytes -> floats. Shared by both nestings below. */
static inline void dsv4_e4m3_block(float *w, const uint8_t *row, int len) {
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256i k7F  = _mm256_set1_epi32(0x7F);
    const __m256i k80  = _mm256_set1_epi32(0x80);
    const __m256i kbia = _mm256_set1_epi32(120 << 23);
    const __m256i k8   = _mm256_set1_epi32(8);
    const __m256  ksub = _mm256_set1_ps(1.0f / 512.0f);
    const __m256  knan = _mm256_set1_ps(NAN);
    for (; i + 8 <= len; i += 8) {
        const __m256i v =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(row + i)));
        const __m256i mag = _mm256_and_si256(v, k7F);
        const __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(v, k80), 24);
        const __m256  nrm = _mm256_castsi256_ps(
            _mm256_add_epi32(_mm256_slli_epi32(mag, 20), kbia));
        const __m256  sub = _mm256_mul_ps(_mm256_cvtepi32_ps(mag), ksub);
        __m256 wv = _mm256_blendv_ps(nrm, sub,
            _mm256_castsi256_ps(_mm256_cmpgt_epi32(k8, mag)));
        wv = _mm256_blendv_ps(wv, knan,
            _mm256_castsi256_ps(_mm256_cmpeq_epi32(mag, k7F)));
        wv = _mm256_or_ps(wv, _mm256_castsi256_ps(sgn));
        _mm256_storeu_ps(w + i, wv);
    }
#endif
    for (; i < len; i++) w[i] = g_dsv4_e4m3[row[i]];
}

/* y[S,O] = x[S,I] @ W[O,I]^T
 *
 *   q8  : [O, I]                        raw e4m3 bytes
 *   e8s : [ceil(O/128), ceil(I/128)]    UE8M0 scales (1 byte, 2^(e-127))
 *
 * The scale MULTIPLIES (it does not divide), same as colibri's fmt=8 path and as
 * `dequant()` in tools/convert_fp8_to_int4.py.
 *
 * TWO NESTINGS, PICKED BY MEASUREMENT.
 *
 * Decode (S == 1) wants the weight bytes decoded straight into the dot product:
 * there is no reuse to amortize, and staging them through a buffer costs a
 * store/load round trip per block. Measured: staging made a 140-token sequential
 * prefill go from 75.6 s of attention to 96.1 s.
 *
 * Prefill (S > 1) wants the opposite. With the input row on the outside a batch is
 * S separate GEMVs, each re-reading all O*I weight bytes -- attention alone is
 * 5.40 GB of them, so batching bought it almost nothing (1.17x). Hoisting the
 * output row out and decoding each block ONCE for all S inputs cuts weight
 * traffic by S: 64.5 s -> 55.8 s on the same prefill.
 *
 * Both nestings accumulate per (s,o) in the same order -- block by block, same
 * partial sums, same scale application -- so they agree bit for bit and the
 * choice cannot change a result.
 */
static inline void dsv4_matmul_fp8_ue8m0(float *y, const float *x,
                                         const uint8_t *q8, const uint8_t *e8s,
                                         int S, int I, int O)
{
    dsv4_e4m3_init();
    const int nbi = dsv4_nblk(I);

    if (S == 1) {
#ifdef _OPENMP
#       pragma omp parallel for schedule(static) if (O >= 256)
#endif
        for (int o = 0; o < O; o++) {
            const uint8_t *row = q8 + (size_t)o * I;
            const uint8_t *sc = e8s + (size_t)(o / DSV4_FP8_BLOCK) * nbi;
            float acc = 0.0f;
            for (int b = 0; b < nbi; b++) {
                const int i0 = b * DSV4_FP8_BLOCK;
                const int i1 = (i0 + DSV4_FP8_BLOCK < I) ? i0 + DSV4_FP8_BLOCK : I;
                float part = 0.0f;
                int k = 0;
#if defined(__AVX2__) && defined(__FMA__)
                /* FUSED decode+FMA: with one input row there is nothing to
                 * amortize, so the decoded weights must never touch memory.
                 * Staging them through a buffer cost 75.6 -> 85.9 s of attention
                 * on a sequential 140-token prefill. */
                const __m256i k7F  = _mm256_set1_epi32(0x7F);
                const __m256i k80  = _mm256_set1_epi32(0x80);
                const __m256i kbia = _mm256_set1_epi32(120 << 23);
                const __m256i k8c  = _mm256_set1_epi32(8);
                const __m256  ksub = _mm256_set1_ps(1.0f / 512.0f);
                const __m256  knan = _mm256_set1_ps(NAN);
                __m256 va = _mm256_setzero_ps();
                for (; k + 8 <= i1 - i0; k += 8) {
                    const __m256i v = _mm256_cvtepu8_epi32(
                        _mm_loadl_epi64((const __m128i *)(row + i0 + k)));
                    const __m256i mag = _mm256_and_si256(v, k7F);
                    const __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(v, k80), 24);
                    const __m256  nrm = _mm256_castsi256_ps(
                        _mm256_add_epi32(_mm256_slli_epi32(mag, 20), kbia));
                    const __m256  sub = _mm256_mul_ps(_mm256_cvtepi32_ps(mag), ksub);
                    __m256 wv = _mm256_blendv_ps(nrm, sub,
                        _mm256_castsi256_ps(_mm256_cmpgt_epi32(k8c, mag)));
                    wv = _mm256_blendv_ps(wv, knan,
                        _mm256_castsi256_ps(_mm256_cmpeq_epi32(mag, k7F)));
                    wv = _mm256_or_ps(wv, _mm256_castsi256_ps(sgn));
                    va = _mm256_fmadd_ps(_mm256_loadu_ps(x + i0 + k), wv, va);
                }
                {
                    __m128 lo = _mm256_castps256_ps128(va);
                    lo = _mm_add_ps(lo, _mm256_extractf128_ps(va, 1));
                    lo = _mm_hadd_ps(lo, lo);
                    lo = _mm_hadd_ps(lo, lo);
                    part = _mm_cvtss_f32(lo);
                }
#endif
                for (; k < i1 - i0; k++) part += x[i0 + k] * g_dsv4_e4m3[row[i0 + k]];
                acc += part * g_dsv4_ue8m0[sc[b]];
            }
            y[o] = acc;
        }
        return;
    }

#ifdef _OPENMP
#   pragma omp parallel for schedule(static) if (O >= 64)
#endif
    for (int o = 0; o < O; o++) {
        const uint8_t *row = q8 + (size_t)o * I;
        const uint8_t *sc = e8s + (size_t)(o / DSV4_FP8_BLOCK) * nbi;

        float acc_s[64];
        float *acc = (S <= 64) ? acc_s : (float *)malloc((size_t)S * sizeof(float));
        for (int s = 0; s < S; s++) acc[s] = 0.0f;

        float wbuf[DSV4_FP8_BLOCK];
        for (int b = 0; b < nbi; b++) {
            const int i0 = b * DSV4_FP8_BLOCK;
            const int i1 = (i0 + DSV4_FP8_BLOCK < I) ? i0 + DSV4_FP8_BLOCK : I;
            const int len = i1 - i0;
            dsv4_e4m3_block(wbuf, row + i0, len);      /* once for all S inputs */
            const float scale = g_dsv4_ue8m0[sc[b]];

            for (int s = 0; s < S; s++) {
                const float *xr = x + (size_t)s * I + i0;
                float part = 0.0f;
                int k = 0;
#if defined(__AVX2__) && defined(__FMA__)
                __m256 va = _mm256_setzero_ps();
                for (; k + 8 <= len; k += 8)
                    va = _mm256_fmadd_ps(_mm256_loadu_ps(xr + k),
                                         _mm256_loadu_ps(wbuf + k), va);
                {
                    __m128 lo = _mm256_castps256_ps128(va);
                    lo = _mm_add_ps(lo, _mm256_extractf128_ps(va, 1));
                    lo = _mm_hadd_ps(lo, lo);
                    lo = _mm_hadd_ps(lo, lo);
                    part = _mm_cvtss_f32(lo);
                }
#endif
                for (; k < len; k++) part += xr[k] * wbuf[k];
                acc[s] += part * scale;
            }
        }

        for (int s = 0; s < S; s++) y[(size_t)s * O + o] = acc[s];
        if (acc != acc_s) free(acc);
    }
}

#endif /* DSV4_FP8_H */
