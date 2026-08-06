/* dsv4_fp8.h — el conjunto denso: FP8-e4m3 con escalas UE8M0 por bloques 128x128.
 *
 * Es el ÚNICO formato de DeepSeek-V4-Flash que colibrì rechaza a propósito.
 * `colibri.c:1485` lo dice con nombre y apellidos:
 *
 *     "DeepSeek-V4 ships the SAME weight layout (FP8 E4M3, 128x128 blocks)
 *      with UE8M0 [...] build does not implement UE8M0 decode: recognizing
 *      the signature and refusing is safer than misreading"
 *
 * O sea que su `fmt=8` ya lee esta geometría —bytes e4m3 crudos + una escala por
 * bloque de 128x128— pero espera la escala en **f32**, como la publica Z.ai para
 * GLM-5.2-FP8. DeepSeek la publica en **UE8M0**: 1 byte, potencia de dos.
 * `qt_resolve_fmt` detecta esa firma y la rechaza en vez de malinterpretarla, lo
 * cual es la decisión correcta — pero deja el denso sin cargar.
 *
 * Aquí está el decodificador que falta. Es pequeño: la escala pasa de leer un
 * f32 a calcular 2^(e-127).
 */

#ifndef DSV4_FP8_H
#define DSV4_FP8_H

#include <math.h>
#include <stdint.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* Tabla de decodificación e4m3-fn, construida una vez.
 *
 * Convención OCP E4M3-FN, que NO es la de IEEE: exp==0xF no está reservado para
 * infinito, y sólo mant==0x7 con exp==0xF es NaN. El máximo finito es por tanto
 * exp=0xF, mant=0x6 -> 448. Tratarlo como IEEE perdería los valores más grandes
 * del formato.
 */
static float g_dsv4_e4m3[256];
/* Escalas UE8M0: el byte ES el exponente, así que las 256 caben en una tabla y
 * el `ldexpf` sale del bucle caliente. Era una llamada a libm por cada bloque de
 * 128 y por cada fila de salida — 8 millones por token. */
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
 *   q8  : [O, I]                bytes e4m3 crudos
 *   e8s : [ceil(O/128), ceil(I/128)]   escalas UE8M0 (1 byte, 2^(e-127))
 *
 * La escala MULTIPLICA (no divide), igual que en la ruta fmt=8 de colibrì y que
 * en el `dequant()` de `tools/convert_fp8_to_int4.py`, que es la referencia
 * autoritativa de cómo se leen estos checkpoints en origen.
 */
static inline void dsv4_matmul_fp8_ue8m0(float *y, const float *x,
                                         const uint8_t *q8, const uint8_t *e8s,
                                         int S, int I, int O)
{
    dsv4_e4m3_init();
    const int nbi = dsv4_nblk(I);

    for (int s = 0; s < S; s++) {
        const float *xr = x + (size_t)s * I;
        float *yr = y + (size_t)s * O;
        /* Las filas de salida son independientes: no hay reducción entre ellas,
         * así que el reparto es trivialmente correcto. En decode S==1 y todo el
         * paralelismo disponible está aquí. */
#ifdef _OPENMP
#       pragma omp parallel for schedule(static) if (O >= 256)
#endif
        for (int o = 0; o < O; o++) {
            const uint8_t *row = q8 + (size_t)o * I;
            const uint8_t *sc = e8s + (size_t)(o / DSV4_FP8_BLOCK) * nbi;
            float acc = 0.0f;
            /* Se recorre por bloques de 128 en I: dentro de cada uno la escala
             * es constante, así que sale del bucle interno y el producto se
             * acumula sin multiplicaciones extra. */
            for (int b = 0; b < nbi; b++) {
                const int i0 = b * DSV4_FP8_BLOCK;
                const int i1 = (i0 + DSV4_FP8_BLOCK < I) ? i0 + DSV4_FP8_BLOCK : I;
                float part = 0.0f;
                int i = i0;
#if defined(__AVX2__) && defined(__FMA__)
                /* Los bytes e4m3 se decodifican por `gather` sobre la LUT de 256
                 * floats: cabe entera en L1, que es la razón de que el gather
                 * salga a cuenta aquí y no en general. */
                __m256 va = _mm256_setzero_ps();
                for (; i + 8 <= i1; i += 8) {
                    const __m256i idx =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(row + i)));
                    const __m256 wv = _mm256_i32gather_ps(g_dsv4_e4m3, idx, 4);
                    va = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i), wv, va);
                }
                {   /* suma horizontal */
                    __m128 lo = _mm256_castps256_ps128(va);
                    __m128 hi = _mm256_extractf128_ps(va, 1);
                    lo = _mm_add_ps(lo, hi);
                    lo = _mm_hadd_ps(lo, lo);
                    lo = _mm_hadd_ps(lo, lo);
                    part = _mm_cvtss_f32(lo);
                }
#endif
                for (; i < i1; i++)
                    part += xr[i] * g_dsv4_e4m3[row[i]];
                acc += part * g_dsv4_ue8m0[sc[b]];
            }
            yr[o] = acc;
        }
    }
}

#endif /* DSV4_FP8_H */
