/* dsv4_weight.h — un peso, en el formato en que vino del checkpoint.
 *
 * El motor NO puede dequantizar a f32 al cargar: el conjunto denso son 6,7 GiB
 * en FP8, que en f32 serían 26,8 GiB, y esta máquina tiene 31,4 GB de RAM en
 * total. Sumando embeddings no cabe.
 *
 * Así que los pesos se quedan tal cual salieron del fichero y el matmul
 * dequantiza sobre la marcha, sin materializar nunca la matriz. Es la misma
 * decisión que toma colibrì con su struct `QT` y sus `fmt=N`.
 *
 * Cada `kind` corresponde a un formato real del checkpoint de
 * DeepSeek-V4-Flash-0731:
 *
 *   F32     los fixtures del modelo tiny (para no romper las 95 validaciones)
 *   BF16    embeddings, lm_head, normas, router, wkv/wgate de los Compressors
 *   FP8B    el denso: e4m3 + escala UE8M0 por bloque de 128x128
 *   MXFP4   los expertos rutados: nibbles e2m1 + escala ue8m0 por cada 32
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
    const void *w;        /* pesos: float*, uint16_t* o uint8_t* según kind */
    const uint8_t *s;     /* escalas; NULL en F32/BF16 */
    int O, I;             /* forma LÓGICA [O, I] */
} DsV4W;

/* Envoltorios para construir el descriptor sin equivocarse de campo. */
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

/* Sub-matriz de `nrows` filas a partir de `row0`.
 *
 * La necesita la proyección de salida agrupada: `wo_a` es [groups*o_lora, dpg] y
 * cada grupo usa su bloque de filas (`model.py` lo hace con un `.view()` y un
 * einsum). Rebanar filas es válido en los cuatro formatos porque todos son
 * row-major, pero con FP8B hay una condición: las escalas van por bloques de
 * 128 filas, así que `row0` debe ser múltiplo de 128. En el modelo real
 * o_lora=1024 = 8x128, así que cuadra. */
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
            fprintf(stderr, "dsv4_w_rows: FP8B exige row0 multiplo de %d (row0=%d)\n",
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
 * `bf16out` redondea la salida, como hacen las capas Linear del modelo. Las del
 * Compressor van en fp32 y piden 0 (ver dsv4_attn.h). */
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
                /* bf16 -> f32 es exacto y sin tablas: son los 16 bits altos del
                 * f32, así que basta extender a 32 y desplazar. */
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
        /* `matmul_mxfp4` es de colibrì y ya trae rutas AVX-512/AVX2. Es `static`
         * en quant.h, así que no se puede declarar `extern`: hay que incluir
         * quant.h en la misma unidad de traducción y definir DSV4_WITH_MXFP4.
         * Los tests del modelo tiny no lo hacen —no tienen pesos MXFP4— y así se
         * evita arrastrar 1.569 líneas de kernels a un binario que no los usa. */
        matmul_mxfp4(y, x, (const uint8_t *)W->w, W->s, S, I, O);
        if (bf16out)
            for (int64_t i = 0; i < (int64_t)S * O; i++)
                y[i] = dsv4_to_bf16(y[i]);
#else
        fprintf(stderr, "dsv4_matmul_w: MXFP4 requiere quant.h "
                        "y -DDSV4_WITH_MXFP4\n");
        exit(1);
#endif
        break;
    default:
        fprintf(stderr, "dsv4_matmul_w: peso sin inicializar (kind=%d)\n", W->kind);
        exit(1);
    }
}

/* Vector f32 desde cualquier formato de 1-D (normas, bias, attn_sink).
 * `out` debe tener `n` floats. */
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
