/* dsv4_math.h — las primitivas nuevas de DeepSeek-V4-Flash.
 *
 * Fase 2 del port. Esto es código destinado a `c/deepseek_v4.c` en colibrì; se
 * mantiene aparte mientras se valida función a función contra el oráculo de la
 * fase 1 (`ref/oracle_tiny.py`), en vez de escribir 2.000 líneas y comparar
 * sólo los logits al final.
 *
 * Aquí está lo que NO existe en el motor GLM de colibrì:
 *   - mHC: el stream residual [b,s,4,dim] con Sinkhorn (hc_split_sinkhorn)
 *   - el scoring sqrtsoftplus del router (GLM usa sigmoid)
 *   - SwiGLU con clamp (swiglu_limit)
 *
 * Referencia: `ref/kernel.py::hc_split_sinkhorn_kernel` (tilelang) y
 * `ref/cpu_kernel/kernel.py` (port Python verificado, 19/19 tests).
 */

#ifndef DSV4_MATH_H
#define DSV4_MATH_H

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define DSV4_MAX_HC 8   /* hc_mult real = 4; holgura para experimentar */

static inline float dsv4_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---------------------------------------------------------------------------
 * Redondeo a bfloat16 (round-to-nearest-even), devuelto como float.
 *
 * No es cosmética: `model.py` computa el pooling del Compressor en fp32 y luego
 * hace `self.norm(kv.to(dtype))`, o sea castea a bf16 ANTES de normalizar. Ese
 * cast es parte de la semántica del modelo, y saltárselo mete un error relativo
 * de ~1,6e-3 que se arrastra por toda la pila de capas.
 *
 * Regla general del port: donde la referencia cambia de precisión, el motor
 * tiene que cambiarla en el mismo sitio.
 * ------------------------------------------------------------------------- */
static inline float dsv4_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    if (((u >> 23) & 0xFF) == 0xFF) return x;          /* NaN/Inf intactos */
    const uint32_t r = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    float y;
    memcpy(&y, &r, sizeof y);
    return y;
}

/* ---------------------------------------------------------------------------
 * Router: función de score.
 *
 * GLM-5.2 usa sigmoid; DeepSeek-V4-Flash usa sqrtsoftplus. Es literalmente la
 * única diferencia del router entre las dos familias — el top-k con bias
 * (noaux_tc), la renormalización y routed_scaling_factor son idénticos.
 *
 * `ln(1+exp(x))` desborda a +inf para x > ~88 en f32, y de ahí salen NaN que
 * envenenan el top-k en silencio: ningún experto gana la comparación y el
 * router acaba devolviendo -1. Para x grande softplus(x) -> x, así que se corta.
 * ------------------------------------------------------------------------- */
static inline float dsv4_sqrtsoftplus(float x) {
    return sqrtf(x > 20.0f ? x : log1pf(expf(x)));
}

/* ---------------------------------------------------------------------------
 * RMSNorm — igual que en el resto de colibrì, aquí por completitud.
 * ------------------------------------------------------------------------- */
static inline void dsv4_rmsnorm(float *out, const float *x, const float *w,
                                int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float sc = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * sc * w[i];
}

/* ---------------------------------------------------------------------------
 * SwiGLU con clamp (`swiglu_limit: 10.0`).
 * `kimi_k3.c` ya tiene una variante acotada (`situf_`), pero con otra fórmula:
 * esta es la de DeepSeek, un recorte duro antes de la puerta.
 * ------------------------------------------------------------------------- */
static inline float dsv4_swiglu(float gate, float up, float limit) {
    if (limit > 0.0f) {
        /* ASIMÉTRICO, y es a propósito (`model.py::Expert.forward`):
         *   up   = clamp(up, min=-limit, max=limit)   -> por los dos lados
         *   gate = clamp(gate, max=limit)             -> SÓLO por arriba
         * Recortar también `gate` por abajo parece lo natural y es incorrecto:
         * SiLU ya satura hacia -inf, así que el límite inferior no hace falta y
         * ponerlo cambia la función. */
        if (gate > limit) gate = limit;
        if (up   >  limit) up   =  limit;
        if (up   < -limit) up   = -limit;
    }
    return (gate / (1.0f + expf(-gate))) * up;   /* silu(gate) * up */
}

/* ---------------------------------------------------------------------------
 * mHC — hc_split_sinkhorn.
 *
 * Descompone mixes[(2+hc)*hc] en (pre, post, comb):
 *
 *   [0    : hc  ] -> pre  = sigmoid(m*scale[0] + base) + eps
 *   [hc   : 2*hc] -> post = 2*sigmoid(m*scale[1] + base)
 *   [2*hc :     ] -> comb = (m*scale[2] + base) como matriz [hc,hc]
 *
 * y comb pasa por: softmax por filas, +eps, normalizar columnas, y luego
 * (iters-1) rondas de (normalizar filas, normalizar columnas). El resultado
 * tiende a doblemente estocástica.
 *
 * OJO con el conteo de iteraciones: la primera ronda es la del softmax, NO una
 * iteración completa. Poner `iters` rondas en vez de `iters-1` es un error que
 * no rompe nada visiblemente —comb sigue siendo casi doblemente estocástica—
 * pero desvía los números lo justo para que la fase 4 no cuadre.
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

    /* softmax por filas (estable: se resta el máximo), luego +eps */
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

    /* normalizar columnas */
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
 * mHC — colapsar las `hc` copias del residual en la entrada de la sub-capa.
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
 * mHC — reexpandir a `hc` copias tras la sub-capa.
 *   h[m][d] = post[m]*y[d] + sum_n comb[m][n]*hres[n][d]
 *
 * `hres` es el residual de ENTRADA a la sub-capa; `h` puede ser el mismo
 * buffer sólo si no se solapan, así que se exige que sean distintos.
 * ------------------------------------------------------------------------- */
static inline void dsv4_hc_expand(float *h, const float *y, const float *post,
                                  const float *comb, const float *hres,
                                  int hc, int dim) {
    for (int j = 0; j < hc; j++) {
        float *dst = h + (size_t)j * dim;
        const float pj = post[j];
        for (int d = 0; d < dim; d++) dst[d] = pj * y[d];
        /* comb va TRANSPUESTA: la salida j acumula sobre la primera dimensión.
         *
         * `model.py::hc_post` hace
         *     sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
         * que desarrollado es  y[j] = sum_i comb[i][j] * residual[i]  — no
         * `sum_j comb[i][j]`, que es lo natural de escribir y lo que yo tenía.
         * Como `comb` sale doblemente estocástica del Sinkhorn, filas y
         * columnas suman 1 y el error es casi invisible en las métricas
         * agregadas: sólo se ve encadenando el bloque entero. */
        for (int i = 0; i < hc; i++) {
            const float c = comb[(size_t)i * hc + j];
            const float *src = hres + (size_t)i * dim;
            for (int d = 0; d < dim; d++) dst[d] += c * src[d];
        }
    }
}

/* ---------------------------------------------------------------------------
 * CSA — pooling con puerta aprendida del `Compressor` (prefill).
 *
 * Es la operación central de Compressed Sparse Attention: funde `ratio` tokens
 * consecutivos en uno solo, ponderándolos con una puerta aprendida:
 *
 *     kv = (kv * (score + ape).softmax(dim=slot)).sum(dim=slot)
 *
 * Tres cosas que hay que entender o los números no salen:
 *
 *   1. **El softmax es POR CANAL, no escalar.** `score` tiene la misma forma
 *      que `kv`, así que cada canal `c` decide su propia mezcla sobre los
 *      slots. No es "pesar tokens", es "pesar tokens por cada dimensión".
 *
 *   2. **`ape` es un embedding de posición DENTRO del bloque** ([ratio, coff*d]),
 *      y se suma al score antes del softmax. Por eso el compressor distingue el
 *      primer token de un grupo del último.
 *
 *   3. **Modo solapado** (`overlap`, que es cuando compress_ratio == 4): las
 *      proyecciones sacan 2*d canales y cada grupo atiende a 2*ratio slots —
 *      la mitad alta de los canales del propio grupo, y la mitad baja del grupo
 *      ANTERIOR. Así los bordes entre bloques no cortan en seco. El primer
 *      grupo no tiene anterior: sus slots bajos van a 0 con score -inf, que
 *      tras el softmax pesan exactamente 0.
 *
 * Sólo cubre el camino de prefill (start_pos == 0) sin resto (seqlen múltiplo
 * de ratio). El camino de decode incremental usa `kv_state`/`score_state` y va
 * aparte.
 *
 *   kv, score : [b, s, coff*d]   con coff = overlap ? 2 : 1
 *   ape       : [ratio, coff*d]
 *   out       : [b, s/ratio, d]
 * ------------------------------------------------------------------------- */
static inline void dsv4_compress_prefill(
        const float *kv, const float *score, const float *ape,
        int b, int s, int ratio, int d, int overlap,
        float *out)
{
    const int coff  = overlap ? 2 : 1;
    const int chan  = coff * d;              /* canales de kv/score/ape */
    const int nslot = overlap ? 2 * ratio : ratio;
    const int ngrp  = s / ratio;

    float *w = (float *)malloc((size_t)nslot * sizeof(float));
    float *v = (float *)malloc((size_t)nslot * sizeof(float));
    if (!w || !v) { fprintf(stderr, "OOM compress\n"); exit(1); }

    for (int bi = 0; bi < b; bi++) {
        for (int g = 0; g < ngrp; g++) {
            float *dst = out + ((size_t)bi * ngrp + g) * d;

            for (int c = 0; c < d; c++) {
                /* reunir (peso, valor) de cada slot para este canal */
                for (int j = 0; j < nslot; j++) {
                    int t, ch;
                    if (!overlap) {
                        t = g * ratio + j;
                        ch = c;
                    } else if (j >= ratio) {
                        t = g * ratio + (j - ratio);
                        ch = d + c;              /* mitad alta: grupo propio */
                    } else if (g == 0) {
                        w[j] = -INFINITY;        /* no hay grupo anterior */
                        v[j] = 0.0f;
                        continue;
                    } else {
                        t = (g - 1) * ratio + j;
                        ch = c;                  /* mitad baja: grupo anterior */
                    }
                    const size_t idx = ((size_t)bi * s + t) * chan + ch;
                    /* ape se indexa por la posición DENTRO del bloque */
                    w[j] = score[idx] + ape[(size_t)(t % ratio) * chan + ch];
                    v[j] = kv[idx];
                }

                /* softmax estable sobre los slots + suma ponderada */
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
 * Indexer: puntuar la KV comprimida y quedarse con las `keep` mejores.
 *
 * BUENA NOTICIA PARA EL PORT: esto es, función a función, el mismo scoring que
 * el indexer DSA que colibrì ya tiene para GLM-5.2 (`colibri.c:3374-3382`):
 *
 *     d0 = dot(q_h, k_t) * rs;
 *     if (d0 > 0) a += w32[h] * d0;      // ReLU y LUEGO el peso por cabeza
 *     isc[t] = a * wsc;                  // wsc = 1/sqrt(n_heads)
 *
 * y DeepSeek-V4-Flash hace exactamente lo mismo:
 *
 *     index_score = (einsum("bshd,btd->bsht", q, kv).relu_()
 *                    * weights.unsqueeze(-1)).sum(dim=2)
 *     weights = weights_proj(x) * (softmax_scale * n_heads**-0.5)
 *
 * (Como rs > 0, relu(d0*rs) == rs*relu(d0): da igual dónde se aplique la
 * escala.) Kimi K3 no tiene nada de esto — usa KDA, una recurrencia delta-rule
 * sin selección.
 *
 * Las diferencias reales frente a GLM son tres, y ninguna toca esta función:
 *   1. V4 puntúa contra la KV **comprimida** (la que llena el Compressor propio
 *      del indexer); GLM contra la KV latente completa.
 *   2. El `q` de V4 pasa por rotación Hadamard + simulación FP4.
 *   3. GLM normaliza las claves con un `k_norm` (LayerNorm); en V4 esa función
 *      la hace el `norm` del compressor del indexer.
 *
 * PRECISIÓN: el scoring corre en **bfloat16**, no en f32, y aquí eso no es un
 * detalle cosmético. Calcularlo en f32 da un orden distinto —y con `index_topk`
 * menor que el número de bloques comprimidos, un CONJUNTO distinto— o sea que
 * la atención acabaría mirando a otras posiciones. Medido: con f32 fallaban
 * 2/512 índices del modelo tiny; con bf16, 0.
 *
 * Se reproduce acumulando en f32 y redondeando a bf16 en cada frontera de
 * tensor (salida del einsum, el peso escalado, la suma sobre cabezas), que es
 * lo que hace torch con tensores bf16. Verificado contra la referencia nativa.
 *
 * Aquí la selección es O(keep · T) por claridad. En producción se usa
 * `partial_select_desc` de colibrì (quickselect O(T) medio) con el mismo
 * criterio de desempate.
 *
 *   q       : [b, s, h, d]   ya rotado y cuantizado
 *   kv      : [b, Tcap, d]   KV comprimida; sólo valen las `nvalid` primeras
 *   weights : [b, s, h]      salida CRUDA de weights_proj (la escala va aquí)
 *   out     : [b, s, keep]   índices, o -1 para relleno no causal
 * ------------------------------------------------------------------------- */
static inline void dsv4_indexer_topk_ex(
        const float *q, const float *kv, const float *weights,
        int b, int s, int h, int d, int Tcap, int nvalid,
        int ratio, int keep, int offset, float softmax_scale,
        int no_causal,   /* 1 = sin máscara: todo lo comprimido es pasado */
        int *out)
{
    const float wsc = softmax_scale / sqrtf((float)h);
    float *sc = (float *)malloc((size_t)nvalid * sizeof(float));
    int *taken = (int *)malloc((size_t)nvalid * sizeof(int));
    if (!sc || !taken) { fprintf(stderr, "OOM indexer\n"); exit(1); }

    for (int bi = 0; bi < b; bi++) {
        for (int si = 0; si < s; si++) {
            /* Límite causal: una posición sólo puede mirar bloques comprimidos
             * ya cerrados. Para las primeras `ratio-1` posiciones esto es 0, o
             * sea que NADA es visible y la fila entera sale a -1.
             *
             * En DECODE no aplica: todo lo comprimido es pasado por
             * construcción. Intentar desactivarlo pasando ratio=1 no funciona
             * —da limit=1 y recorta a un solo bloque—, hace falta el flag. */
            const int limit = no_causal ? nvalid : (si + 1) / ratio;

            for (int t = 0; t < nvalid; t++) {
                if (t >= limit) { sc[t] = -INFINITY; taken[t] = 0; continue; }
                float a = 0.0f;
                for (int hi = 0; hi < h; hi++) {
                    const float *qh = q + (((size_t)bi * s + si) * h + hi) * d;
                    const float *kt = kv + ((size_t)bi * Tcap + t) * d;
                    float dot = 0.0f;
                    for (int i = 0; i < d; i++) dot += qh[i] * kt[i];
                    dot = dsv4_to_bf16(dot);              /* salida del einsum */
                    if (dot > 0.0f) {                     /* ReLU */
                        const float wh = dsv4_to_bf16(
                                weights[((size_t)bi * s + si) * h + hi] * wsc);
                        /* el PRODUCTO se redondea antes de acumular: la suma
                         * sobre cabezas va en f32 y se redondea al final */
                        a += dsv4_to_bf16(wh * dot);
                    }
                }
                sc[t] = dsv4_to_bf16(a);
                taken[t] = 0;
            }

            /* selección: mayor score primero, empates por índice menor */
            int *dst = out + ((size_t)bi * s + si) * keep;
            for (int k = 0; k < keep; k++) {
                int best = -1;
                for (int t = 0; t < nvalid; t++) {
                    if (taken[t]) continue;
                    if (best < 0 || sc[t] > sc[best]) best = t;
                }
                if (best < 0) { dst[k] = -1; continue; }
                taken[best] = 1;
                /* lo que cae fuera del límite causal es relleno, no una
                 * posición: se marca -1 y `sparse_attn` lo ignora. */
                dst[k] = (best >= limit) ? -1 : best + offset;
            }
        }
    }

    free(sc);
    free(taken);
}

/* Envoltorio con máscara causal: el camino de prefill, ya validado. */
static inline void dsv4_indexer_topk(
        const float *q, const float *kv, const float *weights,
        int b, int s, int h, int d, int Tcap, int nvalid,
        int ratio, int keep, int offset, float softmax_scale, int *out)
{
    dsv4_indexer_topk_ex(q, kv, weights, b, s, h, d, Tcap, nvalid, ratio, keep,
                         offset, softmax_scale, 0, out);
}

/* ---------------------------------------------------------------------------
 * sparse_attn — atención sobre las posiciones que eligió el indexer.
 *
 * Port de `ref/kernel.py::sparse_attn_kernel`. Tres detalles que hay que
 * respetar o los números no cuadran:
 *
 *   1. **`kv` hace de K y de V a la vez** ([b,n,d], una sola cabeza: el config
 *      dice `num_key_value_heads: 1`). Es atención MLA absorbida — no hay
 *      proyección V separada.
 *   2. **Los índices -1 son relleno** y se enmascaran a -inf, no a cero.
 *   3. **`attn_sink` va SÓLO en el denominador**: `sum_exp += exp(sink - max)`.
 *      Es un sumidero por cabeza —un "token nulo" aprendido— que deja a una
 *      cabeza no atender a nada. Sumarlo también al numerador es el error fácil
 *      y silencioso: la salida seguiría pareciendo razonable.
 *
 *   q    : [b, m, h, d]
 *   kv   : [b, n, d]
 *   sink : [h]
 *   idxs : [b, m, topk]   con -1 como relleno
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
                /* Si no hay ninguna posición válida, el máximo se queda a 0
                 * (como hace el kernel) y sólo pesa el sumidero. */
                if (mx == -INFINITY) mx = 0.0f;

                float denom = 0.0f;
                for (int k = 0; k < topk; k++) {
                    e[k] = (e[k] == -INFINITY) ? 0.0f : expf(e[k] - mx);
                    denom += e[k];
                }
                denom += expf(sink[hi] - mx);     /* sólo en el denominador */

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
 * RoPE — rotación posicional sobre los últimos `rd` canales.
 *
 * `model.py` la aplica IN-PLACE y en formato **entrelazado**: los canales se
 * leen por pares consecutivos (x0,x1) como un complejo, no partiendo el vector
 * por la mitad. Es el mismo convenio que `rope_interleave` en `colibri.c`.
 *
 *   (x0, x1) * (c, s) = (x0*c - x1*s,  x0*s + x1*c)
 *
 * `inverse` usa el conjugado (des-rotar). Hace falta de verdad: la salida de la
 * atención pasa por una RoPE inversa antes de la proyección de salida
 * (`model.py:539`), cosa fácil de pasar por alto.
 *
 *   x     : [..., rows, (heads,) rd]  se rota in-place
 *   freqs : [rows, rd/2, 2]           (real, imag) por posición y par
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
                    /* `apply_rotary_emb` calcula en complejo f32 pero hace
                     * `y.copy_(x)` sobre el tensor ORIGINAL, que es bf16: el
                     * resultado queda redondeado. Sin esto la RoPE se queda a
                     * 1,6e-3 en vez de ser exacta, y el error se amplifica al
                     * cuantizar después a FP4 (los valores cruzan de punto de
                     * la rejilla e2m1). */
                    v[2 * p]     = dsv4_to_bf16(a * c - b * s);
                    v[2 * p + 1] = dsv4_to_bf16(a * s + b * c);
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * act_quant FP8 (e4m3) por bloques, fusionado quant+dequant.
 *
 * `model.py` lo aplica in-place sobre los canales NO-rope de la KV para simular
 * el QAT: los valores vuelven a bf16 pero ya con la pérdida de FP8. Los canales
 * rope se dejan intactos, que ahí la precisión posicional importa.
 *
 * Con `pow2` (scale_fmt="ue8m0") la escala se redondea a potencia de dos hacia
 * arriba — es lo que hace el formato MXFP y lo que trae el checkpoint.
 * ------------------------------------------------------------------------- */
#define DSV4_FP8_MAX 448.0f

static inline float dsv4_quant_e4m3(float v) {
    /* e4m3: 4 bits de exponente, 3 de mantisa, sin infinitos.
     * Se redondea al representable más cercano vía escalado por potencia de 2. */
    if (v == 0.0f || !isfinite(v)) return v;
    const float a = fabsf(v);
    if (a > DSV4_FP8_MAX) return v > 0 ? DSV4_FP8_MAX : -DSV4_FP8_MAX;
    int ex;
    frexpf(a, &ex);                       /* a = m * 2^ex, m en [0.5,1) */
    if (ex < -5) ex = -5;                 /* rango subnormal de e4m3 */
    const float step = ldexpf(1.0f, ex - 4);   /* 3 bits de mantisa + implícito */
    /* nearbyintf, NO roundf: el hardware FP8 redondea al par más cercano
     * (round-to-nearest-even) y `roundf` redondea las medias hacia afuera. Sólo
     * se nota en los empates exactos, pero ahí falla: un valor que cae en 8,5
     * pasos sale 2,25 en vez de 2,0. Medido contra la referencia. */
    float q = nearbyintf(v / step) * step;
    if (q > DSV4_FP8_MAX) q = DSV4_FP8_MAX;
    if (q < -DSV4_FP8_MAX) q = -DSV4_FP8_MAX;
    return q;
}

/* FP4 e2m1: sólo 8 magnitudes representables. La rejilla se recorre entera
 * porque son 8 comparaciones; no compensa nada más listo. */
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

/* Cuantización por bloques fusionada quant+dequant, in-place.
 *
 * FP8 y FP4 comparten TODO menos la rejilla de redondeo y el máximo del
 * formato: mismo amax por bloque, misma escala redondeada a potencia de dos
 * (ue8m0), mismo reescalado. Tenerlas separadas invitaba a que una recibiera
 * una corrección y la otra no. */
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
 * Transformada de Walsh-Hadamard sobre la última dimensión, escalada 1/sqrt(n).
 *
 * `model.py::rotate_activation` la usa para repartir la información entre
 * dimensiones antes de cuantizar a FP4/FP8. Es ORTOGONAL, así que rotar q y k a
 * la vez no cambia sus productos escalares: existe sólo para que la
 * cuantización se porte mejor, no para alterar la atención.
 *
 * Mariposa iterativa in-place. `n` debe ser potencia de dos (lo es: 128, 256).
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
 * Compressor completo (prefill): pooling con puerta -> bf16 -> RMSNorm ->
 * RoPE -> simulación de cuantización.
 *
 * El detalle que se escapa leyendo por encima: la RoPE de la KV comprimida usa
 * las frecuencias **submuestreadas cada `ratio`** (`freqs_cis[:cutoff:ratio]`),
 * porque cada entrada comprimida representa un bloque de `ratio` tokens y su
 * posición es la del primer token del bloque. Usar `freqs[g]` en vez de
 * `freqs[g*ratio]` da un resultado plausible pero equivocado.
 *
 * `rotate` distingue los dos compressors: el del Indexer rota con Hadamard y
 * cuantiza a FP4 el vector entero; el principal cuantiza a FP8 sólo los canales
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

    /* `model.py` hace `self.norm(kv.to(dtype))`: bf16 ANTES de normalizar */
    for (int64_t i = 0; i < rows * d; i++) out[i] = dsv4_to_bf16(out[i]);

    float *tmp = (float *)malloc((size_t)d * sizeof(float));
    for (int64_t r = 0; r < rows; r++) {
        dsv4_rmsnorm(tmp, out + r * d, norm_w, d, eps);
        for (int i = 0; i < d; i++) out[r * d + i] = dsv4_to_bf16(tmp[i]);
    }
    free(tmp);

    /* RoPE con frecuencias submuestreadas: el grupo g está en la posición
     * g*ratio de la secuencia original. */
    for (int64_t r = 0; r < rows; r++) {
        const int g = (int)(r % ngrp);
        dsv4_rope(out + r * d + (d - rd),
                  freqs + (size_t)(g * ratio) * (rd / 2) * 2,
                  1, 1, 1, rd, 0);
    }

    if (rotate) {
        /* Compressor del Indexer: Hadamard y FP4 sobre el vector entero */
        dsv4_hadamard(out, rows, d);
        dsv4_blockwise_quant(out, rows, d, 32, 1, DSV4_FP4);
    } else {
        /* Compressor principal: FP8 sólo en los canales no-rope. Hay que
         * compactarlos porque no son contiguos entre filas. */
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
 * freqs_cis con escalado YaRN.
 *
 * Port de `model.py::precompute_freqs_cis`. YaRN extiende el contexto sin
 * reentrenar: en vez de dividir TODAS las frecuencias por `factor` —lo que
 * estropearía las de periodo corto, que codifican posición local— interpola
 * sólo las de periodo largo, con una rampa lineal entre dos "rangos de
 * corrección" derivados de beta_fast y beta_slow.
 *
 * DeepSeek-V4-Flash: base 10000, factor 16, original 65536 -> 1M de contexto.
 * Y hay una SEGUNDA tabla con base 160000 (`compress_rope_theta`) para la KV
 * comprimida, porque sus posiciones avanzan de `ratio` en `ratio` y necesitan
 * otra escala de frecuencias.
 *
 *   out : [seqlen, dim/2, 2]  con (cos, sin) por posición y par
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
            /* las de periodo corto (smooth~1) se quedan; las largas se dividen */
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
