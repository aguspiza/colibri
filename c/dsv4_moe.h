/* dsv4_moe.h — el bloque MoE: router + expertos rutados + experto compartido.
 *
 * Es la pieza que conecta con lo que hace especial a colibrì. Los expertos
 * rutados son el 97,5 % de los parámetros del modelo (277B de 284B) y aquí se
 * aplican desde RAM; en el motor real cada uno llega por streaming desde disco
 * justo cuando el router demuestra que hace falta. El resto del bloque —router,
 * pesos, experto compartido— es idéntico en los dos casos.
 *
 * Sigue `ref/model.py::Gate.forward` y `MoE.forward`.
 */

#ifndef DSV4_MOE_H
#define DSV4_MOE_H

#include "dsv4_math.h"
#include "dsv4_attn.h"     /* dsv4_matmul_ex */
#include "dsv4_stream.h"   /* tier de expertos por streaming (fase 3) */

typedef struct {
    int dim, n_experts, topk, inter;
    float route_scale, swiglu_limit;
    int hash;                  /* capas iniciales: ruta por tabla, no por score */
} DsV4MoeCfg;

typedef struct {
    DsV4W gate_w;              /* [n_experts, dim] */
    const float *gate_bias;    /* [n_experts] — NULL en las capas hash */
    const int32_t *tid2eid;    /* [vocab, topk] — sólo en las capas hash */
    /* Expertos residentes en RAM: un descriptor por experto. */
    const DsV4W *e_w1, *e_w2, *e_w3;
    DsV4W s_w1, s_w2, s_w3;             /* experto compartido */

    /* Ruta de STREAMING (fase 3). Si `store` no es NULL, los expertos rutados
     * NO se leen de `e_w*` sino del disco, a través de la caché por capa. El
     * experto compartido se queda residente: se usa en todos los tokens, así
     * que streamearlo sería absurdo — es exactamente la distinción que hace
     * colibrì entre el conjunto denso y el tier rutado. */
    DsV4Store *store;
    int layer;                          /* qué capa pedirle al store */
} DsV4MoeW;

/* ---------------------------------------------------------------------------
 * Router. Tres detalles que hay que respetar:
 *
 *   1. **El scoring va en fp32**, sin redondear (`linear(x.float(), w.float())`).
 *   2. **El bias sólo decide, no pondera.** Se elige el top-k con `score+bias`
 *      pero se pesa con el score CRUDO. El bias es de balanceo de carga
 *      (noaux_tc), no una opinión sobre el experto. Es idéntico al router de
 *      GLM-5.2 en colibrì; lo único que cambia es sqrtsoftplus en vez de sigmoid.
 *   3. **En las capas hash no hay bias** y los índices salen de una tabla
 *      indexada por el id del token — pero los PESOS se siguen calculando con
 *      el router. O sea que el matmul del router se ejecuta igual.
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
            /* top-k por selección directa: con topk<=8 y E=256 gana a ordenar */
            for (int k = 0; k < c->topk; k++) {
                int best = -1;
                float bv = -INFINITY;
                for (int e = 0; e < E; e++) {
                    int taken = 0;
                    for (int j = 0; j < k; j++) if (idx[j] == e) { taken = 1; break; }
                    if (taken) continue;
                    const float v = sc[e] + w->gate_bias[e];   /* el bias SÓLO decide */
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
 * Un experto: SwiGLU con clamp, computado en fp32.
 *
 * El peso de ruteo multiplica el ESTADO INTERMEDIO, antes de w2, no la salida
 * (`x = weights * x; return self.w2(x)`). Matemáticamente da igual porque w2 es
 * lineal, pero en bf16 no: cambia dónde cae el redondeo.
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
 * Bloque MoE completo.
 *
 * `y` acumula en fp32: los expertos rutados primero, el compartido después.
 * El compartido se aplica SIEMPRE y con peso 1 — es el que garantiza que todo
 * token reciba algo aunque el ruteo se desequilibre.
 * ------------------------------------------------------------------------- */
static inline void dsv4_moe_forward(const DsV4MoeCfg *c, const DsV4MoeW *w,
                                    const float *x, const int32_t *ids,
                                    int64_t rows, float *out)
{
    int *idx = (int *)malloc((size_t)rows * c->topk * sizeof(int));
    float *wt = (float *)malloc((size_t)rows * c->topk * sizeof(float));
    dsv4_moe_route(c, w, x, ids, rows, idx, wt);

    /* AVISO sobre las capas hash: aquí se acumulan las `topk` contribuciones,
     * incluso si dos apuntan al mismo experto. La referencia NO hace eso.
     *
     * `MoE.forward` escribe `y[idx] += expert(...)` con `idx` sacado de
     * `torch.where(indices == i)`. Con índices repetidos esa expresión no
     * acumula: lee, suma y reescribe, así que gana la última escritura y una de
     * las dos contribuciones se pierde en silencio.
     *
     * Con el top-k por score no puede pasar (los índices salen distintos por
     * construcción), y una tabla `tid2eid` entrenada tampoco debería repetir.
     * Pero si un checkpoint real trae duplicados, motor y referencia divergen
     * — y el motor tendría razón. Conviene comprobar la tabla al cargar.
     *
     * Se detectó porque el oráculo generaba `tid2eid` muestreando CON
     * reemplazo: 11 de 64 tokens recibían el mismo experto dos veces y el
     * bloque salía con un error de 2,2e-1 mientras el ruteo era exacto. */
    memset(out, 0, (size_t)rows * c->dim * sizeof(float));
    for (int64_t r = 0; r < rows; r++) {
        for (int k = 0; k < c->topk; k++) {
            const int e = idx[r * c->topk + k];
            DsV4W w1, w2, w3;
            if (w->store) {
                /* Del disco, con caché LRU. El resto del cálculo es idéntico:
                 * de dónde vinieron los bytes no puede cambiar el resultado. */
                const int64_t per = (int64_t)c->inter * c->dim;
                const float *blk = dsv4_store_get(w->store, w->layer, e);
                w1 = dsv4_w_f32(blk,           c->inter, c->dim);
                w2 = dsv4_w_f32(blk + per,     c->dim,   c->inter);
                w3 = dsv4_w_f32(blk + 2 * per, c->inter, c->dim);
            } else {
                w1 = w->e_w1[e]; w2 = w->e_w2[e]; w3 = w->e_w3[e];
            }
            dsv4_expert_apply(c, &w1, &w2, &w3, x + r * c->dim,
                              wt[r * c->topk + k], out + r * c->dim);
        }
        dsv4_expert_apply(c, &w->s_w1, &w->s_w2, &w->s_w3,
                          x + r * c->dim, 1.0f, out + r * c->dim);
    }
    free(idx); free(wt);
}

#endif /* DSV4_MOE_H */
