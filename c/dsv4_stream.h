/* dsv4_stream.h — fase 3: los expertos dejan de estar en RAM.
 *
 * Es la razón de ser del port. Los expertos rutados son el 97,5 % del modelo
 * (277B de 284B, 137 GiB en disco), y colibrì no los carga: los **coloca**.
 * Cada token activa 6 de 256 por capa, así que sólo hacen falta ~3,4 GB por
 * token, y esos se leen del disco justo cuando el router demuestra que hacen
 * falta.
 *
 * Aquí está la versión mínima de esa maquinaria —caché LRU por capa, lectura
 * posicionada, promoción— sobre el modelo tiny. La infraestructura real de
 * colibrì (pool de hilos, PILOT, batch-union, O_DIRECT, dual-SSD) es agnóstica
 * al modelo y se engancha igual; lo que había que demostrar es que el motor de
 * DeepSeek-V4 puede alimentarse desde disco **sin cambiar un solo bit del
 * resultado**.
 *
 * Ese es el invariante que colibrì no negocia y el criterio de esta fase:
 *
 *     la colocación decide la VELOCIDAD, nunca la SEMÁNTICA
 *
 * Si un token cambia según el experto viniera de RAM o de disco, el port está
 * mal por rápido que sea.
 */

#ifndef DSV4_STREAM_H
#define DSV4_STREAM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    int eid;                 /* -1 = vacío */
    float *buf;              /* w1 | w2 | w3 contiguos */
    uint64_t used;           /* reloj lógico del LRU */
} DsV4Slot;

typedef struct {
    FILE *f;
    int n_layers, n_experts, cap;
    int64_t floats;          /* floats por experto (w1+w2+w3) */
    DsV4Slot *slots;         /* [n_layers * cap] */
    int *nslot;              /* [n_layers] ocupación de cada caché */
    uint64_t clock, hits, miss, bytes;
} DsV4Store;

/* Escribe los expertos en disco con el layout de colibrì: las tres matrices de
 * cada experto CONTIGUAS, para que una sola lectura traiga el experto entero.
 *
 * En el checkpoint real de DeepSeek no están así —safetensors agrupa todos los
 * `.weight` en una zona y todas las `.scale` en otra, o sea 2 lecturas por
 * experto (§7.3 del análisis)—, pero el principio es el mismo. */
static inline int dsv4_store_write(const char *path, int n_layers, int n_experts,
                                   int64_t floats_per_mat,
                                   const float ***w1, const float ***w2,
                                   const float ***w3)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    for (int L = 0; L < n_layers; L++)
        for (int e = 0; e < n_experts; e++) {
            fwrite(w1[L][e], sizeof(float), (size_t)floats_per_mat, f);
            fwrite(w2[L][e], sizeof(float), (size_t)floats_per_mat, f);
            fwrite(w3[L][e], sizeof(float), (size_t)floats_per_mat, f);
        }
    fclose(f);
    return 0;
}

static inline int dsv4_store_open(DsV4Store *s, const char *path, int n_layers,
                                  int n_experts, int64_t floats_per_expert,
                                  int cache_cap)
{
    memset(s, 0, sizeof *s);
    s->f = fopen(path, "rb");
    if (!s->f) return -1;
    s->n_layers = n_layers;
    s->n_experts = n_experts;
    s->floats = floats_per_expert;
    s->cap = cache_cap;
    s->slots = calloc((size_t)n_layers * cache_cap, sizeof(DsV4Slot));
    s->nslot = calloc((size_t)n_layers, sizeof(int));
    for (int i = 0; i < n_layers * cache_cap; i++) s->slots[i].eid = -1;
    return 0;
}

static inline void dsv4_store_close(DsV4Store *s) {
    for (int i = 0; i < s->n_layers * s->cap; i++) free(s->slots[i].buf);
    free(s->slots); free(s->nslot);
    if (s->f) fclose(s->f);
}

/* Devuelve el experto (w1|w2|w3 contiguos), leyéndolo si no está en caché.
 *
 * Búsqueda lineal como en colibrì: la caché de una capa son unas decenas de
 * slots, y frente a los megabytes que cuesta un fallo, buscar es ruido. */
static inline const float *dsv4_store_get(DsV4Store *s, int layer, int eid) {
    DsV4Slot *lc = s->slots + (size_t)layer * s->cap;
    for (int i = 0; i < s->nslot[layer]; i++)
        if (lc[i].eid == eid) {
            s->hits++;
            lc[i].used = ++s->clock;
            return lc[i].buf;
        }

    s->miss++;
    /* elegir hueco: primero vacío, si no el LRU */
    int dst;
    if (s->nslot[layer] < s->cap) {
        dst = s->nslot[layer]++;
    } else {
        dst = 0;
        for (int i = 1; i < s->cap; i++)
            if (lc[i].used < lc[dst].used) dst = i;
    }
    if (!lc[dst].buf) lc[dst].buf = malloc((size_t)s->floats * sizeof(float));

    const int64_t idx = (int64_t)layer * s->n_experts + eid;
    const int64_t off = idx * s->floats * (int64_t)sizeof(float);
    /* Lectura posicionada. Aquí basta fseek+fread porque el arnés es de un solo
     * hilo; el pool de colibrì usa `pread` precisamente porque el offset de un
     * descriptor es estado compartido y varios hilos se pisarían. */
    if (fseek(s->f, (long)off, SEEK_SET) != 0 ||
        fread(lc[dst].buf, sizeof(float), (size_t)s->floats, s->f)
            != (size_t)s->floats) {
        fprintf(stderr, "[stream] fallo leyendo experto L%d E%d\n", layer, eid);
        exit(1);
    }
    s->bytes += (uint64_t)s->floats * sizeof(float);
    lc[dst].eid = eid;
    lc[dst].used = ++s->clock;
    return lc[dst].buf;
}

static inline double dsv4_store_hitrate(const DsV4Store *s) {
    const uint64_t t = s->hits + s->miss;
    return t ? (double)s->hits / (double)t : 0.0;
}

#endif /* DSV4_STREAM_H */
