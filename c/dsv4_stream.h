/* dsv4_stream.h — where the experts stop living in RAM.
 *
 * This is the whole point of the port. The routed experts are 97.5 % of the model
 * (277B of 284B, 137 GiB on disk), and colibri does not load them: it PLACES
 * them. Each token activates 6 of 256 per layer, so only ~3.4 GB per token is
 * ever needed, and that is read off the disk exactly when the router proves it is
 * needed.
 *
 * What follows is the minimal version of that machinery — per-layer LRU cache,
 * positioned read, promotion — exercised on the tiny model. colibri's real
 * infrastructure (thread pool, PILOT, batch-union, O_DIRECT, multi-drive) is
 * model-agnostic and hooks in the same way; what had to be shown here is that a
 * DeepSeek-V4 engine can be fed from disk WITHOUT CHANGING A SINGLE BIT of the
 * result.
 *
 * That is the invariant colibri does not negotiate, and the acceptance criterion
 * for this stage:
 *
 *     placement decides SPEED, never SEMANTICS
 *
 * If a token changes depending on whether the expert came from RAM or from disk,
 * the port is wrong no matter how fast it is.
 */

#ifndef DSV4_STREAM_H
#define DSV4_STREAM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    int eid;                 /* -1 = empty */
    float *buf;              /* w1 | w2 | w3, contiguous */
    uint64_t used;           /* the LRU's logical clock */
} DsV4Slot;

typedef struct {
    FILE *f;
    int n_layers, n_experts, cap;
    int64_t floats;          /* floats per expert (w1+w2+w3) */
    DsV4Slot *slots;         /* [n_layers * cap] */
    int *nslot;              /* [n_layers] how full each cache is */
    uint64_t clock, hits, miss, bytes;
} DsV4Store;

/* Writes the experts to disk in colibri's layout: an expert's three matrices
 * CONTIGUOUS, so that one read brings the whole expert in.
 *
 * The real DeepSeek checkpoint is not laid out that way — safetensors groups all
 * the `.weight` tensors in one region and all the `.scale` tensors in another, so
 * two reads per expert — but the principle is the same. */
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

/* Returns the expert (w1|w2|w3 contiguous), reading it in if it is not cached.
 *
 * Linear search, as in colibri: a layer's cache is a few dozen slots, and against
 * the megabytes a miss costs, searching is noise. */
static inline const float *dsv4_store_get(DsV4Store *s, int layer, int eid) {
    DsV4Slot *lc = s->slots + (size_t)layer * s->cap;
    for (int i = 0; i < s->nslot[layer]; i++)
        if (lc[i].eid == eid) {
            s->hits++;
            lc[i].used = ++s->clock;
            return lc[i].buf;
        }

    s->miss++;
    /* pick a slot: an empty one first, otherwise the LRU victim */
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
    /* Positioned read. fseek+fread is enough here because the harness is
     * single-threaded; colibri's pool uses `pread` precisely because a
     * descriptor's offset is shared state and several threads would clobber it. */
    if (fseek(s->f, (long)off, SEEK_SET) != 0 ||
        fread(lc[dst].buf, sizeof(float), (size_t)s->floats, s->f)
            != (size_t)s->floats) {
        fprintf(stderr, "[stream] failed reading expert L%d E%d\n", layer, eid);
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
