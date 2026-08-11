/* dsv4_port.c — the independent port's engine: loads the 284B checkpoint and
 * generates text. See docs/deepseek-v4-port.md; c/deepseek_v4.c is the
 * production V4 engine and this is a parallel implementation.
 *
 * Assembles the already-validated primitives (dsv4_*.h) at full scale: 43 layers,
 * 72,317 tensors, 156 GiB on disk.
 *
 * THE MEMORY SPLIT, which is the central design decision:
 *
 *   resident in RAM    8.67 GiB   attention, norms, routers, shared experts,
 *                                 embeddings and lm_head — IN THEIR NATIVE
 *                                 FORMAT, not dequantized
 *   streamed         137.1 GiB    the 11,008 routed experts, read off the NVMe
 *                                 when the router asks for them, with an LRU
 *
 * Dequantizing the dense set to f32 would be 26.8 GiB and would not fit; with
 * descriptors (dsv4_weight.h) the matmul reads the original format and never
 * materializes the matrix. Same decision colibri makes with its `QT` struct.
 *
 * Build:  make -C port engine
 * Usage:  ./dsv4_port <model_dir> "prompt" [n_tokens]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <pthread.h>

#include "compat.h"
#include "json.h"
#include "st.h"
#define DSV4_WITH_MXFP4
#include "quant.h"
#include "tok.h"

#include "omp_tune.h"   /* team sized to physical cores: see the note in main() */

#include "dsv4_fp8.h"
#include "dsv4_weight.h"
#include "dsv4_attn.h"
#include "dsv4_moe.h"
#include "dsv4_block.h"
#include "dsv4_decode.h"

static double now_s(void) {
#ifdef _WIN32
    return (double)clock() / CLOCKS_PER_SEC;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}

/* Profiling: where the time goes. Measure before optimizing. */
static double g_t_attn, g_t_moe, g_t_io, g_t_head;
static uint64_t g_pf_batches, g_pf_reads, g_fb_bytes;
static uint8_t g_seen[(43 * 256 + 7) / 8];   /* layers x experts, one bit each */
static uint64_t g_distinct;
static FILE *g_trace;   /* DSV4_TRACE=file -> dumps (token,layer,expert) */
static int g_tok_no;
/* The gateway's `cap`: in colibri this means cache slots PER LAYER, not in
 * total. It is multiplied by n_layers when the tier is built. */
static int g_cache_per_layer;

/* ---------------------------------------------------------------------------
 * Load policy, in the style of llama.cpp's `--load-mode`.
 *
 * There are two populations of weights with OPPOSITE access regimes, which is why
 * no single answer works:
 *
 *   dense, 8.67 GB   read on EVERY token. Its pages fault once and stay resident
 *                    forever -> mapping wins: loading is lazy and 8.67 GB is not
 *                    duplicated between the heap and the page cache.
 *   experts, 137 GB  each region is touched once and discarded -> mapping LOSES.
 *                    Measured: 23.7 s against 18.9. The slot buffers are reused
 *                    and their pages never fault again, whereas the mapping
 *                    faults on every new region: 13.4 MB per expert is ~3,400
 *                    faults of 4 KB. Trading a memcpy for 3,400 kernel entries
 *                    is not worth it.
 *
 * Mapping is not free, but it buys PRIVATE MEMORY, which is what actually decides
 * who the model fits for. Measured over 14 tokens:
 *
 *   DSV4_LOAD   private   working set   time
 *   read        13.2 GB       13.2 GB   19.3 s   <- default, the fastest
 *   dense        5.4 GB       12.2 GB   21.5 s   +11 %, -7.8 GB private
 *   all          0.6 GB       24.7 GB   24.4 s   +26 %, runs in almost nothing
 *
 * What `dense` buys is not free memory — the working set barely moves — it is
 * that those 8.67 GB go from anonymous to file-backed: under pressure the OS
 * DISCARDS them instead of pushing them to swap. `all` takes the process down to
 * 0.6 GB private in exchange for filling the working set and 26 % of the time.
 *
 * Same reason llama.cpp keeps `--load-mode` instead of choosing for you: there is
 * no good answer for every machine.
 * ------------------------------------------------------------------------- */
typedef enum { LOAD_READ = 0, LOAD_DENSE = 1, LOAD_ALL = 2 } LoadMode;

static LoadMode load_mode(void) {
    const char *s = getenv("DSV4_LOAD");
    if (!s || !strcmp(s, "read")) return LOAD_READ;      /* the default */
    if (!strcmp(s, "dense")) return LOAD_DENSE;
    if (!strcmp(s, "all"))   return LOAD_ALL;
    fprintf(stderr, "DSV4_LOAD must be read|dense|all\n");
    exit(1);
}

#ifdef _WIN32
static uint8_t *map_file(const char *path, uint64_t *len) {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz)) { CloseHandle(h); return NULL; }
    HANDLE m = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(h);                      /* the view keeps the file alive */
    if (!m) return NULL;
    void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    if (p) *len = (uint64_t)sz.QuadPart;
    return (uint8_t *)p;
}
#else
#include <sys/mman.h>
#include <sys/stat.h>
static uint8_t *map_file(const char *path, uint64_t *len) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st)) { close(fd); return NULL; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *len = (uint64_t)st.st_size;
    return (uint8_t *)p;
}
#endif

/* The mapped shards. Pointing the kernel at arbitrary offsets is safe:
 * quant.h does not use a single aligned load, only `loadu` (49 of 49). */
static uint8_t **g_smap;

static void smap_init(shards *S) {
    g_smap = calloc((size_t)S->nfd, sizeof(uint8_t *));
    for (int i = 0; i < S->nfd; i++) {
        uint64_t len = 0;
        g_smap[i] = map_file(S->paths[i], &len);
        if (!g_smap[i]) {
            fprintf(stderr, "could not map %s; falling back to reads\n", S->paths[i]);
            free(g_smap); g_smap = NULL;
            return;
        }
    }
}

/* Pointer to the tensor inside the mapping, or NULL if there is no mapping. */
static uint8_t *smap_ptr(shards *S, const char *nm) {
    if (!g_smap) return NULL;
    st_tensor *t = st_find(S, nm);
    if (!t) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    for (int i = 0; i < S->nfd; i++)
        if (S->fds[i] == t->fd) return g_smap[i] + t->off;
    return NULL;
}

/* ---------------------------------------------------------------------------
 * The streaming expert tier.
 *
 * One expert is 3 MXFP4 matrices plus their scales = 13.4 MB. All 11,008 do not
 * fit (137 GiB), so they are read from the shards on demand and held in an LRU
 * cache. The offsets are resolved once at load time: in decode nothing is looked
 * up by name, it is a `pread` to a known offset.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint8_t *buf[6];        /* w1,w1s, w2,w2s, w3,w3s */
    int layer, expert;
    uint64_t used;
    int pending;            /* reads in flight; 0 = ready to use */
} ExpSlot;

/* One read job: one tensor of one expert into one slot. */
typedef struct { int slot, layer, expert, k; } RdJob;

/* An expert tensor, already resolved: no name lookups on the hot path. */
typedef struct { int shard; int64_t off, nb; } ExpTensor;

typedef struct {
    shards *S;
    int n_layers, n_experts, cap, inter, dim;
    ExpTensor *tens;        /* [(layer*n_experts + e)*6 + k] */
    ExpSlot *slots;
    int nslot;
    uint64_t clock, hits, miss;
    uint64_t bytes;
    int mapped;             /* LOAD_ALL: no buffers, no LRU, read from the map */

    /* UN DESCRIPTOR POR HILO Y SHARD.
     *
     * On Windows, concurrent ReadFile calls on a handle opened in synchronous
     * mode are serialized by the OS itself on the file object, even when an
     * OVERLAPPED is supplied. Measured on this host: cold scattered reads give
     * 1.06 GB/s with one thread and 2.9 GB/s with four *if each has its own
     * descriptor* — sharing one keeps them at a single reader's rate. This was
     * why parallelizing the prefetch changed nothing. */
    int nthreads, nshard;
    int *fd;                /* [thread * nshard + shard] */

    /* POOL PERSISTENTE DE LECTORES.
     *
     * `tier_prefetch` enqueues and returns; the MoE starts computing the first
     * expert while the rest are still arriving. It used to block until all six
     * had landed, so the 4.3 s of compute and the 12.2 s of I/O ran in series.
     * It also removes the setup and teardown of one OpenMP region per layer per
     * token — 602 of them in a 14-token run. */
    RdJob *q;
    int qcap, qhead, qtail;
    pthread_mutex_t mu;
    pthread_cond_t cv_job, cv_done;
    pthread_t *th;
    int stop;
} ExpertTier;

static void tier_pool_start(ExpertTier *T);

static void tier_init(ExpertTier *T, shards *S, int n_layers, int n_experts,
                      int inter, int dim, int cap)
{
    memset(T, 0, sizeof *T);
    T->S = S; T->n_layers = n_layers; T->n_experts = n_experts;
    T->inter = inter; T->dim = dim; T->cap = cap;
    T->mapped = (load_mode() == LOAD_ALL && g_smap != NULL);
    if (T->mapped) cap = T->cap = 1;   /* the OS owns the cache */
    T->slots = calloc((size_t)cap, sizeof(ExpSlot));
    for (int i = 0; i < cap; i++) { T->slots[i].layer = -1; T->slots[i].expert = -1; }

    /* Resolve all 66,048 tensors up front: in decode nothing is looked up by
     * name, it is a pread to an offset that is already known. */
    T->tens = malloc((size_t)n_layers * n_experts * 6 * sizeof(ExpTensor));
    static const char *mats[3] = { "w1", "w2", "w3" };
    for (int l = 0; l < n_layers; l++)
        for (int e = 0; e < n_experts; e++)
            for (int k = 0; k < 3; k++)
                for (int which = 0; which < 2; which++) {
                    char nm[96];
                    snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.%s.%s",
                             l, e, mats[k], which ? "scale" : "weight");
                    st_tensor *t = st_find(S, nm);
                    if (!t) { fprintf(stderr, "missing %s\n", nm); exit(1); }
                    ExpTensor *d = &T->tens[((size_t)l * n_experts + e) * 6 + k * 2 + which];
                    d->off = t->off; d->nb = t->nbytes; d->shard = -1;
                    for (int i = 0; i < S->nfd; i++)
                        if (S->fds[i] == t->fd) { d->shard = i; break; }
                    if (d->shard < 0) { fprintf(stderr, "which shard is %s in?\n", nm); exit(1); }
                }

    /* One descriptor per thread and shard: see the struct comment. */
#ifdef _OPENMP
    T->nthreads = omp_get_max_threads();
#else
    T->nthreads = 1;
#endif
    T->nshard = S->nfd;
    T->fd = malloc((size_t)T->nthreads * T->nshard * sizeof(int));
    for (int t = 0; t < T->nthreads; t++)
        for (int i = 0; i < T->nshard; i++)
            T->fd[t * T->nshard + i] =
                (t == 0) ? S->fds[i] : open(S->paths[i], COMPAT_O_RDONLY);

    tier_pool_start(T);
}

/* Look the expert up in the cache; -1 if it is not there. */
static int tier_find(const ExpertTier *T, int layer, int e)
{
    for (int i = 0; i < T->nslot; i++)
        if (T->slots[i].layer == layer && T->slots[i].expert == e) return i;
    return -1;
}

/* Reserve a slot by LRU and mark it just-used, so that a later reservation
 * WITHIN THE SAME BATCH cannot evict it. */
static int tier_reserve(ExpertTier *T)
{
    int v;
    if (T->nslot < T->cap) v = T->nslot++;
    else {
        /* A slot with reads IN FLIGHT cannot be evicted: the workers are writing
         * into its buffers. At most `topk` are in flight and the cache holds
         * hundreds, so skipping them never leaves the LRU without a candidate. */
        v = -1;
        for (int i = 0; i < T->nslot; i++) {
            if (T->slots[i].pending) continue;
            if (v < 0 || T->slots[i].used < T->slots[v].used) v = i;
        }
        if (v < 0) { fprintf(stderr, "cache too small: everything is in flight\n"); exit(1); }
    }
    T->slots[v].used = ++T->clock;
    return v;
}

/* Allocate the buffers for an expert's 6 tensors and account the bytes. Does not
 * read: the reads are dispatched separately, see `tier_prefetch`. */
/* Every expert has the same shape, so the buffers are allocated once per slot and
 * never touched again: the `realloc` used to sit in the SERIAL part of every
 * prefetch batch, 19 times per layer per token. */
static void tier_alloc(ExpertTier *T, int slot, int layer, int e)
{
    ExpSlot *s = &T->slots[slot];
    const ExpTensor *d = &T->tens[((size_t)layer * T->n_experts + e) * 6];
    for (int k = 0; k < 6; k++) {
        if (!s->buf[k]) s->buf[k] = malloc((size_t)d[k].nb);
        T->bytes += (uint64_t)d[k].nb;
    }
    s->layer = layer; s->expert = e;
}

/* Read a single tensor through the CALLING THREAD's descriptor.
 *
 * No shared state: each thread uses its own fd and writes into a different
 * buffer. Only `pending` is touched under the mutex, in the worker loop. */
static void tier_read_one(ExpertTier *T, int slot, int layer, int e, int k, int tid)
{
    const ExpTensor *d = &T->tens[((size_t)layer * T->n_experts + e) * 6 + k];
    const int fd = T->fd[(size_t)tid * T->nshard + d->shard];
    uint8_t *out = T->slots[slot].buf[k];
    int64_t done = 0;
    while (done < d->nb) {
        const ssize_t got = pread(fd, out + done, (size_t)(d->nb - done), d->off + done);
        if (got <= 0) { fprintf(stderr, "short pread at layer %d expert %d\n", layer, e); exit(1); }
        done += got;
    }
}

/* A worker: pulls jobs off the queue until told to stop. Its index is also the
 * index of its own set of descriptors. */
typedef struct { ExpertTier *T; int tid; } RdArg;

static void *tier_worker(void *arg)
{
    RdArg *a = (RdArg *)arg;
    ExpertTier *T = a->T;
    const int tid = a->tid;
    for (;;) {
        pthread_mutex_lock(&T->mu);
        while (T->qhead == T->qtail && !T->stop)
            pthread_cond_wait(&T->cv_job, &T->mu);
        if (T->stop && T->qhead == T->qtail) { pthread_mutex_unlock(&T->mu); break; }
        const RdJob j = T->q[T->qhead];
        T->qhead = (T->qhead + 1) % T->qcap;
        pthread_mutex_unlock(&T->mu);

        tier_read_one(T, j.slot, j.layer, j.expert, j.k, tid);

        pthread_mutex_lock(&T->mu);
        if (--T->slots[j.slot].pending == 0) pthread_cond_broadcast(&T->cv_done);
        pthread_mutex_unlock(&T->mu);
    }
    return NULL;
}

static void tier_pool_start(ExpertTier *T)
{
    T->qcap = 4096;
    T->q = malloc((size_t)T->qcap * sizeof(RdJob));
    pthread_mutex_init(&T->mu, NULL);
    pthread_cond_init(&T->cv_job, NULL);
    pthread_cond_init(&T->cv_done, NULL);
    T->th = malloc((size_t)T->nthreads * sizeof(pthread_t));
    for (int i = 0; i < T->nthreads; i++) {
        RdArg *a = malloc(sizeof *a);   /* the thread keeps it */
        a->T = T; a->tid = i;
        pthread_create(&T->th[i], NULL, tier_worker, a);
    }
}

/* Enqueue an expert's 6 reads. Called with the mutex already held. */
static void tier_submit(ExpertTier *T, int slot, int layer, int e)
{
    T->slots[slot].pending = 6;
    for (int k = 0; k < 6; k++) {
        T->q[T->qtail] = (RdJob){ slot, layer, e, k };
        T->qtail = (T->qtail + 1) % T->qcap;
    }
    pthread_cond_broadcast(&T->cv_job);
}

/* Wait until a slot has all 6 of its tensors. */
static void tier_wait(ExpertTier *T, int slot)
{
    const double t0 = now_s();
    pthread_mutex_lock(&T->mu);
    while (T->slots[slot].pending) pthread_cond_wait(&T->cv_done, &T->mu);
    pthread_mutex_unlock(&T->mu);
    g_t_io += now_s() - t0;   /* now measures WAITING, not reading: that is the cost */
}

/* Fetch, in one go, the `n` experts the router has just chosen.
 *
 * This is the difference between queue depth 1 and n: an NVMe only gives its
 * bandwidth with several reads in flight, and one at a time it is latency-bound.
 * Slot reservation stays serial — the LRU is global state — and only the reads
 * are spread out. */
static void tier_prefetch(ExpertTier *T, int layer, const int *es, int n)
{
    int slot[16], want[16], nw = 0;
    if (n > 16) n = 16;
    if (T->mapped) {
        /* The OS is ASKED to bring the ranges in, without touching them:
         * touching them would cost the same bandwidth as reading them. It is a
         * hint, not a guarantee; if a page has not arrived, the fault resolves at
         * read time. */
#ifdef _WIN32
        WIN32_MEMORY_RANGE_ENTRY r[16 * 6];
        int nr = 0;
        for (int k = 0; k < n; k++) {
            const ExpTensor *d = &T->tens[((size_t)layer * T->n_experts + es[k]) * 6];
            for (int j = 0; j < 6; j++, nr++) {
                r[nr].VirtualAddress = g_smap[d[j].shard] + d[j].off;
                r[nr].NumberOfBytes  = (SIZE_T)d[j].nb;
            }
        }
        if (nr) PrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR)nr, r, 0);
#else
        for (int k = 0; k < n; k++) {
            const ExpTensor *d = &T->tens[((size_t)layer * T->n_experts + es[k]) * 6];
            for (int j = 0; j < 6; j++)
                madvise(g_smap[d[j].shard] + d[j].off, (size_t)d[j].nb, MADV_WILLNEED);
        }
#endif
        return;
    }
    if (g_trace) for (int k = 0; k < n; k++)
        fprintf(g_trace, "%d %d %d\n", g_tok_no, layer, es[k]);
    for (int k = 0; k < n; k++) {
        int dup = 0;
        for (int j = 0; j < nw; j++) if (want[j] == es[k]) { dup = 1; break; }
        if (dup) continue;
        const int f = tier_find(T, layer, es[k]);
        if (f >= 0) { T->hits++; T->slots[f].used = ++T->clock; continue; }
        T->miss++;
        /* How many DISTINCT experts the whole run asks for: that is the miss
         * count an infinite cache would have, i.e. the ceiling on what growing
         * this one can buy. If we are already close, growing it is pointless. */
        {   const size_t bit = (size_t)layer * T->n_experts + es[k];
            if (!(g_seen[bit >> 3] & (1u << (bit & 7)))) {
                g_seen[bit >> 3] |= (uint8_t)(1u << (bit & 7));
                g_distinct++;
            }
        }
        want[nw] = es[k];
        slot[nw] = tier_reserve(T);
        nw++;
    }
    if (!nw) return;
    g_pf_batches++; g_pf_reads += (uint64_t)nw;

    /* Enqueue and return. The split is PER TENSOR, not per expert: at ~3.2
     * experts per batch, one job per expert would leave the NVMe queue at depth
     * 3.2 instead of 19. */
    for (int j = 0; j < nw; j++) tier_alloc(T, slot[j], layer, want[j]);
    pthread_mutex_lock(&T->mu);
    for (int j = 0; j < nw; j++) tier_submit(T, slot[j], layer, want[j]);
    pthread_mutex_unlock(&T->mu);
}

/* Returns the expert's descriptors. After `tier_prefetch` this always hits; the
 * loading path stays as a safety net for callers that do not prefetch. */
static void tier_get(ExpertTier *T, int layer, int e, DsV4W *w1, DsV4W *w2, DsV4W *w3)
{
    if (T->mapped) {
        /* No copy and no cache of our own: the descriptors point at the page and
         * the cache is the OS's. Costs ~25 % more time (a page fault per 4 KB of
         * new region) and saves the buffers' gigabytes. */
        const ExpTensor *d = &T->tens[((size_t)layer * T->n_experts + e) * 6];
        uint8_t *p[6];
        for (int k = 0; k < 6; k++) {
            p[k] = g_smap[d[k].shard] + d[k].off;
            T->bytes += (uint64_t)d[k].nb;
        }
        *w1 = dsv4_w_mxfp4(p[0], p[1], T->inter, T->dim);
        *w2 = dsv4_w_mxfp4(p[2], p[3], T->dim,   T->inter);
        *w3 = dsv4_w_mxfp4(p[4], p[5], T->inter, T->dim);
        return;
    }

    int hit = tier_find(T, layer, e);
    if (hit < 0) {
        T->miss++;
        hit = tier_reserve(T);
        const uint64_t b0 = T->bytes;
        tier_alloc(T, hit, layer, e);
        pthread_mutex_lock(&T->mu);
        tier_submit(T, hit, layer, e);
        pthread_mutex_unlock(&T->mu);
        g_fb_bytes += T->bytes - b0;
    }
    tier_wait(T, hit);   /* after a prefetch this usually returns immediately */

    ExpSlot *s = &T->slots[hit];
    s->used = ++T->clock;
    *w1 = dsv4_w_mxfp4(s->buf[0], s->buf[1], T->inter, T->dim);
    *w2 = dsv4_w_mxfp4(s->buf[2], s->buf[3], T->dim,   T->inter);
    *w3 = dsv4_w_mxfp4(s->buf[4], s->buf[5], T->inter, T->dim);
}

/* Adapter: the MoE asks for experts by callback and knows nothing about shards. */
static void tier_fetch(void *ctx, int layer, int e,
                       DsV4W *w1, DsV4W *w2, DsV4W *w3) {
    tier_get((ExpertTier *)ctx, layer, e, w1, w2, w3);
}
static void tier_prefetch_cb(void *ctx, int layer, const int *es, int n) {
    tier_prefetch((ExpertTier *)ctx, layer, es, n);
}

/* ---------------------------------------------------------------------------
 * El modelo
 * ------------------------------------------------------------------------- */
typedef struct {
    shards S;
    Tok tok;
    ExpertTier tier;

    int n_layers, dim, vocab, hc, n_hash, sinkhorn_iters;
    int bos, eos;           /* from config.json: 0 and 1 in this checkpoint */
    float norm_eps, hc_eps;
    int *ratios;

    DsV4BlockCfg *cfg;      /* [n_layers] */
    DsV4BlockW   *w;        /* [n_layers] */
    int32_t *tid2eid;       /* [vocab * topk], only when there are hash layers */

    DsV4W embed, head;
    const float *final_norm, *hc_head_fn, *hc_head_base, *hc_head_scale;
    float *freqs;           /* [max_seq, rd/2, 2] */
    int max_seq;
} Model;

/* --- load helpers ------------------------------------------------------- */
static void *raw_of(shards *S, const char *nm) {
    const int64_t n = st_nbytes(S, nm);
    void *p = malloc((size_t)n);
    st_read_raw(S, nm, p, 0);
    return p;
}
static float *vec_f32(shards *S, const char *nm, int n) {
    float *out = malloc((size_t)n * sizeof(float));
    st_tensor *t = st_find(S, nm);
    if (!t) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    if (t->dtype == 2) { st_read_f32(S, nm, out, 0); return out; }   /* F32 */
    uint8_t *r = raw_of(S, nm);                                       /* BF16 */
    DsV4W v = dsv4_w_bf16(r, 1, n);
    dsv4_vec_f32(out, &v, n);
    free(r);
    return out;
}
/* A quantized matrix: FP8 when it has a .scale, BF16 otherwise.
 *
 * This is the bulk of the dense set (attention, embed, lm_head, shared experts),
 * and it comes from the mapping when there is one: nothing is copied and loading
 * is lazy. */
static DsV4W mat_of(shards *S, const char *base, int O, int I) {
    char nw[192], ns[192];
    snprintf(nw, sizeof nw, "%s.weight", base);
    snprintf(ns, sizeof ns, "%s.scale", base);
    uint8_t *wb = smap_ptr(S, nw);
    if (!wb) wb = raw_of(S, nw);
    if (st_has(S, ns)) {
        uint8_t *sb = smap_ptr(S, ns);
        if (!sb) sb = raw_of(S, ns);
        return dsv4_w_fp8b(wb, sb, O, I);
    }
    return dsv4_w_bf16(wb, O, I);
}

static void model_load(Model *M, const char *dir) {
    memset(M, 0, sizeof *M);
    double t0 = now_s();
    st_init(&M->S, dir);
    if (load_mode() != LOAD_READ) smap_init(&M->S);

    /* --- config.json ---------------------------------------------------- */
    char cfgp[1024];
    snprintf(cfgp, sizeof cfgp, "%s/config.json", dir);
    FILE *f = fopen(cfgp, "rb");
    if (!f) { fprintf(stderr, "cannot find %s\n", cfgp); exit(1); }
    fseek(f, 0, SEEK_END); long cn = ftell(f); fseek(f, 0, SEEK_SET);
    char *cbuf = malloc((size_t)cn + 1);
    if (fread(cbuf, 1, (size_t)cn, f) != (size_t)cn) { fprintf(stderr, "config truncated\n"); exit(1); }
    cbuf[cn] = 0; fclose(f);
    char *arena = NULL;
    jval *C = json_parse(cbuf, &arena);
#define GI(k) ((int)json_get(C,k)->num)
#define GF(k) ((float)json_get(C,k)->num)
    M->n_layers = GI("num_hidden_layers");
    M->dim      = GI("hidden_size");
    M->vocab    = GI("vocab_size");
    M->bos      = GI("bos_token_id");
    M->eos      = GI("eos_token_id");
    M->hc       = GI("hc_mult");
    M->n_hash   = GI("num_hash_layers");
    M->sinkhorn_iters = GI("hc_sinkhorn_iters");
    M->norm_eps = GF("rms_norm_eps");
    M->hc_eps   = GF("hc_eps");
    const int heads = GI("num_attention_heads"), hd = GI("head_dim");
    const int rd = GI("qk_rope_head_dim"), q_lora = GI("q_lora_rank");
    const int groups = GI("o_groups"), o_lora = GI("o_lora_rank");
    const int win = GI("sliding_window");
    const int n_exp = GI("n_routed_experts"), topk = GI("num_experts_per_tok");
    const int inter = GI("moe_intermediate_size");
    const int i_heads = GI("index_n_heads"), i_hd = GI("index_head_dim");
    const int i_topk = GI("index_topk");
    const float route_scale = GF("routed_scaling_factor");
    const float swiglu = GF("swiglu_limit");
    jval *rs = json_get(C, "rope_scaling");
    const double base = json_get(C, "rope_theta")->num;
    const double factor = json_get(rs, "factor")->num;
    const int orig = (int)json_get(rs, "original_max_position_embeddings")->num;
    const double bf = json_get(rs, "beta_fast")->num, bs = json_get(rs, "beta_slow")->num;
    jval *cr = json_get(C, "compress_ratios");
    M->ratios = malloc((size_t)M->n_layers * sizeof(int));
    for (int i = 0; i < M->n_layers; i++) M->ratios[i] = (int)cr->kids[i]->num;
#undef GI
#undef GF

    /* Diagnostics ALWAYS go to stderr: in serve mode stdout IS the protocol, and
     * one stray line there throws the gateway off. */
    fprintf(stderr, "config: %d layers, dim %d, %d heads x %d, %d experts top-%d\n",
            M->n_layers, M->dim, heads, hd, n_exp, topk);

    /* --- freqs_cis with YaRN -------------------------------------------- */
    M->max_seq = 2048;                 /* enough for test prompts */
    M->freqs = malloc((size_t)M->max_seq * (rd / 2) * 2 * sizeof(float));
    dsv4_precompute_freqs(M->freqs, rd, M->max_seq, orig, base, factor, bf, bs);

    /* --- globales -------------------------------------------------------- */
    M->embed = mat_of(&M->S, "embed", M->vocab, M->dim);
    M->head  = mat_of(&M->S, "head",  M->vocab, M->dim);
    M->final_norm    = vec_f32(&M->S, "norm.weight", M->dim);
    M->hc_head_fn    = (float *)raw_of(&M->S, "hc_head_fn");
    M->hc_head_base  = (float *)raw_of(&M->S, "hc_head_base");
    M->hc_head_scale = (float *)raw_of(&M->S, "hc_head_scale");

    /* --- per layer ------------------------------------------------------- */
    M->cfg = calloc((size_t)M->n_layers, sizeof(DsV4BlockCfg));
    M->w   = calloc((size_t)M->n_layers, sizeof(DsV4BlockW));
    char nm[192];
    for (int L = 0; L < M->n_layers; L++) {
        DsV4BlockCfg *c = &M->cfg[L];
        DsV4BlockW *w = &M->w[L];
        const int ratio = M->ratios[L];

        c->hc = M->hc; c->sinkhorn_iters = M->sinkhorn_iters;
        c->hc_eps = M->hc_eps; c->norm_eps = M->norm_eps;
        c->attn.dim = M->dim; c->attn.q_lora = q_lora; c->attn.heads = heads;
        c->attn.hd = hd; c->attn.rd = rd; c->attn.groups = groups;
        c->attn.o_lora = o_lora; c->attn.win = win; c->attn.eps = M->norm_eps;
        c->attn.scale = 1.0f / sqrtf((float)hd);
        c->attn.ratio = ratio;
        c->moe.dim = M->dim; c->moe.n_experts = n_exp; c->moe.topk = topk;
        c->moe.inter = inter; c->moe.route_scale = route_scale;
        c->moe.swiglu_limit = swiglu; c->moe.hash = (L < M->n_hash);

#define B(fmt, ...) (snprintf(nm, sizeof nm, fmt, __VA_ARGS__), nm)
        w->attn.wq_a = mat_of(&M->S, B("layers.%d.attn.wq_a", L), q_lora, M->dim);
        w->attn.wq_b = mat_of(&M->S, B("layers.%d.attn.wq_b", L), heads * hd, q_lora);
        w->attn.wkv  = mat_of(&M->S, B("layers.%d.attn.wkv",  L), hd, M->dim);
        w->attn.wo_a = mat_of(&M->S, B("layers.%d.attn.wo_a", L), groups * o_lora,
                              heads * hd / groups);
        w->attn.wo_b = mat_of(&M->S, B("layers.%d.attn.wo_b", L), M->dim, groups * o_lora);
        w->attn.q_norm    = vec_f32(&M->S, B("layers.%d.attn.q_norm.weight", L), q_lora);
        w->attn.kv_norm   = vec_f32(&M->S, B("layers.%d.attn.kv_norm.weight", L), hd);
        w->attn.attn_sink = vec_f32(&M->S, B("layers.%d.attn.attn_sink", L), heads);
        w->attn.freqs = M->freqs;

        if (ratio) {
            const int coff = (ratio == 4) ? 2 : 1;
            w->attn.c_wkv   = mat_of(&M->S, B("layers.%d.attn.compressor.wkv", L),
                                     coff * hd, M->dim);
            w->attn.c_wgate = mat_of(&M->S, B("layers.%d.attn.compressor.wgate", L),
                                     coff * hd, M->dim);
            w->attn.c_norm  = vec_f32(&M->S, B("layers.%d.attn.compressor.norm.weight", L), hd);
            w->attn.c_ape   = (float *)raw_of(&M->S, B("layers.%d.attn.compressor.ape", L));
            if (ratio == 4) {
                c->attn.i_heads = i_heads; c->attn.i_hd = i_hd; c->attn.i_topk = i_topk;
                c->attn.i_scale = 1.0f / sqrtf((float)i_hd);
                w->attn.i_wkv   = mat_of(&M->S, B("layers.%d.attn.indexer.compressor.wkv", L),
                                         coff * i_hd, M->dim);
                w->attn.i_wgate = mat_of(&M->S, B("layers.%d.attn.indexer.compressor.wgate", L),
                                         coff * i_hd, M->dim);
                w->attn.i_norm  = vec_f32(&M->S, B("layers.%d.attn.indexer.compressor.norm.weight", L), i_hd);
                w->attn.i_ape   = (float *)raw_of(&M->S, B("layers.%d.attn.indexer.compressor.ape", L));
                w->attn.i_wq_b  = mat_of(&M->S, B("layers.%d.attn.indexer.wq_b", L),
                                         i_heads * i_hd, q_lora);
                w->attn.i_wproj = mat_of(&M->S, B("layers.%d.attn.indexer.weights_proj", L),
                                         i_heads, M->dim);
            }
        }

        w->attn_norm = vec_f32(&M->S, B("layers.%d.attn_norm.weight", L), M->dim);
        w->ffn_norm  = vec_f32(&M->S, B("layers.%d.ffn_norm.weight", L), M->dim);
        w->hc_attn_fn    = (float *)raw_of(&M->S, B("layers.%d.hc_attn_fn", L));
        w->hc_attn_base  = (float *)raw_of(&M->S, B("layers.%d.hc_attn_base", L));
        w->hc_attn_scale = (float *)raw_of(&M->S, B("layers.%d.hc_attn_scale", L));
        w->hc_ffn_fn     = (float *)raw_of(&M->S, B("layers.%d.hc_ffn_fn", L));
        w->hc_ffn_base   = (float *)raw_of(&M->S, B("layers.%d.hc_ffn_base", L));
        w->hc_ffn_scale  = (float *)raw_of(&M->S, B("layers.%d.hc_ffn_scale", L));

        w->moe.gate_w = mat_of(&M->S, B("layers.%d.ffn.gate", L), n_exp, M->dim);
        if (c->moe.hash) {
            /* the table is shared by every hash layer: loaded once */
            if (!M->tid2eid) {
                snprintf(nm, sizeof nm, "layers.%d.ffn.gate.tid2eid", L);
                const int64_t nb = st_nbytes(&M->S, nm);
                int64_t *r = malloc((size_t)nb);
                st_read_raw(&M->S, nm, r, 0);
                const int64_t n64 = nb / 8;
                M->tid2eid = malloc((size_t)n64 * sizeof(int32_t));
                for (int64_t i = 0; i < n64; i++) M->tid2eid[i] = (int32_t)r[i];
                free(r);
            }
            w->moe.tid2eid = M->tid2eid;
        } else {
            w->moe.gate_bias = vec_f32(&M->S, B("layers.%d.ffn.gate.bias", L), n_exp);
        }
        w->moe.s_w1 = mat_of(&M->S, B("layers.%d.ffn.shared_experts.w1", L), inter, M->dim);
        w->moe.s_w2 = mat_of(&M->S, B("layers.%d.ffn.shared_experts.w2", L), M->dim, inter);
        w->moe.s_w3 = mat_of(&M->S, B("layers.%d.ffn.shared_experts.w3", L), inter, M->dim);
#undef B
        if ((L + 1) % 8 == 0 || L == M->n_layers - 1)
            fprintf(stderr, "\r  layers loaded: %d/%d", L + 1, M->n_layers), fflush(stderr);
    }
    fprintf(stderr, "\n");

    /* --- the expert tier ------------------------------------------------- */
    /* ~13.4 MB per expert. The size governs the I/O, which is the real
     * bottleneck, so it stays adjustable in order to be measurable. */
    { const char *tp = getenv("DSV4_TRACE"); if (tp) g_trace = fopen(tp, "w"); }
    const char *cenv = getenv("DSV4_CACHE");
    const int cache = cenv ? atoi(cenv)
                     : (g_cache_per_layer ? g_cache_per_layer * M->n_layers : 384);
    tier_init(&M->tier, &M->S, M->n_layers, n_exp, inter, M->dim, cache);
    /* Each layer asks the tier for its experts by callback: the MoE knows nothing
     * about shards or about cache policy. */
    for (int L = 0; L < M->n_layers; L++) {
        M->w[L].moe.fetch = tier_fetch;
        M->w[L].moe.prefetch = tier_prefetch_cb;
        M->w[L].moe.fetch_ctx = &M->tier;
        M->w[L].moe.layer = L;
    }
    fprintf(stderr, "experts: streaming with a %d-slot cache out of %d total\n",
            cache, M->n_layers * n_exp);
    fprintf(stderr, "loaded in %.1f s\n\n", now_s() - t0);
}

/* ---------------------------------------------------------------------------
 * Forward for one token (decode) or one prompt (prefill).
 * `h` is the [hc, dim] residual stream of a SINGLE position (batch 1).
 * ------------------------------------------------------------------------- */
static void model_layer_decode(Model *M, int L, DsV4AttnState *st,
                               const float *hin, int pos, int32_t tokid,
                               float *hout)
{
    DsV4BlockCfg *c = &M->cfg[L];
    DsV4BlockW *w = &M->w[L];
    const int dim = M->dim, hc = M->hc;

    float post[DSV4_MAX_HC], comb[DSV4_MAX_HC * DSV4_MAX_HC];
    float *coll = malloc((size_t)dim * sizeof(float));
    float *sub  = malloc((size_t)dim * sizeof(float));
    float *mid  = malloc((size_t)hc * dim * sizeof(float));
    float *tmp  = malloc((size_t)dim * sizeof(float));

    /* --- attention --- */
    dsv4_hc_pre(hin, w->hc_attn_fn, w->hc_attn_scale, w->hc_attn_base,
                hc, dim, c->sinkhorn_iters, c->hc_eps, c->norm_eps,
                coll, post, comb);
    dsv4_rmsnorm(tmp, coll, w->attn_norm, dim, c->norm_eps);
    for (int d = 0; d < dim; d++) coll[d] = dsv4_to_bf16(tmp[d]);
    { const double _t = now_s();
      dsv4_attention_decode(&c->attn, &w->attn, st, coll, pos, sub);
      g_t_attn += now_s() - _t; }
    dsv4_hc_expand(mid, sub, post, comb, hin, hc, dim);

    /* --- MoE --- */
    dsv4_hc_pre(mid, w->hc_ffn_fn, w->hc_ffn_scale, w->hc_ffn_base,
                hc, dim, c->sinkhorn_iters, c->hc_eps, c->norm_eps,
                coll, post, comb);
    dsv4_rmsnorm(tmp, coll, w->ffn_norm, dim, c->norm_eps);
    for (int d = 0; d < dim; d++) coll[d] = dsv4_to_bf16(tmp[d]);
    /* only the hash layers use the token id, but it is always passed */
    { const double _t = now_s();
      dsv4_moe_forward(&c->moe, &w->moe, coll, &tokid, 1, sub);
      g_t_moe += now_s() - _t; }
    dsv4_hc_expand(hout, sub, post, comb, mid, hc, dim);

    free(coll); free(sub); free(mid); free(tmp);
}

/* ---------------------------------------------------------------------------
 * A reusable decode pass.
 *
 * Shared by CLI mode and serve mode: the only difference between generating from
 * the command line and generating through the HTTP gateway is where the prompt
 * comes from and where the tokens go.
 * ------------------------------------------------------------------------- */
typedef struct {
    Model *M;
    DsV4AttnState *st;
    float *h, *h2, *logits, *emb;
    int pos;
    /* The ids ALREADY fed by `run_step`, in order. This is the history the next
     * prompt is compared against in order to reuse the common prefix. It is
     * recorded inside run_step, so it cannot drift out of sync with the state no
     * matter how the loop above changes. */
    int *hist, nhist, hcap;
} Run;

static void run_init(Run *R, Model *M) {
    memset(R, 0, sizeof *R);
    R->M = M;
    R->st = calloc((size_t)M->n_layers, sizeof(DsV4AttnState));
    for (int L = 0; L < M->n_layers; L++)
        dsv4_state_init_full(&R->st[L], 1, M->cfg[L].attn.hd, M->cfg[L].attn.win,
                             M->max_seq, M->cfg[L].attn.ratio, M->cfg[L].attn.i_hd);
    R->h      = malloc((size_t)M->hc * M->dim * sizeof(float));
    R->h2     = malloc((size_t)M->hc * M->dim * sizeof(float));
    R->emb    = malloc((size_t)M->dim * sizeof(float));
    R->logits = malloc((size_t)M->vocab * sizeof(float));
    R->hcap = 65536;
    R->hist = malloc((size_t)R->hcap * sizeof(int));
}

/* Between requests ALL of the attention state has to go, not just the KV ring:
 * every compressed layer carries the compressor's in-progress block and, at
 * ratio 4, the indexer's as well. Reusing one half-way mixes two conversations in
 * a way that raises no error, just strange text. */
static void run_reset(Run *R) {
    Model *M = R->M;
    for (int L = 0; L < M->n_layers; L++) {
        dsv4_state_free(&R->st[L]);
        dsv4_state_init_full(&R->st[L], 1, M->cfg[L].attn.hd, M->cfg[L].attn.win,
                             M->max_seq, M->cfg[L].attn.ratio, M->cfg[L].attn.i_hd);
    }
    R->pos = 0;
    R->nhist = 0;
}

/* One token in, the logits out. */
static const float *run_step(Run *R, int tokid) {
    Model *M = R->M;
    const int dim = M->dim, hc = M->hc;
    {
        const DsV4W row = dsv4_w_rows(&M->embed, tokid, 1);
        const uint16_t *p = (const uint16_t *)row.w;
        for (int d = 0; d < dim; d++) R->emb[d] = dsv4_bf16_to_f32(p[d]);
        for (int m = 0; m < hc; m++)
            memcpy(R->h + (size_t)m * dim, R->emb, (size_t)dim * sizeof(float));
    }
    for (int L = 0; L < M->n_layers; L++) {
        model_layer_decode(M, L, &R->st[L], R->h, R->pos, (int32_t)tokid, R->h2);
        memcpy(R->h, R->h2, (size_t)hc * dim * sizeof(float));
    }
    R->pos++;
    if (R->hist && R->nhist < R->hcap) R->hist[R->nhist++] = tokid;

    float y[8192], nz[8192];
    dsv4_hc_head(R->h, M->hc_head_fn, M->hc_head_scale, M->hc_head_base,
                 hc, dim, M->norm_eps, M->hc_eps, y);
    dsv4_rmsnorm(nz, y, M->final_norm, dim, M->norm_eps);
    for (int d = 0; d < dim; d++) nz[d] = dsv4_to_bf16(nz[d]);
    const double t = now_s();
    dsv4_matmul_w(R->logits, nz, &M->head, 1, 0);
    g_t_head += now_s() - t;
    return R->logits;
}

/* ---------------------------------------------------------------------------
 * Sampling. The CLI uses argmax; the gateway sends temperature and top_p per
 * request, so real nucleus sampling is needed.
 * ------------------------------------------------------------------------- */
static uint64_t g_rng = 0x853c49e6748fea9bULL;
static double rnd01(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

static const float *g_sort_logits;
static int cmp_by_logit(const void *a, const void *b) {
    const float x = g_sort_logits[*(const int *)a];
    const float y = g_sort_logits[*(const int *)b];
    return (x < y) - (x > y);          /* descendente */
}

static int sample_tok(const float *logits, int n, float temp, float top_p) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int v = 1; v < n; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
    static int *idx = NULL;
    static float *pr = NULL;
    if (!idx) { idx = malloc((size_t)n * sizeof(int)); pr = malloc((size_t)n * sizeof(float)); }
    for (int v = 0; v < n; v++) idx[v] = v;
    g_sort_logits = logits;
    qsort(idx, (size_t)n, sizeof(int), cmp_by_logit);

    const float top = logits[idx[0]];
    double sum = 0.0;
    for (int v = 0; v < n; v++) {
        pr[v] = expf((logits[idx[v]] - top) / temp);
        sum += pr[v];
    }
    /* nucleus: cut once the accumulated mass reaches top_p */
    const double cut = (top_p > 0.0f && top_p < 1.0f) ? top_p * sum : sum;
    double acc = 0.0;
    int last = n - 1;
    for (int v = 0; v < n; v++) { acc += pr[v]; if (acc >= cut) { last = v; break; } }

    const double r = rnd01() * acc;
    double c = 0.0;
    for (int v = 0; v <= last; v++) { c += pr[v]; if (c >= r) return idx[v]; }
    return idx[last];
}

/* ---------------------------------------------------------------------------
 * Serve mode: the mux protocol openai_server.py speaks.
 *
 * The same line format as colibri.c and inkling.c — byte for byte, so the
 * gateway can be shared — with one simplification: a single KV slot. The mux
 * allows up to 16 and reuses each conversation's common prefix; here a single
 * slot is kept warm across requests instead, which avoids replicating the
 * compressor and indexer states per slot.
 * ------------------------------------------------------------------------- */
static double rss_gb(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (double)r.ru_maxrss / 1e6;
}

typedef struct {
    char id[64];
    int max_tok;
    float temp, top_p;
    char *payload;
    int plen;
} ServeReq;

/* -1 EOF, 0 nothing useful, 1 cancel the active request, 2 a new request */
static int serve_read_req(ServeReq *q, const char *active) {
    char line[512], cmd[16], id[64];
    if (!fgets(line, sizeof line, stdin)) return -1;
    if (sscanf(line, "%15s %63s", cmd, id) < 2) return 0;
    if (!strcmp(cmd, "CANCEL") || !strcmp(cmd, "STOP"))
        return active && !strcmp(active, id);
    if (strcmp(cmd, "SUBMIT")) return 0;

    int slot, plen, max_tok; float temp, top_p;
    if (sscanf(line, "%*s %*s %d %d %d %f %f", &slot, &plen, &max_tok, &temp, &top_p) != 5
        || plen < 0 || plen > (1 << 24) || max_tok < 1) {
        printf("ERROR %s BAD_FRAME\n", id); fflush(stdout); return 0;
    }
    (void)slot;                        /* single slot: see the note above */
    char *payload = malloc((size_t)plen + 1);
    if (!payload) { printf("ERROR %s BAD_REQUEST\n", id); fflush(stdout); return 0; }
    if (fread(payload, 1, (size_t)plen, stdin) != (size_t)plen) { free(payload); return -1; }
    (void)fgetc(stdin);                /* the \n that closes the frame */
    payload[plen] = 0;
    snprintf(q->id, sizeof q->id, "%s", id);
    q->max_tok = max_tok; q->temp = temp; q->top_p = top_p;
    q->payload = payload; q->plen = plen;
    return 2;
}

static void serve_data(const char *id, const char *p, int n) {
    if (n <= 0) return;
    printf("DATA %s %d\n", id, n);
    fwrite(p, 1, (size_t)n, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

static void serve_one(Run *R, ServeReq *q) {
    Model *M = R->M;
    const int cap = 65536;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = 0;
    ids[np++] = M->bos;                              /* the model expects it */
    np += tok_encode(&M->tok, q->payload, q->plen, ids + np, cap - np);
    if (np <= 1) {
        printf("ERROR %s EMPTY_PROMPT\n", q->id); fflush(stdout); free(ids); return;
    }

    /* PREFIX REUSE (the protocol's "truncate-and-extend").
     *
     * In a conversation, a turn's prompt is the previous one plus the model's
     * reply plus the new message: a pure EXTENSION of what has already been fed.
     * Comparing against the history skips all of those layers and processes only
     * the new tail. Without this, the third turn of a 300-token chat costs ~420 s
     * before the first word appears.
     *
     * When the prompt DIVERGES (the client edits the history, or a different
     * conversation arrives) the state would have to be rewound, and that is not
     * possible here: the KV ring could be, but the compressors and the indexer
     * accumulate an in-progress block that cannot be "un-accumulated". In that
     * case everything is rebuilt, which is correct and merely slower. */
    int reuse = 0;
    while (reuse < R->nhist && reuse < np && R->hist[reuse] == ids[reuse]) reuse++;
    if (reuse < R->nhist || reuse >= np) { run_reset(R); reuse = 0; }
    if (reuse) fprintf(stderr, "[serve] prefix reused: %d of %d tokens\n", reuse, np);

    const uint64_t hit0 = M->tier.hits, miss0 = M->tier.miss;
    const double t0 = now_s();

    int gen = 0, limited = 1, cancelled = 0;
    int tokid = ids[reuse];
    double tdec = 0.0;
    for (int step = reuse; ; step++) {
        const float *lo = run_step(R, tokid);
        if (step + 1 < np) { tokid = ids[step + 1]; continue; }   /* still prefill */
        if (gen == 0) tdec = now_s();

        const int nx = sample_tok(lo, M->vocab, q->temp, q->top_p);
        /* 128805 = <|EOT|>: DeepSeek's chat format uses it in addition to the
         * end-of-sentence id from the config. Stopping only on the latter lets
         * the turn run until max_tokens is exhausted. */
        if (nx == M->eos || nx == 128805) { limited = 0; break; }

        char piece[64];
        const int m = tok_decode(&M->tok, &nx, 1, piece, sizeof piece - 1);
        serve_data(q->id, piece, m);
        gen++;
        tokid = nx;

        /* La pasarela puede cancelar mientras generamos. */
        while (coli_stdin_readable()) {
            ServeReq extra = { 0 };
            const int r = serve_read_req(&extra, q->id);
            if (r < 0) { cancelled = 1; break; }
            if (r == 1) cancelled = 1;
            if (r == 2) {                            /* single slot */
                printf("ERROR %s SLOT_BUSY\n", extra.id); fflush(stdout);
                free(extra.payload);
            }
        }
        if (cancelled) { limited = 0; break; }
        if (gen >= q->max_tok) break;
    }

    const double decode = now_s() - (tdec > 0 ? tdec : t0);
    const uint64_t hits = M->tier.hits - hit0, miss = M->tier.miss - miss0;
    const uint64_t tot = hits + miss;
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen,
           decode > 0 ? gen / decode : 0.0,
           tot ? 100.0 * (double)hits / (double)tot : 0.0,
           rss_gb(), np, limited);
    fflush(stdout);
    free(ids);
}

static void serve_loop(Run *R) {
    setvbuf(stdin, NULL, _IONBF, 0);
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb());
    fflush(stdout);
    for (;;) {
        ServeReq q = { 0 };
        int r;
        do r = serve_read_req(&q, NULL); while (r == 0);
        if (r < 0) return;                       /* EOF: apagado ordenado */
        if (r == 2) { serve_one(R, &q); free(q.payload); }
    }
}

#ifdef _WIN32
#include <shellapi.h>          /* CommandLineToArgvW */
/* ---------------------------------------------------------------------------
 * argv as UTF-8.
 *
 * Windows hands `argv` to the process in the ANSI codepage, not UTF-8. A prompt
 * containing "España" arrives with `ñ` as the single byte 0xF1 (cp1252) instead
 * of the two bytes C3 B1, which is not valid UTF-8. The tokenizer is byte-level
 * BPE, so it does not reject it — it silently tokenizes the stray byte as its own
 * piece, and generation degrades:
 *
 *   in:  "La capital de España es "   ->  the engine sees ...Espa\xF1a...
 *   out: garbage bytes, then plausible text
 *
 * Reading the command line as UTF-16 and converting it ourselves is the only
 * reliable fix; there is no codepage setting a caller can apply that makes the
 * ANSI argv lossless for arbitrary text. SetConsoleOutputCP is the other half of
 * the problem: without it, correct UTF-8 output still renders as mojibake.
 * ------------------------------------------------------------------------- */
static void win_argv_utf8(int *argc, char ***argv) {
    int n = 0;
    LPWSTR *w = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!w || n <= 0) return;
    char **out = (char **)malloc((size_t)(n + 1) * sizeof(char *));
    if (!out) { LocalFree(w); return; }
    for (int i = 0; i < n; i++) {
        const int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, NULL, 0, NULL, NULL);
        out[i] = (char *)malloc((size_t)(len > 0 ? len : 1));
        if (len > 0) WideCharToMultiByte(CP_UTF8, 0, w[i], -1, out[i], len, NULL, NULL);
        else out[i][0] = 0;
    }
    out[n] = NULL;
    LocalFree(w);
    *argc = n;
    *argv = out;
}
#endif

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);   /* so our UTF-8 output renders, not mojibake */
    win_argv_utf8(&argc, &argv);   /* see the note above: argv arrives as ANSI */
#endif

    /* Size the OpenMP team to PHYSICAL cores, no SMT. Measured here: 19.0 s with
     * 16 logical threads against 17.1 s with 8 physical ones, and on top of that
     * the I/O wait drops from 8.8 to 6.6 s because the readers stop fighting the
     * compute over the same cores. At 4 threads it loses on the other side.
     *
     * omp_tune.h deliberately takes ONLY the sizing and leaves the spin-wait out:
     * in an engine that pulls its tokens off the disk, a team spinning idle steals
     * cores from the I/O doing the real work. This engine is in that regime (I/O
     * is 40 % of the time). */
    coli_omp_tune_threads("deepseek_v4");

    /* The gateway launches the engine with SNAP=<dir>, SERVE=1 and
     * NGEN=<max_tokens>, and passes the cache `cap` as argv[1]. */
    const char *sv = getenv("SERVE");
    if (sv && sv[0] == '1') {
#ifdef _WIN32
        /* Without this the CRT's CRLF translation corrupts the sentinels and
         * stalls the byte-counted reads. */
        _setmode(_fileno(stdin),  _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        const char *snap = getenv("SNAP");
        if (!snap || !*snap) { fprintf(stderr, "SERVE=1 needs SNAP=<dir>\n"); return 1; }
        /* MinGW has no setenv, so the cap travels through a global instead. */
        if (argc > 1 && atoi(argv[1]) > 0) g_cache_per_layer = atoi(argv[1]);
        Model M;
        model_load(&M, snap);
        char tp[1024];
        snprintf(tp, sizeof tp, "%s/tokenizer.json", snap);
        tok_load(&M.tok, tp);
        Run R;
        run_init(&R, &M);
        serve_loop(&R);
        return 0;
    }

    const char *dir = (argc > 1) ? argv[1]
        : "C:\\Users\\Gus\\ai\\models\\DeepSeek-V4-Flash-0731";
    const char *prompt = (argc > 2) ? argv[2] : "hola";
    const int ngen = (argc > 3) ? atoi(argv[3]) : 8;

    Model M;
    model_load(&M, dir);

    char tokp[1024];
    snprintf(tokp, sizeof tokp, "%s/tokenizer.json", dir);
    tok_load(&M.tok, tokp);

    /* BOS up front: the checkpoint ships no chat template, but it does have
     * `<|begin_of_sentence|>` (id 0), and the model was trained seeing it. */
    int ids[1024];
    ids[0] = 0;
    int n = 1 + tok_encode(&M.tok, prompt, (int)strlen(prompt), ids + 1, 1000);
    printf("prompt: \"%s\" -> %d tokens (with BOS)\n", prompt, n);

    /* per-layer state */
    DsV4AttnState *st = calloc((size_t)M.n_layers, sizeof(DsV4AttnState));
    for (int L = 0; L < M.n_layers; L++)
        dsv4_state_init_full(&st[L], 1, M.cfg[L].attn.hd, M.cfg[L].attn.win,
                             M.max_seq, M.cfg[L].attn.ratio, M.cfg[L].attn.i_hd);

    const int dim = M.dim, hc = M.hc;
    float *h  = malloc((size_t)hc * dim * sizeof(float));
    float *h2 = malloc((size_t)hc * dim * sizeof(float));
    float *logits = malloc((size_t)M.vocab * sizeof(float));
    float *emb = malloc((size_t)dim * sizeof(float));

    printf("generating %d tokens...\n\n%s", ngen, prompt);
    const double tgen = now_s();
    int pos = 0;
    for (int step = 0; step < n + ngen; step++) {
        /* Always `ids[step]`: for step < n that is the prompt, and beyond it the
         * previous iteration wrote it. Feeding back `ids[step-1]` — which is what
         * this loop did at first — repeats the prompt's last token and derails the
         * generation from the very first step. */
        const int tokid = ids[step];
        g_tok_no = step;

        /* embedding -> hc copias */
        {
            const DsV4W row = dsv4_w_rows(&M.embed, tokid, 1);
            const uint16_t *p = (const uint16_t *)row.w;
            for (int d = 0; d < dim; d++) emb[d] = dsv4_bf16_to_f32(p[d]);
            for (int m = 0; m < hc; m++)
                memcpy(h + (size_t)m * dim, emb, (size_t)dim * sizeof(float));
        }

        for (int L = 0; L < M.n_layers; L++) {
            model_layer_decode(&M, L, &st[L], h, pos, (int32_t)tokid, h2);
            memcpy(h, h2, (size_t)hc * dim * sizeof(float));
        }
        pos++;

        if (step >= n - 1) {
            float y[8192], nz[8192];
            dsv4_hc_head(h, M.hc_head_fn, M.hc_head_scale, M.hc_head_base,
                         hc, dim, M.norm_eps, M.hc_eps, y);
            dsv4_rmsnorm(nz, y, M.final_norm, dim, M.norm_eps);
            for (int d = 0; d < dim; d++) nz[d] = dsv4_to_bf16(nz[d]);
            { const double _t = now_s();
              dsv4_matmul_w(logits, nz, &M.head, 1, 0);
              g_t_head += now_s() - _t; }
            int best = 0;
            for (int v = 1; v < M.vocab; v++) if (logits[v] > logits[best]) best = v;
            if (step < n + ngen - 1) ids[step + 1] = best;
            char piece[64];
            const int m = tok_decode(&M.tok, &best, 1, piece, sizeof piece - 1);
            piece[m] = 0;
            printf("%s", piece);
        }
    }
    const double dt = now_s() - tgen;
    printf("\n\n%d tokens in %.1f s (%.2f tok/s)\n", n + ngen, dt, (n + ngen) / dt);
    printf("profile (total): attention %.1f s | MoE %.1f s | head %.1f s | rest %.1f s\n",
           g_t_attn, g_t_moe, g_t_head, dt - g_t_attn - g_t_moe - g_t_head);
    printf("  I/O: %llu prefetch batches, %.1f experts each, %d readers\n"
           "       %.2f GB via prefetch, %.2f GB via the fallback path\n",
           (unsigned long long)g_pf_batches,
           g_pf_batches ? (double)g_pf_reads / g_pf_batches : 0.0, M.tier.nthreads,
           (double)(M.tier.bytes - g_fb_bytes) / 1e9, (double)g_fb_bytes / 1e9);
    printf("  of the MoE, %.1f s is expert I/O and %.1f s is compute\n",
           g_t_io, g_t_moe - g_t_io);
    printf("experts: %llu hits / %llu miss (%.0f%% hit rate), %.2f GB read\n",
           (unsigned long long)M.tier.hits, (unsigned long long)M.tier.miss,
           100.0 * M.tier.hits / (double)(M.tier.hits + M.tier.miss),
           (double)M.tier.bytes / 1e9);
    printf("  cache of %d slots (%.1f GB); %llu distinct experts in total\n"
           "  -> an infinite cache would miss %llu times instead of %llu\n",
           M.tier.cap, M.tier.cap * 13.4e6 / 1e9,
           (unsigned long long)g_distinct,
           (unsigned long long)g_distinct, (unsigned long long)M.tier.miss);
    return 0;
}
