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

/* ---------------------------------------------------------------------------
 * A HARD CAP on the working set, because otherwise the OS decides.
 *
 * Mapped pages accumulate in the process working set and Windows keeps them there
 * until something pushes back. Measured during a 19701-token build: the working set
 * reached 22.86 GB with 0.55 GB of the machine's 31.4 GB left available, and the
 * page file had grown ~20 GB over the night, filling the disk until reads began
 * failing with EIO.
 *
 * Capping it does not throw the data away -- it moves the excess to the STANDBY
 * list, where it is still in RAM serving re-reads without touching disk, but counts
 * as available for everything else. Measured, same run, seconds apart:
 *
 *     no cap : working set 22.86 GB, available  0.55 GB
 *     8 GB   : working set  8.00 GB, available 16.80 GB, standby 16.8 GB
 *
 * and the throughput difference was noise (254 MB/s against 325 and 230). Clean
 * file pages cost nothing to move, so the cap is close to free.
 *
 * This is the difference between the OS managing our memory and us managing it. An
 * engine whose whole purpose is to stream 137 GB through 31 GB of RAM cannot leave
 * the residency decision to a heuristic that does not know the access pattern.
 * ------------------------------------------------------------------------- */
#ifdef _WIN32
#ifndef QUOTA_LIMITS_HARDWS_MAX_ENABLE
#define QUOTA_LIMITS_HARDWS_MAX_ENABLE 0x00000004
#endif
static void ws_cap(void)
{
    const char *e = getenv("DSV4_WS_MAX_GB");
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return;
    const double ram = (double)ms.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
    /* A quarter of physical RAM by default: it leaves the file cache and the rest of
     * the machine the other three quarters, and on the box this was measured on it
     * lands at 7.9 GB -- the value that was actually validated. `0` disables it. */
    double gb = e ? atof(e) : ram / 4.0;
    if (gb <= 0.0) { if (e) fprintf(stderr, "[ws] cap disabled\n"); return; }
    if (!e) { if (gb < 2.0) gb = 2.0; if (gb > 16.0) gb = 16.0; }

    SIZE_T mx = (SIZE_T)(gb * 1024.0 * 1024.0 * 1024.0);
    SIZE_T mn = (SIZE_T)256 << 20;
    if (mn > mx / 2) mn = mx / 2;
    if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), mn, mx,
                                    QUOTA_LIMITS_HARDWS_MAX_ENABLE))
        fprintf(stderr, "[ws] could not cap the working set (error %lu); the OS keeps "
                        "the residency decision\n", (unsigned long)GetLastError());
    else
        fprintf(stderr, "[ws] working set capped at %.1f GB of %.1f GB of RAM\n",
                gb, ram);
}
#else
/* Linux has no equivalent hard cap; madvise/cgroups are the levers there. */
static void ws_cap(void) {}
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
 * DIRECT I/O for the expert reads.
 *
 * Buffered reads populate the SYSTEM FILE CACHE, and on a machine streaming 137 GB
 * that cache grows without bound: measured at 12.49 GB of 31.4 GB while our own LRU
 * held another 4.4 GB of the same kind of data. Two caches for the same bytes, and
 * only one of them is ours. Worse, that 12.5 GB is what squeezed the engine's own
 * anonymous memory out to the page file -- it grew ~20 GB over a night, filled the
 * disk, and reads eventually failed with EIO. We evicted ourselves.
 *
 * With FILE_FLAG_NO_BUFFERING the reads bypass that cache entirely, so the RAM the
 * engine occupies is the RAM we decided to give it: the LRU and nothing else.
 *
 * The price is alignment. Offset, length AND buffer must be multiples of the sector
 * size, and our tensors sit at arbitrary offsets. So each slot buffer is allocated
 * 4K-aligned with a page of slack, the read starts at the aligned offset BELOW the
 * tensor, and `data` points at the tensor inside it. Nothing is copied.
 *
 * compat_open_direct's contract is what makes this safe to attempt: a misaligned
 * request fails with -1, it never returns wrong bytes. A mistake here is loud.
 * ------------------------------------------------------------------------- */
#define DIO_ALIGN 4096
#define dio_down(x) ((x) & ~(int64_t)(DIO_ALIGN - 1))
#define dio_up(x)   (((x) + DIO_ALIGN - 1) & ~(int64_t)(DIO_ALIGN - 1))

static int dio_on(void) {
    const char *e = getenv("DSV4_DIRECT");
    return e && e[0] == '1';
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
    uint8_t *buf[6];        /* the ALLOCATION: 4K-aligned, with slack for direct I/O */
    uint8_t *data[6];       /* the tensor itself: buf + (off - align_down(off)) */
    int layer, expert;
    uint64_t used;
    int pending;            /* reads in flight; 0 = ready to use */
} ExpSlot;

/* One read job: one tensor of one expert into one slot. */
/* kind 0: one tensor of one expert into an LRU slot (decode).
 * kind 1: one PIECE of a whole layer's contiguous range into a layer buffer, where
 *         `slot` is the buffer and `k` the piece index. */
typedef struct { int slot, layer, expert, k, kind; } RdJob;

/* An expert tensor, already resolved: no name lookups on the hot path. */
typedef struct { int shard; int64_t off, nb; } ExpTensor;

typedef struct {
    shards *S;
    int n_layers, n_experts, cap, inter, dim;
    ExpTensor *tens;        /* [(layer*n_experts + e)*6 + k] */
    ExpSlot *slots;
    int nslot;
    /* `miss` counts expert READS ISSUED and is idempotent: once an expert is
     * reserved, tier_find sees it, so a second prefetch call cannot count it again.
     * `uses` counts how many times the MoE ASKS for an expert. Hit rate is derived
     * from the two.
     *
     * There used to be a `hits` counter incremented in tier_prefetch on every
     * tier_find success, which was fine while prefetch was called once per layer and
     * became nonsense when the sliding window started calling it once per expert in
     * the union: it reported 98.1 %, which is the rate at which the prefetch
     * re-recognizes what it just queued, not the rate at which reads are avoided. */
    uint64_t clock, uses, miss;
    uint64_t bytes;
    int mapped;             /* LOAD_ALL: no buffers, no LRU, read from the map */
    /* Unbuffered for EVERYTHING, deliberately.
     *
     * Letting the per-expert path stay buffered would buy the OS's readahead (360
     * MB/s against 216 on scattered ~2 MB reads), but it costs a second copy of the
     * same bytes in a cache we do not control -- 13 GB of physical RAM spent
     * duplicating what our own LRU is for. On a machine where RAM is the scarce
     * resource, that RAM belongs to the LRU, which knows the access pattern the OS has
     * to guess. Fewer reads beats faster reads when the reads are the same bytes. */
    int direct;             /* DSV4_DIRECT=1: unbuffered reads, our LRU is the cache */
    /* TWO-LAYER DOUBLE BUFFER.
     *
     * The router of layer L depends on layer L-1's output, so WHICH experts a layer
     * wants cannot be known in advance. The STRUCTURE can: prefill walks layers
     * 0..42 and the next chunk walks them again, and with 256 rows a layer's union is
     * huge -- measured 173 of 256 experts, 2.2 GB. So the previous chunk's union for
     * layer L+1 predicts the next one well enough to read it while layer L computes.
     * That is what the linearity buys: we cannot predict the router, and we do not
     * need to.
     *
     * Two layers is the natural depth rather than an approximation: compute L while
     * reading L+1. A third would only help if reads were more than twice as slow as
     * compute, and then the disk is the wall and getting further ahead buys nothing.
     * 2 x 173 experts is ~4.4 GB, which a 384-slot cache already holds.
     *
     * The LRU stays, and matters -- for DECODE. There, two uses of layer L are 258
     * expert reads apart and the cache captures the reuse; in chunked prefill they are
     * ~10,700 apart and it captures none (measured: 1.3 % hit). Same structure, two
     * regimes, and only one of them is about reuse. */
    /* WHOLE-LAYER STREAMING (DSV4_LAYER_STREAM=1).
     *
     * A layer's 256 experts are 1536 tensors that sit CONTIGUOUSLY in a single
     * shard: 3.42 GB of data inside a 3.55 GB range, one internal gap, 96 % dense.
     * That changes the arithmetic completely.
     *
     *   173 scattered experts (68 %)   2.2 GB at 216-360 MB/s  =  6-10 s
     *   the whole range, sequential    3.55 GB at 1514 MB/s    =  2.3 s
     *
     * Reading 61 % MORE bytes takes a THIRD of the time, because this drive gives 7x
     * more sequentially than scattered. So there is nothing to predict and nothing to
     * cache: bring the whole layer, always. The router's choice stops mattering, the
     * 32 % waste of a per-expert union disappears, and an expert index becomes an
     * offset inside the buffer.
     *
     * Two buffers, 3.55 GB each: compute layer L out of one while L+1 lands in the
     * other. */
    int lstream;            /* the whole-layer path is active */
    int lbuf_n;             /* 2 */
    uint8_t *lbuf[2];
    int lbuf_layer[2];      /* which layer each buffer holds, -1 = nothing */
    int lbuf_pend[2];       /* pieces still in flight */
    int64_t *lay_off, *lay_len;   /* [layer] the contiguous range */
    int64_t *lay_base;            /* [layer] lay_off rounded DOWN to 4K, for direct I/O:
                                   * the piece boundaries and the buffer must be
                                   * aligned, so the buffer holds [base, off+len) and
                                   * an expert sits at off - base. */
    int *lay_shard;
    int64_t lspan;          /* the widest layer, i.e. the buffer size */
    int lpieces;            /* how many parallel pieces per layer */

    int *pred, *pred_n;     /* [layer][n_experts]: the previous chunk's union */
    int *cur,  *cur_n;      /* [layer][n_experts]: the one being built now */

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
    /* The contiguous range of every layer's experts, from the offsets already
     * resolved above. Measured: 3.42 GB of tensors inside a 3.55 GB span, in one
     * shard, with a single internal gap. */
    T->lay_off   = (int64_t *)malloc((size_t)n_layers * sizeof(int64_t));
    T->lay_len   = (int64_t *)malloc((size_t)n_layers * sizeof(int64_t));
    T->lay_shard = (int *)malloc((size_t)n_layers * sizeof(int));
    T->lay_base  = (int64_t *)malloc((size_t)n_layers * sizeof(int64_t));
    T->lspan = 0;
    for (int L = 0; L < n_layers; L++) {
        int64_t lo = INT64_MAX, hi = 0;
        int sh = -1, multi = 0;
        for (int e = 0; e < n_experts; e++)
            for (int k = 0; k < 6; k++) {
                const ExpTensor *d = &T->tens[((size_t)L * n_experts + e) * 6 + k];
                if (sh < 0) sh = d->shard; else if (sh != d->shard) multi = 1;
                if (d->off < lo) lo = d->off;
                if (d->off + d->nb > hi) hi = d->off + d->nb;
            }
        T->lay_shard[L] = sh;
        T->lay_off[L] = lo;
        T->lay_base[L] = dio_down(lo);
        T->lay_len[L] = multi ? 0 : hi - T->lay_base[L];   /* 0 disables the fast path */
        if (T->lay_len[L] > T->lspan) T->lspan = T->lay_len[L];
    }

    {   const char *e = getenv("DSV4_LAYER_STREAM");
        T->lstream = (e && e[0] == '1' && T->lspan > 0);
        T->lbuf_n = 2;
        T->lpieces = T->nthreads > 0 ? T->nthreads : 8;
        if (T->lstream) {
            for (int b = 0; b < T->lbuf_n; b++) {
                void *m = NULL;
                if (posix_memalign(&m, DIO_ALIGN,
                                   (size_t)dio_up(T->lspan) + DIO_ALIGN) != 0) m = NULL;
                T->lbuf[b] = (uint8_t *)m;
                T->lbuf_layer[b] = -1;
                if (!T->lbuf[b]) {
                    fprintf(stderr, "[lstream] cannot allocate 2 x %.2f GB; falling "
                                    "back to per-expert reads\n", T->lspan / 1e9);
                    T->lstream = 0;
                }
            }
            if (T->lstream)
                fprintf(stderr, "[lstream] whole-layer reads: 2 buffers of %.2f GB, "
                                "%d pieces each\n", T->lspan / 1e9, T->lpieces);
        }
    }

    T->pred   = (int *)calloc((size_t)n_layers * n_experts, sizeof(int));
    T->cur    = (int *)calloc((size_t)n_layers * n_experts, sizeof(int));
    T->pred_n = (int *)calloc((size_t)n_layers, sizeof(int));
    T->cur_n  = (int *)calloc((size_t)n_layers, sizeof(int));

    T->nshard = S->nfd;
    T->direct = dio_on();
    T->fd = malloc((size_t)T->nthreads * T->nshard * sizeof(int));
    for (int t = 0; t < T->nthreads; t++)
        for (int i = 0; i < T->nshard; i++) {
            int fd = -1;
            if (T->direct) {
                fd = compat_open_direct(S->paths[i]);
                if (fd < 0 && t == 0 && i == 0) {
                    fprintf(stderr, "[dio] cannot open unbuffered; falling back to the "
                                    "file cache\n");
                    T->direct = 0;
                }
            }
            /* Thread 0 reuses the shard's own descriptor in the buffered case; a
             * direct descriptor is never shared with the rest of the loader, which
             * still reads dense weights through the cache. */
            if (fd < 0) fd = (t == 0) ? S->fds[i] : open(S->paths[i], COMPAT_O_RDONLY);
            T->fd[t * T->nshard + i] = fd;
        }
    if (T->direct)
        fprintf(stderr, "[dio] expert reads bypass the system file cache\n");

    tier_pool_start(T);
}

/* Queue the whole range of `layer` into a buffer, and return without waiting. */
static void layer_start(ExpertTier *T, int layer)
{
    if (!T->lstream || layer < 0 || layer >= T->n_layers) return;
    for (int b = 0; b < T->lbuf_n; b++)
        if (T->lbuf_layer[b] == layer) return;          /* already here or coming */
    /* Take the buffer that is not holding the layer being computed. */
    int b = -1;
    for (int i = 0; i < T->lbuf_n; i++)
        if (T->lbuf_pend[i] == 0 && T->lbuf_layer[i] != layer - 1) { b = i; break; }
    if (b < 0) return;                                   /* both busy: skip a beat */
    T->lbuf_layer[b] = layer;
    pthread_mutex_lock(&T->mu);
    T->lbuf_pend[b] = T->lpieces;
    for (int k = 0; k < T->lpieces; k++) {
        RdJob j = { b, layer, 0, k, 1 };
        T->q[T->qtail] = j;
        T->qtail = (T->qtail + 1) % T->qcap;
    }
    pthread_cond_broadcast(&T->cv_job);
    pthread_mutex_unlock(&T->mu);
}

/* Seconds spent BLOCKED waiting for a layer's pieces.
 *
 * The hit counter cannot see this: in streaming mode tier_get issues no read, so
 * it records a hit and then blocks in here for as long as the disk takes. A 99.8 %
 * hit rate is therefore compatible with waiting on I/O the entire time, which is
 * exactly the wrong thing for an instrument to hide. */
static double g_t_lwait = 0.0;

/* The buffer holding `layer`, once its reads are done; -1 if it is not queued. */
static int layer_wait(ExpertTier *T, int layer)
{
    for (int b = 0; b < T->lbuf_n; b++)
        if (T->lbuf_layer[b] == layer) {
            const double t0 = now_s();
            pthread_mutex_lock(&T->mu);
            while (T->lbuf_pend[b]) pthread_cond_wait(&T->cv_done, &T->mu);
            pthread_mutex_unlock(&T->mu);
            g_t_lwait += now_s() - t0;
            return b;
        }
    return -1;
}

/* A chunk is starting: what this chunk used becomes the next one's prediction. */
static void tier_chunk_begin(ExpertTier *T)
{
    if (!T->pred) return;
    for (int L = 0; L < T->n_layers; L++) {
        if (T->cur_n[L] > 0) {
            memcpy(T->pred + (size_t)L * T->n_experts,
                   T->cur  + (size_t)L * T->n_experts,
                   (size_t)T->cur_n[L] * sizeof(int));
            T->pred_n[L] = T->cur_n[L];
        }
        T->cur_n[L] = 0;
    }
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
         * into its buffers. What keeps the LRU from running out of candidates is
         * pf_window() capping the in-flight set at cap/4 -- it used to be that "at
         * most topk are in flight", which stopped being true when prefill started
         * prefetching a whole union. */
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
        if (!s->buf[k]) {
            /* One page of slack covers the worst-case distance from the tensor's
             * offset down to the aligned one, plus the tail rounded up. */
            const size_t cap = (size_t)dio_up(d[k].nb) + 2 * DIO_ALIGN;
            void *m = NULL;
            if (posix_memalign(&m, DIO_ALIGN, cap) != 0 || !m) {
                fprintf(stderr, "out of memory for an expert slot (%zu bytes)\n", cap);
                exit(1);
            }
            s->buf[k] = (uint8_t *)m;
        }
        /* The slack depends on THIS expert's offset, so it is recomputed on every
         * (re)use of the slot, not once at allocation. */
        s->data[k] = s->buf[k] + (T->direct ? (size_t)(d[k].off - dio_down(d[k].off)) : 0);
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
    /* Buffered: read the tensor where it is. Direct: start at the aligned offset
     * below it and read a whole number of blocks; `data` already points past the
     * slack. `need` is what has to arrive; `span` is what may be asked for, and the
     * two differ at the end of a shard, where the last block is short. */
    const int64_t lo   = T->direct ? dio_down(d->off) : d->off;
    const int64_t need = d->off + d->nb - lo;
    const int64_t span = T->direct ? dio_up(need) : need;
    int64_t done = 0;
    while (done < need) {
        const ssize_t got = pread(fd, out + done, (size_t)(span - done), lo + done);
        if (got <= 0) {
            fprintf(stderr, "short %s at layer %d expert %d tensor %d "
                            "(off %lld len %lld)\n", T->direct ? "direct read" : "pread",
                    layer, e, k, (long long)(lo + done), (long long)(span - done));
            exit(1);
        }
        done += got;
    }
}

/* One piece of a layer's range. Big sequential reads are the entire point, so the
 * range is split only as far as there are workers. */
static void tier_read_piece(ExpertTier *T, int buf, int layer, int piece, int tid)
{
    const int64_t len = T->lay_len[layer];
    /* 4K-aligned piece boundaries: required for direct I/O and harmless otherwise. */
    const int64_t per = dio_up((len + T->lpieces - 1) / T->lpieces);
    const int64_t a = per * piece;
    if (a >= len) return;
    int64_t nb = (len - a < per) ? len - a : per;
    if (T->direct) nb = dio_up(nb);            /* the tail may read past the range */
    const int fd = T->fd[(size_t)tid * T->nshard + T->lay_shard[layer]];
    uint8_t *out = T->lbuf[buf] + a;
    const int64_t need = (len - a < per) ? len - a : per;
    int64_t done = 0;
    while (done < need) {
        const ssize_t got = pread(fd, out + done, (size_t)(nb - done),
                                  T->lay_base[layer] + a + done);
        if (got <= 0) {
            fprintf(stderr, "short read on layer %d piece %d\n", layer, piece);
            exit(1);
        }
        done += got;
    }
    T->bytes += (uint64_t)need;
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

        if (j.kind == 1) tier_read_piece(T, j.slot, j.layer, j.k, tid);
        else             tier_read_one(T, j.slot, j.layer, j.expert, j.k, tid);

        pthread_mutex_lock(&T->mu);
        if (j.kind == 1) { if (--T->lbuf_pend[j.slot] == 0) pthread_cond_broadcast(&T->cv_done); }
        else if (--T->slots[j.slot].pending == 0) pthread_cond_broadcast(&T->cv_done);
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
/* How many experts may be in flight at once.
 *
 * This was a hard 16, which was right for decode -- topk is 6, so a layer's union
 * IS about 6 experts and 16 never truncated. Batched prefill broke that silently:
 * with 256 rows the union is ~250 experts, so 16 were prefetched in parallel and
 * the other ~234 were read one at a time, blocking, inside the union loop.
 * Measured while building a 19701-token cache: 193 MB/s from a drive that gives
 * 1514 MB/s to a single sequential reader, with the CPU at 34 %. Not the disk's
 * limit -- ours, for lack of queue depth.
 *
 * The bound is the CACHE, not a constant: tier_reserve cannot evict a slot with
 * reads in flight, so a window near `cap` would leave the LRU with no candidate
 * and abort. A quarter of the cache keeps that invariant with room to spare. */
#define DSV4_PF_MAX 256
static int pf_window(const ExpertTier *T) {
    const char *e = getenv("DSV4_PF_WINDOW");
    int w = e ? atoi(e) : 64;
    if (w < 1) w = 1;
    if (w > DSV4_PF_MAX) w = DSV4_PF_MAX;
    if (T->cap / 4 > 0 && w > T->cap / 4) w = T->cap / 4;
    if (w < 1) w = 1;
    return w;
}

static void tier_prefetch(ExpertTier *T, int layer, const int *es, int n)
{
    int slot[DSV4_PF_MAX], want[DSV4_PF_MAX], nw = 0;
    const int win = pf_window(T);
    if (n > win) n = win;
    if (T->mapped) {
        /* The OS is ASKED to bring the ranges in, without touching them:
         * touching them would cost the same bandwidth as reading them. It is a
         * hint, not a guarantee; if a page has not arrived, the fault resolves at
         * read time. */
#ifdef _WIN32
        WIN32_MEMORY_RANGE_ENTRY r[DSV4_PF_MAX * 6];
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
    if (T->lstream) return;        /* whole layers are streamed; nothing per-expert to do */
    if (g_trace) for (int k = 0; k < n; k++)
        fprintf(g_trace, "%d %d %d\n", g_tok_no, layer, es[k]);
    for (int k = 0; k < n; k++) {
        int dup = 0;
        for (int j = 0; j < nw; j++) if (want[j] == es[k]) { dup = 1; break; }
        if (dup) continue;
        const int f = tier_find(T, layer, es[k]);
        /* Found: nothing to do. Deliberately WITHOUT touching `used`.
         *
         * Recency belongs to real use (tier_get), not to "the prefetch looked at
         * you". Bumping it here inverted the policy once the prefetch window became
         * a sliding one: every call re-finds the experts furthest ahead and marks
         * them as the most recent, while the expert about to be consumed -- queued a
         * while ago and not re-found since -- looks like the oldest and gets evicted
         * first. The LRU was throwing away precisely the next thing it needed. */
        if (f >= 0) continue;
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

static void tier_prefetch_pred(ExpertTier *T, int layer)
{
    if (!T->pred || layer < 0 || layer >= T->n_layers) return;
    const int n = T->pred_n[layer];
    if (n <= 0) return;
    /* Half the cache per layer, so the two layers in flight cannot starve
     * tier_reserve of evictable slots. */
    int lim = T->cap / 2;
    if (lim < 1) lim = 1;
    tier_prefetch(T, layer, T->pred + (size_t)layer * T->n_experts,
                  n < lim ? n : lim);
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

    T->uses++;
    if (T->lstream) {
        /* Straight out of the layer's buffer: no lookup, no cache, no waiting -- the
         * range was read while the previous layer computed. */
        int b = layer_wait(T, layer);
        if (b < 0) { layer_start(T, layer); b = layer_wait(T, layer); }
        if (b >= 0) {
            const ExpTensor *d = &T->tens[((size_t)layer * T->n_experts + e) * 6];
            uint8_t *q[6];
            for (int k = 0; k < 6; k++)
                q[k] = T->lbuf[b] + (d[k].off - T->lay_base[layer]);
            *w1 = dsv4_w_mxfp4(q[0], q[1], T->inter, T->dim);
            *w2 = dsv4_w_mxfp4(q[2], q[3], T->dim,   T->inter);
            *w3 = dsv4_w_mxfp4(q[4], q[5], T->inter, T->dim);
            return;
        }
    }
    /* Record what this layer really used, which is the next chunk's prediction. */
    if (T->cur && T->cur_n[layer] < T->n_experts) {
        const int *row = T->cur + (size_t)layer * T->n_experts;
        int seen = 0;
        for (int i = 0; i < T->cur_n[layer]; i++) if (row[i] == e) { seen = 1; break; }
        if (!seen) T->cur[(size_t)layer * T->n_experts + T->cur_n[layer]++] = e;
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
    *w1 = dsv4_w_mxfp4(s->data[0], s->data[1], T->inter, T->dim);
    *w2 = dsv4_w_mxfp4(s->data[2], s->data[3], T->dim,   T->inter);
    *w3 = dsv4_w_mxfp4(s->data[4], s->data[5], T->inter, T->dim);
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
    float *freqs;           /* [max_seq, rd/2, 2] -- compressed layers */
    float *freqs_w;         /* idem, pure sliding-window layers (no YaRN) */
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

    /* --- freqs_cis with YaRN: TWO tables, chosen per layer ---------------
     *
     * The reference builds freqs_cis inside each Attention, and the parameters
     * depend on whether that layer compresses (model.py, Attention.__init__):
     *
     *   compress_ratio != 0 -> original_seq_len = 65536, base = 160000  (YaRN on)
     *   compress_ratio == 0 -> original_seq_len = 0,     base =  10000  (YaRN OFF,
     *                          "disable YaRN ... in pure sliding-window attention")
     *
     * and that same table serves the layer's window q/kv, its Compressor and its
     * Indexer. Using one table for all 43 layers -- which this engine did -- is
     * wrong twice over: the 38 compressed layers get theta 10000 instead of
     * 160000, and the 5 window-only layers get YaRN applied when it should be
     * off.
     *
     * Both errors GROW WITH POSITION, which is why short prompts looked fine and
     * a 300-token one produced word salad. It also explains why the harness never
     * caught it: check_real builds the table for the single layer under test with
     * that layer's own parameters, so it matched the reference while the engine's
     * shared shortcut did not. */
    /* max_seq sizes the freqs tables AND the compressed-KV region of every
     * layer's state (ncomp = max_seq / ratio). Running past it is not a graceful
     * truncation: `start_pos / ratio` walks off the end of st->kv and the freqs
     * lookup reads past the table. Both silently. A 5k-token system prompt --
     * ordinary for an agent -- would have corrupted the heap.
     *
     * Kept adjustable because it costs memory: the ratio-4 layers allocate
     * (win + max_seq/4) * hd floats each, so 32k of context is ~2.7 GB of state
     * across 43 layers. The bound is enforced in serve_one and in the CLI. */
    const char *msenv = getenv("DSV4_MAX_SEQ");
    M->max_seq = msenv ? atoi(msenv) : 8192;
    if (M->max_seq < 256) M->max_seq = 256;
    const size_t ftab = (size_t)M->max_seq * (rd / 2) * 2;
    M->freqs  = malloc(ftab * sizeof(float));   /* compressed layers */
    M->freqs_w = malloc(ftab * sizeof(float));  /* pure sliding window */
    jval *ctheta = json_get(C, "compress_rope_theta");
    const double cbase = ctheta ? ctheta->num : base;
    dsv4_precompute_freqs(M->freqs,   rd, M->max_seq, orig, cbase, factor, bf, bs);
    dsv4_precompute_freqs(M->freqs_w, rd, M->max_seq, 0,    base,  factor, bf, bs);

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
        w->attn.freqs = ratio ? M->freqs : M->freqs_w;

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

/* ---------------------------------------------------------------------------
 * PROMPT CACHE (a state snapshot on disk).
 *
 * A 20k-token system prompt is ~4.5 h of prefill on this machine, and the FLOP
 * roofline says nothing will make it cheap: 13B active parameters is ~26 GFLOP
 * per token against the 31-35 GFLOP/s these kernels reach. But it is the SAME
 * prefill every session, so the answer is to pay it once and keep the result.
 * llama.cpp keeps --prompt-cache for exactly this.
 *
 * What has to persist is precisely what the decode loop reads: the per-layer
 * attention state, the position, and the token history that prefix reuse compares
 * against. `h`/`h2` are scratch, rebuilt on every step.
 *
 * The arrays are trimmed to what has actually been written (see state_blobs), so
 * the size follows the TOKEN COUNT and not max_seq: a checkpoint is portable to a
 * larger context, and raising DSV4_MAX_SEQ to the model's 262144 neither multiplies
 * the file by 8 nor invalidates it. It reloads in 0.1 s, against hours of prefill.
 * ------------------------------------------------------------------------- */
#define SNAP_MAGIC "DSV4SNP1"
#define SNAP_MAX_BLOBS 6

typedef struct { void *p; size_t n; } Blob;

/* Compressed entries actually written after `ntok` tokens have been fed.
 *
 * dsv4_attention_decode writes entry `start_pos / ratio` (at kv offset
 * `win + that`, and at i_kv offset `that`) for start_pos in [0, ntok), so the
 * written entries are a PREFIX starting at a fixed offset and everything past it is
 * still the zero calloc left.
 *
 * `ntok` comes from Run.pos, NOT from st->n_written. That field is assigned only in
 * dsv4_state_seed_from_prefill and read nowhere, so on the per-token path -- the one
 * the offline builder uses -- it stays 0. Trusting it trimmed away every compressed
 * entry: the window alone still produced a plausible first word, and the answer
 * diverged one token later. Caught by comparing restored output against a cold run,
 * which is the only check that could have caught it. */
static int state_used(const DsV4AttnState *st, int ntok) {
    if (!st->ratio || ntok <= 0) return 0;
    return (ntok - 1) / st->ratio + 1;
}

/* The state's arrays listed in ONE place, so save and load cannot disagree about
 * the layout. The sizes mirror dsv4_state_init_full, except that the two arrays
 * dimensioned by max_seq are cut to the part that has been written.
 *
 * That is what makes a snapshot ~10x smaller AND portable across DSV4_MAX_SEQ: the
 * written prefix sits at the same offset whatever max_seq is -- only the array's
 * total length changes, and the tail is zero on both sides. Without it, raising
 * max_seq to the model's 262144 would multiply every checkpoint by 8 for no gain
 * and invalidate the cache for nothing.
 *
 * With b > 1 the written part is not a contiguous prefix, so the full arrays are
 * written. Both sides read st->b, so they agree either way. */
static int state_blobs(DsV4AttnState *st, int ntok, Blob *b)
{
    const int used  = (st->b == 1) ? state_used(st, ntok) : st->ncomp;
    const size_t ks = (size_t)st->b * st->coff * st->ratio * st->coff * st->hd;
    const size_t is = (size_t)st->b * st->coff * st->ratio * st->coff * st->i_hd;
    int n = 0;
    b[n].p = st->kv;
    b[n++].n = (size_t)st->b * (st->win + used) * st->hd * sizeof(float);
    if (st->ratio) {
        b[n].p = st->kv_state;          b[n++].n = ks * sizeof(float);
        b[n].p = st->score_state;       b[n++].n = ks * sizeof(float);
        if (st->i_hd) {
            b[n].p = st->i_kv;          b[n++].n = (size_t)st->b * used
                                                   * st->i_hd * sizeof(float);
            b[n].p = st->i_kv_state;    b[n++].n = is * sizeof(float);
            b[n].p = st->i_score_state; b[n++].n = is * sizeof(float);
        }
    }
    return n;
}

/* Loading a snapshot built for another MODEL would read the right number of bytes
 * into differently shaped arrays -- silently. max_seq is deliberately NOT here:
 * since the arrays are trimmed to the written prefix, a snapshot is valid at any
 * max_seq large enough to hold it, which is checked per layer instead. That is what
 * lets the context be raised later without paying the prefill again. */
static void snap_fingerprint(Model *M, int32_t *fp) {
    fp[0] = M->n_layers; fp[1] = M->dim; fp[2] = M->vocab;
    fp[3] = M->hc;       fp[4] = 0;
}

static int snap_save(Run *R, const char *path)
{
    Model *M = R->M;
    char part[1200];
    snprintf(part, sizeof part, "%s.part", path);
    FILE *f = fopen(part, "wb");
    if (!f) { fprintf(stderr, "[snap] cannot write %s\n", part); return 0; }

    int32_t fp[5];  snap_fingerprint(M, fp);
    const int32_t head[2] = { R->pos, R->nhist };
    int ok = fwrite(SNAP_MAGIC, 8, 1, f) == 1
          && fwrite(fp, sizeof fp, 1, f) == 1
          && fwrite(head, sizeof head, 1, f) == 1
          && (R->nhist == 0
              || fwrite(R->hist, sizeof(int) * (size_t)R->nhist, 1, f) == 1);
    for (int L = 0; ok && L < M->n_layers; L++) {
        DsV4AttnState *st = &R->st[L];
        const int32_t sc[8] = { st->b, st->hd, st->win, st->ratio,
                                st->coff, st->i_hd, R->pos,
                                state_used(st, R->pos) };
        ok = fwrite(sc, sizeof sc, 1, f) == 1;
        Blob b[SNAP_MAX_BLOBS];
        const int nb = state_blobs(st, R->pos, b);
        for (int i = 0; ok && i < nb; i++)
            if (b[i].n && fwrite(b[i].p, b[i].n, 1, f) != 1) ok = 0;
    }
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        remove(part);
        fprintf(stderr, "[snap] write failed; snapshot discarded rather than left "
                        "half-written\n");
        return 0;
    }
    /* Rename only once it is complete. A kill in the middle of a 1.7 GB write must
     * never leave behind a file that the next start would load as valid. */
    remove(path);
    if (rename(part, path) != 0) { remove(part); return 0; }
    return 1;
}

/* Returns how many tokens were restored, or 0 (state untouched, or reset). */
static int snap_load(Run *R, const char *path, const int *ids, int np)
{
    Model *M = R->M;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    char magic[8];
    int32_t fp[5], head[2], mine[5];
    snap_fingerprint(M, mine);
    if (fread(magic, 8, 1, f) != 1 || memcmp(magic, SNAP_MAGIC, 8) != 0
        || fread(fp, sizeof fp, 1, f) != 1 || memcmp(fp, mine, sizeof fp) != 0
        || fread(head, sizeof head, 1, f) != 1) {
        fprintf(stderr, "[snap] %s is not a snapshot for this model/DSV4_MAX_SEQ; "
                        "ignored\n", path);
        fclose(f); return 0;
    }
    const int pos = head[0], nh = head[1];
    /* nh >= np would leave no token to feed, which is the same condition the
     * in-memory prefix reuse enforces. */
    if (nh <= 0 || nh > R->hcap || nh >= np) { fclose(f); return 0; }

    /* PHASE 1: the history alone. A snapshot from a different conversation is
     * rejected for a few KB, instead of after reading 1.7 GB. */
    int *hist = (int *)malloc((size_t)nh * sizeof(int));
    if (!hist) { fclose(f); return 0; }
    if (fread(hist, sizeof(int) * (size_t)nh, 1, f) != 1
        || memcmp(hist, ids, sizeof(int) * (size_t)nh) != 0) {
        free(hist); fclose(f); return 0;
    }

    /* PHASE 2: the arrays. Past this point a failure has already half-written the
     * state, so it must be reset rather than used. */
    int ok = 1;
    for (int L = 0; ok && L < M->n_layers; L++) {
        DsV4AttnState *st = &R->st[L];
        int32_t sc[8];
        const int32_t want[6] = { st->b, st->hd, st->win,
                                  st->ratio, st->coff, st->i_hd };
        if (fread(sc, sizeof sc, 1, f) != 1 || memcmp(sc, want, sizeof want) != 0) {
            ok = 0; break;
        }
        /* The one thing max_seq still has to satisfy: room for the written part. */
        if (sc[7] > st->ncomp) {
            fprintf(stderr, "[snap] layer %d needs %d compressed entries, this "
                            "DSV4_MAX_SEQ gives %d\n", L, sc[7], st->ncomp);
            ok = 0; break;
        }
        Blob b[SNAP_MAX_BLOBS];
        const int nb = state_blobs(st, sc[6], b);
        for (int i = 0; ok && i < nb; i++)
            if (b[i].n && fread(b[i].p, b[i].n, 1, f) != 1) ok = 0;
        /* Leave the field saying what the seeding path would have left, even though
         * nothing reads it -- a stale 0 here is what caused the trimming bug. */
        st->n_written = sc[6];
    }
    fclose(f);
    if (!ok) {
        fprintf(stderr, "[snap] %s is truncated or inconsistent; starting cold\n", path);
        free(hist); run_reset(R); return 0;
    }
    memcpy(R->hist, hist, sizeof(int) * (size_t)nh);
    R->nhist = nh;
    R->pos = pos;
    free(hist);
    return nh;
}

/* ---------------------------------------------------------------------------
 * CHECKPOINTS: several snapshots along the prompt, not just one at the end.
 *
 * A single snapshot is all-or-nothing. Our state cannot be rewound -- the
 * compressors and the indexer accumulate an in-progress block that cannot be
 * un-accumulated -- so a prompt that diverges from the cached one at token 15000
 * cannot use a 20000-token snapshot AT ALL, and the whole prefill is paid again.
 *
 * With a checkpoint every DSV4_CKPT_EVERY tokens, the deepest one BEFORE the
 * divergence is used and only the tail is prefilled. llama.cpp added
 * --ctx-checkpoints for this, and for the same underlying reason: an SWA state
 * cannot be rolled back either.
 *
 * One file per checkpoint, named by its token count, so the directory is scanned
 * deepest-first and the arrays of a rejected candidate are never read.
 * ------------------------------------------------------------------------- */
#ifdef _WIN32
#include <direct.h>
#define ck_mkdir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define ck_mkdir(p) mkdir(p, 0755)
#endif
#include <dirent.h>

static void ck_path(char *buf, size_t n, const char *dir, int ntok) {
    snprintf(buf, n, "%s/ck_%08d.bin", dir, ntok);
}

/* The token counts present in `dir`, largest first. */
static int ck_list(const char *dir, int *out, int max)
{
    DIR *d = opendir(dir);
    if (!d) return 0;
    int n = 0;
    struct dirent *e;
    while (n < max && (e = readdir(d))) {
        int ntok = 0;
        if (sscanf(e->d_name, "ck_%d.bin", &ntok) != 1 || ntok <= 0) continue;
        int i = n++;
        while (i > 0 && out[i - 1] < ntok) { out[i] = out[i - 1]; i--; }
        out[i] = ntok;
    }
    closedir(d);
    return n;
}

/* The deepest checkpoint whose history is a prefix of this prompt. */
static int snap_load_best(Run *R, const char *dir, const int *ids, int np)
{
    int toks[512];
    const int n = ck_list(dir, toks, 512);
    for (int i = 0; i < n; i++) {
        if (toks[i] >= np) continue;       /* would leave no token to feed */
        char path[1200];
        ck_path(path, sizeof path, dir, toks[i]);
        const int got = snap_load(R, path, ids, np);
        if (got) return got;
    }
    return 0;
}

/* BOS, but only once -- and in ONE place.
 *
 * The gateway's chat template already emits <|begin_of_sentence|> itself, and it
 * survives tokenization as id 0 because tok.h treats added_tokens as atomic.
 * Prepending unconditionally produced [0, 0, ...], a double BOS the model never
 * saw in training. So the text is tokenized first and BOS added only if missing;
 * that stays correct whichever renderer the caller uses, and for a raw
 * /v1/completions prompt too.
 *
 * Shared by serve mode and the offline cache builder: a single token of difference
 * between them would make the snapshot's history fail to match, and the symptom
 * would be a silent cache miss after hours of work. */
static int encode_with_bos(Model *M, const char *text, int len, int *ids, int cap)
{
    int n = tok_encode(&M->tok, text, len, ids + 1, cap - 1);
    if (n < 1) return 0;
    if (ids[1] == M->bos) memmove(ids, ids + 1, (size_t)n * sizeof(int));
    else { ids[0] = M->bos; n++; }
    return n;
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
 * Chunked prefill.
 *
 * The prompt used to go through the decode path one token at a time, which is
 * what made a long prompt unusable: a 5k-token system prompt is ~100 minutes of
 * time-to-first-token. Two costs dominate and batching attacks both.
 *
 *   ATTENTION re-reads its 5.40 GB of resident weights FOR EVERY TOKEN, and at 18
 *   GB/s that is 0.30 s each. As a batch it is 5.40 GB per CHUNK: the projections
 *   become GEMMs whose arithmetic intensity rises with the chunk, so they stop
 *   being memory-bound.
 *
 *   EXPERT I/O drops because a chunk shares experts. Measured on the real routing
 *   trace, reads per token against the sequential LRU baseline of 124:
 *   chunk 16 -> 96, chunk 64 -> 52, chunk 101 -> 42. dsv4_moe_forward already
 *   walks the UNION of the chunk's experts, so this needs no new code -- only
 *   rows > 1.
 *
 * The state is what makes it correct: dsv4_state_seed_from_prefill rebuilds what
 * `n` sequential decode steps would have left, and check_real verifies that by
 * decoding past the seam (bit-identical on all three layer kinds, and the test is
 * known to fail when the seeding is broken).
 *
 * Returns the logits for the LAST position, which is the only one generation
 * needs.
 * ------------------------------------------------------------------------- */
/* Progress for a long offline build: the total, where THIS run started (so a resumed
 * checkpoint does not flatter the rate), and when it began. Zero unless the builder
 * sets them, so serve mode stays quiet. */
static int g_progress = 0, g_prog_total = 0, g_prog_from = 0;
static double g_prog_t0 = 0.0;
static double g_cpu_u0 = 0.0, g_cpu_k0 = 0.0;

/* Process CPU, user and kernel, ACCUMULATED since the process started.
 *
 * Sampling utilization over a 30-second window and comparing it against another
 * window from another run is not a measurement: it put an instantaneous 88 % next to
 * an instantaneous 37 % and a wall-clock average, and concluded the engine had become
 * compute-bound. Averages have to be compared with averages.
 *
 * The split is the diagnostic. If the extra CPU is KERNEL it is I/O overhead; if it
 * is USER the compute itself grew, which would be a bug and not a trade. */
static void cpu_times(double *user, double *krnl)
{
    *user = *krnl = 0.0;
#ifdef _WIN32
    FILETIME c, e, k, u;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
        ULARGE_INTEGER ku, uu;
        ku.LowPart = k.dwLowDateTime; ku.HighPart = k.dwHighDateTime;
        uu.LowPart = u.dwLowDateTime; uu.HighPart = u.dwHighDateTime;
        *krnl = (double)ku.QuadPart / 1e7;
        *user = (double)uu.QuadPart / 1e7;
    }
#endif
}

static const float *run_prefill(Run *R, const int *ids, int n, int pos0,
                                int batch_attn)
{
    Model *M = R->M;
    const int dim = M->dim, hc = M->hc;
    const size_t stride = (size_t)hc * dim;

    /* Batched attention numbers positions from 0, so it cannot continue an existing
     * KV. The caller only asks for it at pos0 == 0; this is belt and braces, and it
     * has to come before the chunk is sized. */
    if (batch_attn && pos0 != 0) batch_attn = 0;

    /* The chunk bounds MEMORY -- every buffer below is sized by it, at ~416 KB per
     * token -- and with whole-layer streaming it also sets the I/O per token, which
     * is what makes 1024 the default rather than 256.
     *
     * A layer's whole range is read once per chunk, so the bytes per token are
     * 43 x 3.55 GB / chunk:
     *
     *      chunk    GB/token    against the per-expert union (0.37 GB/token)
     *         64      2.38           6.4x worse
     *        256      0.60           1.6x worse
     *        512      0.30           0.8x
     *       1024      0.15           0.4x
     *
     * The crossover is around 384: past it, reading every layer whole moves FEWER
     * bytes than fetching the 68 % the router actually asked for, and does it
     * sequentially. The MoE batches better too, since dsv4_expert_apply_rows gets
     * 1024 rows per expert instead of 256 to amortize the MXFP4 decode over.
     *
     * The price is ~426 MB of buffers at 1024 tokens, against ~107 MB at 256. On a
     * machine with 26 GB of physical RAM actually available that is not a constraint;
     * it was chosen when the budget was believed to be 8 GB. */
    const char *ckenv = getenv("DSV4_CHUNK");
    int chunk = ckenv ? atoi(ckenv) : 1024;
    if (chunk < 16) chunk = 16;
    if (batch_attn || n < chunk) chunk = n;
    float *h    = malloc((size_t)chunk * stride * sizeof(float));
    float *mid  = malloc((size_t)chunk * stride * sizeof(float));
    float *coll = malloc((size_t)chunk * dim * sizeof(float));
    float *sub  = malloc((size_t)chunk * dim * sizeof(float));
    float *post = malloc((size_t)chunk * hc * sizeof(float));
    float *comb = malloc((size_t)chunk * hc * hc * sizeof(float));
    float *tmp  = malloc((size_t)chunk * dim * sizeof(float));  /* per token: one shared buffer would race */
    int32_t *cid = malloc((size_t)chunk * sizeof(int32_t));

    /* Capture buffers, sized for the widest chunk. Reused across layers and
     * chunks: one allocation, not 43 per chunk. */
    const int coff2 = 2;                       /* overlap layers need two halves */
    const int hd = M->cfg[0].attn.hd;
    int i_hd_max = 0, ratio_min = 0;
    for (int L = 0; L < M->n_layers; L++) {
        if (M->cfg[L].attn.i_hd > i_hd_max) i_hd_max = M->cfg[L].attn.i_hd;
        if (M->cfg[L].attn.ratio && (!ratio_min || M->cfg[L].attn.ratio < ratio_min))
            ratio_min = M->cfg[L].attn.ratio;
    }
    if (!ratio_min) ratio_min = 1;
    DsV4Capture cap;
    memset(&cap, 0, sizeof cap);
    cap.kv      = malloc((size_t)chunk * hd * sizeof(float));
    cap.ckv     = malloc((size_t)chunk * coff2 * hd * sizeof(float));
    cap.csc     = malloc((size_t)chunk * coff2 * hd * sizeof(float));
    cap.kv_comp = malloc((size_t)(chunk / ratio_min + 2) * hd * sizeof(float));
    if (i_hd_max) {
        cap.ikv       = malloc((size_t)chunk * coff2 * i_hd_max * sizeof(float));
        cap.isc       = malloc((size_t)chunk * coff2 * i_hd_max * sizeof(float));
        cap.i_kv_comp = malloc((size_t)(chunk / ratio_min + 2) * i_hd_max * sizeof(float));
    }
    int *tk = malloc((size_t)chunk * M->cfg[0].attn.win * sizeof(int));
    int lastrow = 0;                  /* row of `h` holding the last token */

    /* Why batched attention cannot be chunked: dsv4_attention_prefill derives each
     * token's RoPE position as `r % s`, so it assumes the batch starts at 0. A
     * second chunk would need the position offset, the window indices reaching
     * back into the previous chunk's KV, and the already-compressed blocks
     * attended -- prefill-with-existing-cache, still to do. The per-token path
     * below has none of those constraints, which is why it is the one that lifts
     * the length limit. */
    const int step_n = batch_attn ? n : chunk;
    for (int base = 0; base < n; base += step_n) {
        const int cs = (base + step_n <= n) ? step_n : n - base;

        /* This chunk's uses become the next chunk's prediction. */
        tier_chunk_begin(&M->tier);
        /* Layer 0 has nobody ahead of it to have queued its experts. */
        if (M->tier.lstream) layer_start(&M->tier, 0);
        else                 tier_prefetch_pred(&M->tier, 0);

        /* embedding -> hc copies, for the whole chunk */
        /* Independent per token, so it is parallel. R->emb is gone from here: one
         * shared scratch buffer across threads is a race, and the first hc copy is a
         * perfectly good place to decode into. */
#ifdef _OPENMP
#       pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < cs; t++) {
            cid[t] = (int32_t)ids[base + t];
            const DsV4W row = dsv4_w_rows(&M->embed, ids[base + t], 1);
            const uint16_t *pw = (const uint16_t *)row.w;
            float *dst = h + (size_t)t * stride;
            for (int d = 0; d < dim; d++) dst[d] = dsv4_bf16_to_f32(pw[d]);
            for (int m = 1; m < hc; m++)
                memcpy(dst + (size_t)m * dim, dst, (size_t)dim * sizeof(float));
        }

        for (int L = 0; L < M->n_layers; L++) {
            DsV4BlockCfg *c = &M->cfg[L];
            DsV4BlockW *w = &M->w[L];

            /* Queue the NEXT layer's predicted experts before doing any of this
             * layer's work, so those reads run under this layer's attention and MoE
             * instead of being waited for. Layer L's own experts were queued here one
             * iteration ago; the MoE's own prefetch call, with the router's real
             * union, is left as the correction pass for whatever the prediction
             * missed. */
            /* Whole-layer streaming reads L+1's entire contiguous range while
             * this layer computes; the per-expert prediction is the fallback. */
            if (M->tier.lstream) layer_start(&M->tier, L + 1);
            else                 tier_prefetch_pred(&M->tier, L + 1);

            /* --- attention, batched --------------------------------------- */
            /* Per-token and independent: mHC, the norm and the bf16 rounding touch
             * only row t. Measured before this: 28 % of eight cores with 9 % blocked
             * on I/O, so ~63 % of the wall clock was these loops running on one
             * thread while the other seven idled. */
#ifdef _OPENMP
#           pragma omp parallel for schedule(static)
#endif
            for (int t = 0; t < cs; t++) {
                dsv4_hc_pre(h + (size_t)t * stride, w->hc_attn_fn, w->hc_attn_scale,
                            w->hc_attn_base, hc, dim, c->sinkhorn_iters, c->hc_eps,
                            c->norm_eps, coll + (size_t)t * dim,
                            post + (size_t)t * hc, comb + (size_t)t * hc * hc);
                dsv4_rmsnorm(tmp + (size_t)t * dim, coll + (size_t)t * dim,
                             w->attn_norm, dim, c->norm_eps);
                for (int d = 0; d < dim; d++)
                    coll[(size_t)t * dim + d] = dsv4_to_bf16(tmp[(size_t)t * dim + d]);
            }
            { const double _t = now_s();
              if (batch_attn) {
                  const int ntopk = dsv4_window_topk_prefill(tk, 1, cs, c->attn.win);
                  dsv4_attention_prefill_cap(&c->attn, &w->attn, coll, tk, 1, cs,
                                             ntopk, 0, NULL, 0, 0, sub, &cap);
                  /* Seed so the tokens after the prompt continue from exactly the
                   * state token-at-a-time decoding would have produced. */
                  dsv4_state_seed_from_prefill(&R->st[L], &c->attn, &w->attn, &cap, cs);
              } else {
                  /* Per-token attention keeps the state itself, which is what lets
                   * a chunk start anywhere. Only the MoE below is batched. */
                  for (int t = 0; t < cs; t++)
                      dsv4_attention_decode(&c->attn, &w->attn, &R->st[L],
                                            coll + (size_t)t * dim, pos0 + base + t,
                                            sub + (size_t)t * dim);
              }
              g_t_attn += now_s() - _t; }
#ifdef _OPENMP
#           pragma omp parallel for schedule(static)
#endif
            for (int t = 0; t < cs; t++)
                dsv4_hc_expand(mid + (size_t)t * stride, sub + (size_t)t * dim,
                               post + (size_t)t * hc, comb + (size_t)t * hc * hc,
                               h + (size_t)t * stride, hc, dim);

            /* --- MoE, batched: one call, so the expert union applies ------- */
#ifdef _OPENMP
#           pragma omp parallel for schedule(static)
#endif
            for (int t = 0; t < cs; t++) {
                dsv4_hc_pre(mid + (size_t)t * stride, w->hc_ffn_fn, w->hc_ffn_scale,
                            w->hc_ffn_base, hc, dim, c->sinkhorn_iters, c->hc_eps,
                            c->norm_eps, coll + (size_t)t * dim,
                            post + (size_t)t * hc, comb + (size_t)t * hc * hc);
                dsv4_rmsnorm(tmp + (size_t)t * dim, coll + (size_t)t * dim,
                             w->ffn_norm, dim, c->norm_eps);
                for (int d = 0; d < dim; d++)
                    coll[(size_t)t * dim + d] = dsv4_to_bf16(tmp[(size_t)t * dim + d]);
            }
            { const double _t = now_s();
              dsv4_moe_forward(&c->moe, &w->moe, coll, cid, cs, sub);
              g_t_moe += now_s() - _t; }
#ifdef _OPENMP
#           pragma omp parallel for schedule(static)
#endif
            for (int t = 0; t < cs; t++)
                dsv4_hc_expand(h + (size_t)t * stride, sub + (size_t)t * dim,
                               post + (size_t)t * hc, comb + (size_t)t * hc * hc,
                               mid + (size_t)t * stride, hc, dim);
        }

        /* The state now covers base+cs tokens; record them so prefix reuse and
         * the next chunk agree with the decode path. */
        for (int t = 0; t < cs; t++)
            if (R->hist && R->nhist < R->hcap) R->hist[R->nhist++] = ids[base + t];
        R->pos = pos0 + base + cs;
        lastrow = cs - 1;

        /* Per-CHUNK progress, not per checkpoint.
         *
         * A 19700-token build checkpoints every 4096 tokens: four lines in seven
         * hours, which leaves "how far along is it" to be guessed from I/O counters
         * and a carried-over bytes-per-token constant. The chunk loop is the only
         * place that knows the exact position, so it is the only place that can
         * answer it. */
        if (g_progress) {
            const int done = pos0 + base + cs;
            const double el = now_s() - g_prog_t0;
            const double per = (done > g_prog_from) ? el / (done - g_prog_from) : 0.0;
            /* The LRU's own numbers go here too. Claiming a cache is "172 slots
             * against 301" and comparing throughput is worthless if the slots are
             * not actually filling, and private memory did not move between those
             * two settings -- which is exactly the kind of thing that has to be
             * read off the cache, not inferred from the process's memory. */
            const ExpertTier *T = &R->M->tier;
            double cu, ck; cpu_times(&cu, &ck);
            const double ncore = (double)(omp_get_max_threads() > 0
                                          ? omp_get_max_threads() : 8);
            fprintf(stderr, "[prog] %d/%d  %.1f%%  %.2f s/token  %.0f min elapsed  "
                            "eta %.1f h  | cpu %.0f%% (usr %.0f sys %.0f)"
                            "  | wait %.0f%%  | lru %d/%d slots  %.1f%% hit  %.0f GB read\n",
                    done, g_prog_total,
                    g_prog_total ? 100.0 * done / g_prog_total : 0.0,
                    per, el / 60.0,
                    (g_prog_total > done) ? (g_prog_total - done) * per / 3600.0 : 0.0,
                    el > 0 ? 100.0 * ((cu - g_cpu_u0) + (ck - g_cpu_k0)) / (el * ncore) : 0.0,
                    el > 0 ? 100.0 * (cu - g_cpu_u0) / (el * ncore) : 0.0,
                    el > 0 ? 100.0 * (ck - g_cpu_k0) / (el * ncore) : 0.0,
                    /* BOTH waits, whichever path this configuration uses.
                     *
                     * This reported only g_t_lwait, which exists solely in the
                     * whole-layer path, so a per-expert run showed `wait 0%` because
                     * the counter was never touched -- and that 0 was then cited as
                     * evidence that I/O was not the bottleneck, for a configuration
                     * whose waiting all happens in tier_wait (g_t_io). An instrument
                     * that reads zero because it is watching the wrong branch is worse
                     * than none. */
                    el > 0 ? 100.0 * (g_t_lwait + g_t_io) / el : 0.0,
                    T->nslot, T->cap,
                    T->uses ? 100.0 * (1.0 - (double)T->miss / (double)T->uses) : 0.0,
                    (double)T->bytes / 1e9);
        }
    }

    /* Logits for the last position only.
     *
     * `h` holds the CURRENT chunk, not the whole prompt, so this has to index the
     * last row of the last chunk -- `n - 1` was a read past the end of the buffer
     * for every prompt that took more than one chunk, and the symptom was fluent
     * garbage rather than a crash. */
    {
        float y[8192], nz[8192];
        dsv4_hc_head(h + (size_t)lastrow * stride, M->hc_head_fn, M->hc_head_scale,
                     M->hc_head_base, hc, dim, M->norm_eps, M->hc_eps, y);
        dsv4_rmsnorm(nz, y, M->final_norm, dim, M->norm_eps);
        for (int d = 0; d < dim; d++) nz[d] = dsv4_to_bf16(nz[d]);
        const double _t = now_s();
        dsv4_matmul_w(R->logits, nz, &M->head, 1, 0);
        g_t_head += now_s() - _t;
    }

    free(h); free(mid); free(coll); free(sub); free(post); free(comb);
    free(tmp); free(cid); free(tk);
    free(cap.kv); free(cap.ckv); free(cap.csc); free(cap.kv_comp);
    free(cap.ikv); free(cap.isc); free(cap.i_kv_comp);
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
    const int np = encode_with_bos(M, q->payload, q->plen, ids, cap);
    if (np < 1) {
        printf("ERROR %s EMPTY_PROMPT\n", q->id); fflush(stdout); free(ids); return;
    }

    /* CAPTURE THE PROMPT, once per process.
     *
     * The agent's opening prompt is what the cache has to be built for, and this
     * is the only place where the exact bytes the tokenizer sees are available --
     * dumping it from the gateway would risk a renderer difference, and a single
     * token of difference makes the snapshot miss. */
    const char *dumpf = getenv("DSV4_DUMP_PROMPT");
    static int dumped = 0;
    if (dumpf && !dumped) {
        dumped = 1;
        FILE *d = fopen(dumpf, "wb");
        if (d) {
            fwrite(q->payload, 1, (size_t)q->plen, d);
            fclose(d);
            fprintf(stderr, "[snap] prompt captured: %d bytes, %d tokens -> %s\n",
                    q->plen, np, dumpf);
        }
    }

    /* Refuse rather than corrupt. Past max_seq the compressed-KV index walks off
     * the end of st->kv and the freqs lookup reads past its table -- both
     * silently, which is the worst possible failure for a long system prompt.
     *
     * CONTEXT_EXCEEDED, not BAD_REQUEST: the gateway maps the former to a 400 that
     * names the limit and the length, so a client that knows how to compact a
     * conversation gets the chance to. BAD_REQUEST became an opaque 500
     * ("engine failed to process the request") that a pi agent retried ten times. */
    if (np + q->max_tok > M->max_seq) {
        fprintf(stderr, "[serve] prompt %d + %d tokens exceeds max_seq %d "
                        "(raise DSV4_MAX_SEQ)\n", np, q->max_tok, M->max_seq);
        printf("ERROR %s CONTEXT_EXCEEDED %d %d\n", q->id, np + q->max_tok, M->max_seq);
        fflush(stdout); free(ids); return;
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
    /* PROMPT CACHE, tried only when there is nothing better in memory: the live
     * state is always at least as good as a file, and reading 1.7 GB to replace it
     * would be a pure loss. */
    const char *snapd = getenv("DSV4_CACHE_DIR");
    if (snapd && R->nhist == 0) {
        const double t_l = now_s();
        const int got = snap_load_best(R, snapd, ids, np);
        if (got) fprintf(stderr, "[snap] restored %d of %d tokens in %.1f s\n",
                         got, np, now_s() - t_l);
    }

    int reuse = 0;
    while (reuse < R->nhist && reuse < np && R->hist[reuse] == ids[reuse]) reuse++;
    if (reuse < R->nhist || reuse >= np) { run_reset(R); reuse = 0; }
    if (reuse) fprintf(stderr, "[serve] prefix reused: %d of %d tokens\n", reuse, np);

    const uint64_t hit0 = M->tier.uses, miss0 = M->tier.miss;
    const double t0 = now_s();

    int gen = 0, limited = 1, cancelled = 0;
    int tokid = ids[reuse];
    double tdec = 0.0;

    /* BATCHED PREFILL for the prompt, when it is allowed.
     *
     * Only from a fresh state (reuse == 0): dsv4_attention_prefill numbers
     * positions from 0, so it cannot continue an existing KV. And only up to
     * DSV4_PREFILL_MAX, because the batch costs ~416 KB per token.
     *
     * Outside those bounds the old token-at-a-time path still runs, so nothing
     * regresses -- it is just slower. */
    int step = reuse;
    const float *lo = NULL;
    /* Two batched paths, and NO length limit any more.
     *
     *   one batch, batched attention   fastest (1.51x on 140 tokens), but prefill
     *                                  numbers positions from 0, so it needs a
     *                                  fresh state, and the batch costs ~416 KB
     *                                  per token -- hence the cap.
     *   chunked, per-token attention   works from any position and at any length,
     *                                  and still batches the MoE, which is 59 % of
     *                                  the time.
     *
     * A long prompt, or one following a reused prefix, no longer falls all the way
     * back to one token at a time. */
    const char *pfenv = getenv("DSV4_PREFILL_MAX");
    const int pfmax = pfenv ? atoi(pfenv) : 512;
    if (np - reuse > 1) {
        const int one_batch = (reuse == 0 && np <= pfmax);
        const double t_pf = now_s();
        lo = run_prefill(R, ids + reuse, np - reuse, reuse, one_batch);
        const double dt_pf = now_s() - t_pf;
        fprintf(stderr, "[serve] prefill (%s): %d tokens in %.1f s (%.3f s/token)\n",
                one_batch ? "batched attn" : "chunked", np - reuse, dt_pf,
                dt_pf / (np - reuse));
        step = np - 1;                 /* the prompt is done; lo is its last logits */

        /* Save straight after the prefill, not at the end of the request: this
         * exists to protect hours of work, and a kill during generation must not
         * throw it away. Only after an EXPENSIVE prefill, so the short tail of a
         * follow-up turn does not rewrite 1.7 GB every turn. */
        const char *smenv = getenv("DSV4_SNAP_MIN");
        const int smin = smenv ? atoi(smenv) : 512;
        if (snapd && np - reuse >= smin) {
            char path[1200];
            ck_mkdir(snapd);
            ck_path(path, sizeof path, snapd, R->nhist);
            const double t_s = now_s();
            if (snap_save(R, path))
                fprintf(stderr, "[snap] checkpoint at %d tokens saved in %.1f s\n",
                        R->nhist, now_s() - t_s);
        }
    }

    for (; ; step++) {
        if (!lo) lo = run_step(R, tokid);
        if (step + 1 < np) { tokid = ids[step + 1]; lo = NULL; continue; }  /* still prefill */
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
        lo = NULL;                     /* next position needs its own forward */

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
    const uint64_t uses = M->tier.uses - hit0, miss = M->tier.miss - miss0;
    const uint64_t hits = uses > miss ? uses - miss : 0;
    const uint64_t tot = uses;
    fprintf(stderr, "[prof] attn %.1f s | moe %.1f s (io %.1f) | head %.1f s | %.1f GB read\n",
            g_t_attn, g_t_moe, g_t_io, g_t_head, (double)M->tier.bytes / 1e9);
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
    /* stderr too, and this one is not cosmetic. Redirected to a file, stderr is
     * BLOCK buffered, so progress does not appear until 4 KB accumulate or the
     * process exits: BUILD_CACHE ran 65 minutes with a 362-byte log, and the
     * `[build] checkpoint ...` lines a watcher was waiting for were sitting in the
     * buffer. For a mode whose whole purpose is to run for hours unattended, a log
     * that only shows up at the end is a defect, not an inconvenience. */
    setvbuf(stderr, NULL, _IONBF, 0);
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
    /* The cache size, BEFORE the mode branches.
     *
     * This lived inside the SERVE branch, so BUILD_CACHE ignored argv[1] entirely: a
     * night of "LRU 172 against 301" comparisons all ran with the default 384, which
     * is why private memory never moved between them and why the comparisons were
     * meaningless. DSV4_CACHE (total slots) was the only knob that worked. */
    if (argc > 1 && atoi(argv[1]) > 0) g_cache_per_layer = atoi(argv[1]);

    coli_omp_tune_threads("deepseek_v4");
    ws_cap();                      /* before anything is mapped or read */

    /* The gateway launches the engine with SNAP=<dir>, SERVE=1 and
     * NGEN=<max_tokens>, and passes the cache `cap` as argv[1]. */
    /* OFFLINE CACHE BUILD.
     *
     *   BUILD_CACHE=1 SNAP=<model dir> DSV4_PROMPT_FILE=<captured prompt>
     *   DSV4_STATE_CACHE=<snapshot> deepseek_v4
     *
     * Prefill the prompt, write the snapshot, exit. No client, so nothing times out
     * while hours of prefill run, and the result is a file the server picks up on
     * its first request. That is the whole point: the cost is paid once, detached
     * from any request. */
    const char *bc = getenv("BUILD_CACHE");
    if (bc && bc[0] == '1') {
        const char *snapdir = getenv("SNAP");
        const char *pf = getenv("DSV4_PROMPT_FILE");
        const char *out = getenv("DSV4_CACHE_DIR");
        if (!snapdir || !*snapdir || !pf || !out) {
            fprintf(stderr, "BUILD_CACHE=1 needs SNAP=<dir>, DSV4_PROMPT_FILE and "
                            "DSV4_CACHE_DIR\n");
            return 1;
        }
        ck_mkdir(out);
        FILE *pfh = fopen(pf, "rb");
        if (!pfh) { fprintf(stderr, "cannot read %s\n", pf); return 1; }
        fseek(pfh, 0, SEEK_END);
        const long plen = ftell(pfh);
        fseek(pfh, 0, SEEK_SET);
        char *ptext = (char *)malloc((size_t)plen + 1);
        if (!ptext || fread(ptext, 1, (size_t)plen, pfh) != (size_t)plen) {
            fprintf(stderr, "cannot read %s\n", pf); return 1;
        }
        ptext[plen] = 0;
        fclose(pfh);

        Model M;
        model_load(&M, snapdir);
        char tp[1024];
        snprintf(tp, sizeof tp, "%s/tokenizer.json", snapdir);
        tok_load(&M.tok, tp);
        Run R;
        run_init(&R, &M);

        const int cap = 65536;
        int *ids = (int *)malloc((size_t)cap * sizeof(int));
        const int np = encode_with_bos(&M, ptext, (int)plen, ids, cap);
        if (np < 1) { fprintf(stderr, "empty prompt\n"); return 1; }
        if (np > M.max_seq) {
            fprintf(stderr, "prompt is %d tokens, DSV4_MAX_SEQ is %d\n", np, M.max_seq);
            return 1;
        }
        /* ALL BUT THE LAST TOKEN, deliberately.
         *
         * A snapshot covering the whole prompt is unusable for that same prompt:
         * generating needs the logits of the last position, and those only exist
         * after feeding a token. snap_load therefore requires nh < np, the same
         * condition the in-memory prefix reuse enforces. Stopping one token short
         * leaves exactly that token for run_step, so replaying the captured prompt
         * verbatim is a hit -- and a longer follow-up prompt is one too. */
        if (np < 2) { fprintf(stderr, "prompt too short to cache\n"); return 1; }
        const int nbuild = np - 1;

        /* RESUME. Hours of prefill are exactly the thing that must not have to be
         * repeated because the build was interrupted -- or because the engine was
         * rebuilt to make it faster. snap_load_best already picks the deepest
         * checkpoint whose history is a prefix, which is precisely this. */
        int done0 = snap_load_best(&R, out, ids, np);
        if (done0 > 0)
            fprintf(stderr, "[build] resuming from checkpoint at %d tokens\n", done0);

        /* Prefill in slices, saving a checkpoint after each. run_prefill already
         * takes an absolute start position and maintains the state itself, so a
         * slice is the same work as the equivalent stretch of one long call --
         * which is what makes checkpointing free here rather than a rewrite. */
        const char *ckev = getenv("DSV4_CKPT_EVERY");
        int every = ckev ? atoi(ckev) : 4096;
        if (every < 64) every = 64;
        fprintf(stderr, "[build] %ld bytes -> %d tokens; prefilling %d, checkpoint "
                        "every %d\n", plen, np, nbuild, every);
        const double t0 = now_s();
        g_progress = 1; g_prog_total = nbuild; g_prog_from = done0; g_prog_t0 = t0;
        cpu_times(&g_cpu_u0, &g_cpu_k0);   /* so model load is out of the average */
        for (int done = done0; done < nbuild; ) {
            const int take = (nbuild - done < every) ? nbuild - done : every;
            run_prefill(&R, ids + done, take, done, 0);
            done += take;
            char path[1200];
            ck_path(path, sizeof path, out, R.nhist);
            if (!snap_save(&R, path)) return 1;
            fprintf(stderr, "[build] checkpoint %d/%d tokens (%.1f s elapsed)\n",
                    R.nhist, nbuild, now_s() - t0);
        }
        const double dt = now_s() - t0;
        const int did = nbuild - done0;
        fprintf(stderr, "[build] done: %d tokens in %.1f s (%.3f s/token)\n",
                did > 0 ? did : nbuild, dt, dt / (did > 0 ? did : nbuild));
        return 0;
    }

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
            /* Stop at end-of-turn, like the serve path does. Without this the CLI
             * ran past it and printed the marker as literal text, then kept
             * rambling: "Respuesta: Berlin.<|end_of_sentence|>: el arte
             * renacentista". Easy to read as the model degenerating when it had
             * actually finished. */
            if (best == M.eos || best == 128805 /* <|EOT|> */) break;
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
    printf("experts: %llu uses / %llu reads (%.0f%% avoided), %.2f GB read\n",
           (unsigned long long)M.tier.uses, (unsigned long long)M.tier.miss,
           M.tier.uses ? 100.0 * (1.0 - M.tier.miss / (double)M.tier.uses) : 0.0,
           (double)M.tier.bytes / 1e9);
    printf("  cache of %d slots (%.1f GB); %llu distinct experts in total\n"
           "  -> an infinite cache would miss %llu times instead of %llu\n",
           M.tier.cap, M.tier.cap * 13.4e6 / 1e9,
           (unsigned long long)g_distinct,
           (unsigned long long)g_distinct, (unsigned long long)M.tier.miss);
    return 0;
}
