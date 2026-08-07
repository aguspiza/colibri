/* deepseek_v4.c — el motor: carga el checkpoint de 284B y genera texto.
 *
 * Monta las primitivas ya validadas (dsv4_*.h) a escala real: 43 capas, 72.317
 * tensores, 156 GiB en disco.
 *
 * REPARTO DE MEMORIA, que es la decisión de diseño central:
 *
 *   residente en RAM   8,67 GiB   atención, normas, routers, expertos
 *                                 compartidos, embeddings y lm_head — EN SU
 *                                 FORMATO NATIVO, sin dequantizar
 *   por streaming    137,1 GiB   los 11.008 expertos rutados, leídos del NVMe
 *                                 cuando el router los pide, con caché LRU
 *
 * Dequantizar el denso a f32 serían 26,8 GiB y no cabría; con descriptores
 * (`dsv4_weight.h`) el matmul lee el formato original y nunca materializa la
 * matriz. Es la misma decisión que toma colibrì con su struct `QT`.
 *
 * Build:  make -C port engine
 * Uso:    ./deepseek_v4 <dir_modelo> "prompt" [n_tokens]
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

/* Perfil: dónde se va el tiempo. Antes de optimizar, medir. */
static double g_t_attn, g_t_moe, g_t_io, g_t_head;
static uint64_t g_pf_batches, g_pf_reads, g_fb_bytes;
static uint8_t g_seen[(43 * 256 + 7) / 8];   /* capas x expertos, 1 bit cada uno */
static uint64_t g_distinct;
static FILE *g_trace;   /* DSV4_TRACE=fichero -> vuelca (token,capa,experto) */
static int g_tok_no;
/* `cap` de la pasarela: en colibrì significa slots de caché POR CAPA, no en
 * total. Se multiplica por n_layers al construir el tier. */
static int g_cache_per_layer;

/* ---------------------------------------------------------------------------
 * Política de carga, al estilo del `--load-mode` de llama.cpp.
 *
 * Hay dos poblaciones de pesos con regímenes OPUESTOS, y por eso no vale una
 * respuesta única:
 *
 *   denso, 8,67 GB   se lee en CADA token. Sus páginas fallan una vez y se
 *                    quedan residentes para siempre -> mapear gana: la carga
 *                    es perezosa y no se duplican 8,67 GB entre el montón y la
 *                    caché de páginas.
 *   expertos, 137 GB cada región se toca una vez y se descarta -> mapear
 *                    PIERDE. Medido: 23,7 s frente a 18,9. Los buffers de los
 *                    slots se reutilizan y sus páginas no vuelven a fallar,
 *                    mientras que el mapeo falla en cada región nueva: 13,4 MB
 *                    por experto son ~3.400 fallos de 4 KB. Cambiar un memcpy
 *                    por 3.400 entradas al kernel no sale a cuenta.
 *
 * Mapear no sale gratis pero compra MEMORIA PRIVADA, que es lo que de verdad
 * limita a quién le cabe el modelo. Medido con 14 tokens:
 *
 *   DSV4_LOAD   privado   working set   tiempo
 *   read         13,2 GB      13,2 GB    19,3 s   <- por defecto, lo más rápido
 *   dense         5,4 GB      12,2 GB    21,5 s   +11 %, -7,8 GB privados
 *   all           0,6 GB      24,7 GB    24,4 s   +26 %, corre en casi nada
 *
 * Lo que compra `dense` no es memoria libre —el working set apenas baja— sino
 * que esos 8,67 GB pasan de anónimos a respaldados por fichero: bajo presión el
 * SO los DESCARTA en vez de mandarlos al swap. `all` lleva al proceso a 0,6 GB
 * privados a cambio de llenar el working set y de un 26 % de tiempo.
 *
 * Es la misma razón por la que llama.cpp mantiene `--load-mode` en vez de
 * elegir por ti: no hay una respuesta buena para todas las máquinas.
 * ------------------------------------------------------------------------- */
typedef enum { LOAD_READ = 0, LOAD_DENSE = 1, LOAD_ALL = 2 } LoadMode;

static LoadMode load_mode(void) {
    const char *s = getenv("DSV4_LOAD");
    if (!s || !strcmp(s, "read")) return LOAD_READ;      /* por defecto */
    if (!strcmp(s, "dense")) return LOAD_DENSE;
    if (!strcmp(s, "all"))   return LOAD_ALL;
    fprintf(stderr, "DSV4_LOAD debe ser read|dense|all\n");
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
    CloseHandle(h);                      /* la vista mantiene vivo el fichero */
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

/* Los shards mapeados. Es seguro apuntar el kernel a offsets arbitrarios:
 * `quant.h` no usa ni una carga alineada, sólo `loadu` (49 de 49). */
static uint8_t **g_smap;

static void smap_init(shards *S) {
    g_smap = calloc((size_t)S->nfd, sizeof(uint8_t *));
    for (int i = 0; i < S->nfd; i++) {
        uint64_t len = 0;
        g_smap[i] = map_file(S->paths[i], &len);
        if (!g_smap[i]) {
            fprintf(stderr, "no pude mapear %s; se cargara leyendo\n", S->paths[i]);
            free(g_smap); g_smap = NULL;
            return;
        }
    }
}

/* Puntero al tensor dentro del mapeo, o NULL si no hay mapeo. */
static uint8_t *smap_ptr(shards *S, const char *nm) {
    if (!g_smap) return NULL;
    st_tensor *t = st_find(S, nm);
    if (!t) { fprintf(stderr, "falta %s\n", nm); exit(1); }
    for (int i = 0; i < S->nfd; i++)
        if (S->fds[i] == t->fd) return g_smap[i] + t->off;
    return NULL;
}

/* ---------------------------------------------------------------------------
 * Tier de expertos por streaming.
 *
 * Un experto son 3 matrices MXFP4 + sus escalas = 13,4 MB. No caben los 11.008
 * (137 GiB), así que se leen bajo demanda de los shards y se guardan en una
 * caché LRU por capa. Los offsets se resuelven una vez al cargar: en decode no
 * se busca por nombre, se hace `pread` a un offset conocido.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint8_t *buf[6];        /* w1,w1s, w2,w2s, w3,w3s */
    int layer, expert;
    uint64_t used;
    int pending;            /* lecturas en vuelo; 0 = listo para usar */
} ExpSlot;

/* Un trabajo de lectura: un tensor de un experto en un slot. */
typedef struct { int slot, layer, expert, k; } RdJob;

/* Un tensor de experto, ya resuelto: sin buscar por nombre en el camino
 * caliente. */
typedef struct { int shard; int64_t off, nb; } ExpTensor;

typedef struct {
    shards *S;
    int n_layers, n_experts, cap, inter, dim;
    ExpTensor *tens;        /* [(layer*n_experts + e)*6 + k] */
    ExpSlot *slots;
    int nslot;
    uint64_t clock, hits, miss;
    uint64_t bytes;
    int mapped;             /* LOAD_ALL: sin buffers ni LRU, se lee del mapeo */

    /* UN DESCRIPTOR POR HILO Y SHARD.
     *
     * En Windows, las ReadFile concurrentes sobre un handle abierto en modo
     * síncrono las serializa el propio SO sobre el objeto fichero, aunque se
     * les pase un OVERLAPPED. Medido en este equipo: lecturas dispersas en frío
     * dan 1,06 GB/s con un hilo y 2,9 GB/s con cuatro *si cada uno tiene su
     * propio descriptor* — compartiéndolo se quedan en el ritmo de uno solo.
     * Era el motivo de que paralelizar el prefetch no cambiase nada. */
    int nthreads, nshard;
    int *fd;                /* [thread * nshard + shard] */

    /* POOL PERSISTENTE DE LECTORES.
     *
     * `tier_prefetch` encola y vuelve; el MoE se pone a calcular el primer
     * experto mientras los demás siguen llegando. Antes bloqueaba hasta tener
     * los seis, así que los 4,3 s de cómputo y los 12,2 de I/O iban en serie.
     * También quita el montaje y desmontaje de una región OpenMP por capa y
     * token, que eran 602 en una generación de 14. */
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
    if (T->mapped) cap = T->cap = 1;   /* la caché la lleva el SO */
    T->slots = calloc((size_t)cap, sizeof(ExpSlot));
    for (int i = 0; i < cap; i++) { T->slots[i].layer = -1; T->slots[i].expert = -1; }

    /* Resuelve los 66.048 tensores de una vez: en decode no se busca por
     * nombre, se hace pread a un offset ya conocido. */
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
                    if (!t) { fprintf(stderr, "falta %s\n", nm); exit(1); }
                    ExpTensor *d = &T->tens[((size_t)l * n_experts + e) * 6 + k * 2 + which];
                    d->off = t->off; d->nb = t->nbytes; d->shard = -1;
                    for (int i = 0; i < S->nfd; i++)
                        if (S->fds[i] == t->fd) { d->shard = i; break; }
                    if (d->shard < 0) { fprintf(stderr, "shard de %s?\n", nm); exit(1); }
                }

    /* Un descriptor por hilo y shard: ver el comentario de la struct. */
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

/* Busca el experto en la caché; -1 si no está. */
static int tier_find(const ExpertTier *T, int layer, int e)
{
    for (int i = 0; i < T->nslot; i++)
        if (T->slots[i].layer == layer && T->slots[i].expert == e) return i;
    return -1;
}

/* Reserva un slot por LRU y lo marca como recién usado, de forma que una
 * reserva posterior DENTRO DE LA MISMA TANDA no pueda expulsarlo. */
static int tier_reserve(ExpertTier *T)
{
    int v;
    if (T->nslot < T->cap) v = T->nslot++;
    else {
        /* Un slot con lecturas EN VUELO no se puede expulsar: los workers están
         * escribiendo en sus buffers. Como mucho hay `topk` a la vez y la caché
         * tiene cientos, así que saltárselos nunca deja al LRU sin candidato. */
        v = -1;
        for (int i = 0; i < T->nslot; i++) {
            if (T->slots[i].pending) continue;
            if (v < 0 || T->slots[i].used < T->slots[v].used) v = i;
        }
        if (v < 0) { fprintf(stderr, "cache demasiado pequena: todo en vuelo\n"); exit(1); }
    }
    T->slots[v].used = ++T->clock;
    return v;
}

/* Reserva los buffers de los 6 tensores de un experto y anota los bytes. No
 * lee: la lectura se reparte aparte, ver `tier_prefetch`. */
/* Todos los expertos tienen la misma forma, así que los buffers se reservan una
 * vez por slot y no se vuelven a tocar: el `realloc` estaba en la parte SERIE
 * de cada tanda de prefetch, 19 veces por capa y por token. */
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

/* Lee un tensor suelto por el descriptor DEL HILO que llama.
 *
 * Sin estado compartido: cada hilo usa su propio fd y escribe en un buffer
 * distinto. Sólo `pending` se toca bajo el mutex, en el bucle del worker. */
static void tier_read_one(ExpertTier *T, int slot, int layer, int e, int k, int tid)
{
    const ExpTensor *d = &T->tens[((size_t)layer * T->n_experts + e) * 6 + k];
    const int fd = T->fd[(size_t)tid * T->nshard + d->shard];
    uint8_t *out = T->slots[slot].buf[k];
    int64_t done = 0;
    while (done < d->nb) {
        const ssize_t got = pread(fd, out + done, (size_t)(d->nb - done), d->off + done);
        if (got <= 0) { fprintf(stderr, "pread corto en capa %d experto %d\n", layer, e); exit(1); }
        done += got;
    }
}

/* Un worker: saca trabajos de la cola hasta que le dicen que pare. Su índice
 * es también el índice de su juego de descriptores. */
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
        RdArg *a = malloc(sizeof *a);   /* lo conserva el hilo */
        a->T = T; a->tid = i;
        pthread_create(&T->th[i], NULL, tier_worker, a);
    }
}

/* Encola las 6 lecturas de un experto. Con el mutex ya tomado. */
static void tier_submit(ExpertTier *T, int slot, int layer, int e)
{
    T->slots[slot].pending = 6;
    for (int k = 0; k < 6; k++) {
        T->q[T->qtail] = (RdJob){ slot, layer, e, k };
        T->qtail = (T->qtail + 1) % T->qcap;
    }
    pthread_cond_broadcast(&T->cv_job);
}

/* Espera a que un slot tenga sus 6 tensores. */
static void tier_wait(ExpertTier *T, int slot)
{
    const double t0 = now_s();
    pthread_mutex_lock(&T->mu);
    while (T->slots[slot].pending) pthread_cond_wait(&T->cv_done, &T->mu);
    pthread_mutex_unlock(&T->mu);
    g_t_io += now_s() - t0;   /* ahora mide ESPERA, no lectura: es lo que cuesta */
}

/* Trae de golpe los `n` expertos que el router acaba de elegir.
 *
 * Es la diferencia entre profundidad de cola 1 y n: un NVMe da su ancho de
 * banda con varias lecturas en vuelo, y de una en una se queda en la latencia.
 * La reserva de slots va en serie —el LRU es estado global— y sólo las lecturas
 * se reparten. */
static void tier_prefetch(ExpertTier *T, int layer, const int *es, int n)
{
    int slot[16], want[16], nw = 0;
    if (n > 16) n = 16;
    if (T->mapped) {
        /* Se le PIDE al SO que traiga los rangos, sin tocarlos: tocarlos
         * costaría el mismo ancho de banda que leerlos. Es una pista, no una
         * garantía; si la página no ha llegado, el fallo se resuelve al leer. */
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
        /* Cuántos expertos DISTINTOS pide la generación entera: es el número de
         * fallos que tendría una caché infinita, o sea el techo de lo que puede
         * dar agrandar ésta. Si ya estamos cerca, crecer no sirve de nada. */
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

    /* Se encola y se vuelve. Se reparte por TENSOR y no por experto: con ~3,2
     * expertos por tanda, un trabajo por experto dejaría la cola del NVMe a
     * profundidad 3,2 en vez de 19. */
    for (int j = 0; j < nw; j++) tier_alloc(T, slot[j], layer, want[j]);
    pthread_mutex_lock(&T->mu);
    for (int j = 0; j < nw; j++) tier_submit(T, slot[j], layer, want[j]);
    pthread_mutex_unlock(&T->mu);
}

/* Devuelve los descriptores del experto. Tras `tier_prefetch` siempre acierta;
 * el camino de carga queda como red de seguridad para quien no lo use. */
static void tier_get(ExpertTier *T, int layer, int e, DsV4W *w1, DsV4W *w2, DsV4W *w3)
{
    if (T->mapped) {
        /* Sin copia y sin caché propia: los descriptores apuntan a la página y
         * la caché es la del SO. Cuesta ~25 % más tiempo (fallos de página por
         * cada 4 KB de región nueva) y ahorra los GB de los buffers. */
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
    tier_wait(T, hit);   /* tras el prefetch suele volver sin bloquear */

    ExpSlot *s = &T->slots[hit];
    s->used = ++T->clock;
    *w1 = dsv4_w_mxfp4(s->buf[0], s->buf[1], T->inter, T->dim);
    *w2 = dsv4_w_mxfp4(s->buf[2], s->buf[3], T->dim,   T->inter);
    *w3 = dsv4_w_mxfp4(s->buf[4], s->buf[5], T->inter, T->dim);
}

/* Adaptador: el MoE pide expertos por callback y no sabe nada de shards. */
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
    int bos, eos;           /* del config.json: 0 y 1 en este checkpoint */
    float norm_eps, hc_eps;
    int *ratios;

    DsV4BlockCfg *cfg;      /* [n_layers] */
    DsV4BlockW   *w;        /* [n_layers] */
    int32_t *tid2eid;       /* [vocab * topk], sólo si hay capas hash */

    DsV4W embed, head;
    const float *final_norm, *hc_head_fn, *hc_head_base, *hc_head_scale;
    float *freqs;           /* [max_seq, rd/2, 2] */
    int max_seq;
} Model;

/* --- helpers de carga --------------------------------------------------- */
static void *raw_of(shards *S, const char *nm) {
    const int64_t n = st_nbytes(S, nm);
    void *p = malloc((size_t)n);
    st_read_raw(S, nm, p, 0);
    return p;
}
static float *vec_f32(shards *S, const char *nm, int n) {
    float *out = malloc((size_t)n * sizeof(float));
    st_tensor *t = st_find(S, nm);
    if (!t) { fprintf(stderr, "falta %s\n", nm); exit(1); }
    if (t->dtype == 2) { st_read_f32(S, nm, out, 0); return out; }   /* F32 */
    uint8_t *r = raw_of(S, nm);                                       /* BF16 */
    DsV4W v = dsv4_w_bf16(r, 1, n);
    dsv4_vec_f32(out, &v, n);
    free(r);
    return out;
}
/* Matriz cuantizada: FP8 si tiene .scale, BF16 si no.
 *
 * Es el grueso del conjunto denso (atención, embed, lm_head, compartidos), y
 * sale del mapeo cuando lo hay: no se copia nada y la carga es perezosa. */
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
    if (!f) { fprintf(stderr, "no encuentro %s\n", cfgp); exit(1); }
    fseek(f, 0, SEEK_END); long cn = ftell(f); fseek(f, 0, SEEK_SET);
    char *cbuf = malloc((size_t)cn + 1);
    if (fread(cbuf, 1, (size_t)cn, f) != (size_t)cn) { fprintf(stderr, "config corto\n"); exit(1); }
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

    /* Los diagnosticos van a stderr SIEMPRE: en modo serve, stdout es el
     * protocolo, y una linea suelta ahi descoloca a la pasarela. */
    fprintf(stderr, "config: %d capas, dim %d, %d cabezas x %d, %d expertos top-%d\n",
            M->n_layers, M->dim, heads, hd, n_exp, topk);

    /* --- freqs_cis con YaRN --------------------------------------------- */
    M->max_seq = 2048;                 /* suficiente para prompts de prueba */
    M->freqs = malloc((size_t)M->max_seq * (rd / 2) * 2 * sizeof(float));
    dsv4_precompute_freqs(M->freqs, rd, M->max_seq, orig, base, factor, bf, bs);

    /* --- globales -------------------------------------------------------- */
    M->embed = mat_of(&M->S, "embed", M->vocab, M->dim);
    M->head  = mat_of(&M->S, "head",  M->vocab, M->dim);
    M->final_norm    = vec_f32(&M->S, "norm.weight", M->dim);
    M->hc_head_fn    = (float *)raw_of(&M->S, "hc_head_fn");
    M->hc_head_base  = (float *)raw_of(&M->S, "hc_head_base");
    M->hc_head_scale = (float *)raw_of(&M->S, "hc_head_scale");

    /* --- por capa -------------------------------------------------------- */
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
            /* la tabla es la misma para todas las capas hash: se carga una vez */
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
            fprintf(stderr, "\r  capas cargadas: %d/%d", L + 1, M->n_layers), fflush(stderr);
    }
    fprintf(stderr, "\n");

    /* --- tier de expertos ------------------------------------------------ */
    /* ~13,4 MB por experto. El tamaño manda sobre el I/O, que es el cuello de
     * botella real, así que se deja ajustable para poder medirlo. */
    { const char *tp = getenv("DSV4_TRACE"); if (tp) g_trace = fopen(tp, "w"); }
    const char *cenv = getenv("DSV4_CACHE");
    const int cache = cenv ? atoi(cenv)
                     : (g_cache_per_layer ? g_cache_per_layer * M->n_layers : 384);
    tier_init(&M->tier, &M->S, M->n_layers, n_exp, inter, M->dim, cache);
    /* Cada capa pide sus expertos al tier por callback: el MoE no sabe nada de
     * shards ni de política de caché. */
    for (int L = 0; L < M->n_layers; L++) {
        M->w[L].moe.fetch = tier_fetch;
        M->w[L].moe.prefetch = tier_prefetch_cb;
        M->w[L].moe.fetch_ctx = &M->tier;
        M->w[L].moe.layer = L;
    }
    fprintf(stderr, "expertos: streaming con cache de %d slots de %d totales\n",
            cache, M->n_layers * n_exp);
    fprintf(stderr, "cargado en %.1f s\n\n", now_s() - t0);
}

/* ---------------------------------------------------------------------------
 * Forward de un token (decode) o de un prompt (prefill).
 * `h` es el stream residual [hc, dim] de UNA posición (batch 1).
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

    /* --- atención --- */
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
    /* el id del token sólo lo usan las capas hash, pero se pasa siempre */
    { const double _t = now_s();
      dsv4_moe_forward(&c->moe, &w->moe, coll, &tokid, 1, sub);
      g_t_moe += now_s() - _t; }
    dsv4_hc_expand(hout, sub, post, comb, mid, hc, dim);

    free(coll); free(sub); free(mid); free(tmp);
}

/* ---------------------------------------------------------------------------
 * Un pase de decode reutilizable.
 *
 * Lo comparten el modo CLI y el modo serve: la única diferencia entre generar
 * desde la línea de órdenes y desde la pasarela HTTP es de dónde viene el
 * prompt y a dónde van los tokens.
 * ------------------------------------------------------------------------- */
typedef struct {
    Model *M;
    DsV4AttnState *st;
    float *h, *h2, *logits, *emb;
    int pos;
    /* Los ids YA metidos por `run_step`, en orden. Es el historial contra el
     * que se compara el prompt siguiente para reaprovechar el prefijo comun.
     * Se anota dentro de run_step, asi que no puede desincronizarse del
     * estado por mucho que cambie el bucle de arriba. */
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

/* Entre peticiones hay que tirar TODO el estado de atención, no sólo el anillo
 * KV: cada capa comprimida arrastra el bloque en curso del compresor y, si es
 * ratio 4, el del indexer. Reaprovechar uno a medias mezcla dos conversaciones
 * de una forma que no da error, sólo texto raro. */
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

/* Un token adentro, los logits afuera. */
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
 * Muestreo. El CLI usa argmax; la pasarela manda temperatura y top_p por
 * petición, así que hace falta el nucleus de verdad.
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
    /* nucleus: se corta cuando la masa acumulada llega a top_p */
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
 * Modo serve: el protocolo mux que habla `openai_server.py`.
 *
 * Mismo formato de línea que `colibri.c` e `inkling.c` —byte a byte, para que
 * la pasarela sea la misma— con una simplificación: un solo slot de KV. El mux
 * admite hasta 16 y reaprovecha el prefijo común de cada conversación; aquí
 * cada petición reinicia el estado. Es correcto, sólo más lento en turnos
 * largos, y evita replicar los estados de compresor e indexer por slot.
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

/* -1 EOF, 0 nada util, 1 cancelar la peticion activa, 2 nueva peticion */
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
    (void)slot;                        /* un solo slot: ver la nota de arriba */
    char *payload = malloc((size_t)plen + 1);
    if (!payload) { printf("ERROR %s BAD_REQUEST\n", id); fflush(stdout); return 0; }
    if (fread(payload, 1, (size_t)plen, stdin) != (size_t)plen) { free(payload); return -1; }
    (void)fgetc(stdin);                /* el \n que cierra el frame */
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
    ids[np++] = M->bos;                              /* el modelo lo espera */
    np += tok_encode(&M->tok, q->payload, q->plen, ids + np, cap - np);
    if (np <= 1) {
        printf("ERROR %s EMPTY_PROMPT\n", q->id); fflush(stdout); free(ids); return;
    }

    /* REAPROVECHAR EL PREFIJO (el "truncate-and-extend" del protocolo).
     *
     * En una conversación, el prompt de un turno es el anterior más la
     * respuesta del modelo más el mensaje nuevo: una EXTENSIÓN pura de lo que
     * ya se metió. Comparando con el historial se saltan todas esas capas y
     * sólo se procesa la cola nueva. Sin esto, el tercer turno de una charla de
     * 300 tokens cuesta ~420 s antes de escribir la primera palabra.
     *
     * Cuando el prompt DIVERGE (el cliente edita el historial, o llega otra
     * conversación) haría falta retroceder el estado, y eso aquí no se puede:
     * el anillo KV sí, pero los compresores y el indexer acumulan el bloque en
     * curso y no se pueden "desacumular". En ese caso se reinicia entero, que
     * es correcto y sólo más lento. */
    int reuse = 0;
    while (reuse < R->nhist && reuse < np && R->hist[reuse] == ids[reuse]) reuse++;
    if (reuse < R->nhist || reuse >= np) { run_reset(R); reuse = 0; }
    if (reuse) fprintf(stderr, "[serve] prefijo reaprovechado: %d de %d tokens\n", reuse, np);

    const uint64_t hit0 = M->tier.hits, miss0 = M->tier.miss;
    const double t0 = now_s();

    int gen = 0, limited = 1, cancelled = 0;
    int tokid = ids[reuse];
    double tdec = 0.0;
    for (int step = reuse; ; step++) {
        const float *lo = run_step(R, tokid);
        if (step + 1 < np) { tokid = ids[step + 1]; continue; }   /* aún prefill */
        if (gen == 0) tdec = now_s();

        const int nx = sample_tok(lo, M->vocab, q->temp, q->top_p);
        /* 128805 = <|EOT|>: el formato de chat de DeepSeek lo usa además del
         * end-of-sentence del config. Parar sólo en el segundo deja el turno
         * corriendo hasta agotar max_tokens. */
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
            if (r == 2) {                            /* un solo slot */
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

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    /* La pasarela lanza el motor con SNAP=<dir>, SERVE=1 y NGEN=<max_tokens>,
     * y le pasa el `cap` de la caché como argv[1]. */
    const char *sv = getenv("SERVE");
    if (sv && sv[0] == '1') {
#ifdef _WIN32
        /* Sin esto, la traduccion CRLF del CRT corrompe los centinelas y deja
         * colgadas las lecturas por bytes contados (colibri #195). */
        _setmode(_fileno(stdin),  _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        const char *snap = getenv("SNAP");
        if (!snap || !*snap) { fprintf(stderr, "SERVE=1 necesita SNAP=<dir>\n"); return 1; }
        /* MinGW no trae setenv, así que el cap viaja por variable global. */
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

    /* BOS delante: el checkpoint no trae plantilla de chat, pero sí
     * `<|begin_of_sentence|>` (id 0), y el modelo se entrenó viéndolo. */
    int ids[1024];
    ids[0] = 0;
    int n = 1 + tok_encode(&M.tok, prompt, (int)strlen(prompt), ids + 1, 1000);
    printf("prompt: \"%s\" -> %d tokens (con BOS)\n", prompt, n);

    /* estado por capa */
    DsV4AttnState *st = calloc((size_t)M.n_layers, sizeof(DsV4AttnState));
    for (int L = 0; L < M.n_layers; L++)
        dsv4_state_init_full(&st[L], 1, M.cfg[L].attn.hd, M.cfg[L].attn.win,
                             M.max_seq, M.cfg[L].attn.ratio, M.cfg[L].attn.i_hd);

    const int dim = M.dim, hc = M.hc;
    float *h  = malloc((size_t)hc * dim * sizeof(float));
    float *h2 = malloc((size_t)hc * dim * sizeof(float));
    float *logits = malloc((size_t)M.vocab * sizeof(float));
    float *emb = malloc((size_t)dim * sizeof(float));

    printf("generando %d tokens...\n\n%s", ngen, prompt);
    const double tgen = now_s();
    int pos = 0;
    for (int step = 0; step < n + ngen; step++) {
        /* `ids[step]` siempre: para step < n es el prompt, y a partir de ahí lo
         * escribió la iteración anterior. Realimentar `ids[step-1]` —que es lo
         * que hacía antes— repite el último token del prompt y descarrila la
         * generación desde el primer paso. */
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
    printf("\n\n%d tokens en %.1f s (%.2f tok/s)\n", n + ngen, dt, (n + ngen) / dt);
    printf("perfil (total): atencion %.1f s | MoE %.1f s | head %.1f s | resto %.1f s\n",
           g_t_attn, g_t_moe, g_t_head, dt - g_t_attn - g_t_moe - g_t_head);
    printf("  I/O: %llu tandas de prefetch, %.1f expertos por tanda, %d lectores\n"
           "       %.2f GB por prefetch, %.2f GB por el camino de respaldo\n",
           (unsigned long long)g_pf_batches,
           g_pf_batches ? (double)g_pf_reads / g_pf_batches : 0.0, M.tier.nthreads,
           (double)(M.tier.bytes - g_fb_bytes) / 1e9, (double)g_fb_bytes / 1e9);
    printf("  del MoE, %.1f s son I/O de expertos y %.1f s computo\n",
           g_t_io, g_t_moe - g_t_io);
    printf("expertos: %llu hits / %llu miss (%.0f%% acierto), %.2f GB leidos\n",
           (unsigned long long)M.tier.hits, (unsigned long long)M.tier.miss,
           100.0 * M.tier.hits / (double)(M.tier.hits + M.tier.miss),
           (double)M.tier.bytes / 1e9);
    printf("  cache de %d slots (%.1f GB); %llu expertos distintos en total\n"
           "  -> una cache infinita tendria %llu fallos en vez de %llu\n",
           M.tier.cap, M.tier.cap * 13.4e6 / 1e9,
           (unsigned long long)g_distinct,
           (unsigned long long)g_distinct, (unsigned long long)M.tier.miss);
    return 0;
}
