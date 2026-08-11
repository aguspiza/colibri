# DeepSeek-V4-Flash: from 0.16 to 0.83 tok/s

Optimization log for the independent port (`c/dsv4_port.c`, see
[deepseek-v4-port.md](deepseek-v4-port.md)) on the real 284B checkpoint. Several
findings apply to any CPU V4 engine, not just this one — in particular the
Windows read serialization, Zen 2's slow `gather`, and the OpenMP team sizing.

It records
the changes that worked **and the ones that did not**, because four of the five
obvious-looking hypotheses turned out to be false, and killing them with a
measurement cost far less than implementing them.

Host: Ryzen 5700U (Zen 2, 8 cores / 16 threads, **no AVX-512**), 31.4 GB RAM,
checkpoint on a DRAM-less NVMe. 43 layers, 256 experts top-6, 156 GiB on disk.

## Summary

| | tok/s | what changed |
|---|---|---|
| starting point | 0.16 | — |
| vectorized kernels | 0.42 | AVX2+FMA and OpenMP in the FP8 and BF16 paths |
| one descriptor per thread | 0.51 | Windows was serializing the reads |
| arithmetic instead of `gather` | 0.65 | Zen 2's gather is slow |
| asynchronous prefetch | 0.70 | persistent reader pool |
| OpenMP team on physical cores | **0.83** | 8 threads beat 16 |
| long run (101 tokens) | 0.78 → 0.83 | cold start amortizes |

Current profile, 14 tokens with a 512-slot cache:

```
16.8 s   attention 4.2 s | MoE 12.1 s | head 0.3 s | rest 0.5 s
         of the MoE, 6.6 s is I/O wait and 5.5 s is compute
```

In every case the generated text is **identical** and the 95 oracle checks still
pass. No optimization here touches semantics.

## Method: measure before touching anything

Every change came out of a profile, not an intuition. The intuitions were wrong
almost every time:

- *"the bottleneck is the MoE, it moves 97.5 % of the parameters"* → it was
  **attention**, at 55 % of the time.
- *"1.28 GB/s is this NVMe's ceiling"* → it was **one reader's** ceiling.
- *"batching tokens cuts I/O by 1.78x"* → it makes it **worse**.
- *"more RAM for the cache would help"* → the time is **flat**.
- *"memory-mapping removes a copy, so it must be faster"* → 25 % slower.

### The page-cache trap

Three disk benchmarks in this work gave false results — 7.6 / 4.9 / 4.0 GB/s —
because each pass re-read the same bytes out of the OS page cache: they were
measuring RAM, not disk. On Windows `posix_fadvise(POSIX_FADV_DONTNEED)` is not
enough to drop them.

**The only trustworthy measurement is one where every pass touches bytes nobody
has read before.** Cold, with disjoint groups, the drive gives:

| | 1 thread | 4 threads | 8+ |
|---|---|---|---|
| scattered 4 MB reads | 1.06 GB/s | 2.73 GB/s | ~2.9 GB/s |

That 2.9 GB/s is the number every I/O figure below has to be judged against.

## What worked

### 1. The port's own kernels were single-threaded scalar C

The first profile split the 76.8 s of a 12-token run like this: attention 35.5 s
(55 %), MoE 27.3 s, `lm_head` 1.9 s.

Attention dominating was the tell: it does ~4.6 GMAC per token and took 3.5 s,
while the MoE did 6.5 GMAC in 1.1 s. **4.5x slower per MAC.** The reason is that
the MoE goes through colibrì's `matmul_mxfp4`, which already has AVX2 paths,
while the rest had been written for readability and for verifiability against
the reference implementation.

Vectorizing with AVX2+FMA and splitting the output rows across cores: **2.6x**.
The UE8M0 scale moves into a 256-float table — it used to be one `ldexpf` per
128-block per output row, eight million libm calls per token.

### 2. The e4m3 `gather` cost another 2x

Even vectorized, the FP8 kernel sat at **14 GFLOP/s** while MXFP4 reached 33 on
the same hardware. Attention re-reads its 5.40 GB of resident weights on every
token, which worked out to 8.4 GB/s effective — too low to be a memory ceiling.

The cause was decoding e4m3 with `_mm256_i32gather_ps` over a 256-float LUT. It
fits entirely in L1, but **on Zen 2 a gather executes as 8 sequenced accesses
and does not pipeline with the rest of the loop**.

Bit arithmetic does much better:

- `exp != 0`: `mag << 20` puts the mantissa in bits 22..20 and the exponent in
  26..23; adding `120 << 23` fixes the bias (e4m3 uses 7, f32 uses 127).
- `exp == 0`: subnormal, the value is `mant * 2^-9`.
- `mag == 0x7F`: NaN under OCP E4M3-FN. The formula would yield 480, so it is
  preserved with a `blend`. The checkpoint contains none — verified, the maximum
  is 0x7E = 448 — but if one ever shows up it is better for it to propagate than
  to be read as a valid number.

Attention **9.0 → 4.4 s**, i.e. 18 GB/s: *now* it is the memory bandwidth.

### 3. Windows was serializing the "parallel" reads

With compute fixed, I/O became 58 % of the time at 1.28 GB/s. I assumed that was
the drive's ceiling, because the prefetch already issued the top-k reads across
16 threads. But 1.28 is exactly **one** reader's rate.

`compat_pread` does `ReadFile` + `OVERLAPPED` on a handle opened in synchronous
mode, and **Windows serializes I/O on the file object** even when an `OVERLAPPED`
is supplied. The parallelism was only apparent.

The fix is **one descriptor per thread and shard** (48 x 16 handles, opened at
load time). I/O from 20.6 to 12.3 s, bandwidth 1.28 → 1.98 GB/s.

Two things left the hot path along the way: the offsets of the 66,048 expert
tensors (now resolved once at load) and the buffers' `realloc`, which sat in the
**serial** section of every prefetch batch.

### 4. Asynchronous prefetch with a persistent pool

The prefetch blocked until all six experts had arrived and only then computed:
the MoE's 5.2 s of compute and the 12.3 s of I/O ran in series. With a persistent
reader pool, `tier_prefetch` enqueues and returns, and the MoE works on the first
expert while the rest are still on their way.

I/O wait 12.3 → 9.7 s. It also removes the setup and teardown of one OpenMP
region per layer per token — 602 of them in a 14-token run.

One correctness detail: the LRU **must not evict a slot with reads in flight**,
because the workers are writing into its buffers. `tier_reserve` skips them; at
most `topk` are in flight and the cache holds hundreds, so it never runs out of
candidates.

### 5. The OpenMP team, sized to physical cores

`kimi_k3.c` and `olmoe.c` already called `coli_omp_tune_threads` at startup; this
engine did not, and ran on all 16 logical threads:

| threads | time | I/O wait | MoE compute |
|---|---|---|---|
| 16 (logical) | 19.0 s | 8.8 s | 5.3 s |
| **8 (physical)** | **17.1 s** | **6.6 s** | 5.6 s |
| 4 | 19.0 s | 5.6 s | 6.9 s |

11 % for free. The interesting part is that **I/O wait drops too**: the reader
pool is sized from `omp_get_max_threads()`, so shrinking the team stops the
readers from fighting the compute over the same cores.

`omp_tune.h` deliberately takes only the sizing and leaves the spin-wait out,
and this engine is exactly the case that justifies it: with I/O at 40 % of the
time, a team spinning idle would steal cores from the work that matters.

### 6. Prefill: batch the prompt, not the decode

Decode batching loses (see below), and that led to the wrong conclusion for a
while: that batching was settled. It is not — **prefill is a different problem**.
The tokens of a prompt are all known at once, so there is no sequential
dependency to break, and the win comes from the same expert union that decode
cannot exploit.

A 140-token prompt, one token at a time, was **202.4 s** and read **270.8 GB**.
Batched it reads **76.9 GB** — 3.5x less — because a chunk of 140 rows visits
each distinct expert once instead of once per row.

Two paths, both measured on the same prompt:

| prefill | TTFT | why |
|---|---|---|
| one token at a time | 202.4 s | baseline |
| chunk 16 (9 chunks) | 190.2 s | little to amortize per chunk |
| chunk 48 (3 chunks) | 181.4 s | |
| **chunk 256 (1 chunk)** | **~157 s** | default |
| one batch + batched attention | ~140 s | fastest, needs a fresh state |

Batched attention is the faster of the two but it derives each token's RoPE
position as `r % s`, so it assumes the batch starts at position 0: it is only
valid for a whole prompt from an empty cache, and it costs ~416 KB per token, so
it is capped (`DSV4_PREFILL_MAX`, 512). The chunked path has neither constraint —
absolute positions, state maintained per token — so it works at any length and
from any position, including after a reused prefix. **There is no length limit
any more**; a 5k-token prompt is chunked, not decoded one token at a time.

`DSV4_CHUNK` (default 256) bounds memory, not correctness: 5k tokens cost ~107 MB
instead of ~2 GB. More chunks is slower, which is why the default is the largest
chunk that stays cheap.

Two things were verified rather than assumed. The harness checks
`prefill == N sequential decodes` bit-for-bit at lengths 32/127/128/129/200/300
(the window is 128 and the compressor ratio 4/128, so the interesting cases are
around the block boundaries), and the seam test decodes 8 more tokens past the
prompt to catch a state that is *nearly* right. On the engine, all four rows of
the table above generate **identical text**.

That last check earned its keep immediately: indexing the last row of the whole
prompt instead of the last row of the current chunk read past the end of the
buffer, and the symptom was not a crash but a *fluent wrong answer*.

## What did NOT work

### Batching tokens at decode time (and with it MTP/DSpark)

*This is about decode. Prefill batches and it wins -- finding 6.*

It looked like the obvious lever. Consecutive tokens share experts, so a batch of
5 asks for 724 distinct layer-experts instead of 5x258 = 1290: **1.78x fewer
requests**. And 5 is exactly `dspark_block_size` in `config.json`.

But that is the wrong comparison. The real baseline is not 258 requests per token
but **124.2 cache misses**, because the LRU was already exploiting that same
overlap — and doing it better. Simulating the cache policy over the real routing
trace (the simulator reproduces **exactly** the 2,980 misses the engine reports):

| batch | misses/token | vs sequential |
|---|---|---|
| 1 | **124.2** | — |
| 2 | 138.5 | 0.90x |
| 5 | 162.2 | 0.77x |
| 8 | 138.1 | 0.90x |

Batching makes it **worse**. In sequential order each layer is revisited every
258 accesses, which fits in 512 slots; in batches of 5 it is revisited every 724,
which does not — so batching converts cache hits into deduplication and loses on
the exchange. Above ~2,048 slots the misses saturate at the *compulsory* ones —
each distinct expert read once — where there is nothing left for batching to
improve.

The only saving left was amortizing the 5.40 GB/token of attention weights, and
that turned out far cheaper to get by fixing the kernel (finding 2).

**And more RAM does not change it.** With the kernels fixed, the engine is *at
its roofline* for nearly half the time:

| | time | GFLOP/s | limited by |
|---|---|---|---|
| attention | 4.2 s | 31 | FLOPs **and** bandwidth simultaneously |
| MoE compute | 5.2 s | 35 | FLOPs |

The 5.40 GB/token of attention weights at 18 GB/s gives exactly the measured 0.30
s/token, and those same 0.30 s give 31 GFLOP/s — precisely what the best kernel
in the repo achieves on this hardware. Both limits coincide, so amortizing weight
traffic across a batch frees nothing: it hits the FLOP wall immediately.

That is what settles the question. **Speculative decoding trades FLOPs for fewer
sequential steps** — it *increases* total FLOPs, since it computes tokens that
get thrown away — and it pays off when FLOPs are cheap, which is the GPU case.
Here they are not: 47 % of the time is already on the AVX2 roofline of 8 Zen 2
cores. There is no budget to spend.

The port's MoE implements the **expert-union** traversal anyway — each expert is
applied to every row that selected it — because it is correct and any batched
path needs it. With `rows == 1` the accumulation order is unchanged and decode
produces the same bits.

### Growing the expert cache

Sweep on the real engine, 14 tokens:

| cache | GB | hit rate | bytes read | time |
|---|---|---|---|---|
| 128 | 1.7 | 0 % | 48.29 GB | 20.5 s |
| 256 | 3.4 | 19 % | 44.03 GB | 20.5 s |
| 512 | 6.9 | 49 % | 24.69 GB | **20.0 s** |
| 768 | 10.3 | 55 % | 21.69 GB | 21.1 s |
| 1024 | 13.7 | 57 % | 20.87 GB | 21.9 s |

The time is **flat** while the bytes vary by 2.3x. And at 101 tokens, where the
simulator predicted the big cache would win easily (45 misses/token with 2048
slots against 114 with 512), the real measurement stays flat too: **130.8 s with
512 against 129.5 s with 1024**, reading 32 % less.

The explanation: **the engine's LRU is redundant with the OS page cache**, which
holds exactly the same bytes. Growing ours adds no capacity, it takes capacity
away from the OS in order to duplicate what the OS already had. The proof is in
the effective bandwidth: with a 128-slot cache the engine "reads" 48.29 GB in
9.8 s = **4.9 GB/s**, well above the 2.9 the drive delivers. Those bytes came
from RAM.

**A warning about the simulator**: it models only the engine's LRU and cannot see
the page cache underneath, so it overestimates the benefit of more slots. It is
reliable for counting *requests* — it reproduces the measured misses exactly —
but not for predicting *time*.

Practical consequence: `DSV4_CACHE` exists for experimenting, but raising it does
not pay. A small cache leaves more RAM to the OS, which manages it better.

### Memory-mapping the shards (as a speed play)

With `pread` the bytes travel twice: disk → page cache → our buffer, and the copy
is paid even when the page was already in RAM. Mapping the shards and pointing
the MXFP4 kernel straight at the page looked like it would remove the copy *and*
the cache duplication in one move. It is also safe: `quant.h` does not use a
single aligned load, only `loadu` (49 of 49).

Implemented with `CreateFileMapping`/`MapViewOfFile` plus
`PrefetchVirtualMemory` to ask the OS for the ranges ahead of time — without
touching them, which would cost the same bandwidth as reading them — it comes out
**worse**: 23.7 s against 19.4.

The profile explains why: I/O wait drops to 0.0 s but MoE compute jumps from 5.1
to 18.4 s. The page faults happen *inside* the matmul. And the reason is
structural:

**The slot buffers are reused, so their pages are mapped once and never fault
again. The mapping faults on every new region** — 13.4 MB per expert is ~3,400
faults of 4 KB, each one a trip into the kernel. Trading a `memcpy` (10-20 GB/s,
no kernel entry) for 3,400 traps per expert is nowhere near worth it.
`PrefetchVirtualMemory` turns them into soft faults, but soft faults still cost
~1 µs each.

The general lesson: **mmap wins when pages are reused and loses when you sweep
constantly over new data**, which is exactly the access pattern of a streaming
expert tier.

## `DSV4_LOAD`: mapping does not buy speed, it buys private memory

Dropping mmap for being slow would have been a mistake, because speed was not the
axis that mattered. llama.cpp's `--no-mmap` did two things at once and in recent
versions it is deprecated in favour of `--load-mode`
(`none | mmap | mlock | mmap+mlock | dio`), precisely because **there is no
single right answer for every machine**. The same applies here.

Measured over 14 tokens, averaging two passes:

| `DSV4_LOAD` | private | working set | system free | time |
|---|---|---|---|---|
| `read` (default) | 13.2 GB | 13.2 GB | 8.2 GB | **19.3 s** |
| `dense` | **5.4 GB** | 12.2 GB | 8.4 GB | 21.5 s |
| `all` | **0.6 GB** | 24.7 GB | 0.6 GB | 24.4 s |

- **`read`** — everything through `pread` into private buffers. Fastest.
- **`dense`** — maps the dense set (attention, `embed`, `lm_head`, shared
  experts) and leaves the routed experts on `pread`. Each population gets the
  policy that suits it. Cuts private memory by 7.8 GB for 11 % of the time.
- **`all`** — maps the experts too and drops the LRU cache entirely. The process
  settles at **0.6 GB private**: it fits on machines where `read` will not even
  start, in exchange for 26 %.

The important nuance is **what `dense` actually buys**. It does not lower the
working set — the pages stay resident, just shared with the OS cache — it
converts 8.67 GB of **anonymous** memory into **file-backed** memory. Under
pressure the OS can discard those pages and re-read them from disk instead of
pushing them to the page file. It is more robust, not more free.

And beware the metric: the first attempt measured *working set* and concluded
that mapping **increased** consumption (24.9 GB), because touched file pages
count toward the process working set. For "how much RAM does this process need"
the number to read is **private bytes**.

The second llama.cpp axis, residency (`mlock` / `VirtualLock`), is not
implemented. It matters little here: the dense set is touched on every token, so
it stays resident on its own, and pinning it on Windows means adjusting the
process working-set quota.

All three modes generate **the same text**.

## What is left

- **I/O is 39 %** of the time (6.6 s of 16.8). The floor is the distinct experts
  that simply have to be read: 1,529 across 14 tokens, about 20.5 GB, which at
  2.9 GB/s is ~7 s. We are essentially at it.
- **A second SSD is the only large lever left.** colibrì already ships the whole
  mechanism and it is model-agnostic: `COLI_MODEL_DIRS` splits the shards across
  drives (each holding a **distinct** subset, no duplication),
  `COLI_MODEL_MIRROR` keeps copies and splits reads across them, and
  `COLI_DISK_WEIGHTS` sets the split — or measures it with a bandwidth probe at
  startup. Since expert reads already go through a thread pool with one
  descriptor per thread, they would parallelize across drives almost by
  themselves. With two comparable NVMes the ceiling would go from 2.9 to
  ~5.5 GB/s and I/O would roughly halve: **~13 s instead of 16.8, i.e. 1.05
  tok/s**.
  On top of that there is `coli mirror`, which ranks by usage from `.coli_usage`
  and stages the hot experts on the fast drive — a good match for what was
  measured, since the traffic concentrates in relatively few distinct experts.
  *It does not help with the drives currently attached:* D: and E: are
  mechanical USB disks, far slower than the NVMe, so splitting toward them would
  hurt unless the weights were made very asymmetric.
- **What is NOT left**: memory-mapping (measured, loses), growing the cache
  (measured, flat), batching tokens at *decode* time and MTP/DSpark (measured,
  loses). Each with its reasoning above. Batching at *prefill* time is a
  different question and it does work -- finding 6.
- **Better overlap**: layer L+1's attention depends on layer L's MoE, so the
  `attn -> router -> read -> MoE` chain is inherently serial. The only overlap
  available is the one already done inside a layer.
