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

### 7. The prompt cache: pay the prefill once, ever

Finding 6 took a 20k-token prompt from "impossible" to ~4.5 h. That is still not a
usable agent backend, and no engineering fixes it: 13B active parameters is
~26 GFLOP per token against the 31-35 GFLOP/s these kernels reach, so ~0.8 s/token
is the floor with infinite RAM and infinite disk. 20265 tokens is hours by
arithmetic.

But it is the SAME prefill every session. So it is paid once and kept.

`DSV4_CACHE_DIR` holds snapshots of the state that decoding actually reads: the
per-layer attention arrays, the position, and the token history prefix reuse
compares against (`h`/`h2` are scratch). Measured on a 140-token prompt:

| | TTFT | text |
|---|---|---|
| cold | 177.0 s | `' Berlin.'` |
| restored | **3.0 s** | `' Berlin.'` |

The snapshot is 0.136 GB at `DSV4_MAX_SEQ=8192` and reloads in 0.1 s. **Its size
follows max_seq, not the number of tokens**, because that is how the arrays are
allocated -- a checkpoint at token 2000 costs the same as one at 20000 (~0.46 GB
at 32768).

**Checkpoints, because our state cannot be rewound.** The compressors and the
indexer accumulate an in-progress block that cannot be un-accumulated, so a prompt
that diverges from the cached one at token 15000 cannot use a 20000-token snapshot
*at all*. `DSV4_CKPT_EVERY` (4096) writes one every N tokens and the deepest one
that is still a prefix wins:

| divergent prompt | prefill | TTFT | text |
|---|---|---|---|
| no cache | 83 tokens | 95.3 s | `' Roma.'` |
| checkpoint at 64 | 19 tokens | **27.4 s** | `' Roma.'` |

llama.cpp added `--ctx-checkpoints` for the same reason: an SWA state cannot be
rolled back either. Two things were taken from its design and one was not: the key
is an exact token comparison (its `get_common_prefix`), the store is a set of
entries picked by depth -- but there is no hash. With one candidate at a time and
80 KB of ids there is nothing to optimize, and a collision would produce wrong
output silently; reading the ids first and the 0.46 GB of arrays only on a match
does the same job without that risk. (llama.cpp does not hash tokens either; its
FNV-1a is for image bitmaps.)

Chunking cost nothing to add: `run_prefill` already takes an absolute start
position and maintains the state itself, so a slice is the same work as the
equivalent stretch of one long call.

At the measured 1.12 s/token the interval is a straight trade: 4096 -> at most
76 min re-prefilled on a divergence, 2048 -> 38 min at twice the files. Five
checkpoints for a 20k prompt is ~2.3 GB.

**The runbook, and why it cuts at the system boundary.** The prompt is captured by
the engine (`DSV4_DUMP_PROMPT`) rather than by the gateway, because that is the only
place the exact bytes the tokenizer sees are available -- a renderer difference of
one token would be a silent miss after hours of work. Then:

```
python3 dsv4_cut_prompt.py captured.txt stable.txt       # cut + verify
BUILD_CACHE=1 SNAP=<model> DSV4_PROMPT_FILE=stable.txt   DSV4_CACHE_DIR=cache DSV4_MAX_SEQ=32768 DSV4_CKPT_EVERY=4096 deepseek_v4
```

**The cache can only extend to the first thing that changes between sessions**, and
finding that is the whole game. Two candidates, both measured on the real agent
prompt (20249 tokens):

| cut at | cached | left per session |
|---|---|---|
| first `<｜User｜>` (end of system) | 20243 (99.97 %) | 6 tokens |
| `Today is <date> ... directory is '<cwd>'` | **19701 (97.3 %)** | 548 tokens, ~10 min |

The first one looks better and is wrong: that line is volatile every single day, so
a cache built past it **expires at midnight**. It sits at token 19701, 97.3 % of the
way in, so cutting before it costs 2.7 % of the prefill and buys a cache that does
not. Everything stable that happens to sit after it -- here a `<critical>` block --
is unreachable, which is what a prefix means, not an oversight.

That was caught by reading the captured prompt, not by the machinery: nothing in the
engine can tell a stable token from a volatile one.

The cut is verified rather than assumed, because a byte offset is not necessarily a
token boundary and byte-level BPE merges can span it. The role marker happens to be
safe (it is an added token, atomic to pre-tokenization); `Today is` is not
guaranteed to be, so the script checks that the truncated text's tokens are a true
prefix and walks the offset backwards until they are. A silent mismatch would only
surface after hours of prefill. (Note the marker is spelled with FULLWIDTH VERTICAL
LINE, U+FF5C: the ASCII `<|User|>` never matches.)

Note that `DSV4_MAX_SEQ` must be the same when building and when serving: it is part
of the fingerprint, so a mismatch is rejected -- correctly, but at the cost of the
whole build.

**The arrays are trimmed to what was written, and that buys more than size.** The
compressed entries are written at `win + start_pos/ratio`, i.e. a prefix at a fixed
offset, so storing only that prefix makes a snapshot **portable across
`DSV4_MAX_SEQ`** -- the offset does not move when max_seq changes, only the array's
total length. So max_seq left the fingerprint, and raising the context later (the
model allows 262144) neither multiplies the files by 8 nor invalidates the cache.
Verified by building at 8192 and loading at 16384: same text, and both equal to a
cold run.

The size now follows the position: 24.4 / 25.2 / 25.4 MB for checkpoints at 64 / 128
/ 140 tokens, against a flat 136 MB before. A checkpoint at `p` costs ~20 MB fixed
(the window ring and the in-progress blocks) plus ~13.5 KB per token, so the series
for interval N over T tokens is proportional to T squared over 2N -- halving N
doubles the disk. It does not make fine-grained checkpointing cheaper, it makes it
possible: 19 checkpoints over 19701 tokens is 3.3 GB trimmed, against 8.7 GB at
max_seq 32768 and 70 GB at 262144. (Deltas would fix the redundancy -- the
compressed region is append-only, so only ~20 MB per checkpoint is genuinely new --
and are not implemented.)

**The trimming shipped broken first, and only one kind of test caught it.** The used
count came from `st->n_written`, which is assigned solely in
`dsv4_state_seed_from_prefill` and read nowhere, so on the per-token path -- the one
the offline builder uses -- it stays 0 and every compressed entry was trimmed away.
The restored state still produced a plausible first word from the window alone and
diverged one token later. The portability check passed the whole time: the cache was
self-consistent, just wrong. Only comparing restored output against a **cold run**
found it, and the giveaway in hindsight is that all three files were byte-identical
in size.

Two failure modes are handled rather than hoped about: the file is written to
`.part` and renamed only when complete, so a kill during a 0.46 GB write cannot
leave something the next start would load as valid; and the fingerprint
(layers, max_seq, dim, vocab, hc) is checked, because a snapshot from another
`DSV4_MAX_SEQ` would read the right number of bytes into differently shaped
arrays -- silently.

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

| `DSV4_LOAD` | private | working set | time |
|---|---|---|---|
| `read` (default) | 13.2 GB | 13.2 GB | **19.3 s** |
| `dense` | **5.4 GB** | 12.2 GB | 21.5 s |
| `all` | **0.6 GB** | 24.7 GB | 24.4 s |

("system free" is deliberately not a column. During the long prefill it read 1.1 GB
in *both* `read` and `dense`, which says nothing: in one case the shortfall is
anonymous memory that can only be relieved by swapping, in the other it is
file-backed pages the OS drops instantly. Free memory was never the metric.)

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

### Mapping is not just lazy loading -- it is accepting the OS's eviction policy

The virtual address space is free in the literal sense: 156 GB of it costs nothing
on 64-bit, and pages that are never touched are never read. But that is not the
whole price. Mapping a page also hands the OS the decision of **when to take it out
of physical RAM**, and the OS does not know our access pattern.

That reframes what `dense` does, and not in its favour. The dense set is the one
population we know with certainty: it is needed on *every single token*, forever.
`dense` takes precisely that -- the thing with no uncertainty in it -- and makes it
evictable, in order to save private bytes. It is backwards. The population that
*should* be left to the OS is the routed experts: 137 GB, uncertain, and measured
flat (see "Growing the expert cache").

Measured during a 19701-token prefill in `dense` mode, the loss did not
materialize: the working set sat at 11.39 GB and grew by ~1 MB over 30 s, so the OS
was keeping the mapped dense pages resident. The point is that this holds only while
nothing else asks for RAM. It is a hope about policy, not a property of the design.

**So the second llama.cpp axis, residency (`mlock` / `VirtualLock`), is the missing
piece rather than a detail.** The earlier note here said it "matters little,
because the dense set is touched on every token, so it stays resident on its own" --
that is the hope, stated as if it were a guarantee. Mapping *and pinning* the dense
ranges is the configuration `dense` is trying to be: file-backed (no commit charge,
never written to the page file) **and** never evicted (no policy delegated). On
Windows it means raising the process working-set quota, which is why it is not free
to add, not why it does not matter.

### The asymmetry, which is the general rule

The useful question is never "mmap or not". It is **whether our information beats
the OS's** for that population:

| population | what we know | who should decide |
|---|---|---|
| routed experts (137 GB) | the router looks **one layer** ahead, and that margin is already spent on the prefetch pool | the OS -- and measured flat, so nothing is lost |
| dense set (8.67 GB) | needed on every token, with certainty | **us** -- never cede this one |

The expert side is not a concession, it is a measurement: our LRU turned out to be
redundant with the page cache (time flat 20.0-21.9 s while bytes read varied 2.3x,
at 4.9 GB/s effective against 2.9 GB/s from disk). There is no long horizon to
exploit, because layer L's routing depends on layer L-1's output.

The dense side needs no measurement at all, which is exactly why ceding it is the
mistake.

A related trap, for completeness: comparing `read` and `dense` by CPU-seconds per
wall-second **does not work**, because attention cost grows with sequence position,
so two samples taken at different points in the same prefill are not comparable
even for identical configurations. The valid comparison is s/token over the same
token range.

All three modes generate **the same text**.

## Memory: the engine was not managing it, the OS was

The port streams 137 GB through 31.4 GB of RAM on purpose. If the OS is paging, the
premise is refuted -- it is doing a worse version of the same job, behind our back,
against the same disk we are already saturating. It was.

A six-hour cache build filled the disk and ended with reads failing with `EIO`. The
accounting, all measured:

| | |
|---|---|
| engine private memory (`read` mode) | 14.0 GB (8.67 dense + ~5 of LRU) |
| **system file cache, filled by our own preads** | **12.5 GB** |
| other 281 processes | 3.8 GB |
| kernel pools | 1.6 GB |
| total against 31.4 GB of RAM | **31.9 GB** |

Over the limit, and the eviction rule decides the rest: clean file pages are dropped
for free, anonymous pages must be **written** first. So Windows pushed the engine's
14 GB to the page file, which grew ~20 GB, filled the disk, and then large reads
started failing. **The engine's own streaming evicted the engine.** What got written
to the page file was ~14 GB of duplicates of bytes already sitting in the
`.safetensors` next to them.

The commit counter alone did not predict it -- 37.1 GB against 31.4 GB of RAM
implied ~6 GB of paging, and the page file grew 20. The missing term was the system
cache, which is in neither our working set nor our commit.

Four separate mechanisms, one per link in the chain:

| link | fix | knob |
|---|---|---|
| our reads fill the OS cache | **direct I/O**, unbuffered | `DSV4_DIRECT=1` |
| dense weights are anonymous, so eviction means swap | map them | `DSV4_LOAD=dense` |
| the LRU has no budget | size it from RAM (still a fixed 384 by default) | `argv[1]` |
| residency is the OS's call | hard working-set cap | `DSV4_WS_MAX_GB` |

None of them is sufficient alone, and two of them are traps on their own:

- **`read`** keeps the fastest expert path but holds 14 GB of anonymous memory, which
  is the thing that swaps.
- **`all`** takes private memory to 0.6 GB, and then the working set grows until only
  0.5 GB of the machine is available. Worse, it removes the reader pool: experts
  arrive as synchronous page faults, at queue depth ~1. Measured, same prompt,
  minutes apart: **141 MB/s and 28 % CPU with `all`, against 360 MB/s and 54 % with
  `dense`** -- 686 fault-driven I/Os per second of 211 KB each, versus explicit
  parallel preads. `PrefetchVirtualMemory` is advisory; when compute catches up with
  it, the fault is synchronous and nothing is pipelined.

The working-set cap is the cheapest of the four. It does not discard anything: the
excess moves to the **standby list**, still in RAM serving re-reads, but counted as
available.

| | working set | available |
|---|---|---|
| no cap | 22.86 GB | 0.55 GB |
| capped at 8 GB | 8.00 GB | 16.80 GB |

with the throughput difference inside the noise (254 MB/s against 325 and 230).
Clean file pages cost nothing to move.

## The expert cache: right for decode, useless for prefill, and measured wrong

### What the LRU is actually for

Two uses of layer L are 258 expert reads apart in DECODE (42 other layers x 6
experts), which a 384-slot cache absorbs -- that is the original 124.2
misses/token against 258 requests. In CHUNKED PREFILL they are ~10,700 apart,
because a chunk visits every layer's whole union before coming back. No cache
short of the entire model survives that gap.

Measured, 512 real tokens, chunk 128, 384 slots:

```
[prog] 512/512  1.05 s/token  | lru 384/384 slots  0.0% hit  290 GB read
```

**0.0 %.** Not "flat", not "redundant with the page cache" as this document
previously said -- zero. The cache must exist and must be correct, because decode
depends on it, but in prefill it cannot be a reuse cache. It can only be a buffer
pool.

### Three measurement bugs, all mine

**`argv[1]` was never read in BUILD_CACHE.** The parse sat inside the `SERVE`
branch, so three relaunches with caches of "172, 301, 344 slots" all ran with the
default 384. That is why private memory never moved between them, and why every
LRU-size comparison from that night is void. `DSV4_CACHE` was the only knob that
worked; the parse now happens before the mode branches.

**The hit counter measured the prefetch, not the cache.** `hits` was incremented in
`tier_prefetch` on every `tier_find` success. That was fine while prefetch ran once
per layer, and became nonsense when the window became a sliding one and it ran once
per expert in the union: it reported **98.1 %**, which is the rate at which the
prefetch recognizes what it just queued. Accounting now counts `uses` (in
`tier_get`, one per real request) against reads issued, both idempotent.

**The prefetch inverted the eviction policy.** `tier_prefetch` also bumped
`slots[f].used` on a find. With a sliding window, the experts furthest ahead get
re-found on every call and look freshest, while the one about to be consumed --
queued a while ago, not re-found since -- looks oldest. **The LRU was evicting
precisely the next expert it needed.** Recency now belongs to real use only.

### What linearity actually buys

The router of layer L depends on layer L-1's output, so *which* experts a layer
wants is not knowable ahead. Reading that as "nothing is knowable" was a mistake:
the *structure* is fixed. Prefill walks layers 0..42 and the next chunk walks them
again, and with 256 rows a layer's union is 173 of 256 experts (2.2 GB, measured
from the read counters). The previous chunk's union predicts the next one well
enough to fetch it while the current layer computes.

Before this, the only prefetch was the current layer's union, taken from the
router -- i.e. requested exactly when it was already needed. The chain
`attention -> router -> read -> MoE` is serial per layer, which is why the engine
sat at 37-54 % CPU with the disk at 183-360 MB/s of the 1514 MB/s it can deliver.

### And then the layout made prediction unnecessary

A layer's 256 experts are 1536 tensors sitting **contiguously in a single shard**:

```
capa  0: 256 expertos, 1536 tensores, shards [1]
         rango 3.55 GB  |  bytes utiles 3.42 GB  |  huecos internos: 1
```

96 % dense, one gap. So:

| strategy | bytes | rate | per layer |
|---|---|---|---|
| 173 scattered experts (68 %) | 2.2 GB | 216-360 MB/s | 6-10 s |
| the whole range, sequential | 3.55 GB | 1514 MB/s (measured, one thread) | **2.3 s** |

**Reading 61 % more bytes takes a third of the time**, because this drive gives ~7x
more sequentially than scattered. Which removes the problem instead of optimizing
it: there is nothing to predict and nothing to cache, because the whole layer comes
every time. The router's choice stops mattering, the 32 % waste of a partial union
disappears, and an expert index becomes an offset into a buffer.

Two buffers of 3.55 GB -- compute L out of one while L+1 lands in the other -- which
is 7.1 GB, and the reason two is the right depth rather than three: a third only
helps if reads were more than twice as slow as compute, and then the disk is the
wall and getting further ahead buys nothing.

### Measured: it takes replacing BOTH of the OS's services

Each piece lost on its own, and for reasons that cancel. The same 512 real tokens,
four chunks, identical work:

| | s/token | bytes | throughput |
|---|---|---|---|
| per-expert, buffered | **1.05** | 290 GB | 0.54 GB/s |
| per-expert, direct I/O | -- | -- | 0.22 GB/s, 116 IOPS |
| whole-layer, buffered | 1.22 | ~600 GB | 0.49 GB/s |
| **whole-layer + direct I/O** | **1.06** | **762 GB** | **1.40 GB/s** |
| whole-layer, 16 logical threads | 1.29 | | |

The OS cache was providing TWO services: reuse caching, where our knowledge is
strictly better (0.0 % reuse measured -- it caches 13 GB that is never re-read), and
**readahead plus request merging**, where it was genuinely working (756 requests of
475 KB against direct I/O's 116 of 1.9 MB). Direct I/O with scattered reads removed
both and replaced neither, which is why it lost. Whole-layer streaming IS the
replacement for the second: 8 pieces of 444 MB is better readahead than the OS can
infer, because we know the whole layer will be used and it has to guess.

Together they read **2.6x more bytes in the same wall time**, at 93 % of the drive's
measured sequential ceiling, with no OS cache and the LRU never even allocated
(`lru 0/8 slots`). Text verified identical.

Equal time, not faster -- but the position dependence should differ, and that is the
reason to prefer it on a long build. The per-expert path degraded 1.32 -> 1.52 -> 1.78
s/token over successive 4096-token blocks; whole-layer bytes are constant by
construction, the same 3.55 GB at token 500 as at token 19000.

**16 logical threads lost here too** (1.29 against 1.22), even at 88 % CPU where the
decode-era explanation -- readers fighting compute for cores -- no longer applies.
The MXFP4 decode is ALU- and cache-bound rather than latency-bound, so SMT only
halves L1/L2 per thread.

### And the utilization numbers that led here were not measurements

The 37 % -> 88 % CPU that justified calling the engine compute-bound was two
instantaneous 30-second samples from two different runs, placed next to a wall-clock
average. With the same compute and 3x the utilization the wall time should have
fallen 3x; it rose. The kernel-copy explanation offered for that does not survive
arithmetic either: 600 GB copied over 627 s is ~60 s of memcpy, 1-2 % of eight cores,
not 58 points.

So the engine now reports, per chunk, run-averaged CPU split into user and kernel,
and the seconds spent BLOCKED in `layer_wait`. That last one matters because the hit
counter cannot see it: in streaming mode `tier_get` issues no read, records a hit, and
then blocks for as long as the disk takes -- a 99.8 % hit rate is compatible with
waiting on I/O the whole time.

The original 3-4x projection was arithmetic from two measured rates, not a result --
43 layers x 3.55 GB per 256-token chunk at 1514 MB/s would be 0.39 s/token against
the 1.32-1.78 observed. Whether the drive holds that rate across 152 GB per chunk,
and whether compute then becomes the wall, is the next thing to measure and not
something to claim here.

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
