# DeepSeek-V4-Flash — an independent port (`c/dsv4_port.c`)

A second, self-contained implementation of **DeepSeek-V4-Flash** (284B total /
13B active, 1M context), written from the reference implementation upward with
every primitive validated in isolation before assembly.

> **Read this first.** `c/deepseek_v4.c` is the *production* V4 engine and this
> is not it. This port was written independently and in parallel; upstream's
> engine is further along (unit build, DSpark in `c/deepseek_v4_dspark.inc`,
> `deepseek_v4` arch in `openai_server.py`). What this file is still good for:
>
> - **the pre-tokenizer finding** below, which is a real bug the production path
>   still has (see `pretok_chunk_dsv4` in `tok.h`, now shared by both);
> - a **per-primitive reference** of the new architecture, each piece pinned to a
>   measured error against DeepSeek's own reference code;
> - the measurement log in
>   [deepseek-v4-performance.md](deepseek-v4-performance.md), whose findings
>   apply to any CPU V4 engine.

It runs and generates text:

```
$ ./dsv4_port /models/DeepSeek-V4-Flash-0731 "La capital de Francia es" 8
config: 43 layers, dim 4096, 64 heads x 512, 256 experts top-6

La capital de Francia es una de las ciudades más visitadas del mundo

14 tokens in 16.8 s (0.83 tok/s)
```

On a Ryzen 5700U (Zen 2, **no AVX-512**) with the checkpoint on an NVMe. It also
speaks the mux serve protocol, so `openai_server.py` drives it over HTTP.

## Why the model is a good fit

Measured on the real checkpoint, not estimated:

| | GLM-5.2 | DeepSeek-V4-Flash |
|---|---|---|
| On disk | 372 GB | **156 GiB** |
| Resident | 17B → 9.9 GB in int4 | **8.67 GiB at native precision** |
| Expert traffic per token | ~11.4 GB | **~3.45 GB** |
| Routed experts | 256, top-8 | 256, **top-6** |

The experts are 97.5 % of the model (277B of 284B) and ship as **native MXFP4**:
e2m1 packed two nibbles per byte, with one ue8m0 scale per 32 values. That is
exactly what `matmul_mxfp4` already consumes for Kimi K3, so they stream straight
out of the Hugging Face shards **with no conversion**.

## The tokenizer bug, which is still live

DeepSeek does **not** resolve its pre-tokenizer as a single alternation the way
cl100k / o200k / kimi do. It chains a `Sequence` of three `Split` stages with
`behavior=Isolated`, and **the first one isolates digits**. That ordering is not
equivalent to folding everything into one regex. The minimal case:

```
"\  0"    DeepSeek -> "\", "  ", "0"        (the double space, ONE token)
          cl100k   -> "\", " ", " ", "0"
```

Splitting the digit off first leaves the run of spaces with nothing after it
inside its own chunk, so the run is grouped whole. `pretok_chunk_dsv4` implements
stage 0 (digits in groups of ≤3) and stage 1 (han + kana) as a pre-pass and hands
the gaps to `pretok_chunk`, whose alternation matches stage 2 on everything
measured.

This matters beyond this port: `tok.h` had no V4 pre-tokenizer at all, and the
production engine calls `tok_encode`, so DeepSeek prompts were falling through to
the generic path. Verified against `tokenizers`: **399 cases, 0 differences**,
including the case above.

*Two other patches this port used to carry are now upstream's and better:* the
old-format merges (`"left right"` as one string) and the explicitly-tagged
dtypes in `st.h` — upstream gives each dtype its own code, accepts the
`F8_E4M3FN`/`F8_E8M0FNU`/`U64` aliases, and refuses them in the float readers.

## What the architecture adds

None of this exists in the GLM engine. Every piece is validated against
DeepSeek's reference implementation on the real checkpoint.

| piece | file | verification |
|---|---|---|
| **mHC** — residual `[b,s,4,dim]` with Sinkhorn | `dsv4_math.h` | rel err ~1e-7 |
| **CSA** — Compressor with a learned gate | `dsv4_math.h` | error 0.00e+00 |
| **Indexer** over the compressed KV | `dsv4_math.h` | 0/1536 indices differ |
| `sparse_attn` with a per-head sink | `dsv4_math.h` | 0 outside 1 bf16 ULP |
| Interleaved RoPE, forward and inverse | `dsv4_math.h` | error 0.00e+00 |
| FP8-e4m3 with **UE8M0** scales | `dsv4_fp8.h` | 4.82e-07 |
| `sqrtsoftplus` router + hash routing | `dsv4_moe.h` | 0/48 indices differ |
| Attention block (all 3 kinds) | `dsv4_attn.h` | 3.6e-03 to 4.3e-03 |
| Incremental **decode** path | `dsv4_decode.h` | 3.8e-03 to 5.9e-03 |
| MoE block, 256 MXFP4 experts | `dsv4_moe.h` | 1.67e-03 |

The 1e-3 figures are bf16 accumulation (ε = 3.9e-3), the same band the reference
model's own layers land in.

### The dense set: colibrì was refusing it on purpose

`colibri.c` named this model explicitly:

> *"DeepSeek-V4 ships the SAME weight layout (FP8 E4M3, 128x128 blocks) with
> UE8M0 [...] recognizing the signature and refusing is safer than misreading"*

`fmt=8` reads that geometry but expected the scale in f32, the way Z.ai publishes
it; DeepSeek publishes it as 1-byte UE8M0. `dsv4_fp8.h` was the missing decoder —
upstream has since added UE8M0 to `fmt=8` itself, which is the better home for it.

Watch the **OCP E4M3-FN** convention: `exp==0xF` is *not* reserved for infinity,
only `mant==0x7` is, so the largest finite value is 448. Treating it as IEEE
would silently lose the format's top magnitudes.

## Memory split

This is the central design decision, and the reason it fits at all:

```
resident in RAM     8.67 GiB   attention, norms, routers, shared experts,
                               embeddings and lm_head — IN THEIR NATIVE FORMAT
streamed          137.10 GiB   the 11,008 routed experts, off the NVMe, with LRU
```

The descriptors in `dsv4_weight.h` let the matmul read FP8/BF16/MXFP4 directly
and **never materialize the dequantized matrix**. Dequantizing the dense set to
f32 would be 26.8 GiB and would not fit. It is the same decision colibrì makes
with its `QT` struct.

`DSV4_LOAD` selects how the bytes get there (`read` / `dense` / `all`) and trades
speed for private memory; see the performance log for the measurements.

## MTP / DSpark, and why this port skips it

4,705 tensors (6.5 %). Three blocks with `markov_head` and `confidence_head` —
the MTP stack *is* the DSpark implementation. `config.json` describes it fully:

```
dspark_block_size: 5          num_nextn_predict_layers: 1
dspark_target_layer_ids: [40, 41, 42]
dspark_markov_rank: 256       dspark_noise_token_id: 128799
```

**It fits in memory**: of its 10.12 GiB on disk only **0.55 GiB is dense**
(attention, `shared_experts`, `gate`, the `hc_*` params and the two heads); the
other 9.56 GiB are experts and stream like the rest. Resident would go from 8.67
to 9.22 GiB. Each block is an ordinary block — attention + a 256-expert MoE +
shared expert — so it reuses the already-validated primitives; what is new is
`markov_head`, `confidence_head` and the acceptance loop.

With greedy decode, acceptance is **exact and trivial**: accept the longest
prefix whose draft tokens match the main model's argmax, no rejection sampling
and no confidence threshold. The validation criterion is as good as it gets — the
text has to come out identical.

**But it does not pay off on this host.** Measured, not assumed: the I/O saving
is *negative* with a 512-slot cache, and the draft costs three extra layers of
expert reads per cycle. Worse, after the kernels were fixed the engine sits on
its AVX2 roofline 47 % of the time, and speculative decoding spends FLOPs to buy
fewer sequential steps. The full numbers are in the performance log. Upstream's
engine does implement DSpark, which is the right call for hosts where FLOPs are
cheaper than this one's.

Two traps for anyone implementing it: **there are 3 blocks, not 1** —
`num_nextn_predict_layers: 1` in `config.json` does not correspond to
`n_mtp_layers` in `model.py`, and the count has to be derived from the checkpoint
(`mtp.0` has `main_proj`/`main_norm`, `mtp.2` has
`norm`/`markov_head`/`confidence_head`/`hc_head_*`, `mtp.1` has neither). And
**DeepSeek does not publish the acceptance loop**: `generate.py` is a plain
autoregressive decode, so the reference repo gives you the drafting but not the
verification.

## HTTP: the mux serve protocol

`dsv4_port.c` implements the same line protocol as `colibri.c` and `inkling.c`
(`SUBMIT`/`STOP`/`CANCEL` → `DATA`/`DONE`/`ERROR`), so `openai_server.py` drives
it unchanged. Verified end to end against the 284B checkpoint:
`/v1/completions`, `/v1/chat/completions` with and without streaming, multi-turn,
and `/v1/messages`.

One KV slot only. Prefix reuse across turns is implemented — the engine records
the ids it has fed and skips the common prefix — which is what makes multi-turn
usable: without it, a third turn over 300 tokens of history costs ~420 s before
the first word, because prefill runs one token at a time.

When the prompt **diverges** instead of extending, the state is rebuilt from
scratch. Rolling back is not possible here: the ring KV could be rewound, but the
CSA compressors and the indexer accumulate an in-progress block that cannot be
un-accumulated. That is the same wall that closed off DSpark.

The checkpoint ships no `chat_template.jinja`, but its vocabulary carries the
V3/R1 role tokens (`<|User|>` 128803, `<|Assistant|>` 128804), which is what the
gateway renders.

## A caveat about the validation criterion

The router's top-k is **discrete**, and the ~1e-3 of bf16 rounding that attention
accumulates is enough to flip a near-tied token to a different expert: measured,
3 of 448. That is not a defect — any implementation differing by one ULP does the
same — but it means teacher-forcing against `transformers` **will not give 32/32
exact**, and the acceptance criterion has to be set knowing that. It is the same
problem this repo already documents for its own DSA top-k.

## Still missing

- **`coli` integration**: the launcher does not know this binary; the gateway has
  to be pointed at it by hand with `--engine`.
- **Real streaming**: colibrì's own thread pool, `experts_apply_union`, PILOT,
  `.coli_usage`, `O_DIRECT`, multi-drive. Model-agnostic infrastructure that
  already exists. Demonstrated on a reduced model that the semantics do not
  change (logits bit-identical with a 3-of-8 expert cache).
- **Long context**: CSA/HCA exist for 1M tokens; what has been exercised here
  reaches 128.
