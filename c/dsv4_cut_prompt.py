#!/usr/bin/env python3
"""Cut a captured prompt at the system boundary, and prove the cut is safe.

A snapshot built from the whole first request diverges as soon as a new session
opens with a different first message -- at token ~20000, which costs a full
checkpoint interval to re-prefill. Cutting at the first `<|User|>` caches only the
system prompt, so EVERY future session hits from token 0.

Why the cut is safe at all: `<|User|>` is an added token, which tok.h (and the HF
tokenizer) treat as atomic, so pre-tokenization splits there and no BPE merge can
span the boundary. That makes the truncated text's token sequence a true prefix of
the full one. This is not obvious and it is not true of an arbitrary character
offset, so it is checked here rather than assumed -- a silent mismatch would only
surface after hours of prefill.

    python3 cut_prompt.py <captured> <out> [model dir]
"""
import sys, pathlib
from tokenizers import Tokenizer

"""DeepSeek writes its role markers with FULLWIDTH VERTICAL LINE (U+FF5C), not the
ASCII pipe: the token is `<｜User｜>`. The ASCII spelling simply never matches,
so both are tried and the ASCII one is only a fallback for another renderer."""
MARKERS = ("<｜User｜>", "<|User|>")

def main(src, dst, model):
    text = pathlib.Path(src).read_text(encoding="utf-8")
    i = -1
    for m in MARKERS:
        i = text.find(m)
        if i >= 0:
            break
    if i < 0:
        sys.exit("none of %s found in %s: nothing to cut at" % (MARKERS, src))
    head = text[:i]

    tok = Tokenizer.from_file(str(pathlib.Path(model) / "tokenizer.json"))
    ids_full = tok.encode(text, add_special_tokens=False).ids
    ids_head = tok.encode(head, add_special_tokens=False).ids
    if ids_full[:len(ids_head)] != ids_head:
        sys.exit("the cut is NOT a token prefix; refusing to build a cache that "
                 "would silently miss")

    pathlib.Path(dst).write_text(head, encoding="utf-8")
    print("full   : %d bytes, %d tokens" % (len(text.encode()), len(ids_full)))
    print("system : %d bytes, %d tokens  -> %s" % (len(head.encode()), len(ids_head), dst))
    print("checked: the system tokens are an exact prefix of the full prompt")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3
         else "C:/Users/Gus/ai/models/DeepSeek-V4-Flash-0731")
