#!/usr/bin/env python3
"""Cut a captured prompt at the last STABLE byte, and prove the cut is safe.

The cache is a strict prefix, so it can only extend as far as the first thing in
the prompt that changes between sessions. Two of those, both measured on the real
agent prompt:

  * the first role marker `<｜User｜>` -- the user's message. Caching past it means a
    new session with a different first message diverges there.
  * `Today is <date>, and the current working directory is '<cwd>'` -- volatile
    every single day. It sits at token 19701 of 20249 (97.3 % in), so cutting
    before it keeps 19701 tokens and leaves 548 to prefill per session (~10 min at
    the measured 1.12 s/token). Caching past it would make the whole cache expire
    at midnight.

Anything stable that happens to sit AFTER a volatile line is unreachable -- that is
what a prefix means, not an oversight.

Why the cut needs checking: a byte offset is not necessarily a token boundary, and
byte-level BPE merges can span it, so the truncated text's tokens would not be a
prefix of the full one. The role marker is safe because it is an added token
(atomic to pre-tokenization); an arbitrary offset is not. So the cut is verified,
and walked backwards until it holds. A silent mismatch here would only surface
after hours of prefill.

    python3 cut_prompt.py <captured> <out> [--at MARKER] [--model DIR]

Default MARKER is `Today is`, the earliest volatile string in this agent's prompt.
Use `--at '<｜User｜>'` to cut at the end of the system prompt instead.
"""
import sys, pathlib
from tokenizers import Tokenizer

MODEL = "C:/Users/Gus/ai/models/DeepSeek-V4-Flash-0731"
ROLE_MARKERS = ("<｜User｜>", "<|User|>")   # U+FF5C, not the ASCII pipe


def safe_cut(tok, text, off, ids_full, max_back=64):
    """The longest prefix of `text` ending at or before `off` whose tokens are a
    true prefix of the whole prompt's."""
    for back in range(max_back):
        head = text[:off - back]
        ids = tok.encode(head, add_special_tokens=False).ids
        if ids_full[:len(ids)] == ids:
            return head, ids, back
    return None, None, None


def main(argv):
    src, dst = argv[0], argv[1]
    marker, model = "Today is", MODEL
    i = 2
    while i < len(argv) - 1:
        if argv[i] == "--at":
            marker = argv[i + 1]
        elif argv[i] == "--model":
            model = argv[i + 1]
        else:
            sys.exit("unknown option %r" % argv[i])
        i += 2

    text = pathlib.Path(src).read_text(encoding="utf-8")
    cands = [marker] if marker != "role" else list(ROLE_MARKERS)
    off = -1
    for m in cands:
        off = text.find(m)
        if off >= 0:
            marker = m
            break
    if off < 0:
        sys.exit("marker %r not found in %s: nothing to cut at" % (marker, src))

    tok = Tokenizer.from_file(str(pathlib.Path(model) / "tokenizer.json"))
    ids_full = tok.encode(text, add_special_tokens=False).ids
    head, ids, back = safe_cut(tok, text, off, ids_full)
    if head is None:
        sys.exit("no token-aligned cut within 64 bytes before %r; refusing to build "
                 "a cache that would silently miss" % marker)

    pathlib.Path(dst).write_text(head, encoding="utf-8")
    print("full    : %d bytes, %d tokens" % (len(text.encode()), len(ids_full)))
    print("cached  : %d bytes, %d tokens (%.1f%%) -> %s"
          % (len(head.encode()), len(ids), 100.0 * len(ids) / len(ids_full), dst))
    print("per call: %d tokens left to prefill = %.0f min at 1.12 s/token"
          % (len(ids_full) - len(ids), (len(ids_full) - len(ids)) * 1.12 / 60))
    print("cut at  : %r%s" % (marker, "" if back == 0 else
                              " (walked back %d bytes to a token boundary)" % back))
    print("checked : those tokens are an exact prefix of the full prompt")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
