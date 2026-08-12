#!/usr/bin/env python3
"""Cut a captured prompt at the last STABLE byte, and prove the cut is safe.

The cache is a strict prefix, so it can only extend as far as the first thing in the
prompt that changes between sessions. Two of those, both measured on the real agent
prompt (20249 tokens):

  * the first role marker `<｜User｜>` -- the user's message. Caching past it means a
    new session with a different first message diverges there.
  * `Today is <date>, and the current working directory is '<cwd>'` -- volatile every
    single day. It sits at token 19701, 97.3 % of the way in, so cutting before it
    costs 2.7 % of the prefill and buys a cache that does not expire at midnight.

EVERYTHING HERE IS BYTES, and that is not fussiness. The first version read and wrote
with pathlib's text mode: `write_text` translates \\n to \\r\\n on Windows, so the file
it produced carried 597 carriage returns the engine's dump did not have -- 1120 extra
tokens, and a prompt that could never match what the agent sends. Six hours of prefill
would have built a cache with a 0 % hit rate.

The check missed it because it verified the string in memory and then wrote the file
through a translating writer. So the verification now RE-READS THE FILE IT WROTE and
tokenizes those bytes. Verify the artifact, not the intention.

    python3 cut_prompt.py <captured> <out> [--at MARKER] [--model DIR]
"""
import sys, pathlib
from tokenizers import Tokenizer

MODEL = "C:/Users/Gus/ai/models/DeepSeek-V4-Flash-0731"
# DeepSeek spells its role markers with FULLWIDTH VERTICAL LINE (U+FF5C), not the
# ASCII pipe. The ASCII form never matches; it is only a fallback for another renderer.
ROLE_MARKERS = ("<｜User｜>", "<|User|>")


def ids_of(tok, raw):
    return tok.encode(raw.decode("utf-8"), add_special_tokens=False).ids


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

    raw = pathlib.Path(src).read_bytes()
    cands = [marker] if marker != "role" else list(ROLE_MARKERS)
    off = -1
    for m in cands:
        off = raw.find(m.encode("utf-8"))
        if off >= 0:
            marker = m
            break
    if off < 0:
        sys.exit("marker %r not found in %s: nothing to cut at" % (marker, src))

    tok = Tokenizer.from_file(str(pathlib.Path(model) / "tokenizer.json"))
    ids_full = ids_of(tok, raw)

    # walk back to a byte offset that is also a token boundary (and a valid UTF-8 one)
    head = back = None
    for b in range(64):
        cand = raw[:off - b]
        try:
            ids = ids_of(tok, cand)
        except UnicodeDecodeError:
            continue                      # cut inside a multi-byte character
        if ids_full[:len(ids)] == ids:
            head, back = cand, b
            break
    if head is None:
        sys.exit("no token-aligned cut within 64 bytes before %r; refusing to build a "
                 "cache that would silently miss" % marker)

    pathlib.Path(dst).write_bytes(head)

    # THE check: re-read what landed on disk. Anything that mangles bytes on the way
    # out -- newline translation, encoding, a stray editor -- has to fail here.
    back_raw = pathlib.Path(dst).read_bytes()
    if back_raw != head:
        sys.exit("%s does not contain the bytes that were written (%d vs %d): refusing"
                 % (dst, len(back_raw), len(head)))
    ids_file = ids_of(tok, back_raw)
    if ids_full[:len(ids_file)] != ids_file:
        sys.exit("the tokens of %s are NOT a prefix of the full prompt: refusing" % dst)
    if back_raw.count(b"\r") != raw[:len(back_raw)].count(b"\r"):
        sys.exit("%s has different carriage returns than the capture: refusing" % dst)

    print("full    : %d bytes, %d tokens" % (len(raw), len(ids_full)))
    print("cached  : %d bytes, %d tokens (%.1f%%) -> %s"
          % (len(back_raw), len(ids_file), 100.0 * len(ids_file) / len(ids_full), dst))
    print("per call: %d tokens left to prefill = %.0f min at 1.12 s/token"
          % (len(ids_full) - len(ids_file), (len(ids_full) - len(ids_file)) * 1.12 / 60))
    print("cut at  : %r%s" % (marker, "" if back == 0 else
                              " (walked back %d bytes to a token boundary)" % back))
    print("checked : re-read from disk; %d CR in both; tokens are an exact prefix"
          % back_raw.count(b"\r"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
