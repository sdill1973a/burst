#!/usr/bin/env python3
"""open_codec — frame 0x03, the OPEN codec: any small GGUF model via llama.cpp + mismatch-resistant coding.

Bellard's ts_sms is the better compressor but ships only for x86. This one runs wherever llama.cpp runs (phones
included) and survives the two ends computing slightly different probabilities (Hu & Tang, arXiv 2601.17684):
each token is coded by a probability bucket plus just enough bits of a fixed random code to tell it from every other
token the far end could confuse it with, given a certified mismatch factor c.

Wire: [0x03][model id][c×10][tail id][varint n_tokens][MRC bits]. Both ends need the same GGUF (sha-pinned below), the same
PRIME, the same bucket table (buckets_v1.json, trained once, shipped with the code), the same code seed.
The model id byte is advisory: every encoder in use (this file, the Android app) writes 1 and no decoder reads it; the
model actually used is MODEL_PATH (the base model). It is left at 1 so 0x03 frames stay byte-identical across ends.

This module also holds what frame 0x08 (det_codec.py) shares with 0x03: PRIME, the learned tail (TAIL_N, prime_for),
MODEL_PATH, and detok.
Measured 2026-10-21 (qwen2.5-0.5b q8_0): x86 batched encoder vs x86 step decoder 100/100 SMS exact at 2.2x; vs ARM
kernels (qemu) the probability gap reached 2.86x, so the default c is 3.3.
"""
import hashlib, json, os, sys
import numpy as np
import mrc

MODELS = {1: ("qwen2.5-0.5b-instruct-q8_0.gguf", "sha256:ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"),
          2: ("qwen2.5-0.5b-base-q8_0.gguf", "sha256:d5163baa48d1797b1d274f271aa1fd152e3f07248011f7ea386f3d0667dce535")}
# The base model is the default: +2% on real 0x08 traffic over instruct, same speed (measured 10-25). Same geometry.
MODEL_PATH = os.environ.get("BURST_GGUF", "qwen2.5-0.5b-base-q8_0.gguf")
PRIME = "The following is a short plain-English message sent between two people over a radio.\n\n"
BUCKETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckets_v1.json")
_llm = None


def llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=max(1, (os.cpu_count() or 2) // 2),
                     logits_all=True, verbose=False)
    return _llm


def _sm(u):
    u = np.asarray(u, np.float64); e = np.exp(u - u.max()); return e / e.sum()


_warned = False


def _coder(c):
    """The bucket table is part of the 0x03 wire: buckets_v1.json must sit beside this file on both ends. Without it the
    coder falls back to mrc's built-in prior — still self-consistent, but NOT interoperable with ends that have the file."""
    global _warned
    freq = None
    if os.path.exists(BUCKETS):
        freq = json.load(open(BUCKETS))["freq"]
    elif not _warned:
        _warned = True
        print(f"open_codec: {BUCKETS} not found — using the built-in bucket prior; 0x03 frames will not interoperate "
              "with ends that have the table", file=sys.stderr)
    return mrc.MRC(llm().n_vocab(), c=c, bucket_freq=freq)


def _wv(n, out):
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return


def _rv(buf, i):
    shift = val = 0
    while True:
        b = buf[i]; i += 1; val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


# THE LEARNED TAIL: both ends keep the messages they have confirmed (exact on arrival,
# or fixed by the reader and proven by the hash; the sender keeps what it sent). The last TAIL_N of them, in order, are
# appended to the PRIME — shared history was the cheapest prime in the arms test (+5.6% at a sixth of a pack's compute).
# The header carries ONE byte: the low byte of sha256 over that tail, so the far end can tell which history was used and
# fall back to the base prime (tail id 0x00) when its own tail does not match. Divergence is therefore a smaller burst
# lost, never a wrong decode: the hash still rules.
TAIL_N = int(os.environ.get("BURST_TAIL_N", "4"))
C_DEFAULT = float(os.environ.get("BURST_C", "3.3"))     # c per pair: 3.3 = emulated-ARM worst case; ~1.6 measured on same-build pairs


def tail_id(tail):
    return (hashlib.sha256("\n".join(tail).encode()).digest()[0] | 1) if tail else 0     # never 0 for a real tail


TAIL_BYTES = 1500    # the learned tail is capped: oldest messages drop until it fits (10-24: two 4,000-char pastes in
                     # the tail pushed every prime past the 2,048-token context — 0x08 refused both ways). Codec.kt: same rule.


def fit_tail(tail):
    t = list(tail)
    while t and sum(len(m.encode()) for m in t) > TAIL_BYTES:
        t = t[1:]
    return t


def prime_for(tail):
    tail = fit_tail(tail) if tail else tail
    return PRIME + ("".join(t.rstrip("\n") + "\n\n" for t in tail) if tail else "")


def crunch(text, c=None, model_id=1, tail=()):
    c = C_DEFAULT if c is None else c
    tail = list(tail)[-TAIL_N:]
    L = llm(); pt = L.tokenize(prime_for(tail).encode(), add_bos=False); toks = L.tokenize(text.encode(), add_bos=False)
    L.reset(); L.eval(pt + toks); S = np.array(L.scores[:len(pt) + len(toks)])
    probs = [_sm(S[len(pt) - 1 + i]) for i in range(len(toks))]
    it = iter(probs)
    data, _ = _coder(c).encode(toks, lambda ctx: next(it), [0])
    head = bytearray([0x03, model_id, round(c * 10), tail_id(tail)]); _wv(len(toks), head)
    return bytes(head) + data


def detok(L, ids):
    """ids -> bytes, exactly as the phone's BurstNative.detokenize (burst_jni.cpp): each piece into a 256-byte buffer.
    llama-cpp-python's own detokenize uses 32 bytes and silently cuts Qwen's long whitespace tokens: a pasted ASCII-art
    link came back short and failed its check, and every carried link after it with it (measured 10-24)."""
    import ctypes
    import llama_cpp
    buf, out = ctypes.create_string_buffer(256), bytearray()
    for t in ids:
        k = llama_cpp.llama_token_to_piece(L._model.vocab, llama_cpp.llama_token(int(t)), buf, 256, 0, False)
        if k > 0:
            out += buf.raw[:k]
    return bytes(out)


def uncrunch(b, marked=False, forced=None, tails=()):
    """text; with marked=True: (text, spots) for every token the decoder doubted. tails = candidate learned tails (newest
    first); the one whose id matches the header is used, else the base prime (and the far end's tail is reported lost)."""
    assert b[0] == 0x03
    model_id, c, tid = b[1], b[2] / 10, b[3]
    n, i = _rv(b, 4)
    tail = next((list(t)[-TAIL_N:] for t in tails if tail_id(list(t)[-TAIL_N:]) == tid), []) if tid else []
    if tid and not tail:
        print(f"open_codec: far end used a history tail (id {tid:02x}) I do not hold — decoding on the base prime", file=sys.stderr)
    L = llm(); pt = L.tokenize(prime_for(tail).encode(), add_bos=False)
    L.reset(); L.eval(pt); st = {"n": len(pt)}

    def probs(ctx):
        if len(ctx) > 1:
            L.eval([ctx[-1]]); st["n"] += 1
        return _sm(L.scores[st["n"] - 1])
    toks, marks = _coder(c).decode_marked(b[i:], n, probs, [0], forced=forced)
    text = detok(L, toks).decode("utf-8", errors="replace")
    if not marked:
        return text
    # the sus spots, with the pop-up's options: [{"i", "start", "end", "why", "guess", "alts": [str...]}]
    spots, at = [], 0
    for i, (t, m) in enumerate(zip(toks, marks)):
        piece = detok(L, [t]).decode("utf-8", errors="replace"); end = at + len(piece)
        if m:
            spots.append({"i": i, "start": at, "end": end, "why": m["why"], "guess": piece,
                          "alts": [(a, detok(L, [a]).decode("utf-8", errors="replace")) for a in m.get("alts", [])]})
        at = end
    return text, spots


def render(text, spots, open_="\u27e6", close="\u27e7"):
    """the text with every sus token wrapped: ⟦like this⟧ — the reader fills the gap, the hash says if they got it."""
    out, at = [], 0
    for sp in sorted(spots, key=lambda s: s["start"]):
        a, b = sp["start"], sp["end"]
        if a < at: continue
        out.append(text[at:a]); out.append(open_ + text[a:b] + close); at = b
    out.append(text[at:]); return "".join(out)


def build_buckets(texts, out=BUCKETS):
    """Train the bucket table once from sample text; ship the JSON with the code."""
    L = llm(); pt = L.tokenize(PRIME.encode(), add_bos=False)
    base = mrc.MRC(L.n_vocab()); hist = np.ones(base.K)
    for t in texts:
        toks = L.tokenize(t.encode(), add_bos=False); L.reset(); L.eval(pt + toks); S = np.array(L.scores[:len(pt) + len(toks)])
        for i, tk in enumerate(toks):
            hist[base.bucket(_sm(S[len(pt) - 1 + i])[tk])] += 1
    json.dump({"version": 1, "prime": PRIME, "model": MODELS[1][0], "freq": [float(x) for x in hist]}, open(out, "w"))
    return hist


if __name__ == "__main__":
    import random, re, base64
    if len(sys.argv) == 3 and sys.argv[1] == "c":          # CLI: c "text" -> base64 bytes ; d <base64> -> text
        print(base64.b64encode(crunch(sys.argv[2])).decode())
        sys.exit()
    if len(sys.argv) == 3 and sys.argv[1] == "d":
        sys.stdout.write(uncrunch(base64.b64decode(sys.argv[2])))
        sys.exit()
    if sys.argv[1:] == ["build-buckets"]:                 # retrain buckets_v1.json (UCI SMS ham + Pride and Prejudice);
                                                          # the shipped table came from this. Changing it changes the 0x03 wire.
        sms = [l.split("\t", 1)[1].strip() for l in open(os.environ.get("SMS_CORPUS", "SMSSpamCollection"),
               encoding="utf-8", errors="replace") if l.startswith("ham\t")]
        random.Random(20261020).shuffle(sms)
        txt = open(os.environ.get("PROSE_CORPUS", "pp.txt"), encoding="utf-8").read().replace("\r", "")
        body = txt[txt.rindex("CHAPTER I"):txt.index("*** END OF")]
        paras = [re.sub(r"\s+", " ", p).strip() for p in body.split("\n\n")]
        paras = [p for p in paras if 250 <= len(p) <= 700]; random.Random(1813).shuffle(paras)
        h = build_buckets(sms[800:900] + paras[100:140])
        print("buckets built:", int(h.sum()), "tokens")
    else:
        for t in ["Checking in. How goes it? Sound off.",
                  "Water is up over the low crossing on the creek road. Trucks are getting through but cars should take the highway. "
                  "Power is out on the east side of town and the co-op says it will be back by morning. Nobody is hurt."]:
            b = crunch(t); back = uncrunch(b)
            print(f"{len(t):4d} chars -> {len(b):4d} B  exact={back == t}")
