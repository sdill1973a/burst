#!/usr/bin/env python3
"""wire 2 proofs (mock model, no weights). `python3 test_wire2.py [false-accept-N]`"""
import hashlib, sys
import numpy as np
import det_codec as dc


class Counting(dc.MockModel):
    n_vocab = 512; resets = 0
    def reset(self, p):
        self.resets += 1; return super().reset(p)
    def tokenize(self, text):                           # bytes in, 3-hex-digit ids out: detokenize below is injective,
        return list(text.encode())                      # so wrong decodes are distinct texts, i.e. independent trials
    def detokenize(self, ids):
        return bytes(ids).decode("latin-1") if all(i < 256 for i in ids) else "".join("%03x" % i for i in ids)
    def _logits(self):                                  # small vocab: 200k decodes stay cheap on a shared box
        key = hashlib.sha256(repr((self.seed, self.ctx)).encode()).digest()
        return np.random.default_rng(int.from_bytes(key[:8], "little")).standard_normal(512).astype(np.float32) * 3


def refused(f):
    try:
        f()
    except dc.DecodeFailed as e:
        return str(e)
    raise AssertionError("was accepted")


def avail_gb():
    return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20


m = Counting()
H = ["first line", "second line"]
# old wire: unchanged, still decodes
old = dc.crunch("old frame still works", m, tail=H)
assert old[0] == 0x08 and dc.uncrunch(old, m, tails=[H]) == "old frame still works"
print("old 0x08 frame: decoded exact")
# new wire round trip, one candidate decoded
f = dc.crunch2("wire two works", m, tail=H)
assert f[0] == 0x0E
wrong = [["x%d" % i] for i in range(8)]
m.resets = 0
assert dc.uncrunch2(f, m, tails=wrong + [H]) == "wire two works"
print(f"0x0E frame: exact; {len(wrong) + 2} candidates held, {m.resets} decoded")
m.resets = 0
print("absent history refused:", refused(lambda: dc.uncrunch2(dc.crunch2("z", m, tail=["only sender has this"]), m, tails=wrong)))
# splice
A, B = ["part one of A ", "part two of A"], ["part one of B ", "part two of B"]
wa, wb = dc.check3("".join(A)), dc.check3("".join(B))
a0 = dc.link2(0x5A, 0, 2, dc.crunch2(A[0], m), wa); b1 = dc.link2(0x5A, 1, 2, dc.crunch2(B[1], m))
mid, i, t, whole, inner = dc.parse_link2(a0); assert (mid, i, t, whole) == (0x5A, 0, 2, wa)
mid1, i1, t1, w1, inner1 = dc.parse_link2(b1); assert (mid1, i1, w1) == (0x5A, 1, None)
parts = [dc.uncrunch2(inner, m), dc.uncrunch2(inner1, m)]
print(f"splice A0+B1 links individually verify: {parts!r}")
print("splice refused:", refused(lambda: dc.join2(parts, whole)))
assert dc.join2(A, wa) == "".join(A)
print("honest chain A0+A1: joined exact")
_ = wb
N = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if N:
    target = "the message actually sent"
    c16, c24 = dc._check(target), dc.check3(target)
    f16 = f24 = 0; seen = set()
    body = bytes(np.random.default_rng(1).integers(0, 256, 12, dtype=np.uint8))
    for k in range(N):
        if k % 5000 == 0 and avail_gb() < 3:
            sys.exit(f"STOPPED at {k}: available memory under 3 GB")
        # a decode under a wrong history: the real decoder path, a distinct prime each trial
        text = dc._decode_with(body, 6, m, ["wrong history %d" % k])
        f16 += dc._check(text) == c16; f24 += dc.check3(text) == c24
        if k < 100000: seen.add(text)
    print(f"false accepts in {N} wrong-history decodes: 16-bit={f16} (expect {N / 65536:.2f})  24-bit={f24} (expect {N / 16777216:.4f}); distinct texts in first 100000: {len(seen)}")
