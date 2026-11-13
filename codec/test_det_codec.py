#!/usr/bin/env python3
"""test_det_codec — proves detrc + frame 0x08 both ways (every check here must be able to fail).

    python3 test_det_codec.py            coder units + edge tables + mock-model round trips + refusals (numpy only:
                                         no model, no llama.cpp; builds build/libdetrc.so with cc on first run)
    python3 test_det_codec.py --golden   also: the golden vectors the Android app checks itself against
                                         (vectors08.json): with the real model, every frame must be re-encoded
                                         byte-identical, decode exact under its history, and refuse when a byte flips.
                                         Needs BURST_GGUF, libdetlm.so (make -f Makefile.detlm) and llama-cpp-python.
                                         Vectors: $BURST_VECTORS, else ../android/app/src/main/assets/vectors08.json.
"""
import base64, json, os, random, sys, time
import numpy as np

import det_codec as dc

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL = 1 << dc.TOTAL_BITS
V = 151936
VECTORS = os.environ.get("BURST_VECTORS") or os.path.join(HERE, "..", "android", "app", "src", "main", "assets", "vectors08.json")

# a synthetic conversation for the mock model (the mock tokenizes bytes, so any text works): 64 messages, 1-3 phrases each
PHRASES = ["Checking in.", "How goes it?", "On my way, ETA 20 min.", "Copy that.", "Battery at 40%, heading back.",
           "Water is up over the low crossing on the creek road.", "Camp is at the second pullout past the bridge.",
           "Bring ice and the big tarp.", "Did you get my last message?", "Nothing came through on my end.",
           "Power is out on the east side of town.", "Nobody is hurt.", "Lat 44.4280, lon -110.5885.", "See you Sunday.",
           "Line one.\n\nLine  two, with a double space.", "Ünïcödé — and an emoji 🙂 too."]
_r = random.Random(20261024)
MSGS = [" ".join(_r.choice(PHRASES) for _ in range(_r.randint(1, 3))) for _ in range(64)]


def ok(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        raise SystemExit("FAILED: " + msg)


# ---------------------------------------------------------------------------------------------------------------- units
def test_freq_edges():
    print("freq tables (sum 2^24, every entry >= 1):")
    flat = dc.freq_table(np.zeros(V, np.float32))
    ok(flat.sum() == TOTAL and flat.min() >= 1 and flat.max() - flat.min() <= V, f"flat: min {flat.min()} max {flat.max()}")
    one = np.zeros(V, np.float32); one[5] = 1e4
    f = dc.freq_table(one)
    ok(f.sum() == TOTAL and f[5] == TOTAL - V + 1 and f.min() == 1, f"one certain token: freq {f[5]} = 2^24-V+1, rest 1")
    nan = np.full(V, np.nan, np.float32)
    f = dc.freq_table(nan)
    ok(f.sum() == TOTAL and f.min() >= 1, "all-NaN row is a valid (flat) table")
    inf = np.zeros(V, np.float32); inf[1] = np.inf; inf[2] = -np.inf; inf[3] = np.nan
    f = dc.freq_table(inf)
    ok(f.sum() == TOTAL and f[1] == TOTAL - V + 1 and f[2] == 1 and f[3] == 1, "+inf wins, -inf and NaN get the floor")
    den = np.full(V, 1e-40, np.float32)
    f = dc.freq_table(den)
    ok(f.sum() == TOTAL and f.min() >= 1, "denormal row is a valid table")
    # sharpness: cost under the table tracks -log2 softmax within the floor's tax
    rng = np.random.default_rng(1)
    worst = 0.0
    for scale in (0.5, 3.0, 10.0, 30.0):
        lg = (rng.standard_normal(V) * scale).astype(np.float32)
        f = dc.freq_table(lg)
        ok(f.sum() == TOTAL and f.min() >= 1, f"random scale {scale}: valid table")
        for s in [int(lg.argmax()), int(rng.integers(V)), int(lg.argmin())]:
            cost = -np.log2(f[s] / TOTAL); ideal = dc.ideal_bits(lg, s)
            worst = max(worst, cost - ideal)
    ok(worst < 0.05, f"table cost - ideal cost <= {worst:.4f} bits on the sampled tokens (floor tax ~0.013)")
    # the quantum: sub-2^-16 float noise cannot reach the table
    lg = (rng.standard_normal(V) * 3).astype(np.float32)
    lg2 = lg.copy(); lg2[100] = np.nextafter(lg2[100], np.float32(1e9))
    ok(np.array_equal(dc.freq_table(lg), dc.freq_table(lg2)), "a 1-ulp nudge below the Q16.16 quantum changes nothing")
    lg2[int(lg.argmax())] += 0.01
    ok(not np.array_equal(dc.freq_table(lg), dc.freq_table(lg2)), "a 0.01 change on the top token does (the function is not blind)")


def test_coder():
    print("range coder (explicit tables, V=151936):")
    rng = np.random.default_rng(2)
    for n, name in [(10000, "10k symbols, mixed sharpness"), (300, "300 symbols, floor-heavy")]:
        syms, tabs, ideal = [], [], 0.0
        for i in range(n):
            scale = [0.0, 2.0, 8.0, 30.0][i % 4] if name.startswith("10k") else 30.0
            lg = (rng.standard_normal(V) * scale).astype(np.float32)
            f = dc.freq_table(lg)
            s = int(lg.argmax()) if (i % 7) else int(rng.integers(V))     # mostly likely tokens, some freq-1 ones
            syms.append(s); tabs.append(f); ideal += -np.log2(f[s] / TOTAL)
        t0 = time.time()
        e = dc.Encoder()
        for f, s in zip(tabs, syms): e.put(f, s)
        data = e.finish(); t1 = time.time()
        d = dc.Decoder(data)
        back = [d.get(f) for f in tabs]
        ok(back == syms, f"{name}: round trip exact ({len(data)} B, enc {t1 - t0:.2f}s dec {time.time() - t1:.2f}s)")
        ok(len(data) * 8 <= ideal + 8 + n * 2 ** -20, f"{name}: {len(data) * 8} bits vs table-ideal {ideal:.1f} (+{len(data) * 8 - ideal:.2f})")
    # degenerate streams: all-certain (the carry/0xFF paths), a single symbol, empty
    one = np.zeros(V, np.float32); one[5] = 1e4; f = dc.freq_table(one)
    e = dc.Encoder()
    for _ in range(5000): e.put(f, 5)
    data = e.finish(); d = dc.Decoder(data)
    ok(all(d.get(f) == 5 for _ in range(5000)), f"5000 certain symbols -> {len(data)} B (ideal {5000 * -np.log2(f[5] / TOTAL):.1f} bits)")
    e = dc.Encoder(); e.put(f, 6); data = e.finish(); d = dc.Decoder(data)
    ok(d.get(f) == 6, f"one floor symbol -> {len(data)} B")
    e = dc.Encoder(); ok(e.finish() == b"", "empty stream -> 0 B")
    # a table that does not sum to 2^24 is refused by the C, not silently coded
    bad = f.copy(); bad[0] += 1
    try:
        dc.Encoder().put(bad, 0); ok(False, "bad table refused")
    except ValueError:
        ok(True, "a table whose sum is not 2^24 is refused")
    # garbage in: the decoder must return in-range symbols and never crash
    d = dc.Decoder(bytes(rng.integers(0, 256, 64, dtype=np.uint8)))
    got = [d.get(tabs[i % len(tabs)]) for i in range(200)]
    ok(all(0 <= g < V for g in got), "200 symbols from random bytes: all in range, no crash")
    d = dc.Decoder(b"")
    ok(0 <= d.get(tabs[0]) < V, "decoder on an empty body returns a symbol (the check refuses it later)")


# ----------------------------------------------------------------------------------------------------------- mock e2e
def test_mock_roundtrip():
    print("frame 0x08 round trip on the mock model, all %d messages, 4-message learned tails:" % len(MSGS))
    m = dc.MockModel(); total = 0; t0 = time.time()
    for i, text in enumerate(MSGS):
        tail = MSGS[max(0, i - 4):i]
        frame = dc.crunch(text, m, tail=tail)
        assert frame[0] == 0x08
        # the receiver holds its own history; offer newest-first candidates, in sync first then one lagging
        cands = [MSGS[max(0, i - 4):i], MSGS[max(0, i - 5):i - 1] if i else []]
        back = dc.uncrunch(frame, m, tails=cands)
        if back != text:
            ok(False, f"message {i} not exact")
        total += len(frame)
    ok(True, f"{len(MSGS)}/{len(MSGS)} exact, {total} B total, {time.time() - t0:.1f}s")
    # the receiver lagging by one (its newest candidate is not the sender's tail) still lands on the right one
    i = 40; frame = dc.crunch(MSGS[i], m, tail=MSGS[i - 4:i])
    back = dc.uncrunch(frame, m, tails=[MSGS[i - 3:i + 1], MSGS[i - 4:i], MSGS[i - 5:i - 1]])
    ok(back == MSGS[i], "receiver one message ahead: second candidate matches")
    back = dc.uncrunch(dc.crunch(MSGS[i], m, tail=()), m, tails=[MSGS[i - 4:i]])
    ok(back == MSGS[i], "sender with no history: base prime found as the last candidate")


def test_refusals():
    print("refusals (never shown as text):")
    m = dc.MockModel(); i = 50; text = MSGS[i]; tail = MSGS[i - 4:i]
    frame = dc.crunch(text, m, tail=tail)
    assert dc.uncrunch(frame, m, tails=[tail]) == text

    def refused(f, what, tails=(tail,), exact_allowed=False):
        """the contract: the exact text, or a refusal — never a wrong text shown"""
        try:
            got = dc.uncrunch(f, m, tails=tails)
        except dc.DecodeFailed as e:
            ok(True, f"{what}: refused ({str(e)[:50]})"); return
        if exact_allowed and got == text:
            ok(True, f"{what}: decoded EXACT (the flip stayed inside the final interval — same message, check passes)"); return
        ok(False, f"{what}: was SHOWN as {got!r}")

    body_at = dc._rv(frame, 3)[1]                         # first rc byte, after the varint token count
    for k in (body_at, body_at + 3, len(frame) // 2):
        f = bytearray(frame); f[k] ^= 0x40; refused(bytes(f), f"body byte {k} flipped")
    f = bytearray(frame); f[-1] ^= 0x40                   # the LAST byte is the loose one: the minimal flush picks any
    refused(bytes(f), f"last body byte flipped", exact_allowed=True)   # value in [low, low+range), so a flip can stay inside
    f = bytearray(frame); f[1] ^= 0x01; refused(bytes(f), "check byte flipped")
    f = bytearray(frame); f[3] ^= 0x01; refused(bytes(f), "token count changed")
    refused(frame[:-3], "frame truncated by 3 bytes")
    refused(frame[:3], "frame cut to the header")
    refused(b"\x08\x00\x00\xff\xff\xff\xff\x7f" + frame[4:], "varint token count 2^35 (over the cap, no model steps)")
    refused(b"\x07" + frame[1:], "wrong codec byte")
    tiny = bytearray(b"\x08\x00\x00"); dc._wv(2000, tiny); tiny += b"\x55\x55\x55"   # 10-30: 3-byte body, 2,000 tokens
    try:
        dc.uncrunch(bytes(tiny), m, tails=[tail]); ok(False, "tiny frame claiming 2,000 tokens: decoded")
    except dc.DecodeFailed as e:
        ok("implausible" in str(e), f"tiny frame claiming 2,000 tokens: refused before any model step ({str(e)[:44]})")
    refused(frame, "only wrong histories offered", tails=[MSGS[i - 5:i - 1], MSGS[i - 6:i - 2], MSGS[i - 3:i + 1]])
    random.seed(5)
    for k in range(3):
        refused(b"\x08" + bytes(random.randrange(256) for _ in range(20)), f"random 23-byte frame #{k}")


# ---------------------------------------------------------------------------------------------------- golden vectors
def test_golden():
    print(f"golden vectors (real model, detlm): {VECTORS}")
    if not os.path.exists(VECTORS):
        ok(False, "vectors08.json not found (set BURST_VECTORS)")
    vec = json.load(open(VECTORS))
    m = dc.DetlmModel(); t0 = time.time()
    for i, v in enumerate(vec):
        frame = base64.b64decode(v["frame_b64"])
        ok(m.tokenize(v["text"]) == v["tokens"], f"[{i}] tokenizer ids match")
        ok(dc.crunch(v["text"], m, tail=v["tail"]) == frame, f"[{i}] re-encoded byte-identical ({len(frame)} B)")
        h = v["history"]                                   # the receiver's candidates: newest window first, then older ones
        cands = [h[max(0, len(h) - 4 - k):len(h) - k] for k in range(dc.MAX_TAILS) if len(h) - k > 0]
        ok(dc.uncrunch(frame, m, tails=cands) == v["text"], f"[{i}] decodes exact, history found by the check")
        if i < 2:
            bad = bytearray(frame); bad[4 + (len(frame) - 4) // 2] ^= 0x5A
            try:
                dc.uncrunch(bytes(bad), m, tails=cands); ok(False, f"[{i}] flipped byte was SHOWN")
            except dc.DecodeFailed:
                ok(True, f"[{i}] flipped byte refused")
    big = bytes([0x08]) + frame[1:3] + bytes([0xF8, 0x0F]) + frame[4:]      # claims 2,040 tokens: over the context
    try:
        dc.uncrunch(big, m, tails=[["x" * 1400]]); ok(False, "over-context token count was SHOWN")
    except dc.DecodeFailed:
        ok(True, "a token count that cannot fit the context is refused before decoding")
    ok(True, f"{len(vec)} vectors, {time.time() - t0:.1f}s, detlm {m.version}")


if __name__ == "__main__":
    test_freq_edges(); test_coder(); test_mock_roundtrip(); test_refusals()
    if "--golden" in sys.argv:
        test_golden()
    print("ALL PASSED")
