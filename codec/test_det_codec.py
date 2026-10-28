#!/usr/bin/env python3
"""test_det_codec — proves detrc + frame 0x08 both ways (an organ must be able to fail).

    python3 test_det_codec.py            coder units + edge tables + mock round trip over bench_convo.json + refusals
    python3 test_det_codec.py --llama    also: bytes on air for the last 32 bench messages with llama.cpp as the stand-in
                                         model (prime = open_codec.prime_for(previous 4)), vs the 912 B baseline, and the
                                         x86-vs-aarch64 identity gate on REAL logits (detrc_test file ... under qemu)
Run with services/.venv-burst/bin/python from the runtime (llama_cpp lives there).
"""
import json, os, random, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import det_codec as dc

TOTAL = 1 << dc.TOTAL_BITS
BENCH = json.load(open(os.path.join(HERE, "bench_convo.json")))
V = 151936


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
    print("frame 0x08 round trip on the mock model, all %d bench messages, 4-message learned tails:" % len(BENCH))
    m = dc.MockModel(); total = 0; t0 = time.time()
    for i, text in enumerate(BENCH):
        tail = BENCH[max(0, i - 4):i]
        frame = dc.crunch(text, m, tail=tail)
        assert frame[0] == 0x08
        # the receiver holds its own history; offer newest-first candidates, in sync first then one lagging
        cands = [BENCH[max(0, i - 4):i], BENCH[max(0, i - 5):i - 1] if i else []]
        back = dc.uncrunch(frame, m, tails=cands)
        if back != text:
            ok(False, f"message {i} not exact")
        total += len(frame)
    ok(True, f"{len(BENCH)}/{len(BENCH)} exact, {total} B total, {time.time() - t0:.1f}s")
    # the receiver lagging by one (its newest candidate is not the sender's tail) still lands on the right one
    i = 40; frame = dc.crunch(BENCH[i], m, tail=BENCH[i - 4:i])
    back = dc.uncrunch(frame, m, tails=[BENCH[i - 3:i + 1], BENCH[i - 4:i], BENCH[i - 5:i - 1]])
    ok(back == BENCH[i], "receiver one message ahead: second candidate matches")
    back = dc.uncrunch(dc.crunch(BENCH[i], m, tail=()), m, tails=[BENCH[i - 4:i]])
    ok(back == BENCH[i], "sender with no history: base prime found as the last candidate")


def test_refusals():
    print("refusals (never shown as text):")
    m = dc.MockModel(); i = 50; text = BENCH[i]; tail = BENCH[i - 4:i]
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

    body_at = 3 + 1 + (len(frame) > 130)                 # first rc byte (varint is 1 or 2 bytes here)
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
    refused(frame, "only wrong histories offered", tails=[BENCH[i - 5:i - 1], BENCH[i - 6:i - 2], BENCH[i - 3:i + 1]])
    random.seed(5)
    for k in range(3):
        refused(b"\x08" + bytes(random.randrange(256) for _ in range(20)), f"random 23-byte frame #{k}")


# -------------------------------------------------------------------------------------------------- llama stand-in
def bench_llama():
    print("llama.cpp stand-in: last 32 bench messages, tail = previous 4, encode + decode on this box:")
    m = dc.LlamaModel()
    rows, syms = [], []                                     # real logits for the cross-arch gate
    on_air = ideal_total = 0.0; rc_total = ntok = 0; t0 = time.time(); lines = []
    for i in range(len(BENCH) - 32, len(BENCH)):
        text, tail = BENCH[i], BENCH[i - 4:i]
        frame = dc.crunch(text, m, tail=tail)
        back = dc.uncrunch(frame, m, tails=[tail, BENCH[i - 5:i - 1]])
        if back != text:
            ok(False, f"message {i} not exact under llama stand-in")
        toks = m.tokenize(text); lg = m.reset(m.tokenize(dc.open_codec.prime_for(tail))); ideal = 0.0
        for t in toks:
            ideal += dc.ideal_bits(lg, t)
            if len(rows) < 48:
                rows.append(lg.copy()); syms.append(t)
            lg = m.step(t)
        on_air += len(frame); rc_total += len(frame) - 3 - (1 if len(toks) < 128 else 2); ideal_total += ideal; ntok += len(toks)
        lines.append(f"    {i:2d} {len(text):4d} ch {len(toks):3d} tok  frame {len(frame):3d} B  ideal {ideal / 8:6.1f} B  {text[:48]!r}")
    print("\n".join(lines))
    print(f"  32 messages, {ntok} tokens: ON AIR {int(on_air)} B (frames incl. 0x08 + 2-byte check + varint) vs 912 B baseline "
          f"(open codec 0x03 848 + 2 B check each) — {912 / on_air:.2f}x smaller; rc bytes {rc_total} vs float ideal "
          f"{ideal_total / 8:.1f} B ({rc_total * 8 / ntok:.2f} vs {ideal_total / ntok:.2f} bits/token); {time.time() - t0:.0f}s")
    ok(on_air < 912, "beats the 912 B baseline")
    # cross-arch identity on REAL logits: the C harness on x86 and on aarch64 under qemu, same rows, cmp
    b = os.path.join(HERE, "build"); os.makedirs(b, exist_ok=True)
    np.stack(rows).astype(np.float32).tofile(os.path.join(b, "real_logits.f32"))
    np.array(syms, np.uint32).tofile(os.path.join(b, "real_syms.u32"))
    subprocess.run(["make", "-s", "-C", HERE, "build/detrc_test_x86", "build/detrc_test_aarch64"], check=True)
    outs = []
    for tag, cmd in (("x86", [os.path.join(b, "detrc_test_x86")]), ("aarch64", ["/usr/bin/qemu-aarch64", os.path.join(b, "detrc_test_aarch64")])):
        r = subprocess.run(cmd + ["file", os.path.join(b, "real_logits.f32"), str(V), os.path.join(b, "real_syms.u32"),
                                  os.path.join(b, f"real_freq_{tag}.bin"), os.path.join(b, f"real_rc_{tag}.bin")],
                           capture_output=True, text=True, check=True)
        outs.append(r.stdout.strip()); print(f"    {tag:8s} {r.stdout.strip()}")
    same = (open(os.path.join(b, "real_freq_x86.bin"), "rb").read() == open(os.path.join(b, "real_freq_aarch64.bin"), "rb").read()
            and open(os.path.join(b, "real_rc_x86.bin"), "rb").read() == open(os.path.join(b, "real_rc_aarch64.bin"), "rb").read())
    ok(same and outs[0] == outs[1], f"{len(rows)} real logit rows: freq tables and coded bytes IDENTICAL x86 vs aarch64")
    # and the Python/ctypes path produces the same bytes as the C harness for those rows
    e = dc.Encoder()
    for lg, s in zip(rows, syms): e.put_logits(lg, s)
    ok(e.finish() == open(os.path.join(b, "real_rc_x86.bin"), "rb").read(), "ctypes path bytes == C harness bytes")


if __name__ == "__main__":
    test_freq_edges(); test_coder(); test_mock_roundtrip(); test_refusals()
    if "--llama" in sys.argv:
        bench_llama()
    print("ALL PASSED")
