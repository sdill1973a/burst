#!/usr/bin/env python3
"""repro — rerun Burst's compression numbers byte for byte, on public text, and compare frames by sha256.

    python repro.py paste  FILE [--chapter 1] [--chars N]   # one long paste, chained exactly as on air
    python repro.py lines  FILE [--sms-seed 20261024 --n 200] # many short messages, each alone (no history)

paste: the text is split exactly as the phone's Send does (burst-android Chain.parts + Chain.split, which sent the
7.7x paste): parts of at most 4,000 chars; in each, rough links of ~380 chars at sentence then word boundaries, each
crunched for real and halved at a word boundary until it fits LIMIT bytes. Link 0 of a part is primed on no history; every later link is a CARRY (0x0A) link — the same
0x08 frame, primed on the previous link's text. Each link rides in a chain header [0x12][msg id][i<<4 | n-1]. The
msg id is random on air; here it is fixed at 0x00 so two runs give the same bytes (it does not change any length).

Counted: every byte of every frame, chain headers included (what the radio carries as Burst payload; the mesh's own
packet header is not counted). Every link is decoded back and must equal its text exactly, or the run fails.

lines: one message per line, each its own 0x08 frame primed on nothing. --sms-seed reads the UCI SMS Spam
Collection (tab-separated label/text): ham only, shuffled with random.Random(seed), messages under 20 chars dropped,
first --n kept.

Model: qwen2.5-0.5b-base-q8_0.gguf (sha256 d5163baa48d1797b...), set BURST_GGUF to its path. Engine: detlm
(libdetlm.so, `make -f Makefile.detlm`), bit-identical across machines, so the frame sha256 should match anywhere.
"""
import argparse, hashlib, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import det_codec as dc
import open_codec as oc

LIMIT = 200         # bytes per radio payload, as burst_rx.LIMIT / the phone's part size
CHAIN = 0x12        # [0x12][msg id][i<<4 | n-1] + frame
CARRY = 0x0A        # a link coded as 0x08 but primed on carry(tail, previous link's text)
MSG_ID = 0x00       # random on air; fixed here for byte-identical reruns


def carry(tail, prev):
    return (list(tail)[-oc.TAIL_N:] + [prev])[-oc.TAIL_N:]


GUESS = 380         # Chain.kt: chars per link to try first
ONE_MAX = 2400      # Chain.kt: longest text tried as one packet
PART = 4000         # Chain.kt: a paste longer than this goes as several chains
PARTS = 0x0B        # [0x0B][paste id][part i, from 1][of n], one packet before each part of a multi-part paste


def sentences(t):
    return [x for x in re.findall(r".*?(?:[.!?]\s+|$)", t, re.S) if x] or [t]


def words(t):
    return [x for x in re.findall(r"\S*\s*", t) if x]


def crunch_or_none(text, m, tail):
    try:
        return dc.crunch(text, m, tail=list(tail))
    except ValueError:
        return None


def parts(text, size=PART):
    """Chain.parts: a long paste cut into messages of at most `size` chars, sentence then word boundaries."""
    if len(text) <= size:
        return [text]
    out, cur = [], ""
    for s in sentences(text):
        for w in (words(s) if len(s) > size else [s]):
            if len(cur) + len(w) > size and cur:
                out.append(cur); cur = ""
            cur += w
    return out + ([cur] if cur else [])


def chain(text, m, tail=()):
    """Chain.split (the phone, which sent the 7.7x paste): rough links of ~GUESS chars at sentence then word
    boundaries; each crunched for real, halved at a word boundary until it fits; links after the first are CARRY."""
    one = crunch_or_none(text, m, tail) if len(text) <= ONE_MAX else None
    if one is not None and len(one) <= LIMIT:
        return [text], [one]
    rough, cur = [], ""
    for s in sentences(text):
        if len(cur) + len(s) <= GUESS:
            cur += s; continue
        if cur:
            rough.append(cur); cur = ""
        if len(s) <= GUESS:
            cur = s; continue
        for w in words(s):
            if len(cur) + len(w) > GUESS and cur:
                rough.append(cur); cur = ""
            cur += w
    if cur:
        rough.append(cur)
    texts, bodies = [], []
    def fit(piece):
        t = carry(tail, texts[-1]) if texts else list(tail)
        b = crunch_or_none(piece, m, t)
        if b is not None and len(b) + 3 <= LIMIT:
            if texts and b[0] == 0x08:
                b = bytes([CARRY]) + b[1:]
            bodies.append(b); texts.append(piece); return
        w = words(piece)
        if len(w) > 1:
            fit("".join(w[:len(w) // 2])); fit("".join(w[len(w) // 2:])); return
        if len(piece) <= 1:
            raise ValueError("one character will not fit a packet")
        cut = len(piece) // 2; fit(piece[:cut]); fit(piece[cut:])
    for r in rough:
        fit(r)
    if len(bodies) > 16:
        raise OverflowError(f"needs {len(bodies)} packets; the cap is 16")
    return texts, [bytes([CHAIN, MSG_ID, (i << 4) | (len(bodies) - 1)]) + b for i, b in enumerate(bodies)]


def paste_frames(text, m, tail=()):
    """MainActivity send loop: parts(), each part its own chain on the shared tail (not carried across parts), a
    0x0B packet before each part when there is more than one; a part over 16 links is halved and requeued."""
    queue, texts, frames, done = parts(text), [], [], 0
    while queue:
        part = queue.pop(0)
        try:
            t, f = chain(part, m, tail)
        except OverflowError:
            if len(part) > 200:
                queue[:0] = parts(part, len(part) // 2 + 1); continue
            raise
        if done + len(queue) > 0:
            texts.append(None); frames.append(bytes([PARTS, MSG_ID, done + 1, done + 1 + len(queue)]))
        texts += t; frames += f; done += 1
    return texts, frames


def decode_paste(frames, m, tail=()):
    tails = [list(tail)] if tail else []
    got, prev = [], None
    for f in frames:
        if f[0] == PARTS:
            prev = None; continue
        if f[0] != CHAIN:
            got.append(dc.uncrunch(f, m, tails=tails)); prev = None; continue
        body = f[3:]
        if body[0] == CARRY:
            got.append(dc.uncrunch(bytes([0x08]) + body[1:], m, tails=[carry(tail, prev)]))
        else:
            got.append(dc.uncrunch(body, m, tails=tails))
        prev = got[-1]
    return "".join(got)


def chapter(text, n):
    """Chapter n of a Gutenberg book: from its 'CHAPTER <roman>.' line (or 'Chapter 1') to the next chapter line."""
    heads = [mm for mm in re.finditer(r"^(CHAPTER|Chapter) [IVXLC\d]+\.?\]?\s*$", text, re.M)]
    return text[heads[n - 1].start():heads[n].start()].strip() + "\n"


def report(label, chars, frames):
    blob = b"".join(frames)
    print(f"{label}: {chars} chars -> {len(frames)} packet(s), {len(blob)} B on air, "
          f"{chars / len(blob):.3f}x, exact, frames sha256 {hashlib.sha256(blob).hexdigest()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("paste", "lines"))
    ap.add_argument("file")
    ap.add_argument("--chapter", type=int, help="take chapter N (Gutenberg 'CHAPTER I.' headings)")
    ap.add_argument("--chars", type=int, help="first N chars, cut back to the last whitespace")
    ap.add_argument("--sms-seed", type=int, help="file is the UCI SMS Spam Collection: ham, seeded shuffle")
    ap.add_argument("--start", help="begin at the first occurrence of this phrase (unambiguous; --chapter can hit a contents line)")
    ap.add_argument("--flow", action="store_true",
                    help="join the book's hard line breaks into flowing text, as a person pastes it (10-27 on-air test)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    raw = open(a.file, encoding="utf-8", errors="replace").read().replace("\r\n", "\n").lstrip("﻿")
    m = dc.DetlmModel(n_threads=a.threads)
    print(f"model {os.path.basename(oc.MODEL_PATH)} sha256 "
          f"{hashlib.sha256(open(oc.MODEL_PATH, 'rb').read()).hexdigest()[:16]}... detlm {m.version}")
    if a.mode == "paste":
        text = chapter(raw, a.chapter) if a.chapter else raw
        if a.start:
            i = raw.find(a.start)
            if i < 0:
                sys.exit(f"--start phrase not found: {a.start!r}")
            text = raw[i:]
        if a.flow:
            text = " ".join(text.split())
        if a.chars and len(text) > a.chars:
            text = text[:a.chars]
            text = text[:max(text.rfind(" "), text.rfind("\n"))]
        texts, frames = paste_frames(text, m)
        back = decode_paste(frames, m)
        assert back == text, "decode is not exact"
        for t, f in zip(texts, frames):
            print(f"  {'0x0B part header' if t is None else f'link {len(t):4d} chars'} -> {len(f):3d} B")
        report("paste", len(text), frames)
        print(f"text sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    else:
        if a.sms_seed is not None:
            msgs = [l.split("\t", 1)[1].strip() for l in raw.splitlines() if l.startswith("ham\t")]
            random.Random(a.sms_seed).shuffle(msgs)
            msgs = [x for x in msgs if len(x) >= 20][:a.n]
        else:
            msgs = [l for l in raw.splitlines() if l.strip()]
        frames, chars, skipped = [], 0, 0
        for x in msgs:
            try:
                fr = paste_frames(x, m)[1]
            except ValueError as e:
                skipped += 1; print(f"  not sendable as 0x08 ({e}): {x[:40]!r}"); continue
            assert decode_paste(fr, m) == x, f"decode is not exact: {x[:40]!r}"
            frames += fr; chars += len(x)
        print(f"{len(msgs) - skipped} messages (avg {chars / max(1, len(msgs) - skipped):.1f} chars), {skipped} skipped")
        report("lines", chars, frames)


if __name__ == "__main__":
    main()
