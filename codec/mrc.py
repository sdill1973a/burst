#!/usr/bin/env python3
"""burst/mrc.py — mismatch-resistant coding (Hu & Tang, arXiv 2601.17684), implemented from the paper.

Survives encoder/decoder probabilities that differ by up to a FACTOR c per token (|ln p - ln p'| < ln c) — the size of
gap two different builds or chips really produce (measured here: up to ~1.6x). Per token:
  bucket index of p(x) (prefix-free code)  +  the first m+1 bits of x's fixed random code  +  one terminator bit,
where m = the longest code prefix x shares with any other token whose probability could be confused with it.
The decoder rebuilds the candidate set from ITS OWN probabilities and picks the code that matches the stream longest.
"""
import heapq, math
import numpy as np


class MRC:
    def __init__(self, vocab_size, c=10 / 3, alpha=1.804, floor=1e-12, seed=20260124, bucket_freq=None):
        self.V, self.c = vocab_size, c
        self.weak_expect = 0.5        # 'weak' fires when expected chance matches in the candidate set exceed this (sus_test tunes it)
        self.L = math.ceil(math.log2(vocab_size))
        codes = np.random.default_rng(seed).permutation(1 << self.L)[:vocab_size]
        self.codes = codes.astype(np.int64)
        self.inv = {int(v): i for i, v in enumerate(codes)}
        edges = [1.0]
        while edges[-1] > floor:
            edges.append(edges[-1] / alpha)
        edges.append(0.0)
        self.hi = np.array(edges[:-1]); self.lo = np.array(edges[1:])       # bucket k = (lo[k], hi[k]]
        self.K = len(self.hi)
        freq = bucket_freq if bucket_freq is not None else [1.0 / (k + 1) ** 1.5 for k in range(self.K)]
        self.bcode = _huffman(freq)                                           # bucket k -> bit string
        self.bdec = {v: k for k, v in self.bcode.items()}

    def bucket(self, p):
        return int(np.searchsorted(-self.hi, -p, side="right") - 1) if p > 0 else self.K - 1

    def _lcp(self, a, others):
        """Longest common prefix length (in L-bit codes) of code a with each of others."""
        x = np.bitwise_xor(others, a)
        lz = np.where(x == 0, self.L, self.L - np.floor(np.log2(np.maximum(x, 1))).astype(np.int64) - 1)
        return lz

    def encode_token(self, probs, x):
        p = probs[x]; k = self.bucket(p)
        lo, hi, c2 = self.lo[k], self.hi[k], self.c ** 2
        U = np.nonzero((probs > lo / c2) & (probs < c2 * hi))[0]
        U = U[U != x]
        m = int(self._lcp(self.codes[x], self.codes[U]).max()) if len(U) else 0
        code = format(int(self.codes[x]), f"0{self.L}b")
        bits = self.bcode[k] + code[:m + 1]
        if m + 1 < self.L:
            bits += "1" if code[m + 1] == "0" else "0"
        return bits

    def decode_token(self, probs, stream, pos, strict=False, force=None):
        """One token back, plus HOW SURE: the decoder owns its guess ("make the codec own its guesses and
        mark them as sus"). Returns (x, pos, sus) where sus is None for a clean read or a dict naming the doubt:
          empty : no candidate in the c-window (mismatch beyond the certified factor) — strict=True raises instead;
                  otherwise the window is widened and the best code match taken, flagged
          tie   : two candidates match the stream equally far — the pick is the likelier one, flagged
          weak  : the match is short for the size of the candidate set (expected chance matches > 0.5), flagged
        A wrong pick can desync the stream; later tokens then come out marked too, which is the point."""
        cur = ""
        while cur not in self.bdec:                                           # prefix-free bucket code
            if pos >= len(stream):                                            # ran off the end: a desync upstream ate the
                if strict: raise ValueError("stream ended early — desync")    # stream; guess the likeliest token, flagged
                top = np.argsort(-probs)[:6]
                return int(top[0]), len(stream), {"why": "end", "alts": [int(t) for t in top[1:]]}
            cur += stream[pos]; pos += 1
        k = self.bdec[cur]
        lo, hi = self.lo[k], self.hi[k]
        Uh = np.nonzero((probs > lo / self.c) & (probs < self.c * hi))[0]
        sus = None
        if len(Uh) == 0:
            if strict:
                raise ValueError("empty candidate set — mismatch exceeded the certified factor")
            c2 = self.c ** 3
            Uh = np.nonzero((probs > lo / c2) & (probs < c2 * hi))[0]
            if len(Uh) == 0:
                Uh = np.argsort(-probs)[:64]
            sus = {"why": "empty", "k": k}
        z = stream[pos:pos + self.L].ljust(self.L, "0")
        if force is not None:                                                 # the reader's pick: take it, resync on it
            n = int(self._lcp(int(z, 2), self.codes[np.array([force])])[0])
            pos += self.L if n >= self.L else n + 1
            return int(force), pos, None
        lcp = self._lcp(int(z, 2), self.codes[Uh])
        n = int(lcp.max())
        top = Uh[lcp == n]
        best = int(top[np.argmax(probs[top])]) if len(top) > 1 else int(Uh[int(np.argmax(lcp))])
        if sus is None and len(top) > 1:
            sus = {"why": "tie", "k": k, "n": n, "ties": int(len(top))}
        expect = (len(Uh) - 1) * 2.0 ** (-(n - 1)) if n > 0 else float(len(Uh) - 1)
        if sus is None and expect > self.weak_expect:
            sus = {"why": "weak", "k": k, "n": n, "set": int(len(Uh)), "expect": round(expect, 2)}
        if sus is not None:                                                   # the options for the pop-up: near matches by likelihood
            near = Uh[lcp >= max(n - 1, 0)]; near = near[near != best]
            sus["alts"] = [int(t) for t in near[np.argsort(-probs[near])][:5]]
        pos += self.L if n >= self.L else n + 1
        return best, pos, sus

    def encode(self, tokens, probs_fn, ctx):
        ctx, out = list(ctx), []
        for t in tokens:
            out.append(self.encode_token(probs_fn(ctx), t)); ctx.append(t)
        bits = "".join(out)
        return _pack(bits), len(bits)

    def decode(self, data, n_tokens, probs_fn, ctx, strict=False):
        """tokens only (the old contract); strict=True raises on an empty candidate set as before."""
        return self.decode_marked(data, n_tokens, probs_fn, ctx, strict=strict)[0]

    def decode_marked(self, data, n_tokens, probs_fn, ctx, strict=False, forced=None):
        """(tokens, marks): marks[i] is None or the sus dict for token i (with 'alts', the pop-up's options).
        forced = {i: token} are the reader's picks; the stream resyncs on each one, so later marks can clear."""
        stream = "".join(f"{b:08b}" for b in data)
        ctx, out, marks, pos = list(ctx), [], [], 0
        for i in range(n_tokens):
            x, pos, sus = self.decode_token(probs_fn(ctx), stream, pos, strict=strict, force=(forced or {}).get(i))
            out.append(x); marks.append(sus); ctx.append(x)
        return out, marks


def _pack(bits):
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def _huffman(freq):
    heap = [(f, i, {i: ""}) for i, f in enumerate(freq)]
    heapq.heapify(heap); n = len(heap)
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap); f2, _, b = heapq.heappop(heap)
        d = {k: "0" + v for k, v in a.items()}; d.update({k: "1" + v for k, v in b.items()})
        heapq.heappush(heap, (f1 + f2, n, d)); n += 1
    return heap[0][2]
