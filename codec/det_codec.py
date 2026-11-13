#!/usr/bin/env python3
"""det_codec — burst frame 0x08: an exact range coder over a bit-identical model.

    frame = [0x08][2-byte check = sha256(text)[:2]][varint n_tokens][range-coder bytes]

Model and coder are implied by the codec byte: no model id, no c, no tail id. The history window (the learned tail,
open_codec.prime_for) is found by the receiver trying candidate tails newest-first and letting the 2-byte check pick;
nothing that fails the check is ever shown. Both ends must compute the SAME logits bit for bit — that is detlm's job
(detlm.c); detrc (detrc.c via ctypes) turns identical logits into identical bytes.

Model protocol (pluggable, so the coder is testable without model weights):
    m.n_vocab                      int
    m.tokenize(text) -> list[int]  m.detokenize(ids) -> str
    m.reset(prime_ids) -> logits   float32[V]: next-token logits after the prime
    m.step(tok) -> logits          float32[V]: next-token logits after feeding one more token
    m.n_ctx                        int or None: context size; a decode that would overrun it is refused up front
Three implementations here: DetlmModel (the real engine), LlamaModel (llama.cpp as a stand-in — exact only encoder-and-decoder-on-the-same-box, same
eval path) and MockModel (seeded hash model, byte tokens, no weights).

API:  crunch(text, model, tail=()) -> bytes      uncrunch(frame, model, tails=()) -> text  (raises DecodeFailed)
"""
import ctypes, hashlib, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import open_codec                               # PRIME, prime_for, TAIL_N — one definition, shared with 0x03
except ImportError:                                 # source tree: open_codec.py sits one directory up
    sys.path.insert(0, os.path.dirname(HERE))
    import open_codec

CODEC_BYTE = 0x08
TOK_PER_BYTE, TOK_SLACK = 16, 64  # his 334 real frames: max 2.9 tok/byte, one repeat 42 tok in 2 B; 5x headroom
MAX_TOKENS = 2048                                   # a flipped varint byte must not become a million model steps
MAX_TAILS = 6                                       # candidate histories tried, newest first
TOTAL_BITS = 24


class DecodeFailed(ValueError):
    """The check did not match under any candidate history: refused, never shown."""


# ---------------------------------------------------------------------------------------------------------------------
# detrc via ctypes
# ---------------------------------------------------------------------------------------------------------------------
_lib = None


def _build_lib():
    so = os.path.join(HERE, "build", "libdetrc.so")
    srcs = [os.path.join(HERE, f) for f in ("detrc.c", "detrc.h", "detrc_exptab.h")]
    if not os.path.exists(so) or any(os.path.getmtime(s) > os.path.getmtime(so) for s in srcs):
        os.makedirs(os.path.dirname(so), exist_ok=True)
        subprocess.run([os.environ.get("CC", "cc"), "-O2", "-ffp-contract=off", "-std=c99", "-fPIC", "-shared",
                        srcs[0], "-o", so], check=True)
    return so


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(_build_lib())
        u8p, f32p, u32p = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)
        L.detrc_version.restype = ctypes.c_int
        L.detrc_freq.argtypes = [f32p, ctypes.c_int32, u32p]; L.detrc_freq.restype = ctypes.c_int
        L.detrc_enc_new.restype = ctypes.c_void_p
        L.detrc_enc_put.argtypes = [ctypes.c_void_p, u32p, ctypes.c_int32, ctypes.c_int32]; L.detrc_enc_put.restype = ctypes.c_int
        L.detrc_enc_put_logits.argtypes = [ctypes.c_void_p, f32p, ctypes.c_int32, ctypes.c_int32]; L.detrc_enc_put_logits.restype = ctypes.c_int
        L.detrc_enc_finish.argtypes = [ctypes.c_void_p]; L.detrc_enc_finish.restype = ctypes.c_int64
        L.detrc_enc_data.argtypes = [ctypes.c_void_p]; L.detrc_enc_data.restype = ctypes.c_void_p
        L.detrc_enc_free.argtypes = [ctypes.c_void_p]
        L.detrc_dec_new.argtypes = [u8p, ctypes.c_int64]; L.detrc_dec_new.restype = ctypes.c_void_p
        L.detrc_dec_get.argtypes = [ctypes.c_void_p, u32p, ctypes.c_int32]; L.detrc_dec_get.restype = ctypes.c_int32
        L.detrc_dec_get_logits.argtypes = [ctypes.c_void_p, f32p, ctypes.c_int32]; L.detrc_dec_get_logits.restype = ctypes.c_int32
        L.detrc_dec_free.argtypes = [ctypes.c_void_p]
        _lib = L
    return _lib


def _f32(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def freq_table(logits):
    """the deterministic integer frequency table (sum 2^24, every entry >= 1) — exposed for tests and the identity gate"""
    a, p = _f32(logits)
    out = np.empty(len(a), np.uint32)
    if lib().detrc_freq(p, len(a), out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))):
        raise ValueError("detrc_freq failed")
    return out


class Encoder:
    def __init__(self):
        self.L = lib(); self.h = self.L.detrc_enc_new()

    def put_logits(self, logits, sym):
        a, p = _f32(logits)
        if self.L.detrc_enc_put_logits(self.h, p, len(a), int(sym)):
            raise ValueError("detrc_enc_put_logits failed")

    def put(self, freq, sym):
        f = np.ascontiguousarray(freq, dtype=np.uint32)
        if self.L.detrc_enc_put(self.h, f.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(f), int(sym)):
            raise ValueError("detrc_enc_put failed (bad table?)")

    def finish(self):
        n = self.L.detrc_enc_finish(self.h)
        if n < 0:
            raise MemoryError("detrc_enc_finish")
        return ctypes.string_at(self.L.detrc_enc_data(self.h), n) if n else b""

    def __del__(self):
        if getattr(self, "h", None):
            self.L.detrc_enc_free(self.h); self.h = None


class Decoder:
    def __init__(self, data):
        self.L = lib(); self.buf = (ctypes.c_uint8 * max(1, len(data))).from_buffer_copy(data or b"\0")
        self.h = self.L.detrc_dec_new(self.buf, len(data))

    def get_logits(self, logits):
        a, p = _f32(logits)
        s = self.L.detrc_dec_get_logits(self.h, p, len(a))
        if s < 0:
            raise ValueError("detrc_dec_get_logits failed")
        return s

    def get(self, freq):
        f = np.ascontiguousarray(freq, dtype=np.uint32)
        s = self.L.detrc_dec_get(self.h, f.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(f))
        if s < 0:
            raise ValueError("detrc_dec_get failed (bad table?)")
        return s

    def __del__(self):
        if getattr(self, "h", None):
            self.L.detrc_dec_free(self.h); self.h = None


# ---------------------------------------------------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------------------------------------------------
def _check(text):
    return hashlib.sha256(text.encode()).digest()[:2]


def _wv(n, out):
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return


def _rv(buf, i, limit=MAX_TOKENS):
    shift = val = 0
    while True:
        if i >= len(buf) or shift > 28:
            raise DecodeFailed("bad token count")
        b = buf[i]; i += 1; val |= (b & 0x7F) << shift
        if not b & 0x80:
            if val > limit:
                raise DecodeFailed(f"token count {val} over the cap {limit}")
            return val, i
        shift += 7


def crunch(text, model, tail=()):
    """text -> 0x08 frame. tail = the learned history (the last TAIL_N confirmed messages, oldest first)."""
    tail = list(tail)[-open_codec.TAIL_N:]
    toks = model.tokenize(text)
    if model.detokenize(toks) != text:
        raise ValueError("text does not survive tokenize/detokenize; not sendable as 0x08")
    if len(toks) > MAX_TOKENS:
        raise ValueError(f"{len(toks)} tokens is over the 0x08 cap of {MAX_TOKENS}")
    enc = Encoder()
    lg = model.reset(model.tokenize(open_codec.prime_for(tail)))
    for t in toks:
        enc.put_logits(lg, t)
        lg = model.step(t)
    head = bytearray([CODEC_BYTE]) + _check(text); _wv(len(toks), head)
    return bytes(head) + enc.finish()


def _decode_with(body, n, model, tail):
    prime = model.tokenize(open_codec.prime_for(tail))
    n_ctx = getattr(model, "n_ctx", None)
    if n_ctx and len(prime) + n > n_ctx:            # cannot be a frame this model encoded under this history
        raise DecodeFailed(f"{len(prime)} prime + {n} message tokens is over the {n_ctx}-token context")
    dec = Decoder(body)
    lg = model.reset(prime)
    toks = []
    for _ in range(n):
        t = dec.get_logits(lg)
        toks.append(t)
        lg = model.step(t)
    return model.detokenize(toks)


def uncrunch(frame, model, tails=(), max_tails=MAX_TAILS):
    """0x08 frame -> text, or DecodeFailed. tails = candidate histories newest first (each a list of messages, oldest
    first); the base prime (no history) is always tried last. The first candidate whose decode passes the check wins."""
    if not frame or frame[0] != CODEC_BYTE:
        raise DecodeFailed("not a 0x08 frame")
    if len(frame) < 4:
        raise DecodeFailed("frame too short")
    chk = bytes(frame[1:3])
    n, i = _rv(frame, 3)
    body = bytes(frame[i:])
    if n > TOK_PER_BYTE * len(body) + TOK_SLACK:     # 10-30 review: a 6-byte frame claiming 2,048 tokens = minutes of evals
        raise DecodeFailed(f"token count {n} implausible for {len(body)} body bytes")
    cands = [list(t)[-open_codec.TAIL_N:] for t in list(tails)[:max_tails]]
    if [] not in cands:
        cands.append([])
    for tail in cands:
        try:
            text = _decode_with(body, n, model, tail)
        except (ValueError, UnicodeError, RuntimeError):  # over the context, an engine error, or a stream the
            continue                                      # detokenizer cannot render: this candidate does not match
        if _check(text) == chk:
            return text
    raise DecodeFailed("plaintext check failed under every candidate history: wrong decode, not shown")


# ---------------------------------------------------------------------------------------------------------------------
# wire 2 (additive: 0x08/0x0A/0x12 above are unchanged and always still decode)
#   single  [0x0E][tail id][3-byte check][varint n][range-coder bytes]     carried link: same with 0x0F
#   chain   [0x14][mid][idx<<4|total-1][3-byte whole-message check, link 0 ONLY][inner frame]
# tail id = sha256(the exact prime text)[0], so the receiver decodes only candidates carrying that id (one, unless a
# 1/256 collision). The whole-message check is what stops links of two messages assembling into one nobody sent.
# ---------------------------------------------------------------------------------------------------------------------
CODEC_V2, CARRY_V2, CHAIN_V2 = 0x0E, 0x0F, 0x14


def check3(text):
    return hashlib.sha256(text.encode()).digest()[:3]


def tail_id(tail):
    tail = list(tail)[-open_codec.TAIL_N:]
    return hashlib.sha256(open_codec.prime_for(tail).encode()).digest()[0]


def ask_key(frame):
    """the wider ASK/resend key: both ends hold these exact frame bytes, so no check collision can pick another message"""
    return hashlib.sha256(bytes(frame)).digest()[:4]


def crunch2(text, model, tail=(), byte=CODEC_V2):
    tail = list(tail)[-open_codec.TAIL_N:]
    toks = model.tokenize(text)
    if model.detokenize(toks) != text:
        raise ValueError("text does not survive tokenize/detokenize; not sendable as wire 2")
    if len(toks) > MAX_TOKENS:
        raise ValueError(f"{len(toks)} tokens is over the cap of {MAX_TOKENS}")
    enc = Encoder()
    lg = model.reset(model.tokenize(open_codec.prime_for(tail)))
    for t in toks:
        enc.put_logits(lg, t)
        lg = model.step(t)
    head = bytearray([byte, tail_id(tail)]) + check3(text); _wv(len(toks), head)
    return bytes(head) + enc.finish()


def uncrunch2(frame, model, tails=()):
    """wire-2 frame -> text, or DecodeFailed. Only candidates whose tail id matches are decoded; base prime is a candidate."""
    if not frame or frame[0] not in (CODEC_V2, CARRY_V2):
        raise DecodeFailed("not a wire-2 frame")
    if len(frame) < 6:
        raise DecodeFailed("frame too short")
    tid, chk = frame[1], bytes(frame[2:5])
    n, i = _rv(frame, 5)
    body = bytes(frame[i:])
    if n > TOK_PER_BYTE * len(body) + TOK_SLACK:
        raise DecodeFailed(f"token count {n} implausible for {len(body)} body bytes")
    cands = [list(t)[-open_codec.TAIL_N:] for t in tails]
    if [] not in cands:
        cands.append([])
    for tail in [t for t in cands if tail_id(t) == tid]:
        try:
            text = _decode_with(body, n, model, tail)
        except (ValueError, UnicodeError, RuntimeError):
            continue
        if check3(text) == chk:
            return text
    raise DecodeFailed("wire 2: no candidate history passed the check: not shown")


def link2(mid, idx, total, inner, whole=None):
    if not 1 <= total <= 16 or not 0 <= idx < total or (idx == 0) != (whole is not None):
        raise ValueError("bad chain link: link 0, and only link 0, carries the whole-message check")
    return bytes([CHAIN_V2, mid, (idx << 4) | (total - 1)]) + (bytes(whole) if idx == 0 else b"") + bytes(inner)


def parse_link2(b):
    """-> (mid, idx, total, whole or None, inner frame)"""
    if len(b) < 4 or b[0] != CHAIN_V2:
        raise DecodeFailed("not a wire-2 chain link")
    idx, total = b[2] >> 4, (b[2] & 0x0F) + 1
    at = 6 if idx == 0 else 3
    if idx >= total or len(b) <= at:
        raise DecodeFailed("malformed wire-2 chain link")
    return b[1], idx, total, (bytes(b[3:6]) if idx == 0 else None), bytes(b[at:])


def join2(texts, whole):
    """the assembled message, only if the whole-message check from link 0 passes"""
    text = "".join(texts)
    if check3(text) != bytes(whole):
        raise DecodeFailed("whole-message check failed: links do not belong to one message, not shown")
    return text


# ---------------------------------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------------------------------
class LlamaModel:
    """llama.cpp as the stand-in engine. Exact ONLY when encoder and decoder run the same build on the same box with the
    same eval path: the prime is evaluated as one batch, then every message token one at a time, on both sides
    (measured 10-24: step-vs-step logits bit-identical even at 16 threads; batched-vs-stepped differ by up to 0.34)."""

    def __init__(self, path=None, n_ctx=1024, n_threads=None):
        from llama_cpp import Llama
        path = path or open_codec.MODEL_PATH
        self.n_ctx = n_ctx
        self.L = Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads or max(1, (os.cpu_count() or 2) // 2),
                       logits_all=False, verbose=False)
        self.n_vocab = self.L.n_vocab()

    def tokenize(self, text):
        return self.L.tokenize(text.encode(), add_bos=False, special=False)

    def detokenize(self, ids):
        return open_codec.detok(self.L, ids).decode("utf-8", errors="replace")

    def _logits(self):
        return np.ctypeslib.as_array(self.L._ctx.get_logits(), shape=(self.n_vocab,)).copy()

    def reset(self, prime_ids):
        self.L.reset(); self.L.eval(list(prime_ids)); return self._logits()

    def step(self, tok):
        self.L.eval([int(tok)]); return self._logits()


class DetlmModel:
    """detlm (detlm.c via ctypes, libdetlm.so): the bit-identical engine — the SAME logits on every CPU it is built
    for, so frames encoded on one machine decode on another exactly. Tokenizer: llama.cpp's, vocab-only load (integer,
    deterministic, no weights). Call pattern, kept identical to the C detcodec on both ends: the prime as ONE batch
    (logits of its last token), then every message token one at a time."""

    def __init__(self, path=None, n_ctx=2048, n_threads=4, so=None):
        from llama_cpp import Llama
        path = path or open_codec.MODEL_PATH
        self.tok = Llama(model_path=path, vocab_only=True, verbose=False)
        self.L = ctypes.CDLL(so or os.path.join(HERE, "libdetlm.so"))
        L = self.L; i32p, f32p = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
        L.detlm_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]; L.detlm_load.restype = ctypes.c_void_p
        L.detlm_free.argtypes = [ctypes.c_void_p]
        L.detlm_reset.argtypes = [ctypes.c_void_p]
        L.detlm_n_vocab.argtypes = [ctypes.c_void_p]; L.detlm_n_vocab.restype = ctypes.c_int
        L.detlm_n_pos.argtypes = [ctypes.c_void_p]; L.detlm_n_pos.restype = ctypes.c_int
        L.detlm_n_ctx.argtypes = [ctypes.c_void_p]; L.detlm_n_ctx.restype = ctypes.c_int
        L.detlm_eval.argtypes = [ctypes.c_void_p, i32p, ctypes.c_int, ctypes.c_int, f32p]; L.detlm_eval.restype = ctypes.c_int
        L.detlm_version.restype = ctypes.c_char_p
        self.h = L.detlm_load(path.encode(), int(n_ctx), int(n_threads))
        if not self.h:
            raise RuntimeError(f"detlm_load failed on {path}")
        self.n_vocab = L.detlm_n_vocab(self.h)
        self.n_ctx = L.detlm_n_ctx(self.h)
        self.version = L.detlm_version().decode()

    def tokenize(self, text):
        return self.tok.tokenize(text.encode(), add_bos=False, special=False)

    def detokenize(self, ids):
        return open_codec.detok(self.tok, ids).decode("utf-8", errors="replace")

    def _eval(self, ids):
        a = np.ascontiguousarray(ids, dtype=np.int32)
        lg = np.empty(self.n_vocab, np.float32)
        if self.L.detlm_eval(self.h, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(a), 1,
                             lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))):
            raise RuntimeError("detlm_eval failed (context full or bad token id)")
        return lg

    def reset(self, prime_ids):
        self.L.detlm_reset(self.h); return self._eval(list(prime_ids))

    def step(self, tok):
        return self._eval([int(tok)])

    def close(self):
        if getattr(self, "h", None):
            self.L.detlm_free(self.h); self.h = None

    def __del__(self):
        self.close()


class MockModel:
    """A deterministic fake: tokens are UTF-8 bytes (ids 0..255) inside a 151,936 vocabulary; logits come from a seeded
    generator keyed on the WHOLE context (so different histories give different streams, as a real model would), with
    a context-chosen sharpness so flat, peaky and floor cases all occur."""
    n_vocab = 151936
    n_ctx = None

    def __init__(self, seed=20261024):
        self.seed = seed; self.ctx = []

    def tokenize(self, text):
        return list(text.encode())

    def detokenize(self, ids):
        return bytes(i if i < 256 else 0x3F for i in ids).decode("utf-8", errors="replace")   # garbage id -> '?'

    def _logits(self):
        key = hashlib.sha256(repr((self.seed, self.ctx)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        scale = [0.0, 1.0, 4.0, 12.0, 40.0][key[8] % 5]
        lg = (rng.standard_normal(self.n_vocab) * scale).astype(np.float32)
        lg[:256] += 6.0                               # bytes are likelier than the unused ids, as a real tokenizer would be
        lg[key[9]] += 20.0                            # one strong favourite per context
        return lg

    def reset(self, prime_ids):
        self.ctx = list(prime_ids); return self._logits()

    def step(self, tok):
        self.ctx.append(int(tok)); return self._logits()


def ideal_bits(logits, sym):
    """-log2 softmax(logits)[sym] in float64 — reporting only, never on the coding path"""
    u = np.asarray(logits, np.float64); u = u - u.max(); return float(np.log2(np.exp(u).sum()) - u[sym] / np.log(2))


if __name__ == "__main__":
    m = MockModel()
    for t in ["Checking in. How goes it? Sound off.", "ok", "Line one.\n\nLine  two, with a double space.\n"]:
        f = crunch(t, m, tail=["earlier message"]); back = uncrunch(f, m, tails=[["earlier message"]])
        print(f"{len(t):4d} chars -> {len(f):4d} B  exact={back == t}")
