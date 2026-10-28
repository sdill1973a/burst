/* detrc.c — exact integer range coder + deterministic logits→frequency function. See detrc.h.
 *
 * NOTHING in this file does floating-point arithmetic. The logits are read as uint32 bit patterns and turned into
 * Q16.16 fixed-point nats with shifts; exp(-d) comes from two 256-entry Q31 tables (detrc_exptab.h, frozen constants).
 * So the frequency table — and therefore every coded byte — is a pure function of the input bits, the same on x86-64,
 * arm64, or anything else with 64-bit unsigned integers.
 *
 * Coder (LZMA shape, widened): low holds 57 bits (bit 56 = carry), range is kept in [2^48, 2^56) by byte-wise
 * renormalisation, r = range >> 24 >= 2^24 so the quantisation loss is <= 2^-24 per symbol. The leading byte LZMA
 * would emit is always zero and is not written; the decoder primes itself with 7 raw bytes. The final flush picks the
 * fewest bytes that still land inside the last interval and strips trailing zeros (the decoder reads past the end as 0).
 */
#include "detrc.h"
#include "detrc_exptab.h"
#include <stdlib.h>
#include <string.h>

#define RC_WINDOW_BITS 56
#define RC_TOP         (1ULL << 48)                 /* renormalise below this */
#define RC_MASK56      ((1ULL << RC_WINDOW_BITS) - 1)
#define RC_MASK48      ((1ULL << 48) - 1)

int detrc_version(void) { return DETRC_VERSION; }

/* ------------------------------------------------------------------------------------------------------------------
 * logits -> frequencies
 * ---------------------------------------------------------------------------------------------------------------- */

#define Q_CLAMP (1 << 30)              /* |logit| capped at 2^14 nats (Q16.16) */
#define INV_LN2_Q30 1549082005ULL      /* round(2^30 / ln 2) */

/* float32 bits -> Q16.16 (truncated toward zero), clamped to +-Q_CLAMP; NaN -> -Q_CLAMP (never the next token) */
static int32_t q16_from_bits(uint32_t b) {
    uint32_t exp = (b >> 23) & 0xFF, man = b & 0x7FFFFF;
    int neg = (b >> 31) & 1;
    int64_t v;
    if (exp == 0xFF) {                 /* inf or nan */
        if (man) return -Q_CLAMP;
        return neg ? -Q_CLAMP : Q_CLAMP;
    }
    if (exp == 0) return 0;            /* zero / denormal: below Q16 resolution anyway */
    int s = (int)exp - 134;            /* value = (man|2^23) * 2^(exp-127-23); Q16 adds 16 */
    uint64_t m = man | 0x800000u;
    if (s >= 0) {
        if (s > 7) return neg ? -Q_CLAMP : Q_CLAMP;      /* >= 2^14 */
        v = (int64_t)(m << s);
        if (v > Q_CLAMP) v = Q_CLAMP;
    } else {
        int t = -s;
        v = t >= 32 ? 0 : (int64_t)(m >> t);
    }
    return (int32_t)(neg ? -v : v);
}

/* 2^31 * exp(-d) for d in Q16.16 nats, d >= 0; <= 2^31, exact to ~1 part in 2^30 */
static uint64_t exp_neg_q31(uint64_t d) {
    uint64_t t = (d * INV_LN2_Q30) >> 30;        /* d / ln2 in Q16.16 (d < 2^31 so the product < 2^62) */
    uint64_t n = t >> 16, f = t & 0xFFFF;
    if (n >= 32) return 0;
    uint64_t w = (DETRC_TA[f >> 8] * DETRC_TB[f & 0xFF]) >> 31;   /* Q31 * Q31 >> 31 = Q31; product < 2^62 */
    return w >> n;
}

int detrc_freq(const float *logits, int32_t n, uint32_t *freq) {
    if (n < 1 || n > DETRC_MAX_SYMS || !logits || !freq) return -1;
    int32_t i, imax = 0, qmax;
    uint32_t bits;
    memcpy(&bits, &logits[0], 4);
    qmax = q16_from_bits(bits);
    for (i = 1; i < n; i++) {
        memcpy(&bits, &logits[i], 4);
        int32_t q = q16_from_bits(bits);
        if (q > qmax) { qmax = q; imax = i; }
    }
    uint64_t W = 0;
    for (i = 0; i < n; i++) {
        memcpy(&bits, &logits[i], 4);
        int64_t d = (int64_t)qmax - (int64_t)q16_from_bits(bits);   /* 0 .. 2^31 */
        uint64_t w = exp_neg_q31((uint64_t)d);
        freq[i] = (uint32_t)w;                                       /* w <= 2^31 fits */
        W += w;
    }
    /* W >= 2^31 (the max token contributes exactly 2^31); scale so the floor leaves room for the +1 on every token */
    uint64_t K = (uint64_t)DETRC_TOTAL - (uint64_t)n;               /* n <= 2^20 < 2^24 */
    uint64_t used = 0;
    for (i = 0; i < n; i++) {
        uint64_t f = 1 + ((uint64_t)freq[i] * K) / W;                /* 2^31 * 2^24 < 2^64 */
        freq[i] = (uint32_t)f;
        used += f;
    }
    freq[imax] += (uint32_t)(DETRC_TOTAL - used);                    /* 0 <= remainder < n */
    return 0;
}

/* ------------------------------------------------------------------------------------------------------------------
 * shared: cumulative table, validated (sum must be exactly DETRC_TOTAL, every entry >= 1)
 * ---------------------------------------------------------------------------------------------------------------- */
static int build_cum(uint32_t **cum, int32_t *cum_n, const uint32_t *freq, int32_t n) {
    if (n < 1 || n > DETRC_MAX_SYMS || !freq) return -1;
    if (*cum_n < n + 1) {
        uint32_t *p = (uint32_t *)realloc(*cum, sizeof(uint32_t) * (size_t)(n + 1));
        if (!p) return -1;
        *cum = p; *cum_n = n + 1;
    }
    uint64_t acc = 0;
    for (int32_t i = 0; i < n; i++) {
        if (freq[i] == 0) return -1;
        (*cum)[i] = (uint32_t)acc;
        acc += freq[i];
        if (acc > DETRC_TOTAL) return -1;
    }
    if (acc != DETRC_TOTAL) return -1;
    (*cum)[n] = DETRC_TOTAL;
    return 0;
}

/* ------------------------------------------------------------------------------------------------------------------
 * encoder
 * ---------------------------------------------------------------------------------------------------------------- */
struct detrc_enc {
    uint64_t low, range;
    uint8_t cache; int first;
    uint64_t pending;
    uint8_t *buf; size_t len, cap;
    uint32_t *cum; int32_t cum_n;
    int finished, oom;
    uint32_t *ftmp; int32_t ftmp_n;
};

detrc_enc *detrc_enc_new(void) {
    detrc_enc *e = (detrc_enc *)calloc(1, sizeof *e);
    if (!e) return NULL;
    e->low = 0; e->range = RC_MASK56; e->cache = 0; e->first = 1; e->pending = 0;
    return e;
}

static void emit(detrc_enc *e, uint8_t b) {
    if (e->len == e->cap) {
        size_t nc = e->cap ? e->cap * 2 : 256;
        uint8_t *p = (uint8_t *)realloc(e->buf, nc);
        if (!p) { e->oom = 1; return; }
        e->buf = p; e->cap = nc;
    }
    e->buf[e->len++] = b;
}

static void shift_low(detrc_enc *e) {
    uint8_t top = (uint8_t)(e->low >> 48);
    uint8_t carry = (uint8_t)(e->low >> 56);          /* 0 or 1 */
    if (top != 0xFF || carry) {
        if (!e->first) emit(e, (uint8_t)(e->cache + carry));
        e->first = 0;
        while (e->pending) { emit(e, (uint8_t)(0xFF + carry)); e->pending--; }
        e->cache = top;
    } else {
        e->pending++;
    }
    e->low = (e->low & RC_MASK48) << 8;
}

int detrc_enc_put(detrc_enc *e, const uint32_t *freq, int32_t n, int32_t sym) {
    if (!e || e->finished || sym < 0 || sym >= n) return -1;
    if (build_cum(&e->cum, &e->cum_n, freq, n)) return -1;
    uint64_t r = e->range >> DETRC_TOTAL_BITS;
    e->low += (uint64_t)e->cum[sym] * r;
    e->range = (uint64_t)freq[sym] * r;
    while (e->range < RC_TOP) { shift_low(e); e->range <<= 8; }
    return e->oom ? -1 : 0;
}

int detrc_enc_put_logits(detrc_enc *e, const float *logits, int32_t n, int32_t sym) {
    if (!e || n < 1 || n > DETRC_MAX_SYMS) return -1;
    if (e->ftmp_n < n) {
        uint32_t *p = (uint32_t *)realloc(e->ftmp, sizeof(uint32_t) * (size_t)n);
        if (!p) return -1;
        e->ftmp = p; e->ftmp_n = n;
    }
    if (detrc_freq(logits, n, e->ftmp)) return -1;
    return detrc_enc_put(e, e->ftmp, n, sym);
}

int64_t detrc_enc_finish(detrc_enc *e) {
    if (!e) return -1;
    if (e->finished) return e->oom ? -1 : (int64_t)e->len;
    e->finished = 1;
    /* the fewest whole bytes whose value still lies in [low, low+range): round low up to a 2^(56-8k) boundary */
    uint64_t hi = e->low + e->range;                 /* < 2^58, no overflow */
    int k;
    for (k = 0; k < 7; k++) {
        uint64_t mask = (1ULL << (RC_WINDOW_BITS - 8 * k)) - 1;
        uint64_t v = (e->low + mask) & ~mask;
        if (v < hi) { e->low = v; break; }
    }
    for (int i = 0; i < k; i++) shift_low(e);
    /* what is left of low below the emitted bytes is zero; flush the cache with any carry still in low */
    uint8_t carry = (uint8_t)(e->low >> 56);
    if (!e->first) emit(e, (uint8_t)(e->cache + carry));
    e->first = 0;
    while (e->pending) { emit(e, (uint8_t)(0xFF + carry)); e->pending--; }
    while (e->len && e->buf[e->len - 1] == 0) e->len--;   /* the decoder reads past the end as zeros */
    return e->oom ? -1 : (int64_t)e->len;
}

const uint8_t *detrc_enc_data(const detrc_enc *e) { return e ? e->buf : NULL; }

void detrc_enc_free(detrc_enc *e) {
    if (!e) return;
    free(e->buf); free(e->cum); free(e->ftmp); free(e);
}

/* ------------------------------------------------------------------------------------------------------------------
 * decoder
 * ---------------------------------------------------------------------------------------------------------------- */
struct detrc_dec {
    uint64_t code, range;
    const uint8_t *buf; int64_t len, pos;
    uint32_t *cum; int32_t cum_n;
    uint32_t *ftmp; int32_t ftmp_n;
};

static uint8_t next_byte(detrc_dec *d) {
    if (d->pos < d->len) return d->buf[d->pos++];
    d->pos++;
    return 0;
}

detrc_dec *detrc_dec_new(const uint8_t *buf, int64_t len) {
    detrc_dec *d = (detrc_dec *)calloc(1, sizeof *d);
    if (!d) return NULL;
    d->buf = buf; d->len = buf ? len : 0; d->pos = 0;
    d->range = RC_MASK56; d->code = 0;
    for (int i = 0; i < 7; i++) d->code = (d->code << 8) | next_byte(d);
    return d;
}

int32_t detrc_dec_get(detrc_dec *d, const uint32_t *freq, int32_t n) {
    if (!d) return -1;
    if (build_cum(&d->cum, &d->cum_n, freq, n)) return -1;
    uint64_t r = d->range >> DETRC_TOTAL_BITS;
    uint64_t v = d->code / r;
    if (v >= DETRC_TOTAL) v = DETRC_TOTAL - 1;       /* only a corrupted stream gets here; stay in bounds */
    int32_t lo = 0, hi = n;                          /* find s: cum[s] <= v < cum[s+1] */
    while (hi - lo > 1) {
        int32_t mid = lo + (hi - lo) / 2;
        if (d->cum[mid] <= v) lo = mid; else hi = mid;
    }
    d->code -= (uint64_t)d->cum[lo] * r;
    d->range = (uint64_t)freq[lo] * r;
    while (d->range < RC_TOP) { d->code = (d->code << 8) | next_byte(d); d->range <<= 8; }
    return lo;
}

int32_t detrc_dec_get_logits(detrc_dec *d, const float *logits, int32_t n) {
    if (!d || n < 1 || n > DETRC_MAX_SYMS) return -1;
    if (d->ftmp_n < n) {
        uint32_t *p = (uint32_t *)realloc(d->ftmp, sizeof(uint32_t) * (size_t)n);
        if (!p) return -1;
        d->ftmp = p; d->ftmp_n = n;
    }
    if (detrc_freq(logits, n, d->ftmp)) return -1;
    return detrc_dec_get(d, d->ftmp, n);
}

void detrc_dec_free(detrc_dec *d) {
    if (!d) return;
    free(d->cum); free(d->ftmp); free(d);
}
