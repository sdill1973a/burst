/* detrc — exact integer range coder + deterministic logits→frequency function for Burst.
 *
 * Same C on both ends of the radio (x86-64 desktop, arm64 phone). detrc_freq reads the float32 logits by their BITS
 * and does only integer arithmetic from there, so identical float inputs give identical frequency tables on any
 * chip, whatever the compiler does with floating point. The coder is LZMA-shaped: 56-bit window, byte renormalisation,
 * carry propagated through a cache byte. Build: cc -O2 -ffp-contract=off (no libm is used, so -ffast-math could not
 * even bite here, but keep it off for the engine next door).
 *
 * Frequency tables: n symbols, every freq[i] >= 1, sum == DETRC_TOTAL exactly. n must be <= DETRC_MAX_SYMS.
 */
#ifndef DETRC_H
#define DETRC_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DETRC_VERSION     1
#define DETRC_TOTAL_BITS  24
#define DETRC_TOTAL       (1u << DETRC_TOTAL_BITS)
#define DETRC_MAX_SYMS    (1 << 20)          /* 1,048,576 — Qwen2.5 has 151,936 */

int detrc_version(void);

/* logits[n] (float32, any values incl. NaN/Inf) -> freq[n]: sum == DETRC_TOTAL, each >= 1. Returns 0, or -1 if n is out
 * of range (1..DETRC_MAX_SYMS). Pure integer math after reading the float bits. */
int detrc_freq(const float *logits, int32_t n, uint32_t *freq);

/* ---- encoder (symbol at a time) ---- */
typedef struct detrc_enc detrc_enc;
detrc_enc *detrc_enc_new(void);
/* code symbol sym (0..n-1) under freq[n] (as produced by detrc_freq, or any table with sum == DETRC_TOTAL, all >= 1) */
int detrc_enc_put(detrc_enc *e, const uint32_t *freq, int32_t n, int32_t sym);
/* convenience: detrc_freq + detrc_enc_put */
int detrc_enc_put_logits(detrc_enc *e, const float *logits, int32_t n, int32_t sym);
/* flush with the fewest bytes that still decode; returns the byte count (data via detrc_enc_data). Call once. */
int64_t detrc_enc_finish(detrc_enc *e);
const uint8_t *detrc_enc_data(const detrc_enc *e);
void detrc_enc_free(detrc_enc *e);

/* ---- decoder (symbol at a time) ---- */
typedef struct detrc_dec detrc_dec;
/* buf may be shorter than the encoder's output: bytes past the end read as 0 (the encoder strips trailing zeros) */
detrc_dec *detrc_dec_new(const uint8_t *buf, int64_t len);
/* decode one symbol under freq[n]; returns 0..n-1, or -1 on a bad table. Never crashes on garbage input. */
int32_t detrc_dec_get(detrc_dec *d, const uint32_t *freq, int32_t n);
int32_t detrc_dec_get_logits(detrc_dec *d, const float *logits, int32_t n);
void detrc_dec_free(detrc_dec *d);

#ifdef __cplusplus
}
#endif
#endif
