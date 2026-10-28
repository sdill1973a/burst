/* det08 — frame 0x08 on top of detlm (bit-identical logits) + detrc (exact range coder).
 *
 *   frame = [0x08][2-byte check = sha256(text)[:2]][varint n_tokens][range-coder bytes]
 *
 * The SAME call pattern as the reference det_codec/detcodec: the prime is evaluated as ONE batch
 * (logits of its last token), then every message token one at a time. Text never enters here: the check is computed
 * where the text is (Kotlin) and passed in; the caller tries candidate primes and lets the check pick.
 * Build with -O2 -ffp-contract=off, never -ffast-math (the detlm engine depends on it; this file has no float math).
 */
#ifndef DET08_H
#define DET08_H
#include <stdint.h>
#include "detlm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DET08_CODEC_BYTE 0x08
#define DET08_MAX_TOKENS 2048          /* a flipped varint byte must not become a million model steps */

/* bytes the encoder may need for nm tokens (header + worst-case 3 B/token + flush slack) */
#define DET08_FRAME_CAP(nm) (16 + 4 * (nm))

/* Encode: prime[np] then msg[nm]; writes the whole frame into out (cap out_cap). Returns the frame length, or -1
 * (bad sizes / context full / out too small). reason (optional) gets a static string on failure. */
int det08_encode(detlm *m, const uint8_t check[2], const int32_t *prime, int np, const int32_t *msg, int nm,
                 uint8_t *out, int out_cap, const char **reason);

/* Header only: check bytes, token count, offset of the coder bytes. 0 ok, -1 refused (reason set). */
int det08_parse(const uint8_t *frame, int flen, uint8_t check[2], uint32_t *n, int *body_off, const char **reason);

/* Decode under one prime: ids[n] (cap ids_cap). Returns n (>= 0), or -1 refused/error (reason set). The check is NOT
 * verified here — the caller detokenizes and compares sha256(text)[:2], because the text lives on the other side. */
int det08_decode(detlm *m, const int32_t *prime, int np, const uint8_t *frame, int flen,
                 int32_t *ids, int ids_cap, const char **reason);

#ifdef __cplusplus
}
#endif
#endif
