/* detlm — deterministic scalar inference engine for ONE model:
 * Qwen2.5-0.5B q8_0 (GGUF), base or instruct (same geometry). Token ids in, float32 logits out.
 *
 * Contract: for the same token stream (same tokens, same batch split or not),
 * the logits are BIT-IDENTICAL on every platform this is built for, provided the
 * build uses  -O2 -ffp-contract=off -fno-math-errno  and never -ffast-math/-Ofast.
 * All reductions are spelled out in a fixed order in detlm.c; the only libm
 * calls are sqrtf (IEEE correctly rounded). Weights stay q8_0 in memory (mmap).
 * The logits (float[n_vocab]) are the only contract consumers rely on.
 */
#ifndef DETLM_H
#define DETLM_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct detlm detlm;

/* Load a GGUF file. n_ctx = KV cache length in tokens (e.g. 2048).
 * n_threads >= 1 (row-parallel matmuls; results do not depend on the count).
 * Returns NULL on error (message on stderr). */
detlm *detlm_load(const char *gguf_path, int n_ctx, int n_threads);
void   detlm_free(detlm *m);

/* Forget everything in the KV cache (next token goes to position 0). */
void   detlm_reset(detlm *m);
/* Keep only the first n_pos positions of the KV cache (n_pos <= current). Returns 0 or -1. */
int    detlm_rewind(detlm *m, int n_pos);

int    detlm_n_vocab(const detlm *m);
int    detlm_n_ctx(const detlm *m);
int    detlm_n_pos(const detlm *m);       /* tokens currently in the cache */
int    detlm_set_threads(detlm *m, int n_threads);

/* Append n tokens at the current position and evaluate them (batched prompt
 * eval; arithmetic per element is identical to feeding them one at a time).
 * If n_logits > 0, writes the logits of the LAST n_logits tokens into
 * logits[n_logits][n_vocab] (row j = logits after tokens[n - n_logits + j]).
 * n_logits == 0 -> no lm_head work at all. Returns 0, or -1 on error
 * (context full, bad token id, n_logits > n). */
int    detlm_eval(detlm *m, const int32_t *tokens, int n, int n_logits, float *logits);

/* Peak resident set size in KiB as reported by the OS (0 if unavailable). */
long   detlm_peak_rss_kib(void);

const char *detlm_version(void);

#ifdef __cplusplus
}
#endif
#endif
