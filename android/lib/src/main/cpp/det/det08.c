/* det08.c — frame 0x08 encode/decode core, shared by the JNI (burst_jni.cpp) and the host harness (host_test.c).
 * Mirrors the reference detcodec's do_encode/do_decode, minus the text parsing. No floating-point arithmetic here.
 */
#include "det08.h"
#include "detrc.h"
#include <stdlib.h>
#include <string.h>

int det08_encode(detlm *m, const uint8_t check[2], const int32_t *prime, int np, const int32_t *msg, int nm,
                 uint8_t *out, int out_cap, const char **reason) {
    const char *why = NULL; int rc = -1;
    int V = detlm_n_vocab(m), ctx = detlm_n_ctx(m);
    float *lg = NULL; detrc_enc *e = NULL;
    if (np <= 0) { why = "empty prime"; goto done; }
    if (nm < 0 || nm > DET08_MAX_TOKENS) { why = "message over the token cap"; goto done; }
    if (np + nm > ctx) { why = "prime + message over the context"; goto done; }
    if (out_cap < 8) { why = "output buffer too small"; goto done; }
    lg = (float *)malloc((size_t)V * sizeof(float));
    if (!lg) { why = "out of memory"; goto done; }
    detlm_reset(m);
    if (detlm_eval(m, prime, np, 1, lg) != 0) { why = "prime eval failed"; goto done; }
    e = detrc_enc_new();
    if (!e) { why = "out of memory"; goto done; }
    for (int i = 0; i < nm; i++) {
        int32_t t = msg[i];
        if (t < 0 || t >= V) { why = "bad token id"; goto done; }
        if (detrc_enc_put_logits(e, lg, V, t) != 0) { why = "enc_put failed"; goto done; }
        if (detlm_eval(m, &t, 1, 1, lg) != 0) { why = "step eval failed"; goto done; }
    }
    {
        int64_t nb = detrc_enc_finish(e);
        if (nb < 0) { why = "enc_finish failed"; goto done; }
        int hl = 0;
        out[hl++] = DET08_CODEC_BYTE; out[hl++] = check[0]; out[hl++] = check[1];
        unsigned v = (unsigned)nm;
        do { uint8_t b = v & 0x7F; v >>= 7; out[hl++] = (uint8_t)(b | (v ? 0x80 : 0)); } while (v);
        if (hl + nb > out_cap) { why = "output buffer too small"; goto done; }
        memcpy(out + hl, detrc_enc_data(e), (size_t)nb);
        rc = hl + (int)nb;
    }
done:
    if (e) detrc_enc_free(e);
    free(lg);
    if (reason) *reason = why;
    return rc;
}

int det08_parse(const uint8_t *frame, int flen, uint8_t check[2], uint32_t *n, int *body_off, const char **reason) {
    const char *why = NULL;
    if (flen < 4) why = "frame too short";
    else if (frame[0] != DET08_CODEC_BYTE) why = "not a 0x08 frame";
    else {
        uint32_t v = 0; int shift = 0, i = 3;
        for (;;) {
            if (i >= flen || shift > 28) { why = "bad token count"; break; }
            uint8_t b = frame[i++]; v |= (uint32_t)(b & 0x7F) << shift;
            if (!(b & 0x80)) break;
            shift += 7;
        }
        if (!why) {
            if (v > DET08_MAX_TOKENS) why = "token count over the cap";
            else { check[0] = frame[1]; check[1] = frame[2]; *n = v; *body_off = i; }
        }
    }
    if (reason) *reason = why;
    return why ? -1 : 0;
}

int det08_decode(detlm *m, const int32_t *prime, int np, const uint8_t *frame, int flen,
                 int32_t *ids, int ids_cap, const char **reason) {
    const char *why = NULL; int rc = -1;
    uint8_t chk[2]; uint32_t n = 0; int off = 0;
    float *lg = NULL; detrc_dec *d = NULL;
    if (det08_parse(frame, flen, chk, &n, &off, &why) != 0) goto done;
    if (np <= 0) { why = "empty prime"; goto done; }
    if ((int)n > ids_cap) { why = "ids buffer too small"; goto done; }
    if ((int)n + np > detlm_n_ctx(m)) { why = "token count over the context"; goto done; }
    {
        int V = detlm_n_vocab(m);
        lg = (float *)malloc((size_t)V * sizeof(float));
        if (!lg) { why = "out of memory"; goto done; }
        detlm_reset(m);
        if (detlm_eval(m, prime, np, 1, lg) != 0) { why = "prime eval failed"; goto done; }
        d = detrc_dec_new(frame + off, flen - off);
        if (!d) { why = "out of memory"; goto done; }
        for (uint32_t k = 0; k < n; k++) {
            int32_t t = detrc_dec_get_logits(d, lg, V);
            if (t < 0) { why = "dec_get failed"; goto done; }
            ids[k] = t;
            if (k + 1 < n && detlm_eval(m, &t, 1, 1, lg) != 0) { why = "step eval failed"; goto done; }
        }
        rc = (int)n;
    }
done:
    if (d) detrc_dec_free(d);
    free(lg);
    if (reason) *reason = why;
    return rc;
}
