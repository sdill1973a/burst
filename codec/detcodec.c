/* detcodec — frame 0x08 on the phone: detlm (bit-identical logits) + detrc (exact range coder), no Python, no
 * tokenizer. Token ids come in as text; the 2-byte plaintext check (sha256(text)[:2]) is computed where the text is
 * (the host side) and passed in, because this binary never sees text.
 *
 *   detcodec MODEL.gguf encode JOB OUT [--threads N] [--ctx N]
 *       JOB line:  <check hex4> | <prime ids...> | <message ids...>
 *       OUT line:  <frame hex> n=<n> t_prime=<s> t_tok=<s>
 *   detcodec MODEL.gguf decode JOB OUT [--threads N] [--ctx N]
 *       JOB line:  <prime ids...> | <frame hex>
 *       OUT line:  ok check=<hex4> n=<n> t_prime=<s> t_tok=<s> | <ids...>
 *              or  refused <reason>
 *
 * Call pattern on both verbs, the same as det_codec.py: the prime is evaluated as ONE batch (logits of its last
 * token), then every message token one at a time. Frame = [0x08][check 2 B][varint n][rc bytes]; n is capped at
 * MAX_TOKENS and at the free context, so a flipped varint byte never becomes a million model steps.
 */
#define _POSIX_C_SOURCE 200809L
#include "detlm.h"
#include "detrc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <time.h>

#define CODEC_BYTE 0x08
#define MAX_TOKENS 2048

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

/* parse "t0 t1 ..." up to the next '|' or end; returns count, advances *pp past the '|' if present; -1 on bad id */
static int parse_ids(const char **pp, int32_t *out, int cap) {
    const char *p = *pp; int n = 0;
    for (;;) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '|') { p++; break; }
        if (*p == 0 || *p == '\n' || *p == '\r') break;
        char *e; long v = strtol(p, &e, 10);
        if (e == p) return -1;
        if (n >= cap) return -1;
        out[n++] = (int32_t)v; p = e;
    }
    *pp = p; return n;
}

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* parse hex digits up to '|' / end into buf; returns byte count or -1 */
static int parse_hex(const char **pp, uint8_t *buf, int cap) {
    const char *p = *pp; int n = 0;
    while (*p == ' ' || *p == '\t') p++;
    for (;;) {
        if (*p == 0 || *p == '\n' || *p == '\r' || *p == ' ' || *p == '|') break;
        int h = hexval(p[0]), l = hexval(p[1]);
        if (h < 0 || l < 0 || n >= cap) return -1;
        buf[n++] = (uint8_t)(h * 16 + l); p += 2;
    }
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '|') p++;
    *pp = p; return n;
}

static void put_hex(FILE *f, const uint8_t *b, size_t n) { for (size_t i = 0; i < n; i++) fprintf(f, "%02x", b[i]); }

static int do_encode(detlm *m, const char *line, FILE *out, float *lg, int32_t *ids, int cap) {
    int V = detlm_n_vocab(m), ctx = detlm_n_ctx(m);
    const char *p = line; uint8_t chk[2];
    if (parse_hex(&p, chk, 2) != 2) { fprintf(out, "error bad check\n"); return -1; }
    int np = parse_ids(&p, ids, cap);
    if (np <= 0) { fprintf(out, "error bad prime\n"); return -1; }
    int nm = parse_ids(&p, ids + np, cap - np);
    if (nm < 0 || nm > MAX_TOKENS || np + nm > ctx) { fprintf(out, "error bad message (n=%d)\n", nm); return -1; }
    detlm_reset(m);
    double t0 = now();
    if (detlm_eval(m, ids, np, 1, lg) != 0) { fprintf(out, "error prime eval\n"); return -1; }
    double t1 = now();
    detrc_enc *e = detrc_enc_new();
    for (int i = 0; i < nm; i++) {
        int32_t t = ids[np + i];
        if (detrc_enc_put_logits(e, lg, V, t) != 0) { fprintf(out, "error enc_put\n"); detrc_enc_free(e); return -1; }
        if (detlm_eval(m, &t, 1, 1, lg) != 0) { fprintf(out, "error step eval\n"); detrc_enc_free(e); return -1; }
    }
    double t2 = now();
    int64_t nb = detrc_enc_finish(e);
    if (nb < 0) { fprintf(out, "error enc_finish\n"); detrc_enc_free(e); return -1; }
    uint8_t head[8]; int hl = 0;
    head[hl++] = CODEC_BYTE; head[hl++] = chk[0]; head[hl++] = chk[1];
    { unsigned v = (unsigned)nm; do { uint8_t b = v & 0x7F; v >>= 7; head[hl++] = (uint8_t)(b | (v ? 0x80 : 0)); } while (v); }
    put_hex(out, head, (size_t)hl); put_hex(out, detrc_enc_data(e), (size_t)nb);
    fprintf(out, " n=%d t_prime=%.4f t_tok=%.4f\n", nm, t1 - t0, t2 - t1);
    detrc_enc_free(e);
    return 0;
}

static int do_decode(detlm *m, const char *line, FILE *out, float *lg, int32_t *ids, int cap, uint8_t *frame, int fcap) {
    int V = detlm_n_vocab(m), ctx = detlm_n_ctx(m);
    const char *p = line;
    int np = parse_ids(&p, ids, cap);
    if (np <= 0) { fprintf(out, "error bad prime\n"); return -1; }
    int fl = parse_hex(&p, frame, fcap);
    if (fl < 0) { fprintf(out, "error bad frame hex\n"); return -1; }
    if (fl < 4) { fprintf(out, "refused frame too short\n"); return 0; }
    if (frame[0] != CODEC_BYTE) { fprintf(out, "refused not a 0x08 frame\n"); return 0; }
    uint32_t n = 0; int shift = 0, i = 3;
    for (;;) {
        if (i >= fl || shift > 28) { fprintf(out, "refused bad token count\n"); return 0; }
        uint8_t b = frame[i++]; n |= (uint32_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
    }
    if (n > MAX_TOKENS) { fprintf(out, "refused token count %u over the cap %d\n", n, MAX_TOKENS); return 0; }
    if ((int)n + np > ctx) { fprintf(out, "refused token count %u over the context\n", n); return 0; }
    detlm_reset(m);
    double t0 = now();
    if (detlm_eval(m, ids, np, 1, lg) != 0) { fprintf(out, "error prime eval\n"); return -1; }
    double t1 = now();
    detrc_dec *d = detrc_dec_new(frame + i, fl - i);
    for (uint32_t k = 0; k < n; k++) {
        int32_t t = detrc_dec_get_logits(d, lg, V);
        if (t < 0) { fprintf(out, "error dec_get\n"); detrc_dec_free(d); return -1; }
        ids[np + k] = t;
        if (k + 1 < n && detlm_eval(m, &t, 1, 1, lg) != 0) { fprintf(out, "error step eval\n"); detrc_dec_free(d); return -1; }
    }
    double t2 = now();
    detrc_dec_free(d);
    fprintf(out, "ok check=%02x%02x n=%u t_prime=%.4f t_tok=%.4f |", frame[1], frame[2], n, t1 - t0, t2 - t1);
    for (uint32_t k = 0; k < n; k++) fprintf(out, " %d", ids[np + k]);
    fprintf(out, "\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s MODEL (encode|decode) JOB OUT [--threads N] [--ctx N]\n", argv[0]); return 2; }
    const char *model = argv[1], *verb = argv[2], *job = argv[3], *outp = argv[4];
    int threads = 1, ctx = 2048;
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }
    int enc = !strcmp(verb, "encode");
    if (!enc && strcmp(verb, "decode")) { fprintf(stderr, "verb must be encode or decode\n"); return 2; }
    FILE *jf = fopen(job, "r"); if (!jf) { fprintf(stderr, "cannot read %s\n", job); return 1; }
    FILE *out = fopen(outp, "w"); if (!out) { fprintf(stderr, "cannot write %s\n", outp); return 1; }
    double tl = now();
    detlm *m = detlm_load(model, ctx, threads);
    if (!m) return 1;
    fprintf(stderr, "detcodec: detlm %s detrc v%d loaded in %.2fs threads=%d ctx=%d\n", detlm_version(), detrc_version(), now() - tl, threads, ctx);
    int V = detlm_n_vocab(m);
    float *lg = (float *)malloc((size_t)V * sizeof(float));
    int32_t *ids = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
    int fcap = 1 << 16; uint8_t *frame = (uint8_t *)malloc((size_t)fcap);
    size_t cap = 1 << 20; char *line = (char *)malloc(cap);
    int rc = 0, lines = 0;
    while (fgets(line, (int)cap, jf)) {
        const char *q = line; while (*q == ' ') q++;
        if (*q == 0 || *q == '\n' || *q == '#') continue;
        lines++;
        int r = enc ? do_encode(m, line, out, lg, ids, ctx) : do_decode(m, line, out, lg, ids, ctx, frame, fcap);
        if (r != 0) rc = 1;
        fflush(out);
    }
    fprintf(stderr, "detcodec: %d lines, peak_rss_mib=%.0f\n", lines, detlm_peak_rss_kib() / 1024.0);
    fclose(out); fclose(jf);
    free(line); free(frame); free(ids); free(lg);
    detlm_free(m);
    return rc;
}
