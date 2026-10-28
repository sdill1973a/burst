/* detlm_cli — drive detlm over token streams; prints timings and a sha256 of the logits.
 *
 *   detlm_cli MODEL.gguf [--ctx N] [--threads N] [--prompt N] [--out logits.f32] (--streams FILE | --tokens "t0 t1 ...")
 *
 * Stream file: one stream per line:  m t0 t1 ... tk   -> after a reset, the first --prompt N tokens are
 * evaluated as one batch, the rest one at a time (decode path); the logits of the LAST m positions are
 * appended to --out (raw little-endian float32 [m][n_vocab]) and folded into the sha256.
 * --tokens: a single stream with m = all positions.
 */
#define _POSIX_C_SOURCE 200809L
#include "detlm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ---- sha256 (public domain style, straightforward) */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t n; } sha256_t;
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
#define ROR(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void sha256_block(sha256_t *s, const uint8_t *p) {
    uint32_t w[64], a,b,c,d,e,f,g,h;
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i]<<24 | (uint32_t)p[4*i+1]<<16 | (uint32_t)p[4*i+2]<<8 | p[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i-15],7) ^ ROR(w[i-15],18) ^ (w[i-15]>>3);
        uint32_t s1 = ROR(w[i-2],17) ^ ROR(w[i-2],19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a=s->h[0]; b=s->h[1]; c=s->h[2]; d=s->h[3]; e=s->h[4]; f=s->h[5]; g=s->h[6]; h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROR(e,6) ^ ROR(e,11) ^ ROR(e,25), ch = (e & f) ^ (~e & g), t1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = ROR(a,2) ^ ROR(a,13) ^ ROR(a,22), mj = (a & b) ^ (a & c) ^ (b & c), t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d; s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}
static void sha256_init(sha256_t *s) {
    static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h, iv, sizeof iv); s->len = 0; s->n = 0;
}
static void sha256_update(sha256_t *s, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data; s->len += n;
    while (n) {
        size_t take = 64 - s->n; if (take > n) take = n;
        memcpy(s->buf + s->n, p, take); s->n += take; p += take; n -= take;
        if (s->n == 64) { sha256_block(s, s->buf); s->n = 0; }
    }
}
static void sha256_final(sha256_t *s, char hex[65]) {
    uint64_t bits = s->len * 8; uint8_t pad = 0x80;
    sha256_update(s, &pad, 1);
    uint8_t z = 0; while (s->n != 56) sha256_update(s, &z, 1);
    uint8_t L[8]; for (int i = 0; i < 8; i++) L[i] = (uint8_t)(bits >> (56 - 8*i));
    sha256_update(s, L, 8);
    for (int i = 0; i < 8; i++) sprintf(hex + 8*i, "%08x", s->h[i]);
    hex[64] = 0;
}

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

/* feed logits into the hash as little-endian bytes regardless of host endianness */
static void hash_logits(sha256_t *s, const float *lg, size_t n) {
    uint8_t buf[4096]; size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t b; memcpy(&b, &lg[i], 4);
        buf[k++] = (uint8_t)b; buf[k++] = (uint8_t)(b >> 8); buf[k++] = (uint8_t)(b >> 16); buf[k++] = (uint8_t)(b >> 24);
        if (k == sizeof buf) { sha256_update(s, buf, k); k = 0; }
    }
    if (k) sha256_update(s, buf, k);
}

static void write_le(FILE *f, const float *lg, size_t n) {
    for (size_t i = 0; i < n; i++) { uint32_t b; memcpy(&b, &lg[i], 4);
        uint8_t o[4] = { (uint8_t)b, (uint8_t)(b >> 8), (uint8_t)(b >> 16), (uint8_t)(b >> 24) }; fwrite(o, 1, 4, f); }
}

static int run_stream(detlm *m, const int32_t *tok, int n, int m_last, int prompt_n, FILE *out, sha256_t *total,
                      double *t_prompt, double *t_decode, int *n_prompt, int *n_decode) {
    int V = detlm_n_vocab(m);
    if (m_last > n) m_last = n;
    if (prompt_n > n) prompt_n = n;
    float *lg = (float *)malloc((size_t)(m_last > 0 ? m_last : 1) * (size_t)V * sizeof(float));
    int first = n - m_last;                               /* first position whose logits we keep */
    detlm_reset(m);
    double t0 = now();
    int nl = prompt_n - first; if (nl < 0) nl = 0;
    if (detlm_eval(m, tok, prompt_n, nl, lg) != 0) { free(lg); return -1; }
    double t1 = now();
    for (int i = prompt_n; i < n; i++) {
        int want = i >= first;
        if (detlm_eval(m, tok + i, 1, want, want ? lg + (size_t)(i - first) * (size_t)V : NULL) != 0) { free(lg); return -1; }
    }
    double t2 = now();
    *t_prompt += t1 - t0; *t_decode += t2 - t1; *n_prompt += prompt_n; *n_decode += n - prompt_n;
    sha256_t one; sha256_init(&one);
    hash_logits(&one, lg, (size_t)m_last * (size_t)V);
    hash_logits(total, lg, (size_t)m_last * (size_t)V);
    char hex[65]; sha256_final(&one, hex);
    if (out) write_le(out, lg, (size_t)m_last * (size_t)V);
    printf("stream n=%d prompt=%d decode=%d logits=%d t_prompt=%.3f t_decode=%.3f sha256=%s\n",
           n, prompt_n, n - prompt_n, m_last, t1 - t0, t2 - t1, hex);
    fflush(stdout);
    free(lg);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [--ctx N] [--threads N] [--prompt N] [--out F] (--streams F | --tokens \"...\")\n", argv[0]); return 2; }
    const char *model = argv[1], *streams = NULL, *tokens = NULL, *outp = NULL;
    int ctx = 2048, threads = 1, prompt_n = 1 << 30;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--streams") && i + 1 < argc) streams = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = argv[++i];
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }
    double tl = now();
    detlm *m = detlm_load(model, ctx, threads);
    if (!m) return 1;
    printf("detlm %s loaded in %.2fs n_vocab=%d ctx=%d threads=%d\n", detlm_version(), now() - tl, detlm_n_vocab(m), ctx, threads);
    FILE *out = outp ? fopen(outp, "wb") : NULL;
    if (outp && !out) { fprintf(stderr, "cannot write %s\n", outp); return 1; }
    sha256_t total; sha256_init(&total);
    double tp = 0, td = 0; int np = 0, nd = 0; int rc = 0;
    int32_t *tok = (int32_t *)malloc(sizeof(int32_t) * (size_t)ctx);
    if (tokens) {
        int n = 0; const char *p = tokens; char *e;
        while (n < ctx) { long v = strtol(p, &e, 10); if (e == p) break; tok[n++] = (int32_t)v; p = e; }
        rc = run_stream(m, tok, n, n, prompt_n, out, &total, &tp, &td, &np, &nd);
    } else if (streams) {
        FILE *f = fopen(streams, "r");
        if (!f) { fprintf(stderr, "cannot read %s\n", streams); return 1; }
        size_t cap = 1 << 20; char *line = (char *)malloc(cap);
        while (fgets(line, (int)cap, f)) {
            const char *p = line; char *e;
            long ml = strtol(p, &e, 10); if (e == p) continue; p = e;
            int n = 0;
            while (n < ctx) { long v = strtol(p, &e, 10); if (e == p) break; tok[n++] = (int32_t)v; p = e; }
            if (n == 0) continue;
            if ((rc = run_stream(m, tok, n, (int)ml, prompt_n, out, &total, &tp, &td, &np, &nd)) != 0) break;
        }
        free(line); fclose(f);
    } else { fprintf(stderr, "need --streams or --tokens\n"); return 2; }
    if (out) fclose(out);
    char hex[65]; sha256_final(&total, hex);
    printf("TOTAL prompt_tokens=%d t_prompt=%.3f (%.1f tok/s) decode_tokens=%d t_decode=%.3f (%.1f tok/s, %.1f ms/tok) peak_rss_mib=%.0f\n",
           np, tp, tp > 0 ? np / tp : 0.0, nd, td, td > 0 ? nd / td : 0.0, nd > 0 ? 1000.0 * td / nd : 0.0, detlm_peak_rss_kib() / 1024.0);
    printf("SHA256 %s\n", hex);
    free(tok);
    detlm_free(m);
    return rc ? 1 : 0;
}
