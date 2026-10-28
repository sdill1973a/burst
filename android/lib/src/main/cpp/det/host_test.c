/* host_test — the app's 0x08 code path (det08.c + detlm.c + detrc.c, the SAME sources the APK compiles) run on the
 * box against frames the box's det_codec.py produced (golden/export08.py -> golden/host08.jobs).
 *
 *   host_test MODEL.gguf JOBS [--threads N]
 *   JOBS line:  <check hex4> | <prime ids> | <msg ids> | <frame hex>
 *
 * Per line: encode(prime, msg) must equal the frame byte for byte; decode(prime, frame) must give msg exactly;
 * the frame with one middle coder byte flipped must decode to DIFFERENT ids (the harness can fail — a flip that
 * still decodes right is reported, since the 2-byte check in Kotlin is what finally refuses it).
 * Build:  cc -O2 -ffp-contract=off -fno-math-errno -std=c11 -o host_test host_test.c det08.c detlm.c detrc.c -lpthread -lm
 */
#define _POSIX_C_SOURCE 200809L
#include "det08.h"
#include "detlm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

static int parse_ids(const char **pp, int32_t *out, int cap) {
    const char *p = *pp; int n = 0;
    for (;;) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '|') { p++; break; }
        if (*p == 0 || *p == '\n' || *p == '\r') break;
        char *e; long v = strtol(p, &e, 10);
        if (e == p || n >= cap) return -1;
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

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.gguf JOBS [--threads N]\n", argv[0]); return 2; }
    int threads = 2;
    for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
    FILE *jf = fopen(argv[2], "r"); if (!jf) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    double t0 = now();
    detlm *m = detlm_load(argv[1], 2048, threads);
    if (!m) return 1;
    fprintf(stderr, "host_test: detlm %s loaded in %.2fs, threads=%d\n", detlm_version(), now() - t0, threads);
    int ctx = detlm_n_ctx(m);
    int32_t *ids = malloc((size_t)ctx * sizeof(int32_t)), *got = malloc((size_t)ctx * sizeof(int32_t));
    uint8_t *frame = malloc(1 << 16), *mine = malloc(1 << 16), *bad = malloc(1 << 16);
    char *line = malloc(1 << 20);
    int lines = 0, enc_same = 0, dec_same = 0, flip_differs = 0, flip_tested = 0, errors = 0;
    double t_enc = 0, t_dec = 0;
    while (fgets(line, 1 << 20, jf)) {
        const char *p = line; while (*p == ' ') p++;
        if (*p == 0 || *p == '\n' || *p == '#') continue;
        lines++;
        uint8_t chk[2];
        if (parse_hex(&p, chk, 2) != 2) { printf("line %d: bad check\n", lines); errors++; continue; }
        int np = parse_ids(&p, ids, ctx);
        int nm = np > 0 ? parse_ids(&p, ids + np, ctx - np) : -1;
        int fl = nm >= 0 ? parse_hex(&p, frame, 1 << 16) : -1;
        if (np <= 0 || nm < 0 || fl < 0) { printf("line %d: bad job\n", lines); errors++; continue; }
        const char *why = NULL;
        double a = now();
        int ml = det08_encode(m, chk, ids, np, ids + np, nm, mine, 1 << 16, &why);
        t_enc += now() - a;
        int same = ml == fl && memcmp(mine, frame, (size_t)fl) == 0;
        enc_same += same;
        if (!same) printf("line %d: ENCODE DIFFERS (%s) mine=%d B box=%d B\n", lines, why ? why : "bytes", ml, fl);
        a = now();
        int n = det08_decode(m, ids, np, frame, fl, got, ctx, &why);
        t_dec += now() - a;
        int dsame = n == nm && memcmp(got, ids + np, (size_t)nm * sizeof(int32_t)) == 0;
        dec_same += dsame;
        if (!dsame) printf("line %d: DECODE DIFFERS (%s) n=%d want %d\n", lines, why ? why : "ids", n, nm);
        /* negative: flip a middle coder byte */
        uint8_t c2[2]; uint32_t nn; int off;
        if (fl > 6 && det08_parse(frame, fl, c2, &nn, &off, &why) == 0 && fl - off >= 2) {
            memcpy(bad, frame, (size_t)fl); int fi = off + (fl - off) / 2; bad[fi] ^= 0x5A;
            flip_tested++;
            int bn = det08_decode(m, ids, np, bad, fl, got, ctx, &why);
            int differs = bn != nm || memcmp(got, ids + np, (size_t)nm * sizeof(int32_t)) != 0;
            flip_differs += differs;
            if (!differs) printf("line %d: flipped byte %d still decoded to the same ids (check would pass!)\n", lines, fi);
        }
        printf("line %d: n=%d frame=%d B encode %s decode %s\n", lines, nm, fl, same ? "IDENTICAL" : "DIFF", dsame ? "EXACT" : "DIFF");
        fflush(stdout);
    }
    printf("host_test: %d jobs · encode identical %d/%d · decode exact %d/%d · flipped-byte differs %d/%d · errors %d · enc %.2fs dec %.2fs · peak_rss_mib=%.0f\n",
           lines, enc_same, lines, dec_same, lines, flip_differs, flip_tested, errors, t_enc, t_dec, detlm_peak_rss_kib() / 1024.0);
    fclose(jf); detlm_free(m);
    free(ids); free(got); free(frame); free(mine); free(bad); free(line);
    return (enc_same == lines && dec_same == lines && flip_differs == flip_tested && errors == 0 && lines > 0) ? 0 : 1;
}
