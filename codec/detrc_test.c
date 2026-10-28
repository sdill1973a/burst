/* detrc_test.c — the cross-architecture harness for detrc: same inputs in, hash of the freq tables and the coded bytes
 * out, decoded back in C and compared. Build it twice (x86 gcc, aarch64 NDK clang -static under qemu) and cmp.
 *
 *   detrc_test synth <freq.bin> <rc.bin>                    synthetic rows (integer-generated float bit patterns + edge rows)
 *   detrc_test file <logits.f32> <V> <syms.u32> <freq.bin> <rc.bin>   real rows (V float32 each) + one uint32 symbol per row
 *
 * -DCONTROL_FLOAT swaps in a float softmax (libm expf, float sums) for the frequency function: the negative control —
 * if that build's hashes ALSO agree across chips the harness could not see a difference, so it proves nothing.
 */
#include "detrc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef CONTROL_FLOAT
#include <math.h>
static int freq_fn(const float *lg, int32_t n, uint32_t *freq) {
    float m = lg[0];
    for (int32_t i = 1; i < n; i++) if (lg[i] > m) m = lg[i];
    float s = 0.f;
    for (int32_t i = 0; i < n; i++) s += expf(lg[i] - m);
    uint64_t used = 0; int32_t imax = 0;
    for (int32_t i = 0; i < n; i++) {
        float p = expf(lg[i] - m) / s;
        uint32_t f = 1 + (uint32_t)(p * (float)(DETRC_TOTAL - n));
        freq[i] = f; used += f;
        if (lg[i] > lg[imax]) imax = i;
    }
    if (used > DETRC_TOTAL) return -1;
    freq[imax] += (uint32_t)(DETRC_TOTAL - used);
    return 0;
}
#else
#define freq_fn detrc_freq
#endif

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static uint32_t lcg_state = 20261024u;
static uint32_t lcg(void) { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }

static float bits_to_float(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

/* synthetic rows: V = 151936; 40 rows of random float bit patterns with sane exponents (|x| ~ 2^-8 .. 2^8), then edge rows */
#define SV 151936
static int synth(float *rows, int32_t *syms, int maxrows) {
    int r = 0;
    for (; r < 40; r++) {
        float *row = rows + (size_t)r * SV;
        uint32_t spread = 112 + (lcg() >> 28);                      /* exponent base 112..127 -> scale 2^-15 .. 2^0 */
        for (int32_t i = 0; i < SV; i++) {
            uint32_t x = lcg();
            uint32_t e = spread + ((x >> 23) & 7);
            row[i] = bits_to_float((x & 0x807FFFFFu) | (e << 23));
        }
        row[lcg() % SV] = bits_to_float(0x41A00000u);                /* 20.0: a peak somewhere */
        syms[r] = (int32_t)(lcg() % SV);
    }
    /* edge rows */
    float *row;
    row = rows + (size_t)r * SV; for (int32_t i = 0; i < SV; i++) row[i] = 0.f;           syms[r++] = 7;      /* flat */
    row = rows + (size_t)r * SV; for (int32_t i = 0; i < SV; i++) row[i] = 0.f; row[5] = 1e4f; syms[r++] = 5;  /* one certain */
    row = rows + (size_t)r * SV; for (int32_t i = 0; i < SV; i++) row[i] = 0.f; row[5] = 1e4f; syms[r++] = 6;  /* the floor */
    row = rows + (size_t)r * SV; for (int32_t i = 0; i < SV; i++) row[i] = bits_to_float(0x7FC00000u); syms[r++] = 9; /* all NaN */
    row = rows + (size_t)r * SV; for (int32_t i = 0; i < SV; i++) row[i] = 0.f; row[1] = bits_to_float(0x7F800000u); row[2] = bits_to_float(0xFF800000u); syms[r++] = 2; /* +inf, -inf */
    row = rows + (size_t)r * SV; for (int32_t i = 0; i < SV; i++) row[i] = bits_to_float(0x00000001u); row[3] = -1e-30f; syms[r++] = 3; /* denormals */
    row = rows + (size_t)r * SV; for (int32_t i = 0; i < SV; i++) row[i] = -3.0e38f; row[0] = 3.0e38f; syms[r++] = SV - 1; /* huge spread, last sym */
    for (int k = 0; k < 20 && r < maxrows; k++, r++) {               /* near-certain streaks: carries and 0xFF runs */
        row = rows + (size_t)r * SV; for (int32_t i = 0; i < SV; i++) row[i] = -30.f; row[k] = 25.f; syms[r] = k;
    }
    return r;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: see header\n"); return 2; }
    float *rows; int32_t *syms; int32_t V; int R;
    const char *ffreq, *frc;
    if ((!strcmp(argv[1], "synth") || !strcmp(argv[1], "synth1ulp")) && argc == 4) {
        V = SV; R = 80;
        rows = (float *)malloc(sizeof(float) * (size_t)V * R); syms = (int32_t *)malloc(sizeof(int32_t) * R);
        R = synth(rows, syms, R); ffreq = argv[2]; frc = argv[3];
        if (argv[1][5] == '1') {                                      /* control: ONE input float changed (row 3, token 1234 -> 21.0) */
            /* measured 10-24: a 1-ulp or 2^12-ulp move on these |x| <= 2^7 rows is below the Q16.16 quantum and changed
             * nothing — by design (sub-quantum float noise cannot reach the bytes); a real change must show in the hash */
            rows[(size_t)3 * SV + 1234] = bits_to_float(0x41A80000u);
        }
    } else if (!strcmp(argv[1], "file") && argc == 7) {
        V = atoi(argv[3]); ffreq = argv[5]; frc = argv[6];
        FILE *f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 2; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        R = (int)(sz / ((long)V * 4));
        rows = (float *)malloc((size_t)sz); if (fread(rows, 1, (size_t)sz, f) != (size_t)sz) return 2; fclose(f);
        syms = (int32_t *)malloc(sizeof(int32_t) * R);
        f = fopen(argv[4], "rb"); if (!f) { perror(argv[4]); return 2; }
        if (fread(syms, 4, (size_t)R, f) != (size_t)R) { fprintf(stderr, "syms short\n"); return 2; } fclose(f);
    } else { fprintf(stderr, "usage: see header\n"); return 2; }

    uint32_t *freq = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)V);
    FILE *of = fopen(ffreq, "wb");
    detrc_enc *e = detrc_enc_new();
    uint64_t hf = 1469598103934665603ULL;
    double ideal_bits = 0;
    for (int r = 0; r < R; r++) {
        if (freq_fn(rows + (size_t)r * V, V, freq)) { fprintf(stderr, "freq failed row %d\n", r); return 1; }
        uint64_t sum = 0; for (int32_t i = 0; i < V; i++) { if (!freq[i]) { fprintf(stderr, "zero freq\n"); return 1; } sum += freq[i]; }
        if (sum != DETRC_TOTAL) { fprintf(stderr, "row %d sum %llu != total\n", r, (unsigned long long)sum); return 1; }
        hf = fnv(hf, freq, sizeof(uint32_t) * (size_t)V);
        fwrite(freq, sizeof(uint32_t), (size_t)V, of);
        if (detrc_enc_put(e, freq, V, syms[r])) { fprintf(stderr, "put failed row %d\n", r); return 1; }
    }
    fclose(of);
    int64_t n = detrc_enc_finish(e);
    FILE *orc = fopen(frc, "wb"); fwrite(detrc_enc_data(e), 1, (size_t)n, orc); fclose(orc);
    uint64_t hr = fnv(1469598103934665603ULL, detrc_enc_data(e), (size_t)n);
    /* decode back */
    detrc_dec *d = detrc_dec_new(detrc_enc_data(e), n);
    int bad = 0;
    for (int r = 0; r < R; r++) {
        freq_fn(rows + (size_t)r * V, V, freq);
        int32_t s = detrc_dec_get(d, freq, V);
        if (s != syms[r]) { bad++; if (bad < 5) fprintf(stderr, "row %d: decoded %d want %d\n", r, s, syms[r]); }
    }
    printf("rows=%d V=%d freq_fnv=%016llx rc_bytes=%lld rc_fnv=%016llx roundtrip=%s\n", R, V,
           (unsigned long long)hf, (long long)n, (unsigned long long)hr, bad ? "FAIL" : "ok");
    (void)ideal_bits;
    detrc_dec_free(d); detrc_enc_free(e); free(freq); free(rows); free(syms);
    return bad ? 1 : 0;
}
