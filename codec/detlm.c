/* detlm.c — deterministic scalar inference for Qwen2.5-0.5B q8_0 (GGUF).
 *
 * Determinism rules:
 *  - float32 activations; every reduction in the fixed order written here;
 *  - build with -O2 -ffp-contract=off -fno-math-errno, never -ffast-math / -Ofast;
 *  - no libm transcendentals: own expf, own double sin/cos (RoPE table), own rounding;
 *    sqrtf and division are IEEE correctly rounded and allowed;
 *  - matmul inputs are quantized to int8 blocks of 32 exactly like llama.cpp's q8_0
 *    activation path: the 32-term int8*int8 block dot is an exact int32 (order-free),
 *    the cross-block accumulation is float32 in ascending block order;
 *  - threads split OUTPUT ROWS only; no reduction ever crosses a thread.
 * Negative controls: -DDETLM_REORDER_SUM accumulates blocks in reverse order
 * (must change the logits hash); -ffp-contract=fast on aarch64 fuses mul+add (ditto).
 */
#define _GNU_SOURCE
#include "detlm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>      /* sqrtf only */
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define DETLM_VERSION "0.1.0"
#define QK 32          /* q8_0 block size */
#define BLK 34         /* bytes per q8_0 block: f16 d + 32 int8 */
#define MAXNB 256      /* max blocks per row we support (n_ff=4864 -> 152) */
#define CHUNK 32       /* tokens per internal batch chunk (bounds scratch memory) */

/* ------------------------------------------------------------------ bit helpers */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    float f;
    if (exp == 0) {                       /* zero / subnormal: mant * 2^-24, exact */
        f = (float)mant * 5.9604644775390625e-08f;
        if (sign) f = -f;
        return f;
    }
    if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);   /* inf / nan */
    else           bits = sign | ((exp + 112u) << 23) | (mant << 13);
    memcpy(&f, &bits, 4);
    return f;
}

/* float32 -> float16, round to nearest even, bit exact (no float ops) */
static inline uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = x & 0x80000000u;
    x ^= sign;
    uint16_t h;
    if (x > 0x7f800000u) h = 0x7e00;                 /* nan */
    else if (x >= 0x477ff000u) h = 0x7c00;           /* >= 65520 rounds to inf */
    else if (x < 0x38800000u) {                      /* below 2^-14: half subnormal */
        if (x < 0x33000000u) h = 0;                  /* < 2^-26 -> 0 (2^-25 is a tie -> even -> 0 too) */
        else {
            uint32_t e = x >> 23;                    /* 102..112 */
            uint32_t m = (x & 0x7fffffu) | 0x800000u;
            uint32_t shift = 126u - e;               /* 14..24 */
            uint32_t q = m >> shift;
            uint32_t rem = m & ((1u << shift) - 1u);
            uint32_t half = 1u << (shift - 1);
            if (rem > half || (rem == half && (q & 1u))) q++;
            h = (uint16_t)q;                         /* q may become 0x400 = smallest normal */
        }
    } else {
        uint32_t r = x + 0xfffu + ((x >> 13) & 1u) - (112u << 23);
        h = (uint16_t)(r >> 13);
    }
    return (uint16_t)(h | (sign >> 16));
}

/* round half away from zero, exact (|x| < 2^23; larger values are already integers) */
static inline float det_roundf(float x) {
    if (x >= 8388608.0f || x <= -8388608.0f) return x;
    float t = (float)(int32_t)x;          /* truncation toward zero */
    float fr = x - t;                     /* exact */
    if (fr >= 0.5f) t = t + 1.0f;
    else if (fr <= -0.5f) t = t - 1.0f;
    return t;
}

static inline double det_round(double x) {
    if (x >= 4503599627370496.0 || x <= -4503599627370496.0) return x;
    double t = (double)(int64_t)x;
    double fr = x - t;
    if (fr >= 0.5) t = t + 1.0;
    else if (fr <= -0.5) t = t - 1.0;
    return t;
}

/* exp(x) for float, own implementation: Cody-Waite reduction + degree-7 Taylor + exponent scaling. */
static float det_expf(float x) {
    if (x > 88.0f) x = 88.0f;
    if (x < -87.0f) return 0.0f;
    float kf = det_roundf(x * 1.44269502162933349609375f);   /* 1/ln2 */
    float r  = x - kf * 0.693145751953125f;                   /* ln2_hi, 12 significant bits */
    r = r - kf * 1.42860676533018727786838531494140625e-06f;  /* ln2_lo */
    float p = 1.0f / 5040.0f;
    p = p * r + (1.0f / 720.0f);
    p = p * r + (1.0f / 120.0f);
    p = p * r + (1.0f / 24.0f);
    p = p * r + (1.0f / 6.0f);
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    int32_t k = (int32_t)kf;
    uint32_t bits = (uint32_t)(k + 127) << 23;
    float scale; memcpy(&scale, &bits, 4);
    return p * scale;
}

/* sin and cos of a double, own implementation (fdlibm-style pi/2 reduction, Taylor on |r|<=pi/4).
 * Used only to build the RoPE table at load; accuracy ~1e-14, cast to float afterwards. */
static void det_sincos(double x, double *s_out, double *c_out) {
    const double invpio2 = 0x1.45f306dc9c883p-1;   /* 2/pi */
    const double pio2_1  = 0x1.921fb544p+0;        /* first 33 bits of pi/2 */
    const double pio2_1t = 0x1.0b4611a626331p-34;  /* pi/2 - pio2_1 */
    double fn = det_round(x * invpio2);
    double r  = x - fn * pio2_1;                   /* exact: fn < 2^20, pio2_1 has 33 bits */
    r = r - fn * pio2_1t;
    int64_t n = (int64_t)fn;
    double r2 = r * r;
    /* sin r = r * (1 - r2/6 + r2^2/120 - ...) up to r^13 */
    double sp = 1.0 / 6227020800.0;
    sp = sp * r2 - (1.0 / 39916800.0);
    sp = sp * r2 + (1.0 / 362880.0);
    sp = sp * r2 - (1.0 / 5040.0);
    sp = sp * r2 + (1.0 / 120.0);
    sp = sp * r2 - (1.0 / 6.0);
    sp = sp * r2 + 1.0;
    double sr = r * sp;
    /* cos r = 1 - r2/2 + r2^2/24 - ... up to r^14 */
    double cp = -1.0 / 87178291200.0;
    cp = cp * r2 + (1.0 / 479001600.0);
    cp = cp * r2 - (1.0 / 3628800.0);
    cp = cp * r2 + (1.0 / 40320.0);
    cp = cp * r2 - (1.0 / 720.0);
    cp = cp * r2 + (1.0 / 24.0);
    cp = cp * r2 - 0.5;
    cp = cp * r2 + 1.0;
    double cr = cp;
    switch (n & 3) {
        case 0: *s_out = sr;  *c_out = cr;  break;
        case 1: *s_out = cr;  *c_out = -sr; break;
        case 2: *s_out = -sr; *c_out = -cr; break;
        default:*s_out = -cr; *c_out = sr;  break;
    }
}

/* ------------------------------------------------------------------ thread pool */
typedef void (*job_fn)(void *arg, int i0, int i1);

struct pool {
    int n;                      /* total threads incl. main */
    pthread_t *th;
    pthread_mutex_t mu;
    pthread_cond_t cv_start, cv_done;
    int gen, n_done, stop;
    job_fn fn; void *arg; int n_items;
};

struct worker_arg { struct pool *p; int idx; };

static void pool_chunk(struct pool *p, int idx) {
    int i0 = (int)((long)p->n_items * idx / p->n);
    int i1 = (int)((long)p->n_items * (idx + 1) / p->n);
    if (i1 > i0) p->fn(p->arg, i0, i1);
}

static void *worker_main(void *a) {
    struct worker_arg *wa = (struct worker_arg *)a;
    struct pool *p = wa->p; int idx = wa->idx; free(wa);
    int my_gen = 0;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (p->gen == my_gen && !p->stop) pthread_cond_wait(&p->cv_start, &p->mu);
        if (p->stop) { pthread_mutex_unlock(&p->mu); return NULL; }
        my_gen = p->gen;
        pthread_mutex_unlock(&p->mu);
        pool_chunk(p, idx);
        pthread_mutex_lock(&p->mu);
        p->n_done++;
        pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
}

static int pool_start(struct pool *p, int n) {
    memset(p, 0, sizeof *p);
    if (n < 1) n = 1;
    p->n = n;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_start, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    if (n > 1) {
        p->th = (pthread_t *)calloc((size_t)n, sizeof(pthread_t));
        for (int i = 1; i < n; i++) {
            struct worker_arg *wa = (struct worker_arg *)malloc(sizeof *wa);
            wa->p = p; wa->idx = i;
            if (pthread_create(&p->th[i], NULL, worker_main, wa) != 0) { free(wa); p->n = i; break; }
        }
    }
    return 0;
}

static void pool_stop(struct pool *p) {
    pthread_mutex_lock(&p->mu);
    p->stop = 1;
    pthread_cond_broadcast(&p->cv_start);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->th[i], NULL);
    free(p->th);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_start);
    pthread_cond_destroy(&p->cv_done);
}

static void pool_run(struct pool *p, job_fn fn, void *arg, int n_items) {
    if (p->n == 1 || n_items < p->n) { fn(arg, 0, n_items); return; }
    pthread_mutex_lock(&p->mu);
    p->fn = fn; p->arg = arg; p->n_items = n_items; p->n_done = 0; p->gen++;
    pthread_cond_broadcast(&p->cv_start);
    pthread_mutex_unlock(&p->mu);
    pool_chunk(p, 0);
    pthread_mutex_lock(&p->mu);
    while (p->n_done < p->n - 1) pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

/* ------------------------------------------------------------------ model */
struct layer {
    const uint8_t *wq, *wk, *wv, *wo, *wg, *wu, *wd;   /* q8_0 */
    const float *bq, *bk, *bv, *attn_norm, *ffn_norm;  /* f32 */
};

struct detlm {
    int n_embd, n_ff, n_head, n_head_kv, n_layer, n_vocab, head_dim, n_rot, n_ctx, kv_dim;
    float eps, rope_base;
    const uint8_t *tok_embd, *output;
    const float *output_norm;
    struct layer *layers;
    float *kcache, *vcache;        /* [n_layer][n_ctx][kv_dim] */
    float *rope_cos, *rope_sin;    /* [n_ctx][n_rot/2] */
    int n_pos;
    void *map; size_t map_len;
    struct pool pool;
    int trace;
};

/* ---- GGUF reader */
struct rd { const uint8_t *p; size_t n, pos; int err; };
static uint64_t rd_u(struct rd *r, int nbytes) {
    if (r->pos + (size_t)nbytes > r->n) { r->err = 1; return 0; }
    uint64_t v = 0;
    for (int i = 0; i < nbytes; i++) v |= (uint64_t)r->p[r->pos + (size_t)i] << (8 * i);
    r->pos += (size_t)nbytes;
    return v;
}
static const char *rd_str(struct rd *r, uint64_t *len) {
    uint64_t n = rd_u(r, 8);
    if (r->err || r->pos + n > r->n) { r->err = 1; *len = 0; return ""; }
    const char *s = (const char *)(r->p + r->pos);
    r->pos += n; *len = n; return s;
}
/* read a scalar value as double (strings/arrays are skipped and return 0) */
static double rd_value(struct rd *r, uint32_t t) {
    uint64_t len;
    switch (t) {
        case 0: return (double)(uint8_t)rd_u(r, 1);
        case 1: return (double)(int8_t)rd_u(r, 1);
        case 2: return (double)(uint16_t)rd_u(r, 2);
        case 3: return (double)(int16_t)rd_u(r, 2);
        case 4: return (double)(uint32_t)rd_u(r, 4);
        case 5: return (double)(int32_t)rd_u(r, 4);
        case 6: { uint32_t b = (uint32_t)rd_u(r, 4); float f; memcpy(&f, &b, 4); return (double)f; }
        case 7: return (double)(uint8_t)rd_u(r, 1);
        case 8: rd_str(r, &len); return 0;
        case 9: { uint32_t et = (uint32_t)rd_u(r, 4); uint64_t n = rd_u(r, 8);
                  for (uint64_t i = 0; i < n && !r->err; i++) rd_value(r, et);
                  return 0; }
        case 10: return (double)(uint64_t)rd_u(r, 8);
        case 11: return (double)(int64_t)rd_u(r, 8);
        case 12: { uint64_t b = rd_u(r, 8); double d; memcpy(&d, &b, 8); return d; }
        default: r->err = 1; return 0;
    }
}

struct tinfo { const char *name; uint64_t nlen; uint32_t ndim; uint64_t ne[4]; uint32_t type; uint64_t off; };

static const struct tinfo *find_tensor(const struct tinfo *t, int n, const char *name) {
    size_t L = strlen(name);
    for (int i = 0; i < n; i++) if (t[i].nlen == L && memcmp(t[i].name, name, L) == 0) return &t[i];
    return NULL;
}

static const void *need(const struct tinfo *t, int n, const uint8_t *base, size_t base_len, const char *name,
                        uint32_t type, uint64_t ne0, uint64_t ne1) {
    const struct tinfo *ti = find_tensor(t, n, name);
    if (!ti) { fprintf(stderr, "detlm: missing tensor %s\n", name); return NULL; }
    if (ti->type != type) { fprintf(stderr, "detlm: tensor %s has type %u, want %u\n", name, ti->type, type); return NULL; }
    if (ti->ne[0] != ne0 || (ne1 && (ti->ndim < 2 || ti->ne[1] != ne1)) || (!ne1 && ti->ndim != 1)) {
        fprintf(stderr, "detlm: tensor %s has unexpected shape\n", name); return NULL;
    }
    uint64_t nel = ne0 * (ne1 ? ne1 : 1);
    uint64_t bytes = (type == 8) ? (nel / QK) * BLK : nel * 4;
    if (type == 8 && (ne0 % QK) != 0) { fprintf(stderr, "detlm: %s not block aligned\n", name); return NULL; }
    if (ti->off + bytes > base_len) { fprintf(stderr, "detlm: tensor %s out of file bounds\n", name); return NULL; }
    return base + ti->off;
}

const char *detlm_version(void) { return DETLM_VERSION; }
int detlm_n_vocab(const detlm *m) { return m->n_vocab; }
int detlm_n_ctx(const detlm *m) { return m->n_ctx; }
int detlm_n_pos(const detlm *m) { return m->n_pos; }
void detlm_reset(detlm *m) { m->n_pos = 0; }
int detlm_rewind(detlm *m, int n_pos) { if (n_pos < 0 || n_pos > m->n_pos) return -1; m->n_pos = n_pos; return 0; }
int detlm_set_threads(detlm *m, int n) { pool_stop(&m->pool); pool_start(&m->pool, n); return m->pool.n; }

long detlm_peak_rss_kib(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256]; long v = 0;
    while (fgets(line, sizeof line, f)) if (strncmp(line, "VmHWM:", 6) == 0) { v = atol(line + 6); break; }
    fclose(f);
    return v;
}

detlm *detlm_load(const char *path, int n_ctx, int n_threads) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "detlm: cannot open %s\n", path); return NULL; }
    struct stat st; fstat(fd, &st);
    size_t len = (size_t)st.st_size;
    void *map = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { fprintf(stderr, "detlm: mmap failed\n"); return NULL; }
    const uint8_t *base = (const uint8_t *)map;
    struct rd r = { base, len, 0, 0 };
    if (len < 24 || memcmp(base, "GGUF", 4) != 0) { fprintf(stderr, "detlm: not a GGUF file\n"); munmap(map, len); return NULL; }
    r.pos = 4;
    uint32_t version = (uint32_t)rd_u(&r, 4);
    uint64_t n_tensors = rd_u(&r, 8), n_kv = rd_u(&r, 8);
    if (version < 2 || n_tensors > 100000 || n_kv > 100000) { fprintf(stderr, "detlm: bad GGUF header\n"); munmap(map, len); return NULL; }

    detlm *m = (detlm *)calloc(1, sizeof *m);
    m->map = map; m->map_len = len;
    uint32_t alignment = 32;
    int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0; double eps = 1e-6, rope_base = 10000.0;
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        uint64_t klen; const char *k = rd_str(&r, &klen);
        uint32_t t = (uint32_t)rd_u(&r, 4);
        if (r.err) break;
#define KEY(s) (klen == strlen(s) && memcmp(k, s, klen) == 0)
        double v = rd_value(&r, t);
        if (KEY("general.alignment")) alignment = (uint32_t)v;
        else if (KEY("qwen2.block_count")) n_layer = (int)v;
        else if (KEY("qwen2.embedding_length")) n_embd = (int)v;
        else if (KEY("qwen2.feed_forward_length")) n_ff = (int)v;
        else if (KEY("qwen2.attention.head_count")) n_head = (int)v;
        else if (KEY("qwen2.attention.head_count_kv")) n_head_kv = (int)v;
        else if (KEY("qwen2.attention.layer_norm_rms_epsilon")) eps = v;
        else if (KEY("qwen2.rope.freq_base")) rope_base = v;
        else if (KEY("general.architecture")) { /* string skipped by rd_value; checked via tensor names */ }
#undef KEY
    }
    struct tinfo *ti = (struct tinfo *)calloc((size_t)n_tensors, sizeof *ti);
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        ti[i].name = rd_str(&r, &ti[i].nlen);
        ti[i].ndim = (uint32_t)rd_u(&r, 4);
        if (ti[i].ndim > 4) { r.err = 1; break; }
        for (uint32_t d = 0; d < ti[i].ndim; d++) ti[i].ne[d] = rd_u(&r, 8);
        ti[i].type = (uint32_t)rd_u(&r, 4);
        ti[i].off = rd_u(&r, 8);
    }
    if (r.err) { fprintf(stderr, "detlm: truncated GGUF header\n"); goto fail; }
    size_t data_start = (r.pos + alignment - 1) / alignment * alignment;
    const uint8_t *data = base + data_start; size_t data_len = len - data_start;

    if (!n_layer || !n_embd || !n_ff || !n_head || !n_head_kv) { fprintf(stderr, "detlm: not a qwen2 GGUF (missing qwen2.* keys)\n"); goto fail; }
    m->n_layer = n_layer; m->n_embd = n_embd; m->n_ff = n_ff; m->n_head = n_head; m->n_head_kv = n_head_kv;
    m->head_dim = n_embd / n_head; m->n_rot = m->head_dim; m->kv_dim = n_head_kv * m->head_dim;
    m->eps = (float)eps; m->rope_base = (float)rope_base;
    if (m->head_dim * n_head != n_embd || n_head % n_head_kv != 0 || m->n_rot != 64 || n_embd % QK || n_ff % QK) {
        fprintf(stderr, "detlm: unsupported geometry (this engine is for Qwen2.5-0.5B: head_dim 64)\n"); goto fail;
    }
    {
        const struct tinfo *e = find_tensor(ti, (int)n_tensors, "token_embd.weight");
        if (!e || e->ndim != 2) { fprintf(stderr, "detlm: no token_embd.weight\n"); goto fail; }
        m->n_vocab = (int)e->ne[1];
    }
    m->tok_embd = (const uint8_t *)need(ti, (int)n_tensors, data, data_len, "token_embd.weight", 8, (uint64_t)n_embd, (uint64_t)m->n_vocab);
    /* tied embeddings (Qwen2.5-0.5B): no output.weight in the file, the output head is token_embd */
    m->output = find_tensor(ti, (int)n_tensors, "output.weight")
        ? (const uint8_t *)need(ti, (int)n_tensors, data, data_len, "output.weight", 8, (uint64_t)n_embd, (uint64_t)m->n_vocab)
        : m->tok_embd;
    m->output_norm = (const float *)need(ti, (int)n_tensors, data, data_len, "output_norm.weight", 0, (uint64_t)n_embd, 0);
    if (!m->tok_embd || !m->output || !m->output_norm) goto fail;
    m->layers = (struct layer *)calloc((size_t)n_layer, sizeof *m->layers);
    for (int l = 0; l < n_layer; l++) {
        char nm[96]; struct layer *L = &m->layers[l];
#define T(field, suffix, type, ne0, ne1) \
        snprintf(nm, sizeof nm, "blk.%d.%s", l, suffix); \
        L->field = (const void *)need(ti, (int)n_tensors, data, data_len, nm, type, (uint64_t)(ne0), (uint64_t)(ne1)); \
        if (!L->field) goto fail;
        T(wq, "attn_q.weight", 8, n_embd, n_embd)
        T(wk, "attn_k.weight", 8, n_embd, m->kv_dim)
        T(wv, "attn_v.weight", 8, n_embd, m->kv_dim)
        T(wo, "attn_output.weight", 8, n_embd, n_embd)
        T(wg, "ffn_gate.weight", 8, n_embd, n_ff)
        T(wu, "ffn_up.weight", 8, n_embd, n_ff)
        T(wd, "ffn_down.weight", 8, n_ff, n_embd)
        T(bq, "attn_q.bias", 0, n_embd, 0)
        T(bk, "attn_k.bias", 0, m->kv_dim, 0)
        T(bv, "attn_v.bias", 0, m->kv_dim, 0)
        T(attn_norm, "attn_norm.weight", 0, n_embd, 0)
        T(ffn_norm, "ffn_norm.weight", 0, n_embd, 0)
#undef T
    }
    free(ti); ti = NULL;

    if (n_ctx < 1) n_ctx = 2048;
    m->n_ctx = n_ctx;
    m->kcache = (float *)calloc((size_t)n_layer * (size_t)n_ctx * (size_t)m->kv_dim, sizeof(float));
    m->vcache = (float *)calloc((size_t)n_layer * (size_t)n_ctx * (size_t)m->kv_dim, sizeof(float));
    m->rope_cos = (float *)malloc((size_t)n_ctx * (size_t)(m->n_rot / 2) * sizeof(float));
    m->rope_sin = (float *)malloc((size_t)n_ctx * (size_t)(m->n_rot / 2) * sizeof(float));
    if (!m->kcache || !m->vcache || !m->rope_cos || !m->rope_sin) { fprintf(stderr, "detlm: out of memory\n"); goto fail; }
    {
        /* inv_freq[i] = base^(-i/32): the 32nd root of base is five nested correctly-rounded sqrt. */
        double root = (double)m->rope_base;
        for (int s = 0; s < 5; s++) root = sqrt(root);
        double inv_root = 1.0 / root;
        double inv_freq[32];
        inv_freq[0] = 1.0;
        for (int i = 1; i < 32; i++) inv_freq[i] = inv_freq[i - 1] * inv_root;
        for (int p = 0; p < n_ctx; p++) {
            for (int i = 0; i < 32; i++) {
                double theta = (double)p * inv_freq[i], s, c;
                det_sincos(theta, &s, &c);
                m->rope_cos[p * 32 + i] = (float)c;
                m->rope_sin[p * 32 + i] = (float)s;
            }
        }
    }
    m->n_pos = 0;
    m->trace = getenv("DETLM_TRACE") != NULL;
    pool_start(&m->pool, n_threads);
    return m;
fail:
    free(ti);
    detlm_free(m);
    return NULL;
}

void detlm_free(detlm *m) {
    if (!m) return;
    if (m->pool.n) pool_stop(&m->pool);
    free(m->layers); free(m->kcache); free(m->vcache); free(m->rope_cos); free(m->rope_sin);
    if (m->map) munmap(m->map, m->map_len);
    free(m);
}

/* ------------------------------------------------------------------ kernels */

/* quantize n_tok rows of n floats into int8 blocks (llama.cpp q8_0 activation semantics) */
static void quantize_rows(const float *x, int n_tok, int n, float *xd, int8_t *xq) {
    int nb = n / QK;
    for (int t = 0; t < n_tok; t++) {
        for (int b = 0; b < nb; b++) {
            const float *src = x + (size_t)t * n + (size_t)b * QK;
            float amax = 0.0f;
            for (int j = 0; j < QK; j++) { float v = src[j]; if (v < 0.0f) v = -v; if (v > amax) amax = v; }
            float d = amax / 127.0f;
            float id = (d != 0.0f) ? 1.0f / d : 0.0f;
            uint16_t dh = f32_to_f16(d);
            xd[(size_t)t * nb + b] = f16_to_f32(dh);
            int8_t *dst = xq + ((size_t)t * nb + b) * QK;
            for (int j = 0; j < QK; j++) { float v = src[j] * id; dst[j] = (int8_t)det_roundf(v); }
        }
    }
}

struct mm_job { const uint8_t *W; int n_in, n_out, n_tok; const float *xd; const int8_t *xq; float *y; };

/* y[t][r] = sum over blocks b (ascending) of (float)(sum_j wq[j]*xq[j]) * (dw[b]*dx[t][b]) */
static void mm_rows(void *arg, int r0, int r1) {
    struct mm_job *J = (struct mm_job *)arg;
    const int nb = J->n_in / QK;
    float dw[MAXNB];
    for (int r = r0; r < r1; r++) {
        const uint8_t *wrow = J->W + (size_t)r * (size_t)nb * BLK;
        for (int b = 0; b < nb; b++) { uint16_t h; memcpy(&h, wrow + (size_t)b * BLK, 2); dw[b] = f16_to_f32(h); }
        for (int t = 0; t < J->n_tok; t++) {
            const int8_t *xq = J->xq + (size_t)t * (size_t)nb * QK;
            const float *xd = J->xd + (size_t)t * nb;
            float acc = 0.0f;
#ifdef DETLM_REORDER_SUM
            for (int b = nb - 1; b >= 0; b--) {
#else
            for (int b = 0; b < nb; b++) {
#endif
                const int8_t *wq = (const int8_t *)(wrow + (size_t)b * BLK + 2);
                const int8_t *xb = xq + (size_t)b * QK;
                int32_t sumi = 0;
                for (int j = 0; j < QK; j++) sumi += (int32_t)wq[j] * (int32_t)xb[j];
                float scale = dw[b] * xd[b];
                float term = (float)sumi * scale;
                acc = acc + term;
            }
            J->y[(size_t)t * J->n_out + r] = acc;
        }
    }
}

static void matmul(detlm *m, const uint8_t *W, int n_in, int n_out, int n_tok, const float *xd, const int8_t *xq, float *y) {
    struct mm_job J = { W, n_in, n_out, n_tok, xd, xq, y };
    pool_run(&m->pool, mm_rows, &J, n_out);
}

/* RMSNorm as ggml does it: double sum of squares, mean, 1/sqrtf(mean+eps), x*scale, then * weight */
static void rms_norm(const float *x, const float *w, int n, float *y, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    double mean = ss / (double)n;
    float scale = 1.0f / sqrtf((float)(mean + (double)eps));
    for (int i = 0; i < n; i++) { float v = x[i] * scale; y[i] = v * w[i]; }
}

static uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct attn_job { detlm *m; int layer, n_tok, pos0; const float *q; float *out; float *scores; };

/* one (token, head) pair per item; heads of one token share the kv head via GQA */
static void attn_items(void *arg, int i0, int i1) {
    struct attn_job *A = (struct attn_job *)arg; detlm *m = A->m;
    const int hd = m->head_dim, nh = m->n_head, gq = m->n_head / m->n_head_kv;
    const float *kc = m->kcache + (size_t)A->layer * (size_t)m->n_ctx * (size_t)m->kv_dim;
    const float *vc = m->vcache + (size_t)A->layer * (size_t)m->n_ctx * (size_t)m->kv_dim;
    const float kq_scale = 1.0f / sqrtf((float)hd);
    for (int it = i0; it < i1; it++) {
        int t = it / nh, h = it % nh, kvh = h / gq;
        int pos = A->pos0 + t;                       /* attend to positions 0..pos */
        const float *q = A->q + (size_t)t * m->n_embd + (size_t)h * hd;
        float *s = A->scores + (size_t)it * (size_t)m->n_ctx;
        float mx = -3.4028234663852886e+38f;
        for (int j = 0; j <= pos; j++) {
            const float *k = kc + (size_t)j * m->kv_dim + (size_t)kvh * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; d++) dot = dot + q[d] * k[d];
            float sc = dot * kq_scale;
            s[j] = sc;
            if (sc > mx) mx = sc;
        }
        float sum = 0.0f;
        for (int j = 0; j <= pos; j++) { float e = det_expf(s[j] - mx); s[j] = e; sum = sum + e; }
        float inv = 1.0f / sum;
        float *o = A->out + (size_t)t * m->n_embd + (size_t)h * hd;
        for (int d = 0; d < hd; d++) o[d] = 0.0f;
        for (int j = 0; j <= pos; j++) {
            float p = s[j] * inv;
            const float *v = vc + (size_t)j * m->kv_dim + (size_t)kvh * hd;
            for (int d = 0; d < hd; d++) o[d] = o[d] + p * v[d];
        }
    }
}

static int eval_chunk(detlm *m, const int32_t *tokens, int n, int n_logits, float *logits) {
    const int E = m->n_embd, F = m->n_ff, KV = m->kv_dim, hd = m->head_dim;
    const int nbE = E / QK, nbF = F / QK;
    float *x    = (float *)malloc((size_t)n * E * sizeof(float));
    float *xn   = (float *)malloc((size_t)n * E * sizeof(float));
    float *q    = (float *)malloc((size_t)n * E * sizeof(float));
    float *k    = (float *)malloc((size_t)n * KV * sizeof(float));
    float *v    = (float *)malloc((size_t)n * KV * sizeof(float));
    float *att  = (float *)malloc((size_t)n * E * sizeof(float));
    float *g    = (float *)malloc((size_t)n * F * sizeof(float));
    float *u    = (float *)malloc((size_t)n * F * sizeof(float));
    float *xd   = (float *)malloc((size_t)n * nbF * sizeof(float));
    int8_t *xq  = (int8_t *)malloc((size_t)n * F);
    float *sc   = (float *)malloc((size_t)n * m->n_head * (size_t)m->n_ctx * sizeof(float));
    if (!x || !xn || !q || !k || !v || !att || !g || !u || !xd || !xq || !sc) { fprintf(stderr, "detlm: out of memory\n"); goto oom; }

    /* embeddings: dequantize q8_0 rows */
    for (int t = 0; t < n; t++) {
        const uint8_t *row = m->tok_embd + (size_t)tokens[t] * (size_t)nbE * BLK;
        for (int b = 0; b < nbE; b++) {
            uint16_t h; memcpy(&h, row + (size_t)b * BLK, 2); float d = f16_to_f32(h);
            const int8_t *qs = (const int8_t *)(row + (size_t)b * BLK + 2);
            for (int j = 0; j < QK; j++) x[(size_t)t * E + (size_t)b * QK + j] = d * (float)qs[j];
        }
    }
    if (m->trace) fprintf(stderr, "trace embd      %016llx\n", (unsigned long long)fnv1a(x, (size_t)n * E * 4, 1469598103934665603ull));

    for (int l = 0; l < m->n_layer; l++) {
        const struct layer *L = &m->layers[l];
        float *kc = m->kcache + (size_t)l * (size_t)m->n_ctx * KV;
        float *vc = m->vcache + (size_t)l * (size_t)m->n_ctx * KV;
        for (int t = 0; t < n; t++) rms_norm(x + (size_t)t * E, L->attn_norm, E, xn + (size_t)t * E, m->eps);
        quantize_rows(xn, n, E, xd, xq);
        matmul(m, L->wq, E, E, n, xd, xq, q);
        matmul(m, L->wk, E, KV, n, xd, xq, k);
        matmul(m, L->wv, E, KV, n, xd, xq, v);
        for (int t = 0; t < n; t++) {
            int pos = m->n_pos + t;
            const float *cs = m->rope_cos + (size_t)pos * 32, *sn = m->rope_sin + (size_t)pos * 32;
            float *qt = q + (size_t)t * E, *kt = k + (size_t)t * KV, *vt = v + (size_t)t * KV;
            for (int i = 0; i < E; i++) qt[i] = qt[i] + L->bq[i];
            for (int i = 0; i < KV; i++) { kt[i] = kt[i] + L->bk[i]; vt[i] = vt[i] + L->bv[i]; }
            /* RoPE, NEOX style: pair (i, i+32) within each head */
            for (int h = 0; h < m->n_head; h++) {
                float *p = qt + (size_t)h * hd;
                for (int i = 0; i < 32; i++) {
                    float x0 = p[i], x1 = p[i + 32];
                    p[i]      = x0 * cs[i] - x1 * sn[i];
                    p[i + 32] = x0 * sn[i] + x1 * cs[i];
                }
            }
            for (int h = 0; h < m->n_head_kv; h++) {
                float *p = kt + (size_t)h * hd;
                for (int i = 0; i < 32; i++) {
                    float x0 = p[i], x1 = p[i + 32];
                    p[i]      = x0 * cs[i] - x1 * sn[i];
                    p[i + 32] = x0 * sn[i] + x1 * cs[i];
                }
            }
            memcpy(kc + (size_t)pos * KV, kt, (size_t)KV * sizeof(float));
            memcpy(vc + (size_t)pos * KV, vt, (size_t)KV * sizeof(float));
        }
        {
            struct attn_job A = { m, l, n, m->n_pos, q, att, sc };
            pool_run(&m->pool, attn_items, &A, n * m->n_head);
        }
        quantize_rows(att, n, E, xd, xq);
        matmul(m, L->wo, E, E, n, xd, xq, xn);
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] = x[i] + xn[i];
        /* FFN */
        for (int t = 0; t < n; t++) rms_norm(x + (size_t)t * E, L->ffn_norm, E, xn + (size_t)t * E, m->eps);
        quantize_rows(xn, n, E, xd, xq);
        matmul(m, L->wg, E, F, n, xd, xq, g);
        matmul(m, L->wu, E, F, n, xd, xq, u);
        for (size_t i = 0; i < (size_t)n * F; i++) {
            float a = g[i];
            float s = a / (1.0f + det_expf(-a));     /* SiLU as ggml: x / (1 + exp(-x)) */
            g[i] = s * u[i];
        }
        quantize_rows(g, n, F, xd, xq);
        matmul(m, L->wd, F, E, n, xd, xq, xn);
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] = x[i] + xn[i];
        if (m->trace) fprintf(stderr, "trace layer %2d  %016llx\n", l, (unsigned long long)fnv1a(x, (size_t)n * E * 4, 1469598103934665603ull));
    }
    m->n_pos += n;
    if (n_logits > 0) {
        int t0 = n - n_logits;
        for (int t = t0; t < n; t++) rms_norm(x + (size_t)t * E, m->output_norm, E, xn + (size_t)(t - t0) * E, m->eps);
        if (m->trace) fprintf(stderr, "trace outnorm   %016llx\n", (unsigned long long)fnv1a(xn, (size_t)n_logits * E * 4, 1469598103934665603ull));
        quantize_rows(xn, n_logits, E, xd, xq);
        matmul(m, m->output, E, m->n_vocab, n_logits, xd, xq, logits);
    }
    free(x); free(xn); free(q); free(k); free(v); free(att); free(g); free(u); free(xd); free(xq); free(sc);
    return 0;
oom:
    free(x); free(xn); free(q); free(k); free(v); free(att); free(g); free(u); free(xd); free(xq); free(sc);
    return -1;
}

int detlm_eval(detlm *m, const int32_t *tokens, int n, int n_logits, float *logits) {
    if (n < 0 || n_logits < 0 || n_logits > n || (n_logits && !logits)) return -1;
    if (m->n_pos + n > m->n_ctx) { fprintf(stderr, "detlm: context full (%d + %d > %d)\n", m->n_pos, n, m->n_ctx); return -1; }
    for (int i = 0; i < n; i++) if (tokens[i] < 0 || tokens[i] >= m->n_vocab) { fprintf(stderr, "detlm: bad token id %d\n", tokens[i]); return -1; }
    int done = 0;
    while (done < n) {
        int c = n - done; if (c > CHUNK) c = CHUNK;
        /* logits wanted for tokens with index >= n - n_logits */
        int first_logit = n - n_logits;
        int nl = 0;
        if (done + c > first_logit) nl = done + c - (first_logit > done ? first_logit : done);
        float *dst = nl ? logits + (size_t)((done + c - nl) - first_logit) * (size_t)m->n_vocab : NULL;
        if (eval_chunk(m, tokens + done, c, nl, dst) != 0) return -1;
        done += c;
    }
    return 0;
}
