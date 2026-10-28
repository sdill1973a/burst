// burst_jni — the smallest bridge the open codec needs: load a GGUF, tokenize/detokenize exactly as the box's
// llama-cpp-python does (no BOS, no special parsing), and return next-token logits for chosen positions.
// The coder itself (MRC) lives in Kotlin; this file only runs the model.
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include "llama.h"
#include "ggml-backend.h"

#define TAG "burst"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static int g_pos = 0;          // tokens already in the KV cache

static const llama_vocab *vocab() { return llama_model_get_vocab(g_model); }

extern "C" JNIEXPORT void JNICALL
Java_com_gtc_burst_BurstNative_init(JNIEnv *env, jobject, jstring libDir) {
    const char *p = env->GetStringUTFChars(libDir, nullptr);
    ggml_backend_load_all_from_path(p);
    env->ReleaseStringUTFChars(libDir, p);
    llama_backend_init();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gtc_burst_BurstNative_load(JNIEnv *env, jobject, jstring path, jint nThreads) {
    const char *p = env->GetStringUTFChars(path, nullptr);
    llama_model_params mp = llama_model_default_params();
    g_model = llama_model_load_from_file(p, mp);
    env->ReleaseStringUTFChars(path, p);
    if (!g_model) { LOGE("model load failed"); return JNI_FALSE; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = nThreads; cp.n_threads_batch = nThreads;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) { LOGE("context init failed"); return JNI_FALSE; }
    g_pos = 0;
    return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_gtc_burst_BurstNative_nVocab(JNIEnv *, jobject) { return llama_vocab_n_tokens(vocab()); }

extern "C" JNIEXPORT jintArray JNICALL
Java_com_gtc_burst_BurstNative_tokenize(JNIEnv *env, jobject, jbyteArray utf8) {
    jsize n = env->GetArrayLength(utf8);
    std::string s(n, '\0');
    env->GetByteArrayRegion(utf8, 0, n, reinterpret_cast<jbyte *>(&s[0]));
    std::vector<llama_token> t(n + 8);
    int k = llama_tokenize(vocab(), s.data(), (int) s.size(), t.data(), (int) t.size(), false, false);
    if (k < 0) { t.resize(-k); k = llama_tokenize(vocab(), s.data(), (int) s.size(), t.data(), (int) t.size(), false, false); }
    jintArray out = env->NewIntArray(k);
    env->SetIntArrayRegion(out, 0, k, reinterpret_cast<const jint *>(t.data()));
    return out;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_gtc_burst_BurstNative_detokenize(JNIEnv *env, jobject, jintArray toks) {
    jsize n = env->GetArrayLength(toks);
    std::vector<jint> t(n);
    env->GetIntArrayRegion(toks, 0, n, t.data());
    std::string out; char buf[256];
    for (jint id : t) {
        int k = llama_token_to_piece(vocab(), id, buf, sizeof buf, 0, false);
        if (k > 0) out.append(buf, k);
    }
    jbyteArray r = env->NewByteArray((jsize) out.size());
    env->SetByteArrayRegion(r, 0, (jsize) out.size(), reinterpret_cast<const jbyte *>(out.data()));
    return r;
}

extern "C" JNIEXPORT void JNICALL
Java_com_gtc_burst_BurstNative_reset(JNIEnv *, jobject) {
    llama_memory_clear(llama_get_memory(g_ctx), true);
    g_pos = 0;
}

// Feed tokens after what is already cached; return logits (row-major, n_vocab each) for every fed position >= fromIdx
// (index into this call's tokens). Empty array on failure.
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_gtc_burst_BurstNative_eval(JNIEnv *env, jobject, jintArray toks, jint fromIdx) {
    jsize n = env->GetArrayLength(toks);
    std::vector<jint> t(n);
    env->GetIntArrayRegion(toks, 0, n, t.data());
    const int V = llama_vocab_n_tokens(vocab());
    std::vector<float> rows;
    const int B = 512;
    llama_batch batch = llama_batch_init(B, 0, 1);
    for (int s = 0; s < n; s += B) {
        int m = std::min(B, n - s);
        batch.n_tokens = m;
        for (int j = 0; j < m; j++) {
            batch.token[j] = t[s + j]; batch.pos[j] = g_pos + j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
            batch.logits[j] = (s + j) >= fromIdx;
        }
        if (llama_decode(g_ctx, batch) != 0) { llama_batch_free(batch); LOGE("decode failed"); return env->NewFloatArray(0); }
        for (int j = 0; j < m; j++) {
            if ((s + j) < fromIdx) continue;
            const float *lg = llama_get_logits_ith(g_ctx, j);
            rows.insert(rows.end(), lg, lg + V);
        }
        g_pos += m;
    }
    llama_batch_free(batch);
    jfloatArray r = env->NewFloatArray((jsize) rows.size());
    env->SetFloatArrayRegion(r, 0, (jsize) rows.size(), rows.data());
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// det — frame 0x08: detlm (bit-identical logits, parses the same GGUF itself) + detrc (exact range coder).
// One resident engine, loaded lazily by Kotlin (Det.kt) and released when idle; a mutex serialises every call.
// The tokenizer stays llama.cpp's above (same vocab as the box).
// ---------------------------------------------------------------------------------------------------------------
#include <mutex>
#include "det/detlm.h"
#include "det/det08.h"

static detlm *g_det = nullptr;
static std::mutex g_det_mu;
static std::string g_det_reason;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gtc_burst_BurstNative_detLoad(JNIEnv *env, jobject, jstring path, jint nThreads) {
    std::lock_guard<std::mutex> lk(g_det_mu);
    if (g_det) return JNI_TRUE;
    const char *p = env->GetStringUTFChars(path, nullptr);
    g_det = detlm_load(p, 2048, nThreads > 0 ? nThreads : 1);
    env->ReleaseStringUTFChars(path, p);
    if (!g_det) { LOGE("detlm load failed"); return JNI_FALSE; }
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_gtc_burst_BurstNative_detFree(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lk(g_det_mu);
    if (g_det) { detlm_free(g_det); g_det = nullptr; }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gtc_burst_BurstNative_detLoaded(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lk(g_det_mu);
    return g_det ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gtc_burst_BurstNative_detVersion(JNIEnv *env, jobject) { return env->NewStringUTF(detlm_version()); }

extern "C" JNIEXPORT jstring JNICALL
Java_com_gtc_burst_BurstNative_detLastReason(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lk(g_det_mu);
    return env->NewStringUTF(g_det_reason.c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_gtc_burst_BurstNative_detPeakRssKib(JNIEnv *, jobject) { return (jlong) detlm_peak_rss_kib(); }

// check[2] + prime ids + message ids -> the whole 0x08 frame, or null (reason via detLastReason)
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_gtc_burst_BurstNative_detEncode(JNIEnv *env, jobject, jbyteArray check, jintArray prime, jintArray msg) {
    std::lock_guard<std::mutex> lk(g_det_mu);
    if (!g_det) { g_det_reason = "detlm not loaded"; return nullptr; }
    if (env->GetArrayLength(check) != 2) { g_det_reason = "check must be 2 bytes"; return nullptr; }
    uint8_t chk[2]; env->GetByteArrayRegion(check, 0, 2, reinterpret_cast<jbyte *>(chk));
    jsize np = env->GetArrayLength(prime), nm = env->GetArrayLength(msg);
    std::vector<int32_t> p(np), t(nm);
    if (np) env->GetIntArrayRegion(prime, 0, np, p.data());
    if (nm) env->GetIntArrayRegion(msg, 0, nm, t.data());
    std::vector<uint8_t> out(DET08_FRAME_CAP(nm));
    const char *why = nullptr;
    int n = det08_encode(g_det, chk, p.data(), np, t.data(), nm, out.data(), (int) out.size(), &why);
    if (n < 0) { g_det_reason = why ? why : "encode failed"; LOGE("det08 encode: %s", g_det_reason.c_str()); return nullptr; }
    g_det_reason.clear();
    jbyteArray r = env->NewByteArray(n);
    env->SetByteArrayRegion(r, 0, n, reinterpret_cast<const jbyte *>(out.data()));
    return r;
}

// prime ids + frame -> message ids under THAT prime, or null (refused / error; reason via detLastReason).
// The 2-byte check is verified by the caller after detokenizing.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_gtc_burst_BurstNative_detDecode(JNIEnv *env, jobject, jintArray prime, jbyteArray frame) {
    std::lock_guard<std::mutex> lk(g_det_mu);
    if (!g_det) { g_det_reason = "detlm not loaded"; return nullptr; }
    jsize np = env->GetArrayLength(prime), fl = env->GetArrayLength(frame);
    std::vector<int32_t> p(np); std::vector<uint8_t> f(fl);
    if (np) env->GetIntArrayRegion(prime, 0, np, p.data());
    if (fl) env->GetByteArrayRegion(frame, 0, fl, reinterpret_cast<jbyte *>(f.data()));
    std::vector<int32_t> ids(DET08_MAX_TOKENS);
    const char *why = nullptr;
    int n = det08_decode(g_det, p.data(), np, f.data(), fl, ids.data(), (int) ids.size(), &why);
    if (n < 0) { g_det_reason = why ? why : "decode failed"; return nullptr; }
    g_det_reason.clear();
    jintArray r = env->NewIntArray(n);
    env->SetIntArrayRegion(r, 0, n, reinterpret_cast<const jint *>(ids.data()));
    return r;
}
