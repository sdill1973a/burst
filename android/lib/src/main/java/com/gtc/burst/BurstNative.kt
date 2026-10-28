package com.gtc.burst

/** The model, nothing else: load a GGUF, tokenize like llama-cpp-python (no BOS, no special parsing), next-token logits. */
object BurstNative {
    init { System.loadLibrary("ai-chat") }

    external fun init(nativeLibDir: String)
    external fun load(path: String, nThreads: Int): Boolean
    external fun nVocab(): Int
    external fun tokenize(utf8: ByteArray): IntArray
    external fun detokenize(tokens: IntArray): ByteArray
    external fun reset()
    /** Feed tokens after the cache; logits rows (nVocab each) for every fed index >= fromIdx. */
    external fun eval(tokens: IntArray, fromIdx: Int): FloatArray

    // --- det: frame 0x08 (detlm bit-identical engine + detrc exact range coder); see Det.kt for load/idle policy ---
    external fun detLoad(path: String, nThreads: Int): Boolean
    external fun detFree()
    external fun detLoaded(): Boolean
    external fun detVersion(): String
    external fun detLastReason(): String
    external fun detPeakRssKib(): Long
    /** check = sha256(text)[:2]; returns the whole 0x08 frame, or null (detLastReason says why). */
    external fun detEncode(check: ByteArray, primeIds: IntArray, msgIds: IntArray): ByteArray?
    /** Message ids decoded under this prime, or null (refused: detLastReason). The check is verified by the caller. */
    external fun detDecode(primeIds: IntArray, frame: ByteArray): IntArray?
}
