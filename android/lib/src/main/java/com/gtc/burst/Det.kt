package com.gtc.burst

import android.util.Log
import kotlin.concurrent.thread

/**
 * The deterministic engine behind frame 0x08 (detlm), resident only while it is being used: loaded lazily on the first
 * 0x08 encode/decode (~1 s from the same model.gguf llama.cpp already mapped — the weight pages are shared by the OS,
 * the extra is detlm's KV cache + scratch, ~520 MiB peak measured on the box), released after IDLE_MS without a call.
 */
object Det {
    @Volatile var modelPath: String? = null
    @Volatile var threads: Int = 2
    const val IDLE_MS = 10 * 60 * 1000L
    @Volatile private var lastUsed = 0L
    private var reaper: Thread? = null

    /** True when the engine is loaded (loading it now if needed). False = no model path yet or load failed. */
    @Synchronized fun ensure(): Boolean {
        val p = modelPath ?: return false
        if (!BurstNative.detLoaded()) {
            val t0 = System.currentTimeMillis()
            if (!BurstNative.detLoad(p, threads)) { Log.e("burst", "detlm load failed: $p"); return false }
            Log.i("burst", "detlm ${BurstNative.detVersion()} loaded in ${System.currentTimeMillis() - t0} ms, $threads threads")
            if (reaper == null) reaper = thread(isDaemon = true, name = "det-reaper") {
                while (true) {
                    Thread.sleep(30_000)
                    synchronized(this) {
                        if (BurstNative.detLoaded() && System.currentTimeMillis() - lastUsed > IDLE_MS) {
                            BurstNative.detFree(); Log.i("burst", "detlm released (idle)")
                        }
                    }
                }
            }
        }
        lastUsed = System.currentTimeMillis()
        return true
    }

    fun touch() { lastUsed = System.currentTimeMillis() }

    /** Run a native det call with the engine guaranteed resident for its whole duration: the block holds the same lock
     *  the reaper takes, so an idle release can never land between ensure() and the call (Q's read, 10-24).
     *  Returns null when the engine is unavailable. */
    @Synchronized fun <T> use(block: () -> T): T? {
        if (!ensure()) return null
        try { return block() } finally { lastUsed = System.currentTimeMillis() }
    }
}
