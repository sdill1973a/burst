package com.gtc.burst

import android.util.Log

/** Logcat for message content (plaintext, wire bytes, admin replies): written only when [verbose] is on, which the app
 *  sets for debug or test-hook builds. Counts, sizes and statuses go through Log directly. */
object BurstLog {
    @Volatile var verbose = false
    fun content(tag: String, msg: String) { if (verbose) Log.i(tag, msg) }
}
