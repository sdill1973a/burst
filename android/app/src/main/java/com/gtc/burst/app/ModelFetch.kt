package com.gtc.burst.app

import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * First run: fetch the codec's model — a generic small base model, the same file on every phone (nothing personal in it,
 * nothing about you leaves the phone). Qwen2.5-0.5B base Q8_0 (mradermacher) from Hugging Face, sha256-checked against the pin the
 * reference encoder uses; a file that does not match is deleted, never loaded. Resumes a partial download.
 */
object ModelFetch {
    // Pinned to a repository commit, not resolve/main, so the URL cannot move under the sha256 pin below.
    const val URL_ = "https://huggingface.co/mradermacher/Qwen2.5-0.5B-GGUF/resolve/c20fdc01c83cad0e38968fb98bc5476042d1c8db/Qwen2.5-0.5B.Q8_0.gguf"
    const val SHA = "d5163baa48d1797b1d274f271aa1fd152e3f07248011f7ea386f3d0667dce535"
    const val SIZE = 531_068_256L

    /** Blocks; progress(percent). True when [dest] holds the pinned file. A dropped connection resumes where it
     *  stopped, up to 30 tries (mobile links drop mid-download). */
    fun ensure(dest: File, progress: (Int) -> Unit): Boolean {
        var err: Exception? = null
        repeat(30) { n ->
            try { return once(dest, progress) } catch (e: java.io.IOException) { err = e; Thread.sleep(minOf(30_000L, 2_000L * (n + 1))) }
        }
        throw err ?: java.io.IOException("download failed")
    }

    private fun once(dest: File, progress: (Int) -> Unit): Boolean {
        if (dest.exists() && dest.length() == SIZE) return true
        val part = File(dest.path + ".part")
        // A complete (or overlong) .part must never be re-requested: a Range past the end is a 416 forever.
        if (part.exists() && part.length() >= SIZE) {
            if (part.length() == SIZE && finish(part, dest)) return true
            part.delete()
        }
        var have = if (part.exists()) part.length() else 0L
        val c = URL(URL_).openConnection() as HttpURLConnection
        c.instanceFollowRedirects = true; c.connectTimeout = 20_000; c.readTimeout = 60_000
        if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
        when (c.responseCode) {
            200 -> have = 0L
            206 -> {}
            416 -> { part.delete(); throw java.io.IOException("range not satisfiable; restarting the download") }
            else -> return false
        }
        c.inputStream.use { inp ->
            java.io.FileOutputStream(part, have > 0).use { out ->
                val buf = ByteArray(1 shl 16); var last = -1
                while (true) {
                    val n = inp.read(buf); if (n < 0) break
                    out.write(buf, 0, n); have += n
                    val pct = (have * 100 / SIZE).toInt(); if (pct != last) { last = pct; progress(pct) }
                }
            }
        }
        return finish(part, dest)
    }

    /** Hash the whole .part: the pinned file becomes [dest]; anything else is deleted, never loaded. */
    private fun finish(part: File, dest: File): Boolean {
        val md = MessageDigest.getInstance("SHA-256")
        part.inputStream().use { val buf = ByteArray(1 shl 20); while (true) { val n = it.read(buf); if (n < 0) break; md.update(buf, 0, n) } }
        val hex = md.digest().joinToString("") { "%02x".format(it) }
        if (hex != SHA) { part.delete(); return false }
        return part.renameTo(dest)
    }
}
