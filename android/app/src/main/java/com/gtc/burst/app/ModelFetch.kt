package com.gtc.burst.app

import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * First run: fetch the codec's model — a generic small base model, the same file on every phone (nothing personal in it,
 * nothing about you leaves the phone). Qwen2.5-0.5B base Q8_0 (mradermacher) from Hugging Face, sha256-checked against the pin the
 * box uses (open_codec.MODELS[2]); a file that does not match is deleted, never loaded. Resumes a partial download.
 */
object ModelFetch {
    const val URL_ = "https://huggingface.co/mradermacher/Qwen2.5-0.5B-GGUF/resolve/main/Qwen2.5-0.5B.Q8_0.gguf"
    const val SHA = "d5163baa48d1797b1d274f271aa1fd152e3f07248011f7ea386f3d0667dce535"
    const val SIZE = 531_068_256L

    /** Blocks; progress(percent). True when [dest] holds the pinned file. A dropped connection resumes where it
     *  stopped, up to 30 tries (first real run on a phone, 10-24: the link dropped at 46 MB). */
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
        var have = if (part.exists()) part.length() else 0L
        val c = URL(URL_).openConnection() as HttpURLConnection
        c.instanceFollowRedirects = true; c.connectTimeout = 20_000; c.readTimeout = 60_000
        if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
        if (c.responseCode == 200) have = 0L else if (c.responseCode != 206) return false
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
        val md = MessageDigest.getInstance("SHA-256")
        part.inputStream().use { val buf = ByteArray(1 shl 20); while (true) { val n = it.read(buf); if (n < 0) break; md.update(buf, 0, n) } }
        val hex = md.digest().joinToString("") { "%02x".format(it) }
        if (hex != SHA) { part.delete(); return false }
        return part.renameTo(dest)
    }
}
