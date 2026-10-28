package com.gtc.burst

import kotlin.math.max

/**
 * Mismatch-resistant coding (Hu & Tang, arXiv 2601.17684), ported line for line from codec/mrc.py.
 * The two tables numpy would generate (the random code permutation and the Huffman bucket codes) are NOT regenerated
 * here: they are exported once from the box (golden/export.py) and shipped, so both ends hold the same bits.
 */
class Mrc(
    val codes: IntArray, val L: Int,
    private val hi: DoubleArray, private val lo: DoubleArray,
    private val bcode: Map<Int, String>, val c: Double,
) {
    private val K = hi.size
    private val bdec: Map<String, Int> = bcode.entries.associate { it.value to it.key }

    /** numpy: searchsorted(-hi, -p, side="right") - 1  ==  (count of hi[j] >= p) - 1. */
    fun bucket(p: Double): Int {
        if (p <= 0.0) return K - 1
        var n = 0
        for (h in hi) if (h >= p) n++ else break
        return n - 1
    }

    private fun lcp(a: Int, b: Int): Int {
        val x = a xor b
        return if (x == 0) L else L - (32 - Integer.numberOfLeadingZeros(x))
    }

    private fun bits(code: Int): String = Integer.toBinaryString(code).padStart(L, '0')

    fun encodeToken(probs: DoubleArray, x: Int): String {
        val p = probs[x]; val k = bucket(p)
        val c2 = c * c; val loK = lo[k] / c2; val hiK = c2 * hi[k]
        var m = 0
        val cx = codes[x]
        for (i in probs.indices) {
            if (i == x) continue
            val q = probs[i]
            if (q > loK && q < hiK) m = max(m, lcp(cx, codes[i]))
        }
        val code = bits(cx)
        val sb = StringBuilder(bcode.getValue(k)).append(code, 0, minOf(m + 1, L))
        if (m + 1 < L) sb.append(if (code[m + 1] == '0') '1' else '0')
        return sb.toString()
    }

    class Read(val token: Int, val pos: Int, val sus: String?)

    fun decodeToken(probs: DoubleArray, stream: String, start: Int): Read {
        var pos = start
        var cur = ""
        while (cur !in bdec) {
            if (pos >= stream.length) {                       // ran off the end: likeliest token, flagged
                var best = 0
                for (i in probs.indices) if (probs[i] > probs[best]) best = i
                return Read(best, stream.length, "end")
            }
            cur += stream[pos]; pos++
        }
        val k = bdec.getValue(cur)
        var sus: String? = null
        var uh = window(probs, lo[k] / c, c * hi[k])
        if (uh.isEmpty()) {
            val c3 = c * c * c
            uh = window(probs, lo[k] / c3, c3 * hi[k])
            if (uh.isEmpty()) uh = probs.indices.sortedByDescending { probs[it] }.take(64).toIntArray()
            sus = "empty"
        }
        val zs = if (pos + L <= stream.length) stream.substring(pos, pos + L) else stream.substring(minOf(pos, stream.length)).padEnd(L, '0')
        val z = Integer.parseInt(zs, 2)
        var n = -1
        val lcps = IntArray(uh.size) { lcp(z, codes[uh[it]]).also { v -> if (v > n) n = v } }
        // numpy: top = Uh[lcp == n]; best = top[argmax(probs[top])] if len(top) > 1 else Uh[argmax(lcp)]
        var best = -1; var ties = 0
        for (j in uh.indices) if (lcps[j] == n) {
            ties++
            if (best < 0 || probs[uh[j]] > probs[best]) best = uh[j]
        }
        if (sus == null && ties > 1) sus = "tie"
        pos += if (n >= L) L else n + 1
        return Read(best, pos, sus)
    }

    private fun window(probs: DoubleArray, a: Double, b: Double): IntArray {
        var cnt = 0
        for (q in probs) if (q > a && q < b) cnt++
        val out = IntArray(cnt); var j = 0
        for (i in probs.indices) if (probs[i] > a && probs[i] < b) out[j++] = i
        return out
    }

    companion object {
        fun pack(bits: String): ByteArray {
            val padded = bits + "0".repeat((8 - bits.length % 8) % 8)
            return ByteArray(padded.length / 8) { Integer.parseInt(padded.substring(it * 8, it * 8 + 8), 2).toByte() }
        }
        fun unpack(data: ByteArray, from: Int): String {
            val sb = StringBuilder()
            for (i in from until data.size) sb.append(Integer.toBinaryString(data[i].toInt() and 0xFF).padStart(8, '0'))
            return sb.toString()
        }
    }
}
