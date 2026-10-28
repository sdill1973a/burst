package com.gtc.burst

import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.roundToInt

/**
 * The open codec, same wire as codec/open_codec.py:
 *   [0x03][model id][c×10][tail id][varint n_tokens][MRC bits]
 * Shared history (10-23, his perpetual exchange): the prime can carry the last TAIL_N confirmed messages of the
 * conversation; header byte 3 = tail id (sha256 of the joined tail, low byte |1; 0 = base prime), as open_codec.py.
 *
 * Frame 0x08 (10-24, codec/): the deterministic codec — detlm computes the SAME logits bit for
 * bit on the phone and the box, so a plain range coder replaces the mismatch margin (~2x content per byte):
 *   [0x08][2-byte check = sha256(text)[:2]][varint n_tokens][range-coder bytes]
 * No model id, no c, no tail id: the receiver tries candidate histories newest-first (cap MAX_TAILS_08) then the base
 * prime, and the check picks; nothing that fails the check is ever shown (refused, not ⟦unverified⟧).
 */
class Codec(codesBin: ByteArray, tablesJson: String) {
    private val t = JSONObject(tablesJson)
    val prime: String = t.getString("prime")
    val cDefault: Double = t.getDouble("c_default")
    private val L = t.getInt("L")
    private val hi = t.getJSONArray("hi").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
    private val lo = t.getJSONArray("lo").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
    private val bcode = t.getJSONObject("bcode").let { o -> o.keys().asSequence().associate { it.toInt() to o.getString(it) } }
    private val codes: IntArray = ByteBuffer.wrap(codesBin).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().let { b -> IntArray(b.remaining()).also { b.get(it) } }

    private fun mrc(c: Double) = Mrc(codes, L, hi, lo, bcode, c)
    private fun softmax(logits: FloatArray, off: Int, n: Int): DoubleArray {
        var mx = Double.NEGATIVE_INFINITY
        for (i in 0 until n) if (logits[off + i] > mx) mx = logits[off + i].toDouble()
        val e = DoubleArray(n); var s = 0.0
        for (i in 0 until n) { e[i] = exp(logits[off + i] - mx); s += e[i] }
        for (i in 0 until n) e[i] /= s
        return e
    }

    companion object {
        const val TAIL_N = 4
        const val MAX_TAILS_08 = 6          // candidate histories tried for a 0x08 frame, newest first (box: det_codec.MAX_TAILS)
        const val MAX_TOKENS_08 = 2048
        const val TAIL_BYTES = 1500          // open_codec.TAIL_BYTES
        /** Dev toggle: send 0x08 to everyone regardless of whether the peer has shown it reads it (MainActivity: prefs/intent). */
        @Volatile var force08 = false
        fun tailId(tail: List<String>): Int =
            if (tail.isEmpty()) 0 else (java.security.MessageDigest.getInstance("SHA-256").digest(tail.joinToString("\n").toByteArray())[0].toInt() and 0xFF) or 1
        fun sha2(bytes: ByteArray): ByteArray = java.security.MessageDigest.getInstance("SHA-256").digest(bytes).copyOf(2)
    }
    /** The learned tail, capped: oldest messages drop until it fits TAIL_BYTES — exactly open_codec.fit_tail (10-24). */
    private fun fitTail(tail: List<String>): List<String> {
        var t = tail; while (t.isNotEmpty() && t.sumOf { it.toByteArray().size } > TAIL_BYTES) t = t.drop(1); return t
    }
    private fun primeFor(tail: List<String>) = prime + fitTail(tail).joinToString("") { it.trimEnd('\n') + "\n\n" }

    fun crunchRaw(text: String, c: Double = cDefault, tail: List<String> = emptyList()): ByteArray {
        val pt = BurstNative.tokenize(primeFor(tail).toByteArray())
        val toks = BurstNative.tokenize(text.toByteArray())
        val V = BurstNative.nVocab()
        BurstNative.reset()
        val rows = BurstNative.eval(pt + toks, pt.size - 1)          // row i = next-token logits after pt + toks[:i]
        val m = mrc(c)
        val bits = StringBuilder()
        for (i in toks.indices) bits.append(m.encodeToken(softmax(rows, i * V, V), toks[i]))
        val head = java.io.ByteArrayOutputStream()
        head.write(byteArrayOf(0x03, 1, (c * 10).roundToInt().toByte(), tailId(tail).toByte()))
        var n = toks.size
        while (true) { val b = n and 0x7F; n = n ushr 7; head.write(b or (if (n != 0) 0x80 else 0)); if (n == 0) break }
        head.write(Mrc.pack(bits.toString()))
        return head.toByteArray()
    }

    /** codec = the wire byte that carried it (0x00 / 0x03 / 0x08); reason = why a 0x08 was refused (text is then ""). */
    class Decoded(val text: String, val sus: Int, val verified: Boolean = true, val codec: Int = 0, val reason: String? = null,
                  val tail: List<String>? = null)

    /** burst.py's wire: [codec][2-byte check = sha256(text)[:2]][body]. 0x03 = open codec, 0x08 = det codec, 0x00 = plain
     *  UTF-8 when that is smaller. The check lets the receiver REFUSE a wrong decode instead of showing it (box, 10-21). */
    private fun check(text: String) = sha2(text.toByteArray())

    /** The whole 0x08 frame (its header already carries the check), or null when the text cannot go as 0x08
     *  (does not survive tokenize/detokenize, over the token cap, engine unavailable) — the caller falls back to 0x03. */
    fun crunch08(text: String, tail: List<String> = emptyList()): ByteArray? {
        val tb = text.toByteArray()
        val toks = BurstNative.tokenize(tb)
        if (toks.size > MAX_TOKENS_08 || !BurstNative.detokenize(toks).contentEquals(tb)) return null
        val pt = BurstNative.tokenize(primeFor(tail.takeLast(TAIL_N)).toByteArray())
        return Det.use { BurstNative.detEncode(sha2(tb), pt, toks) }
    }

    /** use08: the peer has shown it reads 0x08 (History.reads08) or the dev toggle is on; else 0x03 as before. */
    fun crunch(text: String, c: Double = cDefault, tail: List<String> = emptyList(), use08: Boolean = force08): ByteArray {
        val chk = check(text)
        val plain = byteArrayOf(0) + chk + text.toByteArray()
        // A peer that reads 0x08 never gets the inexact 0x03: when 0x08 cannot take this text, it goes plain (exact).
        // 10-24: two parts of his paste fell back to 0x03 and arrived as UNVERIFIED garbage.
        if (use08) return crunch08(text, tail)?.takeIf { it.size < plain.size } ?: plain
        val packed = crunchRaw(text, c, tail).let { raw -> byteArrayOf(raw[0]) + chk + raw.copyOfRange(1, raw.size) }
        return if (packed.size < plain.size) packed else plain
    }

    fun uncrunch(b: ByteArray, tails: List<List<String>> = emptyList()): Decoded {
        val chk = b.copyOfRange(1, 3)
        return when (b[0].toInt()) {
            0x00 -> String(b.copyOfRange(3, b.size), Charsets.UTF_8).let { Decoded(it, 0, check(it).contentEquals(chk), 0) }
            0x03 -> uncrunchRaw(byteArrayOf(3) + b.copyOfRange(3, b.size), tails).let { Decoded(it.text, it.sus, check(it.text).contentEquals(chk), 3, tail = it.tail) }
            0x08 -> uncrunch08(b, tails)
            else -> throw IllegalArgumentException("not a burst (codec byte ${b[0]})")
        }
    }

    /** 0x08: try the candidate histories newest first (cap MAX_TAILS_08), then the base prime; the check picks. A frame
     *  that passes under no candidate is REFUSED: text "", verified=false, reason set — never shown as content. */
    fun uncrunch08(b: ByteArray, tails: List<List<String>> = emptyList()): Decoded {
        require(b.isNotEmpty() && b[0].toInt() == 0x08) { "not a 0x08 frame" }
        if (b.size < 4) return Decoded("", 0, false, 8, "frame too short")
        val chk = b.copyOfRange(1, 3)
        val cands = tails.take(MAX_TAILS_08).map { it.takeLast(TAIL_N) }.distinct().toMutableList()
        if (cands.none { it.isEmpty() }) cands.add(emptyList())
        var why = "check failed under every candidate history (${cands.size} tried): wrong decode, not shown"
        for (tail in cands) {
            val pt = BurstNative.tokenize(primeFor(tail).toByteArray())
            val r = Det.use { BurstNative.detDecode(pt, b) to BurstNative.detLastReason() } ?: return Decoded("", 0, false, 8, "detlm unavailable")
            val ids = r.first
            if (ids == null) { why = "refused: ${r.second}"; if (why.contains("not a 0x08")) break else continue }   // a too-long prime must not stop the base-prime try (10-24)
            val bytes = BurstNative.detokenize(ids)
            if (sha2(bytes).contentEquals(chk)) return Decoded(String(bytes, Charsets.UTF_8), 0, true, 8, tail = tail)
        }
        return Decoded("", 0, false, 8, why)
    }

    /** [tails] = candidate histories; the one whose id matches the header is used. No match → base prime, flagged. */
    fun uncrunchRaw(b: ByteArray, tails: List<List<String>> = emptyList()): Decoded {
        require(b[0].toInt() == 0x03) { "not an open-codec burst" }
        val c = (b[2].toInt() and 0xFF) / 10.0
        val tid = b[3].toInt() and 0xFF
        val tail = if (tid == 0) emptyList() else tails.firstOrNull { tailId(it) == tid } ?: emptyList()
        var i = 4; var shift = 0; var n = 0
        while (true) { val x = b[i++].toInt() and 0xFF; n = n or ((x and 0x7F) shl shift); if (x and 0x80 == 0) break; shift += 7 }
        val stream = Mrc.unpack(b, i)
        val V = BurstNative.nVocab()
        val pt = BurstNative.tokenize(primeFor(tail).toByteArray())
        BurstNative.reset()
        var row = BurstNative.eval(pt, pt.size - 1)
        val m = mrc(c)
        val out = IntArray(n); var pos = 0; var sus = 0
        for (k in 0 until n) {
            val r = m.decodeToken(softmax(row, 0, V), stream, pos)
            out[k] = r.token; pos = r.pos; if (r.sus != null) sus++
            if (k + 1 < n) row = BurstNative.eval(intArrayOf(r.token), 0)
        }
        return Decoded(String(BurstNative.detokenize(out), Charsets.UTF_8), sus, codec = 3, tail = tail)
    }
}
