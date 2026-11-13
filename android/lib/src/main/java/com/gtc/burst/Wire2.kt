package com.gtc.burst

import java.security.MessageDigest

/**
 * Wire 2 bytes, no engine and no Android here, byte for byte the box's det_codec.py (shared vectors: golden/wire2_vectors.py):
 *   single  [0x0E][tail id][3-byte check][varint n][range-coder bytes]     carried link: 0x0F
 *   chain   [0x14][mid][idx<<4|total-1][3-byte whole-message check, link 0 only][inner frame]
 *   ASK     [0x15][sha256(the refused frame)[:4]]      missing links  [0x13][mid][16-bit mask, little end first]
 * Receiving is always on. Sending stays on the old wire until SEND is true.
 */
object Wire2 {
    /** Build constant: send wire 2. Default off: the old wire, exactly as before. */
    const val SEND = true

    const val CODEC: Byte = 0x0E
    const val CARRY: Byte = 0x0F
    const val CHAIN: Byte = 0x14
    const val MISSING: Byte = 0x13
    const val ASK: Byte = 0x15

    fun sha(b: ByteArray): ByteArray = MessageDigest.getInstance("SHA-256").digest(b)
    fun check3(text: ByteArray): ByteArray = sha(text).copyOf(3)
    fun askKey(frame: ByteArray): ByteArray = sha(frame).copyOf(4)
    fun askFrame(frame: ByteArray): ByteArray = byteArrayOf(ASK) + askKey(frame)
    fun isWire2(p: ByteArray) = p.isNotEmpty() && (p[0] == CODEC || p[0] == CHAIN)

    fun primeFor(prime: String, tail: List<String>): String {
        var t = tail
        while (t.isNotEmpty() && t.sumOf { it.toByteArray().size } > Codec.TAIL_BYTES) t = t.drop(1)
        return prime + t.joinToString("") { it.trimEnd('\n') + "\n\n" }
    }
    fun tailId(primeText: String): Int = sha(primeText.toByteArray())[0].toInt() and 0xFF

    /** A native 0x08 frame -> wire 2. Same varint and body; the old 2-byte check is the first 2 bytes of the 3-byte one. */
    fun fromDet08(det08: ByteArray, tid: Int, carried: Boolean, text: ByteArray): ByteArray =
        byteArrayOf(if (carried) CARRY else CODEC, tid.toByte()) + check3(text) + det08.copyOfRange(3, det08.size)
    fun toDet08(f: ByteArray): ByteArray = byteArrayOf(0x08) + f.copyOfRange(2, 4) + f.copyOfRange(5, f.size)

    class Link(val mid: Int, val idx: Int, val total: Int, val whole: ByteArray?, val inner: ByteArray)

    fun link(mid: Int, idx: Int, total: Int, inner: ByteArray, whole: ByteArray?): ByteArray {
        require(total in 1..16 && idx in 0 until total && (idx == 0) == (whole != null)) { "bad chain link" }
        return byteArrayOf(CHAIN, mid.toByte(), ((idx shl 4) or (total - 1)).toByte()) + (whole ?: ByteArray(0)) + inner
    }
    fun parseLink(b: ByteArray): Link? {
        if (b.size < 4 || b[0] != CHAIN) return null
        val h = b[2].toInt() and 0xFF; val idx = h shr 4; val total = (h and 0x0F) + 1; val at = if (idx == 0) 6 else 3
        if (idx >= total || b.size <= at) return null
        return Link(b[1].toInt() and 0xFF, idx, total, if (idx == 0) b.copyOfRange(3, 6) else null, b.copyOfRange(at, b.size))
    }
    fun missingFrame(mid: Int, mask: Int) = byteArrayOf(MISSING, mid.toByte(), (mask and 0xFF).toByte(), (mask shr 8).toByte())
    fun join(texts: List<String>, whole: ByteArray): String? =
        texts.joinToString("").takeIf { check3(it.toByteArray()).contentEquals(whole) }
    fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
    fun unhex(s: String) = ByteArray(s.length / 2) { s.substring(it * 2, it * 2 + 2).toInt(16).toByte() }
}
