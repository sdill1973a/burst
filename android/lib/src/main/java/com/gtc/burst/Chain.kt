package com.gtc.burst

import kotlin.random.Random

/**
 * Long messages: the same chain wire as codec/burst.py —
 *   one link = [0x12][msg id][index<<4 | total-1] + a checked body (0x03…/0x08…/0x00…), up to 16 links,
 * split at sentence then word boundaries; every link keeps its own whitespace, so joining the decoded links gives the
 * text back exactly. A message that fits one packet goes out as a bare burst, as before.
 * (No missing-link request (0x13) on the phone yet: a lost link shows as an incomplete message.)
 */
object Chain {
    const val LIMIT = 200            // bytes per packet, same as burst.py
    private const val GUESS = 380    // chars per link to try first (~2.3 chars/byte measured), checked by real crunch
    private const val ONE_MAX = 2400 // longest text worth trying as one packet (boilerplate ran 11.6 chars/byte, 10-24)
    const val PART = 4000            // a paste longer than this goes as several messages, each its own chain (≤16 links)
    const val PARTS: Byte = 0x0B     // [0x0B][paste id][part i, from 1][of n], before each part of a long paste — burst_rx.PARTS
    const val CARRY: Byte = 0x0A     // a chain link coded 0x08 but primed on carry(tail, previous link) — burst_rx.CARRY

    /** Exactly burst_rx.carry: our shared tail plus the previous link's text, last TAIL_N kept. */
    fun carry(tail: List<String>, prev: String) = (tail.takeLast(Codec.TAIL_N) + prev).takeLast(Codec.TAIL_N)

    /** Cut a long paste into messages of ≤ PART chars at sentence (else word) boundaries; joined they are the text. */
    fun parts(text: String, size: Int = PART): List<String> {
        if (text.length <= size) return listOf(text)
        val out = ArrayList<String>(); var cur = ""
        for (s in sentences(text)) for (w in if (s.length > size) words(s) else listOf(s)) {
            if (cur.length + w.length > size && cur.isNotEmpty()) { out.add(cur); cur = "" }
            cur += w
        }
        if (cur.isNotEmpty()) out.add(cur)
        return out
    }

    private fun sentences(t: String) = Regex(".*?(?:[.!?]\\s+|$)", RegexOption.DOT_MATCHES_ALL).findAll(t)
        .map { it.value }.filter { it.isNotEmpty() }.toList().ifEmpty { listOf(t) }

    private fun words(t: String) = Regex("\\S*\\s*").findAll(t).map { it.value }.filter { it.isNotEmpty() }.toList()

    /** Returns the packets to send, in order. use08 = this peer reads the deterministic codec (or the dev toggle). */
    fun split(c: Codec, text: String, tail: List<String> = emptyList(), use08: Boolean = Codec.force08): List<ByteArray> {
        // A whole long paste must never go to the model in one piece: past its context the native eval returns nothing
        // and the old code died on "length=0; index=0" (his paste, 10-24 14:56). One packet never holds more than this.
        val one = if (text.length <= ONE_MAX) runCatching { c.crunch(text, tail = tail, use08 = use08) }.getOrNull() else null
        if (one != null && one.size <= LIMIT) return listOf(one)
        val rough = ArrayList<String>(); var cur = ""
        for (s in sentences(text)) {
            if (cur.length + s.length <= GUESS) { cur += s; continue }
            if (cur.isNotEmpty()) { rough.add(cur); cur = "" }
            if (s.length <= GUESS) { cur = s; continue }
            for (w in words(s)) { if (cur.length + w.length > GUESS && cur.isNotEmpty()) { rough.add(cur); cur = "" }; cur += w }
        }
        if (cur.isNotEmpty()) rough.add(cur)
        val bodies = ArrayList<ByteArray>()
        var prev: String? = null                     // the text of the link before: every later 0x08 link is primed on it
        fun fit(piece: String) {                     // crunch; if it is still too big, halve at a word boundary and retry
            val t = prev?.let { carry(tail, it) } ?: tail
            val b = runCatching { c.crunch(piece, tail = t, use08 = use08) }.getOrNull()
            if (b != null && b.size + 3 <= LIMIT) {
                if (prev != null && b[0] == 0x08.toByte()) b[0] = CARRY  // 0x0A: decode me after the link before (10-24)
                bodies.add(b); prev = piece; return
            }
            val w = words(piece)
            if (w.size > 1) { fit(w.subList(0, w.size / 2).joinToString("")); fit(w.subList(w.size / 2, w.size).joinToString("")); return }
            // one run with no spaces (a table line, a path): cut it by characters, never mid surrogate pair (10-24)
            require(piece.length > 1) { "one character will not fit a packet" }
            var cut = piece.length / 2; if (Character.isHighSurrogate(piece[cut - 1])) cut++
            fit(piece.substring(0, cut)); fit(piece.substring(cut))
        }
        rough.forEach(::fit)
        require(bodies.size <= 16) { "needs ${bodies.size} packets; the cap is 16" }
        val mid = Random.nextInt(256)
        return bodies.mapIndexed { i, b -> byteArrayOf(0x12, mid.toByte(), ((i shl 4) or (bodies.size - 1)).toByte()) + b }
    }

    /** A reassembled message: verified08 = at least one link was a 0x08 frame that passed its check (the peer reads 0x08). */
    class Whole(val text: String, val total: Int, val verified08: Boolean, val allVerified: Boolean)

    /** Receiver side: feed every 0x12 link; returns the whole text once all links of that message are in. */
    class Assembler(private val c: Codec) {
        private val held = HashMap<String, Array<String?>>()
        private val saw08 = HashMap<String, Boolean>()
        private val bad = HashMap<String, Boolean>()
        private val raw = HashMap<String, HashMap<Int, ByteArray>>()   // carried links waiting for the one before
        private val DEFER = "\u0000defer"
        private fun mark(key: String, parts: Array<String?>, idx: Int, d: Codec.Decoded) {
            if (d.verified && d.codec == 8) saw08[key] = true
            if (!d.verified) bad[key] = true
            parts[idx] = if (d.verified) d.text else "⟦unverified: ${d.text.ifEmpty { d.reason ?: "refused" }}⟧"
        }
        fun add(from: Long, b: ByteArray, tails: List<List<String>> = emptyList()): Whole? {
            val mid = b[1].toInt() and 0xFF; val idx = (b[2].toInt() and 0xFF) shr 4; val total = (b[2].toInt() and 0x0F) + 1
            val key = "$from:$mid"
            val parts = held.getOrPut(key) { arrayOfNulls(total) }
            if (b.size > 3 && b[3] == CARRY) { raw.getOrPut(key) { HashMap() }[idx] = b.copyOfRange(3, b.size); parts[idx] = DEFER }
            else mark(key, parts, idx, c.uncrunch(b.copyOfRange(3, b.size), tails))
            if (parts.any { it == null }) return null
            for (i in parts.indices) if (parts[i] == DEFER) {                // in order: each needs the one before it
                val prev = parts.getOrNull(i - 1)
                val body = raw[key]!![i]!!
                val d = if (prev == null || prev.startsWith("⟦unverified")) Codec.Decoded("", 0, false, 8, "the link it was primed on is refused")
                        else c.uncrunch(byteArrayOf(0x08) + body.copyOfRange(1, body.size), tails.map { carry(it, prev) } + listOf(listOf(prev)))
                mark(key, parts, i, d)
            }
            raw.remove(key)
            held.remove(key)
            return Whole(parts.joinToString(""), total, saw08.remove(key) == true, bad.remove(key) != true)
        }
        fun progress(from: Long, b: ByteArray): String {
            val p = held["$from:${b[1].toInt() and 0xFF}"] ?: return ""
            return "${p.count { it != null }}/${p.size}"
        }
    }
}
