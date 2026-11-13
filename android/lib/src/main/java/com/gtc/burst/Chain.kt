package com.gtc.burst

import kotlin.random.Random

/**
 * Long messages: the same chain wire as the reference encoder —
 *   one link = [0x12][msg id][index<<4 | total-1] + a checked body (0x03…/0x08…/0x00…), up to 16 links,
 * split at sentence then word boundaries; every link keeps its own whitespace, so joining the decoded links gives the
 * text back exactly. A message that fits one packet goes out as a bare burst, as before.
 * (No missing-link request (0x13) on the phone yet: a lost link shows as an incomplete message.)
 */
object Chain {
    const val LIMIT = 200            // bytes per packet, same as the reference encoder
    private const val GUESS = 380    // chars per link to try first (~2.3 chars/byte measured), checked by real crunch
    private const val ONE_MAX = 2400 // longest text worth trying as one packet (boilerplate measured 11.6 chars/byte)
    const val PART = 4000            // a paste longer than this goes as several messages, each its own chain (≤16 links)
    const val PARTS: Byte = 0x0B     // [0x0B][paste id][part i, from 1][of n], before each part of a long paste
    const val CARRY: Byte = 0x0A     // a chain link coded 0x08 but primed on carry(tail, previous link)

    /** As the reference receiver computes it: our shared tail plus the previous link's text, last TAIL_N kept. */
    fun carry(tail: List<String>, prev: String) = (tail.takeLast(Codec.TAIL_N) + prev).takeLast(Codec.TAIL_N)

    /** Cut a long paste into messages of ≤ PART chars at sentence (else word) boundaries; joined they are the text. */
    fun parts(text: String, size: Int = PART): List<String> {
        if (text.length <= size) return listOf(text)
        val out = ArrayList<String>(); var cur = ""
        for (s in sentences(text)) for (w0 in if (s.length > size) words(s) else listOf(s))
            // review #12 (2026-11-04): one unbroken "word" longer than size came back whole, so sendText's halve-and-retry
            // got the same piece forever. A word past size is cut by characters; every piece is now <= size.
            for (w in if (w0.length > size) w0.chunked(size) else listOf(w0)) {
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
        if (Wire2.SEND && use08) return splitW2(c, text, tail)   // build constant (on since 11-12): the path below is the old wire
        // A whole long paste must never go to the model in one piece: past its context the native eval returns nothing
        // and a whole-text attempt failed with an empty logits array. One packet never holds more than this.
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
                if (prev != null && b[0] == 0x08.toByte()) b[0] = CARRY  // 0x0A: decode me after the link before
                bodies.add(b); prev = piece; return
            }
            val w = words(piece)
            if (w.size > 1) { fit(w.subList(0, w.size / 2).joinToString("")); fit(w.subList(w.size / 2, w.size).joinToString("")); return }
            // one run with no spaces (a table line, a path): cut it by characters, never mid surrogate pair
            require(piece.length > 1) { "one character will not fit a packet" }
            var cut = piece.length / 2; if (Character.isHighSurrogate(piece[cut - 1])) cut++
            fit(piece.substring(0, cut)); fit(piece.substring(cut))
        }
        rough.forEach(::fit)
        require(bodies.size <= 16) { "needs ${bodies.size} packets; the cap is 16" }
        val mid = Random.nextInt(256)
        return bodies.mapIndexed { i, b -> byteArrayOf(0x12, mid.toByte(), ((i shl 4) or (bodies.size - 1)).toByte()) + b }
    }

    /** Wire 2 sending: one 0x0E frame, or a 0x14 chain (0x0E, then 0x0F links each primed on the one before). */
    fun splitW2(c: Codec, text: String, tail: List<String> = emptyList()): List<ByteArray> {
        val one = if (text.length <= ONE_MAX) runCatching { c.crunchW2(text, tail) }.getOrNull() else null
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
        var prev: String? = null
        fun fit(piece: String) {
            val t = prev?.let { carry(tail, it) } ?: tail
            val b = runCatching { c.crunchW2(piece, t, prev != null) }.getOrNull()
            if (b != null && b.size + (if (bodies.isEmpty()) 6 else 3) <= LIMIT) { bodies.add(b); prev = piece; return }
            val w = words(piece)
            if (w.size > 1) { fit(w.subList(0, w.size / 2).joinToString("")); fit(w.subList(w.size / 2, w.size).joinToString("")); return }
            require(piece.length > 1) { "one character will not fit a packet" }
            var cut = piece.length / 2; if (Character.isHighSurrogate(piece[cut - 1])) cut++
            fit(piece.substring(0, cut)); fit(piece.substring(cut))
        }
        rough.forEach(::fit)
        require(bodies.size <= 16) { "needs ${bodies.size} packets; the cap is 16" }
        val mid = Random.nextInt(256); val whole = Wire2.check3(text.toByteArray())
        return bodies.mapIndexed { i, b -> Wire2.link(mid, i, bodies.size, b, if (i == 0) whole else null) }
    }

    sealed class Step {
        object Waiting : Step()
        class Done(val whole: Whole) : Step()
        class Refused(val why: String) : Step()
        class Ask(val frames: List<ByteArray>) : Step()     // 0x15 frames to send: links that failed their check
    }

    /** Receiver for 0x14 chains. Links are held raw and decoded once all are in, then the join must pass link 0's check. */
    class Assembler2(private val decode: (ByteArray, List<List<String>>) -> Codec.Decoded) {
        private class C(val total: Int, var t: Long) { val raw = HashMap<Int, ByteArray>(); var whole: ByteArray? = null; var last = t; var asks = 0; val asked = HashSet<Int>() }
        /** frame = a 0x13 to send to [from]; null = gave up on this chain, shown as incomplete. */
        class Nack(val from: Long, val frame: ByteArray?, val got: List<Int>)
        private val held = HashMap<String, C>()
        private val STALE_MS = 15 * 60_000L
        private val NACK_MS = 45_000L
        val idle: Boolean @Synchronized get() = held.isEmpty()

        @Synchronized fun add(from: Long, b: ByteArray, tails: List<List<String>>, now: Long = System.currentTimeMillis()): Step {
            val l = Wire2.parseLink(b) ?: return Step.Refused("malformed chain link")
            val key = "$from:${l.mid}"
            var c = held[key]
            if (c != null && (c.total != l.total || now - c.t > STALE_MS)) { held.remove(key); c = null }
            if (c == null) { c = C(l.total, now); held[key] = c }
            c.raw[l.idx] = b; c.last = now
            if (l.whole != null) c.whole = l.whole
            if (c.raw.size < c.total) return Step.Waiting
            val texts = arrayOfNulls<String>(c.total); val bad = ArrayList<Int>(); val cascade = HashSet<Int>(); var saw08 = false
            for (i in 0 until c.total) {
                val f = Wire2.parseLink(c.raw[i]!!)!!.inner; val prev = texts.getOrNull(i - 1)
                val d = when {
                    f[0] != Wire2.CODEC && f[0] != Wire2.CARRY && f[0] != 0.toByte() -> null
                    f[0] == Wire2.CARRY && prev == null -> { cascade.add(i); null }   // fine once the link before it is resent
                    f[0] == Wire2.CARRY -> decode(f, tails.map { carry(it, prev!!) }.distinct() + listOf(listOf(prev!!)))
                    else -> decode(f, tails)
                }
                if (d != null && d.verified) { texts[i] = d.text; if (d.codec == 8) saw08 = true } else bad.add(i)
            }
            if (bad.isNotEmpty()) {
                val new = bad.filter { it !in cascade && c.asked.add(it) }
                return if (new.isNotEmpty()) Step.Ask(new.map { Wire2.askFrame(c.raw[it]!!) }) else Step.Waiting
            }
            held.remove(key)
            val w = c.whole ?: return Step.Refused("no whole-message check")
            val text = Wire2.join(texts.map { it!! }, w) ?: return Step.Refused("whole-message check failed: these links do not belong to one message, not shown")
            return Step.Done(Whole(text, c.total, saw08, true))
        }

        /** Chains waiting on a lost link: a 0x13 to send (3 times, each later), or a give-up. */
        @Synchronized fun due(now: Long = System.currentTimeMillis()): List<Nack> {
            val out = ArrayList<Nack>()
            for ((key, c) in held.entries.toList()) {
                val from = key.substringBefore(':').toLong(); val mid = key.substringAfter(':').toInt()
                val missing = (0 until c.total).filter { it !in c.raw }
                val ripe = missing.isNotEmpty() && now - c.last >= NACK_MS * (c.asks + 1)
                if (ripe && c.asks < 3) {
                    c.asks++; c.last = now
                    out.add(Nack(from, Wire2.missingFrame(mid, missing.sumOf { 1 shl it }), c.raw.keys.sorted()))
                } else if (ripe || now - c.t > STALE_MS) { held.remove(key); out.add(Nack(from, null, c.raw.keys.sorted())) }
            }
            return out
        }

        @Synchronized fun progress(from: Long, b: ByteArray): String {
            val c = held["$from:${b[1].toInt() and 0xFF}"] ?: return ""
            return "${c.raw.size}/${c.total}"
        }
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
        private val born = HashMap<String, Long>()                   // when each held chain's first link arrived
        private val STALE_MS = 10 * 60 * 1000L                       // longer than any resend round; then it is a new message
        private fun mark(key: String, parts: Array<String?>, idx: Int, d: Codec.Decoded) {
            if (d.verified && d.codec == 8) saw08[key] = true
            if (!d.verified) bad[key] = true
            // review #6 (2026-11-04): an unverified link's text is never shown, only that it failed and why.
            parts[idx] = if (d.verified) d.text else "⟦unverified: ${d.reason ?: "check failed"}⟧"
        }
        fun add(from: Long, b: ByteArray, tails: List<List<String>> = emptyList()): Whole? {
            val mid = b[1].toInt() and 0xFF; val idx = (b[2].toInt() and 0xFF) shr 4; val total = (b[2].toInt() and 0x0F) + 1
            val key = "$from:$mid"
            // 10-30 review: a stale or different-length chain under the same 1-byte mid must never be spliced into this one
            // (an old verified link joined to a new message reads as "verified"; a longer total indexed past the array).
            val now = System.currentTimeMillis()
            val old = held[key]
            if (old != null && (old.size != total || now - (born[key] ?: 0L) > STALE_MS)) {
                held.remove(key); raw.remove(key); saw08.remove(key); bad.remove(key)
            }
            if (idx >= total) return null
            if (held[key] == null) born[key] = now
            val parts = held.getOrPut(key) { arrayOfNulls(total) }
            if (b.size > 3 && b[3] == CARRY) { raw.getOrPut(key) { HashMap() }[idx] = b.copyOfRange(3, b.size); parts[idx] = DEFER }
            else mark(key, parts, idx, c.uncrunch(b.copyOfRange(3, b.size), tails))
            if (parts.any { it == null }) return null
            for (i in parts.indices) if (parts[i] == DEFER) {                // in order: each needs the one before it
                val prev = parts.getOrNull(i - 1)
                val body = raw[key]!![i]!!
                val d = if (prev == null || prev.startsWith("⟦unverified")) Codec.Decoded("", 0, false, 8, "the link it was primed on is refused")
                        // review #11 (2026-11-04): the base-carried history [prev] went LAST, after up to 9 others, and
                        // uncrunch08 tries only MAX_TAILS_08 (6) — so after a base-prime resend every carried link was
                        // refused. Keep it inside the cap: the newest MAX-1 carries, then [prev].
                        else c.uncrunch(byteArrayOf(0x08) + body.copyOfRange(1, body.size),
                                        tails.map { carry(it, prev) }.distinct().take(Codec.MAX_TAILS_08 - 1) + listOf(listOf(prev)))
                mark(key, parts, i, d)
            }
            raw.remove(key)
            held.remove(key); born.remove(key)
            return Whole(parts.joinToString(""), total, saw08.remove(key) == true, bad.remove(key) != true)
        }
        fun progress(from: Long, b: ByteArray): String {
            val p = held["$from:${b[1].toInt() and 0xFF}"] ?: return ""
            return "${p.count { it != null }}/${p.size}"
        }
    }
}
