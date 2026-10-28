package com.gtc.burst

import android.content.Context
import org.json.JSONArray

/**
 * The conversation's shared history, per peer: both ends keep the same list and the prime carries its last TAIL_N
 * entries, so every exchange grows the shared context and makes the next burst smaller.
 *
 * What joins: a message I RECEIVED whose check passed, and a message I SENT that the far end ACKed. The two ends can
 * still disagree (a lost ACK, two messages crossing), so the receiver tries every nearby window — the header's tail id
 * picks the one the sender used, and the 2-byte check still rules. A miss decodes as ⟦UNVERIFIED⟧, never as fact.
 * The reference receiver uses the same candidates in the same order.
 *
 * reads08: this peer has sent me a 0x08 frame that passed its check, so it runs the deterministic codec and I may send
 * 0x08 to it (never send 0x08 to a peer that has not shown it reads it; 0x03 until then).
 */
class History private constructor(ctx: Context, peer: Long) {
    private val prefs = ctx.getSharedPreferences("burst_history", Context.MODE_PRIVATE)
    private val key = "%08x".format(peer)
    private val items: MutableList<String> =
        JSONArray(prefs.getString(key, "[]")).let { a -> MutableList(a.length()) { a.getString(it) } }

    @Synchronized fun tail(): List<String> = items.takeLast(Codec.TAIL_N)

    @Synchronized fun add(text: String) {
        items.add(text)
        while (items.size > 32) items.removeAt(0)
        prefs.edit().putString(key, JSONArray(items).toString()).apply()
    }

    fun reads08(): Boolean = prefs.getBoolean("$key.reads08", false)
    fun markReads08() { if (!reads08()) prefs.edit().putBoolean("$key.reads08", true).apply() }

    /** Windows the far end may have used: the newest TAIL_N, each one step older, and the newest with one entry missing. */
    @Synchronized fun candidates(): List<List<String>> = candidates(items)

    companion object {
        private val cache = java.util.concurrent.ConcurrentHashMap<Long, History>()
        /** The one History for [peer]: every caller shares it, so @Synchronized guards the same list. */
        fun of(ctx: Context, peer: Long): History = cache.getOrPut(peer) { History(ctx.applicationContext, peer) }

        fun candidates(items: List<String>): List<List<String>> {
            val out = LinkedHashSet<List<String>>()
            for (lag in 0..3) {
                val end = items.size - lag; if (end <= 0) break
                out.add(items.subList(maxOf(0, end - Codec.TAIL_N), end).toList())
            }
            val last = items.takeLast(Codec.TAIL_N + 1)
            for (skip in last.indices) out.add(last.filterIndexed { i, _ -> i != skip }.takeLast(Codec.TAIL_N))
            return out.toList()
        }
    }
}
