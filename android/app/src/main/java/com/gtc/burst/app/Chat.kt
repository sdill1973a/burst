package com.gtc.burst.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** The conversation with one peer, both ways, kept across restarts (last 300 lines). The chat screen draws it, the
 *  notification shows its tail. The debug log stays separate. */
object Chat {
    class Line(val mine: Boolean, val text: String, val ts: Long, var status: String)  // status: "" | sending | delivered | not delivered

    private val lines = HashMap<String, ArrayList<Line>>()
    var onChange: ((String) -> Unit)? = null                 // the visible chat screen redraws on this

    private fun prefs(ctx: Context) = ctx.getSharedPreferences("burst_chat", Context.MODE_PRIVATE)

    @Synchronized fun get(ctx: Context, peer: String): List<Line> = load(ctx, peer).toList()

    @Synchronized private fun load(ctx: Context, peer: String) = lines.getOrPut(peer) {
        ArrayList<Line>().also { out ->
            val a = JSONArray(prefs(ctx).getString(peer, "[]"))
            for (i in 0 until a.length()) a.getJSONObject(i).let {
                out.add(Line(it.getBoolean("m"), it.getString("t"), it.getLong("ts"), it.optString("s")))
            }
        }
    }

    @Synchronized fun add(ctx: Context, peer: String, mine: Boolean, text: String, status: String = ""): Line {
        val l = Line(mine, text, System.currentTimeMillis(), status)
        load(ctx, peer).apply { add(l); while (size > 300) removeAt(0) }
        save(ctx, peer); onChange?.invoke(peer); return l
    }

    @Synchronized fun mark(ctx: Context, peer: String, l: Line, status: String) { l.status = status; save(ctx, peer); onChange?.invoke(peer) }

    private fun save(ctx: Context, peer: String) {
        val a = JSONArray()
        lines[peer].orEmpty().forEach { a.put(JSONObject().put("m", it.mine).put("t", it.text).put("ts", it.ts).put("s", it.status)) }
        prefs(ctx).edit().putString(peer, a.toString()).apply()
    }
}
