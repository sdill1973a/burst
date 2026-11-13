package com.gtc.burst.app

import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.gtc.burst.Wire2
import com.gtc.burst.BurstLog
import com.gtc.burst.BurstNative
import com.gtc.burst.Chain
import com.gtc.burst.Codec
import com.gtc.burst.Det
import com.gtc.burst.History
import com.gtc.burst.MeshBle
import com.gtc.burst.Mqtt
import org.json.JSONArray
import java.io.File
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private lateinit var log: TextView
    private var codec: Codec? get() = Session.codec; set(v) { Session.codec = v }
    private var mesh: MeshBle? get() = Session.mesh; set(v) { Session.mesh = v }
    private var asm: Chain.Assembler? get() = Session.asm; set(v) { Session.asm = v }
    private val asked get() = Session.asked             // checks of refused frames I already asked about (once each)

    private fun sha256(b: ByteArray) = java.security.MessageDigest.getInstance("SHA-256").digest(b)
    private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
    /** The 2-byte check a frame carries (a chain link's sits inside its body), as the reference receiver reads it. */
    private class Early(val from: Long, val to: Long, val ch: Int, val port: Int, val payload: ByteArray)
    private val early get() = Session.early

    private fun chk(p: ByteArray) = if (p[0] == 0x12.toByte()) p.copyOfRange(4, 6) else p.copyOfRange(1, 3)

    /** One link at a time, each waiting for its routing ACK; every delivered frame is indexed by its check (whole
     *  message text) so the peer can ASK for it again. Returns true when every link got through. */
    private fun transmit(dest: Long, packets: List<ByteArray>, ch: Int, label: String, text: String,
                         onLink: (Int) -> Unit = {}, record: Boolean = true): Boolean {
        val m = mesh ?: run { say("Not sent: no radio — tap Connect"); return false }
        if (!m.ready) say("Radio not ready yet — waiting up to 60 s before sending")
        repeat(120) { if (!m.ready) Thread.sleep(500) }              // a send tapped mid-connect must wait, not drop
        if (!m.ready) { say("Not sent: the radio never became ready"); return false }
        for ((i, p) in packets.withIndex()) {
            onLink(i)
            var ok = false
            for (attempt in 1..3) {
                val id = m.send(p, dest, ch)
                BurstLog.content("burst", "TXWIRE " + Base64.encodeToString(p, Base64.NO_WRAP))
                val err = if (dest == MeshBle.BROADCAST) 0.also { Thread.sleep(3000) } else m.awaitAck(id, 30000)
                say("TX %d/%d id %08x -> %s: %s".format(i + 1, packets.size, id, label,
                    when (err) { null -> if (m.localOnly) "sent, not confirmed (only my radio answered)" else "no ACK"; 0 -> "delivered"; else -> "error $err" }))
                if (err == 0) { ok = true; break }
            }
            if (!ok) { say("Stopped: link ${i + 1} never got through"); return false }
        }
        if (dest != MeshBle.BROADCAST && record) {
            val prefs = getSharedPreferences("burst", MODE_PRIVATE); val key = "sent_%08x".format(dest)
            val idx = org.json.JSONObject(prefs.getString(key, "{}")!!)
            for (p in packets) idx.put(hex(if (Wire2.isWire2(p)) Wire2.askKey(p) else chk(p)), text)
            if (packets[0][0] == Wire2.CHAIN) rememberChain2(dest, packets)
            while (idx.length() > 64) idx.remove(idx.keys().next())
            prefs.edit().putString(key, idx.toString()).apply()
        }
        return true
    }

    private val RESEND_WINDOW_MS = 30 * 60 * 1000L

    /** ASK (0x06 + check) from a peer that refused one of my frames: resend that message on the BASE prime, once. */
    private fun answerAsk(from: Long, ch: Int, b: ByteArray) {
        val c = codec ?: return
        val prefs = getSharedPreferences("burst", MODE_PRIVATE); val key = "sent_%08x".format(from)
        val idx = org.json.JSONObject(prefs.getString(key, "{}")!!); val k = hex(b.copyOfRange(1, if (b[0] == Wire2.ASK) 5 else 3))
        val text = idx.optString(k, "")
        if (text.isEmpty()) { say("ASK from !%08x for $k: not one of mine".format(from)); return }
        // Review #14 (2026-11-04): these flags were booleans that never cleared, so once a 2-byte check had been
        // resent, a LATER message with the same check could never be resent again. Now each is a time; it lapses
        // after RESEND_WINDOW_MS, which still blocks the burst of duplicate ASKs one refused chain produces.
        val now = System.currentTimeMillis()
        fun recent(key: String) = runCatching { now - prefs.getLong(key, 0L) < RESEND_WINDOW_MS }.getOrDefault(false)
        if (recent("resent_$k")) { say("ASK for $k: already resent once"); return }
        // One resend per MESSAGE, not per link: a peer asks once for each refused link, and resending the whole part per
        // ASK multiplies airtime (13 asks = 13 full resends). The first ASK resends it; the rest are no-ops.
        val tk = "resent_text_" + hex(sha256(text.toByteArray()).copyOf(6))
        synchronized(Session) {
            if (recent(tk)) { prefs.edit().putLong("resent_$k", now).apply(); say("ASK for $k: its message is already being resent"); return }
            prefs.edit().putLong("resent_$k", now).putLong(tk, now).apply()
        }
        val use08 = Codec.force08 || History.of(applicationContext, from).reads08()
        val packets = Chain.split(c, text, emptyList(), use08)
        say("ASK from !%08x: resending on the base prime (%d packet(s), %d B)".format(from, packets.size, packets.sumOf { it.size }))
        transmit(from, packets, ch, "!%08x".format(from), text)
    }

    /** I refused a 0x08 frame: ask the sender once for it again, without our history. */
    private fun askFor(from: Long, ch: Int, p: ByteArray) {
        if (Wire2.isWire2(p)) {                                              // wire 2: the key is a hash of this exact frame
            val k = hex(Wire2.askKey(p)); if (!asked.add(k)) return
            thread { mesh?.let { m -> if (m.ready) { m.send(Wire2.askFrame(p), from, ch); say("ASK sent to !%08x for $k".format(from)) } } }
            return
        }
        val k = hex(chk(p)); if (!asked.add(k)) return
        thread { mesh?.let { m -> if (m.ready) { m.send(byteArrayOf(0x06) + chk(p), from, ch); say("ASK sent to !%08x for $k".format(from)) } } }
    }

    /** Keep a sent wire 2 chain's link frames, so a 0x13 can be answered with exactly the links that were lost. */
    private fun rememberChain2(dest: Long, packets: List<ByteArray>) {
        val prefs = getSharedPreferences("burst", MODE_PRIVATE); val key = "sent2_%08x".format(dest)
        val idx = org.json.JSONObject(prefs.getString(key, "{}")!!); val mid = (packets[0][1].toInt() and 0xFF).toString()
        idx.remove(mid)
        idx.put(mid, org.json.JSONObject().put("t", System.currentTimeMillis()).put("links", org.json.JSONArray(packets.map { Wire2.hex(it) })).put("miss", org.json.JSONObject()))
        while (idx.length() > 8) idx.remove(idx.keys().next())
        prefs.edit().putString(key, idx.toString()).apply()
    }

    /** 0x13 from a peer missing links of a wire 2 chain of mine: resend just those, each at most 3 times. */
    private fun answerMissing(from: Long, ch: Int, b: ByteArray) {
        val prefs = getSharedPreferences("burst", MODE_PRIVATE); val key = "sent2_%08x".format(from)
        val idx = org.json.JSONObject(prefs.getString(key, "{}")!!); val mid = (b[1].toInt() and 0xFF).toString()
        val mask = (b[2].toInt() and 0xFF) or ((b[3].toInt() and 0xFF) shl 8)
        val e = idx.optJSONObject(mid)
        if (e == null || System.currentTimeMillis() - e.getLong("t") > RESEND_WINDOW_MS) { say("0x13 from !%08x for chain $mid: not one of mine".format(from)); return }
        val links = e.getJSONArray("links"); val miss = e.getJSONObject("miss"); val out = ArrayList<ByteArray>()
        for (i in 0 until links.length()) if (mask shr i and 1 == 1 && miss.optInt("$i", 0) < 3) { miss.put("$i", miss.optInt("$i", 0) + 1); out.add(Wire2.unhex(links.getString(i))) }
        prefs.edit().putString(key, idx.toString()).apply()
        say("0x13 from !%08x: resending %d link(s) of chain $mid".format(from, out.size))
        if (out.isNotEmpty()) transmit(from, out, ch, "!%08x".format(from), "", record = false)
    }

    private val w2Chan = java.util.concurrent.ConcurrentHashMap<Long, Int>()

    /** While a wire 2 chain is held, ask for its lost links; stops when nothing is held. */
    private fun watchMissing() {
        if (!Session.w2Watch.compareAndSet(false, true)) return
        thread(isDaemon = true, name = "w2-missing") {
            try {
                while (true) {
                    Thread.sleep(5000)
                    val a = Session.asm2 ?: break
                    for (n in a.due()) {
                        val f = n.frame
                        if (f == null) say("RX !%08x: INCOMPLETE message, got links %s of a chain".format(n.from, n.got))
                        else mesh?.let { m -> if (m.ready) { m.send(f, n.from, w2Chan[n.from] ?: 0); say("Asked !%08x for missing links".format(n.from)) } }
                    }
                    if (a.idle) break
                }
            } finally { Session.w2Watch.set(false) }
        }
    }

    private fun onChain2(c: Codec, from: Long, to: Long, ch: Int, payload: ByteArray) {
        val app = applicationContext
        val a = Session.asm2 ?: Chain.Assembler2 { f, t -> c.decodeW2Inner(f, t) }.also { Session.asm2 = it }
        val hist = History.of(app, from)
        w2Chan[from] = ch
        val step = runCatching { a.add(from, payload, hist.candidates()) }.getOrElse { say("link undecodable: ${it.message}"); return }
        when (step) {
            is Chain.Step.Waiting -> { say("RX !%08x link %s".format(from, a.progress(from, payload))); if (to != MeshBle.BROADCAST) watchMissing() }
            is Chain.Step.Refused -> say("RX !%08x: chain REFUSED — %s".format(from, step.why))
            is Chain.Step.Ask -> { if (to != MeshBle.BROADCAST) thread { mesh?.let { m -> if (m.ready) for (f in step.frames) m.send(f, from, ch) } }; say("RX !%08x: %d link(s) refused, asked for a resend".format(from, step.frames.size)) }
            is Chain.Step.Done -> {
                if (step.whole.verified08) hist.markReads08()
                hist.add(step.whole.text)
                say("RX !%08x (%d packets, wire 2): %s".format(from, step.whole.total, step.whole.text), content = true)
                BurstService.notify(app, "!%08x".format(from), step.whole.text, ch)
            }
        }
    }

    private fun onSingle2(c: Codec, from: Long, to: Long, ch: Int, payload: ByteArray) {
        val app = applicationContext; val hist = History.of(app, from)
        val d = runCatching { c.uncrunch2(payload, hist.candidates()) }.getOrElse { say("wire 2 undecodable: ${it.message}"); return }
        if (d.verified) {
            hist.add(d.text); hist.markReads08(); BurstService.notify(app, "!%08x".format(from), d.text, ch)
            say("RX !%08x ch%d (%d B, wire 2): %s".format(from, ch, payload.size, d.text), content = true)
        } else { if (to != MeshBle.BROADCAST) askFor(from, ch, payload); say("RX !%08x: REFUSED wire 2 — ${d.reason}".format(from)) }
    }

    /** A line for the on-screen log. [content] = it carries message text: logcat gets it only in debug/test builds. */
    private fun say(s: String, content: Boolean = false) {
        if (content) BurstLog.content("burst", s) else Log.i("burst", s)
        synchronized(Session.log) { Session.log.append(s + "\n") }; Session.sink?.invoke(s)
    }

    private var adminReceiver: android.content.BroadcastReceiver? = null

    override fun onDestroy() {
        adminReceiver?.let { runCatching { unregisterReceiver(it) } }; adminReceiver = null
        if (Session.sink === sink) Session.sink = null; if (Chat.onChange === redraw) Chat.onChange = null; super.onDestroy()
    }
    override fun onResume() { super.onResume(); Session.visible = true; ui.post(tick) }
    override fun onPause() { Session.visible = false; ui.removeCallbacks(tick); super.onPause() }

    // ---- the chat screen ----
    private val ui = android.os.Handler(android.os.Looper.getMainLooper())
    private lateinit var chatBox: android.widget.LinearLayout
    private lateinit var chatScroll: android.widget.ScrollView
    private fun peerId() = findViewById<EditText>(R.id.to).text.toString().trim().let { if (it.startsWith("!")) it else "!$it" }
    private val redraw: (String) -> Unit = { p -> if (p == peerId()) runOnUiThread { drawChat() } }
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private var aiShow: () -> Unit = {}

    /** Bring your own AI (optional): pick an account, pick what to ask, about their last message or what I typed. */
    private fun aiAsk(input: EditText) {
        val accts = Ai.accounts(this)
        if (accts.isEmpty()) { aiAccounts(); return }
        val draft = input.text.toString().trim()
        val theirs = Chat.get(applicationContext, peerId()).lastOrNull { !it.mine }?.text
        val subject = if (draft.isNotEmpty()) draft else theirs ?: run { say("AI: nothing to ask about yet"); return }
        val asks = Ai.ASKS.keys.toList() + "Your own question…"
        android.app.AlertDialog.Builder(this).setTitle(if (draft.isNotEmpty()) "Ask AI about what you typed" else "Ask AI about their last message")
            .setItems(asks.toTypedArray()) { _, i ->
                fun go(instr: String) {
                    val acct = accts.first()
                    say("AI: asking ${acct.name} (${acct.model}) over the internet — not the radio")
                    thread {
                        val r = try { Ai.ask(acct, instr, subject) } catch (e: Exception) { "Error: ${e.message}" }
                        runOnUiThread {
                            android.app.AlertDialog.Builder(this).setTitle(acct.name).setMessage(r)
                                .setPositiveButton("Put in message box") { _, _ -> input.setText(r) }
                                .setNegativeButton("Close", null).show()
                        }
                    }
                }
                if (i < Ai.ASKS.size) go(Ai.ASKS.values.toList()[i]) else {
                    val q = EditText(this)
                    android.app.AlertDialog.Builder(this).setTitle("Your question").setView(q)
                        .setPositiveButton("Ask") { _, _ -> go(q.text.toString()) }.setNegativeButton("Cancel", null).show()
                }
            }.show()
    }

    /** Accounts: the first one is used. Add (from a preset), or remove. Keys are sealed by the Keystore (Ai.kt). */
    private fun aiAccounts() {
        val accts = Ai.accounts(this).toMutableList()
        val rows = accts.map { "${it.name} · ${it.model}" } + "+ Add an account"
        android.app.AlertDialog.Builder(this).setTitle("AI accounts (optional)")
            .setItems(rows.toTypedArray()) { _, i ->
                if (i < accts.size) android.app.AlertDialog.Builder(this).setTitle(accts[i].name)
                    .setPositiveButton("Use first") { _, _ -> val a = accts.removeAt(i); accts.add(0, a); Ai.save(this, accts) }
                    .setNegativeButton("Remove") { _, _ -> accts.removeAt(i); Ai.save(this, accts); aiShow() }.show()
                else {
                    val names = Ai.PRESETS.keys.toList()
                    android.app.AlertDialog.Builder(this).setTitle("Provider").setItems(names.toTypedArray()) { _, p ->
                        val (url0, model0) = Ai.PRESETS.getValue(names[p])
                        val box = android.widget.LinearLayout(this).apply { orientation = android.widget.LinearLayout.VERTICAL; setPadding(dp(16), 0, dp(16), 0) }
                        val url = EditText(this).apply { setText(url0); hint = "Endpoint (OpenAI-compatible)" }
                        val model = EditText(this).apply { setText(model0); hint = "Model" }
                        val key = EditText(this).apply { hint = "API key (blank for your own server)"
                            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD }
                        listOf(url, model, key).forEach { box.addView(it) }
                        android.app.AlertDialog.Builder(this).setTitle(names[p]).setView(box)
                            .setPositiveButton("Save") { _, _ ->
                                accts.add(Ai.Account(names[p], url.text.toString().trim(), model.text.toString().trim(), key.text.toString().trim()))
                                Ai.save(this, accts); aiShow(); say("AI: ${names[p]} saved (key sealed on this phone)")
                            }.setNegativeButton("Cancel", null).show()
                    }.show()
                }
            }.show()
    }

    private fun drawChat() {
        chatBox.removeAllViews()
        val fmt = java.text.SimpleDateFormat("h:mm a", java.util.Locale.US)
        for (l in Chat.get(applicationContext, peerId())) {
            val bubble = TextView(this).apply {
                text = l.text; textSize = 16f; setTextColor(if (l.mine) 0xFFFFFFFF.toInt() else 0xFFF2F4F7.toInt())
                setPadding(dp(14), dp(10), dp(14), dp(10)); setTextIsSelectable(true)
                background = android.graphics.drawable.GradientDrawable().apply {
                    cornerRadius = dp(18).toFloat(); setColor(if (l.mine) 0xFF2F6FEB.toInt() else 0xFF2A3038.toInt())
                }
            }
            val meta = TextView(this).apply {
                textSize = 11f; setTextColor(if (l.status.startsWith("NOT") || l.status == "not delivered") 0xFFE5534B.toInt() else 0xFF6B7480.toInt())
                text = fmt.format(java.util.Date(l.ts)) + when (l.status) { "" -> ""; "not delivered" -> " · NOT delivered"; else -> " · ${l.status}" }
                setPadding(dp(6), dp(2), dp(6), dp(10))
            }
            val g = if (l.mine) android.view.Gravity.END else android.view.Gravity.START
            for (v in listOf(bubble, meta)) chatBox.addView(v, android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                gravity = g; if (v === bubble) { if (l.mine) marginStart = dp(48) else marginEnd = dp(48) } })
        }
        if (Session.typing(peerId())) chatBox.addView(TextView(this).apply {
            text = "typing…"; textSize = 14f; setTextColor(0xFF9AA4B2.toInt()); setTypeface(typeface, android.graphics.Typeface.ITALIC)
            setPadding(dp(14), dp(6), dp(14), dp(10))
        })
        shownTyping = Session.typing(peerId())
        chatScroll.post { chatScroll.fullScroll(android.view.View.FOCUS_DOWN) }
    }
    private var shownTyping = false

    /** The radio as a dot: grey off, amber connecting, green listening. Polled once a second while the screen shows. */
    private val tick = object : Runnable { override fun run() {
        val m = mesh
        val (c, t) = when { m == null -> 0xFF6B7480.toInt() to "Off · tap"; !m.ready -> 0xFFE3A008.toInt() to "Connecting"; else -> 0xFF2EA043.toInt() to "Listening · tap to release" }
        (findViewById<android.view.View>(R.id.dot).background.mutate() as android.graphics.drawable.GradientDrawable).setColor(c)
        findViewById<TextView>(R.id.statusText).text = t
        if (Session.typing(peerId()) != shownTyping) drawChat()     // the dots go up or time out
        ui.postDelayed(this, 1000)
    } }
    private val sink: (String) -> Unit = { s -> runOnUiThread { log.append(s + "\n") } }

    /** Carry the radio's MQTT over the phone's network when the radio asks for it (MQTT on + proxy to client on) and a
     *  broker is set (prefs mqtt_host / mqtt_port / mqtt_user / mqtt_pass, set once per phone). Uplink and downlink both pass
     *  through untouched: the radio already encrypted them. */
    private fun proxy(prefs: android.content.SharedPreferences) {
        val host = prefs.getString("mqtt_host", null)?.takeIf { it.isNotBlank() } ?: return
        val m = mesh ?: return
        Session.mqtt?.close(); Session.mqtt = null
        m.onProxy = { t, d, r -> if (Session.mqtt?.publish(t, d, r) != true) say("MQTT up-link dropped (broker not connected): $t") }
        thread {
            repeat(240) { if (!m.ready) Thread.sleep(500) }
            if (!m.ready || mesh !== m) return@thread
            if (!m.mqttEnabled || !m.mqttProxy) { say("Radio MQTT proxy is off: nothing to carry"); return@thread }
            Session.mqtt = Mqtt(host, prefs.getInt("mqtt_port", 1883), prefs.getString("mqtt_user", "") ?: "",
                prefs.getString("mqtt_pass", "") ?: "", listOf("${m.mqttRoot}/2/e/#"), ::say) { t, d, r -> if (mesh === m) m.proxyDown(t, d, r) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        BurstLog.verbose = BuildConfig.DEBUG || BuildConfig.TEST_HOOKS
        setContentView(R.layout.activity_main)
        log = findViewById(R.id.log)
        synchronized(Session.log) { log.text = Session.log.toString() }   // a rebuilt screen shows what the process already heard
        Session.sink = sink
        chatBox = findViewById(R.id.chat); chatScroll = findViewById(R.id.chatScroll)
        Chat.onChange = redraw
        val settings = findViewById<android.view.View>(R.id.settings)
        findViewById<TextView>(R.id.gear).setOnClickListener { settings.visibility = if (settings.visibility == android.view.View.VISIBLE) android.view.View.GONE else android.view.View.VISIBLE }
        // The status is the listen switch: off → connect; on → let go of the radio so the Meshtastic app can use it
        findViewById<android.view.View>(R.id.status).setOnClickListener {
            if (mesh == null) findViewById<Button>(R.id.connect).performClick()
            else { BurstService.release(this); window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
        }
        val input = findViewById<EditText>(R.id.input)
        val send = findViewById<Button>(R.id.send)
        val radio = findViewById<EditText>(R.id.radio); val toF = findViewById<EditText>(R.id.to); val chF = findViewById<EditText>(R.id.chan)
        val prefs = getSharedPreferences("burst", MODE_PRIVATE)
        val need = arrayOf(android.Manifest.permission.BLUETOOTH_CONNECT, android.Manifest.permission.POST_NOTIFICATIONS)
            .filter { checkSelfPermission(it) != android.content.pm.PackageManager.PERMISSION_GRANTED }
        if (need.isNotEmpty()) requestPermissions(need.toTypedArray(), 1)
        // Back sends Burst to the background instead of closing it, so the radio stays held
        onBackPressedDispatcher.addCallback(this, object : androidx.activity.OnBackPressedCallback(true) {
            override fun handleOnBackPressed() { moveTaskToBack(true) }
        })
        // Intent extras are test hooks (am start --es radio NAME --es to !nodeid ...): honoured only in a TEST_HOOKS build,
        // so no other app on the phone can point the radio or the sends anywhere.
        val hooks = BuildConfig.TEST_HOOKS
        val xRadio = if (hooks) intent.getStringExtra("radio") else null
        val xTo = if (hooks) intent.getStringExtra("to") else null
        (xRadio ?: prefs.getString("radio", null))?.let { radio.setText(it) }
        (xTo ?: prefs.getString("to", null))?.let { toF.setText(it) }
        findViewById<TextView>(R.id.peer).text = MeshBle.name(peerId())
        drawChat()
        // Dev toggle: force frame 0x08 to every peer (else 0x08 only to a peer that has sent me a verified 0x08).
        // Persisted; also settable by the test hook `--ez force08 true|false`.
        val force08 = findViewById<CheckBox>(R.id.force08)
        if (hooks && intent.hasExtra("force08")) prefs.edit().putBoolean("force08", intent.getBooleanExtra("force08", false)).apply()
        Codec.force08 = prefs.getBoolean("force08", true)   // every Burst app reads 0x08, so it is the default
        force08.isChecked = Codec.force08                   // before the listener: the checkbox shows the real setting
        force08.setOnCheckedChangeListener { _, on -> Codec.force08 = on; prefs.edit().putBoolean("force08", on).apply(); say("force 0x08: $on") }
        findViewById<Button>(R.id.connect).setOnClickListener connect@{
            // No radio named yet: pick one from the paired devices, never the first device that happens to be bonded
            if (radio.text.isBlank()) {
                val names = MeshBle(applicationContext, ::say) { _, _, _, _, _ -> }.bondedNames()
                if (names.isEmpty()) { say("Pair your Meshtastic radio in Android Bluetooth settings first"); return@connect }
                android.app.AlertDialog.Builder(this).setTitle("Which radio?").setItems(names.toTypedArray()) { _, i ->
                    radio.setText(names[i]); findViewById<Button>(R.id.connect).performClick()
                }.show()
                return@connect
            }
            window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)   // screen stays on while Burst holds the radio
            mesh?.close()
            prefs.edit().putString("radio", radio.text.toString().trim()).putString("to", toF.text.toString().trim()).apply()
            startForegroundService(android.content.Intent(this, BurstService::class.java))   // keep listening with the screen off
            val app = applicationContext
            mesh = MeshBle(app, ::say) { from, to, ch, port, payload ->
                // Decoding runs the model (seconds): never on the BLE pump thread. One worker keeps arrival order.
                Session.decode.execute { runCatching { onData(from, to, ch, port, payload) }.onFailure { say("RX handler error: ${it.javaClass.simpleName}") } }
            }.also { m -> m.onReady = { thread { sendHello(m) } }; m.connect(radio.text.toString().trim()) }
            proxy(prefs)
            // test hook: send once the radio is ready. Needs a fresh "nonce": Android re-delivers the launch intent when it
            // re-creates the app, and without this the same test message would go out again on every re-create.
            val nonce = if (hooks) intent.getStringExtra("nonce") else null
            if (nonce != null && nonce != prefs.getString("nonce", null)) intent.getStringExtra("msg")?.let { msg ->
                prefs.edit().putString("nonce", nonce).apply()
                thread { val m = mesh ?: return@thread; repeat(60) { if (!m.ready) Thread.sleep(500) }; runOnUiThread { input.setText(msg); send.performClick() } }
            }
        }
        if (codec != null) { send.isEnabled = true }
        else thread {
            val model = File(filesDir, "model.gguf")
            if (!model.exists()) {
                say("First run: downloading the codec model (generic Qwen2.5-0.5B, 531 MB, once) — Wi-Fi recommended")
                // the foreground service + its wake lock keep a screen-off phone from freezing the download
                try { startForegroundService(android.content.Intent(this, BurstService::class.java)) } catch (e: Exception) { say("Keep Burst open until the download finishes") }
                val ok = try { ModelFetch.ensure(model) { p -> runOnUiThread { input.hint = "Getting the model… $p%" }; if (p % 10 == 0) say("Model download $p%") } } catch (e: Exception) { say("Download failed: ${e.message}"); false }
                if (!ok) { say("Model not ready — reopen Burst to resume the download"); return@thread }
            }
            runOnUiThread { input.hint = "Message" }
            BurstNative.init(applicationInfo.nativeLibraryDir)
            val t0 = System.currentTimeMillis()
            val threads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
            if (!BurstNative.load(model.path, threads)) { say("Model failed to load"); return@thread }
            Det.modelPath = model.path; Det.threads = threads          // same GGUF; detlm loads lazily on the first 0x08
            val c = Codec(assets.open("codes.bin").readBytes(), assets.open("tables.json").bufferedReader().readText())
            codec = c
            say("Model loaded in ${System.currentTimeMillis() - t0} ms")
            val held = synchronized(early) { early.filterIsInstance<Early>().also { early.clear() } }
            if (held.isNotEmpty()) say("decoding ${held.size} message(s) that arrived while loading")
            held.forEach { e -> Session.decode.execute { runCatching { onData(e.from, e.to, e.ch, e.port, e.payload) } } }
            if (hooks && intent.getBooleanExtra("selftest", false)) {      // test hook only: an everyday open goes straight to the chat
                selfTest(c)
                // the 0x08 selftest costs ~26 s on a current phone: only with the Force-0x08 toggle on or `--ez selftest08 true`
                if (Codec.force08 || intent.getBooleanExtra("selftest08", false)) selfTest08(c)
            }
            runOnUiThread {
                send.isEnabled = true
                if (mesh == null) findViewById<Button>(R.id.connect).performClick()   // the app's job is to listen: connect on open
            }
        }
        findViewById<Button>(R.id.aiAccounts).setOnClickListener { aiAccounts() }
        val aiBtn = findViewById<Button>(R.id.ai)
        aiBtn.setOnClickListener { aiAsk(input) }
        aiShow = { aiBtn.visibility = if (Ai.accounts(this).isEmpty()) android.view.View.GONE else android.view.View.VISIBLE }
        aiShow()          // opt-in: the compose row shows AI only after an account is added
        send.setOnClickListener {
            val text = input.text.toString(); if (codec == null || text.isBlank()) return@setOnClickListener
            input.setText("")
            val t = toF.text.toString().trim()
            val dest = if (t.equals("all", true) || t.isEmpty()) MeshBle.BROADCAST
                else t.removePrefix("!").toLongOrNull(16)?.takeIf { it in 0..0xFFFFFFFFL } ?: run {
                    say("Not sent: \"$t\" is not a node id (use !hex, like !1a2b3c4d, or all)"); input.setText(text); return@setOnClickListener }
            thread { sendText(dest, text, chF.text.toString().toIntOrNull() ?: 1) }
        }
        // review #13 (2026-11-04): the reply goes back on the channel the message came in on (the notice carries it).
        Session.reply = { dest, text, ch -> sendText(dest, text, ch) }
        if (hooks && adminReceiver == null) registerAdmin()
    }

    /** Test hook (TEST_HOOKS builds only): an AdminMessage for this phone's own radio, sent over adb. Guarded by a token
     *  only `run-as` can write (files/admin_token). Replies land in the log as ADMINRX. Registered once per activity. */
    private fun registerAdmin() {
        val r = object : android.content.BroadcastReceiver() {
            override fun onReceive(c: android.content.Context, i: android.content.Intent) {
                val tok = File(filesDir, "admin_token"); val m = mesh ?: return
                if (!tok.exists() || i.getStringExtra("key") != tok.readText().trim()) { say("ADMIN refused: bad key"); return }
                val b = Base64.decode(i.getStringExtra("b64") ?: return, Base64.DEFAULT)
                thread { m.send(b, m.myNode, 0, 6); say("ADMIN sent ${b.size} B to the radio") }
            }
        }
        registerReceiver(r, android.content.IntentFilter("com.gtc.burst.ADMIN"), RECEIVER_EXPORTED)
        adminReceiver = r
    }

    /** One packet from the radio, on the decode worker (Session.decode). */
    private fun onData(from: Long, to: Long, ch: Int, port: Int, payload: ByteArray) {
        val app = applicationContext
        if (port == 6) BurstLog.content("burst", "ADMINRX " + Base64.encodeToString(payload, Base64.NO_WRAP))
        if (port != 256) return
        val c = codec
        // review #18 (2026-11-04): before the model finished loading, codec was null and a message fell through as
        // "not a burst" — lost. Hold burst frames until it loads; they are decoded in arrival order right after.
        if (c == null && payload.isNotEmpty() && (payload[0].toInt() and 0xFF) in setOf(0x00, 0x03, 0x08, 0x12, 0x0E, 0x14)) {
            synchronized(early) { if (early.size < 64) early.add(Early(from, to, ch, port, payload)) }
            say("RX !%08x: held until the model loads".format(from)); return
        }
        if (payload.size == 1 && payload[0] == 0x09.toByte()) {               // TYPING: the peer's mind is on it now
            Session.typingFrom("!%08x".format(from)); Chat.onChange?.invoke("!%08x".format(from)); return
        }
        if (payload.isNotEmpty() && (payload[0] == 0x0C.toByte() || payload[0] == 0x0D.toByte())) {   // CATCH-UP
            if (to != MeshBle.BROADCAST) onCatchUp(from, ch, payload)
            return
        }
        if (payload.isNotEmpty() && payload[0] != 0x06.toByte()) Session.typingDone("!%08x".format(from))
        if (payload.size >= 3 && payload[0] == 0x06.toByte()) {               // they refused a frame of mine
            if (to != MeshBle.BROADCAST) thread { answerAsk(from, ch, payload) }
            return
        }
        if (payload.size == 5 && payload[0] == Wire2.ASK) { if (to != MeshBle.BROADCAST) thread { answerAsk(from, ch, payload) }; return }
        if (payload.size == 4 && payload[0] == Wire2.MISSING) { if (to != MeshBle.BROADCAST) thread { answerMissing(from, ch, payload) }; return }
        if (c != null && payload.isNotEmpty() && payload[0] == Wire2.CHAIN) { onChain2(c, from, to, ch, payload); return }
        if (c != null && payload.size >= 6 && payload[0] == Wire2.CODEC) { onSingle2(c, from, to, ch, payload); return }
        if (c != null && payload.isNotEmpty() && payload[0] == 0x12.toByte()) {
            val a = asm ?: Chain.Assembler(c).also { asm = it }
            val hist = History.of(app, from)
            val whole = runCatching { a.add(from, payload, hist.candidates()) }.getOrElse { say("link undecodable: ${it.message}"); null }
            if (whole == null) { say("RX !%08x link %s".format(from, a.progress(from, payload))); return }
            if (whole.verified08) hist.markReads08()                 // it reads the det codec: I may send 0x08 to it
            if (whole.allVerified) hist.add(whole.text)
            // review #5 (2026-11-04): a chain with a refused link was silently lost. Every link the sender sent is
            // indexed by its own check, so asking with THIS link's check gets the whole message resent.
            else if (to != MeshBle.BROADCAST) askFor(from, ch, payload)
            say("RX !%08x (%d packets%s): %s".format(from, whole.total, if (whole.verified08) ", 0x08" else "", whole.text), content = true)
            if (whole.allVerified) BurstService.notify(app, "!%08x".format(from), whole.text, ch)   // review #13: its own channel
            return
        }
        val hist = History.of(app, from)
        val cb = payload.firstOrNull()?.toInt() ?: -1
        val text = if (c != null && payload.size > 3 && (cb == 0x03 || cb == 0x00 || cb == 0x08))
            runCatching { c.uncrunch(payload, hist.candidates()).let {
                if (it.verified) {
                    hist.add(it.text)
                    BurstService.notify(app, "!%08x".format(from), it.text, ch)
                    if (it.codec == 8) hist.markReads08()
                    "${it.text}  [${if (it.codec == 8) "0x08, history ${it.tail?.size ?: 0}" else "history ${if (cb == 0x03) payload[5].toInt() and 0xFF else 0}"}]"
                } else if (it.codec == 8) { askFor(from, ch, payload); "(REFUSED 0x08 — ${it.reason}; asked for a resend)" }
                else { if (to != MeshBle.BROADCAST) askFor(from, ch, payload)        // review #5: refused 0x03 single → ASK
                       "(UNVERIFIED — check failed; text withheld, ${it.text.length} chars; asked for a resend)" } } }   // review #6: a failed check is not shown
                .getOrElse { "<undecodable: ${it.message}>" }
            else "<${payload.size} bytes, not a burst>"
        say("RX !%08x ch%d (%d B): %s".format(from, ch, payload.size, text), content = true)
        BurstLog.content("burst", "RXWIRE " + Base64.encodeToString(payload, Base64.NO_WRAP))
    }

    /** CATCH-UP ON RECONNECT (his design 11-06, PLAN.md): on every radio-ready, say HELLO with my last checks; the far end
     *  finds the newest entry we both hold, trims to it, and answers ALIGNED; I trim to the same entry. Drift is fixed
     *  before the next message, so nothing is refused for it. What either side trims is shown, never silently lost. */
    private val helloOut = java.util.concurrent.ConcurrentHashMap<Long, List<ByteArray>>()

    private fun sendHello(m: MeshBle) {
        val dest = runOnUiThreadBlocking { peerId() }.removePrefix("!").toLongOrNull(16)?.takeIf { it in 1..0xFFFFFFFEL } ?: return
        val ch = runCatching { runOnUiThreadBlocking { findViewById<EditText>(R.id.chan).text.toString().toIntOrNull() } }.getOrNull() ?: 0
        val checks = History.of(applicationContext, dest).helloChecks()
        helloOut[dest] = checks
        Thread.sleep(1500)                                                    // let the radio settle after config
        if (!m.ready) return
        m.send(byteArrayOf(0x0C, checks.size.toByte()) + checks.fold(ByteArray(0)) { a, c -> a + c }, dest, ch)
        say("Catch-up: HELLO to !%08x (%d checks)".format(dest, checks.size))
    }

    private fun onCatchUp(from: Long, ch: Int, b: ByteArray) {
        val hist = History.of(applicationContext, from)
        if (b[0] == 0x0C.toByte()) {                                          // their HELLO: align, answer
            val n = if (b.size > 1) b[1].toInt() and 0xFF else 0
            val listed = (0 until n).filter { b.size >= 4 + 2 * it }.map { b.copyOfRange(2 + 2 * it, 4 + 2 * it) }
            val (k, j) = History.alignAnchor(listed, hist.snapshot())
            val gone = hist.trimTo(j + 1)
            mesh?.let { m -> if (m.ready) m.send(byteArrayOf(0x0D, k.toByte()) + (if (k > 0) listed[k - 1] else ByteArray(0)), from, ch) }
            say("Catch-up: HELLO from !%08x -> aligned at %d%s".format(from, k, if (gone.isEmpty()) "" else "; set aside: " + gone.joinToString(" | ") { it.take(60) }))
        } else {                                                              // ALIGNED: their answer to my HELLO
            val sent = helloOut.remove(from) ?: run { say("Catch-up: ALIGNED from !%08x with no HELLO out — ignored".format(from)); return }
            val k = if (b.size > 1) b[1].toInt() and 0xFF else 0
            if (k > 0 && (k > sent.size || b.size < 4 || !b.copyOfRange(2, 4).contentEquals(sent[k - 1]))) {
                say("Catch-up: ALIGNED from !%08x does not match my HELLO — ignored".format(from)); return
            }
            val base = maxOf(0, hist.size() - sent.size)
            val gone = hist.trimTo(if (k > 0) base + k else 0)
            say("Catch-up with !%08x: aligned at %d%s".format(from, k, if (gone.isEmpty()) ", nothing drifted" else "; set aside: " + gone.joinToString(" | ") { it.take(60) }))
        }
    }

    private fun <T> runOnUiThreadBlocking(f: () -> T): T {
        val q = java.util.concurrent.ArrayBlockingQueue<Result<T>>(1)
        runOnUiThread { q.put(runCatching(f)) }
        return q.take().getOrThrow()
    }

    /** Split on our shared history, send link by link, and on delivery add it to that history. True = delivered. */
    private fun sendText(dest: Long, text: String, ch: Int): Boolean {
        val c = codec ?: return false
        val hist = if (dest == MeshBle.BROADCAST) null else History.of(applicationContext, dest)   // broadcasts have no one shared history
        val use08 = Codec.force08 || (hist?.reads08() == true)                    // 0x08 only once the peer has shown it reads it
        val label = if (dest == MeshBle.BROADCAST) "all" else "!%08x".format(dest)
        // The bubble goes up the moment Send is tapped, and every stage shows on it, so the user can see whether it
        // went and whether it is working. Nothing fails into the debug log alone.
        val line = if (dest == MeshBle.BROADCAST) null else Chat.add(applicationContext, label, true, text, "packing…")
        fun mark(s: String) { line?.let { Chat.mark(applicationContext, label, it, s) } }
        val queue = ArrayDeque(Chain.parts(text)); var done = 0
        val pasteId = kotlin.random.Random.nextInt(256)
        while (queue.isNotEmpty()) {
            val part = queue.removeFirst()
            val tag = if (done + queue.size > 0) "part ${done + 1}/${done + 1 + queue.size} · " else ""
            mark("${tag}packing…")
            val packets = runCatching { Chain.split(c, part, hist?.tail() ?: emptyList(), use08) }.getOrElse {
                if (part.length > 200 && "cap is 16" in (it.message ?: "")) {       // over 16 links: halve and go on
                    Chain.parts(part, part.length / 2 + 1).reversed().forEach(queue::addFirst); continue }
                say("Not sent: ${it.message}"); mark("NOT sent — ${tag}${it.message}"); return false }
            val wireByte = packets[0].let { if (it[0] == 0x12.toByte()) it[3] else it[0] }.toInt() and 0xFF
            say("${part.length} chars -> ${packets.size} packet(s), ${packets.sumOf { it.size }} bytes, codec 0x%02x, history ${hist?.tail()?.size ?: 0}".format(wireByte))
            if (done + queue.size > 0) {                             // a long paste: say "part i of n" first
                val n = done + 1 + queue.size
                transmit(dest, listOf(byteArrayOf(Chain.PARTS, pasteId.toByte(), (done + 1).toByte(), n.toByte())), ch, label, part, record = false)
            }
            val ok = transmit(dest, packets, ch, label, part, { i -> mark("${tag}sending ${i + 1}/${packets.size}…") })
            if (!ok) { mark("NOT delivered — ${tag}stopped"); return false }
            hist?.add(part)                                         // delivered: it joins the history we share
            done++
        }
        mark("delivered")
        return true
    }

    /** Golden vectors from the reference encoder: the phone must decode the reference bytes, and its own encode must
     *  round-trip (and is compared byte for byte with the reference). */
    private fun selfTest(c: Codec) {
        val vec = JSONArray(assets.open("vectors.json").bufferedReader().readText())
        var tokOk = 0; var decOk = 0; var rtOk = 0; var sameBytes = 0
        val phoneWire = JSONArray()
        for (i in 0 until vec.length()) {
            val v = vec.getJSONObject(i); val text = v.getString("text")
            val want = v.getJSONArray("tokens").let { a -> IntArray(a.length()) { a.getInt(it) } }
            if (BurstNative.tokenize(text.toByteArray()).contentEquals(want)) tokOk++  // golden vectors are the raw open-codec wire
            val refBytes = Base64.decode(v.getString("wire_b64"), Base64.NO_WRAP)
            val d = c.uncrunchRaw(refBytes); if (d.text == text) decOk++ else say("ref->phone MISS [$i]: ${d.text}", content = true)
            val mine = c.crunchRaw(text)
            if (mine.contentEquals(refBytes)) sameBytes++
            if (c.uncrunchRaw(mine).text == text) rtOk++
            phoneWire.put(Base64.encodeToString(mine, Base64.NO_WRAP))
        }
        say("SELFTEST tokens $tokOk/${vec.length()} · ref->phone exact $decOk/${vec.length()} · phone round trip $rtOk/${vec.length()} · identical bytes $sameBytes/${vec.length()}")
        BurstLog.content("burst", "PHONEWIRE " + phoneWire.toString())
    }

    /** Frame 0x08 (det): reference-encoder frames must decode EXACT through History.candidates (the check picks the
     *  history), the phone's encode must be byte-IDENTICAL to the reference, and a flipped coder byte must be REFUSED. */
    private fun selfTest08(c: Codec) {
        val vec = JSONArray(assets.open("vectors08.json").bufferedReader().readText())
        var decOk = 0; var same = 0; var refused = 0; var negTried = 0; var tailOk = 0
        val t0 = System.currentTimeMillis()
        for (i in 0 until vec.length()) {
            val v = vec.getJSONObject(i); val text = v.getString("text")
            val history = v.getJSONArray("history").let { a -> List(a.length()) { a.getString(it) } }
            val tail = v.getJSONArray("tail").let { a -> List(a.length()) { a.getString(it) } }
            val ref = Base64.decode(v.getString("frame_b64"), Base64.NO_WRAP)
            val d = c.uncrunch(ref, History.candidates(history))
            if (d.verified && d.text == text) { decOk++; if (d.tail == tail) tailOk++ } else say("0x08 ref->phone MISS [$i]: ${d.reason ?: d.text}", content = true)
            val mine = c.crunch08(text, tail)
            if (mine != null && mine.contentEquals(ref)) same++ else say("0x08 bytes DIFFER [$i]: ${mine?.size ?: -1} B vs ref ${ref.size} B ${BurstNative.detLastReason()}")
            if (i < 2 && ref.size > 6) {                            // negative (2 of them: each refusal tries every candidate)
                negTried++
                val bad = ref.copyOf(); val fi = 4 + (ref.size - 4) / 2; bad[fi] = (bad[fi].toInt() xor 0x5A).toByte()
                val n = c.uncrunch(bad, History.candidates(history))
                if (!n.verified) refused++ else say("0x08 flipped byte NOT refused [$i]: ${n.text}")
            }
        }
        say("SELFTEST08 ref->phone exact $decOk/${vec.length()} (history picked ${tailOk}) · identical bytes $same/${vec.length()} · flipped refused $refused/$negTried · ${System.currentTimeMillis() - t0} ms · detlm ${BurstNative.detVersion()} peak ${BurstNative.detPeakRssKib() / 1024} MiB")
    }
}
