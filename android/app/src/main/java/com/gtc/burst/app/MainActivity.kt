package com.gtc.burst.app

import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.gtc.burst.BurstNative
import com.gtc.burst.Chain
import com.gtc.burst.Codec
import com.gtc.burst.Det
import com.gtc.burst.History
import com.gtc.burst.MeshBle
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
    /** The 2-byte check a frame carries (a chain link's sits inside its body), as burst_rx._chk. */
    private fun chk(p: ByteArray) = if (p[0] == 0x12.toByte()) p.copyOfRange(4, 6) else p.copyOfRange(1, 3)

    /** One link at a time, each waiting for its routing ACK; every delivered frame is indexed by its check (whole
     *  message text) so the peer can ASK for it again. Returns true when every link got through. */
    private fun transmit(dest: Long, packets: List<ByteArray>, ch: Int, label: String, text: String,
                         onLink: (Int) -> Unit = {}, record: Boolean = true): Boolean {
        val m = mesh ?: run { say("Not sent: no radio — tap Connect"); return false }
        if (!m.ready) say("Radio not ready yet — waiting up to 60 s before sending")
        repeat(120) { if (!m.ready) Thread.sleep(500) }              // 10-23 22:18: a send tapped mid-connect was dropped silently
        if (!m.ready) { say("Not sent: the radio never became ready"); return false }
        for ((i, p) in packets.withIndex()) {
            onLink(i)
            var ok = false
            for (attempt in 1..3) {
                val id = m.send(p, dest, ch)
                Log.i("burst", "TXWIRE " + Base64.encodeToString(p, Base64.NO_WRAP))
                val err = if (dest == MeshBle.BROADCAST) 0.also { Thread.sleep(3000) } else m.awaitAck(id, 30000)
                say("TX %d/%d id %08x -> %s: %s".format(i + 1, packets.size, id, label,
                    when (err) { null -> "no ACK"; 0 -> "delivered"; else -> "error $err" }))
                if (err == 0) { ok = true; break }
            }
            if (!ok) { say("Stopped: link ${i + 1} never got through"); return false }
        }
        if (dest != MeshBle.BROADCAST && record) {
            val prefs = getSharedPreferences("burst", MODE_PRIVATE); val key = "sent_%08x".format(dest)
            val idx = org.json.JSONObject(prefs.getString(key, "{}")!!)
            for (p in packets) idx.put(hex(chk(p)), text)
            while (idx.length() > 64) idx.remove(idx.keys().next())
            prefs.edit().putString(key, idx.toString()).apply()
        }
        return true
    }

    /** ASK (0x06 + check) from a peer that refused one of my frames: resend that message on the BASE prime, once. */
    private fun answerAsk(from: Long, ch: Int, b: ByteArray) {
        val c = codec ?: return
        val prefs = getSharedPreferences("burst", MODE_PRIVATE); val key = "sent_%08x".format(from)
        val idx = org.json.JSONObject(prefs.getString(key, "{}")!!); val k = hex(b.copyOfRange(1, 3))
        val text = idx.optString(k, "")
        if (text.isEmpty()) { say("ASK from !%08x for $k: not one of mine".format(from)); return }
        if (prefs.getBoolean("resent_$k", false)) { say("ASK for $k: already resent once"); return }
        // One resend per MESSAGE, not per link: the box asks once for each refused link, and each ASK used to resend the
        // whole part — 13 asks became 13 full resends on air (10-24 18:03). The first ASK resends it; the rest are no-ops.
        val tk = "resent_text_" + hex(sha256(text.toByteArray()).copyOf(6))
        synchronized(Session) {
            if (prefs.getBoolean(tk, false)) { prefs.edit().putBoolean("resent_$k", true).apply(); say("ASK for $k: its message is already being resent"); return }
            prefs.edit().putBoolean("resent_$k", true).putBoolean(tk, true).apply()
        }
        val use08 = Codec.force08 || History(applicationContext, from).reads08()
        val packets = Chain.split(c, text, emptyList(), use08)
        say("ASK from !%08x: resending on the base prime (%d packet(s), %d B)".format(from, packets.size, packets.sumOf { it.size }))
        transmit(from, packets, ch, "!%08x".format(from), text)
    }

    /** I refused a 0x08 frame: ask the sender once for it again, without our history. */
    private fun askFor(from: Long, ch: Int, p: ByteArray) {
        val k = hex(chk(p)); if (!asked.add(k)) return
        thread { mesh?.let { m -> if (m.ready) { m.send(byteArrayOf(0x06) + chk(p), from, ch); say("ASK sent to !%08x for $k".format(from)) } } }
    }

    private fun say(s: String) { Log.i("burst", s); synchronized(Session.log) { Session.log.append(s + "\n") }; Session.sink?.invoke(s) }

    override fun onDestroy() { if (Session.sink === sink) Session.sink = null; if (Chat.onChange === redraw) Chat.onChange = null; super.onDestroy() }
    override fun onResume() { super.onResume(); Session.visible = true; ui.post(tick) }
    override fun onPause() { Session.visible = false; ui.removeCallbacks(tick); super.onPause() }

    // ---- the chat screen (10-24) ----
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
        val (c, t) = when { m == null -> 0xFF6B7480.toInt() to "Off · tap"; !m.ready -> 0xFFE3A008.toInt() to "Connecting"; else -> 0xFF2EA043.toInt() to "Listening" }
        (findViewById<android.view.View>(R.id.dot).background.mutate() as android.graphics.drawable.GradientDrawable).setColor(c)
        findViewById<TextView>(R.id.statusText).text = t
        if (Session.typing(peerId()) != shownTyping) drawChat()     // the dots go up or time out
        ui.postDelayed(this, 1000)
    } }
    private val sink: (String) -> Unit = { s -> runOnUiThread { log.append(s + "\n") } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        log = findViewById(R.id.log)
        synchronized(Session.log) { log.text = Session.log.toString() }   // a rebuilt screen shows what the process already heard
        Session.sink = sink
        chatBox = findViewById(R.id.chat); chatScroll = findViewById(R.id.chatScroll)
        Chat.onChange = redraw
        val settings = findViewById<android.view.View>(R.id.settings)
        findViewById<TextView>(R.id.gear).setOnClickListener { settings.visibility = if (settings.visibility == android.view.View.VISIBLE) android.view.View.GONE else android.view.View.VISIBLE }
        findViewById<android.view.View>(R.id.status).setOnClickListener { if (mesh?.ready != true) findViewById<Button>(R.id.connect).performClick() }
        val input = findViewById<EditText>(R.id.input)
        val send = findViewById<Button>(R.id.send)
        val radio = findViewById<EditText>(R.id.radio); val toF = findViewById<EditText>(R.id.to); val chF = findViewById<EditText>(R.id.chan)
        val prefs = getSharedPreferences("burst", MODE_PRIVATE)
        val need = arrayOf(android.Manifest.permission.BLUETOOTH_CONNECT, android.Manifest.permission.POST_NOTIFICATIONS)
            .filter { checkSelfPermission(it) != android.content.pm.PackageManager.PERMISSION_GRANTED }
        if (need.isNotEmpty()) requestPermissions(need.toTypedArray(), 1)
        // Back sends Burst to the background instead of closing it, so the radio stays held (10-24)
        onBackPressedDispatcher.addCallback(this, object : androidx.activity.OnBackPressedCallback(true) {
            override fun handleOnBackPressed() { moveTaskToBack(true) }
        })
        intent.getStringExtra("radio")?.let { radio.setText(it) }       // test hooks: am start --es radio <name> --es to !<node>
        intent.getStringExtra("to")?.let { toF.setText(it) }
        prefs.getString("radio", null)?.let { if (!intent.hasExtra("radio")) radio.setText(it) }
        prefs.getString("to", null)?.let { if (!intent.hasExtra("to")) toF.setText(it) }
        findViewById<TextView>(R.id.peer).text = MeshBle.name(peerId())
        drawChat()
        // Dev toggle: force frame 0x08 to every peer (else 0x08 only to a peer that has sent me a verified 0x08).
        // Persisted; also settable by the test hook `--ez force08 true|false`.
        val force08 = findViewById<CheckBox>(R.id.force08)
        if (intent.hasExtra("force08")) prefs.edit().putBoolean("force08", intent.getBooleanExtra("force08", false)).apply()
        Codec.force08 = prefs.getBoolean("force08", true)   // public build: every Burst app reads 0x08, so it is the default; force08.isChecked = Codec.force08
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
                if (port == 6) Log.i("burst", "ADMINRX " + Base64.encodeToString(payload, Base64.NO_WRAP))
                if (port != 256) return@MeshBle
                val c = codec
                if (payload.size == 1 && payload[0] == 0x09.toByte()) {               // TYPING: the peer's mind is on it now
                    Session.typingFrom("!%08x".format(from)); Chat.onChange?.invoke("!%08x".format(from)); return@MeshBle
                }
                if (payload.isNotEmpty() && payload[0] != 0x06.toByte()) Session.typingDone("!%08x".format(from))
                if (payload.size >= 3 && payload[0] == 0x06.toByte()) {               // they refused a frame of mine
                    if (to != MeshBle.BROADCAST) thread { answerAsk(from, ch, payload) }
                    return@MeshBle
                }
                if (c != null && payload.isNotEmpty() && payload[0] == 0x12.toByte()) {
                    val a = asm ?: Chain.Assembler(c).also { asm = it }
                    val hist = History(app, from)
                    val whole = runCatching { a.add(from, payload, hist.candidates()) }.getOrElse { say("link undecodable: ${it.message}"); null }
                    if (whole == null) { say("RX !%08x link %s".format(from, a.progress(from, payload))); return@MeshBle }
                    if (whole.verified08) hist.markReads08()                 // it reads the det codec: I may send 0x08 to it
                    if (whole.allVerified) hist.add(whole.text)
                    say("RX !%08x (%d packets%s): %s".format(from, whole.total, if (whole.verified08) ", 0x08" else "", whole.text))
                    if (whole.allVerified) BurstService.notify(app, "!%08x".format(from), whole.text)
                    return@MeshBle
                }
                val hist = History(app, from)
                val cb = payload.firstOrNull()?.toInt() ?: -1
                val text = if (c != null && payload.size > 3 && (cb == 0x03 || cb == 0x00 || cb == 0x08))
                    runCatching { c.uncrunch(payload, hist.candidates()).let {
                        if (it.verified) {
                            hist.add(it.text)
                            BurstService.notify(app, "!%08x".format(from), it.text)
                            if (it.codec == 8) hist.markReads08()
                            "${it.text}  [${if (it.codec == 8) "0x08, history ${it.tail?.size ?: 0}" else "history ${if (cb == 0x03) payload[5].toInt() and 0xFF else 0}"}]"
                        } else if (it.codec == 8) { askFor(from, ch, payload); "(REFUSED 0x08 — ${it.reason}; asked for a resend)" }
                        else "(UNVERIFIED — check failed) " + it.text } }
                        .getOrElse { "<undecodable: ${it.message}>" }
                    else "<${payload.size} bytes, not a burst>"
                say("RX !%08x ch%d (%d B): %s".format(from, ch, payload.size, text))
                Log.i("burst", "RXWIRE " + Base64.encodeToString(payload, Base64.NO_WRAP))
            }.also { it.connect(radio.text.toString().trim()) }
            // Admin pass-through (10-24): an AdminMessage for MY radio, from the box over adb. Guarded by a token only
            // `run-as` (debug build, the box) can write: files/admin_token. Replies land in the log as ADMINRX.
            registerReceiver(object : android.content.BroadcastReceiver() {
                override fun onReceive(c: android.content.Context, i: android.content.Intent) {
                    val tok = File(filesDir, "admin_token"); val m = mesh ?: return
                    if (!tok.exists() || i.getStringExtra("key") != tok.readText().trim()) { say("ADMIN refused: bad key"); return }
                    val b = Base64.decode(i.getStringExtra("b64") ?: return, Base64.DEFAULT)
                    thread { m.send(b, m.myNode, 0, 6); say("ADMIN sent ${b.size} B to my radio") }
                }
            }, android.content.IntentFilter("com.gtc.burst.ADMIN"), RECEIVER_EXPORTED)
            // test hook: send once the radio is ready. Needs a fresh "nonce": Android re-delivers the launch intent when it
            // re-creates the app, and without this the same test message went out four more times (10-23).
            val nonce = intent.getStringExtra("nonce")
            if (nonce != null && nonce != prefs.getString("nonce", null)) intent.getStringExtra("msg")?.let { msg ->
                prefs.edit().putString("nonce", nonce).apply()
                thread { val m = mesh!!; repeat(60) { if (!m.ready) Thread.sleep(500) }; runOnUiThread { input.setText(msg); send.performClick() } }
            }
        }
        if (codec != null) { send.isEnabled = true }
        else thread {
            val model = File(filesDir, "model.gguf")
            if (!model.exists()) {
                say("First run: downloading the codec model (generic Qwen2.5-0.5B, 531 MB, once) — Wi-Fi recommended")
                // the foreground service + its wake lock keep a screen-off phone from freezing the download (S25, 10-24: froze at 372 MB)
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
            if (intent.getBooleanExtra("selftest", false)) {      // test hook only: an everyday open goes straight to the chat (10-24)
                selfTest(c)
                // the 0x08 selftest costs ~26 s on the S25: only with the Force-0x08 toggle on or `--ez selftest08 true`
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
            val dest = if (t.equals("all", true) || t.isEmpty()) MeshBle.BROADCAST else t.removePrefix("!").toLong(16)
            thread { sendText(dest, text, chF.text.toString().toIntOrNull() ?: 1) }
        }
        // A reply typed into a message notice goes out the same way, on the channel in the Ch field (10-24)
        Session.reply = { dest, text, _ -> sendText(dest, text, chF.text.toString().toIntOrNull() ?: 1) }
    }

    /** Split on our shared history, send link by link, and on delivery add it to that history. True = delivered. */
    private fun sendText(dest: Long, text: String, ch: Int): Boolean {
        val c = codec ?: return false
        val hist = if (dest == MeshBle.BROADCAST) null else History(applicationContext, dest)   // broadcasts have no one shared history
        val use08 = Codec.force08 || (hist?.reads08() == true)                    // box rule: 0x08 only once the peer has shown it reads it
        val label = if (dest == MeshBle.BROADCAST) "all" else "!%08x".format(dest)
        // The bubble goes up the moment he taps Send, and every stage shows on it — his 10-24: "it is not clear if it
        // went or if it is working". Nothing fails into the debug log alone any more.
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

    /** The box's golden vectors: phone decodes the box's bytes, and the box's decoder must accept the phone's bytes. */
    private fun selfTest(c: Codec) {
        val vec = JSONArray(assets.open("vectors.json").bufferedReader().readText())
        var tokOk = 0; var decOk = 0; var rtOk = 0; var sameBytes = 0
        val phoneWire = JSONArray()
        for (i in 0 until vec.length()) {
            val v = vec.getJSONObject(i); val text = v.getString("text")
            val want = v.getJSONArray("tokens").let { a -> IntArray(a.length()) { a.getInt(it) } }
            if (BurstNative.tokenize(text.toByteArray()).contentEquals(want)) tokOk++  // golden vectors are the raw open-codec wire
            val boxBytes = Base64.decode(v.getString("wire_b64"), Base64.NO_WRAP)
            val d = c.uncrunchRaw(boxBytes); if (d.text == text) decOk++ else say("box->phone MISS [$i]: ${d.text}")
            val mine = c.crunchRaw(text)
            if (mine.contentEquals(boxBytes)) sameBytes++
            if (c.uncrunchRaw(mine).text == text) rtOk++
            phoneWire.put(Base64.encodeToString(mine, Base64.NO_WRAP))
        }
        say("SELFTEST tokens $tokOk/${vec.length()} · box->phone exact $decOk/${vec.length()} · phone round trip $rtOk/${vec.length()} · identical bytes $sameBytes/${vec.length()}")
        Log.i("burst", "PHONEWIRE " + phoneWire.toString())
    }

    /** Frame 0x08 (det): the box's det_codec.py frames must decode EXACT through History.candidates (the check picks the
     *  history), the phone's encode must be byte-IDENTICAL to the box's, and a flipped coder byte must be REFUSED. */
    private fun selfTest08(c: Codec) {
        val vec = JSONArray(assets.open("vectors08.json").bufferedReader().readText())
        var decOk = 0; var same = 0; var refused = 0; var negTried = 0; var tailOk = 0
        val t0 = System.currentTimeMillis()
        for (i in 0 until vec.length()) {
            val v = vec.getJSONObject(i); val text = v.getString("text")
            val history = v.getJSONArray("history").let { a -> List(a.length()) { a.getString(it) } }
            val tail = v.getJSONArray("tail").let { a -> List(a.length()) { a.getString(it) } }
            val box = Base64.decode(v.getString("frame_b64"), Base64.NO_WRAP)
            val d = c.uncrunch(box, History.candidates(history))
            if (d.verified && d.text == text) { decOk++; if (d.tail == tail) tailOk++ } else say("0x08 box->phone MISS [$i]: ${d.reason ?: d.text}")
            val mine = c.crunch08(text, tail)
            if (mine != null && mine.contentEquals(box)) same++ else say("0x08 bytes DIFFER [$i]: ${mine?.size ?: -1} B vs box ${box.size} B ${BurstNative.detLastReason()}")
            if (i < 2 && box.size > 6) {                            // negative (2 of them: each refusal tries every candidate)
                negTried++
                val bad = box.copyOf(); val fi = 4 + (box.size - 4) / 2; bad[fi] = (bad[fi].toInt() xor 0x5A).toByte()
                val n = c.uncrunch(bad, History.candidates(history))
                if (!n.verified) refused++ else say("0x08 flipped byte NOT refused [$i]: ${n.text}")
            }
        }
        say("SELFTEST08 box->phone exact $decOk/${vec.length()} (history picked ${tailOk}) · identical bytes $same/${vec.length()} · flipped refused $refused/$negTried · ${System.currentTimeMillis() - t0} ms · detlm ${BurstNative.detVersion()} peak ${BurstNative.detPeakRssKib() / 1024} MiB")
    }
}
