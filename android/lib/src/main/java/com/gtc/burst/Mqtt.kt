package com.gtc.burst

import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.OutputStream
import java.net.InetSocketAddress
import java.net.Socket
import kotlin.concurrent.thread

/**
 * The smallest MQTT 3.1.1 client the radio's proxy needs: CONNECT with user/password, SUBSCRIBE, PUBLISH at QoS 0,
 * PINGREQ. One socket, one reader thread; on any error it waits and reconnects until [close].
 * Used to carry the radio's MqttClientProxyMessage over the phone's network (what the Meshtastic app does when
 * "proxy to client" is on), so a radio with its own Wi-Fi off still reaches its broker.
 */
class Mqtt(private val host: String, private val port: Int, private val user: String, private val pass: String,
           private val subs: List<String>, private val say: (String) -> Unit,
           private val onMessage: (topic: String, payload: ByteArray, retained: Boolean) -> Unit) {
    @Volatile private var running = true
    @Volatile private var out: OutputStream? = null
    @Volatile private var sock: Socket? = null
    @Volatile var up = false; private set
    // Review #19 (2026-11-04): 6 random digits collided on a shared broker; 64 random bits do not.
    private val clientId = "burst" + java.util.UUID.randomUUID().toString().replace("-", "").take(16)

    init { thread(name = "burst-mqtt") { loop() } }

    fun close() { running = false; up = false; runCatching { sock?.close() } }

    /** QoS 0 publish; dropped (false) while the broker is down — the mesh resends what matters. */
    @Synchronized fun publish(topic: String, payload: ByteArray, retain: Boolean = false): Boolean {
        val o = out ?: return false
        val body = ByteArrayOutputStream().apply { str(topic); write(payload) }.toByteArray()
        return runCatching { packet(o, 0x30 or (if (retain) 1 else 0), body) }.isSuccess
    }

    private fun loop() {
        var wait = 2000L
        while (running) {
            try {
                val s = Socket(); s.connect(InetSocketAddress(host, port), 15000); s.soTimeout = 90_000   // not inside apply{}: there `port` is the socket's own (0)
                sock = s; val o = s.getOutputStream(); val i = DataInputStream(s.getInputStream())
                val c = ByteArrayOutputStream().apply {
                    str("MQTT"); write(4); write(0xC2)                     // level 4; user + password + clean session
                    write(0); write(60)                                    // keep-alive 60 s
                    str(clientId); str(user); str(pass)
                }.toByteArray()
                synchronized(this) { packet(o, 0x10, c) }
                val (t, ack) = read(i)
                if (t != 0x20 || ack.size < 2 || ack[1].toInt() != 0) throw IllegalStateException("broker refused (rc ${ack.getOrNull(1)})")
                out = o; up = true; wait = 2000L
                say("MQTT up $host:$port")
                synchronized(this) {
                    subs.forEachIndexed { n, topic ->
                        packet(o, 0x82, ByteArrayOutputStream().apply { write(0); write(n + 1); str(topic); write(0) }.toByteArray())
                    }
                }
                val pinger = thread(name = "burst-mqtt-ping") {
                    while (running && sock === s) { Thread.sleep(45_000); if (sock === s) synchronized(this) { runCatching { packet(o, 0xC0, ByteArray(0)) } } }
                }
                while (running) {
                    val (type, b) = read(i)
                    if (type and 0xF0 != 0x30) continue                     // SUBACK, PINGRESP: nothing to do
                    val tl = ((b[0].toInt() and 0xFF) shl 8) or (b[1].toInt() and 0xFF)
                    val topic = String(b, 2, tl)
                    val qos = (type shr 1) and 3
                    val start = 2 + tl + (if (qos > 0) 2 else 0)
                    onMessage(topic, b.copyOfRange(start, b.size), type and 1 == 1)
                }
                pinger.interrupt()
            } catch (e: Exception) {
                if (running) say("MQTT down (${e.javaClass.simpleName}: ${e.message}); retry in ${wait / 1000} s")
            }
            up = false; out = null; runCatching { sock?.close() }; sock = null
            if (!running) break
            Thread.sleep(wait); wait = (wait * 2).coerceAtMost(60_000)
        }
    }

    private fun ByteArrayOutputStream.str(s: String) { val b = s.toByteArray(); write(b.size shr 8); write(b.size and 0xFF); write(b) }

    private fun packet(o: OutputStream, type: Int, body: ByteArray) {
        val h = ByteArrayOutputStream(); h.write(type)
        var n = body.size
        do { var d = n % 128; n /= 128; if (n > 0) d = d or 0x80; h.write(d) } while (n > 0)
        o.write(h.toByteArray() + body); o.flush()
    }

    private fun read(i: DataInputStream): Pair<Int, ByteArray> {
        val type = i.readUnsignedByte()
        var len = 0; var mul = 1
        while (true) { val d = i.readUnsignedByte(); len += (d and 0x7F) * mul; if (d and 0x80 == 0) break; mul *= 128; if (mul > 128 * 128 * 128) throw IllegalStateException("bad length") }
        if (len > 1 shl 20) throw IllegalStateException("packet too large")
        val b = ByteArray(len); i.readFully(b); return type to b
    }
}
