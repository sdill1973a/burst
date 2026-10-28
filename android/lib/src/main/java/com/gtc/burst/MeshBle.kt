package com.gtc.burst

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.content.Context
import java.io.ByteArrayOutputStream
import java.util.UUID
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread
import kotlin.random.Random

/**
 * Talks to a Meshtastic radio over BLE directly — the official app 2.8.3 exports no service (MeshService exported=false),
 * so there is no door to borrow. One BLE client per radio: while this holds it, the official app is disconnected.
 * Protocol (meshtastic ble_interface.py): write ToRadio to TORADIO; read FROMRADIO until empty; FROMNUM notifies "more".
 * Protobufs are hand-rolled for the few fields we need (numbers checked against meshtastic.protobuf.mesh_pb2).
 */
@SuppressLint("MissingPermission")
class MeshBle(private val ctx: Context, private val say: (String) -> Unit,
              private val onData: (from: Long, to: Long, channel: Int, port: Int, payload: ByteArray) -> Unit) {
    companion object {
        val SERVICE: UUID = UUID.fromString("6ba1b218-15a8-461f-9fa8-5dcae273eafd")
        val TORADIO: UUID = UUID.fromString("f75c76d2-129e-4dad-a1dd-7866124401e7")
        val FROMRADIO: UUID = UUID.fromString("2c55e69e-4993-11ed-b878-0242ac120002")
        val FROMNUM: UUID = UUID.fromString("ed9da18c-a800-4f66-a670-aa7547e34453")
        val CCCD: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
        const val BROADCAST = 0xFFFFFFFFL
        private const val OP_TIMEOUT_S = 15L
        /** Peer names as the mesh announces them (NodeInfo.user.long_name), "!hex" -> name. Nothing hard-coded. */
        val names = java.util.concurrent.ConcurrentHashMap<String, String>()
        fun name(id: String): String = names[id] ?: id
    }

    private var gatt: BluetoothGatt? = null
    private var wanted: String? = null                                  // set by connect(), cleared by close(): retry while set
    private val main = android.os.Handler(android.os.Looper.getMainLooper())
    @Volatile private var discovered = false
    private var toR: BluetoothGattCharacteristic? = null
    private var fromR: BluetoothGattCharacteristic? = null
    @Volatile var myNode: Long = 0; private set
    @Volatile var ready = false; private set
    private val acks = java.util.concurrent.ConcurrentHashMap<Long, Int>()   // packet id -> routing error (0 = delivered)

    /** Wait for the far end's routing ACK of packet [id]. Returns the error code (0 = delivered), or null on timeout. */
    fun awaitAck(id: Long, ms: Long): Int? {
        val end = System.currentTimeMillis() + ms
        while (System.currentTimeMillis() < end) { acks.remove(id)?.let { return it }; Thread.sleep(200) }
        return null
    }

    // GATT allows one operation in flight; every op goes through this queue and waits for its callback.
    private sealed class Op { class W(val b: ByteArray) : Op(); object R : Op() }
    private val ops = LinkedBlockingQueue<Op>()
    private val done = LinkedBlockingQueue<ByteArray>()          // cannot hold null: a write completion is WROTE
    private val WROTE = ByteArray(0)

    fun bondedNames(): List<String> =
        (ctx.getSystemService(BluetoothManager::class.java).adapter?.bondedDevices ?: emptySet()).mapNotNull { it.name }

    fun connect(namePrefix: String): Boolean {
        val dev: BluetoothDevice = ctx.getSystemService(BluetoothManager::class.java).adapter?.bondedDevices
            ?.firstOrNull { (it.name ?: "").startsWith(namePrefix, ignoreCase = true) } ?: run { say("No paired radio named $namePrefix*"); return false }
        say("Connecting to ${dev.name}…")
        wanted = namePrefix; discovered = false
        gatt = dev.connectGatt(ctx, false, cb, BluetoothDevice.TRANSPORT_LE)
        return true
    }

    fun close() { wanted = null; ready = false; gatt?.close(); gatt = null }

    /** One DataPacket on [port] (PRIVATE_APP = 256). [to] = node num, or BROADCAST. Returns the packet id. */
    fun send(payload: ByteArray, to: Long, channel: Int, port: Int = 256): Long {
        val id = Random.nextLong(1, 0xFFFFFFFFL)
        val data = Pb().varint(1, port.toLong()).bytes(2, payload).out()
        val pkt = Pb().fixed32(2, to).varint(3, channel.toLong()).bytes(4, data).fixed32(6, id).varint(9, 3).varint(10, 1).out()
        ops.put(Op.W(Pb().bytes(1, pkt).out()))
        return id
    }

    private val cb = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(g: BluetoothGatt, status: Int, state: Int) {
            if (state == BluetoothProfile.STATE_CONNECTED) {
                say("BLE connected"); g.requestMtu(512)
                main.postDelayed({ if (!discovered && gatt === g) { say("no MTU answer, discovering anyway"); g.discoverServices() } }, 3000)
            } else if (state == BluetoothProfile.STATE_DISCONNECTED) {
                ready = false; say("BLE disconnected (status $status)")
                g.close(); if (gatt === g) gatt = null
                ops.clear(); done.put(WROTE)                            // free a pump stuck waiting on a callback
                val w = wanted
                if (w != null) { say("retrying in 5 s (is the Meshtastic app holding the radio?)"); main.postDelayed({ if (wanted == w && gatt == null) connect(w) }, 5000) }
            }
        }
        override fun onMtuChanged(g: BluetoothGatt, mtu: Int, status: Int) { say("MTU $mtu ($status)"); if (!discovered) g.discoverServices() }
        override fun onServicesDiscovered(g: BluetoothGatt, status: Int) {
            if (discovered) return; discovered = true
            say("services ($status): ${g.services.size}")
            val s = g.getService(SERVICE) ?: run { say("Not a Meshtastic radio (no service)"); return }
            toR = s.getCharacteristic(TORADIO); fromR = s.getCharacteristic(FROMRADIO)
            val num = s.getCharacteristic(FROMNUM)
            val cccd = num?.getDescriptor(CCCD)
            if (toR == null || fromR == null || num == null || cccd == null) {
                say("Not a Meshtastic radio (a required characteristic is missing)"); return
            }
            g.setCharacteristicNotification(num, true)
            val rc = g.writeDescriptor(cccd, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
            say("cccd write rc=$rc")
        }
        override fun onDescriptorWrite(g: BluetoothGatt, d: BluetoothGattDescriptor, status: Int) {
            say("notify on ($status)")
            done.clear(); thread(name = "mesh-ops") { pump() }
            ops.put(Op.W(Pb().varint(3, Random.nextLong(1, 0x7FFFFFFF)).out()))   // want_config_id → radio dumps its state
            ops.put(Op.R)
        }
        override fun onCharacteristicWrite(g: BluetoothGatt, c: BluetoothGattCharacteristic, status: Int) {
            if (status != BluetoothGatt.GATT_SUCCESS) say("write failed ($status)")
            done.put(WROTE)
        }
        override fun onCharacteristicRead(g: BluetoothGatt, c: BluetoothGattCharacteristic, value: ByteArray, status: Int) {
            done.put(value)
        }
        override fun onCharacteristicChanged(g: BluetoothGatt, c: BluetoothGattCharacteristic, value: ByteArray) {
            if (c.uuid == FROMNUM) ops.put(Op.R)
        }
    }

    /** Wait for the GATT callback of the op in flight; null = the radio stopped answering. */
    private fun awaitOp(): ByteArray? = done.poll(OP_TIMEOUT_S, java.util.concurrent.TimeUnit.SECONDS)
        ?: null.also { say("Radio stopped answering (no GATT callback in $OP_TIMEOUT_S s)") }

    private fun pump() {
        val g = gatt ?: return                                          // one pump per connection: it ends when its gatt does
        val tr = toR ?: return; val fr = fromR ?: return
        while (gatt === g) {
            val op = ops.poll(1, java.util.concurrent.TimeUnit.SECONDS) ?: continue
            if (gatt !== g) { ops.put(op); return }
            when (op) {
                is Op.W -> { g.writeCharacteristic(tr, op.b, BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT); awaitOp() }
                is Op.R -> {                                   // drain: read until the radio returns an empty value
                    while (true) {
                        if (!g.readCharacteristic(fr)) break
                        val v = awaitOp() ?: break
                        if (v.isEmpty()) break
                        // A malformed or hostile packet is dropped; it must never take the process down.
                        try { handleFromRadio(v) } catch (e: Exception) { say("Dropped a malformed packet (${e.javaClass.simpleName})") }
                    }
                }
            }
        }
    }

    private fun nodeName(num: Long?, user: ByteArray?) {
        val n = num ?: return; val u = user?.let { Pb.parse(it) } ?: return
        (u[2] as? ByteArray)?.let { names["!%08x".format(n)] = String(it) }
    }

    private fun handleFromRadio(b: ByteArray) {
        val f = Pb.parse(b)
        (f[3] as? ByteArray)?.let { my -> (Pb.parse(my)[1] as? Long)?.let { myNode = it; say("Radio node !%08x".format(myNode)) } }
        (f[4] as? ByteArray)?.let { ni -> Pb.parse(ni).let { n -> nodeName(n[1] as? Long, n[2] as? ByteArray) } }   // node list
        f[7]?.let { if (!ready) { ready = true; say("Radio ready") } }
        if (!ready) return
        val pkt = (f[2] as? ByteArray)?.let { Pb.parse(it) } ?: return
        android.util.Log.d("burst", "pkt fields ${pkt.keys} from=${pkt[1]} decoded=${pkt[4] != null}")
        val dec = (pkt[4] as? ByteArray)?.let { Pb.parse(it) } ?: return   // encrypted-only packets are not for us
        val port = (dec[1] as? Long ?: 0L).toInt()
        if (port == 4) nodeName(pkt[1] as? Long, dec[2] as? ByteArray)   // NODEINFO_APP: a peer announcing its name
        if (port == 5) {                                             // ROUTING_APP: Routing{error_reason=3}, request_id = our packet id
            val rid = dec[6] as? Long ?: return
            acks[rid] = ((dec[2] as? ByteArray)?.let { Pb.parse(it)[3] as? Long } ?: 0L).toInt(); return
        }
        onData(pkt[1] as? Long ?: 0L, pkt[2] as? Long ?: 0L, (pkt[3] as? Long ?: 0L).toInt(), port, dec[2] as? ByteArray ?: ByteArray(0))
    }

    /** Minimal protobuf writer/reader: varint, fixed32, length-delimited. Reader keeps the LAST value per field.
     *  The reader takes peer-supplied bytes: any truncated or out-of-range field throws [Malformed]. */
    class Pb {
        private val o = ByteArrayOutputStream()
        private fun v(x: Long) { var n = x; while (n and 0x7FL.inv() != 0L) { o.write(((n and 0x7F) or 0x80).toInt()); n = n ushr 7 }; o.write(n.toInt()) }
        fun varint(f: Int, x: Long) = apply { v((f shl 3).toLong()); v(x) }
        fun fixed32(f: Int, x: Long) = apply { v(((f shl 3) or 5).toLong()); for (i in 0 until 4) o.write(((x ushr (8 * i)) and 0xFF).toInt()) }
        fun bytes(f: Int, b: ByteArray) = apply { v(((f shl 3) or 2).toLong()); v(b.size.toLong()); o.write(b) }
        fun out(): ByteArray = o.toByteArray()

        companion object {
            class Malformed(msg: String) : RuntimeException(msg)

            fun parse(b: ByteArray): Map<Int, Any> {
                val m = HashMap<Int, Any>(); var i = 0
                fun need(n: Long) { if (n < 0 || n > b.size - i) throw Malformed("field runs past the end") }
                fun rv(): Long {
                    var r = 0L; var s = 0
                    while (true) {
                        if (i >= b.size) throw Malformed("truncated varint")
                        if (s > 63) throw Malformed("varint too long")
                        val c = b[i++].toInt() and 0xFF; r = r or ((c and 0x7F).toLong() shl s)
                        if (c < 0x80) return r; s += 7
                    }
                }
                while (i < b.size) {
                    val tag = rv(); val f = (tag ushr 3).toInt()
                    when ((tag and 7).toInt()) {
                        0 -> m[f] = rv()
                        1 -> { need(8); i += 8 }
                        2 -> { val n = rv(); need(n); m[f] = b.copyOfRange(i, i + n.toInt()); i += n.toInt() }
                        5 -> { need(4); var x = 0L; for (k in 0 until 4) x = x or ((b[i + k].toLong() and 0xFF) shl (8 * k)); m[f] = x; i += 4 }
                        else -> return m
                    }
                }
                return m
            }
        }
    }
}
