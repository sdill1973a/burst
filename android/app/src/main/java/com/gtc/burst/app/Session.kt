package com.gtc.burst.app

import com.gtc.burst.Chain
import com.gtc.burst.Codec
import com.gtc.burst.MeshBle

/** What lives as long as the process, not the screen: the radio link, the model, the chain assembler and the log.
 *  Android can rebuild the activity after a settings change; if the radio link lived in the activity, the old link would
 *  stay open with nobody showing its messages while the new one opened a second link, and messages would be lost. */
object Session {
    var codec: Codec? = null
    var asm2: Chain.Assembler2? = null
    val w2Watch = java.util.concurrent.atomic.AtomicBoolean(false)
    /** review #18: burst frames that arrived before the model loaded (MainActivity drains it once codec is set). */
    val early = ArrayList<Any>()
    var mesh: MeshBle? = null
    var mqtt: com.gtc.burst.Mqtt? = null             // the radio's broker link, carried by the phone (proxy to client)
    var asm: Chain.Assembler? = null
    val asked = HashSet<String>()
    val log = StringBuilder()
    var sink: ((String) -> Unit)? = null           // the visible activity's log view, if any
    @Volatile var visible = false                    // the chat screen is in front: received messages need no notice
    var reply: ((Long, String, Int) -> Boolean)? = null
    /** Received packets are decoded here, one at a time in arrival order, off the BLE thread. */
    val decode: java.util.concurrent.ExecutorService = java.util.concurrent.Executors.newSingleThreadExecutor { r -> Thread(r, "burst-decode") }   // send path for a notification reply: (dest, text, channel) → delivered

    /** Typing dots (frame 0x09): a peer sends one only when it has actually started composing a reply; the dots clear
     *  when anything else arrives from that peer, or after 3 min. */
    private val typingUntil = HashMap<String, Long>()
    @Synchronized fun typingFrom(peer: String) { typingUntil[peer] = System.currentTimeMillis() + 180_000 }
    @Synchronized fun typingDone(peer: String) { typingUntil.remove(peer) }
    @Synchronized fun typing(peer: String) = (typingUntil[peer] ?: 0) > System.currentTimeMillis()
}
