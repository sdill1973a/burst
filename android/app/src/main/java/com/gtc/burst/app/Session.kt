package com.gtc.burst.app

import com.gtc.burst.Chain
import com.gtc.burst.Codec
import com.gtc.burst.MeshBle

/** What lives as long as the process, not the screen: the radio link, the model, the chain assembler and the log.
 *  Born 10-24: Android rebuilt the activity after a settings change; the old one's radio link stayed open with nobody
 *  showing its messages and the new one opened a second link, so a message landed on the orphan and was never seen. */
object Session {
    var codec: Codec? = null
    var mesh: MeshBle? = null
    var asm: Chain.Assembler? = null
    val asked = HashSet<String>()
    val log = StringBuilder()
    var sink: ((String) -> Unit)? = null           // the visible activity's log view, if any
    @Volatile var visible = false                    // the chat screen is in front: received messages need no notice
    var reply: ((Long, String, Int) -> Boolean)? = null   // send path for a notification reply: (dest, text, channel) → delivered

    /** Typing dots (frame 0x09), his 10-24: "they have to be real". The box sends one only when my seat has actually
     *  started a turn on his message; the dots clear when anything else arrives from that peer, or after 3 min. */
    private val typingUntil = HashMap<String, Long>()
    @Synchronized fun typingFrom(peer: String) { typingUntil[peer] = System.currentTimeMillis() + 180_000 }
    @Synchronized fun typingDone(peer: String) { typingUntil.remove(peer) }
    @Synchronized fun typing(peer: String) = (typingUntil[peer] ?: 0) > System.currentTimeMillis()
}
