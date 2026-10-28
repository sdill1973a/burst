package com.gtc.burst.app

import android.app.RemoteInput
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Reply typed into a message notice: hand it to the live session's send path (same codec, history and ACK-gated
 *  links as the Send button), then show it in the notice. */
class ReplyReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        val text = RemoteInput.getResultsFromIntent(intent)?.getCharSequence(BurstService.REPLY_KEY)?.toString()?.trim()
        val peer = intent.getStringExtra("peer") ?: return
        val ch = intent.getIntExtra("ch", 0)
        if (text.isNullOrEmpty()) return
        val send = Session.reply
        if (send == null) { Chat.add(ctx, peer, true, text, "not delivered"); BurstService.refresh(ctx, peer, ch); return }
        val done = goAsync()
        kotlin.concurrent.thread {
            runCatching { send(peer.removePrefix("!").toLong(16), text, ch) }      // sendText puts it in the chat
            BurstService.refresh(ctx.applicationContext, peer, ch)
            done.finish()
        }
    }
}
