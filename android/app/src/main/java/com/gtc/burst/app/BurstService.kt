package com.gtc.burst.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager

/** Keeps the process (and the radio link the activity holds) alive with the screen off or the app in the background.
 *  Born 10-24: before it, Burst only heard the mesh while it was on screen. Started on Connect; the ongoing
 *  notification is Android's price for that, and tapping it brings Burst back. Messages arrive as their own notices. */
class BurstService : Service() {
    private var lock: PowerManager.WakeLock? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        channels(this)
        val n = Notification.Builder(this, HOLD).setSmallIcon(R.mipmap.ic_launcher_round)
            .setContentTitle("Burst is listening").setOngoing(true)
            .setContentIntent(open(this)).build()
        // 10-27, fresh-phone test: Android 14+ refuses a connectedDevice service until Bluetooth is granted, and the
        // first-run model download starts this service before the user has answered that prompt. That crashed every new
        // install. So: connectedDevice once Bluetooth is granted, dataSync (the download) before; never crash on refusal.
        val bt = checkSelfPermission(android.Manifest.permission.BLUETOOTH_CONNECT) == android.content.pm.PackageManager.PERMISSION_GRANTED
        try {
            startForeground(1, n, if (bt) ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE else ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } catch (e: Exception) { stopSelf(); return START_NOT_STICKY }
        if (lock == null) lock = (getSystemService(POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "burst:radio").also { it.acquire() }
        return START_STICKY
    }

    override fun onDestroy() { lock?.release(); lock = null; super.onDestroy() }
    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        private const val HOLD = "listening";   // MIN: folded to the bottom of the shade, no status-bar icon (10-24)
        private const val MSG = "texts"   // v2 10-24: a channel's sound/vibration can't change after creation, so a new id

        private fun channels(ctx: Context) {
            val nm = ctx.getSystemService(NotificationManager::class.java)
            nm.deleteNotificationChannel("hold")
            nm.createNotificationChannel(NotificationChannel(HOLD, "Listening", NotificationManager.IMPORTANCE_MIN))
            nm.deleteNotificationChannel("messages")
            nm.createNotificationChannel(NotificationChannel(MSG, "Messages", NotificationManager.IMPORTANCE_HIGH).apply {
                enableVibration(true); vibrationPattern = longArrayOf(0, 250, 120, 250)
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            })
        }

        private fun open(ctx: Context) = PendingIntent.getActivity(ctx, 0,
            Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT),
            PendingIntent.FLAG_IMMUTABLE)

        const val REPLY_KEY = "reply"
        private val me = android.app.Person.Builder().setName("Me").build()

        /** A message they sent me: it joins the chat, and one conversation notice per peer shows the tail with an inline
         *  Reply. Alerts and lights the screen. */
        fun notify(ctx: Context, from: String, text: String, ch: Int = 0) { Chat.add(ctx, from, false, text); if (!Session.visible) show(ctx, from, ch, alert = true) }

        /** Redraw the notice quietly after I sent from it (Android drops the reply spinner when the notice updates). */
        fun refresh(ctx: Context, peer: String, ch: Int) = show(ctx, peer, ch, alert = false)

        private fun show(ctx: Context, peer: String, ch: Int, alert: Boolean) {
            channels(ctx)
            val name = com.gtc.burst.MeshBle.name(peer)
            val them = android.app.Person.Builder().setName(name).build()
            val style = Notification.MessagingStyle(me).setConversationTitle(name)
            Chat.get(ctx, peer).takeLast(8).forEach { l ->
                style.addMessage(if (l.status == "not delivered") "${l.text}  (not delivered)" else l.text, l.ts, if (l.mine) null else them)
            }
            val reply = PendingIntent.getBroadcast(ctx, peer.hashCode(),
                Intent(ctx, ReplyReceiver::class.java).putExtra("peer", peer).putExtra("ch", ch),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE)
            val action = Notification.Action.Builder(android.graphics.drawable.Icon.createWithResource(ctx, R.mipmap.ic_launcher_round), "Reply", reply)
                .addRemoteInput(android.app.RemoteInput.Builder(REPLY_KEY).setLabel("Reply to $name").build())
                .setAllowGeneratedReplies(false).build()
            ctx.getSystemService(NotificationManager::class.java).notify(peer.hashCode(),
                Notification.Builder(ctx, MSG).setSmallIcon(R.mipmap.ic_launcher_round).setContentTitle(name)
                    .setVisibility(Notification.VISIBILITY_PUBLIC).setCategory(Notification.CATEGORY_MESSAGE)
                    .setStyle(style).addAction(action).setOnlyAlertOnce(!alert)
                    .setAutoCancel(true).setContentIntent(open(ctx)).build())
            if (!alert) return
            // A dark locked screen only gets an always-on-display icon; light it for 6 s so the lock screen shows the card
            //. The flag is deprecated but still honoured; it never unlocks anything.
            @Suppress("DEPRECATION")
            (ctx.getSystemService(POWER_SERVICE) as PowerManager).newWakeLock(
                PowerManager.SCREEN_BRIGHT_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP, "burst:wake").acquire(6000)
        }
    }
}
