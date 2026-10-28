package com.gtc.burst.app

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Bring your own AI — optional. Burst's compression never needs it: that runs on the phone, on a generic small model.
 * An account here is used only when you tap AI, sends only the text you chose, over the internet, to the provider you
 * named — never over the radio. Any OpenAI-compatible endpoint works (OpenAI, Anthropic, OpenRouter, Gemini, or your
 * own Ollama / llama.cpp server). Keys are sealed with an Android Keystore key that never leaves the phone.
 */
object Ai {
    data class Account(val name: String, val url: String, val model: String, val key: String)

    val PRESETS = linkedMapOf(
        "OpenAI" to ("https://api.openai.com/v1" to "gpt-4o-mini"),
        "Anthropic" to ("https://api.anthropic.com/v1" to "claude-haiku-5-5"),
        "OpenRouter" to ("https://openrouter.ai/api/v1" to "openrouter/auto"),
        "Gemini" to ("https://generativelanguage.googleapis.com/v1beta/openai" to "gemini-2.5-flash"),
        "My own server" to ("https://your-server/v1" to "llama3.2"),
    )
    val ASKS = linkedMapOf(
        "Summarize" to "Summarize this message in one or two short sentences.",
        "Translate to English" to "Translate this message to English. Reply with the translation only.",
        "Explain" to "Explain what this message means, briefly.",
        "Draft a short reply" to "Draft a short, plain reply to this message. Reply with the draft only.",
    )

    private const val ALIAS = "burst_ai"
    private fun secret(): SecretKey {
        val ks = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (ks.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build())
        }.generateKey()
    }
    private fun seal(s: String): String {
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, secret()) }
        return Base64.encodeToString(c.iv + c.doFinal(s.toByteArray()), Base64.NO_WRAP)
    }
    private fun open(s: String): String {
        val b = Base64.decode(s, Base64.NO_WRAP)
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.DECRYPT_MODE, secret(), GCMParameterSpec(128, b, 0, 12)) }
        return String(c.doFinal(b, 12, b.size - 12))
    }

    private fun prefs(ctx: Context) = ctx.getSharedPreferences("burst_ai", Context.MODE_PRIVATE)
    fun accounts(ctx: Context): List<Account> = try {
        JSONArray(open(prefs(ctx).getString("a", null) ?: return emptyList())).let { a ->
            List(a.length()) { a.getJSONObject(it).run { Account(getString("name"), getString("url"), getString("model"), getString("key")) } }
        }
    } catch (e: Exception) { emptyList() }
    fun save(ctx: Context, list: List<Account>) {
        val a = JSONArray(); list.forEach { a.put(JSONObject().put("name", it.name).put("url", it.url).put("model", it.model).put("key", it.key)) }
        prefs(ctx).edit().putString("a", seal(a.toString())).apply()
    }

    /** Blocks. One chat completion; returns the answer text or throws with the provider's error. */
    fun ask(acct: Account, instruction: String, text: String): String {
        val c = URL(acct.url.trimEnd('/') + "/chat/completions").openConnection() as HttpURLConnection
        c.requestMethod = "POST"; c.doOutput = true; c.connectTimeout = 20_000; c.readTimeout = 120_000
        c.setRequestProperty("Content-Type", "application/json")
        if (acct.key.isNotBlank()) c.setRequestProperty("Authorization", "Bearer ${acct.key}")
        val body = JSONObject().put("model", acct.model).put("max_tokens", 600).put("messages", JSONArray()
            .put(JSONObject().put("role", "system").put("content", "$instruction Keep it short: it may be sent over a low-bandwidth radio."))
            .put(JSONObject().put("role", "user").put("content", text)))
        c.outputStream.use { it.write(body.toString().toByteArray()) }
        val ok = c.responseCode in 200..299
        val resp = (if (ok) c.inputStream else c.errorStream)?.bufferedReader()?.readText() ?: ""
        if (!ok) throw RuntimeException("${c.responseCode}: ${resp.take(300)}")
        return JSONObject(resp).getJSONArray("choices").getJSONObject(0).getJSONObject("message").getString("content").trim()
    }
}
