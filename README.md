# Burst

Type a normal message. It goes over Meshtastic in a fraction of the bytes. It arrives exactly as typed, or the far end
refuses it and asks again. It never shows a wrong guess.

An Android app that talks to your Meshtastic radio over Bluetooth, plus the codec source. It sends on the PRIVATE_APP
port (256), so stock firmware carries it like any other data.

## Numbers

Real bytes on the air, headers and checks included. "Exact" means byte-for-byte equal to what was typed.

| Text | Burst | Burst + conversation history | Unishox2 (what firmware uses now) | zstd |
|---|---|---|---|---|
| 239 real typed messages, one user (avg 359 chars) | 4.42× | **5.93×** | 1.67× | 1.47× |
| 200 public SMS from strangers (avg 74 chars) | 3.19× | — (no shared history) | 1.55× | 1.06× |

Every message exact, every arm. The SMS row is 200 seeded messages from the public UCI SMS Spam Collection (ham only). Burst was never tuned on it, though public 2012 text may sit in the model's training data. On very
short messages (under about 10 characters) Unishox2 wins by 1–3 bytes: Burst carries a 2-byte check and a small header.

For reference, Fabrice Bellard's ts_sms (RWKV 169M, x86) scores 4.52× and 3.71× on the same two sets. It is
the best cold result on short texts, but it has no history and doesn't run on a phone.

## How it works

Both ends run the same **generic small language model**: Qwen2.5-0.5B-Instruct, q8_0, 676 MB. It's the same file on every
phone, sha256-pinned, and has nothing personal in it. The model predicts each next token of your message, and only the
surprise goes on the air, range-coded. The engine that computes the predictions (`codec/detlm.c`) is plain integer and
scalar C. It gives bit-identical results on any CPU, so the far end rebuilds the exact same predictions and the exact same
text.

- **Exact or refused.** Every frame carries 2 bytes of the text's hash. A decode that doesn't match is refused, never
  shown. The receiver then asks (`0x06`) for a resend without shared history.
- **It learns the conversation.** The last 4 messages both ends confirmed (received with a passing check, or sent and
  ACKed by the peer's radio) prime the model. Replies get smaller as you talk.
- **Long text** splits into a chain of up to 16 packets. A missing link is asked for and resent.
- **Compression is not encryption.** Your radio does that. Send bursts as direct messages, or on a private channel.
  Never broadcast them on the default channel, whose key is public.

## The app

- First run downloads the model once from Hugging Face (676 MB, Wi-Fi recommended), checks its sha256, and resumes if
  interrupted. After that, nothing leaves the phone but the radio.
- Peer names come from your mesh's node list.
- **Bring your own AI (optional).** Behind the gear, add any OpenAI-compatible account: OpenAI, Anthropic, OpenRouter,
  Gemini, or your own Ollama / llama.cpp server. An AI button then appears. It can summarize, translate, explain, or draft
  a reply to a message. It's used only when you tap it, and it sends only the text you picked, over the internet, to the
  provider you named. Never over the radio. Keys are sealed by an Android Keystore key that never leaves the phone.
  Compression never needs this.

Build: Android Studio or `cd android && ./gradlew assembleRelease -PburstAbis=arm64-v8a`. Needs NDK 29, plus llama.cpp
at commit `9a286ac` (a git submodule: `git submodule update --init`).

## Codec on a computer

`codec/` holds the C engine (`make`) and the Python codec (`det_codec.py`, with llama-cpp-python for the tokenizer).
`test_det_codec.py` runs the golden vectors the app checks itself against.

## Not yet

Proven on the air between one phone (Galaxy S25) and one computer, each on its own radio. A second, unrelated phone is next. iOS isn't started. A smaller model
is not planned, because exactness needs both ends on the same file.

## License

LICENSE_LINE
