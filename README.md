# Burst

Type a normal message. It goes over Meshtastic in a fraction of the bytes. It arrives exactly as typed, or the far end
refuses it and asks again.

An Android app that talks to your Meshtastic radio over Bluetooth, plus the codec source. It sends on the PRIVATE_APP
port (256), so stock firmware carries it like any other data.

## Why

**Most of any message is already known.** In a world of AI, the public record of human history, its books and
documents and how people talk, is common knowledge. A small language model carries a compressed copy of it, and anyone
can have the same one. That shared copy is ground both ends of a conversation already stand on.

**Common knowledge doesn't need to travel. Change does.** Courts don't make anyone prove that water runs downhill.
People don't re-explain the world each time they talk; they assume what's shared and say what's different.
Communication at its best is a stream of deltas of change against a world both sides already hold.

**Radios should work that way.** A Meshtastic message is about 200 characters on a slow, shared channel, and those bytes
matter most where there's nothing else: off-grid, in a disaster. Spending them on what both ends already know wastes the
scarcest thing there is.

**So Burst shares the ground first and sends only the change.** It starts with the common record, held in the model.
Then a pack on the subject, if both sides load one. Then what this conversation has already said. Only the surprise goes
on the air, and it arrives exact, or it's refused.

**It should belong to everyone.** Common knowledge shouldn't be owned, so neither should the tool that uses it. It runs
on an ordinary phone, needs no server or account, and is open source.

## The idea

Shared ground comes in three layers, and each one makes messages smaller:

1. **The common record.** The same generic model on every phone. Nothing personal in it.
2. **A pack on the subject.** A page of text both ends load ahead of time: how weather is reported, a search team's
   protocol, a farm's field names. It's identified by its fingerprint, so both sides know they hold the same one.
3. **The conversation.** The last few messages both ends confirmed.

**The model's copy is blurry.** It predicts the record; it can't recite it. So Burst never trusts a guess. Every
message carries a check, and a decode that doesn't match exactly is refused and asked for again.

**Not just radios.** Anything that has to say a lot through a narrow pipe can work this way, as long as both ends hold
the same model and the same packs.

## Numbers

Real bytes on the air, every Burst header and check included. "Exact" means byte-for-byte equal to what was typed.

**Reproduce these yourself** — the codec is deterministic, so you get the same bytes (same frames sha256) we did.
From `codec/` after `make -f Makefile.detlm`, with the model below and Project Gutenberg's *Pride and Prejudice*
(`pg1342.txt`) and the UCI SMS Spam Collection (`SMSSpamCollection`, inside the zip at
https://archive.ics.uci.edu/dataset/228/sms+spam+collection):

| Public text | chars | packets | bytes | ratio | command |
|---|---|---|---|---|---|
| *Pride and Prejudice*, opening, pasted as one message | 2,449 | 8 | 370 | **6.62×** | `python repro.py paste pg1342.txt --start "It is a truth universally acknowledged" --flow --chars 2450` |
| *Pride and Prejudice*, all of chapter 1 | 4,712 | 16 | 761 | **6.19×** | `python repro.py paste pg1342.txt --chapter 1 --flow` |
| 200 SMS from strangers, one at a time, no history (avg 74 chars) | 14,863 | 200 | 4,582 | **3.24×** | `python repro.py lines SMSSpamCollection --sms-seed 20261024 --n 200` |
| 140 National Weather Service forecasts (10 state capitals, 2026-10-28), one at a time, no pack (avg 86 chars) | 12,072 | 140 | 2,371 | **5.09×** | `python repro.py lines packs/test/nws_2026-10-28.txt` |
| The same 140, with a one-page weather pack on both ends | 12,072 | 140 | 1,560 | **7.74×** | `python repro.py lines packs/test/nws_2026-10-28.txt --pack packs/nws_forecast_style.txt` |

The first row was also sent **on the air** from a fresh install of this app to a second radio: 8 packets, 368 bytes,
exact (6.62×). Every number above decoded back exact. Burst was never tuned on any of these texts. A famous book is part of the
common record the model learned from, so it costs little to send: that's the idea working, not a trick. The weather pack
(`codec/packs/nws_forecast_style.txt`) was written and committed before any test forecast was fetched.

With history the numbers go up, because the model is primed on your recent conversation. One user's own 239 typed
messages (private, so not reproducible here) measured 5.93× with history against 4.42× without, on an earlier model.
For comparison on those same messages: Unishox2 (what firmware uses now) 1.67×, zstd 1.47×, Fabrice Bellard's ts_sms
(RWKV 169M, x86, no history, not on a phone) 4.52×. On very short messages (under about 10 characters) Unishox2 wins by
1–3 bytes: Burst carries a 2-byte check and a small header.

## Use it

**Both people need Burst.** Bursts travel as data on port 256. The stock Meshtastic app doesn't show them, and a phone
without Burst sees nothing.

1. **Install.** Download `burst-0.1.apk` from this repo's Releases page and open it on an Android phone (arm64, Android 13+).
   Android will ask you to allow installs from your browser or file manager.
2. **Pair your radio.** Pair your Meshtastic radio in Android's Bluetooth settings first, the same way the Meshtastic app
   needs. Close the Meshtastic app: only one app can hold the radio's Bluetooth link at a time.
3. **Open Burst.** The first run downloads the model (531 MB, once; use Wi-Fi). Allow Bluetooth and notifications when asked.
4. **Connect.** Tap **Connect** and pick your radio from the list of paired devices. Wait for *Radio ready*. Burst
   remembers the radio and reconnects to it next time.
5. **Send.** In *To*, put the other person's node id, like `!1a2b3c4d` (it's in the Meshtastic app's node list). Set *Ch*
   to a private channel if you use one. Type, then tap **Send**. The log shows how many characters went out and how
   many bytes that took.
6. **Receive.** Incoming bursts appear in the chat and as a notification you can reply to without opening the app.

Messages to `all` on channel 0 go out on the public default channel, where anyone can read them. Use a direct message or
a private channel for anything you wouldn't post on a board.

## How it works

Both ends run the same **generic small language model**: Qwen2.5-0.5B (base), q8_0, 531 MB. It's the same file on every
phone, sha256-pinned, and has nothing personal in it. The model predicts each next token of your message, and only the
surprise goes on the air, range-coded. The engine that computes the predictions (`codec/detlm.c`) is plain integer and
scalar C. It gives bit-identical results on any CPU, so the far end rebuilds the exact same predictions and the exact same
text.

- **Exact or refused.** Every frame carries 2 bytes of the text's hash. A decode that doesn't match is refused, never
  shown. The receiver then asks (`0x06`) for a resend without shared history. A 16-bit check is not proof: a corrupted
  or mismatched decode passes it by chance about 1 time in 65,536 per candidate history tried (up to 7), so roughly 1 in
  9,000 at worst for a frame that decodes wrong.
- **It learns the conversation.** The last 4 messages both ends confirmed (received with a passing check, or sent and
  ACKed by the peer's radio) prime the model. Replies get smaller as you talk.
- **Long text** splits into a chain of up to 16 packets. A missing link is asked for and resent.
- **Compression is not encryption.** Your radio does that. Send bursts as direct messages, or on a private channel.
  Never broadcast them on the default channel, whose key is public.

## The app

- First run downloads the model once from Hugging Face (531 MB, Wi-Fi recommended), checks its sha256, and resumes if
  interrupted. After that, nothing leaves the phone but the radio.
- Peer names come from your mesh's node list.
- **Bring your own AI (optional).** Behind the gear, add any OpenAI-compatible account: OpenAI, Anthropic, OpenRouter,
  Gemini, or your own Ollama / llama.cpp server. An AI button then appears. It can summarize, translate, explain, or draft
  a reply to a message. It's used only when you tap it, and it sends only the text you picked, over the internet, to the
  provider you named. Never over the radio. Keys are sealed by an Android Keystore key that never leaves the phone.
  Compression never needs this.

Build: Android Studio or `cd android && ./gradlew assembleRelease -PburstAbis=arm64-v8a`. Needs NDK 29
(29.0.13113456), plus llama.cpp at commit `9a286ac` (a git submodule: `git submodule update --init`).

## Run the codec

`codec/` holds the C engine (`detlm.c`, `detrc.c`) and the Python codec (`det_codec.py`; llama-cpp-python supplies only
the tokenizer). Tested on Linux x86-64 with gcc and Python 3.12:

```sh
cd codec
pip install -r requirements.txt     # numpy + llama-cpp-python (builds from source: needs cmake and a C++ compiler)
make -f Makefile.detlm              # libdetlm.so, detlm_x86, detcodec_x86 (host only; `make -f Makefile.detlm arm` needs the NDK)
curl -LO https://huggingface.co/mradermacher/Qwen2.5-0.5B-GGUF/resolve/c20fdc01c83cad0e38968fb98bc5476042d1c8db/Qwen2.5-0.5B.Q8_0.gguf
sha256sum Qwen2.5-0.5B.Q8_0.gguf    # must be d5163baa48d1797b1d274f271aa1fd152e3f07248011f7ea386f3d0667dce535
export BURST_GGUF=$PWD/Qwen2.5-0.5B.Q8_0.gguf
python3 test_det_codec.py           # coder + refusal tests (numpy only, no model)
python3 test_det_codec.py --golden  # the golden vectors the app checks itself against, on the real model
curl -LO https://www.gutenberg.org/cache/epub/1342/pg1342.txt
python3 repro.py paste pg1342.txt --start "It is a truth universally acknowledged" --flow --chars 2450
```

The last command is the table's first row; it should print `2449 chars -> 8 packet(s), 370 B on air, 6.619x, exact`
and frames sha256 `97c880349f69abc562aabaa1ff9948505e077ae10c12d7251f01d3c883e44ad1`.

`make` (plain) builds the range coder's cross-architecture harness; `make identity` (with the NDK and qemu-aarch64)
checks x86 and aarch64 produce identical tables and bytes, and that its negative controls differ.

## Not yet

Proven on the air between one phone (Galaxy S25) and one computer, each on its own radio. A second, unrelated phone is next. iOS isn't started. Packs work in the codec (`repro.py --pack`) but the app can't load one yet. A smaller model
is not planned, because exactness needs both ends on the same file.

## Third-party

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT, a git submodule (the app's tokenizer and the 0x03 codec).
- Qwen2.5-0.5B — Apache 2.0, Alibaba Cloud. Downloaded at first run, not bundled.
- GGUF conversion: [mradermacher/Qwen2.5-0.5B-GGUF](https://huggingface.co/mradermacher/Qwen2.5-0.5B-GGUF).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 Scott Dill. The bundled llama.cpp submodule keeps its own MIT license; the model the app downloads (Qwen2.5-0.5B) is Apache 2.0.
