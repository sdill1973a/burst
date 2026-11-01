# burst-codec

The codec behind **Burst**: exact text compression for LoRa / Meshtastic radios. Both ends hold the same small
language model (Qwen2.5-0.5B, Q8_0 GGUF), so only the surprise crosses the air. Measured 4.5–6.6× on real messages,
byte-for-byte exact, with a check the receiver can refuse on.

```
pip install burst-codec
burst-codec selftest                      # coder + refusal tests, no model needed
export BURST_GGUF=/path/to/Qwen2.5-0.5B.Q8_0.gguf   # sha256 d5163baa48d1797b1d274f271aa1fd152e3f07248011f7ea386f3d0667dce535
burst-codec selftest --golden             # the golden vectors the Android app checks itself against
```

In Python: `import burst_codec, det_codec` (frame 0x08, deterministic, the default) or `open_codec` (frame 0x03, any GGUF).
The C range coder builds itself on first use (needs a C compiler).

Apache License 2.0. Copyright 2026 Scott Dill. The model is Apache 2.0 (Qwen).
