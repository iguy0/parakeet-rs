# parakeet-rs
[![Rust](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/parakeet-rs.svg)](https://crates.io/crates/parakeet-rs)

Fast speech recognition with NVIDIA's Parakeet models via ONNX Runtime.

Note: CoreML is unstable with this model. For Apple, use WebGPU EP (uses metal under the hood,dont confuse by its name :-). it's a native GPU standard, not only web) or CPU. But even CPU alone is significantly faster on my Mac M3 16GB compared to Whisper metal! :-)

## Models

### Streaming vs batch/offline

| Model | Mode | State across calls |
|---|---|---|
| EOU | Streaming | Persistent encoder/decoder cache; feed fixed-size chunks in order |
| Nemotron | Streaming | Persistent encoder cache + pre-encode buffer; feed fixed-size chunks in order |
| Multitalker | Streaming | Persistent encoder cache + Sortformer speaker state; feed fixed-size chunks in order |
| Sortformer | Streaming (`diarize_chunk`) or offline (`diarize`) | `diarize_chunk` preserves FIFO/speaker-cache state across calls; `diarize` runs a full file in one call |
| Unified | Hybrid | `transcribe_audio`/`transcribe_file` (offline, no chunking) **or** `transcribe_chunk` + `flush` (buffered streaming with persisted cache) — same struct, `reset()` clears state |
| CTC | Offline/batch | Stateless; one call per clip (~4-5 min limit) |
| TDT | Offline/batch | Stateless; one call per clip (~4-5 min limit) |
| Cohere Transcribe | Offline/batch | Stateless; one call per clip. Long-form audio just means the caller splits it and calls `transcribe_audio` repeatedly — no cache carries over between those calls |
| Ark ASR | Offline/batch | Stateless; one call per clip, fixed-length KV cache sized by the model manifest, greedy-only |

"Streaming" here means the model keeps encoder/decoder state (`*Cache`/`*Handle`) alive between calls so you can feed real-time audio chunk-by-chunk. "Offline/batch" means each call is independent — splitting long audio into pieces is just repeated one-shot calls, not shared streaming state.

**CTC (English-only)**:
```rust
use parakeet_rs::{Parakeet, Transcriber, TimestampMode};

let mut parakeet = Parakeet::from_pretrained(".", None)?;

// Load and transcribe audio (see examples/raw.rs for full example)
let result = parakeet.transcribe_samples(audio, 1600, 1, Some(TimestampMode::Words))?;
println!("{}", result.text);

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**TDT (Multilingual)**: 25 languages with auto-detection
```rust
use parakeet_rs::{ParakeetTDT, Transcriber, TimestampMode};

let mut parakeet = ParakeetTDT::from_pretrained("./tdt", None)?;
let result = parakeet.transcribe_samples(audio, 16000, 1, Some(TimestampMode::Sentences))?;
println!("{}", result.text);

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**EOU (Streaming)**: Real-time ASR with end-of-utterance detection
```rust
use parakeet_rs::ParakeetEOU;

let mut parakeet = ParakeetEOU::from_pretrained("./eou", None)?;

// Prepare your audio (Vec<f32>, 16kHz mono, normalized)
let audio: Vec<f32> = /* your audio samples */;

// Process in 160ms chunks for streaming
const CHUNK_SIZE: usize = 2560; // 160ms at 16kHz
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = parakeet.transcribe(chunk, false)?;
    print!("{}", text);
}
```

**Nemotron (Streaming)**: Cache-aware streaming ASR with punctuation. Two variants share the same API — point `from_pretrained` at whatever directory holds the ONNX files and the loader auto-detects which variant it is:

- **English-only 0.6B** — verbatim English, preserves disfluencies (`um`, `uh`). Best for transcription where every spoken word matters.
- **Multilingual 3.5 0.6B** — [40 language-locales](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) across 3 tiers (19 transcription-ready, 13 broad-coverage, 8 that need fine-tuning to reach production quality based on the NVIDIA). Polished output (proper casing/punctuation, drops disfluencies). Same speed and size as the English-only model.

```rust
use parakeet_rs::{Nemotron, NemotronMode};

let mut model = Nemotron::from_pretrained(path, None)?;

// Multilingual variant: optionally pick a target language. Defaults to "auto"
// (the model picks the language itself). Pass a specific code when you know
// the language it's strictly more accurate. No-op when an English-only model is loaded.
if model.mode() == NemotronMode::Multilingual {
    model.set_target_lang("es-ES")?; // also: "ja-JP", "tr-TR", "auto", ...
}

// Process in 560ms chunks for streaming
const CHUNK_SIZE: usize = 8960; // 560ms at 16kHz
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = model.transcribe_chunk(chunk)?;
    print!("{}", text);
}
```

**Cohere Transcribe (Offline Multilingual)**: 14 languages, punctuation & ITN toggles (yes, "parakeets🦜" talk about more than just NVIDIA right?? :-P)
```toml
parakeet-rs = { version = "0.3", features = ["cohere"] }
```
```rust
use parakeet_rs::CohereASR;

let mut model = CohereASR::from_pretrained("./cohere", None)?;

// audio: Vec<f32>, 16kHz mono (long-form supported)
let text = model.transcribe_audio(&audio, "en", true, false)?; // lang, pnc, itn
println!("{}", text);
```
See `examples/cohere.rs` for a runnable demo.

**Multitalker (Streaming Multi-Speaker ASR)**: Speaker-attributed transcription
```toml
parakeet-rs = { version = "0.3", features = ["multitalker"] }
```
```rust
use parakeet_rs::MultitalkerASR;

let mut model = MultitalkerASR::from_pretrained(
    "./multitalker",             // encoder, decoder, tokenizer
    "sortformer.onnx",           // Sortformer v2 for diarization
    None,
)?;

for chunk in audio.chunks(17920) {  // ~1.12s at 16kHz
    let results = model.transcribe_chunk(chunk)?;
    for r in &results {
        println!("[Speaker {}] {}", r.speaker_id, r.text);
    }
}
```
See `examples/multitalker.rs` for full usage with latency modes.

**Sortformer v2 & v2.1 (Speaker Diarization)**: Streaming diarization (4 speakers by default; 8-speaker Ultra exports supported via ONNX `num_speakers` metadata)
```toml
parakeet-rs = { version = "0.3", features = ["sortformer"] }
```
```rust
use parakeet_rs::sortformer::{Sortformer, DiarizationConfig};

let mut sortformer = Sortformer::with_config(
    "diar_streaming_sortformer_4spk-v2.onnx", // or v2.1.onnx
    None,
    DiarizationConfig::callhome(),  // or dihard3(),custom()
)?;
let segments = sortformer.diarize(audio, 16000, 1)?;
for seg in segments {
    println!("Speaker {} [{:.2}s - {:.2}s]", seg.speaker_id,
        seg.start as f64 / 16_000.0, seg.end as f64 / 16_000.0);
}

// 8-speaker Ultra Sortformer: pass ONNX with num_speakers=8 metadata
// println!("{}", sortformer.num_speakers()); // 8

// For streaming/real-time use, diarize_chunk() preserves state across calls:
let segments = sortformer.diarize_chunk(&audio_chunk_16k_mono)?;
```
See `examples/diarization.rs` for combining with TDT transcription.

See `examples/streaming_diarization.rs` for `diarize_chunk` usage example.

See `scripts/export_diar_sortformer.py` for exporting the model with custom streaming parameters.

**Ark ASR (Offline speech-LLM)**: Whisper-style audio encoder + Qwen2 causal LM, greedy decoding only
```toml
parakeet-rs = { version = "0.3", features = ["ark-asr"] }
```
```rust
use parakeet_rs::{ArkAsr, Transcriber};

let mut model = ArkAsr::from_pretrained("./ark-asr-0.6b-int8-onnx", None)?;
model.set_max_new_tokens(256); // optional, default 256

// audio: Vec<f32>, 16kHz mono, up to max_audio_seconds() (default 30s)
let text = model.transcribe_audio(&audio)?;
println!("{}", text);

// or via the Transcriber trait:
let result = model.transcribe_file("audio.wav", None)?;
println!("{}", result.text);
```
See `examples/ark_asr.rs` for a runnable demo.

## Setup

Weights are downloaded separately. Put every file for one model in the same directory. Each row's first link is the ONNX bundle; the second is the source checkpoint when it lives in another repo.

| Model | ONNX | Source |
|---|---|---|
| CTC 0.6B | [onnx-community/parakeet-ctc-0.6b-ONNX](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX) | [nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) |
| Granite Speech 5.0 470M TurboCTC | [qwertz92/granite-speech-5.0-470m-turboctc-onnx](https://huggingface.co/qwertz92/granite-speech-5.0-470m-turboctc-onnx) | [ibm-granite/granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) |
| TDT 0.6B v3 (fp32 and INT8) | [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Fluid Speech Pro (TDT, verbatim and clean INT8) | [FluidVoice-Labs/fluid-speech-pro-onnx](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-onnx) | [FluidVoice-Labs/fluid-speech-pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) |
| EOU 120M v1 | [altunenes/parakeet-rs `realtime_eou_120m-v1-onnx`](https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx) | [nvidia/parakeet_realtime_eou_120m-v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) |
| Nemotron EN 0.6B fp32 | [altunenes/parakeet-rs `nemotron-speech-streaming-en-0.6b`](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b) | [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) |
| Nemotron EN 0.6B INT8, 560 ms | [investguy/nemotron-speech-streaming-en-0.6b-int8](https://huggingface.co/investguy/nemotron-speech-streaming-en-0.6b-int8) | same NVIDIA checkpoint |
| Nemotron EN 0.6B INT8 / INT4 | [lokkju INT8](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8), [lokkju INT4](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int4) | same NVIDIA checkpoint |
| Nemotron 3.5 multilingual 0.6B | [altunenes/parakeet-rs `nemotron-3.5-asr-streaming-0.6b-onnx`](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3.5-asr-streaming-0.6b-onnx) | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |
| Unified EN 0.6B | [bobNight/parakeet-unified-en-0.6b-onnx](https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx) | [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) |
| Multitalker 0.6B v1 | [smcleod INT8](https://huggingface.co/smcleod/multitalker-parakeet-streaming-0.6b-v1-onnx-int8) | [nvidia/multitalker-parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) |
| Cohere Transcribe 03-2026 | [onnx-community/cohere-transcribe-03-2026-ONNX](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX) | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) |
| Sortformer 4spk v2 and v2.1 | [v2](https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.onnx), [v2.1](https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.1.onnx) | [v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2), [v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) |
| Sortformer Ultra 8spk v1 | [investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx](https://huggingface.co/investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx) | [devsy0117/ultra_diar_streaming_sortformer_8spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1) |
| Nemotron-3 Diarization (offline, low, verylow, ultralow) | [investguy/nemotron3-diarization-onnx](https://huggingface.co/investguy/nemotron3-diarization-onnx) | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) |
| Ark ASR 0.6B INT8 | [Audio8/ark-asr-0.6b-int8-onnx](https://huggingface.co/Audio8/ark-asr-0.6b-int8-onnx) | same repo |
| oruk-fourier-2b INT4 | [investguy/oruk-fourier-2b-int4-onnx](https://huggingface.co/investguy/oruk-fourier-2b-int4-onnx) | [oruk/oruk-fourier-2b](https://huggingface.co/oruk/oruk-fourier-2b) |
| oruk-fourier-2b mixed INT4/INT8 | [investguy/oruk-fourier-2b-int4-mixed-onnx](https://huggingface.co/investguy/oruk-fourier-2b-int4-mixed-onnx) | [oruk/oruk-fourier-2b](https://huggingface.co/oruk/oruk-fourier-2b) |

Granite TurboCTC needs one metadata stamp before `Parakeet::from_pretrained`: `python scripts/patch_ctc_onnx_metadata.py --input onnx/model.onnx --in-place --preset granite-turbo-ctc`.

Fluid Speech Pro loads through the TDT API. Clean mode is the `*.clean.int8.onnx` graphs in that repo, renamed to `encoder-model.int8.onnx` and `decoder_joint-model.int8.onnx`. The license is CC BY 4.0.

Nemotron English INT8 above is the 560 ms export (`--right-context 6`). The 80 ms, 160 ms, and 1120 ms graphs are not published. Multitalker also needs a Sortformer ONNX for diarization. Nemotron-3 diarization files are one latency preset each; start post-processing from `DiarizationConfig::nemotron3()`.

The oruk-fourier-2b bundles are ONNX only. This crate has no loader for them yet. The source model is CC BY-NC 4.0 (its base is [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)); commercial and hosted use need Oruk's licence. ONNX Runtime 1.20 or newer is required (`GatherBlockQuantized`).

GPU support (auto-falls back to CPU if fails):
```toml
parakeet-rs = { version = "0.3", features = ["cuda"] }  # or tensorrt, webgpu, directml, migraphx or other ort supported EPs (check cargo features)
```

```rust
use parakeet_rs::{Parakeet, ExecutionConfig, ExecutionProvider};

let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);
let mut parakeet = Parakeet::from_pretrained(".", Some(config))?;
```

Advanced session configuration via [ort SessionBuilder](https://docs.rs/ort/latest/ort/session/builder/struct.SessionBuilder.html):
```rust
let config = ExecutionConfig::new()
    .with_custom_configure(|builder| builder.with_memory_pattern(false));
```

## Features

Download links for every model are in the table above.

- CTC, Granite TurboCTC, TDT, and Fluid Speech Pro: offline ASR. TDT and Fluid Speech Pro cover 25 languages with automatic language detection
- EOU, Nemotron (English and multilingual 3.5), and Multitalker: streaming ASR
- Unified: offline and buffered streaming RNNT
- Cohere Transcribe: offline multilingual ASR, 14 languages, long-form supported
- Sortformer v2, v2.1, Ultra 8-speaker, and Nemotron-3 Diarization: speaker diarization
- Ark ASR: offline speech-LLM (Whisper encoder and Qwen2 LM, greedy)
- oruk-fourier-2b INT4 and mixed INT4/INT8: ONNX bundles only, no Rust loader in this crate yet
- Token-level timestamps (CTC, TDT)

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float)
- CTC/TDT models have ~4-5 minute audio length limit. For longer files, use streaming models or split into chunks

## License

Code: MIT OR Apache-2.0

Model weights are not in this repository. Each download keeps the license of its source model, linked in the table above.
