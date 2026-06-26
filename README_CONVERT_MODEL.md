# ONNX model export (parakeet-rs)

This project uses a vendored [parakeet-rs](https://github.com/altunenes/parakeet-rs) fork under `parakeet-rs/` for:

- **Python export scripts** (`parakeet-rs/scripts/`) — NeMo checkpoints to ONNX
- **Rust inference** (optional) — our fork extends `Sortformer` for 8-speaker diarization via ONNX metadata

The primary deliverable is ONNX artifacts for downstream runtimes (ONNX Runtime, sherpa-onnx, Hugging Face, etc.). The local `parakeet-rs` changes are structured to be upstream-mergeable.

All export commands are run from this project root unless noted otherwise.

## Environment

NeMo exports are heavy. A working setup looks like:

```bash
conda create -n onnx python=3.12
conda activate onnx
pip install "nemo_toolkit[asr]" onnx onnxruntime soundfile
# PyTorch 2.9+ is supported; export scripts patch torch.onnx.export automatically.
```

For **int8 quantization only** (no NeMo):

```bash
pip install onnx 'onnxruntime>=1.20'
```

Colab also works for the initial fp32 export if you prefer not to install NeMo locally.

## Primary workflow: Nemotron streaming (English 0.6B)

This is the pipeline we use in this repo.

### 1. Download the NeMo checkpoint

Place the `.nemo` file locally, e.g.:

```
nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo
```

Source: [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

### 2. Export fp32 ONNX

```bash
python parakeet-rs/scripts/export_nemotron_streaming.py \
  nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo \
  out_onnx/
```

**Output (`out_onnx/`):**

| File | Description |
|------|-------------|
| `encoder.onnx` | Streaming encoder graph |
| `encoder.onnx.data` | Encoder weights (~2.3 GB external data) |
| `decoder_joint.onnx` | RNNT decoder + joint network |
| `tokenizer.model` | SentencePiece vocabulary |
| `config.json` | Streaming params, cache shapes, test I/O |

#### Streaming latency (chosen at export)

Set `--right-context` when exporting. Quantization does **not** change latency.

| Latency | `--right-context` | Mel frames / step | Samples / step @ 16 kHz |
|---------|-------------------|-------------------|---------------------------|
| 80 ms | 0 | 8 | 1280 |
| 160 ms | 1 | 16 | 2560 |
| **560 ms (default)** | **6** | **56** | **8960** |
| 1120 ms | 13 | 112 | 17920 |

Example (560 ms, default):

```bash
python parakeet-rs/scripts/export_nemotron_streaming.py \
  nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo \
  out_onnx/ \
  --right-context 6
```

`--left-context` defaults to `70` and should usually stay unchanged.

### 3. Quantize to int8

```bash
python parakeet-rs/scripts/quantize_nemotron_streaming.py \
  out_onnx/ \
  out_onnx_int8/ \
  --clean-output
```

**Output (`out_onnx_int8/`)** — same layout as [lokkju/nemotron-speech-streaming-en-0.6b-int8](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8):

| File | Description |
|------|-------------|
| `encoder.onnx` | Quantized encoder (~623 MB, single file) |
| `decoder_joint.onnx` | Quantized decoder/joint (~8.6 MB) |
| `tokenizer.model` | Copied from fp32 export |

Standard filenames (no `.int8` suffix). The directory name indicates quantization.

#### Quantize script options

| Flag | Purpose |
|------|---------|
| `--clean-output` | Remove existing files in `output_dir` before writing |
| `--skip-verify` | Skip fp32 vs int8 numerical comparison |
| `--work-dir PATH` | Temp directory for intermediate files |
| `--keep-work-dir` | Keep temp dir after success (debugging) |

The quantize step uses ONNX Runtime **dynamic int8** (`quantize_dynamic`): encoder weights `QUInt8`, decoder/joint `QInt8`. ORT may log warnings about unquantizable `Slice`/`Tile` tensors — these are expected.

See `out_onnx_int8/README.md` for model-card details on the current bundle.

## Ultra Diar Sortformer 8spk (speaker diarization)

Streaming speaker diarization for up to **8 speakers**, extending NVIDIA's 4-speaker Sortformer via [Ultra-Sortformer](https://github.com/LilDevsy0117/Ultra-Sortformer).

**Published ONNX:** [investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx](https://huggingface.co/investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx)

NeMo source checkpoint (for re-export): [devsy0117/ultra_diar_streaming_sortformer_8spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1)

### 1. Download the ONNX model

```bash
huggingface-cli download investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx \
  --local-dir ultra_diar_streaming_sortformer_8spk_v1_onnx
```

Or download `ultra_diar_streaming_sortformer_8spk_v1.onnx` (~492 MB) from the [Hugging Face repo](https://huggingface.co/investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx).

| File | Description |
|------|-------------|
| `ultra_diar_streaming_sortformer_8spk_v1.onnx` | Single fp32 streaming graph (~492 MB) |

Streaming params are embedded in **ONNX metadata** (`num_speakers=8`, `chunk_len=340`, `right_context=40`, `fifo_len=40`, `spkcache_len=376`). Streaming latency is ~30.4 s per step.

See `ultra_diar_streaming_sortformer_8spk_v1_onnx/README.md` (also on Hugging Face) for full graph I/O, preprocessor requirements, and ONNX Runtime usage.

### 2. Re-export from NeMo (optional)

```bash
conda activate onnx

python parakeet-rs/scripts/export_ultra_diar_8spk.py \
  ultra_diar_streaming_sortformer_8spk_v1/ultra_diar_streaming_sortformer_8spk_v1.nemo \
  ultra_diar_streaming_sortformer_8spk_v1_onnx/ultra_diar_streaming_sortformer_8spk_v1.onnx
```

Place the `.nemo` file from [devsy0117/ultra_diar_streaming_sortformer_8spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1) locally first.

### 3. Inference (parakeet-rs)

Our vendored `parakeet-rs` reads `num_speakers` and streaming constants from ONNX metadata. No code changes are needed when switching between 4spk and 8spk exports.

```bash
cd parakeet-rs
cargo run --release --example streaming-diarization --features sortformer -- \
  /path/to/audio.wav \
  ../ultra_diar_streaming_sortformer_8spk_v1_onnx/ultra_diar_streaming_sortformer_8spk_v1.onnx
```

```rust
use parakeet_rs::sortformer::{Sortformer, DiarizationConfig};

let mut sortformer = Sortformer::with_config(
    "ultra_diar_streaming_sortformer_8spk_v1_onnx/ultra_diar_streaming_sortformer_8spk_v1.onnx",
    None,
    DiarizationConfig::callhome(),
)?;
// sortformer.num_speakers() == 8
// sortformer.latency() ≈ 30.4s — use feed()/flush() for buffered streaming
```

4-speaker NVIDIA exports use `export_diar_sortformer.py` instead; same `Sortformer` API with `num_speakers=4` metadata (or default).

## Other export scripts (parakeet-rs)

These live in `parakeet-rs/scripts/` for models outside the default Nemotron / 8spk diar pipelines.

| Script | Model | Notes |
|--------|-------|-------|
| `export_ultra_diar_8spk.py` | Ultra Sortformer 8spk | Recommended 8-speaker export; ORT smoke test built in |
| `export_diar_sortformer.py` | Sortformer 4spk | Single `.onnx`; configurable chunk/cache args; writes `num_speakers=4` metadata |
| `export_nemotron_streaming_multilingual.py` | Nemotron 3.5 multilingual 0.6B | Same streaming encoder pattern + `prompt_index` input; needs NeMo `@main` |
| `export_parakeet_unified.py` | Parakeet Unified EN 0.6B | Offline/buffered RNNT; optional `--quantize-int8` built in |
| `export_multitalker.py` | Multitalker streaming 0.6B | Speaker-target encoder inputs; optional int8 in-script |
| `export_realtime_eou_120m.py` | Parakeet EOU 120M | Low-latency streaming EOU model; edit paths at top of file |

Each script’s header comment documents model-specific requirements, outputs, and flags. Run with `--help` where argparse is used.

## Project layout

```
onnx/
├── README.md                                    # this file
├── nemotron-speech-streaming-en-0.6b/           # NeMo checkpoint (ASR)
├── ultra_diar_streaming_sortformer_8spk_v1/     # NeMo checkpoint (8spk diar, re-export only)
├── ultra_diar_streaming_sortformer_8spk_v1_onnx/ # Published fp32 ONNX (+ README)
├── out_onnx/                                    # Nemotron fp32 export
├── out_onnx_int8/                               # Nemotron int8 bundle
└── parakeet-rs/                                 # vendored fork (export scripts + Rust inference)
    ├── scripts/
    │   ├── export_nemotron_streaming.py
    │   ├── export_ultra_diar_8spk.py
    │   ├── quantize_nemotron_streaming.py
    │   └── …
    └── src/sortformer.rs                        # metadata-driven num_speakers
```

## Quick reference

### Nemotron 560 ms int8 bundle

```bash
python parakeet-rs/scripts/export_nemotron_streaming.py \
  nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo \
  out_onnx/

python parakeet-rs/scripts/quantize_nemotron_streaming.py \
  out_onnx/ out_onnx_int8/ --clean-output
```

### Ultra Sortformer 8spk diarization

Download: [investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx](https://huggingface.co/investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx)

Re-export from NeMo:

```bash
python parakeet-rs/scripts/export_ultra_diar_8spk.py \
  ultra_diar_streaming_sortformer_8spk_v1/ultra_diar_streaming_sortformer_8spk_v1.nemo \
  ultra_diar_streaming_sortformer_8spk_v1_onnx/ultra_diar_streaming_sortformer_8spk_v1.onnx
```

## References

- parakeet-rs upstream: https://github.com/altunenes/parakeet-rs
- Nemotron base model: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- Example int8 layout: https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8
- Ultra Sortformer 8spk ONNX: https://huggingface.co/investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx
- Ultra Sortformer 8spk NeMo: https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1
- NVIDIA Sortformer 4spk base: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- sherpa-onnx NeMo export scripts: https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo
