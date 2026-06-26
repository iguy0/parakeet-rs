# WebGPU Implementation Plan — parakeet-rs

## Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 0** — Dev environment | **Partial** | Linux: `cargo build --features webgpu` OK; `vulkan-tools` installed; 2× RTX 3090 + NVIDIA 595.58.03 confirmed via `vulkaninfo` |
| **Phase 1** — EP harness (`PARAKEET_EP`) | **Done** | `execution_config()`, `ep_skip_if_unavailable()`, `PARAKEET_WEBGPU_DEVICE_ID` in `tests/common/mod.rs`; wired into `model_benchmark.rs` |
| **Phase 2** — WebGPU session + config | **Done** | `parallel_execution(false)` + `memory_pattern(false)` for WebGPU/DirectML before EP registration; `with_webgpu_device_id()` on `ExecutionConfig` |
| **Phase 3** — Example / benchmark CLI | **Done** | `--ep webgpu` on `streaming.rs` + `raw.rs`; `CARGO_FEATURES` + `PARAKEET_EP` in `run_all.sh` |
| **Phase 4** — CPU vs WebGPU correctness matrix | **Done** | Nemotron-only `webgpu_cpu_wer_parity_smoke`; CSV `ep` column; skip on WebGPU EP failure |
| **Phase 5** — CI compile gate | **Done** | Ubuntu `webgpu-build` + `cargo test --lib`; optional `webgpu-build-windows` compile job in `rust.yml` |
| **Phase 6** — Windows 11 validation | **Done** | Win11 + D3D12: Nemotron WebGPU parity **pass** (WER 1.9% CPU vs 1.9% WebGPU, delta 0%); streaming example `--ep webgpu` OK after arg-parse fix |
| **Phase 7** — Upstream ORT tracking | **Ongoing** | Nemotron fails on ort `2.0.0-rc.12`; monitor rc.13+ for WebGPU fixes |

**Last updated:** 2026-06-26 (Phase 6 Windows 11 D3D12 validation pass)  
**Current phase:** Phase 7/8 — ort bump tracking + README/AGENTS docs  
**Status:** Nemotron WebGPU **works on Windows 11 D3D12** (ort rc.12); still blocked on Linux/Vulkan (Slice validation). Models wired via junctions from `%APPDATA%/com.mwhispr.app/models` or `PARAKEET_NEMOTRON_DIR`.  
**Primary objective:** Develop and validate WebGPU on Linux, with a path to certify behavior for Windows 11 users, without changing CPU defaults or breaking upstream mergeability.

### WebGPU test model policy

**All WebGPU correctness and smoke tests use one canonical model:**

| Field | Value |
|-------|-------|
| Model | **Nemotron Speech Streaming EN 0.6B** (`nemotron-speech-streaming-en-0.6b`) |
| Local weights dir | `models/nemotron-speech-streaming-en-0.6b_int8_onnx/` (override: `PARAKEET_NEMOTRON_DIR`) |
| HF source | `lokkju/nemotron-speech-streaming-en-0.6b-int8` |
| API | `parakeet_rs::Nemotron` — streaming, 560 ms chunks |
| Smoke clip | `samples/smoke/test01_20s.wav` |

Do **not** use Parakeet CTC, TDT, or other families for WebGPU parity unless explicitly expanding the matrix after Nemotron passes. Examples (`streaming.rs`), benchmark harness (`run_all.sh`), and `webgpu_cpu_wer_parity_smoke` should all target Nemotron when `PARAKEET_EP=webgpu` or `--ep webgpu`.

**Known blocker (Linux/Vulkan, ort rc.12):** WebGPU inference on Nemotron may fail on first `transcribe_chunk` (Slice validation) or segfault. Tests must skip gracefully on EP failure until ort fixes land; do not mask failures by falling back to a different model.

---

## 1. Problem Summary

### 1.1 Goal

Enable **opt-in WebGPU acceleration** across parakeet-rs model families so that:

- Developers on **Linux** can build, run, and regression-test with `--features webgpu`.
- **Windows 11** users can rely on the same API (`ExecutionProvider::WebGPU`) with confidence that correctness was validated on D3D12 — not assumed from Linux Vulkan results.

### 1.2 Current state

| Area | Status |
|------|--------|
| Cargo feature `webgpu = ["ort/webgpu"]` | Present in `Cargo.toml` |
| `ExecutionProvider::WebGPU` wiring | Present in `src/execution.rs` (default ort config only) |
| All model loaders accept `ExecutionConfig` | Yes — no per-model EP work needed |
| Examples / benchmarks select EP | **No** — all pass `None` (CPU) |
| CI WebGPU build | **No** |
| Windows validation | **No** |
| WebGPU runtime on Nemotron (Linux) | **Fails** — see §4 |

WebGPU remains **experimental upstream** ([ort docs](https://ort.pyke.io/perf/execution-providers)): incorrect results and crashes are known. parakeet-rs already documents this in `src/execution.rs` L16 and `AGENTS.md`.

### 1.3 Observed failure (diagnostic session 2026-06-26)

On Linux (2× RTX 3090, NVIDIA driver 595.58.03, Vulkan 1.4.329):

- **CPU:** Nemotron int8 loads and transcribes a silence chunk in ~191 ms.
- **WebGPU:** Session loads, then first `transcribe_chunk` fails with ORT WebGPU validation error on a **`Slice`** node, followed by **segfault (exit 139)**.

This is **not** caused by missing Vulkan tooling (confirmed after `vulkan-tools` install). Root cause is upstream ORT WebGPU EP behavior on Nemotron's ONNX graph (dynamic shapes, Slice ops, large encoder weights).

---

## 2. Final Codebase Audit

### 2.1 Execution provider flow

```
Public API (Parakeet, Nemotron, Sortformer, …)
  → Option<ExecutionConfig>  (alias for execution::ModelConfig)
  → execution.rs::apply_to_session_builder()
  → ort SessionBuilder.with_execution_providers([WebGPU, CPU fallback])
  → model_*.rs::build_session() / forward()
  → decode loop (EP-agnostic)
```

WebGPU affects **only ONNX session creation**. Audio prep, decoders, streaming caches, and beam search require no EP-specific changes.

### 2.2 Key files

| File | Role | WebGPU relevance |
|------|------|------------------|
| `Cargo.toml` | `webgpu = ["ort/webgpu"]` feature | Entry point for consumers |
| `src/execution.rs` | EP enum, `ModelConfig`, session builder | **Primary extension target** for WebGPU options |
| `src/lib.rs` | Re-exports `ExecutionProvider`, `ExecutionConfig` | Stable public API |
| `src/model*.rs` | `exec_config.build_session(path)` per ONNX graph | No change unless per-graph EP split needed |
| `tests/common/mod.rs` | Benchmark helpers, env-based model paths | Add `PARAKEET_EP`, `execution_config()` |
| `tests/model_benchmark.rs` | Weight-gated WER/DER suite | Pass `ExecutionConfig`; add WebGPU smoke tests |
| `scripts/benchmark/run_all.sh` | Greedy/beam harness | Add `CARGO_FEATURES`, `PARAKEET_EP` |
| `scripts/benchmark/manifest.toml` | Harness documentation | Document EP env vars |
| `examples/raw.rs`, `examples/streaming.rs` | Manual demos | Add `--ep` CLI flag |
| `.github/workflows/rust.yml` | CI | Add compile-only WebGPU job |
| `README.md`, `AGENTS.md` | User / agent docs | Platform prerequisites, experimental warning |

### 2.3 Model families and WebGPU risk tier

| Model | ONNX graphs | Risk | Rationale |
|-------|-------------|------|-----------|
| Parakeet CTC | 1 static-ish graph | **Low** | Good CPU baseline; **not used for WebGPU tests** (Nemotron is canonical) |
| Parakeet TDT | encoder + decoder_joint | **Medium** | RNNT loop; duration head |
| ParakeetUnified | encoder + decoder_joint | **Medium** | Same RNNT pattern as Nemotron |
| Nemotron | encoder (~653 MB) + decoder_joint | **High** | **Failed on WebGPU** (Slice validation) |
| Parakeet EOU | encoder + decoder_joint | **High** | Streaming state + dynamic shapes |
| Cohere | encoder + decoder merged | **High** | Long-form attention; separate architecture |
| Sortformer | single diar graph | **High** | Custom STFT path; dynamic chunk metadata |
| Multitalker | encoder + decoder + Sortformer | **High** | Combines streaming ASR + diarization |

**Scope decision:** All WebGPU harness and parity tests use **Nemotron streaming EN 0.6B** only. Expand to other families (CTC, Unified, TDT, Sortformer) only after Nemotron WebGPU passes on the target platform or upstream documents a fix. Other models remain in the risk table for future matrix work.

### 2.4 Platform mapping (implicit dependency)

| Dev / deploy OS | WebGPU backend (Dawn) | Driver requirement |
|---------------|----------------------|-------------------|
| Linux | Vulkan (default) | NVIDIA/AMD/Intel Vulkan ICD |
| Windows 11 | D3D12 (primary), D3D11 | DX12 GPU; ORT bundles `dxil.dll` / `dxcompiler.dll` |
| macOS | Metal | Apple Silicon |

**Linux Vulkan success/failure does not predict Windows D3D12 behavior.** Phase 6 is mandatory for Windows 11 claims.

### 2.5 ort prebuilt binary constraints

- `webgpu` and `cuda`/`tensorrt` **cannot** be combined in one prebuilt ort binary.
- Consumers choose at compile time: `features = ["webgpu"]` OR `features = ["cuda"]`.
- Windows NVIDIA users may prefer `cuda`; cross-vendor Windows users: `webgpu` or `directml`.

Dependency pin: `ort = "2.0.0-rc.12"` → ORT ~1.24. Recent upstream fixes (e.g. [buffer segment alignment PR #27853](https://github.com/microsoft/onnxruntime/pull/27853), [Slice int32 PR #26693](https://github.com/microsoft/onnxruntime/pull/26693)) may require bumping ort when released.

### 2.6 Runtime dylibs

ort `copy-dylibs` (default) symlinks `libwebgpu_dawn.so` into `target/debug/` on Linux:

```
target/debug/libwebgpu_dawn.so → ~/.cache/ort.pyke.io/dfbin/.../libwebgpu_dawn.so
```

Packaged apps on Linux may need `RPATH` or install-time dylib placement ([ort runtime dylib docs](https://ort.pyke.io/perf/execution-providers)).

---

## 3. Reproduction Steps

### 3.0 Fresh machine setup (clone → test)

Tracked in git (no download needed):

- `samples/smoke/test01_20s.wav` + `test01_20s_transcript.txt` — 20 s smoke clip for WER tests
- `plans/WEBGPU_IMPLEMENTATION_PLAN.md` — this document
- `scripts/benchmark/run_all.sh`, `manifest.toml` — harness with `CARGO_FEATURES` / `PARAKEET_EP`
- `tests/model_benchmark.rs`, `tests/common/mod.rs` — `webgpu_cpu_wer_parity_smoke` and `PARAKEET_EP` harness

**Not** in git (download locally into `models/`):

```bash
git clone https://github.com/iguy0/parakeet-rs.git
cd parakeet-rs

# Required for all WebGPU tests — Nemotron streaming EN 0.6B int8 (~700 MB)
mkdir -p models/nemotron-speech-streaming-en-0.6b_int8_onnx
huggingface-cli download lokkju/nemotron-speech-streaming-en-0.6b-int8 \
  --local-dir models/nemotron-speech-streaming-en-0.6b_int8_onnx
# Expect: encoder.onnx, encoder.onnx.data, decoder_joint.onnx, tokenizer.model
```

**Linux Vulkan prerequisites:** `vulkan-tools`, GPU driver with Vulkan ICD (`vulkaninfo --summary`).

**Windows 11:** Rust + DX12 GPU; same clone + same Nemotron weights; WebGPU uses D3D12 (see Phase 6).

**Run WebGPU parity smoke (Nemotron streaming, CPU vs WebGPU WER):**

```bash
cargo test --release --test model_benchmark --features webgpu \
  webgpu_cpu_wer_parity_smoke -- --nocapture
```

**Run with WebGPU EP explicitly:**

```bash
PARAKEET_EP=webgpu cargo test --release --test model_benchmark --features webgpu \
  webgpu_cpu_wer_parity_smoke -- --nocapture
```

**Manual examples:**

```bash
cargo run --release --features webgpu --example raw \
  samples/smoke/test01_20s.wav -- --ep webgpu

cargo run --release --features webgpu --example streaming \
  samples/smoke/test01_20s.wav -- --ep webgpu
```

**Benchmark harness:**

```bash
CARGO_FEATURES=webgpu PARAKEET_EP=webgpu ./scripts/benchmark/run_all.sh
```

---

### 3.1 Environment

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd parakeet-rs   # repo root after clone
cargo build --features webgpu --release
```

Prerequisites:

- Nemotron int8 weights at `models/nemotron-speech-streaming-en-0.6b_int8_onnx/` (or `PARAKEET_NEMOTRON_DIR`) — **required for all WebGPU tests**
- Linux Vulkan: `vulkan-tools`, NVIDIA/AMD/Intel driver with Vulkan ICD
- Verify GPU: `vulkaninfo --summary`

### 3.2 Minimal repro (used during audit)

Create a temporary smoke binary (or use Phase 1 harness once implemented):

```rust
use parakeet_rs::{ExecutionConfig, ExecutionProvider, Nemotron};

let cfg = ExecutionConfig::new().with_execution_provider(ExecutionProvider::WebGPU);
let mut model = Nemotron::from_pretrained(nemotron_dir, Some(cfg))?;
let _ = model.transcribe_chunk(&vec![0.0f32; 8960])?;
```

### 3.3 Expected vs actual

| Step | CPU | WebGPU |
|------|-----|--------|
| `cargo build --features webgpu` | N/A | **Pass** |
| `Nemotron::from_pretrained(..., Some(cfg))` | **Pass** | **Pass** (with Dawn warnings) |
| First `transcribe_chunk` | **Pass** (~191 ms) | **Fail** + segfault |

### 3.4 Log output (2026-06-26)

**Dawn warnings (WebGPU init):**

```
Warning: maxDynamicUniformBuffersPerPipelineLayout artificially reduced from 500000 to 16 ...
Warning: maxDynamicStorageBuffersPerPipelineLayout artificially reduced from 500000 to 16 ...
```

**ORT error:**

```
Error: Ort(Error { code: GenericFailure, message: "WebGPU validation failed. [Invalid Buffer (unlabeled)] is invalid.
 - While validating entries[1] as a Buffer.
Expected entry layout: {type: BufferBindingType::Storage, minBindingSize: 4, hasDynamicOffset: 0}
 - While validating [BindGroupDescriptor \"\"Slice\"\"] against [BindGroupLayout (unlabeled)]
 - While calling [Device].CreateBindGroup([BindGroupDescriptor \"\"Slice\"\"]).\"
```

**Process exit:** `139` (SIGSEGV) after error return.

### 3.5 Vulkan verification (post vulkan-tools install)

```
GPU0–GPU3: NVIDIA GeForce RTX 3090, driver 595.58.03, Vulkan 1.4.329
minStorageBufferOffsetAlignment = 16 (0x10)
```

Vulkan stack is healthy; failure is at ORT WebGPU / graph level.

---

## 4. Root Cause Analysis

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| Missing Vulkan drivers | **Ruled out** | `vulkaninfo` sees RTX 3090; `nvidia_icd.json` present |
| parakeet-rs EP wiring bug | **Unlikely** | Same code path works for CPU; ort `WebGPU::default()` is standard |
| Nemotron ONNX graph incompatibility with WebGPU EP | **Primary** | Error on `Slice` node during encoder forward |
| ort version too old for WebGPU fixes | **Contributing** | rc.12 predates Mar 2026 buffer-alignment fix in ORT main |
| Multi-GPU device selection | **Unknown** | Not tested; ort supports `with_device_id` — worth Phase 2 |
| Linux-only Vulkan issue | **Possible but unproven** | Windows D3D12 may differ; must test separately |

**Effect on primary objective:** parakeet-rs is **architecturally ready** but **operationally blocked** for Nemotron WebGPU until upstream ORT fixes land or workarounds (`forceCpuNodeNames`, CPU fallback per-node) are applied. Development can proceed on harness + simpler models + Windows validation in parallel.

---

## 5. Implementation Plan

### Phase 0 — Dev environment (complete checklist)

- [x] Rust toolchain (`~/.cargo/bin/cargo`)
- [x] `cargo build --features webgpu`
- [x] Model weights under `models/`
- [x] NVIDIA driver + Vulkan ICD
- [x] `vulkan-tools` (`vulkaninfo --summary`)
- [ ] Document `PARAKEET_WEBGPU_DEVICE_ID` convention (Phase 1)
- [x] `samples/smoke/` fixture committed (20 s WAV + transcript for cross-machine tests)
- [ ] Optional: install CTC model for general (non-WebGPU) benchmark matrix (`models/parakeet-ctc`)

**Verify:**

```bash
vulkaninfo --summary | grep -A2 'deviceName'
cargo build --features webgpu
```

---

### Phase 1 — EP harness in test common

**Goal:** Single env-driven entry point for execution config, mirroring existing `PARAKEET_*_DIR` pattern.

**File:** `tests/common/mod.rs`

Add:

```rust
pub const EP_ENV: &str = "PARAKEET_EP";
pub const WEBGPU_DEVICE_ID_ENV: &str = "PARAKEET_WEBGPU_DEVICE_ID";

/// Build ExecutionConfig from PARAKEET_EP (default: cpu).
/// Values: cpu | webgpu | cuda | directml | … (compile-time gated)
pub fn execution_config() -> parakeet_rs::ExecutionConfig { ... }
```

Behavior:

| `PARAKEET_EP` | `--features` required | Notes |
|---------------|----------------------|-------|
| unset / `cpu` | default | Current behavior |
| `webgpu` | `webgpu` | Skip test with message if feature not enabled |
| `webgpu` + EP init failure | — | Skip (not fail) in CI without GPU |

**File:** `tests/model_benchmark.rs`

Replace hardcoded `None` configs:

```rust
// Before
Nemotron::from_pretrained(nemotron_dir(), None)

// After
Nemotron::from_pretrained(nemotron_dir(), Some(execution_config()))
```

Add dedicated test (optional, weight-gated):

```rust
#[test]
fn nemotron_streaming_webgpu_smoke() {
    // Only runs when PARAKEET_EP=webgpu and feature enabled
    // Assert: no panic; if inference succeeds, WER within SMOKE_MAX_WER
    // If ORT returns error: eprintln + skip (document known upstream failure)
}
```

**Acceptance criteria:**

```bash
# CPU (unchanged)
cargo test --release --test model_benchmark -- --nocapture

# WebGPU attempt (may skip or fail until ORT fixed)
cargo test --release --test model_benchmark --features webgpu -- --nocapture
PARAKEET_EP=webgpu cargo test --release --test model_benchmark --features webgpu -- --nocapture
```

---

### Phase 2 — WebGPU session flags + config on ExecutionConfig

**Goal:** Match working native Rust ASR projects and ORT DirectML/WebGPU requirements; expose ort WebGPU tuning without breaking existing API.

**Required fix (from [transcribe-rs](https://github.com/cjpais/transcribe-rs/blob/main/src/onnx/session.rs) + [ORT DirectML docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)):** When `execution_provider` is `WebGPU` or `DirectML`, `apply_to_session_builder` must set:

```rust
builder = builder
    .with_parallel_execution(false)?
    .with_memory_pattern(false)?;
```

Apply **before** `with_execution_providers`. Do not apply for CPU/CUDA/CoreML (transcribe-rs excludes WebGPU from `Auto` mode for this reason).

**Verified locally (2026-06-26):** Adding these flags does **not** fix Nemotron Slice validation on ort rc.12 — still required for spec compliance and Windows DirectML.

**File:** `src/execution.rs`

Add optional struct (names tentative):

```rust
#[cfg(feature = "webgpu")]
#[derive(Debug, Clone, Default)]
pub struct WebGpuConfig {
    pub device_id: Option<i32>,
    pub validation_mode: Option<...>,  // ort::ep::webgpu::ValidationMode
    pub preferred_layout: Option<...>,
    pub dawn_backend_type: Option<...>, // Vulkan | D3D12 — useful for forced repro
}

pub struct ModelConfig {
    // existing fields ...
    #[cfg(feature = "webgpu")]
    pub webgpu: Option<WebGpuConfig>,
}
```

Update WebGPU arm:

```rust
ExecutionProvider::WebGPU => {
    let mut ep = ort::ep::WebGPU::default();
    if let Some(wg) = &self.webgpu {
        if let Some(id) = wg.device_id { ep = ep.with_device_id(id); }
        // ...
    }
    builder.with_execution_providers([
        ep.build(),
        CPUExecutionProvider::default().build().error_on_failure(),
    ])?
}
```

Env mapping in `tests/common/mod.rs`:

- `PARAKEET_WEBGPU_DEVICE_ID=0` → `WebGpuConfig { device_id: Some(0), .. }`

**Do not** change defaults; `webgpu: None` keeps current `WebGPU::default()`.

**Acceptance:** Unit test or doc test that config builder chains compile with `feature = "webgpu"`.

---

### Phase 3 — Example and shell harness CLI

**Goal:** Manual testing without editing Rust source.

**Files:** `examples/streaming.rs`, `examples/raw.rs`

Add shared pattern (match existing `--decoding` / `--beam-size` parsing):

```
cargo run --features webgpu --example streaming -- samples/smoke/test01_20s.wav -- --ep webgpu
cargo run --features webgpu --example raw -- samples/smoke/test01_20s.wav -- --ep webgpu
```

Helper (inline or small module in examples — prefer inline to avoid new files unless duplicated 3+ times):

```rust
fn parse_execution_config(args: &[String]) -> Option<ExecutionConfig> { ... }
```

**File:** `scripts/benchmark/run_all.sh`

Add env vars:

```bash
CARGO_FEATURES="${CARGO_FEATURES:-}"   # e.g. webgpu
PARAKEET_EP="${PARAKEET_EP:-cpu}"
```

Pass to cargo:

```bash
cargo run $CARGO_FLAGS --features "$CARGO_FEATURES" --example "$example" ...
# example binary reads PARAKEET_EP or accept --ep forwarded
```

**File:** `scripts/benchmark/manifest.toml` — document new env keys.

---

### Phase 4 — CPU vs WebGPU correctness matrix

**Goal:** Quantify WebGPU accuracy vs CPU baseline; block regressions once ORT works.

**File:** `tests/common/mod.rs`

Extend CSV:

```rust
pub const ASR_CSV_HEADER: &str = "run_id,label,sample,model,mode,decoding,ep,\
audio_secs,proc_secs,rtf,wer,cer";
```

Update `record_asr(...)` to accept `ep: &str`.

**File:** `tests/model_benchmark.rs`

Add test `webgpu_cpu_wer_parity_smoke` (**Nemotron streaming only**):

1. Skip if `nemotron_available()` is false
2. Transcribe smoke clip with CPU → `wer_cpu` (`Nemotron::from_pretrained`, streaming chunks)
3. Transcribe smoke clip with WebGPU → `wer_gpu` (or capture error)
4. If WebGPU succeeds: assert `|wer_gpu - wer_cpu| <= PARAKEET_WEBGPU_MAX_WER_DELTA` (default e.g. 0.05)
5. If WebGPU fails: print skip + link to this plan §4 (known upstream issue on ort rc.12 Linux/Vulkan)

**Remove** CTC branch and `PARAKEET_WEBGPU_NEMOTRON_PARITY` opt-in once refactored.

Run matrix (document in plan; **WebGPU P0 = Nemotron only**):

| Model | Mode | WebGPU priority |
|-------|------|-----------------|
| **Nemotron EN 0.6B** | streaming | **P0** (canonical; currently fails Linux/Vulkan rc.12) |
| CTC | offline | P2 (CPU benchmark only unless matrix expanded) |
| Unified | offline + streaming | P2 |
| TDT | offline | P2 |
| Sortformer | diarization | P3 |

---

### Phase 5 — CI compile gate

**File:** `.github/workflows/rust.yml`

Add job (no GPU required):

```yaml
webgpu-build:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo build --features webgpu
    - run: cargo test --features webgpu --lib  # unit tests only; no model weights
```

Optional Windows compile-only job:

```yaml
webgpu-build-windows:
  runs-on: windows-latest
  steps:
    - run: cargo build --features webgpu
```

Do **not** assert WebGPU inference in GitHub-hosted CI (no GPU). Weight-gated tests stay local/self-hosted.

---

### Phase 6 — Windows 11 validation

**Goal:** Certify D3D12 path independently of Linux Vulkan.

**Procedure:**

1. Windows 11 machine with DX12 GPU (Intel UHD / AMD / NVIDIA all valid).
2. Install Rust, clone repo, download same Nemotron int8 weights.
3. Build: `cargo build --release --features webgpu`
4. Run:
   ```powershell
   $env:PARAKEET_EP = "webgpu"
   cargo test --release --test model_benchmark --features webgpu -- --nocapture
   ```
5. Run examples:
   ```powershell
   cargo run --release --features webgpu --example streaming -- samples\smoke\test01_20s.wav -- --ep webgpu
   ```
6. Record: WER, RTF, any ORT errors, GPU used (Task Manager).

**Compare three baselines:**

| Run | Platform | Backend |
|-----|----------|---------|
| A | Linux | CPU |
| B | Linux | WebGPU (Vulkan) |
| C | Windows 11 | WebGPU (D3D12) |

Acceptance for Windows 11 users (once ORT stable):

- C produces transcript (no crash)
- `wer(C) <= SMOKE_MAX_WER`
- `|wer(C) - wer(A)| <= PARAKEET_WEBGPU_MAX_WER_DELTA`

---

### Phase 7 — Upstream ORT tracking and escalation

**Actions:**

1. **Bump ort** when `2.0.0-rc.13+` (or stable 2.0) ships with WebGPU fixes — re-run Phase 4 matrix.
2. **File upstream issue** if Nemotron encoder still fails after bump:
   - Attach: ORT version, GPU/driver, `vulkaninfo --summary`, minimal repro (encoder-only ORT Run on `encoder.onnx` with WebGPU EP).
   - Reference: [ORT #26690](https://github.com/microsoft/onnxruntime/issues/26690) (Slice), [PR #27853](https://github.com/microsoft/onnxruntime/pull/27853) (buffer alignment).
3. **Workaround research:** ort `WebGPU::with_force_cpu_node_names(...)` for failing Slice nodes — only if upstream slow; document in `execution.rs` comment.

**File:** `Cargo.toml` — bump `ort` version in controlled PR with changelog note.

---

### Phase 8 — Documentation and AGENTS.md

**Files:** `README.md`, `AGENTS.md`

Add concise sections:

- Platform prerequisites (Linux Vulkan, Windows DX12, VC++ runtime)
- Experimental warning (keep prominent)
- Example commands with `--features webgpu` and `--ep webgpu`
- Env vars: `PARAKEET_EP`, `PARAKEET_WEBGPU_DEVICE_ID`
- Windows EP guidance table: CUDA (NVIDIA), DirectML (Windows native), WebGPU (cross-vendor)

Update AGENTS.md "Current local delta" table with WebGPU harness row when Phase 1 lands.

---

## 6. Acceptance Criteria (Release-Ready WebGPU)

WebGPU remains **opt-in experimental** until all of:

- [ ] Phase 1–3 complete (harness + CLI)
- [x] Phase 4: Nemotron smoke passes WER parity CPU vs WebGPU on Linux **or** documented upstream blocker with skip logic (current: blocked on ort rc.12; test skips on WebGPU failure)
- [x] Phase 4 follow-up: refactor `webgpu_cpu_wer_parity_smoke` to Nemotron-only (remove CTC / `PARAKEET_WEBGPU_NEMOTRON_PARITY`)
- [x] Phase 6: Windows 11 smoke passes (transcript + WER threshold)
- [ ] Phase 5: CI compile gate green
- [ ] No change to default EP (CPU)
- [ ] README documents platform requirements and experimental status

Performance (RTF) improvements are **nice-to-have**; correctness and no-crash are **required**.

---

## 7. Risks and Side Effects

| Risk | Mitigation |
|------|------------|
| WebGPU segfault after ORT error | Document; use `validation_mode`; fail-fast at session build where possible |
| CI breaks on missing GPU | Weight-gated + skip-on-EP-failure pattern |
| CSV schema change breaks parsers | Append `ep` column; old rows remain valid |
| `cuda` + `webgpu` feature confusion | Document mutual exclusion in README |
| Dual RTX 3090 wrong GPU selected | `PARAKEET_WEBGPU_DEVICE_ID` in Phase 2 |
| Fork divergence | Keep changes in harness + optional config; no default behavior change |

---

## 8. Quick Command Reference (Target State)

```bash
# Build
cargo build --release --features webgpu

# Benchmark suite (CPU default)
cargo test --release --test model_benchmark --features "sortformer multitalker cohere" -- --nocapture

# WebGPU Nemotron parity smoke (canonical WebGPU test model)
PARAKEET_EP=webgpu PARAKEET_WEBGPU_DEVICE_ID=0 \
  cargo test --release --test model_benchmark --features webgpu \
  webgpu_cpu_wer_parity_smoke -- --nocapture

# Manual streaming demo
cargo run --release --features webgpu --example streaming -- \
  samples/smoke/test01_20s.wav -- --ep webgpu

# Greedy/beam shell harness with WebGPU
CARGO_FEATURES=webgpu PARAKEET_EP=webgpu scripts/benchmark/run_all.sh

# Vulkan sanity check
vulkaninfo --summary
```

---

## 9. Session Context (for future debugging)

**Exploration date:** 2026-06-26  
**Machine:** Linux 6.8.0, 2× NVIDIA RTX 3090, driver 595.58.03  
**ort version:** 2.0.0-rc.12  
**parakeet-rs version:** 0.3.6  
**Model tested:** `models/nemotron-speech-streaming-en-0.6b_int8_onnx`  
**Primary blocker:** ORT WebGPU EP validation failure on Slice node during Nemotron encoder inference (Linux/Vulkan); segfault on error path  
**Not the cause:** Missing Vulkan tooling (resolved), parakeet-rs EP wiring (CPU works with same API)

**Related plans:** `plans/BEAM_SEARCH_IMPLEMENTATION_PLAN.md` (benchmark harness pattern to extend)

### Phase 6 results — Windows 11 (2026-06-26)

| Run | Platform | Backend | Smoke WER | Notes |
|-----|----------|---------|-----------|-------|
| A | Windows 11 | CPU | 1.9% | `webgpu_cpu_wer_parity_smoke`, RTF ~2.2x |
| C | Windows 11 | WebGPU (D3D12) | 1.9% | delta 0.0%; streaming example `--ep webgpu` transcript matches CPU |
| B | Linux | WebGPU (Vulkan) | — | Blocked ort rc.12 Slice validation (test skips) |

Models: junctions from `%APPDATA%/com.mwhispr.app/models/nemotron-speech-streaming-en-0.6b-int8` → `models/nemotron-speech-streaming-en-0.6b_int8_onnx/`.

**Upstream references:**

- [ort execution providers](https://ort.pyke.io/perf/execution-providers)
- [ONNX Runtime WebGPU plugin EP v0.1.0](https://github.com/microsoft/onnxruntime/releases/tag/plugin-ep-webgpu/v0.1.0)
- [ORT Slice FP16 regression #26690](https://github.com/microsoft/onnxruntime/issues/26690)
- [ORT buffer alignment fix #27853](https://github.com/microsoft/onnxruntime/pull/27853)

---

## 10. External Reference Audit (2026-06-26)

Cross-check of working/current Rust projects and official ort/ORT docs against parakeet-rs WebGPU integration.

### 10.1 Reference projects

| Project | Stack | WebGPU approach | Relevance |
|---------|-------|-----------------|-----------|
| **[transcribe-rs](https://github.com/cjpais/transcribe-rs)** | ort 2.x, ASR (incl. Parakeet) | `ort-webgpu` feature; `WebGPU::default().build()`; **mandatory** `parallel_execution(false)` + `memory_pattern(false)`; explicit opt-in (not Auto) | **Primary native Rust reference** — same domain, same ort EP API |
| **[ort](https://github.com/pykeio/ort)** (pykeio) | Bindings crate we use | Documents WebGPU for Windows/macOS/Linux; prebuilt `webgpu` feature; experimental warning | Authoritative API + platform matrix |
| **[ort-web](https://ort.pyke.io/backends/web)** | Browser/WASM | `ort_web::api(FEATURE_WEBGPU)` + session EP registration | Different backend init (`set_api`); **not applicable** to native parakeet-rs |
| **[wonnx](https://github.com/webonnx/wonnx)** | Pure Rust + wgpu | No ONNX Runtime; separate inference engine | Not comparable — different runtime |
| **Microsoft onnxruntime/rust** | Official Rust bindings (experimental) | CPU/GPU download builds; WebGPU via `--use_webgpu` build flag | Lower-level; parakeet-rs correctly uses `ort` instead |
| **parakeet-rs upstream README** | This repo | Author reports WebGPU working on **Apple M3** (Metal via Dawn) | Empirical success on macOS; not proof for Linux/Windows |

### 10.2 Pattern comparison: parakeet-rs vs transcribe-rs

**What parakeet-rs gets right (matches references):**

```toml
# Cargo.toml — same as ort docs
webgpu = ["ort/webgpu"]
```

```rust
// execution.rs — same EP registration pattern as ort docs + transcribe-rs
builder.with_execution_providers([
    ort::ep::WebGPU::default().build(),
    CPUExecutionProvider::default().build().error_on_failure(),
])?
```

All model loaders already accept `Option<ExecutionConfig>` — no per-model EP plumbing needed.

**What parakeet-rs is missing (transcribe-rs + ORT DirectML docs require):**

| Setting | parakeet-rs today | transcribe-rs / ORT requirement |
|---------|-------------------|--------------------------------|
| `with_parallel_execution(false)` | Not set (ort default is false) | **Explicitly set** when WebGPU/DirectML active |
| `with_memory_pattern(false)` | Not set (**mem pattern likely ON**) | **Required** for DirectML; transcribe-rs applies for WebGPU too |
| WebGPU in Auto/fallback | N/A (explicit EP only) | Correct — WebGPU must stay opt-in |
| Windows dylibs | Relies on ort `copy-dylibs` | Same — dxil/dxcompiler copied on Windows builds |

**Local retest:** Applying `parallel_execution(false)` + `memory_pattern(false)` via `with_custom_configure` did **not** resolve Nemotron Slice validation on ort rc.12/Linux — flags are still required for correctness on Windows DirectML and spec compliance.

### 10.3 Windows 11 viability (from official sources)

| Source | Windows behavior |
|--------|------------------|
| [ort EP docs](https://ort.pyke.io/perf/execution-providers) | WebGPU prebuilt binaries for **Windows, macOS, Linux**; Windows uses **DirectX 12 and 11** |
| [ORT WebGPU plugin EP v0.1.0](https://github.com/microsoft/onnxruntime/releases/tag/plugin-ep-webgpu/v0.1.0) | Native binaries for **Windows x64/arm64**; bundles `dxil.dll` / `dxcompiler.dll` |
| [ORT issue #21917](https://github.com/microsoft/onnxruntime/issues/21917) | Dawn backend on Windows is **D3D12** (Vulkan available via build flag but not default) |
| [pykeio/ort #349](https://github.com/pykeio/ort/issues/349) | WebGPU confirmed working on Windows native; helper DLLs required next to executable |
| [ORT DirectML docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) | Windows-native alternative; same session constraints as WebGPU path in transcribe-rs |

**Windows 11 conclusion:** The **API and build path are correct** for Windows — same `features = ["webgpu"]` and `ExecutionProvider::WebGPU`. Success depends on:

1. Adding required session flags in `execution.rs` (Phase 2).
2. Validating on Windows hardware (Phase 6) — **D3D12 may behave differently** than Linux Vulkan for Nemotron.
3. Shipping `libwebgpu_dawn`/helper DLLs via ort `copy-dylibs` or install step.
4. Upstream ORT fixes for complex ASR graphs (Slice nodes in streaming encoders).

**Windows EP recommendation for users:**

| GPU | Recommended EP | Feature |
|-----|----------------|---------|
| NVIDIA | CUDA (production) | `cuda` |
| AMD / Intel / any DX12 | WebGPU or DirectML | `webgpu` or `directml` |
| Cross-platform library | WebGPU | `webgpu` |

Note: `cuda` and `webgpu` **cannot** be combined in one ort prebuilt binary.

### 10.4 Accuracy verdict

| Question | Answer |
|----------|--------|
| Is parakeet-rs WebGPU integration structurally correct? | **Yes** — matches ort + transcribe-rs EP registration |
| Will it work out of the box on Windows 11? | **Plausibly for simpler models**; Nemotron unproven until Windows test + ORT fixes |
| Must we change code before claiming WebGPU support? | **Yes** — add session flags (Phase 2); add harness (Phase 1–4); validate Windows (Phase 6) |
| Is Linux dev sufficient for Windows certification? | **No** — different Dawn backends (Vulkan vs D3D12) |
| Is upstream author's Mac WebGPU claim consistent? | **Yes** — ort ships macOS WebGPU (Metal backend); different path than Linux |

### 10.5 Immediate code change (highest priority)

In `src/execution.rs` `apply_to_session_builder`, after thread/optimization setup and **before** EP match:

```rust
let needs_sequential = matches!(
    self.execution_provider,
    ExecutionProvider::WebGPU | ExecutionProvider::DirectML
);
if needs_sequential {
    builder = builder
        .with_parallel_execution(false)?
        .with_memory_pattern(false)?;
}
```

Gate `ExecutionProvider::WebGPU` behind `#[cfg(feature = "webgpu")]` in the match. This aligns parakeet-rs with the only production-grade native Rust ASR reference (transcribe-rs) and ORT's documented DirectML constraints (same session model transcribe-rs applies to WebGPU).
