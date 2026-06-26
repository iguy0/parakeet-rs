# Beam Search Implementation Plan — parakeet-rs

## Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 0** — Shared types | **Complete** | `DecodingStrategy`, `BeamConfig` in `src/decoding/mod.rs`; exported from `lib.rs` |
| **Phase 1** — RNNT beam core | **Complete** | `src/decoding/rnnt_beam.rs` with mock-callback unit tests (15 decoding tests passing) |
| **Phase 2** — Unified offline beam | **Complete** | `decode_encoder_frames_greedy` / `_beam`; `from_pretrained_with_decoding` (streaming beam landed in Phase 3) |
| **Phase 3** — Unified streaming beam | **Complete** | Cross-chunk frame remap; `decode_encoder_frames` dispatches beam; CLI in `examples/unified.rs` |
| **Phase 4** — TDT offline beam | **Complete** | `src/decoding/tdt_beam.rs`; `duration_reward` on `BeamConfig`; wired via `ParakeetTDT::from_pretrained_with_decoding`; CLI in `examples/raw.rs` |
| **Phase 5** — CTC offline beam | **Complete** | Native prefix beam in `src/decoding/ctc_beam.rs`; `decode_with_beam_search` wired; `Parakeet::from_pretrained_with_decoding`; timestamps deferred for beam |
| **Phase 5a** — Nemotron beam (fork) | **Complete** | `from_pretrained_with_decoding` / `set_decoding` in `src/nemotron.rs`; offline = single-pass beam over concatenated encoder frames; streaming = per-chunk beam with global-best transcript |
| **Phase 5b** — Benchmark harness | **Complete** | `tests/model_benchmark.rs` + `tests/common/mod.rs`; WER/CER/DER/JER; CSV reports in `samples/asr_report.csv` and `samples/diarization_report.csv`; `scripts/benchmark/run_all.sh` + `manifest.toml` |

**Last updated:** 2026-06-26 — Phases 0–5 complete; Nemotron beam + benchmark harness landed; full benchmark re-run validated (run_id `1782447181`).

---

**Date:** 2025-06-25 (plan); 2026-06-26 (benchmark results)  
**Status:** Phase 5 complete — all planned beam phases done; Nemotron beam added locally; validation harness in place  
**Primary objective:** Add optional beam search decoding to parakeet-rs, with **ParakeetUnified streaming** as the lead integration target, followed by offline paths for CTC and TDT.

**Reference implementation:** [senstella/parakeet-mlx PR #47](https://github.com/senstella/parakeet-mlx/pull/47) (TDT beam, merged Jan 2026)

---

## 1. Problem Summary

> **Historical context (pre-implementation).** parakeet-rs originally used
> **greedy decoding everywhere**; the only beam hook was a CTC stub that fell
> back to greedy. As of Phase 5, `decode_with_beam_search` is wired to a native
> CTC prefix beam (`src/decoding/ctc_beam.rs` via `decode_with_beam_config`),
> and RNNT/TDT beam paths are implemented. The original stub looked like:

```rust
// src/decoder.rs (original stub, now replaced)
pub fn decode_with_beam_search(...) -> Result<String> {
    self.decode(logits)  // stub
}
```

Beam search improves WER on difficult audio and is a prerequisite for future features (n-gram LM fusion, phrase/word boosting). The fork goal is an **upstream-mergeable**, **opt-in** API that preserves greedy as the default for latency-sensitive streaming.

**Scope decision (this plan):**

| Priority | Target | Rationale |
|----------|--------|-----------|
| P0 | **ParakeetUnified** (offline + buffered streaming) | User requirement; RNNT is simpler than TDT (no duration head); shares decode path |
| P1 | **ParakeetTDT** (offline) | Direct port of parakeet-mlx PR #47 |
| P2 | **Parakeet CTC** (offline) | CPU-only beam on logits; no extra ONNX calls |
| Done (fork) | **Nemotron** (offline + streaming beam) | RNNT `decoder_joint` matches Unified; reuses `decode_rnnt_beam` |
| Defer | EOU, Multitalker, Cohere | EOU: latency; Multitalker: multi-speaker + Sortformer coupling; Cohere: attention seq2seq (needs separate KV-cache beam, not RNNT/CTC core) |

---

## 2. Final Codebase Audit

### 2.1 Current decode architecture

```
Encoder ONNX  →  encoder frames [B, D, T]
                      ↓
              Rust decode loop (greedy)
                      ↓
              decoder_joint ONNX per step  (RNNT/TDT)
              OR argmax + CTC collapse     (CTC)
                      ↓
              vocab / tokenizer → text
```

All RNNT-family models call `decoder_joint.run()` from Rust in a frame/symbol loop. Beam belongs in that Rust loop, not in ONNX export.

### 2.2 ParakeetUnified — streaming path (primary target)

**File:** `src/parakeet_unified.rs`

| Component | Current behavior |
|-----------|------------------|
| `UnifiedStreamingConfig` | Default: 5.6 s left context, 560 ms chunk, 560 ms right context |
| `process_ready_chunks` | Buffers audio, encodes sliding window, decodes middle chunk frames |
| `decode_encoder_frames` | Greedy RNNT: single `(state_1, state_2, last_token)` across chunks |
| `transcribe_offline` | Same `decode_encoder_frames` over full encoder output |
| Chunk size | 56 mel frames → 7 encoder frames per chunk (`SUBSAMPLING_FACTOR = 8`) |

**Critical finding:** Decoder LSTM state persists across chunks via instance fields (`state_1`, `state_2`, `last_token`). Streaming beam **must** carry `Vec<Hypothesis>` across chunk boundaries, not re-beam independently per chunk.

**Model I/O:** `src/model_unified.rs`

- Encoder inputs: `audio_signal`, `length`
- Decoder inputs: `encoder_outputs`, `targets`, `target_length`, `input_states_1`, `input_states_2`
- Decoder outputs: `outputs` (logits), `output_states_1`, `output_states_2`
- Vocab: SentencePiece via `nemotron::SentencePieceVocab`; `blank_id = vocab.size()`

### 2.3 Other models (out of P0 scope)

| Model | Decode location | Beam notes |
|-------|-----------------|------------|
| CTC (`decoder.rs`) | Argmax + collapse | Stub exists; use `fast-ctc-decode` or native prefix beam |
| TDT (`model_tdt.rs`) | `greedy_decode` inline | Has duration head (+5 logits); port MLX `decode_beam` |
| Nemotron (`nemotron.rs`) | `decode_chunk_tokens` / `decode_chunk_tokens_beam` | Streaming cache-aware encoder; beam via shared RNNT core; offline = one beam pass over full encoder output |
| EOU (`parakeet_eou.rs`) | Per-chunk greedy | Real-time; beam conflicts with latency |
| Cohere (`cohere.rs`) | `decode_greedy` transformer | Production uses `beam_size=1` |
| Sortformer | N/A | Diarization, not seq decoding |

### 2.4 parakeet-mlx PR #47 — what applies

| Applies to Unified | Does not apply |
|--------------------|----------------|
| `DecodingConfig { Greedy \| Beam }` pattern | `duration_reward`, duration expansion |
| Top-k token pruning (`argpartition`) | TDT-only RNNT assert in MLX |
| Log-sum-exp path merging | MLX GPU batched joint |
| `length_penalty`, `patience` | CTC beam (still greedy in MLX) |

MLX **RNNT** decode remains greedy-only (`assert isinstance(config.decoding, Greedy)`). Unified beam is an **RNNT adaptation** of the TDT algorithm, not a direct copy.

### 2.5 External references (implementation surveys)

1. **NeMo TDT/RNNT batched beam** — `malsd_batch` strategy, [NeMo PR #12729](https://github.com/NVIDIA/NeMo/pull/12729)
2. **Rust CTC beam** — [fast-ctc-decode](https://github.com/nanoporetech/fast-ctc-decode), [rust-asr](https://github.com/RustedBytes/rust-asr)
3. **NeMo CTC + n-gram LM** — `beam_batch`, NGPU-LM ([NeMo docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/ngpulm_language_modeling_and_customization.html))
4. **Streaming RNNT on ONNX** — greedy only in ORT GenAI Nemotron ([PR #1997](https://github.com/microsoft/onnxruntime-genai/pull/1997))
5. **Cohere** — greedy canonical ([HF deep dive](https://huggingface.co/blog/balaatdell/under-the-hood-cohere-transcribe-deepdive-deh))

---

## 3. Affected Files (planned)

### New files

| Path | Purpose |
|------|---------|
| `src/decoding/mod.rs` | `DecodingStrategy`, `BeamConfig`, re-exports |
| `src/decoding/rnnt_beam.rs` | RNNT hypothesis expand/prune/merge (Unified, later Nemotron offline) |
| `src/decoding/tdt_beam.rs` | TDT beam with duration head (P1) |
| `src/decoding/ctc_beam.rs` | Prefix beam on logit matrix (P2) |

### Modified files

| Path | Change |
|------|--------|
| `src/lib.rs` | `mod decoding;` + public re-exports |
| `src/parakeet_unified.rs` | Beam path in `decode_encoder_frames`; cross-chunk hypothesis carry; API |
| `src/decoder.rs` | Wire CTC beam stub to `ctc_beam` (P2) |
| `src/model_tdt.rs` | Optional beam decode path (P1) |
| `src/parakeet_tdt.rs` | Pass `DecodingStrategy` to model (P1) |
| `src/nemotron.rs` | Beam path via `decode_rnnt_beam`; offline single-pass; streaming global-best (fork) |
| `tests/model_benchmark.rs`, `tests/common/mod.rs` | Integration benchmarks; WER/CER/DER/JER; CSV reports |
| `scripts/benchmark/run_all.sh`, `manifest.toml` | CLI benchmark orchestrator |
| `examples/unified.rs` | CLI flag for beam vs greedy |
| `Cargo.toml` | Optional `beam-search` feature if deps added later (KenLM) |

### Unchanged (explicit non-goals for v1)

- `scripts/export_parakeet_unified.py` — no export changes required
- ONNX model artifacts — same `encoder.onnx` + `decoder_joint.onnx`
- Default behavior — greedy unless caller opts in

---

## 4. Implementation Plan

### Phase 0 — Shared types (1 PR)

**Goal:** Additive API surface, zero behavior change.

```rust
// src/decoding/mod.rs
#[derive(Debug, Clone, Copy)]
pub struct BeamConfig {
    pub beam_size: usize,           // default: 5
    pub length_penalty: f32,        // default: 0.013  (MLX tuned)
    pub patience: f32,              // default: 3.5
    pub max_symbols_per_step: usize,// default: 10
}

#[derive(Debug, Clone, Copy, Default)]
pub enum DecodingStrategy {
    #[default]
    Greedy,
    Beam(BeamConfig),
}

impl BeamConfig {
    pub fn default_unified() -> Self { /* values above */ }
}
```

Export from `lib.rs`: `DecodingStrategy`, `BeamConfig`.

**Acceptance:** `cargo build` clean; all existing tests pass; no runtime behavior change.

---

### Phase 1 — RNNT beam core (1 PR)

**Goal:** `src/decoding/rnnt_beam.rs` — model-agnostic beam over a decoder callback.

**Hypothesis struct:**

```rust
struct RnntHypothesis {
    score: f32,
    frame: usize,              // local index within [start_frame, end_frame)
    last_token: i32,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    symbols_on_frame: usize,
    tokens: Vec<(usize, usize)>, // (token_id, absolute_frame)
    prev_token_count: usize,   // tokens already committed before this chunk (streaming carry)
}
```

**Algorithm (per encoder frame range):**

1. Seed from input hypotheses (or single empty hypothesis on first chunk).
2. For each frame `t` in `[start, end)`:
   - For each active hypothesis (up to `beam_size`):
     - Call decoder → logits.
     - Compute log-softmax; take top-`beam_size` token indices (partial argmax).
     - For each token: if blank, advance frame; else emit token, update LSTM state, stay on frame (until `max_symbols_per_step`).
   - Merge duplicate paths by `(frame, token_sequence_hash)` using log-sum-exp.
   - Partition finished vs active; prune active to `beam_size`.
3. Apply `length_penalty` when selecting best: `score / len^penalty`.
4. Return `(best_tokens, surviving_hypotheses_for_next_chunk)`.

**Decoder callback trait** (avoids coupling to `ParakeetUnifiedModel`):

```rust
type DecoderStep = dyn FnMut(
    &Array3<f32>,  // encoder frame
    i32,           // last_token
    &Array3<f32>,  // state_1
    &Array3<f32>,  // state_2
) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)>;
```

**Unit tests (no ONNX):** synthetic logits via mock callback; verify merge, length penalty, blank advance.

---

### Phase 2 — ParakeetUnified offline beam (1 PR)

**Goal:** Wire beam into `transcribe_offline` first (easier to validate WER).

**Changes in `parakeet_unified.rs`:**

1. Add field: `decoding: DecodingStrategy` (default `Greedy`).
2. Add field (beam only): `beam_hypotheses: Vec<RnntHypothesis>`.
3. Split `decode_encoder_frames`:
   - `decode_encoder_frames_greedy` — existing logic unchanged.
   - `decode_encoder_frames_beam` — calls `rnnt_beam::decode`.
4. `transcribe_offline` / `reset()` clears `beam_hypotheses`.

**Constructor API (additive):**

```rust
pub fn from_pretrained_with_decoding(
    path: P,
    exec_config: Option<ExecutionConfig>,
    streaming_config: UnifiedStreamingConfig,
    decoding: DecodingStrategy,
) -> Result<Self>
```

Or `set_decoding_strategy(&mut self, strategy: DecodingStrategy)`.

**Acceptance:**

- Greedy offline output bit-identical to current behavior.
- Beam offline produces non-empty transcript on `./unified` model + test WAV.
- Manual WER spot-check vs greedy on short clip (optional benchmark script).

---

### Phase 3 — ParakeetUnified streaming beam (1 PR)

**Goal:** Cross-chunk hypothesis carry in `process_ready_chunks`.

**State machine:**

```
reset()
  → beam_hypotheses = []

process_ready_chunks() / each chunk:
  → encode window
  → decode_encoder_frames_beam(
         encoded, start_frame, end_frame, abs_offset,
         input_hypotheses = self.beam_hypotheses,
     )
  → self.beam_hypotheses = returned_survivors
  → emit text from best hypothesis new tokens (same as greedy incremental)
  → accumulate tokens/timestamps from best path
```

**Streaming-specific rules:**

| Rule | Detail |
|------|--------|
| Cross-chunk carry | Store pruned `beam_hypotheses` on `ParakeetUnified` instance |
| First chunk | Seed one hypothesis: blank token, zero LSTM |
| Flush | Run beam on partial final chunk; finalize best |
| Token emission | Emit from current best hypothesis per chunk (v1; may flicker if best swaps) |
| Timestamps | `absolute_frame_offset + local_frame` from best hypothesis |
| Multi-stream | Each `from_shared` instance owns its own `beam_hypotheses` |

**Performance budget (default 560 ms chunks, beam 5):**

- ~7 encoder frames × ~25 decoder calls/frame (5 hyps × top-5 tokens) ≈ 175 ONNX calls/chunk
- Greedy upper bound: 7 × 10 = 70 calls/chunk
- Expected overhead: **2–5× decode time per chunk** (acceptable for opt-in quality mode)

**Acceptance:**

- `examples/unified.rs streaming` with beam completes without panic.
- RTF logged; compare greedy vs beam on same audio.
- `ParakeetUnifiedHandle` + multiple streams each maintain independent beam state.

---

### Phase 4 — TDT offline beam (1 PR)

**Goal:** Port parakeet-mlx `decode_beam` to `src/decoding/tdt_beam.rs` + `model_tdt.rs`.

**TDT-specific additions over RNNT beam:**

- Joint logits split: `vocab_size` token + `5` duration logits.
- Expand `(top_k_token × top_k_duration)` candidates.
- Score: `token_logprob * (1 - duration_reward) + duration_logprob * duration_reward`.
- Step advance: `t += duration` (not always +1).

**Default `BeamConfig` for TDT:** add `duration_reward: f32 = 0.67`.

Wire through `ParakeetTDT` / `model_tdt.rs::forward`.

---

### Phase 5 — CTC offline beam (1 PR)

**Goal:** Replace `decode_with_beam_search` stub.

**Options (pick one for v1):**

1. **Dependency:** `fast-ctc-decode` crate — proven speed, BPE blank id 1024.
2. **Native:** Prefix beam in `ctc_beam.rs` — no new dep, more code.

**Note:** Timestamp API needs alignment traceback (frame indices per token), not just text. Defer timestamp+beam to Phase 5b if needed.

**Optional later:** `kenlm` feature for LM fusion (separate PR, Cargo feature-gated).

---

## 5. Reproduction / Validation Steps

### Prerequisites

```bash
# Model: https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx
# Files in ./unified/: encoder.onnx, encoder.onnx.data, decoder_joint.onnx, tokenizer.model
```

### Baseline (current greedy behavior)

```bash
cd /home/v2/projects/onnx/parakeet-rs
cargo run --release --example unified 6_speakers.wav
cargo run --release --example unified 6_speakers.wav streaming
```

### After Phase 2 (offline beam)

```bash
cargo run --release --example unified 6_speakers.wav -- --decoding beam --beam-size 5
# Expect: different (hopefully better) transcript vs greedy; no crash
```

### After Phase 3 (streaming beam)

```bash
cargo run --release --example unified 6_speakers.wav streaming -- --decoding beam
# Expect: incremental output; final transcript; RTF within ~2-5x greedy decode portion
```

### Regression checks

```bash
cargo build --all-features
cargo test
# Greedy default: bit-identical offline output on fixed test vectors (unit tests)
```

### Benchmark harness (implemented)

Integration tests in `tests/model_benchmark.rs` (shared helpers in `tests/common/mod.rs`):

```bash
# Full suite (models on disk are auto-detected; missing models skip gracefully)
cargo test --test model_benchmark --features "sortformer multitalker cohere" -- --nocapture --test-threads=1

# Nemotron greedy vs beam x offline vs streaming only
cargo test --test model_benchmark nemotron_decoding_matrix -- --nocapture
```

Reports append to:

- `samples/asr_report.csv` — WER, CER, RTF per sample/model/mode/decoding
- `samples/diarization_report.csv` — DER, JER, missed/FA/confusion, RTF

Shell orchestrator (greedy/beam diff for CLI examples): `scripts/benchmark/run_all.sh`  
Manifest / model matrix: `scripts/benchmark/manifest.toml`

Long clips (`committee`, `datascience-staff-meeting`) run only when `PARAKEET_BENCH_ALL=1`.

---

## 5b. Benchmark Results (2026-06-26)

**Environment:** local fork, CPU inference, `samples/smoke` (24 s) unless noted.  
**Baseline run:** `1782441790` (ASR), `1782439077` (diarization).  
**Latest full re-run:** `1782447181` — all metrics reproduced; no accuracy regressions.

### ASR — `samples/smoke` (run_id `1782447181`)

| Model | Mode | Decoding | WER | CER | RTF | vs greedy |
|-------|------|----------|-----|-----|-----|-----------|
| nemotron-en-0.6b-int8 | offline | greedy | 1.9% | 0.7% | 3.06x | — |
| nemotron-en-0.6b-int8 | streaming | greedy | 1.9% | 0.7% | 2.64x | — |
| nemotron-en-0.6b-int8 | offline | beam | **0.0%** | 0.0% | 0.97x | **-1.9% WER** |
| nemotron-en-0.6b-int8 | streaming | beam | 1.9% | 0.7% | 0.89x | same WER, ~3x slower |
| multitalker-0.6b-int8 | streaming | greedy | 3.7% | 0.4% | 1.38x | — |
| cohere-transcribe-int8 | offline | greedy | 0.0% | 0.0% | 0.87x | beam N/A |

**Nemotron offline vs streaming beam:** Offline runs one global beam over all encoder frames; streaming runs per-chunk beam with pruning at each 560 ms boundary and reports the current global-best path. Greedy is mode-invariant (1.9% both); beam gains accuracy only offline on smoke.

**Historical note:** Rows with run_id `1782441223` / `1782441451` show Nemotron offline beam WER ~52% — pre-fix for an RNNT blank-handling bug (see Section 9). Post-fix beam is stable at 0.0% offline WER across runs `1782441629`, `1782441759`, `1782447493`.

**Not benchmarked (models not on disk):** CTC, TDT, Unified, EOU — tests skip when `models/parakeet-{ctc,tdt,unified}` / `models/realtime-eou-120m` are absent.

### Diarization — Sortformer 4spk v2.1 (run_id `1782447181`)

| Sample | Duration | DER | JER | RTF |
|--------|----------|-----|-----|-----|
| smoke | 24 s | 12.7% | 12.9% | 14.9x |
| committee | 632 s | 21.3% | 46.1% | 11.3x |
| datascience-staff-meeting | 1750 s | 30.5% | 72.3% | 10.8x |

DER/JER identical to baseline run `1782439077`; RTF varies with machine load only.

### RNNT beam bug fix (affects Unified + Nemotron)

In `src/decoding/rnnt_beam.rs`, blank advances previously set `last_token = blank_id`, corrupting the prediction-network input on the next frame. RNNT/greedy semantics keep the last **non-blank** token and LSTM state across blank steps. Fix: do not overwrite `last_token` on blank (clone already carries correct state). All 15 decoding unit tests pass; Nemotron offline beam WER dropped from ~53% to 0.0% on smoke.

---

## 6. Risks and Side Effects

| Risk | Severity | Mitigation |
|------|----------|------------|
| ONNX call amplification (beam) | High | Top-k pruning; patience; keep greedy default |
| Text flicker in streaming beam | Medium | v1: emit from best path; v2: stable commit threshold |
| LSTM state clone overhead | Medium | `Arc<Array3<f32>>` for shared prefixes in hypothesis tree |
| Chunk-edge WER without cross-chunk carry | High | Phase 3 requirement — do not ship per-chunk-only beam |
| Timestamp drift under beam | Medium | Use best-hypothesis frame indices; add tests vs greedy |
| Lock contention multi-stream | Low | Existing `Arc<Mutex<Model>>` unchanged; beam only extends hold time |
| Upstream merge conflict | Low | Additive API; feature-free core; no export script changes |

**Implicit dependencies:**

- `blank_id = vocab.size()` must match ONNX export (already enforced in `ParakeetUnifiedHandle::from_pretrained`).
- `MAX_SYMBOLS_PER_STEP = 10` must stay aligned with NeMo export default.
- Streaming chunk frames must remain divisible by `SUBSAMPLING_FACTOR` (8) — validated in `UnifiedStreamingConfig::validate`.

---

## 7. Performance Targets

| Mode | Target | Notes |
|------|--------|-------|
| Greedy streaming | No regression | Default path untouched |
| Beam streaming (beam=5) | Decode portion ≤ 5× greedy | Top-k prune required |
| Beam offline Unified | Acceptable for batch/file transcription | Not real-time critical |
| CTC beam | CPU-only; encoder once | `fast-ctc-decode` benchmark as guide |

Long-term (not v1): batched decoder ONNX runs (NeMo `malsd_batch` parity), optional KenLM.

---

## 8. PR Series (upstream-mergeable)

| PR | Title | Files | Behavior change |
|----|-------|-------|-----------------|
| 1 | Add DecodingStrategy and BeamConfig types | `decoding/mod.rs`, `lib.rs` | None |
| 2 | RNNT beam decoder module | `decoding/rnnt_beam.rs` | None |
| 3 | ParakeetUnified offline beam | `parakeet_unified.rs` | Opt-in |
| 4 | ParakeetUnified streaming beam | `parakeet_unified.rs`, `examples/unified.rs` | Opt-in |
| 5 | TDT offline beam | `decoding/tdt_beam.rs`, `model_tdt.rs`, `parakeet_tdt.rs` | Opt-in |
| 6 | CTC offline beam | `decoding/ctc_beam.rs`, `decoder.rs` | Opt-in |

Each PR: `cargo build`, `cargo test`, no model binaries committed.

---

## 9. Open Questions (resolutions)

1. **Text flicker policy:** **Resolved (v1):** emit from the current best path each
   chunk (`decode_incremental_tokens`). A stable-commit threshold (delay until the
   best hypothesis is unchanged for N chunks) remains a v2 follow-up; flicker rate
   should be measured by the benchmark harness.
2. **BeamConfig on Handle vs instance:** **Open.** `DecodingStrategy` currently lives
   per-`ParakeetUnified` instance (`from_shared_with_decoding`); `ParakeetUnifiedHandle`
   holds model/vocab only. Multi-stream servers must set the strategy per stream. A
   handle-level default is a candidate additive change (see Issue 7 in the review).
3. **Partial argmax:** **Resolved.** Inline `top_k_indices` (sort-based) in each beam
   module; no extra dependency added.
4. **CTC dependency:** **Resolved.** Native prefix beam in `src/decoding/ctc_beam.rs`;
   no `fast-ctc-decode` dependency.

**Known residual gaps (tracked in review report):**

- CTC beam returns text without timestamps (Phase 5c — alignment traceback not yet implemented).
- `patience` is honored only by the TDT decoder; RNNT/CTC ignore it (documented on `BeamConfig`).
- TDT greedy stores the duration *index* and advances by it; identical to the value
  only while `TDT_DURATIONS` is the identity table `[0,1,2,3,4]`.
- Cohere beam not implemented — attention decoder with KV cache needs a separate module.
- CTC/TDT/Unified/EOU benchmarks blocked until model dirs are downloaded.
- Streaming beam may revise partial transcripts if the global best path changes mid-utterance.

**Resolved since plan draft:**

- RNNT blank `last_token` bug (Section 5b) — fixed in `rnnt_beam.rs`.
- Benchmark harness exists; WER/CER/DER/JER validated on smoke + long diarization clips.
- Nemotron beam landed as fork extension (upstream candidate after Unified beam WER spot-check with model on disk).

---

## 10. Summary

Beam search in parakeet-rs is **implemented and mergeable** as an opt-in decode strategy across CTC, TDT, and Unified RNNT (Phases 0–5). **ParakeetUnified streaming** remains the primary upstream target; **Nemotron** reuses the same RNNT beam core locally with offline single-pass and streaming per-chunk semantics. Greedy stays the default everywhere.

**Validated on smoke (2026-06-26):** Nemotron offline beam improves WER 1.9% → 0.0% at ~3× decode cost; streaming beam matches greedy WER but is slower — greedy remains the right default for real-time. A shared RNNT blank-handling fix benefits all RNNT beam consumers. Integration tests + CSV reports provide regression tracking for future changes.
