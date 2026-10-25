# ASR Performance and Feature-Fit Implementation Plan - parakeet-rs

**Last updated:** 2026-10-25
**Current phase:** Phase 7 partial. The steady-state Cohere decoder matches the merged graph; mean step time dropped about 10 ms and RTFx did not improve.

## Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 0** - Baseline benchmarks (thread sweep, Ark/Granite coverage) | Done | Ark and Granite smoke tests pass. Sweep is in section 10.1. 32 threads is at least 1.3x the 4-thread RTFx for Ark, Granite, and Nemotron; Cohere peaks at 16 threads (1.21x, under the gate) and slows at 32. Defaults stay 4 threads. README and examples were not changed. |
| **Phase 1** - Ark correctness and decode hot path | Done | `ArkLongAudioPolicy` default `Truncate` (stderr warning; no log crate). `transcribe_long` splits on the quietest 100 ms frame in the last 5 s. Audio-token count comes from the encoder output. Argmax uses a `Vec<bool>` mask. `set_repeat_guard` default off. Smoke WER 0.0000, RTFx 1.31 at 4 threads (baseline 1.32; a cold run the same day was 1.21). Argmax mask 12.45x vs HashSet. 72 s `transcribe_long`: 162 words vs 54. `Error` policy rejects that clip. |
| **Phase 2** - Cohere decode copies and long-form helper | Done | Encoder K/V reused after the first decoder step. Smoke transcript byte-identical. 4-thread RTFx 0.64 to 0.71; mean decoder step 390.458 ms to 317.861 ms. `transcribe_audio_long` on committee: WER 0.0625. |
| **Phase 3** - Nemotron streaming `flush()` and feature-cache check | Done | `flush` / `flush_with_tokens` pad the streaming tail like the offline last chunk. Smoke streaming WER stayed 0.0185. RTFx 2.37-2.38 at 4 threads. Mel is 0.3% of staged chunk time, so no incremental STFT. 1 s flush vs offline WER 0.0000. |
| **Phase 4** - Granite long-audio validation and beam memory | Done | Greedy smoke WER stayed 0.0370 (RTFx 16.43 at 4 threads). Smoke beam text byte-identical. One-shot greedy completed committee (632.28 s, WER 0.0959, RTFx 17.76) and datascience-staff-meeting (1750.30 s, WER 0.1131, RTFx 17.30). Committee beam peak RSS 3,099,844 kB to 2,616,988 kB (15.6%, short of the 40% gate). |
| **Phase 5** - Multitalker Sortformer stride decoupling | Done | Opt-in `DiarStrategy::Buffered` (default stays `PerChunk`). Smoke, 2 speakers, 4 threads: PerChunk RTFx 1.04-1.10, WER 0.0000, 25 Sortformer runs; Buffered RTFx 2.85-3.24, WER 0.0000, 3 runs (8.33x fewer runs, about 2.7x RTFx). 1.5x gate met. Encoder batch is fixed at 1. Committee was not run. |
| **Phase 6** - Export-side work (Ark last-token logits, cache buckets) | Done | No script and no Rust change. Ark `llm_kv_cpu_fp32_int8.onnx` already slices the last hidden state before the lm_head (`/Slice` starts=-1, ends=INT64_MAX, axis 1). Cache buckets are a no-go. Cohere encoder K/V projections are already inside an `If` and do not rerun after step 0; the steady-state cost is MatMulNBits plus an `If` output copy. |
| **Phase 7** - Cohere steady-state decoder | Partial | `scripts/export_cohere_steady_decoder.py` replaces the two cross-attn `If` nodes with direct `past_key_values.*.encoder.*` reads. Rust loads `decoder_model_steady[_quantized].onnx` after step 0 when that file is present (`CohereASR::set_use_steady_decoder`, default on). Smoke transcript byte-identical. Same-run steady steps 326.7 ms vs merged tail 337.0 ms; smoke RTFx stayed 0.67. Committee long-form WER stayed 0.0625 (RTFx 0.69 vs Phase 2's 0.64). The profiled 156 ms `If` copy overlaps MatMulNBits. |

**Source:** feasibility review of the five ASR models under `models/` (Nemotron EN, Multitalker, Cohere Transcribe, Ark ASR, Granite TurboCTC).
**Primary objective:** raise throughput and fix correctness gaps in the existing model families with additive, upstream-mergeable changes.
**Non-goals:** new model families, default behavior changes for existing users, vendoring weights.

---

## 1. Baseline (from `samples/asr_report.csv`, 24 s smoke clip, CPU, 56 cores, `intra_threads = 4`)

| Model | Mode | Decoding | RTFx | WER |
|-------|------|----------|------|-----|
| Nemotron EN int8 | offline | greedy | about 3.0 | 0.0185 |
| Nemotron EN int8 | streaming | greedy | about 2.6 | 0.0185 |
| Nemotron EN int8 | offline / streaming | beam | about 0.95 | 0.0000 to 0.0185 |
| Multitalker int8 | streaming | greedy | about 1.4 to 1.5 | 0.0370 |
| Cohere int8 | offline | greedy | 0.64 (re-run 2026-10-24; older CSV rows about 0.87) | 0.0000 |
| Ark ASR int8 | offline | greedy | 1.32 | 0.0000 |
| Granite TurboCTC int8 | offline | greedy | 17.44 | 0.0370 |

RTFx = audio seconds / processing seconds (CSV column `rtf`). The smoke clip is 24.00 s. Ark and Granite rows are `samples/asr_report.csv` run `1792891921` / `1792891932` (`intra_threads` 4). The Cohere re-run is `cohere_offline_wer` the same day (proc 37.230 s, RTFx 0.64, run `1792892389`); the thread sweep's matching 4-thread row was 0.56. Nemotron was not re-run outside the sweep: offline greedy at 4 threads measured RTFx 2.46 there, versus about 3.0 in the older CSV rows. Later phases should compare against these 2026-10-24 numbers.

New `asr_report.csv` rows append `intra_threads`. The file already existed, so its header was left unchanged (it still has no `intra_threads` name). Older rows also predate the `ep` column. Read the trailing integer on rows that have it.

---

## 2. Design Rules (from AGENTS.md)

1. Smallest diff that achieves the goal; prefer additive APIs and optional config fields.
2. No hardcoded local paths; model dirs come from arguments or `PARAKEET_*` env overrides in tests.
3. Defaults for existing users must not change unless a phase explicitly says so and a benchmark justifies it.
4. Feature-gated modules (`ark-asr`, `cohere`, `multitalker`, `sortformer`) keep their gates.
5. One logical change per commit; no model binaries; no `scripts/__pycache__/`.
6. Run `cargo build --all-features` and `cargo test` before any upstream proposal.

---

## 3. Phase 0 - Baseline Benchmarks

**Goal:** measure before changing code. All later phases compare against these numbers.

### 3.1 Work items

| Item | Files | Detail |
|------|-------|--------|
| Ark benchmark test | `tests/model_benchmark.rs`, `tests/common/mod.rs` | New `ark_offline_wer` (feature `ark-asr`), modeled on `cohere_offline_wer`. Env override `PARAKEET_ARK_DIR`, default `models/ark-asr-0.6b-int8-onnx`. Skips when weights absent. |
| Granite benchmark test | same | New `granite_ctc_offline_wer` using `Parakeet::from_pretrained` on `models/granite-speech-5.0-470m-turboctc-onnx`. Env override `PARAKEET_GRANITE_DIR`. Skips when weights absent. |
| Thread sweep | `tests/model_benchmark.rs` | New `asr_thread_sweep` test, `#[ignore]`d. Reads `PARAKEET_SWEEP_THREADS` (comma list, default `4,8,16,32`) and `PARAKEET_SWEEP_MODELS` (default `cohere,ark,granite,nemotron`). Builds each model with `ExecutionConfig::with_intra_threads(n)` and writes `samples/asr_thread_sweep.csv`. |
| CSV columns | `tests/common/mod.rs` | Add an `intra_threads` column to `asr_report.csv` rows (append-only so old rows stay parseable). |
| Run notes | this file, section 9 | Record the final sweep table here. |

### 3.2 Tests

- `ark_offline_wer`: asserts WER on the 20 s smoke clip is at or below `SMOKE_MAX_WER` (default 0.15), matching the existing smoke policy.
- `granite_ctc_offline_wer`: same assertion.
- Both print the skip notice and pass when weights are absent.

### 3.3 Benchmarks and acceptance

- Run one process at a time (CPU contention invalidates RTFx).
- Command:
  `cargo test --release --test model_benchmark --features "sortformer multitalker cohere ark-asr" asr_thread_sweep -- --ignored --nocapture`
- **Acceptance:** `samples/asr_thread_sweep.csv` exists with at least three thread counts per model, and the best thread count per model is recorded in section 9.
- **Decision gate:** if a thread count gives at least 1.3x over 4 threads for a model, document the recommendation in the example and README for that model. Do not change `ModelConfig::default()` in this plan; propose it separately with the data.

---

## 4. Phase 1 - Ark ASR Correctness and Decode Hot Path

**Files:** `src/ark_asr.rs`, `src/model_ark.rs`.

### 4.1 Work items

| # | Change | Detail |
|---|--------|--------|
| 1.1 | Replace silent truncation | In `transcribe_samples_inner`, audio longer than `max_audio_seconds` currently hits `audio.truncate(...)`. Add `ArkLongAudioPolicy { Error, Truncate }` set via `set_long_audio_policy` (default `Truncate` to preserve behavior, with a `log`-level warning when it triggers). Add an optional `transcribe_long` that splits into windows of `max_audio_seconds` at the lowest-energy point in the last 5 s and joins text with a space. |
| 1.2 | Size placeholders from the encoder | Run `encode_audio` before building `input_ids`; use `audio_embeds.shape()[1]` as `n_audio_tokens`. Remove the `audio_token_count` estimate and the `let _ = needed;` branch. Return `Error::Model` if the prompt would exceed `max_total_len`. |
| 1.3 | Boolean mask for argmax | Replace `HashSet<u32>` lookups in `argmax_masked` with a `Vec<bool>` (length `block_from`) built once in `from_pretrained`. |
| 1.4 | Optional repetition guard | Reuse a small n-gram repeat check (same idea as Cohere's `find_ngram_repetition`) behind `set_repeat_guard(bool)`, default off. |

### 4.2 Tests (unit, no weights)

- `argmax_masked_matches_hashset_reference`: random logits and a random bad-id set; assert the boolean-mask version equals a straightforward reference.
- `argmax_masked_all_masked_errors`: all ids masked returns `Error::Model`.
- `assemble_prompt_ids_uses_actual_count`: placeholder count equals the supplied count, including 0 and 1.
- `long_audio_policy_error_returns_audio_error`: use the policy check function in isolation (factor it into a free function so it is testable without a model).
- `split_at_quiet_point_*`: synthetic signal with a silent gap in the last 5 s; assert the split index falls inside the gap.

### 4.3 Tests (weight-gated, in `tests/model_benchmark.rs`)

- `ark_long_audio_no_truncation`: concatenate the smoke clip 3 times (60 s); assert `transcribe_long` output word count is at least 2.5x the single-clip count and the policy `Error` mode returns an error.

### 4.4 Benchmarks and acceptance

- Re-run `ark_offline_wer`; **WER must not regress** versus the Phase 0 row.
- Add a micro-benchmark `bench_argmax_masked` (criterion-free: a `#[test] #[ignore]` that times 1000 iterations over 151,670 logits). **Target:** at least 5x faster than the HashSet version.
- **Acceptance:** no WER regression, no silent truncation in `Error` mode, unit tests green.

---

## 5. Phase 2 - Cohere Decode Copies and Long-Form Helper

**Files:** `src/model_cohere.rs`, `src/cohere.rs`.

### 5.1 Work items

| # | Change | Detail |
|---|--------|--------|
| 2.1 | Stop re-extracting cross-attention K/V | After the first decoder step, keep `present.{i}.encoder.{key,value}` as the cached tensors and do not call `extract_cache` for them again. Only `present.{i}.decoder.*` is read each step. |
| 2.2 | Read layer and head counts from the graph | Derive `NUM_DECODER_LAYERS`, `NUM_KV_HEADS`, `HEAD_DIM` from `decoder.inputs()` names and shapes (`past_key_values.N.decoder.key`). Keep the current values as fallbacks. Replace the 32 hand-written `TensorRef` bindings with a loop that builds `Vec<(String, SessionInputValue)>` (same pattern as `ArkModel::run_llm`). |
| 2.3 | Optional long-form API | `transcribe_audio_long(audio, language, punctuation, itn)` splits at 35 s (configurable) using the quiet-point search in the last 5 s of each window, transcribes each chunk, joins with `""` for `ja`/`zh` and `" "` otherwise. Matches the vendor behavior (`max_audio_clip_s = 35`, `overlap_chunk_second = 5`). |
| 2.4 | Correct the doc comment | Update the `TRAINING_CHUNK_SECS` note and `transcribe_audio` docs: audio over about 35 s is not guaranteed to work; recommend `transcribe_audio_long`. |

### 5.2 Tests (unit)

- `cache_names_generated_for_n_layers`: the input-name generator yields `past_key_values.{i}.{decoder|encoder}.{key|value}` for N layers.
- `split_chunks_cover_all_samples`: chunk boundaries are contiguous, none exceeds the max, last chunk is non-empty.
- `join_policy_cjk_vs_latin`: `ja` and `zh` join with `""`; `en` with `" "`.
- `ngram_repetition_detection` (existing) stays green.

### 5.3 Tests (weight-gated)

- `cohere_offline_wer` (existing) must still pass with the same WER.
- `cohere_decode_output_unchanged`: transcribe the smoke clip before and after (store the Phase 0 transcript text in a fixture `samples/smoke/test01_20s_cohere_baseline.txt` only if small; otherwise compare against `test01_20s_transcript.txt` with a WER ceiling). Output text must be identical to the pre-change run on the same weights.
- `cohere_long_form_wer`: use `samples/committee` or another long clip when `PARAKEET_BENCH_ALL=1`; assert WER below the ceiling in `manifest.toml`.

### 5.4 Benchmarks and acceptance

- Re-run the Cohere row of the thread sweep at the best thread count from Phase 0.
- Add a decode-step timer (`#[ignore]` test) that logs mean ms per decoder step before and after.
- **Acceptance:** identical transcript on the smoke clip; per-step time does not increase; RTFx at the chosen thread count is recorded in section 9.

---

## 6. Phase 3 - Nemotron Streaming Flush and Feature-Cache Check

**Files:** `src/nemotron.rs`.

### 6.1 Work items

| # | Change | Detail |
|---|--------|--------|
| 3.1 | Add `flush()` | Pad the remaining buffered audio with zeros to a full chunk (equivalent to NeMo `flush_size_in_secs`), run one encoder + decode step with `chunk_length = pre_encode_cache + remaining_frames`, return the tokens, and mark the stream finished until `reset()`. Mirror the offline path's last-chunk handling. |
| 3.2 | Streaming/offline parity test hook | Expose no new public API beyond `flush()`; use `transcribe_chunk` + `flush` in tests. |
| 3.3 | Evaluate incremental mel (investigation, not a commitment) | Measure how much of per-chunk time is spent in `compute_mel_spectrogram`. Only implement an incremental STFT if it exceeds 10 percent of chunk time. Record the number in section 9. |

### 6.2 Tests

- Unit: `flush_on_empty_buffer_returns_empty`.
- Unit: `flush_is_idempotent_until_reset`.
- Weight-gated: `nemotron_streaming_flush_parity`: feed the smoke clip in 1 s increments, then `flush()`; the concatenated transcript must equal the offline transcript, or differ by at most the WER ceiling already used by `nemotron_streaming_wer`. Also feed a clip whose length is not a multiple of 560 ms and assert the last words are present.
- Extend `nemotron_streaming_wer` to call `flush()` at the end.

### 6.3 Benchmarks and acceptance

- Re-run the Nemotron streaming rows; **RTFx must stay within 5 percent of Phase 0**; WER must not increase.
- Add a timing split (mel, encoder, decoder) printed by an `#[ignore]` test `nemotron_stage_timing`.

---

## 7. Phase 4 - Granite TurboCTC Validation

**Files:** `src/model.rs`, `src/decoding/ctc_beam.rs`, `src/parakeet.rs` (only if needed); docs in `AGENTS.md`.

### 7.1 Work items

| # | Change | Detail |
|---|--------|--------|
| 4.1 | Long-audio test | Run Granite on the longest local clips (committee, datascience) and record peak RSS and WER. If it completes without chunking, update the AGENTS.md "CTC limit" entry to say the 4-5 minute limit applies to Parakeet FastConformer only. |
| 4.2 | Beam memory | Add a top-k-per-frame path in `ctc_beam` that reads logits as a borrowed slice instead of `forward()` + `to_vec()`. Add `ParakeetModel::forward_view` style access (closure over `&[f32]`) so the beam decoder never owns a full `T x 16384` copy. |
| 4.3 | Metadata sanity check | At load, warn if `ctc_encoder_subsample * ctc_frontend_stack_factor` disagrees with the logit/feature frame ratio (already derivable in `logit_shift_samples_for`). |

### 7.2 Tests

- Unit: `ctc_beam_topk_matches_full_logits` using a small synthetic `T x V` matrix; the top-k variant must produce the same hypothesis as the full-copy variant.
- Unit: `logit_shift_for_granite_defaults` (stack 2, subsample 4, hop 160 gives an 80 ms frame).
- Weight-gated: `granite_ctc_offline_wer` (from Phase 0) and `granite_long_audio_completes` (`PARAKEET_BENCH_ALL=1`, asserts completion and WER ceiling).

### 7.3 Benchmarks and acceptance

- Record greedy RTFx, beam RTFx, and peak RSS (via `/usr/bin/time -v` in `scripts/benchmark/run_all.sh` when `PARAKEET_MEASURE_RSS=1`).
- **Acceptance:** beam peak RSS on a 5 minute clip drops by at least 40 percent versus current; greedy output unchanged.

---

## 8. Phase 5 - Multitalker Sortformer Stride Decoupling

**Files:** `src/multitalker.rs`, possibly `src/sortformer.rs` (additive only).

### 8.1 Work items

| # | Change | Detail |
|---|--------|--------|
| 5.1 | Buffer audio for Sortformer | Use `Sortformer::feed_raw` (already present) so diarization runs once per its native window and the resulting 80 ms probability frames are queued. ASR sub-chunks consume queued frames. |
| 5.2 | Lag handling | If predictions for a chunk are not ready yet, hold the ASR chunk until `feed_raw` has produced frames covering it (adds at most one Sortformer window of latency); expose this as `MultitalkerConfig::diar_lookahead` documented with the latency impact. Default preserves the current behavior. |
| 5.3 | Parallel speaker passes (investigation) | Check whether the multitalker encoder graph accepts batch > 1. If yes, batch active speakers; if not, evaluate running per-speaker sessions on a small thread pool. Decision recorded in section 9 before any code is written. |
| 5.4 | Replace constants | Read `VOCAB_SIZE` and `BLANK_ID` from the tokenizer and decoder output shape. |

### 8.2 Tests

- Unit: `derive_speaker_targets` tests on synthetic prediction matrices (shape, threshold, background mask).
- Weight-gated: `multitalker_smoke` (existing) must keep its WER and speaker count.
- New weight-gated `multitalker_diar_calls_reduced`: count Sortformer session runs via a counter exposed under `#[cfg(test)]`; assert the count drops by at least 4x versus the old per-chunk path on the smoke clip.

### 8.3 Benchmarks and acceptance

- Compare RTFx and WER before and after on `smoke` (2 speakers) and `committee` (6 speakers, `PARAKEET_BENCH_ALL=1`).
- **Acceptance:** RTFx improves by at least 1.5x on the 2-speaker clip with no WER regression above 0.5 points.

---

## 9. Phase 6 - Export-Side Work (scripts only)

| # | Change | Detail |
|---|--------|--------|
| 6.1 | Ark last-token logits | Verify with an ONNX graph walk whether `logits` is computed for every sequence position (the output is dynamic over `seq`). If so, add `scripts/export_ark_last_logits.py` (or a patch script, in the style of `patch_ctc_onnx_metadata.py`) that inserts a `Slice` on the final position before the lm_head. The Rust code already takes the last row, so no Rust change is needed. |
| 6.2 | Ark cache buckets | Optional: export 1024-slot variants and let `runtime_manifest.json` list them; the Rust side would pick the smallest bucket that fits `prompt_len + max_new_tokens`. Only pursue if Phase 1 profiling shows attention over 2048 slots matters. |
| 6.3 | Documentation | Document each script header with the expected latency effect, per the existing export-script convention. |

**Tests:** run the patched graph and the original on the smoke clip; transcripts must be identical. **Benchmark:** record prefill time and total RTFx before and after.

### Result (2026-10-24)

6.1 The single-file `llm_kv_cpu_fp32_int8.onnx` (486 MB) already applies the last-token slice. `logits` is `Cast /Cast_3` of the dequantized `/lm_head/MatMul_quant` (`MatMulInteger`, weight `onnx::MatMul_8500_quantized` `[896, 163958]`). The MatMul input is `/Slice_output_0`, produced by `Slice /Slice` on the final RMSNorm `/core/norm/Mul_1_output_0`. Slice inputs are starts `[-1]` (`/Constant_7`), ends `[INT64_MAX]` (`/Constant_8`), axes `[1]` (`/Constant_6`), steps `[1]` (`/Constant_9`). There is no Gather on the sequence axis before the head. The declared output dim stays dynamic (`Castlogits_dim_1`); at runtime, seq 1, 8, and 32 all return logits `[1, 1, 163958]`. `key_delta_*` / `value_delta_*` stay `[batch, seq, 2, 64]`. `extract_last_logits` uses `start = (seq - 1) * vocab`, which is 0 when seq is 1. No `scripts/patch_ark_last_logits.py`. Same-day `ark_offline_wer` (4 threads, original graph): WER 0.0000, RTFx 1.29, proc 18.658 s, 24.00 s smoke, run `1792899602`. Phase 1's same-day figure was RTFx 1.31 (proc 18.353 s). `ark_long_audio_no_truncation` was not re-run.

6.2 No-go on 1024/512 cache-bucket exports. ONNX Runtime, 4 intra threads, all KV-delta outputs requested: decode seq=1 median 143.0 ms; the same step with only 64 mask bits set was 144.4 ms (the mask does not skip the static 2048 work). Prefill seq=32 median 278.8 ms, seq=128 median 721.5 ms, about 4.5 ms per extra token. A profiled seq=1 forward spends about 60 ms in `self_attn` `Expand_1`/`Expand_2` (materializing K/V across all 2048 slots) and about 9 ms in the lm_head. A 512-slot graph would cut that expand roughly 4x, on the order of 50 ms per forward and about 1.2x end-to-end on this clip, but a 24 s prompt already needs a few hundred audio tokens, so the useful buckets are 1024 and 2048 and each graph is another ~486 MB. Not worth a Rust bucket picker.

6.4 Cohere `decoder_model_merged_quantized.onnx` (loaded with `load_external_data=False`) has no `use_cache_branch` input. `/model/is_first_step` is `Equal(past_key_values.0.encoder.key.shape[2], 0)`. That bool feeds two `If` nodes. `/model/if_cross_attn_kv_cache` then-branch is 16 `MatMulNBits` (k and v for 8 layers) that read `encoder_hidden_states`; the else-branch is `Identity` of `past_key_values.*.encoder.*`. Those projections are not in the main graph. `/model/if_cross_attn_public_cache` publishes `present.*.encoder.*` (then: identity of the projection; else: `Slice` to an empty axis-2 range). A later step with past encoder length 2400 ran 0 ms of then-branch kernels. The `If` itself cost 150.9 ms, which is the else-branch `Identity` copies (~9.6 ms x 16). `MatMulNBits` on that step was 229.2 ms (65 nodes), including lm_head 70.3 ms and one `fc1` 11.5 ms; all eight `MultiHeadAttention` nodes were 11.0 ms. Wall time, 4 threads: later step T=256 median 340.4 ms (this is the ~318 ms Rust figure), first step T=256 median 431.8 ms, later T=2400 median 486.0 ms, first T=2400 median 700.3 ms. A drop-in patch cannot delete the `If` without breaking step 0, which is what fills the encoder cache, and removing the copy still leaves ~230 ms of `MatMulNBits`. No script.

---

## 10. Results Log (fill in as phases complete)

### 10.1 Thread sweep (Phase 0)

Measured 2026-10-24 on the 24.00 s smoke clip, CPU, one process, release build. Values are RTFx. WER did not change with thread count (Cohere 0.0000, Ark 0.0000, Granite 0.0370, Nemotron 0.0185). Source: `samples/asr_thread_sweep.csv` run `1792891946`. This is the record section 3.3 asked to keep (that section names "section 9"; the table is here). Nemotron is the offline `transcribe_audio` path, peak-normalized, same as `nemotron_decoding_matrix`.

| Model | 4 | 8 | 16 | 32 | Best |
|-------|---|---|----|----|------|
| Cohere | 0.56 | 0.65 | 0.68 | 0.40 | 16 |
| Ark | 1.26 | 1.67 | 1.89 | 2.08 | 32 |
| Granite | 17.79 | 31.88 | 32.57 | 46.45 | 32 |
| Nemotron | 2.46 | 3.08 | 3.45 | 3.81 | 32 |

Best means the highest RTFx inside `{4, 8, 16, 32}`, not a searched optimum. Ark, Granite, and Nemotron were still rising at 32. Ratio versus that model's own 4-thread RTFx: Cohere 1.21x (16 threads, under the 1.3x gate; 32 threads is slower than 4), Ark 1.65x, Granite 2.61x, Nemotron 1.55x. `ModelConfig::default()` stays 4 intra threads. A README or example note can recommend 32 for Ark, Granite, and Nemotron, and 16 for Cohere, without changing the default.

### 10.2 Per-phase before/after

Phase 1's "before" is the Phase 0 Ark row. Other before-cells stay empty until that phase measures them. The Cohere and Granite greedy figures above are the comparison points for Phases 2 and 4.

| Phase | Metric | Before | After |
|-------|--------|--------|-------|
| 1 | Ark WER / RTFx | 0.0000 / 1.32 at 4 threads (sweep 1.26) | 0.0000 / 1.31 at 4 threads (cold run the same day: 1.21). Argmax mask 12.45x vs HashSet (178.831 ms vs 2227.320 ms, 151670 logits x 1000, min of two release timings). `transcribe_long` on 72.00 s (smoke x3): 162 words vs 54 (3.0x). `ArkLongAudioPolicy::Error` returns `Error::Audio`. |
| 2 | Cohere RTFx / ms per decoder step | 0.64 RTFx at 4 threads (proc 37.230 s, run `1792892389`; sweep 0.56, 16-thread peak 0.68). Pre-change step timer, same instrumentation: 390.458 ms mean, 79 steps, wall 39.988 s, RTFx 0.60 | 0.71 RTFx at 4 threads (proc 33.896 s, WER 0.0000, run `1792894114`). Step timer: 317.861 ms mean (1.23x), 79 steps, wall 34.492 s, RTFx 0.70. Smoke transcript byte-identical to the pre-change run. `transcribe_audio_long` on committee (632.28 s): WER 0.0625, CER 0.0366, RTFx 0.64, proc 983.171 s, ceiling 0.50 (run `1792895199`). Not re-swept at 16 threads. |
| 3 | Nemotron RTFx / mel share of chunk time | Older CSV streaming about 2.60 (WER 0.0185, proc about 9.1 s). Same-day offline sweep 2.46 at 4 threads. Streaming was not re-measured on 2026-10-24 before this phase. | Streaming greedy smoke WER 0.0185 (unchanged). RTFx 2.38 then 2.37 (proc 10.079 s / 10.128 s, runs `1792895923` / `1792895971`, 4 threads). 3.3% and 3.7% under the same-day offline 2.46 (inside 5%). 8.5% and 8.8% under the older 2.60 CSV row; offline on this machine had already fallen from about 3.0 to 2.46, so that gap is not a flush regression. Stage timing wall 10.274 s: mel 34.285 ms (0.3%), encoder 10015.085 ms (97.5%), decoder 220.790 ms (2.1%). Incremental STFT is not justified. `flush` after 1 s feeds matched offline with WER 0.0000 on the full smoke clip and on a 20.3 s truncation (final words present). |
| 4 | Granite beam peak RSS | Committee beam (632.28 s), owned `forward` + `to_vec`: peak RSS 3,099,844 kB, proc 40.270 s, RTFx 15.70, WER 0.0959 (run `1792896581`, `/usr/bin/time -v`, 4 threads). Smoke beam transcript saved before the refactor. | Committee beam via `with_logits` / `decode_ctc_beam_view`: peak RSS 2,616,988 kB (15.6% lower, 482,856 kB; the 40% gate was not met). Proc 39.453 s, RTFx 16.03, WER 0.0959 (run `1792896979`). Smoke beam text byte-identical; greedy smoke WER 0.0370, RTFx 16.43 (run `1792896787`; baseline 17.44). Greedy long audio, one process peak RSS 5,624,492 kB: committee 632.28 s WER 0.0959 RTFx 17.76 (run `1792896831`); datascience-staff-meeting 1750.30 s WER 0.1131 RTFx 17.30 (run `1792896936`). Both greedy clips finished without chunking. Datascience was not run with beam. No frame-ratio warning on these forwards. |
| 5 | Multitalker RTFx / Sortformer runs | Before any code change, smoke 24.00 s, 4 threads, run `1792897390`: RTFx 1.10 (proc 21.877 s), WER 0.0000, 2 speakers. The unchanged PerChunk path, after a doc-hidden run counter was added, does 25 Sortformer `session.run` calls on that clip. | Buffered WER 0.0000. RTFx 2.85 (proc 8.418 s, run `1792898904`) and 2.98 (proc 8.047 s, run `1792898950`), 3 Sortformer runs. Paired PerChunk in the same process: RTFx 1.04 (proc 23.113 s) and 1.07 (proc 22.416 s), 25 runs, WER 0.0000. Speedup versus that paired PerChunk is 2.74x and 2.78x (above the 1.5x gate). An earlier Buffered pass the same day was RTFx 3.24 (proc 7.399 s, run `1792898828`). Word-normalized text matches PerChunk; capitalization differs. The int8 encoder fixes batch 1 on every input, including the caches (`processed_signal`, `cache_last_channel`, `cache_last_time`, `cache_last_channel_len`, `spk_targets`, `bg_spk_targets`). The joint network is batch 1 and its LSTM states are `[2, 1, 640]`. Speakers cannot share one `session.run`. Separate sessions per speaker were not added. Committee (632 s, 6 speakers) was not run: PerChunk smoke is about 22 s for 24 s, so a PerChunk plus Buffered pair on committee would be on the order of 15-30 minutes once the extra encoder passes are counted. |
| 6 | Ark prefill time / Cohere step | Phase 1 Ark smoke: WER 0.0000, RTFx 1.31 (proc 18.353 s, run `1792893137`). Cohere mean decoder step 317.861 ms (Phase 2). | No patched Ark graph: the export already slices before the lm_head. Same-day original `ark_offline_wer`: WER 0.0000, RTFx 1.29 (proc 18.658 s, run `1792899602`). ORT prefill (4 threads, logits `[1, 1, 163958]`): seq=1 median 143.0 ms, seq=32 median 278.8 ms, seq=128 median 721.5 ms (~4.5 ms/token above a ~140 ms fixed cost, of which ~60 ms is Expand over the static 2048 cache). Cache buckets: no-go. Cohere later-step ORT at encoder length 256: median 340.4 ms, and the K/V `MatMulNBits` then-branch did not run; at length 2400 the `If` output copy was 150.9 ms on top of 229.2 ms of `MatMulNBits`. |
| 7 | Cohere steady step / RTFx | Phase 2: 0.71 RTFx at 4 threads (proc 33.896 s, run `1792894114`), mean step 317.861 ms, 79 steps. Same-day merged-only timing, steady disabled in the same process: mean 337.867 ms, tail mean 337.007 ms, 79 steps, wall 35.969 s, RTFx 0.67 | Steady graph on, smoke transcript byte-identical to merged. 78 steady steps mean 326.675 ms (about 10 ms under that merged tail); the prompt step stays on the merged graph at 436.073 ms. Wall 35.946 s, RTFx 0.67. `cohere_offline_wer` with the steady file: WER 0.0000, RTFx 0.69, proc 34.722 s, run `1792907731`. ORT logits and `present.*.decoder.*` match the merged else-branch (max abs diff 0.0). Python ORT, seq=1, past decoder 32, encoder length 2400, 4 threads: steady median 240.4 ms, merged median 258.3 ms. The profiled `If` kernel is 156.0 ms and does not add to wall time; it overlaps MatMulNBits. First steady sweep at encoder length 2400, threads 1/4/8/16: 710.7 / 317.6 / 273.5 / 335.0 ms. 8 threads is a bit faster than 4, 16 is slower, 1 is much slower. No separate decoder `ExecutionConfig`. lm_head stays `MatMulNBits` (98.3 ms of node time in that 4-thread profile). `transcribe_audio_long` on committee (632.28 s): WER 0.0625, CER 0.0366, RTFx 0.69, proc 914.221 s (run `1792908768`). Phase 2 on the same clip was WER 0.0625, CER 0.0366, RTFx 0.64, proc 983.171 s; those runs are a day apart. |

---

## 11. Verification Checklist (every phase)

1. `cargo build --all-features`
2. `cargo test --lib` (all unit tests, no weights needed)
3. `cargo test --release --test model_benchmark --features "sortformer multitalker cohere ark-asr" -- --nocapture` (weight-gated; run alone, no parallel benchmark processes)
4. Compare CSV rows against the Phase 0 baseline; no WER regression unless explicitly justified.
5. `cargo fmt` and `cargo clippy` on touched files.
6. Update the AGENTS.md "Current local delta" table for any new public API.

## 12. Risks

| Risk | Mitigation |
|------|-----------|
| Thread sweep results are machine-specific | Document as guidance, not as a default change |
| Cohere decode refactor changes numerics | Transcript-identity test on the smoke clip |
| Ark `Error` policy breaks callers | Default stays `Truncate`; the new policy is opt-in |
| Multitalker added latency | Opt-in `diar_lookahead`; default unchanged |
| Export patch alters model behavior | Transcript-identity test before accepting the patched graph. Phase 6 did not write a patch: Ark already slices the last hidden state, and a Cohere graph that dropped the encoder-KV `If` would break step 0. |
| Cohere ~318 ms decoder steps | Phase 7 built that steady-state graph. Smoke transcripts match. Same-run wall time did not improve (steady step 326.7 ms vs merged tail 337.0 ms; RTFx 0.67 either way). The `If` else-branch copy profiles at about 156 ms but overlaps the `MatMulNBits` critical path (lm_head about 70-98 ms inside ~230 ms of node time). |

## 13. Upstream PR Split

1. Benchmark coverage (Phase 0)
2. Ark fixes (Phase 1)
3. Cohere decode and long-form (Phase 2)
4. Nemotron `flush()` (Phase 3)
5. CTC beam memory (Phase 4)
6. Multitalker stride (Phase 5)
7. Export scripts (Phase 6)
