# Upstream Fixes and Dynamo Exports Implementation Plan

## Implementation phase

**Current phase:** Complete. Branch re-review approved `69d66c1` (ready to merge). Work is on local `master`, 11 commits from `6fabc14` through `69d66c1`. No NeMo checkpoint export was run.

| Task | Status |
|------|--------|
| 1. Repeated timestamp words | complete (`82936de`, review approved) |
| 2. Sortformer mel layout | complete (`7f4e81b`, review approved) |
| 3. tokenizers 0.23.2 and API feature flags | complete (`c984b5d`, review approved) |
| 4. Shared dynamo export helper | complete (`6fabc14`, review approved) |
| 5. CTC and TDT export scripts | complete (`b1f3595`, review approved) |
| 6. Nemotron, multilingual, and EOU exporters | complete (`c9eb648`, review approved) |
| 7. Unified RNNT exporter | complete (`cd66cb5`, review approved) |
| 8. Multitalker and diarization exporters | complete (`0467d1c` + `67bf53c`, re-review approved) |

Burn stays out of scope. Each task commits only its own files.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the two upstream correctness fixes (repeated timestamp words, Sortformer mel layout), bump `tokenizers` to 0.23.2 and add the newer ONNX Runtime API feature flags, and switch the Python exporters to the PyTorch dynamo exporter from upstream PR #144.

**Architecture:** Rust changes are local and behavior-preserving except for the two bugs. Dynamo export is a Python-only change: a shared helper writes opset 23 graphs, and each existing script calls that helper instead of legacy `torch.onnx.export` (opset 17, `dynamo=False`). Graph input and output names stay the names `src/` already reads. The Burn backend is out of scope.

**Tech Stack:** Rust 2024, `ort` 2.0.0-rc.15 (already in `Cargo.toml`), `tokenizers` 0.23.2, `ndarray` 0.17. Python exporters run in the `onnx` conda environment with `torch >= 2.9`, `onnx`, `onnxscript`, and `onnx_ir`.

## Global Constraints

- Upstream reference commit: `187dbf3e81d869483cf6daa8ee4a97b2aa458835` (`upstream/master`, tag `v0.4.3`).
- Do not add the `burn`, `wgpu`, `metal`, or `vulkan` Cargo features, and do not make `ort` optional.
- Do not change default chunk sizes, execution-provider fallback, or the process-wide `ort` environment in `src/execution.rs`.
- Sortformer ONNX consumed by this crate uses these names: inputs `chunk`, `chunk_lengths`, `spkcache`, `spkcache_lengths`, `fifo`, `fifo_lengths`; outputs `spkcache_fifo_chunk_preds`, `chunk_pre_encode_embs`, `chunk_pre_encode_lengths`. Keep that contract. Upstream's rewritten `export_diar_sortformer.py` emits `preds_diar` / `preds_hires` instead; do not replace this fork's diarization scripts with that file.
- Keep fork-only scripts and behavior: `export_nemotron3_diarization.py` presets and metadata (`sil_emb`, `scores_boost_latest`, `spkcache_sil_frames_per_spk`), `export_ultra_diar_8spk.py`, `export_cohere_steady_decoder.py`, `patch_ctc_onnx_metadata.py`, `quantize_nemotron_streaming.py`, the Nemotron latency table in `export_nemotron_streaming.py`, and the multitalker fallback encoder path.
- Python export work uses `conda activate onnx`. Install missing packages into that environment (`pip install` / `conda install`). Scripts must not hardcode the environment name.
- Do not commit ONNX weights, model directories, or `scripts/__pycache__/`.
- `cargo fmt` on every Rust edit. Commit after each task.

## File structure

| File | Responsibility |
|------|----------------|
| `src/timestamps.rs` | Stop dropping a repeated word while grouping tokens. |
| `src/sortformer.rs` | Emit C-order mel features so ORT can view them. |
| `Cargo.toml`, `Cargo.lock` | `tokenizers` 0.23.2 and `api-29` / `api-30` / `api-31`. |
| `scripts/onnx_export.py` | Shared dynamo exporter (opset 23), constant folding, external-data save. |
| `scripts/export_parakeet_ctc.py` | New CTC exporter, upstream file verbatim. |
| `scripts/export_parakeet_tdt.py` | New TDT exporter, Ultra via `--ultra`, upstream file verbatim. |
| Existing `scripts/export_*.py` listed in Tasks 5-8 | Call `onnx_export` and drop the `dynamo=False` monkeypatch. |

---

### Task 1: Keep repeated words in timestamps

Upstream `f8d7167`. `group_by_words` compares each finished word with the previous word in lowercase and drops the new one when they match. "the the thing" becomes "the thing" for `TimestampMode::Words` and `TimestampMode::Sentences`. Plain transcript text does not go through this function.

**Files:**
- Modify: `src/timestamps.rs` (`group_by_words`, tests module)
- Test: `src/timestamps.rs` (`timestamps::tests`)

**Interfaces:**
- Consumes: `TimedToken { text, start, end }`, `group_by_words`, `group_by_sentences`
- Produces: `group_by_words(&[TimedToken]) -> Vec<TimedToken>` with repeated words kept, in order, with the same start/end rules as today

- [ ] **Step 1: Write the failing test**

Add this test inside `mod tests` in `src/timestamps.rs`, after `test_repetition_preservation`:

```rust
    #[test]
    fn test_word_grouping_keeps_repeated_words() {
        let tokens: Vec<TimedToken> = ["▁the", "▁the", "▁thing"]
            .iter()
            .enumerate()
            .map(|(i, t)| TimedToken {
                text: (*t).to_string(),
                start: i as f32,
                end: i as f32 + 1.0,
            })
            .collect();
        let words = group_by_words(&tokens);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["the", "the", "thing"]);
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 1.0);
        assert_eq!(words[1].start, 1.0);
        assert_eq!(words[1].end, 2.0);
        assert_eq!(words[2].end, 3.0);

        let sentences = group_by_sentences(&tokens);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "the the thing");
    }
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `cargo test --lib timestamps::tests::test_word_grouping_keeps_repeated_words -- --nocapture`

Expected: FAIL. `texts` is `["the", "thing"]` because the second "the" matches `last_word_lower`.

- [ ] **Step 3: Remove the dedup**

Replace `group_by_words` with:

```rust
pub(crate) fn group_by_words(tokens: &[TimedToken]) -> Vec<TimedToken> {
    let Some(last_token) = tokens.last() else {
        return Vec::new();
    };

    let mut words = Vec::new();
    let mut current_word_text = String::new();
    let mut current_word_start = 0.0;

    for (i, token) in tokens.iter().enumerate() {
        // Space-only tokens (from SentencePiece ▁ word boundaries) act as word separators
        // but don't contribute text. Save current word if we hit one.
        if token.text.trim().is_empty() {
            if !current_word_text.is_empty() {
                words.push(TimedToken {
                    text: current_word_text.clone(),
                    start: current_word_start,
                    end: if i > 0 { tokens[i - 1].end } else { token.end },
                });
                current_word_text.clear();
            }
            continue;
        }

        // Check if this starts a new word (SentencePiece uses ▁ or space prefix)
        // Also treat PURE punctuation marks (like ".", ",") as separate words
        // But NOT contractions like "'re" or "'s" or hyphenations like "-two" (ex. twenty-two) which should attach to previous word
        let is_pure_punctuation =
            !token.text.is_empty() && token.text.chars().all(|c| c.is_ascii_punctuation());

        // Check if this is a contraction or hyphenation suffix
        // These should NOT start a new word - they attach to the previous word
        let token_without_marker = token.text.trim_start_matches('▁').trim_start_matches(' ');
        let is_contraction = token_without_marker.starts_with('\'');
        let is_hyphenation = token_without_marker.starts_with('-');

        let starts_word =
            (token.text.starts_with('▁') || token.text.starts_with(' ') || is_pure_punctuation)
                && !is_contraction
                && !is_hyphenation
                || i == 0;

        if starts_word && !current_word_text.is_empty() {
            words.push(TimedToken {
                text: current_word_text.clone(),
                start: current_word_start,
                end: tokens[i - 1].end,
            });
            current_word_text.clear();
        }

        if current_word_text.is_empty() {
            current_word_start = token.start;
        }

        let token_text = token.text.trim_start_matches('▁').trim_start_matches(' ');
        current_word_text.push_str(token_text);
    }

    if !current_word_text.is_empty() {
        words.push(TimedToken {
            text: current_word_text,
            start: current_word_start,
            end: last_token.end,
        });
    }

    words
}
```

Leave `format_sentence` and `test_repetition_preservation` as they are. That path already keeps "uh uh hello".

- [ ] **Step 4: Run the timestamp tests**

Run: `cargo test --lib timestamps::tests -- --nocapture`

Expected: PASS, including `test_word_grouping`, `test_word_grouping_with_hyphenated_word`, `test_sentence_grouping`, `test_repetition_preservation`, `test_space_token_separates_words_from_digits`, and `test_word_grouping_keeps_repeated_words`.

- [ ] **Step 5: Commit**

```bash
git add src/timestamps.rs
git commit -m "$(cat <<'EOF'
Keep repeated words in word and sentence timestamps.

Case-insensitive dedup in word grouping dropped a real second "the".
EOF
)"
```

---

### Task 2: Sortformer mel features in standard layout

Upstream `d787482` (the chunk-site copy from `df2dd41` was later folded into this). `extract_mel_features` returns `log_mel_spec.t().to_owned().insert_axis(Axis(0))`. ndarray 0.17 `to_owned` keeps the memory order of a contiguous transpose, so the result is Fortran layout. `streaming_update` builds the chunk with `TensorRef::from_array_view`, which rejects that layout. A padded tail chunk is assigned into a fresh C-order array and happens to work, so the failure depends on length.

**Files:**
- Modify: `src/sortformer.rs` (`extract_mel_features` around line 1644, and `mod tests`)
- Test: `src/sortformer.rs` (`sortformer::tests::model_mel_layout_is_standard_and_time_major`)

**Interfaces:**
- Consumes: `Array2<f32>` log-mel in `(mel, time)` order, the layout `stft` and the mel filterbank already produce
- Produces: `fn model_mel_layout(log_mel_spec: Array2<f32>) -> Array3<f32>` with shape `(1, time, mel)`, `is_standard_layout() == true`, values `features[[0, t, m]] == log_mel[[m, t]]`

- [ ] **Step 1: Write the failing test**

Add this test in `src/sortformer.rs` inside `mod tests`, after `stft_concentrates_power_at_expected_bin`:

```rust
    #[test]
    fn model_mel_layout_is_standard_and_time_major() {
        let n_mels = 4;
        let n_frames = 6;
        let log_mel = Array2::from_shape_fn((n_mels, n_frames), |(mel, frame)| {
            (mel * 10 + frame) as f32
        });
        let features = model_mel_layout(log_mel.clone());
        assert!(
            features.is_standard_layout(),
            "ORT from_array_view rejects a Fortran mel tensor"
        );
        assert_eq!(features.shape(), &[1, n_frames, n_mels]);
        assert_eq!(features[[0, 3, 2]], log_mel[[2, 3]]);
    }
```

- [ ] **Step 2: Run the test and confirm it fails to compile**

Run: `cargo test --lib --features sortformer model_mel_layout_is_standard_and_time_major -- --nocapture`

Expected: FAIL to compile. `model_mel_layout` is not defined.

- [ ] **Step 3: Add the helper and call it**

Place this function on `impl Sortformer`, immediately above `extract_mel_features`:

```rust
    /// `(mel, time)` log-mel to the model's `(batch, time, mel)` tensor.
    ///
    /// `Array2::t().to_owned()` keeps Fortran layout. ORT refuses a
    /// non-standard view ("Array has a non-contiguous layout"), and a
    /// full-size chunk is passed through `from_array_view` without another copy.
    fn model_mel_layout(log_mel_spec: Array2<f32>) -> Array3<f32> {
        log_mel_spec
            .t()
            .as_standard_layout()
            .into_owned()
            .insert_axis(Axis(0))
    }
```

In `extract_mel_features`, replace the return line:

```rust
        Ok(log_mel_spec.t().to_owned().insert_axis(Axis(0)))
```

with:

```rust
        Ok(Self::model_mel_layout(log_mel_spec))
```

- [ ] **Step 4: Run the Sortformer unit tests**

Run: `cargo test --lib --features sortformer sortformer::tests -- --nocapture`

Expected: PASS, including `model_mel_layout_is_standard_and_time_major` and `stft_concentrates_power_at_expected_bin`.

- [ ] **Step 5: Commit**

```bash
git add src/sortformer.rs
git commit -m "$(cat <<'EOF'
Force Sortformer mel features into standard layout.

A transposed to_owned tensor stays column-major, and ORT rejects that view on full-size chunks.
EOF
)"
```

---

### Task 3: tokenizers 0.23.2 and ONNX Runtime API feature flags

Upstream `b3a2c22` bumps `tokenizers` from 0.23.1 to 0.23.2. Upstream also exposes `api-29`, `api-30`, and `api-31` so callers can select a newer ORT C API. This crate already depends on `ort` 2.0.0-rc.15 as a required dependency. Leave that required. Do not copy upstream's `ort = { optional = true }` or the `ort` feature that turns it on.

**Files:**
- Modify: `Cargo.toml` (dependencies and `[features]`)
- Modify: `Cargo.lock` (via `cargo update -p tokenizers`)

**Interfaces:**
- Consumes: existing features `api-24` through `api-28`, which are `["ort/api-N"]`
- Produces: `tokenizers` version `0.23.2` with the same features (`default-features = false`, `features = ["onig"]`); new features `api-29`, `api-30`, `api-31` in that same shape

- [ ] **Step 1: Edit Cargo.toml**

Change the tokenizers line to:

```toml
tokenizers = { version = "0.23.2", default-features = false, features = ["onig"] }
```

After the existing `api-28` feature, add:

```toml
api-29 = ["ort/api-29"]
api-30 = ["ort/api-30"]
api-31 = ["ort/api-31"]
```

Leave `default = ["cpu", "ort-defaults", "api-28"]`. Leave `ark-asr`, `webgpu`, `sortformer`, `multitalker`, and `cohere` as they are.

- [ ] **Step 2: Refresh the lockfile and compile**

```bash
cargo update -p tokenizers
cargo check --features "sortformer multitalker cohere ark-asr webgpu"
```

Expected: `cargo update` resolves `tokenizers` 0.23.2. `cargo check` finishes with no errors. `Cargo.lock` contains `name = "tokenizers"` / `version = "0.23.2"`.

- [ ] **Step 3: Confirm the feature flags exist on this ort crate**

```bash
cargo check --no-default-features --features "api-29 api-31 webgpu"
```

Expected: PASS. If `ort` 2.0.0-rc.15 does not export `api-31`, drop only the flags that fail and note the highest flag that compiles in the commit message. Do not bump `ort` past rc.15 in this task.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
Bump tokenizers to 0.23.2 and expose newer ORT API features.

Matches upstream's tokenizer bump and api-29 through api-31 flags without making ONNX Runtime optional.
EOF
)"
```

---

### Task 4: Shared dynamo export helper

Upstream `be7bfd5`, file blob `851dbd9392f7f55bf469c8715e38638cc5604d18`. Every later export task imports this module. `torch.onnx.export(..., dynamo=True, opset_version=23)` writes attention as the ONNX Attention op and LSTM as the ONNX LSTM op. `dynamic_axes` keeps the dict shape the current scripts already pass (axis to a name, or a list of axes). An integer value is an upper bound (`torch.export.Dim.AUTO(max=...)`). A symbolic axis that the exporter freezes to a constant is an error.

`INT8_OPS` omits `Attention`. ORT's dynamic quantizer turns Attention into `com.microsoft.QAttention` with attributes that op does not have. Attention stays fp32.

`save` writes weights to one external data file and leaves tensors under 1 KB inside the `.onnx` file. ORT cannot read the dynamo exporter's shape constants from external data. Pass `data_file=None` for a single-file model (Sortformer).

**Files:**
- Create: `scripts/onnx_export.py`
- Test: `python -m py_compile scripts/onnx_export.py`

**Interfaces:**
- Consumes: `torch >= 2.9`, `onnx`, `onnxscript`, `onnx_ir` (imported inside `fold_constants`)
- Produces:
  - `OPSET = 23`
  - `INT8_OPS = ["MatMul", "Gemm", "Gather", "LSTM"]`
  - `use_sdpa(module) -> int`
  - `exact_position_slices(module) -> int`
  - `export(module, inputs, path, input_names, output_names, dynamic_axes=None) -> None`
  - `export_decoder_joint(model, path) -> None`
  - `fold_constants(model) -> onnx.ModelProto`
  - `save(model, path, data_file=None, metadata=None) -> None`

- [ ] **Step 1: Write `scripts/onnx_export.py`**

Copy the file from the pinned commit and check the blob id:

```bash
git show 187dbf3e81d869483cf6daa8ee4a97b2aa458835:scripts/onnx_export.py > scripts/onnx_export.py
git hash-object scripts/onnx_export.py
```

Expected hash: `851dbd9392f7f55bf469c8715e38638cc5604d18`.

The module is this file (do not edit it in this task):

```python
# ONNX export settings shared by the export scripts in this folder.
#
# Every graph is exported with PyTorch's dynamo exporter (torch.export + onnxscript) at opset 23,
# which writes attention as the ONNX Attention op and keeps LSTMs as the ONNX LSTM op.
#
# Requirements: torch >= 2.9, onnxscript, onnx.

import os
import tempfile
import types

import onnx
import torch

OPSET = 23

# Op types for ONNX Runtime's dynamic int8 quantization. Its default list includes Attention, which
# it then turns into com.microsoft QAttention with attributes that op does not have; attention
# stays fp32.
INT8_OPS = ["MatMul", "Gemm", "Gather", "LSTM"]


def use_sdpa(module):
    """Route NeMo attention through F.scaled_dot_product_attention, so each layer exports as one
    ONNX Attention node (the relative-position term goes in as an additive mask)."""
    count = 0
    for m in module.modules():
        if hasattr(m, "use_pytorch_sdpa"):
            m.use_pytorch_sdpa = True
            m.use_pytorch_sdpa_backends = None
            count += 1
    print(f"  SDPA attention on {count} modules")
    return count


def _rel_pos_forward(self, x, cache_len=0):
    if self.xscale:
        x = x * self.xscale
    input_len = x.size(1) + cache_len
    center_pos = self.pe.size(1) // 2 + 1
    pos_emb = self.pe.narrow(1, center_pos - input_len, 2 * input_len - 1)
    if self.dropout_emb:
        pos_emb = self.dropout_emb(pos_emb)
    return self.dropout(x), pos_emb


def exact_position_slices(module):
    """NeMo's RelPositionalEncoding slices its table as pe[:, start:end]; with a symbolic input
    length torch.export cannot size that slice. narrow() takes the same 2 * L - 1 rows with an
    exact length."""
    count = 0
    for m in module.modules():
        if type(m).__name__ == "RelPositionalEncoding":
            m.forward = types.MethodType(_rel_pos_forward, m)
            count += 1
    return count


def export(module, inputs, path, input_names, output_names, dynamic_axes=None):
    """Export `module` with the dynamo exporter.

    `dynamic_axes` maps input names to their dynamic axes: a list of axes, or a dict from axis to
    an upper bound (an int) or a name (ignored). Some NeMo code only exports with a bound, e.g. a
    streaming encoder slicing its positional table. Output sizes follow from the inputs.
    """
    bounds = {}
    for name, axes in (dynamic_axes or {}).items():
        items = axes.items() if isinstance(axes, dict) else ((a, None) for a in axes)
        bounds[name] = {int(a): (v if isinstance(v, int) else None) for a, v in items}
    dynamic_shapes = tuple(
        {a: torch.export.Dim.AUTO(max=m) if m else torch.export.Dim.AUTO for a, m in bounds[name].items()}
        if name in bounds else None
        for name in input_names
    )
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            tuple(inputs),
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=OPSET,
            dynamo=True,
        )
    # Dim.AUTO lets torch.export fix a size it cannot keep dynamic; refuse that.
    graph_inputs = {v.name: v for v in onnx.load(path, load_external_data=False).graph.input}
    for name, axes in bounds.items():
        dims = graph_inputs[name].type.tensor_type.shape.dim
        fixed = [a for a in axes if dims[a].HasField("dim_value")]
        if fixed:
            raise RuntimeError(f"{name}: axes {fixed} became fixed sizes in the export")


def export_decoder_joint(model, path):
    """Export NeMo's RNNT/TDT decoder + joint (one prediction step) as one .onnx file, with NeMo's
    own example inputs, input/output names and dynamic axes."""
    from nemo.core.classes.common import typecheck
    from nemo.core.classes.exportable import Exportable

    decoder_joint = model.decoder_joint
    decoder_joint.eval()
    typecheck.set_typecheck_enabled(enabled=False)
    for m in decoder_joint.modules():
        if isinstance(m, Exportable) and m is not decoder_joint:
            m._prepare_for_export(noreplace=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "decoder_joint.onnx")
        export(
            decoder_joint,
            decoder_joint.input_example(),
            raw,
            input_names=decoder_joint.input_names,
            output_names=decoder_joint.output_names,
            dynamic_axes=decoder_joint.dynamic_shapes_for_export(),
        )
        save(fold_constants(onnx.load(raw)), path)


def fold_constants(model):
    """Fold every constant subgraph of a small graph into initializers. The dynamo exporter
    builds an LSTM's W and R from the PyTorch weights at runtime (gate reordering); folded,
    they are plain weights like in any other ONNX LSTM."""
    import onnx_ir as ir
    from onnxscript import optimizer

    ir_model = ir.serde.deserialize_model(model)
    optimizer.fold_constants(ir_model, input_size_limit=2**31, output_size_limit=2**31)
    ir.passes.common.RemoveUnusedNodesPass()(ir_model)
    return ir.serde.serialize_model(ir_model)


def save(model, path, data_file=None, metadata=None):
    """Save `model` with its weights in one external data file (or none if `data_file` is None),
    adding `metadata` as metadata_props.

    Tensors under 1 KB stay in the .onnx file: ONNX Runtime cannot read the dynamo export's shape
    constants from external data.
    """
    for key, value in (metadata or {}).items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=str(key), value=str(value)))
    if data_file is None:
        onnx.save_model(model, path)
    else:
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )
```

- [ ] **Step 2: Install the Python packages and compile the helper**

```bash
conda activate onnx
python -c "import torch; print(torch.__version__)"
pip install "torch>=2.9" onnx onnxscript
python -m py_compile scripts/onnx_export.py
```

Expected: compile prints nothing and exits 0. If `torch` is already `>= 2.9`, skip the torch reinstall. `onnx_ir` ships with `onnxscript`; the import is inside `fold_constants`, so `py_compile` does not need it.

- [ ] **Step 3: Commit**

```bash
git add scripts/onnx_export.py
git commit -m "$(cat <<'EOF'
Add the shared dynamo ONNX export helper.

Opset 23 export, SDPA attention, and external-data saving are shared by the export scripts.
EOF
)"
```

---

### Task 5: CTC and TDT export scripts

Upstream `744c2b4` and `cadebd4`. These files do not exist in this fork. Take them verbatim. Ultra is `export_parakeet_tdt.py --ultra <dir>`, not a separate script.

**Files:**
- Create: `scripts/export_parakeet_ctc.py` (blob `c4bc29e3bc2781cc6b04ee35e78e7a248b7e5e39`)
- Create: `scripts/export_parakeet_tdt.py` (blob `8239c5531b09392936a8115dfa60b37a856f9e09`)

**Interfaces:**
- Consumes: `onnx_export.export`, `onnx_export.save`, `onnx_export.use_sdpa`, `onnx_export.export_decoder_joint`
- Produces:
  - `python export_parakeet_ctc.py <output dir> [--model nvidia/parakeet-ctc-0.6b or a .nemo path]` writes `model.onnx`, `model.onnx_data`, `tokenizer.json`. Inputs `input_features` `[batch, frames, 80]` and `attention_mask` `[batch, frames]`. Output `logits`.
  - `python export_parakeet_tdt.py <output dir> [--model nvidia/parakeet-tdt-0.6b-v3] [--ultra <hf dir>] [--check-audio speech.wav]` writes `encoder-model.onnx`, `encoder-model.onnx.data`, `decoder_joint-model.onnx`, `vocab.txt`. Encoder inputs `audio_signal`, `length`. Outputs `outputs`, `encoded_lengths`.

- [ ] **Step 1: Copy both files at the pinned commit**

```bash
git show 187dbf3e81d869483cf6daa8ee4a97b2aa458835:scripts/export_parakeet_ctc.py > scripts/export_parakeet_ctc.py
git show 187dbf3e81d869483cf6daa8ee4a97b2aa458835:scripts/export_parakeet_tdt.py > scripts/export_parakeet_tdt.py
git hash-object scripts/export_parakeet_ctc.py scripts/export_parakeet_tdt.py
```

Expected hashes, in that order:

```text
c4bc29e3bc2781cc6b04ee35e78e7a248b7e5e39
8239c5531b09392936a8115dfa60b37a856f9e09
```

- [ ] **Step 2: Compile them**

```bash
conda activate onnx
python -m py_compile scripts/export_parakeet_ctc.py scripts/export_parakeet_tdt.py
```

Expected: exit 0. A full NeMo export needs a checkpoint and is not part of this task. Do not run the scripts against a model unless a `.nemo` or Hugging Face cache is already present, and do not commit the output directory.

- [ ] **Step 3: Commit**

```bash
git add scripts/export_parakeet_ctc.py scripts/export_parakeet_tdt.py
git commit -m "$(cat <<'EOF'
Add dynamo exporters for Parakeet CTC and TDT.

TDT's --ultra flag loads Moondream's Parakeet Ultra weights into the same graph.
EOF
)"
```

---

### Task 6: Nemotron, multilingual Nemotron, and EOU call the helper

These three scripts already build their own encoder wrappers and name their inputs. Swap the export call and the decoder export. Keep CLI flags, the latency table in `export_nemotron_streaming.py`, metadata keys, and output filenames.

Delete, in each file that has it, the PyTorch 2.9 block that sets `dynamo=False` (`_PATCH_MARKER` / `_legacy_onnx_patched` in the two Nemotron scripts). Dynamo is now the path we want. Drop the `functools` import when nothing else uses it.

Before the encoder export, after the model is loaded and in eval mode:

```python
import onnx_export

onnx_export.use_sdpa(model.encoder)
onnx_export.exact_position_slices(model.encoder)
```

`exact_position_slices` is required wherever `RelPositionalEncoding` slices `pe[:, start:end]` under a symbolic length. Nemotron streaming has that slice. Call it on the encoder module.

**Files:**
- Modify: `scripts/export_nemotron_streaming.py`
- Modify: `scripts/export_nemotron_streaming_multilingual.py`
- Modify: `scripts/export_realtime_eou_120m.py`

**Interfaces:**
- Consumes: `onnx_export.export`, `onnx_export.save`, `onnx_export.export_decoder_joint`, `onnx_export.use_sdpa`, `onnx_export.exact_position_slices`
- Produces: the same output files as today (`encoder.onnx` + `encoder.onnx.data` + `decoder_joint.onnx` + tokenizer). Nemotron metadata keys stay `chunk_size_output_frames` and `pre_encode_cache`.

- [ ] **Step 1: Replace the Nemotron English encoder export**

In `scripts/export_nemotron_streaming.py`, delete the `dynamo=False` monkeypatch (the block from the "PyTorch 2.9+" comment through `torch.onnx.export = _patched_onnx_export`).

Add `import onnx_export` with the other imports. After `model.eval()`, call `onnx_export.use_sdpa(model.encoder)` and `onnx_export.exact_position_slices(model.encoder)`.

Replace the `torch.onnx.export(...)` call that writes `temp_encoder_path` with:

```python
onnx_export.export(
    encoder_wrapper,
    (processed_signal, processed_signal_length,
     cache_last_channel, cache_last_time, cache_last_channel_len),
    temp_encoder_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "processed_signal": {0: "batch", 2: "time"},
        "processed_signal_length": {0: "batch"},
        "encoded": {0: "batch", 1: "time"},
        "encoded_len": {0: "batch"},
    },
)
```

Replace the `onnx.save_model(...)` that writes `encoder.onnx` (the block that appends `chunk_size_output_frames` and `pre_encode_cache` by hand, then saves with `size_threshold=0`) with:

```python
pre_encode_cache = int(getattr(streaming_cfg, "pre_encode_cache_size", [0, 9])[1])
print(f"  Metadata: chunk_size_output_frames={chunk_size}, pre_encode_cache={pre_encode_cache}")
onnx_export.save(
    encoder_model,
    final_encoder_path,
    "encoder.onnx.data",
    {"chunk_size_output_frames": chunk_size, "pre_encode_cache": pre_encode_cache},
)
```

`onnx_export.save` appends those metadata props. Delete the two `metadata_props.append` calls so the keys are not written twice.

Replace the decoder section that calls `model.export(...)` and then searches the directory for a decoder file with:

```python
final_decoder_path = os.path.join(OUTPUT_DIR, "decoder_joint.onnx")
onnx_export.export_decoder_joint(model, final_decoder_path)
print("  Decoder saved: decoder_joint.onnx")
```

Keep the cleanup that deletes files other than `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`, and `tokenizer.model`. Keep the latency preset table and the `config.json` dump.

- [ ] **Step 2: Do the same in the multilingual script**

Apply the same three substitutions in `scripts/export_nemotron_streaming_multilingual.py`: delete the `dynamo=False` patch, call `use_sdpa` and `exact_position_slices` on `model.encoder`, replace `torch.onnx.export` with `onnx_export.export` using that file's existing `input_names`, `output_names`, and `dynamic_axes`, replace the external-data save with `onnx_export.save` while keeping its metadata keys, and replace `model.export(...)` with `onnx_export.export_decoder_joint`.

- [ ] **Step 3: Switch the EOU script**

In `scripts/export_realtime_eou_120m.py`:

```python
import onnx_export
```

After the model is loaded and set to eval, before the encoder wrapper is traced:

```python
onnx_export.use_sdpa(model.encoder)
onnx_export.exact_position_slices(model.encoder)
```

`o_d` is the output directory and `o_p` is its path prefix. Replace:

```python
with torch.no_grad():
    model.export(output=o_p + ".onnx", check_trace=False)
```

with:

```python
onnx_export.export_decoder_joint(model, os.path.join(o_d, "decoder_joint.onnx"))
```

Delete the later loop that renames a decoder file to `decoder_joint.onnx` and deletes extra encoder files. `export_decoder_joint` writes that name directly, and the stateful encoder is written to `enc_onnx_path` below.

Replace the encoder `torch.onnx.export` (the call that passes `(features, feat_lengths, *cache_state)`, `i_names`, `o_names`, and the `dynamic_axes` dict for `audio_signal` / `length` / `outputs` / the three cache inputs) with `onnx_export.export` and those same arguments. Drop `opset_version`. The encoder stays one file at `enc_onnx_path` (`encoder.onnx`); this script does not consolidate an external data file, so do not call `onnx_export.save` for it. Leave the `InferenceSession` check that `cache_last_channel` is an input.

- [ ] **Step 4: Compile**

```bash
conda activate onnx
python -m py_compile scripts/export_nemotron_streaming.py scripts/export_nemotron_streaming_multilingual.py scripts/export_realtime_eou_120m.py
```

Expected: exit 0. Confirm with `rg "dynamo=False|opset_version=17" scripts/export_nemotron_streaming.py scripts/export_nemotron_streaming_multilingual.py scripts/export_realtime_eou_120m.py` that those strings are gone.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_nemotron_streaming.py scripts/export_nemotron_streaming_multilingual.py scripts/export_realtime_eou_120m.py
git commit -m "$(cat <<'EOF'
Export Nemotron and EOU graphs with the dynamo exporter.

Streaming input names, latency metadata, and the Nemotron chunk table stay as they are.
EOF
)"
```

---

### Task 7: Unified RNNT export

`scripts/export_parakeet_unified.py` exports by calling `model.export`, which hits the legacy tracer, and it patches `torch.onnx.export` to force `dynamo=False`. Replace that with an encoder module plus `export_decoder_joint`. Keep this file's CLI (`--quantize-int8`, override config, work dir, tokenizer extraction, artifact names `encoder.onnx` / `decoder_joint.onnx`).

Quantization must pass `op_types_to_quantize=onnx_export.INT8_OPS` so Attention is not rewritten to `QAttention`.

**Files:**
- Modify: `scripts/export_parakeet_unified.py`

**Interfaces:**
- Consumes: `onnx_export.export`, `onnx_export.use_sdpa`, `onnx_export.export_decoder_joint`, `onnx_export.save`, `onnx_export.INT8_OPS`
- Produces: the same `export_model(model_path, output_dir, work_dir, quantize_int8)` entry point and the same output filenames

- [ ] **Step 1: Delete the legacy patch**

Remove `patch_torch_onnx_export` and the call in `main`. Add `import onnx_export`.

- [ ] **Step 2: Add the encoder export**

```python
class OfflineEncoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        return self.encoder(audio_signal=audio_signal, length=length)


def export_encoder(model: Any, dest: Path) -> None:
    audio = torch.randn(1, 16000 * 4) * 0.1
    features, lengths = model.preprocessor(
        input_signal=audio, length=torch.tensor([audio.shape[1]])
    )
    onnx_export.export(
        OfflineEncoder(model.encoder),
        (features, lengths),
        str(dest),
        input_names=["audio_signal", "length"],
        output_names=["outputs", "encoded_lengths"],
        dynamic_axes={"audio_signal": {0: "batch", 2: "time"}, "length": {0: "batch"}},
    )
```

- [ ] **Step 3: Call it from `export_model`**

After `model.to("cpu")`:

```python
    onnx_export.use_sdpa(model.encoder)

    print(f"Exporting raw ONNX graphs to: {raw_export_dir}")
    raw_export_dir.mkdir(parents=True, exist_ok=True)
    raw_encoder = raw_export_dir / "encoder.onnx"
    raw_decoder_joint = raw_export_dir / "decoder_joint.onnx"
    with torch.no_grad():
        export_encoder(model, raw_encoder)
        onnx_export.export_decoder_joint(model, str(raw_decoder_joint))
```

Delete `model.export(...)` and the `find_single(raw_export_dir, "encoder-*.onnx")` / `decoder_joint-*.onnx` lookups. The paths above are the raw files. Keep `save_external_data_onnx` and `save_single_file_onnx`, and implement `save_external_data_onnx` with:

```python
    model = onnx.load(str(src), load_external_data=True)
    onnx_export.save(model, str(dest), data_filename)
```

In `quantize_to_tmp`, add `op_types_to_quantize=onnx_export.INT8_OPS` to `quantize_dynamic`.

- [ ] **Step 4: Compile**

```bash
conda activate onnx
python -m py_compile scripts/export_parakeet_unified.py
```

Expected: exit 0. `rg "model.export|dynamo=False" scripts/export_parakeet_unified.py` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_parakeet_unified.py
git commit -m "$(cat <<'EOF'
Export the unified RNNT graphs with the dynamo exporter.

INT8 quantization skips the Attention op, which ORT's quantizer rewrites incorrectly.
EOF
)"
```

---

### Task 8: Multitalker and the three diarization exporters

Multitalker cannot use `export_decoder_joint`. Its encoder and decoder are custom wrappers (`MultitalkerEncoderExport`, the fallback kernel wrapper, `DecoderJointExport`) because speaker-target hooks are otherwise traced as constants. Keep both encoder paths, the 7 encoder inputs, and the post-export check that `spk_targets` is still a graph input.

Diarization scripts keep the 6-in/3-out names listed in Global Constraints. `export_nemotron3_diarization.py` already builds a dense mask, but the head axis is 1 (`expand(B, 1, Q_LEN, KV_LEN)`). ORT's Attention op does not broadcast the mask; the mask must be `[B, H, Q, K]`.

**Files:**
- Modify: `scripts/export_multitalker.py` (three `torch.onnx.export` calls, around lines 290, 352, and 553)
- Modify: `scripts/export_diar_sortformer.py` (the single `torch.onnx.export`)
- Modify: `scripts/export_ultra_diar_8spk.py` (the single `torch.onnx.export`)
- Modify: `scripts/export_nemotron3_diarization.py` (`install_onnx_attention_shims` and the export call)

**Interfaces:**
- Consumes: `onnx_export.export`, `onnx_export.use_sdpa`, `onnx_export.exact_position_slices`, `onnx_export.save`
- Produces: unchanged CLI and unchanged ONNX names for each script

- [ ] **Step 1: Point each multitalker export at the helper**

Add `import onnx_export`. On the ASR model, before the encoder wrapper is built:

```python
onnx_export.use_sdpa(asr_model.encoder)
onnx_export.exact_position_slices(asr_model.encoder)
```

Replace each of the three `torch.onnx.export` calls with `onnx_export.export`, passing the same module, example tuple, path, `input_names`, `output_names`, and `dynamic_axes`. Delete `opset_version` and `do_constant_folding`. For the decoder wrapper, run the loaded graph through `onnx_export.fold_constants` before the final save so the LSTM weights are initializers:

```python
onnx_export.export(
    wrapper,
    (dummy_encoder, dummy_targets, dummy_state1, dummy_state2),
    raw_path,
    input_names=["encoder_outputs", "targets", "input_states_1", "input_states_2"],
    output_names=["outputs", "prednet_lengths", "states_1", "states_2"],
    dynamic_axes={"encoder_outputs": {1: "enc_time"}},
)
onnx_export.save(onnx_export.fold_constants(onnx.load(raw_path)), output_path)
```

Write `raw_path` under a `tempfile.TemporaryDirectory`, then save to the existing `output_path`. Apply the same raw-then-`fold_constants` sequence only to the decoder. Encoder saves that already consolidate external data should use `onnx_export.save(model_proto, output_path, "encoder.onnx.data")` and must still check that `spk_targets` is a graph input after the save.

Update the requirements comment: `torch >= 2.9`, plus `onnxscript`. Remove "PyTorch < 2.9".

- [ ] **Step 2: Switch the 4-speaker and 8-speaker diarization exports**

In `scripts/export_diar_sortformer.py` and `scripts/export_ultra_diar_8spk.py`, replace `torch.onnx.export` with `onnx_export.export` and the same names:

```python
onnx_export.export(
    model,
    input_example,
    raw_path,
    input_names=["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"],
    output_names=["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"],
    dynamic_axes={
        "chunk": {0: "batch", 1: "time_chunk"},
        "spkcache": {0: "batch", 1: "time_cache"},
        "fifo": {0: "batch", 1: "time_fifo"},
        "spkcache_fifo_chunk_preds": {0: "batch", 1: "time_out"},
        "chunk_pre_encode_embs": {0: "batch", 1: "time_pre_encode"},
    },
)
```

Export into a temp file, load it, then `onnx_export.save(model_onnx, final_path)` with no external data file when the current script writes a single `.onnx`. Copy the existing metadata stamping so it still runs on the proto before `save`. `save` also accepts a `metadata` dict; use that and delete the hand-written `metadata_props.append` loop if every key is a string. Keep `num_speakers` and any `sil_emb` key the script already writes.

If the 4-speaker model has no FlexAttention, do not add the Nemotron-3 shim to it.

- [ ] **Step 3: Expand the Nemotron-3 mask to `[B, H, Q, K]`**

In `install_onnx_attention_shims` inside `scripts/export_nemotron3_diarization.py`, replace `dense_block_mask` with:

```python
    def dense_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device=None, **_):
        b = torch.arange(B or 1, device=device).view(-1, 1, 1, 1)
        h = torch.arange(H or 1, device=device).view(1, -1, 1, 1)
        q = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
        kv = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)
        # ORT Attention does not broadcast the mask. The query axis and the
        # head axis both have to match q.
        dense = mask_mod(b, h, q, kv)
        return dense.expand(b.shape[0], h.shape[1], q.shape[2], kv.shape[3])
```

Leave `sdpa` as `F.scaled_dot_product_attention(..., attn_mask=block_mask)`. Then replace that file's `torch.onnx.export` with `onnx_export.export` using the same six input names and three output names and the same `dynamic_axes`. Export to a temp path. The script's later metadata pass (`sil_emb`, `scores_boost_latest`, `spkcache_sil_frames_per_spk`, preset lengths) must run on the loaded proto, then `onnx_export.save(model_onnx, output_path)` for the single-file graph. Keep `--preset`, `--int8-output`, and `--compare-wav`.

Also call `onnx_export.use_sdpa(model)` after `model.eval()` and before the shim, so layers that already have `use_pytorch_sdpa` take the SDPA path. Keep `install_onnx_attention_shims` for the FlexAttention encoder the SDPA flag does not cover.

- [ ] **Step 4: Compile**

```bash
conda activate onnx
python -m py_compile scripts/export_multitalker.py scripts/export_diar_sortformer.py scripts/export_ultra_diar_8spk.py scripts/export_nemotron3_diarization.py
rg "opset_version|dynamo=False" scripts/export_multitalker.py scripts/export_diar_sortformer.py scripts/export_ultra_diar_8spk.py scripts/export_nemotron3_diarization.py
```

Expected: `py_compile` exits 0. `rg` prints nothing.

- [ ] **Step 5: Record the new scripts in `AGENTS.md`**

In the export-script table, add rows for `scripts/onnx_export.py` (shared dynamo helper, opset 23), `scripts/export_parakeet_ctc.py`, and `scripts/export_parakeet_tdt.py` (`--ultra` for Moondream's Parakeet Ultra). In the Nemotron and diarization rows, say the graphs are exported with `scripts/onnx_export.py`. Do not document Burn.

- [ ] **Step 6: Commit**

```bash
git add scripts/export_multitalker.py scripts/export_diar_sortformer.py scripts/export_ultra_diar_8spk.py scripts/export_nemotron3_diarization.py AGENTS.md
git commit -m "$(cat <<'EOF'
Export multitalker and diarization graphs with the dynamo exporter.

Speaker-target inputs and the 6-in/3-out diarization contract stay put. The Nemotron-3 attention mask is expanded to the full head and query shape ORT Attention requires.
EOF
)"
```

---

## Verification after all tasks

```bash
cargo fmt --check
cargo test --lib timestamps::tests
cargo test --lib --features sortformer sortformer::tests
cargo check --features "sortformer multitalker cohere ark-asr webgpu"
conda activate onnx
python -m py_compile scripts/onnx_export.py scripts/export_parakeet_ctc.py scripts/export_parakeet_tdt.py scripts/export_*.py
```

Expected: each command exits 0.

A real NeMo export (CTC, TDT, Nemotron, diarization) needs checkpoints and is optional. When one is run, confirm the output names with:

```bash
python -c "import onnx,sys; m=onnx.load(sys.argv[1], load_external_data=False); print([i.name for i in m.graph.input]); print([o.name for o in m.graph.output])" path/to/model.onnx
```

Diarization outputs must include `spkcache_fifo_chunk_preds`. CTC output must be `logits`. Do not commit the generated ONNX.

## Out of scope

- Burn backend, generated Burn modules, `examples/burn.rs`, and making `ort` optional.
- Requiring callers to pass an `ort` environment (`ExecutionConfig::with_environment`). This crate keeps `default_environment()`.
- Upstream's Sortformer v3 output pair `preds_diar` / `preds_hires`, and the metadata key `learnable_sil_emb`. This fork's loader reads `spkcache_fifo_chunk_preds` and `sil_emb`.
- Orukeet download script and example.
- Re-running weight-gated WER/DER benchmarks. The timestamp and mel fixes are covered by the unit tests above.
