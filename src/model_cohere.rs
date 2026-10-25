use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{Session, SessionInputValue};
use ort::value::ValueType;
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// Cohere Transcribe architecture constants.
/// Verified against `CohereLabs/cohere-transcribe-03-2026` config and the
/// onnx-community ONNX export I/O signatures. Used when the decoder graph
/// does not expose static `past_key_values.N.decoder.key` shapes.
pub(crate) const NUM_DECODER_LAYERS: usize = 8;
pub(crate) const NUM_KV_HEADS: usize = 8;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const N_MELS: usize = 128;

/// Encoder hidden states.
/// Shape: `[batch=1, T_enc, HIDDEN_SIZE]`
pub(crate) struct CohereEncoderOutput {
    pub(crate) hidden_states: Array3<f32>,
}

/// Per-layer cache tensor for one of `{decoder, encoder}.{key, value}`.
/// Shape: `[batch=1, NUM_KV_HEADS, seq_len, HEAD_DIM]`.
type LayerCache = Array4<f32>;

/// Decoder `past_key_values` cache split into self-attention (`decoder_*`)
/// and cross-attention (`encoder_*`) tensors per layer.
///
/// On the first decoder call all caches are zero-length; the model
/// populates the cross-attention caches from `encoder_hidden_states`. On
/// subsequent calls the model writes new self-attention K/V into the
/// growing decoder caches and reuses the encoder caches as-is.
pub(crate) struct CoherePastKv {
    pub(crate) decoder_k: Vec<LayerCache>,
    pub(crate) decoder_v: Vec<LayerCache>,
    pub(crate) encoder_k: Vec<LayerCache>,
    pub(crate) encoder_v: Vec<LayerCache>,
}

impl CoherePastKv {
    pub(crate) fn empty(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let zero = Array4::<f32>::zeros((1, num_kv_heads, 0, head_dim));
        Self {
            decoder_k: vec![zero.clone(); num_layers],
            decoder_v: vec![zero.clone(); num_layers],
            encoder_k: vec![zero.clone(); num_layers],
            encoder_v: vec![zero; num_layers],
        }
    }

    /// Total decoded tokens accumulated in the self-attention cache.
    pub(crate) fn past_decoder_len(&self) -> usize {
        self.decoder_k.first().map(|t| t.shape()[2]).unwrap_or(0)
    }
}

/// `past_key_values.{i}.{decoder|encoder}.{key|value}` for `n_layers` layers.
pub(crate) fn cache_input_names(n_layers: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(n_layers * 4);
    for i in 0..n_layers {
        for kind in ["decoder", "encoder"] {
            for proj in ["key", "value"] {
                names.push(format!("past_key_values.{i}.{kind}.{proj}"));
            }
        }
    }
    names
}

/// Steady-state decoder filenames, quantized first, nested `onnx/` then flat.
/// The merged graph is still required; these files are optional.
pub(crate) const STEADY_DECODER_CANDIDATES: &[&str] = &[
    "onnx/decoder_model_steady_quantized.onnx",
    "decoder_model_steady_quantized.onnx",
    "onnx/decoder_model_steady.onnx",
    "decoder_model_steady.onnx",
];

/// First existing steady-state decoder under `model_dir`, if any.
pub(crate) fn find_steady_decoder_path(model_dir: &Path) -> Option<PathBuf> {
    STEADY_DECODER_CANDIDATES.iter().find_map(|name| {
        let path = model_dir.join(name);
        path.is_file().then_some(path)
    })
}

/// Non-cache inputs of the steady-state graph. No `encoder_hidden_states`.
///
/// The decode loop builds these feeds directly. This list is the contract
/// checked by unit tests; release builds of the library do not call it.
#[allow(dead_code)]
pub(crate) fn steady_decoder_input_names(n_layers: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(4 + n_layers * 4);
    names.push("input_ids".to_string());
    names.push("attention_mask".to_string());
    names.push("position_ids".to_string());
    names.push("num_logits_to_keep".to_string());
    names.extend(cache_input_names(n_layers));
    names
}

fn present_names_from_past(past: &[String]) -> Vec<String> {
    past.iter()
        .map(|name| name.replacen("past_key_values.", "present.", 1))
        .collect()
}

pub(crate) struct CohereModel {
    encoder: Session,
    decoder: Session,
    /// Steady-state decoder (no first-step `If`, no encoder-cache outputs).
    /// Used for every step after the encoder K/V cache is filled.
    decoder_steady: Option<Session>,
    /// When false, every step uses the merged decoder even if `decoder_steady`
    /// was loaded. Default true when that file is present.
    use_steady: bool,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Precomputed `past_key_values.*` input names, 4 per layer.
    past_names: Vec<String>,
    /// Matching `present.*` output names, same order as `past_names`.
    present_names: Vec<String>,
    /// When set, each successful `run_decoder_step` appends its duration.
    record_steps: bool,
    step_millis: Vec<f64>,
    /// Parallel to `step_millis`: true when that step ran the steady graph.
    step_used_steady: Vec<bool>,
}

impl CohereModel {
    pub(crate) fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Try int8 quantised first, then fp32, then fp16. Both flat
        // (`encoder_model_quantized.onnx` next to the directory root) and
        // nested (`onnx/encoder_model_quantized.onnx` as in the
        // onnx-community HF repo) layouts are supported.
        let encoder_path = Self::find_file(
            model_dir,
            &[
                "onnx/encoder_model_quantized.onnx",
                "encoder_model_quantized.onnx",
                "onnx/encoder_model.onnx",
                "encoder_model.onnx",
                "onnx/encoder_model_fp16.onnx",
                "encoder_model_fp16.onnx",
            ],
        )?;
        let decoder_path = Self::find_file(
            model_dir,
            &[
                "onnx/decoder_model_merged_quantized.onnx",
                "decoder_model_merged_quantized.onnx",
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
                "onnx/decoder_model_merged_fp16.onnx",
                "decoder_model_merged_fp16.onnx",
            ],
        )?;

        let encoder = exec_config.build_session(&encoder_path)?;
        let decoder = exec_config.build_session(&decoder_path)?;
        let (decoder_steady, use_steady) = load_steady_decoder(model_dir, &exec_config)?;
        let (num_layers, num_kv_heads, head_dim) = derive_decoder_dims(&decoder);
        let past_names = cache_input_names(num_layers);
        let present_names = present_names_from_past(&past_names);

        Ok(Self {
            encoder,
            decoder,
            decoder_steady,
            use_steady,
            num_layers,
            num_kv_heads,
            head_dim,
            past_names,
            present_names,
            record_steps: false,
            step_millis: Vec::new(),
            step_used_steady: Vec::new(),
        })
    }

    pub(crate) fn steady_decoder_loaded(&self) -> bool {
        self.decoder_steady.is_some()
    }

    /// Enable the steady-state decoder for steps after the first.
    ///
    /// Has no effect when that graph was not loaded. The default is on
    /// when `decoder_model_steady[_quantized].onnx` was found.
    pub(crate) fn set_use_steady_decoder(&mut self, enabled: bool) {
        self.use_steady = enabled && self.decoder_steady.is_some();
    }

    pub(crate) fn uses_steady_decoder(&self) -> bool {
        self.use_steady && self.decoder_steady.is_some()
    }

    pub(crate) fn empty_past(&self) -> CoherePastKv {
        CoherePastKv::empty(self.num_layers, self.num_kv_heads, self.head_dim)
    }

    pub(crate) fn set_record_steps(&mut self, enabled: bool) {
        self.record_steps = enabled;
        self.step_millis.clear();
        self.step_used_steady.clear();
    }

    pub(crate) fn step_millis(&self) -> &[f64] {
        &self.step_millis
    }

    pub(crate) fn step_used_steady(&self) -> &[bool] {
        &self.step_used_steady
    }

    /// Run the encoder on log-mel features with shape `[1, T, N_MELS]`.
    pub(crate) fn run_encoder(
        &mut self,
        input_features: &Array3<f32>,
    ) -> Result<CohereEncoderOutput> {
        let features_ref = ort::value::TensorRef::<f32>::from_array_view(input_features.view())?;
        let outputs = self.encoder.run(ort::inputs!(
            "input_features" => features_ref,
        ))?;

        let (h_shape, h_data) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract last_hidden_state: {e}")))?;

        let hidden_states = Array3::from_shape_vec(
            (
                h_shape[0] as usize,
                h_shape[1] as usize,
                h_shape[2] as usize,
            ),
            h_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape last_hidden_state: {e}")))?;

        Ok(CohereEncoderOutput { hidden_states })
    }

    /// Run one decoder step. Returns logits for the LAST output position
    /// only (we always pass `num_logits_to_keep=1` since we only need the
    /// next-token distribution for greedy decoding) and the new past_kv.
    ///
    /// `past_kv` is taken by value so encoder K/V populated on the first
    /// step can be moved into the next cache. Later steps do not read
    /// `present.{i}.encoder.*`; those tensors are unchanged after step 0.
    /// When a steady-state session is loaded and enabled, steps with a
    /// non-empty encoder cache run that graph (no `encoder_hidden_states`
    /// input, no `present.*.encoder.*` outputs).
    pub(crate) fn run_decoder_step(
        &mut self,
        tokens: &Array2<i64>,
        past_kv: CoherePastKv,
        encoder_out: &CohereEncoderOutput,
    ) -> Result<(Array1<f32>, CoherePastKv)> {
        let started = self.record_steps.then(std::time::Instant::now);
        let steady = self.uses_steady_decoder()
            && past_kv
                .encoder_k
                .first()
                .is_some_and(|tensor| tensor.shape()[2] > 0);
        let seq_len = tokens.shape()[1];
        let past_len = past_kv.past_decoder_len();
        let total_len = past_len + seq_len;
        let n_layers = self.num_layers;

        // position_ids: [past_len, past_len+1, ..., past_len+seq_len-1]
        let position_ids = Array2::from_shape_vec(
            (1, seq_len),
            (past_len..total_len).map(|i| i as i64).collect(),
        )
        .map_err(|e| Error::Model(format!("position_ids shape mismatch: {e}")))?;

        // attention_mask: [1, total_len] all-ones (covers past + current tokens)
        let attention_mask = Array2::<i64>::from_elem((1, total_len), 1);

        // num_logits_to_keep: scalar i64 = 1 (only need the last position)
        let num_logits = ndarray::Array0::<i64>::from_elem((), 1);

        // Borrow large tensors as views to avoid per-step copies into ORT.
        let tokens_ref = ort::value::TensorRef::<i64>::from_array_view(tokens.view())?;
        let attn_ref = ort::value::TensorRef::<i64>::from_array_view(attention_mask.view())?;
        let pos_ref = ort::value::TensorRef::<i64>::from_array_view(position_ids.view())?;
        let enc_ref = (!steady)
            .then(|| {
                ort::value::TensorRef::<f32>::from_array_view(encoder_out.hidden_states.view())
            })
            .transpose()?;

        let mut feeds: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(5 + self.past_names.len());
        feeds.push((Cow::Borrowed("input_ids"), tokens_ref.into()));
        feeds.push((Cow::Borrowed("attention_mask"), attn_ref.into()));
        feeds.push((Cow::Borrowed("position_ids"), pos_ref.into()));
        feeds.push((
            Cow::Borrowed("num_logits_to_keep"),
            ort::value::Value::from_array(num_logits)?.into(),
        ));
        if let Some(enc_ref) = enc_ref {
            feeds.push((Cow::Borrowed("encoder_hidden_states"), enc_ref.into()));
        }
        for i in 0..n_layers {
            let base = i * 4;
            let dk = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[i].view())?;
            let dv = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[i].view())?;
            let ek = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[i].view())?;
            let ev = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[i].view())?;
            feeds.push((Cow::Owned(self.past_names[base].clone()), dk.into()));
            feeds.push((Cow::Owned(self.past_names[base + 1].clone()), dv.into()));
            feeds.push((Cow::Owned(self.past_names[base + 2].clone()), ek.into()));
            feeds.push((Cow::Owned(self.past_names[base + 3].clone()), ev.into()));
        }

        let outputs = if steady {
            self.decoder_steady
                .as_mut()
                .ok_or_else(|| Error::Model("steady decoder session missing".into()))?
                .run(feeds)?
        } else {
            self.decoder.run(feeds)?
        };

        // logits: [1, n_positions, vocab_size]. With num_logits_to_keep=1
        // n_positions is 1, but we slice the last position defensively in
        // case the model ever returns all positions.
        let (l_shape, l_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;
        let n_positions = l_shape[1] as usize;
        let vocab_size = l_shape[2] as usize;
        let last_start = n_positions.saturating_sub(1) * vocab_size;
        let logits = Array1::from_vec(l_data[last_start..last_start + vocab_size].to_vec());

        let new_past = read_past_kv(&outputs, past_kv, &self.present_names, n_layers)?;

        if let Some(started) = started {
            self.step_millis
                .push(started.elapsed().as_secs_f64() * 1000.0);
            self.step_used_steady.push(steady);
        }
        Ok((logits, new_past))
    }

    fn find_file(dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
        for name in candidates {
            let path = dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "None of {:?} found in {}",
            candidates,
            dir.display()
        )))
    }
}

fn load_steady_decoder(
    model_dir: &Path,
    exec_config: &ExecutionConfig,
) -> Result<(Option<Session>, bool)> {
    let Some(path) = find_steady_decoder_path(model_dir) else {
        return Ok((None, false));
    };
    let session = exec_config.build_session(&path)?;
    if steady_session_accepted(&session) {
        return Ok((Some(session), true));
    }
    eprintln!(
        "parakeet-rs: ignoring {} (not a steady-state Cohere decoder)",
        path.display()
    );
    Ok((None, false))
}

/// Filename match is enough when the variant key is absent. A present key
/// must be `steady`, and the I/O must be the post-step-0 contract.
fn steady_session_accepted(session: &Session) -> bool {
    if let Ok(meta) = session.metadata() {
        if let Some(value) = meta.custom("cohere_decoder_variant") {
            if value != "steady" {
                return false;
            }
        }
    }
    let has_input = |name: &str| session.inputs().iter().any(|input| input.name() == name);
    let has_output = |name: &str| session.outputs().iter().any(|output| output.name() == name);
    has_input("past_key_values.0.encoder.key")
        && !has_input("encoder_hidden_states")
        && has_output("logits")
        && has_output("present.0.decoder.key")
        && !has_output("present.0.encoder.key")
}

/// Layer count, KV heads, and head dim from `past_key_values.N.decoder.key`.
/// Missing or dynamic dimensions fall back to the Cohere Transcribe constants.
fn derive_decoder_dims(decoder: &Session) -> (usize, usize, usize) {
    let mut max_layer: Option<usize> = None;
    let mut kv_heads: Option<usize> = None;
    let mut head_dim: Option<usize> = None;
    for input in decoder.inputs() {
        let Some(rest) = input.name().strip_prefix("past_key_values.") else {
            continue;
        };
        let Some((idx, tail)) = rest.split_once('.') else {
            continue;
        };
        if tail != "decoder.key" {
            continue;
        }
        let Ok(layer) = idx.parse::<usize>() else {
            continue;
        };
        max_layer = Some(max_layer.map_or(layer, |current| current.max(layer)));
        let ValueType::Tensor { shape, .. } = input.dtype() else {
            continue;
        };
        if shape.len() >= 4 {
            if shape[1] > 0 {
                kv_heads = Some(shape[1] as usize);
            }
            if shape[3] > 0 {
                head_dim = Some(shape[3] as usize);
            }
        }
    }
    (
        max_layer
            .map(|layer| layer + 1)
            .filter(|n| *n > 0)
            .unwrap_or(NUM_DECODER_LAYERS),
        kv_heads.filter(|n| *n > 0).unwrap_or(NUM_KV_HEADS),
        head_dim.filter(|n| *n > 0).unwrap_or(HEAD_DIM),
    )
}

/// Decoder K/V are copied out once per step. Encoder K/V are copied only
/// on the first step (zero-length input cache); later steps move the
/// tensors already stored on `past_kv`.
fn read_past_kv(
    outputs: &ort::session::SessionOutputs,
    past_kv: CoherePastKv,
    present_names: &[String],
    n: usize,
) -> Result<CoherePastKv> {
    let reuse_encoder = past_kv.encoder_k.first().is_some_and(|t| t.shape()[2] > 0);
    let mut decoder_k = Vec::with_capacity(n);
    let mut decoder_v = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 4;
        decoder_k.push(extract_cache(outputs, &present_names[base])?);
        decoder_v.push(extract_cache(outputs, &present_names[base + 1])?);
    }
    let (encoder_k, encoder_v) = if reuse_encoder {
        (past_kv.encoder_k, past_kv.encoder_v)
    } else {
        let mut encoder_k = Vec::with_capacity(n);
        let mut encoder_v = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 4;
            encoder_k.push(extract_cache(outputs, &present_names[base + 2])?);
            encoder_v.push(extract_cache(outputs, &present_names[base + 3])?);
        }
        (encoder_k, encoder_v)
    };
    Ok(CoherePastKv {
        decoder_k,
        decoder_v,
        encoder_k,
        encoder_v,
    })
}

/// Extract a 4-D `[1, heads, seq, head_dim]` cache tensor from the decoder
/// outputs by name. One allocation (`to_vec` into `from_shape_vec`).
fn extract_cache(outputs: &ort::session::SessionOutputs, name: &str) -> Result<LayerCache> {
    let (shape, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    Array4::from_shape_vec(
        (
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
        ),
        data.to_vec(),
    )
    .map_err(|e| Error::Model(format!("Failed to reshape {name}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::{
        STEADY_DECODER_CANDIDATES, cache_input_names, find_steady_decoder_path,
        steady_decoder_input_names,
    };

    #[test]
    fn cache_names_generated_for_n_layers() {
        let names = cache_input_names(3);
        assert_eq!(names.len(), 12);
        for i in 0..3 {
            let base = i * 4;
            assert_eq!(names[base], format!("past_key_values.{i}.decoder.key"));
            assert_eq!(
                names[base + 1],
                format!("past_key_values.{i}.decoder.value")
            );
            assert_eq!(names[base + 2], format!("past_key_values.{i}.encoder.key"));
            assert_eq!(
                names[base + 3],
                format!("past_key_values.{i}.encoder.value")
            );
        }
        assert!(cache_input_names(0).is_empty());
    }

    #[test]
    fn steady_input_names_omit_encoder_hidden_states() {
        let names = steady_decoder_input_names(2);
        assert!(
            !names.iter().any(|name| name == "encoder_hidden_states"),
            "{names:?}"
        );
        let past = cache_input_names(2);
        assert!(past.iter().all(|name| names.contains(name)));
        assert_eq!(
            &names[..4],
            [
                "input_ids",
                "attention_mask",
                "position_ids",
                "num_logits_to_keep",
            ]
        );
        assert_eq!(names.len(), 4 + past.len());
        assert!(
            steady_decoder_input_names(0)
                .iter()
                .all(|name| { !name.starts_with("past_key_values.") })
        );
    }

    #[test]
    fn steady_decoder_candidate_search_uses_temp_dir() {
        let dir = std::env::temp_dir().join(format!(
            "cohere_steady_search_{}_{}",
            std::process::id(),
            STEADY_DECODER_CANDIDATES.len()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("onnx")).unwrap();
        assert!(find_steady_decoder_path(&dir).is_none());

        let flat = dir.join("decoder_model_steady.onnx");
        std::fs::write(&flat, b"steady").unwrap();
        assert_eq!(find_steady_decoder_path(&dir).as_ref(), Some(&flat));

        let nested = dir.join("onnx/decoder_model_steady.onnx");
        std::fs::write(&nested, b"nested").unwrap();
        assert_eq!(find_steady_decoder_path(&dir).as_ref(), Some(&nested));

        let quantized = dir.join("onnx/decoder_model_steady_quantized.onnx");
        std::fs::write(&quantized, b"quantized").unwrap();
        assert_eq!(find_steady_decoder_path(&dir).as_ref(), Some(&quantized));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
