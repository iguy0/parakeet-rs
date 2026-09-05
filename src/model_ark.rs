use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{s, Array1, Array2, Array3, Array4};
use ort::session::Session;
use ort::value::TensorRef;
use serde::Deserialize;
use std::path::{Path, PathBuf};

const DEFAULT_ASR_BLOCK_TOKEN_ID_FROM: i64 = 151670;
const DEFAULT_TRANSCRIBE_PROMPT: &str = "Please transcribe this audio.";

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeManifest {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    pub audio_token_id: i64,
    #[serde(default)]
    pub im_end_token_id: Option<i64>,
    #[serde(default)]
    pub stop_token_ids: Vec<i64>,
    pub audio_encoder_default: String,
    pub audio_adapter_default: String,
    #[serde(default = "default_merge_factor")]
    pub audio_merge_factor: usize,
    #[serde(default = "default_encoder_hidden")]
    pub audio_encoder_hidden_size: usize,
    pub embedding_default: String,
    pub llm_default: String,
}

fn default_sample_rate() -> u32 {
    16000
}
fn default_merge_factor() -> usize {
    4
}
fn default_encoder_hidden() -> usize {
    1280
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct LlmMeta {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_total_len: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ArkRuntimeConfig {
    pub manifest: RuntimeManifest,
    pub llm: LlmMeta,
    pub head_dim: usize,
    pub asr_block_token_id_from: i64,
    pub transcribe_prompt: String,
}

pub(crate) struct EmbeddingTable {
    mmap: memmap2::Mmap,
    pub vocab: usize,
    pub hidden: usize,
}

impl EmbeddingTable {
    fn load(path: &Path, hidden: usize) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Config(format!("Failed to mmap {}: {e}", path.display())))?;
        if hidden == 0 || mmap.len() % (hidden * 4) != 0 {
            return Err(Error::Config(format!(
                "embedding table {} size {} is not a multiple of hidden {hidden} * 4",
                path.display(),
                mmap.len()
            )));
        }
        if mmap.as_ptr() as usize % std::mem::align_of::<f32>() != 0 {
            return Err(Error::Config(format!(
                "embedding table {} is not f32-aligned",
                path.display()
            )));
        }
        let vocab = mmap.len() / (hidden * 4);
        Ok(Self {
            mmap,
            vocab,
            hidden,
        })
    }

    fn row(&self, id: usize) -> Result<&[f32]> {
        if id >= self.vocab {
            return Err(Error::Model(format!(
                "token id {id} >= embedding vocab {}",
                self.vocab
            )));
        }
        let start = id * self.hidden;
        let bytes = &self.mmap[start * 4..(start + self.hidden) * 4];
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.hidden) })
    }

    pub(crate) fn gather(&self, ids: &[i64]) -> Result<Array3<f32>> {
        let seq = ids.len();
        let mut out = Array3::<f32>::zeros((1, seq, self.hidden));
        let dst = out
            .as_slice_mut()
            .ok_or_else(|| Error::Model("embedding output is not contiguous".into()))?;
        for (t, &id) in ids.iter().enumerate() {
            if id < 0 {
                return Err(Error::Model(format!("negative token id {id}")));
            }
            let row = self.row(id as usize)?;
            let start = t * self.hidden;
            dst[start..start + self.hidden].copy_from_slice(row);
        }
        Ok(out)
    }
}

pub(crate) struct ArkKvCache {
    pub keys: Vec<Array4<f32>>,
    pub values: Vec<Array4<f32>>,
    inner: usize,
    used: usize,
}

impl ArkKvCache {
    pub(crate) fn zeros(cfg: &ArkRuntimeConfig) -> Self {
        let shape = (
            1,
            cfg.llm.max_total_len,
            cfg.llm.num_key_value_heads,
            cfg.head_dim,
        );
        let inner = cfg.llm.num_key_value_heads * cfg.head_dim;
        let zero = Array4::<f32>::zeros(shape);
        Self {
            keys: vec![zero.clone(); cfg.llm.num_layers],
            values: vec![zero; cfg.llm.num_layers],
            inner,
            used: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        if self.used == 0 || self.inner == 0 {
            return;
        }
        let n = self.used * self.inner;
        for t in self.keys.iter_mut().chain(self.values.iter_mut()) {
            if let Some(slice) = t.as_slice_mut() {
                let end = n.min(slice.len());
                slice[..end].fill(0.0);
            }
        }
        self.used = 0;
    }
}

pub(crate) struct ArkModel {
    encoder: Session,
    adapter: Session,
    embedding: Option<Session>,
    embed_table: Option<EmbeddingTable>,
    llm: Session,
    pub config: ArkRuntimeConfig,
    cache_key_names: Vec<String>,
    cache_value_names: Vec<String>,
    key_delta_names: Vec<String>,
    value_delta_names: Vec<String>,
}

impl ArkModel {
    pub(crate) fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let root = resolve_asset_root(model_dir.as_ref())?;
        let manifest = load_manifest(&root)?;
        let llm_meta = load_llm_meta(&root, &manifest.llm_default)?;
        if llm_meta.num_attention_heads == 0 {
            return Err(Error::Config("num_attention_heads must be > 0".into()));
        }
        let head_dim = llm_meta.hidden_size / llm_meta.num_attention_heads;
        if head_dim == 0 {
            return Err(Error::Config("computed head_dim is 0".into()));
        }

        let encoder = exec_config.build_session(&root.join(&manifest.audio_encoder_default))?;
        let adapter = exec_config.build_session(&root.join(&manifest.audio_adapter_default))?;
        let llm = exec_config.build_session(&root.join(&manifest.llm_default))?;

        let embed_data = embedding_data_path(&root, &manifest.embedding_default);
        let (embed_table, embedding) = if embed_data.is_file() {
            (
                Some(EmbeddingTable::load(&embed_data, llm_meta.hidden_size)?),
                None,
            )
        } else {
            (
                None,
                Some(exec_config.build_session(&root.join(&manifest.embedding_default))?),
            )
        };

        let n = llm_meta.num_layers;
        let mut cache_key_names = Vec::with_capacity(n);
        let mut cache_value_names = Vec::with_capacity(n);
        let mut key_delta_names = Vec::with_capacity(n);
        let mut value_delta_names = Vec::with_capacity(n);
        for i in 0..n {
            cache_key_names.push(format!("cache_key_{i}"));
            cache_value_names.push(format!("cache_value_{i}"));
            key_delta_names.push(format!("key_delta_{i}"));
            value_delta_names.push(format!("value_delta_{i}"));
        }

        Ok(Self {
            encoder,
            adapter,
            embedding,
            embed_table,
            llm,
            config: ArkRuntimeConfig {
                manifest,
                llm: llm_meta,
                head_dim,
                asr_block_token_id_from: DEFAULT_ASR_BLOCK_TOKEN_ID_FROM,
                transcribe_prompt: DEFAULT_TRANSCRIBE_PROMPT.to_string(),
            },
            cache_key_names,
            cache_value_names,
            key_delta_names,
            value_delta_names,
        })
    }

    pub(crate) fn encode_audio(&mut self, audios: &Array3<f32>) -> Result<Array3<f32>> {
        let input = TensorRef::<f32>::from_array_view(audios.view())?;
        let outputs = self.encoder.run(ort::inputs!("audios" => input))?;
        let encoded = extract_array3(&outputs, "encoded_audio_features")?;
        let merged = merge_audio_features(
            &encoded,
            self.config.manifest.audio_merge_factor.max(1),
            self.config.manifest.audio_encoder_hidden_size,
        )?;
        let merged_ref = TensorRef::<f32>::from_array_view(merged.view())?;
        let adapter_out =
            self.adapter
                .run(ort::inputs!("merged_audio_features" => merged_ref))?;
        extract_array3(&adapter_out, "audio_embeddings")
    }

    pub(crate) fn embed_tokens(&mut self, ids: &[i64]) -> Result<Array3<f32>> {
        if let Some(table) = &self.embed_table {
            return table.gather(ids);
        }
        let seq = ids.len();
        let input_ids = Array2::from_shape_vec((1, seq), ids.to_vec())
            .map_err(|e| Error::Model(format!("Failed to shape input_ids: {e}")))?;
        let session = self.embedding.as_mut().ok_or_else(|| {
            Error::Config("no embedding table or embedding ONNX session".into())
        })?;
        let input = TensorRef::<i64>::from_array_view(input_ids.view())?;
        let outputs = session.run(ort::inputs!("input_ids" => input))?;
        extract_array3(&outputs, "inputs_embeds")
    }

    pub(crate) fn embed_one(&mut self, id: i64, dest: &mut Array3<f32>) -> Result<()> {
        if dest.shape() != [1, 1, self.config.llm.hidden_size] {
            return Err(Error::Model("step embedding buffer has the wrong shape".into()));
        }
        if let Some(table) = &self.embed_table {
            let row = table.row(id as usize)?;
            dest.as_slice_mut()
                .ok_or_else(|| Error::Model("step embedding buffer is not contiguous".into()))?
                .copy_from_slice(row);
            return Ok(());
        }
        *dest = self.embed_tokens(&[id])?;
        Ok(())
    }

    /// Run the decoder, scatter KV deltas into `cache`, and return the last-step logits.
    pub(crate) fn run_llm(
        &mut self,
        inputs_embeds: &Array3<f32>,
        attention_mask: &Array2<i64>,
        cache_position: &Array1<i64>,
        cache: &mut ArkKvCache,
        write_start: usize,
    ) -> Result<Array1<f32>> {
        let n = self.config.llm.num_layers;
        let embed_ref = TensorRef::<f32>::from_array_view(inputs_embeds.view())?;
        let mask_ref = TensorRef::<i64>::from_array_view(attention_mask.view())?;
        let pos_ref = TensorRef::<i64>::from_array_view(cache_position.view())?;

        let mut feeds: Vec<(String, ort::session::SessionInputValue<'_>)> =
            Vec::with_capacity(3 + 2 * n);
        feeds.push(("inputs_embeds".into(), embed_ref.into()));
        feeds.push(("attention_mask".into(), mask_ref.into()));
        feeds.push(("cache_position".into(), pos_ref.into()));
        for i in 0..n {
            let key_ref = TensorRef::<f32>::from_array_view(cache.keys[i].view())?;
            let value_ref = TensorRef::<f32>::from_array_view(cache.values[i].view())?;
            feeds.push((self.cache_key_names[i].clone(), key_ref.into()));
            feeds.push((self.cache_value_names[i].clone(), value_ref.into()));
        }

        let outputs = self.llm.run(feeds)?;
        scatter_cache_from_outputs(
            &outputs,
            cache,
            write_start,
            &self.key_delta_names,
            &self.value_delta_names,
        )?;
        extract_last_logits(&outputs, "logits")
    }
}

fn scatter_cache_from_outputs(
    outputs: &ort::session::SessionOutputs<'_>,
    cache: &mut ArkKvCache,
    start: usize,
    key_names: &[String],
    value_names: &[String],
) -> Result<()> {
    let inner = cache.inner.max(1);
    let mut seq_written = 0usize;
    for i in 0..cache.keys.len() {
        seq_written = scatter_one(&outputs[key_names[i].as_str()], &mut cache.keys[i], start, inner)?;
        scatter_one(&outputs[value_names[i].as_str()], &mut cache.values[i], start, inner)?;
    }
    cache.used = cache.used.max(start + seq_written);
    Ok(())
}

fn scatter_one(
    value: &ort::value::Value,
    dest: &mut Array4<f32>,
    start: usize,
    inner: usize,
) -> Result<usize> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract KV delta: {e}")))?;
    if shape.len() != 4 {
        return Err(Error::Model(format!(
            "Expected 4D KV delta, got shape {shape:?}"
        )));
    }
    let seq = shape[1] as usize;
    let end = start + seq;
    let max_len = dest.shape()[1];
    if end > max_len {
        return Err(Error::Model(format!(
            "KV cache write [{start}, {end}) exceeds max_total_len {max_len}"
        )));
    }
    let dst = dest
        .as_slice_mut()
        .ok_or_else(|| Error::Model("KV cache is not contiguous".into()))?;
    let n = seq * inner;
    dst[start * inner..start * inner + n].copy_from_slice(&data[..n]);
    Ok(seq)
}

pub(crate) fn grow_attention_mask(mask: &mut Array2<i64>, valid_len: usize) -> Result<()> {
    let max_total_len = mask.shape()[1];
    if valid_len > max_total_len {
        return Err(Error::Model(format!(
            "valid length exceeds cache length: {valid_len} > {max_total_len}"
        )));
    }
    if valid_len > 0 {
        mask[[0, valid_len - 1]] = 1;
    }
    Ok(())
}

pub(crate) fn reset_attention_mask(mask: &mut Array2<i64>, valid_len: usize) -> Result<()> {
    mask.fill(0);
    let max_total_len = mask.shape()[1];
    if valid_len > max_total_len {
        return Err(Error::Model(format!(
            "valid length exceeds cache length: {valid_len} > {max_total_len}"
        )));
    }
    if let Some(row) = mask.as_slice_mut() {
        row[..valid_len].fill(1);
    }
    Ok(())
}

fn merge_audio_features(
    encoded: &Array3<f32>,
    merge_factor: usize,
    hidden: usize,
) -> Result<Array3<f32>> {
    let bsz = encoded.shape()[0];
    let seq_len = encoded.shape()[1];
    let feat_dim = encoded.shape()[2];
    if feat_dim != hidden {
        return Err(Error::Model(format!(
            "unexpected audio hidden size: {feat_dim} != {hidden}"
        )));
    }

    let data = if seq_len < merge_factor {
        let mut padded = Array3::<f32>::zeros((bsz, merge_factor, hidden));
        padded
            .slice_mut(s![.., ..seq_len, ..])
            .assign(&encoded.slice(s![.., ..seq_len, ..]));
        padded
    } else {
        let trimmed = (seq_len / merge_factor) * merge_factor;
        encoded.slice(s![.., ..trimmed, ..]).to_owned()
    };
    let seq_len = data.shape()[1];
    let out_seq = seq_len / merge_factor;
    data.as_standard_layout()
        .into_owned()
        .into_shape_with_order((bsz, out_seq, hidden * merge_factor))
        .map_err(|e| Error::Model(format!("Failed to reshape merged audio features: {e}")))
}

fn extract_array3(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Array3<f32>> {
    let (shape, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    if shape.len() != 3 {
        return Err(Error::Model(format!(
            "Expected 3D {name}, got shape {shape:?}"
        )));
    }
    Array3::from_shape_vec(
        (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        data.to_vec(),
    )
    .map_err(|e| Error::Model(format!("Failed to reshape {name}: {e}")))
}

fn extract_last_logits(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Array1<f32>> {
    let (shape, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    if shape.len() != 3 {
        return Err(Error::Model(format!(
            "Expected 3D {name}, got shape {shape:?}"
        )));
    }
    let seq = shape[1] as usize;
    let vocab = shape[2] as usize;
    if seq == 0 {
        return Err(Error::Model("LLM returned empty logits".into()));
    }
    let start = (seq - 1) * vocab;
    Ok(Array1::from_vec(data[start..start + vocab].to_vec()))
}

fn embedding_data_path(root: &Path, embedding_onnx: &str) -> PathBuf {
    let onnx = root.join(embedding_onnx);
    onnx.with_extension("data")
}

fn resolve_asset_root(dir: &Path) -> Result<PathBuf> {
    if dir.join("runtime_manifest.json").is_file() {
        return Ok(dir.to_path_buf());
    }
    let nested = dir.join("model");
    if nested.join("runtime_manifest.json").is_file() {
        return Ok(nested);
    }
    Err(Error::Config(format!(
        "runtime_manifest.json not found in {} or {}/model",
        dir.display(),
        dir.display()
    )))
}

fn load_manifest(root: &Path) -> Result<RuntimeManifest> {
    let path = root.join("runtime_manifest.json");
    let text = std::fs::read_to_string(&path)?;
    serde_json::from_str(&text).map_err(|e| {
        Error::Config(format!("Failed to parse {}: {e}", path.display()))
    })
}

fn load_llm_meta(root: &Path, llm_filename: &str) -> Result<LlmMeta> {
    let stem = Path::new(llm_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("llm");
    let candidates = [
        root.join("llm_kv_fp32_qwen_native.json"),
        root.join(format!("{stem}.json")),
        root.join("llm_kv_cpu_fp32_int8.json"),
        root.join("build").join("llm_kv_fp32_qwen_native.json"),
        root.parent()
            .map(|p| p.join("build").join("llm_kv_fp32_qwen_native.json"))
            .unwrap_or_else(|| root.join("build").join("llm_kv_fp32_qwen_native.json")),
    ];
    for path in &candidates {
        if path.is_file() {
            let text = std::fs::read_to_string(path)?;
            return serde_json::from_str(&text).map_err(|e| {
                Error::Config(format!("Failed to parse {}: {e}", path.display()))
            });
        }
    }
    Err(Error::Config(format!(
        "LLM metadata JSON not found next to {llm_filename} in {}",
        root.display()
    )))
}

pub(crate) fn find_tokenizer(dir: &Path) -> Result<PathBuf> {
    for candidate in [
        dir.join("tokenizer.json"),
        dir.join("model").join("tokenizer.json"),
    ] {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(Error::Config(format!(
        "tokenizer.json not found in {}",
        dir.display()
    )))
}
