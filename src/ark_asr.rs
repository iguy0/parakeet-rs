//! ARK-ASR speech-LLM engine.
//!
//! Whisper-style audio encoder + MLP adapter + Qwen2 causal LM with a
//! static pre-allocated KV cache. Consumes the INT8 ONNX package from
//! [`Audio8/ark-asr-0.6b-int8-onnx`](https://huggingface.co/Audio8/ark-asr-0.6b-int8-onnx).
//!
//! Layout (files at the package root, or under `model/`):
//! - `runtime_manifest.json`
//! - `audio_encoder_whisper_int8.onnx`
//! - `audio_encoder_adapter_int8.onnx`
//! - `embedding_fp32.onnx` (+ `embedding_fp32.data`)
//! - `llm_kv_cpu_fp32_int8.onnx`
//! - `tokenizer.json`
//! - `preprocessor_config.json`

use crate::audio::{extract_whisper_logmel, FeatureCache};
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_ark::{
    find_tokenizer, grow_attention_mask, reset_attention_mask, ArkKvCache, ArkModel,
};
use crate::timestamps::TimestampMode;
use crate::transcriber::Transcriber;
use ndarray::{Array1, Array2, Array3, Axis};
use std::collections::HashSet;
use std::path::Path;
use tokenizers::Tokenizer;

const WHISPER_FLOOR_DB: f32 = 8.0;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_MAX_AUDIO_SECONDS: u32 = 30;

/// ARK-ASR-0.6B speech-LLM transcriber.
pub struct ArkAsr {
    model: ArkModel,
    tokenizer: Tokenizer,
    preprocessor: PreprocessorConfig,
    feature_cache: FeatureCache,
    kv_cache: ArkKvCache,
    attention_mask: Array2<i64>,
    step_embed: Array3<f32>,
    step_pos: Array1<i64>,
    bad_token_ids: HashSet<u32>,
    stop_ids: HashSet<u32>,
    prompt_prefix: Vec<i64>,
    audio_token_id: i64,
    prompt_suffix: Vec<i64>,
    max_new_tokens: usize,
    max_audio_seconds: u32,
}

impl ArkAsr {
    /// Load ARK-ASR from a directory containing `runtime_manifest.json` and
    /// the ONNX graphs listed in it.
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let exec = exec_config.unwrap_or_default();
        let model = ArkModel::from_pretrained(model_dir, exec)?;
        let tokenizer_path = find_tokenizer(model_dir)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer.json: {e}")))?;

        let preprocessor = {
            let mut preprocessor = load_preprocessor(model_dir)?;
            if model.config.manifest.sample_rate > 0 {
                preprocessor.sampling_rate = model.config.manifest.sample_rate as usize;
            }
            preprocessor
        };
        let feature_cache = FeatureCache::from_config(&preprocessor);
        let kv_cache = ArkKvCache::zeros(&model.config);
        let hidden = model.config.llm.hidden_size;
        let max_total_len = model.config.llm.max_total_len;
        let attention_mask = Array2::<i64>::zeros((1, max_total_len));
        let step_embed = Array3::<f32>::zeros((1, 1, hidden));
        let step_pos = Array1::<i64>::zeros(1);

        let eos_id = tokenizer.token_to_id("<|im_end|>").or_else(|| {
            model
                .config
                .manifest
                .im_end_token_id
                .and_then(|id| (id >= 0).then_some(id as u32))
        });
        let block_from = model.config.asr_block_token_id_from.max(0) as u32;
        let bad_token_ids = build_bad_token_ids(&tokenizer, eos_id, block_from);
        let stop_ids = model
            .config
            .manifest
            .stop_token_ids
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| id as u32)
            .collect();

        let audio_token_id = model.config.manifest.audio_token_id;
        let (prompt_prefix, prompt_suffix) =
            build_prompt_pieces(&tokenizer, &model.config.transcribe_prompt)?;

        Ok(Self {
            model,
            tokenizer,
            preprocessor,
            feature_cache,
            kv_cache,
            attention_mask,
            step_embed,
            step_pos,
            bad_token_ids,
            stop_ids,
            prompt_prefix,
            audio_token_id,
            prompt_suffix,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            max_audio_seconds: DEFAULT_MAX_AUDIO_SECONDS,
        })
    }

    pub fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    pub fn set_max_new_tokens(&mut self, max: usize) {
        self.max_new_tokens = max.max(1);
    }

    pub fn max_audio_seconds(&self) -> u32 {
        self.max_audio_seconds
    }

    pub fn set_max_audio_seconds(&mut self, seconds: u32) {
        self.max_audio_seconds = seconds.max(1);
    }

    /// Transcribe raw 16 kHz f32 samples (mono or interleaved multi-channel).
    pub fn transcribe_audio(&mut self, audio: &[f32]) -> Result<String> {
        self.transcribe_samples_inner(audio.to_vec(), self.preprocessor.sampling_rate as u32, 1)
            .map(|r| r.text)
    }

    fn transcribe_samples_inner(
        &mut self,
        mut audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<TranscriptionResult> {
        if sample_rate != self.preprocessor.sampling_rate as u32 {
            return Err(Error::Audio(format!(
                "Audio sample rate {} doesn't match expected {}. Please resample your audio first.",
                sample_rate, self.preprocessor.sampling_rate
            )));
        }
        if channels > 1 {
            audio = audio
                .chunks(channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect();
        }
        if audio.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                tokens: Vec::new(),
            });
        }

        let hop = self.preprocessor.hop_length.max(1);
        let max_samples =
            (self.max_audio_seconds as usize).saturating_mul(self.preprocessor.sampling_rate);
        if audio.len() > max_samples {
            audio.truncate(max_samples);
        }

        let merge_factor = self.model.config.manifest.audio_merge_factor.max(1);
        let mel_frames = audio.len() / hop;
        let n_audio_tokens = audio_token_count(mel_frames, merge_factor);

        let features = extract_whisper_logmel(
            &audio,
            &self.feature_cache,
            &self.preprocessor,
            WHISPER_FLOOR_DB,
        )?;
        let audios = features.insert_axis(Axis(0)).as_standard_layout().to_owned();

        let input_ids = assemble_prompt_ids(
            &self.prompt_prefix,
            self.audio_token_id,
            n_audio_tokens,
            &self.prompt_suffix,
        );
        let seq = input_ids.len();
        let ids = Array2::from_shape_vec((1, seq), input_ids.clone())
            .map_err(|e| Error::Model(format!("Failed to shape input_ids: {e}")))?;

        let mut embeds = self.model.embed_tokens(&input_ids)?;
        let audio_embeds = self.model.encode_audio(&audios)?;
        inject_audio(&ids, &mut embeds, &audio_embeds, self.audio_token_id)?;
        let embeds = if embeds.is_standard_layout() {
            embeds
        } else {
            embeds.as_standard_layout().into_owned()
        };

        let token_ids = self.decode_greedy(&embeds)?;
        let text = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {e}")))?
            .trim()
            .to_string();

        Ok(TranscriptionResult {
            text,
            tokens: Vec::new(),
        })
    }

    fn decode_greedy(&mut self, prompt_embeds: &Array3<f32>) -> Result<Vec<u32>> {
        let max_total_len = self.model.config.llm.max_total_len;
        let prompt_len = prompt_embeds.shape()[1];
        if prompt_len + self.max_new_tokens > max_total_len {
            return Err(Error::Model(format!(
                "prompt + max_new_tokens exceeds cache length: {prompt_len} + {} > {max_total_len}",
                self.max_new_tokens
            )));
        }

        self.kv_cache.reset();
        reset_attention_mask(&mut self.attention_mask, prompt_len)?;
        let positions = Array1::from_iter((0..prompt_len).map(|i| i as i64));
        let mut logits = self.model.run_llm(
            prompt_embeds,
            &self.attention_mask,
            &positions,
            &mut self.kv_cache,
            0,
        )?;

        let block_from = self.model.config.asr_block_token_id_from;
        let mut generated = Vec::new();
        let mut total_len = prompt_len;

        for _ in 0..self.max_new_tokens {
            let next = argmax_masked(logits.as_slice().unwrap(), &self.bad_token_ids, block_from)?;
            generated.push(next);
            if self.stop_ids.contains(&next) {
                break;
            }

            self.model.embed_one(next as i64, &mut self.step_embed)?;
            grow_attention_mask(&mut self.attention_mask, total_len + 1)?;
            self.step_pos[0] = total_len as i64;
            logits = self.model.run_llm(
                &self.step_embed,
                &self.attention_mask,
                &self.step_pos,
                &mut self.kv_cache,
                total_len,
            )?;
            total_len += 1;
        }

        Ok(generated)
    }
}

impl Transcriber for ArkAsr {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        _mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        self.transcribe_samples_inner(audio, sample_rate, channels)
    }
}

fn load_preprocessor(model_dir: &Path) -> Result<PreprocessorConfig> {
    #[derive(serde::Deserialize)]
    struct WhisperPreprocessorJson {
        feature_extractor_type: Option<String>,
        feature_size: usize,
        hop_length: usize,
        n_fft: usize,
        #[serde(default)]
        padding_side: Option<String>,
        #[serde(default)]
        padding_value: Option<f32>,
        sampling_rate: usize,
        #[serde(default)]
        win_length: Option<usize>,
    }

    for candidate in [
        model_dir.join("preprocessor_config.json"),
        model_dir.join("model").join("preprocessor_config.json"),
    ] {
        if !candidate.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&candidate)?;
        let json: WhisperPreprocessorJson = serde_json::from_str(&text).map_err(|e| {
            Error::Config(format!("Failed to parse {}: {e}", candidate.display()))
        })?;
        let n_fft = json.n_fft;
        return Ok(PreprocessorConfig {
            feature_extractor_type: json
                .feature_extractor_type
                .unwrap_or_else(|| "WhisperFeatureExtractor".into()),
            feature_size: json.feature_size,
            hop_length: json.hop_length,
            n_fft,
            padding_side: json.padding_side.unwrap_or_else(|| "right".into()),
            padding_value: json.padding_value.unwrap_or(0.0),
            preemphasis: 0.0,
            processor_class: "ArkasrProcessor".into(),
            return_attention_mask: false,
            sampling_rate: json.sampling_rate,
            win_length: json.win_length.unwrap_or(n_fft),
        });
    }

    Err(Error::Config(format!(
        "preprocessor_config.json not found in {}",
        model_dir.display()
    )))
}

fn audio_token_count(mel_frames: usize, merge_factor: usize) -> usize {
    let downsampled = (mel_frames + 1) / 2;
    (downsampled / merge_factor).max(1)
}

fn require_token(tokenizer: &Tokenizer, token: &str) -> Result<i64> {
    tokenizer
        .token_to_id(token)
        .map(|id| id as i64)
        .ok_or_else(|| Error::Tokenizer(format!("tokenizer is missing {token}")))
}

fn build_prompt_pieces(tokenizer: &Tokenizer, user_text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
    let prefix = vec![
        require_token(tokenizer, "<|user|>")?,
        require_token(tokenizer, "<|begin_of_audio|>")?,
    ];
    let mut suffix = vec![require_token(tokenizer, "<|end_of_audio|>")?];
    let encoding = tokenizer
        .encode(user_text, false)
        .map_err(|e| Error::Tokenizer(format!("Failed to encode prompt text: {e}")))?;
    suffix.extend(encoding.get_ids().iter().map(|&id| id as i64));
    suffix.push(require_token(tokenizer, "<|assistant|>")?);
    Ok((prefix, suffix))
}

fn assemble_prompt_ids(
    prefix: &[i64],
    audio_token_id: i64,
    n_audio_tokens: usize,
    suffix: &[i64],
) -> Vec<i64> {
    let mut ids = Vec::with_capacity(prefix.len() + n_audio_tokens + suffix.len());
    ids.extend_from_slice(prefix);
    ids.resize(prefix.len() + n_audio_tokens, audio_token_id);
    ids.extend_from_slice(suffix);
    ids
}

fn inject_audio(
    input_ids: &Array2<i64>,
    embeds: &mut Array3<f32>,
    audio_embeds: &Array3<f32>,
    audio_token_id: i64,
) -> Result<()> {
    let seq = input_ids.shape()[1];
    let hidden = embeds.shape()[2];
    if audio_embeds.shape()[2] != hidden {
        return Err(Error::Model(format!(
            "audio embed hidden {} != token embed hidden {hidden}",
            audio_embeds.shape()[2]
        )));
    }
    let positions: Vec<usize> = (0..seq)
        .filter(|&i| input_ids[[0, i]] == audio_token_id)
        .collect();
    if positions.is_empty() {
        return Ok(());
    }
    let needed = positions.len();
    let avail = audio_embeds.shape()[1];
    for (offset, &pos) in positions.iter().enumerate() {
        if offset < avail {
            embeds
                .slice_mut(ndarray::s![0, pos, ..])
                .assign(&audio_embeds.slice(ndarray::s![0, offset, ..]));
        }
        // remaining positions stay as the placeholder embedding (zero-pad in the Python
        // runtime when audio frames < placeholders; here leftover ids keep token embeds)
        let _ = needed;
    }
    Ok(())
}

fn argmax_masked(scores: &[f32], bad_ids: &HashSet<u32>, block_from: i64) -> Result<u32> {
    let end = if block_from >= 0 {
        (block_from as usize).min(scores.len())
    } else {
        scores.len()
    };
    if end == 0 {
        return Err(Error::Model("empty vocabulary logits".into()));
    }
    let mut best_i = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in scores[..end].iter().enumerate() {
        if !v.is_finite() || bad_ids.contains(&(i as u32)) {
            continue;
        }
        if v > best {
            best = v;
            best_i = i as u32;
        }
    }
    if !best.is_finite() {
        return Err(Error::Model("all logits were masked".into()));
    }
    Ok(best_i)
}

fn build_bad_token_ids(
    tokenizer: &Tokenizer,
    eos_id: Option<u32>,
    block_from: u32,
) -> HashSet<u32> {
    let mut bad = HashSet::new();
    for (id, token) in tokenizer.get_added_tokens_decoder() {
        if Some(id) == eos_id || id >= block_from {
            continue;
        }
        let content = token.content.as_str();
        if token.special || (content.starts_with('<') && content.ends_with('>')) {
            bad.insert(id);
        }
    }
    bad
}
