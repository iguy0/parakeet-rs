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

use crate::audio::{FeatureCache, extract_whisper_logmel};
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_ark::{
    ArkKvCache, ArkModel, find_tokenizer, grow_attention_mask, reset_attention_mask,
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
/// Minimum repeated n-gram length for [`ArkAsr::set_repeat_guard`].
const REPEAT_MIN_LEN: usize = 8;

/// What to do when audio is longer than [`ArkAsr::max_audio_seconds`].
///
/// The default is [`Truncate`](ArkLongAudioPolicy::Truncate), which keeps the
/// previous prefix-only behavior and writes a warning to stderr. The crate
/// does not depend on a logging facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArkLongAudioPolicy {
    /// Return [`Error::Audio`] instead of transcribing a prefix.
    Error,
    /// Keep the first `max_audio_seconds` and warn on stderr.
    #[default]
    Truncate,
}

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
    /// `blocked_mask[id]` is set for ids that argmax must skip.
    /// Length is `min(block_from, vocab)`.
    blocked_mask: Vec<bool>,
    stop_ids: HashSet<u32>,
    prompt_prefix: Vec<i64>,
    audio_token_id: i64,
    prompt_suffix: Vec<i64>,
    max_new_tokens: usize,
    max_audio_seconds: u32,
    long_audio_policy: ArkLongAudioPolicy,
    repeat_guard: bool,
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
        let stop_ids = model
            .config
            .manifest
            .stop_token_ids
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| id as u32)
            .collect();

        let vocab = tokenizer.get_vocab_size(true);
        let block_from = model.config.asr_block_token_id_from.max(0) as u32;
        let blocked_mask = build_blocked_mask(&tokenizer, eos_id, block_from, vocab);

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
            blocked_mask,
            stop_ids,
            prompt_prefix,
            audio_token_id,
            prompt_suffix,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            max_audio_seconds: DEFAULT_MAX_AUDIO_SECONDS,
            long_audio_policy: ArkLongAudioPolicy::Truncate,
            repeat_guard: false,
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

    pub fn long_audio_policy(&self) -> ArkLongAudioPolicy {
        self.long_audio_policy
    }

    pub fn set_long_audio_policy(&mut self, policy: ArkLongAudioPolicy) {
        self.long_audio_policy = policy;
    }

    /// Enable the n-gram repetition guard inside greedy decode.
    ///
    /// Off by default. When on, a repeated token n-gram of length at least 8
    /// ends generation and the repeated copy is dropped, matching the Cohere
    /// decoder's loop break.
    pub fn set_repeat_guard(&mut self, enabled: bool) {
        self.repeat_guard = enabled;
    }

    pub fn repeat_guard(&self) -> bool {
        self.repeat_guard
    }

    /// Transcribe raw 16 kHz f32 samples (mono or interleaved multi-channel).
    pub fn transcribe_audio(&mut self, audio: &[f32]) -> Result<String> {
        self.transcribe_samples_inner(audio.to_vec(), self.preprocessor.sampling_rate as u32, 1)
            .map(|r| r.text)
    }

    /// Transcribe audio that may be longer than [`Self::max_audio_seconds`].
    ///
    /// The clip is split into windows of at most `max_audio_seconds`. Each cut
    /// is the start of the lowest-energy 100 ms frame in the last 5 seconds of
    /// the window. Chunk transcripts are joined with a single space.
    pub fn transcribe_long(&mut self, audio: &[f32]) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        let sample_rate = self.preprocessor.sampling_rate as u32;
        let max_samples = (self.max_audio_seconds as usize).saturating_mul(sample_rate as usize);
        if max_samples == 0 {
            return Err(Error::Audio(
                "max_audio_seconds covers 0 samples at the configured sample rate".into(),
            ));
        }
        let mut offset = 0;
        let mut parts = Vec::new();
        while offset < audio.len() {
            let remaining = &audio[offset..];
            let mut end = if remaining.len() <= max_samples {
                remaining.len()
            } else {
                split_at_quiet_point(remaining, max_samples, sample_rate)
            };
            if end == 0 || end > remaining.len() {
                end = max_samples.min(remaining.len());
            }
            let text = self
                .transcribe_samples_inner(remaining[..end].to_vec(), sample_rate, 1)?
                .text;
            if !text.is_empty() {
                parts.push(text);
            }
            offset += end;
        }
        Ok(parts.join(" "))
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

        let max_samples =
            (self.max_audio_seconds as usize).saturating_mul(self.preprocessor.sampling_rate);
        let kept = enforce_long_audio_policy(audio.len(), max_samples, self.long_audio_policy)?;
        if kept < audio.len() {
            let sr = self.preprocessor.sampling_rate.max(1) as f32;
            eprintln!(
                "ark-asr: audio is {:.2}s, which exceeds max_audio_seconds={}; truncating to {:.2}s",
                audio.len() as f32 / sr,
                self.max_audio_seconds,
                kept as f32 / sr
            );
            audio.truncate(kept);
        }
        if audio.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                tokens: Vec::new(),
            });
        }

        let features = extract_whisper_logmel(
            &audio,
            &self.feature_cache,
            &self.preprocessor,
            WHISPER_FLOOR_DB,
        )?;
        let audios = features
            .insert_axis(Axis(0))
            .as_standard_layout()
            .to_owned();

        let audio_embeds = self.model.encode_audio(&audios)?;
        let n_audio_tokens = audio_embeds.shape()[1];
        let input_ids = assemble_prompt_ids(
            &self.prompt_prefix,
            self.audio_token_id,
            n_audio_tokens,
            &self.prompt_suffix,
        );
        let seq = input_ids.len();
        let max_total_len = self.model.config.llm.max_total_len;
        if seq + self.max_new_tokens > max_total_len {
            return Err(Error::Model(format!(
                "prompt + max_new_tokens exceeds cache length: {seq} + {} > {max_total_len}",
                self.max_new_tokens
            )));
        }
        let ids = Array2::from_shape_vec((1, seq), input_ids.clone())
            .map_err(|e| Error::Model(format!("Failed to shape input_ids: {e}")))?;

        let mut embeds = self.model.embed_tokens(&input_ids)?;
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

        let mut generated = Vec::new();
        let mut total_len = prompt_len;

        for _ in 0..self.max_new_tokens {
            let next = argmax_masked(logits.as_slice().unwrap(), &self.blocked_mask)?;
            generated.push(next);
            if self.stop_ids.contains(&next) {
                break;
            }
            if self.repeat_guard
                && let Some(repeat_len) = find_ngram_repetition(&generated, REPEAT_MIN_LEN)
            {
                generated.truncate(generated.len() - repeat_len);
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
        let json: WhisperPreprocessorJson = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("Failed to parse {}: {e}", candidate.display())))?;
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
    let avail = audio_embeds.shape()[1];
    if positions.len() != avail {
        return Err(Error::Model(format!(
            "audio placeholder count {} != encoder embedding count {avail}",
            positions.len()
        )));
    }
    for (offset, &pos) in positions.iter().enumerate() {
        embeds
            .slice_mut(ndarray::s![0, pos, ..])
            .assign(&audio_embeds.slice(ndarray::s![0, offset, ..]));
    }
    Ok(())
}

fn argmax_masked(scores: &[f32], blocked: &[bool]) -> Result<u32> {
    let end = blocked.len().min(scores.len());
    if end == 0 {
        return Err(Error::Model("empty vocabulary logits".into()));
    }
    let mut best_i = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in scores[..end].iter().enumerate() {
        if !v.is_finite() || blocked[i] {
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

fn build_blocked_mask(
    tokenizer: &Tokenizer,
    eos_id: Option<u32>,
    block_from: u32,
    vocab: usize,
) -> Vec<bool> {
    let limit = (block_from as usize).min(vocab);
    let mut mask = vec![false; limit];
    for (id, token) in tokenizer.get_added_tokens_decoder() {
        if Some(id) == eos_id || (id as usize) >= limit {
            continue;
        }
        let content = token.content.as_str();
        if token.special || (content.starts_with('<') && content.ends_with('>')) {
            mask[id as usize] = true;
        }
    }
    mask
}

/// Keep `sample_count` when it fits. [`ArkLongAudioPolicy::Truncate`] returns
/// `max_samples`. [`ArkLongAudioPolicy::Error`] returns [`Error::Audio`].
fn enforce_long_audio_policy(
    sample_count: usize,
    max_samples: usize,
    policy: ArkLongAudioPolicy,
) -> Result<usize> {
    if sample_count <= max_samples {
        return Ok(sample_count);
    }
    match policy {
        ArkLongAudioPolicy::Truncate => Ok(max_samples),
        ArkLongAudioPolicy::Error => Err(Error::Audio(format!(
            "audio is {sample_count} samples, which exceeds the max_audio_seconds limit of {max_samples} samples"
        ))),
    }
}

fn split_at_quiet_point(audio: &[f32], window_samples: usize, sample_rate: u32) -> usize {
    crate::audio::split_at_quiet_point(audio, window_samples, sample_rate)
}

/// Returns the length of a repeated tail when the last `repeat_len` tokens
/// (`repeat_len >= min_len`) are an exact copy of the preceding segment.
fn find_ngram_repetition<T: PartialEq>(tokens: &[T], min_len: usize) -> Option<usize> {
    let n = tokens.len();
    if min_len == 0 || n < min_len.saturating_mul(2) {
        return None;
    }
    for repeat_len in min_len..=(n / 2) {
        let tail = &tokens[n - repeat_len..];
        let prev = &tokens[n - 2 * repeat_len..n - repeat_len];
        if tail == prev {
            return Some(repeat_len);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 16) as u32
        }
    }

    fn mask_from_ids(limit: usize, bad: &HashSet<u32>) -> Vec<bool> {
        let mut mask = vec![false; limit];
        for &id in bad {
            if (id as usize) < limit {
                mask[id as usize] = true;
            }
        }
        mask
    }

    fn argmax_masked_hashset(
        scores: &[f32],
        bad_ids: &HashSet<u32>,
        block_from: i64,
    ) -> Result<u32> {
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

    #[test]
    fn argmax_masked_matches_hashset_reference() {
        let mut lcg = Lcg(0xA11C_E5);
        for _trial in 0..32 {
            let vocab = 384;
            let block_from = 300i64;
            let mut scores = Vec::with_capacity(vocab);
            let mut bad = HashSet::new();
            for i in 0..vocab {
                let raw = lcg.next_u32();
                let value = if raw % 23 == 0 {
                    f32::NAN
                } else if raw % 29 == 0 {
                    f32::NEG_INFINITY
                } else {
                    (raw as f32) * (10.0 / u32::MAX as f32) - 5.0
                };
                scores.push(value);
                if raw % 17 == 0 {
                    bad.insert(i as u32);
                }
            }
            let limit = (block_from as usize).min(scores.len());
            let mask = mask_from_ids(limit, &bad);
            let masked = argmax_masked(&scores, &mask);
            let reference = argmax_masked_hashset(&scores, &bad, block_from);
            match (masked, reference) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(Error::Model(_)), Err(Error::Model(_))) => {}
                (a, b) => panic!("mask {a:?} != hashset {b:?}"),
            }
        }
    }

    #[test]
    fn argmax_masked_all_masked_errors() {
        let scores = [0.1f32, 1.5, -0.3, 4.0];
        let mask = vec![true; scores.len()];
        let err = argmax_masked(&scores, &mask).unwrap_err();
        assert!(matches!(err, Error::Model(_)));
    }

    #[test]
    fn assemble_prompt_ids_uses_actual_count() {
        let prefix = [1i64, 2];
        let suffix = [9i64];
        let audio_token_id = 5i64;
        for n in [0usize, 1, 17] {
            let ids = assemble_prompt_ids(&prefix, audio_token_id, n, &suffix);
            assert_eq!(ids.len(), prefix.len() + n + suffix.len());
            assert_eq!(&ids[..prefix.len()], &prefix);
            assert_eq!(&ids[prefix.len() + n..], &suffix);
            assert_eq!(ids.iter().filter(|&&id| id == audio_token_id).count(), n);
        }
    }

    #[test]
    fn long_audio_policy_error_returns_audio_error() {
        let err = enforce_long_audio_policy(48_000, 16_000, ArkLongAudioPolicy::Error).unwrap_err();
        assert!(matches!(err, Error::Audio(_)));
        assert_eq!(
            enforce_long_audio_policy(16_000, 16_000, ArkLongAudioPolicy::Error).unwrap(),
            16_000
        );
        assert_eq!(
            enforce_long_audio_policy(48_000, 16_000, ArkLongAudioPolicy::Truncate).unwrap(),
            16_000
        );
    }

    #[test]
    fn split_at_quiet_point_lands_in_gap() {
        let sample_rate = 16_000u32;
        let window = 10 * sample_rate as usize;
        let total = 12 * sample_rate as usize;
        let mut audio = vec![0.25f32; total];
        let gap_start = (8.5 * sample_rate as f32) as usize;
        let gap_len = (0.4 * sample_rate as f32) as usize;
        let gap_end = gap_start + gap_len;
        audio[gap_start..gap_end].fill(0.0);
        let split = split_at_quiet_point(&audio, window, sample_rate);
        assert!(
            split >= gap_start && split < gap_end,
            "split {split} not in gap {gap_start}..{gap_end}"
        );
        assert!(split < window);
    }

    #[test]
    fn split_at_quiet_point_within_window_returns_len() {
        let audio = vec![0.2f32; 1_000];
        assert_eq!(split_at_quiet_point(&audio, 8_000, 16_000), audio.len());
    }

    #[test]
    fn repeat_guard_truncates_repeated_ngram() {
        let mut tokens: Vec<u32> = (0..8).collect();
        tokens.extend_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let repeat = find_ngram_repetition(&tokens, REPEAT_MIN_LEN).expect("repeated 8-gram");
        assert_eq!(repeat, 8);
        tokens.truncate(tokens.len() - repeat);
        assert_eq!(tokens, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(find_ngram_repetition(&[1u32, 2, 1, 2], REPEAT_MIN_LEN).is_none());
    }

    #[test]
    #[ignore = "micro-benchmark: boolean mask vs HashSet argmax"]
    fn bench_argmax_masked() {
        const N: usize = 151_670;
        const ITERS: usize = 1000;
        let mut lcg = Lcg(0x51A5_7);
        let mut scores = Vec::with_capacity(N);
        for _ in 0..N {
            let raw = lcg.next_u32();
            scores.push((raw as f32) * (1.0 / u32::MAX as f32));
        }
        let mut bad = HashSet::new();
        for id in 0..32 {
            bad.insert(id);
        }
        for id in (0..N).step_by(4096) {
            bad.insert(id as u32);
        }
        let mask = mask_from_ids(N, &bad);
        let block_from = N as i64;

        for _ in 0..10 {
            std::hint::black_box(argmax_masked(&scores, &mask).unwrap());
            std::hint::black_box(argmax_masked_hashset(&scores, &bad, block_from).unwrap());
        }

        let time_mask = || {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                std::hint::black_box(argmax_masked(&scores, &mask).unwrap());
            }
            t0.elapsed()
        };
        let time_hash = || {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                std::hint::black_box(argmax_masked_hashset(&scores, &bad, block_from).unwrap());
            }
            t0.elapsed()
        };
        let mask_dur = time_mask().min(time_mask());
        let hash_dur = time_hash().min(time_hash());
        let speedup = hash_dur.as_secs_f64() / mask_dur.as_secs_f64();
        eprintln!(
            "bench_argmax_masked: mask {:.3} ms, hashset {:.3} ms, speedup {:.2}x ({N} logits x {ITERS} iters)",
            mask_dur.as_secs_f64() * 1000.0,
            hash_dur.as_secs_f64() * 1000.0,
            speedup
        );
        assert!(
            speedup >= 5.0,
            "argmax mask speedup {speedup:.2}x is below the 5x target"
        );
    }
}
