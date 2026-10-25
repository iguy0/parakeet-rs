//! Cohere Transcribe ASR engine.
//!
//! 2B parameter Conformer encoder + lightweight Transformer decoder.
//! Takes raw 16 kHz mono f32 audio, returns transcribed text.
//! Supports 14 languages via explicit language selection.
//!
//! Consumes a HuggingFace-standard ONNX export: the encoder takes
//! pre-computed log-mel features and the decoder is a merged graph using
//! the standard `past_key_values.N.{decoder,encoder}.{key,value}` cache
//! convention.
//!
//! [`onnx-community/cohere-transcribe-03-2026-ONNX`](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
//! is one such export (FP32, FP16, INT8, and INT4 variants available).
//! To produce your own from the upstream PyTorch checkpoint, install
//! [`optimum`](https://github.com/huggingface/optimum) and run:
//!
//! ```sh
//! optimum-cli export onnx \
//!     --model CohereLabs/cohere-transcribe-03-2026 \
//!     --task automatic-speech-recognition-with-past \
//!     ./cohere-onnx
//! ```
//!
//! No custom export script is needed for the merged graph - the `cohere_asr`
//! model type is supported by Optimum's standard exporter. An optional
//! steady-state decoder (`decoder_model_steady[_quantized].onnx`, produced
//! by `scripts/export_cohere_steady_decoder.py`) skips the first-step `If`
//! on later tokens. It is used automatically when the file is present.

use crate::audio::{FeatureCache, extract_features_with_cache};
use crate::config::PreprocessorConfig;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_cohere::{CohereEncoderOutput, CohereModel, N_MELS};
use ndarray::{Array2, Axis};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// Special token literals that drive the decoder prompt. The canonical
// prompt structure is produced by CohereAsrProcessor in transformers and
// matches the shape model.generate() expects:
//
//   [▁, <|startofcontext|>, <|startoftranscript|>, <|emo:undefined|>,
//    <|src_lang|>, <|tgt_lang|>, <|pnc|>/<|nopnc|>, <|itn|>/<|noitn|>,
//    <|notimestamp|>, <|nodiarize|>]
//
// `▁` (SentencePiece word boundary) is the model's
// `decoder_start_token_id`. Source and target languages are both the
// same ISO code for pure transcription. Emotion and diarisation are
// fixed to "undefined"/"no" since the model's ASR task does not use
// them.
const TOKEN_WORD_BOUNDARY: &str = "\u{2581}";
const TOKEN_STARTOFCONTEXT: &str = "<|startofcontext|>";
const TOKEN_STARTOFTRANSCRIPT: &str = "<|startoftranscript|>";
const TOKEN_EMO_UNDEFINED: &str = "<|emo:undefined|>";
const TOKEN_ENDOFTEXT: &str = "<|endoftext|>";
const TOKEN_PNC: &str = "<|pnc|>";
const TOKEN_NOPNC: &str = "<|nopnc|>";
const TOKEN_NOTIMESTAMP: &str = "<|notimestamp|>";
const TOKEN_NODIARIZE: &str = "<|nodiarize|>";
const TOKEN_ITN: &str = "<|itn|>";
const TOKEN_NOITN: &str = "<|noitn|>";

/// Hard upper bound on output tokens enforced by the model
/// (`max_position_embeddings = 1024`). The user-configurable
/// `max_decode_tokens` cannot exceed this.
const MAX_DECODE_TOKENS_LIMIT: usize = 1024;

/// Default maximum output tokens per transcription. 512 is enough for
/// ~40 seconds of typical speech at the model's tokenisation rate, which
/// covers the training range (`max_audio_clip_s = 35`).
const DEFAULT_MAX_DECODE_TOKENS: usize = 512;

/// Training chunk length recorded in `preprocessor_config.json`
/// (`max_audio_clip_s`). Vendor docs also set `overlap_chunk_second = 5`,
/// and positional embeddings cap the encoder at about 5000 frames. Audio
/// longer than this is not guaranteed to transcribe; use
/// [`CohereASR::transcribe_audio_long`].
const TRAINING_CHUNK_SECS: f32 = 35.0;

// The 14 languages officially supported by Cohere Transcribe
// (cohere-transcribe-03-2026). The tokenizer contains `<|xx|>` placeholders
// for ~180 ISO codes but only these have trained weights.
// See https://docs.cohere.com/docs/transcribe.
const SUPPORTED_LANGUAGES: &[&str] = &[
    "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh",
];

struct DecoderTokens {
    decoder_start: i64,
    startofcontext: i64,
    sot: i64,
    emo_undefined: i64,
    eos: i64,
    pnc: i64,
    nopnc: i64,
    notimestamp: i64,
    nodiarize: i64,
    itn: i64,
    noitn: i64,
}

impl DecoderTokens {
    fn resolve(tokenizer: &Tokenizer) -> Result<Self> {
        Ok(Self {
            decoder_start: require_token(tokenizer, TOKEN_WORD_BOUNDARY)?,
            startofcontext: require_token(tokenizer, TOKEN_STARTOFCONTEXT)?,
            sot: require_token(tokenizer, TOKEN_STARTOFTRANSCRIPT)?,
            emo_undefined: require_token(tokenizer, TOKEN_EMO_UNDEFINED)?,
            eos: require_token(tokenizer, TOKEN_ENDOFTEXT)?,
            pnc: require_token(tokenizer, TOKEN_PNC)?,
            nopnc: require_token(tokenizer, TOKEN_NOPNC)?,
            notimestamp: require_token(tokenizer, TOKEN_NOTIMESTAMP)?,
            nodiarize: require_token(tokenizer, TOKEN_NODIARIZE)?,
            itn: require_token(tokenizer, TOKEN_ITN)?,
            noitn: require_token(tokenizer, TOKEN_NOITN)?,
        })
    }
}

/// Values on here are mirror from `preprocessor_config.json` in the upstream HF export. they are baked
/// into the encoder's ONNX graph (feature_size=128, hop=160, etc.) and
/// are not user tunable, so we hardcode them rather than requiring the
/// file on disk. If they share onnx script, we could consider something just like we did for the sorftformer.
fn cohere_preprocessor_config() -> PreprocessorConfig {
    PreprocessorConfig {
        feature_extractor_type: "CohereAsrFeatureExtractor".to_string(),
        feature_size: N_MELS,
        hop_length: 160,
        n_fft: 512,
        padding_side: "right".to_string(),
        padding_value: 0.0,
        preemphasis: 0.97,
        processor_class: "CohereAsrProcessor".to_string(),
        return_attention_mask: true,
        sampling_rate: 16000,
        win_length: 400,
    }
}

/// Cohere Transcribe ASR engine.
pub struct CohereASR {
    model: CohereModel,
    tokenizer: Tokenizer,
    /// Mel/STFT parameters (hardcoded — see [`cohere_preprocessor_config`]).
    preprocessor: PreprocessorConfig,
    /// Pre-built mel filterbank + FFT plan ->> reused across every transcribe call.
    feature_cache: FeatureCache,
    /// Map of supported ISO 639-1 language code -> language token id.
    lang_tokens: HashMap<String, i64>,
    tokens: DecoderTokens,
    /// Maximum number of tokens to generate per `transcribe_audio` call.
    /// Defaults to [`DEFAULT_MAX_DECODE_TOKENS`] (512). Capped at
    /// [`MAX_DECODE_TOKENS_LIMIT`] (1024).
    max_decode_tokens: usize,
    /// Window length used by [`Self::transcribe_audio_long`].
    /// Defaults to [`TRAINING_CHUNK_SECS`].
    max_chunk_secs: f32,
}

impl CohereASR {
    /// Load the Cohere Transcribe model from a directory.
    ///
    /// The directory must contain (flat or under `onnx/`):
    /// - one of `encoder_model[_quantized|_fp16].onnx` (+ `.onnx_data` companions)
    /// - one of `decoder_model_merged[_quantized|_fp16].onnx` (+ `.onnx_data`)
    /// - `tokenizer.json`
    ///
    /// Optional, same flat or `onnx/` layout:
    /// `decoder_model_steady.onnx` or `decoder_model_steady_quantized.onnx`.
    /// When present, decode steps after the first use that graph. Disable
    /// with [`Self::set_use_steady_decoder`].
    ///
    /// parameters are hardcoded since they are fixed by the encoder graph.
    ///
    /// This layout matches the [`onnx-community/cohere-transcribe-03-2026-ONNX`](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
    /// HF repository.
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let exec = exec_config.unwrap_or_default();

        let model = CohereModel::from_pretrained(model_dir, exec)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::Config(format!(
                "Missing tokenizer.json in {}",
                model_dir.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer.json: {e}")))?;

        let preprocessor = cohere_preprocessor_config();
        let feature_cache = FeatureCache::from_config(&preprocessor);
        let tokens = DecoderTokens::resolve(&tokenizer)?;

        let mut lang_tokens = HashMap::with_capacity(SUPPORTED_LANGUAGES.len());
        for code in SUPPORTED_LANGUAGES {
            let lit = format!("<|{code}|>");
            if let Some(id) = tokenizer.token_to_id(&lit) {
                lang_tokens.insert((*code).to_string(), id as i64);
            }
        }
        if lang_tokens.is_empty() {
            return Err(Error::Tokenizer(
                "No supported language tokens found in tokenizer.json".into(),
            ));
        }

        Ok(Self {
            model,
            tokenizer,
            preprocessor,
            feature_cache,
            lang_tokens,
            tokens,
            max_decode_tokens: DEFAULT_MAX_DECODE_TOKENS,
            max_chunk_secs: TRAINING_CHUNK_SECS,
        })
    }

    /// Training chunk length (in seconds) recorded in the upstream
    /// `preprocessor_config.json` (`max_audio_clip_s`). Audio past this
    /// length is not guaranteed; [`Self::transcribe_audio_long`] splits
    /// on that window.
    pub fn training_chunk_secs(&self) -> f32 {
        TRAINING_CHUNK_SECS
    }

    /// Window length, in seconds, used by [`Self::transcribe_audio_long`].
    pub fn max_chunk_secs(&self) -> f32 {
        self.max_chunk_secs
    }

    /// Set the window length used by [`Self::transcribe_audio_long`].
    ///
    /// The default is [`Self::training_chunk_secs`] (35). Values that are
    /// not finite or not positive make `transcribe_audio_long` return
    /// [`Error::Config`].
    pub fn set_max_chunk_secs(&mut self, secs: f32) {
        self.max_chunk_secs = secs;
    }

    /// Current maximum number of tokens the decoder will emit per call.
    pub fn max_decode_tokens(&self) -> usize {
        self.max_decode_tokens
    }

    /// Set the maximum number of tokens the decoder will emit per call.
    /// Values above the model's hard limit (1024) are clamped.
    pub fn set_max_decode_tokens(&mut self, max: usize) {
        self.max_decode_tokens = max.clamp(1, MAX_DECODE_TOKENS_LIMIT);
    }

    /// Record per-step decoder durations for the next transcription.
    ///
    /// Default is off. Enabling clears any samples already stored. Read them
    /// with [`Self::decoder_step_millis`] after `transcribe_audio`.
    pub fn set_record_decoder_steps(&mut self, enabled: bool) {
        self.model.set_record_steps(enabled);
    }

    /// Milliseconds spent inside each decoder step of the last recorded call.
    pub fn decoder_step_millis(&self) -> &[f64] {
        self.model.step_millis()
    }

    /// Parallel to [`Self::decoder_step_millis`]: true when that step ran the
    /// steady-state decoder graph.
    pub fn decoder_step_used_steady(&self) -> &[bool] {
        self.model.step_used_steady()
    }

    /// Use the steady-state decoder for steps after the first.
    ///
    /// Default is on when `decoder_model_steady[_quantized].onnx` was found
    /// at load. Passing `false` forces the merged graph for every step.
    /// Passing `true` does nothing when that file was not loaded.
    pub fn set_use_steady_decoder(&mut self, enabled: bool) {
        self.model.set_use_steady_decoder(enabled);
    }

    /// Whether a steady-state decoder graph was loaded.
    pub fn steady_decoder_loaded(&self) -> bool {
        self.model.steady_decoder_loaded()
    }

    /// Transcribe raw 16 kHz mono f32 audio samples.
    ///
    /// `language` is an ISO 639-1 code (e.g. `"en"`, `"fr"`, `"de"`, `"ja"`).
    /// `punctuation` controls whether output includes punctuation and
    /// capitalisation. `itn` enables inverse text normalisation
    /// (e.g. "twenty three" -> "23").
    ///
    /// # Long-form audio
    ///
    /// The model was trained on clips up to 35 s ([`Self::training_chunk_secs`]).
    /// The Hugging Face model card sets `max_audio_clip_s = 35` and
    /// `overlap_chunk_second = 5`, and positional embeddings cover about
    /// 5000 encoder frames. Audio longer than about 35 s is not guaranteed
    /// to transcribe correctly. Use [`Self::transcribe_audio_long`] for
    /// longer clips.
    pub fn transcribe_audio(
        &mut self,
        audio: &[f32],
        language: &str,
        punctuation: bool,
        itn: bool,
    ) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }

        let lang_token = self.lang_tokens.get(language).copied().ok_or_else(|| {
            Error::Config(format!(
                "Unsupported language '{}'. Supported: {:?}",
                language,
                self.supported_languages()
            ))
        })?;

        // 1. Mel features. extract_features_raw returns [T, N_MELS] after
        //    preemphasis + STFT + log-mel + per-feature normalisation, which
        //    matches the CohereAsrFeatureExtractor pipeline. We add a batch
        //    axis to get [1, T, N_MELS] for the encoder.
        //
        // `as_standard_layout().to_owned()` is required because `insert_axis`
        // on a view may produce non-standard strides, but ort::TensorRef
        // needs C-contiguous memory.
        let mel_2d = extract_features_with_cache(
            audio.to_vec(),
            self.preprocessor.sampling_rate as u32,
            1,
            &self.preprocessor,
            &self.feature_cache,
        )?;
        let mel_3d = mel_2d.insert_axis(Axis(0)).as_standard_layout().to_owned();

        // 2. Encoder
        let encoder_out = self.model.run_encoder(&mel_3d)?;

        // 3. Build the canonical Cohere decoder prompt matching what
        //    CohereAsrProcessor in transformers produces. The source and
        //    target language tokens are both the caller's `language` code
        //    since this is pure transcription (no translation).
        let t = &self.tokens;
        let pnc_token = if punctuation { t.pnc } else { t.nopnc };
        let itn_token = if itn { t.itn } else { t.noitn };
        let prompt = vec![
            t.decoder_start,
            t.startofcontext,
            t.sot,
            t.emo_undefined,
            lang_token,
            lang_token,
            pnc_token,
            itn_token,
            t.notimestamp,
            t.nodiarize,
        ];

        // 4. Greedy decode loop
        let token_ids = self.decode_greedy(&prompt, &encoder_out)?;

        // 5. Detokenise (skip special tokens)
        let text = self
            .tokenizer
            .decode(
                &token_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| Error::Tokenizer(format!("Failed to decode tokens: {e}")))?;

        // Strip leading stray punctuation the decoder sometimes emits
        // before the first real token.
        let cleaned = text
            .trim()
            .trim_start_matches(['.', '?', '!', ','])
            .trim()
            .to_string();

        Ok(cleaned)
    }

    /// Transcribe audio that may be longer than [`Self::max_chunk_secs`].
    ///
    /// Each window is at most `max_chunk_secs` (default 35). The cut is the
    /// quietest 100 ms frame in the last 5 s of the window. Chunk texts are
    /// joined with `""` for `ja` and `zh`, and with a single space otherwise.
    pub fn transcribe_audio_long(
        &mut self,
        audio: &[f32],
        language: &str,
        punctuation: bool,
        itn: bool,
    ) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        let sample_rate = self.preprocessor.sampling_rate as u32;
        let max_samples = chunk_sample_limit(self.max_chunk_secs, sample_rate)?;
        let ranges = chunk_ranges(audio, max_samples, sample_rate);
        let mut parts = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            let text = self.transcribe_audio(&audio[start..end], language, punctuation, itn)?;
            if !text.is_empty() {
                parts.push(text);
            }
        }
        Ok(join_chunk_texts(language, &parts))
    }

    /// Greedy autoregressive decode using the merged decoder's growing
    /// `past_key_values` cache. The first call feeds the prompt and lets
    /// the model populate the cross-attention encoder cache; subsequent
    /// calls feed one token at a time.
    fn decode_greedy(
        &mut self,
        prompt: &[i64],
        encoder_out: &CohereEncoderOutput,
    ) -> Result<Vec<i64>> {
        let mut past_kv = self.model.empty_past();
        let mut output_tokens: Vec<i64> = Vec::new();

        // First step: feed entire prompt
        let prompt_tensor = Array2::from_shape_vec((1, prompt.len()), prompt.to_vec())
            .map_err(|e| Error::Model(format!("Prompt tensor shape error: {e}")))?;
        let (logits, new_past) =
            self.model
                .run_decoder_step(&prompt_tensor, past_kv, encoder_out)?;
        past_kv = new_past;

        let mut next_token = argmax(logits.as_slice().unwrap());
        if next_token == self.tokens.eos {
            return Ok(output_tokens);
        }
        output_tokens.push(next_token);

        // Continue one token at a time up to the configured max.
        for _ in 1..self.max_decode_tokens {
            let token_tensor = Array2::from_shape_vec((1, 1), vec![next_token])
                .map_err(|e| Error::Model(format!("Token tensor shape error: {e}")))?;
            let (logits, new_past) =
                self.model
                    .run_decoder_step(&token_tensor, past_kv, encoder_out)?;
            past_kv = new_past;

            next_token = argmax(logits.as_slice().unwrap());
            if next_token == self.tokens.eos {
                break;
            }
            output_tokens.push(next_token);

            // Detect n-gram repetition: if the last N tokens match a
            // previous sequence the model is stuck in a loop.
            if let Some(repeat_len) = find_ngram_repetition(&output_tokens, 8) {
                output_tokens.truncate(output_tokens.len() - repeat_len);
                break;
            }
        }

        Ok(output_tokens)
    }

    /// Sorted list of supported ISO 639-1 language codes.
    pub fn supported_languages(&self) -> Vec<String> {
        let mut langs: Vec<String> = self.lang_tokens.keys().cloned().collect();
        langs.sort();
        langs
    }
}

/// Look up a special token id by literal, returning a clear error if it's
/// not present in the tokenizer vocabulary.
fn require_token(tokenizer: &Tokenizer, literal: &str) -> Result<i64> {
    tokenizer
        .token_to_id(literal)
        .map(|id| id as i64)
        .ok_or_else(|| Error::Tokenizer(format!("Tokenizer is missing required token {literal}")))
}

/// Check if the token sequence ends with a repeated n-gram of length
/// `>= min_len`. Returns `Some(repeat_len)` if the last `repeat_len` tokens
/// are an exact copy of the preceding segment.
fn find_ngram_repetition(tokens: &[i64], min_len: usize) -> Option<usize> {
    let n = tokens.len();
    if n < min_len * 2 {
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

/// Greedy argmax over a slice of f32 logits.
fn argmax(logits: &[f32]) -> i64 {
    crate::tensor_utils::argmax_f32(logits).0 as i64
}

fn chunk_sample_limit(max_chunk_secs: f32, sample_rate: u32) -> Result<usize> {
    if !max_chunk_secs.is_finite() || max_chunk_secs <= 0.0 || sample_rate == 0 {
        return Err(Error::Config(format!(
            "max_chunk_secs must be a positive finite number, got {max_chunk_secs}"
        )));
    }
    let samples = (max_chunk_secs * sample_rate as f32).round() as usize;
    if samples == 0 {
        return Err(Error::Config(
            "max_chunk_secs covers 0 samples at the configured sample rate".into(),
        ));
    }
    Ok(samples)
}

/// Half-open sample ranges covering `audio`. Every range is non-empty and
/// at most `max_samples` long. Cuts inside a window use the quietest 100 ms
/// frame in the last 5 s.
fn chunk_ranges(audio: &[f32], max_samples: usize, sample_rate: u32) -> Vec<(usize, usize)> {
    if audio.is_empty() || max_samples == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut offset = 0;
    while offset < audio.len() {
        let remaining = &audio[offset..];
        let mut end = crate::audio::split_at_quiet_point(remaining, max_samples, sample_rate);
        if end == 0 || end > remaining.len() {
            end = max_samples.min(remaining.len());
        }
        ranges.push((offset, offset + end));
        offset += end;
    }
    ranges
}

fn chunk_separator(language: &str) -> &'static str {
    match language {
        "ja" | "zh" => "",
        _ => " ",
    }
}

fn join_chunk_texts(language: &str, parts: &[String]) -> String {
    parts
        .iter()
        .filter(|part| !part.is_empty())
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(chunk_separator(language))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.9, 0.2]), 3);
        assert_eq!(argmax(&[1.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn test_supported_languages_count() {
        // Cohere Transcribe officially ships trained weights for 14 languages
        assert_eq!(SUPPORTED_LANGUAGES.len(), 14);
    }

    #[test]
    fn test_ngram_repetition_detection() {
        // No repetition
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 5, 6, 7, 8], 4), None);

        // Repeated 4-gram: [1,2,3,4] appears twice
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 1, 2, 3, 4], 4), Some(4));

        // Repeated 8-gram
        let mut tokens = vec![10, 20, 30, 40, 50, 60, 70, 80];
        tokens.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(find_ngram_repetition(&tokens, 8), Some(8));

        // Too short to detect
        assert_eq!(find_ngram_repetition(&[1, 2, 1, 2], 4), None);
    }

    #[test]
    fn split_chunks_cover_all_samples() {
        let sample_rate = 16_000u32;
        let max_samples = 40_000usize;
        let total = 100_000usize;
        let audio = vec![0.2f32; total];
        let ranges = chunk_ranges(&audio, max_samples, sample_rate);
        assert!(!ranges.is_empty());
        assert_eq!(ranges[0].0, 0);
        assert_eq!(ranges.last().unwrap().1, total);
        let mut cursor = 0usize;
        for (start, end) in &ranges {
            assert_eq!(*start, cursor);
            assert!(*end > *start, "empty chunk");
            assert!(end - start <= max_samples);
            cursor = *end;
        }
        let (last_start, last_end) = *ranges.last().unwrap();
        assert!(last_end > last_start);

        let short = vec![0.1f32; 100];
        let one = chunk_ranges(&short, max_samples, sample_rate);
        assert_eq!(one, vec![(0, 100)]);
        assert!(chunk_ranges(&[], max_samples, sample_rate).is_empty());
    }

    #[test]
    fn join_policy_cjk_vs_latin() {
        let parts = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(join_chunk_texts("ja", &parts), "helloworld");
        assert_eq!(join_chunk_texts("zh", &parts), "helloworld");
        assert_eq!(join_chunk_texts("en", &parts), "hello world");
        assert_eq!(chunk_separator("ja"), "");
        assert_eq!(chunk_separator("zh"), "");
        assert_eq!(chunk_separator("en"), " ");
        assert_eq!(
            join_chunk_texts("en", &["a".to_string(), String::new(), "b".to_string()]),
            "a b"
        );
    }
}
