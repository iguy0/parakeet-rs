use crate::audio::{self, FeatureCache};
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::decoder_tdt::ParakeetTDTDecoder;
use crate::decoding::{BeamConfig, DecodingStrategy};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_tdt::ParakeetTDTModel;
use crate::timestamps::{TimestampMode, process_timestamps, rebuild_text};
use crate::transcriber::Transcriber;
use crate::vocab::Vocabulary;
use std::path::{Path, PathBuf};

/// Parakeet TDT model for multilingual ASR
pub struct ParakeetTDT {
    model: ParakeetTDTModel,
    decoder: ParakeetTDTDecoder,
    preprocessor_config: PreprocessorConfig,
    feature_cache: FeatureCache,
    model_dir: PathBuf,
    decoding: DecodingStrategy,
    max_chunk_secs: Option<f32>,
}

/// Optional per-model defaults, read from the `parakeet_rs` object in the model directory's
/// `config.json`:
///
/// ```json
/// { "parakeet_rs": { "decoding": "greedy", "max_chunk_secs": 120 } }
/// ```
///
/// `decoding` is `"greedy"` or `"beam"` and applies only to constructors that do not take an
/// explicit [`DecodingStrategy`]. `max_chunk_secs` makes [`Transcriber::transcribe_samples`] split
/// long audio at quiet points, because the FastConformer encoder fails past roughly 4-5 minutes
/// per call. Missing file, object or keys keep the legacy behavior (greedy, no chunking).
#[derive(Debug, Default, Clone, PartialEq)]
struct TdtSidecar {
    decoding: Option<DecodingStrategy>,
    max_chunk_secs: Option<f32>,
}

impl TdtSidecar {
    fn read(dir: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
            return Self::default();
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            return Self::default();
        };
        let Some(obj) = json.get("parakeet_rs") else {
            return Self::default();
        };
        let decoding = match obj.get("decoding").and_then(|v| v.as_str()) {
            Some("greedy") => Some(DecodingStrategy::Greedy),
            Some("beam") => Some(DecodingStrategy::Beam(BeamConfig::default_tdt())),
            _ => None,
        };
        let max_chunk_secs = obj
            .get("max_chunk_secs")
            .and_then(|v| v.as_f64())
            .filter(|v| *v > 0.0)
            .map(|v| v as f32);
        Self {
            decoding,
            max_chunk_secs,
        }
    }
}

impl ParakeetTDT {
    /// Load Parakeet TDT model from path with optional configuration.
    ///
    /// # Arguments
    /// * `path` - Directory containing encoder-model.onnx, decoder_joint-model.onnx, and vocab.txt
    /// * `config` - Optional execution configuration (defaults to CPU if None)
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let joint_config = config.clone();
        Self::from_pretrained_full(path, config, joint_config, None)
    }

    /// Load Parakeet TDT with an explicit decoding strategy (greedy default, or beam).
    pub fn from_pretrained_with_decoding<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
        decoding: DecodingStrategy,
    ) -> Result<Self> {
        let joint_config = config.clone();
        Self::from_pretrained_full(path, config, joint_config, Some(decoding))
    }

    /// Load Parakeet TDT model from path, running the decoder/joint graph under its own execution
    /// configuration.
    ///
    /// The encoder and the decoder/joint are separate ONNX graphs and do not have to run on the
    /// same execution provider. Splitting them matters when a provider can take one graph and not
    /// the other, and the joint runs once per token, so where it runs is a decision of its own.
    ///
    /// # Arguments
    /// * `path` - Directory containing encoder-model.onnx, decoder_joint-model.onnx, and vocab.txt
    /// * `config` - Optional execution configuration for the encoder (defaults to CPU if None)
    /// * `joint_config` - Optional execution configuration for the decoder/joint (defaults to CPU
    ///   if None)
    pub fn from_pretrained_with_joint_config<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
        joint_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Self::from_pretrained_full(path, config, joint_config, None)
    }

    /// Load Parakeet TDT model from path, with an explicit decoder/joint execution
    /// configuration and decoding strategy (greedy default, or beam). See
    /// [`Self::from_pretrained_with_joint_config`] and [`Self::from_pretrained_with_decoding`]
    /// for details on each parameter.
    pub fn from_pretrained_with_joint_config_and_decoding<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
        joint_config: Option<ExecutionConfig>,
        decoding: DecodingStrategy,
    ) -> Result<Self> {
        Self::from_pretrained_full(path, config, joint_config, Some(decoding))
    }

    fn from_pretrained_full<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
        joint_config: Option<ExecutionConfig>,
        decoding: Option<DecodingStrategy>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let sidecar = TdtSidecar::read(path);
        let decoding = decoding
            .or(sidecar.decoding)
            .unwrap_or(DecodingStrategy::Greedy);

        if !path.is_dir() {
            return Err(Error::Config(format!(
                "TDT model path must be a directory: {}",
                path.display()
            )));
        }

        let vocab_path = path.join("vocab.txt");
        if !vocab_path.exists() {
            return Err(Error::Config(format!(
                "vocab.txt not found in {}",
                path.display()
            )));
        }

        // TDT-specific preprocessor config (128 features instead of 80)
        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        };

        let exec_config = config.unwrap_or_default();
        let joint_exec_config = joint_config.unwrap_or_default();

        // Load vocab first to get the actual vocabulary size
        let vocab = Vocabulary::from_file(&vocab_path)?;
        let vocab_size = vocab.size();

        let model = ParakeetTDTModel::from_pretrained_with_configs(
            path,
            exec_config,
            joint_exec_config,
            vocab_size,
        )?;
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let feature_cache = FeatureCache::from_config(&preprocessor_config);

        Ok(Self {
            model,
            decoder,
            preprocessor_config,
            feature_cache,
            model_dir: path.to_path_buf(),
            decoding,
            max_chunk_secs: sidecar.max_chunk_secs,
        })
    }

    pub fn decoding_strategy(&self) -> DecodingStrategy {
        self.decoding
    }

    pub fn set_decoding_strategy(&mut self, strategy: DecodingStrategy) {
        self.decoding = strategy;
    }

    /// Longest audio (seconds) sent through the encoder in one call. `None` (default unless the
    /// model's `config.json` sets `parakeet_rs.max_chunk_secs`) runs each clip in one call.
    pub fn max_chunk_secs(&self) -> Option<f32> {
        self.max_chunk_secs
    }

    /// Set or clear the chunk length used by `transcribe_samples`. Longer inputs are split at the
    /// quietest 100 ms frame in the last 5 s of each window and the results are joined.
    pub fn set_max_chunk_secs(&mut self, secs: Option<f32>) {
        self.max_chunk_secs = secs.filter(|v| *v > 0.0);
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }
}

impl ParakeetTDT {
    fn transcribe_one(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: TimestampMode,
    ) -> Result<TranscriptionResult> {
        let features = audio::extract_features_with_cache(
            audio,
            sample_rate,
            channels,
            &self.preprocessor_config,
            &self.feature_cache,
        )?;
        let (tokens, frame_indices, durations) =
            self.model.forward_with_decoding(features, self.decoding)?;

        let mut result = self.decoder.decode_with_timestamps(
            &tokens,
            &frame_indices,
            &durations,
            self.preprocessor_config.hop_length,
            self.preprocessor_config.sampling_rate,
        )?;

        // Apply timestamp mode conversion
        result.tokens = process_timestamps(&result.tokens, mode);

        // Rebuild full text from processed tokens
        result.text = rebuild_text(&result.tokens, mode);

        Ok(result)
    }
}

impl Transcriber for ParakeetTDT {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        let mode = mode.unwrap_or(TimestampMode::Tokens);
        let window = self
            .max_chunk_secs
            .map(|secs| (secs * sample_rate as f32) as usize)
            .filter(|w| *w > 0);
        let frames = audio.len() / channels.max(1) as usize;
        let Some(window) = window.filter(|w| frames > *w) else {
            return self.transcribe_one(audio, sample_rate, channels, mode);
        };

        // Long input: down-mix, split at quiet points, transcribe each piece, shift timestamps.
        let mono: Vec<f32> = if channels > 1 {
            audio
                .chunks_exact(channels as usize)
                .map(|f| f.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            audio
        };
        let mut merged = TranscriptionResult {
            text: String::new(),
            tokens: Vec::new(),
        };
        let mut offset = 0usize;
        while offset < mono.len() {
            let remaining = &mono[offset..];
            let mut end = audio::split_at_quiet_point(remaining, window, sample_rate);
            if end == 0 || end > remaining.len() {
                end = window.min(remaining.len());
            }
            let piece = self.transcribe_one(remaining[..end].to_vec(), sample_rate, 1, mode)?;
            let shift = offset as f32 / sample_rate as f32;
            merged.tokens.extend(piece.tokens.into_iter().map(|mut t| {
                t.start += shift;
                t.end += shift;
                t
            }));
            offset += end;
        }
        merged.text = rebuild_text(&merged.tokens, mode);
        Ok(merged)
    }
}
