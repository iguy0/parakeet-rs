use crate::audio::{self, FeatureCache};
use crate::config::PreprocessorConfig;
use crate::decoder::{ParakeetDecoder, TranscriptionResult};
use crate::decoding::DecodingStrategy;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model::ParakeetModel;
use crate::timestamps::{process_timestamps, rebuild_text, TimestampMode};
use crate::transcriber::Transcriber;
use std::path::{Path, PathBuf};

pub struct Parakeet {
    model: ParakeetModel,
    decoder: ParakeetDecoder,
    preprocessor_config: PreprocessorConfig,
    feature_cache: FeatureCache,
    model_dir: PathBuf,
    decoding: DecodingStrategy,
}

impl Parakeet {
    /// Load Parakeet model from path with optional configuration.
    ///
    /// # Arguments
    /// * `path` - Directory containing model files, or path to specific model file
    /// * `config` - Optional execution configuration (defaults to CPU if None)
    ///
    /// # Examples
    /// ```no_run
    /// use parakeet_rs::Parakeet;
    ///
    /// // Load from directory with CPU (default)
    /// let parakeet = Parakeet::from_pretrained(".", None)?;
    ///
    /// // Or load from specific model file
    /// let parakeet = Parakeet::from_pretrained("model_q4.onnx", None)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// For GPU acceleration, enable the corresponding feature (cuda, tensorrt, webgpu, etc.)
    /// and pass an `ExecutionConfig` with the desired execution provider.
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Self::from_pretrained_with_decoding(path, config, DecodingStrategy::Greedy)
    }

    /// Load Parakeet CTC with an explicit decoding strategy (greedy default, or beam).
    pub fn from_pretrained_with_decoding<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
        decoding: DecodingStrategy,
    ) -> Result<Self> {
        let path = path.as_ref();

        // Determine if path is a directory or file
        let (model_path, tokenizer_path, model_dir) = if path.is_dir() {
            let model_path = Self::find_model_file(path)?;
            let tokenizer_path = Self::find_tokenizer(path, &model_path)?;
            (model_path, tokenizer_path, path.to_path_buf())
        } else if path.is_file() {
            let model_dir = path
                .parent()
                .ok_or_else(|| Error::Config("Invalid model path".to_string()))?;
            let tokenizer_path = Self::find_tokenizer(model_dir, path)?;
            (path.to_path_buf(), tokenizer_path, model_dir.to_path_buf())
        } else {
            return Err(Error::Config(format!(
                "Path does not exist: {}",
                path.display()
            )));
        };

        let preprocessor_config = PreprocessorConfig::default();
        let exec_config = config.unwrap_or_default();

        let model = ParakeetModel::from_pretrained_with_config(&model_path, exec_config)?;
        let ctc_config = model.ctc_config();
        let decoder =
            ParakeetDecoder::from_pretrained_with_ctc_config(&tokenizer_path, ctc_config)?;
        let feature_cache = FeatureCache::from_config(&preprocessor_config);

        Ok(Self {
            model,
            decoder,
            preprocessor_config,
            feature_cache,
            model_dir,
            decoding,
        })
    }

    pub fn decoding_strategy(&self) -> DecodingStrategy {
        self.decoding
    }

    pub fn set_decoding_strategy(&mut self, strategy: DecodingStrategy) {
        self.decoding = strategy;
    }

    fn find_model_file(dir: &Path) -> Result<PathBuf> {
        // Priority order: model.onnx > model_fp16.onnx > model_int8.onnx > model_q4.onnx
        let candidates = [
            "model.onnx",
            "model_fp16.onnx",
            "model_int8.onnx",
            "model_q4.onnx",
        ];

        let search_dirs = [dir.to_path_buf(), dir.join("onnx")];
        for search in &search_dirs {
            for candidate in &candidates {
                let path = search.join(candidate);
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        for search in &search_dirs {
            if let Ok(entries) = std::fs::read_dir(search) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                        return Ok(path);
                    }
                }
            }
        }

        Err(Error::Config(format!(
            "No model file (*.onnx) found in directory: {}",
            dir.display()
        )))
    }

    fn find_tokenizer(dir: &Path, model_path: &Path) -> Result<PathBuf> {
        let mut candidates = vec![
            dir.join("tokenizer.json"),
            dir.join("onnx").join("tokenizer.json"),
        ];
        if let Some(parent) = model_path.parent() {
            candidates.push(parent.join("tokenizer.json"));
            if let Some(grand) = parent.parent() {
                candidates.push(grand.join("tokenizer.json"));
            }
        }
        for path in candidates {
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "Required file 'tokenizer.json' not found in {} (or next to the ONNX file)",
            dir.display()
        )))
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }
}

impl Transcriber for Parakeet {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        let features = audio::extract_features_with_frontend(
            audio,
            sample_rate,
            channels,
            &self.preprocessor_config,
            &self.feature_cache,
            &self.model.ctc_config().frontend,
        )?;
        let feature_frames = features.shape()[0];
        let mut result = match self.decoding {
            DecodingStrategy::Greedy => {
                let ids = self.model.forward_frame_ids(features)?;
                let ctc = self.model.ctc_config();
                let frame_shift = ctc.logit_shift_samples_for(
                    self.preprocessor_config.hop_length,
                    feature_frames,
                    ids.len(),
                );
                self.decoder.decode_ids_with_timestamps(
                    &ids,
                    frame_shift,
                    self.preprocessor_config.sampling_rate,
                )?
            }
            DecodingStrategy::Beam(config) => {
                let logits = self.model.forward(features)?;
                let text = self.decoder.decode_with_beam_config(&logits, &config)?;
                TranscriptionResult {
                    text,
                    tokens: Vec::new(),
                }
            }
        };

        let mode = mode.unwrap_or(TimestampMode::Tokens);
        if result.tokens.is_empty() {
            // Beam path: timestamps deferred (Phase 5b); keep decoded text as-is.
            return Ok(result);
        }

        // Process timestamps to requested output mode (greedy path only).
        result.tokens = process_timestamps(&result.tokens, mode);

        // Rebuild full text from processed tokens to ensure consistency
        result.text = rebuild_text(&result.tokens, mode);

        Ok(result)
    }
}
