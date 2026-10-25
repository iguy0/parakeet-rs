//! Multi-talker streaming ASR pipeline.
//!
//! Combines Sortformer speaker diarisation with the multitalker encoder
//! (speaker kernel injection) to produce per-speaker transcriptions from
//! mixed audio. Each active speaker gets an independent encoder cache and
//! decoder state.
//!
//! Architecture:
//! ```text
//! Audio -> [Mel] -> [Sortformer raw preds] -> per-speaker masks
//!                   -> [ASR Encoder(mel, cache_k, spk_k, bg_k)] -> [RNNT Decode] -> text_k
//! ```

use crate::decoder::{TimedToken, TranscriptionResult};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_multitalker::{MultitalkerEncoderCache, MultitalkerModel};
use crate::nemotron::SentencePieceVocab;
use crate::sortformer::{DEFAULT_NUM_SPEAKERS, Sortformer};
use crate::timestamps::{self, TimestampMode};
use crate::transcriber::Transcriber;
use ndarray::{Array2, Array3, ArrayView2, Axis, s};
use std::path::Path;

// Reuse the same audio constants as Nemotron (same encoder architecture)
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

// Encoder arch (same as Nemotron 0.6B)
const NUM_ENCODER_LAYERS: usize = 24;
const HIDDEN_DIM: usize = 1024;
const LEFT_CONTEXT: usize = 70;
const CONV_CONTEXT: usize = 8;

// Decoder fallbacks. The loaded tokenizer and joint-network output replace these
// when they agree (RNNT blank is the class after the SentencePiece pieces).
const FALLBACK_VOCAB_SIZE: usize = 1024;
const FALLBACK_BLANK_ID: usize = 1024;

/// Mel frames (10 ms) per Sortformer frame (80 ms).
const MEL_FRAMES_PER_DIAR_FRAME: usize = 8;
const DECODER_LSTM_DIM: usize = 640;
const MAX_SYMBOLS_PER_STEP: usize = 10;

// Pre-encode cache frames (fixed, independent of latency mode)
const PRE_ENCODE_CACHE: usize = 9;

// Each encoded frame spans 8 mel frames at 10ms hop = 80ms
const SECONDS_PER_ENCODED_FRAME: f32 = 0.08;

/// Activity threshold: a speaker is considered active if any frame in the
/// chunk exceeds this probability.
const SPEAKER_ACTIVITY_THRESHOLD: f32 = 0.3;

/// Word-level timestamp for a single word in a speaker's transcript.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_secs: f32,
    pub end_secs: f32,
}

/// Per-speaker state for the multi-instance architecture.
struct SpeakerInstance {
    encoder_cache: MultitalkerEncoderCache,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    /// Each entry is (token_id, absolute_encoder_frame).
    accumulated_tokens: Vec<(usize, usize)>,
    speaker_id: usize,
}

impl SpeakerInstance {
    fn new(speaker_id: usize, blank_id: usize) -> Self {
        Self {
            encoder_cache: MultitalkerEncoderCache::new(
                NUM_ENCODER_LAYERS,
                LEFT_CONTEXT,
                HIDDEN_DIM,
                CONV_CONTEXT,
            ),
            state_1: Array3::zeros((2, 1, DECODER_LSTM_DIM)),
            state_2: Array3::zeros((2, 1, DECODER_LSTM_DIM)),
            last_token: blank_id as i32,
            accumulated_tokens: Vec::new(),
            speaker_id,
        }
    }
}

/// Per-speaker transcription output.
#[derive(Debug, Clone)]
pub struct SpeakerTranscript {
    pub speaker_id: usize,
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

/// Streaming latency mode controlling the encoder chunk size.
///
/// The multitalker encoder was trained with multi-latency masking, so it can
/// operate at different chunk sizes at inference time. Smaller chunks give
/// lower latency but reduce accuracy because fewer future frames are available
/// to the attention layers.
///
/// Each mode corresponds to an `att_context_size` configuration in the model:
/// the second value is the number of future encoded frames the first layer
/// group can attend to.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LatencyMode {
    /// `[70, 13]` -- 14 encoded frames, 112 mel frames, 1.12s latency.
    /// Highest accuracy. This is the default.
    #[default]
    Normal,
    /// `[70, 6]` -- 7 encoded frames, 56 mel frames, 0.56s latency.
    Low,
    /// `[70, 1]` -- 2 encoded frames, 16 mel frames, 0.16s latency.
    VeryLow,
    /// `[70, 0]` -- 1 encoded frame, 8 mel frames, 0.08s latency.
    /// Lowest accuracy.
    Ultra,
}

impl LatencyMode {
    /// Number of mel spectrogram frames per encoder chunk.
    pub const fn chunk_mel_frames(self) -> usize {
        match self {
            Self::Normal => 112, // 14 * 8
            Self::Low => 56,     //  7 * 8
            Self::VeryLow => 16, //  2 * 8
            Self::Ultra => 8,    //  1 * 8
        }
    }

    /// Number of encoded frames per chunk (after 8x subsampling).
    pub const fn encoded_frames(self) -> usize {
        match self {
            Self::Normal => 14,
            Self::Low => 7,
            Self::VeryLow => 2,
            Self::Ultra => 1,
        }
    }

    /// Approximate latency in seconds.
    pub const fn latency_secs(self) -> f32 {
        match self {
            Self::Normal => 1.12,
            Self::Low => 0.56,
            Self::VeryLow => 0.16,
            Self::Ultra => 0.08,
        }
    }
}

/// How each ASR chunk obtains Sortformer speaker probabilities.
///
/// [`PerChunk`](Self::PerChunk) is the historical behavior. [`Buffered`](Self::Buffered)
/// runs Sortformer on its own window and holds ASR chunks until those frames exist.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DiarStrategy {
    /// Call `Sortformer::diarize_chunk_raw` on every ASR chunk.
    ///
    /// The ASR chunk (1.12 s in [`LatencyMode::Normal`]) is shorter than a
    /// Sortformer window (about 10 s on the default 4-speaker export), so each
    /// call pads up to that window. This is the default.
    #[default]
    PerChunk,
    /// Feed audio with `Sortformer::feed_raw` and queue the 80 ms probability
    /// frames. An ASR chunk runs only once queued frames cover its time range,
    /// and [`MultitalkerASR::flush`] (or the silence tail inside
    /// [`MultitalkerASR::transcribe_audio_multitalker`]) finishes the stream
    /// with `flush_raw`.
    ///
    /// Latency grows by one Sortformer window ([`Sortformer::latency`](crate::sortformer::Sortformer::latency)):
    /// `(chunk_len + right_context) * 80 ms`, about 10 s for the default export.
    /// Text for that window arrives together once the window is ready.
    Buffered,
}

/// Runtime configuration for the multitalker pipeline.
///
/// These settings can be changed between calls to `transcribe_chunk()` via
/// the setter methods on [`MultitalkerASR`]. Changing `latency_mode` requires
/// calling [`MultitalkerASR::reset()`] first (the setter does this automatically).
#[derive(Debug, Clone)]
pub struct MultitalkerConfig {
    /// Maximum number of concurrent speakers to track.
    /// Capped automatically to the loaded Sortformer model's `num_speakers`.
    pub max_speakers: usize,

    /// Minimum speaker activity probability to consider a speaker active
    /// in a given chunk. Higher values require stronger evidence of speech
    /// before creating a speaker instance. Range: 0.0..=1.0.
    pub activity_threshold: f32,

    /// Streaming latency mode. Controls the encoder chunk size and
    /// therefore the latency-accuracy tradeoff.
    pub latency_mode: LatencyMode,

    /// Sortformer stride relative to ASR chunks.
    /// [`DiarStrategy::PerChunk`] keeps one Sortformer call per ASR chunk.
    pub diar_strategy: DiarStrategy,
}

impl Default for MultitalkerConfig {
    fn default() -> Self {
        Self {
            max_speakers: DEFAULT_NUM_SPEAKERS,
            activity_threshold: SPEAKER_ACTIVITY_THRESHOLD,
            latency_mode: LatencyMode::default(),
            diar_strategy: DiarStrategy::default(),
        }
    }
}

impl MultitalkerConfig {
    /// The mel-frame chunk size for the current latency mode.
    pub fn chunk_size(&self) -> usize {
        self.latency_mode.chunk_mel_frames()
    }
}

/// Multi-talker streaming ASR combining Sortformer diarisation with
/// speaker-kernel-injected ASR encoding.
pub struct MultitalkerASR {
    model: MultitalkerModel,
    sortformer: Sortformer,
    vocab: SentencePieceVocab,
    speakers: Vec<SpeakerInstance>,
    config: MultitalkerConfig,
    /// SentencePiece piece count used as the exclusive end of real token ids.
    vocab_size: usize,
    /// RNNT blank class. Equal to `vocab_size` for the current multitalker
    /// export (1024 pieces, blank at 1024).
    blank_id: usize,
    mel_basis: Array2<f32>,
    audio_buffer: Vec<f32>,
    audio_processed: usize,
    chunk_idx: usize,
    /// Queued Sortformer frames for [`DiarStrategy::Buffered`], shape
    /// `[frames, speakers]`, one row per 80 ms from the start of the stream.
    diar_queue: Array2<f32>,
}

impl MultitalkerASR {
    /// Load the multitalker ASR pipeline.
    ///
    /// # Arguments
    /// * `asr_model_dir` - Directory containing encoder.onnx, decoder_joint.onnx, tokenizer.model
    /// * `sortformer_model_path` - Path to Sortformer ONNX model
    /// * `exec_config` - ONNX Runtime execution config (optional)
    pub fn from_pretrained<P: AsRef<Path>, Q: AsRef<Path>>(
        asr_model_dir: P,
        sortformer_model_path: Q,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let asr_dir = asr_model_dir.as_ref();
        let exec = exec_config.unwrap_or_default();

        let vocab = SentencePieceVocab::from_file(asr_dir.join("tokenizer.model"))?;

        let model = MultitalkerModel::from_pretrained(asr_dir, exec.clone())?;
        let (vocab_size, blank_id) = token_layout(vocab.size(), model.decoder_output_classes());
        let sortformer = Sortformer::with_config(
            sortformer_model_path,
            Some(exec.clone()),
            crate::sortformer::DiarizationConfig::default(),
        )?;

        let mel_basis = crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);
        let mut config = MultitalkerConfig::default();
        config.max_speakers = sortformer.num_speakers();
        let n_spk = sortformer.num_speakers();

        Ok(Self {
            model,
            sortformer,
            vocab,
            speakers: Vec::new(),
            config,
            vocab_size,
            blank_id,
            mel_basis,
            audio_buffer: Vec::new(),
            audio_processed: 0,
            chunk_idx: 0,
            diar_queue: Array2::zeros((0, n_spk)),
        })
    }

    /// Reset all state for a new utterance.
    pub fn reset(&mut self) {
        self.speakers.clear();
        self.sortformer.reset_state();
        self.audio_buffer.clear();
        self.audio_processed = 0;
        self.chunk_idx = 0;
        self.diar_queue = Array2::zeros((0, self.sortformer.num_speakers()));
    }

    /// Sortformer ONNX runs since the last [`reset`](Self::reset).
    #[doc(hidden)]
    pub fn sortformer_session_runs(&self) -> u64 {
        self.sortformer.session_runs()
    }

    /// Returns the current multitalker configuration.
    pub fn multitalker_config(&self) -> &MultitalkerConfig {
        &self.config
    }

    /// Set the maximum number of speakers to track.
    ///
    /// Can be called between chunks to adjust mid-session. Existing speaker
    /// instances above the new limit will still produce output for any
    /// already-accumulated tokens, but won't receive new audio.
    pub fn set_max_speakers(&mut self, max_speakers: usize) {
        self.config.max_speakers = max_speakers.clamp(1, self.sortformer.num_speakers());
    }

    /// Set the speaker activity threshold (0.0..=1.0).
    ///
    /// A speaker is considered active in a chunk if any frame's probability
    /// exceeds this value. Lower values are more sensitive (detect quieter
    /// speakers sooner), higher values require stronger evidence.
    pub fn set_activity_threshold(&mut self, threshold: f32) {
        self.config.activity_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set the streaming latency mode.
    ///
    /// This changes the encoder chunk size, trading latency for accuracy.
    /// Because encoder caches are tied to the chunk size, this automatically
    /// calls [`reset()`](Self::reset) to clear all state.
    pub fn set_latency_mode(&mut self, mode: LatencyMode) {
        if self.config.latency_mode != mode {
            self.config.latency_mode = mode;
            self.reset();
        }
    }

    /// Set how ASR chunks obtain Sortformer frames.
    ///
    /// Changing the strategy clears streaming state, including Sortformer
    /// caches and any queued diarization frames.
    pub fn set_diar_strategy(&mut self, strategy: DiarStrategy) {
        if self.config.diar_strategy != strategy {
            self.config.diar_strategy = strategy;
            self.reset();
        }
    }

    /// Returns the number of audio samples the caller should provide per
    /// chunk for the current latency mode. This is `chunk_mel_frames * HOP_LENGTH`.
    pub fn chunk_audio_samples(&self) -> usize {
        self.config.chunk_size() * HOP_LENGTH
    }

    /// Get accumulated per-speaker transcripts.
    pub fn get_transcripts(&self) -> Vec<SpeakerTranscript> {
        self.speakers
            .iter()
            .map(|spk| {
                let valid_ids: Vec<usize> = spk
                    .accumulated_tokens
                    .iter()
                    .filter(|&&(t, _)| t < self.vocab_size)
                    .map(|&(t, _)| t)
                    .collect();
                let words = self.tokens_to_words(&spk.accumulated_tokens);
                SpeakerTranscript {
                    speaker_id: spk.speaker_id,
                    text: self.vocab.decode(&valid_ids),
                    words,
                }
            })
            .collect()
    }

    /// Process one audio chunk in streaming mode.
    ///
    /// Returns per-speaker text deltas for this chunk. Speakers are created
    /// automatically when first detected.
    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<Vec<SpeakerTranscript>> {
        self.audio_buffer.extend_from_slice(audio_chunk);

        if self.config.diar_strategy == DiarStrategy::Buffered {
            let frames = self.sortformer.feed_raw(audio_chunk)?;
            self.push_diar_frames(frames);
            return self.emit_buffered_chunks(false);
        }

        self.emit_per_chunk(audio_chunk)
    }

    /// Finish a [`DiarStrategy::Buffered`] stream.
    ///
    /// Runs `Sortformer::flush_raw` and transcribes ASR chunks that were held
    /// until those 80 ms frames existed. [`DiarStrategy::PerChunk`] already
    /// diarized every chunk, so this returns an empty vec.
    pub fn flush(&mut self) -> Result<Vec<SpeakerTranscript>> {
        if self.config.diar_strategy != DiarStrategy::Buffered {
            return Ok(Vec::new());
        }
        let frames = self.sortformer.flush_raw()?;
        self.push_diar_frames(frames);
        self.emit_buffered_chunks(true)
    }

    /// Per-chunk Sortformer path. One ASR chunk per call, matching the
    /// historical `diarize_chunk_raw(audio_chunk)` behavior.
    fn emit_per_chunk(&mut self, audio_chunk: &[f32]) -> Result<Vec<SpeakerTranscript>> {
        let Some(full_mel) = self.ready_mel()? else {
            return Ok(Vec::new());
        };
        let raw_preds = self.sortformer.diarize_chunk_raw(audio_chunk)?;
        self.consume_asr_chunk(&full_mel, &raw_preds.predictions, None)
    }

    /// Transcribe every buffered ASR chunk whose Sortformer frames are ready.
    ///
    /// `end_of_stream` accepts a short final slice (and, if the queue ends
    /// exactly on a chunk boundary, repeats the last frame) so the tail still
    /// reaches the encoder after `flush_raw`.
    fn emit_buffered_chunks(&mut self, end_of_stream: bool) -> Result<Vec<SpeakerTranscript>> {
        let mut results = Vec::new();
        let frames_per = sortformer_frames_per_asr_chunk(self.config.chunk_size());
        loop {
            let Some(full_mel) = self.ready_mel()? else {
                break;
            };
            let queued = if end_of_stream {
                diar_frames_for_asr_chunk_final(&self.diar_queue, self.chunk_idx, frames_per)
            } else {
                diar_frames_for_asr_chunk(&self.diar_queue, self.chunk_idx, frames_per)
                    .map(|view| view.to_owned())
            };
            let Some(preds) = queued else {
                break;
            };
            // The encoder keeps only the tail of a stretched mask, so a stable
            // turn matches PerChunk tokenization. A turn change or a chunk that
            // ends in silence would hide the earlier speaker; those chunks
            // spread one Sortformer frame across each kept mask slot.
            let threshold = self.config.activity_threshold;
            let spread = trailing_rows_inactive(&preds, threshold)
                || speaker_changes_within(&preds, threshold);
            let kept = if spread {
                Some(self.config.latency_mode.encoded_frames())
            } else {
                None
            };
            results.extend(self.consume_asr_chunk(&full_mel, &preds, kept)?);
        }
        Ok(results)
    }

    /// Mel for the current buffer when at least one ASR chunk of new frames is waiting.
    fn ready_mel(&self) -> Result<Option<Array2<f32>>> {
        if self.audio_buffer.len() < WIN_LENGTH {
            return Ok(None);
        }
        let full_mel = self.compute_mel_spectrogram(&self.audio_buffer)?;
        let processed_mel_frames = self.audio_processed / HOP_LENGTH;
        let available_new_frames = full_mel.shape()[1].saturating_sub(processed_mel_frames);
        if available_new_frames < self.config.chunk_size() {
            Ok(None)
        } else {
            Ok(Some(full_mel))
        }
    }

    fn push_diar_frames(&mut self, frames: Array2<f32>) {
        if frames.nrows() == 0 {
            return;
        }
        if self.diar_queue.nrows() == 0 {
            self.diar_queue = frames;
            return;
        }
        if frames.ncols() != self.diar_queue.ncols() {
            return;
        }
        self.diar_queue =
            ndarray::concatenate(Axis(0), &[self.diar_queue.view(), frames.view()]).unwrap();
    }

    /// Encode and decode one ASR chunk using `diar_preds` (rows are 80 ms frames).
    ///
    /// `kept_tail` is set for [`DiarStrategy::Buffered`]. The encoder graph slices
    /// `spk_targets` to its last `encoded_frames` entries (the pre-encode length),
    /// so those slots receive one Sortformer frame each. [`DiarStrategy::PerChunk`]
    /// passes `None` and keeps the historical stretch across the whole vector.
    fn consume_asr_chunk(
        &mut self,
        full_mel: &Array2<f32>,
        diar_preds: &Array2<f32>,
        kept_tail: Option<usize>,
    ) -> Result<Vec<SpeakerTranscript>> {
        let chunk_size = self.config.chunk_size();
        let processed_mel_frames = self.audio_processed / HOP_LENGTH;
        let active_speakers = active_speaker_ids(
            diar_preds,
            self.config.max_speakers,
            self.config.activity_threshold,
        );

        for &spk_id in &active_speakers {
            if !self.speakers.iter().any(|s| s.speaker_id == spk_id) {
                self.speakers
                    .push(SpeakerInstance::new(spk_id, self.blank_id));
            }
        }

        let expected_size = PRE_ENCODE_CACHE + chunk_size;
        let is_first_chunk = self.chunk_idx == 0;
        let main_start = processed_mel_frames;

        let mel_chunk =
            self.build_mel_chunk(full_mel, main_start, is_first_chunk, expected_size)?;
        let chunk_length = expected_size;

        let chunk_frame_offset = self.chunk_idx * self.config.latency_mode.encoded_frames();
        let mut results = Vec::new();

        for &spk_id in &active_speakers {
            let (spk_targets, bg_spk_targets) = match kept_tail {
                Some(kept) => {
                    Self::derive_speaker_targets_kept(diar_preds, spk_id, chunk_length, kept)?
                }
                None => Self::derive_speaker_targets(diar_preds, spk_id, chunk_length)?,
            };

            let spk_idx = self
                .speakers
                .iter()
                .position(|s| s.speaker_id == spk_id)
                .unwrap();

            let (encoded, enc_len, new_cache) = self.model.run_encoder(
                &mel_chunk,
                chunk_length as i64,
                &self.speakers[spk_idx].encoder_cache,
                &spk_targets,
                &bg_spk_targets,
            )?;
            self.speakers[spk_idx].encoder_cache = new_cache;

            let tokens = self.decode_chunk_for_speaker(
                spk_idx,
                &encoded,
                enc_len as usize,
                chunk_frame_offset,
            )?;
            self.speakers[spk_idx].accumulated_tokens.extend(&tokens);

            let mut text = String::new();
            for &(t, _) in &tokens {
                if t < self.vocab_size {
                    text.push_str(&self.vocab.decode_single(t));
                }
            }

            if !text.is_empty() {
                let words = self.tokens_to_words(&tokens);
                results.push(SpeakerTranscript {
                    speaker_id: spk_id,
                    text,
                    words,
                });
            }
        }

        self.audio_processed += chunk_size * HOP_LENGTH;
        self.chunk_idx += 1;

        let keep_samples = (PRE_ENCODE_CACHE + chunk_size) * HOP_LENGTH + WIN_LENGTH;
        if self.audio_buffer.len() > keep_samples * 2 {
            let remove = self.audio_buffer.len() - keep_samples;
            let actual_remove = remove.min(self.audio_processed);
            self.audio_buffer.drain(0..actual_remove);
            self.audio_processed -= actual_remove;
        }

        Ok(results)
    }

    /// Non-streaming transcription of an audio file.
    pub fn transcribe_file_multitalker<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
    ) -> Result<Vec<SpeakerTranscript>> {
        let (audio, spec) = crate::audio::load_audio(audio_path)?;

        if spec.sample_rate != SAMPLE_RATE as u32 {
            return Err(Error::Audio(format!(
                "Expected {} Hz, got {} Hz",
                SAMPLE_RATE, spec.sample_rate
            )));
        }

        let audio = if spec.channels > 1 {
            audio
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            audio
        };

        self.transcribe_audio_multitalker(&audio)
    }

    /// Non-streaming transcription of raw audio samples.
    pub fn transcribe_audio_multitalker(
        &mut self,
        audio: &[f32],
    ) -> Result<Vec<SpeakerTranscript>> {
        self.reset();

        let audio_chunk_size = self.chunk_audio_samples();
        for chunk in audio.chunks(audio_chunk_size) {
            let chunk_vec = if chunk.len() < audio_chunk_size {
                let mut p = chunk.to_vec();
                p.resize(audio_chunk_size, 0.0);
                p
            } else {
                chunk.to_vec()
            };
            self.transcribe_chunk(&chunk_vec)?;
        }

        // Flush with silence, then (Buffered only) the Sortformer tail.
        let flush_chunk = vec![0.0f32; audio_chunk_size];
        for _ in 0..3 {
            self.transcribe_chunk(&flush_chunk)?;
        }
        self.flush()?;

        Ok(self.get_transcripts())
    }

    /// Derive per-speaker target masks from raw Sortformer predictions.
    ///
    /// For the target speaker k:
    /// - `spk_targets[t] = raw_preds[t, k]`
    /// - `bg_spk_targets[t] = max(raw_preds[t, j]) for j != k`
    ///
    /// The masks are resized/interpolated to match the encoder's time dimension.
    fn derive_speaker_targets(
        diar_preds: &Array2<f32>,
        speaker_id: usize,
        encoder_time: usize,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let diar_frames = diar_preds.nrows();

        let mut spk_vals = Vec::with_capacity(encoder_time);
        let mut bg_vals = Vec::with_capacity(encoder_time);

        for enc_t in 0..encoder_time {
            // Map encoder time to diarisation time (nearest-neighbour)
            let diar_t = if diar_frames > 0 && encoder_time > 0 {
                (enc_t * diar_frames / encoder_time).min(diar_frames - 1)
            } else {
                0
            };

            if diar_t < diar_frames && speaker_id < diar_preds.ncols() {
                let spk_val = diar_preds[[diar_t, speaker_id]];
                let bg_val = (0..diar_preds.ncols())
                    .filter(|&j| j != speaker_id)
                    .map(|j| diar_preds[[diar_t, j]])
                    .fold(0.0f32, f32::max);
                spk_vals.push(spk_val);
                bg_vals.push(bg_val);
            } else {
                // No diarisation data: assume single speaker
                spk_vals.push(1.0);
                bg_vals.push(0.0);
            }
        }

        let spk_targets = Array2::from_shape_vec((1, encoder_time), spk_vals)
            .map_err(|e| Error::Model(format!("spk_targets shape mismatch: {e}")))?;
        let bg_spk_targets = Array2::from_shape_vec((1, encoder_time), bg_vals)
            .map_err(|e| Error::Model(format!("bg_spk_targets shape mismatch: {e}")))?;

        Ok((spk_targets, bg_spk_targets))
    }

    /// Place one Sortformer frame in each mask slot the encoder keeps.
    ///
    /// The multitalker graph slices `spk_targets` / `bg_spk_targets` with a
    /// negative start on the time axis, so only the last `kept` values are
    /// multiplied into the pre-encoded sequence (`kept` equals
    /// [`LatencyMode::encoded_frames`]). Values before that tail are unused.
    fn derive_speaker_targets_kept(
        diar_preds: &Array2<f32>,
        speaker_id: usize,
        encoder_time: usize,
        kept: usize,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let kept = kept.max(1).min(encoder_time.max(1));
        let tail_start = encoder_time.saturating_sub(kept);
        let diar_frames = diar_preds.nrows();
        // The graph reads one mask value per pre-encoded frame. Sample the
        // chunk's Sortformer rows uniformly into those slots (not only the
        // last 140 ms, which drops words when a chunk ends in silence).
        let mut spk_vals = Vec::with_capacity(encoder_time);
        let mut bg_vals = Vec::with_capacity(encoder_time);

        for enc_t in 0..encoder_time {
            if enc_t < tail_start {
                spk_vals.push(0.0);
                bg_vals.push(0.0);
                continue;
            }
            let slot = enc_t - tail_start;
            let diar_t = if diar_frames == 0 {
                0
            } else {
                (slot * diar_frames / kept).min(diar_frames - 1)
            };
            if diar_t < diar_frames && speaker_id < diar_preds.ncols() {
                let spk_val = diar_preds[[diar_t, speaker_id]];
                let bg_val = (0..diar_preds.ncols())
                    .filter(|&j| j != speaker_id)
                    .map(|j| diar_preds[[diar_t, j]])
                    .fold(0.0f32, f32::max);
                spk_vals.push(spk_val);
                bg_vals.push(bg_val);
            } else {
                spk_vals.push(1.0);
                bg_vals.push(0.0);
            }
        }

        let spk_targets = Array2::from_shape_vec((1, encoder_time), spk_vals)
            .map_err(|e| Error::Model(format!("spk_targets shape mismatch: {e}")))?;
        let bg_spk_targets = Array2::from_shape_vec((1, encoder_time), bg_vals)
            .map_err(|e| Error::Model(format!("bg_spk_targets shape mismatch: {e}")))?;
        Ok((spk_targets, bg_spk_targets))
    }

    fn build_mel_chunk(
        &self,
        full_mel: &Array2<f32>,
        main_start: usize,
        is_first_chunk: bool,
        expected_size: usize,
    ) -> Result<Array3<f32>> {
        let total_mel_frames = full_mel.shape()[1];
        let chunk_size = self.config.chunk_size();
        let mut chunk_data = vec![0.0f32; N_MELS * expected_size];

        if is_first_chunk {
            for f in 0..chunk_size.min(total_mel_frames) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = full_mel[[m, f]];
                }
            }
        } else {
            let cache_start = main_start.saturating_sub(PRE_ENCODE_CACHE);
            let cache_frames = main_start - cache_start;
            let cache_offset = PRE_ENCODE_CACHE - cache_frames;

            for f in 0..cache_frames {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + cache_offset + f] =
                        full_mel[[m, cache_start + f]];
                }
            }

            for f in 0..chunk_size.min(total_mel_frames - main_start) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] =
                        full_mel[[m, main_start + f]];
                }
            }
        }

        Array3::from_shape_vec((1, N_MELS, expected_size), chunk_data)
            .map_err(|e| Error::Model(format!("Failed to create mel chunk: {e}")))
    }

    fn decode_chunk_for_speaker(
        &mut self,
        spk_idx: usize,
        encoder_out: &Array3<f32>,
        enc_frames: usize,
        chunk_frame_offset: usize,
    ) -> Result<Vec<(usize, usize)>> {
        let mut tokens = Vec::new();
        let hidden_dim = encoder_out.shape()[1];

        for t in 0..enc_frames {
            let frame = encoder_out.slice(s![0, .., t]).to_owned();
            let frame = frame
                .to_shape((1, 1, hidden_dim))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            let absolute_frame = chunk_frame_offset + t;

            for _ in 0..MAX_SYMBOLS_PER_STEP {
                let (logits, new_state_1, new_state_2) = self.model.run_decoder(
                    &frame,
                    self.speakers[spk_idx].last_token,
                    &self.speakers[spk_idx].state_1,
                    &self.speakers[spk_idx].state_2,
                )?;

                let (max_idx, _) = crate::tensor_utils::argmax_f32_iter(logits.iter().copied());

                if max_idx == self.blank_id {
                    break;
                }

                tokens.push((max_idx, absolute_frame));
                self.speakers[spk_idx].last_token = max_idx as i32;
                self.speakers[spk_idx].state_1 = new_state_1;
                self.speakers[spk_idx].state_2 = new_state_2;
            }
        }

        Ok(tokens)
    }

    /// Convert (token_id, absolute_frame) pairs into word-level timestamps.
    fn tokens_to_words(&self, tokens: &[(usize, usize)]) -> Vec<WordTimestamp> {
        let timed: Vec<TimedToken> = tokens
            .iter()
            .filter(|(id, _)| *id < self.vocab_size)
            .map(|&(id, frame)| TimedToken {
                text: self.vocab.decode_single(id),
                start: frame as f32 * SECONDS_PER_ENCODED_FRAME,
                end: (frame + 1) as f32 * SECONDS_PER_ENCODED_FRAME,
            })
            .collect();

        timestamps::group_by_words(&timed)
            .into_iter()
            .map(|t| WordTimestamp {
                word: t.text,
                start_secs: t.start,
                end_secs: t.end,
            })
            .collect()
    }

    /// Compute mel spectrogram using shared audio utilities.
    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Result<Array2<f32>> {
        if audio.is_empty() {
            return Ok(Array2::zeros((N_MELS, 0)));
        }

        let preemph = crate::audio::apply_preemphasis(audio, PREEMPH);
        let spec = crate::audio::stft(&preemph, N_FFT, HOP_LENGTH, WIN_LENGTH)?;
        let mel = self.mel_basis.dot(&spec);

        Ok(mel.mapv(|x| (x.max(0.0) + LOG_ZERO_GUARD).ln()))
    }
}

/// SentencePiece size plus the joint-network class count.
///
/// Success is `decoder_classes == tokenizer_size + 1` (blank is the extra class).
/// When the joint dimension is missing, blank is `tokenizer_size` for a non-empty
/// tokenizer. Any other combination returns the historical `(1024, 1024)` pair.
pub(crate) fn token_layout(
    tokenizer_size: usize,
    decoder_classes: Option<usize>,
) -> (usize, usize) {
    match decoder_classes {
        Some(classes) if tokenizer_size > 0 && classes == tokenizer_size + 1 => {
            (tokenizer_size, classes - 1)
        }
        None if tokenizer_size > 0 => (tokenizer_size, tokenizer_size),
        _ => (FALLBACK_VOCAB_SIZE, FALLBACK_BLANK_ID),
    }
}

/// Sortformer frames (80 ms) covered by one ASR chunk of `chunk_mel_frames` (10 ms hop).
pub(crate) fn sortformer_frames_per_asr_chunk(chunk_mel_frames: usize) -> usize {
    chunk_mel_frames / MEL_FRAMES_PER_DIAR_FRAME
}

/// Rows `[chunk_index * frames_per_chunk, (chunk_index + 1) * frames_per_chunk)`
/// of `queued`, or `None` when the last row is not queued yet.
pub(crate) fn diar_frames_for_asr_chunk<'a>(
    queued: &'a Array2<f32>,
    chunk_index: usize,
    frames_per_chunk: usize,
) -> Option<ArrayView2<'a, f32>> {
    if frames_per_chunk == 0 {
        return None;
    }
    let start = chunk_index.checked_mul(frames_per_chunk)?;
    let end = start.checked_add(frames_per_chunk)?;
    if queued.nrows() < end {
        None
    } else {
        Some(queued.slice(s![start..end, ..]))
    }
}

/// Like [`diar_frames_for_asr_chunk`], but at end of stream: a short tail is
/// returned as-is, and chunks that start at or beyond the queue repeat the
/// last queued frame so the encoder still steps.
fn diar_frames_for_asr_chunk_final(
    queued: &Array2<f32>,
    chunk_index: usize,
    frames_per_chunk: usize,
) -> Option<Array2<f32>> {
    if let Some(view) = diar_frames_for_asr_chunk(queued, chunk_index, frames_per_chunk) {
        return Some(view.to_owned());
    }
    if frames_per_chunk == 0 || queued.ncols() == 0 {
        return None;
    }
    let start = chunk_index.checked_mul(frames_per_chunk)?;
    if queued.nrows() > start {
        return Some(queued.slice(s![start.., ..]).to_owned());
    }
    if queued.nrows() == 0 {
        return None;
    }
    let last = queued.row(queued.nrows() - 1);
    let ncols = queued.ncols();
    Some(Array2::from_shape_fn(
        (frames_per_chunk, ncols),
        |(_, c)| last[c],
    ))
}

/// True when every speaker is at or below `threshold` on the last two rows
/// (or the only row). Those rows are what a stretched mask leaves in the
/// encoder's kept tail.
fn trailing_rows_inactive(frames: &Array2<f32>, threshold: f32) -> bool {
    let n = frames.nrows();
    if n == 0 {
        return true;
    }
    let start = n.saturating_sub(2);
    (start..n).all(|t| (0..frames.ncols()).all(|s| frames[[t, s]] <= threshold))
}

/// True when the loudest speaker above `threshold` is not the same on every
/// active row. A stretched mask would give the whole chunk to whoever is last.
fn speaker_changes_within(frames: &Array2<f32>, threshold: f32) -> bool {
    let mut seen: Option<usize> = None;
    for t in 0..frames.nrows() {
        let mut best: Option<(usize, f32)> = None;
        for s in 0..frames.ncols() {
            let v = frames[[t, s]];
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((s, v));
            }
        }
        if let Some((s, v)) = best
            && v > threshold
        {
            match seen {
                None => seen = Some(s),
                Some(prev) if prev != s => return true,
                Some(_) => {}
            }
        }
    }
    false
}

/// Speaker columns whose max probability in `diar_preds` exceeds `threshold`.
fn active_speaker_ids(diar_preds: &Array2<f32>, max_speakers: usize, threshold: f32) -> Vec<usize> {
    let mut active = Vec::new();
    for spk_id in 0..max_speakers {
        if spk_id >= diar_preds.ncols() {
            break;
        }
        let max_activity = (0..diar_preds.nrows())
            .map(|t| diar_preds[[t, spk_id]])
            .fold(0.0f32, f32::max);
        if max_activity > threshold {
            active.push(spk_id);
        }
    }
    active
}

/// Implement the Transcriber trait for single-speaker fallback.
/// Runs with spk_targets=1.0 and bg_spk_targets=0.0 (no diarisation),
/// treating the multitalker encoder as a standard streaming ASR encoder.
impl Transcriber for MultitalkerASR {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        _mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        if sample_rate != SAMPLE_RATE as u32 {
            return Err(Error::Audio(format!(
                "Expected {} Hz, got {} Hz",
                SAMPLE_RATE, sample_rate
            )));
        }

        let audio = if channels > 1 {
            audio
                .chunks(channels as usize)
                .map(|c| c.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            audio
        };

        // Single-speaker mode: run encoder with full speaker activity
        self.reset();

        let mel = self.compute_mel_spectrogram(&audio)?;
        let total_frames = mel.shape()[1];

        if total_frames == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                tokens: Vec::new(),
            });
        }

        // Create a single speaker instance
        self.speakers.push(SpeakerInstance::new(0, self.blank_id));

        let chunk_size = self.config.chunk_size();
        let mut buffer_idx = 0;
        let mut chunk_idx = 0;

        while buffer_idx < total_frames {
            let expected_size = PRE_ENCODE_CACHE + chunk_size;

            let is_first = chunk_idx == 0;
            let mel_chunk = self.build_mel_chunk(&mel, buffer_idx, is_first, expected_size)?;
            // Use expected_size consistently (matches transcribe_chunk path)
            let chunk_length = expected_size;

            // Single-speaker: full activity, no background
            let spk_targets = Array2::from_elem((1, chunk_length), 1.0f32);
            let bg_spk_targets = Array2::from_elem((1, chunk_length), 0.0f32);

            let (encoded, enc_len, new_cache) = self.model.run_encoder(
                &mel_chunk,
                chunk_length as i64,
                &self.speakers[0].encoder_cache,
                &spk_targets,
                &bg_spk_targets,
            )?;
            self.speakers[0].encoder_cache = new_cache;

            let chunk_frame_offset = chunk_idx * self.config.latency_mode.encoded_frames();
            let tokens =
                self.decode_chunk_for_speaker(0, &encoded, enc_len as usize, chunk_frame_offset)?;
            self.speakers[0].accumulated_tokens.extend(tokens);

            buffer_idx += chunk_size;
            chunk_idx += 1;
        }

        let valid_ids: Vec<usize> = self.speakers[0]
            .accumulated_tokens
            .iter()
            .filter(|&&(t, _)| t < self.vocab_size)
            .map(|&(t, _)| t)
            .collect();

        let text = self.vocab.decode(&valid_ids);

        Ok(TranscriptionResult {
            text,
            tokens: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preds(rows: &[[f32; 3]]) -> Array2<f32> {
        let mut arr = Array2::zeros((rows.len(), 3));
        for (t, row) in rows.iter().enumerate() {
            for (s, value) in row.iter().enumerate() {
                arr[[t, s]] = *value;
            }
        }
        arr
    }

    #[test]
    fn token_layout_matches_current_model_and_falls_back() {
        assert_eq!(token_layout(1024, Some(1025)), (1024, 1024));
        assert_eq!(token_layout(1024, None), (1024, 1024));
        assert_eq!(token_layout(2048, Some(2049)), (2048, 2048));
        assert_eq!(
            token_layout(0, None),
            (FALLBACK_VOCAB_SIZE, FALLBACK_BLANK_ID)
        );
        assert_eq!(
            token_layout(100, Some(1025)),
            (FALLBACK_VOCAB_SIZE, FALLBACK_BLANK_ID)
        );
        assert_eq!(
            token_layout(1024, Some(1024)),
            (FALLBACK_VOCAB_SIZE, FALLBACK_BLANK_ID)
        );
    }

    #[test]
    fn derive_speaker_targets_shape_background_and_nearest() {
        let diar = preds(&[
            [0.1, 0.2, 0.0],
            [0.4, 0.9, 0.3],
            [0.0, 0.2, 0.8],
            [0.5, 0.1, 0.1],
        ]);
        let (spk, bg) = MultitalkerASR::derive_speaker_targets(&diar, 1, 4).unwrap();
        assert_eq!(spk.shape(), &[1, 4]);
        assert_eq!(bg.shape(), &[1, 4]);
        // encoder_time == diar rows, so each step copies that row.
        assert_eq!(spk.row(0).to_vec(), vec![0.2, 0.9, 0.2, 0.1]);
        // Background is the max of the other speakers on the same row.
        assert_eq!(bg.row(0).to_vec(), vec![0.1, 0.4, 0.8, 0.5]);

        let (narrow, _) = MultitalkerASR::derive_speaker_targets(&diar, 1, 2).unwrap();
        // enc 0 -> diar 0; enc 1 -> 1 * 4 / 2 = 2.
        assert_eq!(narrow.row(0).to_vec(), vec![0.2, 0.2]);

        let (kept, kept_bg) = MultitalkerASR::derive_speaker_targets_kept(&diar, 1, 10, 4).unwrap();
        assert_eq!(kept.shape(), &[1, 10]);
        let kept_row = kept.row(0).to_vec();
        let kept_bg_row = kept_bg.row(0).to_vec();
        assert_eq!(&kept_row[..6], &[0.0; 6]);
        assert_eq!(&kept_row[6..], &[0.2, 0.9, 0.2, 0.1]);
        assert_eq!(&kept_bg_row[6..], &[0.1, 0.4, 0.8, 0.5]);

        let (fallback_spk, fallback_bg) =
            MultitalkerASR::derive_speaker_targets(&diar, 9, 3).unwrap();
        assert_eq!(fallback_spk.row(0).to_vec(), vec![1.0, 1.0, 1.0]);
        assert_eq!(fallback_bg.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn trailing_silence_selects_spread_mask() {
        let speech = preds(&[[0.9, 0.0, 0.0], [0.8, 0.0, 0.0]]);
        assert!(!trailing_rows_inactive(&speech, 0.3));
        let ended = preds(&[[0.9, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.05, 0.0]]);
        assert!(trailing_rows_inactive(&ended, 0.3));
        let handover = preds(&[[0.9, 0.1, 0.0], [0.1, 0.8, 0.0]]);
        assert!(speaker_changes_within(&handover, 0.3));
        assert!(!speaker_changes_within(&speech, 0.3));
    }

    #[test]
    fn active_speakers_respect_threshold() {
        let diar = preds(&[[0.2, 0.31, 0.29], [0.0, 0.1, 0.0]]);
        assert_eq!(active_speaker_ids(&diar, 4, 0.3), vec![1]);
        assert!(active_speaker_ids(&diar, 4, 0.95).is_empty());
        assert!(active_speaker_ids(&diar, 0, 0.0).is_empty());
    }

    #[test]
    fn diar_frames_for_chunk_waits_until_covered() {
        let queued = preds(&[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        assert!(diar_frames_for_asr_chunk(&queued, 0, 0).is_none());
        assert!(diar_frames_for_asr_chunk(&queued, 1, 4).is_none());
        let chunk0 = diar_frames_for_asr_chunk(&queued, 0, 2).unwrap();
        assert_eq!(chunk0.shape(), &[2, 3]);
        assert_eq!(chunk0[[0, 0]], 1.0);
        assert_eq!(chunk0[[1, 0]], 1.0);
        let chunk1 = diar_frames_for_asr_chunk(&queued, 1, 2).unwrap();
        assert_eq!(chunk1[[0, 1]], 1.0);
        assert_eq!(chunk1[[1, 1]], 1.0);
        // Normal latency: 112 mel frames / 8 = 14 Sortformer frames.
        assert_eq!(
            sortformer_frames_per_asr_chunk(LatencyMode::Normal.chunk_mel_frames()),
            14
        );
        assert_eq!(
            sortformer_frames_per_asr_chunk(LatencyMode::Ultra.chunk_mel_frames()),
            1
        );
    }
}
