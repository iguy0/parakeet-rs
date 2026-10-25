use crate::config::CtcConfig;
use crate::decoding::{BeamConfig, decode_ctc_beam_view};
use crate::error::{Error, Result};
use crate::tensor_utils::argmax_f32;
use ndarray::{Array2, ArrayView2};
use std::path::Path;

// Token with its timestamp information
// start and end are in seconds
#[derive(Debug, Clone)]
pub struct TimedToken {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub tokens: Vec<TimedToken>,
}

// CTC decoder for parakeet-ctc-0.6b (and metadata-configured CTC variants)
pub struct ParakeetDecoder {
    tokenizer: tokenizers::Tokenizer,
    blank_id: usize,
    token_id_offset: i32,
}

impl ParakeetDecoder {
    pub fn from_pretrained<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Self::from_pretrained_with_ctc_config(tokenizer_path, CtcConfig::default())
    }

    pub fn from_pretrained_with_ctc_config<P: AsRef<Path>>(
        tokenizer_path: P,
        ctc: CtcConfig,
    ) -> Result<Self> {
        let tokenizer_path = tokenizer_path.as_ref();

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self {
            tokenizer,
            blank_id: ctc.blank_id as usize,
            token_id_offset: ctc.token_id_offset,
        })
    }

    pub fn decode(&self, logits: &Array2<f32>) -> Result<String> {
        let mut token_ids = Vec::with_capacity(logits.shape()[0]);
        for row in logits.rows() {
            token_ids.push(argmax_row(&row));
        }
        let collapsed = self.apply_token_offset(&self.ctc_collapse(&token_ids));
        self.tokenizer
            .decode(&collapsed, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {e}")))
    }

    fn ctc_collapse(&self, token_ids: &[u32]) -> Vec<u32> {
        let mut result = Vec::new();
        let mut prev_token: Option<u32> = None;
        let blank = self.blank_id as u32;

        for &token_id in token_ids {
            if token_id == blank {
                prev_token = Some(token_id);
                continue;
            }

            if Some(token_id) != prev_token {
                result.push(token_id);
            }

            prev_token = Some(token_id);
        }

        result
    }

    fn apply_token_offset(&self, token_ids: &[u32]) -> Vec<u32> {
        if self.token_id_offset == 0 {
            return token_ids.to_vec();
        }
        token_ids
            .iter()
            .filter_map(|&id| {
                let shifted = id as i32 - self.token_id_offset;
                (shifted >= 0).then_some(shifted as u32)
            })
            .collect()
    }

    // CTC collapse with frame tracking for timestamps
    fn ctc_collapse_with_frames(&self, token_ids: &[(u32, usize)]) -> Vec<(u32, usize, usize)> {
        let mut result: Vec<(u32, usize, usize)> = Vec::new();
        let mut prev_token: Option<u32> = None;
        let blank = self.blank_id as u32;

        for &(token_id, frame) in token_ids.iter() {
            if token_id == blank {
                prev_token = Some(token_id);
                continue;
            }

            if Some(token_id) != prev_token {
                if let Some(prev) = prev_token
                    && prev != blank
                {
                    // End previous token
                    if let Some(last) = result.last_mut() {
                        last.2 = frame;
                    }
                }
                // Start new token
                result.push((token_id, frame, frame));
            }

            prev_token = Some(token_id);
        }

        // Close last token
        if let Some(last) = result.last_mut() {
            last.2 = token_ids.len();
        }

        result
    }

    // Decode with token-level timestamps.
    // `frame_shift_samples` is samples per logit frame (hop * stack * encoder subsample).
    pub fn decode_with_timestamps(
        &self,
        logits: &Array2<f32>,
        frame_shift_samples: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        let mut frame_ids = Vec::with_capacity(logits.shape()[0]);
        for row in logits.rows() {
            frame_ids.push(argmax_row(&row));
        }
        self.decode_ids_with_timestamps(&frame_ids, frame_shift_samples, sample_rate)
    }

    /// Greedy CTC collapse + timestamps from per-frame argmax ids (one id per logit frame).
    pub fn decode_ids_with_timestamps(
        &self,
        frame_ids: &[u32],
        frame_shift_samples: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        let token_ids_with_frames: Vec<(u32, usize)> = frame_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(t, id)| (id, t))
            .collect();

        let collapsed_with_frames = self.ctc_collapse_with_frames(&token_ids_with_frames);
        let collapsed_with_frames: Vec<(u32, usize, usize)> = collapsed_with_frames
            .into_iter()
            .filter_map(|(id, start, end)| {
                let shifted = id as i32 - self.token_id_offset;
                (shifted >= 0).then_some((shifted as u32, start, end))
            })
            .collect();

        let token_ids: Vec<u32> = collapsed_with_frames.iter().map(|(id, _, _)| *id).collect();
        let full_text = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {e}")))?;

        let shift = frame_shift_samples.max(1);
        let sr = sample_rate.max(1) as f32;
        let mut timed_tokens = Vec::with_capacity(collapsed_with_frames.len());
        for (id, start_frame, end_frame) in &collapsed_with_frames {
            let added_text = self.piece_text(*id);
            if added_text.is_empty() {
                continue;
            }
            timed_tokens.push(TimedToken {
                text: added_text,
                start: (*start_frame * shift) as f32 / sr,
                end: (*end_frame * shift) as f32 / sr,
            });
        }

        Ok(TranscriptionResult {
            text: full_text,
            tokens: timed_tokens,
        })
    }

    /// Map one tokenizer id to surface text (leading `Ġ`/`▁` become spaces).
    fn piece_text(&self, id: u32) -> String {
        let Some(tok) = self.tokenizer.id_to_token(id) else {
            return String::new();
        };
        if (tok.starts_with("<|") && tok.ends_with("|>"))
            || tok == "[PAD]"
            || tok == "<pad>"
            || tok == "<blank>"
        {
            return String::new();
        }
        tok.replace('▁', " ").replace('Ġ', " ")
    }

    /// CTC prefix beam search (no language model). Timestamp alignment is not
    /// computed here; use greedy [`Self::decode_with_timestamps`] when timestamps
    /// are required.
    pub fn decode_with_beam_search(
        &self,
        logits: &Array2<f32>,
        beam_width: usize,
    ) -> Result<String> {
        let config = BeamConfig {
            beam_size: beam_width.max(1),
            ..BeamConfig::default_ctc()
        };
        self.decode_with_beam_config(logits, &config)
    }

    /// CTC prefix beam search with full [`BeamConfig`].
    pub fn decode_with_beam_config(
        &self,
        logits: &Array2<f32>,
        config: &BeamConfig,
    ) -> Result<String> {
        self.decode_with_beam_config_view(logits.view(), config)
    }

    /// CTC prefix beam search over a borrowed `[time, vocab]` logit view.
    pub fn decode_with_beam_config_view(
        &self,
        logits: ArrayView2<f32>,
        config: &BeamConfig,
    ) -> Result<String> {
        let collapsed = decode_ctc_beam_view(logits, self.blank_id, config);
        let collapsed = self.apply_token_offset(&collapsed);
        let text = self
            .tokenizer
            .decode(&collapsed, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {e}")))?;
        Ok(text)
    }

    pub fn pad_token_id(&self) -> usize {
        self.blank_id
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    pub fn token_id_offset(&self) -> i32 {
        self.token_id_offset
    }
}

fn argmax_row(row: &ndarray::ArrayView1<f32>) -> u32 {
    if let Some(s) = row.as_slice() {
        argmax_f32(s).0 as u32
    } else {
        crate::tensor_utils::argmax_f32_iter(row.iter().copied()).0 as u32
    }
}
