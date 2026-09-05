use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX `metadata_props` keys read by [`CtcConfig::from_metadata`].
/// Must stay in lockstep with `scripts/patch_ctc_onnx_metadata.py`.
pub const CTC_META_BLANK_ID: &str = "ctc_blank_id";
pub const CTC_META_TOKEN_ID_OFFSET: &str = "ctc_token_id_offset";
pub const CTC_META_FRONTEND_DELTAS: &str = "ctc_frontend_deltas";
pub const CTC_META_FRONTEND_DELTA_WIN_LENGTH: &str = "ctc_frontend_delta_win_length";
pub const CTC_META_FRONTEND_STACK_FACTOR: &str = "ctc_frontend_stack_factor";
pub const CTC_META_FRONTEND_NORMALIZATION: &str = "ctc_frontend_normalization";
/// In-graph temporal downsample after the frontend (Granite TurboCTC = 4).
pub const CTC_META_ENCODER_SUBSAMPLE: &str = "ctc_encoder_subsample";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    pub feature_extractor_type: String,
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub padding_side: String,
    pub padding_value: f32,
    pub preemphasis: f32,
    pub processor_class: String,
    pub return_attention_mask: bool,
    pub sampling_rate: usize,
    pub win_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub pad_token_id: usize,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: 80,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["ParakeetForCTC".to_string()],
            vocab_size: 1025,
            pad_token_id: 1024,
        }
    }
}

/// Log-mel post-processing after the shared STFT / mel filterbank.
///
/// [`Self::PerFeatureZNorm`] is the Parakeet-CTC-0.6b path (`ln` + per-feature z-score).
/// [`Self::DbFloorScale`] is the Granite TurboCTC path (`log10`, clamp to max-floor_db, `/4+1`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMode {
    PerFeatureZNorm,
    DbFloorScale { floor_db: f32 },
}

impl Default for NormalizationMode {
    fn default() -> Self {
        Self::PerFeatureZNorm
    }
}

/// Optional CTC frontend extras (deltas, frame stacking, alternate log/norm).
///
/// Defaults preserve Parakeet-CTC-0.6b: no deltas, no stacking, per-feature z-norm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CtcFrontendConfig {
    pub deltas: bool,
    pub delta_win_length: usize,
    pub stack_factor: usize,
    pub normalization: NormalizationMode,
}

impl Default for CtcFrontendConfig {
    fn default() -> Self {
        Self {
            deltas: false,
            delta_win_length: 3,
            stack_factor: 1,
            normalization: NormalizationMode::PerFeatureZNorm,
        }
    }
}

/// Runtime CTC decode + frontend config, typically loaded from ONNX metadata.
///
/// Missing or unparseable keys fall back to Parakeet-CTC-0.6b defaults
/// (`blank_id = 1024`, no token offset, default frontend).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CtcConfig {
    pub blank_id: u32,
    pub token_id_offset: i32,
    /// Encoder time downsample relative to stacked feature frames. `1` = logits
    /// are 1:1 with frontend frames (Parakeet-CTC-0.6b timestamps). Granite
    /// TurboCTC is `4` (two strided conformer blocks).
    pub encoder_subsample: usize,
    pub frontend: CtcFrontendConfig,
}

impl Default for CtcConfig {
    fn default() -> Self {
        Self {
            blank_id: 1024,
            token_id_offset: 0,
            encoder_subsample: 1,
            frontend: CtcFrontendConfig::default(),
        }
    }
}

impl CtcConfig {
    /// Parse CTC settings from ONNX custom metadata. Unknown / invalid values
    /// are ignored field-by-field so unannotated models keep default behavior.
    pub fn from_metadata(keys: &HashMap<String, String>) -> Self {
        let mut cfg = Self::default();

        if let Some(v) = parse_u32(keys.get(CTC_META_BLANK_ID)) {
            cfg.blank_id = v;
        }
        if let Some(v) = parse_i32(keys.get(CTC_META_TOKEN_ID_OFFSET)) {
            cfg.token_id_offset = v;
        }
        if let Some(v) = parse_bool(keys.get(CTC_META_FRONTEND_DELTAS)) {
            cfg.frontend.deltas = v;
        }
        if let Some(v) = parse_usize(keys.get(CTC_META_FRONTEND_DELTA_WIN_LENGTH)) {
            cfg.frontend.delta_win_length = v.max(1);
        }
        if let Some(v) = parse_usize(keys.get(CTC_META_FRONTEND_STACK_FACTOR)) {
            cfg.frontend.stack_factor = v.max(1);
        }
        if let Some(v) = parse_usize(keys.get(CTC_META_ENCODER_SUBSAMPLE)) {
            cfg.encoder_subsample = v.max(1);
        }
        if let Some(v) = keys.get(CTC_META_FRONTEND_NORMALIZATION) {
            if let Some(norm) = parse_normalization(v) {
                cfg.frontend.normalization = norm;
            }
        }

        cfg
    }

    /// Audio samples per CTC logit frame: `hop * stack_factor * encoder_subsample`.
    ///
    /// Granite TurboCTC: 160 * 2 * 4 = 1280 samples (80 ms @ 16 kHz).
    pub fn logit_shift_samples(&self, hop_length: usize) -> usize {
        hop_length.max(1)
            * self.frontend.stack_factor.max(1)
            * self.encoder_subsample.max(1)
    }

    /// Samples per logit frame, using in-graph subsample from metadata when
    /// set, otherwise `feature_frames / logit_frames` for stacked frontends.
    pub fn logit_shift_samples_for(
        &self,
        hop_length: usize,
        feature_frames: usize,
        logit_frames: usize,
    ) -> usize {
        let subsample = if self.encoder_subsample > 1 {
            self.encoder_subsample
        } else if self.frontend.stack_factor > 1 && logit_frames > 0 {
            (feature_frames / logit_frames).max(1)
        } else {
            self.encoder_subsample.max(1)
        };
        hop_length.max(1) * self.frontend.stack_factor.max(1) * subsample
    }
}

fn parse_u32(raw: Option<&String>) -> Option<u32> {
    raw.and_then(|s| s.trim().parse().ok())
}

fn parse_i32(raw: Option<&String>) -> Option<i32> {
    raw.and_then(|s| s.trim().parse().ok())
}

fn parse_usize(raw: Option<&String>) -> Option<usize> {
    raw.and_then(|s| s.trim().parse().ok())
}

fn parse_bool(raw: Option<&String>) -> Option<bool> {
    let s = raw?.trim().to_ascii_lowercase();
    match s.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_normalization(raw: &str) -> Option<NormalizationMode> {
    let s = raw.trim().to_ascii_lowercase();
    if s == "per_feature_z_norm" {
        return Some(NormalizationMode::PerFeatureZNorm);
    }
    if let Some(rest) = s.strip_prefix("db_floor_scale") {
        let floor_db = rest
            .strip_prefix(':')
            .or_else(|| rest.strip_prefix('='))
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .and_then(|v| v.parse().ok())
            .unwrap_or(8.0);
        return Some(NormalizationMode::DbFloorScale { floor_db });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_parakeet_ctc_0_6b() {
        let cfg = CtcConfig::default();
        assert_eq!(cfg.blank_id, 1024);
        assert_eq!(cfg.token_id_offset, 0);
        assert!(!cfg.frontend.deltas);
        assert_eq!(cfg.frontend.delta_win_length, 3);
        assert_eq!(cfg.frontend.stack_factor, 1);
        assert_eq!(cfg.encoder_subsample, 1);
        assert_eq!(cfg.logit_shift_samples(160), 160);
        assert_eq!(
            cfg.frontend.normalization,
            NormalizationMode::PerFeatureZNorm
        );
    }

    #[test]
    fn from_metadata_empty_is_default() {
        assert_eq!(CtcConfig::from_metadata(&HashMap::new()), CtcConfig::default());
    }

    #[test]
    fn from_metadata_granite_turbo_ctc() {
        let mut keys = HashMap::new();
        keys.insert(CTC_META_BLANK_ID.to_string(), "0".into());
        keys.insert(CTC_META_TOKEN_ID_OFFSET.to_string(), "0".into());
        keys.insert(CTC_META_FRONTEND_DELTAS.to_string(), "true".into());
        keys.insert(CTC_META_FRONTEND_DELTA_WIN_LENGTH.to_string(), "3".into());
        keys.insert(CTC_META_FRONTEND_STACK_FACTOR.to_string(), "2".into());
        keys.insert(CTC_META_ENCODER_SUBSAMPLE.to_string(), "4".into());
        keys.insert(
            CTC_META_FRONTEND_NORMALIZATION.to_string(),
            "db_floor_scale:8.0".into(),
        );

        let cfg = CtcConfig::from_metadata(&keys);
        assert_eq!(cfg.blank_id, 0);
        assert_eq!(cfg.token_id_offset, 0);
        assert!(cfg.frontend.deltas);
        assert_eq!(cfg.frontend.stack_factor, 2);
        assert_eq!(cfg.encoder_subsample, 4);
        assert_eq!(cfg.logit_shift_samples(160), 1280);
        assert_eq!(
            cfg.logit_shift_samples_for(160, 200, 50),
            1280
        );
        assert_eq!(
            cfg.frontend.normalization,
            NormalizationMode::DbFloorScale { floor_db: 8.0 }
        );
    }

    #[test]
    fn from_metadata_ignores_invalid_fields() {
        let mut keys = HashMap::new();
        keys.insert(CTC_META_BLANK_ID.to_string(), "not-a-number".into());
        keys.insert(CTC_META_FRONTEND_NORMALIZATION.to_string(), "unknown".into());
        keys.insert(CTC_META_FRONTEND_DELTAS.to_string(), "maybe".into());
        assert_eq!(CtcConfig::from_metadata(&keys), CtcConfig::default());
    }

    #[test]
    fn stacked_frontend_infers_encoder_subsample_from_shapes() {
        let mut keys = HashMap::new();
        keys.insert(CTC_META_FRONTEND_STACK_FACTOR.to_string(), "2".into());
        let cfg = CtcConfig::from_metadata(&keys);
        assert_eq!(cfg.encoder_subsample, 1);
        assert_eq!(cfg.logit_shift_samples_for(160, 200, 50), 1280);
    }
}
