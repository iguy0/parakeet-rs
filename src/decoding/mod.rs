//! Decoding strategies for parakeet-rs ASR models.
//!
//! Greedy decoding remains the default. Opt in to beam search via [`DecodingStrategy`].

mod rnnt_beam;
mod tdt_beam;
mod ctc_beam;

pub use rnnt_beam::{decode_rnnt_beam, RnntBeamOutput, RnntHypothesis};
pub use tdt_beam::{decode_tdt_beam, TdtBeamOutput, TdtHypothesis};
pub use ctc_beam::decode_ctc_beam;

/// Beam search hyperparameters (defaults tuned from parakeet-mlx).
///
/// Fields are shared across model families, but not every field is honored by
/// every decoder. The table below documents current applicability:
///
/// | Field                 | RNNT (Unified) | TDT          | CTC          |
/// |-----------------------|----------------|--------------|--------------|
/// | `beam_size`           | yes            | yes          | yes          |
/// | `length_penalty`      | yes            | yes          | no (unused)  |
/// | `patience`            | no (unused)    | yes          | no (unused)  |
/// | `max_symbols_per_step`| yes            | yes          | no (unused)  |
/// | `duration_reward`     | no (TDT-only)  | yes          | no (TDT-only)|
///
/// Notes:
/// - `patience` controls the finished-hypothesis pool size only in the TDT
///   decoder (`max_candidates = round(beam_size * patience)`). The RNNT and CTC
///   paths prune strictly to `beam_size` and ignore `patience`. Tuning
///   `--patience` therefore has no effect on Unified/CTC today.
/// - `length_penalty` length-normalizes scores in RNNT/TDT selection; the CTC
///   prefix beam ranks by total prefix log-probability and ignores it.
/// - `duration_reward` only weights TDT duration vs token log-probs; it is read
///   by the TDT decoder alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamConfig {
    /// Maximum hypotheses retained after pruning each step. Honored by all models.
    pub beam_size: usize,
    /// Length normalization exponent for hypothesis selection. RNNT/TDT only; CTC ignores.
    pub length_penalty: f32,
    /// Finished-pool expansion factor. TDT only (`beam_size * patience`); RNNT/CTC ignore.
    pub patience: f32,
    /// Cap on symbol emissions per encoder frame. RNNT/TDT only; CTC ignores.
    pub max_symbols_per_step: usize,
    /// TDT-only: weight for duration vs token log-probs in `[0, 1]`.
    pub duration_reward: f32,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self::default_unified()
    }
}

impl BeamConfig {
    /// Defaults for ParakeetUnified RNNT decoding.
    pub fn default_unified() -> Self {
        Self {
            beam_size: 5,
            length_penalty: 0.013,
            patience: 3.5,
            max_symbols_per_step: 10,
            duration_reward: 0.67,
        }
    }

    /// Defaults for ParakeetTDT decoding (same beam hyperparameters as MLX).
    pub fn default_tdt() -> Self {
        Self::default_unified()
    }

    /// Defaults for Parakeet CTC decoding (`beam_size` only; other fields unused).
    pub fn default_ctc() -> Self {
        Self {
            beam_size: 5,
            length_penalty: 0.0,
            patience: 1.0,
            max_symbols_per_step: 10,
            duration_reward: 0.67,
        }
    }
}

/// Decoding strategy for ASR models. Greedy is the default everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DecodingStrategy {
    #[default]
    Greedy,
    Beam(BeamConfig),
}

impl DecodingStrategy {
    pub fn is_beam(self) -> bool {
        matches!(self, Self::Beam(_))
    }

    pub fn beam_config(self) -> Option<BeamConfig> {
        match self {
            Self::Greedy => None,
            Self::Beam(config) => Some(config),
        }
    }
}
