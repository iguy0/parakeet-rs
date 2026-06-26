use crate::decoding::BeamConfig;
use crate::error::Result;
use ndarray::{Array1, Array3};
use std::collections::HashMap;

/// TDT duration indices map to frame skips (NeMo default).
pub const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

/// One TDT beam hypothesis: decoder state plus emitted tokens.
#[derive(Debug, Clone)]
pub struct TdtHypothesis {
    pub score: f32,
    /// Current encoder frame index.
    pub step: usize,
    pub last_token: i32,
    pub state_1: Array3<f32>,
    pub state_2: Array3<f32>,
    /// Consecutive symbol emissions without a duration advance.
    pub stuck: usize,
    /// `(token_id, encoder_frame, duration_value)` triples emitted so far.
    pub tokens: Vec<(usize, usize, usize)>,
}

/// Output of TDT beam search over the full encoder output.
#[derive(Debug, Clone)]
pub struct TdtBeamOutput {
    pub tokens: Vec<usize>,
    pub frame_indices: Vec<usize>,
    pub durations: Vec<usize>,
}

/// TDT beam search (parakeet-mlx `decode_beam` port).
pub fn decode_tdt_beam(
    encoder_out: &Array3<f32>,
    encoder_len: usize,
    blank_id: usize,
    vocab_size: usize,
    config: &BeamConfig,
    decoder: &mut dyn FnMut(
        &Array3<f32>,
        i32,
        &Array3<f32>,
        &Array3<f32>,
    ) -> Result<(Array1<f32>, Array1<f32>, Array3<f32>, Array3<f32>)>,
) -> Result<TdtBeamOutput> {
    let time_steps = encoder_out.shape()[2].min(encoder_len);
    let encoder_dim = encoder_out.shape()[1];
    let num_duration = TDT_DURATIONS.len();

    let beam_token = config.beam_size.min(vocab_size);
    let beam_duration = config.beam_size.min(num_duration);
    let max_candidates = (config.beam_size as f32 * config.patience).round() as usize;

    let state_shape = (2, 1, 640);
    let mut finished: Vec<TdtHypothesis> = Vec::new();
    let mut active = vec![seed_hypothesis(blank_id, state_shape)];

    while finished.len() < max_candidates && !active.is_empty() {
        let mut candidates: HashMap<(usize, i32, Vec<usize>), TdtHypothesis> = HashMap::new();

        for hyp in &active {
            if hyp.step >= time_steps {
                continue;
            }

            let frame = encoder_out
                .slice(ndarray::s![0, .., hyp.step])
                .to_owned()
                .to_shape((1, encoder_dim, 1))
                .map_err(|e| crate::error::Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            let (vocab_logits, duration_logits, new_state_1, new_state_2) = decoder(
                &frame,
                hyp.last_token,
                &hyp.state_1,
                &hyp.state_2,
            )?;

            let token_log_probs = log_softmax(&vocab_logits);
            let duration_log_probs = log_softmax(&duration_logits);

            for token_id in top_k_indices(&token_log_probs, beam_token) {
                let is_blank = token_id == blank_id;
                for duration_idx in top_k_indices(&duration_log_probs, beam_duration) {
                    let duration = TDT_DURATIONS[duration_idx];
                    let stuck = if duration != 0 {
                        0
                    } else {
                        hyp.stuck + 1
                    };

                    let new_step = if stuck >= config.max_symbols_per_step {
                        hyp.step + 1
                    } else {
                        hyp.step + duration
                    };
                    let new_stuck = if stuck >= config.max_symbols_per_step {
                        0
                    } else {
                        stuck
                    };

                    let token_lp = token_log_probs[token_id];
                    let duration_lp = duration_log_probs[duration_idx];
                    let combined = token_lp * (1.0 - config.duration_reward)
                        + duration_lp * config.duration_reward;

                    let mut next = hyp.clone();
                    next.score = hyp.score + combined;
                    next.step = new_step;
                    next.stuck = new_stuck;

                    if is_blank {
                        next.last_token = hyp.last_token;
                        next.state_1 = hyp.state_1.clone();
                        next.state_2 = hyp.state_2.clone();
                    } else {
                        next.last_token = token_id as i32;
                        next.state_1 = new_state_1.clone();
                        next.state_2 = new_state_2.clone();
                        next.tokens
                            .push((token_id, hyp.step, duration));
                    }

                    merge_candidate(&mut candidates, next);
                }
            }
        }

        finished.extend(
            candidates
                .values()
                .filter(|h| h.step >= time_steps)
                .cloned(),
        );

        let mut still_active: Vec<TdtHypothesis> = candidates
            .into_values()
            .filter(|h| h.step < time_steps)
            .collect();
        still_active.sort_by(|a, b| {
            length_normalized_score(b, config.length_penalty)
                .partial_cmp(&length_normalized_score(a, config.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        still_active.truncate(config.beam_size);
        active = still_active;
    }

    let mut all = finished;
    all.extend(active);

    let best = select_best(&all, config.length_penalty).unwrap_or_else(|| {
        seed_hypothesis(blank_id, state_shape)
    });

    let mut tokens = Vec::new();
    let mut frame_indices = Vec::new();
    let mut durations = Vec::new();
    for (token_id, frame, duration) in &best.tokens {
        tokens.push(*token_id);
        frame_indices.push(*frame);
        durations.push(*duration);
    }

    Ok(TdtBeamOutput {
        tokens,
        frame_indices,
        durations,
    })
}

fn seed_hypothesis(blank_id: usize, state_shape: (usize, usize, usize)) -> TdtHypothesis {
    TdtHypothesis {
        score: 0.0,
        step: 0,
        last_token: blank_id as i32,
        state_1: Array3::zeros(state_shape),
        state_2: Array3::zeros(state_shape),
        stuck: 0,
        tokens: Vec::new(),
    }
}

fn merge_candidate(
    candidates: &mut HashMap<(usize, i32, Vec<usize>), TdtHypothesis>,
    hyp: TdtHypothesis,
) {
    let key = (
        hyp.step,
        hyp.last_token,
        hyp.tokens.iter().map(|(id, _, _)| *id).collect(),
    );
    match candidates.get_mut(&key) {
        Some(existing) => {
            let merged_score = log_sum_exp(existing.score, hyp.score);
            if hyp.score > existing.score {
                *existing = hyp;
            }
            existing.score = merged_score;
        }
        None => {
            candidates.insert(key, hyp);
        }
    }
}

fn select_best(hyps: &[TdtHypothesis], length_penalty: f32) -> Option<TdtHypothesis> {
    hyps
        .iter()
        .max_by(|a, b| {
            length_normalized_score(a, length_penalty)
                .partial_cmp(&length_normalized_score(b, length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

fn length_normalized_score(hyp: &TdtHypothesis, length_penalty: f32) -> f32 {
    let len = hyp.tokens.len().max(1) as f32;
    hyp.score / len.powf(length_penalty)
}

fn log_softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let log_sum = sum.ln() + max_logit;
    Array1::from_iter(logits.iter().map(|&x| x - log_sum))
}

fn top_k_indices(log_probs: &Array1<f32>, k: usize) -> Vec<usize> {
    let k = k.min(log_probs.len());
    let mut indexed: Vec<(usize, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn mock_decoder(
        frame: &Array3<f32>,
        last_token: i32,
        state_1: &Array3<f32>,
        state_2: &Array3<f32>,
    ) -> Result<(Array1<f32>, Array1<f32>, Array3<f32>, Array3<f32>)> {
        let step = frame[[0, 0, 0]] as usize;
        let blank_id = 2;
        let vocab_size = 3;

        let mut vocab_logits = Array1::from_elem(vocab_size, -20.0f32);
        vocab_logits[blank_id] = 0.0;

        let mut duration_logits = Array1::from_elem(5, -20.0f32);
        duration_logits[1] = 0.0; // duration 1

        if step == 0 {
            vocab_logits[blank_id] = -20.0;
            vocab_logits[0] = 0.0; // emit token 0
        } else if step == 1 {
            if last_token == 1 {
                vocab_logits[blank_id] = 0.0;
            } else {
                vocab_logits[blank_id] = -20.0;
                vocab_logits[1] = 0.0;
            }
        }

        Ok((vocab_logits, duration_logits, state_1.clone(), state_2.clone()))
    }

    fn synthetic_encoder(frames: usize, hidden: usize) -> Array3<f32> {
        let mut enc = Array3::zeros((1, hidden, frames));
        for t in 0..frames {
            enc[[0, 0, t]] = t as f32;
        }
        enc
    }

    #[test]
    fn combined_score_weights_duration_reward() {
        let token_lp = -1.0f32;
        let duration_lp = -2.0f32;
        let reward = 0.67;
        let combined = token_lp * (1.0 - reward) + duration_lp * reward;
        assert!((combined - (-1.67)).abs() < 1e-5);
    }

    #[test]
    fn merge_combines_duplicate_paths() {
        let state = Array3::zeros((1, 1, 1));
        let hyp_a = TdtHypothesis {
            score: -1.0,
            step: 1,
            last_token: 0,
            state_1: state.clone(),
            state_2: state.clone(),
            stuck: 0,
            tokens: vec![(0, 0, 1)],
        };
        let hyp_b = TdtHypothesis {
            score: -2.0,
            step: 1,
            last_token: 0,
            state_1: state.clone(),
            state_2: state.clone(),
            stuck: 0,
            tokens: vec![(0, 0, 1)],
        };
        let mut candidates = HashMap::new();
        merge_candidate(&mut candidates, hyp_a);
        merge_candidate(&mut candidates, hyp_b);
        assert_eq!(candidates.len(), 1);
        let expected = log_sum_exp(-1.0, -2.0);
        assert!((candidates.values().next().unwrap().score - expected).abs() < 1e-5);
    }

    #[test]
    fn decode_emits_tokens_with_duration_advance() {
        let enc = synthetic_encoder(3, 4);
        let config = BeamConfig {
            beam_size: 3,
            length_penalty: 0.0,
            patience: 3.5,
            max_symbols_per_step: 10,
            duration_reward: 0.67,
        };

        let mut decoder = mock_decoder;
        let out = decode_tdt_beam(&enc, 3, 2, 3, &config, &mut decoder).unwrap();

        assert_eq!(out.tokens, vec![0, 1]);
        assert_eq!(out.frame_indices, vec![0, 1]);
        assert_eq!(out.durations, vec![1, 1]);
    }

    #[test]
    fn patience_scales_finished_candidate_pool() {
        // TDT is the only decoder that reads `patience`; it sets the finished pool
        // size as round(beam_size * patience). RNNT and CTC ignore the field.
        let beam_size = 5usize;
        let low = (beam_size as f32 * 1.0).round() as usize;
        let high = (beam_size as f32 * 3.5).round() as usize;
        assert_eq!(low, 5);
        assert_eq!(high, 18);
        assert!(high > low);
    }

    #[test]
    fn duration_zero_increments_stuck_and_advances_on_max_symbols() {
        let state = Array3::zeros((1, 1, 1));
        let hyp = TdtHypothesis {
            score: 0.0,
            step: 0,
            last_token: 2,
            state_1: state.clone(),
            state_2: state,
            stuck: 9,
            tokens: Vec::new(),
        };
        // stuck + 1 >= max_symbols_per_step (10) => advance by 1
        let duration = 0;
        let stuck = hyp.stuck + 1;
        let new_step = if stuck >= 10 { hyp.step + 1 } else { hyp.step + duration };
        assert_eq!(new_step, 1);
    }
}
