use crate::decoding::BeamConfig;
use crate::error::Result;
use ndarray::{Array1, Array3};
use std::collections::HashMap;

/// One RNNT beam hypothesis: decoder state plus emitted tokens.
#[derive(Debug, Clone)]
pub struct RnntHypothesis {
    pub score: f32,
    /// Local encoder frame index within the current decode range `[start, end)`.
    pub frame: usize,
    pub last_token: i32,
    pub state_1: Array3<f32>,
    pub state_2: Array3<f32>,
    pub symbols_on_frame: usize,
    /// `(token_id, absolute_encoder_frame)` pairs emitted so far.
    pub tokens: Vec<(usize, usize)>,
    /// Number of tokens this path already carried into the current decode pass
    /// (i.e. emitted in prior chunks). `tokens[prev_token_count..]` are the tokens
    /// newly emitted during this pass. Set at pass entry and preserved across
    /// clones so per-path incremental emission stays correct regardless of how
    /// many tokens other beam members hold.
    pub prev_token_count: usize,
}

/// Output of one RNNT beam decode pass over an encoder frame range.
#[derive(Debug, Clone)]
pub struct RnntBeamOutput {
    /// Tokens newly emitted during this decode call (from the length-normalized best path).
    pub new_tokens: Vec<(usize, usize)>,
    /// Hypotheses at `end_frame` to carry into the next streaming chunk.
    pub hypotheses: Vec<RnntHypothesis>,
    /// Length-normalized best hypothesis for this decode pass.
    pub best_hypothesis: Option<RnntHypothesis>,
}

/// Model-agnostic RNNT beam search over a decoder callback.
pub fn decode_rnnt_beam(
    encoder_out: &Array3<f32>,
    start_frame: usize,
    end_frame: usize,
    absolute_frame_offset: usize,
    blank_id: usize,
    config: &BeamConfig,
    input_hypotheses: Vec<RnntHypothesis>,
    decoder: &mut dyn FnMut(
        &Array3<f32>,
        i32,
        &Array3<f32>,
        &Array3<f32>,
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)>,
) -> Result<RnntBeamOutput> {
    let end_frame = end_frame.min(encoder_out.shape()[2]);
    if start_frame >= end_frame {
        return Ok(RnntBeamOutput {
            new_tokens: Vec::new(),
            hypotheses: input_hypotheses,
            best_hypothesis: None,
        });
    }

    let hidden_dim = encoder_out.shape()[1];
    let state_shape = input_hypotheses
        .first()
        .map(|h| h.state_1.dim())
        .unwrap_or((2, 1, 640));

    let mut beam = if input_hypotheses.is_empty() {
        vec![seed_hypothesis(blank_id, start_frame, state_shape)]
    } else {
        // Mark each carried hypothesis's existing tokens as already committed so
        // newly emitted tokens are computed per path, not against a global max.
        let mut seeded = input_hypotheses;
        for h in &mut seeded {
            h.prev_token_count = h.tokens.len();
        }
        seeded
    };

    for frame_idx in start_frame..end_frame {
        let frame = encoder_out
            .slice(ndarray::s![0, .., frame_idx])
            .to_owned()
            .to_shape((1, hidden_dim, 1))
            .map_err(|e| crate::error::Error::Model(format!("Failed to reshape encoder frame: {e}")))?
            .to_owned();

        let absolute_frame = absolute_frame_offset + (frame_idx - start_frame);

        let mut active: Vec<RnntHypothesis> = beam
            .into_iter()
            .filter(|h| h.frame == frame_idx)
            .collect();
        active = prune_beam(active, config);

        let mut advanced = Vec::new();

        for _ in 0..config.max_symbols_per_step {
            if active.is_empty() {
                break;
            }

            let mut symbol_candidates = Vec::new();

            for hyp in &active {
                let (logits, new_state_1, new_state_2) = decoder(
                    &frame,
                    hyp.last_token,
                    &hyp.state_1,
                    &hyp.state_2,
                )?;

                let log_probs = log_softmax(&logits);
                for token_id in top_k_indices(&log_probs, config.beam_size) {
                    let log_prob = log_probs[token_id];
                    let new_score = hyp.score + log_prob;

                    if token_id == blank_id {
                        // Blank only advances time. Per RNNT/greedy semantics the
                        // prediction network input and state stay at the last
                        // non-blank token (already carried in the cloned hyp); do
                        // NOT overwrite last_token with blank or feed it to the
                        // predictor, or the next frame decodes from a corrupted
                        // context.
                        let mut next = hyp.clone();
                        next.score = new_score;
                        next.frame = frame_idx + 1;
                        next.symbols_on_frame = 0;
                        advanced.push(next);
                    } else {
                        let mut next = hyp.clone();
                        next.score = new_score;
                        next.last_token = token_id as i32;
                        next.state_1 = new_state_1.clone();
                        next.state_2 = new_state_2.clone();
                        next.symbols_on_frame = hyp.symbols_on_frame + 1;
                        next.tokens.push((token_id, absolute_frame));
                        symbol_candidates.push(next);
                    }
                }
            }

            active = prune_beam(merge_hypotheses(symbol_candidates), config);

            if !advanced.is_empty() && active.is_empty() {
                break;
            }
        }

        // Greedy parity: paths that hit max_symbols_per_step still advance via blank.
        for hyp in active {
            let (logits, _, _) = decoder(
                &frame,
                hyp.last_token,
                &hyp.state_1,
                &hyp.state_2,
            )?;
            let log_probs = log_softmax(&logits);
            let mut next = hyp;
            next.score += log_probs[blank_id];
            next.frame = frame_idx + 1;
            // Keep last_token/state at the last non-blank (see blank branch above).
            next.symbols_on_frame = 0;
            advanced.push(next);
        }

        beam = prune_beam(merge_hypotheses(advanced), config);
    }

    let best = select_best(&beam, config.length_penalty);
    let new_tokens = best
        .as_ref()
        .map(|h| {
            let from = h.prev_token_count.min(h.tokens.len());
            h.tokens[from..].to_vec()
        })
        .unwrap_or_default();

    // Carry the full pruned beam across the chunk boundary. The frame loop
    // force-advances every surviving path to `end_frame` (blank advance plus the
    // greedy-parity block), so filtering on `frame == end_frame` would only ever
    // drop hypotheses if that invariant broke. Carrying all survivors avoids
    // silently losing valid hypotheses at chunk edges; the caller remaps their
    // `frame` to the next chunk's `start_frame`.
    debug_assert!(
        beam.iter().all(|h| h.frame == end_frame),
        "RNNT beam survivors must all terminate at end_frame ({end_frame})"
    );
    let survivors = beam;

    Ok(RnntBeamOutput {
        new_tokens,
        hypotheses: survivors,
        best_hypothesis: best,
    })
}

fn seed_hypothesis(
    blank_id: usize,
    start_frame: usize,
    state_shape: (usize, usize, usize),
) -> RnntHypothesis {
    RnntHypothesis {
        score: 0.0,
        frame: start_frame,
        last_token: blank_id as i32,
        state_1: Array3::zeros(state_shape),
        state_2: Array3::zeros(state_shape),
        symbols_on_frame: 0,
        tokens: Vec::new(),
        prev_token_count: 0,
    }
}

fn prune_beam(mut hyps: Vec<RnntHypothesis>, config: &BeamConfig) -> Vec<RnntHypothesis> {
    hyps.sort_by(|a, b| {
        length_normalized_score(b, config.length_penalty)
            .partial_cmp(&length_normalized_score(a, config.length_penalty))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hyps.truncate(config.beam_size);
    hyps
}

fn select_best(hyps: &[RnntHypothesis], length_penalty: f32) -> Option<RnntHypothesis> {
    hyps
        .iter()
        .max_by(|a, b| {
            length_normalized_score(a, length_penalty)
                .partial_cmp(&length_normalized_score(b, length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

fn length_normalized_score(hyp: &RnntHypothesis, length_penalty: f32) -> f32 {
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

fn merge_key(hyp: &RnntHypothesis) -> (usize, i32, Vec<usize>) {
    (
        hyp.frame,
        hyp.last_token,
        hyp.tokens.iter().map(|(id, _)| *id).collect(),
    )
}

fn merge_hypotheses(mut hyps: Vec<RnntHypothesis>) -> Vec<RnntHypothesis> {
    let mut merged: HashMap<(usize, i32, Vec<usize>), RnntHypothesis> = HashMap::new();
    for hyp in hyps.drain(..) {
        let key = merge_key(&hyp);
        merged
            .entry(key)
            .and_modify(|existing| {
                existing.score = log_sum_exp(existing.score, hyp.score);
            })
            .or_insert(hyp);
    }
    merged.into_values().collect()
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
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        let d = frame.shape()[1];
        let blank_id = d;
        let vocab = blank_id + 1;

        let mut logits = Array1::from_elem(vocab, -20.0f32);
        logits[blank_id] = 0.0;

        if frame[[0, 1, 0]] > 0.5 {
            if last_token == 1 {
                logits[blank_id] = 0.0;
            } else {
                logits[blank_id] = -20.0;
                logits[1] = 0.0;
            }
        }
        if frame[[0, 2, 0]] > 0.5 {
            if last_token == 2 {
                logits[blank_id] = 0.0;
            } else {
                logits[blank_id] = -20.0;
                logits[2] = 0.0;
            }
        }

        Ok((logits, state_1.clone(), state_2.clone()))
    }

    fn synthetic_encoder(frames: usize, hidden: usize) -> Array3<f32> {
        let mut enc = Array3::zeros((1, hidden, frames));
        if frames > 0 {
            enc[[0, 1, 0]] = 1.0;
        }
        if frames > 1 {
            enc[[0, 2, 1]] = 1.0;
        }
        enc
    }

    #[test]
    fn log_softmax_sums_to_zero_in_log_space() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let lp = log_softmax(&logits);
        let sum: f32 = lp.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn merge_combines_duplicate_paths_with_log_sum_exp() {
        let state = Array3::zeros((1, 1, 1));
        let hyp_a = RnntHypothesis {
            score: -1.0,
            frame: 1,
            last_token: 0,
            state_1: state.clone(),
            state_2: state.clone(),
            symbols_on_frame: 0,
            tokens: vec![(1, 0)],
            prev_token_count: 0,
        };
        let hyp_b = RnntHypothesis {
            score: -2.0,
            frame: 1,
            last_token: 0,
            state_1: state.clone(),
            state_2: state.clone(),
            symbols_on_frame: 0,
            tokens: vec![(1, 0)],
            prev_token_count: 0,
        };
        let merged = merge_hypotheses(vec![hyp_a, hyp_b]);
        assert_eq!(merged.len(), 1);
        let expected = log_sum_exp(-1.0, -2.0);
        assert!((merged[0].score - expected).abs() < 1e-5);
    }

    #[test]
    fn length_penalty_favors_shorter_paths_at_equal_per_token_score() {
        let state = Array3::zeros((1, 1, 1));
        let short = RnntHypothesis {
            score: -1.0,
            frame: 0,
            last_token: 0,
            state_1: state.clone(),
            state_2: state.clone(),
            symbols_on_frame: 0,
            tokens: vec![(1, 0)],
            prev_token_count: 0,
        };
        let long = RnntHypothesis {
            score: -2.0,
            frame: 0,
            last_token: 0,
            state_1: state.clone(),
            state_2: state,
            symbols_on_frame: 0,
            tokens: vec![(1, 0), (2, 1)],
            prev_token_count: 0,
        };
        let best = select_best(&[short, long], 0.5).unwrap();
        assert_eq!(best.tokens.len(), 1);
    }

    #[test]
    fn decode_emits_tokens_and_advances_blank() {
        let enc = synthetic_encoder(2, 4);
        let blank_id = 4;
        let config = BeamConfig {
            beam_size: 3,
            length_penalty: 0.0,
            patience: 3.5,
            max_symbols_per_step: 5,
            duration_reward: 0.67,
        };

        let mut decoder = mock_decoder;
        let out = decode_rnnt_beam(
            &enc,
            0,
            2,
            0,
            blank_id,
            &config,
            Vec::new(),
            &mut decoder,
        )
        .unwrap();

        assert_eq!(out.new_tokens, vec![(1, 0), (2, 1)]);
        assert!(!out.hypotheses.is_empty());
        assert_eq!(out.hypotheses[0].frame, 2);
    }

    #[test]
    fn decode_returns_only_new_tokens_on_continuation() {
        let enc = synthetic_encoder(2, 4);
        let blank_id = 4;
        let config = BeamConfig::default_unified();
        let state = Array3::zeros((1, 1, 1));

        // Carried from a prior chunk: already at frame 1 with one token emitted.
        let seed = RnntHypothesis {
            score: -0.5,
            frame: 1,
            last_token: blank_id as i32,
            state_1: state.clone(),
            state_2: state.clone(),
            symbols_on_frame: 0,
            tokens: vec![(1, 0)],
            prev_token_count: 0,
        };

        let mut decoder = mock_decoder;
        let out = decode_rnnt_beam(
            &enc,
            1,
            2,
            1,
            blank_id,
            &config,
            vec![seed],
            &mut decoder,
        )
        .unwrap();

        assert_eq!(out.new_tokens, vec![(2, 1)]);
    }

    #[test]
    fn survivors_all_terminate_at_end_frame() {
        let enc = synthetic_encoder(2, 4);
        let blank_id = 4;
        let config = BeamConfig::default_unified();

        let mut decoder = mock_decoder;
        let out = decode_rnnt_beam(&enc, 0, 2, 0, blank_id, &config, Vec::new(), &mut decoder)
            .unwrap();

        assert!(!out.hypotheses.is_empty());
        assert!(out.hypotheses.iter().all(|h| h.frame == 2));
    }

    #[test]
    fn multi_chunk_carry_matches_single_pass_transcript() {
        // Decode the same 2-frame utterance two ways:
        //   (a) one pass over [0, 2)
        //   (b) two passes [0, 1) then [1, 2), carrying survivors across the boundary
        // and remapping their frame to the next chunk's start (as the streaming
        // wrapper does). The concatenated tokens must match the single-pass result.
        let enc = synthetic_encoder(2, 4);
        let blank_id = 4;
        let config = BeamConfig::default_unified();
        let mut decoder = mock_decoder;

        let single = decode_rnnt_beam(&enc, 0, 2, 0, blank_id, &config, Vec::new(), &mut decoder)
            .unwrap();

        let chunk0 =
            decode_rnnt_beam(&enc, 0, 1, 0, blank_id, &config, Vec::new(), &mut decoder).unwrap();
        let mut carried = chunk0.hypotheses;
        assert!(!carried.is_empty(), "chunk 0 must carry survivors forward");
        for h in &mut carried {
            h.frame = 1; // remap to next chunk start_frame
        }
        let chunk1 =
            decode_rnnt_beam(&enc, 1, 2, 1, blank_id, &config, carried, &mut decoder).unwrap();

        let mut streamed = chunk0.new_tokens.clone();
        streamed.extend(chunk1.new_tokens.clone());

        let ids = |toks: &[(usize, usize)]| toks.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        assert_eq!(ids(&streamed), ids(&single.new_tokens));
    }
}
