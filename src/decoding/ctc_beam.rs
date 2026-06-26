use crate::decoding::BeamConfig;
use ndarray::Array2;
use std::collections::HashMap;

/// CTC prefix beam search over a logit matrix `[time, vocab]`.
///
/// Returns collapsed token ids (blank removed, repeats merged per CTC rules).
pub fn decode_ctc_beam(logits: &Array2<f32>, blank_id: usize, config: &BeamConfig) -> Vec<u32> {
    let time_steps = logits.shape()[0];
    if time_steps == 0 {
        return Vec::new();
    }

    let vocab_size = logits.shape()[1];
    let beam_size = config.beam_size.max(1);
    let top_k = beam_size.min(vocab_size);

    let mut beam: HashMap<Vec<u32>, CtcBeamEntry> = HashMap::new();
    beam.insert(
        Vec::new(),
        CtcBeamEntry {
            log_pb: 0.0,
            log_pnb: f32::NEG_INFINITY,
        },
    );

    for t in 0..time_steps {
        let log_probs = log_softmax_row(logits.row(t));
        let mut label_indices = top_k_indices(&log_probs, top_k);
        if !label_indices.contains(&blank_id) {
            label_indices.push(blank_id);
        }

        let mut next: HashMap<Vec<u32>, CtcBeamEntry> = HashMap::new();

        for (prefix, entry) in &beam {
            let log_pb = entry.log_pb;
            let log_pnb = entry.log_pnb;
            let log_p_blank = log_probs[blank_id];

            merge_entry(&mut next, prefix.clone(), |e| {
                e.log_pb = log_sum_exp(e.log_pb, log_sum_exp(log_pb, log_pnb) + log_p_blank);
            });

            for &label in &label_indices {
                if label == blank_id {
                    continue;
                }
                let log_p = log_probs[label];
                let token = label as u32;

                if !prefix.is_empty() && prefix[prefix.len() - 1] == token {
                    merge_entry(&mut next, prefix.clone(), |e| {
                        e.log_pnb = log_sum_exp(e.log_pnb, log_pb + log_p);
                    });
                } else {
                    let mut extended = prefix.clone();
                    extended.push(token);
                    merge_entry(&mut next, extended, |e| {
                        e.log_pnb =
                            log_sum_exp(e.log_pnb, log_sum_exp(log_pb, log_pnb) + log_p);
                    });
                }
            }
        }

        beam = prune_beam(next, beam_size);
    }

    beam
        .iter()
        .map(|(prefix, entry)| {
            (
                prefix.clone(),
                prefix_score(entry),
                entry.log_pb,
                entry.log_pnb,
            )
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(prefix, _, _, _)| prefix)
        .unwrap_or_default()
}

#[derive(Debug, Clone)]
struct CtcBeamEntry {
    log_pb: f32,
    log_pnb: f32,
}

fn prefix_score(entry: &CtcBeamEntry) -> f32 {
    log_sum_exp(entry.log_pb, entry.log_pnb)
}

fn merge_entry<F>(map: &mut HashMap<Vec<u32>, CtcBeamEntry>, key: Vec<u32>, update: F)
where
    F: FnOnce(&mut CtcBeamEntry),
{
    match map.get_mut(&key) {
        Some(entry) => update(entry),
        None => {
            let mut entry = CtcBeamEntry {
                log_pb: f32::NEG_INFINITY,
                log_pnb: f32::NEG_INFINITY,
            };
            update(&mut entry);
            map.insert(key, entry);
        }
    }
}

fn prune_beam(mut beam: HashMap<Vec<u32>, CtcBeamEntry>, beam_size: usize) -> HashMap<Vec<u32>, CtcBeamEntry> {
    let mut ranked: Vec<(Vec<u32>, CtcBeamEntry)> = beam.drain().collect();
    ranked.sort_by(|(_, a), (_, b)| {
        prefix_score(b)
            .partial_cmp(&prefix_score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(beam_size);
    ranked.into_iter().collect()
}

fn log_softmax_row(row: ndarray::ArrayView1<f32>) -> Vec<f32> {
    let max_logit = row
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = row.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let log_sum = sum.ln() + max_logit;
    row.iter().map(|&x| x - log_sum).collect()
}

fn top_k_indices(log_probs: &[f32], k: usize) -> Vec<usize> {
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

/// Greedy per-frame argmax + CTC collapse (for tests).
#[cfg(test)]
pub(crate) fn decode_ctc_greedy(logits: &Array2<f32>, blank_id: usize) -> Vec<u32> {
    let time_steps = logits.shape()[0];
    let mut token_ids = Vec::with_capacity(time_steps);
    for t in 0..time_steps {
        let row = logits.row(t);
        let max_idx = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        token_ids.push(max_idx as u32);
    }
    ctc_collapse(&token_ids, blank_id)
}

#[cfg(test)]
fn ctc_collapse(token_ids: &[u32], blank_id: usize) -> Vec<u32> {
    let mut result = Vec::new();
    let mut prev_token: Option<u32> = None;

    for &token_id in token_ids {
        if token_id == blank_id as u32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn logits_from_frame_probs(frame_probs: &[&[(usize, f32)]], vocab: usize) -> Array2<f32> {
        let t = frame_probs.len();
        let mut logits = Array2::from_elem((t, vocab), -20.0f32);
        for (frame, probs) in frame_probs.iter().enumerate() {
            for &(idx, p) in *probs {
                logits[[frame, idx]] = p.ln();
            }
        }
        logits
    }

    #[test]
    fn collapse_removes_blanks_and_repeats() {
        let collapsed = ctc_collapse(&[0, 1, 1, 0, 2, 2, 0], 0);
        assert_eq!(collapsed, vec![1, 2]);
    }

    #[test]
    fn beam_decodes_simple_ab_sequence() {
        // blank=0, a=1, b=2
        let logits = logits_from_frame_probs(
            &[
                &[(0, 0.1), (1, 0.8)],
                &[(0, 0.1), (1, 0.7)],
                &[(0, 0.1), (2, 0.8)],
                &[(0, 0.9)],
            ],
            3,
        );
        let config = BeamConfig {
            beam_size: 3,
            length_penalty: 0.0,
            patience: 1.0,
            max_symbols_per_step: 10,
            duration_reward: 0.67,
        };
        let tokens = decode_ctc_beam(&logits, 0, &config);
        assert_eq!(tokens, vec![1, 2]);
    }

    #[test]
    fn beam_matches_greedy_on_clean_path() {
        let logits = logits_from_frame_probs(
            &[
                &[(0, 0.05), (1, 0.9)],
                &[(0, 0.9)],
                &[(0, 0.05), (2, 0.9)],
                &[(0, 0.9)],
            ],
            3,
        );
        let config = BeamConfig {
            beam_size: 3,
            length_penalty: 0.0,
            patience: 1.0,
            max_symbols_per_step: 10,
            duration_reward: 0.67,
        };
        assert_eq!(
            decode_ctc_beam(&logits, 0, &config),
            decode_ctc_greedy(&logits, 0)
        );
    }
}
