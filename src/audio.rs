use crate::config::{CtcFrontendConfig, NormalizationMode, PreprocessorConfig};
use crate::error::{Error, Result};
use hound::{WavReader, WavSpec};
use ndarray::{s, Array2};
use realfft::RealToComplex;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

/// Cached, reusable mel filterbank + FFT plan keyed to a preprocessor
/// cfgs.
pub struct FeatureCache {
    pub mel_basis: Array2<f32>,
    /// HTK / unnormalized filterbank used by Granite-style (`DbFloorScale`) frontends.
    pub mel_basis_htk: Array2<f32>,
    /// `[freq, n_mels]` transpose of `mel_basis` for `[time, freq] @ W` GEMM.
    mel_basis_t: Array2<f32>,
    /// `[freq, n_mels]` transpose of `mel_basis_htk`.
    mel_basis_htk_t: Array2<f32>,
    pub fft_plan: Arc<dyn RealToComplex<f32>>,
    hann: Vec<f32>,
    hann_periodic: Vec<f32>,
}

impl FeatureCache {
    pub fn from_config(config: &PreprocessorConfig) -> Self {
        let mel_basis =
            create_mel_filterbank_slaney(config.n_fft, config.feature_size, config.sampling_rate);
        let mel_basis_htk =
            create_mel_filterbank_htk(config.n_fft, config.feature_size, config.sampling_rate);
        let mel_basis_t = mel_basis.t().as_standard_layout().into_owned();
        let mel_basis_htk_t = mel_basis_htk.t().as_standard_layout().into_owned();
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(config.n_fft);
        Self {
            mel_basis,
            mel_basis_htk,
            mel_basis_t,
            mel_basis_htk_t,
            fft_plan,
            hann: hann_window(config.win_length),
            hann_periodic: hann_window_periodic(config.win_length),
        }
    }
}

pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, WavSpec)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read float samples: {e}")))?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read int samples: {e}")))?,
    };

    Ok((samples, spec))
}

pub fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    let mut result = audio.to_vec();
    apply_preemphasis_in_place(&mut result, coef);
    result
}

/// First-order preemphasis in place. Walks backward so `x[i-1]` is still original.
fn apply_preemphasis_in_place(audio: &mut [f32], coef: f32) {
    if audio.len() < 2 {
        return;
    }
    for i in (1..audio.len()).rev() {
        audio[i] -= coef * audio[i - 1];
    }
}

fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

/// `torch.hann_window(..., periodic=True)` used by torchaudio MelSpectrogram / torch.stft.
fn hann_window_periodic(window_length: usize) -> Vec<f32> {
    if window_length == 0 {
        return Vec::new();
    }
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / window_length as f32).cos())
        .collect()
}

// We use proper FFT here instead of naive DFT because the model was trained
// on correctly computed spectrograms. Naive DFT produces wrong frequency bins
// and the model outputs all blank tokens. realfft (real-valued FFT wrapper around
// RustFFT) gives us O(n log n) performance and numerically correct results.
pub fn stft(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>> {
    let window = hann_window(win_length);
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(n_fft);
    stft_with_plan_window(audio, &plan, n_fft, hop_length, &window)
}

fn stft_with_plan_window(
    audio: &[f32],
    plan: &Arc<dyn RealToComplex<f32>>,
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> Result<Array2<f32>> {
    let frames = stft_zero_pad_time_major(audio, plan, n_fft, hop_length, window)?;
    Ok(frames.t().as_standard_layout().into_owned())
}

fn window_into_fft_input(input: &mut [f32], frame: &[f32], window: &[f32], window_offset: usize) {
    let win_length = window.len();
    input[..window_offset].fill(0.0);
    input[window_offset + win_length..].fill(0.0);
    for i in 0..win_length {
        input[window_offset + i] = frame[i] * window[i];
    }
}

/// Power STFT with NeMo zero-pad. Layout `[time, freq]` (contiguous per frame).
fn stft_zero_pad_time_major(
    audio: &[f32],
    plan: &Arc<dyn RealToComplex<f32>>,
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> Result<Array2<f32>> {
    let win_length = window.len();
    let pad_amount = n_fft / 2;
    let mut padded = vec![0.0f32; pad_amount];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad_amount, 0.0);

    let num_frames = audio.len() / hop_length;
    let freq_bins = n_fft / 2 + 1;
    let mut spectrogram = Array2::<f32>::zeros((num_frames, freq_bins));
    let window_offset = (n_fft - win_length) / 2;

    let mut input = vec![0.0f32; n_fft];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        let frame = &padded[start + window_offset..start + window_offset + win_length];
        window_into_fft_input(&mut input, frame, window, window_offset);

        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;

        let mut row = spectrogram.row_mut(frame_idx);
        if let Some(dest) = row.as_slice_mut() {
            for (d, c) in dest.iter_mut().zip(output.iter()) {
                *d = c.norm_sqr();
            }
        } else {
            for (k, c) in output.iter().enumerate().take(freq_bins) {
                row[k] = c.norm_sqr();
            }
        }
    }

    Ok(spectrogram)
}

pub fn stft_with_plan(
    audio: &[f32],
    plan: &Arc<dyn RealToComplex<f32>>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>> {
    let window = hann_window(win_length);
    stft_with_plan_window(audio, plan, n_fft, hop_length, &window)
}

// Slaney mel scale (again librosa)
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.06875177742094912;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

/// librosa Slaney-scale filterbank with `norm="slaney"` (Whisper / HuggingFace default).
pub fn create_mel_filterbank_slaney(
    n_fft: usize,
    n_mels: usize,
    sample_rate: usize,
) -> Array2<f32> {
    create_mel_filterbank(n_fft, n_mels, sample_rate)
}

pub fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));

    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);

    // Mel cent freq
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    // FFT bin freq
    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    // librosa's ramp
    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();

    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            filterbank[[i, k]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }

    // Slaney norm
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
        for k in 0..freq_bins {
            filterbank[[i, k]] *= enorm as f32;
        }
    }

    filterbank
}

fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// torchaudio default MelSpectrogram filterbank: HTK scale, `norm=None`.
pub fn create_mel_filterbank_htk(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));
    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_htk(0.0);
    let mel_max = hz_to_mel_htk(fmax);
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz_htk(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();
    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            filterbank[[i, k]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }
    filterbank
}

/// Centered, reflect-padded STFT matching torchaudio `MelSpectrogram` / `torch.stft`.
/// Layout `[time, freq]` (contiguous per frame for FFT writes and GEMM).
fn stft_centered_reflect_time_major(
    audio: &[f32],
    plan: &Arc<dyn RealToComplex<f32>>,
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> Result<Array2<f32>> {
    let win_length = window.len();
    let pad_amount = n_fft / 2;
    let padded = reflect_pad_1d(audio, pad_amount);
    let window_offset = (n_fft - win_length) / 2;
    // torch.stft(center=True): 1 + T // hop
    let num_frames = if hop_length == 0 {
        0
    } else {
        1 + audio.len() / hop_length
    };
    let freq_bins = n_fft / 2 + 1;
    let mut spectrogram = Array2::<f32>::zeros((num_frames, freq_bins));

    let mut input = vec![0.0f32; n_fft];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        input[..window_offset].fill(0.0);
        input[window_offset + win_length..].fill(0.0);
        for i in 0..win_length {
            let s = start + window_offset + i;
            input[window_offset + i] = if s < padded.len() {
                padded[s] * window[i]
            } else {
                0.0
            };
        }
        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;

        let mut row = spectrogram.row_mut(frame_idx);
        if let Some(dest) = row.as_slice_mut() {
            for (d, c) in dest.iter_mut().zip(output.iter()) {
                *d = c.norm_sqr();
            }
        } else {
            for (k, c) in output.iter().enumerate().take(freq_bins) {
                row[k] = c.norm_sqr();
            }
        }
    }
    Ok(spectrogram)
}

fn reflect_pad_1d(audio: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 {
        return audio.to_vec();
    }
    if audio.is_empty() {
        return vec![0.0; pad * 2];
    }
    let len = audio.len();
    let mut out = vec![0.0f32; len + 2 * pad];
    out[pad..pad + len].copy_from_slice(audio);
    for i in 0..pad {
        out[i] = audio[reflect_index(i as isize - pad as isize, len)];
        out[pad + len + i] = audio[reflect_index((len + i) as isize, len)];
    }
    out
}

fn reflect_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let n = len as isize;
    let period = 2 * (n - 1);
    let mut x = idx % period;
    if x < 0 {
        x += period;
    }
    if x >= n {
        (period - x) as usize
    } else {
        x as usize
    }
}

/// Extract mel spectrogram features from raw audio samples.
///
/// The `cache` holds the mel filterbank and FFT plan built once at model
/// load — these are deterministic from `config` and identical across calls,
/// so reusing them avoids rebuilding ~15-20 µs of arithmetic per request.
///
/// Uses the Parakeet-CTC-0.6b frontend (`ln` + per-feature z-norm, no deltas
/// or stacking). CTC models with ONNX metadata should call
/// [`extract_features_with_frontend`] instead.
pub fn extract_features_with_cache(
    audio: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    config: &PreprocessorConfig,
    cache: &FeatureCache,
) -> Result<Array2<f32>> {
    extract_features_with_frontend(
        audio,
        sample_rate,
        channels,
        config,
        cache,
        &CtcFrontendConfig::default(),
    )
}

/// Like [`extract_features_with_cache`], with optional CTC frontend extras
/// (alternate log/norm, deltas, frame stacking).
///
/// [`CtcFrontendConfig::default`] takes the same path as
/// [`extract_features_with_cache`].
pub fn extract_features_with_frontend(
    mut audio: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    config: &PreprocessorConfig,
    cache: &FeatureCache,
    frontend: &CtcFrontendConfig,
) -> Result<Array2<f32>> {
    if sample_rate != config.sampling_rate as u32 {
        return Err(Error::Audio(format!(
            "Audio sample rate {} doesn't match expected {}. Please resample your audio first.",
            sample_rate, config.sampling_rate
        )));
    }

    if channels > 1 {
        let mono: Vec<f32> = audio
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();
        audio = mono;
    }

    // NeMo / Parakeet-CTC uses first-order preemphasis. Granite-style
    // DbFloorScale frontends do not; skip so default PreprocessorConfig
    // preemphasis cannot leak into those models.
    if matches!(
        frontend.normalization,
        NormalizationMode::PerFeatureZNorm
    ) {
        apply_preemphasis_in_place(&mut audio, config.preemphasis);
    }

    let stack_factor = frontend.stack_factor.max(1);
    let mut features = match frontend.normalization {
        NormalizationMode::PerFeatureZNorm => {
            if stack_factor > 1 {
                pad_audio_for_stacking(&mut audio, config.hop_length, stack_factor);
            }
            let spectrogram = stft_zero_pad_time_major(
                &audio,
                &cache.fft_plan,
                config.n_fft,
                config.hop_length,
                &cache.hann,
            )?;
            let mut mel = spectrogram.dot(&cache.mel_basis_t);
            normalize_per_feature_z_norm(&mut mel);
            mel
        }
        NormalizationMode::DbFloorScale { floor_db } => {
            granite_logmel(&audio, cache, config, stack_factor, floor_db)?
        }
    };

    let win = frontend.delta_win_length.max(3);
    if frontend.deltas && stack_factor > 1 && win == 3 {
        features = stack_with_win3_deltas(&features, stack_factor);
    } else {
        if frontend.deltas {
            features = apply_deltas(&features, win);
        }
        if stack_factor > 1 {
            features = stack_frames(&features, stack_factor);
        }
    }

    Ok(features)
}

/// Whisper-style log-mel for ARK-ASR / HuggingFace `WhisperFeatureExtractor`.
///
/// Centered reflect STFT, Slaney-normalized mel, `log10` clamp to
/// `(max - floor_db)` then `/4 + 1`. Returns `[n_mels, time]` (encoder layout).
pub fn extract_whisper_logmel(
    audio: &[f32],
    cache: &FeatureCache,
    config: &PreprocessorConfig,
    floor_db: f32,
) -> Result<Array2<f32>> {
    let spectrogram = stft_centered_reflect_time_major(
        audio,
        &cache.fft_plan,
        config.n_fft,
        config.hop_length.max(1),
        &cache.hann_periodic,
    )?;
    let mut logmel = spectrogram.dot(&cache.mel_basis_t);
    normalize_db_floor_inplace(&mut logmel, floor_db);
    Ok(logmel.t().as_standard_layout().into_owned())
}

/// Granite / torchaudio frontend: centered reflect STFT, HTK mel, `log10` dB floor.
/// Pads the waveform so the last stack group is filled, then slices to `n_frames`.
fn granite_logmel(
    audio: &[f32],
    cache: &FeatureCache,
    config: &PreprocessorConfig,
    stack_factor: usize,
    floor_db: f32,
) -> Result<Array2<f32>> {
    let hop = config.hop_length.max(1);
    let stack = stack_factor.max(1);
    let mel_frames = audio.len() / hop;
    let n_frames = stack * mel_frames.div_ceil(stack).max(1);
    let need = (n_frames.saturating_sub(1)) * hop + 1;
    let spectrogram = if audio.len() < need {
        let mut padded = audio.to_vec();
        padded.resize(need, 0.0);
        stft_centered_reflect_time_major(
            &padded,
            &cache.fft_plan,
            config.n_fft,
            hop,
            &cache.hann_periodic,
        )?
    } else {
        stft_centered_reflect_time_major(
            audio,
            &cache.fft_plan,
            config.n_fft,
            hop,
            &cache.hann_periodic,
        )?
    };
    let time = spectrogram.nrows().min(n_frames);
    let sliced = spectrogram.slice(s![..time, ..]);
    let mut mel = sliced.dot(&cache.mel_basis_htk_t);
    normalize_db_floor_inplace(&mut mel, floor_db);
    Ok(mel)
}

fn pad_audio_for_stacking(audio: &mut Vec<f32>, hop_length: usize, stack_factor: usize) {
    if hop_length == 0 || stack_factor <= 1 {
        return;
    }
    let mel_frames = audio.len() / hop_length;
    let n_frames = stack_factor * mel_frames.div_ceil(stack_factor);
    let need = n_frames * hop_length;
    if audio.len() < need {
        audio.resize(need, 0.0);
    }
}

/// NeMo log + per-feature z-norm. `mel` is `[time, n_mels]` power spectrogram.
fn normalize_per_feature_z_norm(mel: &mut Array2<f32>) {
    let log_zero_guard: f32 = 2.0f32.powi(-24);
    for x in mel.iter_mut() {
        *x = (*x + log_zero_guard).ln();
    }

    let num_frames = mel.shape()[0];
    let num_features = mel.shape()[1];

    if num_frames <= 1 {
        return;
    }

    for feat_idx in 0..num_features {
        let mut column = mel.column_mut(feat_idx);
        let mean: f32 = column.iter().sum::<f32>() / num_frames as f32;
        let variance: f32 =
            column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (num_frames as f32 - 1.0);
        let std = variance.sqrt() + 1e-5;

        for val in column.iter_mut() {
            *val = (*val - mean) / std;
        }
    }
}

/// Granite TurboCTC log-mel: `log10`, clamp to `(max - floor_db)`, then `/4 + 1`.
/// `mel` is `[time, n_mels]` power spectrogram, updated in place.
fn normalize_db_floor_inplace(mel: &mut Array2<f32>, floor_db: f32) {
    let mut mx = f32::NEG_INFINITY;
    for x in mel.iter_mut() {
        *x = x.max(1e-10).log10();
        if *x > mx {
            mx = *x;
        }
    }
    if mx.is_finite() {
        let floor = mx - floor_db;
        for x in mel.iter_mut() {
            *x = x.max(floor) / 4.0 + 1.0;
        }
    }
}

/// Concatenate static features with temporal deltas along the feature axis.
/// Matches `torchaudio.functional.compute_deltas` (regression kernel, replicate pad).
fn apply_deltas(features: &Array2<f32>, win_length: usize) -> Array2<f32> {
    let time = features.shape()[0];
    let feat = features.shape()[1];
    let mut out = Array2::<f32>::zeros((time, feat * 2));
    if time == 0 {
        return out;
    }
    let n = (win_length.saturating_sub(1)) / 2;
    let denom: f32 = 2.0 * (1..=n).map(|i| (i * i) as f32).sum::<f32>();
    for t in 0..time {
        let tm = t.saturating_sub(n).min(time - 1);
        let tp = (t + n).min(time - 1);
        for f in 0..feat {
            let x = features[[t, f]];
            out[[t, f]] = x;
            if n == 0 || denom == 0.0 {
                continue;
            }
            // win_length=3 => n=1, denom=2, delta = 0.5 * (x[t+1] - x[t-1])
            if n == 1 {
                out[[t, feat + f]] = 0.5 * (features[[tp, f]] - features[[tm, f]]);
                continue;
            }
            let mut acc = 0.0f32;
            for k in 0..=(2 * n) {
                let offset = k as isize - n as isize;
                let src = (t as isize + offset).clamp(0, time as isize - 1) as usize;
                acc += (offset as f32 / denom) * features[[src, f]];
            }
            out[[t, feat + f]] = acc;
        }
    }
    out
}

/// Concatenate groups of `stack_factor` consecutive frames along the feature axis.
/// Right-pads with zeros when `time` is not divisible by `stack_factor`.
fn stack_frames(features: &Array2<f32>, stack_factor: usize) -> Array2<f32> {
    let time = features.shape()[0];
    let feat = features.shape()[1];
    if stack_factor <= 1 || time == 0 {
        return features.clone();
    }
    let padded_time = time.div_ceil(stack_factor) * stack_factor;
    let out_time = padded_time / stack_factor;
    let out_feat = feat * stack_factor;
    let mut out = Array2::<f32>::zeros((out_time, out_feat));
    for t in 0..out_time {
        for s in 0..stack_factor {
            let src_t = t * stack_factor + s;
            if src_t >= time {
                break;
            }
            out.slice_mut(s![t, s * feat..(s + 1) * feat])
                .assign(&features.row(src_t));
        }
    }
    out
}

/// Fuse `apply_deltas(..., 3)` and `stack_frames` into one allocation.
fn stack_with_win3_deltas(features: &Array2<f32>, stack_factor: usize) -> Array2<f32> {
    let time = features.shape()[0];
    let feat = features.shape()[1];
    if time == 0 {
        return Array2::zeros((0, feat * 2 * stack_factor.max(1)));
    }
    let padded_time = time.div_ceil(stack_factor) * stack_factor;
    let out_time = padded_time / stack_factor;
    let pair = feat * 2;
    let mut out = Array2::<f32>::zeros((out_time, pair * stack_factor));
    let last = time - 1;
    for t in 0..out_time {
        for s in 0..stack_factor {
            let src_t = t * stack_factor + s;
            if src_t >= time {
                break;
            }
            let tm = src_t.saturating_sub(1);
            let tp = (src_t + 1).min(last);
            let base = s * pair;
            for f in 0..feat {
                let x = features[[src_t, f]];
                out[[t, base + f]] = x;
                out[[t, base + feat + f]] = 0.5 * (features[[tp, f]] - features[[tm, f]]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sine wave at the given frequency and sample rate.
    fn sine_wave(freq_hz: f32, sample_rate: usize, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn stft_concentrates_power_at_expected_bin() {
        // 1kHz sine at 16kHz sample rate, 1 second
        let n_fft = 512;
        let hop_length = 160;
        let win_length = 400;
        let sample_rate = 16000;
        let audio = sine_wave(1000.0, sample_rate, sample_rate);

        let spec = stft(&audio, n_fft, hop_length, win_length).unwrap();

        // Expected bin for 1kHz: freq_hz * n_fft / sample_rate = 1000 * 512 / 16000 = 32
        let expected_bin = 32;
        let freq_bins = n_fft / 2 + 1;
        let num_frames = spec.shape()[1];

        // Check that bin 32 has the highest power in most frames (skip edge frames)
        let mut correct_frames = 0;
        for frame in 2..num_frames.saturating_sub(2) {
            let mut max_bin = 0;
            let mut max_power = 0.0f32;
            for bin in 0..freq_bins {
                if spec[[bin, frame]] > max_power {
                    max_power = spec[[bin, frame]];
                    max_bin = bin;
                }
            }
            if max_bin == expected_bin {
                correct_frames += 1;
            }
        }

        let interior_frames = num_frames.saturating_sub(4);
        assert!(
            correct_frames > interior_frames / 2,
            "Expected bin {expected_bin} to dominate in most frames, but only {correct_frames}/{interior_frames}"
        );
    }

    #[test]
    fn stft_output_shape_is_correct() {
        let n_fft = 512;
        let hop_length = 160;
        let win_length = 400;
        let audio = vec![0.0f32; 16000]; // 1 second of silence

        let spec = stft(&audio, n_fft, hop_length, win_length).unwrap();

        let freq_bins = n_fft / 2 + 1;
        assert_eq!(spec.shape()[0], freq_bins);
        assert_eq!(spec.shape()[1], audio.len() / hop_length);
    }
}
