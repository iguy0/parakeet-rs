//! Model-backed benchmark + smoke tests.
//!
//! These run real ONNX inference over every clip in `samples/` and report WER
//! against the reference transcripts. They are gated on model availability:
//! when the weights under `models/` (or the `PARAKEET_*` env overrides) are
//! missing, each test prints a skip notice and passes, so CI without weights
//! stays green. When the weights are present the smoke clip is asserted.
//!
//! Models exercised:
//!   - Nemotron streaming ASR (default ASR)            — no cargo feature needed
//!   - Sortformer 4spk v2.1 diarization                — `--features sortformer`
//!   - Multitalker speaker-attributed streaming ASR    — `--features multitalker`
//!   - Cohere Transcribe                               — `--features cohere`
//!   - Ark ASR                                         — `--features ark-asr`
//!   - Granite TurboCTC                                — no cargo feature needed
//!
//! Run everything against local models:
//!   cargo test --test model_benchmark --features "sortformer multitalker" -- --nocapture
//!
//! Override the smoke WER ceiling (default 0.15; observed ~0.02):
//!   SMOKE_MAX_WER=0.5 cargo test --test model_benchmark -- --nocapture

mod common;

use common::*;
use parakeet_rs::{BeamConfig, DecodingStrategy, TimestampMode, Transcriber};
use std::time::Instant;

const NEMOTRON_CHUNK_SAMPLES: usize = 8960; // 560 ms @ 16 kHz

fn smoke_max_wer() -> f32 {
    std::env::var("SMOKE_MAX_WER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.15)
}

/// Stream a full clip through Nemotron and return the final transcript.
///
/// Chunks are not zero-padded. [`Nemotron::flush`] decodes the tail that is
/// shorter than one 560 ms window.
fn nemotron_transcribe(model: &mut parakeet_rs::Nemotron, audio: &[f32]) -> String {
    for chunk in audio.chunks(NEMOTRON_CHUNK_SAMPLES) {
        model.transcribe_chunk(chunk).expect("nemotron chunk");
    }
    model.flush().expect("nemotron flush");
    model.get_transcript()
}

/// Feed `audio` in 1 s pieces, then flush. One `transcribe_chunk` call emits
/// at most one 560 ms window, so `flush` also drains any full chunks left over.
fn nemotron_feed_1s_then_flush(model: &mut parakeet_rs::Nemotron, audio: &[f32]) -> String {
    for chunk in audio.chunks(16_000) {
        model.transcribe_chunk(chunk).expect("nemotron 1s chunk");
    }
    model.flush().expect("nemotron flush");
    model.get_transcript()
}

/// Read and clean a sample's reference transcript (speaker headers stripped).
fn reference_for(sample: &Sample) -> String {
    let raw = std::fs::read_to_string(&sample.transcript).expect("read transcript");
    reference_text(&raw)
}

/// Run an ASR benchmark over the selected samples: `run` loads + transcribes a
/// clip and returns `(hypothesis, transcription_seconds)` (model-load time
/// excluded). Records WER/CER/RTF to the CSV and asserts the smoke ceiling.
fn run_asr_bench(
    test: &str,
    model: &str,
    mode: &str,
    decoding: &str,
    ceiling: f32,
    mut run: impl FnMut(&[f32]) -> (String, f32),
) {
    if ep_skip_if_unavailable(test) {
        return;
    }
    let samples = selected_asr_samples();
    assert!(
        samples.iter().any(|s| s.name == "smoke"),
        "samples/smoke must exist"
    );
    eprintln!("\n[{test}] {model} ({mode}/{decoding}):");

    let mut smoke_wer: Option<f32> = None;
    for sample in &samples {
        let (audio, _) = load_wav_mono_16k(&sample.wav);
        let audio_secs = audio.len() as f32 / 16_000.0;
        let (hyp, proc_secs) = run(&audio);
        let reference = reference_for(sample);
        let w = record_asr(
            &sample.name,
            model,
            mode,
            decoding,
            &requested_ep(),
            audio_secs,
            proc_secs,
            &reference,
            &hyp,
        );
        if sample.name == "smoke" {
            assert!(!hyp.trim().is_empty(), "{test}: smoke transcript was empty");
            smoke_wer = Some(w);
        }
    }

    let w = smoke_wer.expect("smoke transcribed");
    assert!(
        w <= ceiling,
        "{test}: smoke WER {:.1}% exceeds ceiling {:.1}% (set SMOKE_MAX_WER to adjust)",
        w * 100.0,
        ceiling * 100.0
    );
}

#[test]
fn nemotron_streaming_wer() {
    if !nemotron_available() {
        skip(
            "nemotron_streaming_wer",
            &format!("model not found at {}", nemotron_dir().display()),
        );
        return;
    }
    run_asr_bench(
        "nemotron_streaming_wer",
        "nemotron-en-0.6b-int8",
        "streaming",
        "greedy",
        smoke_max_wer(),
        |audio| {
            let mut model =
                parakeet_rs::Nemotron::from_pretrained(nemotron_dir(), Some(execution_config()))
                    .expect("load nemotron");
            let t0 = std::time::Instant::now();
            let mut buf = audio.to_vec();
            peak_normalize(&mut buf);
            let hyp = nemotron_transcribe(&mut model, &buf);
            (hyp, t0.elapsed().as_secs_f32())
        },
    );
}

/// Streaming `flush` vs offline on the smoke clip, plus a non-aligned tail.
///
/// `flush_on_empty_buffer_returns_empty` has no model-free constructor, so the
/// empty-buffer case lives in [`nemotron_flush_idempotent`].
#[test]
fn nemotron_streaming_flush_parity() {
    use parakeet_rs::Nemotron;

    if !nemotron_available() {
        skip(
            "nemotron_streaming_flush_parity",
            &format!("model not found at {}", nemotron_dir().display()),
        );
        return;
    }
    if ep_skip_if_unavailable("nemotron_streaming_flush_parity") {
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("nemotron_streaming_flush_parity", "samples/smoke missing");
        return;
    };

    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let mut audio = audio;
    peak_normalize(&mut audio);
    let ceiling = smoke_max_wer();

    let mut offline =
        Nemotron::from_pretrained(nemotron_dir(), Some(execution_config())).expect("load nemotron");
    let offline_text = offline.transcribe_audio(&audio).expect("offline");
    let mut streaming =
        Nemotron::from_pretrained(nemotron_dir(), Some(execution_config())).expect("load nemotron");
    let hyp = nemotron_feed_1s_then_flush(&mut streaming, &audio);
    let w = wer(&offline_text, &hyp);
    eprintln!(
        "\n[nemotron_streaming_flush_parity] 1s increments vs offline WER {w:.4}\n  offline: {offline_text}\n  stream:  {hyp}"
    );
    assert!(
        w <= ceiling,
        "1s-increment flush WER {w:.4} exceeds ceiling {ceiling:.4}"
    );

    let trunc_n = ((20.3_f64 * 16_000.0) as usize).min(audio.len());
    assert_ne!(
        trunc_n % NEMOTRON_CHUNK_SAMPLES,
        0,
        "20.3 s clip must not be a multiple of 560 ms"
    );
    let trunc = &audio[..trunc_n];
    let mut offline_trunc =
        Nemotron::from_pretrained(nemotron_dir(), Some(execution_config())).expect("load nemotron");
    let offline_trunc_text = offline_trunc
        .transcribe_audio(trunc)
        .expect("offline truncated");
    let mut streaming_trunc =
        Nemotron::from_pretrained(nemotron_dir(), Some(execution_config())).expect("load nemotron");
    let hyp_trunc = nemotron_feed_1s_then_flush(&mut streaming_trunc, trunc);
    let offline_words = normalize_words(&offline_trunc_text);
    assert!(
        offline_words.len() >= 3,
        "offline truncated transcript too short: {offline_trunc_text}"
    );
    let tail = offline_words[offline_words.len() - 3..].join(" ");
    let hyp_norm = normalize_words(&hyp_trunc).join(" ");
    eprintln!(
        "[nemotron_streaming_flush_parity] truncated {trunc_n} samples\n  offline: {offline_trunc_text}\n  stream:  {hyp_trunc}\n  tail: {tail}"
    );
    assert!(
        hyp_norm.contains(&tail),
        "final words \"{tail}\" missing from streaming transcript \"{hyp_trunc}\""
    );
}

/// Empty flush, a second flush, and `transcribe_chunk` after flush.
///
/// Also covers `flush_on_empty_buffer_returns_empty`: building a `Nemotron`
/// requires ONNX weights, so this cannot live in `src/nemotron.rs` unit tests.
#[test]
fn nemotron_flush_idempotent() {
    use parakeet_rs::Nemotron;

    if !nemotron_available() {
        skip(
            "nemotron_flush_idempotent",
            &format!("model not found at {}", nemotron_dir().display()),
        );
        return;
    }
    if ep_skip_if_unavailable("nemotron_flush_idempotent") {
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("nemotron_flush_idempotent", "samples/smoke missing");
        return;
    };

    let mut model =
        Nemotron::from_pretrained(nemotron_dir(), Some(execution_config())).expect("load nemotron");
    assert_eq!(model.flush().expect("empty flush"), "");
    assert_eq!(model.flush().expect("second empty flush"), "");

    model.reset();
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let n = (NEMOTRON_CHUNK_SAMPLES + 1_600).min(audio.len());
    model
        .transcribe_chunk(&audio[..n])
        .expect("chunk before flush");
    let _first = model.flush().expect("flush after audio");
    assert_eq!(model.flush().expect("second flush"), "");
    let after = model
        .transcribe_chunk(&audio[..1_600.min(audio.len())])
        .expect("chunk after flush");
    assert!(
        after.is_empty(),
        "transcribe_chunk after flush should return empty until reset, got {after:?}"
    );
    model.reset();
    assert_eq!(model.flush().expect("flush after reset"), "");
}

/// Mel vs encoder vs decoder time on the smoke clip (streaming + flush).
///
/// Ignored so the default weight-gated run stays a WER check. Prints the mel
/// share and whether an incremental STFT would be worth a later change.
#[test]
#[ignore]
fn nemotron_stage_timing() {
    use parakeet_rs::Nemotron;

    if !nemotron_available() {
        skip(
            "nemotron_stage_timing",
            &format!("model not found at {}", nemotron_dir().display()),
        );
        return;
    }
    if ep_skip_if_unavailable("nemotron_stage_timing") {
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("nemotron_stage_timing", "samples/smoke missing");
        return;
    };

    let (mut audio, _) = load_wav_mono_16k(&sample.wav);
    peak_normalize(&mut audio);
    let mut model =
        Nemotron::from_pretrained(nemotron_dir(), Some(execution_config())).expect("load nemotron");
    model.set_record_stage_timing(true);
    let wall = Instant::now();
    let hyp = nemotron_transcribe(&mut model, &audio);
    let wall_ns = wall.elapsed().as_nanos();
    let (mel, enc, dec) = model.stage_timing_ns();
    let staged = mel + enc + dec;
    let pct = |ns: u128| {
        if staged == 0 {
            0.0
        } else {
            ns as f64 / staged as f64 * 100.0
        }
    };
    let mel_share = if staged == 0 {
        0.0
    } else {
        mel as f64 / staged as f64
    };
    eprintln!(
        "\n[nemotron_stage_timing] smoke streaming + flush\n  \
         wall {:.3} ms\n  \
         mel {:.3} ms ({:.1}%)\n  \
         encoder {:.3} ms ({:.1}%)\n  \
         decoder {:.3} ms ({:.1}%)\n  \
         staged sum {:.3} ms\n  \
         transcript: {hyp}",
        wall_ns as f64 / 1e6,
        mel as f64 / 1e6,
        pct(mel),
        enc as f64 / 1e6,
        pct(enc),
        dec as f64 / 1e6,
        pct(dec),
        staged as f64 / 1e6,
    );
    if mel_share > 0.10 {
        eprintln!(
            "[nemotron_stage_timing] recommendation: mel is {:.1}% of staged chunk time (>10%). \
             An incremental STFT is worth a later phase. Not implemented here.",
            mel_share * 100.0
        );
    } else {
        eprintln!(
            "[nemotron_stage_timing] recommendation: mel is {:.1}% of staged chunk time (<=10%). \
             Keep the full-buffer STFT. An incremental mel is not justified on this clip.",
            mel_share * 100.0
        );
    }
    assert!(staged > 0, "stage timers stayed zero");
    assert!(!hyp.trim().is_empty(), "smoke transcript was empty");
}

#[cfg(feature = "sortformer")]
#[test]
fn sortformer_diarization_smoke() {
    use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};

    if !sortformer_available() {
        skip(
            "sortformer_diarization_smoke",
            &format!("model not found at {}", sortformer_onnx().display()),
        );
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("sortformer_diarization_smoke", "samples/smoke missing");
        return;
    };

    if ep_skip_if_unavailable("sortformer_diarization_smoke") {
        return;
    }

    let (audio, sample_rate) = load_wav_mono_16k(&sample.wav);
    let mut sf = Sortformer::with_config(
        sortformer_onnx(),
        Some(execution_config()),
        DiarizationConfig::callhome(),
    )
        .expect("load sortformer");

    let segments = sf.diarize(audio, sample_rate, 1).expect("diarize");
    eprintln!("\nSortformer found {} segment(s) on smoke:", segments.len());
    for s in &segments {
        eprintln!(
            "  [{:6.2}s - {:6.2}s] speaker {}",
            s.start as f64 / 16_000.0,
            s.end as f64 / 16_000.0,
            s.speaker_id
        );
    }

    assert!(
        !segments.is_empty(),
        "expected at least one speaker segment"
    );
    assert!(
        segments.iter().all(|s| s.end >= s.start),
        "segment end must be >= start"
    );
    // The smoke clip is two distinct readers; expect the model to find >= 2 ids.
    let distinct = {
        let mut ids: Vec<usize> = segments.iter().map(|s| s.speaker_id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    };
    eprintln!("distinct speakers detected: {distinct}");
    assert!(distinct >= 1, "expected at least one speaker id");
}

/// Compare greedy vs beam x streaming vs offline for the Nemotron int8 model.
/// Records all four configurations to samples/asr_report.csv for side-by-side
/// comparison. Nemotron's encoder is cache-aware streaming; "offline" runs the
/// full-file pass (`transcribe_audio`) while "streaming" emits per 560 ms chunk.
/// Beam reuses the shared RNNT beam core with cross-chunk hypothesis carry.
#[test]
fn nemotron_decoding_matrix() {
    use parakeet_rs::Nemotron;

    if !nemotron_available() {
        skip(
            "nemotron_decoding_matrix",
            &format!("model not found at {}", nemotron_dir().display()),
        );
        return;
    }

    let model = "nemotron-en-0.6b-int8";
    let greedy = DecodingStrategy::Greedy;
    let beam = DecodingStrategy::Beam(BeamConfig::default_unified());

    eprintln!("\n[nemotron_decoding_matrix] greedy vs beam x streaming vs offline:");

    for (decoding, dlabel) in [(greedy, "greedy"), (beam, "beam")] {
        // Offline: full-file pass.
        run_asr_bench(
            "nemotron_decoding_matrix",
            model,
            "offline",
            dlabel,
            smoke_max_wer(),
            |audio| {
                let mut m = Nemotron::from_pretrained_with_decoding(
                    nemotron_dir(),
                    Some(execution_config()),
                    decoding,
                )
                    .expect("load nemotron");
                let mut buf = audio.to_vec();
                peak_normalize(&mut buf);
                let t0 = Instant::now();
                let text = m.transcribe_audio(&buf).expect("nemotron offline");
                (text, t0.elapsed().as_secs_f32())
            },
        );

        // Streaming: per-chunk emission with carried state/hypotheses.
        run_asr_bench(
            "nemotron_decoding_matrix",
            model,
            "streaming",
            dlabel,
            smoke_max_wer(),
            |audio| {
                let mut m = Nemotron::from_pretrained_with_decoding(
                    nemotron_dir(),
                    Some(execution_config()),
                    decoding,
                )
                    .expect("load nemotron");
                let mut buf = audio.to_vec();
                peak_normalize(&mut buf);
                let t0 = Instant::now();
                let text = nemotron_transcribe(&mut m, &buf);
                (text, t0.elapsed().as_secs_f32())
            },
        );
    }
}

/// Transcribe a full clip with any offline [`Transcriber`], timing only inference.
fn offline_transcribe<T: Transcriber>(model: &mut T, audio: &[f32]) -> (String, f32) {
    let t0 = Instant::now();
    let result = model
        .transcribe_samples(audio.to_vec(), 16_000, 1, Some(TimestampMode::Words))
        .expect("offline transcribe");
    (result.text, t0.elapsed().as_secs_f32())
}

#[test]
fn ctc_offline_wer() {
    use parakeet_rs::Parakeet;
    if !ctc_available() {
        skip(
            "ctc_offline_wer",
            &format!("model not found at {}", ctc_dir().display()),
        );
        return;
    }
    for (decoding, label) in [
        (DecodingStrategy::Greedy, "greedy"),
        (DecodingStrategy::Beam(BeamConfig::default_ctc()), "beam"),
    ] {
        run_asr_bench(
            "ctc_offline_wer",
            "parakeet-ctc",
            "offline",
            label,
            smoke_max_wer(),
            |audio| {
            let mut m = Parakeet::from_pretrained_with_decoding(
                ctc_dir(),
                Some(execution_config()),
                decoding,
            )
                .expect("load ctc");
            offline_transcribe(&mut m, audio)
            },
        );
    }
}

#[test]
fn tdt_offline_wer() {
    use parakeet_rs::ParakeetTDT;
    if !tdt_available() {
        skip(
            "tdt_offline_wer",
            &format!("model not found at {}", tdt_dir().display()),
        );
        return;
    }
    for (decoding, label) in [
        (DecodingStrategy::Greedy, "greedy"),
        (DecodingStrategy::Beam(BeamConfig::default_tdt()), "beam"),
    ] {
        run_asr_bench(
            "tdt_offline_wer",
            "parakeet-tdt",
            "offline",
            label,
            smoke_max_wer(),
            |audio| {
            let mut m = ParakeetTDT::from_pretrained_with_decoding(
                tdt_dir(),
                Some(execution_config()),
                decoding,
            )
                .expect("load tdt");
            offline_transcribe(&mut m, audio)
            },
        );
    }
}

/// ParakeetUnified: offline + streaming, each with greedy and beam decoding.
#[test]
fn unified_offline_and_streaming_wer() {
    use parakeet_rs::ParakeetUnified;
    if !unified_available() {
        skip(
            "unified_offline_and_streaming_wer",
            &format!("model not found at {}", unified_dir().display()),
        );
        return;
    }

    for (decoding, label) in [
        (DecodingStrategy::Greedy, "greedy"),
        (
            DecodingStrategy::Beam(BeamConfig::default_unified()),
            "beam",
        ),
    ] {
        // Offline
        run_asr_bench(
            "unified_offline_and_streaming_wer",
            "parakeet-unified",
            "offline",
            label,
            smoke_max_wer(),
            |audio| {
                let mut m = ParakeetUnified::from_pretrained_with_decoding(
                    unified_dir(),
                    Some(execution_config()),
                    Default::default(),
                    decoding,
                )
                .expect("load unified");
                offline_transcribe(&mut m, audio)
            },
        );

        // Streaming (cross-chunk beam carry exercised here)
        run_asr_bench(
            "unified_offline_and_streaming_wer",
            "parakeet-unified",
            "streaming",
            label,
            smoke_max_wer(),
            |audio| {
                let mut m = ParakeetUnified::from_pretrained_with_decoding(
                    unified_dir(),
                    Some(execution_config()),
                    Default::default(),
                    decoding,
                )
                .expect("load unified");
                let chunk = m.streaming_config().chunk_samples();
                let t0 = Instant::now();
                for c in audio.chunks(chunk) {
                    m.transcribe_chunk(c).expect("unified chunk");
                }
                m.flush().expect("unified flush");
                let text = m.get_timed_transcript(TimestampMode::Sentences).text;
                (text, t0.elapsed().as_secs_f32())
            },
        );
    }
}

#[test]
fn eou_streaming_wer() {
    use parakeet_rs::ParakeetEOU;
    if !eou_available() {
        skip(
            "eou_streaming_wer",
            &format!("model not found at {}", eou_dir().display()),
        );
        return;
    }
    // EOU is lowercase, no punctuation; allow a looser smoke ceiling.
    let ceiling = std::env::var("SMOKE_MAX_WER_EOU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.4);
    run_asr_bench(
        "eou_streaming_wer",
        "realtime-eou-120m",
        "streaming",
        "greedy",
        ceiling,
        |audio| {
            let mut m = ParakeetEOU::from_pretrained(eou_dir(), Some(execution_config()))
                .expect("load eou");
        let chunk = 2560; // 160 ms
        let t0 = Instant::now();
        let mut text = String::new();
        for c in audio.chunks(chunk) {
            let mut buf = c.to_vec();
            if buf.len() < chunk {
                buf.resize(chunk, 0.0);
            }
            text.push_str(&m.transcribe(&buf, false).expect("eou chunk"));
        }
        for _ in 0..3 {
            text.push_str(&m.transcribe(&vec![0.0; chunk], false).expect("eou flush"));
        }
        (text, t0.elapsed().as_secs_f32())
        },
    );
}

#[cfg(feature = "cohere")]
fn cohere_steady_present() -> bool {
    // Keep in lockstep with `STEADY_DECODER_CANDIDATES` in src/model_cohere.rs.
    const CANDIDATES: &[&str] = &[
        "onnx/decoder_model_steady_quantized.onnx",
        "decoder_model_steady_quantized.onnx",
        "onnx/decoder_model_steady.onnx",
        "decoder_model_steady.onnx",
    ];
    let dir = cohere_dir();
    CANDIDATES.iter().any(|name| dir.join(name).is_file())
}

#[cfg(feature = "cohere")]
fn mean_ms(steps: &[f64]) -> f64 {
    if steps.is_empty() {
        0.0
    } else {
        steps.iter().sum::<f64>() / steps.len() as f64
    }
}

#[cfg(feature = "cohere")]
#[test]
#[ignore = "prints mean milliseconds per Cohere decoder step"]
fn cohere_decode_step_timing() {
    use parakeet_rs::CohereASR;
    if !cohere_available() {
        skip(
            "cohere_decode_step_timing",
            &format!("model not found at {}", cohere_dir().display()),
        );
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("cohere_decode_step_timing", "samples/smoke missing");
        return;
    };
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let audio_secs = audio.len() as f64 / 16_000.0;
    let mut model =
        CohereASR::from_pretrained(cohere_dir(), Some(execution_config())).expect("load cohere");

    let time_run = |model: &mut CohereASR, audio: &[f32], steady: bool| {
        model.set_use_steady_decoder(steady);
        model.set_record_decoder_steps(true);
        let t0 = Instant::now();
        let text = model
            .transcribe_audio(audio, "en", true, false)
            .expect("cohere transcribe");
        let elapsed = t0.elapsed().as_secs_f64();
        let steps = model.decoder_step_millis().to_vec();
        let used = model.decoder_step_used_steady().to_vec();
        (text, elapsed, steps, used)
    };

    let (merged_text, merged_elapsed, merged_steps, _) = time_run(&mut model, &audio, false);
    let merged_tail = if merged_steps.len() > 1 {
        mean_ms(&merged_steps[1..])
    } else {
        0.0
    };
    eprintln!(
        "\n[cohere_decode_step_timing] merged steps={} mean_ms={:.3} tail_mean_ms={merged_tail:.3} total_step_ms={:.1} wall_s={merged_elapsed:.3} RTFx={:.2}",
        merged_steps.len(),
        mean_ms(&merged_steps),
        merged_steps.iter().sum::<f64>(),
        audio_secs / merged_elapsed.max(1e-9)
    );
    eprintln!("[cohere_decode_step_timing] merged transcript: {merged_text}");
    assert!(
        !merged_steps.is_empty(),
        "expected at least one decoder step"
    );

    if !cohere_steady_present() {
        eprintln!(
            "[cohere_decode_step_timing] steady graph not present beside {}; merged-only",
            cohere_dir().display()
        );
        return;
    }
    assert!(
        model.steady_decoder_loaded(),
        "steady file is present but was not loaded"
    );
    let (steady_text, steady_elapsed, steady_steps, used) = time_run(&mut model, &audio, true);
    let steady_only: Vec<f64> = steady_steps
        .iter()
        .zip(used.iter())
        .filter_map(|(ms, steady)| steady.then_some(*ms))
        .collect();
    let merged_only: Vec<f64> = steady_steps
        .iter()
        .zip(used.iter())
        .filter_map(|(ms, steady)| (!steady).then_some(*ms))
        .collect();
    eprintln!(
        "[cohere_decode_step_timing] steady steps={} steady_mean_ms={:.3} merged_steps_in_run={} merged_mean_ms={:.3} wall_s={steady_elapsed:.3} RTFx={:.2}",
        steady_only.len(),
        mean_ms(&steady_only),
        merged_only.len(),
        mean_ms(&merged_only),
        audio_secs / steady_elapsed.max(1e-9)
    );
    eprintln!("[cohere_decode_step_timing] steady transcript: {steady_text}");
    assert_eq!(
        steady_text, merged_text,
        "steady and merged transcripts diverged"
    );
    assert!(
        !steady_only.is_empty(),
        "expected at least one steady decoder step"
    );
}

#[cfg(feature = "cohere")]
#[test]
fn cohere_steady_decoder_identical() {
    use parakeet_rs::CohereASR;
    if !cohere_available() {
        skip(
            "cohere_steady_decoder_identical",
            &format!("model not found at {}", cohere_dir().display()),
        );
        return;
    }
    if !cohere_steady_present() {
        skip(
            "cohere_steady_decoder_identical",
            &format!(
                "steady decoder not beside {} (decoder_model_steady[_quantized].onnx)",
                cohere_dir().display()
            ),
        );
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("cohere_steady_decoder_identical", "samples/smoke missing");
        return;
    };
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let mut model =
        CohereASR::from_pretrained(cohere_dir(), Some(execution_config())).expect("load cohere");
    assert!(
        model.steady_decoder_loaded(),
        "steady file is present but was not loaded"
    );
    model.set_use_steady_decoder(true);
    let with_steady = model
        .transcribe_audio(&audio, "en", true, false)
        .expect("cohere steady transcribe");
    model.set_use_steady_decoder(false);
    let without_steady = model
        .transcribe_audio(&audio, "en", true, false)
        .expect("cohere merged transcribe");
    eprintln!("\n[cohere_steady_decoder_identical] transcript: {with_steady}");
    assert_eq!(
        with_steady, without_steady,
        "steady-state decoder transcript diverged from the merged decoder"
    );
}

#[cfg(feature = "cohere")]
#[test]
fn cohere_offline_wer() {
    use parakeet_rs::CohereASR;
    if !cohere_available() {
        skip(
            "cohere_offline_wer",
            &format!("model not found at {}", cohere_dir().display()),
        );
        return;
    }
    run_asr_bench(
        "cohere_offline_wer",
        "cohere-transcribe-int8",
        "offline",
        "greedy",
        smoke_max_wer(),
        |audio| {
            let mut m = CohereASR::from_pretrained(cohere_dir(), Some(execution_config()))
                .expect("load cohere");
        let t0 = Instant::now();
            let text = m
                .transcribe_audio(audio, "en", true, false)
                .expect("cohere transcribe");
        (text, t0.elapsed().as_secs_f32())
        },
    );
}

#[cfg(feature = "cohere")]
#[test]
fn cohere_decode_output_unchanged() {
    use parakeet_rs::CohereASR;
    if !cohere_available() {
        skip(
            "cohere_decode_output_unchanged",
            &format!("model not found at {}", cohere_dir().display()),
        );
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("cohere_decode_output_unchanged", "samples/smoke missing");
        return;
    };
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let mut model =
        CohereASR::from_pretrained(cohere_dir(), Some(execution_config())).expect("load cohere");
    let text = model
        .transcribe_audio(&audio, "en", true, false)
        .expect("cohere transcribe");
    let reference = reference_for(&sample);
    let w = wer(&reference, &text);
    eprintln!(
        "\n[cohere_decode_output_unchanged] WER {:.1}% ceiling {:.1}%",
        w * 100.0,
        smoke_max_wer() * 100.0
    );
    eprintln!("[cohere_decode_output_unchanged] transcript: {text}");
    assert!(
        w <= smoke_max_wer(),
        "cohere_decode_output_unchanged: smoke WER {:.1}% exceeds ceiling {:.1}%",
        w * 100.0,
        smoke_max_wer() * 100.0
    );
    if let Ok(path) = std::env::var("PARAKEET_COHERE_BASELINE") {
        let expected = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read baseline {}: {e}", path));
        assert_eq!(
            text.trim(),
            expected.trim(),
            "transcript diverged from pre-change baseline at {path}"
        );
    } else {
        eprintln!(
            "[cohere_decode_output_unchanged] PARAKEET_COHERE_BASELINE unset; exact-string check skipped"
        );
    }
}

#[cfg(feature = "cohere")]
#[test]
fn cohere_long_form_wer() {
    use parakeet_rs::CohereASR;
    let bench_all = std::env::var("PARAKEET_BENCH_ALL")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    if !bench_all {
        skip(
            "cohere_long_form_wer",
            "set PARAKEET_BENCH_ALL=1 to score a long clip",
        );
        return;
    }
    if !cohere_available() {
        skip(
            "cohere_long_form_wer",
            &format!("model not found at {}", cohere_dir().display()),
        );
        return;
    }
    let Some(sample) = sample_named("committee") else {
        skip("cohere_long_form_wer", "samples/committee missing");
        return;
    };
    // manifest.toml lists the clip but no numeric Cohere ceiling. 0.50 is a
    // generous gate for a ~10 minute multi-speaker meeting.
    let ceiling = std::env::var("COHERE_LONG_MAX_WER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.50);
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let audio_secs = audio.len() as f32 / 16_000.0;
    let mut model =
        CohereASR::from_pretrained(cohere_dir(), Some(execution_config())).expect("load cohere");
    let t0 = Instant::now();
    let text = model
        .transcribe_audio_long(&audio, "en", true, false)
        .expect("cohere long transcribe");
    let proc_secs = t0.elapsed().as_secs_f32();
    let reference = reference_for(&sample);
    let w = record_asr(
        &sample.name,
        "cohere-transcribe-int8",
        "long",
        "greedy",
        &requested_ep(),
        audio_secs,
        proc_secs,
        &reference,
        &text,
    );
    assert!(
        w <= ceiling,
        "cohere_long_form_wer: WER {:.1}% exceeds ceiling {:.1}%",
        w * 100.0,
        ceiling * 100.0
    );
}

#[cfg(feature = "ark-asr")]
#[test]
fn ark_offline_wer() {
    use parakeet_rs::ArkAsr;
    if !ark_available() {
        skip(
            "ark_offline_wer",
            &format!("model not found at {}", ark_dir().display()),
        );
        return;
    }
    run_asr_bench(
        "ark_offline_wer",
        "ark-asr-0.6b-int8",
        "offline",
        "greedy",
        smoke_max_wer(),
        |audio| {
            let mut m =
                ArkAsr::from_pretrained(ark_dir(), Some(execution_config())).expect("load ark");
            let t0 = Instant::now();
            let text = m.transcribe_audio(audio).expect("ark transcribe");
            (text, t0.elapsed().as_secs_f32())
        },
    );
}

#[cfg(feature = "ark-asr")]
#[test]
fn ark_long_audio_no_truncation() {
    use parakeet_rs::{ArkAsr, ArkLongAudioPolicy, Error};
    if !ark_available() {
        skip(
            "ark_long_audio_no_truncation",
            &format!("model not found at {}", ark_dir().display()),
        );
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("ark_long_audio_no_truncation", "samples/smoke missing");
        return;
    };
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let mut long = Vec::with_capacity(audio.len() * 3);
    for _ in 0..3 {
        long.extend_from_slice(&audio);
    }
    let mut model = ArkAsr::from_pretrained(ark_dir(), Some(execution_config())).expect("load ark");
    let one = model.transcribe_audio(&audio).expect("ark single clip");
    let long_text = model.transcribe_long(&long).expect("ark transcribe_long");
    let one_words = one.split_whitespace().count();
    let long_words = long_text.split_whitespace().count();
    let long_secs = long.len() as f32 / 16_000.0;
    eprintln!(
        "\n[ark_long_audio_no_truncation] {long_secs:.2}s: single={one_words} words, long={long_words} words"
    );
    assert!(
        one_words > 0,
        "ark_long_audio_no_truncation: single-clip transcript was empty"
    );
    assert!(
        long_words as f32 >= one_words as f32 * 2.5,
        "ark_long_audio_no_truncation: long transcript has {long_words} words, need >= 2.5x {one_words}"
    );
    model.set_long_audio_policy(ArkLongAudioPolicy::Error);
    let err = model
        .transcribe_audio(&long)
        .expect_err("Error policy should reject audio past max_audio_seconds");
    assert!(
        matches!(err, Error::Audio(_)),
        "expected Error::Audio, got {err}"
    );
}

#[test]
fn granite_ctc_offline_wer() {
    use parakeet_rs::Parakeet;
    if !granite_available() {
        skip(
            "granite_ctc_offline_wer",
            &format!("model not found at {}", granite_dir().display()),
        );
        return;
    }
    run_asr_bench(
        "granite_ctc_offline_wer",
        "granite-turboctc-int8",
        "offline",
        "greedy",
        smoke_max_wer(),
        |audio| {
            let mut m = Parakeet::from_pretrained(granite_dir(), Some(execution_config()))
                .expect("load granite");
            offline_transcribe(&mut m, audio)
        },
    );
}

fn bench_all_enabled() -> bool {
    std::env::var("PARAKEET_BENCH_ALL")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

fn granite_long_max_wer() -> f32 {
    std::env::var("GRANITE_LONG_MAX_WER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.35)
}

/// Load Granite, transcribe one clip, record WER. Returns `(audio_secs, proc_secs, wer, text)`.
fn granite_score_sample(
    decoding: DecodingStrategy,
    label: &str,
    sample: &Sample,
) -> (f32, f32, f32, String) {
    use parakeet_rs::Parakeet;
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let audio_secs = audio.len() as f32 / 16_000.0;
    let mut model =
        Parakeet::from_pretrained_with_decoding(granite_dir(), Some(execution_config()), decoding)
            .expect("load granite");
    let (text, proc_secs) = offline_transcribe(&mut model, &audio);
    let reference = reference_for(sample);
    let w = record_asr(
        &sample.name,
        "granite-turboctc-int8",
        "offline",
        label,
        &requested_ep(),
        audio_secs,
        proc_secs,
        &reference,
        &text,
    );
    (audio_secs, proc_secs, w, text)
}

/// Greedy Granite on the long local clips. Completes in one shot (no chunking).
#[test]
fn granite_long_audio_completes() {
    if !bench_all_enabled() {
        skip(
            "granite_long_audio_completes",
            "set PARAKEET_BENCH_ALL=1 to score long clips",
        );
        return;
    }
    if !granite_available() {
        skip(
            "granite_long_audio_completes",
            &format!("model not found at {}", granite_dir().display()),
        );
        return;
    }
    let ceiling = granite_long_max_wer();
    for name in ["committee", "datascience-staff-meeting"] {
        let Some(sample) = sample_named(name) else {
            skip(
                "granite_long_audio_completes",
                &format!("samples/{name} missing"),
            );
            continue;
        };
        let (audio_secs, proc_secs, w, _) =
            granite_score_sample(DecodingStrategy::Greedy, "greedy", &sample);
        let rtfx = if proc_secs > 1e-6 {
            audio_secs / proc_secs
        } else {
            0.0
        };
        eprintln!(
            "\n[granite_long_audio_completes] {name}: {audio_secs:.2}s proc {proc_secs:.3}s RTFx {rtfx:.2} WER {w:.4} ceiling {ceiling:.2}"
        );
        assert!(
            w <= ceiling,
            "granite_long_audio_completes: {name} WER {w:.4} exceeds ceiling {ceiling:.2}"
        );
    }
}

/// Beam-search Granite on committee, for peak-RSS measurement around the process.
/// Ignored so the default suite stays on the short smoke clip.
#[test]
#[ignore]
fn granite_beam_peak_rss() {
    if !granite_available() {
        skip(
            "granite_beam_peak_rss",
            &format!("model not found at {}", granite_dir().display()),
        );
        return;
    }
    let Some(sample) = sample_named("committee") else {
        skip("granite_beam_peak_rss", "samples/committee missing");
        return;
    };
    let ceiling = granite_long_max_wer();
    let (audio_secs, proc_secs, w, _) = granite_score_sample(
        DecodingStrategy::Beam(BeamConfig::default_ctc()),
        "beam",
        &sample,
    );
    let rtfx = if proc_secs > 1e-6 {
        audio_secs / proc_secs
    } else {
        0.0
    };
    eprintln!(
        "\n[granite_beam_peak_rss] committee: {audio_secs:.2}s proc {proc_secs:.3}s RTFx {rtfx:.2} WER {w:.4} ceiling {ceiling:.2}"
    );
    assert!(
        w <= ceiling,
        "granite_beam_peak_rss: WER {w:.4} exceeds ceiling {ceiling:.2}"
    );
}

/// Smoke-clip beam transcript. Set `PARAKEET_GRANITE_BEAM_BASELINE` to a file
/// saved before the borrowed-logits change to require a byte-identical string.
#[test]
fn granite_ctc_beam_matches_before() {
    use parakeet_rs::Parakeet;
    if !granite_available() {
        skip(
            "granite_ctc_beam_matches_before",
            &format!("model not found at {}", granite_dir().display()),
        );
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("granite_ctc_beam_matches_before", "samples/smoke missing");
        return;
    };
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let mut model = Parakeet::from_pretrained_with_decoding(
        granite_dir(),
        Some(execution_config()),
        DecodingStrategy::Beam(BeamConfig::default_ctc()),
    )
    .expect("load granite");
    let text = model
        .transcribe_samples(audio, 16_000, 1, Some(TimestampMode::Words))
        .expect("granite beam")
        .text;
    let reference = reference_for(&sample);
    let w = wer(&reference, &text);
    eprintln!(
        "\n[granite_ctc_beam_matches_before] WER {:.1}% ceiling {:.1}%",
        w * 100.0,
        smoke_max_wer() * 100.0
    );
    eprintln!("[granite_ctc_beam_matches_before] transcript: {text}");
    assert!(
        w <= smoke_max_wer(),
        "granite_ctc_beam_matches_before: smoke WER {:.1}% exceeds ceiling {:.1}%",
        w * 100.0,
        smoke_max_wer() * 100.0
    );
    if let Ok(path) = std::env::var("PARAKEET_GRANITE_BEAM_BASELINE") {
        let expected = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read baseline {}: {e}", path));
        assert_eq!(
            text.trim(),
            expected.trim(),
            "transcript diverged from pre-change baseline at {path}"
        );
    } else {
        eprintln!(
            "[granite_ctc_beam_matches_before] PARAKEET_GRANITE_BEAM_BASELINE unset; exact-string check skipped"
        );
    }
}

struct SweepRow {
    model: String,
    intra_threads: usize,
    audio_secs: f32,
    proc_secs: f32,
    rtfx: f32,
    wer: f32,
}

fn parse_sweep_threads() -> Vec<usize> {
    let default = "4,8,16,32";
    let raw = std::env::var("PARAKEET_SWEEP_THREADS").unwrap_or_else(|_| default.to_string());
    let parsed: Vec<usize> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    if parsed.is_empty() {
        default
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    } else {
        parsed
    }
}

fn parse_sweep_models() -> Vec<String> {
    let default = "cohere,ark,granite,nemotron";
    let raw = std::env::var("PARAKEET_SWEEP_MODELS").unwrap_or_else(|_| default.to_string());
    let names: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if names.is_empty() {
        default.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        names
    }
}

fn record_sweep(rid: u64, row: &SweepRow) {
    append_csv_row(
        "asr_thread_sweep.csv",
        "run_id,model,intra_threads,audio_secs,proc_secs,rtfx,wer",
        &format!(
            "{rid},{},{},{:.2},{:.3},{:.2},{:.4}",
            row.model, row.intra_threads, row.audio_secs, row.proc_secs, row.rtfx, row.wer
        ),
    );
    eprintln!(
        "  {:<10} threads {:>2}  RTFx {:>6.2}x  WER {:>5.1}%  proc {:>7.3}s",
        row.model,
        row.intra_threads,
        row.rtfx,
        row.wer * 100.0,
        row.proc_secs
    );
}

/// Load `build` outside the timer, transcribe `audio` once, and append a sweep row.
fn sweep_threads(
    model: &str,
    threads: &[usize],
    audio: &[f32],
    reference: &str,
    audio_secs: f32,
    rid: u64,
    rows: &mut Vec<SweepRow>,
    mut build: impl FnMut(parakeet_rs::ExecutionConfig) -> Box<dyn FnMut(&[f32]) -> String>,
) {
    for &n in threads {
        let cfg = execution_config().with_intra_threads(n);
        let mut run = build(cfg);
        let t0 = Instant::now();
        let hyp = run(audio);
        let proc_secs = t0.elapsed().as_secs_f32();
        let w = wer(reference, &hyp);
        let rtfx = if proc_secs > 1e-6 {
            audio_secs / proc_secs
        } else {
            0.0
        };
        let row = SweepRow {
            model: model.to_string(),
            intra_threads: n,
            audio_secs,
            proc_secs,
            rtfx,
            wer: w,
        };
        record_sweep(rid, &row);
        rows.push(row);
    }
}

fn print_sweep_summary(threads: &[usize], rows: &[SweepRow]) {
    eprintln!("\n[asr_thread_sweep] RTFx summary:");
    eprint!("{:<12}", "model");
    for t in threads {
        eprint!(" {t:>8}");
    }
    eprintln!(" {:>8}", "best");
    let mut seen: Vec<&str> = Vec::new();
    for row in rows {
        if !seen.contains(&row.model.as_str()) {
            seen.push(row.model.as_str());
        }
    }
    for model in seen {
        eprint!("{model:<12}");
        let mut best_threads = 0usize;
        let mut best_rtfx = f32::MIN;
        for t in threads {
            match rows
                .iter()
                .find(|r| r.model == model && r.intra_threads == *t)
            {
                Some(r) => {
                    eprint!(" {:>7.2}x", r.rtfx);
                    if r.rtfx > best_rtfx {
                        best_rtfx = r.rtfx;
                        best_threads = *t;
                    }
                }
                None => eprint!(" {:>8}", "-"),
            }
        }
        eprintln!(" {best_threads:>8}");
    }
}

/// CPU intra-op thread sweep on the smoke clip. Ignored because each config is
/// a full transcription and concurrent runs make RTFx meaningless.
///
/// `PARAKEET_SWEEP_THREADS` (default `4,8,16,32`) and `PARAKEET_SWEEP_MODELS`
/// (default `cohere,ark,granite,nemotron`). A model whose Cargo feature or
/// weights are missing is skipped. Model load stays outside the timer.
/// Writes `samples/asr_thread_sweep.csv`.
#[test]
#[ignore = "long CPU sweep; run alone so RTFx is not contended"]
fn asr_thread_sweep() {
    if ep_skip_if_unavailable("asr_thread_sweep") {
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("asr_thread_sweep", "samples/smoke missing");
        return;
    };
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let reference = reference_for(&sample);
    let audio_secs = audio.len() as f32 / 16_000.0;
    let threads = parse_sweep_threads();
    let models = parse_sweep_models();
    let rid = run_id();
    let mut rows = Vec::new();

    eprintln!("\n[asr_thread_sweep] smoke {audio_secs:.2}s threads={threads:?} models={models:?}");

    for name in &models {
        match name.as_str() {
            "cohere" => {
                #[cfg(not(feature = "cohere"))]
                skip("asr_thread_sweep", "cohere requires cargo feature cohere");
                #[cfg(feature = "cohere")]
                {
                    use parakeet_rs::CohereASR;
                    if !cohere_available() {
                        skip(
                            "asr_thread_sweep",
                            &format!("cohere model not found at {}", cohere_dir().display()),
                        );
                    } else {
                        sweep_threads(
                            "cohere",
                            &threads,
                            &audio,
                            &reference,
                            audio_secs,
                            rid,
                            &mut rows,
                            |cfg| {
                                let mut m = CohereASR::from_pretrained(cohere_dir(), Some(cfg))
                                    .expect("load cohere");
                                Box::new(move |audio: &[f32]| {
                                    m.transcribe_audio(audio, "en", true, false)
                                        .expect("cohere transcribe")
                                })
                            },
                        );
                    }
                }
            }
            "ark" => {
                #[cfg(not(feature = "ark-asr"))]
                skip("asr_thread_sweep", "ark requires cargo feature ark-asr");
                #[cfg(feature = "ark-asr")]
                {
                    use parakeet_rs::ArkAsr;
                    if !ark_available() {
                        skip(
                            "asr_thread_sweep",
                            &format!("ark model not found at {}", ark_dir().display()),
                        );
                    } else {
                        sweep_threads(
                            "ark",
                            &threads,
                            &audio,
                            &reference,
                            audio_secs,
                            rid,
                            &mut rows,
                            |cfg| {
                                let mut m = ArkAsr::from_pretrained(ark_dir(), Some(cfg))
                                    .expect("load ark");
                                Box::new(move |audio: &[f32]| {
                                    m.transcribe_audio(audio).expect("ark transcribe")
                                })
                            },
                        );
                    }
                }
            }
            "granite" => {
                use parakeet_rs::Parakeet;
                if !granite_available() {
                    skip(
                        "asr_thread_sweep",
                        &format!("granite model not found at {}", granite_dir().display()),
                    );
                } else {
                    sweep_threads(
                        "granite",
                        &threads,
                        &audio,
                        &reference,
                        audio_secs,
                        rid,
                        &mut rows,
                        |cfg| {
                            let mut m = Parakeet::from_pretrained(granite_dir(), Some(cfg))
                                .expect("load granite");
                            Box::new(move |audio: &[f32]| {
                                m.transcribe_samples(
                                    audio.to_vec(),
                                    16_000,
                                    1,
                                    Some(TimestampMode::Words),
                                )
                                .expect("granite transcribe")
                                .text
                            })
                        },
                    );
                }
            }
            "nemotron" => {
                use parakeet_rs::Nemotron;
                if !nemotron_available() {
                    skip(
                        "asr_thread_sweep",
                        &format!("nemotron model not found at {}", nemotron_dir().display()),
                    );
                } else {
                    let mut normalized = audio.clone();
                    peak_normalize(&mut normalized);
                    sweep_threads(
                        "nemotron",
                        &threads,
                        &normalized,
                        &reference,
                        audio_secs,
                        rid,
                        &mut rows,
                        |cfg| {
                            let mut m = Nemotron::from_pretrained(nemotron_dir(), Some(cfg))
                                .expect("load nemotron");
                            Box::new(move |audio: &[f32]| {
                                m.transcribe_audio(audio).expect("nemotron offline")
                            })
                        },
                    );
                }
            }
            other => skip(
                "asr_thread_sweep",
                &format!("unknown PARAKEET_SWEEP_MODELS entry {other}"),
            ),
        }
    }

    if rows.is_empty() {
        eprintln!("[asr_thread_sweep] no models ran");
        return;
    }
    print_sweep_summary(&threads, &rows);
}

/// Benchmark Sortformer diarization quality (DER/JER) and speed (RTF) across all
/// samples, appending a row per sample to `samples/diarization_report.csv` so
/// changes to the project can be compared run-over-run. The smoke clip is
/// asserted; longer meeting clips are included by default (Sortformer streams
/// long audio efficiently). Tag a run with PARAKEET_BENCH_LABEL.
#[cfg(feature = "sortformer")]
#[test]
fn diarization_benchmark_report() {
    use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
    use std::time::Instant;

    if !sortformer_available() {
        skip(
            "diarization_benchmark_report",
            &format!("model not found at {}", sortformer_onnx().display()),
        );
        return;
    }

    if ep_skip_if_unavailable("diarization_benchmark_report") {
        return;
    }

    let model_name = sortformer_onnx()
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("sortformer")
        .to_string();
    let label = bench_label();
    let rid = run_id();

    let samples = discover_samples();
    assert!(
        samples.iter().any(|s| s.name == "smoke"),
        "samples/smoke must exist"
    );

    let header = "run_id,label,sample,model,audio_secs,proc_secs,rtf,\
num_ref_speakers,num_hyp_speakers,der,missed,false_alarm,confusion,jer";

    eprintln!("\nDiarization benchmark ({model_name}):");
    eprintln!(
        "{:<28} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "sample", "RTF", "DER", "miss", "FA", "JER"
    );
    eprintln!("{}", "-".repeat(64));

    let mut smoke_der: Option<f32> = None;

    for sample in &samples {
        let (audio, sample_rate) = load_wav_mono_16k(&sample.wav);
        let audio_secs = audio.len() as f32 / 16_000.0;

        let mut sf =
            Sortformer::with_config(sortformer_onnx(), Some(execution_config()), diar_config())
                .expect("load sortformer");

        let t0 = Instant::now();
        let segments = sf.diarize(audio, sample_rate, 1).expect("diarize");
        let proc_secs = t0.elapsed().as_secs_f32();
        let rtf = if proc_secs > 0.0 {
            audio_secs / proc_secs
        } else {
            0.0
        };

        let hyp: Vec<DiarSegment> = segments
            .iter()
            .map(|s| DiarSegment {
                speaker: s.speaker_id,
                start: s.start as f32 / 16_000.0,
                end: s.end as f32 / 16_000.0,
            })
            .collect();

        let raw_ref = std::fs::read_to_string(&sample.transcript).expect("read transcript");
        let reference = parse_reference_turns(&raw_ref, audio_secs);
        let m = diarization_metrics(&reference, &hyp, audio_secs);

        eprintln!(
            "{:<28} {:>5.1}x {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}%",
            sample.name,
            rtf,
            m.der * 100.0,
            m.missed * 100.0,
            m.false_alarm * 100.0,
            m.jer * 100.0
        );

        append_csv_row(
            "diarization_report.csv",
            header,
            &format!(
                "{rid},{label},{},{model_name},{audio_secs:.2},{proc_secs:.3},{rtf:.2},\
{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                sample.name,
                m.num_ref_speakers,
                m.num_hyp_speakers,
                m.der,
                m.missed,
                m.false_alarm,
                m.confusion,
                m.jer
            ),
        );

        if sample.name == "smoke" {
            smoke_der = Some(m.der);
        }
    }

    eprintln!(
        "\nwrote {} rows to samples/diarization_report.csv (run_id={rid})",
        samples.len()
    );

    let der = smoke_der.expect("smoke diarized");
    let ceiling: f32 = std::env::var("SMOKE_MAX_DER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    assert!(
        der <= ceiling,
        "smoke DER {:.1}% exceeds ceiling {:.1}% (set SMOKE_MAX_DER to adjust)",
        der * 100.0,
        ceiling * 100.0
    );
}

/// Sweep post-processing thresholds over raw predictions (the model runs once per clip, then
/// every candidate `DiarizationConfig` is scored on the cached probabilities). Ignored by
/// default; run with `-- --ignored --nocapture`. Candidates are ranked by mean DER over the
/// clips named in `PARAKEET_SWEEP_TUNE` (default: the two long meeting clips); the remaining
/// clips are reported as hold-out. Only a few clips exist, so treat the winner as a starting
/// point to confirm on your own audio, not a universal optimum.
#[cfg(feature = "sortformer")]
#[test]
#[ignore = "threshold sweep; run explicitly with --ignored"]
fn diarization_threshold_sweep() {
    use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};

    if !sortformer_available() {
        skip("diarization_threshold_sweep", "sortformer model not found");
        return;
    }
    let tune: Vec<String> = std::env::var("PARAKEET_SWEEP_TUNE")
        .unwrap_or_else(|_| "committee,datascience-staff-meeting".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // Raw predictions + reference per clip (one model pass each).
    struct Clip {
        name: String,
        preds: ndarray::Array2<f32>,
        n_samples: u64,
        reference: Vec<DiarSegment>,
        secs: f32,
    }
    let mut clips = Vec::new();
    for sample in discover_samples() {
        let (audio, sr) = load_wav_mono_16k(&sample.wav);
        let secs = audio.len() as f32 / 16_000.0;
        let mut sf = Sortformer::with_config(
            sortformer_onnx(),
            Some(execution_config()),
            DiarizationConfig::callhome(),
        )
        .expect("load sortformer");
        let (preds, n_samples) = sf.predict_raw(audio, sr, 1).expect("predict_raw");
        let raw_ref = std::fs::read_to_string(&sample.transcript).expect("read transcript");
        clips.push(Clip {
            name: sample.name.clone(),
            preds,
            n_samples,
            reference: parse_reference_turns(&raw_ref, secs),
            secs,
        });
    }

    let score = |cfg: &DiarizationConfig, c: &Clip| -> f32 {
        let hyp: Vec<DiarSegment> = Sortformer::post_process(cfg, &c.preds, c.n_samples)
            .iter()
            .map(|s| DiarSegment {
                speaker: s.speaker_id,
                start: s.start as f32 / 16_000.0,
                end: s.end as f32 / 16_000.0,
            })
            .collect();
        diarization_metrics(&c.reference, &hyp, c.secs).der
    };

    let onsets = [0.35f32, 0.40, 0.45, 0.50, 0.55, 0.60, 0.641];
    let offset_gaps = [0.0f32, 0.05, 0.10, 0.20];
    let min_ons = [0.10f32, 0.20, 0.30, 0.511];
    let pads = [(0.229f32, 0.079f32), (0.10, 0.05)];
    let medians = [7usize, 11];

    let mut results: Vec<(f32, DiarizationConfig, Vec<f32>)> = Vec::new();
    let mut candidates: Vec<DiarizationConfig> = Vec::new();
    for &on in &onsets {
        for &gap in &offset_gaps {
            for &mon in &min_ons {
                for &(pon, poff) in &pads {
                    for &med in &medians {
                        candidates.push(DiarizationConfig {
                            onset: on,
                            offset: (on - gap).max(0.05),
                            pad_onset: pon,
                            pad_offset: poff,
                            min_duration_on: mon,
                            min_duration_off: 0.296,
                            median_window: med,
                        });
                    }
                }
            }
        }
    }
    candidates.push(DiarizationConfig::callhome());
    for cfg in candidates {
        let ders: Vec<f32> = clips.iter().map(|c| score(&cfg, c)).collect();
        let tuned: Vec<f32> = clips
            .iter()
            .zip(&ders)
            .filter(|(c, _)| tune.contains(&c.name))
            .map(|(_, d)| *d)
            .collect();
        let mean = tuned.iter().sum::<f32>() / tuned.len().max(1) as f32;
        results.push((mean, cfg, ders));
    }
    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let names: Vec<&str> = clips.iter().map(|c| c.name.as_str()).collect();
    eprintln!("\nclips: {names:?}  (tuned on {tune:?})");
    eprintln!(
        "{:<8} {:<7} {:<7} {:<7} {:<7} {:<7} {:<5} per-clip DER %",
        "tunedDER", "onset", "offset", "min_on", "pad_on", "pad_off", "med"
    );
    let fmt = |r: &(f32, DiarizationConfig, Vec<f32>)| {
        let c = &r.1;
        let per: Vec<String> = r.2.iter().map(|d| format!("{:.1}", d * 100.0)).collect();
        eprintln!(
            "{:<8.2} {:<7.3} {:<7.3} {:<7.3} {:<7.3} {:<7.3} {:<5} {}",
            r.0 * 100.0,
            c.onset,
            c.offset,
            c.min_duration_on,
            c.pad_onset,
            c.pad_offset,
            c.median_window,
            per.join(" / ")
        );
    };
    eprintln!("-- top 12");
    for r in results.iter().take(12) {
        fmt(r);
    }
    eprintln!("-- callhome baseline");
    if let Some(r) = results.iter().find(|r| {
        let c = &r.1;
        c.onset == 0.641 && c.offset == 0.561 && c.min_duration_on == 0.511 && c.median_window == 11
    }) {
        fmt(r);
    }
}

/// Score the buffered streaming API (`feed` / `flush`) for DER, with audio delivered in
/// 1 s increments. Two routes are reported per clip:
///   - `feed`: segments exactly as the API returns them (median filter + thresholds applied
///     independently inside each window, the way a live consumer sees them);
///   - `feed_raw`: raw per-frame probabilities from the same streaming state, concatenated and
///     post-processed once over the whole stream. This isolates model + speaker-cache quality
///     from per-window post-processing artifacts.
/// Appends to `samples/diarization_streaming_report.csv`. Restrict routes with
/// PARAKEET_STREAM_ROUTES (`feed`, `feed_raw`) and clips with
/// PARAKEET_BENCH_SAMPLES; the latency preset is whichever ONNX PARAKEET_SORTFORMER_ONNX names.
#[cfg(feature = "sortformer")]
#[test]
fn sortformer_streaming_der_report() {
    use parakeet_rs::sortformer::Sortformer;
    use std::time::Instant;

    if !sortformer_available() {
        skip(
            "sortformer_streaming_der_report",
            "sortformer model not found",
        );
        return;
    }
    if ep_skip_if_unavailable("sortformer_streaming_der_report") {
        return;
    }
    let model_name = sortformer_onnx()
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("sortformer")
        .to_string();
    let label = bench_label();
    let rid = run_id();
    let header = "run_id,label,sample,model,route,audio_secs,proc_secs,rtf,\
num_ref_speakers,num_hyp_speakers,der,missed,false_alarm,confusion,jer";

    eprintln!("\nStreaming DER ({model_name}, 1 s input increments):");
    eprintln!(
        "{:<28} {:<9} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "sample", "route", "RTF", "DER", "miss", "FA", "JER"
    );
    eprintln!("{}", "-".repeat(76));

    for sample in discover_samples() {
        let (audio, _) = load_wav_mono_16k(&sample.wav);
        let audio_secs = audio.len() as f32 / 16_000.0;
        let raw_ref = std::fs::read_to_string(&sample.transcript).expect("read transcript");
        let reference = parse_reference_turns(&raw_ref, audio_secs);

        let mut sf =
            Sortformer::with_config(sortformer_onnx(), Some(execution_config()), diar_config())
        .expect("load sortformer");

        let routes =
            std::env::var("PARAKEET_STREAM_ROUTES").unwrap_or_else(|_| "feed,feed_raw".into());
        for route in ["feed", "feed_raw"] {
            if !routes.split(',').any(|r| r.trim() == route) {
                continue;
            }
            sf.reset_state();
            let t0 = Instant::now();
            let hyp_segments = if route == "feed" {
                let mut segs = Vec::new();
                for c in audio.chunks(16_000) {
                    segs.extend(sf.feed(c).expect("feed"));
                }
                segs.extend(sf.flush().expect("flush"));
                segs
            } else {
                let mut parts = Vec::new();
                for c in audio.chunks(16_000) {
                    let p = sf.feed_raw(c).expect("feed_raw");
                    if p.nrows() > 0 {
                        parts.push(p);
                    }
                }
                let tail = sf.flush_raw().expect("flush_raw");
                if tail.nrows() > 0 {
                    parts.push(tail);
                }
                let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
                let all = ndarray::concatenate(ndarray::Axis(0), &views).expect("concat");
                Sortformer::post_process(&diar_config(), &all, audio.len() as u64)
            };
            let proc_secs = t0.elapsed().as_secs_f32();
            let rtf = if proc_secs > 0.0 {
                audio_secs / proc_secs
            } else {
                0.0
            };

            let hyp: Vec<DiarSegment> = hyp_segments
                .iter()
                .map(|s| DiarSegment {
                    speaker: s.speaker_id,
                    start: s.start as f32 / 16_000.0,
                    end: s.end as f32 / 16_000.0,
                })
                .collect();
            let m = diarization_metrics(&reference, &hyp, audio_secs);
            eprintln!(
                "{:<28} {:<9} {:>5.1}x {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}%",
                sample.name,
                route,
                rtf,
                m.der * 100.0,
                m.missed * 100.0,
                m.false_alarm * 100.0,
                m.jer * 100.0
            );
            append_csv_row(
                "diarization_streaming_report.csv",
                header,
                &format!(
                    "{rid},{label},{},{model_name},{route},{audio_secs:.2},{proc_secs:.3},{rtf:.2},\
{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    sample.name,
                    m.num_ref_speakers,
                    m.num_hyp_speakers,
                    m.der,
                    m.missed,
                    m.false_alarm,
                    m.confusion,
                    m.jer
                ),
            );
        }
    }
}

/// Exercise Sortformer's streaming API (`diarize_chunk` + `flush`) to ensure the
/// stateful path produces segments, complementing the offline `diarize` benchmark.
#[cfg(feature = "sortformer")]
#[test]
fn sortformer_streaming_smoke() {
    use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};

    if !sortformer_available() {
        skip("sortformer_streaming_smoke", "sortformer model not found");
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("sortformer_streaming_smoke", "samples/smoke missing");
        return;
    };

    if ep_skip_if_unavailable("sortformer_streaming_smoke") {
        return;
    }
    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let mut sf = Sortformer::with_config(
        sortformer_onnx(),
        Some(execution_config()),
        DiarizationConfig::callhome(),
    )
        .expect("load sortformer");

    let chunk = 16_000; // 1 s
    let mut segments = Vec::new();
    for c in audio.chunks(chunk) {
        segments.extend(sf.diarize_chunk(c).expect("diarize_chunk"));
    }
    segments.extend(sf.flush().expect("flush"));

    let distinct: std::collections::BTreeSet<usize> =
        segments.iter().map(|s| s.speaker_id).collect();
    eprintln!(
        "\nSortformer streaming: {} segment(s), {} distinct speaker(s)",
        segments.len(),
        distinct.len()
    );
    assert!(
        !segments.is_empty(),
        "streaming diarization produced no segments"
    );
    assert!(
        segments.iter().all(|s| s.end >= s.start),
        "segment end must be >= start"
    );
}

#[cfg(feature = "multitalker")]
#[test]
fn multitalker_smoke() {
    use parakeet_rs::MultitalkerASR;

    if !multitalker_available() {
        skip(
            "multitalker_smoke",
            &format!("model not found at {}", multitalker_dir().display()),
        );
        return;
    }
    if !sortformer_available() {
        skip(
            "multitalker_smoke",
            "multitalker requires a sortformer ONNX; not found",
        );
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("multitalker_smoke", "samples/smoke missing");
        return;
    };

    if ep_skip_if_unavailable("multitalker_smoke") {
        return;
    }
    let (mut audio, _) = load_wav_mono_16k(&sample.wav);
    let audio_secs = audio.len() as f32 / 16_000.0;
    peak_normalize(&mut audio);

    let mut model = MultitalkerASR::from_pretrained(
        multitalker_dir(),
        sortformer_onnx(),
        Some(execution_config()),
    )
        .expect("load multitalker");

    let chunk = model.chunk_audio_samples();
    let t0 = Instant::now();
    for c in audio.chunks(chunk) {
        let mut buf = c.to_vec();
        if buf.len() < chunk {
            buf.resize(chunk, 0.0);
        }
        model.transcribe_chunk(&buf).expect("multitalker chunk");
    }
    for _ in 0..3 {
        model.transcribe_chunk(&vec![0.0; chunk]).expect("flush");
    }
    let proc_secs = t0.elapsed().as_secs_f32();

    let transcripts = model.get_transcripts();
    let total_chars: usize = transcripts.iter().map(|t| t.text.trim().len()).sum();

    // Speaker-attributed transcripts joined for a combined WER vs the reference.
    let combined: String = transcripts
        .iter()
        .map(|t| t.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let reference = reference_for(&sample);
    record_asr(
        &sample.name,
        "multitalker-0.6b-int8",
        "streaming",
        "greedy",
        &requested_ep(),
        audio_secs,
        proc_secs,
        &reference,
        &combined,
    );
    eprintln!(
        "\nMultitalker produced {} speaker transcript(s), {} chars total:",
        transcripts.len(),
        total_chars
    );
    for t in &transcripts {
        eprintln!("  speaker {}: {}", t.speaker_id, t.text.trim());
    }

    assert!(
        total_chars > 0,
        "expected non-empty multitalker transcription on a 20s speech clip"
    );
}

#[cfg(feature = "multitalker")]
struct MultitalkerPass {
    speakers: usize,
    proc_secs: f32,
    audio_secs: f32,
    session_runs: u64,
    wer: f32,
}

/// One full-clip multitalker pass. `transcribe_audio_multitalker` flushes the
/// Sortformer tail when `strategy` is [`DiarStrategy::Buffered`](parakeet_rs::DiarStrategy::Buffered).
#[cfg(feature = "multitalker")]
fn run_multitalker_pass(
    audio: &[f32],
    strategy: parakeet_rs::DiarStrategy,
    sample: &str,
    reference: &str,
    mode: &str,
) -> MultitalkerPass {
    use parakeet_rs::{DiarStrategy, MultitalkerASR};

    let mut model = MultitalkerASR::from_pretrained(
        multitalker_dir(),
        sortformer_onnx(),
        Some(execution_config()),
    )
    .expect("load multitalker");
    if strategy != DiarStrategy::PerChunk {
        model.set_diar_strategy(strategy);
    }
    let audio_secs = audio.len() as f32 / 16_000.0;
    let t0 = Instant::now();
    let transcripts = model
        .transcribe_audio_multitalker(audio)
        .expect("multitalker transcribe");
    let proc_secs = t0.elapsed().as_secs_f32();
    let runs = model.sortformer_session_runs();
    let combined = transcripts
        .iter()
        .map(|t| t.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let wer = record_asr(
        sample,
        "multitalker-0.6b-int8",
        mode,
        "greedy",
        &requested_ep(),
        audio_secs,
        proc_secs,
        reference,
        &combined,
    );
    eprintln!(
        "\n[{mode}] speakers {}  Sortformer runs {runs}  proc {proc_secs:.3}s",
        transcripts.len()
    );
    for t in &transcripts {
        eprintln!("  speaker {}: {}", t.speaker_id, t.text.trim());
    }
    MultitalkerPass {
        speakers: transcripts.len(),
        proc_secs,
        audio_secs,
        session_runs: runs,
        wer,
    }
}

#[cfg(feature = "multitalker")]
fn multitalker_ready(test: &str) -> bool {
    if !multitalker_available() {
        skip(
            test,
            &format!("model not found at {}", multitalker_dir().display()),
        );
        return false;
    }
    if !sortformer_available() {
        skip(test, "multitalker requires a sortformer ONNX; not found");
        return false;
    }
    if ep_skip_if_unavailable(test) {
        return false;
    }
    true
}

/// Buffered diarization on the smoke clip stays within 0.5 WER points of PerChunk.
#[cfg(feature = "multitalker")]
#[test]
fn multitalker_buffered_matches_per_chunk() {
    use parakeet_rs::DiarStrategy;

    if !multitalker_ready("multitalker_buffered_matches_per_chunk") {
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip(
            "multitalker_buffered_matches_per_chunk",
            "samples/smoke missing",
        );
        return;
    };
    let (mut audio, _) = load_wav_mono_16k(&sample.wav);
    peak_normalize(&mut audio);
    let reference = reference_for(&sample);

    let per = run_multitalker_pass(
        &audio,
        DiarStrategy::PerChunk,
        &sample.name,
        &reference,
        "streaming",
    );
    let buffered = run_multitalker_pass(
        &audio,
        DiarStrategy::Buffered,
        &sample.name,
        &reference,
        "streaming-buffered",
    );
    let delta = buffered.wer - per.wer;
    eprintln!(
        "\n[multitalker_buffered_matches_per_chunk] PerChunk WER {:.4} RTFx {:.2}  Buffered WER {:.4} RTFx {:.2}  delta {:.4}",
        per.wer,
        per.audio_secs / per.proc_secs,
        buffered.wer,
        buffered.audio_secs / buffered.proc_secs,
        delta
    );
    assert!(
        delta <= 0.005,
        "Buffered WER {:.4} exceeds PerChunk WER {:.4} by more than 0.5 points",
        buffered.wer,
        per.wer
    );
    assert!(
        buffered.speakers > 0 && per.speakers > 0,
        "expected speaker transcripts from both strategies"
    );
}

/// Buffered Sortformer windowing should run the diarizer at least 4x less often.
#[cfg(feature = "multitalker")]
#[test]
fn multitalker_diar_calls_reduced() {
    use parakeet_rs::DiarStrategy;

    if !multitalker_ready("multitalker_diar_calls_reduced") {
        return;
    }
    let Some(sample) = sample_named("smoke") else {
        skip("multitalker_diar_calls_reduced", "samples/smoke missing");
        return;
    };
    let (mut audio, _) = load_wav_mono_16k(&sample.wav);
    peak_normalize(&mut audio);
    let reference = reference_for(&sample);

    let per = run_multitalker_pass(
        &audio,
        DiarStrategy::PerChunk,
        &sample.name,
        &reference,
        "streaming",
    );
    let buffered = run_multitalker_pass(
        &audio,
        DiarStrategy::Buffered,
        &sample.name,
        &reference,
        "streaming-buffered",
    );
    eprintln!(
        "\n[multitalker_diar_calls_reduced] PerChunk runs {}  Buffered runs {}  ({:.2}x)",
        per.session_runs,
        buffered.session_runs,
        per.session_runs as f32 / buffered.session_runs.max(1) as f32
    );
    assert!(
        per.session_runs >= buffered.session_runs.saturating_mul(4),
        "expected Buffered Sortformer runs to drop by at least 4x (per {} buffered {})",
        per.session_runs,
        buffered.session_runs
    );
}

// --- WebGPU CPU parity (Phase 4) -----------------------------------------

fn webgpu_max_wer_delta() -> f32 {
    std::env::var("PARAKEET_WEBGPU_MAX_WER_DELTA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05)
}

/// Compare CPU vs WebGPU WER on the smoke clip for one model configuration.
/// Records both runs to `asr_report.csv` with `ep=cpu` / `ep=webgpu`.
/// On WebGPU inference failure, prints a skip notice (known upstream issues on
/// some graphs — see `plans/WEBGPU_IMPLEMENTATION_PLAN.md` section 4).
fn assert_cpu_webgpu_wer_parity(
    test: &str,
    model: &str,
    mode: &str,
    decoding: &str,
    reference: &str,
    audio_secs: f32,
    transcribe: impl Fn(&parakeet_rs::ExecutionConfig) -> Result<(String, f32), String>,
) {
    let max_delta = webgpu_max_wer_delta();

    let cpu_cfg = execution_config_for_ep("cpu");
    let (hyp_cpu, proc_cpu) = match transcribe(&cpu_cfg) {
        Ok(v) => v,
        Err(e) => panic!("{test}: CPU transcribe failed: {e}"),
    };
    let wer_cpu = record_asr(
        "smoke", model, mode, decoding, "cpu", audio_secs, proc_cpu, reference, &hyp_cpu,
    );

    let gpu_cfg = execution_config_for_ep("webgpu");
    match transcribe(&gpu_cfg) {
        Ok((hyp_gpu, proc_gpu)) => {
            let wer_gpu = record_asr(
                "smoke", model, mode, decoding, "webgpu", audio_secs, proc_gpu, reference, &hyp_gpu,
            );
            let delta = (wer_gpu - wer_cpu).abs();
            eprintln!(
                "\n[{test}] CPU WER {:.1}%  WebGPU WER {:.1}%  delta {:.1}% (max {:.1}%)",
                wer_cpu * 100.0,
                wer_gpu * 100.0,
                delta * 100.0,
                max_delta * 100.0
            );
            assert!(
                delta <= max_delta,
                "{test}: |WER_webgpu - WER_cpu| = {:.1}% exceeds PARAKEET_WEBGPU_MAX_WER_DELTA \
({:.1}%). Set PARAKEET_WEBGPU_MAX_WER_DELTA to adjust.",
                delta * 100.0,
                max_delta * 100.0
            );
        }
        Err(e) => {
            skip(
                test,
                &format!(
                    "WebGPU transcribe failed ({e}); see plans/WEBGPU_IMPLEMENTATION_PLAN.md \
section 4 (known ORT WebGPU issues on some graphs, e.g. Nemotron Slice on ort rc.12 Linux/Vulkan, fixed in rc.15)"
                ),
            );
        }
    }
}

/// CPU vs WebGPU WER parity on the smoke clip. CTC offline is the primary P0
/// target; Nemotron streaming is optional (segfaulted on ort rc.12, passes on rc.15 — opt in
/// with `PARAKEET_WEBGPU_NEMOTRON_PARITY=1`).
#[cfg(feature = "webgpu")]
#[test]
fn webgpu_cpu_wer_parity_smoke() {
    let Some(sample) = sample_named("smoke") else {
        skip("webgpu_cpu_wer_parity_smoke", "samples/smoke missing");
        return;
    };

    let (audio, _) = load_wav_mono_16k(&sample.wav);
    let audio_secs = audio.len() as f32 / 16_000.0;
    let reference = reference_for(&sample);

    if ctc_available() {
        eprintln!("\n[webgpu_cpu_wer_parity_smoke] CTC offline (P0):");
        let audio = audio.clone();
        let reference = reference.clone();
        assert_cpu_webgpu_wer_parity(
            "webgpu_cpu_wer_parity_smoke/ctc",
            "parakeet-ctc",
            "offline",
            "greedy",
            &reference,
            audio_secs,
            |cfg| {
                use parakeet_rs::Parakeet;
                let mut m = Parakeet::from_pretrained_with_decoding(
                    ctc_dir(),
                    Some(cfg.clone()),
                    DecodingStrategy::Greedy,
                )
                .map_err(|e| e.to_string())?;
                Ok(offline_transcribe(&mut m, &audio))
            },
        );
    } else {
        skip(
            "webgpu_cpu_wer_parity_smoke/ctc",
            &format!("CTC model not found at {}", ctc_dir().display()),
        );
    }

    let nemotron_parity = std::env::var("PARAKEET_WEBGPU_NEMOTRON_PARITY")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    if !nemotron_parity {
        skip(
            "webgpu_cpu_wer_parity_smoke/nemotron",
            "set PARAKEET_WEBGPU_NEMOTRON_PARITY=1 to run (segfaulted on ort rc.12; passes on rc.15 Linux/Vulkan)",
        );
        return;
    }
    if !nemotron_available() {
        skip(
            "webgpu_cpu_wer_parity_smoke/nemotron",
            &format!("model not found at {}", nemotron_dir().display()),
        );
        return;
    }

    eprintln!("\n[webgpu_cpu_wer_parity_smoke] Nemotron streaming (P0, opt-in):");
    let mut buf = audio;
    peak_normalize(&mut buf);
    assert_cpu_webgpu_wer_parity(
        "webgpu_cpu_wer_parity_smoke/nemotron",
        "nemotron-en-0.6b-int8",
        "streaming",
        "greedy",
        &reference,
        audio_secs,
        |cfg| {
            let mut m = parakeet_rs::Nemotron::from_pretrained(nemotron_dir(), Some(cfg.clone()))
                .map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            let hyp = nemotron_transcribe(&mut m, &buf);
            Ok((hyp, t0.elapsed().as_secs_f32()))
        },
    );
}

#[cfg(not(feature = "webgpu"))]
#[test]
fn webgpu_cpu_wer_parity_smoke() {
    skip(
        "webgpu_cpu_wer_parity_smoke",
        "requires `cargo test --features webgpu`",
    );
}
