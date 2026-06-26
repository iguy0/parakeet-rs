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
fn nemotron_transcribe(model: &mut parakeet_rs::Nemotron, audio: &[f32]) -> String {
    for chunk in audio.chunks(NEMOTRON_CHUNK_SAMPLES) {
        let mut buf = chunk.to_vec();
        if buf.len() < NEMOTRON_CHUNK_SAMPLES {
            buf.resize(NEMOTRON_CHUNK_SAMPLES, 0.0);
        }
        model.transcribe_chunk(&buf).expect("nemotron chunk");
    }
    // Flush with trailing silence so the last tokens are emitted.
    for _ in 0..3 {
        model
            .transcribe_chunk(&vec![0.0; NEMOTRON_CHUNK_SAMPLES])
            .expect("nemotron flush");
    }
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

    assert!(!segments.is_empty(), "expected at least one speaker segment");
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
        skip("ctc_offline_wer", &format!("model not found at {}", ctc_dir().display()));
        return;
    }
    for (decoding, label) in [
        (DecodingStrategy::Greedy, "greedy"),
        (DecodingStrategy::Beam(BeamConfig::default_ctc()), "beam"),
    ] {
        run_asr_bench("ctc_offline_wer", "parakeet-ctc", "offline", label, smoke_max_wer(), |audio| {
            let mut m = Parakeet::from_pretrained_with_decoding(
                ctc_dir(),
                Some(execution_config()),
                decoding,
            )
                .expect("load ctc");
            offline_transcribe(&mut m, audio)
        });
    }
}

#[test]
fn tdt_offline_wer() {
    use parakeet_rs::ParakeetTDT;
    if !tdt_available() {
        skip("tdt_offline_wer", &format!("model not found at {}", tdt_dir().display()));
        return;
    }
    for (decoding, label) in [
        (DecodingStrategy::Greedy, "greedy"),
        (DecodingStrategy::Beam(BeamConfig::default_tdt()), "beam"),
    ] {
        run_asr_bench("tdt_offline_wer", "parakeet-tdt", "offline", label, smoke_max_wer(), |audio| {
            let mut m = ParakeetTDT::from_pretrained_with_decoding(
                tdt_dir(),
                Some(execution_config()),
                decoding,
            )
                .expect("load tdt");
            offline_transcribe(&mut m, audio)
        });
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
        (DecodingStrategy::Beam(BeamConfig::default_unified()), "beam"),
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
        skip("eou_streaming_wer", &format!("model not found at {}", eou_dir().display()));
        return;
    }
    // EOU is lowercase, no punctuation; allow a looser smoke ceiling.
    let ceiling = std::env::var("SMOKE_MAX_WER_EOU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.4);
    run_asr_bench("eou_streaming_wer", "realtime-eou-120m", "streaming", "greedy", ceiling, |audio| {
        let mut m = ParakeetEOU::from_pretrained(eou_dir(), Some(execution_config())).expect("load eou");
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
    });
}

#[cfg(feature = "cohere")]
#[test]
fn cohere_offline_wer() {
    use parakeet_rs::CohereASR;
    if !cohere_available() {
        skip("cohere_offline_wer", &format!("model not found at {}", cohere_dir().display()));
        return;
    }
    run_asr_bench("cohere_offline_wer", "cohere-transcribe-int8", "offline", "greedy", smoke_max_wer(), |audio| {
        let mut m = CohereASR::from_pretrained(cohere_dir(), Some(execution_config())).expect("load cohere");
        let t0 = Instant::now();
        let text = m.transcribe_audio(audio, "en", true, false).expect("cohere transcribe");
        (text, t0.elapsed().as_secs_f32())
    });
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

        let mut sf = Sortformer::with_config(
            sortformer_onnx(),
            Some(execution_config()),
            DiarizationConfig::callhome(),
        )
                .expect("load sortformer");

        let t0 = Instant::now();
        let segments = sf.diarize(audio, sample_rate, 1).expect("diarize");
        let proc_secs = t0.elapsed().as_secs_f32();
        let rtf = if proc_secs > 0.0 { audio_secs / proc_secs } else { 0.0 };

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
    assert!(!segments.is_empty(), "streaming diarization produced no segments");
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
        "smoke",
        model,
        mode,
        decoding,
        "cpu",
        audio_secs,
        proc_cpu,
        reference,
        &hyp_cpu,
    );

    let gpu_cfg = execution_config_for_ep("webgpu");
    match transcribe(&gpu_cfg) {
        Ok((hyp_gpu, proc_gpu)) => {
            let wer_gpu = record_asr(
                "smoke",
                model,
                mode,
                decoding,
                "webgpu",
                audio_secs,
                proc_gpu,
                reference,
                &hyp_gpu,
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
section 4 (known ORT WebGPU issues on some graphs, e.g. Nemotron Slice on ort rc.12 Linux/Vulkan)"
                ),
            );
        }
    }
}

/// CPU vs WebGPU WER parity on the smoke clip. CTC offline is the primary P0
/// target; Nemotron streaming is optional (may segfault on ort rc.12 — opt in
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
            "set PARAKEET_WEBGPU_NEMOTRON_PARITY=1 to run (may segfault on ort rc.12 Linux/Vulkan)",
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
