//! Shared support for model-backed integration tests.
//!
//! These tests run against real ONNX weights and the WAV clips under `samples/`.
//! Weights are large and never committed (see AGENTS.md), so every model entry
//! point here resolves to a default path under `models/` that can be overridden
//! by an environment variable, and tests skip gracefully when the weights are
//! absent (CI without models stays green).

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Default model locations (overridable via env vars).
pub const NEMOTRON_DIR_ENV: &str = "PARAKEET_NEMOTRON_DIR";
pub const MULTITALKER_DIR_ENV: &str = "PARAKEET_MULTITALKER_DIR";
pub const SORTFORMER_ONNX_ENV: &str = "PARAKEET_SORTFORMER_ONNX";
pub const SORTFORMER_8SPK_ONNX_ENV: &str = "PARAKEET_SORTFORMER_8SPK_ONNX";
pub const EP_ENV: &str = "PARAKEET_EP";
pub const WEBGPU_DEVICE_ID_ENV: &str = "PARAKEET_WEBGPU_DEVICE_ID";

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn resolve(env_key: &str, default_rel: &str) -> PathBuf {
    match std::env::var(env_key) {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => repo_root().join(default_rel),
    }
}

pub fn nemotron_dir() -> PathBuf {
    resolve(
        NEMOTRON_DIR_ENV,
        "models/nemotron-speech-streaming-en-0.6b_int8_onnx",
    )
}

pub fn multitalker_dir() -> PathBuf {
    resolve(
        MULTITALKER_DIR_ENV,
        "models/multitalker-parakeet-streaming-0.6b-v1-onnx-int8",
    )
}

pub fn sortformer_onnx() -> PathBuf {
    resolve(
        SORTFORMER_ONNX_ENV,
        "models/sortformer-4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.onnx",
    )
}

pub fn sortformer_8spk_onnx() -> PathBuf {
    resolve(
        SORTFORMER_8SPK_ONNX_ENV,
        "models/ultra_diar_streaming_sortformer_8spk_v1_onnx/ultra_diar_streaming_sortformer_8spk_v1.onnx",
    )
}

/// True if every named file exists inside `dir`.
pub fn dir_has(dir: &Path, files: &[&str]) -> bool {
    dir.is_dir() && files.iter().all(|f| dir.join(f).exists())
}

pub fn nemotron_available() -> bool {
    dir_has(
        &nemotron_dir(),
        &["encoder.onnx", "decoder_joint.onnx", "tokenizer.model"],
    )
}

pub fn multitalker_available() -> bool {
    let d = multitalker_dir();
    d.join("tokenizer.model").exists()
        && (d.join("encoder.int8.onnx").exists() || d.join("encoder.onnx").exists())
        && (d.join("decoder_joint.int8.onnx").exists() || d.join("decoder_joint.onnx").exists())
}

pub fn sortformer_available() -> bool {
    sortformer_onnx().exists()
}

pub fn sortformer_8spk_available() -> bool {
    sortformer_8spk_onnx().exists()
}

// Additional ASR model dirs (env-overridable; default under models/). None are
// committed; tests skip when absent.
pub fn ctc_dir() -> PathBuf {
    resolve("PARAKEET_CTC_DIR", "models/parakeet-ctc")
}
pub fn tdt_dir() -> PathBuf {
    resolve("PARAKEET_TDT_DIR", "models/parakeet-tdt")
}
pub fn unified_dir() -> PathBuf {
    resolve("PARAKEET_UNIFIED_DIR", "models/parakeet-unified")
}
pub fn eou_dir() -> PathBuf {
    resolve("PARAKEET_EOU_DIR", "models/realtime-eou-120m")
}
pub fn cohere_dir() -> PathBuf {
    resolve("PARAKEET_COHERE_DIR", "models/cohere-transcribe-03-2026-int8")
}

pub fn ctc_available() -> bool {
    let d = ctc_dir();
    d.is_dir()
        && d.join("tokenizer.json").exists()
        && std::fs::read_dir(&d)
            .map(|rd| {
                rd.flatten().any(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("onnx")
                })
            })
            .unwrap_or(false)
}
pub fn tdt_available() -> bool {
    dir_has(
        &tdt_dir(),
        &["encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"],
    )
}
pub fn unified_available() -> bool {
    let d = unified_dir();
    d.join("tokenizer.model").exists()
        && (d.join("encoder.onnx").exists() || d.join("encoder.int8.onnx").exists())
        && (d.join("decoder_joint.onnx").exists() || d.join("decoder_joint.int8.onnx").exists())
}
pub fn eou_available() -> bool {
    dir_has(&eou_dir(), &["encoder.onnx", "decoder_joint.onnx", "tokenizer.json"])
}
pub fn cohere_available() -> bool {
    let d = cohere_dir();
    if !d.is_dir() || !d.join("tokenizer.json").exists() {
        return false;
    }
    // Mirror model_cohere candidate resolution (int8 quantized preferred), both
    // flat and nested `onnx/` layouts.
    let enc = [
        "onnx/encoder_model_quantized.onnx",
        "encoder_model_quantized.onnx",
        "onnx/encoder_model.onnx",
        "encoder_model.onnx",
        "onnx/encoder_model_fp16.onnx",
        "encoder_model_fp16.onnx",
    ];
    let dec = [
        "onnx/decoder_model_merged_quantized.onnx",
        "decoder_model_merged_quantized.onnx",
        "onnx/decoder_model_merged.onnx",
        "decoder_model_merged.onnx",
        "onnx/decoder_model_merged_fp16.onnx",
        "decoder_model_merged_fp16.onnx",
    ];
    enc.iter().any(|f| d.join(f).exists()) && dec.iter().any(|f| d.join(f).exists())
}

/// Emit a standard skip notice (visible with `cargo test -- --nocapture`).
pub fn skip(test: &str, reason: &str) {
    eprintln!("[skip] {test}: {reason}");
}

// --- Samples -------------------------------------------------------------

/// One discovered sample: a WAV clip plus its reference transcript.
#[derive(Debug, Clone)]
pub struct Sample {
    pub name: String,
    pub wav: PathBuf,
    pub transcript: PathBuf,
}

/// Discover every sample under `samples/`: each subdirectory that contains a
/// `*.wav` and a matching `*_transcript.txt`. New sample folders are picked up
/// automatically, so adding a clip wires it into the suite with no code change.
pub fn discover_samples() -> Vec<Sample> {
    let mut out = Vec::new();
    let root = repo_root().join("samples");
    let Ok(entries) = std::fs::read_dir(&root) else {
        return out;
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();

    for dir in dirs {
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let mut wav: Option<PathBuf> = None;
        let mut transcript: Option<PathBuf> = None;
        if let Ok(files) = std::fs::read_dir(&dir) {
            for f in files.flatten() {
                let p = f.path();
                match p.extension().and_then(|e| e.to_str()) {
                    Some("wav") => wav = Some(p),
                    Some("txt")
                        if p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.ends_with("_transcript.txt"))
                            .unwrap_or(false) =>
                    {
                        transcript = Some(p)
                    }
                    _ => {}
                }
            }
        }
        if let (Some(wav), Some(transcript)) = (wav, transcript) {
            out.push(Sample { name, wav, transcript });
        }
    }
    out
}

pub fn sample_named(name: &str) -> Option<Sample> {
    discover_samples().into_iter().find(|s| s.name == name)
}

// --- Audio ---------------------------------------------------------------

/// Load a WAV as 16 kHz mono f32. Returns `(samples, sample_rate)`.
/// Panics on non-16 kHz input (every sample clip is already 16 kHz).
pub fn load_wav_mono_16k(path: &Path) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let spec = reader.spec();

    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("read f32 samples"),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()
            .expect("read i16 samples"),
    };

    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }

    assert_eq!(spec.sample_rate, 16000, "samples must be 16 kHz");
    (audio, spec.sample_rate)
}

/// Peak-normalize in place (matches examples/streaming.rs).
pub fn peak_normalize(audio: &mut [f32]) {
    let max_val = audio.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_val > 1e-6 {
        for s in audio.iter_mut() {
            *s /= max_val + 1e-5;
        }
    }
}

// --- Reference transcript parsing & WER ----------------------------------

/// Strip speaker/timestamp scaffolding from a reference transcript and return
/// the spoken text. Removes `Speaker N` header lines and `M:SS` timestamps.
pub fn reference_text(raw: &str) -> String {
    let mut words = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip standalone speaker-header lines like "Speaker 1  0:02".
        let lower = trimmed.to_lowercase();
        if lower.starts_with("speaker ") && line_is_header(trimmed) {
            continue;
        }
        words.push(trimmed.to_string());
    }
    words.join(" ")
}

/// A header line is "Speaker <id>" optionally followed by a "M:SS" timestamp
/// and nothing else.
fn line_is_header(line: &str) -> bool {
    let mut toks = line.split_whitespace();
    match (toks.next(), toks.next()) {
        (Some(s), Some(id)) if s.eq_ignore_ascii_case("speaker") => {
            if id.trim_end_matches(|c: char| !c.is_alphanumeric())
                .chars()
                .all(|c| c.is_ascii_digit())
            {
                // Remaining tokens (if any) must all look like timestamps.
                toks.all(is_timestamp)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_timestamp(tok: &str) -> bool {
    let t = tok.trim();
    let parts: Vec<&str> = t.split(':').collect();
    parts.len() >= 2 && parts.iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

/// Normalize text for WER: lowercase, drop punctuation, collapse whitespace.
pub fn normalize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|w| w.to_string())
        .collect()
}

/// Word error rate of `hypothesis` against `reference` (0.0 = perfect).
pub fn wer(reference: &str, hypothesis: &str) -> f32 {
    let r = normalize_words(reference);
    let h = normalize_words(hypothesis);
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&r, &h) as f32 / r.len() as f32
}

/// Character error rate against the normalized reference string.
pub fn cer(reference: &str, hypothesis: &str) -> f32 {
    let r: Vec<char> = normalize_words(reference).join(" ").chars().collect();
    let h: Vec<char> = normalize_words(hypothesis).join(" ").chars().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&r, &h) as f32 / r.len() as f32
}

/// Generic Levenshtein distance over slices of comparable items.
pub fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ai) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bj) in b.iter().enumerate() {
            let cost = if ai == bj { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

// --- Diarization metrics (DER / JER) -------------------------------------

/// A timed speaker segment in seconds (model-agnostic; used for both reference
/// turns and hypothesis segments).
#[derive(Debug, Clone, Copy)]
pub struct DiarSegment {
    pub speaker: usize,
    pub start: f32,
    pub end: f32,
}

/// Frame-based diarization error breakdown (no collar, 10 ms frames).
#[derive(Debug, Clone, Copy, Default)]
pub struct DiarMetrics {
    /// Diarization Error Rate = (missed + false_alarm + confusion) / ref_speech.
    pub der: f32,
    pub missed: f32,
    pub false_alarm: f32,
    pub confusion: f32,
    /// Jaccard Error Rate, averaged over reference speakers.
    pub jer: f32,
    pub num_ref_speakers: usize,
    pub num_hyp_speakers: usize,
}

const FRAME_STEP: f32 = 0.01; // 10 ms

/// Parse `M:SS` / `H:MM:SS` into seconds.
fn parse_clock(tok: &str) -> Option<f32> {
    let parts: Vec<&str> = tok.trim().split(':').collect();
    if parts.len() < 2 || parts.iter().any(|p| p.is_empty() || !p.chars().all(|c| c.is_ascii_digit())) {
        return None;
    }
    let mut secs = 0f32;
    for p in &parts {
        secs = secs * 60.0 + p.parse::<f32>().ok()?;
    }
    Some(secs)
}

/// Build a reference diarization timeline from a transcript whose speaker turns
/// are marked by `Speaker N  M:SS` headers. Each turn runs from its timestamp to
/// the next turn's timestamp (last turn ends at `audio_secs`).
pub fn parse_reference_turns(raw: &str, audio_secs: f32) -> Vec<DiarSegment> {
    let mut starts: Vec<(usize, f32)> = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if !trimmed.to_lowercase().starts_with("speaker ") || !line_is_header(trimmed) {
            continue;
        }
        let mut toks = trimmed.split_whitespace();
        let _ = toks.next(); // "Speaker"
        let id_tok = toks.next().unwrap_or("");
        let spk: usize = id_tok
            .trim_end_matches(|c: char| !c.is_alphanumeric())
            .parse()
            .map(|n: usize| n.saturating_sub(1)) // 1-based -> 0-based
            .unwrap_or(0);
        let start = toks.find_map(parse_clock).unwrap_or(0.0);
        starts.push((spk, start));
    }

    let mut turns = Vec::with_capacity(starts.len());
    for i in 0..starts.len() {
        let (spk, start) = starts[i];
        let end = if i + 1 < starts.len() {
            starts[i + 1].1
        } else {
            audio_secs.max(start)
        };
        if end > start {
            turns.push(DiarSegment { speaker: spk, start, end });
        }
    }
    turns
}

fn timeline_end(segs: &[DiarSegment], audio_secs: f32) -> f32 {
    segs.iter().fold(audio_secs, |m, s| m.max(s.end))
}

fn remap(speaker: usize) -> usize {
    speaker
}

/// Compute frame-based DER and JER between reference turns and hypothesis
/// segments (both in seconds). Reference is treated as single-speaker-per-frame
/// (transcript turns do not encode overlap); hypothesis may overlap. Hypothesis
/// speaker ids are mapped to reference ids greedily by maximum frame overlap.
pub fn diarization_metrics(
    reference: &[DiarSegment],
    hypothesis: &[DiarSegment],
    audio_secs: f32,
) -> DiarMetrics {
    let end = timeline_end(reference, audio_secs).max(timeline_end(hypothesis, audio_secs));
    let n_frames = (end / FRAME_STEP).ceil() as usize;
    if n_frames == 0 {
        return DiarMetrics::default();
    }

    let ref_ids: Vec<usize> = unique_speakers(reference);
    let hyp_ids: Vec<usize> = unique_speakers(hypothesis);

    // Per-frame reference label (single) and hypothesis active set.
    let mut ref_frame = vec![usize::MAX; n_frames];
    for s in reference {
        for f in frame_range(s, n_frames) {
            ref_frame[f] = remap(s.speaker);
        }
    }
    let mut hyp_frame: Vec<Vec<usize>> = vec![Vec::new(); n_frames];
    for s in hypothesis {
        for f in frame_range(s, n_frames) {
            if !hyp_frame[f].contains(&s.speaker) {
                hyp_frame[f].push(s.speaker);
            }
        }
    }

    // Greedy hyp->ref mapping by overlap.
    let map = greedy_map(&hyp_ids, &ref_ids, &ref_frame, &hyp_frame);

    let (mut missed, mut fa, mut conf, mut total_ref) = (0u64, 0u64, 0u64, 0u64);
    for f in 0..n_frames {
        let n_ref = if ref_frame[f] != usize::MAX { 1u64 } else { 0 };
        let n_sys = hyp_frame[f].len() as u64;
        let n_correct = if n_ref == 1
            && hyp_frame[f]
                .iter()
                .any(|h| map.get(h).copied() == Some(ref_frame[f]))
        {
            1u64
        } else {
            0
        };
        missed += n_ref.saturating_sub(n_sys);
        fa += n_sys.saturating_sub(n_ref);
        conf += n_ref.min(n_sys) - n_correct;
        total_ref += n_ref;
    }

    let denom = total_ref.max(1) as f32;
    let jer = jaccard_error_rate(&ref_ids, &ref_frame, &hyp_frame, &map);

    DiarMetrics {
        der: (missed + fa + conf) as f32 / denom,
        missed: missed as f32 / denom,
        false_alarm: fa as f32 / denom,
        confusion: conf as f32 / denom,
        jer,
        num_ref_speakers: ref_ids.len(),
        num_hyp_speakers: hyp_ids.len(),
    }
}

fn frame_range(s: &DiarSegment, n_frames: usize) -> std::ops::Range<usize> {
    let a = (s.start / FRAME_STEP).floor() as usize;
    let b = ((s.end / FRAME_STEP).ceil() as usize).min(n_frames);
    a.min(n_frames)..b.max(a.min(n_frames))
}

fn unique_speakers(segs: &[DiarSegment]) -> Vec<usize> {
    let mut v: Vec<usize> = segs.iter().map(|s| s.speaker).collect();
    v.sort_unstable();
    v.dedup();
    v
}

fn greedy_map(
    hyp_ids: &[usize],
    ref_ids: &[usize],
    ref_frame: &[usize],
    hyp_frame: &[Vec<usize>],
) -> std::collections::HashMap<usize, usize> {
    // Overlap counts between each (hyp, ref) pair.
    let mut pairs: Vec<(u64, usize, usize)> = Vec::new();
    for &h in hyp_ids {
        for &r in ref_ids {
            let mut c = 0u64;
            for f in 0..ref_frame.len() {
                if ref_frame[f] == r && hyp_frame[f].contains(&h) {
                    c += 1;
                }
            }
            if c > 0 {
                pairs.push((c, h, r));
            }
        }
    }
    pairs.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let mut map = std::collections::HashMap::new();
    let mut used_ref = std::collections::HashSet::new();
    for (_, h, r) in pairs {
        if map.contains_key(&h) || used_ref.contains(&r) {
            continue;
        }
        map.insert(h, r);
        used_ref.insert(r);
    }
    map
}

fn jaccard_error_rate(
    ref_ids: &[usize],
    ref_frame: &[usize],
    hyp_frame: &[Vec<usize>],
    map: &std::collections::HashMap<usize, usize>,
) -> f32 {
    if ref_ids.is_empty() {
        return 0.0;
    }
    // Invert the mapping: ref id -> hyp id.
    let mut ref_to_hyp = std::collections::HashMap::new();
    for (&h, &r) in map {
        ref_to_hyp.insert(r, h);
    }
    let mut sum = 0f32;
    for &r in ref_ids {
        let h = ref_to_hyp.get(&r).copied();
        let (mut inter, mut union) = (0u64, 0u64);
        for f in 0..ref_frame.len() {
            let in_ref = ref_frame[f] == r;
            let in_hyp = h.map(|h| hyp_frame[f].contains(&h)).unwrap_or(false);
            if in_ref || in_hyp {
                union += 1;
            }
            if in_ref && in_hyp {
                inter += 1;
            }
        }
        sum += if union > 0 {
            1.0 - inter as f32 / union as f32
        } else {
            0.0
        };
    }
    sum / ref_ids.len() as f32
}

// --- CSV report ----------------------------------------------------------

/// Append a row to a CSV under `samples/`, writing the header if the file is new.
/// Reports are stored in `samples/` so diarization quality/speed can be compared
/// across code changes (sort by `run_id`, tag a change via PARAKEET_BENCH_LABEL).
pub fn append_csv_row(filename: &str, header: &str, row: &str) {
    use std::io::Write;
    let path = repo_root().join("samples").join(filename);
    let exists = path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    if !exists {
        writeln!(f, "{header}").expect("write csv header");
    }
    writeln!(f, "{row}").expect("write csv row");
}

/// Seconds since the Unix epoch, used as a sortable run identifier.
pub fn run_id() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Optional human label for a benchmark run (e.g. "before-fix").
pub fn bench_label() -> String {
    std::env::var("PARAKEET_BENCH_LABEL").unwrap_or_default()
}

/// Normalized `PARAKEET_EP` value (`cpu` when unset).
pub fn requested_ep() -> String {
    std::env::var(EP_ENV)
        .unwrap_or_else(|_| "cpu".to_string())
        .to_ascii_lowercase()
}

/// Skip when `PARAKEET_EP` requests an EP that was not compiled in.
pub fn ep_skip_if_unavailable(test: &str) -> bool {
    match requested_ep().as_str() {
        "cpu" | "" => false,
        #[cfg(not(feature = "webgpu"))]
        "webgpu" => {
            skip(
                test,
                &format!("{EP_ENV}=webgpu requires `cargo test --features webgpu`"),
            );
            true
        }
        #[cfg(feature = "webgpu")]
        "webgpu" => false,
        #[cfg(not(feature = "cuda"))]
        "cuda" => {
            skip(test, &format!("{EP_ENV}=cuda requires `cargo test --features cuda`"));
            true
        }
        #[cfg(feature = "cuda")]
        "cuda" => false,
        #[cfg(not(feature = "directml"))]
        "directml" => {
            skip(
                test,
                &format!("{EP_ENV}=directml requires `cargo test --features directml`"),
            );
            true
        }
        #[cfg(feature = "directml")]
        "directml" => false,
        #[cfg(not(feature = "coreml"))]
        "coreml" => {
            skip(
                test,
                &format!("{EP_ENV}=coreml requires `cargo test --features coreml`"),
            );
            true
        }
        #[cfg(feature = "coreml")]
        "coreml" => false,
        #[cfg(not(feature = "tensorrt"))]
        "tensorrt" => {
            skip(
                test,
                &format!("{EP_ENV}=tensorrt requires `cargo test --features tensorrt`"),
            );
            true
        }
        #[cfg(feature = "tensorrt")]
        "tensorrt" => false,
        #[cfg(not(feature = "migraphx"))]
        "migraphx" => {
            skip(
                test,
                &format!("{EP_ENV}=migraphx requires `cargo test --features migraphx`"),
            );
            true
        }
        #[cfg(feature = "migraphx")]
        "migraphx" => false,
        #[cfg(not(feature = "openvino"))]
        "openvino" => {
            skip(
                test,
                &format!("{EP_ENV}=openvino requires `cargo test --features openvino`"),
            );
            true
        }
        #[cfg(feature = "openvino")]
        "openvino" => false,
        other => {
            skip(test, &format!("unknown {EP_ENV}={other}"));
            true
        }
    }
}

/// Build [`parakeet_rs::ExecutionConfig`] for an explicit EP name (`cpu`, `webgpu`, …).
pub fn execution_config_for_ep(ep: &str) -> parakeet_rs::ExecutionConfig {
    use parakeet_rs::{ExecutionConfig, ExecutionProvider};

    let ep = ep.to_ascii_lowercase();
    let mut cfg = ExecutionConfig::new();
    match ep.as_str() {
        "cpu" | "" => cfg,
        #[cfg(feature = "webgpu")]
        "webgpu" => {
            cfg = cfg.with_execution_provider(ExecutionProvider::WebGPU);
            if let Ok(id_str) = std::env::var(WEBGPU_DEVICE_ID_ENV) {
                if let Ok(id) = id_str.parse::<i32>() {
                    cfg = cfg.with_webgpu_device_id(id);
                }
            }
            cfg
        }
        #[cfg(not(feature = "webgpu"))]
        "webgpu" => cfg,
        #[cfg(feature = "cuda")]
        "cuda" => cfg.with_execution_provider(ExecutionProvider::Cuda),
        #[cfg(not(feature = "cuda"))]
        "cuda" => cfg,
        #[cfg(feature = "directml")]
        "directml" => cfg.with_execution_provider(ExecutionProvider::DirectML),
        #[cfg(not(feature = "directml"))]
        "directml" => cfg,
        #[cfg(feature = "coreml")]
        "coreml" => cfg.with_execution_provider(ExecutionProvider::CoreML),
        #[cfg(not(feature = "coreml"))]
        "coreml" => cfg,
        #[cfg(feature = "tensorrt")]
        "tensorrt" => cfg.with_execution_provider(ExecutionProvider::TensorRT),
        #[cfg(not(feature = "tensorrt"))]
        "tensorrt" => cfg,
        #[cfg(feature = "migraphx")]
        "migraphx" => cfg.with_execution_provider(ExecutionProvider::MIGraphX),
        #[cfg(not(feature = "migraphx"))]
        "migraphx" => cfg,
        #[cfg(feature = "openvino")]
        "openvino" => cfg.with_execution_provider(ExecutionProvider::OpenVINO),
        #[cfg(not(feature = "openvino"))]
        "openvino" => cfg,
        _ => cfg,
    }
}

/// Build [`parakeet_rs::ExecutionConfig`] from `PARAKEET_EP` (default: CPU).
pub fn execution_config() -> parakeet_rs::ExecutionConfig {
    execution_config_for_ep(&requested_ep())
}

pub const ASR_CSV_HEADER: &str = "run_id,label,sample,model,mode,decoding,ep,\
audio_secs,proc_secs,rtf,wer,cer";

/// Record one ASR result to `samples/asr_report.csv`, print a row, and return WER.
#[allow(clippy::too_many_arguments)]
pub fn record_asr(
    sample: &str,
    model: &str,
    mode: &str,
    decoding: &str,
    ep: &str,
    audio_secs: f32,
    proc_secs: f32,
    reference: &str,
    hyp: &str,
) -> f32 {
    let w = wer(reference, hyp);
    let c = cer(reference, hyp);
    let rtf = if proc_secs > 1e-6 { audio_secs / proc_secs } else { 0.0 };
    eprintln!(
        "  {:<26} {:<9} {:<6} {:<6} RTF {:>6.1}x  WER {:>5.1}%  CER {:>5.1}%",
        sample,
        mode,
        decoding,
        ep,
        rtf,
        w * 100.0,
        c * 100.0
    );
    append_csv_row(
        "asr_report.csv",
        ASR_CSV_HEADER,
        &format!(
            "{},{},{sample},{model},{mode},{decoding},{ep},{audio_secs:.2},{proc_secs:.3},\
{rtf:.2},{:.4},{:.4}",
            run_id(),
            bench_label(),
            w,
            c
        ),
    );
    w
}

/// Samples selected for an ASR run: always smoke, plus the rest when
/// PARAKEET_BENCH_ALL is set (the longer clips are minutes long).
pub fn selected_asr_samples() -> Vec<Sample> {
    let bench_all = std::env::var("PARAKEET_BENCH_ALL")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    discover_samples()
        .into_iter()
        .filter(|s| s.name == "smoke" || bench_all)
        .collect()
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn der_zero_with_relabeled_speakers() {
        // Reference: spk0 [0,1), spk1 [1,2). Hypothesis identical but ids swapped.
        let reference = vec![
            DiarSegment { speaker: 0, start: 0.0, end: 1.0 },
            DiarSegment { speaker: 1, start: 1.0, end: 2.0 },
        ];
        let hypothesis = vec![
            DiarSegment { speaker: 7, start: 0.0, end: 1.0 },
            DiarSegment { speaker: 3, start: 1.0, end: 2.0 },
        ];
        let m = diarization_metrics(&reference, &hypothesis, 2.0);
        assert!(m.der < 1e-6, "der={}", m.der);
        assert!(m.jer < 1e-6, "jer={}", m.jer);
    }

    #[test]
    fn der_counts_missed_and_confusion() {
        let reference = vec![
            DiarSegment { speaker: 0, start: 0.0, end: 1.0 },
            DiarSegment { speaker: 1, start: 1.0, end: 2.0 },
        ];
        // Hyp: correct first half, silence second half -> 50% missed.
        let hypothesis = vec![DiarSegment { speaker: 5, start: 0.0, end: 1.0 }];
        let m = diarization_metrics(&reference, &hypothesis, 2.0);
        assert!((m.der - 0.5).abs() < 0.02, "der={}", m.der);
        assert!(m.missed > 0.4, "missed={}", m.missed);
    }

    #[test]
    fn parse_reference_turns_uses_next_start_as_end() {
        let raw = "Speaker 1  0:02\nhello\n\nSpeaker 2  0:10\nworld";
        let turns = parse_reference_turns(raw, 22.0);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, 0);
        assert!((turns[0].start - 2.0).abs() < 1e-6);
        assert!((turns[0].end - 10.0).abs() < 1e-6);
        assert!((turns[1].end - 22.0).abs() < 1e-6);
    }

    #[test]
    fn wer_basic() {
        assert_eq!(wer("the quick brown fox", "the quick brown fox"), 0.0);
        assert!((wer("the quick brown fox", "the quik brown fox") - 0.25).abs() < 1e-6);
        assert_eq!(wer("", ""), 0.0);
    }

    #[test]
    fn reference_strips_speaker_headers() {
        let raw = "Speaker 1  0:02\nHello there world.\n\nSpeaker 2  0:10\nGeneral Kenobi.";
        let txt = reference_text(raw);
        assert!(!txt.to_lowercase().contains("speaker 1"));
        assert!(txt.contains("Hello there world."));
        assert!(txt.contains("General Kenobi."));
    }

    #[test]
    fn normalize_drops_punctuation_and_case() {
        assert_eq!(normalize_words("Hello, World!"), vec!["hello", "world"]);
    }
}
