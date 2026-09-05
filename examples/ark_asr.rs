/*
ARK-ASR-0.6B speech-LLM — offline transcription.

Whisper-style encoder + Qwen2 decoder. Download the INT8 ONNX package from:
https://huggingface.co/Audio8/ark-asr-0.6b-int8-onnx

Usage:
  cargo run --release --example ark_asr --features ark-asr -- \
    <model_dir> <audio.wav> [max_new_tokens]

Examples:
  cargo run --release --example ark_asr --features ark-asr -- \
    ./models/ark-asr-0.6b-int8-onnx samples/smoke/test01_20s.wav
*/

#[cfg(feature = "ark-asr")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{ArkAsr, Transcriber};
    use std::env;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model_dir> <audio.wav> [max_new_tokens]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let audio_path = &args[2];
    let max_new_tokens = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256usize);

    println!("Loading ARK-ASR model...");
    let load_start = Instant::now();
    let mut model = ArkAsr::from_pretrained(model_dir, None)?;
    model.set_max_new_tokens(max_new_tokens);
    println!("Loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    let start = Instant::now();
    let result = model.transcribe_file(audio_path, None)?;
    let elapsed = start.elapsed().as_secs_f32();

    let reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let duration_secs =
        reader.duration() as f32 / spec.sample_rate.max(1) as f32;

    println!("\n{}", result.text);
    println!(
        "\nTranscribed in {:.2}s (audio {:.2}s, RTF {:.2}x)",
        elapsed,
        duration_secs,
        if elapsed > 0.0 {
            duration_secs / elapsed
        } else {
            0.0
        }
    );

    Ok(())
}

#[cfg(not(feature = "ark-asr"))]
fn main() {
    eprintln!("Rebuild with --features ark-asr to run this example.");
    std::process::exit(1);
}
