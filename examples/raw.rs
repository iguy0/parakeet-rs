/*
Demonstrates using transcribe_samples()

This example shows manual audio loading and calling transcribe_samples() directly
with sample_rate and channels instead of using transcribe_file()

Usage:
cargo run --example raw 6_speakers.wav
cargo run --example raw 6_speakers.wav -- --decoding beam --beam-size 5
cargo run --example raw 6_speakers.wav tdt
cargo run --example raw 6_speakers.wav tdt -- --decoding beam --beam-size 5
cargo run --features webgpu --example raw 6_speakers.wav -- --ep webgpu

WARNING: TDT model has sequence length limitations (~8-10 minutes max).
For longer audio files, you must split into chunks (e.g., 5-minute segments)
and transcribe each chunk separately. Attempting to transcribe 25+ minute
audio files in one call will cause ONNX runtime errors.
Otherwise you will likely get a error like:
"Error: Ort(Error { code: RuntimeException, msg: "Non-zero status code returned while running Add node. Name:'/layers.0/self_attn/Add_2' Status Message: /Users/runner/work/ort-artifacts/ort-artifacts/onnxruntime/onnxruntime/core/providers/cpu/math/element_wise_ops.h:540 void onnxruntime::BroadcastIterator::Init(ptrdiff_t, ptrdiff_t) axis == 1 || axis == largest was false. })"
*/

use parakeet_rs::{
    BeamConfig, DecodingStrategy, ExecutionConfig, ExecutionProvider, Parakeet, ParakeetTDT,
    TimestampMode, Transcriber,
};
use std::env;
use std::time::Instant;

/// Build execution config from `--ep <name>` (after `--`) or `PARAKEET_EP` env (default: cpu).
fn parse_execution_config(flag_args: &[String]) -> Option<ExecutionConfig> {
    let ep_name = flag_args
        .windows(2)
        .find(|w| w[0] == "--ep")
        .map(|w| w[1].clone())
        .or_else(|| std::env::var("PARAKEET_EP").ok())
        .unwrap_or_else(|| "cpu".to_string())
        .to_ascii_lowercase();

    let mut cfg = ExecutionConfig::new();
    match ep_name.as_str() {
        "cpu" | "" => {}
        #[cfg(feature = "webgpu")]
        "webgpu" => {
            cfg = cfg.with_execution_provider(ExecutionProvider::WebGPU);
            if let Ok(id_str) = std::env::var("PARAKEET_WEBGPU_DEVICE_ID") {
                if let Ok(id) = id_str.parse::<i32>() {
                    cfg = cfg.with_webgpu_device_id(id);
                }
            }
            eprintln!("[execution provider: webgpu]");
        }
        #[cfg(not(feature = "webgpu"))]
        "webgpu" => {
            eprintln!("Warning: --ep webgpu ignored (rebuild with --features webgpu)");
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            cfg = cfg.with_execution_provider(ExecutionProvider::Cuda);
            eprintln!("[execution provider: cuda]");
        }
        #[cfg(not(feature = "cuda"))]
        "cuda" => {
            eprintln!("Warning: --ep cuda ignored (rebuild with --features cuda)");
        }
        #[cfg(feature = "directml")]
        "directml" => {
            cfg = cfg.with_execution_provider(ExecutionProvider::DirectML);
            eprintln!("[execution provider: directml]");
        }
        #[cfg(not(feature = "directml"))]
        "directml" => {
            eprintln!("Warning: --ep directml ignored (rebuild with --features directml)");
        }
        other => eprintln!("Warning: unknown execution provider '{other}', using cpu"),
    }

    if ep_name == "cpu" || ep_name.is_empty() {
        None
    } else {
        Some(cfg)
    }
}

fn parse_decoding(args: &[String], default_beam: fn() -> BeamConfig) -> DecodingStrategy {
    let mut use_beam = false;
    let mut beam_size = default_beam().beam_size;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--decoding" => {
                i += 1;
                if i < args.len() && args[i] == "beam" {
                    use_beam = true;
                }
            }
            "--beam-size" => {
                i += 1;
                if i < args.len() {
                    if let Ok(size) = args[i].parse::<usize>() {
                        beam_size = size.max(1);
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    if use_beam {
        DecodingStrategy::Beam(BeamConfig {
            beam_size,
            ..default_beam()
        })
    } else {
        DecodingStrategy::Greedy
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    let audio_path = if args.len() > 1 {
        &args[1]
    } else {
        "6_speakers.wav"
    };

    // Load audio manually using hound (or any other audio library)
    // remember if you use raw audio API, you need to handle audio preprocessing yourself!
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    println!(
        "Audio info: {}Hz, {} channel(s)",
        spec.sample_rate, spec.channels
    );

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    let use_tdt = args.iter().any(|a| a == "tdt");
    let flag_args: Vec<String> = args
        .iter()
        .skip_while(|a| *a != "--")
        .skip(1)
        .cloned()
        .collect();
    let exec_config = parse_execution_config(&flag_args);

    if use_tdt {
        let decoding = parse_decoding(&flag_args, BeamConfig::default_tdt);
        match decoding {
            DecodingStrategy::Greedy => println!("Loading TDT model (decoding: greedy)..."),
            DecodingStrategy::Beam(c) => println!(
                "Loading TDT model (decoding: beam, size={}, duration_reward={})...",
                c.beam_size, c.duration_reward
            ),
        }
        let mut parakeet =
            ParakeetTDT::from_pretrained_with_decoding("./tdt", exec_config.clone(), decoding)?;

        // Use transcribe_samples() with raw parameters and timestamp mode
        let result = parakeet.transcribe_samples(
            audio,
            spec.sample_rate,
            spec.channels,
            Some(TimestampMode::Sentences),
        )?;

        println!("{}", result.text);
        println!("\nSentencess:");
        for segment in result.tokens.iter() {
            println!(
                "[{:.2}s - {:.2}s]: {}",
                segment.start, segment.end, segment.text
            );
        }
    } else {
        let decoding = parse_decoding(&flag_args, BeamConfig::default_ctc);
        match decoding {
            DecodingStrategy::Greedy => println!("Loading CTC model (decoding: greedy)..."),
            DecodingStrategy::Beam(c) => println!(
                "Loading CTC model (decoding: beam, size={})...",
                c.beam_size
            ),
        }
        let mut parakeet =
            Parakeet::from_pretrained_with_decoding(".", exec_config.clone(), decoding)?;

        // CTC model doesn't predict punctuation (lowercase alphabet only)
        // This means no sentence boundaries. we use Words mode instead of Sentences
        let result = parakeet.transcribe_samples(
            audio,
            spec.sample_rate,
            spec.channels,
            Some(TimestampMode::Words),
        )?;

        println!("{}", result.text);

        // Access word-level timestamps (showing first 10 for brevity)
        // Note: CTC generates word-level timestamps but cannot segment into sentences
        // due to lack of punctuation prediction - this is a model limitation if I not mistake
        println!("\nWords (first 10):");
        for word in result.tokens.iter().take(10) {
            println!("[{:.2}s - {:.2}s]: {}", word.start, word.end, word.text);
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "\n✓ Transcription completed in {:.2}s",
        elapsed.as_secs_f32()
    );

    Ok(())
}
