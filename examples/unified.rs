/*
Unified model: offline + buffered streaming transcription

Offline:
cargo run --release --example unified 6_speakers.wav
cargo run --release --example unified 6_speakers.wav -- --decoding beam --beam-size 5

Streaming:
cargo run --release --example unified 6_speakers.wav streaming
cargo run --release --example unified 6_speakers.wav streaming -- --decoding beam

---

Download model from: https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx
Files: encoder.onnx, encoder.onnx.data, decoder_joint.onnx, tokenizer.model
Place in: ./unified/
*/

use parakeet_rs::{BeamConfig, DecodingStrategy, ParakeetUnified, TimestampMode, Transcriber};
use std::env;
use std::io::Write;
use std::time::Instant;

struct CliOptions {
    audio_path: String,
    streaming: bool,
    decoding: DecodingStrategy,
}

fn parse_cli(args: &[String]) -> CliOptions {
    let mut audio_path = "6_speakers.wav".to_string();
    let mut streaming = false;
    let mut use_beam = false;
    let mut beam_size = BeamConfig::default_unified().beam_size;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "streaming" => streaming = true,
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
            flag if flag.starts_with('-') => {}
            path => audio_path = path.to_string(),
        }
        i += 1;
    }

    let decoding = if use_beam {
        DecodingStrategy::Beam(BeamConfig {
            beam_size,
            ..BeamConfig::default_unified()
        })
    } else {
        DecodingStrategy::Greedy
    };

    CliOptions {
        audio_path,
        streaming,
        decoding,
    }
}

fn decoding_label(decoding: DecodingStrategy) -> String {
    match decoding {
        DecodingStrategy::Greedy => "greedy".to_string(),
        DecodingStrategy::Beam(config) => format!("beam (size={})", config.beam_size),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    let cli = parse_cli(&args);

    // Load audio
    let mut reader = hound::WavReader::open(&cli.audio_path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz, got {}Hz", spec.sample_rate).into());
    }

    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }

    let duration = audio.len() as f32 / 16000.0;
    println!("Audio: {:.1}s, {}Hz, {} ch", duration, spec.sample_rate, spec.channels);

    let mut model = ParakeetUnified::from_pretrained_with_decoding(
        "./unified",
        None,
        Default::default(),
        cli.decoding,
    )?;
    let load_time = start_time.elapsed();
    println!(
        "Model loaded in {:.2}s (decoding: {})",
        load_time.as_secs_f32(),
        decoding_label(cli.decoding)
    );

    if cli.streaming {
        let config = model.streaming_config();
        let chunk_size = config.chunk_samples();
        println!("Streaming mode: {:.0}ms chunks", config.chunk_secs * 1000.0);

        let transcribe_start = Instant::now();
        print!("Streaming: ");

        for chunk in audio.chunks(chunk_size) {
            let text = model.transcribe_chunk(chunk)?;
            if !text.is_empty() {
                print!("{}", text);
                std::io::stdout().flush()?;
            }
        }

        let remaining = model.flush()?;
        if !remaining.is_empty() {
            print!("{}", remaining);
        }

        let result = model.get_timed_transcript(TimestampMode::Sentences);
        println!("\n\nFinal: {}", result.text);

        println!("\nSentences:");
        for segment in &result.tokens {
            println!("[{:.2}s - {:.2}s]: {}", segment.start, segment.end, segment.text);
        }

        let elapsed = transcribe_start.elapsed();
        println!(
            "Transcribed in {:.2}s (audio: {:.1}s, RTF: {:.2}x)",
            elapsed.as_secs_f32(),
            duration,
            duration / elapsed.as_secs_f32()
        );
    } else {
        println!("Offline mode (with word timestamps)");

        let transcribe_start = Instant::now();
        let result = model.transcribe_samples(
            audio,
            spec.sample_rate,
            spec.channels,
            Some(TimestampMode::Words),
        )?;
        let elapsed = transcribe_start.elapsed();

        println!("Result: {}", result.text);

        println!("\nWords (first 20):");
        for word in result.tokens.iter().take(20) {
            println!("[{:.2}s - {:.2}s]: {}", word.start, word.end, word.text);
        }

        println!(
            "Transcribed in {:.2}s (audio: {:.1}s, RTF: {:.2}x)",
            elapsed.as_secs_f32(),
            duration,
            duration / elapsed.as_secs_f32()
        );
    }

    Ok(())
}
