/*
Ultra Sortformer 8-speaker streaming diarization

Download the model (~492 MB fp32):
https://huggingface.co/investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx

  huggingface-cli download investguy/ultra_diar_streaming_sortformer_8spk_v1_onnx \
    --local-dir models/ultra_diar_streaming_sortformer_8spk_v1_onnx

Download test audio (6 speakers):
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

Usage:
cargo run --example diarization-8spk --features sortformer 6_speakers.wav
cargo run --example diarization-8spk --features sortformer 6_speakers.wav path/to/model.onnx
cargo run --example diarization-8spk --features sortformer 6_speakers.wav path/to/model.onnx --full
*/

#[cfg(feature = "sortformer")]
use hound;
#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
#[cfg(feature = "sortformer")]
use std::env;
#[cfg(feature = "sortformer")]
use std::time::Instant;

const DEFAULT_MODEL: &str =
    "models/ultra_diar_streaming_sortformer_8spk_v1_onnx/ultra_diar_streaming_sortformer_8spk_v1.onnx";

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("Error: This example requires the 'sortformer' feature.");
        eprintln!(
            "Run with: cargo run --example diarization-8spk --features sortformer <audio.wav>"
        );
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let start_time = Instant::now();
        let args: Vec<String> = env::args().collect();
        let audio_path = args
            .get(1)
            .expect("Usage: diarization-8spk <audio.wav> [model.onnx] [--full]");
        let model_path = args
            .get(2)
            .filter(|s| *s != "--full")
            .map(String::as_str)
            .unwrap_or(DEFAULT_MODEL);
        let use_full_file = args.iter().any(|s| s == "--full");

        let mut reader = hound::WavReader::open(audio_path)?;
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

        let duration = audio.len() as f32 / 16_000.0;
        println!("Loaded {:.1}s of audio", duration);

        let mut sortformer = Sortformer::with_config(
            model_path,
            None,
            DiarizationConfig::ultra_8spk(),
        )?;

        println!(
            "Ultra Sortformer 8spk | speakers={} | chunk_len={} | right_context={} | latency={:.2}s | mode={}",
            sortformer.num_speakers(),
            sortformer.chunk_len,
            sortformer.right_context,
            sortformer.latency(),
            if use_full_file { "full" } else { "streaming" }
        );

        let feed_chunk_size = 320; // 20 ms at 16 kHz
        let mut total_segments = 0;

        if use_full_file {
            println!("\nFull-file diarization:");
            println!("{}", "-".repeat(60));
            let segments = sortformer.diarize(audio, 16_000, 1)?;
            for seg in &segments {
                println!(
                    "  [{:06.2}s - {:06.2}s] Speaker {}",
                    seg.start as f64 / 16_000.0,
                    seg.end as f64 / 16_000.0,
                    seg.speaker_id
                );
            }
            total_segments = segments.len();
        } else {
            println!(
                "\nStreaming diarization (feeding {}ms chunks):",
                feed_chunk_size * 1000 / 16_000
            );
            println!("{}", "-".repeat(60));

            for chunk in audio.chunks(feed_chunk_size) {
                let segments = sortformer.feed(chunk)?;
                for seg in &segments {
                    println!(
                        "  [{:06.2}s - {:06.2}s] Speaker {}",
                        seg.start as f64 / 16_000.0,
                        seg.end as f64 / 16_000.0,
                        seg.speaker_id
                    );
                }
                total_segments += segments.len();
            }

            let final_segments = sortformer.flush()?;
            for seg in &final_segments {
                println!(
                    "  [{:06.2}s - {:06.2}s] Speaker {} (flush)",
                    seg.start as f64 / 16_000.0,
                    seg.end as f64 / 16_000.0,
                    seg.speaker_id
                );
            }
            total_segments += final_segments.len();
        }

        println!("{}", "-".repeat(60));
        println!(
            "Done: {} segments in {:.2}s",
            total_segments,
            start_time.elapsed().as_secs_f32()
        );

        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
