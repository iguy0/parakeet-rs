/*
Speaker Diarization with NVIDIA Sortformer v2 (Streaming)

Download the Sortformer v2 model:
https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.onnx
Or download the Sortformer v2.1 model:
https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.1.onnx
Download test audio:
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

Usage:
cargo run --example streaming-diarization --features sortformer <audio.wav> [model.onnx] [config]

Configs: auto (default), callhome, dihard3, ultra-8spk, sensitive

8-speaker example:
cargo run --example streaming-diarization --features sortformer 6_speakers.wav \\
  ../ultra_diar_streaming_sortformer_8spk_v1_onnx/ultra_diar_streaming_sortformer_8spk_v1.onnx \\
  ultra-8spk
*/

#[cfg(feature = "sortformer")]
use hound;
#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
#[cfg(feature = "sortformer")]
use std::env;
#[cfg(feature = "sortformer")]
use std::time::Instant;

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("Error: This example requires the 'sortformer' feature.");
        eprintln!("Run with: cargo run --example streaming_diarization --features sortformer <audio.wav>");
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let start_time = Instant::now();
        let args: Vec<String> = env::args().collect();
        let audio_path = args.get(1).expect(
            "Usage: streaming-diarization <audio.wav> [model.onnx]",
        );
        let model_path = args
            .get(2)
            .map(String::as_str)
            .unwrap_or("diar_streaming_sortformer_4spk-v2.1.onnx");
        let config_label = args
            .get(3)
            .filter(|s| *s != "--full")
            .map(|s| s.as_str())
            .unwrap_or("auto");
        let use_full_file = args.iter().any(|s| s == "--full");
        let diar_config = match config_label {
            "callhome" => DiarizationConfig::callhome(),
            "dihard3" => DiarizationConfig::dihard3(),
            "ultra-8spk" | "8spk" => DiarizationConfig::ultra_8spk(),
            "sensitive" => {
                let mut cfg = DiarizationConfig::custom(0.42, 0.32);
                cfg.min_duration_on = 0.15;
                cfg.min_duration_off = 0.10;
                cfg.median_window = 7;
                cfg
            }
            "auto" => {
                if model_path.contains("8spk") {
                    DiarizationConfig::ultra_8spk()
                } else {
                    DiarizationConfig::callhome()
                }
            }
            other => {
                return Err(format!(
                    "Unknown config '{other}'. Use: auto, callhome, dihard3, ultra-8spk, sensitive"
                )
                .into());
            }
        };

        // Load audio
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

        // Create Sortformer
        let mut sortformer = Sortformer::with_config(model_path, None, diar_config)?;

        println!(
            "Model: {} | config={} | mode={} | speakers={} chunk_len={} right_context={} latency={:.2}s",
            model_path,
            config_label,
            if use_full_file { "full" } else { "streaming" },
            sortformer.num_speakers(),
            sortformer.chunk_len,
            sortformer.right_context,
            sortformer.latency()
        );

        // simulate real-time streaming: feed small chunks (for instance 20ms = 320 samples)
        // In practice, real world these would come from gsttreamer, mic etc ofc
        let feed_chunk_size = 320; // 20ms at 16kHz
        let mut total_segments = 0;

        if use_full_file {
            println!("\nFull-file diarization (diarize):");
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
            println!("\nStreaming diarization (feeding {}ms chunks):", feed_chunk_size * 1000 / 16_000);
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
