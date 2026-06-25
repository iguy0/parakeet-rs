#!/usr/bin/env python3
"""Quantize Nemotron streaming ONNX exports to dynamic int8.

This script does **not** choose streaming latency — that is fixed when you run
export_nemotron_streaming.py (--right-context). Quantization only compresses
weights; graph shapes come from the fp32 export.

Input (from export_nemotron_streaming.py):

  <model_dir>/
    encoder.onnx
    encoder.onnx.data
    decoder_joint.onnx
    config.json            # used for verification; not copied to int8 bundle
    tokenizer.model

Output (HuggingFace int8 layout — same as lokkju/nemotron-speech-streaming-en-0.6b-int8):

  <output_dir>/
    encoder.onnx           # single file (~650 MB), QUInt8 dynamic quant
    decoder_joint.onnx
    tokenizer.model

Standard filenames; the directory name indicates int8 (e.g. out_onnx_int8/).

Settings
--------
Positional:
  model_dir     Directory with fp32 ONNX from export_nemotron_streaming.py
  output_dir    Directory for int8 artifacts (separate from fp32)

Options:
  --work-dir       Temp dir for intermediate quant files (default: <output_dir>/.quantize_work)
  --skip-verify    Skip fp32 vs int8 numerical comparison
  --keep-work-dir  Keep temp dir after success (debugging)
  --clean-output   Delete existing files in output_dir before writing

Streaming latency (set at export, not here)
-------------------------------------------
  ms     export --right-context   mel frames/step   samples/step @ 16 kHz
  80     0                        8                 1280
  160    1                        16                2560
  560    6 (export default)       56                8960
  1120   13                       112               17920

After export, set CHUNK_SIZE in parakeet-rs/src/nemotron.rs to mel frames/step.

Requirements: pip install onnx 'onnxruntime>=1.20'

Examples:
  python quantize_nemotron_streaming.py out_onnx/ out_onnx_int8/
  python quantize_nemotron_streaming.py out_onnx/ out_onnx_int8/ --clean-output
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

REQUIRED_ENCODER_INPUTS = {
    "processed_signal",
    "processed_signal_length",
    "cache_last_channel",
    "cache_last_time",
    "cache_last_channel_len",
}
REQUIRED_ENCODER_OUTPUTS = {
    "encoded",
    "encoded_len",
    "cache_last_channel_next",
    "cache_last_time_next",
    "cache_last_channel_len_next",
}
REQUIRED_DECODER_INPUTS = {
    "encoder_outputs",
    "targets",
    "target_length",
    "input_states_1",
    "input_states_2",
}
REQUIRED_DECODER_OUTPUTS = {
    "outputs",
    "output_states_1",
    "output_states_2",
}

DEFAULT_DECODER_LSTM_DIM = 640
DEFAULT_DECODER_LSTM_LAYERS = 2

OUTPUT_ENCODER = "encoder.onnx"
OUTPUT_DECODER = "decoder_joint.onnx"
OUTPUT_TOKENIZER = "tokenizer.model"

# Must match export_nemotron_streaming.py STREAMING_LATENCY_MS.
STREAMING_LATENCY_MS = {
    0: 80,
    1: 160,
    6: 560,
    13: 1120,
}
HOP_LENGTH_MS = 10
SAMPLE_RATE = 16000


def check_runtime_versions() -> None:
    import onnx

    ort_version = tuple(int(part) for part in ort.__version__.split(".")[:2])
    if ort_version < (1, 20):
        raise RuntimeError(
            f"onnxruntime {ort.__version__} is too old for onnx {onnx.__version__}. "
            "Install onnxruntime>=1.20 (pip install 'onnxruntime>=1.20')."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Nemotron streaming ONNX models to dynamic int8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Streaming latency is chosen in export_nemotron_streaming.py "
            "(--right-context), not in this script.\n\n"
            "  ms     --right-context   mel frames   samples @ 16 kHz\n"
            "  80     0                 8            1280\n"
            "  160    1                 16           2560\n"
            "  560    6                 56           8960\n"
            "  1120   13                112          17920\n"
        ),
    )
    parser.add_argument(
        "model_dir",
        help="Directory containing fp32 encoder.onnx, decoder_joint.onnx, and config.json",
    )
    parser.add_argument(
        "output_dir",
        help="Directory for int8 outputs (encoder.onnx, decoder_joint.onnx, tokenizer.model)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Temporary directory for intermediate quantization artifacts "
        "(default: <output_dir>/.quantize_work)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip numerical verification against the fp32 models",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the temporary work directory after completion",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing files in output_dir before writing int8 artifacts",
    )
    return parser.parse_args()


def format_size(num_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}T"


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def validate_graph_io(
    model_path: Path,
    required_inputs: set[str],
    required_outputs: set[str],
) -> None:
    model = onnx.load_model(str(model_path), load_external_data=False)
    input_names = {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}

    missing_inputs = sorted(required_inputs - input_names)
    missing_outputs = sorted(required_outputs - output_names)
    if missing_inputs or missing_outputs:
        raise RuntimeError(
            f"Validation failed for {model_path.name}: "
            f"missing inputs {missing_inputs}, missing outputs {missing_outputs}"
        )


def save_single_file_onnx(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load_model(str(src), load_external_data=True)
    remove_if_exists(dest)
    onnx.save_model(model, str(dest))


def quantize_to_tmp(
    src: Path,
    tmp_dest: Path,
    weight_type: QuantType,
    use_external_data: bool,
) -> Path:
    tmp_dest.parent.mkdir(parents=True, exist_ok=True)
    for existing in tmp_dest.parent.glob(f"{tmp_dest.name}*"):
        existing.unlink()

    quantize_dynamic(
        model_input=str(src),
        model_output=str(tmp_dest),
        weight_type=weight_type,
        use_external_data_format=use_external_data,
    )
    return tmp_dest


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_inputs(model_dir: Path) -> None:
    required = [
        model_dir / "encoder.onnx",
        model_dir / "encoder.onnx.data",
        model_dir / "decoder_joint.onnx",
        model_dir / "config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing required files in {model_dir}: {missing_names}. "
            "Run export_nemotron_streaming.py first."
        )


def validate_outputs(output_dir: Path) -> None:
    required = [
        output_dir / OUTPUT_ENCODER,
        output_dir / OUTPUT_DECODER,
        output_dir / OUTPUT_TOKENIZER,
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Missing quantized artifact: {path}")

    validate_graph_io(
        output_dir / OUTPUT_ENCODER,
        REQUIRED_ENCODER_INPUTS,
        REQUIRED_ENCODER_OUTPUTS,
    )
    validate_graph_io(
        output_dir / OUTPUT_DECODER,
        REQUIRED_DECODER_INPUTS,
        REQUIRED_DECODER_OUTPUTS,
    )


def copy_tokenizer(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    src = model_dir / OUTPUT_TOKENIZER
    if not src.exists():
        raise FileNotFoundError(f"Missing {OUTPUT_TOKENIZER} in {model_dir}")
    shutil.copy2(src, output_dir / OUTPUT_TOKENIZER)


def describe_export_config(config: dict) -> None:
    right_context = config.get("right_context")
    left_context = config.get("left_context")
    chunk_frames = config.get("chunk_size_output_frames")
    parts = []
    if left_context is not None:
        parts.append(f"left_context={left_context}")
    if right_context is not None:
        parts.append(f"right_context={right_context}")
        preset_ms = STREAMING_LATENCY_MS.get(right_context)
        if preset_ms is not None:
            mel_frames = preset_ms // HOP_LENGTH_MS
            samples = mel_frames * (SAMPLE_RATE // 1000) * HOP_LENGTH_MS
            parts.append(f"latency={preset_ms} ms")
            parts.append(f"mel_frames/step={mel_frames}")
            parts.append(f"samples/step={samples}")
        elif chunk_frames is not None:
            approx_ms = chunk_frames * 80
            parts.append(f"~latency={approx_ms} ms (custom)")
    if parts:
        print(f"  Export settings: {', '.join(parts)}")
        if right_context == 13:
            print("  Note: parakeet-rs CHUNK_SIZE must be 112 for this export.")
        elif right_context == 6:
            print("  Note: parakeet-rs CHUNK_SIZE must be 56 for this export.")


def build_encoder_feeds(config: dict) -> dict[str, np.ndarray]:
    mel_shape = config["test_input"]["mel_shape"]
    mel_length = config["test_input"]["mel_length"]
    cache_shapes = config["cache_shapes"]

    return {
        "processed_signal": np.random.randn(*mel_shape).astype(np.float32) * 0.1,
        "processed_signal_length": np.array([mel_length], dtype=np.int64),
        "cache_last_channel": np.zeros(cache_shapes["cache_last_channel"], dtype=np.float32),
        "cache_last_time": np.zeros(cache_shapes["cache_last_time"], dtype=np.float32),
        "cache_last_channel_len": np.zeros(
            cache_shapes["cache_last_channel_len"], dtype=np.int64
        ),
    }


def build_decoder_feeds(
    encoder_frame: np.ndarray,
    blank_id: int,
    lstm_dim: int = DEFAULT_DECODER_LSTM_DIM,
    lstm_layers: int = DEFAULT_DECODER_LSTM_LAYERS,
) -> dict[str, np.ndarray]:
    return {
        "encoder_outputs": encoder_frame.astype(np.float32),
        "targets": np.array([[blank_id]], dtype=np.int32),
        "target_length": np.array([1], dtype=np.int32),
        "input_states_1": np.zeros((lstm_layers, 1, lstm_dim), dtype=np.float32),
        "input_states_2": np.zeros((lstm_layers, 1, lstm_dim), dtype=np.float32),
    }


def compare_outputs(
    name: str,
    fp32_outputs: list[np.ndarray],
    int8_outputs: list[np.ndarray],
    output_names: list[str],
) -> None:
    print(f"\n  {name} numerical comparison (fp32 vs int8):")
    for out_name, fp32_arr, int8_arr in zip(output_names, fp32_outputs, int8_outputs):
        if fp32_arr.shape != int8_arr.shape:
            raise RuntimeError(
                f"{name} output {out_name}: shape mismatch "
                f"fp32={fp32_arr.shape}, int8={int8_arr.shape}"
            )
        diff = float(np.max(np.abs(fp32_arr - int8_arr)))
        mean_diff = float(np.mean(np.abs(fp32_arr - int8_arr)))
        print(f"    {out_name:30s} max={diff:.6f}  mean={mean_diff:.6f}")


def run_session(model_path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    output_names = [output.name for output in session.get_outputs()]
    return session.run(output_names, feeds)


def verify_quantization(model_dir: Path, output_dir: Path, config: dict) -> None:
    print("\nVerifying quantized models against fp32 references...")

    encoder_feeds = build_encoder_feeds(config)
    fp32_encoder_out = run_session(model_dir / "encoder.onnx", encoder_feeds)
    int8_encoder_out = run_session(output_dir / OUTPUT_ENCODER, encoder_feeds)

    encoder_output_names = [
        "encoded",
        "encoded_len",
        "cache_last_channel_next",
        "cache_last_time_next",
        "cache_last_channel_len_next",
    ]
    compare_outputs("Encoder", fp32_encoder_out, int8_encoder_out, encoder_output_names)

    # Encoder output is [batch, hidden_dim, time]; decoder expects one frame.
    encoded_frame = fp32_encoder_out[0][:, :, :1]

    blank_id = config.get("blank_id", config.get("vocab_size", 1024))
    decoder_feeds = build_decoder_feeds(encoded_frame, blank_id=blank_id)

    fp32_decoder_out = run_session(model_dir / "decoder_joint.onnx", decoder_feeds)
    int8_decoder_out = run_session(output_dir / OUTPUT_DECODER, decoder_feeds)

    decoder_output_names = ["outputs", "output_states_1", "output_states_2"]
    compare_outputs("Decoder", fp32_decoder_out, int8_decoder_out, decoder_output_names)

    logits_diff = float(np.max(np.abs(fp32_decoder_out[0] - int8_decoder_out[0])))
    if logits_diff > 1.0:
        print(
            f"\n  Warning: decoder logits max diff is {logits_diff:.4f}. "
            "This can happen with dynamic int8; run end-to-end WER checks before deployment."
        )
    else:
        print("\n  Verification passed (graph I/O preserved, outputs within expected tolerance).")


def print_summary(model_dir: Path, output_dir: Path) -> None:
    print(f"\nQuantization complete:")
    print(f"  fp32 source : {model_dir}")
    print(f"  int8 output : {output_dir}")
    artifact_names = [OUTPUT_ENCODER, OUTPUT_DECODER, OUTPUT_TOKENIZER]
    for artifact_name in artifact_names:
        path = output_dir / artifact_name
        if path.exists():
            print(f"    {artifact_name:<24} {format_size(path.stat().st_size)}")


def quantize_models(model_dir: Path, output_dir: Path, work_dir: Path) -> None:
    encoder_fp32 = model_dir / "encoder.onnx"
    decoder_fp32 = model_dir / "decoder_joint.onnx"

    encoder_tmp = work_dir / "encoder.int8.tmp.onnx"
    decoder_tmp = work_dir / "decoder_joint.int8.tmp.onnx"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Quantizing encoder to int8 (dynamic, QUInt8 weights)...")
    quantize_to_tmp(
        encoder_fp32,
        encoder_tmp,
        QuantType.QUInt8,
        use_external_data=True,
    )
    save_single_file_onnx(
        encoder_tmp,
        output_dir / OUTPUT_ENCODER,
    )

    print("Quantizing decoder_joint to int8 (dynamic, QInt8 weights)...")
    quantize_to_tmp(
        decoder_fp32,
        decoder_tmp,
        QuantType.QInt8,
        use_external_data=False,
    )
    save_single_file_onnx(decoder_tmp, output_dir / OUTPUT_DECODER)

    print(f"Copying {OUTPUT_TOKENIZER}...")
    copy_tokenizer(model_dir, output_dir)


def main() -> None:
    args = parse_args()
    check_runtime_versions()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output_dir / ".quantize_work"
    )

    validate_inputs(model_dir)
    config = load_config(model_dir)
    describe_export_config(config)

    if args.clean_output and output_dir.exists():
        print(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    ensure_clean_dir(work_dir)
    try:
        quantize_models(model_dir, output_dir, work_dir)
        validate_outputs(output_dir)

        if not args.skip_verify:
            verify_quantization(model_dir, output_dir, config)

        print_summary(model_dir, output_dir)
    finally:
        if not args.keep_work_dir and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
