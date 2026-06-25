#!/usr/bin/env python3
"""Export ultra_diar_streaming_sortformer_8spk_v1 to ONNX.

Extends export_diar_sortformer.py with 8-speaker defaults and ORT verification.

Recommended streaming params (model README):
  chunk_len=340, right_context=40, fifo_len=40,
  spkcache_update_period=300, spkcache_len=376

Usage:
  python export_ultra_diar_8spk.py model.nemo output.onnx
  python export_ultra_diar_8spk.py model.nemo output.onnx --skip-verify

Requirements: nemo_toolkit[asr], torch, onnx, onnxruntime
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import torch

# 8spk inference defaults from ultra_diar_streaming_sortformer_8spk_v1 README
DEFAULT_CHUNK_LEN = 340
DEFAULT_RIGHT_CONTEXT = 40
DEFAULT_FIFO_LEN = 40
DEFAULT_SPKCACHE_UPDATE_PERIOD = 300
DEFAULT_SPKCACHE_LEN = 376
DEFAULT_NUM_SPEAKERS = 8

FEAT_DIM = 128
EMB_DIM = 512
SUBSAMPLING = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ultra_diar_streaming_sortformer_8spk_v1 to ONNX",
    )
    parser.add_argument("input_path", help="Path to .nemo checkpoint")
    parser.add_argument("output_path", help="Path for exported .onnx file")
    parser.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN)
    parser.add_argument("--right-context", type=int, default=DEFAULT_RIGHT_CONTEXT)
    parser.add_argument("--fifo-len", type=int, default=DEFAULT_FIFO_LEN)
    parser.add_argument(
        "--spkcache-update-period",
        type=int,
        default=DEFAULT_SPKCACHE_UPDATE_PERIOD,
    )
    parser.add_argument("--spkcache-len", type=int, default=DEFAULT_SPKCACHE_LEN)
    parser.add_argument("--num-speakers", type=int, default=DEFAULT_NUM_SPEAKERS)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use strict=True when loading .nemo (default: False)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX Runtime inference smoke test",
    )
    return parser.parse_args()


def onnx_forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
    chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(
        x=chunk, lengths=chunk_lengths
    )
    chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

    concat_embs = torch.cat([spkcache, fifo, chunk_pre_encode_embs], dim=1)
    concat_lens = spkcache_lengths + fifo_lengths + chunk_pre_encode_lengths

    spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = (
        self.frontend_encoder(
            processed_signal=concat_embs,
            processed_signal_length=concat_lens,
            bypass_pre_encode=True,
        )
    )

    spkcache_fifo_chunk_preds = self.forward_infer(
        spkcache_fifo_chunk_fc_encoder_embs,
        spkcache_fifo_chunk_fc_encoder_lengths,
    )

    return spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths


def export_model(args: argparse.Namespace) -> Path:
    from nemo.collections.asr.models import SortformerEncLabelModel

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Loading model from: {args.input_path}")

    model = SortformerEncLabelModel.restore_from(
        restore_path=args.input_path,
        map_location="cpu",
        strict=args.strict,
    )
    model.eval()

    sm = model.sortformer_modules
    sm.chunk_len = args.chunk_len
    sm.chunk_right_context = args.right_context
    sm.fifo_len = args.fifo_len
    sm.spkcache_update_period = args.spkcache_update_period
    sm.spkcache_len = args.spkcache_len
    sm._check_streaming_parameters()

    print(
        "Streaming config:",
        f"chunk_len={sm.chunk_len}",
        f"right_context={sm.chunk_right_context}",
        f"fifo_len={sm.fifo_len}",
        f"spkcache_len={sm.spkcache_len}",
        f"spkcache_update_period={sm.spkcache_update_period}",
        f"num_speakers={args.num_speakers}",
    )

    model.forward = types.MethodType(onnx_forward, model)

    mel_frames = (args.chunk_len + args.right_context) * SUBSAMPLING
    batch_size = 1

    chunk = torch.randn(batch_size, mel_frames, FEAT_DIM)
    chunk_lengths = torch.tensor([mel_frames], dtype=torch.long)
    spkcache = torch.zeros(batch_size, args.spkcache_len, EMB_DIM)
    spkcache_lengths = torch.tensor([0], dtype=torch.long)
    fifo = torch.zeros(batch_size, args.fifo_len, EMB_DIM)
    fifo_lengths = torch.tensor([0], dtype=torch.long)

    input_example = (
        chunk,
        chunk_lengths,
        spkcache,
        spkcache_lengths,
        fifo,
        fifo_lengths,
    )

    print(f"  chunk:    {tuple(chunk.shape)}")
    print(f"  spkcache: {tuple(spkcache.shape)}")
    print(f"  fifo:     {tuple(fifo.shape)}")

    torch.onnx.export(
        model,
        input_example,
        str(output_path),
        input_names=[
            "chunk",
            "chunk_lengths",
            "spkcache",
            "spkcache_lengths",
            "fifo",
            "fifo_lengths",
        ],
        output_names=[
            "spkcache_fifo_chunk_preds",
            "chunk_pre_encode_embs",
            "chunk_pre_encode_lengths",
        ],
        dynamic_axes={
            "chunk": {0: "batch", 1: "time_chunk"},
            "spkcache": {0: "batch", 1: "time_cache"},
            "fifo": {0: "batch", 1: "time_fifo"},
            "spkcache_fifo_chunk_preds": {0: "batch", 1: "time_out"},
            "chunk_pre_encode_embs": {0: "batch", 1: "time_pre_encode"},
        },
        opset_version=17,
        dynamo=False,
        verbose=False,
    )

    model_onnx = onnx.load(str(output_path))
    print("\nONNX inputs:")
    for inp in model_onnx.graph.input:
        dims = [
            d.dim_param if d.dim_param else d.dim_value
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"  {inp.name}: {dims}")

    print("\nONNX outputs:")
    for out in model_onnx.graph.output:
        dims = [
            d.dim_param if d.dim_param else d.dim_value
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"  {out.name}: {dims}")

    metadata = {
        "chunk_len": str(args.chunk_len),
        "fifo_len": str(args.fifo_len),
        "spkcache_len": str(args.spkcache_len),
        "right_context": str(args.right_context),
        "num_speakers": str(args.num_speakers),
        "model": "ultra_diar_streaming_sortformer_8spk_v1",
    }
    del model_onnx.metadata_props[:]
    for key, value in metadata.items():
        model_onnx.metadata_props.append(
            onnx.StringStringEntryProto(key=key, value=value)
        )

    onnx.save(model_onnx, str(output_path))
    print(f"\nExported to: {output_path}")
    print("Metadata:", metadata)
    return output_path


def verify_onnx(
    onnx_path: Path,
    args: argparse.Namespace,
) -> None:
    import onnxruntime as ort

    mel_frames = (args.chunk_len + args.right_context) * SUBSAMPLING
    batch_size = 1

    chunk = np.random.randn(batch_size, mel_frames, FEAT_DIM).astype(np.float32)
    chunk_lengths = np.array([mel_frames], dtype=np.int64)
    spkcache = np.zeros((batch_size, args.spkcache_len, EMB_DIM), dtype=np.float32)
    spkcache_lengths = np.array([0], dtype=np.int64)
    fifo = np.zeros((batch_size, args.fifo_len, EMB_DIM), dtype=np.float32)
    fifo_lengths = np.array([0], dtype=np.int64)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    outputs = session.run(
        None,
        {
            "chunk": chunk,
            "chunk_lengths": chunk_lengths,
            "spkcache": spkcache,
            "spkcache_lengths": spkcache_lengths,
            "fifo": fifo,
            "fifo_lengths": fifo_lengths,
        },
    )

    preds, pre_encode_embs, pre_encode_lengths = outputs
    expected_pre_encode_frames = mel_frames // SUBSAMPLING

    print("\nORT verification:")
    print(f"  spkcache_fifo_chunk_preds: {preds.shape}")
    print(f"  chunk_pre_encode_embs:     {pre_encode_embs.shape}")
    print(f"  chunk_pre_encode_lengths:  {pre_encode_lengths}")

    if preds.ndim != 3:
        raise RuntimeError(f"Expected 3D preds, got shape {preds.shape}")
    if preds.shape[-1] != args.num_speakers:
        raise RuntimeError(
            f"Expected {args.num_speakers} speaker outputs, got {preds.shape[-1]}"
        )
    if not np.isfinite(preds).all():
        raise RuntimeError("Predictions contain NaN or Inf")
    if preds.shape[0] != batch_size:
        raise RuntimeError(f"Unexpected batch size in preds: {preds.shape[0]}")

    if pre_encode_embs.shape[-1] != EMB_DIM:
        raise RuntimeError(
            f"Expected embedding dim {EMB_DIM}, got {pre_encode_embs.shape[-1]}"
        )
    if int(pre_encode_lengths[0]) != expected_pre_encode_frames:
        raise RuntimeError(
            "Unexpected pre_encode length: "
            f"{int(pre_encode_lengths[0])} != {expected_pre_encode_frames}"
        )

    # Sigmoid outputs should be in [0, 1] for BCE diarization head
    if preds.min() < -0.01 or preds.max() > 1.01:
        print(
            f"  Warning: preds range [{preds.min():.4f}, {preds.max():.4f}] "
            "outside typical [0, 1]"
        )
    else:
        print(f"  preds range: [{preds.min():.4f}, {preds.max():.4f}]")

    print("  ORT smoke test passed")


def main() -> int:
    args = parse_args()
    output_path = export_model(args)
    if not args.skip_verify:
        verify_onnx(output_path, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
