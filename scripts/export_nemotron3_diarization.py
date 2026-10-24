#!/usr/bin/env python3
"""Export nvidia/Nemotron-3-Diarization (8-speaker streaming Sortformer) to ONNX.

The exported graph keeps the same 6-input / 3-output contract as the other
Sortformer exports (see export_diar_sortformer.py / export_ultra_diar_8spk.py),
so `parakeet_rs::sortformer::Sortformer` loads it without a new code path:

  inputs : chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths
  outputs: spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths

The checkpoint is a high-resolution model (10 ms predictions); NeMo's
`forward_for_export` average-pools them back to 80 ms frames inside the graph.

Extra ONNX metadata_props (all optional for the Rust runtime):
  num_speakers, chunk_len, fifo_len, spkcache_len, right_context   (existing keys)
  spkcache_sil_frames_per_spk, scores_boost_latest                 (cache compression)
  sil_emb  comma-separated learnable silence embedding (fc_d_model floats);
           replaces the running-mean silence embedding when present

Streaming presets (frames are 80 ms; latency = (chunk_len + right_context) * 80 ms,
excluding compute). spkcache_len=264 for every preset.

  preset    latency  chunk_len  right_context  fifo_len  update_period
  offline   30.4 s      340          40            40         300
  low        1.04 s       9            4           264         222
  verylow    0.64 s       6            2           264         222
  ultralow   0.32 s       3            1           264         222

Requirements: Python >= 3.12, torch, onnx, onnxruntime, and a NeMo Speech build
that registers TransformerEncoder with RoPE (NeMo `main`; neither nemo-toolkit
2.7.3 nor the 3.0.0 PyPI release can restore this checkpoint). Point PYTHONPATH
at such a checkout instead of upgrading your environment, for example:

  PYTHONPATH=/path/to/NeMo-Speech python export_nemotron3_diarization.py \\
      Nemotron-3-Diarization.nemo out/nemotron3_diar_offline.onnx --preset offline

INT8 (optional): --int8-output PATH writes a dynamic-INT8 copy (QInt8 weights) of the
fp32 graph. Only the weight MatMuls inside the 31 encoder layers are quantized
(w_qkv, out_proj, FFN); the attention score matmuls, pre_encode projection,
encoder_proj and the speaker head stay fp32, because the encoder embeddings are
fed back into the speaker cache / FIFO and quantization noise there accumulates
across chunks. --compare-wav WAV (16 kHz mono) then reports fp32-vs-INT8 agreement
on real speech. Requires onnxruntime >= 1.20.

Usage:
  python export_nemotron3_diarization.py model.nemo output.onnx [--preset NAME]
  python export_nemotron3_diarization.py model.nemo fp32.onnx --int8-output int8.onnx \\
      --compare-wav samples/smoke/test01_20s.wav
  python export_nemotron3_diarization.py model.nemo output.onnx --chunk-len 9 \\
      --right-context 4 --fifo-len 264 --spkcache-update-period 222
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import torch

PRESETS = {
    # name: (chunk_len, right_context, fifo_len, spkcache_update_period)
    "offline": (340, 40, 40, 300),
    "low": (9, 4, 264, 222),
    "verylow": (6, 2, 264, 222),
    "ultralow": (3, 1, 264, 222),
}
DEFAULT_SPKCACHE_LEN = 264
FEAT_DIM = 128
EMB_DIM = 512
SUBSAMPLING = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Nemotron-3-Diarization to ONNX")
    p.add_argument("input_path", help="Path to Nemotron-3-Diarization.nemo")
    p.add_argument("output_path", help="Path for exported .onnx file")
    p.add_argument("--preset", choices=sorted(PRESETS), default="offline")
    p.add_argument("--chunk-len", type=int)
    p.add_argument("--right-context", type=int)
    p.add_argument("--fifo-len", type=int)
    p.add_argument("--spkcache-update-period", type=int)
    p.add_argument("--spkcache-len", type=int, default=DEFAULT_SPKCACHE_LEN)
    p.add_argument("--skip-verify", action="store_true", help="Skip ORT-vs-PyTorch check")
    p.add_argument("--int8-output", help="Also write a dynamic-INT8 copy to this path")
    p.add_argument("--compare-wav", help="16 kHz mono WAV for the fp32-vs-INT8 comparison")
    args = p.parse_args()
    chunk, right, fifo, period = PRESETS[args.preset]
    args.chunk_len = args.chunk_len if args.chunk_len is not None else chunk
    args.right_context = args.right_context if args.right_context is not None else right
    args.fifo_len = args.fifo_len if args.fifo_len is not None else fifo
    if args.spkcache_update_period is None:
        args.spkcache_update_period = period
    return args


def install_onnx_attention_shims() -> None:
    """Replace NeMo's FlexAttention path with ONNX-traceable equivalents.

    NeMo's TransformerEncoder builds a FlexAttention `block_mask` and calls
    `flex_attention`; neither can be traced to ONNX. For attn_mode="full" the
    mask is only key padding (kv_idx < length[b]), so it is materialised as a
    dense boolean mask and attention runs as scaled_dot_product_attention. The
    weights, RoPE and every other op are untouched.
    """
    import torch.nn.functional as F
    from nemo.collections.asr.modules import transformer_encoder as te

    def dense_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device=None, **_):
        b = torch.arange(B, device=device).view(B, 1, 1, 1)
        h = torch.zeros(1, 1, 1, 1, dtype=torch.long, device=device)
        q = torch.arange(Q_LEN, device=device).view(1, 1, Q_LEN, 1)
        kv = torch.arange(KV_LEN, device=device).view(1, 1, 1, KV_LEN)
        return mask_mod(b, h, q, kv).expand(B, 1, Q_LEN, KV_LEN)

    def sdpa(q, k, v, block_mask=None, score_mod=None, **_):
        if score_mod is not None:
            raise NotImplementedError("score_mod (rel_pos) is not used by this model")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)

    te.create_block_mask = dense_block_mask
    te._transformer_utils._get_flex_attention = lambda x: sdpa


def load_model(args: argparse.Namespace):
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(
        restore_path=args.input_path, map_location="cpu", strict=True
    )
    model.eval()
    sm = model.sortformer_modules
    sm.chunk_len = args.chunk_len
    sm.chunk_right_context = args.right_context
    sm.fifo_len = args.fifo_len
    sm.spkcache_update_period = args.spkcache_update_period
    sm.spkcache_len = args.spkcache_len
    sm._check_streaming_parameters()
    return model


def example_inputs(args: argparse.Namespace, seed: int = 0):
    """Non-trivial example: half-full speaker cache and FIFO."""
    g = torch.Generator().manual_seed(seed)
    mel_frames = (args.chunk_len + args.right_context) * SUBSAMPLING
    chunk = torch.randn(1, mel_frames, FEAT_DIM, generator=g)
    chunk_lengths = torch.tensor([mel_frames], dtype=torch.long)
    spkcache = torch.randn(1, args.spkcache_len, EMB_DIM, generator=g)
    spkcache_lengths = torch.tensor([args.spkcache_len // 2], dtype=torch.long)
    fifo = torch.randn(1, args.fifo_len, EMB_DIM, generator=g)
    fifo_lengths = torch.tensor([args.fifo_len // 2], dtype=torch.long)
    return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths


def export_model(args: argparse.Namespace, model) -> Path:
    import onnx as _onnx

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sm = model.sortformer_modules
    num_speakers = int(sm.n_spk)

    print(
        "Streaming config:",
        f"chunk_len={sm.chunk_len}",
        f"right_context={sm.chunk_right_context}",
        f"fifo_len={sm.fifo_len}",
        f"spkcache_len={sm.spkcache_len}",
        f"update_period={sm.spkcache_update_period}",
        f"num_speakers={num_speakers}",
        f"high_resolution={model.high_resolution}",
    )

    # Reference outputs from NeMo's own (FlexAttention) path, before patching.
    refs = {}
    if not args.skip_verify:
        for seed in (1, 2):
            inputs = example_inputs(args, seed=seed)
            with torch.no_grad():
                refs[seed] = [t.cpu().numpy() for t in model.forward_for_export(*inputs)]
    print("encoder attn_mode:", model.encoder.attn_mode,
          "causal_tail_len:", getattr(model.encoder, "causal_tail_len", None))
    install_onnx_attention_shims()

    # Trace the NeMo export forward directly (same 6-in / 3-out contract).
    model.forward = types.MethodType(
        lambda self, *a: self.forward_for_export(*a), model
    )
    inputs = example_inputs(args)

    torch.onnx.export(
        model,
        inputs,
        str(out),
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

    metadata = {
        "model": "Nemotron-3-Diarization",
        "chunk_len": str(args.chunk_len),
        "fifo_len": str(args.fifo_len),
        "spkcache_len": str(args.spkcache_len),
        "right_context": str(args.right_context),
        "num_speakers": str(num_speakers),
        "spkcache_sil_frames_per_spk": str(int(sm.spkcache_sil_frames_per_spk)),
        "scores_boost_latest": str(float(sm.scores_boost_latest)),
    }
    if getattr(sm, "use_learnable_sil_emb", False):
        sil = sm.learnable_sil_emb.detach().cpu().float().numpy()
        metadata["sil_emb"] = ",".join(f"{v:.9g}" for v in sil)

    # Keep the weights in an external-data file if the graph is large
    proto = _onnx.load(str(out))
    del proto.metadata_props[:]
    for k, v in metadata.items():
        proto.metadata_props.append(_onnx.StringStringEntryProto(key=k, value=v))
    _onnx.save(proto, str(out))

    print("ONNX inputs :", [i.name for i in proto.graph.input])
    print("ONNX outputs:", [o.name for o in proto.graph.output])
    print("Metadata keys:", sorted(metadata))
    print(f"Exported to: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out, refs


def verify_onnx(onnx_path: Path, args: argparse.Namespace, model, refs) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    worst = 0.0
    for seed in (1, 2):
        inputs = example_inputs(args, seed=seed)
        ref = refs[seed]
        got = sess.run(None, {n: t.numpy() for n, t in zip(names, inputs)})
        for label, r, g in zip(
            ("preds", "pre_encode_embs", "pre_encode_lengths"), ref, got
        ):
            r = np.asarray(r)
            if r.shape != g.shape:
                raise RuntimeError(f"{label}: shape {g.shape} != torch {r.shape}")
            if not np.isfinite(g.astype(np.float64)).all():
                raise RuntimeError(f"{label}: NaN/Inf in ONNX output")
            err = float(np.abs(r.astype(np.float64) - g).max())
            print(f"  seed {seed} {label:20s} shape={g.shape} max_abs_err={err:.2e}")
            if label != "pre_encode_lengths":
                worst = max(worst, err)

    # Smaller spkcache/fifo (early-stream shapes) must also run: dynamic axes.
    chunk, chunk_lengths, *_ = example_inputs(args)
    empty = np.zeros((1, 0, EMB_DIM), dtype=np.float32)
    zero = np.array([0], dtype=np.int64)
    out = sess.run(
        None,
        {
            "chunk": chunk.numpy(),
            "chunk_lengths": chunk_lengths.numpy(),
            "spkcache": empty,
            "spkcache_lengths": zero,
            "fifo": empty,
            "fifo_lengths": zero,
        },
    )
    print(f"  empty-cache run: preds {out[0].shape}")
    if out[0].shape[-1] != int(model.sortformer_modules.n_spk):
        raise RuntimeError("Unexpected speaker dimension")
    if worst > 1e-3:
        raise RuntimeError(f"ORT vs PyTorch max abs error {worst:.2e} > 1e-3")
    print("  ORT verification passed")


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """Dynamic INT8 of the encoder-layer weight MatMuls only (see module docstring)."""
    import re

    from onnxruntime.quantization import QuantType, quantize_dynamic

    proto = onnx.load(str(fp32_path))
    weight_matmul = re.compile(
        r"^/encoder/layers\.\d+/(attn/(w_qkv|out_proj)|ffn/net/net\.\d+)/MatMul$"
    )
    keep_fp32 = [
        n.name for n in proto.graph.node
        if n.op_type in ("MatMul", "Gemm", "Conv") and not weight_matmul.match(n.name)
    ]
    n_quant = sum(
        1 for n in proto.graph.node if n.op_type == "MatMul" and weight_matmul.match(n.name)
    )
    print(f"INT8: quantizing {n_quant} weight MatMuls; keeping {len(keep_fp32)} nodes fp32")
    if n_quant == 0:
        raise RuntimeError("No encoder weight MatMuls matched; node naming changed?")

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = int8_path.with_suffix(".tmp.onnx")
    quantize_dynamic(
        str(fp32_path),
        str(tmp),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=keep_fp32,
    )
    q = onnx.load(str(tmp))
    del q.metadata_props[:]
    q.metadata_props.extend(proto.metadata_props)
    q.metadata_props.append(onnx.StringStringEntryProto(key="quantization", value="dynamic_int8_encoder_weights"))
    onnx.save(q, str(int8_path))
    tmp.unlink()
    print(
        f"INT8 written: {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB; "
        f"fp32 {fp32_path.stat().st_size / 1e6:.1f} MB)"
    )


def compare_fp32_int8(fp32_path: Path, int8_path: Path, model, wav_path: str) -> None:
    """Run both graphs on real speech (empty cache, one pass) and report agreement."""
    import time
    import wave

    import onnxruntime as ort

    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16 kHz mono WAV"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)
    with torch.no_grad():
        feats, flen = model.preprocessor(
            input_signal=audio, length=torch.tensor([audio.shape[1]])
        )
    frames = (int(flen[0]) // SUBSAMPLING) * SUBSAMPLING
    chunk = feats.transpose(1, 2)[:, :frames, :].contiguous().numpy().astype(np.float32)
    empty = np.zeros((1, 0, EMB_DIM), dtype=np.float32)
    zero = np.array([0], dtype=np.int64)
    feed = {
        "chunk": chunk,
        "chunk_lengths": np.array([frames], dtype=np.int64),
        "spkcache": empty,
        "spkcache_lengths": zero,
        "fifo": empty,
        "fifo_lengths": zero,
    }
    res = {}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        so = ort.SessionOptions()
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        sess.run(None, feed)  # warm-up
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        res[name] = (out, time.perf_counter() - t0)

    (p32, e32, _), t32 = res["fp32"]
    (p8, e8, _), t8 = res["int8"]
    d = np.abs(p32 - p8)
    bin32, bin8 = p32 > 0.5, p8 > 0.5
    act32, act8 = bin32.any(-1), bin8.any(-1)
    print(f"\nfp32 vs INT8 on {wav_path} ({frames // SUBSAMPLING} frames of 80 ms):")
    print(f"  preds     max_abs_diff={d.max():.4f} mean_abs_diff={d.mean():.5f}")
    print(f"  embeddings mean_abs_diff={np.abs(e32 - e8).mean():.5f} (fp32 mean_abs={np.abs(e32).mean():.4f})")
    print(f"  per-speaker decision agreement (p>0.5): {(bin32 == bin8).mean() * 100:.3f}%")
    print(f"  speech/non-speech frame agreement:      {(act32 == act8).mean() * 100:.3f}%")
    print(f"  active speakers fp32={int(bin32.any(1).sum())} int8={int(bin8.any(1).sum())}")
    print(f"  CPU time single pass: fp32 {t32:.2f}s  int8 {t8:.2f}s  ({t32 / t8:.2f}x)")


def main() -> int:
    args = parse_args()
    print(f"torch {torch.__version__}; loading {args.input_path}")
    model = load_model(args)
    out, refs = export_model(args, model)
    if not args.skip_verify:
        verify_onnx(out, args, model, refs)
    if args.int8_output:
        int8 = Path(args.int8_output)
        quantize_int8(out, int8)
        if args.compare_wav:
            compare_fp32_int8(out, int8, model, args.compare_wav)
    return 0


if __name__ == "__main__":
    sys.exit(main())
