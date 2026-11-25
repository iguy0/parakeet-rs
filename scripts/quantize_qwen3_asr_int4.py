#!/usr/bin/env python3
"""Weight-only INT4 / mixed INT4+INT8 quantization of sherpa-onnx style Qwen3-ASR exports.

Input is the FP32 three-graph bundle written by the sherpa-onnx Qwen3-ASR exporter
(https://github.com/Wasser1462/Qwen3-ASR-onnx): ``conv_frontend.onnx``,
``encoder.onnx``, ``decoder.onnx`` plus external weight files. The same layout is
produced for Qwen3-ASR-0.6B / 1.7B and for derivatives such as oruk-fourier-2b.

Recipes
-------
int4   Everything that can be 4-bit is 4-bit: every constant-weight MatMul of the
       frontend, encoder and decoder (including the untied ``lm_head``) becomes
       ``MatMulNBits`` (4 bit) and the decoder token embedding becomes
       ``GatherBlockQuantized`` (4 bit). Only graph pieces that have no 4-bit
       operator stay in FP32: the three Conv2d layers of the frontend, norms,
       biases and the non-constant attention MatMuls.

mixed  Same, but the accuracy-sensitive areas are capped at INT8 instead of INT4:
       the whole audio path (frontend ``conv_out``, all encoder MatMuls including
       the output projector), the decoder ``lm_head``, and the decoder ``v_proj``
       and ``down_proj`` of every layer. The remaining decoder projections
       (q, k, o, gate, up) and the token embedding stay INT4.
       Rationale: published Qwen3-ASR quantization studies find the audio encoder
       carries most of the 4-bit WER loss and that sub-8-bit encoders can drift
       into repetition loops on long audio. A teacher-forced KL study of the
       oruk-fourier-2b decoder (12 committee chunks, KL against FP32 logits) ranked
       the decoder areas by benefit per byte: v_proj and down_proj together cut the
       all-INT4 KL by more than half, edge layers alone barely helped, and
       promoting q/k/o as well gave diminishing returns. Extra decoder nodes can
       be promoted with ``--int8-regex`` (full match on the node name).

Block size: 4-bit weights use 32 for ``int4`` (smallest) and 16 for ``mixed``
(about 20% lower decoder KL for roughly 0.25 GB of extra scales); override with
``--block-size``.

Algorithms: ``rtn`` (round-to-nearest, default, symmetric) or ``hqq`` (data-free
half-quadratic quantization, needs ``torch``). Use ``rtn`` on CPU: HQQ writes
asymmetric weights with zero points, which miss the fast int8 MatMulNBits kernel;
with ONNX Runtime 1.27 on an AVX2 Xeon it decoded about 15x slower (0.3 vs 5 tok/s).
HQQ cannot quantize Gather, so the embedding always uses RTN.

Run inside the ``onnx`` conda environment:

    python scripts/quantize_qwen3_asr_int4.py SRC_DIR OUT_DIR --recipe int4
    python scripts/quantize_qwen3_asr_int4.py SRC_DIR OUT_DIR --recipe mixed --algo hqq

The output directory holds ``conv_frontend.onnx``, ``encoder.onnx`` and
``decoder.onnx`` (each with a single ``.data`` file), the ``tokenizer`` directory
copied from SRC_DIR when present, and ``quantization_recipe.json``. The graph
interfaces are unchanged, so the bundle loads wherever the FP32 one did (the
loader needs ONNX Runtime >= 1.20 for ``GatherBlockQuantized``).
"""

import argparse
import collections
import json
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path

import onnx
from onnx import version_converter
from onnx.external_data_helper import load_external_data_for_model
from onnxruntime.quantization import quant_utils
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    HQQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

GRAPHS = ("conv_frontend", "encoder", "decoder")
# Layer 0 is exported without a numeric suffix (/v_proj/MatMul), later layers have one.
MIXED_DECODER_INT8 = re.compile(r"/(v|down)_proj(_\d+)?/MatMul")
TARGET_OPSET = 21  # first opset with INT4 tensors


def int8_node_names(graph_name, model, recipe, extra_regex):
    """Names of constant-weight MatMuls that stay at 8 bit instead of 4 bit."""
    inits = {i.name for i in model.graph.initializer}
    names = set()
    extra = [re.compile(r) for r in extra_regex]
    for node in model.graph.node:
        if node.op_type != "MatMul" or not any(i in inits for i in node.input):
            continue
        keep8 = False
        if recipe == "mixed":
            keep8 = (
                graph_name in ("conv_frontend", "encoder")
                or node.name == "/lm_head/MatMul"
                or bool(MIXED_DECODER_INT8.fullmatch(node.name))
            )
        if graph_name == "decoder" and any(r.fullmatch(node.name) for r in extra):
            keep8 = True
        if keep8:
            names.add(node.name)
    return names


def load_normalized(src_dir, name, work):
    """Load ``name.onnx`` as opset 21 with its external data, without ever
    serialising the (multi GB) weights: convert a structure-only copy first."""
    shell = onnx.load(str(src_dir / f"{name}.onnx"), load_external_data=False)
    # The exporter declares the default domain twice ("" and "ai.onnx"); keep one.
    versions = {o.version for o in shell.opset_import if o.domain in ("", "ai.onnx")}
    if len(versions) != 1:
        raise SystemExit(f"{name}: ambiguous default opset {versions}")
    src_version = versions.pop()
    kept = [o for o in shell.opset_import if o.domain not in ("", "ai.onnx")]
    del shell.opset_import[:]
    shell.opset_import.extend(kept)
    shell.opset_import.append(onnx.helper.make_opsetid("", src_version))
    if src_version != TARGET_OPSET:
        shell = version_converter.convert_version(shell, TARGET_OPSET)
    shell.ir_version = max(shell.ir_version, 10)  # INT4 initializers need IR >= 10
    # Tensors still point at the source's external files; read them from there.
    load_external_data_for_model(shell, str(src_dir))
    return shell, src_version


def make_config(algo, bits, block_size, accuracy_level, op_types):
    if algo == "hqq":
        return HQQWeightOnlyQuantConfig(
            block_size=block_size,
            bits=bits,
            op_types_to_quantize=op_types,
            quant_axes=(("MatMul", 0), ("Gather", 1)),
        )
    return DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=op_types,
        quant_axes=(("MatMul", 0), ("Gather", 1)),
        bits=bits,
    )


def run_pass(model, bits, block_size, accuracy_level, algo, op_types, exclude=()):
    cfg = make_config(algo, bits, block_size, accuracy_level, op_types)
    q = MatMulNBitsQuantizer(
        model,
        bits=bits,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=accuracy_level,
        nodes_to_exclude=sorted(exclude),
        algo_config=cfg,
    )
    q.process()
    return q.model.model


def quantize_graph(model, int8_names, args):
    # Pass 1: INT4 MatMuls (RTN or HQQ), skipping the INT8 set.
    model = run_pass(model, 4, args.block_size, args.accuracy_level, args.algo, ("MatMul",), int8_names)
    # Pass 2: INT4 token-embedding Gather. Only RTN supports Gather.
    model = run_pass(model, 4, args.block_size, args.accuracy_level, "rtn", ("Gather",))
    if int8_names:
        # Pass 3: constant-weight MatMuls still in FP32 (the INT8 set) become 8 bit.
        model = run_pass(model, 8, args.block_size_int8, args.accuracy_level, args.algo, ("MatMul",))
    return model


def stamp_accuracy_level(model, level):
    """HQQ configs carry no accuracy_level; set it on every MatMulNBits node so
    all algorithms run with the same compute precision."""
    if not level:
        return
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = [a for a in node.attribute if a.name == "accuracy_level"]
        if attrs:
            attrs[0].i = level
        else:
            node.attribute.append(onnx.helper.make_attribute("accuracy_level", level))


def summarize(model):
    counts = collections.Counter()
    for node in model.graph.node:
        if node.op_type == "MatMulNBits":
            bits = next(a.i for a in node.attribute if a.name == "bits")
            counts[f"MatMulNBits{bits}"] += 1
        elif node.op_type == "GatherBlockQuantized":
            counts["GatherBlockQuantized4"] += 1
        elif node.op_type in ("MatMul", "Conv"):
            counts[node.op_type + "_fp32"] += 1
    return dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="FP32 sherpa-onnx Qwen3-ASR bundle directory")
    p.add_argument("out", type=Path, help="new output directory")
    p.add_argument("--recipe", choices=("int4", "mixed"), required=True)
    p.add_argument("--algo", choices=("rtn", "hqq"), default="rtn")
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="4-bit block size, power of two >= 16 (default: 32 for int4, 16 for mixed)",
    )
    p.add_argument("--block-size-int8", type=int, default=64, help="8-bit block size")
    p.add_argument(
        "--accuracy-level",
        type=int,
        default=4,
        help="MatMulNBits accuracy_level (4 = int8 activations, fastest on CPU; 0 = unset, float compute)",
    )
    p.add_argument(
        "--int8-regex",
        action="append",
        default=[],
        help="decoder MatMul node names (full match) to keep at INT8. Layer 0 has no numeric suffix "
        "(/v_proj/MatMul, /v_proj_1/MatMul, ...), e.g. '/(v|down)_proj(_\\d+)?/MatMul'",
    )
    p.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=GRAPHS)
    args = p.parse_args()
    if args.block_size is None:
        args.block_size = 32 if args.recipe == "int4" else 16
    if args.block_size is None:
        args.block_size = 32 if args.recipe == "int4" else 16
    # The quantizer logs one line per graph node.
    logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)
    if args.out.exists():
        p.error(f"{args.out} already exists")
    for g in args.graphs:
        if not (args.src / f"{g}.onnx").exists():
            p.error(f"missing {args.src / (g + '.onnx')}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "recipe": args.recipe,
        "algo": args.algo,
        "block_size_int4": args.block_size,
        "block_size_int8": args.block_size_int8,
        "accuracy_level": args.accuracy_level,
        "int8_regex": args.int8_regex,
        "graphs": {},
    }
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        work = Path(tmp)
        final = work / "out"
        final.mkdir()
        for name in args.graphs:
            print(f"[{name}] loading and converting to opset {TARGET_OPSET}", flush=True)
            model, src_opset = load_normalized(args.src, name, work)
            int8 = int8_node_names(name, model, args.recipe, args.int8_regex)
            print(f"[{name}] INT8 MatMuls: {len(int8)}; quantizing ({args.algo})", flush=True)
            model = quantize_graph(model, int8, args)
            stamp_accuracy_level(model, args.accuracy_level)
            counts = summarize(model)
            print(f"[{name}] {counts}", flush=True)
            out_path = final / f"{name}.onnx"
            onnx.save_model(
                model,
                str(out_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{name}.onnx.data",
                size_threshold=1024,
            )
            report["graphs"][name] = {"source_opset": src_opset, "ops": counts}
            del model
        tok = args.src / "tokenizer"
        if tok.is_dir():
            shutil.copytree(tok, final / "tokenizer")
        (final / "quantization_recipe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        final.rename(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
