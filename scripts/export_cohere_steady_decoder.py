"""Derive a Cohere Transcribe steady-state decoder from a merged ONNX graph.

Optimum's `decoder_model_merged*.onnx` runs the first step and every later
step in one graph. `/model/is_first_step` is
`Equal(past_key_values.0.encoder.key.shape[2], 0)` and feeds two `If` nodes:

  if_cross_attn_kv_cache
    then: MatMulNBits projecting `encoder_hidden_states` into cross K/V
    else: Identity of `past_key_values.{i}.encoder.{key,value}`
  if_cross_attn_public_cache
    then: Identity of those projections, published as `present.{i}.encoder.*`
    else: Slice of the past encoder cache down to an empty axis-2 range

After step 0 the else branches only copy the encoder K/V (about 150 ms/step
at encoder length 2400 on 4 CPU threads in the Phase 6 profile) and the
projections do not run. This script keeps the else behavior without the
`If`: cross-attention reads `past_key_values.{i}.encoder.{key,value}`
directly, `encoder_hidden_states` and `present.{i}.encoder.*` are removed,
and the then-branch MatMulNBits weights are dropped from the graph. Logits
and `present.{i}.decoder.*` stay. The first decode step must still use the
merged graph; parakeet-rs does that when
`decoder_model_steady.onnx` or `decoder_model_steady_quantized.onnx` sits
next to the merged decoder (flat or under `onnx/`).

Remaining initializers keep their original external-data offsets. Each
referenced `.onnx_data` file is placed beside `--output` as a hardlink when
the filesystem allows it (no extra bytes). ONNX Runtime rejects a symlink
whose target resolves outside the model directory, so a cross-filesystem
output falls back to a copy. The decoder external data is about 200 MB, not
the multi-GB encoder. `onnx.checker.check_model` runs with the working
directory set to the source file's directory so it reads the real
`.onnx_data` file.

Metadata: `cohere_decoder_variant=steady`.

Usage:
  python scripts/export_cohere_steady_decoder.py \\
      --decoder onnx/decoder_model_merged_quantized.onnx \\
      --output /tmp/cohere_steady/onnx/decoder_model_steady_quantized.onnx

  # Also symlink encoder, tokenizer, and the merged decoder into a model dir
  # that `CohereASR::from_pretrained` can load (PARAKEET_COHERE_DIR):
  python scripts/export_cohere_steady_decoder.py \\
      --decoder onnx/decoder_model_merged_quantized.onnx \\
      --output /tmp/cohere_steady/onnx/decoder_model_steady_quantized.onnx \\
      --assemble-dir /tmp/cohere_steady

  python scripts/export_cohere_steady_decoder.py \\
      --decoder onnx/decoder_model_merged_quantized.onnx \\
      --output /tmp/cohere_steady/onnx/decoder_model_steady_quantized.onnx \\
      --benchmark

Requires: pip install onnx onnxruntime numpy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

VARIANT_KEY = "cohere_decoder_variant"
VARIANT_VALUE = "steady"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a steady-state Cohere decoder graph (no first-step If) "
            "from a merged decoder ONNX file."
        )
    )
    parser.add_argument(
        "--decoder",
        required=True,
        type=Path,
        help="Path to decoder_model_merged[_quantized|_fp16].onnx",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Destination .onnx path. Use decoder_model_steady.onnx or "
            "decoder_model_steady_quantized.onnx so parakeet-rs can find it."
        ),
    )
    parser.add_argument(
        "--assemble-dir",
        type=Path,
        default=None,
        help=(
            "Optional model directory. Symlink tokenizer.json, the source "
            "onnx/ siblings (encoder, merged decoder, external data), and "
            "leave --output as the steady graph."
        ),
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the single-step ORT comparison against the merged else branch.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time the steady graph at intra_op threads 1, 4, 8, and 16.",
    )
    return parser.parse_args()


def external_location(tensor: onnx.TensorProto) -> str | None:
    if tensor.data_location != TensorProto.EXTERNAL:
        return None
    for entry in tensor.external_data:
        if entry.key == "location":
            return entry.value
    return None


def else_identity_sources(if_node: onnx.NodeProto) -> dict[str, str]:
    """Map each If output to the past_key_values input its else Identity reads."""
    else_graph = None
    for attr in if_node.attribute:
        if attr.name == "else_branch":
            else_graph = attr.g
            break
    if else_graph is None:
        raise SystemExit(f"{if_node.name}: missing else_branch")
    if len(else_graph.output) != len(if_node.output):
        raise SystemExit(
            f"{if_node.name}: else outputs {len(else_graph.output)} "
            f"!= If outputs {len(if_node.output)}"
        )
    produced: dict[str, str] = {}
    for node in else_graph.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            raise SystemExit(
                f"{if_node.name}: else branch has {node.op_type}, expected Identity"
            )
        source = node.input[0]
        if ".encoder." not in source or not source.startswith("past_key_values."):
            raise SystemExit(f"{if_node.name}: else Identity reads {source}")
        produced[node.output[0]] = source
    mapping: dict[str, str] = {}
    for index, output in enumerate(if_node.output):
        subgraph_out = else_graph.output[index].name
        if subgraph_out not in produced:
            raise SystemExit(f"{if_node.name}: no Identity for else output {subgraph_out}")
        mapping[output] = produced[subgraph_out]
    return mapping


def find_if_nodes(graph: onnx.GraphProto) -> tuple[onnx.NodeProto, onnx.NodeProto]:
    kv_cache = None
    public_cache = None
    for node in graph.node:
        if node.op_type != "If":
            continue
        outputs = list(node.output)
        if outputs and all(name.startswith("cross_attn.") and ".encoder." in name for name in outputs):
            kv_cache = node
        elif outputs and all(
            name.startswith("present.") and ".encoder." in name for name in outputs
        ):
            public_cache = node
    if kv_cache is None or public_cache is None:
        names = [n.name for n in graph.node if n.op_type == "If"]
        raise SystemExit(f"expected cross-attn K/V and public-cache If nodes, found {names}")
    return kv_cache, public_cache


def is_present_encoder(name: str) -> bool:
    return name.startswith("present.") and ".encoder." in name


def rewrite_and_prune(model: onnx.ModelProto) -> dict[str, int]:
    graph = model.graph
    before_nodes = len(graph.node)
    before_inits = len(graph.initializer)
    kv_cache, public_cache = find_if_nodes(graph)
    mapping = else_identity_sources(kv_cache)

    for node in graph.node:
        if node is kv_cache or node is public_cache:
            continue
        for index, name in enumerate(node.input):
            replacement = mapping.get(name)
            if replacement is not None:
                node.input[index] = replacement

    drop_ids = {id(kv_cache), id(public_cache)}
    for index in range(len(graph.node) - 1, -1, -1):
        if id(graph.node[index]) in drop_ids:
            del graph.node[index]

    for index in range(len(graph.output) - 1, -1, -1):
        if is_present_encoder(graph.output[index].name):
            del graph.output[index]

    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            if output:
                producers[output] = node

    needed: set[int] = set()
    stack = [output.name for output in graph.output if output.name]
    while stack:
        name = stack.pop()
        node = producers.get(name)
        if node is None or id(node) in needed:
            continue
        needed.add(id(node))
        for inp in node.input:
            if inp:
                stack.append(inp)

    for index in range(len(graph.node) - 1, -1, -1):
        if id(graph.node[index]) not in needed:
            del graph.node[index]

    used: set[str] = set()
    produced: set[str] = set()
    for node in graph.node:
        for output in node.output:
            if output:
                produced.add(output)
        for inp in node.input:
            if inp:
                used.add(inp)
    for output in graph.output:
        used.add(output.name)

    if "encoder_hidden_states" in used:
        raise SystemExit("encoder_hidden_states is still consumed after pruning")
    if any(is_present_encoder(output.name) for output in graph.output):
        raise SystemExit("present.*.encoder.* outputs remain")
    if any(node.op_type == "If" for node in graph.node):
        raise SystemExit("If nodes remain after pruning")

    for index in range(len(graph.input) - 1, -1, -1):
        if graph.input[index].name not in used:
            del graph.input[index]
    input_names = {item.name for item in graph.input}

    for index in range(len(graph.initializer) - 1, -1, -1):
        if graph.initializer[index].name not in used:
            del graph.initializer[index]
    init_names = {item.name for item in graph.initializer}

    for index in range(len(graph.sparse_initializer) - 1, -1, -1):
        if graph.sparse_initializer[index].values.name not in used:
            del graph.sparse_initializer[index]

    for index in range(len(graph.value_info) - 1, -1, -1):
        if graph.value_info[index].name not in used and graph.value_info[index].name not in produced:
            del graph.value_info[index]

    missing = sorted(name for name in used if name not in produced and name not in input_names and name not in init_names)
    if missing:
        raise SystemExit(f"pruned graph has unresolved inputs: {missing[:8]}")

    set_metadata(model, VARIANT_KEY, VARIANT_VALUE)
    return {
        "nodes_before": before_nodes,
        "nodes_after": len(graph.node),
        "initializers_before": before_inits,
        "initializers_after": len(graph.initializer),
        "inputs": len(graph.input),
        "outputs": len(graph.output),
    }


def set_metadata(model: onnx.ModelProto, key: str, value: str) -> None:
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def symlink_file(src: Path, dest: Path) -> None:
    src = src.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.resolve() == src:
            return
        raise SystemExit(f"{dest} exists and does not point at {src}")
    dest.symlink_to(src)


def place_external_data_file(src: Path, dest: Path) -> str:
    """Put `src` at `dest` as a regular file.

    A symlink that resolves outside the model directory is rejected by ONNX
    Runtime ("External data path escapes model directory"). A hardlink is a
    regular file and shares the inode. Copy only when the hardlink fails.
    """
    src = src.resolve()
    if dest.is_symlink():
        dest.unlink()
    if dest.exists():
        src_stat = src.stat()
        dest_stat = dest.stat()
        if dest_stat.st_dev == src_stat.st_dev and dest_stat.st_ino == src_stat.st_ino:
            return "hardlink"
        if dest.is_file() and dest_stat.st_size == src_stat.st_size:
            return "existing-file"
        raise SystemExit(f"{dest} exists and is not the source data file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError as exc:
        print(
            f"hardlink failed ({exc}); copying {src.name} "
            f"({src.stat().st_size} bytes) to {dest}",
            flush=True,
        )
        shutil.copy2(src, dest)
        return "copy"


def link_external_data(model: onnx.ModelProto, src_dir: Path, out_path: Path) -> list[str]:
    placed: list[str] = []
    seen: set[str] = set()
    for tensor in model.graph.initializer:
        location = external_location(tensor)
        if location is None or location in seen:
            continue
        seen.add(location)
        src = src_dir / location
        if not src.exists():
            raise SystemExit(f"external data missing: {src}")
        dest = out_path.parent / location
        if dest.exists() and dest.resolve() == src.resolve() and not dest.is_symlink():
            placed.append(f"{location}:same-file")
            continue
        how = place_external_data_file(src, dest)
        placed.append(f"{location}:{how}")
    return placed


def assemble_model_dir(assemble_dir: Path, decoder_path: Path, steady_path: Path) -> None:
    """Symlink a from_pretrained layout around the steady graph."""
    src_onnx = decoder_path.parent
    src_root = src_onnx.parent
    assemble_dir.mkdir(parents=True, exist_ok=True)
    linked_tokenizer = False
    for candidate in (src_root / "tokenizer.json", src_onnx / "tokenizer.json"):
        if candidate.is_file():
            symlink_file(candidate, assemble_dir / "tokenizer.json")
            linked_tokenizer = True
            break
    if not linked_tokenizer:
        raise SystemExit(f"tokenizer.json not found beside {decoder_path}")

    dest_onnx = assemble_dir / "onnx"
    dest_onnx.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_onnx.iterdir()):
        if not src.is_file() and not src.is_symlink():
            continue
        if src.resolve() == steady_path.resolve():
            continue
        dest = dest_onnx / src.name
        # A regular file already placed here (hardlink or copy of external
        # data) must stay a regular file. Replacing it with a symlink makes
        # ONNX Runtime reject the steady graph.
        if dest.exists() and not dest.is_symlink():
            continue
        symlink_file(src, dest)


def feed_for(
    session_inputs: list,
    *,
    batch: int,
    seq: int,
    past_decoder: int,
    past_encoder: int,
    heads: int,
    head_dim: int,
    hidden: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    total = past_decoder + seq
    feeds: dict[str, np.ndarray] = {}
    for item in session_inputs:
        name = item.name
        if name == "input_ids":
            feeds[name] = rng.integers(0, 1000, size=(batch, seq), dtype=np.int64)
        elif name == "attention_mask":
            feeds[name] = np.ones((batch, total), dtype=np.int64)
        elif name == "position_ids":
            feeds[name] = np.arange(past_decoder, total, dtype=np.int64)[None, :]
        elif name == "num_logits_to_keep":
            feeds[name] = np.array(1, dtype=np.int64)
        elif name == "encoder_hidden_states":
            feeds[name] = rng.standard_normal((batch, past_encoder, hidden), dtype=np.float32)
        elif name.startswith("past_key_values.") and ".decoder." in name:
            feeds[name] = rng.standard_normal(
                (batch, heads, past_decoder, head_dim), dtype=np.float32
            )
        elif name.startswith("past_key_values.") and ".encoder." in name:
            feeds[name] = rng.standard_normal(
                (batch, heads, past_encoder, head_dim), dtype=np.float32
            )
        else:
            raise SystemExit(f"do not know how to fill input {name}")
    return feeds


def session_options(threads: int, profile: bool = False):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = profile
    return options


def compare_to_merged(merged_path: Path, steady_path: Path) -> None:
    import onnxruntime as ort

    merged = ort.InferenceSession(
        str(merged_path), session_options(4), providers=["CPUExecutionProvider"]
    )
    steady = ort.InferenceSession(
        str(steady_path), session_options(4), providers=["CPUExecutionProvider"]
    )
    steady_names = {item.name for item in steady.get_inputs()}
    if "encoder_hidden_states" in steady_names:
        raise SystemExit("steady graph still has encoder_hidden_states")
    rng = np.random.default_rng(0)
    merged_feed = feed_for(
        merged.get_inputs(),
        batch=1,
        seq=1,
        past_decoder=8,
        past_encoder=24,
        heads=8,
        head_dim=128,
        hidden=1024,
        rng=rng,
    )
    # Share the cache tensors so both graphs see the same else-branch inputs.
    steady_feed = {name: merged_feed[name] for name in steady_names}
    merged_outs = [item.name for item in merged.get_outputs()]
    steady_outs = [item.name for item in steady.get_outputs()]
    merged_values = dict(zip(merged_outs, merged.run(merged_outs, merged_feed), strict=True))
    steady_values = dict(zip(steady_outs, steady.run(steady_outs, steady_feed), strict=True))
    worst = 0.0
    for name, steady_value in steady_values.items():
        merged_value = merged_values[name]
        if steady_value.shape != merged_value.shape:
            raise SystemExit(
                f"{name}: shape {steady_value.shape} != merged {merged_value.shape}"
            )
        delta = float(np.max(np.abs(steady_value.astype(np.float64) - merged_value.astype(np.float64))))
        worst = max(worst, delta)
        if delta > 0.0:
            raise SystemExit(f"{name}: max abs diff {delta}")
    print(f"check: steady vs merged else-branch max abs diff {worst}", flush=True)


def benchmark(steady_path: Path, merged_path: Path) -> None:
    import onnxruntime as ort

    def median_ms(path: Path, threads: int, past_encoder: int, repeats: int = 5) -> float:
        session = ort.InferenceSession(
            str(path), session_options(threads), providers=["CPUExecutionProvider"]
        )
        rng = np.random.default_rng(1)
        feed = feed_for(
            session.get_inputs(),
            batch=1,
            seq=1,
            past_decoder=32,
            past_encoder=past_encoder,
            heads=8,
            head_dim=128,
            hidden=1024,
            rng=rng,
        )
        outputs = [item.name for item in session.get_outputs()]
        session.run(outputs, feed)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            session.run(outputs, feed)
            samples.append((time.perf_counter() - start) * 1000.0)
        samples.sort()
        return samples[len(samples) // 2]

    print("benchmark: median ms, seq=1, past_decoder=32", flush=True)
    for past_encoder in (256, 2400):
        for threads in (1, 4, 8, 16):
            steady_ms = median_ms(steady_path, threads, past_encoder)
            print(
                f"  steady enc={past_encoder} threads={threads} median_ms={steady_ms:.1f}",
                flush=True,
            )
        merged_ms = median_ms(merged_path, 4, past_encoder)
        print(
            f"  merged enc={past_encoder} threads=4 median_ms={merged_ms:.1f}",
            flush=True,
        )

    session = ort.InferenceSession(
        str(steady_path), session_options(4, profile=True), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(2)
    feed = feed_for(
        session.get_inputs(),
        batch=1,
        seq=1,
        past_decoder=32,
        past_encoder=2400,
        heads=8,
        head_dim=128,
        hidden=1024,
        rng=rng,
    )
    outputs = [item.name for item in session.get_outputs()]
    session.run(outputs, feed)
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    by_op: dict[str, float] = {}
    lm_head_ms = 0.0
    for event in events:
        if event.get("cat") != "Node":
            continue
        dur_ms = float(event.get("dur", 0)) / 1000.0
        op_name = str(event.get("args", {}).get("op_name", ""))
        by_op[op_name] = by_op.get(op_name, 0.0) + dur_ms
        node_name = str(event.get("name", ""))
        if "lm_head" in node_name:
            lm_head_ms += dur_ms
    ranked = sorted(by_op.items(), key=lambda item: item[1], reverse=True)[:8]
    print("profile enc=2400 threads=4 (sum of node dur, ms):", flush=True)
    for op_name, total in ranked:
        print(f"  {op_name} {total:.1f}", flush=True)
    print(f"  lm_head nodes {lm_head_ms:.1f}", flush=True)
    profile_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    decoder_path = args.decoder
    if not decoder_path.is_file():
        raise SystemExit(f"decoder not found: {decoder_path}")
    model = onnx.load(str(decoder_path), load_external_data=False)
    stats = rewrite_and_prune(model)
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Checker opens external data relative to the process cwd and rejects
    # symlinks. Run it from the source directory, where the .onnx_data file
    # is a regular file, then symlink that file beside the output.
    cwd = Path.cwd()
    try:
        os.chdir(decoder_path.parent)
        onnx.checker.check_model(model)
    finally:
        os.chdir(cwd)
    onnx.save(model, str(out_path))
    locations = link_external_data(model, decoder_path.parent, out_path)
    saved = onnx.load(str(out_path), load_external_data=False)
    variant = {prop.key: prop.value for prop in saved.metadata_props}.get(VARIANT_KEY)
    if variant != VARIANT_VALUE:
        raise SystemExit(f"metadata {VARIANT_KEY}={variant!r}")
    external = sum(1 for tensor in saved.graph.initializer if external_location(tensor))
    print(
        "wrote {path} ({size} bytes) nodes {before}->{after} "
        "initializers {ib}->{ia} inputs {inputs} outputs {outputs} "
        "external_tensors {external} data {locations}".format(
            path=out_path,
            size=out_path.stat().st_size,
            before=stats["nodes_before"],
            after=stats["nodes_after"],
            ib=stats["initializers_before"],
            ia=stats["initializers_after"],
            inputs=stats["inputs"],
            outputs=stats["outputs"],
            external=external,
            locations=locations,
        ),
        flush=True,
    )
    if args.assemble_dir is not None:
        assemble_model_dir(args.assemble_dir, decoder_path, out_path)
        print(f"assembled model dir {args.assemble_dir}", flush=True)
    if not args.skip_check:
        compare_to_merged(decoder_path, out_path)
    if args.benchmark:
        benchmark(out_path, decoder_path)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
