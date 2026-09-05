"""Stamp CTC frontend/decode metadata onto a single-file ONNX graph.

parakeet-rs reads these `metadata_props` keys at load time (see
`CtcConfig::from_metadata` in src/config.rs). Models without the keys keep
Parakeet-CTC-0.6b defaults (blank_id=1024, no token offset, per-feature z-norm).

Keys (must stay in lockstep with src/config.rs):
  ctc_blank_id
  ctc_token_id_offset
  ctc_frontend_deltas
  ctc_frontend_delta_win_length
  ctc_frontend_stack_factor
  ctc_frontend_normalization   # per_feature_z_norm | db_floor_scale:<floor_db>
  ctc_encoder_subsample        # in-graph time downsample; Granite = 4

Usage:
  python scripts/patch_ctc_onnx_metadata.py \\
      --input onnx/model.onnx --output onnx/model.ctcmeta.onnx \\
      --preset granite-turbo-ctc

  python scripts/patch_ctc_onnx_metadata.py \\
      --input onnx/model.onnx --in-place --preset granite-turbo-ctc

Granite (community ONNX, after download):
  huggingface-cli download qwertz92/granite-speech-5.0-470m-turboctc-onnx \\
      --local-dir models/granite-turbo-ctc-onnx
  python scripts/patch_ctc_onnx_metadata.py \\
      --input models/granite-turbo-ctc-onnx/onnx/model.onnx \\
      --output models/granite-turbo-ctc-onnx/onnx/model.ctcmeta.onnx \\
      --preset granite-turbo-ctc

Requires: pip install onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx

# Keep in lockstep with src/config.rs CTC_META_* constants.
CTC_META_BLANK_ID = "ctc_blank_id"
CTC_META_TOKEN_ID_OFFSET = "ctc_token_id_offset"
CTC_META_FRONTEND_DELTAS = "ctc_frontend_deltas"
CTC_META_FRONTEND_DELTA_WIN_LENGTH = "ctc_frontend_delta_win_length"
CTC_META_FRONTEND_STACK_FACTOR = "ctc_frontend_stack_factor"
CTC_META_FRONTEND_NORMALIZATION = "ctc_frontend_normalization"
CTC_META_ENCODER_SUBSAMPLE = "ctc_encoder_subsample"

PRESETS = {
    "granite-turbo-ctc": {
        CTC_META_BLANK_ID: "0",
        # IBM unified tokenizer: <|blank|> is vocab id 0 (16384 tokens, no +1 offset).
        CTC_META_TOKEN_ID_OFFSET: "0",
        CTC_META_FRONTEND_DELTAS: "true",
        CTC_META_FRONTEND_DELTA_WIN_LENGTH: "3",
        CTC_META_FRONTEND_STACK_FACTOR: "2",
        CTC_META_FRONTEND_NORMALIZATION: "db_floor_scale:8.0",
        # Two strided conformer blocks after stack-by-2: 100 Hz -> 12.5 Hz (80 ms).
        CTC_META_ENCODER_SUBSAMPLE: "4",
    },
}


def set_metadata(model: onnx.ModelProto, props: dict[str, str]) -> None:
    existing = {p.key: i for i, p in enumerate(model.metadata_props)}
    for key, value in props.items():
        if key in existing:
            model.metadata_props[existing[key]].value = value
        else:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write CTC metadata_props onto an ONNX graph for parakeet-rs."
    )
    parser.add_argument("--input", required=True, help="Path to source ONNX file")
    parser.add_argument(
        "--output",
        help="Path to write the patched ONNX file (required unless --in-place)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input (refuses to run without this flag or --output)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Named preset (overridden by any explicit flags below)",
    )
    parser.add_argument("--blank-id", type=int, help="CTC blank token id")
    parser.add_argument("--token-offset", type=int, help="Subtract from model ids before tokenizer decode")
    parser.add_argument("--deltas", dest="deltas", action="store_true", help="Enable temporal deltas")
    parser.add_argument("--no-deltas", dest="deltas", action="store_false", help="Disable temporal deltas")
    parser.set_defaults(deltas=None)
    parser.add_argument("--delta-win-length", type=int, help="Delta regression window (torchaudio default 5, Granite 3)")
    parser.add_argument("--stack-factor", type=int, help="Frame-stacking factor (1 = none)")
    parser.add_argument(
        "--encoder-subsample",
        type=int,
        help="In-graph temporal downsample after the frontend (Granite TurboCTC: 4)",
    )
    parser.add_argument(
        "--normalization",
        choices=["per_feature_z_norm", "db_floor_scale"],
        help="Log/normalization mode",
    )
    parser.add_argument(
        "--floor-db",
        type=float,
        default=None,
        help="db_floor_scale floor in dB (default 8.0 when that mode is selected)",
    )
    return parser.parse_args()


def build_props(args: argparse.Namespace) -> dict[str, str]:
    props: dict[str, str] = {}
    if args.preset:
        props.update(PRESETS[args.preset])

    if args.blank_id is not None:
        props[CTC_META_BLANK_ID] = str(args.blank_id)
    if args.token_offset is not None:
        props[CTC_META_TOKEN_ID_OFFSET] = str(args.token_offset)
    if args.deltas is not None:
        props[CTC_META_FRONTEND_DELTAS] = "true" if args.deltas else "false"
    if args.delta_win_length is not None:
        props[CTC_META_FRONTEND_DELTA_WIN_LENGTH] = str(args.delta_win_length)
    if args.stack_factor is not None:
        props[CTC_META_FRONTEND_STACK_FACTOR] = str(args.stack_factor)
    if args.encoder_subsample is not None:
        props[CTC_META_ENCODER_SUBSAMPLE] = str(args.encoder_subsample)
    if args.normalization is not None:
        if args.normalization == "db_floor_scale":
            floor = args.floor_db if args.floor_db is not None else 8.0
            props[CTC_META_FRONTEND_NORMALIZATION] = f"db_floor_scale:{floor}"
        else:
            props[CTC_META_FRONTEND_NORMALIZATION] = "per_feature_z_norm"
    elif args.floor_db is not None:
        props[CTC_META_FRONTEND_NORMALIZATION] = f"db_floor_scale:{args.floor_db}"

    if not props:
        raise SystemExit(
            "No metadata to write. Pass --preset and/or explicit CTC flags."
        )
    return props


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    if args.in_place and args.output:
        print("Pass either --in-place or --output, not both.", file=sys.stderr)
        return 1
    if not args.in_place and not args.output:
        print("Refusing to overwrite input; pass --output or --in-place.", file=sys.stderr)
        return 1

    output_path = input_path if args.in_place else Path(args.output)
    props = build_props(args)

    print(f"Loading {input_path} (external data not loaded)...")
    model = onnx.load(str(input_path), load_external_data=False)
    set_metadata(model, props)

    if args.in_place:
        tmp = input_path.with_suffix(input_path.suffix + ".tmp")
        onnx.save(model, str(tmp))
        shutil.move(str(tmp), str(input_path))
        print(f"Updated in place: {input_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_path))
        print(f"Wrote {output_path}")

    print("Metadata:")
    for key, value in sorted(props.items()):
        print(f"  {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
