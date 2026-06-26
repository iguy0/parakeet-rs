#!/usr/bin/env bash
#
# parakeet-rs beam-search benchmark orchestrator.
#
# Runs each available model in greedy and beam modes, captures the final
# transcript and RTF, writes a CSV/JSON summary, and reports the greedy-vs-beam
# transcript difference (character/word edit distance). No WER by default; pass a
# reference transcript per clip to enable WER (see compare_greedy_beam.py).
#
# Generic and portable: model directories and clips are taken from environment
# variables. Missing models are skipped gracefully (logged, not failed) so the
# same script works on any developer's machine and in CI nightly jobs.
#
# Usage:
#   scripts/benchmark/run_all.sh
#   PARAKEET_MODELS_ROOT=/data/models CLIP=clean_30s.wav BEAM_SIZE=10 \
#       scripts/benchmark/run_all.sh
#
# Environment variables (all optional):
#   PARAKEET_MODELS_ROOT  Base dir for default model paths (default: repo root ".")
#   CTC_DIR               CTC model dir       (default: $PARAKEET_MODELS_ROOT)
#   TDT_DIR               TDT model dir       (default: $PARAKEET_MODELS_ROOT/tdt)
#   UNIFIED_DIR           Unified model dir   (default: $PARAKEET_MODELS_ROOT/unified)
#   CLIP                  Audio clip path     (default: 6_speakers.wav)
#   BEAM_SIZE             Beam width          (default: 5)
#   OUT_DIR               Results dir         (default: scripts/benchmark/results)
#   CARGO_FLAGS           Extra cargo flags   (default: --release)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODELS_ROOT="${PARAKEET_MODELS_ROOT:-.}"
CTC_DIR="${CTC_DIR:-$MODELS_ROOT}"
TDT_DIR="${TDT_DIR:-$MODELS_ROOT/tdt}"
UNIFIED_DIR="${UNIFIED_DIR:-$MODELS_ROOT/unified}"
CLIP="${CLIP:-6_speakers.wav}"
BEAM_SIZE="${BEAM_SIZE:-5}"
OUT_DIR="${OUT_DIR:-scripts/benchmark/results}"
CARGO_FLAGS="${CARGO_FLAGS:---release}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/$STAMP"
RUN_DIR="$OUT_DIR/$STAMP"
CSV="$RUN_DIR/summary.csv"
COMPARE_PY="$REPO_ROOT/scripts/benchmark/compare_greedy_beam.py"

echo "model,mode,decoding,beam_size,rtf,transcript_chars,status,transcript_file" > "$CSV"

log() { printf '[bench] %s\n' "$*" >&2; }

# Pull "Result:" (offline) or "Final:" (streaming) transcript line.
extract_transcript() {
  awk '/^Final: /{sub(/^Final: /,"");print;found=1} /^Result: /{sub(/^Result: /,"");r=$0} END{if(!found && r!="")print r}' "$1"
}

# Pull the RTF value from a "... RTF: 1.23x)" line.
extract_rtf() {
  grep -oE 'RTF: [0-9.]+x' "$1" | head -n1 | grep -oE '[0-9.]+' || true
}

# run_case <model_key> <example> <mode> <decoding: greedy|beam>
run_case() {
  local key="$1" example="$2" mode="$3" decoding="$4"
  local log_file="$RUN_DIR/${key}_${decoding}.log"
  local txt_file="$RUN_DIR/${key}_${decoding}.txt"

  local -a cmd=(cargo run $CARGO_FLAGS --example "$example" -- "$CLIP")
  [ -n "$mode" ] && cmd+=("$mode")
  cmd+=(-- --decoding "$decoding")
  [ "$decoding" = "beam" ] && cmd+=(--beam-size "$BEAM_SIZE")

  log "running $key/$mode/$decoding: ${cmd[*]}"
  if "${cmd[@]}" >"$log_file" 2>&1; then
    local transcript rtf chars
    transcript="$(extract_transcript "$log_file")"
    printf '%s\n' "$transcript" > "$txt_file"
    rtf="$(extract_rtf "$log_file")"
    chars="${#transcript}"
    echo "$key,$mode,$decoding,$([ "$decoding" = beam ] && echo "$BEAM_SIZE" || echo 1),${rtf:-NA},$chars,ok,$txt_file" >> "$CSV"
  else
    log "FAILED $key/$mode/$decoding (see $log_file)"
    echo "$key,$mode,$decoding,NA,NA,0,failed,$log_file" >> "$CSV"
  fi
}

# diff_case: compare greedy vs beam transcripts for a model key.
diff_case() {
  local key="$1"
  local g="$RUN_DIR/${key}_greedy.txt" b="$RUN_DIR/${key}_beam.txt"
  [ -s "$g" ] && [ -s "$b" ] || return 0
  if command -v python3 >/dev/null 2>&1; then
    log "greedy vs beam diff for $key:"
    python3 "$COMPARE_PY" --greedy "$g" --beam "$b" | tee "$RUN_DIR/${key}_diff.json" >&2
  fi
}

# has_model <dir> <file1> [file2 ...]
has_model() {
  local dir="$1"; shift
  [ -d "$dir" ] || return 1
  local f
  for f in "$@"; do
    [ -e "$dir/$f" ] || return 1
  done
  return 0
}

if [ ! -f "$CLIP" ]; then
  log "WARNING: clip '$CLIP' not found in repo root; examples may fail to open it."
fi

# --- CTC (offline) ---
if has_model "$CTC_DIR" model.onnx tokenizer.json; then
  run_case ctc raw "" greedy
  run_case ctc raw "" beam
  diff_case ctc
else
  log "skip CTC: model not found in '$CTC_DIR'"
fi

# --- TDT (offline) ---
if has_model "$TDT_DIR" encoder-model.onnx decoder_joint-model.onnx vocab.txt; then
  run_case tdt raw tdt greedy
  run_case tdt raw tdt beam
  diff_case tdt
else
  log "skip TDT: model not found in '$TDT_DIR'"
fi

# --- Unified (offline + streaming) ---
if has_model "$UNIFIED_DIR" encoder.onnx decoder_joint.onnx tokenizer.model; then
  run_case unified_offline unified "" greedy
  run_case unified_offline unified "" beam
  diff_case unified_offline
  run_case unified_streaming unified streaming greedy
  run_case unified_streaming unified streaming beam
  diff_case unified_streaming
else
  log "skip Unified: model not found in '$UNIFIED_DIR'"
fi

log "done. summary: $CSV"
cat "$CSV" >&2
