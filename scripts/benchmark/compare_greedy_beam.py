#!/usr/bin/env python3
"""Compare two transcripts (greedy vs beam) and optionally score against a reference.

Pure stdlib: computes character- and word-level Levenshtein edit distance and
normalized rates. No external dependencies required. If `jiwer` is installed and
`--reference` is given, WER is also reported via jiwer for cross-checking.

Usage:
  compare_greedy_beam.py --greedy GREEDY.txt --beam BEAM.txt [--reference REF.txt]
  compare_greedy_beam.py --greedy-text "..." --beam-text "..."

Output: a single JSON object on stdout.
"""
import argparse
import json
import sys


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join(s.strip().split())


def rates(hyp, ref):
    ref_w = norm(ref).split()
    hyp_w = norm(hyp).split()
    wer = levenshtein(ref_w, hyp_w) / max(1, len(ref_w))
    cer = levenshtein(norm(ref), norm(hyp)) / max(1, len(norm(ref)))
    return wer, cer


def read(path_or_none, text_or_none):
    if text_or_none is not None:
        return text_or_none
    if path_or_none:
        with open(path_or_none, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--greedy")
    p.add_argument("--beam")
    p.add_argument("--greedy-text")
    p.add_argument("--beam-text")
    p.add_argument("--reference")
    args = p.parse_args()

    greedy = norm(read(args.greedy, args.greedy_text))
    beam = norm(read(args.beam, args.beam_text))

    out = {
        "greedy_chars": len(greedy),
        "beam_chars": len(beam),
        "char_edit_distance": levenshtein(greedy, beam),
        "word_edit_distance": levenshtein(greedy.split(), beam.split()),
    }
    out["char_edit_rate"] = out["char_edit_distance"] / max(1, len(greedy))
    out["word_edit_rate"] = out["word_edit_distance"] / max(1, len(greedy.split()))

    if args.reference:
        ref = read(args.reference, None)
        g_wer, g_cer = rates(greedy, ref)
        b_wer, b_cer = rates(beam, ref)
        out["greedy_wer"] = g_wer
        out["greedy_cer"] = g_cer
        out["beam_wer"] = b_wer
        out["beam_cer"] = b_cer
        out["wer_delta_beam_minus_greedy"] = b_wer - g_wer
        try:
            import jiwer  # type: ignore
            out["greedy_wer_jiwer"] = jiwer.wer(norm(ref), greedy)
            out["beam_wer_jiwer"] = jiwer.wer(norm(ref), beam)
        except Exception:
            pass

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
