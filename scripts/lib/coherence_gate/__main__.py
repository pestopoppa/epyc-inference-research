"""CLI:  PYTHONPATH=scripts/lib python3 -m coherence_gate --base b.jsonl --cand c.jsonl [--truth t.jsonl]
          [--tokenizer tokenizer.json | --tokenize-url http://127.0.0.1:8083 | --text-only] [--out v.json]

Exit codes: 0 gate PASS, 1 gate FAIL, 2 gate INCOMPLETE, 3 refused (bad rows / fake token ids /
no token source). The full verdict record (schema epyc.coherence_gate.v1) goes to --out, or stdout.
No judge is wired here: tier-2 items come out NEEDS_REVIEW, so the gate is INCOMPLETE until a caller
injects a judge through the library API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import gate as G
from . import tokens as T

EXIT = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}


def _file_id(p: str) -> dict:
    return {"path": str(Path(p).resolve()), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="coherence_gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base/anchor arm rows (jsonl)")
    ap.add_argument("--cand", required=True, help="candidate arm rows (jsonl)")
    ap.add_argument("--truth", help="ground truth rows (jsonl; question_pool shape accepted)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--tokenizer", help="HF tokenizer.json for rows without token_ids")
    src.add_argument("--tokenize-url", help="llama-server base URL whose /tokenize fills missing token_ids")
    ap.add_argument("--text-only", action="store_true",
                    help="DECLARE text-only degeneracy (surrogate tokens, uncalibrated) when ids are absent")
    ap.add_argument("--anchor-json", help="JSON object naming the base arm (recorded on the verdict), e.g. "
                    '\'{"source_commit": "...", "binary_sha256": "...", "linkage_sha256": "..."}\'')
    ap.add_argument("--out", help="write the verdict record here (default: stdout)")
    ap.add_argument("--summary", action="store_true", help="print only the aggregate to stdout")
    a = ap.parse_args(argv)
    try:
        tok = (T.HFTokenizer(a.tokenizer) if a.tokenizer else
               T.ServerTokenizer(a.tokenize_url) if a.tokenize_url else None)
        base, cand = G.load_jsonl(a.base), G.load_jsonl(a.cand)
        truth = G.load_jsonl(a.truth) if a.truth else None
        anchor = json.loads(a.anchor_json) if a.anchor_json else None
        rep = G.evaluate(base, cand, truth, tokenizer=tok, allow_text_only=a.text_only, anchor=anchor)
    except (T.TokenProvenanceError, G.GateInputError, OSError, ValueError) as e:
        print(f"coherence_gate: REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001 - a crash must never exit 1 (= gate FAIL) or 0
        print(f"coherence_gate: ERROR (no verdict): {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    rep["inputs"] = {"base": _file_id(a.base), "cand": _file_id(a.cand),
                     "truth": _file_id(a.truth) if a.truth else None}
    blob = json.dumps(rep, indent=1, sort_keys=False, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(blob + "\n", encoding="utf-8")
    if a.summary or a.out:
        print(json.dumps(rep["aggregate"], indent=1))
    else:
        print(blob)
    return EXIT[rep["aggregate"]["gate"]]


if __name__ == "__main__":
    sys.exit(main())
