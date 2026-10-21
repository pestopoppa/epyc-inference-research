#!/usr/bin/env python3
"""YARN-AGG-1: re-measure token counts of the fetched oolong-synth contexts with the served model's tokenizer.

No server, no inference: reads ``tokenizer.json`` (the served model's own, same vocabulary) offline with the
``tokenizers`` package (research venv). One count per distinct ``context_window_id`` (the 25 questions of a
context share it). Writes a JSON report: per nominal length, the real-token distribution, the zone from
``retokenize.label_zone`` and a CONSTRUCTIBILITY verdict for each REAL-token target (128K/240K native,
400K/500K beyond): a target is constructible only if some contexts land in its zone within 10% of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.benchmark.yarn_gate import retokenize as rt  # noqa: E402

TOKENIZER = "/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw/tokenizer.json"
#: A target is "constructible" with >= this many contexts (half of the 16 per length) inside +-10% of it and in the
#: expected zone. The threshold was chosen after seeing the counts; the raw counts are reported beside the flag.
MIN_CONTEXTS = 8
TARGETS = {"128K": 131072, "240K": 245760, "400K": 409600, "500K": 512000}


def targets_report(per_ctx: list[dict]) -> dict:
    out = {}
    for name, t in TARGETS.items():
        near = [c for c in per_ctx if abs(c["real_n"] - t) <= rt.NOMINAL_MISMATCH_FRAC * t]
        zone_ok = [c for c in near if c["zone"] == ("native" if t < rt.NATIVE_CTX else "beyond-native")]
        out[name] = {"target_real": t, "n_contexts_within_10pct": len(near), "n_in_expected_zone": len(zone_ok),
                     "constructible": len(zone_ok) >= MIN_CONTEXTS}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    per_ctx, seen = [], set()
    for p in sorted(Path(a.dir).glob("oolong_synth_*.jsonl")):
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                key = (r["context_len"], r["context_window_id"])
                if key in seen:
                    continue
                seen.add(key)
                n = len(tok.encode(r["context_window_text"], add_special_tokens=False).ids)
                per_ctx.append({"context_len": r["context_len"], "context_window_id": r["context_window_id"], "real_n": n,
                                "bin": rt.bin_label(n), "zone": rt.label_zone(n, source="model_tokenizer_json"),
                                "dataset": r.get("dataset"), "n_chars": len(r["context_window_text"])})
    by = {}
    for c in per_ctx:
        by.setdefault(c["context_len"], []).append(c)
    summary = {str(L): rt.cell_report([{**c} for c in rs], L) for L, rs in sorted(by.items())}
    rep = {"tokenizer": a.tokenizer, "tokenizer_sha256": hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest(),
           "n_source": "model_tokenizer_json", "n_contexts": len(per_ctx), "per_nominal_length": summary,
           "targets": targets_report(per_ctx), "contexts": per_ctx}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({"per_nominal_length": summary, "targets": rep["targets"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
