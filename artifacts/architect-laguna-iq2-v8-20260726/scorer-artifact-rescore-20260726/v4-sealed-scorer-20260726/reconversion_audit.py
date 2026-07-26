#!/usr/bin/env python3
"""Arm-neutral offline reconversion audit for the Laguna SWE investigation.

This tool does not score patches.  It replays only the shared converter's patch
construction over four immutable capture files and records the historical
before-counts beside the current-converter after-counts.  Any source or
converter drift is a hard failure, so a changed tool cannot silently rewrite a
cross-arm comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/mnt/raid0/llm/epyc-inference-research")
DEFAULT_CONVERTER = REPO / "artifacts/architect-code-eval-20260724/convert_sr_to_patch.py"
CONVERTER_SHA256 = "6bd2302dda3e5139cc6faabcc5639bdcf85b27895f93a9181cbb53dd65749507"
ARMS = (
    {
        "name": "Laguna_promptfix_v4",
        "pq": REPO / "artifacts/architect-laguna-iq2-v8-20260726/scorer-artifact-rescore-20260726/clean-full40-promptfix-20260726/run-20260726T220759Z/pq.jsonl",
        "pq_sha256": "a2ce92399d87c8f5f15b285ad27eca3cf0328f1b80eec1fec81933ed782cd81a",
        "before": {"empty_patches": 11, "blocks_skipped": 3},
    },
    {
        "name": "Laguna_historical_diagnostic_only",
        "pq": REPO / "artifacts/architect-laguna-iq2-v8-20260726/attempt-02-port18089/swe_oracle/pq.jsonl",
        "pq_sha256": "32e7c7885508d3b3d4f74d9e7457ce21a6d1cea9fd2c7ea7780469056cb4f0e4",
        "before": {"empty_patches": 13, "blocks_skipped": 8},
    },
    {
        "name": "A3_same_era_banked",
        "pq": REPO / "artifacts/architect-same-era-v8-20260726/live-20260726T201413Z/A3_27b_dense/swebench_oracle/pq.jsonl",
        "pq_sha256": "acedbad8b0396a0d37bf11215cb32c0ea7e71a2ffe767e9bfd2d8e6748091701",
        "before": {"empty_patches": 4, "blocks_skipped": 0},
    },
    {
        "name": "A4_same_era_banked",
        "pq": REPO / "artifacts/architect-same-era-v8-20260726/live-20260726T201413Z/A4_35b_a3b/swe/pq.jsonl",
        "pq_sha256": "0ff795d1fc638f78aabcc6e1587f451b04feecb7d59fc0b64c83ba32b97bb3b0",
        "before": {"empty_patches": 13, "blocks_skipped": 14},
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_converter(path: Path):
    if sha256(path) != CONVERTER_SHA256:
        raise RuntimeError("shared converter source drift; reconversion is refused")
    spec = importlib.util.spec_from_file_location("arm_neutral_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import shared converter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconvert(converter, arm: dict) -> dict:
    path = arm["pq"]
    actual_sha = sha256(path)
    if actual_sha != arm["pq_sha256"]:
        raise RuntimeError(f"capture source drift for {arm['name']}: {actual_sha}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    ids = [row.get("id") for row in rows]
    if len(rows) != 40 or len(set(ids)) != 40:
        raise RuntimeError(f"{arm['name']} does not have an immutable 40-ID denominator")
    empty = applied = skipped = 0
    for row in rows:
        if row.get("finish_reason") == "length":
            patch, did_apply, did_skip = "", 0, 0
        else:
            patch, did_apply, did_skip = converter.apply_blocks(
                converter.rows[row["id"]], row.get("response", ""))
        empty += not bool(patch)
        applied += did_apply
        skipped += did_skip
    return {
        "name": arm["name"], "pq": str(path), "pq_sha256": actual_sha,
        "rows": len(rows), "before": arm["before"],
        "after": {"empty_patches": empty, "blocks_skipped": skipped, "blocks_applied": applied},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--out", type=Path, default=HERE / "arm_neutral_reconversion_audit.json")
    args = parser.parse_args()
    converter = load_converter(args.converter)
    result = {
        "schema": "epyc.laguna-swe-arm-neutral-reconversion-audit.v1",
        "converter": {"path": str(args.converter), "sha256": CONVERTER_SHA256},
        "arms": [reconvert(converter, arm) for arm in ARMS],
        "rule": "Before counts are preserved historical observations; after counts are a same-converter audit, not an official SWE verdict.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "arms": len(result["arms"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
