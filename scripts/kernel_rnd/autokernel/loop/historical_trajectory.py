#!/usr/bin/env python3
"""Publish the compact, source-attributed AutoKernel champion trajectory.

This is deliberately a curated normalizer, not a logfile indexer.  Historical
measurements span several pre-contract eras, so each admitted checkpoint names
an exact source and literals which that source must still contain.  A missing or
changed source fails publication instead of silently changing the dashboard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

SCHEMA = "epyc.autokernel.historical_trajectory.v1"
FILENAME = "historical-trajectory.json"
PRODUCTION = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
PRISTINE = "c51e4dabf9268d43f243b230cf59ca0482f2a282"
CURRENT = "ef81196d5bdd4190b46dff4ae7eecc333a46c8ce"
DEEPSEEK = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
QWEN_GPU = "Qwen3.8-27B-Q8_0"
QWEN_CPU = "Qwen3.8-Flash-Next"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path, required: Iterable[str]) -> dict[str, str]:
    body = path.read_bytes()
    raw = body.decode("utf-8") if required else ""
    missing = [item for item in required if item not in raw]
    if missing:
        raise ValueError(f"{path}: required evidence changed or missing: {missing}")
    return {"path": str(path), "sha256": _digest(path)}


def _legacy_receipt(path: Path, surface: str, rounded_gain: float) -> dict[str, str]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("surface") != surface or round(float(body.get("effect_pct")), 3) != rounded_gain:
        raise ValueError(f"{path}: historical receipt no longer matches its ledger checkpoint")
    return {"path": str(path), "sha256": _digest(path)}


def _point(*, commit: str, at: str, model: str, surface: str, recipe: str,
           era: str, gain: float | None, baseline: str, baseline_label: str,
           state: str, source: dict[str, str], pairs: int | None = None,
           note: str | None = None, bounded_gain: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "commit": commit, "recorded_at": at, "model": model,
        "surface": surface, "recipe": recipe, "era": era,
        "gain_pct": gain, "baseline": {"commit": baseline,
        "label": baseline_label}, "evidence_state": state, "evidence": source,
    }
    if pairs is not None:
        out["pairs"] = pairs
    if note:
        out["note"] = note
    if bounded_gain is not None:
        out["bounded_gain_pct"] = bounded_gain
    return out


def build(*, store: Path, research: Path, root_repo: Path) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    model_contract = _source(
        research / "docs/design/autokernel-production-shaped-rung.md",
        ["measures on **DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M**"])

    legacy = [
        ("cb6173725876db0a82a3fb39359362723b83978a", "2026-08-31T14:32:04.086425Z", "tg128", 15.326),
        ("aba5a8155cdd7d2d2a0ea5ef947b860bb3ac322f", "2026-08-31T15:54:37.996231Z", "tg128", 16.180),
        ("32fad0188a16c1fe7c324fc50bc3922a397ee881", "2026-08-31T19:17:09.613804Z", "tg128", 19.333),
        ("48a004cd2b3a3abacc66416e75241a8472ff0468", "2026-09-01T02:06:15.652968Z", "tg128", 13.096),
        ("c0d42d81c15597208e35c5f15ca5dbd913144c10", "2026-09-01T05:37:42.922776Z", "tg128", 15.065),
        ("1f7f36e4566e7d043ef68972f514734a5686b145", "2026-09-01T06:21:55.142306Z", "tg128", 16.494),
        ("9e18beb0036860f87cde32a77350f12fda8c1793", "2026-09-01T07:19:23.754557Z", "tg128", 17.937),
        ("7d2ea88b551aafbd963d1c23180d05f0323fef92", "2026-09-01T13:04:43.146499Z", "dec-b4", 6.666),
        ("db18f39377674657c18c84094e0d7a48b6d3a11c", "2026-09-01T18:49:15.019902Z", "dec-b4", 12.186),
    ]
    for commit, at, surface, gain in legacy:
        receipt = store / f"champion-vs-production.{commit[:12]}.json"
        evidence = _legacy_receipt(receipt, surface, gain)
        evidence["model_contract_path"] = model_contract["path"]
        evidence["model_contract_sha256"] = model_contract["sha256"]
        points.append(_point(commit=commit, at=at, model=DEEPSEEK,
            surface=surface, recipe="gpu-loop-production-shaped-v1", era="gpu-deepseek-loop",
            gain=gain, baseline=PRODUCTION, baseline_label="production-consolidated-v9",
            state="serving_verified_direct", source=evidence, pairs=20))

    a272_path = research / "artifacts/autokernel-champ-a2728701-ab/champion-a2728701-vs-v9.json"
    a272 = _source(a272_path, ["a2728701530d2b76a71939509afbeb2386e53751",
                               '"effect_pct": 12.618108071254252', DEEPSEEK])
    points.append(_point(commit="a2728701530d2b76a71939509afbeb2386e53751",
        at="2026-08-31T10:25:23Z", model=DEEPSEEK, surface="tg128",
        recipe="gpu-loop-production-shaped-v1", era="gpu-deepseek-loop", gain=12.618108071254252,
        baseline=PRODUCTION, baseline_label="production-consolidated-v9",
        state="serving_verified_direct", source=a272, pairs=20))

    gpu_specs = [
        ("b0eb4fab4729f0f4b813390c8f05235b42642d60", "2026-09-03T14:54:34Z", "dec-b4", 22.44286910166302, 5),
        ("732389d6d9d08338fe2ad2457bf8f44205914a7f", "2026-09-01T19:48:03Z", "dec-b4", -1.4144466097860575, 20),
        ("bff30cebee0de4440dd7a6ef3607e59b970268ce", "2026-09-07T19:58:50Z", "tg128", 5.633302338439394, 20),
    ]
    for commit, at, surface, gain, pairs in gpu_specs:
        src = _source(store / f"champion-vs-production.{commit[:12]}.json",
                      [f'"effect_pct": {gain}', QWEN_GPU, f'"surface": "{surface}"'])
        note = "historical ancestor; current champion is ef81196d" if commit.startswith("bff30") else None
        points.append(_point(commit=commit, at=at, model=QWEN_GPU, surface=surface,
            recipe="gpu-production-model-direct-ab", era="gpu-qwen38-loop", gain=gain,
            baseline=PRODUCTION, baseline_label="production-consolidated-v9",
            state="serving_verified_direct", source=src, pairs=pairs, note=note))

    # The status ledger's +27.363% record was later overwritten by a same-commit
    # direct -1.414% receipt. Preserve it, but never draw it as an equivalent fact.
    ledger = _source(store / "experiments.db", [])
    points.append(_point(commit="732389d6d9d08338fe2ad2457bf8f44205914a7f",
        at="2026-09-01T20:03:12.914982Z", model=QWEN_GPU, surface="dec-b4",
        recipe="gpu-production-model-direct-ab", era="gpu-qwen38-loop", gain=27.363,
        baseline=PRODUCTION, baseline_label="production-consolidated-v9",
        state="overwritten_conflicting_producer_record", source=ledger, pairs=20,
        note="status-ledger result conflicts with the retained same-commit direct receipt (-1.414%)"))

    gemma_commit = "732389d6d9d08338fe2ad2457bf8f44205914a7f"
    gemma_src = _source(store / f"champion-vs-production.{gemma_commit[:12]}.gemma-4-26B-A4B-it-Q4_K_M.json",
                        ['"effect_pct": 7.205928784886151', 'gemma-4-26B-A4B-it-Q4_K_M', '"pairs": 20'])
    points.append(_point(commit=gemma_commit, at="2026-09-04T00:00:00Z",
        model="gemma-4-26B-A4B-it-Q4_K_M", surface="dec-b4",
        recipe="gpu-production-model-direct-ab", era="gpu-cross-model-transfer", gain=7.205928784886151,
        baseline=PRODUCTION, baseline_label="production-consolidated-v9",
        state="serving_verified_direct", source=gemma_src, pairs=20))

    cpu_6f = _source(root_repo / "docs/design/inf70-cpu-fold-into-champion-20260907.md",
        ["6f032c48", "1.4834", "1.6934", "c51e4dabf"])
    cpu_9c = _source(root_repo / "handoffs/active/cpu-decode-roofline-program.md",
        ["9c4f73e2", "1.5149", "1.7151", "c51e4dabf", "SUPERSEDED"])
    cpu_new_path = Path("/mnt/raid0/llm/epyc-inference-research/data/inf70-retest1-2026-09-08/CHAMPION-FINAL.md")
    cpu_new = _source(cpu_new_path, ["ef81196d", "1.8255", "2.1857", "c51e4dabf", "80.8", "117.3"])
    cpu_rows = [
        ("6f032c48db14e43f53a636df3ac02f97844181bd", "2026-09-06T22:18:06Z", "mtp-decode", "old-harness-defaults", 48.34, "whole_candidate_direct", cpu_6f, None),
        ("6f032c48db14e43f53a636df3ac02f97844181bd", "2026-09-06T22:18:06Z", "plain-decode", "old-harness-defaults", 69.34, "whole_candidate_direct", cpu_6f, None),
        ("9c4f73e2965ff347593ef335d52aea585af92106", "2026-09-07T13:01:06Z", "mtp-decode", "old-harness-shim-off", 51.49, "superseded_whole_candidate_direct", cpu_9c, None),
        ("9c4f73e2965ff347593ef335d52aea585af92106", "2026-09-07T13:01:06Z", "plain-decode", "old-harness-shim-off", 71.51, "superseded_whole_candidate_direct", cpu_9c, None),
        (CURRENT, "2026-09-08T08:53:12Z", "mtp-decode", "canonical-hot-process-thp-disable", 82.55, "whole_candidate_direct_bounded", cpu_new, 80.8),
        (CURRENT, "2026-09-08T08:53:12Z", "plain-decode", "canonical-hot-process-thp-disable", 118.57, "whole_candidate_direct_bounded", cpu_new, 117.3),
    ]
    for commit, at, surface, recipe, gain, state, src, bound in cpu_rows:
        points.append(_point(commit=commit, at=at, model=QWEN_CPU, surface=surface,
            recipe=recipe, era=("inf70-canonical-hot" if commit == CURRENT else "inf70-old-harness"),
            gain=gain, baseline=PRISTINE, baseline_label="pristine-upstream-c51e4dabf",
            state=state, source=src, bounded_gain=bound))

    points.sort(key=lambda p: (p["recorded_at"], p["model"], p["surface"], p["recipe"], p["evidence_state"]))
    return {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_epochs": [{"commit": PRODUCTION, "label": "production-consolidated-v9",
            "effective_at": "2026-08-11T00:00:00Z"}],
        "points": points,
        "missing_checkpoints": [{"commit": CURRENT, "baseline": {"commit": PRODUCTION,
            "label": "production-consolidated-v9"}, "evidence_state": "missing_production_ab",
            "note": "current champion ef81196d has no direct production A/B; bff30ce is historical only"}],
        "active_campaign": {"id": "aku12a-glm53-five-loop", "model": "GLM-5.3-Flash",
            "store": "/mnt/raid0/llm/tmp/aku12a-glm53-five-loop-store",
            "champion_of_record": "c463f601bd39d0e313b744c214b8c22f9455bcd3",
            "expected_keep_count": 23, "expected_gain_pct_vs_cor": 0.8303809823155373},
    }


def write(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, output)
    finally:
        try: os.unlink(name)
        except FileNotFoundError: pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("/mnt/raid0/llm/autokernel/loop-memory"))
    parser.add_argument("--research", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--root-repo", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    write(build(store=args.store, research=args.research, root_repo=args.root_repo), args.store / FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
