#!/usr/bin/env python3
"""Add explicit response-integrity accounting to the sealed FA-4b M2 analysis.

This is a successor report, not a modification of the sealed producer or its
predeclared quality test. A failed or missing capture still makes M2 refuse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import v11_fa4b_quality_m2 as m2
except ImportError:
    import v11_fa4b_quality_m2 as m2


def classify_response(raw: str, content: str) -> dict[str, bool]:
    response = json.loads(raw)
    if not isinstance(response, dict) or not isinstance(content, str):
        raise ValueError("invalid response or content")
    return {"nonfinite_json": not m2.finite_json(response),
            "garbage_text": bool(m2.garbage(content))}


def analyze(manifest: dict, output: Path) -> dict:
    # M2 enforces exact coverage, seals, request/response joins, whole-server
    # log health, and refuses failure records before a count can be published.
    report = m2.analyze(manifest, output)
    questions = {q["id"]: q for q in manifest["scored"]}
    for kv in m2.KVS:
        for draft in m2.DRAFTS:
            cell = report["cells"][f"{kv}:{draft}"]
            counts = {}
            for arm in m2.ARMS:
                counts[arm] = {}
                for phase in m2.PHASES:
                    ids = ([q["id"] for q in manifest["scored"]] if phase == "score"
                           else manifest["greedy_ids"])
                    folder = output / arm / f"{kv}-{draft}" / phase
                    rows = {}
                    for path in sorted(folder.glob("[0-9][0-9][0-9]-*.json")):
                        row = m2.read_json(path)
                        qid = row.get("id")
                        if qid not in questions or qid in rows:
                            raise ValueError("unknown or duplicate observation")
                        m2.validate_record(row, phase, questions[qid], manifest, arm, kv, draft)
                        rows[qid] = row
                    for qid in ids:
                        historical = m2.historical_row(manifest, phase, arm, kv, draft, questions[qid])
                        if historical is not None:
                            if qid in rows:
                                raise ValueError("duplicate historical observation")
                            rows[qid] = historical
                    if set(rows) != set(ids):
                        raise ValueError("incomplete response set")
                    flags = [classify_response(rows[qid]["raw_response_text"], rows[qid]["content"])
                             for qid in ids]
                    counts[arm][phase] = {"responses": len(flags),
                                          "nonfinite_json": sum(f["nonfinite_json"] for f in flags),
                                          "garbage_text": sum(f["garbage_text"] for f in flags)}
            cell["response_integrity"] = counts
            if any(v["nonfinite_json"] or v["garbage_text"]
                   for arm_counts in counts.values() for v in arm_counts.values()):
                raise ValueError("invalid response passed M2 validation")
    report["response_integrity_scope"] = (
        "JSON numeric finiteness and DF2-9 client text rule for all accepted responses; "
        "M2 separately rejects failed requests and unhealthy whole-server logs. "
        "No independent logits or hidden-state finiteness claim.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--scorer-root", type=Path,
                        help="retained orchestrator checkout; all source hashes must match the M2 seal")
    args = parser.parse_args()
    if args.scorer_root:
        root = args.scorer_root.resolve()
        m2.ORCH_ROOT = root
        m2.ORCH_BENCH = root / "scripts/benchmark"
        m2.ORACLE = root / "data/benchmark/livecodebench_oracle_manifest.json"
        m2.NATIVE_FILES = {
            **{name: root / "scripts/benchmark" / name for name in
               ("seeding_scoring.py", "seeding_types.py", "debug_scorer.py", "livecodebench_oracle.py")},
            "measurement_guards.py": root / "src/autopilot_core/measurement_guards.py",
            "livecodebench_oracle_manifest.json": root / "data/benchmark/livecodebench_oracle_manifest.json",
        }
    m2.write_new(args.report, analyze(m2.read_json(args.manifest), args.output_dir))


if __name__ == "__main__":
    main()
