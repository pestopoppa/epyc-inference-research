#!/usr/bin/env python3
"""Materialize the A0 objective-verifier reviewer floor.

This is a no-inference baseline for reviewer-model ablations. It emits a
ledger-shaped ``decisions.jsonl`` where the reviewer returns the gold label:
``accept`` rows become ``approve`` and ``reject`` rows become ``reject``.
The output is useful as a ceiling/floor reference for calibration plumbing; it
is not a deployable reviewer model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CORPUS = Path("/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl")
DEFAULT_REVIEWER_ID = "a0_objective_verifier_floor"
DEFAULT_RUBRIC_VERSION = "objective_verifier_floor_v1"
DEFAULT_ERA = "p_rev1_attested"
DEFAULT_PROTOCOL = "p_rev1"
DEFAULT_ATTESTATION = "MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719"


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_row_ids(path: Path) -> list[str]:
    row_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        row_id = stripped.split("#", 1)[0].strip()
        if row_id:
            row_ids.append(row_id)
    return row_ids


def corpus_by_row_id(corpus_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(corpus_path):
        row_id = row.get("row_id") or row.get("candidate_id")
        if isinstance(row_id, str) and row_id:
            rows[row_id] = row
    return rows


def decision_for_gold(gold_label: str) -> str:
    normalized = gold_label.strip().lower()
    if normalized in {"accept", "pass"}:
        return "approve"
    if normalized in {"reject", "fail"}:
        return "reject"
    raise ValueError(f"unsupported gold_label for A0 floor: {gold_label!r}")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "a0"


def decision_id_for(reviewer_id: str, row_id: str) -> str:
    digest = hashlib.sha1(f"{reviewer_id}\0{row_id}\0a0".encode("utf-8")).hexdigest()[:24]
    return f"{safe_name(reviewer_id)}-rev-{digest}"


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    artifacts_dir = output_dir / "artifacts"
    decisions_path = output_dir / "decisions.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.measurement_protocol == "p_rev1" and not args.protocol_attestation:
        raise ValueError("--protocol-attestation is required for p_rev1")

    corpus = corpus_by_row_id(args.corpus.expanduser().resolve())
    row_ids = read_row_ids(args.row_ids_file.expanduser().resolve())
    missing = [row_id for row_id in row_ids if row_id not in corpus]
    if missing:
        raise ValueError("row ids missing from corpus: " + ", ".join(missing[:5]))

    generated_at = datetime.now(timezone.utc).isoformat()
    decisions: list[dict[str, Any]] = []
    for index, row_id in enumerate(row_ids, start=1):
        row = corpus[row_id]
        gold_label = str(row.get("gold_label") or "")
        decision = decision_for_gold(gold_label)
        event_path = artifacts_dir / f"{safe_name(row_id)}.objective_floor.json"
        event = {
            "schema": "reviewer_a0_objective_floor.event.v1",
            "row_id": row_id,
            "row_index": index,
            "gold_label": gold_label,
            "decision": decision,
            "source": "gold_label_objective_floor",
        }
        write_json(event_path, event)
        decisions.append(
            {
                "candidate_id": row_id,
                "confidence": 1.0,
                "corpus_id": row.get("corpus_id"),
                "decision": decision,
                "decision_id": decision_id_for(args.reviewer_id, row_id),
                "domain": row.get("domain"),
                "era": args.era,
                "event_source_path": str(event_path),
                "family_match_flag": None,
                "gold_instrument_version": row.get("gold_instrument_version"),
                "gold_label": gold_label,
                "gold_source": row.get("gold_source"),
                "grading_model": None,
                "latency_ms": 0.0,
                "rationale_cause_match": None,
                "reviewer_model_quant": args.reviewer_id,
                "rubric_version": args.rubric_version,
                "tokens": 0,
                "tripwire": decision == "reject",
            }
        )

    decisions_path.write_text(
        "".join(canonical_json(row).replace("\n", " ") + "\n" for row in decisions),
        encoding="utf-8",
    )
    label_counts = Counter(str(row["gold_label"]).lower() for row in decisions)
    decision_counts = Counter(str(row["decision"]).lower() for row in decisions)
    manifest = {
        "schema": "reviewer_a0_objective_floor_run_manifest.v1",
        "calibration_command": (
            "python3 /mnt/raid0/llm/epyc-orchestrator/scripts/analysis/"
            "reviewer_calibration_report.py "
            f"--decisions {decisions_path} "
            f"--corpus {args.corpus.expanduser().resolve()} "
            f"--run-manifest {manifest_path} --k 2 --print"
        ),
        "decisions_path": str(decisions_path),
        "generated_at": generated_at,
        "measurement_note": "A0 objective-verifier no-inference floor; emits gold-label decisions.",
        "measurement_protocol": args.measurement_protocol,
        "n_scored": len(decisions),
        "observation_only": args.measurement_protocol != "p_rev1",
        "protocol_attestation": args.protocol_attestation,
    }
    summary = {
        "schema": "reviewer_a0_objective_floor_summary.v1",
        "decisions_path": str(decisions_path),
        "generated_at": generated_at,
        "label_counts": dict(sorted(label_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "measurement_protocol": args.measurement_protocol,
        "n": len(decisions),
        "observation_only": manifest["observation_only"],
        "output_dir": str(output_dir),
        "protocol_attestation": args.protocol_attestation,
        "reviewer_id": args.reviewer_id,
        "row_ids_file": str(args.row_ids_file.expanduser().resolve()),
        "run_manifest": str(manifest_path),
    }
    write_json(manifest_path, manifest)
    write_json(summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--row-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewer-id", default=DEFAULT_REVIEWER_ID)
    parser.add_argument("--rubric-version", default=DEFAULT_RUBRIC_VERSION)
    parser.add_argument("--era", default=DEFAULT_ERA)
    parser.add_argument("--measurement-protocol", default=DEFAULT_PROTOCOL, choices=("p_rev1", "observation"))
    parser.add_argument("--protocol-attestation", default=DEFAULT_ATTESTATION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = materialize(parse_args(argv))
    print(f"a0 objective floor wrote {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
