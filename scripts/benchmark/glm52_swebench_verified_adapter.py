#!/usr/bin/env python3
"""No-inference SWE-bench-Verified adapter for GLM-5.2 reviewer gates.

This adapter materializes SWE-bench-Verified rows into a mechanical patch-review
oracle contract. It does not start Docker, run tests, launch servers, or load
models; downstream reviewer-quality gates can consume the normalized rows later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = "glm52_swebench_verified_adapter.v1"
TASK_KIND = "patch_review_oracle"
GOLD_SOURCE = "swe-bench-verified"
DEFAULT_PARQUET_PATH = Path("/mnt/raid0/llm/datasets/swe-bench-verified/data/test-00000-of-00001.parquet")


@dataclass(frozen=True)
class SwebenchVerifiedRow:
    row_id: str
    repo: str
    instance_id: str
    problem_statement: str
    task: str
    patch: str
    candidate: str
    base_commit: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    gold_label: str
    gold_source: str
    gold_confidence: str
    gold_instrument_version: str
    task_kind: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_version(path: Path) -> str:
    return f"file-sha256:{sha256_file(path)}"


def require_text(row: dict[str, Any], key: str, *, source: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: missing non-empty string field {key!r}")
    return value


def normalize_test_list(value: Any, *, key: str, source: str) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: field {key!r} must be a JSON list string") from exc
        value = decoded
    if not isinstance(value, list):
        raise ValueError(f"{source}: field {key!r} must be a list")
    tests = [item for item in value if isinstance(item, str) and item.strip()]
    if len(tests) != len(value):
        raise ValueError(f"{source}: field {key!r} must contain only non-empty strings")
    return tests


def stable_row_id(repo: str, instance_id: str, base_commit: str, patch: str) -> str:
    key = "\x00".join([repo, instance_id, base_commit, patch])
    return f"glm52-swebench-verified:{hashlib.sha1(key.encode('utf-8')).hexdigest()[:20]}"


def normalize_swebench_verified_record(
    raw: dict[str, Any],
    *,
    row_index: int,
    gold_instrument_version: str,
) -> SwebenchVerifiedRow:
    source = f"swe-bench-verified row {row_index}"
    repo = require_text(raw, "repo", source=source)
    instance_id = require_text(raw, "instance_id", source=source)
    problem_statement = require_text(raw, "problem_statement", source=source)
    patch = require_text(raw, "patch", source=source)
    base_commit = require_text(raw, "base_commit", source=source)
    fail_to_pass = normalize_test_list(raw.get("FAIL_TO_PASS"), key="FAIL_TO_PASS", source=source)
    pass_to_pass = normalize_test_list(raw.get("PASS_TO_PASS"), key="PASS_TO_PASS", source=source)
    return SwebenchVerifiedRow(
        row_id=stable_row_id(repo, instance_id, base_commit, patch),
        repo=repo,
        instance_id=instance_id,
        problem_statement=problem_statement,
        task=problem_statement,
        patch=patch,
        candidate=patch,
        base_commit=base_commit,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        gold_label="accept",
        gold_source=GOLD_SOURCE,
        gold_confidence="test_oracle",
        gold_instrument_version=gold_instrument_version,
        task_kind=TASK_KIND,
        provenance={
            "source_row_index": row_index,
            "test_patch_present": bool(str(raw.get("test_patch") or "").strip()),
            "difficulty": raw.get("difficulty"),
            "version": raw.get("version"),
            "environment_setup_commit": raw.get("environment_setup_commit"),
        },
    )


def normalize_swebench_verified_records(
    records: Iterable[dict[str, Any]],
    *,
    gold_instrument_version: str,
) -> list[SwebenchVerifiedRow]:
    return [
        normalize_swebench_verified_record(raw, row_index=idx, gold_instrument_version=gold_instrument_version)
        for idx, raw in enumerate(records)
    ]


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SWE-bench-Verified parquet loading requires pyarrow. "
            "Run with `uv run --with pyarrow scripts/benchmark/glm52_swebench_verified_adapter.py ...` "
            "or call normalize_swebench_verified_records() with record dicts."
        ) from exc
    table = pq.read_table(path)
    return table.to_pylist()


def load_swebench_verified(path: Path) -> list[SwebenchVerifiedRow]:
    return normalize_swebench_verified_records(read_parquet_records(path), gold_instrument_version=source_version(path))


def stable_selection_hash(seed_key: str, row_id: str) -> str:
    return hashlib.sha1(f"{seed_key}\x00{row_id}".encode("utf-8")).hexdigest()


def select_rows(rows: list[SwebenchVerifiedRow], *, n: int, seed: int) -> list[SwebenchVerifiedRow]:
    if n <= 0:
        return []
    selected = sorted(rows, key=lambda row: stable_selection_hash(str(seed), row.row_id))
    return selected[:n]


def summarize_rows(rows: list[SwebenchVerifiedRow]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "repo_counts": dict(Counter(row.repo for row in rows)),
        "gold_label_counts": dict(Counter(row.gold_label for row in rows)),
        "task_kind_counts": dict(Counter(row.task_kind for row in rows)),
        "fail_to_pass_total": sum(len(row.FAIL_TO_PASS) for row in rows),
        "pass_to_pass_total": sum(len(row.PASS_TO_PASS) for row in rows),
    }


def iter_jsonl_rows(rows: Iterable[SwebenchVerifiedRow]) -> Iterator[str]:
    for row in rows:
        yield json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_swebench_verified(args.path)
    selected = select_rows(rows, n=args.n, seed=args.seed)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run",
        "execution": {
            "docker_allowed": False,
            "inference_allowed": False,
            "server_or_model_load_allowed": False,
            "live_execution_allowed": False,
        },
        "dataset": {
            "kind": GOLD_SOURCE,
            "path": str(args.path),
            "file_sha256": sha256_file(args.path),
            "gold_instrument_version": source_version(args.path),
            "available": summarize_rows(rows),
            "selected": summarize_rows(selected),
            "selected_row_ids": [row.row_id for row in selected],
        },
        "oracle_contract": {
            "task_kind": TASK_KIND,
            "gold_label": "accept",
            "gold_source": GOLD_SOURCE,
            "gold_confidence": "test_oracle",
            "candidate_field": "patch",
            "task_field": "problem_statement",
            "oracle_fields": ["FAIL_TO_PASS", "PASS_TO_PASS"],
        },
        "refusal_reasons": [] if selected else ["no selected rows"],
    }


def write_selected_rows(path: Path, rows: list[SwebenchVerifiedRow], selected_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row.row_id in selected_ids:
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=DEFAULT_PARQUET_PATH)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--out-plan", type=Path)
    ap.add_argument("--out-rows-jsonl", type=Path)
    args = ap.parse_args(argv)

    plan = build_plan(args)
    if args.out_plan:
        write_json(args.out_plan, plan)
    if args.out_rows_jsonl:
        rows = load_swebench_verified(args.path)
        write_selected_rows(args.out_rows_jsonl, rows, set(plan["dataset"]["selected_row_ids"]))
    if not args.out_plan:
        print(canonical_json(plan))
    return 0 if plan["dataset"]["selected"]["n"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
