#!/usr/bin/env python3
"""Build a no-evaluation metadata successor for the published six-arm table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "expanded-six-arm-v4-tail-replay-20260727"
ORIGINAL_TABLE = ORIGINAL / "expanded_six_arm_table.json"
ORIGINAL_FINALIZATION = ORIGINAL / "finalization.sha256"
ORIGINAL_TABLE_SHA256 = "a5081b84645b089e0989d81bb148682f7c54d2c0ff0da03fcc7ba7e816a4724b"
ORIGINAL_FINALIZATION_SHA256 = "c7112a218ec3cddb8a6e56a0c5b3edcc4ef715a7becd67214f483edea75d5364"
APPENDED = {"A3-tc", "A3-ff"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_original_finalization() -> dict[str, Any]:
    if sha256(ORIGINAL_TABLE) != ORIGINAL_TABLE_SHA256 or sha256(ORIGINAL_FINALIZATION) != ORIGINAL_FINALIZATION_SHA256:
        fail("original package/table binding drifted")
    expected = {}
    for line in ORIGINAL_FINALIZATION.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    actual = {str(path.relative_to(ORIGINAL)): sha256(path) for path in ORIGINAL.rglob("*")
              if path.is_file() and path.name != "finalization.sha256"}
    if expected != actual:
        fail("original package finalization ledger drifted")
    table = json.loads(ORIGINAL_TABLE.read_text())
    if table.get("status") != "APPEND_ONLY_SUCCESSOR_NO_ROLE_DECISION" or len(table.get("rows", [])) != 6:
        fail("original table is not the expected finalized six-arm package")
    return table


def corrected_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in table["rows"]:
        row = dict(source)
        report = Path(row["report"])
        if row["arm"] in APPENDED:
            target = ORIGINAL / row["arm"] / report.name
            corrected = f"{row['arm']}/{report.name}"
        else:
            target = report
            corrected = str(report)
        if not target.is_file() or sha256(target) != row["report_sha256"]:
            fail(f"report binding drifted for {row['arm']}")
        row["report"] = corrected
        rows.append(row)
    return rows


def preflight() -> dict[str, Any]:
    table = verify_original_finalization()
    rows = corrected_rows(table)
    if any(row["report"].startswith("/") for row in rows if row["arm"] in APPENDED):
        fail("corrected appended report path is not package-relative")
    return {"status": "PRECHECK_OK", "no_evaluation": True, "no_inference": True,
            "original_table_sha256": ORIGINAL_TABLE_SHA256, "rows": len(rows)}


def publish(output: Path) -> None:
    table = verify_original_finalization(); rows = corrected_rows(table)
    if output.exists():
        fail("successor output already exists; refusing overwrite")
    output.mkdir(parents=True)
    successor = {"schema_version": "epyc.expanded-six-arm-v4-report-path-correction.v1",
                 "status": "FINALIZED_METADATA_ONLY_NO_EVAL_NO_INFERENCE",
                 "original_package": {"path": str(ORIGINAL), "table_sha256": ORIGINAL_TABLE_SHA256,
                                      "finalization_sha256": ORIGINAL_FINALIZATION_SHA256},
                 "correction": "A3-tc and A3-ff report paths changed from stale staging absolutes to package-relative paths; all rows, scores, report hashes, and non-path metadata are unchanged.",
                 "rows": rows, "source_table_metadata": {key: value for key, value in table.items() if key != "rows"}}
    write_json(output / "path_correction_successor.json", successor)
    (output / "finalization.sha256").write_text(f"{sha256(output / 'path_correction_successor.json')}  path_correction_successor.json\n")
    print(json.dumps({"status": "FINALIZED", "output": str(output)}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--preflight", action="store_true"); group.add_argument("--publish", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.preflight: print(json.dumps(preflight(), sort_keys=True))
        else:
            if not args.output_dir: fail("--publish requires --output-dir")
            publish(args.output_dir)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
