#!/usr/bin/env python3
"""Validate prospective matched-input finite and stress-set records, offline.

This is an input/custody and finiteness control, not a numerical or promotion
verdict. The native backend's existing accuracy and OFF exactness gates apply.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def validate(plan: dict, records: list[dict], base: Path) -> dict:
    if plan.get("schema") != "fa4a-exact-control.v1":
        raise ValueError("plan schema")
    cases = plan.get("case_ids")
    repeats = plan.get("repeats")
    if (not isinstance(cases, list) or not cases or len(cases) != len(set(cases))
            or any(not isinstance(case, str) or not case for case in cases)
            or type(repeats) is not int or repeats < 1):
        raise ValueError("plan cases or repeats")
    wanted = {(case, bias, arm, repeat) for case in cases for bias in (256, 1024)
              for arm in ("ON", "OFF") for repeat in range(1, repeats + 1)}
    indexed = {}
    for record in records:
        key = (record.get("case_id"), record.get("bias"), record.get("arm"), record.get("repeat"))
        if key not in wanted or key in indexed:
            raise ValueError("unexpected or duplicate case")
        digest = record.get("input_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid input digest")
        n = record.get("nonfinite_elements")
        if type(n) is not int or n < 0:
            raise ValueError("invalid nonfinite count")
        path = base / record.get("raw_file", "")
        raw_digest = record.get("raw_sha256")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != raw_digest:
            raise ValueError("raw evidence missing or changed")
        raw = json.loads(path.read_text())
        for field in ("case_id", "bias", "arm", "repeat", "input_sha256", "nonfinite_elements"):
            if raw.get(field) != record.get(field):
                raise ValueError("raw evidence disagrees with index")
        input_path = base / record.get("input_file", "")
        if not input_path.is_file() or hashlib.sha256(input_path.read_bytes()).hexdigest() != digest:
            raise ValueError("input bytes missing or changed")
        indexed[key] = record
    if set(indexed) != wanted:
        raise ValueError("incomplete case matrix")
    for case in cases:
        for bias in (256, 1024):
            digests = {indexed[(case, bias, arm, repeat)]["input_sha256"]
                       for arm in ("ON", "OFF") for repeat in range(1, repeats + 1)}
            if len(digests) != 1:
                raise ValueError("matched input identity differs")
    finite_failures = [list(key) for key, row in indexed.items()
                       if key[1] == 256 and row["nonfinite_elements"]]
    membership = {arm: {case: [indexed[(case, 1024, arm, repeat)]["nonfinite_elements"]
                               for repeat in range(1, repeats + 1)] for case in cases}
                  for arm in ("ON", "OFF")}
    return {"schema": "fa4a-exact-control-report.v1", "cases_per_bias_arm_repeat": len(cases),
            "repeats": repeats, "bias256_nonfinite_cases": finite_failures,
            "bias256_finite_control": "FAIL" if finite_failures else "PASS",
            "bias1024_nonfinite_elements_by_case": membership,
            "numeric_accuracy": "NOT_EVALUATED", "promotion": "NOT_EVALUATED"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    records = json.loads(args.records.read_text())
    result = validate(plan, records, args.records.parent)
    with args.output.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
