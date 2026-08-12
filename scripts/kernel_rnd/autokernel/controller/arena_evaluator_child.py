#!/usr/bin/env python3
"""Strict, claim-blind AgentKernelArena candidate evaluator child.

The parent owns the device claim, sampler, sandbox process and durable receipt.
This child can only consume one self-hashed request and emit one JSON result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "epyc.autokernel.arena_evaluator_child_request.v1"
RESULT_SCHEMA = "epyc.autokernel.arena_evaluator_child_result.v1"
BASELINE_SCHEMA = "epyc.autokernel.arena_baseline_cases.v1"


class EvaluatorChildError(RuntimeError):
    """The evaluator child input or vendor output violated its strict contract."""


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def verify_self_hash(payload: Mapping[str, Any], label: str) -> None:
    claimed = payload.get("receipt_sha256")
    bare = {key: value for key, value in payload.items()
            if key != "receipt_sha256"}
    if not isinstance(claimed, str) or canonical_sha256(bare) != claimed:
        raise EvaluatorChildError(f"{label} self-hash does not verify")


def _json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluatorChildError(f"{label} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise EvaluatorChildError(f"{label} keys must be strings")
        return {key: _json_value(item, label)
                for key, item in sorted(value.items())}
    raise EvaluatorChildError(f"{label} is not strict JSON data")


def serialize_baseline_cases(cases: Sequence[Any]) -> dict[str, Any]:
    rows = []
    for case in cases:
        test_case_id = getattr(case, "test_case_id", None)
        execution_time = getattr(case, "execution_time_ms", None)
        if (not isinstance(test_case_id, str) or not test_case_id
                or isinstance(execution_time, bool)
                or not isinstance(execution_time, (int, float))
                or not math.isfinite(float(execution_time))):
            raise EvaluatorChildError("baseline case identity or timing is invalid")
        rows.append({
            "test_case_id": test_case_id,
            "shape": _json_value(getattr(case, "shape", None), "baseline shape"),
            "execution_time_ms": float(execution_time),
            "metadata": _json_value(
                getattr(case, "metadata", None), "baseline metadata"),
        })
    return self_hash({"schema": BASELINE_SCHEMA, "cases": rows})


def reconstruct_baseline_cases(document: Mapping[str, Any], case_type: Any) -> list[Any]:
    verify_self_hash(document, "baseline cases")
    rows = document.get("cases")
    if document.get("schema") != BASELINE_SCHEMA or not isinstance(rows, list):
        raise EvaluatorChildError("baseline cases schema is invalid")
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
                "test_case_id", "shape", "execution_time_ms", "metadata"}:
            raise EvaluatorChildError("baseline case fields are invalid")
        result.append(case_type(
            test_case_id=row["test_case_id"], shape=row["shape"],
            execution_time_ms=row["execution_time_ms"], metadata=row["metadata"]))
    return result


def _strict_evaluation(raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "pass_compilation", "pass_correctness", "best_optimized_execution_time",
        "average_speedup", "valid_baseline_cases", "valid_optimized_cases",
        "compilation_error_message", "correctness_error_message",
    }
    if set(raw) - allowed:
        raise EvaluatorChildError("vendor evaluation emitted undeclared fields")
    result = {key: _json_value(raw.get(key), f"evaluation.{key}")
              for key in sorted(allowed)}
    if not isinstance(result["pass_compilation"], bool) \
            or not isinstance(result["pass_correctness"], bool):
        raise EvaluatorChildError("vendor evaluation pass fields are invalid")
    for key in ("valid_baseline_cases", "valid_optimized_cases"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) \
                or result[key] < 0:
            raise EvaluatorChildError(f"vendor evaluation {key} is invalid")
    return result


def evaluate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    verify_self_hash(request, "evaluator child request")
    required = {
        "schema", "campaign_id", "claim_campaign_id", "task_id", "arm_id",
        "checkpoint_hours", "phase", "evaluation_ordinal", "workspace",
        "config_sha256", "arena_root", "vendor_evaluator_sha256",
        "evaluator_python", "baseline_cases", "outer_baseline_receipt_sha256",
        "authority", "receipt_sha256",
    }
    if "attempt_id" in request:
        required.add("attempt_id")
    if set(request) != required or request.get("schema") != REQUEST_SCHEMA:
        raise EvaluatorChildError("evaluator child request schema is invalid")
    if request.get("phase") not in {
            "controller_intermediate_evaluation",
            "centralized_final_evaluation"}:
        raise EvaluatorChildError("evaluator child phase is invalid")
    for field in ("config_sha256", "vendor_evaluator_sha256",
                  "outer_baseline_receipt_sha256"):
        value = request.get(field)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise EvaluatorChildError(f"evaluator child {field} is invalid")
    python_identity = request.get("evaluator_python")
    if (not isinstance(python_identity, Mapping)
            or Path(str(python_identity.get("resolved_path"))).resolve()
            != Path(sys.executable).resolve()
            or hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
            != python_identity.get("sha256")):
        raise EvaluatorChildError("evaluator Python identity drifted")
    workspace = Path(str(request.get("workspace"))).resolve()
    config_path = workspace / "config.yaml"
    if not workspace.is_dir() or workspace.is_symlink() or not config_path.is_file():
        raise EvaluatorChildError("evaluator workspace is unsafe")
    expected_config = request.get("config_sha256")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != expected_config:
        raise EvaluatorChildError("evaluator config identity drifted")
    arena_root = Path(str(request.get("arena_root"))).resolve()
    sys.path.insert(0, str(arena_root))
    try:
        import yaml  # type: ignore[import-not-found]
        from src import evaluator  # type: ignore[import-not-found]
        from src.testcases import TestCaseResult  # type: ignore[import-not-found]
    except ImportError as exc:
        raise EvaluatorChildError("pinned Arena evaluator cannot be imported") from exc
    expected_vendor = request.get("vendor_evaluator_sha256")
    if hashlib.sha256(Path(evaluator.__file__).read_bytes()).hexdigest() != expected_vendor:
        raise EvaluatorChildError("vendor evaluator identity drifted")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise EvaluatorChildError("Arena task config must be an object")
    baseline = reconstruct_baseline_cases(request["baseline_cases"], TestCaseResult)
    logger = logging.getLogger(f"autokernel.arena.child.{Path.cwd().name}")
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.INFO)
    evaluation = evaluator.evaluate_kernel(workspace, config, baseline, logger, None)
    if not isinstance(evaluation, Mapping):
        raise EvaluatorChildError("vendor evaluator returned a non-object")
    return self_hash({
        "schema": RESULT_SCHEMA,
        "request_receipt_sha256": request["receipt_sha256"],
        "baseline_cases_sha256": request["baseline_cases"]["receipt_sha256"],
        "outer_baseline_receipt_sha256": request["outer_baseline_receipt_sha256"],
        "evaluation": _strict_evaluation(evaluation),
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise EvaluatorChildError("evaluator child request must be an object")
        result = evaluate_request(request)
    except (EvaluatorChildError, OSError, json.JSONDecodeError,
            KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"schema": RESULT_SCHEMA, "status": "error",
                          "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
