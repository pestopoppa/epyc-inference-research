#!/usr/bin/env python3
"""Fail-closed validator for one matched A3/A4 raw promptfix capture."""
from __future__ import annotations

import ast
import hashlib
import json
import shlex
import sys
from pathlib import Path

QUESTION_SHA256 = "4b03ad7703bbf2dbaa1eb91b3313cc3cab2892672db87f6242ffd1d489e76375"
RUNNER_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
WATCHDOG_SHA256 = "f4bd45b9617ca880a92be506d741038df65d457f0923f07bc3db7091a7303055"
BINARY_PATH = "/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server"
BINARY_SHA256 = "112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882"
KERNEL = "production-consolidated-v8"
KERNEL_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
ARM_SPECS = {
    "A3_27B_dense_v8_matched_laguna_promptfix_3072": {
        "label": "A3_27B_dense",
        "model": "/mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf",
        "model_sha256": "9408dcb356cc061a05c139e5647cbde0698ff980c6a69f7fc214e9989f86cfa8",
    },
    "A4_35B_A3B_v8_matched_laguna_promptfix_3072": {
        "label": "A4_35B_A3B",
        "model": "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
        "model_sha256": "93dd505d5b4d3f6adcef8c3b6b35465f7537379893f80b87b9ddc2baa62ca557",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: str) -> dict[str, int | str]:
    encoded = value.encode()
    return {"chars": len(value), "utf8_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def capture_schema(path: Path) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CAPTURE_SCHEMA_VERSION":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    raise RuntimeError("runner has no literal CAPTURE_SCHEMA_VERSION")


def expected_server_argv(model: str) -> list[str]:
    return [
        "env", "GGML_IQK=1", "LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin",
        "taskset", "-c", "184-191", BINARY_PATH, "-m", model,
        "--host", "127.0.0.1", "--port", "18093", "--metrics", "--slots",
        "--jinja", "--reasoning", "off", "--device", "ROCm0", "-ngl", "all",
        "-fa", "on", "-np", "1", "-c", "49152", "-t", "8", "-tb", "8",
        "-b", "2048", "-ub", "2048", "-ctk", "f16", "-ctv", "f16",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
    ]


def validate_identity(
    arm: str,
    provenance: dict[str, object],
    server_argv: list[str],
    actual_hashes: dict[str, str],
) -> None:
    spec = ARM_SPECS.get(arm)
    if spec is None:
        raise RuntimeError("unknown arm identity")
    expected_provenance = {
        "schema": "a3_a4_matched_promptfix_capture.v1",
        "label": spec["label"],
        "arm": arm,
        "kernel": KERNEL,
        "kernel_head": KERNEL_HEAD,
        "model": spec["model"],
        "model_sha256": spec["model_sha256"],
        "binary": BINARY_PATH,
        "binary_sha256": BINARY_SHA256,
        "question_sha256": QUESTION_SHA256,
        "runner_source_sha256": RUNNER_SHA256,
        "watchdog_source_sha256": WATCHDOG_SHA256,
        "capture_only": True,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise RuntimeError(f"provenance identity mismatch: {key}")
    expected_hashes = {
        "model": spec["model_sha256"],
        "binary": BINARY_SHA256,
        "runner": RUNNER_SHA256,
        "watchdog": WATCHDOG_SHA256,
        "questions": QUESTION_SHA256,
    }
    for key, expected in expected_hashes.items():
        if actual_hashes.get(key) != expected:
            raise RuntimeError(f"actual artifact hash mismatch: {key}")
    if server_argv != expected_server_argv(str(spec["model"])):
        raise RuntimeError("server argv semantic contract mismatch")


def validate(run: Path, arm: str, *, verify_large_files: bool = True) -> dict[str, object]:
    run = run.resolve()
    required_names = (
        "pq.jsonl", "pq.live-status.json", "runner.json", "runner_source.py",
        "capture_integrity_watchdog.py", "questions_pinned_40.json",
        "expected_question_ids.json", "provenance.json", "server.argv",
    )
    required = [run / name for name in required_names]
    if any(not path.is_file() for path in required):
        raise RuntimeError("missing required raw-capture artifact")
    spec = ARM_SPECS.get(arm)
    if spec is None:
        raise RuntimeError("unknown arm identity")
    actual_hashes = {
        "runner": sha256(run / "runner_source.py"),
        "watchdog": sha256(run / "capture_integrity_watchdog.py"),
        "questions": sha256(run / "questions_pinned_40.json"),
        "model": sha256(Path(str(spec["model"]))) if verify_large_files else str(spec["model_sha256"]),
        "binary": sha256(Path(BINARY_PATH)) if verify_large_files else BINARY_SHA256,
    }
    provenance = json.loads((run / "provenance.json").read_text())
    validate_identity(arm, provenance, shlex.split((run / "server.argv").read_text()), actual_hashes)
    source = run / "runner_source.py"
    if capture_schema(source) != "v7_quality_gate_capture.v4":
        raise RuntimeError("runner capture schema mismatch")
    question_bytes = (run / "questions_pinned_40.json").read_bytes()
    questions = json.loads(question_bytes)
    expected_ids = json.loads((run / "expected_question_ids.json").read_text())
    if len(questions) != 40 or [row.get("id") for row in questions] != expected_ids or len(set(expected_ids)) != 40:
        raise RuntimeError("question denominator mismatch")
    rows = [json.loads(line) for line in (run / "pq.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 40 or [row.get("id") for row in rows] != expected_ids:
        raise RuntimeError("raw rows are not the exact pinned denominator")
    for index, row in enumerate(rows):
        if row.get("suite") != "swebench_oracle" or row.get("arm") != arm or row.get("seed") != 42 or row.get("rep") != 0:
            raise RuntimeError(f"row {index}: draw identity mismatch")
        if row.get("capture_schema_version") != "v7_quality_gate_capture.v4" or row.get("runner_source_sha256") != RUNNER_SHA256:
            raise RuntimeError(f"row {index}: capture provenance mismatch")
        if row.get("request_error") or row.get("finish_reason") == "request_error" or row.get("prompt") != questions[index]["prompt"]:
            raise RuntimeError(f"row {index}: request error or prompt mismatch")
        for field in ("prompt", "response", "reasoning"):
            if not isinstance(row.get(field), str) or row.get(f"{field}_fingerprint") != fingerprint(row[field]):
                raise RuntimeError(f"row {index}: incomplete {field} capture")
    status = json.loads((run / "pq.live-status.json").read_text())
    expected_status = {
        "complete": True, "completed_draws": 40, "expected_draws": 40,
        "request_error_rows": 0, "artifact_integrity_fail_closed": False,
        "suite": "swebench_oracle", "arm": arm,
        "schema_version": "v7_quality_gate_capture.v4", "runner_source_sha256": RUNNER_SHA256,
    }
    if any(status.get(key) != value for key, value in expected_status.items()):
        raise RuntimeError("live status does not attest a clean complete capture")
    summary = json.loads((run / "runner.json").read_text())
    suites = summary.get("suites", [])
    if len(suites) != 1 or suites[0].get("suite") != "swebench_oracle" or suites[0].get("n") != 40 or suites[0].get("errors") != 0:
        raise RuntimeError("runner summary denominator or errors mismatch")
    return {"status": "VALID", "rows": 40, "arm": arm, "capture_schema_version": "v7_quality_gate_capture.v4"}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise RuntimeError(f"usage: {argv[0]} RUN_DIR EXPECTED_ARM")
    print(json.dumps(validate(Path(argv[1]), argv[2]), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except RuntimeError as exc:
        print(f"matched A3/A4 validator: {exc}", file=sys.stderr)
        raise SystemExit(1)
