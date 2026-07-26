from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
QUESTION_SHA = "4b03ad7703bbf2dbaa1eb91b3313cc3cab2892672db87f6242ffd1d489e76375"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = load_module("matched_capture_validator", "validate_matched_capture.py")
CONTINUATION = load_module("continuation_validator", "validate_27b_continuation.py")


def good_provenance(arm: str) -> dict[str, object]:
    spec = CAPTURE.ARM_SPECS[arm]
    return {
        "schema": "a3_a4_matched_promptfix_capture.v1",
        "label": spec["label"],
        "arm": arm,
        "kernel": CAPTURE.KERNEL,
        "kernel_head": CAPTURE.KERNEL_HEAD,
        "model": spec["model"],
        "model_sha256": spec["model_sha256"],
        "binary": CAPTURE.BINARY_PATH,
        "binary_sha256": CAPTURE.BINARY_SHA256,
        "question_sha256": CAPTURE.QUESTION_SHA256,
        "runner_source_sha256": CAPTURE.RUNNER_SHA256,
        "watchdog_source_sha256": CAPTURE.WATCHDOG_SHA256,
        "capture_only": True,
    }


def good_hashes(arm: str) -> dict[str, str]:
    return {
        "model": CAPTURE.ARM_SPECS[arm]["model_sha256"],
        "binary": CAPTURE.BINARY_SHA256,
        "runner": CAPTURE.RUNNER_SHA256,
        "watchdog": CAPTURE.WATCHDOG_SHA256,
        "questions": CAPTURE.QUESTION_SHA256,
    }


def build_continuation_fixture(root: Path, *, complete: bool) -> None:
    source = CONTINUATION.EXPECTED_ROOT / "instrument"
    instrument = root / "instrument"
    instrument.mkdir(parents=True)
    for name in (
        "v7_quality_gate_runner.py", "capture_integrity_watchdog.py",
        "convert_sr_to_patch.py", "questions_swe_oracle.json",
        "questions_livecodebench_hard.json",
    ):
        shutil.copy2(source / name, instrument / name)
    shutil.copy2(source / "identity.json", instrument / "identity.json")
    if complete:
        for arm in CONTINUATION.ARMS:
            arm_dir = root / arm
            arm_dir.mkdir()
            for short_suite, (suite, denominator) in CONTINUATION.SUITES.items():
                status = {
                    "schema_version": "v7_quality_gate_capture.v4",
                    "runner_source_sha256": CONTINUATION.RUNNER_SHA256,
                    "suite": suite,
                    "arm": arm,
                    "complete": True,
                    "completed_draws": denominator,
                    "expected_draws": denominator,
                    "request_error_rows": 0,
                    "artifact_integrity_fail_closed": False,
                }
                (arm_dir / f"{short_suite}.sealed.live-status.json").write_text(json.dumps(status))
                (arm_dir / f"{short_suite}.sealed.jsonl").write_text("{}\n" * denominator)
                summary = {"suites": [{"suite": suite, "n": denominator, "errors": 0}]}
                (arm_dir / f"{short_suite}.summary.json").write_text(json.dumps(summary))
    marker = root / "continuation.complete"
    marker.write_text("2026-07-26T23:59:00Z\n")
    future = max(path.stat().st_mtime_ns for path in root.rglob("*")) + 1_000_000
    os.utime(marker, ns=(future, future))


def test_exact_terminal_laguna_promptfix_question_bytes_are_pinned() -> None:
    subprocess.run(["python3", str(HERE / "prepare_matched_questions.py")], check=True)
    raw = (HERE / "questions_pinned_40.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == QUESTION_SHA
    rows = json.loads(raw)
    assert len(rows) == 40
    assert len({row["id"] for row in rows}) == 40
    assert all(row["prompt"].endswith("mandatory.") for row in rows)


def test_package_self_test_is_non_inferencing() -> None:
    result = subprocess.run(["bash", str(HERE / "run_matched_a3_a4_promptfix.sh"), "--self-test"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "no inference" in result.stdout


def test_manifest_pins_all_execution_identities() -> None:
    subprocess.run(["python3", str(HERE / "prepare_matched_questions.py")], check=True)
    manifest = json.loads((HERE / "prepared_manifest.json").read_text())
    assert manifest["question_sha256"] == QUESTION_SHA
    assert manifest["binary"] == {"path": CAPTURE.BINARY_PATH, "sha256": CAPTURE.BINARY_SHA256}
    assert manifest["kernel"] == {"branch": CAPTURE.KERNEL, "head": CAPTURE.KERNEL_HEAD}
    for arm_name, spec in (("A3_27B_dense", CAPTURE.ARM_SPECS["A3_27B_dense_v8_matched_laguna_promptfix_3072"]), ("A4_35B_A3B", CAPTURE.ARM_SPECS["A4_35B_A3B_v8_matched_laguna_promptfix_3072"])):
        assert manifest["arms"][arm_name]["model"] == spec["model"]
        assert manifest["arms"][arm_name]["model_sha256"] == spec["model_sha256"]
        assert manifest["arms"][arm_name]["hash_evidence"]


@pytest.mark.parametrize("tamper", ["model", "binary", "hash", "server_args", "watchdog"])
def test_capture_identity_gate_rejects_tampering(tamper: str) -> None:
    arm = "A3_27B_dense_v8_matched_laguna_promptfix_3072"
    provenance = copy.deepcopy(good_provenance(arm))
    hashes = good_hashes(arm)
    server_argv = CAPTURE.expected_server_argv(str(CAPTURE.ARM_SPECS[arm]["model"]))
    if tamper == "model":
        provenance["model"] = "/tmp/wrong.gguf"
    elif tamper == "binary":
        provenance["binary"] = "/tmp/wrong-server"
    elif tamper == "hash":
        hashes["model"] = "0" * 64
    elif tamper == "server_args":
        server_argv = server_argv[:-2]
    elif tamper == "watchdog":
        hashes["watchdog"] = "f" * 64
    with pytest.raises(RuntimeError):
        CAPTURE.validate_identity(arm, provenance, server_argv, hashes)


def test_continuation_gate_rejects_fake_partial_marker(tmp_path: Path) -> None:
    build_continuation_fixture(tmp_path, complete=False)
    with pytest.raises(RuntimeError, match="missing terminal output"):
        CONTINUATION.validate(tmp_path, enforce_exact_root=False)


def test_continuation_gate_accepts_only_complete_eight_status_fixture(tmp_path: Path) -> None:
    build_continuation_fixture(tmp_path, complete=True)
    assert CONTINUATION.validate(tmp_path, enforce_exact_root=False)["suite_statuses"] == 8
    broken = tmp_path / CONTINUATION.ARMS[2] / "lcb_hard.sealed.live-status.json"
    status = json.loads(broken.read_text())
    status["request_error_rows"] = 1
    broken.write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="request_error_rows"):
        CONTINUATION.validate(tmp_path, enforce_exact_root=False)


def test_first_live_status_wait_is_liveness_checked_and_not_30_seconds() -> None:
    script = (HERE / "run_matched_a3_a4_promptfix.sh").read_text()
    assert "INITIAL_STATUS_TIMEOUT_S=300" in script
    assert 'wait_for_live_status "$RUN_DIR/pq.live-status.json" "$runner_pid"' in script
    assert 'kill -0 "$runner_pid"' in script
    assert '--startup-grace-s "$INITIAL_STATUS_TIMEOUT_S"' in script
    assert "--startup-grace-s 30 --" not in script
    assert "validate_27b_continuation.py" in script
    assert script.index("run_arm A3_27B_dense") < script.index("run_arm A4_35B_A3B")
