#!/usr/bin/env python3
"""Validate the exact terminal 27B continuation before A3/A4 recapture."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

EXPECTED_ROOT = Path(
    "/mnt/raid0/llm/epyc-inference-research/artifacts/architect-27b-finetunes-v8-20260726/"
    "live-20260726T1750Z/continuation-27b-v8"
)
MANIFEST_PATH = Path(
    "/mnt/raid0/llm/epyc-inference-research/artifacts/architect-27b-finetunes-v8-20260726/"
    "finetune_bench_manifest.json"
)
MANIFEST_SHA256 = "dab6b5ac7548cb36b9bade443a2a1dcc3a5b377e87add9429da1918be8f8d9ae"
IDENTITY_SHA256 = "6212109e18668e6c7f6ec488cbb573af4bea61ef90ec0ba67391578b4b30cbc2"
BINARY_SHA256 = "112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882"
RUNNER_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
WATCHDOG_SHA256 = "f4bd45b9617ca880a92be506d741038df65d457f0923f07bc3db7091a7303055"
CONVERTER_SHA256 = "6bd2302dda3e5139cc6faabcc5639bdcf85b27895f93a9181cbb53dd65749507"
QUESTION_SHAS = {
    "swe_oracle": "f82a5191274048f2fdf432df7a0ebf4017ad982b954d6aa075326a1302df1c3c",
    "lcb_hard": "d51e56f601e3d153910d086b35c6aea94f4d903bab0427c8a49ffe895a6287c4",
}
MODELS = {
    "thinkingcap": {
        "path": "/mnt/raid0/llm/models/ThinkingCap-Qwen3.6-27B-GGUF/ThinkingCap-Qwen3.6-27B-Q8_0.gguf",
        "sha256": "efcb358ef86f07cf24bfd617a66bb0baa7220e9dd1c31b7d7beacd7b49e67d93",
    },
    "stock_non_mtp": {
        "path": "/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf",
        "sha256": "5927dc06c2b19f732fb6e2a6546dff4c130b552f2ab5f91feb3daafe43897b2a",
    },
    "fable_non_mtp": {
        "path": "/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf",
        "sha256": "2fff409d4a22e0cb11fb0ecfafed1c669b9808f7e6bc499036c6e85297f14f4d",
    },
    "fable_mtp": {
        "path": "/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q8_0.gguf",
        "sha256": "041c175f03b76adb70077ba470258f6b916ec4f5f066077377ef96396c3dd1d0",
    },
}
ARMS = (
    "A3-tc-quality__thinkingcap",
    "A3-ff-quality__stock_non_mtp",
    "A3-ff-quality__fable_non_mtp",
    "A3-ff-embedded-mtp__fable_mtp",
)
SUITES = {
    "swe_oracle": ("swebench_oracle", 40),
    "lcb_hard": ("livecodebench_hard", 53),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate(root: Path, *, enforce_exact_root: bool = True) -> dict[str, object]:
    root = root.resolve()
    if enforce_exact_root and root != EXPECTED_ROOT.resolve():
        fail("continuation output path is not the pinned terminal path")
    marker = root / "continuation.complete"
    identity_path = root / "instrument/identity.json"
    if not marker.is_file() or not identity_path.is_file():
        fail("terminal marker or instrument identity is missing")
    if sha256(identity_path) != IDENTITY_SHA256:
        fail("continuation instrument identity file hash mismatch")
    try:
        datetime.fromisoformat(marker.read_text().strip().replace("Z", "+00:00"))
    except ValueError:
        fail("terminal marker is not an ISO timestamp")
    identity = json.loads(identity_path.read_text())
    expected_identity = {
        "manifest_sha256": MANIFEST_SHA256,
        "server_sha256": BINARY_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "watchdog_sha256": WATCHDOG_SHA256,
        "converter_sha256": CONVERTER_SHA256,
        "capture_schema_version": "v7_quality_gate_capture.v4",
        "question_sha256": QUESTION_SHAS,
    }
    for key, value in expected_identity.items():
        if identity.get(key) != value:
            fail(f"instrument identity mismatch: {key}")
    for key, expected in MODELS.items():
        actual = identity.get("models", {}).get(key, {})
        if actual.get("path") != expected["path"] or actual.get("sha256") != expected["sha256"]:
            fail(f"instrument model identity mismatch: {key}")
    instrument = root / "instrument"
    hashed_inputs = {
        instrument / "v7_quality_gate_runner.py": RUNNER_SHA256,
        instrument / "capture_integrity_watchdog.py": WATCHDOG_SHA256,
        instrument / "convert_sr_to_patch.py": CONVERTER_SHA256,
        instrument / "questions_swe_oracle.json": QUESTION_SHAS["swe_oracle"],
        instrument / "questions_livecodebench_hard.json": QUESTION_SHAS["lcb_hard"],
    }
    for path, expected in hashed_inputs.items():
        if not path.is_file() or sha256(path) != expected:
            fail(f"continuation instrument hash mismatch: {path.name}")
    if not MANIFEST_PATH.is_file() or sha256(MANIFEST_PATH) != MANIFEST_SHA256:
        fail("27B continuation manifest drift")
    newest_evidence_ns = identity_path.stat().st_mtime_ns
    for arm in ARMS:
        for short_suite, (suite, denominator) in SUITES.items():
            status_path = root / arm / f"{short_suite}.sealed.live-status.json"
            rows_path = root / arm / f"{short_suite}.sealed.jsonl"
            summary_path = root / arm / f"{short_suite}.summary.json"
            if any(not path.is_file() for path in (status_path, rows_path, summary_path)):
                fail(f"missing terminal output for {arm}/{short_suite}")
            status = json.loads(status_path.read_text())
            expected_status = {
                "schema_version": "v7_quality_gate_capture.v4",
                "runner_source_sha256": RUNNER_SHA256,
                "suite": suite,
                "arm": arm,
                "complete": True,
                "completed_draws": denominator,
                "expected_draws": denominator,
                "request_error_rows": 0,
                "artifact_integrity_fail_closed": False,
            }
            for key, value in expected_status.items():
                if status.get(key) != value:
                    fail(f"terminal status mismatch for {arm}/{short_suite}: {key}")
            rows = [line for line in rows_path.read_text().splitlines() if line.strip()]
            if len(rows) != denominator:
                fail(f"terminal row denominator mismatch for {arm}/{short_suite}")
            summary = json.loads(summary_path.read_text())
            suites = summary.get("suites", [])
            if len(suites) != 1 or suites[0].get("suite") != suite or suites[0].get("n") != denominator or suites[0].get("errors") != 0:
                fail(f"terminal summary mismatch for {arm}/{short_suite}")
            newest_evidence_ns = max(newest_evidence_ns, status_path.stat().st_mtime_ns, rows_path.stat().st_mtime_ns, summary_path.stat().st_mtime_ns)
    if marker.stat().st_mtime_ns < newest_evidence_ns:
        fail("terminal marker predates continuation evidence")
    return {"status": "VALID", "root": str(root), "arms": 4, "suite_statuses": 8}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} CONTINUATION_ROOT", file=sys.stderr)
        return 2
    print(json.dumps(validate(Path(argv[1])), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except RuntimeError as exc:
        print(f"27B continuation validator: {exc}", file=sys.stderr)
        raise SystemExit(1)
