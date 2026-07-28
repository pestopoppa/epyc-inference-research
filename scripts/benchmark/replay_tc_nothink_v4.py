#!/usr/bin/env python3
"""Seal and officially score the banked FG-3 ThinkingCap no-think SWE capture.

This is a deterministic scoring-only successor.  It accepts exactly the
completed ``A3-tc-nothink__thinkingcap`` capture, runs the frozen v4
SEARCH/REPLACE converter, and invokes the pinned SWE-bench harness.  It never
contacts a model endpoint or generates completions.  It does not amend the
earlier four- or six-arm tables; a table owner may consume this sealed result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CANONICAL_REPO = Path("/mnt/raid0/llm/epyc-inference-research")
CAPTURE_ARM = "A3-tc-nothink__thinkingcap"
CAPTURE_SCHEMA = "v7_quality_gate_capture.v4"
RUNNER_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
EXPECTED: dict[str, tuple[str, str]] = {
    "capture": (
        "artifacts/architect-27b-finetunes-v8-20260726/fg3-tc-nothink-validation-20260727/swe_oracle_nothink.sealed.jsonl",
        "b3e063c8d32cf0e87eea2b57603ce0e7a55aaaccca065fbd1482bf53e6643903",
    ),
    "capture_summary": (
        "artifacts/architect-27b-finetunes-v8-20260726/fg3-tc-nothink-validation-20260727/swe_oracle_nothink.summary.json",
        "4bc8e0e05c350b028585e22648fd296401922f33c2966ce75ea094640603a975",
    ),
    "capture_argv": (
        "artifacts/architect-27b-finetunes-v8-20260726/fg3-tc-nothink-validation-20260727/swe_oracle_nothink.evaluator.argv",
        "aae1b8fefd32c8f8800668ba02051ce6c9d37c90baef47e2b4057a512b8ae858",
    ),
    "capture_done": (
        "artifacts/architect-27b-finetunes-v8-20260726/fg3-tc-nothink-validation-20260727/capture.done",
        "8221ac66be71558c921fb44cfb66f7997699aea754d917763882d6d9eddc836e",
    ),
    "frozen_table": (
        "artifacts/architect-same-era-v8-20260726/final-4arm-v4-tail-replay-20260727/runs/final-4arm-v4-tail-replay-20260727T080703Z/final_4arm_table.json",
        "f440680d7a6ae2169b35202627de7889441c530b4976f23fd4140e84be8f2e50",
    ),
    "authority_finalization": (
        "artifacts/architect-same-era-v8-20260726/final-4arm-v4-tail-replay-20260727/runs/final-4arm-v4-tail-replay-20260727T080703Z/finalization.sha256",
        "21a5de5c36018487bf7c6bf00e5e46a6cc2c5d2773a12cbba8279d34094aea23",
    ),
    "authority_ids": (
        "artifacts/architect-same-era-v8-20260726/final-4arm-v4-tail-replay-20260727/runs/final-4arm-v4-tail-replay-20260727T080703Z/A3/raw_capture.sealed.jsonl",
        "1fa07aff0ae46b1300104b87a3ea96daaaa8dda9c9a028dd6e22b1879effc3c2",
    ),
    "converter": (
        "artifacts/architect-same-era-v8-20260726/final-4arm-v4-tail-replay-20260727/runs/final-4arm-v4-tail-replay-20260727T080703Z/A3/converter_v4.sealed.py",
        "6bd2302dda3e5139cc6faabcc5639bdcf85b27895f93a9181cbb53dd65749507",
    ),
    "harness": (
        "artifacts/architect-same-era-v8-20260726/final-4arm-v4-tail-replay-20260727/runs/final-4arm-v4-tail-replay-20260727T080703Z/A3/harness_module.sealed.py",
        "6959f0b4e4eaf979771f529b88e3e9df1daa7fe86bc4291feec2e7d320bf7f2e",
    ),
    "dataset": (
        "artifacts/architect-same-era-v8-20260726/final-4arm-v4-tail-replay-20260727/runs/final-4arm-v4-tail-replay-20260727T080703Z/A3/swebench_verified.sealed.json",
        "b087b5dad72b3e765a6cf93a9e7d516d8796698a0fd358abb73c6627df19f66e",
    ),
}
SWEBENCH_PYTHON = Path("/mnt/raid0/llm/epyc-inference-research/.venv-swebench/bin/python")
CANONICAL_REPOS_REL = "artifacts/architect-code-eval-20260724/swebench_repos"
CPUSET_ENV = "SWEBENCH_EVAL_CPUSET"
CANONICAL_CPUSET = "112-119"
ADAPTER_SOURCE = Path(__file__).with_name("swebench_cpuset_adapter.py")
DOCKER_BUILD_SOURCE = Path("/mnt/raid0/llm/epyc-inference-research/.venv-swebench/lib/python3.12/site-packages/swebench/harness/docker_build.py")
DOCKER_BUILD_SHA256 = "5278842b60a7d38256f95f93c915dc84de2b8b4f286e9baae1b19280f768e484"


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class Inputs:
    root: Path

    def path(self, key: str) -> Path:
        return self.root / EXPECTED[key][0]

    def digest(self, key: str) -> str:
        return EXPECTED[key][1]


def fail(message: str) -> None:
    raise ReplayError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def text_fingerprint(value: str) -> dict[str, int | str]:
    encoded = value.encode("utf-8")
    return {"chars": len(value), "utf8_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def verify_inputs(inputs: Inputs) -> None:
    for key in EXPECTED:
        path = inputs.path(key)
        if not path.is_file() or sha256(path) != inputs.digest(key):
            fail(f"pinned input drifted: {key}: {path}")


def authority_ids(inputs: Inputs) -> list[str]:
    rows = load_jsonl(inputs.path("authority_ids"))
    ids = [row.get("id") for row in rows]
    if len(ids) != 40 or len(set(ids)) != 40 or not all(isinstance(value, str) for value in ids):
        fail("frozen authority does not bind an ordered 40-ID denominator")
    return ids


def validate_capture(inputs: Inputs, ids: list[str]) -> list[dict[str, Any]]:
    if inputs.path("capture_done").read_text().strip() != "DONE":
        fail("FG-3 capture completion marker is not terminal")
    argv = inputs.path("capture_argv").read_text()
    required_argv = ("--no-enable-thinking", f"--arm {CAPTURE_ARM}", "--n 40", "--max-tokens 3072")
    if any(token not in argv for token in required_argv) or " --enable-thinking" in argv:
        fail("FG-3 evaluator argv is not the pinned no-think contract")
    summary = json.loads(inputs.path("capture_summary").read_text())
    meta = summary.get("meta", {})
    if (meta.get("arm") != CAPTURE_ARM or meta.get("enable_thinking") is not False
            or meta.get("n_per_suite") != 40 or meta.get("max_tokens") != 3072
            or meta.get("seed") != 42 or meta.get("runner_source_sha256") != RUNNER_SHA256):
        fail("FG-3 capture summary identity drifted")
    rows = load_jsonl(inputs.path("capture"))
    if [row.get("id") for row in rows] != ids:
        fail("FG-3 capture does not preserve the frozen 40-ID order")
    for row in rows:
        if (row.get("capture_schema_version") != CAPTURE_SCHEMA or row.get("arm") != CAPTURE_ARM
                or row.get("suite") != "swebench_oracle" or row.get("seed") != 42
                or row.get("rep") != 0 or row.get("runner_source_sha256") != RUNNER_SHA256):
            fail(f"FG-3 capture identity drifted at {row.get('id')}")
        if row.get("request_error") or row.get("finish_reason") == "request_error":
            fail(f"FG-3 capture has request-error evidence at {row.get('id')}")
        for field in ("prompt", "response", "reasoning"):
            if not isinstance(row.get(field), str) or row.get(f"{field}_fingerprint") != text_fingerprint(row[field]):
                fail(f"FG-3 capture lacks a complete fingerprinted {field} at {row.get('id')}")
    return rows


def installed_harness() -> Path:
    if not SWEBENCH_PYTHON.is_file():
        fail("pinned SWE-bench interpreter is unavailable")
    result = subprocess.run(
        [str(SWEBENCH_PYTHON), "-c", "import inspect,swebench.harness.run_evaluation as m; print(inspect.getsourcefile(m))"],
        text=True, capture_output=True, check=False,
    )
    path = Path(result.stdout.strip())
    if result.returncode or not path.is_file() or sha256(path) != EXPECTED["harness"][1]:
        fail("installed SWE-bench harness SHA-256 drifted")
    return path


def installed_docker_build() -> Path:
    if not SWEBENCH_PYTHON.is_file():
        fail("pinned SWE-bench interpreter is unavailable")
    result = subprocess.run(
        [str(SWEBENCH_PYTHON), "-c", "import inspect,swebench.harness.docker_build as m; print(inspect.getsourcefile(m))"],
        text=True,
        capture_output=True,
        check=False,
    )
    path = Path(result.stdout.strip())
    if result.returncode or path != DOCKER_BUILD_SOURCE or not path.is_file() or sha256(path) != DOCKER_BUILD_SHA256:
        fail("installed SWE-bench docker_build SHA-256 drifted")
    return path


def validate_repo_mirror(inputs: Inputs, ids: list[str]) -> None:
    dataset = {row.get("instance_id"): row for row in json.loads(inputs.path("dataset").read_text())}
    mirror = inputs.root / CANONICAL_REPOS_REL
    for instance_id in ids:
        row = dataset.get(instance_id)
        if not isinstance(row, dict) or not isinstance(row.get("repo"), str) or not isinstance(row.get("base_commit"), str):
            fail(f"dataset lacks identity for {instance_id}")
        git_dir = mirror / row["repo"].replace("/", "__")
        result = subprocess.run(["git", "--git-dir", str(git_dir), "rev-parse", f"{row['base_commit']}^{{commit}}"], text=True, capture_output=True, check=False)
        if result.returncode or result.stdout.strip() != row["base_commit"]:
            fail(f"repo mirror lacks the required base commit for {instance_id}")


def preflight(inputs: Inputs) -> None:
    verify_inputs(inputs)
    ids = authority_ids(inputs)
    validate_capture(inputs, ids)
    installed_harness()
    installed_docker_build()
    validate_repo_mirror(inputs, ids)
    print(json.dumps({"status": "PRECHECK_OK", "arm": CAPTURE_ARM, "no_inference": True, "denominator": len(ids)}, sort_keys=True))


def stage_converter(stage: Path, inputs: Inputs) -> types.ModuleType:
    authority = stage / "authority"
    authority.mkdir(parents=True)
    converter = authority / "converter_v4.sealed.py"
    # The frozen converter imports its sibling by this historical name.
    dataset = authority / "swebench_verified.json"
    shutil.copyfile(inputs.path("converter"), converter)
    shutil.copyfile(inputs.path("dataset"), dataset)
    if sha256(converter) != inputs.digest("converter") or sha256(dataset) != inputs.digest("dataset"):
        fail("staged converter or dataset hash mismatch")
    os.symlink(inputs.root / CANONICAL_REPOS_REL, authority / "swebench_repos")
    module = types.ModuleType("fg3_frozen_converter")
    module.__file__ = str(converter)
    exec(compile(converter.read_text(), str(converter), "exec"), module.__dict__)
    return module


def convert(rows: list[dict[str, Any]], converter: types.ModuleType) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, int]]:
    predictions: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    counts = {"blocks_applied": 0, "blocks_skipped": 0, "empty_patch_count": 0, "length_cap_model_failures": 0}
    for row in rows:
        blocks: list[dict[str, Any]] = []
        if row.get("finish_reason") == "length":
            patch, applied, skipped = "", 0, 0
            counts["length_cap_model_failures"] += 1
        else:
            patch, applied, skipped = converter.apply_blocks(converter.rows[row["id"]], row["response"], blocks)
        diagnostic = converter.row_diagnostic(row, patch, blocks, RUNNER_SHA256)
        if not diagnostic.get("scoring_eligible"):
            fail(f"frozen v4 converter integrity failed at {row['id']}")
        if row.get("finish_reason") == "length":
            diagnostic.update({"empty_patch": True, "empty_patch_reason": "model_length_cap", "conversion_disposition": "model_failure_length_cap"})
        predictions.append({"instance_id": row["id"], "model_name_or_path": "A3-tc_ThinkingCap_Q8_nothink", "model_patch": patch})
        diagnostics.append(diagnostic)
        counts["blocks_applied"] += applied
        counts["blocks_skipped"] += skipped
        counts["empty_patch_count"] += not bool(patch)
    return predictions, diagnostics, counts


def nonrecovery_ledger(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    skipped, empty = [], []
    for diagnostic in diagnostics:
        for block in diagnostic.get("blocks", []):
            if str(block.get("outcome", "")).startswith("skipped_"):
                skipped.append({"instance_id": diagnostic["instance_id"], "block_index": block["block_index"], "outcome": block["outcome"],
                                "search_sha256": block["search_sha256"], "replace_sha256": block["replace_sha256"],
                                "classification": "model_contract_failure", "additional_recovery_attempted": False,
                                "disposition": "preserve_frozen_v4_outcome_no_match_broadening"})
        if diagnostic.get("empty_patch"):
            empty.append({"instance_id": diagnostic["instance_id"], "finish_reason": diagnostic.get("finish_reason"),
                          "empty_patch_reason": diagnostic.get("empty_patch_reason"), "additional_recovery_attempted": False,
                          "disposition": "preserve_frozen_v4_outcome_no_match_broadening"})
    return {"schema_version": "epyc.fg3-tc-nothink-v4-nonrecovery-ledger.v1", "status": "EXHAUSTIVE_PINNED_V4_OUTCOME",
            "arm": CAPTURE_ARM, "skipped_blocks": skipped, "empty_patch_rows": empty,
            "aggregate": {"skipped_block_count": len(skipped), "empty_patch_row_count": len(empty)}}


def validate_ledger(diagnostics: list[dict[str, Any]], ledger: dict[str, Any]) -> None:
    actual_skips = {(diag["instance_id"], block["block_index"], block["outcome"], block["search_sha256"], block["replace_sha256"])
                    for diag in diagnostics for block in diag.get("blocks", []) if str(block.get("outcome", "")).startswith("skipped_")}
    ledger_skips = {(row["instance_id"], row["block_index"], row["outcome"], row["search_sha256"], row["replace_sha256"])
                    for row in ledger["skipped_blocks"]}
    actual_empty = {diag["instance_id"] for diag in diagnostics if diag.get("empty_patch")}
    if ledger_skips != actual_skips or {row["instance_id"] for row in ledger["empty_patch_rows"]} != actual_empty:
        fail("non-recovery ledger is not exhaustive")
    if any(row.get("additional_recovery_attempted") is not False for row in ledger["skipped_blocks"] + ledger["empty_patch_rows"]):
        fail("non-recovery ledger contains unsupported recovery")


def seal(
    stage: Path,
    inputs: Inputs,
    ids: list[str],
    rows: list[dict[str, Any]],
    converter: types.ModuleType,
    sealer: Path,
    sealer_source: Path,
    adapter_source: Path,
    harness_source: Path,
    docker_build_source: Path,
) -> Path:
    arm = stage / "A3-tc-nothink"
    arm.mkdir()
    for key, name in (("capture", "raw_capture.sealed.jsonl"), ("capture_summary", "capture_summary.sealed.json"),
                      ("capture_argv", "capture.evaluator.argv"), ("capture_done", "capture.done"),
                      ("converter", "converter_v4.sealed.py"), ("dataset", "swebench_verified.sealed.json"),
                      ("harness", "harness_module.sealed.py")):
        destination = arm / name
        shutil.copyfile(inputs.path(key), destination)
        if sha256(destination) != inputs.digest(key):
            fail(f"sealed copy mismatch: {key}")
    adapter = stage / "swebench_cpuset_adapter.sealed.py"
    shutil.copyfile(adapter_source, adapter)
    if sha256(adapter) != sha256(adapter_source):
        fail("sealed CPU-isolation adapter hash mismatch")
    predictions, diagnostics, counts = convert(rows, converter)
    if len(predictions) != 40 or [row["instance_id"] for row in predictions] != ids:
        fail("prediction denominator/order drifted")
    predictions_path = arm / "predictions.sealed.json"
    diagnostics_path = arm / "conversion_diagnostics.sealed.jsonl"
    ledger_path = arm / "nonrecovery_ledger.sealed.json"
    write_json(predictions_path, predictions)
    diagnostics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics))
    ledger = nonrecovery_ledger(diagnostics)
    validate_ledger(diagnostics, ledger)
    write_json(ledger_path, ledger)
    write_json(arm / "manifest.json", {
        "schema_version": "epyc.fg3-tc-nothink-v4-replay.v1", "status": "READY_FOR_OFFICIAL_SCORING", "arm": CAPTURE_ARM,
        "requested_ids": ids, "counts": counts, "no_inference": True, "length_caps_are_model_failures": True,
        "frozen_authority": {key: {"source": EXPECTED[key][0], "sha256": inputs.digest(key)} for key in ("frozen_table", "authority_finalization", "converter", "dataset", "harness", "authority_ids")},
        "capture": {key: {"source": EXPECTED[key][0], "sha256": inputs.digest(key)} for key in ("capture", "capture_summary", "capture_argv", "capture_done")},
        "sealer": {
            "source": str(sealer_source),
            "sealed_path": str(Path("..") / sealer.name),
            "sha256": sha256(sealer),
        },
        "cpu_isolation": {
            "environment_variable": CPUSET_ENV,
            "cpuset_cpus": CANONICAL_CPUSET,
            "atomic_create": True,
            "inspect_before_start": True,
            "original_pinned_harness": {
                "run_evaluation": {
                    "frozen_authority_source": EXPECTED["harness"][0],
                    "frozen_authority_sha256": inputs.digest("harness"),
                    "installed_source": str(harness_source),
                    "installed_sha256": sha256(harness_source),
                },
                "docker_build": {
                    "installed_source": str(docker_build_source),
                    "installed_sha256": sha256(docker_build_source),
                },
            },
            "adapter": {
                "source": str(adapter_source),
                "sealed_path": str(Path("..") / adapter.name),
                "sha256": sha256(adapter),
            },
        },
        "sealed": {path.name: sha256(path) for path in sorted(arm.iterdir()) if path.is_file()},
    })
    return arm


def revalidate_sealed_arm(stage: Path, arm: Path, ids: list[str]) -> None:
    """Reject a scorer that mutates any sealed input or conversion output."""
    manifest = json.loads((arm / "manifest.json").read_text())
    sealed = manifest.get("sealed")
    if not isinstance(sealed, dict):
        fail("sealed manifest is malformed")
    for name, digest in sealed.items():
        path = arm / name
        if not path.is_file() or not isinstance(digest, str) or sha256(path) != digest:
            fail(f"sealed artifact drifted during official scoring: {name}")
    source = manifest.get("sealer")
    if not isinstance(source, dict) or not isinstance(source.get("sealed_path"), str) or not isinstance(source.get("sha256"), str):
        fail("sealed manifest lacks a sealer source binding")
    sealer = (arm / source["sealed_path"]).resolve()
    if not sealer.is_file() or stage not in sealer.parents or sha256(sealer) != source["sha256"]:
        fail("sealed sealer source drifted during official scoring")
    diagnostics = load_jsonl(arm / "conversion_diagnostics.sealed.jsonl")
    if [row.get("instance_id") for row in diagnostics] != ids:
        fail("sealed diagnostics denominator/order drifted")
    validate_ledger(diagnostics, json.loads((arm / "nonrecovery_ledger.sealed.json").read_text()))
    isolation = manifest.get("cpu_isolation")
    if not isinstance(isolation, dict) or isolation.get("cpuset_cpus") != CANONICAL_CPUSET:
        fail("sealed manifest lacks the canonical CPU-isolation contract")
    adapter = isolation.get("adapter")
    if not isinstance(adapter, dict) or not isinstance(adapter.get("sealed_path"), str) or not isinstance(adapter.get("sha256"), str):
        fail("sealed manifest lacks CPU-isolation adapter provenance")
    adapter_path = (arm / adapter["sealed_path"]).resolve()
    if not adapter_path.is_file() or stage not in adapter_path.parents or sha256(adapter_path) != adapter["sha256"]:
        fail("sealed CPU-isolation adapter drifted during official scoring")


def evaluation_command(arm: Path, ids: list[str], run_id: str, adapter: Path) -> list[str]:
    return ["/usr/bin/taskset", "-c", CANONICAL_CPUSET, str(SWEBENCH_PYTHON), str(adapter),
            "--dataset_name", str(arm / "swebench_verified.sealed.json"), "--predictions_path", str(arm / "predictions.sealed.json"),
            "--instance_ids", *ids, "--max_workers", "8", "--timeout", "1800", "--cache_level", "env", "--run_id", run_id,
            "--report_dir", str(arm / "report")]


def run_official(arm: Path, ids: list[str], run_id: str, adapter: Path, adapter_sha256: str) -> None:
    if not adapter.is_file() or sha256(adapter) != adapter_sha256:
        fail("sealed CPU-isolation adapter is unavailable or drifted")
    installed_harness()
    installed_docker_build()
    environment = os.environ.copy()
    environment[CPUSET_ENV] = CANONICAL_CPUSET
    command = evaluation_command(arm, ids, run_id, adapter)
    result = subprocess.run(command, text=True, capture_output=True, cwd=arm, env=environment, check=False)
    (arm / "command.txt").write_text(f"{CPUSET_ENV}={CANONICAL_CPUSET} " + " ".join(command) + "\n")
    (arm / "stdout.log").write_text(result.stdout)
    (arm / "stderr.log").write_text(result.stderr)
    (arm / "exit_code").write_text(f"{result.returncode}\n")
    if result.returncode:
        fail(f"official SWE evaluation exited {result.returncode}")


def find_report(arm: Path) -> Path:
    reports = []
    for path in arm.rglob("*.json"):
        if path.name in {"manifest.json", "predictions.sealed.json"}:
            continue
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "resolved_ids" in value:
            reports.append(path)
    if len(reports) != 1:
        fail(f"{arm.name} has {len(reports)} official report candidates")
    return reports[0]


def validate_report(arm: Path, ids: list[str]) -> dict[str, Any]:
    report_path = find_report(arm)
    report = json.loads(report_path.read_text())
    names = ("submitted_ids", "completed_ids", "empty_patch_ids", "resolved_ids", "unresolved_ids", "error_ids")
    lists = {name: report.get(name, []) for name in names}
    if any(not isinstance(value, list) or not all(isinstance(item, str) for item in value) or len(value) != len(set(value)) for value in lists.values()):
        fail("official report contains duplicate or malformed ID lists")
    sets = {name: set(value) for name, value in lists.items()}
    count_fields = {"submitted_ids": "submitted_instances", "completed_ids": "completed_instances", "empty_patch_ids": "empty_patch_instances",
                    "resolved_ids": "resolved_instances", "unresolved_ids": "unresolved_instances", "error_ids": "error_instances"}
    if any(report.get(field) != len(sets[name]) for name, field in count_fields.items()) or sets["submitted_ids"] != set(ids) or report.get("error_instances") != 0:
        fail("official report has denominator drift or harness errors")
    predictions = json.loads((arm / "predictions.sealed.json").read_text())
    expected_empty = {row["instance_id"] for row in predictions if not row["model_patch"]}
    if (sets["completed_ids"] & sets["empty_patch_ids"] or sets["resolved_ids"] & sets["unresolved_ids"]
            or sets["completed_ids"] | sets["empty_patch_ids"] != set(ids)
            or sets["resolved_ids"] | sets["unresolved_ids"] != sets["completed_ids"]
            or sets["empty_patch_ids"] != expected_empty):
        fail("official report partition is invalid")
    result = {"arm": CAPTURE_ARM, "resolved": len(sets["resolved_ids"]), "denominator": len(ids),
              "percent_resolved": 100 * len(sets["resolved_ids"]) / len(ids), "empty_patch_failures": len(expected_empty),
              "harness_errors": 0, "report": str(report_path.relative_to(arm.parent)), "report_sha256": sha256(report_path),
              "quality_provenance": {"same_era_generation": True, "quality_transfer_to_v8_eligible": True,
                                     "transfer_basis": "complete FG-3 v8 no-think capture replayed through frozen v4 converter",
                                     "speed_or_throughput_transfer_claim": False}}
    write_json(arm / "report_validation.json", result)
    return result


def final_ledger(stage: Path) -> None:
    files = sorted(path for path in stage.rglob("*") if path.is_file() and path.name != "finalization.sha256")
    (stage / "finalization.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(stage)}\n" for path in files))


def execute(inputs: Inputs, output: Path) -> None:
    preflight(inputs)
    ids = authority_ids(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        fail("publication lock already exists; another publisher owns this target")
    try:
        if output.exists():
            fail("output already exists; immutable successor creation refuses overwrite")
        os.write(descriptor, f"pid={os.getpid()} target={output}\n".encode())
        stage = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
        stage.mkdir()
        try:
            write_json(stage / "state.json", {"status": "STAGING_NOT_FINAL", "target": str(output), "created_at": datetime.now(UTC).isoformat()})
            sealer_source = Path(__file__).resolve()
            adapter_source = ADAPTER_SOURCE.resolve()
            if not adapter_source.is_file():
                fail("CPU-isolation adapter source is unavailable")
            sealer = stage / "replay_tc_nothink_v4.sealed.py"
            shutil.copyfile(sealer_source, sealer)
            if sha256(sealer) != sha256(sealer_source):
                fail("sealed sealer source hash mismatch")
            rows = validate_capture(inputs, ids)
            converter = stage_converter(stage, inputs)
            harness_source = installed_harness()
            docker_build_source = installed_docker_build()
            arm = seal(
                stage,
                inputs,
                ids,
                rows,
                converter,
                sealer,
                sealer_source,
                adapter_source,
                harness_source,
                docker_build_source,
            )
            run_official(
                arm,
                ids,
                output.name,
                stage / "swebench_cpuset_adapter.sealed.py",
                sha256(adapter_source),
            )
            revalidate_sealed_arm(stage, arm, ids)
            verify_inputs(inputs)
            validate_capture(inputs, ids)
            installed_harness()
            validate_repo_mirror(inputs, ids)
            validation = validate_report(arm, ids)
            validation["sealer"] = {
                "source": str(sealer_source),
                "sealed_path": str(sealer.relative_to(stage)),
                "sha256": sha256(sealer),
            }
            write_json(stage / "result.json", validation)
            write_json(stage / "state.json", {"status": "FINALIZED", "target": str(output)})
            final_ledger(stage)
            if output.exists():
                fail("output appeared during staging; refusing overwrite")
            os.rename(stage, output)
        except Exception as exc:
            write_json(stage / "state.json", {"status": "FAILED_NOT_FINAL", "target": str(output), "error": str(exc)})
            raise
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
    print(json.dumps({"status": "FINALIZED", "run_root": str(output)}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-repo", type=Path, default=CANONICAL_REPO)
    args = parser.parse_args(argv)
    try:
        inputs = Inputs(args.source_repo.resolve())
        if args.preflight:
            preflight(inputs)
        else:
            if args.output_dir is None:
                fail("--execute requires --output-dir")
            execute(inputs, args.output_dir.resolve())
    except (OSError, ReplayError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
