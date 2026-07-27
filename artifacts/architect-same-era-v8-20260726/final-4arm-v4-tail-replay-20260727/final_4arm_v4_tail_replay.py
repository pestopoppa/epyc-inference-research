#!/usr/bin/env python3
"""Deterministically replay the final four-arm SWE table without inference.

This package implements the 2026-07-27 operator directive.  It applies the
pinned v4 SEARCH/REPLACE semantics to three July-24 banked response tails and
the terminal Laguna v4 capture, then runs the official harness sequentially.
The pre-v4 captures are admissible *only* for this final historical table; the
exception is written into each sealed arm manifest and cannot establish an E8
same-era quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = Path("/mnt/raid0/llm/epyc-inference-research")
ROOT = Path("/mnt/raid0/llm/epyc-root")
ORCHESTRATOR = Path("/mnt/raid0/llm/epyc-orchestrator")
CONVERTER = REPO / "artifacts/architect-code-eval-20260724/convert_sr_to_patch.py"
DATASET = REPO / "artifacts/architect-code-eval-20260724/swebench_verified.json"
QUESTIONS = REPO / "artifacts/architect-code-eval-20260724/questions_swebench_oracle.json"
SWEBENCH_PYTHON = REPO / ".venv-swebench/bin/python"
TASKSET = Path("/usr/bin/taskset")
HARNESS_CPUSET = "112-119"
HF_HOME = "/mnt/raid0/llm/cache/huggingface"
CONVERTER_SHA256 = "6bd2302dda3e5139cc6faabcc5639bdcf85b27895f93a9181cbb53dd65749507"
DATASET_SHA256 = "b087b5dad72b3e765a6cf93a9e7d516d8796698a0fd358abb73c6627df19f66e"
QUESTIONS_SHA256 = "f82a5191274048f2fdf432df7a0ebf4017ad982b954d6aa075326a1302df1c3c"
RUNNER_LAGUNA = (
    REPO
    / "artifacts/architect-laguna-iq2-v8-20260726/scorer-artifact-rescore-20260726"
    / "clean-full40-promptfix-20260726/run-20260726T220759Z/runner_source.py"
)
RUNNER_LAGUNA_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
STOPPING_RULE = {
    "final_table": "final_4arm_table.json",
    "finality": "FINAL_FOR_THIS_ERA_PER_OPERATOR_DIRECTIVE",
    "architect_tier_decision": (
        "If A3 leads or ties at top, the architect-tier quality question is closed; "
        "the role choice falls to throughput/residency economics."
    ),
    "confirmation_policy": (
        "The powered-160 confirmation is shelved unless this corrected table reorders "
        "an outcome that changes an actual deployment decision."
    ),
    "regeneration_policy": (
        "Regenerate only if generation evidence is defective; scorer/converter/extractor "
        "defects use deterministic replay of banked outputs."
    ),
}
KERNEL_PARITY_BASIS = "operator-certified v7-to-v8 kernel Delta 0.0pp exact quality parity"
ARMS = (
    {
        "name": "A1",
        "label": "A1_122b_iq2",
        "raw": REPO / "artifacts/architect-code-eval-20260724/swe_A1_122b_iq2/pq.jsonl",
        "raw_sha256": "f4dff35ee646993459f6536e4e8e6db80870a5691980bc158987f433e90c7c33",
        "legacy": True,
    },
    {
        "name": "A3",
        "label": "A3_27b_dense",
        "raw": REPO / "artifacts/architect-code-eval-20260724/swe_A3_27b_dense/pq.jsonl",
        "raw_sha256": "1fa07aff0ae46b1300104b87a3ea96daaaa8dda9c9a028dd6e22b1879effc3c2",
        "legacy": True,
    },
    {
        "name": "A4",
        "label": "A4_35b_a3b",
        "raw": REPO / "artifacts/architect-code-eval-20260724/swe_A4_35b_a3b/pq.jsonl",
        "raw_sha256": "548fe5b396e98f7b84a64746e1a3c8af12ecec559d92a63edae0b9c1852a629c",
        "legacy": True,
    },
    {
        "name": "Laguna",
        "label": "Laguna_S_2_1_UD_IQ2_M_v8",
        "raw": (
            REPO
            / "artifacts/architect-laguna-iq2-v8-20260726/scorer-artifact-rescore-20260726"
            / "clean-full40-promptfix-20260726/run-20260726T220759Z/pq.jsonl"
        ),
        "raw_sha256": "a2ce92399d87c8f5f15b285ad27eca3cf0328f1b80eec1fec81933ed782cd81a",
        "legacy": False,
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quality_provenance(arm: dict[str, Any]) -> dict[str, Any]:
    if arm["legacy"]:
        return {
            "same_era_generation": False,
            "quality_transfer_to_v8_eligible": True,
            "transfer_basis": KERNEL_PARITY_BASIS,
            "current_era_quality_decision_input": True,
            "speed_or_throughput_transfer_claim": False,
        }
    return {
        "same_era_generation": True,
        "quality_transfer_to_v8_eligible": True,
        "transfer_basis": "v8 same-era terminal Laguna v4 capture",
        "current_era_quality_decision_input": True,
        "speed_or_throughput_transfer_claim": False,
    }


def git_snapshot(path: Path) -> dict[str, str]:
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, text=True, capture_output=True)
    status = subprocess.run(["git", "-C", str(path), "status", "--short"], check=True, text=True, capture_output=True)
    return {"path": str(path), "head": head.stdout.strip(), "status": status.stdout}


def task_affinity(pid: int) -> list[dict[str, str]]:
    result = []
    for task in sorted((Path("/proc") / str(pid) / "task").glob("*"), key=lambda path: int(path.name)):
        status = task / "status"
        if not status.is_file():
            continue
        allowed = next((line.split("\t", 1)[1].strip() for line in status.read_text().splitlines()
                        if line.startswith("Cpus_allowed_list:")), "")
        result.append({"tid": task.name, "cpuset": allowed})
    return result


def fable_coexistence_evidence() -> dict[str, Any]:
    """Capture a live Fable workload as correctness-only coexistence evidence."""
    process_rows = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], check=True, text=True, capture_output=True).stdout.splitlines()
    parsed = []
    for line in process_rows:
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            parsed.append((int(fields[0]), int(fields[1]), fields[2]))
    found = []
    for pid, ppid, argv in parsed:
        if "continue_thinkingcap_and_fable.sh" not in argv:
            continue
        inference = []
        for child_pid, child_ppid, child_argv in parsed:
            if child_ppid != pid or not ("llama-server" in child_argv or "v7_quality_gate_runner.py" in child_argv):
                continue
            affinity = task_affinity(child_pid)
            inference.append({
                "pid": child_pid, "argv": child_argv, "thread_affinity": affinity,
                "expected_cpuset": "184-191",
                "cpuset_matches_expected": bool(affinity) and all(row["cpuset"] == "184-191" for row in affinity),
            })
        found.append({
            "controller_pid": pid, "controller_ppid": ppid, "controller_argv": argv,
            "allowed_inference_processes": inference,
            "scope": "correctness-only coexistence evidence; no timing or throughput claim",
        })
    return {"present": bool(found), "processes": found}


def check_no_official_swe_harness() -> None:
    rows = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True).stdout.splitlines()
    active = [line.strip() for line in rows if "swebench.harness.run_evaluation" in line and str(os.getpid()) not in line.split(None, 1)[0]]
    if active:
        fail("another official SWE harness is active: " + " | ".join(active))
    if shutil.which("docker"):
        containers = subprocess.run(["docker", "ps", "--format", "{{.ID}} {{.Image}} {{.Names}} {{.Command}}"], check=True, text=True, capture_output=True).stdout
        if any(token in containers.lower() for token in ("swebench", "swe-bench")):
            fail("an official SWE Docker container is active")


def pinned_ids() -> list[str]:
    if sha256(QUESTIONS) != QUESTIONS_SHA256:
        fail("pinned question source SHA-256 drifted")
    rows = json.loads(QUESTIONS.read_text())
    ids = [row.get("id") for row in rows]
    if len(ids) != 40 or len(set(ids)) != 40 or not all(isinstance(value, str) for value in ids):
        fail("question source is not the exact ordered 40-ID oracle")
    return ids


def validate_raw_rows(arm: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    raw = Path(arm["raw"])
    if not raw.is_file() or sha256(raw) != arm["raw_sha256"]:
        fail(f"{arm['name']} raw capture source drifted")
    rows = load_jsonl(raw)
    actual_ids = [row.get("id") for row in rows]
    if actual_ids != ids:
        fail(f"{arm['name']} does not preserve the pinned 40 IDs and order")
    if any(bool(row.get("request_error")) for row in rows):
        fail(f"{arm['name']} has request-error rows")
    if arm["legacy"]:
        if any(row.get("capture_schema_version") for row in rows):
            fail(f"{arm['name']} is not a July-24 legacy capture as declared")
    else:
        if any(row.get("capture_schema_version") != "v7_quality_gate_capture.v4" for row in rows):
            fail("Laguna is not a complete v4 capture")
        if not RUNNER_LAGUNA.is_file() or sha256(RUNNER_LAGUNA) != RUNNER_LAGUNA_SHA256:
            fail("Laguna reviewed v4 runner source drifted")
    return rows


def load_converter() -> Any:
    if not CONVERTER.is_file() or sha256(CONVERTER) != CONVERTER_SHA256:
        fail("pinned v4 converter SHA-256 drifted")
    spec = importlib.util.spec_from_file_location("final_4arm_pinned_converter", CONVERTER)
    if spec is None or spec.loader is None:
        fail("cannot load pinned v4 converter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_authorities() -> tuple[list[str], Any, Path]:
    ids = pinned_ids()
    if not DATASET.is_file() or sha256(DATASET) != DATASET_SHA256:
        fail("canonical SWE-bench_Verified snapshot SHA-256 drifted")
    dataset = json.loads(DATASET.read_text())
    if len(dataset) != 500 or len({row.get("instance_id") for row in dataset}) != 500:
        fail("canonical SWE-bench_Verified snapshot has unexpected shape")
    if not set(ids) <= {row["instance_id"] for row in dataset}:
        fail("pinned question IDs are not all in the canonical snapshot")
    converter = load_converter()
    if not SWEBENCH_PYTHON.is_file():
        fail("pinned SWE-bench Python is unavailable")
    # Discover the harness through its pinned interpreter and bind that source at
    # seal time.  The package never imports the harness through this interpreter.
    discovered = subprocess.run(
        [str(SWEBENCH_PYTHON), "-c", "import inspect,swebench.harness.run_evaluation as m; print(inspect.getsourcefile(m))"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    harness = Path(discovered)
    if not harness.is_file():
        fail("pinned SWE-bench harness module is unavailable")
    return ids, converter, harness


def convert_rows(converter: Any, arm: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    applied = skipped = empty = 0
    for row in rows:
        block_diagnostics: list[dict[str, Any]] = []
        if row.get("finish_reason") == "length":
            patch, did_apply, did_skip = "", 0, 0
        else:
            patch, did_apply, did_skip = converter.apply_blocks(
                converter.rows[row["id"]], row.get("response", ""), block_diagnostics,
            )
        diagnostic = converter.row_diagnostic(row, patch, block_diagnostics, (
            sha256(RUNNER_LAGUNA) if not arm["legacy"] else None
        ))
        # The converter's v4 CLI intentionally rejects schema-less evidence.  The
        # operator directive permits these three immutable July-24 tails only for
        # this historical final table; conversion/matching itself is unchanged.
        if arm["legacy"]:
            diagnostic["operator_directive_legacy_tail_exception"] = True
            diagnostic["operator_directive_scope"] = "final_4arm_historical_table_only"
        predictions.append({
            "instance_id": row["id"],
            "model_name_or_path": arm["label"],
            "model_patch": patch,
        })
        diagnostics.append(diagnostic)
        applied += did_apply
        skipped += did_skip
        empty += not bool(patch)
    if not arm["legacy"] and any(not diagnostic["scoring_eligible"] for diagnostic in diagnostics):
        fail("Laguna v4 capture-integrity validation failed")
    return predictions, diagnostics, {
        "blocks_applied": applied,
        "blocks_skipped": skipped,
        "empty_patch_count": empty,
        "length_rows_forced_empty": sum(row.get("finish_reason") == "length" for row in rows),
    }


def ledger_for(arm: dict[str, Any], diagnostics: list[dict[str, Any]], bindings: dict[str, Any]) -> dict[str, Any]:
    skipped_blocks = []
    empty_rows = []
    for diagnostic in diagnostics:
        for block in diagnostic.get("blocks", []):
            if str(block.get("outcome", "")).startswith("skipped_"):
                skipped_blocks.append({
                    "instance_id": diagnostic["instance_id"],
                    "block_index": block["block_index"],
                    "outcome": block["outcome"],
                    "search_sha256": block["search_sha256"],
                    "replace_sha256": block["replace_sha256"],
                    "additional_recovery_attempted": False,
                    "disposition": "preserve_pinned_v4_outcome_no_match_broadening",
                })
        if diagnostic["empty_patch"]:
            empty_rows.append({
                "instance_id": diagnostic["instance_id"],
                "finish_reason": diagnostic.get("finish_reason"),
                "empty_patch_reason": diagnostic["empty_patch_reason"],
                "conversion_disposition": diagnostic["conversion_disposition"],
                "additional_recovery_attempted": False,
                "disposition": "preserve_pinned_v4_outcome_no_match_broadening",
            })
    return {
        "schema_version": "epyc.final-4arm-v4-tail-replay-nonrecovery-ledger.v1",
        "status": "EXHAUSTIVE_PINNED_V4_OUTCOME",
        "arm": arm["name"],
        "operator_directive": "2026-07-27 bench closure: deterministic-tail rescoring; no matching broadening",
        "legacy_capture_exception": bool(arm["legacy"]),
        "quality_provenance": quality_provenance(arm),
        "inputs": bindings,
        "skipped_blocks": skipped_blocks,
        "empty_patch_rows": empty_rows,
        "aggregate": {
            "diagnostic_skipped_block_count": len(skipped_blocks),
            "ledger_skipped_block_count": len(skipped_blocks),
            "empty_patch_row_count": len(empty_rows),
        },
    }


def validate_conversion(arm: dict[str, Any], ids: list[str], predictions: list[dict[str, str]], diagnostics: list[dict[str, Any]], ledger: dict[str, Any]) -> None:
    if [row["instance_id"] for row in predictions] != ids or len(predictions) != 40:
        fail(f"{arm['name']} prediction denominator/order drifted")
    if any(set(row) != {"instance_id", "model_name_or_path", "model_patch"} for row in predictions):
        fail(f"{arm['name']} prediction schema drifted")
    if len(diagnostics) != 40 or [row["instance_id"] for row in diagnostics] != ids:
        fail(f"{arm['name']} diagnostics do not bind all 40 rows")
    actual_skips = [
        (diagnostic["instance_id"], block["block_index"], block["outcome"], block["search_sha256"], block["replace_sha256"])
        for diagnostic in diagnostics for block in diagnostic.get("blocks", [])
        if str(block.get("outcome", "")).startswith("skipped_")
    ]
    ledger_skips = [
        (row["instance_id"], row["block_index"], row["outcome"], row["search_sha256"], row["replace_sha256"])
        for row in ledger["skipped_blocks"]
    ]
    if sorted(actual_skips) != sorted(ledger_skips):
        fail(f"{arm['name']} non-recovery ledger is not exhaustive")
    if any(row.get("additional_recovery_attempted") is not False
           for row in ledger["skipped_blocks"] + ledger["empty_patch_rows"]):
        fail(f"{arm['name']} ledger makes an unsupported recovery assertion")


def seal_arm(run_root: Path, arm: dict[str, Any], ids: list[str], converter: Any, harness: Path) -> Path:
    rows = validate_raw_rows(arm, ids)
    predictions, diagnostics, counts = convert_rows(converter, arm, rows)
    arm_dir = run_root / arm["name"]
    arm_dir.mkdir(parents=True)
    raw_copy = arm_dir / "raw_capture.sealed.jsonl"
    converter_copy = arm_dir / "converter_v4.sealed.py"
    dataset_copy = arm_dir / "swebench_verified.sealed.json"
    harness_copy = arm_dir / "harness_module.sealed.py"
    predictions_path = arm_dir / "predictions.sealed.json"
    diagnostics_path = arm_dir / "conversion_diagnostics.sealed.jsonl"
    ledger_path = arm_dir / "nonrecovery_ledger.sealed.json"
    shutil.copyfile(arm["raw"], raw_copy)
    shutil.copyfile(CONVERTER, converter_copy)
    shutil.copyfile(DATASET, dataset_copy)
    shutil.copyfile(harness, harness_copy)
    write_json(predictions_path, predictions)
    diagnostics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics))
    bindings = {
        "raw_capture": {"source": str(arm["raw"]), "sha256": sha256(raw_copy)},
        "converter_source": {"source": str(CONVERTER), "sha256": sha256(converter_copy)},
        "dataset": {"source": str(DATASET), "sha256": sha256(dataset_copy)},
        "harness_module": {"source": str(harness), "sha256": sha256(harness_copy)},
        "predictions": {"sha256": sha256(predictions_path)},
        "diagnostics": {"sha256": sha256(diagnostics_path)},
    }
    ledger = ledger_for(arm, diagnostics, bindings)
    write_json(ledger_path, ledger)
    bindings["nonrecovery_ledger"] = {"sha256": sha256(ledger_path)}
    validate_conversion(arm, ids, predictions, diagnostics, ledger)
    manifest = {
        "schema_version": "epyc.final-4arm-v4-tail-replay-arm.v1",
        "status": "READY_FOR_OFFICIAL_SCORING",
        "arm": arm["name"],
        "requested_ids": ids,
        "counts": counts,
        "operator_directive_provenance": {
            "directive_date": "2026-07-27",
            "classification": (
                "legacy_banked_tail_accepted_only_for_final_historical_era_table"
                if arm["legacy"] else "terminal_v4_capture_rebuilt_with_pinned_converter"
            ),
            "quality_provenance": quality_provenance(arm),
            "no_generation_or_inference": True,
            "length_finish_reason_forced_empty": True,
            "matching_semantics": "pinned_v4_exact_then_trailing_whitespace_then_unique_indent_only",
            "stopping_rule": STOPPING_RULE,
        },
        "sealed": bindings,
    }
    write_json(arm_dir / "manifest.json", manifest)
    return arm_dir


def find_report(arm_dir: Path) -> Path:
    reports = []
    for path in arm_dir.rglob("*.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "error_instances" in value and "resolved_ids" in value:
            reports.append(path)
    if len(reports) != 1:
        fail(f"{arm_dir.name} has {len(reports)} official report candidates")
    return reports[0]


def validate_report(arm_dir: Path, ids: list[str]) -> dict[str, Any]:
    manifest = json.loads((arm_dir / "manifest.json").read_text())
    predictions = json.loads((arm_dir / "predictions.sealed.json").read_text())
    report_path = find_report(arm_dir)
    report = json.loads(report_path.read_text())
    expected = set(ids)
    listed = {field: set(report.get(field, [])) for field in (
        "submitted_ids", "completed_ids", "empty_patch_ids", "resolved_ids", "unresolved_ids", "error_ids",
    )}
    if any(len(report.get(field, [])) != len(listed[field]) for field in listed):
        fail(f"{arm_dir.name} report has duplicate IDs")
    count_fields = {
        "submitted_ids": "submitted_instances",
        "completed_ids": "completed_instances",
        "empty_patch_ids": "empty_patch_instances",
        "resolved_ids": "resolved_instances",
        "unresolved_ids": "unresolved_instances",
        "error_ids": "error_instances",
    }
    if any(report.get(count_field) != len(listed[field]) for field, count_field in count_fields.items()):
        fail(f"{arm_dir.name} report count does not match its ID list")
    expected_empty = {row["instance_id"] for row in predictions if not row["model_patch"]}
    if (
        report.get("submitted_instances") != 40 or listed["submitted_ids"] != expected
        or report.get("error_instances") != 0 or listed["error_ids"]
        or listed["empty_patch_ids"] != expected_empty
        or listed["completed_ids"] | listed["empty_patch_ids"] != expected
        or listed["completed_ids"] & listed["empty_patch_ids"]
        or listed["resolved_ids"] | listed["unresolved_ids"] != listed["completed_ids"]
        or listed["resolved_ids"] & listed["unresolved_ids"]
    ):
        fail(f"{arm_dir.name} report has denominator drift or harness errors")
    result = {
        "arm": manifest["arm"], "report": str(report_path), "report_sha256": sha256(report_path),
        "resolved": len(listed["resolved_ids"]), "denominator": 40,
        "percent_resolved": 100.0 * len(listed["resolved_ids"]) / 40,
        "empty_patch_failures": len(expected_empty), "harness_errors": 0,
        "quality_provenance": manifest["operator_directive_provenance"]["quality_provenance"],
    }
    write_json(arm_dir / "report_validation.json", result)
    return result


def run_arm(arm_dir: Path, arm: dict[str, Any], ids: list[str], run_id: str) -> None:
    check_no_official_swe_harness()
    command = [
        str(TASKSET), "-c", HARNESS_CPUSET, str(SWEBENCH_PYTHON), "-m", "swebench.harness.run_evaluation",
        "--dataset_name", str(arm_dir / "swebench_verified.sealed.json"),
        "--predictions_path", str(arm_dir / "predictions.sealed.json"),
        "--instance_ids", *ids, "--max_workers", "8", "--timeout", "1800", "--cache_level", "env",
        "--run_id", f"{run_id}-{arm['name'].lower()}", "--report_dir", str(arm_dir / "report"),
    ]
    (arm_dir / "command.txt").write_text(" ".join(map(shlex_quote, command)) + "\n")
    env = dict(os.environ)
    env["HF_HOME"] = HF_HOME
    result = subprocess.run(command, text=True, capture_output=True, cwd=arm_dir, env=env)
    (arm_dir / "stdout.log").write_text(result.stdout)
    (arm_dir / "stderr.log").write_text(result.stderr)
    (arm_dir / "exit_code").write_text(f"{result.returncode}\n")
    if result.returncode:
        fail(f"{arm['name']} official SWE harness exited {result.returncode}")


def shlex_quote(value: str) -> str:
    # ``shlex.join`` is unavailable in a few older system helpers used to inspect evidence.
    import shlex
    return shlex.quote(value)


def finalize(run_root: Path, ids: list[str]) -> None:
    results = [validate_report(run_root / arm["name"], ids) for arm in ARMS]
    results.sort(key=lambda row: (-row["resolved"], row["arm"]))
    table = {
        "schema_version": "epyc.final-4arm-v4-tail-replay-table.v1",
        "status": "FINAL_FOR_THIS_ERA_PER_OPERATOR_DIRECTIVE",
        "scope": "FINAL current-era quality decision input: A1/A3/A4 transfer through certified Delta 0.0pp parity; Laguna is v8 same-era generation. No speed or throughput transfer claim.",
        "ranking": results,
        "stopping_rule": STOPPING_RULE,
    }
    write_json(run_root / "final_4arm_table.json", table)
    digest_paths = sorted(path for path in run_root.rglob("*") if path.is_file() and path.name not in {"finalization.sha256"})
    (run_root / "finalization.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(run_root)}\n" for path in digest_paths))


def preflight() -> None:
    ids, _converter, _harness = validate_authorities()
    for arm in ARMS:
        validate_raw_rows(arm, ids)
    if not TASKSET.is_file():
        fail("taskset is unavailable")
    check_no_official_swe_harness()
    print(json.dumps({"status": "PRECHECK_OK", "arms": [arm["name"] for arm in ARMS], "docker_invoked": False,
                      "fable_coexistence": fable_coexistence_evidence()}, sort_keys=True))


def execute() -> None:
    ids, converter, harness = validate_authorities()
    run_id = "final-4arm-v4-tail-replay-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = HERE / "runs" / run_id
    if run_root.exists():
        fail("run ID collision")
    run_root.mkdir(parents=True)
    check_no_official_swe_harness()
    provenance = {
        "schema_version": "epyc.final-4arm-v4-tail-replay.v1", "run_id": run_id,
        "operator_directive": "2026-07-27 bench closure", "no_inference": True,
        "replay_source_sha256": sha256(Path(__file__)), "converter_sha256": CONVERTER_SHA256,
        "dataset_sha256": DATASET_SHA256, "questions_sha256": QUESTIONS_SHA256,
        "stopping_rule": STOPPING_RULE,
        "quality_provenance": {arm["name"]: quality_provenance(arm) for arm in ARMS},
        "git": {"research": git_snapshot(REPO), "root": git_snapshot(ROOT), "orchestrator": git_snapshot(ORCHESTRATOR)},
        "fable_coexistence_pre": fable_coexistence_evidence(),
    }
    write_json(run_root / "package_provenance.json", provenance)
    for arm in ARMS:
        arm_dir = seal_arm(run_root, arm, ids, converter, harness)
        run_arm(arm_dir, arm, ids, run_id)
    provenance["fable_coexistence_post"] = fable_coexistence_evidence()
    write_json(run_root / "package_provenance.json", provenance)
    finalize(run_root, ids)
    print(json.dumps({"status": "FINALIZED", "run_root": str(run_root)}, sort_keys=True))


def finalize_existing(run_id: str) -> None:
    if not run_id.startswith("final-4arm-v4-tail-replay-"):
        fail("invalid final replay run ID")
    run_root = HERE / "runs" / run_id
    if not run_root.is_dir():
        fail("existing run is absent")
    ids, _converter, _harness = validate_authorities()
    for arm in ARMS:
        arm_dir = run_root / arm["name"]
        if not arm_dir.is_dir() or (arm_dir / "exit_code").read_text().strip() != "0":
            fail(f"{arm['name']} has no successful official harness evidence")
    finalize(run_root, ids)
    print(json.dumps({"status": "FINALIZED_EXISTING", "run_root": str(run_root)}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--finalize-existing", metavar="RUN_ID")
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            preflight()
        elif args.execute:
            execute()
        else:
            finalize_existing(args.finalize_existing)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
