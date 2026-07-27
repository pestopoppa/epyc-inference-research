#!/usr/bin/env python3
"""Append two banked 27B SWE arms to the immutable final four-arm v4 table.

This is intentionally a one-purpose deterministic replay package.  It neither
generates completions nor rewrites the historical four-arm authority.  The
ThinkingCap and Fable non-MTP captures are converted with the authority's
sealed v4 converter, evaluated by its sealed official harness, and emitted as
an append-only six-row successor table.  Fable MTP remains outside this probe.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AUTH_ROOT = REPO / "artifacts/architect-same-era-v8-20260726/final-4arm-v4-tail-replay-20260727"
AUTH_RUN = AUTH_ROOT / "runs/final-4arm-v4-tail-replay-20260727T080703Z"
AUTH_SCRIPT = AUTH_ROOT / "final_4arm_v4_tail_replay.py"
FROZEN_TABLE = AUTH_RUN / "final_4arm_table.json"
FROZEN_FINALIZATION = AUTH_RUN / "finalization.sha256"
FROZEN_TABLE_SHA256 = "f440680d7a6ae2169b35202627de7889441c530b4976f23fd4140e84be8f2e50"
FROZEN_FINALIZATION_SHA256 = "21a5de5c36018487bf7c6bf00e5e46a6cc2c5d2773a12cbba8279d34094aea23"
CONVERTER = AUTH_RUN / "A3/converter_v4.sealed.py"
HARNESS = AUTH_RUN / "A3/harness_module.sealed.py"
DATASET = AUTH_RUN / "A3/swebench_verified.sealed.json"
CONVERTER_SHA256 = "6bd2302dda3e5139cc6faabcc5639bdcf85b27895f93a9181cbb53dd65749507"
HARNESS_SHA256 = "6959f0b4e4eaf979771f529b88e3e9df1daa7fe86bc4291feec2e7d320bf7f2e"
DATASET_SHA256 = "b087b5dad72b3e765a6cf93a9e7d516d8796698a0fd358abb73c6627df19f66e"
CANONICAL_REPOS = REPO / "artifacts/architect-code-eval-20260724/swebench_repos"
SWEBENCH_PYTHON = REPO / ".venv-swebench/bin/python"
CAPTURE_ROOT = HERE / "live-20260726T1750Z/continuation-27b-v8"
RUNNER_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
CAPTURE_SCHEMA = "v7_quality_gate_capture.v4"
ARMS = (
    {"name": "A3-tc", "label": "A3-tc_ThinkingCap_Q8", "capture_arm": "A3-tc-quality__thinkingcap",
     "raw": CAPTURE_ROOT / "A3-tc-quality__thinkingcap/swe_oracle.sealed.jsonl",
     "raw_sha256": "95328eafa1dcca4a7d262d147a84f38a9656b725b17d66527f6867c8674394a6",
     "license_gate": "PENDING_NO_DECLARED_LICENSE"},
    {"name": "A3-ff", "label": "A3-ff_Fable_Fusion_Q8_non_MTP", "capture_arm": "A3-ff-quality__fable_non_mtp",
     "raw": CAPTURE_ROOT / "A3-ff-quality__fable_non_mtp/swe_oracle.sealed.jsonl",
     "raw_sha256": "a9f3bf2c65869ef0819416189b2478cfed157dd88bcb4cd62d68d913b034990a",
     "license_gate": "PENDING_FABLE_ABLITERATION_SCREEN"},
)


def fail(message: str) -> None:
    raise RuntimeError(message)


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


def validate_repo_mirror(dataset: Path, ids: list[str]) -> None:
    rows = {row.get("instance_id"): row for row in json.loads(dataset.read_text())}
    if not set(ids) <= set(rows):
        fail("verified dataset does not contain the authority denominator")
    for instance_id in ids:
        row = rows[instance_id]
        repo, commit = row.get("repo"), row.get("base_commit")
        if not isinstance(repo, str) or not isinstance(commit, str):
            fail(f"dataset lacks repo/base_commit identity for {instance_id}")
        git_dir = CANONICAL_REPOS / repo.replace("/", "__")
        result = subprocess.run(["git", "--git-dir", str(git_dir), "rev-parse", f"{commit}^{{commit}}"],
                                text=True, capture_output=True)
        if result.returncode or result.stdout.strip() != commit:
            fail(f"repo mirror lacks the required base commit for {instance_id}")


def validate_authority_sources(ids: list[str] | None = None) -> list[str]:
    expected = {
        FROZEN_TABLE: FROZEN_TABLE_SHA256, FROZEN_FINALIZATION: FROZEN_FINALIZATION_SHA256,
        CONVERTER: CONVERTER_SHA256, HARNESS: HARNESS_SHA256, DATASET: DATASET_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            fail(f"frozen authority drifted: {path}")
    table = json.loads(FROZEN_TABLE.read_text())
    if table.get("status") != "FINAL_FOR_THIS_ERA_PER_OPERATOR_DIRECTIVE" or len(table.get("ranking", [])) != 4:
        fail("frozen four-arm table is not the terminal authority")
    resolved_ids = ids or ids_from_authority()
    validate_repo_mirror(DATASET, resolved_ids)
    return resolved_ids


def stage_authority(stage_root: Path, ids: list[str]) -> Any:
    """Copy immutable converter/dataset and bind the verified mirror by symlink."""
    validate_authority_sources(ids)
    authority_dir = stage_root / "authority"
    authority_dir.mkdir(parents=True)
    staged_converter = authority_dir / "convert_sr_to_patch.py"
    staged_dataset = authority_dir / "swebench_verified.json"
    shutil.copyfile(CONVERTER, staged_converter)
    shutil.copyfile(DATASET, staged_dataset)
    if sha256(staged_converter) != CONVERTER_SHA256 or sha256(staged_dataset) != DATASET_SHA256:
        fail("staged authority copy hash mismatch")
    os.symlink(CANONICAL_REPOS, authority_dir / "swebench_repos")
    module = types.ModuleType("frozen_four_arm_converter")
    module.__file__ = str(staged_converter)
    exec(compile(staged_converter.read_text(), str(staged_converter), "exec"), module.__dict__)
    return module


def installed_harness_path() -> Path:
    if not SWEBENCH_PYTHON.is_file():
        fail("pinned SWE-bench interpreter is unavailable")
    result = subprocess.run([str(SWEBENCH_PYTHON), "-c", "import inspect,swebench.harness.run_evaluation as m; print(inspect.getsourcefile(m))"],
                            text=True, capture_output=True)
    path = Path(result.stdout.strip())
    if result.returncode or not path.is_file() or sha256(path) != HARNESS_SHA256:
        fail("installed SWE-bench harness SHA-256 drifted")
    return path


def ids_from_authority() -> list[str]:
    copied_questions = AUTH_RUN / "A3/raw_capture.sealed.jsonl"
    rows = load_jsonl(copied_questions)
    ids = [row.get("id") for row in rows]
    if len(ids) != 40 or len(set(ids)) != 40 or not all(isinstance(value, str) for value in ids):
        fail("frozen authority does not bind an ordered 40-ID denominator")
    return ids


def validate_capture(arm: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    raw = Path(arm["raw"])
    if not raw.is_file() or sha256(raw) != arm["raw_sha256"]:
        fail(f"{arm['name']} banked capture drifted")
    rows = load_jsonl(raw)
    if [row.get("id") for row in rows] != ids:
        fail(f"{arm['name']} does not preserve the authority 40-ID order")
    for row in rows:
        if (row.get("capture_schema_version") != CAPTURE_SCHEMA or row.get("arm") != arm["capture_arm"]
                or row.get("suite") != "swebench_oracle" or row.get("seed") != 42 or row.get("rep") != 0
                or row.get("runner_source_sha256") != RUNNER_SHA256):
            fail(f"{arm['name']} capture identity drifted at {row.get('id')}")
        if row.get("request_error") or row.get("finish_reason") == "request_error":
            fail(f"{arm['name']} has request-error evidence at {row.get('id')}")
        for field in ("prompt", "response", "reasoning"):
            if not isinstance(row.get(field), str) or row.get(f"{field}_fingerprint") != text_fingerprint(row[field]):
                fail(f"{arm['name']} lacks a complete fingerprinted {field} at {row.get('id')}")
    return rows


def convert(arm: dict[str, Any], rows: list[dict[str, Any]], converter: Any) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, int]]:
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
            fail(f"{arm['name']} v4 integrity failed at {row['id']}")
        if row.get("finish_reason") == "length":
            diagnostic.update({"empty_patch": True, "empty_patch_reason": "model_length_cap",
                               "conversion_disposition": "model_failure_length_cap"})
        predictions.append({"instance_id": row["id"], "model_name_or_path": arm["label"], "model_patch": patch})
        diagnostics.append(diagnostic)
        counts["blocks_applied"] += applied
        counts["blocks_skipped"] += skipped
        counts["empty_patch_count"] += not bool(patch)
    return predictions, diagnostics, counts


def quality_provenance() -> dict[str, Any]:
    return {"same_era_generation": True, "quality_transfer_to_v8_eligible": True,
            "transfer_basis": "complete v8 banked capture replayed through frozen final-table v4 converter",
            "current_era_quality_decision_input": True, "speed_or_throughput_transfer_claim": False}


def nonrecovery_ledger(arm: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    skipped, empty = [], []
    for diagnostic in diagnostics:
        for block in diagnostic.get("blocks", []):
            if str(block.get("outcome", "")).startswith("skipped_"):
                skipped.append({"instance_id": diagnostic["instance_id"], "block_index": block["block_index"],
                                "outcome": block["outcome"], "search_sha256": block["search_sha256"],
                                "replace_sha256": block["replace_sha256"], "classification": "model_contract_failure",
                                "additional_recovery_attempted": False,
                                "disposition": "preserve_frozen_v4_outcome_no_match_broadening"})
        if diagnostic.get("empty_patch"):
            empty.append({"instance_id": diagnostic["instance_id"], "finish_reason": diagnostic.get("finish_reason"),
                          "empty_patch_reason": diagnostic.get("empty_patch_reason"),
                          "classification": "model_failure_length_cap" if diagnostic.get("finish_reason") == "length" else "model_contract_failure",
                          "additional_recovery_attempted": False,
                          "disposition": "preserve_frozen_v4_outcome_no_match_broadening"})
    return {"schema_version": "epyc.append-27b-v4-nonrecovery-ledger.v1", "status": "EXHAUSTIVE_PINNED_V4_OUTCOME",
            "arm": arm["name"], "skipped_blocks": skipped, "empty_patch_rows": empty,
            "aggregate": {"skipped_block_count": len(skipped), "empty_patch_row_count": len(empty)}}


def validate_ledger(diagnostics: list[dict[str, Any]], ledger: dict[str, Any]) -> None:
    skips = {(row["instance_id"], row["block_index"], row["outcome"], row["search_sha256"], row["replace_sha256"])
             for row in ledger["skipped_blocks"]}
    actual = {(diag["instance_id"], block["block_index"], block["outcome"], block["search_sha256"], block["replace_sha256"])
              for diag in diagnostics for block in diag.get("blocks", []) if str(block.get("outcome", "")).startswith("skipped_")}
    if skips != actual or {row["instance_id"] for row in ledger["empty_patch_rows"]} != {diag["instance_id"] for diag in diagnostics if diag.get("empty_patch")}:
        fail("non-recovery ledger is not exhaustive")
    if any(row.get("additional_recovery_attempted") is not False for row in ledger["skipped_blocks"] + ledger["empty_patch_rows"]):
        fail("non-recovery ledger contains unsupported recovery")


def seal_arm(run_root: Path, arm: dict[str, Any], ids: list[str], converter: Any) -> Path:
    rows = validate_capture(arm, ids)
    predictions, diagnostics, counts = convert(arm, rows, converter)
    if [row["instance_id"] for row in predictions] != ids or len(predictions) != 40:
        fail(f"{arm['name']} prediction denominator/order drifted")
    arm_dir = run_root / arm["name"]
    arm_dir.mkdir(parents=True)
    raw_copy, converter_copy, dataset_copy, harness_copy = (arm_dir / name for name in (
        "raw_capture.sealed.jsonl", "converter_v4.sealed.py", "swebench_verified.sealed.json", "harness_module.sealed.py"))
    for source, destination, digest in ((arm["raw"], raw_copy, arm["raw_sha256"]), (CONVERTER, converter_copy, CONVERTER_SHA256),
                                        (DATASET, dataset_copy, DATASET_SHA256), (HARNESS, harness_copy, HARNESS_SHA256)):
        shutil.copyfile(source, destination)
        if sha256(destination) != digest:
            fail(f"sealed copy hash mismatch: {source}")
    predictions_path, diagnostics_path, ledger_path = arm_dir / "predictions.sealed.json", arm_dir / "conversion_diagnostics.sealed.jsonl", arm_dir / "nonrecovery_ledger.sealed.json"
    write_json(predictions_path, predictions)
    diagnostics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics))
    ledger = nonrecovery_ledger(arm, diagnostics)
    validate_ledger(diagnostics, ledger)
    write_json(ledger_path, ledger)
    write_json(arm_dir / "manifest.json", {
        "schema_version": "epyc.append-27b-v4-authority-arm.v1", "status": "READY_FOR_OFFICIAL_SCORING",
        "arm": arm["name"], "requested_ids": ids, "counts": counts,
        "no_inference": True, "length_caps_are_model_failures": True,
        "frozen_four_arm_authority": {"table": str(FROZEN_TABLE), "sha256": FROZEN_TABLE_SHA256},
        "sealed": {"raw_capture_sha256": sha256(raw_copy), "converter_sha256": sha256(converter_copy),
                   "dataset_sha256": sha256(dataset_copy), "harness_sha256": sha256(harness_copy),
                   "predictions_sha256": sha256(predictions_path), "diagnostics_sha256": sha256(diagnostics_path),
                   "nonrecovery_ledger_sha256": sha256(ledger_path)},
    })
    return arm_dir


def run_arm(arm_dir: Path, arm: dict[str, Any], ids: list[str], run_id: str) -> None:
    installed_harness_path()
    command = ["/usr/bin/taskset", "-c", "112-119", str(SWEBENCH_PYTHON),
               "-m", "swebench.harness.run_evaluation", "--dataset_name", str(arm_dir / "swebench_verified.sealed.json"),
               "--predictions_path", str(arm_dir / "predictions.sealed.json"), "--instance_ids", *ids,
               "--max_workers", "8", "--timeout", "1800", "--cache_level", "env", "--run_id", f"{run_id}-{arm['name'].lower()}",
               "--report_dir", str(arm_dir / "report")]
    result = subprocess.run(command, text=True, capture_output=True, cwd=arm_dir)
    (arm_dir / "command.txt").write_text(" ".join(command) + "\n")
    (arm_dir / "stdout.log").write_text(result.stdout); (arm_dir / "stderr.log").write_text(result.stderr)
    (arm_dir / "exit_code").write_text(f"{result.returncode}\n")
    if result.returncode:
        fail(f"{arm['name']} official SWE evaluation exited {result.returncode}")


def run_arms_concurrently(sealed: list[tuple[Path, dict[str, Any]]], ids: list[str], run_id: str) -> None:
    """Run the two independent sealed replays concurrently; each keeps its v4 8-worker contract."""
    installed_harness_path()
    processes: list[tuple[Path, dict[str, Any], list[str], subprocess.Popen[str]]] = []
    for index, (arm_dir, arm) in enumerate(sealed):
        cpuset = "112-119" if index == 0 else "120-127"
        command = ["/usr/bin/taskset", "-c", cpuset, str(SWEBENCH_PYTHON),
                   "-m", "swebench.harness.run_evaluation", "--dataset_name", str(arm_dir / "swebench_verified.sealed.json"),
                   "--predictions_path", str(arm_dir / "predictions.sealed.json"), "--instance_ids", *ids,
                   "--max_workers", "8", "--timeout", "1800", "--cache_level", "env", "--run_id", f"{run_id}-{arm['name'].lower()}",
                   "--report_dir", str(arm_dir / "report")]
        (arm_dir / "command.txt").write_text(" ".join(command) + "\n")
        processes.append((arm_dir, arm, command, subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=arm_dir)))
    failures = []
    for arm_dir, arm, _command, process in processes:
        stdout, stderr = process.communicate()
        (arm_dir / "stdout.log").write_text(stdout); (arm_dir / "stderr.log").write_text(stderr)
        (arm_dir / "exit_code").write_text(f"{process.returncode}\n")
        if process.returncode:
            failures.append(f"{arm['name']}={process.returncode}")
    if failures:
        fail("official SWE evaluation failed: " + ", ".join(failures))


def find_report(arm_dir: Path) -> Path:
    reports = [path for path in arm_dir.rglob("*.json") if path.name not in {"manifest.json", "predictions.sealed.json"}
               and isinstance(json.loads(path.read_text()), dict) and "resolved_ids" in json.loads(path.read_text())]
    if len(reports) != 1:
        fail(f"{arm_dir.name} has {len(reports)} official report candidates")
    return reports[0]


def validate_report(arm_dir: Path, ids: list[str]) -> dict[str, Any]:
    report_path = find_report(arm_dir); report = json.loads(report_path.read_text())
    names = ("submitted_ids", "completed_ids", "empty_patch_ids", "resolved_ids", "unresolved_ids", "error_ids")
    raw_lists = {name: report.get(name, []) for name in names}
    if any(not isinstance(values, list) or not all(isinstance(value, str) for value in values)
           or len(values) != len(set(values)) for values in raw_lists.values()):
        fail(f"{arm_dir.name} official report contains duplicate or malformed ID lists")
    listed = {name: set(raw_lists[name]) for name in names}
    counts = {"submitted_ids": "submitted_instances", "completed_ids": "completed_instances", "empty_patch_ids": "empty_patch_instances",
              "resolved_ids": "resolved_instances", "unresolved_ids": "unresolved_instances", "error_ids": "error_instances"}
    if any(report.get(total) != len(listed[name]) for name, total in counts.items()) or listed["submitted_ids"] != set(ids) or report.get("error_instances") != 0:
        fail(f"{arm_dir.name} official report has denominator drift or harness errors")
    predictions = json.loads((arm_dir / "predictions.sealed.json").read_text())
    expected_empty = {row["instance_id"] for row in predictions if not row["model_patch"]}
    if (listed["completed_ids"] & listed["empty_patch_ids"]
            or listed["resolved_ids"] & listed["unresolved_ids"]
            or listed["completed_ids"] | listed["empty_patch_ids"] != set(ids)
            or listed["resolved_ids"] | listed["unresolved_ids"] != listed["completed_ids"]
            or listed["empty_patch_ids"] != expected_empty):
        fail(f"{arm_dir.name} official report partition is invalid")
    result = {"arm": arm_dir.name, "resolved": len(listed["resolved_ids"]), "denominator": 40,
              "percent_resolved": 100 * len(listed["resolved_ids"]) / 40, "empty_patch_failures": len(expected_empty),
              "harness_errors": 0, "quality_provenance": quality_provenance(),
              "report": str(report_path.relative_to(arm_dir.parent)), "report_sha256": sha256(report_path)}
    write_json(arm_dir / "report_validation.json", result)
    return result


def finalize(run_root: Path, ids: list[str]) -> None:
    frozen = json.loads(FROZEN_TABLE.read_text())
    rows = list(frozen["ranking"]) + [validate_report(run_root / arm["name"], ids) for arm in ARMS]
    table = {"schema_version": "epyc.expanded-six-arm-v4-authority-table.v1", "status": "APPEND_ONLY_SUCCESSOR_NO_ROLE_DECISION",
             "frozen_four_arm_authority": {"path": str(FROZEN_TABLE), "sha256": FROZEN_TABLE_SHA256, "status": frozen["status"]},
             "rows": rows, "scope": "Six-row deterministic comparison. Frozen ranking is preserved; appended rows are not a deployment or role decision.",
             "license_gate_status": {"A3-tc": "PENDING_NO_DECLARED_LICENSE", "A3-ff": "PENDING_FABLE_ABLITERATION_SCREEN"},
             "mtp_disposition": "A3-ff embedded-MTP remains a separate probe and is absent from this table."}
    write_json(run_root / "expanded_six_arm_table.json", table)
    files = sorted(path for path in run_root.rglob("*") if path.is_file() and path.name != "finalization.sha256")
    (run_root / "finalization.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(run_root)}\n" for path in files))


def revalidate_before_finalization(stage_root: Path, ids: list[str]) -> None:
    validate_authority_sources(ids)
    installed_harness_path()
    for arm in ARMS:
        validate_capture(arm, ids)
    authority_dir = stage_root / "authority"
    if sha256(authority_dir / "convert_sr_to_patch.py") != CONVERTER_SHA256 or sha256(authority_dir / "swebench_verified.json") != DATASET_SHA256:
        fail("staged authority drifted before finalization")
    validate_repo_mirror(authority_dir / "swebench_verified.json", ids)


def execute(output: Path) -> None:
    ids = ids_from_authority()
    validate_authority_sources(ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        fail("publication lock already exists; another publisher owns this target")
    try:
        os.write(descriptor, f"pid={os.getpid()} target={output}\n".encode())
        if output.exists():
            fail("output already exists; immutable successor creation refuses overwrite")
        stage = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
        stage.mkdir()
        try:
            write_json(stage / "state.json", {"status": "STAGING_NOT_FINAL", "target": str(output), "created_at": datetime.now(UTC).isoformat()})
            converter = stage_authority(stage, ids)
            write_json(stage / "package_provenance.json", {"schema_version": "epyc.append-27b-v4-authority.v1", "created_at": datetime.now(UTC).isoformat(),
                "no_inference": True, "frozen_table_sha256": FROZEN_TABLE_SHA256, "frozen_converter_sha256": CONVERTER_SHA256,
                "arms": [{"name": arm["name"], "capture_sha256": arm["raw_sha256"], "license_gate": arm["license_gate"]} for arm in ARMS]})
            sealed = [(seal_arm(stage, arm, ids, converter), arm) for arm in ARMS]
            run_arms_concurrently(sealed, ids, output.name)
            revalidate_before_finalization(stage, ids)
            # This state is itself sealed by finalization.sha256, before the
            # only publication rename can occur.
            write_json(stage / "state.json", {"status": "FINALIZED", "target": str(output)})
            finalize(stage, ids)
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


def preflight() -> None:
    ids = ids_from_authority(); validate_authority_sources(ids); installed_harness_path()
    for arm in ARMS: validate_capture(arm, ids)
    print(json.dumps({"status": "PRECHECK_OK", "no_inference": True, "arms": [arm["name"] for arm in ARMS]}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--preflight", action="store_true"); group.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.preflight: preflight()
        else:
            if not args.output_dir: fail("--execute requires --output-dir")
            execute(args.output_dir)
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
