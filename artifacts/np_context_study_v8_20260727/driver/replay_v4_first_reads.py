#!/usr/bin/env python3
"""Seal v8 first-read captures for deterministic replay without regeneration.

``--seal`` consumes only completed ThinkingCap/Fable quality captures.  It
replays the pinned v4 SEARCH/REPLACE converter and LiveCodeBench executable
scorer over those saved responses, copying every authority into a new,
immutable package.  ``--official-score`` is intentionally a separate opt-in:
it refuses to run until the concurrent Laguna-Q4 CPU run has a terminal
cleanup receipt.  This module never starts a model server.
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
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
# driver -> np_context_study_v8_20260727 -> artifacts -> research root
REPO = HERE.parents[2]
ART = HERE.parent
CANON = REPO / "artifacts" / "architect-code-eval-20260724"
# The terminal same-era table used the sealed v4 converter, not the mutable
# canonical converter.  New first reads must use that exact instrument.
CONVERTER = REPO / "artifacts/architect-27b-finetunes-v8-20260726/expanded-six-arm-v4-tail-replay-20260727/A3-tc/converter_v4.sealed.py"
HARNESS = REPO / "artifacts/architect-27b-finetunes-v8-20260726/expanded-six-arm-v4-tail-replay-20260727/A3-tc/harness_module.sealed.py"
SWE_QUESTIONS = CANON / "questions_swebench_oracle.json"
LCB_QUESTIONS = CANON / "questions_livecodebench_hard.json"
SWE_DATASET = CANON / "swebench_verified.json"
RUNNER = REPO / "scripts/benchmark/v7_quality_gate_runner.py"
LCB_SCORER = REPO / "scripts/benchmark/code_exec_scorer.py"
Q4_AUTHORITY = REPO / "scripts/benchmark/laguna_q4_cpu_bench_runner.py"
SWEBENCH_PYTHON = REPO / ".venv-swebench/bin/python"
TASKSET = Path("/usr/bin/taskset")
HARNESS_CPUSET = "184-191"

CAPTURE_SCHEMA = "v7_quality_gate_capture.v4"
CONVERTER_SHA256 = "6bd2302dda3e5139cc6faabcc5639bdcf85b27895f93a9181cbb53dd65749507"
HARNESS_SHA256 = "6959f0b4e4eaf979771f529b88e3e9df1daa7fe86bc4291feec2e7d320bf7f2e"
SWE_QUESTIONS_SHA256 = "f82a5191274048f2fdf432df7a0ebf4017ad982b954d6aa075326a1302df1c3c"
LCB_QUESTIONS_SHA256 = "d51e56f601e3d153910d086b35c6aea94f4d903bab0427c8a49ffe895a6287c4"
RUNNER_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
LCB_SCORER_SHA256 = "12b8c9408d4b2f606929e37316c3f1c3d8f6252925dfb7bf6bdea541c3ef23cc"
SWE_DATASET_SHA256 = "b087b5dad72b3e765a6cf93a9e7d516d8796698a0fd358abb73c6627df19f66e"
Q4_AUTHORITY_SHA256 = "38392828f3e17b23dbf5d7ae596afc3ed23c29ead52e6fcd3a775d63bdadc242"

ARMS = (
    {"name": "thinkingcap_q8", "label": "A3_tc_thinkingcap_q8", "mtp_depth": 4},
    {"name": "fable_non_mtp_q8", "label": "A3_ff_fable_non_mtp_q8", "mtp_depth": 0},
    {"name": "fable_mtp_q8", "label": "A3_ff_fable_mtp_q8", "mtp_depth": 1},
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def text_fingerprint(value: str) -> dict[str, int | str]:
    payload = value.encode("utf-8")
    return {"chars": len(value), "utf8_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def pin(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        fail(f"{label} SHA-256 drifted")


def static_authorities() -> tuple[list[str], dict[str, dict[str, Any]], Any, Any, Path]:
    pin(CONVERTER, CONVERTER_SHA256, "v4 converter")
    pin(SWE_QUESTIONS, SWE_QUESTIONS_SHA256, "SWE question source")
    pin(LCB_QUESTIONS, LCB_QUESTIONS_SHA256, "LCB question source")
    pin(SWE_DATASET, SWE_DATASET_SHA256, "SWE dataset")
    pin(RUNNER, RUNNER_SHA256, "capture runner")
    pin(LCB_SCORER, LCB_SCORER_SHA256, "LCB executable scorer")
    pin(Q4_AUTHORITY, Q4_AUTHORITY_SHA256, "Laguna Q4 official-score authority")
    swe = json.loads(SWE_QUESTIONS.read_text())
    lcb = json.loads(LCB_QUESTIONS.read_text())
    swe_ids = [row.get("id") for row in swe]
    if len(swe_ids) != 40 or len(set(swe_ids)) != 40 or not all(isinstance(item, str) for item in swe_ids):
        fail("SWE source is not the exact 40-ID denominator")
    lcb_ids = [row.get("id") for row in lcb]
    if len(lcb_ids) != 53 or len(set(lcb_ids)) != 53 or not all(isinstance(item, str) for item in lcb_ids):
        fail("LCB source is not the exact 53-ID denominator")
    # The sealed converter locates its immutable SWE checkout relative to
    # __file__.  Execute its byte-identical source with the canonical authority
    # location so matching still reads the pinned repository snapshots.
    converter = types.ModuleType("v8_first_read_converter")
    converter.__file__ = str(CANON / "convert_sr_to_patch.py")
    exec(compile(CONVERTER.read_text(), converter.__file__, "exec"), converter.__dict__)
    scorer_spec = importlib.util.spec_from_file_location("v8_first_read_lcb_scorer", LCB_SCORER)
    if not scorer_spec or not scorer_spec.loader:
        fail("cannot load pinned replay authorities")
    scorer = importlib.util.module_from_spec(scorer_spec)
    scorer_spec.loader.exec_module(scorer)
    pin(HARNESS, HARNESS_SHA256, "sealed SWE harness")
    if not SWEBENCH_PYTHON.is_file():
        fail("pinned SWE-bench interpreter is unavailable")
    installed_harness = Path(
        subprocess.run(
            [
                str(SWEBENCH_PYTHON),
                "-c",
                "import inspect,swebench.harness.run_evaluation as m; print(inspect.getsourcefile(m))",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    pin(installed_harness, HARNESS_SHA256, "installed SWE harness")
    return swe_ids, {row["id"]: row for row in lcb}, converter, scorer, HARNESS


def capture_path(arm: dict[str, Any], suite: str) -> Path:
    return ART / arm["label"] / f"quality_{suite}" / "per_question.jsonl"


def complete_rows(arm: dict[str, Any], suite: str, expected_ids: list[str]) -> list[dict[str, Any]]:
    raw = capture_path(arm, suite)
    summary = raw.with_name("summary.json")
    if not raw.is_file() or not summary.is_file():
        fail(f"{arm['name']} {suite} is incomplete (raw capture or summary absent)")
    rows = read_jsonl(raw)
    if [row.get("id") for row in rows] != expected_ids:
        fail(f"{arm['name']} {suite} denominator/order drifted")
    summary_data = json.loads(summary.read_text())
    suite_rows = summary_data.get("suites")
    if not isinstance(suite_rows, list) or len(suite_rows) != 1:
        fail(f"{arm['name']} {suite} summary shape drifted")
    declared = suite_rows[0]
    if declared.get("suite") != suite or declared.get("n") != len(expected_ids) or declared.get("errors") != 0:
        fail(f"{arm['name']} {suite} summary is not a clean completed capture")
    expected_arm = f"{arm['label']}_rb1024_{suite}"
    for row in rows:
        if (row.get("capture_schema_version") != CAPTURE_SCHEMA or row.get("runner_source_sha256") != RUNNER_SHA256
                or row.get("suite") != suite or row.get("arm") != expected_arm or row.get("seed") != 42
                or row.get("rep") != 0 or row.get("request_error")):
            fail(f"{arm['name']} {suite} row {row.get('id')} capture contract drifted")
        for field in ("prompt", "response", "reasoning"):
            if not isinstance(row.get(field), str) or row.get(f"{field}_fingerprint") != text_fingerprint(row[field]):
                fail(f"{arm['name']} {suite} row {row.get('id')} {field} fingerprint drifted")
    return rows


def seal_swe(arm: dict[str, Any], rows: list[dict[str, Any]], converter: Any) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, int]]:
    predictions, diagnostics = [], []
    counts = {"blocks_applied": 0, "blocks_skipped": 0, "empty_patch_count": 0, "length_forced_empty": 0}
    for row in rows:
        blocks: list[dict[str, Any]] = []
        if row.get("finish_reason") == "length":
            patch, applied, skipped = "", 0, 0
            counts["length_forced_empty"] += 1
        else:
            patch, applied, skipped = converter.apply_blocks(converter.rows[row["id"]], row["response"], blocks)
        diagnostic = converter.row_diagnostic(row, patch, blocks, RUNNER_SHA256)
        if not diagnostic.get("scoring_eligible"):
            fail(f"{arm['name']} SWE row {row['id']} is not v4 scoring eligible")
        if row.get("finish_reason") == "length":
            diagnostic.update(empty_patch=True, empty_patch_reason="model_length_forced_empty",
                              conversion_disposition="model_length_contract_failure")
        predictions.append({"instance_id": row["id"], "model_name_or_path": arm["label"], "model_patch": patch})
        diagnostics.append(diagnostic)
        counts["blocks_applied"] += applied
        counts["blocks_skipped"] += skipped
        counts["empty_patch_count"] += int(not bool(patch))
    return predictions, diagnostics, counts


def replay_lcb(arm: dict[str, Any], rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]], scorer: Any) -> dict[str, Any]:
    replayed = []
    for row in rows:
        question = questions[row["id"]]
        config = question.get("scoring_config")
        if question.get("scoring_method") != "code_execution" or not isinstance(config, dict):
            fail(f"LCB scoring contract drifted for {row['id']}")
        result = scorer.score_code(row["response"], config.get("test_cases", []), config.get("language", "python"),
                                   timeout=int(config.get("timeout", 10)))
        replayed.append({"id": row["id"], "response_sha256": row["response_fingerprint"]["sha256"], "result": result})
    correct = sum(bool(row["result"].get("correct")) for row in replayed)
    return {"schema_version": "epyc.v8-first-read-lcb-replay.v1", "arm": arm["name"], "denominator": 53,
            "correct": correct, "accuracy": correct / 53, "rows": replayed,
            "authority": {"path": str(LCB_SCORER), "sha256": LCB_SCORER_SHA256}, "no_inference": True}


def write_digest(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "seal.sha256")
    (root / "seal.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in files))


def verify_digest(root: Path) -> None:
    digest = root / "seal.sha256"
    if not digest.is_file():
        fail("sealed replay package has no hash ledger")
    declared: dict[str, str] = {}
    for line in digest.read_text().splitlines():
        value, relative = line.split("  ", 1)
        if len(value) != 64 or relative in declared:
            fail("sealed replay package hash ledger is malformed")
        declared[relative] = value
    actual = {str(path.relative_to(root)): sha256(path) for path in root.rglob("*")
              if path.is_file() and path.name != "seal.sha256"}
    if declared != actual:
        fail("sealed replay package hash ledger does not match immutable contents")


def seal() -> Path:
    swe_ids, lcb_questions, converter, scorer, harness = static_authorities()
    lcb_ids = list(lcb_questions)
    # Validate every input before allocating an output location.  A partial
    # capture must leave no package that a later scorer could mistake for sealed.
    captures = {
        arm["name"]: (complete_rows(arm, "swebench_oracle", swe_ids),
                      complete_rows(arm, "livecodebench_hard", lcb_ids))
        for arm in ARMS
    }
    run = ART / "v4_first_read_replays" / ("v4-first-read-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    if run.exists():
        fail("run ID collision")
    run.mkdir(parents=True)
    provenance: dict[str, Any] = {"schema_version": "epyc.v8-first-read-v4-replay.v1", "status": "SEALED_NO_DOCKER",
        "created_at": datetime.now(UTC).isoformat(), "no_inference": True, "docker_invoked": False,
        "authorities": {str(path): sha256(path) for path in (CONVERTER, SWE_QUESTIONS, LCB_QUESTIONS, SWE_DATASET, RUNNER, LCB_SCORER, harness)},
        "arms": {}}
    for arm in ARMS:
        swe_rows, lcb_rows = captures[arm["name"]]
        arm_dir = run / arm["name"]
        arm_dir.mkdir()
        predictions, diagnostics, counts = seal_swe(arm, swe_rows, converter)
        lcb = replay_lcb(arm, lcb_rows, lcb_questions, scorer)
        for name, source in (("swe_raw_capture.sealed.jsonl", capture_path(arm, "swebench_oracle")),
                             ("lcb_raw_capture.sealed.jsonl", capture_path(arm, "livecodebench_hard"))):
            shutil.copyfile(source, arm_dir / name)
        write_json(arm_dir / "swe_predictions.sealed.json", predictions)
        (arm_dir / "swe_conversion_diagnostics.sealed.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics))
        write_json(arm_dir / "lcb_deterministic_replay.json", lcb)
        provenance["arms"][arm["name"]] = {"label": arm["label"], "mtp_depth": arm["mtp_depth"], "swe": counts,
            "lcb_correct": lcb["correct"], "raw_capture_sha256": {"swe": sha256(capture_path(arm, "swebench_oracle")),
            "lcb": sha256(capture_path(arm, "livecodebench_hard"))}}
    shutil.copyfile(CONVERTER, run / "authority_converter_v4.py")
    shutil.copyfile(LCB_SCORER, run / "authority_lcb_scorer.py")
    shutil.copyfile(harness, run / "authority_swebench_harness.py")
    write_json(run / "provenance.json", provenance)
    write_digest(run)
    verify_digest(run)
    return run


def require_q4_terminal(path: Path) -> None:
    # A successful terminal receipt is deliberately required before a Docker
    # workload can compete with the CPU Laguna campaign.
    summary = path / "summary.json"
    swe_cleanup, lcb_cleanup = path / "swe_oracle.cleanup.json", path / "lcb_hard.cleanup.json"
    docker = path / "swe_docker_terminal.json"
    if not all(candidate.is_file() for candidate in (summary, swe_cleanup, lcb_cleanup, docker)):
        fail("Laguna Q4 terminal summary, per-suite cleanup, or Docker receipt is absent")
    summary_data, docker_data = json.loads(summary.read_text()), json.loads(docker.read_text())
    if summary_data.get("schema") != "epyc.laguna_q4_cpu_v8.summary.v2" or summary_data.get("status") != "ok":
        fail("Laguna Q4 is not terminally successful")
    if (docker_data.get("execution_error") or docker_data.get("postflight_errors")
            or docker_data.get("residual_unproven_ids") or docker_data.get("cleanup_failed_ids")):
        fail("Laguna Q4 Docker terminal receipt is not clean")
    for receipt in (swe_cleanup, lcb_cleanup):
        data = json.loads(receipt.read_text())
        if data.get("members_after_kill") or data.get("port_free") is not True:
            fail(f"Laguna Q4 cleanup receipt is not clean: {receipt.name}")


def load_q4_authority() -> Any:
    """Use the already-reviewed Docker lifecycle and report validator verbatim."""
    pin(Q4_AUTHORITY, Q4_AUTHORITY_SHA256, "Laguna Q4 official-score authority")
    spec = importlib.util.spec_from_file_location("laguna_q4_official_authority", Q4_AUTHORITY)
    if not spec or not spec.loader:
        fail("cannot load reviewed Laguna Q4 official-score authority")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_score(run: Path, q4_terminal: Path) -> Path:
    require_q4_terminal(q4_terminal)
    verify_digest(run)
    q4 = load_q4_authority()
    successor = run.parent / (run.name + "-official-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    if successor.exists():
        fail("official-score successor ID collision")
    successor.mkdir(parents=True)
    before_images = subprocess.run([str(q4.DOCKER), "images", "--no-trunc", "--format", "{{.Repository}}@{{.ID}}"],
                                   text=True, capture_output=True, check=True).stdout
    before_containers = subprocess.run([str(q4.DOCKER), "ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.State}} {{.Names}}"],
                                       text=True, capture_output=True, check=True).stdout
    evidence: dict[str, Any] = {"schema_version": "epyc.v8-first-read-official-successor.v1", "sealed_parent": str(run),
        "sealed_parent_digest_verified": True, "q4_terminal": str(q4_terminal), "docker_images_before": before_images.splitlines(),
        "docker_containers_before": before_containers.splitlines(),
        "q4_official_authority": {"path": str(Q4_AUTHORITY), "sha256": Q4_AUTHORITY_SHA256},
        "arms": {}}
    run_ids: list[str] = []
    try:
        for arm in ARMS:
            sealed = run / arm["name"] / "swe_predictions.sealed.json"
            predictions = json.loads(sealed.read_text())
            expected_ids = [row["instance_id"] for row in predictions]
            if len(expected_ids) != 40 or len(set(expected_ids)) != 40:
                fail(f"{arm['name']} sealed prediction denominator drifted")
            arm_out = successor / arm["name"]
            arm_out.mkdir()
            run_id = f"{successor.name}-{arm['name']}"
            run_ids.append(run_id)
            command = [str(TASKSET), "-c", HARNESS_CPUSET, str(SWEBENCH_PYTHON), "-m", "swebench.harness.run_evaluation",
                       "--dataset_name", str(SWE_DATASET), "--predictions_path", str(sealed), "--instance_ids", *expected_ids,
                       "--max_workers", "8", "--timeout", "1800", "--cache_level", "env", "--run_id", run_id,
                       "--report_dir", str(arm_out / "report")]
            (arm_out / "official_command.txt").write_text(" ".join(command) + "\n")
            result = q4.run_owned_command(command, cwd=arm_out, env={**q4.clean_env(), "HF_HOME": "/mnt/raid0/llm/cache/huggingface"},
                                          timeout_s=q4.OFFICIAL_SWE_TIMEOUT_S, label=f"official SWE {arm['name']}")
            (arm_out / "official.stdout").write_text(result.stdout)
            (arm_out / "official.stderr").write_text(result.stderr)
            if result.returncode:
                fail(f"official SWE harness exited {result.returncode} for {arm['name']}")
            reports = [path for path in arm_out.rglob("*.json") if path.name != "swe_predictions.sealed.json"
                       and isinstance(json.loads(path.read_text()), dict) and "resolved_ids" in json.loads(path.read_text())]
            if len(reports) != 1:
                fail(f"{arm['name']} has {len(reports)} official report candidates")
            validation = q4.validate_official_swe_report(reports[0], {"empty_patch_ids": [row["instance_id"] for row in predictions if not row["model_patch"]]})
            evidence["arms"][arm["name"]] = validation | {"report": str(reports[0]), "report_sha256": sha256(reports[0])}
    finally:
        after_images = subprocess.run([str(q4.DOCKER), "images", "--no-trunc", "--format", "{{.Repository}}@{{.ID}}"],
                                      text=True, capture_output=True, check=True).stdout
        after_containers = subprocess.run([str(q4.DOCKER), "ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.State}} {{.Names}}"],
                                          text=True, capture_output=True, check=True).stdout
        new_ids = sorted(set(q4.parse_docker_container_rows(after_containers)) - set(q4.parse_docker_container_rows(before_containers)))
        inspections, removed, unproven = {}, [], []
        for container_id in new_ids:
            inspected = q4.docker_operation([str(q4.DOCKER), "inspect", container_id], q4.clean_env())
            if inspected["error"]:
                unproven.append(container_id)
                continue
            payload = json.loads(str(inspected["stdout"]))
            inspections[container_id] = payload[0] if isinstance(payload, list) and payload else {}
            if not any(q4.campaign_container_owned(inspections[container_id], run_id) for run_id in run_ids):
                unproven.append(container_id)
                continue
            removed_op = q4.docker_operation([str(q4.DOCKER), "rm", "-f", container_id], q4.clean_env())
            if removed_op["error"]:
                unproven.append(container_id)
            else:
                removed.append(container_id)
        final_containers = subprocess.run([str(q4.DOCKER), "ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.State}} {{.Names}}"],
                                          text=True, capture_output=True, check=True).stdout
        # The reviewed validator enforces no residual/unowned container and no
        # mutation of pre-existing containers after owned-container removal.
        if unproven:
            fail(f"official SWE left unproven containers: {unproven}")
        q4.validate_docker_container_transition(before_containers, final_containers)
        if before_images != after_images:
            fail("official SWE changed Docker image IDs")
        evidence["docker_images_after"] = after_images.splitlines()
        evidence["docker_containers_after"] = after_containers.splitlines()
        evidence["docker_containers_final"] = final_containers.splitlines()
        evidence["new_container_inspections"] = inspections
        evidence["removed_owned_container_ids"] = removed
        write_json(successor / "official_provenance.json", evidence)
        write_digest(successor)
    return successor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--static-preflight", action="store_true")
    group.add_argument("--seal", action="store_true")
    group.add_argument("--official-score", metavar="SEALED_RUN")
    parser.add_argument("--q4-terminal-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.static_preflight:
            static_authorities()
            print(json.dumps({"status": "STATIC_PREFLIGHT_OK", "docker_invoked": False}, sort_keys=True))
        elif args.seal:
            print(json.dumps({"status": "SEALED_NO_DOCKER", "run": str(seal())}, sort_keys=True))
        else:
            if args.q4_terminal_dir is None:
                fail("--q4-terminal-dir is required with --official-score")
            successor = official_score(Path(args.official_score), args.q4_terminal_dir)
            print(json.dumps({"status": "OFFICIAL_SWE_COMPLETE", "run": str(successor)}, sort_keys=True))
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
