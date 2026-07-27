from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("append_27b", HERE / "append_27b_v4_authority.py")
assert SPEC and SPEC.loader
append = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(append)


def test_preflight_binds_frozen_authority_and_complete_captures() -> None:
    append.preflight()


def test_capture_rejects_request_error(tmp_path: Path) -> None:
    ids = [f"x-{number}" for number in range(40)]
    raw = tmp_path / "capture.jsonl"
    rows = [{"id": item, "capture_schema_version": append.CAPTURE_SCHEMA, "arm": "test", "suite": "swebench_oracle", "seed": 42, "rep": 0,
             "runner_source_sha256": append.RUNNER_SHA256, "request_error": "failed", "finish_reason": "request_error",
             "prompt": "p", "response": "r", "reasoning": "", "prompt_fingerprint": append.text_fingerprint("p"),
             "response_fingerprint": append.text_fingerprint("r"), "reasoning_fingerprint": append.text_fingerprint("")} for item in ids]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    arm = {"name": "test", "capture_arm": "test", "raw": raw, "raw_sha256": append.sha256(raw)}
    with pytest.raises(RuntimeError, match="request-error"):
        append.validate_capture(arm, ids)


def test_finalize_preserves_four_arm_rows_and_pending_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [f"x-{number}" for number in range(40)]
    row_keys = {"denominator": 40, "empty_patch_failures": 0, "harness_errors": 0, "percent_resolved": 1.0,
                "quality_provenance": append.quality_provenance(), "report": "frozen", "report_sha256": "a" * 64, "resolved": 1}
    frozen = tmp_path / "frozen.json"; frozen.write_text(json.dumps({"ranking": [{"arm": name, **row_keys} for name in ("A1", "A3", "A4", "Laguna")], "status": "FINAL"}))
    monkeypatch.setattr(append, "FROZEN_TABLE", frozen)
    for arm in append.ARMS:
        arm_dir = tmp_path / arm["name"]; (arm_dir / "report").mkdir(parents=True)
        report = {"submitted_ids": ids, "submitted_instances": 40, "completed_ids": ids, "completed_instances": 40,
                  "empty_patch_ids": [], "empty_patch_instances": 0, "resolved_ids": ids[:2], "resolved_instances": 2,
                  "unresolved_ids": ids[2:], "unresolved_instances": 38, "error_ids": [], "error_instances": 0}
        (arm_dir / "report" / "official.json").write_text(json.dumps(report))
        (arm_dir / "predictions.sealed.json").write_text(json.dumps([
            {"instance_id": instance_id, "model_name_or_path": arm["name"], "model_patch": "patch"} for instance_id in ids]))
    append.finalize(tmp_path, ids)
    table = json.loads((tmp_path / "expanded_six_arm_table.json").read_text())
    assert [row["arm"] for row in table["rows"]] == ["A1", "A3", "A4", "Laguna", "A3-tc", "A3-ff"]
    assert table["license_gate_status"]["A3-tc"] == "PENDING_NO_DECLARED_LICENSE"
    assert table["mtp_disposition"].startswith("A3-ff embedded-MTP")
    assert {tuple(sorted(row)) for row in table["rows"]} == {tuple(sorted(table["rows"][0]))}
    assert all(not row["report"].startswith("/") for row in table["rows"][-2:])


@pytest.mark.parametrize(
    ("field", "overlapping_ids"),
    [
        ("empty_patch_ids", ["x-0"]),
        ("unresolved_ids", ["x-0"]),
    ],
)
def test_validate_report_rejects_overlapping_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, overlapping_ids: list[str]
) -> None:
    ids = [f"x-{number}" for number in range(40)]
    arm_dir = tmp_path / "A3-test"; report_dir = arm_dir / "report"; report_dir.mkdir(parents=True)
    report = {
        "submitted_ids": ids, "submitted_instances": 40,
        "completed_ids": ids, "completed_instances": 40,
        "empty_patch_ids": [], "empty_patch_instances": 0,
        "resolved_ids": ids, "resolved_instances": 40,
        "unresolved_ids": [], "unresolved_instances": 0,
        "error_ids": [], "error_instances": 0,
    }
    report[field] = overlapping_ids
    report[field.replace("_ids", "_instances")] = len(overlapping_ids)
    (report_dir / "official.json").write_text(json.dumps(report))
    (arm_dir / "predictions.sealed.json").write_text(json.dumps([
        {"instance_id": instance_id, "model_name_or_path": "test", "model_patch": "patch"} for instance_id in ids
    ]))
    monkeypatch.setattr(append, "find_report", lambda _arm_dir: report_dir / "official.json")
    with pytest.raises(RuntimeError, match="partition is invalid"):
        append.validate_report(arm_dir, ids)


def test_authority_dataset_and_installed_harness_drift_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_dataset = tmp_path / "dataset.json"; bad_dataset.write_text("[]")
    monkeypatch.setattr(append, "DATASET", bad_dataset)
    with pytest.raises(RuntimeError, match="frozen authority drifted"):
        append.validate_authority_sources(["x"])

    harness = tmp_path / "run_evaluation.py"; harness.write_text("different")
    class Result:
        returncode = 0
        stdout = str(harness) + "\n"
        stderr = ""
    monkeypatch.setattr(append.subprocess, "run", lambda *_args, **_kwargs: Result())
    with pytest.raises(RuntimeError, match="installed SWE-bench harness"):
        append.installed_harness_path()


def test_failed_evaluation_stays_staged_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "published"
    ids = [f"x-{number}" for number in range(40)]
    monkeypatch.setattr(append, "ids_from_authority", lambda: ids)
    monkeypatch.setattr(append, "validate_authority_sources", lambda _ids: None)
    monkeypatch.setattr(append, "stage_authority", lambda _root, _ids: object())
    def fake_seal(root: Path, arm: dict, _ids: list[str], _converter: object) -> Path:
        path = root / arm["name"]; path.mkdir(); return path
    monkeypatch.setattr(append, "seal_arm", fake_seal)
    monkeypatch.setattr(append, "run_arms_concurrently", lambda *_args: (_ for _ in ()).throw(RuntimeError("eval failed")))
    with pytest.raises(RuntimeError, match="eval failed"):
        append.execute(output)
    assert not output.exists()
    stages = list(tmp_path.glob(".published.staging-*"))
    assert len(stages) == 1
    assert json.loads((stages[0] / "state.json").read_text())["status"] == "FAILED_NOT_FINAL"


def test_successful_publish_seals_finalized_state_and_respects_target_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "published"; ids = [f"x-{number}" for number in range(40)]
    monkeypatch.setattr(append, "ids_from_authority", lambda: ids)
    monkeypatch.setattr(append, "validate_authority_sources", lambda _ids: None)
    monkeypatch.setattr(append, "stage_authority", lambda _root, _ids: object())
    monkeypatch.setattr(append, "seal_arm", lambda root, arm, _ids, _converter: (root / arm["name"]).mkdir() or root / arm["name"])
    monkeypatch.setattr(append, "run_arms_concurrently", lambda *_args: None)
    monkeypatch.setattr(append, "revalidate_before_finalization", lambda *_args: None)
    def fake_finalize(root: Path, _ids: list[str]) -> None:
        assert json.loads((root / "state.json").read_text())["status"] == "FINALIZED"
        (root / "finalization.sha256").write_text("sealed\n")
    monkeypatch.setattr(append, "finalize", fake_finalize)
    append.execute(output)
    assert json.loads((output / "state.json").read_text())["status"] == "FINALIZED"
    assert not (tmp_path / ".published.publish.lock").exists()

    locked = tmp_path / "locked"; (tmp_path / ".locked.publish.lock").write_text("other")
    with pytest.raises(RuntimeError, match="publication lock"):
        append.execute(locked)


def test_nonrecovery_ledger_is_exhaustive() -> None:
    diagnostics = [{"instance_id": "one", "empty_patch": True, "finish_reason": "length", "empty_patch_reason": "model_length_cap",
                    "blocks": [{"block_index": 0, "outcome": "skipped_search_not_found", "search_sha256": "a" * 64, "replace_sha256": "b" * 64}]}]
    ledger = append.nonrecovery_ledger({"name": "test"}, diagnostics)
    append.validate_ledger(diagnostics, ledger)
    ledger["empty_patch_rows"] = []
    with pytest.raises(RuntimeError, match="not exhaustive"):
        append.validate_ledger(diagnostics, ledger)
