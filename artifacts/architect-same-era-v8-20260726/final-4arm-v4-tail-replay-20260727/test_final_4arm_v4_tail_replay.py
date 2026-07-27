from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("final_4arm_replay", HERE / "final_4arm_v4_tail_replay.py")
assert SPEC and SPEC.loader
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _ids() -> list[str]:
    return [f"repo__case-{index}" for index in range(40)]


def _rows(ids: list[str], *, error: bool = False) -> list[dict]:
    return [
        {"id": instance_id, "response": "payload", "finish_reason": "stop", "request_error": "bad" if error and index == 0 else ""}
        for index, instance_id in enumerate(ids)
    ]


def test_validate_raw_rows_rejects_order_and_request_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = _ids()
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in _rows(ids)))
    arm = {"name": "test", "raw": raw, "raw_sha256": replay.sha256(raw), "legacy": True}
    assert replay.validate_raw_rows(arm, ids)[0]["id"] == ids[0]
    swapped = _rows(ids); swapped[0], swapped[1] = swapped[1], swapped[0]
    raw.write_text("".join(json.dumps(row) + "\n" for row in swapped))
    arm["raw_sha256"] = replay.sha256(raw)
    with pytest.raises(RuntimeError, match="order"):
        replay.validate_raw_rows(arm, ids)
    raw.write_text("".join(json.dumps(row) + "\n" for row in _rows(ids, error=True)))
    arm["raw_sha256"] = replay.sha256(raw)
    with pytest.raises(RuntimeError, match="request-error"):
        replay.validate_raw_rows(arm, ids)


def test_conversion_forces_length_empty_and_ledger_is_exhaustive(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = _ids()
    class FakeConverter:
        rows = {instance_id: {} for instance_id in ids}
        @staticmethod
        def apply_blocks(_inst, _response, diagnostics):
            diagnostics.append({"block_index": 0, "outcome": "skipped_search_not_found", "search_sha256": "a" * 64, "replace_sha256": "b" * 64})
            return "patch", 1, 1
        @staticmethod
        def row_diagnostic(row, patch, blocks, _runner):
            return {
                "instance_id": row["id"], "finish_reason": row["finish_reason"], "empty_patch": not bool(patch),
                "empty_patch_reason": "empty_response" if not patch else None,
                "conversion_disposition": "model_truncation_empty_patch" if row["finish_reason"] == "length" else "converted",
                "blocks": blocks,
            }
    rows = _rows(ids); rows[3]["finish_reason"] = "length"
    arm = {"name": "A1", "label": "A1", "legacy": True}
    predictions, diagnostics, counts = replay.convert_rows(FakeConverter(), arm, rows)
    assert predictions[3]["model_patch"] == ""
    assert counts["length_rows_forced_empty"] == 1
    ledger = replay.ledger_for(arm, diagnostics, {"converter_source": {"sha256": "c" * 64}})
    replay.validate_conversion(arm, ids, predictions, diagnostics, ledger)
    assert ledger["aggregate"]["diagnostic_skipped_block_count"] == 39
    assert ledger["aggregate"]["empty_patch_row_count"] == 1
    assert ledger["status"] == "EXHAUSTIVE_PINNED_V4_OUTCOME"
    assert all("safe_arm_neutral_recovery_exists" not in row for row in ledger["skipped_blocks"])
    assert all(row["additional_recovery_attempted"] is False for row in ledger["skipped_blocks"])


def test_conversion_validation_rejects_missing_skip_ledger() -> None:
    ids = _ids()
    arm = {"name": "A1", "legacy": True}
    predictions = [{"instance_id": instance_id, "model_name_or_path": "A1", "model_patch": ""} for instance_id in ids]
    diagnostics = [{"instance_id": instance_id, "empty_patch": True, "blocks": [], "finish_reason": "stop", "empty_patch_reason": "x", "conversion_disposition": "x"} for instance_id in ids]
    diagnostics[0]["blocks"] = [{"block_index": 0, "outcome": "skipped_search_not_found", "search_sha256": "a" * 64, "replace_sha256": "b" * 64}]
    ledger = replay.ledger_for(arm, diagnostics, {})
    ledger["skipped_blocks"] = []
    with pytest.raises(RuntimeError, match="not exhaustive"):
        replay.validate_conversion(arm, ids, predictions, diagnostics, ledger)


def test_report_validator_requires_exact_counts_and_zero_harness_errors(tmp_path: Path) -> None:
    ids = _ids()
    arm_dir = tmp_path / "A1"; (arm_dir / "report").mkdir(parents=True)
    predictions = [
        {"instance_id": instance_id, "model_name_or_path": "A1", "model_patch": "" if index == 0 else "patch"}
        for index, instance_id in enumerate(ids)
    ]
    (arm_dir / "manifest.json").write_text(json.dumps({
        "arm": "A1",
        "operator_directive_provenance": {"quality_provenance": {"current_era_quality_decision_input": True}},
    }))
    (arm_dir / "predictions.sealed.json").write_text(json.dumps(predictions))
    report = {
        "submitted_ids": ids, "submitted_instances": 40,
        "completed_ids": ids[1:], "completed_instances": 39,
        "empty_patch_ids": ids[:1], "empty_patch_instances": 1,
        "resolved_ids": ids[1:20], "resolved_instances": 19,
        "unresolved_ids": ids[20:], "unresolved_instances": 20,
        "error_ids": [], "error_instances": 0,
    }
    report_path = arm_dir / "report" / "official.json"; report_path.write_text(json.dumps(report))
    assert replay.validate_report(arm_dir, ids)["resolved"] == 19
    report["resolved_instances"] = 20; report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="count does not match"):
        replay.validate_report(arm_dir, ids)


def test_quality_transfer_provenance_is_current_era_quality_only() -> None:
    legacy = replay.quality_provenance({"legacy": True})
    laguna = replay.quality_provenance({"legacy": False})
    assert legacy["same_era_generation"] is False
    assert legacy["quality_transfer_to_v8_eligible"] is True
    assert legacy["transfer_basis"] == replay.KERNEL_PARITY_BASIS
    assert laguna["same_era_generation"] is True
    assert laguna["quality_transfer_to_v8_eligible"] is True
    assert all(not value["speed_or_throughput_transfer_claim"] for value in (legacy, laguna))


def test_active_official_harness_blocks_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        stdout = "999 python -m swebench.harness.run_evaluation --run_id another\n"
    monkeypatch.setattr(replay.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setattr(replay.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="another official SWE harness"):
        replay.check_no_official_swe_harness()
