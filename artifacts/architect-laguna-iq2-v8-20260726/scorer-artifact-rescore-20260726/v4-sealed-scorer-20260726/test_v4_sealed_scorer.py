from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_seals_only_terminal_v4_capture() -> None:
    result = subprocess.run(["python3", str(HERE / "build_v4_scorer_package.py")], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((HERE / "sealed_package_manifest.json").read_text())
    assert manifest["status"] == "READY_FOR_OFFICIAL_SCORING"
    assert len(manifest["requested_ids"]) == 40
    ledger = json.loads((HERE / "v4_skip_disposition_and_supersession.json").read_text())
    assert ledger["status"] == "TERMINAL_V4_REVIEWED_NONRECOVERY_DISPOSITION"
    assert len(ledger["skipped_blocks"]) == 3
    assert "fullcapture5" in ledger["supersession"]["rule"]


def test_arm_neutral_audit_preserves_all_four_before_counts(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    result = subprocess.run(["python3", str(HERE / "reconversion_audit.py"), "--out", str(out)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    audit = json.loads(out.read_text())
    assert [arm["name"] for arm in audit["arms"]] == [
        "Laguna_promptfix_v4", "Laguna_historical_diagnostic_only", "A3_same_era_banked", "A4_same_era_banked"
    ]
    assert all(arm["rows"] == 40 for arm in audit["arms"])
    assert audit["arms"][0]["before"] == {"empty_patches": 11, "blocks_skipped": 3}


def test_arm_neutral_audit_rejects_converter_drift(tmp_path: Path) -> None:
    source = HERE.parents[3] / "artifacts/architect-code-eval-20260724/convert_sr_to_patch.py"
    drifted = tmp_path / "converter.py"
    shutil.copyfile(source, drifted)
    drifted.write_text(drifted.read_text() + "\n# test drift\n")
    result = subprocess.run(["python3", str(HERE / "reconversion_audit.py"), "--converter", str(drifted)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "source drift" in result.stderr


@pytest.fixture
def scorer_contract():
    validator = _module("v4_validator_fixture", "validate_official_swebench_report.py")
    manifest = json.loads((HERE / "sealed_package_manifest.json").read_text())
    ids = manifest["requested_ids"]
    predictions = json.loads((HERE / "predictions_v4.json").read_text())
    empty = [
        row["instance_id"]
        for row in predictions
        if row["model_patch"] == ""
    ]
    completed = [instance_id for instance_id in ids if instance_id not in set(empty)]
    assert len(completed) == 29
    assert len(empty) == 11
    incomplete = [f"unsubmitted__case-{index}" for index in range(460)]
    report = {
        "schema_version": 2,
        "total_instances": 500,
        "submitted_ids": ids,
        "submitted_instances": 40,
        "completed_ids": completed,
        "completed_instances": 29,
        "empty_patch_ids": empty,
        "empty_patch_instances": 11,
        "resolved_ids": completed[:18],
        "resolved_instances": 18,
        "unresolved_ids": completed[18:],
        "unresolved_instances": 11,
        "error_ids": [],
        "error_instances": 0,
        "incomplete_ids": incomplete,
        "incomplete_instances": 460,
    }
    return validator, manifest, empty, report


def test_post_score_validator_accepts_29_completed_11_empty_and_460_unsubmitted(
    scorer_contract,
) -> None:
    validator, manifest, empty, report = scorer_contract
    result = validator.validate_selected_report(report, manifest["requested_ids"], empty)
    assert result == {
        "status": "VALID",
        "protocol_id": "laguna-swe-v4-selected-instance-postscore.v1",
        "denominator": 40,
        "completed_instances": 29,
        "empty_patch_instances": 11,
        "resolved_instances": 18,
        "unresolved_instances": 11,
        "error_instances": 0,
        "unsubmitted_incomplete_instances": 460,
        "percent_resolved_over_40": 45.0,
    }


@pytest.mark.parametrize(
    "field",
    [
        "submitted_ids",
        "completed_ids",
        "empty_patch_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
        "incomplete_ids",
    ],
)
def test_post_score_validator_rejects_duplicate_ids(
    scorer_contract,
    field: str,
) -> None:
    validator, manifest, empty, report = scorer_contract
    broken = copy.deepcopy(report)
    source = broken[field] or [manifest["requested_ids"][0]]
    broken[field] = [*source, source[0]]
    with pytest.raises(ValueError, match="duplicate"):
        validator.validate_selected_report(broken, manifest["requested_ids"], empty)


@pytest.mark.parametrize(
    "field",
    [
        "submitted_instances",
        "completed_instances",
        "empty_patch_instances",
        "resolved_instances",
        "unresolved_instances",
        "error_instances",
        "incomplete_instances",
    ],
)
def test_post_score_validator_rejects_numeric_list_mismatch(
    scorer_contract,
    field: str,
) -> None:
    validator, manifest, empty, report = scorer_contract
    broken = copy.deepcopy(report)
    broken[field] += 1
    with pytest.raises(ValueError, match="does not match"):
        validator.validate_selected_report(broken, manifest["requested_ids"], empty)


def test_post_score_validator_rejects_swapped_empty_patch_id(scorer_contract) -> None:
    validator, manifest, empty, report = scorer_contract
    broken = copy.deepcopy(report)
    forged_empty = broken["completed_ids"][0]
    formerly_empty = broken["empty_patch_ids"][0]
    broken["empty_patch_ids"][0] = forged_empty
    broken["completed_ids"][0] = formerly_empty
    broken["resolved_ids"][0] = formerly_empty
    with pytest.raises(ValueError, match="do not match sealed predictions"):
        validator.validate_selected_report(broken, manifest["requested_ids"], empty)


def test_post_score_validator_rejects_wrong_partitions_and_errors(scorer_contract) -> None:
    validator, manifest, empty, report = scorer_contract
    missing = copy.deepcopy(report)
    missing["completed_ids"] = missing["completed_ids"][:-1]
    missing["completed_instances"] -= 1
    missing["unresolved_ids"] = missing["unresolved_ids"][:-1]
    missing["unresolved_instances"] -= 1
    with pytest.raises(ValueError, match="do not partition submitted"):
        validator.validate_selected_report(missing, manifest["requested_ids"], empty)

    errored = copy.deepcopy(report)
    errored["error_ids"] = [manifest["requested_ids"][0]]
    errored["error_instances"] = 1
    with pytest.raises(ValueError, match="reported selected-instance errors"):
        validator.validate_selected_report(errored, manifest["requested_ids"], empty)


def test_package_contract_rejects_prediction_hash_drift(
    scorer_contract,
    tmp_path: Path,
) -> None:
    validator, _manifest, _empty, _report = scorer_contract
    copied = tmp_path / "package"
    shutil.copytree(HERE, copied)
    predictions = copied / "predictions_v4.json"
    predictions.write_text(predictions.read_text() + "\n")
    with pytest.raises(ValueError, match="predictions SHA"):
        validator.load_package_contract(copied)


def test_cli_binds_report_hash_and_rejects_expected_hash_drift(
    scorer_contract,
    tmp_path: Path,
) -> None:
    _validator, _manifest, _empty, report = scorer_contract
    report_path = tmp_path / "official.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    command = [
        "python3",
        str(HERE / "validate_official_swebench_report.py"),
        "--package",
        str(HERE),
        "--report",
        str(report_path),
        "--expected-report-sha256",
        report_sha,
        "--out",
        str(receipt),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    validation = json.loads(receipt.read_text())
    assert validation["bindings"]["report_sha256"] == report_sha
    assert validation["bindings"]["predictions_sha256"] == json.loads(
        (HERE / "sealed_package_manifest.json").read_text()
    )["predictions_sha256"]
    command[command.index(report_sha)] = "0" * 64
    drifted = subprocess.run(command, text=True, capture_output=True)
    assert drifted.returncode != 0
    assert "report SHA drifted" in drifted.stderr
