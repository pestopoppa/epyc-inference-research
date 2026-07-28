from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tc_nothink_replay", HERE / "replay_tc_nothink_v4.py")
assert SPEC and SPEC.loader
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def make_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[replay.Inputs, list[str]]:
    ids = [f"repo__issue-{number}" for number in range(40)]
    paths = {
        "capture": "capture.jsonl", "capture_summary": "summary.json", "capture_argv": "capture.argv", "capture_done": "capture.done",
        "frozen_table": "authority/table.json", "authority_finalization": "authority/finalization.sha256",
        "authority_ids": "authority/ids.jsonl", "converter": "authority/converter.py", "harness": "authority/harness.py", "dataset": "authority/dataset.json",
    }
    rows = []
    for number, instance_id in enumerate(ids):
        row = {
            "id": instance_id, "capture_schema_version": replay.CAPTURE_SCHEMA, "arm": replay.CAPTURE_ARM,
            "suite": "swebench_oracle", "seed": 42, "rep": 0, "runner_source_sha256": replay.RUNNER_SHA256,
            "request_error": "", "finish_reason": "length" if number == 0 else "stop",
            "prompt": f"prompt-{number}", "response": f"patch-{number}", "reasoning": "",
        }
        for field in ("prompt", "response", "reasoning"):
            row[f"{field}_fingerprint"] = replay.text_fingerprint(row[field])
        rows.append(row)
    _write(tmp_path / paths["capture"], "".join(json.dumps(row) + "\n" for row in rows))
    _write(tmp_path / paths["capture_summary"], json.dumps({"meta": {
        "arm": replay.CAPTURE_ARM, "enable_thinking": False, "n_per_suite": 40, "max_tokens": 3072,
        "seed": 42, "runner_source_sha256": replay.RUNNER_SHA256,
    }}))
    _write(tmp_path / paths["capture_argv"], f"runner --no-enable-thinking --arm {replay.CAPTURE_ARM} --n 40 --max-tokens 3072")
    _write(tmp_path / paths["capture_done"], "DONE\n")
    _write(tmp_path / paths["frozen_table"], json.dumps({"status": "FINAL"}))
    _write(tmp_path / paths["authority_finalization"], "sealed\n")
    _write(tmp_path / paths["authority_ids"], "".join(json.dumps({"id": instance_id}) + "\n" for instance_id in ids))
    _write(tmp_path / paths["harness"], "# pinned harness\n")
    _write(tmp_path / paths["dataset"], json.dumps([{"instance_id": instance_id, "repo": "repo/project", "base_commit": "deadbeef"} for instance_id in ids]))
    converter = """import json\nfrom pathlib import Path\nrows = {row['instance_id']: row for row in json.loads((Path(__file__).parent / 'swebench_verified.json').read_text())}\ndef apply_blocks(_row, response, blocks):\n    blocks.append({'block_index': 0, 'outcome': 'applied', 'search_sha256': 'a' * 64, 'replace_sha256': 'b' * 64})\n    return response, 1, 0\ndef row_diagnostic(row, patch, blocks, _runner):\n    return {'instance_id': row['id'], 'finish_reason': row['finish_reason'], 'empty_patch': not bool(patch), 'blocks': blocks, 'scoring_eligible': True}\n"""
    _write(tmp_path / paths["converter"], converter)
    expected = {key: (relative, _sha(tmp_path / relative)) for key, relative in paths.items()}
    monkeypatch.setattr(replay, "EXPECTED", expected)
    monkeypatch.setattr(replay, "CANONICAL_REPOS_REL", "mirror")
    monkeypatch.setattr(replay, "installed_harness", lambda: tmp_path / "authority/harness.py")
    monkeypatch.setattr(replay, "validate_repo_mirror", lambda _inputs, _ids: None)
    return replay.Inputs(tmp_path), ids


def test_preflight_binds_only_complete_nothink_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, _ids = make_inputs(tmp_path, monkeypatch)
    replay.preflight(inputs)
    rows = [json.loads(line) for line in inputs.path("capture").read_text().splitlines()]
    rows[0]["arm"] = "A3-tc-quality__thinkingcap"
    inputs.path("capture").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(replay.ReplayError, match="pinned input drifted"):
        replay.preflight(inputs)


def test_execute_seals_full_fingerprints_nonrecovery_report_and_immutable_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, ids = make_inputs(tmp_path, monkeypatch)

    def fake_official(arm: Path, expected_ids: list[str], _run_id: str) -> None:
        predictions = json.loads((arm / "predictions.sealed.json").read_text())
        empty = [row["instance_id"] for row in predictions if not row["model_patch"]]
        completed = [item for item in expected_ids if item not in empty]
        report = {
            "submitted_ids": expected_ids, "submitted_instances": 40,
            "completed_ids": completed, "completed_instances": len(completed),
            "empty_patch_ids": empty, "empty_patch_instances": len(empty),
            "resolved_ids": completed[:2], "resolved_instances": 2,
            "unresolved_ids": completed[2:], "unresolved_instances": len(completed) - 2,
            "error_ids": [], "error_instances": 0,
        }
        report_dir = arm / "report"
        report_dir.mkdir()
        (report_dir / "official.json").write_text(json.dumps(report))

    monkeypatch.setattr(replay, "run_official", fake_official)
    output = tmp_path / "published"
    replay.execute(inputs, output)
    assert json.loads((output / "state.json").read_text())["status"] == "FINALIZED"
    assert len((output / "finalization.sha256").read_text().splitlines()) >= 10
    result = json.loads((output / "result.json").read_text())
    assert result["denominator"] == 40
    assert result["empty_patch_failures"] == 1
    ledger = json.loads((output / "A3-tc-nothink/nonrecovery_ledger.sealed.json").read_text())
    assert ledger["aggregate"]["empty_patch_row_count"] == 1
    manifest = json.loads((output / "A3-tc-nothink/manifest.json").read_text())
    assert manifest["sealer"]["sealed_path"] == "../replay_tc_nothink_v4.sealed.py"
    assert result["sealer"]["sealed_path"] == "replay_tc_nothink_v4.sealed.py"
    sealer = output / "replay_tc_nothink_v4.sealed.py"
    sealer.write_text(sealer.read_text() + "# tampered\n")
    with pytest.raises(replay.ReplayError, match="sealed sealer source drifted"):
        replay.revalidate_sealed_arm(output, output / "A3-tc-nothink", ids)
    assert not list(tmp_path.glob(".published.publish.lock"))
    with pytest.raises(replay.ReplayError, match="immutable successor"):
        replay.execute(inputs, output)


def test_report_validation_rejects_empty_patch_partition_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, ids = make_inputs(tmp_path, monkeypatch)
    arm = tmp_path / "arm"
    arm.mkdir()
    (arm / "predictions.sealed.json").write_text(json.dumps([
        {"instance_id": instance_id, "model_patch": "" if index == 0 else "patch"} for index, instance_id in enumerate(ids)
    ]))
    report = {
        "submitted_ids": ids, "submitted_instances": 40, "completed_ids": ids, "completed_instances": 40,
        "empty_patch_ids": [], "empty_patch_instances": 0, "resolved_ids": ids, "resolved_instances": 40,
        "unresolved_ids": [], "unresolved_instances": 0, "error_ids": [], "error_instances": 0,
    }
    report_dir = arm / "report"
    report_dir.mkdir()
    (report_dir / "official.json").write_text(json.dumps(report))
    with pytest.raises(replay.ReplayError, match="partition is invalid"):
        replay.validate_report(arm, ids)
