"""Private create-once snapshot controls for the two analysis report producers."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import eval_suite_discriminability as eval_report  # noqa: E402
import mf_vbs1_verify_before_stop as mf_report  # noqa: E402


def _entry(raw: bytes = b'{"fixture":true}\n') -> dict:
    return {"path": "data/fixture.jsonl", "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw), "row_count": 1,
            "normalized_row_count": 1, "_raw_bytes": raw}


def _seal_eval(report: dict, inputs: list[dict], path: Path) -> dict:
    return eval_report._seal_report(report, inputs, path)


def _seal_mf(report: dict, inputs: list[dict], path: Path) -> dict:
    return mf_report._seal_report(report, str(REPO), inputs, str(path))


@pytest.mark.parametrize("seal", [_seal_eval, _seal_mf])
def test_snapshots_are_private_and_identical_paths_deduplicate(tmp_path, seal):
    report_path = tmp_path / "report.json"
    sealed = seal({"config": {}}, [_entry(), _entry()], report_path)
    provenance = sealed["native_provenance"]
    assert len(provenance["inputs"]) == 1
    snapshot_root = report_path.with_name("report.json.native")
    snapshot_dir = Path(provenance["inputs"][0]["snapshot_path"]).parent
    paths = [snapshot_root, snapshot_dir, *snapshot_dir.iterdir()]
    assert paths[0].stat().st_mode & 0o777 == 0o700
    assert paths[1].stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths[2:])


@pytest.mark.parametrize("seal", [_seal_eval, _seal_mf])
def test_conflicting_duplicate_input_path_refuses_without_overwriting(tmp_path, seal):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="conflicting bytes or counts"):
        seal({"config": {}}, [_entry(), _entry(b'{"fixture":false}\n')], path)
    assert not path.with_name("report.json.native").exists()


@pytest.mark.parametrize("seal", [_seal_eval, _seal_mf])
def test_existing_snapshot_mismatch_refuses_instead_of_replacing(tmp_path, seal):
    path = tmp_path / "report.json"
    first = seal({"config": {}}, [_entry()], path)
    snapshot = Path(first["native_provenance"]["inputs"][0]["snapshot_path"])
    snapshot.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="different or non-regular"):
        seal({"config": {}}, [_entry()], path)
    assert snapshot.read_bytes() == b"tampered"


@pytest.mark.parametrize("seal", [_seal_eval, _seal_mf])
def test_symlinked_snapshot_root_is_refused_without_chmod_target(tmp_path, seal):
    path = tmp_path / "report.json"
    target = tmp_path / "unrelated"
    target.mkdir(mode=0o755)
    original_mode = target.stat().st_mode & 0o777
    path.with_name("report.json.native").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="not a real directory"):
        seal({"config": {}}, [_entry()], path)
    assert target.stat().st_mode & 0o777 == original_mode
