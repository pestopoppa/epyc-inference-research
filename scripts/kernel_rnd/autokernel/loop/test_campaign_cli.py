"""Hermetic tests for the offline-only campaign consumer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from . import campaign, campaign_cli
from .test_campaign import _manifest, _target


def _file_artifact(tmp_path, kind: str, ref: str, content: bytes) -> dict:
    path = tmp_path / f"{kind}-{ref}"
    path.write_bytes(content)
    return {"schema": campaign.ARTIFACT_SCHEMA, "kind": kind, "ref": ref,
            "path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def _snapshot(tmp_path, *, include_second_model: bool = False) -> dict:
    model = {"model-a": _file_artifact(tmp_path, "model", "model-a", b"model-a")}
    if include_second_model:
        model["model-b"] = _file_artifact(tmp_path, "model", "model-b", b"model-b")
    return {
        "schema": campaign_cli.REGISTRY_SNAPSHOT_SCHEMA,
        "artifacts": {
            "source": {
                "ef81196d5": _file_artifact(tmp_path, "source", "ef81196d5", b"kernel"),
                "1d9733f1": _file_artifact(tmp_path, "source", "1d9733f1", b"recipes"),
            },
            "model": model,
            "build": {
                "build-a": _file_artifact(tmp_path, "build", "build-a", b"build"),
                "production-v9": _file_artifact(
                    tmp_path, "build", "production-v9", b"baseline"),
            },
            "recipe": {
                "recipe-a": _file_artifact(tmp_path, "recipe", "recipe-a", b"recipe")},
        },
    }


def _inputs(tmp_path, manifest: dict, snapshot: dict):
    manifest_path = tmp_path / "campaign.json"
    snapshot_path = tmp_path / "registry.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    return manifest_path, snapshot_path


def test_stdout_resolution_is_explicitly_unverified_and_never_admission_ready(
        tmp_path, capsys):
    paths = _inputs(tmp_path, _manifest(production=[_target("prod")]), _snapshot(tmp_path))
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1])]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["schema"] == campaign_cli.DRY_RESOLUTION_SCHEMA
    assert body["admission_ready"] is False
    assert body["disposition"] == "resolved_unverified"
    assert body["verification"] == {
        "requested": False, "status": "not_requested",
        "checked_artifacts": 0, "errors": []}


def test_out_uses_durable_json_publication_and_can_be_reused_as_previous(tmp_path):
    paths = _inputs(tmp_path, _manifest(production=[_target("prod")]), _snapshot(tmp_path))
    output = tmp_path / "nested" / "resolution.json"
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1]),
        "--out", str(output)]) == 0
    previous = campaign_cli.load_previous(output)
    assert previous.request_id == "request-1"
    assert not list(output.parent.glob(".campaign-*"))


@pytest.mark.parametrize("malformation", [
    {"artifacts": {}},
    {"schema": "legacy", "artifacts": {}},
    {"schema": campaign_cli.REGISTRY_SNAPSHOT_SCHEMA, "artifacts": {}, "extra": 1},
])
def test_malformed_or_legacy_snapshot_returns_two_with_explanation(
        tmp_path, capsys, malformation):
    paths = _inputs(tmp_path, _manifest(production=[_target("prod")]), malformation)
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1])]) == 2
    assert "refused" in capsys.readouterr().err


def test_malformed_manifest_returns_two_with_explanation(tmp_path, capsys):
    paths = _inputs(tmp_path, {"schema": "legacy"}, _snapshot(tmp_path))
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1])]) == 2
    assert "manifest" in capsys.readouterr().err


def test_missing_target_is_partial_json_and_does_not_block_ready_peer(tmp_path, capsys):
    raw = _manifest(production=[_target("ready"), _target("missing", model="absent")])
    paths = _inputs(tmp_path, raw, _snapshot(tmp_path))
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1])]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["disposition"] == "partial"
    assert body["summary"]["resolution_dispositions"] == {
        "missing_artifact": 1, "ready": 1}


def test_explicit_verification_accepts_small_regular_files(tmp_path, capsys):
    paths = _inputs(tmp_path, _manifest(production=[_target("prod")]), _snapshot(tmp_path))
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1]),
        "--verify-artifacts"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["disposition"] == "verified_resolution"
    assert body["verification"]["status"] == "passed"
    assert body["verification"]["checked_artifacts"] == 6
    assert body["admission_ready"] is False


def test_changed_artifact_refuses_only_its_target_and_reports_error_separately(
        tmp_path, capsys):
    snapshot = _snapshot(tmp_path, include_second_model=True)
    raw = _manifest(production=[_target("bad", model="model-a"),
                                _target("peer", model="model-b", context=8192)])
    paths = _inputs(tmp_path, raw, snapshot)
    model_path = snapshot["artifacts"]["model"]["model-a"]["path"]
    with open(model_path, "wb") as stream:
        stream.write(b"changed-after-snapshot")
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1]),
        "--verify-artifacts"]) == 0
    body = json.loads(capsys.readouterr().out)
    by_id = {row["target_ids"][0]: row for row in body["target_dispositions"]}
    assert by_id["bad"]["verification_status"] == "failed"
    assert by_id["peer"]["verification_status"] == "passed"
    assert body["verification"]["errors"][0]["reason"] == "sha256_mismatch"
    assert all(target["status"] == "ready"
               for target in body["resolved_campaign"]["targets"])


def test_missing_artifact_file_fails_verification_without_erasing_resolution(
        tmp_path, capsys):
    snapshot = _snapshot(tmp_path)
    paths = _inputs(tmp_path, _manifest(production=[_target("prod")]), snapshot)
    model_path = snapshot["artifacts"]["model"]["model-a"]["path"]
    Path(model_path).unlink()
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1]),
        "--verify-artifacts"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["verification"]["errors"][0]["reason"] == "missing"
    assert body["resolved_campaign"]["targets"][0]["status"] == "ready"
    assert body["target_dispositions"][0]["verification_status"] == "failed"


def test_previous_identity_is_not_replaced_by_a_moved_registry_during_verification(
        tmp_path, capsys):
    snapshot = _snapshot(tmp_path)
    paths = _inputs(tmp_path, _manifest(production=[_target("prod")]), snapshot)
    out = tmp_path / "first.json"
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1]),
        "--out", str(out)]) == 0
    first = json.loads(out.read_text())
    pinned = first["resolved_campaign"]["targets"][0]["execution"]["model"]
    with open(pinned["path"], "wb") as stream:
        stream.write(b"changed")
    snapshot["artifacts"]["model"]["model-a"] = _file_artifact(
        tmp_path, "model", "replacement", b"replacement") | {"ref": "model-a"}
    paths[1].write_text(json.dumps(snapshot), encoding="utf-8")
    assert campaign_cli.main([
        "--manifest", str(paths[0]), "--registry-snapshot", str(paths[1]),
        "--previous", str(out), "--verify-artifacts"]) == 0
    replay = json.loads(capsys.readouterr().out)
    replay_model = replay["resolved_campaign"]["targets"][0]["execution"]["model"]
    assert replay_model == pinned
    assert replay["verification"]["errors"][0]["path"] == pinned["path"]


def test_legacy_previous_resolution_is_rejected(tmp_path):
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"schema": "legacy"}), encoding="utf-8")
    with pytest.raises(campaign.ManifestError, match="unsupported schema"):
        campaign_cli.load_previous(previous)
