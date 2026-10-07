from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.harness_p0.hs19d_p0_summary import (
    ROOT_VERDICT,
    ROOT_VERDICT_BLOB,
    ROOT_VERDICT_SHA256,
    load_pinned_verdict,
    summarize_verdict,
)


def _verdict(detail: str, *, passed: bool = True, check_ok: bool = True) -> dict:
    return {
        "pass": passed,
        "parent_session_id": "private-parent-id",
        "child_session_ids": ["private-child-id"],
        "checks": [{"check": "S1-one-linked-child", "ok": check_ok, "detail": detail}],
    }


def test_summary_projects_counts_without_exporting_private_detail_or_claims():
    detail = (
        "parent task parts: 1 (1 completed); child ids: ['private-child-id']; "
        "child export present; child task calls: 0"
    )
    report = summarize_verdict(
        _verdict(detail),
        {"path": ROOT_VERDICT, "git_blob": ROOT_VERDICT_BLOB,
         "sha256": ROOT_VERDICT_SHA256, "checkout_commit": "pinned-root"},
    )
    encoded = json.dumps(report)
    assert report["records_in_scope"] == 1
    assert report["counts"]["parent_task_parts"]["value"] == 1
    assert report["counts"]["child_task_calls"]["value"] == 0
    assert "private-parent-id" not in encoded and "private-child-id" not in encoded
    assert "tool-call correctness" in report["interpretation"]
    for unknown in (
        "emitted_tool_calls", "tap_emitted_call_retention",
        "served_template_parallel_tool_support", "child_call_to_tool_join",
    ):
        assert report["counts"][unknown] == {"status": "unknown", "value": None, "observed_from": None}


def test_missing_count_is_unknown_not_zero_and_failed_summary_is_not_promoted():
    source = {"path": ROOT_VERDICT, "git_blob": ROOT_VERDICT_BLOB,
              "sha256": ROOT_VERDICT_SHA256, "checkout_commit": None}
    partial = summarize_verdict(_verdict("parent task parts: 1 only"), source)
    assert partial["counts"]["parent_task_parts"]["value"] == 1
    assert partial["counts"]["child_task_calls"]["status"] == "unknown"
    failed = summarize_verdict(_verdict("parent task parts: 1; child task calls: 0", passed=False), source)
    assert failed["summary_record_verified"] is False
    assert failed["counts"]["parent_task_parts"]["status"] == "unknown"
    assert failed["counts"]["child_task_calls"]["status"] == "unknown"
    ambiguous = summarize_verdict(
        _verdict("parent task parts: 1; parent task parts: 2; child task calls: 0"), source
    )
    assert ambiguous["counts"]["parent_task_parts"]["status"] == "unknown"
    duplicate = _verdict("parent task parts: 1; child task calls: 0")
    duplicate["checks"].append(duplicate["checks"][0])
    rejected = summarize_verdict(duplicate, source)
    assert rejected["summary_record_verified"] is False
    assert rejected["counts"]["child_task_calls"]["status"] == "unknown"


def test_pinned_source_loader_refuses_byte_drift_and_symlink(tmp_path: Path, monkeypatch):
    import subprocess

    from scripts.harness_p0 import hs19d_p0_summary as module

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    artifact = tmp_path / ROOT_VERDICT
    artifact.parent.mkdir(parents=True)
    data = b'{"pass": true, "checks": []}\n'
    artifact.write_bytes(data)
    subprocess.run(["git", "-C", str(tmp_path), "add", ROOT_VERDICT], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=fixture", "-c",
        "user.email=fixture@example.invalid", "commit", "-qm", "fixture",
    ], check=True)
    blob = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", f"HEAD:{ROOT_VERDICT}"], text=True
    ).strip()
    monkeypatch.setattr(module, "ROOT_VERDICT_BLOB", blob)
    monkeypatch.setattr(module, "ROOT_VERDICT_SHA256", hashlib.sha256(data).hexdigest())

    with pytest.raises(ValueError, match="fixed accepted path"):
        load_pinned_verdict(tmp_path, "../outside.json")
    loaded, source = load_pinned_verdict(tmp_path)
    assert loaded["pass"] is True
    assert source["git_blob"] == blob
    hardlink = tmp_path / "linked-verdict"
    os.link(artifact, hardlink)
    with pytest.raises(ValueError, match="single-link"):
        load_pinned_verdict(tmp_path)
    hardlink.unlink()
    artifact.unlink()
    os.mkfifo(artifact)
    with pytest.raises(ValueError, match="single-link regular file"):
        load_pinned_verdict(tmp_path)
    artifact.unlink()
    artifact.write_bytes(data + b" ")
    with pytest.raises(ValueError, match="pinned source identity"):
        load_pinned_verdict(tmp_path)
    artifact.unlink()
    artifact.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        load_pinned_verdict(tmp_path)


def test_actual_pinned_root_verdict_projects_only_sanitized_counts():
    root_value = os.environ.get("HS19D_ROOT_CHECKOUT")
    if not root_value:
        pytest.skip("native P0 capture supplies the separately pinned ROOT checkout")
    root = Path(root_value)
    verdict, source = load_pinned_verdict(root)
    report = summarize_verdict(verdict, source)
    rendered = json.dumps(report)
    private_ids = [verdict.get("parent_session_id"), *verdict.get("child_session_ids", [])]
    assert source["git_blob"] == ROOT_VERDICT_BLOB
    assert source["sha256"] == ROOT_VERDICT_SHA256
    assert report["summary_record_verified"] is True
    assert report["counts"]["parent_task_parts"]["value"] == 1
    assert report["counts"]["child_task_calls"]["value"] == 0
    assert all(identifier not in rendered for identifier in private_ids if identifier)
    assert report["counts"]["emitted_tool_calls"]["value"] is None
    assert report["counts"]["served_template_parallel_tool_support"]["value"] is None
