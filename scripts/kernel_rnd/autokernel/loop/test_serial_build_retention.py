import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

from . import serial_build_retention as retention


def _git(tmp_path: Path) -> tuple[Path, str]:
    tree = tmp_path / "source"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-C", str(tree), "config", "user.name", "fixture"], check=True)
    subprocess.run(["git", "-C", str(tree), "config", "user.email", "fixture@example.invalid"], check=True)
    (tree / "source.c").write_text("int x;\n")
    subprocess.run(["git", "-C", str(tree), "add", "source.c"], check=True)
    subprocess.run(["git", "-C", str(tree), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    return tree, commit


def _state(parent: Path, name: str, source: Path, commit: str, stamp: int) -> Path:
    root = parent / name
    batch = root / "batches" / "batch-000000"
    build = root / "targets" / "target" / "builds"
    lane = build / "lane0"
    batch.mkdir(parents=True)
    lane.mkdir(parents=True)
    (lane / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n")
    (lane / "object.o").write_bytes(b"generated" * 100)
    (batch / "measurement.json").write_text('{"unique":"receipt"}\n')
    continuation = {
        "input_argv": ["--target-id", "target", "--worktree", str(source),
                       "--worker-build-root", str(build), "--out", str(batch)],
        "current_anchor": {"path": str(root / "store" / "anchor-gen-001"),
                           "commit": commit},
        "cor_anchor": None,
    }
    raw = json.dumps(continuation).encode()
    result = batch / "loop-continuation.json"
    result.write_bytes(raw)
    state = {"schema": "epyc.autokernel.serial_run.v1", "active": None,
             "config_digest": "f" * 64,
             "last_results": {"0": {"path": str(result),
                                      "sha256": hashlib.sha256(raw).hexdigest()}}}
    (root / "serial.lock").touch()
    state_path = root / "serial-state.json"
    state_path.write_text(json.dumps(state))
    os.utime(state_path, ns=(stamp, stamp))
    return root


def test_pressure_plan_preserves_current_recent_and_locked_states(tmp_path):
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    _state(tmp_path, "old", source, commit, 10)
    locked = _state(tmp_path, "locked", source, commit, 20)
    recent = _state(tmp_path, "recent", source, commit, 30)
    descriptor = os.open(locked / "serial.lock", os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        planned = retention.plan(
            tmp_path, current, free_bytes=0, trigger_free_bytes=1, hard_floor_free_bytes=0,
            target_free_bytes=10**9, recent_state_caches=1, max_build_dirs=4)
    finally:
        os.close(descriptor)
    assert [Path(row["state_root"]).name for row in planned["selected"]] == ["old"]
    assert planned["protected_recent"] == [str(recent / "targets/target/builds")]
    assert "locked" not in json.dumps(
        [planned["selected"], planned["protected_recent"]])
    assert {"state_root": str(locked),
            "reason": "locked_active_or_not_serial_state"} in planned["skipped"]


def test_execute_revalidates_then_removes_only_generated_build_root(tmp_path):
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    old = _state(tmp_path, "old", source, commit, 10)
    build = old / "targets/target/builds"
    receipt = old / "batches/batch-000000/measurement.json"
    planned = retention.plan(
        tmp_path, current, free_bytes=0, trigger_free_bytes=1, hard_floor_free_bytes=0,
        target_free_bytes=10**9, recent_state_caches=0, max_build_dirs=1)
    preview = retention.execute(planned, dry_run=True)
    assert build.is_dir() and preview["removed"] == []
    result = retention.execute(planned)
    assert not build.exists()
    assert receipt.read_text() == '{"unique":"receipt"}\n'
    assert source.is_dir()
    assert result["reclaimed_bytes"] > 0


def test_unreconciled_active_state_is_never_a_candidate(tmp_path):
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    root = _state(tmp_path, "unreconciled", source, commit, 10)
    state_path = root / "serial-state.json"
    state = json.loads(state_path.read_text())
    state["active"] = {"pid": 999999, "batch_dir": "unreconciled"}
    state_path.write_text(json.dumps(state))
    planned = retention.plan(
        tmp_path, current, free_bytes=0, trigger_free_bytes=1, hard_floor_free_bytes=0,
        target_free_bytes=10**9, recent_state_caches=0, max_build_dirs=4)
    assert planned["selected"] == []
    assert (root / "targets/target/builds").is_dir()


def test_failed_quarantine_removal_restores_retryable_build_root(tmp_path, monkeypatch):
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    old = _state(tmp_path, "old", source, commit, 10)
    build = old / "targets/target/builds"
    planned = retention.plan(
        tmp_path, current, free_bytes=0, trigger_free_bytes=1, hard_floor_free_bytes=0,
        target_free_bytes=10**9, recent_state_caches=0, max_build_dirs=1)
    real_rmtree = retention.shutil.rmtree
    def partial_failure(path):
        (path / "lane0/CMakeCache.txt").unlink()
        raise OSError("injected")
    monkeypatch.setattr(retention.shutil, "rmtree", partial_failure)
    failed = retention.execute(planned)
    assert failed["removed"] == []
    assert failed["skipped"][0]["reason"].startswith("OSError: injected")
    assert build.is_dir()
    assert not list(build.parent.glob(".builds.retention-*"))
    assert json.loads((old / retention.RETRY_FILENAME).read_text()) == {
        "schema": retention.RETRY_SCHEMA, "build_root": str(build),
        "source_commit": commit, "recipe_digest": planned["selected"][0]["recipe_digest"]}

    monkeypatch.setattr(retention.shutil, "rmtree", real_rmtree)
    retry = retention.plan(
        tmp_path, current, free_bytes=0, trigger_free_bytes=1, hard_floor_free_bytes=0,
        target_free_bytes=10**9, recent_state_caches=0, max_build_dirs=1)
    assert retry["selected"][0]["build_root"] == str(build)
    assert retention.execute(retry)["removed"][0]["build_root"] == str(build)
    assert not build.exists()
    assert not (old / retention.RETRY_FILENAME).exists()


GIB = 1024 ** 3


def _disk(monkeypatch, gib):
    monkeypatch.setattr(retention.shutil, "disk_usage",
                        lambda _path: type("Usage", (), {"free": int(gib * GIB)})())


def _policy(**overrides):
    return {"trigger_free_bytes": 250 * GIB, "target_free_bytes": 320 * GIB,
            "hard_floor_free_bytes": 100 * GIB, "recent_state_caches": 0,
            "max_build_dirs": 4} | overrides


def test_defaults_are_reachable_on_the_production_host():
    assert retention.DEFAULT_HARD_FLOOR_FREE_BYTES == 100 * GIB
    assert retention.DEFAULT_TRIGGER_FREE_BYTES == 250 * GIB
    assert retention.DEFAULT_TARGET_FREE_BYTES == 320 * GIB


def test_above_trigger_is_a_logged_noop(tmp_path, monkeypatch):
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    old = _state(tmp_path, "old", source, commit, 10)
    _disk(monkeypatch, 300)
    planned = retention.plan(tmp_path, current, **_policy())
    assert planned["selected"] == []
    assert {"state_root": str(old), "build_root": str(old / "targets/target/builds"),
            "reason": "free_at_or_above_trigger"}.items() <= next(
        row for row in planned["skipped"] if row["state_root"] == str(old)).items()
    result = retention.execute(planned)
    decision = retention.decide(planned, result)
    assert decision == decision | {"verdict": "noop_above_trigger", "refuse": False}
    record = retention.append_decision(current, planned, result, decision)
    assert (old / "targets/target/builds").is_dir()
    [line] = (current / retention.DECISION_LOG).read_text().splitlines()
    assert json.loads(line) == record and record["reclaimed_bytes"] == 0


def test_below_trigger_reclaims_and_logs(tmp_path, monkeypatch):
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    old = _state(tmp_path, "old", source, commit, 10)
    _disk(monkeypatch, 200)
    planned = retention.plan(tmp_path, current, **_policy(target_free_bytes=200 * GIB + 1,
                                                          trigger_free_bytes=200 * GIB + 1))
    assert [row["state_root"] for row in planned["selected"]] == [str(old)]
    result = retention.execute(planned)
    assert not (old / "targets/target/builds").exists()
    assert result["reclaimed_bytes"] > 0
    _disk(monkeypatch, 201)  # disk_usage after the reclaim
    result["free_bytes_after"] = retention.shutil.disk_usage(tmp_path).free
    decision = retention.decide(planned, result)
    assert decision["verdict"] == "reclaimed_to_target" and not decision["refuse"]
    record = retention.append_decision(current, planned, result, decision)
    assert record["removed"] == [{"build_root": str(old / "targets/target/builds"),
                                  "bytes": planned["selected"][0]["bytes"]}]
    assert record["free_bytes_before"] == 200 * GIB


def test_unreachable_target_above_floor_warns_and_continues(tmp_path, monkeypatch):
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    old = _state(tmp_path, "old", source, commit, 10)
    _disk(monkeypatch, 150)
    planned = retention.plan(tmp_path, current, **_policy())
    result = retention.execute(planned)
    assert result["removed"][0]["build_root"] == str(old / "targets/target/builds")
    decision = retention.decide(planned, result)
    assert decision["verdict"] == "target_unreachable_continuing"
    assert decision["refuse"] is False
    assert decision["message"].startswith("WARNING:")
    assert "150.0 -> 150.0 GiB" in decision["message"]


def test_below_hard_floor_refuses(tmp_path, monkeypatch):
    current = tmp_path / "current"
    current.mkdir()
    _disk(monkeypatch, 50)
    planned = retention.plan(tmp_path, current, **_policy())
    result = retention.execute(planned)
    decision = retention.decide(planned, result)
    assert decision["verdict"] == "refused_below_hard_floor" and decision["refuse"]
    assert "hard safety floor" in decision["message"]


def test_live_build_root_of_the_launch_is_never_selected(tmp_path):
    # A relaunch may point --target-root into a retired sibling state; that
    # sibling's build root is the one the new run builds in.
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    old = _state(tmp_path, "old", source, commit, 10)
    planned = retention.plan(tmp_path, current, free_bytes=0,
                             protected_paths=[old / "targets"],
                             **_policy(recent_state_caches=0))
    assert planned["selected"] == []
    assert planned["protected_live"] == [str((old / "targets").resolve())]
    assert [row["reason"] for row in planned["skipped"]
            if row["state_root"] == str(old)] == ["protected_live_build_root"]


def test_floor_above_trigger_is_an_invalid_policy(tmp_path):
    import pytest
    with pytest.raises(retention.BuildRetentionRefused):
        retention.plan(tmp_path, tmp_path / "current",
                       **_policy(hard_floor_free_bytes=300 * GIB))
