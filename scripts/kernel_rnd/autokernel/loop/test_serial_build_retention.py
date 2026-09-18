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
            tmp_path, current, free_bytes=0, trigger_free_bytes=1,
            target_free_bytes=10**9, recent_state_caches=1, max_build_dirs=4)
    finally:
        os.close(descriptor)
    assert [Path(row["state_root"]).name for row in planned["selected"]] == ["old"]
    assert planned["protected_recent"] == [str(recent / "targets/target/builds")]
    assert "locked" not in json.dumps(planned)


def test_execute_revalidates_then_removes_only_generated_build_root(tmp_path):
    source, commit = _git(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    old = _state(tmp_path, "old", source, commit, 10)
    build = old / "targets/target/builds"
    receipt = old / "batches/batch-000000/measurement.json"
    planned = retention.plan(
        tmp_path, current, free_bytes=0, trigger_free_bytes=1,
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
        tmp_path, current, free_bytes=0, trigger_free_bytes=1,
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
        tmp_path, current, free_bytes=0, trigger_free_bytes=1,
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
        tmp_path, current, free_bytes=0, trigger_free_bytes=1,
        target_free_bytes=10**9, recent_state_caches=0, max_build_dirs=1)
    assert retry["selected"][0]["build_root"] == str(build)
    assert retention.execute(retry)["removed"][0]["build_root"] == str(build)
    assert not build.exists()
    assert not (old / retention.RETRY_FILENAME).exists()
