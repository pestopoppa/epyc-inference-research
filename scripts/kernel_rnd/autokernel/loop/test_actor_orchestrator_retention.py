"""OAB-17-R4: sidecar retention and the `role` in the orchestrator bundle seal."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from autokernel.loop import actor_context, actor_orchestrator as orch, actors

OLD = 1_000_000_000.0  # an mtime far older than any `days` cutoff


def _name(i: int) -> str:
    return f"20260101T0000{i:02d}-{i:012x}.provenance.json"


def _make(d: Path, i: int, mtime: float) -> Path:
    p = d / _name(i)
    p.write_text("{}", encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    orch._PENDING.clear()
    monkeypatch.delenv(orch.SIDECAR_KEEP_ENV, raising=False)
    monkeypatch.delenv(orch.SIDECAR_DAYS_ENV, raising=False)
    yield
    orch._PENDING.clear()


def test_keeps_exactly_the_newest_n_that_are_also_old(tmp_path):
    files = [_make(tmp_path, i, OLD + i) for i in range(10)]
    removed = orch.sweep_sidecars(tmp_path, keep=4, days=14)
    assert sorted(removed) == sorted(f.name for f in files[:6])
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f.name for f in files[6:])


def test_recent_files_survive_beyond_n(tmp_path):
    now = 2_000_000_000.0
    old = [_make(tmp_path, i, OLD + i) for i in range(3)]
    fresh = [_make(tmp_path, i, now - 3600 + i) for i in range(3, 6)]
    removed = orch.sweep_sidecars(tmp_path, keep=1, days=14, now=now)
    # keep=1 retains only fresh[-1] by count; the other fresh ones by age; old ones go.
    assert sorted(removed) == sorted(f.name for f in old)
    assert all(f.exists() for f in fresh)


def test_days_zero_is_count_only(tmp_path):
    now = 2_000_000_000.0
    files = [_make(tmp_path, i, now + i) for i in range(5)]
    removed = orch.sweep_sidecars(tmp_path, keep=2, days=0, now=now + 10)
    assert sorted(removed) == sorted(f.name for f in files[:3])


def test_never_touches_foreign_files_dirs_or_symlinks(tmp_path):
    target = tmp_path.parent / (tmp_path.name + "-target.provenance.json")
    target.write_text("keep me", encoding="utf-8")
    link = tmp_path / _name(50)  # a matching NAME that is a symlink
    link.symlink_to(target)
    foreign = [tmp_path / "notes.txt", tmp_path / "x.schema.json",
               tmp_path / "20260101T000000-abc.provenance.json",     # not 12 hex
               tmp_path / (_name(60) + ".bak")]
    for f in foreign:
        f.write_text("x", encoding="utf-8")
        os.utime(f, (OLD, OLD))
    subdir = tmp_path / _name(70)  # a matching NAME that is a directory
    subdir.mkdir()
    (subdir / "inner.txt").write_text("x", encoding="utf-8")
    own = _make(tmp_path, 1, OLD)
    removed = orch.sweep_sidecars(tmp_path, keep=0, days=0)
    assert removed == [own.name] and not own.exists()
    assert link.is_symlink() and target.read_text(encoding="utf-8") == "keep me"
    assert all(f.exists() for f in foreign)
    assert subdir.is_dir() and (subdir / "inner.txt").exists()


def test_a_pending_provenance_path_is_never_swept(tmp_path):
    pending = _make(tmp_path, 1, OLD)
    orch._PENDING["somewhere"] = pending
    gone = _make(tmp_path, 2, OLD)
    assert orch.sweep_sidecars(tmp_path, keep=0, days=0) == [gone.name]
    assert pending.exists()


def test_missing_dir_is_quiet(tmp_path):
    assert orch.sweep_sidecars(tmp_path / "absent") == []


def test_env_configures_the_policy_and_bad_values_use_defaults(tmp_path, monkeypatch):
    for i in range(5):
        _make(tmp_path, i, OLD + i)
    monkeypatch.setenv(orch.SIDECAR_KEEP_ENV, "2")
    assert len(orch.sweep_sidecars(tmp_path)) == 3
    monkeypatch.setenv(orch.SIDECAR_KEEP_ENV, "junk")
    monkeypatch.setenv(orch.SIDECAR_DAYS_ENV, "-3")
    assert orch.sweep_sidecars(tmp_path) == []  # defaults: 200 / 14


def test_argv_sweeps_before_allocating_its_own_sidecar(tmp_path, monkeypatch):
    ws = tmp_path / "workers" / "lane0"
    ws.mkdir(parents=True)
    call_dir = ws.parent / orch.SIDECAR_DIR
    call_dir.mkdir()
    stale = [_make(call_dir, i, OLD + i) for i in range(3)]
    monkeypatch.setenv(orch.SIDECAR_KEEP_ENV, "1")
    backend = actors.backend_for("orch:auto", "high")
    argv = backend.argv("P", ws, read_only=True)
    mine = Path(argv[argv.index("--provenance-out") + 1])
    assert not stale[0].exists() and not stale[1].exists() and stale[2].exists()
    assert mine.parent == call_dir  # its own (not yet written) path is untouched


def test_the_orchestrator_seal_carries_the_role(tmp_path):
    backend = actors.backend_for("orch:auto", "high")
    planner = actors.AgentPlanner(workspace=tmp_path, backend=backend)
    for role in ("planner", "author"):
        bundle = actor_context.orchestrator_bundle("## Already tried\nx\n", role=role)
        planner._sealed("PROMPT", bundle, None, role)
        assert bundle.payload["manifest"]["role"] == role
    orch._STAGED.clear()
