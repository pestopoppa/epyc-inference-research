"""Exercise the actual run.py publisher without actors, inference or a live store."""

from __future__ import annotations

import ast
from collections.abc import Mapping
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from . import roofline_coverage as rc, status


def _outcomes(progress=False):
    return [
        SimpleNamespace(
            to_attempt=lambda value=(i + 1 if progress else 0): {
                "effect_fraction": value,
                "status": "kept" if progress else "measured_null",
            }
        )
        for i in range(8)
    ]


def _native_publisher(root):
    """Compile the unchanged native closure body; replace only its enclosing run inputs."""
    source = Path(rc.__file__).with_name("run.py")
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "publish")
    gap = {"triggered": True, "uncovered_nodes": [{"file": "repack.cpp"}]}
    env = {
        "__name__": "autokernel.loop.run",
        "__package__": "autokernel.loop",
        "status": status,
        "Mapping": Mapping,
        "json": json,
        "args": SimpleNamespace(
            store=root,
            surface="fixture",
            serving_pairs=1,
            pairs=1,
            model="fixture",
            iterations=8,
            worktree=root,
            out=None,
        ),
        "epoch": "e",
        "current_anchor_commit": ["a"],
        "direct_launch": None,
        "floor": 1.0,
        "selected_identity": None,
        "experimental": False,
        "runtime_status": [None],
        "anchor_guard_seen": [],
        "accumulator_state": lambda: {},
        "pending_view": [[]],
        "gpu_reading": lambda rows: None,
        "actor_health": lambda rows: {},
        "history_view": [None],
        "scratch_registry": [None],
        "uncovered_gap_state": [gap],
        "_git": lambda *args: "a",
        "HEARTBEAT_S": 30,
        "roofline_coverage": rc,
    }
    bindings = [
        n
        for n in ast.walk(tree)
        if (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "scope_gap_publisher" for t in n.targets)
        )
        or (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "scope_gap_error"
        )
    ]
    assert len(bindings) == 2  # actual per-run state, not a test-created substitute
    exec(compile(ast.Module(body=[*bindings, node], type_ignores=[]), str(source), "exec"), env)
    return env["publish"], env


def test_actual_publisher_repeated_beats_preserve_first_artifact_and_print_once(tmp_path, capsys):
    publish, _ = _native_publisher(tmp_path)
    with patch.object(status, "write_json", wraps=status.write_json) as atomic:
        publish("running", _outcomes())
        target = tmp_path / rc.SCOPE_GAP_FILENAME
        original, mtime = target.read_bytes(), target.stat().st_mtime_ns
        for _ in range(20):
            publish("running", _outcomes())
        assert target.read_bytes() == original
        assert target.stat().st_mtime_ns == mtime
        calls = [c for c in atomic.call_args_list if c.args[1] == rc.SCOPE_GAP_FILENAME]
        assert len(calls) == 1
    assert capsys.readouterr().out.count("STAGNATION + UNCOVERED") == 1
    body = json.loads((tmp_path / status.STATUS_FILENAME).read_text())
    assert body["schema"] == status.STATUS_SCHEMA
    assert "scope_gap_error" not in body


def test_changed_gap_and_reentry_write_once_per_transition(tmp_path, capsys):
    publish, env = _native_publisher(tmp_path)
    with patch.object(rc, "write_scope_gap", wraps=rc.write_scope_gap) as writes:
        publish("running", _outcomes())
        gap = env["uncovered_gap_state"][0]
        env["uncovered_gap_state"][0] = dict(reversed(list(gap.items())))
        publish("running", _outcomes())  # mapping order is not a new gap
        assert writes.call_count == 1
        env["uncovered_gap_state"][0] = {**gap, "uncovered_nodes": [{"file": "other.cpp"}]}
        publish("running", _outcomes())
        publish("running", _outcomes())
        assert writes.call_count == 2
        publish("running", _outcomes(progress=True))
        publish("running", _outcomes())
        assert writes.call_count == 3
        env["uncovered_gap_state"][0] = None
        publish("running", _outcomes())
        env["uncovered_gap_state"][0] = gap
        publish("running", _outcomes())
        assert writes.call_count == 4
    assert capsys.readouterr().out.count("STAGNATION + UNCOVERED") == 4


def test_atomic_failure_keeps_prior_artifact_retries_and_clears_error(tmp_path, capsys):
    publish, env = _native_publisher(tmp_path)
    publish("running", _outcomes())
    target = tmp_path / rc.SCOPE_GAP_FILENAME
    before = target.read_bytes()
    env["uncovered_gap_state"][0] = {"triggered": True, "uncovered_nodes": [{"file": "new.cpp"}]}
    real_replace = status.os.replace
    failures = []

    def fail_scope_only(src, dst):
        if Path(dst).name == rc.SCOPE_GAP_FILENAME:
            failures.append(str(src))
            raise OSError(f"fixture full {len(failures)} at {src}")
        return real_replace(src, dst)

    with patch.object(status.os, "replace", side_effect=fail_scope_only):
        for _ in range(3):
            publish("running", _outcomes())
        assert target.read_bytes() == before
        assert not list(tmp_path.glob(".scope-gap-*"))
        body = json.loads((tmp_path / status.STATUS_FILENAME).read_text())
        assert body["scope_gap_error"]["type"] == "OSError"
        assert body["scope_gap_error"]["message"].startswith("fixture full 3 at ")
        assert body["uncovered_gap"]["uncovered_nodes"][0]["file"] == "new.cpp"
    assert capsys.readouterr().out.count("scope-gap stagnation hook skipped") == 1
    publish("running", _outcomes())
    assert target.read_bytes() != before
    assert "scope_gap_error" not in json.loads((tmp_path / status.STATUS_FILENAME).read_text())
    assert capsys.readouterr().out.count("STAGNATION + UNCOVERED") == 1


@pytest.mark.parametrize("bad_gap", [["bad shape"], {"triggered": True, "bad": object()}])
def test_malformed_gap_is_nonfatal_structured_and_rate_limited_with_recovery(
    tmp_path, capsys, bad_gap
):
    publish, env = _native_publisher(tmp_path)
    valid = env["uncovered_gap_state"][0]
    env["uncovered_gap_state"][0] = bad_gap
    for _ in range(3):
        publish("running", _outcomes())
    body = json.loads((tmp_path / status.STATUS_FILENAME).read_text())
    assert body["state"] == "running" and body["scope_gap_error"]["type"] == "TypeError"
    assert "uncovered_gap" not in body
    assert capsys.readouterr().out.count("scope-gap stagnation hook skipped") == 1
    env["uncovered_gap_state"][0] = valid
    publish("running", _outcomes())
    assert "scope_gap_error" not in json.loads((tmp_path / status.STATUS_FILENAME).read_text())
    env["uncovered_gap_state"][0] = bad_gap
    publish("running", _outcomes())
    assert capsys.readouterr().out.count("scope-gap stagnation hook skipped") == 1


def test_direct_hook_still_raises_for_its_malformed_output(tmp_path):
    with pytest.raises(AttributeError):
        rc.stagnation_scope_gap_hook(
            ["bad"], {"consecutive_measured_nulls": 5}, store_root=tmp_path
        )
