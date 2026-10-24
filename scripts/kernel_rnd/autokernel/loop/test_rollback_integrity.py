"""Exercise the production accumulator rollback with injected process failures.

Bind main's compiled closures directly: entering main would claim hardware and
launch actors. These are the actual function bodies, with build/measurement and
git process boundaries replaced, rather than assertions about source spelling.
"""
import json
from pathlib import Path
import subprocess
from types import CodeType, FunctionType, SimpleNamespace

import pytest

from . import accumulate, loop, run
from .test_loop import _comparison, _Critic, _Planner


PRIOR = "b" * 40
CANDIDATE = "c" * 40


def _bind(name, bindings):
    code = next(code for code in run.main.__code__.co_consts
                if isinstance(code, CodeType) and code.co_name == name)

    def cell(value):
        return (lambda: value).__closure__[0]

    return FunctionType(code, run.__dict__, name,
                        closure=tuple(cell(bindings.get(key)) for key in code.co_freevars))


def _accumulator(tmp_path, monkeypatch, failure=None):
    head = [CANDIDATE]
    events = []
    reset = [False]
    bundle = accumulate.Bundle(champion_of_record="a" * 40, tip=PRIOR)
    original_keeps = [{"kept_commit": CANDIDATE}]

    def process(argv, **kwargs):
        assert argv[:3] == ["git", "-C", str(tmp_path / "source")]
        assert kwargs == dict(capture_output=True, text=True, timeout=600)
        args = argv[3:]
        stage = "reset" if args[0] == "reset" else "verify" if reset[0] else "head"
        events.append(stage)
        if failure and stage == failure[0]:
            if failure[1] == "exit":
                return subprocess.CompletedProcess(argv, 128, "", "injected git failure")
            if failure[1] == "timeout":
                raise subprocess.TimeoutExpired(argv, 600)
            if failure[1] == "spawn":
                raise OSError("injected git spawn failure")
        if args == ["reset", "--hard", PRIOR]:
            reset[0] = True
            if failure != ("verify", "wrong_head"):
                head[0] = PRIOR
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert args == ["rev-parse", "HEAD"]
        return subprocess.CompletedProcess(argv, 0, head[0] + "\n", "")

    def compare(*args, **kwargs):
        events.append("measure")
        # Two independent negative observations, then the restored prior tip.
        effect = -0.1 if events.count("measure") <= 2 else 0.01
        if events.count("measure") > 2:
            assert head[0] == PRIOR
        return SimpleNamespace(to_dict=lambda: {"effect": effect})

    def promote():
        assert head[0] == PRIOR
        events.append("build")

    monkeypatch.setattr(run.subprocess, "run", process)
    bindings = dict(
        args=SimpleNamespace(worktree=tmp_path / "source", store=tmp_path / "store",
                             accumulate_bench_every_keeps=1),
        bundle=[bundle], accum_policy=accumulate.AccumulatorPolicy(),
        accum_bench_forced=[False], serving_floor_pct=1.0, serving_recipe=object(),
        direct_launch=object(), cor_build=[tmp_path / "cor"],
        anchor_build=[tmp_path / "candidate"], cpu_compare=compare,
        publish=lambda *a, **kw: None, promote_anchor=promote,
        original_source_keeps=original_keeps)
    bindings["_accumulate_after_keep"] = _bind("_accumulate_after_keep", bindings)
    return _bind("accumulate_after_keep", bindings), events, head, bundle, original_keeps


@pytest.mark.parametrize("failure", [
    ("reset", "exit"), ("reset", "timeout"), ("reset", "spawn"),
    ("verify", "exit"), ("verify", "timeout"), ("verify", "spawn"),
    ("verify", "wrong_head"),
])
def test_failed_rollback_aborts_before_build_or_remeasurement(tmp_path, monkeypatch, failure):
    callback, events, head, bundle, original_keeps = _accumulator(
        tmp_path, monkeypatch, failure)
    with pytest.raises(loop.RunAborted, match="interaction-regression rollback"):
        callback("regressing-keep")
    assert events.count("measure") == 2
    assert "build" not in events
    assert bundle.tip == PRIOR
    assert original_keeps == [{"kept_commit": CANDIDATE}]
    assert not (tmp_path / "store" / "serving").exists()
    assert not list((tmp_path / "store").rglob("resolution-*.json"))


def test_successful_rollback_rebuilds_and_rechecks_restored_tip(tmp_path, monkeypatch):
    callback, events, head, bundle, original_keeps = _accumulator(tmp_path, monkeypatch)
    with pytest.raises(loop.InteractionRegression, match="restored prior tip"):
        callback("regressing-keep")
    assert events == ["head", "measure", "measure", "reset", "verify", "build", "measure"]
    assert head[0] == bundle.tip == PRIOR
    assert original_keeps == []
    resolution = next((tmp_path / "store").rglob("resolution-*.json"))
    row = json.loads(resolution.read_text())
    restored = json.loads(Path(row["restored_tip_recheck"]["path"]).read_text())
    assert row["candidate_tip"] == CANDIDATE
    assert restored["measured_tip"] == PRIOR


@pytest.mark.parametrize("failure", [("reset", "timeout"), ("reset", "exit")])
def test_failed_rollback_never_returns_candidate_as_kept(tmp_path, monkeypatch, failure):
    callback, events, *_ = _accumulator(tmp_path, monkeypatch, failure)

    def commit(hypothesis, paths, comparison):
        callback(hypothesis.mechanism_id)
        return CANDIDATE

    with pytest.raises(loop.RunAborted) as aborted:
        loop.iterate(planner=_Planner(), critic=_Critic([], []), context={},
                     measure=lambda *a: _comparison(0.05),
                     gate=lambda *a: (True, []), commit=commit)
    assert aborted.value.inflight_keep["committed_head"] is None
    assert "build" not in events


@pytest.mark.parametrize("check", [True, False])
def test_git_exit_check_contract(tmp_path, monkeypatch, check):
    def process(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 128, "partial identity", "failed")

    monkeypatch.setattr(run.subprocess, "run", process)
    if check:
        with pytest.raises(run.GitError):
            run._git(tmp_path, "rev-parse", "missing", check=check)
    else:
        assert run._git(tmp_path, "rev-parse", "missing", check=check) == "partial identity"


def test_git_success_returns_stripped_stdout(tmp_path, monkeypatch):
    monkeypatch.setattr(run.subprocess, "run", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0, PRIOR + "\n", ""))
    assert run._git(tmp_path, "rev-parse", "HEAD") == PRIOR
