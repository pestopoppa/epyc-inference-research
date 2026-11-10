"""C10 floor 2026-11-10: EPYC_ROOT_REPO is the hub view (no `repos/`), and the claim
refused although the orchestrator was on PYTHONPATH. Resolution must accept an explicit
EPYC_ORCHESTRATOR_REPO or a sys.path owner, and still refuse when nothing resolves."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

from scripts.kernel_rnd.autokernel.loop import ak_check, claim


def _orchestrator(root: Path) -> Path:
    runtime = root / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "cpu_region_lock.py").write_text("", encoding="utf-8")
    return root


def _view(tmp_path: Path) -> Path:
    view = tmp_path / "views" / "epyc-root-main"
    (view / ".git").mkdir(parents=True)  # a real clone: .git is a directory, no repos/
    return view


def test_view_without_repos_resolves_through_sys_path(tmp_path):
    view, orch = _view(tmp_path), _orchestrator(tmp_path / "epyc-orchestrator")
    env = {"EPYC_ROOT_REPO": str(view)}
    got = claim.resolve_orchestrator_root(env, [str(tmp_path / "other"), str(orch)])
    assert got == orch.resolve()


def test_view_without_repos_resolves_through_explicit_env(tmp_path):
    view, orch = _view(tmp_path), _orchestrator(tmp_path / "epyc-orchestrator")
    env = {"EPYC_ROOT_REPO": str(view), "EPYC_ORCHESTRATOR_REPO": str(orch)}
    assert claim.resolve_orchestrator_root(env, []) == orch.resolve()


def test_view_without_repos_refuses_when_nothing_resolves(tmp_path):
    view = _view(tmp_path)
    with pytest.raises(claim.ClaimRefused, match="does not resolve the orchestrator"):
        claim.resolve_orchestrator_root({"EPYC_ROOT_REPO": str(view)}, [str(tmp_path)])


def test_explicit_env_that_is_not_an_owner_refuses_even_if_path_has_one(tmp_path):
    view, orch = _view(tmp_path), _orchestrator(tmp_path / "epyc-orchestrator")
    env = {"EPYC_ROOT_REPO": str(view), "EPYC_ORCHESTRATOR_REPO": str(tmp_path / "nope")}
    with pytest.raises(claim.ClaimRefused, match="EPYC_ORCHESTRATOR_REPO"):
        claim.resolve_orchestrator_root(env, [str(orch)])


def test_root_layout_still_wins_over_sys_path(tmp_path):
    root = tmp_path / "root"
    layout = _orchestrator(root / "repos" / "epyc-orchestrator")
    other = _orchestrator(tmp_path / "other-orch")
    got = claim.resolve_orchestrator_root({"EPYC_ROOT_REPO": str(root)}, [str(other)])
    assert got == layout.resolve()


def test_ak_check_mirror_agrees(tmp_path, monkeypatch):
    view, orch = _view(tmp_path), _orchestrator(tmp_path / "epyc-orchestrator")
    monkeypatch.setenv("EPYC_ROOT_REPO", str(view))
    monkeypatch.delenv("EPYC_ORCHESTRATOR_REPO", raising=False)
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    assert ak_check._orchestrator_root() is None
    monkeypatch.setattr(sys, "path", [str(tmp_path), str(orch)])
    assert ak_check._orchestrator_root() == orch.resolve()
    monkeypatch.setenv("EPYC_ORCHESTRATOR_REPO", str(tmp_path / "nope"))
    assert ak_check._orchestrator_root() is None


@pytest.mark.parametrize("mode", ["pythonpath", "env", "neither"])
def test_subprocess_ensure_importable_under_view(tmp_path, mode):
    """The real entry point, in a child, under the live-run env shape."""
    view, orch = _view(tmp_path), _orchestrator(tmp_path / "epyc-orchestrator")
    research_root = Path(__file__).parents[4]
    env = {k: v for k, v in os.environ.items()
           if k not in {"PYTHONPATH", "EPYC_ORCHESTRATOR_REPO"}}
    env["EPYC_ROOT_REPO"] = str(view)
    env["PYTHONPATH"] = "."
    if mode == "pythonpath":
        env["PYTHONPATH"] = os.pathsep.join((".", str(orch)))
    elif mode == "env":
        env["EPYC_ORCHESTRATOR_REPO"] = str(orch)
    result = subprocess.run(
        [sys.executable, "-c",
         "from scripts.kernel_rnd.autokernel.loop import claim\n"
         "claim._ensure_orchestrator_importable()\n"],
        cwd=research_root, env=env, capture_output=True, text=True, timeout=30)
    if mode == "neither":
        assert result.returncode != 0
        assert "does not resolve the orchestrator CPU-lock owner" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
