"""Hermetic tests for the shared frozen-kernel mutation policy."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from . import kernel_mutation_guard as guard


def _git(repo: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if check and done.returncode:
        raise AssertionError(done.stderr)
    return done.stdout.strip()


def _repo(path: Path, *, branch: str = "experimental") -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", branch)
    _git(path, "-c", "user.name=Guard Test", "-c",
         "user.email=guard@example.invalid", "commit", "--allow-empty", "-qm", "base")
    return path


def _branch(repo: Path) -> str | None:
    value = _git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    return value or None


def test_frozen_kernel_root_configuration_is_complete() -> None:
    assert guard.FROZEN_PRODUCTION_ROOTS == frozenset({
        Path("/mnt/raid0/llm/llama.cpp"),
        Path("/mnt/raid0/llm/whisper.cpp"),
        Path("/mnt/raid0/llm/qwentts.cpp"),
    })


@pytest.mark.parametrize("configured_root", sorted(guard.FROZEN_PRODUCTION_ROOTS))
def test_each_configured_kernel_root_is_refused_with_temporary_fixture(
        tmp_path, monkeypatch, configured_root):
    repo = _repo(tmp_path / configured_root.name)
    monkeypatch.setattr(guard, "FROZEN_PRODUCTION_ROOTS", frozenset({repo}))

    with pytest.raises(guard.FrozenKernelMutationRefused,
                       match="canonical frozen production checkout"):
        guard.ensure_kernel_mutation_allowed(repo, _branch(repo))


@pytest.mark.parametrize("production_branch", [
    "production-consolidated-v999",
    "production-speech-v999",
])
def test_production_branch_families_are_refused(tmp_path, production_branch):
    repo = _repo(tmp_path / "repo", branch=production_branch)

    with pytest.raises(guard.FrozenKernelMutationRefused,
                       match="frozen production branch"):
        guard.ensure_kernel_mutation_allowed(repo, _branch(repo))


def test_detached_head_in_canonical_root_is_still_refused(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "production")
    _git(repo, "checkout", "-q", "--detach")
    monkeypatch.setattr(guard, "FROZEN_PRODUCTION_ROOTS", frozenset({repo}))

    assert _branch(repo) is None
    with pytest.raises(guard.FrozenKernelMutationRefused,
                       match="canonical frozen production checkout"):
        guard.ensure_kernel_mutation_allowed(repo, _branch(repo))


def test_symlink_alias_of_canonical_root_is_refused(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "production")
    alias = tmp_path / "production-alias"
    alias.symlink_to(repo, target_is_directory=True)
    monkeypatch.setattr(guard, "FROZEN_PRODUCTION_ROOTS", frozenset({repo}))

    with pytest.raises(guard.FrozenKernelMutationRefused,
                       match="canonical frozen production checkout"):
        guard.ensure_kernel_mutation_allowed(alias, _branch(alias))


def test_experimental_linked_worktree_sharing_production_objects_is_allowed(
        tmp_path, monkeypatch):
    production = _repo(tmp_path / "production", branch="production-speech-v999")
    linked = tmp_path / "experimental"
    _git(production, "worktree", "add", "-qb", "candidate-work", str(linked))
    monkeypatch.setattr(guard, "FROZEN_PRODUCTION_ROOTS", frozenset({production}))

    assert Path(_git(linked, "rev-parse", "--path-format=absolute",
                     "--git-common-dir")).resolve() == (
        Path(_git(production, "rev-parse", "--path-format=absolute",
                  "--git-common-dir")).resolve())
    guard.ensure_kernel_mutation_allowed(linked, _branch(linked))
