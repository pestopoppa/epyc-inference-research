#!/usr/bin/env python3
"""Champion source identity and immutable measurement history.

A champion commit identifies re-executable source; it is not evidence that a build,
benchmark, or scientific result occurred. Those records live separately in the
immutable experiment journal, with negatives retained alongside accepted attempts.
Advancing source never creates or upgrades an artifact/evidence claim.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from ..controller import experiments
from . import kernel_mutation_guard


class RatchetRefused(RuntimeError):
    """The champion branch could not be advanced."""


_AMBIENT_REPO_ENV = ("GIT_DIR", "GIT_WORK_TREE")


def _git_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an isolated Git environment without mutating the caller's mapping."""
    command_env = os.environ.copy()
    for name in (*_AMBIENT_REPO_ENV, "GIT_INDEX_FILE"):
        command_env.pop(name, None)
    if overrides:
        command_env.update(overrides)
    return command_env


def _git(repo: Path, *args: str, check: bool = True,
         env: Mapping[str, str] | None = None, input_text: str | None = None,
         raw_output: bool = False) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=600,
                          env=_git_env(env), input=input_text)
    if check and done.returncode != 0:
        raise RatchetRefused(f"git {' '.join(args)}: {done.stderr.strip()[:400]}")
    return done.stdout if raw_output else done.stdout.strip()


def _verified_repo(repo: Path) -> Path:
    redirected = [name for name in _AMBIENT_REPO_ENV if name in os.environ]
    if redirected:
        raise RatchetRefused(
            f"ambient Git repository redirection is not allowed: {', '.join(redirected)}")
    requested = repo.resolve()
    actual = Path(_git(requested, "rev-parse", "--show-toplevel")).resolve()
    if actual != requested:
        raise RatchetRefused(
            f"repository path must be its actual root: requested {requested}, root {actual}")
    return actual


def _validated_paths(repo: Path, paths: tuple[str, ...]) -> tuple[str, ...]:
    if not paths:
        raise RatchetRefused("a champion advance must name the files it changes")
    validated = []
    for raw in paths:
        if not isinstance(raw, str) or not raw:
            raise RatchetRefused("accepted paths must be non-empty strings")
        parts = raw.split("/")
        if (Path(raw).is_absolute() or raw != raw.strip() or raw.startswith(":")
                or any(part in {"", ".", ".."} for part in parts)
                or any(char in raw for char in "*?[")
                or any(ord(char) < 32 or ord(char) == 127 for char in raw)):
            raise RatchetRefused(f"unsafe accepted path {raw!r}; literal relative files only")
        candidate = repo / raw
        object_type = _git(repo, "cat-file", "-t", f"HEAD:{raw}", check=False)
        if candidate.is_dir() or object_type == "tree":
            raise RatchetRefused(f"accepted path is a directory, not a file: {raw!r}")
        if raw in validated:
            raise RatchetRefused(f"duplicate accepted path {raw!r}")
        validated.append(raw)
    return tuple(validated)


def _hook(repo: Path, name: str, env: Mapping[str, str], *args: str,
          check: bool = True) -> None:
    hook_name = _git(repo, "rev-parse", "--git-path", f"hooks/{name}")
    hook = Path(hook_name)
    if not hook.is_absolute():
        hook = repo / hook
    if not hook.is_file() or not os.access(hook, os.X_OK):
        return
    command_env = _git_env(env)
    try:
        done = subprocess.run([str(hook), *args], cwd=repo, env=command_env,
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        if check:
            raise RatchetRefused(f"{name} hook could not run: {exc}") from exc
        print(f"warning: {name} hook could not report after committed ref update: {exc}",
              file=sys.stderr)
        return
    if check and done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()[:400]
        raise RatchetRefused(f"{name} hook refused champion commit: {detail}")
    if not check and done.returncode != 0:
        print(f"warning: {name} hook reported after committed ref update: "
              f"{(done.stderr or done.stdout).strip()[:400]}", file=sys.stderr)


def _changed_paths(repo: Path, expected: str, env: Mapping[str, str]) -> set[str]:
    raw = _git(repo, "diff", "--cached", "--name-only", "-z", expected, "--",
               env=env, raw_output=True)
    return {path for path in raw.split("\0") if path}


def _require_private_diff(repo: Path, expected: str, paths: tuple[str, ...],
                          env: Mapping[str, str]) -> None:
    changed = _changed_paths(repo, expected, env)
    if not changed:
        raise RatchetRefused("accepted paths produced no committable change")
    unexpected = changed - set(paths)
    if unexpected:
        raise RatchetRefused(
            f"commit hook staged paths outside the accepted patch: {sorted(unexpected)}")
    # A hook may inspect or even format the accepted working files, but the private
    # index must still name their explicit current bytes. It may never substitute a
    # peer's differently staged version.
    _git(repo, "diff", "--quiet", "--", *paths, env=env)


def _verify_head_binding(repo: Path, symbolic: str, expected: str) -> None:
    current_root = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()
    if current_root != repo:
        raise RatchetRefused(
            f"repository root changed before commit: expected {repo}, found {current_root}")
    current_symbolic = _git(repo, "symbolic-ref", "-q", "HEAD", check=False)
    _guard_kernel_mutation(repo, current_symbolic)
    if current_symbolic != symbolic:
        before = symbolic or "detached HEAD"
        after = current_symbolic or "detached HEAD"
        raise RatchetRefused(f"HEAD binding changed before commit: {before} -> {after}")
    current_head = _git(repo, "rev-parse", "HEAD")
    if current_head != expected:
        raise RatchetRefused(
            f"HEAD moved before commit: expected {expected}, found {current_head}")


def _guard_kernel_mutation(repo: Path, branch: str | None) -> None:
    try:
        kernel_mutation_guard.ensure_kernel_mutation_allowed(repo, branch)
    except kernel_mutation_guard.FrozenKernelMutationRefused as exc:
        raise RatchetRefused(str(exc)) from exc


def keep(repo: Path, *, branch: str, message: str, paths: tuple[str, ...]) -> str:
    """CAS an accepted source patch onto the current branch and return its commit.

    A private index is seeded from the exact captured HEAD and stages only validated
    literal files from the working tree, including tracked deletions. ``keep`` itself
    never resets or rewrites the caller's shared index or peer working files; normal
    hooks retain their ordinary ability to edit the working tree. After the ref
    advances that untouched index can legitimately appear stale relative to the new
    HEAD; do not "repair" it, even for selected paths, because those entries may
    contain distinct peer staging.

    Normal pre-commit, prepare-commit-msg, and commit-msg constraints run against the
    private index. ``commit-tree`` preserves Git author/committer identity and optional
    configured signing; atomic ``update-ref <new> <expected-old>`` prevents overwriting
    a concurrent advance. Post-commit is notification-only and runs after the CAS; as
    with normal Git it cannot veto or roll back an already-created commit.

    This commits source identity only. It infers no build, artifact, measurement, or
    evidence claim.
    """
    repo = _verified_repo(Path(repo))
    accepted = _validated_paths(repo, paths)
    symbolic = _git(repo, "symbolic-ref", "-q", "HEAD", check=False)
    _guard_kernel_mutation(repo, symbolic)
    if branch == "HEAD":
        target = symbolic or "HEAD"
        no_deref = not symbolic
    else:
        short = branch.removeprefix("refs/heads/")
        _git(repo, "check-ref-format", "--branch", short)
        target = f"refs/heads/{short}"
        if symbolic != target:
            current = symbolic.removeprefix("refs/heads/") if symbolic else "detached HEAD"
            raise RatchetRefused(
                f"requested branch {short!r} is not checked out (current: {current})")
        no_deref = False
    expected = _git(repo, "rev-parse", "HEAD")

    with tempfile.TemporaryDirectory(prefix="autokernel-private-index-") as tmp:
        index = Path(tmp) / "index"
        env = {"GIT_INDEX_FILE": str(index), "GIT_LITERAL_PATHSPECS": "1"}
        _git(repo, "read-tree", expected, env=env)
        _git(repo, "add", "-A", "--", *accepted, env=env)
        _require_private_diff(repo, expected, accepted, env)
        _hook(repo, "pre-commit", env)

        message_path = Path(tmp) / "COMMIT_EDITMSG"
        cleaned = _git(repo, "stripspace", input_text=message)
        if not cleaned:
            raise RatchetRefused("champion commit message is empty after cleanup")
        message_path.write_text(cleaned + "\n", encoding="utf-8")
        _hook(repo, "prepare-commit-msg", env, str(message_path), "message")
        _hook(repo, "commit-msg", env, str(message_path))
        if not message_path.read_text(encoding="utf-8").strip():
            raise RatchetRefused("commit-msg hook left an empty champion message")

        _require_private_diff(repo, expected, accepted, env)
        tree = _git(repo, "write-tree", env=env)
        commit_args = ["commit-tree", tree, "-p", expected, "-F", str(message_path)]
        signing = _git(repo, "config", "--type=bool", "--default=false", "--get",
                       "commit.gpgSign")
        if signing == "true":
            commit_args.insert(1, "-S")
        new_head = _git(repo, *commit_args, env=env)
        # update-ref rejects a symref verification and an update of its referent as
        # duplicate updates in one transaction. Recheck both immediately before the
        # CAS; expected-old still atomically protects the captured parent.
        _verify_head_binding(repo, symbolic, expected)
        update_args = ["update-ref"]
        if no_deref:
            update_args.append("--no-deref")
        update_args.extend([target, new_head, expected])
        _git(repo, *update_args, env=env)
        try:
            _hook(repo, "post-commit", env, check=False)
        except Exception as exc:
            # The ref has landed. A notification failure cannot turn success into an
            # ambiguous caller-visible exception claiming no commit was made.
            print(f"warning: post-commit notification failed after committed ref "
                  f"update: {exc}", file=sys.stderr)
        return new_head


def record(store_root: Path, attempt: Mapping[str, Any], *, epoch: str,
           recorded_at: str, campaign_id: str) -> bool:
    """Append one attempt to durable memory and refresh `experiments.md`.

    Idempotent on attempt identity, so a resumed loop re-recording its own rows
    cannot inflate the history it will later read back.
    """
    with experiments.ExperimentStore(store_root) as store:
        added = store.record(attempt, epoch=epoch, recorded_at=recorded_at,
                             campaign_id=campaign_id)
        store.write_markdown(epoch=epoch)
        return added


def recall(store_root: Path, *, epoch: str, limit: int = 40,
           ranking_authorized: bool = False) -> list[dict]:
    """What has been tried, cross-epoch records marked stale.

    Recency order by default. `ranking_authorized` is `P-AK-SEARCH-1-A3`'s epoch-scoped
    ranking, off unless the caller asks (`run.py --rank-prior-experiments`): it returns
    an order of merit instead, with cross-epoch magnitudes redacted. The default is
    unchanged, and passing the flag is a decision with a name on it.
    """
    with experiments.ExperimentStore(store_root) as store:
        return store.recall(epoch=epoch, limit=limit,
                            ranking_authorized=ranking_authorized)


def epoch_for(*, anchor_commit: str, build_recipe: Mapping[str, Any],
              host_state: Mapping[str, Any] | None = None) -> str:
    return experiments.epoch_sha256(anchor_commit=anchor_commit,
                                    build_recipe=build_recipe,
                                    host_state=host_state)


__all__ = ["RatchetRefused", "epoch_for", "keep", "recall", "record"]
