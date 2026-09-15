"""Pre-build integrity gates for planner-authored kernel candidates.

The actor's path list is a claim, not an isolation boundary.  This module derives
the complete candidate from git, refuses protected or undeclared files, and gives
the caller a tree object that can be compared with the eventual keep commit.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Iterable, Sequence


class IntegrityRefused(RuntimeError):
    """A durable science refusal: the candidate must not reach a build."""

    def __init__(self, refusal_class: str, reason: str):
        self.refusal_class = refusal_class
        super().__init__(f"{refusal_class}: {reason}")


@dataclass(frozen=True)
class SpecialCaseFinding:
    kind: str
    path: str
    line: str


@dataclass(frozen=True)
class CandidateIntegrity:
    paths: tuple[str, ...]
    tree: str
    findings: tuple[SpecialCaseFinding, ...]

    @property
    def needs_confirm(self) -> bool:
        return bool(self.findings)


_PROTECTED_PREFIXES = ("tests/", "tools/llama-bench/", "tools/server/",
                       "examples/", "scripts/")
_LITERAL_PREDICATE = re.compile(
    r"(?:\b(?:ne\d|type|op|op_id)\b|->(?:ne|type|op)\b|\[(?:[0-3])\])"
    r"[^;{}]*(?:==|!=|<=|>=|<|>)[^;{}]*"
    r"(?:\b\d+\b|GGML_TYPE_[A-Z0-9_]+|GGML_OP_[A-Z0-9_]+)")
_MUTABLE_STATE = re.compile(
    r"\b(?:static\s+(?!const\b|constexpr\b)|(?:std::)?atomic\s*<|thread_local\b)")


def _git(worktree: Path, *args: str, input_bytes: bytes | None = None,
         env: dict[str, str] | None = None) -> bytes:
    done = subprocess.run(["git", "-C", str(worktree), *args], input=input_bytes,
                          capture_output=True, timeout=300, env=env)
    if done.returncode:
        raise IntegrityRefused("integrity_instrument_failed",
                               done.stderr.decode(errors="replace").strip())
    return done.stdout


def _normal_path(raw: str) -> str:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise IntegrityRefused("invalid_declared_path", repr(raw))
    return path.as_posix()


def dirty_paths(worktree: Path) -> tuple[str, ...]:
    """Return every tracked, staged, renamed and untracked dirty path."""
    fields = _git(worktree, "status", "--porcelain=v1", "-z",
                  "--untracked-files=all").split(b"\0")
    paths: set[str] = set()
    index = 0
    while index < len(fields) and fields[index]:
        field = fields[index]
        if len(field) < 4:
            raise IntegrityRefused("integrity_instrument_failed", "malformed git status")
        status, raw = field[:2], field[3:]
        paths.add(_normal_path(raw.decode("utf-8", "surrogateescape")))
        if b"R" in status or b"C" in status:
            index += 1
            if index >= len(fields) or not fields[index]:
                raise IntegrityRefused("integrity_instrument_failed", "truncated rename")
            paths.add(_normal_path(fields[index].decode("utf-8", "surrogateescape")))
        index += 1
    return tuple(sorted(paths))


def _protected(path: str) -> bool:
    return (path.startswith(_PROTECTED_PREFIXES)
            or PurePosixPath(path).name == "CMakeLists.txt")


def _candidate_tree(worktree: Path) -> str:
    """Materialize the full working tree in a temporary index without changing it."""
    fd, name = tempfile.mkstemp(prefix="autokernel-integrity-index-")
    os.close(fd)
    try:
        env = dict(os.environ, GIT_INDEX_FILE=name)
        os.unlink(name)  # read-tree requires an absent or valid index, not an empty file
        _git(worktree, "read-tree", "HEAD", env=env)
        _git(worktree, "add", "-A", "--", env=env)
        return _git(worktree, "write-tree", env=env).decode().strip()
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _added_lines(worktree: Path) -> Iterable[tuple[str, str]]:
    diff = _git(worktree, "diff", "--no-ext-diff", "--unified=0", "HEAD", "--")
    path = ""
    for raw in diff.decode("utf-8", "replace").splitlines():
        if raw.startswith("+++ b/"):
            path = raw[6:]
        elif raw.startswith("+") and not raw.startswith("+++") and path:
            yield path, raw[1:].strip()


def screen_special_cases(worktree: Path) -> tuple[SpecialCaseFinding, ...]:
    findings: list[SpecialCaseFinding] = []
    for path, line in _added_lines(worktree):
        if _LITERAL_PREDICATE.search(line):
            findings.append(SpecialCaseFinding("literal_shape_predicate", path, line))
        if _MUTABLE_STATE.search(line):
            findings.append(SpecialCaseFinding("hot_path_mutable_state", path, line))
    return tuple(findings)


def validate_candidate(worktree: Path, declared_paths: Sequence[str]) -> CandidateIntegrity:
    declared = tuple(sorted({_normal_path(str(path)) for path in declared_paths}))
    dirty = dirty_paths(worktree)
    if dirty != declared:
        raise IntegrityRefused("dirty_set_mismatch",
                               f"declared={list(declared)!r} dirty={list(dirty)!r}")
    protected = [path for path in dirty if _protected(path)]
    if protected:
        raise IntegrityRefused("oracle_bench_source_modified", repr(protected))
    outside = [path for path in dirty
               if not (path.startswith("ggml/src/") or path.startswith("src/"))]
    if outside:
        raise IntegrityRefused("outside_kernel_allowlist", repr(outside))
    return CandidateIntegrity(dirty, _candidate_tree(worktree),
                              screen_special_cases(worktree))


def assert_measured_tree(worktree: Path, expected_tree: str) -> None:
    actual = _candidate_tree(worktree)
    if actual != expected_tree:
        raise IntegrityRefused("measured_tree_changed",
                               f"measured={expected_tree} current={actual}")


def assert_kept_commit(worktree: Path, commit: str, expected_tree: str) -> None:
    actual = _git(worktree, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if actual != expected_tree:
        raise IntegrityRefused("kept_tree_mismatch",
                               f"measured={expected_tree} kept={actual}")
