"""Read a repository's HEAD commit from its git files, without spawning git.

Used for provenance stamps (the commit a long-lived process STARTED on), where a
subprocess per stamp would be wasteful and a failing ``git`` binary must never
break the caller. Handles plain clones, linked worktrees (``.git`` file +
``commondir``) and packed refs. Returns None whenever it cannot tell.
"""

from __future__ import annotations

from pathlib import Path


def _git_dir(repo: Path) -> Path | None:
    dot_git = repo / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():  # worktree: "gitdir: <path>"
        try:
            text = dot_git.read_text().strip()
        except OSError:
            return None
        if text.startswith("gitdir:"):
            path = Path(text.split(":", 1)[1].strip())
            return path if path.is_absolute() else (repo / path).resolve()
    return None


def resolve_git_head(repo: Path) -> str | None:
    """HEAD commit sha of ``repo`` read from the git files (no subprocess)."""
    git_dir = _git_dir(repo)
    if git_dir is None:
        return None
    try:
        head = (git_dir / "HEAD").read_text().strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None
    ref = head.split(":", 1)[1].strip()
    # A linked worktree keeps HEAD privately but refs in the common dir.
    common = git_dir
    commondir_file = git_dir / "commondir"
    if commondir_file.is_file():
        try:
            common = (git_dir / commondir_file.read_text().strip()).resolve()
        except OSError:
            pass
    for base in (git_dir, common):
        try:
            return (base / ref).read_text().strip() or None
        except OSError:
            continue
    try:
        for line in (common / "packed-refs").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    except OSError:
        pass
    return None
