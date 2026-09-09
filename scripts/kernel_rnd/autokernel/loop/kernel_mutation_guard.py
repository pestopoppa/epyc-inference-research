"""Shared refusal policy for mutations of frozen production kernel checkouts."""
from __future__ import annotations

from pathlib import Path


FROZEN_PRODUCTION_ROOTS = frozenset(Path(value) for value in (
    "/mnt/raid0/llm/llama.cpp",
    "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp",
))
FROZEN_PRODUCTION_BRANCH_PREFIXES = (
    "production-consolidated-",
    "production-speech-",
)


class FrozenKernelMutationRefused(RuntimeError):
    """A requested Git mutation targets a frozen production checkout or branch."""


def ensure_kernel_mutation_allowed(repo_root: Path, branch: str | None) -> None:
    """Refuse a canonical frozen worktree or checked-out production branch.

    The worktree identity is intentionally independent of its Git common directory:
    an experimental linked worktree may share the production object's database while
    retaining its own allowed root and branch.
    """
    resolved_root = Path(repo_root).resolve()
    frozen_roots = {Path(root).resolve() for root in FROZEN_PRODUCTION_ROOTS}
    if resolved_root in frozen_roots:
        raise FrozenKernelMutationRefused(
            "canonical frozen production checkout is refused")

    short_branch = (branch or "").removeprefix("refs/heads/")
    if short_branch.startswith(FROZEN_PRODUCTION_BRANCH_PREFIXES):
        raise FrozenKernelMutationRefused(
            f"frozen production branch is refused: {short_branch}")
