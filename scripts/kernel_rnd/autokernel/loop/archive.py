#!/usr/bin/env python3
"""The champion branch is the only durable state.

No intermediate artifact needs to be trusted, because there is none. The champion is
a git commit: anyone can check it out and re-measure it. "Did we improve?" is
answered by rebuilding and benching, not by reading a receipt about a run that
happened last Tuesday.

That is the whole reason this replaces 4,491 uses of `receipt`. A receipt exists to
make a RECORD trustworthy; a commit is trustworthy because it is re-executable. The
old loop chose verification by proof where verification by reproduction was
available -- a `llama-bench` re-run costs 90 seconds.

`experiments.md` is the other half: every attempt, with its mechanism and its
measured effect, negatives written up as carefully as wins. It is the planner's
memory and therefore the loop's real product.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping

from ..controller import experiments


class RatchetRefused(RuntimeError):
    """The champion branch could not be advanced."""


def _git(repo: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=600)
    if check and done.returncode != 0:
        raise RatchetRefused(f"git {' '.join(args)}: {done.stderr.strip()[:400]}")
    return done.stdout.strip()


def champion_head(repo: Path, branch: str) -> str:
    return _git(repo, "rev-parse", branch)


def keep(repo: Path, *, branch: str, message: str, paths: tuple[str, ...]) -> str:
    """Commit an accepted patch onto the champion branch, and return the new head.

    Pathspec-limited staging, never a blanket `git add`: this repo is shared, and a
    bare commit takes whatever is in the index INCLUDING a peer's staged files.
    """
    if not paths:
        raise RatchetRefused("a champion advance must name the files it changes")
    _git(repo, "add", "--", *paths)
    staged = _git(repo, "diff", "--cached", "--name-only")
    if not staged:
        raise RatchetRefused("nothing staged; the patch produced no committable change")
    _git(repo, "commit", "-q", "-m", message)
    return champion_head(repo, "HEAD")


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


def recall(store_root: Path, *, epoch: str, limit: int = 40) -> list[dict]:
    """What has been tried, most recent first, cross-epoch records marked stale."""
    with experiments.ExperimentStore(store_root) as store:
        return store.recall(epoch=epoch, limit=limit)


def epoch_for(*, anchor_commit: str, build_recipe: Mapping[str, Any],
              host_state: Mapping[str, Any] | None = None) -> str:
    return experiments.epoch_sha256(anchor_commit=anchor_commit,
                                    build_recipe=build_recipe,
                                    host_state=host_state)


__all__ = ["RatchetRefused", "champion_head", "epoch_for", "keep", "recall", "record"]
