#!/usr/bin/env python3
"""DRAFT — the concurrent path's real-world wiring: git worktrees, refs, timing.

`pipeline` is pure control flow with every side effect injected. This module is the
other half: it turns a lane number into a real detached worktree with its own build
directory, turns "advance the champion" into an atomic ref update, and fixes the
accounting that only made sense with one lane.

OPT-IN AND REVERSIBLE. `run.py --workers 1` is today's sequential path, byte for byte;
nothing in this module is reached. `--workers N` is the only way in.

THE CHAMPION IS A BRANCH IN A WORKTREE
--------------------------------------
`/mnt/raid0/llm/tmp/ak-loop-tree` is a git worktree of `/mnt/raid0/llm/llama.cpp` with
`ak/loop-champion-20260828` checked out. Git refuses to check one branch out in two
worktrees, so lanes run DETACHED at the champion commit and only `advance_champion`
moves the branch.

NEVER `git worktree prune` OR `git gc` IN THIS REPO. On 2026-08-12 that destroyed all
five lanes mid-run. A lane whose directory is momentarily absent (a stale NFS handle,
a lane being re-provisioned) is indistinguishable to `prune` from a lane that is gone
for good, and `prune` resolves that ambiguity by deleting. Nothing here calls either.

NOT SAFE TO RUN BESIDE A SEQUENTIAL RUN. `advance_champion` resets the champion tree,
so a `--workers N` run and a `--workers 1` run must never hold the same champion tree
at once. In practice the GPU claim already enforces this -- both paths run under one
`claim.hold()` -- but the claim is about the device and this is about the tree, so it
is stated rather than inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Sequence

from . import archive, loop as loop_mod, pipeline

#: The frozen production clone the champion worktree belongs to.
SOURCE_REPO = Path("/mnt/raid0/llm/llama.cpp")
#: The champion worktree: this branch is checked out HERE and nowhere else.
CHAMPION_TREE = Path("/mnt/raid0/llm/tmp/ak-loop-tree")
CHAMPION_BRANCH = "ak/loop-champion-20260828"
#: One detached worktree per lane, one build directory per lane. Both must be
#: disjoint: two lanes cmake-configuring one build directory produce a `llama-bench`
#: that belongs to neither of them.
WORKER_ROOT = Path("/mnt/raid0/llm/tmp/ak-loop-lanes")
WORKER_BUILD_ROOT = Path("/mnt/raid0/llm/tmp/ak-loop-builds")

#: More lanes than this queue on the serialized tail instead of adding throughput,
#: and each one costs a full source tree plus a full build tree on the raid.
MAX_WORKERS = 8


def _git(repo: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=600)
    if check and done.returncode != 0:
        raise archive.RatchetRefused(
            f"git -C {repo} {' '.join(args)}: {done.stderr.strip()[:400]}")
    return done.stdout.strip()


# ------------------------------------------------------------------ provisioning


def check_lanes_are_disjoint(workers: Sequence[pipeline.Worker]) -> None:
    """Refuse a pool whose lanes would overwrite each other.

    A shared build directory does not fail loudly: lane A configures it, lane B
    reconfigures it, and A then benches a binary built from B's patch. That is a
    measurement attributed to the wrong candidate, which is the exact class of defect
    this rebuild exists to close, so it is checked before anything starts rather than
    discovered in the numbers.
    """
    for label, values in (("worktree", [w.worktree for w in workers]),
                          ("build", [w.build_dir for w in workers]),
                          ("name", [w.name for w in workers])):
        if len(set(values)) != len(values):
            raise ValueError(
                f"two lanes share a {label}: {values}. Each lane needs its own, or "
                f"one lane's measurement is of another lane's patch")


def provision(count: int, *, source: Path = SOURCE_REPO,
              champion_tree: Path = CHAMPION_TREE,
              champion_branch: str = CHAMPION_BRANCH,
              root: Path = WORKER_ROOT, build_root: Path = WORKER_BUILD_ROOT,
              execute: bool = False) -> list[pipeline.Worker]:
    """Plan (and, only with `execute=True`, create) one detached worktree per lane.

    `execute` defaults to False so the plan can be inspected and unit-tested without
    mutating a repo another run may be using -- the check that lanes are disjoint is
    worth more before the trees exist than after.

    Each lane is created with `git worktree add --detach <path> <sha>`, never
    `--branch`: the champion branch is checked out in `champion_tree` and git will
    refuse a second checkout of it. Detaching is not a workaround, it is the design --
    only the serialized tail is allowed to move the branch.

    An existing lane directory is REUSED, never deleted. Deleting is how five lanes
    were lost once already, and a lane tree is cheap to reset and expensive to rebuild.
    """
    if not 1 <= count <= MAX_WORKERS:
        raise ValueError(f"{count} lanes: outside 1..{MAX_WORKERS}. Beyond the tail's "
                         f"saturation point lanes only queue, and each costs a tree")
    head = champion_head(champion_tree, champion_branch) if execute else "HEAD"
    workers = [pipeline.Worker(name=f"lane{index}",
                               worktree=root / f"lane{index}",
                               build_dir=build_root / f"lane{index}")
               for index in range(count)]
    check_lanes_are_disjoint(workers)
    if any(worker.worktree == champion_tree for worker in workers):
        raise ValueError("a lane points at the champion tree; lanes must be detached "
                         "worktrees of their own")
    if not execute:
        return workers
    for worker in workers:
        if (worker.worktree / ".git").exists():
            # Reuse. `reset_to_champion` will detach it at the current champion.
            continue
        worker.worktree.parent.mkdir(parents=True, exist_ok=True)
        _git(source, "worktree", "add", "--detach", str(worker.worktree), head)
        worker.build_dir.mkdir(parents=True, exist_ok=True)
    return workers


# ------------------------------------------------------------------ the champion


def champion_head(champion_tree: Path = CHAMPION_TREE,
                  branch: str = CHAMPION_BRANCH) -> str:
    """The champion BRANCH, not any worktree's HEAD.

    Lanes are detached, so their HEAD is a base, not the champion. Reading the branch
    ref is the only reading that means "what a keep must build on".
    """
    return _git(champion_tree, "rev-parse", branch)


def reset_to_champion(worker: pipeline.Worker, *,
                      champion_tree: Path = CHAMPION_TREE,
                      branch: str = CHAMPION_BRANCH) -> str:
    """Detach the lane at the current champion and RETURN the sha it landed on.

    The sha is read ONCE and then checked out BY SHA. Checking out the branch name
    instead would read the ref a second time, so the value returned to the pipeline
    could name a commit the tree is not actually on -- and that value is the base every
    later staleness check is made against. A base that does not describe the tree makes
    the whole check ornamental.

    `clean` is scoped to the source directories a patch may touch, so a lane's build
    directory (which lives outside the tree) and any operator scratch survive.
    """
    head = champion_head(champion_tree, branch)
    # --force, and reset BEFORE clean. Without the force, the previous iteration's
    # uncommitted patch blocks the checkout outright:
    #
    #   error: Your local changes to ggml/src/ggml-cuda/quantize.cu would be
    #   overwritten by checkout
    #
    # which killed four of seven lanes in run 16. Discarding is the whole point of
    # this function -- the lane is being returned to the champion, and the patch it is
    # discarding was already written to <store>/patches before the gate ran.
    _git(worker.worktree, "checkout", "--detach", "--force", head)
    _git(worker.worktree, "reset", "--hard", head)
    _git(worker.worktree, "clean", "-fd", "ggml/", "src/", check=False)
    return head


def commit_message(hypothesis, comparison) -> str:
    """One spelling of the champion commit subject, shared by both run paths."""
    return (f"{hypothesis.mechanism_id}: {comparison.effect * 100:+.3f}% "
            f"on {comparison.surface} over {comparison.pairs} pairs")


def advance_champion(worker: pipeline.Worker, hypothesis, paths, comparison, *,
                     champion_tree: Path = CHAMPION_TREE,
                     branch: str = CHAMPION_BRANCH) -> str:
    """Commit the lane's patch and move the champion BRANCH onto it.

    The sequential path commits with `branch="HEAD"`, which works because its worktree
    has the champion branch checked out. A lane does not: it is detached, so committing
    there advances a detached HEAD and produces a commit that no ref points at. It
    would be measured, reported as kept, and then collected. So the ref move is
    explicit.

    `update-ref <ref> <new> <old>` is a compare-and-swap: it refuses if the ref is not
    still at `old`. This runs under the serialized tail's lock and after a staleness
    check, so it should never fire -- which is exactly why it is here. It costs
    nothing and it is the only thing standing between a bug in the lock discipline and
    a silently lost keep.

    The champion tree still has the OLD commit in its index and working tree after a
    ref move, so it is reset onto the new head. Nothing may be working in that tree
    during a pooled run.
    """
    base = _git(worker.worktree, "rev-parse", "HEAD")
    new_head = archive.keep(worker.worktree, branch="HEAD",
                            message=commit_message(hypothesis, comparison),
                            paths=tuple(paths))
    _git(champion_tree, "update-ref", f"refs/heads/{branch}", new_head, base)
    _git(champion_tree, "reset", "--hard", new_head)
    return new_head


# ------------------------------------------------------------------ accounting


class PhaseClock:
    """Per-LANE phase accounting. The single global mark stops meaning anything.

    `run.py`'s `note_phase` holds one `(label, started)` pair for the whole process.
    With W lanes, lane B's label overwrites lane A's mark and the elapsed interval is
    charged to whichever lane wrote last -- the numbers are not merely noisy, they are
    attributed to the wrong phase.

    The totals here are LANE-seconds, not wall-seconds: with W lanes they can sum to
    W x the wall clock. `wall_seconds` is reported alongside them so nobody reads a
    sum larger than the run's own duration as a contradiction.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._marks: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def note(self, worker_name: str, label: str) -> None:
        now = time.monotonic()
        with self._lock:
            previous = self._marks.get(worker_name)
            if previous is not None:
                prior_label, started = previous
                self.seconds[prior_label] = (
                    self.seconds.get(prior_label, 0.0) + (now - started))
            self._marks[worker_name] = (label, now)

    def close(self) -> None:
        """Charge every lane's open interval. Without this the last phase each lane
        was in -- often the longest -- is missing from the totals entirely."""
        now = time.monotonic()
        with self._lock:
            for label, started in self._marks.values():
                self.seconds[label] = self.seconds.get(label, 0.0) + (now - started)
            self._marks.clear()

    def totals(self) -> dict[str, float]:
        with self._lock:
            return {label: round(value, 1) for label, value in
                    sorted(self.seconds.items(), key=lambda kv: -kv[1])}


@dataclass
class PoolResult:
    """What a pooled run produced, plus the accounting a pooled run changes."""
    outcomes: list[loop_mod.Outcome] = field(default_factory=list)
    phase_seconds: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0
    tail_seconds: float = 0.0
    superseded: int = 0

    def to_dict(self, *, workers: int) -> dict[str, Any]:
        return {
            "workers": workers,
            "wall_seconds": round(self.wall_seconds, 1),
            # The fraction of the run that could not be overlapped. This is the number
            # that decides whether another lane would buy anything: once it approaches
            # 1.0 every extra lane only queues.
            "tail_seconds": round(self.tail_seconds, 1),
            "tail_fraction": (round(self.tail_seconds / self.wall_seconds, 4)
                              if self.wall_seconds > 0 else None),
            "superseded": self.superseded,
            # LANE-seconds. May exceed wall_seconds by up to the lane count.
            "phase_lane_seconds": self.phase_seconds,
        }


# ------------------------------------------------------------------ the driver


def drive(*, workers: Sequence[pipeline.Worker], make_planner, make_critic,
          build_context: Callable[[], dict], make_gate, make_measure,
          record: Callable[[loop_mod.Outcome], None], iterations: int | None,
          should_stop: Callable[[], bool] | None = None,
          champion_tree: Path = CHAMPION_TREE, branch: str = CHAMPION_BRANCH,
          reset: Callable[[pipeline.Worker], str] | None = None,
          commit: Callable[..., str] | None = None,
          on_step: Callable[[str, str], None] | None = None) -> PoolResult:
    """Run `iterations` iterations across `workers` lanes and report the accounting.

    Everything device-shaped is still injected; this only binds the git side and the
    per-lane clock. The GPU claim is NOT taken here: `run.py` holds one claim for the
    whole process, and a second `flock` on a second file descriptor in the same
    process would refuse itself.
    """
    check_lanes_are_disjoint(workers)
    clock = PhaseClock()
    tail = pipeline.SerializedTail(lambda: champion_head(champion_tree, branch))

    def step(worker_name: str, label: str) -> None:
        clock.note(worker_name, label)
        if on_step is not None:
            on_step(worker_name, label)

    started = time.monotonic()
    outcomes = pipeline.run_pool(
        workers=workers, make_planner=make_planner, make_critic=make_critic,
        build_context=build_context, make_gate=make_gate, make_measure=make_measure,
        commit=commit or (lambda worker, hypothesis, paths, comparison:
                          advance_champion(worker, hypothesis, paths, comparison,
                                           champion_tree=champion_tree, branch=branch)),
        champion_head=lambda: champion_head(champion_tree, branch),
        reset_to_champion=reset or (lambda worker: reset_to_champion(
            worker, champion_tree=champion_tree, branch=branch)),
        record=record, iterations=iterations, on_step=step, tail=tail,
        should_stop=should_stop)
    clock.close()
    return PoolResult(outcomes=outcomes, phase_seconds=clock.totals(),
                      wall_seconds=time.monotonic() - started,
                      tail_seconds=tail.tail_seconds, superseded=tail.superseded)


__all__ = ["CHAMPION_BRANCH", "CHAMPION_TREE", "MAX_WORKERS", "PhaseClock",
           "PoolResult", "SOURCE_REPO", "WORKER_BUILD_ROOT", "WORKER_ROOT",
           "advance_champion", "champion_head", "check_lanes_are_disjoint",
           "commit_message", "drive", "provision", "reset_to_champion"]


def promote_anchor(build_dir: Path, store: Path) -> Path:
    """Make a kept candidate build the new anchor, and return its path.

    THE WHOLE POINT. If the anchor does not advance, every effect is measured against
    a baseline the champion already beat -- so it is CUMULATIVE, a patch that regresses
    the champion still clears the floor, and recovering the marginal means inferring it
    from two noisy numbers with sqrt(2) the uncertainty. Runs 13 and 14 both committed
    patches that way. An advancing anchor makes every measurement directly marginal and
    the whole problem stops existing.

    MOVED, never copied: the next iteration rebuilds into the candidate slot, and an
    anchor sharing that path is an anchor that ends up measuring against itself.

    Lives here rather than inside `run.main` so a test can EXECUTE it. The predecessor
    was a closure, and its test asserted that the string "shutil.move" appeared in the
    source. It passed while `shutil` was never imported, so the promotion raised
    NameError on the first real keep and the anchor silently never advanced.
    """
    if not (build_dir / "bin").is_dir():
        raise ValueError(
            f"{build_dir} has no bin/: refusing to promote a build that produced no "
            f"binary, which would make every later comparison measure nothing")
    generation = len(list(store.glob("anchor-gen-*"))) + 1
    promoted = store / f"anchor-gen-{generation:03d}"
    shutil.move(str(build_dir), str(promoted))
    return promoted


#: Only the CURRENT anchor is needed. A patch is either in the champion or it is not,
#: and the anchor build is just that commit compiled -- git has the commits, and a
#: rebuild is 63 seconds. Keeping spares was caution without a use: promotion happens
#: inside the exclusive tail session, so no lane is ever measuring against the old
#: anchor while it is replaced, and superseded concurrent work is preserved elsewhere
#: entirely (its patch in <store>/patches, its hypothesis in the journal). Anchor
#: generations never protected that work and were never going to.
ANCHOR_GENERATIONS_KEPT = 1

#: Drop this file in the store to stop a continuous run at the next iteration boundary.
#: A file rather than a signal because it works from any shell, any session, and needs
#: no pid: the operator should never have to find a process to stop the loop politely.
STOP_SENTINEL = "STOP"


def stop_requested(store: Path) -> bool:
    """True once the operator has asked the loop to wind down."""
    return (store / STOP_SENTINEL).exists()


def prune_anchor_generations(store: Path, *, keep: int = ANCHOR_GENERATIONS_KEPT,
                             current: Path | None = None) -> list[Path]:
    """Delete superseded anchor builds, never the one in use.

    Returns what was removed. The current anchor is excluded explicitly rather than by
    assuming it is the newest: promotion and pruning are separate steps, and an
    assumption that they stay in step is the kind that holds until it does not.
    """
    generations = sorted(store.glob("anchor-gen-*"))
    protected = {current.resolve()} if current else set()
    doomed = [g for g in generations[:-keep] if g.resolve() not in protected]
    for path in doomed:
        shutil.rmtree(path, ignore_errors=True)
    return doomed
