#!/usr/bin/env python3
"""Run several iterations concurrently, with one serialized tail.

WHY
---
Run 11 spent 75.1 minutes on 10 iterations and 11.9 of them on the device -- 15.8%.
The other 63 minutes were four sequential `gpt-5.6-sol` calls per iteration at high
reasoning effort, plus a build and the op oracle. Those four calls are inherently
sequential WITHIN an iteration: each one consumes the previous one's output. So the
only way to hide that latency is to overlap ACROSS iterations.

THE SPLIT
---------
Concurrent:  propose, critic pass 1, author, critic pass 2 -- pure actor latency, no
             device, no build lane, no champion write.
Serialized:  build, op oracle, A/B, champion commit -- all four contend. The build
             takes 64 of the 88 lane cores, the oracle and the A/B need the one GPU,
             and the commit advances a branch.

Since the serialized tail is ~16% of wall time, three or four workers saturate it
before it becomes the constraint. This buys throughput without weakening one statistic
-- which matters, because five of the six instrument defects found on 2026-08-29 were
forms of measuring less, and going faster by measuring less is the failure this whole
rebuild exists to correct.

THE CHAMPION IS THE SHARED MUTABLE THING
----------------------------------------
`ak-loop-tree` is a git WORKTREE of the frozen production clone, on branch
`ak/loop-champion-20260828`. Git refuses to check one branch out in two worktrees, so
each worker runs DETACHED at the champion commit and only the serialized tail advances
the branch.

That creates the one genuine hazard here: a worker authors against champion C0 while
another worker's keep advances the champion to C1. Measuring the first candidate
against C1 is wrong -- it was never built on it -- and against C0 is stale, because a
kept patch is supposed to compound. So a candidate whose base has moved is recorded as
`superseded` and its hypothesis returns to the pool with its reasons intact. It is not
a failure of the science and must never be recorded as one.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import traceback
from pathlib import Path
import threading
from typing import Any, Callable, Sequence

from . import loop as loop_mod

#: Beyond this, lanes queue on the serialized tail rather than adding throughput.
#: Derived from measurement, not chosen. Run 13 recorded per-phase wall time:
#:
#:     proposing a hypothesis   62.7 min  54.4%   concurrent
#:     critic pass 1            21.4 min  18.6%   concurrent
#:     authoring                 9.4 min   8.2%   concurrent
#:     critic pass 2             6.1 min   5.3%   concurrent
#:     measuring A/B            10.4 min   9.0%   SERIALIZED
#:     building and gating       5.2 min   4.5%   SERIALIZED
#:
#: The tail is 13.5% of an iteration, so 1/0.135 = 7.4 lanes saturate it. Seven keeps
#: the device busy without a queue forming in front of it. The planner alone is 54%,
#: which is why overlapping ACROSS iterations is the only lever: those four calls are
#: sequential WITHIN an iteration by construction.
DEFAULT_WORKERS = 7

#: Consecutive errored iterations ACROSS the whole pool -- not per lane -- before the
#: run gives up. Moved here from `loop.py` when the sequential CLI path was deleted:
#: an unbraked `--iterations 0` pool against a systematically broken setup (every
#: build failing, say) would spin forever, which is run-18-shaped waste with no brake.
#: One-off faults reset the count. Refusals, nulls, supersessions, transients and
#: mid-formation stops are NOT errors and never trip it -- they are the loop working.
MAX_CONSECUTIVE_ERRORS = 3


class Superseded(loop_mod.TailRefused):
    """The champion advanced while this candidate was being authored.

    Extends the loop's TailRefused so `iterate` catches it where the hypothesis is
    still in scope and carries it out (as `self.hypothesis`; both heads are named in
    the message). A formed-but-unmeasured candidate is a QUEUE ENTRY, not a loss: its
    patch may still help against the champion that displaced it, and the planner is
    told to look at these first.
    """

    hypothesis = None


@dataclass
class Worker:
    """One concurrent lane: its own worktree, its own build directory."""
    name: str
    worktree: Path
    build_dir: Path


@dataclass
class Budget:
    """A shared iteration count, or none at all.

    `remaining=None` means run until told to stop. That is the difference between a
    tool someone runs and a loop that works: the previous shape exited after N
    iterations, so every continuation needed a human to start it again.

    `should_stop` is checked on every draw, so a stop is honoured at the next
    iteration boundary rather than mid-measurement -- a lane holding the device
    finishes what it started, publishes, and only then declines to take more.
    """
    remaining: int | None
    should_stop: Callable[[], bool] | None = None
    drawn: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self) -> bool:
        with self._lock:
            if self.should_stop is not None and self.should_stop():
                return False
            if self.remaining is not None:
                if self.remaining <= 0:
                    return False
                self.remaining -= 1
            self.drawn += 1
            return True


class SerializedTail:
    """Build, oracle, measure and commit -- one worker at a time.

    Also the arbiter of champion staleness, because that decision can only be made
    while holding the lock: any check made outside it is immediately racy.
    """

    def __init__(self, champion_head: Callable[[], str],
                 lock: threading.Lock | None = None) -> None:
        self._lock = lock or threading.Lock()
        self._champion_head = champion_head
        self.tail_seconds = 0.0
        self.superseded = 0

    def _check_base(self, base_head: str) -> None:
        """Caller MUST hold the lock: a staleness check made outside it is racy."""
        current = self._champion_head()
        if current != base_head:
            self.superseded += 1
            raise Superseded(
                f"formed against {base_head[:12]}, champion advanced to "
                f"{current[:12]} before it could be measured — NOT refuted, and not "
                f"a failure of the science: reconsider it against the new champion")

    @contextmanager
    def session(self, base_head: str, clock=None):
        """Hold the tail for ONE candidate attempt: build, oracle, A/B and commit.

        Three separate acquisitions let a peer's keep land between them and turn an
        already-measured candidate into a stale one -- probed at 4 lanes, 20 A/B runs
        executed and 5 recorded a comparison. Held for the attempt, released between
        patch rounds so authoring never serializes.
        """
        import time as _time
        clock = clock or _time.monotonic
        with self._lock:
            self._check_base(base_head)
            started = clock()
            try:
                yield
            finally:
                self.tail_seconds += clock() - started

def run_pool(*, workers: Sequence[Worker], make_planner, make_critic, build_context,
             make_gate, make_measure, commit, champion_head: Callable[[], str],
             reset_to_champion: Callable[[Worker], str],
             record: Callable[[loop_mod.Outcome], None],
             iterations: int | None,
             on_step: Callable[[str, str], None] | None = None,
             tail: SerializedTail | None = None,
             should_stop: Callable[[], bool] | None = None) -> list[loop_mod.Outcome]:
    """Drive `iterations` iterations across `workers` concurrent lanes.

    Every side effect is injected, exactly as in `loop.iterate`, so the whole pool is
    testable with no GPU, no build toolchain and no API key.
    """
    budget = Budget(iterations, should_stop=should_stop)
    tail = tail or SerializedTail(champion_head)
    outcomes: list[loop_mod.Outcome] = []
    outcomes_lock = threading.Lock()
    aborted: list[BaseException] = []
    consecutive_errors = [0]

    def keep(outcome: loop_mod.Outcome) -> None:
        """Append, record, and arm THE breaker -- one place, under one lock.

        The count is pool-wide because the thing it detects is pool-wide: a broken
        SETUP errors every lane, while a fault that belongs to one hypothesis is
        interleaved with other lanes' healthy outcomes and resets the count. Tripping
        zeroes the budget so every lane stops at its next draw, and the `RunAborted`
        re-raised after the join makes the run exit non-zero, publishing why --
        exactly the anchor-guard abort's path.
        """
        with outcomes_lock:
            outcomes.append(outcome)
            record(outcome)
            # `lane_error` is the only error status left: `iteration_error` was the
            # deleted sequential driver's spelling of the same containment (R21-7).
            if outcome.status == "lane_error":
                consecutive_errors[0] += 1
                if consecutive_errors[0] >= MAX_CONSECUTIVE_ERRORS and not aborted:
                    budget.remaining = 0
                    aborted.append(loop_mod.RunAborted(
                        f"{consecutive_errors[0]} consecutive iterations failed "
                        f"across the pool (last: "
                        f"{outcome.reasons[0] if outcome.reasons else 'unknown'}); "
                        f"the setup is broken, not any one hypothesis — stopping "
                        f"rather than spending the budget re-proving it"))
            else:
                consecutive_errors[0] = 0

    def lane(worker: Worker) -> None:
        planner, critic = make_planner(worker), make_critic(worker)
        while budget.take():
            # INSIDE the try. This sat outside it, so a failure here killed the whole
            # thread rather than costing one iteration -- run 16 lost four of seven
            # lanes that way, silently, while the run carried on looking healthy at
            # reduced capacity. Everything an iteration does must be containable.
            try:
                base = reset_to_champion(worker)
            except Exception as exc:      # noqa: BLE001
                keep(loop_mod.Outcome(
                    "lane_error", None,
                    [f"lane {worker.name} could not reach the champion: "
                     f"{type(exc).__name__}: {exc}",
                     traceback.format_exc()[-1500:]]))
                continue

            # All three run INSIDE one `tail.session`, so they need no lock of their
            # own: the session is the atomic unit and the staleness check happens once,
            # at its start, while the lock is held.
            def gate(hypothesis, paths, _w=worker):
                return make_gate(_w)(hypothesis, paths)

            def measure(hypothesis, paths, _w=worker):
                return make_measure(_w)(hypothesis, paths)

            def commit_one(hypothesis, paths, comparison, _w=worker):
                return commit(_w, hypothesis, paths, comparison)

            def step(label, _w=worker):
                if on_step is not None:
                    on_step(_w.name, label)

            try:
                # `should_abandon` is the stop predicate itself: once a stop is asked,
                # a lane still FORMING abandons at its next stage boundary instead of
                # completing multi-minute actor calls for results nobody will use
                # (run 20 drained for ~50 min at 0% GPU that way). A lane holding the
                # tail is unaffected -- `iterate` never polls inside the tail session.
                outcome = loop_mod.iterate(
                    planner=planner, critic=critic, context=build_context(),
                    measure=measure, gate=gate, commit=commit_one, on_step=step,
                    tail_session=lambda _b=base: tail.session(_b),
                    should_abandon=should_stop)
            except Superseded as exc:
                # `iterate` already converted this into an Outcome carrying the
                # hypothesis; reaching here means it escaped before one was formed.
                outcome = loop_mod.Outcome(
                    "superseded", getattr(exc, "hypothesis", None), [str(exc)])
            except loop_mod.RunAborted as exc:
                # Run-ending, not lane-ending: the anchor is SHARED, so a refused anchor
                # voids every lane's next measurement. Zeroing the budget stops them all
                # at their next draw -- continuous runs too: `remaining` is no longer None.
                aborted.append(exc)
                budget.remaining = 0
                outcome = loop_mod.Outcome("run_aborted", None, [str(exc)])
            except Exception as exc:      # noqa: BLE001 -- deliberate, see below
                # A lane used to die on ANY exception that was not Superseded:
                # RatchetRefused, a git timeout, an OSError. The thread ended with a
                # traceback on stderr, the budget draw was lost, and nothing was
                # published -- probed, a commit raising on 3 of 20 iterations returned
                # 17 outcomes for a budget of 20. At seven lanes that loses work seven
                # times as fast. The traceback is recorded so it cannot hide.
                outcome = loop_mod.Outcome(
                    "lane_error", None,
                    [f"lane {worker.name}: {type(exc).__name__}: {exc}",
                     traceback.format_exc()[-1500:]])
            keep(outcome)

    threads = [threading.Thread(target=lane, args=(w,), name=w.name, daemon=True)
               for w in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if aborted:
        # Re-raised after every lane has published, so the run ends the way a crashed
        # run does -- `run.py` publishes `failed` -- rather than returning a normal
        # result list that reads as a completed run.
        raise aborted[0]
    return outcomes


__all__ = ["Budget", "DEFAULT_WORKERS", "MAX_CONSECUTIVE_ERRORS", "SerializedTail",
           "Superseded", "Worker", "run_pool"]
