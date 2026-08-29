#!/usr/bin/env python3
"""The concurrent pipeline, exercised with no GPU, no toolchain and no API key.

Every side effect `run_pool` performs is injected, so these tests are about the only
things concurrency can break here: whether the serialized tail is actually serialized,
whether formation actually overlaps (if it does not, the module buys nothing), whether
the iteration budget stays exact under contention, and whether a candidate whose base
moved is recorded as `superseded` rather than as a scientific failure.

TWO OF THESE TESTS CARRY THEIR OWN MUTATION TEST. A concurrency assertion that would
also pass on a sequential implementation, or an overlap detector that cannot detect
overlap, proves nothing -- and "a check that passes for the wrong reason" is the
failure mode this rebuild exists to close. So `TheSerializedTailIsMutuallyExclusive`
re-runs its detector against a lock that does not lock and requires it to FAIL, and
`FormationGenuinelyOverlaps` blocks every lane on a barrier that a sequential runner
can never release.

The `Worker` paths below are deliberately non-existent: nothing in `pipeline` touches
a filesystem, and a test that quietly started needing a real worktree would be
measuring something other than the pipeline.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import unittest

from autokernel.loop import bench, gates, loop, pipeline


def _sha(n: int) -> str:
    """A 40-char fake commit whose first 12 characters are unique to `n`.

    `Superseded` truncates both shas to 12, so ids that differ only after character 12
    would let the reason name two heads that read identically -- the message would be
    technically correct and operationally useless.
    """
    return f"{n:02d}" * 20


def _hypothesis(mechanism: str = "akm-q5-bit-deposit") -> loop.Hypothesis:
    return loop.Hypothesis(
        mechanism_id=mechanism, statement="bit-deposit the qh scatter",
        falsifier="no VGPR reduction below 64",
        target_surface="ggml/src/ggml-cuda/vecdotq.cuh",
        target_symbol="vec_dot_q5_0_q8_1_impl")


def _comparison(effect: float = 0.05, *, floor: float = 1.0) -> bench.Comparison:
    return bench.Comparison(
        surface="tg128", anchor_samples=[100.0],
        candidate_samples=[100.0 * (1 + effect)], effect=effect,
        estimator="median_over_median", pairs=5, noise_floor_pct=floor,
        residency={"invocations": 10, "resident": 10})


def _workers(count: int) -> list[pipeline.Worker]:
    return [pipeline.Worker(f"lane{index}",
                            Path(f"/nonexistent/ak-lane{index}"),
                            Path(f"/nonexistent/build-lane{index}"))
            for index in range(count)]


class _Champion:
    """The one piece of shared mutable state, faked: a ref that lanes advance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n = 0
        self.advances = 0

    def head(self) -> str:
        with self._lock:
            return _sha(self._n)

    def advance(self) -> str:
        with self._lock:
            self._n += 1
            self.advances += 1
            return _sha(self._n)


class _Planner:
    def __init__(self, hypothesis: loop.Hypothesis | None = None) -> None:
        self.hypothesis = hypothesis or _hypothesis()

    def propose(self, context):
        return self.hypothesis

    def author(self, hypothesis, context):
        return ("ggml/src/ggml-cuda/vecdotq.cuh",)


class _Critic:
    def review_hypothesis(self, hypothesis, context):
        return loop.Review(True)

    def review_patch(self, hypothesis, paths, context):
        return loop.Review(True)


class _Recorder:
    """A deliberately NON-atomic recorder.

    `run_pool` calls `record` under its own lock. If it ever stops doing so, the
    read-modify-write below loses counts across the yield point -- which is the whole
    point of instrumenting it this way rather than with a `list.append`, since
    `list.append` is atomic in CPython and would pass either way.
    """

    def __init__(self) -> None:
        self.rows: list[loop.Outcome] = []
        self.count = 0

    def __call__(self, outcome: loop.Outcome) -> None:
        seen = self.count
        time.sleep(0)          # a scheduler yield point between the read and the write
        self.count = seen + 1
        self.rows.append(outcome)


def _drive(*, workers, iterations, champion=None, planner=None, critic=None,
           gate=None, measure=None, commit=None, reset=None, record=None,
           on_step=None, tail=None):
    """Run the pool with harmless defaults. Every argument is an injection point."""
    champion = champion or _Champion()
    shared_planner = planner or _Planner()
    shared_critic = critic or _Critic()

    def default_gate(worker):
        return lambda hypothesis, paths: (True, [gates.Verdict("compile", True)])

    def default_measure(worker):
        return lambda hypothesis, paths: _comparison(0.02, floor=3.452)

    def default_commit(worker, hypothesis, paths, comparison):
        return champion.advance()

    def default_reset(worker):
        return champion.head()

    outcomes = pipeline.run_pool(
        workers=workers,
        make_planner=lambda worker: shared_planner,
        make_critic=lambda worker: shared_critic,
        build_context=dict,
        make_gate=gate or default_gate,
        make_measure=measure or default_measure,
        commit=commit or default_commit,
        champion_head=champion.head,
        reset_to_champion=reset or default_reset,
        record=record or (lambda outcome: None),
        iterations=iterations,
        on_step=on_step,
        tail=tail)
    return outcomes, champion


class _Overlap:
    """Counts how many callers are inside a region at once, and the peak."""

    def __init__(self) -> None:
        self._guard = threading.Lock()   # guards the OBSERVATION, never the region
        self.live = 0
        self.peak = 0

    def enter(self) -> None:
        with self._guard:
            self.live += 1
            self.peak = max(self.peak, self.live)

    def leave(self) -> None:
        with self._guard:
            self.live -= 1


class _NotALock:
    """A lock-shaped object that does not lock. Used to mutation-test the detector."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _WatchedLock:
    """A real lock that can say whether it is currently held."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.held = False

    def __enter__(self):
        self._lock.acquire()
        self.held = True
        return self

    def __exit__(self, *_exc):
        self.held = False
        self._lock.release()
        return False


# ---------------------------------------------------------------- (a) exclusion


class TheSerializedTailIsMutuallyExclusive(unittest.TestCase):
    """The property the whole design rests on.

    The build takes 64 of the 88 lane cores, the op oracle and the A/B need the one
    MI210, and the commit advances a branch. Two lanes inside that region at once is
    not slow, it is wrong: it is two builds fighting for the same cores, two
    workloads on a device claimed for one, and two writers on one ref.
    """

    def _hammer(self, tail, *, threads=8, calls=25):
        overlap = _Overlap()
        spans: list[tuple[float, float]] = []
        spans_lock = threading.Lock()

        def work():
            overlap.enter()
            started = time.monotonic()
            time.sleep(0.001)
            ended = time.monotonic()
            overlap.leave()
            with spans_lock:
                spans.append((started, ended))
            return None

        def lane():
            for _ in range(calls):
                tail.call(_sha(0), work)

        runners = [threading.Thread(target=lane, name=f"t{i}") for i in range(threads)]
        for runner in runners:
            runner.start()
        for runner in runners:
            runner.join(timeout=60)
        return overlap, spans

    def test_no_two_tail_calls_are_ever_inside_the_tail_at_once(self):
        tail = pipeline.SerializedTail(lambda: _sha(0))
        overlap, spans = self._hammer(tail)
        self.assertEqual(overlap.peak, 1,
                         f"{overlap.peak} lanes were inside the serialized tail at "
                         f"once; the build, the device and the champion ref all "
                         f"assume exactly one")
        self.assertEqual(len(spans), 8 * 25, "every call must have completed")

    def test_no_two_tail_spans_overlap_in_time(self):
        """A second, independent instrument: intervals, not a counter.

        A counter and a pair of timestamps fail differently, and agreeing is worth
        more than either alone.
        """
        tail = pipeline.SerializedTail(lambda: _sha(0))
        _, spans = self._hammer(tail, threads=6, calls=15)
        spans.sort()
        for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(
                first_end, second_start,
                "two tail spans overlap in wall time: the tail is not serialized")

    def test_the_overlap_detector_actually_detects_overlap(self):
        """Mutation test. Without this, every assertion above could be vacuous.

        Same instrument, same threads, a lock that does not lock. If the detector
        cannot see overlap HERE, its silence above means nothing.
        """
        tail = pipeline.SerializedTail(lambda: _sha(0), lock=_NotALock())
        overlap, _ = self._hammer(tail, threads=8, calls=10)
        self.assertGreater(overlap.peak, 1,
                           "the detector reported no overlap even with the lock "
                           "removed, so it cannot detect overlap at all")


# ---------------------------------------------------------------- (b) overlap


class FormationGenuinelyOverlaps(unittest.TestCase):
    """If formation does not overlap, this module buys nothing at all.

    Run 11: 63 of 75.1 minutes were four sequential high-effort actor calls per
    iteration. They are sequential WITHIN an iteration by construction, so the only
    place that latency can hide is ACROSS iterations.
    """

    def test_every_lane_is_inside_formation_at_the_same_moment(self):
        """This test CANNOT pass on a sequential runner.

        Each lane blocks in `propose` on a barrier that only releases when all four
        have arrived. One-lane-at-a-time execution never releases it: the barrier
        breaks on its timeout and `broke` is non-empty.
        """
        parties = 4
        barrier = threading.Barrier(parties)
        overlap = _Overlap()
        broke: list[str] = []

        class _Blocking(_Planner):
            def propose(self, context):
                overlap.enter()
                try:
                    barrier.wait(timeout=20)
                except threading.BrokenBarrierError:
                    broke.append("barrier never filled")
                finally:
                    overlap.leave()
                return self.hypothesis

        outcomes, _ = _drive(workers=_workers(parties), iterations=parties,
                             planner=_Blocking())

        self.assertEqual(broke, [],
                         f"only some of the {parties} lanes reached formation before "
                         f"the barrier timed out: formation is running one lane at a "
                         f"time, which is the sequential loop with extra machinery")
        self.assertEqual(overlap.peak, parties,
                         f"peak concurrent lanes in formation was {overlap.peak}, "
                         f"expected {parties}")
        self.assertEqual(len(outcomes), parties)

    def test_formation_overlaps_while_the_tail_stays_exclusive(self):
        """Both halves of the split at once -- the claim the module actually makes."""
        parties = 3
        barrier = threading.Barrier(parties)
        formation = _Overlap()
        tail_overlap = _Overlap()
        broke: list[str] = []

        class _Blocking(_Planner):
            def propose(self, context):
                formation.enter()
                try:
                    barrier.wait(timeout=20)
                except threading.BrokenBarrierError:
                    broke.append("barrier never filled")
                finally:
                    formation.leave()
                return self.hypothesis

        def gate(worker):
            def run(hypothesis, paths):
                tail_overlap.enter()
                time.sleep(0.002)
                tail_overlap.leave()
                return True, [gates.Verdict("compile", True)]
            return run

        _drive(workers=_workers(parties), iterations=parties,
               planner=_Blocking(), gate=gate)

        self.assertEqual(broke, [], "formation did not overlap")
        self.assertEqual(formation.peak, parties)
        self.assertEqual(tail_overlap.peak, 1,
                         "the tail overlapped: the build, the device and the "
                         "champion ref each assume exactly one lane")


# ---------------------------------------------------------------- (c) budget


class TheBudgetIsExact(unittest.TestCase):
    """`--iterations N` must mean N, not N +/- the number of lanes.

    An over-run spends device time nobody authorised; an under-run silently shortens
    a campaign. Both are invisible in a summary that only prints what it produced.

    SCOPE. These tests establish exactness for injections that RETURN. They do not
    cover an injection that RAISES anything other than `Superseded`: such an exception
    escapes `lane`, the thread dies with a traceback on stderr, and the budget draw it
    already made produces no outcome. Measured 2026-08-29: a `commit` raising on three
    of twenty iterations returned 17 outcomes for a budget of 20, silently. Closing
    that is a change to `run_pool`, so it is reported rather than asserted here.
    """

    def test_n_iterations_across_more_lanes_yields_exactly_n_outcomes(self):
        for lanes, iterations in ((4, 10), (3, 7), (8, 13)):
            with self.subTest(lanes=lanes, iterations=iterations):
                recorder = _Recorder()
                outcomes, _ = _drive(workers=_workers(lanes), iterations=iterations,
                                     record=recorder)
                self.assertEqual(len(outcomes), iterations,
                                 f"{lanes} lanes produced {len(outcomes)} outcomes "
                                 f"for a budget of {iterations}")
                self.assertEqual(recorder.count, iterations)

    def test_the_budget_is_exact_under_contention_in_the_tail(self):
        """The interesting case: lanes queueing on the tail, not running free."""
        def slow_gate(worker):
            def run(hypothesis, paths):
                time.sleep(0.003)
                return True, [gates.Verdict("compile", True)]
            return run

        outcomes, _ = _drive(workers=_workers(6), iterations=20, gate=slow_gate)
        self.assertEqual(len(outcomes), 20)

    def test_more_lanes_than_iterations_leaves_some_lanes_with_nothing(self):
        """A lane that draws nothing must exit, not spin and not steal."""
        claimed: list[str] = []
        claimed_lock = threading.Lock()

        def reset(worker):
            with claimed_lock:
                claimed.append(worker.name)
            return _sha(0)

        outcomes, _ = _drive(workers=_workers(8), iterations=2, reset=reset)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(len(claimed), 2,
                         "a lane drew from the budget without producing an outcome")
        self.assertLessEqual(len(set(claimed)), 2,
                             "more lanes started an iteration than the budget allows")

    def test_a_zero_budget_starts_no_iteration_at_all(self):
        outcomes, champion = _drive(workers=_workers(4), iterations=0)
        self.assertEqual(outcomes, [])
        self.assertEqual(champion.advances, 0)

    def test_the_shared_budget_hands_out_exactly_its_count(self):
        """`Budget` on its own, hammered: the primitive the exactness rests on."""
        budget = pipeline.Budget(500)
        taken: list[bool] = []
        taken_lock = threading.Lock()

        def lane():
            mine = 0
            while budget.take():
                mine += 1
            with taken_lock:
                taken.append(mine)

        runners = [threading.Thread(target=lane) for _ in range(12)]
        for runner in runners:
            runner.start()
        for runner in runners:
            runner.join(timeout=60)
        self.assertEqual(sum(taken), 500,
                         f"12 lanes drew {sum(taken)} times from a budget of 500")
        self.assertEqual(budget.remaining, 0)


# ---------------------------------------------------------------- (d) superseded


class ASupersededCandidateIsNotAScientificFailure(unittest.TestCase):
    """A moved base is a scheduling fact. Recording it as a null teaches the planner
    that a live mechanism was tested and found wanting, which is the same class of
    error as a fabricated refusal -- and it is durable, because the archive is the
    planner's memory.
    """

    def _superseded_outcome(self):
        champion = _Champion()

        class _Advancing(_Planner):
            def author(self, hypothesis, context):
                # Another lane's keep lands between this lane's reset and its tail.
                champion.advance()
                return ("ggml/src/ggml-cuda/vecdotq.cuh",)

        outcomes, _ = _drive(workers=_workers(1), iterations=1,
                             champion=champion, planner=_Advancing())
        self.assertEqual(len(outcomes), 1)
        return outcomes[0]

    def test_the_status_is_exactly_superseded(self):
        outcome = self._superseded_outcome()
        self.assertEqual(outcome.status, "superseded")

    def test_it_is_not_recorded_as_a_measurement_or_a_refusal(self):
        outcome = self._superseded_outcome()
        for wrong in ("measured_null", "refused_at_formation", "kept",
                      "planner_transient"):
            self.assertNotEqual(
                outcome.status, wrong,
                f"a candidate whose base moved was recorded as {wrong}: the science "
                f"never happened, and this row will be read back as if it had")
        self.assertIsNone(outcome.comparison,
                          "a superseded candidate was never measured")

    def test_the_reason_names_both_the_base_and_the_current_head(self):
        outcome = self._superseded_outcome()
        reason = " | ".join(outcome.reasons)
        self.assertIn(_sha(0)[:12], reason,
                      "the reason must name the base the lane authored against")
        self.assertIn(_sha(1)[:12], reason,
                      "the reason must name the head that superseded it")
        self.assertNotEqual(_sha(0)[:12], _sha(1)[:12],
                            "the fixture's two heads must be distinguishable after "
                            "the 12-character truncation, or this test is vacuous")

    def test_the_row_written_to_durable_memory_carries_the_status_and_reason(self):
        row = self._superseded_outcome().to_attempt()
        self.assertEqual(row["status"], "superseded")
        self.assertIn("champion is now", row["reason"])

    def test_the_tail_counts_supersessions_for_the_operator(self):
        champion = _Champion()

        class _Advancing(_Planner):
            def author(self, hypothesis, context):
                champion.advance()
                return ("a.cu",)

        tail = pipeline.SerializedTail(champion.head)
        _drive(workers=_workers(1), iterations=3, champion=champion,
               planner=_Advancing(), tail=tail)
        self.assertEqual(tail.superseded, 3,
                         "every supersession must be visible as a rate, not just as "
                         "individual rows: a high rate means the lane count is wrong")


# ---------------------------------------------------------------- (e) the race


class TheStalenessCheckHappensInsideTheLock(unittest.TestCase):
    """The check is only meaningful where the ref cannot move under it.

    Outside the lock the sequence is: read head, ref advances, enter tail, measure
    against a base that is now two commits behind. The measurement would be of a tree
    nobody ever built, reported against an anchor it never saw.
    """

    def test_the_champion_ref_is_only_ever_read_with_the_tail_lock_held(self):
        """The property itself, asserted directly rather than raced for.

        The race below is corroboration, but as an instrument it is weak: it catches a
        check moved outside the lock only when a peer's commit happens to land in the
        gap, which was 2 runs in 12 when it was measured. This one is deterministic --
        it asks the lock, at the moment the ref is read, whether it is held.
        """
        lock = _WatchedLock()
        held_at_each_read: list[bool] = []

        def head() -> str:
            held_at_each_read.append(lock.held)
            return _sha(0)

        tail = pipeline.SerializedTail(head, lock=lock)
        tail.call(_sha(0), lambda: "work")
        tail.call(_sha(0), lambda: "work")

        self.assertEqual(len(held_at_each_read), 2,
                         "the champion ref was not read once per tail call, so the "
                         "staleness check is not running at all")
        self.assertTrue(
            all(held_at_each_read),
            "the champion ref was read WITHOUT the tail lock held: between that read "
            "and entering the tail, a peer's keep can advance the branch, and the "
            "candidate is then built, gated or measured against a base it never saw")

    def _race(self, lanes=5, iterations=16):
        champion = _Champion()
        observed: list[tuple[str, str]] = []
        observed_lock = threading.Lock()
        kept = []

        # `make_measure` is not handed the base, so capture it at the reset instead.
        bases: dict[str, str] = {}
        bases_lock = threading.Lock()

        def reset(worker):
            head = champion.head()
            with bases_lock:
                bases[worker.name] = head
            return head

        def measure_for(worker):
            def run(hypothesis, paths):
                with bases_lock:
                    base = bases[worker.name]
                with observed_lock:
                    observed.append((base, champion.head()))
                time.sleep(0.001)
                return _comparison(0.05, floor=1.0)
            return run

        def commit(worker, hypothesis, paths, comparison):
            # A long hold widens the window a peer could slip through. In the real
            # loop the tail is held for a whole build and A/B -- minutes, not
            # milliseconds -- so a short hold here understates the race badly.
            time.sleep(0.010)
            head = champion.advance()
            kept.append(head)
            return head

        outcomes, _ = _drive(workers=_workers(lanes), iterations=iterations,
                             champion=champion, reset=reset, measure=measure_for,
                             commit=commit)
        return outcomes, observed, kept

    def test_no_lane_ever_measures_against_a_base_it_did_not_author_on(self):
        outcomes, observed, kept = self._race()
        for base, head_at_measure in observed:
            self.assertEqual(
                base, head_at_measure,
                f"a lane measured against {head_at_measure[:12]} while its patch was "
                f"authored on {base[:12]}: the staleness check ran outside the lock")
        self.assertEqual(len(outcomes), 16, "the budget must still be exact")

    def test_the_race_this_test_constructs_actually_happens(self):
        """Guards the test above against passing because nothing raced.

        If no lane is ever superseded, `observed` is trivially consistent and the
        assertion proves nothing about the lock.
        """
        outcomes, observed, kept = self._race()
        statuses = [outcome.status for outcome in outcomes]
        self.assertGreater(statuses.count("superseded"), 0,
                           "no lane was superseded, so the race never occurred and "
                           "the staleness assertion is vacuous")
        self.assertGreater(len(kept), 0, "no lane ever advanced the champion")
        self.assertGreater(len(observed), 0, "no lane ever reached a measurement")

    def test_the_champion_never_advances_twice_from_the_same_base(self):
        """Two lanes advancing the ref from one base is the lost-commit race."""
        champion = _Champion()
        advanced_from: list[str] = []
        advanced_lock = threading.Lock()

        def commit(worker, hypothesis, paths, comparison):
            with advanced_lock:
                advanced_from.append(champion.head())
            time.sleep(0.001)
            return champion.advance()

        def measure(worker):
            return lambda hypothesis, paths: _comparison(0.05, floor=1.0)

        _drive(workers=_workers(5), iterations=15, champion=champion,
               measure=measure, commit=commit)
        self.assertEqual(len(advanced_from), len(set(advanced_from)),
                         "the champion was advanced twice from the same base: one of "
                         "those two keeps is not on the branch it claims to be on")


# ---------------------------------------------------------------- (f) recording


class RecordingIsThreadSafe(unittest.TestCase):
    """The outcomes list and `record` are the run's only outputs besides the branch.

    A lost outcome is a measurement the campaign paid for and cannot read back; a
    duplicated one inflates the history the planner reads as evidence.
    """

    def _unique_run(self, lanes=8, iterations=200):
        counter = {"n": 0}
        counter_lock = threading.Lock()
        recorder = _Recorder()

        class _Unique(_Planner):
            def propose(self, context):
                with counter_lock:
                    counter["n"] += 1
                    index = counter["n"]
                return _hypothesis(f"akm-{index:04d}")

        outcomes, _ = _drive(workers=_workers(lanes), iterations=iterations,
                             planner=_Unique(), record=recorder)
        return outcomes, recorder

    def test_no_outcome_is_lost_or_duplicated(self):
        outcomes, recorder = self._unique_run()
        ids = [outcome.hypothesis.mechanism_id for outcome in outcomes]
        self.assertEqual(len(ids), 200, f"{len(ids)} outcomes for a 200 budget")
        self.assertEqual(len(set(ids)), 200,
                         f"{200 - len(set(ids))} outcomes were duplicated")

    def test_every_outcome_reaches_record_exactly_once(self):
        outcomes, recorder = self._unique_run()
        self.assertEqual(recorder.count, len(outcomes),
                         f"record ran {recorder.count} times for {len(outcomes)} "
                         f"outcomes: the non-atomic counter lost updates, so record "
                         f"is being called concurrently")
        self.assertEqual([id(o) for o in recorder.rows], [id(o) for o in outcomes],
                         "the recorded rows and the returned list disagree")

    def test_the_recorder_used_here_can_actually_lose_updates(self):
        """Mutation test for the instrument in the test above."""
        recorder = _Recorder()
        outcome = loop.Outcome("measured_null")

        def hammer():
            for _ in range(300):
                recorder(outcome)

        runners = [threading.Thread(target=hammer) for _ in range(12)]
        for runner in runners:
            runner.start()
        for runner in runners:
            runner.join(timeout=60)
        self.assertLess(recorder.count, 3600,
                        "the unsynchronised recorder did not lose a single update, "
                        "so it cannot detect an unsynchronised caller either")


# ---------------------------------------------------------------- (g) reporting


class ReportingMustNeverKillALane(unittest.TestCase):
    """Same contract as the sequential loop: a heartbeat is not allowed to be fatal.

    `on_step` publishes status and stamps phase timing. In a pool it also shells out
    to git for the champion head, which is exactly the kind of call that fails
    transiently -- and a dead lane silently shrinks the campaign.
    """

    def test_a_raising_on_step_does_not_shorten_the_run(self):
        def explode(worker_name, label):
            raise RuntimeError(f"status write failed for {worker_name} at {label}")

        outcomes, _ = _drive(workers=_workers(4), iterations=10, on_step=explode)
        self.assertEqual(len(outcomes), 10,
                         "a raising on_step killed lanes: the budget was drawn but "
                         "the iterations produced nothing")
        self.assertEqual({outcome.status for outcome in outcomes}, {"measured_null"})

    def test_on_step_is_told_which_lane_is_reporting(self):
        """A per-phase timing that cannot say WHICH lane is a global mark again."""
        seen: list[tuple[str, str]] = []
        seen_lock = threading.Lock()

        def note(worker_name, label):
            with seen_lock:
                seen.append((worker_name, label))

        _drive(workers=_workers(3), iterations=6, on_step=note)
        names = {worker_name for worker_name, _ in seen}
        self.assertTrue(names, "on_step was never called")
        self.assertTrue(names <= {"lane0", "lane1", "lane2"}, f"unexpected lanes {names}")
        self.assertIn("measuring A/B on the device", {label for _, label in seen})

    def test_the_pool_runs_with_no_on_step_at_all(self):
        outcomes, _ = _drive(workers=_workers(3), iterations=5, on_step=None)
        self.assertEqual(len(outcomes), 5)


# ------------------------------------------------- provisioning (Task 2 support)


class WorkerProvisioningIsPlannedBeforeItIsExecuted(unittest.TestCase):
    """Two lanes sharing a build directory is a silent correctness failure: lane A
    cmake-configures it, lane B configures it differently, and A's `llama-bench`
    binary is B's. The plan is therefore checkable without touching git -- and
    `provision` refuses to run `git worktree add` unless explicitly told to.
    """

    def test_the_plan_gives_every_lane_its_own_tree_and_build_directory(self):
        from autokernel.loop import pool
        planned = pool.provision(4, execute=False)
        self.assertEqual(len(planned), 4)
        self.assertEqual(len({worker.worktree for worker in planned}), 4,
                         "two lanes share a worktree")
        self.assertEqual(len({worker.build_dir for worker in planned}), 4,
                         "two lanes share a build directory")
        self.assertEqual(len({worker.name for worker in planned}), 4)

    def test_the_plan_never_reuses_the_champion_tree_as_a_lane(self):
        from autokernel.loop import pool
        planned = pool.provision(3, execute=False)
        for worker in planned:
            self.assertNotEqual(worker.worktree, pool.CHAMPION_TREE,
                                "a lane pointed at the champion tree: git refuses to "
                                "check one branch out twice, and a lane that edited "
                                "it would be editing the champion in place")

    def test_a_pool_refuses_lanes_that_would_collide(self):
        from autokernel.loop import pool
        shared = pipeline.Worker("a", Path("/nonexistent/t"), Path("/nonexistent/b"))
        twin = pipeline.Worker("b", Path("/nonexistent/t2"), Path("/nonexistent/b"))
        with self.assertRaises(ValueError) as caught:
            pool.check_lanes_are_disjoint([shared, twin])
        self.assertIn("build", str(caught.exception))

    def test_the_commit_message_names_the_mechanism_and_the_measured_effect(self):
        from autokernel.loop import pool
        message = pool.commit_message(_hypothesis("akm-x"), _comparison(0.041))
        self.assertIn("akm-x", message)
        self.assertIn("+4.100%", message)
        self.assertIn("tg128", message)


if __name__ == "__main__":
    unittest.main()


class DistinctAttemptsMustNotCollapse(unittest.TestCase):
    """`record` silently dropped attempts that shared a status, mechanism and reason.

    `_attempt_id` hashed `attempt.get("turn")`, and `Outcome.to_attempt()` never emits
    a `turn` key -- it was always None, so the identity collapsed to
    (campaign, status, mechanism, reason). Two `planner_transient` rows 40 minutes
    apart, both reading "authoring returned no changed paths", hashed identically; the
    second was dropped and `record` returned False with nobody reading the result.

    The planner reads this store back. A history missing its own repetitions is
    exactly the blindness the store exists to remove -- and concurrency makes it
    acute, because repetitive statuses are the ones several lanes emit at once.
    """

    def _record(self, root, at):
        from autokernel.loop import archive
        return archive.record(
            root, {"status": "planner_transient",
                   "reason": "authoring returned no changed paths"},
            epoch="e" * 64, recorded_at=at, campaign_id="ak-loop")

    def _rows(self, root):
        import sqlite3
        return list(sqlite3.connect(root / "experiments.db").execute(
            "SELECT COUNT(*) FROM experiments"))[0][0]

    def test_two_attempts_with_identical_text_are_both_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, "2026-08-29T10:00:00Z")
            self._record(root, "2026-08-29T10:40:00Z")
            self.assertEqual(self._rows(root), 2,
                             "a repeated transient is a real event, not a duplicate")

    def test_the_idempotency_it_was_protecting_still_holds(self):
        """A resumed loop re-recording the SAME attempt must not inflate history."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(root, "2026-08-29T10:40:00Z")
            added = self._record(root, "2026-08-29T10:40:00Z")
            self.assertFalse(added)
            self.assertEqual(self._rows(root), 1)
