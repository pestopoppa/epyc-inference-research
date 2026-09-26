"""Best-of-N concurrent authoring (`bestof.AuthorPanel`), with fake concurrent backends.

What is under test: N authors genuinely run AT THE SAME TIME (a barrier only a concurrent
runner can release), the first passing diff wins and the other call is ended through the
actor stop path (one test runs REAL child processes and checks the process group died),
the pool-budget math and its refusals, N=1 staying the single path byte for byte, the
scratch-worktree lifecycle on every exit path (allocated from the flow-level registry,
released, never pruned), and the per-author record on the outcome and in the metrics rows.

Most tests allocate through the REAL flow-level registry (`scratch.ScratchRegistry`); a
small double stands in where a test needs a registry shape the real one does not have
(an early `release`, a shared parent). The panel never creates, sweeps or removes a
worktree itself.
"""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tokenize
import unittest
from types import SimpleNamespace
from unittest import mock

from autokernel.loop import (actor_metrics, actors, archive, bestof, bench, gates, loop,
                             pipeline, scratch)
from autokernel.loop.loop import Abstain, ActorStopped, ActorTransient, Hypothesis, Review

TARGET = "ggml/src/kernel.c"
OTHER = "ggml/src/other.c"
HYP = Hypothesis("akh-bestof", "statement", "falsifier", f"{TARGET}: f()", "f")


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], input=input_bytes,
                          capture_output=True, timeout=120)
    if done.returncode:
        raise AssertionError(f"git {args}: {done.stderr.decode()}")
    return done.stdout.decode().strip()


def _worktrees(repo: Path) -> set[str]:
    listing = _git(repo, "worktree", "list", "--porcelain")
    return {line.split(" ", 1)[1] for line in listing.splitlines() if line.startswith("worktree ")}


class FakeScope:
    """One `registry.scope(...)`: marker-owned worktrees, released on exit."""

    def __init__(self, registry, name, release_method=True):
        self.registry, self.name, self.held = registry, name, []
        if not release_method:
            self.release = None      # type: ignore[assignment]

    def worktree(self, repo, base_commit, name):
        path = (self.registry.root / self.registry.layout(self.name, name)).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(Path(repo), "worktree", "add", "--detach", str(path), base_commit)
        self.held.append((Path(repo), path))
        self.registry.events.append(("create", str(path)))
        self.registry.counters["worktrees_created"] += 1
        return path

    def release(self, path):
        for repo, held in list(self.held):
            if held == Path(path):
                _git(repo, "worktree", "remove", "--force", str(held))
                self.held.remove((repo, held))
                self.registry.events.append(("release", str(held)))
                self.registry.counters["worktrees_removed"] += 1

    def close(self):
        for repo, held in list(self.held):
            _git(repo, "worktree", "remove", "--force", str(held))
            self.held.remove((repo, held))
            self.registry.events.append(("scope_exit_release", str(held)))
            self.registry.counters["worktrees_removed"] += 1


class FakeRegistry:
    def __init__(self, root: Path, *, free=True, release_method=True, shared_parent=False):
        self.root, self.free, self.release_method = Path(root), free, release_method
        self.events: list[tuple[str, str]] = []
        self.counters = {"worktrees_created": 0, "worktrees_removed": 0}
        self.ensure_calls: list[int] = []
        self.shared_parent = shared_parent

    def layout(self, scope_name, name):
        return f"{scope_name}/{name}" if self.shared_parent else f"{scope_name}/{name}/tree"

    @contextmanager
    def scope(self, kind, name):
        assert kind == "call"
        scope = FakeScope(self, name, self.release_method)
        self.events.append(("scope_open", name))
        try:
            yield scope
        finally:
            scope.close()
            self.events.append(("scope_close", name))

    def ensure_free(self, nbytes):
        self.ensure_calls.append(nbytes)
        return self.free

    def stats(self):
        return dict(self.counters)


class Fixture(unittest.TestCase):
    """A real repo, a detached lane worktree at `base`, a patch store."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = self.root / "repo"
        (self.repo / "ggml/src").mkdir(parents=True)
        (self.repo / TARGET).write_text("int f(void) { return 1; }\n")
        (self.repo / OTHER).write_text("int g(void) { return 2; }\n")
        _git(self.repo, "init", "-q")
        _git(self.repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
        _git(self.repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
        self.base = _git(self.repo, "rev-parse", "HEAD")
        self.workers = self.root / "workers"
        self.lane = self.workers / "lane0"
        _git(self.repo, "worktree", "add", "--detach", str(self.lane), self.base)
        self.store = self.root / "store"
        self.registry = self.real_registry()
        self.stop = threading.Event()

    def real_registry(self, min_free_bytes=0):
        return scratch.ScratchRegistry(self.root / "scratch", owner={"test": self.id()},
                                       min_free_bytes=min_free_bytes)

    def tearDown(self):
        self._tmp.cleanup()

    def panel(self, authors, *, specs="off,medium", registry=None, validator=None, **kw):
        specs = bestof.parse_authors(specs)
        return bestof.AuthorPanel(
            lane="lane0", specs=specs,
            make_author=lambda spec, ws, stop: authors[spec.thinking](spec, Path(ws), stop),
            scratch=registry or self.registry, budget=bestof.author_budget(len(specs)),
            validator=validator or bestof.integrity_validator,
            retain=lambda **patch: archive.retain_patch_bytes(self.store, **patch),
            should_stop=self.stop.is_set, **kw)

    def call(self, panel, solo=None, lane=None, context=None):
        records = []
        solo = solo or (lambda h, c: (_ for _ in ()).throw(AssertionError("solo called")))
        result = panel(HYP, context or {"k": "v"}, lane=lane or (self.lane, self.base),
                       solo=solo, record=records.append)
        return result, records

    def assert_no_scratch_left(self):
        root = str(Path(self.registry.root).resolve())
        left = {p for p in _worktrees(self.repo) if p.startswith(root)}
        self.assertEqual(left, set(), "a scratch worktree survived the panel")
        for kind in ("author", "worktrees"):
            home = Path(root) / kind
            self.assertFalse(home.exists() and any(home.iterdir()), f"{home} not empty")
        for kind, path in getattr(self.registry, "events", []):
            if kind == "create":
                self.assertFalse(Path(path).exists(), path)


class Editor:
    """An author that edits the target in ITS workspace and reports it."""

    def __init__(self, spec, ws, stop, *, text="int f(void) { return 7; }\n", path=TARGET,
                 barrier=None, delay=0.0, log=None, rows=None):
        self.spec, self.ws, self.stop = spec, ws, stop
        self.text, self.path, self.barrier, self.delay, self.log = text, path, barrier, delay, log
        self.rows = rows

    def author(self, hypothesis, context):
        if self.log is not None:
            self.log.append(("start", self.spec.label, time.monotonic()))
        if self.barrier is not None:
            self.barrier.wait(timeout=10)      # only a CONCURRENT runner releases it
        time.sleep(self.delay)
        (self.ws / self.path).write_text(self.text)
        if self.rows is not None:
            _write_metric_rows(self.ws, self.spec, **self.rows)
        if self.log is not None:
            self.log.append(("end", self.spec.label, time.monotonic()))
        return (self.path,)


class Blocker:
    """An author whose call runs until the panel's stop reaches it (the C22 path)."""

    def __init__(self, spec, ws, stop, *, barrier=None, log=None, edit=True):
        self.spec, self.ws, self.stop, self.barrier, self.log, self.edit = \
            spec, ws, stop, barrier, log, edit
        self.stopped = threading.Event()

    def author(self, hypothesis, context):
        if self.log is not None:
            self.log.append(("start", self.spec.label, time.monotonic()))
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        if self.edit:
            (self.ws / TARGET).write_text("int f(void) { return 99; } /* partial */\n")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.stop():
                self.stopped.set()
                if self.log is not None:
                    self.log.append(("stopped", self.spec.label, time.monotonic()))
                raise ActorStopped("stop asked during the actor call; group ended")
            time.sleep(0.01)
        raise AssertionError("the panel never stopped the losing author")


def _write_metric_rows(ws: Path, spec, *, steps, decoded):
    target = ws.parent / actor_metrics.REPLY_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    with open(target / actor_metrics.CALL_LOG_NAME, "a") as fh:
        fh.write(json.dumps({"schema": actor_metrics.METRICS_SCHEMA, "role": "author",
                             "wall_s": 1.5, "failure_class": None,
                             "opencode": {"totals": {"steps": steps, "decoded_tokens": decoded,
                                                     "compactions": 0}},
                             "budgets": {"thinking": spec.thinking}}) + "\n")
        fh.write(json.dumps({"schema": "vb-ak-seat.call.v1", "sealed": True}) + "\n")


# --------------------------------------------------------------------------- pool budget


class PoolBudgetMath(unittest.TestCase):

    def test_best_of_two_on_the_196608_pool(self):
        budget = bestof.author_budget(2)
        self.assertEqual((budget.context_limit, budget.output_limit, budget.compaction_at),
                         (90_112, 16_384, 73_728))
        self.assertEqual(budget.context_limit, (196_608 - 16_384) // 2)
        self.assertLessEqual(2 * budget.context_limit + budget.reserve, 196_608)

    def test_computed_from_n_and_the_pool_size(self):
        self.assertEqual(bestof.author_budget(1).context_limit, 180_224)
        self.assertEqual(bestof.author_budget(3).context_limit, 60_074)
        small = bestof.author_budget(2, pool_tokens=131_072)
        self.assertEqual(small.context_limit, (131_072 - 16_384) // 2)
        for n in (1, 2, 3):
            b = bestof.author_budget(n)
            self.assertLessEqual(n * b.context_limit + b.reserve, b.pool_tokens)

    def test_refuses_configs_that_exceed_the_pool(self):
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "compacts"):
            bestof.author_budget(4)       # 45,056 each: output 16,384 leaves < 32,768
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "outside"):
            bestof.author_budget(5)
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "outside"):
            bestof.author_budget(0)
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "reserve"):
            bestof.author_budget(2, pool_tokens=16_384)
        with self.assertRaisesRegex(bestof.PoolBudgetRefused, "compacts"):
            bestof.author_budget(2, pool_tokens=90_000)

    def test_parse_authors(self):
        specs = bestof.parse_authors("off,medium")
        self.assertEqual([(s.label, s.thinking) for s in specs],
                         [("a0-off", "off"), ("a1-medium", "medium")])
        with self.assertRaisesRegex(ValueError, "unknown"):
            bestof.parse_authors("off,high")
        with self.assertRaisesRegex(ValueError, "unknown"):
            bestof.parse_authors("off,medium", allowed=("default", "off"))
        with self.assertRaisesRegex(ValueError, "slots"):
            bestof.parse_authors("off,off,off,off,off")
        with self.assertRaises(ValueError):
            bestof.parse_authors("off,,medium")


class RunPyAuthorPlan(unittest.TestCase):
    """`--actor-authors` resolution: explicit values refuse, the default degrades."""

    def args(self, authors=None, workers=1, pool=196_608):
        return SimpleNamespace(actor_authors=authors, workers=workers, actor_pool_tokens=pool)

    def setUp(self):
        from autokernel.loop import run
        self.run = run
        self.with_medium = mock.patch.object(run.actor_opencode_config, "THINKING_CHOICES",
                                             ("default", "off", "medium"))

    def test_default_is_off_medium_when_the_build_has_medium(self):
        with self.with_medium:
            plan = self.run._author_plan(self.args(), "opencode")
        self.assertTrue(plan.panel)
        self.assertEqual([s.thinking for s in plan.specs], ["off", "medium"])
        # Asymmetric by thinking mode (lane/ak-authfail-20260926).
        self.assertEqual([(m.label, m.context_limit, m.output_limit)
                          for m in plan.budget.members],
                         [("a0-off", 65_536, 16_384), ("a1-medium", 114_688, 40_960)])
        self.assertIsNone(plan.budget.context_limit)
        self.assertEqual(self.run.DEFAULT_ACTOR_AUTHORS, "off,medium")

    def test_default_degrades_loudly_without_medium_and_explicit_refuses(self):
        with mock.patch.object(self.run.actor_opencode_config, "THINKING_CHOICES",
                               ("default", "off")):
            plan = self.run._author_plan(self.args(), "opencode")
            self.assertFalse(plan.panel)
            self.assertIn("unavailable", plan.note)
            with self.assertRaises(ValueError):
                self.run._author_plan(self.args("off,medium"), "opencode")

    def test_n1_and_single_are_the_single_path(self):
        with self.with_medium:
            self.assertFalse(self.run._author_plan(self.args("single"), "opencode").panel)
            self.assertFalse(self.run._author_plan(self.args("medium"), "opencode").panel)

    def test_several_lanes_or_a_non_opencode_planner(self):
        with self.with_medium:
            self.assertFalse(self.run._author_plan(self.args(workers=7), "opencode").panel)
            with self.assertRaisesRegex(ValueError, "workers"):
                self.run._author_plan(self.args("off,medium", workers=2), "opencode")
            self.assertFalse(self.run._author_plan(self.args(), "codex").panel)
            with self.assertRaisesRegex(ValueError, "opencode"):
                self.run._author_plan(self.args("off,medium"), "codex")
            with self.assertRaises(bestof.PoolBudgetRefused):
                self.run._author_plan(self.args("off,medium", pool=60_000), "opencode")


# --------------------------------------------------------------------------- the race


class ConcurrentRace(Fixture):

    def test_both_authors_run_concurrently(self):
        # A sequential runner deadlocks on the barrier (timeout -> BrokenBarrierError).
        barrier, log = threading.Barrier(2), []
        panel = self.panel({
            "off": lambda s, w, st: Editor(s, w, st, barrier=barrier, log=log, delay=0.2),
            "medium": lambda s, w, st: Editor(s, w, st, barrier=barrier, log=log, delay=0.2,
                                              text="int f(void) { return 8; }\n")})
        paths, records = self.call(panel)
        starts = {label: t for kind, label, t in log if kind == "start"}
        ends = {label: t for kind, label, t in log if kind == "end"}
        self.assertEqual(set(starts), {"a0-off", "a1-medium"})
        self.assertEqual(set(ends), {"a0-off", "a1-medium"})
        self.assertLess(max(starts.values()), min(ends.values()), "calls did not overlap")
        self.assertGreater(records[0]["overlap_s"], 0.0)
        self.assertEqual(tuple(paths), (TARGET,))
        self.assert_no_scratch_left()

    def test_first_passing_wins_and_the_other_is_cancelled(self):
        barrier, log, blockers = threading.Barrier(2), [], []

        def blocker(s, w, st):
            b = Blocker(s, w, st, barrier=barrier, log=log)
            blockers.append(b)
            return b

        panel = self.panel({"off": lambda s, w, st: Editor(s, w, st, barrier=barrier, log=log),
                            "medium": blocker})
        paths, records = self.call(panel)
        self.assertEqual(tuple(paths), (TARGET,))
        # The winner's diff is on the real lane.
        self.assertEqual((self.lane / TARGET).read_text(), "int f(void) { return 7; }\n")
        self.assertTrue(blockers[0].stopped.is_set())
        row = records[0]
        self.assertEqual((row["selection"], row["winner"]), ("first_passing", "a0-off"))
        members = {m["label"]: m for m in row["members"]}
        self.assertEqual(members["a0-off"]["result"], "won")
        self.assertEqual(members["a1-medium"]["result"], "cancelled")
        self.assertEqual(members["a1-medium"]["outcome"], "stopped")
        self.assertTrue(members["a0-off"]["validation"]["passed"])
        self.assert_no_scratch_left()

    def test_a_real_registry_releases_the_losers_whole_footprint_early(self):
        """With the run registry's `Scope.release`, a finished loser's tree, ak-check
        dir and home dir (its per-call opencode config) go at once, in that order,
        before the panel's call scope ends."""
        released = []
        real = scratch.Scope.release

        def spy(scope, path):
            released.append((scope.closed, Path(path)))
            return real(scope, path)
        panel = self.panel({"off": Editor, "medium": lambda s, w, st: Blocker(s, w, st)})
        with mock.patch.object(scratch.Scope, "release", spy):
            self.call(panel)
        loser = [p for _, p in released if "a1-medium" in str(p)]
        self.assertEqual([p.name for p in loser[:3]], ["tree", bestof.CHECK_DIR_NAME,
                                                       loser[2].name])
        self.assertEqual(loser[2], loser[0].parent)
        self.assertTrue(all(not closed for closed, _ in released))
        self.assert_no_scratch_left()

    def test_with_an_early_release_the_loser_goes_first(self):
        self.registry = FakeRegistry(self.root / "fake-scratch")
        panel = self.panel({"off": Editor, "medium": lambda s, w, st: Blocker(s, w, st)})
        self.call(panel)
        releases = [p for kind, p in self.registry.events if kind == "release"]
        self.assertEqual(len(releases), 2)
        self.assertIn("a1-medium", releases[0])      # released as soon as a0 won
        self.assert_no_scratch_left()

    def test_a_failing_first_finisher_does_not_win(self):
        # The fast author edits OUTSIDE the target surface; the slow one passes.
        panel = self.panel({
            "off": lambda s, w, st: Editor(s, w, st, path=OTHER, text="int g(void){return 3;}\n"),
            "medium": lambda s, w, st: Editor(s, w, st, delay=0.3)})
        paths, records = self.call(panel)
        row = records[0]
        self.assertEqual((row["selection"], row["winner"]), ("first_passing", "a1-medium"))
        members = {m["label"]: m for m in row["members"]}
        self.assertEqual(members["a0-off"]["result"], "lost")
        self.assertFalse(members["a0-off"]["validation"]["passed"])
        self.assertEqual((self.lane / OTHER).read_text(), "int g(void) { return 2; }\n")
        self.assertEqual((self.lane / TARGET).read_text(), "int f(void) { return 7; }\n")
        self.assert_no_scratch_left()

    def test_no_winner_picks_the_better_diff_by_the_existing_rules(self):
        # a0 edits nothing in the target (score 0); a1 edits the target AND a stray file
        # but reports only the target: the integrity screen refuses (score 1). a1 got
        # further, so its diff goes to the lane and the normal gates judge it.
        class Sloppy(Editor):
            def author(self, hypothesis, context):
                (self.ws / OTHER).write_text("int g(void) { return 5; }\n")
                return super().author(hypothesis, context)

        panel = self.panel({
            "off": lambda s, w, st: Editor(s, w, st, path=OTHER, text="int g(void){return 3;}\n"),
            "medium": lambda s, w, st: Sloppy(s, w, st, delay=0.1)})
        paths, records = self.call(panel)
        row = records[0]
        self.assertEqual((row["selection"], row["winner"]), ("best_failing", "a1-medium"))
        self.assertEqual(tuple(paths), (TARGET,))
        self.assertEqual((self.lane / OTHER).read_text(), "int g(void) { return 5; }\n")
        self.assert_no_scratch_left()

    def test_both_abstain_records_both(self):
        class Abstainer(Editor):
            def author(self, hypothesis, context):
                return Abstain(f"{self.spec.label} cannot")

        panel = self.panel({"off": Abstainer, "medium": Abstainer})
        result, records = self.call(panel)
        self.assertIsInstance(result, Abstain)
        self.assertIn("a0-off cannot", result.reason)
        self.assertIn("a1-medium cannot", result.reason)
        self.assertEqual(records[0]["selection"], "none")
        self.assertEqual({m["outcome"] for m in records[0]["members"]}, {"abstained"})
        self.assert_no_scratch_left()

    def test_both_transient_raises_the_provider_failure(self):
        class Flaky(Editor):
            def author(self, hypothesis, context):
                raise actors.ProviderTransient(f"{self.spec.label} 503")

        panel = self.panel({"off": Flaky, "medium": Flaky})
        with self.assertRaisesRegex(ActorTransient, "every author failed"):
            self.call(panel)
        self.assert_no_scratch_left()

    def test_a_run_stop_ends_every_author_and_the_round(self):
        barrier = threading.Barrier(3)

        def blocker(s, w, st):
            return Blocker(s, w, st, barrier=barrier)

        panel = self.panel({"off": blocker, "medium": blocker})
        stopper = threading.Thread(target=lambda: (barrier.wait(timeout=10), self.stop.set()))
        stopper.start()
        with self.assertRaises(ActorStopped):
            self.call(panel)
        stopper.join()
        self.assert_no_scratch_left()

    def test_later_rounds_seed_every_author_with_the_lane_diff(self):
        (self.lane / TARGET).write_text("int f(void) { return 1; } /* round 1 */\n")

        class Reviser(Editor):
            def author(self, hypothesis, context):
                seen = (self.ws / TARGET).read_text()
                assert "round 1" in seen, seen
                (self.ws / TARGET).write_text(seen.replace("round 1", "round 2"))
                return (TARGET,)

        panel = self.panel({"off": Reviser,
                            "medium": lambda s, w, st: Blocker(s, w, st, edit=False)})
        paths, records = self.call(panel)
        self.assertEqual((self.lane / TARGET).read_text(),
                         "int f(void) { return 1; } /* round 2 */\n")
        self.assertIsNotNone(records[0]["seed_patch_sha256"])
        self.assert_no_scratch_left()


class RealProcessStop(Fixture):
    """The loser is a REAL actor process tree ended through `_run_stoppable` (C22)."""

    SLEEPER = r"""
import os, subprocess, sys, time
grand = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(sys.argv[1], "w") as fh:
    fh.write(f"{os.getpid()} {grand.pid}")
time.sleep(120)
"""
    EDITOR = r"""
import json, os, sys, time
deadline = time.time() + 20
while not (os.path.exists(sys.argv[1]) and open(sys.argv[1]).read().strip()):
    if time.time() > deadline: sys.exit(3)
    time.sleep(0.02)
with open("ggml/src/kernel.c", "w") as fh:
    fh.write("int f(void) { return 42; }\n")
print(json.dumps({"paths": ["ggml/src/kernel.c"]}))
"""

    def setUp(self):
        super().setUp()
        self.marker = self.root / "pids"
        self._patches = [mock.patch.object(actors, "STOP_POLL_S", 0.05),
                         mock.patch.object(actors, "STOP_GRACE_S", 3.0)]
        for patch in self._patches:
            patch.start()

    def tearDown(self):
        for patch in self._patches:
            patch.stop()
        if self.marker.exists():
            for token in self.marker.read_text().split():
                try:
                    os.kill(int(token), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
        super().tearDown()

    def _planner(self, script, ws, stop):
        class Script(actors.Backend):
            def argv(inner, prompt, workspace, *, read_only=False):
                return [sys.executable, "-c", script, str(self.marker)]
        backend = Script(kind="opencode", model="test/script", effort="high",
                         binary=sys.executable)
        return actors.AgentPlanner(workspace=ws, backend=backend, timeout_s=60,
                                   should_stop=stop)

    def test_the_losing_actor_process_group_is_ended(self):
        panel = self.panel({"off": lambda s, w, st: self._planner(self.EDITOR, w, st),
                            "medium": lambda s, w, st: self._planner(self.SLEEPER, w, st)})
        paths, records = self.call(panel)
        self.assertEqual(tuple(paths), (TARGET,))
        self.assertEqual((self.lane / TARGET).read_text(), "int f(void) { return 42; }\n")
        members = {m["label"]: m for m in records[0]["members"]}
        self.assertEqual(members["a1-medium"]["result"], "cancelled")
        pids = [int(t) for t in self.marker.read_text().split()]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
            time.sleep(0.05)
        self.assertFalse(any(_alive(pid) for pid in pids), "the loser's process tree survived")
        self.assert_no_scratch_left()


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state not in ("Z", "X")


# --------------------------------------------------------------------------- lifecycle


class ScratchLifecycle(Fixture):

    def test_worktrees_are_created_at_the_base_and_removed(self):
        seen = []

        class Peek(Editor):
            def author(self, hypothesis, context):
                seen.append((_git(self.ws, "rev-parse", "HEAD"), self.ws))
                return super().author(hypothesis, context)

        with mock.patch.object(self.registry, "ensure_free",
                               wraps=self.registry.ensure_free) as guard:
            _paths, records = self.call(self.panel(
                {"off": Peek, "medium": lambda s, w, st: Peek(s, w, st, delay=0.2)}))
        self.assertEqual({head for head, _ws in seen}, {self.base})
        self.assertEqual(len({ws.parent for _h, ws in seen}), 2, "authors share a parent")
        for _head, ws in seen:
            self.assertFalse(ws.parent.exists(), "a member dir survived the scope")
        guard.assert_called_once_with(2 * bestof.SCRATCH_BYTES_PER_AUTHOR)
        stats = self.registry.stats()
        # Per member: its home dir, its worktree, its ak-check dir.
        self.assertEqual((stats["allocated"], stats["released"]), (6, 6))
        moved = records[0]["scratch"]
        self.assertEqual((moved["delta_allocated"], moved["delta_released"]), (6, 6))
        self.assertGreater(moved["delta_bytes_freed"], 0)
        self.assert_no_scratch_left()

    def test_member_trees_are_marked_while_they_run(self):
        seen = []

        class Peek(Editor):
            def author(self, hypothesis, context):
                seen.append(((self.ws.parent / scratch.MARKER).is_file(),
                             Path(str(self.ws) + scratch.SIDECAR_SUFFIX).is_file()))
                return super().author(hypothesis, context)

        self.call(self.panel({"off": Peek, "medium": Peek}))
        self.assertEqual(seen, [(True, True), (True, True)])
        self.assert_no_scratch_left()

    def test_scope_exit_releases_without_an_early_release_method(self):
        self.registry = FakeRegistry(self.root / "fake-scratch", release_method=False)
        self.call(self.panel({"off": Editor,
                              "medium": lambda s, w, st: Blocker(s, w, st)}))
        kinds = [kind for kind, _p in self.registry.events]
        self.assertEqual(kinds.count("scope_exit_release"), 2)
        self.assert_no_scratch_left()

    def test_removed_on_an_exception_after_selection(self):
        class LaneMover(Editor):
            def author(inner, hypothesis, context):
                (self.lane / OTHER).write_text("changed under the panel\n")
                return super(LaneMover, inner).author(hypothesis, context)

        with self.assertRaisesRegex(bestof.PanelSetupRefused, "lane changed"):
            self.call(self.panel({"off": LaneMover,
                                  "medium": lambda s, w, st: Blocker(s, w, st)}))
        self.assert_no_scratch_left()

    def test_removed_when_a_member_factory_or_validator_raises(self):
        def broken(s, w, st):
            raise RuntimeError("factory fault")

        def validator(*args, **kwargs):
            raise RuntimeError("validator fault")

        with self.assertRaisesRegex(RuntimeError, "factory fault"):
            self.call(self.panel({"off": broken, "medium": broken}))
        self.assert_no_scratch_left()
        paths, records = self.call(self.panel({"off": Editor, "medium": Editor},
                                              validator=validator))
        self.assertEqual(records[0]["selection"], "best_failing")
        self.assert_no_scratch_left()

    def test_removed_on_budget_exhaustion(self):
        class Spent(Editor):
            def author(self, hypothesis, context):
                (self.ws / TARGET).write_text("half\n")
                raise actors.ActorBudgetExhausted("budget_exhausted: author call ended")

        with self.assertRaises(ActorTransient):
            self.call(self.panel({"off": Spent, "medium": Spent}))
        self.assert_no_scratch_left()

    def test_free_space_guard_falls_back_to_the_single_author(self):
        self.registry = self.real_registry(min_free_bytes=10 ** 18)
        solo_calls = []

        def solo(h, ctx):
            solo_calls.append(h)
            (self.lane / TARGET).write_text("solo\n")
            return (TARGET,)

        paths, records = self.call(self.panel({"off": Editor, "medium": Editor}), solo=solo)
        self.assertEqual(solo_calls, [HYP])
        self.assertEqual(tuple(paths), (TARGET,))
        self.assertEqual(records[0]["selection"], "fallback_single")
        self.assertIn("space", records[0]["fallback_reason"])
        self.assertEqual(self.registry.stats()["allocated"], 0)
        self.assertEqual(self.registry.stats()["guard_refusals"], 1)

    def test_scratch_trees_sharing_a_parent_fall_back(self):
        # The actor writes its per-call opencode config into workspace.parent: two
        # members there would run each other's thinking mode.
        self.registry = FakeRegistry(self.root / "fake-scratch", shared_parent=True)
        paths, records = self.call(self.panel({"off": Editor, "medium": Editor}),
                                   solo=lambda h, c: (TARGET,))
        self.assertEqual(records[0]["selection"], "fallback_single")
        self.assertIn("share the parent", records[0]["fallback_reason"])
        self.assert_no_scratch_left()

    def test_the_panel_never_prunes_gcs_or_manages_worktrees_itself(self):
        source = Path(bestof.__file__).read_text(encoding="utf-8")
        strings = [tok.string for tok in tokenize.generate_tokens(io.StringIO(source).readline)
                   if tok.type == tokenize.STRING]
        joined = "\n".join(strings)
        for forbidden in (r"""^[rbuf]*['"]prune['"]$""", r"""^[rbuf]*['"]gc['"]$"""):
            self.assertFalse(any(re.match(forbidden, s) for s in strings), forbidden)
        # Spelled in pieces, like test_scratch's scanner, so this file is not itself
        # an offender under that package-wide ban.
        self.assertNotIn("worktree " + "pr" + "une", joined)
        self.assertNotRegex(source, r"""["']worktree["']\s*,\s*["'](add|remove|prune)["']""")
        self.assertNotIn("rmtree", source)


# --------------------------------------------------------------------------- ak-check

#: A stand-in for `ak_check.py` with its CLI and exit contract (0 pass, 1 fail, 2
#: refused): decides by what the member wrote into the target.
FAKE_AK_CHECK = r"""
import argparse, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--op-test", action="store_true")
ap.add_argument("--lane"); ap.add_argument("--build-dir"); ap.add_argument("--scratch")
ap.add_argument("--base")
a = ap.parse_args()
assert a.op_test and a.build_dir == "/anchor/build", a
assert os.path.isfile(os.path.join(a.scratch, ".ak-scratch-owner")), "scratch not marked"
assert os.path.dirname(a.scratch) == os.path.dirname(a.lane), (a.scratch, a.lane)
open(os.path.join(a.scratch, "obj.o"), "w").write("x" * 4096)
text = open(os.path.join(a.lane, "ggml/src/kernel.c")).read()
if "return 7" in text:
    print("ak-check op-test: PASS"); sys.exit(0)
if "return 8" in text:
    print("ak-check op-test: FAIL\n--- ggml/src/kernel.c: ok\n--- test-backend-ops -o MUL_MAT: 3/4 passed")
    sys.exit(1)
if "REFUSE" in text:
    print("ak-check op-test: REFUSED\nREFUSED: the loop is measuring"); sys.exit(2)
print("ak-check op-test: FAIL\n--- ggml/src/kernel.c: FAIL\nerror: bad intrinsic"); sys.exit(1)
"""
CONTEXT = {"target": {"recipe": {"build_dir": "/anchor/build"}}}


class AkCheckWinner(Fixture):

    def setUp(self):
        super().setUp()
        self.script = self.root / "fake_ak_check.py"
        self.script.write_text(FAKE_AK_CHECK)
        self.validator = bestof.chain_validators(
            bestof.integrity_validator, bestof.ak_check_validator(self.script, timeout_s=60))

    def text(self, value):
        return f"int f(void) {{ {value} }}\n"

    def test_the_first_author_passing_ak_check_wins(self):
        panel = self.panel({
            "off": lambda s, w, st: Editor(s, w, st, text=self.text("return 9;")),
            "medium": lambda s, w, st: Editor(s, w, st, delay=0.3, text=self.text("return 7;"))},
            validator=self.validator)
        _paths, records = self.call(panel, context=CONTEXT)
        row = records[0]
        self.assertEqual((row["selection"], row["winner"]), ("first_passing", "a1-medium"))
        members = {m["label"]: m for m in row["members"]}
        self.assertEqual(members["a0-off"]["validation"]["checks"]["ak-check"]["returncode"], 1)
        self.assertEqual(members["a1-medium"]["validation"]["checks"]["ak-check"]["returncode"], 0)
        self.assertEqual((self.lane / TARGET).read_text(), self.text("return 7;"))
        self.assert_no_scratch_left()

    def test_no_pass_prefers_the_failure_that_reached_the_op_test(self):
        panel = self.panel({
            "off": lambda s, w, st: Editor(s, w, st, text=self.text("return 9;")),
            "medium": lambda s, w, st: Editor(s, w, st, delay=0.3, text=self.text("return 8;"))},
            validator=self.validator)
        _paths, records = self.call(panel, context=CONTEXT)
        row = records[0]
        self.assertEqual((row["selection"], row["winner"]), ("best_failing", "a1-medium"))
        members = {m["label"]: m for m in row["members"]}
        self.assertGreater(members["a1-medium"]["validation"]["score"],
                           members["a0-off"]["validation"]["score"])
        self.assert_no_scratch_left()

    def test_a_refused_check_falls_back_to_the_integrity_screen(self):
        panel = self.panel({
            "off": lambda s, w, st: Editor(s, w, st, text=self.text("return 1; /* REFUSE */")),
            "medium": lambda s, w, st: Blocker(s, w, st)}, validator=self.validator)
        _paths, records = self.call(panel, context=CONTEXT)
        row = records[0]
        self.assertEqual((row["selection"], row["winner"]), ("first_passing", "a0-off"))
        check = {m["label"]: m for m in row["members"]}["a0-off"]["validation"]["checks"]
        self.assertTrue(check["ak-check"]["inconclusive"])
        self.assert_no_scratch_left()

    def test_no_anchor_build_dir_is_inconclusive(self):
        result = bestof.ak_check_validator(self.script)(HYP, self.lane, self.base, [TARGET],
                                                         lambda: False, context={})
        self.assertTrue(result.passed)
        self.assertTrue(result.checks["ak-check"]["inconclusive"])


# --------------------------------------------------------------------------- metrics


class PanelRecord(Fixture):

    def test_per_author_metrics_patches_and_rows(self):
        panel = self.panel({
            "off": lambda s, w, st: Editor(s, w, st, rows={"steps": 5, "decoded": 1200}),
            "medium": lambda s, w, st: Editor(s, w, st, delay=0.3,
                                              text="int f(void) { return 9; }\n",
                                              rows={"steps": 11, "decoded": 5400})})
        paths, records = self.call(panel)
        row = records[0]
        self.assertEqual(row["schema"], bestof.PANEL_SCHEMA)
        self.assertEqual(row["pool"]["context_limit"], 90_112)
        members = {m["label"]: m for m in row["members"]}
        self.assertEqual({m["thinking"] for m in members.values()}, {"off", "medium"})
        self.assertEqual((members["a0-off"]["steps"], members["a0-off"]["decoded_tokens"]),
                         (5, 1200))
        self.assertEqual((members["a1-medium"]["steps"], members["a1-medium"]["decoded_tokens"]),
                         (11, 5400))
        self.assertEqual({m["result"] for m in members.values()}, {"won", "lost"})
        for member in members.values():
            self.assertIsNotNone(member["wall_s"])
            self.assertIn("passed", member["validation"])
            # Both authors' patches are in the patch store, for the record.
            patch = Path(member["patch"]["patch_file"])
            self.assertTrue(patch.is_file())
            self.assertIn(f"lane0.{member['label']}", patch.name)
        self.assertEqual(len(list((self.store / "patches").glob("*.patch"))), 2)
        # The panel row and the moved, annotated actor-call rows are in the LANE's
        # reply dir; the scratch copies are gone (no double count).
        replies = self.workers / actor_metrics.REPLY_DIR_NAME
        panel_rows = [json.loads(line) for line in
                      (replies / bestof.PANEL_LOG).read_text().splitlines()]
        self.assertEqual(panel_rows[0]["panel_id"], row["panel_id"])
        calls = [json.loads(line) for line in
                 (replies / actor_metrics.CALL_LOG_NAME).read_text().splitlines()]
        metric = [c for c in calls if c.get("schema") == actor_metrics.METRICS_SCHEMA]
        self.assertEqual({c["author_panel"]["label"] for c in metric}, {"a0-off", "a1-medium"})
        self.assertEqual({c["author_panel"]["panel_id"] for c in metric}, {row["panel_id"]})
        sealed = [c for c in calls if c.get("schema") == "vb-ak-seat.call.v1"]
        self.assertEqual(sealed, [{"schema": "vb-ak-seat.call.v1", "sealed": True}] * 2)
        # Summarized over EVERYTHING (lane reply dir and the scratch parents, which the
        # double keeps): each author call is counted exactly once.
        summary = actor_metrics.summarize(self.root)
        self.assertEqual(summary["roles"]["author"]["calls"], 2)
        self.assertEqual(summary["roles"]["author"]["decoded_tokens_total"], 6600)


# --------------------------------------------------------------------------- the loop


class _Planner:
    def __init__(self):
        self.author_calls = 0

    def propose(self, context):
        return HYP

    def author(self, hypothesis, context):
        self.author_calls += 1
        raise AssertionError("the single author must not run under a panel")


class _Critic:
    def __init__(self, patch_error=None):
        self.patch_error = patch_error

    def review_hypothesis(self, hypothesis, context):
        return Review(True)

    def review_patch(self, hypothesis, paths, context):
        if self.patch_error is not None:
            raise self.patch_error
        return Review(False, "patch rejected by the critic")


class LoopIntegration(Fixture):

    def iterate(self, panel, critic):
        return loop.iterate(
            planner=_Planner(), critic=critic, context={}, patch_rounds=1,
            hypothesis_rounds=1,
            measure=lambda h, p: (_ for _ in ()).throw(AssertionError("no measure")),
            gate=lambda h, p: (True, []), commit=lambda *a: "head",
            author_lane=(self.lane, self.base), author_panel=panel)

    def test_critic2_checkpoint_retains_the_winning_diff(self):
        panel = self.panel({"off": Editor, "medium": lambda s, w, st: Blocker(s, w, st)})
        outcome = self.iterate(panel, _Critic(patch_error=ActorTransient("critic 401")))
        self.assertEqual(outcome.status, "planner_transient")
        stages = [ck["stage"] for ck in outcome.resume_checkpoints]
        self.assertIn("critic2", stages)
        # The owner retains the LANE's diff for the critic2 checkpoint: the winner's.
        self.assertEqual((self.lane / TARGET).read_text(), "int f(void) { return 7; }\n")
        self.assertEqual(len(outcome.author_panels), 1)
        self.assertEqual(outcome.to_attempt()["author_panels"][0]["winner"], "a0-off")
        self.assert_no_scratch_left()

    def test_a_rejected_patch_row_carries_its_panel(self):
        abandoned = []
        panel = self.panel({"off": Editor, "medium": lambda s, w, st: Blocker(s, w, st)})
        outcome = loop.iterate(
            planner=_Planner(), critic=_Critic(), context={}, patch_rounds=1,
            hypothesis_rounds=1, measure=lambda h, p: None, gate=lambda h, p: (True, []),
            commit=lambda *a: "head", author_lane=(self.lane, self.base), author_panel=panel,
            record_abandoned=abandoned.append)
        # lane/ak-keephyp-20260926: an ACCEPTED hypothesis whose patch rounds all end
        # rejected stays pending, so the iteration ends `patch_rounds_exhausted`.
        self.assertEqual(outcome.status, loop.PATCH_ROUNDS_EXHAUSTED)
        self.assertEqual(abandoned[0].status, "patch_rejected")
        self.assertEqual(abandoned[0].author_panels[0]["winner"], "a0-off")
        self.assertEqual(len(outcome.author_panels), 1)


class SingleAuthorPathUnchanged(unittest.TestCase):
    """N=1: no panel is built and nothing about the single path changes."""

    def _run(self, **extra):
        seen = []
        real = loop.iterate

        def spy(**kwargs):
            seen.append(sorted(kwargs))
            return loop.Outcome("abstained", None, ["x"])

        with mock.patch.object(loop, "iterate", spy):
            pipeline.run_pool(
                workers=[pipeline.Worker("lane0", Path("/nonexistent/lane0"),
                                         Path("/nonexistent/b0"))],
                make_planner=lambda w: object(), make_critic=lambda w: object(),
                build_context=dict, make_gate=lambda w: None, make_measure=lambda w: None,
                commit=lambda *a: "h", champion_head=lambda: "c" * 40,
                reset_to_champion=lambda w: "c" * 40, record=lambda o: None, iterations=1,
                **extra)
        del real
        return seen

    def test_run_pool_passes_no_panel_for_n1(self):
        baseline = self._run()
        self.assertNotIn("author_panel", baseline[0])
        self.assertEqual(self._run(make_author_panel=lambda w: None), baseline)
        self.assertIn("author_panel", self._run(make_author_panel=lambda w: "panel")[0])

    def test_iterate_without_a_panel_calls_the_planner_author_directly(self):
        calls = []

        class P:
            def propose(self, context):
                return HYP

            def author(self, hypothesis, context):
                calls.append((hypothesis, dict(context)))
                return Abstain("n1")

        outcome = loop.iterate(planner=P(), critic=_Critic(), context={"a": 1},
                               measure=lambda h, p: None, gate=lambda h, p: (True, []),
                               commit=lambda *a: "h")
        self.assertEqual(outcome.status, loop.AUTHORING_FAILED)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("author_panels", outcome.to_attempt())
        self.assertEqual(outcome.author_panels, [])


if __name__ == "__main__":
    unittest.main()
