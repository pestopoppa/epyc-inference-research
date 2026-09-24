"""DS41-C22: a stop must reach an in-flight actor call, and a stopped actor is never retried.

DS41 run 7 (2026-09-24 10:17): SIGTERM to `run.py` set the drain flag, but the planner
call IS the forming stage, so nothing polled the flag until the call returned. When the
actor was TERM'd directly, `_with_backoff` read its rc -15 as a provider transient and
launched a new actor. These tests run REAL child processes (a Python sleeper that also
spawns a grandchild), because the property under test is that the process TREE dies.
"""
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

from autokernel.loop import actors, loop as loop_mod
from autokernel.loop.loop import ActorStopped, Hypothesis

# The child writes its pid and its grandchild's pid, then sleeps. A grandchild shows
# that the whole process group is ended, not just the direct child.
SLEEPER = r"""
import os, subprocess, sys, time
grand = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(sys.argv[1], "w") as fh:
    fh.write(f"{os.getpid()} {grand.pid}")
time.sleep(120)
"""

# Writes its pid and a marker, then TERMs ITSELF: an actor that died of a signal.
SELF_TERM = r"""
import os, signal, sys
with open(sys.argv[1], "w") as fh:
    fh.write(str(os.getpid()))
signal.signal(signal.SIGTERM, signal.SIG_DFL)
os.kill(os.getpid(), signal.SIGTERM)
"""


def _alive(pid: int) -> bool:
    """True if `pid` exists and is not a zombie. Container init may not reap, so a
    killed grandchild can linger as a zombie; that is dead for this purpose."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state not in ("Z", "X")


def _wait_for(path: Path, timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.02)
    raise AssertionError(f"child never wrote {path}")


class _ScriptBackend(actors.Backend):
    """An opencode-shaped backend whose argv runs a local Python script."""

    script: str = ""
    marker: str = ""

    def argv(self, prompt, workspace, *, read_only=False):
        return [sys.executable, "-c", self.script, self.marker]


def _backend(script: str, marker: Path) -> _ScriptBackend:
    backend = _ScriptBackend(kind="opencode", model="test/sleeper", effort="high",
                             binary=sys.executable)
    object.__setattr__(backend, "script", script)
    object.__setattr__(backend, "marker", str(marker))
    return backend


class _Fast:
    """Tight poll and grace so each test takes about a second, not STOP_GRACE_S."""

    def setUp(self):
        self._patches = [mock.patch.object(actors, "STOP_POLL_S", 0.05),
                         mock.patch.object(actors, "STOP_GRACE_S", 3.0)]
        for patch in self._patches:
            patch.start()
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.ws = root / "lane"
        self.ws.mkdir()
        self.marker = root / "pids"

    def tearDown(self):
        for patch in self._patches:
            patch.stop()
        # Never leave a sleeper behind if an assertion failed mid-test.
        if self.marker.exists():
            for token in self.marker.read_text().split():
                try:
                    os.kill(int(token), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
        self._tmp.cleanup()


class StopDuringActorCall(_Fast, unittest.TestCase):

    def test_a_stop_mid_call_ends_the_whole_tree_and_raises_actor_stopped(self):
        backend = _backend(SLEEPER, self.marker)
        stop = {"asked": False}

        def should_stop():
            if not stop["asked"] and self.marker.exists() and self.marker.read_text().strip():
                stop["asked"] = True       # the loop was told to stop mid-call
            return stop["asked"]

        started = time.monotonic()
        with self.assertRaises(ActorStopped) as caught:
            actors._run_agent("p", workspace=self.ws, timeout_s=60, backend=backend,
                              should_stop=should_stop)
        self.assertLess(time.monotonic() - started, 30)
        self.assertIn("not retried", str(caught.exception))
        child, grand = (int(t) for t in _wait_for(self.marker).split())
        deadline = time.monotonic() + 5
        while (_alive(child) or _alive(grand)) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(_alive(child), "the actor survived the stop")
        self.assertFalse(_alive(grand), "the actor's child survived: the group was not ended")
        # The call is still accounted for: a stopped call is evidence too.
        log = (self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG).read_text()
        self.assertIn('"returncode": -15', log)

    def test_the_planner_launches_exactly_once_when_stopped(self):
        """The run-7 shape end to end: planner -> backoff -> actor; stop mid-call."""
        backend = _backend(SLEEPER, self.marker)
        launches = []
        real_popen = actors.subprocess.Popen

        def counting_popen(*args, **kwargs):
            launches.append(args[0])
            return real_popen(*args, **kwargs)

        def should_stop():
            return self.marker.exists() and bool(self.marker.read_text().strip())

        planner = actors.AgentPlanner(workspace=self.ws, backend=backend, timeout_s=60,
                                      should_stop=should_stop)
        started = time.monotonic()
        with mock.patch.object(actors.subprocess, "Popen", side_effect=counting_popen):
            with self.assertRaises(ActorStopped):
                planner.propose({})
        self.assertEqual(len(launches), 1, "a stopped actor was relaunched")
        # The first backoff is BACKOFF_S[0] = 30 s: finishing well inside it proves the
        # stopped call slept no backoff before giving up.
        self.assertLess(time.monotonic() - started, actors.BACKOFF_S[0] / 2)

    def test_a_signal_death_while_a_stop_is_asked_is_a_stop_not_a_transient(self):
        """Someone TERMs the actor directly as part of stopping the run (10:17:49)."""
        backend = _backend(SELF_TERM, self.marker)

        def should_stop():
            return self.marker.exists() and bool(self.marker.read_text().strip())

        with self.assertRaises(ActorStopped):
            actors._run_agent("p", workspace=self.ws, timeout_s=60, backend=backend,
                              should_stop=should_stop)


class SignalDeathWithoutStop(_Fast, unittest.TestCase):
    """rc < 0 with NO stop asked keeps today's behaviour: a transient, retried.

    Deliberate: an operator killing ONE hung actor (or earlyoom taking it) wants the
    call retried; only a stop of the run means "do not relaunch"."""

    def test_it_is_still_a_provider_transient(self):
        backend = _backend(SELF_TERM, self.marker)
        with self.assertRaises(actors.ProviderTransient) as caught:
            actors._run_agent("p", workspace=self.ws, timeout_s=60, backend=backend,
                              should_stop=lambda: False)
        self.assertNotIsInstance(caught.exception, ActorStopped)
        self.assertIn("actor exited -15", str(caught.exception))

    def test_without_a_stop_predicate_the_old_path_is_unchanged(self):
        backend = _backend(SELF_TERM, self.marker)
        with mock.patch.object(actors, "_run_stoppable") as stoppable:
            with self.assertRaises(actors.ProviderTransient):
                actors._run_agent("p", workspace=self.ws, timeout_s=60, backend=backend)
        stoppable.assert_not_called()


class BackoffHonoursStop(unittest.TestCase):

    def test_no_attempt_is_drawn_once_a_stop_is_asked(self):
        call = mock.Mock(return_value="ok")
        with self.assertRaises(ActorStopped):
            actors._with_backoff(call, sleep=lambda _s: None, should_stop=lambda: True)
        call.assert_not_called()

    def test_an_actor_stopped_from_the_call_is_never_retried(self):
        call = mock.Mock(side_effect=ActorStopped("stopped"))
        slept = []
        with self.assertRaises(ActorStopped):
            actors._with_backoff(call, sleep=slept.append, should_stop=lambda: False)
        self.assertEqual((call.call_count, slept), (1, []))

    def test_a_stop_during_the_backoff_ends_it_without_another_attempt(self):
        state = {"slept": 0.0}
        call = mock.Mock(side_effect=actors.ProviderTransient("exited 1"))

        def sleep(seconds):
            state["slept"] += seconds

        def should_stop():
            return state["slept"] >= 2.0      # stop arrives two seconds into a 30 s backoff

        with mock.patch.object(actors, "STOP_POLL_S", 1.0):
            with self.assertRaises(ActorStopped):
                actors._with_backoff(call, sleep=sleep, should_stop=should_stop)
        self.assertEqual(call.call_count, 1)
        self.assertLess(state["slept"], actors.BACKOFF_S[0])

    def test_without_a_stop_predicate_the_backoff_is_unchanged(self):
        slept = []
        call = mock.Mock(side_effect=[actors.ProviderTransient("x"), "ok"])
        self.assertEqual(actors._with_backoff(call, sleep=slept.append), ("ok", 1))
        self.assertEqual(slept, [actors.BACKOFF_S[0]])


class IterateRecordsTheStop(unittest.TestCase):

    HYP = Hypothesis(mechanism_id="akm-x", statement="s", falsifier="f",
                     target_surface="a.cpp", target_symbol="f")

    def _iterate(self, planner, critic):
        return loop_mod.iterate(
            planner=planner, critic=critic, context={},
            measure=mock.Mock(side_effect=AssertionError("no measurement after a stop")),
            gate=mock.Mock(side_effect=AssertionError("no gate after a stop")),
            commit=mock.Mock(side_effect=AssertionError("no commit after a stop")))

    def test_a_stop_during_proposing_is_stopped_mid_formation(self):
        planner = mock.Mock()
        planner.propose.side_effect = ActorStopped("stop asked")
        outcome = self._iterate(planner, mock.Mock())
        self.assertEqual(outcome.status, "stopped_mid_formation")
        self.assertEqual(planner.propose.call_count, 1)

    def test_a_stop_during_the_critic_still_names_the_hypothesis(self):
        planner = mock.Mock()
        planner.propose.return_value = self.HYP
        critic = mock.Mock()
        critic.review_hypothesis.side_effect = ActorStopped("stop asked")
        outcome = self._iterate(planner, critic)
        self.assertEqual(outcome.status, "stopped_mid_formation")
        self.assertEqual(outcome.hypothesis, self.HYP)
        planner.author.assert_not_called()


if __name__ == "__main__":
    unittest.main()
