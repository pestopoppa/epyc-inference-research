"""Stubbed CPU campaign continuation through the installed iterator, without inference."""
import json
import os
from pathlib import Path
import signal
import time
import traceback
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from . import cpu_profile, pool, run, serial_run as sr, serving
from . import test_existing_cpu_run as cpu_fixture
from . import test_promotion_targets as promotion_fixture
from . import test_resume as resume_fixture
from . import pipeline, resume
from .test_cpu_screen import _inputs as cpu_inputs


def test_experimental_keep_continues_into_a_completed_next_batch():
    """A kept tip is not the champion of record (DS41-C45)."""
    original_main = run.main
    completed = []

    def first_then_next(argv):
        assert original_main(argv) == 0
        first = Path(sr.option(argv, "--out"))
        continuation = first / "loop-continuation.json"
        row, _ = sr.load_completed(continuation)
        startup = Path(sr.option(argv, "--anchor-build")).resolve()
        assert row["current_anchor"]["path"] != str(startup)
        resumed = first.parent / "resumed"
        # All actor, build and serving doubles remain installed by the CPU fixture.
        # The actual claim context, pool, checkpoint/keep and result paths run.
        with mock.patch.object(run.serving, "calibrate_floor",
                               side_effect=AssertionError("unexpected recalibration")):
            assert original_main([*argv, "--resume-run", str(continuation),
                                  "--out", str(resumed), "--iterations", "1"]) == 0
        assert row["cor_anchor"] is not None, "experimental continuation lost its COR"
        assert row["cor_anchor"]["path"] == str(startup)
        result = json.loads((resumed / "loop-run.json").read_text())
        assert len(result["iterations"]) == 1
        assert result["iterations"][0]["status"] == "measured_null"
        next_row, _ = sr.load_completed(resumed / "loop-continuation.json")
        assert next_row["cor_anchor"] == row["cor_anchor"]
        assert Path(next_row["cor_anchor"]["path"]).exists()
        completed.append(next_row)
        return 0

    with mock.patch.object(run, "main", first_then_next):
        cpu_fixture.test_existing_main_cpu_five_iterations_preserves_canonical_champion(
            False, expected_actor_calls=6, expected_claim_cycles=2,
            result_expectation=lambda *_args: None)
    assert len(completed) == 1


def test_signal_during_stub_calibration_publishes_stopped_and_releases_claim():
    fixture = promotion_fixture.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    old_term, old_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    held, observed = [], []
    original_main = run.main
    try:
        launch, prompts, options, _ = cpu_inputs(fixture)
        requests = prompts.requests(("glm-fixed2029",), launch.template)

        @contextmanager
        def hold(*_args, **_kwargs):
            held.append("open")
            try:
                yield {"device_id": "synthetic-cpu"}
            finally:
                held.append("close")

        def observe(recipe, build, port, *, resolved_recipe, frozen_requests, **_kwargs):
            assert held == ["open"] and frozen_requests == requests
            resolved_recipe.validate_launch(recipe, build, port)
            observed.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
            return 10.0

        def invoke(argv):
            argv += options + ["--serving-pairs", "3", "--serving-instrument",
                serving.MATCHED_INSTRUMENT, "--out", str(fixture.root / "result"),
                "--allow-untested-actor-seats", "synthetic fixture actors"]
            with mock.patch.object(run.claim, "hold_cpu", hold), \
                    mock.patch.object(run.claim, "hold", side_effect=AssertionError("GPU claim")), \
                    mock.patch.object(run.os, "sched_getaffinity", return_value={0, 1}), \
                    mock.patch.object(run.os, "sched_setaffinity"), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q5_0")), \
                    mock.patch.object(cpu_profile, "profile_loop", side_effect=cpu_profile.CpuProfileRefused("stub")), \
                    mock.patch.object(serving, "_measure_once", observe), \
                    mock.patch.object(serving, "_matched_frame", return_value={}), \
                    mock.patch.object(serving, "_matched_plan", return_value={"orders": [["anchor", "candidate"]]}), \
                    mock.patch.object(pool, "provision", side_effect=AssertionError("stopped before pool")):
                return original_main(argv)

        with mock.patch.object(run, "main", invoke):
            rc, builds, planners, _, log = fixture._run_one_keep()
        assert rc == 0, log
        assert observed == [True] and held == ["open", "close"]
        assert not builds and not planners
        result = json.loads((fixture.root / "result/loop-run.json").read_text())
        assert result["iterations"] == []
        assert result["continuation"]["terminal"] == "stopped"
        sr.load_completed(fixture.root / "result/loop-continuation.json")
        assert not serving.floor_path(fixture.store, launch.template, frozen_requests=requests).exists()
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        fixture.doCleanups()


@pytest.mark.parametrize("stop_kind", ["signal", "stop_file"])
def test_signal_during_formation_retains_patch_and_releases_claim(tmp_path, stop_kind):
    """Signal the owned campaign child while its stub author is in formation."""
    marker, error = tmp_path / "ready", tmp_path / "error"
    child = os.fork()
    if child == 0:
        try:
            original_main = run.main

            def interrupted(argv):
                original_factory = run.actors.AgentPlanner

                def planner(*args, **kwargs):
                    actor = original_factory(*args, **kwargs)
                    author = actor.author
                    should_stop = kwargs["should_stop"]

                    def wait_after_edit(*author_args):
                        paths = author(*author_args)
                        marker.write_text(sr.option(argv, "--store"))
                        deadline = time.monotonic() + 20
                        while not should_stop() and time.monotonic() < deadline:
                            time.sleep(.01)
                        assert should_stop(), "signal did not reach campaign"
                        return paths
                    actor.author = wait_after_edit
                    return actor

                with mock.patch.object(run.actors, "AgentPlanner", planner):
                    return original_main(argv)

            def check(result, _measured, builds, fixture, _selected):
                assert not builds
                assert [row["status"] for row in result["iterations"]] == ["stopped_mid_formation"]
                patches = list((fixture.store / "patches").glob("cpu-1.lane0.*.patch"))
                assert patches and b"cpu-1" in patches[0].read_bytes()

            with mock.patch.object(run, "main", interrupted):
                cpu_fixture.test_existing_main_cpu_five_iterations_preserves_canonical_champion(
                    False, expected_actor_calls=1, expected_oracle_calls=0,
                    result_expectation=check)
        except BaseException:
            error.write_text(traceback.format_exc())
            os._exit(1)
        os._exit(0)

    identity = (os.stat(f"/proc/{child}").st_uid, os.getsid(child), os.getpgid(child),
                Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[19])
    reaped = False
    try:
        deadline = time.monotonic() + 25
        while not marker.exists() and time.monotonic() < deadline:
            pid, status = os.waitpid(child, os.WNOHANG)
            if pid:
                reaped = True
                raise AssertionError(f"campaign exited before marker: {status}: "
                                     f"{error.read_text() if error.exists() else ''}")
            time.sleep(.01)
        assert marker.exists(), "formation marker timed out"
        if stop_kind == "signal":
            assert identity == (os.stat(f"/proc/{child}").st_uid, os.getsid(child),
                                os.getpgid(child), Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[19])
            os.kill(child, signal.SIGTERM)
        else:
            (Path(marker.read_text()) / "STOP").touch()
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            pid, status = os.waitpid(child, os.WNOHANG)
            if pid:
                reaped = True
                assert os.waitstatus_to_exitcode(status) == 0, error.read_text() if error.exists() else status
                break
            time.sleep(.01)
        assert reaped, "campaign did not terminate after SIGTERM"
        assert not Path(f"/proc/{child}").exists()
    finally:
        if not reaped:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)


def test_killed_resumed_build_can_be_audited_and_reopened(tmp_path):
    """A killed claimed build keeps its patch and claim until audited reopen."""
    fixture = resume_fixture.Fixture()
    fixture.setUp()
    child, reaped = None, False
    marker = tmp_path / "build-ready"
    try:
        fixture.refuse_once()
        (refused,) = resume_fixture.rows_with(fixture.store, "gate_refused")
        retained = Path(refused["retained_patch"]["patch_file"])
        patch_bytes = retained.read_bytes()
        queue, report = fixture.prepare()
        (checkpoint,) = report["queued"]
        checkpoint_id = checkpoint["checkpoint_id"]
        child = os.fork()
        if child == 0:
            try:
                def stalled_gate(_hypothesis, _paths):
                    marker.touch()
                    while True:
                        time.sleep(.05)

                pipeline.run_pool(
                    workers=[pipeline.Worker("lane0", fixture.repo, fixture.tmp / "build")],
                    make_planner=lambda _w: resume_fixture.NoActors(fixture),
                    make_critic=lambda _w: resume_fixture.NoActors(fixture),
                    build_context=dict, make_gate=lambda _w: stalled_gate,
                    make_measure=lambda _w: lambda _h, _p: resume_fixture.comparison(),
                    commit=lambda *_args: fixture.fail("unexpected keep"),
                    champion_head=lambda: fixture.anchor,
                    reset_to_champion=lambda _w: fixture.anchor,
                    record=fixture.owner.record, iterations=1, next_resume=queue.take)
            finally:
                os._exit(1)
        identity = (os.stat(f"/proc/{child}").st_uid, os.getsid(child), os.getpgid(child),
                    Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[19])
        deadline = time.monotonic() + 25
        while not marker.exists() and time.monotonic() < deadline:
            pid, status = os.waitpid(child, os.WNOHANG)
            if pid:
                reaped = True
                raise AssertionError(f"campaign exited before build: {status}")
            time.sleep(.01)
        assert marker.exists(), "resumed build never started"
        assert fixture.claims()[checkpoint_id]["result_status"] is None
        assert retained.read_bytes() == patch_bytes
        assert identity == (os.stat(f"/proc/{child}").st_uid, os.getsid(child), os.getpgid(child),
                            Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[19])
        os.kill(child, signal.SIGKILL)
        _pid, status = os.waitpid(child, 0)
        reaped = True
        assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
        assert not Path(f"/proc/{child}").exists()
        # The owner contract deliberately requires an audited operator reopen;
        # a dead worker is not silently granted a duplicate claim.
        assert fixture.claims()[checkpoint_id]["state"] == "resumed"
        assert retained.read_bytes() == patch_bytes
        reason = "fixture: captured build process killed and reaped"
        plan = resume.reopen_plan(fixture.store, checkpoint_id=checkpoint_id,
                                  reason=reason, anchor_commit=fixture.anchor,
                                  rules_fingerprint=resume_fixture.CHANGED_RULES)
        assert plan["refused"] is None
        resume.reopen_apply(fixture.store, plan, reason=reason)
        resume_fixture.reset(fixture.repo)
        next_queue, _ = fixture.prepare()
        actors = resume_fixture.NoActors(fixture)
        (outcome,) = pipeline.run_pool(
            workers=[pipeline.Worker("lane0", fixture.repo, fixture.tmp / "build")],
            make_planner=lambda _w: actors, make_critic=lambda _w: actors,
            build_context=dict, make_gate=lambda _w: resume_fixture.passing_gate,
            make_measure=lambda _w: lambda _h, _p: resume_fixture.comparison(),
            commit=lambda *_args: fixture.fail("unexpected keep"),
            champion_head=lambda: fixture.anchor,
            reset_to_champion=lambda _w: fixture.anchor,
            record=fixture.owner.record, iterations=1, next_resume=next_queue.take)
        assert outcome.status == "measured_null" and outcome.resumed_from == checkpoint_id
        assert fixture.claims()[checkpoint_id]["result_status"] == "measured_null"
        assert retained.read_bytes() == patch_bytes
    finally:
        if child and not reaped:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
        fixture.doCleanups()


def test_refused_checkpoint_yields_and_preserves_retained_patch():
    """Reuse the actual pool queue regression and check its durable artifacts."""
    fixture = resume_fixture.ARefusedResumeYieldsToTheNextQueuedCheckpoint(
        "test_the_next_queued_checkpoint_resumes_in_the_same_iteration")
    fixture.setUp()
    try:
        fixture.test_the_next_queued_checkpoint_resumes_in_the_same_iteration()
        rejected, = resume_fixture.rows_with(fixture.store, "resume_rejected")
        claims = fixture.claims()
        assert claims[rejected["resumed_from"]]["result_status"] == "resume_rejected"
        original, = [row for row in resume_fixture.rows_with(fixture.store, "gate_refused")
                     if row["mechanism_id"] == "akm-build"]
        retained = Path(original["retained_patch"]["patch_file"])
        assert retained.is_file() and b"EDITED" in retained.read_bytes()
        assert retained.with_suffix(".json").is_file()
    finally:
        fixture.doCleanups()
