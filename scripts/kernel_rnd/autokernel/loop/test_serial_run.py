"""Owned tiny children and original loop fixtures; no hardware or providers."""
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from . import run, serial_run as sr
from . import test_promotion_targets as promotion_fixture
from . import test_existing_cpu_run as cpu_fixture
from .test_legacy_targets import _argv, _resolved


CHILD = r'''
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from scripts.kernel_rnd.autokernel.loop import serial_run as sr, campaign_cli, legacy_targets
argv = sys.argv[1:]
out = Path(sr.option(argv, "--out"))
target_id = sr.option(argv, "--target-id")
root = out.parents[1]
with (root / "seen.jsonl").open("a") as f:
    f.write(json.dumps({"argv": argv, "pid": __import__('os').getpid()}) + "\n")
mode = (root / "mode").read_text() if (root / "mode").exists() else "good"
if mode == "missing":
    sys.exit(0)
if mode == "fail":
    sys.exit(3)
stopped = [False]
signal.signal(signal.SIGTERM, lambda *_args: stopped.__setitem__(0, True))
if mode in {"wait", "stop"}:
    (out / "ready").touch()
    if mode == "stop":
        (root / "STOP").touch()
    until = time.monotonic() + 10
    while not stopped[0] and time.monotonic() < until:
        time.sleep(.01)
resolved = campaign_cli.load_previous(Path(sr.option(argv, "--resolved-campaign")))
cpu = sr.option(argv, "--cpu-serving-launch") is not None
selected = legacy_targets.select_target(resolved, target_id, cpu_serving=cpu)
identity = {"campaign_id": resolved.campaign_id, "request_id": resolved.request_id,
            "manifest_digest": resolved.manifest_digest, "selected_id": target_id,
            "scope": "cpu_serving_selected_workload" if cpu else "legacy_gpu_screen",
            "original_target": selected.to_dict()}
count = 0 if stopped[0] else int(sr.option(argv, "--iterations"))
prior = sr.option(argv, "--resume-run")
anchor = Path(sr.option(argv, "--anchor-build"))
if prior:
    original, _ = sr.load_completed(Path(prior), expected_binding=sr.input_binding(argv))
    anchor = Path(original["current_anchor"]["path"])
row = sr.continuation(argv=argv, binding=sr.input_binding(argv),
    terminal="stopped" if stopped[0] else "complete",
    worktree=sr.option(argv, "--worktree"),
    branch=sr.option(argv, "--experimental-branch") if cpu else sr.champion.CANONICAL_BRANCH,
    model=selected.execution.model.path, selected_target=identity,
    anchor_build=anchor, anchor_commit="a" * 40,
    cor_build=None if cpu else anchor, cor_commit=None if cpu else "a" * 40,
    iterations_requested=int(sr.option(argv, "--iterations")),
    outcomes=[SimpleNamespace(status="measured_null") for _ in range(count)])
(out / "loop-run.json").write_text(json.dumps({"fixture": "not measurement evidence"}))
if mode == "wrong_args":
    row["input_argv"].append("--forged")
    row["input_argv_sha256"] = sr._digest(row["input_argv"])
(out / "loop-continuation.json").write_text(json.dumps(row))
'''


def _inputs(tmp_path, monkeypatch, *, mode="good", rounds=2):
    state = tmp_path / "router"
    state.mkdir()
    (state / "mode").write_text(mode)
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    monkeypatch.setattr(sr, "_child_command", lambda argv: [sr.sys.executable, str(child), *argv])
    # Source imports remain pinned to the tested checkout, not ambient installations.
    monkeypatch.setenv("PYTHONPATH", str(Path(sr.__file__).resolve().parents[4]))
    files = []
    for backend in ("cpu", "gpu"):
        target_root = tmp_path / backend
        target_root.mkdir()
        resolved, launch = _resolved(backend=backend, seed=backend == "cpu")
        argv = _argv(target_root, resolved, launch, cpu=backend == "cpu")
        argv.remove("--dry-run")
        argv += ["--worker-root", str(target_root / "workers"),
                 "--worker-build-root", str(target_root / "builds")]
        if backend == "cpu":
            argv += ["--cpu-calibrate-serving", "5"]
        path = target_root / "args.json"
        path.write_text(json.dumps(argv))
        files += ["--target-args", str(path)]
    return state, [*files, "--batch-iterations", "1", "--rounds", str(rounds), "--state-dir", str(state)]


def test_actual_children_rotate_reuse_inputs_and_stop_at_finite_budget(tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch)
    assert sr.main(argv) == 0
    saved = json.loads((state / "serial-state.json").read_text())
    seen = [json.loads(line) for line in (state / "seen.jsonl").read_text().splitlines()]
    assert len(seen) == saved["next_batch"] == 4
    assert [sr.option(row["argv"], "--cpu-serving-launch") is not None for row in seen] == [True, False, True, False]
    assert sr.option(seen[0]["argv"], "--cpu-calibrate-serving") == "5"
    assert sr.option(seen[2]["argv"], "--cpu-calibrate-serving") is None
    assert sr.option(seen[2]["argv"], "--resume-run").endswith("batch-000000/loop-continuation.json")
    assert sr.option(seen[3]["argv"], "--resume-run").endswith("batch-000001/loop-continuation.json")
    assert len({sr.option(row["argv"], "--out") for row in seen}) == 4
    assert saved["active"] is None and not saved["failed_targets"]
    assert sr.main(argv) == 0  # Completed restart does not replay four iterations.
    assert len((state / "seen.jsonl").read_text().splitlines()) == 4
    for row in seen:
        with pytest.raises(ProcessLookupError):
            os.kill(row["pid"], 0)


@pytest.mark.parametrize("mode", ["missing", "wrong_args", "fail"])
def test_no_success_without_matching_terminal_and_all_failed_continuous_exits(
        tmp_path, monkeypatch, mode):
    state, argv = _inputs(tmp_path, monkeypatch, mode=mode, rounds=0)
    assert sr.main(argv) == 1
    saved = json.loads((state / "serial-state.json").read_text())
    assert len(saved["failed_targets"]) == 2
    assert not saved["last_results"] and saved["active"] is None
    assert len((state / "seen.jsonl").read_text().splitlines()) == 2


def test_stop_during_child_drains_captured_pid_and_persists_across_restart(tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, mode="stop", rounds=0)
    assert sr.main(argv) == 0
    seen = [json.loads(line) for line in (state / "seen.jsonl").read_text().splitlines()]
    assert len(seen) == 1 and (state / "STOP").exists()
    row, _ = sr.load_completed(state / "batches/batch-000000/loop-continuation.json")
    assert row["terminal"] == "stopped" and row["iterations_completed"] == 0
    assert sr.main(argv) == 0
    assert len((state / "seen.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(seen[0]["pid"], 0)


def test_post_popen_status_failure_drains_before_another_child(tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, mode="wait", rounds=1)
    original = sr.status.write
    spawned = []
    real_popen = sr.subprocess.Popen

    def spawn(*args, **kwargs):
        assert not spawned or spawned[-1].poll() is not None
        child = real_popen(*args, **kwargs)
        spawned.append(child)
        return child

    def write(*args, **kwargs):
        if kwargs["state"] == "running":
            # A real I/O fault immediately after Popen, before its wait loop.
            # Stop the session too: there must not be a second target launch.
            (state / "STOP").touch()
            raise OSError("fixture status publication failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(sr.subprocess, "Popen", spawn)
    monkeypatch.setattr(sr.status, "write", write)
    assert sr.main(argv) == 1
    assert len(spawned) == 1 and spawned[0].poll() is not None
    saved = json.loads((state / "serial-state.json").read_text())
    assert saved["active"] is None


def test_continuation_does_not_read_large_original_measurement_result(tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, rounds=1)
    assert sr.main(argv) == 0
    directory = state / "batches/batch-000000"
    with (directory / "loop-run.json").open("wb") as stream:
        stream.truncate(400 * 1024 * 1024)  # Sparse fixture, not a model or measurement.
    original = sr._read

    def read(path, **kwargs):
        assert path.name != "loop-run.json"
        return original(path, **kwargs)

    monkeypatch.setattr(sr, "_read", read)
    row, _ = sr.load_completed(directory / "loop-continuation.json")
    assert row["iterations_completed"] == 1


@pytest.mark.parametrize("field", ["current_anchor", "selected_target", "binding", "terminal", "result_file"])
def test_continuation_tampering_refused(tmp_path, monkeypatch, field):
    state, argv = _inputs(tmp_path, monkeypatch, rounds=1)
    assert sr.main(argv) == 0
    path = state / "batches/batch-000000/loop-continuation.json"
    row = json.loads(path.read_text())
    expected = row["binding"]
    row[field] = None
    path.write_text(json.dumps(row))
    with pytest.raises(sr.SerialRefused):
        sr.load_completed(path, expected_binding=expected)


def test_existing_run_emits_actual_current_and_cor_then_resumes_without_anchor_rebuild():
    fixture = promotion_fixture.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        original_main = run.main
        batch = [0]
        first = fixture.root / "batch0"
        second = fixture.root / "batch1"
        starts = []
        real_startup = run.champion.verify_startup

        def startup(**kwargs):
            starts.append(kwargs["anchor_build"])
            return real_startup(**kwargs)

        def invoke(argv):
            out = first if batch[0] == 0 else second
            extra = [] if batch[0] == 0 else ["--resume-run", str(first / "loop-continuation.json")]
            return original_main([*argv, "--out", str(out), *extra])

        with mock.patch.object(run, "main", invoke), \
                mock.patch.object(run.champion, "verify_startup", startup):
            rc, _calls, _planners, _scratch, log = fixture._run_one_keep()
            assert rc == 0, log
            original, _ = sr.load_completed(first / "loop-continuation.json")
            assert original["current_anchor"]["path"] != str(fixture.startup_anchor)
            assert original["cor_anchor"] == {"path": str(fixture.startup_anchor), "commit": fixture.tip}
            # Ordinary restart must not silently point the old COR at the new tip.
            def missing_cor(argv):
                argv = list(argv)
                argv[argv.index("--anchor-build") + 1] = original["current_anchor"]["path"]
                return original_main(argv)

            with mock.patch.object(run, "main", missing_cor), \
                    pytest.raises(run.champion.StartupRefused, match="original --cor-build"):
                fixture._run_one_keep()
            # Existing ancestor acceptance is insufficient for an exact COR join.
            with pytest.raises(sr.SerialRefused, match="exact recorded arm"):
                sr.verify_exact_anchor(fixture.startup_anchor, fixture.repo,
                                       original["current_anchor"]["commit"])
            sr.verify_exact_anchor(fixture.startup_anchor, fixture.repo, fixture.tip[:12])
            batch[0] = 1
            rc, calls, _planners, _scratch, log = fixture._run_one_keep()
            assert rc == 0, log
        assert starts[-1] == Path(original["current_anchor"]["path"])
        assert all(call["dest"] != Path(original["current_anchor"]["path"]) for call in calls)
        resumed, _ = sr.load_completed(second / "loop-continuation.json")
        assert resumed["cor_anchor"] == original["cor_anchor"]
        assert Path(original["cor_anchor"]["path"]).exists()
    finally:
        fixture.doCleanups()


def test_actual_cpu_post_keep_resume_rebinds_original_launch_without_recalibration():
    original_main = run.main
    resumed = []

    def first_then_resume_dry_run(argv):
        assert original_main(argv) == 0
        first = Path(sr.option(argv, "--out"))
        row, _ = sr.load_completed(first / "loop-continuation.json")
        assert row["cor_anchor"] is None
        assert row["current_anchor"]["path"] != sr.option(argv, "--anchor-build")
        # Keep all original recipe/request arguments and source identity checks.
        # Only this second pass is dry-run; the original five-iteration fixture
        # already used the real pool/keep path and synthetic hardware observations.
        with mock.patch.object(run.serving, "calibrate_floor", side_effect=AssertionError("recalibration")), \
                mock.patch.object(run, "_cpu_arm", wraps=run._cpu_arm) as rebind:
            rc = original_main([*argv, "--resume-run", str(first / "loop-continuation.json"),
                                "--out", str(first.parent / "resumed"), "--dry-run"])
        assert rc == 0
        assert Path(rebind.call_args.args[1]) == Path(row["current_anchor"]["path"])
        resumed.append(row)
        return 0

    with mock.patch.object(run, "main", first_then_resume_dry_run):
        cpu_fixture.test_existing_main_cpu_five_iterations_preserves_canonical_champion(False)
    assert len(resumed) == 1
