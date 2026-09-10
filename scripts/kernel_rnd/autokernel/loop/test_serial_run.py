"""Owned tiny children and original loop fixtures; no hardware or providers."""
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import pytest

from . import cpu_screen, run, scheduling, serial_run as sr, serial_scheduling as ss, source_loo
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
held = None
selection_path = sr.option(argv, "--scheduler-selection")
if selection_path and mode == "original_cost":
    # Actual original private flock/PID/interval writer, no hardware or measurement.
    import fcntl
    from scripts.kernel_rnd.autokernel.loop import claim, scheduling
    from scripts.kernel_rnd.autokernel.loop.measurement_capture import ArtifactStore
    selection = scheduling.Selection.from_dict(json.loads(Path(selection_path).read_text()))
    with (root / "fixture-cost.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owner = claim.HeldCpuClaim({"device_id": "cpu", "cpu_list": "0"},
            [root / "fixture-cost.lock"],
            region_fraction=selection.proposal.estimated_claims.physical_region_fraction,
            affinity=("0",))
        try:
            time.sleep(.02)
        finally:
            owner._closing()
            fcntl.flock(lock, fcntl.LOCK_UN)
            owner._released_now()
    store = ArtifactStore(out / "held-claim-artifacts")
    try:
        artifact = claim.publish_intervals(store, selection, [owner], target=identity)
    finally:
        store.close()
    held = {"schema": sr.HELD_REFERENCE_SCHEMA, "selection_digest": selection.digest,
            "evidence": artifact.to_dict()}
elif selection_path:
    from scripts.kernel_rnd.autokernel.loop import scheduling, serial_scheduling as ss
    from scripts.kernel_rnd.autokernel.loop.measurement_capture import ArtifactStore
    selection = scheduling.Selection.from_dict(json.loads(Path(selection_path).read_text()))
    if mode == "fail_preclaim":
        (out / "loop-preclaim-failure.json").write_text(json.dumps({
            "schema": "epyc.autokernel.preclaim_failure.v1",
            "selection_digest": selection.digest, "target": identity,
            "error_type": "ClaimRefused"}))
        sys.exit(3)
    proposal = selection.proposal
    def observation(pid, suffix):
        inode = sum(map(ord, suffix))
        return {"observed_at": 1.0, "started_monotonic_s": 1.0,
                "ended_monotonic_s": 1.1, "owner_pid": pid, "error": None,
                "status": "held", "locks": [{"path": "/locks/" + suffix,
                "device": 1, "inode": inode, "path_unchanged": True,
                "owners": [{"pid": pid, "kernel_row": "original"}],
                "same_holder": True}]}
    def component(device, start, end, fraction, suffix, gpu):
        pid = __import__('os').getpid()
        domain = {"kind": "direct_loop", "clock": "monotonic", "pid": pid,
                  "boot_id": "fixture-boot", "process_start_ticks": 1, "error": None}
        opened = observation(pid, suffix)
        inode = opened["locks"][0]["inode"]
        return {"context_id": ss._digest({"domain": domain, "started_at": start,
                                          "locks": opened["locks"]}),
                "domain": domain, "ownership_generation": 1, "allocation_generation": 1,
                "started_at": start, "ended_at": end, "device_id": device,
                "physical_claim_ids": [f"fixture-boot:flock:1:{inode}"],
                "physical_region_fraction": fraction, "gpu_device_ids": gpu,
                "memory_reservation_bytes": proposal.estimated_claims.memory_reservation_bytes,
                "affinity_cores": ["0"], "open": opened,
                "close": observation(pid, suffix), "released": True}
    base = float(int(out.name.rsplit("-", 1)[-1]) * 100)
    components = [component("cpu", base + 1.0, base + 4.0,
                            proposal.estimated_claims.physical_region_fraction,
                            "cpu", [])]
    if proposal.backend == "gpu":
        components[0]["ended_at"] = base + 6.0
        components.append(component(proposal.estimated_claims.gpu_devices[0], base + 2.0,
                                    base + 5.0,
                                    0.0, "gpu", list(proposal.estimated_claims.gpu_devices)))
    store = ArtifactStore(out / "held-claim-artifacts")
    try:
        artifact = store.write("direct-held-intervals", {
            "schema": ss.INTERVAL_SCHEMA, "selection": selection.to_dict(),
            "selection_digest": selection.digest, "target": identity,
            "components": components}).to_dict()
    finally:
        store.close()
    held = {"schema": ss.REFERENCE_SCHEMA, "selection_digest": selection.digest,
            "evidence": artifact}
    if mode == "fail_postclaim_publish":
        sys.exit(3)
    if mode == "fail_held":
        (out / "loop-held-claims.json").write_text(json.dumps(held))
        sys.exit(3)
    if mode == "gpu_missing_zero" and target_id == "gpu":
        (out / "loop-held-claims.json").write_text(json.dumps(held))
        sys.exit(0)
result_only_failure = mode == "gpu_result_only_failure" and target_id == "gpu"
count = 0 if stopped[0] else int(sr.option(argv, "--iterations"))
prior = sr.option(argv, "--resume-run")
anchor = Path(sr.option(argv, "--anchor-build"))
if prior:
    original, _ = sr.load_completed(Path(prior), expected_binding=sr.resume_binding(argv))
    anchor = Path(original["current_anchor"]["path"])
row = sr.continuation(argv=argv, binding=sr.input_binding(argv),
    terminal="stopped" if stopped[0] else "complete",
    worktree=sr.option(argv, "--worktree"),
    branch=sr.option(argv, "--experimental-branch") if cpu else sr.champion.CANONICAL_BRANCH,
    model=selected.execution.model.path, selected_target=identity,
    anchor_build=anchor, anchor_commit="a" * 40,
    cor_build=None if cpu else anchor, cor_commit=None if cpu else "a" * 40,
    iterations_requested=int(sr.option(argv, "--iterations")),
    outcomes=[SimpleNamespace(status="bench_failed") for _ in range(count)]
    if result_only_failure else
    [SimpleNamespace(status="measured_null") for _ in range(count)],
    **({"runtime_recipe_reference": {"locator": "runtime-selection-fixture",
       "sha256": "b" * 64, "verified": True}}
       if mode == "runtime_ref" and cpu else {}),
    **({"held_claim_evidence": held} if held else {}))
if result_only_failure:
    (out / "loop-run.json").write_text(json.dumps({
        "schema": "epyc.autokernel.loop_run.v1",
        "epoch": "e" * 64, "anchor_commit": "a" * 40,
        "surface": "serving:fixture", "pairs": 5, "noise_floor_pct": 1.0,
        "elapsed_s": 1.0, "workers": 1,
        "iterations": [{"status": "bench_failed", "turn_recorded_at":
                        "2026-09-10T00:00:00Z", "reason": "GPU setup failed"}],
        "phase_seconds": {"setup": 1.0}, "phase_seconds_are_lane_seconds": True,
        "pool": {"workers": 1, "wall_seconds": 1.0, "tail_seconds": 1.0,
                 "tail_fraction": 1.0, "superseded": 0},
        "continuation": row,
        "target": identity,
        "runtime_preparation": {"status": "fixture"},
        "launch_snapshot": "f" * 64, "floor_request_digest": "d" * 64,
        **({"held_claim_evidence": held} if held else {}),
    }))
    sys.exit(3)  # Crash/failure after the full result, before the routing receipt.
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
        resolved, launch = _resolved(backend=backend, seed=backend == "cpu",
                                     changes={"target_id": backend})
        argv = _argv(target_root, resolved, launch, cpu=backend == "cpu")
        argv[argv.index("--target-id") + 1] = backend
        argv.remove("--dry-run")
        argv += ["--worker-root", str(target_root / "workers"),
                 "--worker-build-root", str(target_root / "builds")]
        if backend == "cpu":
            argv += ["--cpu-calibrate-serving", "5"]
        path = target_root / "args.json"
        path.write_text(json.dumps(argv))
        files += ["--target-args", str(path)]
    return state, [*files, "--batch-iterations", "1", "--rounds", str(rounds), "--state-dir", str(state)]


def _scheduled(tmp_path, argv):
    from .test_scheduling import config, proposal, vector
    target_files = [Path(argv[index + 1]) for index, value in enumerate(argv)
                    if value == "--target-args"]
    targets = [json.loads(path.read_text()) for path in target_files]
    bindings = sr._scheduler_bindings(targets)
    rows = {}
    for selected_id, binding in bindings.items():
        backend = binding["backend"]
        rows[selected_id] = proposal(
            selected_id, backend=backend, target=binding["target_revision"],
            alias=binding["alias_identity"],
            claims=vector(fraction=0.5,
                          gpus=("mi210_0",) if backend == "gpu" else (), memory=0))
        rows[selected_id]["eligibility_ref"] = binding["eligibility_ref"]
    manifest = {"schema": ss.MANIFEST_SCHEMA, "scheduler_id": "serial-integration",
                "config": config(noncoverage_slots=2,
                                 capacity=vector(gpus=("mi210_0",), memory=10000)),
                "targets": rows}
    path = tmp_path / "scheduler.json"
    path.write_text(json.dumps(manifest))
    return [*argv, "--scheduler-manifest", str(path)]


def test_scheduled_actual_children_bind_fresh_selection_and_account_original_intervals(
        tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, rounds=2)
    argv = _scheduled(tmp_path, argv)
    assert sr.main(argv) == 0
    saved = json.loads((state / "serial-state.json").read_text())
    scheduler_state = scheduling.SchedulerState.from_dict(saved["scheduler_state"])
    assert scheduler_state.campaign_attempts == 4
    assert len(scheduler_state.accounted_receipts) >= 4
    seen = [json.loads(line)["argv"] for line in (state / "seen.jsonl").read_text().splitlines()]
    assert all(sr.option(row, "--scheduler-selection") for row in seen)
    assert sr.option(seen[0], "--scheduler-selection") != sr.option(
        seen[1], "--scheduler-selection")
    assert sr.option(seen[1], "--resume-run").endswith(
        "batch-000000/loop-continuation.json")
    assert sr.input_binding(seen[0]) == sr.input_binding(seen[1])


def test_scheduled_failed_child_accounts_only_original_released_claim(tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, mode="fail_held", rounds=1)
    argv = _scheduled(tmp_path, argv)
    assert sr.main(argv) == 1
    saved = json.loads((state / "serial-state.json").read_text())
    scheduler_state = scheduling.SchedulerState.from_dict(saved["scheduler_state"])
    assert scheduler_state.campaign_attempts == 2
    assert {record.outcome for record in scheduler_state.accounted_receipts} == {"failed"}
    assert len(saved["failed_targets"]) == 2
    assert "cost_forecast" not in saved  # released failed prefixes are charged, not successful forecasts


def test_scheduled_preclaim_failures_settle_each_selection_without_held_evidence(
        tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, mode="fail_preclaim", rounds=1)
    argv = _scheduled(tmp_path, argv)
    assert sr.main(argv) == 1
    saved = json.loads((state / "serial-state.json").read_text())
    scheduler_state = scheduling.SchedulerState.from_dict(saved["scheduler_state"])
    assert scheduler_state.campaign_attempts == 0
    assert scheduler_state.issued_selection_digests == ()
    assert scheduler_state.accounted_receipts == ()
    assert scheduler_state.receipts == ()
    assert len(saved["failed_targets"]) == 2
    assert not list((state / "batches").glob("*/loop-held-claims.json"))
    seen = (state / "seen.jsonl").read_bytes()
    assert sr.main(argv) == 1
    assert (state / "seen.jsonl").read_bytes() == seen


def test_postclaim_publication_failure_cannot_use_preclaim_settlement(tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, mode="fail_postclaim_publish", rounds=1)
    argv = _scheduled(tmp_path, argv)
    with pytest.raises(ss.SerialSchedulingRefused,
                       match="issued selection awaits settlement"):
        sr.main(argv)
    saved = json.loads((state / "serial-state.json").read_text())
    scheduler_state = scheduling.SchedulerState.from_dict(saved["scheduler_state"])
    assert scheduler_state.issued_selection_digests
    assert scheduler_state.campaign_attempts == 0
    assert scheduler_state.receipts == ()
    assert len(saved["failed_targets"]) == 1
    assert not list((state / "batches").glob("*/loop-preclaim-failure.json"))


def test_scheduled_restart_accounts_original_completed_child_before_new_selection(
        tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, rounds=1)
    argv = _scheduled(tmp_path, argv)
    with mock.patch.object(sr, "_scheduled_account", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            sr.main(argv)
    crashed = json.loads((state / "serial-state.json").read_text())
    assert crashed["active"] is not None and crashed["next_batch"] == 0
    assert sr.main(argv) == 0
    recovered = json.loads((state / "serial-state.json").read_text())
    scheduler_state = scheduling.SchedulerState.from_dict(recovered["scheduler_state"])
    assert scheduler_state.campaign_attempts == 2
    assert recovered["last_reconciliation"]["batch_number"] == 0
    assert len((state / "seen.jsonl").read_text().splitlines()) == 2


def test_original_child_cost_changes_next_selection_and_recovers_once(tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, mode="original_cost", rounds=3)
    argv = argv[:2] + argv[4:]  # One synthetic CPU target; no real host resource locks.
    target_path = Path(argv[1])
    target = sr._without(json.loads(target_path.read_text()), {"--cpu-calibrate-serving"})
    target_path.write_text(json.dumps(target))
    argv = _scheduled(tmp_path, argv)
    original_account = sr._scheduled_account

    def crash_after_second_original_account(*args, **kwargs):
        result = original_account(*args, **kwargs)
        if args[0]["scheduler_state"]["campaign_attempts"] == 1:
            raise KeyboardInterrupt("original second child completed before state save")
        return result

    with mock.patch.object(sr, "_scheduled_account", side_effect=crash_after_second_original_account):
        with pytest.raises(KeyboardInterrupt):
            sr.main(argv)
    first = json.loads((state / "batches/batch-000000/scheduler-selection.json").read_text())
    second = json.loads((state / "batches/batch-000001/scheduler-selection.json").read_text())
    crashed = json.loads((state / "serial-state.json").read_text())
    samples = crashed["cost_forecast"]["targets"]["cpu"]["samples"]
    assert len(samples) == 1
    assert second["proposal"]["estimated_duration_seconds"] == samples[0]["held_seconds"]
    assert second["proposal"]["estimated_duration_seconds"] < first["proposal"]["estimated_duration_seconds"]
    assert sr.main(argv) == 0
    saved = json.loads((state / "serial-state.json").read_text())
    assert saved["scheduler_state"]["campaign_attempts"] == 3
    assert saved["last_reconciliation"]["batch_number"] == 1
    assert len(saved["cost_forecast"]["targets"]["cpu"]["samples"]) == 3
    assert saved["cost_forecast"]["policy"] == ss.COST_FORECAST_POLICY
    seen = [json.loads(row) for row in (state / "seen.jsonl").read_text().splitlines()]
    assert len(seen) == 3 and all(not Path(f"/proc/{row['pid']}").exists() for row in seen)
    assert sr.main(argv) == 0
    assert json.loads((state / "serial-state.json").read_text())["cost_forecast"] == saved["cost_forecast"]
    assert len((state / "seen.jsonl").read_text().splitlines()) == 3


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


def test_gpu_full_result_recovers_missing_continuation_without_inventing_measured_null(
        tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch,
                          mode="gpu_result_only_failure", rounds=1)
    argv = ["--target-args", argv[3], *argv[4:]]
    argv = _scheduled(tmp_path, argv)
    assert sr.main(argv) == 0
    saved = json.loads((state / "serial-state.json").read_text())
    gpu_path = state / "batches/batch-000000/loop-continuation.json"
    gpu, gpu_sha = sr.load_completed(gpu_path)
    assert gpu["outcome_counts"] == {"bench_failed": 1}
    assert saved["last_results"]["0"] == {
        "path": str(gpu_path), "sha256": gpu_sha}
    assert saved["failed_targets"] == {}
    assert {row["outcome"] for row in
            saved["scheduler_state"]["accounted_receipts"]} == {"failed"}
    full = json.loads((gpu_path.parent / "loop-run.json").read_text())
    assert full["iterations"][0]["reason"] == "GPU setup failed"


def test_zero_exit_missing_gpu_terminal_settles_scheduler_failure(tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, mode="gpu_missing_zero", rounds=1)
    argv = ["--target-args", argv[3], *argv[4:]]
    argv = _scheduled(tmp_path, argv)
    assert sr.main(argv) == 1
    saved = json.loads((state / "serial-state.json").read_text())
    assert saved["failed_targets"]
    assert {row["outcome"] for row in
            saved["scheduler_state"]["accounted_receipts"]} == {"failed"}
    assert saved["active"] is None and saved["next_batch"] == 1


def test_runtime_recipe_reference_is_target_local_digest_bound_and_replay_stable(
        tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, mode="runtime_ref", rounds=2)
    assert sr.main(argv) == 0
    seen = [json.loads(line)["argv"]
            for line in (state / "seen.jsonl").read_text().splitlines()]
    assert sr.option(seen[0], "--runtime-recipe-reference") is None
    assert sr.option(seen[1], "--runtime-recipe-reference") is None  # GPU has no CPU recipe.
    reference_path = Path(sr.option(seen[2], "--runtime-recipe-reference"))
    assert reference_path.name == "runtime-recipe-reference.json"
    assert json.loads(reference_path.read_text()) == {
        "locator": "runtime-selection-fixture", "sha256": "b" * 64, "verified": True}
    assert sr.option(seen[3], "--runtime-recipe-reference") is None
    # Recovery recreates the same child argv and accepts only the immutable bytes.
    first_path = state / "batches/batch-000000/loop-continuation.json"
    prior = {"path": str(first_path),
             "sha256": hashlib.sha256(first_path.read_bytes()).hexdigest()}
    replayed = sr._batch_argv(
        json.loads(Path(argv[1]).read_text()), prior, 1, reference_path.parent)
    assert replayed == seen[2]
    reference_path.write_text(json.dumps({
        "locator": "changed", "sha256": "b" * 64, "verified": True}))
    with pytest.raises(sr.SerialRefused, match="routing reference changed"):
        sr._batch_argv(json.loads(Path(argv[1]).read_text()), prior, 1,
                       reference_path.parent)


def test_serialized_targets_without_source_keeps_preserve_only_own_resume(
        tmp_path, monkeypatch):
    state, argv = _inputs(tmp_path, monkeypatch, rounds=2)
    paths = [Path(argv[index + 1]) for index, value in enumerate(argv)
             if value == "--target-args"]
    first = json.loads(paths[0].read_text())
    second = json.loads(paths[1].read_text())
    second[second.index("--worktree") + 1] = sr.option(first, "--worktree")
    first += ["--experimental-branch", sr.champion.CANONICAL_BRANCH]
    paths[0].write_text(json.dumps(first))
    paths[1].write_text(json.dumps(second))
    assert sr.main(argv) == 0
    seen = [json.loads(line)["argv"]
            for line in (state / "seen.jsonl").read_text().splitlines()]
    assert sr.option(seen[0], "--source-anchor-continuation") is None
    assert sr.option(seen[1], "--source-anchor-continuation") is None
    assert sr.option(seen[2], "--resume-run").endswith(
        "batch-000000/loop-continuation.json")
    assert sr.option(seen[2], "--source-anchor-continuation") is None
    assert sr.option(seen[2], "--source-anchor-sha256") is None


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
        if args[0][0] == "git":
            return real_popen(*args, **kwargs)
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


@pytest.mark.parametrize("validation_failure,cross_worktree", [
    (False, False), (True, False), (False, True), ("gate", True),
])
def test_actual_cpu_keep_cross_target_then_older_history_uses_latest_source(
        validation_failure, cross_worktree):
    original_main = run.main
    observed = []

    def exercise(argv):
        assert original_main(argv) == 0
        first = Path(sr.option(argv, "--out"))
        first_row, first_sha = sr.load_completed(first / "loop-continuation.json")
        assert first_row["selected_target"]["selected_id"] == "target-a"
        source_keeps = (first_row.get("source_lineage_keeps")
                        or first_row.get("experimental_source_keeps"))
        assert isinstance(source_keeps, list) and len(source_keeps) == 1
        origin_receipt = run.surface_fold.reopen_reference(
            source_keeps[0])
        origin_capture_id = origin_receipt.comparison["belief_capture"]["capture_id"]
        origin_export = (Path(sr.option(argv, "--store")) / "serving-beliefs"
                         / f"{origin_capture_id}.json")
        assert origin_export.is_file()
        origin_export_bytes = origin_export.read_bytes()

        second = first.parent / "target-b"
        target_b = list(argv)
        target_b[target_b.index("--target-id") + 1] = "target-b"
        target_b[target_b.index("--store") + 1] = str(first.parent / "store-b")
        if cross_worktree:
            target_b_tree = first.parent / "target-b-tree"
            target_b_branch = "ak/experimental/cpu-target-b"
            run._git(Path(first_row["worktree"]), "worktree", "add", "-b",
                     target_b_branch, str(target_b_tree), origin_receipt.parent_commit)
            assert run._git(target_b_tree, "branch", "--show-current") == target_b_branch
            target_b[target_b.index("--worktree") + 1] = str(target_b_tree)
            target_b[target_b.index("--experimental-branch") + 1] = target_b_branch
        target_b += ["--source-anchor-continuation", str(first / "loop-continuation.json"),
                     "--source-anchor-sha256", first_sha, "--validate-source-continuation",
                     "--iterations", "1", "--out", str(second)]
        scheduler_manifest = sr._derived_scheduler_manifest(
            (target_b,), Path(sr.option(target_b, "--resolved-campaign")), 1)
        scheduler_state = scheduling.initial_state(
            scheduler_manifest.config, scheduler_manifest.scheduler_id)
        _state, validation_selection, _index = ss.select_target(
            scheduler_manifest, scheduler_state, ("target-b",), now=1,
            stage_number=0, validation_ids=frozenset({"target-b"}))
        selection_path = first.parent / "target-b-validation-selection.json"
        selection_path.write_text(json.dumps(validation_selection.to_dict()))
        assert original_main([*target_b, "--scheduler-selection", str(selection_path),
                              "--out", str(first.parent / "target-b-dry"), "--dry-run"]) == 0
        observer = run.serving._measure_once
        observer_state = dict(zip(observer.__code__.co_freevars,
                                  (cell.cell_contents for cell in observer.__closure__)))
        held_rows = observer_state["held"]
        @cpu_fixture.contextmanager
        def validation_hold(cpu_list):
            assert cpu_list == "0-95"
            assert held_rows[-1] is False
            held_rows[-1] = True
            try:
                yield {"device_id": "cpu", "regions": ["q0", "q1", "q2", "q3"]}
            finally:
                held_rows[-1] = False

        if not validation_failure:
            origin = first.parent / "target-a-origin-validation"
            origin_argv = [*argv, "--resume-run", str(first / "loop-continuation.json"),
                "--source-anchor-continuation", str(first / "loop-continuation.json"),
                "--source-anchor-sha256", first_sha, "--validate-source-continuation",
                "--iterations", "1", "--out", str(origin)]
            replayed_exports = []
            with mock.patch.object(run.claim, "hold_cpu", validation_hold), \
                    mock.patch.object(run.serving, "compare",
                                      side_effect=AssertionError("exact keep must be reused")), \
                    mock.patch.object(run.serving, "calibrate_floor",
                                      side_effect=AssertionError("exact keep must not recalibrate")), \
                    mock.patch.object(run.serving_beliefs.PlannerFeedback, "exported",
                                      lambda _self, path: replayed_exports.append(Path(path))):
                assert original_main(origin_argv) == 0
            assert replayed_exports == [origin_export]
            assert origin_export.read_bytes() == origin_export_bytes
            origin_row, _sha = sr.load_completed(origin / "loop-continuation.json")
            origin_validation = run.surface_validation.reopen_reference(
                origin_row["source_validation"])
            assert origin_validation["intended_target"] is True
            assert origin_validation["disposition"] == "passed"

        validation_result = (mock.patch.object(
            run.serving, "compare", side_effect=run.loop.MeasurementInvalid(
                "synthetic invalid target comparison", {"status": "measurement_invalid"}))
            if validation_failure is True else nullcontext())
        original_compile = run.gates.compiles

        def validation_compile(*args, **kwargs):
            if Path(args[1]).name == "whole-source-candidate-build":
                return run.gates.Verdict("compile", False, "synthetic target recipe refusal")
            return original_compile(*args, **kwargs)

        gate_result = (mock.patch.object(run.gates, "compiles", validation_compile)
                       if validation_failure == "gate" else nullcontext())
        original_oracle = run.gates.op_correctness
        validation_oracles = []

        def validation_oracle(build, **kwargs):
            if Path(build).name == "whole-source-candidate-build":
                validation_oracles.append((Path(build), kwargs))
                return run.gates.Verdict("correctness", True, "synthetic target oracle")
            return original_oracle(build, **kwargs)

        oracle_result = (mock.patch.object(run.gates, "op_correctness", validation_oracle)
                         if cross_worktree else nullcontext())
        validation_exports = []
        with mock.patch.object(run.claim, "hold_cpu", validation_hold), validation_result, \
                gate_result, oracle_result, \
                mock.patch.object(run.serving_beliefs.PlannerFeedback, "exported",
                                  lambda _self, path: validation_exports.append(Path(path))):
            assert original_main(target_b) == 0
        second_row, second_sha = sr.load_completed(second / "loop-continuation.json")
        assert second_row["selected_target"]["selected_id"] == "target-b"
        # A validation stage consumes the propagated source without authoring
        # another candidate; ordinary research can continue in later stages.
        if cross_worktree:
            assert second_row["current_anchor"]["commit"] == origin_receipt.parent_commit
            assert second_row["current_anchor"] != first_row["current_anchor"]
        else:
            assert second_row["current_anchor"] == first_row["current_anchor"]
        validation = run.surface_validation.reopen_reference(second_row["source_validation"])
        assert validation["target"]["selected_id"] == "target-b"
        if not cross_worktree:
            assert validation["candidate_anchor"] == first_row["current_anchor"]
        assert validation["original_anchor"]["commit"] != validation["source_commit"]
        assert validation["disposition"] == ("pending" if validation_failure else "passed")
        assert validation["source_keep_ids"]
        if validation_failure:
            assert validation_exports == []
        else:
            assert len(validation_exports) == 1
            assert validation_exports[0].is_file()
            assert validation_exports[0] != origin_export
        if cross_worktree:
            candidate_build = second / "whole-source-candidate-build"
            assert validation["candidate_anchor"] == {
                "path": str(candidate_build.resolve()),
                "commit": first_row["current_anchor"]["commit"]}
            if validation_failure == "gate":
                assert validation["recipe_execution_digest"] is None
                assert validation["failure"]["type"] == "target_recipe_gate_refused"
                assert validation_oracles == []
            else:
                assert candidate_build.is_dir()
                assert validation_oracles[0][0] == candidate_build
                assert validation_oracles[0][1]["backend"] == "CPU"
                authored = first.parent / "target-b-shared-author"
                authored_argv = [*target_b[:target_b.index("--source-anchor-continuation")],
                    "--resume-run", str(second / "loop-continuation.json"),
                    "--source-anchor-continuation", str(first / "loop-continuation.json"),
                    "--source-anchor-sha256", first_sha, "--iterations", "4",
                    "--out", str(authored)]
                assert original_main(authored_argv) == 0
                authored_row, authored_sha = sr.load_completed(
                    authored / "loop-continuation.json")
                assert run._git(Path(first_row["worktree"]), "merge-base", "--is-ancestor",
                                first_row["current_anchor"]["commit"],
                                authored_row["current_anchor"]["commit"]) == ""
                assert authored_row["worktree"] == str(target_b_tree.resolve())
                b_next = first.parent / "target-b-shared-resume"
                assert original_main([*target_b[:target_b.index(
                    "--source-anchor-continuation")], "--resume-run",
                    str(authored / "loop-continuation.json"),
                    "--source-anchor-continuation",
                    str(authored / "loop-continuation.json"), "--source-anchor-sha256",
                    authored_sha, "--iterations", "1", "--out", str(b_next),
                    "--dry-run"]) == 0
                a_validation = first.parent / "target-a-shared-validation"
                a_validation_argv = list(argv)
                a_validation_argv[a_validation_argv.index("--anchor-build") + 1] = \
                    first_row["current_anchor"]["path"]
                assert original_main([*a_validation_argv, "--resume-run",
                    str(first / "loop-continuation.json"),
                    "--source-anchor-continuation",
                    str(authored / "loop-continuation.json"), "--source-anchor-sha256",
                    authored_sha, "--validate-source-continuation", "--iterations", "1",
                    "--out", str(a_validation)]) == 0
                a_validation_row, _a_validation_sha = sr.load_completed(
                    a_validation / "loop-continuation.json")
                assert run.surface_validation.reopen_reference(
                    a_validation_row["source_validation"])["disposition"] == "passed"
                loo_references = []
                for name, disposition in (("loo", "neutral"),
                                          ("rebaseline", "passed")):
                    path = first.parent / f"{name}-result.json"
                    raw = json.dumps({"operation": name, "disposition": disposition,
                        "deletion_authorized": False,
                        "target": first_row["selected_target"],
                        "assembled_commit": authored_row["current_anchor"]["commit"]},
                                     sort_keys=True).encode() + b"\n"
                    path.write_bytes(raw)
                    loo_references.append({"path": str(path),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "disposition": disposition})
                loo_calls = []
                def execute_loo(**kwargs):
                    assert held_rows[-1] is True
                    loo_calls.append(kwargs)
                    return {"loo": [loo_references[0]],
                            "rebaseline": loo_references[1]}
                a_loo = first.parent / "target-a-source-loo"
                with mock.patch.object(source_loo, "execute_surface",
                                       side_effect=execute_loo), \
                        mock.patch.object(run.serving, "compare",
                            side_effect=AssertionError("exact validation must not remeasure")), \
                        mock.patch.object(run.gates, "compiles",
                            side_effect=AssertionError("exact validation must not rebuild")):
                    assert original_main([*a_validation_argv, "--resume-run",
                        str(a_validation / "loop-continuation.json"),
                        "--source-anchor-continuation",
                        str(authored / "loop-continuation.json"), "--source-anchor-sha256",
                        authored_sha, "--validate-source-continuation",
                        "--validate-source-loo", "--iterations", "1",
                        "--out", str(a_loo)]) == 0
                a_loo_row, _a_loo_sha = sr.load_completed(
                    a_loo / "loop-continuation.json")
                assert len(loo_calls) == 1, run.surface_validation.reopen_reference(
                    a_loo_row["source_validation"])
                assert a_loo_row["source_loo"] == {
                    "loo": [loo_references[0]], "rebaseline": loo_references[1]}
                a_resume = first.parent / "target-a-shared-resume"
                assert original_main([*argv, "--resume-run",
                    str(a_loo / "loop-continuation.json"),
                    "--source-anchor-continuation",
                    str(authored / "loop-continuation.json"), "--source-anchor-sha256",
                    authored_sha, "--iterations", "1", "--out", str(a_resume)]) == 0
                a_resume_row, _a_resume_sha = sr.load_completed(
                    a_resume / "loop-continuation.json")
                assert run._git(Path(first_row["worktree"]), "merge-base", "--is-ancestor",
                                authored_row["current_anchor"]["commit"],
                                a_resume_row["current_anchor"]["commit"]) == ""
                assert a_resume_row["selected_target"]["selected_id"] == "target-a"
                assert a_resume_row["worktree"] == first_row["worktree"]
                observed.append((first_row, a_resume_row))
                return 0
            # A newer keep can advance the live producer checkout without making
            # this immutable target-B verdict unreadable.
            newer = first.parent / "target-a-newer-source"
            assert original_main([*argv, "--resume-run",
                                  str(first / "loop-continuation.json"),
                                  "--out", str(newer)]) == 0
            newer_row, newer_sha = sr.load_completed(
                newer / "loop-continuation.json")
            reopened, reopened_sha = sr.load_completed(
                second / "loop-continuation.json")
            assert reopened == second_row and reopened_sha == second_sha
            # A target-local branch fork remains retained but cannot replace the
            # one forward shared-source pointer.
            (target_b_tree / "kernel.c").write_text("target-b divergent keep\n")
            run._git(target_b_tree, "add", "kernel.c")
            run._git(target_b_tree, "-c", "user.email=t@t", "-c", "user.name=t",
                     "commit", "-m", "target B divergent keep")
            divergent_commit = run._git(target_b_tree, "rev-parse", "HEAD")
            source_key = sr._source_owner_key(target_b)
            legacy_key = sr._digest({
                "worktree": str(Path(newer_row["worktree"]).resolve()),
                "branch": newer_row["branch"]})
            legacy_state = {"source_results": {legacy_key: {
                "path": str(newer / "loop-continuation.json"), "sha256": newer_sha}}}
            assert sr._source_result(legacy_state, target_b) == {
                "path": str(newer / "loop-continuation.json"), "sha256": newer_sha}
            assert legacy_state["source_results"][source_key]["sha256"] == newer_sha
            source_state = {"source_results": {source_key: {
                "path": str(newer / "loop-continuation.json"), "sha256": newer_sha}}}
            sr._remember_source_result(source_state, {
                "experimental_source_keeps": [{"retained": True}],
                "current_anchor": {"commit": divergent_commit}}, target_b,
                {"path": "/retained/target-b", "sha256": "f" * 64})
            assert source_state["source_results"][source_key]["sha256"] == newer_sha
            assert newer_row["current_anchor"]["commit"] != divergent_commit
            observed.append((first_row, second_row))
            return 0
        if validation_failure:
            retry = first.parent / "target-b-validation-retry"
            target_b_retry = list(argv)
            target_b_retry[target_b_retry.index("--target-id") + 1] = "target-b"
            target_b_retry[target_b_retry.index("--store") + 1] = str(first.parent / "store-b")
            target_b_retry += ["--resume-run", str(second / "loop-continuation.json"),
                "--source-anchor-continuation", str(second / "loop-continuation.json"),
                "--source-anchor-sha256", second_sha, "--validate-source-continuation",
                "--iterations", "1", "--out", str(retry)]
            with mock.patch.object(run.claim, "hold_cpu", validation_hold):
                assert original_main(target_b_retry) == 0
            retry_row, retry_sha = sr.load_completed(retry / "loop-continuation.json")
            retry_validation = run.surface_validation.reopen_reference(
                retry_row["source_validation"])
            assert retry_validation["disposition"] == "passed"
            assert retry_validation["original_anchor"] == validation["original_anchor"]
            second, second_row, second_sha = retry, retry_row, retry_sha

        # A completed validation verdict does not turn the target into a blocked
        # validation worker.  Its next ordinary child can continue source research.
        research = first.parent / "target-b-research"
        target_b_research = list(argv)
        target_b_research[target_b_research.index("--target-id") + 1] = "target-b"
        target_b_research[target_b_research.index("--store") + 1] = str(first.parent / "store-b")
        target_b_research += ["--resume-run", str(second / "loop-continuation.json"),
                              "--out", str(research)]
        assert original_main(target_b_research) == 0
        research_row, research_sha = sr.load_completed(research / "loop-continuation.json")
        assert research_row["current_anchor"] != second_row["current_anchor"]

        # A keeps its own request/history receipt while consuming B's newer exact
        # source/build.  Dry-run proves startup/rebinding without another proposal.
        third = first.parent / "target-a-resumed"
        target_a = [*argv, "--resume-run", str(first / "loop-continuation.json"),
                    "--source-anchor-continuation", str(research / "loop-continuation.json"),
                    "--source-anchor-sha256", research_sha,
                    "--validate-source-continuation", "--iterations", "1",
                    "--out", str(third), "--dry-run"]
        target_a[target_a.index("--anchor-build") + 1] = \
            first_row["current_anchor"]["path"]
        starts = []
        real_startup = run.champion.verify_startup

        def startup(**kwargs):
            starts.append(kwargs["anchor_build"])
            return real_startup(**kwargs)

        with mock.patch.object(run.champion, "verify_startup", startup):
            assert original_main(target_a) == 0
        assert starts == [Path(research_row["current_anchor"]["path"])]
        observed.append((first_row, second_row))
        return 0

    with mock.patch.object(run, "main", exercise):
        cpu_fixture.test_existing_main_cpu_five_iterations_preserves_canonical_champion(
            False, enrolled_pair=True,
            expected_claim_cycles=5 if cross_worktree and not validation_failure else 2)
    assert len(observed) == 1


def test_pending_validation_requires_ordinary_search_before_retry_for_each_target():
    for selected_id in ("only-target", "cpu", "gpu"):
        entry = {"latest_reference": {"path": f"/{selected_id}", "sha256": "a" * 64},
                 "disposition": "pending", "retry_after_search_count": 1,
                 "attempts": 1, "history_digest": "b" * 64}
        assert not sr._validation_retry_due(entry, 0)
        assert sr._validation_retry_due(entry, 1)
    assert not sr._validation_retry_due(
        dict(entry, disposition="failed", retry_after_search_count=0), 99)


def test_shared_source_routes_authoring_only_to_original_owner(tmp_path):
    source_reference = {"path": "/retained/source", "sha256": "a" * 64}
    source = {"selected_target": {"selected_id": "owner"}}

    def prepare(argv, _prior, _directory, **_kwargs):
        return list(argv), {"scope": "full"}

    with mock.patch.object(sr, "load_completed", return_value=(source, "a" * 64)), \
            mock.patch.object(cpu_screen, "prepare_batch", side_effect=prepare):
        owner = sr._batch_argv(
            ["--target-id", "owner"], None, 1, tmp_path / "owner",
            source_prior=source_reference)
        other = sr._batch_argv(
            ["--target-id", "other"], None, 1, tmp_path / "other",
            source_prior=source_reference)
        validation = sr._batch_argv(
            ["--target-id", "other"], None, 1, tmp_path / "validation",
            source_prior=source_reference, validate_source=True)
    assert sr.option(owner, "--source-anchor-continuation") == "/retained/source"
    assert sr.option(other, "--source-anchor-continuation") is None
    assert sr.option(validation, "--source-anchor-continuation") == "/retained/source"
    assert "--validate-source-continuation" in validation


def test_required_source_validation_uses_production_and_every_keep_author():
    targets = [["--target-id", value] for value in ("a", "b", "prod", "optional")]
    identities = {value: {"selected_id": value,
        "original_target": {"enrolled_as": ["production"] if value == "prod" else ["seed"]}}
        for value in ("a", "b", "prod", "optional")}
    source_reference = {"path": "/source", "sha256": "a" * 64}
    source = {"current_anchor": {"commit": "3" * 40},
              "source_lineage_keeps": [{"locator": "a"}, {"locator": "b"}]}
    receipts = [mock.Mock(selected_target=identities["a"], kept_commit="2" * 40),
                mock.Mock(selected_target=identities["b"], kept_commit="3" * 40)]
    rows = {}
    validations = {}
    for value, intended in (("a", False), ("b", True), ("prod", False)):
        reference = {"locator": value, "sha256": value.encode().hex().ljust(64, "0")}
        validations["subject-" + value] = {
            "latest_reference": reference, "disposition": "passed"}
        rows[value] = {"source_commit": "3" * 40, "target": identities[value],
                       "intended_target": intended, "disposition": "passed"}
    state = {"source_results": {}, "source_validations": validations,
             "last_results": {}}
    with mock.patch.object(sr, "_source_result", return_value=source_reference), \
            mock.patch.object(sr, "load_completed", return_value=(source, "a" * 64)), \
            mock.patch.object(sr, "_selected_identity",
                side_effect=lambda argv: identities[sr.option(argv, "--target-id")]), \
            mock.patch.object(sr, "_validation_subject",
                side_effect=lambda _state, argv, _index, _commit:
                    "subject-" + sr.option(argv, "--target-id")), \
            mock.patch.object(run.surface_fold, "reopen_reference", side_effect=receipts), \
            mock.patch.object(run.surface_validation, "reopen_reference",
                side_effect=lambda reference: rows[reference["locator"]]):
        aggregate = sr._required_source_validation(state, targets)
    assert aggregate["required_target_ids"] == ["a", "b", "prod"]
    assert aggregate["intended_target_id"] == "b"
    assert aggregate["disposition"] == "passed"
    assert aggregate["missing_target_ids"] == []
    state["required_source_validation"] = aggregate
    with mock.patch.object(sr, "_required_source_validation", return_value=None):
        sr._refresh_required_source_validation(state, targets)
    assert state["required_source_validation"] is None


def test_required_source_loo_routes_each_validated_target_once(tmp_path):
    targets = [["--target-id", "a"], ["--target-id", "b"]]
    state = {"required_source_validation": {"disposition": "passed", "rows": [
        {"selected_id": "a", "subject": "subject-a"},
        {"selected_id": "b", "subject": "subject-b"}]}, "source_loo": {},
        "source_search_counts": {}}
    assert sr._pending_source_loo(state, targets) == {0: "subject-a", 1: "subject-b"}
    references = []
    for name, disposition in (("loo", "neutral"), ("rebaseline", "passed")):
        path = tmp_path / f"{name}.json"
        raw = json.dumps({"operation": name, "disposition": disposition,
                          "deletion_authorized": False},
                         sort_keys=True).encode() + b"\n"
        path.write_bytes(raw)
        references.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                           "disposition": disposition})
    state["source_loo"]["subject-a"] = {"result": {
        "loo": [references[0]], "rebaseline": references[1]},
        "disposition": "passed", "attempts": 1, "retry_after_search_count": 0}
    assert sr._pending_source_loo(state, targets) == {1: "subject-b"}
    inconclusive_path = tmp_path / "inconclusive.json"
    inconclusive_raw = json.dumps({"operation": "loo", "disposition": "inconclusive",
                                   "deletion_authorized": False},
                                  sort_keys=True).encode() + b"\n"
    inconclusive_path.write_bytes(inconclusive_raw)
    state["source_loo"]["subject-b"] = {"result": {
        "loo": [{"path": str(inconclusive_path),
                 "sha256": hashlib.sha256(inconclusive_raw).hexdigest(),
                 "disposition": "inconclusive"}],
        "rebaseline": references[1]}, "disposition": "pending", "attempts": 1,
        "retry_after_search_count": 1}
    assert sr._pending_source_loo(state, targets) == {}
    state["source_search_counts"]["1"] = 1
    assert sr._pending_source_loo(state, targets) == {1: "subject-b"}
