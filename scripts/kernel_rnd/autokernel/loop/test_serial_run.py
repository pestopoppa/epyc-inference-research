"""Owned tiny children and original loop fixtures; no hardware or providers."""
import hashlib
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from . import run, scheduling, serial_run as sr, serial_scheduling as ss
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
if selection_path:
    from scripts.kernel_rnd.autokernel.loop import scheduling, serial_scheduling as ss
    from scripts.kernel_rnd.autokernel.loop.measurement_capture import ArtifactStore
    selection = scheduling.Selection.from_dict(json.loads(Path(selection_path).read_text()))
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
    if mode == "fail_held":
        (out / "loop-held-claims.json").write_text(json.dumps(held))
        sys.exit(3)
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
    outcomes=[SimpleNamespace(status="measured_null") for _ in range(count)],
    **({"runtime_recipe_reference": {"locator": "runtime-selection-fixture",
       "sha256": "b" * 64, "verified": True}}
       if mode == "runtime_ref" and cpu else {}),
    **({"held_claim_evidence": held} if held else {}))
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


def test_serialized_targets_share_latest_source_anchor_without_losing_own_resume(
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
    assert sr.option(seen[1], "--source-anchor-continuation").endswith(
        "batch-000000/loop-continuation.json")
    assert sr.option(seen[2], "--resume-run").endswith(
        "batch-000000/loop-continuation.json")
    assert sr.option(seen[2], "--source-anchor-continuation").endswith(
        "batch-000001/loop-continuation.json")
    source_path = Path(sr.option(seen[2], "--source-anchor-continuation"))
    assert sr.option(seen[2], "--source-anchor-sha256") == hashlib.sha256(
        source_path.read_bytes()).hexdigest()


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


def test_actual_cpu_keep_cross_target_then_older_history_uses_latest_source():
    original_main = run.main
    observed = []

    def exercise(argv):
        assert original_main(argv) == 0
        first = Path(sr.option(argv, "--out"))
        first_row, first_sha = sr.load_completed(first / "loop-continuation.json")
        assert first_row["selected_target"]["selected_id"] == "target-a"

        second = first.parent / "target-b"
        target_b = list(argv)
        target_b[target_b.index("--target-id") + 1] = "target-b"
        target_b[target_b.index("--store") + 1] = str(first.parent / "store-b")
        target_b += ["--source-anchor-continuation", str(first / "loop-continuation.json"),
                     "--source-anchor-sha256", first_sha, "--out", str(second)]
        assert original_main(target_b) == 0
        second_row, second_sha = sr.load_completed(second / "loop-continuation.json")
        assert second_row["selected_target"]["selected_id"] == "target-b"
        assert second_row["current_anchor"] != first_row["current_anchor"]

        # A keeps its own request/history receipt while consuming B's newer exact
        # source/build.  Dry-run proves startup/rebinding without another proposal.
        third = first.parent / "target-a-resumed"
        target_a = [*argv, "--resume-run", str(first / "loop-continuation.json"),
                    "--source-anchor-continuation", str(second / "loop-continuation.json"),
                    "--source-anchor-sha256", second_sha, "--out", str(third), "--dry-run"]
        starts = []
        real_startup = run.champion.verify_startup

        def startup(**kwargs):
            starts.append(kwargs["anchor_build"])
            return real_startup(**kwargs)

        with mock.patch.object(run.champion, "verify_startup", startup):
            assert original_main(target_a) == 0
        assert starts == [Path(second_row["current_anchor"]["path"])]
        observed.append((first_row, second_row))
        return 0

    with mock.patch.object(run, "main", exercise):
        cpu_fixture.test_existing_main_cpu_five_iterations_preserves_canonical_champion(
            False, enrolled_pair=True)
    assert len(observed) == 1
