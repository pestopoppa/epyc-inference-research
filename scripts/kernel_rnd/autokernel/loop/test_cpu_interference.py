"""Ordinary load stays noise; original CPU placement contradictions stay invalid."""
from unittest import mock
import json
import sqlite3

import pytest

from . import archive, loop, residency, serving
from .test_cpu_lifecycle_facts import _fixture, _proc
from .test_loop import _Critic, _Planner, _hypothesis, drive_single_lane
from .test_resolved_recipe import BUILD, _resolve


def _host(root):
    (root / "stat").write_text("cpu 900 10 200 300 400 5 6 7 8 9\n" + "cpu0 0\n" * 3000)
    (root / "meminfo").write_text("MemAvailable: 1 kB\nSwapFree: 0 kB\nSwapTotal: 900 kB\n")
    (root / "vmstat").write_text("pswpin 123456\npswpout 987654\n")
    (root / "pressure").mkdir(exist_ok=True)
    (root / "pressure/memory").write_text(
        "some avg10=99.0 avg60=90.0 avg300=80.0 total=999999\n"
        "full avg10=89.0 avg60=80.0 avg300=70.0 total=888888\n")


def _foreign(root, *, pid=99, ticks=1, start=10, comm="llama-server", cpus="0-95"):
    target = root / str(pid)
    _proc(target, pid, cpus=cpus)
    fields = ["0"] * 37
    fields[0], fields[11], fields[19], fields[36] = "S", str(ticks), str(start), "0"
    (target / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")


def test_host_noise_covers_unattached_setup_and_during_request_not_a_blocker(tmp_path):
    sampler = _fixture(tmp_path)
    _host(tmp_path)
    _foreign(tmp_path)
    original = sampler._target
    sampler._target = None
    sampler._sample()
    setup = sampler.observation["samples"][0]
    assert setup["host"]["memory_psi"]["some"]["avg10"] == 99.0
    assert setup["host"]["cpu_ticks"]["guest"] == 8
    sampler._target = original
    sampler.phase("measurement")
    _foreign(tmp_path, ticks=123, comm="build-and-llama-server")
    sampler._sample()
    row = sampler.observation["samples"][1]
    activity = row["non_target_activity"]
    assert activity["classification"] == "unproven"
    assert activity["active_intervals"][0]["cpu_tick_delta"] == 122
    assert activity["active_intervals"][0]["before"]["start_ticks"] == 10
    assert row["host"]["swap_pages"]["pswpout"] == 987654
    assert residency.cpu_lifecycle_invalidity(sampler.observation, "0-95") == []
    assert sampler.observation["foreign_inference_classification"] == "not_available"


@pytest.mark.parametrize("change", ["reuse", "regression", "missing"])
def test_non_target_gaps_do_not_imply_absence_or_competing_inference(tmp_path, change):
    sampler = _fixture(tmp_path)
    _host(tmp_path)
    _foreign(tmp_path, ticks=100)
    sampler._sample()
    _foreign(tmp_path, ticks=1 if change == "regression" else 200,
             start=20 if change == "reuse" else 10)
    if change == "missing":
        (tmp_path / "99/status").unlink()
    sampler.phase("measurement")
    sampler._sample()
    activity = sampler.observation["samples"][-1]["non_target_activity"]
    assert activity["incomplete"] and activity["errors"]
    assert not activity["active_intervals"]
    assert residency.cpu_lifecycle_invalidity(sampler.observation, "0-95") == []


def test_census_is_count_and_time_bounded_with_unknown_coverage(tmp_path):
    sampler = _fixture(tmp_path, max_processes=1)
    _foreign(tmp_path)
    _foreign(tmp_path, pid=98)
    sampler._sample()
    row = sampler.observation["samples"][-1]["non_target_activity"]
    assert row["processes_read"] == 1 and row["incomplete"]
    with mock.patch.object(residency.time, "monotonic", return_value=20.0):
        row = sampler._non_target_activity(sampler._target, 10.0)
    assert row["processes_read"] == 0 and row["incomplete"]


def test_only_post_ready_original_contradictions_invalidate(tmp_path):
    sampler = _fixture(tmp_path)
    _proc(tmp_path / "4321/task/4322", 4322, ticks=101, cpus="96-97")
    sampler.phase("placement")
    sampler._sample()
    assert residency.cpu_lifecycle_invalidity(sampler.observation, "0-95") == []
    sampler.phase("measurement")
    sampler._sample()
    assert residency.cpu_lifecycle_invalidity(sampler.observation, "0-95") == []
    sampler._sample()
    failures = residency.cpu_lifecycle_invalidity(sampler.observation, "0-95")
    assert failures[0]["condition"] == "task_affinity_outside_original_recipe"
    assert failures[0]["observations"] == 2
    _proc(tmp_path / "4321", 4321, ticks=999)
    sampler._sample()
    assert residency.cpu_lifecycle_invalidity(sampler.observation, "0-95")[-1]["condition"] == "original_target_identity_changed"
    detached = sampler.observation
    for row in detached["samples"]:
        row["crosses_phase_boundary"] = True
    assert residency.cpu_lifecycle_invalidity(detached, "0-95") == []


def test_invalid_candidate_preserves_exact_arm_and_does_not_replace_a_sample():
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=1, cpu_list="0-95")
    resolved = _resolve(recipe, backend="cpu")
    requests = (("original", b'{"prompt":"literal whitespace", "n_predict":8}'),)
    invalid = loop.MeasurementInvalid("original placement mismatch", {"original": "raw"})
    with mock.patch.object(serving, "_measure_once", side_effect=[25.0, invalid]) as measure:
        with pytest.raises(loop.MeasurementInvalid) as failed:
            serving.compare(recipe, BUILD, BUILD, pairs=5, floor_pct=None,
                anchor_resolved_recipe=resolved, candidate_resolved_recipe=resolved,
                frozen_requests=requests)
    assert measure.call_count == 2  # never silently substitute a fresh candidate sample
    assert all(call.kwargs["frozen_requests"] == requests for call in measure.call_args_list)
    row = failed.value.record
    assert row["failed_arm"] == "candidate" and row["pair_index"] == 0
    assert row["anchor_raw_samples"] == [25.0] and row["candidate_raw_samples"] == []
    assert row["candidate_resolved_recipe"] == resolved.to_dict()
    assert "effect" not in row and "decisive" not in row


def test_existing_pool_and_archive_keep_invalid_distinct_from_null(tmp_path):
    captured = []
    original = {"schema": "epyc.autokernel.serving_invalid_comparison.v1",
                "status": "measurement_invalid", "invalid_arm": {"identity": "original"}}
    def invalid(*args):
        raise loop.MeasurementInvalid("original placement mismatch", original)
    def record(outcome):
        captured.append(outcome.to_attempt())
        archive.record(tmp_path, outcome.to_attempt(), epoch="fixture",
                       recorded_at="2026-09-10T00:00:00Z", campaign_id="fixture")
    commit = mock.Mock(side_effect=AssertionError("invalid is never a keep"))
    result = drive_single_lane(planner=_Planner(), critic=_Critic([], []),
        measure=invalid, gate=lambda *args: (True, []), commit=commit,
        iterations=1, record=record)
    outcome = result[0]
    assert outcome.status == "measurement_invalid" and outcome.hypothesis == _hypothesis()
    assert outcome.comparison is None and outcome.invalid_measurement == original
    assert "effect_fraction" not in captured[0] and "comparison" not in captured[0]
    retained = archive.recall(tmp_path, epoch="fixture")
    assert retained[0]["status"] == "measurement_invalid"
    with sqlite3.connect(tmp_path / "experiments.db") as db:
        payload = json.loads(db.execute("SELECT payload FROM experiments").fetchone()[0])
    assert payload["invalid_measurement"] == original
    commit.assert_not_called()


@pytest.mark.parametrize("runtime_only", [False, True])
def test_actual_main_reschedules_original_source_or_runtime_once(runtime_only):
    from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion
    test_existing_main_cpu_five_iterations_preserves_canonical_champion(
        False, runtime_only=runtime_only, invalid_once=True)


@pytest.mark.parametrize("mode", ["budget", "stop", "stop_after_archive", "twice", "cleanup"])
def test_existing_tail_reschedule_limits_keep_original_invalid_record(tmp_path, mode):
    from . import pipeline, run
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=1, cpu_list="0-95")
    resolved = _resolve(recipe, backend="cpu")
    calls, archived, resets, gates_seen = [], [], [], []
    stopped = [False]
    original_invalid = []
    def sample(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return 25.0
        if mode == "cleanup":
            raise RuntimeError("owned server teardown unresolved")
        if len(calls) == 2 or mode == "twice":
            if mode == "stop":
                stopped[0] = True
            raise loop.MeasurementInvalid("original placement mismatch", {"raw": len(calls)})
        assert archived[0]["status"] == "measurement_invalid"
        return 25.0
    def measure(*args):
        return run._serving_comparison(lambda: serving.compare(recipe, BUILD, BUILD,
            pairs=1, floor_pct=None, anchor_resolved_recipe=resolved,
            candidate_resolved_recipe=resolved), "fixture")
    def record(outcome):
        archived.append(json.loads(json.dumps(outcome.to_attempt())))
        if outcome.status == "measurement_invalid":
            original_invalid.append(json.loads(json.dumps(outcome.invalid_measurement)))
        if mode == "stop_after_archive":
            stopped[0] = True
    def gate(*args):
        gates_seen.append(True)
        return True, []
    worker = pipeline.Worker("lane", tmp_path / "source", BUILD)
    with mock.patch.object(serving, "_measure_once", side_effect=sample):
        outcomes = pipeline.run_pool(workers=[worker], make_planner=lambda w: _Planner(),
            make_critic=lambda w: _Critic([], []), build_context=dict,
            make_gate=lambda w: gate, make_measure=lambda w: measure,
            commit=mock.Mock(side_effect=AssertionError("no keep")), champion_head=lambda: "original",
            reset_to_champion=lambda w: resets.append(True) or "original", record=record,
            iterations=1 if mode in {"budget", "cleanup"} else 2,
            should_stop=lambda: stopped[0])
    assert len(resets) == len(gates_seen) == 1
    assert len(calls) == (3 if mode == "twice" else 2)
    assert [outcome.to_attempt()["status"] for outcome in outcomes] == [row["status"] for row in archived]
    if mode == "cleanup":
        assert outcomes[0].status == "lane_error" and not original_invalid
    else:
        assert outcomes[0].status == "measurement_invalid" and outcomes[0].comparison is None
        assert original_invalid[0]["anchor_raw_samples"] == [25.0]
        assert original_invalid[0]["candidate_raw_samples"] == []
        if mode == "stop_after_archive":
            assert outcomes[-1].status == "stopped_before_reschedule"
        elif mode == "twice":
            assert len(outcomes) == 2 and outcomes[-1].status == "measurement_invalid"
