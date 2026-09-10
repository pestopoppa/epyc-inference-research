"""Factual CPU telemetry on the existing serving path; no compute or real child."""
from __future__ import annotations

import json
import threading
from unittest import mock

import pytest

from . import residency, serving
from .test_cpu_serving_recipe import _launch
from .test_resolved_recipe import BUILD, _resolve
from .test_serving_residency import _proof, _sampler_class


def _proc(root, pid, *, ticks=100, cpus="0-95", mems="0-1"):
    root.mkdir(parents=True, exist_ok=True)
    # Kernel stat fields 3..22: state + eighteen intermediate fields + starttime.
    (root / "stat").write_text(f"{pid} (fixture worker) S " + "0 " * 18 + f"{ticks} 0\n")
    (root / "status").write_text(
        f"Name:\tfixture\nCpus_allowed_list:\t{cpus}\nMems_allowed_list:\t{mems}\n")


def _fixture(tmp_path, **budgets):
    _proc(tmp_path / "4321", 4321)
    _proc(tmp_path / "4321/task/4321", 4321)
    _proc(tmp_path / "4321/task/4322", 4322, ticks=101)
    boot = tmp_path / "sys/kernel/random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text("synthetic-boot-id\n")
    sampler = residency.CpuLifecycleSampler(proc_root=tmp_path, **budgets)
    sampler.attach_target(4321)
    return sampler


def test_raw_thread_change_during_request_and_numa_permissions_are_preserved(tmp_path):
    sampler = _fixture(tmp_path)
    sampler.phase("load")
    sampler._sample()
    sampler.phase("warmup")
    _proc(tmp_path / "4321/task/4322", 4322, ticks=101, cpus="96-97", mems="2")
    sampler._sample()
    sampler.phase("measurement")
    _proc(tmp_path / "4321/task/4322", 4322, ticks=101)
    sampler._sample()
    sampler.checkpoint("measurement_end")
    sampler.phase("teardown")
    sampler._sample()
    sampler.finish()
    result = sampler.observation
    assert result["target"] == {"pid": 4321, "start_ticks": 100}
    assert result["status"] == "observed"
    assert result["scope"] == "process_and_task_allowed_lists_not_numa_page_placement"
    assert [row["phase"] for row in result["samples"]] == ["load", "warmup", "measurement", "teardown"]
    assert [row["tasks"][1]["Cpus_allowed_list"] for row in result["samples"]] == ["0-95", "96-97", "0-95", "0-95"]
    assert result["samples"][1]["tasks"][1]["Mems_allowed_list"] == "2"
    assert result["cpu_placement"] == result["contention"] == "unproven"
    assert result["shutdown_resolved"] and not result["truncated"]
    result["samples"].clear()
    assert len(sampler.observation["samples"]) == 4  # detached archival snapshot


def test_process_reuse_never_rebinds_and_task_reuse_keeps_original_start_ticks(tmp_path):
    sampler = _fixture(tmp_path)
    sampler._sample()
    _proc(tmp_path / "4321/task/4322", 4322, ticks=999)
    sampler._sample()
    assert [row["tasks"][1]["start_ticks"] for row in sampler.observation["samples"]] == [101, 999]
    _proc(tmp_path / "4321", 4321, ticks=200)
    sampler._sample()
    row = sampler.observation["samples"][-1]
    assert "attached process identity changed" in row["error"]
    assert row["process"] is None and row["tasks"] == []
    assert sampler.observation["target"]["start_ticks"] == 100


@pytest.mark.parametrize("failure", ["missing", "malformed", "oversized", "reuse"])
def test_failed_task_read_keeps_partial_facts_and_error(tmp_path, failure):
    sampler = _fixture(tmp_path)
    path = tmp_path / "4321/task/4322/status"
    if failure == "missing":
        path.unlink()
    elif failure == "malformed":
        path.write_text("Cpus_allowed_list: 9-0\nMems_allowed_list: 0\n")
    elif failure == "oversized":
        path.write_text("x" * 16385)
    else:
        original = sampler._text

        def read(target):
            result = original(target)
            if target == path:
                _proc(path.parent, 4322, ticks=999)
            return result

        sampler._text = read
    sampler._sample()
    result = sampler.observation
    assert result["status"] == "partial"
    assert result["samples"][0]["error"]
    assert result["samples"][0]["tasks"][0]["id"] == 4321
    assert result["cpu_placement"] == "unproven"


@pytest.mark.parametrize("budget", [{"max_samples": 1}, {"max_bytes": 1}, {"max_tasks": 1}])
def test_bounds_are_explicit_not_silent_complete_coverage(tmp_path, budget):
    sampler = _fixture(tmp_path, **budget)
    sampler._sample()
    sampler._sample()
    sampler.finish()
    result = sampler.observation
    if "max_tasks" in budget:
        assert "task count bound exceeded" in result["samples"][0]["error"]
    else:
        assert result["truncated"]
    assert result["cpu_placement"] == "unproven"
    if "max_bytes" in budget:
        assert result["status"] == "unavailable" and not result["samples"]


def test_phase_crossing_and_final_read_overrun_remain_diagnostic(tmp_path):
    sampler = _fixture(tmp_path, max_sample_s=0.1)
    clock = [10.0]
    original = sampler._task_ids

    def ids(root):
        result = original(root)
        if sampler._phase == "measurement":
            sampler.checkpoint("measurement_end")
        else:
            clock[0] += 0.2  # final census read, after the last per-task deadline check
        return result

    sampler._task_ids = ids
    with mock.patch.object(residency.time, "monotonic", side_effect=lambda: clock[0]):
        sampler.phase("measurement")
        sampler._sample()
    row = sampler.observation["samples"][0]
    assert row["phase"] == "measurement" and row["phase_at_end"] == "measurement_end"
    assert row["crosses_phase_boundary"]
    assert "sample duration bound exceeded" in row["error"]
    sampler.finish()
    sampler._sample()
    assert len(sampler.observation["samples"]) == 1  # no after-close substitute


def test_real_observer_thread_over_synthetic_proc_stops_and_retains_gap(tmp_path):
    sampler = _fixture(tmp_path, interval=0.01)
    captured = threading.Event()
    original = sampler._sample

    def sample():
        original()
        captured.set()

    sampler._sample = sample
    sampler.start()
    try:
        assert captured.wait(1)
    finally:
        sampler.finish()
    result = sampler.observation
    assert result["boot_id"] == "synthetic-boot-id"
    assert result["shutdown_resolved"] and result["samples"]
    assert result["markers"][0]["phase"] == "setup"
    assert result["markers"][-1]["phase"] == "closed"
    assert result["samples"][0]["gap_before_s"] is None


def test_gap_is_preserved_and_error_only_rows_are_not_observed(tmp_path):
    sampler = _fixture(tmp_path)
    with mock.patch.object(residency.time, "monotonic", return_value=10.0):
        sampler._sample()
    with mock.patch.object(residency.time, "monotonic", return_value=15.0):
        sampler._sample()
    assert sampler.observation["samples"][1]["gap_before_s"] == 5.0
    sampler._samples.clear()
    _proc(tmp_path / "4321", 4321, ticks=999)
    sampler._sample()
    assert sampler.observation["status"] == "unavailable"
    assert all(row["error"] for row in sampler.observation["samples"])
    sampler.finish()


def test_actual_measure_path_invalidates_observed_outside_affinity_after_owned_teardown(tmp_path):
    sampler = _fixture(tmp_path)
    # The owning serving method calls attach_target itself after its mocked Popen.
    sampler._target = None
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=1, cpu_list="0-95")
    resolved = _resolve(recipe, backend="cpu")
    reads = []

    class Response:
        def read(self):
            reads.append(sampler._phase)
            _proc(tmp_path / "4321/task/4322", 4322, ticks=101,
                  cpus="96-97")
            sampler._sample()  # synthetic observation boundary inside the actual request
            return json.dumps({"timings": {"predicted_n": recipe.n_predict,
                                          "predicted_per_second": 25.0}}).encode()

    process = mock.Mock(pid=4321)
    process.poll.return_value = None
    evidence = []
    with mock.patch.object(serving.residency, "CpuLifecycleSampler", return_value=sampler), \
            mock.patch.object(sampler, "start", side_effect=sampler.phase), \
            mock.patch.object(serving.subprocess, "Popen", return_value=process), \
            mock.patch.object(serving.urllib.request, "urlopen", return_value=Response()), \
            mock.patch.object(serving.residency, "Sampler", _sampler_class(_proof())), \
            mock.patch.object(serving, "verify_env_readback"):
        with pytest.raises(serving.MeasurementInvalid) as failed:
            serving._measure_once(recipe, BUILD, 18311, evidence=evidence, resolved_recipe=resolved)
    assert reads == ["warmup", "measurement"]
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(30)
    assert failed.value.record["resolved_recipe"] == resolved.to_dict()
    assert failed.value.record["recipe"] == recipe.to_dict()
    assert failed.value.record["teardown"] == "terminated"
    assert failed.value.record["observed_rate_not_admissible_tok_s"] == 25.0
    assert failed.value.record["residency"] == evidence[0]
    facts = evidence[0]["cpu_lifecycle"]
    assert [row["tasks"][1]["Cpus_allowed_list"] for row in facts["samples"]] == ["96-97", "96-97"]
    assert failed.value.record["failed_conditions"][0]["observations"] == 2
    assert [row["phase"] for row in facts["markers"]] == ["setup", "load", "placement", "health",
                                                         "warmup", "measurement", "measurement_end",
                                                         "teardown", "closed"]
    assert evidence[0]["cpu_placement"] == evidence[0]["contention"] == "unproven"
    assert serving._residency_fold(evidence)["cpu_placement"] == "unproven"


def test_diagnostic_failure_does_not_change_success_or_teardown_and_gpu_unchanged():
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=1)
    evidence = []
    with mock.patch.object(serving.residency, "CpuLifecycleSampler", side_effect=OSError("fixture")):
        value, seen, _ = _launch(recipe, _resolve(recipe, backend="cpu"), _proof(), evidence)
    assert value == 25.0 and seen["terminated"]
    assert evidence[0]["cpu_lifecycle_error"] == "OSError: fixture"
    gpu = serving.Recipe(name="gpu", model="/m", np=1)
    with mock.patch.object(serving.residency, "CpuLifecycleSampler", side_effect=AssertionError("GPU")) as ctor:
        _launch(gpu, _resolve(gpu), _proof(), [])
    ctor.assert_not_called()


def test_hook_and_snapshot_failures_cannot_replace_serving_result():
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=1)
    observer = mock.Mock()
    observer.start.side_effect = OSError("start diagnostic")
    observer.phase.side_effect = OSError("phase diagnostic")
    observer.attach_target.side_effect = OSError("attach diagnostic")
    observer.note_hook_failure.side_effect = OSError("error diagnostic")
    type(observer).observation = mock.PropertyMock(side_effect=OSError("snapshot diagnostic"))
    evidence = []
    with mock.patch.object(serving.residency, "CpuLifecycleSampler", return_value=observer):
        value, seen, _ = _launch(recipe, _resolve(recipe, backend="cpu"), _proof(), evidence)
    assert value == 25.0 and seen["terminated"]
    assert evidence[0]["cpu_lifecycle_error"] == "OSError: snapshot diagnostic"
    observer.finish.assert_called()
