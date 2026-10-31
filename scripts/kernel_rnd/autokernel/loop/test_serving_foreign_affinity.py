"""INF-70 sibling overlap and live tick deltas, entirely synthetic proc/sysfs."""
import time
from unittest import mock

from . import residency


def proc(root, pid, cpus, ticks=0, start=100):
    path = root / str(pid)
    path.mkdir(exist_ok=True, parents=True)
    fields = ["0"] * 50
    fields[0] = "S"
    fields[11] = str(ticks)
    fields[19] = str(start)
    (path / "stat").write_text(f"{pid} (worker with spaces) " + " ".join(fields))
    (path / "status").write_text(f"Cpus_allowed_list:\t{cpus}\nMems_allowed_list:\t0\n")
    if not (path / "exe").is_symlink():
        (path / "exe").symlink_to("/synthetic/bin/cc1plus")
    return path


def topology(root, count=96):
    for cpu in range(count * 2):
        path = root / f"cpu{cpu}"
        (path / "topology").mkdir(parents=True)
        (path / "node0").mkdir()
        (path / "topology/thread_siblings_list").write_text(
            f"{cpu % count},{cpu % count + count}\n")


def census(sampler, target_cpus="0-95"):
    return sampler._non_target_activity({"pid": 10}, time.monotonic() + 10,
                                       [{"Cpus_allowed_list": target_cpus}])


def test_full_96_core_smt_map_detects_184_191_foreign_burst(tmp_path):
    sysfs, procfs = tmp_path / "sys", tmp_path / "proc"
    topology(sysfs)
    proc(procfs, 10, "0-95")
    proc(procfs, 20, "184-191", ticks=100)
    sampler = residency.CpuLifecycleSampler(proc_root=procfs, sysfs_cpu_root=sysfs)
    first = census(sampler)
    assert first["target_physical_affinity_footprint"] == list(range(192))
    assert not first["active_intervals"]  # one reading proves no CPU delta
    proc(procfs, 20, "184-191", ticks=250)
    second = census(sampler)
    interval, = second["active_intervals"]
    assert interval["cpu_tick_delta"] == 150
    assert interval["potential_physical_overlap"] == list(range(184, 192))
    assert interval["after"]["exe"] == "/synthetic/bin/cc1plus"
    assert second["classification"] == "unproven"  # no new inference/load grading rule
    assert not second["incomplete"]
    assert sampler.observation["topology"]["physical_sibling_groups"] == [
        [cpu, cpu + 96] for cpu in range(96)]


def test_disjoint_cores_and_idle_counters_are_not_reported_as_overlap(tmp_path):
    sysfs, procfs = tmp_path / "sys", tmp_path / "proc"
    topology(sysfs)
    proc(procfs, 20, "184-191", ticks=100)
    sampler = residency.CpuLifecycleSampler(proc_root=procfs, sysfs_cpu_root=sysfs)
    census(sampler, "0-7")
    assert not census(sampler, "0-7")["active_intervals"]
    proc(procfs, 20, "184-191", ticks=105)
    interval, = census(sampler, "0-7")["active_intervals"]
    assert interval["potential_physical_overlap"] == []


def test_missing_topology_is_unknown_never_disjoint(tmp_path):
    procfs = tmp_path / "proc"
    proc(procfs, 20, "184-191", ticks=100)
    sampler = residency.CpuLifecycleSampler(proc_root=procfs, sysfs_cpu_root=tmp_path / "missing")
    census(sampler)
    proc(procfs, 20, "184-191", ticks=120)
    row = census(sampler)
    assert row["incomplete"] and row["errors"]
    assert row["target_physical_affinity_footprint"] is None
    assert row["active_intervals"][0]["potential_physical_overlap"] is None


def test_reused_pid_cannot_create_a_cpu_delta(tmp_path):
    sysfs, procfs = tmp_path / "sys", tmp_path / "proc"
    topology(sysfs)
    proc(procfs, 20, "184-191", ticks=100)
    sampler = residency.CpuLifecycleSampler(proc_root=procfs, sysfs_cpu_root=sysfs)
    census(sampler)
    proc(procfs, 20, "184-191", ticks=500, start=200)
    row = census(sampler)
    assert row["incomplete"]
    assert not row["active_intervals"]
    assert any("reused" in error for error in row["errors"])


def test_serving_gpu_pid_hook_retains_affinity_observation_and_closes_collector():
    observer = mock.Mock()
    observer.observation = {"samples": [{"non_target_activity": {"classification": "unproven"}}]}
    sampler = residency.Sampler()
    with mock.patch.object(residency, "CpuLifecycleSampler", return_value=observer):
        sampler.watch_pid(123)
    observer.attach_target.assert_called_once_with(123)
    observer.start.assert_not_called()  # no background collector outside a sampling window
    assert sampler.proof["cpu_affinity_observation"] == observer.observation
    sampler.__exit__()
    observer.finish.assert_called_once()


def test_failed_pid_capture_remains_explicit_and_does_not_fail_gpu_sampler():
    sampler = residency.Sampler()
    with mock.patch.object(residency, "CpuLifecycleSampler", side_effect=OSError("unreadable")):
        sampler.watch_pid(123)
    assert "OSError: unreadable" in sampler.proof["cpu_affinity_observation_error"]
    assert "cpu_affinity_observation" not in sampler.proof


def test_collector_snapshot_and_shutdown_errors_do_not_replace_serving_result():
    observer = mock.Mock()
    observer.finish.side_effect = OSError("finish failed")
    type(observer).observation = mock.PropertyMock(side_effect=OSError("snapshot failed"))
    sampler = residency.Sampler()
    with mock.patch.object(residency, "CpuLifecycleSampler", return_value=observer):
        sampler.watch_pid(123)
    assert "snapshot failed" in sampler.proof["cpu_affinity_observation_error"]
    sampler.__exit__()
    observer.finish.assert_called_once()
    assert "snapshot failed" in sampler.proof["cpu_affinity_observation_error"]


def test_gpu_sampling_context_starts_and_closes_attached_cpu_collector():
    observer = mock.Mock()
    with mock.patch.object(residency, "CpuLifecycleSampler", return_value=observer):
        with residency.Sampler() as sampler:
            sampler.watch_pid(123)
            observer.start.assert_called_once_with("load")
    observer.finish.assert_called_once()


def test_real_serving_export_retains_smt_overlap_without_new_gpu_veto(tmp_path):
    import json
    from . import serving
    from .test_resolved_recipe import BUILD, _resolve

    sysfs, procfs = tmp_path / "sys", tmp_path / "proc"
    topology(sysfs)
    target = proc(procfs, 4321, "0-95")
    proc(target / "task", 4321, "0-95")
    proc(procfs, 20, "184-191", ticks=100)
    observer = residency.CpuLifecycleSampler(proc_root=procfs, sysfs_cpu_root=sysfs)
    recipe = serving.Recipe(name="gpu", model="/m", np=1)
    process = mock.Mock(pid=4321)
    process.poll.return_value = None
    reads = 0

    class Response:
        def read(self):
            nonlocal reads
            reads += 1
            proc(procfs, 20, "184-191", ticks=100 + 20 * reads)
            observer._sample()  # synthetic proc facts inside real warmup/request boundaries
            return json.dumps({"timings": {"predicted_n": recipe.n_predict,
                                          "predicted_per_second": 25.0}}).encode()

    evidence = []
    with mock.patch.object(serving, "require_free_port"), \
            mock.patch.object(serving, "ServerPortOwner"), \
            mock.patch.object(serving.subprocess, "Popen", return_value=process), \
            mock.patch.object(serving.urllib.request, "urlopen", return_value=Response()), \
            mock.patch.object(serving, "verify_env_readback"), \
            mock.patch.object(residency.Sampler, "_loop"), \
            mock.patch.object(residency, "CpuLifecycleSampler", return_value=observer), \
            mock.patch.object(observer, "start", side_effect=observer.phase):
        rate = serving._measure_once(recipe, BUILD, 18311, evidence=evidence,
                                     resolved_recipe=_resolve(recipe))
    assert rate == 25.0
    facts = evidence[0]["cpu_affinity_observation"]
    rows = [interval for sample in facts["samples"]
            for interval in sample["non_target_activity"]["active_intervals"]]
    assert any(row["potential_physical_overlap"] == list(range(184, 192)) for row in rows)
    assert facts["shutdown_resolved"]
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(180)


def test_missing_target_task_census_cannot_claim_disjoint_footprint(tmp_path):
    sysfs, procfs = tmp_path / "sys", tmp_path / "proc"
    topology(sysfs)
    proc(procfs, 20, "184-191", ticks=100)
    sampler = residency.CpuLifecycleSampler(proc_root=procfs, sysfs_cpu_root=sysfs)
    row = sampler._non_target_activity({"pid": 10}, time.monotonic() + 10)
    assert row["incomplete"]
    assert row["target_physical_affinity_footprint"] is None
