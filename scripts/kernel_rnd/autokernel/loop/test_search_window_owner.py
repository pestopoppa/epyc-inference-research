"""Bounded raw observation tests; fixture facts never stand in for search verdicts."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import threading
import time
import copy
import queue
from types import SimpleNamespace

import pytest

from ..execution import microbench
from . import driver_execution as de
from . import planned_serving as ps
from . import search_window as sw
from . import observation_binding as ob
from . import measurement_capture as mc
from . import unified_worker as uw
from .test_unified_worker import _prepared, _start


def configuration(tmp_path, cpus=(0,)):
    for name in ("proc", "sys", "claims", "storage"):
        (tmp_path / name).mkdir(exist_ok=True)
    for cpu in cpus:
        root = tmp_path / "sys" / f"cpu{cpu}"
        (root / "cpufreq").mkdir(parents=True, exist_ok=True)
        (root / "topology").mkdir(exist_ok=True)
        for name, value in (("scaling_cur_freq", "3000000"), ("cpuinfo_min_freq", "1000000"),
                            ("cpuinfo_max_freq", "4000000")):
            (root / "cpufreq" / name).write_text(value)
        (root / "topology" / "physical_package_id").write_text("0")
    (tmp_path / "proc" / "loadavg").write_text("0 0 0 1/1 1")
    (tmp_path / "proc" / "uptime").write_text("10000 9000")
    (tmp_path / "proc" / "locks").write_text("")
    (tmp_path / "claims" / "cpu_region.GLOBAL.fixture.lock").write_text("")
    return sw.InstalledSearchWindowConfiguration(str(tmp_path / "claims"),
        str(tmp_path / "proc"), str(tmp_path / "sys"), str(tmp_path / "storage"),
        ("fixture",), microbench.HostStatePolicy(nominal_khz=3000000), 1,
        sw.source_digest())


def test_procfs_size_zero_is_read_to_bounded_eof():
    path = Path("/proc/self/stat")
    assert path.stat().st_size == 0
    row = sw.read_raw_file(path, 65536)
    assert row["error"] is None
    assert sw._raw(row).startswith(f"{os.getpid()} (".encode())


def test_fifo_symlink_and_oversize_sources_refuse_without_blocking(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    started = time.monotonic()
    assert "regular" in sw.read_raw_file(fifo, 16)["error"]
    assert time.monotonic() - started < 0.5
    original = tmp_path / "original"
    original.write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(original)
    assert sw.read_raw_file(link, 16)["error"] is not None
    assert "bound" in sw.read_raw_file(original, 2)["error"]


@pytest.mark.parametrize("cpus", [48, 96])
def test_representative_cpu_snapshot_does_not_pay_queue_sleep_per_read(tmp_path, monkeypatch, cpus):
    ids = tuple(range(cpus))
    config = configuration(tmp_path, ids)
    clock = [1000.0]
    read = sw.read_raw_file
    def timed_read(path, maximum):
        row = read(path, maximum)
        clock[0] += 0.0002  # labelled synthetic 0.2ms filesystem-read latency
        return row
    monkeypatch.setattr(sw.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sw, "read_raw_file", timed_read)
    snapshot = sw.BoundedWindowSnapshot(config, ids, census=False, marker="measurement")
    operations = 0
    while not snapshot.complete:
        snapshot.step()
        operations += 1
    assert operations == 2 * cpus + 4
    assert snapshot.ended - snapshot.started < config.limits.max_sample_duration_s
    state = sw.host_state(snapshot.body(), ids)
    classification, check = config.host_policy.frequency_verdict(state, under_load=False)
    assert classification == microbench.FREQUENCY_DEFERRED_IDLE
    assert check.outcome != "PASS"  # raw availability is never workload evidence


def test_preparation_requirements_are_explicit_original_outputs(tmp_path):
    config = configuration(tmp_path)
    assert config.preparation_requirements() == (
        "original_serving_cell_calibration_material", "original_serving_cell_executed_control_material")
    with pytest.raises(sw.SearchWindowRefused):
        replace(config, package_energy_paths=[[0, "/energy", "/max"]])


@pytest.mark.parametrize("census", [False, True])
def test_final_read_overrun_invalidates_even_a_complete_snapshot(tmp_path, monkeypatch, census):
    config = configuration(tmp_path)
    clock = [1000.0]
    read = sw.read_raw_file
    monkeypatch.setattr(sw.time, "monotonic", lambda: clock[0])
    snapshot = sw.BoundedWindowSnapshot(config, (0,), census=census, marker="fixture")
    def final_read(path, maximum):
        row = read(path, maximum)
        if not snapshot._jobs:
            clock[0] = snapshot.deadline + 0.001
        return row
    monkeypatch.setattr(sw, "read_raw_file", final_read)
    while not snapshot.complete:
        snapshot.step()
    assert snapshot.complete
    assert "snapshot duration exceeded" in snapshot.errors


def test_continuous_valid_notices_do_not_starve_one_step_poll(tmp_path):
    prepared = _prepared(tmp_path)
    authority = uw.ParentUnitEvidenceAuthority(max_records=64)
    stopped, notices, polls = threading.Event(), [], []
    class Producer(de.UnknownParentEvidenceProducer):
        def _phase_for(self, notice):
            notices.append(notice["key"])
            return {"outcome": "unavailable"}
        def _poll_live_evidence(self):
            polls.append(len(notices))
            if len(notices) == 20:
                self._stop.set()
                stopped.set()
    producer = Producer(authority, prepared)
    start = _start(prepared)
    unit = prepared.plan.expected_units[0]
    fence = ps.StageFence("fixture-fence", unit.unit_id, unit.process_id,
        start.lineage_id, start.grant_id, start.container_id,
        start.clock_domain, start.provider_deadline)
    for number in range(20):
        # Public bounded transport API only, never a seeded receipt registry.
        authority.request_observation_phase(start=start, unit=unit, fence=fence,
            request={"phase": "health", "synthetic_notice_index": number})
    producer.start()
    assert stopped.wait(2)
    producer.stop_and_join()
    assert polls == list(range(1, 21))
    assert producer.stopped


def lifecycle_case(tmp_path, *, transient=False):
    from .test_lifecycle_observation import _session, _write_process, _set_ticks
    session, _probe, proc, pressure, *_ = _session(tmp_path)
    session.start()
    try:
        session.phase("load")
        session.attach_target(101)
        session.phase("placement")
        session.phase("health")
        session.phase("warmup")
        session.checkpoint("warmup_during")
        session.phase("measurement")
        if transient:
            # Real bounded reader sees a labelled synthetic foreign process
            # only between endpoints; no argv classifier or evidence verifier.
            _write_process(proc, 202, start=200, ticks=1, cpus="0", cgroup="/foreign")
            session.checkpoint("foreign_born")
            _set_ticks(proc, 202, 9)
            pressure.write_text("some avg10=0 avg60=0 avg300=0 total=20\n"
                                "full avg10=0 avg60=0 avg300=0 total=3\n")
            session.checkpoint("foreign_active")
            (proc / "202").rename(proc / "departed-fixture-process")
            session.checkpoint("foreign_gone")
        else:
            session.checkpoint("measurement_during")
        session.checkpoint("measurement_end")
        session.phase("teardown")
    finally:
        observation = session.finish()
    markers = {row["phase"]: {"boundary_monotonic_s": row["monotonic_s"]}
               for row in observation["phase_boundaries"]}
    markers["measurement_end"] = {"boundary_monotonic_s": next(row["marker_monotonic_s"]
        for row in observation["samples"] if row["marker_label"] == "measurement_end")}
    return observation, markers


def test_contamination_between_endpoints_retains_original_interval_and_memory(tmp_path):
    observation, markers = lifecycle_case(tmp_path, transient=True)
    result = sw.lifecycle_window_facts(observation, markers, [], sw.WindowLimits())
    phase = result["phases"]["measurement"]
    endpoints = [observation["samples"][0], observation["samples"][-1]]
    assert all(202 not in [row["pid"] for row in sample["processes"]] for sample in endpoints)
    assert any(row["pid"] == 202 and row["start_ticks"] == 200
               and row["process_total_cpu_tick_delta"] == 8
               for row in phase["potential_foreign_overlap"])
    assert any(row["memory_delta"] and row["memory_delta"]["psi_some_total"] == 10
               for row in phase["intervals"])
    assert phase["coverage"] == "unknown"  # census transitions never disappear
    assert phase["purpose"].startswith("unknown")
    assert result["no_concurrent_work"].startswith("unknown")


def test_marker_join_is_separate_from_during_phase_coverage(tmp_path):
    observation, markers = lifecycle_case(tmp_path)
    good = sw.lifecycle_window_facts(observation, markers, [], sw.WindowLimits())
    assert good["marker_join"] == "observed"
    assert good["phases"]["measurement"]["coverage"] == "observed"
    changed = copy.deepcopy(markers)
    changed["health"]["boundary_monotonic_s"] += .001
    bad = sw.lifecycle_window_facts(observation, changed, [], sw.WindowLimits())
    assert bad["marker_join"] == "unknown"
    assert bad["phases"]["measurement"]["coverage"] == "observed"
    assert bad["coverage"] == "unknown"


def test_cross_phase_interval_is_diagnostic_not_measured_delta(tmp_path):
    observation, markers = lifecycle_case(tmp_path)
    result = sw.lifecycle_window_facts(observation, markers, [], sw.WindowLimits())
    phase = result["phases"]["measurement"]
    boundary = markers["measurement"]["boundary_monotonic_s"]
    assert any(row["start_monotonic_s"] < boundary for row in phase["boundary_crossing_intervals"])
    assert all(row["start_monotonic_s"] >= boundary for row in phase["intervals"])


def test_zero_duration_and_post_close_host_samples_cannot_supply_coverage(tmp_path):
    observation, markers = lifecycle_case(tmp_path)
    markers["measurement_end"] = copy.deepcopy(markers["measurement"])
    end = markers["measurement_end"]["boundary_monotonic_s"]
    post = [{"started": end + i, "ended": end + i + .01, "complete": True, "errors": []}
            for i in (1, 2)]
    result = sw.lifecycle_window_facts(observation, markers, post, sw.WindowLimits())
    assert result["phases"]["measurement"]["during_sample_count"] == 0
    assert result["host_during_sample_count"] == 0
    assert result["host_measurement_coverage"] == result["coverage"] == "unknown"


@pytest.mark.parametrize("cpus", [48, 96])
def test_actual_parent_poll_loop_uses_zero_queue_wait_between_bounded_reads(tmp_path, monkeypatch, cpus):
    """Actual owner/parent loop, synthetic latency only; no receipt or grant issued."""
    from .test_driver_execution import _as_observed_v2
    from .test_unified_worker import _measure
    from .test_worker_lifecycle import Harness
    from .test_lifecycle_observation import _budgets
    ids = tuple(range(cpus))
    config = configuration(tmp_path, ids)
    prepared, instrument = _as_observed_v2(_prepared(tmp_path), _measure,
        search_window_configuration=config)
    start = replace(_start(prepared), provider_deadline=time.monotonic() + 30)
    unit = prepared.plan.expected_units[0]
    binding = ob.ObservationUnitBinding.from_dict(ob.ObservationUnitBinding(
        "scheduler-fixture", unit.unit_id, unit.process_id, "fixture-fence", start.clock_domain,
        start.child_process.boot_id, {"worker_id": start.worker_id,
            "worker_incarnation": start.worker_generation, "grant_id": start.grant_id,
            "grant_generation": start.grant_generation, "container_identity": ob._plain(start.cgroup_identity)},
        start.container_id, "fixture-claim", {"logical_cpus": list(ids), "gpu_devices": []},
        {"logical_cpus": list(ids), "numa_nodes": [0], "thp_mode": "madvise"},
        (), (), .1, .5, _budgets(max_cpu_ids=cpus), instrument).to_dict())
    harness = Harness(provider=False)
    store = mc.ArtifactStore(prepared.artifact_root)
    try:
        owner = sw.SameAttemptWindowOwner(config=config, prepared=prepared,
            lifecycle=harness.engine, store=store)
        owner.open_unit(start=start, binding=binding, claim={"fixture": "scheduler-only"})
        clock, waits, completed = [time.monotonic()], [], []
        real_read = sw.read_raw_file
        def timed_read(path, maximum):
            row = real_read(path, maximum)
            clock[0] += .0002  # representative labelled raw-file latency
            return row
        monkeypatch.setattr(sw.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(sw, "read_raw_file", timed_read)
        context = SimpleNamespace(unit_id=unit.unit_id, binding=binding, nonce=start.nonce,
                                  descendant_event={"fixture": "scheduler-only"})
        for phase in ("health", "warmup", "measurement"):
            owner.marker(context=context, packet={"phase": phase, "boundary_monotonic_s": clock[0]})

        class ClockedAuthority(uw.ParentUnitEvidenceAuthority):
            def next_notice(self, *, timeout):
                waits.append(timeout)
                clock[0] += timeout  # observe actual parent-supplied queue timeout
                raise queue.Empty

        class Producer(de.UnknownParentEvidenceProducer):
            def _evidence_poll_delay(self):
                return owner.poll_delay()
            def _poll_live_evidence(self):
                owner.poll()
                row = owner._units[unit.unit_id]
                if row["samples"]:
                    completed.extend(row["samples"])
                    self._stop.set()

        producer = Producer(ClockedAuthority(), prepared)
        # Same execution thread throughout, actual _run routes queue timeouts
        # through owner.poll_delay and polls exactly one bounded operation.
        producer._run()
        assert not producer._errors
        assert len(completed) == 1 and not completed[0]["errors"]
        first_zero = waits.index(0.0)
        assert sum(waits[:first_zero]) == pytest.approx(config.limits.cadence_s)
        assert all(0 < delay <= .05 for delay in waits[:first_zero])
        assert len(waits[first_zero:]) == 2 * cpus + 3  # remaining bounded reads
        assert all(delay == 0 for delay in waits[first_zero:])
        assert completed[0]["ended"] - completed[0]["started"] < config.limits.max_sample_duration_s
        assert not owner._receipts and not harness.acquisitions
    finally:
        store.close()
        harness.close()

