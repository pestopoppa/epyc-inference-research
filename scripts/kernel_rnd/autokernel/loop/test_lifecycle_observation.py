"""Tiny fake-proc/sys tests; no host sampling or inference is performed."""
from __future__ import annotations

import copy
import os
import threading
import time

import pytest

from . import lifecycle_observation as lo


class Clock:
    def __init__(self, step=0.01):
        self.value, self.step, self.lock = 0.0, step, threading.Lock()

    def __call__(self):
        with self.lock:
            self.value += self.step
            return self.value


def _stat(pid, start, ticks, cpu=0, comm="fixture"):
    fields = ["0"] * 40
    fields[0], fields[11], fields[12] = "S", str(ticks), "0"
    fields[19], fields[36] = str(start), str(cpu)
    return f"{pid} ({comm}) " + " ".join(fields) + "\n"


def _write_process(proc, pid, *, start, ticks, cpus="0", cgroup="/owned",
                   maps="", kfd=False, comm="fixture"):
    root = proc / str(pid)
    (root / "fd").mkdir(parents=True, exist_ok=True)
    (root / "stat").write_text(_stat(pid, start, ticks, comm=comm))
    (root / "status").write_text(
        f"Cpus_allowed_list:\t{cpus}\nMems_allowed_list:\t0\n")
    (root / "cgroup").write_text(f"0::{cgroup}\n")
    (root / "numa_maps").write_text(
        "00400000 default kernelpagesize_kB=2048 N0=2\n")
    (root / "smaps_rollup").write_text(
        "Rss: 40 kB\nAnonHugePages: 4 kB\nShmemPmdMapped: 0 kB\n"
        "FilePmdMapped: 8 kB\n")
    (root / "maps").write_text(maps)
    if kfd:
        (root / "fd" / "9").symlink_to("/dev/kfd")


def _set_ticks(proc, pid, ticks, *, start=None, comm="fixture"):
    old = lo.parse_proc_stat((proc / str(pid) / "stat").read_text(), pid)
    (proc / str(pid) / "stat").write_text(
        _stat(pid, start or old["start_ticks"], ticks, comm=comm))


def _fixture(tmp_path, *, asymmetric=False):
    proc, cpu, cgroups = tmp_path / "proc", tmp_path / "cpu", tmp_path / "cgroup"
    proc.mkdir()
    owned = cgroups / "owned"
    owned.mkdir(parents=True)
    for number, siblings, node in ((0, "0,4", 0), (4, "0,4", 0),
                                   (1, "1", 1), (2, "2", 0)):
        root = cpu / f"cpu{number}"
        (root / "topology").mkdir(parents=True)
        if asymmetric and number == 4:
            siblings = "4"
        (root / "topology" / "thread_siblings_list").write_text(siblings)
        (root / f"node{node}").mkdir()
    boot, pressure, thp = tmp_path / "boot", tmp_path / "pressure", tmp_path / "thp"
    boot.write_text("boot-fixture\n")
    pressure.write_text(
        "some avg10=0 avg60=0 avg300=0 total=10\n"
        "full avg10=0 avg60=0 avg300=0 total=2\n")
    thp.write_text("always [madvise] never\n")
    (proc / "meminfo").write_text(
        "MemAvailable: 1000 kB\nSwapFree: 500 kB\nSwapTotal: 500 kB\n")
    (proc / "vmstat").write_text("pswpin 0\npswpout 0\n")
    vram, kfd = tmp_path / "global-vram", tmp_path / "kfd"
    vram.write_text("0\n")
    kfd.mkdir()
    probe = lo.FilesystemProbe(
        proc_root=proc, sysfs_cpu_root=cpu, boot_id_path=boot,
        memory_psi_path=pressure, thp_enabled_path=thp, cgroup_root=cgroups,
        global_vram_paths={"gpu0": vram})
    container = lo._stat_identity(owned)
    container["path"] = "/owned"
    return probe, proc, pressure, vram, kfd, container


def _budgets(**overrides):
    result = {"max_samples": 32, "max_pending_markers": 16, "max_processes": 16,
              "max_proc_entries": 64, "max_retained_bytes": 4 * 1024 * 1024,
              "max_read_bytes": 65536, "max_map_entries": 32, "max_fd_entries": 16,
              "max_cpu_ids": 32, "max_numa_rows": 16, "max_dso_entries": 8,
              "phase_ack_timeout_s": 0.2, "join_timeout_s": 0.2,
              "max_probe_duration_s": 0.2}
    result.update(overrides)
    return result


def _context(container, *, backend="cpu", artifact=None, budgets=None, runtime=()):
    return {"schema": lo.CONTEXT_SCHEMA, "observation_id": "obs-1",
            "backend": backend, "instrument_identity_digest": "a" * 64,
            "recipe_identity_digest": "b" * 64, "clock_domain": "monotonic-fixture",
            "cadence_s": 10.0, "gap_limit_s": 11.0, "boot_id": "boot-fixture",
            "worker_binding": {"worker_id": "worker-1", "worker_incarnation": 2,
                               "grant_id": "grant-1", "grant_generation": 3,
                               "container_identity": container},
            "requested_effective_state": {
                "logical_cpus": [0], "numa_nodes": [0], "thp_mode": "madvise"},
            "held_claim": {"logical_cpus": [0],
                           "gpu_devices": ["gpu0"] if backend == "gpu" else []},
            "runtime_witness_keys": list(runtime),
            "required_gpu_dsos": [artifact] if artifact else [],
            "budgets": budgets or _budgets()}


def _resolver(context, *, start=100):
    def resolve(pid):
        return {"pid": pid, "start_ticks": start, "boot_id": "boot-fixture",
                "worker_binding": copy.deepcopy(context["worker_binding"]),
                "binding_ref": "provider-event:owned-child"}
    return resolve


def _session(tmp_path, *, backend="cpu", artifact=None, foreign=None,
             runtime=None, budgets=None, gpu_adapter=True):
    probe, proc, pressure, vram, kfd, container = _fixture(tmp_path)
    maps = ""
    if artifact:
        major, minor = os.major(artifact["dev"]), os.minor(artifact["dev"])
        maps = (f"7f0-7f1 r-xp 00000000 {major:02x}:{minor:02x} "
                f"{artifact['ino']} {artifact['path']}\n")
    _write_process(proc, 101, start=100, ticks=1, maps=maps, kfd=backend == "gpu")
    if backend == "gpu" and gpu_adapter:
        def adapter(binding, devices, _budget):
            root = kfd / str(binding["pid"])
            start = int((root / "start_ticks").read_text())
            boot = (root / "boot_id").read_text().strip()
            if start != binding["start_ticks"] or boot != "boot-fixture":
                return {"status": "unknown", "allocated_bytes": {},
                        "evidence_ref": None, "reason": "fixture identity mismatch"}
            return {"status": "observed",
                    "allocated_bytes": {
                        device: int((root / f"vram_bytes.{device}").read_text())
                        for device in devices},
                    "evidence_ref": "fake-adapter:target-allocation", "reason": None}
        probe.target_gpu_adapter = adapter
    context = _context(container, backend=backend, artifact=artifact,
                       budgets=budgets, runtime=("knob",) if runtime else ())
    session = lo.ObservationSession(
        context, probe=probe, owned_identity_resolver=_resolver(context),
        foreign_verifier=foreign, runtime_verifier=runtime, monotonic=Clock(),
        wall_clock=lambda: "2026-09-09T00:00:00Z")
    return session, probe, proc, pressure, vram, kfd, context


def _complete(session, *, attach=True):
    session.start()
    session.phase("load")
    if attach:
        session.attach_target(101)
    for phase in ("placement", "health", "warmup", "measurement", "teardown"):
        session.phase(phase)
        if phase == "measurement":
            session.checkpoint("measurement_end")
    return session.finish()


def test_topology_expands_split_siblings_and_keeps_asymmetric_numa(tmp_path):
    probe, *_ = _fixture(tmp_path)
    topology = probe.topology(_budgets())
    assert lo.physical_footprint([0], topology, max_cpu_ids=32) == [0, 4]
    assert topology["cpu_to_numa_node"] == {"0": 0, "1": 1, "2": 0, "4": 0}
    with pytest.raises(lo.ObservationError, match="budget"):
        lo.parse_cpu_list("0-1000000000", max_cpu_ids=32)


def test_asymmetric_sibling_partition_is_rejected(tmp_path):
    probe, *_ = _fixture(tmp_path, asymmetric=True)
    with pytest.raises(lo.ObservationError, match="asymmetric"):
        probe.topology(_budgets())


def test_closed_context_requires_integer_worker_and_prepared_artifact(tmp_path):
    _, _, _, _, _, container = _fixture(tmp_path)
    context = _context(container)
    assert lo.validate_context(context)["worker_binding"]["worker_incarnation"] == 2
    with pytest.raises(lo.ObservationError, match="unknown fields"):
        lo.validate_context(context | {"future": 1})
    bad = copy.deepcopy(context)
    bad["worker_binding"]["worker_incarnation"] = "2"
    with pytest.raises(lo.ObservationError, match="integer"):
        lo.validate_context(bad)


def test_whole_lifecycle_has_real_load_window_and_cpu_residency_na(tmp_path):
    record = _complete(_session(tmp_path)[0])
    assert [row["phase"] for row in record["phase_boundaries"]] == list(lo.PHASES)
    assert record["load_window"]["start"]["phase"] == "load"
    assert record["load_window"]["end"]["phase"] == "health"
    assert record["held_claim"]["physical_cpus"] == [0, 4]
    assert record["residency"]["status"] == "not_applicable"
    assert record["verdict"]["status"] == "not_evaluated"
    assert record["shutdown"] == {"status": "resolved", "successor_permitted": True,
                                  "late_writes_accepted": False}
    assert lo.validate_observation(record) == record
    with pytest.raises(lo.ObservationError, match="unknown fields"):
        lo.validate_observation(record | {"future": True})
    nested = copy.deepcopy(record)
    nested["samples"][0]["future"] = True
    body = {key: value for key, value in nested.items() if key != "content_sha256"}
    nested["content_sha256"] = lo._digest(body)
    with pytest.raises(lo.ObservationError, match="unknown fields"):
        lo.validate_observation(nested)


def test_trusted_resolver_binds_actual_post_launch_pid_boot_worker_and_cgroup_inode(tmp_path):
    session, _, _, _, _, _, _ = _session(tmp_path)
    record = _complete(session)
    assert record["target_binding"]["pid"] == 101
    assert record["target_binding"]["binding_ref"] == "provider-event:owned-child"
    assert all(sample["target"]["container_identity"] == record["worker_binding"][
        "container_identity"] for sample in record["samples"] if sample["target"])
    target = next(sample["target"] for sample in record["samples"] if sample["target"])
    assert target["smaps_rollup_kb"]["Rss"] == 40
    assert target["numa_maps"][0]["kernel_page_size_kb"] == 2048


def test_wrong_cgroup_inode_is_retained_as_unknown_target_evidence(tmp_path):
    session, _, _, _, _, _, context = _session(tmp_path)
    context["worker_binding"]["container_identity"]["ino"] += 1
    session.context["worker_binding"]["container_identity"]["ino"] += 1
    record = _complete(session)
    assert record["completeness"] == "unknown"
    assert any("cgroup inode identity differs" in (sample["target_error"] or "")
               for sample in record["samples"])


def test_missing_foreign_and_runtime_verifiers_remain_unknown(tmp_path):
    session, _, proc, _, _, _, _ = _session(tmp_path)
    _write_process(proc, 202, start=200, ticks=1, cpus="4", comm="llama-server")
    session.start()
    _set_ticks(proc, 202, 8, comm="llama-server")
    session.phase("load")
    for phase in ("placement", "health", "warmup", "measurement", "teardown"):
        session.phase(phase)
    record = session.finish()
    overlaps = [item for row in record["intervals"]
                for item in row["potential_foreign_overlap"]]
    assert overlaps[0]["verification"] == {
        "status": "unknown", "kind": None, "evidence_ref": None,
        "reason": "verifier_unavailable"}
    assert not any("policy_effect" in item for item in overlaps)


def test_trusted_foreign_verifier_distinguishes_ordinary_without_policy(tmp_path):
    def verifier(_process):
        return {"status": "verified", "kind": "ordinary",
                "evidence_ref": "provider:ordinary-build", "reason": None}
    session, _, proc, _, _, _, _ = _session(tmp_path, foreign=verifier)
    _write_process(proc, 202, start=200, ticks=1, cpus="4")
    session.start()
    _set_ticks(proc, 202, 5)
    session.phase("load")
    for phase in ("placement", "health", "warmup", "measurement", "teardown"):
        session.phase(phase)
    record = session.finish()
    overlap = next(item for row in record["intervals"]
                   for item in row["potential_foreign_overlap"])
    assert overlap["process_total_cpu_tick_delta"] == 4
    assert overlap["potential_physical_claim_overlap"] == [0, 4]
    assert overlap["verification"]["kind"] == "ordinary"


def test_counter_reset_pid_reuse_and_born_disappeared_are_not_silent(tmp_path):
    session, _, proc, _, _, _, _ = _session(tmp_path)
    _write_process(proc, 202, start=200, ticks=9)
    session.start()
    _set_ticks(proc, 202, 1)
    session.phase("load")
    _set_ticks(proc, 202, 2, start=201)
    session.phase("placement")
    for phase in ("health", "warmup", "measurement", "teardown"):
        session.phase(phase)
    record = session.finish()
    codes = {issue["code"] for issue in record["issues"]}
    assert {"counter_reset", "pid_reuse", "process_census_changed"} <= codes
    assert record["completeness"] == "unknown"


def test_pid_start_is_rechecked_after_target_multi_file_read(tmp_path):
    base, proc, _, _, _, container = _fixture(tmp_path)
    _write_process(proc, 101, start=100, ticks=1)

    class ReusingProbe(lo.FilesystemProbe):
        inside_target = False

        def _target(self, *args, **kwargs):
            self.inside_target = True
            try:
                return super()._target(*args, **kwargs)
            finally:
                self.inside_target = False

        def _process(self, pid, topology, budget):
            row = super()._process(pid, topology, budget)
            if self.inside_target:
                _set_ticks(self.proc_root, pid, row["cpu_ticks"], start=101)
            return row

    probe = ReusingProbe(
        proc_root=base.proc_root, sysfs_cpu_root=base.sysfs_cpu_root,
        boot_id_path=base.boot_id_path, memory_psi_path=base.memory_psi_path,
        thp_enabled_path=base.thp_enabled_path, cgroup_root=base.cgroup_root,
        target_gpu_adapter=base.target_gpu_adapter)
    context = _context(container)
    session = lo.ObservationSession(
        context, probe=probe, owned_identity_resolver=_resolver(context),
        monotonic=Clock(), wall_clock=lambda: "2026-09-09T00:00:00Z")
    record = _complete(session)
    assert any("changed during multi-file read" in (sample["target_error"] or "")
               for sample in record["samples"])
    assert record["completeness"] == "unknown"


def test_disappearing_process_gap_retains_memory_and_pressure_data(tmp_path):
    session, _, proc, _, _, _, _ = _session(tmp_path)
    _write_process(proc, 202, start=200, ticks=1)
    (proc / "202" / "status").unlink()
    record = _complete(session, attach=False)
    assert any(sample["census_gaps"] for sample in record["samples"])
    assert all(sample["memory"]["MemAvailable"] == 1000 for sample in record["samples"])
    assert record["completeness"] == "unknown"


def test_pressure_burst_is_in_warmup_window_not_after_finish(tmp_path):
    session, _, _, pressure, _, _, _ = _session(tmp_path)
    session.start()
    for phase in ("load", "placement", "health"):
        session.phase(phase)
    pressure.write_text(
        "some avg10=1 avg60=0 avg300=0 total=40\n"
        "full avg10=0 avg60=0 avg300=0 total=7\n")
    for phase in ("warmup", "measurement", "teardown"):
        session.phase(phase)
    record = session.finish()
    warmup = next(row for row in record["intervals"] if row["phase"] == "warmup")
    assert warmup["memory_delta"]["psi_some_total"] == 30
    pressure.write_text(pressure.read_text().replace("total=40", "total=400"))
    assert max(row["memory_psi"]["some"]["total"] for row in record["samples"]) == 40


def test_gpu_positive_requires_prepared_mapping_target_kfd_and_target_allocation(tmp_path):
    dso = tmp_path / "libfixture.so"
    dso.write_bytes(b"prepared DSO")
    artifact = lo.prepare_artifact_identity(dso, max_bytes=1024)
    session, _, _, _, global_vram, kfd, _ = _session(
        tmp_path, backend="gpu", artifact=artifact)
    target = kfd / "101"
    target.mkdir()
    (target / "start_ticks").write_text("100\n")
    (target / "boot_id").write_text("boot-fixture\n")
    (target / "vram_bytes.gpu0").write_text("4096\n")
    global_vram.write_text("999999\n")
    record = _complete(session)
    assert record["residency"]["status"] == "observed"
    sample = next(row for row in record["samples"] if row["phase"] == "measurement")
    assert sample["target"]["required_gpu_dsos"][0]["matched_mapping_count"] == 1
    assert sample["gpu"]["target_attribution"]["allocated_bytes"] == {"gpu0": 4096}


def test_global_vram_and_open_kfd_without_target_attribution_stay_unknown(tmp_path):
    dso = tmp_path / "libfixture.so"
    dso.write_bytes(b"prepared DSO")
    artifact = lo.prepare_artifact_identity(dso, max_bytes=1024)
    session, _, _, _, global_vram, _, _ = _session(
        tmp_path, backend="gpu", artifact=artifact, gpu_adapter=False)
    global_vram.write_text("999999\n")
    record = _complete(session)
    assert record["residency"]["status"] == "unknown"
    measurement = next(row for row in record["samples"] if row["phase"] == "measurement")
    assert measurement["gpu"]["global_vram_bytes"] == {"gpu0": 999999}
    assert measurement["gpu"]["target_attribution"]["status"] == "unknown"


def test_two_in_generation_target_samples_with_zero_allocation_are_not_observed(tmp_path):
    dso = tmp_path / "libfixture.so"
    dso.write_bytes(b"prepared DSO")
    artifact = lo.prepare_artifact_identity(dso, max_bytes=1024)
    session, _, _, _, _, kfd, _ = _session(tmp_path, backend="gpu", artifact=artifact)
    target = kfd / "101"
    target.mkdir()
    (target / "start_ticks").write_text("100\n")
    (target / "boot_id").write_text("boot-fixture\n")
    (target / "vram_bytes.gpu0").write_text("0\n")
    record = _complete(session)
    assert record["residency"]["measurement_samples"] == 2
    assert record["residency"]["status"] == "not_observed"


def test_memory_subprobe_failure_retains_independent_gpu_facts(tmp_path):
    dso = tmp_path / "libfixture.so"
    dso.write_bytes(b"prepared DSO")
    artifact = lo.prepare_artifact_identity(dso, max_bytes=1024)
    session, _, proc, _, _, kfd, _ = _session(tmp_path, backend="gpu", artifact=artifact)
    target = kfd / "101"
    target.mkdir()
    (target / "start_ticks").write_text("100\n")
    (target / "boot_id").write_text("boot-fixture\n")
    (target / "vram_bytes.gpu0").write_text("4096\n")
    (proc / "meminfo").unlink()
    record = _complete(session)
    sample = next(row for row in record["samples"] if row["phase"] == "measurement")
    assert sample["memory"] is None
    assert sample["gpu"]["target_attribution"]["allocated_bytes"] == {"gpu0": 4096}
    assert any(error["probe"] == "memory" for error in sample["subprobe_errors"])
    assert record["observer_cost"]["failed_read_count"] > 0
    assert record["observer_cost"]["completed_read_count"] >= record[
        "observer_cost"]["failed_read_count"]


def test_dso_path_replacement_cannot_match_prepared_inode(tmp_path):
    dso = tmp_path / "libfixture.so"
    dso.write_bytes(b"old")
    artifact = lo.prepare_artifact_identity(dso, max_bytes=1024)
    replacement_path = tmp_path / "replacement.so"
    replacement_path.write_bytes(b"new")
    os.replace(replacement_path, dso)
    session, _, proc, _, _, _, _ = _session(tmp_path, backend="gpu", artifact=artifact)
    # Fixture maps are formed from the prepared identity, proving matching is inode-based;
    # overwrite maps with the replacement inode to model a path-identical replacement.
    replacement = dso.stat()
    major, minor = os.major(replacement.st_dev), os.minor(replacement.st_dev)
    (proc / "101" / "maps").write_text(
        f"7f0-7f1 r-xp 0 {major:02x}:{minor:02x} {replacement.st_ino} {dso}\n")
    record = _complete(session)
    measurement = next(row for row in record["samples"] if row["phase"] == "measurement")
    assert measurement["target"]["required_gpu_dsos"][0]["matched_mapping_count"] == 0


def test_same_inode_dso_modification_invalidates_prepared_metadata(tmp_path):
    dso = tmp_path / "libfixture.so"
    dso.write_bytes(b"old")
    artifact = lo.prepare_artifact_identity(dso, max_bytes=1024)
    dso.write_bytes(b"new")
    assert dso.stat().st_ino == artifact["ino"]
    session, *_ = _session(tmp_path, backend="gpu", artifact=artifact)
    record = _complete(session)
    measurement = next(row for row in record["samples"] if row["phase"] == "measurement")
    dso_evidence = measurement["target"]["required_gpu_dsos"][0]
    assert dso_evidence["matched_mapping_count"] == 1
    assert dso_evidence["prepared_identity_current"] is False
    assert dso_evidence["current_identity_error"] == "prepared artifact metadata changed"
    assert record["residency"]["status"] == "unknown"


def test_mapped_deleted_prepared_inode_is_retained_but_not_upgraded(tmp_path):
    dso = tmp_path / "libfixture.so"
    dso.write_bytes(b"old")
    artifact = lo.prepare_artifact_identity(dso, max_bytes=1024)
    session, *_ = _session(tmp_path, backend="gpu", artifact=artifact)
    dso.unlink()
    record = _complete(session)
    measurement = next(row for row in record["samples"] if row["phase"] == "measurement")
    evidence = measurement["target"]["required_gpu_dsos"][0]
    assert evidence["matched_mapping_count"] == 1
    assert evidence["prepared_identity_current"] is False
    assert "FileNotFoundError" in evidence["current_identity_error"]


def test_runtime_witness_requires_trusted_verifier_and_preserves_states(tmp_path):
    states = iter(("unsupported", "compiled_unproven", "fired_under_target"))

    def verifier(_key, _target):
        state = next(states, "fired_under_target")
        return {"status": state,
                "evidence_ref": "trace:positive-control" if state == "fired_under_target" else None,
                "reason": None}

    session, *_ = _session(tmp_path, runtime=verifier)
    record = _complete(session)
    observed = [sample["runtime_witnesses"][0]["status"] for sample in record["samples"]]
    assert {"unsupported", "compiled_unproven", "fired_under_target"} <= set(observed)


def test_byte_entry_sample_and_total_retention_budgets_fail_explicitly(tmp_path):
    budgets = _budgets(max_samples=2, max_read_bytes=128, max_retained_bytes=1,
                       phase_ack_timeout_s=0.05)
    session, _, proc, _, _, _, _ = _session(tmp_path, budgets=budgets)
    (proc / "meminfo").write_text("x" * 129)
    session.start()
    session.phase("load")
    for phase in ("placement", "health", "warmup", "measurement", "teardown"):
        session.phase(phase)
    record = session.finish()
    codes = {issue["code"] for issue in record["issues"]}
    assert "sample_failed" in codes
    assert "sample_budget_exhausted" in codes
    assert "retained_byte_budget_exhausted" in codes
    assert record["observer_cost"]["sample_budget"] == 2
    assert record["observer_cost"]["dropped_marker_count"] > 0
    assert record["observer_cost"]["dropped_sample_count"] == 2


def test_proc_entry_budget_is_distinct_from_numeric_process_budget(tmp_path):
    budgets = _budgets(max_processes=2, max_proc_entries=32)
    session, _, proc, _, _, _, _ = _session(tmp_path, budgets=budgets)
    for index in range(20):
        (proc / f"nonnumeric-{index}").write_text("fixture")
    record = _complete(session)
    assert not any(error["probe"] == "process_census"
                   for sample in record["samples"] for error in sample["subprobe_errors"])


def test_repeated_sample_exhaustion_aggregates_one_bounded_issue(tmp_path):
    session, *_ = _session(tmp_path, budgets=_budgets(max_samples=1))
    with session._condition:
        session._scheduled = 1
        for index in range(20):
            assert session._enqueue_locked("setup", "periodic", float(index),
                                           "2026-09-09T00:00:00Z") is None
    issues = [row for row in session._issues if row["code"] == "sample_budget_exhausted"]
    assert len(issues) == 1
    assert issues[0]["counts"]["dropped_marker_count"] == 20


def test_truthful_read_windows_detect_sampling_gap(tmp_path):
    session, *_ = _session(tmp_path)
    session.monotonic = Clock(step=20.0)
    record = _complete(session)
    assert all(sample["read_ended_monotonic_s"] >= sample["read_started_monotonic_s"]
               for sample in record["samples"])
    assert any(issue["code"] == "sampling_gap" for issue in record["issues"])
    assert record["completeness"] == "unknown"


def test_permanently_blocked_reader_finishes_bounded_and_freezes_late_diagnostics(tmp_path):
    probe, proc, _, _, _, container = _fixture(tmp_path)
    _write_process(proc, 101, start=100, ticks=1)
    entered, release = threading.Event(), threading.Event()

    class BlockingProbe(lo.FilesystemProbe):
        def capture(self, **kwargs):
            entered.set()
            release.wait(5.0)
            return super().capture(**kwargs)

    blocking = BlockingProbe(
        proc_root=probe.proc_root, sysfs_cpu_root=probe.sysfs_cpu_root,
        boot_id_path=probe.boot_id_path, memory_psi_path=probe.memory_psi_path,
        thp_enabled_path=probe.thp_enabled_path, cgroup_root=probe.cgroup_root,
        target_gpu_adapter=probe.target_gpu_adapter)
    context = _context(container, budgets=_budgets(
        phase_ack_timeout_s=0.01, join_timeout_s=0.02))
    callbacks = []
    session = lo.ObservationSession(
        context, probe=blocking, owned_identity_resolver=_resolver(context),
        monotonic=Clock(), wall_clock=lambda: "2026-09-09T00:00:00Z",
        record_callback=lambda row: callbacks.append(row) or "artifact:1")
    session.start()
    assert entered.wait(0.2)
    for phase in ("load", "placement", "health", "warmup", "measurement", "teardown"):
        session.phase(phase)
    before = time.monotonic()
    record = session.finish()
    assert time.monotonic() - before < 0.2
    assert record["shutdown"]["status"] == "unresolved"
    assert record["observer_cost"]["reader_cost_status"] == "unknown"
    assert record["observer_cost"]["unresolved_read_count"] == 1
    assert record["observer_cost"]["completed_read_count"] == 0
    assert session.successor_permitted is False
    frozen = session.record()
    release.set()
    assert session._thread is not None
    session._thread.join(0.5)
    assert session.record() == frozen
    assert callbacks == [frozen]
    assert session.reconcile_shutdown() is True
    assert session.successor_permitted is True
    assert session.record() == frozen


def test_loaded_instrument_identity_pins_closure_or_marks_unproven():
    factor = 3

    def measured(value):
        return value * factor

    identity = lo.loaded_instrument_identity(
        measurement_callable=measured, clock_callable=time.monotonic,
        supporting_callables=[lo.parse_proc_stat], used_constants={"phase": "measurement"},
        dependency_packages=[])
    assert identity["measurement_callable"]["configuration_status"] == "pinned"
    assert identity["clock_callable"]["configuration_status"] == "unproven"
    assert identity["configuration_complete"] is False
    assert lo.validate_instrument_identity(identity) == identity
    changed = lo.loaded_instrument_identity(
        measurement_callable=lambda value: value * 4, clock_callable=time.monotonic,
        supporting_callables=[lo.parse_proc_stat], used_constants={"phase": "measurement"},
        dependency_packages=[])
    assert changed["sha256"] != identity["sha256"]
    changed_constant = lo.loaded_instrument_identity(
        measurement_callable=measured, clock_callable=time.monotonic,
        supporting_callables=[lo.parse_proc_stat], used_constants={"phase": "warmup"},
        dependency_packages=[])
    assert changed_constant["sha256"] != identity["sha256"]

    class Evaluator:
        def __init__(self, scale):
            self.scale = scale

        def measure(self, value):
            return value * self.scale

    first = lo.callable_identity(Evaluator(3).measure)
    second = lo.callable_identity(Evaluator(4).measure)
    assert first["configuration_status"] == "pinned"
    assert first["configuration_sha256"] != second["configuration_sha256"]


def test_loaded_instrument_identity_includes_slots_with_instance_dict():
    class Configured:
        __slots__ = ("factor",)

        def __init__(self, factor):
            self.factor = factor

    class Evaluator(Configured):
        __slots__ = ("__dict__",)

        def measure(self, value):
            return value * self.factor

    first = lo.callable_identity(Evaluator(3).measure)
    second = lo.callable_identity(Evaluator(4).measure)
    assert first["configuration_status"] == "pinned"
    assert first["configuration_sha256"] != second["configuration_sha256"]


def test_loaded_instrument_identity_does_not_pin_hidden_native_base_payload():
    class Evaluator(float):
        __slots__ = ()

        def measure(self, value):
            return float(self) * value

    first = lo.callable_identity(Evaluator(3).measure)
    second = lo.callable_identity(Evaluator(4).measure)
    assert first["configuration_status"] == "unproven"
    assert second["configuration_status"] == "unproven"


def test_reader_cost_counts_completed_reads_even_when_samples_are_not_retained(tmp_path):
    session, *_ = _session(tmp_path, budgets=_budgets(max_retained_bytes=1))
    record = _complete(session)
    cost = record["observer_cost"]
    assert cost["sample_count"] == 0
    assert cost["dropped_sample_count"] == cost["oversized_sample_count"] > 0
    assert cost["completed_read_count"] == cost["dropped_sample_count"]
    assert cost["failed_read_count"] == 0
    assert cost["reader_seconds"] > 0
    assert cost["reader_cost_status"] == "complete"
    assert cost["unresolved_read_count"] == 0


def test_unjoined_idle_reader_finishes_with_unknown_zero_active_cost(tmp_path):
    probe, _, _, _, _, container = _fixture(tmp_path)
    entered, release = threading.Event(), threading.Event()

    class IdleReaderSession(lo.ObservationSession):
        def _run(self):
            entered.set()
            release.wait(5.0)

    context = _context(container, budgets=_budgets(
        phase_ack_timeout_s=0.01, join_timeout_s=0.02))
    session = IdleReaderSession(
        context, probe=probe, owned_identity_resolver=_resolver(context),
        monotonic=Clock(), wall_clock=lambda: "2026-09-09T00:00:00Z")
    session.start()
    assert entered.wait(0.2)
    record = session.finish()
    assert record["shutdown"]["status"] == "unresolved"
    assert record["observer_cost"]["reader_cost_status"] == "unknown"
    assert record["observer_cost"]["unresolved_read_count"] == 0
    frozen = session.record()
    release.set()
    assert session._thread is not None
    session._thread.join(0.5)
    assert session.record() == frozen
    assert session.reconcile_shutdown() is True
    assert session.record() == frozen


@pytest.mark.parametrize("duration,cadence,markers,expected", [
    (32.0, 0.1, 9, 329), (1.0, 0.25, 9, 13), (1.01, 0.25, 9, 14),
    (0.1, 1.0, 0, 1), (5.0, 1.0, 9, 14),
])
def test_required_sample_capacity(duration, cadence, markers, expected):
    assert lo.required_sample_capacity(max_duration_s=duration,
        cadence_s=cadence, nonperiodic_samples=markers) == expected


@pytest.mark.parametrize("field,value", [
    ("max_duration_s", True), ("max_duration_s", 0), ("max_duration_s", -1),
    ("max_duration_s", float("inf")), ("max_duration_s", float("nan")),
    ("cadence_s", True), ("cadence_s", 0), ("cadence_s", -1),
    ("cadence_s", float("inf")), ("cadence_s", float("nan")),
    ("nonperiodic_samples", True), ("nonperiodic_samples", -1),
    ("nonperiodic_samples", 1.5),
])
def test_sample_capacity_refuses_invalid_numeric_inputs(field, value):
    values = {"max_duration_s": 32.0, "cadence_s": 0.1, "nonperiodic_samples": 9}
    with pytest.raises(lo.ObservationError):
        lo.required_sample_capacity(**(values | {field: value}))


def test_sample_capacity_refuses_overflow_without_allocating_samples():
    with pytest.raises(lo.ObservationError, match="finite"):
        lo.required_sample_capacity(max_duration_s=1e308, cadence_s=1e-308,
                                    nonperiodic_samples=9)


class ScriptedCondition:
    """Single-thread deterministic scheduler harness, not an evidence probe."""

    def __init__(self, wait):
        self.wait_callback = wait
        self.depth = 0
        self.waits = []

    def __enter__(self):
        self.depth += 1

    def __exit__(self, *_args):
        self.depth -= 1

    def wait(self, timeout):
        self.waits.append(timeout)
        self.wait_callback(timeout)
        return True  # Wakeups do not themselves establish expiry.

    def notify(self):
        pass


def _scripted_reader(tmp_path):
    session, *_ = _session(tmp_path)
    session.context["cadence_s"] = 1.0
    session._started = session._accepting = True
    session._phase = "placement"
    clock = [0.0]
    samples = []
    condition = ScriptedCondition(lambda timeout: clock.__setitem__(0, clock[0] + timeout))
    session._condition = condition
    def time_pair():
        assert condition.depth == 0, "clock read must remain outside state lock"
        return clock[0], "2026-09-09T00:00:00Z"
    session._time_pair = time_pair
    session._capture_marker = lambda marker: marker
    def commit(sample, marker):
        session._active_reads -= 1
        samples.append((sample["phase"], sample["kind"], sample["marker_monotonic_s"]))
        marker["done"].set()
        session._stopping = True
    session._commit = commit
    return session, clock, samples, condition


def test_empty_spurious_wakes_keep_due_time_without_early_or_postponed_samples(tmp_path):
    session, clock, samples, condition = _scripted_reader(tmp_path)
    def wake(_remaining):
        assert len(condition.waits) <= 4, "spurious wake restarted the cadence forever"
        clock[0] += 0.25
    condition.wait_callback = wake
    session._run()
    assert condition.waits == [1.0, 0.75, 0.5, 0.25]
    assert samples == [("placement", "periodic", 1.0)]
    assert session._scheduled == 1


@pytest.mark.parametrize("race", ["phase", "checkpoint", "stop"])
def test_unlocked_periodic_clock_race_does_not_queue_stale_phase_or_overtake_marker(tmp_path, race):
    session, clock, samples, _ = _scripted_reader(tmp_path)
    original_clock = session._time_pair
    clock_calls = 0
    def racing_clock():
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 2:
            with session._condition:
                if race == "stop":
                    session._stopping = True
                else:
                    if race == "phase":
                        session._phase = "health"
                    session._enqueue_locked(session._phase,
                        "boundary" if race == "phase" else "checkpoint", clock[0],
                        "2026-09-09T00:00:00Z", None if race == "phase" else "fixture-checkpoint")
        return original_clock()
    session._time_pair = racing_clock
    session._run()
    expected = ([] if race == "stop" else [
        ("health", "boundary", 1.0) if race == "phase" else ("placement", "checkpoint", 1.0)])
    assert samples == expected
    assert session._scheduled == len(expected)
    assert not session._pending


@pytest.mark.parametrize("shortfall", [0, 1])
def test_exact_periodic_allowance_plus_nine_hooks_reaches_capacity_boundary(tmp_path, shortfall):
    # Zero-cost scripted reads exercise the worst count, not hardware evidence:
    # all 100 periodic deadlines elapse in placement before the last five hooks.
    session, clock, samples, condition = _scripted_reader(tmp_path)
    duration, cadence, periodic_allowance = 100.0, 1.0, 100
    capacity = lo.required_sample_capacity(max_duration_s=duration,
        cadence_s=cadence, nonperiodic_samples=9)
    assert capacity == periodic_allowance + 9
    session.context["budgets"]["max_samples"] = capacity - shortfall
    session._phase = "setup"
    session._wait_marker = lambda *_args: None
    periodic_count = 0
    hooks = []
    def commit(sample, marker):
        nonlocal periodic_count
        session._active_reads -= 1
        samples.append((sample["phase"], sample["kind"], sample["marker_monotonic_s"]))
        marker["done"].set()
        if sample["kind"] == "periodic":
            periodic_count += 1
            assert periodic_count <= periodic_allowance
            if periodic_count == periodic_allowance:
                session.phase("health")
            return
        hooks.append((sample["phase"], sample["kind"], sample["marker_label"]))
        if sample["kind"] == "target_attached":
            session.phase("placement")
        elif sample["kind"] == "checkpoint":
            session.phase("teardown")
            if shortfall:
                session._stopping = True  # The last hook was explicitly refused.
        elif sample["phase"] == "setup":
            session.phase("load")
        elif sample["phase"] == "load":
            session.attach_target(101)
        elif sample["phase"] == "health":
            session.phase("warmup")
        elif sample["phase"] == "warmup":
            session.phase("measurement")
        elif sample["phase"] == "measurement":
            session.checkpoint("measurement_end")
        elif sample["phase"] == "teardown":
            session._stopping = True
    session._commit = commit
    with session._condition:
        session._enqueue_locked("setup", "boundary", 0.0, "2026-09-09T00:00:00Z")
    session._run()
    expected_hooks = [
        ("setup", "boundary", None), ("load", "boundary", None),
        ("load", "target_attached", None), ("placement", "boundary", None),
        ("health", "boundary", None), ("warmup", "boundary", None),
        ("measurement", "boundary", None),
        ("measurement", "checkpoint", "measurement_end"),
        ("teardown", "boundary", None)]
    assert clock[0] == duration
    assert condition.waits == [cadence] * periodic_allowance
    assert [row for row in samples if row[1] == "periodic"] == [
        ("placement", "periodic", float(index)) for index in range(1, 101)]
    assert hooks == (expected_hooks if not shortfall else expected_hooks[:-1])
    assert len(samples) == session._scheduled == capacity - shortfall
    assert session._dropped_markers == shortfall
    assert [row["code"] for row in session._issues] == (
        ["sample_budget_exhausted"] if shortfall else [])
    assert not session._pending


def test_sized_long_placement_retains_measurement_end_and_teardown(tmp_path):
    cadence, duration = 0.02, 2.0
    capacity = lo.required_sample_capacity(max_duration_s=duration,
        cadence_s=cadence, nonperiodic_samples=9)
    session, *_ = _session(tmp_path, budgets=_budgets(max_samples=capacity))
    session.context.update(cadence_s=cadence, gap_limit_s=0.1)
    session.monotonic = time.monotonic
    observed_placement = threading.Event()
    capture = session._capture_marker
    periodic_count = 0
    def count(sample):
        nonlocal periodic_count
        value = capture(sample)
        if sample["kind"] == "periodic" and sample["phase"] == "placement":
            periodic_count += 1
            if periodic_count == 8:
                observed_placement.set()
        return value
    session._capture_marker = count
    session.start()
    try:
        session.phase("load")
        session.attach_target(101)
        session.phase("placement")
        assert observed_placement.wait(1.0)
        for phase in ("health", "warmup", "measurement"):
            session.phase(phase)
        session.checkpoint("measurement_end")
    finally:
        session.phase("teardown")
        record = session.finish()
    assert record["ended_monotonic_s"] - record["started_monotonic_s"] <= duration
    assert record["observer_cost"]["sample_count"] <= capacity
    assert record["observer_cost"]["dropped_marker_count"] == 0
    markers = [(row["phase"], row["kind"], row["marker_label"]) for row in record["samples"]]
    assert ("measurement", "checkpoint", "measurement_end") in markers
    assert ("teardown", "boundary", None) in markers
    assert {row["phase"] for row in record["samples"]} == set(lo.PHASES)
    assert not any(row["code"] == "sample_budget_exhausted" for row in record["issues"])
