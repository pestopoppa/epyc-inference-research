"""Hermetic binding tests: fake proc/sys and one tiny owned Python child only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import observation_binding as ob
from . import serving
from .test_lifecycle_observation import (Clock, _budgets, _fixture, _resolver,
                                         _set_ticks, _write_process)


def _instrument(store):
    return ob.seal_loaded_instrument(store=store,
        measurement_callable=serving._measure_once,
        fence_clock=time.monotonic, serving_timer=time.time)


def _binding(container, instrument):
    value = ob.ObservationUnitBinding(
        "obs-unit-1", "unit-1", "process-generation-1", "fence-1",
        "fixture-monotonic", "boot-fixture",
        {"worker_id": "worker-1", "worker_incarnation": 2,
         "grant_id": "grant-1", "grant_generation": 3,
         "container_identity": container},
        "container-1", "journal:active-claim", {"logical_cpus": [0], "gpu_devices": []},
        {"logical_cpus": [0], "numa_nodes": [0], "thp_mode": "madvise"},
        (), (), 10.0, 11.0, _budgets(), instrument)
    return ob.ObservationUnitBinding.from_dict(value.to_dict())


def test_loaded_real_code_identity_matches_tiny_child(tmp_path):
    store = mc.ArtifactStore(tmp_path / "artifacts")
    try:
        parent = ob.loaded_planned_serving_identity(
            measurement_callable=serving._measure_once,
            fence_clock=time.monotonic, serving_timer=time.time)
        program = """
import json, time
from autokernel.loop import observation_binding as ob
from autokernel.loop import serving
row = ob.loaded_planned_serving_identity(measurement_callable=serving._measure_once,
    fence_clock=time.monotonic, serving_timer=time.time)
print(json.dumps(ob._plain(row), sort_keys=True))
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str((__import__("pathlib").Path.cwd() / "scripts/kernel_rnd"))
        child = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, env=env)
        pid = child.pid
        print(f"owned_test_child_pid={pid}")
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 0, stderr
        assert json.loads(stdout) == ob._plain(parent)
        assert json.loads(stdout)["sha256"] == parent["sha256"]
        assert parent["clock_callable"]["implementation_status"] == "pinned"
        assert parent["supporting_callables"][0]["implementation_status"] == "pinned"
        assert parent["configuration_complete"] is True
        provenance = parent["used_constants"]["builtin_callable_provenance"]
        assert provenance["fence_clock"]["provider_artifact"]["sha256"]
        assert provenance["serving_timer"]["clock"]["name"] == "time"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        store.close()


def test_provider_artifact_change_during_hash_is_refused(tmp_path, monkeypatch):
    artifact = tmp_path / "fixture-provider.so"
    artifact.write_bytes(b"original-provider-bytes")
    real_read = os.read
    changed = False

    def changing_read(fd, count):
        nonlocal changed
        value = real_read(fd, count)
        if value and not changed:
            changed = True
            artifact.write_bytes(b"replaced-provider-bytes")
        return value

    monkeypatch.setattr(os, "read", changing_read)
    with pytest.raises(ob.ObservationBindingError, match="changed"):
        ob._artifact_identity(artifact)


def test_provider_artifact_fifo_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "provider.fifo"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(ob.ObservationBindingError, match="bounded file"):
        ob._artifact_identity(fifo)
    assert time.monotonic() - started < 1.0


def test_provider_maps_read_is_bounded_before_join(monkeypatch):
    real_open = os.open
    real_read = os.read
    maps_fd = None

    def fake_open(path, flags, *args, **kwargs):
        nonlocal maps_fd
        if Path(path) == Path("/proc/self/maps"):
            maps_fd = real_open("/dev/zero", os.O_RDONLY | os.O_NONBLOCK)
            return maps_fd
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "read", real_read)
    monkeypatch.setattr(ob, "_MAX_MAPS_BYTES", 1024)
    with pytest.raises(ob.ObservationBindingError, match="maps exceed byte bound"):
        ob._mapped_executable_provider(1)


def test_real_session_seal_reopen_preserves_all_phase_facts_unknown_without_adapters(tmp_path):
    probe, proc, _, _, _, container = _fixture(tmp_path)
    _write_process(proc, 101, start=100, ticks=1)
    _write_process(proc, 202, start=200, ticks=2, cpus="0", cgroup="/foreign")
    store = mc.ArtifactStore(tmp_path / "artifacts")
    try:
        instrument = _instrument(store)
        binding = _binding(container, instrument)
        context = {"schema": lo.CONTEXT_SCHEMA, "observation_id": binding.observation_id,
            "backend": "cpu", "instrument_identity_digest": instrument.identity_sha256,
            "recipe_identity_digest": "b" * 64, "clock_domain": binding.clock_domain,
            "cadence_s": binding.cadence_s, "gap_limit_s": binding.gap_limit_s,
            "boot_id": binding.boot_id, "worker_binding": ob._plain(binding.worker_binding),
            "requested_effective_state": ob._plain(binding.requested_effective_state),
            "held_claim": ob._plain(binding.held_claim), "runtime_witness_keys": [],
            "required_gpu_dsos": [], "budgets": ob._plain(binding.budgets)}
        session = lo.ObservationSession(context, probe=probe,
                                        owned_identity_resolver=_resolver(context),
                                        monotonic=Clock())
        session.start()
        session.phase("load")
        session.attach_target(101)
        for tick, phase in enumerate(
                ("placement", "health", "warmup", "measurement", "teardown"), 3):
            _set_ticks(proc, 202, tick, start=200)
            session.phase(phase)
        record = session.finish()
        reference = ob.seal_observation(store=store, binding=binding, record=record)
        link = ob.validate_reopened_observation(reference, store=store,
            expected={"unit_id": "unit-1", "process_generation_id": "process-generation-1",
                      "fence_id": "fence-1", "active_claim_ref": "journal:active-claim",
                      "container_id": "container-1",
                      "capture_context": {"worker_id": "worker-1",
                                          "worker_incarnation": 2,
                                          "grant_id": "grant-1"},
                      },
            instrument=instrument)
        assert set(link.phase_facts) == set(ob.SEMANTIC_PHASES)
        assert link.observation_status == "unknown"
        assert link.purpose_status == "unknown"
        assert link.runtime_status == "unknown"
        assert link.gpu_status == "unknown"
        assert reference.successor_permitted is True
        def purpose(_kind, candidate, **_kwargs):
            kind = "model_inference" if candidate["phase"] == "measurement" else "ordinary"
            return {"status": "verified", "kind": kind,
                    "evidence_ref": f"parent:{kind}:{candidate['phase']}"}
        verified = ob.validate_reopened_observation(reference, store=store,
            expected={"unit_id": "unit-1", "process_generation_id": "process-generation-1",
                      "fence_id": "fence-1", "active_claim_ref": "journal:active-claim",
                      "container_id": "container-1", "capture_context": {
                          "worker_id": "worker-1", "worker_incarnation": 2,
                          "grant_id": "grant-1"}}, instrument=instrument,
            verifiers=ob.ParentObservationVerifiers(purpose=purpose))
        assert verified.phase_facts["measurement"]["inference"]
        assert any(verified.phase_facts[phase]["ordinary"]
                   for phase in ("load", "placement", "warmup", "teardown"))
    finally:
        store.close()


def test_child_verified_label_is_never_parent_verification(tmp_path):
    # The parent status helper accepts only the explicit parent adapter result.
    candidate = {"verification": {"status": "verified", "kind": "model_inference",
                                   "evidence_ref": "child:claim", "reason": None}}
    status, kind, ref = ob._parent_status(None, "purpose", candidate,
                                          {"schema": "fixture"}, {"unit_id": "unit-1"})
    assert (status, kind, ref) == ("unknown", None, None)


def test_closed_reference_rejects_unknown_fields():
    row = {"schema": ob.OBSERVATION_REFERENCE_SCHEMA, "observation_id": "obs",
           "unit_id": "unit", "process_generation_id": "pg", "fence_id": "fence",
           "active_claim_ref": "claim", "target_pid": 1, "target_start_ticks": 2,
           "descendant_binding_ref": "descendant", "worker_id": "worker",
           "worker_generation": 1, "grant_id": "grant", "grant_generation": 1,
           "container_id": "container", "instrument_identity_sha256": "a" * 64,
           "observation_content_sha256": "b" * 64, "shutdown_status": "resolved",
           "successor_permitted": True,
           "artifact": {"locator": "x.json", "sha256": "c" * 64, "verified": True}}
    row = ob._hashed(row, "reference_digest")
    row["legacy_verified"] = True
    with pytest.raises(ob.ObservationBindingError):
        ob.LifecycleObservationReference.from_dict(row)
