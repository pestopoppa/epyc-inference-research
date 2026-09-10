"""Actual temporary flock lifecycle; no physical region or model execution."""
from contextlib import contextmanager
import fcntl
import os
import sys
from types import ModuleType

import pytest

from . import claim, measurement_capture as mc, observation_binding as ob


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("observation_fails", [False, True])
def test_existing_cpu_claim_context_observes_same_actual_locks_and_closes(
        tmp_path, monkeypatch, failed, observation_fails):
    # Fixture provider acquires real temporary kernel flocks. No physical region,
    # orchestrator telemetry or host preflight is changed by this test.
    modules = {name: ModuleType(name) for name in (
        "src.runtime.cpu_region_lock", "src.runtime.instance_topology", "src.runtime.region_lock_cli")}
    paths = [tmp_path / "role.lock", tmp_path / "global.lock"]
    acquisitions = []

    @contextmanager
    def held(*args, **kwargs):
        acquisitions.append(True)
        handles = [path.open("a") for path in paths]
        try:
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield {"fixture-region": paths[0]}
        finally:
            for handle in handles:
                handle.close()

    modules["src.runtime.cpu_region_lock"].cpu_region_lock = held
    modules["src.runtime.cpu_region_lock"].global_region_lock_path = lambda region: paths[1]
    modules["src.runtime.instance_topology"].cpu_list_to_regions = lambda cpus: ["fixture-region"]
    modules["src.runtime.instance_topology"].ATOMIC_REGIONS = ("fixture-region",)
    modules["src.runtime.region_lock_cli"]._preflight = lambda **kwargs: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    class OriginalFailure(Exception):
        pass

    try:
        with claim.hold_cpu("0") as receipt:
            original = dict(receipt)
            opened = receipt.observe()
            with pytest.raises(claim.ClaimRefused, match="owning release"):
                receipt.retained_interval()
            # An actual second temporary flock, not a GPU or production lock.
            with claim.hold(tmp_path / "fixture-device.lock", device_id="fixture-device") as gpu:
                assert gpu.observe()["status"] == "held"
            closed = receipt.observe()
            assert opened["status"] == closed["status"] == "held"
            assert opened["owner_pid"] == closed["owner_pid"] == os.getpid()
            assert [(row["device"], row["inode"]) for row in opened["locks"]] == [
                (row["device"], row["inode"]) for row in closed["locks"]]
            assert dict(receipt) == original and len(acquisitions) == 1
            if observation_fails:
                def broken():
                    raise RuntimeError("original close observer failed")
                monkeypatch.setattr(receipt, "_closing", broken)
            if failed:
                raise OriginalFailure("retained original failure")
    except OriginalFailure as exc:
        assert failed and str(exc) == "retained original failure"
    assert receipt.observe()["status"] == "unavailable"
    assert "no longer active" in receipt.observe()["error"]
    cpu_row, gpu_row = receipt.retained_interval(), gpu.retained_interval()
    assert cpu_row["started_at"] < gpu_row["started_at"] < gpu_row["ended_at"] < cpu_row["ended_at"]
    assert cpu_row["domain"] == gpu_row["domain"]
    assert cpu_row["domain"]["kind"] == "direct_loop"
    assert cpu_row["domain"]["process_start_ticks"] > 0
    assert cpu_row["physical_region_fraction"] == 1.0
    assert cpu_row["affinity_cores"] == ["0"]
    assert cpu_row["released"] and cpu_row["open"]["status"] == "held"
    assert cpu_row["close"]["status"] == ("unavailable" if observation_fails else "held")
    if observation_fails:
        assert "original close observer failed" in cpu_row["close"]["error"]
    assert cpu_row["ownership_generation"] == cpu_row["allocation_generation"] == 1
    # No new incarnation is reconstructed from a prior receipt.
    with claim.hold_cpu("0") as later:
        pass
    assert later.retained_interval()["context_id"] != cpu_row["context_id"]
    from . import scheduling as s
    from .test_scheduling import config, proposal
    cfg = s.SchedulerConfig.from_dict(config())
    _state, selected = s.select_stage(cfg, s.initial_state(cfg, "original"), [proposal()], now=0)
    store = mc.ArtifactStore(tmp_path / "held-artifacts")
    try:
        reference = claim.publish_intervals(store, selected, [receipt, gpu], target={"fixture": True})
        retained = store.read(reference.locator, reference.sha256)
        assert ob._plain(retained)["components"] == [cpu_row, gpu_row]
        assert retained["selection_digest"] == selected.digest
        with pytest.raises(claim.ClaimRefused, match="duplicate"):
            claim.publish_intervals(store, selected, [receipt, receipt], target={"fixture": True})
    finally:
        store.close()


def test_gpu_observer_constructor_failure_releases_acquired_lock(tmp_path, monkeypatch):
    path = tmp_path / "fixture-device.lock"
    original = claim.HeldCpuClaim

    def broken(*args, **kwargs):
        raise RuntimeError("original constructor failed")

    monkeypatch.setattr(claim, "HeldCpuClaim", broken)
    with pytest.raises(RuntimeError, match="original constructor failed"):
        with claim.hold(path, device_id="fixture-device"):
            pytest.fail("constructor failure yielded a claim")
    monkeypatch.setattr(claim, "HeldCpuClaim", original)
    with claim.hold(path, device_id="fixture-device") as recovered:
        assert recovered.observe()["status"] == "held"


def test_gpu_close_observer_failure_is_unavailable_and_releases(tmp_path, monkeypatch):
    path = tmp_path / "fixture-device.lock"
    with claim.hold(path, device_id="fixture-device") as original:
        def broken():
            raise RuntimeError("original close observer failed")
        monkeypatch.setattr(original, "_closing", broken)
    assert original.retained_interval()["close"]["status"] == "unavailable"
    with claim.hold(path, device_id="fixture-device") as recovered:
        assert recovered.observe()["status"] == "held"
