"""Native private-path owners prove waiting retries release quiet, not just metadata."""
import inspect
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from . import claim, scheduling, serial_scheduling as ss
from .measurement_capture import ArtifactStore
from .test_scheduling import config, proposal, vector


@pytest.fixture
def native(tmp_path, monkeypatch):
    claim._ensure_orchestrator_importable()
    import importlib
    quiet = importlib.import_module('src.runtime.gpu_quiet_lock')
    regions = importlib.import_module('src.runtime.cpu_region_lock')
    cli = importlib.import_module('src.runtime.region_lock_cli')
    # Fresh worker API check: fail rather than silently use an older unbounded owner.
    assert 'region_attempt_s' in inspect.signature(quiet.gpu_quiet_then_regions).parameters
    assert Path(quiet.__file__).resolve() == Path('/mnt/raid0/llm/epyc-orchestrator/src/runtime/gpu_quiet_lock.py')
    monkeypatch.setenv('ORCHESTRATOR_TMP_DIR', str(tmp_path / 'private-native-locks'))
    monkeypatch.setenv('ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT', '1')
    monkeypatch.setenv('EPYC_LOCK_FIFO', '0')
    monkeypatch.setattr(cli, '_preflight', lambda **kwargs: None)
    assert quiet.gpu_quiet_lock_path().parent == tmp_path / 'private-native-locks'
    # A CPU-only claim or preflight refusal need not create the quiet file.
    # Establish readable private custody through the ORIGINAL native provider;
    # None stays an unknown observation, never an alias for no live owners.
    with quiet.gpu_quiet_lock('exclusive', role='private-fixture-custody', timeout_s=1):
        assert quiet.lock_owners() == {os.getpid(): 'exclusive'}
    assert quiet.lock_owners() == {}
    return quiet, regions


def _probe_on_backoff(native, monkeypatch):
    quiet, regions = native
    backoffs = []
    original_time = quiet.time

    def sleep(seconds):
        # Only the combined provider's own backoff is intercepted. Actual native
        # CPU flock waits keep their real clock/sleep and deadline behavior.
        assert seconds == quiet._BACKOFF_S == 0.2
        with quiet.gpu_quiet_lock('exclusive', role='private-gpu-probe', timeout_s=.1):
            assert quiet.lock_owners() == {os.getpid(): 'exclusive'}
            # The CPU peer remains an original native owner throughout this probe.
            path = regions.global_region_lock_path('q0')
            assert str(path).startswith(str(quiet.gpu_quiet_lock_path().parent))
            assert claim.observe_gpu_quiet(path)['owners'][0]['pid'] == os.getpid()
            backoffs.append(seconds)
        original_time.sleep(seconds)

    proxy = SimpleNamespace(**{key: getattr(original_time, key) for key in dir(original_time)
                               if not key.startswith('__')})
    proxy.sleep = sleep
    monkeypatch.setattr(quiet, 'time', proxy)
    return backoffs


def test_native_cpu_wait_releases_quiet_between_busy_region_attempts(native, monkeypatch):
    quiet, regions = native
    backoffs = _probe_on_backoff(native, monkeypatch)
    with regions.cpu_region_lock('private-peer', {'q0'}, timeout_s=1):
        with pytest.raises(regions.CpuRegionLockTimeout):
            with claim.hold_cpu('0', gpu_quiet=True):
                pytest.fail('CPU owner entered over the occupied original region')
        assert backoffs
        assert quiet.lock_owners() == {}


def test_native_old_full_budget_attempt_has_no_quiet_free_backoff(native, monkeypatch):
    quiet, regions = native
    backoffs = _probe_on_backoff(native, monkeypatch)
    with regions.cpu_region_lock('private-peer', {'q0'}, timeout_s=1):
        with pytest.raises(regions.CpuRegionLockTimeout):
            with quiet.gpu_quiet_then_regions('private-old-control', {'q0'}, gpu_quiet='shared',
                                              timeout_s=1, region_attempt_s=5):
                pytest.fail('old control entered over the occupied original region')
        assert backoffs == []
        assert quiet.lock_owners() == {}


def test_native_stop_keeps_yielded_cpu_and_quiet_released_then_v1_reopens(native, tmp_path):
    quiet, regions = native
    with claim.hold_cpu('0', gpu_quiet=True) as receipt:
        lease = claim.yield_lease(receipt)
        assert receipt.observe()['status'] == 'held'
        assert quiet.lock_owners() == {os.getpid(): 'shared'}
        lease.release(reason='actor:planner')
        with regions.cpu_region_lock('private-peer', {'q0'}, timeout_s=1):
            with pytest.raises(claim.ClaimRefused, match='stop requested'):
                lease.reacquire(reason='timing', should_stop=lambda: True, poll_s=1)
            assert not lease.held
            assert quiet.lock_owners() == {}
        lease.reacquire(reason='timing', poll_s=1)
        assert receipt.observe()['status'] == 'held'
        assert quiet.lock_owners() == {os.getpid(): 'shared'}
        assert lease.generation == 2
    assert quiet.lock_owners() == {}
    cfg = scheduling.SchedulerConfig.from_dict(config())
    _state, selected = scheduling.select_stage(cfg, scheduling.initial_state(cfg, 'private'),
        [proposal(claims=vector(fraction=.25, memory=0))], now=0)
    store = ArtifactStore(tmp_path / 'held-claim-artifacts')
    target = {'private': 'original-cpu-v1'}
    try:
        artifact = claim.publish_intervals(store, selected, [receipt], target=target)
        body = store.read(artifact.locator, artifact.sha256)
        assert body['schema'] == ss.INTERVAL_SCHEMA == 'epyc.autokernel.direct_held_intervals.v1'
    finally:
        store.close()
    reopened, = ss.reopen_held_receipts(tmp_path,
        {'schema': ss.REFERENCE_SCHEMA, 'selection_digest': selected.digest,
         'evidence': artifact.to_dict()}, selection=selected, target=target)
    assert reopened.physical_region_fraction == .25 and reopened.gpu_device_ids == ()
    assert reopened.schema == scheduling.RECEIPT_SCHEMA
