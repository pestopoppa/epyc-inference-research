"""Original native CPU generations, not a reconstructed window-hint discount."""
from copy import deepcopy
from dataclasses import replace
import json
import os
import time
from unittest import mock

import pytest

from . import claim, scheduling as s, serial_scheduling as ss, serial_run as sr
from .measurement_capture import ArtifactStore, _plain
from .test_cpu_quiet_region_backoff import native
from .test_scheduling import config, proposal, vector


def _selected(*, seed=None):
    cfg = s.SchedulerConfig.from_dict(config(capacity=vector(memory=0)))
    state, selected = s.select_stage(cfg, s.initial_state(cfg, 'cpu-generations'),
        [proposal(claims=vector(fraction=.25, memory=0), seed=seed)], now=0)
    return cfg, state, selected


def _retain(tmp_path, selected, receipt, *, cpu_segments=True, change=None):
    target = {'private': 'original-cpu-generations'}
    store = ArtifactStore(tmp_path / 'held-claim-artifacts')
    try:
        artifact = claim.publish_intervals(store, selected, [receipt], target=target,
                                           cpu_segments=cpu_segments)
        body = _plain(store.read(artifact.locator, artifact.sha256))
        if change:
            change(body)
            artifact = store.write('direct-held-intervals', body)
    finally:
        store.close()
    reference = {'schema': ss.REFERENCE_SCHEMA, 'selection_digest': selected.digest,
                 'evidence': artifact.to_dict()}
    return body, ss.reopen_held_receipts(tmp_path, reference, selection=selected, target=target)


def _yielded(native):
    quiet, _regions = native
    with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as receipt:
        lease = claim.yield_lease(receipt)
        assert receipt.observe()['status'] == 'held'
        for generation in (1, 2):
            assert lease.generation == generation
            lease.release(reason='original-actor')
            # Actual exclusive native owner during the released gap proves this
            # period has neither CPU-region nor SHARED quiet ownership from lease.
            with quiet.gpu_quiet_lock('exclusive', role='private-probe', timeout_s=.1):
                assert quiet.lock_owners() == {os.getpid(): 'exclusive'}
                time.sleep(.03)
            lease.reacquire(reason='original-build', poll_s=1)
        # Corrupting the hint ledger must not alter original segment evidence.
        for row in lease.segments:
            row['acquired_at'] = -999
            row['released_at'] = 999
    return receipt


def test_native_segments_exclude_actual_yielded_gaps_from_cpu_cost_and_forecast(native, tmp_path):
    cfg, state, selected = _selected()
    receipt = _yielded(native)
    body, receipts = _retain(tmp_path, selected, receipt)
    assert body['schema'] == ss.INTERVAL_SCHEMA_V2 and body['phases'] == []
    assert [row['ownership_generation'] for row in body['components']] == [1, 2, 3]
    assert len({row['context_id'] for row in body['components']}) == 3
    assert all(row['released'] and row['open']['status'] == row['close']['status'] == 'held'
               for row in body['components'])
    durations = sum(row.ended_at - row.started_at for row in receipts)
    span = receipts[-1].ended_at - receipts[0].started_at
    assert span - durations >= .05
    assert all(row.schema == s.RECEIPT_SCHEMA_V2 and row.physical_region_fraction == .25
               and not row.gpu_device_ids for row in receipts)
    settled = s.account_stage_components(cfg, state, selected, receipts, outcome='valid_comparison',
                                         cpu_stage_elapsed=body['stage_elapsed'])
    view = s.charge_receipts(settled.receipts)
    assert view.physical_region_seconds == pytest.approx(durations * .25)
    # Wall-time admission/seed/campaign controls retain their original span.
    assert settled.campaign_charged_seconds == pytest.approx(
        body['stage_elapsed']['ended_at'] - body['stage_elapsed']['started_at'])
    history = ss.retain_cost_sample(None, 'cpu', 'v2-scope', selected, receipts)
    assert history['targets']['cpu']['samples'][0]['held_seconds'] == pytest.approx(durations)
    assert ss.retain_cost_sample(history, 'cpu', 'v2-scope', selected, receipts) == history
    assert s.account_stage_components(cfg, settled, selected, receipts,
                                     outcome='valid_comparison', cpu_stage_elapsed=body['stage_elapsed']) == settled


@pytest.mark.parametrize('change,match', [
    (lambda b: b['components'][1].update(ownership_generation=1), 'sequential'),
    (lambda b: b['components'][1].update(released=False), 'release'),
    (lambda b: b['components'][1]['domain'].update(process_start_ticks=0), 'identity'),
    (lambda b: b['components'][1].update(physical_region_fraction=.5), 'ownership'),
    (lambda b: b['components'].append(deepcopy(b['components'][-1])), 'sequential'),
])
def test_new_reader_refuses_changed_original_segments(native, tmp_path, change, match):
    _cfg, _state, selected = _selected()
    receipt = _yielded(native)
    with pytest.raises(ss.SerialSchedulingRefused, match=match):
        _retain(tmp_path, selected, receipt, change=change)


def test_uncertain_original_release_refuses_even_after_native_locks_are_gone(native, tmp_path):
    _cfg, _state, selected = _selected()
    with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as receipt:
        lease = claim.yield_lease(receipt)
        owner = lease._slot[0]
        original_exit = owner.__exit__
        def failing_exit(*error):
            original_exit(*error)
            raise RuntimeError('original exit completion uncertain')
        owner.__exit__ = failing_exit
        with pytest.raises(RuntimeError, match='completion uncertain'):
            lease.release(reason='failed-owner')
        assert native[0].lock_owners() == {}
        lease.reacquire(reason='cleanup', poll_s=1)
    with pytest.raises(claim.ClaimRefused, match='capture unavailable'):
        _retain(tmp_path, selected, receipt)


def test_close_observation_failure_refuses_original_segment(native, tmp_path, monkeypatch):
    _cfg, _state, selected = _selected()
    with claim.hold_cpu('0', capture_segments=True) as receipt:
        lease = claim.yield_lease(receipt)
        def broken():
            raise RuntimeError('original close observation unavailable')
        monkeypatch.setattr(lease._current_capture, '_closing', broken)
    with pytest.raises(claim.ClaimRefused, match='capture unavailable'):
        _retain(tmp_path, selected, receipt)


def test_new_reader_preserves_default_v1_envelope(native, tmp_path):
    _cfg, _state, selected = _selected()
    with claim.hold_cpu('0') as receipt:
        lease = claim.yield_lease(receipt)
        lease.release(reason='actor')
        time.sleep(.03)
        lease.reacquire(reason='build', poll_s=1)
    body, (reopened,) = _retain(tmp_path, selected, receipt, cpu_segments=False)
    assert body['schema'] == ss.INTERVAL_SCHEMA and 'phases' not in body
    assert reopened.schema == s.RECEIPT_SCHEMA and reopened.ownership_generation == 1
    assert reopened.ended_at - reopened.started_at >= .03
    with pytest.raises(claim.ClaimRefused, match='capture unavailable'):
        _retain(tmp_path, selected, receipt)


def test_cpu_forecast_capture_modes_do_not_mix_and_default_v1_scope_is_unchanged():
    _cfg, state, selected = _selected()
    body = {'input_argv': [], 'current_anchor': {'commit': 'a'}, 'cor_anchor': {'commit': 'b'}}
    scope = sr._cost_body_scope(body, selected.proposal, state)
    assert sr._cost_body_scope(dict(body, input_argv=['--cpu-held-intervals-v2', 'off']),
                               selected.proposal, state) == scope
    native_scope = sr._cost_body_scope(dict(body, input_argv=['--cpu-held-intervals-v2', 'on']),
                                      selected.proposal, state)
    assert native_scope != scope
    # Prospective mode follows the next original argv, never the old continuation's mode.
    assert sr._cost_body_scope(body, selected.proposal, state, cpu_capture_mode='on') == native_scope


def test_cancelled_reacquire_adds_no_original_generation(native, tmp_path):
    _cfg, _state, selected = _selected()
    with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as receipt:
        lease = claim.yield_lease(receipt)
        lease.release(reason='actor')
        with native[1].cpu_region_lock('private-peer', {'q0'}, timeout_s=1):
            with pytest.raises(claim.ClaimRefused, match='stop requested'):
                lease.reacquire(reason='timing', should_stop=lambda: True, poll_s=1)
            assert not lease.held and lease.generation == 1
            assert native[0].lock_owners() == {}
        lease.reacquire(reason='cleanup', poll_s=1)
    body, receipts = _retain(tmp_path, selected, receipt)
    assert [row['ownership_generation'] for row in body['components']] == [1, 2]
    assert len(receipts) == 2


def test_completed_v2_scope_already_yielded_settles_only_closed_original_generation(native, tmp_path):
    _cfg, _state, selected = _selected()
    with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as receipt:
        lease = claim.yield_lease(receipt)
        lease.release(reason='actor-stop')
        time.sleep(.03)
    assert receipt._closed['status'] != 'held'
    body, (reopened,) = _retain(tmp_path, selected, receipt)
    assert len(body['components']) == 1 and reopened.ownership_generation == 1
    assert reopened.ended_at < receipt._ended_at - .02
    assert native[0].lock_owners() == {}


def test_cancelled_v2_scope_ends_while_peer_holds_regions_without_reacquiring(native, tmp_path):
    _cfg, _state, selected = _selected()
    peer = native[1].cpu_region_lock('private-peer', {'q0'}, timeout_s=1)
    peer_entered = False
    try:
        with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as receipt:
            lease = claim.yield_lease(receipt)
            lease.release(reason='actor-stop')
            peer.__enter__()
            peer_entered = True
            with pytest.raises(claim.ClaimRefused, match='stop requested'):
                lease.reacquire(reason='cancelled', should_stop=lambda: True, poll_s=1)
        # The completed outer context did not wait for, release or adopt this peer.
        assert peer_entered and not lease.held and lease.generation == 1
        body, receipts = _retain(tmp_path, selected, receipt)
        assert len(body['components']) == len(receipts) == 1
    finally:
        if peer_entered:
            peer.__exit__(None, None, None)


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_capture_baseexception_still_exits_original_native_owner(native, tmp_path, monkeypatch, error_type):
    _cfg, _state, selected = _selected()
    with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as receipt:
        lease = claim.yield_lease(receipt)
        def interrupted():
            raise error_type('original capture interrupted')
        monkeypatch.setattr(lease._current_capture, '_closing', interrupted)
        with pytest.raises(error_type, match='capture interrupted'):
            lease.release(reason='interrupted')
        assert not lease.held and native[0].lock_owners() == {}
        # An independent original provider proves the CPU flock was also released.
        with native[1].cpu_region_lock('private-probe', {'q0'}, timeout_s=.1):
            pass
    with pytest.raises(claim.ClaimRefused, match='capture unavailable'):
        _retain(tmp_path, selected, receipt)


@pytest.mark.parametrize('rejection', ['paths', 'observation', 'interrupt'])
def test_rejected_acquired_native_owner_poisoned_and_released(native, tmp_path, monkeypatch, rejection):
    _cfg, _state, selected = _selected()
    with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as receipt:
        lease = claim.yield_lease(receipt)
        lease.release(reason='actor')
        if rejection == 'paths':
            # Rejection occurs after the real native owner acquired both flocks.
            monkeypatch.setitem(receipt['lock_paths'], 'q0', '/different/original-path')
            error_type = claim.ClaimRefused
        elif rejection == 'observation':
            monkeypatch.setattr(receipt, 'observe', lambda: {'status': 'unavailable', 'error': 'lost'})
            error_type = claim.ClaimRefused
        else:
            def interrupted():
                raise KeyboardInterrupt('native acquisition observation interrupted')
            monkeypatch.setattr(receipt, 'observe', interrupted)
            error_type = KeyboardInterrupt
        with pytest.raises(error_type):
            lease.reacquire(reason='rejected', poll_s=1)
        assert not lease.held and lease.generation == 1 and native[0].lock_owners() == {}
        with native[1].cpu_region_lock('private-probe', {'q0'}, timeout_s=.1):
            pass
        # Restore normal observers before outer close; the latched omission still refuses.
        monkeypatch.undo()
    with pytest.raises(claim.ClaimRefused, match='capture unavailable'):
        _retain(tmp_path, selected, receipt)


@pytest.mark.parametrize("failed_construction", [1, 2])
def test_original_native_entry_fault_cannot_settle_as_preclaim_zero(native, tmp_path, monkeypatch, failed_construction):
    from . import run
    cfg, state, selected = _selected()
    target = {'private': 'original-enter-fault'}
    monkeypatch.setattr(sr, '_selected_identity', lambda argv: target)
    def native_entered():
        run._publish_cpu_native_acquired(tmp_path, selected, target)
    original_constructor = claim.HeldCpuClaim
    constructed = [0]
    def observer_fault(*args, **kwargs):
        constructed[0] += 1
        if constructed[0] == failed_construction:
            raise RuntimeError('original observer construction failed after native entry')
        return original_constructor(*args, **kwargs)
    monkeypatch.setattr(claim, 'HeldCpuClaim', observer_fault)
    with pytest.raises(RuntimeError, match='after native entry'):
        with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True,
                            on_acquired=native_entered):
            pytest.fail('failed observer yielded an original claim')
    assert native[0].lock_owners() == {}
    with native[1].cpu_region_lock('private-probe', {'q0'}, timeout_s=.1):
        pass
    acquired = json.loads((tmp_path / 'loop-claim-acquired.json').read_text())
    assert acquired['native_cpu_provider_entered'] is True
    # A conflicting old zero-route marker cannot override actual native entry.
    run._publish_preclaim_failure(tmp_path, selected, target, RuntimeError('wrong zero route'))
    manifest = ss.SerialSchedulerManifest('cpu-generations', cfg, {'cpu': selected.proposal})
    active = {'scheduler_selection': selected.to_dict(), 'scheduler_selection_sha256': selected.digest}
    with pytest.raises(sr.SerialRefused, match='post-claim'):
        sr._scheduled_failure_account({'scheduler_state': state.to_dict()}, manifest, active,
                                      tmp_path, [])
    assert state.campaign_attempts == 0 and state.issued_selection_digests


def test_durable_v2_refusal_is_authoritative_before_preclaim_route(tmp_path, monkeypatch):
    from . import run
    cfg, state, selected = _selected()
    target = {'private': 'original-refusal'}
    monkeypatch.setattr(sr, '_selected_identity', lambda argv: target)
    run._publish_preclaim_failure(tmp_path, selected, target, RuntimeError('wrong zero route'))
    (tmp_path / 'loop-held-claims-refused.json').write_text(json.dumps({
        'schema': 'epyc.autokernel.cpu_segment_refusal.v1', 'selection_digest': selected.digest,
        'target': target, 'error': 'original native release uncertain'}))
    manifest = ss.SerialSchedulerManifest('cpu-generations', cfg, {'cpu': selected.proposal})
    active = {'scheduler_selection': selected.to_dict(), 'scheduler_selection_sha256': selected.digest}
    with pytest.raises(sr.SerialRefused, match='original CPU segment capture refused'):
        sr._scheduled_failure_account({'scheduler_state': state.to_dict()}, manifest, active,
                                      tmp_path, [])


def test_legacy_v1_and_gpu_components_still_require_contiguous_intervals(native, tmp_path):
    cfg, state, selected = _selected()
    original = _yielded(native)
    _body, receipts = _retain(tmp_path, selected, original)
    v1 = tuple(replace(row, schema=s.RECEIPT_SCHEMA) for row in receipts)
    with pytest.raises(s.SchedulingRefused, match='contiguous'):
        s.account_stage_components(cfg, state, selected, v1, outcome='valid_comparison')


def test_gpu_v2_component_gaps_remain_refused(native, tmp_path):
    _cfg, _state, cpu_selected = _selected()
    original = _yielded(native)
    _body, cpu_receipts = _retain(tmp_path, cpu_selected, original)
    cfg = s.SchedulerConfig.from_dict(config(capacity=vector(gpus=('gpu0',), memory=0)))
    state, selected = s.select_stage(cfg, s.initial_state(cfg, 'gpu-continuity'),
        [proposal(backend='gpu', claims=vector(fraction=.25, gpus=('gpu0',), memory=0))], now=0)
    invalid = tuple(replace(row, proposal_id=selected.proposal.proposal_id, backend='gpu',
                            gpu_device_ids=('gpu0',)) for row in cpu_receipts)
    with pytest.raises(s.SchedulingRefused, match='contiguous'):
        s.account_stage_components(cfg, state, selected, invalid, outcome='valid_comparison')


def test_before_native_preflight_failure_remains_genuine_preclaim(native, tmp_path, monkeypatch):
    from . import run
    from src.runtime import region_lock_cli
    cfg, state, selected = _selected()
    target = {'private': 'before-native-refusal'}
    monkeypatch.setattr(sr, '_selected_identity', lambda argv: target)
    monkeypatch.setattr(region_lock_cli, '_preflight', lambda **kwargs: 'private before-native refusal')
    with pytest.raises(claim.ClaimRefused, match='before-native refusal') as failed:
        with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True,
                            on_acquired=lambda: pytest.fail('native owner entered')):
            pytest.fail('preflight refusal yielded a CPU context')
    run._publish_preclaim_failure(tmp_path, selected, target, failed.value)
    assert not (tmp_path / 'loop-claim-acquired.json').exists()
    assert not (tmp_path / 'loop-held-claims.json').exists()
    assert native[0].lock_owners() == {}
    manifest = ss.SerialSchedulerManifest('cpu-generations', cfg, {'cpu': selected.proposal})
    active = {'scheduler_selection': selected.to_dict(), 'scheduler_selection_sha256': selected.digest}
    settled = sr._scheduled_failure_account({'scheduler_state': state.to_dict()}, manifest, active,
                                            tmp_path, [])
    # A native preflight refusal retires only the issued selection. Actual held
    # receipts are required to charge attempts, time, seed budgets or debt.
    assert state.issued_selection_digests == (selected.digest,)
    assert settled == replace(state, issued_selection_digests=())
    with pytest.raises(s.SchedulingRefused, match='not an issued selection'):
        sr._scheduled_failure_account({'scheduler_state': settled.to_dict()}, manifest, active,
                                      tmp_path, [])


@pytest.mark.parametrize('seed', [None, 'native-seed'])
def test_original_scope_wall_budget_includes_trailing_yield_without_cpu_cost(native, tmp_path, monkeypatch, seed):
    from types import SimpleNamespace
    cfg, state, selected = _selected(seed=seed)
    cfg = s.SchedulerConfig.from_dict(config(capacity=vector(memory=0), max_stage_seconds=900))
    state, selected = s.select_stage(cfg, s.initial_state(cfg, 'cpu-generations'),
        [proposal(claims=vector(fraction=.25, memory=0), seed=seed)], now=0)
    original_time = claim.time
    now = [0.0]
    # Controlled original capture clock; real private native flock ownership.
    clock = SimpleNamespace(**{key: getattr(original_time, key) for key in dir(original_time)
                               if not key.startswith('__')})
    clock.monotonic = lambda: now[0]
    monkeypatch.setattr(claim, 'time', clock)
    with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as receipt:
        now[0] = 2.0
        claim.yield_lease(receipt).release(reason='actor-stop')
        now[0] = 1002.0
    body, (held,) = _retain(tmp_path, selected, receipt)
    assert held.ended_at - held.started_at == 2.0
    elapsed = body['stage_elapsed']
    assert elapsed['ended_at'] - elapsed['started_at'] == 1002.0
    settled = s.account_stage_components(cfg, state, selected, (held,), outcome='failed',
                                         cpu_stage_elapsed=elapsed)
    assert s.charge_receipts(settled.receipts).physical_region_seconds == .5
    assert settled.campaign_charged_seconds == 1002.0
    assert settled.successor_fences  # Original failed-stage D fence remains armed.
    assert s.recover_abstained_overrun_fence(cfg, settled) == settled
    # A separate v1 abstained overrun must not clear this CPU-v2 wall fence.
    legacy = replace(held, schema=s.RECEIPT_SCHEMA, receipt_id='legacy-abstained',
                     started_at=2000., ended_at=3001.)
    legacy_record = s.AccountedReceipt(receipt_id=legacy.receipt_id, receipt_digest=legacy.digest,
                                       selection_digest='f' * 64, outcome='abstained')
    mixed = replace(settled, receipts=(*settled.receipts, legacy),
                    accounted_receipts=(*settled.accounted_receipts, legacy_record))
    assert s.recover_abstained_overrun_fence(cfg, mixed) == mixed
    legacy_only = replace(settled, receipts=(legacy,), accounted_receipts=(legacy_record,))
    assert s.recover_abstained_overrun_fence(cfg, legacy_only).successor_fences == ()
    if seed:
        assert settled.seed_accounts[0].charged_seconds == 1002.0
        assert not settled.seed_accounts[0].boosted
    history = ss.retain_cost_sample(None, 'cpu', 'native-scope', selected, (held,))
    assert history['targets']['cpu']['samples'][0]['held_seconds'] == 2.0
    assert ss.duration_forecast(history, 'cpu', 'native-scope', proposal=selected.proposal,
                                max_stage_seconds=10)['estimated_duration_seconds'] == 2.0
    changed = dict(elapsed, ended_at=1003.0)
    with pytest.raises(s.SchedulingRefused, match='original receipt binding'):
        s.account_stage_components(cfg, settled, selected, (held,), outcome='failed',
                                   cpu_stage_elapsed=changed)


@pytest.mark.parametrize('change', [
    lambda b: b.pop('stage_elapsed'),
    lambda b: b['stage_elapsed'].update(completed=False),
    lambda b: b['stage_elapsed'].update(ended_at=0),
    lambda b: b['stage_elapsed']['domain'].update(process_start_ticks=0),
    lambda b: b['stage_elapsed'].update(physical_region_fraction=.25),
])
def test_reader_refuses_unproved_or_claim_bearing_stage_clock(native, tmp_path, change):
    _cfg, _state, selected = _selected()
    original = _yielded(native)
    with pytest.raises(ss.SerialSchedulingRefused):
        _retain(tmp_path, selected, original, change=change)


def test_exited_child_cpu_v2_runtime_recovery_joins_original_outer_holder(native, tmp_path, monkeypatch):
    import subprocess
    import sys
    from . import runtime_recovery as recovery
    _cfg, _state, selected = _selected()
    target = {'scope': 'cpu_serving_selected_workload', 'selected_id': 'cpu',
              'private': 'exited-original-child'}
    monkeypatch.setattr(sr, '_selected_identity', lambda argv: target)
    (tmp_path / 'selection.json').write_text(json.dumps(selected.to_dict()))
    (tmp_path / 'target.json').write_text(json.dumps(target))
    child_code = r'''
import json, sys, time
from pathlib import Path
from scripts.kernel_rnd.autokernel.loop import claim, scheduling, status, runtime_recovery as recovery
from scripts.kernel_rnd.autokernel.loop.measurement_capture import ArtifactStore
root = Path(sys.argv[1]); batch = root / 'batch'; batch.mkdir()
claim._ensure_orchestrator_importable()
from src.runtime import region_lock_cli
region_lock_cli._preflight = lambda **kwargs: None
selected = scheduling.Selection.from_dict(json.loads((root / 'selection.json').read_text()))
target = json.loads((root / 'target.json').read_text())
with claim.hold_cpu('0', gpu_quiet=True, capture_segments=True) as outer:
    original = recovery.holder_identity(outer)
    lease = claim.yield_lease(outer)
    lease.release(reason='original-actor')
    time.sleep(.02)
    lease.reacquire(reason='original-runtime', poll_s=1)
store = ArtifactStore(batch / 'held-claim-artifacts')
try:
    artifact = claim.publish_intervals(store, selected, [outer], target=target, cpu_segments=True)
finally:
    store.close()
held = {'schema': 'epyc.autokernel.direct_held_reference.v1', 'selection_digest': selected.digest,
        'evidence': artifact.to_dict()}
status.write_json(batch, 'loop-held-claims.json', held)
(root / 'store').mkdir()
runtime = ArtifactStore(root / 'store' / 'runtime-preparation')
try:
    interrupted = runtime.write('original-runtime-interruption', {
        'schema': 'epyc.autokernel.direct_runtime_interruption.v1', 'holder': original,
        'reason': 'between_launch_budget'})
finally:
    runtime.close()
status.write_json(batch, 'loop-runtime-interruption.json', interrupted.to_dict())
(root / 'original-holder.json').write_text(json.dumps(original))
'''
    child = subprocess.Popen([sys.executable, '-c', child_code, str(tmp_path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 0, (stdout, stderr)
    original = json.loads((tmp_path / 'original-holder.json').read_text())
    assert original['domain']['pid'] == child.pid
    directory = tmp_path / 'batch'
    argv = ['--target-id', 'cpu', '--store', str(tmp_path / 'store'), '--out', str(directory)]
    domain = original['domain']
    active = {'target_index': 0, 'selected_id': 'cpu', 'store': str(tmp_path / 'store'),
              'batch_dir': str(directory), 'input_argv_sha256': sr._digest(argv), 'pid': child.pid,
              'process_identity': {'pid': child.pid, 'start_ticks': domain['process_start_ticks'],
                                   'boot_id': domain['boot_id']},
              'scheduler_selection': selected.to_dict(), 'scheduler_selection_sha256': selected.digest}
    reference = recovery.retain(directory, active, argv)
    reopened = recovery.reopen(reference, current_argv=argv)
    assert {'context_id': reopened['context_id'], 'domain': reopened['domain']} == original
    assert 'physical_region_fraction' not in reopened and 'physical_claim_ids' not in reopened
    with claim.hold_cpu('0', gpu_quiet=True) as new:
        replaced = recovery.replacement(reference=reference, original_holder=original, new_holder=new)
        assert replaced['holder'] == recovery.holder_identity(new)
        assert replaced['holder'] != original
    # A valid evidence file never overrides an authoritative failed-capture marker.
    (directory / 'loop-held-claims-refused.json').write_text(json.dumps({
        'schema': 'epyc.autokernel.cpu_segment_refusal.v1', 'selection_digest': selected.digest,
        'target': target, 'error': 'original capture uncertain'}))
    with pytest.raises(sr.SerialRefused, match='original CPU segment capture refused'):
        recovery.reopen(reference)


def test_cpu_segment_writer_refuses_gpu_phases_without_writing(tmp_path):
    _cfg, _state, selected = _selected()
    receipt = claim.HeldCpuClaim({"device_id": "cpu"}, [])
    store = ArtifactStore(tmp_path / "held-claim-artifacts")
    try:
        with mock.patch.object(store, "write") as write:
            with pytest.raises(claim.ClaimRefused, match="cannot share a bundle"):
                claim.publish_intervals(store, selected, [receipt], target={"cpu": True},
                                        phases=[], cpu_segments=True)
            write.assert_not_called()
    finally:
        store.close()
