"""Real scheduler/child/IPC/Journal preparation; synthetic observation boundaries.

The existing mock resource provider grants no real hardware authority. The tiny
owned child is not a model, and unknown scientific witnesses stay unknown.
"""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest

from . import driver_execution as execution, experiment_plan as ep
from . import campaign_control, scheduling
from . import measurement_capture as mc, observation_binding as ob, planned_serving as ps
from . import serving_preparation as prep, unified_driver as driver, unified_worker as worker
from . import worker_lifecycle as lifecycle
from . import lifecycle_observation as observation
from .test_driver_execution import _owned_stack
from .test_driver_execution import _observation_configuration as _base_observation_configuration
from .test_serving_preparation import neutral_copy, request


def _observation_configuration(prepared):
    configured = _base_observation_configuration(prepared)
    required = observation.required_sample_capacity(
        max_duration_s=prepared.max_stage_seconds + prepared.teardown_seconds,
        cadence_s=configured.cadence_s, nonperiodic_samples=len(observation.PHASES) + 2)
    result = replace(configured, budgets={**ob._plain(configured.budgets),
                                         "max_samples": required})
    assert ob.validate_planned_sample_capacity(result,
        max_stage_seconds=prepared.max_stage_seconds,
        teardown_seconds=prepared.teardown_seconds) == required
    return result


MEASURE_SOURCE = r'''
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from autokernel.loop import worker_lifecycle as wl

def observed_measure(template, build_dir, port, **kwargs):
    session = kwargs['observation_session']
    server = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        cpus = [int(value) for value in os.environ['FIXTURE_CPUS'].split(',')]
        os.sched_setaffinity(server.pid, cpus)
        identity = wl.process_identity(server.pid)
        with Path(os.environ['FIXTURE_PID_LOG']).open('a') as stream:
            stream.write(f'{identity.pid} {identity.start_ticks}\n')
        root = Path(os.environ['FIXTURE_PROC']) / str(server.pid)
        (root / 'fd').mkdir(parents=True)
        fields = ['0'] * 40
        fields[0], fields[11], fields[19], fields[36] = 'S', '1', str(identity.start_ticks), '0'
        (root / 'stat').write_text(f'{server.pid} (synthetic-calibration) ' + ' '.join(fields) + '\n')
        nodes = '0' if os.environ.get('FIXTURE_CONTAMINATED') == 'yes' else '0-1'
        (root / 'status').write_text('Cpus_allowed_list:\t' + ','.join(map(str, cpus))
                                   + '\nMems_allowed_list:\t' + nodes + '\n')
        container = session.context['worker_binding']['container_identity']['path']
        captured = session.context['worker_binding']['container_identity']
        info = Path(container).stat()
        assert (captured['dev'], captured['ino'], captured['mode']) == (
            info.st_dev, info.st_ino, info.st_mode), 'fixture container identity drift'
        (root / 'cgroup').write_text(f'0::{container}\n')
        (root / 'numa_maps').write_text('00400000 default kernelpagesize_kB=2048 N0=2\n')
        (root / 'smaps_rollup').write_text('Rss: 40 kB\nAnonHugePages: 4 kB\nShmemPmdMapped: 0 kB\nFilePmdMapped: 8 kB\n')
        (root / 'maps').write_text('')
        session.start()
        session.phase('load')
        session.attach_target(server.pid)
        for phase in ('placement', 'health', 'warmup', 'measurement'):
            session.phase(phase)
        time.sleep(0.04)  # Explicit synthetic interval, sampled DURING the fixture.
        session.checkpoint('measurement_end')
        prompt_id, body = kwargs['frozen_requests'][0]
        kwargs['observation'].append({
            'schema': 'epyc.autokernel.serving_observation.v1', 'process_pid': server.pid,
            'requests': [{'phase': 'measurement', 'slot_index': 0, 'prompt_id': prompt_id,
                'request_sha256': hashlib.sha256(body).hexdigest(),
                'predicted_n': template.n_predict, 'predicted_per_second': 10.0,
                'terminal': True, 'error': None}],
            'residency': {'status': 'synthetic'}, 'teardown': 'terminated', 'failure': None})
        return 10.0
    finally:
        session.phase('teardown')
        server.terminate()
        server.wait(timeout=2)
        session.finish()
'''


def install_observation_fixture(tmp_path, monkeypatch, *, contaminate_initial=False,
                                installed_measure=False):
    from .test_worker_lifecycle import MockOwnedContainer

    def original_container_identity(self):
        info = self.path.stat()
        return {'path': str(self.path), 'dev': info.st_dev, 'ino': info.st_ino,
                'uid': info.st_uid, 'nlink': info.st_nlink, 'mode': info.st_mode}

    # Match the observer's actual stat identity contract, including file-type
    # bits. The inherited mock masks these bits for non-observation tests.
    monkeypatch.setattr(MockOwnedContainer, 'identity', original_container_identity)
    measure_path = tmp_path / 'preparation_fixture.py'
    measure_path.write_text(MEASURE_SOURCE)
    spec = importlib.util.spec_from_file_location('preparation_fixture', measure_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'preparation_fixture', module)
    spec.loader.exec_module(module)
    fixture_cpus = sorted(os.sched_getaffinity(0))[:4]
    source_root = Path(worker.__file__).resolve().parents[2]
    pid_log = tmp_path / 'owned-calibration-pids.txt'
    child_path = tmp_path / 'unified_worker.py'
    child_path.write_text(f'''
import argparse, os, sys, types
from pathlib import Path
sys.path.insert(0, {str(source_root)!r})
yaml = types.ModuleType('yaml')
yaml.YAMLError = ValueError
yaml.safe_load = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError())
sys.modules['yaml'] = yaml
from autokernel.loop import unified_worker as uw, lifecycle_observation as lo
sys.path.insert(0, {str(tmp_path)!r})
from preparation_fixture import observed_measure
if {installed_measure!r}:
    from autokernel.loop import serving
    serving._measure_once = observed_measure
parser = argparse.ArgumentParser()
for key in ('start', 'control', 'result'):
    parser.add_argument('--' + key + '-fd', type=int, required=True)
args = parser.parse_args()
root = Path({str(tmp_path)!r}) / 'native-probe'
proc, cpu, cgroups = root / 'proc', root / 'cpu', root / 'cgroup'
proc.mkdir(parents=True, exist_ok=True)
(cgroups / 'owned').mkdir(parents=True, exist_ok=True)
for index, number in enumerate({fixture_cpus!r}):
    item = cpu / f'cpu{{number}}'
    (item / 'topology').mkdir(parents=True, exist_ok=True)
    (item / 'topology' / 'thread_siblings_list').write_text(str(number))
    (item / f'node{{index % 2}}').mkdir(exist_ok=True)
(root / 'boot').write_text(Path('/proc/sys/kernel/random/boot_id').read_text())
(root / 'pressure').write_text('some avg10=0 avg60=0 avg300=0 total=10\\nfull avg10=0 avg60=0 avg300=0 total=2\\n')
(root / 'thp').write_text('always [madvise] never\\n')
(proc / 'meminfo').write_text('MemAvailable: 1000 kB\\nSwapFree: 500 kB\\nSwapTotal: 500 kB\\n')
(proc / 'vmstat').write_text('pswpin 0\\npswpout 0\\n')
(root / 'global-vram').write_text('0\\n')
probe = lo.FilesystemProbe(proc_root=proc, sysfs_cpu_root=cpu,
    boot_id_path=root / 'boot', memory_psi_path=root / 'pressure',
    thp_enabled_path=root / 'thp', cgroup_root=Path('/'),
    global_vram_paths={{'gpu0': root / 'global-vram'}})
os.environ['FIXTURE_PROC'] = str(proc)
os.environ['FIXTURE_PID_LOG'] = {str(pid_log)!r}
os.environ['FIXTURE_CPUS'] = {','.join(map(str, fixture_cpus))!r}
os.environ['FIXTURE_CONTAMINATED'] = Path({str(tmp_path / 'synthetic-placement-state')!r}).read_text()
uw.run_from_fds(start_fd=args.start_fd, control_fd=args.control_fd, result_fd=args.result_fd,
    _test_membership_probe=lambda _start: None, _test_measure=observed_measure,
    _test_observation_probe=probe)
''')
    bootstrap = tmp_path / 'worker_bootstrap.py'
    bootstrap.write_bytes(Path(lifecycle.__file__).with_name('worker_bootstrap.py').read_bytes())
    original_open = worker.PlannedWorkerInvocation.open

    def open_fixture(cls, prepared, authority):
        invocation = original_open(prepared, authority)
        invocation.argv = (*invocation.argv[:3], str(child_path), *invocation.argv[4:])
        original_request = prep.CalibrationPreparationDispatch.from_dict(prepared.dispatch).request
        contaminated = (contaminate_initial and original_request.attempt_ordinal == 0
                        and original_request.block_membership[0]['block_index'] == 0)
        (tmp_path / 'synthetic-placement-state').write_text('yes' if contaminated else 'no')
        return invocation

    monkeypatch.setattr(worker.PlannedWorkerInvocation, 'open', classmethod(open_fixture))
    original_popen = lifecycle.subprocess.Popen

    def popen_fixture(argv, *args, **kwargs):
        if isinstance(argv, tuple) and len(argv) > 4 and Path(argv[4]).name == 'worker_bootstrap.py':
            argv = (*argv[:4], str(bootstrap), *argv[5:])
        return original_popen(argv, *args, **kwargs)

    monkeypatch.setattr(lifecycle.subprocess, 'Popen', popen_fixture)
    return module.observed_measure, pid_log


def make_requests(tmp_path, original_driver, engine, measurement, *, max_attempts=1,
                  kinds=('aa', 'neutral')):
    target = next(iter(original_driver.runtime_anchors.recipes))
    anchor = original_driver.runtime_anchors.recipes[target]
    original = request(tmp_path, kind='neutral')
    stats = replace(original.declaration.statistics, commitment=replace(
        original.declaration.statistics.commitment, campaign_id=original_driver.resolved.campaign_id))
    declared = replace(original.declaration, campaign_id=original_driver.resolved.campaign_id,
        target_revision=target, statistics=stats,
        aa_pair=prep.PreparationArmPair('aa', anchor, anchor, None),
        neutral_pair=neutral_copy(anchor) if 'neutral' in kinds else None,
        retry_policy=prep.PreparationRetryPolicy(max_attempts, ('failed', 'contaminated')),
        resources=engine.config.capacity, frame={**dict(original.declaration.frame),
            'quant': original_driver.profiles[target].quant})
    store = mc.ArtifactStore(original_driver.controller.store / 'unified-native-artifacts')
    try:
        loaded = ob.seal_loaded_instrument(store=store, measurement_callable=measurement,
            fence_clock=time.monotonic, serving_timer=time.time).to_dict()
    finally:
        store.close()
    requests = []
    positions = range(2) if max_attempts > 1 else (0,)
    for kind, block, attempt in ((kind, block, attempt) for kind in kinds
                                for block in positions for attempt in range(max_attempts)):
        prototype = request(tmp_path, kind=kind, block_start=block,
            blocks=1 if max_attempts > 1 else 2, attempt=attempt, max_attempts=max_attempts)
        pair = declared.aa_pair if kind == 'aa' else declared.neutral_pair
        plan = ep.ExperimentPlan.from_dict({**prototype.plan.to_dict(),
            'campaign_id': declared.campaign_id, 'target_revision': target, 'loaded_instrument': loaded,
            'anchor_identity': ps.arm_identity(pair.anchor.template, pair.anchor, loaded_instrument=loaded),
            'candidate_identity': ps.arm_identity(pair.candidate.template, pair.candidate, loaded_instrument=loaded)})
        stage = replace(prototype.stage_proposal, target_revision=target, eligibility_ref=declared.digest,
            frontier_id=target, production_frontier=True, estimated_claims=declared.resources, full_region=True)
        requests.append(replace(prototype, declaration=declared, plan=plan, stage_proposal=stage))
    return tuple(requests)


def test_actual_public_v3_child_collection_settlement_and_numeric_pool(tmp_path, monkeypatch):
    original, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    monkeypatch.undo()  # No injected worker completion or result authority.
    # The inherited fixture provider names search receipts; this fixture owns
    # only calibration stages and records that actual selected class.
    provider = controller._lifecycle_provider
    original_close = provider.close_held_receipt

    def close_calibration(**kwargs):
        trusted = original_close(**kwargs)
        return replace(trusted, receipt=replace(trusted.receipt, stage_class='calibration',
                                                memory_reservation_bytes=0))

    monkeypatch.setattr(provider, 'close_held_receipt', close_calibration)
    executor = None
    pid_log = None
    try:
        measurement, pid_log = install_observation_fixture(tmp_path, monkeypatch)
        requests = make_requests(tmp_path, original, engine, measurement)
        instance = driver.UnifiedCampaignDriver(resolved_campaign=original.resolved,
            controller=controller, scheduler_engine=engine, profiles=original.profiles,
            evidence_index=original.evidence, runtime_anchors=original.runtime_anchors,
            runtime_dimensions={}, experiment_plans={}, profile_requests={}, actor_identities={},
            native_artifact_sink_ref=original.sink_ref, calibration_requests=requests,
            execution_inputs={item.declaration.target_revision: driver.ExecutionInput(
                item.declaration.target_revision, item.prompts, item.declaration.max_stage_seconds,
                item.declaration.teardown_seconds, item.plan.loaded_instrument["identity_sha256"])
                for item in requests},
            executable_work_kinds={driver.CALIBRATION_WORK_KIND})
        first = instance.tick(now=1.0)
        prepared = instance.materialize_calibration(first)
        assert prepared.schema == worker.PREPARED_SCHEMA_V3
        executor = execution.UnifiedDriverExecution(driver=instance, controller=controller,
            observation_configuration=_observation_configuration(prepared))
        receipts = []
        for ordinal in range(2):
            outcome = first if ordinal == 0 else instance.tick(now=2.0)
            assert outcome.status == 'intent_recorded', outcome.to_dict()
            if ordinal == 1:
                def refuse_settlement(_request):
                    raise OSError('explicit fixture interruption before Journal settlement')

                with monkeypatch.context() as interrupted:
                    interrupted.setattr(controller, 'unified_driver_settle', refuse_settlement)
                    with pytest.raises(execution.DriverExecutionUncertain, match='settlement append/reply'):
                        executor.execute(outcome)
                before_retry_pids = pid_log.read_text()
                # Both original CAS chunks now exist, but only the first one
                # was settled. Presence must not license a complete solve.
                raw_chunks = [json.loads(path.read_bytes())
                              for path in (controller.store / 'unified-native-artifacts').glob('*.json')]
                assert sum(row.get('schema') == 'epyc.autokernel.collected_calibration_chunk.v1'
                           for row in raw_chunks) == 2
                assert instance.preparation_owner.solve_collected(requests[0].declaration) is None
                assert len(instance.preparation_owner.disposition()['collected']) == 1
            receipt = executor.execute(outcome)
            assert receipt.settlement_request['outcome'] == 'calibration'
            assert receipt.schema == execution.EXECUTION_RECEIPT_SCHEMA_V2
            assert execution.DriverExecutionReceipt.from_dict(receipt.to_dict()) == receipt
            if ordinal == 1:
                assert pid_log.read_text() == before_retry_pids  # Exact finished-attempt retry, no launch.
            receipts.append(receipt)
        assert instance.preparation_owner.pending_requests() == ()
        history = controller.unified_driver_preparation_history(requests)
        assert len(history['records']) == 2
        assert all(row['settlement']['outcome'] == 'calibration' for row in history['records'])
        solved = instance.preparation_owner.solve_collected(requests[0].declaration)
        assert solved is not None
        retained = instance.preparation_owner.reopen_solve(solved, declaration=requests[0].declaration)
        assert retained['qualification'] == 'unavailable'
        assert retained['ranking_authorized'] is False
        assert retained['numeric_solve'] is not None or retained['numeric_reasons']
        assert instance.preparation_owner.solve_collected(requests[0].declaration) == solved
        assert len(controller.unified_driver_preparation_history(requests)['records']) == 2
        references = [item for receipt in receipts for item in receipt.settlement_request['terminal_refs']
                      if item.startswith('calibration-collected:')]
        assert len(references) == 2
        for item in references:
            assert json.loads(item.removeprefix('calibration-collected:'))['declaration_digest'] == requests[0].declaration.digest
        for changed in ('numeric_solve', 'numeric_reasons', 'qualification_debt'):
            body = prep._plain(retained)
            body[changed] = {'forged': True} if changed == 'numeric_solve' else ['self-consistent forgery']
            with_store = mc.ArtifactStore(controller.store / 'unified-native-artifacts')
            try:
                forged = with_store.write(f'calibration-solve:{requests[0].declaration.digest}', body)
            finally:
                with_store.close()
            forged_ref = prep.CollectedCalibrationReference(requests[0].declaration.digest,
                                                            forged.locator, forged.sha256)
            with pytest.raises(prep.PreparationRefused, match='original owning derivation'):
                instance.preparation_owner.reopen_solve(forged_ref, declaration=requests[0].declaration)
        receipt_row = receipts[-1].to_dict()
        for mutation in ('downgrade', 'foreign', 'duplicate', 'failed', 'runtime'):
            row = json.loads(json.dumps(receipt_row))
            refs = row['settlement_request']['terminal_refs']
            if mutation == 'downgrade':
                row['schema'] = execution.EXECUTION_RECEIPT_SCHEMA
            elif mutation == 'foreign':
                ref = json.loads(refs[-2].removeprefix('calibration-collected:'))
                ref['declaration_digest'] = 'f' * 64
                refs[-2] = 'calibration-collected:' + prep._bytes(ref).decode()
            elif mutation == 'duplicate':
                refs.insert(-1, refs[-2])
            elif mutation == 'failed':
                row['settlement_request']['outcome'] = 'failed'
            else:
                row['settlement_request']['selection']['proposal']['stage_class'] = 'search'
                row['settlement_request']['receipt']['stage_class'] = 'search'
                row['settlement_request']['selection']['proposal_digest'] = driver._digest(
                    row['settlement_request']['selection']['proposal'])
            row.pop('receipt_digest')
            row['receipt_digest'] = driver._digest(row)
            with pytest.raises(execution.DriverExecutionRefused):
                execution.DriverExecutionReceipt.from_dict(row)

        executor.close()
        executor = None
        controller.close()
        recovered_engine = scheduling.SchedulerEngine(engine.config,
            scheduling.initial_state(engine.config, original.resolved.campaign_id))
        controller = campaign_control.CampaignController(original.resolved, controller.store,
            snapshot_version=3, scheduler_engine=recovered_engine, readiness_check=lambda: (True, None),
            lifecycle_provider=provider)
        controller.__enter__()
        recovered_owner = prep.InstalledServingPreparationOwner(controller=controller, requests=requests)
        try:
            before_recovery_pids = pid_log.read_text()
            assert recovered_owner.pending_requests() == ()
            assert len(recovered_owner.disposition()['collected']) == 2
            # Recompute only the pure numeric diagnostic from exact settled
            # original samples. No new IDs, workers, measurements or accounting.
            assert recovered_owner.solve_collected(requests[0].declaration) == solved
            assert pid_log.read_text() == before_recovery_pids
            for receipt in receipts:
                restored = execution.DriverExecutionReceipt.from_dict(receipt.to_dict())
                assert execution.UnifiedDriverExecution.retry_durable(controller, restored)['status'] == 'duplicate'
            assert len(controller.unified_driver_preparation_history(requests)['records']) == 2
        finally:
            recovered_owner.close()
    finally:
        if executor is not None:
            executor.close()
        controller.close()
        if pid_log is not None and pid_log.exists():
            for line in pid_log.read_text().splitlines():
                pid, ticks = map(int, line.split())
                assert not lifecycle.same_process(lifecycle.ProcessIdentity(
                    pid, ticks, lifecycle.process_identity(os.getpid()).boot_id))


def test_actual_observed_placement_failure_uses_only_predeclared_pair_retry(tmp_path, monkeypatch):
    from . import lifecycle_observation as lo, native_parent_service as native
    original, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    monkeypatch.undo()
    executor, pid_log = None, None
    provider = controller._lifecycle_provider
    original_close = provider.close_held_receipt

    def close_calibration(**kwargs):
        trusted = original_close(**kwargs)
        return replace(trusted, receipt=replace(trusted.receipt, stage_class='calibration',
                                                memory_reservation_bytes=0))

    monkeypatch.setattr(provider, 'close_held_receipt', close_calibration)
    original_native_init = native.NativeParentEvidenceService.__init__

    def factual_init(self, *args, **kwargs):
        root = tmp_path / 'native-probe'
        kwargs['runtime_probe'] = lo.FilesystemProbe(proc_root=root / 'proc',
            sysfs_cpu_root=root / 'cpu', boot_id_path=root / 'boot', cgroup_root=Path('/'))
        original_native_init(self, *args, **kwargs)

    # Only the raw filesystem observation boundary is synthetic. The installed
    # producer, placement reducer, registry and capture replay are unchanged.
    monkeypatch.setattr(native.NativeParentEvidenceService, '__init__', factual_init)
    try:
        measurement, pid_log = install_observation_fixture(tmp_path, monkeypatch,
                                                          contaminate_initial=True)
        requests = make_requests(tmp_path, original, engine, measurement, max_attempts=2, kinds=('aa',))
        requests = tuple(replace(item, plan=ep.ExperimentPlan.from_dict({**item.plan.to_dict(),
            'required_witnesses': ['identity', 'placement']})) for item in requests)
        instance = driver.UnifiedCampaignDriver(resolved_campaign=original.resolved,
            controller=controller, scheduler_engine=engine, profiles=original.profiles,
            evidence_index=original.evidence, runtime_anchors=original.runtime_anchors,
            runtime_dimensions={}, experiment_plans={}, profile_requests={}, actor_identities={},
            native_artifact_sink_ref=original.sink_ref, calibration_requests=requests,
            execution_inputs={item.declaration.target_revision: driver.ExecutionInput(
                item.declaration.target_revision, item.prompts, item.declaration.max_stage_seconds,
                item.declaration.teardown_seconds, item.plan.loaded_instrument["identity_sha256"])
                for item in requests},
            executable_work_kinds={driver.CALIBRATION_WORK_KIND})
        first = instance.tick(now=1.0)
        prepared = instance.materialize_calibration(first)
        executor = execution.UnifiedDriverExecution(driver=instance, controller=controller,
            observation_configuration=_observation_configuration(prepared),
            native_evidence_configuration=native.NativeFactualEvidenceConfiguration())
        first_receipt = executor.execute(first)
        assert first_receipt.settlement_request['outcome'] == 'invalid'
        first_request = prep.CalibrationPreparationDispatch.from_dict(prepared.dispatch).request
        assert first_request.attempt_ordinal == 0
        collection = next(item for item in first_receipt.settlement_request['terminal_refs']
                          if item.startswith('calibration-collected:'))
        reference = prep.CollectedCalibrationReference.from_dict(json.loads(
            collection.removeprefix('calibration-collected:')))
        chunk = instance.preparation_owner.reopen_chunk(reference, request=first_request)
        assert chunk['raw_status'] == 'contaminated'
        assert any(row['raw_unit']['witnesses']['placement']['status'] == 'fail'
                   for row in chunk['observations'])
        pending = instance.preparation_owner.pending_requests()
        retry = next(item for item in pending if item.chunk_identity == first_request.chunk_identity)
        assert retry.attempt_ordinal == 1
        assert retry.digest in {item.digest for item in requests}
        assert retry.logical_membership == first_request.logical_membership
        assert retry.plan.expected_units[0].arm != first_request.plan.expected_units[0].arm
        assert not ({item.process_id for item in retry.plan.expected_units}
                    & {item.process_id for item in first_request.plan.expected_units})
        completed = []
        for now in (2.0, 3.0):
            next_outcome = instance.tick(now=now)
            assert next_outcome.status == 'intent_recorded', next_outcome.to_dict()
            selected = instance.materialize_calibration(next_outcome)
            selected_request = prep.CalibrationPreparationDispatch.from_dict(selected.dispatch).request
            receipt = executor.execute(next_outcome)
            assert receipt.settlement_request['outcome'] == 'calibration'
            completed.append(selected_request.digest)
        assert retry.digest in completed
        assert instance.preparation_owner.pending_requests() == ()
        history = controller.unified_driver_preparation_history(requests)
        assert len(history['records']) == 3
        assert sorted(row['settlement']['outcome'] for row in history['records']) == [
            'calibration', 'calibration', 'invalid']
        # Three distinct original pairs were executed; the successful other
        # block's unused retry never acquired or launched.
        assert len(pid_log.read_text().splitlines()) == 6
        assert instance.preparation_owner.solve_collected(first_request.declaration) is None
    finally:
        if executor is not None:
            executor.close()
        controller.close()
        if pid_log is not None and pid_log.exists():
            for line in pid_log.read_text().splitlines():
                pid, ticks = map(int, line.split())
                assert not lifecycle.same_process(lifecycle.ProcessIdentity(
                    pid, ticks, lifecycle.process_identity(os.getpid()).boot_id))
