"""Hermetic connector tests; fake provider facts are not live containment proof."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from . import campaign_control
from . import driver_execution as de
from . import experiment_plan
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import lifecycle_observation as lo
from .. import journal as journal_module
from . import planned_serving
from . import observation_binding as ob
from . import scheduling
from . import unified_driver
from . import unified_worker
from . import worker_lifecycle as wl
from .test_unified_driver import runtime_driver
from .test_unified_planner import scheduler
from .test_unified_worker import _measure
from .test_lifecycle_observation import _budgets
from .test_worker_lifecycle import MockOwnedContainer


class HeldProvider:
    """Explicit test provider; its claims prove no real resource ownership."""

    def __init__(self, root):
        self.root = root
        self.authorization = None
        self.claim_threads = []

    def authorize(self, request, container_id, deadline):
        del request
        grant = wl.GrantReceipt("test-grant", 1, deadline, wl.monotonic_clock_domain())
        self.authorization = wl.AuthorizedLaunch(
            grant, container_id, MockOwnedContainer(self.root, container_id))
        return self.authorization

    def refresh(self, authorization, deadline):
        del deadline
        return authorization.grant

    def release(self, authorization, deadline):
        del authorization, deadline
        return True

    def inspect_pending(self, identity, deadline):
        del identity, deadline
        return wl.PendingAcquisitionInspection(
            "absent", None, "test provider certifies no acquisition")

    def describe_active_observation_claim(
            self, *, authorization, request, unit_id,
            process_generation_id, deadline):
        del request, unit_id, process_generation_id, deadline
        self.claim_threads.append(threading.current_thread().name)
        body = {"schema": wl.ACTIVE_OBSERVATION_CLAIM_SCHEMA,
                "grant_id": authorization.grant.grant_id,
                "grant_generation": authorization.grant.generation,
                "container_id": authorization.container_id,
                "held_claim": {"logical_cpus": sorted(os.sched_getaffinity(0))[:4],
                               "gpu_devices": []},
                "active_claim_ref": f"fixture-claim:{authorization.container_id}"}
        return {**body, "claim_digest": wl._digest(body)}

    def close_held_receipt(self, *, authorization, request, worker_id,
                           worker_generation, container_identity,
                           lifecycle_started_at, released_at, deadline):
        del container_identity, deadline
        receipt = scheduling.HeldClaimReceipt(
            f"held:{worker_id}:{worker_generation}", request.request_id, "cpu", "search",
            lifecycle_started_at, released_at, worker_generation,
            authorization.grant.generation, (f"test:{authorization.container_id}",),
            0.5, (), 1024, ("0",), {request.request_id: 1.0})
        return wl.TrustedHeldClaimReceipt(
            request.request_id, request.plan_digest, worker_id, worker_generation,
            authorization.grant.grant_id, authorization.grant.generation,
            authorization.container_id, authorization.grant.clock_domain,
            lifecycle_started_at, released_at, receipt)


class FixtureScoredParentEvidenceProducer(de.UnknownParentEvidenceProducer):
    """Fixture-only scorer deriving passes from captured parent-owned facts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixture_placement = {}

    def _record_target(self, binding_key, binding, target):
        super()._record_target(binding_key, binding, target)
        actual = sorted(os.sched_getaffinity(target["pid"]))
        if (actual != list(binding.held_claim["logical_cpus"])
                or actual != list(binding.requested_effective_state["logical_cpus"])):
            raise de.DriverExecutionRefused(
                "fixture target affinity differs from parent claim/request")
        self._fixture_placement[binding_key] = unified_worker._digest({
            "pid": target["pid"], "start_ticks": target["start_ticks"],
            "logical_cpus": actual, "binding_ref": target["binding_ref"]})

    def _reopen_fixture_observation(self, binding):
        matches = []
        for path in sorted(Path(self.prepared.artifact_root).glob("*.json"))[:128]:
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if (row.get("schema") == lo.OBSERVATION_SCHEMA
                    and row.get("observation_id") == binding.observation_id):
                matches.append(lo.validate_observation(row))
        if len(matches) != 1:
            raise de.DriverExecutionRefused(
                "fixture parent cannot reopen one exact lifecycle observation")
        return matches[0]

    def _completion_for(self, fence, observation):
        matches = [(key, binding) for key, binding in self._bindings.items()
                   if key[2:] == (fence.unit_id, fence.process_generation_id,
                                  fence.fence_id)]
        if len(matches) != 1:
            raise de.DriverExecutionRefused(
                "fixture scoring lacks one exact parent observation binding")
        key, binding = matches[0]
        target = self._targets.get(key)
        placement_ref = self._fixture_placement.get(key)
        requests = observation.get("requests")
        unit = next(item for item in self.plan.expected_units
                    if item.unit_id == fence.unit_id)
        prompts = {item["prompt_id"]: item
                   for item in self.prepared.prompts.to_dict()["prompts"]}
        exact_requests = isinstance(requests, (list, tuple)) \
            and tuple(row.get("prompt_id") for row in requests) == unit.expected_prompt_ids \
            and all(row.get("request_sha256") == prompts[row["prompt_id"]]["request_digest"]
                    and row.get("predicted_n") == prompts[row["prompt_id"]]["n_predict"]
                    and row.get("terminal") is True and row.get("error") is None
                    for row in requests if hasattr(row, "get"))
        lifecycle_record = self._reopen_fixture_observation(binding)
        clean_overlap = all(
            not interval["potential_foreign_overlap"]
            for interval in lifecycle_record["intervals"]
            if interval["phase"] in {"load", "placement", "warmup",
                                     "measurement", "teardown"})
        if (target is None or placement_ref is None
                or target["worker_binding"] != binding.worker_binding
                or target["binding_ref"] == ""
                or observation.get("process_pid") != target["pid"]
                or not exact_requests or not clean_overlap):
            raise de.DriverExecutionRefused(
                "fixture scoring parent facts do not bind a completed target")
        response_ref = "fixture-only-parent-evidence:" + unified_worker._digest({
            "fence_id": fence.fence_id, "binding": binding.to_dict(),
            "target": unified_worker._plain(target),
            "observation": unified_worker._plain(observation)})
        contention_ref = "fixture-only-parent-evidence:" + lifecycle_record["content_sha256"]
        witnesses = {
            "native-capture-v1": experiment_plan.Witness("pass", response_ref),
            "contention": experiment_plan.Witness("pass", contention_ref),
            "placement": experiment_plan.Witness(
                "pass", "fixture-only-parent-evidence:" + placement_ref)}
        witnesses["residency"] = experiment_plan.Witness("unknown", None)
        return planned_serving.StageCompletion(
            fence.fence_id, True, witnesses, "clean", None)


class DirectUnitAuthority:
    """Test-only adapter feeding the same bounded parent evidence cache."""

    def __init__(self, parent, start):
        self.parent, self.start = parent, start

    def admit(self, *, sequence, plan_digest, unit, prior_completion_digest):
        del plan_digest, prior_completion_digest
        return unified_worker.UnitPermit(
            sequence, unit.unit_id, unit.process_id, f"test-fence:{sequence}",
            self.start.provider_deadline, self.start.grant_id,
            self.start.grant_generation, self.start.container_id, True, "test held")

    def complete(self, *, sequence, fence, observation):
        key, completion = self.parent.request_completion(
            start=self.start, sequence=sequence, fence=fence, observation=observation)
        deadline = time.monotonic() + 2
        while completion is None and time.monotonic() < deadline:
            time.sleep(0.001)
            completion = self.parent.poll_completion(key)
        assert completion is not None
        return completion

    def verify_continuation(self, raw, plan, prompts, previous_lineage_id):
        del raw, plan, prompts, previous_lineage_id
        return False


def _resume(controller, campaign_id):
    digest = campaign_control.command_digest(
        operation="resume", payload={}, campaign_id=campaign_id, config_generation=1)
    controller.apply_command({
        "schema": campaign_control.COMMAND_SCHEMA, "campaign_id": campaign_id,
        "config_generation": 1, "request_id": "resume-execution",
        "operation": "resume", "payload": {}, "payload_digest": digest,
        "expected_control_revision": 0})


def _command(controller, operation, request_id):
    payload = {}
    digest = campaign_control.command_digest(
        operation=operation, payload=payload,
        campaign_id=controller.resolved.campaign_id,
        config_generation=controller.config_generation)
    return controller.apply_command({
        "schema": campaign_control.COMMAND_SCHEMA,
        "campaign_id": controller.resolved.campaign_id,
        "config_generation": controller.config_generation,
        "request_id": request_id, "operation": operation, "payload": payload,
        "payload_digest": digest,
        "expected_control_revision": controller.control_revision})


def _open_fds():
    result = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            result[int(name)] = os.readlink(f"/proc/self/fd/{name}")
        except FileNotFoundError:
            pass
    return result


def test_driver_requires_concrete_parent_observation_verifiers(tmp_path, monkeypatch):
    driver, controller, _lifecycle, _engine = _owned_stack(tmp_path, monkeypatch)
    try:
        with pytest.raises(de.DriverExecutionRefused, match="concrete parent"):
            de.UnifiedDriverExecution(
                driver=driver, controller=controller, observation_verifiers=object())
        connector = de.UnifiedDriverExecution(
            driver=driver, controller=controller,
            observation_verifiers=ob.ParentObservationVerifiers())
        connector.close()
    finally:
        controller.close()


def _owned_stack(tmp_path, monkeypatch, *, return_code=0, recipe=None):
    driver, _old, enrolled, _target, target_digest = runtime_driver(
        git_source=True, recipe=recipe)
    original_input = driver.execution_inputs[target_digest]
    prompt_body = original_input.prompt_manifest.to_dict()
    prompt_body["prompts"] = prompt_body["prompts"][:1]
    prompt_body["digest"] = unified_driver._digest({
        key: value for key, value in prompt_body.items() if key != "digest"})
    prompts = planned_serving.FrozenPromptManifest.from_dict(prompt_body)
    driver.execution_inputs = {target_digest: unified_driver.ExecutionInput(
        target_digest, prompts, original_input.max_stage_seconds,
        original_input.teardown_seconds, original_input.instrument_id)}
    plan_key, original_plan = next(iter(driver.experiment_plans.items()))
    plan_body = original_plan.to_dict()
    for unit in plan_body["expected_units"]:
        unit["expected_prompt_ids"] = ["p1"]
    driver.experiment_plans = {plan_key: experiment_plan.ExperimentPlan.from_dict(plan_body)}
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    containers = tmp_path / "containers"
    containers.mkdir(mode=0o700)
    provider = HeldProvider(containers)
    controller = campaign_control.CampaignController(
        enrolled, tmp_path / "controller", snapshot_version=3,
        scheduler_engine=engine, readiness_check=lambda: (True, None),
        lifecycle_provider=provider)
    controller.__enter__()
    _resume(controller, enrolled.campaign_id)
    driver.controller = controller
    driver.scheduler = engine

    lifecycle = controller._worker_lifecycle
    assert lifecycle is not None

    def fake_wait(process, fd, nonce, contract_digest, request, authorization, held_grant,
                  deadline, *, planned_invocation, container_identity, worker_id,
                  worker_generation, container_id):
        del fd, authorization
        start = unified_worker.WorkerStart(
            nonce, request.request_id, planned_invocation.prepared.prepared_digest,
            request.plan_digest, request.lineage_id, request.stage_id,
            lifecycle.binding.campaign_id, lifecycle.binding.config_digest,
            lifecycle.binding.config_generation, lifecycle.binding.supervisor_id,
            lifecycle.binding.supervisor_incarnation, worker_id, worker_generation,
            held_grant.grant_id, held_grant.generation, container_id,
            wl.process_identity(os.getpid()), container_identity, held_grant.clock_domain,
            deadline)
        planned_invocation.start = start
        authority = DirectUnitAuthority(planned_invocation.parent_authority, start)
        times = iter(f"2026-09-09T00:00:{index:02d}Z" for index in range(20))
        reference = unified_worker.run_prepared_stage(
            planned_invocation.prepared, start, _test_authority=authority,
            _test_membership_probe=lambda _: None, _test_measure=_measure,
            clock=lambda: 1.0, wall_clock=times.__next__)
        planned_invocation._reference = reference
        planned_invocation._reference_digest = unified_worker._digest(reference.to_dict())
        return {"schema": wl.OUTCOME_SCHEMA, "nonce": nonce,
                "contract_digest": contract_digest, "child_pid": process.pid,
                "return_code": return_code, "forwarded_signal": None, "stdout_bytes": 0,
                "stderr_bytes": 0, "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "stdout_truncated": False, "stderr_truncated": False}

    monkeypatch.setattr(lifecycle, "_wait_outcome", fake_wait)
    return driver, controller, lifecycle, engine


def _as_observed_v2(prepared, measurement_callable, *, scientific_adapters=None,
                    search_window_configuration=None):
    store = mc.ArtifactStore(prepared.artifact_root)
    try:
        instrument = ob.seal_loaded_instrument(
            store=store, measurement_callable=measurement_callable,
            fence_clock=time.monotonic, serving_timer=time.time,
            scientific_adapters=scientific_adapters,
            search_window_configuration=search_window_configuration)
    finally:
        store.close()
    plan_row = prepared.plan.to_dict() | {
        "schema": experiment_plan.PLAN_SCHEMA_V2,
        "loaded_instrument": instrument.to_dict(),
        "anchor_identity": planned_serving.arm_identity(
            prepared.runtime_pair.anchor.template, prepared.runtime_pair.anchor,
            loaded_instrument=instrument.to_dict()),
        "candidate_identity": planned_serving.arm_identity(
            prepared.runtime_pair.candidate.template, prepared.runtime_pair.candidate,
            loaded_instrument=instrument.to_dict())}
    plan = experiment_plan.ExperimentPlan.from_dict(plan_row)
    dispatch = unified_worker._plain(prepared.dispatch)
    proposal_row = dispatch["proposal"] | {"experiment_plan_digest": plan.digest}
    proposal = unified_worker.up.UnifiedProposal.from_dict(proposal_row)
    dispatch["proposal"] = proposal.to_dict()
    dispatch["experiment_intent"]["proposal_digest"] = proposal.digest
    dispatch["experiment_intent"]["experiment_plan_digest"] = plan.digest
    dispatch = unified_worker.up.DispatchRequest(**dispatch).to_dict()
    body = prepared.body() | {
        "schema": unified_worker.PREPARED_SCHEMA_V2, "dispatch": dispatch,
        "plan": plan.to_dict(), "capture_context_base":
            unified_worker._plain(prepared.capture_context_base)
            | {"instrument_id": instrument.identity_sha256}}
    return unified_worker.PreparedPlannedServingStage.from_dict(
        {**body, "prepared_digest": unified_worker._digest(body)}), instrument


def _observation_configuration(prepared):
    requested = {recipe.execution_digest: {
        "logical_cpus": sorted(os.sched_getaffinity(0))[:4], "numa_nodes": [0, 1],
        "thp_mode": "madvise"}
        for recipe in (prepared.runtime_pair.anchor, prepared.runtime_pair.candidate)}
    return ob.ParentObservationConfiguration(
        # Original owning T0 collection runs while placement is still held.
        # A 10 ms / 64-sample fixture exhausted its capacity before later
        # mandatory markers under scheduling pressure. Pin a bounded ~25 s
        # periodic allowance; retain the 4 MiB byte cap and real marker checks.
        requested, {}, 0.1, 0.2, _budgets(
            max_samples=256, phase_ack_timeout_s=0.5, join_timeout_s=0.5))


def _run_real_controller_child_v2_capture_and_restart(
        tmp_path, monkeypatch, *, producer_type, recipe=None,
        scientific_adapters=None, search_window_configuration=None):
    driver, controller, lifecycle, engine = _owned_stack(
        tmp_path, monkeypatch, recipe=recipe)
    # Restore the actual lifecycle watcher; this test does not inject a completion.
    monkeypatch.undo()
    issued = driver.tick(now=1.0)
    prepared = driver.materialize_runtime(issued)
    measure_module_path = tmp_path / "fixture_measure.py"
    measure_module_path.write_text("""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from autokernel.loop import worker_lifecycle as wl

def _stat(pid, start, ticks):
    fields = ['0'] * 40
    fields[0], fields[11], fields[12] = 'S', str(ticks), '0'
    fields[19], fields[36] = str(start), '0'
    return f'{pid} (fixture-server) ' + ' '.join(fields) + '\\n'

def observed_measure(template, build_dir, port, **kwargs):
    del build_dir, port
    session = kwargs['observation_session']
    preparation_root = os.environ.get('AUTOKERNEL_MODEL_PREPARATION_ROOT')
    if preparation_root:
        selected = kwargs['resolved_recipe']
        matches = []
        for path in Path(preparation_root).iterdir():
            if not path.is_file():
                continue
            try:
                row = json.loads(path.read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if row.get('schema') == 'epyc.autokernel.scheduled_model_preparation_binding.v1':
                preparation = row['preparation']
                if (preparation['recipe_execution_digest'] == selected.execution_digest
                        and preparation['entry_path'] == selected.model.path
                        and preparation['entry_sha256'] == selected.model.sha256):
                    matches.append(row)
        if len(matches) != 1:
            raise AssertionError('measurement started before model preparation receipt')
    proc_root = Path(os.environ['AUTOKERNEL_FIXTURE_PROC_ROOT'])
    pid_log = Path(os.environ['AUTOKERNEL_FIXTURE_PID_LOG'])
    server = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fixture_cpus = {int(value) for value in
                    os.environ['AUTOKERNEL_FIXTURE_CPUS'].split(',')}
    os.sched_setaffinity(server.pid, fixture_cpus)
    identity = wl.process_identity(server.pid)
    with pid_log.open('a', encoding='ascii') as stream:
        stream.write(f'{identity.pid} {identity.start_ticks}\\n')
    root = proc_root / str(server.pid)
    (root / 'fd').mkdir(parents=True, exist_ok=True)
    (root / 'stat').write_text(_stat(server.pid, identity.start_ticks, 1))
    (root / 'status').write_text(
        'Cpus_allowed_list:\\t' + ','.join(str(value) for value in sorted(fixture_cpus))
        + '\\nMems_allowed_list:\\t0-1\\n')
    container = session.context['worker_binding']['container_identity']['path']
    (root / 'cgroup').write_text(f'0::{{container}}\\n')
    (root / 'numa_maps').write_text(
        '00400000 default kernelpagesize_kB=2048 N0=2\\n')
    (root / 'smaps_rollup').write_text(
        'Rss: 40 kB\\nAnonHugePages: 4 kB\\nShmemPmdMapped: 0 kB\\nFilePmdMapped: 8 kB\\n')
    (root / 'maps').write_text('')
    try:
        session.start()
        session.phase('load')
        session.attach_target(server.pid)
        for phase in ('placement', 'health', 'warmup', 'measurement'):
            session.phase(phase)
        # A labelled synthetic interval, not model work: let both actual
        # observer loops run DURING the fixture's measurement window.
        import time
        time.sleep(0.04)
        session.checkpoint('measurement_end')
        prompt_id, body = kwargs['frozen_requests'][0]
        kwargs['observation'].append({
            'schema': 'epyc.autokernel.serving_observation.v1',
            'process_pid': server.pid,
            'requests': [{'phase': 'measurement', 'slot_index': 0,
                'prompt_id': prompt_id, 'request_sha256': hashlib.sha256(body).hexdigest(),
                'predicted_n': template.n_predict, 'predicted_per_second': 10.0,
                'terminal': True, 'error': None}],
            'residency': {'status': 'mocked'}, 'teardown': 'terminated', 'failure': None})
        return 10.0
    finally:
        session.phase('teardown')
        server.terminate()
        server.wait(timeout=2)
        session.finish()
""", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("fixture_measure", measure_module_path)
    assert spec is not None and spec.loader is not None
    measure_module = importlib.util.module_from_spec(spec)
    sys.modules["fixture_measure"] = measure_module
    spec.loader.exec_module(measure_module)
    prepared, _instrument = _as_observed_v2(
        prepared, measure_module.observed_measure,
        scientific_adapters=scientific_adapters,
        search_window_configuration=search_window_configuration)
    store = mc.ArtifactStore(prepared.artifact_root)
    validator = nc.NativeCaptureValidator(
        binding=nc.NativeCaptureBinding(
            prepared.capture_context_base["campaign_id"],
            prepared.capture_context_base["config_digest"],
            prepared.capture_context_base["config_generation"],
            prepared.capture_context_base["supervisor_id"],
            prepared.capture_context_base["supervisor_incarnation"]),
        store=store, fence_provider=lambda *_args: None,
        observation_verifiers=ob.ParentObservationVerifiers())
    controller.register_native_capture(validator)
    authority = unified_worker.ParentUnitEvidenceAuthority(
        max_records=len(prepared.plan.expected_units) * (
            3 + len(unified_worker.OBSERVATION_WINDOW_MARKERS)))
    producer = producer_type(
        authority, prepared, lifecycle, _observation_configuration(prepared))
    invocation = unified_worker.PlannedWorkerInvocation.open(prepared, authority)
    probe_root = tmp_path / "fixture-probe"
    pid_log = tmp_path / "owned-serving-pids.txt"
    worker_script = tmp_path / "unified_worker.py"
    bootstrap_script = tmp_path / "worker_bootstrap.py"
    bootstrap_script.write_bytes(
        Path(wl.__file__).with_name("worker_bootstrap.py").read_bytes())
    source_root = Path.cwd() / "scripts/kernel_rnd"
    fixture_cpus = sorted(os.sched_getaffinity(0))[:4]
    worker_script.write_text(f"""
import argparse, os, sys, types
from pathlib import Path
sys.path.insert(0, {str(source_root)!r})
yaml = types.ModuleType('yaml')
class UnavailableYamlError(Exception):
    pass
yaml.YAMLError = UnavailableYamlError
yaml.safe_load = lambda *_args, **_kwargs: (_ for _ in ()).throw(UnavailableYamlError())
sys.modules['yaml'] = yaml
from autokernel.loop import unified_worker as uw
from autokernel.loop import lifecycle_observation as lo
sys.path.insert(0, {str(tmp_path)!r})
from fixture_measure import observed_measure
parser = argparse.ArgumentParser()
parser.add_argument('--start-fd', type=int, required=True)
parser.add_argument('--control-fd', type=int, required=True)
parser.add_argument('--result-fd', type=int, required=True)
args = parser.parse_args()
root = Path({str(probe_root)!r})
proc, cpu, cgroups = root / 'proc', root / 'cpu', root / 'cgroup'
proc.mkdir(parents=True)
(cgroups / 'owned').mkdir(parents=True)
for index, number in enumerate({fixture_cpus!r}):
    siblings, node = str(number), index % 2
    item = cpu / f'cpu{{number}}'
    (item / 'topology').mkdir(parents=True)
    (item / 'topology' / 'thread_siblings_list').write_text(siblings)
    (item / f'node{{node}}').mkdir()
Path({str(probe_root / 'boot')!r}).write_text(
    Path('/proc/sys/kernel/random/boot_id').read_text(), encoding='ascii')
(root / 'pressure').write_text(
    'some avg10=0 avg60=0 avg300=0 total=10\\nfull avg10=0 avg60=0 avg300=0 total=2\\n')
(root / 'thp').write_text('always [madvise] never\\n')
(proc / 'meminfo').write_text(
    'MemAvailable: 1000 kB\\nSwapFree: 500 kB\\nSwapTotal: 500 kB\\n')
(proc / 'vmstat').write_text('pswpin 0\\npswpout 0\\n')
(root / 'global-vram').write_text('0\\n')
(root / 'kfd').mkdir()
probe = lo.FilesystemProbe(proc_root=proc, sysfs_cpu_root=cpu,
    boot_id_path=root / 'boot', memory_psi_path=root / 'pressure',
    thp_enabled_path=root / 'thp', cgroup_root=cgroups,
    global_vram_paths={{'gpu0': root / 'global-vram'}})
os.environ['AUTOKERNEL_FIXTURE_PROC_ROOT'] = {str(probe_root / 'proc')!r}
os.environ['AUTOKERNEL_FIXTURE_PID_LOG'] = {str(pid_log)!r}
os.environ['AUTOKERNEL_FIXTURE_CPUS'] = {','.join(str(value) for value in fixture_cpus)!r}
os.environ['AUTOKERNEL_MODEL_PREPARATION_ROOT'] = {
    str(prepared.artifact_root) if scientific_adapters is not None else ''!r}
uw.run_from_fds(start_fd=args.start_fd, control_fd=args.control_fd,
    result_fd=args.result_fd, _test_membership_probe=lambda _start: None,
    _test_measure=observed_measure,
    _test_observation_probe=probe)
""", encoding="utf-8")
    invocation._worker_path = worker_script
    invocation.argv = (
        invocation._interpreter, "-I", "-B", str(worker_script),
        "--start-fd", str(invocation._start_read),
        "--control-fd", str(invocation._control_child),
        "--result-fd", str(invocation._result_write))
    real_popen = wl.subprocess.Popen

    def fixture_bootstrap(argv, *args, **kwargs):
        if (isinstance(argv, tuple) and len(argv) > 4
                and Path(argv[4]).name == "worker_bootstrap.py"):
            argv = (*argv[:4], str(bootstrap_script), *argv[5:])
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(wl.subprocess, "Popen", fixture_bootstrap)
    request = invocation.stage_request(
        request_id="v2-real-fixture", lineage_id="v2-lineage",
        stage_id="v2-stage", control_revision=controller.control_revision)
    producer.start()
    terminal = None
    try:
        terminal = controller.run_worker_stage(request, planned_invocation=invocation)
    finally:
        producer.stop_and_join()
    assert terminal.accepted
    claim_threads = controller._lifecycle_provider.claim_threads
    assert all(name == "autokernel-parent-evidence" for name in claim_threads)
    checks_per_unit = 1 if search_window_configuration is None else 2
    assert len(claim_threads) >= checks_per_unit * len(prepared.plan.expected_units)
    start = invocation.start
    assert start is not None
    reference = invocation.result_reference()
    fence = controller.worker_result_fence(terminal)
    with controller.native_capture_callback() as capture:
        receipts = unified_worker.ingest_deferred_result(
            reference, prepared=prepared, start=start, terminal=terminal,
            fence=fence, capture_transaction=capture)
    assert len(receipts) == 2  # one sealed native carrier per arm
    descendant_events = [row for row in controller._active_worker_events
                         if row["event"] == "OWNED_DESCENDANT_CAPTURED"]
    # Terminal projection clears the active list; durable replay below proves retention.
    assert descendant_events == []
    identities = []
    for line in pid_log.read_text(encoding="ascii").splitlines():
        pid, ticks = (int(value) for value in line.split())
        identities.append(wl.ProcessIdentity(
            pid, ticks, wl.process_identity(os.getpid()).boot_id))
    assert len(identities) == len(prepared.plan.expected_units)
    assert all(not wl.same_process(identity) for identity in identities)
    for identity in identities:
        print(f"OWNED_V2_SERVING_PID pid={identity.pid} "
              f"start_ticks={identity.start_ticks} alive=false")

    store.close()
    controller_store = controller.store
    resolved = driver.resolved
    provider = controller._lifecycle_provider
    replay_engine = scheduling.SchedulerEngine(
        engine.config, scheduling.initial_state(engine.config, resolved.campaign_id))
    controller.close()
    restarted = campaign_control.CampaignController(
        resolved, controller_store, snapshot_version=3, scheduler_engine=replay_engine,
        readiness_check=lambda: (True, None), lifecycle_provider=provider)
    restarted.__enter__()
    try:
        # A restart replays the terminal but deliberately does not re-authorize it
        # as current-owner state in the new supervisor incarnation.
        assert restarted.worker_terminal_for_request(
            request_id=request.request_id, plan_digest=request.plan_digest,
            lineage_id=request.lineage_id, stage_id=request.stage_id) is None
        assert all(restarted.native_capture(item.record_id) is not None
                   for item in receipts)
        lifecycle_rows = [entry.payload for entry in restarted._journal.read_all()
                          if entry.kind == journal_module.KIND_WORKER_LIFECYCLE]
        captured = [row for row in lifecycle_rows
                    if row["event"] == "OWNED_DESCENDANT_CAPTURED"]
        assert len(captured) == len(prepared.plan.expected_units)
        projection = wl.project_events(lifecycle_rows)
        assert not projection.active
        replayed_terminal = projection.terminal[terminal.worker_id]
        assert replayed_terminal["data"]["result_digest"] == terminal.result_digest
        assert restarted._acquisition_projection.pending is None
    finally:
        restarted.close()


def test_real_controller_child_v2_capture_and_restart(tmp_path, monkeypatch):
    """Fixture-scored path retained for the actual registered ROOT projector probe."""
    _run_real_controller_child_v2_capture_and_restart(
        tmp_path, monkeypatch, producer_type=FixtureScoredParentEvidenceProducer)
    rows = journal_module.Journal(str(tmp_path / "controller" / "journal")).read_all()
    native = [row.payload["carrier"] for row in rows
              if row.kind == journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED]
    assert len(native) == 2 and all(row["status"] == "measurement" for row in native)
    for carrier in native:
        assert carrier["admissible_view"]["complete"] is True
        for selected in carrier["admissible_view"]["selected_rows"]:
            assert all(witness["status"] == "pass" and witness["ref"].startswith(
                "fixture-only-parent-evidence:")
                for name, witness in selected["witnesses"].items()
                if name in {"native-capture-v1", "contention", "placement"})


def test_real_controller_child_v2_diagnostic_retention(tmp_path, monkeypatch):
    """Unknown parent evidence remains diagnostic and is never silently upgraded."""
    _run_real_controller_child_v2_capture_and_restart(
        tmp_path, monkeypatch, producer_type=de.UnknownParentEvidenceProducer)
    rows = journal_module.Journal(str(tmp_path / "controller" / "journal")).read_all()
    native = [row.payload["carrier"] for row in rows
              if row.kind == journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED]
    assert len(native) == 2
    assert all(row["status"] == "diagnostic"
               and row["diagnostic_reason"] == "zero scored independent launches"
               for row in native)
    for carrier in native:
        reasons = carrier["admissible_view"]["rejection_reasons"]
        assert reasons and all(any(
            "required witness native-capture-v1 is not passed with a ref" in reason
            for reason in unit_reasons) for unit_reasons in reasons.values())


def test_nonzero_owned_terminal_is_charged_once_without_native_acceptance(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(
        tmp_path, monkeypatch, return_code=7)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    calls = {"lifecycle": 0}
    real_run = lifecycle.run_stage

    def count_run(*args, **kwargs):
        calls["lifecycle"] += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "run_stage", count_run)
    real_stop = de.UnknownParentEvidenceProducer.stop_and_join

    def stop_then_report(*args, **kwargs):
        real_stop(*args, **kwargs)
        raise RuntimeError("injected producer shutdown report")

    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "stop_and_join", stop_then_report)
    try:
        issued = driver.tick(now=1.0)
        receipt = connector.execute(issued)
        assert receipt.settlement_request["outcome"] == "failed"
        assert receipt.native_measurement_ids == ()
        assert receipt.result_reference is None
        assert receipt.terminal["return_code"] == 7
        assert engine.accounting_view().receipt_count == 1
        assert connector.execute(issued).to_dict() == receipt.to_dict()
        assert calls == {"lifecycle": 1}
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()


def test_accepted_terminal_with_stopped_producer_error_is_coherent_failed_receipt(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    original = de.UnknownParentEvidenceProducer.stop_and_join

    def stop_then_report(producer, *args, **kwargs):
        original(producer, *args, **kwargs)
        raise RuntimeError("producer reported after stopping")

    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "stop_and_join", stop_then_report)
    try:
        issued = driver.tick(now=1.0)
        receipt = connector.execute(issued)
        assert receipt.terminal["accepted"] is True
        assert receipt.result_reference is not None
        assert receipt.native_measurement_ids == ()
        assert receipt.settlement_request["outcome"] == "failed"
        assert de.DriverExecutionReceipt.from_dict(receipt.to_dict()) == receipt
        assert connector.execute(issued).to_dict() == receipt.to_dict()
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()


def test_unresolved_producer_thread_fences_successor_after_charging_attempt(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    original_run = de.UnknownParentEvidenceProducer._run
    original_stop = de.UnknownParentEvidenceProducer.stop_and_join
    release = threading.Event()
    calls = {"stop": 0}

    def blocked_after_work(producer):
        original_run(producer)
        release.wait(2)

    def bounded_stop(producer, *_args, **_kwargs):
        calls["stop"] += 1
        return original_stop(producer, timeout=0.02)

    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "_run", blocked_after_work)
    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "stop_and_join", bounded_stop)
    try:
        first = driver.tick(now=1.0)
        receipt = connector.execute(first)
        assert receipt.terminal["accepted"] is True
        assert receipt.settlement_request["outcome"] == "failed"
        assert engine.accounting_view().receipt_count == 1
        assert connector.execute(first).to_dict() == receipt.to_dict()
        successor = unified_driver.DriverOutcome(
            first.status, first.reasons, "f" * 64,
            unified_driver._thaw(first.selection))
        with pytest.raises(de.DriverExecutionUncertain, match="remains alive"):
            connector.execute(successor)
        assert engine.accounting_view().receipt_count == 1
        with pytest.raises(de.DriverExecutionUncertain, match="teardown remains unresolved"):
            connector.close()
        assert connector.execute(first).to_dict() == receipt.to_dict()
        release.set()
        connector.close()
        with pytest.raises(de.DriverExecutionRefused, match="closed"):
            connector.execute(first)
    finally:
        release.set()
        if not connector._closed:
            connector.close()
        controller.close()
    assert calls["stop"] >= 2


def test_partially_started_live_producer_is_retained_until_close_retry(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, _engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    original_run = de.UnknownParentEvidenceProducer._run
    original_start = de.UnknownParentEvidenceProducer.start
    original_stop = de.UnknownParentEvidenceProducer.stop_and_join
    release = threading.Event()

    def blocked_after_work(producer):
        original_run(producer)
        release.wait(2)

    def start_then_fail(producer):
        original_start(producer)
        raise RuntimeError("producer failed after thread start")

    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "_run", blocked_after_work)
    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "start", start_then_fail)
    monkeypatch.setattr(
        de.UnknownParentEvidenceProducer, "stop_and_join",
        lambda producer, *_args, **_kwargs: original_stop(producer, timeout=0.02))
    try:
        with pytest.raises(RuntimeError, match="after thread start"):
            connector.execute(issued)
        assert connector._successor_fence is not None
        assert len(connector._unresolved_producers) == 1
        with pytest.raises(de.DriverExecutionUncertain, match="teardown remains unresolved"):
            connector.close()
        release.set()
        connector.close()
    finally:
        release.set()
        if not connector._closed:
            connector.close()
        controller.close()


def test_pause_refusal_has_no_acquisition_and_same_issue_can_resume(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    try:
        issued = driver.tick(now=1.0)
        assert _command(controller, "pause", "pause-before-worker")["completed"]
        assert controller.worker_attempt_status(
            request_id="unseen-request", plan_digest="1" * 64,
            lineage_id="unseen-lineage", stage_id="unseen-stage") == "unknown"
        with pytest.raises(campaign_control.ControlRefused, match="admission is closed"):
            connector.execute(issued)
        assert connector._launched == set()
        assert controller._acquisition_projection.pending is None
        assert controller._worker_projection.active == {}
        assert _command(controller, "resume", "resume-before-worker")["completed"]
        receipt = connector.execute(issued)
        assert receipt.settlement_request["outcome"] == "invalid"
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()


def test_invocation_fds_close_when_stage_request_or_producer_start_fails(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, _engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    # Materialization pins the controller-owned artifact root once; this descriptor
    # is intentionally lifetime-scoped and is not an invocation channel.
    driver.materialize_runtime(issued)
    baseline = _open_fds()
    original_request = unified_worker.PlannedWorkerInvocation.stage_request
    original_start = de.UnknownParentEvidenceProducer.start
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                unified_worker.PlannedWorkerInvocation, "stage_request",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    ValueError("injected request failure")))
            with pytest.raises(ValueError, match="request failure"):
                connector.execute(issued)
        assert _open_fds() == baseline

        with monkeypatch.context() as scoped:
            def start_then_fail(producer):
                original_start(producer)
                raise RuntimeError("injected producer failure")

            scoped.setattr(
                de.UnknownParentEvidenceProducer, "start",
                start_then_fail)
            with pytest.raises(RuntimeError, match="producer failure"):
                connector.execute(issued)
        assert _open_fds() == baseline
        assert connector._launched == set()
        assert original_request is unified_worker.PlannedWorkerInvocation.stage_request
        assert original_start is de.UnknownParentEvidenceProducer.start
        assert connector.execute(issued).settlement_request["outcome"] == "invalid"
    finally:
        connector.close()
        controller.close()


def test_concurrent_duplicate_submission_runs_one_controller_owned_worker(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = lifecycle._wait_outcome

    def blocking_wait(*args, **kwargs):
        calls.append("worker")
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "_wait_outcome", blocking_wait)
    results, errors = [], []

    def execute():
        try:
            results.append(connector.execute(issued))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=execute)
    second = threading.Thread(target=execute)
    try:
        first.start()
        assert entered.wait(2)
        assert controller._worker_run_active is True
        assert controller.publish_snapshot()["active_worker"] is not None
        second.start()
        time.sleep(0.02)
        assert second.is_alive()
        release.set()
        first.join(3)
        second.join(3)
        assert not errors and len(results) == 2
        assert results[0].to_dict() == results[1].to_dict()
        assert calls == ["worker"]
        assert engine.accounting_view().receipt_count == 1
        assert controller._worker_run_active is False
    finally:
        release.set()
        first.join(3)
        second.join(3)
        connector.close()
        controller.close()


def test_close_waits_for_active_execute_and_then_fences_new_submission(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, _engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = lifecycle._wait_outcome

    def blocking_wait(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "_wait_outcome", blocking_wait)
    results, errors = [], []

    def execute():
        try:
            results.append(connector.execute(issued))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=execute)
    closer = threading.Thread(target=lambda: (connector.close(), closed.set()))
    try:
        worker.start()
        assert entered.wait(2)
        closer.start()
        time.sleep(0.02)
        assert not closed.is_set()
        release.set()
        worker.join(3)
        closer.join(3)
        assert not errors and len(results) == 1 and closed.is_set()
        with pytest.raises(de.DriverExecutionRefused, match="closed"):
            connector.execute(issued)
    finally:
        release.set()
        worker.join(3)
        closer.join(3)
        connector.close()
        controller.close()


def test_selected_runtime_reaches_native_journal_invalid_accounting_and_replay(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    try:
        issued = driver.tick(now=1.0)
        receipt = connector.execute(issued)
        assert receipt.settlement_request["outcome"] == "invalid"
        assert len(receipt.native_measurement_ids) == 2
        assert all(controller.native_capture(item) is not None
                   for item in receipt.native_measurement_ids)
        assert engine.accounting_view().receipt_count == 1
        entries = controller._journal.read_all()
        assert any(item.kind == journal_module.KIND_WORKER_ACQUISITION for item in entries)
        assert any(item.kind == journal_module.KIND_WORKER_LIFECYCLE for item in entries)
        assert controller.publish_snapshot()["active_worker"] is None
        assert controller._worker_run_active is False
        assert controller.worker_attempt_status(
            request_id=receipt.request_id, plan_digest=receipt.terminal["plan_digest"],
            lineage_id=receipt.lineage_id, stage_id=receipt.stage_id) == "terminal"
        assert connector.execute(issued).to_dict() == receipt.to_dict()
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()

    replay_engine = scheduling.SchedulerEngine(
        engine.config, scheduling.initial_state(engine.config, driver.resolved.campaign_id))
    with campaign_control.CampaignController(
            driver.resolved, tmp_path / "controller", snapshot_version=3,
            scheduler_engine=replay_engine) as replayed:
        assert replay_engine.accounting_view().receipt_count == 1
        assert replayed.worker_attempt_status(
            request_id=receipt.request_id, plan_digest=receipt.terminal["plan_digest"],
            lineage_id=receipt.lineage_id, stage_id=receipt.stage_id) == "unknown"
        duplicate = de.UnifiedDriverExecution.retry_durable(replayed, receipt)
        assert duplicate["status"] == "duplicate"
        assert replay_engine.accounting_view().receipt_count == 1


def test_default_missing_provider_is_safe_to_retry_without_acquisition(
        tmp_path, monkeypatch):
    driver, _old, enrolled, _target, _digest = runtime_driver(git_source=True)
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    with campaign_control.CampaignController(
            enrolled, tmp_path / "controller", snapshot_version=3,
            scheduler_engine=engine, readiness_check=lambda: (True, None)) as controller:
        _resume(controller, enrolled.campaign_id)
        driver.controller, driver.scheduler = controller, engine
        connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
        issued = driver.tick(now=1.0)
        real_stop = de.UnknownParentEvidenceProducer.stop_and_join

        def noisy_stop(*args, **kwargs):
            real_stop(*args, **kwargs)
            raise RuntimeError("secondary producer stop failure")

        monkeypatch.setattr(de.UnknownParentEvidenceProducer, "stop_and_join", noisy_stop)
        for _ in range(2):
            with pytest.raises(wl.WaitingAuthority, match="provider is unavailable"):
                connector.execute(issued)
        assert engine.accounting_view().receipt_count == 0
        assert controller._native_records == {}
        assert connector._launched == set()
        assert controller._acquisition_projection.pending is None
        assert controller._worker_projection.active == {}
        connector.close()


def test_ambiguous_authorization_never_relaunches_without_reconciliation(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    provider = controller._lifecycle_provider
    original_authorize = provider.authorize
    provider.authorize = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        TimeoutError("test authorization reply lost"))
    try:
        with pytest.raises(de.DriverExecutionUncertain, match="exact terminal recovery"):
            connector.execute(issued)
        with pytest.raises(de.DriverExecutionUncertain, match="not rerun"):
            connector.execute(issued)
        assert controller._acquisition_projection.pending is not None
        assert engine.accounting_view().receipt_count == 0
        provider.authorize = original_authorize
        assert controller.reconcile_workers() is None
        assert controller._acquisition_projection.pending is None
        with pytest.raises(de.DriverExecutionUncertain, match="not rerun"):
            connector.execute(issued)
    finally:
        provider.authorize = original_authorize
        connector.close()
        controller.close()


def test_exact_provider_denial_is_durable_and_safe_to_retry(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    provider = controller._lifecycle_provider
    original_authorize = provider.authorize
    calls = {"authorize": 0}

    def deny_once(request, container_id, deadline):
        calls["authorize"] += 1
        if calls["authorize"] == 1:
            return wl.AuthorizationDenied(
                request.request_id, container_id, "test capacity unavailable")
        return original_authorize(request, container_id, deadline)

    provider.authorize = deny_once
    try:
        with pytest.raises(wl.WaitingAuthority, match="capacity unavailable"):
            connector.execute(issued)
        assert connector._launched == set()
        assert controller._acquisition_projection.pending is None
        selection = scheduling.Selection.from_dict(issued.selection)
        assert controller.worker_attempt_status(
            request_id=selection.proposal.proposal_id,
            plan_digest=driver.materialize_runtime(issued).plan.digest,
            lineage_id=f"driver:{issued.transition_id}",
            stage_id=f"runtime:{issued.transition_id}") == "not_acquired"
        receipt = connector.execute(issued)
        assert receipt.terminal["worker_generation"] == 2
        assert engine.accounting_view().receipt_count == 1
        acquisition_rows = [
            item.payload for item in controller._journal.read_all()
            if item.kind == journal_module.KIND_WORKER_ACQUISITION]
        assert [row["phase"] for row in acquisition_rows[:2]] == ["INTENT", "RESOLVED"]
        assert acquisition_rows[1]["data"]["outcome"] == "denied"
        assert lifecycle._worker_generation == 2
    finally:
        provider.authorize = original_authorize
        connector.close()
        controller.close()


def test_denial_proof_does_not_cross_controller_incarnation_or_iterate_history(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    prepared = driver.materialize_runtime(issued)
    selection = scheduling.Selection.from_dict(issued.selection)
    provider = controller._lifecycle_provider
    provider.authorize = lambda request, container_id, _deadline: wl.AuthorizationDenied(
        request.request_id, container_id, "test denied")
    with pytest.raises(wl.WaitingAuthority, match="test denied"):
        connector.execute(issued)
    assert controller.worker_attempt_status(
        request_id=selection.proposal.proposal_id, plan_digest=prepared.plan.digest,
        lineage_id=f"driver:{issued.transition_id}",
        stage_id=f"runtime:{issued.transition_id}") == "not_acquired"
    connector.close()
    controller.close()

    replay_engine = scheduling.SchedulerEngine(
        engine.config, scheduling.initial_state(engine.config, driver.resolved.campaign_id))
    with campaign_control.CampaignController(
            driver.resolved, tmp_path / "controller", snapshot_version=3,
            scheduler_engine=replay_engine) as replayed:
        class NonIterableAttemptIndex(set):
            def __iter__(self):
                raise AssertionError("logical-attempt lookup must not iterate history")

        replayed._worker_historical_logical_attempts = NonIterableAttemptIndex(
            replayed._worker_historical_logical_attempts)
        assert replayed.worker_attempt_status(
            request_id=selection.proposal.proposal_id, plan_digest=prepared.plan.digest,
            lineage_id=f"driver:{issued.transition_id}",
            stage_id=f"runtime:{issued.transition_id}") == "unknown"


def test_lost_settlement_reply_retries_native_and_accounting_without_execution(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    original = controller.unified_driver_settle
    calls = {"settle": 0, "lifecycle": 0}
    real_run = lifecycle.run_stage

    def count_run(*args, **kwargs):
        calls["lifecycle"] += 1
        return real_run(*args, **kwargs)

    def lose_reply(value):
        result = original(value)
        calls["settle"] += 1
        if calls["settle"] == 1:
            raise OSError("reply lost after durable settlement")
        return result

    monkeypatch.setattr(lifecycle, "run_stage", count_run)
    monkeypatch.setattr(controller, "unified_driver_settle", lose_reply)
    try:
        with pytest.raises(de.DriverExecutionUncertain, match="settlement"):
            connector.execute(issued)
        assert engine.accounting_view().receipt_count == 1
        receipt = connector.execute(issued)
        assert receipt.settlement_receipt["status"] == "duplicate"
        assert calls == {"settle": 2, "lifecycle": 1}
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()


def test_settlement_verifier_refuses_caller_mutation_and_receipt_is_detached(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    try:
        receipt = connector.execute(driver.tick(now=1.0))
        restored = de.DriverExecutionReceipt.from_dict(receipt.to_dict())
        assert restored.to_dict() == receipt.to_dict()
        malformed = receipt.to_dict()
        malformed["settlement_request"]["outcome"] = "valid_comparison"
        malformed["receipt_digest"] = unified_driver._digest({
            key: value for key, value in malformed.items() if key != "receipt_digest"})
        with pytest.raises(de.DriverExecutionRefused, match="settlement request differs"):
            de.DriverExecutionReceipt.from_dict(malformed)
        changed = receipt.to_dict()["settlement_request"]
        changed["outcome"] = "valid_comparison"
        with pytest.raises(de.DriverExecutionRefused, match="differs from the owned"):
            connector._verify_settlement(changed)
        with pytest.raises(campaign_control.ControlRefused, match="conflicts"):
            controller.unified_driver_settle(changed)
        assert receipt.settlement_request["outcome"] == "invalid"
        assert engine.accounting_view().receipt_count == 1
        assert not any(thread.name == "autokernel-parent-evidence" and thread.is_alive()
                       for thread in __import__("threading").enumerate())
    finally:
        connector.close()
        controller.close()
