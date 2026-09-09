"""Hermetic original HTTP byte capture; no model, broker grant, or kernel execution."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import time
import urllib.error

import pytest

from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_server_response as ns
from . import observation_binding as ob
from . import planned_serving as ps
from . import serving
from .test_native_parent_evidence import _case
from .test_planned_serving import _plan, _prompts, _recipes, Provider
from .test_resolved_recipe import _resolve
from .test_serving_residency import _proof, _sampler_class


@pytest.fixture
def original(tmp_path):
    case = _case(tmp_path)
    ctx = case["context"]
    requests = ctx.prompts.requests(ctx.unit.expected_prompt_ids, ctx.template)
    capture = ns.ServerResponseCapture(store=case["store"], plan=ctx.plan, unit=ctx.unit,
        fence=ctx.fence, recipe=ctx.recipe, prompts=ctx.prompts, frozen_requests=requests)
    rows = []
    for phase in ns.PHASES:
        for index, (name, request) in enumerate(requests):
            response = json.dumps({"content": f"answer {index}", "tokens": [index, index + 1],
                "stop": True, "timings": {"predicted_n": 2, "predicted_per_second": 25.0}}).encode()
            rows.append(ns.RawServerResponse(phase, index, name, request, response, 1.5, 2.5, None))
    receipt = capture.seal(rows, process_pid=101,
        request_started_monotonic_s=1.0, request_ended_monotonic_s=3.0)
    case.update(capture=capture, rows=rows, receipt=receipt, requests=requests)
    yield case
    case["store"].close()


def reopen(case, receipt=None):
    return ns.reopen_unit(case["receipt"] if receipt is None else receipt,
        store=case["store"], expected_frame=case["capture"].frame,
        expected_requests=case["requests"], expected_pid=101)


def test_original_every_phase_slot_request_response_byte_reopens(original):
    rows = reopen(original)
    assert len(rows) == 4
    for retained, raw in zip(rows, original["rows"]):
        assert bytes.fromhex(retained["raw"]["request_hex"]) == raw.request
        assert bytes.fromhex(retained["raw"]["response_hex"]) == raw.response
        assert retained["response"]["tokens"] == tuple(json.loads(raw.response)["tokens"])
        assert "seed" not in retained["request"]  # v1 absence is not repaired on read
    assert all(row["implementation_status"] == row["configuration_status"] == "pinned"
               for row in original["receipt"]["source_identity"]["callables"])
    assert not any("status" in item for item in rows)


@pytest.mark.parametrize("field", ["plan_digest", "unit_id", "arm", "process_generation_id",
    "prompt_manifest_digest", "prompt_ids", "fence", "recipe", "loaded_instrument"])
def test_one_parent_frame_fact_mutation_refuses(original, field):
    body = ob._plain(original["receipt"])
    body["frame"][field] = "foreign"
    with pytest.raises(ns.ServerResponseRefused, match="parent frame"):
        reopen(original, body)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "duplicate", "pid", "interval", "source"])
def test_unit_receipt_one_fact_refuses(original, mutation):
    body = ob._plain(original["receipt"])
    if mutation == "missing":
        body["responses"].pop()
    elif mutation == "extra":
        body["responses"].append(body["responses"][0])
    elif mutation == "reordered":
        body["responses"].reverse()
    elif mutation == "duplicate":
        body["responses"][1] = body["responses"][0]
    elif mutation == "pid":
        body["process_pid"] += 1
    elif mutation == "interval":
        body["retention_started_monotonic_s"] = 0
    elif mutation == "source":
        body["source_identity"]["max_slots"] += 1
    with pytest.raises((ns.ServerResponseRefused, mc.CaptureError)):
        reopen(original, body)


@pytest.mark.parametrize("field", ["request_hex", "response_hex", "sequence", "phase", "prompt_id"])
def test_resealed_one_raw_fact_refuses(original, field):
    body = ob._plain(original["receipt"])
    ref = body["responses"][0]
    raw = ob._plain(original["store"].read(ref["locator"], ref["sha256"]))
    raw[field] = 19 if field == "sequence" else "00" if field.endswith("hex") else "foreign"
    changed = original["store"].write("server-response:0", raw)
    body["responses"][0] = changed.to_dict()
    with pytest.raises(ns.ServerResponseRefused):
        reopen(original, body)


def test_repeat_seal_and_unbounded_or_mutable_raw_refuse(original):
    with pytest.raises(ns.ServerResponseRefused, match="already sealed"):
        original["capture"].seal(original["rows"], process_pid=101,
            request_started_monotonic_s=1.0, request_ended_monotonic_s=3.0)
    row = original["rows"][0]
    for value in (bytearray(row.request), b"x" * (ns.MAX_REQUEST_BYTES + 1)):
        with pytest.raises(ns.ServerResponseRefused):
            replace(row, request=value)


def test_aggregate_worst_case_refuses_before_requests(original, monkeypatch):
    ctx = original["context"]
    monkeypatch.setattr(ns, "MAX_TOTAL_RAW_BYTES", 1)
    with pytest.raises(ns.ServerResponseRefused, match="aggregate raw-byte budget"):
        ns.ServerResponseCapture(store=original["store"], plan=ctx.plan, unit=ctx.unit,
            fence=ctx.fence, recipe=ctx.recipe, prompts=ctx.prompts,
            frozen_requests=original["requests"])


@pytest.mark.parametrize("failure", ["oversized", "invalid_json", "http_error"])
def test_failed_http_retains_explicit_invalid_evidence_and_closes(original, monkeypatch, failure):
    ctx = original["context"]
    capture = ns.ServerResponseCapture(store=original["store"], plan=ctx.plan, unit=ctx.unit,
        fence=ctx.fence, recipe=ctx.recipe, prompts=ctx.prompts,
        frozen_requests=original["requests"])
    streams, terminated = [], []
    class Process:
        pid = 101
        def poll(self): return None
        def terminate(self): terminated.append(True)
        def wait(self, _timeout): return 0
    def urlopen(request, **_kwargs):
        raw = (b"x" * (ns.MAX_RESPONSE_BYTES + 1) if failure == "oversized" else
               b"not-json" if failure == "invalid_json" else b'{"error":"unavailable"}')
        stream = io.BytesIO(b"" if isinstance(request, str) else raw)
        streams.append(stream)
        if not isinstance(request, str) and failure == "http_error":
            raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, stream)
        return stream
    monkeypatch.setattr(serving.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(serving.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(serving, "verify_env_readback", lambda *a, **k: None)
    monkeypatch.setattr(serving.residency, "Sampler", _sampler_class(_proof(peak=0, kfd=0)))
    observed = []
    with pytest.raises(serving.ServerDied, match="slots failed"):
        serving._measure_once(ctx.template, ctx.recipe.build_dir, ctx.recipe.port,
            resolved_recipe=ctx.recipe, frozen_requests=original["requests"],
            observation=observed, response_capture=capture)
    assert terminated == [True] and all(stream.closed for stream in streams)
    retained = ns.reopen_unit(observed[0]["server_responses"], store=original["store"],
        expected_frame=capture.frame, expected_requests=original["requests"], expected_pid=101)
    assert len(retained) == 4
    assert all(row["raw"]["error"] for row in retained)
    assert all(not row["terminal"] and row["error"] for row in observed[0]["requests"])
    if failure == "oversized":
        assert all(row["raw"]["response_hex"] is None for row in retained)
    elif failure == "invalid_json":
        assert all(row["response"] is None for row in retained)
    else:
        assert all(row["response"]["error"] == "unavailable" for row in retained)


def test_actual_selected_request_mutation_refuses_before_launch(original, monkeypatch):
    ctx = original["context"]
    requests = list(original["requests"])
    requests[0] = (requests[0][0], b"{}")
    monkeypatch.setattr(serving.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ns.ServerResponseRefused, match="selected requests"):
        serving._measure_once(ctx.template, ctx.recipe.build_dir, ctx.recipe.port,
            resolved_recipe=ctx.recipe, frozen_requests=requests, response_capture=original["capture"])


def test_incomplete_selected_recorder_identity_refuses_before_plan_issue(tmp_path, monkeypatch):
    identity = lo.callable_identity
    def incomplete(value):
        row = identity(value)
        if value is ns.reopen_unit:
            row = {**row, "implementation_status": "unproven", "implementation_sha256": None}
        return row
    monkeypatch.setattr(lo, "callable_identity", incomplete)
    with pytest.raises(ob.ObservationBindingError, match="response producer identity is incomplete"):
        ob.loaded_planned_serving_identity(measurement_callable=serving._measure_once,
            fence_clock=time.monotonic, serving_timer=time.time)


@pytest.mark.parametrize("mode", ["missing", "changed"])
def test_old_instrument_cannot_be_repinned_with_current_response_producer(original, mode):
    from .. import schemas
    frame = ob._plain(original["capture"].frame)
    reference = ob.LoadedInstrumentReference.from_dict(frame["loaded_instrument"])
    identity = ob._plain(original["store"].read(reference.artifact.locator, reference.artifact.sha256))
    if mode == "missing":
        del identity["used_constants"]["server_response_source"]
    else:
        identity["used_constants"]["server_response_source"]["max_total_raw_bytes"] += 1
    identity.pop("sha256")
    identity["sha256"] = schemas.content_hash(identity)
    stored = original["store"].write(f"loaded-instrument:{identity['sha256']}", identity)
    frame["loaded_instrument"] = ob.LoadedInstrumentReference(
        identity["sha256"], True, stored).to_dict()
    with pytest.raises(ns.ServerResponseRefused, match="original write-time response producer"):
        ns._instrument_source(original["store"], frame)


def test_actual_controller_child_socket_http_capture_and_restart(tmp_path, monkeypatch):
    from pathlib import Path
    import socket
    from . import native_parent_service as service
    from . import native_parent_receipt_replay as replay
    from . import native_capture_control as nc
    from . import resolved_recipe as rr
    from . import test_unified_driver as driver_fixtures
    from .test_driver_execution import _run_real_controller_child_v2_capture_and_restart
    text_writer = Path.write_text
    fixture_patch = pytest.MonkeyPatch()
    services = []
    registries, scopes = [], []
    # Select before enrollment/plan freeze; never probe the fixture's shared :8000.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        fixture_port = reservation.getsockname()[1]
    assert fixture_port != 8000
    canonical_recipe = driver_fixtures.canonical_recipe

    def isolated_recipe(**kwargs):
        recipe = canonical_recipe(**kwargs)
        command = list(recipe.command_argv)
        command[command.index("--port") + 1] = str(fixture_port)
        return rr.resolve_canonical_launch(recipe.template, build_dir=recipe.build_dir,
            command_argv=command, topology_prefix=recipe.topology_prefix,
            launch_environment=dict(recipe.launch_env), artifact_identities={
                "model": recipe.model.to_dict(),
                "drafter": None if recipe.drafter is None else recipe.drafter.to_dict(),
                "executable": recipe.executable.to_dict(),
                "dsos": [item.to_dict() for item in recipe.dsos]},
            backend=recipe.backend, environment_policy=recipe.environment_policy,
            port=fixture_port, runtime_binary_dir=recipe.runtime_binary_dir,
            runtime_ld_paths=recipe.runtime_ld_paths, provenance=dict(recipe.provenance))

    replayer = replay.NativeParentReceiptReplayer()
    validator_init = nc.NativeCaptureValidator.__init__
    def patched_validator(self, *args, **kwargs):
        validator_init(self, *args, **{**kwargs, "parent_receipt_replayer": replayer})

    def fixture_writer(path, text, *args, **kwargs):
        if path == tmp_path / "fixture_measure.py":
            text = _CONTAINED_HTTP_FIXTURE
        return text_writer(path, text, *args, **kwargs)

    def producer_factory(authority, prepared, lifecycle, configuration):
        root = tmp_path / "fixture-probe"
        registry = replay.IssuedNativeEvidenceRegistry(artifact_root=prepared.artifact_root,
            max_units=len(prepared.plan.expected_units))
        scope = replayer.using(registry)
        scope.__enter__()
        registries.append(registry)
        scopes.append(scope)
        producer = service.NativeParentEvidenceService(authority, prepared, lifecycle,
            configuration, registry=registry, runtime_probe=lo.FilesystemProbe(proc_root=root / "proc",
                sysfs_cpu_root=root / "cpu", boot_id_path=root / "boot", cgroup_root=root / "cgroup"))
        services.append(producer)
        return producer

    fixture_patch.setattr(Path, "write_text", fixture_writer)
    fixture_patch.setattr(nc.NativeCaptureValidator, "__init__", patched_validator)
    fixture_patch.setattr(driver_fixtures, "canonical_recipe", isolated_recipe)
    try:
        _run_real_controller_child_v2_capture_and_restart(
            tmp_path, monkeypatch, producer_type=producer_factory)
        producer = services[0]
        assert producer.stopped
        assert len(registries[0]._entries) == len(producer.prepared.plan.expected_units)
        assert len(producer._unit_producers) == len(producer.prepared.plan.expected_units)
        store = mc.ArtifactStore(producer.prepared.artifact_root)
        try:
            for entry in registries[0]._entries.values():
                unit = producer._unit_producers[entry.context.unit_id]
                assert unit.context.recipe.port == fixture_port
                assert unit.context.recipe.command_argv[
                    unit.context.recipe.command_argv.index("--port") + 1] == str(fixture_port)
                assert unit._result.completion.stage_witnesses["correctness"].status == "unknown"
                ref = entry.request["native_observation"]
                native = store.read(ref["locator"], ref["sha256"])
                selected = native["selected_observation"]
                frame = ns._frame(unit.context.plan, unit.context.unit, unit.context.fence,
                                  unit.context.recipe, unit.context.prompts)
                requests = unit.context.prompts.requests(unit.context.unit.expected_prompt_ids,
                                                        unit.context.template)
                retained = ns.reopen_unit(selected["server_responses"], store=store,
                    expected_frame=frame, expected_requests=requests,
                    expected_pid=unit.context.descendant_event["data"]["process"]["pid"])
                assert len(retained) == 2 * len(requests)
                assert tuple((row["raw"]["phase"], row["raw"]["slot_index"], row["raw"]["prompt_id"])
                    for row in retained) == tuple((phase, index, name) for phase in ns.PHASES
                        for index, (name, _body) in enumerate(requests))
                assert all(row["raw"]["error"] is None and row["response"]["stop"] is True
                           for row in retained)
                assert all(len(row["response"]["tokens"]) == row["request"]["n_predict"]
                           for row in retained)
                assert all(row["response"]["content"] == row["request"]["prompt"] for row in retained)
                expected_pid = unit.context.descendant_event["data"]["process"]["pid"]
                assert all(type(row["response"]["server_pid"]) is int
                           and row["response"]["server_pid"] == expected_pid for row in retained)
        finally:
            store.close()
    finally:
        for scope in reversed(scopes):
            scope.__exit__(None, None, None)
        fixture_patch.undo()


_CONTAINED_HTTP_FIXTURE = r'''
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock
from autokernel.loop import serving, worker_lifecycle as wl

HTTP_SERVER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        request = json.loads(body)
        raw = json.dumps({'content': request['prompt'], 'stop': True, 'server_pid': os.getpid(),
            'tokens': list(range(request['n_predict'])),
            'timings': {'predicted_n': request['n_predict'], 'predicted_per_second': 10.0}}).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(raw)))
        self.end_headers(); self.wfile.write(raw)
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""

def observed_measure(template, build_dir, port, **kwargs):
    session = kwargs['observation_session']
    actual_popen = subprocess.Popen
    def popen(_argv, **_kwargs):
        server = actual_popen([sys.executable, '-c', HTTP_SERVER, str(port)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cpus = {int(value) for value in os.environ['AUTOKERNEL_FIXTURE_CPUS'].split(',')}
        os.sched_setaffinity(server.pid, cpus)
        identity = wl.process_identity(server.pid)
        with Path(os.environ['AUTOKERNEL_FIXTURE_PID_LOG']).open('a', encoding='ascii') as stream:
            stream.write(f'{identity.pid} {identity.start_ticks}\n')
        root = Path(os.environ['AUTOKERNEL_FIXTURE_PROC_ROOT']) / str(server.pid)
        (root / 'fd').mkdir(parents=True, exist_ok=True)
        fields = ['0'] * 40
        fields[0], fields[11], fields[12] = 'S', '1', '0'
        fields[19], fields[36] = str(identity.start_ticks), str(min(cpus))
        (root / 'stat').write_text(f'{server.pid} (fixture-http) ' + ' '.join(fields) + '\n')
        (root / 'status').write_text('Cpus_allowed_list:\t' + ','.join(str(value) for value in sorted(cpus))
            + '\nMems_allowed_list:\t0-1\n')
        container = session.context['worker_binding']['container_identity']['path']
        (root / 'cgroup').write_text(f'0::{container}\n')
        (root / 'numa_maps').write_text('00400000 default kernelpagesize_kB=2048 N0=2\n')
        (root / 'smaps_rollup').write_text('Rss: 40 kB\nAnonHugePages: 4 kB\nShmemPmdMapped: 0 kB\nFilePmdMapped: 8 kB\n')
        (root / 'maps').write_text('')
        return server
    class Sampler:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        @property
        def proof(self):
            return {'peak_vram_bytes': 0, 'median_vram_bytes': 0, 'vram_reads': 0,
                'peak_kfd_processes': 0, 'sclk_min_mhz': 0, 'sclk_max_mhz': 0,
                'clock_stable': True, 'samples': 0, 'resident': False}
    with mock.patch.object(serving.subprocess, 'Popen', popen), \
            mock.patch.object(serving.residency, 'Sampler', Sampler), \
            mock.patch.object(serving, 'verify_env_readback', lambda *a, **k: None):
        return serving._measure_once(template, build_dir, port, **kwargs)
'''


def test_real_http_raw_capture_persists_after_requests_before_owned_teardown(tmp_path, monkeypatch):
    sent, returned, events = [], [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
        def do_POST(self):
            request = self.rfile.read(int(self.headers["Content-Length"]))
            sent.append(request)
            parsed = json.loads(request)
            response = (" {\n" + json.dumps({"content": parsed["prompt"], "tokens": [11, 12],
                "stop": True, "timings": {"predicted_n": parsed["n_predict"],
                "predicted_per_second": 25.0}})[1:]).encode()
            returned.append(response)
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    store = mc.ArtifactStore(tmp_path / "responses")
    class Process:
        pid = 4321
        def poll(self):
            return None
        def terminate(self):
            events.append("teardown")
        def wait(self, _timeout):
            return 0
    class Observation:
        shutdown_resolved = True
        def start(self, _phase): pass
        def phase(self, phase): events.append(phase)
        def attach_target(self, _pid): pass
        def checkpoint(self, label): events.append(label)
        def finish(self): pass
    write = store.write
    def tracked_write(namespace, body):
        if namespace.startswith("server-response:"):
            assert "measurement_end" in events and "teardown" not in events
            events.append("persist")
        return write(namespace, body)
    monkeypatch.setattr(store, "write", tracked_write)
    monkeypatch.setattr(serving.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(serving, "verify_env_readback", lambda *a, **k: None)
    monkeypatch.setattr(serving.residency, "Sampler", _sampler_class(_proof(peak=0, kfd=0)))
    try:
        at, ct, _a, _c = _recipes()
        anchor = _resolve(at, backend="cpu", port=http.server_port)
        candidate = _resolve(ct, backend="cpu", policy=_a.environment_policy.to_dict(), port=http.server_port)
        plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
        from . import experiment_plan as ep
        instrument = ob.seal_loaded_instrument(store=store, measurement_callable=serving._measure_once,
            fence_clock=time.monotonic, serving_timer=time.time)
        body = plan.to_dict()
        body.update(schema=ep.PLAN_SCHEMA_V2, loaded_instrument=instrument.to_dict(),
            anchor_identity=ps.arm_identity(at, anchor, loaded_instrument=instrument.to_dict()),
            candidate_identity=ps.arm_identity(ct, candidate, loaded_instrument=instrument.to_dict()))
        body.pop("digest", None)
        plan = ep.ExperimentPlan.from_dict(body)
        unit = plan.expected_units[0]
        requests = prompts.requests(unit.expected_prompt_ids, at)
        fence = Provider().admit(plan.digest, unit, ps.STAGES)
        capture = ns.ServerResponseCapture(store=store, plan=plan, unit=unit, fence=fence,
            recipe=anchor, prompts=prompts, frozen_requests=requests)
        observations = []
        value = serving._measure_once(at, anchor.build_dir, anchor.port,
            resolved_recipe=anchor, frozen_requests=requests, observation=observations,
            observation_session=Observation(), response_capture=capture)
        assert value == 50.0  # original sum of per-slot server rates, not wall throughput
        unit_receipt = observations[0]["server_responses"]
        reopened = ns.reopen_unit(unit_receipt, store=store, expected_frame=capture.frame,
            expected_requests=requests, expected_pid=4321)
        assert len(reopened) == len(sent) == len(returned) == 4
        assert sorted(bytes.fromhex(row["raw"]["response_hex"]) for row in reopened) == sorted(returned)
        assert sorted(bytes.fromhex(row["raw"]["request_hex"]) for row in reopened) == sorted(sent)
        assert events.index("measurement_end") < events.index("persist") < events.index("teardown")
        assert all("seed" not in row["request"] for row in reopened)
    finally:
        store.close()
        http.shutdown()
        http.server_close()
        thread.join(2)
        assert not thread.is_alive()
