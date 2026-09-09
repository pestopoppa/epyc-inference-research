"""Actual installed CPU producer with synthetic perf/HTTP children; no hardware tests."""
from contextlib import closing
import hashlib
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from unittest.mock import patch

import pytest

from . import cpu_profile as cp, native_model_preparation as mp, planned_serving as ps
from . import profile_preparation as pp, resolved_recipe as rr, scheduling as sched
from . import serving, standalone_inputs as si, unified_driver as ud
from . import measurement_capture as mc
from .test_profile_preparation_runtime import SyntheticSelectedProfileProvider, _start
from .test_standalone_inputs import _document, _verifier
from .test_unified_driver import runtime_driver, _prompt_manifest
from .test_unified_planner import runtime_anchor
from . import test_unified_planner as planner_fixture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path, *, perf_failure="", server_failure=""):
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"synthetic model bytes, not a GGUF model")
    dso = build / "bin/libggml.so"
    shutil.copyfile(Path("/lib/x86_64-linux-gnu/libm.so.6").resolve(), dso)
    events = tmp_path / "child-events"
    python = str(Path(sys.executable).resolve())
    server = build / "bin/llama-server"
    server.write_text(f"#!{python}\n" + f'''
import ctypes,json,os,sys,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
time.sleep(0.05)
ctypes.CDLL({str(dso)!r})
events=Path({str(events)!r})
with events.open('a') as f: f.write('server '+str(os.getpid())+'\\n')
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{{}}')
 def do_POST(self):
  req=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  assert set(req)=={{'prompt','n_predict','temperature','top_p','top_k','cache_prompt'}}
  assert req['cache_prompt'] is False
  time.sleep(0.04)
  data=b'bad-json' if {server_failure!r}=='json' else json.dumps({{'content':'synthetic',
   'stop':{server_failure!r}!='truncated','timings':{{'predicted_n':1 if {server_failure!r}=='eos' else req['n_predict'],
   'predicted_per_second':20.0}}}}).encode()
  self.send_response(500 if {server_failure!r}=='http' else 200);self.end_headers();self.wfile.write(data)
HTTPServer(('127.0.0.1',int(sys.argv[sys.argv.index('--port')+1])),Handler).serve_forever()
''')
    server.chmod(0o700)
    perf = tmp_path / "synthetic-perf"
    perf.write_text(f"#!{python}\n" + f'''
import json,os,select,signal,sys,time
from pathlib import Path
events=Path({str(events)!r})
with events.open('a') as f: f.write('perf '+str(os.getpid())+'\\n')
failure={perf_failure!r}
if sys.argv[1]=='--version':
 print('wrong' if failure=='version' else 'SYNTHETIC-test-only');sys.exit(0)
if sys.argv[1]=='script':
 data=json.loads(Path(sys.argv[sys.argv.index('-i')+1]).read_text())
 if failure=='script_stderr': sys.stderr.write('x'*200000);sys.stderr.flush()
 for stamp in data['times']:
  print(str(data['pid'])+'/'+str(data['pid'])+' '+format(stamp,'.9f')+': 100 cycles:u: 0000 synthetic_kernel ({str(dso)})')
 sys.exit(0)
if failure=='denied': sys.stderr.write('synthetic permission denied');sys.exit(7)
kind=sys.argv[1];pid=int(sys.argv[sys.argv.index('-p')+1])
output=Path(sys.argv[sys.argv.index('-o')+1])
ctl,ack=map(int,next(x for x in sys.argv if x.startswith('--control=fd:')).split(':')[1].split(','))
done=False;enabled=False;times=[]
def stop(*args):
 global done
 done=True
signal.signal(signal.SIGINT,stop)
while not done:
 if enabled: times.append(time.monotonic())
 ready,_,_=select.select([ctl],[],[],0.002)
 if ready:
  cmd=os.read(ctl,4096)
  if not cmd: break
  if failure=='verbose': os.write(2,b'x'*200000)
  if failure=='bad_ack': os.write(ack,b'nope\\n');continue
  if cmd==b'enable\\n': enabled=True
  if cmd==b'disable\\n':
   enabled=False
   if failure=='disable_ack': continue
  os.write(ack,b'ack\\n')
if kind=='record': output.write_text(json.dumps({{'pid':pid,'times':times}}))
else:
 output.write_text('\\n'.join(json.dumps({{'counter-value':'100','unit':'','event':event,
  'event-runtime':100,'pcnt-running':100}}) for event in {cp.EVENTS!r}))
''')
    perf.chmod(0o700)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    cpus = sorted(os.sched_getaffinity(0))[:4]
    template = serving.Recipe(name="synthetic-cpu-profile", model=str(model), device="none",
        ngl=0, cpu_list=",".join(map(str, cpus)), threads=4, np=1, ctx=128, n_predict=2)
    argv = template.server_argv(build, port)
    artifacts = {"model": {"schema": rr.ARTIFACT_SCHEMA, "role": "model", "path": str(model), "sha256": sha(model)},
        "drafter": None,
        "executable": {"schema": rr.ARTIFACT_SCHEMA, "role": "executable", "path": str(server), "sha256": sha(server)},
        "dsos": [{"schema": rr.ARTIFACT_SCHEMA, "role": "dso", "path": str(dso), "sha256": sha(dso)}]}
    recipe = rr.resolve_canonical_launch(template, build_dir=build, command_argv=argv[3:],
        topology_prefix=argv[:3], launch_environment=template.server_env(build, base={}),
        artifact_identities=artifacts, backend="cpu", port=port,
        environment_policy={"schema": rr.ENVIRONMENT_POLICY_SCHEMA, "version": "synthetic-v1",
            "measurement_keys": [], "allowed_inherit_keys": [], "witnesses": {}},
        runtime_binary_dir=str(build / "bin"), runtime_ld_paths=[str(build / "bin")],
        provenance={"export_sha256": "d"*64, "instance_mode": "full", "source:launcher": "e"*64})
    original_artifact = planner_fixture.artifact
    def issued_artifact(kind, ref, digit):
        value = original_artifact(kind, ref, digit)
        if kind == "build" and ref.endswith(":executable"):
            value.update(path=str(server), sha256=sha(server))
        return value
    # Author the actual executable in the fixture's original registry, before
    # resolve_manifest/target enrollment; never patch a verifier or settled row.
    with patch.object(planner_fixture, "artifact", issued_artifact):
        seed, _, resolved, target, target_digest = runtime_driver(git_source=True, recipe=recipe)
    document = _document(tmp_path)
    (tmp_path / "resolved.json").write_text(json.dumps(resolved.to_dict()))
    document["driver_config"].update({"runtime_anchors": {target_digest: runtime_anchor(target, recipe)},
        "runtime_dimensions": {}, "experiment_plans": {}, "execution_inputs": {}, "profiles": {}})
    from . import unified_planner as up
    recipe = up.prepare_runtime_anchors(resolved,
        document["driver_config"]["runtime_anchors"]).recipes[target_digest]
    arm = ps.arm_identity(template, recipe)
    loaded = {"target_revision_digest": target_digest, "model_digest": sha(model), "quantization": "Q8_0",
        "recipe_digest": recipe.snapshot_digest, "executable_digest": arm["executable_digest"],
        "dso_digest": arm["dso_set_digest"]}
    manifest_body = {"model_path": str(model), "files": [{"path": ".", "sha256": sha(model)}]}
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(json.dumps({"schema": "epyc.autokernel.model_identity.v1", **manifest_body}))
    prep = {"schema": mp.SPEC_SCHEMA, "target_revision_digest": target_digest,
        "recipe_execution_digest": recipe.execution_digest, "entry_path": str(model), "entry_sha256": sha(model),
        "inventory_identity": {"model_id": str(model), "model_manifest": str(manifest),
            "model_manifest_sha256": sha(manifest), "model_sha256": cp._digest(manifest_body)}}
    prep["preparation_digest"] = cp._digest(prep)
    prompts = _prompt_manifest(recipe).to_dict()
    prompts["prompts"] = prompts["prompts"][:1]
    prompts["digest"] = cp._digest({key: value for key, value in prompts.items() if key != "digest"})
    storage = tmp_path / "artifacts"
    storage.mkdir(mode=0o700)
    config = {"schema": cp.CONFIG_SCHEMA, "mode": cp.MODE, "loaded_identity": loaded,
        "resolved_recipe": recipe.to_dict(), "prompt_manifest": prompts, "model_preparation": prep,
        "profiler": {"path": str(perf), "sha256": sha(perf), "version": "SYNTHETIC-test-only",
            "server_interpreter": {"path": python, "sha256": sha(python)}},
        "source_closure": cp.source_identity(), "storage": str(storage),
        "budgets": {"max_stage_seconds": 20.0, "teardown_seconds": 2.0, "control_seconds": 0.5,
            "reduce_seconds": 2.0, "max_raw_file_bytes": 1024**2, "max_total_raw_bytes": 32*1024**2,
            "max_parser_bytes": 1024**2, "max_rows": 10000, "max_symbols": 16,
            "max_metadata_bytes": 4*1024**2}}
    config_file = tmp_path / "cpu-profile-config.json"
    config_file.write_bytes(cp._canonical(config))
    binding = cp.build_installed_cpu_profile_binding({"path": str(config_file), "sha256": sha(config_file)},
                                                    valid_for_seconds=120.0)
    stage = sched.StageProposal(proposal_id="profile:synthetic-cpu", submitted_at=1.0, backend="cpu",
        target_revision=target_digest, alias_identity=target.workload_signature,
        frontier_id=target_digest, production_frontier=True, seed_id=None, stage_class="prerequisite",
        estimated_duration_seconds=22.0, estimated_claims=sched.ResourceVector(1.0, (), 0),
        eligible=True, eligibility_ref="1"*64, reservation_kind=None, full_region=True,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    request = ud.ProfilePreparationRequest.from_dict({"schema": ud.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": target_digest, "stage_proposal": stage.to_dict(),
        "profile_contract": {"schema": ud.PROFILE_CONTRACT_SCHEMA, "adapter_id": cp.MECHANISM_ID,
                             "adapter_digest": binding.adapter_digest}})
    document["driver_config"]["profile_requests"] = {target_digest: request.to_dict()}
    document["manifest_digest"] = si._digest({key: value for key, value in document.items() if key != "manifest_digest"})
    materialized = si.materialize(si.StartupManifest.from_dict(document))
    containers = tmp_path / "containers"
    containers.mkdir()
    provider = SyntheticSelectedProfileProvider(containers, {target_digest: request})
    registry = si.ProviderRegistry({"fixture-lifecycle": si.ProviderBinding(lifecycle_provider=provider),
        "fixture-readiness": si.ProviderBinding(readiness_check=lambda: (True, None))},
        evidence_verifiers={"fixture-evidence": _verifier()},
        profile_bindings={cp.MECHANISM_ID: pp.InstalledProfilePreparationBinding({target_digest: binding})})
    return materialized, registry, target_digest, config, events, request


def assert_children_gone(events):
    if events.exists():
        for row in events.read_text().splitlines():
            assert not Path(f"/proc/{int(row.split()[1])}").exists(), row


def test_actual_installed_producer_to_settlement_planner_and_restart(tmp_path):
    materialized, registry, target, config, events, request = fixture(tmp_path)
    controller, runtime = _start(materialized, registry)
    try:
        outcome = runtime.tick()
        snapshot = controller.current_verified_profile_result(target)
        assert snapshot is not None, outcome
        assert snapshot["settlement"]["outcome"] == "prerequisite"
        profile = runtime.profile_executor.planner_profiles(time.monotonic())[target]
        assert profile.opportunities == ()
        assert profile.observation_states == ("unknown",)
        assert profile.hotspots and "synthetic_kernel" in profile.hotspots[0]
        with closing(mc.ArtifactStore(Path(config["storage"]))) as store:
            original = cp.reopen_capture(snapshot["profile_event"]["artifact_identity"],
                store=store, config=config, request=request.to_dict())
            # New content addresses cannot replace any original factual join.
            for changed_fact in ("request", "source", "counter_window", "mapping_device", "response", "raw"):
                altered = copy.deepcopy(cp._plain(original))
                if changed_fact == "request":
                    altered["request_digest"] = "0" * 64
                elif changed_fact == "source":
                    altered["source_closure"]["schema"] = "foreign"
                elif changed_fact == "counter_window":
                    altered["phases"][0]["counter_scope"]["enable_transition"]["sent"] = 1e100
                elif changed_fact == "mapping_device":
                    altered["phases"][0]["target_before"]["loaded_dso_mappings"][0]["artifact"]["dev"] += 1
                elif changed_fact == "response":
                    altered["phases"][0]["response"]["response_hex"] = b'{"stop":false,"timings":{"predicted_n":1}}'.hex()
                else:
                    altered["phases"][0]["script"]["sha256"] = "0" * 64
                forged = store.write("test-only-one-fact:" + changed_fact, altered)
                with pytest.raises(cp.CpuProfileRefused):
                    cp.reopen_capture(forged.to_dict(), store=store, config=config, request=request.to_dict())
        assert len(original["phases"]) == 2
        for phase in original["phases"]:
            controls = phase["counter_scope"]
            assert controls["enable_transition"]["sent"] <= controls["enable_transition"]["acknowledged"] <= phase["response"]["start"]
            assert phase["response"]["end"] <= controls["disable_transition"]["sent"] <= controls["disable_transition"]["acknowledged"]
            assert "not exact request" in controls["scope"]
            assert phase["target_before"]["loaded_dso_mappings"]
            assert phase["target_after"]["argv"] == phase["target_before"]["argv"]
        assert original["processes"]["server"]["ppid"] == original["processes"]["producer"]["pid"]
        assert "not exact CPU cost" in original["limitations"][0]
        before = events.read_text()
    finally:
        runtime.close()
        controller.close()
    assert_children_gone(events)

    # Reconstruct the scheduler from its original immutable startup input.
    restarted = si.materialize(si.StartupManifest.from_dict(materialized.manifest.to_dict()))
    controller, runtime = _start(restarted, registry)
    try:
        assert target in runtime.profile_executor.planner_profiles(time.monotonic())
        assert events.read_text() == before
    finally:
        runtime.close()
        controller.close()


@pytest.mark.parametrize("boundary", ["popen", "identity"])
def test_spawn_failure_keeps_fd_and_direct_child_cleanup_ownership(tmp_path, monkeypatch, boundary):
    _, _, _, config, _, request = fixture(tmp_path)
    created = []
    original_popen = cp.subprocess.Popen
    def launch(*args, **kwargs):
        if boundary == "popen":
            raise OSError("synthetic Popen refusal")
        child = original_popen(*args, **kwargs)
        created.append(child)
        return child
    def identity(pid):
        raise OSError("synthetic identity read refusal")
    with closing(mc.ArtifactStore(Path(config["storage"]))) as store:
        capture = cp.CpuProfileCapture(config=cp.CpuProfileConfig.from_dict(config), request=request.to_dict(), store=store)
        capture.target = {"pid": os.getpid()}
        before = set(os.listdir("/proc/self/fd"))
        monkeypatch.setattr(cp.subprocess, "Popen", launch)
        if boundary == "identity":
            monkeypatch.setattr(cp.wl, "process_identity", identity)
        with pytest.raises((OSError, cp.CpuProfileRefused)):
            capture._spawn("record", "warmup")
        assert not capture.active
        assert set(os.listdir("/proc/self/fd")) == before
        for child in created:
            assert child.poll() is not None
            assert not Path(f"/proc/{child.pid}").exists()


def test_insufficient_aggregate_is_refused_before_any_child(tmp_path):
    _, _, _, config, events, _ = fixture(tmp_path)
    config["budgets"]["max_total_raw_bytes"] = 2 * config["budgets"]["max_raw_file_bytes"]
    with pytest.raises(cp.CpuProfileRefused, match="aggregate"):
        cp.CpuProfileConfig.from_dict(config)
    assert not events.exists()


@pytest.mark.parametrize("failure", ["denied", "bad_ack", "disable_ack", "verbose", "script_stderr", "version"])
def test_tool_failures_retain_failed_terminal_cost_and_cleanup(tmp_path, failure):
    materialized, registry, target, config, events, request = fixture(tmp_path, perf_failure=failure)
    controller, runtime = _start(materialized, registry)
    try:
        runtime.tick()
        assert controller.current_verified_profile_result(target) is None
        assert not runtime.profile_executor.planner_profiles(time.monotonic())
        assert not controller._actor_profile_execution_reservations
        assert len(controller._actor_profile_execution_settlements) == 1
        held, terminal = next(iter(controller._actor_profile_execution_settlements.values()))
        assert terminal.return_code != 0
        assert held.ended_at > held.started_at
    finally:
        runtime.close()
        controller.close()
    assert_children_gone(events)


@pytest.mark.parametrize("failure", ["json", "http", "truncated"])
def test_bad_http_cannot_publish_profile(tmp_path, failure):
    materialized, registry, target, config, events, request = fixture(tmp_path, server_failure=failure)
    controller, runtime = _start(materialized, registry)
    try:
        runtime.tick()
        assert controller.current_verified_profile_result(target) is None
    finally:
        runtime.close()
        controller.close()
    assert_children_gone(events)


def test_natural_eos_is_completed_full_request_not_fixed_generated_length(tmp_path):
    materialized, registry, target, config, events, request = fixture(tmp_path, server_failure="eos")
    controller, runtime = _start(materialized, registry)
    try:
        runtime.tick()
        snapshot = controller.current_verified_profile_result(target)
        assert snapshot is not None
        with closing(mc.ArtifactStore(Path(config["storage"]))) as store:
            original = cp.reopen_capture(snapshot["profile_event"]["artifact_identity"],
                store=store, config=config, request=request.to_dict())
        assert [phase["observed_predicted_n"] for phase in original["phases"]] == [1, 1]
        requested = rr.resolved_recipe_from_dict(config["resolved_recipe"]).template.n_predict
        assert requested > 1
        assert [json.loads(bytes.fromhex(phase["response"]["request_hex"]))["n_predict"]
                for phase in original["phases"]] == [requested, requested]
    finally:
        runtime.close()
        controller.close()
    assert_children_gone(events)
