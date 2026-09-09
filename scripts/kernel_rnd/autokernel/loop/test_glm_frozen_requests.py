"""Original GLM request transport; HTTP observations are fixtures, never a model run."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading

import pytest

from .. import schemas
from . import planned_serving as ps, resolved_recipe as rr, serving
from .test_planned_serving import _prompts
from .test_resolved_recipe import BUILD, _artifacts, _policy
from .test_serving_residency import _proof, _sampler_class


def _request():
    # Original depth3/48-thread client request, not the rejected depth2 experiment:
    # glm53-validation-20260908/artifact/run/
    # evidence-profile-fixed2029-mtp-t48-20260909T065605Z/probe/decode/request.json
    return json.loads((Path(__file__).parent / "fixtures/glm53-cached-decode-request.json").read_text())


def _manifest(request, *, schema=ps.PROMPT_SCHEMA_V2):
    row = {"prompt_id": "glm-fixed2029", "request": request,
           "request_digest": hashlib.sha256(ps._canonical(request)).hexdigest()}
    body = {"schema": schema, "version": "glm-fixed2029-depth3", "prompts": [row]}
    return ps.FrozenPromptManifest.from_dict({**body, "digest": schemas.content_hash(body)})


def _recipe():
    return serving.Recipe(name="glm-cpu-depth3", model="/models/GLM-5.3-Flash-00001-of-00006.gguf",
        device="none", ngl=0, threads=48, cpu_list="0-95", np=1, ctx=8192,
        n_predict=512, temperature=0.0, top_k=1,
        spec_decode={"type": "draft-mtp", "draft_n_max": 3},
        extra_flags=("--spec-draft-p-min", "0", "--reasoning", "off", "--no-mmap"))


def _canonical_launch(port):
    command = _recipe().server_argv(BUILD, port)[3:] + ["--no-webui", "-lv", "4"]
    prefix = ["numactl", "--interleave=all", "--", "taskset", "-c", "0-95"]
    template = rr.canonical_recipe_projection(name="glm-cpu-depth3", command_argv=command,
        topology_prefix=prefix, n_predict=512, temperature=0.0, top_k=1)
    resolved = rr.resolve_canonical_launch(template, build_dir=BUILD, command_argv=command,
        topology_prefix=prefix, launch_environment={"LD_LIBRARY_PATH": str(BUILD / "bin")},
        artifact_identities=_artifacts(template), backend="cpu", environment_policy=_policy(),
        port=port, runtime_binary_dir=str(BUILD / "bin"), runtime_ld_paths=(str(BUILD / "bin"),),
        provenance={"export_sha256": "a" * 64, "instance_mode": "full", "source:fixture": "b" * 64})
    return template, resolved


def test_exact_token_request_roundtrip_immutable_and_legacy_unchanged():
    request = _request()
    assert len(request["prompt"]) == 2029 and request["seed"] == 42
    manifest = _manifest(request)
    expected = ps._canonical(request)
    assert manifest.requests(("glm-fixed2029",), _recipe()) == (("glm-fixed2029", expected),)
    assert "top_p" not in json.loads(expected)
    assert ps.FrozenPromptManifest.from_dict(manifest.to_dict()) == manifest
    request["prompt"][0] = 0
    assert manifest.prompts[0].body == expected
    assert isinstance(manifest.prompts[0].prompt, tuple)
    old = _prompts(serving.Recipe("old-gpu", "/m"))
    assert old.schema == ps.PROMPT_SCHEMA
    assert ps.FrozenPromptManifest.from_dict(old.to_dict()).to_dict() == old.to_dict()
    assert not any(key in json.loads(old.prompts[0].body)
                   for key in ("seed", "stream", "return_tokens", "ignore_eos"))
    with pytest.raises(ps.PlannedServingError):
        _manifest(_request(), schema=ps.PROMPT_SCHEMA)
    row = old.to_dict()
    row["prompts"][0]["cache_prompt"] = True
    with pytest.raises(ps.PlannedServingError, match="cache_prompt=false"):
        ps.FrozenPromptManifest.from_dict(row)


@pytest.mark.parametrize("key,value", [
    ("prompt", []), ("prompt", [True]), ("prompt", [-1]), ("prompt", [2 ** 31]),
    ("prompt", [1.5]), ("seed", True), ("seed", -1), ("seed", 2 ** 32),
    ("cache_prompt", 1), ("ignore_eos", 1), ("return_tokens", 1), ("stream", True),
    ("temperature", float("inf")), ("top_p", None), ("unknown", "no"),
])
def test_closed_v2_request_negatives(key, value):
    request = _request()
    request[key] = value
    with pytest.raises((ps.PlannedServingError, ValueError)):
        _manifest(request)


def test_request_identity_covers_cache_seed_tokens_and_workload():
    baseline = _manifest(_request())
    for key, value in (("seed", 43), ("cache_prompt", False), ("prompt", [154822, 154824])):
        request = _request()
        request[key] = value
        changed = _manifest(request)
        assert changed.digest != baseline.digest
        assert changed.prompts[0].request_digest != baseline.prompts[0].request_digest
    with pytest.raises(ps.PlannedServingError, match="workload"):
        baseline.requests(("glm-fixed2029",), replace(_recipe(), n_predict=1))
    row = baseline.to_dict()
    row["prompts"][0]["request"]["seed"] = 43
    with pytest.raises(ps.PlannedServingError, match="request digest"):
        ps.FrozenPromptManifest.from_dict(row)


@pytest.mark.parametrize("flag,value", [("--spec-draft-p-min", "nan"),
    ("--spec-draft-p-min", "-0.1"), ("--spec-draft-p-min", "1.1"),
    ("--spec-draft-p-min", "no"), ("-lv", "-1"), ("-lv", "1.5")])
def test_glm_canonical_flag_values_refuse(flag, value):
    _, original = _canonical_launch(18311)
    argv = list(original.command_argv)
    argv[argv.index(flag) + 1] = value
    with pytest.raises(rr.ResolutionError):
        rr.canonical_recipe_projection(name="bad-glm", command_argv=argv,
                                       topology_prefix=original.topology_prefix)


def test_native_request_byte_bound_and_legacy_row_cannot_smuggle_options(monkeypatch):
    from . import native_server_response
    monkeypatch.setattr(native_server_response, "MAX_REQUEST_BYTES", 64)
    with pytest.raises(ps.PlannedServingError, match="bounded|bound"):
        _manifest(_request())
    old = _prompts(serving.Recipe("old-gpu", "/m")).to_dict()
    old["prompts"][0]["seed"] = 42
    with pytest.raises(ps.PlannedServingError, match="unknown"):
        ps.FrozenPromptManifest.from_dict(old)


def test_actual_http_payload_and_cpu_depth3_launch_unchanged(monkeypatch):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"stop": True, "tokens": [42],
                "timings": {"predicted_n": 512, "predicted_per_second": 9.0}}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    seen = {}
    class Process:
        pid = 4321
        def poll(self):
            return None

        def terminate(self):
            seen["terminated"] = True

        def wait(self, timeout):
            return 0

    def popen(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs["env"]
        return Process()

    try:
        port = server.server_address[1]
        recipe, resolved = _canonical_launch(port)
        manifest = _manifest(_request())
        requests = manifest.requests(("glm-fixed2029",), recipe)
        monkeypatch.setattr(serving.subprocess, "Popen", popen)
        monkeypatch.setattr(serving.residency, "Sampler", _sampler_class(_proof(peak=0, median=0, kfd=0)))
        monkeypatch.setattr(serving, "verify_env_readback", lambda *_a, **_k: None)
        ledger, observations = [], []
        assert serving._measure_once(recipe, BUILD, port, resolved_recipe=resolved,
            frozen_requests=requests, observation=ledger, evidence=observations) == 9.0
        assert received == [requests[0][1], requests[0][1]]  # original warmup then measurement
        argv = seen["argv"]
        assert argv == list(resolved.argv)
        for flag, value in (("-ngl", "0"), ("--device", "none"), ("-t", "48"),
                            ("-np", "1"), ("--spec-type", "draft-mtp"),
                            ("--spec-draft-n-max", "3"), ("--spec-draft-p-min", "0")):
            assert argv[argv.index(flag) + 1] == value
        assert argv[:6] == ["numactl", "--interleave=all", "--", "taskset", "-c", "0-95"]
        assert "--no-mmap" in argv and "--no-webui" in argv
        assert "-md" not in argv and seen["terminated"]
        request_rows = ledger[0]["requests"]
        assert [row["phase"] for row in request_rows] == ["warmup", "measurement"]
        assert all(row["request_sha256"] == manifest.prompts[0].request_digest for row in request_rows)
        assert observations[0]["gpu_residency"] == serving.RESIDENCY_NOT_APPLICABLE
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
