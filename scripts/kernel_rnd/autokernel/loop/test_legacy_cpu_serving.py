"""Existing comparator, real local HTTP children; synthetic rates, not model evidence."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import pytest

from . import resolved_recipe as rr, serving
from .test_resolved_recipe import _artifacts, _policy, _resolve
from .test_serving_residency import _proof, _sampler_class


def _requests():
    return tuple((f"prompt-{slot}", json.dumps({
        "prompt": f"original workload {slot}", "n_predict": 8, "temperature": 0.0,
        "top_k": 1, "seed": 17, "cache_prompt": False, "ignore_eos": True,
        "return_tokens": True, "stream": False}, separators=(",", ":")).encode())
        for slot in range(2))


def _server(build, template, port, rate):
    binary = build / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    raw_log, pid_log = build / "requests.jsonl", build / "pids"
    binary.write_text(f'''#!{sys.executable}
import http.server, json, os, sys
from pathlib import Path
Path({str(pid_log)!r}).open("a").write(str(os.getpid()) + "\\n")
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        with open({str(raw_log)!r}, "a") as stream: stream.write(json.dumps(raw.hex()) + "\\n")
        request = json.loads(raw)
        body = json.dumps({{"stop": True, "timings": {{"predicted_n": request["n_predict"],
            "predicted_per_second": {rate!r}}}}}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
class Server(http.server.HTTPServer): allow_reuse_address = True
Server(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler).serve_forever()
''')
    binary.chmod(0o700)
    dso = binary.parent / "libggml.so"
    dso.write_bytes(b"fixture-not-a-loaded-DSO")
    artifacts = _artifacts(template, build=build)
    for key in ("model", "executable"):
        artifacts[key]["sha256"] = hashlib.sha256(Path(artifacts[key]["path"]).read_bytes()).hexdigest()
    artifacts["dsos"][0]["sha256"] = hashlib.sha256(dso.read_bytes()).hexdigest()
    resolved = rr.resolve_canonical_launch(template, build_dir=build,
        command_argv=template.server_argv(build, port), topology_prefix=(),
        launch_environment=template.server_env(build, base={}),
        artifact_identities=artifacts, backend="cpu", environment_policy=_policy(),
        port=port, runtime_binary_dir=None, runtime_ld_paths=(),
        provenance={"export_sha256": "1" * 64, "instance_mode": "full", "source:fixture": "synthetic"})
    return resolved, raw_log, pid_log


def test_actual_cpu_http_comparison_and_request_bound_calibration(tmp_path, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    model = tmp_path / "not-a-model"
    model.write_bytes(b"no inference is performed")
    recipe = serving.Recipe(name="cpu-http", model=str(model), device="none", ngl=0,
                            np=2, n_predict=8, cpu_list=None)
    anchor, a_log, a_pids = _server(tmp_path / "anchor", recipe, port, 10.0)
    candidate, c_log, c_pids = _server(tmp_path / "candidate", recipe, port, 12.0)
    monkeypatch.setattr(serving.residency, "Sampler", _sampler_class(_proof(peak=0, median=0, kfd=0)))
    requests = _requests()
    floor = serving.calibrate_floor(recipe, Path(anchor.build_dir), samples=2, port=port,
                                    resolved_recipe=anchor, frozen_requests=requests)
    path = serving.write_floor(tmp_path / "floors", recipe, floor, frozen_requests=requests)
    reading = serving.load_floor(tmp_path / "floors", recipe, frozen_requests=requests)
    assert reading.verified and reading.path == path
    assert reading.request_digest == serving.request_digest(recipe, requests)
    row = serving.compare(recipe, Path(anchor.build_dir), Path(candidate.build_dir), pairs=1,
        floor_pct=reading.floor_pct, floor_request_digest=reading.request_digest, port=port,
        anchor_resolved_recipe=anchor, candidate_resolved_recipe=candidate, frozen_requests=requests)
    assert row["anchor_samples"] == [20.0] and row["candidate_samples"] == [24.0]
    assert row["effect"] == pytest.approx(0.2) and row["decisive"] is True
    assert row["request_digest"] == reading.request_digest == row["floor_request_digest"]
    assert row["residency"]["gpu_residency"] == serving.RESIDENCY_NOT_APPLICABLE
    assert row["residency"]["cpu_placement"] == "unproven"
    expected = sorted(body.hex() for _, body in requests)
    for log, launches in ((a_log, 3), (c_log, 1)):
        bodies = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(bodies) == launches * 2 * recipe.np
        assert sorted(bodies) == sorted(expected * (launches * 2))
    for log, count in ((a_pids, 3), (c_pids, 1)):
        pids = [int(line) for line in log.read_text().splitlines()]
        assert len(pids) == count
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)


def test_forwarding_refuses_partial_or_mismatched_arm_before_any_measurement(monkeypatch):
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=2)
    anchor = _resolve(recipe, backend="cpu", build=Path("/a"))
    candidate = _resolve(recipe, backend="cpu", build=Path("/c"))
    def forbidden(*args, **kwargs):
        pytest.fail("invalid arm pairing reached a measurement")
    monkeypatch.setattr(serving, "_measure_once", forbidden)
    with pytest.raises(serving.RecipeError, match="both"):
        serving.compare(recipe, Path("/a"), Path("/c"), pairs=1, floor_pct=None,
                        anchor_resolved_recipe=anchor)
    with pytest.raises(rr.ResolutionError):
        serving.compare(recipe, Path("/a"), Path("/wrong"), pairs=1, floor_pct=None,
                        anchor_resolved_recipe=anchor, candidate_resolved_recipe=candidate)
    with pytest.raises(serving.ServingFloorMismatch, match="request"):
        serving.compare(recipe, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0,
                        frozen_requests=_requests())


def test_custom_requests_cannot_borrow_or_overwrite_legacy_floor(tmp_path):
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=2)
    requests = _requests()
    legacy = serving.write_floor(tmp_path, recipe, {"floor_pct": 1.0})
    before = legacy.read_bytes()
    assert serving.load_floor(tmp_path, recipe, frozen_requests=requests).floor_pct is None
    assert serving.floor_path(tmp_path, recipe, frozen_requests=requests) != legacy
    row = {"floor_pct": 2.0, "recipe_hash": recipe.recipe_hash,
           "request_digest": serving.request_digest(recipe, requests)}
    path = serving.write_floor(tmp_path, recipe, row, frozen_requests=iter(requests))
    assert legacy.read_bytes() == before
    changed = ((requests[0][0], requests[0][1] + b" "), requests[1])
    assert serving.load_floor(tmp_path, recipe, frozen_requests=changed).floor_pct is None
    with pytest.raises(serving.ServingFloorMismatch):
        serving.write_floor(tmp_path, recipe, row, frozen_requests=changed)
    with pytest.raises(serving.ServingFloorMismatch):
        serving.write_floor(tmp_path, recipe, row)
    row.pop("request_digest")
    path.write_text(json.dumps(row))
    with pytest.raises(serving.ServingFloorMismatch):
        serving.load_floor(tmp_path, recipe, frozen_requests=requests)


def test_uncalibrated_custom_requests_still_measure_without_decisive_keep(monkeypatch):
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=2)
    requests = _requests()
    calls = []
    def measure(recipe, build, port, **kwargs):
        calls.append(kwargs["frozen_requests"])
        return 10.0 if build == Path("/a") else 12.0
    monkeypatch.setattr(serving, "_measure_once", measure)
    row = serving.compare(recipe, Path("/a"), Path("/c"), pairs=1, floor_pct=None,
                          frozen_requests=requests)
    assert row["decisive"] is None and row["floor_request_digest"] is None
    assert calls == [requests, requests]
