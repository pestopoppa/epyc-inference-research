"""Long-context surface (audit C1/C4) -- fakes only.

No model, no inference: the tokenizer is a code-point stub, the server is either an
injected `post` callable or a tiny local HTTP child that speaks the slot/completion
protocol with synthetic timings, and every A/B uses a mocked `_measure_once`.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import socket
import sys
from types import SimpleNamespace

import pytest

from . import actor_context, actors, cpu_profile, gates, longctx, longctx_tools
from . import resolved_recipe as rr, serial_run, serving
from .test_legacy_cpu_serving import FIXTURE_PARENT_WATCH
from .test_resolved_recipe import _artifacts, _policy
from .test_serving_residency import _proof, _sampler_class

DEPTH, TAIL = 128, 16


def _encode(text: str) -> list[int]:
    return [ord(char) % 50000 for char in text]


def _template(model="/models/lc-fixture.gguf") -> serving.Recipe:
    return serving.Recipe(name="lc-fixture", model=model, device="none", ngl=0, threads=4,
                          np=1, ctx=512, n_predict=8, temperature=0.0, top_k=1,
                          cpu_list=None)


def _launch(build: Path, *, port=18655, model="/models/lc-fixture.gguf", artifacts=None):
    base = _template(model)
    command = base.server_argv(build, port) + ["--no-mmap", "--no-webui"]
    template = rr.canonical_recipe_projection(name=base.name, command_argv=command,
                                              topology_prefix=(), n_predict=8,
                                              temperature=0.0, top_k=1)
    return rr.resolve_canonical_launch(
        template, build_dir=build, command_argv=command, topology_prefix=(),
        launch_environment={"LD_LIBRARY_PATH": str(build / "bin")},
        artifact_identities=artifacts or _artifacts(template, build=build), backend="cpu",
        environment_policy=_policy(), port=port, runtime_binary_dir=str(build / "bin"),
        runtime_ld_paths=(str(build / "bin"),),
        provenance={"export_sha256": "a" * 64, "instance_mode": "full",
                    "source:fixture": "b" * 64})


def _spec(tmp_path: Path, *, logs=()) -> longctx.Spec:
    corpus = " ".join(f"word{i}" for i in range(200))
    manifest_path = tmp_path / "lc.prompt-manifest.json"
    manifest, body = longctx_tools.build(
        template=_template(), target_id="lc-target", encode=_encode, corpus_text=corpus,
        fmt="raw", depth=DEPTH, tail=TAIL, version="lc-fixture-v1",
        manifest_path=manifest_path, production_logs=list(logs), source={"corpus": "synthetic"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec_path = tmp_path / "lc.longctx.json"
    spec_path.write_text(json.dumps(body), encoding="utf-8")
    return longctx.Spec.load(spec_path)


class FakeServer:
    """An in-process llama-server for `SurfaceLaunch.post`: slot files and a token cache."""

    def __init__(self, spec, *, a_rate=100.0, b_rate=20.0, identity_drift=False):
        self.spec, self.calls, self.cached, self.saved = spec, [], 0, {}
        self.a_rate, self.b_rate, self.identity_drift = a_rate, b_rate, identity_drift

    def __call__(self, port, path, body, timeout):
        request = json.loads(body)
        self.calls.append((path, body))
        if path.startswith("/slots/0?action=save"):
            self.saved[request["filename"]] = self.cached
            return json.dumps({"n_saved": self.cached, "n_written": 10}).encode()
        if path.startswith("/slots/0?action=restore"):
            if request["filename"] not in self.saved:
                raise longctx.LongCtxRefused("POST restore -> HTTP 400: file not found")
            self.cached = self.saved[request["filename"]]
            return json.dumps({"n_restored": self.cached,
                               "timings": {"restore_ms": 3.0}}).encode()
        prompt = request["prompt"]
        reused = self.cached if request["cache_prompt"] and len(prompt) > self.cached else 0
        self.cached = len(prompt)
        tokens = [7] * request["n_predict"]
        if self.identity_drift and len([c for c in self.calls if "restore" in c[0]]):
            tokens[-1] = 8
        return json.dumps({"stop": True, "tokens": tokens, "timings": {
            "prompt_n": len(prompt) - reused, "prompt_per_second": self.a_rate,
            "predicted_n": request["n_predict"], "predicted_per_second": self.b_rate}}).encode()


# ------------------------------------------------------------------ spec and launch

def test_spec_builds_from_corpus_and_refuses_tampering(tmp_path):
    spec = _spec(tmp_path)
    assert spec.depth == DEPTH and len(spec.tail) == TAIL
    assert spec.prefix + spec.probe == tuple(json.loads(spec.request_b[1])["prompt"])
    assert json.loads(spec.request_a)["prompt"] == list(spec.prefix + spec.tail)
    assert json.loads(spec.request_a)["n_predict"] == 1
    assert spec.body["ctx"] >= DEPTH + len(spec.probe) + 32
    assert spec.requests(_template()) == (spec.request_b,)
    body = dict(spec.body, tail=[1, 2, 3])
    (tmp_path / "bad.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(longctx.LongCtxRefused, match="digest"):
        longctx.Spec.load(tmp_path / "bad.json")
    body["digest"] = longctx.spec_digest(body)
    body["bounds"] = {"a_prompt_n_max": 1, "b_prompt_n_max": 999}
    body["digest"] = longctx.spec_digest(body)
    (tmp_path / "bad.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(longctx.LongCtxRefused, match="bound"):
        longctx.Spec.load(tmp_path / "bad.json")


def test_derived_launch_is_the_target_at_depth_with_a_slot_dir(tmp_path):
    spec = _spec(tmp_path)
    base = _launch(tmp_path / "anchor")
    long = longctx.derive_launch(base, spec, tmp_path / "slots")
    assert long.template.ctx == spec.body["ctx"]
    assert long.template.name == f"lc-fixture-lc0k-{spec.digest[:8]}"
    assert long.command_argv[-2:] == ("--slot-save-path", str((tmp_path / "slots").resolve()))
    assert long.command_argv[long.command_argv.index("-c") + 1] == str(spec.body["ctx"])
    assert long.template.recipe_hash != base.template.recipe_hash
    assert long.execution_digest != base.execution_digest
    assert spec.requests(long.template) == (spec.request_b,)
    surface = longctx.Surface(spec, store=tmp_path)
    name = surface.slot_name(long)
    assert name.startswith("a" * 16 + "-" + long.execution_digest[:16])
    wide = _launch(tmp_path / "anchor")
    wide = replace(wide, template=replace(wide.template, np=2))
    with pytest.raises(longctx.LongCtxRefused, match="np=1"):
        longctx.derive_launch(wide, spec, tmp_path / "slots")


# ------------------------------------------------------------------ per-launch protocol

def test_measure_launch_restores_then_a_then_restores_then_b(tmp_path):
    spec = _spec(tmp_path)
    server = FakeServer(spec)
    server.saved["s.bin"] = DEPTH
    launch = longctx.SurfaceLaunch(spec, "s.bin", post=server)
    warm = launch.serve(1, 0, "warmup", spec.request_b)
    measured = launch.serve(1, 0, "measurement", spec.request_b, capture=True)
    assert [path.split("?")[-1] if "?" in path else path for path, _ in server.calls] == [
        "action=restore", "/completion", "action=restore", "/completion"]
    assert server.calls[1][1] == spec.request_a and server.calls[3][1] == spec.request_b[1]
    assert warm[:3] == (1, 0.0, True) and warm[3]["error"] is None
    assert warm[3]["longctx"]["prompt_n"] == TAIL
    assert measured[:3] == (8, 20.0, True) and measured[3]["longctx"]["prompt_n"] == len(spec.probe)
    assert measured[4].request == spec.request_b[1]
    record = launch.launch_record([warm[3], measured[3]])
    assert record["a"]["prompt_per_second"] == 100.0 and record["b"]["predicted_per_second"] == 20.0
    assert record["a"]["restore"]["n_restored"] == DEPTH


def test_refusals_a_failed_restore_and_an_unreused_prefix(tmp_path):
    spec = _spec(tmp_path)
    server = FakeServer(spec)
    launch = longctx.SurfaceLaunch(spec, "missing.bin", post=server)
    row = launch.serve(1, 0, "measurement", spec.request_b)[3]
    assert "HTTP 400" in row["error"] and server.calls[-1][0].endswith("restore")
    server.saved["short.bin"] = 3                     # restored the wrong state
    row = longctx.SurfaceLaunch(spec, "short.bin", post=server).serve(
        1, 0, "measurement", spec.request_b)[3]
    assert "LONGCTX_REFUSED slot restore" in row["error"]

    def no_reuse(port, path, body, timeout):          # the server re-prefilled everything
        reply = json.loads(server(port, path, body, timeout))
        if path == "/completion":
            reply["timings"]["prompt_n"] = len(json.loads(body)["prompt"])
        return json.dumps(reply).encode()
    server.saved["s.bin"] = DEPTH
    row = longctx.SurfaceLaunch(spec, "s.bin", post=no_reuse).serve(
        1, 0, "measurement", spec.request_b)[3]
    assert "LONGCTX_REFUSED request B evaluated prompt_n" in row["error"]
    with pytest.raises(longctx.LongCtxRefused, match="foreign"):
        launch.serve(1, 0, "measurement", ("p", b"{}"))


def test_generation_saves_the_slot_and_runs_the_identity_gate_once(tmp_path):
    spec = _spec(tmp_path)
    receipt = tmp_path / "identity.json"
    server = FakeServer(spec)
    launch = longctx.SurfaceLaunch(spec, "s.bin", mode="generate",
                                   identity_receipt=str(receipt), post=server)
    row = launch.serve(1, 0, "warmup", spec.request_b)[3]
    assert row["error"] is None and server.saved["s.bin"] == DEPTH
    assert json.loads(server.calls[0][1])["prompt"] == list(spec.prefix)
    assert json.loads(receipt.read_text())["passed"] is True
    assert launch.serve(1, 0, "measurement", spec.request_b)[:3] == (0, 0.0, True)
    drift = longctx.SurfaceLaunch(spec, "t.bin", mode="generate", identity_receipt=str(receipt),
                                  post=FakeServer(spec, identity_drift=True))
    row = drift.serve(1, 0, "warmup", spec.request_b)[3]
    assert "identity FAILED" in row["error"]
    assert json.loads(receipt.read_text())["passed"] is False


# ------------------------------------------------------------------ verdict and A/B

def _records(rates):
    return [{"longctx": {"a": {"prompt_per_second": rate}}} for rate in rates]


def test_verdict_judges_decode_and_prefill_against_their_own_floors():
    floor = {"anchor_residency": _records([100, 101, 99, 100.5, 99.5] * 5),
             "candidate_residency": _records([100.2, 99.8, 100.1, 99.9, 100] * 5)}
    row = {"effect": 0.04, "decisive": True, "noise_floor_pct": 2.0,
           "anchor_residency": _records([100] * 5), "candidate_residency": _records([100.1] * 5)}
    verdict = longctx.verdict(row, floor, 5)
    assert verdict["passed"] is True and verdict["prefill_decisive"] is False
    row["candidate_residency"] = _records([80] * 5)
    verdict = longctx.verdict(row, floor, 5)
    assert verdict["passed"] is False and verdict["prefill_regressed"] is True
    row.update(candidate_residency=_records([100] * 5), effect=-0.05)
    assert longctx.verdict(row, floor, 5)["decode_regressed"] is True
    row["decisive"] = None
    assert longctx.verdict(row, floor, 5)["passed"] is False


def test_surface_compare_calibrates_the_target_floor_once_then_gates(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    surface = longctx.Surface(spec, store=tmp_path / "store")
    anchor = surface.launch_for(_launch(tmp_path / "anchor"))
    candidate = surface.launch_for(_launch(tmp_path / "candidate"))
    launches = []
    prefill = {"candidate": 100.5}

    def measure(recipe, build, port, *, longctx=None, evidence=None, **kwargs):
        launches.append((longctx.mode, Path(build).name))
        if longctx.mode == "generate":
            (surface.slot_dir / longctx.slot_filename).write_bytes(b"slot")
            Path(longctx.identity_receipt).write_text(json.dumps(
                {"schema": "epyc.autokernel.longctx_identity.v1", "passed": True}))
            return 0.0
        n = len(launches)
        rate = 100.0 + (n % 5) * 0.1 if Path(build).name == "anchor" else prefill["candidate"]
        evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                         "status": "not_applicable", "window_start": float(n),
                         "window_end": n + 0.5, "longctx": {"a": {"prompt_per_second": rate}}})
        return 20.0 + (n % 3) * 0.01 if Path(build).name == "anchor" else 22.0

    monkeypatch.setattr(serving, "_measure_once", measure)
    row = surface.compare(anchor, candidate, pairs=5, log=lambda *_: None)
    assert launches[0] == ("generate", "anchor")
    assert len(launches) == 1 + 48 + 10
    assert row["recipe"] == anchor.template.name and row["decisive"] is True
    assert row["longctx"]["passed"] is True and row["longctx"]["prefill_floor_pct"] > 0
    assert list(surface.floor_store.glob("serving-floor.*.json"))
    launches.clear()
    prefill["candidate"] = 70.0
    row = surface.compare(anchor, candidate, pairs=5, log=lambda *_: None)
    assert len(launches) == 10                      # slot and floor reused
    assert row["longctx"]["passed"] is False and row["longctx"]["prefill_regressed"] is True


def test_attention_routes_are_selected_by_route_name(monkeypatch):
    sgemm = SimpleNamespace(target_surface="ggml/src/ggml-cpu/llamafile/sgemm.cpp",
                            target_symbol="gemm4xN")
    assert longctx.attention_route(sgemm) is None
    fa = replace(gates.CPU_SOURCE_ROUTES[0], route="cpu_fa_schedule",
                 path="ggml/src/ggml-cpu/ops.cpp",
                 symbols=("ggml_compute_forward_flash_attn_ext_f16",))
    monkeypatch.setattr(gates, "CPU_SOURCE_ROUTES", (*gates.CPU_SOURCE_ROUTES, fa))
    hypothesis = SimpleNamespace(target_surface="ggml/src/ggml-cpu/ops.cpp",
                                 target_symbol="ggml_compute_forward_flash_attn_ext_f16")
    assert longctx.attention_route(hypothesis) == "cpu_fa_schedule"
    assert not any(longctx.ATTENTION_ROUTE.search(route.route)
                   for route in (*gates.CPU_SOURCE_ROUTES[:-1], *gates.CPU_MULTI_FILE_ROUTES))


# ------------------------------------------------------------------ real _measure_once hook

def _fake_server(build: Path) -> None:
    binary = build / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text(f'''#!{sys.executable}
{FIXTURE_PARENT_WATCH}
import http.server, json, sys
from pathlib import Path
argv = sys.argv
slots = Path(argv[argv.index("--slot-save-path") + 1])
state = {{"cached": 0}}
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, body, code=200):
        self.send_response(code); self.end_headers(); self.wfile.write(json.dumps(body).encode())
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "action=save" in self.path:
            (slots / request["filename"]).write_text(str(state["cached"]))
            return self.reply({{"n_saved": state["cached"]}})
        if "action=restore" in self.path:
            path = slots / request["filename"]
            if not path.is_file():
                return self.reply({{"error": "missing"}}, 400)
            state["cached"] = int(path.read_text())
            return self.reply({{"n_restored": state["cached"], "timings": {{"restore_ms": 1}}}})
        prompt = request["prompt"]
        reused = state["cached"] if len(prompt) > state["cached"] else 0
        state["cached"] = len(prompt)
        self.reply({{"stop": True, "tokens": [5] * request["n_predict"], "timings": {{
            "prompt_n": len(prompt) - reused, "prompt_per_second": 123.0,
            "predicted_n": request["n_predict"], "predicted_per_second": 17.0}}}})
class Server(http.server.HTTPServer): allow_reuse_address = True
Server(("127.0.0.1", int(argv[argv.index("--port") + 1])), Handler).serve_forever()
''')
    binary.chmod(0o700)
    (binary.parent / "libggml.so").write_bytes(b"fixture-not-a-loaded-DSO")


def test_measure_once_hook_with_a_local_fake_server(tmp_path, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    spec = _spec(tmp_path)
    build = tmp_path / "anchor"
    _fake_server(build)
    monkeypatch.setattr(serving.residency, "Sampler", _sampler_class(_proof(peak=0, median=0, kfd=0)))
    surface = longctx.Surface(spec, store=tmp_path / "store")
    arm = surface.launch_for(_launch(build, port=port))
    launch = surface.ensure_slot(arm)                         # real _measure_once, generate
    assert (surface.slot_dir / launch.slot_filename).read_text() == str(DEPTH)
    assert json.loads(next(surface.slot_dir.glob("identity-*.json")).read_text())["passed"]
    evidence = []
    value = serving._measure_once(arm.template, build, port, resolved_recipe=arm,
                                  frozen_requests=spec.requests(arm.template),
                                  evidence=evidence, longctx=launch)
    assert value == 17.0
    facts = evidence[-1]["longctx"]
    assert facts["a"]["prompt_n"] == TAIL and facts["a"]["prompt_per_second"] == 123.0
    assert facts["b"]["prompt_n"] == len(spec.probe) and facts["b"]["predicted_per_second"] == 17.0
    assert surface.ensure_slot(arm) == launch                  # cached: no relaunch


# ------------------------------------------------------------------ opt-out: unchanged

def test_targets_that_do_not_opt_in_launch_exactly_as_before(tmp_path, monkeypatch):
    """No `longctx` reaches `_measure_once` from compare/calibrate/perf capture unless
    passed, the planner prompt and target card are unchanged, and the resume binding
    ignores the flag."""
    spec = _spec(tmp_path)
    base = _launch(tmp_path / "anchor")
    candidate = _launch(tmp_path / "candidate")
    requests = spec.requests(base.template)
    seen = []

    def legacy_measure(recipe, build, port, *, evidence, resolved_recipe=None,
                       frozen_requests=None):                  # the pre-longctx signature
        seen.append(Path(build).name)
        evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                         "status": "not_applicable"})
        return 10.0

    monkeypatch.setattr(serving, "_measure_once", legacy_measure)
    serving.compare(base.template, Path(base.build_dir), Path(candidate.build_dir), pairs=1,
                    floor_pct=None, port=base.port, anchor_resolved_recipe=base,
                    candidate_resolved_recipe=candidate, frozen_requests=requests)
    serving.calibrate_floor(base.template, Path(base.build_dir), samples=2, port=base.port,
                            resolved_recipe=base, frozen_requests=requests)
    assert seen == ["anchor", "candidate", "anchor", "anchor"]

    class Stop(Exception):
        pass

    class Capture:
        budgets, cleanup_uncertain = {}, None
        def verify_version(self): return {}
        def _remaining(self, timeout): return 10
        def abort(self, reason): pass

    def perf_measure(recipe, build, port, *, boot_timeout_s, resolved_recipe, frozen_requests,
                     cpu_profile_capture):
        raise Stop
    monkeypatch.setattr(cpu_profile.CpuProfileCapture, "for_loop",
                        classmethod(lambda cls, *a, **k: Capture()))
    monkeypatch.setattr(serving, "_measure_once", perf_measure)
    (tmp_path / "store").mkdir()
    with pytest.raises(Stop):
        cpu_profile.profile_loop(base, spec.manifest, store_root=tmp_path / "store")

    context = {"target": {"recipe": {"backend": "cpu"}, "scope": "experimental"},
               "cpu_profile": {"status": "unavailable", "reason": "fixture"}}
    plain = actors.render_context(context)
    assert actors.LONG_CONTEXT_HEADER not in plain
    assert not any("long-context" in line for line in actor_context.target_card(context["target"]))
    histogram = {"servers": {"8074": {"requests": 2, "ctx_p50": 1, "ctx_p90": 2, "ctx_max": 3,
                                      "buckets": [{"bucket": "0k-8k", "requests": 2,
                                                   "decode_wall_share": 1.0,
                                                   "decode_tok_s": 40.0,
                                                   "prefill_wall_share": 1.0}]}}}
    opted = dict(context, long_context=longctx.planner_context(
        spec, histogram=histogram, cpu_profile=None, node_profile=None))
    opted["target"] = dict(context["target"], long_context=longctx.target_card(spec))
    rendered = actors.render_context(opted)
    assert actors.LONG_CONTEXT_HEADER in rendered and "| 0k-8k | 2 |" in rendered
    assert f"- long-context target depth (tokens): {DEPTH}" in actor_context.target_card(
        opted["target"])
    argv = ["--model", "/m.gguf", "--store", "/s"]
    assert serial_run.resume_binding(argv + ["--longctx-surface", "/x.json"]) == \
        serial_run.resume_binding(argv)


def test_run_wiring_is_guarded_by_the_opt_in_flag():
    """Every run.py entry into the surface sits behind `longctx_surface is not None`
    (`longctx_reprofile` returns first thing when it is None), so a target without
    `--longctx-surface` takes exactly the pre-existing measure/keep/profile/planner path."""
    source = (Path(__file__).parent / "run.py").read_text(encoding="utf-8").splitlines()
    entries = re.compile(r"longctx_compare\(worker|longctx_keep_gate\(worker,|"
                         r"longctx\.planner_context\(|longctx\.target_card\(")
    for index, line in enumerate(source):
        if entries.search(line) and "def " not in line and "row = longctx_compare" not in line:
            window = "\n".join(source[max(0, index - 3):index + 5])
            assert "longctx_surface is not None" in window, (index + 1, line)
    body = "\n".join(source)
    assert re.search(r"def longctx_reprofile\(\) -> None:\n(?:.*\n){1,4}?\s+if longctx_surface "
                     r"is None:\n\s+return", body)
    assert 'parser.add_argument("--longctx-surface", type=Path,' in body


def test_production_histogram_buckets_decode_wall_by_context(tmp_path):
    log = tmp_path / "llama-server-8074.log"
    lines = []
    for task, (ctx, eval_ms) in enumerate([(300, 1000.0), (40000, 3000.0), (150000, 6000.0)]):
        lines += [f"I slot print_timing: id  0 | task {task} | prompt eval time =  500.00 ms /"
                  f"   100 tokens (5 ms per token, 200 tokens per second)",
                  f"I slot print_timing: id  0 | task {task} |        eval time =  {eval_ms} ms /"
                  f"    50 tokens (20 ms per token, 50 tokens per second)",
                  f"I slot      release: id  0 | task {task} | stop processing: n_tokens = "
                  f"{ctx}, truncated = 0"]
    log.write_text("\n".join(lines) + "\n")
    body = longctx.parse_server_logs([log])
    server = body["servers"]["8074"]
    assert server["requests"] == 3 and server["ctx_max"] == 150000
    shares = {row["bucket"]: row["decode_wall_share"] for row in server["buckets"]}
    assert shares == {"0k-8k": 0.1, "32k-64k": 0.3, ">=128k": 0.6}
