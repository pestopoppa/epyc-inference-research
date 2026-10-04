"""Local-model actor calls through the orchestrator passthrough (`actor_passthrough`,
operator 2026-10-04). Offline: no orchestrator and no model server is contacted -- the
opencode wire test answers from recording mocks on ephemeral ports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from autokernel.loop import actor_passthrough as route
from autokernel.loop import actor_serving as serving
from autokernel.loop import actors
from autokernel.loop import run as run_mod

GPU = "http://127.0.0.1:8083/v1"
CPU = "http://127.0.0.1:8074/v1"
HOSTED = "https://api.deepseek.com/v1"
ORCH = "http://127.0.0.1:59999"   # never contacted: only ever written into configs


def _on(spec: str = "on", url: str = ORCH):
    return mock.patch.dict(os.environ, {route.ENV: spec, route.URL_ENV: url})


def _off():
    patcher = mock.patch.dict(os.environ, {}, clear=False)
    patcher.start()
    os.environ.pop(route.ENV, None)
    return patcher


def _params(window=65_536) -> serving.ServingParams:
    return serving.ServingParams(base_url=GPU, idle_timeout_ms=2_500_000,
                                 context_window=window, compact_at=int(window * 0.76),
                                 prefill_tps=400.0, prefill_samples=5, ctx_source="props",
                                 idle_source="measured")


class Mapping_(unittest.TestCase):

    def test_parse_roles(self):
        self.assertEqual(route.parse_roles(""), {})
        self.assertEqual(route.parse_roles("off"), {})
        self.assertEqual(route.parse_roles("on"), route.DEFAULT_ROLES)
        self.assertEqual(route.parse_roles("qwen-gpu=architect_critic, x.y=worker_general"),
                         {"qwen-gpu": "architect_critic", "x.y": "worker_general"})
        for bad in ("qwen-gpu", "qwen-gpu=Arch", "=a", "a=b,a=c"):
            with self.assertRaises(ValueError, msg=bad):
                route.parse_roles(bad)

    def test_wire_url(self):
        env = {route.ENV: "on", route.URL_ENV: ORCH + "/"}
        self.assertEqual(route.wire_url("qwen-gpu", GPU, env),
                         ORCH + "/v1/passthrough/architect_critic")
        self.assertEqual(route.wire_url("qwen-local", CPU, env),
                         ORCH + "/v1/passthrough/architect_general")
        self.assertIsNone(route.wire_url("deepseek", HOSTED, env))           # hosted
        self.assertIsNone(route.wire_url("qwen-gpu", GPU, {}))               # off
        already = ORCH + "/v1/passthrough/architect_critic"
        self.assertIsNone(route.wire_url("other", already, env))             # pre-routed
        with self.assertRaises(route.UnroutedLocalProvider):
            route.wire_url("llama-local", "http://localhost:8099/v1", env)
        self.assertEqual(route.orchestrator_url({}), route.DEFAULT_URL)

    def test_codex_override_and_cpu_conflict(self):
        self.assertEqual(route.codex_base_url_override("local", ORCH + "/v1/passthrough/r"),
                         ["-c", f'model_providers.local.base_url="{ORCH}/v1/passthrough/r"'])
        routed = {"qwen-gpu/m": "w1", "qwen-local/m": "w2"}
        urls = {"qwen-gpu/m": GPU, "qwen-local/m": CPU}
        regions = {GPU: frozenset(), CPU: frozenset({"q0", "q1", "q2", "q3"})}
        self.assertEqual(route.cpu_conflict(routed, urls.get, regions.get), ["qwen-local/m"])

    def test_unknown_topology_fails_closed(self):
        with mock.patch("autokernel.loop.claim._ensure_orchestrator_importable",
                        side_effect=RuntimeError("no orchestrator")):
            self.assertTrue(route.orchestrator_cpu_regions(CPU))


class SeatWiring(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()
        self.addCleanup(_off().stop)

    def _patched(self, base_url, params=None):
        for p in (mock.patch.object(actors, "_provider_base_url", return_value=base_url),
                  mock.patch.object(serving, "resolve", return_value=params or _params())):
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _config(env) -> dict:
        return json.loads(Path(env["OPENCODE_CONFIG"]).read_text())

    def test_plain_seat_local_provider_is_routed_with_f1_from_the_backing_server(self):
        self._patched(GPU)
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        seat = actors.ActorSeat(bounded=False, serving_f1=True, context_limit=180_224,
                                output_limit=8_192)
        with _on(), mock.patch.object(serving, "resolve", return_value=_params()) as resolved:
            env = actors._seat_call(seat, backend, "planner", self.ws, {})
        resolved.assert_called_once_with(GPU)      # F1 reads the BACKING server
        options = self._config(env)["provider"]["qwen-gpu"]["options"]
        self.assertEqual(options["baseURL"], ORCH + "/v1/passthrough/architect_critic")
        self.assertEqual(options["headers"], {"X-Client-Id": route.CLIENT_ID})
        self.assertEqual(options["headerTimeout"], 2_500_000 + route.LOCK_WAIT_ALLOWANCE_MS)
        self.assertEqual(options["chunkTimeout"], 2_500_000)
        recorded = json.loads(env[actors.SEAT_ENV_SERVING])
        self.assertEqual(recorded["route"]["role"], "architect_critic")
        self.assertEqual(recorded["route"]["backing_url"], GPU)
        self.assertEqual(recorded["f1"]["context_window"], 65_536)

    def test_routing_without_f1_still_routes(self):
        self._patched(CPU)
        backend = actors.backend_for("qwen-local/qwen3.8-flash-next", "high")
        with _on():
            env = actors._seat_call(actors.ActorSeat(bounded=False), backend, "critic",
                                    self.ws, {})
            no_seat = actors._seat_call(None, backend, "critic", self.ws, {})
        for e in (env, no_seat):
            options = self._config(e)["provider"]["qwen-local"]["options"]
            self.assertEqual(options["baseURL"], ORCH + "/v1/passthrough/architect_general")
            self.assertEqual(options["headerTimeout"],
                             serving.CLIENT_DEFAULT_IDLE_MS + route.LOCK_WAIT_ALLOWANCE_MS)

    def test_bounded_seat_is_routed(self):
        self._patched(GPU)
        planner = actors.AgentPlanner(
            workspace=self.ws, backend=actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
            seat=actors.ActorSeat(bounded=True, serving_f1=True, planner_think_budget=8000))
        with _on(), actors._scratch_scope(self.ws).scope("call", "planner"):
            _backend, env = planner._seated("planner", {})
            entry = self._config(env)["provider"]["qwen-gpu"]
        self.assertEqual(entry["options"]["baseURL"], ORCH + "/v1/passthrough/architect_critic")
        self.assertEqual(entry["models"]["qwen3.8-27b"]["options"]["thinking_budget_tokens"], 8000)

    def test_hosted_and_cli_backends_are_untouched(self):
        self._patched(HOSTED)
        seat = actors.ActorSeat(bounded=False)
        backend = actors.backend_for("deepseek/deepseek-v4-flash", "max")
        with mock.patch.dict(os.environ, {}):
            os.environ.pop(route.ENV, None)
            off = actors._seat_call(seat, backend, "planner", self.ws, {})
        with _on():
            on = actors._seat_call(seat, backend, "planner", self.ws, {})
            for model in ("gpt-6.1-sol", "claude-opus-5-5"):
                cli = actors.backend_for(model, "high")
                self.assertEqual(actors._seat_serving(seat, cli, "planner", {"context_limit": 1}),
                                 ({"context_limit": 1}, {}, {}))
                self.assertIsNone(actors._seat_call(seat, cli, "planner", self.ws, {}))
        self.assertEqual(self._config(on), self._config(off))
        self.assertNotIn(actors.SEAT_ENV_SERVING, on)

    def test_off_is_byte_identical(self):
        self._patched(GPU)
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        seat = actors.ActorSeat(bounded=False, serving_f1=True)
        env = actors._seat_call(seat, backend, "planner", self.ws, {})
        self.assertNotIn("baseURL", json.dumps(self._config(env)))
        self.assertNotIn("route", json.loads(env[actors.SEAT_ENV_SERVING]))

    def test_unrouted_local_provider_is_refused(self):
        self._patched("http://127.0.0.1:8099/v1")
        backend = actors.backend_for("other-local/m", "high")
        with _on(), self.assertRaises(route.UnroutedLocalProvider):
            actors._seat_call(actors.ActorSeat(bounded=False), backend, "planner", self.ws, {})


class SchemaRepair(unittest.TestCase):

    def test_repair_turn_goes_through_the_passthrough(self):
        class Resp:
            def read(self):
                return json.dumps({"choices": [{"message": {"content": '{"paths": []}'}}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"
            ws.mkdir(parents=True)
            with _on(), mock.patch.object(actors, "_provider_base_url", return_value=GPU), \
                    mock.patch("urllib.request.urlopen", return_value=Resp()) as opened:
                actors._schema_repair("edited nothing", schema=actors.PATHS_SCHEMA,
                                      backend=actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
                                      workspace=ws)
            request = opened.call_args.args[0]
            self.assertEqual(request.full_url,
                             ORCH + "/v1/passthrough/architect_critic/chat/completions")
            self.assertEqual(request.get_header("X-client-id"), route.CLIENT_ID)


class RunFlags(unittest.TestCase):

    def test_apply_routing(self):
        env: dict[str, str] = {route.ENV: "stale"}
        ns = argparse.Namespace(actor_local_via_orchestrator="off")
        self.assertEqual(run_mod._apply_actor_routing(ns, [], env), {})
        self.assertNotIn(route.ENV, env)       # off clears an inherited knob
        self.assertEqual(run_mod._apply_actor_routing(argparse.Namespace(), [], env), {})
        backends = [actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
                    actors.backend_for("gpt-6.1-sol", "high"),
                    actors.backend_for("claude-opus-5-5", "high")]
        ns = argparse.Namespace(actor_local_via_orchestrator="on",
                                actor_local_orchestrator_roles=route.DEFAULT_ROLES_SPEC)
        env[route.URL_ENV] = ORCH
        with mock.patch.object(actors, "_provider_base_url", return_value=GPU):
            routed = run_mod._apply_actor_routing(ns, backends, env)
        self.assertEqual(routed, {"qwen-gpu/qwen3.8-27b": ORCH + "/v1/passthrough/architect_critic"})
        self.assertEqual(env[route.ENV], route.DEFAULT_ROLES_SPEC)
        with mock.patch.object(actors, "_provider_base_url", return_value="http://localhost:8099/v1"), \
                self.assertRaises(ValueError):
            run_mod._apply_actor_routing(
                ns, [actors.backend_for("other-local/m", "high")], env)

    def test_cli_defaults_off(self):
        self.assertIn("--actor-local-via-orchestrator", open(run_mod.__file__).read())


# -------------------------------------------------------------------------------------
# The wire: the installed opencode, per-call config from the seat, against two recording
# mocks -- the rebased "raw port" one (the harness's) and one standing in for :8000.
# -------------------------------------------------------------------------------------
from autokernel.loop import test_actor_author_thinking as wire  # noqa: E402


class _Orchestrator(wire._Recorder):
    bodies: list = []

    def do_POST(self):
        type(self).bodies.append({"method": "HEADERS", "path": self.path,
                                  "client": self.headers.get("X-Client-Id")})
        super().do_POST()


@unittest.skipUnless(wire.OPENCODE and wire.REAL_GLOBAL.is_file()
                     and GPU in wire.REAL_GLOBAL.read_text(errors="replace"),
                     "needs the installed opencode and the host's qwen-gpu global config")
class OpencodeWire(unittest.TestCase):

    def test_per_call_base_url_sends_the_call_to_the_passthrough(self):
        from http.server import ThreadingHTTPServer
        _Orchestrator.bodies = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Orchestrator)
        port = server.server_address[1]
        self.assertNotIn(port, (8000, 8074, 8083))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with _on(url=f"http://127.0.0.1:{port}"):
                records, config, err = wire.run_opencode_against_mock(
                    "planner", actors.ActorSeat(bounded=False), timeout_s=60)
        finally:
            server.shutdown()
            server.server_close()
        self.assertTrue(config["provider"]["qwen-gpu"]["options"]["baseURL"].endswith(
            "/v1/passthrough/architect_critic"))
        routed = [r for r in _Orchestrator.bodies if r["method"] == "HEADERS"]
        self.assertTrue(routed, f"nothing reached the passthrough mock; stderr: {err}")
        self.assertTrue(all(r["path"] == "/v1/passthrough/architect_critic/chat/completions"
                            for r in routed), routed)
        self.assertTrue(all(r["client"] == route.CLIENT_ID for r in routed), routed)
        self.assertFalse(wire._chat_bodies(records), "the raw-port mock got a chat completion")


if __name__ == "__main__":
    unittest.main()
