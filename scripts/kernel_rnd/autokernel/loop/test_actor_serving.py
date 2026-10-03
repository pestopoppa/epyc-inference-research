"""UFH14-B1 F1: client serving parameters derived per local server (`actor_serving`), and
their wiring into the opencode actor seats. Offline: no server is contacted (`/props` is
injected or `resolve` is patched, serving records are fixture lines)."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from autokernel.loop import actor_opencode_config as seat_config
from autokernel.loop import actor_serving as serving
from autokernel.loop import actors
from autokernel.loop import run as run_mod

LOCAL = "http://127.0.0.1:8083/v1"


def _record(port=8083, prompt_n=40_000, prompt_ms=80_000.0, cache_n=0, **extra) -> str:
    rec = {"schema": "epyc.orchestrator.serving_call.v1",
           "server": {"base_url": f"http://localhost:{port}", "port": port},
           "timings": {"prompt_n": prompt_n, "prompt_ms": prompt_ms, "cache_n": cache_n}}
    rec.update(extra)
    return json.dumps(rec)


def _props(n_ctx=262_144) -> dict:
    return {"default_generation_settings": {"n_ctx": n_ctx}, "total_slots": 4}


class Facts(unittest.TestCase):

    def test_locality_and_root(self):
        self.assertTrue(serving.is_local("http://127.0.0.1:8083/v1"))
        self.assertTrue(serving.is_local("http://localhost:8074/v1"))
        self.assertFalse(serving.is_local("https://api.deepseek.com/v1"))
        self.assertFalse(serving.is_local(None))
        self.assertEqual(serving.server_root("http://127.0.0.1:8083/v1/"), "http://127.0.0.1:8083")
        self.assertEqual(serving.port_of(LOCAL), 8083)

    def test_per_request_ctx_reads_the_slot_n_ctx_like_context_limits(self):
        self.assertEqual(serving.per_request_ctx(_props(262_144)), 262_144)
        self.assertIsNone(serving.per_request_ctx({}))
        self.assertIsNone(serving.per_request_ctx({"default_generation_settings": {"n_ctx": 0}}))
        self.assertIsNone(serving.per_request_ctx(None))

    def test_samples_keep_only_long_prefills_on_this_port(self):
        lines = [_record(), _record(port=8070), _record(prompt_n=4_000, prompt_ms=4_000.0),
                 "not json", json.dumps([1]),
                 _record(timings=None, result={"prompt_tokens": 30_000, "cached_prompt_tokens": 5_000,
                                               "prompt_eval_ms": 50_000.0}),
                 json.dumps({"caller": {"port": 8083},
                             "timings": {"prompt_n": 20_000, "prompt_ms": 40_000.0}})]
        samples = serving.prefill_samples(8083, lines)
        self.assertEqual([s.prompt_n for s in samples], [40_000, 25_000, 20_000])
        self.assertEqual(samples[1].ctx, 30_000)
        self.assertAlmostEqual(samples[0].rate, 500.0)
        self.assertEqual(serving.prefill_samples(None, lines), [])

    def test_window_rate_is_scaled_to_the_window_and_low_quantile(self):
        few = serving.prefill_samples(8083, [_record(), _record()])
        self.assertIsNone(serving.window_prefill_tps(few, 262_144))
        samples = [serving.PrefillSample(80_000, 80_000 / 487 * 1000, 80_000)] * 3
        rate = serving.window_prefill_tps(samples, 157_000)
        # 487 tok/s at 80k predicts ~350 at 157k (F12 measured ~340).
        self.assertAlmostEqual(rate, 487 * (80_000 / 157_000) ** 0.5, places=3)
        self.assertGreater(rate, 330)
        self.assertLess(rate, 360)
        # Enough samples near the window: the far ones are dropped (no 3x under-prediction).
        mixed = ([serving.PrefillSample(2_000, 2_000 / 850 * 1000, 2_000)] * 5
                 + [serving.PrefillSample(150_000, 150_000 / 340 * 1000, 150_000)] * 3)
        self.assertGreater(serving.window_prefill_tps(mixed, 157_000), 320)
        # A sample at or above the window is not scaled up.
        big = [serving.PrefillSample(200_000, 400_000.0, 200_000)] * 3
        self.assertAlmostEqual(serving.window_prefill_tps(big, 100_000), 500.0)

    def test_idle_timeout_measured_fallback_and_clamps(self):
        self.assertEqual(serving.idle_timeout_s(None, 300.0), (14_400, "fallback_unmeasured"))
        self.assertEqual(serving.idle_timeout_s(262_144, None), (14_400, "fallback_unmeasured"))
        seconds, source = serving.idle_timeout_s(262_144, 263.0)
        self.assertEqual(source, "measured")
        self.assertEqual(seconds, -(-2 * 262_144 / 263.0 * 1.25 // 1))
        self.assertGreater(seconds, 300)   # far above the 300 s client defaults (D1)
        self.assertEqual(serving.idle_timeout_s(8_192, 5_000.0)[0], serving.MIN_IDLE_S)
        self.assertEqual(serving.idle_timeout_s(1_000_000, 10.0)[0], serving.MAX_IDLE_S)


class Derive(unittest.TestCase):

    def test_hosted_providers_get_nothing(self):
        self.assertIsNone(serving.derive("https://api.deepseek.com/v1", props=_props()))
        self.assertIsNone(serving.resolve("https://api.deepseek.com/v1"))

    def test_measured_server(self):
        lines = [_record(prompt_n=150_000, prompt_ms=150_000 / 340 * 1000)] * 5
        params = serving.derive(LOCAL, props=_props(262_144), record_lines=lines)
        self.assertEqual(params.context_window, 262_144)
        self.assertEqual(params.ctx_source, "props")
        self.assertEqual(params.compact_at, int(262_144 * 0.76))
        self.assertEqual(params.idle_source, "measured")
        self.assertEqual(params.prefill_samples, 5)
        self.assertGreater(params.idle_timeout_ms, 300_000)
        self.assertLessEqual(params.idle_timeout_ms, serving.MAX_IDLE_S * 1000)
        self.assertEqual(params.to_dict()["base_url"], LOCAL)

    def test_f12_shape_reproduces_the_proven_compaction(self):
        params = serving.derive(LOCAL, n_ctx=196_608, record_lines=[])
        # F12 ran compaction at 150k on a 196,608 window.
        self.assertAlmostEqual(params.compact_at, 150_000, delta=1_000)
        self.assertEqual(params.idle_timeout_ms, 14_400_000)  # unmeasured: the ceiling
        self.assertEqual(params.idle_source, "fallback_unmeasured")
        self.assertEqual(params.ctx_source, "given")

    def test_unknown_window_leaves_context_alone(self):
        params = serving.derive(LOCAL, props={}, record_lines=[])
        self.assertIsNone(params.context_window)
        self.assertIsNone(params.compact_at)
        self.assertEqual(params.idle_timeout_ms, 14_400_000)

    def test_records_are_read_from_the_orchestrator_log_and_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / serving.SERVING_RECORDS_REL
            log.parent.mkdir(parents=True)
            log.write_text("\n".join([_record()] * 2) + "\n")
            Path(str(log) + ".1").write_text(_record() + "\n")
            with mock.patch.dict("os.environ", {serving.ORCHESTRATOR_ROOT_ENV: tmp,
                                                serving.SERVING_RECORDS_ENV: ""}):
                self.assertEqual(len(serving.serving_record_paths()), 2)
                params = serving.derive(LOCAL, props=_props(65_536))
        self.assertEqual(params.prefill_samples, 3)
        self.assertEqual(params.idle_source, "measured")

    def test_resolve_caches_per_base_url(self):
        serving.clear_cache()
        self.addCleanup(serving.clear_cache)
        with mock.patch.object(serving, "derive", return_value="P") as derive:
            self.assertEqual(serving.resolve(LOCAL), "P")
            self.assertEqual(serving.resolve(LOCAL + "/"), "P")
        derive.assert_called_once()


def _params(window=262_144, idle_ms=2_500_000):
    return serving.ServingParams(base_url=LOCAL, idle_timeout_ms=idle_ms, context_window=window,
                                 compact_at=serving.compact_threshold(window), prefill_tps=263.0,
                                 prefill_samples=5, ctx_source="props", idle_source="measured")


class Projections(unittest.TestCase):

    def test_opencode_limits(self):
        p = _params(262_144)          # compact_at 199,229
        # The run.py defaults already compact below 0.76 of the window: unchanged.
        self.assertEqual(serving.opencode_limits(p, context_limit=180_224, output_limit=32_000),
                         (180_224, 32_000))
        # Unknown window: unchanged.
        self.assertEqual(serving.opencode_limits(None, context_limit=0, output_limit=0), (0, 0))
        # 0 (never compact proactively) -> compaction at compact_at.
        ctx, out = serving.opencode_limits(p, context_limit=0, output_limit=16_384)
        self.assertEqual(ctx - out, p.compact_at)
        # A small server: never above the window, compaction never past compact_at.
        small = _params(65_536)
        ctx, out = serving.opencode_limits(small, context_limit=180_224, output_limit=8_192)
        self.assertLessEqual(ctx, 65_536)
        self.assertLessEqual(ctx - out, small.compact_at)
        ctx, out = serving.opencode_limits(_params(16_384), context_limit=180_224,
                                           output_limit=40_960)
        self.assertLess(out, ctx)
        self.assertLessEqual(ctx, 16_384)

    def test_opencode_provider_options_raise_header_and_chunk_timeouts(self):
        block = serving.opencode_provider_options("qwen-gpu/qwen3.8-27b", _params())
        self.assertEqual(block, {"provider": {"qwen-gpu": {"options": {
            "headerTimeout": 2_500_000, "chunkTimeout": 2_500_000}}}})
        self.assertEqual(serving.opencode_provider_options("qwen-gpu/x", None), {})

    def test_think_budget_block_and_fields(self):
        self.assertEqual(serving.think_budget_fields(0), {})
        fields = serving.think_budget_fields(8000)
        self.assertEqual(fields["thinking_budget_tokens"], 8000)
        self.assertIn("Thinking budget", fields["reasoning_budget_message"])
        block = serving.opencode_think_budget("qwen-gpu/qwen3.8-27b", 8000)
        self.assertEqual(block["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"]["options"],
                         fields)
        with self.assertRaises(ValueError):
            serving.think_budget_fields(-1)

    def test_codex_overrides_are_the_f12_knobs(self):
        p = _params(196_608, idle_ms=3_600_000)
        self.assertEqual(serving.codex_config_overrides(p, "local27b"), [
            "-c", "model_providers.local27b.stream_idle_timeout_ms=3600000",
            "-c", "model_context_window=196608",
            "-c", f"model_auto_compact_token_limit={p.compact_at}"])
        self.assertEqual(serving.codex_config_overrides(None, "x"), [])

    def test_serving_block_merges_over_both_config_builders(self):
        lane = Path("/tmp/lane")
        block = seat_config._merge(
            serving.opencode_provider_options("qwen-gpu/qwen3.8-27b", _params()),
            serving.opencode_think_budget("qwen-gpu/qwen3.8-27b", 8000))
        plain = seat_config.build_plain_config(role="planner", lane=lane,
                                               model="qwen-gpu/qwen3.8-27b", thinking="medium",
                                               serving=block)
        entry = plain["provider"]["qwen-gpu"]
        self.assertEqual(entry["options"]["chunkTimeout"], 2_500_000)
        # Merged with, not replacing, the reasoning kwargs on the same model entry.
        self.assertIn("chat_template_kwargs", entry["models"]["qwen3.8-27b"]["options"])
        self.assertEqual(entry["models"]["qwen3.8-27b"]["options"]["thinking_budget_tokens"], 8000)
        bounded = seat_config.build_actor_config(role="planner", lane=lane, model="qwen-gpu/m",
                                                 instructions_path=Path("/tmp/i.md"),
                                                 serving={"provider": {"qwen-gpu": {
                                                     "options": {"headerTimeout": 1}}}})
        self.assertEqual(bounded["provider"]["qwen-gpu"]["options"]["headerTimeout"], 1)
        self.assertEqual(bounded["compaction"], {"prune": True})
        # No serving block: byte-identical to the historical builders.
        self.assertEqual(seat_config.build_plain_config(role="critic", lane=lane),
                         seat_config.build_plain_config(role="critic", lane=lane, serving={}))

    def test_seat_label_suffixes(self):
        self.assertEqual(seat_config.seat_label("plain"), "plain")
        self.assertEqual(seat_config.seat_label("plain", serving="f1f2", think_budget=8000),
                         "plain+f1f2+tb8k")


class SeatWiring(unittest.TestCase):
    """`_seat_call` / `AgentPlanner._seated` with F1 and the thinking budget."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()

    def _patched(self, base_url):
        patches = [mock.patch.object(actors, "_provider_base_url", return_value=base_url),
                   mock.patch.object(serving, "resolve", return_value=_params(65_536))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _config(self, env) -> dict:
        return json.loads(Path(env["OPENCODE_CONFIG"]).read_text())

    def test_plain_seat_local_provider_gets_f1(self):
        self._patched(LOCAL)
        seat = actors.ActorSeat(bounded=False, serving_f1=True, context_limit=180_224,
                                output_limit=8_192, planner_think_budget=8000)
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        env = actors._seat_call(seat, backend, "critic", self.ws, {})
        conf = self._config(env)
        self.assertEqual(conf["provider"]["qwen-gpu"]["options"],
                         {"headerTimeout": 2_500_000, "chunkTimeout": 2_500_000})
        limit = conf["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"]["limit"]
        self.assertLessEqual(limit["context"], 65_536)
        self.assertLessEqual(limit["context"] - limit["output"], int(65_536 * 0.76))
        # The critic never carries the planner's thinking budget.
        self.assertNotIn("options", conf["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"])
        recorded = json.loads(env[actors.SEAT_ENV_SERVING])
        self.assertTrue(recorded["local"])
        self.assertEqual(recorded["f1"]["context_window"], 65_536)
        self.assertIn("+f1", env[actors.SEAT_ENV_ARM])
        budgets = json.loads(env[actors.SEAT_ENV_BUDGETS])
        self.assertEqual(budgets["context_limit"], limit["context"])
        self.assertEqual(budgets["serving"], "f1")

    def test_hosted_provider_is_untouched(self):
        self._patched("https://api.deepseek.com/v1")
        backend = actors.backend_for("deepseek/deepseek-v4-flash", "max")
        on = actors.ActorSeat(bounded=False, serving_f1=True, planner_think_budget=8000)
        env = actors._seat_call(on, backend, "planner", self.ws, {})
        conf = self._config(env)
        self.assertNotIn("provider", conf)
        recorded = json.loads(env[actors.SEAT_ENV_SERVING])
        self.assertEqual(recorded["think_budget_skipped"], "provider_not_local")

    def test_knobs_off_is_byte_identical(self):
        self._patched(LOCAL)
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        seat = actors.ActorSeat(bounded=False, context_limit=180_224, output_limit=8_192)
        env = actors._seat_call(seat, backend, "planner", self.ws, {})
        self.assertNotIn(actors.SEAT_ENV_SERVING, env)
        self.assertEqual(self._config(env), seat_config.build_plain_config(
            role="planner", lane=self.ws, model="qwen-gpu/qwen3.8-27b",
            context_limit=180_224, output_limit=8_192))
        self.assertNotIn("+f1", env[actors.SEAT_ENV_ARM])

    def test_codex_and_claude_backends_never_change(self):
        seat = actors.ActorSeat(serving_f1=True, planner_think_budget=8000, answer_protocol="f2")
        for model in ("gpt-6.1-sol", "claude-opus-5-5"):
            backend = actors.backend_for(model, "high")
            self.assertEqual(actors._seat_serving(seat, backend, "planner", {"context_limit": 1}),
                             ({"context_limit": 1}, {}, {}))
            self.assertIsNone(actors._seat_call(seat, backend, "planner", self.ws, {}))

    def test_bounded_planner_gets_think_budget_and_f1(self):
        self._patched(LOCAL)
        planner = actors.AgentPlanner(
            workspace=self.ws, backend=actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
            seat=actors.ActorSeat(bounded=True, serving_f1=True, planner_think_budget=8000,
                                  answer_protocol="f2", planner_budget_s=4500,
                                  context_limit=180_224, output_limit=8_192))
        with actors._scratch_scope(self.ws).scope("call", "planner"):
            backend, env = planner._seated("planner", {})
            conf = self._config(env)
            entry = conf["provider"]["qwen-gpu"]
            self.assertEqual(entry["options"]["headerTimeout"], 2_500_000)
            self.assertEqual(entry["models"]["qwen3.8-27b"]["options"]["thinking_budget_tokens"],
                             8000)
            self.assertTrue(env[actors.SEAT_ENV_ARM].startswith("bounded"))
            self.assertIn("+f1f2", env[actors.SEAT_ENV_ARM])
            self.assertIn("+tb8k", env[actors.SEAT_ENV_ARM])
            _, author_env = planner._seated("author", {})
            author = self._config(author_env)["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"]
            self.assertNotIn("thinking_budget_tokens", author.get("options", {}))
            self.assertNotIn("+f2", author_env[actors.SEAT_ENV_ARM])


class RunFlags(unittest.TestCase):

    def _ns(self, **kw):
        base = dict(actor_context_limit=0, actor_output_limit=0, actor_planner_output_limit=0,
                    actor_author_output_limit=0, actor_planner_budget_s=4500,
                    actor_author_budget_s=0, actor_planner_salvage_s=900,
                    actor_serving_f1="on", actor_answer_protocol="f2",
                    actor_answer_force_frac=0.65, actor_planner_think_budget=8000)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_seat_fields_and_validation(self):
        self.assertEqual(run_mod._actor_serving(self._ns()), {
            "serving_f1": True, "answer_protocol": "f2", "answer_force_frac": 0.65,
            "planner_think_budget": 8000})
        self.assertIsNone(run_mod._actor_budget_error(self._ns()))
        self.assertIn("force-frac", run_mod._actor_budget_error(self._ns(actor_answer_force_frac=1.0)))
        self.assertIn("think-budget",
                      run_mod._actor_budget_error(self._ns(actor_planner_think_budget=-1)))

    def test_old_namespaces_are_all_off(self):
        self.assertEqual(run_mod._actor_serving(argparse.Namespace()), {
            "serving_f1": False, "answer_protocol": "off", "answer_force_frac": 0.65,
            "planner_think_budget": 0})


# -------------------------------------------------------------------------------------
# The wire: the installed opencode against a recording mock (no model server anywhere;
# the harness is test_actor_author_thinking's: scratch HOME/XDG, global config rebased).
# -------------------------------------------------------------------------------------
from autokernel.loop import test_actor_author_thinking as wire  # noqa: E402


class _SlowHeaders(wire._Recorder):
    """Holds every chat completion's response headers for DELAY_S (a silent prefill),
    and records whether the client had already hung up when the server answered."""
    DELAY_S = 4.0

    def do_POST(self):
        import select
        import socket
        if self.path.endswith("/chat/completions"):
            time.sleep(self.DELAY_S)
            gone = False
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable:
                    gone = self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                gone = True
            type(self).bodies.append({"method": "HUNGUP" if gone else "ANSWERED",
                                      "path": self.path})
            if gone:
                return
        try:
            super().do_POST()
        except OSError:
            pass


@unittest.skipUnless(wire.OPENCODE and wire.REAL_GLOBAL.is_file()
                     and "http://127.0.0.1:8083/v1" in wire.REAL_GLOBAL.read_text(errors="replace"),
                     "needs the installed opencode and the host's qwen-gpu global config")
class OpencodeWire(unittest.TestCase):

    def test_planner_body_carries_the_thinking_budget(self):
        records, config, err = wire.run_opencode_against_mock("planner", actors.ActorSeat(
            bounded=False, planner_think_budget=8000))
        bodies = wire._chat_bodies(records)
        self.assertTrue(bodies, f"opencode sent no chat completion; stderr: {err}")
        main = [b for b in bodies if b.get("tools")] or bodies
        for body in main:
            self.assertEqual(body.get("thinking_budget_tokens"), 8000)
            self.assertIn("Thinking budget", body.get("reasoning_budget_message", ""))

    def _hangups(self, idle_ms: int) -> list[str]:
        params = dataclasses.replace(_params(), idle_timeout_ms=idle_ms)
        with mock.patch.object(serving, "resolve", return_value=params), \
                mock.patch.object(wire, "_Recorder", _SlowHeaders):
            records, config, err = wire.run_opencode_against_mock(
                "planner", actors.ActorSeat(bounded=False, serving_f1=True), timeout_s=40)
        self.assertEqual(config["provider"]["qwen-gpu"]["options"]["headerTimeout"], idle_ms)
        marks = [r["method"] for r in records if r["method"] in ("HUNGUP", "ANSWERED")]
        self.assertTrue(marks, f"no chat completion reached the mock; stderr: {err}")
        return marks

    def test_header_timeout_from_the_per_call_config_governs_the_wait(self):
        # Below the silent interval: opencode hangs up before the server answers (D1) ...
        self.assertIn("HUNGUP", self._hangups(1_500))
        # ... and the F1 value above it waits the prefill out.
        self.assertNotIn("HUNGUP", self._hangups(30_000))


if __name__ == "__main__":
    unittest.main()
