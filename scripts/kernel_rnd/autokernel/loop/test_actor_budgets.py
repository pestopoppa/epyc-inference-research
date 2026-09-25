"""OAB-22 / OAB-23: per-call budgets for the opencode planner/author seat, offline.

DS41 run 10 (2026-09-25 18:49Z): the first planner call ran 61 min and reached 183,710
tokens of :8083's 196,608-token unified KV pool; a full pool under MTP crashes the
server. What must hold:

* limits: every per-call OPENCODE_CONFIG (plain, bounded, critic) sets `limit.context`
  and `limit.output` on the call's own provider/model, and after opencode's
  global-then-OPENCODE_CONFIG deep merge the global provider definition (npm, name,
  options.baseURL) is intact. `_Opencode` below replicates the 1.18.31 binary's merge
  (`uW` -> mergeDeep), provider-model parse, `maxOutputTokens` and `isOverflow`, as
  documented in `actor_opencode_config`;
* concise: the rule rides the planner and author prompts only when on, only on opencode;
  every knob off is the historical prompt, config and arm label byte for byte;
* budget: a call past its wall budget is ended through the stop path, recorded as
  `failure_class: budget_exhausted`, a complete reply is salvaged, anything else raises
  `ActorBudgetExhausted` that is never retried, and it is distinct from the timeout.
The budget tests run REAL child processes (the property is that the call ends).
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

from autokernel.loop import actor_metrics, actor_opencode_config as aoc, actors
from autokernel.loop import loop as loop_mod

MODEL = "qwen-gpu/qwen3.8-27b"
#: The global provider definitions as ~/.config/opencode/opencode.jsonc carries them.
GLOBAL = {
    "provider": {
        "qwen-local": {"npm": "@ai-sdk/openai-compatible",
                       "name": "Local llama-server (EPYC CPU)",
                       "options": {"baseURL": "http://127.0.0.1:8074/v1"},
                       "models": {"qwen3.8-flash-next": {"name": "Qwen3.8-Flash-Next"}}},
        "qwen-gpu": {"npm": "@ai-sdk/openai-compatible",
                     "name": "Local llama-server (MI210, Qwen3.8-27B)",
                     "options": {"baseURL": "http://127.0.0.1:8083/v1"},
                     "models": {"qwen3.8-27b": {"name": "Qwen3.8-27B Q8_0 (author)"}}},
    },
    "permission": {"bash": {"mount*": "deny"}},
}
REAL_GLOBAL = Path.home() / ".config" / "opencode" / "opencode.jsonc"
POOL_TOKENS = 196_608   # :8083, np4, --kv-unified


def _merge_deep(a: dict, b: dict) -> dict:
    """remeda mergeDeep, which opencode's `uW` uses: plain objects merge key by key,
    everything else (arrays included) is replaced by the later value."""
    out = dict(a)
    for key, value in b.items():
        out[key] = (_merge_deep(a[key], value)
                    if isinstance(a.get(key), dict) and isinstance(value, dict) else value)
    return out


class _Opencode:
    """The model one call runs on, as opencode 1.18.31 resolves it from its configs."""

    OUTPUT_TOKEN_MAX = 32_000   # ProviderTransform.OUTPUT_TOKEN_MAX (M7)

    def __init__(self, seat_config: dict, global_config: dict = GLOBAL):
        self.merged = _merge_deep(global_config, seat_config)

    def provider(self, model: str) -> dict:
        return self.merged["provider"][model.split("/", 1)[0]]

    def limit(self, model: str) -> dict:
        # limit:{context:C.limit?.context??_?.limit?.context??0, input:..., output:...??0}
        entry = self.provider(model)["models"][model.split("/", 1)[1]]
        limit = entry.get("limit") or {}
        return {"context": limit.get("context", 0), "input": limit.get("input"),
                "output": limit.get("output", 0)}

    def max_output_tokens(self, model: str) -> int:
        # function by($,Z=M7){return Math.min($.limit.output,Z)||Z}
        return min(self.limit(model)["output"], self.OUTPUT_TOKEN_MAX) or self.OUTPUT_TOKEN_MAX

    def usable(self, model: str) -> int:
        # vn(): context 0 -> 0; input set -> input - reserved; else context - maxOutputTokens
        limit = self.limit(model)
        if limit["context"] == 0:
            return 0
        return max(0, limit["context"] - self.max_output_tokens(model))

    def overflows(self, model: str, total_tokens: int) -> bool:
        # vl(): compaction.auto false or context 0 -> never; else total >= usable
        if (self.merged.get("compaction") or {}).get("auto") is False:
            return False
        if self.limit(model)["context"] == 0:
            return False
        return total_tokens >= self.usable(model)


def _jsonc(text: str) -> dict:
    """Strip // comments outside strings (the global config's only jsonc feature)."""
    out, in_str, i = [], False, 0
    while i < len(text):
        char = text[i]
        if in_str:
            out.append(char)
            if char == "\\":
                out.append(text[i + 1])
                i += 1
            elif char == '"':
                in_str = False
        elif char == '"':
            in_str = True
            out.append(char)
        elif text.startswith("//", i):
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        else:
            out.append(char)
        i += 1
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))


class ModelLimitsConfig(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lane = Path(self._tmp.name) / "lane"
        self.lane.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _configs(self, **limits):
        plain = aoc.build_plain_config(role="planner", lane=self.lane, model=MODEL, **limits)
        bounded = aoc.build_actor_config(role="planner", lane=self.lane, model=MODEL,
                                         instructions_path=self.lane / "i.md", **limits)
        critic = aoc.build_plain_config(role="critic", lane=self.lane, model=MODEL,
                                        lane_guard=True, **limits)
        return {"plain": plain, "bounded": bounded, "critic": critic}

    def test_every_seat_config_sets_both_limits_on_the_call_model(self):
        for name, config in self._configs(context_limit=131072, output_limit=8192).items():
            with self.subTest(seat=name):
                self.assertEqual(
                    config["provider"],
                    {"qwen-gpu": {"models": {"qwen3.8-27b": {
                        "limit": {"context": 131072, "output": 8192}}}}})
                self.assertIs(config["snapshot"], False)

    def test_the_merge_keeps_the_global_provider_definition(self):
        for name, config in self._configs(context_limit=131072, output_limit=8192).items():
            with self.subTest(seat=name):
                oc = _Opencode(config)
                provider = oc.provider(MODEL)
                self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
                self.assertEqual(provider["options"], {"baseURL": "http://127.0.0.1:8083/v1"})
                self.assertEqual(provider["models"]["qwen3.8-27b"]["name"],
                                 "Qwen3.8-27B Q8_0 (author)")
                # The other provider and the global deny-list are untouched.
                self.assertEqual(oc.merged["provider"]["qwen-local"],
                                 GLOBAL["provider"]["qwen-local"])
                self.assertEqual(oc.merged["permission"]["bash"]["mount*"], "deny")

    @unittest.skipUnless(REAL_GLOBAL.is_file(), "no global opencode config on this host")
    def test_the_merge_keeps_the_real_global_provider_definition(self):
        real = _jsonc(REAL_GLOBAL.read_text(encoding="utf-8"))
        if "qwen-gpu" not in real.get("provider", {}):
            self.skipTest("the host's global config defines no qwen-gpu provider")
        config = self._configs(context_limit=131072, output_limit=8192)["plain"]
        merged = _Opencode(config, real).provider(MODEL)
        for key, value in real["provider"]["qwen-gpu"].items():
            if key != "models":
                self.assertEqual(merged[key], value)
        self.assertEqual({k: v for k, v in merged["models"]["qwen3.8-27b"].items()
                          if k != "limit"}, real["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"])

    def test_opencode_semantics_compact_before_the_pool_fills(self):
        oc = _Opencode(self._configs(context_limit=131072, output_limit=8192)["plain"])
        self.assertEqual(oc.max_output_tokens(MODEL), 8192)        # -> max_tokens
        self.assertEqual(oc.usable(MODEL), 131072 - 8192)
        self.assertFalse(oc.overflows(MODEL, 122_879))
        self.assertTrue(oc.overflows(MODEL, 122_880))
        # Worst next request: usable + one step of plain-seat tool output (50 KB default
        # ~ 15k tokens) + O decoded -- still well inside the unified pool.
        self.assertLess(oc.usable(MODEL) + 15_000 + 8192, POOL_TOKENS - 3 * 10_000)

    def test_without_limits_opencode_never_compacts_and_decodes_32k(self):
        """The run-10 state: a config-only model with no limit."""
        oc = _Opencode(self._configs()["plain"])
        self.assertEqual(oc.max_output_tokens(MODEL), 32_000)
        self.assertFalse(oc.overflows(MODEL, 183_710))

    def test_limits_off_are_byte_identical_configs(self):
        base = aoc.build_plain_config(role="author", lane=self.lane, lane_guard=True)
        self.assertEqual(aoc.build_plain_config(role="author", lane=self.lane, lane_guard=True,
                                                model=MODEL, context_limit=0, output_limit=0),
                         base)
        bounded = aoc.build_actor_config(role="planner", lane=self.lane,
                                         instructions_path=self.lane / "i.md")
        self.assertEqual(aoc.build_actor_config(role="planner", lane=self.lane, model=MODEL,
                                                instructions_path=self.lane / "i.md"), bounded)
        self.assertNotIn("provider", base)

    def test_bad_limits_are_refused(self):
        with self.assertRaises(ValueError):
            aoc.model_limits(MODEL, context_limit=8192, output_limit=8192)
        with self.assertRaises(ValueError):
            aoc.model_limits("no-provider", context_limit=131072)
        with self.assertRaises(ValueError):
            aoc.model_limits(MODEL, context_limit=-1)

    def test_one_limit_alone_keeps_the_other_at_the_opencode_default(self):
        limits = aoc.model_limits(MODEL, context_limit=131072)
        entry = limits["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"]["limit"]
        self.assertEqual(entry, {"context": 131072, "output": 0})
        oc = _Opencode(limits)
        self.assertEqual(oc.usable(MODEL), 131072 - 32_000)

    def test_labels(self):
        self.assertEqual(aoc.seat_label("plain"), "plain")
        self.assertEqual(aoc.seat_label("plain", lane_guard=True), "plain+lane-guard")
        self.assertEqual(aoc.seat_label("bounded", context_limit=131072, output_limit=8192,
                                        concise=True, budget_s=2700),
                         "bounded+ctx128k+out8k+concise+budget2700s")


class SeatWiring(unittest.TestCase):
    """What `_seat_call` / `_seated` write for a real opencode backend (no call is run)."""

    SEAT = dict(context_limit=131072, output_limit=8192, concise=True,
                planner_budget_s=2700)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()
        self.backend = actors.backend_for(MODEL, "high")

    def tearDown(self):
        self._tmp.cleanup()

    def test_plain_seat_config_carries_limits_and_the_arm_names_them(self):
        seat = actors.ActorSeat(bounded=False, **self.SEAT)
        env = actors._seat_call(seat, self.backend, "planner", self.ws, {})
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        self.assertEqual(config["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"]["limit"],
                         {"context": 131072, "output": 8192})
        self.assertEqual(env[actors.SEAT_ENV_ARM], "plain+ctx128k+out8k+concise+budget2700s")
        self.assertEqual(json.loads(env[actors.SEAT_ENV_BUDGETS]),
                         {"concise": True, "context_limit": 131072, "output_limit": 8192})

    def test_the_critic_gets_the_limits_but_no_concise_or_budget(self):
        seat = actors.ActorSeat(bounded=False, **self.SEAT)
        env = actors._seat_call(seat, self.backend, "critic", self.ws, {})
        self.assertEqual(env[actors.SEAT_ENV_ARM], "plain+ctx128k+out8k")
        self.assertIn("provider", json.loads(Path(env["OPENCODE_CONFIG"]).read_text()))

    def test_bounded_seat_config_carries_limits(self):
        planner = actors.AgentPlanner(workspace=self.ws, backend=self.backend,
                                      seat=actors.ActorSeat(bounded=True, **self.SEAT))
        _backend, env = planner._seated("author", {})
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        self.assertEqual(config["provider"]["qwen-gpu"]["models"]["qwen3.8-27b"]["limit"],
                         {"context": 131072, "output": 8192})
        self.assertEqual(env[actors.SEAT_ENV_ARM], "bounded+ctx128k+out8k+concise")

    def test_knobs_off_seat_is_the_historical_call(self):
        env = actors._seat_call(actors.ActorSeat(bounded=False), self.backend, "planner",
                                self.ws, {})
        self.assertNotIn(actors.SEAT_ENV_ARM, env)
        self.assertNotIn(actors.SEAT_ENV_BUDGETS, env)
        self.assertEqual(json.loads(Path(env["OPENCODE_CONFIG"]).read_text()),
                         {"$schema": "https://opencode.ai/config.json", "snapshot": False})


class ConcisePrompt(unittest.TestCase):

    HYP = loop_mod.Hypothesis(mechanism_id="akm-x", statement="s", falsifier="f",
                              target_surface="ggml/src/ggml-cuda/a.cu", target_symbol="k")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _prompts(self, backend, seat):
        seen = {}

        def fake(prompt, **kw):
            role = "planner" if kw["schema"] is actors.HYPOTHESIS_SCHEMA else "author"
            seen[role] = (prompt, kw.get("budget_s"))
            if role == "planner":
                return json.dumps({"mechanism_id": "akm-y", "statement": "s",
                                   "falsifier": "f", "target_surface": "a.cu",
                                   "target_symbol": "k"})
            return json.dumps({"abstain": "no"})
        planner = actors.AgentPlanner(workspace=self.ws, backend=backend, seat=seat)
        with mock.patch.object(actors, "_run_agent", side_effect=fake):
            planner.propose({})
            planner.author(self.HYP, {})
        return seen

    def test_knobs_off_prompts_are_byte_identical(self):
        backend = actors.backend_for(MODEL, "high")
        control = self._prompts(backend, None)
        off = self._prompts(backend, actors.ActorSeat(bounded=False))
        self.assertEqual(off, control)
        for prompt, budget in control.values():
            self.assertNotIn(actors.CONCISE_RULE, prompt)
            self.assertIsNone(budget)

    def test_concise_rides_planner_and_author_prompts(self):
        backend = actors.backend_for(MODEL, "high")
        control = self._prompts(backend, None)
        on = self._prompts(backend, actors.ActorSeat(bounded=False, concise=True))
        for role in ("planner", "author"):
            self.assertEqual(on[role][0], control[role][0] + "\n\n" + actors.CONCISE_RULE)
        for phrase in ("derive each fact once", "reuse results instead of re-deriving",
                       "under ~4,000 tokens", "the JSON object only",
                       "no draft you then replace"):
            self.assertIn(phrase, actors.CONCISE_RULE)

    def test_budgets_reach_the_call_by_role(self):
        backend = actors.backend_for(MODEL, "high")
        seen = self._prompts(backend, actors.ActorSeat(bounded=False, planner_budget_s=2700))
        self.assertEqual(seen["planner"][1], 2700)
        self.assertIsNone(seen["author"][1])

    def test_codex_prompts_never_change(self):
        backend = actors.backend_for("gpt-5.6-sol", "high")
        control = self._prompts(backend, None)
        self.assertEqual(self._prompts(backend, actors.ActorSeat(
            bounded=False, concise=True, planner_budget_s=2700)), control)


# ------------------------------------------------------------------------ budget path

HYPOTHESIS = json.dumps({"mechanism_id": "akm-budget", "statement": "fuse the loads",
                         "falsifier": "no tg gain", "target_surface": "ggml/src/a.cu",
                         "target_symbol": "k"})

#: Writes its pid, prints `sys.argv[2]` (a reply or nothing), then sleeps: an actor that
#: never exits on its own -- the run-9 endless chain / run-10 61-minute call.
LINGER = r"""
import os, sys, time
with open(sys.argv[1], "w") as fh:
    fh.write(str(os.getpid()))
sys.stdout.write(sys.argv[2]); sys.stdout.flush()
time.sleep(120)
"""


class _ScriptBackend(actors.Backend):
    script: str = ""
    marker: str = ""
    reply: str = ""

    def argv(self, prompt, workspace, *, read_only=False):
        return [sys.executable, "-c", self.script, self.marker, self.reply]


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state not in ("Z", "X")


class BudgetPath(unittest.TestCase):

    def setUp(self):
        self._patches = [mock.patch.object(actors, "STOP_POLL_S", 0.05),
                         mock.patch.object(actors, "STOP_GRACE_S", 3.0)]
        for patch in self._patches:
            patch.start()
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.ws = root / "lane"
        self.ws.mkdir()
        self.marker = root / "pid"

    def tearDown(self):
        for patch in self._patches:
            patch.stop()
        if self.marker.exists() and self.marker.read_text().strip():
            try:
                os.kill(int(self.marker.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        self._tmp.cleanup()

    def _backend(self, reply: str) -> _ScriptBackend:
        backend = _ScriptBackend(kind="opencode", model="test/linger", effort="high",
                                 binary=sys.executable)
        object.__setattr__(backend, "script", LINGER)
        object.__setattr__(backend, "marker", str(self.marker))
        object.__setattr__(backend, "reply", reply)
        return backend

    def _rows(self) -> list[dict]:
        log = self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
        return [json.loads(line) for line in log.read_text().splitlines()]

    def _metrics(self) -> dict:
        rows = [r for r in self._rows() if r.get("schema") == actor_metrics.METRICS_SCHEMA]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def _assert_dead(self):
        pid = int(self.marker.read_text())
        deadline = time.monotonic() + 5
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(_alive(pid), "the actor survived its budget")

    def test_a_complete_reply_is_salvaged_and_recorded_budget_exhausted(self):
        started = time.monotonic()
        env = {actors.SEAT_ENV_BUDGETS: json.dumps(
            {"context_limit": 131072, "output_limit": 8192, "concise": True})}
        text = actors._run_agent("p", workspace=self.ws, timeout_s=60,
                                 backend=self._backend(HYPOTHESIS),
                                 schema=actors.HYPOTHESIS_SCHEMA, env=env, budget_s=1)
        self.assertLess(time.monotonic() - started, 30)
        self.assertIn('"akm-budget"', text)
        self._assert_dead()
        row = self._metrics()
        self.assertEqual(row["failure_class"], "budget_exhausted")
        self.assertTrue(row["salvaged"])
        self.assertFalse(row["timed_out"])
        self.assertEqual(row["budgets"]["budget_s"], 1)
        self.assertTrue(row["budgets"]["budget_exhausted"])
        self.assertEqual(row["budgets"]["context_limit"], 131072)
        self.assertEqual(row["budgets"]["output_limit"], 8192)
        self.assertTrue(row["budgets"]["concise"])
        self.assertIsNone(row["budgets"]["compacted"])   # no export: unknown, not "no"

    def test_an_incomplete_reply_raises_budget_exhausted(self):
        with self.assertRaises(actors.ActorBudgetExhausted) as caught:
            actors._run_agent("p", workspace=self.ws, timeout_s=60,
                              backend=self._backend('{"mechanism_id": "akm-half"'),
                              schema=actors.HYPOTHESIS_SCHEMA, budget_s=1)
        self.assertTrue(str(caught.exception).startswith("budget_exhausted"))
        self.assertIn("not retried", str(caught.exception))
        self._assert_dead()
        row = self._metrics()
        self.assertEqual(row["failure_class"], "budget_exhausted")
        self.assertFalse(row["salvaged"])
        # The raw streams are kept, like every other call.
        replies = list((self.ws.parent / actors.ACTOR_REPLY_DIR).glob("*.stdout"))
        self.assertEqual(len(replies), 1)

    def test_an_abstention_is_not_salvaged_from_a_budget_kill(self):
        with self.assertRaises(actors.ActorBudgetExhausted):
            actors._run_agent("p", workspace=self.ws, timeout_s=60,
                              backend=self._backend('{"abstain": "ran out of ideas"}'),
                              schema=actors.HYPOTHESIS_SCHEMA, budget_s=1)

    def test_the_timeout_stays_the_timeout(self):
        with self.assertRaises(actors.ProviderTransient) as caught:
            actors._run_agent("p", workspace=self.ws, timeout_s=1,
                              backend=self._backend(""), schema=actors.HYPOTHESIS_SCHEMA,
                              budget_s=30)
        self.assertNotIsInstance(caught.exception, actors.ActorBudgetExhausted)
        self.assertIn("exceeded 1s", str(caught.exception))
        row = self._metrics()
        self.assertTrue(row["timed_out"])
        self.assertIsNone(row["failure_class"])
        self.assertFalse(row["budgets"]["budget_exhausted"])

    def test_the_planner_is_never_relaunched_after_its_budget(self):
        launches = []
        real_popen = actors.subprocess.Popen

        def counting_popen(*args, **kwargs):
            launches.append(args[0])
            return real_popen(*args, **kwargs)
        planner = actors.AgentPlanner(
            workspace=self.ws, backend=self._backend("thinking..."), timeout_s=60,
            seat=actors.ActorSeat(bounded=False, planner_budget_s=1))
        started = time.monotonic()
        with mock.patch.object(actors.subprocess, "Popen", side_effect=counting_popen):
            with self.assertRaises(actors.ActorBudgetExhausted):
                planner.propose({})
        self.assertEqual(len(launches), 1, "a budget-exhausted planner was relaunched")
        self.assertLess(time.monotonic() - started, actors.BACKOFF_S[0] / 2)
        row = self._metrics()
        self.assertTrue(row["seat_arm"].endswith("+budget1s"))

    def test_the_planner_salvages_a_complete_reply_at_its_budget(self):
        planner = actors.AgentPlanner(
            workspace=self.ws, backend=self._backend(HYPOTHESIS), timeout_s=60,
            seat=actors.ActorSeat(bounded=False, planner_budget_s=1))
        hypothesis = planner.propose({})
        self.assertEqual(hypothesis.mechanism_id, "akm-budget")

    def test_iterate_records_it_as_an_iteration_transient_with_the_reason(self):
        planner = mock.Mock()
        planner.propose.side_effect = actors.ActorBudgetExhausted(
            "budget_exhausted: the planner call spent its 2700s budget")
        outcome = loop_mod.iterate(
            planner=planner, critic=mock.Mock(), context={},
            measure=mock.Mock(side_effect=AssertionError("no measurement")),
            gate=mock.Mock(side_effect=AssertionError("no gate")),
            commit=mock.Mock(side_effect=AssertionError("no commit")))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertTrue(outcome.reasons[0].startswith("budget_exhausted"))
        self.assertEqual(planner.propose.call_count, 1)

    def test_backoff_never_retries_a_spent_budget(self):
        calls = []

        def call():
            calls.append(1)
            raise actors.ActorBudgetExhausted("budget_exhausted: spent")
        with self.assertRaises(actors.ActorBudgetExhausted):
            actors._with_backoff(call, sleep=lambda _s: None)
        self.assertEqual(len(calls), 1)


class MetricsExport(unittest.TestCase):

    def test_output_capped_steps_count_length_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.json"
            msgs = [{"info": {"role": "user", "sessionID": "s"}, "parts": []}]
            for finish in ("tool-calls", "length", "stop"):
                msgs.append({"info": {"role": "assistant", "finish": finish,
                                      "tokens": {"input": 1, "output": 1}}, "parts": []})
            path.write_text(json.dumps({"messages": msgs}))
            stats = actor_metrics.parse_export(path)
        self.assertEqual(stats["output_capped_steps"], 1)
        self.assertIn("output_capped_steps", actor_metrics.TOTAL_FIELDS)

    def test_budgets_block_reads_compaction_and_cap_from_the_export(self):
        env = {actors.SEAT_ENV_BUDGETS: json.dumps({"context_limit": 131072,
                                                    "output_limit": 8192, "concise": False})}
        stats = {"totals": {"compactions": 2, "output_capped_steps": 1},
                 "context_max_tokens": 120000}
        block = actors._budgets_of(env, 2700, False, stats)
        self.assertEqual(block["compacted"], True)
        self.assertEqual(block["compactions"], 2)
        self.assertEqual(block["output_capped_steps"], 1)
        self.assertEqual(block["context_max_tokens"], 120000)
        self.assertIsNone(actors._budgets_of({}, None, False, stats))


class CliKnobs(unittest.TestCase):

    def _args(self, **kw):
        base = dict(actor_context_limit=aoc.DEFAULT_CONTEXT_LIMIT,
                    actor_output_limit=aoc.DEFAULT_OUTPUT_LIMIT,
                    actor_planner_output_limit=aoc.DEFAULT_PLANNER_OUTPUT_LIMIT,
                    actor_author_output_limit=aoc.DEFAULT_AUTHOR_OUTPUT_LIMIT,
                    actor_concise="on",
                    actor_planner_budget_s=2700, actor_author_budget_s=0,
                    actor_timeout_s=7200)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_defaults_and_mapping(self):
        from autokernel.loop import run
        self.assertEqual((aoc.DEFAULT_CONTEXT_LIMIT, aoc.DEFAULT_OUTPUT_LIMIT), (131072, 8192))
        args = self._args()
        self.assertEqual(run._actor_limits(args),
                         {"context_limit": 131072, "output_limit": 8192,
                          "planner_output_limit": 16384, "author_output_limit": 32768})
        self.assertEqual(run._actor_budgets(args), {"concise": True, "planner_budget_s": 2700,
                                                    "author_budget_s": 0})
        self.assertIsNone(run._actor_budget_error(args))
        self.assertEqual(run._moot_budgets(args), [])

    def test_the_parser_defaults_are_on(self):
        from autokernel.loop import run
        source = Path(run.__file__).read_text(encoding="utf-8")
        for flag, default in (("--actor-concise", 'default="on"'),
                              ("--actor-planner-budget-s", "default=2700"),
                              ("--actor-author-budget-s", "default=0"),
                              ("--actor-context-limit", "DEFAULT_CONTEXT_LIMIT"),
                              ("--actor-output-limit", "DEFAULT_OUTPUT_LIMIT")):
            block = source[source.index(f'"{flag}"'):][:300]
            self.assertIn(default, block, flag)

    def test_invalid_and_moot_knobs(self):
        from autokernel.loop import run
        self.assertIn("below", run._actor_budget_error(
            self._args(actor_context_limit=8192, actor_output_limit=8192)))
        self.assertIn(">= 0", run._actor_budget_error(self._args(actor_planner_budget_s=-1)))
        self.assertEqual(run._moot_budgets(self._args(actor_timeout_s=1800)),
                         ["--actor-planner-budget-s=2700"])


if __name__ == "__main__":
    unittest.main()
