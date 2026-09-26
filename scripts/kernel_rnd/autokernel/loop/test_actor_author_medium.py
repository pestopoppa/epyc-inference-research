"""Operator 2026-09-26: author thinking MEDIUM, author output at its max, the action rule.

DS41 run 10g's author (27B, thinking off) wrote five AVX-512 patches the DeepSeek critic
rejected; with thinking uncapped it deliberated 74k tokens and never edited. The operator
turned author thinking back on at MEDIUM, raised the author's output cap, set the pool
budget (context 180,224 for every role, author output 40,960, planner/critic 16,384) and
added an author prompt rule pushing it to act instead of deliberating.

What must hold:

* thinking: `--actor-author-thinking medium` (the default) puts
  `chat_template_kwargs: {"enable_thinking": true, "reasoning_effort": "medium"}` in the
  AUTHOR's per-call model options only (plain and bounded seat); planner and critic never;
* output: the author's default `limit.output` is 40,960, above opencode's 32,000 clamp,
  so its call env raises the clamp (`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`); the
  planner/critic (16,384) env never carries it; the wire test proves max_tokens=40960;
* pool budget: context <= 196,608 - 16,384 and every output < context - 32,768;
* action rule: appended to the AUTHOR prompt only, opencode only; off (and the library
  default) = the historical prompts byte for byte.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actor_opencode_config as aoc, actors
from autokernel.loop import loop as loop_mod
from autokernel.loop.test_actor_author_thinking import (
    OPENCODE, REAL_GLOBAL, _chat_bodies, _model_entry, run_opencode_against_mock)

MODEL = "qwen-gpu/qwen3.8-27b"
MEDIUM = {"enable_thinking": True, "reasoning_effort": "medium"}


def _run_args(**kw) -> argparse.Namespace:
    base = dict(actor_context_limit=aoc.DEFAULT_CONTEXT_LIMIT,
                actor_output_limit=aoc.DEFAULT_OUTPUT_LIMIT,
                actor_planner_output_limit=aoc.DEFAULT_PLANNER_OUTPUT_LIMIT,
                actor_author_output_limit=aoc.DEFAULT_AUTHOR_OUTPUT_LIMIT,
                actor_author_thinking=aoc.DEFAULT_AUTHOR_THINKING,
                actor_author_action_rule="on", actor_concise="on",
                actor_planner_budget_s=2700, actor_author_budget_s=0, actor_timeout_s=7200)
    base.update(kw)
    return argparse.Namespace(**base)


def _default_seat(bounded: bool = False) -> actors.ActorSeat:
    from autokernel.loop import run
    args = _run_args()
    return actors.ActorSeat(bounded=bounded, **run._actor_limits(args),
                            **run._actor_budgets(args), **run._actor_thinking(args))


class Defaults(unittest.TestCase):

    def test_operator_defaults(self):
        self.assertEqual(aoc.DEFAULT_AUTHOR_THINKING, "medium")
        self.assertEqual(aoc.THINKING_CHOICES, ("default", "off", "medium"))
        self.assertEqual((aoc.DEFAULT_CONTEXT_LIMIT, aoc.DEFAULT_AUTHOR_OUTPUT_LIMIT,
                          aoc.DEFAULT_PLANNER_OUTPUT_LIMIT), (180224, 40960, 16384))
        self.assertEqual(aoc.MAX_CONTEXT_LIMIT, 196608 - 16384)

    def test_parser_wiring(self):
        from autokernel.loop import run
        source = Path(run.__file__).read_text(encoding="utf-8")
        block = source[source.index('"--actor-author-action-rule"'):][:200]
        self.assertIn('default="on"', block)
        block = source[source.index('"--actor-author-thinking"'):][:300]
        self.assertIn("DEFAULT_AUTHOR_THINKING", block)
        # The critic's seat never takes the author knobs.
        critic = source[source.index("make_critic=lambda worker"):][:600]
        self.assertNotIn("_actor_thinking", critic)
        self.assertEqual(run._actor_thinking(_run_args()),
                         {"author_thinking": "medium", "author_action_rule": True})
        self.assertEqual(run._actor_thinking(_run_args(actor_author_action_rule="off",
                                                       actor_author_thinking="default")),
                         {"author_thinking": "default", "author_action_rule": False})

    def test_pool_budget_validator(self):
        from autokernel.loop import run
        self.assertIsNone(run._actor_budget_error(_run_args()))
        self.assertIn("must be <= 180224",
                      run._actor_budget_error(_run_args(actor_context_limit=196608)))
        self.assertIn("full pool plus MTP",
                      run._actor_budget_error(_run_args(actor_context_limit=180225)))
        # output < context - 32768, per role.
        self.assertIsNone(run._actor_budget_error(_run_args(actor_author_output_limit=147455)))
        self.assertIn("--actor-author-output-limit",
                      run._actor_budget_error(_run_args(actor_author_output_limit=147456)))
        self.assertIn("--actor-planner-output-limit", run._actor_budget_error(
            _run_args(actor_context_limit=65536, actor_planner_output_limit=32768)))
        # The operator's split, spelled out: author compacts at ~139k, planner at ~164k.
        self.assertEqual(180224 - 40960, 139264)
        self.assertEqual(180224 - 16384, 163840)


class ThinkingMedium(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()
        self.backend = actors.backend_for(MODEL, "high")

    def tearDown(self):
        self._tmp.cleanup()

    def test_model_thinking_medium_block(self):
        self.assertEqual(aoc.model_thinking(MODEL, "medium"),
                         {"provider": {"qwen-gpu": {"models": {"qwen3.8-27b": {
                             "options": {"chat_template_kwargs": MEDIUM}}}}}})
        block = aoc.model_thinking(MODEL, "medium")
        _model_entry(block)["options"]["chat_template_kwargs"]["reasoning_effort"] = "high"
        self.assertEqual(aoc.THINKING_MEDIUM_OPTIONS["chat_template_kwargs"], MEDIUM)
        with self.assertRaises(ValueError):
            aoc.model_thinking(MODEL, "high")

    def test_plain_seat_author_only(self):
        seat = _default_seat()
        env = actors._seat_call(seat, self.backend, "author", self.ws, {})
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        self.assertEqual(_model_entry(config), {
            "limit": {"context": 180224, "output": 40960},
            "options": {"chat_template_kwargs": MEDIUM}})
        self.assertEqual(env[aoc.OUTPUT_TOKEN_MAX_ENV], "40960")
        self.assertEqual(env[actors.SEAT_ENV_ARM],
                         "plain+ctx176k+out40k+think-medium+concise+act-rule")
        applied = json.loads(env[actors.SEAT_ENV_BUDGETS])
        self.assertEqual((applied["thinking"], applied["action_rule"]), ("medium", True))
        block = actors._budgets_of(env, None, False, None)
        self.assertEqual((block["thinking"], block["action_rule"]), ("medium", True))
        for role in ("planner", "critic"):
            with self.subTest(role=role):
                env = actors._seat_call(seat, self.backend, role, self.ws, {})
                config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
                self.assertEqual(_model_entry(config),
                                 {"limit": {"context": 180224, "output": 16384}})
                self.assertNotIn(aoc.OUTPUT_TOKEN_MAX_ENV, env)
                self.assertNotIn("think", env[actors.SEAT_ENV_ARM])
                self.assertNotIn("act-rule", env[actors.SEAT_ENV_ARM])
                applied = json.loads(env[actors.SEAT_ENV_BUDGETS])
                self.assertNotIn("thinking", applied)
                self.assertNotIn("action_rule", applied)
                self.assertFalse(actors._budgets_of(env, None, False, None)["action_rule"])

    def test_bounded_seat_author_only(self):
        planner = actors.AgentPlanner(workspace=self.ws, backend=self.backend,
                                      seat=_default_seat(bounded=True))
        _backend, env = planner._seated("author", {})
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        self.assertEqual(_model_entry(config)["options"], {"chat_template_kwargs": MEDIUM})
        self.assertEqual(env[aoc.OUTPUT_TOKEN_MAX_ENV], "40960")
        self.assertIn("+think-medium", env[actors.SEAT_ENV_ARM])
        _backend, env = planner._seated("planner", {})
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        self.assertNotIn("options", _model_entry(config))
        self.assertNotIn(aoc.OUTPUT_TOKEN_MAX_ENV, env)

    def test_output_ceiling_env(self):
        self.assertEqual(aoc.output_ceiling_env(0), {})
        self.assertEqual(aoc.output_ceiling_env(32000), {})
        self.assertEqual(aoc.output_ceiling_env(32001),
                         {"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "32001"})

    def test_per_author_values_are_injectable_per_call(self):
        """Best-of-N: a mixed pair (off + medium) on its own pool share, per call."""
        base = _default_seat()
        pair = {"off": base.for_author(thinking="off", context_limit=90112,
                                       output_limit=16384),
                "medium": base.for_author(thinking="medium", context_limit=90112,
                                          output_limit=16384)}
        self.assertEqual(base.author_thinking, "medium")   # the run's seat is untouched
        self.assertEqual(base.limits_for("author"),
                         {"context_limit": 180224, "output_limit": 40960})
        expected = {"off": {"enable_thinking": False}, "medium": MEDIUM}
        for mode, seat in pair.items():
            with self.subTest(mode=mode):
                ws = self.ws.parent / f"lane-{mode}"
                ws.mkdir()
                env = actors._seat_call(seat, self.backend, "author", ws, {})
                config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
                self.assertEqual(_model_entry(config), {
                    "limit": {"context": 90112, "output": 16384},
                    "options": {"chat_template_kwargs": expected[mode]}})
                self.assertNotIn(aoc.OUTPUT_TOKEN_MAX_ENV, env)   # 16384 < 32000
                self.assertIn(f"+think-{mode}", env[actors.SEAT_ENV_ARM])
                # The planner of the same seat is untouched by the author values.
                self.assertEqual(seat.limits_for("planner"),
                                 {"context_limit": 90112, "output_limit": 16384})
                self.assertEqual(seat.thinking_for("planner"), "default")
        # The builders take every value per call, explicitly.
        self.assertEqual(_model_entry(aoc.build_plain_config(
            role="author", lane=self.ws, model=MODEL, context_limit=90112,
            output_limit=16384, thinking="off")), {
                "limit": {"context": 90112, "output": 16384},
                "options": {"chat_template_kwargs": {"enable_thinking": False}}})
        self.assertEqual(base.for_author(), base)
        with self.assertRaises(ValueError):
            base.for_author(thinking="high")

    def test_off_label_is_unchanged(self):
        seat = actors.ActorSeat(bounded=False, author_thinking="off")
        env = actors._seat_call(seat, self.backend, "author", self.ws, {})
        self.assertEqual(env[actors.SEAT_ENV_ARM], "plain+think-off")
        self.assertEqual(aoc.seat_label("plain", thinking_off=True), "plain+think-off")


class ActionRule(unittest.TestCase):

    HYP = loop_mod.Hypothesis(mechanism_id="akm-x", statement="s", falsifier="f",
                              target_surface="ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp",
                              target_symbol="mul_mat_qX_K_q8_2_X4_T")

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
            seen[role] = prompt
            if role == "planner":
                return json.dumps({"mechanism_id": "akm-y", "statement": "s",
                                   "falsifier": "f", "target_surface": "a.cpp",
                                   "target_symbol": "k"})
            return json.dumps({"abstain": "no"})
        planner = actors.AgentPlanner(workspace=self.ws, backend=backend, seat=seat)
        with mock.patch.object(actors, "_run_agent", side_effect=fake):
            planner.propose({})
            planner.author(self.HYP, {})
        return seen

    def test_rides_the_author_prompt_only(self):
        backend = actors.backend_for(MODEL, "high")
        control = self._prompts(backend, actors.ActorSeat(bounded=False, concise=True))
        on = self._prompts(backend, actors.ActorSeat(bounded=False, concise=True,
                                                     author_action_rule=True))
        self.assertEqual(on["planner"], control["planner"])
        self.assertEqual(on["author"],
                         control["author"] + "\n\n" + actors.AUTHOR_ACTION_RULE)
        self.assertTrue(on["author"].endswith(actors.CONCISE_RULE + "\n\n"
                                              + actors.AUTHOR_ACTION_RULE))

    def test_off_is_byte_identical(self):
        backend = actors.backend_for(MODEL, "high")
        control = self._prompts(backend, None)
        self.assertEqual(self._prompts(backend, actors.ActorSeat(bounded=False)), control)
        self.assertEqual(self._prompts(backend, actors.ActorSeat(
            bounded=False, author_action_rule=False, author_thinking="medium")), control)
        for prompt in control.values():
            self.assertNotIn(actors.AUTHOR_ACTION_RULE, prompt)

    def test_codex_prompts_never_change(self):
        backend = actors.backend_for("gpt-5.6-sol", "high")
        control = self._prompts(backend, None)
        self.assertEqual(self._prompts(backend, actors.ActorSeat(
            bounded=False, author_action_rule=True)), control)

    def test_rule_content(self):
        rule = actors.AUTHOR_ACTION_RULE
        for phrase in ("think briefly, then act", "small verified steps",
                       "read the exact lines you will change", "edit them immediately",
                       "re-read the edited region", "Never re-derive a data layout",
                       "copy the existing idiom", "Q4Bits", "BlockPermuter", "iqk_common.h",
                       "/usr/lib/gcc/x86_64-linux-gnu/15/include/",
                       "never invent an intrinsic name", "consistent with its callers",
                       "strictly inside the admitted scope", "dispatch or selector",
                       "abstain with the reason", "instead of writing a partial patch"):
            self.assertIn(phrase, rule)
        self.assertLess(len(rule), 1000)


class ActionRuleGitNexus(unittest.TestCase):
    """The GitNexus sentence: repo derived from the anchor worktree, allowed by the guard."""

    def setUp(self):
        from autokernel.loop import test_actor_seat_trim_guard as guard
        self.guard = guard
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root, self.build = guard._anchor(self.tmp)
        self.lane = self.tmp / "targets" / "abc" / "workers" / "lane0"
        self.lane.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _author_prompt(self, seat, context):
        seen = {}

        def fake(prompt, **kw):
            seen["prompt"] = prompt
            return json.dumps({"abstain": "no"})
        planner = actors.AgentPlanner(workspace=self.lane, seat=seat,
                                      backend=actors.backend_for(MODEL, "high"))
        with mock.patch.object(actors, "_run_agent", side_effect=fake):
            planner.author(ActionRule.HYP, context)
        return seen["prompt"]

    def test_repo_is_the_anchor_trees_absolute_path(self):
        # The DS41 anchor is registered in GitNexus as `llama.cpp`, the frozen production
        # tree's name too: only the absolute path targets it unambiguously.
        self.assertEqual(aoc.gitnexus_repo_for(self.build), str(self.root))
        self.assertTrue(Path(aoc.gitnexus_repo_for(self.build)).is_absolute())
        self.assertIsNone(aoc.gitnexus_repo_for(None))

    def test_the_author_prompt_names_the_anchor_repo(self):
        context = {"target": {"recipe": {"backend": "cpu", "build_dir": str(self.build)}}}
        repo = str(self.root)
        rule = actors.author_action_rule(repo)
        control = self._author_prompt(actors.ActorSeat(bounded=False), context)
        on = self._author_prompt(actors.ActorSeat(bounded=False, author_action_rule=True),
                                 context)
        self.assertEqual(on, control + "\n\n" + rule)
        self.assertIn(f"gitnexus context <symbol> --repo {repo}", on)
        self.assertIn(f'gitnexus query "<concept>" --repo {repo}', on)
        self.assertIn("then read only the lines you will change", on)
        # Every base sentence is kept, after the GitNexus one.
        self.assertTrue(rule.startswith("Author action rule: think briefly, then act. "
                                        "Use GitNexus"))
        self.assertTrue(rule.endswith(actors.AUTHOR_ACTION_RULE[len(
            "Author action rule: think briefly, then act. "):]))
        # No anchor build dir: the base rule, no GitNexus sentence.
        self.assertEqual(self._author_prompt(
            actors.ActorSeat(bounded=False, author_action_rule=True), {}),
            self._author_prompt(actors.ActorSeat(bounded=False), {}) + "\n\n"
            + actors.AUTHOR_ACTION_RULE)
        self.assertNotIn("GitNexus", actors.AUTHOR_ACTION_RULE)

    def test_the_author_config_allows_gitnexus_and_still_denies_builds(self):
        repo = aoc.gitnexus_repo_for(self.build)
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                if bounded:
                    config = aoc.build_actor_config(
                        role="author", lane=self.lane, build_dir=self.build, model=MODEL,
                        instructions_path=self.tmp / "i.md", lane_guard=True,
                        trim_tools=True, trim_instructions=True)
                    agent = config.get("agent", {}).get(aoc.AGENT_NAMES["author"], {})
                    oc = self.guard._Opencode(config, agent.get("permission"))
                else:
                    oc = self.guard._Opencode(aoc.build_plain_config(
                        role="author", lane=self.lane, build_dir=self.build,
                        lane_guard=True, trim_tools=True, trim_instructions=True))
                for command in (f"gitnexus context mul_mat_qX_K_q8_2_X4_T --repo {repo}",
                                f'gitnexus query "Q4Bits BlockPermuter" --repo {repo}',
                                f"gitnexus context Q4Bits --repo {repo} --content",
                                "grep -n _mm512_permutexvar_epi32 /usr/lib/gcc/"
                                "x86_64-linux-gnu/15/include/avx512fintrin.h"):
                    self.assertEqual(oc.bash(command), "allow", command)
                for command in ("cmake --build build -j", "make -j 16",
                                "g++ -march=native -c y.cpp", "ninja -C build",
                                # Index writers stay denied, bare or aimed at the anchor.
                                "gitnexus analyze", "gitnexus analyze --force",
                                f"gitnexus analyze {repo}", "cd x && gitnexus analyze .",
                                f"gitnexus clean --repo {repo}", "gitnexus wiki",
                                # The anchor path outside the two read forms: denied.
                                f"ls {repo}", f"gitnexus impact Q4Bits --repo {repo}",
                                f"cat {repo}/ggml/src/x.c"):
                    self.assertEqual(oc.bash(command), "deny", command)

    def test_planner_and_critic_get_no_gitnexus_allow(self):
        repo = aoc.gitnexus_repo_for(self.build)
        for role in ("planner", "critic"):
            with self.subTest(role=role):
                oc = self.guard._Opencode(aoc.build_plain_config(
                    role=role, lane=self.lane, build_dir=self.build,
                    lane_guard=True, trim_tools=True, trim_instructions=True))
                self.assertEqual(oc.bash(f"gitnexus context Q4Bits --repo {repo}"), "deny")
                self.assertEqual(oc.bash("gitnexus analyze"), "deny")


@unittest.skipUnless(OPENCODE and REAL_GLOBAL.is_file()
                     and "http://127.0.0.1:8083/v1" in REAL_GLOBAL.read_text(errors="replace"),
                     "needs the installed opencode and the host's qwen-gpu global config")
class OpencodeMockBodyMedium(unittest.TestCase):
    """The installed opencode against a recording mock (no model server anywhere)."""

    def test_default_author_body(self):
        records, _config, err = run_opencode_against_mock("author", _default_seat())
        bodies = _chat_bodies(records)
        self.assertTrue(bodies, f"opencode sent no chat completion; stderr: {err}")
        for body in bodies:
            self.assertEqual(body.get("chat_template_kwargs"), MEDIUM)
        # 40960 > opencode's 32000 clamp: the env ceiling lets it reach the wire.
        self.assertTrue(any(b.get("max_tokens") == 40960 for b in bodies),
                        [b.get("max_tokens") for b in bodies])

    def test_default_planner_body(self):
        records, _config, err = run_opencode_against_mock("planner", _default_seat())
        bodies = _chat_bodies(records)
        self.assertTrue(bodies, f"opencode sent no chat completion; stderr: {err}")
        for body in bodies:
            self.assertNotIn("chat_template_kwargs", body)
        self.assertTrue(any(b.get("max_tokens") == 16384 for b in bodies))


if __name__ == "__main__":
    unittest.main()
