"""The actor layer, tested without spending a provider call.

The two things that must hold before this touches a real API: consecutive failures
back off (a codex 401 produced 284 failures in 23 minutes with zero delay), and the
context bundle actually carries what the old planner never received.
"""
import copy
import json
import subprocess
from pathlib import Path
import unittest
from unittest import mock

from autokernel.loop import actors
from autokernel.loop.loop import Hypothesis


class Backoff(unittest.TestCase):

    def test_a_transient_streak_backs_off_exponentially(self):
        slept = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise actors.ProviderTransient("401")
            return "ok"

        result, streak = actors._with_backoff(flaky, sleep=slept.append)
        self.assertEqual(result, "ok")
        self.assertEqual(streak, 2)
        self.assertEqual(slept, [actors.BACKOFF_S[0], actors.BACKOFF_S[1]])

    def test_it_gives_up_rather_than_spinning(self):
        """284 failures in 23 minutes is what no bound looks like."""
        slept = []

        def always():
            raise actors.ProviderTransient("401")

        with self.assertRaises(actors.ProviderTransient) as caught:
            actors._with_backoff(always, sleep=slept.append)
        self.assertIn("consecutive", str(caught.exception))
        self.assertEqual(len(slept), len(actors.BACKOFF_S) - 1)

    def test_a_first_try_success_sleeps_not_at_all(self):
        slept = []
        result, streak = actors._with_backoff(lambda: "fine", sleep=slept.append)
        self.assertEqual((result, streak, slept), ("fine", 0, []))


class JsonExtraction(unittest.TestCase):

    def test_it_takes_the_last_complete_object(self):
        text = ('thinking out loud {"draft": 1}\nfinal answer:\n'
                '{"mechanism_id": "akm-x", "statement": "s"}\n')
        self.assertEqual(actors._extract_json(text)["mechanism_id"], "akm-x")

    def test_nested_objects_do_not_confuse_it(self):
        self.assertEqual(
            actors._extract_json('{"a": {"b": 2}, "c": 3}')["c"], 3)

    def test_no_json_is_a_transient_not_a_crash(self):
        with self.assertRaises(actors.ProviderTransient):
            actors._extract_json("I could not complete this task.")

    def test_malformed_json_is_a_transient(self):
        with self.assertRaises(actors.ProviderTransient):
            actors._extract_json("{not: valid}")


class ContextBundle(unittest.TestCase):
    """Everything rendered here is something the old loop measured and discarded."""

    def test_the_profile_reaches_the_actor(self):
        text = actors.render_context({"kernel_hotspots": [
            {"signature": "mul_mat_vec_q<Q4_K>", "total_duration_ns": 700000,
             "calls": 13803, "share_of_device_time": 0.7}]})
        self.assertIn("mul_mat_vec_q<Q4_K>", text)
        self.assertIn("70.00%", text)
        self.assertIn("13803", text)

    def test_an_absent_profile_says_so_rather_than_inviting_a_guess(self):
        text = actors.render_context({})
        self.assertIn("no profile yet", text)

    def test_prior_refusals_are_rendered_as_things_to_answer(self):
        text = actors.render_context(
            {"prior_hypothesis_rejections": ["already measured null in epoch 4de6"]})
        self.assertIn("already measured null in epoch 4de6", text)
        self.assertIn("answer these, do not re-derive", text)

    def test_a_stale_epoch_record_is_marked_not_comparable(self):
        text = actors.render_context({"prior_experiments": [
            {"mechanism_id": "akm-old", "status": "screened_out",
             "effect_fraction": 0.001, "stale_epoch": True}]})
        self.assertIn("akm-old", text)
        self.assertIn("STALE EPOCH", text)
        self.assertIn("NUMBER is not", text)

    def test_a_refusal_reason_from_memory_is_carried(self):
        text = actors.render_context({"prior_experiments": [
            {"mechanism_id": "akm-old", "status": "authoring_refused",
             "refusal_reason": "derives undeclared symbols ['<file-scope>']"}]})
        self.assertIn("derives undeclared symbols", text)

    def test_the_operator_inbox_is_surfaced(self):
        text = actors.render_context({"inbox": ["try IQ4_XS at the 64-VGPR knee"]})
        self.assertIn("IQ4_XS", text)
        self.assertIn("Operator suggestions", text)

    def test_repeated_family_failures_force_a_higher_level_diagnostic(self):
        rows = [
            {"status": "measured_null", "mechanism_id": f"akm-barrier-{name}",
             "statement": "change the OpenMP barrier implementation"}
            for name in ("spin", "yield", "tree")
        ]
        text = actors.render_context({"prior_experiments": rows})
        self.assertIn("DIMINISHING-RETURNS ESCAPE", text)
        self.assertIn("synchronization/barrier", text)
        self.assertIn("MUST target one of graph scheduling", text)
        self.assertIn("expert/load balance", text)

    def test_two_family_failures_do_not_force_an_early_escape(self):
        rows = [
            {"status": "refused_at_formation", "mechanism_id": f"akm-q4k-{name}"}
            for name in ("a", "b")
        ]
        text = actors.render_context({"prior_experiments": rows})
        self.assertNotIn("DIMINISHING-RETURNS ESCAPE", text)

    def test_harness_failures_do_not_exhaust_a_mechanism_family(self):
        rows = [
            {"status": "bench_failed", "mechanism_id": f"akm-barrier-{name}"}
            for name in ("a", "b", "c")
        ]
        text = actors.render_context({"prior_experiments": rows})
        self.assertNotIn("DIMINISHING-RETURNS ESCAPE", text)


class PlannerContract(unittest.TestCase):

    def test_original_cpu_target_reaches_planner_author_and_critic_without_gpu_constraints(self):
        from .test_glm_frozen_requests import _canonical_launch

        _template, selected = _canonical_launch(18311)
        target = {"scope": "experimental candidate, NOT canonical champion",
                  "recipe": selected.to_dict(), "requests": "/original/frozen-prompts.json",
                  "hotspot_status": "CPU profile unavailable; do not infer GPU hotspots"}
        context = {"target": target, "program": "Preserve original target; source edits only."}
        hypothesis = {"mechanism_id": "akm-cpu", "statement": "inspect CPU loop",
                      "falsifier": "paired serving regresses",
                      "target_surface": "ggml/src/ggml-cpu/ggml-cpu.cpp", "target_symbol": "cpu_loop"}
        captured = []

        def reply(prompt, **kwargs):
            captured.append(prompt)
            if prompt.startswith("Implement"):
                return json.dumps({"paths": [hypothesis["target_surface"]]})
            if prompt.startswith("Review"):
                return '{"accepted": true, "reason": "fixture reply"}'
            return json.dumps(hypothesis)

        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        critic = actors.AgentCritic(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent", side_effect=reply), \
                mock.patch.object(actors.subprocess, "run", return_value=mock.Mock(
                    stdout=" M " + hypothesis["target_surface"])):
            proposed = planner.propose(context)
            self.assertEqual(planner.author(proposed, context), (hypothesis["target_surface"],))
            self.assertTrue(critic.review_hypothesis(proposed, context).accepted)
            for prompt in captured:
                self.assertIn(json.dumps(target, indent=2, sort_keys=True), prompt)
                self.assertNotIn("AMD MI210", prompt)
                self.assertNotIn("ggml/src/ggml-cuda/", prompt)
            self.assertTrue(captured[0].startswith(
                "You are proposing ONE kernel optimisation for llama.cpp on the CPUs"))
            self.assertIn("do not invent timing evidence", captured[0])
            self.assertIn("DO NOT BUILD, COMPILE, BENCHMARK OR TEST", captured[1])
            self.assertIn(json.dumps({"paths": [hypothesis["target_surface"]]}), captured[1])
            self.assertIn("invents unavailable evidence as an established fact", captured[2])
            captured.clear()
            planner.propose({})
            planner.author(proposed, {})
            critic.review_hypothesis(proposed, {})
        self.assertTrue(captured[0].startswith(
            "You are proposing ONE kernel optimisation for llama.cpp on an AMD MI210 (gfx90a, ROCm 6.2)."))
        self.assertIn('"target_surface": "<one path under ggml/src/ggml-cuda/>"', captured[0])
        self.assertIn('{"paths": ["ggml/src/ggml-cuda/<file>"]}', captured[1])
        self.assertIn("negligible device-time share", captured[2])

    def test_cpu_planner_uses_existing_ab_instead_of_promising_unsupported_trace(self):
        context = {"target": {"resource_class": "cpu"},
                   "prior_hypothesis_rejections": [
                       "no frozen trace establishes the eligible-call fraction"]}
        payload = ('{"mechanism_id": "akm-row", "statement": "bound row loop", '
                   '"falsifier": "matched A/B is non-positive", '
                   '"target_surface": "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp", '
                   '"target_symbol": "mul_mat"}')
        with mock.patch.object(actors, "_run_agent", return_value=payload) as run:
            actors.AgentPlanner(workspace=Path("/tmp")).propose(context)
        prompt = run.call_args.args[0]
        self.assertIn("Do not make an unsupported trace or counter a prerequisite", prompt)
        self.assertIn("existing matched A/B can test", prompt)

    def test_a_complete_hypothesis_parses(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        payload = ('{"mechanism_id": "akm-q4k-branchless", "statement": "s", '
                   '"falsifier": "f", "target_surface": "ggml/src/ggml-cuda/mmvq.cu", '
                   '"target_symbol": "vec_dot_q4_K_q8_1"}')
        with mock.patch.object(actors, "_run_agent", return_value=payload):
            got = planner.propose({})
        self.assertIsInstance(got, Hypothesis)
        self.assertEqual(got.mechanism_id, "akm-q4k-branchless")

    def test_an_incomplete_hypothesis_is_a_transient(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"mechanism_id": "akm-x"}'):
            with self.assertRaises(actors.ProviderTransient) as caught:
                planner.propose({})
        self.assertIn("missing", str(caught.exception))

    def test_proposal_can_abstain_with_a_reason(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"abstain": "profile has no reachable hot path"}'):
            got = planner.propose({})
        self.assertIsInstance(got, actors.Abstain)
        self.assertEqual(got.reason, "profile has no reachable hot path")

    def test_unconfigured_runtime_treatment_is_not_a_provider_transient(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        context = {"target": {"resource_class": "cpu"},
                   "runtime_preparation": {"status": "unavailable",
                       "reason": "explicit prospective statistics are missing"}}
        payload = ('{"mechanism_id": "akm-threads", "statement": "s", '
                   '"falsifier": "f", "target_surface": "threads", '
                   '"target_symbol": "threads", '
                   '"runtime_treatment": {"kind": "threads", "candidate": 32}}')
        with mock.patch.object(actors, "_run_agent", return_value=payload) as invoked:
            got = planner.propose(context)
        self.assertIsInstance(got, actors.Abstain)
        self.assertIn("explicit prospective statistics", got.reason)
        self.assertNotIn("Alternatively propose ONE runtime treatment", invoked.call_args.args[0])

    def test_reasonless_abstention_is_a_malformed_provider_reply(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent", return_value='{"abstain": ""}'):
            with self.assertRaises(actors.ProviderTransient):
                planner.propose({})

    def test_legacy_empty_author_paths_are_an_abstention(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent", return_value='{"paths": []}'):
            got = planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertIsInstance(got, actors.Abstain)
        self.assertEqual(got.reason, "authoring returned no changed paths")

    def test_authoring_can_abstain_without_dirty_path_check(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"abstain": "required API is unavailable"}'), \
                mock.patch.object(actors.subprocess, "run") as status:
            got = planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertIsInstance(got, actors.Abstain)
        status.assert_not_called()

    def test_author_prompt_names_abstention_as_a_correct_result(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"abstain": "infeasible"}') as run:
            planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        prompt = run.call_args.args[0]
        self.assertIn("abstaining is a correct science result", prompt)
        self.assertIn('{"abstain":', prompt)

    def test_abstention_history_feeds_the_next_planner_context(self):
        text = actors.render_context({"prior_experiments": [{
            "status": "abstained", "mechanism_id": "akm-infeasible",
            "refusal_reason": "required primitive is absent"}]})
        self.assertIn("`akm-infeasible` → abstained", text)
        self.assertIn("required primitive is absent", text)


class CriticContract(unittest.TestCase):

    def test_cpu_payoff_evidence_is_post_authoring_not_a_formation_gate(self):
        context = {"target": {"recipe": {"backend": "cpu"}},
                   "cpu_profile": {"status": "unavailable", "reason": "fixture"}}
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"accepted": true}') as run:
            review = actors.AgentCritic(workspace=Path("/tmp")).review_hypothesis(
                Hypothesis("akm-row", "bound row loop", "matched A/B is non-positive",
                           "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp", "mul_mat"), context)
        self.assertTrue(review.accepted)
        prompt = run.call_args.args[0]
        self.assertIn("Do NOT reject", prompt)
        self.assertIn("ordinary post-authoring falsifiers", prompt)
        self.assertIn("source reachability or safety", prompt)

    def test_a_reasonless_rejection_is_made_explicit_not_crashed_on(self):
        """The loop refuses a reasonless rejection; the critic must not hand it one."""
        critic = actors.AgentCritic(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"accepted": false}'):
            review = critic.review_hypothesis(
                Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertFalse(review.accepted)
        self.assertIn("without stating a reason", review.reason)

    def test_an_acceptance_passes_through(self):
        critic = actors.AgentCritic(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"accepted": true}'):
            self.assertTrue(critic.review_hypothesis(
                Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {}).accepted)

    def test_critic_names_identity_independence_and_evidence(self):
        critic = actors.AgentCritic(workspace=Path("/tmp"))
        context = {"actor_provenance": {
            "planner": actors.PLANNER_DEFAULT.describe(),
            "critic": critic.backend.describe(),
        }}
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"accepted": true}'):
            review = critic.review_hypothesis(
                Hypothesis("akm-x", "s", "f", "a.cu", "sym"), context)
        self.assertEqual(review.validator_identity, critic.backend.describe())
        self.assertEqual(review.validator_kind, "llm_critic")
        self.assertEqual(review.independence, "different_family")
        self.assertEqual(review.evidence_inspected,
                         ("review subject", "rejection grounds", "planner context"))


class PlaceholderEchoes(unittest.TestCase):
    """The first real run parsed the prompt's own template as the answer.

    `_extract_json` takes the LAST complete JSON object in stdout, and codex echoes
    the requested shape while reasoning. So `{"paths": ["<relative path you
    changed>"]}` -- the spec, not the reply -- was parsed as a result, and the loop
    recorded a transient while codex was still working.
    """

    def test_the_template_is_recognised_as_a_placeholder(self):
        for echoed in ("<relative path you changed>", "akm-<short-slug>",
                       "<the function you will change>", "e.g. mmvq.cu",
                       "  YOUR PATH HERE  "):
            self.assertTrue(actors._is_placeholder(echoed), echoed)

    def test_a_real_path_is_not_a_placeholder(self):
        for real in ("ggml/src/ggml-cuda/vecdotq.cuh", "mmvq.cu",
                     "akm-q4k-branchless"):
            self.assertFalse(actors._is_placeholder(real), real)

    def test_a_real_answer_containing_an_angle_bracket_survives(self):
        """The guard's first version listed a bare `"<"` and retired three
        consecutive hypotheses the planner had answered correctly.

        A falsifier states a threshold and a statement names a C++ template, so both
        legitimately carry `<`. Rejecting them is the guard forbidding its own
        compliant idiom -- the failure class this rebuild exists to remove.
        """
        for real in (
            "pp512 median delta < 0.97% over 5 alternating pairs",
            "mul_mat_vec_q<(ggml_type)12, 1, true> loses its per-block branch",
            "no change if occupancy stays < 8 waves/SIMD",
            "vec_dot_q4_K_q8_1 dispatch count drops, and tg128 is unchanged",
        ):
            self.assertFalse(actors._is_placeholder(real), real)

    def test_authoring_refuses_an_echoed_template(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"paths": ["<relative path you changed>"]}'):
            with self.assertRaises(actors.ProviderTransient) as caught:
                planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertIn("echoed the prompt template", str(caught.exception))

    def test_a_hypothesis_echoing_the_template_is_refused(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        payload = ('{"mechanism_id": "akm-<short-slug>", "statement": "s", '
                   '"falsifier": "f", "target_surface": "a.cu", '
                   '"target_symbol": "<the function you will change>"}')
        with mock.patch.object(actors, "_run_agent", return_value=payload):
            with self.assertRaises(actors.ProviderTransient) as caught:
                planner.propose({})
        self.assertIn("echoed the prompt template", str(caught.exception))


class TheWorktreeIsTheGroundTruth(unittest.TestCase):
    """An actor that SAYS it changed a file and did not must not pass."""

    def test_an_unchanged_worktree_refuses_the_claimed_path(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"paths": ["ggml/src/ggml-cuda/mmvq.cu"]}'), \
             mock.patch.object(actors.subprocess, "run") as ran:
            ran.return_value = mock.Mock(stdout="")          # git status: clean
            with self.assertRaises(actors.ProviderTransient) as caught:
                planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertIn("worktree is unchanged", str(caught.exception))

    def test_a_genuinely_changed_worktree_passes(self):
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"paths": ["ggml/src/ggml-cuda/mmvq.cu"]}'), \
             mock.patch.object(actors.subprocess, "run") as ran:
            ran.return_value = mock.Mock(stdout=" M ggml/src/ggml-cuda/mmvq.cu")
            got = planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertEqual(got, ("ggml/src/ggml-cuda/mmvq.cu",))


class TheActorMustNotSpendTheLoopsCompute(unittest.TestCase):
    def test_the_authoring_prompt_forbids_building(self):
        """codex started its own `cmake --build -j 16` on the first real run."""
        planner = actors.AgentPlanner(workspace=Path("/tmp"))
        captured = {}

        def capture(prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"paths": ["ggml/src/ggml-cuda/mmvq.cu"]}'

        with mock.patch.object(actors, "_run_agent", side_effect=capture), \
             mock.patch.object(actors.subprocess, "run") as ran:
            ran.return_value = mock.Mock(stdout=" M ggml/src/ggml-cuda/mmvq.cu")
            planner.author(Hypothesis("akm-x", "s", "f", "a.cu", "sym"), {})
        self.assertIn("DO NOT BUILD", captured["prompt"])
        self.assertIn("the loop owns the build", captured["prompt"].lower())


if __name__ == "__main__":
    unittest.main()


class Backends(unittest.TestCase):
    """The per-role model split is the operator's choice; these pin its wiring.

    Each assertion names an exact argv token, so a mutation that drops a flag,
    reorders the prompt, or un-quotes the codex TOML value is visible AND counted.
    """

    def test_the_defaults_are_the_operator_choice(self):
        self.assertEqual(actors.PLANNER_DEFAULT.describe(), "codex:gpt-5.6-sol@high")
        self.assertEqual(actors.CRITIC_DEFAULT.describe(), "claude:claude-fable-5-1@medium")
        self.assertIs(actors.AgentPlanner(workspace=Path("/tmp")).backend,
                      actors.PLANNER_DEFAULT)
        self.assertIs(actors.AgentCritic(workspace=Path("/tmp")).backend,
                      actors.CRITIC_DEFAULT)

    def test_a_claude_model_routes_to_the_claude_cli(self):
        b = actors.backend_for("claude-opus-5", "high")
        argv = b.argv("PROMPT", Path("/ws"))
        self.assertEqual(argv[0], actors.CLAUDE)
        self.assertEqual(argv[1], "-p")
        self.assertIn("--dangerously-skip-permissions", argv)
        self.assertIn("--no-session-persistence", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "claude-opus-5")
        self.assertEqual(argv[argv.index("--effort") + 1], "high")
        self.assertEqual(argv[argv.index("--output-format") + 1], "text")
        note = argv[argv.index("--append-system-prompt") + 1]
        self.assertIn("DETACHED git worktree", note)
        self.assertIn("Never build", note)
        self.assertEqual(argv[-1], "PROMPT")

    def test_critic_backends_are_constructed_read_only(self):
        claude = actors.backend_for("claude-opus-5", "high").argv(
            "PROMPT", Path("/ws"), read_only=True)
        self.assertNotIn("--dangerously-skip-permissions", claude)
        self.assertEqual(claude[claude.index("--permission-mode") + 1], "plan")
        critic_note = claude[claude.index("--append-system-prompt") + 1]
        self.assertIn("read-only AutoKernel critic", critic_note)
        self.assertIn("Do not edit", critic_note)
        codex = actors.backend_for("gpt-5.6-sol", "high").argv(
            "PROMPT", Path("/ws"), read_only=True)
        self.assertEqual(codex[codex.index("-s") + 1], "read-only")

    def test_any_other_model_routes_to_codex_with_quoted_toml_effort(self):
        b = actors.backend_for("gpt-5.6-sol", "high")
        argv = b.argv("PROMPT", Path("/ws"))
        self.assertEqual(argv[:3], [actors.CODEX, "exec", "--skip-git-repo-check"])
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-5.6-sol")
        # codex parses -c as TOML: an unquoted bare word is rejected, so the quotes
        # are load-bearing, not cosmetic.
        self.assertEqual(argv[argv.index("-c") + 1], 'model_reasoning_effort="high"')
        self.assertEqual(argv[argv.index("-C") + 1], "/ws")
        self.assertEqual(argv[-1], "PROMPT")

    def test_a_provider_slash_model_routes_to_opencode(self):
        b = actors.backend_for("deepseek/deepseek-v4-flash", "max")
        self.assertEqual(b.kind, "opencode")
        argv = b.argv("PROMPT", Path("/ws"))
        self.assertEqual(argv[0], actors.OPENCODE)
        self.assertEqual(argv[1], "run")
        self.assertIn("--auto", argv)
        self.assertEqual(argv[argv.index("--dir") + 1], "/ws")
        self.assertEqual(argv[argv.index("-m") + 1], "deepseek/deepseek-v4-flash")
        # opencode calls reasoning effort a "variant"; "max" must reach it.
        self.assertEqual(argv[argv.index("--variant") + 1], "max")
        # The prompt rides stdin: opencode re-quotes and backslash-escapes any
        # positional that contains a space (DS41 2026-09-24).
        self.assertNotIn("PROMPT", argv)
        self.assertEqual(b.stdin_payload("PROMPT"), "PROMPT")
        self.assertNotIn("--agent", argv)
        named = actors.Backend("opencode", "q/m", "high", actors.OPENCODE, agent="autokernel-planner")
        agent_argv = named.argv("PROMPT", Path("/ws"))
        self.assertEqual(agent_argv[agent_argv.index("--agent") + 1], "autokernel-planner")

    def test_bounded_seat_writes_a_per_run_config_outside_the_worktree(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            profiles = Path(tmp) / "store" / "cpu-profiles"
            (profiles / "cpu-raw-abc").mkdir(parents=True)
            context = {"cpu_profile": {"record": str(profiles / "cpu-raw-abc" / "measurement-record.data")}}
            planner = actors.AgentPlanner(workspace=ws, backend=actors.backend_for("q/m", "high"),
                                          seat=actors.ActorSeat(tools_python="/py"))
            backend, env = planner._seated("planner", context)
            self.assertEqual(backend.agent, "autokernel-planner")
            config = Path(env["OPENCODE_CONFIG"])
            self.assertEqual(config.parent, ws.parent, "the config must not ride into the diff")
            self.assertFalse(list(ws.iterdir()))
            body = json.loads(config.read_text())
            command = next(iter(body["mcp"].values()))["command"]
            self.assertEqual(command[0], "/py")
            self.assertEqual(command[command.index("--root") + 1], str(ws))
            self.assertEqual(command[command.index("--profiles") + 1], str(profiles))

    def test_plain_seat_and_non_opencode_backends_are_untouched(self):
        ws = Path("/ws")
        for backend, seat in ((actors.backend_for("q/m", "high"), actors.ActorSeat(bounded=False)),
                              (actors.backend_for("q/m", "high"), None),
                              (actors.backend_for("gpt-5.6-sol", "high"), actors.ActorSeat())):
            planner = actors.AgentPlanner(workspace=ws, backend=backend, seat=seat)
            self.assertEqual(planner._seated("planner", {}), (backend, None))

    def test_run_agent_passes_env_and_logs_the_call(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            done = subprocess.CompletedProcess(args=["x"], returncode=0, stdout="{}", stderr="")
            with mock.patch.object(actors.subprocess, "run", return_value=done) as ran:
                actors._run_agent("p", workspace=ws, backend=actors.backend_for("q/m", "high"),
                                  env={"OPENCODE_CONFIG": "/c.json"})
            # The actor's own call, not the actor_metrics session-list calls this seat
            # now also makes (they carry no `env` kwarg) -- find it by that kwarg
            # rather than assuming it is the last (or only) `subprocess.run` call.
            main_call = next(c for c in ran.call_args_list if "env" in c.kwargs)
            self.assertEqual(main_call.kwargs["env"]["OPENCODE_CONFIG"], "/c.json")
            self.assertIn("PATH", main_call.kwargs["env"], "env extends, never replaces")
            rows = (ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG).read_text().splitlines()
            row = json.loads(rows[-1])
            self.assertEqual((row["returncode"], row["prompt_chars"], row["opencode_config"]), (0, 1, "/c.json"))

    def test_target_dedupe_points_repeats_at_the_first_copy(self):
        dsos = [{"path": f"/lib/lib{i}.so", "sha256": "a" * 64} for i in range(4)]
        target = {"recipe": {"dsos": dsos, "threads": 48},
                  "scope": {"full": {"dsos": dsos, "argv": ["-t", "48"]}}}
        out = actors._dedupe_subtrees(target)
        self.assertEqual(out["recipe"]["dsos"], dsos)
        self.assertEqual(out["scope"]["full"]["dsos"], "<same as $.recipe.dsos>")
        self.assertEqual(out["scope"]["full"]["argv"], ["-t", "48"], "small subtrees are never folded")
        rendered = actors.render_context({"target": target})
        self.assertIn("<same as $.recipe.dsos>", rendered)

    def test_shared_history_drops_hashes_and_clips_prose_but_keeps_caveats(self):
        shared = {"use": "suggestions only", "rows": [
            {"mechanism_id": "akm-x", "status": "measured_null", "attempt_id": "f" * 64,
             "result_sha256": "e" * 64, "refusal_reason": "r" * 2000,
             "unknown_reason": "original window was cold"}]}
        row = actors._slim_shared_history(shared)["rows"][0]
        self.assertNotIn("attempt_id", row); self.assertNotIn("result_sha256", row)
        self.assertEqual(row["unknown_reason"], "original window was cold")
        self.assertTrue(row["refusal_reason"].endswith("[clipped 1500 chars]"))
        self.assertEqual(actors._slim_shared_history({"no": "rows"}), {"no": "rows"})

    def test_codex_and_claude_keep_the_prompt_in_argv(self):
        for model in ("gpt-5.6-sol", "claude-fable-5-1"):
            b = actors.backend_for(model, "high")
            self.assertEqual(b.argv("PROMPT", Path("/ws"))[-1], "PROMPT")
            self.assertIsNone(b.stdin_payload("PROMPT"))

    def test_run_agent_feeds_the_opencode_prompt_on_stdin(self):
        b = actors.backend_for("prov/model", "high")
        done = mock.Mock(returncode=0, stdout="{}", stderr="")
        with mock.patch.object(actors.subprocess, "run", return_value=done) as ran, \
             mock.patch.object(actors, "_persist_reply"):
            actors._run_agent('say "hi" to me', workspace=Path("/ws"), backend=b)
        self.assertEqual(ran.call_args.kwargs["input"], 'say "hi" to me')
        self.assertNotIn('say "hi" to me', ran.call_args.args[0])

    def test_an_unknown_kind_refuses_rather_than_guessing(self):
        with self.assertRaises(ValueError):
            actors.Backend("gemini", "x", "y", "/bin/x").argv("p", Path("/ws"))

    def test_run_agent_invokes_the_backend_argv_in_the_workspace(self):
        b = actors.backend_for("claude-fable-5-1", "medium")
        done = mock.Mock(returncode=0, stdout="{}", stderr="")
        with mock.patch.object(actors.subprocess, "run", return_value=done) as ran:
            actors._run_agent("PROMPT", workspace=Path("/ws"), backend=b)
        argv = ran.call_args.args[0]
        self.assertEqual(argv, b.argv("PROMPT", Path("/ws")))
        self.assertEqual(ran.call_args.kwargs["cwd"], "/ws")



class RawReplyPersistenceAndStreamFallback(unittest.TestCase):
    """A complete reply must never become a transient because of which stream
    carried it, and every raw exchange must be on disk (DS41 2026-09-24)."""

    def _done(self, stdout: str, stderr: str, rc: int = 0):
        return subprocess.CompletedProcess(args=["x"], returncode=rc, stdout=stdout, stderr=stderr)

    def test_reply_on_stderr_is_still_handed_to_the_parser(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            body = '{"mechanism_id":"m","statement":"s","falsifier":"f","target_surface":"t","target_symbol":"y"}'
            with mock.patch.object(actors.subprocess, "run",
                                   return_value=self._done("chrome only\n", "final: " + body)):
                raw = actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"))
            self.assertEqual(actors._extract_json(raw)["mechanism_id"], "m")
            replies = sorted(p for p in (ws.parent / actors.ACTOR_REPLY_DIR).iterdir()
                             if p.name != actors.ACTOR_CALL_LOG)
            self.assertEqual([p.suffix for p in replies], [".stderr", ".stdout"])
            self.assertIn("final:", replies[0].read_text())
            self.assertFalse(list(ws.iterdir()), "nothing may land inside the worker tree")

    def test_incomplete_stdout_no_longer_suppresses_a_complete_stderr_reply(self):
        """TD-21.30(a): the old schema-BLIND probe treated any parseable JSON on
        stdout as "an answer is here", so an incomplete object on stdout (right
        shape, wrong content) silently discarded a COMPLETE object on stderr
        instead of handing both to `_parse_reply`."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            incomplete_stdout = '{"type": "progress", "step": 3}'
            complete_stderr = 'final: {"accepted": true, "reason": "looks correct"}'
            with mock.patch.object(actors.subprocess, "run",
                                   return_value=self._done(incomplete_stdout, complete_stderr)):
                raw = actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"),
                                        read_only=True, schema=actors.REVIEW_SCHEMA)
            self.assertEqual(actors._extract_json(raw), {"accepted": True, "reason": "looks correct"})

    def test_schema_blind_probe_still_used_when_no_schema_is_known(self):
        """`schema=None` (a call site with nothing to give) keeps the original
        "any JSON found" behaviour rather than refusing to ever merge."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            with mock.patch.object(actors.subprocess, "run",
                                   return_value=self._done("chrome only\n", 'final: {"x": 1}')):
                raw = actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"))
            self.assertIn('"x": 1', raw)

    def test_stdout_reply_is_returned_unchanged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            with mock.patch.object(actors.subprocess, "run",
                                   return_value=self._done('{"abstain":"x"}', '{"not":"this"}')):
                raw = actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"))
            self.assertEqual(raw, '{"abstain":"x"}')

    def test_nonzero_exit_is_still_a_transient_and_still_persisted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            with mock.patch.object(actors.subprocess, "run", return_value=self._done("", "boom", rc=3)):
                with self.assertRaises(actors.ProviderTransient):
                    actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"))
            self.assertTrue(any(p.name.endswith("-rc3.stderr") for p in (ws.parent / actors.ACTOR_REPLY_DIR).iterdir()))

    def test_nonzero_exit_with_a_complete_reply_is_salvaged(self):
        """DS41 2026-09-24 09:55: opencode exited 1 after a recovered tool error and
        its complete hypothesis was retried from zero."""
        import tempfile
        body = ('{"mechanism_id":"m","statement":"s","falsifier":"f",'
                '"target_surface":"a/b.cpp","target_symbol":"fn"}')
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            with mock.patch.object(actors.subprocess, "run",
                                   return_value=self._done("report...\n" + body, "Error: trim", rc=1)):
                raw = actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"),
                                        schema=actors.HYPOTHESIS_SCHEMA)
            self.assertEqual(actors._extract_json(raw)["mechanism_id"], "m")
            self.assertTrue(any(p.name.endswith("-rc1.stdout") for p in (ws.parent / actors.ACTOR_REPLY_DIR).iterdir()))

    def test_nonzero_exit_with_an_incomplete_object_stays_a_transient(self):
        """A crashed critic whose stray JSON lacks `accepted` must not become a rejection."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            for out in ('{"type": "error", "message": "rate limited"}', ""):
                with mock.patch.object(actors.subprocess, "run", return_value=self._done(out, "", rc=1)):
                    with self.assertRaises(actors.ProviderTransient):
                        actors._run_agent("p", workspace=ws, backend=actors.backend_for("gpt-6-sol", "high"),
                                          read_only=True, schema=actors.REVIEW_SCHEMA)

    def test_a_long_real_stdout_is_captured_whole_with_the_final_json_last(self):
        """A real child process, no mock: 300 KB of chrome and then the reply. The
        reply is the tail, which a truncating pipe would lose first."""
        import sys
        import tempfile
        code = ("import sys; sys.stdout.write('x' * 300000 + '\\n'); "
                "sys.stdout.write('{\"accepted\": true, \"reason\": \"ok\"}'); sys.stdout.flush()")
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            backend = actors.backend_for("gpt-6-sol", "high")
            with mock.patch.object(actors.Backend, "argv", return_value=[sys.executable, "-c", code]):
                raw = actors._run_agent("p", workspace=ws, backend=backend, read_only=True,
                                        schema=actors.REVIEW_SCHEMA)
            self.assertGreater(len(raw), 300000)
            self.assertEqual(actors._extract_json(raw), {"accepted": True, "reason": "ok"})

    def test_signal_death_is_never_salvaged(self):
        """A killed actor never finished: its stdout may be a compaction summary."""
        import tempfile
        body = ('{"mechanism_id":"m","statement":"s","falsifier":"f",'
                '"target_surface":"a/b.cpp","target_symbol":"fn"}')
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            with mock.patch.object(actors.subprocess, "run", return_value=self._done(body, "", rc=-15)):
                with self.assertRaises(actors.ProviderTransient):
                    actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"),
                                      schema=actors.HYPOTHESIS_SCHEMA)

    def test_compaction_summary_quoting_the_template_is_not_a_reply(self):
        """Bounded-seat A/B 2026-09-24: opencode's compaction summary on stdout quoted
        the output contract, and `{"abstain":"<reason>"}` was taken as the answer."""
        summary = ('## Important Details\n- **Output contract**: Reply with ONE json object '
                   '`{"mechanism_id":"akm-<slug>","statement":"...","falsifier":"...",'
                   '"target_surface":"<one source path>","target_symbol":"<function>"}` '
                   'or `{"abstain":"<reason>"}`.\n')
        real = ('{"mechanism_id":"akm-x","statement":"s","falsifier":"f",'
                '"target_surface":"a/b.cpp","target_symbol":"fn"}')
        self.assertEqual(actors._extract_json(real + "\n" + summary)["mechanism_id"], "akm-x",
                         "a real object beats a later template echo")
        self.assertTrue(actors._is_template_echo(actors._extract_json(summary)),
                        "with nothing else, the echo is still surfaced for the guards")
        with self.assertRaises(actors.ProviderTransient):
            actors._abstention({"abstain": "<reason>"})
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            with mock.patch.object(actors.subprocess, "run", return_value=self._done(summary, "", rc=1)):
                with self.assertRaises(actors.ProviderTransient):
                    actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"),
                                      schema=actors.HYPOTHESIS_SCHEMA)
            with self.assertRaises(actors.ProviderTransient):
                actors._parse_reply(summary, schema=actors.HYPOTHESIS_SCHEMA,
                                    backend=actors.backend_for("gpt-6-sol", "high"), workspace=ws)

    def test_nonzero_exit_abstention_is_salvaged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            with mock.patch.object(actors.subprocess, "run",
                                   return_value=self._done('{"abstain": "nothing reachable"}', "", rc=1)):
                raw = actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"),
                                        schema=actors.HYPOTHESIS_SCHEMA)
            self.assertEqual(actors._extract_json(raw), {"abstain": "nothing reachable"})


    def test_timeout_persists_the_partial_output(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True)
            exc = subprocess.TimeoutExpired(cmd=["x"], timeout=1, output=b"half a patch", stderr=b"chrome")
            with mock.patch.object(actors.subprocess, "run", side_effect=exc):
                with self.assertRaises(actors.ProviderTransient):
                    actors._run_agent("p", workspace=ws, backend=actors.backend_for("prov/model", "high"))
            files = sorted((ws.parent / actors.ACTOR_REPLY_DIR).iterdir())
            self.assertTrue(any(f.name.endswith("-rc-1.stdout") for f in files))
            self.assertIn("half a patch", [f.read_text() for f in files if f.suffix == ".stdout"][0])


class SchemaRepairTurn(unittest.TestCase):
    """A reply whose JSON is missing or incomplete gets ONE schema-constrained
    repair turn on the agent's own local server (typed-decision-plane TD-1
    idiom); it never invents, and non-local backends keep the old path."""

    def _ws(self, tmp):
        ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True); return ws

    def _server_seq(self, contents):
        class Resp:
            def __init__(self, body): self._b = body
            def read(self): return self._b
            def __enter__(self): return self
            def __exit__(self, *a): return False
        payloads = [Resp(json.dumps({"choices": [{"message": {"content": c}}]}).encode()) for c in contents]
        return mock.patch("urllib.request.urlopen", side_effect=payloads)

    def _server(self, content: str):
        class Resp:
            def __init__(self, body): self._b = body
            def read(self): return self._b
            def __enter__(self): return self
            def __exit__(self, *a): return False
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        return mock.patch.object(actors.urllib.request, "urlopen", return_value=Resp(payload)) \
            if hasattr(actors, "urllib") else mock.patch("urllib.request.urlopen", return_value=Resp(payload))

    def test_prose_only_reply_is_repaired_into_the_schema(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            fixed = {"mechanism_id": "m", "statement": "s", "falsifier": "f",
                     "target_surface": "ggml/src/ggml-cpu/ops.cpp", "target_symbol": "ggml_vec_dot_q4_K"}
            report = "I investigated and propose hoisting the scale load in ggml_vec_dot_q4_K (ops.cpp)."
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 self._server_seq([json.dumps({"explicitly_declines": False, "reason": ""}), json.dumps(fixed)]):
                body = actors._parse_reply(report, schema=actors.HYPOTHESIS_SCHEMA,
                                           backend=actors.backend_for("prov/model", "high"), workspace=ws)
            self.assertEqual(body, fixed)
            self.assertTrue(any("-rc0.stdout" in p.name for p in (ws.parent / actors.ACTOR_REPLY_DIR).iterdir()))

    def test_incomplete_object_is_repaired_and_complete_object_is_not_touched(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            fixed = {"accepted": False, "reason": "false"}  # "false" is literally in the raw reply
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 self._server_seq([json.dumps(fixed), json.dumps(fixed)]) as srv:
                body = actors._parse_reply('{"accepted": false}', schema=actors.REVIEW_SCHEMA,
                                           backend=actors.backend_for("prov/model", "high"), workspace=ws)
                self.assertEqual(body, fixed)
                calls = srv.call_count
                body2 = actors._parse_reply('{"accepted": true, "reason": ""}', schema=actors.REVIEW_SCHEMA,
                                            backend=actors.backend_for("prov/model", "high"), workspace=ws)
                self.assertEqual(srv.call_count, calls, "a complete object must not trigger a repair turn")
                self.assertTrue(body2["accepted"])

    def test_non_local_backend_keeps_the_transient(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            with self._server('{"never":1}') as srv:
                with self.assertRaises(actors.ProviderTransient):
                    actors._parse_reply("no json here", schema=actors.HYPOTHESIS_SCHEMA,
                                        backend=actors.backend_for("gpt-6-sol", "high"), workspace=ws)
                self.assertEqual(srv.call_count, 0)

    def test_explicit_decline_is_mapped_by_the_boolean_stage(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 self._server(json.dumps({"explicitly_declines": True, "reason": "nothing reachable"})) as srv:
                body = actors._parse_reply("I cannot propose anything here.", schema=actors.HYPOTHESIS_SCHEMA,
                                           backend=actors.backend_for("prov/model", "high"), workspace=ws)
            self.assertEqual(body, {"abstain": "nothing reachable"}); self.assertEqual(srv.call_count, 1)

    def test_abstention_passes_through_without_a_repair(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 self._server('{"x":1}') as srv:
                body = actors._parse_reply('{"abstain": "nothing reachable"}', schema=actors.HYPOTHESIS_SCHEMA,
                                           backend=actors.backend_for("prov/model", "high"), workspace=ws)
                self.assertEqual(body, {"abstain": "nothing reachable"}); self.assertEqual(srv.call_count, 0)


    def test_empty_reply_is_never_repaired(self):
        """DS41 2026-09-24 10:09: an empty retry reply was repaired into an invented
        hypothesis ("src/verify/replay.ts") that the critic then had to reject."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 self._server('{"x":1}') as srv:
                for raw in ("", "   \n", "ok"):
                    with self.assertRaises(actors.ProviderTransient):
                        actors._parse_reply(raw, schema=actors.HYPOTHESIS_SCHEMA,
                                            backend=actors.backend_for("prov/model", "high"), workspace=ws)
                self.assertEqual(srv.call_count, 0, "no constrained turn may run over an empty report")

    def test_repair_that_names_a_file_the_report_never_mentions_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            invented = {"mechanism_id": "replay-verification", "statement": "s", "falsifier": "f",
                        "target_surface": "src/verify/replay.ts", "target_symbol": "replay"}
            report = ("Profiling shows mul_mat_qX_K_q8_2_X4_T in iqk_gemm_kquants.cpp dominates the "
                      "drafter; I have not settled on a mechanism yet.")
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 self._server_seq([json.dumps({"explicitly_declines": False, "reason": ""}),
                                   json.dumps(invented)]):
                with self.assertRaises(actors.ProviderTransient) as caught:
                    actors._parse_reply(report, schema=actors.HYPOTHESIS_SCHEMA,
                                        backend=actors.backend_for("prov/model", "high"), workspace=ws)
            self.assertIn("target_surface", str(caught.exception))

    def test_grounding_accepts_the_real_09_55_target_fields(self):
        report = ("... the drafter Q4_K iqk X4 gemm ... ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp "
                  "mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1> inner K-block loop ...")
        body = {"target_surface": "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp :: mul_mat_qX_K_q8_2_X4_T "
                                  "(template <typename Dequantizer, int nrc_y>, lines ~814-869)",
                "target_symbol": "mul_mat_qX_K_q8_2_X4_T<(anonymous namespace)::DequantizerQ4K_AVX2, 1>"}
        self.assertEqual(actors._ungrounded_fields(body, report, actors.HYPOTHESIS_SCHEMA), [])

    def test_review_parse_never_asks_the_decline_question(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 self._server_seq([json.dumps({"accepted": False, "reason": "the hoist is invalid"})]) as srv:
                body = actors._parse_reply("The hoist is invalid. I reject it.", schema=actors.REVIEW_SCHEMA,
                                           backend=actors.backend_for("prov/model", "high"), workspace=ws)
            self.assertEqual(body, {"accepted": False, "reason": "the hoist is invalid"})
            self.assertEqual(srv.call_count, 1)

    def test_wrong_typed_fished_value_is_not_short_circuited(self):
        """TD-21.30(b): `{"accepted": "true", ...}` has both REVIEW_SCHEMA required
        keys but the WRONG TYPE for `accepted` (a string, not a boolean) -- only
        required-key presence used to gate the skip-repair decision, so this
        would previously have been returned unrepaired and mistyped."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            fixed = {"accepted": True, "reason": "true stated explicitly"}
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 self._server(json.dumps(fixed)) as srv:
                body = actors._parse_reply(
                    '{"accepted": "true", "reason": "true stated explicitly"}',
                    schema=actors.REVIEW_SCHEMA,
                    backend=actors.backend_for("prov/model", "high"), workspace=ws)
            self.assertEqual(body, fixed)
            self.assertEqual(srv.call_count, 1, "a wrong-typed required field must trigger repair")
            self.assertIs(type(body["accepted"]), bool)

    def test_local_validator_matches_jsonschema_when_both_are_available(self):
        """Whichever validator TD-21.30(b) actually uses, type errors on a
        REQUIRED field are caught, not just its presence."""
        wrong_type = {"accepted": "true", "reason": "r"}
        right_type = {"accepted": True, "reason": "r"}
        self.assertFalse(actors._schema_valid(wrong_type, actors.REVIEW_SCHEMA))
        self.assertTrue(actors._schema_valid(right_type, actors.REVIEW_SCHEMA))
        self.assertFalse(actors._schema_valid({"accepted": True, "reason": "r", "extra": 1},
                                              actors.REVIEW_SCHEMA),
                         "additionalProperties: False must be enforced too")

    def test_repair_request_carries_the_model_field(self):
        """TD-21.30(e): a multi-model local endpoint would 400 on a repair
        request with no `model` field. The wire model is the bare name after
        the opencode `provider/` prefix, matching what the endpoint serves."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            captured: dict = {}

            class Resp:
                def __init__(self, body): self._b = body
                def read(self): return self._b
                def __enter__(self): return self
                def __exit__(self, *a): return False

            def fake_urlopen(request, timeout=None):
                captured["body"] = json.loads(request.data.decode())
                content = json.dumps({"accepted": True, "reason": "ok stated"})
                return Resp(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                actors._parse_reply("ok stated explicitly", schema=actors.REVIEW_SCHEMA,
                                    backend=actors.backend_for("myprovider/mymodel-7b", "high"),
                                    workspace=ws)
            self.assertEqual(captured["body"].get("model"), "mymodel-7b")

    def test_repaired_number_the_report_never_stated_is_refused(self):
        """The orchestrator's evidence guard (`structured_output/repair.py`)
        exists because a repair turn INVENTS a required field the reply never
        stated (live: an unstated `tier` repaired to `tier=2`). Mirrored here
        in `_ungrounded_fields`, which `_parse_reply` already applies to every
        repaired value -- kept local per the reference note, not imported."""
        schema = {"type": "object",
                 "properties": {"accepted": {"type": "boolean"}, "tier": {"type": "integer"}},
                 "required": ["accepted", "tier"], "additionalProperties": False}
        report = "I looked at this and it seems fine overall."
        self.assertEqual(actors._ungrounded_fields({"accepted": True, "tier": 2}, report, schema),
                         ["tier"])
        self.assertEqual(
            actors._ungrounded_fields({"accepted": True, "tier": 2},
                                      "I'd call this tier 2 work.", schema),
            [], "a number actually stated in the report is not invented")

    def test_repaired_short_string_the_report_never_stated_is_refused(self):
        schema = {"type": "object", "properties": {"accepted": {"type": "boolean"},
                                                    "reason": {"type": "string"}},
                 "required": ["accepted", "reason"], "additionalProperties": False}
        self.assertEqual(
            actors._ungrounded_fields({"accepted": False, "reason": "not applicable"},
                                      "I decline this for other reasons.", schema),
            ["reason"])
        self.assertEqual(
            actors._ungrounded_fields({"accepted": False, "reason": "other reasons"},
                                      "I decline this for other reasons.", schema),
            [])

    def test_long_string_and_boolean_leaves_are_evidence_exempt(self):
        schema = {"type": "object",
                 "properties": {"accepted": {"type": "boolean"},
                                "reason": {"type": "string"}},
                 "required": ["accepted", "reason"], "additionalProperties": False}
        long_paraphrase = "x" * 41  # over _MAX_EVIDENCE_LEAF_CHARS: a paraphrase, not a copy
        self.assertEqual(
            actors._ungrounded_fields({"accepted": True, "reason": long_paraphrase},
                                      "totally unrelated report text", schema),
            [])

    def test_mechanism_id_may_be_derived_but_a_template_echo_is_still_refused(self):
        """mechanism_id is the one field the extraction instruction explicitly
        lets the model DERIVE as a slug -- exempt from grounding/evidence --
        but our own template echoed back (`akm-<short-slug>`) is never a
        derivation."""
        report = "I looked at the drafter and found nothing worth changing yet."
        derived = {"mechanism_id": "akm-drafter-idle", "statement": "s", "falsifier": "f",
                  "target_surface": "a.cpp", "target_symbol": "fn"}
        self.assertNotIn("mechanism_id",
                        actors._ungrounded_fields(derived, report, actors.HYPOTHESIS_SCHEMA))
        echoed = {**derived, "mechanism_id": "akm-<short-slug>"}
        self.assertIn("mechanism_id",
                      actors._ungrounded_fields(echoed, report, actors.HYPOTHESIS_SCHEMA))


# --------------------------------------------------------------------------- TD-21.35: relax_required wire schema
#
# A `required` field in a JSON-schema-to-GBNF grammar FORCES a value, real or
# not (orchestrator-side live finding, 2026-09-24: an unstated `tier` repaired
# to a fabricated `tier=2`). Mirrors
# `epyc-orchestrator:src/structured_output/repair.py`
# `TestRelaxRequiredForWireUnit`/`TestRelaxRequiredIntegration` -- read there
# for the reference coverage this reproduces locally.

NESTED_UNION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "outer": {
            "type": "object",
            "properties": {
                "kind": {"const": "widget"},
                "size": {"type": "integer"},
            },
            "required": ["kind", "size"],
            "additionalProperties": False,
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "score": {"type": "number"}},
                "required": ["id", "score"],
                "additionalProperties": False,
            },
        },
        "choice": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"type": {"const": "a"}, "value": {"type": "string"}},
                    "required": ["type", "value"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"type": {"enum": ["b"]}, "count": {"type": "integer"}},
                    "required": ["type", "count"],
                    "additionalProperties": False,
                },
            ]
        },
    },
    "required": ["outer", "items"],
}
SIMPLE_RELAX_SCHEMA = {"type": "object",
                       "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                       "required": ["name", "count"], "additionalProperties": False}
ORDERED_RELAX_SCHEMA = {"type": "object",
                        "properties": {"zeta": {"type": "string"}, "alpha": {"type": "string"},
                                      "mu": {"type": "string"}},
                        "required": ["zeta", "alpha", "mu"], "additionalProperties": False}


class RelaxRequiredForWireUnit(unittest.TestCase):
    """Direct unit coverage of `_relax_required_for_wire` -- nested objects,
    array `items`, `oneOf` branches and property order, with `const`/single-
    `enum` discriminators kept and every other `required` entry dropped."""

    def test_top_level_required_dropped(self):
        relaxed = actors._relax_required_for_wire(NESTED_UNION_SCHEMA)
        self.assertNotIn("required", relaxed)

    def test_nested_object_required_dropped_but_const_kept(self):
        relaxed = actors._relax_required_for_wire(NESTED_UNION_SCHEMA)
        outer = relaxed["properties"]["outer"]
        self.assertEqual(outer["required"], ["kind"])  # `size` dropped, `kind` (const) kept

    def test_array_items_required_dropped(self):
        relaxed = actors._relax_required_for_wire(NESTED_UNION_SCHEMA)
        item_schema = relaxed["properties"]["items"]["items"]
        self.assertNotIn("required", item_schema)

    def test_oneof_branches_keep_only_their_discriminator(self):
        relaxed = actors._relax_required_for_wire(NESTED_UNION_SCHEMA)
        branch_a, branch_b = relaxed["properties"]["choice"]["oneOf"]
        self.assertEqual(branch_a["required"], ["type"])  # `value` dropped
        self.assertEqual(branch_b["required"], ["type"])  # single-enum discriminator kept

    def test_non_required_content_untouched(self):
        relaxed = actors._relax_required_for_wire(NESTED_UNION_SCHEMA)
        self.assertEqual(relaxed["properties"]["outer"]["properties"]["size"], {"type": "integer"})
        self.assertIs(relaxed["additionalProperties"], False)
        self.assertEqual(relaxed["properties"]["choice"]["oneOf"][1]["properties"]["type"],
                         {"enum": ["b"]})

    def test_does_not_mutate_input(self):
        original = copy.deepcopy(NESTED_UNION_SCHEMA)
        actors._relax_required_for_wire(NESTED_UNION_SCHEMA)
        self.assertEqual(NESTED_UNION_SCHEMA, original)

    def test_all_non_discriminator_required_schema_loses_required_entirely(self):
        relaxed = actors._relax_required_for_wire(SIMPLE_RELAX_SCHEMA)
        self.assertNotIn("required", relaxed)

    def test_required_with_no_sibling_properties_is_dropped(self):
        relaxed = actors._relax_required_for_wire({"type": "object", "required": ["x"]})
        self.assertNotIn("required", relaxed)

    def test_non_mapping_and_scalar_schemas_pass_through(self):
        self.assertIs(actors._relax_required_for_wire(True), True)
        self.assertEqual(actors._relax_required_for_wire({"type": "string"}), {"type": "string"})

    def test_property_declaration_order_is_preserved(self):
        # llama.cpp's grammar converter walks optional properties in declared
        # order (json-schema-to-grammar.cpp:688-725) -- relaxation must never
        # reorder `properties`.
        relaxed = actors._relax_required_for_wire(ORDERED_RELAX_SCHEMA)
        self.assertEqual(list(relaxed["properties"]), ["zeta", "alpha", "mu"])


class RelaxRequiredIntegration(unittest.TestCase):
    """`_schema_repair`/`_parse_reply` end to end: the schema handed to the
    server is relaxed (const/single-enum discriminators survive); the RESULT
    is still validated against the ORIGINAL schema, so an omitted required
    field fails honestly instead of being invented."""

    def _ws(self, tmp):
        ws = Path(tmp) / "workers" / "lane0"; ws.mkdir(parents=True); return ws

    class _Resp:
        def __init__(self, body): self._b = body
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _capturing_urlopen(self, captured, content):
        def fake_urlopen(request, timeout=None):
            captured.setdefault("bodies", []).append(json.loads(request.data.decode()))
            payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            return self._Resp(payload)
        return fake_urlopen

    def test_wire_schema_seen_by_server_has_required_relaxed_except_discriminator(self):
        import tempfile
        schema = {"type": "object",
                 "properties": {"kind": {"const": "x"}, "detail": {"type": "string"}},
                 "required": ["kind", "detail"], "additionalProperties": False}
        captured: dict = {}
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 mock.patch("urllib.request.urlopen",
                           side_effect=self._capturing_urlopen(
                               captured, json.dumps({"kind": "x", "detail": "d stated in report"}))):
                actors._schema_repair("report says detail: d stated in report", schema=schema,
                                     backend=actors.backend_for("prov/model", "high"), workspace=ws)
        wire_schema = captured["bodies"][0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(wire_schema.get("required"), ["kind"])

    def test_relax_required_false_keeps_required_on_the_wire(self):
        import tempfile
        captured: dict = {}
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 mock.patch("urllib.request.urlopen",
                           side_effect=self._capturing_urlopen(
                               captured, json.dumps({"name": "widget", "count": 1}))):
                actors._schema_repair("widget", schema=SIMPLE_RELAX_SCHEMA,
                                     backend=actors.backend_for("prov/model", "high"), workspace=ws,
                                     relax_required=False)
        wire_schema = captured["bodies"][0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(wire_schema.get("required"), ["name", "count"])

    def test_omitted_required_field_fails_instead_of_being_invented(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            captured: dict = {}
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 mock.patch("urllib.request.urlopen",
                           side_effect=self._capturing_urlopen(captured, json.dumps({"name": "widget"}))):
                with self.assertRaises(actors.ProviderTransient) as caught:
                    actors._parse_reply("a widget, no count given anywhere in the text",
                                        schema=SIMPLE_RELAX_SCHEMA,
                                        backend=actors.backend_for("prov/model", "high"), workspace=ws)
            # both were relaxed away on the wire (neither is a discriminator)
            self.assertNotIn("required", captured["bodies"][0]["response_format"]["json_schema"]["schema"])
            self.assertIn("invalid against the original", str(caught.exception))

    def test_supplied_required_field_still_repairs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            captured: dict = {}
            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 mock.patch("urllib.request.urlopen",
                           side_effect=self._capturing_urlopen(
                               captured, json.dumps({"name": "widget", "count": 3}))):
                body = actors._parse_reply("widget, and the count stated in the report is 3",
                                           schema=SIMPLE_RELAX_SCHEMA,
                                           backend=actors.backend_for("prov/model", "high"), workspace=ws)
            self.assertEqual(body, {"name": "widget", "count": 3})

    def test_decline_probe_schema_is_never_relaxed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._ws(tmp)
            captured: dict = {}

            def fake_urlopen(request, timeout=None):
                body = json.loads(request.data.decode())
                captured.setdefault("bodies", []).append(body)
                schema = body["response_format"]["json_schema"]["schema"]
                if "explicitly_declines" in schema.get("properties", {}):
                    content = json.dumps({"explicitly_declines": False, "reason": ""})
                else:
                    content = json.dumps({
                        "mechanism_id": "widen-alignment", "statement": "s", "falsifier": "f",
                        "target_surface": "a/b.cpp", "target_symbol": "align_fn"})
                payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                return self._Resp(payload)

            with mock.patch.object(actors, "_provider_base_url", return_value="http://127.0.0.1:1/v1"), \
                 mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                actors._parse_reply(
                    "I propose renaming the buffer align_fn to widen alignment in a/b.cpp.",
                    schema=actors.HYPOTHESIS_SCHEMA,
                    backend=actors.backend_for("prov/model", "high"), workspace=ws)
            decline_wire_schema = captured["bodies"][0]["response_format"]["json_schema"]["schema"]
            self.assertEqual(sorted(decline_wire_schema.get("required", [])),
                             ["explicitly_declines", "reason"])
