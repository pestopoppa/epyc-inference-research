"""No-network tests for the YARN-JUDGE-1 cloud judge, the cross-family labeller and the deterministic label rules.
The orchestrator transport is real (loaded by path); the CLI process is replaced by a fake ``run_fn``.

    python3 -m unittest scripts/benchmark/yarn_gate/tests/test_cloud_judge.py   # from the research repo root
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.yarn_gate import cloud_judge as cjm, judge_validation_kit as kit  # noqa: E402
from scripts.benchmark.yarn_gate.local_judge import JudgeError  # noqa: E402

REGISTRY = textwrap.dedent("""\
    schema: epyc.orchestrator.cloud_judges.v1
    judges:
      codex-luna-low: {transport: codex_exec, model: gpt-5.6-luna, effort: low, timeout_s: 300}
      sonnet-5-5-low: {transport: claude_cli, model: claude-sonnet-5-5, effort: low, timeout_s: 300}
    """)


def claude_envelope(obj: dict, model: str = "claude-sonnet-5-5") -> str:
    return json.dumps({"type": "result", "is_error": False, "structured_output": obj, "total_cost_usd": 0.01,
                       "usage": {"input_tokens": 100, "output_tokens": 7}, "modelUsage": {model: {}}})


@unittest.skipUnless(cjm.CLOUD_JUDGES_PY.is_file(), "orchestrator cloud_judges.py not present")
class TestCloud(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        reg = Path(self.tmp.name) / "cloud_judges.yaml"
        reg.write_text(REGISTRY)
        self._old = os.environ.get("ORCHESTRATOR_CLOUD_JUDGES_FILE")
        os.environ["ORCHESTRATOR_CLOUD_JUDGES_FILE"] = str(reg)
        self.cj = cjm.load_cloud_judges()

    def tearDown(self):
        if self._old is None:
            os.environ.pop("ORCHESTRATOR_CLOUD_JUDGES_FILE", None)
        else:
            os.environ["ORCHESTRATOR_CLOUD_JUDGES_FILE"] = self._old
        self.tmp.cleanup()

    def test_claude_judge_verdict_usage_and_pin(self):
        seen = []

        def run(argv, stdin, timeout, cwd):
            seen.append((argv, stdin))
            return 0, claude_envelope({"verdict": "correct" if "Paris" in stdin.split("Candidate answer:")[1]
                                       else "incorrect"}), ""

        caller = cjm.CloudCaller(self.cj, cjm.resolve_spec(self.cj, "sonnet-5-5-low", transport="claude_cli"),
                                 run_fn=run)
        j = cjm.CloudJudge(caller)
        self.assertEqual(j.judge("Capital of France?", "Paris", "Paris.")["verdict"], 1)
        self.assertEqual(j.judge("Capital of France?", "Paris", "Lyon")["verdict"], 0)
        self.assertIn("--model", seen[0][0])
        self.assertEqual(seen[0][0][seen[0][0].index("--model") + 1], "claude-sonnet-5-5")
        ident = caller.identity()
        self.assertEqual(ident["usage"]["calls"], 2)
        self.assertEqual(ident["usage"]["input_tokens"], 200)
        self.assertEqual(ident["envelope_models"], ["claude-sonnet-5-5"])
        pin = j.pin.claim_fields()
        self.assertEqual(pin["judge_kind"], "cloud-claude_cli")
        self.assertIsNone(pin["temperature"])
        self.assertFalse(pin["comparable_to_published"])
        # root scripts/vidya/adapters/yarn_gate_agg.py JUDGE_PIN_KEYS must all be present on a cloud pin
        for k in ("model", "prompt_version", "prompt_sha256", "temperature", "seed", "max_tokens"):
            self.assertIn(k, pin)

    def test_failure_is_unavailable_never_zero(self):
        caller = cjm.CloudCaller(self.cj, cjm.resolve_spec(self.cj, "sonnet-5-5-low"),
                                 run_fn=lambda *a: (1, "", "not logged in"))
        rows = kit.judge_items([{"item_id": "j0", "question": "q", "reference": "r", "candidate": "c"}],
                               cjm.CloudJudge(caller))
        self.assertIsNone(rows[0]["verdict"])
        self.assertTrue(rows[0]["raw"].startswith("unavailable"))
        with self.assertRaises(JudgeError):
            caller.preflight()

    def test_wrong_transport_refused(self):
        with self.assertRaises(ValueError):
            cjm.resolve_spec(self.cj, "codex-luna-low", transport="claude_cli")
        with self.assertRaises(ValueError):
            cjm.resolve_spec(self.cj, "nope")

    def test_labeller_rules_first_then_model_with_reference_in_prompt(self):
        prompts = []

        def run(argv, stdin, timeout, cwd):
            prompts.append(stdin)
            out = Path(argv[argv.index("-o") + 1])
            out.write_text(json.dumps({"label": 1, "rationale": "same entity"}))
            return 0, "", ""

        caller = cjm.CloudCaller(self.cj, cjm.resolve_spec(self.cj, "codex-luna-low", transport="codex_exec"),
                                 run_fn=run)
        items = [{"item_id": "a", "task": "simpleqa", "question": "q1", "reference": "454", "candidate": "2000"},
                 {"item_id": "b", "task": "hotpotqa", "question": "Context: ...", "reference": "John Francis Kelly",
                  "candidate": "John F. Kelly"}]
        ctx = {"b": {"task": "hotpotqa", "resolved": True, "full_question": "Who?", "dataset_answer": "John Francis Kelly",
                     "question_type": "bridge", "supporting_facts": [{"title": "Kelly", "sentence": "John Francis Kelly ..."}]}}
        with tempfile.TemporaryDirectory() as d:
            part = Path(d) / "l.partial.jsonl"
            rows = {r["item_id"]: r for r in kit.label_items(items, ctx, caller, workers=1, partial=part)}
            self.assertEqual(rows["a"]["label_source"], "rule")
            self.assertEqual(rows["a"]["label"], 0)
            self.assertEqual(rows["b"]["label_source"], "codex-luna-low")
            self.assertEqual(rows["b"]["label"], 1)
            self.assertEqual(len(prompts), 1)
            self.assertIn("John Francis Kelly", prompts[0])
            self.assertIn("Who?", prompts[0])
            # resume: a decided row is never re-asked
            kit.label_items(items, ctx, caller, workers=1, partial=part)
            self.assertEqual(len(prompts), 1)


class TestRules(unittest.TestCase):
    def test_rules(self):
        r = kit.rule_label
        self.assertEqual(r("x", "[Max turns (10) reached]")[0], 0)
        self.assertEqual(r("x", '{"success": false, "error": "boom"}')[0], 0)
        self.assertEqual(r("A Little Princess", "#### A Little Princess")[0], 1)
        self.assertEqual(r("Atlético Madrid", "atletico madrid.")[0], 1)
        self.assertEqual(r("no", "Yes")[0], 0)
        self.assertEqual(r("350 meters", "3100")[0], 0)
        self.assertEqual(r("4080", "4080 ft")[0], 1)
        self.assertEqual(r("13", "#### 7 years")[0], 0)
        self.assertIsNone(r("John Francis Kelly", "John F. Kelly"))  # alias: not a rule's call
        self.assertIsNone(r("4080 ft", "4080 m"))  # equal number, conflicting unit
        self.assertIsNone(r("Gurgaon", "a long " + "word " * 20))  # long text is a reading task
        self.assertEqual(kit.final_answer("#### Neil ####\n\nprose #### Neil ####"), "Neil")

    def test_kappa_ci(self):
        pairs = [(1, 1)] * 10 + [(0, 0)] * 35 + [(1, 0)] * 2 + [(0, 1)] * 3
        ci = kit.kappa_ci(pairs)
        k = kit.cohen_kappa(pairs)
        self.assertLess(ci["analytic"][0], k)
        self.assertGreater(ci["analytic"][1], k)
        self.assertLessEqual(ci["bootstrap"][0], k)
        self.assertEqual(ci, kit.kappa_ci(pairs))  # seeded: deterministic
        self.assertIsNone(kit.kappa_ci([(1, 1)] * 5)["analytic"])
        a = kit.agreement([{"item_id": "x", "human_label": "1"}], [{"item_id": "x", "verdict": 1}])
        self.assertIn("cohen_kappa_ci95", a)


if __name__ == "__main__":
    unittest.main()
