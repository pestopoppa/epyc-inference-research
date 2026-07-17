#!/usr/bin/env python3
"""Self-contained tests for the LongCoT-Mini post-run aggregator.

K-LCM-1 (intake-386 / RE-4).  The research repo ships no pytest, so this is a
stdlib ``unittest`` module with a ``unittest.main()`` runner in ``__main__``::

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        scripts/benchmark/tests/test_score_longcot_run.py

Every test is INFERENCE-FREE and DATASET-FREE: the aggregator is driven with a
SYNTHETIC run-output fixture (a few chemistry/chess/cs/math rows + one logic
row + one canary-leak row) and a SYNTHETIC prompt index (never the Arrow
dataset, never a server), mirroring the score_tulving_run test convention.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_BENCHMARK_DIR = _TESTS_DIR.parent
_SCRIPTS_DIR = _BENCHMARK_DIR.parent
_RESEARCH_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_RESEARCH_ROOT), str(_SCRIPTS_DIR), str(_BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import longcot_mini_adapter as lcm  # noqa: E402
import score_longcot_run as slr  # noqa: E402

_CANARY_LEAK = "LEAK-CANARY-1234-abcd"


def _prompt(pid, domain, gold_value, canary, *, is_scorable=True, template="t"):
    """Build a prompt_dict of the exact shape LongCoTMiniAdapter emits."""
    return {
        "id": pid,
        "suite": "longcot_mini",
        "scoring_method": lcm.SCORING_METHOD,
        "scoring_config": {"is_scorable": is_scorable, "domain": domain,
                           "template": template, "canary": canary},
        "metadata": {"question_id": pid, "domain": domain, "template": template,
                     "canary": canary, "is_scorable": is_scorable,
                     "gold_raw": json.dumps(gold_value), "gold_value": gold_value},
    }


def _build_fixture():
    """A 6-row run: 5 scorable (4 correct incl. 1 canary-leaked) + 1 logic.

    Expected aggregates (all exact):
      scorable rows        = 5
      scorable correct     = 4   -> overall accuracy 4/5 = 0.80
      leak-free scorable   = 4 (3 correct) -> clean accuracy 3/4 = 0.75
      canary leaks         = 1 (on a scorable chess row)
      unscorable logic     = 1
      per-domain accuracy  : chemistry 1/1, chess 2/2, cs 0/1, math 1/1
    """
    prompts = {
        "longcot_mini_chem_ok":
            _prompt("longcot_mini_chem_ok", "chemistry", "CCO", "canary-chem"),
        "longcot_mini_chess_ok":
            _prompt("longcot_mini_chess_ok", "chess", "391365", "canary-chess"),
        "longcot_mini_chess_leak":
            _prompt("longcot_mini_chess_leak", "chess", "42", _CANARY_LEAK),
        "longcot_mini_cs_bad":
            _prompt("longcot_mini_cs_bad", "cs", [1, 2, 3], "canary-cs"),
        "longcot_mini_math_ok":
            _prompt("longcot_mini_math_ok", "math", ["16", "13"], "canary-math"),
        "longcot_mini_logic":
            _prompt("longcot_mini_logic", "logic", None, "canary-logic",
                    is_scorable=False),
    }
    responses = {
        "longcot_mini_chem_ok": "reasoning...\nsolution = CCO",
        "longcot_mini_chess_ok": "think...\nsolution = 391365",
        # canary leaks in the reasoning; answer still structurally correct
        "longcot_mini_chess_leak":
            f"I recall the tag {_CANARY_LEAK} from training.\nsolution = 42",
        "longcot_mini_cs_bad": "solution = [3, 2, 1]",   # order wrong -> FAIL
        "longcot_mini_math_ok": "solution = [16, 13]",
        "longcot_mini_logic": "solution = [[1,2],[3,4]]",  # unscorable
    }
    payload = {
        "run_id": "test_run",
        "model_role": "worker",
        "config_name": "baseline",
        "results": {"longcot_mini": {
            qid: {"question_id": qid, "prompt": "p", "response": responses[qid],
                  "tokens_per_second": 10.0}
            for qid in prompts
        }},
    }
    return prompts, payload


class TestLoadRunRows(unittest.TestCase):
    def setUp(self):
        self.prompts, self.payload = _build_fixture()

    def _write(self, obj, suffix=".json", jsonl=False):
        tmp = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        if jsonl:
            for row in obj:
                tmp.write(json.dumps(row) + "\n")
        else:
            json.dump(obj, tmp)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_loads_run_benchmark_payload_shape(self):
        path = self._write(self.payload)
        rows = slr.load_run_rows(path)
        self.assertEqual(len(rows), 6)
        ids = {qid for qid, _ in rows}
        self.assertEqual(ids, set(self.payload["results"]["longcot_mini"]))
        # each row carries a model response
        self.assertTrue(all("response" in row for _, row in rows))

    def test_loads_jsonl(self):
        jsonl_rows = [
            {"question_id": "longcot_mini_chem_ok", "response": "solution = CCO"},
            {"question_id": "longcot_mini_math_ok", "response": "solution = [16, 13]"},
        ]
        path = self._write(jsonl_rows, suffix=".jsonl", jsonl=True)
        rows = slr.load_run_rows(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual([qid for qid, _ in rows],
                         ["longcot_mini_chem_ok", "longcot_mini_math_ok"])


class TestScoreRunPayload(unittest.TestCase):
    def setUp(self):
        self.prompts, self.payload = _build_fixture()
        self.rows = list(self.payload["results"]["longcot_mini"].items())
        self.scored = slr.score_run_payload(
            self.rows, self.prompts,
            run_meta={"run_id": "test_run", "model_role": "worker",
                      "config_name": "baseline"},
        )
        self.summary = self.scored["summary"]

    def test_row_counts(self):
        self.assertEqual(self.summary["rows_in_run"], 6)
        self.assertEqual(self.summary["scorable_rows"], 5)
        self.assertEqual(self.summary["scorable_correct"], 4)
        self.assertEqual(self.summary["missing_from_prompt_index"], 0)

    def test_overall_accuracy_exact(self):
        # 4 correct of 5 scorable, canary-agnostic
        self.assertEqual(self.summary["overall_accuracy"], 0.8)

    def test_canary_reported_separately_and_clean_accuracy(self):
        # the leak is a SEPARATE signal, not folded into pass/fail
        self.assertEqual(self.summary["canary_leak_count"], 1)
        self.assertEqual(self.summary["canary_leaks_on_scorable_rows"], 1)
        # a leak invalidates that row's reading -> clean accuracy drops the row:
        # 3 correct of 4 leak-free scorable rows
        self.assertEqual(self.summary["clean_scorable_rows"], 4)
        self.assertEqual(
            self.summary["overall_accuracy_excluding_canary_leaks"], 0.75)

    def test_unscorable_logic_excluded_from_accuracy(self):
        self.assertEqual(self.summary["unscorable_logic_rows"], 1)
        # logic never enters the scorable denominator or any domain bucket
        self.assertNotIn("logic", self.scored["per_domain"])

    def test_per_domain_accuracy_exact(self):
        pd = self.scored["per_domain"]
        self.assertEqual(set(pd), set(slr.SCORABLE_DOMAINS))
        self.assertEqual((pd["chemistry"]["correct"], pd["chemistry"]["total"],
                          pd["chemistry"]["accuracy"]), (1, 1, 1.0))
        self.assertEqual((pd["chess"]["correct"], pd["chess"]["total"],
                          pd["chess"]["accuracy"]), (2, 2, 1.0))
        self.assertEqual(pd["chess"]["canary_leaks"], 1)
        self.assertEqual((pd["cs"]["correct"], pd["cs"]["total"],
                          pd["cs"]["accuracy"]), (0, 1, 0.0))
        self.assertEqual((pd["math"]["correct"], pd["math"]["total"],
                          pd["math"]["accuracy"]), (1, 1, 1.0))

    def test_grade_is_observation(self):
        self.assertEqual(self.summary["grade"], "observation")
        self.assertIn("OBSERVATION", self.summary["grade_note"])

    def test_deterministic_repeat(self):
        again = slr.score_run_payload(self.rows, self.prompts)
        self.assertEqual(again["summary"]["overall_accuracy"],
                         self.summary["overall_accuracy"])
        self.assertEqual(again["per_domain"], self.scored["per_domain"])


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.prompts, _ = _build_fixture()

    def test_missing_prompt_id_counted(self):
        rows = [("longcot_mini_ghost", {"response": "solution = 1"})]
        scored = slr.score_run_payload(rows, self.prompts)
        self.assertEqual(scored["summary"]["missing_from_prompt_index"], 1)
        self.assertEqual(scored["missing_prompt_ids"], ["longcot_mini_ghost"])
        self.assertEqual(scored["summary"]["scorable_rows"], 0)

    def test_inline_metadata_fallback_resolves_without_index(self):
        # a JSONL-style row that carries its own metadata scores with no index
        inline = _prompt("x", "math", ["16", "13"], "canary-x")
        row = dict(inline, response="solution = [16, 13]")
        scored = slr.score_run_payload([("x", row)], prompt_index={})
        self.assertEqual(scored["summary"]["scorable_rows"], 1)
        self.assertEqual(scored["summary"]["scorable_correct"], 1)
        self.assertEqual(scored["summary"]["missing_from_prompt_index"], 0)


class TestMarkdownAndCLI(unittest.TestCase):
    def setUp(self):
        self.prompts, self.payload = _build_fixture()

    def test_render_markdown_includes_key_metrics(self):
        rows = list(self.payload["results"]["longcot_mini"].items())
        scored = slr.score_run_payload(rows, self.prompts)
        md = slr.render_markdown(scored, Path("run.json"))
        self.assertIn("# LongCoT-Mini Run Score", md)
        self.assertIn("Overall accuracy: 0.8000", md)
        self.assertIn("excluding canary-leaked rows): 0.7500", md)
        self.assertIn("Canary leaks: 1", md)
        self.assertIn("Unscorable logic rows", md)
        # per-domain table rows
        self.assertIn("| chess | 2 | 2 | 1.0000 | 1 |", md)
        self.assertIn("OBSERVATION-grade", md)

    def test_main_writes_json_and_md(self):
        # inject the synthetic prompt index so main() needs no Arrow dataset
        orig = slr.build_prompt_index
        slr.build_prompt_index = lambda *a, **k: self.prompts
        self.addCleanup(lambda: setattr(slr, "build_prompt_index", orig))

        with tempfile.TemporaryDirectory() as d:
            run_path = Path(d) / "run.json"
            run_path.write_text(json.dumps(self.payload))
            out_json = Path(d) / "reports" / "score.json"
            rc = slr.main([str(run_path), "--output", str(out_json)])
            self.assertEqual(rc, 0)
            self.assertTrue(out_json.exists())
            md_path = out_json.with_suffix(".md")
            self.assertTrue(md_path.exists())
            report = json.loads(out_json.read_text())
            self.assertEqual(report["summary"]["overall_accuracy"], 0.8)
            self.assertEqual(report["summary"]["canary_leak_count"], 1)
            self.assertEqual(report["summary"]["unscorable_logic_rows"], 1)
            self.assertEqual(report["summary"]["run_id"], "test_run")


if __name__ == "__main__":
    unittest.main(verbosity=2)
