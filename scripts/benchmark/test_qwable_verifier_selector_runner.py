#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest import TestCase

import sys

sys.path.insert(0, str(Path(__file__).parent))

import qwable_verifier_selector_runner as runner


class TestQwableVerifierSelectorRunner(TestCase):
    def test_dry_run_writes_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = runner.main(
                    [
                        "--output-dir",
                        str(output_dir),
                        "--port-base",
                        "19140",
                        "--limit",
                        "2",
                        "--n-candidates",
                        "3",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertIn("mode: dry_run", stdout.getvalue())
            plan = json.loads((output_dir / "plan.json").read_text())
            self.assertEqual(plan["schema"], "qwable_verifier_selector_plan.v1")
            self.assertEqual(plan["mode"], "dry_run")
            self.assertEqual(plan["evidence_grade"], "observation")
            self.assertEqual(plan["n_candidates"], 3)
            self.assertEqual(plan["servers"]["beneficiary"]["port"], 19140)
            self.assertEqual(plan["servers"]["verifier"]["port"], 19141)
            self.assertIn("/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server", plan["server_bin"])
            self.assertIn("Qwen3.6-35B-A3B-MTP-Q8_0.gguf", plan["servers"]["beneficiary"]["model_path"])
            self.assertIn("Qwable-v1.IQ4_XS.gguf", plan["servers"]["verifier"]["model_path"])
            self.assertTrue((output_dir / "commands.sh").exists())

    def test_parse_index_prefers_final_marker(self) -> None:
        self.assertEqual(runner.parse_index("reasoning\nFINAL: 2", 5), (2, "marker"))
        self.assertEqual(runner.parse_index("Candidate 4 is bad. best candidate is 1", 5), (1, "marker"))
        self.assertEqual(runner.parse_index("I considered 9 then 3", 4), (3, "lastint"))
        self.assertEqual(runner.parse_index("no parse", 4), (0, "fallback"))

    def test_summarize_rows_gap_recovered(self) -> None:
        rows = [
            {"pass_at_1": False, "verifier_pass": True, "oracle_pass_at_n": True, "has_passing": True, "verifier_selected_passing": True},
            {"pass_at_1": True, "verifier_pass": True, "oracle_pass_at_n": True, "has_passing": True, "verifier_selected_passing": True},
            {"pass_at_1": False, "verifier_pass": False, "oracle_pass_at_n": True, "has_passing": True, "verifier_selected_passing": False},
        ]
        summary = runner.summarize_rows(rows)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["pass_at_1"], 1)
        self.assertEqual(summary["verifier_selected"], 2)
        self.assertEqual(summary["oracle_pass_at_n"], 3)
        self.assertEqual(summary["gap_recovered"], 0.5)
        self.assertEqual(summary["selection_accuracy"], 2 / 3)

    def test_load_questions_filters_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "pool.jsonl"
            pool.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "b", "suite": "cruxeval", "tier": 1}),
                        json.dumps({"id": "a", "suite": "cruxeval", "tier": 2}),
                        json.dumps({"id": "c", "suite": "other", "tier": 1}),
                    ]
                )
                + "\n"
            )
            rows = runner.load_questions(pool, "cruxeval", "1")
            self.assertEqual([row["id"] for row in rows], ["b"])
            rows_all = runner.load_questions(pool, "cruxeval", "")
            self.assertEqual([row["id"] for row in rows_all], ["a", "b"])

    def test_verifier_prompt_uses_bounded_candidate_excerpt(self) -> None:
        args = runner.parse_args(["--candidate-chars", "8"])
        question = {"prompt": "Pick the exact answer."}
        candidates = [{"index": 0, "verifier_excerpt": "abcdefghi"}, {"index": 1, "verifier_excerpt": "xyz"}]
        prompt = runner.verifier_prompt(question, candidates, args)
        self.assertIn("### Candidate 0\nabcdefghi", prompt)
        self.assertIn("### Candidate 1\nxyz", prompt)
        self.assertNotIn("candidate['text']", prompt)

    def test_extract_final_answer_uses_scoring_pattern(self) -> None:
        question = {
            "scoring_config": {
                "extract_pattern": "<answer>(.*?)</answer>",
            }
        }
        text = "reasoning\n<answer>[(4, 1), (2, 3)]</answer>\nmore text"
        self.assertEqual(runner.extract_final_answer(text, question), "[(4, 1), (2, 3)]")

    def test_candidate_verifier_excerpt_prefers_extracted_answer(self) -> None:
        args = runner.parse_args(["--candidate-chars", "8", "--candidate-answer-chars", "6"])
        question = {
            "scoring_config": {
                "extract_pattern": "<answer>(.*?)</answer>",
            }
        }
        answer, excerpt = runner.candidate_verifier_excerpt("abcdefghi\n<answer>correct-answer</answer>", question, args)
        self.assertEqual(answer, "correct-answer")
        self.assertIn("Extracted final answer:\ncorrec", excerpt)
        self.assertIn("Bounded candidate context:\nabcdefgh", excerpt)
