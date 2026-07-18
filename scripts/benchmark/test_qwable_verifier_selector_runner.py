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
            {
                "pass_at_1": False,
                "verifier_pass": True,
                "oracle_pass_at_n": True,
                "has_passing": True,
                "verifier_selected_passing": True,
                "verifier_parse": "marker",
                "verifier_finish_reason": "stop",
            },
            {
                "pass_at_1": True,
                "verifier_pass": True,
                "oracle_pass_at_n": True,
                "has_passing": True,
                "verifier_selected_passing": True,
                "verifier_parse": "lastint",
                "verifier_finish_reason": "length",
            },
            {
                "pass_at_1": False,
                "verifier_pass": False,
                "oracle_pass_at_n": True,
                "has_passing": True,
                "verifier_selected_passing": False,
                "verifier_parse": "marker",
                "verifier_finish_reason": "stop",
            },
        ]
        summary = runner.summarize_rows(rows)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["pass_at_1"], 1)
        self.assertEqual(summary["verifier_selected"], 2)
        self.assertEqual(summary["oracle_pass_at_n"], 3)
        self.assertEqual(summary["gap_recovered"], 0.5)
        self.assertEqual(summary["selection_accuracy"], 2 / 3)
        self.assertEqual(summary["verifier_parse_modes"], {"marker": 2, "lastint": 1})
        self.assertEqual(summary["verifier_finish_reasons"], {"stop": 2, "length": 1})

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
        self.assertIn("First solve the problem independently", prompt)

    def test_verifier_prompt_can_skip_solve_first(self) -> None:
        args = runner.parse_args(["--no-verifier-solve-first"])
        question = {"prompt": "Pick the exact answer."}
        candidates = [{"index": 0, "verifier_excerpt": "Extracted final answer:\nA"}]
        prompt = runner.verifier_prompt(question, candidates, args)
        self.assertIn("Do not solve the problem from scratch", prompt)
        self.assertNotIn("First solve the problem independently", prompt)

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

    def test_verifier_payload_defaults_to_thinking_enabled(self) -> None:
        args = runner.parse_args([])
        payload = runner.verifier_payload(
            {"prompt": "Pick."},
            [{"index": 0, "verifier_excerpt": "Extracted final answer:\nok"}],
            args,
        )
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("Do not explain", payload["messages"][0]["content"])

    def test_verifier_payload_can_disable_thinking(self) -> None:
        args = runner.parse_args(["--no-verifier-thinking"])
        payload = runner.verifier_payload(
            {"prompt": "Pick."},
            [{"index": 0, "verifier_excerpt": "Extracted final answer:\nok"}],
            args,
        )
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertIn("Do not explain", payload["messages"][0]["content"])

    def test_plan_records_verifier_controls(self) -> None:
        args = runner.parse_args(["--no-verifier-thinking", "--no-verifier-solve-first"])
        plan = runner.build_plan(args)
        self.assertFalse(plan["request"]["verifier_thinking"])
        self.assertFalse(plan["request"]["verifier_solve_first"])

    def test_replay_plan_uses_existing_candidates_without_beneficiary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            artifact.mkdir()
            row = {
                "status": "ok",
                "qid": "cruxeval_output_0057",
                "suite": "cruxeval",
                "tier": 1,
                "has_passing": True,
                "n_passing": 1,
                "verifier_selected_index": 0,
                "verifier_selected_passing": False,
                "candidates": [
                    {"index": 0, "correct": False, "verifier_excerpt": "Extracted final answer:\nno"},
                    {"index": 1, "correct": True, "verifier_excerpt": "Extracted final answer:\n1"},
                ],
            }
            (artifact / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            args = runner.parse_args([
                "--replay-artifact",
                str(artifact),
                "--replay-known-misses",
                "--replay-permute-solvable",
            ])

            plan = runner.build_replay_plan(args)

            self.assertEqual(plan["schema"], "qwable_verifier_selector_replay_plan.v1")
            self.assertTrue(plan["replay"]["no_beneficiary_regeneration"])
            self.assertEqual(set(plan["servers"]), {"verifier"})
            self.assertEqual(plan["replay"]["source_rows"], 1)
            self.assertEqual(plan["replay"]["planned_cases"], 2)
            self.assertEqual(plan["replay_cases"][0]["source_order"], [0, 1])
            self.assertIn([1, 0], [case["source_order"] for case in plan["replay_cases"]])

    def test_replay_only_misses_filters_non_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            artifact.mkdir()
            rows = [
                {
                    "status": "ok",
                    "qid": "miss",
                    "has_passing": True,
                    "verifier_selected_passing": False,
                    "candidates": [{"index": 0, "correct": False}, {"index": 1, "correct": True}],
                },
                {
                    "status": "ok",
                    "qid": "hit",
                    "has_passing": True,
                    "verifier_selected_passing": True,
                    "candidates": [{"index": 0, "correct": True}],
                },
                {
                    "status": "ok",
                    "qid": "unsolved",
                    "has_passing": False,
                    "verifier_selected_passing": None,
                    "candidates": [{"index": 0, "correct": False}],
                },
            ]
            (artifact / "results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = runner.parse_args(["--replay-artifact", str(artifact), "--replay-only-misses"])

            cases = runner.build_replay_cases(args)

            self.assertEqual([case["qid"] for case in cases], ["miss"])

    def test_replay_case_payload_uses_relabelled_candidates(self) -> None:
        args = runner.parse_args(["--replay-artifact", "/tmp/nonexistent"])
        question = {"prompt": "Choose the exact answer."}
        case = {
            "qid": "q1",
            "candidates": [
                {"index": 0, "source_index": 3, "verifier_excerpt": "Extracted final answer:\n1"},
                {"index": 1, "source_index": 0, "verifier_excerpt": "Extracted final answer:\nno"},
            ],
        }

        payload = runner.replay_case_payload(case, args, {"q1": question})

        user_prompt = payload["messages"][1]["content"]
        self.assertIn("### Candidate 0\nExtracted final answer:\n1", user_prompt)
        self.assertIn("### Candidate 1\nExtracted final answer:\nno", user_prompt)
        self.assertIn("First solve the problem independently", user_prompt)
