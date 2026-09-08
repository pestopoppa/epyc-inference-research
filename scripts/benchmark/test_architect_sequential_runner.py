from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import architect_sequential_runner as runner


def _question(qid: str, difficulty: int, expected: str = "A") -> dict:
    return {
        "id": qid,
        "prompt": f"prompt-{qid}",
        "expected": expected,
        "difficulty_key": difficulty,
        "scoring_method": "multiple_choice",
        "scoring_config": {},
    }


def _meta(text: str, *, error: str = "") -> dict:
    return {"text": text, "error": error, "finish_reason": "stop"}


class TestArchitectSequentialRunner(unittest.TestCase):
    def test_order_questions_is_descending_and_requires_a_priori_key(self):
        ordered = runner.order_questions(
            [_question("tie-b", 3), _question("easy", 1), _question("tie-a", 3)],
            difficulty_field="difficulty_key",
        )

        self.assertEqual([item["id"] for item in ordered], ["tie-a", "tie-b", "easy"])
        with self.assertRaisesRegex(ValueError, "a-priori difficulty"):
            runner.order_questions([{"id": "missing"}], difficulty_field="difficulty_key")

    def test_interleaves_complete_pairs_and_stops_on_separation(self):
        calls: list[tuple[str, str]] = []

        def fake_query(url: str, prompt: str, **_kwargs):
            calls.append((url, prompt))
            return _meta("A" if url.endswith("candidate") else "B")

        policy = runner.SequentialPolicy(confirm_e=1.05, budget=99)
        with self.subTest("interleaved pair"):
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as directory:
                temp = Path(directory)
                result = runner.run_interleaved(
                    suite="future_hard", questions=[_question("hard", 5), _question("easy", 1)],
                    arms=[runner.Arm("baseline", "http://baseline"), runner.Arm("candidate", "http://candidate")],
                    baseline_arm="baseline", candidate_arm="candidate", difficulty_field="difficulty_key",
                    saturation=runner.SaturationPolicy(2, 1.0), output=temp / "result.json",
                    capture_out=temp / "capture.jsonl", seed=42, max_tokens=64, temperature=0.7,
                    policy=policy, query=fake_query,
                )

                self.assertEqual(calls, [("http://baseline", "prompt-hard"), ("http://candidate", "prompt-hard")])
                self.assertEqual(result["stop_reason"], "separation:candidate")
                self.assertEqual(result["candidate_eprocess"]["k"], 1)
                capture = [json.loads(line) for line in (temp / "capture.jsonl").read_text().splitlines()]
                self.assertEqual([row["arm"] for row in capture], ["baseline", "candidate"])

    def test_saturation_stops_only_after_a_completed_difficulty_tier(self):
        calls: list[str] = []

        def fake_query(url: str, _prompt: str, **_kwargs):
            calls.append(url)
            return _meta("A")

        policy = runner.SequentialPolicy(first_lambda=0.0, lambda_cap=0.0, budget=99)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            temp = Path(directory)
            result = runner.run_interleaved(
                suite="future_hard", questions=[_question("hard", 5), _question("easy", 1)],
                arms=[runner.Arm("baseline", "http://baseline"), runner.Arm("candidate", "http://candidate")],
                baseline_arm="baseline", candidate_arm="candidate", difficulty_field="difficulty_key",
                saturation=runner.SaturationPolicy(1, 1.0), output=temp / "result.json",
                capture_out=temp / "capture.jsonl", seed=42, max_tokens=64, temperature=0.7,
                policy=policy, query=fake_query,
            )

        self.assertEqual(calls, ["http://baseline", "http://candidate"])
        self.assertEqual(result["stop_reason"], "saturation:difficulty_key=5")
        self.assertEqual(result["candidate_eprocess"]["k"], 1)

    # ── CJ-11: three-valued scoring ──────────────────────────────────────
    # This caller used to read
    #   "correct": bool(response) and score_response(response, expected, question)
    # in which any truthy third value coerces to a PASS. Each positive assertion
    # below is paired with a mutation that removes the signal under test.

    def _run_one(self, candidate_text: str, baseline_text: str, **kwargs):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            temp = Path(directory)
            result = runner.run_interleaved(
                suite="future_hard", questions=[_question("hard", 5)],
                arms=[runner.Arm("baseline", "http://baseline"),
                      runner.Arm("candidate", "http://candidate")],
                baseline_arm="baseline", candidate_arm="candidate",
                difficulty_field="difficulty_key",
                saturation=runner.SaturationPolicy(1, 1.0),
                output=temp / "result.json", capture_out=temp / "capture.jsonl",
                seed=42, max_tokens=64, temperature=0.7,
                query=lambda url, _p, **_k: _meta(
                    candidate_text if url.endswith("candidate") else baseline_text),
                **kwargs,
            )
            capture = [json.loads(line) for line
                       in (temp / "capture.jsonl").read_text().splitlines() if line.strip()]
        return result, capture

    def test_undecidable_arm_is_excluded_from_the_eprocess_and_never_a_pass(self):
        """(a) A NON-EMPTY but unparseable candidate answer -- so the historical
        `bool(response)` guard does not mask it -- must not become a pass and
        must not move the e-process."""
        unparseable = "I think C is likely, or maybe D, hard to say"
        result, capture = self._run_one(unparseable, "A")

        pair = result["pairs"][0]
        self.assertTrue(pair["paired_complete"], "transport was fine")
        self.assertFalse(pair["paired_decided"])
        self.assertEqual(result["undecided_pairs"], 1)
        self.assertEqual(result["provisional_transport_pairs"], 0)
        self.assertEqual(pair["sequential"]["state"], "not_updated_undecided")
        self.assertEqual(pair["sequential"]["causes"], {"candidate": "unparsed"})

        # the e-process saw NOTHING -- neither arm was credited or debited
        self.assertEqual(result["candidate_eprocess"]["k"], 0)
        self.assertEqual(result["baseline_eprocess"]["k"], 0)

        cand = pair["arms"]["candidate"]
        self.assertIs(cand["correct"], False)
        self.assertEqual(cand["verdict"], "out-of-coverage")
        self.assertEqual(cand["cause"], "unparsed")
        # the old idiom, applied to what the row now carries, still cannot pass
        self.assertFalse(bool(cand["response"]) and cand["correct"])
        # the decided arm is unaffected and carries no cause
        base = pair["arms"]["baseline"]
        self.assertIs(base["correct"], True)
        self.assertEqual(base["verdict"], "pass")
        self.assertIsNone(base["cause"])

        # the capture stream carries the same three-valued record
        self.assertEqual({row["arm"]: row["verdict"] for row in capture},
                         {"candidate": "out-of-coverage", "baseline": "pass"})

    def test_mutation_a_decided_pair_does_move_the_eprocess(self):
        """MUTATION of the exclusion: with a PARSEABLE candidate answer the very
        same harness updates the e-process. Without this, the assertions above
        would pass on a runner that never updates anything."""
        result, _capture = self._run_one("D", "A")
        pair = result["pairs"][0]
        self.assertTrue(pair["paired_decided"])
        self.assertEqual(result["undecided_pairs"], 0)
        self.assertEqual(result["candidate_eprocess"]["k"], 1)
        self.assertEqual(result["baseline_eprocess"]["k"], 1)
        self.assertNotIn("undecided", pair)

    def test_mutation_a_truthy_third_value_still_cannot_reach_correct(self):
        """MUTATION of (a) against the REAL code path: force the scorer to hand
        back a truthy non-True third value -- what an in-place widening of
        `score_response` would have produced -- and prove the caller's guard
        (`verdict is True`, not `and`) refuses it anyway.

        First assert the mutation is live: under the OLD expression this same
        value WOULD have been a pass."""
        widened = "out-of-coverage"
        self.assertTrue(bool("a non-empty response") and widened,
                        "the injected value must be truthy or the mutation is inert")

        real = runner.score_response_or_error
        runner.score_response_or_error = lambda *_a, **_k: (widened, None)
        try:
            result, capture = self._run_one("some answer", "some answer")
        finally:
            runner.score_response_or_error = real

        for row in result["pairs"][0]["arms"].values():
            self.assertIs(row["correct"], False)
            self.assertFalse(bool(row["response"]) and row["correct"])
        self.assertTrue(all(row["correct"] is False for row in capture))

    def test_undecided_arm_is_excluded_from_the_saturation_rate(self):
        """An undecided arm is in neither numerator nor denominator of the tier
        accuracy. Observable: at min_accuracy=1.0 a tier where the candidate is
        1 right + 1 undecided SATURATES (1/1), where folding the undecided in as
        a wrong answer would have given 1/2 and never saturated."""
        unparseable = "I think C is likely, or maybe D, hard to say"
        answers = {"prompt-h1": "A", "prompt-h2": unparseable, "prompt-easy": "A"}

        def fake_query(url: str, prompt: str, **_kwargs):
            return _meta("A" if url.endswith("baseline") else answers[prompt])

        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            temp = Path(directory)
            result = runner.run_interleaved(
                suite="future_hard",
                questions=[_question("h1", 5), _question("h2", 5), _question("easy", 1)],
                arms=[runner.Arm("baseline", "http://baseline"),
                      runner.Arm("candidate", "http://candidate")],
                baseline_arm="baseline", candidate_arm="candidate",
                difficulty_field="difficulty_key",
                saturation=runner.SaturationPolicy(1, 1.0),
                output=temp / "result.json", capture_out=temp / "capture.jsonl",
                seed=42, max_tokens=64, temperature=0.7, query=fake_query,
            )

        self.assertEqual(result["undecided_pairs"], 1)
        self.assertEqual(result["stop_reason"], "saturation:difficulty_key=5")
        # the easy tier was never reached, proving the stop actually fired
        self.assertEqual([pair["id"] for pair in result["pairs"]], ["h1", "h2"])
        # MUTATION: had the undecided arm been counted as a wrong answer the
        # candidate rate would be 1/2, below min_accuracy -- no stop.
        self.assertLess(1 / 2, 1.0)

    def test_transport_failure_is_recorded_without_eprocess_update(self):
        def fake_query(url: str, _prompt: str, **_kwargs):
            return _meta("A" if url.endswith("candidate") else "", error="timeout" if url.endswith("baseline") else "")

        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            temp = Path(directory)
            result = runner.run_interleaved(
                suite="future_hard", questions=[_question("hard", 5)],
                arms=[runner.Arm("baseline", "http://baseline"), runner.Arm("candidate", "http://candidate")],
                baseline_arm="baseline", candidate_arm="candidate", difficulty_field="difficulty_key",
                saturation=runner.SaturationPolicy(1, 1.0), output=temp / "result.json",
                capture_out=temp / "capture.jsonl", seed=42, max_tokens=64, temperature=0.7,
                query=fake_query,
            )

        self.assertEqual(result["provisional_transport_pairs"], 1)
        self.assertEqual(result["candidate_eprocess"]["k"], 0)
        self.assertEqual(result["pairs"][0]["sequential"]["state"], "not_updated_transport_failure")
