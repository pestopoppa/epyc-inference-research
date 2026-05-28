"""Unit tests for v4_quality_gate_compare.py.

Exercises compute_mad, token1_match, has_runtime_failure, compare, and verdict
with synthetic JSONs that have known expected outcomes. No actual GGUF or
server required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Import the module under test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_quality_gate_compare as c


# ---------------------------------------------------------------------------
# Synthetic-data fixtures
# ---------------------------------------------------------------------------


def make_prompt(pid: str, category: str, tokens_text: list[str],
                logprobs: list[float], error: str | None = None) -> dict:
    """Build a synthetic per-prompt result entry matching the runner's shape."""
    base = {
        "id": pid,
        "category": category,
        "prompt": "synthetic test prompt",
        "tokens_text": tokens_text,
        "logprobs": logprobs,
        "token_count": len(tokens_text),
    }
    if error is not None:
        base["error"] = error
        base["token_count"] = 0
    return base


def make_result_set(prompts: list[dict], model_path: str = "/test/model.gguf",
                    binary: str = "/test/bin/llama-server",
                    n_tokens_requested: int = 1) -> dict:
    """Build a synthetic top-level runner output.

    Defaults to n_tokens_requested=1 so existing 3-token synthetic prompts
    pass the strict per-prompt token-count check; the dedicated strict-token
    test class overrides this to exercise the gate.
    """
    return {
        "model_path": model_path,
        "binary": binary,
        "n_tokens_requested": n_tokens_requested,
        "n_prompts": len(prompts),
        "seed": 1,
        "temperature": 0,
        "top_k": 1,
        "prompts": prompts,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestComputeMAD(unittest.TestCase):

    def test_identical_logprobs_zero_mad(self):
        a = [-0.5, -1.2, -0.8]
        b = [-0.5, -1.2, -0.8]
        self.assertEqual(c.compute_mad(a, b), 0.0)

    def test_constant_offset(self):
        a = [-0.5, -1.0, -2.0]
        b = [-0.6, -1.1, -2.1]
        self.assertAlmostEqual(c.compute_mad(a, b), 0.1, places=6)

    def test_empty_returns_none(self):
        self.assertIsNone(c.compute_mad([], [-0.5]))
        self.assertIsNone(c.compute_mad([-0.5], []))
        self.assertIsNone(c.compute_mad([], []))

    def test_mismatched_lengths_uses_min(self):
        # min length = 2; MAD over first 2 only
        a = [-1.0, -2.0, -3.0]
        b = [-1.5, -2.5]
        self.assertAlmostEqual(c.compute_mad(a, b), 0.5, places=6)

    def test_none_entries_skipped(self):
        a = [-0.5, None, -1.0]
        b = [-0.6, -1.0, -1.1]
        # only positions 0 and 2 contribute: |(-0.5)-(-0.6)| + |(-1.0)-(-1.1)| = 0.1 + 0.1, avg 0.1
        self.assertAlmostEqual(c.compute_mad(a, b), 0.1, places=6)

    def test_nan_returns_none(self):
        a = [-0.5, float("nan")]
        b = [-0.6, -1.0]
        self.assertIsNone(c.compute_mad(a, b))

    def test_inf_returns_none(self):
        a = [-0.5, float("-inf")]
        b = [-0.6, -1.0]
        self.assertIsNone(c.compute_mad(a, b))

    def test_non_numeric_skipped(self):
        a = [-0.5, "garbage", -1.0]
        b = [-0.6, -1.0, -1.1]
        # only positions 0 and 2 contribute
        self.assertAlmostEqual(c.compute_mad(a, b), 0.1, places=6)


class TestToken1Match(unittest.TestCase):

    def test_same_first_token_true(self):
        self.assertTrue(c.token1_match(["hello", "world"], ["hello", "there"]))

    def test_different_first_token_false(self):
        self.assertFalse(c.token1_match(["hello", "world"], ["hi", "world"]))

    def test_empty_either_side_false(self):
        self.assertFalse(c.token1_match([], ["hello"]))
        self.assertFalse(c.token1_match(["hello"], []))
        self.assertFalse(c.token1_match([], []))


class TestRuntimeFailure(unittest.TestCase):

    def test_error_key_is_failure(self):
        self.assertTrue(c.has_runtime_failure({"error": "assert failed", "token_count": 0}))

    def test_empty_tokens_is_failure(self):
        self.assertTrue(c.has_runtime_failure({"token_count": 0}))

    def test_normal_is_not_failure(self):
        self.assertFalse(c.has_runtime_failure({"token_count": 64, "logprobs": [-0.5]}))


class TestVerdict(unittest.TestCase):

    def test_clean_pass(self):
        summary = {"n_prompts": 20, "n_pass_mad": 19, "n_token1_match": 17,
                   "n_runtime_fail": 0, "max_mad_threshold": 0.05}
        ok, text = c.verdict(summary, 18, 15)
        self.assertTrue(ok)
        self.assertIn("PASS", text)

    def test_runtime_failure_blocks_pass(self):
        # Even if MAD + token-1 thresholds met, a runtime fail is auto-FAIL
        summary = {"n_prompts": 20, "n_pass_mad": 20, "n_token1_match": 20,
                   "n_runtime_fail": 1, "max_mad_threshold": 0.05}
        ok, text = c.verdict(summary, 18, 15)
        self.assertFalse(ok)
        self.assertIn("runtime failure", text)

    def test_below_mad_threshold_fails(self):
        summary = {"n_prompts": 20, "n_pass_mad": 17, "n_token1_match": 17,
                   "n_runtime_fail": 0, "max_mad_threshold": 0.05}
        ok, text = c.verdict(summary, 18, 15)
        self.assertFalse(ok)
        self.assertIn("MAD", text)
        self.assertIn("17/20", text)

    def test_below_token1_threshold_fails(self):
        summary = {"n_prompts": 20, "n_pass_mad": 19, "n_token1_match": 14,
                   "n_runtime_fail": 0, "max_mad_threshold": 0.05}
        ok, text = c.verdict(summary, 18, 15)
        self.assertFalse(ok)
        self.assertIn("token-1", text)
        self.assertIn("14/20", text)

    def test_at_threshold_passes(self):
        # Exactly at threshold should pass (≥ is inclusive)
        summary = {"n_prompts": 20, "n_pass_mad": 18, "n_token1_match": 15,
                   "n_runtime_fail": 0, "max_mad_threshold": 0.05}
        ok, _ = c.verdict(summary, 18, 15)
        self.assertTrue(ok)


class TestCompareEndToEnd(unittest.TestCase):

    def test_all_identical_passes(self):
        # 20 prompts, all logprobs exactly identical → trivially passes
        prompts = []
        for i in range(20):
            tokens = [f"tok{i}_{j}" for j in range(64)]
            logprobs = [-0.5 - 0.01 * j for j in range(64)]
            prompts.append(make_prompt(f"p{i:02d}", "short_factual", tokens, logprobs))
        epyc = make_result_set(prompts)
        ref = make_result_set([p.copy() for p in prompts])
        rows, summary = c.compare(epyc, ref, max_mad=0.05)
        self.assertEqual(summary["n_pass_mad"], 20)
        self.assertEqual(summary["n_token1_match"], 20)
        self.assertEqual(summary["n_runtime_fail"], 0)
        ok, _ = c.verdict(summary, 18, 15)
        self.assertTrue(ok)

    def test_small_constant_offset_within_tolerance(self):
        # All prompts have a 0.02-nat offset; tolerance is 0.05 → all pass
        epyc_prompts = []
        ref_prompts = []
        for i in range(20):
            tokens = [f"tok{i}_{j}" for j in range(64)]
            base = [-1.0 - 0.01 * j for j in range(64)]
            epyc_prompts.append(make_prompt(f"p{i:02d}", "x", tokens, base))
            shifted = [v - 0.02 for v in base]
            ref_prompts.append(make_prompt(f"p{i:02d}", "x", tokens, shifted))
        rows, summary = c.compare(make_result_set(epyc_prompts),
                                   make_result_set(ref_prompts), max_mad=0.05)
        self.assertEqual(summary["n_pass_mad"], 20)
        self.assertEqual(summary["n_token1_match"], 20)
        ok, _ = c.verdict(summary, 18, 15)
        self.assertTrue(ok)

    def test_large_offset_fails_mad(self):
        # 0.1-nat offset; tolerance 0.05 → all fail MAD
        epyc_prompts = []
        ref_prompts = []
        for i in range(20):
            tokens = [f"tok{i}_{j}" for j in range(64)]
            base = [-1.0 - 0.01 * j for j in range(64)]
            epyc_prompts.append(make_prompt(f"p{i:02d}", "x", tokens, base))
            shifted = [v - 0.10 for v in base]
            ref_prompts.append(make_prompt(f"p{i:02d}", "x", tokens, shifted))
        rows, summary = c.compare(make_result_set(epyc_prompts),
                                   make_result_set(ref_prompts), max_mad=0.05)
        self.assertEqual(summary["n_pass_mad"], 0)
        ok, text = c.verdict(summary, 18, 15)
        self.assertFalse(ok)
        self.assertIn("MAD", text)

    def test_token1_divergence_fails_even_with_low_mad(self):
        # MAD low but different first token in 6 prompts → fails token-1 gate (<15)
        epyc_prompts = []
        ref_prompts = []
        for i in range(20):
            base = [-1.0] * 64
            epyc_tokens = ["A"] * 64
            ref_tokens = ["A"] * 64 if i < 14 else ["B"] + ["A"] * 63
            epyc_prompts.append(make_prompt(f"p{i:02d}", "x", epyc_tokens, base))
            ref_prompts.append(make_prompt(f"p{i:02d}", "x", ref_tokens, base))
        rows, summary = c.compare(make_result_set(epyc_prompts),
                                   make_result_set(ref_prompts), max_mad=0.05)
        self.assertEqual(summary["n_pass_mad"], 20)  # MAD = 0
        self.assertEqual(summary["n_token1_match"], 14)  # only 14/20
        ok, text = c.verdict(summary, 18, 15)
        self.assertFalse(ok)
        self.assertIn("token-1", text)

    def test_runtime_failure_blocks_pass(self):
        # All prompts pass MAD + token-1; one has explicit error → auto-FAIL
        epyc_prompts = []
        ref_prompts = []
        for i in range(20):
            tokens = ["A"] * 64
            base = [-1.0] * 64
            if i == 5:
                epyc_prompts.append(make_prompt(f"p{i:02d}", "x", [], [],
                                                 error="GGML_ASSERT failure"))
            else:
                epyc_prompts.append(make_prompt(f"p{i:02d}", "x", tokens, base))
            ref_prompts.append(make_prompt(f"p{i:02d}", "x", tokens, base))
        rows, summary = c.compare(make_result_set(epyc_prompts),
                                   make_result_set(ref_prompts), max_mad=0.05)
        self.assertGreaterEqual(summary["n_runtime_fail"], 1)
        ok, text = c.verdict(summary, 18, 15)
        self.assertFalse(ok)
        self.assertIn("runtime failure", text)

    def test_missing_prompt_id_one_side_is_runtime_fail(self):
        # Reference has an extra prompt; treated as runtime failure on EPYC side
        epyc_prompts = [make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-1.0] * 64)
                        for i in range(19)]
        ref_prompts = [make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-1.0] * 64)
                       for i in range(20)]
        rows, summary = c.compare(make_result_set(epyc_prompts),
                                   make_result_set(ref_prompts), max_mad=0.05)
        # 19 prompts pass, 1 marked runtime-fail (missing on EPYC)
        self.assertEqual(summary["n_runtime_fail"], 1)


class TestStrictTokenCount(unittest.TestCase):
    """Per-prompt token-count enforcement — catches the false-pass on truncated runs."""

    def test_one_token_per_prompt_fails_when_64_requested(self):
        # 20 prompts × 1 token each, n_tokens_requested=64 → all marked runtime_failure
        epyc_p = [make_prompt(f"p{i:02d}", "x", ["A"], [-1.0]) for i in range(20)]
        ref_p = [make_prompt(f"p{i:02d}", "x", ["A"], [-1.0]) for i in range(20)]
        epyc = make_result_set(epyc_p, n_tokens_requested=64)
        ref = make_result_set(ref_p, n_tokens_requested=64)
        rows, summary = c.compare(epyc, ref, max_mad=0.05)
        self.assertEqual(summary["n_runtime_fail"], 20,
                         "20 short prompts must all be runtime_failure")
        ok, text = c.verdict(summary, 18, 15)
        self.assertFalse(ok, "Strict-token gate must FAIL on 1-token-per-prompt run")
        self.assertIn("runtime failure", text)

    def test_partial_truncation_some_pass(self):
        # 18 full-length + 2 truncated → 2 runtime_failure → verdict FAIL
        epyc_p = ([make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-1.0] * 64) for i in range(18)]
                  + [make_prompt(f"p{i:02d}", "x", ["A"] * 10, [-1.0] * 10) for i in (18, 19)])
        ref_p = [make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-1.0] * 64) for i in range(20)]
        rows, summary = c.compare(make_result_set(epyc_p, n_tokens_requested=64),
                                   make_result_set(ref_p, n_tokens_requested=64),
                                   max_mad=0.05)
        self.assertEqual(summary["n_runtime_fail"], 2)
        ok, _ = c.verdict(summary, 18, 15)
        self.assertFalse(ok)

    def test_explicit_min_tokens_override(self):
        # CLI override: --min-tokens-per-prompt=10 with 64-token n_tokens_requested
        # should allow shorter runs to pass.
        epyc_p = [make_prompt(f"p{i:02d}", "x", ["A"] * 20, [-1.0] * 20) for i in range(20)]
        ref_p = [make_prompt(f"p{i:02d}", "x", ["A"] * 20, [-1.0] * 20) for i in range(20)]
        rows, summary = c.compare(make_result_set(epyc_p, n_tokens_requested=64),
                                   make_result_set(ref_p, n_tokens_requested=64),
                                   max_mad=0.05,
                                   min_tokens_per_prompt=10)
        self.assertEqual(summary["n_runtime_fail"], 0)
        self.assertEqual(summary["n_pass_mad"], 20)


class TestStrictPromptCount(unittest.TestCase):
    """expected_n_prompts enforcement — catches truncated-run false-pass."""

    def test_below_expected_count_marks_missing_slots(self):
        epyc_p = [make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-1.0] * 64) for i in range(15)]
        ref_p = [make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-1.0] * 64) for i in range(15)]
        rows, summary = c.compare(make_result_set(epyc_p, n_tokens_requested=64),
                                   make_result_set(ref_p, n_tokens_requested=64),
                                   max_mad=0.05,
                                   expected_n_prompts=20)
        self.assertEqual(summary["n_runtime_fail"], 5,
                         "5 missing slots out of 20 expected")
        ok, _ = c.verdict(summary, 18, 15)
        self.assertFalse(ok)

    def test_none_disables_strict_count(self):
        # library callers can opt out
        epyc_p = [make_prompt(f"p{i:02d}", "x", ["A"], [-1.0]) for i in range(3)]
        ref_p = [make_prompt(f"p{i:02d}", "x", ["A"], [-1.0]) for i in range(3)]
        rows, summary = c.compare(make_result_set(epyc_p),
                                   make_result_set(ref_p),
                                   max_mad=0.05,
                                   expected_n_prompts=None)
        self.assertEqual(summary["n_runtime_fail"], 0)


class TestCLIIntegration(unittest.TestCase):
    """Smoke-test the script as a CLI: write two synthetic JSONs and verify exit code."""

    def test_cli_pass_exit_zero(self):
        prompts = [make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-1.0] * 64)
                   for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            epyc_path = tmp / "epyc.json"
            ref_path = tmp / "ref.json"
            out_path = tmp / "report.md"
            epyc_path.write_text(json.dumps(make_result_set(prompts)))
            ref_path.write_text(json.dumps(make_result_set(prompts)))
            script = Path(__file__).resolve().parent / "v4_quality_gate_compare.py"
            result = subprocess.run(
                [sys.executable, str(script),
                 "--epyc", str(epyc_path),
                 "--reference", str(ref_path),
                 "--output", str(out_path)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertTrue(out_path.exists())
            report = out_path.read_text()
            self.assertIn("**Verdict**: PASS", report)

    def test_cli_fail_exit_one(self):
        # Big offset → fails MAD → exit 1
        epyc_p = [make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-1.0] * 64) for i in range(20)]
        ref_p = [make_prompt(f"p{i:02d}", "x", ["A"] * 64, [-5.0] * 64) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            epyc_path = tmp / "epyc.json"
            ref_path = tmp / "ref.json"
            out_path = tmp / "report.md"
            epyc_path.write_text(json.dumps(make_result_set(epyc_p)))
            ref_path.write_text(json.dumps(make_result_set(ref_p)))
            script = Path(__file__).resolve().parent / "v4_quality_gate_compare.py"
            result = subprocess.run(
                [sys.executable, str(script),
                 "--epyc", str(epyc_path),
                 "--reference", str(ref_path),
                 "--output", str(out_path)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)

    def test_cli_default_min_tokens_64_blocks_misconfigured_run(self):
        """If a runner is misconfigured to capture only 1 token per prompt and
        the JSON says n_tokens_requested=1, the library auto-derive would let
        it pass. The CLI hard default of 64 (per §Merge Gates) must block it.
        """
        prompts = [make_prompt(f"p{i:02d}", "x", ["A"], [-1.0]) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            epyc_path = tmp / "epyc.json"
            ref_path = tmp / "ref.json"
            out_path = tmp / "report.md"
            # Both sides have only 1 token per prompt, n_tokens_requested=1.
            # No way for the JSON metadata alone to flag this.
            epyc_path.write_text(json.dumps(make_result_set(prompts, n_tokens_requested=1)))
            ref_path.write_text(json.dumps(make_result_set(prompts, n_tokens_requested=1)))
            script = Path(__file__).resolve().parent / "v4_quality_gate_compare.py"
            result = subprocess.run(
                [sys.executable, str(script),
                 "--epyc", str(epyc_path),
                 "--reference", str(ref_path),
                 "--output", str(out_path)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1,
                             "CLI must FAIL on 1-token-per-prompt even with "
                             "n_tokens_requested=1 in JSON")
            report = out_path.read_text()
            self.assertIn("**Verdict**: FAIL", report)
            self.assertIn("truncated", report.lower())

    def test_cli_min_tokens_override_allows_short_runs(self):
        """Side experiments can use --min-tokens-per-prompt=1 to permit
        short runs (e.g., debug-only token-level diffs).
        """
        prompts = [make_prompt(f"p{i:02d}", "x", ["A"], [-1.0]) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            epyc_path = tmp / "epyc.json"
            ref_path = tmp / "ref.json"
            out_path = tmp / "report.md"
            epyc_path.write_text(json.dumps(make_result_set(prompts, n_tokens_requested=1)))
            ref_path.write_text(json.dumps(make_result_set(prompts, n_tokens_requested=1)))
            script = Path(__file__).resolve().parent / "v4_quality_gate_compare.py"
            result = subprocess.run(
                [sys.executable, str(script),
                 "--epyc", str(epyc_path),
                 "--reference", str(ref_path),
                 "--output", str(out_path),
                 "--min-tokens-per-prompt", "1"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0,
                             f"override should pass: {result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
