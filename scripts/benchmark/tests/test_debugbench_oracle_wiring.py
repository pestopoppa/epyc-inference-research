"""The live pool builder emits the rebuilt debugbench oracle, and this repo can score it.

`benchmarks/prompts/question_pool.jsonl` is built here (question_pool.py ->
dataset_adapters.DebugBenchAdapter), while the eval tower scores it with the
ORCHESTRATOR copy of debug_scorer.py. Fixing only the orchestrator's adapter would
have left the live builder emitting the vacuous oracle, so these tests pin the
research side of the wiring:

  * the adapter emits `programmatic` / `code_patch`, not a solution prefix;
  * echoing the buggy code FAILS and the reference solution PASSES;
  * THIS repo's debug_scorer.py is a delegation shim to the orchestrator's B7
    copy (2026-08-23, scorer-fork-drift-audit-2026-07-22 residual row), so it
    agrees with the orchestrator's on everything, including raising on an
    unknown verifier instead of falling back to a case-insensitive substring
    match on `expected` — a `code_patch` row can never be silently scored by a
    weaker oracle.

Origin: epyc-root `artifacts/audit/debugbench-oracle-vacuity-20260812.md`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
for _p in (str(BENCHMARK_DIR), str(BENCHMARK_DIR.parent), str(BENCHMARK_DIR.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_scorer_spec = importlib.util.spec_from_file_location(
    "research_debug_scorer_under_test", BENCHMARK_DIR / "debug_scorer.py")
research_scorer = importlib.util.module_from_spec(_scorer_spec)
sys.modules["research_debug_scorer_under_test"] = research_scorer
_scorer_spec.loader.exec_module(research_scorer)

_orch_spec = importlib.util.spec_from_file_location(
    "orch_debug_scorer_under_test",
    "/mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/debug_scorer.py")
orch_scorer = importlib.util.module_from_spec(_orch_spec)
sys.modules["orch_debug_scorer_under_test"] = orch_scorer
_orch_spec.loader.exec_module(orch_scorer)

import dataset_adapters  # noqa: E402


BUGGY = """\
class Solution {
    public int longestCycle(int[] edges) {
        int ans = -1;
        for (int i = 0; i <= edges.length; i++) {
            ans = Math.max(ans, walk(edges, i));
        }
        return ans;
    }
}"""

SOLUTION = BUGGY.replace("i <= edges.length", "i < edges.length")


def _upstream_row(language: str = "java") -> dict:
    return {
        "question": "Return the length of the longest cycle.",
        "buggy_code": BUGGY,
        "solution": SOLUTION,
        "bug_explanation": "loop bound is off by one",
        "examples": [],
        "constraints": "",
        "language": language,
        "level": "medium",
        "slug": "longest-cycle-in-a-graph",
        "category": "logic error",
    }


def _row(language: str = "java") -> dict:
    return dataset_adapters.DebugBenchAdapter()._row_to_prompt(0, _upstream_row(language))


def test_the_live_pool_builder_emits_the_diff_oracle() -> None:
    row = _row()
    assert row["scoring_method"] == "programmatic"
    assert row["scoring_config"]["verifier"] == "code_patch"
    assert row["expected"] != SOLUTION[:100]


def test_echoing_the_buggy_code_fails_the_row_this_repo_builds() -> None:
    """The decisive test, on the rows that actually reach the pool."""
    row = _row()
    assert research_scorer.score_answer(
        BUGGY, row["expected"], row["scoring_method"], row["scoring_config"]) is False


def test_the_reference_solution_passes_the_row_this_repo_builds() -> None:
    row = _row()
    assert research_scorer.score_answer(
        SOLUTION, row["expected"], row["scoring_method"], row["scoring_config"]) is True


def test_both_repo_copies_of_the_scorer_agree_on_the_same_row() -> None:
    """The two copies diverged long ago; on this oracle they must not."""
    row = _row()
    for answer, expected_verdict in ((BUGGY, False), (SOLUTION, True)):
        research_verdict = research_scorer.score_answer(
            answer, row["expected"], row["scoring_method"], row["scoring_config"])
        orch_verdict = orch_scorer.score_answer(
            answer, row["expected"], row["scoring_method"], row["scoring_config"])
        assert research_verdict is expected_verdict
        assert orch_verdict is expected_verdict


def test_the_delegated_scorer_refuses_an_unknown_verifier() -> None:
    """The research path is B7 now: fail-closed, not a weaker oracle.

    Pre-2026-08-23, this repo's `_score_programmatic` answered an unknown
    verifier with a case-insensitive substring match on `expected` instead of
    raising (pinned by this test in its original form), so deleting `code_patch`
    from its table made the row quietly scored by something else. Since the
    delegation shim (scorer-fork-drift-audit-2026-07-22 residual row) the
    research path inherits B7's SCORE-25 rejection: an unknown verifier raises,
    so an unscoreable row fails closed instead of being quietly mis-scored.
    """
    row = _row()
    verifiers_without_code_patch = dict(row["scoring_config"], verifier="not_a_verifier")
    with pytest.raises(ValueError):
        research_scorer.score_answer(
            "some prose that happens to mention nothing relevant",
            row["expected"], "programmatic", verifiers_without_code_patch)
    with pytest.raises(ValueError):
        research_scorer.score_answer(
            row["expected"], row["expected"], "programmatic",
            verifiers_without_code_patch)


def test_the_builder_drops_a_row_it_cannot_validate() -> None:
    upstream = dict(_upstream_row(), solution=BUGGY)
    assert dataset_adapters.DebugBenchAdapter()._row_to_prompt(0, upstream) == {}


def test_python_rows_are_no_longer_scored_by_an_always_false_oracle() -> None:
    """1,414 upstream python rows shipped `code_execution` with no test_code."""
    assert research_scorer.score_answer(
        SOLUTION, SOLUTION[:100], "code_execution",
        {"language": "python", "timeout": 30}) is False
    row = _row("python3")
    assert row["scoring_method"] == "programmatic"
    assert research_scorer.score_answer(
        SOLUTION, row["expected"], row["scoring_method"], row["scoring_config"]) is True


def test_the_buggy_code_is_not_truncated_out_of_the_prompt() -> None:
    long_buggy = BUGGY.replace(
        "int ans = -1;", "int ans = -1;\n" + "        int pad = 0;\n" * 90)
    upstream = dict(_upstream_row(), buggy_code=long_buggy)
    row = dataset_adapters.DebugBenchAdapter()._row_to_prompt(0, upstream)
    assert len(long_buggy) > 1000
    assert long_buggy in row["prompt"]
