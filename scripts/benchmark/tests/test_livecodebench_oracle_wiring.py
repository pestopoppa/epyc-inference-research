"""The live pool builder emits the rebuilt livecodebench oracle, and this repo can score it.

`benchmarks/prompts/question_pool.jsonl` is built HERE (question_pool.py ->
dataset_adapters.LiveCodeBenchAdapter), while the eval tower scores it with the
ORCHESTRATOR copy of debug_scorer.py. Fixing only the orchestrator's adapter would
have left the live builder re-emitting the vacuous oracle on the next rebuild, so
these tests pin the research side of the wiring:

  * all 2,360 rows used to carry `expected == "def "` — ONE value across the whole
    suite — scored by `substring`. Measured through the real scorer over every
    upstream row, `def solve(): pass` passed 100% of them and echoing the prompt
    passed 100% of them;
  * the adapter now emits `code_execution` with per-question `entry_point_cases`
    and DROPS the 1,656 rows for which no case can be manufactured, rather than
    downgrading them back to a string match;
  * THIS repo's diverged debug_scorer.py agrees with the orchestrator's on the
    `entry_point` oracle — an unported copy would score these rows differently.

Origin: epyc-root `artifacts/audit/unscoreable-rows-livecodebench-cruxeval-mah-20260812.md`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
for _p in (str(BENCHMARK_DIR), str(BENCHMARK_DIR.parent), str(BENCHMARK_DIR.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset_adapters as da  # noqa: E402

_scorer_spec = importlib.util.spec_from_file_location(
    "research_scorer_for_lcb", BENCHMARK_DIR / "debug_scorer.py")
research_scorer = importlib.util.module_from_spec(_scorer_spec)
sys.modules["research_scorer_for_lcb"] = research_scorer
_scorer_spec.loader.exec_module(research_scorer)

_orch_spec = importlib.util.spec_from_file_location(
    "orch_scorer_for_lcb",
    "/mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/debug_scorer.py")
orch_scorer = importlib.util.module_from_spec(_orch_spec)
sys.modules["orch_scorer_for_lcb"] = orch_scorer
_orch_spec.loader.exec_module(orch_scorer)

UPSTREAM = Path(
    "/mnt/raid0/llm/hf-home/hub/datasets--greengerong--leetcode/snapshots/"
    "00f2d466dc0f00f65a0b6938c4c11a57f721db81/leetcode-train.jsonl"
)
_SNAPSHOT_MISSING = not UPSTREAM.exists()

CORRECT = (
    "def twoSum(nums, target):\n"
    "    for i in range(len(nums)):\n"
    "        for j in range(i + 1, len(nums)):\n"
    "            if nums[i] + nums[j] == target:\n"
    "                return [i, j]\n"
    "    return []\n"
)


def _fenced(code: str) -> str:
    return f"```python\n{code}\n```"


def _two_sum_row() -> dict:
    with UPSTREAM.open() as handle:
        return json.loads(handle.readline())


def _adapter() -> "da.LiveCodeBenchAdapter":
    adapter = da.LiveCodeBenchAdapter()
    adapter._scoreable_cache = None
    return adapter


def test_the_manifest_is_reachable_from_this_repo() -> None:
    """It lives in epyc-orchestrator, beside the scorer it was validated against."""
    assert Path(da._LIVECODEBENCH_MANIFEST_PATH).exists(), (
        f"{da._LIVECODEBENCH_MANIFEST_PATH} missing — regenerate with "
        "epyc-orchestrator/scripts/benchmark/livecodebench_oracle.py --manifest"
    )
    assert da.livecodebench_manifest()["oracles"]


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_the_builder_emits_an_executable_oracle_not_the_string_def() -> None:
    question = _adapter()._row_to_prompt(0, _two_sum_row())
    assert question["scoring_method"] == "code_execution"
    assert question["expected"] != "def "
    assert question["scoring_config"]["entry_point"] == question["expected"]
    assert question["scoring_config"]["entry_point_cases"]
    assert question["scoring_config"].get("test_code") is None


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_the_builder_states_the_required_signature_in_the_prompt() -> None:
    """Without it the oracle measures function-name guessing."""
    question = _adapter()._row_to_prompt(0, _two_sum_row())
    assert "def twoSum(nums, target):" in question["prompt"]


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_both_scorer_copies_agree_on_the_two_directions() -> None:
    """The orchestrator copy scores the pool; this copy must not disagree."""
    question = _adapter()._row_to_prompt(0, _two_sum_row())
    stub = _fenced("def twoSum(nums, target):\n    pass")
    for scorer in (orch_scorer, research_scorer):
        args = (question["expected"], question["scoring_method"],
                question["scoring_config"])
        assert bool(scorer.score_answer(question["prompt"], *args)) is False
        assert bool(scorer.score_answer(stub, *args)) is False
        assert bool(scorer.score_answer(_fenced(CORRECT), *args)) is True


def test_this_repos_scorer_runs_entry_point_cases_instead_of_a_zero_arg_assert() -> None:
    """Why the port exists — pinned, so the reason cannot be forgotten.

    This copy used to answer an `entry_point` oracle with
    `assert f() == <expected text>`. On a function that takes arguments that can
    NEVER pass, so a correct answer scored False: the half-a-suite trap this repo
    already hit once on debugbench's python rows, in the opposite direction from
    the vacuous `"def "` oracle but with the same net effect — a number that does
    not measure the model.
    """
    config = {
        "language": "python",
        "timeout": 10,
        "entry_point": "double",
        "entry_point_cases": [
            {"args": [2], "expected": 4},
            {"args": [3], "expected": 6},
        ],
    }
    good = _fenced("def double(n):\n    return n * 2")
    bad = _fenced("def double(n):\n    return 4")
    assert bool(research_scorer.score_answer(good, "double", "code_execution", config)) is True
    assert bool(research_scorer.score_answer(bad, "double", "code_execution", config)) is False


def test_this_repos_scorer_refuses_an_entry_point_oracle_with_no_executable_cases() -> None:
    """Refuse loudly rather than book a scorer defect as a wrong answer.

    The orchestrator copy — the one the eval tower scores this pool with — already
    raises here. A copy that silently returns False instead disagrees with the
    authoritative scorer about what a row MEANS.
    """
    config = {"language": "python", "timeout": 10, "entry_point": "double"}
    with pytest.raises(Exception):
        research_scorer.score_answer(
            _fenced("def double(n):\n    return n * 2"), "4", "code_execution", config)


def test_a_row_with_no_validated_oracle_is_dropped_not_downgraded() -> None:
    row = {"slug": "no-such-problem-in-the-manifest", "title": "X", "content": "Y"}
    assert _adapter()._row_to_prompt(0, row) is None


def test_a_missing_manifest_empties_the_suite_rather_than_scoring_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed. A fail-open default would resume scoring `"def "` forever."""
    monkeypatch.setattr(da, "_LIVECODEBENCH_MANIFEST", None)
    monkeypatch.setattr(da, "_LIVECODEBENCH_MANIFEST_PATH", "/nonexistent/x.json")
    assert da.livecodebench_manifest() == {"oracles": {}}
    monkeypatch.setattr(da, "_LIVECODEBENCH_MANIFEST", None)


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_extraction_books_the_drops_under_their_real_reason() -> None:
    """`empty_prompt` would misname 1,656 rows whose prompt is fine and oracle is not."""
    adapter = _adapter()
    adapter._dataset = [
        json.loads(line) for line in UPSTREAM.read_text().splitlines() if line.strip()
    ]
    rows = adapter.extract_all()
    assert len(rows) == len(da.livecodebench_manifest()["oracles"])
    assert adapter.dropped_by_reason == {"no_validated_oracle": 2360 - len(rows)}
    assert "empty_prompt" not in adapter.dropped_by_reason


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_the_emitted_suite_no_longer_shares_one_expected_value() -> None:
    """The defect in one line: 2,360 rows, 1 distinct `expected`, `"def "`."""
    adapter = _adapter()
    adapter._dataset = [
        json.loads(line) for line in UPSTREAM.read_text().splitlines() if line.strip()
    ]
    rows = adapter.extract_all()
    assert len({row["expected"] for row in rows}) > len(rows) / 2
    assert {row["scoring_method"] for row in rows} == {"code_execution"}


@pytest.mark.skipif(_SNAPSHOT_MISSING, reason="upstream leetcode snapshot not cached")
def test_sampling_draws_only_from_rows_that_have_an_oracle() -> None:
    """`sample(n)` must return n scoreable questions, not n minus the drops."""
    adapter = _adapter()
    adapter._dataset = [
        json.loads(line) for line in UPSTREAM.read_text().splitlines() if line.strip()
    ]
    assert len(adapter.sample(20)) == 20
    stratified = adapter.sample(30, stratify=True)
    assert len(stratified) == 30
    assert len({question["tier"] for question in stratified}) == 3
