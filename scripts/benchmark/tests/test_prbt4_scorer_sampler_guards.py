"""PRB-T4 (2026-09-17) guards for the TALE harness and the shared pool path.

Three defects surfaced by the PRB-T4 TALE-EP run (research ``b4d38ebc``):

1. ``livecodebench`` accuracy was vacuous. The live ``question_pool.jsonl``
   predates the 2026-08-12 oracle rebuild, so its rows still score
   ``substring 'def '``, which every Python answer contains.
2. ``mmlu_pro`` never ran. The adapter shipped ``scoring_config={}`` for an A-J
   suite, and the shared scorer only knows A-H, so every I/J gold is
   unscoreable.
3. The ``math`` sample was 100% ``gsm8k``. The harness took the first ``n``
   rows in file order, and the pool lists 1,319 gsm8k rows before its 500
   MATH-500 rows.

These tests use synthetic pools and the pinned local source snapshots only.
No server is contacted.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import dataset_adapters as da  # noqa: E402
import debug_scorer  # noqa: E402  (shim -> orchestrator B7 scorer)
import eval_tale_budget as tb  # noqa: E402
import eval_trimr  # noqa: E402
import question_pool as qp  # noqa: E402

SCORER_HAS_LABELS = hasattr(debug_scorer, "_choice_labels")
needs_fixed_scorer = pytest.mark.skipif(
    not SCORER_HAS_LABELS,
    reason="shared orchestrator clone predates the PRB-T4 scorer fix (no choice_labels support)",
)

TEN = [f"option {i}" for i in range(10)]
STALE_LCB = {
    "id": "leetcode_two-sum", "suite": "livecodebench", "prompt": "Two Sum",
    "expected": "def ", "scoring_method": "substring",
    "scoring_config": {"language": "python", "timeout": 30,
                       "case_sensitive": True, "substring": "def "},
}
STALE_MMLU = {
    "id": "mmlu_pro_business_00000", "suite": "mmlu_pro", "prompt": "Q",
    "expected": "I", "scoring_method": "multiple_choice", "scoring_config": {},
}
FIXED_MMLU = dict(STALE_MMLU, scoring_config={"choices": TEN, "choice_labels": "ABCDEFGHIJ"})


def _math_population() -> list[dict]:
    rows = [{"id": f"gsm8k_{i:05d}", "suite": "math"} for i in range(1319)]
    subjects = {"Algebra": 124, "Intermediate Algebra": 97, "Prealgebra": 82,
                "Number Theory": 62, "Precalculus": 56, "Geometry": 41,
                "Counting & Probability": 38}
    for subject, count in subjects.items():
        rows += [{"id": f"math500_{subject}_{i:05d}", "suite": "math"} for i in range(count)]
    return rows


def _write_pool(path: Path, rows: list[dict]) -> Path:
    header = {"__pool_metadata__": True, "generated_at": "2026-07-27T00:00:00+00:00",
              "total_questions": len(rows), "suites": dict(Counter(r["suite"] for r in rows))}
    with path.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


# ── 3. sampler ──────────────────────────────────────────────────────────────


def test_source_stratum_keys() -> None:
    assert qp.source_stratum({"id": "gsm8k_00012"}) == "gsm8k"
    assert qp.source_stratum({"id": "math500_Counting & Probability_00001"}) == "math500_Counting & Probability"
    assert qp.source_stratum({"id": "olympiadbench_geometry_00007"}) == "olympiadbench_geometry"
    assert qp.source_stratum({"id": "leetcode_two-sum"}) == "leetcode"


def test_math_sample_is_proportional_not_first_n() -> None:
    pop = _math_population()
    picked = qp.stratified_sample(pop, 150, seed=42)
    assert len(picked) == 150 and len({r["id"] for r in picked}) == 150
    gsm = sum(r["id"].startswith("gsm8k_") for r in picked)
    # 1319/1819 of 150 = 108.8
    assert gsm in (108, 109), gsm
    subjects = {qp.source_stratum(r) for r in picked if not r["id"].startswith("gsm8k_")}
    assert len(subjects) == 7


def test_sample_is_deterministic_and_file_order_independent() -> None:
    pop = _math_population()
    shuffled = list(pop)
    random.Random(7).shuffle(shuffled)
    a = [r["id"] for r in qp.stratified_sample(pop, 60, seed=42)]
    b = [r["id"] for r in qp.stratified_sample(shuffled, 60, seed=42)]
    c = [r["id"] for r in qp.stratified_sample(pop, 60, seed=43)]
    assert a == b
    assert a != c


def test_sample_larger_than_population_returns_everything() -> None:
    pop = _math_population()[:5]
    assert sorted(r["id"] for r in qp.stratified_sample(pop, 50, seed=1)) == sorted(r["id"] for r in pop)


def test_tale_loader_no_longer_takes_the_first_n_rows(tmp_path: Path) -> None:
    pool = _write_pool(tmp_path / "pool.jsonl", _math_population())
    comp: dict = {}
    qs = tb.load_questions(["math"], 150, seed=42, pool_path=pool, composition=comp)
    assert any(q["id"].startswith("math500_") for q in qs), "regressed to first-n (gsm8k only)"
    assert comp["math"]["population"]["gsm8k"] == 1319
    assert sum(comp["math"]["sample"].values()) == 150


def test_trimr_loader_no_longer_takes_the_first_n_rows(tmp_path: Path, monkeypatch) -> None:
    pool = _write_pool(tmp_path / "pool.jsonl", _math_population())
    monkeypatch.setattr(eval_trimr, "POOL_PATH", pool)
    qs = eval_trimr.load_questions(["math"], 40)
    assert any(q["id"].startswith("math500_") for q in qs)


# ── 1 + 2. oracle preflight and three-valued scoring ────────────────────────


def test_stale_livecodebench_row_is_flagged_before_inference() -> None:
    assert "vacuous code oracle" in tb.oracle_defect(STALE_LCB)


def test_stale_mmlu_pro_row_is_flagged_before_inference() -> None:
    assert "unscoreable multiple_choice gold" in tb.oracle_defect(STALE_MMLU)


@needs_fixed_scorer
def test_fixed_mmlu_pro_row_passes_preflight_and_scores_both_ways() -> None:
    assert tb.oracle_defect(FIXED_MMLU) is None
    assert tb.score_question("I", FIXED_MMLU) is True
    assert tb.score_question("Answer: J", FIXED_MMLU) is False
    assert tb.score_question("I don't know", FIXED_MMLU) is False


def test_code_execution_rows_are_not_flagged() -> None:
    row = {"id": "leetcode_x", "suite": "livecodebench", "expected": "f",
           "scoring_method": "code_execution",
           "scoring_config": {"entry_point": "f", "entry_point_cases": [[[1], 1]]}}
    assert tb.oracle_defect(row) is None


@needs_fixed_scorer
def test_known_wrong_code_no_longer_passes_the_stale_row() -> None:
    """`def solve(): pass` used to score True here. Now the row is excluded."""
    assert tb.score_question("def solve():\n    pass", STALE_LCB) is None


def test_main_refuses_a_defective_suite_without_sending_a_request(tmp_path: Path) -> None:
    pool = _write_pool(tmp_path / "pool.jsonl", [STALE_LCB, STALE_MMLU])

    def boom(*a, **k):
        raise AssertionError("no request may be sent for a defective suite")

    with pytest.raises(SystemExit) as exc:
        tb.main(["--suites", "livecodebench", "mmlu_pro", "--n-questions", "5",
                 "--pool", str(pool), "--endpoint", "http://127.0.0.1:9"],
                poster=boom, fetcher=boom)
    assert exc.value.code == 2


def test_summary_excludes_unscoreable_rows_from_accuracy() -> None:
    rows = [
        tb.TrialResult("a", "s", "baseline", "", "", correct=True, source="x"),
        tb.TrialResult("b", "s", "baseline", "", "", correct=None, source="x"),
        tb.TrialResult("c", "s", "baseline", "", "", correct=False, source="y"),
    ]
    cell = tb.summarize(rows)["s"]["baseline"]
    assert cell["accuracy"] == 0.5
    assert cell["n_scored"] == 2 and cell["n_unscoreable"] == 1
    assert cell["by_source"]["x"] == {"n": 2, "n_scored": 1, "correct": 1, "accuracy": 1.0}


# ── adapters re-derived from the pinned source snapshots ────────────────────

MMLU_SNAPSHOT_MISSING = not da.MMLUProAdapter.SNAPSHOT_PARQUET.is_file()
LCB_SNAPSHOT_MISSING = not da.LiveCodeBenchAdapter.SNAPSHOT_JSONL.is_file()


def test_mmlu_pro_gold_is_derived_from_the_index_and_cross_checked() -> None:
    row = {"question": "Q", "options": TEN, "answer": "I", "answer_index": 8}
    assert da.MMLUProAdapter.gold_letter(row) == "I"
    for bad in ({"answer": "H"}, {"answer_index": 10}, {"answer_index": None}, {"options": []}):
        with pytest.raises(ValueError):
            da.MMLUProAdapter.gold_letter({**row, **bad})


@needs_fixed_scorer
@pytest.mark.skipif(MMLU_SNAPSHOT_MISSING, reason="MMLU-Pro parquet snapshot not cached")
def test_every_rebuilt_mmlu_pro_row_scores_right_and_wrong_correctly() -> None:
    adapter = da.MMLUProAdapter()
    rows = adapter.extract_all()
    assert len(rows) == 12032 and adapter.dropped_rows == 0
    assert sum(r["expected"] in "IJ" for r in rows) == 2053
    for q in rows:
        gold, cfg = q["expected"], q["scoring_config"]
        wrong = "A" if gold != "A" else "B"
        assert debug_scorer.score_answer(gold, gold, "multiple_choice", cfg) is True
        assert debug_scorer.score_answer(wrong, gold, "multiple_choice", cfg) is False


@pytest.mark.skipif(LCB_SNAPSHOT_MISSING, reason="leetcode snapshot not cached")
def test_refreshed_pool_replaces_stale_livecodebench_and_keeps_other_rows(tmp_path: Path) -> None:
    other = {"id": "gsm8k_00000", "suite": "math", "prompt": "1+1", "expected": "2",
             "scoring_method": "exact_match", "scoring_config": {}}
    src = _write_pool(tmp_path / "live.jsonl", [other, STALE_LCB])
    out = tmp_path / "refreshed.jsonl"
    report = qp.refresh_suites(["livecodebench"], source_path=src, output_path=out)
    assert report["livecodebench"]["removed"] == 1
    assert report["livecodebench"]["added"] == 704

    lines = out.read_text().splitlines()
    header = json.loads(lines[0])
    assert header["suites"]["livecodebench"] == 704 and header["total_questions"] == 705
    assert lines[1] == json.dumps(other)  # untouched rows are byte-preserved
    rows = [json.loads(l) for l in lines[1:]]
    lcb = [r for r in rows if r["suite"] == "livecodebench"]
    assert all(r["scoring_method"] == "code_execution" for r in lcb)
    assert all(tb.oracle_defect(r) is None for r in lcb)

    two_sum = next(r for r in lcb if r["id"] == "leetcode_two-sum")
    right = ("```python\ndef twoSum(nums, target):\n"
             "    seen = {}\n"
             "    for i, x in enumerate(nums):\n"
             "        if target - x in seen:\n"
             "            return [seen[target - x], i]\n"
             "        seen[x] = i\n```")
    assert tb.score_question(right, two_sum) is True
    assert tb.score_question("```python\ndef twoSum(nums, target):\n    return [0, 0]\n```", two_sum) is False
    assert tb.score_question("```python\ndef solve():\n    pass\n```", two_sum) is False


def test_refresh_refuses_to_overwrite_the_live_pool_by_default() -> None:
    with pytest.raises(qp.PoolBuildInvariantError, match="live pool"):
        qp.refresh_suites(["livecodebench"], output_path=qp.POOL_FILE)
