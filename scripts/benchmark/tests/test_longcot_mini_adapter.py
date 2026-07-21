#!/usr/bin/env python3
"""Self-contained tests for the LongCoT-Mini adapter + deterministic scorer.

K-LCM-1 (intake-386 / RE-4).  Runs under pytest if available, and — because the
research repo ships no pytest — also stands alone via the stdlib runner in
``__main__``::

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        scripts/benchmark/tests/test_longcot_mini_adapter.py

Every test is INFERENCE-FREE: the scorer is exercised against SYNTHETIC model
answers (never a server).  Dataset-backed tests skip cleanly if the Arrow data
is absent.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _BENCHMARK_DIR.parent
_RESEARCH_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_RESEARCH_ROOT), str(_SCRIPTS_DIR), str(_BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import longcot_mini_adapter as lcm  # noqa: E402
from longcot_mini_adapter import (  # noqa: E402
    LongCoTMiniAdapter,
    detect_canary_leak,
    score_structural,
)

_DATA_DIR = Path("/mnt/raid0/llm/epyc-inference-research/data/longcot-mini")


class _Skip(Exception):
    """Raised by a test to signal 'skipped' to the stdlib runner."""


def _require_data():
    if not _DATA_DIR.exists():
        raise _Skip(f"dataset not present at {_DATA_DIR}")


def _synthetic_prompt(domain, template, gold_value, canary="CANARY-UUID-XYZ",
                      is_scorable=True, question_id="q0"):
    """Build a prompt_dict of the exact shape _row_to_prompt emits (no dataset)."""
    return {
        "id": f"longcot_mini_{question_id}",
        "suite": "longcot_mini",
        "prompt": "solve it. Return your solution in the format: solution = ...",
        "expected": json.dumps(gold_value, ensure_ascii=False) if is_scorable else "",
        "scoring_method": lcm.SCORING_METHOD,
        "scoring_config": {"is_scorable": is_scorable, "domain": domain,
                           "template": template, "canary": canary},
        "metadata": {"question_id": question_id, "domain": domain,
                     "template": template, "canary": canary,
                     "is_scorable": is_scorable, "gold_raw": json.dumps(gold_value),
                     "gold_value": gold_value},
    }


# ── scorer: scalar (case-sensitive) ───────────────────────────────────────────

def test_chemistry_smiles_exact_and_case_sensitive():
    gold = "C1(CCCC1)NC1=CC=CC=2N1N=C(C2C2=CC(=NC=C2)F)C2=CC=C(C=C2)F"
    resp_ok = f"long reasoning...\nsolution = {gold}"
    assert score_structural(resp_ok, gold)["correct"] is True
    # SMILES is case-sensitive (lowercase = aromatic): a case flip must FAIL.
    resp_case = f"solution = {gold.lower()}"
    assert score_structural(resp_case, gold)["correct"] is False
    # a different molecule fails
    assert score_structural("solution = CCO", gold)["correct"] is False


def test_chess_number_string_bridges_int_and_string():
    gold = "391365"  # gold is a JSON string
    # model emits a bare integer — numeric normalization must bridge it
    assert score_structural("solution = 391365", gold)["correct"] is True
    # model emits the quoted string
    assert score_structural('solution = "391365"', gold)["correct"] is True
    assert score_structural("solution = 391366", gold)["correct"] is False


def test_chess_fen_is_case_sensitive():
    gold = "8/r7/kn3p2/1pr1pPpp/NP2PbPP/5K2/3B4/1bR1bB2 w - - 122 349"
    assert score_structural(f"solution = {gold}", gold)["correct"] is True
    # whitespace runs collapse (harmless for FEN single-spaces)
    assert score_structural(f"solution =   {gold}  ", gold)["correct"] is True
    # a black-rook vs white-rook case flip changes the board → FAIL
    assert score_structural("solution = 8/R7/kn3p2/1pr1pPpp/NP2PbPP/5K2/3B4/1bR1bB2 w - - 122 349", gold)["correct"] is False


# ── scorer: structured containers ─────────────────────────────────────────────

def test_cs_object_key_order_irrelevant():
    gold = {"Q1": "(M_1*(M_2*(M_3*M_4)))", "Q2": 3159991384, "Q3": 3, "Q4": 3, "Q5": 0}
    # reorder keys + int-as-string for Q2 → must still match structurally
    reordered = '{"Q5": 0, "Q2": "3159991384", "Q1": "(M_1*(M_2*(M_3*M_4)))", "Q4": 3, "Q3": 3}'
    assert score_structural(f"solution = {reordered}", gold)["correct"] is True
    wrong = dict(gold, Q3=4)
    assert score_structural(f"solution = {json.dumps(wrong)}", gold)["correct"] is False


def test_cs_int_array():
    gold = [15, 392, 2790]
    assert score_structural("solution = [15, 392, 2790]", gold)["correct"] is True
    # order matters for a list
    assert score_structural("solution = [392, 15, 2790]", gold)["correct"] is False


def test_math_list_bridges_numeric_and_keeps_expressions():
    gold = ["16", "13", "54", "89"]
    # model emits bare ints — numeric normalization bridges per element
    assert score_structural("solution = [16, 13, 54, 89]", gold)["correct"] is True
    # expression element compared as normalized string
    gold2 = ["2013^{4025}", "2692", "26"]
    assert score_structural('solution = ["2013^{4025}", 2692, 26]', gold2)["correct"] is True
    assert score_structural("solution = [16, 13, 54, 90]", gold)["correct"] is False


def test_multiline_container_solution_is_parsed():
    gold = [1, 2, 3]
    resp = "here is my answer:\nsolution = [1,\n 2,\n 3]\nthat's it"
    assert score_structural(resp, gold)["correct"] is True


def test_last_solution_marker_wins():
    gold = "42"
    # an earlier echoed 'solution = ...' (e.g. the format instruction) is ignored
    resp = "The format is solution = <integer>. Let me think... solution = 42"
    assert score_structural(resp, gold)["correct"] is True


def test_no_solution_marker_is_incorrect():
    r = score_structural("I could not solve this.", "42")
    assert r["correct"] is False and r["reason"] == "no_solution_marker"


# ── canary / contamination ────────────────────────────────────────────────────

def test_canary_leak_detection():
    canary = "e5625385-9b03-4787-82de-14b17369d703"
    assert detect_canary_leak(f"...{canary}...", canary) is True
    assert detect_canary_leak("clean output", canary) is False


# ── adapter convenience method ────────────────────────────────────────────────

def test_compute_score_for_result_scorable_and_canary():
    canary = "CANARY-UUID-XYZ"
    pd = _synthetic_prompt("chess", "piece_combinations", "391365", canary=canary)
    ok = LongCoTMiniAdapter.compute_score_for_result("solution = 391365", pd)
    assert ok["correct"] is True and ok["is_scorable"] is True
    assert ok["canary_leak"] is False and ok["domain"] == "chess"
    # A leaked canary typically surfaces in the model's reasoning, not glued to
    # the answer: score stays correct AND the contamination flag fires.
    leaked = LongCoTMiniAdapter.compute_score_for_result(
        f"I recall the tag {canary} from training.\nsolution = 391365", pd)
    assert leaked["correct"] is True and leaked["canary_leak"] is True


def test_compute_score_for_result_unscorable_is_none():
    pd = _synthetic_prompt("logic", "Sudoku", None, is_scorable=False)
    r = LongCoTMiniAdapter.compute_score_for_result("solution = [[1,2],[3,4]]", pd)
    assert r["correct"] is None and r["reason"] == "unscorable_null_gold"
    assert r["is_scorable"] is False


# ── dataset-backed load tests (skip if data absent) ───────────────────────────

def test_default_load_is_scorable_only():
    _require_data()
    adapter = LongCoTMiniAdapter()
    adapter._ensure_loaded()
    n = adapter.total_available
    assert n == 402, f"expected 402 scorable rows, got {n}"
    domains = {r["domain"] for r in adapter._dataset}
    assert domains == set(lcm.SCORABLE_DOMAINS), domains
    assert "logic" not in domains  # unscorable domain excluded by default


def test_include_unscorable_loads_all_507():
    _require_data()
    adapter = LongCoTMiniAdapter(include_unscorable=True)
    adapter._ensure_loaded()
    assert adapter.total_available == 507, adapter.total_available
    domains = {r["domain"] for r in adapter._dataset}
    assert "logic" in domains


def test_row_shape_and_canary_carried():
    _require_data()
    adapter = LongCoTMiniAdapter()
    q = adapter.sample(n=1, seed=42)[0]
    for key in ("id", "prompt", "expected", "scoring_method", "scoring_config",
                "metadata"):
        assert key in q, key
    assert q["scoring_method"] == "structural_exact_match"
    assert q["scoring_config"]["canary"]           # non-empty canary carried
    assert q["metadata"]["canary"] == q["scoring_config"]["canary"]
    assert "solution =" in q["prompt"].lower()
    # end-to-end: a synthetic correct answer scores True against the real gold
    gold = q["metadata"]["gold_value"]
    resp = f"...\nsolution = {json.dumps(gold, ensure_ascii=False)}"
    assert LongCoTMiniAdapter.compute_score_for_result(resp, q)["correct"] is True


def test_registration_and_get_adapter():
    from dataset_adapters import ADAPTER_SUITES, get_adapter
    assert "longcot_mini" in ADAPTER_SUITES
    adapter = get_adapter("longcot_mini")
    assert adapter is not None
    assert adapter.suite_name == "longcot_mini"


def test_suites_load_suite_bridge():
    _require_data()
    from suites import load_suite
    suite = load_suite("longcot_mini")
    assert suite is not None
    assert len(suite.questions) == 402
    q0 = suite.questions[0]
    assert q0.prompt and q0.expected  # prompt + gold flow through the bridge


# ── stdlib runner ─────────────────────────────────────────────────────────────

def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except _Skip as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
