"""Tests for niah_scorer (E1a strict + lenient). Synthetic haystacks only, no inference.

Run: python -m pytest scripts/benchmark/test_niah_scorer.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import niah_scorer as ns


def _haystack(seed: int, n_words: int = 400):
    rng = random.Random(seed)
    key = f"key_{rng.randint(1000, 9999)}"
    value = f"value_{rng.randint(100000, 999999)}"
    words = [f"word{rng.randint(0, 10000)}" for _ in range(n_words)]
    words.insert(rng.randrange(n_words), f"The special {key} is {value}.")
    return " ".join(words), key, value


# ── the E1a distinction: format failure vs retrieval failure ────────────────

def test_bare_answer_passes_both():
    r = ns.score_niah("value_123456", "value_123456")
    assert r == {"decidable": True, "strict": True, "lenient": True, "cause": None}


def test_outer_whitespace_is_forgiven_by_strict():
    assert ns.score_niah("  value_1\n", "value_1")["strict"] is True


def test_narrative_answer_is_format_only_failure():
    """The arXiv:2603.02615 A.3 case: right answer, long narrative, strict scorer says 0."""
    resp = "After scanning the document, I found that the special key_4242 is VALUE_777777."
    r = ns.score_niah(resp, "value_777777")
    assert r["strict"] is False and r["lenient"] is True


@pytest.mark.parametrize("resp", [
    "Value_1",                      # case
    "ｖａｌｕｅ_１",  # NFKC fullwidth -> ascii
    "value​_1",                # zero-width space (Cf)
    "﻿value_1",                # BOM
    "The answer:\n\n  value_1  .",  # prose + punctuation
    "**value_1**",                  # markdown
])
def test_lenient_normalizations(resp):
    r = ns.score_niah(resp, "value_1")
    assert r["lenient"] is True
    assert r["strict"] is False


def test_whitespace_collapse_inside_multiword_needle():
    assert ns.score_lenient("it is the  blue\n\tcanary ok", "blue canary")


@pytest.mark.parametrize("resp", [
    "value_123",        # needle is a prefix of a longer token
    "xvalue_12",        # suffix of a longer token
    "value_12_extra",   # underscore is a word char
    "value_13",         # wrong answer
    "",                 # empty
])
def test_lenient_rejects_non_matches(resp):
    assert ns.score_lenient(resp, "value_12") is False


def test_wrong_answer_fails_both():
    r = ns.score_niah("The special key is value_000001.", "value_999999")
    assert r["strict"] is False and r["lenient"] is False


def test_none_response_is_decidable_fail():
    r = ns.score_niah(None, "value_1")
    assert r["decidable"] is True and r["strict"] is False and r["lenient"] is False


@pytest.mark.parametrize("gold", [None, "", "   ", "​", []])
def test_empty_reference_is_undecidable(gold):
    r = ns.score_niah("anything", gold)
    assert r["decidable"] is False and r["strict"] is None and r["cause"] == "no_reference"


def test_list_of_golds_any_match():
    r = ns.score_niah("Paris", ["paris, france", "Paris"])
    assert r["strict"] is True and r["lenient"] is True


# ── invariant and determinism ──────────────────────────────────────────────

def test_strict_implies_lenient_property():
    rng = random.Random(7)
    alphabet = ["a", "B", "1", "_", " ", "\n", "é", "Ａ", ".", "-"]
    for _ in range(3000):
        gold = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
        resp = gold if rng.random() < 0.5 else "".join(
            rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
        r = ns.score_niah(resp, gold)
        if r["decidable"] and r["strict"]:
            assert r["lenient"], (resp, gold)


def test_synthetic_haystack_batch_and_aggregate():
    recs = []
    for i in range(20):
        _, key, value = _haystack(i)
        if i < 8:
            resp = value                                   # clean
        elif i < 14:
            resp = f"The special {key} is {value}."        # format-only failure
        elif i < 18:
            resp = f"I could not find {key}."              # retrieval failure
        else:
            resp = value.upper()                           # case-only failure
        recs.append({"id": f"s{i}", "response": resp, "expected": value})
    recs.append({"id": "bad", "response": "x", "expected": ""})
    rep = ns.score_records(recs)
    s = rep["summary"]
    assert s["n_items"] == 21 and s["n_scored"] == 20 and s["n_undecidable"] == 1
    assert s["strict_correct"] == 8 and s["lenient_correct"] == 16
    assert s["strict_accuracy"] == pytest.approx(0.4)
    assert s["lenient_accuracy"] == pytest.approx(0.8)
    assert s["format_gap"] == pytest.approx(0.4)
    assert s["format_only_failures"] == 8
    assert s["metric_direction"]["strict_accuracy"] == "higher_better"
    assert s["scorer_id"] == ns.SCORER_ID
    # deterministic: identical input -> identical output
    assert json.dumps(rep, sort_keys=True) == json.dumps(ns.score_records(recs), sort_keys=True)


def test_aggregate_empty_is_none_not_zero():
    s = ns.aggregate([])
    assert s["strict_accuracy"] is None and s["lenient_accuracy"] is None
    assert s["format_gap"] is None


def test_ruler_adapter_niah_tasks_score_without_converter():
    """Wiring: the in-repo RULER NIAH adapter's prompt dicts score directly."""
    lca = pytest.importorskip("long_context_adapters")
    ad = lca.RULERAdapter(context_length=512, num_examples=5)
    ad._ensure_loaded()
    prompts = [ad._row_to_prompt(i, row) for i, row in enumerate(ad._dataset)]
    assert all(p["expected"] in p["prompt"] for p in prompts)
    recs = [{"id": p["id"], "response": f"The value is {p['expected']}.",
             "expected": p["expected"]} for p in prompts]
    s = ns.score_records(recs)["summary"]
    assert s["n_scored"] == 5 and s["strict_accuracy"] == 0.0 and s["lenient_accuracy"] == 1.0


def test_cli_replay_is_byte_deterministic(tmp_path):
    inp = tmp_path / "out.jsonl"
    rows = [{"qid": "a", "answer": "The special key_1 is value_9.", "gold": "value_9"},
            {"qid": "b", "answer": "value_8", "gold": "value_8"}]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    o1, o2 = tmp_path / "r1.json", tmp_path / "r2.json"
    args = ["--in", str(inp), "--response-field", "answer", "--expected-field", "gold",
            "--id-field", "qid"]
    assert ns.main(args + ["--out", str(o1)]) == 0
    assert ns.main(args + ["--out", str(o2)]) == 0
    assert o1.read_bytes() == o2.read_bytes()
    rep = json.loads(o1.read_text())
    assert rep["summary"]["strict_correct"] == 1 and rep["summary"]["lenient_correct"] == 2
    assert [i["id"] for i in rep["items"]] == ["a", "b"]


def test_cli_rejects_bad_jsonl(tmp_path):
    inp = tmp_path / "bad.jsonl"
    inp.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ns.main(["--in", str(inp)])
