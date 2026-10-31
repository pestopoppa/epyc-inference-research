"""Replay the original G2-CONC amendment-5 fixture set without a tokenizer or model.

Inputs were generated independently of the A4 triggering answer on 2026-10-24,
with real Qwen3.8-27B token ids, under ws58-g2conc-20261024/a5-fixtures/.
Ambiguous templated lists are retained as loop controls, not correctness labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import coherence_gate as CG
from coherence_gate import degeneracy as D

FIXTURES = json.loads((Path(__file__).parent / "fixtures/g2conc_v21.json").read_text())
BY_NAME = {row["name"]: row for row in FIXTURES}


@pytest.mark.parametrize("row", FIXTURES, ids=lambda row: row["name"])
def test_original_amendment_fixtures(row):
    verdict = D.classify(row["text"], row["ids"], "eos")
    assert verdict["classifier"] == "degeneracy.v2.1"
    assert verdict["semantic"] == "unchecked"
    if row["label"] == "amb":
        # These rows intentionally have no correctness label; preserve loop detection.
        assert "loop" in verdict["reasons"]
    else:
        assert verdict["cls"] == ("OK" if row["label"] == "legit" else "DEGENERATE")


def test_old_stats_remain_v2_and_do_not_claim_printable_ascii_replay():
    # Original v2 fires on 12% token concentration; full-text v2.1 needs 20%.
    stats = {"n": 1000, "uniq": .04, "top": .12, "run": 2, "words": 700,
             "ascii_ok": .97, "finish": "length", "cls": "SALAD"}
    verdict = D.classify_stats(stats)
    assert verdict["classifier"] == "degeneracy.v2"
    assert verdict["cls"] == "DEGENERATE"
    assert verdict["reasons"] == ["uniq+stuck"]
    assert "unchecked" in verdict["loop"]


@pytest.mark.parametrize("name,expected", [("legit_latex_Z7", "PASS"),
                                           ("deg_token_loop", "REGRESSION")])
def test_shared_gate_uses_v21_with_truth_and_real_tokens(name, expected):
    row = BY_NAME[name]
    anchor = BY_NAME["legit_prose"]
    base = {"id": "math", "text": anchor["text"], "token_ids": anchor["ids"],
            "answer_text": "D", "finish": "eos"}
    cand = {"id": "math", "text": row["text"], "token_ids": row["ids"],
            "answer_text": "D", "finish": "eos"}
    report = CG.evaluate([base], [cand], {"math": {"expected": "D",
                                                "scoring_method": "multiple_choice"}})
    assert report["degeneracy_classifier"] == "degeneracy.v2.1"
    item = report["items"][0]
    assert item["verdict"] == expected
    assert item["degeneracy"]["cand"]["classifier"] == report["degeneracy_classifier"]


def test_v1_garbage_character_stat_is_still_frozen():
    row = BY_NAME["legit_latex_Z7"]
    v1 = D.classify_v1(row["text"], row["ids"], "eos")
    v21 = D.classify(row["text"], row["ids"], "eos")
    assert v1["cls"] == "SALAD"
    assert "ascii" in v1["salad_reasons"]
    assert v21["ascii_ok_v1"] == v1["ascii_ok"]
    assert v21["ascii_ok"] == 1.0


@pytest.mark.parametrize("top_count,expected", [(199, "OK"), (200, "DEGENERATE")])
def test_low_uniqueness_needs_twenty_percent_concentration(top_count, expected):
    ids = [0] * top_count + [1 + k % 49 for k in range(1000 - top_count)]
    random.Random(33).shuffle(ids)
    verdict = D.classify("word " * 1000, ids, "length")
    assert verdict["run"] < 5 and verdict["loop"] is None
    assert verdict["cls"] == expected


@pytest.mark.parametrize("run,expected", [(4, "OK"), (5, "DEGENERATE")])
def test_low_uniqueness_needs_five_consecutive_tokens(run, expected):
    ids = [1 + k % 49 for k in range(1000 - run)]
    random.Random(17).shuffle(ids)
    verdict = D.classify("word " * 1000, [0] * run + ids, "length")
    assert verdict["top"] < .20 and verdict["loop"] is None
    assert verdict["cls"] == expected
