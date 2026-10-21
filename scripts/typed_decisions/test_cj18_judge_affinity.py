from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).with_name("cj18_judge_affinity.py")
_SPEC = importlib.util.spec_from_file_location("cj18_judge_affinity_test_target", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)
INVALID, _stats, affinity_table, load_rows = _MOD.INVALID, _MOD._stats, _MOD.affinity_table, _MOD.load_rows


def _rows(a, b, gold):
    return [{"row_id": "r", "gold": gold, "outputs": {"a": a, "b": b}}]


def test_identical_judges_have_ec_one_and_positive_capa():
    gold = [True, False, True, False]
    wrong = [True, True, True, False]  # 3/4 right
    s = _stats(_rows(wrong, wrong, gold), "a", "b")
    assert s["ec"] == 1.0 and s["capa"] > 0.9


def test_independent_accuracy_only_agreement_gives_capa_zero():
    # a always right; b right on half -> agreement is exactly what accuracies imply
    gold = [True, True, True, True]
    s = _stats(_rows(gold, [True, True, False, False], gold), "a", "b")
    assert s["capa"] == pytest.approx(0.0)


def test_invalid_never_agrees_and_counts_wrong():
    gold = [True, False]
    s = _stats(_rows([INVALID, INVALID], [INVALID, INVALID], gold), "a", "b")
    assert s["n"] == 2 and s["acc_a"] == 0 and s["ec"] == 1.0 and s["c_obs"] == 0.0


def test_undefined_capa_is_none():
    gold = [True, False]
    s = _stats(_rows(gold, gold, gold), "a", "b")  # both perfect: c_exp = 1
    assert s["capa"] is None


def test_frozen_fixture_has_one_judge_pair_and_no_role_pairs():
    t = affinity_table(load_rows(), resamples=200)
    assert t["rows"] == 6 and t["judges"] == ["llm", "typed"] and t["generating_roles"] == []
    assert t["items_per_actor"] == {"llm": 18, "typed": 18}
    assert t["invalid_per_actor"] == {"llm": 15, "typed": 0}
    assert len(t["pairs"]) == 1 and t["pairs"][0]["n"] == 18
