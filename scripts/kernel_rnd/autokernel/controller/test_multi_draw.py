"""R24-10 multi-draw arithmetic and aggregation (pure Python)."""
from __future__ import annotations

import pytest

from autokernel.controller import multi_draw as md


def test_three_draw_oracle_misses_p015_about_956_percent():
    assert md.miss_probability(0.015, 3) == pytest.approx(0.9557, abs=1e-3)


def test_draws_needed_for_p015_at_95():
    n = md.draws_needed(0.015)
    assert md.miss_probability(0.015, n) <= 0.05 < md.miss_probability(0.015, n - 1)
    assert n <= md.DEFAULT_DRAWS * 2 and n > 3


def test_seed_schedule_deterministic_and_distinct():
    a, b = md.seed_schedule(7, 100), md.seed_schedule(7, 100)
    assert a == b and len(set(a)) == 100
    assert md.seed_schedule(8, 5) != a[:5]


def test_clean_run_passes_and_bounds_rate():
    rec = md.run_draws("ARGMAX", "M=33", lambda s: True, base_seed=1)
    assert rec["passes"] and rec["mismatches"] == 0
    assert rec["rate_upper_95"] == pytest.approx(0.0295, abs=1e-3)


def test_rare_failure_is_found_with_100_draws():
    bad = set(md.seed_schedule(3, 100)[17:19])
    rec = md.run_draws("TOP_K", "M=127", lambda s: s not in bad, base_seed=3)
    assert not rec["passes"] and rec["mismatches"] == 2
    assert rec["mismatch_rate"] == 0.02 and set(rec["mismatch_seeds"]) == bad


def test_refused_draw_is_never_a_pass():
    def boom(seed):
        raise RuntimeError("tool crashed")
    rec = md.run_draws("ARGMAX", "M=1", boom, base_seed=1, draws=10)
    assert not rec["passes"] and rec["refused"] == 10 and rec["mismatch_rate"] is None
    assert not md.run_draws("ARGMAX", "M=1", lambda s: None, base_seed=1, draws=5)["passes"]


def test_unregistered_op_refused_and_summary():
    with pytest.raises(ValueError):
        md.run_draws("MUL_MAT", "x", lambda s: True, base_seed=1)
    ok = md.run_draws("ARGMAX", "a", lambda s: True, base_seed=1, draws=5)
    bad = md.run_draws("ARGMAX", "b", lambda s: False, base_seed=1, draws=5)
    summary = md.summarize([ok, bad])
    assert not summary["passes"] and summary["failed_cases"] == ["b"]
    assert not md.summarize([])["passes"]
