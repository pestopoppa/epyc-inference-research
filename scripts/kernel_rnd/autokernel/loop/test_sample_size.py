"""AK-EVAL-NPLAN: pre-registered pair counts from stored floor rows. Pure; no launches."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from . import bench, loop, run, sample_size, serving

#: A synthetic A/A calibration table in floor_rows' shape {k: p95 floor %}.
ROWS = {1: 3.6, 5: 1.6, 9: 1.2, 16: 0.9}


def _floor_at(n):
    """run.noise_floor_pct's rule over ROWS: max(sigma1/sqrt(n), measured row <= n)."""
    return max(ROWS[1] / n ** 0.5, ROWS[max(k for k in ROWS if k <= n)])


def _hypothesis(fraction=None):
    return loop.Hypothesis("h", "s", "f", "pp512", "x",
                           expected_outcome=None if fraction is None else
                           {"direction": "improve", "min_effect_fraction": fraction})


@pytest.mark.parametrize("fraction,pairs,rule", [
    (None, 5, "n_min_no_declared_effect"),
    (0.03, 5, "smallest_n_floor_below_target"),
    (0.0135, 9, "smallest_n_floor_below_target"),
    (0.0100, 16, "smallest_n_floor_below_target"),
    (0.0050, 20, "n_max_cap_target_below_floor"),
])
def test_pairs_chosen_from_the_calibration_rows_and_target_effect(fraction, pairs, rule):
    plan = sample_size.plan_pairs(_floor_at, sample_size.target_effect_pct(_hypothesis(fraction)))
    assert (plan["pairs"], plan["rule"]) == (pairs, rule)
    assert plan["fixed_before_measurement"] is True


def test_uncalibrated_surface_refuses():
    with pytest.raises(sample_size.Uncalibrated):
        sample_size.plan_pairs(lambda n: None, 2.0)


def test_planned_n_is_still_a_predeclared_matched_plan():
    plan = sample_size.plan_pairs(_floor_at, 1.0)
    matched = serving._matched_plan(plan["pairs"])
    row = {"schema": "epyc.autokernel.serving_ab.v2", "pairs": plan["pairs"], "measurement_plan": matched}
    assert serving.comparison_instrument_matches(row, instrument=serving.MATCHED_INSTRUMENT,
                                                 pairs=plan["pairs"])
    assert len(matched["orders"]) == plan["pairs"] and matched["stopping"] == "fixed_pairs"


def test_run_plans_once_before_measuring_and_n_never_moves():
    args = SimpleNamespace(plan_pairs="on", plan_pairs_max=20, pairs=5, surface="pp512",
                           model=Path("/m.gguf"), store=Path("/s"), bench_kv_type=None)
    calls = []

    def floor(surface, n, model, store=None, kv_type=None):
        calls.append(n)
        return _floor_at(n)
    with mock.patch.object(run, "noise_floor_pct", side_effect=floor):
        pairs, bar, basis = run._planned_bench_pairs(args, _hypothesis(0.0135), 1.6)
    assert (pairs, bar) == (9, _floor_at(9)) and basis["pairs"] == 9
    seen = []

    def run_once(binary, model, **_):
        seen.append(str(binary))
        return 100.0, {"resident": True, "peak_vram_bytes": 1 << 31, "peak_kfd_processes": 1}
    with mock.patch.object(bench, "run_once", side_effect=run_once):
        out = bench.compare(bench.Arm("a", Path("/a")), bench.Arm("c", Path("/c")), Path("/m"),
                            pp=512, tg=0, pairs=pairs, noise_floor_pct=bar, warmup_pairs=0,
                            pairs_basis=basis)
    assert len(seen) == 2 * 9 and out.to_dict()["pairs_basis"]["pairs"] == 9
    args.plan_pairs = "off"
    assert run._planned_bench_pairs(args, _hypothesis(0.0135), 1.6) == (5, 1.6, None)
