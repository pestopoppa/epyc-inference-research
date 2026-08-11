from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "run_inf37_iq2_fancy_simd_ab.py"
SPEC = importlib.util.spec_from_file_location("inf37_iq2_ab", PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def sql(rows: list[tuple[str, float, int]]) -> str:
    head = """CREATE TABLE test_backend_ops (
op_params TEXT, supported INTEGER, passed INTEGER, time_us REAL, n_runs INTEGER);
"""
    return head + "".join(
        "INSERT INTO test_backend_ops VALUES "
        f"('type_a=iq2_xxs,type_b=f32,m=4096,n={n},k=14336',1,1,{time_us},{runs});\n"
        for n, time_us, runs in rows)


def test_balanced_orders_are_exact() -> None:
    orders = runner.balanced_orders(10)
    assert orders.count(("baseline", "candidate")) == 5
    assert orders.count(("candidate", "baseline")) == 5
    with pytest.raises(ValueError):
        runner.balanced_orders(3)


def test_parse_sql_rows_requires_exact_cells() -> None:
    rows = runner.parse_sql_rows(sql([(1, 100.0, 10), (512, 200.0, 5)]))
    assert [row["n"] for row in rows] == [1, 512]
    with pytest.raises(RuntimeError, match="expected exact cells"):
        runner.parse_sql_rows(sql([(1, 100.0, 10)]))


def test_summary_uses_lower_time_as_better() -> None:
    invocations = []
    for block in range(4):
        for arm, scale in (("baseline", 1.0), ("candidate", 0.8)):
            invocations.append({
                "block": block, "arm": arm,
                "rows": [
                    {"n": 1, "time_us": 100 * scale},
                    {"n": 512, "time_us": 200 * scale},
                ],
            })
    summary = runner.summarize(invocations, 4)
    assert summary["direction"] == "lower_is_better"
    assert all(row["all_candidate_faster"] for row in summary["cells"])
    assert all(row["median_candidate_speedup_fraction"] == pytest.approx(0.25)
               for row in summary["cells"])
