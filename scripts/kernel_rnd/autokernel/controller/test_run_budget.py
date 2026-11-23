"""AK-CONT-BUDGET: run ceilings hold from the run's own ledger, offline."""
import json

import pytest

from autokernel.controller import run_budget as rb

T0 = 1_800_000_000


def _row(wall=10.0, tokens=None, lane="lane-0", backend="opencode"):
    totals = None if tokens is None else {"totals": {"decoded_tokens": tokens}}
    return {"schema": rb.METRICS_SCHEMA, "role": "planner", "backend_kind": backend, "lane": lane,
            "wall_s": wall, "opencode": totals,
            "ts": "2030-01-01T00:00:00Z"}


def _budget(tmp_path, rows=(), **ceilings):
    log = tmp_path / "actor-replies"
    log.mkdir(parents=True)
    (log / "actor-calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    clock = [float(T0)]
    return rb.RunBudget(rb.Ceilings(**ceilings), tmp_path, started_at=T0, clock=lambda: clock[0], monotonic=lambda: clock[0]), clock


def test_inert_without_ceilings(tmp_path):
    budget, _ = _budget(tmp_path, [_row()])
    assert budget.tripped is False


def test_metered_tokens_debit_lane_and_trip_at_the_cap(tmp_path):
    budget, _ = _budget(tmp_path, [_row(tokens=600), _row(tokens=500, lane="lane-1")], decoded_tokens=1000)
    assert budget.check(force=True).startswith("decoded_tokens 1100")
    snapshot = budget.snapshot()
    assert snapshot["lanes"]["lane-0"]["decoded_tokens"] == 600
    assert [call["decoded_tokens"] for call in snapshot["calls"]] == [600, 500]


@pytest.mark.parametrize("row", [_row(tokens=None, backend="codex"), _row(tokens=0, backend="local")])
def test_unmetered_or_zero_token_seat_fails_the_token_cap_closed_and_debits_wall(tmp_path, row):
    budget, _ = _budget(tmp_path, [row], decoded_tokens=10**9)
    assert "unknown" in budget.check(force=True)
    assert budget.snapshot()["lanes"]["lane-0"]["actor_wall_s"] == row["wall_s"]


def test_unmetered_seat_is_fine_without_a_token_cap(tmp_path):
    budget, _ = _budget(tmp_path, [_row()], gpu_s=100.0)
    assert budget.check(force=True) is None


def test_gpu_and_wall_ceilings_latch(tmp_path):
    budget, clock = _budget(tmp_path, gpu_s=100.0, wall_s=1000.0)
    token = budget.begin_arm("gpu")
    clock[0] += 60.0
    budget.debit_stage("benchmark", 60.0, gpu=True, lane="a")
    assert budget.check(force=True) is None
    clock[0] += 40.0
    assert budget.check(force=True).startswith("gpu_s")
    budget.end_arm(token)
    assert budget.tripped is True

    other, clock = _budget(tmp_path / "x", wall_s=1000.0)
    clock[0] += 1000.0
    assert other.check(force=True).startswith("wall_s")


def test_rows_from_before_the_run_and_a_torn_line_are_not_debited(tmp_path):
    early = dict(_row(tokens=99), ts="2000-01-01T00:00:00Z")
    budget, _ = _budget(tmp_path, [early, _row(tokens=7)], decoded_tokens=1000)
    with (tmp_path / "actor-replies" / "actor-calls.jsonl").open("a") as stream:
        stream.write('{"schema": "torn')
    assert budget.check(force=True) is None
    assert budget.snapshot()["decoded_tokens"] == 7


def test_cpu_arm_budget_excludes_yielded_time_and_counts_idle(tmp_path):
    budget, clock = _budget(tmp_path, cpu_s=10.0)
    token = budget.begin_arm("cpu")
    clock[0] += 4
    budget.pause_arm(token)
    clock[0] += 100
    assert budget.check(force=True) is None
    budget.resume_arm(token)
    clock[0] += 6
    assert budget.check(force=True).startswith("cpu_s")
    budget.end_arm(token)
    assert budget.snapshot()["arm_seconds_claimed"]["cpu"] == 10


def test_overlapping_arm_claims_sum_and_stages_do_not_double_charge(tmp_path):
    budget, clock = _budget(tmp_path, gpu_s=100)
    first = budget.begin_arm("gpu")
    clock[0] += 3
    second = budget.begin_arm("gpu")
    clock[0] += 4
    budget.debit_stage("nested", 100, gpu=True)
    budget.end_arm(first)
    budget.end_arm(second)
    snapshot = budget.snapshot()
    assert snapshot["arm_seconds_claimed"]["gpu"] == 11
    assert snapshot["arm_seconds_budget"]["surfaces"]["gpu"]["spent"] == 11


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_arm_ceiling_requires_finite_positive_seconds(value):
    with pytest.raises(ValueError, match='finite and positive'):
        rb.Ceilings(cpu_s=value)
