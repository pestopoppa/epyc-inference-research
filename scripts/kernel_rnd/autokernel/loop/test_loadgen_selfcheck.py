"""AKR-SG-1 acceptance: an injected 50 ms scheduler delay voids the run."""
import pytest

from . import loadgen_selfcheck as lg


def test_injected_50ms_scheduler_delay_voids_run():
    _, log = lg.run_open_loop(lambda i: i, 12, 50.0, inject_delay_s=0.05)
    rep = log.report()
    assert rep["voided"] is True
    assert rep["scheduling_delay_ms"]["median_ms"] > 10.0
    with pytest.raises(lg.LoadDriverVoided):
        log.require_valid()


def test_clean_run_is_not_voided_and_states_rates():
    _, log = lg.run_open_loop(lambda i: i, 12, 50.0)
    rep = log.require_valid(requested_n=12)
    assert rep["voided"] is False and rep["void_reasons"] == []
    assert rep["requested_rate_hz"] == 50.0
    assert abs(rep["achieved_rate_hz"] - 50.0) / 50.0 <= 0.05


def test_rate_departure_voids_even_with_low_delay():
    log = lg.SchedulingLog(requested_rate_hz=10.0)
    for k in range(6):                       # intended 10/s, actually 5/s, zero delay
        log.record(k * 0.1, k * 0.2)
    rep = log.report()
    assert rep["voided"] and any("departs" in r for r in rep["void_reasons"])


def test_short_start_count_voids():
    log = lg.SchedulingLog()
    log.record(0.0, 0.0)
    assert log.report(requested_n=4)["voided"]


def test_headline_is_over_all_sessions_started():
    s = [{"status": "ok", "success": True}, {"status": "failed"},
         {"status": "cancelled"}, {"status": "ok", "success": False}]
    h = lg.session_headline(s)
    assert h["sessions_started"] == 4 and h["solved"] == 1
    assert h["success_rate_over_started"] == 0.25 and h["failed"] == 1 and h["cancelled"] == 1
