"""AK-PROF-TWO-DENOM: wall share, launch gap, count validity."""
from __future__ import annotations

from . import hotspots as hs

HDR = "Kernel_Name,Start_Timestamp,End_Timestamp\n"


def _csv(spec):  # [(name, dur, n)]
    rows, t = [], 0
    for name, dur, n in spec:
        for _ in range(n):
            rows.append(f"{name},{t},{t + dur}")
            t += dur + 1
    return HDR + "\n".join(rows) + "\n"


LONG = _csv([("gated_delta_net_k", 100, 70), ("mul_mat_q", 300, 70), ("rms", 10, 10)])
SHORT = _csv([("gated_delta_net_k", 100, 10), ("mul_mat_q", 300, 10), ("rms", 10, 10)])


def test_unit_work_cancels_shared_cost():
    w = hs.unit_work(LONG, SHORT, 2)
    assert w["gated_delta_net_k"] == (3000.0, 30.0)
    assert "rms" not in w          # identical in both arms -> cancels


def test_two_denominators_and_gap():
    w = hs.unit_work(LONG, SHORT, 2)           # kernel sum 12000
    r = hs.two_denominator(w, wall_ns=15000, expected_calls={"gated_delta_net": (30, 36)})
    assert r["launch_gap_ns"] == 3000 and abs(r["launch_gap_frac"] - 0.2) < 1e-9
    top = {h.signature: h for h in r["hotspots"]}["gated_delta_net_k"]
    assert abs(top.share_of_device_time - 0.25) < 1e-9
    assert abs(top.share_of_wall - 0.2) < 1e-9
    assert r["count_ok"] is True and "launch_gap 20.0%" in hs.render_two_denominator(r)


def test_no_wall_is_absent_not_fabricated():
    r = hs.two_denominator(hs.unit_work(LONG, SHORT, 2), wall_ns=None)
    assert r["wall_ns"] is None and r["launch_gap_ns"] is None and r["launch_gap_frac"] is None
    assert r["wall_valid"] is None and r["count_ok"] is None
    assert all(h.share_of_wall is None and "share_of_wall" not in h.to_dict() for h in r["hotspots"])
    assert "n/a" in hs.render_two_denominator(r)
    assert hs.two_denominator({}, wall_ns=0)["launch_gap_ns"] is None


def test_count_validity_negative_controls():
    w = hs.unit_work(LONG, SHORT, 2)
    bad = hs.two_denominator(w, wall_ns=15000, expected_calls={"gated_delta_net": (12,)})
    assert bad["count_ok"] is False and "FAILED" in hs.render_two_denominator(bad)
    missing = hs.two_denominator(w, wall_ns=15000, expected_calls={"no_such_kernel": (1,)})
    assert missing["count_ok"] is False          # a check on a kernel that never ran fails


def test_kernel_sum_over_wall_is_flagged_not_negative_gap():
    r = hs.two_denominator(hs.unit_work(LONG, SHORT, 2), wall_ns=5000)
    assert r["wall_valid"] is False and r["launch_gap_ns"] is None
    assert "WARNING" in hs.render_two_denominator(r)


def test_legacy_parse_unchanged_and_dict_stable():
    rows = hs.parse_kernel_trace(LONG)
    assert rows[0].share_of_wall is None
    assert set(rows[0].to_dict()) == {"signature", "total_duration_ns", "calls", "share_of_device_time"}
