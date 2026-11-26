"""AUDIT D1-D3 of AK-BNSIG-CENSUS-KERNEL-KEYS: time-only signatures reach the planner, survive refreshes, and the
signature ledger converges live <- committed through the ak_runend --merge-static-from flow."""
import json

from autokernel import bnsig_planner as bp
from autokernel.loop import census_coverage, census_update as cu
from autokernel.loop.test_census_kernel_keys import _trace
from autokernel.loop.census_extract import extract_trace_kernels

SIG = {"class": "latency_bound", "classes": ["latency_bound"], "ruled_out": ["over_fetch"], "rule": "x"}
GRIDS = {"rms_norm_f32ILi1024ELb1E": 8192, "unary_gated_op_kernelIXadL_ZL7op_silufEEf": 139264}


def _census():
    c = cu.empty_census()
    cu.apply(c, extract_trace_kernels(_trace()), "r", "a" * 64, "2026-11-23T00:00:00Z")
    cu.apply_signature(c, dict.fromkeys(GRIDS, SIG), "s", "b" * 64, "2026-11-25T00:00:00Z", "bld", grids=GRIDS)
    return c


def test_time_only_signature_reaches_planner_view_and_render():
    v = bp.planner_view(_census())
    assert v["status"] == "observed" and [h["share_of_round"] for h in v["hotspots"]] == [0.075, 0.027]
    assert any("rms_norm_f32<1024" in ln for ln in bp.render(v))
    assert bp.planner_view(cu.empty_census()) == {"status": "off"}  # zero signed: prompt stays byte-identical


def test_signature_survives_time_only_refresh_and_flags_stale():
    c = _census()
    o = dict(extract_trace_kernels(_trace())[0], us_per_call=60.0)
    assert cu.apply_time_only(c, o, "r2", "c" * 64, "t")
    sig = c["time_only"][cu.rid(o)]["signature"]
    assert sig["result_sha256"] == "b" * 64 and sig["stale_vs_timing"] is True


def test_merge_static_from_unions_signature_ledger_end_to_end(tmp_path, monkeypatch):
    live = cu.empty_census()
    cu.apply(live, extract_trace_kernels(_trace()), "r", "a" * 64, "2026-11-23T00:00:00Z")
    live["signature_applied"] = [{"row": "old", "sha256": "d" * 64}]
    lp, cp = tmp_path / "live.json", tmp_path / "committed.json"
    lp.write_text(json.dumps(live)), cp.write_text(json.dumps(_census()))
    monkeypatch.setattr(census_coverage, "load_registry", lambda *a, **k: {})
    rep = cu.run(None, None, str(lp), None, None, None, merge_static_from=str(cp))
    got = json.loads(lp.read_text())
    assert "signature_applied" in rep["merge_static"]
    assert [a["sha256"][0] for a in got["signature_applied"]] == ["d", "b"]
    assert sum(bool(r.get("signature")) for r in got["time_only"].values()) == 2
    again = cu.run(None, None, str(lp), None, None, None, merge_static_from=str(cp))
    assert "signature_applied" not in (again["merge_static"] or [])


def test_copied_signature_is_stale_when_live_timing_moved():
    """Fable probe B: live refreshed the kernel's timing; a committed signature copied onto it must say stale."""
    sig = {"class": "issue_bound", "measured_at": "2026-11-25T22:21:03Z", "timing_sha256": "old", "stale_vs_timing": False}
    committed = {"time_only": {"k": {"result_sha256": "old", "signature": sig}}, "signature_applied": [{"sha256": "s"}]}
    live = {"time_only": {"k": {"result_sha256": "new", "measured_at": "2026-11-26T09:00:00Z"}}}
    assert bp.union_signature_applied(live, committed)
    assert live["time_only"]["k"]["signature"]["stale_vs_timing"] is True
    live2 = {"time_only": {"k": {"result_sha256": "old"}}}
    bp.union_signature_applied(live2, committed)
    assert live2["time_only"]["k"]["signature"]["stale_vs_timing"] is False
    assert sig["stale_vs_timing"] is False   # the committed copy is never mutated
