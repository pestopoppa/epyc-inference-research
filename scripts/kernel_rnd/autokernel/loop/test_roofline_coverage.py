"""Unit tests for the 2026-10-06 autonomy self-check (`roofline_coverage.py`).

Covers: a covered gap producing no warning, an uncovered gap producing a warning
plus `scope_gap.json`, and prefill-contaminated path-level data being ignored in
favour of the per-node rows -- the three fixture classes the handoff asked for, plus
a Q38FN-like profile (IQ3_S/IQ4_NL MoE expert matmul, low achieved bandwidth, no
admitted route) as the end-to-end verdict.
"""
from __future__ import annotations

import json

import pytest

from . import roofline_coverage as rc

#: A single-expert-sized Q4_K weight: 5120x2304 elements -> 46080 blocks of 256
#: elements/144 bytes -> ~6.33 MB. At 2000us/35 evals this reads at ~116 GB/s,
#: comfortably under a 430 GB/s ceiling -- a realistic "partial headroom" gap,
#: not the physically-impossible multi-TB/s figure a batched (x384-expert) shape
#: would produce from the same wall time.
Q4_K_SHAPE = ([5120, 2304, 1], [2304, 1, 1])
#: Same element count, IQ3_S layout (256 elements/110 bytes): ~4.84 MB, ~88.7 GB/s
#: at the same timing -- ~20% of ceiling, matching the 10-40%-of-ceiling range the
#: 2026-10-06 scope audit measured for these kernels.
IQ3_S_SHAPE = ([5120, 2304, 1], [2304, 1, 1])


def _node(op, src0_type, src0_ne, dst_ne, wall_us, evals, *, idx=1, name=None):
    return {"idx": idx, "op": op, "name": name or f"node_{idx}", "src0_type": src0_type,
            "src0_ne": list(src0_ne), "dst_ne": list(dst_ne), "wall_us": wall_us,
            "evals": evals}


# ---- tensor_bytes / node_achieved_bandwidth -------------------------------------

def test_tensor_bytes_q4_k_matches_known_block_size():
    # QK_K=256 elements per 144-byte block; one full block.
    assert rc.tensor_bytes("q4_k", [256, 1, 1]) == 144
    # Partial block still costs a whole block (ceil division).
    assert rc.tensor_bytes("q4_k", [1, 1, 1]) == 144


def test_tensor_bytes_unresolved_type_is_none_not_zero():
    assert rc.tensor_bytes("bf16", [4, 4, 1]) is None


def test_node_achieved_bandwidth_needs_positive_evals_and_wall():
    base = _node("MUL_MAT", "q8_0", [8192, 5120, 1], [5120, 2, 1], 100.0, 10)
    assert rc.node_achieved_bandwidth(base) is not None
    assert rc.node_achieved_bandwidth({**base, "evals": 0}) is None
    assert rc.node_achieved_bandwidth({**base, "wall_us": 0}) is None
    assert rc.node_achieved_bandwidth({**base, "src0_type": "bf16"}) is None


# ---- route resolution / coverage --------------------------------------------------

def test_resolve_route_file_symbol_is_explicit_and_case_insensitive():
    assert rc.resolve_route_file_symbol("mul_mat_id", "IQ3_S") == (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp", "DequantizerIQ3S")
    assert rc.resolve_route_file_symbol("MUL_MAT", "bf16") is None


def test_dense_q4_k_dot_route_is_covered_by_the_ad_hoc_admission():
    # Ad-hoc admitted (gates.py ~1725-1736): the DENSE Q4_K/Q5_K dot route's
    # dequantizer bodies, which `cpu_source_routes` alone would miss (it only
    # covers `CPU_SOURCE_ROUTES`'s table-driven routes, not gates.py's inline
    # structural rules).
    file_path, symbol = rc.ROUTE_FILE_SYMBOL[("MUL_MAT", "q4_k")]
    assert rc.route_is_admitted(file_path, symbol) is True
    assert (file_path, symbol) in rc.AD_HOC_ADMITTED_PAIRS


def test_iq3_s_and_repack_routes_are_not_admitted_today():
    # Matches the 2026-10-06 lowbit-scope audit verdict: both OUT of scope.
    iq3 = rc.ROUTE_FILE_SYMBOL[("MUL_MAT_ID", "iq3_s")]
    repack = rc.ROUTE_FILE_SYMBOL[("MUL_MAT_ID", "q6_k")]
    assert rc.route_is_admitted(*iq3) is False
    assert rc.route_is_admitted(*repack) is False


# ---- per_node_roofline / route_coverage_map / uncovered_gap_summary --------------

def test_per_node_roofline_ranks_by_gap_share_and_skips_unresolved_types():
    src0, dst = Q4_K_SHAPE
    nodes = [
        _node("MUL_MAT", "q4_k", src0, dst, 2000.0, 35, idx=1),
        _node("MUL_MAT", "bf16", [100, 100, 1], [100, 1, 1], 1.0, 1, idx=2),
    ]
    rows = rc.per_node_roofline(nodes, total_wall_us=10_000.0, ceiling_gb_s=430.0)
    assert len(rows) == 1  # the bf16 row is unresolved, not a fabricated zero gap
    assert rows[0]["idx"] == 1
    assert 0.0 <= rows[0]["headroom_fraction"] <= 1.0
    assert rows[0]["achieved_gb_s"] < rows[0]["ceiling_gb_s"]
    assert rows[0]["gap_share"] > 0


def test_per_node_roofline_refuses_non_positive_total_wall_or_ceiling():
    nodes = [_node("MUL_MAT", "q8_0", [8192, 5120, 1], [5120, 2, 1], 100.0, 10)]
    with pytest.raises(rc.RooflineRefused):
        rc.per_node_roofline(nodes, total_wall_us=0.0, ceiling_gb_s=430.0)
    with pytest.raises(rc.RooflineRefused):
        rc.per_node_roofline(nodes, total_wall_us=100.0, ceiling_gb_s=0.0)


def test_covered_gap_never_triggers_the_uncovered_warning():
    """A gap sitting entirely in an admitted route: no warning, no scope_gap trigger."""
    src0, dst = Q4_K_SHAPE
    nodes = [_node("MUL_MAT", "q4_k", src0, dst, 2000.0, 35, idx=1)]
    report = rc.from_node_dump({"nodes": nodes, "total_wall_us": 10_000.0},
                               ceiling_gb_s=430.0)
    assert report["gap_table"][0]["gap_share"] > 0  # a real, nonzero gap ...
    assert report["gap_table"][0]["covered"] is True  # ... that IS covered
    assert report["uncovered_gap"]["triggered"] is False
    assert report["uncovered_gap"]["uncovered_gap_share"] == 0.0
    assert report["uncovered_gap"]["uncovered_nodes"] == []


def test_uncovered_gap_triggers_warning_and_names_file_and_symbol():
    """A gap sitting in iq3_s (no admitted route): triggers, names the real file."""
    src0, dst = IQ3_S_SHAPE
    nodes = [_node("MUL_MAT_ID", "iq3_s", src0, dst, 2000.0, 35, idx=1)]
    report = rc.from_node_dump({"nodes": nodes, "total_wall_us": 10_000.0},
                               ceiling_gb_s=430.0)
    row = report["gap_table"][0]
    assert row["gap_share"] > 0
    assert row["covered"] is False
    assert row["file"] == "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp"
    assert row["symbol"] == "DequantizerIQ3S"
    gap = report["uncovered_gap"]
    assert gap["triggered"] is True
    assert gap["uncovered_fraction"] == pytest.approx(1.0)
    assert gap["uncovered_nodes"][0]["file"] == "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp"


def test_unresolved_type_counts_as_uncovered_not_silently_safe():
    """`covered is None` (no route map entry at all) must not read as covered."""
    rows = [{"idx": 1, "op": "MUL_MAT", "name": "n", "src0_type": "bf16",
            "time_share": 0.9, "achieved_gb_s": 10.0, "ceiling_gb_s": 430.0,
            "headroom_fraction": 0.98, "gap_share": 0.88}]
    covered = rc.route_coverage_map(rows)
    assert covered[0]["covered"] is None
    summary = rc.uncovered_gap_summary(covered)
    assert summary["uncovered_gap_share"] == pytest.approx(0.88)
    assert summary["triggered"] is True


# ---- prefill-contaminated path data is ignored -----------------------------------

def test_from_node_dump_ignores_the_paths_aggregate_entirely():
    """`paths`/`weight_paths` is the prefill-pooled aggregate the 2026-10-06 roofline
    audit found contaminated (README: "discard path-level GBs_on_wall"). Swapping it
    out for wildly different, internally-inconsistent numbers must not change the
    per-node report at all -- `from_node_dump` must never read that key."""
    src0, dst = IQ3_S_SHAPE
    nodes = [_node("MUL_MAT_ID", "iq3_s", src0, dst, 2000.0, 35, idx=1)]
    clean = rc.from_node_dump({"nodes": nodes, "total_wall_us": 10_000.0},
                              ceiling_gb_s=430.0)
    contaminated = rc.from_node_dump(
        {"nodes": nodes, "total_wall_us": 10_000.0,
         "paths": [{"path": "expert_mul_mat_id", "calls": 1, "compute_us": 1,
                    "wall_us": 1, "bytes": 999_999_999_999,
                    "GBs_on_compute": 1e9, "GBs_on_wall": 1e9}]},
        ceiling_gb_s=430.0)
    assert clean == contaminated


def test_from_node_dump_requires_positive_total_wall_us():
    with pytest.raises(rc.RooflineRefused):
        rc.from_node_dump({"nodes": [], "total_wall_us": 0}, ceiling_gb_s=430.0)


# ---- stagnation hook / scope_gap.json ---------------------------------------------

def _stagnant_signal():
    return {"mode": "observe_only", "window": 20, "measured_points": 6,
            "best_effect_slope_per_measurement": -0.001, "consecutive_measured_nulls": 0}


def _progressing_signal():
    return {"mode": "observe_only", "window": 20, "measured_points": 6,
            "best_effect_slope_per_measurement": 0.02, "consecutive_measured_nulls": 0}


def test_is_stagnant_reads_slope_and_null_streak():
    assert rc.is_stagnant(_stagnant_signal()) is True
    assert rc.is_stagnant(_progressing_signal()) is False
    assert rc.is_stagnant({"best_effect_slope_per_measurement": None,
                           "consecutive_measured_nulls": 5}) is True
    assert rc.is_stagnant(None) is False


def test_stagnation_hook_writes_scope_gap_only_when_both_hold(tmp_path):
    src0, dst = IQ3_S_SHAPE
    nodes = [_node("MUL_MAT_ID", "iq3_s", src0, dst, 2000.0, 35, idx=1)]
    report = rc.from_node_dump({"nodes": nodes, "total_wall_us": 10_000.0},
                               ceiling_gb_s=430.0)
    uncovered = report["uncovered_gap"]
    assert uncovered["triggered"] is True

    # Progressing: no write even though the gap is uncovered.
    assert rc.stagnation_scope_gap_hook(uncovered, _progressing_signal(),
                                        store_root=tmp_path) is None
    assert not (tmp_path / rc.SCOPE_GAP_FILENAME).exists()

    # Stagnant AND uncovered: writes the machine-readable scope_gap.json.
    path = rc.stagnation_scope_gap_hook(uncovered, _stagnant_signal(), store_root=tmp_path)
    assert path == tmp_path / rc.SCOPE_GAP_FILENAME
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["schema"] == rc.SCOPE_GAP_SCHEMA
    assert body["triggered"] is True
    assert body["uncovered_nodes"][0]["file"] == "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp"


def test_stagnation_hook_is_a_noop_without_a_triggered_gap(tmp_path):
    src0, dst = Q4_K_SHAPE
    covered_nodes = [_node("MUL_MAT", "q4_k", src0, dst, 2000.0, 35, idx=1)]
    report = rc.from_node_dump({"nodes": covered_nodes, "total_wall_us": 10_000.0},
                               ceiling_gb_s=430.0)
    assert rc.stagnation_scope_gap_hook(report["uncovered_gap"], _stagnant_signal(),
                                        store_root=tmp_path) is None
    assert rc.stagnation_scope_gap_hook(None, _stagnant_signal(),
                                        store_root=tmp_path) is None


def test_write_scope_gap_is_machine_readable_and_idempotent(tmp_path):
    summary = {"schema": rc.SCHEMA, "threshold": 0.3, "total_gap_share": 0.5,
              "uncovered_gap_share": 0.5, "uncovered_fraction": 1.0, "triggered": True,
              "uncovered_nodes": []}
    first = rc.write_scope_gap(tmp_path, summary, now="2026-10-06T00:00:00Z")
    second = rc.write_scope_gap(tmp_path, summary, now="2026-10-06T00:00:01Z")
    assert first == second == tmp_path / rc.SCOPE_GAP_FILENAME
    body = json.loads(second.read_text(encoding="utf-8"))
    assert body["generated_at"] == "2026-10-06T00:00:01Z"
    assert body["triggered"] is True


# ---- Q38FN-like end-to-end verdict ------------------------------------------------

def test_q38fn_like_profile_flags_iq3_s_and_iq4_nl_moe_as_the_uncovered_headroom(tmp_path):
    """Reproduces the shape of the 2026-10-06 incident: IQ3_S/IQ4_NL MoE expert
    matmul running at 10-40% of the host's measured ceiling, with the admitted-route
    table covering only the dense Q4_K/Q5_K dot and MoE dispatch wiring -- never the
    IQ3_S/IQ4_NL dequant bodies themselves."""
    nodes = [
        # IQ3_S expert matmul: ~115 MB/call at ~2ms/call -> ~58 GB/s, 13% of ceiling.
        _node("MUL_MAT_ID", "iq3_s", [256, 4096, 256], [4096, 8, 1], 70_000.0, 35,
             idx=1, name="ffn_moe_iq3s"),
        # IQ4_NL expert matmul: ~19 MB/call at ~0.57ms/call -> ~33 GB/s, 8% of ceiling.
        _node("MUL_MAT_ID", "iq4_nl", [32, 4096, 256], [4096, 8, 1], 20_000.0, 35,
             idx=2, name="ffn_moe_iq4nl"),
        # Dense Q4_K dot IS covered (ad-hoc admission) -- must not show as uncovered.
        _node("MUL_MAT", "q4_k", [5120, 2304, 1], [2304, 1, 1], 5_000.0, 35,
             idx=3, name="attn_q4k_dot"),
    ]
    report = rc.from_node_dump({"nodes": nodes, "total_wall_us": 100_000.0},
                               ceiling_gb_s=430.0)
    gap = report["uncovered_gap"]
    assert gap["triggered"] is True
    uncovered_files = {row["file"] for row in gap["uncovered_nodes"]}
    assert uncovered_files == {
        "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp",
        "ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp",
    }
    covered_names = {row["name"] for row in report["gap_table"] if row["covered"] is True}
    assert covered_names == {"attn_q4k_dot"}
    uncovered_names = {row["name"] for row in report["gap_table"] if row["covered"] is False}
    assert uncovered_names == {"ffn_moe_iq3s", "ffn_moe_iq4nl"}
    # The stagnation hook fires for this shape once the lane has actually stalled.
    path = rc.stagnation_scope_gap_hook(
        gap, {"best_effect_slope_per_measurement": 0.0, "measured_points": 10,
             "consecutive_measured_nulls": 0},
        store_root=tmp_path)
    assert path == tmp_path / rc.SCOPE_GAP_FILENAME
    body = json.loads(path.read_text(encoding="utf-8"))
    assert {row["file"] for row in body["uncovered_nodes"]} == uncovered_files
