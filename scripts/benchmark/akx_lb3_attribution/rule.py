"""The LB3.2 predeclared rule (PREDECLARATION.md in the launcher dir; same S/F shape as the frozen 2026-10-22 rule,
with F = op WALL share on the critical path instead of a CPU-sample share). Pure arithmetic over the two arms'
summaries; nothing here reads a perf.data or starts anything.

    S(N)   = (tG(N, 64k) - tG(N, 512)) / tG(N, 64k)        G arm = champion store image, llama-bench avg_ns
    F(N)   = W_fa(N, 64k) / T_M(N, 64k)                    M arm = measurement build, FLASH_ATTN_EXT node wall
                                                            (thread 0, through the graph barrier) over the timed step
    LANE_LARGE        max_{N=2..5} F(N) >= 10%
    LANE_SMALL        else, max_{N=2..5} S(N) < 10%
    SPLIT_UNRESOLVED  else; the lane is seeded on SEED_CLASS = argmax_c mean_{N=2..5} dW_c(N) / T_M(N, 64k),
                      dW_c(N) = W_c(N, 64k) - W_c(N, 512), over node classes, the solo-run residual and host phases
    INVALID           a step not OK, a (N, d) cell missing in either arm, an M cell at N=2..5 x 64k with < 8 usable
                      repetitions, |tM/tG - 1| > 5% or a coverage defect > 10%, or a truncated eval file
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

NS = (1, 2, 3, 4, 5, 8)
GATE_NS = (2, 3, 4, 5)
D_LO, D_HI = 512, 65536
THRESH = 0.10
MAX_OVERHEAD = 0.05
MAX_COVERAGE_DEFECT = 0.10
MIN_REPS = 8


def cells(rows: Sequence[dict]) -> Dict[Tuple[int, int], dict]:
    """llama-bench jsonl rows (test order) -> {(n_prompt, n_depth): row + its 1-based test index}."""
    out = {}
    for i, r in enumerate(rows, start=1):
        out[(r.get("n_prompt"), r.get("n_depth"))] = dict(r, test_idx=i)
    return out


def class_table(m: dict) -> Dict[str, float]:
    """One M-arm test rollup -> {class: us per step}: node classes, solo residual, host phases split into the
    per-input set_input classes, the rest of set_inputs (graph build/reuse + alloc), and in-graph_compute overhead."""
    t: Dict[str, float] = {}
    for c, v in (m.get("node_wall_us") or {}).items():
        t["node:" + c] = v
    t["node:solo_runs"] = m.get("solo_wall_us") or 0.0
    h = m.get("host_us") or {}
    inputs = {k: v for k, v in h.items() if k.startswith("input:")}
    t.update({"host:" + k: v for k, v in inputs.items()})
    t["host:apply_ubatch"] = h.get("mctx_apply", 0.0)
    t["host:build_alloc"] = h.get("set_inputs", 0.0) - sum(inputs.values())
    t["host:graph_overhead"] = h.get("graph_compute", 0.0) - (m.get("eval_span_us") or 0.0)
    t["host:unattributed"] = m.get("unattributed_us") or 0.0
    return t


def decide(g_rows: Sequence[dict], m_rows: Sequence[dict], m_roll: Dict[int, dict],
           steps_ok: bool = True, eval_truncated: bool = False,
           g_perf: Optional[Dict[int, dict]] = None) -> dict:
    G, M = cells(g_rows), cells(m_rows)
    invalid: List[str] = []
    if not steps_ok:
        invalid.append("a gating step (G or M) is not OK")
    if eval_truncated:
        invalid.append("M eval file truncated")
    res: Dict[int, dict] = {}
    for N in NS:
        cell = {}
        glo, ghi, mlo, mhi = G.get((N, D_LO)), G.get((N, D_HI)), M.get((N, D_LO)), M.get((N, D_HI))
        if not (glo and ghi and mlo and mhi):
            invalid.append(f"N={N}: a (N, d) cell is missing in G or M")
            continue
        tg_lo, tg_hi, tm_lo, tm_hi = glo["avg_ns"], ghi["avg_ns"], mlo["avg_ns"], mhi["avg_ns"]
        rhi, rlo = m_roll.get(mhi["test_idx"], {}), m_roll.get(mlo["test_idx"], {})
        cell["S_depth_share"] = round((tg_hi - tg_lo) / tg_hi, 4)
        cell["S_M_depth_share"] = round((tm_hi - tm_lo) / tm_hi, 4)
        cell["overhead_d512"] = round(tm_lo / tg_lo - 1, 4)
        cell["overhead_d64k"] = round(tm_hi / tg_hi - 1, 4)
        cell["t_ms"] = dict(G_d512=round(tg_lo / 1e6, 3), G_d64k=round(tg_hi / 1e6, 3),
                            M_d512=round(tm_lo / 1e6, 3), M_d64k=round(tm_hi / 1e6, 3))
        cell["F_fa_wall_share"] = (rhi.get("share") or {}).get("fa")
        cell["reps_used_d64k"] = rhi.get("reps_used", 0)
        cell["coverage_defect_d64k"] = rhi.get("coverage_defect")
        if rhi.get("t_step_us") and rlo.get("t_step_us"):
            hi_t, lo_t = class_table(rhi), class_table(rlo)
            T = rhi["t_step_us"]
            cell["class_share_d64k"] = {k: round(v / T, 5) for k, v in sorted(hi_t.items())}
            cell["class_depth_delta_share"] = {k: round((hi_t.get(k, 0.0) - lo_t.get(k, 0.0)) / T, 5)
                                               for k in sorted(set(hi_t) | set(lo_t))}
        if g_perf is not None:
            gp = g_perf.get(ghi["test_idx"], {})
            cell["F_cpu_sample_share_G"] = (gp.get("share") or {}).get("fa")
            cell["G_perf_samples"] = gp.get("samples")
            cell["G_perf_reps_used"] = gp.get("reps_used")
        if N in GATE_NS:
            if cell["reps_used_d64k"] < MIN_REPS:
                invalid.append(f"N={N}: {cell['reps_used_d64k']} usable M repetitions at 64k < {MIN_REPS}")
            if abs(cell["overhead_d64k"]) > MAX_OVERHEAD:
                invalid.append(f"N={N}: measurement-build overhead {cell['overhead_d64k']:+.2%} at 64k exceeds "
                               f"{MAX_OVERHEAD:.0%}")
            cd = cell["coverage_defect_d64k"]
            if cd is None or cd > MAX_COVERAGE_DEFECT:
                invalid.append(f"N={N}: coverage defect {cd} at 64k (> {MAX_COVERAGE_DEFECT:.0%} or unknown)")
            if cell["F_fa_wall_share"] is None:
                invalid.append(f"N={N}: no FA wall share at 64k")
        res[N] = cell
    out = dict(rule="PREDECLARATION.md (LB3.2)", thresholds=dict(F=THRESH, S=THRESH, overhead=MAX_OVERHEAD,
               coverage_defect=MAX_COVERAGE_DEFECT, min_reps=MIN_REPS), cells=res)
    if invalid:
        out.update(verdict="INVALID", why=invalid)
        return out
    Fm = max(res[N]["F_fa_wall_share"] for N in GATE_NS)
    Sm = max(res[N]["S_depth_share"] for N in GATE_NS)
    keys = sorted({k for N in GATE_NS for k in res[N].get("class_depth_delta_share", {})})
    mean_delta = {k: round(sum(res[N]["class_depth_delta_share"].get(k, 0.0) for N in GATE_NS) / len(GATE_NS), 5)
                  for k in keys}
    seed = max(mean_delta, key=lambda k: mean_delta[k]) if mean_delta else None
    out.update(max_F_N2to5=Fm, max_S_N2to5=Sm, mean_depth_delta_share_N2to5=mean_delta, seed_class=seed)
    out["verdict"] = "LANE_LARGE" if Fm >= THRESH else ("LANE_SMALL" if Sm < THRESH else "SPLIT_UNRESOLVED")
    return out
