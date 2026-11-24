"""census_extract.py -- PURE RESULT -> census observation extractors (split out of census_update.py, 2026-11-23).

census_update.py owns the census store (apply/rerank/seeds/doc/journal); this module owns only the parsing of a
filed RESULT.json into observations (dicts of model/kernel_class/shape/n/bytes_per_call/us_per_call). No census state.
"""
from __future__ import annotations

import json
import os
import re

from .census_coverage import FALLBACK_N, served_n

SERVED_N = set(served_n())  # LEGACY import-time union (seed-linking default); extraction takes the per-run per-path census_coverage.n_report
Q8_SHAPE_CLASS = {  # 27B Qwen3.8 Q8_0 tensor shapes -> (class name, calls per verify forward)
    "17408x5120": ("ffn_gate_up", 128), "5120x17408": ("ffn_down", 64), "10240x5120": ("gdn_attn_qkv", 48),
    "6144x5120": ("gdn_attn_gate", 48), "5120x6144": ("ssm_out+attn_output", 64), "12288x5120": ("attn_q", 16),
}
# Census model keys by RESULT id substring / RESULT `model_key` (alias: the Q38FN row writes a shorter key than the census).
CPU_MODEL_BY_ID = (("a3b", "a3b-q8_0-cpu"), ("q38fn", "q38fn-ud-iq4xs-cpu"))
CPU_MODEL_ALIAS = {"q38fn-iq4xs-cpu": "q38fn-ud-iq4xs-cpu", "a3b-q8_0-cpu": "a3b-q8_0-cpu"}
TRACE_DENSE_BYTES = 2.04e9  # A3B dense Q8_0 bytes per verify forward (C14 census seed constant)
TRACE_Q8_CLASS = {"1024x5120": ("attn_kv_v5i", 32), "248320x5120": ("lm_head_v5i", 1)}
# Static bytes per decode step (n=1 row) per family, Qwen3.6-35B-A3B-MTP-Q8_0.gguf tensor table (Q8_0 = 34 B/32 w).
A3B_CPU_FAMILY_BYTES = {
    "MUL_MAT:gdn_proj": 1_069_000_000,          # 30 x (attn_qkv + attn_gate + ssm_out)
    "MUL_MAT:lm_head": 540_000_000,             # output [2048, 248320]
    "MUL_MAT_ID:exps_gate_up": 713_000_000,     # 40 x 8 experts x 2 x 2048x512
    "MUL_MAT_ID:exps_down": 357_000_000,        # 40 x 8 experts x 512x2048
    "MUL_MAT:fullattn_proj": 300_000_000,       # 10 x (q + o + k/v), approx
    "MUL_MAT:shared_expert": 138_000_000,       # 41 x 3 x 2048x512
}
MMID_CLASS = {"gate_up": ("mmid_gate_up", 40), "down": ("mmid_down", 40)}  # A3B, 40 MoE layers
GDN_CELL = re.compile(r"sv(\d+):ns(\d+):nt(\d+):K(\d+):(bf16|f32)")
# CPU standalone-op harness (quarter, one op x one layer, DRAM-streaming pools): (class suffix, calls per round).
# calls: 40 MoE layers (MMID_CLASS), 41 shared-expert calls (A3B_CPU_FAMILY_BYTES). Class names carry `@ubench` so they
# never collide with the in-graph per-step census rows (different time basis: per layer call vs per decode step).
UBENCH_CLASS = {("mmid", "base"): ("MUL_MAT_ID:exps_gate_up@ubench", 40), ("shexp", "base"): ("MUL_MAT:shared_expert@ubench", 41),
                ("shexp", "down"): ("MUL_MAT:shexp_down@ubench", 41)}



def _n_ok(model: str, n: int, rep: dict | None) -> bool:
    """n admitted for this census model: `rep` = census_coverage.n_report() (per-path, per run); model absent from it ->
    FALLBACK_N (fail-open, journaled by the caller). No `rep` -> the import-time SERVED_N (legacy callers)."""
    if rep is None:
        return n in SERVED_N
    return n in (rep.get("by_model") or {}).get(model, FALLBACK_N)


def extract(res: dict, rep: dict | None = None) -> list[dict]:
    """Pure: RESULT dict -> census observations (no I/O). `rep` = census_coverage.n_report() (per-path n, per run)."""
    out = []
    op = res.get("op") or {}
    cells = op.get("cells_us") or {}
    for key in sorted(cells):
        m = re.fullmatch(r"(\d+)x(\d+):n(\d+)", key)
        arms = cells[key]
        if not m or not _n_ok("27b-q8_0", int(m.group(3)), rep) or not isinstance(arms, dict):
            continue
        arm = next((a for a in ("V", "ON", "N", "OFF") if isinstance(arms.get(a), (int, float))), None)
        if arm is None:
            continue
        shape = f"{m.group(1)}x{m.group(2)}"
        cls, calls = Q8_SHAPE_CLASS.get(shape, (f"dense_q8_{shape}", 1))
        us = float(arms[arm])
        b = int(m.group(1)) * int(m.group(2)) * 34 // 32
        out.append({"model": "27b-q8_0", "kernel_class": cls + ("_v5i" if arm == "V" else ""), "shape": shape,
                    "n": int(m.group(3)), "arm": arm, "bytes_per_call": b, "us_per_call": us,
                    "calls_per_round": calls})
    table = op.get("table") or {}
    for key in sorted(table):
        m = re.fullmatch(r"([a-z_]+):n(\d+)", key)
        row = table[key]
        if not m or not _n_ok("a3b-q8_0", int(m.group(2)), rep) or not isinstance(row, dict):
            continue
        us, g = row.get("us_on"), row.get("gbps_on")
        if not isinstance(us, (int, float)) or not isinstance(g, (int, float)):
            continue
        cls, calls = MMID_CLASS.get(m.group(1), ("mmid_" + m.group(1), 1))
        out.append({"model": "a3b-q8_0", "kernel_class": cls, "shape": m.group(1), "n": int(m.group(2)), "arm": "ON",
                    "bytes_per_call": int(round(g * us * 1e3)), "us_per_call": float(us), "calls_per_round": calls})
    return out + extract_trace(res, rep) + extract_time_only(res) + extract_vattn_ipb(res)


def extract_trace(res: dict, rep: dict | None = None) -> list[dict]:
    """Pure: GPU *trace* RESULT (`summary` block, no `op`) -> observations. Only values the RESULT states directly:
    - `summary.tbo_gbps["<M>x<K>:n<N>"]` = tbo Q8_0 MUL_MAT GB/s (27B V5I rows); bytes M*K*34/32, arm V.
    - `summary.dense_OFF|dense_ON.apparent_gbps_mean` = A3B dense verify aggregate at n=4 over TRACE_DENSE_BYTES.
    Grid-derived per-kernel byte estimates (mmvq rows, mmid unique experts, gv_fattn KV) are NOT re-derived here:
    their byte basis is hand-calibrated per RESULT, so those stay a manual fold. Pure `op` RESULTs are untouched."""
    out, s = [], res.get("summary")
    if not isinstance(s, dict) or isinstance(res.get("op"), dict):
        return out
    tbo = s.get("tbo_gbps")
    for key in sorted(tbo) if isinstance(tbo, dict) else []:
        m = re.fullmatch(r"(\d+)x(\d+):n(\d+)", key)
        g = tbo[key]
        if not m or not _n_ok("27b-q8_0", int(m.group(3)), rep) or not isinstance(g, (int, float)) or g <= 0:
            continue
        shape = f"{m.group(1)}x{m.group(2)}"
        cls, calls = TRACE_Q8_CLASS.get(shape, (f"dense_q8_{shape}_v5i", 1))
        b = int(m.group(1)) * int(m.group(2)) * 34 // 32
        out.append({"model": "27b-q8_0", "kernel_class": cls, "shape": shape, "n": int(m.group(3)), "arm": "V",
                    "bytes_per_call": b, "us_per_call": round(b / (g * 1e3), 3), "calls_per_round": calls,
                    "bytes_basis": "tbo MUL_MAT q8_0 V5I MF_INPLACE=2 (trace RESULT summary.tbo_gbps)"})
    for arm, cls in (("dense_OFF", "dense_verify_aggregate"), ("dense_ON", "dense_verify_aggregate_v5i")):
        g = (s.get(arm) or {}).get("apparent_gbps_mean") if isinstance(s.get(arm), dict) else None
        if isinstance(g, (int, float)) and g > 0:
            out.append({"model": "a3b-q8_0", "kernel_class": cls, "shape": "2.04GB", "n": 4,
                        "arm": arm[6:], "bytes_per_call": int(TRACE_DENSE_BYTES),
                        "us_per_call": round(TRACE_DENSE_BYTES / (g * 1e3), 3), "calls_per_round": 1,
                        "bytes_basis": "seed constant 2.04e9 B dense Q8_0 per forward (C14 census) / trace RESULT"})
    return out


def cpu_config(result_path: str | None, cfg: str) -> str:
    """Thread/quarter config from the row's server argv (out/servers/<cfg>-L1.argv.json); 'unknown' if absent."""
    if not result_path:
        return "unknown"
    p = os.path.join(os.path.dirname(os.path.abspath(result_path)), "servers", f"{cfg}-L1.argv.json")
    try:
        a = json.load(open(p))
    except (OSError, ValueError):
        return "unknown"
    a = [str(x) for x in (a.get("argv", []) if isinstance(a, dict) else a)]
    t = int(a[a.index("-t") + 1]) if "-t" in a else 0
    # 96 threads = the whole host (NUMA_FULL rows, e.g. ak-c21-census-refresh-cpu): census ceiling key "full-host"
    # 72 threads on q0-q2 (cores 0-71, --interleave=0,1,2): ceiling key "q0q2" = 3 x quarter
    return {24: "quarter-24t", 48: "half-48t", 72: "q0q2", 96: "full-host"}.get(t, f"{t}t")


def extract_cpu(res: dict, result_path: str | None = None) -> list[dict]:
    """CPU gap-census rows: configs[<cfg>].ranked_families at one row per step (plain decode only; MTP steps carry
    a row-dependent byte count and are skipped, deterministically)."""
    out = []
    rid_ = str(res.get("id") or res.get("row") or "")
    if isinstance(res.get("table"), dict) and res.get("quarter"):
        return extract_ubench(res)
    if isinstance(res.get("per_family_by_unit"), dict):
        return extract_cpu_diag(res, result_path)
    model = CPU_MODEL_ALIAS.get(str(res.get("model_key") or ""))
    model = model or next((m for tag, m in CPU_MODEL_BY_ID if tag in rid_), None)
    if model is None:  # unknown model: no bytes basis, so no record (never guess a census key)
        return out
    for cfg, c in sorted((res.get("configs") or {}).items()):
        if not isinstance(c, dict) or c.get("mean_verify_rows") != 1.0:
            continue
        conf = cpu_config(result_path, cfg)
        for f in c.get("ranked_families") or []:
            b = f.get("bytes_per_step")  # the row's own GGUF byte table wins (Q38FN: bytes_per_step per family)
            if not isinstance(b, (int, float)) or b <= 0:
                b = A3B_CPU_FAMILY_BYTES.get(f.get("family")) if model == "a3b-q8_0-cpu" else None
            ms = f.get("ms_per_timing_step")
            if b is None or not isinstance(ms, (int, float)) or ms <= 0:
                continue
            dm = re.search(r"-(d\d+k)$", cfg)  # depth variants share one quarter config: shape quarter-24t-d1k
            out.append({"device": "cpu", "model": model, "kernel_class": f["family"],
                        "shape": conf + (f"-{dm.group(1)}" if dm else ""), "n": 1, "arm": cfg, "bytes_per_call": int(b), "us_per_call": round(ms * 1e3, 1),
                        "calls_per_round": 1, "share_of_round": round(float(f.get("share_pct", 0)) / 100, 4),
                        "config": conf})
    return out


def _idx_share(result_path: str | None, unit: str) -> float | None:
    """Weight share of the indexer family's node wall time from the unit's own prof.json: bf16 MUL_MAT nodes = the two indexer
    weight GEMVs, f32 src0 with ne0 == 128 = the pooled-K score GEMM. None when either side is absent (never guessed)."""
    try:
        nodes = json.load(open(os.path.join(os.path.dirname(result_path or ""), "servers", unit + ".prof.json")))["nodes"]
    except (OSError, ValueError, KeyError):
        return None
    mm = [n for n in nodes if n.get("op") == "MUL_MAT"]
    w = sum(n["wall_us"] for n in mm if n["src0_type"] == "bf16")
    s = sum(n["wall_us"] for n in mm if n["src0_type"] == "f32" and n["src0_ne"][0] == 128)
    return w / (w + s) if w > 0 and s > 0 else None


def extract_cpu_diag(res: dict, result_path: str | None = None) -> list[dict]:
    """Guarded quiet-host CPU diagnostic RESULT (`per_family_by_unit`, per-unit `guard.quiet`) -> quiet_host observations.
    Only the depth(s) the RESULT names as its subject (`evidence.gemv_gbps_<depth>_mean`) are emitted; replicate units
    (`<cfg>-L1`, `-L2`) of one config are averaged on ms/step; a unit whose own guard is not quiet is dropped."""
    rid_, units = str(res.get("id") or ""), res.get("units") or {}
    model = next((m for tag, m in CPU_MODEL_BY_ID if tag in rid_), None)
    subject = {m.group(1) for k in (res.get("evidence") or {}) if (m := re.fullmatch(r"gemv_gbps_(d\d+k)_mean", k))}
    if model is None or res.get("status") != "complete" or not subject:
        return []
    out = []
    for fam, by_unit in sorted(res["per_family_by_unit"].items()):
        cfgs = {}
        for unit, v in sorted(by_unit.items()):
            m = re.fullmatch(r"(.+-(d\d+k))-L\d+", unit)
            if m and m.group(2) in subject and ((units.get(unit) or {}).get("guard") or {}).get("quiet") is True and v.get("ms_per_step", 0) > 0:
                cfgs.setdefault((m.group(1), m.group(2)), []).append(dict(v, unit=unit))
        for (cfg, depth), vs in sorted(cfgs.items()):
            ms = sum(v["ms_per_step"] for v in vs) / len(vs)
            conf = cpu_config(result_path, cfg)
            rec = {"device": "cpu", "model": model, "kernel_class": fam, "shape": f"{conf}-{depth}", "n": 1, "arm": cfg,
                   "bytes_per_call": int(round(sum(v["gbps"] * v["ms_per_step"] for v in vs) / len(vs) * 1e6)), "us_per_call": round(ms * 1e3, 1),
                   "calls_per_round": 1, "config": conf, "quiet_host": True, "replicates": len(vs)}
            shares = [_idx_share(result_path, v["unit"]) for v in vs] if fam == "MUL_MAT:indexer" else [None]
            if None in shares:
                out.append(rec)
                continue  # classifier split (weight GEMVs vs score GEMM): weight keeps the counted bytes, score is time-only and latency-bound
            sh = sum(shares) / len(shares)
            out.append(dict(rec, kernel_class=fam + "_weight", us_per_call=round(ms * sh * 1e3, 1), split_from=fam))
            out.append(_t(model, fam + "_score", rec["shape"], 1, ms * (1 - sh) * 1e3, "pooled-K bytes not in the RESULT (prof node shape is stamped once)",
                          arm=cfg, config=conf, quiet_host=True, bound="compute_latency", split_from=fam))
    return out


def _t(model: str, cls: str, shape: str, n: int, us: float, basis: str, calls: int = 1, **extra) -> dict:
    """Time-only observation: bytes_per_call is None (the RESULT gives no byte basis; never invented)."""
    return dict({"model": model, "kernel_class": cls, "shape": shape, "n": n, "bytes_per_call": None, "us_per_call": round(float(us), 3),
                 "calls_per_round": calls, "bytes_basis": "unknown: " + basis}, **extra)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _model_of(res: dict) -> str | None:
    """Census model key from the RESULT's own row/task text only (27b / a3b); None = unattributable, never guessed."""
    txt = f"{res.get('row') or ''} {res.get('task') or ''}".lower()
    return "a3b-q8_0" if "a3b" in txt else "27b-q8_0" if "27b" in txt else None


def extract_gdn(res: dict) -> list[dict]:
    """GDN op microbench `cells["svS:nsN:ntT:KK:dtype"]{us_<arm>: us}` (fused OFF/ON, dispatch r3/disp, colwidth c1/c2/c4) or
    verify-base `{us_per_layer}`. Time only: the cell key carries no head count, so the state bytes are not derivable."""
    out = []
    cells = res.get("cells")
    for key in sorted(cells) if isinstance(cells, dict) else []:
        m, c = GDN_CELL.fullmatch(key), cells[key]
        if not m or not isinstance(c, dict):
            continue
        arms = {k[3:]: v for k, v in c.items() if k.startswith("us_") and k != "us_per_layer" and not k.endswith("_info")}
        if _num(c.get("us_per_layer")):
            arms["verify_base"] = c["us_per_layer"]
        for arm, us in sorted(arms.items()):
            if _num(us) and us > 0:
                out.append(_t("gdn-op", f"gdn_core:{arm}", key, int(m.group(3)), us, "GDN op cell key has no head count (state bytes underivable)"))
    return out


def extract_dtree(res: dict) -> list[dict]:
    """DTREE rowcost (`summary.ms_per_verify_by_T[T][class]` ms per verify, `slopes_ms_per_row["a->b"][class].median` ms per extra
    row) and host-gap (`summary.ms_per_round_by_arm[arm][field]` ms per round). Times only; the model is not stated -> `unattributed`."""
    out, s = [], res.get("summary") or {}
    for t, row in sorted((s.get("ms_per_verify_by_T") or {}).items(), key=lambda kv: int(kv[0])):
        out += [_t("unattributed", f"verify_ms:{c}", f"T{t}", int(t), ms * 1e3, "per-class verify wall time, no per-class byte basis")
                for c, ms in sorted(row.items()) if _num(ms)]
    for span, row in sorted((s.get("slopes_ms_per_row") or {}).items()):
        m = re.fullmatch(r"(\d+)->(\d+)", span)
        out += [_t("unattributed", f"rowslope:{c}", f"T{span}", int(m.group(2)), v["median"] * 1e3,
                   "us per extra verify row (median of reps; may be negative)") for c, v in sorted(row.items())
                if m and isinstance(v, dict) and _num(v.get("median"))]
    for arm, row in sorted((s.get("ms_per_round_by_arm") or {}).items()):
        out += [_t("unattributed", f"hostgap:{f}", arm, 1, ms * 1e3, "host-gap phase per round, no byte basis")
                for f, ms in sorted(row.items()) if _num(ms)]
    return out


def extract_trace_time(res: dict) -> list[dict]:
    """rocprof trace RESULT (`summary.trace[tr-<arm>-rN].by_class[phase].cat_ms_per_seg`, `verify_dense_ms`; getrows A/B
    `summary.<arm>.draft_ms|verify_ms`). Times per decode segment (n=0 means per segment, not a batch width) averaged over the
    reps of an arm. Kernel byte bases are hand-calibrated per RESULT, so bytes stay unknown. Needs a model from row/task text."""
    s, model = res.get("summary") or {}, _model_of(res)
    if model is None or not isinstance(s, dict):
        return []
    acc: dict[tuple, list] = {}
    for t in (s.get("trace") or {}).values():
        if not isinstance(t, dict):
            continue
        arm = str(t.get("arm"))
        if _num(t.get("verify_dense_ms")):
            acc.setdefault(("trace_verify:dense_mm", arm), []).append(t["verify_dense_ms"])
        for ph, c in (t.get("by_class") or {}).items():
            if _num((c or {}).get("gpu_ms_per_seg")):
                acc.setdefault((f"trace_{ph}:gpu_total", arm), []).append(c["gpu_ms_per_seg"])
            for cat, ms in ((c or {}).get("cat_ms_per_seg") or {}).items():
                if _num(ms):
                    acc.setdefault((f"trace_{ph}:{cat}", arm), []).append(ms)
    for arm in ("OFF", "ON"):
        for f in ("draft_ms", "verify_ms"):
            if _num((s.get(arm) or {}).get(f)) if isinstance(s.get(arm), dict) else False:
                acc.setdefault((f"e2e_{f}", arm), []).append(s[arm][f])
    return [_t(model, cls, arm, 0, sum(v) / len(v) * 1e3, "rocprof trace time per segment; kernel byte basis not in RESULT", reps=len(v))
            for (cls, arm), v in sorted(acc.items())]


def extract_time_only(res: dict) -> list[dict]:
    """All time-only extractors (GDN cells, DTREE rowcost/host-gap, trace by-class). Pure op/table RESULTs yield nothing."""
    return extract_gdn(res) + extract_dtree(res) + extract_trace_time(res)


def extract_ubench(res: dict) -> list[dict]:
    """CPU standalone-op harness RESULT (`table["[<mode>|]<arm>|N<n>"]{us_per_layer_median, bytes_per_layer}`, top-level `quarter`).
    Only arms in UBENCH_CLASS (production-default `base`, shexp `down`) are emitted; knob arms are experiments, invalid arms
    (`arm_invalid`) are dropped. Bytes are the row's own `bytes_per_layer` (N>1 mmid: uniform-random expert ids -> an upper bound).
    quiet_host only when the ceil-pre pass is valid with SMT-probe p50 <= 1% and post-drift <= 5%."""
    if not isinstance(res.get("table"), dict) or not res.get("quarter") or res.get("status") != "complete":
        return []
    rid_ = str(res.get("id") or "")
    model = next((m for tag, m in CPU_MODEL_BY_ID if tag in rid_), None)
    cc = res.get("ceiling") or {}
    pre = next(iter(cc.get("passes") or []), {})
    quiet = bool(pre.get("valid") and _num(pre.get("probe_p50")) and pre["probe_p50"] <= 1.0
                 and abs((cc.get("post") or {}).get("drift", 1.0)) <= 0.05)
    default_mode = "shexp" if "shexp" in rid_ else "mmid"
    out = []
    for key, v in sorted(res["table"].items()):
        parts = key.split("|")
        mode, arm, n = (parts if len(parts) == 3 else [default_mode, *parts])
        cls = UBENCH_CLASS.get((mode, arm))
        us, b = v.get("us_per_layer_median"), v.get("bytes_per_layer")
        if model is None or cls is None or not (_num(us) and _num(b) and us > 0 and b > 0) or f"{mode}|{arm}" in (res.get("arm_invalid") or []):
            continue
        o = {"device": "cpu", "model": model, "kernel_class": cls[0], "shape": "quarter-24t-ubench", "n": int(n[1:]), "arm": arm,
             "bytes_per_call": int(b), "us_per_call": float(us), "calls_per_round": cls[1], "config": "quarter-24t",
             "bytes_basis": "row bytes_per_layer (standalone op harness)"}
        out.append(dict(o, quiet_host=True) if quiet else o)
    return out


def extract_vattn_ipb(res: dict) -> list[dict]:
    """GPU verify-attention profiler census RESULT (`shapes[]{kv, nq, kv_bytes, dur_us, achieved_gbps, valu_util, counters}`,
    e.g. ak-rka-ipb-census-gpu-20261124-r2): 27B verify FA over q8_0 KV (GQA). Bytes are the row's own kv_bytes; one
    observation per (kv, nq). calls_per_round = 1: the RESULT is per-kernel-call and states no layer count (never invented)."""
    shapes = res.get("shapes")
    if not isinstance(shapes, list) or res.get("complete") is not True or "ipb" not in str(res.get("row") or ""):
        return []
    # rka-ipb rows profile the 27B verify path (AutoKernel 2026-11-24); the row id carries no model tag
    model = _model_of(res) or ("27b-q8_0" if "rka" in str(res.get("row") or "") else None)
    if model is None:
        return []
    out = []
    for sh in shapes:
        us, b, kv, nq = sh.get("dur_us"), sh.get("kv_bytes"), sh.get("kv"), sh.get("nq")
        if not (_num(us) and _num(b) and us > 0 and b > 0 and _num(kv) and _num(nq)):
            continue
        c = sh.get("counters") or {}
        out.append({"model": model, "kernel_class": "gv_fattn_q8kv", "shape": f"kv{int(kv)}", "n": int(nq), "arm": "C21",
                    "bytes_per_call": int(b), "us_per_call": float(us), "calls_per_round": 1,
                    "bytes_basis": "row kv_bytes (q8_0 K+V read per verify-attention call)",
                    "calls_basis": "per call; the RESULT states no per-round layer count",
                    "valu_util": sh.get("valu_util"), "waves": c.get("SQ_WAVES")})
    return out
