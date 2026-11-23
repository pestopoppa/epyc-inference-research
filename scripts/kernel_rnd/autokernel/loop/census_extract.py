"""census_extract.py -- PURE RESULT -> census observation extractors (split out of census_update.py, 2026-11-23).

census_update.py owns the census store (apply/rerank/seeds/doc/journal); this module owns only the parsing of a
filed RESULT.json into observations (dicts of model/kernel_class/shape/n/bytes_per_call/us_per_call). No census state.
"""
from __future__ import annotations

import json
import os
import re

SERVED_N = {1, 3, 4, 8, 16}
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



def extract(res: dict) -> list[dict]:
    """Pure: RESULT dict -> census observations (no I/O)."""
    out = []
    op = res.get("op") or {}
    cells = op.get("cells_us") or {}
    for key in sorted(cells):
        m = re.fullmatch(r"(\d+)x(\d+):n(\d+)", key)
        arms = cells[key]
        if not m or int(m.group(3)) not in SERVED_N or not isinstance(arms, dict):
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
        if not m or int(m.group(2)) not in SERVED_N or not isinstance(row, dict):
            continue
        us, g = row.get("us_on"), row.get("gbps_on")
        if not isinstance(us, (int, float)) or not isinstance(g, (int, float)):
            continue
        cls, calls = MMID_CLASS.get(m.group(1), ("mmid_" + m.group(1), 1))
        out.append({"model": "a3b-q8_0", "kernel_class": cls, "shape": m.group(1), "n": int(m.group(2)), "arm": "ON",
                    "bytes_per_call": int(round(g * us * 1e3)), "us_per_call": float(us), "calls_per_round": calls})
    return out + extract_trace(res)


def extract_trace(res: dict) -> list[dict]:
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
        if not m or int(m.group(3)) not in SERVED_N or not isinstance(g, (int, float)) or g <= 0:
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
    return {24: "quarter-24t", 48: "half-48t"}.get(t, f"{t}t")


def extract_cpu(res: dict, result_path: str | None = None) -> list[dict]:
    """CPU gap-census rows: configs[<cfg>].ranked_families at one row per step (plain decode only; MTP steps carry
    a row-dependent byte count and are skipped, deterministically)."""
    out = []
    rid_ = str(res.get("id") or res.get("row") or "")
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
