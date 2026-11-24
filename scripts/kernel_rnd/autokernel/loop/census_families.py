"""census_families.py: roll the per-(path, op) census surfaces up into op FAMILIES and rank the families (RSL-R4).

A family is the same kernel problem seen on different models/paths (flash-attn-verify on 27B and A3B; mmid-moe on the GPU and
the CPU). score(family) = sum over surfaces of w x saving_ms_per_step, where w comes from `model_weights` (default 1):
the weight is looked up by PATH first (e.g. "a3b-mtp-mi210" -- how a campaign separates "served today" from "served on the
next GPU"), then by the path's MODEL (e.g. "a3b-q8_0"), then 1. Surfaces whose saving is None add nothing and are counted.
Serving-knob arms ({"name", "path"?, "step_ms"?, "vs_control_pct"?}) are listed beside the families: priced only when they
carry a screened vs_control_pct (a THROUGHPUT gain vs control, 100*(tok_s_arm-tok_s_ctrl)/tok_s_ctrl, serving_knobs.py), else
saving=None, 'unpriced'. Step saving = step_ms x (1 - 1/(1 + pct/100)); a negative pct clamps to 0 and stays priced at 0.
Read-only on census_update; this module owns the family table, census_update.OP_MAP stays the (model, class)->(path, op) map.
"""
from . import census_update as cu

UNFAMILIED = "unfamilied"
KNOB = "serving-knob"

_FAMILY = {  # (path, op) -> family. Every cu.OP_MAP value MUST be here (test_every_op_map_entry_has_a_family).
    ("27b-df2-mi210", "verify dense Q8 GEMV/GEMM (q8mf6, V5I)"): "dense-q8-gemv",
    ("27b-df2-mi210", "attn_k/v 1024x5120 (q8sk)"): "dense-q8-gemv",
    ("a3b-mtp-mi210", "dense MMVQ (attn, GDN proj, shexp, lm_head)"): "dense-q8-gemv",
    ("a3b-mtp-cpu", "fullattn_proj"): "dense-q8-gemv",
    ("a3b-mtp-cpu", "gdn_proj"): "dense-q8-gemv",
    ("a3b-mtp-cpu", "shared expert N>=2"): "dense-q8-gemv",
    ("flash-next-cpu", "forward m1 dense (UD-IQ4_XS)"): "dense-iq4xs-gemv",
    ("flash-next-cpu", "fullattn_proj"): "dense-iq4xs-gemv",
    ("flash-next-cpu", "gdn_proj"): "dense-iq4xs-gemv",
    ("flash-next-cpu", "indexer"): "dense-iq4xs-gemv",
    ("flash-next-cpu", "indexer_weight"): "dense-iq4xs-gemv",
    ("flash-next-cpu", "other_weight"): "dense-iq4xs-gemv",
    ("flash-next-cpu", "shared_expert"): "dense-iq4xs-gemv",
    ("27b-df2-mi210", "lm_head 248320x5120"): "lm-head-gemv",
    ("a3b-mtp-cpu", "lm_head"): "lm-head-gemv",
    ("flash-next-cpu", "lm_head"): "lm-head-gemv",
    ("27b-df2-mi210", "gv_fattn + merge (verify FA)"): "flash-attn-verify",
    ("27b-df2-mi210", "q4_0 KV attention (gv_fattn, q4_0 KV)"): "flash-attn-verify",
    ("a3b-mtp-mi210", "FA / gv (10 layers, f16 KV)"): "flash-attn-verify",
    ("a3b-mtp-mi210", "MUL_MAT_ID experts n=3"): "mmid-moe",
    ("a3b-mtp-cpu", "MUL_MAT_ID exps_down"): "mmid-moe",
    ("a3b-mtp-cpu", "MUL_MAT_ID exps_gate_up"): "mmid-moe",
    ("flash-next-cpu", "exps_down"): "mmid-moe",
    ("flash-next-cpu", "exps_gate_up"): "mmid-moe",
    ("a3b-mtp-mi210", "router mul_mat_f F32 [2048,256]"): "moe-router",
    ("a3b-mtp-mi210", "router chain softmax/argsort/get_rows"): "moe-router",
    ("a3b-mtp-cpu", "router (CPU)"): "moe-router",
    ("flash-next-cpu", "moe_router"): "moe-router",
    # parity ops with no census class yet (OP_MAP has no entry): still families, so their share is not lost
    ("27b-df2-mi210", "GDN recurrence (fused tree GDN)"): "gdn-recurrence",
    ("27b-df2-mi210", "drafter BF16 hipBLAS (not served; training deferred)"): "drafter-gemm",
    ("27b-df2-mi210", "actquant (q8mf_pre_i8 + q8sk_quantize)"): "act-quantize",
    ("27b-df2-mi210", "draft q8ip_permute"): "act-quantize",
    ("a3b-mtp-mi210", "quantize_q8_1"): "act-quantize",
    ("27b-df2-mi210", "launch-bound tail (eltwise/norm/copy/rope/conv)"): "launch-bound-tail",
    ("a3b-mtp-mi210", "k_bin_bcast / scale"): "launch-bound-tail",
}


def op_family(path: str, op: str) -> str:
    """Family of one (path, op); UNFAMILIED when absent (rank() reports those so a new op cannot hide)."""
    return _FAMILY.get((path, op), UNFAMILIED)


def unmapped_op_map_entries() -> list:
    """OP_MAP values with no family (must be empty; the test fails otherwise)."""
    return sorted(v for v in set(cu.OP_MAP.values()) if v not in _FAMILY)


def _path_model() -> dict:
    return {path: model for (model, _kc), (path, _op) in cu.OP_MAP.items()}


def weight_of(path: str, model_weights: dict | None, path_model: dict | None = None) -> float:
    """Path key beats model key beats 1."""
    mw = model_weights or {}
    if path in mw:
        return float(mw[path])
    model = (path_model or _path_model()).get(path)
    return float(mw[model]) if model in mw else 1.0


def _knob_row(a: dict, steps: dict, mw, pm) -> dict:
    pct, path = a.get("vs_control_pct"), a.get("path")
    step = a.get("step_ms") or steps.get(path)
    saving = None if pct is None or step is None else round(step * (1 - 1 / (1 + max(0.0, pct) / 100.0)), 3)
    w = weight_of(path, mw, pm) if path else 1.0
    return {"family": f"{KNOB}:{a.get('name')}", "kind": "knob", "path": path, "vs_control_pct": pct,
            "saving_ms_per_step": saving, "score": None if saving is None else round(w * saving, 3),
            "weight": w, "status": "unpriced" if saving is None else "priced", "members": []}


def rank(surfaces: list, model_weights: dict | None = None, knob_arms: list | None = None) -> list:
    """Families (and knob arms) sorted by weighted score, unpriced last. Each row: family, kind, score, n_surfaces,
    n_unpriced, status (priced|unpriced), members [{path, op, weight, saving_ms_per_step}]."""
    pm = _path_model()
    fam = {}
    for s in surfaces:
        f = fam.setdefault(op_family(s["path"], s["op"]), [])
        w, sv = weight_of(s["path"], model_weights, pm), s.get("saving_ms_per_step")
        f.append({"path": s["path"], "op": s["op"], "weight": w, "saving_ms_per_step": sv,
                  "weighted": None if sv is None else w * sv})
    rows = []
    for name, ms in fam.items():
        priced = [m["weighted"] for m in ms if m["weighted"] is not None]
        rows.append({"family": name, "kind": "family", "score": round(sum(priced), 3) if priced else None,
                     "n_surfaces": len(ms), "n_unpriced": len(ms) - len(priced),
                     "status": "priced" if priced else "unpriced",
                     "members": sorted(ms, key=lambda m: -(m["weighted"] or 0))})
    steps = {s["path"]: s["step_ms"] for s in surfaces if s.get("step_ms")}
    rows += [_knob_row(a, steps, model_weights, pm) for a in knob_arms or []]
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0), r["family"]))
    return rows
