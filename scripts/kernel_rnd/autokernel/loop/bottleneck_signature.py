"""AK-BNSIG P1: classify ONE GPU kernel's limiter against the stream references S16 / S64.

Pure module (no I/O beyond ``from_result``'s caller-supplied dict). Lifted from the diag row
``ak-c22-q8wt-op-diag-gpu-20261125`` (job.py parse_counters :405, derive :562, decide :684), with the
row-specific rules (q8wt witness, Q8WT_BEATS, CFGS/DEC4, deeper-pipeline comparison) left out.

References: S16 = stream2 pattern-0 kernel at 16 waves/CU, 16 KB in flight per CU (the healthy-bandwidth
point); S64 = the same kernel at 64 KB/CU (over-subscribed). Each over-subscription axis takes its
DIRECTION from S16 -> S64 rather than assuming "higher is worse": on the 2026-11-25 diag RESULT,
wait_any FALLS from 0.962 (S16) to 0.563 (S64), so the diag's ``wait_any >= midpoint`` leg failed against
its own S64 reference. Before any kernel is classified the references are classified themselves
(``self_check``): S16 must come out ``healthy_bw`` and S64 ``over_subscribed``, otherwise every result is
``reference_incoherent``.

Host quirk kept: MemUnitBusy / MemUnitStalled read 0 per dispatch on this host
(ak-vattn-limiter-counters-gpu-20261124 P2, 1126 dispatches), so mem_busy / mem_stalled are derived from
the raw TA counters of the single C7 pass (TA_BUSY_avr, TA_ADDR_STALLED_BY_TC_CYCLES_sum); the SDK values
are kept as ``sdk_*`` for the record only. The axis is named ``mem_stalled`` here and ``mem_stalled_ta``
in gpu_counter_signals.SIGNATURE_AXES; it is NOT gpu_counter_signals' ``mem_unit_stalled_pct``
(TCP_TCP_TA_DATA_STALL_CYCLES).
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Callable, Iterable, Mapping

SCHEMA = "epyc.autokernel.bottleneck_signature.v1"
NCU = 104
SIMD_NUM = 416.0
REJECT_MAX = 0.25
TH = {"overfetch": 1.15, "ea_rel": 1.15, "issue": 0.60, "issue_soft": 0.40, "wait_inst_soft": 0.40,
      "latency_wait": 0.50, "latency_issue_max": 0.40, "latency_busy_rel": 0.80, "oversub_mid": 0.5, "oversub_firm": 0.75, "valu_rel": 2.0, "band_sep_rel": 0.25,
      "healthy_bw_frac": 0.90, "ref_min_sep": 0.05}
CLASSES = ("harness", "reference_incoherent", "over_fetch", "issue_bound", "over_subscribed", "latency_bound",
           "not_in_flight", "healthy_bw", "unclassified")
# The axes reported for every kernel, and the subset that defines over-subscription.
AXES = ("fetch_over_bytes", "ea_req_per_kb", "mem_stalled", "mem_busy", "wait_any", "wait_inst", "issue_busy",
        "valu_per_kb", "lds_per_kb", "tcp_pending_stall")
OVERSUB_AXES = ("mem_stalled", "wait_any")
REQUIRED = ("fetch_over_bytes", "ea_req_per_kb", "mem_stalled", "mem_busy", "wait_any", "wait_inst", "issue_busy")
LEVER = {"over_fetch": "layout / bytes per weight", "over_subscribed": "in-flight shaping, fewer WGs",
         "issue_bound": "fewer instructions per byte", "latency_bound": "occupancy / pipelining",
         "not_in_flight": "more loads in flight"}
P_RX = re.compile(r"ILi(\d+)E")
# rocprofv3 pmc passes for a signature (C3 MemUnit* dropped: reads 0 here). Re-exported as
# gpu_counter_signals.SIGNATURE_PASSES, the loop's counter-declaration surface.
PASSES_SIG = {
    "C1": ("FETCH_SIZE", "TCC_EA_RDREQ", "TCC_EA_RDREQ_32B", "TCC_EA_RDREQ_DRAM"),
    "C4": ("SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES", "SQ_WAIT_INST_ANY", "SQ_WAIT_ANY", "GRBM_GUI_ACTIVE"),
    "C5": ("SQ_ACTIVE_INST_VALU", "SQ_ACTIVE_INST_ANY", "GRBM_GUI_ACTIVE", "VALUBusy"),
    "C6": ("SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_INSTS_VMEM_RD", "SQ_INSTS_LDS"),
    "C7": ("TA_BUSY_avr", "TA_ADDR_STALLED_BY_TC_CYCLES_sum", "TCP_PENDING_STALL_CYCLES_sum", "GRBM_GUI_ACTIVE"),
}


# ---------------------------------------------------------------- counters (lifted from the diag row)
def parse_counters(rows: Iterable[Mapping[str, Any]], select: Callable[[Mapping[str, Any]], str]):
    """({counter: mean per selected dispatch} (instances summed per dispatch), n dispatches, meta, {family: n}, grids,
    P values). select(row) -> family name of the row; 'main' rows are the ones measured. Absent counters are absent,
    never zero."""
    per: dict = {}
    meta: dict = {}
    fd: dict = {}
    grids, ps = set(), set()
    for r in rows:
        f = select(r)
        fd.setdefault(f, set()).add(r.get("Dispatch_Id"))
        if f != "main":
            continue
        grids.add(int(r.get("Grid_Size") or 0))
        m = P_RX.search(r.get("Kernel_Name") or "")
        if m:
            ps.add(int(m.group(1)))
        key = (r.get("Dispatch_Id"), r.get("Counter_Name"))
        per[key] = per.get(key, 0.0) + float(r.get("Counter_Value") or 0)
        meta = meta or {k: r.get(k) for k in ("Grid_Size", "Workgroup_Size", "LDS_Block_Size", "VGPR_Count",
                                              "SGPR_Count", "Scratch_Size")}
    names = {n for (_, n) in per}
    out = {n: statistics.mean([v for (_, nn), v in per.items() if nn == n]) for n in names}
    return out, len({dd for (dd, _) in per}), meta, {k: len(v) for k, v in fd.items()}, sorted(grids), sorted(ps)


def merge(dst: dict, pas: str, ctrs: Iterable[str], counters: Mapping[str, float]) -> None:
    """Fold one pass's counters into dst, keeping a pass-qualified copy (per-pass GRBM stays separate)."""
    ctrs = set(ctrs)
    dst.update({k: v for k, v in counters.items() if k in ctrs})
    dst.update({f"{pas}.{k}": v for k, v in counters.items() if k in ctrs})


def derive(c: Mapping[str, float], nbytes: float, ncu: int = NCU, simd: float = SIMD_NUM) -> dict:
    """Derived axes from merged counters (diag job.py:562). A metric with a missing input is None."""
    def r(n, d):
        return (c[n] / c[d]) if c.get(n) is not None and c.get(d) else None
    kb = nbytes / 1024.0
    hit, miss = c.get("TCC_HIT_sum"), c.get("TCC_MISS_sum")
    out = {"fetch_over_bytes": c["FETCH_SIZE"] / kb if c.get("FETCH_SIZE") is not None else None,
           "ea_req_per_kb": c["TCC_EA_RDREQ"] / kb if c.get("TCC_EA_RDREQ") is not None else None,
           "frac_32b": r("TCC_EA_RDREQ_32B", "TCC_EA_RDREQ"),
           "tcc_hit": hit / (hit + miss) if hit is not None and miss is not None and hit + miss > 0 else None,
           "sdk_mem_unit_busy": c.get("MemUnitBusy"), "sdk_mem_unit_stalled": c.get("MemUnitStalled"),
           "wait_any": r("SQ_WAIT_ANY", "SQ_WAVE_CYCLES"), "wait_inst": r("SQ_WAIT_INST_ANY", "SQ_WAVE_CYCLES"),
           "mfma_per_kb": c["SQ_INSTS_MFMA"] / kb if c.get("SQ_INSTS_MFMA") is not None else None,
           "valu_per_kb": c["SQ_INSTS_VALU"] / kb if c.get("SQ_INSTS_VALU") is not None else None,
           "lds_per_kb": c["SQ_INSTS_LDS"] / kb if c.get("SQ_INSTS_LDS") is not None else None}
    g = c.get("C5.GRBM_GUI_ACTIVE") or c.get("GRBM_GUI_ACTIVE")
    g4 = c.get("C4.GRBM_GUI_ACTIVE") or c.get("GRBM_GUI_ACTIVE")
    out["issue_busy"] = 4 * c["SQ_ACTIVE_INST_ANY"] / (simd * g) if c.get("SQ_ACTIVE_INST_ANY") is not None and g else None
    out["valu_busy"] = 4 * c["SQ_ACTIVE_INST_VALU"] / (simd * g) if c.get("SQ_ACTIVE_INST_VALU") is not None and g else None
    g7 = c.get("C7.GRBM_GUI_ACTIVE") or c.get("GRBM_GUI_ACTIVE")
    # MemUnit quirk: SDK MemUnitBusy/Stalled read 0 here, so the axes come from the raw TA counters of pass C7.
    out["mem_busy"] = c["TA_BUSY_avr"] / g7 if c.get("TA_BUSY_avr") is not None and g7 else None
    out["mem_stalled"] = c["TA_ADDR_STALLED_BY_TC_CYCLES_sum"] / (ncu * g7) \
        if c.get("TA_ADDR_STALLED_BY_TC_CYCLES_sum") is not None and g7 else None
    out["tcp_pending_stall"] = c["TCP_PENDING_STALL_CYCLES_sum"] / (ncu * g7) \
        if c.get("TCP_PENDING_STALL_CYCLES_sum") is not None and g7 else None
    out["waves_per_simd_approx"] = 4 * c["SQ_WAVE_CYCLES"] / (simd * g4) if c.get("SQ_WAVE_CYCLES") is not None and g4 else None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def mean_axes(ds: list[Mapping[str, Any]]) -> dict:
    """Arithmetic mean per axis over several shapes (the diag's `signatures`; its docstring said geomean)."""
    if not ds:
        return {}
    keys = sorted({k for d in ds for k in d})
    return {k: (round(statistics.mean([d[k] for d in ds if d.get(k) is not None]), 4)
                if any(d.get(k) is not None for d in ds) else None) for k in keys}


# ---------------------------------------------------------------- references
def reference_bands(s16: Mapping[str, Any], s64: Mapping[str, Any]) -> dict:
    """{axis: {s16, s64, direction (+1 rises / -1 falls under over-subscription), mid, coherent}}."""
    out = {}
    for a in OVERSUB_AXES:
        lo, hi = s16.get(a), s64.get(a)
        if lo is None or hi is None:
            out[a] = {"s16": lo, "s64": hi, "direction": None, "mid": None, "coherent": False}
            continue
        d = hi - lo
        out[a] = {"s16": lo, "s64": hi, "direction": (1 if d > 0 else -1) if d else 0,
                  "mid": round(lo + TH["oversub_mid"] * d, 4), "coherent": abs(d) >= TH["ref_min_sep"]}
    return out


def oversub_legs(axes: Mapping[str, Any], bands: Mapping[str, Any]) -> dict:
    """{axis: True if past the midpoint toward S64, False if not, None if unknown}."""
    out = {}
    for a, b in bands.items():
        v = axes.get(a)
        out[a] = None if v is None or b.get("mid") is None or not b.get("direction") else \
            (v - b["mid"]) * b["direction"] >= 0
    return out


def band_positions(axes: Mapping[str, Any], s16: Mapping[str, Any], s64: Mapping[str, Any]) -> dict:
    """{axis: (v - S16) / (S64 - S16)}: 0 = at the healthy reference, 1 = at the over-subscribed one. None when unknown
    or when the references do not separate on that axis (|S64 - S16| < band_sep_rel x max(|S16|, |S64|))."""
    out = {}
    for a in AXES:
        v, lo, hi = axes.get(a), s16.get(a), s64.get(a)
        sep = None if lo is None or hi is None else abs(hi - lo)
        out[a] = None if v is None or not sep or sep < TH["band_sep_rel"] * max(abs(lo), abs(hi)) \
            else round((v - lo) / (hi - lo), 4)
    return out


# Rule order = design note section 6 (over_fetch, over_subscribed, issue_bound, latency_bound). Every rule is evaluated
# and EVERY firing class is reported (a combined limiter needs both levers); the order only picks `primary`, which is the
# first NON-marginal firing class (else the first firing one). over_subscribed leans on mem_stalled alone: wait_any is
# confounded with instruction density (an issue-heavy kernel waits less), so it is reported as support only.
def classify(axes: Mapping[str, Any], s16: Mapping[str, Any], bands: Mapping[str, Any],
             bw_frac: float | None = None, s64: Mapping[str, Any] | None = None) -> dict:
    """{class (= primary), primary, classes, marginal, rules, ruled_out, band_position, oversub_legs}.
    bw_frac = achieved GB/s / S16 GB/s."""
    s64 = s64 or {}
    legs = oversub_legs(axes, bands)
    pos = band_positions(axes, s16, s64)
    missing = [a for a in REQUIRED if axes.get(a) is None]
    if missing:
        return {"class": "harness", "primary": "harness", "classes": ["harness"], "marginal": [],
                "rule": "missing_axes:" + ",".join(missing), "rules": {}, "ruled_out": [], "band_position": pos,
                "oversub_legs": legs}
    ea_ref, valu_ref, wi_ref = s16.get("ea_req_per_kb"), s16.get("valu_per_kb"), s64.get("wait_inst")
    # issue_bound needs a discriminator the pure stream lacks: S64 itself has wait_inst 0.41 >= the old soft bar.
    disc = bool(valu_ref and axes.get("valu_per_kb") is not None and axes["valu_per_kb"] >= TH["valu_rel"] * valu_ref) or \
        bool(wi_ref is not None and axes["wait_inst"] > wi_ref)
    ms = pos.get("mem_stalled")
    tests = [
        ("over_fetch", axes["fetch_over_bytes"] > TH["overfetch"] or
         bool(ea_ref and axes["ea_req_per_kb"] > TH["ea_rel"] * ea_ref),
         f"fetch_over_bytes>{TH['overfetch']} or ea_req_per_kb>{TH['ea_rel']}*S16"),
        ("over_subscribed", ms is not None and bands["mem_stalled"].get("direction") == 1 and ms >= TH["oversub_mid"],
         f"mem_stalled band position >= {TH['oversub_mid']} (marginal below {TH['oversub_firm']}); wait_any support only"),
        ("issue_bound", axes["issue_busy"] >= TH["issue"] or (axes["issue_busy"] >= TH["issue_soft"] and disc),
         f"issue_busy>={TH['issue']} or (issue_busy>={TH['issue_soft']} and (valu_per_kb>={TH['valu_rel']}*S16 "
         "or wait_inst>S64))"),
        ("latency_bound", bool(s16.get("mem_busy")) and axes["mem_busy"] < TH["latency_busy_rel"] * s16["mem_busy"]
         and axes["wait_any"] >= TH["latency_wait"] and axes["issue_busy"] < TH["latency_issue_max"],
         f"mem_busy<{TH['latency_busy_rel']}*S16 and wait_any>={TH['latency_wait']} and issue_busy<{TH['latency_issue_max']}"),
    ]
    classes = [n for n, hit, _r in tests if hit]
    rules = {n: r for n, hit, r in tests if hit}
    marginal = ["over_subscribed"] if "over_subscribed" in classes and ms < TH["oversub_firm"] else []
    if not classes:
        name = "unclassified" if bw_frac is None else "healthy_bw" if bw_frac >= TH["healthy_bw_frac"] else "not_in_flight"
        rules = {name: f"no limiter fired; GB/s vs S16 {bw_frac} (healthy >= {TH['healthy_bw_frac']})"}
        primary, classes = name, [name]
    else:
        primary = next((c for c in classes if c not in marginal), classes[0])
    return {"class": primary, "primary": primary, "classes": classes, "marginal": marginal, "rule": rules[primary],
            "rules": rules, "ruled_out": [n for n, hit, _r in tests if not hit], "band_position": pos,
            "oversub_legs": legs}


def self_check(s16: Mapping[str, Any], s64: Mapping[str, Any], s64_bw_frac: float | None = None) -> dict:
    """S16 must classify healthy_bw and S64 over_subscribed against the bands they define.

    Because bands are DERIVED from the two references, the classification half is close to tautological (S16 sits at
    band position 0, S64 at 1, and a swap is self-consistent). The check's real content is the two ABSOLUTE anchors
    below: mem_stalled must rise S16 -> S64, and S64's GB/s must not beat S16's; plus a minimum separation per axis."""
    bands = reference_bands(s16, s64)
    got16 = classify(s16, s16, bands, 1.0, s64)["class"]
    got64 = classify(s64, s16, bands, s64_bw_frac, s64)["class"]
    incoherent = [a for a, b in bands.items() if not b["coherent"]]
    # Direction-from-references is symmetric under an S16<->S64 swap, so one axis carries an ABSOLUTE anchor: the
    # TA address path stalled by TC must rise under over-subscription (that is what over-subscribed means), and a
    # supplied S64 GB/s must not beat S16. Every other axis's direction is free.
    if bands["mem_stalled"]["direction"] != 1 and "mem_stalled" not in incoherent:
        incoherent.append("mem_stalled")
    if s64_bw_frac is not None and s64_bw_frac > 1.0:
        incoherent.append("gbps")
    ok = got16 == "healthy_bw" and got64 == "over_subscribed" and not incoherent
    return {"ok": ok, "S16": got16, "S64": got64, "incoherent_axes": incoherent, "bands": bands}


def signature(kernel: str, axes: Mapping[str, Any], s16: Mapping[str, Any], s64: Mapping[str, Any], *,
              refs: Mapping[str, Any] | None = None, gbps: float | None = None, s16_gbps: float | None = None,
              s64_gbps: float | None = None, rejected_frac: float | None = None) -> dict:
    """One kernel's signature document (schema epyc.autokernel.bottleneck_signature.v1)."""
    frac = (lambda g: round(g / s16_gbps, 4) if g is not None and s16_gbps else None)
    chk = self_check(s16, s64, frac(s64_gbps))
    try:  # a vendored copy (row job dirs) has no package; the sha is then None, never invented
        from .gpu_counter_signals import signature_counter_set_sha256
        cs = signature_counter_set_sha256()
    except ImportError:
        cs = None
    doc = {"schema": SCHEMA, "kernel": kernel, "axes": {a: axes.get(a) for a in AXES}, "counter_set_sha256": cs,
           "refs": dict(refs or {}, S16={a: s16.get(a) for a in AXES}, S64={a: s64.get(a) for a in AXES},
                        s16_gbps=s16_gbps, s64_gbps=s64_gbps),
           "bands": chk["bands"], "reference_check": {k: chk[k] for k in ("ok", "S16", "S64", "incoherent_axes")},
           "gbps": gbps, "bw_frac_of_s16": frac(gbps), "thresholds": dict(TH)}
    if rejected_frac is not None and rejected_frac > REJECT_MAX:
        res = {"class": "harness", "primary": "harness", "classes": ["harness"], "marginal": [],
               "rule": f"rejected_frac {rejected_frac} > {REJECT_MAX}", "ruled_out": [],
               "oversub_legs": oversub_legs(axes, chk["bands"])}
    elif not chk["ok"]:
        res = {"class": "reference_incoherent", "primary": "reference_incoherent", "classes": ["reference_incoherent"],
               "marginal": [],
               "rule": f"self-check: S16->{chk['S16']} (want healthy_bw), S64->{chk['S64']} (want over_subscribed)"
                       + (f", incoherent axes {chk['incoherent_axes']}" if chk["incoherent_axes"] else ""),
               "ruled_out": [], "oversub_legs": oversub_legs(axes, chk["bands"])}
    else:
        res = classify(axes, s16, chk["bands"], frac(gbps), s64)
    doc.update(res, lever=LEVER.get(res["class"]), levers={c: LEVER[c] for c in res.get("classes", []) if c in LEVER})
    return doc


def from_result(result: Mapping[str, Any], kernel_key: str, *, gbps: float | None = None,
                ref_keys: tuple[str, str] = ("S16", "S64")) -> dict:
    """Signature of `kernel_key` from a diag-row RESULT.json dict (summary.signatures / counters_per_kernel)."""
    s = result.get("summary") or {}
    sigs = dict(s.get("counters_per_kernel") or {}, **(s.get("signatures") or {}))
    k16, k64 = ref_keys
    sg = s.get("stream_67m_sig_gbps") or {}
    return signature(kernel_key, sigs.get(kernel_key) or {}, sigs.get(k16) or {}, sigs.get(k64) or {},
                     refs={"S16_id": k16, "S64_id": k64, "row": result.get("row")}, gbps=gbps,
                     s16_gbps=sg.get(k16), s64_gbps=sg.get(k64), rejected_frac=s.get("rejected_frac"))
