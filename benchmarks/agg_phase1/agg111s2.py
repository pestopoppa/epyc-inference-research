#!/usr/bin/env python3
"""AGG-1.11 step 2: the ``agg_pipeline.v2`` typed L4 stage re-run on the 13 AGG-1.3/1.4 r3 misses (library; ``run_agg111s2.py`` is
the CLI). Implements ``PREDECLARED_AGG111S2.md`` literally.

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an injected backend.
Reuses ``agg16`` (in-memory store copies, the recording backend, the M1 cell rule), ``agg112`` (descriptions, the typed window
record and its fact shape), ``agg1314_zero_cell_diag`` (the frozen 24 + 25 cell scorer) and the pinned orchestrator's
``facts_typed`` (plan, typed call, quote rule, reconcile).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import agg112 as G112
import agg1314_zero_cell_diag as Z
import agg16 as G16
import phase1_arms as X
from agg111_triage import num_re, squash
from phase1_arms import P

HERE = Path(__file__).resolve().parent
PREDECLARED_FILE = "PREDECLARED_AGG111S2.md"
PREDECLARED_SHA = "60d069edfff786c40219c1c14364734417a69f4b3031faadb10998a0ed116bed"
CORPUS_ID = P.CORPUS_ID
SHA256 = X.sha256_file
STORE_OF_GROUP = {"a": "agg13", "b": "agg14"}
N_TARGETS, N_A, N_B = 13, 10, 3
N_CELLS13, N_CELLS14 = 24, 25
PILOT_CELLS = (
    ("COCACOLA_2022_10K", "long_term_debt", "2022"), ("3M_2022_10K", "total_assets", "2021"),
    ("3M_2022_10K", "total_current_liabilities", "2021"), ("MICROSOFT_2022_10K", "capex", "2021"),
    ("JOHNSON_JOHNSON_2022_10K", "total_assets", "2021"),
)
PILOT_FUTILE_AT = 0                     # the pilot fills 0 of 5 -> futility stop
PILOT_WALL_MAX_S = 600.0                # budget stop (2x the box's 300 s)
BUDGET_X = 2.0
RATE_S_PER_CALL = G112.RATE_S_PER_CALL  # MEASURED, AGG-1.6 r2 arm T: 201 calls in 1946.1 s
SERVER_START_S = 120.0                  # ESTIMATE
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_PARTIAL = (X.RC_OK, X.RC_PREFLIGHT, X.RC_GATE, X.RC_PRECOND,
                                                                   X.RC_BUDGET, X.RC_PARTIAL)


# --------------------------------------------------------------------------- inputs
def key(doc: str, field: str, fy: str) -> str:
    return f"{doc}|{field}|{fy}"


def load_targets(set_dir: Path = HERE) -> list[dict[str, Any]]:
    tri = json.loads((set_dir / "agg111_triage.json").read_text())
    out = []
    for c in tri["cells"]:
        if c["group"] in STORE_OF_GROUP:
            t = {"doc": c["doc"], "field": c["field"], "fy": str(c["fy"]), "gold": float(c["gold"]), "group": c["group"],
                 "store": STORE_OF_GROUP[c["group"]], "primary": c["primary"], "l4_class": c.get("l4_class")}
            t["key"] = key(t["doc"], t["field"], t["fy"])
            out.append(t)
    return sorted(out, key=lambda t: (t["store"], t["doc"], t["field"]))


def load_cells(set_dir: Path = HERE) -> dict[str, list[dict[str, Any]]]:
    out = {"agg13": Z.load_cells13(set_dir / "agg13_cells.csv"), "agg14": Z.load_cells14(set_dir / "agg14_cells.csv")}
    for cs in out.values():
        for c in cs:
            c["key"] = key(c["doc"], c["field"], c["fy"])
    return out


def load_inputs(set_dir: Path = HERE) -> dict[str, Any]:
    return {"targets": load_targets(set_dir), "cells": load_cells(set_dir), "set_dir": set_dir}


def input_problems(inp: dict[str, Any], set_dir: Path = HERE, *, check_pin: bool = True) -> list[str]:
    out = []
    if check_pin:
        f = set_dir / PREDECLARED_FILE
        got = SHA256(f) if f.exists() else None
        if got != PREDECLARED_SHA:
            out.append(f"{PREDECLARED_FILE} sha256 {got} != {PREDECLARED_SHA}")
    t = inp["targets"]
    n = (len(t), sum(x["group"] == "a" for x in t), sum(x["group"] == "b" for x in t))
    if n != (N_TARGETS, N_A, N_B):
        out.append(f"targets (n, a, b) {n} != {(N_TARGETS, N_A, N_B)}")
    keys = {x["key"] for x in t}
    out += [f"pilot cell {k} is not a target" for k in PILOT_CELLS if key(*k) not in keys]
    if (len(inp["cells"]["agg13"]), len(inp["cells"]["agg14"])) != (N_CELLS13, N_CELLS14):
        out.append(f"cells {len(inp['cells']['agg13'])}/{len(inp['cells']['agg14'])} != {N_CELLS13}/{N_CELLS14}")
    return out


# --------------------------------------------------------------------------- the plan (zero model)
def ft():
    return G112.orch_symbols()["ft"]


def doc_windows(store: Any) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for w in ft().split_windows(store, CORPUS_ID, P.CFG.window_chars):
        out.setdefault(w["doc_id"], []).append(w)
    return out


def doc_plans(targets: list[dict[str, Any]], wins_by_store: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, Any]]:
    """Per (store, doc): the distinct union windows of the document's TARGET fields, who chose each, and per target field its own
    union window ids (the pilot scores a cell from those)."""
    m = ft()
    out: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in targets:
        groups.setdefault((t["store"], t["doc"]), []).append(t)
    for (store, doc), ts in sorted(groups.items()):
        fy = ts[0]["fy"]
        if any(t["fy"] != fy for t in ts):
            raise ValueError(f"{store}/{doc}: targets with two fiscal years")
        descs = G112.descs_for(fy)
        fields = sorted({t["field"] for t in ts})
        wins, chosen = m.doc_call_plan(wins_by_store[store][doc], fields, descs)
        per_field = {f: [w["chunk_id"] for w in m.union_candidates(wins_by_store[store][doc], f, descs[f])] for f in fields}
        out[f"{store}:{doc}"] = {"store": store, "doc": doc, "fy": fy, "fields": fields, "windows": wins,
                                 "chosen": {k: sorted(v) for k, v in chosen.items()}, "per_field": per_field,
                                 "n_windows": len(wins), "chars": sum(len(w["text"]) for w in wins)}
    return out


def pilot_window_ids(plans: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """(plan id, chunk id) of every pilot cell's union windows, in pilot order, deduplicated."""
    seen: list[tuple[str, str]] = []
    for doc, field, fy in PILOT_CELLS:
        pid = next(k for k, p in plans.items() if p["doc"] == doc and p["fy"] == fy)
        for cid in plans[pid]["per_field"][field]:
            if (pid, cid) not in seen:
                seen.append((pid, cid))
    return seen


def estimates(plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    calls = sum(p["n_windows"] for p in plans.values())
    pilot = len(pilot_window_ids(plans))
    return {"label": "ESTIMATE", "s_per_call": round(RATE_S_PER_CALL, 3), "calls": calls, "pilot_calls": pilot,
            "pilot_s": round(pilot * RATE_S_PER_CALL, 1), "server_start_s": SERVER_START_S,
            "total_s": round(calls * RATE_S_PER_CALL + SERVER_START_S, 1),
            "per_doc": {k: p["n_windows"] for k, p in plans.items()}}


# --------------------------------------------------------------------------- scoring (frozen M1 rule)
def score_store(path: Path, cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return {c["key"]: Z.diagnose_cell(c, Z.facts_for(conn, c["doc"], c["field"])) for c in cells}
    finally:
        conn.close()


def canonical_ok(path: Path, cells: list[dict[str, Any]]) -> dict[str, bool]:
    from src.aggregation.store import select_canonical  # noqa: PLC0415
    return G16.canon_ok_fy(select_canonical(G16.saved_rows(path))[0], cells)


def m1(diag: dict[str, dict[str, Any]]) -> int:
    return sum(1 for d in diag.values() if d["verdict"] == "HIT")


def compare(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fixed = sorted(k for k in after if after[k]["verdict"] == "HIT" and before[k]["verdict"] != "HIT")
    broken = sorted(k for k in before if before[k]["verdict"] == "HIT" and after[k]["verdict"] != "HIT")
    return {"n": len(before), "before": m1(before), "after": m1(after), "fixed": fixed, "regressed": broken,
            "transitions": {k: [before[k]["verdict"], after[k]["verdict"]] for k in before if before[k]["verdict"] != after[k]["verdict"]}}


# --------------------------------------------------------------------------- the typed calls
def window_unit(pid: str, chunk_id: str) -> str:
    return f"tw:{pid}:{chunk_id}"


def win_records(rec: P.Recorder) -> dict[str, dict[str, Any]]:
    return {r["unit"]: r for r in rec.records if r.get("kind") == "t_window" and r.get("final")}


def run_window(rec: P.Recorder, tb: Any, rb: Any, pid: str, plan: dict[str, Any], w: dict[str, Any], guard: Callable[[], None]) -> dict[str, Any]:
    """ONE typed call (``agg112.run_window`` with this design's unit key and fiscal-year descriptions)."""
    guard()
    t0 = time.time()
    mark = rb.mark()
    status, res = "ok", None
    try:
        res = tb.call_window(w, G112.descs_for(plan["fy"]))
    except Exception as e:  # noqa: BLE001 - recorded per window; a re-grant retries it
        status = f"error:{type(e).__name__}:{str(e)[:200]}"
    cs = G16.calls_stats(rb.since(mark))
    return rec.emit("t_window", unit=window_unit(pid, w["chunk_id"]), plan=pid, doc=plan["doc"], chunk_id=w["chunk_id"],
                    status=status, ok=status == "ok", parsed=bool(res and res.parsed), bad=bool(res and res.bad) and status == "ok",
                    chosen_by=plan["chosen"][w["chunk_id"]], chars=len(w["text"]), **cs,
                    wall_s=time.time() - t0, facts=[G112.full_fact(f) for f in res.facts] if res else [], final=status == "ok")


def cell_facts(rec: P.Recorder, plan: dict[str, Any], pid: str, field: str) -> list[dict[str, Any]] | None:
    """A field's facts from ITS union windows only: keep_slot, the quote rule, reconcile. None while a window is missing."""
    m = ft()
    byw = win_records(rec)
    raw: list[dict[str, Any]] = []
    for cid in plan["per_field"][field]:
        r = byw.get(window_unit(pid, cid))
        if r is None:
            return None
        raw.extend(f for f in r["facts"] if m.keep_slot(f, {field}))
    kept, _drop = m.drop_unquoted(raw)
    return _reconcile(kept)


def _reconcile(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.aggregation import facts_llm  # noqa: PLC0415
    return facts_llm.reconcile(list(facts))[0]


def pilot_eval(rec: P.Recorder, plans: dict[str, dict[str, Any]], targets: list[dict[str, Any]], *, booked_s: float) -> dict[str, Any]:
    by = {t["key"]: t for t in targets}
    cells = {}
    for doc, field, fy in PILOT_CELLS:
        t = by[key(doc, field, fy)]
        pid = f"{t['store']}:{doc}"
        facts = cell_facts(rec, plans[pid], pid, field) or []
        cells[t["key"]] = G16.cell_hit(facts, t)
    ids = {window_unit(p, c) for p, c in pilot_window_ids(plans)}
    recs = [r for u, r in win_records(rec).items() if u in ids]
    wall = sum(float(r.get("wall_s") or 0) for r in recs)
    calls = sum(p["n_windows"] for p in plans.values())
    projected = wall / max(len(recs), 1) * calls + SERVER_START_S
    ev = {"filled": sum(cells.values()), "of": len(cells), "calls": len(recs), "wall_s": round(wall, 1),
          "projected_total_s": round(projected, 1), "booked_s": booked_s, "budget_stops": []}
    if wall > PILOT_WALL_MAX_S:
        ev["budget_stops"].append("pilot_wall_over_600s")
    if projected > BUDGET_X * booked_s:
        ev["budget_stops"].append("projected_wall_over_2x_booked")
    ev["futility"] = ev["filled"] <= PILOT_FUTILE_AT
    return ev


def finish_doc(rec: P.Recorder, tb: Any, pid: str, plan: dict[str, Any]) -> dict[str, Any]:
    m = ft()
    byw = win_records(rec)
    kept_in: list[dict[str, Any]] = []
    recs = []
    for w in plan["windows"]:
        r = byw[window_unit(pid, w["chunk_id"])]
        recs.append(r)
        kept_in.extend(f for f in r["facts"] if m.keep_slot(f, set(plan["chosen"][w["chunk_id"]])))
    out = tb.finish_doc(CORPUS_ID, kept_in, plan["fields"])
    return rec.emit("doc_l4", unit=f"doc_l4:{pid}", plan=pid, doc=plan["doc"], n_windows=len(recs),
                    llm_calls=sum(r["calls"] for r in recs), prefill_tokens=sum(r["prefill_tokens"] for r in recs),
                    wall_s=sum(r["wall_s"] for r in recs), bad_replies=sum(1 for r in recs if r.get("bad")),
                    n_facts_in=len(kept_in), n_facts_kept=len(out["facts"]), n_quote_dropped=len(out.get("quote_dropped", [])),
                    by_field={f: sum(1 for k in out["facts"] if k.get("field") == f) for f in plan["fields"]}, final=True)


# --------------------------------------------------------------------------- the 25-backfill gap diagnosis (zero model)
def _gold_close(v: float, unit: str | None, gold: float) -> bool:
    return Z._close(float(v) * Z.UNIT_SCALE.get(unit, 1.0), gold, Z.TOL) or Z._close(float(v), gold, Z.TOL)


def _facts_all(path: Path, doc: str, field: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return Z.facts_for(conn, doc, field)
    finally:
        conn.close()


def gap_diagnosis(cells14: list[dict[str, Any]], *, agg14_store: Path, r2_store: Path, ingest_store: Path,
                  ingest_results: Path) -> dict[str, Any]:
    """PREDECLARED_AGG111S2.md, the ordered first-match rule for every cell HIT on AGG-1.4 r3 and not HIT on the AGG-1.12 r2 store."""
    m = ft()
    b14, b12 = score_store(agg14_store, cells14), score_store(r2_store, cells14)
    lost = [c for c in cells14 if b14[c["key"]]["verdict"] == "HIT" and b12[c["key"]]["verdict"] != "HIT"]
    gained = sorted(c["key"] for c in cells14 if b12[c["key"]]["verdict"] == "HIT" and b14[c["key"]]["verdict"] != "HIT")
    tw = {}
    for r in X.read_jsonl(ingest_results):
        if r.get("kind") == "t_window" and r.get("final"):
            tw[r["unit"]] = r
    rows = []
    wins_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for c in lost:
        doc, field, gold, fy = c["doc"], c["field"], c["gold"], c["fy"]
        ev: dict[str, Any] = {"cell": c["key"], "gold": gold, "agg14_r3_won_by": b14[c["key"]]["candidate"],
                              "r2_verdict": b12[c["key"]]["verdict"], "r2_candidate": b12[c["key"]]["candidate"]}
        r2 = [f for f in _facts_all(r2_store, doc, field) if f.get("value") is not None]
        ing = [f for f in _facts_all(ingest_store, doc, field) if f.get("value") is not None]
        step = None
        r2_gold = [f for f in r2 if _gold_close(f["value"], f.get("unit"), gold)]
        if r2_gold and not any(fy in str(f.get("period") or "") for f in r2_gold):
            step, ev["evidence"] = "PERIOD", r2_gold[:3]
        elif not r2_gold and any(_gold_close(f["value"], f.get("unit"), gold) and fy in str(f.get("period") or "") for f in ing):
            step, ev["evidence"] = "VALUE_IN_QUOTE_POSTHOC", [f for f in ing if _gold_close(f["value"], f.get("unit"), gold)][:3]
        else:
            if doc not in wins_cache:
                st = G16.mem_store(r2_store, [doc])
                try:
                    wins_cache.update(doc_windows(st))
                finally:
                    st.close()
            wins = wins_cache.get(doc, [])
            descs = G112.descs_for(fy)
            union = m.union_candidates(wins, field, descs[field])
            rx = num_re(gold)
            holding = [w["chunk_id"] for w in union if rx.search(squash(w["text"]))]
            ev["union_windows"] = [w["chunk_id"] for w in union]
            ev["gold_windows_in_union"] = holding
            if not holding:
                any_win = [w["chunk_id"] for w in wins if rx.search(squash(w["text"]))]
                step, ev["evidence"] = "RETRIEVAL", {"gold_windows_in_doc": any_win[:10]}
            else:
                rws = [f for cid in [w["chunk_id"] for w in union]
                       for f in (tw.get(G112.window_unit(doc, cid)) or {}).get("facts", []) if f.get("field") == field]
                ev["windows_with_saved_reply"] = sum(1 for w in union if G112.window_unit(doc, w["chunk_id"]) in tw)
                gr = [f for f in rws if f.get("value") is not None and _gold_close(f["value"], f.get("unit"), gold)]
                gp = [f for f in gr if fy in str(f.get("period") or "")]
                gq = [f for f in gp if m.value_in_quote(f) is not None]
                if not gr:
                    step, ev["evidence"] = "EXTRACTION", [{k: f.get(k) for k in ("period", "value", "unit", "quote", "chunk_id")} for f in rws[:6]]
                elif not gp:
                    step, ev["evidence"] = "PERIOD_L4_ROW", gr[:3]
                elif not gq:
                    step, ev["evidence"] = "VALUE_IN_QUOTE", gp[:3]
                elif not any(_gold_close(f["value"], f.get("unit"), gold) and fy in str(f.get("period") or "") for f in r2):
                    step, ev["evidence"] = "CANONICAL", gq[:3]
                else:
                    step, ev["evidence"] = "OTHER", gq[:3]
        ev["step"] = step
        rows.append(ev)
    by: dict[str, int] = {}
    for r in rows:
        by[r["step"]] = by.get(r["step"], 0) + 1
    return {"agg14_r3_hits": m1(b14), "r2_hits": m1(b12), "n": len(cells14), "lost": len(rows), "gained_on_r2": gained,
            "by_step": by, "cells": rows, "inputs": {"agg14_store": str(agg14_store), "r2_store": str(r2_store),
                                                    "ingest_store": str(ingest_store), "ingest_results": str(ingest_results)}}


# --------------------------------------------------------------------------- the run
def _result(out: Path, **res: Any) -> dict[str, Any]:
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = PREDECLARED_SHA
    X._atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    return res


def run(rec: P.Recorder, out: Path, inp: dict[str, Any], backend: Any, *, seeds: dict[str, Path], diag_inputs: dict[str, Path],
        guard: Callable[[], None], booked_s: float, smoke: bool = False) -> dict[str, Any]:
    m = ft()
    targets = inp["targets"]
    work = {name: out / f"store_{name}.sqlite" for name in seeds}
    for name, src in seeds.items():
        if not work[name].exists():
            P.copy_store(src, work[name])
            rec.emit("store_copied", src=str(src), dst=str(work[name]), sha256=SHA256(work[name]), final=False)
    stores = {name: P.open_store(p) for name, p in work.items()}
    try:
        wins = {name: doc_windows(st) for name, st in stores.items()}
        plans = doc_plans(targets, wins)
        rb = G16.RecordingBackend(backend)
        tbs = {name: m.TypedBackfill(st, rb) for name, st in stores.items()}
        wmap = {(pid, w["chunk_id"]): w for pid, p in plans.items() for w in p["windows"]}
        if smoke:
            pid = min(plans, key=lambda k: (plans[k]["chars"], k))
            w = plans[pid]["windows"][0]
            run_window(rec, tbs[plans[pid]["store"]], rb, pid, plans[pid], w, guard)
            return {"complete": False, "smoke": True, "rc": RC_OK}
        if "before" not in rec.done:
            rec.emit("before", unit="before", m1={n: m1(score_store(seeds[n], inp["cells"][n])) for n in seeds})
        for pid, cid in pilot_window_ids(plans):
            if window_unit(pid, cid) not in rec.done:
                run_window(rec, tbs[plans[pid]["store"]], rb, pid, plans[pid], wmap[(pid, cid)], guard)
        if "pilot_done" not in rec.done:
            missing = [u for u in (window_unit(p, c) for p, c in pilot_window_ids(plans)) if u not in rec.done]
            if missing:
                return {"complete": False, "reason": f"pilot: {len(missing)} windows errored; re-grant retries them", "rc": RC_PARTIAL}
            ev = pilot_eval(rec, plans, targets, booked_s=booked_s)
            rec.emit("pilot_done", unit="pilot_done", **ev)    # COUNTS ONLY
            if not ev["futility"] and ev["budget_stops"]:      # re-book; the next grant continues past the pilot
                rec.emit("stop", reason="budget_stop:" + "+".join(ev["budget_stops"]), final=False)
                return {"complete": False, "budget_stop": "+".join(ev["budget_stops"]), "rc": RC_BUDGET}
        pd = next(r for r in rec.records if r.get("unit") == "pilot_done")
        if pd["futility"]:
            return _result(out, complete=True, futility_stop="pilot", verdict="agg111s2_futility", pilot=pd, rc=RC_GATE)
        for pid, p in plans.items():
            for w in p["windows"]:
                if window_unit(pid, w["chunk_id"]) not in rec.done:
                    run_window(rec, tbs[p["store"]], rb, pid, p, w, guard)
        for pid, p in plans.items():
            if not all(window_unit(pid, w["chunk_id"]) in rec.done for w in p["windows"]):
                return {"complete": False, "reason": f"{pid}: windows errored; re-grant retries them", "rc": RC_PARTIAL}
            if f"doc_l4:{pid}" not in rec.done:
                finish_doc(rec, tbs[p["store"]], pid, p)
    finally:
        for st in stores.values():
            st.close()
    return finish(rec, out, inp, seeds=seeds, work=work, diag_inputs=diag_inputs, pilot=pd)


def finish(rec: P.Recorder, out: Path, inp: dict[str, Any], *, seeds: dict[str, Path], work: dict[str, Path],
           diag_inputs: dict[str, Path], pilot: dict[str, Any]) -> dict[str, Any]:
    cmp_, canon = {}, {}
    for name in seeds:
        cells = inp["cells"][name]
        cmp_[name] = compare(score_store(seeds[name], cells), score_store(work[name], cells))
        cb, ca = canonical_ok(seeds[name], cells), canonical_ok(work[name], cells)
        canon[name] = {"before": sum(cb.values()), "after": sum(ca.values()),
                       "regressed": sorted(k for k in cb if cb[k] and not ca[k]), "fixed": sorted(k for k in ca if ca[k] and not cb[k])}
    tot = {"n": sum(v["n"] for v in cmp_.values()), "before": sum(v["before"] for v in cmp_.values()),
           "after": sum(v["after"] for v in cmp_.values()), "regressed": sum(len(v["regressed"]) for v in cmp_.values())}
    fixed = {k for v in cmp_.values() for k in v["fixed"]}
    tk = [t["key"] for t in inp["targets"]]
    fill = {"of_13": sum(k in fixed for k in tk),
            "of_10_agg13": sum(k in fixed for t, k in zip(inp["targets"], tk) if t["group"] == "a"),
            "of_3_agg14": sum(k in fixed for t, k in zip(inp["targets"], tk) if t["group"] == "b")}
    diag = gap_diagnosis(inp["cells"]["agg14"], agg14_store=seeds["agg14"], **diag_inputs)
    docs = [r for r in rec.records if r.get("kind") == "doc_l4"]
    wins = list(win_records(rec).values())
    cost = {"calls": sum(r["calls"] for r in wins), "prefill_tokens": sum(r["prefill_tokens"] for r in wins),
            "wall_s": round(sum(r["wall_s"] for r in wins), 1), "bad_replies": sum(1 for r in wins if r.get("bad")),
            "quote_dropped": sum(r.get("n_quote_dropped", 0) for r in docs)}
    verdict = "agg111s2_regression" if tot["regressed"] else "agg111s2_done"
    rec.emit("summary", unit="summary", m1=cmp_, total=tot, fill=fill, cost=cost, gap=diag["by_step"])
    return _result(out, complete=True, verdict=verdict, m1=cmp_, m1_total=tot, target_fill=fill, canonical=canon, cost=cost,
                   gap_diagnosis=diag, pilot=pilot, post_hoc=True, in_sample=True, rc=RC_OK)
