#!/usr/bin/env python3
"""AGG-1.9: targeted L4 fill of the 26 never-extracted AGG-1.1 cells plus the canonicalisation guards (library;
``run_agg19.py`` is the CLI).

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an injected
backend (``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in tests).

Implements ``PREDECLARED_AGG19.md`` and ``PREDECLARED_AGG19_AMENDMENT1_20261112.md`` literally. The L4 unit is
``phase1_arms.l4_unit_fy`` (unchanged; the amended ``facts_llm.py`` of orchestrator 1a408ce4 is the stage). Phases:

* ``probe``     zero model: the canonical guard variants over the 211 cells of a store, and the 14 saved store-gap SQLs re-run
                on it (``agg19_guard_probe.json`` / ``agg19_offline_before_fill.json``).
* ``preflight`` zero model: pins, the 26 cells, the L4 reuse test, the probe recomputed and compared with the frozen one.
* ``fill``      CPU: copy the base store, seed the 198 AGG-1.1 units that are NOT targets as cache hits, pilot 5 cells, the other
                21, then M1 before/after, the canonical rule, the 14 SQL re-runs and the retained-cell regression gate.

Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped, 4 precondition, 5 budget stop, 75 partial.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
P0DIR = HERE.parent / "agg_phase0"
for _p in (str(P0DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg_phase0 as A  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
from agg1314_zero_cell_diag import diagnose_cell  # noqa: E402

# --------------------------------------------------------------------------- pins
PREDECLARED_SHA = "4eed298ac95022de532f0a30844ab6d7af684e261e7e1a3e4109ae749ad7917c"
AMENDMENT1_FILE = "PREDECLARED_AGG19_AMENDMENT1_20261112.md"
AMENDMENT1_SHA = "dbe8f858850a76a7ddc9f21a3c27bc3a720f34b2605582a2e8e7f0fc6f75b625"
FROZEN = {
    "agg18_miss_table.json": "97ebad89a70e2802d4d1c12a5418389ee1471ae24709c8e747c42ba05dafa193",
    "agg19_cells.csv": "31dffe2f0255c461bb31d25eb5286b278d34441d24ad5df58180a5bf8e5de08b",
}
PROBE_FILE = "agg19_guard_probe.json"
PROBE_SHA = "dcb716ba02ec4e060dd1faf19cf848d981ed59d032a686ceeba9b6d3610b37a7"
PERIOD = "2022"
N_CELLS, N_GOLD = 26, 211
EXPECTED_BEFORE = {"raw_hits": 185, "canonical_ok_v1": 177}
STATES = {"missing": 20, "wrong": 5, "unit1000": 1}
PILOT_INDEX = (0, 5, 10, 15, 20)
PILOT_MIN_FILLED = 2
PILOT_WALL_S = 300
BOOKED_S = 1400.0                            # amendment 1: measured-rate projection (the box ESTIMATE of 600 s is likely low)
BUDGET_X = 2.0
M1_GATE = X.M1_GATE
TOL = 0.005                                  # the frozen rel_tol / AGG-1.8 TOL
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_PARTIAL = (X.RC_OK, X.RC_PREFLIGHT, X.RC_GATE, X.RC_PRECOND,
                                                                   X.RC_BUDGET, X.RC_PARTIAL)
SHA256 = X.sha256_file
GUARDS = ("thousands_unit", "attributable_net_income")


# --------------------------------------------------------------------------- inputs and pins
def load_inputs(set_dir: Path = HERE) -> dict[str, Any]:
    with (set_dir / "agg19_cells.csv").open(newline="") as fh:
        cells = [{"doc": r["doc_name"], "field": r["metric"], "fy": r["fiscal_year"], "gold": float(r["gold_usd_m"]),
                  "state": r["agg18_state"]} for r in csv.DictReader(fh)]
    corpus = json.loads((set_dir / "corpus.json").read_text())
    base = X.load_inputs(corpus=set_dir / "corpus.json", golds=set_dir / "golds.jsonl",
                         verification=set_dir / "verification_record.json", facts_sheet=set_dir / "facts_sheet.csv")
    cells211 = X.m1_cells(set_dir / "facts_sheet.csv", X.verified_agg(base), [d["doc_name"] for d in corpus["docs"]])
    miss = json.loads((set_dir / "agg18_miss_table.json").read_text())
    golds = {}
    for line in (set_dir / "golds.jsonl").read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            golds[g["id"]] = g
    store_gap = [r["qid"] for r in miss["rows"] if r["category"] == "store_gap_m1"]
    return {"cells": sorted(cells, key=lambda c: (c["doc"], c["field"])), "corpus": corpus, "cells211": cells211,
            "golds": golds, "store_gap_qids": store_gap, "miss": miss}


def pilot_cells(inp: dict[str, Any]) -> list[dict[str, Any]]:
    return [inp["cells"][i] for i in PILOT_INDEX]


def unit_of(c: dict[str, Any]) -> str:
    return f"l4:{c['field']}:{c['doc']}"


def pin_problems(set_dir: Path = HERE, *, predeclared_sha: str = PREDECLARED_SHA,
                 amendment1_sha: str = AMENDMENT1_SHA, probe_sha: str = PROBE_SHA,
                 frozen: dict[str, str] | None = None) -> list[str]:
    out = []
    for name, want in ((("PREDECLARED_AGG19.md", predeclared_sha), (AMENDMENT1_FILE, amendment1_sha),
                        (PROBE_FILE, probe_sha))):
        p = set_dir / name
        got = SHA256(p) if p.exists() else None
        if got != want:
            out.append(f"{name} sha256 {got} != recorded {want}")
    for name, want in (FROZEN if frozen is None else frozen).items():
        p = set_dir / name
        got = SHA256(p) if p.exists() else None
        if got != want:
            out.append(f"frozen input {name} sha256 {got} != pinned {want}")
    return out


def cells_problems(inp: dict[str, Any], *, expect: bool = True) -> list[str]:
    out = []
    cells = inp["cells"]
    keys = [(c["doc"], c["field"]) for c in cells]
    if len(keys) != len(set(keys)):
        out.append("duplicate (document, field) cells")
    docs = {d["doc_name"] for d in inp["corpus"]["docs"]}
    for c in cells:
        if c["doc"] not in docs:
            out.append(f"{c['doc']}: not a corpus document")
        if c["field"] not in A.METRICS:
            out.append(f"{c['field']}: no L4 description in the frozen METRICS table")
        if str(c["fy"]) != PERIOD:
            out.append(f"{c['doc']}|{c['field']}: fiscal year {c['fy']} != {PERIOD}")
    g211 = {(d, f): v for d, f, v in inp["cells211"]}
    for c in cells:
        gv = g211.get((c["doc"], c["field"]))
        if gv is None or abs(gv - c["gold"]) > 1e-6 * max(1.0, abs(gv)):
            out.append(f"{c['doc']}|{c['field']}: gold {c['gold']} is not the verified 211-cell value {gv}")
    if expect:
        if len(cells) != N_CELLS:
            out.append(f"{len(cells)} cells, PREDECLARED expects {N_CELLS}")
        by_state: dict[str, int] = {}
        for c in cells:
            by_state[c["state"]] = by_state.get(c["state"], 0) + 1
        if by_state != STATES:
            out.append(f"cell states {by_state} != {STATES}")
        if len(inp["cells211"]) != N_GOLD:
            out.append(f"{len(inp['cells211'])} verified cells, PREDECLARED expects {N_GOLD}")
        if len(inp["store_gap_qids"]) != 14:
            out.append(f"{len(inp['store_gap_qids'])} store-gap questions, expected 14")
    return out


def l4_reuse_check19(agg11_job: Path, orch_root: Path, research_arms: Path | None = None) -> dict[str, Any]:
    """The AGG-1.1 reuse rule against the AGG-1.1 job's pin, with the amended L4 stage file expected at ``X.AMENDED_L4``."""
    res = X.l4_reuse_check(agg11_job, orch_root, research_arms, amended=X.AMENDED_L4)
    res["against"] = ("AGG-1.1 job pin (" + str(agg11_job) + "), L4 stage as amended 2026-11-12 (AGG-1.3/1.4 amendment 2)")
    return res


# --------------------------------------------------------------------------- pure measurement
def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(abs(a) - abs(b)) <= tol * abs(b) + 1e-9


def fact_rows(conn: sqlite3.Connection, corpus_id: str) -> list[dict[str, Any]]:
    from src.aggregation.store import FACT_COLUMNS  # noqa: PLC0415
    cur = conn.execute(f"SELECT d.corpus_id, {', '.join('f.' + c for c in FACT_COLUMNS)} FROM facts f JOIN documents d "
                       "ON f.doc_hash = d.doc_hash WHERE d.corpus_id = ?", [corpus_id])
    return [dict(zip(("corpus_id",) + FACT_COLUMNS, r)) for r in cur.fetchall()]


def canon_ok(canon: list[dict[str, Any]], cells: list[tuple[str, str, float]]) -> dict[str, bool]:
    """Per verified cell: is the canonical FY2022 value within tolerance of the gold (the AGG-1.8 ``cell_state`` rule)?"""
    by = {(r["doc_id"], r["field"]): r["value"] for r in canon if str(r["period"]) == PERIOD}
    return {f"{d}|{f}": (by.get((d, f)) is not None and _close(by[(d, f)], g)) for d, f, g in cells}


def raw_hits(rows: list[dict[str, Any]], cells: list[tuple[str, str, float]]) -> dict[str, bool]:
    hits = X.m1_hits(rows, cells, unit_scale=X.M1R_UNIT_SCALE)
    return {f"{d}|{f}": bool(h) for (d, f, _g), h in zip(cells, hits)}


def variant_table(rows: list[dict[str, Any]], cells: list[tuple[str, str, float]]) -> dict[str, Any]:
    """The canonical variants over the cells, ``canonical.v1`` as the control: ok counts, fixed and broken cells."""
    from src.aggregation.store import CANONICAL_RULE, CANONICAL_RULE_V2, select_canonical  # noqa: PLC0415
    base = canon_ok(select_canonical(rows)[0], cells)
    out: dict[str, Any] = {"control": {"rule": CANONICAL_RULE, "n_cells": len(cells), "n_ok": sum(base.values())},
                           "variants": {}}
    for name, g in (("attributable_net_income", ("attributable_net_income",)), ("thousands_unit", ("thousands_unit",)),
                    ("both", GUARDS)):
        ok = canon_ok(select_canonical(rows, rule=CANONICAL_RULE_V2, guards=g)[0], cells)
        fixed = sorted(k for k in ok if ok[k] and not base[k])
        broken = sorted(k for k in ok if base[k] and not ok[k])
        by_field: dict[str, dict[str, int]] = {}
        for k in fixed:
            by_field.setdefault(k.split("|")[1], {"fixed": 0, "broken": 0})["fixed"] += 1
        for k in broken:
            by_field.setdefault(k.split("|")[1], {"fixed": 0, "broken": 0})["broken"] += 1
        out["variants"][name] = {"guards": list(g), "n_ok": sum(ok.values()), "fixed": fixed, "broken": broken,
                                 "by_field": by_field, "adopted": bool(fixed) and not broken}
    out["adopted_guards"] = [g for g in GUARDS if out["variants"][g]["adopted"]]
    return out


def adopted_guards(probe: dict[str, Any]) -> tuple[str, ...]:
    return tuple(probe.get("adopted_guards") or ())


def apply_rule(store, corpus_id: str, guards: tuple[str, ...]) -> dict[str, Any]:
    """Write ``facts_canonical`` under the adopted rule: ``canonical.v2`` with the adopted guards, else ``canonical.v1``."""
    from src.aggregation.store import CANONICAL_RULE, CANONICAL_RULE_V2  # noqa: PLC0415
    if guards:
        return store.refresh_canonical(corpus_id, rule=CANONICAL_RULE_V2, guards=guards)
    return store.refresh_canonical(corpus_id, rule=CANONICAL_RULE)


# --------------------------------------------------------------------------- the 14 saved store-gap SQLs
def sql_scores(rows: list[Any] | None, q: dict[str, Any]) -> bool:
    """The AGG-1.8 SQL scorer (``agg18_miss_table.sql_scores``): first-row string match for argmax/argmin, the first three
    names for top3, exact count, 0.5 percent relative tolerance on a sum."""
    if not rows:
        return False
    op, gold = q["op"], q["gold"]
    try:
        if op in ("argmax", "argmin"):
            return str(rows[0][0]).lower() == str(gold).lower()
        if op == "top3":
            return [str(r[0]).lower() for r in rows[:3]] == [str(g).lower() for g in gold]
        v, g = float(rows[0][0]), float(gold)
        return abs(v - g) <= (0 if op.startswith("count") else TOL * abs(g)) + 1e-9
    except Exception:  # noqa: BLE001
        return False


def saved_sql(answers: Path, qids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in X.read_jsonl(answers):
        if r.get("kind") == "answer" and r.get("arm") == "P-large" and r.get("final") and r.get("qid") in qids:
            out[r["qid"]] = r
    return out


def replay_sql(conn: sqlite3.Connection, saved: dict[str, dict[str, Any]], golds: dict[str, dict[str, Any]]
               ) -> dict[str, Any]:
    """Re-execute each saved SQL string on ``conn``'s ``facts_canonical`` (no model) and score it."""
    per = {}
    for qid in sorted(saved):
        sql = saved[qid].get("sql") or ""
        try:
            rows = conn.execute(sql).fetchall()
            err = None
        except sqlite3.Error as e:
            rows, err = None, f"{type(e).__name__}: {e}"
        per[qid] = {"sql": sql, "correct": sql_scores(rows, golds[qid]), "error": err,
                    "first_rows": [list(r) for r in (rows or [])[:3]], "gold": golds[qid]["gold"],
                    "aggregate_op": golds[qid]["op"], "was_correct_in_agg11": saved[qid].get("correct") is True}
    return {"n": len(per), "n_correct": sum(1 for v in per.values() if v["correct"]), "per_question": per}


def gold_corrected_upper_bound() -> dict[str, Any]:
    """From AGG-1.8 (``agg18_miss_table.json``): the saved SQL on gold-corrected cells flips 14 of 14."""
    return {"n": 14, "n_correct": 14, "source": "agg18_miss_table.json rows rescore_on_gold_cells_correct"}


# --------------------------------------------------------------------------- probe phase (zero model)
def probe_base(base_store: Path, inp: dict[str, Any], answers: Path | None, corpus_id: str) -> dict[str, Any]:
    c = sqlite3.connect(f"file:{base_store}?immutable=1", uri=True)
    try:
        rows = fact_rows(c, corpus_id)
        out = variant_table(rows, inp["cells211"])
        before_hits = raw_hits(rows, inp["cells211"])
        out["raw_hits"] = {"found": sum(before_hits.values()), "n": len(before_hits),
                           "m1": sum(before_hits.values()) / len(before_hits)}
        if answers is not None and answers.exists():
            saved = saved_sql(answers, inp["store_gap_qids"])
            out["saved_sql_on_agg11_store"] = replay_sql(c, saved, inp["golds"])
            from src.aggregation.store import CANONICAL_RULE_V2, select_canonical  # noqa: PLC0415
            g = adopted_guards(out)
            mem = sqlite3.connect(":memory:")
            c.backup(mem)
            mem.execute("DELETE FROM facts_canonical")
            canon, _ = select_canonical(rows, rule=CANONICAL_RULE_V2, guards=g) if g else select_canonical(rows)
            from src.aggregation.store import CANONICAL_COLUMNS  # noqa: PLC0415
            for r in canon:
                mem.execute(f"INSERT INTO facts_canonical ({', '.join(CANONICAL_COLUMNS)}) "
                            f"VALUES ({', '.join('?' for _ in CANONICAL_COLUMNS)})", [r[k] for k in CANONICAL_COLUMNS])
            out["saved_sql_on_adopted_rule_before_fill"] = {"guards": list(g), **replay_sql(mem, saved, inp["golds"])}
            mem.close()
        out["agg11_cell_decomposition"] = inp["miss"]["cell_decomposition"]
        return out
    finally:
        c.close()


# --------------------------------------------------------------------------- CPU phase
def cell_found(conn: sqlite3.Connection, c: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    facts = [{"doc_id": c["doc"], "field": c["field"], **r} for r in (
        {"period": p, "value": v, "unit": u, "extractor": e, "page": pg, "quote": q}
        for p, v, u, e, pg, q in conn.execute(
            "SELECT period, value, unit, extractor, page, quote FROM facts WHERE doc_id = ? AND field = ?",
            [c["doc"], c["field"]]))]
    hit = X.m1_hits(facts, [(c["doc"], c["field"], c["gold"])], unit_scale=X.M1R_UNIT_SCALE)[0]
    diag = diagnose_cell({"doc": c["doc"], "field": c["field"], "gold": c["gold"], "fy": PERIOD},
                         [{k: f[k] for k in ("period", "value", "unit", "extractor", "page", "quote")} for f in facts])
    return bool(hit), {"class": diag["verdict"], "candidate": diag["candidate"]}


def seed_units19(rec: P.Recorder, agg11_results: Path, targets: set[str]) -> dict[str, Any]:
    """Seed every AGG-1.1 L4 unit as a cache hit EXCEPT the target units, which run again."""
    ok = X.l4_ok(X.read_jsonl(agg11_results))
    n = 0
    for unit, r in sorted(ok.items()):
        if unit in targets or unit in rec.done:
            continue
        rec.emit("l4_unit", **{k: v for k, v in r.items() if k not in ("kind", "ts", "reused", "reused_from")},
                 reused=True, reused_from=str(agg11_results))
        n += 1
    seeded = [r for r in rec.records if r["kind"] == "l4_unit" and r.get("reused")]
    return {"seeded": n, "units": len(seeded), "targets_not_seeded": len(targets & set(ok)), **X.tally(seeded)}


def _result(out: Path, **res: Any) -> dict[str, Any]:
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = PREDECLARED_SHA
    X._atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    return res


def _gate_stop(rec: P.Recorder, out: Path, gates: list[str], where: str, **detail: Any) -> dict[str, Any]:
    rec.emit("gate_tripped", unit=f"gate_tripped:{where}", gates=gates, where=where, **detail)
    rec.emit("stop", reason="phase1_gate_tripped", gates=gates, where=where, final=False)
    res = _result(out, complete=True, futility_stop="+".join(gates), gate_tripped=gates, verdict="phase1_gate_tripped",
                  where=where, rc=RC_GATE, **detail)
    res["rc"] = RC_GATE
    return res


def pilot_eval(filled: int, n: int, wall_s: float, n_units: int) -> dict[str, Any]:
    budget = []
    if wall_s > PILOT_WALL_S:
        budget.append("pilot_wall_over_300s")
    proj = wall_s / n * n_units if n else 0.0
    if proj > BUDGET_X * BOOKED_S:
        budget.append("projected_wall_over_2x_booked")
    return {"filled": filled, "n": n, "wall_s": wall_s, "projected_wall_s": proj,
            "gates_tripped": ["pilot_fills_lt_2_of_5"] if filled < PILOT_MIN_FILLED else [], "budget_stops": budget}


def run_fill(rec: P.Recorder, out: Path, inp: dict[str, Any], orch_root: Path, backend, *, base_store: Path,
             agg11_results: Path, agg11_answers: Path | None, guard: Callable[[], None], probe: dict[str, Any],
             smoke: bool = False, corpus_id: str | None = None, reuse_ok: bool = True,
             cfg: P.Cfg = P.CFG) -> dict[str, Any]:
    corpus_id = corpus_id or X.CORPUS_ID
    if not reuse_ok:
        rec.emit("precondition_failed", reason="L4 reuse test failed: the earlier units are not cache hits")
        return {"complete": False, "rc": RC_PRECOND}
    store_path = out / "store.sqlite"
    if not store_path.exists():
        if not base_store.exists():
            rec.emit("precondition_failed", reason=f"no base store {base_store}")
            return {"complete": False, "rc": RC_PRECOND}
        P.copy_store(base_store, store_path)
        rec.emit("store_copied", src=str(base_store), sha256=SHA256(store_path), final=False)
    store = P.open_store(store_path)
    try:
        cells, cells211 = inp["cells"], inp["cells211"]
        targets = {unit_of(c) for c in cells}
        if "reuse_seeded" not in rec.done:
            rec.emit("reuse_seeded", unit="reuse_seeded", **seed_units19(rec, agg11_results, targets))
        if "before" not in rec.done:
            rows0 = fact_rows(store.conn, corpus_id)
            from src.aggregation.store import select_canonical  # noqa: PLC0415
            h0 = raw_hits(rows0, cells211)
            c0 = canon_ok(select_canonical(rows0)[0], cells211)
            rec.emit("before", unit="before", raw_found=sum(h0.values()), raw_m1=sum(h0.values()) / len(h0),
                     canonical_ok_v1=sum(c0.values()), raw_hit_cells=sorted(k for k, v in h0.items() if v),
                     canonical_ok_cells=sorted(k for k, v in c0.items() if v), n_facts=len(rows0))
        before = next(r for r in rec.records if r.get("unit") == "before")
        pilot = pilot_cells(inp)
        rest = [c for c in cells if c not in pilot]
        order = pilot + rest
        for i, c in enumerate(order):
            unit = unit_of(c)
            if unit not in rec.done:
                units = P.windows(store, corpus_id, cfg.window_chars)
                X.l4_unit_fy(rec, store, backend, units, c["field"], c["doc"], guard, cfg, fy=PERIOD)
                if unit not in rec.done:
                    return {"complete": False, "reason": f"{unit} errored; re-grant retries it", "rc": RC_PARTIAL}
                found, diag = cell_found(store.conn, c)
                rec.emit("cell_fill", unit=f"cellfill:{c['field']}:{c['doc']}", doc=c["doc"], field=c["field"],
                         gold=c["gold"], state_before=c["state"], found=found, **diag)
                if smoke:
                    return {"complete": False, "smoke": True, "rc": RC_OK}
            if i == len(pilot) - 1 and "pilot_done" not in rec.done:
                fills = {r["unit"]: r for r in rec.records if r.get("kind") == "cell_fill"}
                filled = sum(1 for p in pilot if fills[f"cellfill:{p['field']}:{p['doc']}"]["found"])
                wall = sum((X.l4_ok(rec.records)[unit_of(p)].get("wall_s") or 0.0) for p in pilot)
                ev = pilot_eval(filled, len(pilot), wall, len(order))
                rec.emit("pilot_done", unit="pilot_done", **ev)
                if ev["gates_tripped"]:
                    return _gate_stop(rec, out, ev["gates_tripped"], "pilot", pilot=ev)
                if ev["budget_stops"]:
                    rec.emit("stop", reason="budget_stop:" + "+".join(ev["budget_stops"]), final=False, pilot=ev)
                    return {"complete": False, "budget_stop": "+".join(ev["budget_stops"]), "rc": RC_BUDGET, **ev}
        return _finish(rec, out, store, inp, corpus_id, before, probe, agg11_answers, order)
    finally:
        store.close()


def regressions(before: dict[str, Any], after_ok: dict[str, bool], after_hits: dict[str, bool]) -> dict[str, Any]:
    """Retained cells that were right before and are not right after, on both measures (PREDECLARED DONE WHEN 3)."""
    can = sorted(k for k in before["canonical_ok_cells"] if not after_ok.get(k))
    raw = sorted(k for k in before["raw_hit_cells"] if not after_hits.get(k))
    return {"ok": not can and not raw, "canonical_regressed": can, "raw_regressed": raw,
            "n_retained_canonical": len(before["canonical_ok_cells"]), "n_retained_raw": len(before["raw_hit_cells"])}


def _finish(rec, out, store, inp, corpus_id, before, probe, agg11_answers, order) -> dict[str, Any]:
    from src.aggregation.store import CANONICAL_RULE, select_canonical  # noqa: PLC0415
    ok = X.l4_ok(rec.records)
    fresh_units = [r for r in ok.values() if not r.get("reused")]
    fresh = X.tally(fresh_units)
    cells211 = inp["cells211"]
    rows = fact_rows(store.conn, corpus_id)
    hits = raw_hits(rows, cells211)
    v1 = canon_ok(select_canonical(rows)[0], cells211)
    guards = adopted_guards(probe)
    var = variant_table(rows, cells211)          # the guard variants again, on the FILLED store (informational)
    stats = apply_rule(store, corpus_id, guards)
    after_ok = canon_ok(store.canonical_facts(corpus_id), cells211)
    reg = regressions(before, after_ok, hits)
    fills = {r["unit"]: r for r in rec.records if r.get("kind") == "cell_fill"}
    per_cell = [{"doc": c["doc"], "field": c["field"], "gold": c["gold"], "state_before": c["state"],
                 "filled": fills.get(f"cellfill:{c['field']}:{c['doc']}", {}).get("found"),
                 "class": fills.get(f"cellfill:{c['field']}:{c['doc']}", {}).get("class"),
                 "canonical_v1": v1[f"{c['doc']}|{c['field']}"], "canonical_adopted": after_ok[f"{c['doc']}|{c['field']}"]}
                for c in inp["cells"]]
    n_filled = sum(1 for p in per_cell if p["filled"])
    m1 = {"before": {"found": before["raw_found"], "n": len(cells211), "m1": before["raw_m1"]},
          "after": {"found": sum(hits.values()), "n": len(cells211), "m1": sum(hits.values()) / len(cells211)},
          "gate": M1_GATE, "gained": sum(hits.values()) - before["raw_found"],
          "canonical_ok": {"before_v1": before["canonical_ok_v1"], "after_v1": sum(v1.values()),
                           "after_adopted": sum(after_ok.values()), "adopted_guards": list(guards),
                           "rule": stats["rule"]}}
    sql14 = None
    if agg11_answers is not None and agg11_answers.exists():
        saved = saved_sql(agg11_answers, inp["store_gap_qids"])
        sql14 = {**replay_sql(store.conn, saved, inp["golds"]), "rule": stats["rule"], "guards": list(guards),
                 "agg11_correct": sum(1 for r in saved.values() if r.get("correct") is True),
                 "gold_corrected_upper_bound": gold_corrected_upper_bound()}
    rec.emit("gates_cpu", unit="gates_cpu", m1=m1, regressions=reg, sql14=sql14, n_filled=n_filled, per_cell=per_cell,
             variants_after_fill=var, canonical=stats, fresh=fresh)
    tripped = [] if reg["ok"] else ["retained_cell_regression"]
    return _result(out, complete=True, futility_stop=None, gate_tripped=tripped,
                   verdict="phase1_gate_tripped" if tripped else "agg_1_9_done", m1=m1["after"]["m1"], m1_detail=m1,
                   regressions=reg, sql14=sql14, n_filled=n_filled, n_cells=len(inp["cells"]), per_cell=per_cell,
                   variants_after_fill=var, canonical=stats, fresh=fresh, order=[unit_of(c) for c in order],
                   default_canonical_rule=CANONICAL_RULE, rc=RC_GATE if tripped else RC_OK)
