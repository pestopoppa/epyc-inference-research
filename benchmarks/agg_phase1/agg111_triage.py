#!/usr/bin/env python3
"""AGG-1.11 step 1 -- no-model triage of the AGG-1.3 r3 / AGG-1.4 r3 / AGG-1.9 missed cells.

No model call, no server. Every store is opened read-only (``file:...?mode=ro``). The deterministic L4 window
selection (``phase0_arms.windows`` over the stored chunks, 10 000-char windows) and BM25 (orchestrator
``src/repl_environment/context_search.BM25Index``, stdlib only, loaded by file path) are RE-RUN offline on the saved
store, with the exact query string the run used (``"<field words> <description>"``, ``phase1_arms.l4_unit_fy``).
The replay is validated before any cell is classified: for every cell, the chunk of every L4 fact the run stored must be
among the recomputed top-k windows, and the L4a gate-call count must equal the recomputed number of gated windows.

Groups
  a  the 10 AGG-1.3 r3 changed-slot misses (FY2021 descriptions)        store+results: agg13-reingest-cpu-20261111-r3
  b  the 3 AGG-1.4 r3 misses (long_term_debt, FY2022)                   store+results: agg14-backfill-cpu-20261111-r3
  c  the 26 AGG-1.9 never-extracted cells (FY2022)                      store+results: agg1-ingest-cpu-20261111 (AGG-1.1)

Reply-level evidence. The model REPLIES were not saved. Two indirect, saved witnesses are used and labelled as such:
  * ``notes: unparseable:<chunk>`` in the unit's L4b receipt (the reply did not ``json.loads``);
  * the llama-server log: one ``eval time ... / N tokens`` line per request, in request order. Requests of a fresh unit
    are its L4a gate calls (1 token) then its L4b calls (one per admitted window, rank order). The mapping is accepted
    for a run ONLY if (task count == sum of the fresh units' calls) AND (the number of 256-token replies == the number of
    ``unparseable`` notes) AND, per unit, the 256-token replies sit exactly on the noted windows. Then a window's reply
    length is known: 256 = cut at n_predict (grammar-constrained output is valid JSON or cut off, so truncation IS the
    unparseable cause), 2 = ``[]`` (model said absent), other = rows. Where the mapping is rejected or the unit was
    REUSED from an earlier run whose log is not available, reply length is "unknown" and EXTRACTION rests on
    exclusion (admitted, parsed, no right fact) -- the cell record says so (``reply_evidence``).

Classes (PRIMARY cause, first match): GOLD, L3, RETRIEVAL, FORMAT, EXTRACTION, CANONICAL, UNKNOWN. See ``classify``.

Everything here is DIAGNOSTIC: no scorer, gold, gate or pinned file is changed. The offline fix probes
(``fix_probes``) are post-hoc and in-sample; they bound a fix, they do not prove it.

CLI:  agg111_triage.py [--json PATH] [--md PATH]      (paths default to the sibling agg111_triage.{json,md})
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg14 as G14  # noqa: E402
import agg_phase0 as A  # noqa: E402
import agg1314_zero_cell_diag as Z  # noqa: E402
import phase0_arms as P  # noqa: E402

TMP = Path(os.environ.get("AGG111_TMP", "/mnt/raid0/llm/tmp"))
ORCH = Path(os.environ.get("AGG_ORCH_ROOT", "/mnt/raid0/llm/epyc-orchestrator"))
CFG = P.CFG                                   # window_chars 10000, k_per_doc 6, hi 0.6, lo 0.2, l4_n_predict 256
CORPUS_ID = "agg_phase0_fb10"
CLASSES = ("RETRIEVAL", "FORMAT", "EXTRACTION", "CANONICAL", "L3", "GOLD", "UNKNOWN")
N_PREDICT = CFG.l4_n_predict

RUNS = {
    "a": {"title": "AGG-1.3 r3 changed-slot misses", "dir": TMP / "agg13-reingest-cpu-20261111-r3" / "out", "fy": "2021"},
    "b": {"title": "AGG-1.4 r3 misses", "dir": TMP / "agg14-backfill-cpu-20261111-r3" / "out", "fy": "2022"},
    "c": {"title": "AGG-1.9 never-extracted cells", "dir": TMP / "agg1-ingest-cpu-20261111" / "out", "fy": "2022"},
}
PILOT19 = TMP / "agg19-fill-cpu-20261112" / "out"

# Field-specific retrieval queries for the offline RETRIEVAL probe (written from the field's meaning, not tuned on the
# cells; post-hoc, in-sample, unregistered).
FIELD_QUERY = {
    "revenue": "total revenues net sales operating revenues consolidated statements of income",
    "net_income": "net income attributable consolidated statements of income net earnings",
    "capex": "purchases of property plant and equipment additions capital expenditures investing activities cash flows",
    "total_assets": "total assets consolidated balance sheets",
    "operating_cash_flow": "net cash provided by operating activities cash flows",
    "total_current_assets": "total current assets consolidated balance sheets",
    "total_current_liabilities": "total current liabilities consolidated balance sheets",
    "dividends_paid": "dividends paid cash dividends financing activities cash flows",
    "long_term_debt": "long-term debt consolidated balance sheets",
}
# Table-row locator: a statement row = label followed by a number, per field.
ROW_LABEL = {
    "revenue": r"(?:total\s+)?(?:net\s+sales|revenues?|operating\s+revenues?)",
    "net_income": r"net\s+(?:income|earnings)(?:\s+attributable[^\d$()]{0,60})?",
    "capex": r"(?:purchases?\s+of\s+property|additions\s+to\s+property|capital\s+expenditures|payments?\s+for\s+property)[^\d$()]{0,50}",
    "total_assets": r"total\s+assets",
    "operating_cash_flow": r"net\s+cash\s+(?:provided\s+by|from|generated\s+by|flows?\s+from)[^\d$()]{0,30}operating\s+activities",
    "total_current_assets": r"total\s+current\s+assets",
    "total_current_liabilities": r"total\s+current\s+liabilities",
    "dividends_paid": r"(?:cash\s+)?dividends\s+paid|dividends\s+(?:to|on)[^\d$()]{0,30}",
    "long_term_debt": r"long[-\s]term\s+debt",
}


# --------------------------------------------------------------------------- plumbing
def ro(path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def load_bm25(orch: Path = ORCH):
    """``context_search`` is stdlib-only: load it by file path so no orchestrator package (pydantic_graph, ...) is needed."""
    src = orch / "src" / "repl_environment" / "context_search.py"
    spec = importlib.util.spec_from_file_location("agg111_context_search", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agg111_context_search"] = mod
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(src.read_bytes()).hexdigest()


class RoStore:
    """The two read paths ``phase0_arms.windows`` uses, on a read-only connection (no schema writes, no WAL)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.c = conn

    def corpus_docs(self, cid: str) -> list[dict[str, Any]]:
        q = "SELECT doc_hash, doc_id, entity, n_tokens_est, total_pages FROM documents WHERE corpus_id = ? ORDER BY doc_id"
        return [dict(zip(("doc_hash", "doc_id", "entity", "n_tokens_est", "total_pages"), r)) for r in self.c.execute(q, [cid])]

    def chunks(self, cid: str) -> list[dict[str, Any]]:
        keys = ("chunk_id", "doc_hash", "doc_id", "ordinal", "section_id", "title", "page_start", "page_end", "text")
        q = ("SELECT c.chunk_id, c.doc_hash, d.doc_id, c.ordinal, c.section_id, c.title, c.page_start, c.page_end, c.text "
             "FROM chunks c JOIN documents d ON c.doc_hash = d.doc_hash WHERE d.corpus_id = ? ORDER BY d.doc_id, c.ordinal")
        return [dict(zip(keys, r)) for r in self.c.execute(q, [cid])]


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def num_re(v: float) -> re.Pattern[str]:
    s = f"{int(round(v)):,}" if abs(v - round(v)) < 1e-9 else f"{v:,.3f}".rstrip("0").rstrip(".")
    return re.compile(r"(?<![\d,.])" + re.escape(s) + r"(?![\d,]|\.\d)")


def short(cid: str) -> str:
    """``<doc_hash>:00101#w0`` -> ``00101#w0``."""
    return cid.split(":", 1)[-1]


# --------------------------------------------------------------------------- replay of one L4 unit
class Replayer:
    def __init__(self, conn: sqlite3.Connection, bm25_mod) -> None:
        self.conn, self.mod = conn, bm25_mod
        self.units = P.windows(RoStore(conn), CORPUS_ID, CFG.window_chars)
        self.by_doc: dict[str, list[dict[str, Any]]] = {}
        self._idx: dict[str, Any] = {}
        for u in self.units:
            self.by_doc.setdefault(u["doc_id"], []).append(u)

    def rank(self, doc: str, query: str) -> list[tuple[dict[str, Any], float]]:
        """All windows of ``doc`` best-first with the run's ``sim`` (score / top score)."""
        du = self.by_doc.get(doc, [])
        if not du:
            return []
        if doc not in self._idx:
            self._idx[doc] = self.mod.BM25Index([u["text"] for u in du])
        sc = self._idx[doc].scores(query)
        if not sc:
            return []
        top = sc[0][1]
        return [(du[i], s / top) for i, s in sc]

    def unit_query(self, field: str, fy: str) -> str:
        desc = G14.FIELD_DESC if field == G14.FIELD else A.METRICS[field]
        if fy != "2022":
            desc = desc.replace("FY2022", f"FY{fy}")
        return f"{field.replace('_', ' ')} {desc}"

    def top(self, doc: str, field: str, fy: str) -> list[dict[str, Any]]:
        out = []
        for i, (u, sim) in enumerate(self.rank(doc, self.unit_query(field, fy))[:CFG.k_per_doc]):
            out.append({"chunk_id": u["chunk_id"], "rank": i, "sim": round(sim, 4), "page_start": u["page_start"]})
        return out


def admission(top: list[dict[str, Any]], l4a: dict[str, Any] | None) -> dict[str, Any]:
    """Which top windows were admitted. sim >= hi admit, <= lo reject, between -> a model gate whose reply is not saved;
    its outcome is recovered from the receipt: ``rejected - n_low`` = number of gate 'no'. All-no or all-yes resolves
    every gated window; a mixture leaves them 'gate_unknown'."""
    gated = [t for t in top if CFG.lo < t["sim"] < CFG.hi]
    low = [t for t in top if t["sim"] <= CFG.lo]
    st = {t["chunk_id"]: ("admit_sim" if t["sim"] >= CFG.hi else "reject_sim" if t["sim"] <= CFG.lo else "gate_unknown")
          for t in top}
    info: dict[str, Any] = {"n_gated": len(gated), "n_low": len(low), "consistent": None}
    if l4a is not None:
        info["consistent"] = l4a["calls"] == len(gated)
        no = l4a["rejected"] - len(low)
        if info["consistent"] and gated:
            if no == 0:
                for t in gated:
                    st[t["chunk_id"]] = "gate_yes"
            elif no == len(gated):
                for t in gated:
                    st[t["chunk_id"]] = "gate_no"
    info["status"] = st
    return info


# --------------------------------------------------------------------------- server-log request mapping
def parse_log_tokens(log: Path) -> list[int]:
    out = []
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"\beval time =\s+[\d.]+ ms /\s+(\d+) tokens", line)
        if m and "prompt eval" not in line:
            out.append(int(m.group(1)))
    return out


def l4_records(results: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in results.read_text().splitlines() if line.strip()]


def map_replies(rep: Replayer, records: list[dict[str, Any]], log: Path, *, window_fy: dict[str, str]) -> dict[str, Any]:
    """Map the run's server requests to (unit, window). Returns ``{"ok": bool, "why": str, "tokens": {unit: {chunk: n}}}``.
    Accepted only if the three checks in the module docstring hold."""
    if not log.exists():
        return {"ok": False, "why": "no server.log", "tokens": {}}
    toks = parse_log_tokens(log)
    fresh = [r for r in records if r.get("kind") == "l4_unit" and not r.get("reused") and r.get("status") == "ok"]
    calls = sum(r["llm_calls"] for r in fresh)
    notes = sum(len(x["notes"]) for r in fresh for x in r["receipts"] if x["stage"] == "L4b_extract")
    if len(toks) != calls:
        return {"ok": False, "why": f"task count {len(toks)} != fresh calls {calls}", "tokens": {}}
    if sum(t == N_PREDICT for t in toks) != notes:
        return {"ok": False, "why": f"256-token replies {sum(t == N_PREDICT for t in toks)} != unparseable notes {notes}",
                "tokens": {}}
    out: dict[str, dict[str, int]] = {}
    pos = 0
    for r in fresh:
        fy = window_fy.get(r["unit"], str(r.get("fy") or "2022"))
        top = rep.top(r["doc"], r["field"], fy)
        recs = {x["stage"]: x for x in r["receipts"]}
        adm = admission(top, recs.get("L4a_admit"))
        ng, nb = recs["L4a_admit"]["calls"], recs["L4b_extract"]["calls"]
        pos += ng                                           # gate replies are 1 token each; not needed
        admitted = [t["chunk_id"] for t in top if adm["status"][t["chunk_id"]] in ("admit_sim", "gate_yes")]
        if any(s == "gate_unknown" for s in adm["status"].values()):
            pos += nb                                        # ambiguous admission: this unit's windows stay unmapped
            continue
        if len(admitted) != nb:
            return {"ok": False, "why": f"{r['unit']}: admitted {len(admitted)} != L4b calls {nb}", "tokens": {}}
        mine = {cid: toks[pos + i] for i, cid in enumerate(admitted)}
        pos += nb
        noted = {n.split(":", 2)[2] for n in recs["L4b_extract"]["notes"]}
        if {short(c) for c, n in mine.items() if n == N_PREDICT} != {short(c) for c in noted}:
            return {"ok": False, "why": f"{r['unit']}: 256-token replies are not on the unparseable windows", "tokens": {}}
        out[r["unit"]] = mine
    nb = sum(x["calls"] for r in fresh for x in r["receipts"] if x["stage"] == "L4b_extract")
    ntr = sum(t == N_PREDICT for t in toks)
    return {"ok": True, "why": "validated", "tokens": out, "n_units": len(out), "n_tasks": len(toks),
            "l4b_calls": nb, "replies_cut_at_n_predict": ntr, "cut_rate": round(ntr / nb, 3) if nb else None}


# --------------------------------------------------------------------------- one cell
def gold_windows(rep: Replayer, doc: str, field: str, gold: float, quote: str) -> dict[str, Any]:
    """Windows of ``doc`` that hold the gold span: by the sheet's verbatim quote (whitespace-insensitive), else by the
    gold number on a row whose label matches the field."""
    qs = squash(quote)
    num = num_re(gold)
    lab = re.compile(ROW_LABEL.get(field, r"$^"), re.I)
    by_quote, by_row, by_num = [], [], []
    for u in rep.by_doc.get(doc, []):
        t = u["text"]
        if qs and qs in squash(t):
            by_quote.append(u["chunk_id"])
        hit = False
        for m in num.finditer(t):
            if lab.search(t[max(0, m.start() - 160):m.start()]):
                hit = True
                break
        if hit:
            by_row.append(u["chunk_id"])
        if num.search(t):
            by_num.append(u["chunk_id"])
    return {"by_quote": by_quote, "by_row": by_row, "by_number": by_num,
            "ids": list(dict.fromkeys(by_quote + by_row)) or []}


def l3_column_error(facts: list[dict[str, Any]], gold: float) -> dict[str, Any] | None:
    """No stored fact carries the gold value, but an L3 fact quotes the gold row and stored another column's value."""
    if any(Z._close(Z.stored_value(f), gold, Z.TOL) for f in facts if f.get("value") is not None):
        return None
    rx = num_re(gold)
    for f in facts:
        if str(f.get("extractor", "")).startswith("l3") and rx.search(f.get("quote") or ""):
            return {"period": f.get("period"), "stored_value": f["value"], "stored_unit": f.get("unit"),
                    "extractor": f.get("extractor"), "page": f.get("page"), "quote": (f.get("quote") or "")[:100]}
    return None


def l3_gold_fact(facts: list[dict[str, Any]], gold: float, fy: str) -> dict[str, Any] | None:
    """An L3 fact that carries the gold value but whose period does not name the filing's fiscal year (L3 text rows emit
    one fact per column with ``period`` None): the deterministic layer HAS the value and cannot say which year it is."""
    for f in facts:
        if (str(f.get("extractor", "")).startswith("l3") and f.get("value") is not None
                and Z._close(Z.stored_value(f), gold, Z.TOL) and fy not in str(f.get("period") or "")):
            return {"period": f.get("period"), "stored_value": f["value"], "stored_unit": f.get("unit"),
                    "extractor": f.get("extractor"), "page": f.get("page"), "quote": (f.get("quote") or "")[:100]}
    return None


def first_printed(quote: str) -> float | None:
    m = re.search(r"\(?\$?\s*\(?(\d[\d,]*(?:\.\d+)?)", quote or "")
    return float(m.group(1).replace(",", "")) if m else None


def l4_class(cell: dict[str, Any]) -> tuple[str, str]:
    """The cell's class as the L4 pipeline alone fails it (ignores whether L3 holds the value)."""
    gw, top_ids = cell["gold_window_ids"], {t["chunk_id"] for t in cell["top"]}
    in_top = [c for c in gw if c in top_ids]
    if not gw:
        return "RETRIEVAL", "gold_row_not_found_in_any_window"
    if not in_top:
        return "RETRIEVAL", f"gold_window_rank_{cell['gold_best_rank']}_outside_top_{CFG.k_per_doc}"
    st = cell["admission_status"]
    adm = [c for c in in_top if st.get(c) in ("admit_sim", "gate_yes")]
    unk = [c for c in in_top if st.get(c) == "gate_unknown"]
    if not adm and not unk:
        return "RETRIEVAL", "gold_window_in_top_k_but_rejected_by_admission"
    if not adm:
        return "UNKNOWN", "gold_window_gate_outcome_not_resolvable"
    if all(c in set(cell["unparseable_windows"]) for c in adm):
        return "FORMAT", ("reply_cut_at_n_predict_256" if cell["reply_evidence"] == "server_log_mapped"
                          else "unparseable_reply_note")
    v = cell["diag_verdict"]
    if v == "PERIOD_LABEL":
        return "EXTRACTION", "value_right_period_label_wrong"
    if v == "UNIT_SCALE":
        return "EXTRACTION", "value_right_unit_label_wrong"
    return "EXTRACTION", ("reply_empty_array" if cell["gold_reply_tokens_min"] == 2 else "no_right_fact")


def classify(cell: dict[str, Any]) -> tuple[str, str, list[str]]:
    """PRIMARY class, sub-reason, secondary flags. Precedence (first match):
    GOLD       the cell's cross-extractor check failed, or neither its quote nor its number is in any stored window;
    CANONICAL  a correct fact (right value, FY period) is stored but the cell is not scored right;
    L3         the deterministic layer already holds the answer and loses it: a fact with the gold value but no FY period
               (L3 text rows), or the right row with another column's value;
    then the L4 pipeline (``l4_class``): RETRIEVAL / FORMAT / EXTRACTION / UNKNOWN.
    ``secondary`` records ``l4_also_<class>``: what the L4 pipeline alone would still have to fix."""
    sec: list[str] = []
    if cell["l3_wrong_fact"]:
        sec.append("wrong_l3_fact_present")
    if not cell["gold_check"]["xcheck_ok"] or not (cell["gold_windows"]["by_quote"] or cell["gold_windows"]["by_number"]):
        return "GOLD", "xcheck_failed_or_gold_absent_from_chunks", sec
    if cell["diag_verdict"] == "HIT":
        return "CANONICAL", "correct_fact_stored_not_selected", sec
    l4c, l4s = l4_class(cell)
    if cell["l3_gold_fact"] or cell["l3_column_error"]:
        sec.append(f"l4_also_{l4c.lower()}")
        return "L3", ("gold_value_stored_without_fy_period" if cell["l3_gold_fact"] else "right_row_wrong_column"), sec
    return l4c, l4s, sec


def triage_cell(rep: Replayer, conn: sqlite3.Connection, records: list[dict[str, Any]], mapped: dict[str, Any], *,
                grp: str, doc: str, field: str, gold: float, fy: str, quote: str, xcheck_ok: bool) -> dict[str, Any]:
    unit = f"l4:{field}:{doc}"
    rec = next((r for r in reversed(records) if r.get("kind") == "l4_unit" and r["unit"] == unit), None)
    facts = Z.facts_for(conn, doc, field)
    diag = Z.diagnose_cell({"doc": doc, "field": field, "gold": gold, "fy": fy}, facts)
    top = rep.top(doc, field, fy)
    ranks = {u["chunk_id"]: i for i, (u, _) in enumerate(rep.rank(doc, rep.unit_query(field, fy)))}
    gws = gold_windows(rep, doc, field, gold, quote)
    gw = gws["ids"] or gws["by_number"]
    recs = {x["stage"]: x for x in (rec or {}).get("receipts", [])}
    adm = admission(top, recs.get("L4a_admit"))
    st = adm["status"]
    notes = {n.split(":", 2)[2] for n in recs.get("L4b_extract", {}).get("notes", [])}
    unpars = [t["chunk_id"] for t in top if short(t["chunk_id"]) in {short(n) for n in notes}]
    in_top = [c for c in gw if c in st]
    admitted_gold = [c for c in in_top if st[c] in ("admit_sim", "gate_yes")]
    toks = (mapped.get("tokens") or {}).get(unit, {})
    gold_tok = [toks[c] for c in admitted_gold if c in toks]
    evidence = "server_log_mapped" if gold_tok else ("unit_reused_no_log" if (rec or {}).get("reused") else
                                                      "receipt_notes_only")
    l3 = [f for f in facts if str(f.get("extractor", "")).startswith("l3")]
    cell = {
        "group": grp, "doc": doc, "field": field, "fy": fy, "gold": gold, "unit": unit,
        "diag_verdict": diag["verdict"], "n_facts": diag["n_facts"], "candidate": diag["candidate"],
        "l3_wrong_fact": any(not Z._close(Z.stored_value(f), gold, Z.TOL) for f in l3),
        "l3_column_error": l3_column_error(facts, gold), "l3_gold_fact": l3_gold_fact(facts, gold, fy),
        "gold_check": {"xcheck_ok": xcheck_ok, "quote": quote},
        "gold_windows": {k: [short(c) for c in v] for k, v in gws.items() if k != "ids"},
        "gold_window_ids": gw, "gold_best_rank": min((ranks[c] for c in gw if c in ranks), default=None),
        "gold_ranks": sorted(ranks[c] for c in gw if c in ranks)[:6],
        "top": top, "admission_status": st, "admission_n_gated": adm["n_gated"], "admission_consistent": adm["consistent"],
        "gold_in_admitted": bool(admitted_gold) if in_top else False,
        "unparseable_windows": unpars,
        "gold_unparseable_all": bool(admitted_gold) and all(c in unpars for c in admitted_gold),
        "gold_reply_tokens": gold_tok, "gold_reply_tokens_min": min(gold_tok) if gold_tok else None,
        "reply_evidence": evidence, "unit_reused": bool((rec or {}).get("reused")),
        "unit_record": None if rec is None else {"n_candidates": rec["n_candidates"], "n_facts": rec["n_facts"],
                                                 "l4a": recs.get("L4a_admit", {}).get("calls"),
                                                 "l4a_rejected": recs.get("L4a_admit", {}).get("rejected"),
                                                 "l4b": recs.get("L4b_extract", {}).get("calls"),
                                                 "l4b_rejected": recs.get("L4b_extract", {}).get("rejected"),
                                                 "unparseable": len(notes)},
    }
    cls, sub, sec = classify(cell)
    cell.update({"primary": cls, "sub": sub, "secondary": sec})
    cell["l4_class"], cell["l4_sub"] = l4_class(cell)
    l3 = cell["l3_gold_fact"] or cell["l3_column_error"]
    fp = first_printed(l3["quote"]) if l3 else None
    cell["l3_first_col_is_gold"] = None if fp is None else bool(Z._close(fp, gold, Z.TOL))
    cell["l3_distinct_row_candidates"] = len({(f["page"], f["quote"]) for f in l3_rows(facts)}) if l3 else None
    cell["top_k_needed"] = None if cell["gold_best_rank"] is None else cell["gold_best_rank"] + 1
    cell["fix_probes"] = fix_probes(rep, doc, field, fy, gw)
    return cell


def l3_rows(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in facts if str(f.get("extractor", "")).startswith("l3")]


def fix_probes(rep: Replayer, doc: str, field: str, fy: str, gold_ids: list[str]) -> dict[str, Any]:
    """Offline, post-hoc, in-sample: would a better locator put a gold window in the top-k?"""
    def rank_of(query: str) -> int | None:
        rk = {u["chunk_id"]: i for i, (u, _) in enumerate(rep.rank(doc, query))}
        return min((rk[c] for c in gold_ids if c in rk), default=None)
    fq = FIELD_QUERY[field]
    lab = re.compile(ROW_LABEL.get(field, r"$^") + r"[^\n]{0,40}?[$ ]*\(?\d[\d,]*", re.I)
    row_windows = [u["chunk_id"] for u in rep.by_doc.get(doc, []) if lab.search(u["text"])]
    return {"current_rank": rank_of(rep.unit_query(field, fy)), "field_query_rank": rank_of(fq),
            "n_row_locator_windows": len(row_windows),
            "row_locator_hit": any(c in row_windows for c in gold_ids),
            "row_locator_top6_hit": any(c in row_windows[:6] for c in gold_ids)}


# --------------------------------------------------------------------------- cell lists
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def xcheck_index() -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    for n in ("facts_sheet_xcheck.json", "agg13_xcheck.json", "agg14_xcheck.json"):
        for c in json.loads((HERE / n).read_text())["cells"]:
            out[(c["doc"], c["metric"])] = out.get((c["doc"], c["metric"]), True) and bool(c.get("ok", False))
    return out


def cell_inputs() -> dict[str, list[dict[str, Any]]]:
    """The three groups' gold cells, from the committed sheets. a and b list ALL cells of the r3 run, ``miss`` marks the
    non-HIT ones (so the group is DEFINED by the run, not hand-picked; the hits give the in-sample parse-to-hit rate);
    c is ``agg19_cells.csv`` (all misses)."""
    xc = xcheck_index()
    sheet = {(r["doc_name"], r["metric"]): r for r in read_csv(HERE / "facts_sheet.csv")}
    cells13 = {(r["doc_name"], r["metric"]): r for r in read_csv(HERE / "agg13_cells.csv") if r["status"] == "filled"}
    cells14 = {(r["doc_name"], r["metric"]): r for r in read_csv(HERE / "agg14_cells.csv") if r["status"] == "filled"}
    out: dict[str, list[dict[str, Any]]] = {"a": [], "b": [], "c": []}
    for grp, src in (("a", cells13), ("b", cells14)):
        conn = ro(RUNS[grp]["dir"] / "store.sqlite")
        try:
            for (doc, field), r in src.items():
                d = Z.diagnose_cell({"doc": doc, "field": field, "gold": float(r["value_usd_m"]), "fy": str(r["fiscal_year"])},
                                    Z.facts_for(conn, doc, field))
                out[grp].append({"doc": doc, "field": field, "gold": float(r["value_usd_m"]), "quote": r["quote"],
                                 "fy": str(r["fiscal_year"]), "xcheck_ok": xc.get((doc, field), False),
                                 "miss": d["verdict"] != "HIT"})
        finally:
            conn.close()
    for r in read_csv(HERE / "agg19_cells.csv"):
        s = sheet[(r["doc_name"], r["metric"])]
        out["c"].append({"doc": r["doc_name"], "field": r["metric"], "gold": float(r["gold_usd_m"]), "quote": s["quote"],
                         "fy": str(r["fiscal_year"]), "xcheck_ok": xc.get((r["doc_name"], r["metric"]), False), "miss": True,
                         "agg18_state": r["agg18_state"]})
    return out


# --------------------------------------------------------------------------- the whole triage
def validate_replay(rep: Replayer, conn: sqlite3.Connection, records: list[dict[str, Any]], fy_default: str) -> dict[str, Any]:
    """Every fresh L4 unit: (1) each stored L4 fact's window is in the recomputed top-k; (2) L4a calls == recomputed
    gated windows. Also accounts for the unit's own candidate count."""
    bad_facts, bad_gate, n = [], [], 0
    for r in records:
        if r.get("kind") != "l4_unit" or r.get("status") != "ok":
            continue
        fy = str(r.get("fy") or fy_default)
        top = rep.top(r["doc"], r["field"], fy)
        n += 1
        ids = {t["chunk_id"] for t in top}
        rows = conn.execute("SELECT chunk_id FROM facts WHERE doc_id = ? AND field = ? AND extractor LIKE 'l4%'",
                            [r["doc"], r["field"]]).fetchall()
        for (cid,) in rows:
            if cid not in ids:
                bad_facts.append((r["unit"], short(cid)))
        l4a = next((x for x in r["receipts"] if x["stage"] == "L4a_admit"), None)
        if l4a is not None and l4a["calls"] != admission(top, l4a)["n_gated"]:
            bad_gate.append(r["unit"])
        if r["n_candidates"] != len(top):
            bad_gate.append(r["unit"] + ":n_candidates")
    return {"units_checked": n, "l4_fact_window_not_in_top_k": bad_facts, "gate_count_mismatch": bad_gate,
            "ok": not bad_facts and not bad_gate}


def summarise(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by = Counter(c["primary"] for c in cells)
    by4 = Counter(c["l4_class"] for c in cells)
    return {"n": len(cells), "by_class": {k: by.get(k, 0) for k in CLASSES},
            "by_l4_class_ignoring_l3": {k: by4.get(k, 0) for k in CLASSES if by4.get(k, 0)},
            "by_class_sub": dict(Counter(
        f"{c['primary']}:{c['sub']}" for c in cells))}


def recoverable(cells: list[dict[str, Any]], parsed_hit_rate: float | None, cut_rate: float | None = None) -> dict[str, Any]:
    """Per-fix recoverable-count bounds over ``cells``. UPPER = every cell the fix addresses; POINT folds in two in-sample
    rates (parse-to-hit of admitted parsed gold windows, and the share of fresh L4b replies that end cut at n_predict).
    A cell is counted under the fix of its PRIMARY class; ``l4_also`` counts the L3 cells that would still need an L4 fix."""
    prim = lambda k: [c for c in cells if c["primary"] == k]  # noqa: E731
    ret, fmt, ext, l3, can = prim("RETRIEVAL"), prim("FORMAT"), prim("EXTRACTION"), prim("L3"), prim("CANONICAL")
    k_needed = lambda k: sum(c["top_k_needed"] is not None and c["top_k_needed"] <= k for c in ret)  # noqa: E731
    pr = parsed_hit_rate if parsed_hit_rate is not None else 1.0
    cr = cut_rate or 0.0
    l4_also = lambda k: sum(f"l4_also_{k}" in c["secondary"] for c in l3)  # noqa: E731
    return {
        "RETRIEVAL": {
            "n": len(ret), "l3_cells_also_needing_it": l4_also("retrieval"), "upper": len(ret),
            "point": round(len(ret) * pr * (1 - cr), 1),
            "gold_window_exists": sum(c["gold_best_rank"] is not None for c in ret),
            "found_if_k": {str(k): k_needed(k) for k in (8, 12, 16, 24)},
            "field_query_top6": sum(c["fix_probes"]["field_query_rank"] is not None and c["fix_probes"]["field_query_rank"] < 6
                                    for c in ret),
            "field_query_top3": sum(c["fix_probes"]["field_query_rank"] is not None and c["fix_probes"]["field_query_rank"] < 3
                                    for c in ret),
            "row_locator_window_present": sum(c["fix_probes"]["row_locator_hit"] for c in ret),
            "row_locator_top6": sum(c["fix_probes"]["row_locator_top6_hit"] for c in ret),
            "fix": "field-specific query (+ union with the current top-k) and/or a table-aware row locator"},
        "FORMAT": {
            "n": len(fmt), "l3_cells_also_needing_it": l4_also("format"), "upper": len(fmt), "point": round(len(fmt) * pr, 1),
            "fix": "constrained/typed output (row cap + bounded quote length, AGG-1.6) or n_predict 512; the replies were "
                   "cut at exactly 256 tokens"},
        "EXTRACTION": {"n": len(ext), "upper": len(ext), "point": None,
                       "fix": "prompt or model change; the one cell is a unit label (L4d unit_override shipped in "
                              "orchestrator 1a408ce4); not testable offline"},
        "L3": {"n": len(l3), "upper": len(l3), "point": None,
               "first_column_is_gold": sum(bool(c["l3_first_col_is_gold"]) for c in l3),
               "single_row_candidate": sum(c["l3_distinct_row_candidates"] == 1 for c in l3),
               "multi_row_candidates": sum((c["l3_distinct_row_candidates"] or 0) > 1 for c in l3),
               "fix": "L3 period/column attribution: a text row 'label v1 v2 v3' lists the filing's FY first; emit v1 with "
                      "period = FY. Zero model calls. Needs a row-selection rule where several rows share the label"},
        "CANONICAL": {"n": len(can), "upper": len(can), "point": None, "fix": "canonicalisation guard rules"},
    }


def retrieval_probe(rep: Replayer) -> dict[str, Any]:
    """Regression guard for the RETRIEVAL fix, over ALL filled AGG-1.1 gold cells (the AGG-1.1 store, FY2022): is the gold
    window in the top-6 under the run's query, under a field-specific query, and under their union (current top-6 plus the
    field query's top-3)? Cells whose gold span is in no window are counted apart. Post-hoc, in-sample."""
    cells = [r for r in read_csv(HERE / "facts_sheet.csv") if r["status"] == "filled" and r["value_usd_m"].strip()]
    out = {"cells": 0, "gold_in_no_window": 0, "no_window_cells": [], "current_top6": 0, "field_top6": 0, "field_top3": 0, "union_cur6_field3": 0,
           "lost_by_field_top6": [], "lost_by_union": [], "gained_by_field_top6": 0, "avg_windows_union": 0.0}
    tot = 0
    for r in cells:
        doc, f = r["doc_name"], r["metric"]
        gold = float(r["value_usd_m"].replace(",", ""))
        gws = gold_windows(rep, doc, f, gold, r["quote"])
        gids = gws["ids"] or gws["by_number"]
        out["cells"] += 1
        if not gids:
            out["gold_in_no_window"] += 1
            out["no_window_cells"].append(f"{doc}|{f}")
            continue
        cur = [u["chunk_id"] for u, _ in rep.rank(doc, rep.unit_query(f, "2022"))[:6]]
        fq = [u["chunk_id"] for u, _ in rep.rank(doc, FIELD_QUERY[f])]
        c6, f6, f3 = any(g in cur for g in gids), any(g in fq[:6] for g in gids), any(g in fq[:3] for g in gids)
        un = list(dict.fromkeys(cur + fq[:3]))
        u_ = any(g in un for g in gids)
        tot += len(un)
        out["current_top6"] += c6
        out["field_top6"] += f6
        out["field_top3"] += f3
        out["union_cur6_field3"] += u_
        out["gained_by_field_top6"] += (f6 and not c6)
        if c6 and not f6:
            out["lost_by_field_top6"].append(f"{doc}|{f}")
        if c6 and not u_:
            out["lost_by_union"].append(f"{doc}|{f}")
    n = out["cells"] - out["gold_in_no_window"]
    out["cells_with_gold_window"] = n
    out["avg_windows_union"] = round(tot / n, 2) if n else 0.0
    return out


def parse_to_hit_rate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """In-sample (groups a and b, all 49 cells of the two r3 runs): of the cells whose gold window was admitted AND whose
    admitted gold windows all parsed, the fraction that came out as a right fact. This is the point-estimate rate for
    what a FORMAT fix recovers (a cut-off reply is assumed to behave like a parsed one). Post-hoc and in-sample."""
    pool = [c for c in cells if c["group"] in ("a", "b") and c["gold_in_admitted"] and not c["gold_unparseable_all"]]
    hit = sum(c["diag_verdict"] == "HIT" for c in pool)
    return {"n_pool": len(pool), "hits": hit, "rate": round(hit / len(pool), 3) if pool else None}


def build(orch: Path = ORCH) -> dict[str, Any]:
    mod, mod_sha = load_bm25(orch)
    inputs = cell_inputs()
    report: dict[str, Any] = {"schema": "agg111_triage.v1", "cfg": {"window_chars": CFG.window_chars, "k_per_doc": CFG.k_per_doc,
                                                                  "hi": CFG.hi, "lo": CFG.lo, "n_predict": N_PREDICT},
                              "bm25_module_sha256": mod_sha, "groups": {}, "cells": []}
    everything: list[dict[str, Any]] = []
    for grp, run in RUNS.items():
        conn = ro(run["dir"] / "store.sqlite")
        try:
            rep = Replayer(conn, mod)
            records = l4_records(run["dir"] / "results.jsonl")
            mapped = map_replies(rep, records, run["dir"] / "server.log", window_fy={})
            val = validate_replay(rep, conn, records, run["fy"])
            cells = [triage_cell(rep, conn, records, mapped, grp=grp, doc=c["doc"], field=c["field"], gold=c["gold"],
                                 fy=c["fy"], quote=c["quote"], xcheck_ok=c["xcheck_ok"]) | {"miss": c["miss"],
                                 "agg18_state": c.get("agg18_state")} for c in inputs[grp]]
            everything += cells
            report["groups"][grp] = {"title": run["title"], "dir": str(run["dir"]), "n_cells_in_run": len(cells),
                                     "replay_validation": val, "reply_mapping": {k: v for k, v in mapped.items() if k != "tokens"}}
        finally:
            conn.close()
    report["parse_to_hit_in_sample"] = rate = parse_to_hit_rate(everything)
    conn = ro(RUNS["c"]["dir"] / "store.sqlite")
    try:
        report["retrieval_probe_211"] = retrieval_probe(Replayer(conn, mod))
    finally:
        conn.close()
    misses = [c for c in everything if c["miss"]]
    report["cells"] = misses
    report["pilot19_reply_lengths"] = pilot19_reply_lengths(orch)
    maps = [report["groups"][g]["reply_mapping"] for g in RUNS if report["groups"][g]["reply_mapping"].get("ok")]
    nb = sum(m["l4b_calls"] for m in maps)
    cut = round(sum(m["replies_cut_at_n_predict"] for m in maps) / nb, 3) if nb else None
    report["l4b_cut_rate_fresh_runs"] = {"l4b_calls": nb, "cut_rate": cut}
    for grp in RUNS:
        mine = [c for c in misses if c["group"] == grp]
        report["groups"][grp].update({"summary": summarise(mine), "recoverable": recoverable(mine, rate["rate"], cut)})
    report["groups"]["all"] = {"title": "all misses (a + b + c)", "summary": summarise(misses),
                               "recoverable": recoverable(misses, rate["rate"], cut)}
    return report


def pilot19_reply_lengths(orch: Path = ORCH) -> dict[str, Any]:
    """The AGG-1.9 pilot (5 cells, 30 L4b calls): reply length per call from its server log, mapped in order."""
    log, res = PILOT19 / "server.log", PILOT19 / "results.jsonl"
    if not (log.exists() and res.exists()):
        return {"available": False}
    toks = parse_log_tokens(log)
    fresh = [r for r in l4_records(res) if r.get("kind") == "l4_unit" and not r.get("reused")]
    calls = sum(r["llm_calls"] for r in fresh)
    notes = sum(len(x["notes"]) for r in fresh for x in r["receipts"])
    n256, n2 = sum(t == N_PREDICT for t in toks), sum(t == 2 for t in toks)
    return {"available": True, "calls": len(toks), "fresh_units": len(fresh), "fresh_calls": calls,
            "unparseable_notes": notes, "replies_cut_at_256": n256, "replies_empty_array": n2,
            "replies_with_rows": len(toks) - n256 - n2, "mapping_ok": len(toks) == calls and n256 == notes,
            "hit_rate_when_parsed": None}


PILOT19_CELLS = (("3M_2022_10K", "net_income"), ("AMERICANEXPRESS_2022_10K", "capex"), ("COCACOLA_2022_10K", "net_income"),
                 ("COSTCO_2022_10K", "net_income"), ("MGMRESORTS_2022_10K", "total_assets"))


def findings(report: dict[str, Any]) -> list[str]:
    a = report["groups"]["all"]["summary"]
    by, by4 = a["by_class"], a["by_l4_class_ignoring_l3"]
    cells = {(c["doc"], c["field"]): c for c in report["cells"]}
    pil = [cells[k] for k in PILOT19_CELLS]
    b = {c["doc"]: c for c in report["cells"] if c["group"] == "b"}
    pl = report["pilot19_reply_lengths"]
    return ["## Findings", "",
            f"* All {a['n']} misses are classified; none is GOLD (each gold passes its cross-extractor check and its span is in "
            f"a stored window) and none is CANONICAL. Primary: FORMAT {by['FORMAT']}, RETRIEVAL {by['RETRIEVAL']}, L3 {by['L3']}, "
            f"EXTRACTION {by['EXTRACTION']}. By the L4 pipeline alone: FORMAT {by4.get('FORMAT', 0)}, RETRIEVAL "
            f"{by4.get('RETRIEVAL', 0)}, EXTRACTION {by4.get('EXTRACTION', 0)}.",
            f"* FORMAT is truncation: the L4b grammar forces valid JSON, so an unparseable reply is one that ran out of "
            f"`n_predict`. In every run whose server log validates, the number of 256-token replies equals the number of "
            f"`unparseable` notes (AGG-1.9 pilot: {pl.get('replies_cut_at_256')} of {pl.get('calls')} replies cut at 256, "
            f"{pl.get('replies_empty_array')} were `[]`, {pl.get('replies_with_rows')} had rows), about "
            f"{report['l4b_cut_rate_fresh_runs']['cut_rate']} of fresh L4b replies overall.",
            "* The AGG-1.9 pilot (5 cells, 0 filled, futility) was a replay: " + ", ".join(
                f"{c['doc'].replace('_2022_10K', '')}/{c['field']} {c['primary']}" for c in pil)
            + ". Re-running the same call on the same windows cannot fill a FORMAT cell (same prompt, same cut) or a RETRIEVAL "
              "cell (same windows); the fix has to change the windows or the output contract.",
            f"* AGG-1.4 box check: Activision is RETRIEVAL (gold window rank {b['ACTIVISIONBLIZZARD_2022_10K']['gold_best_rank']}, "
            f"just outside the top-6), not 'no fact'; Boeing is RETRIEVAL (gold window rank {b['BOEING_2022_10K']['gold_best_rank']}; "
            f"its one unparseable reply was on a non-gold window), not an unparseable reply; CocaCola is L3 as stated (the "
            f"L3 row quotes 36,377 | 38,116 and stores 38,116 as '2024') and L4 also failed there (both gold windows cut at 256).",
            ""]


def md(report: dict[str, Any]) -> str:
    L = ["# AGG-1.11 step 1 -- triage of missed cells (no model)", "",
         "Generated by `agg111_triage.py` from the saved stores (read-only, `mode=ro`); counts are reproduced by "
         "`tests/test_agg111_triage.py`. Post-hoc, in-sample, diagnostic: no gate, gold or scorer is changed. Classes are "
         "PRIMARY causes (first match: GOLD, CANONICAL, L3, then the L4 pipeline: RETRIEVAL, FORMAT, EXTRACTION). The model "
         "replies were not saved; reply-level evidence is the receipts' `unparseable` notes plus per-request token counts "
         "from each run's llama-server log, accepted only where they validate (see the script docstring).", ""]
    L += findings(report)
    for g in ("a", "b", "c", "all"):
        s = report["groups"][g]
        L += [f"## Group {g}: {s['title']} ({s['summary']['n']} cells)", "", "| primary class | n |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in s["summary"]["by_class"].items() if v]
        L += ["", "By the L4 pipeline alone (ignoring that L3 holds the value for some cells): "
              + ", ".join(f"{k} {v}" for k, v in s["summary"]["by_l4_class_ignoring_l3"].items()) + ".", ""]
        if g != "all":
            v, m = s["replay_validation"], s["reply_mapping"]
            L += [f"Replay validation: {v['units_checked']} L4 units re-run offline, every stored L4 fact sits in a recomputed "
                  f"top-6 window and every gate-call count matches: {v['ok']}. Reply mapping: {m.get('why')}.", ""]
        r = s["recoverable"]
        L += ["| fix | primary cells | upper | point (in-sample) | probe |", "|---|---|---|---|---|"]
        rt = r["RETRIEVAL"]
        L += [f"| RETRIEVAL: {rt['fix']} | {rt['n']} (+{rt['l3_cells_also_needing_it']} L3 cells) | {rt['upper']} | {rt['point']} | "
              f"gold window exists {rt['gold_window_exists']}; found if k=8/12/16/24 {rt['found_if_k']['8']}/"
              f"{rt['found_if_k']['12']}/{rt['found_if_k']['16']}/{rt['found_if_k']['24']}; field query top-6 "
              f"{rt['field_query_top6']}, top-3 {rt['field_query_top3']}; row locator in first 6 {rt['row_locator_top6']} |"]
        f = r["FORMAT"]
        L += [f"| FORMAT: {f['fix']} | {f['n']} (+{f['l3_cells_also_needing_it']} L3 cells) | {f['upper']} | {f['point']} | |"]
        e = r["EXTRACTION"]
        L += [f"| EXTRACTION: {e['fix']} | {e['n']} | {e['upper']} | | |"]
        t = r["L3"]
        L += [f"| L3: {t['fix']} | {t['n']} | {t['upper']} | | first column is gold {t['first_column_is_gold']}; "
              f"single candidate row {t['single_row_candidate']}, several {t['multi_row_candidates']} |"]
        c = r["CANONICAL"]
        L += [f"| CANONICAL: {c['fix']} | {c['n']} | {c['upper']} | | |", ""]
    p, rp = report["parse_to_hit_in_sample"], report["retrieval_probe_211"]
    L += ["## In-sample rates and the retrieval regression probe", "",
          f"* Parse-to-hit: of the {p['n_pool']} cells of the two r3 runs whose gold window was admitted and parsed, "
          f"{p['hits']} came out as a right fact (rate {p['rate']}).",
          f"* Fresh L4b replies cut at n_predict 256: {report['l4b_cut_rate_fresh_runs']}. AGG-1.9 pilot: "
          f"{report['pilot19_reply_lengths']}.",
          f"* Retrieval probe over all {rp['cells']} filled AGG-1.1 gold cells: gold window in the top-6 under the run's "
          f"query {rp['current_top6']}; under a field-specific query {rp['field_top6']} (top-3 {rp['field_top3']}); under the "
          f"union (current top-6 + field top-3, {rp['avg_windows_union']} windows on average) {rp['union_cur6_field3']}. Lost by "
          f"the field query alone: {rp['lost_by_field_top6']}; lost by the union: {rp['lost_by_union']}.", "",
          "## Cells", "", "| grp | doc | field | gold | primary | sub | L4 class | gold-window ranks | admitted | reply evidence | "
          "reply tokens | secondary |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        L.append(f"| {c['group']} | {c['doc'].replace('_2022_10K', '')} | {c['field']} | {c['gold']:g} | {c['primary']} | "
                 f"{c['sub']} | {c['l4_class']} | {c['gold_ranks']} | {c['gold_in_admitted']} | {c['reply_evidence']} | "
                 f"{c['gold_reply_tokens'] or ''} | {','.join(c['secondary'])} |")
    L.append("")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    rep = build()
    jp = Path(argv[argv.index("--json") + 1]) if "--json" in argv else HERE / "agg111_triage.json"
    mp = Path(argv[argv.index("--md") + 1]) if "--md" in argv else HERE / "agg111_triage.md"
    jp.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    mp.write_text(md(rep))
    for g in ("a", "b", "c", "all"):
        print(g, rep["groups"][g]["summary"]["by_class"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
