#!/usr/bin/env python3
"""AGG-0.4 Phase-0 arms: L4 ingest, 27B audit, P-large / P-small / B1 / B2 / B-today, scoring.

Imported by ``run_phase0.py``. Every model call goes through an injected backend
(``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in
tests); this module never launches or signals a process.

Phases (one keeper job each side of the hardware split, see PREDECLARED.md):

* ``cpu``         M5 manifest check (zero LLM), then lazy L4 over the 8 schema-seed
                  fields (worker_general 35B, CPU quarter), L4d, ingest record, M1.
                  Futility 1 (P-small built an L2-L4 index) and futility 2 (M1 < 0.70)
                  stop here; the GPU job's precondition then never passes.
* ``gpu-audit``   L4c: 27B re-extraction of ``facts_llm.audit_sample`` (np4 server).
* ``gpu-answers`` all answer arms on the 27B np1 196k server, then the verdict.

Every record is appended with flush+fsync as soon as it exists; a re-run skips
any ``unit`` already recorded (resumable per question / per (field, doc) / per fact).
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agg_phase0 as A  # noqa: E402

CORPUS_ID = "agg_phase0_fb10"

# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Cfg:
    window_chars: int = 10_000      # L4 candidate window (fits a 4k np4 slot for the audit)
    k_per_doc: int = 6              # L1 top-K windows per (field, doc)
    hi: float = 0.6                 # sim = BM25 / top BM25 in the doc; >= hi admit
    lo: float = 0.2                 # <= lo reject; between -> logprob gate
    l4_n_predict: int = 256
    sql_n_predict: int = 160
    answer_n_predict: int = 384
    note_n_predict: int = 256
    b1_max_calls: int = 30
    b1_search_k: int = 8
    b1_read_chars: int = 12_000
    np1_window: int = 196_608
    sql_row_cap: int = 60
    quote_chars: int = 160
    rel_tol: float = 0.005


CFG = Cfg()


# --------------------------------------------------------------------------- records
class Recorder:
    """Append-only JSONL, one flushed + fsynced line per record; knows finished units."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    self.records.append(json.loads(line))
                except ValueError:
                    continue  # a torn last line from a killed run is ignored, never repaired
        self.done = {r["unit"] for r in self.records if r.get("unit") and r.get("final", True)}

    def emit(self, kind: str, **rec: Any) -> dict[str, Any]:
        r = {"kind": kind, "ts": time.time(), **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records.append(r)
        if r.get("unit") and r.get("final", True):
            self.done.add(r["unit"])
        return r


class Stop(Exception):
    """Deadline / stopfile: the job exits 75 and the next grant resumes."""


def make_guard(deadline: float | None, stopfile: str | None) -> Callable[[], None]:
    def guard() -> None:
        if stopfile and os.path.exists(stopfile):
            raise Stop("stopfile")
        if deadline and time.time() >= deadline:
            raise Stop("deadline")
    return guard


# --------------------------------------------------------------------------- scoring
# Deterministic scorer (PREDECLARED.md, amended pre-data 2026-11-10; replaces CJ-8).
_SCALE = {"thousand": 1e-3, "thousands": 1e-3, "k": 1e-3, "million": 1.0, "millions": 1.0,
          "m": 1.0, "mm": 1.0, "mn": 1.0, "billion": 1e3, "billions": 1e3, "bn": 1e3, "b": 1e3,
          "trillion": 1e6}
_NUM = re.compile(r"(\()?(-|−)?\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(%|percent|thousands?|millions?|billions?|"
                  r"trillion|bn|mm|mn|[kmb])?\b\)?", re.I)


def final_line(text: str) -> str:
    """The text after the last ``FINAL:``; else the last non-empty line."""
    if not text:
        return ""
    i = text.rfind("FINAL:")
    if i >= 0:
        return text[i + 6:].strip().splitlines()[0].strip() if text[i + 6:].strip() else ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def parse_numbers(text: str, *, money_scale: bool = True) -> list[tuple[float, int]]:
    """``(value, decimals)`` for each number; money scale words normalise to USD millions."""
    out = []
    for m in _NUM.finditer(text or ""):
        paren, minus, digits, unit = m.group(1), m.group(2), m.group(3), (m.group(4) or "").lower()
        if re.fullmatch(r"(19|20)\d\d", digits) and not unit:
            continue  # a bare year is never an answer value
        v = float(digits.replace(",", ""))
        dec = len(digits.split(".")[1]) if "." in digits else 0
        if money_scale and unit in _SCALE:
            v *= _SCALE[unit]
        if minus or (paren and m.group(0).endswith(")")):
            v = -v
        out.append((v, dec))
    return out


def num_match(answer: float, gold: float, gold_decimals: int, rel_tol: float) -> bool:
    half_unit = 0.5 * 10 ** (-gold_decimals) if gold_decimals > 0 else 0.0
    return abs(answer - gold) <= max(rel_tol * abs(gold), half_unit) + 1e-12


def norm_name(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in {"the", "inc", "corp", "corporation", "company", "co",
                                                "plc", "ltd", "llc"}]
    return " ".join(toks)


def score_agg(gold: dict[str, Any], answer_text: str, rel_tol: float = CFG.rel_tol) -> bool | None:
    """True/False; None when no FINAL value could be read (counted as not correct)."""
    fl = final_line(answer_text)
    if not fl:
        return None
    op, g = gold["op"], gold["gold"]
    if op in ("argmax", "argmin"):
        return norm_name(fl) == norm_name(str(g))
    if op == "top3":
        items = [norm_name(x) for x in re.split(r"[;,\n]|\band\b", fl) if norm_name(x)]
        return items[:3] == [norm_name(x) for x in g] and len(items) == 3
    nums = parse_numbers(fl)
    if not nums:
        return None
    if op in ("count_gt", "count_lt"):
        return int(round(nums[0][0])) == int(g) and float(nums[0][0]).is_integer()
    if op == "sum":
        return num_match(nums[0][0], float(g), 0, rel_tol)
    raise ValueError(op)


def score_single(key: dict[str, Any], answer_text: str, rel_tol: float = CFG.rel_tol) -> bool | None:
    """``key`` from single_fact_keys.json: optional ``yesno``, ``numbers`` [{value, decimals, signed}]."""
    fl = final_line(answer_text)
    if not fl:
        return None
    if key.get("yesno"):
        m = re.match(r"\s*(yes|no)\b", fl, re.I)
        if not m or m.group(1).lower() != key["yesno"]:
            return False
    found = parse_numbers(fl)
    for want in key.get("numbers", []):
        ok = any(num_match(v if want.get("signed") else abs(v), want["value"] if want.get("signed")
                           else abs(want["value"]), want.get("decimals", 0), rel_tol) for v, _ in found)
        if not ok:
            return False
    return True


# --------------------------------------------------------------------------- store helpers
def open_store(path: Path):
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    return CorpusStore(path)


def windows(store, corpus_id: str, window_chars: int) -> list[dict[str, Any]]:
    """Deterministic L4 candidate units: each stored chunk split at <= window_chars (line breaks)."""
    ent = {d["doc_hash"]: d["entity"] for d in store.corpus_docs(corpus_id)}
    out = []
    for c in store.chunks(corpus_id):
        text = c["text"] or ""
        parts, cur = [], ""
        for line in text.splitlines(keepends=True):
            while len(line) > window_chars:  # one over-long line: hard split, still lossless
                if cur:
                    parts.append(cur)
                    cur = ""
                parts.append(line[:window_chars])
                line = line[window_chars:]
            if len(cur) + len(line) > window_chars and cur:
                parts.append(cur)
                cur = ""
            cur += line
        if cur:
            parts.append(cur)
        for i, p in enumerate(parts):
            out.append({"chunk_id": f"{c['chunk_id']}#w{i}", "doc_hash": c["doc_hash"],
                        "doc_id": c["doc_id"], "entity": ent.get(c["doc_hash"]),
                        "page_start": c["page_start"], "title": c["title"], "text": p})
    return out


def bm25_top(units: list[dict[str, Any]], query: str, k: int) -> list[dict[str, Any]]:
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    if not units:
        return []
    idx = BM25Index([u["text"] for u in units])
    scored = idx.scores(query)[:k]
    if not scored:
        return []
    top = scored[0][1]
    return [dict(units[i], sim=s / top) for i, s in scored]


def doc_text(store, corpus_id: str, doc_id: str) -> str:
    return "\n\n".join(c["text"] or "" for c in store.chunks(corpus_id) if c["doc_id"] == doc_id)


def doc_tokens(store, corpus_id: str, doc_id: str) -> int:
    return sum(d["n_tokens_est"] or 0 for d in store.corpus_docs(corpus_id) if d["doc_id"] == doc_id)


def single_doc_store(store, doc_id: str):
    """In-memory store holding ONE document (P-small's corpus), so its manifest is its own."""
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    small = CorpusStore(":memory:")
    cid = f"psmall:{doc_id}"
    src = store.conn
    for row in src.execute("SELECT * FROM documents WHERE corpus_id = ? AND doc_id = ?",
                           [CORPUS_ID, doc_id]).fetchall():
        row = list(row)
        row[1] = cid
        small.conn.execute(f"INSERT INTO documents VALUES ({','.join('?' * len(row))})", row)
        for c in src.execute("SELECT * FROM chunks WHERE doc_hash = ?", [row[0]]).fetchall():
            small.conn.execute(f"INSERT INTO chunks VALUES ({','.join('?' * len(c))})", list(c))
    small.conn.commit()
    return small, cid


def copy_store(src: Path, dst: Path) -> None:
    """sqlite online backup (read-only on the source; never touches its WAL)."""
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    with d:
        s.backup(d)
    s.close()
    d.close()


# --------------------------------------------------------------------------- M1
def m1_cells(set_dir: Path, verified_agg: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    """(doc_name, metric, value) gold cells the verified agg questions read (status filled)."""
    sheet = A.read_sheet(set_dir / "facts_sheet.csv")
    need: set[tuple[str, str]] = set()
    for g in verified_agg:
        metrics = re.split(r"[-/]", g["expr"])
        for d in A.DOCS:
            if d in g.get("excluded_by_rule", []):
                continue
            for m in metrics:
                need.add((d, m))
    cells = []
    for (d, m) in sorted(need):
        r = sheet.get((d, m))
        if r and r["status"] == "filled" and r["value_usd_m"].strip():
            cells.append((d, m, float(r["value_usd_m"].replace(",", ""))))
    return cells


def m1_found(facts: Iterable[dict[str, Any]], cells, rel_tol: float = CFG.rel_tol) -> int:
    from src.aggregation.facts_llm import UNIT_SCALE  # noqa: PLC0415
    by: dict[tuple[str, str], list[float]] = {}
    for f in facts:
        if f.get("value") is None or "2022" not in str(f.get("period") or ""):
            continue
        v = float(f["value"])
        if f.get("unit") in UNIT_SCALE:
            v *= UNIT_SCALE[f["unit"]]
        by.setdefault((f.get("doc_id"), f.get("field")), []).append(v)
    n = 0
    for d, m, gv in cells:
        if any(abs(abs(v) - abs(gv)) <= rel_tol * abs(gv) + 1e-9 for v in by.get((d, m), [])):
            n += 1
    return n


# --------------------------------------------------------------------------- CPU phase
def m5_manifest(rec: Recorder, store, single: list[dict[str, Any]], *, planner_mod) -> list[str]:
    """Futility 1 (zero LLM): plan each P-small question; build only what its path builds."""
    from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
    tripped = []
    for q in single:
        unit = f"manifest:P-small:{q['id']}"
        if unit in rec.done:
            prev = next(r for r in rec.records if r.get("unit") == unit)
            tripped += [q["id"]] if set(prev["built_layers"]) & {"L2", "L3", "L4"} else []
            continue
        small, cid = single_doc_store(store, q["doc_name"])
        facts = planner_mod.PlannerFacts(corpus_tokens=small.corpus_tokens(cid), n_docs=1)
        dec = planner_mod.plan(q["question"], facts)
        if "L1" in dec.layers:  # repl_nav / retrieval read L1; L2 is not built in Phase 0
            IndexRegistry(small).get_or_build(cid, "l1_bm25")
        built = sorted(small.built_layers(cid))
        rec.emit("manifest", unit=unit, arm="P-small", qid=q["id"], path=dec.path,
                 planned_layers=list(dec.layers), built_layers=built, decided_by=dec.decided_by,
                 planner_llm_called=dec.llm_called, corpus_tokens=facts.corpus_tokens)
        if set(built) & {"L2", "L3", "L4"}:
            tripped.append(q["id"])
    return tripped


def run_cpu(rec: Recorder, store, set_dir: Path, backend, *, agg, single, guard, smoke=False,
            cfg: Cfg = CFG) -> dict[str, Any]:
    from src.aggregation import facts_llm, planner  # noqa: PLC0415
    tripped = m5_manifest(rec, store, single, planner_mod=planner)
    if tripped:
        rec.emit("stop", reason="futility_1_m5_index_built", qids=tripped)
        return {"complete": True, "futility_stop": "M5", "m1": None}
    units = windows(store, CORPUS_ID, cfg.window_chars)
    docs = sorted({u["doc_id"] for u in units})
    if smoke:  # smallest document, first field: one real (field, doc) unit
        sizes = {d: sum(len(u["text"]) for u in units if u["doc_id"] == d) for d in docs}
        docs = [min(docs, key=lambda d: sizes[d])]
    fields = list(A.METRICS)[:1] if smoke else list(A.METRICS)
    for fname in fields:
        for d in docs:
            unit = f"l4:{fname}:{d}"
            if unit in rec.done:
                continue
            guard()
            t0 = time.time()
            cands = bm25_top([u for u in units if u["doc_id"] == d],
                             f"{fname.replace('_', ' ')} {A.METRICS[fname]}", cfg.k_per_doc)
            bf = facts_llm.LazyFieldBackfill(store, backend, n_predict=cfg.l4_n_predict, id_slot=None,
                                             hi=cfg.hi, lo=cfg.lo)
            calls0, pre0 = backend.calls, backend.prefill_tokens
            try:
                out = bf.fill(CORPUS_ID, fname, A.METRICS[fname], cands)
                status = "ok"
            except Exception as e:  # noqa: BLE001 — recorded per unit, the run continues
                out, status = {"facts": [], "conflicts": [], "receipts": [], "audit_queue": []}, \
                    f"error:{type(e).__name__}:{str(e)[:200]}"
            rec.emit("l4_unit", unit=unit, field=fname, doc=d, status=status, ok=status == "ok",
                     n_candidates=len(cands), n_facts=len(out["facts"]),
                     n_conflicts=len(out["conflicts"]), receipts=out["receipts"],
                     llm_calls=backend.calls - calls0, prefill_tokens=backend.prefill_tokens - pre0,
                     wall_s=time.time() - t0, final=status == "ok")
    if smoke:
        return {"complete": False, "smoke": True}
    l4 = [r for r in rec.records if r["kind"] == "l4_unit" and r.get("ok")]
    n_l4 = len({r["unit"] for r in l4})
    if n_l4 < len(A.METRICS) * len(docs):
        return {"complete": False, "reason": "l4 units with errors; re-grant retries them"}
    rec.emit("ingest", unit="ingest", prefill_tokens=sum(r["prefill_tokens"] for r in l4),
             corpus_tokens=store.corpus_tokens(CORPUS_ID), llm_calls=sum(r["llm_calls"] for r in l4),
             n_chunks=len(store.chunks(CORPUS_ID)), n_windows=len(units),
             cache_n=sum(x.get("cache_n", 0) for r in l4 for x in r["receipts"]))
    cells = m1_cells(set_dir, agg)
    found = m1_found(store.facts(CORPUS_ID), cells)
    m1 = found / len(cells) if cells else None
    rec.emit("m1", unit="m1", gold_facts=len(cells), found=found, m1=m1)
    stop = m1 is None or m1 < A.KILL["m1_min"]
    if stop:
        rec.emit("stop", reason="futility_2_m1", m1=m1)
    return {"complete": True, "m1": m1, "futility_stop": "M1" if stop else None}


# --------------------------------------------------------------------------- GPU audit
def run_audit(rec: Recorder, store, backend, *, guard, smoke=False, cfg: Cfg = CFG) -> dict[str, Any]:
    from src.aggregation import facts_llm  # noqa: PLC0415
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    facts = store.facts(CORPUS_ID, extractor=facts_llm.EXTRACTOR_L4)
    queue = facts_llm.audit_sample(facts)
    wins = {u["chunk_id"]: u for u in windows(store, CORPUS_ID, cfg.window_chars)}
    n = 0
    for f in queue:
        unit = f"audit:{f['fact_id']}"
        if unit in rec.done:
            continue
        guard()
        w = wins.get(f["chunk_id"])
        r = facts_llm.StageReceipt("L4c_audit")
        status = "ok"
        try:
            rows = facts_llm.extract(w, f["field"], A.METRICS.get(f["field"], f["field"]), backend,
                                     n_predict=cfg.l4_n_predict, id_slot=None, receipt=r) if w else []
            agree = any(str(x["period"]) == str(f["period"]) and abs(
                x["value"] * facts_llm.UNIT_SCALE.get(x["unit"], 1.0) - f["value"])
                <= cfg.rel_tol * abs(f["value"]) + 1e-9 for x in rows)
            verdict = "audited_ok" if agree else "disputed"
            store.conn.execute("UPDATE facts SET audit_status = ? WHERE fact_id = ?",
                               [verdict, f["fact_id"]])
            store.conn.commit()
        except OutOfWindow:
            status, verdict = "out_of_window", None
        rec.emit("audit", unit=unit, fact_id=f["fact_id"], field=f["field"], doc=f["doc_id"],
                 status=status, verdict=verdict, calls=r.calls, prefill_tokens=r.prompt_tokens,
                 cache_n=r.cache_n)
        n += 1
        if smoke:
            break
    done = all(f"audit:{f['fact_id']}" in rec.done for f in queue)
    if done and not smoke and "audit_done" not in rec.done:
        rec.emit("audit_done", unit="audit_done", n_queue=len(queue),
                 disputed=sum(1 for r in rec.records if r["kind"] == "audit" and r.get("verdict") == "disputed"))
    return {"complete": done, "n_audited_now": n, "n_queue": len(queue)}


# --------------------------------------------------------------------------- answer arms
ANSWER_RULE = ("End your reply with one line 'FINAL: <answer>'. For a company, write exactly one name "
               "from this list: {names}. For several companies, separate names with commas. For an "
               "amount, write the number in USD millions. For a count, write the integer.")


def _ans_prompt(question: str, context: str, names: list[str]) -> str:
    return (f"{context}\n\nQuestion: {question}\n" + ANSWER_RULE.format(names="; ".join(names)))


class Tally:
    def __init__(self, backend) -> None:
        self.b, self.c0, self.p0 = backend, backend.calls, backend.prefill_tokens

    @property
    def calls(self) -> int:
        return self.b.calls - self.c0

    @property
    def prefill(self) -> int:
        return self.b.prefill_tokens - self.p0


def whole_doc_answer(backend, store, doc_id: str, question: str, names, cfg: Cfg) -> tuple[str, str]:
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    if doc_tokens(store, CORPUS_ID, doc_id) + cfg.answer_n_predict > cfg.np1_window * 1.25:
        return "", "out_of_window"  # clearly over: no call; the server is authoritative otherwise
    ctx = f"Filing ({doc_id}):\n{doc_text(store, CORPUS_ID, doc_id)}"
    try:
        # cache_prompt=False: P-small and B-today each pay a full prefill, so M5's prefill parity
        # never depends on which arm happened to warm the slot (PREDECLARED amendment 4).
        r = backend.chat(_ans_prompt(question, ctx, names), n_predict=cfg.answer_n_predict,
                         cache_prompt=False, id_slot=0)
    except OutOfWindow:
        return "", "out_of_window"
    return r["text"], "ok"


def agent_answer(backend, chunks: list[dict[str, Any]], question: str, names, cfg: Cfg,
                 bm25=None) -> tuple[str, str]:
    """B1 (and P-small repl_nav/retrieval): SEARCH / READ / FINAL loop, <= b1_max_calls calls."""
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    idx = bm25 or BM25Index([c["text"] or "" for c in chunks])
    by_id = {c["chunk_id"]: c for c in chunks}
    grammar = 'root ::= ("SEARCH: " | "READ: " | "FINAL: ") [^\\n]+'
    head = ("You answer a question about a corpus of FY2022 10-K filings using tools. Each turn reply with "
            "exactly one line: 'SEARCH: <keywords>' (lexical search, returns chunk ids), 'READ: <chunk_id>' "
            "(returns the chunk text) or 'FINAL: <answer>'.\n" + ANSWER_RULE.format(names="; ".join(names))
            + f"\nQuestion: {question}\n")
    hist = ""
    for _ in range(cfg.b1_max_calls):
        try:
            r = backend.chat(head + hist + "Next action:", n_predict=120, grammar=grammar, id_slot=0)
        except OutOfWindow:
            return "", "out_of_window"
        act = r["text"].strip()
        hist += f"\n> {act}\n"
        if act.startswith("FINAL:"):
            return act, "ok"
        arg = act.split(":", 1)[1].strip() if ":" in act else ""
        if act.startswith("SEARCH:"):
            hits = idx.scores(arg)[:cfg.b1_search_k]
            hist += "\n".join(f"[{chunks[i]['chunk_id']}] {chunks[i].get('doc_id')} | "
                              f"{(chunks[i].get('title') or '')[:80]} | "
                              f"{(chunks[i]['text'] or '')[:200].replace(chr(10), ' ')}" for i, _ in hits) or "(no hits)"
        elif act.startswith("READ:"):
            c = by_id.get(arg)
            hist += (c["text"] or "")[:cfg.b1_read_chars] if c else "(unknown chunk id)"
    return "", "budget_exhausted"


def sql_answer(backend, store, question: str, names, cfg: Cfg) -> tuple[str, str, dict[str, Any]]:
    from src.aggregation import sql_grammar  # noqa: PLC0415
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    facts = store.facts(CORPUS_ID)
    cat = sql_grammar.catalog_from_facts(facts)
    gbnf = sql_grammar.build_sql_gbnf(cat)
    fdesc = "\n".join(f"- {f}: {A.METRICS.get(f, '(deterministic L3 field)')}" for f in cat.fields
                      if f in A.METRICS) or "(none)"
    prompt = ("Write ONE SQLite SELECT over table facts(entity, doc_id, period, field, value, unit, quote, "
              "page). value is in USD millions for money fields; period text contains the fiscal year. "
              f"Fields:\n{fdesc}\nCompany names (entity): {'; '.join(names)}\nQuestion: {question}\nSQL:")
    info: dict[str, Any] = {}
    try:
        sql = backend.chat(prompt, n_predict=cfg.sql_n_predict, grammar=gbnf, id_slot=0)["text"].strip()
    except OutOfWindow:
        return "", "out_of_window", info
    ok, why = sql_grammar.validate_sql(sql, cat)
    info.update(sql=sql, sql_valid=ok, sql_reason=why)
    rows: list[Any] = []
    if ok:
        try:
            cur = store.conn.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchmany(cfg.sql_row_cap)]
        except sqlite3.Error as e:
            info["sql_error"] = str(e)[:200]
    info["n_rows"] = len(rows)
    info["backfill_needed"] = ok and not rows  # lazy L4 for a new field is a CPU-job step
    for r in rows:
        if isinstance(r.get("quote"), str):
            r["quote"] = r["quote"][:cfg.quote_chars]
    ctx = f"SQL: {sql}\nRows ({len(rows)}):\n" + "\n".join(json.dumps(r, default=str) for r in rows)
    try:
        ans = backend.chat(_ans_prompt(question, ctx, names), n_predict=cfg.answer_n_predict, id_slot=0)
    except OutOfWindow:
        return "", "out_of_window", info
    return ans["text"], "ok", info


def b2_notes_prepass(rec, backend, store, qs, cfg: Cfg, guard) -> None:
    """Whole-document read per (doc, question), DOC-OUTER so one filing's prefix is reused from the
    slot cache across the 5 questions (filing first, question last). One record per note."""
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    for d in sorted({x["doc_id"] for x in store.corpus_docs(CORPUS_ID)}):
        over = doc_tokens(store, CORPUS_ID, d) + cfg.note_n_predict > cfg.np1_window * 1.25
        for q in qs:
            unit = f"b2note:{q['id']}:{d}"
            if unit in rec.done:
                continue
            guard()
            if over:
                rec.emit("b2_note", unit=unit, qid=q["id"], doc=d, status="out_of_window",
                         calls=0, prefill_tokens=0, note="")
                continue
            t = Tally(backend)
            try:
                r = backend.chat(f"Filing ({d}):\n{doc_text(store, CORPUS_ID, d)}\n\nA question about a "
                                 f"corpus of filings: {q['question']}\nFrom THIS filing only, state the "
                                 "company's values needed to answer it (USD millions), briefly.",
                                 n_predict=cfg.note_n_predict, id_slot=0)
                st, note = "ok", r["text"]
            except OutOfWindow:
                st, note = "out_of_window", ""
            rec.emit("b2_note", unit=unit, qid=q["id"], doc=d, status=st, calls=t.calls,
                     prefill_tokens=t.prefill, note=note)


def b2_answer(rec, backend, q, names, cfg: Cfg) -> tuple[str, str, int, int]:
    """Aggregate call over the question's per-doc notes; (text, status, calls, prefill) incl. notes."""
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    notes = [r for r in rec.records if r.get("kind") == "b2_note" and r.get("qid") == q["id"]]
    calls, pre = sum(r["calls"] for r in notes), sum(r["prefill_tokens"] for r in notes)
    if any(r["status"] == "out_of_window" for r in notes):
        return "", "out_of_window", calls, pre
    t = Tally(backend)
    try:
        r = backend.chat(_ans_prompt(q["question"], "Per-filing notes:\n" + "\n".join(
            f"{n['doc']}: {n['note']}" for n in notes), names), n_predict=cfg.answer_n_predict, id_slot=0)
    except OutOfWindow:
        return "", "out_of_window", calls + t.calls, pre + t.prefill
    return r["text"], "ok", calls + t.calls, pre + t.prefill


def run_answers(rec: Recorder, store, backend, *, agg, single, keys, names, guard, smoke=False,
                cfg: Cfg = CFG) -> dict[str, Any]:
    from src.aggregation import planner  # noqa: PLC0415
    from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    corpus_chunks = store.chunks(CORPUS_ID)
    corpus_bm25 = BM25Index([c["text"] or "" for c in corpus_chunks])
    C = store.corpus_tokens(CORPUS_ID)
    sql_present = bool(store.facts(CORPUS_ID))
    built = sorted(store.built_layers(CORPUS_ID))

    def correct(qclass, q, text):
        return score_agg(q, text) if qclass == "agg" else score_single(keys[q["id"]], text)

    plan_items = ([("P-small", "single", q) for q in single] + [("B-today", "single", q) for q in single]
                  + [("P-large", "agg", q) for q in agg] + [("P-large", "single", q) for q in single]
                  + [("B1", "agg", q) for q in agg] + [("B1", "single", q) for q in single]
                  + [("B2", "agg", q) for q in agg[:5]])
    n = 0
    for i, (arm, qclass, q) in enumerate(plan_items):
        unit = f"answer:{arm}:{q['id']}"
        if unit in rec.done:
            continue
        guard()
        t = Tally(backend)
        t0 = time.time()
        extra: dict[str, Any] = {}
        if arm == "P-small":
            small, cid = single_doc_store(store, q["doc_name"])
            dec = planner.plan(q["question"], planner.PlannerFacts(corpus_tokens=small.corpus_tokens(cid), n_docs=1))
            extra.update(path=dec.path, decided_by=dec.decided_by, receipt_stub=dec.receipt.get("stub", False))
            if dec.path == "whole_doc":
                text, st = whole_doc_answer(backend, store, q["doc_name"], q["question"], names, cfg)
            else:
                IndexRegistry(small).get_or_build(cid, "l1_bm25")
                text, st = agent_answer(backend, small.chunks(cid), q["question"], names, cfg)
            rec.emit("manifest", unit=f"manifest_run:P-small:{q['id']}", arm="P-small", qid=q["id"],
                     built_layers=sorted(small.built_layers(cid)))
        elif arm == "B-today":
            extra["route"] = "ingest_long_context (27B whole-doc; served on the np1 27B regardless)"
            text, st = whole_doc_answer(backend, store, q["doc_name"], q["question"], names, cfg)
        elif arm == "P-large":
            dec = planner.plan(q["question"], planner.PlannerFacts(
                corpus_tokens=C, n_docs=len(A.DOCS), questions_so_far=i + 1,
                available_layers=tuple(built), sql_fields_present=sql_present))
            extra.update(path=dec.path, decided_by=dec.decided_by, receipt_stub=dec.receipt.get("stub", False))
            if dec.path == "sql":
                text, st, info = sql_answer(backend, store, q["question"], names, cfg)
                extra.update(info)
            else:
                text, st = agent_answer(backend, corpus_chunks, q["question"], names, cfg, bm25=corpus_bm25)
        elif arm == "B1":
            text, st = agent_answer(backend, corpus_chunks, q["question"], names, cfg, bm25=corpus_bm25)
        else:  # B2: notes for all 5 questions first (doc-outer), then this question's aggregate
            b2_notes_prepass(rec, backend, store, agg[:5], cfg, guard)
            text, st, b2_calls, b2_pre = b2_answer(rec, backend, q, names, cfg)
        calls, pre = (b2_calls, b2_pre) if arm == "B2" else (t.calls, t.prefill)
        ok = correct(qclass, q, text) if st == "ok" else None
        rec.emit("answer", unit=unit, arm=arm, qid=q["id"], qclass=qclass, status=st, correct=ok,
                 calls=calls, prefill_tokens=pre, answer=text[-600:], final=st in ("ok", "out_of_window",
                                                                                 "budget_exhausted"),
                 wall_s=time.time() - t0, **extra)
        n += 1
        if smoke:
            return {"complete": False, "smoke": True}
    return {"complete": all(f"answer:{a}:{q['id']}" in rec.done for a, _, q in plan_items)}
