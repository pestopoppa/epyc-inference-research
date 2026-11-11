#!/usr/bin/env python3
"""AGG-1.5 mid-size retrieval parity (M5): planner ``retrieval`` path vs ``context.search``.

New module; it imports (never edits) the Phase 0 arms (``phase0_arms``: Recorder, scorer, Tally, guard) and
the pinned orchestrator tree (``src.aggregation`` planner / store / registry, ``src.repl_environment``
``ContextBundle``). Frozen rules: ``PREDECLARED_AGG15.md`` (its sha256 is passed with
``--expect-predeclared-sha`` and recorded in every results file).

Arms, one question at a time, both on the SAME model loop (SEARCH / READ / FINAL over pointers, same prompt,
same grammar, same caps): only the retrieval backend differs.

* ``P-retrieval``    the planner scopes the question to its one document (``planner.scoped_facts``), plans it
                     (``planner.plan`` -> ``retrieval``, layers L0 L1 L2), and the pointers come from the STORE:
                     ``IndexRegistry.get_or_build(corpus, 'l1_bm25')`` over the stored chunk table. The index is
                     keyed by (corpus, content hash, builder, model) and built once per document, then reused.
* ``B-context.search`` today's flow: a fresh ``ContextBundle`` per request holding the document as one section,
                     ``bundle.search`` pointers ({section, start, end}) read through ``bundle.get``. The bundle
                     builds its lazy ``ContextSearchIndex`` on its first search, once PER REQUEST (``context_bundle.py``
                     lines 580-636), so every question pays the index build again.

Dense retrieval (L2) is off in both arms (no embedder in a private job; the registry reserves ``l2_dense``);
``context.search`` then runs its documented ``lexical`` mode. No L3 / L4 index is ever built: the parity store
holds one corpus per document with L0 chunks only, and its manifest is read after every P question.

CLI (stdlib until a phase needs the orchestrator):
    --build-questions   derive the 20 questions + verification record from the frozen sheet and the PDFs
                        (deterministic, no model); needs pypdfium2 + pypdf (``--libpath`` for site-packages dirs)
    --preflight         zero compute: file hashes, PDF pins, golds, PREDECLARED sha256
    --dry-run           zero model: preflight + plan check + the full pipeline on a FAKE backend in a temp dir
                        (store build, planner scope, both retrieval backends, scoring, verdict)
    (default)           run the arms against ``--url`` (a private llama-server the job owns)
    --score             verdict only, from ``--out/results.jsonl``

Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 stop (futility / budget / plan), 4 precondition,
75 partial (deadline / stopfile; resumable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
sys.path.insert(0, str(HERE))
import agg_phase0 as A0  # noqa: E402
import phase0_arms as P0  # noqa: E402

SCHEMA = "agg15_parity_questions.v1"
CORPUS_PREFIX = "agg15:"
# the 4 documents of the 28 with 200k < C <= 1M tokens (all four are in the Phase 0 store), largest first
DOCS = ("JPMORGAN_2022_10K", "PEPSICO_2022_10K", "AMERICANWATERWORKS_2022_10K", "AES_2022_10K")
MID_LO, MID_HI = 200_000, 1_000_000
PER_DOC, MAX_PER_DOC, N_QUESTIONS = 5, 6, 20
SELECT_SALT = "agg15-v1"
ARMS = ("P-retrieval", "B-context.search")
PILOT_N = 5
PILOT_BUDGET_S = 900.0
PILOT_FUTILE_ACC = 2          # pilot accuracy difference > 2 questions
PILOT_FUTILE_PREFILL = 2000   # pilot max (P - B) prefill per question > 2k tokens
# AMENDMENT 2026-11-11 (PREDECLARED_AGG15_AMENDMENT_20261111.md, POST-DATA, operator decision): a READ returns at most
# READ_SPAN_CHARS characters in BOTH arms, the span a context.search hit covers (ContextSearchIndex CHUNK_MAX_CHARS = 800).
# r1 (PREDECLARED_AGG15.md) read a stored chunk whole, up to 12,000 characters (avg 3.1k), against B's 800-character span.
READ_SPAN_CHARS = 800
READ_CHARS = READ_SPAN_CHARS  # recorded in the meta record; r1 recorded 12000
R1_READ_CHARS = 12_000        # the amended-away r1 value, kept for the record only
ALREADY_READ = "(already read: {ptr}; its text is earlier in this conversation. Read a different pointer or answer.)"
SEARCH_K = 8
NO_L = ("L2", "L3", "L4")
STORE_DEFAULT = Path("/mnt/raid0/llm/tmp/agg0-arms-cpu-20261110-r2/out/store.sqlite")
# the box's M5 tolerances are the Phase 0 kill constants (imported, not copied)
ACC_TOL = A0.KILL["m5_acc_tol_questions"]
CALLS_TOL = A0.KILL["m5_calls_tol"]
PREFILL_TOL = A0.KILL["m5_prefill_tol"]

RULE = "End your reply with one line 'FINAL: <answer>'. For an amount, write the number in USD millions."
GRAMMAR = 'root ::= ("SEARCH: " | "READ: " | "FINAL: ") [^\\n]+'


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- question derivation
def _hkey(doc: str, metric: str) -> str:
    return hashlib.sha256(f"{SELECT_SALT}|{doc}|{metric}".encode()).hexdigest()


def select_cells(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deterministic: eligible = filled, confidence high, verified, with a quote, in the 4 documents.
    Per document the first PER_DOC by sha256(salt|doc|metric); the rest of the 20 from the leftover pool in
    hash order, at most MAX_PER_DOC per document. Returned in round-robin document order (so the pilot,
    the first 5, touches all four documents)."""
    el = [r for r in rows if r["doc_name"] in DOCS and r["status"] == "filled" and r["confidence"] == "high"
          and r["verified"] == "true" and r["quote"].strip()]
    by_doc = {d: sorted((r for r in el if r["doc_name"] == d), key=lambda r: _hkey(d, r["metric"])) for d in DOCS}
    chosen = {d: by_doc[d][:PER_DOC] for d in DOCS}
    pool = sorted((r for d in DOCS for r in by_doc[d][PER_DOC:]), key=lambda r: _hkey(r["doc_name"], r["metric"]))
    for r in pool:
        if sum(len(v) for v in chosen.values()) >= N_QUESTIONS:
            break
        if len(chosen[r["doc_name"]]) < MAX_PER_DOC:
            chosen[r["doc_name"]].append(r)
    if sum(len(v) for v in chosen.values()) != N_QUESTIONS:
        raise ValueError("cannot select %d questions: %s" % (N_QUESTIONS, {d: len(v) for d, v in chosen.items()}))
    for d in DOCS:
        chosen[d].sort(key=lambda r: _hkey(d, r["metric"]))
    out, i = [], 0
    while len(out) < N_QUESTIONS:
        for d in DOCS:
            if i < len(chosen[d]):
                out.append(chosen[d][i])
        i += 1
    return out


def label_of(quote: str) -> str:
    """The statement line label of a sheet quote: the quote without its footnote mark and trailing figures."""
    s = re.sub(r"\((?:[a-z]|\d)\)(?=\s*[$\d(])", "", quote.strip())
    return re.sub(r"[\s$()\d,.\-\u2013]+$", "", s).strip()


def make_question(qid: str, row: dict[str, str]) -> dict[str, Any]:
    value = float(row["value_usd_m"])
    label = label_of(row["quote"])
    q = (f"In {row['company']}'s Form 10-K for fiscal year 2022, what amount is reported for \"{label}\" in the "
         "consolidated financial statements for fiscal 2022? Give the absolute value in USD millions, as the "
         "statement prints it.")
    dec = 0 if value == int(value) else len(str(value).split(".")[1])
    return {"id": qid, "doc_name": row["doc_name"], "company": row["company"], "metric": row["metric"],
            "question": q, "label": label, "gold_value_usd_m": value, "page": int(row["page"]),
            "quote": row["quote"], "key": {"numbers": [{"value": abs(value), "decimals": dec}]},
            "source": "agg_phase1/facts_sheet.csv (filled, confidence high, verified)"}


def build_questions(sheet: Path, corpus: Path, tokens: dict[str, int]) -> dict[str, Any]:
    import csv  # noqa: PLC0415
    rows = list(csv.DictReader(sheet.open(newline="")))
    cells = select_cells(rows)
    qs = [make_question(f"p15_{i + 1:02d}", r) for i, r in enumerate(cells)]
    docs = {d["doc_name"]: d for d in json.loads(corpus.read_text())["docs"]}
    return {"schema": SCHEMA, "corpus_sha256": sha_file(corpus), "sheet_sha256": sha_file(sheet),
            "select_salt": SELECT_SALT, "mid_range_tokens": [MID_LO, MID_HI],
            "docs": [{"doc_name": d, "company": docs[d]["company"], "pdf": docs[d]["pdf"],
                      "pdf_sha256": docs[d]["sha256"], "n_tokens_est": tokens[d],
                      "n_questions": sum(1 for q in qs if q["doc_name"] == d)} for d in DOCS],
            "questions": qs}


def phase0_store_tokens(store: Path) -> dict[str, int]:
    c = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        return {r[0]: int(r[1]) for r in c.execute("SELECT doc_id, n_tokens_est FROM documents")}
    finally:
        c.close()


def verify_questions(qdoc: dict[str, Any], libpath: list[str], store: Path | None) -> dict[str, Any]:
    """Deterministic gold verification against the PDF text, no model: the quote is found on its stated page in
    pypdfium2 AND pypdf text, the gold equals a number printed in the quote, and (when a store is given) the quote
    is present in the stored chunk text of that document (the gold is reachable by retrieval)."""
    import verify_phase1_cells as V  # noqa: PLC0415
    sys.path[:0] = libpath
    import pypdf  # noqa: PLC0415
    import pypdfium2 as pdfium  # noqa: PLC0415
    pdfs = {d["doc_name"]: d["pdf"] for d in qdoc["docs"]}
    cache: dict[str, tuple[list[str], Any]] = {}
    sdoc: dict[str, str] = {}
    if store:
        c = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
        for d in pdfs:
            sdoc[d] = V.sq("".join(r[0] or "" for r in c.execute(
                "SELECT c.text FROM chunks c JOIN documents d ON c.doc_hash = d.doc_hash WHERE d.doc_id = ? "
                "ORDER BY c.ordinal", [d])))
        c.close()
    res = []
    for q in qdoc["questions"]:
        d = q["doc_name"]
        if d not in cache:
            pd = pdfium.PdfDocument(pdfs[d])
            cache[d] = ([pd[i].get_textpage().get_text_range() for i in range(len(pd))], pypdf.PdfReader(pdfs[d]))
        tx, rd = cache[d]
        qs = V.sq(q["quote"])
        nstr = V.number_strings(q["gold_value_usd_m"])
        r = {"qid": q["id"], "doc": d, "page": q["page"],
             "pdfium_found": qs in V.sq(tx[q["page"] - 1]),
             "pypdf_found": qs in V.sq(rd.pages[q["page"] - 1].extract_text()),
             "value_in_quote": any(n in q["quote"] for n in nstr),
             "pages_with_value": sum(1 for t in tx if any(n in t for n in nstr)),
             "label_in_quote": q["label"] in q["quote"]}
        if store:
            r["quote_in_store_text"] = qs in sdoc[d]
        r["verified"] = bool(r["pdfium_found"] and r["pypdf_found"] and r["value_in_quote"]
                             and r["label_in_quote"] and r.get("quote_in_store_text", True))
        res.append(r)
    return {"schema": "agg15_parity_verification.v1", "method": "deterministic: PDF text (pypdfium2 + pypdf) and "
            "stored chunk text; no model", "n": len(res), "n_verified": sum(r["verified"] for r in res),
            "questions": {r["qid"]: r for r in res}}


# --------------------------------------------------------------------------- scoring / verdict
def _final_answers(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in records:
        if r.get("kind") == "answer" and r.get("final") and r.get("arm") in ARMS:
            by.setdefault(r["qid"], {})[r["arm"]] = r
    return by


def paired_stats(records: list[dict[str, Any]], qids: list[str] | None = None) -> dict[str, Any]:
    by = _final_answers(records)
    common = sorted(q for q, a in by.items() if len(a) == 2 and (qids is None or q in qids))
    p = {q: by[q]["P-retrieval"] for q in common}
    b = {q: by[q]["B-context.search"] for q in common}
    ok = lambda rs: sum(1 for r in rs.values() if r.get("correct") is True)  # noqa: E731
    return {"n": len(common), "acc_P": ok(p), "acc_B": ok(b),
            "acc_diff": abs(ok(p) - ok(b)) if common else None,
            "max_extra_calls": max((p[q]["calls"] - b[q]["calls"] for q in common), default=None),
            "max_extra_prefill": max((p[q]["prefill_tokens"] - b[q]["prefill_tokens"] for q in common), default=None),
            "median_calls": {"P": A0._median([p[q]["calls"] for q in common]),
                             "B": A0._median([b[q]["calls"] for q in common])},
            "median_prefill": {"P": A0._median([p[q]["prefill_tokens"] for q in common]),
                               "B": A0._median([b[q]["prefill_tokens"] for q in common])},
            "wall_s": sum(r.get("wall_s", 0.0) for rs in (p, b) for r in rs.values())}


def score(records: list[dict[str, Any]], n_expected: int = N_QUESTIONS) -> dict[str, Any]:
    """The M5 parity verdict (zero model). Components: accuracy, calls, prefill, no_l3_l4, no_rebuild_per_request.
    ``verdict``: ``parity_holds`` | ``phase1_gate_tripped`` (``failing`` names the components) | ``incomplete``."""
    st = paired_stats(records)
    by = _final_answers(records)
    p_rows = [r for q in by.values() for a, r in q.items() if a == "P-retrieval"]
    b_rows = [r for q in by.values() for a, r in q.items() if a == "B-context.search"]
    built = sorted({l for r in records if r.get("kind") == "manifest" and r.get("arm") == "P-retrieval"
                    for l in r.get("built_layers", []) if l in NO_L})
    seen: dict[str, int] = {}
    for r in sorted(p_rows, key=lambda r: r["qid"]):
        seen[r["doc"]] = seen.get(r["doc"], 0) + int(r.get("index_builds", 0))
    p_builds, b_builds = sum(seen.values()), sum(int(r.get("index_builds", 0)) for r in b_rows)
    p_s, b_s = sum(r.get("index_build_s", 0.0) for r in p_rows), sum(r.get("index_build_s", 0.0) for r in b_rows)
    comp = {"accuracy": st["n"] > 0 and st["acc_diff"] <= ACC_TOL,
            "calls": st["n"] > 0 and st["max_extra_calls"] <= CALLS_TOL,
            "prefill": st["n"] > 0 and st["max_extra_prefill"] <= PREFILL_TOL,
            "no_l3_l4": not built and any(r.get("kind") == "manifest" for r in records),
            "no_rebuild_per_request": bool(p_rows) and all(v <= 1 for v in seen.values())
            and p_builds <= len(seen)}
    failing = [k for k, v in comp.items() if not v]
    complete = st["n"] >= n_expected
    verdict = "incomplete" if not complete else ("phase1_gate_tripped" if failing else "parity_holds")
    return {"n_pairs": st["n"], "acc_P": st["acc_P"], "acc_B": st["acc_B"], "acc_diff": st["acc_diff"],
            "max_extra_calls": st["max_extra_calls"], "max_extra_prefill": st["max_extra_prefill"],
            "median_calls": st["median_calls"], "median_prefill": st["median_prefill"],
            "indexes_built": built,
            "index_builds": {"P": p_builds, "B": b_builds, "P_per_doc": seen},
            "index_build_s": {"P": p_s, "B": b_s},
            "saving": {"index_builds": b_builds - p_builds, "index_build_s": b_s - p_s},
            "tolerances": {"acc_questions": ACC_TOL, "calls": CALLS_TOL, "prefill": PREFILL_TOL},
            "components": comp, "failing": failing, "complete": complete, "verdict": verdict}


def pilot_check(records: list[dict[str, Any]], pilot_qids: list[str]) -> dict[str, Any]:
    st = paired_stats(records, pilot_qids)
    fut = []
    if st["n"] == len(pilot_qids):
        if st["acc_diff"] > PILOT_FUTILE_ACC:
            fut.append("accuracy")
        if st["max_extra_prefill"] > PILOT_FUTILE_PREFILL:
            fut.append("prefill")
    over = st["wall_s"] > PILOT_BUDGET_S
    return {**st, "futility": fut, "over_budget": over, "budget_s": PILOT_BUDGET_S}


# --------------------------------------------------------------------------- stores and pointers
def build_parity_store(src: Path, dst_path: Path, doc_ids: tuple[str, ...]):
    """L0-only store: one corpus per document ('agg15:<doc_id>') copied from the Phase 0 store (read-only);
    result_json is dropped (no L3 text rows are ever read). Idempotent. Returns the CorpusStore."""
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    dst = CorpusStore(dst_path)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        for d in doc_ids:
            cid = CORPUS_PREFIX + d
            if dst.corpus_docs(cid):
                continue
            rows = s.execute("SELECT * FROM documents WHERE doc_id = ?", [d]).fetchall()
            if len(rows) != 1:
                raise ValueError(f"{d}: {len(rows)} documents in {src}")
            row = list(rows[0])
            row[1], row[8] = cid, None
            dst.conn.execute(f"INSERT INTO documents VALUES ({','.join('?' * len(row))})", row)
            for c in s.execute("SELECT * FROM chunks WHERE doc_hash = ?", [row[0]]).fetchall():
                dst.conn.execute(f"INSERT INTO chunks VALUES ({','.join('?' * len(c))})", list(c))
            dst.conn.commit()
    finally:
        s.close()
    return dst


def plan_question(store, planner, q: dict[str, Any], docs_meta: list[dict[str, Any]]) -> dict[str, Any]:
    pf, why = planner.scoped_facts(q["question"], docs_meta, questions_so_far=1, available_layers=(),
                                   sql_fields_present=False)
    dec = planner.plan(q["question"], pf)
    return {"scope_doc": pf.scope_doc, "scope_reason": why, "corpus_tokens": pf.corpus_tokens,
            "path": dec.path, "layers": list(dec.layers), "decided_by": dec.decided_by,
            "planner_llm_called": dec.llm_called}


_WORD = re.compile(r"[a-z0-9]+")


def hit_window(text: str, query: str, width: int = READ_SPAN_CHARS) -> str:
    """The ``width``-character window of ``text`` around the best lexical hit of ``query`` (the P arm's read span, the
    counterpart of a context.search hit span). Candidates are centred on each occurrence of a query word; a window is
    ranked by (the whole query appears as a phrase, distinct query words, query-word occurrences), ties to the earliest.
    No query word present: the start of the text. Pure and deterministic; ``len(result) <= width``."""
    if len(text) <= width:
        return text
    words = [w for w in _WORD.findall(query.lower()) if len(w) > 1]
    if not words:
        return text[:width]
    uniq = sorted(set(words))
    low = re.sub(r"\s+", " ", text.lower())
    if len(low) != len(text):  # whitespace folding must not shift offsets: fall back to the raw lower-cased text
        low = text.lower()
    phrase = " ".join(words)
    best, best_start = (-1, -1, -1), 0
    centres = sorted({m.start() for w in uniq for m in re.finditer(re.escape(w), low)})
    for c in centres:
        start = min(max(c - width // 2, 0), len(text) - width)
        win = low[start:start + width]
        score = (int(phrase in win), sum(1 for w in uniq if w in win), sum(win.count(w) for w in uniq))
        if score > best:
            best, best_start = score, start
    return text[best_start:best_start + width] if best[1] > 0 else text[:width]


class PArm:
    """Planner ``retrieval`` pointers from the store's L1 index (built once per (corpus, content hash))."""

    def __init__(self, store, registry, doc_id: str) -> None:
        self.store, self.registry, self.cid = store, registry, CORPUS_PREFIX + doc_id
        self.chunks = {c["chunk_id"]: c for c in store.chunks(self.cid)}
        self.builds, self.build_s = 0, 0.0
        self.dup_reads = 0
        self.hit_query: dict[str, str] = {}  # chunk id -> the last SEARCH query whose hits returned it

    def open(self) -> None:
        key = self.registry.key(self.cid, "l1_bm25")
        fresh = key not in self.registry._live
        t0 = time.perf_counter()
        self.handle = self.registry.get_or_build(self.cid, "l1_bm25")
        self.builds = int(fresh)
        self.build_s = time.perf_counter() - t0 if fresh else 0.0

    def search(self, query: str) -> list[tuple[str, str]]:
        out = []
        for cid, _score in self.handle.search(query, SEARCH_K):
            c = self.chunks[cid]
            self.hit_query[cid] = query
            out.append((cid, f"{(c.get('title') or '')[:80]} | {(c['text'] or '')[:200]}"))
        return out

    def read_key(self, ptr: str) -> Any:
        ptr = ptr.strip()
        return ptr if ptr in self.chunks else None

    def read(self, ptr: str) -> str | None:
        """At most READ_SPAN_CHARS characters of the chunk, the window around the hit for the query that returned it
        (the same read span a context.search hit covers). A pointer never returned by a search has no hit: its start."""
        c = self.chunks.get(ptr.strip())
        if c is None:
            return None
        return hit_window(c["text"] or "", self.hit_query.get(ptr.strip(), ""), READ_SPAN_CHARS)


class BArm:
    """Today's ``context.search``: a fresh ContextBundle per request; its lazy index builds on first search."""

    def __init__(self, doc_id: str, text: str) -> None:
        from src.repl_environment.context_bundle import ContextBundle, Section  # noqa: PLC0415
        self.doc_id = doc_id
        self.bundle = ContextBundle([Section(doc_id, text, "text", False, "10-K filing text")], search_enabled=True)
        self.bundle.set_embedder(None)  # dense off: the documented lexical mode, same as P (no L2)
        self.builds, self.build_s = 0, 0.0
        self.dup_reads = 0
        self._text = text

    def open(self) -> None:
        pass

    def search(self, query: str) -> list[tuple[str, str]]:
        t0 = time.perf_counter()
        hits = self.bundle.search(query, SEARCH_K)
        if self.builds == 0 and getattr(self.bundle, "_search_index", None) is not None:
            self.builds = 1
            self.build_s = time.perf_counter() - t0  # first search = lazy index build + one query
        return [(f"{h['start']}-{h['end']}", f"line {h['line']} | {self._text[h['start']:h['end']][:200]}")
                for h in hits]

    @staticmethod
    def _span(ptr: str) -> tuple[int, int] | None:
        m = re.fullmatch(r"(\d+)-(\d+)", ptr.strip())
        if not m:
            return None
        s, e = int(m.group(1)), int(m.group(2))
        return s, s + min(max(e - s, 0), READ_SPAN_CHARS)

    def read_key(self, ptr: str) -> Any:
        return self._span(ptr)

    def read(self, ptr: str) -> str | None:
        sp = self._span(ptr)
        if sp is None:
            return None
        return self.bundle.get(self.doc_id, max_chars=sp[1] - sp[0], offset=sp[0])


def retrieval_loop(backend, question: str, arm, cfg: P0.Cfg, trace: list[str]) -> tuple[str, str]:
    """The shared SEARCH / READ / FINAL loop (<= single_doc_max_calls); identical for both arms, repeated-read guard included."""
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    head = ("You answer a question about ONE FY2022 10-K filing using tools. Each turn reply with exactly one "
            "line: 'SEARCH: <keywords>' (lexical search of the filing, returns pointers), 'READ: <pointer>' "
            f"(returns the text at that pointer) or 'FINAL: <answer>'.\n{RULE}\nQuestion: {question}\n")
    hist, seen = "", set()
    for _ in range(cfg.single_doc_max_calls):
        try:
            r = backend.chat(head + hist + "Next action:", n_predict=120, grammar=GRAMMAR, id_slot=0)
        except OutOfWindow:
            return "", "out_of_window"
        act = r["text"].strip()
        trace.append(act[:120])
        hist += f"\n> {act}\n"
        if act.startswith("FINAL:"):
            return act, "ok"
        arg = act.split(":", 1)[1].strip() if ":" in act else ""
        if act.startswith("SEARCH:"):
            hits = arm.search(arg) if arg else []
            hist += "\n".join(f"[{p}] {s.replace(chr(10), ' ')}" for p, s in hits) or "(no hits)"
        elif act.startswith("READ:"):
            key = arm.read_key(arg)
            if key is not None and key in seen:  # AMENDMENT 2026-11-11: a repeated READ costs a note, not the text
                arm.dup_reads += 1
                hist += ALREADY_READ.format(ptr=arg)
                continue
            t = arm.read(arg)
            if t is not None and key is not None:
                seen.add(key)
            hist += t if t is not None else "(unknown pointer)"
    return "", "budget_exhausted"


# --------------------------------------------------------------------------- run
def run_arms(rec: P0.Recorder, src_store: Path, parity_path: Path, backend, qdoc: dict[str, Any], *,
             guard: Callable[[], None], smoke: bool = False, cfg: P0.Cfg = P0.CFG) -> dict[str, Any]:
    from src.aggregation import planner  # noqa: PLC0415
    from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
    qs = qdoc["questions"]
    store = build_parity_store(src_store, parity_path, DOCS)
    m0 = [m for d in DOCS for m in store.manifest(CORPUS_PREFIX + d) if m["layer"] in NO_L]  # L1 is the P arm's own
    if "store" not in rec.done:
        rec.emit("store", unit="store", path=str(parity_path), forbidden_manifest_rows=len(m0),
                 docs={d["doc_name"]: d["n_tokens_est"] for d in qdoc["docs"]})
    docs_meta = [d for d in store.conn.execute(
        "SELECT doc_id, entity, n_tokens_est FROM documents").fetchall()]
    docs_meta = [{"doc_id": a, "entity": b, "n_tokens_est": c} for a, b, c in docs_meta]
    bad = []
    for q in qs:  # zero-LLM plan check first: every question must scope to ONE mid-size doc and plan `retrieval`
        unit = f"plan:{q['id']}"
        pl = plan_question(store, planner, q, docs_meta)
        ok = (pl["path"] == "retrieval" and pl["scope_doc"] == q["doc_name"] and pl["scope_reason"] == "single_entity"
              and MID_LO < pl["corpus_tokens"] <= MID_HI and not pl["planner_llm_called"])
        if unit not in rec.done:
            rec.emit("plan", unit=unit, qid=q["id"], ok=ok, **pl)
        if not ok:
            bad.append(q["id"])
    if m0 or bad:
        rec.emit("stop", reason="futility_plan_or_manifest", bad_plan=bad, forbidden_manifest_rows=len(m0), final=False)
        return {"complete": True, "futility_stop": "plan", "bad_plan": bad, "forbidden_manifest_rows": len(m0)}
    prior = next((r for r in rec.records if r.get("unit") == "verdict" and r.get("stage") == "pilot"
                  and r.get("verdict") == "phase1_gate_tripped"), None)
    if prior:  # a futility verdict stands: a rerun never runs the remaining questions past it
        return {"complete": True, "verdict": {k: v for k, v in prior.items() if k not in ("kind", "ts")}}
    registry = IndexRegistry(store)
    pilot_ids = [q["id"] for q in qs[:PILOT_N]]
    chunks_by_doc = {d: "\n\n".join(c["text"] or "" for c in store.chunks(CORPUS_PREFIX + d)) for d in DOCS}
    for q in qs:
        for arm_name in ARMS:
            unit = f"answer:{arm_name}:{q['id']}"
            if unit in rec.done:
                continue
            guard()
            t0, t = time.time(), P0.Tally(backend)
            arm = PArm(store, registry, q["doc_name"]) if arm_name == "P-retrieval" else BArm(q["doc_name"], chunks_by_doc[q["doc_name"]])
            arm.open()
            trace: list[str] = []
            text, st = retrieval_loop(backend, q["question"], arm, cfg, trace)
            correct = P0.score_single(q["key"], text) if st == "ok" else None
            if arm_name == "P-retrieval":
                rec.emit("manifest", unit=f"manifest:{q['id']}", arm="P-retrieval", qid=q["id"],
                         built_layers=sorted(store.built_layers(CORPUS_PREFIX + q["doc_name"])))
            rec.emit("answer", unit=unit, arm=arm_name, qid=q["id"], doc=q["doc_name"], qclass="single",
                     status=st, correct=correct, calls=t.calls, prefill_tokens=t.prefill, answer=text[-300:],
                     index_builds=arm.builds, index_build_s=arm.build_s, dup_reads=arm.dup_reads, loop_actions=trace,
                     final=st in ("ok", "out_of_window", "budget_exhausted"), wall_s=time.time() - t0)
        if smoke:
            return {"complete": False, "smoke": True}
        if q["id"] == pilot_ids[-1] and "pilot" not in rec.done:
            pc = pilot_check(rec.records, pilot_ids)
            rec.emit("pilot", unit="pilot", **pc)
            if pc["futility"] or pc["over_budget"]:
                reason = "budget_stop_rebook" if pc["over_budget"] and not pc["futility"] else "phase1_gate_tripped"
                v = {"verdict": reason, "stage": "pilot", "failing": pc["futility"], "pilot": pc}
                rec.emit("verdict", unit="verdict", final=reason == "phase1_gate_tripped", **v)
                return {"complete": True, "verdict": v}
    s = score(rec.records)
    if s["complete"] and "verdict" not in rec.done:
        rec.emit("verdict", unit="verdict", stage="full", **s)
    return {"complete": s["complete"], "verdict": s}


# --------------------------------------------------------------------------- fake backend (dry-run, tests)
class FakeBackend:
    """No model. SEARCH the quoted label, READ the first pointer, FINAL the figure printed after the label
    (0 when absent). Bills len(prompt)//4 tokens. Exercises both retrieval backends for real."""

    def __init__(self) -> None:
        self.calls = self.prefill_tokens = self.cache_n = 0
        self._last = ""

    def chat(self, prompt, *, n_predict, grammar=None, cache_prompt=True, id_slot=None, logprobs=None):
        self.calls += 1
        lcp = len(os.path.commonprefix([self._last, prompt])) if cache_prompt else 0  # slot-cache prefix reuse
        self._last = prompt
        self.prefill_tokens += (len(prompt) - lcp) // 4
        label = (re.search(r'what amount is reported for "([^"]+)"', prompt) or re.search(r'"([^"]+)"', prompt)).group(1)
        acts = re.findall(r"\n> (SEARCH|READ|FINAL): ?", prompt)
        if not acts:
            text = f"SEARCH: {label}"
        elif acts[-1] == "SEARCH":
            m = re.search(r"\n\[([^\]]+)\]", prompt.rsplit("> SEARCH:", 1)[1])
            text = f"READ: {m.group(1)}" if m else "FINAL: 0"
        else:
            body = prompt.rsplit("> READ:", 1)[1]
            m = re.search(re.escape(label) + r"[^\d\n]{0,12}\(?([\d,]+(?:\.\d+)?)", body, re.I)
            text = f"FINAL: {m.group(1)}" if m else "FINAL: 0"
        return {"text": text, "prompt_tokens": len(prompt) // 4, "cache_n": 0, "finish_reason": "stop"}


# --------------------------------------------------------------------------- preflight / CLI
def load_questions(d: Path) -> dict[str, Any]:
    return json.loads((d / "parity_questions.json").read_text())


def preflight(d: Path, expect_sha: str | None = None) -> tuple[bool, dict[str, Any]]:
    problems: list[str] = []
    qdoc = load_questions(d)
    ver = json.loads((d / "parity_verification.json").read_text())
    pre = sha_file(d / "PREDECLARED_AGG15.md")
    if expect_sha and pre != expect_sha:
        problems.append(f"PREDECLARED_AGG15.md sha256 {pre} != expected {expect_sha}")
    if qdoc.get("schema") != SCHEMA or len(qdoc["questions"]) != N_QUESTIONS:
        problems.append("question file schema or count")
    c1 = json.loads((d / "corpus.json").read_text())
    if qdoc["corpus_sha256"] != sha_file(d / "corpus.json") or c1["n_docs"] != 28:
        problems.append("corpus.json sha256 differs from the one the questions were built on")
    pins = {x["doc_name"]: x for x in c1["docs"]}
    for dd in qdoc["docs"]:
        p = Path(dd["pdf"])
        if pins[dd["doc_name"]]["sha256"] != dd["pdf_sha256"]:
            problems.append(f"pin mismatch {dd['doc_name']}")
        elif not p.exists() or A0.sha256_file(p) != dd["pdf_sha256"]:
            problems.append(f"pdf missing or sha256 mismatch {dd['doc_name']}")
        if not (MID_LO < dd["n_tokens_est"] <= MID_HI):
            problems.append(f"{dd['doc_name']} outside 200k < C <= 1M")
    for q in qdoc["questions"]:
        v = ver["questions"].get(q["id"], {})
        if not v.get("verified"):
            problems.append(f"{q['id']} not verified")
    info = {"predeclared_sha256": pre, "questions_sha256": sha_file(d / "parity_questions.json"),
            "verification_sha256": sha_file(d / "parity_verification.json"), "n_questions": len(qdoc["questions"]),
            "n_verified": sum(1 for q in qdoc["questions"] if ver["questions"].get(q["id"], {}).get("verified")),
            "docs": {x["doc_name"]: x["n_questions"] for x in qdoc["docs"]}, "problems": problems}
    return not problems, info


def reachability(qdoc: dict[str, Any], store, registry) -> dict[str, Any]:
    """Zero model: does each arm's top-k for the quoted label contain the gold quote? (informational)"""
    import verify_phase1_cells as V  # noqa: PLC0415
    chunks = {d: "\n\n".join(c["text"] or "" for c in store.chunks(CORPUS_PREFIX + d)) for d in DOCS}
    out = {a: 0 for a in ARMS}
    for q in qdoc["questions"]:
        qs = V.sq(q["quote"])
        pa = PArm(store, registry, q["doc_name"])
        pa.open()
        top = [pa.chunks[cid]["text"] or "" for cid, _ in pa.handle.search(q["label"], SEARCH_K)]
        out["P-retrieval"] += any(qs in V.sq(t) for t in top)
        ba = BArm(q["doc_name"], chunks[q["doc_name"]])
        hits = ba.bundle.search(q["label"], SEARCH_K)
        out["B-context.search"] += any(qs in V.sq(chunks[q["doc_name"]][h["start"]:h["end"]]) for h in hits)
    return out


def dry_run(d: Path, store_src: Path, expect_sha: str | None) -> tuple[int, dict[str, Any]]:
    ok, info = preflight(d, expect_sha)
    if not ok:
        return 2, {"preflight": info}
    if not store_src.exists():
        return 4, {"error": f"no Phase 0 store at {store_src}", "preflight": info}
    qdoc = load_questions(d)
    global PILOT_FUTILE_PREFILL, PREFILL_TOL
    saved = PILOT_FUTILE_PREFILL, PREFILL_TOL
    PILOT_FUTILE_PREFILL = PREFILL_TOL = 10**9  # the fake model's token bill is meaningless: exercise all 20 questions
    try:
        return _dry_run(d, store_src, info, qdoc)
    finally:
        PILOT_FUTILE_PREFILL, PREFILL_TOL = saved


def _dry_run(d: Path, store_src: Path, info: dict[str, Any], qdoc: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="agg15-dry-") as tmp:
        from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
        rec = P0.Recorder(Path(tmp) / "results.jsonl")
        res = run_arms(rec, store_src, Path(tmp) / "parity_store.sqlite", FakeBackend(), qdoc, guard=lambda: None)
        store = build_parity_store(store_src, Path(tmp) / "parity_store.sqlite", DOCS)
        reach = reachability(qdoc, store, IndexRegistry(store))
        plans = [r for r in rec.records if r["kind"] == "plan"]
        return 0, {"fake_backend": True, "note": "plumbing check only: a fake model reads the retrieved text; no accuracy, "
                   "cost or parity claim, and the prefill gates are disabled so all 20 questions run", "preflight": info, "plans_ok": sum(1 for r in plans if r["ok"]),
                   "plans": len(plans), "paths": sorted({r["path"] for r in plans}),
                   "reachability_top_k": {**reach, "of": len(qdoc["questions"])},
                   "result": res.get("verdict") or res, "pilot": next((r for r in rec.records if r["kind"] == "pilot"), None)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--store-src", type=Path, default=STORE_DEFAULT, help="the Phase 0 store (read-only)")
    ap.add_argument("--parity-store", type=Path)
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--orch-root", type=Path)
    ap.add_argument("--libpath", action="append", default=[])
    ap.add_argument("--expect-predeclared-sha")
    ap.add_argument("--build-questions", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.orch_root:
        sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    d = a.set_dir
    if a.build_questions:
        qdoc = build_questions(d / "facts_sheet.csv", d / "corpus.json", phase0_store_tokens(a.store_src))
        (d / "parity_questions.json").write_text(json.dumps(qdoc, indent=1) + "\n")
        v = verify_questions(qdoc, a.libpath, a.store_src)
        (d / "parity_verification.json").write_text(json.dumps(v, indent=1) + "\n")
        print(json.dumps({"n": len(qdoc["questions"]), "n_verified": v["n_verified"],
                          "docs": {x["doc_name"]: x["n_questions"] for x in qdoc["docs"]}}))
        return 0 if v["n_verified"] == v["n"] else 2
    if a.preflight:
        ok, info = preflight(d, a.expect_predeclared_sha)
        print(json.dumps(info, indent=1))
        return 0 if ok else 2
    if a.dry_run:
        rc, out = dry_run(d, a.store_src, a.expect_predeclared_sha)
        print(json.dumps(out, indent=1, default=str))
        return rc
    if a.out is None:
        ap.error("--out is required to run or score")
    if a.score:
        recs = [json.loads(l) for l in (a.out / "results.jsonl").read_text().splitlines() if l.strip()]
        print(json.dumps(score(recs), indent=1, default=str))
        return 0
    ok, info = preflight(d, a.expect_predeclared_sha)
    print(json.dumps(info, indent=1))
    if not ok:
        return 2
    if not a.url or not a.parity_store:
        ap.error("--url and --parity-store are required to run")
    if not a.store_src.exists():
        return 4
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P0.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
    backend = LlamaServerBackend(a.url, model_id=a.model_id)
    rec = P0.Recorder(a.out / "results.jsonl")
    rec.emit("meta", smoke=smoke, preflight=info, url=a.url, model_id=a.model_id, cfg=dict(vars(P0.CFG)),
             read_chars=READ_CHARS, read_span_chars=READ_SPAN_CHARS, search_k=SEARCH_K, final=False)
    try:
        res = run_arms(rec, a.store_src, a.parity_store, backend, load_questions(d), guard=guard, smoke=smoke)
    except P0.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return 75
    res.update(predeclared_sha256=info["predeclared_sha256"])
    if not smoke:
        (a.out / "result.parity.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    if res.get("futility_stop") or (res.get("verdict") or {}).get("verdict") == "budget_stop_rebook":
        return 3
    return 0 if (res.get("complete") or smoke) else 75


if __name__ == "__main__":
    sys.exit(main())
