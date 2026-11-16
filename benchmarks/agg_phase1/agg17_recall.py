#!/usr/bin/env python3
"""AGG-1.7: ``context.search`` lexical recall on the AGG-1.5 labels (ZERO model, no server, no GPU).

Question (PREDECLARED_AGG17.md): why is the gold quote in the top 8 passage hits for only 16 of the 20 AGG-1.5 label
queries (misses p15_09, p15_10, p15_11, p15_16), and does a cheap ranking change raise recall@8 on the 20 labels
without losing any of the recorded r1/r2 SEARCH strings (``agg15_trace_queries_r1_r2.json``)?

The engine is the orchestrator's own (pinned tree): ``chunk_sections`` (800-char line-aware passages), ``tokenize`` and
``BM25Index``, dense off. The baseline is ``ContextSearchIndex.search`` (lexical) over the same corpus text the P arm
and ``context.search`` index (stored chunks of the r3 parity store, joined by blank lines). Candidates are RESEARCH-SIDE
re-rankers over that engine's own BM25 candidates; each carries a new builder-version label
(``l1_passage_bm25.v2-cand-*``) and never edits ``l1_passage_bm25.v1`` (the AGG-1.5 parity claim stays on v1).

Phases (in order; every phase result is a file in --out, so a rerun resumes and skips what is done):
  baseline   -> recall@8 for the labels and the trace queries; asserts 16/20 and the 4 recorded misses (else rc 4)
  diagnose   -> per miss: gold passage(s), its rank and BM25 score, the top-8 cutoff, per-term tf/idf in gold vs top 1
  candidates -> c1_label_norm, c2_line_label, c3_norm_line_label, c4_numeric_row (predeclared order); STOP-ON-ANSWER
                after the first candidate at 20/20 labels with no trace query lost; l3_row_pointers and dense_l2 are
                recorded as not available (checked, not assumed)
  verdict    -> result.agg17.json (complete, verdict answer|futility|partial_gain)

rc 0 complete / smoke ok, 4 baseline does not reproduce (infrastructure stop), 5 STOP file seen (resumable), 2 inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "agg17_recall.v1"
HERE = Path(__file__).resolve().parent
CORPUS_PREFIX = "agg15:"
DOCS = ("JPMORGAN_2022_10K", "PEPSICO_2022_10K", "AMERICANWATERWORKS_2022_10K", "AES_2022_10K")
K = 8
EXPECT_BASE = 16
EXPECT_MISSES = ["p15_09", "p15_10", "p15_11", "p15_16"]
CAND_ORDER = ("c1_label_norm", "c2_line_label", "c3_norm_line_label", "c4_numeric_row")
#: c1: statement boilerplate removed from a query (predeclared; designed from the 45 trace strings' vocabulary only)
STOP_TERMS = frozenset({"consolidated", "statement", "statements", "financial", "sheet", "sheets", "the", "of", "in",
                        "for", "and", "millions", "fiscal", "year", "ended", "december", "31", "2022", "2021"})
#: c1 never drops a query to nothing (falls back to the raw tokens)
NUM_TOKEN = re.compile(r"^\(?\$?\(?[\d,]+(\.\d+)?\)?%?$")


def sq(s: str) -> str:  # == verify_phase1_cells.sq (the AGG-1.5 gold test)
    return re.sub(r"[\s$]+", "", s)


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rec(out: Path, kind: str, **kw: Any) -> None:
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps({"kind": kind, "ts": time.time(), "unit": kw.pop("unit", kind), **kw}) + "\n")


def stop_seen(stopfile: str | None) -> bool:
    return bool(stopfile) and os.path.exists(stopfile)


# ------------------------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, orch: Path, store: Path):
        sys.path.insert(0, str(orch))
        os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
        from src.aggregation.store import CorpusStore  # noqa: PLC0415
        from src.repl_environment import context_search as cs  # noqa: PLC0415
        self.cs = cs
        st = CorpusStore(store)
        self.text = {d: "\n\n".join(c["text"] or "" for c in st.chunks(CORPUS_PREFIX + d)) for d in DOCS}
        st.close()
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
        try:
            self.l3_rows = con.execute("SELECT COUNT(*) FROM tables_").fetchone()[0]
        except sqlite3.Error:
            self.l3_rows = None
        finally:
            con.close()
        self.idx: dict[str, Any] = {}
        self.build_s: dict[str, float] = {}
        for d in DOCS:
            t0 = time.perf_counter()
            self.idx[d] = cs.ContextSearchIndex([_Sec(d, self.text[d])])
            self.build_s[d] = round(time.perf_counter() - t0, 3)

    def base(self, doc: str, q: str) -> list[int]:
        ix = self.idx[doc]
        hits, _ = ix.search(q[: self.cs.QUERY_MAX_CHARS], K, section=None, embedder=None)
        by = {(c.start, c.end): i for i, c in enumerate(ix.chunks)}
        return [by[(h["start"], h["end"])] for h in hits]

    def bm25(self, doc: str, q: str) -> list[tuple[int, float]]:
        return self.idx[doc].bm25.scores(q)

    def gold_ids(self, doc: str, quote: str) -> list[int]:
        g = sq(quote)
        return [i for i, c in enumerate(self.idx[doc].chunks) if g in sq(c.text)]


class _Sec:
    def __init__(self, name: str, text: str):
        self.name, self.text = name, text


# ------------------------------------------------------------------------------------------- candidates
def norm_query(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9_]+", q.replace('"', " "))
    keep = [t for t in toks if t.lower() not in STOP_TERMS]
    if not keep:
        keep = [t for t in toks if t.lower() != "the"] or toks
    return " ".join(keep) or q


def line_label_re(q: str) -> re.Pattern[str] | None:
    toks = re.findall(r"[A-Za-z0-9]+", q.replace('"', " "))
    if not toks:
        return None
    phrase = r"[^A-Za-z0-9\n]+".join(re.escape(t) for t in toks)
    # a table row: the phrase starts the line (optional footnote mark), then a number, $ or ( on the same line
    return re.compile(r"(?im)^[ \t]*" + phrase + r"(\s*\([a-z0-9]\))?[ \t:]*[$(]?\s*[\d(]")


def numeric_frac(text: str) -> float:
    toks = text.split()
    return sum(1 for t in toks if NUM_TOKEN.match(t)) / max(1, len(toks))


def rank(eng: Engine, cand: str, doc: str, q: str) -> list[int]:
    ix = eng.idx[doc]
    if cand == "c1_label_norm":
        return [i for i, _ in eng.bm25(doc, norm_query(q))[:K]]
    qq = norm_query(q) if cand == "c3_norm_line_label" else q
    pool = eng.bm25(doc, qq)[: eng.cs.CANDIDATES_PER_RANKER]
    if cand in ("c2_line_label", "c3_norm_line_label"):
        rx = line_label_re(qq)
        key = [(-(1 if rx and rx.search(ix.chunks[i].text) else 0), -s, i) for i, s in pool]
    elif cand == "c4_numeric_row":
        key = [(0, -(s * (1.0 + numeric_frac(ix.chunks[i].text))), i) for i, s in pool]
    else:
        raise ValueError(cand)
    return [k[2] for k in sorted(key)][:K]


# ------------------------------------------------------------------------------------------- evaluation
def evaluate(eng: Engine, ranker, labels: list[dict], traces: list[dict]) -> dict[str, Any]:
    t0 = time.perf_counter()
    lab = {q["id"]: bool(set(ranker(q["doc_name"], q["label"])) & set(q["_gold"])) for q in labels}
    tr = {t["key"]: bool(set(ranker(t["doc"], t["query"])) & set(t["_gold"])) for t in traces}
    n = len(labels) + len(traces)
    return {"labels_hit": sum(lab.values()), "labels_of": len(lab), "label_misses": sorted(k for k, v in lab.items() if not v),
            "trace_hit": sum(tr.values()), "trace_of": len(tr), "trace_misses": sorted(k for k, v in tr.items() if not v),
            "trace_records_hit": sum(tr[t["key"]] * t["n_records"] for t in traces),
            "trace_records_of": sum(t["n_records"] for t in traces),
            "ms_per_query": round((time.perf_counter() - t0) * 1000 / max(1, n), 3), "_lab": lab, "_tr": tr}


def diagnose(eng: Engine, q: dict) -> dict[str, Any]:
    doc, ix = q["doc_name"], eng.idx[q["doc_name"]]
    sc = eng.bm25(doc, q["label"])
    pos = {i: r for r, (i, _) in enumerate(sc)}
    terms = [t for t in dict.fromkeys(eng.cs.tokenize(q["label"]))]
    top1 = sc[0][0] if sc else None

    def tf(i: int | None) -> dict[str, int]:
        if i is None:
            return {}
        c = Counter(eng.cs.tokenize(ix.chunks[i].text))
        return {t: c.get(t, 0) for t in terms}
    golds = []
    for g in q["_gold"]:
        c = ix.chunks[g]
        golds.append({"chunk": g, "start": c.start, "line": c.line, "len_tokens": len(eng.cs.tokenize(c.text)),
                      "bm25_rank": (pos[g] + 1) if g in pos else None, "bm25_score": dict(sc).get(g), "tf": tf(g),
                      "line_label_row": bool(line_label_re(q["label"]) and line_label_re(q["label"]).search(c.text))})
    return {"qid": q["id"], "doc": doc, "label": q["label"], "query_tokens": terms,
            "idf": {t: round(ix.bm25._idf.get(t, 0.0), 4) for t in terms},
            "n_passages": len(ix.chunks), "n_positive": len(sc), "cutoff_score_rank8": sc[K - 1][1] if len(sc) >= K else None,
            "top1": {"chunk": top1, "score": sc[0][1] if sc else None, "tf": tf(top1),
                     "len_tokens": len(eng.cs.tokenize(ix.chunks[top1].text)) if top1 is not None else None,
                     "head": ix.chunks[top1].text[:160] if top1 is not None else None},
            "n_passages_with_all_terms": sum(1 for i, _ in sc if all(v > 0 for v in tf(i).values())),
            "gold": golds, "gold_split_across_passages": not golds}


def load_inputs(set_dir: Path, eng: Engine) -> tuple[list[dict], list[dict]]:
    qd = json.load(open(set_dir / "parity_questions.json"))
    labels = qd["questions"]
    for q in labels:
        q["_gold"] = eng.gold_ids(q["doc_name"], q["quote"])
    by = {q["id"]: q for q in labels}
    raw = json.load(open(set_dir / "agg15_trace_queries_r1_r2.json"))
    agg: dict[str, dict] = {}
    for r in raw:
        key = f"{r['qid']}|{r['query']}"
        a = agg.setdefault(key, {"key": key, "qid": r["qid"], "doc": by[r["qid"]]["doc_name"], "query": r["query"],
                                 "_gold": by[r["qid"]]["_gold"], "n_records": 0})
        a["n_records"] += 1
    return labels, list(agg.values())


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def done(out: Path, name: str) -> dict | None:
    p = out / f"phase_{name}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def save(out: Path, name: str, d: dict) -> None:
    tmp = out / f".phase_{name}.json.tmp"
    tmp.write_text(json.dumps(d, indent=1) + "\n")
    tmp.replace(out / f"phase_{name}.json")


def copy_store(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(str(dst) + ".tmp")
    try:
        s.backup(d)
        d.execute("PRAGMA journal_mode=DELETE")
        d.commit()
    finally:
        s.close()
        d.close()
    os.replace(str(dst) + ".tmp", dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orch-root", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True, help="the r3 parity store (read-only source; copied once)")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stopfile")
    ap.add_argument("--smoke", action="store_true", help="baseline phase only, then stop")
    ap.add_argument("--dry-run", action="store_true", help="inputs and engine import only; no index, no writes")
    a = ap.parse_args(argv)
    for p in (a.store, a.set_dir / "parity_questions.json", a.set_dir / "agg15_trace_queries_r1_r2.json",
              a.orch_root / "src/repl_environment/context_search.py"):
        if not p.exists():
            print(f"missing input {p}")
            return 2
    if a.dry_run:
        sys.path.insert(0, str(a.orch_root))
        from src.repl_environment import context_search as cs  # noqa: PLC0415,F401
        from src.aggregation.store import CorpusStore  # noqa: PLC0415,F401
        print(json.dumps({"dry_run": True, "store": str(a.store), "store_bytes": a.store.stat().st_size,
                          "orch_context_search_sha256": sha_file(a.orch_root / "src/repl_environment/context_search.py"),
                          "candidates": CAND_ORDER, "k": K, "expect_base": EXPECT_BASE, "expect_misses": EXPECT_MISSES,
                          "out": str(a.out)}, indent=1))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    if stop_seen(a.stopfile):
        print("stop-file")
        return 5
    copy = a.out / "store_copy.sqlite"
    copy_store(a.store, copy)
    eng = Engine(a.orch_root, copy)
    labels, traces = load_inputs(a.set_dir, eng)
    inputs = {"store_sha256_copy": sha_file(copy), "context_search_sha256": sha_file(a.orch_root / "src/repl_environment/context_search.py"),
              "questions_sha256": sha_file(a.set_dir / "parity_questions.json"),
              "traces_sha256": sha_file(a.set_dir / "agg15_trace_queries_r1_r2.json"),
              "n_labels": len(labels), "n_trace_unique": len(traces), "n_trace_records": sum(t["n_records"] for t in traces),
              "build_s": eng.build_s, "n_passages": {d: len(eng.idx[d].chunks) for d in DOCS}, "l3_table_rows": eng.l3_rows,
              "labels_without_gold_passage": [q["id"] for q in labels if not q["_gold"]]}

    base = done(a.out, "baseline")
    if base is None:
        ev = evaluate(eng, eng.base, labels, traces)
        base = {"builder": "l1_passage_bm25.v1 (context.search lexical)", "inputs": inputs, **strip(ev),
                "lab": ev["_lab"], "tr": ev["_tr"],
                "reproduces": ev["labels_hit"] == EXPECT_BASE and ev["label_misses"] == EXPECT_MISSES}
        save(a.out, "baseline", base)
    rec(a.out, "baseline", labels_hit=base["labels_hit"], labels_of=base["labels_of"], trace_hit=base["trace_hit"],
        trace_of=base["trace_of"], reproduces=base["reproduces"], status="ok" if base["reproduces"] else "error:baseline")
    if not base["reproduces"]:
        print(f"baseline does not reproduce 16/20: {base['labels_hit']} misses {base['label_misses']}")
        return 4
    if a.smoke:
        print("smoke ok")
        return 0

    if done(a.out, "diagnose") is None:
        save(a.out, "diagnose", {"misses": [diagnose(eng, q) for q in labels if q["id"] in EXPECT_MISSES]})
        rec(a.out, "diagnose", status="ok")

    answered = None
    for c in CAND_ORDER:
        r = done(a.out, c)
        if r is None:
            if stop_seen(a.stopfile):
                print("stop-file")
                return 5
            ev = evaluate(eng, lambda d, q, c=c: rank(eng, c, d, q), labels, traces)
            lost = sorted(k for k, v in base["tr"].items() if v and not ev["_tr"][k])
            lab_lost = sorted(k for k, v in base["lab"].items() if v and not ev["_lab"][k])
            moved = sorted(k for k in EXPECT_MISSES if ev["_lab"][k])
            r = {"candidate": c, "builder_version": f"l1_passage_bm25.v2-cand-{c}", **strip(ev),
                 "misses_fixed": moved, "labels_lost": lab_lost, "trace_lost": lost,
                 "answer": ev["labels_hit"] == len(labels) and not lost,
                 "cost": {"ms_per_query": ev["ms_per_query"], "extra_index_build": "none (re-ranks the v1 BM25 top-64)"}}
            save(a.out, c, r)
            rec(a.out, "candidate", unit=c, labels_hit=r["labels_hit"], trace_hit=r["trace_hit"], status="ok")
        if r["answer"]:
            answered = c
            break  # STOP-ON-ANSWER

    cands = [x for x in (done(a.out, c) for c in CAND_ORDER) if x]
    any_moved = any(x["misses_fixed"] for x in cands)
    verdict = "answer" if answered else ("partial_gain" if any_moved else "futility")
    result = {"schema": SCHEMA, "complete": True, "phase": "agg17", "verdict": verdict, "answered_by": answered,
              "recall_at_8": {"baseline": {"labels": f"{base['labels_hit']}/{base['labels_of']}",
                                           "trace_unique": f"{base['trace_hit']}/{base['trace_of']}",
                                           "trace_records": f"{base['trace_records_hit']}/{base['trace_records_of']}"},
                              **{x["candidate"]: {"labels": f"{x['labels_hit']}/{x['labels_of']}",
                                                  "trace_unique": f"{x['trace_hit']}/{x['trace_of']}",
                                                  "trace_records": f"{x['trace_records_hit']}/{x['trace_records_of']}",
                                                  "misses_fixed": x["misses_fixed"], "labels_lost": x["labels_lost"],
                                                  "trace_lost": x["trace_lost"], "ms_per_query": x["ms_per_query"]} for x in cands}},
              "not_run": [c for c in CAND_ORDER if not done(a.out, c)],
              "not_available": {"l3_row_pointers": f"tables_ rows in the r3 store = {eng.l3_rows} (the parity store carries no L3)",
                                "dense_l2": "no embedder: the box forbids a model or server (dense off is the measured mode)"},
              "baseline_ms_per_query": base["ms_per_query"], "inputs": inputs,
              "held_out": "candidates are research re-rankers labelled l1_passage_bm25.v2-cand-*; l1_passage_bm25.v1 and the AGG-1.5 parity claim are untouched"}
    (a.out / "result.agg17.json").write_text(json.dumps(result, indent=1) + "\n")
    rec(a.out, "verdict", verdict=verdict, status="ok")
    print(json.dumps({k: result[k] for k in ("verdict", "answered_by", "recall_at_8", "not_run")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
