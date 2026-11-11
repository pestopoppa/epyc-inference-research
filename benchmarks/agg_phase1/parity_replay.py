#!/usr/bin/env python3
"""AGG-1.5 offline replay (zero model, no server): why r1 and r2 of the P arm looped, and the r3 fix.

Replays SEARCH query strings through three offline backends over the SAME document text and compares where the gold
quote ranks and what a hit shows the model:

* ``old_P``  the r1/r2 P arm: registry ``l1_bm25`` over the stored chunks (3-10k characters each, one BM25 document per
             chunk), a hit shown as ``title[:80] | text[:200]`` (the HEAD of the chunk).
* ``B``      ``ContextBundle.search`` (today's ``context.search``, lexical): 800-character line-aware passages, a hit
             shown as ``line N | span[:200]``.
* ``new_P``  the r3 P arm: registry ``l1_passage_bm25`` (``context.search``'s engine over the corpus text rebuilt from the
             store, built once), same presentation as B.

``gold`` is the question's verified quote (whitespace-folded) found in the hit's full text. ``visible`` is the question's
label found in the 200 characters the model is shown for the gold hit (what it needs to pick that hit).

CLI: ``parity_replay.py --store <parity store> --orch-root <orch> --out <json>`` replays the 20 label queries and the
SEARCH strings in ``agg15_trace_queries_r1_r2.json``. ``--extract-traces <json> --trace run=results.jsonl ...`` writes
that file from finished runs (it is committed, so the replay needs no /mnt/raid0/llm/tmp results).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
sys.path.insert(0, str(HERE))
import parity_m5 as M  # noqa: E402
import verify_phase1_cells as V  # noqa: E402

TOP = 50          # ranks are searched to here; None below it
SHOWN = 8         # the loop's SEARCH_K
_WORD = re.compile(r"[a-z0-9]+")


def r2_hit_window(text: str, query: str, width: int = 800) -> str:
    """The r2 P arm's 800-character read window (removed in r3; kept to measure what r2 reads contained)."""
    if len(text) <= width:
        return text
    words = [w for w in _WORD.findall(query.lower()) if len(w) > 1]
    if not words:
        return text[:width]
    uniq = sorted(set(words))
    low = re.sub(r"\s+", " ", text.lower())
    if len(low) != len(text):
        low = text.lower()
    phrase = " ".join(words)
    best, best_start = (-1, -1, -1), 0
    for c in sorted({m.start() for w in uniq for m in re.finditer(re.escape(w), low)}):
        start = min(max(c - width // 2, 0), len(text) - width)
        win = low[start:start + width]
        score = (int(phrase in win), sum(1 for w in uniq if w in win), sum(win.count(w) for w in uniq))
        if score > best:
            best, best_start = score, start
    return text[best_start:best_start + width] if best[1] > 0 else text[:width]


def extract_traces(paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Every SEARCH string of every finished answer record, in order: ``{run, arm, qid, status, calls, query}``."""
    out = []
    for run, p in paths.items():
        for line in Path(p).read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("kind") != "answer":
                continue
            for a in r["loop_actions"]:
                if a.startswith("SEARCH:"):
                    out.append({"run": run, "arm": r["arm"], "qid": r["qid"], "status": r["status"],
                                "calls": r["calls"], "query": a[len("SEARCH:"):].strip()})
    return out


def _fold(s: str) -> str:
    return " ".join(s.lower().split())


def _gold_figure(q: dict[str, Any]) -> str:
    v = abs(float(q["gold_value_usd_m"]))
    return f"{v:,.0f}" if v == int(v) else f"{v:,.1f}"


class Replay:
    """The three offline backends over one parity store. Needs the orchestrator tree on ``sys.path``."""

    def __init__(self, store, qdoc: dict[str, Any]) -> None:
        from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
        self.store, self.reg = store, IndexRegistry(store)
        self.q = {q["id"]: q for q in qdoc["questions"]}
        self.text = {d: "\n\n".join(c["text"] or "" for c in store.chunks(M.CORPUS_PREFIX + d)) for d in M.DOCS}
        self.chunks = {d: {c["chunk_id"]: c for c in store.chunks(M.CORPUS_PREFIX + d)} for d in M.DOCS}
        self.parm = {d: M.PArm(store, self.reg, d) for d in M.DOCS}
        for a in self.parm.values():
            a.open()
        self.barm = {d: M.BArm(d, self.text[d]) for d in M.DOCS}  # one bundle per doc: the replay needs no rebuild

    # each backend returns [(full_text_of_hit, shown_text)] best first
    def old_p(self, doc: str, query: str, k: int = TOP) -> list[tuple[str, str]]:
        h = self.reg.get_or_build(M.CORPUS_PREFIX + doc, M.R2_P_BUILDER)
        out = []
        for cid, _ in h.search(query, k):
            c = self.chunks[doc][cid]
            out.append((c["text"] or "", f"{(c.get('title') or '')[:80]} | {(c['text'] or '')[:200]}"))
        return out

    def b(self, doc: str, query: str, k: int = TOP) -> list[tuple[str, str]]:
        text = self.text[doc]
        hits = self.barm[doc].bundle.search(query, k)
        return [(text[h["start"]:h["end"]], M.snippet(h["line"], text[h["start"]:h["end"]])) for h in hits]

    def new_p(self, doc: str, query: str, k: int = TOP) -> list[tuple[str, str]]:
        h_ = self.parm[doc].handle
        return [(h_.span(h["start"], h["end"]), M.snippet(h["line"], h_.span(h["start"], h["end"])))
                for h in h_.search(query, k)]

    def pointers(self, backend: str, doc: str, query: str) -> list[str]:
        if backend == "new_P":
            return [f"{h['start']}-{h['end']}" for h in self.parm[doc].handle.search(query, TOP)]
        return [f"{h['start']}-{h['end']}" for h in self.barm[doc].bundle.search(query, TOP)]

    def measure(self, qid: str, query: str) -> dict[str, Any]:
        q = self.q[qid]
        quote, label, doc = V.sq(q["quote"]), _fold(q["label"]), q["doc_name"]
        row: dict[str, Any] = {"qid": qid, "query": query}
        for name, fn in (("old_P", self.old_p), ("B", self.b), ("new_P", self.new_p)):
            hits = fn(doc, query)
            rank = next((i + 1 for i, (full, _) in enumerate(hits) if quote in V.sq(full)), None)
            row[name] = {"gold_rank": rank, "in_top8": bool(rank and rank <= SHOWN),
                         "visible": bool(rank and label in _fold(hits[rank - 1][1]))}
            if rank:  # where the quote sits in the hit, against the 200 characters the model is shown
                full = V.sq(hits[rank - 1][0])
                row[name]["hit_chars"] = len(hits[rank - 1][0])
                row[name]["quote_offset"] = full.find(quote)
                row[name]["quote_in_shown"] = quote in V.sq(hits[rank - 1][0][:200])
            if name == "old_P" and rank:  # what the r2 800-character READ of that chunk returned
                row[name]["r2_read_has_figure"] = _gold_figure(q) in r2_hit_window(hits[rank - 1][0], query)
        row["new_P_equals_B"] = self.pointers("new_P", doc, query) == self.pointers("B", doc, query)
        return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {"n": len(rows)}
    for name in ("old_P", "B", "new_P"):
        d[name] = {"gold_in_top8": sum(x[name]["in_top8"] for x in rows),
                   "gold_hit_label_visible_in_shown_text": sum(x[name]["visible"] for x in rows),
                   "gold_in_top8_and_visible": sum(x[name]["in_top8"] and x[name]["visible"] for x in rows)}
    for name in ("old_P", "B", "new_P"):
        found = [x[name] for x in rows if x[name]["gold_rank"]]
        offs = sorted(x["quote_offset"] for x in found)
        d[name].update({"gold_top1": sum(x[name]["gold_rank"] == 1 for x in rows),
                        "gold_hit_median_chars": sorted(x["hit_chars"] for x in found)[len(found) // 2] if found else None,
                        "gold_quote_median_offset": offs[len(offs) // 2] if offs else None,
                        "gold_quote_inside_shown_200": sum(x["quote_in_shown"] for x in found), "gold_found": len(found)})
    d["old_P_r2_read_has_figure_when_gold_chunk_found"] = [
        sum(1 for x in rows if x["old_P"]["gold_rank"] and x["old_P"].get("r2_read_has_figure")),
        sum(1 for x in rows if x["old_P"]["gold_rank"])]
    d["new_P_ranking_identical_to_B"] = sum(x["new_P_equals_B"] for x in rows)
    return d


def run_replay(store, qdoc: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any]:
    rp = Replay(store, qdoc)
    label_rows = [rp.measure(q["id"], q["label"]) for q in qdoc["questions"]]
    trace_rows = [{**rp.measure(t["qid"], t["query"]), "run": t["run"], "arm": t["arm"], "status": t["status"]}
                  for t in traces]
    sizes = {d: {"chunks": len(rp.chunks[d]),
                 "mean_chars": round(sum(len(c["text"] or "") for c in rp.chunks[d].values()) / len(rp.chunks[d])),
                 "max_chars": max(len(c["text"] or "") for c in rp.chunks[d].values())} for d in M.DOCS}
    return {"schema": "agg15_replay.v1", "method": "offline, no model: three backends over the same corpus text",
            "stored_chunk_sizes": sizes, "passage_chars": 800,
            "label_queries": {"summary": summarize(label_rows), "rows": label_rows},
            "trace_queries": {"summary": summarize(trace_rows), "rows": trace_rows}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path)
    ap.add_argument("--orch-root", type=Path)
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--trace", action="append", default=[], metavar="RUN=results.jsonl")
    ap.add_argument("--extract-traces", type=Path, help="write the SEARCH strings of the --trace runs here and exit")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.extract_traces:
        paths = {t.split("=", 1)[0]: Path(t.split("=", 1)[1]) for t in a.trace}
        a.extract_traces.write_text(json.dumps(extract_traces(paths), indent=1) + "\n")
        return 0
    import os  # noqa: PLC0415
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    sys.path.insert(0, str(a.orch_root))
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    out = run_replay(CorpusStore(a.store), M.load_questions(a.set_dir),
                     json.loads((a.set_dir / "agg15_trace_queries_r1_r2.json").read_text()))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"stored_chunk_sizes": out["stored_chunk_sizes"], "label": out["label_queries"]["summary"],
                      "trace": out["trace_queries"]["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
