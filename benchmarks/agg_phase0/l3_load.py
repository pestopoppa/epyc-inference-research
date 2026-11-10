#!/usr/bin/env python3
"""AGG-0.2/0.3 keeper job: parse the 10 FinanceBench 10-Ks once, load the corpus store,
build L1 + L3, and report the L3-only M1. CPU only, ZERO model calls.

Run it with the orchestrator venv python and the pinned orchestrator tree::

    python -I l3_load.py --out OUT --orch-root /path/to/pinned/orch [--smoke] [--only DOC]

Why no model is needed: parse = ``DocumentPreprocessor.preprocess`` with
``PDF_EXTRACTOR=opendataloader`` + ``ORCHESTRATOR_ODL_STRUCTURED=1`` (local JVM
OpenDataLoader, no OCR server, no figure description); chunking, the sqlite store,
BM25 (L1) and L3 (tables + regex) are deterministic Python. ``l4_facts_llm`` and
``l2_dense`` are never touched, and the job forces ``ORCHESTRATOR_ODL_TABLE_BACKEND=local``
(the ``hybrid`` backend would call a server).

Early, numeric, per-document records are appended (flush + fsync) to
``OUT/results.jsonl`` as each stage finishes:

  meta | probe | parse | store | l3_doc | doc_done | doc_fail | stop | corpus | m1_l3 | done

Resumable per document: a ``doc_done`` record whose ``pdf_sha256`` matches is skipped;
a parsed document is cached under ``OUT/cache`` (atomic write) so a run killed between
parse and store does not re-parse; the store ingest is idempotent per content hash.

Grant contract (filler_run15): ``FILLER_DEADLINE`` (epoch, start no new document after
it), ``FILLER_STOPFILE``, ``FILLER_SMOKE=1`` (or ``--smoke``: the smallest document only).
Exit codes: 0 complete (or smoke ok), 75 partial (deadline/stop; re-grant resumes),
2 pin/input failure, 3 a document failed to parse.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m1_l3_report as M1  # noqa: E402

CORPUS_ID = "agg_phase0_fb10"
SCHEMA = "agg_l3_load.v1"
JOB_ENV = {
    "PDF_EXTRACTOR": "opendataloader",
    "ORCHESTRATOR_ODL_STRUCTURED": "1",
    "ORCHESTRATOR_ODL_TABLE_BACKEND": "local",
    "ORCHESTRATOR_FEATURE_AGG_LAYER": "1",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class Recorder:
    """Append-only JSONL, one fsynced line per record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **rec: Any) -> dict[str, Any]:
        row = {"kind": kind, "ts": time.time(), **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:  # a torn last line from a kill: ignore, never repair
                continue
        return out


def real_parse(pdf: Path) -> Any:
    """The existing document flow, local ODL, no OCR server. Returns DocumentPreprocessResult."""
    from src.services.document_preprocessor import (  # noqa: PLC0415
        DocumentPreprocessor, PreprocessingConfig)

    pre = DocumentPreprocessor(PreprocessingConfig(extract_figures=False, describe_figures=False,
                                                   max_pages=5000))
    res = asyncio.run(pre.preprocess({"task_type": "doc",
                                      "inputs": [{"type": "path", "value": str(pdf)}]}))
    if not res.success or res.document_result is None:
        raise RuntimeError(res.error or "preprocess returned no document_result")
    return res.document_result


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def order_docs(corpus: dict[str, Any], only: str | None) -> list[dict[str, Any]]:
    docs = sorted(corpus["docs"], key=lambda d: (d["bytes"], d["doc_name"]))  # smallest first
    if only:
        docs = [d for d in docs if d["doc_name"] == only]
        if not docs:
            raise SystemExit(f"--only {only!r} not in corpus.json")
    return docs


def doc_l3_facts(store: Any, corpus_id: str, doc_id: str) -> tuple[list[dict[str, Any]], int, int]:
    from src.aggregation import facts_det  # noqa: PLC0415

    chunks = [c for c in store.chunks(corpus_id) if c["doc_id"] == doc_id]
    by_id = {c["chunk_id"]: c for c in chunks}
    ent = {d["doc_hash"]: d["entity"] for d in store.corpus_docs(corpus_id)}
    facts: list[dict[str, Any]] = []
    n_tab = 0
    for t in (t for t in store.tables(corpus_id) if t["doc_id"] == doc_id):
        got = facts_det.facts_from_table(t, (by_id.get(t["chunk_id"]) or {}).get("text", ""))
        n_tab += len(got)
        facts.extend(got)
    n_rx = 0
    for c in chunks:
        got = facts_det.facts_from_text(c, ent.get(c["doc_hash"]))
        n_rx += len(got)
        facts.extend(got)
    return facts, n_tab, n_rx


def run(args: argparse.Namespace, parse_fn: Callable[[Path], Any] = real_parse,
        now: Callable[[], float] = time.time) -> int:
    for k, v in JOB_ENV.items():
        os.environ[k] = v
    out: Path = args.out
    rec = Recorder(out / "results.jsonl")
    corpus = json.loads((args.set_dir / "corpus.json").read_text())
    docs = order_docs(corpus, args.only)
    smoke = args.smoke or os.environ.get("FILLER_SMOKE") == "1"
    if smoke:
        docs = docs[:1]
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    stopfile = os.environ.get("FILLER_STOPFILE")
    sys.path.insert(0, str(args.orch_root))
    from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    from src.models.document import DocumentPreprocessResult  # noqa: PLC0415

    done = {r["doc"]: r for r in rec.read() if r["kind"] == "doc_done"}
    sheet_sha = sha256_file(args.set_dir / "facts_sheet.csv")
    rec.emit("meta", schema=SCHEMA, smoke=smoke, orch_root=str(args.orch_root), orch_pin=args.orch_pin,
             research_pin=args.research_pin, corpus_json_sha256=sha256_file(args.set_dir / "corpus.json"),
             facts_sheet_sha256=sheet_sha, model_calls=0, env=JOB_ENV, n_docs_planned=len(docs),
             already_done=sorted(done))
    # probe: every pdf present and pinned BEFORE any parse (fail closed, numeric)
    bad = [d["doc_name"] for d in docs
           if not Path(d["pdf"]).exists() or sha256_file(Path(d["pdf"])) != d["sha256"]]
    rec.emit("probe", n_docs=len(docs), n_pdf_ok=len(docs) - len(bad), bad=bad,
             pdf_bytes=sum(d["bytes"] for d in docs))
    if bad:
        return 2
    store = CorpusStore(out / "store.sqlite")
    failed = 0
    partial = False
    for d in docs:
        name = d["doc_name"]
        if name in done and done[name].get("pdf_sha256") == d["sha256"]:
            rec.emit("skip", doc=name, reason="doc_done")
            continue
        if (stopfile and os.path.exists(stopfile)) or (deadline and now() >= deadline):
            rec.emit("stop", reason="stopfile" if stopfile and os.path.exists(stopfile) else "deadline",
                     next_doc=name)
            partial = True
            break
        t0 = now()
        cache = out / "cache" / f"{name}.{d['sha256'][:12]}.json"
        try:
            if cache.exists():
                blob = json.loads(cache.read_text())
                result = DocumentPreprocessResult.from_cache_dict(blob)
                parse_s, cached = blob.get("_parse_s"), True
            else:
                tp = now()
                result = parse_fn(Path(d["pdf"]))
                parse_s, cached = now() - tp, False
                payload = result.to_cache_dict()
                payload["_parse_s"] = parse_s
                _atomic_write(cache, json.dumps(payload, default=str))
            pages_seen = max((s.page_end for s in result.sections), default=0)
            rec.emit("parse", doc=name, pdf_bytes=d["bytes"], parse_s=parse_s, cached=cached,
                     total_pages=result.total_pages, pages_seen=pages_seen,
                     n_sections=len(result.sections), n_tables=len(result.tables),
                     n_chars=sum(len(s.content) for s in result.sections),
                     failed_pages=len(result.failed_pages),
                     maxrss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
            rc = store.ingest(result, corpus_id=CORPUS_ID, doc_id=name, entity=d["company"])
            rec.emit("store", doc=name, status=rc.status, doc_hash=rc.doc_hash, n_chunks=rc.n_chunks,
                     n_tables=rc.n_tables, n_tokens_est=rc.n_tokens_est)
            facts, n_tab, n_rx = doc_l3_facts(store, CORPUS_ID, name)
            store.write_facts(facts)
            units: dict[str, int] = {}
            for f in facts:
                units[f["unit"]] = units.get(f["unit"], 0) + 1
            rec.emit("l3_doc", doc=name, n_facts=len(facts), n_table_facts=n_tab, n_regex_facts=n_rx,
                     fields=len({f["field"] for f in facts}), units=units, llm_calls=0, prefill_tokens=0)
            rec.emit("doc_done", doc=name, pdf_sha256=d["sha256"], doc_hash=rc.doc_hash,
                     wall_s=now() - t0)
        except Exception as e:  # record and continue; the run ends rc 3
            failed += 1
            rec.emit("doc_fail", doc=name, error=f"{type(e).__name__}: {e}"[:500], wall_s=now() - t0)
    if partial:
        store.close()
        return 75
    if smoke:
        rec.emit("done", smoke=True, failed=failed)
        store.close()
        return 3 if failed else 0
    finished = {r["doc"] for r in rec.read() if r["kind"] == "doc_done"}
    if failed or finished != {d["doc_name"] for d in corpus["docs"]}:
        rec.emit("done", complete=False, failed=failed, missing=sorted(
            {d["doc_name"] for d in corpus["docs"]} - finished))
        store.close()
        return 3 if failed else 75
    reg = IndexRegistry(store)
    reg.get_or_build(CORPUS_ID, "l1_bm25")
    facts = reg.get_or_build(CORPUS_ID, "l3_facts_det")
    rec.emit("corpus", n_docs=len(store.corpus_docs(CORPUS_ID)), corpus_tokens=store.corpus_tokens(CORPUS_ID),
             n_chunks=len(store.chunks(CORPUS_ID)), n_tables=len(store.tables(CORPUS_ID)),
             n_facts=len(facts), built_layers=sorted(store.built_layers(CORPUS_ID)),
             corpus_content_hash=store.corpus_content_hash(CORPUS_ID), llm_calls=0, prefill_tokens=0)
    store.close()
    rep = M1.report(out / "store.sqlite", args.set_dir / "facts_sheet.csv")
    _atomic_write(out / "m1_l3_report.json", json.dumps(rep, indent=1, sort_keys=True) + "\n")
    rec.emit("m1_l3", **{k: rep[k] for k in ("gold_cells", "found", "m1_l3_only", "m1_status",
                                             "n_facts", "cells_with_any_l3_candidate", "sheet_cells",
                                             "candidate_coverage", "facts_sheet_sha256")})
    result = {"schema": SCHEMA, "complete": True, "n_docs": len(finished), "m1_l3_only": rep["m1_l3_only"],
              "m1_status": rep["m1_status"], "model_calls": 0}
    _atomic_write(out / "result.json", json.dumps(result, indent=1) + "\n")
    rec.emit("done", complete=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--orch-root", type=Path, required=True, help="pinned orchestrator tree (src/ on sys.path)")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--only", default=None, help="one doc_name (smoke / debugging)")
    ap.add_argument("--smoke", action="store_true", help="smallest document only")
    ap.add_argument("--orch-pin", default="", help="recorded in meta")
    ap.add_argument("--research-pin", default="", help="recorded in meta")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
