#!/usr/bin/env python3
"""AGG-1.5 persistence re-check (ZERO model, no server, no GPU): does the P arm's passage index survive a process boundary?

r3 (agg15-parity-gpu-20261111-r3) failed ONE component, ``no_rebuild_per_request``: P built 5 indexes (JPMORGAN 2, the
others 1) because the smoke process and the resumed main run were two processes and the registry builder
``l1_passage_bm25`` cached in memory only. The orchestrator now persists that index next to the store
(``src/aggregation/registry.py``, keyed by corpus, content hash and builder version). This script replays the 20 AGG-1.5
label queries through the P arm's retrieval in TWO SEPARATE PROCESSES against a COPY of the r3 parity store and asserts:

  * process 1 builds one index per document (4) and persists it; process 2 builds 0 and LOADS all 4;
  * the two processes return IDENTICAL hits (chunk, span, line, bit-identical score, in order) for all 20 queries;
  * at most 1 build per document across both processes together (the r3 gate's own test);
  * the manifest of the copy records ``origin='loaded'`` with n_built == 1 and n_loaded == 1 per document;
  * the r3 source store is byte-identical before and after (it is only read).

It also records, informationally, where the gold quote ranks for P and for B (``context.search``) on the label queries:
the B recall weakness (gold in the top 8 on only 16 of 20) is a retrieval property of the shared engine, not a
persistence property, and is tracked separately (AGG-1.7).

    python3 parity_persist_check.py --orch-root <orch tree at/after the persistence commit> \\
        --store <r3 parity_store.sqlite> --work-dir <empty scratch dir> --out <result json>

Exit 0 when every assertion holds, 1 otherwise. The check is the re-run of the ``no_rebuild_per_request`` component only;
it does not (and cannot) replace a GPU run of accuracy / calls / prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
sys.path.insert(0, str(HERE))
import parity_m5 as M  # noqa: E402
import verify_phase1_cells as V  # noqa: E402

SCHEMA = "agg15_parity_persist_check.v1"
BUILDER = M.P_BUILDER
R3_STORE = Path("/mnt/raid0/llm/tmp/agg15-parity-gpu-20261111-r3/out/parity_store.sqlite")


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copy_store(src: Path, dst: Path) -> dict[str, Any]:
    """A consistent copy of the r3 parity store (sqlite backup API, source opened read-only), with the manifest rows of
    the passage builder removed so the copy's counters start from nothing. Returns what was removed."""
    if dst.exists():
        raise FileExistsError(f"{dst} exists: the check needs a fresh work dir")
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
        d.execute("PRAGMA journal_mode=DELETE")
        rows = d.execute("SELECT COUNT(*) FROM index_manifest WHERE builder = ?", [BUILDER]).fetchone()[0]
        d.execute("DELETE FROM index_manifest WHERE builder = ?", [BUILDER])
        d.commit()
    finally:
        s.close()
        d.close()
    return {"manifest_rows_removed_from_copy": rows}


# --------------------------------------------------------------------------------------- the child (one process)
def child(orch_root: Path, store_path: Path, set_dir: Path, tag: str) -> dict[str, Any]:
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    sys.path.insert(0, str(orch_root))
    from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
    from src.aggregation.store import CorpusStore  # noqa: PLC0415

    qdoc = M.load_questions(set_dir)
    store = CorpusStore(store_path)
    reg = IndexRegistry(store)
    text = {d: "\n\n".join(c["text"] or "" for c in store.chunks(M.CORPUS_PREFIX + d)) for d in M.DOCS}
    handles: dict[str, Any] = {}
    rows = []
    for q in qdoc["questions"]:
        doc = q["doc_name"]
        if doc not in handles:  # exactly what PArm.open does: one get_or_build per document
            handles[doc] = reg.get_or_build(M.CORPUS_PREFIX + doc, BUILDER)
        h = handles[doc]
        hits = h.search(q["label"], M.SEARCH_K)
        gold = V.sq(q["quote"])
        b_hits = M.BArm(doc, text[doc]).bundle.search(q["label"], M.SEARCH_K)
        rows.append({
            "qid": q["id"], "doc": doc, "label": q["label"],
            "p_hits": [[x["chunk_id"], x["start"], x["end"], x["line"], x["score"]] for x in hits],
            "p_gold_in_top8": any(gold in V.sq(h.span(x["start"], x["end"])) for x in hits),
            "b_hits": [[x["start"], x["end"], x["line"], x["score"]] for x in b_hits],
            "b_gold_in_top8": any(gold in V.sq(text[doc][x["start"]:x["end"]]) for x in b_hits),
            "p_equals_b": [[x["start"], x["end"], x["line"]] for x in hits]
            == [[x["start"], x["end"], x["line"]] for x in b_hits]})
    per_doc: dict[str, dict[str, Any]] = {}
    for e in reg.events:
        doc = e["corpus_id"][len(M.CORPUS_PREFIX):]
        per_doc[doc] = {"origin": e["origin"], "seconds": e["seconds"], "n_items": e["n_items"]}
    out = {"tag": tag, "pid": os.getpid(), "built": reg.built, "loaded": reg.loaded, "per_doc": per_doc,
           "invalid_files": reg.invalid, "persist_errors": reg.persist_errors, "rows": rows,
           "manifest": [m for m in store.manifest() if m["builder"] == BUILDER]}
    store.close()
    return out


# --------------------------------------------------------------------------------------- the parent
def run_child(orch_root: Path, store: Path, set_dir: Path, tag: str, out: Path) -> dict[str, Any]:
    env = {**os.environ, "ORCHESTRATOR_FEATURE_AGG_LAYER": "1"}
    cp = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", tag, "--orch-root", str(orch_root),
                         "--store", str(store), "--set-dir", str(set_dir), "--out", str(out)],
                        capture_output=True, text=True, env=env, timeout=1800)
    if cp.returncode != 0:
        raise RuntimeError(f"child {tag} failed ({cp.returncode}): {cp.stderr[-1500:]}")
    return json.loads(out.read_text())


def git_head(root: Path) -> dict[str, Any]:
    def g(*a: str) -> str:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "tree_clean": g("status", "--porcelain", "--untracked-files=no") == ""}


def assess(p1: dict[str, Any], p2: dict[str, Any]) -> dict[str, Any]:
    docs = sorted(set(p1["per_doc"]) | set(p2["per_doc"]))
    builds_total = {d: sum(int(p["per_doc"].get(d, {}).get("origin") == "built") for p in (p1, p2)) for d in docs}
    manifest_ok = len(p2["manifest"]) == len(docs) and all(
        m["origin"] == "loaded" and m["n_built"] == 1 and m["n_loaded"] == 1 for m in p2["manifest"])
    hits_identical = [a["p_hits"] for a in p1["rows"]] == [b["p_hits"] for b in p2["rows"]]
    checks = {
        "p1_built_one_per_doc": p1["built"] == len(docs) and p1["loaded"] == 0
        and all(v["origin"] == "built" for v in p1["per_doc"].values()),
        "p2_built_zero": p2["built"] == 0,
        "p2_loaded_every_doc": p2["loaded"] == len(docs) and all(v["origin"] == "loaded" for v in p2["per_doc"].values()),
        "hits_identical_across_processes": hits_identical and len(p1["rows"]) == len(p2["rows"]) == 20,
        "at_most_one_build_per_doc_total": all(v <= 1 for v in builds_total.values())
        and sum(builds_total.values()) <= len(docs),
        "manifest_records_loaded_vs_built": manifest_ok,
        "no_invalid_files_no_persist_errors": not (p1["invalid_files"] or p1["persist_errors"]
                                                   or p2["invalid_files"] or p2["persist_errors"]),
        "distinct_processes": p1["pid"] != p2["pid"],
    }
    return {"checks": checks, "builds_total_per_doc": builds_total, "p_equals_b_queries": sum(
        r["p_equals_b"] for r in p1["rows"]), "passed": all(checks.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orch-root", type=Path, required=True)
    ap.add_argument("--store", type=Path, default=R3_STORE, help="the r3 parity store (read-only source)")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--work-dir", type=Path, help="an empty scratch directory (the copy and its index dir live here)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:  # one process of the pair: write its result and exit
        res = child(a.orch_root, a.store, a.set_dir, a.child)
        a.out.write_text(json.dumps(res))
        return 0
    if not a.work_dir or not a.out:
        ap.error("--work-dir and --out are required")
    a.work_dir.mkdir(parents=True, exist_ok=True)
    if any(a.work_dir.iterdir()):
        ap.error(f"--work-dir {a.work_dir} is not empty")
    src_sha_before = sha_file(a.store)
    copy = a.work_dir / "parity_store.sqlite"
    note = copy_store(a.store, copy)
    t0 = time.time()
    p1 = run_child(a.orch_root, copy, a.set_dir, "process_1", a.work_dir / "p1.json")
    p2 = run_child(a.orch_root, copy, a.set_dir, "process_2", a.work_dir / "p2.json")
    wall_s = round(time.time() - t0, 2)
    src_sha_after = sha_file(a.store)
    verdict = assess(p1, p2)
    verdict["checks"]["r3_source_store_untouched"] = src_sha_before == src_sha_after
    verdict["passed"] = all(verdict["checks"].values())
    idx_dir = copy.with_name(copy.name + ".indexes")
    qdoc = M.load_questions(a.set_dir)
    p_gold, b_gold = (sum(r[k] for r in p1["rows"]) for k in ("p_gold_in_top8", "b_gold_in_top8"))
    miss = [r["qid"] for r in p1["rows"] if not r["b_gold_in_top8"]]
    result = {
        "schema": SCHEMA, "question": "does l1_passage_bm25 survive a process boundary with 0 rebuilds and identical hits",
        "method": "zero model; replay the 20 label queries through the P arm in two separate processes over a copy of the r3 store",
        "passed": verdict["passed"], "checks": verdict["checks"],
        "builds": {"process_1": p1["built"], "process_2": p2["built"], "loaded_process_2": p2["loaded"],
                   "per_doc_total_across_processes": verdict["builds_total_per_doc"]},
        "inputs": {"r3_store": str(a.store), "r3_store_sha256": src_sha_before, **note,
                   "orch": git_head(a.orch_root),
                   "registry_py_sha256": sha_file(a.orch_root / "src/aggregation/registry.py"),
                   "questions_sha256": sha_file(a.set_dir / "parity_questions.json"),
                   "n_questions": len(qdoc["questions"]), "builder": BUILDER},
        "index_file_bytes": {f.name: f.stat().st_size for f in sorted(idx_dir.glob("*.json.gz"))},
        "processes": {t: {k: p[k] for k in ("pid", "built", "loaded", "per_doc", "invalid_files", "persist_errors", "manifest")}
                      for t, p in (("process_1", p1), ("process_2", p2))},
        "hits_digest_sha256": hashlib.sha256(json.dumps([r["p_hits"] for r in p1["rows"]]).encode()).hexdigest(),
        "retrieval_informational": {
            "note": "gold = the question's verified quote (whitespace/$-folded) inside a top-8 hit's text, label query",
            "P_gold_in_top8": p_gold, "B_gold_in_top8": b_gold, "of": len(p1["rows"]),
            "P_ranking_identical_to_B_queries": verdict["p_equals_b_queries"],
            "B_gold_missing_qids": miss},
        "wall_s_two_processes": wall_s,
        "rows": [{k: r[k] for k in ("qid", "doc", "label", "p_gold_in_top8", "b_gold_in_top8", "p_equals_b")}
                 | {"p_top8": r["p_hits"]} for r in p1["rows"]],
    }
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": result["checks"], "builds": result["builds"],
                      "retrieval_informational": result["retrieval_informational"]}, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
