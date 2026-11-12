#!/usr/bin/env python3
"""AGG-1.12 frozen reference: the AGG-1.1 comparator and the held-out partition of the 211 gold cells (no model).

Reads, never writes: the AGG-1.1 answers store (the one P-large read, after the L4c audit and canonical.v1) and the AGG-1.1
GPU ``results.jsonl``; writes ``agg112_reference.json`` next to this file. ``PREDECLARED_AGG112.md`` pins its sha256, so the
comparator and the partition are fixed BEFORE any AGG-1.12 data exists.

Contents
* ``agg11``      per-cell raw hit (``phase1_arms.m1_hits`` over ``facts``) and canonical.v1 ok (the AGG-1.8 ``cell_state`` rule) of
                 every one of the 211 cells; M1, M3, per-document hits, the pilot-document cells; per-question P-large correctness,
                 calls and prefill of the 58 questions, accuracy and M4 medians.
* ``partition``  which of the 211 cells were TOUCHED while the AGG-1.8 / AGG-1.11 / AGG-1.6 / AGG-1.9 / AGG-1.12 designs were made
                 (named in a miss table, a triage row, a target list, a probe list), and which were not. Provenance is by script
                 from the committed files, per cell, with the reason(s); nothing here is a judgement call.
* ``questions``  how many of the 58 questions read ONLY untouched cells (none: every question reads cells that were touched).

CLI:  agg112_reference.py --orch-root DIR [--store PATH] [--results PATH] [--out JSON]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "agg112_reference.v1"
OUT_FILE = "agg112_reference.json"
DEFAULT_STORE = Path("/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out/store.sqlite")
DEFAULT_RESULTS = Path("/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out/results.jsonl")
DEFAULT_CPU_RESULT = Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/result.cpu.json")
DEFAULT_PILOT_RESULT = Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/result.pilot.json")
STORE_SHA = "ba5cf95076f61ec0373699021970c1d8102da93b0d887b2b2958e72cd2e7b063"
RESULTS_SHA = "c921910e634f05e1b3403926ba1fe9679285603994727d63b0608d7aab57c95a"
CORPUS_ID = "agg_phase0_fb10"
PILOT_DOCS = ("MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K", "3M_2022_10K")
EXPECTED = {"cells": 211, "raw_hits": 185, "canonical_ok_v1": 177, "p_large_correct": 18, "n_questions": 58}

# the files the partition reads (all committed under benchmarks/agg_phase1/)
PARTITION_SOURCES = ("agg111_triage.json", "agg18_miss_table.json", "agg19_cells.csv", "agg19_guard_probe.json")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def key(d: str, f: str) -> str:
    return f"{d}|{f}"


# --------------------------------------------------------------------------- the partition (pure, from committed files)
def touched_reasons(set_dir: Path = HERE) -> dict[str, list[str]]:
    """``cell key -> reasons`` for every (document, field) named in a design artifact. A cell is TOUCHED when it appears in any of:

    * ``agg111_triage``      one of the 26 AGG-1.9 cells the triage classified (group c; groups a / b are the AGG-1.3 FY2021 slots and
                             the AGG-1.4 long_term_debt cells: other filings or other fields, so they are NOT 211 cells even when
                             a document name matches);
    * ``agg18_bad_cell``     the AGG-1.8 miss table's bad cells (26 never extracted correctly, 8 canonicalisation misses, 1 spurious);
    * ``agg18_tier_probe``   named in the AGG-1.8 blanket l4-over-l3 tier probe (7 fixed, 19 broken), the evidence the guards were
                             narrowed against;
    * ``agg19_cell``         the AGG-1.9 fill list (a copy of the 26).

    Cells only SCORED IN AGGREGATE by the retrieval probe (gold window held in 206/211), the l3_period variant table and the
    canonical guard probe are NOT touched by this definition: those counted all 211 cells without naming or reading any of them.
    They are weakly in-sample for the component choices, and the held-out report says so."""
    import csv  # noqa: PLC0415
    why: dict[str, list[str]] = {}

    def add(k: str, r: str) -> None:
        why.setdefault(k, [])
        if r not in why[k]:
            why[k].append(r)

    tri = json.loads((set_dir / "agg111_triage.json").read_text())
    for c in tri["cells"]:
        if c["group"] == "c":
            add(key(c["doc"], c["field"]), "agg111_triage")
    miss = json.loads((set_dir / "agg18_miss_table.json").read_text())
    for c in miss["bad_cells"]:
        add(key(c["doc"], c["field"]), "agg18_bad_cell")
    for name in ("fixed", "broken"):
        for k in miss["tier_probe"][name]:
            d, f = k.split("|")[:2]
            add(key(d, f), "agg18_tier_probe")
    with (set_dir / "agg19_cells.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            add(key(r["doc_name"], r["metric"]), "agg19_cell")
    return why


def partition(cells: list[tuple[str, str, float]], why: dict[str, list[str]]) -> dict[str, Any]:
    keys = [key(d, f) for d, f, _g in cells]
    touched = [k for k in keys if k in why]
    held = [k for k in keys if k not in why]
    return {"n_cells": len(keys), "n_touched": len(touched), "n_held_out": len(held), "touched": touched, "held_out": held,
            "reasons": {k: why[k] for k in touched},
            "touched_named_outside_211": sorted(k for k in why if k not in set(keys))}


def questions_clean(golds: list[dict[str, Any]], verified: dict[str, Any], touched: set[str]) -> dict[str, Any]:
    """Verified questions whose evidence cells are all untouched."""
    clean, n = [], 0
    per: dict[str, int] = {}
    for g in golds:
        if not verified.get(g["id"], {}).get("verified"):
            continue
        n += 1
        ev = {key(e["doc"], e["metric"]) for e in g["evidence"]}
        per[g["id"]] = len(ev & touched)
        if not ev & touched:
            clean.append(g["id"])
    return {"n_verified": n, "n_with_only_untouched_evidence": len(clean), "ids": clean,
            "touched_evidence_cells_per_question_min": min(per.values()) if per else None,
            "touched_evidence_cells_per_question_max": max(per.values()) if per else None}


# --------------------------------------------------------------------------- the AGG-1.1 comparator
def per_doc(cells: list[tuple[str, str, float]], hits: dict[str, bool]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for d, f, _g in cells:
        b = out.setdefault(d, {"gold": 0, "found": 0})
        b["gold"] += 1
        b["found"] += int(bool(hits[key(d, f)]))
    return dict(sorted(out.items()))


def p_large_table(records: list[dict[str, Any]], qids: list[str]) -> dict[str, Any]:
    ans = {r["qid"]: r for r in records if r.get("kind") == "answer" and r.get("arm") == "P-large" and r.get("final")}
    per = {q: {"correct": ans[q].get("correct") is True, "calls": ans[q]["calls"], "prefill_tokens": ans[q]["prefill_tokens"],
               "path": ans[q].get("path")} for q in qids if q in ans}
    import statistics  # noqa: PLC0415
    n = len(per)
    return {"n": n, "n_correct": sum(v["correct"] for v in per.values()),
            "accuracy": sum(v["correct"] for v in per.values()) / n if n else None,
            "median_calls": statistics.median(v["calls"] for v in per.values()) if per else None,
            "median_prefill": statistics.median(v["prefill_tokens"] for v in per.values()) if per else None,
            "per_question": per}


def build(orch_root: Path, store_path: Path, results_path: Path, set_dir: Path = HERE,
          cpu_result: Path = DEFAULT_CPU_RESULT, pilot_result: Path = DEFAULT_PILOT_RESULT) -> dict[str, Any]:
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    if str(orch_root) not in sys.path:
        sys.path.insert(0, str(orch_root))
    import agg19 as G19  # noqa: PLC0415
    import phase1_arms as X  # noqa: PLC0415
    from src.aggregation.store import select_canonical  # noqa: PLC0415
    inp = G19.load_inputs(set_dir)
    cells = inp["cells211"]
    store_sha, results_sha = sha256_file(store_path), sha256_file(results_path)
    conn = sqlite3.connect(f"file:{store_path}?immutable=1", uri=True)
    try:
        rows = G19.fact_rows(conn, CORPUS_ID)
    finally:
        conn.close()
    raw = G19.raw_hits(rows, cells)
    can = G19.canon_ok(select_canonical(rows)[0], cells)
    why = touched_reasons(set_dir)
    part = partition(cells, why)
    touched = set(part["touched"])
    base = X.load_inputs(corpus=set_dir / "corpus.json", golds=set_dir / "golds.jsonl",
                         verification=set_dir / "verification_record.json", facts_sheet=set_dir / "facts_sheet.csv")
    golds = base["golds"]
    qids = [g["id"] for g in X.verified_agg(base)]
    recs = X.read_jsonl(results_path)
    pl = p_large_table(recs, qids)
    cpu = json.loads(cpu_result.read_text())
    pilot = json.loads(pilot_result.read_text())
    pilot_keys = [key(d, f) for d, f, _g in cells if d in PILOT_DOCS]
    held = set(part["held_out"])
    out: dict[str, Any] = {
        "schema": SCHEMA, "model_calls": 0,
        "sources": {"store": str(store_path), "store_sha256": store_sha, "results": str(results_path),
                    "results_sha256": results_sha, "partition_files": {n: sha256_file(set_dir / n) for n in PARTITION_SOURCES}},
        "agg11": {
            "n_cells": len(cells), "raw_found": sum(raw.values()), "raw_m1": sum(raw.values()) / len(cells),
            "canonical_v1_ok": sum(can.values()), "raw_hit_cells": sorted(k for k, v in raw.items() if v),
            "canonical_v1_ok_cells": sorted(k for k, v in can.items() if v),
            "per_doc_raw": per_doc(cells, raw),
            "pilot": {"docs": list(PILOT_DOCS), "n_cells": len(pilot_keys), "found": sum(raw[k] for k in pilot_keys),
                      "m1": sum(raw[k] for k in pilot_keys) / len(pilot_keys), "cells": pilot_keys,
                      "recorded_pilot_m1_in_agg11": pilot["pilot_m1"]},
            "m3": cpu["m3"], "m3_incremental": cpu.get("m3_incremental"),
            "p_large": pl},
        "partition": part,
        "partition_hit_rates_agg11": {
            "held_out": {"n": len(held), "raw_found": sum(raw[k] for k in held), "canonical_v1_ok": sum(can[k] for k in held)},
            "touched": {"n": len(touched), "raw_found": sum(raw[k] for k in touched),
                        "canonical_v1_ok": sum(can[k] for k in touched)}},
        "questions": questions_clean(golds, base["verified"], touched),
    }
    got = {"cells": len(cells), "raw_hits": out["agg11"]["raw_found"], "canonical_ok_v1": out["agg11"]["canonical_v1_ok"],
           "p_large_correct": pl["n_correct"], "n_questions": pl["n"]}
    out["expected_matches"] = got == EXPECTED
    out["expected"], out["got"] = EXPECTED, got
    out["store_sha256_ok"] = store_sha == STORE_SHA and results_sha == RESULTS_SHA
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orch-root", type=Path, required=True)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=HERE / OUT_FILE)
    a = ap.parse_args(argv)
    res = build(a.orch_root, a.store, a.results, a.set_dir)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}: expected_matches={res['expected_matches']} store_sha256_ok={res['store_sha256_ok']}")
    return 0 if res["expected_matches"] and res["store_sha256_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
