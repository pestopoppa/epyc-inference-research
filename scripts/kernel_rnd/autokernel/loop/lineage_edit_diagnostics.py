#!/usr/bin/env python3
"""AK-lineage diagnostics over recorded proposals: proposal-level cycling detector + 9-label edit taxonomy.

Read-only (`mode=ro`, scalar columns only), stdlib only, no model call. Observe-only, nonpromotable.

Scope and honesty: the line-level EvoReplay detector lives in the root repo
(`scripts/autokernel/lineage_diagnostics.py`) and REFUSES records without an authoring-bound source capture, so it
cannot run on historical stores. This module is the *proposal-statement-level* counterpart that historical
`experiments.db` rows can support:
  cycling  a proposal is "recycled" when its (target_symbol, normalized-statement shingle set) matches an earlier
           proposal of the same store at Jaccard >= `threshold` (default 0.6) AND at least one proposal with a
           different key lies between them (an A..B..A return, not an immediate retry). `first_seen` rows are
           "added". cycling_rate = recycled / proposals.
  taxonomy the 9 EvoReplay edit labels, assigned by ordered keyword rules over the statement text. This is a
           deterministic PROXY label, not an authoring-bound classification; `unlabeled` is reported, never forced.
Yield vs frequency per label: n, share of proposals, measured n, kept n, keep rate, mean/max effect, and
`yield_over_frequency = keep_share / proposal_share` (>1 = over-yields its frequency).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Sequence

SCHEMA = "epyc.autokernel.lineage_edit_diagnostics.v1"
LABELS = ("bug_fix", "external_dependency", "architectural_change", "composition", "local_refinement",
          "pruning", "refactor", "efficiency", "hyperparameter_tuning")
MEASURED = frozenset({"kept", "keep_candidate", "measured_null", "regression", "runtime_observed"})
#: ordered: first matching rule wins. Specific signals before generic ones.
_RULES = (
    ("bug_fix", r"\b(fix|bug|correct(ness)?|race|overflow|off-by-one|wrong|incorrect|crash|nan)\b"),
    ("external_dependency", r"\b(library|rocblas|hipblas|hipblaslt|aiter|composable|cutlass|intrinsic|vendor|upstream)\b"),
    ("hyperparameter_tuning", r"\b(tune|tuning|threshold|block size|tile size|unroll factor|occupancy|launch bounds|"
                              r"num_threads|n_threads|chunk size|prefetch distance|constant)\b"),
    ("pruning", r"\b(remove|delete|drop|eliminate|skip|elide|dead|redundant|unnecessary)\b"),
    ("composition", r"\b(fuse|fused|fusion|combine|merge|epilogue|prologue|two-kernel|single kernel)\b"),
    ("architectural_change", r"\b(redesign|restructure|new (kernel|path|algorithm|layout)|rewrite|repack|re-layout|"
                             r"layout|partition|split-k|stream-k|workgroup|wave64|cohort|specializ\w*)\b"),
    ("refactor", r"\b(refactor|rename|hoist|inline|simplif\w*|clean ?up|deduplicate)\b"),
    ("efficiency", r"\b(vectori[sz]e\w*|coalesc\w*|branchless|prefetch|cache|lds|register|latency|bandwidth|"
                   r"throughput|faster|reduce|float4|packed|unconditional|straight-line)\b"),
    ("local_refinement", r"\b(replace|change|adjust|modify|improve|use|add|collapse|preserve)\b"),
)
_COMPILED = tuple((lab, re.compile(rx, re.I)) for lab, rx in _RULES)
_TOK = re.compile(r"[a-z_][a-z0-9_]{2,}")
_STOP = frozenset("the and for with that this from into each when are its has have not but can one all per via".split())


def classify(statement: str | None) -> str:
    for lab, rx in _COMPILED:
        if statement and rx.search(statement):
            return lab
    return "unlabeled"


def shingles(statement: str | None) -> frozenset:
    toks = [t for t in _TOK.findall((statement or "").lower()) if t not in _STOP]
    return frozenset(zip(toks, toks[1:])) or frozenset((t,) for t in toks)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def load(db: Path) -> list[dict]:
    con = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT attempt_id,recorded_at,campaign_id,target_symbol,statement,status,effect_fraction "
                           "FROM experiments WHERE statement IS NOT NULL AND statement != '' "
                           "ORDER BY recorded_at,rowid").fetchall()
    finally:
        con.close()
    return [{"id": a, "campaign": c, "symbol": s or "?", "statement": st, "status": status, "effect": e}
            for a, _t, c, s, st, status, e in rows]


def detect_cycling(rows: Sequence[dict], threshold: float = 0.6) -> dict:
    """Mark each proposal added / recycled. Compare only against earlier rows with the same target_symbol."""
    sigs = [shingles(r["statement"]) for r in rows]
    by_sym: dict[str, list[int]] = {}
    flags, matches = [], []
    for i, r in enumerate(rows):
        hit = None
        for j in reversed(by_sym.get(r["symbol"], [])):
            if jaccard(sigs[i], sigs[j]) >= threshold:
                # an immediate retry (no different-key proposal between) is not a cycle return
                if any(rows[k]["symbol"] != r["symbol"] for k in range(j + 1, i)):
                    hit = j
                    break
        flags.append(hit is not None)
        matches.append(hit)
        by_sym.setdefault(r["symbol"], []).append(i)
    n = len(rows)
    rec = sum(flags)
    return {"proposals": n, "recycled": rec, "added": n - rec, "cycling_rate": rec / n if n else None,
            "flags": flags, "matches": matches}


def taxonomy_yield(rows: Sequence[dict], flags: Sequence[bool] | None = None) -> dict:
    n = len(rows)
    kept_total = sum(1 for r in rows if r["status"] == "kept")
    out = {}
    for lab in (*LABELS, "unlabeled"):
        sel = [(i, r) for i, r in enumerate(rows) if classify(r["statement"]) == lab]
        meas = [r for _, r in sel if r["status"] in MEASURED]
        kept = sum(1 for _, r in sel if r["status"] == "kept")
        effs = [r["effect"] for r in meas if isinstance(r["effect"], (int, float)) and r["effect"] == r["effect"]]
        share = len(sel) / n if n else 0.0
        ks = kept / kept_total if kept_total else None
        out[lab] = {"n": len(sel), "share": share, "measured": len(meas), "kept": kept,
                    "keep_rate_of_measured": kept / len(meas) if meas else None,
                    "mean_effect": sum(effs) / len(effs) if effs else None, "max_effect": max(effs) if effs else None,
                    "yield_over_frequency": (ks / share) if (ks is not None and share) else None,
                    "recycled": sum(1 for i, _ in sel if flags and flags[i])}
    return out


def analyze(db: Path, threshold: float = 0.6) -> dict:
    rows = load(db)
    cyc = detect_cycling(rows, threshold)
    flags = cyc.pop("flags")
    cyc.pop("matches")
    kept_r = sum(1 for f, r in zip(flags, rows) if f and r["status"] == "kept")
    kept_a = sum(1 for f, r in zip(flags, rows) if not f and r["status"] == "kept")
    meas_r = sum(1 for f, r in zip(flags, rows) if f and r["status"] in MEASURED)
    meas_a = sum(1 for f, r in zip(flags, rows) if not f and r["status"] in MEASURED)
    return {"path": str(db), "threshold": threshold, "cycling": cyc,
            "keep_rate_recycled": kept_r / meas_r if meas_r else None,
            "keep_rate_added": kept_a / meas_a if meas_a else None,
            "taxonomy": taxonomy_yield(rows, flags)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stores", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--pretty", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps({"schema": SCHEMA, "observe_only": True, "nonpromotable": True,
                      "labels_are_text_proxy": True, "line_level_detector": "root scripts/autokernel/lineage_diagnostics.py "
                      "(refuses historical unbound records)",
                      "stores": [analyze(s, a.threshold) for s in a.stores]}, sort_keys=True,
                     indent=2 if a.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
