#!/usr/bin/env python3
"""AGG-0.4 Phase-0 runner SKELETON (keeper runs it later under normal grants).

Not runnable end to end yet: the arm bodies are the keeper-wiring points and
raise ``NotImplementedError`` with their contract. What IS implemented:

* ``--preflight``: zero-compute checks — corpus PDFs present with pinned sha256,
  every question to run is verified in ``verification_record.json``, golds
  complete, PREDECLARED.md sha256 captured. Exit 2 on any failure.
* stage ordering with the predeclared stop-on-futility points (M5 manifest
  first, M1 before the GPU window is requested);
* append-only ``results.jsonl`` with a numeric record per stage, written early;
* final scoring via ``agg_phase0.score`` (M1-M5 + kill gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agg_phase0 as A  # noqa: E402

ARMS = ("P-large", "P-small", "B1", "B2", "B-today")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_set(d: Path) -> dict[str, Any]:
    golds = [json.loads(l) for l in (d / "golds.jsonl").read_text().splitlines() if l.strip()]
    single = [json.loads(l) for l in (d / "single_fact.jsonl").read_text().splitlines() if l.strip()]
    ver = json.loads((d / "verification_record.json").read_text())["questions"]
    return {"golds": golds, "single": single, "verified": ver,
            "corpus": json.loads((d / "corpus.json").read_text())}


def preflight(d: Path) -> tuple[bool, dict[str, Any]]:
    s = load_set(d)
    problems: list[str] = []
    for doc in s["corpus"]["docs"]:
        p = Path(doc["pdf"])
        if doc["sha256"] is None:
            problems.append(f"unpinned/missing pdf {doc['doc_name']}")
        elif not p.exists() or A.sha256_file(p) != doc["sha256"]:
            problems.append(f"pdf sha256 mismatch {doc['doc_name']}")
    run_agg = [g for g in s["golds"] if s["verified"].get(g["id"], {}).get("verified")]
    run_single = [q for q in s["single"] if s["verified"].get(q["id"], {}).get("verified")]
    for g in run_agg:
        if g["gold"] is None:
            problems.append(f"{g['id']} verified but gold incomplete")
    if not run_agg:
        problems.append("no verified aggregation questions")
    if not run_single:
        problems.append("no verified single-fact questions")
    info = {"predeclared_sha256": sha(d / "PREDECLARED.md"),
            "n_agg_verified": len(run_agg), "n_agg_excluded": len(s["golds"]) - len(run_agg),
            "n_single_verified": len(run_single),
            "n_single_excluded": len(s["single"]) - len(run_single), "problems": problems}
    return not problems, info


class Recorder:
    def __init__(self, path: Path, meta: dict[str, Any]) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.emit({"kind": "meta", **meta})

    def emit(self, rec: dict[str, Any]) -> None:
        rec = {"ts": time.time(), **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()


# ---------------------------------------------------------------- keeper wiring
def stage_ingest(rec: Recorder, corpus: dict[str, Any]) -> None:
    """Parse (PDFRouter/ODL) -> CorpusStore.ingest per doc -> L1 + L3 via IndexRegistry.

    Emit ``{"kind": "ingest", prefill_tokens, corpus_tokens, llm_calls, n_chunks}`` per doc
    and ``{"kind": "m1", gold_facts, found}`` after L4d. Needs ORCHESTRATOR_FEATURE_AGG_LAYER=1
    and a CPU-quarter grant for L4 (worker_general, -np 2, n_predict cap, cache_prompt+id_slot).
    """
    raise NotImplementedError("keeper wiring: AGG-0.4 ingest stage")


def stage_arm(rec: Recorder, arm: str, questions: list[dict[str, Any]]) -> None:
    """Emit one ``{"kind": "answer", arm, qid, qclass, correct, calls, prefill_tokens}`` per
    question; P-small also emits ``{"kind": "manifest", arm, qid, built_layers}``."""
    raise NotImplementedError(f"keeper wiring: AGG-0.4 arm {arm}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None, help="results dir (required to run)")
    ap.add_argument("--preflight", action="store_true", help="checks only, zero compute")
    a = ap.parse_args(argv)
    ok, info = preflight(a.set_dir)
    print(json.dumps(info, indent=1))
    if a.preflight or not ok:
        return 0 if ok else 2
    if a.out is None:
        ap.error("--out is required to run")
    s = load_set(a.set_dir)
    rec = Recorder(a.out / "results.jsonl", {"preflight": info})
    verified = lambda qid: s["verified"].get(qid, {}).get("verified")  # noqa: E731
    agg = [g for g in s["golds"] if verified(g["id"])]
    single = [q for q in s["single"] if verified(q["id"])]
    # Futility 1: M5 manifest (P-small) first — zero compute beyond the planner.
    stage_arm(rec, "P-small", single)
    stage_arm(rec, "B-today", single)
    # CPU quarter: ingest + L4; Futility 2: M1 before requesting the GPU window.
    stage_ingest(rec, s["corpus"])
    recs = [json.loads(l) for l in (a.out / "results.jsonl").read_text().splitlines()]
    partial = A.score([r for r in recs if r["kind"] != "meta"])
    if "M1" in partial["kill_gate_tripped"] or partial["M5"]["indexes_built"]:
        rec.emit({"kind": "stop", "reason": "futility", "partial": partial})
        return 3
    stage_arm(rec, "P-large", agg + single)
    stage_arm(rec, "B1", agg + single)
    stage_arm(rec, "B2", agg[:5])
    recs = [json.loads(l) for l in (a.out / "results.jsonl").read_text().splitlines()]
    final = A.score([r for r in recs if r["kind"] not in ("meta", "stop")])
    rec.emit({"kind": "verdict", **final})
    print(json.dumps(final, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
