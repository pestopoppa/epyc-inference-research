#!/usr/bin/env python3
"""AGG-1.1 step B: Phase 1 run phases (library; ``run_phase1.py`` is the CLI).

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an
injected backend (``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in tests).

This file implements ``benchmarks/agg_phase1/PREDECLARED.md`` literally and changes NONE of its frozen parts:
questions, golds, the deterministic scorer, the metric definitions and the gate thresholds are the Phase 0 ones
(``agg_phase0`` / ``phase0_arms``, imported, never copied or edited; ``PHASE0_PINS`` and ``FROZEN_P1`` pin their
sha256 and the tests assert the pins).

Phases (``run_phase1.py --phase``), checkpoints numbered as in PREDECLARED "Checkpoints and stopping rules":

* ``preflight``   0. zero compute: pins, PDFs, 58 of 58 verified, ``build_phase1.py check``, L4 reuse test.
* ``l3``          1. L3 load of the 18 new documents into a copy of the Phase 0 store (no model); futility
                  ``>= 3`` of 18 fail to parse -> exit 6.
* ``pilot``       2. L4 ingest of MICROSOFT, JOHNSON_JOHNSON, 3M (24 units, <= 1,800 s) and its futility checks.
* ``ingest``      3-4. remaining L4 units, fsynced per unit and resumable, interim look after 10 of the 18 new
                  documents (futility only), final look (M1, M3); writes ``result.cpu.json``.
* ``gpu-audit``   5a. L4c audit sample (Phase 0 recipe, unmodified).
* ``gpu-answers`` 5b. P-large on the 58, then B1 on the 58 (continuity first, then new by id) with the
                  pre-declared B1 stop-on-answer after 30 questions; then the verdict.
* ``score``       verdict only (no model).

Exit codes: 0 phase complete (or smoke ok), 2 preflight/input failure, 3 gate tripped (``phase1_gate_tripped``,
the GPU window is never requested), 4 precondition missing, 5 budget stop (re-book, NOT a gate), 6 L3 parse
futility (infrastructure defect: stop, fix, dated amendment, rerun), 75 partial (deadline / stopfile; the next
grant resumes).

Every record is appended with flush + fsync as soon as it exists (``phase0_arms.Recorder``); a re-run skips every
``unit`` already recorded.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
P0DIR = HERE.parent / "agg_phase0"
for _p in (str(P0DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg_phase0 as A  # noqa: E402
import l3_load as L3  # noqa: E402
import phase0_arms as P  # noqa: E402
from phase0_arms import bm25_top  # noqa: E402,F401  (used by ``l4_unit`` under the name run_cpu uses)

# --------------------------------------------------------------------------- pins (frozen inputs)
# The store keeps the Phase 0 corpus id: the base store is a copy of the Phase 0 store and every Phase 0 helper
# (windows, sql_answer, run_audit) reads ``phase0_arms.CORPUS_ID``. ``configure`` keeps both in step.
CORPUS_ID = P.CORPUS_ID

PREDECLARED_SHA = "2d4a532197ec316d253b9009646be6586cc0a0b4db639daf2e7a0af0af4b78af"
PHASE0_PREDECLARED_SHA = "976a5fe87e3db33d6a5e4547ee4bb6cae324f44d107980342b62f7b6212b8bc9"
MANIFEST_DIGEST = "5c834d5e05c224365405a4a9d99e5b3184d55447991df91840c54af5562b3705"
ORCH_PIN = "9a0ab00d88ffdccab3dd9f25d7f5db26c93b7069"
PHASE0_RESEARCH_PIN = "13afe97c8650d0b272846dd0993e688269478149"

# sha256 of the Phase 1 frozen inputs, exactly the table in PREDECLARED.md
FROZEN_P1 = {
    "corpus.json": "bdeb7a978f036a8e2599bc6db752899d5d18bd57f5fbb8ac52a6dd3a1da41f84",
    "facts_sheet.csv": "9f1bbfe34857532306da65ccbad6a6fceed3ab7db9c9fd3fe24da239cfe074aa",
    "facts_sheet_xcheck.json": "49b747d0d08abf0d29a99ac1bc552ac29fa35b7084c5b7dcd45a4f20a840df1c",
    "aggregation_questions.json": "92e9d389f65fb6d5576ba276095076e9edcfedb36916d01a890e4b01cb30939f",
    "golds.jsonl": "fbe97354a73d23e32f84224d2eb629be1813ed208839a0a82c43a120e1b35e1f",
    "verification_record.json": "704352675d2e5fa60b247099367cba6290acfca786c231eaaabde5db5e6b386b",
}
# sha256 of the Phase 0 code the Phase 1 scorer, metrics and L4 stage live in (blobs at research 13afe97c, the
# Phase 0 CPU/GPU job pin). Phase 1 imports these files; it never edits them.
PHASE0_PINS = {
    "phase0_arms.py": "df98936c38c1f11d6f2822fa72baf7967f8182c8c01baf7fdd3668421aa37f41",
    "agg_phase0.py": "7d2c48c86869ea31a5e5c04dfcfb34990e6386ee68d99a7c4ee2dc80f025d8cd",
    "l3_load.py": "621e08c8aa5644414f5f59a04129e65fb1c1351bdef39a56d7de06a1c61ac67f",
    "m1_l3_report.py": "dc9d28c06f5e639d971af55a6caccdeb7a7ffbed589b37c7ef1cf97387209be4",
    "PREDECLARED.md": PHASE0_PREDECLARED_SHA,
}

PILOT_DOCS = ("MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K", "3M_2022_10K")
PILOT_RANKS = (5, 11, 17)          # of the 18 new documents, by extracted size
N_NEW, N_P0, N_CELLS = 18, 10, 211
N_QUESTIONS = 58
M1_GATE = 0.75                     # box value; the Phase 0 kill was 0.70
M1_PILOT_MIN = 0.60                # pilot futility on its 24 cells
M4_MEDIAN_PREFILL_MAX = 2510       # Phase 0 median 510 + 2,000
M4_MEDIAN_CALLS_SOFT = 3           # Phase 0 median 2 + 1
CALLS_PER_UNIT_RESTIMATE = 12      # "stop and re-estimate above 12"
L3_FUTILITY_FAILS = 3
INTERIM_NEW_DOCS = 10
PILOT_WALL_S = 1800
B1_LOOK_AT = 30
B1_STOP_Z = 3.0
BOOKED_L4_S = {"reuse": 7500.0, "fresh": 14200.0}   # PREDECLARED budget table, upper figure
L4_HARD_STOP_X = 1.5               # hard stop at 1.5x the pilot-based projection
PILOT_BUDGET_X = 2.0               # projected ingest wall > 2x the booked estimate -> budget stop
L4_ORCH_FILES = ("src/aggregation/facts_llm.py", "src/aggregation/llm_backend.py")
L4_ORCH_INFO_FILES = ("src/aggregation/store.py", "src/aggregation/__init__.py",
                      "src/typed_decisions/confidence.py")

RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_L3, RC_PARTIAL = 0, 2, 3, 4, 5, 6, 75

# 2026-11-12 (AGG-1.3 / AGG-1.4 amendment 2): the L4 stage the RERUN units use. ``facts_llm.py`` differs from the AGG-1.1
# pin by the unit-label cross-check in ``reconcile`` (orchestrator tests/unit/test_aggregation_unit_override.py); the
# prompt, grammar and extraction are byte-identical. ``l4_reuse_check(..., amended=AMENDED_L4)`` expects this sha256 for
# the rerun units; the seeded AGG-1.1 cache-hit units stay as AGG-1.1 produced them.
AMENDED_L4 = {"src/aggregation/facts_llm.py": "c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451"}
BASE_FY = "2022"                   # the fiscal year the frozen ``agg_phase0.METRICS`` descriptions name


def configure(corpus_id: str) -> None:
    """Point this module and the Phase 0 helpers it calls at one corpus id."""
    global CORPUS_ID
    CORPUS_ID = corpus_id
    P.CORPUS_ID = corpus_id


sha256_file = A.sha256_file


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def digest_docs(docs: list[dict[str, Any]]) -> str:
    """Same canonical digest as ``build_phase1.digest_docs`` (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(docs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    L3._atomic_write(path, text)


# --------------------------------------------------------------------------- inputs
def load_inputs(*, corpus: Path, golds: Path, verification: Path, facts_sheet: Path) -> dict[str, Any]:
    g = [json.loads(l) for l in golds.read_text().splitlines() if l.strip()]
    ver = json.loads(verification.read_text())["questions"]
    return {"corpus": json.loads(corpus.read_text()), "golds": g, "verified": ver, "sheet": facts_sheet}


def verified_agg(inp: dict[str, Any]) -> list[dict[str, Any]]:
    """Only verified questions run; unverified = excluded, never assumed (PREDECLARED)."""
    return [g for g in inp["golds"] if inp["verified"].get(g["id"], {}).get("verified")]


def load_sizes(path: Path) -> dict[str, int]:
    return json.loads(path.read_text())["chars"]


def new_docs(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for d in corpus["docs"] if not d.get("phase0")]


def p0_docs(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for d in corpus["docs"] if d.get("phase0")]


def size_ranks(corpus: dict[str, Any], sizes: dict[str, int]) -> dict[str, int]:
    """Rank (1 = smallest) of each NEW document by extracted size; ties broken by name."""
    order = sorted(new_docs(corpus), key=lambda d: (sizes[d["doc_name"]], d["doc_name"]))
    return {d["doc_name"]: i + 1 for i, d in enumerate(order)}


def new_doc_order(corpus: dict[str, Any], sizes: dict[str, int]) -> list[str]:
    """Pilot documents first (declared order), then the other new documents by ascending extracted size."""
    ranks = size_ranks(corpus, sizes)
    rest = sorted((n for n in ranks if n not in PILOT_DOCS), key=lambda n: ranks[n])
    return list(PILOT_DOCS) + rest


def plan_units(corpus: dict[str, Any], sizes: dict[str, int], reuse_ok: bool) -> list[dict[str, Any]]:
    """Every L4 (field, doc) unit in run order. ``stage`` is reused | pilot | ingest.

    Reuse: the 80 Phase 0 units are cache hits (never re-run) and come first. No reuse: the Phase 0 documents
    are ingested fresh AFTER the pilot, before the remaining new documents (the pilot decides first)."""
    out: list[dict[str, Any]] = []

    def add(doc: str, stage: str) -> None:
        for f in A.METRICS:
            out.append({"unit": f"l4:{f}:{doc}", "doc": doc, "field": f, "stage": stage})

    p0 = sorted(d["doc_name"] for d in p0_docs(corpus))
    order = new_doc_order(corpus, sizes)
    if reuse_ok:
        for d in p0:
            add(d, "reused")
    for d in order[:len(PILOT_DOCS)]:
        add(d, "pilot")
    if not reuse_ok:
        for d in p0:
            add(d, "ingest")
    for d in order[len(PILOT_DOCS):]:
        add(d, "ingest")
    return out


# --------------------------------------------------------------------------- M1
def m1_cells(sheet_path: Path, verified: list[dict[str, Any]], docs: Iterable[str]
             ) -> list[tuple[str, str, float]]:
    """(doc_name, metric, value) for every FILLED cell a verified question reads (``expr`` metrics and the
    ``where`` metric), skipping a question's ``excluded_by_rule`` documents. ``phase0_arms.m1_cells`` with
    the document list as a parameter instead of ``A.DOCS``."""
    sheet = A.read_sheet(sheet_path)
    need: set[tuple[str, str]] = set()
    for g in verified:
        metrics = re.split(r"[-/]", g["expr"])
        if g.get("where"):
            metrics += re.split(r"[-/]", g["where"]["expr"])
        for d in docs:
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


def m1_hits(facts: Iterable[dict[str, Any]], cells: list[tuple[str, str, float]], *,
            unit_scale: dict[str, float] | None = None, rel_tol: float = P.CFG.rel_tol) -> list[bool]:
    """Per cell: found? Exactly ``phase0_arms.m1_found`` (raw ``facts`` table, a period containing 2022, value
    within ``rel_tol`` after unit scale, no leniency for a 1000x unit error) but one boolean per cell."""
    if unit_scale is None:
        from src.aggregation.facts_llm import UNIT_SCALE as unit_scale  # noqa: PLC0415
    by: dict[tuple[str, str], list[float]] = {}
    for f in facts:
        if f.get("value") is None or "2022" not in str(f.get("period") or ""):
            continue
        v = float(f["value"])
        if f.get("unit") in unit_scale:
            v *= unit_scale[f["unit"]]
        by.setdefault((f.get("doc_id"), f.get("field")), []).append(v)
    return [any(abs(abs(v) - abs(gv)) <= rel_tol * abs(gv) + 1e-9 for v in by.get((d, m), []))
            for d, m, gv in cells]


def m1_summary(cells: list[tuple[str, str, float]], hits: list[bool], p0_names: set[str]) -> dict[str, Any]:
    """M1 overall, for the Phase 0 documents and for the new documents, and per document."""
    per: dict[str, dict[str, int]] = {}
    for (d, _m, _v), h in zip(cells, hits):
        b = per.setdefault(d, {"gold": 0, "found": 0})
        b["gold"] += 1
        b["found"] += int(h)

    def agg(names: Iterable[str]) -> dict[str, Any]:
        names = list(names)
        g = sum(per[n]["gold"] for n in names if n in per)
        f = sum(per[n]["found"] for n in names if n in per)
        return {"gold": g, "found": f, "m1": (f / g) if g else None}

    return {"overall": agg(per), "phase0_docs": agg(n for n in per if n in p0_names),
            "new_docs": agg(n for n in per if n not in p0_names), "per_doc": dict(sorted(per.items()))}


# --------------------------------------------------------------------------- M3 / gates
def l4_ok(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """unit -> its first finished (ok) ``l4_unit`` record, reused ones included."""
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.get("kind") == "l4_unit" and r.get("ok") is True and r.get("unit") not in out:
            out[r["unit"]] = r
    return out


def tally(recs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    recs = list(recs)
    return {"n_units": len(recs), "llm_calls": sum(r["llm_calls"] for r in recs),
            "prefill_tokens": sum(r["prefill_tokens"] for r in recs),
            "wall_s": sum(r.get("wall_s") or 0.0 for r in recs)}


def m3_values(prefill: float, calls: float, corpus_tokens: float, chunks: float) -> dict[str, Any]:
    return {"prefill_ratio": prefill / corpus_tokens if corpus_tokens else None,
            "calls_per_100_chunks": 100 * calls / chunks if chunks else None,
            "prefill_tokens": prefill, "llm_calls": calls, "corpus_tokens": corpus_tokens, "n_chunks": chunks}


def m3_trips(m3: dict[str, Any]) -> bool:
    """Phase 0 kill bounds, imported: prefill / corpus tokens > 2.0 or calls / 100 chunks > 150."""
    pr, cp = m3["prefill_ratio"], m3["calls_per_100_chunks"]
    return (pr is None or cp is None or pr > A.KILL["m3_prefill_ratio_max"]
            or cp > A.KILL["m3_calls_per_100_chunks_max"])


def doc_stats(store, corpus_id: str) -> dict[str, dict[str, int]]:
    out = {d["doc_id"]: {"tokens": d["n_tokens_est"] or 0, "chunks": 0} for d in store.corpus_docs(corpus_id)}
    for c in store.chunks(corpus_id):
        if c["doc_id"] in out:
            out[c["doc_id"]]["chunks"] += 1
    return out


def sum_stats(stats: dict[str, dict[str, int]], docs: Iterable[str]) -> tuple[int, int]:
    docs = list(docs)
    return (sum(stats[d]["tokens"] for d in docs if d in stats), sum(stats[d]["chunks"] for d in docs if d in stats))


# --------------------------------------------------------------------------- preflight
def parse_source_sha(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            out[rel.strip()] = h
    return out


def _l4_loop_ast(tree: ast.AST) -> list[str]:
    """AST dumps of the per-unit statements of the ``for fname`` / ``for d`` loop in ``run_cpu`` (the unit key
    assignment, then everything after the skip-if), or ``[]`` if the loop is not found."""
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "run_cpu":
            for outer in ast.walk(fn):
                if isinstance(outer, ast.For) and isinstance(outer.target, ast.Name) and outer.target.id == "fname":
                    for inner in outer.body:
                        if isinstance(inner, ast.For) and isinstance(inner.target, ast.Name) \
                                and inner.target.id == "d":
                            b = inner.body
                            return [ast.dump(b[0])] + [ast.dump(s) for s in b[2:]]
    return []


def l4_unit_ast(source_path: Path | None = None) -> list[str]:
    """The same statement list for ``phase1_arms.l4_unit``: unit key, then everything after the skip-if and
    before the final ``return``."""
    tree = ast.parse((source_path or Path(__file__)).read_text())
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "l4_unit":
            b = fn.body  # [import, unit =, if skip, ...L4 statements..., return True]
            return [ast.dump(b[1])] + [ast.dump(s) for s in b[3:-1]]
    return []


def l4_reuse_check(p0_job: Path, orch_root: Path, research_arms: Path | None = None, *,
                   amended: dict[str, str] | None = None) -> dict[str, Any]:
    """PREDECLARED checkpoint 0: the L4 stage code diff against the Phase 0 CPU job's SOURCE.sha256 pin.

    Empty diff on (a) ``facts_llm.py`` and ``llm_backend.py`` of the orchestrator tree, (b) the Phase 0 research
    ``phase0_arms.py`` (it holds ``windows``, ``bm25_top``, ``Cfg`` and the L4 loop of ``run_cpu``; Phase 1 does
    not edit it) and (c) the Phase 1 ``l4_unit`` statements AST-equal to ``run_cpu``'s L4 loop. Then the 80
    Phase 0 L4 units are cache hits; otherwise all 224 units run fresh and M3 is measured fresh.

    ``amended`` (AGG-1.3 / AGG-1.4 amendment 2, 2026-11-12): ``{orchestrator rel path: expected sha256}`` for the L4 stage
    file(s) the RERUN units use. Such a file must equal its AMENDED sha256 instead of the AGG-1.1 pin (the pin is still
    reported as ``phase0``), and ``fy_stage_ok`` (``l4_fy_stage_problems``) must hold. Every other gating file, and the
    ``l4_unit`` AST, are checked against the pin exactly as before."""
    research_arms = research_arms or Path(P.__file__)
    res: dict[str, Any] = {"p0_job": str(p0_job), "orch_root": str(orch_root), "files": {}, "problems": []}
    try:
        pins = parse_source_sha(p0_job / "SOURCE.sha256")
    except OSError as e:
        res["problems"].append(f"cannot read Phase 0 SOURCE.sha256: {e}")
        res["reuse_ok"] = False
        return res
    for rel in L4_ORCH_FILES + L4_ORCH_INFO_FILES:
        cur = orch_root / rel
        p0 = pins.get(f"orch/{rel}")
        now = sha256_file(cur) if cur.exists() else None
        if amended and rel in amended:
            res["files"][rel] = {"phase0": p0, "phase1": now, "equal": now is not None and now == amended[rel],
                                 "gating": rel in L4_ORCH_FILES, "amended_expect": amended[rel],
                                 "equals_phase0_pin": p0 is not None and p0 == now}
            continue
        res["files"][rel] = {"phase0": p0, "phase1": now, "equal": p0 is not None and p0 == now,
                             "gating": rel in L4_ORCH_FILES}
    rel_arms = "res/benchmarks/agg_phase0/phase0_arms.py"
    arms_now = sha256_file(research_arms)
    res["files"]["research:phase0_arms.py"] = {"phase0": pins.get(rel_arms), "phase1": arms_now,
                                               "equal": pins.get(rel_arms) == arms_now, "gating": True}
    p0_arms = p0_job / rel_arms
    ast_equal = False
    if p0_arms.exists() and sha256_file(p0_arms) == pins.get(rel_arms):
        ref = _l4_loop_ast(ast.parse(p0_arms.read_text()))
        mine = l4_unit_ast()
        ast_equal = bool(ref) and ref == mine
    else:
        res["problems"].append("Phase 0 pinned phase0_arms.py missing or not equal to its SOURCE.sha256 entry")
    res["l4_loop_ast_equal"] = ast_equal
    res["diff_empty"] = all(v["equal"] for v in res["files"].values() if v["gating"]) and ast_equal
    if amended:
        res["amended"] = dict(amended)
        res["fy_stage_problems"] = l4_fy_stage_problems()
        res["problems"] += [f"fy stage: {p}" for p in res["fy_stage_problems"]]
    res["reuse_ok"] = res["diff_empty"] and not res["problems"]
    return res


def preflight(inp: dict[str, Any], set_dir: Path, *, pins: dict[str, str] | None = FROZEN_P1,
              predeclared_sha: str | None = PREDECLARED_SHA, check_pdfs: bool = True,
              run_build_check: bool = True, orch_root: Path | None = None, p0_job: Path | None = None,
              sizes: dict[str, int] | None = None, expect_cells: int | None = N_CELLS,
              strict_p0_pins: bool = True) -> tuple[bool, dict[str, Any]]:
    """Zero-compute checks (PREDECLARED checkpoint 0). ``ok`` is False on any problem."""
    problems: list[str] = []
    info: dict[str, Any] = {}
    corpus = inp["corpus"]
    if pins is not None:
        for name, want in pins.items():
            p = set_dir / name
            got = sha256_file(p) if p.exists() else None
            if got != want:
                problems.append(f"frozen input {name} sha256 {got} != pinned {want}")
    if predeclared_sha is not None:
        pd = set_dir / "PREDECLARED.md"
        got = sha256_file(pd) if pd.exists() else None
        info["predeclared_sha256"] = got
        if got != predeclared_sha:
            problems.append(f"PREDECLARED.md sha256 {got} != recorded {predeclared_sha}")
    if strict_p0_pins:
        for name, want in PHASE0_PINS.items():
            p = P0DIR / name
            got = sha256_file(p) if p.exists() else None
            if got != want:
                problems.append(f"Phase 0 file {name} sha256 {got} != pin {want}")
    docs = corpus["docs"]
    digest = digest_docs(docs)
    info.update(n_docs=len(docs), n_new=len(new_docs(corpus)), n_phase0=len(p0_docs(corpus)),
                manifest_digest=digest)
    if corpus.get("manifest_digest_sha256") != digest:
        problems.append("corpus manifest digest differs from its docs array")
    if pins is not None and digest != MANIFEST_DIGEST:
        problems.append("corpus manifest digest differs from the PREDECLARED pin")
    if check_pdfs:
        bad = [d["doc_name"] for d in docs
               if not Path(d["pdf"]).exists() or sha256_file(Path(d["pdf"])) != d["sha256"]]
        info["n_pdf_ok"] = len(docs) - len(bad)
        if bad:
            problems.append(f"pdf missing or sha256 mismatch: {bad}")
    run_agg = verified_agg(inp)
    info.update(n_questions=len(inp["golds"]), n_agg_verified=len(run_agg),
                n_agg_excluded=len(inp["golds"]) - len(run_agg))
    for g in run_agg:
        if g["gold"] is None:
            problems.append(f"{g['id']} verified but its gold is incomplete")
    if not run_agg:
        problems.append("no verified aggregation questions")
    if pins is not None and len(run_agg) != N_QUESTIONS:
        problems.append(f"{len(run_agg)} verified questions, PREDECLARED has {N_QUESTIONS} of {N_QUESTIONS}")
    cells = m1_cells(inp["sheet"], run_agg, [d["doc_name"] for d in docs])
    p0n = {d["doc_name"] for d in p0_docs(corpus)}
    info["m1_cells"] = {"total": len(cells), "phase0_docs": sum(1 for c in cells if c[0] in p0n),
                        "new_docs": sum(1 for c in cells if c[0] not in p0n)}
    if expect_cells is not None and len(cells) != expect_cells:
        problems.append(f"{len(cells)} M1 cells, PREDECLARED has {expect_cells}")
    if sizes is not None:
        ranks = size_ranks(corpus, sizes)
        got = tuple(ranks.get(n) for n in PILOT_DOCS)
        info["pilot_ranks"] = dict(zip(PILOT_DOCS, got))
        if got != PILOT_RANKS:
            problems.append(f"pilot documents are ranks {got} of the new documents, PREDECLARED has {PILOT_RANKS}")
    if run_build_check:
        # build_phase1.py check recomputes golds with float sum(): CPython 3.12 sums more exactly than 3.11, so
        # run it under the system 3.12 interpreter, never the 3.11 orchestrator venv.
        py = "/usr/bin/python3" if sys.version_info < (3, 12) and Path("/usr/bin/python3").exists() else sys.executable
        cp = subprocess.run([py, "-I", "-B", str(HERE / "build_phase1.py"), "check", "--out", str(set_dir)],
                            capture_output=True, text=True, timeout=300)
        info["build_check"] = {"rc": cp.returncode, "python": py, "out": (cp.stdout + cp.stderr)[-300:]}
        if cp.returncode != 0:
            problems.append("build_phase1.py check exited non-zero")
    if orch_root is not None and p0_job is not None:
        info["l4_reuse"] = l4_reuse_check(p0_job, orch_root)
    info["problems"] = problems
    return not problems, info


# --------------------------------------------------------------------------- reuse seeding
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def seed_reuse(rec: P.Recorder, p0_results: Path) -> dict[str, Any]:
    """Copy the Phase 0 ok L4 units into this run's records as cache hits (``reused: true``) and record the
    Phase 0 ingest numbers beside them. Idempotent: a unit already recorded is not copied twice."""
    p0 = read_jsonl(p0_results)
    ok = l4_ok(p0)
    n = 0
    for unit, r in sorted(ok.items()):
        if unit in rec.done:
            continue
        keep = {k: v for k, v in r.items() if k not in ("kind", "ts")}
        rec.emit("l4_unit", **keep, reused=True, reused_from=str(p0_results))
        n += 1
    ing = next((r for r in p0 if r.get("kind") == "ingest"), None)
    if ing and "p0_ingest" not in rec.done:
        rec.emit("p0_ingest", unit="p0_ingest", **{k: ing[k] for k in (
            "prefill_tokens", "corpus_tokens", "llm_calls", "n_chunks") if k in ing})
    return {"seeded": n, "units": len(ok), **tally(ok.values())}


# --------------------------------------------------------------------------- L3 load of the new documents
def run_l3(rec: P.Recorder, out: Path, inp: dict[str, Any], sizes: dict[str, int], orch_root: Path, *,
           store_src: Path | None, guard: Callable[[], None], parse_fn: Callable[[Path], Any] | None = None,
           now: Callable[[], float] = time.time, smoke: bool = False, corpus_id: str | None = None,
           orch_pin: str = "", research_pin: str = "") -> dict[str, Any]:
    """Checkpoint 1 (CPU, no model). The base store is a sqlite backup of the Phase 0 store; each new document is
    parsed with the existing document flow (local ODL, no OCR server), stored, L3-extracted and recorded with the
    same record kinds as ``l3_load.py``. Fewer than 3 failures are recorded ``doc_fail`` (final): their M1 cells
    count as misses. Three failures stop the run (rc 6)."""
    parse_fn = parse_fn or L3.real_parse
    corpus_id = corpus_id or CORPUS_ID
    for k, v in L3.JOB_ENV.items():
        os.environ[k] = v
    corpus = inp["corpus"]
    order = new_doc_order(corpus, sizes)
    if smoke:
        order = sorted(order, key=lambda n: sizes[n])[:1]
    by_name = {d["doc_name"]: d for d in corpus["docs"]}
    store_path = out / "store.sqlite"
    if not store_path.exists():
        if store_src is None or not store_src.exists():
            rec.emit("precondition_failed", reason=f"no store at {store_path} and no usable --store-src")
            return {"complete": False, "rc": RC_PRECOND}
        P.copy_store(store_src, store_path)
        rec.emit("store_copied", src=str(store_src), sha256=sha256_file(store_path), final=False)
    from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    from src.models.document import DocumentPreprocessResult  # noqa: PLC0415
    store = CorpusStore(store_path)
    have = {d["doc_id"] for d in store.corpus_docs(corpus_id)}
    missing_p0 = sorted(d["doc_name"] for d in p0_docs(corpus) if d["doc_name"] not in have)
    if missing_p0:
        rec.emit("precondition_failed", reason=f"base store lacks Phase 0 documents {missing_p0}")
        store.close()
        return {"complete": False, "rc": RC_PRECOND}
    rec.emit("meta", phase="l3", smoke=smoke, schema="agg_phase1_l3.v1", model_calls=0, orch_pin=orch_pin,
             research_pin=research_pin, n_new_planned=len(order), corpus_id=corpus_id, final=False)
    failed = {r["doc"] for r in rec.records if r["kind"] == "doc_fail"}
    cache_dir = out / "cache"
    for name in order:
        d = by_name[name]
        unit = f"l3:{name}"
        if unit in rec.done:
            continue
        guard()
        if not Path(d["pdf"]).exists() or sha256_file(Path(d["pdf"])) != d["sha256"]:
            rec.emit("doc_fail", unit=unit, doc=name, error="pdf missing or sha256 mismatch", wall_s=0.0, final=True)
            failed.add(name)
        else:
            t0 = now()
            cache = cache_dir / f"{name}.{d['sha256'][:12]}.json"
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
                rec.emit("parse", doc=name, pdf_bytes=d["bytes"], parse_s=parse_s, cached=cached,
                         total_pages=result.total_pages, pages_seen=max((s.page_end for s in result.sections), default=0),
                         n_sections=len(result.sections), n_tables=len(result.tables),
                         n_chars=sum(len(s.content) for s in result.sections),
                         failed_pages=len(result.failed_pages))
                rc = store.ingest(result, corpus_id=corpus_id, doc_id=name, entity=d["company"])
                rec.emit("store", doc=name, status=rc.status, doc_hash=rc.doc_hash, n_chunks=rc.n_chunks,
                         n_tables=rc.n_tables, n_tokens_est=rc.n_tokens_est)
                facts, n_tab, n_rx = L3.doc_l3_facts(store, corpus_id, name)
                store.write_facts(facts)
                units: dict[str, int] = {}
                for f in facts:
                    units[f["unit"]] = units.get(f["unit"], 0) + 1
                rec.emit("l3_doc", doc=name, n_facts=len(facts), n_table_facts=n_tab, n_regex_facts=n_rx,
                         fields=len({f["field"] for f in facts}), units=units, llm_calls=0, prefill_tokens=0)
                rec.emit("doc_done", unit=unit, doc=name, pdf_sha256=d["sha256"], doc_hash=rc.doc_hash,
                         wall_s=now() - t0, final=True)
            except Exception as e:  # noqa: BLE001 - recorded per document, the run continues until the futility rule
                rec.emit("doc_fail", unit=unit, doc=name, error=f"{type(e).__name__}: {e}"[:500],
                         wall_s=now() - t0, final=True)
                failed.add(name)
        if len(failed) >= L3_FUTILITY_FAILS:
            rec.emit("stop", reason="futility_l3_parse", failed=sorted(failed), final=False)
            store.close()
            res = {"complete": True, "infrastructure_stop": "l3_parse_futility", "failed": sorted(failed),
                   "rc": RC_L3}
            return res
    if smoke:
        store.close()
        return {"complete": False, "smoke": True, "failed": sorted(failed), "rc": RC_OK}
    reg = IndexRegistry(store)
    reg.get_or_build(corpus_id, "l1_bm25")
    facts = reg.get_or_build(corpus_id, "l3_facts_det")
    rec.emit("corpus", n_docs=len(store.corpus_docs(corpus_id)), corpus_tokens=store.corpus_tokens(corpus_id),
             n_chunks=len(store.chunks(corpus_id)), n_tables=len(store.tables(corpus_id)), n_facts=len(facts),
             built_layers=sorted(store.built_layers(corpus_id)), corpus_content_hash=store.corpus_content_hash(corpus_id),
             failed=sorted(failed), llm_calls=0, prefill_tokens=0, final=False)
    cells = m1_cells(inp["sheet"], verified_agg(inp), [d["doc_name"] for d in corpus["docs"]])
    l3_facts = [f for f in store.facts(corpus_id) if str(f.get("extractor") or "").startswith("l3")]
    hits = m1_hits(l3_facts, cells, unit_scale=M1R_UNIT_SCALE)
    p0n = {d["doc_name"] for d in p0_docs(corpus)}
    summ = m1_summary(cells, hits, p0n)
    rec.emit("m1_l3", kind_note="L3-only, informational (the gate reads M1 after L4d)", gold_facts=len(cells),
             found=sum(hits), m1_l3_only=summ["overall"]["m1"], new_docs=summ["new_docs"],
             phase0_docs=summ["phase0_docs"], final=False)
    store.close()
    res = {"complete": True, "phase": "l3", "n_new_docs": len(order), "failed": sorted(failed),
           "m1_l3_only": summ["overall"]["m1"], "model_calls": 0, "rc": RC_OK}
    return res


# L3 facts carry units as USD_millions / USD_thousands / USD_billions / USD / unknown; the L4 UNIT_SCALE is the
# same table (millions = 1). Kept here so the L3-only report needs no orchestrator import for the scale.
M1R_UNIT_SCALE = {"USD_thousands": 1e-3, "USD_millions": 1.0, "USD_billions": 1e3, "USD": 1e-6}


# --------------------------------------------------------------------------- L4 (the L3-loaded store -> facts)
def l4_unit(rec, store, backend, units, fname, d, guard, cfg: P.Cfg = P.CFG):
    from src.aggregation import facts_llm  # noqa: PLC0415
    unit = f"l4:{fname}:{d}"
    if unit in rec.done:
        return False
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
    return True


# --------------------------------------------------------------------------- L4 with a per-document fiscal year
def l4_description(desc: str, fy: Any = None) -> str:
    """The field description rendered for one document's fiscal year. ``METRICS`` names ``FY2022`` in every description;
    a filing of another fiscal year gets that year instead (AGG-1.3 amendment 2). ``fy`` None or 2022 returns ``desc``
    itself, so Phase 0 / AGG-1.1 prompts and BM25 queries are byte-identical."""
    fy = BASE_FY if fy in (None, "") else str(fy)
    if fy == BASE_FY:
        return desc
    if f"FY{BASE_FY}" not in desc:
        raise ValueError(f"description does not name FY{BASE_FY}: {desc!r}")
    return desc.replace(f"FY{BASE_FY}", f"FY{fy}")


def doc_fiscal_year(entry: dict[str, Any], corpus: dict[str, Any] | None = None) -> str:
    """The document's own fiscal year from the corpus manifest: the entry's ``fiscal_year`` (set on the AGG-1.3 replaced
    slots), else the manifest's ``fiscal_year`` (AGG-1.1: 2022), else the base year."""
    return str(entry.get("fiscal_year") or (corpus or {}).get("fiscal_year") or BASE_FY)


def l4_fy_stage_problems() -> list[str]:
    """Zero-compute self-check of the fiscal-year rendering: FY2022 is the identity on every frozen description, any
    other year renders exactly the FY2022 text with the year swapped and names no other year."""
    out = []
    for f, d in A.METRICS.items():
        if l4_description(d, None) != d or l4_description(d, "2022") != d or l4_description(d, 2022) != d:
            out.append(f"{f}: FY2022 rendering is not the frozen description")
        r = l4_description(d, "2021")
        if r != d.replace("FY2022", "FY2021") or "2022" in r:
            out.append(f"{f}: FY2021 rendering is wrong: {r!r}")
    return out


def l4_unit_fy(rec, store, backend, units, fname, d, guard, cfg: P.Cfg = P.CFG, *, fy: Any = None):
    """``l4_unit`` with the field description rendered for the document's fiscal year (``fy``). For ``fy`` None / 2022
    the description, the BM25 query, the extraction prompt and the emitted record are those of ``l4_unit`` (the test
    ``FyRenderingTests`` runs both side by side). The AGG-1.1 function itself is NOT edited: it stays AST-equal to
    ``phase0_arms.run_cpu``'s L4 loop, which is what the L4 reuse test compares. Extra record fields: ``fy``,
    ``n_unit_overrides``, ``unit_overrides`` (the L4d unit-label cross-check notes)."""
    from src.aggregation import facts_llm  # noqa: PLC0415
    unit = f"l4:{fname}:{d}"
    if unit in rec.done:
        return False
    guard()
    t0 = time.time()
    desc = l4_description(A.METRICS[fname], fy)
    cands = bm25_top([u for u in units if u["doc_id"] == d],
                     f"{fname.replace('_', ' ')} {desc}", cfg.k_per_doc)
    bf = facts_llm.LazyFieldBackfill(store, backend, n_predict=cfg.l4_n_predict, id_slot=None,
                                     hi=cfg.hi, lo=cfg.lo)
    calls0, pre0 = backend.calls, backend.prefill_tokens
    try:
        out = bf.fill(CORPUS_ID, fname, desc, cands)
        status = "ok"
    except Exception as e:  # noqa: BLE001 — recorded per unit, the run continues
        out, status = {"facts": [], "conflicts": [], "receipts": [], "audit_queue": []}, \
            f"error:{type(e).__name__}:{str(e)[:200]}"
    ovr = list(out.get("unit_overrides") or [])
    rec.emit("l4_unit", unit=unit, field=fname, doc=d, status=status, ok=status == "ok",
             n_candidates=len(cands), n_facts=len(out["facts"]),
             n_conflicts=len(out["conflicts"]), receipts=out["receipts"],
             llm_calls=backend.calls - calls0, prefill_tokens=backend.prefill_tokens - pre0,
             wall_s=time.time() - t0, final=status == "ok",
             fy=BASE_FY if fy in (None, "") else str(fy), n_unit_overrides=len(ovr), unit_overrides=ovr[:20])
    return True



def _doc_complete(rec: P.Recorder, doc: str) -> bool:
    return all(f"l4:{f}:{doc}" in rec.done for f in A.METRICS)


def _p1_state(rec: P.Recorder) -> dict[str, Any]:
    """Last ``pilot_done`` / ``interim`` records, by unit key."""
    return {r["unit"]: r for r in rec.records if r.get("unit") in ("pilot_done", "interim10")}


def _l3_failed(rec: P.Recorder) -> set[str]:
    done = {r["doc"] for r in rec.records if r["kind"] == "doc_done"}
    return {r["doc"] for r in rec.records if r["kind"] == "doc_fail"} - done


def _write_cpu_result(out: Path, **res: Any) -> dict[str, Any]:
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = res.get("predeclared_sha256") or PREDECLARED_SHA
    _atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    return res


def _gate_stop(rec: P.Recorder, out: Path, gates: list[str], where: str, **detail: Any) -> dict[str, Any]:
    rec.emit("gate_tripped", unit=f"gate_tripped:{where}", gates=gates, where=where, **detail)
    rec.emit("stop", reason="phase1_gate_tripped", gates=gates, where=where, final=False)
    res = _write_cpu_result(out, complete=True, futility_stop="+".join(gates), gate_tripped=gates,
                            verdict="phase1_gate_tripped", where=where, m1=detail.get("m1"), m3_passed=False,
                            rc=RC_GATE)
    res["rc"] = RC_GATE
    return res


def _budget_stop(rec: P.Recorder, reason: str, **detail: Any) -> dict[str, Any]:
    rec.emit("stop", reason=f"budget_stop:{reason}", final=False, **detail)
    return {"complete": False, "budget_stop": reason, "rc": RC_BUDGET, **detail}


def _cells_for(inp: dict[str, Any]) -> list[tuple[str, str, float]]:
    return m1_cells(inp["sheet"], verified_agg(inp), [d["doc_name"] for d in inp["corpus"]["docs"]])


def pilot_eval(*, ok: dict[str, dict[str, Any]], plan: list[dict[str, Any]], stats: dict[str, dict[str, int]],
               pilot_hits: list[bool], reused_tally: dict[str, Any], total_tokens: int, total_chunks: int,
               booked_l4_s: float, reuse_ok: bool) -> dict[str, Any]:
    """PREDECLARED checkpoint 2, pure arithmetic. Projection is linear in (field, doc) units."""
    pilot_units = [p["unit"] for p in plan if p["stage"] == "pilot"]
    run_units = [p for p in plan if p["stage"] != "reused"]
    pil = tally(ok[u] for u in pilot_units if u in ok)
    n = max(pil["n_units"], 1)
    per = {"calls": pil["llm_calls"] / n, "prefill": pil["prefill_tokens"] / n, "wall_s": pil["wall_s"] / n}
    docs = sorted({p["doc"] for p in plan if p["stage"] == "pilot"})
    ptok, pchunks = sum_stats(stats, docs)
    proj_calls = reused_tally["llm_calls"] + per["calls"] * len(run_units)
    proj_pre = reused_tally["prefill_tokens"] + per["prefill"] * len(run_units)
    proj_m3 = m3_values(proj_pre, proj_calls, total_tokens, total_chunks)
    proj_wall = per["wall_s"] * len(run_units)
    found = sum(pilot_hits)
    pm1 = found / len(pilot_hits) if pilot_hits else None
    gates = []
    if m3_trips(proj_m3):
        gates.append("M3")
    if pm1 is None or pm1 < M1_PILOT_MIN:
        gates.append("M1")
    budget = []
    if proj_wall > PILOT_BUDGET_X * booked_l4_s:
        budget.append("projected_ingest_wall_over_2x_booked")
    if per["calls"] > CALLS_PER_UNIT_RESTIMATE:
        budget.append("calls_per_unit_over_12")
    return {"n_pilot_units": pil["n_units"], "pilot": pil, "per_unit": per,
            "pilot_calls_per_100_chunks": (100 * pil["llm_calls"] / pchunks) if pchunks else None,
            "pilot_prefill_ratio": (pil["prefill_tokens"] / ptok) if ptok else None,
            "pilot_m1": {"gold": len(pilot_hits), "found": found, "m1": pm1},
            "units_to_run": len(run_units), "projected_m3": proj_m3, "projected_l4_wall_s": proj_wall,
            "booked_l4_s": booked_l4_s, "booking_basis": "reuse" if reuse_ok else "fresh",
            "rebook_cap_s": 1.25 * proj_wall, "hard_stop_s": L4_HARD_STOP_X * proj_wall,
            "gates_tripped": gates, "budget_stops": budget}


def run_l4_stage(rec: P.Recorder, out: Path, store, backend, inp: dict[str, Any], sizes: dict[str, int], *,
                 stage: str, reuse_ok: bool, guard: Callable[[], None], smoke: bool = False,
                 cfg: P.Cfg = P.CFG, corpus_id: str | None = None, booked_l4_s: float | None = None,
                 pilot_wall_s: float = PILOT_WALL_S) -> dict[str, Any]:
    """Checkpoints 2-4. ``stage`` = ``pilot`` (24 units, gates on the projection) or ``ingest`` (all remaining
    units, interim and final looks). Smoke: ONE real unit on the smallest loaded new document, then return."""
    corpus_id = corpus_id or CORPUS_ID
    corpus = inp["corpus"]
    booked = booked_l4_s if booked_l4_s else BOOKED_L4_S["reuse" if reuse_ok else "fresh"]
    plan = plan_units(corpus, sizes, reuse_ok)
    order = new_doc_order(corpus, sizes)
    cells = _cells_for(inp)
    p0n = {d["doc_name"] for d in p0_docs(corpus)}
    failed = _l3_failed(rec)
    stats = doc_stats(store, corpus_id)
    state = _p1_state(rec)
    if stage == "ingest" and "pilot_done" not in state and not smoke:
        rec.emit("precondition_failed", reason="the pilot has not completed (run --phase pilot first)")
        return {"complete": False, "rc": RC_PRECOND}
    if stage == "pilot" and "pilot_done" in state and not smoke:
        pd = state["pilot_done"]
        if pd.get("gates_tripped"):
            return {"complete": True, "phase": "pilot", "gate_tripped": pd["gates_tripped"], "rc": RC_GATE}
        if pd.get("budget_stops"):
            return {"complete": False, "budget_stop": "+".join(pd["budget_stops"]), "rc": RC_BUDGET}
        return {"complete": True, "phase": "pilot", "rc": RC_OK, "already_done": True}
    if stage == "ingest" and "pilot_done" in state and state["pilot_done"].get("gates_tripped"):
        return {"complete": True, "already_stopped": True, "rc": RC_GATE}
    units = P.windows(store, corpus_id, cfg.window_chars)
    for p in plan:
        if p["doc"] in failed and p["stage"] != "reused" and f"l4skip:{p['doc']}" not in rec.done and not smoke:
            rec.emit("l4_skip", unit=f"l4skip:{p['doc']}", doc=p["doc"],
                     reason="doc_fail at L3: no chunks; its cells count as M1 misses")
    todo = [p for p in plan if p["stage"] != "reused" and (stage == "ingest" or p["stage"] == "pilot")
            and p["doc"] not in failed]
    if smoke:  # any stage: the smoke document is the smallest LOADED new document, not necessarily a pilot one
        loaded = [p for p in plan if p["stage"] != "reused" and p["doc"] not in p0n and p["doc"] not in failed
                  and stats.get(p["doc"], {}).get("chunks")]
        smallest = min((p["doc"] for p in loaded), key=lambda d: sizes[d], default=None)
        todo = [p for p in loaded if p["doc"] == smallest][:1]
    pilot_units = [q["unit"] for q in plan if q["stage"] == "pilot"]
    if stage == "ingest" and not smoke:
        early = _interim_look(rec, out, store, inp, plan, order, stats, cells, corpus_id=corpus_id, failed=failed)
        if early:
            return early
    for p in todo:
        if p["unit"] in rec.done:
            continue
        if stage == "pilot" and not smoke:
            ok_now = l4_ok(rec.records)
            used = tally(ok_now[u] for u in pilot_units if u in ok_now)["wall_s"]
            if used > pilot_wall_s:
                rec.emit("pilot_bound_hit", unit="pilot_bound_hit", wall_s=used, bound_s=pilot_wall_s)
                return _budget_stop(rec, "pilot_wall_over_1800s", wall_s=used)
        l4_unit(rec, store, backend, units, p["field"], p["doc"], guard, cfg)
        if smoke:
            return {"complete": False, "smoke": True, "rc": RC_OK}
        if stage == "ingest":
            res = _ingest_checks(rec, out, store, inp, plan, order, stats, cells, state=state,
                                 corpus_id=corpus_id, failed=failed)
            if res:
                return res
    if smoke:
        return {"complete": False, "smoke": True, "rc": RC_OK}
    ok = l4_ok(rec.records)
    if stage == "pilot":
        if not all(u in ok for u in pilot_units if u.split(":", 2)[2] not in failed):
            return {"complete": False, "reason": "pilot units with errors; re-grant retries them", "rc": RC_PARTIAL}
        pcells = [c for c in cells if c[0] in PILOT_DOCS]
        hits = m1_hits(store.facts(corpus_id), pcells)
        reused = tally(r for r in ok.values() if r.get("reused"))
        ev = pilot_eval(ok=ok, plan=plan, stats=stats, pilot_hits=hits, reused_tally=reused,
                        total_tokens=sum(v["tokens"] for v in stats.values()),
                        total_chunks=sum(v["chunks"] for v in stats.values()), booked_l4_s=booked,
                        reuse_ok=reuse_ok)
        rec.emit("pilot_done", unit="pilot_done", **ev)
        _atomic_write(out / "result.pilot.json", json.dumps(
            {"phase": "pilot", "complete": True, **ev, "predeclared_sha256": PREDECLARED_SHA}, indent=1,
            default=str) + "\n")
        if ev["gates_tripped"]:
            return _gate_stop(rec, out, ev["gates_tripped"], "pilot", m1=ev["pilot_m1"]["m1"], pilot=ev)
        if ev["budget_stops"]:
            return _budget_stop(rec, "+".join(ev["budget_stops"]), pilot=ev)
        return {"complete": True, "phase": "pilot", "rc": RC_OK, **ev}
    return _finish_ingest(rec, out, store, inp, plan, stats, cells, p0n, corpus_id=corpus_id, failed=failed,
                          reuse_ok=reuse_ok)


def _evaluated_docs(rec: P.Recorder, plan: list[dict[str, Any]], failed: set[str]) -> list[str]:
    """Documents whose 8 L4 units are all recorded (reused ones included)."""
    return [d for d in sorted({p["doc"] for p in plan}) if d not in failed and _doc_complete(rec, d)]


def _interim_look(rec, out, store, inp, plan, order, stats, cells, *, corpus_id, failed) -> dict[str, Any] | None:
    """PREDECLARED checkpoint 3, futility only: after 10 of the 18 new documents (pilot included) the M1 gate is
    mathematically unreachable (``found + not_yet_evaluated < 0.75 x 211``) or the running M3 is over a kill bound.
    Never an early GO. Evaluated once, the first time ten new documents are complete."""
    new_done = [d for d in order if d not in failed and _doc_complete(rec, d)]
    if len(new_done) < INTERIM_NEW_DOCS or "interim10" in rec.done:
        return None
    evaluated = _evaluated_docs(rec, plan, failed)
    hits = m1_hits(store.facts(corpus_id), cells)
    found = sum(h for c, h in zip(cells, hits) if c[0] in evaluated)
    nye = sum(1 for c in cells if c[0] not in evaluated)
    unreachable = found + nye < M1_GATE * len(cells)
    tok, chunks = sum_stats(stats, evaluated)
    ok = {u: r for u, r in l4_ok(rec.records).items() if u.split(":", 2)[2] in evaluated}
    m3 = m3_values(sum(r["prefill_tokens"] for r in ok.values()), sum(r["llm_calls"] for r in ok.values()),
                   tok, chunks)
    gates = (["M1"] if unreachable else []) + (["M3"] if m3_trips(m3) else [])
    rec.emit("interim10", unit="interim10", n_new_docs_done=len(new_done), docs_evaluated=len(evaluated),
             found=found, not_yet_evaluated=nye, gate_cells=len(cells), m1_unreachable=unreachable,
             running_m3=m3, gates_tripped=gates)
    return _gate_stop(rec, out, gates, "interim", m1=None, running_m3=m3) if gates else None


def _ingest_checks(rec, out, store, inp, plan, order, stats, cells, *, state, corpus_id, failed
                   ) -> dict[str, Any] | None:
    """After every unit: the budget stops, then the interim look when ten new documents are complete."""
    pilot = state["pilot_done"]
    fresh = tally(r for r in l4_ok(rec.records).values() if not r.get("reused"))
    if fresh["n_units"] >= 24 and fresh["llm_calls"] / fresh["n_units"] > CALLS_PER_UNIT_RESTIMATE:
        return _budget_stop(rec, "calls_per_unit_over_12", calls_per_unit=fresh["llm_calls"] / fresh["n_units"])
    if 0 < pilot["hard_stop_s"] < fresh["wall_s"]:
        return _budget_stop(rec, "l4_wall_over_1.5x_pilot_projection", wall_s=fresh["wall_s"],
                            hard_stop_s=pilot["hard_stop_s"])
    return _interim_look(rec, out, store, inp, plan, order, stats, cells, corpus_id=corpus_id, failed=failed)


def _finish_ingest(rec, out, store, inp, plan, stats, cells, p0n, *, corpus_id, failed, reuse_ok
                   ) -> dict[str, Any]:
    ok = l4_ok(rec.records)
    expected = [p["unit"] for p in plan if p["doc"] not in failed]
    if not all(u in ok for u in expected):
        return {"complete": False, "reason": "l4 units with errors; re-grant retries them", "rc": RC_PARTIAL}
    allr = list(ok.values())
    fresh = tally(r for r in allr if not r.get("reused"))
    reused = tally(r for r in allr if r.get("reused"))
    total_tokens = sum(v["tokens"] for v in stats.values())
    total_chunks = sum(v["chunks"] for v in stats.values())
    p0_ing = next((r for r in rec.records if r.get("kind") == "p0_ingest"), None)
    new_names = [d for d in stats if d not in p0n]
    inc_tokens, inc_chunks = sum_stats(stats, new_names)
    gated = m3_values(sum(r["prefill_tokens"] for r in allr), sum(r["llm_calls"] for r in allr),
                      total_tokens, total_chunks)
    incremental = m3_values(fresh["prefill_tokens"], fresh["llm_calls"], inc_tokens if reuse_ok else total_tokens,
                            inc_chunks if reuse_ok else total_chunks)
    rec.emit("ingest", unit="ingest", prefill_tokens=gated["prefill_tokens"], corpus_tokens=total_tokens,
             llm_calls=gated["llm_calls"], n_chunks=total_chunks, n_windows=len(P.windows(store, corpus_id,
                                                                                         P.CFG.window_chars)),
             cache_n=sum(x.get("cache_n", 0) for r in allr for x in r["receipts"]),
             reused_units=reused["n_units"], reused=reused, incremental=incremental, phase0_recorded=p0_ing,
             calls_per_unit_new=(fresh["llm_calls"] / fresh["n_units"]) if fresh["n_units"] else None,
             prefill_per_unit_new=(fresh["prefill_tokens"] / fresh["n_units"]) if fresh["n_units"] else None,
             wall_s_new=fresh["wall_s"])
    hits = m1_hits(store.facts(corpus_id), cells)
    summ = m1_summary(cells, hits, p0n)
    m1 = summ["overall"]["m1"]
    rec.emit("m1", unit="m1", gold_facts=len(cells), found=sum(hits), m1=m1, phase0_docs=summ["phase0_docs"],
             new_docs=summ["new_docs"], per_doc=summ["per_doc"], doc_fail=sorted(failed))
    gates = (["M1"] if m1 is None or m1 < M1_GATE else []) + (["M3"] if m3_trips(gated) else [])
    if gates:
        return _gate_stop(rec, out, gates, "final", m1=m1, m3=gated)
    res = _write_cpu_result(out, complete=True, futility_stop=None, gate_tripped=[], m1=m1, m3=gated,
                            m3_incremental=incremental, m3_passed=True, m1_phase0_docs=summ["phase0_docs"],
                            m1_new_docs=summ["new_docs"], doc_fail=sorted(failed), l4_reused=reuse_ok, rc=RC_OK)
    return res


# --------------------------------------------------------------------------- GPU: P-large + B1
def paired_stats(p_correct: dict[str, bool], b_correct: dict[str, bool], qids: list[str]) -> dict[str, Any]:
    """b = only P-large right, c = only B1 right (None counts as not correct); z = (b - c) / sqrt(b + c)."""
    both = onlyp = onlyb = neither = 0
    for q in qids:
        a, b = bool(p_correct.get(q)), bool(b_correct.get(q))
        both += a and b
        onlyp += a and not b
        onlyb += b and not a
        neither += not a and not b
    d = onlyp + onlyb
    z = (onlyp - onlyb) / math.sqrt(d) if d else 0.0
    return {"n": len(qids), "both_right": both, "only_p_large": onlyp, "only_b1": onlyb, "neither": neither,
            "p_large_acc": (both + onlyp) / len(qids) if qids else None,
            "b1_acc": (both + onlyb) / len(qids) if qids else None, "z": z}


def b1_look(records: Iterable[dict[str, Any]], agg_ids: list[str], n: int | None = None
            ) -> dict[str, Any] | None:
    """PREDECLARED stop-on-answer for B1 only: after B1 has answered ``n`` questions (the first ``n`` in run order)
    stop iff P-large minus B1 on those ``n``, paired, is >= 3 SE (SE from the discordant pairs). ``None`` until
    ``n`` B1 answers exist."""
    n = n or B1_LOOK_AT
    recs = list(records)
    b1 = {r["qid"]: r for r in recs if r.get("kind") == "answer" and r.get("arm") == "B1" and r.get("final")}
    pl = {r["qid"]: r for r in recs if r.get("kind") == "answer" and r.get("arm") == "P-large" and r.get("final")}
    first = agg_ids[:n]
    if len(first) < n or not all(q in b1 for q in first):
        return None
    st = paired_stats({q: r.get("correct") is True for q, r in pl.items()},
                      {q: r.get("correct") is True for q, r in b1.items()}, first)
    st["stop"] = st["z"] >= B1_STOP_Z
    return st


def p_large_answer(backend, store, q: dict[str, Any], *, scope: tuple[str | None, str, int], planner,
                   docs_meta, built: list[str], sql_present: bool, names, cfg: P.Cfg, corpus_chunks,
                   corpus_bm25, doc_chunks: Callable[[str], list[dict[str, Any]]]
                   ) -> tuple[str, str, dict[str, Any]]:
    """One P-large aggregation answer; the branch logic of ``phase0_arms.run_answers`` unchanged."""
    extra: dict[str, Any] = {}
    scope_doc, scope_why, qsf = scope
    pf, _ = planner.scoped_facts(q["question"], docs_meta, questions_so_far=qsf,
                                 available_layers=tuple(built), sql_fields_present=sql_present)
    dec = planner.plan(q["question"], pf)
    extra.update(path=dec.path, decided_by=dec.decided_by, receipt_stub=dec.receipt.get("stub", False),
                 scope_doc=scope_doc, scope_reason=scope_why, planner_questions_so_far=qsf,
                 planner_corpus_tokens=pf.corpus_tokens)
    if dec.path == "sql":
        text, st, info = P.sql_answer(backend, store, q["question"], names, cfg)
        extra.update(info)
    elif scope_doc:
        text, st = P.single_doc_answer(backend, store, scope_doc, q["question"], names, cfg, dec.path,
                                       doc_chunks(scope_doc), extra)
    else:
        text, st = P.agent_answer(backend, corpus_chunks, q["question"], names, cfg, bm25=corpus_bm25)
    return text, st, extra


def run_answers(rec: P.Recorder, store, backend, *, agg: list[dict[str, Any]], names: list[str],
                guard: Callable[[], None], smoke: bool = False, cfg: P.Cfg = P.CFG,
                corpus_id: str | None = None) -> dict[str, Any]:
    """Checkpoint 5b: P-large on every verified question, then B1 in the same order (continuity first, then new
    by id; ``golds.jsonl`` order). No P-small / B-today / B2 (PREDECLARED: not in AGG-1.1)."""
    from src.aggregation import planner  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    corpus_id = corpus_id or CORPUS_ID
    corpus_chunks = store.chunks(corpus_id)
    corpus_bm25 = BM25Index([c["text"] for c in corpus_chunks])
    docs_meta = store.corpus_docs(corpus_id)
    sql_present = bool(store.facts(corpus_id))
    built = sorted(store.built_layers(corpus_id))
    canon = store.refresh_canonical(corpus_id)
    if "canonical" not in rec.done:
        rec.emit("canonical", unit="canonical", **canon)
    plan_items = [("P-large", "agg", q) for q in agg]
    scopes = P._scope_plan(planner, docs_meta, plan_items)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for c in corpus_chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    ids = [q["id"] for q in agg]

    def emit_answer(arm: str, q: dict[str, Any], text: str, st: str, t: P.Tally, t0: float,
                    extra: dict[str, Any]) -> None:
        ok = P.score_agg(q, text) if st == "ok" else None
        rec.emit("answer", unit=f"answer:{arm}:{q['id']}", arm=arm, qid=q["id"], qclass="agg", status=st,
                 correct=ok, calls=t.calls, prefill_tokens=t.prefill, answer=text[-600:],
                 final=st in ("ok", "out_of_window", "budget_exhausted"), wall_s=time.time() - t0, **extra)

    for q in agg:
        unit = f"answer:P-large:{q['id']}"
        if unit in rec.done:
            continue
        guard()
        t, t0 = P.Tally(backend), time.time()
        text, st, extra = p_large_answer(backend, store, q, scope=scopes[unit], planner=planner,
                                         docs_meta=docs_meta, built=built, sql_present=sql_present, names=names,
                                         cfg=cfg, corpus_chunks=corpus_chunks, corpus_bm25=corpus_bm25,
                                         doc_chunks=lambda d: by_doc.get(d, []))
        emit_answer("P-large", q, text, st, t, t0, extra)
        if smoke:
            return {"complete": False, "smoke": True}
    for q in agg:
        unit = f"answer:B1:{q['id']}"
        if "b1_look" in rec.done:
            look = next(r for r in rec.records if r.get("unit") == "b1_look")
            if look.get("stop"):
                break
        if unit in rec.done:
            pass
        else:
            guard()
            t, t0 = P.Tally(backend), time.time()
            text, st = P.agent_answer(backend, corpus_chunks, q["question"], names, cfg, bm25=corpus_bm25)
            emit_answer("B1", q, text, st, t, t0, {})
            if smoke:
                return {"complete": False, "smoke": True}
        if "b1_look" not in rec.done:
            look = b1_look(rec.records, ids)
            if look is not None:
                rec.emit("b1_look", unit="b1_look", **look)
                if look["stop"]:
                    rec.emit("b1_early_stop", unit="b1_early_stop", n_answered=look["n"], z=look["z"],
                             only_p_large=look["only_p_large"], only_b1=look["only_b1"])
                    break
    b1_stopped = "b1_early_stop" in rec.done
    done = all(f"answer:P-large:{q['id']}" in rec.done for q in agg) and (
        b1_stopped or all(f"answer:B1:{q['id']}" in rec.done for q in agg))
    return {"complete": done, "b1_early_stop": b1_stopped}


# --------------------------------------------------------------------------- verdict
def _median(xs: list[float]) -> float:
    return A._median(xs)


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return s[k]


def sql_latency(store, answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Seconds of the generated-SQL execution on the 28-document store (reported, no gate). The generated SQL
    text is in each answer record; it is re-executed read-only against the same store, once per answer."""
    ts = []
    for r in answers:
        if r.get("sql") and r.get("sql_valid"):
            t0 = time.perf_counter()
            try:
                store.conn.execute(r["sql"]).fetchall()
            except Exception:  # noqa: BLE001 - a failing statement has no latency to report
                continue
            ts.append(time.perf_counter() - t0)
    walls = [r["wall_s"] for r in answers if r.get("wall_s") is not None]
    return {"n_sql": len(ts), "sql_exec_median_s": _median(ts) if ts else None, "sql_exec_p95_s": _pct(ts, 0.95),
            "sql_exec_max_s": max(ts) if ts else None, "question_wall_median_s": _median(walls) if walls else None,
            "question_wall_max_s": max(walls) if walls else None,
            "phase0_comparator": "mean 7 s, max 26 s over 20 questions, LLM calls included"}


def verdict(cpu_records: list[dict[str, Any]], own_records: list[dict[str, Any]], agg: list[dict[str, Any]],
            *, n_excluded: int = 0, store=None) -> dict[str, Any]:
    """Gates M1, M3 (read from the CPU job's records), M4 (computed here) and informational M2."""
    ing = next((r for r in cpu_records if r.get("kind") == "ingest"), None)
    m1r = next((r for r in cpu_records if r.get("kind") == "m1"), None)
    ans = [r for r in own_records if r.get("kind") == "answer" and r.get("final")]
    pl = {r["qid"]: r for r in ans if r["arm"] == "P-large"}
    b1 = {r["qid"]: r for r in ans if r["arm"] == "B1"}
    ids = [q["id"] for q in agg]
    is_cont = lambda q: q.get("continuity", not q["id"].startswith("p1_"))  # noqa: E731
    cont = [q["id"] for q in agg if is_cont(q)]
    new = [q["id"] for q in agg if not is_cont(q)]
    m1 = m1r["m1"] if m1r else None
    m3 = m3_values(ing["prefill_tokens"], ing["llm_calls"], ing["corpus_tokens"], ing["n_chunks"]) if ing else None
    trips: list[str] = []
    if m1 is None or m1 < M1_GATE:
        trips.append("M1")
    if m3 is None or m3_trips(m3):
        trips.append("M3")
    plr = [pl[q] for q in ids if q in pl]
    p_complete = len(plr) == len(ids)
    m4 = {"n": len(plr), "median_calls": _median([r["calls"] for r in plr]) if plr else None,
          "median_prefill": _median([r["prefill_tokens"] for r in plr]) if plr else None}
    if p_complete and (m4["median_calls"] > A.KILL["median_calls_max"] or m4["median_calls"] > M4_MEDIAN_CALLS_SOFT
                       or m4["median_prefill"] > M4_MEDIAN_PREFILL_MAX):
        trips.append("M4")
    pc = {q: r["correct"] is True for q, r in pl.items()}
    bc = {q: r["correct"] is True for q, r in b1.items()}
    answered = [q for q in ids if q in b1]
    m2 = {"informational": True, "b1_early_stop": any(r.get("kind") == "b1_early_stop" for r in own_records),
          "n_p_large": len(pl), "n_b1_answered": len(b1), "n_excluded_unverified": n_excluded,
          "p_large_acc_all": (sum(pc.values()) / len(pc)) if pc else None,
          "paired_on_b1_answered": paired_stats(pc, bc, answered),
          "continuity": paired_stats(pc, bc, [q for q in cont if q in b1]),
          "new": paired_stats(pc, bc, [q for q in new if q in b1])}
    b1_complete = m2["b1_early_stop"] or all(q in b1 for q in ids)
    complete = p_complete and b1_complete
    out = {"M1": m1, "M3": m3, "M4": m4, "M2": m2, "gates_tripped": trips,
           "n_excluded_unverified": n_excluded, "complete": complete,
           "verdict": ("phase1_gate_tripped" if trips else "agg_1_1_done") if complete else "incomplete"}
    if m1r:
        out["M1_detail"] = {k: m1r.get(k) for k in ("gold_facts", "found", "phase0_docs", "new_docs")}
    if ing:
        out["M3_incremental"] = ing.get("incremental")
        out["M3_phase0_recorded"] = ing.get("phase0_recorded")
    if store is not None:
        out["sql_latency"] = sql_latency(store, plr)
    return out


def cpu_ready(result: dict[str, Any]) -> tuple[bool, str]:
    """GPU precondition on the CPU row's ``result.cpu.json``: complete, no gate tripped, M1 >= 0.75, M3 passed."""
    ok = (result.get("complete") is True and not result.get("futility_stop") and not result.get("gate_tripped")
          and (result.get("m1") or 0) >= M1_GATE and result.get("m3_passed") is True)
    return ok, (f"complete={result.get('complete')} futility={result.get('futility_stop')} m1={result.get('m1')} "
                f"m3_passed={result.get('m3_passed')}")
