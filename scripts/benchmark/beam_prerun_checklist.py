#!/usr/bin/env python3
"""M-12c: the BEAM pre-run do-not-copy checklist (items 1-8) as a mechanical check.

Zero compute: reads run directories (``scored-beam-*.json`` summary) and an optional per-arm
RECEIPT, and reports each item as PASS / FAIL / UNKNOWN with the evidence. UNKNOWN is the
honest state when the run left no receipt; it is never promoted to PASS. A run is
``cleared`` only when all 8 items are PASS.

The 2026-09-18 runs predate receipts, so items 2, 3 (partly), 5, 6 and 7 come back UNKNOWN for
them; that is the recovery gap M-12c names, now machine-visible.

Receipt (``beam_run_receipt.json`` beside the run's scored file; written by the launcher)::

  {"arm": "rag", "reader": {"model": "...", "max_tokens": 1024, "temperature": 0.0,
                            "prompt_sha256": "...", "closed_book_instruction": false},
   "ingest": {"roles": ["user", "assistant"], "chunking": "pair_chunk", "retrieval_budget": 8000},
   "dataset": {"path": ".../100K-00000-of-00001.parquet", "sha256": "..."}}

Item map (M-12c): (1) BEAM's own fold, never a binarised micro-average; (2) both roles
ingested; (3) synthesis max_tokens not below need; (4) judge != reader, fixed across arms;
(5) dataset staged with a verified digest; (6) identical retrieval budget/chunking, control vs
arm; (7) prompt-matched arms incl. the closed-book instruction; (8) external reference curves
used only as floors (far below = configuration defect).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
REQUIRED_ROLES = {"user", "assistant"}
BEAM_FOLD_FLOOR = 0.557  # BEAM issue #125 Raw ChromaDB arm, BEAM fold (author fold 0.490)
FLOOR_MARGIN = 0.10  # "far below the floor": more than this under it
FLOOR_ARMS = {"rag", "trace"}  # the retrieval arms the Raw-ChromaDB floor is comparable to
DEFAULT_MIN_MAX_TOKENS = 1024  # parameter, not a measured need: pass the real one in
PROVENANCE_SHA = "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace"


def _item(n: int, status: str, why: str) -> dict[str, Any]:
    return {"item": n, "status": status, "why": why}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(model: str | None) -> str:
    return Path(str(model or "")).name.lower().removesuffix(".gguf")


def check_arms(
    arms: Sequence[Mapping[str, Any]],
    *,
    min_max_tokens: int = DEFAULT_MIN_MAX_TOKENS,
    expected_dataset_sha: str = PROVENANCE_SHA,
    hash_file=sha256_file,
) -> dict[str, Any]:
    """``arms``: dicts with ``arm``, ``summary`` (scored-run summary) and optional ``receipt``."""
    items: list[dict[str, Any]] = []
    summaries = {a["arm"]: a.get("summary") or {} for a in arms}
    receipts = {a["arm"]: a.get("receipt") for a in arms}
    names = sorted(summaries)

    # (1) the BEAM fold, not a binarised micro-average.
    bad = [n for n in names if summaries[n].get("fold") != "beam_macro"
           or summaries[n].get("headline") is None]
    items.append(_item(1, FAIL if bad else PASS,
                       f"arms without a beam_macro headline: {bad}" if bad
                       else "every arm headline is the beam_macro fold; micro/binarised are secondary"))

    # (2) both roles ingested.
    roles = {n: set(((receipts[n] or {}).get("ingest") or {}).get("roles") or []) for n in names}
    if any(not receipts[n] or not roles[n] for n in names):
        items.append(_item(2, UNKNOWN, "no ingest.roles receipt for: "
                           f"{[n for n in names if not roles[n]]}"))
    else:
        short = [n for n in names if not REQUIRED_ROLES <= roles[n]]
        items.append(_item(2, FAIL if short else PASS,
                           f"arms missing a role: {short}" if short else "user+assistant ingested"))

    # (3) max_tokens not below need; observed truncation is evidence against.
    trunc = {n: int(summaries[n].get("truncated_rows") or 0) for n in names}
    mt = {n: ((receipts[n] or {}).get("reader") or {}).get("max_tokens") for n in names}
    low = [n for n in names if mt[n] is not None and mt[n] < min_max_tokens]
    if low:
        items.append(_item(3, FAIL, f"max_tokens below {min_max_tokens}: {low}"))
    elif any(mt[n] is None for n in names):
        items.append(_item(3, UNKNOWN, f"no reader.max_tokens receipt for "
                           f"{[n for n in names if mt[n] is None]}; observed truncated rows {trunc}"))
    else:
        items.append(_item(3, PASS if not any(trunc.values()) else FAIL,
                           f"max_tokens>={min_max_tokens}; truncated rows {trunc}"))

    # (4) judge != reader and identical across arms.
    judges = {n: summaries[n].get("judge_model") for n in names}
    readers = {n: _norm(((receipts[n] or {}).get("reader") or {}).get("model")
                        or summaries[n].get("model_role")) for n in names}
    if any(not j for j in judges.values()):
        items.append(_item(4, UNKNOWN, "an arm has no judge_model"))
    elif len({_norm(j) for j in judges.values()}) != 1:
        items.append(_item(4, FAIL, f"judge differs across arms: {judges}"))
    elif any(_norm(judges[n]) == readers[n] for n in names):
        items.append(_item(4, FAIL, "judge model equals reader model"))
    else:
        items.append(_item(4, PASS, f"judge {next(iter(judges.values()))} fixed; reader differs"))

    # (5) dataset digest verified.
    ds = {n: (receipts[n] or {}).get("dataset") or {} for n in names}
    if any(not d.get("path") for d in ds.values()):
        items.append(_item(5, UNKNOWN, "no dataset path/sha256 receipt (checked_against_dataset is "
                           "not a digest)"))
    else:
        problems = []
        for n, d in ds.items():
            actual = hash_file(Path(d["path"])) if Path(d["path"]).is_file() else None
            if actual is None:
                problems.append(f"{n}: file missing")
            elif actual != expected_dataset_sha or d.get("sha256") not in (None, actual):
                problems.append(f"{n}: digest mismatch")
        items.append(_item(5, FAIL if problems else PASS, "; ".join(problems) or "digest verified"))

    # (6) identical retrieval budget and chunking for control and arm under test.
    ing = {n: (receipts[n] or {}).get("ingest") or {} for n in names if n in FLOOR_ARMS}
    if len(ing) < 2:
        items.append(_item(6, UNKNOWN, "needs both control (rag) and arm (trace) receipts"))
    else:
        keys = {n: (i.get("retrieval_budget"), i.get("chunking")) for n, i in ing.items()}
        if any(None in k for k in keys.values()):
            items.append(_item(6, UNKNOWN, f"incomplete budget/chunking receipts: {keys}"))
        else:
            items.append(_item(6, PASS if len(set(keys.values())) == 1 else FAIL, f"{keys}"))

    # (7) prompt-matched arms including the closed-book instruction.
    rd = {n: (receipts[n] or {}).get("reader") or {} for n in names}
    if any("prompt_sha256" not in r or "closed_book_instruction" not in r for r in rd.values()):
        items.append(_item(7, UNKNOWN, "no reader.prompt_sha256 / closed_book_instruction receipt"))
    else:
        same = len({(r["prompt_sha256"], r["closed_book_instruction"]) for r in rd.values()}) == 1
        items.append(_item(7, PASS if same else FAIL,
                           "reader prompt and closed-book instruction identical" if same
                           else f"arms differ: {[(n, r['prompt_sha256'][:8], r['closed_book_instruction']) for n, r in rd.items()]}"))

    # (8) floors: far below = configuration defect, never a result.
    far = [n for n in names if n in FLOOR_ARMS
           and (summaries[n].get("headline") or 0) < BEAM_FOLD_FLOOR - FLOOR_MARGIN]
    items.append(_item(8, FAIL if far else PASS,
                       f"retrieval arms far below the {BEAM_FOLD_FLOOR} floor: {far} -> configuration defect"
                       if far else f"no retrieval arm more than {FLOOR_MARGIN} under the {BEAM_FOLD_FLOOR} sanity floor"))

    return {"schema": "epyc.beam_prerun_checklist.v1", "arms": names, "items": items,
            "cleared": all(i["status"] == PASS for i in items)}


def load_arm(run_dir: Path, arm: str) -> dict[str, Any]:
    scored = sorted(run_dir.glob("scored-beam-*.json"))
    summary = json.loads(scored[0].read_text())["summary"] if scored else {}
    rp = run_dir / "beam_run_receipt.json"
    return {"arm": arm, "summary": summary,
            "receipt": json.loads(rp.read_text()) if rp.is_file() else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BEAM pre-run checklist (M-12c)")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--min-max-tokens", type=int, default=DEFAULT_MIN_MAX_TOKENS)
    args = ap.parse_args(argv)
    arms = [load_arm(Path(spec.split("=", 1)[1]), spec.split("=", 1)[0]) for spec in args.arm]
    rep = check_arms(arms, min_max_tokens=args.min_max_tokens)
    print(json.dumps(rep, indent=2))
    return 0 if rep["cleared"] else 1


if __name__ == "__main__":
    sys.exit(main())
