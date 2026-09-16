#!/usr/bin/env python3
"""Fold a judged BEAM run offline and (optionally) emit its belief-kernel sidecar.

It performs NO inference. The input is a JUDGED artifact — per-question nugget
verdicts already produced by the served judge — and this module only applies
the fold (``beam_scoring.fold_beam``) and writes the result. Nothing here opens
a socket or loads a model.

Input (JSON)::

    {
      "run_id": "...", "model_role": "...", "split": "100K",
      "judge_model": "...", "judge_prompt_version": "beam-unified-b2da22ea+question/v1",
      "records": [
        {"question_id": "beam_100K_1_abstention_0", "ability": "abstention",
         "nugget_verdicts": [1.0]},
        {"question_id": "...", "ability": "event_ordering",
         "nugget_verdicts": [0.5, 1.0, 0.0], "tau_norm": 0.75},
        ...
      ]
    }

Output: ``{"summary": {...}, "per_question": [...]}``. The summary carries the
BEAM-fold headline AND the secondary diagnostics (rubric-item micro-average,
binarised pass count) side by side, plus the judge identity — four judges have
put BEAM on four different axes, so a headline without its judge is not a
comparable number.

Belief-kernel write side (SC68): ``--belief-measurements`` emits a
producer-authored ``belief_measurements.jsonl`` beside ``--out-json`` through
``epyc-root:scripts/vidya/adapters/beam_memory_capture.py``. It requires
``--arm`` because the judged artifact does not record it and a tuple that guesses
the arm claims warrant the run never captured.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

from beam_scoring import (
    FOLD_VERSION,
    JUDGE_PROMPT_VERSION,
    QUESTION_IN_JUDGE_PROMPT,
    BEAMFoldError,
    fold_beam,
)


def build_prompt_index(adapter) -> dict[str, dict]:
    """Prompt dicts by id, so a judged artifact can be checked against the dataset."""
    return {p["id"]: p for p in adapter.extract_all()}


def score_judged_payload(payload: Mapping[str, Any],
                         prompt_index: Mapping[str, dict] | None = None) -> dict:
    """Fold one judged payload. With ``prompt_index``, each record must match its prompt.

    The check refuses a record whose ability or nugget count disagrees with the
    dataset — a verdict list shorter than the rubric would silently re-weight the
    per-question mean.
    """
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise BEAMFoldError("judged payload carries no records")
    unjudged = payload.get("unjudged") or []
    if unjudged:
        # judge_beam_run.py lists questions it could not judge; folding without them
        # would silently re-weight their abilities. Re-judge first.
        reasons = sorted({str(u.get("reason")) for u in unjudged if isinstance(u, Mapping)})
        raise BEAMFoldError(f"{len(unjudged)} question(s) were never judged ({reasons}); "
                            "re-run judge_beam_run.py before folding")
    unknown_ids = 0
    if prompt_index is not None:
        for record in records:
            prompt = prompt_index.get(record.get("question_id"))
            if prompt is None:
                unknown_ids += 1
                continue
            cfg = prompt["scoring_config"]
            if record.get("ability") != cfg["ability"]:
                raise BEAMFoldError(f"{record['question_id']}: ability {record.get('ability')!r} "
                                    f"!= dataset {cfg['ability']!r}")
            if len(record.get("nugget_verdicts") or []) != len(cfg["nuggets"]):
                raise BEAMFoldError(f"{record['question_id']}: {len(record.get('nugget_verdicts') or [])} "
                                    f"verdicts for {len(cfg['nuggets'])} nuggets")
        if unknown_ids:
            raise BEAMFoldError(f"{unknown_ids} record(s) name question ids the dataset lacks")

    folded = fold_beam(records)
    per_question = folded.pop("per_question")
    summary = {
        "scorer_version": FOLD_VERSION,
        "run_id": payload.get("run_id"),
        "model_role": payload.get("model_role"),
        "split": payload.get("split"),
        "judge_model": payload.get("judge_model"),
        "judge_prompt_version": payload.get("judge_prompt_version") or JUDGE_PROMPT_VERSION,
        "question_in_judge_prompt": payload.get("question_in_judge_prompt",
                                                QUESTION_IN_JUDGE_PROMPT),
        "checked_against_dataset": prompt_index is not None,
        # M-12 B2: the arm each judged row was produced under, as the harness recorded it.
        "context_mode_by_row": dict(payload.get("context_mode_by_row") or {"unrecorded": 0}),
        **folded,
    }
    return {"summary": summary, "per_question": per_question}


_ROOT_CANDIDATES = (
    os.environ.get("EPYC_ROOT", ""),
    "/mnt/raid0/llm/epyc-root",
    "/workspace",
)


def _load_belief_capture():
    """Import ``beam_memory_capture`` from epyc-root, or explain why it is unavailable."""
    tried = []
    for root in _ROOT_CANDIDATES:
        if not root:
            continue
        module = Path(root) / "scripts" / "vidya" / "adapters" / "beam_memory_capture.py"
        tried.append(str(module))
        if not module.is_file():
            continue
        spec = importlib.util.spec_from_file_location("beam_memory_capture", module)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            continue
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    raise SystemExit("--belief-measurements needs epyc-root's beam_memory_capture.py "
                     "(set EPYC_ROOT). Looked in: " + ", ".join(tried))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judged", type=Path, help="Judged BEAM run JSON (per-nugget verdicts)")
    parser.add_argument("--out-json", type=Path, default=None, help="Write the folded report")
    parser.add_argument("--check-dataset", action="store_true",
                        help="Check every record against the local BEAM dataset")
    parser.add_argument("--data-dir", type=Path, default=None)
    belief = parser.add_argument_group("belief kernel (SC68)")
    belief.add_argument("--belief-measurements", action="store_true",
                        help="Write belief_measurements.jsonl beside --out-json")
    belief.add_argument("--arm", default=None,
                        help="M-12b arm: full (Vanilla, memory-off) | rag (pair_chunk BM25 "
                             "control) | trace (arm under test)")
    belief.add_argument("--run-id", default=None, help="Run id override")
    belief.add_argument("--category", default="CANDIDATE",
                        choices=("OPTIMUM", "BASELINE", "CANDIDATE"))
    args = parser.parse_args()

    payload = json.loads(args.judged.read_text())
    prompt_index = None
    if args.check_dataset:
        from long_context_adapters import BEAMAdapter
        adapter = BEAMAdapter(data_dir=args.data_dir, split=payload.get("split") or "100K",
                              context_mode="full")
        prompt_index = build_prompt_index(adapter)
        if not prompt_index:
            raise SystemExit(f"--check-dataset: no BEAM data loaded "
                             f"({adapter.accounting_summary()['degraded_sources']})")
    scored = score_judged_payload(payload, prompt_index)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(scored, indent=2) + "\n")

    if args.belief_measurements:
        if not args.out_json:
            raise SystemExit("--belief-measurements requires --out-json: the sidecar attests "
                             "the scored artifact, so it must be durable first")
        if not args.arm:
            raise SystemExit("--belief-measurements requires --arm; the judged artifact does "
                             "not record it and nothing may be guessed on read")
        # B2: the rows record which arm produced them; --arm may only restate it.
        seen = {k for k, v in scored["summary"]["context_mode_by_row"].items() if v}
        if seen != {args.arm}:
            raise SystemExit(
                f"--arm {args.arm!r} disagrees with the arm the result rows record "
                f"({scored['summary']['context_mode_by_row']}); refusing to emit belief rows")
        capture = _load_belief_capture()
        run_id = args.run_id or scored["summary"].get("run_id")
        if not run_id:
            raise SystemExit("--belief-measurements needs a run id (--run-id)")
        sidecar = capture.write_belief_measurements(
            args.out_json, summary=scored["summary"], run_id=str(run_id),
            producer="score_beam_run.py", arm=args.arm, category=args.category)
        print(f"belief sidecar: {sidecar}")

    print(json.dumps(scored["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
