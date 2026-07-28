#!/usr/bin/env python3
"""Build the pinned co-critic task set for the Phase-3 shadow bake-off (P3-1).

Zero-inference, deterministic construction (PROPOSAL implementation -- see
docs/design/p3-shadow-bakeoff-spec.md section "Co-critic corpus construction
(PROPOSAL)").  Candidates are mined from BANKED per-question capture
artifacts of the 2026-07-24 architect code-eval campaign: LCB-hard responses
from arms A3/A4/A1, each carrying an executable code-execution-oracle label
(``correct``) recorded at capture time.  Nothing here contacts a model.

Task shape (constraint 3): [candidate solution + typed review request] ->
the model under test emits a typed verdict in the reviewer-control-plane
ReviewDecision shape (review_decision.schema.json vocabulary).  Gold =
``known_correct`` / ``known_wrong`` from the executable oracle.

The output is runner-compatible: a JSON list of question rows consumable by
``v7_quality_gate_runner.py --questions-in`` (suite ``p3_cocritic_v1``).
Inline runner scoring of these rows is JUNK BY DESIGN (like swebench_oracle
``__patch__`` rows); the authoritative scorer is the deterministic replay
scorer ``p3_bakeoff_critic_score.py``.

The critic set is SWAPPABLE: any file matching this schema can be pinned
into the bake-off manifest; the runner and scorer are generic over it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_bakeoff_common import (  # noqa: E402
    CRITIC_SUITE,
    CRITIC_TASKS_SCHEMA_VERSION,
    GOLD_CORRECT,
    GOLD_WRONG,
    load_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)

RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
CODE_EVAL_ART = RESEARCH_ROOT / "artifacts/architect-code-eval-20260724"
DEFAULT_SOURCES = (
    CODE_EVAL_ART / "lcb_A3_27b_dense/pq.jsonl",
    CODE_EVAL_ART / "lcb_A4_35b_a3b/pq.jsonl",
    CODE_EVAL_ART / "lcb_A1_122b_iq2/pq.jsonl",
)
DEFAULT_QUESTIONS = CODE_EVAL_ART / "questions_livecodebench_hard.json"

REVIEW_REQUEST_TEMPLATE = """You are a code reviewer acting as a judicial gate: you review a candidate solution against its task, you do NOT author or rewrite code.

## Task given to the author
{task_prompt}

## Candidate solution under review
{candidate}

## Review request (typed decision)
Respond with ONLY one JSON object, no prose before or after, with exactly these fields:
{{"decision": "<verdict>", "confidence": <number 0..1>, "blocking": {{"tripwire": <true|false>}}}}
where "decision" is exactly one of:
- "approve": the candidate correctly and completely solves the task as specified.
- "request_changes": the candidate is defective (wrong logic, wrong I/O contract, fails plausible inputs) but fixable.
- "reject": the candidate is defective and not worth iterating on.
- "reject_to_empty": the candidate is worse than no answer at all.
- "request_evidence": you cannot decide without running tests or seeing more evidence.
- "abstain": the package is insufficient and no permitted evidence request would resolve it.
- "escalate": defer to a higher review tier.
Set "blocking.tripwire" to true ONLY if a hard invariant is violated (e.g. the program does not read stdin / write stdout as the task requires, or cannot run at all).
"confidence" is how sure you are the DECISION itself is right -- not how good the candidate is."""


def eligible(row: dict) -> bool:
    """Keep only clean, committed model outputs as review candidates.

    Truncated (length-capped) and empty rows are excluded: their gold label
    would reflect a token budget, not solution quality, and an empty
    candidate is trivially rejectable (degenerate discrimination).
    Request-error rows are transport artifacts, never candidates.
    """
    return bool(
        row.get("suite") == "livecodebench_hard"
        and row.get("response")
        and row.get("finish_reason") == "stop"
        and not row.get("request_error")
        and not row.get("truncated")
        and not row.get("empty_response")
    )


def mine_candidates(sources: list[Path]) -> tuple[list[dict], list[dict]]:
    """Return (candidates, provenance).  Deterministic; dedupes by response hash."""
    seen: set[str] = set()
    candidates: list[dict] = []
    provenance: list[dict] = []
    for src in sources:
        digest = sha256_file(src)
        provenance.append({"path": str(src), "sha256": digest})
        for row in load_jsonl(src):
            if not eligible(row):
                continue
            rhash = sha256_text(row["response"])
            if rhash in seen:
                continue
            seen.add(rhash)
            candidates.append(
                {
                    "question_id": row["id"],
                    "source_arm": row.get("arm", "unknown"),
                    "source_path": str(src),
                    "source_sha256": digest,
                    "response": row["response"],
                    "response_sha256": rhash,
                    "gold_label": GOLD_CORRECT if row.get("correct") else GOLD_WRONG,
                    "completion_tokens": row.get("completion_tokens", 0),
                }
            )
    return candidates, provenance


def balance_select(candidates: list[dict], per_class: int, seed: int) -> list[dict]:
    """Deterministic class-balanced, question-stratified selection.

    Round-robin across question ids within each gold class so no single
    question dominates the corpus; order and tie-breaks fixed by ``seed``.
    """
    rng = random.Random(seed)
    by_class: dict[str, dict[str, list[dict]]] = {GOLD_CORRECT: {}, GOLD_WRONG: {}}
    for cand in sorted(candidates, key=lambda c: (c["question_id"], c["response_sha256"])):
        by_class[cand["gold_label"]].setdefault(cand["question_id"], []).append(cand)
    selected: list[dict] = []
    for label in (GOLD_CORRECT, GOLD_WRONG):
        queues = by_class[label]
        qids = sorted(queues)
        rng.shuffle(qids)
        for qid in qids:
            rng.shuffle(queues[qid])
        picked = 0
        while picked < per_class:
            progressed = False
            for qid in qids:
                if picked >= per_class:
                    break
                if queues[qid]:
                    selected.append(queues[qid].pop(0))
                    picked += 1
                    progressed = True
            if not progressed:
                break  # class exhausted below target -- recorded, not fatal
    return selected


def build_tasks(selected: list[dict], questions_by_id: dict[str, dict]) -> list[dict]:
    tasks = []
    for cand in selected:
        question = questions_by_id[cand["question_id"]]
        prompt = REVIEW_REQUEST_TEMPLATE.format(
            task_prompt=question["prompt"], candidate=cand["response"]
        )
        task_id = (
            f"crit_{cand['question_id']}_{cand['source_arm']}"
            f"_{cand['response_sha256'][:8]}"
        )
        tasks.append(
            {
                "id": task_id,
                "suite": CRITIC_SUITE,
                "tier": 2,
                "prompt": prompt,
                # Sentinel expected (never sent to the model; runner requires
                # non-empty).  Inline runner scoring is junk by design; the
                # replay scorer is authoritative (swebench __patch__ pattern).
                "expected": "__typed_verdict__",
                "scoring_method": "exact_match",
                "scoring_config": {
                    "deferred_scorer": "p3_bakeoff_critic_score.py",
                    "gold_label": cand["gold_label"],
                    "gold_source": "code_execution_oracle@capture",
                    "provenance": {
                        "question_id": cand["question_id"],
                        "source_arm": cand["source_arm"],
                        "source_path": cand["source_path"],
                        "source_sha256": cand["source_sha256"],
                        "response_sha256": cand["response_sha256"],
                        "candidate_completion_tokens": cand["completion_tokens"],
                    },
                },
            }
        )
    # Deterministic interleave of classes by id hash (stable, seed-free).
    tasks.sort(key=lambda t: sha256_text(t["id"]))
    return tasks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sources", nargs="+", type=Path,
                   default=[Path(s) for s in DEFAULT_SOURCES],
                   help="Banked per-question capture JSONLs to mine")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
                   help="Pinned question file supplying original task prompts")
    p.add_argument("--per-class", type=int, default=60,
                   help="Target candidates per gold class (default: 60)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True,
                   help="Output critic tasks JSON path")
    args = p.parse_args()

    questions = json.loads(args.questions.read_text())
    questions_by_id = {q["id"]: q for q in questions}
    candidates, provenance = mine_candidates(list(args.sources))
    missing = sorted(
        {c["question_id"] for c in candidates} - set(questions_by_id)
    )
    if missing:
        print(f"[critic-build] FATAL: candidate question ids missing from "
              f"question file: {missing}", file=sys.stderr)
        return 1
    selected = balance_select(candidates, args.per_class, args.seed)
    tasks = build_tasks(selected, questions_by_id)
    n_correct = sum(
        1 for t in tasks
        if t["scoring_config"]["gold_label"] == GOLD_CORRECT
    )
    payload = {
        "schema_version": CRITIC_TASKS_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite": CRITIC_SUITE,
        "builder": {
            "source": str(Path(__file__).name),
            "source_sha256": sha256_file(Path(__file__)),
            "seed": args.seed,
            "per_class_target": args.per_class,
        },
        "sources": provenance,
        "questions_file": {
            "path": str(args.questions),
            "sha256": sha256_file(args.questions),
        },
        "n_tasks": len(tasks),
        "prevalence": {
            GOLD_CORRECT: n_correct,
            GOLD_WRONG: len(tasks) - n_correct,
        },
        "candidate_pool": {
            "mined": len(candidates),
            "eligible_note": "finish_reason=stop, non-empty, no request_error, deduped",
        },
        # Runner-compatible shape: v7_quality_gate_runner.load_questions()
        # replays pinned files via pinned["suites"][suite_name].
        "suites": {CRITIC_SUITE: tasks},
    }
    digest = write_json(args.output, payload, sort_keys=False)
    print(f"[critic-build] {len(tasks)} tasks "
          f"({n_correct} {GOLD_CORRECT} / {len(tasks) - n_correct} {GOLD_WRONG}) "
          f"-> {args.output} sha256={digest[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
