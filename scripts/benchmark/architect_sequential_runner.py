#!/usr/bin/env python3
"""Interleaved, paired sequential evaluator for future architect suites.

The existing quality-gate runner intentionally evaluates one arm at a time.
That is appropriate for already-pinned historical campaigns, but cannot make
an early stopping decision: the second arm has not answered the same question
yet.  This runner instead issues both arms for each question before advancing
to the next (easier) question, and records the paired e-process state after
every complete pair.

It is deliberately fail-closed.  A live run requires a pre-pinned manifest,
an a-priori numeric difficulty key on every item, explicit saturation
parameters, and exactly two named arms.  It imports the established
``EProcessState`` / ``SequentialPolicy`` implementation from the orchestrator;
the statistical primitive is not copied or re-parameterised here.

This program performs inference only when invoked without ``--dry-run``.
During E5 Stage-B it may be syntax-checked and dry-run planned, but must not
be used against an endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from v7_quality_gate_runner import query_server_meta, score_response  # noqa: E402


CAPTURE_SCHEMA_VERSION = "architect_interleaved_sequential.v1"


def _load_sequential_primitives():
    """Import the one authoritative e-process implementation.

    The explicit path is an operational dependency, rather than silently
    falling back to a local statistical copy if the orchestrator checkout is
    missing.  A different implementation would invalidate the claimed reuse.
    """
    root = Path(
        os.environ.get("EPYC_ORCHESTRATOR_ROOT", "/mnt/raid0/llm/epyc-orchestrator")
    ).resolve()
    if not (root / "src" / "autopilot_core" / "sequential_verdict.py").is_file():
        raise RuntimeError(
            "authoritative sequential_verdict.py not found; set "
            "EPYC_ORCHESTRATOR_ROOT to the orchestrator checkout"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.autopilot_core.sequential_verdict import (  # pylint: disable=import-outside-toplevel
        DEFAULT_POLICY,
        STATE_CONFIRMED,
        STATE_REFUTED,
        EProcessState,
        SequentialPolicy,
    )
    return DEFAULT_POLICY, STATE_CONFIRMED, STATE_REFUTED, EProcessState, SequentialPolicy


(
    DEFAULT_POLICY,
    STATE_CONFIRMED,
    STATE_REFUTED,
    EProcessState,
    SequentialPolicy,
) = _load_sequential_primitives()


@dataclass(frozen=True)
class Arm:
    """One named endpoint in a paired comparison."""

    label: str
    url: str


@dataclass(frozen=True)
class SaturationPolicy:
    """Operator-declared ceiling rule; no hidden near-ceiling constants."""

    min_items: int
    min_accuracy: float

    def __post_init__(self) -> None:
        if self.min_items < 1:
            raise ValueError("saturation min_items must be at least 1")
        if not 0.0 <= self.min_accuracy <= 1.0:
            raise ValueError("saturation min_accuracy must be in [0, 1]")


def _difficulty_value(question: dict[str, Any], field: str) -> float:
    """Return a numeric, model-independent ordering value or fail closed."""
    if field not in question:
        raise ValueError(
            f"question {question.get('id', '<missing-id>')!r} lacks required "
            f"a-priori difficulty field {field!r}"
        )
    raw = question[field]
    if isinstance(raw, bool):
        raise ValueError(f"difficulty key {field!r} must be numeric, not bool")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"difficulty key {field!r} must be numeric: {raw!r}") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"difficulty key {field!r} must be finite: {raw!r}")
    return value


def order_questions(
    questions: Sequence[dict[str, Any]], *, difficulty_field: str,
) -> list[dict[str, Any]]:
    """Order hardest first, with deterministic ID tie-breaking.

    The caller supplies the already-pinned manifest.  This function only reads
    the manifest's declared a-priori key; it never derives an order from either
    arm's outcomes.
    """
    normalized: list[tuple[float, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for question in questions:
        qid = str(question.get("id", "")).strip()
        if not qid:
            raise ValueError("every pinned question needs a non-empty id")
        if qid in seen_ids:
            raise ValueError(f"duplicate pinned question id: {qid}")
        seen_ids.add(qid)
        normalized.append((_difficulty_value(question, difficulty_field), qid, question))
    return [question for _, _, question in sorted(normalized, key=lambda item: (-item[0], item[1]))]


def parse_arm(value: str) -> Arm:
    """Parse ``LABEL=URL`` without accepting ambiguous arm labels."""
    label, sep, url = value.partition("=")
    if not sep or not label.strip() or not url.strip():
        raise argparse.ArgumentTypeError("--arm must be LABEL=URL")
    return Arm(label.strip(), url.rstrip("/"))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _pair_stop_reason(candidate_state, baseline_state, *, policy: Any) -> str | None:
    if candidate_state.state_name(policy) == STATE_CONFIRMED:
        return "separation:candidate"
    if baseline_state.state_name(policy) == STATE_CONFIRMED:
        return "separation:baseline"
    if (
        candidate_state.state_name(policy) == STATE_REFUTED
        and baseline_state.state_name(policy) == STATE_REFUTED
    ):
        return "futility"
    return None


def run_interleaved(
    *,
    suite: str,
    questions: Sequence[dict[str, Any]],
    arms: Sequence[Arm],
    baseline_arm: str,
    candidate_arm: str,
    difficulty_field: str,
    saturation: SaturationPolicy,
    output: Path,
    capture_out: Path,
    seed: int,
    max_tokens: int,
    temperature: float,
    endpoint: str = "chat",
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
    policy: Any = DEFAULT_POLICY,
    query: Callable[..., dict[str, Any]] = query_server_meta,
) -> dict[str, Any]:
    """Run exactly two arms per question and atomically preserve each pair.

    Transport failures are captured but do not update either e-process or make
    an early decision.  That avoids treating a missing arm answer as evidence.
    """
    by_label = {arm.label: arm for arm in arms}
    if len(arms) != 2 or len(by_label) != 2:
        raise ValueError("exactly two uniquely labelled --arm values are required")
    if baseline_arm not in by_label or candidate_arm not in by_label:
        raise ValueError("--baseline-arm and --candidate-arm must name supplied arms")
    if baseline_arm == candidate_arm:
        raise ValueError("baseline and candidate arms must differ")
    if capture_out.exists() and capture_out.stat().st_size:
        raise ValueError("capture output already exists; refuse an ambiguous partial resume")

    ordered = order_questions(questions, difficulty_field=difficulty_field)
    candidate_state = EProcessState()
    baseline_state = EProcessState()
    tier_counts: dict[float, dict[str, dict[str, int]]] = {}
    pairs: list[dict[str, Any]] = []
    stop_reason: str | None = None

    result: dict[str, Any] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "meta": {
            "suite": suite,
            "arms": [asdict(arm) for arm in arms],
            "baseline_arm": baseline_arm,
            "candidate_arm": candidate_arm,
            "difficulty_field": difficulty_field,
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "endpoint": endpoint,
            "top_p": top_p,
            "top_k": top_k,
            "enable_thinking": enable_thinking,
            "sequential_policy": asdict(policy),
            "saturation_policy": asdict(saturation),
        },
        "pairs": pairs,
        "stop_reason": None,
        "provisional_transport_pairs": 0,
    }
    capture_out.parent.mkdir(parents=True, exist_ok=True)
    with capture_out.open("x", encoding="utf-8") as capture:
        for index, question in enumerate(ordered):
            qid = str(question["id"])
            difficulty = _difficulty_value(question, difficulty_field)
            expected = str(question.get("expected", "")).strip()
            if not expected:
                raise ValueError(f"question {qid} has no expected answer")
            arm_rows: dict[str, dict[str, Any]] = {}
            for arm in arms:
                meta = query(
                    arm.url, question["prompt"], max_tokens=max_tokens,
                    temperature=temperature, seed=seed, endpoint=endpoint,
                    top_p=top_p, top_k=top_k, enable_thinking=enable_thinking,
                )
                response = str(meta.get("text") or "")
                error = str(meta.get("error") or "")
                arm_rows[arm.label] = {
                    "arm": arm.label,
                    "url": arm.url,
                    "correct": bool(response) and score_response(response, expected, question),
                    "response": response,
                    "finish_reason": str(meta.get("finish_reason") or ""),
                    "request_error": error,
                }

            complete = not any(row["request_error"] for row in arm_rows.values())
            pair = {
                "sequence_index": index,
                "suite": suite,
                "id": qid,
                "difficulty_key": difficulty,
                "arms": arm_rows,
                "paired_complete": complete,
            }
            if complete:
                candidate_correct = bool(arm_rows[candidate_arm]["correct"])
                baseline_correct = bool(arm_rows[baseline_arm]["correct"])
                delta = int(candidate_correct) - int(baseline_correct)
                candidate_state, candidate_update = candidate_state.update(delta, policy=policy)
                baseline_state, baseline_update = baseline_state.update(-delta, policy=policy)
                pair["sequential"] = {
                    "candidate": {"z": delta, "update": asdict(candidate_update)},
                    "baseline": {"z": -delta, "update": asdict(baseline_update)},
                }
                stop_reason = _pair_stop_reason(candidate_state, baseline_state, policy=policy)
            else:
                result["provisional_transport_pairs"] += 1
                pair["sequential"] = {"state": "not_updated_transport_failure"}

            tier = tier_counts.setdefault(
                difficulty, {arm.label: {"n": 0, "correct": 0} for arm in arms}
            )
            for label, row in arm_rows.items():
                tier[label]["n"] += 1
                tier[label]["correct"] += int(bool(row["correct"]))
                capture.write(json.dumps({
                    "schema_version": CAPTURE_SCHEMA_VERSION,
                    "suite": suite,
                    "id": qid,
                    "sequence_index": index,
                    "difficulty_key": difficulty,
                    **row,
                    "paired_complete": complete,
                }) + "\n")
            capture.flush()
            os.fsync(capture.fileno())
            pairs.append(pair)

            at_tier_boundary = (
                index == len(ordered) - 1
                or _difficulty_value(ordered[index + 1], difficulty_field) != difficulty
            )
            if stop_reason is None and at_tier_boundary:
                saturated = all(
                    stats["n"] >= saturation.min_items
                    and stats["correct"] / stats["n"] >= saturation.min_accuracy
                    for stats in tier.values()
                )
                if saturated and index < len(ordered) - 1:
                    stop_reason = f"saturation:difficulty_key={difficulty:g}"
            result["candidate_eprocess"] = {
                "wealth": candidate_state.wealth,
                "k": candidate_state.k,
                "state": candidate_state.state_name(policy),
            }
            result["baseline_eprocess"] = {
                "wealth": baseline_state.wealth,
                "k": baseline_state.k,
                "state": baseline_state.state_name(policy),
            }
            result["stop_reason"] = stop_reason
            _atomic_write(output, result)
            if stop_reason is not None:
                break
    return result


def _load_pinned_questions(path: Path, suite: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    suites = payload.get("suites") if isinstance(payload, dict) else None
    if not isinstance(suites, dict) or not isinstance(suites.get(suite), list):
        raise ValueError("--questions-in must contain a suites[SUITE] pinned list")
    return suites[suite]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-in", required=True, type=Path,
                        help="Pinned manifest containing suites[SUITE]")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True,
                        help="Exactly twice: LABEL=URL")
    parser.add_argument("--baseline-arm", required=True)
    parser.add_argument("--candidate-arm", required=True)
    parser.add_argument("--difficulty-field", default="difficulty_key")
    parser.add_argument("--saturation-min-items", required=True, type=int)
    parser.add_argument("--saturation-min-accuracy", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--endpoint", choices=("chat", "completion"), default="chat")
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None)
    parser.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capture-out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the planned hardest-first order without endpoints")
    args = parser.parse_args()
    questions = _load_pinned_questions(args.questions_in, args.suite)
    saturation = SaturationPolicy(args.saturation_min_items, args.saturation_min_accuracy)
    ordered = order_questions(questions, difficulty_field=args.difficulty_field)
    if args.dry_run:
        print(json.dumps({
            "suite": args.suite,
            "difficulty_field": args.difficulty_field,
            "ordered_ids": [str(question["id"]) for question in ordered],
            "arms": [asdict(arm) for arm in args.arm],
            "saturation_policy": asdict(saturation),
        }, indent=2))
        return 0
    run_interleaved(
        suite=args.suite, questions=ordered, arms=args.arm,
        baseline_arm=args.baseline_arm, candidate_arm=args.candidate_arm,
        difficulty_field=args.difficulty_field, saturation=saturation,
        output=args.output, capture_out=args.capture_out, seed=args.seed,
        max_tokens=args.max_tokens, temperature=args.temperature,
        endpoint=args.endpoint, top_p=args.top_p, top_k=args.top_k,
        enable_thinking=args.enable_thinking,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
