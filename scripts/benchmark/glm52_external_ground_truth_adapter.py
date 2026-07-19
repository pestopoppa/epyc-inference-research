#!/usr/bin/env python3
"""No-inference adapter for GLM-5.2 external reviewer ground-truth gates.

This module normalizes judge/reviewer datasets already present on disk into one
pairwise row contract. It intentionally does not launch GLM, Docker, or any
server. Live execution can consume the normalized rows later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

SCHEMA = "glm52_external_ground_truth_adapter.v1"
PAIRWISE_DECISIONS = ("A", "B")
DEFAULT_MAX_FIELD_CHARS = 24000
DEFAULT_CONTEXT_LENGTH = 12000
DEFAULT_COMPLETION_TOKENS = 128
DEFAULT_PROMPT_GUARD_TOKENS = 256


@dataclass(frozen=True)
class PairwiseRow:
    row_id: str
    task_kind: str
    task: str
    candidate: str
    candidate_b: str
    gold_label: str
    gold_source: str
    gold_instrument_version: str
    source_benchmark: str
    source_suite: str
    source_row_id: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_confidence"] = "external_ground_truth"
        payload["domain"] = "judge_quality"
        payload["candidate_a"] = payload["candidate"]
        return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_row_id(source_benchmark: str, source_suite: str, source_row_id: str, *parts: str) -> str:
    key = "\x00".join([source_benchmark, source_suite, source_row_id, *parts])
    return f"glm52-ext:{source_benchmark}:{source_suite}:{hashlib.sha1(key.encode('utf-8')).hexdigest()[:20]}"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def require_text(row: dict[str, Any], key: str, *, source: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: missing non-empty string field {key!r}")
    return value


def source_version(path: Path) -> str:
    return f"local-file-sha256:{sha256_file(path)}"


def normalize_judgebench_label(label: Any) -> str | None:
    if label == "A>B":
        return "A"
    if label == "B>A":
        return "B"
    return None


def load_judgebench(path: Path, *, suite: str) -> list[PairwiseRow]:
    version = source_version(path)
    rows: list[PairwiseRow] = []
    for raw in read_jsonl(path):
        gold = normalize_judgebench_label(raw.get("label"))
        if gold is None:
            continue
        source_row_id = str(raw.get("pair_id") or raw.get("original_id") or len(rows))
        task = require_text(raw, "question", source="judgebench")
        cand_a = require_text(raw, "response_A", source="judgebench")
        cand_b = require_text(raw, "response_B", source="judgebench")
        rows.append(
            PairwiseRow(
                row_id=stable_row_id("judgebench", suite, source_row_id, task, cand_a, cand_b),
                task_kind="pairwise",
                task=task,
                candidate=cand_a,
                candidate_b=cand_b,
                gold_label=gold,
                gold_source="judgebench",
                gold_instrument_version=version,
                source_benchmark="judgebench",
                source_suite=suite,
                source_row_id=source_row_id,
                provenance={
                    "scoring_method": "exact_match",
                    "native_label": raw.get("label"),
                    "source": raw.get("source"),
                    "response_model": raw.get("response_model"),
                },
            )
        )
    return rows


def normalize_llmbar_label(label: Any) -> str | None:
    if label in (1, "1", "A", "a"):
        return "A"
    if label in (2, "2", "B", "b"):
        return "B"
    return None


def load_llmbar(path: Path, *, suite: str) -> list[PairwiseRow]:
    version = source_version(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"LLMBar file must contain a JSON list: {path}")
    rows: list[PairwiseRow] = []
    for idx, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{idx}: expected JSON object")
        gold = normalize_llmbar_label(raw.get("label"))
        if gold is None:
            continue
        source_row_id = str(raw.get("id") or idx)
        task = require_text(raw, "input", source="llmbar")
        cand_a = require_text(raw, "output_1", source="llmbar")
        cand_b = require_text(raw, "output_2", source="llmbar")
        rows.append(
            PairwiseRow(
                row_id=stable_row_id("llmbar", suite, source_row_id, task, cand_a, cand_b),
                task_kind="pairwise",
                task=task,
                candidate=cand_a,
                candidate_b=cand_b,
                gold_label=gold,
                gold_source="llmbar",
                gold_instrument_version=version,
                source_benchmark="llmbar",
                source_suite=suite,
                source_row_id=source_row_id,
                provenance={"scoring_method": "exact_match", "native_label": raw.get("label")},
            )
        )
    return rows


def judge_score(raw_score: Any, key: str) -> float | None:
    if not isinstance(raw_score, dict):
        return None
    value = raw_score.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def load_judgelm(path: Path, *, suite: str = "val_5k", score_key: str = "rougeLsum") -> list[PairwiseRow]:
    version = source_version(path)
    rows: list[PairwiseRow] = []
    for raw in read_jsonl(path):
        scores = raw.get("score")
        if not isinstance(scores, list) or len(scores) != 2:
            continue
        score_a = judge_score(scores[0], score_key)
        score_b = judge_score(scores[1], score_key)
        if score_a is None or score_b is None or score_a == score_b:
            continue
        gold = "A" if score_a > score_b else "B"
        source_row_id = str(raw.get("question_id") or len(rows))
        try:
            task = require_text(raw, "question_body", source="judgelm")
            cand_a = require_text(raw, "answer1_body", source="judgelm")
            cand_b = require_text(raw, "answer2_body", source="judgelm")
        except ValueError:
            continue
        rows.append(
            PairwiseRow(
                row_id=stable_row_id("judgelm", suite, source_row_id, task, cand_a, cand_b),
                task_kind="pairwise",
                task=task,
                candidate=cand_a,
                candidate_b=cand_b,
                gold_label=gold,
                gold_source="judgelm",
                gold_instrument_version=version,
                source_benchmark="judgelm",
                source_suite=suite,
                source_row_id=source_row_id,
                provenance={
                    "scoring_method": "exact_match",
                    "teacher_score_key": score_key,
                    "score_a": score_a,
                    "score_b": score_b,
                },
            )
        )
    return rows


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet datasets require pandas/pyarrow for adapter materialization; "
            "use `uv run --with pandas --with pyarrow ...` or pass records in tests."
        ) from exc
    return pd.read_parquet(path).to_dict(orient="records")


def load_rewardbench_records(
    records: Iterable[dict[str, Any]],
    *,
    suite: str,
    version: str,
) -> list[PairwiseRow]:
    rows: list[PairwiseRow] = []
    for idx, raw in enumerate(records):
        task = require_text(raw, "prompt", source="rewardbench")
        cand_a = require_text(raw, "chosen", source="rewardbench")
        cand_b = require_text(raw, "rejected", source="rewardbench")
        source_row_id = str(raw.get("id") or idx)
        rows.append(
            PairwiseRow(
                row_id=stable_row_id("reward-bench", suite, source_row_id, task, cand_a, cand_b),
                task_kind="pairwise",
                task=task,
                candidate=cand_a,
                candidate_b=cand_b,
                gold_label="A",
                gold_source="reward-bench",
                gold_instrument_version=version,
                source_benchmark="reward-bench",
                source_suite=str(raw.get("subset") or suite),
                source_row_id=source_row_id,
                provenance={"scoring_method": "exact_match", "native_label": "chosen>rejected"},
            )
        )
    return rows


def load_rewardbench(path: Path, *, suite: str = "filtered") -> list[PairwiseRow]:
    return load_rewardbench_records(read_parquet_records(path), suite=suite, version=source_version(path))


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        return as_string_list(value.tolist())
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def is_tie_record(raw: dict[str, Any]) -> bool:
    if str(raw.get("subset") or "").strip().lower() == "ties":
        return True
    for key in ("tie", "ties", "is_tie"):
        if raw.get(key) is True:
            return True
    label = str(raw.get("label") or raw.get("winner") or "").strip().lower()
    return label in {"tie", "ties", "equal"}


def load_rewardbench2_records(
    records: Iterable[dict[str, Any]],
    *,
    suite: str,
    version: str,
) -> list[PairwiseRow]:
    rows: list[PairwiseRow] = []
    for idx, raw in enumerate(records):
        if is_tie_record(raw):
            continue
        task = require_text(raw, "prompt", source="rewardbench2")
        chosen = as_string_list(raw.get("chosen"))
        rejected = as_string_list(raw.get("rejected"))
        if not chosen or not rejected:
            continue
        base_id = str(raw.get("id") or idx)
        for chosen_idx, cand_a in enumerate(chosen):
            for rejected_idx, cand_b in enumerate(rejected):
                source_row_id = f"{base_id}:{chosen_idx}:{rejected_idx}"
                rows.append(
                    PairwiseRow(
                        row_id=stable_row_id("reward-bench-2", suite, source_row_id, task, cand_a, cand_b),
                        task_kind="pairwise",
                        task=task,
                        candidate=cand_a,
                        candidate_b=cand_b,
                        gold_label="A",
                        gold_source="reward-bench-2",
                        gold_instrument_version=version,
                        source_benchmark="reward-bench-2",
                        source_suite=str(raw.get("subset") or suite),
                        source_row_id=source_row_id,
                        provenance={"scoring_method": "exact_match", "native_label": "chosen>rejected"},
                    )
                )
    return rows


def load_rewardbench2(path: Path, *, suite: str = "test") -> list[PairwiseRow]:
    return load_rewardbench2_records(read_parquet_records(path), suite=suite, version=source_version(path))


def truncate_middle(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    marker = "\n\n[... truncated by GLM external adapter ...]\n\n"
    return text[:keep_head] + marker + text[-keep_tail:], True


def build_pairwise_prompt(row: PairwiseRow | dict[str, Any], *, max_field_chars: int = DEFAULT_MAX_FIELD_CHARS) -> tuple[str, dict[str, Any]]:
    data = row.to_dict() if isinstance(row, PairwiseRow) else row
    task, task_truncated = truncate_middle(str(data.get("task") or ""), max_field_chars)
    cand_a, a_truncated = truncate_middle(str(data.get("candidate") or data.get("candidate_a") or ""), max_field_chars)
    cand_b, b_truncated = truncate_middle(str(data.get("candidate_b") or ""), max_field_chars)
    prompt = (
        "You are judging two candidate responses for the same task. Choose the better answer.\n"
        "Return exactly one JSON object with keys decision and confidence. decision must be A or B.\n\n"
        f"TASK:\n{task}\n\n"
        f"CANDIDATE A:\n{cand_a}\n\n"
        f"CANDIDATE B:\n{cand_b}\n\n"
        "PairwiseDecision JSON only:"
    )
    return prompt, {
        "task_truncated": task_truncated,
        "candidate_a_truncated": a_truncated,
        "candidate_b_truncated": b_truncated,
        "max_field_chars": max_field_chars,
        "prompt_chars": len(prompt),
        "task_chars_original": len(str(data.get("task") or "")),
        "candidate_a_chars_original": len(str(data.get("candidate") or data.get("candidate_a") or "")),
        "candidate_b_chars_original": len(str(data.get("candidate_b") or "")),
    }


def fit_pairwise_prompt_to_budget(
    row: PairwiseRow | dict[str, Any],
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
    prompt_context_guard_tokens: int = DEFAULT_PROMPT_GUARD_TOKENS,
    max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
    token_counter: Callable[[str], int] = lambda text: len(text.split()),
) -> dict[str, Any]:
    max_prompt_tokens = context_length - max_completion_tokens - prompt_context_guard_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("prompt token budget is non-positive")
    attempts: list[dict[str, Any]] = []
    field_chars = max_field_chars
    for _ in range(10):
        prompt, trunc = build_pairwise_prompt(row, max_field_chars=field_chars)
        token_count = token_counter(prompt)
        attempt = dict(trunc)
        attempt["field_chars"] = field_chars
        attempt["prompt_token_count"] = token_count
        attempts.append(attempt)
        if token_count <= max_prompt_tokens:
            return {
                "prompt": prompt,
                "prompt_token_count": token_count,
                "prompt_token_max": max_prompt_tokens,
                "prompt_fit_attempts": attempts,
                "truncation": trunc,
            }
        field_chars = max(512, int(field_chars * 0.65))
    raise ValueError(f"prompt still exceeds budget {max_prompt_tokens}; last={attempts[-1] if attempts else None}")


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_pairwise_decision_text(text: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidate = extract_json_object(text)
    if candidate is None:
        stripped = text.strip().upper()
        if stripped in PAIRWISE_DECISIONS:
            return {"decision": stripped, "confidence": None}, None
        return None, {"reason": "no_json", "detail": "no JSON object found"}
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, {"reason": "json_decode_error", "detail": str(exc)}
    if not isinstance(obj, dict):
        return None, {"reason": "not_object", "detail": "top-level JSON is not object"}
    decision = str(obj.get("decision") or "").strip().upper()
    if decision not in PAIRWISE_DECISIONS:
        return None, {"reason": "schema_invalid", "detail": "decision must be A or B"}
    confidence = obj.get("confidence")
    if confidence is not None and (
        not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not (0 <= confidence <= 1)
    ):
        return None, {"reason": "schema_invalid", "detail": "confidence must be null or a number in [0, 1]"}
    return {"decision": decision, "confidence": confidence}, None


def score_pairwise_text(text: str, gold_label: str) -> dict[str, Any]:
    parsed, failure = parse_pairwise_decision_text(text)
    if failure is not None or parsed is None:
        return {"decision": "parse_error", "correct": False, "parse_failure": failure}
    decision = parsed["decision"]
    return {"decision": decision, "correct": decision == gold_label, "parse_failure": None, "confidence": parsed.get("confidence")}


def stable_row_hash(seed_key: str, row_id: str) -> str:
    return hashlib.sha1(f"{seed_key}\x00{row_id}".encode("utf-8")).hexdigest()


def select_balanced_rows(rows: list[PairwiseRow], *, n: int, seed_key: str) -> list[PairwiseRow]:
    if n <= 0:
        return []
    by_label: dict[str, list[PairwiseRow]] = defaultdict(list)
    for row in rows:
        by_label[row.gold_label].append(row)
    for label_rows in by_label.values():
        label_rows.sort(key=lambda r: stable_row_hash(seed_key, r.row_id))
    labels = [label for label in PAIRWISE_DECISIONS if by_label.get(label)]
    if not labels:
        return []
    target_each = n // len(labels)
    remainder = n % len(labels)
    selected: list[PairwiseRow] = []
    for idx, label in enumerate(labels):
        selected.extend(by_label[label][: target_each + (1 if idx < remainder else 0)])
    if len(selected) < n:
        selected_ids = {row.row_id for row in selected}
        leftovers = [row for row in rows if row.row_id not in selected_ids]
        leftovers.sort(key=lambda r: stable_row_hash(seed_key, r.row_id))
        selected.extend(leftovers[: n - len(selected)])
    selected.sort(key=lambda r: stable_row_hash(seed_key, r.row_id))
    return selected[:n]


def summarize_rows(rows: list[PairwiseRow]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "gold_label_counts": dict(Counter(row.gold_label for row in rows)),
        "source_counts": dict(Counter(f"{row.source_benchmark}|{row.source_suite}" for row in rows)),
        "task_kind_counts": dict(Counter(row.task_kind for row in rows)),
        "max_prompt_tokens_policy": DEFAULT_CONTEXT_LENGTH,
    }


def load_dataset(kind: str, path: Path, *, suite: str, score_key: str) -> list[PairwiseRow]:
    if kind == "judgebench":
        return load_judgebench(path, suite=suite)
    if kind == "llmbar":
        return load_llmbar(path, suite=suite)
    if kind == "judgelm":
        return load_judgelm(path, suite=suite, score_key=score_key)
    if kind == "reward-bench":
        return load_rewardbench(path, suite=suite)
    if kind == "reward-bench-2":
        return load_rewardbench2(path, suite=suite)
    raise ValueError(f"unsupported dataset kind: {kind}")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_dataset(args.dataset, args.path, suite=args.suite, score_key=args.score_key)
    selected = select_balanced_rows(rows, n=args.n, seed_key=f"{args.seed}:{args.dataset}:{args.suite}")
    prompt_fit_refusals: list[str] = []
    for row in selected:
        try:
            fit_pairwise_prompt_to_budget(
                row,
                context_length=args.context_length,
                max_completion_tokens=args.max_tokens,
                prompt_context_guard_tokens=args.prompt_guard_tokens,
                max_field_chars=args.max_field_chars,
            )
        except ValueError as exc:
            prompt_fit_refusals.append(f"{row.row_id}: {exc}")
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run",
        "dataset": {
            "kind": args.dataset,
            "path": str(args.path),
            "suite": args.suite,
            "score_key": args.score_key if args.dataset == "judgelm" else None,
            "file_sha256": sha256_file(args.path),
            "available": summarize_rows(rows),
            "selected": summarize_rows(selected),
            "selected_row_ids": [row.row_id for row in selected],
        },
        "request_policy": {
            "context_length": args.context_length,
            "max_tokens": args.max_tokens,
            "prompt_guard_tokens": args.prompt_guard_tokens,
            "max_field_chars": args.max_field_chars,
            "response_schema": {"decision": list(PAIRWISE_DECISIONS), "confidence": "number|null"},
        },
        "execution_allowed": bool(selected) and not prompt_fit_refusals,
        "refusal_reasons": prompt_fit_refusals + ([] if selected else ["no selected rows"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["judgebench", "llmbar", "judgelm", "reward-bench", "reward-bench-2"], required=True)
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--score-key", default="rougeLsum", help="JudgeLM score key to compare.")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS)
    ap.add_argument("--prompt-guard-tokens", type=int, default=DEFAULT_PROMPT_GUARD_TOKENS)
    ap.add_argument("--max-field-chars", type=int, default=DEFAULT_MAX_FIELD_CHARS)
    ap.add_argument("--out-plan", type=Path)
    ap.add_argument("--out-rows-jsonl", type=Path)
    args = ap.parse_args(argv)

    plan = build_plan(args)
    if args.out_plan:
        write_json(args.out_plan, plan)
    if args.out_rows_jsonl:
        rows = load_dataset(args.dataset, args.path, suite=args.suite, score_key=args.score_key)
        selected_ids = set(plan["dataset"]["selected_row_ids"])
        args.out_rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_rows_jsonl.open("w", encoding="utf-8") as fh:
            for row in rows:
                if row.row_id in selected_ids:
                    fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    if not args.out_plan:
        print(canonical_json(plan))
    return 0 if plan["execution_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
