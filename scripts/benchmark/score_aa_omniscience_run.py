#!/usr/bin/env python3
from __future__ import annotations

"""Score an AA-Omniscience benchmark run with deterministic F1 labels.

This is a non-inference scorer for the G10/G11/G12 factual-risk lane. It maps
raw benchmark responses onto the 4-class AA label vocabulary:

* CORRECT: token F1 is at or above the configured threshold.
* PARTIAL_ANSWER: non-abstained answer has partial token overlap.
* INCORRECT: non-abstained answer has no token overlap.
* NOT_ATTEMPTED: explicit abstention or empty answer.

The output is suitable for factual-risk aggregation, but it is intentionally
marked as deterministic F1 scoring rather than LLM-as-judge scoring.
"""

import argparse
import json
import re
import string
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_adapters import AAOmniscienceAdapter  # noqa: E402


DEFAULT_RUN_DIR = PROJECT_ROOT / "benchmarks" / "results" / "runs" / "20260620_035613"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "package_g" / "omniscience"
DEFAULT_CONFIGS = ("baseline", "moe4", "moe6")


@dataclass(frozen=True)
class QuestionMeta:
    question_id: str
    expected: str
    domain: str
    topic: str
    tier: int | None
    scoring_config: dict[str, Any]


@dataclass(frozen=True)
class Score:
    label: str
    extracted_answer: str
    f1: float
    precision: float
    recall: float
    threshold: float


def normalize_text(text: str) -> str:
    """Normalize text for token-F1 scoring."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def extract_answer(response: str, pattern: str = r"<answer>(.*?)</answer>") -> str:
    """Extract the final answer tag, or fall back to the last non-empty line."""
    matches = re.findall(pattern, response or "", re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (response or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def is_abstention(answer: str, patterns: list[str]) -> bool:
    if not answer.strip():
        return True
    return any(re.search(pattern, answer, re.IGNORECASE) for pattern in patterns)


def token_f1(prediction: str, expected: str) -> tuple[float, float, float]:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(expected).split()
    if not gold_tokens:
        return (1.0, 1.0, 1.0) if not pred_tokens else (0.0, 0.0, 0.0)
    if not pred_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_response(response: str, expected: str, scoring_config: dict[str, Any]) -> Score:
    pattern = scoring_config.get("extract_pattern", r"<answer>(.*?)</answer>")
    threshold = float(scoring_config.get("threshold", 0.8))
    abstention_patterns = list(scoring_config.get("abstention_patterns", []))

    extracted = extract_answer(response, pattern)
    if is_abstention(extracted, abstention_patterns):
        return Score("NOT_ATTEMPTED", extracted, 0.0, 0.0, 0.0, threshold)

    precision, recall, f1 = token_f1(extracted, expected)
    if f1 >= threshold:
        label = "CORRECT"
    elif f1 > 0:
        label = "PARTIAL_ANSWER"
    else:
        label = "INCORRECT"
    return Score(label, extracted, f1, precision, recall, threshold)


def load_question_meta() -> dict[str, QuestionMeta]:
    adapter = AAOmniscienceAdapter()
    questions = adapter.extract_all()
    meta: dict[str, QuestionMeta] = {}
    for question in questions:
        prompt = question.get("prompt", "")
        match = re.search(
            r"You are answering questions about (?P<domain>.*?), and in particular (?P<topic>.*?)\.",
            prompt,
            re.DOTALL,
        )
        domain = match.group("domain").strip() if match else "unknown"
        topic = match.group("topic").strip() if match else "unknown"
        qid = str(question["id"])
        meta[qid] = QuestionMeta(
            question_id=qid,
            expected=str(question.get("expected", "")).strip(),
            domain=domain,
            topic=topic,
            tier=question.get("tier"),
            scoring_config=dict(question.get("scoring_config", {})),
        )
    return meta


def load_config_result(run_dir: Path, role: str, config: str) -> dict[str, Any]:
    path = run_dir / f"{role}_{config}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_scored_rows(run_dir: Path, role: str, configs: tuple[str, ...]) -> list[dict[str, Any]]:
    meta_by_qid = load_question_meta()
    rows: list[dict[str, Any]] = []

    for config in configs:
        data = load_config_result(run_dir, role, config)
        suite_rows = data.get("results", {}).get("omniscience", {})
        for qid, result in sorted(suite_rows.items()):
            meta = meta_by_qid.get(qid)
            if meta is None:
                continue
            score = score_response(result.get("response", ""), meta.expected, meta.scoring_config)
            rows.append({
                "run_id": data.get("run_id"),
                "role": role,
                "model": data.get("model_path", "unknown"),
                "config": config,
                "question_id": qid,
                "domain": meta.domain,
                "topic": meta.topic,
                "tier": meta.tier,
                "expected_answer": meta.expected,
                "response": result.get("response", ""),
                "extracted_answer": score.extracted_answer,
                "label_4class": score.label,
                "outcome": score.label,
                "label_source": "aa_omniscience_deterministic_f1",
                "source": "aa_omniscience",
                "scoring_method": "deterministic_f1",
                "f1": score.f1,
                "precision": score.precision,
                "recall": score.recall,
                "threshold": score.threshold,
                "tokens_per_second": result.get("tokens_per_second"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "total_time_ms": result.get("total_time_ms"),
            })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_config: dict[str, Counter[str]] = defaultdict(Counter)
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    f1_by_config: dict[str, list[float]] = defaultdict(list)
    tps_by_config: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        config = str(row["config"])
        domain = str(row["domain"])
        label = str(row["label_4class"])
        by_config[config][label] += 1
        by_domain[domain][label] += 1
        f1_by_config[config].append(float(row["f1"]))
        tps = row.get("tokens_per_second")
        if isinstance(tps, int | float):
            tps_by_config[config].append(float(tps))

    def shape(counter: Counter[str]) -> dict[str, Any]:
        total = sum(counter.values())
        correct = counter.get("CORRECT", 0)
        incorrect = counter.get("INCORRECT", 0)
        partial = counter.get("PARTIAL_ANSWER", 0)
        not_attempted = counter.get("NOT_ATTEMPTED", 0)
        denom = incorrect + partial + not_attempted
        return {
            "total": total,
            "label_counts": dict(sorted(counter.items())),
            "accuracy": correct / total if total else None,
            "hallucination_rate": incorrect / denom if denom else None,
            "omniscience_index": (
                0.5 * (correct / total) + 0.5 * (1 - incorrect / denom)
                if total and denom
                else None
            ),
        }

    summary = {
        "row_count": len(rows),
        "scoring_method": "deterministic_f1",
        "configs": {},
        "domains": {},
    }
    for config, counter in sorted(by_config.items()):
        config_summary = shape(counter)
        f1_values = f1_by_config[config]
        tps_values = tps_by_config[config]
        config_summary["avg_f1"] = sum(f1_values) / len(f1_values) if f1_values else None
        config_summary["avg_tokens_per_second"] = sum(tps_values) / len(tps_values) if tps_values else None
        summary["configs"][config] = config_summary
    summary["domains"] = {domain: shape(counter) for domain, counter in sorted(by_domain.items())}
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--role", default="frontdoor")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = tuple(args.configs)
    basename = args.basename or f"{args.role}_{args.run_dir.name}_aa_omniscience"
    jsonl_path = args.output_dir / f"{basename}.jsonl"
    summary_path = args.output_dir / f"{basename}_summary.json"

    rows = iter_scored_rows(args.run_dir, args.role, configs)
    summary = summarize(rows)
    summary.update({
        "run_dir": str(args.run_dir),
        "role": args.role,
        "configs_scored": list(configs),
        "jsonl_path": str(jsonl_path),
    })

    write_jsonl(jsonl_path, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
