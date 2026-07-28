#!/usr/bin/env python3
"""Observation-grade offline garbage-gate producer for authoritative E5 W0 runs.

This adapter deliberately owns no scorer or anomaly semantics. It imports the
governed B7 scorer and repetition detector from epyc-orchestrator, reads saved
responses only, and emits the JSONL contract consumed by server_numa_np_sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_QUESTION_POOL = RESEARCH_ROOT / "benchmarks/prompts/question_pool.jsonl"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
CAPTURE_METHODS = {
    "code_execution",
    "exact_match",
    "f1",
    "multiple_choice",
    "programmatic",
    "substring",
}


@dataclass(frozen=True)
class GovernedPrimitives:
    score_answer: Callable[[str, Any, str, dict[str, Any] | None], bool]
    extract_multiple_choice_letter: Callable[[str], str | None]
    extract_multiple_choice_text_index: Callable[[str, list[Any]], int | None]
    extract_code_block: Callable[[str, str], str | None]
    detect_repetition_loop: Callable[[str], bool]
    repetition_loop_threshold: Callable[[], float]
    scorer_path: Path
    anomaly_path: Path
    anomaly_config_path: Path


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load governed module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_governed_primitives(orchestrator_root: Path) -> GovernedPrimitives:
    scorer_path = orchestrator_root / "scripts/benchmark/debug_scorer.py"
    anomaly_path = orchestrator_root / "src/pipeline_monitor/anomaly.py"
    if not scorer_path.is_file() or not anomaly_path.is_file():
        raise FileNotFoundError("governed scorer or anomaly primitive is missing")
    scorer = _load_module("e5_w0_debug_scorer", scorer_path)
    anomaly = _load_module("e5_w0_anomaly", anomaly_path)
    return GovernedPrimitives(
        score_answer=scorer.score_answer,
        extract_multiple_choice_letter=scorer._extract_multiple_choice_letter,
        extract_multiple_choice_text_index=scorer._extract_multiple_choice_text_index,
        extract_code_block=scorer._extract_code_block,
        detect_repetition_loop=anomaly.detect_repetition_loop,
        repetition_loop_threshold=lambda: float(
            anomaly._get_signal_param("repetition_loop", "threshold", 0.4)
        ),
        scorer_path=scorer_path,
        anomaly_path=anomaly_path,
        anomaly_config_path=anomaly_path.parents[2] / "orchestration/anomaly_signals.yaml",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def question_pool_by_qid(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_jsonl(path)
    pool: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qid = row.get("id")
        if qid is None:
            continue
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"invalid question-pool id in {path}")
        pool.setdefault(qid, []).append(row)
    if not pool:
        raise ValueError(f"no question rows in {path}")
    return pool


def strip_completed_think_blocks(text: str) -> str:
    """Mirror debug_scorer: remove completed tags only; leave incomplete tags."""
    return THINK_BLOCK_RE.sub("", text).strip()


def parse_ok_for_response(
    response_text: str, scoring_method: str, scoring_config: dict[str, Any], primitives: GovernedPrimitives
) -> bool:
    if scoring_method not in CAPTURE_METHODS:
        raise ValueError(f"unknown or unsupported B7 capture method: {scoring_method}")
    post_think = strip_completed_think_blocks(response_text)
    if not post_think:
        return False
    if scoring_method == "multiple_choice":
        if primitives.extract_multiple_choice_letter(post_think) is not None:
            return True
        choices = scoring_config.get("choices")
        return isinstance(choices, list) and primitives.extract_multiple_choice_text_index(post_think, choices) is not None
    if scoring_method == "code_execution":
        language = str(scoring_config.get("language") or "python")
        return primitives.extract_code_block(post_think, language) is not None
    return True


def validate_coverage(
    run_dir: Path, selected_qids: list[str], cells: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    if len(selected_qids) != len(set(selected_qids)):
        raise ValueError(f"duplicate selected qid in {run_dir}")
    cell_ids = [row.get("cell_id") for row in cells]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids):
        raise ValueError(f"invalid cell_id in {run_dir}/cells.jsonl")
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError(f"duplicate cell_id in {run_dir}/cells.jsonl")
    expected = {(cell_id, qid) for cell_id in cell_ids for qid in selected_qids}
    observed: set[tuple[str, str]] = set()
    for row in responses:
        cell_id, qid = row.get("cell_id"), row.get("qid")
        if not isinstance(cell_id, str) or not isinstance(qid, str):
            raise ValueError(f"response missing cell_id or qid in {run_dir}")
        pair = (cell_id, qid)
        if pair not in expected:
            raise ValueError(f"unexpected response coverage pair {pair} in {run_dir}")
        if pair in observed:
            raise ValueError(f"duplicate response coverage pair {pair} in {run_dir}")
        observed.add(pair)
    if len(responses) != len(expected) or observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"response coverage mismatch in {run_dir}; missing={missing[:3]}")
    return sorted(cell_ids), selected_qids


def score_run(
    run_dir: Path, pool: dict[str, list[dict[str, Any]]], primitives: GovernedPrimitives
) -> dict[str, Any]:
    selected_path = run_dir / "selected_prompts.jsonl"
    cells_path = run_dir / "cells.jsonl"
    responses_path = run_dir / "responses.jsonl"
    for path in (selected_path, cells_path, responses_path):
        if not path.is_file():
            raise FileNotFoundError(f"required W0 input missing: {path}")
    selected = read_jsonl(selected_path)
    selected_qids = [row.get("qid") for row in selected]
    if any(not isinstance(qid, str) or not qid or not isinstance(row.get("prompt"), str) for row, qid in zip(selected, selected_qids)):
        raise ValueError(f"invalid selected qid in {selected_path}")
    missing_pool = sorted(set(selected_qids) - set(pool))
    if missing_pool:
        raise ValueError(f"selected qids missing from question pool: {missing_pool[:3]}")
    selected_questions: dict[str, dict[str, Any]] = {}
    for row in selected:
        qid = row["qid"]
        matches = [candidate for candidate in pool[qid] if candidate.get("prompt") == row["prompt"]]
        distinct_metadata = {
            json.dumps(
                {
                    "expected": candidate.get("expected"),
                    "scoring_method": candidate.get("scoring_method"),
                    "scoring_config": candidate.get("scoring_config") or {},
                },
                sort_keys=True,
            )
            for candidate in matches
        }
        if len(distinct_metadata) != 1:
            raise ValueError(f"selected qid has no unique scoring metadata: {qid}")
        selected_questions[qid] = matches[0]
    cells = read_jsonl(cells_path)
    responses = read_jsonl(responses_path)
    cell_ids, selected_qids = validate_coverage(run_dir, selected_qids, cells, responses)

    scores: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    for response in sorted(responses, key=lambda row: (str(row["cell_id"]), str(row["qid"]))):
        if response.get("http_status") != 200:
            raise ValueError(f"non-200 response for {(response.get('cell_id'), response.get('qid'))}")
        text = response.get("response_text")
        if not isinstance(text, str):
            raise ValueError(f"missing response_text for {(response.get('cell_id'), response.get('qid'))}")
        question = selected_questions[response["qid"]]
        method = question.get("scoring_method")
        if not isinstance(method, str):
            raise ValueError(f"unknown scoring method for {response['qid']}")
        config = question.get("scoring_config") or {}
        if not isinstance(config, dict):
            raise ValueError(f"invalid scoring_config for {response['qid']}")
        try:
            primitives.score_answer(text, question.get("expected"), method, config)
        except Exception as exc:
            raise RuntimeError(f"governed scorer failed for {(response['cell_id'], response['qid'])}") from exc
        parse_ok = parse_ok_for_response(text, method, config, primitives)
        scores.append({
            "cell_id": response["cell_id"],
            "qid": response["qid"],
            "parse_ok": parse_ok,
            "repetition_loop": bool(primitives.detect_repetition_loop(text)),
        })
        methods[method] += 1

    output_path = run_dir / "offline_scores.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=False) + "\n")
    parse_failures = Counter(row["cell_id"] for row in scores if not row["parse_ok"])
    repetition_flags = Counter(row["cell_id"] for row in scores if row["repetition_loop"])
    provenance = {
        "artifact_type": "e5_w0_offline_scores_observation_grade",
        "run_dir": str(run_dir),
        "contract": {"fields": ["cell_id", "qid", "parse_ok", "repetition_loop"], "row_count": len(scores)},
        "coverage": {"cells": len(cell_ids), "qids": len(selected_qids), "expected_rows": len(cell_ids) * len(selected_qids)},
        "methods": dict(sorted(methods.items())),
        "garbage_gate_observation": {
            "max_parse_failures_per_cell": 2,
            "repetition_loop_threshold": primitives.repetition_loop_threshold(),
            "parse_failures_by_cell": dict(sorted(parse_failures.items())),
            "repetition_flags_by_cell": dict(sorted(repetition_flags.items())),
        },
        "hashes": {
            "offline_scores_jsonl": sha256_file(output_path),
            "selected_prompts_jsonl": sha256_file(selected_path),
            "cells_jsonl": sha256_file(cells_path),
            "responses_jsonl": sha256_file(responses_path),
            "question_pool_jsonl": None,
            "debug_scorer_py": sha256_file(primitives.scorer_path),
            "anomaly_py": sha256_file(primitives.anomaly_path),
            "anomaly_signals_yaml": sha256_file(primitives.anomaly_config_path),
        },
    }
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--question-pool", type=Path, default=DEFAULT_QUESTION_POOL)
    parser.add_argument("--orchestrator-root", type=Path, default=DEFAULT_ORCHESTRATOR_ROOT)
    args = parser.parse_args(argv)
    if not args.question_pool.is_file():
        parser.error(f"question pool not found: {args.question_pool}")
    primitives = load_governed_primitives(args.orchestrator_root)
    pool = question_pool_by_qid(args.question_pool)
    pool_hash = sha256_file(args.question_pool)
    for run_dir in args.run_dir:
        provenance = score_run(run_dir, pool, primitives)
        provenance["hashes"]["question_pool_jsonl"] = pool_hash
        provenance_path = run_dir / "offline_scores.provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {run_dir / 'offline_scores.jsonl'} and {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
