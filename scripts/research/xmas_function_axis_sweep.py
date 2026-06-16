#!/usr/bin/env python3
"""Prepare and summarize true function-axis X-MAS sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

XMAS_DOMAINS: tuple[str, ...] = (
    "math",
    "code",
    "knowledge",
    "long_context",
    "reasoning",
)

XMAS_FUNCTIONS: tuple[str, ...] = (
    "solve",
    "verify",
    "plan",
    "refine",
    "extract",
)

DEFAULT_MANIFEST = (
    Path("/mnt/raid0/llm/epyc-inference-research")
    / "data"
    / "research"
    / "xmas_function_axis_manifest.v1.yaml"
)
DEFAULT_QUESTION_POOL = (
    Path("/mnt/raid0/llm/epyc-inference-research")
    / "benchmarks"
    / "prompts"
    / "question_pool.jsonl"
)

PROMPT_WRAPPERS: dict[str, str] = {
    "solve_direct": (
        "Function: solve. Produce the final answer for the task below using "
        "the task's requested answer format.\n\n{prompt}"
    ),
    "verify_answer": (
        "Function: verify. Check whether the task's expected answer is "
        "consistent with the prompt. Return <answer>valid</answer> if it is "
        "consistent, otherwise <answer>invalid</answer>, then give one brief "
        "reason.\n\nPrompt:\n{prompt}\n\nExpected answer: {expected}"
    ),
    "plan_solution": (
        "Function: plan. Write a concise plan for solving the task without "
        "giving the final answer. Return 3-6 ordered steps.\n\n{prompt}"
    ),
    "refine_answer": (
        "Function: refine. Improve the draft answer for clarity and correctness "
        "while preserving the requested final-answer format.\n\nPrompt:\n"
        "{prompt}\n\nDraft answer: {expected}"
    ),
    "extract_answer": (
        "Function: extract. Extract only the final answer requested by the "
        "task. Put it inside <answer></answer> tags.\n\nPrompt:\n{prompt}"
    ),
}


def load_manifest(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    validate_manifest(loaded)
    return loaded


def validate_manifest(manifest: dict[str, Any]) -> None:
    cells = manifest.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("manifest missing cells mapping")
    for domain in XMAS_DOMAINS:
        domain_cells = cells.get(domain)
        if not isinstance(domain_cells, dict):
            raise ValueError(f"manifest missing cells.{domain}")
        for function in XMAS_FUNCTIONS:
            cell = domain_cells.get(function)
            if not isinstance(cell, dict):
                raise ValueError(f"manifest missing cells.{domain}.{function}")
            _validate_cell(domain, function, cell, manifest)


def _validate_cell(
    domain: str,
    function: str,
    cell: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    task_ids_ref = cell.get("task_ids_ref")
    task_ids = cell.get("task_ids")
    if task_ids_ref:
        refs = manifest.get("domain_task_sets")
        if not isinstance(refs, dict) or task_ids_ref not in refs:
            raise ValueError(f"cells.{domain}.{function} has unknown task_ids_ref")
    elif not isinstance(task_ids, list) or not task_ids:
        raise ValueError(f"cells.{domain}.{function} needs task_ids or task_ids_ref")
    wrapper = cell.get("prompt_wrapper")
    if wrapper not in PROMPT_WRAPPERS:
        raise ValueError(f"cells.{domain}.{function} has unknown prompt_wrapper")
    if not isinstance(cell.get("scoring_family"), str):
        raise ValueError(f"cells.{domain}.{function} missing scoring_family")
    if not isinstance(cell.get("failure_policy"), str):
        raise ValueError(f"cells.{domain}.{function} missing failure_policy")


def load_question_pool(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("__pool_metadata__"):
                continue
            task_id = row.get("id")
            if isinstance(task_id, str):
                out[task_id] = row
    return out


def build_requests(
    manifest: dict[str, Any],
    question_pool: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    model_capture_profiles = _model_capture_profiles(manifest)
    for domain in XMAS_DOMAINS:
        for function in XMAS_FUNCTIONS:
            cell = manifest["cells"][domain][function]
            for source_id in _cell_task_ids(cell, manifest):
                source = question_pool.get(source_id)
                if source is None:
                    raise ValueError(f"unknown question_pool id: {source_id}")
                prompt = PROMPT_WRAPPERS[cell["prompt_wrapper"]].format(
                    prompt=source.get("prompt", ""),
                    expected=source.get("expected", ""),
                )
                requests.append({
                    "request_id": f"{domain}:{function}:{source_id}",
                    "domain": domain,
                    "function": function,
                    "cell": f"{domain}:{function}",
                    "source_task_id": source_id,
                    "source_suite": source.get("suite"),
                    "prompt": prompt,
                    "expected": _expected_for_function(function, source),
                    "source_expected": source.get("expected"),
                    "source_scoring_method": source.get("scoring_method"),
                    "scoring_family": cell["scoring_family"],
                    "failure_policy": cell["failure_policy"],
                    "capture_profile": cell.get("capture_profile", "default"),
                    "model_capture_profiles": model_capture_profiles,
                })
    return requests


def _model_capture_profiles(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = manifest.get("models", {})
    profiles = manifest.get("capture_profiles", {})
    if not isinstance(models, dict) or not isinstance(profiles, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for model_id, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            continue
        profile_name = str(model_cfg.get("capture_profile") or "default")
        profile = profiles.get(profile_name, {})
        out[str(model_id)] = {
            "url": model_cfg.get("url"),
            "capture_profile": profile_name,
            "chat_template_kwargs": (
                profile.get("chat_template_kwargs")
                if isinstance(profile, dict)
                else None
            ),
        }
    return out


def _cell_task_ids(cell: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    if cell.get("task_ids_ref"):
        refs = manifest["domain_task_sets"][cell["task_ids_ref"]]
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"empty task_ids_ref: {cell['task_ids_ref']}")
        return [str(item) for item in refs]
    return [str(item) for item in cell["task_ids"]]


def _expected_for_function(function: str, source: dict[str, Any]) -> str:
    if function == "verify":
        return "valid"
    return str(source.get("expected") or "")


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
            "wall_total": 0.0,
            "ok": 0,
            "failures": Counter(),
        }
    )
    for row in rows:
        domain = str(row["domain"])
        function = str(row["function"])
        model_id = str(row["model_id"])
        if domain not in XMAS_DOMAINS or function not in XMAS_FUNCTIONS:
            raise ValueError(f"invalid X-MAS cell in result row: {domain}:{function}")
        bucket = buckets[(domain, function, model_id)]
        bucket["total"] += 1
        if bool(row.get("correct")):
            bucket["correct"] += 1
        if bool(row.get("ok", True)):
            bucket["ok"] += 1
        bucket["wall_total"] += float(row.get("wall_s") or 0.0)
        failure_class = row.get("failure_class")
        if failure_class:
            bucket["failures"][str(failure_class)] += 1

    table: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    cell_winners: dict[str, str] = {}
    for domain in XMAS_DOMAINS:
        table[domain] = {}
        for function in XMAS_FUNCTIONS:
            model_rows = {
                model_id: _bucket_metrics(data)
                for (d, f, model_id), data in buckets.items()
                if d == domain and f == function
            }
            if not model_rows:
                raise ValueError(f"no result rows for cell {domain}:{function}")
            winner = _choose_winner(model_rows)
            table[domain][function] = model_rows
            cell_winners[f"{domain}:{function}"] = winner

    return {
        "table": table,
        "cell_winners": cell_winners,
        "winner_rule": "correct_desc_then_wall_mean_s_asc",
        "derivation_mode": "function_axis_sweep",
    }


def _bucket_metrics(bucket: dict[str, Any]) -> dict[str, Any]:
    total = int(bucket["total"])
    correct = int(bucket["correct"])
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "wall_mean_s": float(bucket["wall_total"]) / max(total, 1),
        "ok": int(bucket["ok"]),
        "failures": dict(bucket["failures"]),
    }


def _choose_winner(model_rows: dict[str, dict[str, Any]]) -> str:
    return min(
        model_rows,
        key=lambda model_id: (
            -int(model_rows[model_id]["correct"]),
            float(model_rows[model_id]["wall_mean_s"]),
            model_id,
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--question-pool", type=Path, default=DEFAULT_QUESTION_POOL)
    parser.add_argument("--emit-requests", type=Path)
    parser.add_argument("--results-jsonl", type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        if args.emit_requests:
            requests = build_requests(
                manifest,
                load_question_pool(args.question_pool),
            )
            write_jsonl(args.emit_requests, requests)
            print(f"Wrote {len(requests)} requests to {args.emit_requests}")
        if args.results_jsonl:
            if args.summary_out is None:
                raise ValueError("--results-jsonl requires --summary-out")
            rows = read_jsonl(args.results_jsonl)
            payload = {
                "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "manifest": str(args.manifest),
                "n_tasks": len({row.get("request_id") for row in rows}),
                "n_models": len({row.get("model_id") for row in rows}),
                "summary": summarize_results(rows),
            }
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"Wrote summary to {args.summary_out}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not args.emit_requests and not args.results_jsonl:
        print("Manifest validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
