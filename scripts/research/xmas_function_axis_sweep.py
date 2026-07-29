#!/usr/bin/env python3
"""Prepare and summarize true function-axis X-MAS sweeps."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from answer_scoring import extract_letter_answer

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


def filter_requests(
    requests: list[dict[str, Any]],
    *,
    domain: str | None = None,
    function: str | None = None,
    cell: str | None = None,
    source_task_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return requests matching optional sweep-slice filters."""
    if cell is not None:
        if ":" not in cell:
            raise ValueError("--cell must use domain:function syntax")
        raw_domain, raw_function = cell.split(":", 1)
        domain = raw_domain
        function = raw_function
    if domain is not None and domain not in XMAS_DOMAINS:
        raise ValueError(f"unknown X-MAS domain: {domain}")
    if function is not None and function not in XMAS_FUNCTIONS:
        raise ValueError(f"unknown X-MAS function: {function}")
    return [
        request for request in requests
        if (domain is None or request["domain"] == domain)
        and (function is None or request["function"] == function)
        and (
            source_task_id is None
            or request["source_task_id"] == source_task_id
        )
    ]


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


def summarize_results(
    rows: list[dict[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
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
                if not require_complete:
                    continue
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


def normalise_text(value: str) -> str:
    """Normalize text for low-cost auto-scoring."""
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text or "", flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else (text or "").strip()


def score_response(request: dict[str, Any], answer: str) -> tuple[bool, str]:
    """Return (correct, failure_class) for auto-scorable sweep cells."""
    if not answer:
        return False, "empty_content"
    scoring_family = request.get("scoring_family")
    expected = str(request.get("expected") or "").strip()
    extracted = extract_answer(answer)
    normalized_expected = normalise_text(expected)
    normalized_answer = normalise_text(extracted)

    if scoring_family == "binary_validity":
        if normalized_answer.startswith("valid"):
            return expected == "valid", ""
        if normalized_answer.startswith("invalid"):
            return expected == "invalid", ""
        return False, "parse_failure"

    if scoring_family == "rubric":
        return _score_plan_structure(answer)

    source_method = str(request.get("source_scoring_method") or "substring")
    if source_method == "multiple_choice":
        letter = expected.upper()
        if not letter or letter not in "ABCD":
            return False, "scoring_error"
        return extract_letter_answer(extracted) == letter, ""
    if source_method in {"exact_match", "substring", "f1"}:
        return normalized_expected in normalized_answer, ""
    return normalized_expected in normalized_answer, ""


def _score_plan_structure(answer: str) -> tuple[bool, str]:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    step_lines = [
        line for line in lines
        if re.match(r"^(?:\d+[\.\)]|[-*])\s+", line)
    ]
    if 3 <= len(step_lines) <= 8:
        return True, ""
    return False, "rubric_unscored"


def _health_url(api_url: str) -> str:
    return api_url.rstrip("/").removesuffix("/v1") + "/health"


def preflight_idle(manifest: dict[str, Any]) -> list[str]:
    """Return health-gate errors for configured model endpoints."""
    errors: list[str] = []
    models = manifest.get("models", {})
    if not isinstance(models, dict):
        return ["manifest models must be a mapping"]
    for model_id, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            errors.append(f"model {model_id} config must be a mapping")
            continue
        url = model_cfg.get("url")
        if not isinstance(url, str) or not url:
            errors.append(f"model {model_id} missing url")
            continue
        try:
            with urllib.request.urlopen(_health_url(url), timeout=5.0) as resp:
                health = json.loads(resp.read())
        except Exception as exc:
            errors.append(f"{model_id} unreachable: {exc}")
            continue
        if health.get("status") != "ok":
            errors.append(f"{model_id} status={health.get('status')!r}")
        if int(health.get("slots_processing", 0) or 0) > 0:
            errors.append(f"{model_id} busy: slots_processing={health.get('slots_processing')}")
    return errors


async def query_model(
    client: Any,
    request: dict[str, Any],
    model_id: str,
    model_cfg: dict[str, Any],
    *,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": request["prompt"]}],
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    chat_template_kwargs = model_cfg.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict):
        body["chat_template_kwargs"] = chat_template_kwargs

    t0 = time.monotonic()
    try:
        payload = _post_json(
            f"{model_cfg['url'].rstrip('/')}/chat/completions",
            body,
            timeout_s,
        )
        content = payload["choices"][0]["message"].get("content") or ""
        usage = payload.get("usage", {}) or {}
        return {
            "ok": True,
            "wall_s": time.monotonic() - t0,
            "answer": content,
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
        }
    except Exception as exc:
        return {
            "ok": False,
            "wall_s": time.monotonic() - t0,
            "answer": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": repr(exc),
        }


def _post_json(url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time",
                f"{timeout_s:.3f}",
                "--connect-timeout",
                "5",
                "--header",
                "Content-Type: application/json",
                "--data-binary",
                json.dumps(body),
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s + 2.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"request exceeded {timeout_s:.1f}s wall timeout") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(detail) from exc
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("chat completion response must be a mapping")
    return payload


async def run_requests(
    requests: list[dict[str, Any]],
    *,
    limit_requests: int | None = None,
    output_path: Path | None = None,
    skip_keys: set[tuple[str, str]] | None = None,
    timeout_s: float = 600.0,
) -> list[dict[str, Any]]:
    selected = requests[:limit_requests] if limit_requests else requests
    skipped = skip_keys or set()
    rows: list[dict[str, Any]] = []
    output_fh = output_path.open("a", encoding="utf-8") if output_path else None
    for request in selected:
        model_profiles = request.get("model_capture_profiles", {})
        if not isinstance(model_profiles, dict):
            raise ValueError(f"request {request['request_id']} missing model profiles")
        for model_id, model_cfg in model_profiles.items():
            row_key = (str(request["request_id"]), str(model_id))
            if row_key in skipped:
                continue
            if not isinstance(model_cfg, dict):
                raise ValueError(f"model profile {model_id} must be a mapping")
            result = await query_model(
                None,
                request,
                str(model_id),
                model_cfg,
                timeout_s=timeout_s,
            )
            correct, failure_class = score_response(request, result["answer"])
            row = {
                "request_id": request["request_id"],
                "domain": request["domain"],
                "function": request["function"],
                "cell": request["cell"],
                "source_task_id": request["source_task_id"],
                "source_suite": request["source_suite"],
                "model_id": model_id,
                "correct": correct,
                "failure_class": failure_class,
                "wall_s": result["wall_s"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "answer_excerpt": (result["answer"] or "")[:300],
                "expected": request["expected"],
                "scoring_family": request["scoring_family"],
                "source_scoring_method": request["source_scoring_method"],
                "ok": result["ok"],
                "error": result.get("error"),
            }
            rows.append(row)
            if output_fh is not None:
                output_fh.write(json.dumps(row, sort_keys=True) + "\n")
                output_fh.flush()
            print(
                f"  [{model_id:<22} {request['cell']:<22} "
                f"{request['source_task_id']:<36}] correct={correct} "
                f"wall={result['wall_s']:.1f}s"
                + (" ERROR" if not result["ok"] else "")
            )
    if output_fh is not None:
        output_fh.close()
    return rows


def completed_result_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        request_id = row.get("request_id")
        model_id = row.get("model_id")
        if isinstance(request_id, str) and isinstance(model_id, str):
            keys.add((request_id, model_id))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--question-pool", type=Path, default=DEFAULT_QUESTION_POOL)
    parser.add_argument("--emit-requests", type=Path)
    parser.add_argument("--run-out", type=Path)
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--domain", choices=XMAS_DOMAINS)
    parser.add_argument("--function", choices=XMAS_FUNCTIONS)
    parser.add_argument("--cell", help="domain:function request filter")
    parser.add_argument("--source-task-id", help="single source task id filter")
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--skip-health-gate", action="store_true")
    parser.add_argument("--results-jsonl", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--allow-partial-summary", action="store_true")
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        action_taken = False
        if args.emit_requests:
            action_taken = True
            requests = build_requests(
                manifest,
                load_question_pool(args.question_pool),
            )
            requests = filter_requests(
                requests,
                domain=args.domain,
                function=args.function,
                cell=args.cell,
                source_task_id=args.source_task_id,
            )
            if args.limit_requests:
                requests = requests[:args.limit_requests]
            write_jsonl(args.emit_requests, requests)
            print(f"Wrote {len(requests)} requests to {args.emit_requests}")
        if args.run_out:
            action_taken = True
            requests = build_requests(
                manifest,
                load_question_pool(args.question_pool),
            )
            requests = filter_requests(
                requests,
                domain=args.domain,
                function=args.function,
                cell=args.cell,
                source_task_id=args.source_task_id,
            )
            if not args.skip_health_gate:
                errors = preflight_idle(manifest)
                if errors:
                    raise ValueError("health gate failed: " + "; ".join(errors))
            args.run_out.parent.mkdir(parents=True, exist_ok=True)
            existing = completed_result_keys(args.run_out)
            rows = asyncio.run(
                run_requests(
                    requests,
                    limit_requests=args.limit_requests,
                    output_path=args.run_out,
                    skip_keys=existing,
                    timeout_s=args.request_timeout_s,
                )
            )
            print(
                f"Appended {len(rows)} result rows to {args.run_out} "
                f"({len(existing)} existing skipped)"
            )
        if args.results_jsonl:
            action_taken = True
            if args.summary_out is None:
                raise ValueError("--results-jsonl requires --summary-out")
            rows = read_jsonl(args.results_jsonl)
            payload = {
                "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "manifest": str(args.manifest),
                "n_tasks": len({row.get("request_id") for row in rows}),
                "n_models": len({row.get("model_id") for row in rows}),
                "summary": summarize_results(
                    rows,
                    require_complete=not args.allow_partial_summary,
                ),
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

    if not action_taken:
        print("Manifest validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
