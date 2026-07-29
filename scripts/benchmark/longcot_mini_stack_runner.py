#!/usr/bin/env python3
"""Run LongCoT-Mini against already-resident optimized stack ports.

This runner is intentionally narrow for RE-4/K-LCM-1. It does not start or stop
model servers. Instead it verifies provided llama-server ports and fans out a
single role across those ports, preserving the existing ``run_benchmark.py``
result JSON shape so ``score_longcot_run.py`` remains the deterministic scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from suites import Question, load_suite  # noqa: E402

SUITE_NAME = "longcot_mini"

PROMPT_MODE_STANDARD = "standard"
PROMPT_MODE_CONCISE_SOLUTION = "concise_solution"
PROMPT_MODES = (PROMPT_MODE_STANDARD, PROMPT_MODE_CONCISE_SOLUTION)

CONCISE_SOLUTION_SYSTEM = (
    "You are solving a deterministic benchmark. Compute the answer internally, "
    "but do not show derivation, scratch work, caveats, or prose. Return exactly "
    "one final-answer line in this form: solution = <value>"
)
CONCISE_SOLUTION_SUFFIX = (
    "\n\nOutput contract for this run: do not show your derivation. Return exactly "
    "one line, and that line must be: solution = <value>"
)
LONGCOT_STEP_BY_STEP_PREAMBLE = (
    "Solve this problem step by step and return the final solution at the end."
)
SOLUTION_MARKER_GRAMMAR = r'root ::= "solution = " ([^\n])+ "\n"?'
# Case-insensitive terminal-marker probe used to short-circuit the two-phase
# protocol when Phase 1 already emitted a ``solution =`` line (mirrors the
# adapter's ``_SOLUTION_RE`` anchor so the runner and scorer agree on presence).
SOLUTION_MARKER_RE = re.compile(r"solution\s*=\s*", re.IGNORECASE)
# Phase-2 user turn: forces exactly the final-answer line (grammar-constrained).
FINAL_ANSWER_INSTRUCTION = (
    "Output ONLY your final answer now, as a single line, exactly: "
    "solution = <value>"
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_role_ports(items: Iterable[str]) -> dict[str, list[int]]:
    roles: dict[str, list[int]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"role ports must be role=port[,port...], got {item!r}")
        role, raw_ports = item.split("=", 1)
        role = role.strip()
        ports = [int(p.strip()) for p in raw_ports.split(",") if p.strip()]
        if not role or not ports:
            raise ValueError(f"invalid role port mapping {item!r}")
        roles[role] = ports
    return roles


def _read_probe_ids(path: Path) -> set[str]:
    """Read a newline-delimited ``question_id`` allowlist (``#`` comments ok)."""
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            ids.add(token)
    return ids


def _load_domain_map(suite_name: str) -> dict[str, str]:
    """Map ``question.id`` -> domain via the registered dataset adapter.

    The runner's ``Question`` objects drop the per-row domain, so the stratified
    ``--limit-per-domain`` probe rehydrates it from the adapter's ``metadata``
    (same source the deterministic scorer uses). Pure data read, no inference.
    """
    try:
        from dataset_adapters import get_adapter
    except ImportError:  # pragma: no cover - path shim
        sys.path.insert(0, str(_BENCHMARK_DIR))
        from dataset_adapters import get_adapter
    adapter = get_adapter(suite_name)
    if adapter is None:
        return {}
    dmap: dict[str, str] = {}
    for item in adapter.extract_all():
        meta = item.get("metadata") or {}
        dmap[str(item.get("id"))] = str(meta.get("domain", ""))
    return dmap


def _select_questions(
    questions: list[Question],
    *,
    probe_ids: set[str] | None = None,
    limit_per_domain: int | None = None,
    domain_map: dict[str, str] | None = None,
) -> list[Question]:
    """Deterministically subset ``questions`` for a probe run (no sampling).

    ``probe_ids`` — keep exactly the listed ids (input order preserved).
    ``limit_per_domain`` — group by ``domain_map`` and keep the first ``N`` per
    domain by sorted ``question_id`` (domains processed in sorted order).
    """
    if probe_ids is not None:
        return [q for q in questions if q.id in probe_ids]
    if limit_per_domain is not None:
        domain_map = domain_map or {}
        by_domain: dict[str, list[Question]] = {}
        for q in questions:
            by_domain.setdefault(domain_map.get(q.id, ""), []).append(q)
        selected: list[Question] = []
        for domain in sorted(by_domain):
            ordered = sorted(by_domain[domain], key=lambda q: q.id)
            selected.extend(ordered[:limit_per_domain])
        return selected
    return questions


def _port_open(port: int, host: str = "127.0.0.1", timeout_s: float = 1.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _http_json(url: str, payload: dict[str, Any] | None, timeout_s: int) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _health_ok(port: int, host: str = "127.0.0.1") -> bool:
    if not _port_open(port, host=host):
        return False
    try:
        data = _http_json(f"http://{host}:{port}/health", None, timeout_s=2)
    except Exception:
        return False
    return data.get("status") == "ok"


def _completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
    content = data.get("content")
    return content if isinstance(content, str) else ""


def _prompt_for_mode(prompt: str, prompt_mode: str) -> str:
    if prompt_mode == PROMPT_MODE_STANDARD:
        return prompt
    if prompt_mode == PROMPT_MODE_CONCISE_SOLUTION:
        body = prompt.strip()
        if body.startswith(LONGCOT_STEP_BY_STEP_PREAMBLE):
            body = body[len(LONGCOT_STEP_BY_STEP_PREAMBLE):].lstrip()
        return (
            "Answer-only LongCoT-Mini item. Do not reason aloud; compute privately "
            "and return only the final answer.\n\n"
            f"{body}{CONCISE_SOLUTION_SUFFIX}"
        )
    raise ValueError(f"unknown prompt mode {prompt_mode!r}")


def _run_question(
    *,
    host: str,
    port: int,
    role: str,
    question: Question,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    endpoint: str,
    disable_thinking: bool,
    prompt_mode: str,
    force_solution_grammar: bool,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    start = time.time()
    url = (
        f"http://{host}:{port}/v1/chat/completions"
        if endpoint == "chat"
        else f"http://{host}:{port}/completion"
    )
    prompt = _prompt_for_mode(question.prompt, prompt_mode)
    if endpoint == "chat":
        messages = [{"role": "user", "content": prompt}]
        if prompt_mode == PROMPT_MODE_CONCISE_SOLUTION:
            messages.insert(0, {"role": "system", "content": CONCISE_SOLUTION_SYSTEM})
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if force_solution_grammar:
            payload["grammar"] = SOLUTION_MARKER_GRAMMAR
    else:
        if prompt_mode == PROMPT_MODE_CONCISE_SOLUTION:
            prompt = CONCISE_SOLUTION_SYSTEM + "\n\n" + prompt
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
            "cache_prompt": False,
        }
        if force_solution_grammar:
            payload["grammar"] = SOLUTION_MARKER_GRAMMAR

    # Reproducibility: seed is threaded only when explicitly provided so a
    # flags-absent (v1) invocation produces byte-identical payloads.
    if seed is not None:
        payload["seed"] = seed

    row: dict[str, Any] = {
        "question_id": question.id,
        "prompt": prompt,
        "expected": question.expected,
        "port": port,
        "role": role,
        "endpoint": endpoint,
        "prompt_mode": prompt_mode,
        "force_solution_grammar": force_solution_grammar,
        # This runner does not request completion probabilities, so it must
        # never fabricate a numeric confidence for downstream calibration.
        "confidence": None,
        "confidence_is_real": False,
        "confidence_source": "not_collected",
    }
    try:
        data = _http_json(url, payload, timeout_s=timeout_s)
        elapsed_s = time.time() - start
        text = _completion_text(data)
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        completion_tokens = (
            usage.get("completion_tokens")
            or timings.get("predicted_n")
            or len(text.split())
        )
        prompt_tokens = usage.get("prompt_tokens") or timings.get("prompt_n")
        tps = timings.get("predicted_per_second")
        if not tps and completion_tokens and elapsed_s > 0:
            tps = float(completion_tokens) / elapsed_s
        row.update(
            {
                "response": text,
                "success": True,
                "elapsed_s": elapsed_s,
                "tokens_per_second": tps,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw_usage": usage,
            }
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        row.update(
            {
                "response": "",
                "success": False,
                "elapsed_s": time.time() - start,
                "tokens_per_second": None,
                "error": str(exc),
                # A transport/decoder failure is an infra observation, not a
                # wrong answer.  The deterministic scorer consumes this
                # explicit exclusion rather than treating its empty response
                # as a scored failure (REL-1).
                "error_type": "infra_error",
                "excluded_from_scoring": True,
                "exclusion_reason": "infra_error",
            }
        )
    return question.id, row


def _run_question_two_phase(
    *,
    host: str,
    port: int,
    role: str,
    question: Question,
    reasoning_budget: int,
    final_answer_max_tokens: int,
    temperature: float,
    timeout_s: int,
    disable_thinking: bool,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Two-phase forced-final-answer generation (RE-4 v2 protocol).

    Phase 1: free CoT, ``max_tokens=reasoning_budget``, no grammar. If Phase 1
    already emitted a ``solution =`` marker the second call is short-circuited
    (0 extra HTTP calls). Otherwise Phase 2 feeds Phase-1's reasoning back
    verbatim and applies ``SOLUTION_MARKER_GRAMMAR`` to THAT turn only, forcing
    a terminal answer line. ``response`` splices the forced line onto the
    reasoning so the scorer's last-marker anchor lands on it. Chat endpoint only.
    """
    start = time.time()
    url = f"http://{host}:{port}/v1/chat/completions"
    prompt = _prompt_for_mode(question.prompt, PROMPT_MODE_STANDARD)

    def _payload(
        messages: list[dict[str, Any]],
        max_tokens: int,
        *,
        grammar: str | None = None,
    ) -> dict[str, Any]:
        p: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if disable_thinking:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        if grammar is not None:
            p["grammar"] = grammar
        if seed is not None:
            p["seed"] = seed
        return p

    row: dict[str, Any] = {
        "question_id": question.id,
        "prompt": prompt,
        "expected": question.expected,
        "port": port,
        "role": role,
        "endpoint": "chat",
        "prompt_mode": PROMPT_MODE_STANDARD,
        "two_phase": True,
        "reasoning_budget": reasoning_budget,
        # Two-phase RE-4 requests do not ask llama-server for probabilities.
        # Keep the absence explicit and fail closed for calibration consumers.
        "confidence": None,
        "confidence_is_real": False,
        "confidence_source": "not_collected",
    }
    try:
        # ── Phase 1: free CoT, no grammar ─────────────────────────────────────
        p1_messages = [{"role": "user", "content": prompt}]
        data1 = _http_json(url, _payload(p1_messages, reasoning_budget), timeout_s=timeout_s)
        text1 = _completion_text(data1)
        usage1 = data1.get("usage") or {}
        timings1 = data1.get("timings") or {}
        reasoning_tokens = (
            usage1.get("completion_tokens")
            or timings1.get("predicted_n")
            or len(text1.split())
        )
        prompt_tokens = usage1.get("prompt_tokens") or timings1.get("prompt_n")

        # ── Short-circuit test (B): Phase-1 already has a marker ──────────────
        if SOLUTION_MARKER_RE.search(text1):
            response = text1
            phase2_used = False
            final_answer_tokens = 0
            usage2: dict[str, Any] = {}
        else:
            # ── Phase 2: forced final line (grammar on THIS turn only) ────────
            p2_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text1},
                {"role": "user", "content": FINAL_ANSWER_INSTRUCTION},
            ]
            data2 = _http_json(
                url,
                _payload(
                    p2_messages,
                    final_answer_max_tokens,
                    grammar=SOLUTION_MARKER_GRAMMAR,
                ),
                timeout_s=timeout_s,
            )
            text2 = _completion_text(data2)
            usage2 = data2.get("usage") or {}
            timings2 = data2.get("timings") or {}
            final_answer_tokens = (
                usage2.get("completion_tokens")
                or timings2.get("predicted_n")
                or len(text2.split())
            )
            # Last-marker anchor lands on the forced line; reasoning stays in row.
            response = text1.rstrip() + "\n" + text2.strip()
            phase2_used = True

        elapsed_s = time.time() - start
        reasoning_tokens = int(reasoning_tokens or 0)
        final_answer_tokens = int(final_answer_tokens or 0)
        completion_tokens = reasoning_tokens + final_answer_tokens
        tps = (float(completion_tokens) / elapsed_s) if completion_tokens and elapsed_s > 0 else None
        row.update(
            {
                "response": response,
                "success": True,
                "elapsed_s": elapsed_s,
                "tokens_per_second": tps,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "final_answer_tokens": final_answer_tokens,
                # Explicit phase-labelled accounting so the accuracy-vs-token
                # ladder curve is computable from the artifacts alone.
                "phase1_tokens": reasoning_tokens,
                "phase2_tokens": final_answer_tokens,
                "total_tokens": completion_tokens,
                "phase2_used": phase2_used,
                "text1_len": len(text1),
                "raw_usage": usage1,
                "raw_usage_phase2": usage2,
            }
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        row.update(
            {
                "response": "",
                "success": False,
                "elapsed_s": time.time() - start,
                "tokens_per_second": None,
                "error": str(exc),
                "reasoning_tokens": 0,
                "final_answer_tokens": 0,
                "phase1_tokens": 0,
                "phase2_tokens": 0,
                "total_tokens": 0,
                "phase2_used": False,
                "text1_len": 0,
                "error_type": "infra_error",
                "excluded_from_scoring": True,
                "exclusion_reason": "infra_error",
            }
        )
    return question.id, row


def _write_role_result(
    path: Path,
    *,
    run_id: str,
    role: str,
    rows: dict[str, dict[str, Any]],
    model_path: str | None = None,
    started_at: str | None = None,
) -> None:
    tps = [
        float(row["tokens_per_second"])
        for row in rows.values()
        if row.get("tokens_per_second") is not None
    ]
    payload = {
        "model_role": role,
        "model_path": model_path,
        "config_name": "baseline",
        "run_id": run_id,
        "timestamp": started_at or _utcnow_iso(),
        "summary": {
            "avg_tokens_per_second": (sum(tps) / len(tps)) if tps else None,
            "avg_algorithmic_score": None,
            "questions_tested": len(rows),
            "questions_passed": 0,
        },
        "results": {SUITE_NAME: dict(sorted(rows.items()))},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_role(
    *,
    role: str,
    ports: list[int],
    questions: list[Question],
    output_path: Path,
    run_id: str,
    host: str = "127.0.0.1",
    endpoint: str = "chat",
    max_tokens: int = 2048,
    temperature: float = 0.6,
    timeout_s: int = 900,
    disable_thinking: bool = True,
    prompt_mode: str = PROMPT_MODE_STANDARD,
    force_solution_grammar: bool = False,
    resume: bool = True,
    two_phase: bool = False,
    reasoning_budget: int | None = None,
    final_answer_max_tokens: int = 64,
    seed: int | None = None,
) -> dict[str, Any]:
    unhealthy = [p for p in ports if not _health_ok(p, host=host)]
    if unhealthy:
        raise RuntimeError(f"{role}: unhealthy stack ports: {unhealthy}")

    # In two-phase mode the reasoning budget is the Phase-1 cap and overrides
    # --max-tokens; fall back to max_tokens when no budget was supplied.
    phase1_cap = reasoning_budget if reasoning_budget is not None else max_tokens

    started_at = _utcnow_iso()
    rows: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        rows.update(existing.get("results", {}).get(SUITE_NAME, {}))

    pending = [q for q in questions if q.id not in rows]
    with cf.ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures: dict[cf.Future[tuple[str, dict[str, Any]]], Question] = {}
        for idx, question in enumerate(pending):
            port = ports[idx % len(ports)]
            if two_phase:
                fut = pool.submit(
                    _run_question_two_phase,
                    host=host,
                    port=port,
                    role=role,
                    question=question,
                    reasoning_budget=phase1_cap,
                    final_answer_max_tokens=final_answer_max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    disable_thinking=disable_thinking,
                    seed=seed,
                )
            else:
                fut = pool.submit(
                    _run_question,
                    host=host,
                    port=port,
                    role=role,
                    question=question,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    endpoint=endpoint,
                    disable_thinking=disable_thinking,
                    prompt_mode=prompt_mode,
                    force_solution_grammar=force_solution_grammar,
                    seed=seed,
                )
            futures[fut] = question
        for fut in cf.as_completed(futures):
            qid, row = fut.result()
            rows[qid] = row
            _write_role_result(output_path, run_id=run_id, role=role, rows=rows, started_at=started_at)

    _write_role_result(output_path, run_id=run_id, role=role, rows=rows, started_at=started_at)
    failures = [qid for qid, row in rows.items() if not row.get("success", True)]
    result: dict[str, Any] = {
        "role": role,
        "ports": ports,
        "output_path": str(output_path),
        "questions": len(questions),
        "rows": len(rows),
        "failures": failures,
    }
    if two_phase:
        p2_flags = [bool(r.get("phase2_used")) for r in rows.values() if "phase2_used" in r]
        rtoks = [
            r["reasoning_tokens"]
            for r in rows.values()
            if r.get("reasoning_tokens") is not None
        ]
        result["phase2_used_rate"] = (sum(p2_flags) / len(p2_flags)) if p2_flags else None
        result["mean_reasoning_tokens"] = (sum(rtoks) / len(rtoks)) if rtoks else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role-ports", action="append", required=True)
    parser.add_argument("--output-dir", default="benchmarks/results/runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--endpoint", choices=("chat", "completion"), default="chat")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-thinking", action="store_true")
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default=PROMPT_MODE_STANDARD)
    parser.add_argument("--force-solution-grammar", action="store_true")
    parser.add_argument("--summary-out", default=None)
    # RE-4 v2 two-phase protocol (free CoT → forced solution= terminal turn).
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=None,
        help="Phase-1 max_tokens under --two-phase; overrides --max-tokens.",
    )
    parser.add_argument("--final-answer-max-tokens", type=int, default=64)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed threaded into every payload (production: 42). "
        "Omitted → no seed key (v1 payloads byte-identical).",
    )
    probe = parser.add_mutually_exclusive_group()
    probe.add_argument(
        "--probe-ids",
        default=None,
        help="Path to a newline-delimited question_id allowlist (probe subset).",
    )
    probe.add_argument(
        "--limit-per-domain",
        type=int,
        default=None,
        help="Keep the first N rows per domain by sorted question_id (probe).",
    )
    args = parser.parse_args(argv)

    suite = load_suite(SUITE_NAME)
    if suite is None:
        raise SystemExit(f"missing suite {SUITE_NAME}")
    questions = list(suite.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    # Deterministic probe subsetting (no sampling): explicit id list, or the
    # stratified first-N-per-domain slice used by the non-saturation probe.
    if args.probe_ids:
        questions = _select_questions(
            questions, probe_ids=_read_probe_ids(Path(args.probe_ids))
        )
    elif args.limit_per_domain is not None:
        questions = _select_questions(
            questions,
            limit_per_domain=args.limit_per_domain,
            domain_map=_load_domain_map(SUITE_NAME),
        )
    role_ports = _parse_role_ports(args.role_ports)

    run_root = Path(args.output_dir) / args.run_id
    summary = {
        "schema_version": "longcot_mini_stack_runner.v1",
        "run_id": args.run_id,
        "suite": SUITE_NAME,
        "started_at": _utcnow_iso(),
        "endpoint": args.endpoint,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "prompt_mode": args.prompt_mode,
        "force_solution_grammar": args.force_solution_grammar,
        "roles": [],
    }
    # v2 additions are gated so a flags-absent (v1) invocation emits an
    # unchanged summary schema.
    if args.two_phase:
        summary["two_phase"] = True
        summary["reasoning_budget"] = (
            args.reasoning_budget if args.reasoning_budget is not None else args.max_tokens
        )
        summary["final_answer_max_tokens"] = args.final_answer_max_tokens
    if args.seed is not None:
        summary["seed"] = args.seed
    exit_code = 0
    for role, ports in role_ports.items():
        output_path = run_root / f"{role}_baseline.json"
        role_summary = run_role(
            role=role,
            ports=ports,
            questions=questions,
            output_path=output_path,
            run_id=args.run_id,
            host=args.host,
            endpoint=args.endpoint,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
            disable_thinking=not args.allow_thinking,
            prompt_mode=args.prompt_mode,
            force_solution_grammar=args.force_solution_grammar,
            resume=not args.no_resume,
            two_phase=args.two_phase,
            reasoning_budget=args.reasoning_budget,
            final_answer_max_tokens=args.final_answer_max_tokens,
            seed=args.seed,
        )
        summary["roles"].append(role_summary)
        if role_summary["failures"]:
            exit_code = 2
    summary["ended_at"] = _utcnow_iso()
    summary_path = Path(args.summary_out) if args.summary_out else run_root / "stack_runner_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
