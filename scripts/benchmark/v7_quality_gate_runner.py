#!/usr/bin/env python3
"""V7 quality-gate runner: evaluate MMLU-Pro + GPQA-Diamond on a kernel.

Samples questions from registered suites, queries a llama-server instance,
scores multiple-choice answers, and writes per-suite accuracy JSON ready for
v7_quality_gate_compare.py.

Usage:
    v7_quality_gate_runner.py --port 18072 --output results.json \
        --suites mmlu_pro gpqa --n 200 --seed 42 --endpoint chat

Output JSON shape:
    {
      "meta": {"kernel": "v7-experimental", "binary": "...", "models": "...", "timestamp": "..."},
      "suites": [
        {"suite": "mmlu_pro", "accuracy": 0.82, "n": 200, "correct": 164,
         "per_tier": {"1": {"accuracy": 0.85, "n": 50, "correct": 42}, ...}},
        {"suite": "gpqa", "accuracy": 0.63, "n": 100, "correct": 63, ...}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REQUEST_TIMEOUT_S = int(os.environ.get("RUNNER_REQUEST_TIMEOUT_S", "1800"))


def wait_for_server(url: str, timeout: int = 120) -> None:
    """Wait for llama-server /health to return ok."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode().strip().lower()
                if "ok" in body:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server at {url} did not become healthy within {timeout}s")


def query_server(*args, **kwargs) -> str:
    """Backwards-compatible wrapper: response text only."""
    return query_server_meta(*args, **kwargs)["text"]


def query_server_meta(
    url: str,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 42,
    endpoint: str = "chat",
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
) -> dict:
    """Query llama-server; return text plus why generation stopped.

    finish_reason matters: a response cut off at max_tokens scores wrong for a
    budget reason, not a reasoning reason, and must be counted separately.
    """
    if endpoint == "chat":
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if enable_thinking is not None:
            # enable_thinking is only honoured on the /v1/chat/completions path.
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        request_path = "/v1/chat/completions"
    elif endpoint == "completion":
        payload = {
            "model": "",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "top_k": 1,
            "logprobs": 0,
        }
        request_path = "/v1/completions"
    else:
        raise ValueError(f"unsupported endpoint: {endpoint}")

    req = urllib.request.Request(
        f"{url}{request_path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            result = json.loads(resp.read().decode())
            choices = result.get("choices", [])
            if not choices:
                return {"text": "", "reasoning": "", "finish_reason": "no_choices",
                        "completion_tokens": 0, "error": ""}
            choice = choices[0]
            if endpoint == "chat":
                message = choice.get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                text = str(content or "")
            else:
                text = choice.get("text", "")
            reasoning = ""
            if endpoint == "chat":
                reasoning = str(message.get("reasoning_content") or "")
            usage = result.get("usage", {}) or {}
            timings = result.get("timings", {}) or {}
            return {
                "text": text,
                "reasoning": reasoning,
                "finish_reason": choice.get("finish_reason", ""),
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                # server-reported per-request rates (single-request view)
                "decode_tok_s": timings.get("predicted_per_second", 0.0),
                "prompt_tok_s": timings.get("prompt_per_second", 0.0),
                "error": "",
            }
    except Exception as e:
        print(f"  [runner] query failed: {e}", file=sys.stderr)
        return {"text": "", "reasoning": "", "finish_reason": "request_error",
                "completion_tokens": 0, "error": str(e)[:300]}


# Canonical scoring primitives live in answer_scoring (single source; see
# handoffs/active/scoring-infra-standardization.md). Re-exported here so this
# module's public scoring API is unchanged for existing importers.
from answer_scoring import (  # noqa: F401
    extract_letter_answer, _normalize_numeric, parse_math_number,
    _latex_to_sympy_str, _sympy_expr, _split_top,
    _canon_elem, _is_set_answer, score_math_symbolic,
    gold_symbolically_parseable, extract_boxed, score_math_numeric,
    _first_pattern_match, extract_exact_answer, score_response,
)

def load_questions(
    suite_name: str,
    n: int,
    seed: int,
    stratify: bool = False,
    questions_in: Path | None = None,
    limit: int = 0,
) -> list:
    """Sample a suite's questions, or replay a previously pinned item set.

    Pinning matters for the architect bench: arms are compared paired, and
    the CPU arm runs in a later session. Re-sampling there would silently
    change the item set and break the pairing, so the first arm writes the
    manifest and every later arm replays it verbatim.
    """
    if questions_in is not None:
        pinned = json.loads(Path(questions_in).read_text())
        items = pinned["suites"][suite_name] if "suites" in pinned else pinned
        return items[:limit] if limit else items

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from dataset_adapters import get_adapter
    except ImportError:
        sys.path.insert(
            0,
            str(Path(__file__).resolve().parent.parent / "benchmark"),
        )
        from dataset_adapter_modules.registry import get_adapter

    adapter = get_adapter(suite_name)
    if adapter is None:
        return []
    return adapter.sample(n=n, seed=seed, stratify=stratify)


def run_suite(
    suite_name: str,
    url: str,
    n: int,
    seed: int,
    stratify: bool = False,
    max_tokens: int = 64,
    endpoint: str = "chat",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
    repeats: int = 1,
    per_question_out=None,
    questions_in: Path | None = None,
    limit: int = 0,
    arm: str = "",
    concurrency: int = 1,
) -> dict:
    """Run eval on a single suite and return per-suite results."""
    questions = load_questions(suite_name, n, seed, stratify, questions_in, limit)
    if not questions:
        return {"suite": suite_name, "accuracy": 0, "n": 0, "correct": 0,
                "error": "no questions sampled"}

    correct = 0
    total = 0
    errors = 0
    truncated = 0
    per_tier: dict[str, dict] = {}
    per_item: dict[str, dict] = {}
    tok_acc = [0, 0]  # [completion, prompt] tokens generated THIS run (excludes resumed)
    suite_t0 = time.monotonic()

    # Idempotent resume: never re-query a (suite, question, seed) already on
    # disk. Lets an interrupted run resume, and an avg@k top-up add only new
    # seeds, without re-spending inference on results already collected. A
    # shared JSONL contains multiple suites, so rows from another suite must
    # neither suppress this suite's work nor be folded into its counters.
    already: set = set()
    if per_question_out is not None:
        done_path = getattr(per_question_out, "name", None)
        if done_path and Path(done_path).exists():
            for line in Path(done_path).read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("suite") != suite_name:
                    continue
                key = (r.get("id"), r.get("seed"))
                if key in already:
                    continue
                already.add(key)
                tier = str(r.get("tier", 2))
                qid = r.get("id", "")
                per_tier.setdefault(tier, {"correct": 0, "n": 0})
                per_item.setdefault(qid, {"correct": 0, "n": 0})
                per_tier[tier]["n"] += 1
                per_item[qid]["n"] += 1
                total += 1
                if r.get("correct"):
                    correct += 1
                    per_tier[tier]["correct"] += 1
                    per_item[qid]["correct"] += 1
                if r.get("empty_response"):
                    errors += 1
                if r.get("truncated"):
                    truncated += 1
            if already:
                print(f"  [runner] resume: {len(already)} (id,seed) draws already "
                      f"on disk — folded in, not re-queried", file=sys.stderr)

    # Repeats are the OUTER loop on purpose. Iterating questions outermost
    # would mean an interrupted avg@k run had all k draws for early questions
    # and none for late ones -- a subset-biased score. Sweeping the full
    # question set once per repeat instead means every completed pass is a
    # valid avg@1, and partial work degrades to avg@(k-1) plus a fragment.
    for rep in range(repeats):
        # Distinct seed per repeat so avg@k samples k independent draws
        # instead of re-running one deterministic path k times.
        rep_seed = seed + rep
        if repeats > 1:
            print(f"  [runner] {suite_name}: pass {rep+1}/{repeats} "
                  f"(seed {rep_seed})", file=sys.stderr)

        # Build this pass's work list, skipping (id,seed) already on disk.
        pending = []
        for i, q in enumerate(questions):
            # Do NOT uppercase: fine for A-J / digits, but corrupts LaTeX gold
            # (\frac -> \FRAC). score_response applies case rules per method.
            expected = str(q.get("expected", "")).strip()
            if not expected:
                continue
            qid = q.get("id", f"{suite_name}_{i:04d}")
            if (qid, rep_seed) in already:
                continue  # already collected — never re-query (idempotent resume)
            pending.append((i, q, qid, expected))

        def _work(item):
            """Pure: query + score one question. No shared-state mutation, so it
            runs safely in a thread pool; the main thread applies the record."""
            i, q, qid, expected = item
            meta = query_server_meta(
                url, q["prompt"], max_tokens=max_tokens, temperature=temperature,
                seed=rep_seed, endpoint=endpoint, top_p=top_p, top_k=top_k,
                enable_thinking=enable_thinking)
            response = meta["text"]
            if not response:
                is_correct, got = False, ""
            else:
                is_correct = score_response(response, expected, q)
                _method = q.get("scoring_method", "multiple_choice")
                if _method == "multiple_choice":
                    got = extract_letter_answer(response)
                elif _method in ("math_numeric", "math_symbolic"):
                    got = extract_boxed(response)
                else:
                    got = extract_exact_answer(response, q.get("scoring_config", {}) or {})
            return (i, q, qid, expected, meta, response, is_correct, got)

        def _apply(res):
            """Main-thread only: fold one result into counters + JSONL."""
            nonlocal correct, total, errors, truncated
            i, q, qid, expected, meta, response, is_correct, got = res
            tier = str(q.get("tier", 2))
            per_tier.setdefault(tier, {"correct": 0, "n": 0})
            per_item.setdefault(qid, {"correct": 0, "n": 0})
            per_tier[tier]["n"] += 1
            per_item[qid]["n"] += 1
            total += 1
            if meta.get("finish_reason") == "length":
                truncated += 1
            if not response:
                errors += 1
            elif is_correct:
                correct += 1
                per_tier[tier]["correct"] += 1
                per_item[qid]["correct"] += 1
            if per_question_out is not None:
                # Written per result, not at completion: an interrupted run keeps
                # everything collected so far.
                per_question_out.write(json.dumps({
                    "arm": arm, "suite": suite_name, "id": qid, "tier": tier,
                    "rep": rep, "seed": rep_seed, "expected": expected,
                    "extracted": got, "correct": bool(is_correct),
                    "empty_response": not response,
                    "finish_reason": meta.get("finish_reason", ""),
                    "truncated": meta.get("finish_reason") == "length",
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "decode_tok_s": round(meta.get("decode_tok_s", 0.0), 2),
                    "request_error": meta.get("error", ""),
                    "reasoning_chars": len(meta.get("reasoning") or ""),
                    "empty_content_with_reasoning": (
                        not response and bool(meta.get("reasoning"))),
                    "response": response[-4000:],
                }) + "\n")
                per_question_out.flush()
            tok_acc[0] += meta.get("completion_tokens", 0)
            tok_acc[1] += meta.get("prompt_tokens", 0)

        if concurrency > 1 and len(pending) > 1:
            # Client-side concurrency; server serves them from its -np slots.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = {pool.submit(_work, it): it for it in pending}
                for n, fut in enumerate(as_completed(futs), 1):
                    _apply(fut.result())  # main thread — no lock needed
                    if n % 25 == 0:
                        print(f"  [runner] {suite_name}: pass {rep+1}/{repeats} "
                              f"{n}/{len(pending)} ({correct}/{total} correct so far)",
                              file=sys.stderr)
        else:
            for n, it in enumerate(pending, 1):
                _apply(_work(it))
                if n % 25 == 0:
                    print(f"  [runner] {suite_name}: pass {rep+1}/{repeats} "
                          f"{n}/{len(pending)} ({correct}/{total} correct so far)",
                          file=sys.stderr)

    accuracy = correct / total if total > 0 else 0.0

    tier_results = {}
    for t, data in sorted(per_tier.items()):
        tier_results[t] = {
            "accuracy": data["correct"] / data["n"] if data["n"] > 0 else 0,
            "n": data["n"],
            "correct": data["correct"],
        }

    suite_wall = time.monotonic() - suite_t0
    return {
        "suite": suite_name,
        "accuracy": accuracy,
        "n": total,
        "n_questions": len(per_item),
        "repeats": repeats,
        "correct": correct,
        "errors": errors,
        "truncated": truncated,
        "per_tier": tier_results,
        "per_item": per_item,
        # Throughput (this run only; excludes resumed draws). Aggregate =
        # tokens generated across all concurrent slots / wall-clock.
        "throughput": {
            "concurrency": concurrency,
            "wall_s": round(suite_wall, 1),
            "completion_tokens": tok_acc[0],
            "prompt_tokens": tok_acc[1],
            "aggregate_decode_tok_s": round(tok_acc[0] / suite_wall, 1) if suite_wall > 0 else 0,
            "aggregate_total_tok_s": round((tok_acc[0] + tok_acc[1]) / suite_wall, 1) if suite_wall > 0 else 0,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="V7 quality-gate runner")
    p.add_argument("--port", type=int, default=18072,
                   help="llama-server port (default: 18072)")
    p.add_argument("--host", default="localhost",
                   help="llama-server host (default: localhost)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON path")
    p.add_argument("--suites", nargs="+", default=["mmlu_pro", "gpqa"],
                   help="Suites to evaluate (default: mmlu_pro gpqa)")
    p.add_argument("--n", type=int, default=200,
                   help="Questions per suite (default: 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling (default: 42)")
    p.add_argument("--stratify", action="store_true",
                   help="Use stratified sampling (equal per tier)")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Max tokens for model response (default: 64)")
    p.add_argument("--endpoint", choices=["chat", "completion"], default="chat",
                   help="llama-server API endpoint mode (default: chat)")
    p.add_argument("--kernel", default="v7-candidate",
                   help="Kernel label for output metadata")
    p.add_argument("--binary", default="",
                   help="Binary path for output metadata")
    p.add_argument("--models", default="",
                   help="Model path(s) for output metadata")
    p.add_argument("--timeout", type=int, default=120,
                   help="Server health check timeout (seconds)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0). Sampling-sensitive "
                        "suites should use the production temperature, not 0.")
    p.add_argument("--top-p", type=float, default=None,
                   help="Sampling top_p (default: server default)")
    p.add_argument("--top-k", type=int, default=None,
                   help="Sampling top_k (default: server default)")
    p.add_argument("--enable-thinking", dest="enable_thinking",
                   action="store_true", default=None,
                   help="Send chat_template_kwargs.enable_thinking=true")
    p.add_argument("--no-enable-thinking", dest="enable_thinking",
                   action="store_false",
                   help="Send chat_template_kwargs.enable_thinking=false")
    p.add_argument("--repeats", type=int, default=1,
                   help="Draws per question (avg@k). Each repeat uses seed+rep.")
    p.add_argument("--per-question-out", type=Path, default=None,
                   help="JSONL path for per-question records (written incrementally; "
                        "required for paired/McNemar analysis)")
    p.add_argument("--questions-out", type=Path, default=None,
                   help="Write the sampled item set here so later arms can replay it")
    p.add_argument("--questions-in", type=Path, default=None,
                   help="Replay a pinned item set instead of sampling (paired arms)")
    p.add_argument("--limit", type=int, default=0,
                   help="Use only the first N items of a pinned set (ablations)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Concurrent in-flight requests (client-side); match to the "
                        "server's -np slots. 1 = sequential.")
    p.add_argument("--arm", default="",
                   help="Arm label recorded in per-question records")
    args = p.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"[runner] Waiting for server at {url}...", file=sys.stderr)
    wait_for_server(url, timeout=args.timeout)
    print(f"[runner] Server healthy", file=sys.stderr)

    # Determine binary path
    binary = args.binary or os.environ.get("LLAMA_BINARY", "")

    if args.questions_out:
        pinned = {
            s: load_questions(s, args.n, args.seed, args.stratify)
            for s in args.suites
        }
        args.questions_out.write_text(json.dumps({"suites": pinned}, indent=2))
        print(f"[runner] Pinned item set written to {args.questions_out}",
              file=sys.stderr)

    pq_handle = None
    if args.per_question_out:
        args.per_question_out.parent.mkdir(parents=True, exist_ok=True)
        pq_handle = args.per_question_out.open("a")

    suites_results = []
    start = time.monotonic()

    questions_in = args.questions_in or args.questions_out

    for suite in args.suites:
        print(f"\n[runner] Evaluating {suite} (n={args.n}, seed={args.seed})...",
              file=sys.stderr)
        result = run_suite(
            suite, url, args.n, args.seed,
            stratify=args.stratify,
            max_tokens=args.max_tokens,
            endpoint=args.endpoint,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_thinking=args.enable_thinking,
            repeats=args.repeats,
            per_question_out=pq_handle,
            questions_in=questions_in,
            limit=args.limit,
            arm=args.arm,
            concurrency=args.concurrency,
        )
        suites_results.append(result)
        acc = result.get("accuracy", 0)
        n = result.get("n", 0)
        print(f"[runner] {suite}: {acc:.1%} ({result.get('correct',0)}/{n})",
              file=sys.stderr)

    elapsed = time.monotonic() - start
    if pq_handle is not None:
        pq_handle.close()

    output = {
        "meta": {
            "kernel": args.kernel,
            "binary": binary,
            "models": args.models or "unknown",
            "arm": args.arm,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 1),
            "n_per_suite": args.n,
            "seed": args.seed,
            "stratify": args.stratify,
            "endpoint": args.endpoint,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "enable_thinking": args.enable_thinking,
            "repeats": args.repeats,
            "max_tokens": args.max_tokens,
            "questions_pinned": str(questions_in) if questions_in else "",
        },
        "suites": suites_results,
    }

    args.output.write_text(json.dumps(output, indent=2))
    print(f"\n[runner] Results written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
