#!/usr/bin/env python3
"""TALE dynamic budget estimation evaluation.

Compares three brevity strategies on existing eval suites:

1. Baseline — no brevity constraint
2. Static word limits — Action 12 format templates (50w math, 60w general)
3. TALE-EP self-estimated budget — model estimates its own budget (zero-shot
   pre-pass), then answers "{question} Let's think step by step and use less
   than {beta} tokens". ``--budget-unit words`` restores the legacy
   "Answer in under {beta} words" variant.

Measures accuracy, token count, latency and OAA/PTI per condition. The TALE
estimator call is charged: every row carries answer-only cost
(``total_tokens``/``elapsed_s``, comparable with pre-PRB-T4 rows) and
``*_incl_estimator`` totals. Run metadata (endpoint, served model/GGUF identity,
temperature, seed, budget unit) goes to ``<output>.meta.json``; per suite x
condition aggregates to ``<output>.summary.json``.

Reference: TALE (Han et al., "Token-Budget-Aware LLM Reasoning",
arXiv:2412.18547).

Usage:
    python eval_tale_budget.py --suites math general --n-questions 20
    python eval_tale_budget.py --suites math --model-port 8080 --dry-run
    python eval_tale_budget.py --endpoint http://127.0.0.1:8083 \
        --suites math --temperature 0.2 --chat-template-kwargs '{"enable_thinking": false}'
"""

# Scoring: delegates to orchestrator B7 debug_scorer (2026-08-23) — no pre-B7 semantics

from __future__ import annotations

import argparse
import hashlib
import os
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
POOL_PATH = SCRIPT_DIR.parent.parent / "benchmarks" / "prompts" / "question_pool.jsonl"
RESULTS_DIR = SCRIPT_DIR.parent.parent / "data" / "tale_budget"

# Static word limits from Action 12 (per suite type)
STATIC_LIMITS = {
    "math": "Show essential steps only, under 50 words. Final answer on last line.",
    "gsm8k": "Show essential steps only, under 50 words. Final answer on last line.",
    "general": "Answer in under 60 words, key point only. No preamble.",
    "hotpotqa": "Answer in under 60 words, key point only. No preamble.",
    "gpqa": "Letter + ONE sentence justification.",
    "agentic": "Answer in under 60 words, key point only. No preamble.",
    "coder": "Output code only.",
}

# Legacy word-unit prompts (pre-PRB-T4). Kept behind ``--budget-unit words`` so
# earlier numbers stay reproducible; they are NOT the published intervention.
TALE_PREPASS_PROMPT_WORDS = (
    "Estimate how many words you need to answer this question correctly.\n"
    "Reply with ONLY a number.\n\n"
    "Question: {question}\n\n"
    "Words needed:"
)
TALE_CONSTRAINT_TEMPLATE_WORDS = "Answer in under {beta} words.\n\n"

# Token-unit prompts following TALE-EP (Han et al., "Token-Budget-Aware LLM
# Reasoning", arXiv:2412.18547): zero-shot token-budget estimate, then the
# question followed by "Let's think step by step and use less than {budget}
# tokens". Default since PRB-T4.
TALE_PREPASS_PROMPT_TOKENS = (
    "Task: Analyze the given question and estimate the minimum number of "
    "tokens required to generate a complete and accurate response. "
    "Please give the response by strictly following this format: [[budget]], "
    "for example, Budget: [[12]].\n\n"
    "Question: {question}"
)
TALE_CONSTRAINT_TEMPLATE_TOKENS = (
    "\n\nLet's think step by step and use less than {beta} tokens."
)

# Back-compat aliases (legacy names referred to the word prompts).
TALE_PREPASS_PROMPT = TALE_PREPASS_PROMPT_WORDS
TALE_CONSTRAINT_TEMPLATE = TALE_CONSTRAINT_TEMPLATE_WORDS

BUDGET_UNITS = ("tokens", "words")
# (min, max, fallback) per unit. Words keeps the original [10, 500] / 60.
BUDGET_CLAMPS = {"words": (10, 500, 60), "tokens": (10, 4096, 100)}
ESTIMATOR_MAX_TOKENS = 32

# Injectable HTTP seams: poster(url, json_body, timeout) -> response dict;
# fetcher(url, timeout) -> response dict. Defaults use httpx lazily.
Poster = Callable[[str, dict, float], dict]
Fetcher = Callable[[str, float], dict]


def _httpx_post(url: str, body: dict, timeout: float) -> dict:
    import httpx

    resp = httpx.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _httpx_get(url: str, timeout: float) -> dict:
    import httpx

    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def resolve_base_url(endpoint: str | None, host: str = "localhost", port: int = 8080) -> str:
    """Return an OpenAI-compatible base URL without trailing ``/`` or ``/v1``.

    ``--endpoint`` wins; otherwise ``http://{host}:{port}`` (legacy flags).
    """
    if not endpoint:
        return f"http://{host}:{port}"
    url = endpoint.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    if "://" not in url:
        url = f"http://{url}"
    return url


@dataclass
class TrialResult:
    question_id: str
    suite: str
    condition: str  # "baseline", "static", "tale"
    prompt: str
    response: str
    correct: bool | None = None
    # Answer-only cost (unchanged meaning; comparable with pre-PRB-T4 rows).
    total_tokens: int = 0
    elapsed_s: float = 0.0
    tale_budget: int | None = None  # Only for TALE condition
    budget_unit: str | None = None  # "tokens"|"words" for TALE, else None
    # Estimator pre-pass cost (TALE only; 0 for other arms).
    estimator_tokens: int = 0
    estimator_s: float = 0.0
    estimator_response: str | None = None
    # Total cost incl. the estimator call (== answer-only for non-TALE arms).
    total_tokens_incl_estimator: int = 0
    elapsed_s_incl_estimator: float = 0.0
    temperature: float | None = None
    seed: int | None = None
    served_model: str | None = None  # "model" field echoed by the server


@dataclass
class GenConfig:
    """Request configuration shared by every call in a run."""

    base_url: str = "http://localhost:8080"
    temperature: float = 0.0
    seed: int | None = 42
    max_tokens: int = 8192
    model: str | None = None
    chat_template_kwargs: dict | None = None
    budget_unit: str = "tokens"
    timeout: float = 1200.0
    poster: Poster | None = None


def load_questions(suites: list[str], n_questions: int) -> list[dict[str, Any]]:
    """Load questions from question_pool.jsonl filtered by suite."""
    if not POOL_PATH.exists():
        log.error("Question pool not found: %s", POOL_PATH)
        log.error("Run: python question_pool.py --build")
        sys.exit(1)

    questions: list[dict[str, Any]] = []
    suite_counts: dict[str, int] = {s: 0 for s in suites}

    with open(POOL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            if q.get("__pool_metadata__"):
                continue
            suite = q.get("suite", "")
            if suite not in suites:
                continue
            if suite_counts[suite] >= n_questions:
                continue
            questions.append(q)
            suite_counts[suite] += 1
            if all(c >= n_questions for c in suite_counts.values()):
                break

    log.info(
        "Loaded %d questions: %s",
        len(questions),
        ", ".join(f"{s}={c}" for s, c in suite_counts.items()),
    )
    return questions


def generate_response(
    prompt: str,
    host: str = "localhost",
    port: int = 8080,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    *,
    base_url: str | None = None,
    seed: int | None = None,
    model: str | None = None,
    chat_template_kwargs: dict | None = None,
    poster: Poster | None = None,
    timeout: float = 1200.0,
    meta: dict | None = None,
) -> tuple[str, int, float]:
    """Generate a response from an OpenAI-compatible server.

    Returns (response_text, completion_tokens, elapsed_seconds). If ``meta``
    is given it receives ``served_model`` and ``usage`` from the response.
    """
    url = f"{base_url or resolve_base_url(None, host, port)}/v1/chat/completions"
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    if model:
        body["model"] = model
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs

    post = poster or _httpx_post
    t0 = time.monotonic()
    data = post(url, body, timeout)
    elapsed = time.monotonic() - t0

    message = data["choices"][0]["message"]
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens", len(text) // 4)

    if reasoning:
        text = f"<think>\n{reasoning}\n</think>\n{text}"
    if meta is not None:
        meta["served_model"] = data.get("model")
        meta["usage"] = usage

    return text, completion_tokens, elapsed


def _gen(prompt: str, cfg: GenConfig, max_tokens: int | None = None,
         meta: dict | None = None) -> tuple[str, int, float]:
    return generate_response(
        prompt,
        max_tokens=cfg.max_tokens if max_tokens is None else max_tokens,
        temperature=cfg.temperature,
        base_url=cfg.base_url,
        seed=cfg.seed,
        model=cfg.model,
        chat_template_kwargs=cfg.chat_template_kwargs,
        poster=cfg.poster,
        timeout=cfg.timeout,
        meta=meta,
    )


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def score_question(answer: str, question: dict[str, Any]) -> bool:
    """Score an answer against expected. Uses debug_scorer if available."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from debug_scorer import score_answer

    return score_answer(
        answer=strip_think_blocks(answer),
        expected=question.get("expected", ""),
        scoring_method=question.get("scoring_method", "exact_match"),
        scoring_config=question.get("scoring_config"),
    )


@dataclass
class BudgetEstimate:
    budget: int
    unit: str
    tokens: int
    elapsed_s: float
    raw: str


def parse_budget(text: str, unit: str = "tokens") -> int | None:
    """Extract a budget: prefer ``[[N]]`` (TALE format), else first integer."""
    lo, hi, _ = BUDGET_CLAMPS[unit]
    clean = strip_think_blocks(text)
    match = re.search(r"\[\[\s*(\d+)\s*\]\]", clean) or re.search(r"\d+", clean)
    if not match:
        return None
    value = int(match.group(1) if match.groups() else match.group())
    return max(lo, min(value, hi))


def build_tale_prompt(question_text: str, beta: int, unit: str = "tokens") -> str:
    if unit == "tokens":
        return question_text + TALE_CONSTRAINT_TEMPLATE_TOKENS.format(beta=beta)
    if unit == "words":
        return TALE_CONSTRAINT_TEMPLATE_WORDS.format(beta=beta) + question_text
    raise ValueError(f"Unknown budget unit: {unit}")


def estimate_tale_budget_full(question_text: str, cfg: GenConfig) -> BudgetEstimate:
    """Run the TALE pre-pass and keep its cost (tokens + wall time)."""
    unit = cfg.budget_unit
    template = TALE_PREPASS_PROMPT_TOKENS if unit == "tokens" else TALE_PREPASS_PROMPT_WORDS
    prompt = template.format(question=question_text)
    text, tokens, elapsed = _gen(prompt, cfg, max_tokens=ESTIMATOR_MAX_TOKENS)
    budget = parse_budget(text, unit)
    if budget is None:
        budget = BUDGET_CLAMPS[unit][2]
        log.warning("TALE pre-pass returned no number: %r, defaulting to %d %s",
                    strip_think_blocks(text), budget, unit)
    return BudgetEstimate(budget, unit, int(tokens), elapsed, text)


def estimate_tale_budget(
    question_text: str,
    host: str = "localhost",
    port: int = 8080,
) -> int:
    """Legacy word-unit API: returns only the clamped budget (cost discarded)."""
    cfg = GenConfig(base_url=resolve_base_url(None, host, port),
                    temperature=0.0, seed=None, budget_unit="words")
    return estimate_tale_budget_full(question_text, cfg).budget


def run_trial(
    question: dict[str, Any],
    condition: str,
    cfg: GenConfig,
    estimate: BudgetEstimate | None = None,
) -> TrialResult:
    """Run a single trial (one question, one condition)."""
    q_text = question.get("prompt", question.get("question", ""))
    suite = question.get("suite", "unknown")
    q_id = question.get("id", question.get("question_id", "?"))

    if condition == "baseline":
        prompt = q_text
    elif condition == "static":
        limit = STATIC_LIMITS.get(suite, STATIC_LIMITS["general"])
        prompt = f"{limit}\n\n{q_text}"
    elif condition == "tale":
        if estimate is None:
            raise ValueError("tale condition requires a budget estimate")
        prompt = build_tale_prompt(q_text, estimate.budget, estimate.unit)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    meta: dict[str, Any] = {}
    text, tokens, elapsed = _gen(prompt, cfg, meta=meta)
    correct = score_question(text, question)

    is_tale = condition == "tale"
    est_tok = estimate.tokens if is_tale else 0
    est_s = estimate.elapsed_s if is_tale else 0.0
    return TrialResult(
        question_id=str(q_id),
        suite=suite,
        condition=condition,
        prompt=prompt,
        response=text,
        correct=correct,
        total_tokens=int(tokens),
        elapsed_s=round(elapsed, 2),
        tale_budget=estimate.budget if is_tale else None,
        budget_unit=estimate.unit if is_tale else None,
        estimator_tokens=est_tok,
        estimator_s=round(est_s, 2),
        estimator_response=estimate.raw if is_tale else None,
        total_tokens_incl_estimator=int(tokens) + est_tok,
        elapsed_s_incl_estimator=round(elapsed + est_s, 2),
        temperature=cfg.temperature,
        seed=cfg.seed,
        served_model=meta.get("served_model"),
    )


def run_evaluation(
    questions: list[dict[str, Any]],
    conditions: list[str],
    cfg: GenConfig,
    dry_run: bool = False,
) -> list[TrialResult]:
    """Run all trials across questions and conditions."""
    results: list[TrialResult] = []

    for i, q in enumerate(questions):
        q_text = q.get("prompt", q.get("question", ""))
        q_id = q.get("id", q.get("question_id", "?"))
        suite = q.get("suite", "unknown")
        log.info("[%d/%d] %s q=%s", i + 1, len(questions), suite, q_id)

        if dry_run:
            for cond in conditions:
                log.info("  DRY-RUN %s: would send %d-char prompt", cond, len(q_text))
            continue

        # Estimate TALE budget once per question; its cost is charged to the tale row.
        estimate = None
        if "tale" in conditions:
            estimate = estimate_tale_budget_full(q_text, cfg)
            log.info("  TALE budget estimate: %d %s (%d tok, %.1fs)",
                     estimate.budget, estimate.unit, estimate.tokens, estimate.elapsed_s)

        for cond in conditions:
            result = run_trial(q, cond, cfg, estimate)
            results.append(result)
            status = "correct" if result.correct else "wrong"
            log.info(
                "  %s: %s, %d tokens (%d incl. est), %.1fs",
                cond, status, result.total_tokens,
                result.total_tokens_incl_estimator, result.elapsed_s,
            )

    return results


# ---------------------------------------------------------------------------
# Serving identity
# ---------------------------------------------------------------------------

def stat_gguf(path: str, do_hash: bool = False) -> dict[str, Any]:
    """Stat a local GGUF (size/mtime). SHA-256 only when explicitly asked."""
    info: dict[str, Any] = {"path": path}
    try:
        st = os.stat(path)
    except OSError as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["size_bytes"] = st.st_size
    info["mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
    if do_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
    return info


def fetch_serving_identity(
    base_url: str,
    fetcher: Fetcher | None = None,
    model_id: str | None = None,
    gguf_path: str | None = None,
    hash_gguf: bool = False,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Record which model actually serves ``base_url``.

    Queries ``/v1/models`` and llama-server ``/props`` (``model_path``);
    failures are recorded, not raised. ``model_id``/``gguf_path`` override.
    """
    get = fetcher or _httpx_get
    ident: dict[str, Any] = {"base_url": base_url, "errors": {}}

    try:
        models = get(f"{base_url}/v1/models", timeout)
        ids = [m.get("id") for m in (models.get("data") or []) if isinstance(m, dict)]
        ident["v1_models"] = ids
    except Exception as exc:  # noqa: BLE001 - identity probing is best-effort
        ident["errors"]["v1_models"] = f"{type(exc).__name__}: {exc}"
        ids = []

    props_path = None
    try:
        props = get(f"{base_url}/props", timeout)
        props_path = props.get("model_path")
        ident["props"] = {
            k: props.get(k)
            for k in ("model_path", "build_info", "total_slots", "chat_template_caps")
            if k in props
        }
        dgs = props.get("default_generation_settings") or {}
        if "n_ctx" in dgs:
            ident["props"]["n_ctx"] = dgs["n_ctx"]
    except Exception as exc:  # noqa: BLE001
        ident["errors"]["props"] = f"{type(exc).__name__}: {exc}"

    ident["model_id"] = model_id or (ids[0] if ids else None)
    ident["model_id_source"] = "override" if model_id else ("v1_models" if ids else None)
    path = gguf_path or props_path
    ident["gguf_path"] = path
    ident["gguf_path_source"] = "override" if gguf_path else ("props" if props_path else None)
    if path:
        ident["gguf"] = stat_gguf(path, do_hash=hash_gguf)
    if not ident["errors"]:
        del ident["errors"]
    return ident


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: list[TrialResult]) -> dict[str, Any]:
    """Per suite x condition cost/accuracy, answer-only AND incl. estimator.

    ``net_token_change_vs_baseline`` = mean(total incl. estimator) of the arm
    minus mean(answer tokens) of baseline, as a fraction of the latter.
    """
    out: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[TrialResult]] = {}
    for r in results:
        groups.setdefault((r.suite, r.condition), []).append(r)
        groups.setdefault(("__all__", r.condition), []).append(r)
    for (suite, cond), rows in sorted(groups.items()):
        scored = [r for r in rows if r.correct is not None]
        cell = {
            "n": len(rows),
            "accuracy": _mean([1.0 if r.correct else 0.0 for r in scored]),
            "mean_tokens_answer_only": _mean([r.total_tokens for r in rows]),
            "mean_tokens_incl_estimator": _mean([r.total_tokens_incl_estimator for r in rows]),
            "mean_elapsed_s_answer_only": _mean([r.elapsed_s for r in rows]),
            "mean_elapsed_s_incl_estimator": _mean([r.elapsed_s_incl_estimator for r in rows]),
            "mean_estimator_tokens": _mean([r.estimator_tokens for r in rows]),
            "mean_estimator_s": _mean([r.estimator_s for r in rows]),
        }
        budgets = [r.tale_budget for r in rows if r.tale_budget is not None]
        if budgets:
            sb = sorted(budgets)
            cell["budget"] = {"unit": rows[0].budget_unit, "min": sb[0], "max": sb[-1],
                              "median": sb[len(sb) // 2], "mean": _mean(budgets)}
        out.setdefault(suite, {})[cond] = cell
    for suite, conds in out.items():
        base = conds.get("baseline")
        if not base or not base["mean_tokens_answer_only"]:
            continue
        b = base["mean_tokens_answer_only"]
        for cond, cell in conds.items():
            cell["net_token_change_vs_baseline"] = (cell["mean_tokens_incl_estimator"] - b) / b
            cell["answer_token_change_vs_baseline"] = (cell["mean_tokens_answer_only"] - b) / b
            cell["accuracy_delta_pp_vs_baseline"] = 100.0 * (cell["accuracy"] - base["accuracy"])
    return out


def save_results(
    results: list[TrialResult],
    output_path: Path,
    run_meta: dict[str, Any] | None = None,
) -> None:
    """Save per-question JSONL plus ``.meta.json`` and ``.summary.json`` sidecars."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    if run_meta is not None:
        output_path.with_suffix(".meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summarize(results), indent=2) + "\n"
    )
    log.info("Saved %d results to %s", len(results), output_path)


def print_summary(results: list[TrialResult]) -> None:
    """Print per-condition summary with OAA/PTI."""
    from eval_metrics import compute_batch_oaa

    conditions = sorted(set(r.condition for r in results))
    print("\n" + "=" * 70)
    print("TALE Budget Evaluation Summary")
    print("=" * 70)

    for cond in conditions:
        cond_results = [asdict(r) for r in results if r.condition == cond]
        metrics = compute_batch_oaa(cond_results, alpha=0.5)
        n = len(cond_results)
        correct = sum(1 for r in cond_results if r.get("correct"))
        print(f"\n  {cond.upper()} ({n} questions, {correct} correct):")
        print(f"    Accuracy:   {metrics['accuracy']:.1%}")
        print(f"    OAA (a=.5): {metrics['oaa']:.4f}")
        print(f"    PTI:        {metrics['pti']:.6f}")
        print(f"    Avg tokens: {metrics['avg_tokens']:.0f} (answer-only)")
        tot = _mean([r["total_tokens_incl_estimator"] for r in cond_results])
        print(f"    Avg tokens: {tot:.0f} (total incl. estimator)")
        print(f"    Avg time:   {_mean([r['elapsed_s'] for r in cond_results]):.2f}s answer-only, "
              f"{_mean([r['elapsed_s_incl_estimator'] for r in cond_results]):.2f}s incl. estimator")
        print(f"    Ref tokens: {metrics['reference_tokens']}")

    # Per-suite breakdown
    suites = sorted(set(r.suite for r in results))
    if len(suites) > 1:
        print(f"\n{'─' * 70}")
        print("Per-suite breakdown:")
        for suite in suites:
            print(f"\n  [{suite}]")
            for cond in conditions:
                subset = [
                    asdict(r)
                    for r in results
                    if r.condition == cond and r.suite == suite
                ]
                if not subset:
                    continue
                metrics = compute_batch_oaa(subset, alpha=0.5)
                print(
                    f"    {cond:10s}: acc={metrics['accuracy']:.0%} "
                    f"oaa={metrics['oaa']:.3f} "
                    f"avg_tok={metrics['avg_tokens']:.0f} "
                    f"tot_tok={_mean([r['total_tokens_incl_estimator'] for r in subset]):.0f}"
                )

    # TALE budget distribution
    tale_results = [r for r in results if r.condition == "tale" and r.tale_budget]
    if tale_results:
        budgets = [r.tale_budget for r in tale_results]
        unit = tale_results[0].budget_unit or "words"
        print(f"\n{'─' * 70}")
        print(f"TALE budget distribution ({unit}): min={min(budgets)}, max={max(budgets)}, "
              f"median={sorted(budgets)[len(budgets)//2]}, "
              f"mean={sum(budgets)/len(budgets):.0f}")

    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TALE dynamic budget estimation evaluation",
    )
    parser.add_argument(
        "--suites", nargs="+", default=["math", "general"],
        help="Suite names to evaluate (default: math general)",
    )
    parser.add_argument(
        "--n-questions", type=int, default=20,
        help="Max questions per suite (default: 20)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["baseline", "static", "tale"],
        choices=["baseline", "static", "tale"],
        help="Conditions to run (default: all three)",
    )
    parser.add_argument(
        "--endpoint", "--base-url", dest="endpoint", default=None,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8083 "
             "(overrides --model-host/--model-port; a trailing /v1 is accepted)",
    )
    parser.add_argument(
        "--model-host", default="localhost",
        help="Model server host (default: localhost; ignored with --endpoint)",
    )
    parser.add_argument(
        "--model-port", type=int, default=8080,
        help="Model server port (default: 8080; ignored with --endpoint)",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Model id to record (and send as request 'model'); default: from /v1/models",
    )
    parser.add_argument(
        "--gguf-path", default=None,
        help="Local GGUF path to record (stat only); default: /props model_path",
    )
    parser.add_argument(
        "--hash-gguf", action="store_true",
        help="Also SHA-256 the GGUF (slow on multi-GB files)",
    )
    parser.add_argument(
        "--budget-unit", choices=BUDGET_UNITS, default="tokens",
        help="TALE budget unit: tokens (published TALE-EP, default) or legacy words",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for every call (default: 0.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Request seed (default: 42; pass -1 to omit)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="max_tokens for answer calls (default: 8192; estimator uses %d)" % ESTIMATOR_MAX_TOKENS,
    )
    parser.add_argument(
        "--chat-template-kwargs", default=None,
        help='JSON object sent as chat_template_kwargs, e.g. \'{"enable_thinking": false}\'',
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path (default: data/tale_budget/YYYYMMDD_HHMMSS.jsonl); "
             "writes .meta.json and .summary.json sidecars",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load questions and show prompts without sending to model",
    )
    return parser


def build_config(args: argparse.Namespace, poster: Poster | None = None) -> GenConfig:
    ctk = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    if ctk is not None and not isinstance(ctk, dict):
        raise SystemExit("--chat-template-kwargs must be a JSON object")
    return GenConfig(
        base_url=resolve_base_url(args.endpoint, args.model_host, args.model_port),
        temperature=args.temperature,
        seed=None if args.seed is not None and args.seed < 0 else args.seed,
        max_tokens=args.max_tokens,
        model=args.model_id,
        chat_template_kwargs=ctk,
        budget_unit=args.budget_unit,
        poster=poster,
    )


def main(argv: list[str] | None = None, poster: Poster | None = None,
         fetcher: Fetcher | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = build_config(args, poster)

    questions = load_questions(args.suites, args.n_questions)
    if not questions:
        log.error("No questions loaded. Check suite names and question_pool.jsonl.")
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — %d questions, conditions: %s, endpoint: %s",
                 len(questions), args.conditions, cfg.base_url)
        run_evaluation(questions, args.conditions, cfg, dry_run=True)

        # Show sample prompts for each condition
        sample = questions[0]
        q_text = sample.get("prompt", sample.get("question", ""))
        suite = sample.get("suite", "unknown")
        print(f"\n--- Sample prompts for suite={suite} ---\n")
        print(f"BASELINE:\n{q_text[:200]}...\n")
        limit = STATIC_LIMITS.get(suite, STATIC_LIMITS["general"])
        print(f"STATIC:\n{limit}\n\n{q_text[:200]}...\n")
        demo = 45 if cfg.budget_unit == "words" else 200
        print(f"TALE ({cfg.budget_unit}, assuming budget={demo}):\n"
              f"{build_tale_prompt(q_text[:200], demo, cfg.budget_unit)}\n")
        return

    identity = fetch_serving_identity(
        cfg.base_url, fetcher, model_id=args.model_id,
        gguf_path=args.gguf_path, hash_gguf=args.hash_gguf,
    )
    log.info("Serving identity: model_id=%s gguf=%s",
             identity.get("model_id"), identity.get("gguf_path"))
    run_meta: dict[str, Any] = {
        "schema": "tale_budget_run.v2",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv if argv is None else list(argv),
        "suites": args.suites,
        "n_questions": args.n_questions,
        "conditions": args.conditions,
        "budget_unit": cfg.budget_unit,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "max_tokens": cfg.max_tokens,
        "estimator_max_tokens": ESTIMATOR_MAX_TOKENS,
        "chat_template_kwargs": cfg.chat_template_kwargs,
        "serving": identity,
    }

    results = run_evaluation(questions, args.conditions, cfg)

    if not results:
        log.warning("No results generated.")
        return

    run_meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_meta["served_models_seen"] = sorted({r.served_model for r in results if r.served_model})

    # Save
    if args.output:
        output_path = args.output
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"{ts}.jsonl"
    save_results(results, output_path, run_meta)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
