#!/usr/bin/env python3
"""CME-1: judge a stored BEAM run once PER NUGGET and write the artifact ``score_beam_run.py`` folds.

Pipeline::

    run_benchmark.py (suite "beam")  ->  results["beam"][qid]["response"]
    judge_beam_run.py                ->  judged JSON (per-nugget 0 / 0.5 / 1 verdicts)
    score_beam_run.py                ->  BEAM-fold headline + SC68 belief sidecar

This is the one step in the pipeline that calls a model: the served judge, never
the model under test. Each nugget is one judge call made with
``beam_scoring.build_nugget_judge_prompt``, which passes the probing question
(CME-3), and each reply is parsed by ``beam_scoring.parse_nugget_verdict``. The
transport is the orchestrator's
``debug_scorer.request_llm_judge_text``, reached through this repo's
``debug_scorer`` delegation shim. It provides the same endpoint resolution and
the same fail-closed error taxonomy as every other llm_judge item. The boolean
``llm_judge`` path refuses ``per_nugget`` items, so it cannot binarise the scale.

Failure is recorded, never folded. A question whose judge call is unavailable, or
whose reply does not parse, is listed under ``unjudged`` with its reason and never
scored 0.0. ``score_beam_run.py`` refuses a payload whose ``unjudged`` list is not
empty. Verdicts are persisted per question to ``<out>.partial.jsonl``. A rerun
keeps the judged questions from that file and retries every unjudged one.

The judge identity (``--judge-model``) is required: a BEAM headline with no judge
recorded cannot be compared to anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).parent))

from beam_scoring import (  # noqa: E402
    BEAMFoldError,
    JUDGE_PROMPT_VERSION,
    NuggetVerdictError,
    QUESTION_IN_JUDGE_PROMPT,
    judge_question,
)

JUDGED_SCHEMA = "epyc.beam_judged_run.v1"
#: Enough room for BEAM's {"score", "reason"} reply; the score is all that is parsed.
NUGGET_MAX_TOKENS = 512
NUGGET_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "enum": [0.0, 0.5, 1.0]},
        "reason": {"type": "string"},
    },
    "required": ["score"],
}

Judge = Callable[[str], str]

# ── anthropic backend (HS-PROF-1, operator decision 2026-10-22: Claude via API, pinned id) ──
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"  # the ONLY place the key is read; never logged or recorded
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
ANTHROPIC_MAX_RETRIES = 6
ANTHROPIC_BACKOFF_BASE_S = 2.0
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 529})
USAGE_KEYS = ("input_tokens", "output_tokens", "calls")


class MissingApiKey(RuntimeError):
    """ANTHROPIC_API_KEY is unset or empty."""


class AnthropicUsage:
    """Running token totals plus the set of ``model`` values the API echoed back."""

    def __init__(self) -> None:
        self.totals = {k: 0 for k in USAGE_KEYS}
        self.response_models: set[str] = set()

    def snapshot(self) -> dict[str, int]:
        return dict(self.totals)


def _http_post(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    """The one network call (patched in tests). Returns (status, body); HTTP errors are returned, not raised."""
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def anthropic_judge(model_id: str, usage: AnthropicUsage, *, timeout: float | None = None,
                    post: Callable[..., tuple[int, bytes]] | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> Judge:
    """A ``judge(prompt) -> text`` over the Anthropic Messages API (stdlib only).

    The prompt is sent verbatim as the single user message, exactly as the served path
    sends it. temperature 0, bounded ``NUGGET_MAX_TOKENS``. 429/5xx and network errors
    are retried with exponential backoff, then surface as :class:`JudgeUnavailable`.
    The key is read from the environment at call time and appears in no message or record.
    """
    post = post or _http_post
    key = os.environ.get(ANTHROPIC_KEY_ENV, "")
    if not key.strip():
        raise MissingApiKey(f"{ANTHROPIC_KEY_ENV} is unset: export it in the operator shell "
                            f"(the judge backend 'anthropic' reads it from the environment only)")
    headers = {"x-api-key": key.strip(), "anthropic-version": ANTHROPIC_VERSION,
               "content-type": "application/json"}

    def judge(prompt: str) -> str:
        body = json.dumps({"model": model_id, "max_tokens": NUGGET_MAX_TOKENS, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        last = "no attempt"
        for attempt in range(ANTHROPIC_MAX_RETRIES + 1):
            try:
                status, raw = post(ANTHROPIC_URL, headers, body, timeout or 120.0)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                status, raw = 0, str(exc).encode()
            if status == 200:
                try:
                    data = json.loads(raw)
                    text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
                except (ValueError, KeyError, TypeError) as exc:
                    raise JudgeUnavailable(f"anthropic: malformed 200 body: {type(exc).__name__}") from exc
                u = data.get("usage") or {}
                usage.totals["input_tokens"] += int(u.get("input_tokens") or 0)
                usage.totals["output_tokens"] += int(u.get("output_tokens") or 0)
                usage.totals["calls"] += 1
                if data.get("model"):
                    usage.response_models.add(str(data["model"]))
                return text
            last = f"HTTP {status}: {raw[:160].decode('utf-8', 'replace')}" if status else f"network: {raw[:160].decode('utf-8', 'replace')}"
            if status and status not in _RETRY_STATUS:
                raise JudgeUnavailable(f"anthropic: {last}")  # 4xx other than 429: do not hammer
            if attempt < ANTHROPIC_MAX_RETRIES:
                sleep(ANTHROPIC_BACKOFF_BASE_S * (2 ** attempt))
        raise JudgeUnavailable(f"anthropic: gave up after {ANTHROPIC_MAX_RETRIES + 1} attempts ({last})")

    return judge


def estimate_judge_calls(payload: Mapping[str, Any], prompt_index: Mapping[str, dict],
                         done: Mapping[str, Any]) -> dict:
    """Judge calls still to make and their average prompt size, computed BEFORE any call."""
    from beam_scoring import build_nugget_judge_prompt
    calls = chars = 0
    for qid, row in (payload.get("results", {}).get("beam") or {}).items():
        prompt = prompt_index.get(qid)
        if qid in done or prompt is None or not isinstance(row.get("response"), str) or not row["response"].strip():
            continue
        cfg = prompt["scoring_config"]
        for item in cfg.get("nuggets") or []:
            calls += 1
            chars += len(build_nugget_judge_prompt(cfg.get("probing_question", ""), item, row["response"]))
    avg = chars / calls if calls else 0.0
    return {"judge_calls": calls, "avg_prompt_chars": round(avg), "total_prompt_chars": chars,
            "approx_input_tokens": round(chars / 4),  # ~4 chars/token: an estimate, not a count
            "approx_output_tokens_max": calls * NUGGET_MAX_TOKENS}


class JudgeUnavailable(RuntimeError):
    """The served judge could not return a verdict for one nugget."""


def served_judge(scoring_config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Judge:
    """A ``judge(prompt) -> text`` bound to the orchestrator llm_judge transport."""
    import debug_scorer  # the delegation shim -> orchestrator B7 scorer

    config = {**dict(scoring_config), **{k: v for k, v in overrides.items() if v is not None}}

    def judge(prompt: str) -> str:
        try:
            return debug_scorer.request_llm_judge_text(
                prompt, config, max_tokens=NUGGET_MAX_TOKENS,
                output_schema=NUGGET_OUTPUT_SCHEMA)
        except debug_scorer.ScoringUnavailableError as exc:
            raise JudgeUnavailable(str(exc)) from exc

    return judge


def _load_partial(path: Path) -> dict[str, dict]:
    """Previously JUDGED questions only. An ``unjudged`` row (judge down, bad verdict,
    empty response) is a failure to retry, never a result to carry forward. Carrying
    it forward would make a rerun after the judge recovers exit 3 forever."""
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            # Fail before any judge call or append: a resumed v2 run cannot
            # relabel untagged/v1 verdicts. Preserve the historical partial.
            if "nugget_verdicts" in row:
                if row.get("judge_prompt_version") != JUDGE_PROMPT_VERSION:
                    raise BEAMFoldError("partial judged row has missing or incompatible judge_prompt_version")
                done[row["question_id"]] = row
    return done


def judge_run(
    payload: Mapping[str, Any],
    prompt_index: Mapping[str, dict],
    *,
    judge_for: Callable[[Mapping[str, Any]], Judge],
    partial_path: Path | None = None,
    usage_probe: Callable[[], dict[str, int]] | None = None,
) -> dict:
    """Judge every stored BEAM response. Returns ``{"records", "unjudged"}``.

    ``judge_for(scoring_config)`` builds the judge for one question, so a test
    can inject a fake and production binds the served transport.
    """
    results = payload.get("results", {}).get("beam")
    if not isinstance(results, Mapping) or not results:
        raise BEAMFoldError("result file carries no results['beam'] rows")
    done = _load_partial(partial_path) if partial_path else {}
    records: list[dict] = []
    unjudged: list[dict] = []
    for question_id in sorted(results):
        row = results[question_id]
        if question_id in done:
            records.append(done[question_id])
            continue
        prompt = prompt_index.get(question_id)
        if prompt is None:
            outcome = {"question_id": question_id, "reason": "unknown_question_id"}
        elif not isinstance(row.get("response"), str) or not row["response"].strip():
            outcome = {"question_id": question_id, "reason": "empty_response"}
        else:
            before = usage_probe() if usage_probe else None
            try:
                outcome = judge_question(prompt, row["response"],
                                         judge_for(prompt["scoring_config"]))
                outcome["judge_prompt_version"] = JUDGE_PROMPT_VERSION
                if usage_probe:  # per-question delta, so a resumed run still totals correctly
                    after = usage_probe()
                    outcome["judge_usage"] = {k: after[k] - before[k] for k in after}
            except JudgeUnavailable as exc:
                outcome = {"question_id": question_id, "reason": "judge_unavailable",
                           "detail": str(exc)[:240]}
            except NuggetVerdictError as exc:
                outcome = {"question_id": question_id, "reason": "unparseable_verdict",
                           "detail": str(exc)[:240]}
        (records if "nugget_verdicts" in outcome else unjudged).append(outcome)
        if partial_path is not None:
            with open(partial_path, "a") as handle:
                handle.write(json.dumps(outcome, sort_keys=True) + "\n")
    return {"records": records, "unjudged": unjudged}


def recorded_arms(payload: Mapping[str, Any]) -> dict[str, int]:
    """How many result rows each M-12b arm produced, from the rows' ``provenance``
    (B2). Rows written before the harness recorded it count as ``"unrecorded"``."""
    counts: dict[str, int] = {}
    for row in (payload.get("results", {}).get("beam") or {}).values():
        arm = (row.get("provenance") or {}).get("context_mode") or "unrecorded"
        counts[arm] = counts.get(arm, 0) + 1
    return dict(sorted(counts.items()))


#: ``finish_reason`` meaning the reader's answer hit ``max_tokens`` (M-12c(3)).
TRUNCATED_FINISH_REASON = "length"


def finish_reason_summary(payload: Mapping[str, Any]) -> dict:
    """Per-row ``finish_reason`` counts and the truncated question ids of a result file."""
    rows = payload.get("results", {}).get("beam") or {}
    counts: dict[str, int] = {}
    for row in rows.values():
        key = row.get("finish_reason") or "unrecorded"
        counts[key] = counts.get(key, 0) + 1
    truncated = sorted(q for q, row in rows.items()
                       if row.get("finish_reason") == TRUNCATED_FINISH_REASON)
    return {"finish_reason_by_row": dict(sorted(counts.items())),
            # None when no row recorded a finish_reason (pre-B3 run): zero would be a guess.
            "truncated_rows": (len(truncated) if any(r.get("finish_reason") for r in rows.values())
                               else None),
            "truncated_question_ids": truncated}


def build_judged_payload(payload: Mapping[str, Any], judged: Mapping[str, Any], *,
                         split: str, judge_model: str, judge_backend: str = "served",
                         judge_model_id: str | None = None,
                         response_models: list[str] | None = None) -> dict:
    if any(row.get("judge_prompt_version") != JUDGE_PROMPT_VERSION for row in judged["records"]):
        raise BEAMFoldError("cannot label judged rows with missing or incompatible judge_prompt_version")
    identity: dict[str, Any] = {"judge_backend": judge_backend}
    if judge_backend == "anthropic":
        totals = {k: 0 for k in USAGE_KEYS}
        for row in judged["records"]:
            for k, v in (row.get("judge_usage") or {}).items():
                totals[k] = totals.get(k, 0) + int(v)
        identity.update({"judge_model_id": judge_model_id,
                         "judge_response_models": sorted(response_models or []),
                         "judge_usage_totals": totals})
    return {
        **identity,
        "schema": JUDGED_SCHEMA,
        "run_id": payload.get("run_id"),
        "model_role": payload.get("model_role"),
        "config_name": payload.get("config_name"),
        "split": split,
        "judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "question_in_judge_prompt": QUESTION_IN_JUDGE_PROMPT,
        "result_questions": len(payload.get("results", {}).get("beam", {})),
        "context_mode_by_row": recorded_arms(payload),
        **finish_reason_summary(payload),
        "records": list(judged["records"]),
        "unjudged": list(judged["unjudged"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=Path, help="run_benchmark result JSON with results['beam']")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--judge-model", default=None,
                        help="Identity of the judge that answers (model + quant). Required for "
                             "--judge-backend served; defaults to --judge-model-id for anthropic")
    parser.add_argument("--judge-backend", choices=("served", "anthropic"), default="served",
                        help="served = llama-server via the orchestrator transport (default); "
                             "anthropic = Messages API, key from $ANTHROPIC_API_KEY only")
    parser.add_argument("--judge-model-id", default=None,
                        help=f"Pinned API model id (anthropic backend; default {DEFAULT_ANTHROPIC_MODEL})")
    parser.add_argument("--split", default="100K")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--judge-url", default=None, help="Direct llama-server judge base URL")
    parser.add_argument("--judge-role", default=None, help="Orchestrator force_role for the judge")
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args(argv)
    usage: AnthropicUsage | None = None
    if args.judge_backend == "anthropic":
        args.judge_model_id = args.judge_model_id or DEFAULT_ANTHROPIC_MODEL
        args.judge_model = args.judge_model or args.judge_model_id
        if not os.environ.get(ANTHROPIC_KEY_ENV, "").strip():
            print(f"BLOCKED: {ANTHROPIC_KEY_ENV} unset (export it in the operator shell; "
                  f"it is read from the environment only)", file=sys.stderr)
            return 5
        usage = AnthropicUsage()
    else:
        if args.judge_model_id:
            parser.error("--judge-model-id applies to --judge-backend anthropic only")
        if not args.judge_model:
            parser.error("--judge-model is required for --judge-backend served")

    from long_context_adapters import BEAMAdapter

    payload = json.loads(args.result.read_text())
    # Nuggets and ids are identical across the M-12b arms, so the index is built from the
    # full-history arm whatever $BEAM_CONTEXT_MODE says (no retriever, no trace store).
    adapter = BEAMAdapter(data_dir=args.data_dir, split=args.split, context_mode="full")
    prompt_index = {p["id"]: p for p in adapter.extract_all()}
    if not prompt_index:
        raise SystemExit(f"no BEAM {args.split} data loaded: "
                         f"{adapter.accounting_summary()['degraded_sources']}")
    overrides = {"judge_url": args.judge_url, "judge_role": args.judge_role,
                 "timeout": args.timeout}
    partial = args.out_json.with_name(args.out_json.name + ".partial.jsonl")
    # Validate resume provenance before creating directories or opening writers.
    done = _load_partial(partial)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    if usage is not None:
        est = estimate_judge_calls(payload, prompt_index, done)
        print("judge estimate (before any call): " + json.dumps(
            {**est, "backend": "anthropic", "model_id": args.judge_model_id,
             "already_judged_questions": len(done)}), file=sys.stderr)
        one = anthropic_judge(args.judge_model_id, usage, timeout=args.timeout)
        judge_for = lambda cfg: one  # noqa: E731
        probe = usage.snapshot
    else:
        judge_for = lambda cfg: served_judge(cfg, overrides)  # noqa: E731
        probe = None
    judged = judge_run(payload, prompt_index, judge_for=judge_for, partial_path=partial,
                       usage_probe=probe)
    out = build_judged_payload(payload, judged, split=args.split, judge_model=args.judge_model,
                               judge_backend=args.judge_backend, judge_model_id=args.judge_model_id,
                               response_models=sorted(usage.response_models) if usage else None)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"judged": len(out["records"]), "unjudged": len(out["unjudged"]),
                      "out": str(args.out_json)}))
    return 0 if not out["unjudged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
