#!/usr/bin/env python3
"""Canonical judge-suite harness for Qwen3.6-27B-MTP-Q8_0 on architect_general (:8083).

PURPOSE
    Close the three accepted gaps declared in
    epyc-orchestrator/orchestration/accepted_gaps.yaml (expiry 2026-09-01):
      - architect_general        / "Missing overall quality prior"
      - coder_escalation         / "Missing overall quality prior"
      - qwen36_27b_mtp_q8_local  / "Missing overall quality prior"

    by producing a `quality_score` measured on the SAME instrument the other
    roles' numbers were measured on: the canonical Claude-as-Judge 0-3 suite.

WHAT THIS DOES NOT DO
    It launches nothing. It starts, stops, restarts and kills nothing. It takes
    the URL of an ALREADY-RUNNING server and sends scoring requests to it, in
    the `score` subcommand only. `selftest` and `report` never touch the
    network. `score --dry-run` never touches the network either.

SUBCOMMANDS
    selftest   Offline validation: loader, prompt construction, context
               preflight, failure classifier, judge-packet construction and
               aggregation, including deliberate pass and deliberate fail cases.
    score      Capture responses from a running server. Emits a capture JSONL
               and per-suite judge packets. Does NOT assign scores.
    report     Join a capture JSONL with a judge CSV and emit the
               pre-registered report. Offline.

THREE-PHASE DESIGN, AND WHY
    Capture and judging are deliberately separate processes. The canonical
    instrument's judge is a Claude agent scoring in-context (see JUDGE section
    below), not a function this script can call. Splitting the phases also means
    a judging mistake never costs another inference run: the capture is
    complete, fingerprinted and replayable, so re-judging is free.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Canonical code reuse. We import the project's own loader and judge-prompt
# builder rather than reimplementing them, so this harness cannot drift from
# the instrument it claims to be running.
# ---------------------------------------------------------------------------
RESEARCH_REPO = Path("/mnt/raid0/llm/epyc-inference-research")
BENCH_SCRIPTS = RESEARCH_REPO / "scripts" / "benchmark"
if str(BENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPTS))

from suites import (  # noqa: E402  canonical loader; pulls in context_generator
    Question,
    Suite,
    get_inference_params,
    load_suite,
)
import score_with_claude as swc  # noqa: E402  canonical judge prompt + rubric

PROMPTS_DIR = str(RESEARCH_REPO / "benchmarks" / "prompts" / "v1")

# ===========================================================================
# SECTION 1 — THE INSTRUMENT, AND WHAT ITS NUMBERS MAY BE COMPARED AGAINST
# ===========================================================================
#
# This block is the PRE-REGISTERED REPORTING CONTRACT. It is committed to
# source before any 27B response exists. `report` emits exactly these blocks,
# with these denominators, and refuses to emit an "overall" number that is not
# one of them.

# The six suites the 2026-05-04 Claude-as-Judge anchor actually scored.
# Verified against benchmarks/results/reviews/may4_run/qwen36_q8_0_baseline.csv:
# 61 rows, every question id still present in today's YAML, scores summing to
# exactly 170 out of 183.
MAY4_COMPARABLE_SUITES = (
    "agentic",
    "coder",
    "general",
    "instruction_precision",
    "math",
    "thinking",
)

# The two suites present in the 79-question set but ABSENT from the anchor.
NON_ANCHOR_SUITES = ("tool_compliance", "long_context")

ALL_CANONICAL_SUITES = MAY4_COMPARABLE_SUITES + NON_ANCHOR_SUITES

# Suites excluded from `vl.yaml` / `deepseek-v4-quality-gate.yaml` etc: those
# are different instruments and are never loaded here.

POINTS_PER_QUESTION = 3  # 0-3 Claude-as-Judge rubric

REPORTING_CONTRACT = """
PRE-REGISTERED REPORTING CONTRACT  (fixed before any 27B data existed)

1. HEADLINE NUMBER — "may4_comparable"
   Denominator: 61 questions x 3 points = 183.
   Suites: agentic(10), coder(10), general(10), instruction_precision(11),
           math(10), thinking(10).
   THIS is the number that may be placed beside frontdoor's 170/183 (92.9%),
   because it is the same 61 question ids, the same 0-3 rubric and the same
   judge protocol. Report as "N/183 (P%)".

2. SECONDARY — "tool_compliance"
   Denominator: 9 questions x 3 = 27. Reported as its own line.
   MUST NOT be added into the 183 denominator. The anchor never ran it, so a
   combined 70-question number is comparable to nothing.

3. BLOCKED — "long_context"
   Denominator: 9 questions x 3 = 27. Expected to be reported as
   NOT_RUN/blocked on architect_general's production serving shape, with the
   per-question context-budget arithmetic attached. See SECTION 2.

4. FULL CANONICAL — "full_canonical_8suite"
   Denominator: 79 questions x 3 = 237. Emitted ONLY when all 79 questions
   carry an eligible judged score. It is NOT comparable to 170/183 and every
   emission of it is stamped comparable_to_frontdoor_anchor=false.

5. FAILURE TAXONOMY — reported separately from, and never folded into, the
   score. Mechanical classes (request_error, timeout, empty, think_only,
   unclosed_think, truncated) are HARNESS/FORMAT observations. "Wrong" is not
   a mechanical class: only the judge may call an answer wrong. A difference in
   mechanical-failure rate between two models is a serving/plumbing artifact
   until proven otherwise and must never be reported as a quality delta.

6. INELIGIBLE ROWS — any row whose judged score is <0, or whose capture
   fingerprint does not verify, is counted in `ineligible` and EXCLUDED from
   both numerator and denominator. If ineligible > 0, every aggregate in the
   report is stamped decision_grade=false.

7. WHAT THIS NUMBER MAY NOT BE COMPARED AGAINST
   - SWE-bench Verified 23/40 (57.5%). Different instrument, different task
     family, different scale. Never present the two on one axis.
   - roles.qwen36_q8_0.performance.quality_score = 73.8 (2026-04-20). A
     different run on a different date over a 7-suite percentage basis that
     INCLUDES tool_compliance. It is not the 170/183 anchor.
   - Any number produced with --temp-zero or a non-default
     --max-tokens-multiplier, which are stamped
     comparable_to_frontdoor_anchor=false at source.
""".strip()

# ===========================================================================
# SECTION 2 — PRODUCTION SERVING SHAPE
# ===========================================================================
#
# Sourced from epyc-orchestrator/orchestration/model_registry.yaml,
# server_mode.architect_general (read 2026-08-02). We score against the
# PRODUCTION shape because the registry number must describe production.

PRODUCTION_SHAPE = {
    "role": "architect_general",
    "url": "http://127.0.0.1:8083",
    "model": "Qwen3.6-27B-MTP-Q8_0.gguf",
    "device": "MI210 ROCm0",
    "n_ctx": 65536,
    "slots": 8,
    "kv_quant": {"k": "q8_0", "v": "q8_0"},
    "spec_type": "draft-mtp",
    "draft_max": 4,
    # 65536 / 8 slots. THIS, not n_ctx, is the per-request context budget.
    "ctx_per_slot": 65536 // 8,
    # Load-bearing: architect_general is routed to /v1/chat/completions
    # precisely so this kwarg applies (src/llm_primitives/backend.py J12).
    "chat_template_kwargs": {"enable_thinking": False},
    "endpoint": "/v1/chat/completions",
}

# Conservative chars-per-token for the offline context preflight. Real Qwen3.6
# BPE averages ~3.6-4.0 chars/token on English prose; 3.0 deliberately
# OVER-estimates token count so the preflight errs toward refusing a request
# that would overflow a slot rather than discovering it mid-run.
CHARS_PER_TOKEN_CONSERVATIVE = 3.0

DEFAULT_SEED = 42

SELF_PATH = Path(__file__).resolve()
CAPTURE_SCHEMA = "judge_suite_27b.capture.v1"

RUNBOOK = """
OPERATOR RUNBOOK — canonical judge suite on Qwen3.6-27B-MTP-Q8_0
================================================================

PRECONDITION
  architect_general must ALREADY be serving on :8083. This harness starts
  nothing. If the stack is down, bring it up by your normal route first; do
  not let this tool be the reason a server is launched.

STEP 0 — offline validation (safe any time, no network)
  python3 run_judge_suite.py selftest
  python3 run_judge_suite.py score --dry-run          # plan + context preflight

STEP 1 — capture (THE ONLY STEP THAT SENDS REQUESTS)
  python3 run_judge_suite.py score \\
      --url http://127.0.0.1:8083 \\
      --outdir /mnt/raid0/llm/tmp/judge-suite-27b/run-$(date +%Y%m%d)

  Defaults are the production shape: suite-declared temperature and max_tokens,
  seed 42, enable_thinking=false, /v1/chat/completions, multiplier 1.
  Sequential, one request at a time, so it adds one slot of load to a
  16-slot-capable process rather than competing with production traffic.
  Every question is fsync'd to capture.jsonl as it completes, and
  capture.live-status.json is rewritten each time — the run is resumable
  (--resume is on by default) and safe to interrupt at any question boundary.

EXPECTED WALL CLOCK  (70 questions: the 61 anchor questions + tool_compliance 9)
  Worst case, every question runs to its token cap = 228,864 generated tokens:
      @47.8 t/s (registry optimized)   ~80 min
      @40   t/s (role decode estimate) ~95 min
      @19.8 t/s (registry contended)   ~193 min
  Realistic: with enable_thinking=false Qwen3.6 answers moderate prompts in a
  few hundred tokens, so a mean of 600-900 completion tokens is the expectation:
      ~18-26 min uncontended, ~35-53 min if the GPU is contended.
  Prefill is negligible: the longest non-long_context prompt is ~1,000 chars.
  Budget 30 min, do not be alarmed by 90.

IF A QUESTION TIMES OUT
  It is recorded, not retried silently. The row gets request_error set,
  failure_class=timeout, and is marked producer_request_error so the judge
  never sees it and it is EXCLUDED from both numerator and denominator. Then:
    1. Do NOT extend the timeout and rerun the whole suite. Rerun only the
       affected questions:  score --suites <suite>  (resume skips the rest).
    2. If the same question times out twice, that is a finding, not noise —
       record it. Per-suite timeouts are the suite's own declared values
       (90-300 s); a 4096-token generation at 40 t/s needs ~102 s, so a coder
       or math timeout at 300 s means the model is ruminating, which is the
       failure mode the rubric scores 0 anyway.
    3. Never delete a timed-out row to make the denominator look better. If
       ineligible > 0 the report stamps decision_grade=false, and that is the
       honest outcome.

STEP 2 — judge (canonical: Claude subagent, in-context)
  score writes judge_packets/<suite>.json, each carrying the rubric verbatim.
  Hand ONE packet per suite to a Claude subagent; it returns the 5-column CSV
  (suite,question_id,tokens_per_second,claude_score,score_reason). Concatenate
  the per-suite CSVs into one scores.csv (single header).
  Alternative transport if the operator stands up a judge server:
    python3 <research>/scripts/benchmark/score_with_claude.py --judge-url ...
  — nothing is listening on the historical judge port 8199, and no judge model
  is declared in the launch manifest, so record WHICH model judged.

STEP 3 — report (offline)
  python3 run_judge_suite.py report \\
      --capture <outdir>/capture.jsonl \\
      --scores  <outdir>/scores.csv \\
      --judge-label "claude-subagent-in-context 2026-08-02" \\
      --out     <outdir>/report.json

  The headline is N/183. That is the number that closes the three accepted
  gaps and the only one that may sit beside frontdoor's 170/183.
""".strip()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def source_sha256() -> str:
    """SHA-256 of this file, stamped into every capture row."""
    return _sha256(SELF_PATH.read_bytes())


def fingerprint(text: str) -> dict[str, Any]:
    enc = text.encode("utf-8")
    return {"chars": len(text), "utf8_bytes": len(enc), "sha256": _sha256(enc)}


# ===========================================================================
# SECTION 3 — FAILURE CLASSIFICATION BY REASON
# ===========================================================================
#
# Mechanical, deterministic, and strictly separate from the quality score.
#
# The codebase has been bitten twice by conflating these:
#   - substring scoring is comma-brittle (a correct answer scored wrong
#     because of punctuation);
#   - a cross-arm parse-failure gap was once read as a quality gap when it was
#     a scorer bug.
# So: this function may say a response is MALFORMED. It may never say a
# response is WRONG. Only the judge may do that. `report` keeps the two in
# separate columns and refuses to add them together.

FAILURE_CLASSES = (
    "request_error",   # transport failed; no response body exists
    "timeout",         # read timeout; no usable response
    "empty",           # response is whitespace-only
    "unclosed_think",  # <think> opened, never closed: model never left CoT
    "think_only",      # think block closed but nothing substantive after it
    "truncated",       # finish_reason == "length": hit the token cap
    "ok",              # no mechanical defect; correctness is the judge's call
)

THINK_TAIL_MIN_CHARS = 20  # matches the canonical rubric's "no visible answer"


def classify_failure(
    response: str | None,
    finish_reason: str | None,
    request_error: str | None,
) -> tuple[str, list[str]]:
    """Return (primary_class, all_flags). Never returns a correctness verdict."""
    flags: list[str] = []

    if request_error:
        cls = "timeout" if "timed out" in request_error.lower() or "timeout" in request_error.lower() else "request_error"
        return cls, [cls]

    text = (response or "").strip()
    if not text:
        flags.append("empty")

    if "<think>" in text:
        if "</think>" not in text:
            flags.append("unclosed_think")
        else:
            tail = text.rsplit("</think>", 1)[-1].strip()
            if len(tail) < THINK_TAIL_MIN_CHARS:
                flags.append("think_only")

    if finish_reason == "length":
        flags.append("truncated")

    # Precedence: the most upstream defect wins as the primary class.
    for cls in ("empty", "unclosed_think", "think_only", "truncated"):
        if cls in flags:
            return cls, flags
    return "ok", flags or ["ok"]


# ===========================================================================
# SECTION 4 — SUITE LOADING AND CONTEXT PREFLIGHT
# ===========================================================================


@dataclass
class PlannedQuestion:
    suite: str
    question_id: str
    tier: int
    name: str
    prompt: str
    temperature: float
    max_tokens: int
    timeout: int
    est_prompt_tokens: int
    ctx_needed: int
    fits_slot: bool
    fits_full_ctx: bool


@dataclass
class Plan:
    questions: list[PlannedQuestion] = field(default_factory=list)
    blocked: list[PlannedQuestion] = field(default_factory=list)
    shape: dict[str, Any] = field(default_factory=dict)
    params_by_suite: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def runnable_suites(self) -> list[str]:
        return sorted({q.suite for q in self.questions})


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN_CONSERVATIVE) + 1


def build_plan(
    suite_names: Iterable[str],
    *,
    shape: dict[str, Any],
    max_tokens_multiplier: int = 1,
    temp_zero: bool = False,
    prompts_dir: str = PROMPTS_DIR,
) -> Plan:
    """Load suites via the canonical loader and preflight every question.

    long_context prompts are materialised here by suites.py, which calls
    context_generator.build_full_prompt — that is why long_context's real size
    is knowable offline and can be refused before a single request is sent.
    """
    plan = Plan(shape=dict(shape))
    ctx_per_slot = shape["ctx_per_slot"]
    full_ctx = shape["n_ctx"]

    for name in suite_names:
        suite = load_suite(name, prompts_dir)
        if suite is None:
            raise SystemExit(f"suite not found: {name}")
        params = get_inference_params(suite)
        max_tokens = int(params["max_tokens"]) * max_tokens_multiplier
        temperature = 0.0 if temp_zero else float(params["temperature"])
        timeout = int(params["timeout"])
        plan.params_by_suite[name] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "suite_declared": dict(params),
        }
        for q in suite.questions:
            ept = est_tokens(q.prompt)
            need = ept + max_tokens
            pq = PlannedQuestion(
                suite=name,
                question_id=q.id,
                tier=q.tier,
                name=q.name,
                prompt=q.prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                est_prompt_tokens=ept,
                ctx_needed=need,
                fits_slot=need <= ctx_per_slot,
                fits_full_ctx=need <= full_ctx,
            )
            (plan.questions if pq.fits_slot else plan.blocked).append(pq)

    plan.questions.sort(key=lambda q: (q.suite, q.question_id))
    plan.blocked.sort(key=lambda q: (q.suite, q.question_id))
    return plan


# ===========================================================================
# SECTION 5 — CAPTURE (the only subcommand that touches the network)
# ===========================================================================


def _http_json(url: str, payload: dict | None, timeout: int, method: str = "POST") -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe_serving_shape(url: str) -> dict[str, Any]:
    """DERIVE the serving shape from the running server. Never inference.

    WHY THIS EXISTS
    ---------------
    This harness was born single-arm, so its shape lived in a hardcoded
    PRODUCTION_SHAPE dict pinned to architect_general/:8083. `--url` overrode
    only the request target, so pointing it at any other server planned the run
    against the 27B's slot budget and blocked-question set while happily talking
    to a different model. It reported `n_ctx=65536 slots=8` for a server serving
    262144/16. Derived value, hardcoded key — it did not fail, it answered
    confidently about the wrong process.

    A shape is a property of the RUNNING SERVER, so it is read from the server.
    `default_generation_settings.n_ctx` is llama-server's PER-SLOT budget
    (n_ctx / n_parallel), which is the number that actually bounds a request.

    Raises on an unreadable/unparseable server rather than falling back to a
    default: a fabricated shape silently mis-plans the whole run.
    """
    props = _http_json(f"{url}/props", None, timeout=10, method="GET")
    if not isinstance(props, dict):
        raise SystemExit(f"{url}/props did not return an object; cannot derive shape")

    dgs = props.get("default_generation_settings") or {}
    ctx_per_slot = dgs.get("n_ctx")
    slots = props.get("total_slots")
    if not ctx_per_slot or not slots:
        raise SystemExit(
            f"{url}/props lacks the fields needed to derive a shape "
            f"(default_generation_settings.n_ctx={ctx_per_slot!r}, total_slots={slots!r}). "
            "Refusing to plan against a guessed shape."
        )

    model_path = props.get("model_path") or ""
    return {
        "url": url,
        "shape_source": "probed:/props",
        "model": os.path.basename(model_path) or None,
        "model_path": model_path or None,
        "slots": int(slots),
        "ctx_per_slot": int(ctx_per_slot),
        "n_ctx": int(ctx_per_slot) * int(slots),
        # Not probeable from /props; these are protocol choices this harness
        # makes, identical for every arm, and are asserted rather than derived.
        "endpoint": "/v1/chat/completions",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def verify_live_shape(url: str, shape: dict[str, Any]) -> dict[str, Any]:
    """Read the server's own view of itself. Not inference: no tokens generated.

    We verify the LIVE shape instead of trusting the registry, because a
    registry row describes intent and a running process describes reality.
    """
    out: dict[str, Any] = {"url": url, "checked_at": time.time()}
    try:
        out["props"] = _http_json(f"{url}/props", None, timeout=10, method="GET")
    except Exception as exc:  # noqa: BLE001
        out["props_error"] = repr(exc)
    try:
        out["models"] = _http_json(f"{url}/v1/models", None, timeout=10, method="GET")
    except Exception as exc:  # noqa: BLE001
        out["models_error"] = repr(exc)

    props = out.get("props") or {}
    live_n_ctx = props.get("n_ctx") or (props.get("default_generation_settings") or {}).get("n_ctx")
    out["live_n_ctx"] = live_n_ctx
    out["expected_n_ctx"] = shape["n_ctx"]
    # Compare against what the server was PROBED to offer, not against the
    # measurement budget. Those differ whenever a larger-slotted arm is held
    # down for cross-arm comparability, and comparing against the budget
    # reports a mismatch for a server that is in fact exactly as expected.
    probed_slot = shape.get("probed_ctx_per_slot", shape["ctx_per_slot"])
    out["expected_ctx_per_slot"] = probed_slot
    out["measurement_ctx_budget"] = shape.get("measurement_ctx_budget", shape["ctx_per_slot"])
    # llama-server reports n_ctx per slot on some builds and total on others;
    # accept either, but record which one matched so the operator can see it.
    out["n_ctx_match"] = live_n_ctx in (shape["n_ctx"], probed_slot) if live_n_ctx else None
    return out


def ask(
    url: str,
    pq: PlannedQuestion,
    *,
    seed: int,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    """One scoring request against an already-running server. Never launches."""
    messages = [{"role": "user", "content": pq.prompt}]
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": pq.max_tokens,
        "temperature": pq.temperature,
        "seed": seed,
        "stream": False,
    }
    if enable_thinking is not None:
        # Only meaningful on /v1/chat/completions. On /completion it is inert
        # and Qwen3.6 emits <think> regardless.
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    started = time.time()
    try:
        body = _http_json(f"{url}/v1/chat/completions", payload, timeout=pq.timeout)
        elapsed = time.time() - started
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        response = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = body.get("usage") or {}
        ct = usage.get("completion_tokens") or 0
        return {
            "response": response,
            "reasoning": reasoning,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "latency_s": round(elapsed, 3),
            "tokens_per_second": round(ct / elapsed, 2) if elapsed > 0 and ct else 0.0,
            "request_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "response": "",
            "reasoning": "",
            "finish_reason": None,
            "usage": {},
            "latency_s": round(time.time() - started, 3),
            "tokens_per_second": 0.0,
            "request_error": repr(exc),
        }


def make_capture_row(pq: PlannedQuestion, result: dict[str, Any], *, seed: int,
                     src_sha: str, enable_thinking: bool | None) -> dict[str, Any]:
    cls, flags = classify_failure(
        result["response"], result["finish_reason"], result["request_error"]
    )
    return {
        "capture_schema_version": CAPTURE_SCHEMA,
        "runner_source_sha256": src_sha,
        "suite": pq.suite,
        "question_id": pq.question_id,
        "tier": pq.tier,
        "name": pq.name,
        "prompt": pq.prompt,
        "response": result["response"],
        "reasoning": result["reasoning"],
        "prompt_fingerprint": fingerprint(pq.prompt),
        "response_fingerprint": fingerprint(result["response"]),
        "reasoning_fingerprint": fingerprint(result["reasoning"]),
        "finish_reason": result["finish_reason"],
        "usage": result["usage"],
        "latency_s": result["latency_s"],
        "tokens_per_second": result["tokens_per_second"],
        "request_error": result["request_error"],
        "failure_class": cls,
        "failure_flags": flags,
        "request_params": {
            "temperature": pq.temperature,
            "max_tokens": pq.max_tokens,
            "timeout": pq.timeout,
            "seed": seed,
            "enable_thinking": enable_thinking,
            "endpoint": "/v1/chat/completions",
        },
    }


def verify_row(row: dict[str, Any]) -> str:
    """Fail-closed capture check, mirroring the project's capture contract."""
    if row.get("capture_schema_version") != CAPTURE_SCHEMA:
        return "wrong_schema"
    for f in ("prompt", "response", "reasoning"):
        if not isinstance(row.get(f), str):
            return f"missing_{f}"
        if row.get(f"{f}_fingerprint") != fingerprint(row[f]):
            return f"{f}_fingerprint_mismatch"
    if row.get("request_error"):
        return "producer_request_error"
    return "eligible"


# ===========================================================================
# SECTION 6 — JUDGE PACKETS
# ===========================================================================
#
# THE JUDGE. The canonical instrument is Claude-as-Judge on a 0-3 scale, per
# /workspace/.claude/commands/benchmark.md. The 2026-05-04 anchor was produced
# by "one Claude subagent per config, scored in-context"
# (benchmarks/results/reviews/SCORING_SUMMARY_2026-05-03.md). There are two
# supported transports, and both use the SAME rubric text, imported from
# score_with_claude.py so it cannot drift:
#
#   agent  (default, canonical) — emit one packet per suite; a Claude subagent
#          reads the packet and writes the 5-column review CSV. This is
#          literally how 170/183 was made.
#   server — POST score_with_claude.build_judge_input()'s byte-exact payload to
#          an HTTP judge (--judge-url). NOTE: nothing is listening on the
#          historical judge port 8199, and no judge model is declared in the
#          launch manifest, so this transport requires the operator to stand a
#          judge up first and to record which model served as judge.


def build_judge_packets(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    packets: dict[str, dict[str, Any]] = {}
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_suite[r["suite"]].append(r)

    for suite, srows in sorted(by_suite.items()):
        items = []
        for r in sorted(srows, key=lambda x: x["question_id"]):
            ji = swc.build_judge_input(r["question_id"], suite, r["prompt"], r["response"])
            items.append({
                "suite": suite,
                "question_id": r["question_id"],
                "tier": r["tier"],
                "prompt": r["prompt"],
                "response": r["response"],
                "tokens_per_second": r["tokens_per_second"],
                "finish_reason": r["finish_reason"],
                "failure_class": r["failure_class"],
                "capture_status": verify_row(r),
                "scorer_input_sha256": ji["scorer_input_sha256"],
                "scorer_input_utf8_bytes": ji["scorer_input_utf8_bytes"],
            })
        packets[suite] = {
            "suite": suite,
            "rubric_system_prompt": swc.SYSTEM_PROMPT.format(calibration=swc.CALIBRATION_EXAMPLES),
            "output_csv_columns": ["suite", "question_id", "tokens_per_second",
                                   "claude_score", "score_reason"],
            "instructions": (
                "Score every item 0-3 using rubric_system_prompt verbatim. Emit ONE csv row "
                "per item with the columns in output_csv_columns. Do not skip items. If an "
                "item cannot be scored, emit claude_score=-1 with the reason; it will be "
                "counted ineligible and excluded from both numerator and denominator, never "
                "silently averaged."
            ),
            "items": items,
        }
    return packets


# ===========================================================================
# SECTION 7 — AGGREGATION
# ===========================================================================


def aggregate(
    capture_rows: list[dict[str, Any]],
    scores: dict[tuple[str, str], tuple[int, str]],
    *,
    run_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Join capture + judged scores into the pre-registered report blocks."""
    run_meta = run_meta or {}
    per_q: list[dict[str, Any]] = []
    for r in capture_rows:
        key = (r["suite"], r["question_id"])
        raw = scores.get(key)
        cap_status = verify_row(r)
        if raw is None:
            score, reason, elig = -1, "no_judged_score", "unjudged"
        else:
            score, reason = raw
            if cap_status != "eligible":
                score, elig = -1, cap_status
            else:
                elig = "eligible" if score >= 0 else "judge_ineligible"
        per_q.append({
            "suite": r["suite"],
            "question_id": r["question_id"],
            "tier": r["tier"],
            "claude_score": score,
            "score_reason": reason,
            "eligibility": elig,
            "failure_class": r["failure_class"],
            "failure_flags": r["failure_flags"],
            "finish_reason": r["finish_reason"],
            "tokens_per_second": r["tokens_per_second"],
            "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
        })

    def block(suites: Iterable[str], label: str) -> dict[str, Any]:
        want = set(suites)
        rows = [p for p in per_q if p["suite"] in want]
        elig = [p for p in rows if p["eligibility"] == "eligible"]
        inelig = [p for p in rows if p["eligibility"] != "eligible"]
        total = sum(p["claude_score"] for p in elig)
        denom = len(elig) * POINTS_PER_QUESTION
        return {
            "label": label,
            "questions_scored": len(elig),
            "questions_ineligible": len(inelig),
            "score": total,
            "denominator": denom,
            "pct": round(total / denom * 100, 1) if denom else None,
            "pass_rate_ge2": round(
                sum(1 for p in elig if p["claude_score"] >= 2) / len(elig) * 100, 1
            ) if elig else None,
            "decision_grade": len(inelig) == 0 and len(elig) > 0,
            "ineligible_detail": sorted(
                {p["eligibility"] for p in inelig}
            ) if inelig else [],
        }

    per_suite = {}
    for s in sorted({p["suite"] for p in per_q}):
        per_suite[s] = block([s], s)

    may4 = block(MAY4_COMPARABLE_SUITES, "may4_comparable")
    # The headline is only comparable if ALL SIX anchor suites are complete at
    # their full question counts. A partial 61-set is not 61.
    expected_may4 = 61
    may4["complete_61"] = may4["questions_scored"] == expected_may4
    may4["comparable_to_frontdoor_anchor"] = bool(
        may4["complete_61"]
        and may4["decision_grade"]
        and may4["denominator"] == 183
        and not run_meta.get("deviations")
    )
    may4["anchor"] = {
        "role": "frontdoor",
        "model": "Qwen3.6-35B-A3B-MTP-Q8_0",
        "score": "170/183 (92.9%)",
        "date": "2026-05-04",
        "method": "Claude-as-Judge 0-3, in-context Claude subagent",
        "source": "benchmarks/results/reviews/may4_run/qwen36_q8_0_baseline.csv",
    }

    full = block(ALL_CANONICAL_SUITES, "full_canonical_8suite")
    full["complete_79"] = full["questions_scored"] == 79
    full["comparable_to_frontdoor_anchor"] = False
    full["comparability_note"] = (
        "237-point basis over 8 suites. The 170/183 anchor is a 183-point basis "
        "over 6 suites. These are different denominators over different question "
        "sets and MUST NOT be compared."
    )
    # Contract clause 4: emit a full-canonical number ONLY at 79/79. Below that,
    # suppress it — a partial 8-suite total on a shrunken denominator is exactly
    # the kind of number that gets copied into a registry and read as an overall.
    if not full["complete_79"]:
        full["suppressed"] = True
        full["suppressed_reason"] = (
            f"only {full['questions_scored']}/79 questions carry an eligible score; "
            "per reporting-contract clause 4 no full-canonical number is emitted"
        )
        full["score"] = None
        full["denominator"] = None
        full["pct"] = None
        full["pass_rate_ge2"] = None
        full["decision_grade"] = False
    else:
        full["suppressed"] = False

    taxonomy: dict[str, Any] = {"overall": defaultdict(int), "by_suite": defaultdict(lambda: defaultdict(int))}
    for p in per_q:
        taxonomy["overall"][p["failure_class"]] += 1
        taxonomy["by_suite"][p["suite"]][p["failure_class"]] += 1
    taxonomy["overall"] = dict(taxonomy["overall"])
    taxonomy["by_suite"] = {k: dict(v) for k, v in taxonomy["by_suite"].items()}
    mech = sum(v for k, v in taxonomy["overall"].items() if k != "ok")
    taxonomy["mechanical_failures"] = mech
    taxonomy["mechanical_failure_rate_pct"] = (
        round(mech / len(per_q) * 100, 1) if per_q else None
    )
    taxonomy["interpretation"] = (
        "Mechanical classes describe FORM, not correctness. A mechanical-failure "
        "rate that differs from another model's is a serving/plumbing/scorer "
        "artifact until independently shown otherwise; it is not a quality delta. "
        "Only claude_score carries a correctness verdict."
    )

    tps = [p["tokens_per_second"] for p in per_q if p["tokens_per_second"]]

    return {
        "instrument": "canonical Claude-as-Judge 0-3 suite (benchmarks/prompts/v1)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_meta": run_meta,
        "reporting_contract": REPORTING_CONTRACT,
        "headline_may4_comparable": may4,
        "tool_compliance": per_suite.get("tool_compliance", block(["tool_compliance"], "tool_compliance")),
        "long_context": per_suite.get("long_context", block(["long_context"], "long_context")),
        "full_canonical_8suite": full,
        "per_suite": per_suite,
        "failure_taxonomy": taxonomy,
        "throughput": {
            "median_tps": round(statistics.median(tps), 2) if tps else None,
            "n": len(tps),
        },
        "not_comparable_against": [
            "SWE-bench Verified 23/40 (57.5%) — different instrument, different scale",
            "roles.qwen36_q8_0.performance.quality_score 73.8 (2026-04-20) — different run, 7-suite percentage basis",
        ],
        "per_question": sorted(per_q, key=lambda p: (p["suite"], p["question_id"])),
    }


def load_scores_csv(path: Path) -> dict[tuple[str, str], tuple[int, str]]:
    out: dict[tuple[str, str], tuple[int, str]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                sc = int(row["claude_score"])
            except (KeyError, TypeError, ValueError):
                sc = -1
            out[(row["suite"], row["question_id"])] = (sc, row.get("score_reason", ""))
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ===========================================================================
# SECTION 8 — SUBCOMMANDS
# ===========================================================================


def render_plan(plan: Plan) -> str:
    lines = []
    shape = plan.shape
    lines.append(
        f"shape source : {shape.get('shape_source', 'declared')}"
        + (f"   model={shape['model']}" if shape.get("model") else "")
        + (f"   url={shape['url']}" if shape.get("url") else "")
    )
    if shape.get("measurement_ctx_budget") and \
            shape.get("probed_ctx_per_slot") and \
            shape["measurement_ctx_budget"] != shape["probed_ctx_per_slot"]:
        lines.append(
            f"ctx budget   : {shape['measurement_ctx_budget']} "
            f"(server offers {shape['probed_ctx_per_slot']}/slot; held down for comparability)"
        )
    lines.append(f"serving shape: n_ctx={shape['n_ctx']} slots={shape['slots']} "
                 f"-> ctx_per_slot={shape['ctx_per_slot']}")
    lines.append("")
    by_suite: dict[str, list[PlannedQuestion]] = defaultdict(list)
    for q in plan.questions:
        by_suite[q.suite].append(q)
    blocked_by_suite: dict[str, list[PlannedQuestion]] = defaultdict(list)
    for q in plan.blocked:
        blocked_by_suite[q.suite].append(q)

    lines.append(f"{'suite':24s} {'run':>4s} {'blocked':>8s}  temp  max_tok  timeout")
    for s in sorted(set(by_suite) | set(blocked_by_suite)):
        p = plan.params_by_suite[s]
        lines.append(f"{s:24s} {len(by_suite[s]):4d} {len(blocked_by_suite[s]):8d}"
                     f"  {p['temperature']:<5.2f} {p['max_tokens']:7d} {p['timeout']:8d}")
    lines.append("")
    lines.append(f"TOTAL runnable={len(plan.questions)}  blocked={len(plan.blocked)}")
    if plan.blocked:
        lines.append("")
        lines.append("BLOCKED — context budget exceeds one slot:")
        for q in plan.blocked:
            lines.append(
                f"  {q.suite}/{q.question_id:32s} est_prompt={q.est_prompt_tokens:6d} "
                f"+ max_tokens={q.max_tokens} = {q.ctx_needed:6d} > slot {plan.shape['ctx_per_slot']}"
                f"   fits_full_n_ctx={q.fits_full_ctx}"
            )
    return "\n".join(lines)


def cmd_score(args: argparse.Namespace) -> int:
    suites = args.suites or list(ALL_CANONICAL_SUITES)

    # ---- shape is DERIVED from the server, never assumed -------------------
    # --dry-run promises "no network access at all", so it cannot probe. It
    # plans against the declared shape and says so, loudly, rather than
    # silently presenting a declared plan as a live one.
    if args.dry_run:
        shape = dict(PRODUCTION_SHAPE)
        shape["shape_source"] = "declared:PRODUCTION_SHAPE (--dry-run cannot probe)"
    else:
        shape = probe_serving_shape(args.url)

    # ---- measurement budget is a property of the EXPERIMENT ---------------
    # Distinct from the server's capability. A cross-arm comparison is only
    # apples-to-apples if every arm plans and generates under ONE budget, so a
    # larger-slotted arm is deliberately held down to the smaller one.
    probed_slot = int(shape["ctx_per_slot"])
    if args.ctx_budget:
        if args.ctx_budget > probed_slot:
            raise SystemExit(
                f"--ctx-budget {args.ctx_budget} exceeds this server's per-slot context "
                f"({probed_slot}). The budget must fit the slot; raise -c / lower -np, "
                "or lower the budget."
            )
        shape["ctx_per_slot"] = args.ctx_budget
    shape["probed_ctx_per_slot"] = probed_slot
    shape["measurement_ctx_budget"] = int(shape["ctx_per_slot"])

    plan = build_plan(
        suites, shape=shape,
        max_tokens_multiplier=args.max_tokens_multiplier,
        temp_zero=args.temp_zero,
    )

    deviations = []
    if args.temp_zero:
        deviations.append("temp_zero: overrides suite-declared temperature; breaks comparability with 170/183")
    if args.max_tokens_multiplier != 1:
        deviations.append(f"max_tokens_multiplier={args.max_tokens_multiplier}: non-default")
    if args.enable_thinking:
        deviations.append("enable_thinking=true: production shape is false")
    if args.ctx_budget and args.ctx_budget != probed_slot:
        deviations.append(
            f"ctx_budget={args.ctx_budget} held below this server's {probed_slot}/slot "
            "for cross-arm comparability"
        )
    if shape.get("model") and shape["model"] != PRODUCTION_SHAPE["model"]:
        deviations.append(
            f"model={shape['model']} is NOT the canonical arm "
            f"({PRODUCTION_SHAPE['model']}); this run does not describe architect_general"
        )

    print(render_plan(plan))
    if deviations:
        print("\nDEVIATIONS FROM PRODUCTION/ANCHOR SHAPE:")
        for d in deviations:
            print(f"  ! {d}")

    if args.dry_run:
        print("\n--dry-run: no network access performed, nothing launched, no requests sent.")
        return 0

    if not plan.questions:
        print("nothing runnable", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    capture_path = outdir / "capture.jsonl"
    status_path = outdir / "capture.live-status.json"

    live = verify_live_shape(args.url, shape)
    (outdir / "live_shape.json").write_text(json.dumps(live, indent=2))
    print(f"\nlive shape probe -> {outdir / 'live_shape.json'} "
          f"(live_n_ctx={live.get('live_n_ctx')} match={live.get('n_ctx_match')})")

    enable_thinking = True if args.enable_thinking else False
    src_sha = source_sha256()
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, str]] = set()
    if capture_path.exists() and args.resume:
        rows = read_jsonl(capture_path)
        done = {(r["suite"], r["question_id"]) for r in rows if not r.get("request_error")}
        print(f"resume: {len(done)} rows already captured")

    fh = open(capture_path, "a")
    try:
        for i, pq in enumerate(plan.questions, 1):
            if (pq.suite, pq.question_id) in done:
                continue
            res = ask(args.url, pq, seed=args.seed, enable_thinking=enable_thinking)
            row = make_capture_row(pq, res, seed=args.seed, src_sha=src_sha,
                                   enable_thinking=enable_thinking)
            rows.append(row)
            # Persist per question: every persisted unit is a drain point.
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            status_path.write_text(json.dumps({
                "schema": CAPTURE_SCHEMA, "completed": len(rows),
                "total": len(plan.questions), "updated_at": time.time(),
                "last": f"{pq.suite}/{pq.question_id}",
                "failure_counts": {c: sum(1 for r in rows if r["failure_class"] == c)
                                   for c in FAILURE_CLASSES},
            }, indent=2))
            print(f"  [{i}/{len(plan.questions)}] {pq.suite}/{pq.question_id} "
                  f"{row['failure_class']} {row['tokens_per_second']} t/s "
                  f"({(res['usage'] or {}).get('completion_tokens', 0)} tok)")
    finally:
        fh.close()

    packets = build_judge_packets(rows)
    pdir = outdir / "judge_packets"
    pdir.mkdir(exist_ok=True)
    for suite, packet in packets.items():
        (pdir / f"{suite}.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False))
    print(f"\ncapture  -> {capture_path} ({len(rows)} rows)")
    print(f"packets  -> {pdir} ({len(packets)} suites)")
    print("\nNo scores assigned. Run the judge, then `report`.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.capture))
    scores = load_scores_csv(Path(args.scores)) if args.scores else {}
    meta = {
        "capture": str(args.capture),
        "scores": str(args.scores) if args.scores else None,
        "judge": args.judge_label,
        "deviations": [],
    }
    if args.deviation:
        meta["deviations"] = list(args.deviation)
    rep = aggregate(rows, scores, run_meta=meta)
    out = Path(args.out) if args.out else None
    if out:
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"report -> {out}")
    print(render_report(rep))
    return 0


def render_report(rep: dict[str, Any]) -> str:
    L = []
    A = L.append
    A("=" * 72)
    A("CANONICAL JUDGE SUITE — Qwen3.6-27B-MTP-Q8_0 (architect_general)")
    A("=" * 72)
    h = rep["headline_may4_comparable"]
    A("")
    A(f"HEADLINE (may4_comparable, 6 suites): {h['score']}/{h['denominator']} "
      f"({h['pct']}%)  pass>=2 {h['pass_rate_ge2']}%")
    A(f"  questions scored {h['questions_scored']} (expect 61)   "
      f"complete_61={h['complete_61']}  decision_grade={h['decision_grade']}")
    A(f"  COMPARABLE TO frontdoor 170/183 (92.9%)?  {h['comparable_to_frontdoor_anchor']}")
    if h["ineligible_detail"]:
        A(f"  ineligible: {h['questions_ineligible']} {h['ineligible_detail']}")
    A("")
    def line(b: dict[str, Any], label: str, extra: str = "") -> str:
        if not b["questions_scored"]:
            return f"{label}: NOT RUN (0 questions scored) {extra}"
        return (f"{label}: {b['score']}/{b['denominator']} ({b['pct']}%) "
                f"n={b['questions_scored']} {extra}")

    A(line(rep["tool_compliance"], "tool_compliance", "[separate line, NOT added to /183]"))
    A(line(rep["long_context"], "long_context",
           "[expected NOT RUN: every question exceeds the 8192-token production slot]"))
    f = rep["full_canonical_8suite"]
    if f.get("suppressed"):
        A(f"full_canonical_8suite: SUPPRESSED — {f['suppressed_reason']}")
    else:
        A(f"full_canonical_8suite: {f['score']}/{f['denominator']} ({f['pct']}%) "
          f"complete_79=True  comparable_to_anchor=False")
    A("")
    A("PER SUITE")
    for s, b in sorted(rep["per_suite"].items()):
        A(f"  {s:24s} {b['score']:4d}/{b['denominator']:<4d} "
          f"{'' if b['pct'] is None else str(b['pct'])+'%':>7s}  "
          f"pass>=2 {b['pass_rate_ge2']}%  ineligible={b['questions_ineligible']}")
    A("")
    A("FAILURE TAXONOMY (form, not correctness — never a quality delta)")
    for k, v in sorted(rep["failure_taxonomy"]["overall"].items()):
        A(f"  {k:16s} {v}")
    A(f"  mechanical failure rate: {rep['failure_taxonomy']['mechanical_failure_rate_pct']}%")
    A("")
    A("NOT COMPARABLE AGAINST:")
    for n in rep["not_comparable_against"]:
        A(f"  - {n}")
    return "\n".join(L)


# ===========================================================================
# SECTION 9 — SELFTEST (fully offline)
# ===========================================================================

def _synthetic_rows() -> list[dict[str, Any]]:
    """Fixtures with KNOWN-correct outcomes, covering pass and fail paths."""
    src = "SELFTEST"

    def row(suite, qid, response, finish="stop", err=None, tier=1):
        cls, flags = classify_failure(response, finish, err)
        return {
            "capture_schema_version": CAPTURE_SCHEMA, "runner_source_sha256": src,
            "suite": suite, "question_id": qid, "tier": tier, "name": qid,
            "prompt": f"PROMPT for {suite}/{qid}",
            "response": response or "", "reasoning": "",
            "prompt_fingerprint": fingerprint(f"PROMPT for {suite}/{qid}"),
            "response_fingerprint": fingerprint(response or ""),
            "reasoning_fingerprint": fingerprint(""),
            "finish_reason": finish, "usage": {"completion_tokens": 100},
            "latency_s": 1.0, "tokens_per_second": 40.0, "request_error": err,
            "failure_class": cls, "failure_flags": flags,
            "request_params": {"seed": DEFAULT_SEED},
        }

    return [
        # 1. GOOD: well-formed, correct. Judge will score 3.
        row("math", "sel_good", "The answer is 15. Step 1: ... Step 2: ... Therefore 15."),
        # 2. DELIBERATELY WRONG: well-formed, confidently WRONG.
        #    Mechanically indistinguishable from #1 — only the judge can fail it.
        row("math", "sel_wrong", "The answer is 9,999. Step 1: ... Therefore 9,999."),
        # 3. DELIBERATELY MALFORMED: never leaves the think block.
        row("math", "sel_malformed", "<think>I should consider the cases carefully and", finish="length"),
        # 4. EMPTY.
        row("math", "sel_empty", "   ", finish="stop"),
        # 5. TRUNCATED but substantive.
        row("math", "sel_truncated", "The answer begins as follows and continues at length", finish="length"),
        # 6. TRANSPORT failure.
        row("math", "sel_error", "", finish=None, err="TimeoutError('timed out')"),
        # 7. think block that CLOSED with a real answer -> not a failure.
        row("math", "sel_think_closed", "<think>deliberating</think>The answer is 15, computed as shown above."),
    ]


def cmd_selftest(args: argparse.Namespace) -> int:  # noqa: C901
    ok = True
    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))

    print("=" * 72)
    print("SELFTEST (offline — no network, no server, nothing launched)")
    print("=" * 72)

    print("\n[1] Canonical loader + prompt construction")
    counts = {}
    for s in ALL_CANONICAL_SUITES:
        su = load_suite(s, PROMPTS_DIR)
        counts[s] = len(su.questions)
    check("8 canonical suites load", len(counts) == 8, str(counts))
    check("total = 79 questions", sum(counts.values()) == 79, f"got {sum(counts.values())}")
    check("may4 six suites = 61 questions",
          sum(counts[s] for s in MAY4_COMPARABLE_SUITES) == 61,
          f"got {sum(counts[s] for s in MAY4_COMPARABLE_SUITES)}")
    check("61 x 3 = 183 (the anchor denominator)",
          sum(counts[s] for s in MAY4_COMPARABLE_SUITES) * POINTS_PER_QUESTION == 183)
    lc = load_suite("long_context", PROMPTS_DIR)
    biggest = max(len(q.prompt) for q in lc.questions)
    check("long_context prompts materialised via context_generator.build_full_prompt",
          biggest > 40000, f"largest prompt {biggest} chars")

    print("\n[2] Context preflight against the production serving shape")
    plan = build_plan(ALL_CANONICAL_SUITES, shape=PRODUCTION_SHAPE)
    check("all 9 long_context questions BLOCKED on an 8192-token slot",
          len([q for q in plan.blocked if q.suite == "long_context"]) == 9,
          f"blocked={len(plan.blocked)}")
    check("no non-long_context question is blocked",
          all(q.suite == "long_context" for q in plan.blocked))
    check("70 questions runnable on the production shape",
          len(plan.questions) == 70, f"got {len(plan.questions)}")
    check("61 anchor questions all runnable",
          len([q for q in plan.questions if q.suite in MAY4_COMPARABLE_SUITES]) == 61)
    over = [q for q in plan.questions if q.ctx_needed > PRODUCTION_SHAPE["ctx_per_slot"]]
    check("no runnable question exceeds one slot", not over)

    print("\n[3] Failure classifier — distinct reasons, distinctly")
    cases = [
        ("well-formed correct",   "The answer is 15.", "stop", None, "ok"),
        ("DELIBERATELY WRONG",    "The answer is 9,999.", "stop", None, "ok"),
        ("DELIBERATELY MALFORMED","<think>still thinking", "length", None, "unclosed_think"),
        ("think closed, answered","<think>x</think>The answer is 15, as shown.", "stop", None, "ok"),
        ("think closed, no answer","<think>x</think> ok", "stop", None, "think_only"),
        ("empty",                 "   ", "stop", None, "empty"),
        ("truncated substantive", "A long partial answer that got cut", "length", None, "truncated"),
        ("transport timeout",     "", None, "TimeoutError('timed out')", "timeout"),
        ("transport error",       "", None, "URLError('connection refused')", "request_error"),
    ]
    for label, resp, fin, err, expect in cases:
        got, _ = classify_failure(resp, fin, err)
        check(f"{label:24s} -> {expect}", got == expect, f"got {got}")
    print("  NOTE: 'DELIBERATELY WRONG' classifies as ok — that is CORRECT.")
    print("        A wrong answer has no FORM defect. Only the judge may call it wrong.")
    print("        'DELIBERATELY MALFORMED' is caught here, pre-judge, and is reported")
    print("        as a harness/serving artifact, never as a quality delta.")

    print("\n[4] Judge packet construction (rubric imported from score_with_claude.py)")
    rows = _synthetic_rows()
    packets = build_judge_packets(rows)
    p = packets["math"]
    check("packet built for the suite", "math" in packets)
    check("packet carries the canonical 0-3 rubric verbatim",
          "0-3 scale" in p["rubric_system_prompt"] and "Scoring rubric" in p["rubric_system_prompt"])
    check("packet CSV columns match the May-4 anchor schema",
          p["output_csv_columns"] == ["suite", "question_id", "tokens_per_second",
                                      "claude_score", "score_reason"])
    check("every captured row is in the packet", len(p["items"]) == len(rows))
    check("scorer input is fingerprinted (no silent truncation)",
          all(i["scorer_input_sha256"] and i["scorer_input_utf8_bytes"] > 0 for i in p["items"]))
    err_item = [i for i in p["items"] if i["question_id"] == "sel_error"][0]
    check("transport-failed row is marked ineligible BEFORE the judge sees it",
          err_item["capture_status"] == "producer_request_error", err_item["capture_status"])

    print("\n[5] Capture integrity — fail closed on tampering")
    good = dict(rows[0])
    check("clean row verifies", verify_row(good) == "eligible")
    tampered = dict(rows[0]); tampered["response"] = tampered["response"] + " EDITED"
    check("tampered response is REJECTED",
          verify_row(tampered) == "response_fingerprint_mismatch", verify_row(tampered))

    print("\n[6] Aggregation — proving the scorer can PASS")
    perfect_rows = [r for r in rows if r["question_id"] in
                    ("sel_good", "sel_wrong", "sel_think_closed")]
    perfect_scores = {(r["suite"], r["question_id"]): (3, "correct") for r in perfect_rows}
    rep_pass = aggregate(perfect_rows, perfect_scores)
    b = rep_pass["per_suite"]["math"]
    check("all-3 fixture -> 9/9 = 100.0%",
          (b["score"], b["denominator"], b["pct"]) == (9, 9, 100.0),
          f"{b['score']}/{b['denominator']} {b['pct']}%")
    check("pass_rate_ge2 = 100.0", b["pass_rate_ge2"] == 100.0)
    check("no mechanical failures in the clean fixture",
          rep_pass["failure_taxonomy"]["mechanical_failures"] == 0)

    print("\n[7] Aggregation — proving the scorer can FAIL, for the right reason")
    mixed_scores = {
        ("math", "sel_good"): (3, "correct"),
        ("math", "sel_wrong"): (0, "wrong final answer: 9999 != 15"),
        ("math", "sel_think_closed"): (3, "correct"),
    }
    rep_fail = aggregate(perfect_rows, mixed_scores)
    b2 = rep_fail["per_suite"]["math"]
    check("a judged-wrong answer LOWERS the score: 6/9 = 66.7%",
          (b2["score"], b2["denominator"], b2["pct"]) == (6, 9, 66.7),
          f"{b2['score']}/{b2['denominator']} {b2['pct']}%")
    check("the wrong answer contributed 0 while staying failure_class=ok",
          [q for q in rep_fail["per_question"] if q["question_id"] == "sel_wrong"][0]["failure_class"] == "ok")
    check("pass_rate_ge2 drops to 66.7", b2["pass_rate_ge2"] == 66.7)
    check("PASS and FAIL fixtures give different results",
          rep_pass["per_suite"]["math"]["pct"] != b2["pct"])

    print("\n[8] Malformed and unjudged rows are quarantined, not averaged in")
    bad_rows = [r for r in rows if r["question_id"] in
                ("sel_good", "sel_malformed", "sel_empty", "sel_error")]
    bad_scores = {
        ("math", "sel_good"): (3, "correct"),
        ("math", "sel_malformed"): (0, "never left think block"),
        ("math", "sel_empty"): (0, "empty"),
        # sel_error deliberately absent: no judged score at all.
    }
    rep_bad = aggregate(bad_rows, bad_scores)
    b3 = rep_bad["per_suite"]["math"]
    check("transport-failed row is EXCLUDED from the denominator",
          b3["denominator"] == 9, f"denominator {b3['denominator']} (3 rows x 3)")
    check("it is counted as ineligible instead", b3["questions_ineligible"] == 1)
    check("decision_grade=false when anything is ineligible", b3["decision_grade"] is False)
    tax = rep_bad["failure_taxonomy"]["overall"]
    check("failure taxonomy separates the reasons",
          tax.get("unclosed_think") == 1 and tax.get("empty") == 1
          and tax.get("timeout") == 1 and tax.get("ok") == 1, str(tax))

    print("\n[9] Comparability gates")
    check("a 3-question fixture is NOT flagged comparable to 170/183",
          rep_pass["headline_may4_comparable"]["comparable_to_frontdoor_anchor"] is False)
    check("full_canonical_8suite is ALWAYS stamped not-comparable",
          rep_pass["full_canonical_8suite"]["comparable_to_frontdoor_anchor"] is False)
    check("an incomplete full_canonical_8suite emits NO number (contract clause 4)",
          rep_pass["full_canonical_8suite"]["suppressed"] is True
          and rep_pass["full_canonical_8suite"]["pct"] is None)
    check("SWE-bench is listed as not-comparable",
          any("SWE-bench" in n for n in rep_pass["not_comparable_against"]))
    dev_rep = aggregate(perfect_rows, perfect_scores,
                        run_meta={"deviations": ["temp_zero"]})
    check("a declared deviation forces comparable=false",
          dev_rep["headline_may4_comparable"]["comparable_to_frontdoor_anchor"] is False)

    print("\n" + "=" * 72)
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    print("=" * 72)
    if args.show_contract:
        print("\n" + REPORTING_CONTRACT)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Canonical judge-suite harness for Qwen3.6-27B-MTP-Q8_0. "
                    "Never launches, stops or kills anything.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="offline validation; no network")
    st.add_argument("--show-contract", action="store_true")
    st.set_defaults(func=cmd_selftest)

    rb = sub.add_parser("runbook", help="print the operator runbook and reporting contract")
    rb.set_defaults(func=lambda a: (print(RUNBOOK), print("\n\n" + REPORTING_CONTRACT), 0)[-1])

    sc = sub.add_parser("score", help="capture responses from an ALREADY-RUNNING server")
    sc.add_argument("--url", default=PRODUCTION_SHAPE["url"])
    sc.add_argument(
        "--ctx-budget", type=int, default=0,
        help="Per-request context budget for THIS MEASUREMENT, in tokens. Default 0 "
             "= use whatever the probed server offers per slot. Set it to hold a "
             "larger-slotted arm down to a smaller arm's budget so a cross-arm "
             "comparison is apples-to-apples. Must not exceed the server's slot.",
    )
    sc.add_argument("--outdir", default="/mnt/raid0/llm/tmp/judge-suite-27b/run")
    sc.add_argument("--suites", nargs="*", default=None,
                    help=f"default: the anchor six + tool_compliance. all: {ALL_CANONICAL_SUITES}")
    sc.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sc.add_argument("--max-tokens-multiplier", type=int, default=1,
                    help="DEFAULT 1. The production slot is 8192 tokens; the anchor's "
                         "model carried max_tokens_multiplier=4 (16384) which CANNOT fit here.")
    sc.add_argument("--temp-zero", action="store_true",
                    help="override suite-declared temperature with 0.0 (breaks comparability)")
    sc.add_argument("--enable-thinking", action="store_true",
                    help="production shape is enable_thinking=FALSE; this deviates")
    sc.add_argument("--resume", action="store_true", default=True)
    sc.add_argument("--dry-run", action="store_true",
                    help="print the plan and preflight; perform NO network access at all")
    sc.set_defaults(func=cmd_score)

    rp = sub.add_parser("report", help="join capture + judge CSV; offline")
    rp.add_argument("--capture", required=True)
    rp.add_argument("--scores", help="judge review CSV")
    rp.add_argument("--out", help="write report JSON here")
    rp.add_argument("--judge-label", default="claude-subagent-in-context",
                    help="WHICH judge produced the scores; recorded in the report")
    rp.add_argument("--deviation", action="append",
                    help="declare a deviation; forces comparable_to_anchor=false")
    rp.set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
