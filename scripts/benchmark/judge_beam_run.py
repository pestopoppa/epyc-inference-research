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
import dataclasses
import importlib.util
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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

# ── cloud backend (HS-PROF-1, operator 2026-10-22: Claude via the claude_cli cloud-judge transport, subscription auth) ──
# Reuses the orchestrator's typed_decisions.cloud_judges (registry orchestration/cloud_judges.yaml): `claude -p --json-schema
# --restricted --tools "" --strict-mcp-config`, prompt on stdin, empty tmp cwd, own process group, timeout kill. No API key
# is handled anywhere in this file.
_CLOUD_JUDGES_PATH = Path("/mnt/raid0/llm/epyc-orchestrator/src/typed_decisions/cloud_judges.py")
_CLOUD_KEY = "epyc_orch_cloud_judges"
DEFAULT_CLOUD_JUDGE = "sonnet-low"
DEFAULT_CLOUD_WORKERS = 2
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
              "cache_creation_input_tokens", "total_cost_usd", "calls")


def load_cloud_judges():
    """The orchestrator's cloud_judges module, loaded by absolute path (same pattern as the debug_scorer shim)."""
    if _CLOUD_KEY in sys.modules:
        return sys.modules[_CLOUD_KEY]
    if not _CLOUD_JUDGES_PATH.is_file():
        raise ImportError(f"orchestrator cloud_judges not found: {_CLOUD_JUDGES_PATH}")
    spec = importlib.util.spec_from_file_location(_CLOUD_KEY, _CLOUD_JUDGES_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_CLOUD_KEY] = mod
    spec.loader.exec_module(mod)
    return mod


class CloudJudge:
    """``judge(prompt) -> text`` over one registered cloud judge; keeps THIS instance's usage.

    One instance per question (judge_run reads ``.usage`` / ``.models`` afterwards), so concurrent
    questions never share counters. The claude envelope is captured by wrapping ``run_fn``
    (the registry transport returns only the schema object, not usage/cost).
    """

    def __init__(self, cj, spec, model: str | None, *, run_fn=None):
        self._cj, self._spec, self._model = cj, spec, model
        self._inner = run_fn or cj._default_run
        self.usage = {k: 0 for k in USAGE_KEYS}
        self.models: set[str] = set()

    def _run(self, argv, stdin_text, timeout_s, cwd):
        rc, out, err = self._inner(argv, stdin_text, timeout_s, cwd)
        env = self._cj._first_object(out) or {}
        u = env.get("usage") if isinstance(env.get("usage"), dict) else {}
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.usage[k] += int(u.get(k) or 0)
        self.usage["total_cost_usd"] += float(env.get("total_cost_usd") or 0.0)
        self.usage["calls"] += 1
        mu = env.get("modelUsage")
        if isinstance(mu, dict):
            self.models.update(str(k) for k in mu)
        return rc, out, err

    def __call__(self, prompt: str) -> str:
        try:
            res = self._cj.run_cloud_judge(self._spec, prompt, NUGGET_OUTPUT_SCHEMA,
                                           model=self._model, run_fn=self._run)
        except self._cj.CloudJudgeError as exc:
            raise JudgeUnavailable(str(exc)) from exc
        return json.dumps(res.payload)


def cloud_preflight(cj, spec, *, run_fn=None) -> None:
    """One trivial call (60 s cap): fail fast if ``claude`` is missing or not logged in."""
    probe_schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    if run_fn is None and not (shutil.which("claude") or Path(cj.CLAUDE_BIN).exists()):
        raise JudgeUnavailable("claude CLI not found on PATH")
    try:
        cj.run_cloud_judge(dataclasses.replace(spec, timeout_s=60.0), 'Reply with {"ok": true}.',
                           probe_schema, run_fn=run_fn)
    except cj.CloudJudgeError as exc:
        raise JudgeUnavailable(f"claude not authenticated or unreachable: {exc}") from exc


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
            "approx_output_tokens": calls * 60}


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
    workers: int = 1,
) -> dict:
    """Judge every stored BEAM response. Returns ``{"records", "unjudged"}``.

    ``judge_for(scoring_config)`` builds the judge for one question, so a test
    can inject a fake and production binds the served transport. ``workers`` > 1 judges that many
    QUESTIONS concurrently (nuggets within a question stay sequential); results are assembled in
    sorted question order and the partial file is appended under a lock as each question finishes.
    A judge exposing ``usage`` / ``models`` has them recorded on its question's row.
    """
    results = payload.get("results", {}).get("beam")
    if not isinstance(results, Mapping) or not results:
        raise BEAMFoldError("result file carries no results['beam'] rows")
    done = _load_partial(partial_path) if partial_path else {}
    lock = threading.Lock()

    def one(question_id: str) -> dict:
        row = results[question_id]
        prompt = prompt_index.get(question_id)
        if prompt is None:
            outcome = {"question_id": question_id, "reason": "unknown_question_id"}
        elif not isinstance(row.get("response"), str) or not row["response"].strip():
            outcome = {"question_id": question_id, "reason": "empty_response"}
        else:
            judge = judge_for(prompt["scoring_config"])
            try:
                outcome = judge_question(prompt, row["response"], judge)
                outcome["judge_prompt_version"] = JUDGE_PROMPT_VERSION
            except JudgeUnavailable as exc:
                outcome = {"question_id": question_id, "reason": "judge_unavailable",
                           "detail": str(exc)[:240]}
            except NuggetVerdictError as exc:
                outcome = {"question_id": question_id, "reason": "unparseable_verdict",
                           "detail": str(exc)[:240]}
            if isinstance(getattr(judge, "usage", None), dict):
                outcome["judge_usage"] = dict(judge.usage)  # spent even when the verdict failed
                outcome["judge_response_models"] = sorted(getattr(judge, "models", ()))
        if partial_path is not None:
            with lock, open(partial_path, "a") as handle:
                handle.write(json.dumps(outcome, sort_keys=True) + "\n")
        return outcome

    todo = [q for q in sorted(results) if q not in done]
    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fresh = dict(zip(todo, pool.map(one, todo)))
    else:
        fresh = {q: one(q) for q in todo}
    records: list[dict] = []
    unjudged: list[dict] = []
    for question_id in sorted(results):
        outcome = done[question_id] if question_id in done else fresh[question_id]
        (records if "nugget_verdicts" in outcome else unjudged).append(outcome)
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
                         cloud_judge: str | None = None) -> dict:
    if any(row.get("judge_prompt_version") != JUDGE_PROMPT_VERSION for row in judged["records"]):
        raise BEAMFoldError("cannot label judged rows with missing or incompatible judge_prompt_version")
    identity: dict[str, Any] = {"judge_backend": judge_backend}
    if judge_backend == "cloud":
        totals: dict[str, float] = {k: 0 for k in USAGE_KEYS}
        models: set[str] = set()
        for row in list(judged["records"]) + list(judged["unjudged"]):  # unjudged calls were paid for too
            for k, v in (row.get("judge_usage") or {}).items():
                totals[k] = totals.get(k, 0) + v
            models.update(row.get("judge_response_models") or ())
        totals["total_cost_usd"] = round(totals["total_cost_usd"], 6)
        identity.update({"cloud_judge": cloud_judge, "judge_response_models": sorted(models),
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
                             "--judge-backend served; defaults to the cloud judge's pinned model for cloud")
    parser.add_argument("--judge-backend", choices=("served", "cloud"), default="served",
                        help="served = llama-server via the orchestrator transport (default); "
                             "cloud = orchestration/cloud_judges.yaml claude_cli (subscription CLI, no API key)")
    parser.add_argument("--cloud-judge", default=DEFAULT_CLOUD_JUDGE,
                        help="cloud_judges.yaml registry name (cloud backend)")
    parser.add_argument("--judge-model-id", default=None,
                        help="Override the registry entry's model; must be in its allow-list")
    parser.add_argument("--workers", type=int, default=DEFAULT_CLOUD_WORKERS,
                        help="questions judged concurrently (cloud backend; default 2)")
    parser.add_argument("--split", default="100K")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--judge-url", default=None, help="Direct llama-server judge base URL")
    parser.add_argument("--judge-role", default=None, help="Orchestrator force_role for the judge")
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args(argv)
    cj = spec = None
    if args.judge_backend == "cloud":
        cj = load_cloud_judges()
        registry = cj.load_registry()
        if args.cloud_judge not in registry:
            parser.error(f"unknown --cloud-judge {args.cloud_judge!r}; registry has {sorted(registry)}")
        spec = registry[args.cloud_judge]
        if spec.transport != "claude_cli":
            parser.error("HS-PROF-1 cloud judge must be a claude_cli entry (judge != reader family)")
        try:
            args.judge_model_id = spec.resolve_model(args.judge_model_id)
        except ValueError as exc:
            parser.error(str(exc))
        args.judge_model = args.judge_model or f"{args.cloud_judge}:{args.judge_model_id}"
    else:
        if args.judge_model_id:
            parser.error("--judge-model-id applies to --judge-backend cloud only")
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
    workers = 1
    if cj is not None:
        est = estimate_judge_calls(payload, prompt_index, done)
        print("judge estimate (before any call): " + json.dumps(
            {**est, "backend": "cloud", "cloud_judge": args.cloud_judge, "model": args.judge_model_id,
             "workers": args.workers, "already_judged_questions": len(done)}), file=sys.stderr)
        if est["judge_calls"]:
            try:
                cloud_preflight(cj, spec)
            except JudgeUnavailable as exc:
                print(f"BLOCKED: {exc}", file=sys.stderr)
                return 5
        workers = max(1, args.workers)
        judge_for = lambda cfg: CloudJudge(cj, spec, args.judge_model_id)  # noqa: E731
    else:
        judge_for = lambda cfg: served_judge(cfg, overrides)  # noqa: E731
    judged = judge_run(payload, prompt_index, judge_for=judge_for, partial_path=partial, workers=workers)
    out = build_judged_payload(payload, judged, split=args.split, judge_model=args.judge_model,
                               judge_backend=args.judge_backend,
                               cloud_judge=args.cloud_judge if cj else None)
    if cj:
        out["judge_model_id"] = args.judge_model_id
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"judged": len(out["records"]), "unjudged": len(out["unjudged"]),
                      "out": str(args.out_json)}))
    return 0 if not out["unjudged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
