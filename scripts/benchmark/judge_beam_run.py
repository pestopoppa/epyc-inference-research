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
empty. Verdicts are persisted per question to ``<out>.partial.jsonl`` and a
rerun resumes from that file.

The judge identity (``--judge-model``) is required: a BEAM headline with no judge
recorded cannot be compared to anything.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["question_id"]] = row
    return done


def judge_run(
    payload: Mapping[str, Any],
    prompt_index: Mapping[str, dict],
    *,
    judge_for: Callable[[Mapping[str, Any]], Judge],
    partial_path: Path | None = None,
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
            prior = done[question_id]
            (records if "nugget_verdicts" in prior else unjudged).append(prior)
            continue
        prompt = prompt_index.get(question_id)
        if prompt is None:
            outcome = {"question_id": question_id, "reason": "unknown_question_id"}
        elif not isinstance(row.get("response"), str) or not row["response"].strip():
            outcome = {"question_id": question_id, "reason": "empty_response"}
        else:
            try:
                outcome = judge_question(prompt, row["response"],
                                         judge_for(prompt["scoring_config"]))
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


def build_judged_payload(payload: Mapping[str, Any], judged: Mapping[str, Any], *,
                         split: str, judge_model: str) -> dict:
    return {
        "schema": JUDGED_SCHEMA,
        "run_id": payload.get("run_id"),
        "model_role": payload.get("model_role"),
        "config_name": payload.get("config_name"),
        "split": split,
        "judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "question_in_judge_prompt": QUESTION_IN_JUDGE_PROMPT,
        "result_questions": len(payload.get("results", {}).get("beam", {})),
        "records": list(judged["records"]),
        "unjudged": list(judged["unjudged"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=Path, help="run_benchmark result JSON with results['beam']")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--judge-model", required=True,
                        help="Identity of the judge that answers (model + quant)")
    parser.add_argument("--split", default="100K")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--judge-url", default=None, help="Direct llama-server judge base URL")
    parser.add_argument("--judge-role", default=None, help="Orchestrator force_role for the judge")
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args(argv)

    from long_context_adapters import BEAMAdapter

    payload = json.loads(args.result.read_text())
    adapter = BEAMAdapter(data_dir=args.data_dir, split=args.split)
    prompt_index = {p["id"]: p for p in adapter.extract_all()}
    if not prompt_index:
        raise SystemExit(f"no BEAM {args.split} data loaded: "
                         f"{adapter.accounting_summary()['degraded_sources']}")
    overrides = {"judge_url": args.judge_url, "judge_role": args.judge_role,
                 "timeout": args.timeout}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out_json.with_name(args.out_json.name + ".partial.jsonl")
    judged = judge_run(payload, prompt_index,
                       judge_for=lambda cfg: served_judge(cfg, overrides),
                       partial_path=partial)
    out = build_judged_payload(payload, judged, split=args.split, judge_model=args.judge_model)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"judged": len(out["records"]), "unjudged": len(out["unjudged"]),
                      "out": str(args.out_json)}))
    return 0 if not out["unjudged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
