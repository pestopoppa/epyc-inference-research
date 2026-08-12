#!/usr/bin/env python3
"""Score benchmark responses using Claude-as-Judge.

Reads responses from run_benchmark.py result JSON, scores each on a 0-3 scale
using the same rubric as prior scoring runs. Outputs a review CSV compatible
with rebuild_summary.py.

Usage:
    python3 score_with_claude.py --result-json <path> --output <csv>
    python3 score_with_claude.py --result-json benchmarks/results/runs/20260326_150453/nemotron_cascade_2_baseline.json
"""
import argparse
import csv
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

# Calibration examples from prior scoring runs (qwen35_q4km + reap_246b)
# These anchor the scoring rubric so scores are consistent across models.
CALIBRATION_EXAMPLES = """
## Scoring calibration anchors:

Generic level anchors only. Per-question calibration examples were STRIPPED
2026-08-12 (architect-model-selection-bench.md L176): they named specific
question_ids with the scores they received on OTHER models, priming the judge
on item identity before it read the answer. Absolute score levels are NOT
comparable across this boundary (pre-strip runs, e.g. the 2026-08-02
head-to-head, were scored with the priming); A-vs-B comparisons where both
arms share one judge version are unaffected.

- Score 3: fully correct AND complete; every asked part answered; format
  constraints met exactly.
- Score 2: right approach with real progress, but incomplete, truncated, or
  a secondary part wrong or missing.
- Score 1: engages the problem but does not solve it; major parts missing or
  the core answer wrong despite relevant work.
- Score 0: wrong, absent, hallucinated, or nothing visible outside think tags.
"""

SYSTEM_PROMPT = """You are a benchmark scorer evaluating LLM responses on a 0-3 scale.

## Scoring rubric:
- **3**: Correct, complete, well-structured answer. All parts of the question addressed.
- **2**: Mostly correct but incomplete (truncation, missing parts) OR correct approach with minor errors.
- **1**: Partially relevant but significantly incomplete, wrong approach, or major errors.
- **0**: Wrong answer, hallucination, complete truncation (stuck in think block), or failure to address the question.

## Important scoring rules:
- If the response is entirely inside <think> tags with no visible answer after </think>, score 0 (truncation failure).
- If the response has a valid answer after </think> but was cut short, score based on what's visible (typically 1-2).
- Hallucinated facts or data not in the prompt = score 0.
- For instruction_precision: exact constraint compliance required for score 3.
- For agentic: valid JSON tool calls required for score 3.
- For math: correct final answer required for score 3; correct approach but wrong answer = score 1-2.
- Think-block leakage (visible deliberation) is a format issue, reduce by 1 point if answer is otherwise correct.

{calibration}

## Output format:
Respond with ONLY a JSON object: {{"score": <0-3>, "reason": "<1-2 sentence explanation>"}}
Do not include any other text."""

CURRENT_CAPTURE_SCHEMA = "v7_quality_gate_capture.v4"

# --- Suite retirement (binding, machine-readable) ---------------------------
# Origin: 2026-08-02 judge-suite head-to-head (data/judge_suite_headtohead_20260802):
# 50 of 68 questions scored a perfect 3 for BOTH arms; `general` was 10/10
# both-perfect, `thinking` 9/10, `math` 8/9. That is a ceiling artifact, not
# model equivalence (published benchmarks separate the same pair on 8 of 8
# axes). Those suites carry no discriminating information at the >=27B tier and
# are RETIRED for any comparative/keep-drop read there. The retirement data
# lives in SUITE_RETIREMENTS_PATH; consumption is FAIL-CLOSED: a missing or
# invalid sidecar is an error, never an implicit "nothing is retired".
RETIREMENT_SCHEMA = "suite_retirement.v1"
SUITE_RETIREMENTS_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks/prompts/v1/suite_retirements.json"
)


class SuiteRetirementError(RuntimeError):
    """The suite-retirement sidecar is missing or invalid (fail-closed)."""


def load_suite_retirements(path: Path | None = None) -> dict[str, dict]:
    """Load the suite-retirement sidecar, refusing loudly on any defect.

    Returns the ``retired_for_discrimination`` mapping (suite name -> entry).
    An absent, unreadable, or malformed sidecar raises SuiteRetirementError:
    without the sidecar no suite can be certified as discriminating, so the
    caller must refuse to present comparative output rather than silently
    treat every suite as live.
    """
    path = Path(path) if path is not None else SUITE_RETIREMENTS_PATH
    refusal = (
        "FAIL-CLOSED: without a valid retirement sidecar no suite can be "
        "certified as discriminating; refusing to produce comparative output. "
        f"Expected sidecar: {path}"
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SuiteRetirementError(
            f"suite-retirement sidecar unreadable ({exc}). {refusal}") from exc
    except json.JSONDecodeError as exc:
        raise SuiteRetirementError(
            f"suite-retirement sidecar is not valid JSON ({exc}). {refusal}") from exc
    if not isinstance(data, dict) or data.get("schema") != RETIREMENT_SCHEMA:
        raise SuiteRetirementError(
            f"suite-retirement sidecar schema is not {RETIREMENT_SCHEMA!r}. {refusal}")
    retired = data.get("retired_for_discrimination")
    if not isinstance(retired, dict):
        raise SuiteRetirementError(
            f"suite-retirement sidecar has no retired_for_discrimination map. {refusal}")
    for suite, entry in retired.items():
        if not isinstance(entry, dict):
            raise SuiteRetirementError(
                f"retirement entry for {suite!r} is not an object. {refusal}")
        if not isinstance(entry.get("min_params_b"), (int, float)):
            raise SuiteRetirementError(
                f"retirement entry for {suite!r} lacks numeric min_params_b. {refusal}")
        for field in ("tier", "both_perfect", "measured", "reason"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise SuiteRetirementError(
                    f"retirement entry for {suite!r} lacks {field}. {refusal}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise SuiteRetirementError(
                f"retirement entry for {suite!r} lacks an evidence list. {refusal}")
    return retired


# Total-parameter count from a model path/name: take the LARGEST `<num>B` token
# not preceded by a letter/digit, so active-expert suffixes like `-A10B` in
# `122B-A10B` are ignored while `122B` is kept.
_PARAMS_B_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)[bB](?![A-Za-z0-9])")


def parse_model_params_b(name: str) -> float | None:
    """Best-effort total-parameter count (billions) from a model path or name."""
    if not name:
        return None
    matches = _PARAMS_B_RE.findall(os.path.basename(str(name)))
    if not matches:
        return None
    return max(float(m) for m in matches)


def retirement_stamp(suite: str, retirements: dict[str, dict],
                     params_b: float | None) -> str:
    """Return the loud non-discriminating stamp for a retired suite, else "".

    Tier handling is fail-closed: a model whose parameter count cannot be
    resolved is treated as at-tier, because an unknown tier cannot certify the
    suite as discriminating.
    """
    entry = retirements.get(suite)
    if entry is None:
        return ""
    if params_b is not None and params_b < float(entry["min_params_b"]):
        return ""
    tier_note = "" if params_b is not None else " model-tier-unresolved:fail-closed"
    return (
        f"RETIRED_NON_DISCRIMINATING tier={entry['tier']} "
        f"both_perfect={entry['both_perfect']} measured={entry['measured']} "
        f"evidence={entry['evidence'][0]}{tier_note}"
    )


def text_identity(text: str) -> dict[str, int | str]:
    """Return lossless UTF-8 identity metadata for an input string."""
    encoded = text.encode("utf-8")
    return {
        "utf8_chars": len(text),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def producer_fingerprint(text: str) -> dict[str, int | str]:
    """Match the runner's persisted capture fingerprint exactly."""
    encoded = text.encode("utf-8")
    return {"chars": len(text), "utf8_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest()}


def capture_eligibility(row: dict, *, allow_provisional_legacy: bool,
                        expected_source_sha256: str | None,
                        pinned_prompt: str | None) -> str:
    """Return an auditable status before a response is sent to a judge."""
    schema = row.get("capture_schema_version")
    if schema != CURRENT_CAPTURE_SCHEMA:
        if allow_provisional_legacy:
            return "provisional_legacy_capture"
        return "missing_or_legacy_capture"
    if row.get("runner_source_sha256") != expected_source_sha256:
        return "invalid_runner_source"
    if row.get("request_error"):
        return "producer_request_error"
    for field in ("prompt", "response", "reasoning"):
        if not isinstance(row.get(field), str):
            return f"missing_{field}"
    for field in ("prompt", "response", "reasoning"):
        if row.get(f"{field}_fingerprint") != producer_fingerprint(row[field]):
            return f"{field}_fingerprint_mismatch"
    if pinned_prompt is None or row["prompt"] != pinned_prompt:
        return "prompt_not_bound_to_pinned_questions"
    return "eligible"


def load_pinned_prompts(path: Path) -> dict[tuple[str, str], str]:
    """Read runner ``--questions-out`` evidence as immutable scorer bindings."""
    data = json.loads(path.read_text(encoding="utf-8"))
    suites = data.get("suites", {})
    if not isinstance(suites, dict):
        raise ValueError("pinned questions must contain a suites object")
    prompts = {}
    for suite, questions in suites.items():
        if not isinstance(questions, list):
            raise ValueError(f"pinned questions for {suite} are not a list")
        for index, question in enumerate(questions):
            if not isinstance(question, dict) or not isinstance(question.get("prompt"), str):
                raise ValueError(f"pinned question {suite}/{index} has no prompt")
            question_id = question.get("id", f"{suite}_{index:04d}")
            prompts[(suite, str(question_id))] = question["prompt"]
    return prompts


def build_judge_input(question_id: str, suite: str, prompt: str, response: str) -> dict:
    """Build the complete scorer payload and its audit metadata.

    This boundary is deliberately lossless: callers must either submit the full
    prompt and response or mark the row ineligible before an LLM judge sees it.
    """
    user_msg = f"""Score this response.

**Suite**: {suite}
**Question ID**: {question_id}

**PROMPT**:
{prompt}

**RESPONSE**:
{response}

Score (0-3) with reason:"""

    payload = {
        "model": "judge",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(calibration=CALIBRATION_EXAMPLES)},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 150,
        "temperature": 0.0
    }
    scorer_input = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")).encode("utf-8")
    return {
        "payload": payload,
        "serialized_payload": scorer_input,
        "prompt_identity": text_identity(prompt),
        "response_identity": text_identity(response),
        "scorer_input_utf8_bytes": len(scorer_input),
        "scorer_input_sha256": hashlib.sha256(scorer_input).hexdigest(),
    }


def score_response(question_id: str, suite: str, prompt: str, response: str,
                   judge_url: str, judge_input_budget_bytes: int | None = None,
                   scorer_input: dict | None = None) -> tuple[int, str]:
    """Score a single response using the judge LLM without truncating its input."""
    scorer_input = scorer_input or build_judge_input(
        question_id, suite, prompt, response
    )
    if (judge_input_budget_bytes is not None
            and scorer_input["scorer_input_utf8_bytes"] > judge_input_budget_bytes):
        return (
            -1,
            "provisional_input_over_budget: "
            f"{scorer_input['scorer_input_utf8_bytes']} bytes exceeds configured "
            f"judge budget {judge_input_budget_bytes} bytes; row not scored",
        )

    # Send exactly the byte sequence whose fingerprint is persisted in the review row.
    data = scorer_input["serialized_payload"]
    req = urllib.request.Request(
        f"{judge_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        raw = resp.read().decode()
        r = json.loads(raw)
        content = r["choices"][0]["message"]["content"].strip()

        # Parse JSON from response
        # Handle cases where model wraps in markdown
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)
        return int(result["score"]), result["reason"]
    except Exception as e:
        return -1, f"SCORING ERROR: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score benchmark responses with Claude-as-Judge")
    parser.add_argument("--result-json", required=True, help="Path to run_benchmark result JSON")
    parser.add_argument("--output", help="Output CSV path (default: reviews/<role>_<config>.csv)")
    parser.add_argument("--judge-url", default="http://localhost:8199",
                        help="URL of the judge LLM server")
    parser.add_argument("--judge-model", help="Path to judge model GGUF (auto-start server)")
    parser.add_argument("--judge-threads", type=int, default=96, help="Threads for judge server")
    parser.add_argument(
        "--judge-input-budget-bytes", type=int,
        help=("Maximum UTF-8 bytes for the complete judge payload. Rows above this "
              "budget are recorded as provisional and are never truncated or scored."),
    )
    parser.add_argument("--allow-provisional-legacy", action="store_true",
                        help="Inspect legacy captures only; output remains provisional and exits nonzero")
    parser.add_argument("--producer-source", type=Path,
                        help="Reviewed v4 runner source; required when scoring v4 capture")
    parser.add_argument("--pinned-questions", type=Path,
                        help="Runner --questions-out artifact; required when scoring v4 capture")
    args = parser.parse_args()
    if args.judge_input_budget_bytes is not None and args.judge_input_budget_bytes <= 0:
        parser.error("--judge-input-budget-bytes must be positive")

    # FAIL-CLOSED retirement gate: refuse to score at all when the sidecar is
    # missing or invalid. Deleting the retirement metadata must be loud, never
    # a silent return to "every suite is discriminating".
    try:
        suite_retirements = load_suite_retirements()
    except SuiteRetirementError as exc:
        parser.error(str(exc))

    # Load results
    with open(args.result_json) as f:
        data = json.load(f)
    raw_rows = [qdata for suite_data in data.get("results", {}).values()
                if isinstance(suite_data, dict) for qdata in suite_data.values()
                if isinstance(qdata, dict) and "response" in qdata]
    has_current_capture = any(row.get("capture_schema_version") == CURRENT_CAPTURE_SCHEMA for row in raw_rows)
    if has_current_capture and (args.producer_source is None or args.pinned_questions is None):
        parser.error("v4 capture scoring requires --producer-source and --pinned-questions")
    if args.producer_source is not None and not args.producer_source.is_file():
        parser.error("--producer-source must name a readable file")
    if args.pinned_questions is not None and not args.pinned_questions.is_file():
        parser.error("--pinned-questions must name a readable file")
    expected_source_sha256 = (
        hashlib.sha256(args.producer_source.read_bytes()).hexdigest()
        if args.producer_source is not None else None
    )
    try:
        pinned_prompts = load_pinned_prompts(args.pinned_questions) if args.pinned_questions else {}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid --pinned-questions: {exc}")

    role = data["model_role"]
    config = data["config_name"]
    # Model tier for retirement scoping: model_path first, then role/config as
    # weaker hints. None (unresolvable) is treated as at-tier by
    # retirement_stamp (fail-closed).
    params_b = None
    for tier_source in (data.get("model_path"), role, config):
        params_b = parse_model_params_b(str(tier_source or ""))
        if params_b is not None:
            break

    if not args.output:
        args.output = f"/mnt/raid0/llm/epyc-inference-research/benchmarks/results/reviews/{role}_{config}.csv"

    # Collect all questions
    questions = []
    for suite_name, suite_data in data["results"].items():
        if not isinstance(suite_data, dict):
            continue
        for qid, qdata in suite_data.items():
            if not isinstance(qdata, dict) or "response" not in qdata:
                continue
            capture_status = capture_eligibility(
                qdata, allow_provisional_legacy=args.allow_provisional_legacy,
                expected_source_sha256=expected_source_sha256,
                pinned_prompt=pinned_prompts.get((suite_name, str(qid))),
            )
            questions.append({
                "suite": suite_name,
                "question_id": qid,
                "prompt": qdata.get("prompt", ""),
                "response": qdata.get("response", ""),
                "tokens_per_second": qdata.get("tokens_per_second", 0),
                "finish_reason": qdata.get("finish_reason"),
                "usage": qdata.get("usage"),
                "capture_status": capture_status,
                "suite_retirement": retirement_stamp(
                    suite_name, suite_retirements, params_b),
            })

    print(f"Scoring {len(questions)} responses for {role}/{config}")
    print(f"Judge: {args.judge_url}")
    print(f"Output: {args.output}")

    # Score each
    results = []
    for i, q in enumerate(questions):
        if q["capture_status"] != "eligible":
            results.append({
                "suite": q["suite"], "question_id": q["question_id"],
                "tokens_per_second": round(q["tokens_per_second"], 1),
                "claude_score": -1, "score_reason": q["capture_status"],
                "score_eligibility": q["capture_status"],
                "suite_retirement": q["suite_retirement"],
                "prompt_utf8_chars": "", "prompt_utf8_bytes": "", "prompt_sha256": "",
                "response_utf8_chars": "", "response_utf8_bytes": "", "response_sha256": "",
                "scorer_input_utf8_bytes": "", "scorer_input_sha256": "",
                "finish_reason": q["finish_reason"],
                "usage": json.dumps(q["usage"], ensure_ascii=False, sort_keys=True)
                if q["usage"] is not None else "",
            })
            continue
        scorer_input = build_judge_input(
            q["question_id"], q["suite"], q["prompt"], q["response"]
        )
        score, reason = score_response(
            q["question_id"], q["suite"], q["prompt"], q["response"],
            args.judge_url, args.judge_input_budget_bytes, scorer_input
        )
        eligible = score >= 0
        results.append({
            "suite": q["suite"],
            "question_id": q["question_id"],
            "tokens_per_second": round(q["tokens_per_second"], 1),
            "claude_score": score,
            "score_reason": reason,
            "score_eligibility": "eligible" if eligible else (
                "provisional_input_over_budget" if reason.startswith(
                    "provisional_input_over_budget:") else "scoring_error"
            ),
            "suite_retirement": q["suite_retirement"],
            "prompt_utf8_chars": scorer_input["prompt_identity"]["utf8_chars"],
            "prompt_utf8_bytes": scorer_input["prompt_identity"]["utf8_bytes"],
            "prompt_sha256": scorer_input["prompt_identity"]["sha256"],
            "response_utf8_chars": scorer_input["response_identity"]["utf8_chars"],
            "response_utf8_bytes": scorer_input["response_identity"]["utf8_bytes"],
            "response_sha256": scorer_input["response_identity"]["sha256"],
            "scorer_input_utf8_bytes": scorer_input["scorer_input_utf8_bytes"],
            "scorer_input_sha256": scorer_input["scorer_input_sha256"],
            "finish_reason": q["finish_reason"],
            "usage": json.dumps(q["usage"], ensure_ascii=False, sort_keys=True)
            if q["usage"] is not None else "",
        })
        status = f"{'✓' if score >= 0 else '✗'} {q['suite']}/{q['question_id']}: {score}/3"
        print(f"  [{i+1}/{len(questions)}] {status} — {reason[:80]}")

    # Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "suite", "question_id", "tokens_per_second", "claude_score", "score_reason",
            "score_eligibility", "suite_retirement",
            "prompt_utf8_chars", "prompt_utf8_bytes", "prompt_sha256",
            "response_utf8_chars", "response_utf8_bytes", "response_sha256",
            "scorer_input_utf8_bytes", "scorer_input_sha256", "finish_reason", "usage",
        ])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    ineligible = [r for r in results if r["claude_score"] < 0]
    if ineligible:
        print(f"\n{'='*60}")
        print(f"SCORING PROVISIONAL: {role}/{config}")
        print(f"{'='*60}")
        print(
            f"{len(ineligible)}/{len(results)} rows are ineligible; "
            "no aggregate score is decision-grade."
        )
        print(f"\nResults written to {args.output}")
        return 2

    valid = results
    if valid:
        # Retired suites are still scored and recorded, but they carry no
        # discriminating information at this model tier: they are excluded
        # from the comparative aggregate and every mention is stamped.
        discriminating = [r for r in valid if not r["suite_retirement"]]
        retired_stamps = {r["suite"]: r["suite_retirement"]
                          for r in valid if r["suite_retirement"]}
        print(f"\n{'='*60}")
        print(f"SCORING COMPLETE: {role}/{config}")
        print(f"{'='*60}")
        if discriminating:
            total_score = sum(r["claude_score"] for r in discriminating)
            total_max = len(discriminating) * 3
            passed = sum(1 for r in discriminating if r["claude_score"] >= 2)
            scope = " (discriminating suites only)" if retired_stamps else ""
            print(f"Total{scope}: {total_score}/{total_max} ({total_score/total_max*100:.0f}%)")
            print(f"Passed (≥2){scope}: {passed}/{len(discriminating)} "
                  f"({passed/len(discriminating)*100:.0f}%)")

        # Per-suite
        from collections import defaultdict
        per_suite = defaultdict(list)
        for r in valid:
            per_suite[r["suite"]].append(r["claude_score"])
        for suite, scores in sorted(per_suite.items()):
            s = sum(scores)
            m = len(scores) * 3
            p = sum(1 for x in scores if x >= 2)
            line = f"  {suite}: {s}/{m} ({s/m*100:.0f}%) | {p}/{len(scores)} pass"
            if suite in retired_stamps:
                line += f"  *** NON-DISCRIMINATING: {retired_stamps[suite]} ***"
            print(line)

        if retired_stamps:
            print("\n*** SUITE RETIREMENT ***")
            print("Retired suites present: " + ", ".join(sorted(retired_stamps)))
            print("Their scores are recorded but carry NO discriminating information")
            print("at this model tier and MUST NOT feed any comparative/keep-drop read.")
            print(f"Metadata: {SUITE_RETIREMENTS_PATH}")

        if not discriminating:
            print("\n" + "!" * 60)
            print("NO DISCRIMINATING SUITES IN THIS RUN: every scored suite is")
            print("RETIRED at this model tier. This output supports NO")
            print("comparative/keep-drop read.")
            print("!" * 60)
            print(f"\nResults written to {args.output}")
            return 3

    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
