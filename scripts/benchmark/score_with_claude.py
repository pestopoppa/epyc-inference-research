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
            "score_eligibility", "prompt_utf8_chars", "prompt_utf8_bytes", "prompt_sha256",
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
        total_score = sum(r["claude_score"] for r in valid)
        total_max = len(valid) * 3
        passed = sum(1 for r in valid if r["claude_score"] >= 2)
        print(f"\n{'='*60}")
        print(f"SCORING COMPLETE: {role}/{config}")
        print(f"{'='*60}")
        print(f"Total: {total_score}/{total_max} ({total_score/total_max*100:.0f}%)")
        print(f"Passed (≥2): {passed}/{len(valid)} ({passed/len(valid)*100:.0f}%)")

        # Per-suite
        from collections import defaultdict
        per_suite = defaultdict(list)
        for r in valid:
            per_suite[r["suite"]].append(r["claude_score"])
        for suite, scores in sorted(per_suite.items()):
            s = sum(scores)
            m = len(scores) * 3
            p = sum(1 for x in scores if x >= 2)
            print(f"  {suite}: {s}/{m} ({s/m*100:.0f}%) | {p}/{len(scores)} pass")

    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
