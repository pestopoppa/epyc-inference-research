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
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Calibration examples from prior scoring runs (qwen35_q4km + reap_246b)
# These anchor the scoring rubric so scores are consistent across models.
CALIBRATION_EXAMPLES = """
## Scoring calibration examples (from prior runs on other models):

### Score 3 examples:
- math/t1_q1_word_problem: "Perfect step-by-step solution: $120 after 20% discount then $129.60 after 8% tax. All three parts correct."
- coder/t2_q2_debug_complex: "Correctly identifies the race condition: read-yield-write non-atomicity. Fix uses asyncio.Lock correctly."
- agentic/t1_q3_nested_params: "Perfect JSON tool call with correct nested parameters. Clean output."
- instruction_precision/t2_q1_resist_elaboration: "Perfect: outputs only the number 4 with nothing else."
- general/t2_q1_synthesis: "Excellent synthesis balancing all three stakeholder needs. Concise and actionable."

### Score 2 examples:
- agentic/t3_q1_competing_constraints: "Structured plan but generic — lacks concrete tool call JSON."
- coder/t1_q2_refactor: "Speed version correct; memory version truncated."
- general/t3_q2_system_failure: "Good analysis but hit token cap, monitoring section cut off."
- instruction_precision/t1_q2_word_limit: "Good description, word count approximately in range but slightly ambiguous."
- math/t3_q1_analysis: "M-test approach correct but truncated before completion."

### Score 1 examples:
- coder/t3_q3_algorithmic_hardness: "Weak adversary argument; truncated before parts 2 and 3."
- general/t2_q3_schedule: "Extensive constraint analysis but never produces a final schedule. Truncated."
- math/t3_q2_combinatorics: "Involution not constructed — approaches problem but doesn't solve it."
- thinking/t3_q2_causal_inference: "Correct reasoning in think block but never produces visible final answer."

### Score 0 examples:
- coder/t1_q1_algorithm: "Entire response trapped in think block. Hit token cap while deliberating. Never produces answer."
- general/t1_q2_multistep: "HALLUCINATED elephant not in original list." (wait — this scored 3 on Qwen3.5. On REAP-246B it scored 0 for hallucination.)
- instruction_precision/t3_q2_cascading_constraints: "Multiple retry attempts but none satisfy all constraints. No valid final answer."
- math/t3_q3_probability_theory: "Entire tokens spent in think block. Never exits to produce visible answer."
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

def score_response(question_id: str, suite: str, prompt: str, response: str,
                   judge_url: str) -> tuple[int, str]:
    """Score a single response using the judge LLM."""
    user_msg = f"""Score this response.

**Suite**: {suite}
**Question ID**: {question_id}

**PROMPT**:
{prompt[:2000]}

**RESPONSE**:
{response[:4000]}

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

    data = json.dumps(payload).encode()
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


def main():
    parser = argparse.ArgumentParser(description="Score benchmark responses with Claude-as-Judge")
    parser.add_argument("--result-json", required=True, help="Path to run_benchmark result JSON")
    parser.add_argument("--output", help="Output CSV path (default: reviews/<role>_<config>.csv)")
    parser.add_argument("--judge-url", default="http://localhost:8199",
                        help="URL of the judge LLM server")
    parser.add_argument("--judge-model", help="Path to judge model GGUF (auto-start server)")
    parser.add_argument("--judge-threads", type=int, default=96, help="Threads for judge server")
    args = parser.parse_args()

    # Load results
    with open(args.result_json) as f:
        data = json.load(f)

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
            questions.append({
                "suite": suite_name,
                "question_id": qid,
                "prompt": qdata.get("prompt", ""),
                "response": qdata.get("response", ""),
                "tokens_per_second": qdata.get("tokens_per_second", 0),
            })

    print(f"Scoring {len(questions)} responses for {role}/{config}")
    print(f"Judge: {args.judge_url}")
    print(f"Output: {args.output}")

    # Score each
    results = []
    for i, q in enumerate(questions):
        score, reason = score_response(
            q["question_id"], q["suite"], q["prompt"], q["response"],
            args.judge_url
        )
        results.append({
            "suite": q["suite"],
            "question_id": q["question_id"],
            "tokens_per_second": round(q["tokens_per_second"], 1),
            "claude_score": score,
            "score_reason": reason,
        })
        status = f"{'✓' if score >= 0 else '✗'} {q['suite']}/{q['question_id']}: {score}/3"
        print(f"  [{i+1}/{len(questions)}] {status} — {reason[:80]}")

    # Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "question_id", "tokens_per_second",
                                                "claude_score", "score_reason"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    valid = [r for r in results if r["claude_score"] >= 0]
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


if __name__ == "__main__":
    main()
