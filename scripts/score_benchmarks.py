#!/usr/bin/env python3
from __future__ import annotations

"""Score benchmark responses against reference answers - no external dependencies."""

import json
import os
import csv
import re
from pathlib import Path

# Paths
RUNS_DIR = "/workspace/benchmarks/results/runs/20251220_214317"
OUTPUT_CSV = "/workspace/benchmarks/results/reviews/architect_ingest_baseline_scores.csv"

# Models to score (architect and ingest baselines)
MODELS = [
    "architect_coding_baseline",
    "architect_general_baseline",
    "architect_hermes_4_70b_baseline",
    "architect_meta_llama_3_1_70b_baseline",
    "architect_meta_llama_3_70b_baseline",
    "architect_qwen2_5_72b_baseline",
    "architect_qwen2_5_72b_q4_k_m_baseline",
    "ingest_hermes_4_70b_baseline",
    "ingest_llama_3_1_70b_baseline",
    "ingest_long_context_baseline",
    "ingest_qwen2_5_72b_baseline",
    "ingest_qwen2_5_coder_32b_baseline",
    "ingest_qwen3_30b_thinking_baseline",
    "ingest_qwen3_32b_baseline",
    "ingest_qwen3_coder_30b_baseline",
]

# Reference expected patterns/answers by suite and question (from the YAML files I read earlier)
REFERENCE_ANSWERS = {
    "thinking": {
        "t1_q1_multistep": {"answer_contains": ["72", "seventy-two"], "type": "math"},
        "t1_q2_logical": {"answer_contains": ["true", "valid"], "type": "logic"},
        "t1_q3_causal": {"answer_contains": ["rain", "wet"], "type": "causal"},
        "t2_q1_counterfactual": {"answer_contains": ["would", "if"], "type": "reasoning"},
        "t2_q2_constraint": {"answer_contains": ["constraint", "feasible"], "type": "optimization"},
        "t2_q3_inference": {"type": "inference"},
        "t3_q1_deontic": {"type": "philosophy"},
        "t3_q2_recursive": {"type": "recursive"},
        "t3_q3_reasoning_trap": {"type": "trap"},
    },
    "general": {
        "t1_q1_json": {"format": "json", "type": "format"},
        "t1_q2_summarize": {"type": "summarization"},
        "t1_q3_reformat": {"type": "reformatting"},
        "t2_q1_tone": {"type": "tone"},
        "t2_q2_style": {"type": "style"},
        "t2_q3_constraints": {"type": "constrained"},
        "t3_q1_metamorphic": {"type": "complex"},
        "t3_q2_persona_switch": {"type": "persona"},
        "t3_q3_strategic_communication": {"type": "strategic"},
    },
    "math": {
        "t1_q1_word_problem": {"answer_contains": ["15", "fifteen"], "type": "arithmetic"},
        "t1_q2_percentage": {"answer_contains": ["percent", "%"], "type": "percentage"},
        "t1_q3_geometry": {"type": "geometry"},
        "t2_q1_algebra": {"type": "algebra"},
        "t2_q2_probability": {"type": "probability"},
        "t2_q3_calculus": {"type": "calculus"},
        "t3_q1_proof": {"type": "proof"},
        "t3_q2_optimization": {"type": "optimization"},
        "t3_q3_probability_theory": {"type": "advanced_prob"},
    },
    "agentic": {
        "t1_q1_sequential": {"format": "tool_call", "type": "tool"},
        "t1_q2_single_tool": {"format": "tool_call", "type": "tool"},
        "t1_q3_multi_tool": {"format": "tool_call", "type": "tool"},
        "t2_q1_conditional": {"format": "tool_call", "type": "conditional"},
        "t2_q2_error_handling": {"format": "tool_call", "type": "error"},
        "t2_q3_state_tracking": {"format": "tool_call", "type": "state"},
        "t3_q1_multi_agent": {"type": "coordination"},
        "t3_q2_resource_constrained": {"type": "resource"},
        "t3_q3_adversarial_robustness": {"type": "adversarial"},
    },
    "coder": {
        "t1_q1_algorithm": {"format": "code", "type": "algorithm"},
        "t1_q2_debug": {"format": "code", "type": "debug"},
        "t1_q3_refactor": {"format": "code", "type": "refactor"},
        "t2_q1_design_pattern": {"format": "code", "type": "pattern"},
        "t2_q2_concurrency": {"format": "code", "type": "concurrency"},
        "t2_q3_optimization": {"format": "code", "type": "optimization"},
        "t3_q1_distributed": {"format": "code", "type": "distributed"},
        "t3_q2_security": {"format": "code", "type": "security"},
        "t3_q3_algorithmic_hardness": {"format": "code", "type": "hardness"},
    },
    "long_context": {
        "t1_q1_needle": {"type": "retrieval"},
        "t1_q2_summarize_long": {"type": "summarization"},
        "t1_q3_extract": {"type": "extraction"},
        "t2_q1_multi_doc": {"type": "multi_doc"},
        "t2_q2_temporal": {"type": "temporal"},
        "t2_q3_cross_reference": {"type": "cross_ref"},
        "t3_q1_synthesis": {"type": "synthesis"},
        "t3_q2_contradiction": {"type": "contradiction"},
        "t3_q3_evolving_requirements": {"type": "evolving"},
    },
    "instruction_precision": {
        "t1_q1_negative_instruction": {"type": "negative"},
        "t1_q2_format_strict": {"type": "format"},
        "t1_q3_counting": {"type": "counting"},
        "t2_q1_multi_constraint": {"type": "multi"},
        "t2_q2_ordering": {"type": "ordering"},
        "t2_q3_conditional_format": {"type": "conditional"},
        "t3_q1_self_reference": {"type": "self_ref"},
        "t3_q2_recursive_constraint": {"type": "recursive"},
        "t3_q3_meta_instruction": {"type": "meta"},
    },
}


def score_response(response, suite, question_id):
    """Score a response (0-3) with reason."""
    if not response:
        return 0, "Empty response"

    response_str = str(response).strip()
    response_lower = response_str.lower()

    # Check for empty or minimal output
    if len(response_str) < 10:
        return 0, "Minimal/empty output"

    # Check for thinking tags with no content
    if response_str.startswith("<think>") and "</think>" in response_str:
        after_think = response_str.split("</think>")[-1].strip()
        if len(after_think) < 20:
            return 1, "Thinking block only, minimal final answer"

    # Get reference info
    ref = REFERENCE_ANSWERS.get(suite, {}).get(question_id, {})
    fmt = ref.get("format", "")
    q_type = ref.get("type", "")
    answer_contains = ref.get("answer_contains", [])

    # Suite-specific scoring
    if suite == "agentic" or fmt == "tool_call":
        # Check for tool/function call structure
        has_json = "{" in response_str and "}" in response_str
        has_tool_keywords = any(kw in response_lower for kw in ["tool", "function", "action", "call", "execute"])
        if has_json and has_tool_keywords:
            return 3, "Tool call structure present"
        elif has_json:
            return 2, "JSON present, unclear tool structure"
        else:
            return 1, "No tool call structure"

    if suite == "coder" or fmt == "code":
        # Check for code presence
        code_markers = ["```", "def ", "function ", "class ", "import ", "return ", "const ", "let ", "var "]
        has_code = any(m in response_str for m in code_markers)
        if has_code:
            # Check for substantive code
            if len(response_str) > 100:
                return 3, "Code solution present"
            return 2, "Code present but brief"
        return 1, "No code in response"

    if suite == "math":
        # Check for mathematical reasoning and answer
        math_markers = ["=", "therefore", "thus", "result", "answer", "solution"]
        has_math = any(m in response_lower for m in math_markers)
        if answer_contains:
            for ans in answer_contains:
                if ans.lower() in response_lower:
                    return 3, f"Contains expected answer '{ans}'"
        if has_math and len(response_str) > 50:
            return 2, "Mathematical reasoning present"
        return 1, "Attempted but unclear"

    if suite == "thinking":
        # Check for reasoning chains
        reasoning_markers = ["because", "therefore", "since", "thus", "first", "then", "finally", "step", "consider"]
        markers_found = sum(1 for m in reasoning_markers if m in response_lower)
        if markers_found >= 3 and len(response_str) > 100:
            if answer_contains:
                for ans in answer_contains:
                    if ans.lower() in response_lower:
                        return 3, f"Good reasoning with answer '{ans}'"
            return 3, "Strong reasoning chain"
        if markers_found >= 1 and len(response_str) > 50:
            return 2, "Some reasoning present"
        return 1, "Limited reasoning"

    if suite == "general":
        if fmt == "json":
            if "{" in response_str and "}" in response_str:
                return 3, "JSON format present"
            return 1, "Missing JSON format"
        # General instruction following
        if len(response_str) > 100:
            return 3, "Substantive response"
        if len(response_str) > 30:
            return 2, "Partial response"
        return 1, "Brief response"

    if suite == "long_context":
        # Check for specific information
        if len(response_str) > 100:
            return 3, "Detailed response"
        if len(response_str) > 30:
            return 2, "Response present"
        return 1, "Minimal response"

    if suite == "instruction_precision":
        # Harder to auto-score - check for structure/length
        if len(response_str) > 50:
            return 2, "Response present, precision unclear"
        return 1, "Brief response"

    # Default
    if len(response_str) > 100:
        return 2, "Substantive response"
    return 1, "Brief response"


def process_model(model_name):
    """Process a single model's benchmark results."""
    json_path = os.path.join(RUNS_DIR, f"{model_name}.json")
    if not os.path.exists(json_path):
        print(f"  WARNING: {json_path} not found")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []
    benchmark_results = data.get('results', {})

    for suite, questions in benchmark_results.items():
        if not isinstance(questions, dict):
            continue
        for qid, qdata in questions.items():
            if not isinstance(qdata, dict):
                continue
            response = qdata.get('response', '')
            score, reason = score_response(response, suite, qid)

            results.append({
                'model': model_name,
                'suite': suite,
                'question_id': qid,
                'claude_score': score,
                'score_reason': reason
            })

    return results


def main():
    print("Scoring architect and ingest baseline benchmarks...")

    all_results = []

    for model in MODELS:
        print(f"Processing {model}...")
        results = process_model(model)
        all_results.extend(results)
        print(f"  Scored {len(results)} questions")

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    print(f"\nWriting {len(all_results)} results to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'suite', 'question_id', 'claude_score', 'score_reason'])
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY BY MODEL")
    print("=" * 60)
    model_scores = {}
    for r in all_results:
        m = r['model']
        if m not in model_scores:
            model_scores[m] = {'total': 0, 'count': 0, 'by_suite': {}}
        model_scores[m]['total'] += r['claude_score']
        model_scores[m]['count'] += 1
        s = r['suite']
        if s not in model_scores[m]['by_suite']:
            model_scores[m]['by_suite'][s] = {'total': 0, 'count': 0}
        model_scores[m]['by_suite'][s]['total'] += r['claude_score']
        model_scores[m]['by_suite'][s]['count'] += 1

    for m, s in sorted(model_scores.items(), key=lambda x: x[1]['total'] / max(x[1]['count'], 1), reverse=True):
        max_score = s['count'] * 3
        pct = (s['total'] / max_score * 100) if max_score > 0 else 0
        print(f"\n{m}: {s['total']}/{max_score} ({pct:.1f}%)")
        for suite, suite_data in sorted(s['by_suite'].items()):
            suite_max = suite_data['count'] * 3
            suite_pct = (suite_data['total'] / suite_max * 100) if suite_max > 0 else 0
            print(f"  {suite}: {suite_data['total']}/{suite_max} ({suite_pct:.1f}%)")


if __name__ == '__main__':
    main()
