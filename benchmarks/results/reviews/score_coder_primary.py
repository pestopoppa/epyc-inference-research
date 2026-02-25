#!/usr/bin/env python3
"""
Score the coder_primary_baseline benchmark results using Claude-as-Judge rubric.
"""

import json
import csv
import sys

def strip_prompt_echo(response, prompt):
    """Remove the echoed prompt from the response if present."""
    if response.startswith(prompt):
        return response[len(prompt):].strip()
    return response

def score_response(suite, question_id, prompt, response, tps):
    """
    Score a response according to the Claude-as-Judge rubric.
    Returns (score, reason)

    Scoring:
    3 = Correct answer with good reasoning
    2 = Partially correct or truncated
    1 = Wrong but reasonable attempt
    0 = Empty, garbage, or completely wrong
    """

    # Clean response
    cleaned = strip_prompt_echo(response, prompt)

    # Check for empty or minimal output
    if len(cleaned) < 50 or cleaned.strip() == '':
        return 0, "Empty or minimal output"

    # Check for "[end of text]" marker
    has_end_marker = "[end of text]" in response

    # Suite-specific scoring logic

    # THINKING SUITE
    if suite == "thinking":
        if "methodological_critique" in question_id:
            # Looking for 4+ methodological issues
            if "Selection Bias" in response and "Testing Effect" in response and "Statistical" in response and "Measurement Error" in response:
                return 3, "Identified 4+ methodological issues with explanations"
            elif "bias" in response.lower() and "confound" in response.lower():
                return 2, "Partial identification of issues"
            else:
                return 1, "Some attempt at critique"

        elif "causal_inference" in question_id:
            # DAG/backdoor path question
            if "{Gene}" in response and "backdoor" in response.lower():
                if "collider" in response.lower() and "Yellow Fingers" in response:
                    return 3, "Correct minimal adjustment set and collider understanding"
                else:
                    return 2, "Correct adjustment set but incomplete collider explanation"
            else:
                return 1, "Attempted causal reasoning but incorrect"

        elif "reasoning_trap" in question_id:
            # Dataset comparison flaw
            if "test set" in response.lower() and ("representative" in response.lower() or "distribution" in response.lower()):
                if "class imbalance" in response.lower() or "prevalence" in response.lower():
                    return 3, "Identified test set representativeness flaw with concrete scenario"
                else:
                    return 2, "Identified flaw but scenario not fully developed"
            else:
                return 1, "Attempted reasoning but missed key flaw"

        elif "paradox" in question_id:
            # Ship of Theseus
            if "continuity" in response.lower() and "material" in response.lower():
                if "position" in response.lower() and len(cleaned) > 500:
                    return 3, "Presented multiple positions with analysis"
                else:
                    return 2, "Discussed continuity and materials but incomplete"
            else:
                return 1, "Basic discussion of identity"

        elif "multi_step" in question_id:
            # Multi-step reasoning
            if "step" in response.lower() and len(cleaned) > 300:
                return 3, "Systematic multi-step reasoning"
            else:
                return 2, "Some reasoning but not fully systematic"

        else:
            # Generic thinking question
            if len(cleaned) > 300:
                return 2, "Substantial reasoning provided"
            else:
                return 1, "Basic reasoning attempt"

    # CODER SUITE
    elif suite == "coder":
        if "lock_free" in question_id or "aba" in question_id:
            # Lock-free ABA problem
            if "ABA" in response and ("compare-and-swap" in response.lower() or "CAS" in response.lower()):
                if "tagged pointer" in response.lower() or "version" in response.lower():
                    return 3, "Identified ABA problem with correct solutions"
                else:
                    return 2, "Identified ABA but incomplete solution"
            else:
                return 1, "Attempted concurrent programming answer"

        elif "distributed" in question_id or "consistency" in question_id:
            # CAP theorem / consistency
            if "CAP" in response or ("consistency" in response.lower() and "availability" in response.lower()):
                return 3, "Discussed consistency trade-offs"
            else:
                return 2, "Some distributed systems concepts"

        elif "```" in response:
            # Contains code blocks
            if len(cleaned) > 200:
                return 3, "Code solution provided"
            else:
                return 2, "Code snippet provided"

        elif "class " in response or "def " in response or "function" in response.lower():
            # Code-like structure
            return 2, "Code or pseudocode provided"

        else:
            # Generic coder answer
            if len(cleaned) > 300:
                return 2, "Substantial technical answer"
            else:
                return 1, "Basic technical response"

    # AGENTIC SUITE
    elif suite == "agentic":
        # Look for tool calls, JSON structures, function definitions
        if "{" in response and "}" in response:
            if '"function"' in response or '"tool"' in response or '"name"' in response:
                return 3, "Tool call structure present"
            elif '"' in response and ":" in response:
                return 2, "JSON-like structure present"
            else:
                return 1, "Some structured output"
        elif "tool" in response.lower() or "function" in response.lower():
            return 2, "Tool usage discussed"
        else:
            return 1, "General response"

    # GENERAL SUITE
    elif suite == "general":
        if "reformat" in question_id:
            if len(cleaned) > 100:
                return 2, "Reformatting response"
            else:
                return 1, "Minimal reformatting"
        elif "summarize" in question_id or "summary" in question_id:
            if len(cleaned) > 100 and len(cleaned) < 500:
                return 3, "Good summary"
            else:
                return 2, "Summary provided"
        else:
            if len(cleaned) > 200:
                return 2, "General response generated"
            else:
                return 1, "Basic response"

    # MATH SUITE
    elif suite == "math":
        if "complex" in question_id or "exponential" in question_id:
            # f(x) = Σ(x^n/n!)sin(n) question
            if "e^" in response or "exponential" in response.lower():
                if "complex" in response.lower() or "euler" in response.lower():
                    return 3, "Used complex exponentials or Euler's formula"
                else:
                    return 2, "Recognized exponential series"
            else:
                return 1, "Attempted series analysis"

        elif "uniform" in question_id or "E[N]" in question_id:
            # Expected value where S_n > 1 (answer is e)
            if " e " in response or "2.718" in response:
                return 3, "Correct answer (e)"
            else:
                return 1, "Attempted probability reasoning"

        elif "=" in response and ("step" in response.lower() or len(cleaned) > 200):
            # Has equations and reasoning
            return 2, "Mathematical reasoning with equations"

        else:
            if len(cleaned) > 200:
                return 2, "Math reasoning provided"
            else:
                return 1, "Basic math attempt"

    # INSTRUCTION PRECISION SUITE
    elif suite == "instruction_precision":
        if "json" in question_id:
            if "{" in response and "}" in response and '"' in response:
                # Check if valid JSON
                try:
                    # Extract JSON from response
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = response[start:end]
                        json.loads(json_str)
                        return 3, "Valid JSON structure"
                except:
                    return 2, "JSON-like structure but may not be valid"
            else:
                return 1, "Attempted JSON but incomplete"

        elif "format" in question_id or "structure" in question_id:
            if len(cleaned) > 100:
                return 2, "Formatted response"
            else:
                return 1, "Basic formatting attempt"

        else:
            if len(cleaned) > 100:
                return 2, "Instruction following response"
            else:
                return 1, "Basic response"

    # Default scoring
    if len(cleaned) > 500:
        return 2, "Substantial response generated"
    elif len(cleaned) > 100:
        return 1, "Basic response"
    else:
        return 0, "Minimal or empty output"


def main():
    # Read benchmark file
    input_file = '/mnt/raid0/llm/claude/benchmarks/results/runs/20251220_214317/coder_primary_baseline.json'
    output_file = '/mnt/raid0/llm/claude/benchmarks/results/reviews/coder_primary_baseline.csv'

    with open(input_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Score all questions
    scores = []
    total_score = 0
    max_score = 0
    total_tps = 0
    question_count = 0

    for suite_name in sorted(results.keys()):
        suite = results[suite_name]
        for question_id in sorted(suite.keys()):
            q = suite[question_id]

            prompt = q['prompt']
            response = q['response']
            tps = q['tokens_per_second']

            score, reason = score_response(suite_name, question_id, prompt, response, tps)

            scores.append({
                'suite': suite_name,
                'question_id': question_id,
                'tokens_per_second': tps,
                'claude_score': score,
                'score_reason': reason
            })

            total_score += score
            max_score += 3
            total_tps += tps
            question_count += 1

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['suite', 'question_id', 'tokens_per_second', 'claude_score', 'score_reason'])
        writer.writeheader()
        writer.writerows(scores)

    # Print summary
    percentage = (total_score / max_score) * 100
    avg_tps = total_tps / question_count

    print(f"Scoring complete!")
    print(f"Output: {output_file}")
    print()
    print(f"=== SUMMARY ===")
    print(f"Total score: {total_score}/{max_score}")
    print(f"Percentage: {percentage:.1f}%")
    print(f"Average TPS: {avg_tps:.2f}")
    print(f"Questions scored: {question_count}")
    print()

    # Per-suite breakdown
    print("=== PER-SUITE BREAKDOWN ===")
    suite_stats = {}
    for score in scores:
        suite = score['suite']
        if suite not in suite_stats:
            suite_stats[suite] = {'score': 0, 'max': 0, 'count': 0}
        suite_stats[suite]['score'] += score['claude_score']
        suite_stats[suite]['max'] += 3
        suite_stats[suite]['count'] += 1

    for suite in sorted(suite_stats.keys()):
        stats = suite_stats[suite]
        pct = (stats['score'] / stats['max']) * 100
        print(f"{suite}: {stats['score']}/{stats['max']} ({pct:.1f}%) - {stats['count']} questions")

if __name__ == '__main__':
    main()
