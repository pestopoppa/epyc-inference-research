#!/usr/bin/env python3
from __future__ import annotations

"""
Benchmark Output Scoring Script
===============================
Scores thinking rubric outputs against predefined criteria using pattern matching.
No API calls required - can run fully offline.

Usage:
    python3 score_outputs.py [--input-dir DIR] [--output FILE] [--summary]

Output format: JSONL with one entry per scored output
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# =============================================================================
# SCORING CRITERIA
# =============================================================================

CRITERIA = {
    # T1: Baseline (straightforward)
    "t1_q1_algorithm": {
        "question": "Sort 10 mostly-sorted items. Quicksort or insertion sort?",
        "correct_answer": "insertion sort",
        "score_3": [r"insertion\s*sort", r"nearly.?sorted", r"O\(n\)"],
        "score_2": [r"insertion"],
        "score_1": [r"small", r"sorted"],
        "wrong_indicators": [r"quicksort.*better", r"choose.*quicksort"],
    },
    "t1_q2_threadsafe": {
        "question": "Is self.count += 1 thread-safe in Python?",
        "correct_answer": "no",
        "score_3": [r"not?\s*(thread.?safe|atomic)", r"race\s*condition", r"GIL.*not.*guarantee"],
        "score_2": [r"no[^t]|not\b", r"unsafe"],
        "score_1": [r"depend", r"sometimes"],
        "wrong_indicators": [r"yes.*thread.?safe", r"is\s+thread.?safe", r"atomic"],
    },

    # T2: Medium-Hard
    "t2_q1_dict_reuse": {
        "question": "Pre-allocate global dict and clear() vs create new dict each time?",
        "correct_answer": "B) create new",
        "score_3": [r"(B|new|create).*\b(better|prefer|recommend)", r"clear\(\).*not.*free", r"GC.*efficient"],
        "score_2": [r"\bB\b|new dict|create new"],
        "score_1": [r"garbage.?collect", r"memory"],
        "wrong_indicators": [r"\bA\b.*better", r"pre.?allocat.*better", r"clear.*better"],
    },
    "t2_q2_cache_bug": {
        "question": "Find the bug in the cache (TOCTOU race condition)",
        "correct_answer": "race condition - check outside lock",
        "score_3": [r"race\s*condition", r"TOCTOU|time.?of.?check", r"check.*outside.*lock|lock.*after.*check"],
        "score_2": [r"race|concurrent", r"double.?check"],
        "score_1": [r"lock|thread|parallel"],
        "wrong_indicators": [],
    },
    "t2_q3_api_design": {
        "question": "Return style for library function that might fail?",
        "correct_answer": "C) raise Exception",
        "score_3": [r"\bC\b.*\b(best|prefer|pythonic)", r"raise.*exception.*pythonic", r"EAFP"],
        "score_2": [r"\bC\b|raise.*exception|exception"],
        "score_1": [r"explicit", r"error.*handl"],
        "wrong_indicators": [r"\bA\b.*best", r"\bB\b.*best", r"tuple.*better", r"dict.*better"],
    },

    # T3: Very Hard
    "t3_q1_dependency": {
        "question": "Minimum sequential startup phases for service dependencies",
        "correct_answer": "4 phases",
        "score_3": [r"\b4\b.*phase", r"four.*phase", r"phase.*\b4\b"],
        "score_2": [r"\b4\b|\bfour\b"],
        "score_1": [r"\b5\b|\bfive\b", r"phase"],  # 5 is close but not optimal
        "wrong_indicators": [r"\b[123]\b.*phase", r"\b[6-9]\b.*phase"],
    },
    "t3_q2_vector_clock": {
        "question": "Vector clock merge calculations",
        "correct_answer": "P2=[2,3,1], P3=[2,3,3]",
        "score_3": [r"\[2,\s*3,\s*1\].*\[2,\s*3,\s*3\]", r"2,3,1.*2,3,3"],
        "score_2": [r"\[2,\s*3,\s*1\]|\[2,\s*3,\s*3\]"],  # Got one right
        "score_1": [r"max|element.?wise|increment"],
        "wrong_indicators": [],
    },
    "t3_q3_type_system": {
        "question": "TypeVar issue with first_or_default([], None)",
        "correct_answer": "T gets bound to None type",
        "score_3": [r"T.*bound.*None|None.*type|TypeVar.*None", r"infer.*None"],
        "score_2": [r"None|type.*issue|TypeVar"],
        "score_1": [r"type|generic"],
        "wrong_indicators": [r"no.*issue|works.*fine|correct.*signature"],
    },
    "t3_q4_probability": {
        "question": "Expected MEDIAN latency over many requests",
        "correct_answer": "50ms",
        "score_3": [r"\b50\s*ms|\b50\b.*median|median.*\b50\b"],
        "score_2": [r"\b50\b"],
        "score_1": [r"median", r"middle"],
        "wrong_indicators": [r"\b53\.33|53\.3\b", r"mean|average|expected.*value"],  # Common mistake: calculating mean
    },
}


def _extract_answer(content: str) -> str:
    """Extract the answer portion from model output."""
    # Try to find content after </think> tag
    think_match = re.search(r'</think>\s*(.*?)(?:EOF by user|common_perf_print|llama_perf|$)',
                            content, re.DOTALL | re.IGNORECASE)
    if think_match:
        answer = think_match.group(1).strip()
        if answer:
            return answer

    # Fallback: find assistant response
    assistant_match = re.search(r'assistant\s*\n(.*?)(?:EOF by user|common_perf_print|llama_perf|$)',
                                content, re.DOTALL)
    if assistant_match:
        return assistant_match.group(1).strip()

    # Last resort: take last meaningful chunk before timing info
    lines = content.split('\n')
    answer_lines = []
    for line in reversed(lines):
        if 'common_perf_print' in line or 'llama_perf' in line or 'EOF by user' in line:
            continue
        if line.strip():
            answer_lines.insert(0, line)
            if len(answer_lines) > 20:
                break

    return '\n'.join(answer_lines).strip()


def _extract_speed(content: str) -> Optional[float]:
    """Extract tokens per second from output."""
    match = re.search(r'(\d+\.?\d*)\s*tokens per second', content)
    if match:
        return float(match.group(1))
    return None


def _extract_acceptance(content: str) -> Optional[float]:
    """Extract acceptance rate for speculative decoding."""
    match = re.search(r'accept\s*=\s*(\d+\.?\d*)%', content)
    if match:
        return float(match.group(1))
    return None


def score_answer(test_name: str, answer: str) -> dict:
    """Score an answer against criteria."""
    if test_name not in CRITERIA:
        return {"score": -1, "reason": "Unknown test", "matched": []}

    criteria = CRITERIA[test_name]
    answer_lower = answer.lower()

    # Check for wrong indicators first
    for pattern in criteria.get("wrong_indicators", []):
        if re.search(pattern, answer_lower):
            return {
                "score": 0,
                "reason": f"Wrong indicator matched: {pattern}",
                "matched": [pattern],
                "correct_answer": criteria["correct_answer"]
            }

    # Check for score 3 (best)
    matched_3 = [p for p in criteria.get("score_3", []) if re.search(p, answer_lower)]
    if len(matched_3) >= 2:
        return {
            "score": 3,
            "reason": "Correct with strong reasoning",
            "matched": matched_3,
            "correct_answer": criteria["correct_answer"]
        }

    # Check for score 2
    matched_2 = [p for p in criteria.get("score_2", []) if re.search(p, answer_lower)]
    if matched_2 or len(matched_3) == 1:
        return {
            "score": 2,
            "reason": "Correct answer",
            "matched": matched_3 + matched_2,
            "correct_answer": criteria["correct_answer"]
        }

    # Check for score 1
    matched_1 = [p for p in criteria.get("score_1", []) if re.search(p, answer_lower)]
    if matched_1:
        return {
            "score": 1,
            "reason": "Partially correct",
            "matched": matched_1,
            "correct_answer": criteria["correct_answer"]
        }

    return {
        "score": 0,
        "reason": "No correct indicators found",
        "matched": [],
        "correct_answer": criteria["correct_answer"]
    }


def parse_filename(filename: str) -> Optional[dict]:
    """Parse model name, config, and test from filename."""
    # Format: ModelName_config_testname.txt
    # Example: Qwen3-4B-Thinking-Q8_baseline_t1_q1_algorithm.txt

    basename = os.path.splitext(filename)[0]

    # Find test name (t1_q1_xxx, t2_q1_xxx, etc.)
    test_match = re.search(r'(t[123]_q[0-9]_\w+)$', basename)
    if not test_match:
        return None

    test_name = test_match.group(1)
    prefix = basename[:test_match.start()].rstrip('_')

    # Find config (baseline, spec_k8, moe4, etc.)
    config_match = re.search(r'_(baseline|spec_k\d+|moe\d+|lookup)$', prefix)
    if config_match:
        config = config_match.group(1)
        model_name = prefix[:config_match.start()]
    else:
        config = "unknown"
        model_name = prefix

    return {
        "model": model_name,
        "config": config,
        "test": test_name
    }


def score_file(filepath: Path) -> Optional[dict]:
    """Score a single output file."""
    parsed = parse_filename(filepath.name)
    if not parsed:
        return None

    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return None

    answer = _extract_answer(content)
    speed = _extract_speed(content)
    acceptance = _extract_acceptance(content)

    score_result = score_answer(parsed["test"], answer)

    return {
        "timestamp": datetime.now().isoformat(),
        "file": filepath.name,
        "model": parsed["model"],
        "config": parsed["config"],
        "test": parsed["test"],
        "score": score_result["score"],
        "max_score": 3,
        "reason": score_result["reason"],
        "correct_answer": score_result.get("correct_answer", ""),
        "matched_patterns": score_result.get("matched", []),
        "speed_tps": speed,
        "acceptance_pct": acceptance,
        "answer_preview": answer[:500] if answer else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Score benchmark outputs")
    parser.add_argument("--input-dir", default="/mnt/raid0/llm/tmp/thinking_rubric_results",
                       help="Directory containing output files")
    parser.add_argument("--output", default="/mnt/raid0/llm/claude/benchmarks/results/scores.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary statistics")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Score all files
    results = []
    files = sorted(input_dir.glob("*.txt"))

    print(f"Scoring {len(files)} files from {input_dir}")

    for filepath in files:
        result = score_file(filepath)
        if result:
            results.append(result)

    # Write results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Wrote {len(results)} scores to {output_file}")

    # Print summary
    # Print summary
    print("\n" + "="*70)
    print("SCORING SUMMARY")
    print("="*70)

    # Group by model
    models = {}
    for r in results:
        key = f"{r['model']}_{r['config']}"
        if key not in models:
            models[key] = {"scores": [], "speeds": []}
        models[key]["scores"].append(r["score"])
        if r["speed_tps"]:
            models[key]["speeds"].append(r["speed_tps"])

    print(f"\n{'Model + Config':<45} {'Avg Score':>10} {'Total':>8} {'Avg TPS':>10}")
    print("-"*75)

    for key in sorted(models.keys()):
        data = models[key]
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        total = sum(data["scores"])
        max_total = len(data["scores"]) * 3
        avg_speed = sum(data["speeds"]) / len(data["speeds"]) if data["speeds"] else 0

        print(f"{key:<45} {avg_score:>10.2f} {total:>4}/{max_total:<3} {avg_speed:>10.1f}")

    # Per-question breakdown
    print("\n" + "-"*70)
    print("Per-Question Scores (all models):")
    print("-"*70)

    questions = {}
    for r in results:
        if r["test"] not in questions:
            questions[r["test"]] = []
        questions[r["test"]].append(r["score"])

    for test in sorted(questions.keys()):
        scores = questions[test]
        avg = sum(scores) / len(scores) if scores else 0
        dist = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in scores:
            dist[s] = dist.get(s, 0) + 1
        print(f"  {test:<25} avg={avg:.2f}  dist=[0:{dist[0]}, 1:{dist[1]}, 2:{dist[2]}, 3:{dist[3]}]")

    return 0


if __name__ == "__main__":
    exit(main())
