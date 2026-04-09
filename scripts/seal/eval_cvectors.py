#!/usr/bin/env python3
"""Evaluate SEAL control vectors for reasoning conciseness.

Tests a control vector at a given scaling factor (or baseline) against a set of
math problems, measuring token count, correctness, and brevity ratio.

Because llama.cpp loads control vectors at server startup via
--control-vector-scaled <path>,<factor>, each scaling factor requires a server
restart. This script tests ONE configuration per invocation. Use --run-all mode
for an interactive workflow that prompts between scaling factors.

Usage:
    # Baseline only (no control vector)
    python eval_cvectors.py --model-port 8080 --baseline-only

    # Single scaling factor (server must already be running with that vector/factor)
    python eval_cvectors.py --model-port 8080 --cvector /tmp/seal-concise.gguf --scaling 0.5

    # Interactive multi-factor workflow
    python eval_cvectors.py --model-port 8080 --cvector /tmp/seal-concise.gguf --scaling 0.3,0.5,0.7 --run-all

Requires a running llama-server instance.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Evaluation problem set — 20 problems with known answers
# ---------------------------------------------------------------------------

EVAL_PROBLEMS = [
    # Algebra
    {
        "problem": "Solve for x: 4x - 7 = 21.",
        "expected_answer": "7",
        "category": "algebra",
    },
    {
        "problem": "Solve for x: x^2 - 7x + 10 = 0. Give the larger root.",
        "expected_answer": "5",
        "category": "algebra",
    },
    {
        "problem": "What is the sum of the first 20 positive integers?",
        "expected_answer": "210",
        "category": "algebra",
    },
    {
        "problem": "Simplify: (x^2 - 4) / (x + 2) for x ≠ -2.",
        "expected_answer": "x - 2",
        "category": "algebra",
    },
    {
        "problem": "Solve for x: 3^(2x) = 81.",
        "expected_answer": "2",
        "category": "algebra",
    },
    # Geometry
    {
        "problem": "Find the area of a circle with radius 6. Give the exact answer in terms of pi.",
        "expected_answer": "36pi",
        "category": "geometry",
    },
    {
        "problem": "Find the hypotenuse of a right triangle with legs 9 and 12.",
        "expected_answer": "15",
        "category": "geometry",
    },
    {
        "problem": "What is the volume of a cube with side length 5?",
        "expected_answer": "125",
        "category": "geometry",
    },
    {
        "problem": "A rectangle has length 14 and width 9. What is its perimeter?",
        "expected_answer": "46",
        "category": "geometry",
    },
    {
        "problem": "Find the area of an equilateral triangle with side 10. Round to the nearest integer.",
        "expected_answer": "43",
        "category": "geometry",
    },
    # Number theory
    {
        "problem": "Find the GCD of 84 and 120.",
        "expected_answer": "12",
        "category": "number_theory",
    },
    {
        "problem": "How many trailing zeros does 25! have?",
        "expected_answer": "6",
        "category": "number_theory",
    },
    {
        "problem": "What is 17 mod 5?",
        "expected_answer": "2",
        "category": "number_theory",
    },
    {
        "problem": "How many prime numbers are between 1 and 30?",
        "expected_answer": "10",
        "category": "number_theory",
    },
    {
        "problem": "What is the LCM of 8 and 14?",
        "expected_answer": "56",
        "category": "number_theory",
    },
    # Word problems
    {
        "problem": "A train travels at 80 km/h for 3 hours. How far does it go (in km)?",
        "expected_answer": "240",
        "category": "word_problem",
    },
    {
        "problem": "A shirt costs $40. It is on sale for 25% off. What is the sale price?",
        "expected_answer": "30",
        "category": "word_problem",
    },
    {
        "problem": "How many ways can you choose 3 items from 7? (Combinations.)",
        "expected_answer": "35",
        "category": "word_problem",
    },
    {
        "problem": "The average of five numbers is 12. What is their sum?",
        "expected_answer": "60",
        "category": "word_problem",
    },
    {
        "problem": "A fair coin is flipped 3 times. What is the probability of exactly 2 heads? Express as a fraction.",
        "expected_answer": "3/8",
        "category": "word_problem",
    },
]

# ---------------------------------------------------------------------------
# HTTP client for llama-server
# ---------------------------------------------------------------------------


def query_server(
    port: int,
    prompt: str,
    n_predict: int = 2048,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> dict:
    """Send a completion request to llama-server and return the response dict.

    Returns dict with keys: content, tokens_predicted, timings, etc.
    Raises ConnectionError on failure.
    """
    url = f"http://127.0.0.1:{port}/completion"
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to llama-server on port {port}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Answer extraction and checking
# ---------------------------------------------------------------------------


def extract_answer(response_text: str) -> str:
    """Extract the final answer from a model response.

    Strategy:
    1. Look for \\boxed{...}
    2. Look for "the answer is ..."
    3. Fall back to the last number/fraction in the response
    """
    # \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response_text)
    if boxed:
        return boxed[-1].strip()

    # "the answer is X" pattern
    answer_match = re.search(
        r"(?:the answer is|answer:|final answer:?)\s*([^\n.]+)",
        response_text,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).strip().rstrip(".")

    # Last number or fraction in the text
    numbers = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", response_text)
    if numbers:
        return numbers[-1]

    return ""


def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison."""
    ans = ans.strip().lower()
    # Remove LaTeX formatting
    ans = ans.replace("\\", "").replace("$", "").replace(",", "")
    # Normalize pi representations
    ans = re.sub(r"(\d+)\s*\\?pi", r"\1pi", ans)
    ans = re.sub(r"(\d+)\s*π", r"\1pi", ans)
    return ans


def check_answer(extracted: str, expected: str) -> bool:
    """Check if the extracted answer matches the expected answer."""
    norm_ext = normalize_answer(extracted)
    norm_exp = normalize_answer(expected)

    if norm_ext == norm_exp:
        return True

    # Try numeric comparison
    try:
        # Handle fractions
        def to_float(s: str) -> float:
            if "/" in s:
                num, den = s.split("/", 1)
                return float(num) / float(den)
            return float(s)

        return abs(to_float(norm_ext) - to_float(norm_exp)) < 1e-6
    except (ValueError, ZeroDivisionError):
        pass

    # Check if expected is contained in extracted
    if norm_exp in norm_ext:
        return True

    return False


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def evaluate_problems(
    port: int,
    problems: list[dict],
    label: str,
) -> dict:
    """Run all problems against the server and return aggregate results."""
    per_problem = []
    correct = 0

    for i, prob in enumerate(problems, 1):
        prompt = (
            f"Solve this problem.\n\nProblem: {prob['problem']}\n\nAnswer:"
        )

        print(f"  [{i}/{len(problems)}] {prob['category']}: ", end="", flush=True)

        try:
            t0 = time.monotonic()
            resp = query_server(port, prompt)
            elapsed = time.monotonic() - t0
        except ConnectionError as e:
            print(f"ERROR — {e}")
            per_problem.append({
                "problem": prob["problem"],
                "category": prob["category"],
                "expected": prob["expected_answer"],
                "extracted": "",
                "correct": False,
                "tokens": 0,
                "chars": 0,
                "error": str(e),
            })
            continue

        content = resp.get("content", "")
        tokens = resp.get("tokens_predicted", len(content.split()))
        extracted = extract_answer(content)
        is_correct = check_answer(extracted, prob["expected_answer"])

        if is_correct:
            correct += 1

        status = "OK" if is_correct else "WRONG"
        print(
            f"{status} (extracted={extracted!r}, expected={prob['expected_answer']!r}, "
            f"{tokens} tok, {elapsed:.1f}s)"
        )

        per_problem.append({
            "problem": prob["problem"],
            "category": prob["category"],
            "expected": prob["expected_answer"],
            "extracted": extracted,
            "correct": is_correct,
            "tokens": tokens,
            "chars": len(content),
            "elapsed_s": round(elapsed, 2),
        })

    n = len(problems)
    accuracy = correct / n if n > 0 else 0.0
    avg_tokens = sum(p["tokens"] for p in per_problem) / n if n > 0 else 0.0
    avg_chars = sum(p["chars"] for p in per_problem) / n if n > 0 else 0.0

    return {
        "label": label,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": n,
        "avg_tokens": round(avg_tokens, 1),
        "avg_chars": round(avg_chars, 1),
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------


def print_summary_table(all_results: dict, baseline_key: str = "baseline") -> dict:
    """Print a summary table and return the summary dict."""
    baseline = all_results.get(baseline_key)
    if baseline is None:
        print("No baseline results — cannot compute reductions.")
        return {}

    header = f"{'Scaling':<10}| {'Accuracy':>8} | {'Avg Tokens':>10} | {'Token Δ':>10} | {'Verdict':>7}"
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)

    best_scaling = None
    best_reduction = 0.0
    best_accuracy_delta = 0.0

    for key, res in all_results.items():
        acc_str = f"{res['accuracy'] * 100:.1f}%"
        tok_str = f"{res['avg_tokens']:.0f}"

        if key == baseline_key:
            print(f"{'baseline':<10}| {acc_str:>8} | {tok_str:>10} | {'—':>10} | {'—':>7}")
        else:
            if baseline["avg_tokens"] > 0:
                reduction = (
                    (baseline["avg_tokens"] - res["avg_tokens"])
                    / baseline["avg_tokens"]
                    * 100
                )
            else:
                reduction = 0.0

            acc_delta_pp = (res["accuracy"] - baseline["accuracy"]) * 100
            verdict = "PASS" if acc_delta_pp >= -2.0 else "FAIL"

            print(
                f"{key:<10}| {acc_str:>8} | {tok_str:>10} | "
                f"{reduction:>+9.1f}% | {verdict:>7}"
            )

            if verdict == "PASS" and reduction > best_reduction:
                best_reduction = reduction
                best_scaling = key
                best_accuracy_delta = acc_delta_pp

    print()

    summary = {}
    if best_scaling is not None:
        summary = {
            "best_scaling": best_scaling,
            "token_reduction_pct": round(best_reduction, 1),
            "accuracy_delta_pp": round(best_accuracy_delta, 1),
        }
        print(
            f"Best passing config: {best_scaling} "
            f"({best_reduction:+.1f}% tokens, {best_accuracy_delta:+.1f}pp accuracy)"
        )
    elif baseline_key in all_results and len(all_results) == 1:
        print("Baseline only — no scaling factors to compare.")
    else:
        print("No scaling factor passed the <2pp accuracy drop threshold.")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SEAL control vectors for reasoning conciseness."
    )
    parser.add_argument(
        "--model-port",
        type=int,
        required=True,
        help="Port of running llama-server",
    )
    parser.add_argument(
        "--cvector",
        type=str,
        default=None,
        help="Path to the control vector file (omit for baseline-only)",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated scaling factors (default: 0.3,0.5,0.7)",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=20,
        help="Number of problems to evaluate (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seal_eval_results.json",
        help="Output JSON file path (default: seal_eval_results.json)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run baseline only (no control vector)",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Interactive mode: run baseline + all scaling factors, prompting for server restart between each",
    )
    args = parser.parse_args()

    problems = EVAL_PROBLEMS[: args.n_problems]
    scaling_factors = [float(s.strip()) for s in args.scaling.split(",")]

    all_results = {}
    output_path = Path(args.output)

    # ── Baseline ──────────────────────────────────────────────────────────
    if args.baseline_only or args.run_all:
        print("=" * 60)
        print("Running BASELINE (no control vector)")
        print("=" * 60)
        if args.run_all and not args.baseline_only:
            print("Ensure the server is running WITHOUT a control vector.")
            input("Press Enter to continue (or Ctrl+C to abort)...")
        all_results["baseline"] = evaluate_problems(
            args.model_port, problems, "baseline"
        )

    # ── Scaling factors ───────────────────────────────────────────────────
    if not args.baseline_only:
        if args.cvector is None:
            print(
                "Error: --cvector is required for non-baseline runs.",
                file=sys.stderr,
            )
            sys.exit(1)

        for factor in scaling_factors:
            label = f"scaled_{factor}"
            print()
            print("=" * 60)
            print(f"Evaluating scaling factor: {factor}")
            print("=" * 60)
            if args.run_all:
                print(
                    f"\nRestart the server with:\n"
                    f"  llama-server -m <model> "
                    f"--control-vector-scaled {args.cvector},{factor}\n"
                )
                input("Press Enter when the server is ready (or Ctrl+C to abort)...")
            all_results[label] = evaluate_problems(
                args.model_port, problems, label
            )

    # ── Summary ───────────────────────────────────────────────────────────
    summary = print_summary_table(all_results)

    # ── Write JSON output ─────────────────────────────────────────────────
    output_data = {
        "model_port": args.model_port,
        "cvector_path": args.cvector,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_problems": len(problems),
        "results": all_results,
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
