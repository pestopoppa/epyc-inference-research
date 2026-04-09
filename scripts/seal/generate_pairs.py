#!/usr/bin/env python3
"""Generate contrastive prompt pairs for SEAL control vector training.

Produces positive.txt (concise reasoning) and negative.txt (verbose reasoning)
for use with llama.cpp's cvector-generator tool.

Usage:
    python generate_pairs.py [--output-dir DIR] [--n-pairs N]

Examples:
    # Generate all pairs to the script directory
    python generate_pairs.py

    # Generate 50 pairs to a custom directory
    python generate_pairs.py --output-dir /tmp/seal-pairs --n-pairs 50

    # Use a custom problem file
    python generate_pairs.py --problems-file my_problems.txt --output-dir /tmp/seal-pairs
"""

import argparse
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hardcoded problem bank — diverse MATH-benchmark-style problems
# Categories: algebra, geometry, number_theory, word_problems, logic
# ---------------------------------------------------------------------------

PROBLEMS = [
    # ── Algebra (20) ──────────────────────────────────────────────────────
    "Solve for x: 3x + 7 = 22.",
    "Solve for x: 2(x - 3) + 5 = 3x - 1.",
    "Simplify: (2x^2 + 3x - 5) - (x^2 - 4x + 2).",
    "Factor completely: x^2 - 5x + 6.",
    "Solve for x: x^2 - 9x + 20 = 0.",
    "If f(x) = 2x^2 - 3x + 1, find f(4).",
    "Solve the system: 2x + y = 7, x - y = 2.",
    "Simplify: (3a^2b)(2ab^3).",
    "Solve for x: |2x - 5| = 9.",
    "Find the sum of the arithmetic series: 3 + 7 + 11 + ... + 99.",
    "Solve for x: log_2(x) + log_2(x - 2) = 3.",
    "If the roots of x^2 + bx + 12 = 0 are 3 and 4, find b.",
    "Simplify: (x^3 - 8) / (x - 2).",
    "Solve for x: 5^(x+1) = 125.",
    "Find the vertex of the parabola y = x^2 - 6x + 5.",
    "Solve: sqrt(2x + 3) = 5.",
    "Expand and simplify: (2x - 3)^3.",
    "Solve the inequality: 3x - 7 > 2x + 5.",
    "Find the 10th term of the geometric sequence 2, 6, 18, 54, ...",
    "Solve for x: (x + 2)/(x - 1) = 3.",

    # ── Geometry (15) ─────────────────────────────────────────────────────
    "Find the area of a triangle with base 12 and height 8.",
    "A circle has radius 7. Find its circumference.",
    "Find the hypotenuse of a right triangle with legs 5 and 12.",
    "What is the area of a trapezoid with parallel sides 6 and 10, and height 4?",
    "Find the volume of a cylinder with radius 3 and height 10.",
    "In triangle ABC, angle A = 40° and angle B = 75°. Find angle C.",
    "A rectangle has perimeter 30 and width 7. Find its length.",
    "Find the area of a regular hexagon with side length 6.",
    "What is the surface area of a sphere with radius 5?",
    "Two parallel lines are cut by a transversal. One angle is 65°. Find its supplementary co-interior angle.",
    "Find the diagonal of a rectangle with sides 8 and 15.",
    "A cone has radius 4 and slant height 9. Find its lateral surface area.",
    "Find the arc length of a 60° sector in a circle of radius 12.",
    "The legs of an isosceles right triangle each measure 10. Find the hypotenuse.",
    "Find the area of a circle inscribed in a square of side 8.",

    # ── Number Theory (15) ────────────────────────────────────────────────
    "Is 91 prime? Explain.",
    "Find the greatest common divisor of 48 and 180.",
    "How many positive divisors does 360 have?",
    "What is the remainder when 2^100 is divided by 7?",
    "Find the least common multiple of 12, 18, and 20.",
    "Express 0.363636... as a fraction in lowest terms.",
    "How many primes are there between 50 and 80?",
    "Find the sum of all positive divisors of 28.",
    "What is 7^4 mod 13?",
    "How many trailing zeros does 50! have?",
    "Find the last two digits of 3^200.",
    "Is 2^31 - 1 prime? (Mersenne prime check.)",
    "Find all integer solutions to 3x ≡ 5 (mod 7).",
    "What is the digital root of 987654?",
    "How many integers from 1 to 100 are coprime to 30?",

    # ── Word Problems (20) ────────────────────────────────────────────────
    "A train travels at 60 mph for 2.5 hours. How far does it go?",
    "If 8 workers can build a wall in 6 days, how long would 12 workers take?",
    "A store offers 20% off a $45 item. What is the sale price?",
    "Mix 5 liters of 30% acid with 3 liters of 50% acid. What is the resulting concentration?",
    "A fair die is rolled twice. What is the probability both rolls show the same number?",
    "John is twice as old as Mary. In 5 years, the sum of their ages will be 40. How old is Mary now?",
    "A car depreciates 15% per year. After 3 years, a $20000 car is worth how much?",
    "How many ways can 5 books be arranged on a shelf?",
    "A bag has 4 red and 6 blue balls. Two are drawn without replacement. P(both red)?",
    "A boat goes 20 km upstream in 4 hours and returns in 2 hours. Find the speed of the current.",
    "Investment of $1000 at 5% annual compound interest for 3 years yields how much?",
    "Three coins are tossed. What is the probability of at least 2 heads?",
    "A pipe fills a tank in 6 hours, another empties it in 9 hours. Both open: how long to fill?",
    "The average of 5 numbers is 18. When a 6th number is added, the average becomes 20. What is the 6th number?",
    "How many diagonals does a convex 10-gon have?",
    "A ladder 13 feet long leans against a wall. The base is 5 feet from the wall. How high up does the ladder reach?",
    "A rectangular garden is 3 meters longer than it is wide. Its area is 108 m^2. Find its dimensions.",
    "Two trains approach each other at 80 km/h and 120 km/h from 400 km apart. When do they meet?",
    "A committee of 3 is chosen from 8 people. How many ways?",
    "Water flows into a pool at 3 m^3/hr and out at 1.5 m^3/hr. The pool holds 45 m^3. How long to fill from empty?",

    # ── Logic Puzzles (10) ────────────────────────────────────────────────
    "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
    "If it rains, the ground is wet. The ground is wet. Did it rain?",
    "A says 'B always lies.' B says 'A always tells the truth.' Who is the liar?",
    "You have a 3-liter jug and a 5-liter jug. How do you measure exactly 4 liters?",
    "Three light switches in one room control three bulbs in another. You may enter the bulb room once. How do you determine which switch controls which bulb?",
    "If the day after tomorrow is two days before Thursday, what day is today?",
    "Five people finish a race. Amy beats Bob. Carol beats Dave. Bob beats Carol. Eve finishes last. Who won?",
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "What is the minimum number of moves to transfer 4 disks in the Tower of Hanoi?",
    "You have 12 coins, one counterfeit (lighter). Using a balance scale, what is the minimum weighings needed to find it?",
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

POSITIVE_TEMPLATE = (
    "Solve this problem. Be maximally concise — give only the essential "
    "steps and final answer.\n\nProblem: {problem}"
)

NEGATIVE_TEMPLATE = (
    "Solve this problem. Show every step of your reasoning in full detail, "
    "explaining your thought process thoroughly.\n\nProblem: {problem}"
)


def load_problems_from_file(path: Path) -> list[str]:
    """Read problems from a text file, one per line, skipping blanks."""
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                problems.append(line)
    return problems


def generate_pairs(
    problems: list[str],
    n_pairs: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return (positive_prompts, negative_prompts) lists paired by index."""
    rng = random.Random(seed)
    pool = list(problems)
    rng.shuffle(pool)

    if n_pairs is not None:
        pool = pool[:n_pairs]

    positives = [POSITIVE_TEMPLATE.format(problem=p) for p in pool]
    negatives = [NEGATIVE_TEMPLATE.format(problem=p) for p in pool]
    return positives, negatives


def write_file(path: Path, lines: list[str]) -> None:
    """Write one prompt per line. Prompts must not contain newlines."""
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            # Collapse any internal newlines to spaces so each prompt is one line
            f.write(line.replace("\n", " ") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate contrastive prompt pairs for SEAL control vector training."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write positive.txt and negative.txt (default: script dir)",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        help="Number of pairs to generate (default: all available problems)",
    )
    parser.add_argument(
        "--problems-file",
        type=Path,
        default=None,
        help="Optional file with custom problems, one per line",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    # Select problem source
    if args.problems_file is not None:
        if not args.problems_file.exists():
            print(f"Error: problems file not found: {args.problems_file}", file=sys.stderr)
            sys.exit(1)
        problems = load_problems_from_file(args.problems_file)
        print(f"Loaded {len(problems)} problems from {args.problems_file}")
    else:
        problems = PROBLEMS
        print(f"Using {len(problems)} built-in problems")

    if not problems:
        print("Error: no problems available.", file=sys.stderr)
        sys.exit(1)

    # Generate pairs
    positives, negatives = generate_pairs(problems, n_pairs=args.n_pairs, seed=args.seed)

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pos_path = args.output_dir / "positive.txt"
    neg_path = args.output_dir / "negative.txt"

    write_file(pos_path, positives)
    write_file(neg_path, negatives)

    print(f"Wrote {len(positives)} pairs:")
    print(f"  positive: {pos_path}")
    print(f"  negative: {neg_path}")
    print(f"  seed:     {args.seed}")


if __name__ == "__main__":
    main()
