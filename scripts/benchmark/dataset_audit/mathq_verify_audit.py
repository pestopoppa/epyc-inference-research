#!/usr/bin/env python3
"""MathQ-Verify dataset audit (NIB2-03, EV-0).

Applies stages 1-3 of the MathQ-Verify pipeline (intake-379,
arxiv:2505.13903) to math-suite questions in the EPYC question pool.
Stage 5 (completeness) is intentionally skipped per the paper's
ablation insight (hurts F1 by +0.57pp). Stage 4 (consistency between
atomic assumptions and conclusions) is deferred to a follow-up because
it requires LLM-based atomic decomposition — out of scope for a
non-inference audit.

Stages implemented:
  Stage 1 InstValid   — regex checks: missing punctuation, malformed LaTeX,
                        unclosed delimiters, empty prompts
  Stage 2 Clean       — strip boilerplate, normalize whitespace/quotes,
                        unify ``$...$`` vs ``$$...$$``, report diffs
  Stage 3 Parse       — best-effort LaTeX parse via sympy if antlr4 present;
                        otherwise regex-based LaTeX sanity checks

Outputs:
  - ``question_pool_math_flagged.jsonl`` — flagged questions with reason codes
  - audit report (markdown) at ``--report`` path (default: stdout)

Incremental write (per ``feedback_incremental_persistence``): flagged
questions are streamed to the output JSONL as they are found, not
buffered in memory.

Usage:
    python3 mathq_verify_audit.py \\
        --input /mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl \\
        --output /mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool_math_flagged.jsonl \\
        --report /workspace/progress/2026-04/mathq-verify-audit-2026-04-21.md \\
        --suites aime,math,olympiadbench
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Default math-suite set
DEFAULT_MATH_SUITES = {"aime", "math", "olympiadbench", "physreason"}

# Stage 1 regexes
_EMPTY_OR_TRIVIAL = re.compile(r"^\s*$")
_UNCLOSED_DOLLAR = re.compile(r"(?<!\\)\$[^$]*$", re.MULTILINE)  # lone $ at EOL
_UNCLOSED_BRACE = re.compile(r"\\\w+\{[^}]*$", re.MULTILINE)       # \foo{... no close
_UNBALANCED_DELIM = re.compile(r"\\(left|right)(?=[^\w])")
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")

# Stage 2 normalization
_MULTI_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_FANCY_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_DOUBLE_DOLLAR = re.compile(r"\$\$([^$]+?)\$\$")


def load_question_pool(path: Path) -> list[dict[str, Any]]:
    """Load the JSONL pool, skipping the metadata line."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("__pool_metadata__"):
                continue
            out.append(rec)
    return out


def stage1_inst_valid(prompt: str) -> list[str]:
    """Return list of Stage 1 reason codes triggered by this prompt."""
    codes = []
    if _EMPTY_OR_TRIVIAL.match(prompt or ""):
        codes.append("S1_empty_prompt")
        return codes  # no point checking further
    # Balanced $-delimiters: count $ (non-escaped). Odd count = unclosed.
    non_escaped_dollars = len(re.findall(r"(?<!\\)\$", prompt))
    if non_escaped_dollars % 2 != 0:
        codes.append("S1_unbalanced_dollar")
    # Balanced braces inside \command{...}
    open_braces = prompt.count("{")
    close_braces = prompt.count("}")
    if open_braces != close_braces:
        codes.append("S1_unbalanced_braces")
    # \left / \right pairing
    lefts = len(re.findall(r"\\left[\(\[\{]", prompt))
    rights = len(re.findall(r"\\right[\)\]\}]", prompt))
    if lefts != rights:
        codes.append("S1_unbalanced_left_right")
    # No ending punctuation (question mark, period, or explicit answer request)
    tail = (prompt[-40:] if len(prompt) > 40 else prompt).rstrip()
    if tail and not re.search(r"[\.\?\!\:](?:\s*\*+)?\s*$", tail) and \
       not re.search(r"(answer|find|compute|evaluate|determine|prove)\b", tail, re.IGNORECASE):
        codes.append("S1_missing_terminator")
    # Unknown LaTeX commands (very permissive — only catches obvious typos)
    # Skip this check for now — too noisy without a whitelist.
    return codes


def stage2_clean(prompt: str) -> tuple[str, list[str]]:
    """Return (cleaned_prompt, list_of_applied_transformations)."""
    original = prompt
    transforms: list[str] = []
    # Normalize fancy quotes
    new = prompt.translate(_FANCY_QUOTES)
    if new != prompt:
        transforms.append("S2_normalize_quotes")
        prompt = new
    # Collapse multi-whitespace
    new = _MULTI_WS.sub(" ", prompt)
    if new != prompt:
        transforms.append("S2_collapse_whitespace")
        prompt = new
    # Collapse 3+ newlines
    new = _MULTI_NL.sub("\n\n", prompt)
    if new != prompt:
        transforms.append("S2_collapse_newlines")
        prompt = new
    # Unify $$...$$ → $...$ for inline (keep display-math semantics via later pass)
    # Only flag, don't rewrite, since $$ has semantic meaning in LaTeX display mode.
    if _DOUBLE_DOLLAR.search(prompt):
        transforms.append("S2_display_math_present")  # informational, not an error
    # Strip trailing whitespace
    new = prompt.rstrip()
    if new != prompt:
        transforms.append("S2_strip_trailing_ws")
        prompt = new
    return prompt, transforms


def _try_sympy_parse(prompt: str) -> tuple[bool, str | None]:
    """Attempt to parse inline LaTeX chunks via sympy. Returns (ok, error).

    Gated on a working antlr4 install — without it, parse_latex raises on
    nearly every non-trivial LaTeX expression, producing false positives.
    Returns ``(False, reason)`` when the toolchain is incomplete; caller
    should skip the Stage 3 parse check in that case.
    """
    try:
        import antlr4  # noqa: F401  # ErrorListener import chain
        from antlr4.error.ErrorListener import ErrorListener  # noqa: F401
        from sympy.parsing.latex import parse_latex  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return False, f"sympy_parse_unavailable: {type(exc).__name__}"
    chunks = re.findall(r"(?<!\\)\$([^$]{1,200})\$", prompt)
    parse_errors = []
    for chunk in chunks[:3]:
        if len(chunk.strip()) < 2:
            continue
        try:
            parse_latex(chunk)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(str(exc)[:80])
    if parse_errors:
        return True, "; ".join(parse_errors[:2])
    return True, None


def stage3_parse(prompt: str) -> list[str]:
    """Stage 3 parse check. Best-effort — tolerant of missing deps."""
    codes = []
    # Regex-based sanity: any `\command` without matching shape
    # Lots of false positives possible; only flag commands that clearly take args but have none.
    for cmd in ("frac", "sqrt", "sum", "int", "prod"):
        pattern = re.compile(rf"\\{cmd}(?![a-zA-Z])")
        for m in pattern.finditer(prompt):
            after = prompt[m.end():m.end() + 4]
            if not after.lstrip().startswith(("{", "(", "[", "_")):
                codes.append(f"S3_malformed_{cmd}")
                break
    ok, err = _try_sympy_parse(prompt)
    if ok and err:
        codes.append("S3_latex_parse_error")
    return codes


def audit_pool(
    pool: list[dict[str, Any]],
    suites: set[str],
    output: Path,
) -> dict[str, Any]:
    """Run stages 1-3 on each question in selected suites, streaming flagged to output."""
    stats = {
        "total_scanned": 0,
        "total_flagged": 0,
        "stage1_codes": Counter(),
        "stage2_codes": Counter(),
        "stage3_codes": Counter(),
        "suites_scanned": Counter(),
        "suite_flag_rate": {},
    }
    suite_total: Counter[str] = Counter()
    suite_flagged: Counter[str] = Counter()
    # Incremental JSONL write per feedback_incremental_persistence
    with open(output, "w") as out_f:
        for rec in pool:
            suite = rec.get("suite", "unknown")
            if suite not in suites:
                continue
            prompt = rec.get("prompt", "")
            stats["total_scanned"] += 1
            stats["suites_scanned"][suite] += 1
            suite_total[suite] += 1

            s1 = stage1_inst_valid(prompt)
            _, s2_transforms = stage2_clean(prompt)
            s3 = stage3_parse(prompt)

            # Stage-2 transformations are informational; they don't flag a question.
            # Stages 1 and 3 produce flag codes.
            flag_codes = s1 + s3

            for c in s1:
                stats["stage1_codes"][c] += 1
            for c in s2_transforms:
                stats["stage2_codes"][c] += 1
            for c in s3:
                stats["stage3_codes"][c] += 1

            if flag_codes:
                stats["total_flagged"] += 1
                suite_flagged[suite] += 1
                flagged_rec = {
                    "id": rec.get("id"),
                    "suite": suite,
                    "reason_codes": flag_codes,
                    "stage2_transforms": s2_transforms,
                    "prompt_preview": (prompt[:200] + "…") if len(prompt) > 200 else prompt,
                }
                out_f.write(json.dumps(flagged_rec) + "\n")
                out_f.flush()

    for suite, total in suite_total.items():
        flagged = suite_flagged.get(suite, 0)
        stats["suite_flag_rate"][suite] = {
            "total": total,
            "flagged": flagged,
            "rate_pct": round(100.0 * flagged / total, 2) if total else 0.0,
        }
    return stats


def write_report(stats: dict[str, Any], report_path: Path | None) -> str:
    lines = []
    lines.append("# MathQ-Verify Audit Report")
    lines.append("")
    lines.append(f"**Date**: 2026-04-21  ")
    lines.append(f"**Script**: `scripts/benchmark/dataset_audit/mathq_verify_audit.py` (NIB2-03)  ")
    lines.append(f"**Source**: intake-379 MathQ-Verify (arxiv:2505.13903), stages 1-3 only  ")
    lines.append(f"**Total scanned**: {stats['total_scanned']}  ")
    lines.append(f"**Total flagged**: {stats['total_flagged']} ({100.0 * stats['total_flagged'] / max(stats['total_scanned'], 1):.2f}%)  ")
    lines.append("")
    lines.append("## Per-suite flag rate")
    lines.append("")
    lines.append("| Suite | Total | Flagged | Rate |")
    lines.append("|-------|-------|---------|------|")
    for suite, info in sorted(stats["suite_flag_rate"].items()):
        lines.append(f"| {suite} | {info['total']} | {info['flagged']} | {info['rate_pct']:.2f}% |")
    lines.append("")
    lines.append("## Stage 1 (InstValid) reason-code distribution")
    lines.append("")
    if stats["stage1_codes"]:
        for code, count in stats["stage1_codes"].most_common():
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("_No stage 1 violations._")
    lines.append("")
    lines.append("## Stage 2 (Clean) applied transformations")
    lines.append("")
    if stats["stage2_codes"]:
        for code, count in stats["stage2_codes"].most_common():
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("_No stage 2 transformations applied (pool already clean)._")
    lines.append("")
    lines.append("## Stage 3 (Parse) reason-code distribution")
    lines.append("")
    if stats["stage3_codes"]:
        for code, count in stats["stage3_codes"].most_common():
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("_No stage 3 parse issues detected (note: may reflect missing antlr4/math-verify dependency)._")
    lines.append("")
    lines.append("## Out of scope")
    lines.append("")
    lines.append("- **Stage 4 (Consistent)** — requires LLM-based atomic decomposition (inference-gated); deferred to a follow-up work item.")
    lines.append("- **Stage 5 (Complete)** — skipped per paper ablation insight (hurts F1 by +0.57pp, introduces false positives).")
    lines.append("")
    body = "\n".join(lines)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(body + "\n")
    return body


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input", type=Path,
        default=Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl"),
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool_math_flagged.jsonl"),
    )
    p.add_argument("--report", type=Path, default=None, help="Write audit report to path (markdown).")
    p.add_argument(
        "--suites", type=str, default=",".join(sorted(DEFAULT_MATH_SUITES)),
        help="Comma-separated suite names to audit.",
    )
    args = p.parse_args()

    suites = {s.strip() for s in args.suites.split(",") if s.strip()}
    pool = load_question_pool(args.input)
    stats = audit_pool(pool, suites, args.output)
    body = write_report(stats, args.report)
    if not args.report:
        print(body)
    else:
        print(f"Scanned {stats['total_scanned']} questions across {len(stats['suites_scanned'])} suites; flagged {stats['total_flagged']}.")
        print(f"Flagged records: {args.output}")
        print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
