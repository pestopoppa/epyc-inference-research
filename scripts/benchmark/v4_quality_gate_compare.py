#!/usr/bin/env python3
"""V4 quality-gate comparator: diff EPYC logprobs vs reference logprobs.

Inputs are two JSON files produced by v4_quality_gate_runner.py (or an
equivalent on the reference engine — antirez fork on Mac or ds4). Both must
have the same prompt IDs and the same n_tokens_requested.

Output is a Markdown report + an exit code (0 PASS, 1 FAIL).

Gate criteria (per handoffs/active/deepseek-v4-flash-cpu-port.md §Merge Gates):
  - Per-prompt MAD ≤ 0.05 nats
  - ≥ 18 of 20 prompts pass per-prompt tolerance
  - ≥ 15 of 20 prompts emit same first token under greedy
  - No assert/segfault/NaN (manifested as missing logprobs / empty tokens)

Usage:
    v4_quality_gate_compare.py --epyc PATH --reference PATH --output PATH [...]

Override the gates with --max-mad / --min-prompt-pass / --min-token1-pass if
you want to tighten or loosen for a side experiment (not for the canonical
merge gate; the documented thresholds are 0.05 / 18 / 15).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core MAD computation
# ---------------------------------------------------------------------------


def compute_mad(epyc_lp: list[float], ref_lp: list[float]) -> float | None:
    """Mean absolute log-prob difference over matched-length token sequences.

    Returns None if either side has no valid logprobs (assert/NaN proxy).
    """
    if not epyc_lp or not ref_lp:
        return None
    n = min(len(epyc_lp), len(ref_lp))
    if n == 0:
        return None
    diffs: list[float] = []
    for i in range(n):
        a, b = epyc_lp[i], ref_lp[i]
        if a is None or b is None:
            continue
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            return None  # NaN/Inf is a hard failure indicator
        diffs.append(abs(a - b))
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def token1_match(epyc_tokens: list[str], ref_tokens: list[str]) -> bool:
    """Did both engines emit the same first token under greedy decoding?"""
    if not epyc_tokens or not ref_tokens:
        return False
    return epyc_tokens[0] == ref_tokens[0]


def has_runtime_failure(prompt_result: dict, min_tokens: int = 1) -> bool:
    """Detect assert/segfault/NaN proxies + short-output runs.

    A prompt result with fewer than `min_tokens` captured tokens is treated as
    a runtime failure. The pre-fix MAD-over-min(len) logic falsely PASSED runs
    where the server emitted only 1 token per prompt; with min_tokens set from
    the runner's n_tokens_requested, short runs now fail the gate.
    """
    if "error" in prompt_result:
        return True
    if prompt_result.get("token_count", 0) < min_tokens:
        return True
    # NaN/Inf in logprobs are caught by compute_mad returning None
    return False


# ---------------------------------------------------------------------------
# Comparison loop + report
# ---------------------------------------------------------------------------


def compare(epyc: dict, reference: dict, max_mad: float,
            expected_n_prompts: int | None = None,
            min_tokens_per_prompt: int | None = None) -> tuple[list[dict], dict]:
    """Per-prompt comparison loop. Returns (rows, summary).

    Args:
        epyc: EPYC runner JSON (must have prompts[] and n_tokens_requested)
        reference: reference runner JSON (same shape)
        max_mad: per-prompt MAD threshold (nats)
        expected_n_prompts: required number of prompts on each side (default 20
            per §Merge Gates). A run with fewer prompts gets the missing slots
            flagged as runtime_failure rows.
        min_tokens_per_prompt: minimum token_count per prompt to count as a
            non-failure. If None, derived as min(epyc.n_tokens_requested,
            reference.n_tokens_requested). Any prompt with fewer tokens is
            flagged runtime_failure (this prevents 1-token-per-prompt runs
            from falsely passing the gate).
    """
    epyc_by_id = {p["id"]: p for p in epyc["prompts"]}
    ref_by_id = {p["id"]: p for p in reference["prompts"]}
    all_ids = sorted(set(epyc_by_id) | set(ref_by_id))

    # Derive expected per-prompt token count from runner metadata if not given.
    if min_tokens_per_prompt is None:
        epyc_n = epyc.get("n_tokens_requested")
        ref_n = reference.get("n_tokens_requested")
        if isinstance(epyc_n, int) and isinstance(ref_n, int):
            min_tokens_per_prompt = min(epyc_n, ref_n)
        elif isinstance(epyc_n, int):
            min_tokens_per_prompt = epyc_n
        elif isinstance(ref_n, int):
            min_tokens_per_prompt = ref_n
        else:
            min_tokens_per_prompt = 1  # last resort; existing-pass behavior

    rows: list[dict] = []
    n_pass_mad = 0
    n_token1_match = 0
    n_runtime_fail = 0

    # If a strict expected_n_prompts is set and either side falls short, add
    # synthetic runtime-failure rows for the missing slots. The CLI layer
    # defaults this to 20 (per §Merge Gates); library callers can pass None to
    # disable the strict prompt-count check.
    if expected_n_prompts is not None and (
        len(epyc_by_id) < expected_n_prompts or len(ref_by_id) < expected_n_prompts
    ):
        missing_count = expected_n_prompts - len(all_ids)
        for i in range(max(0, missing_count)):
            rows.append({
                "id": f"<missing-{i+1}>",
                "category": "",
                "error": (
                    f"truncated run: expected {expected_n_prompts} prompts, "
                    f"got {len(all_ids)} (epyc={len(epyc_by_id)}, "
                    f"ref={len(ref_by_id)})"
                ),
                "mad": None,
                "mad_pass": False,
                "token1_match": False,
                "runtime_failure": True,
            })
            n_runtime_fail += 1

    for pid in all_ids:
        e = epyc_by_id.get(pid)
        rf = ref_by_id.get(pid)
        row: dict = {"id": pid}
        if e is None or rf is None:
            row["error"] = f"missing on {'epyc' if e is None else 'reference'} side"
            row["mad"] = None
            row["mad_pass"] = False
            row["token1_match"] = False
            row["runtime_failure"] = True
            n_runtime_fail += 1
            rows.append(row)
            continue
        if (has_runtime_failure(e, min_tokens=min_tokens_per_prompt)
                or has_runtime_failure(rf, min_tokens=min_tokens_per_prompt)):
            row["category"] = e.get("category", "?")
            e_short = e.get("token_count", 0) < min_tokens_per_prompt
            r_short = rf.get("token_count", 0) < min_tokens_per_prompt
            if e_short or r_short:
                row["error"] = (
                    f"truncated tokens: epyc={e.get('token_count',0)} "
                    f"ref={rf.get('token_count',0)} "
                    f"need ≥ {min_tokens_per_prompt}"
                )
            else:
                row["error"] = "runtime failure (empty tokens / explicit error)"
            row["mad"] = None
            row["mad_pass"] = False
            row["token1_match"] = False
            row["runtime_failure"] = True
            n_runtime_fail += 1
            rows.append(row)
            continue
        mad = compute_mad(e["logprobs"], rf["logprobs"])
        t1 = token1_match(e["tokens_text"], rf["tokens_text"])
        row["category"] = e.get("category", "?")
        row["epyc_tokens_n"] = len(e["logprobs"])
        row["ref_tokens_n"] = len(rf["logprobs"])
        row["mad"] = mad
        row["mad_pass"] = mad is not None and mad <= max_mad
        row["token1_match"] = t1
        row["runtime_failure"] = mad is None  # MAD=None indicates NaN/empty
        if row["runtime_failure"]:
            n_runtime_fail += 1
        if row["mad_pass"]:
            n_pass_mad += 1
        if t1:
            n_token1_match += 1
        rows.append(row)
    summary = {
        "n_prompts": len(all_ids),
        "n_pass_mad": n_pass_mad,
        "n_token1_match": n_token1_match,
        "n_runtime_fail": n_runtime_fail,
        "max_mad_threshold": max_mad,
        "expected_n_prompts": expected_n_prompts,
        "min_tokens_per_prompt": min_tokens_per_prompt,
    }
    return rows, summary


def verdict(summary: dict, min_prompt_pass: int, min_token1_pass: int) -> tuple[bool, str]:
    """Apply the §Merge Gates rule and return (pass, explanation)."""
    if summary["n_runtime_fail"] > 0:
        return False, (
            f"FAIL: {summary['n_runtime_fail']} prompt(s) had runtime failure "
            f"(assert/segfault/NaN/empty). Any runtime failure is automatic "
            f"FAIL regardless of other prompt scores."
        )
    if summary["n_pass_mad"] < min_prompt_pass:
        return False, (
            f"FAIL: only {summary['n_pass_mad']}/{summary['n_prompts']} prompts "
            f"passed per-prompt MAD ≤ {summary['max_mad_threshold']} nats; "
            f"need ≥ {min_prompt_pass}."
        )
    if summary["n_token1_match"] < min_token1_pass:
        return False, (
            f"FAIL: only {summary['n_token1_match']}/{summary['n_prompts']} prompts "
            f"had token-1 exact match; need ≥ {min_token1_pass}."
        )
    return True, (
        f"PASS: {summary['n_pass_mad']}/{summary['n_prompts']} MAD-pass "
        f"(≥{min_prompt_pass}) + {summary['n_token1_match']}/{summary['n_prompts']} "
        f"token-1 (≥{min_token1_pass}) + zero runtime failures."
    )


def render_markdown(rows: list[dict], summary: dict, verdict_text: str,
                    epyc_meta: dict, ref_meta: dict, max_mad: float,
                    min_prompt_pass: int, min_token1_pass: int) -> str:
    lines: list[str] = []
    lines.append("# DeepSeek-V4 Quality-Gate Report")
    lines.append("")
    lines.append(f"**Verdict**: {verdict_text}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- EPYC side: `{epyc_meta.get('model_path','?')}` via `{epyc_meta.get('binary','?')}`")
    lines.append(f"- Reference: `{ref_meta.get('model_path','?')}` via `{ref_meta.get('binary','?')}`")
    lines.append(f"- Tokens per prompt: {epyc_meta.get('n_tokens_requested','?')} (EPYC) "
                 f"/ {ref_meta.get('n_tokens_requested','?')} (ref)")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Pass? |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| Per-prompt MAD ≤ x nats | {max_mad} | per-row | — |")
    lines.append(f"| Prompts passing MAD | ≥ {min_prompt_pass} | {summary['n_pass_mad']} | "
                 f"{'✓' if summary['n_pass_mad'] >= min_prompt_pass else '✗'} |")
    lines.append(f"| Token-1 exact match | ≥ {min_token1_pass} | {summary['n_token1_match']} | "
                 f"{'✓' if summary['n_token1_match'] >= min_token1_pass else '✗'} |")
    lines.append(f"| Runtime failures | 0 | {summary['n_runtime_fail']} | "
                 f"{'✓' if summary['n_runtime_fail'] == 0 else '✗'} |")
    lines.append("")
    lines.append("## Per-prompt detail")
    lines.append("")
    lines.append("| ID | Category | MAD (nats) | MAD-pass | Token-1 match | Notes |")
    lines.append("|---|---|---:|:---:|:---:|---|")
    for row in rows:
        mad_disp = f"{row['mad']:.4f}" if row.get("mad") is not None else "—"
        mad_pass = "✓" if row["mad_pass"] else "✗"
        t1 = "✓" if row["token1_match"] else "✗"
        note = row.get("error", "")
        lines.append(f"| {row['id']} | {row.get('category','')} | {mad_disp} | "
                     f"{mad_pass} | {t1} | {note} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="V4 quality-gate comparator")
    p.add_argument("--epyc", required=True, type=Path,
                   help="EPYC-side JSON from v4_quality_gate_runner.py")
    p.add_argument("--reference", required=True, type=Path,
                   help="Reference JSON (Mac fork / ds4) — same shape")
    p.add_argument("--output", required=True, type=Path,
                   help="Output markdown report path")
    p.add_argument("--max-mad", type=float, default=0.05,
                   help="Per-prompt MAD threshold in nats (default: 0.05)")
    p.add_argument("--min-prompt-pass", type=int, default=18,
                   help="Min prompts passing MAD (default: 18 of 20)")
    p.add_argument("--min-token1-pass", type=int, default=15,
                   help="Min prompts with token-1 match (default: 15 of 20)")
    p.add_argument("--expected-n-prompts", type=int, default=20,
                   help="Required number of prompts on each side (default: 20)")
    p.add_argument("--min-tokens-per-prompt", type=int, default=None,
                   help="Required token_count per prompt. Default: min of EPYC "
                        "and reference n_tokens_requested from runner JSON.")
    args = p.parse_args()

    with args.epyc.open() as f:
        epyc = json.load(f)
    with args.reference.open() as f:
        reference = json.load(f)

    rows, summary = compare(epyc, reference, args.max_mad,
                            expected_n_prompts=args.expected_n_prompts,
                            min_tokens_per_prompt=args.min_tokens_per_prompt)
    passed, verdict_text = verdict(summary, args.min_prompt_pass, args.min_token1_pass)
    report = render_markdown(rows, summary, verdict_text, epyc, reference,
                             args.max_mad, args.min_prompt_pass, args.min_token1_pass)
    args.output.write_text(report)
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
