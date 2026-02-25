#!/usr/bin/env python3
from __future__ import annotations

"""Look up a benchmark question by ID: reconstruct prompt from HuggingFace, show all saved eval answers.

Usage:
    python3 scripts/benchmark/lookup_question.py hellaswag_27747
    python3 scripts/benchmark/lookup_question.py arc_Mercury_7111178
    python3 scripts/benchmark/lookup_question.py mmlu_abstract_algebra_01498
    python3 scripts/benchmark/lookup_question.py gsm8k_00042
"""

import json
import re
import sys
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

from dataset_adapters import get_adapter


# ── ID parsing ──────────────────────────────────────────────────────────────

# Maps ID prefix → (suite_name, dataset_within_adapter)
# The dataset_within_adapter is used to disambiguate multi-source adapters
PREFIX_MAP = [
    # thinking suite
    ("hellaswag_",  "thinking",  "hellaswag"),
    ("arc_",        "thinking",  "arc"),
    # general suite
    ("mmlu_",       "general",   "mmlu"),
    # math suite
    ("gsm8k_",      "math",      "gsm8k"),
    ("math500_",    "math",      "math500"),
    # coder suite
    ("humaneval_",  "coder",     "humaneval"),
    ("mbpp_",       "coder",     "mbpp"),
    # instruction_precision suite
    ("ifeval_",     "instruction_precision", "ifeval"),
    # hard benchmarks
    ("gpqa_",       "gpqa",      "gpqa"),
    ("simpleqa_",   "simpleqa",  "simpleqa"),
    ("hotpot_",     "hotpotqa",  "hotpotqa"),
    ("leetcode_",   "livecodebench", "livecodebench"),
    ("cruxeval_",   "cruxeval",  "cruxeval"),
    ("bcb_",        "bigcodebench", "bigcodebench"),
    ("debugbench_", "debugbench", "debugbench"),
    ("usaco_",      "usaco",     "usaco"),
    ("gaia_",       "gaia",      "gaia"),
    # YAML-based debug suites (unique prefixes)
    ("ma_hard_",    "mode_advantage_hard", "mode_advantage_hard"),
    ("ma_",         "mode_advantage",      "mode_advantage"),
]


def parse_question_id(qid: str) -> tuple[str, str, str]:
    """Return (suite, sub_dataset, suffix) from a question ID."""
    for prefix, suite, sub in PREFIX_MAP:
        if qid.startswith(prefix):
            return suite, sub, qid[len(prefix):]
    # Fallback: try the first token before _
    parts = qid.split("_", 1)
    return parts[0], parts[0], parts[1] if len(parts) > 1 else ""


def _find_yaml_question(suite: str, qid: str) -> dict | None:
    """Look up a question by ID directly from suite YAML files."""
    yaml_dirs = [
        PROJECT_ROOT / "benchmarks" / "prompts" / "v1",
        PROJECT_ROOT / "benchmarks" / "prompts" / "debug",
    ]
    for d in yaml_dirs:
        yaml_path = d / f"{suite}.yaml"
        if not yaml_path.exists():
            continue
        try:
            import yaml
            with yaml_path.open() as f:
                data = yaml.safe_load(f)
            prompts = data.get("prompts", {})
            if qid in prompts:
                q = prompts[qid]
                return {
                    "id": qid,
                    "suite": suite,
                    "tier": q.get("tier", 1),
                    "scoring_method": q.get("scoring_method", data.get("scoring_method", "?")),
                    "expected": q.get("expected", "?"),
                    "prompt": q.get("prompt", "(no prompt)"),
                }
        except Exception as exc:
            print(f"  Error reading {yaml_path}: {exc}", file=sys.stderr)
    return None


def find_question_in_adapter(suite: str, sub_dataset: str, suffix: str, qid: str) -> dict | None:
    """Load the adapter, scan for the matching row, return its prompt dict."""
    adapter = get_adapter(suite)
    if adapter is None:
        # Try YAML-based suites directly
        return _find_yaml_question(suite, qid)

    adapter._ensure_loaded()

    # Strategy: scan the dataset looking for a row whose generated ID matches qid.
    # For large datasets we try to extract the numeric index from the suffix first.

    # -- Fast path: extract numeric index from suffix --
    idx_match = re.search(r"(\d+)$", suffix)
    if idx_match:
        candidate_idx = int(idx_match.group(1))
        prompt_dict = _try_index(adapter, suite, sub_dataset, candidate_idx, qid)
        if prompt_dict:
            return prompt_dict

    # -- Slow path: linear scan (capped) --
    total = adapter.total_available
    cap = min(total, 50_000)
    for i in range(cap):
        try:
            prompt_dict = _build_prompt_at(adapter, suite, sub_dataset, i)
            if prompt_dict and prompt_dict.get("id") == qid:
                return prompt_dict
        except (IndexError, KeyError):
            continue

    return None


def _try_index(adapter, suite: str, sub: str, idx: int, qid: str) -> dict | None:
    """Try building a prompt at a specific index and check if ID matches."""
    try:
        prompt_dict = _build_prompt_at(adapter, suite, sub, idx)
        if prompt_dict and prompt_dict.get("id") == qid:
            return prompt_dict
    except (IndexError, KeyError):
        pass
    return None


def _build_prompt_at(adapter, suite: str, sub: str, idx: int) -> dict | None:
    """Build a prompt dict at index idx, handling multi-source adapters."""
    # For single-source adapters, use _row_to_prompt directly
    ds = adapter._dataset
    if ds is None or idx >= len(ds):
        return None

    # Multi-source adapters need offset calculation
    if suite == "thinking":
        arc_len = len(adapter._arc) if adapter._arc else 0
        if sub == "arc" and idx < arc_len:
            return adapter._arc_prompt(idx)
        elif sub == "hellaswag":
            hs_len = len(adapter._hellaswag) if adapter._hellaswag else 0
            if idx < hs_len:
                return adapter._hellaswag_prompt(idx)
    elif suite == "math":
        gsm_len = len(adapter._gsm8k) if adapter._gsm8k else 0
        if sub == "gsm8k" and idx < gsm_len:
            return adapter._gsm8k_prompt(idx, adapter._gsm8k[idx])
        elif sub == "math500":
            m500_len = len(adapter._math500) if adapter._math500 else 0
            if idx < m500_len:
                return adapter._math500_prompt(idx, adapter._math500[idx])
    elif suite == "coder":
        he_len = len(adapter._humaneval) if adapter._humaneval else 0
        if sub == "humaneval" and idx < he_len:
            return adapter._humaneval_prompt(idx)
        elif sub == "mbpp":
            mbpp_len = len(adapter._mbpp) if adapter._mbpp else 0
            if idx < mbpp_len:
                return adapter._mbpp_prompt(idx)
    elif suite == "cruxeval":
        raw_len = len(adapter._raw_dataset) if adapter._raw_dataset else 0
        if sub == "cruxeval" and "output" in str(idx):
            pass  # handled by generic path
        if idx < raw_len:
            return adapter._output_prompt(idx)
    else:
        # Single-source adapters
        row = ds[idx]
        if hasattr(row, '__getitem__'):
            return adapter._row_to_prompt(idx, row)

    return None


# ── Eval result search ──────────────────────────────────────────────────────

def find_eval_results(qid: str) -> list[tuple[str, dict]]:
    """Search all 3way eval JSONL files for this question ID.
    Returns list of (filename, parsed_json) tuples.
    """
    eval_dir = PROJECT_ROOT / "benchmarks" / "results" / "eval"
    results = []
    for path in sorted(eval_dir.glob("3way_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Quick string check before parsing
                if qid not in line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("question_id") == qid:
                        results.append((path.name, data))
                except json.JSONDecodeError:
                    continue
    return results


# ── Formatting ──────────────────────────────────────────────────────────────

ROLE_ORDER = [
    "frontdoor:direct",
    "frontdoor:repl",
    "architect_general:delegated",
    "architect_coding:delegated",
]

PASS_SYM = {True: "\033[32mPASS\033[0m", False: "\033[31mFAIL\033[0m"}


def fmt_question(prompt_dict: dict) -> str:
    """Format the full question section."""
    lines = [
        "=" * 72,
        f"  Question ID : {prompt_dict['id']}",
        f"  Suite       : {prompt_dict['suite']}",
        f"  Tier        : T{prompt_dict.get('tier', '?')}",
        f"  Scoring     : {prompt_dict.get('scoring_method', '?')}",
        f"  Expected    : {prompt_dict.get('expected', '?')}",
        "=" * 72,
        "",
        prompt_dict.get("prompt", "(no prompt)"),
        "",
    ]
    return "\n".join(lines)


def fmt_role_result(role_key: str, rr: dict) -> str:
    """Format one role's result."""
    passed = rr.get("passed", False)
    elapsed = rr.get("elapsed_seconds", 0)
    tps = rr.get("predicted_tps", 0)
    tokens = rr.get("tokens_generated", 0)
    gen_ms = rr.get("generation_ms", 0)
    answer = rr.get("answer", "")

    # Truncate long answers
    answer_display = answer
    if len(answer_display) > 300:
        answer_display = answer_display[:297] + "..."

    lines = [
        f"  --- {role_key} ---",
        f"  Result  : {PASS_SYM[passed]}",
        f"  Answer  : {answer_display}",
        f"  Elapsed : {elapsed:.2f}s  |  Tokens: {tokens}  |  Speed: {tps:.1f} t/s  |  Gen: {gen_ms:.0f}ms",
    ]
    tools = rr.get("tools_called", [])
    if tools:
        lines.append(f"  Tools   : {', '.join(tools)}")
    return "\n".join(lines)


def fmt_eval_result(filename: str, data: dict) -> str:
    """Format a complete eval result entry."""
    lines = [
        "-" * 72,
        f"  Eval file : {filename}",
        f"  Timestamp : {data.get('timestamp', '?')}",
        "",
    ]

    # Role results
    role_results = data.get("role_results", {})
    for role_key in ROLE_ORDER:
        if role_key in role_results:
            lines.append(fmt_role_result(role_key, role_results[role_key]))
            lines.append("")

    # Any extra roles not in the standard order
    for role_key in sorted(role_results.keys()):
        if role_key not in ROLE_ORDER:
            lines.append(fmt_role_result(role_key, role_results[role_key]))
            lines.append("")

    # Reward summary
    rewards = data.get("rewards", {})
    meta = data.get("metadata", {})
    tool_adv = meta.get("tool_advantage", "?")
    architect_role = meta.get("architect_role", "?")
    architect_eval = meta.get("architect_eval", {})
    best_architect = architect_eval.get("best", "?")
    heuristic_pick = architect_eval.get("heuristic_would_pick", "?")

    lines.append("  --- Rewards ---")
    for k, v in sorted(rewards.items()):
        lines.append(f"    {k:20s} : {v}")
    lines.append(f"    {'tool_advantage':20s} : {tool_adv}")
    lines.append(f"    {'architect_selected':20s} : {architect_role}")
    lines.append(f"    {'best_architect':20s} : {best_architect}")
    lines.append(f"    {'heuristic_pick':20s} : {heuristic_pick}")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: lookup_question.py <question_id>")
        print("Examples:")
        print("  lookup_question.py hellaswag_27747")
        print("  lookup_question.py arc_Mercury_7111178")
        print("  lookup_question.py mmlu_computer_security_01498")
        sys.exit(1)

    qid = sys.argv[1]
    # Handle suite/qid format from seeding logs (e.g. "thinking/hellaswag_21243", "agentic/t1_q1_sequential")
    explicit_suite = None
    if "/" in qid:
        explicit_suite, qid = qid.rsplit("/", 1)
    suite, sub_dataset, suffix = parse_question_id(qid)
    # If user provided explicit suite, override the prefix-based guess
    # (needed for YAML suites sharing t1_/t2_/t3_ prefixes)
    if explicit_suite:
        suite = explicit_suite
        sub_dataset = explicit_suite
    print(f"Parsed: suite={suite}, sub={sub_dataset}, suffix={suffix}")

    # Section 1: Reconstruct question from HuggingFace
    print("\nLoading dataset adapter...")
    prompt_dict = find_question_in_adapter(suite, sub_dataset, suffix, qid)
    if prompt_dict:
        print(fmt_question(prompt_dict))
    else:
        print(f"\n  [!] Could not reconstruct question from adapter (suite={suite})")
        print(f"      The question may use a non-standard ID or the dataset is unavailable.")
        print()

    # Section 2: Find eval results
    eval_results = find_eval_results(qid)
    if eval_results:
        print(f"Found {len(eval_results)} eval result(s):\n")
        for filename, data in eval_results:
            print(fmt_eval_result(filename, data))
            print()
    else:
        print("No eval results found for this question ID.")
        eval_dir = PROJECT_ROOT / "benchmarks" / "results" / "eval"
        files = list(eval_dir.glob("3way_*.jsonl"))
        if files:
            print(f"  Searched {len(files)} file(s) in {eval_dir}")
        else:
            print(f"  No 3way_*.jsonl files found in {eval_dir}")


if __name__ == "__main__":
    main()
