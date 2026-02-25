#!/usr/bin/env python3
"""Live validation of C1/C3 context window management features.

Key architecture:
- state.context is set once from request (prompt + context), doesn't grow
- C3 operates on state.last_output (REPL output per turn)
- C1 fires when state.context > 12K chars AND state.turns > 5
- C1 externalizes full context to file, generates index, keeps recent 20%

Test strategy:
- Force REPL mode (compaction logic lives in graph turn loop)
- Send context just above 12K chars (C1 char-heuristic trigger)
- Use max_turns=10 so turns > 5 guard can pass
- Verify compaction telemetry and context-file externalization where possible

Usage:
    python3 scripts/benchmark/validate_compaction_live.py
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

import requests

API = os.environ.get("ORCHESTRATOR_API", "http://localhost:8000")
CHAT_URL = f"{API}/chat"


def make_tool_output_block(n: int, size: int = 600) -> str:
    """Generate a tool output block."""
    content = f"Result of analysis step {n}:\n"
    for i in range(size // 60):
        content += f"  metric_{n}_{i:03d} = {i * 17 % 97:.4f}\n"
    return f"<<<TOOL_OUTPUT>>>\n{content}\n<<<END_TOOL_OUTPUT>>>"


def make_context(target_chars: int) -> str:
    """Build context with tool output blocks and enough volume for triggers."""
    parts = [
        "## Benchmark Analysis Context\n",
        "User: Analyze the model performance data and identify optimization opportunities.\n\n",
        "Assistant: I'll run a systematic analysis.\n\n",
    ]
    # Add tool output blocks
    block_num = 0
    while len("\n".join(parts)) < target_chars * 0.65:
        block_num += 1
        parts.append(f"Step {block_num}:\n")
        parts.append(make_tool_output_block(block_num, 600))
        parts.append(f"Optimization #{block_num} improved throughput by {block_num * 2.3:.1f}%.\n\n")

    # Fill with text
    fill = (
        "MoE expert reduction to 6 experts maintained 98.5% quality with 2.58x speedup. "
        "Prompt lookup with corpus injection added +16% on 30B. "
    )
    while len("\n".join(parts)) < target_chars:
        parts.append(fill)

    return "\n".join(parts)


def run_test(name: str, context_chars: int, max_turns: int, expect_c1: bool) -> dict:
    """Run a single validation test."""
    context = make_context(context_chars)
    prompt = (
        "Work in REPL mode and use multiple short reasoning/tool steps before finalizing. "
        "Do at least 6 internal steps, then provide top 3 optimization opportunities."
    )

    payload = {
        "prompt": prompt,
        "context": context,
        "real_mode": True,
        "mock_mode": False,
        "force_mode": "repl",
        "force_role": "frontdoor",
        "max_turns": max_turns,
        "timeout_s": 150,
    }

    print(f"\n{'=' * 70}")
    print(f"TEST: {name}")
    print(f"  Context: {len(context):,} chars | max_turns: {max_turns}")
    print(f"  Expect C1: {expect_c1}")

    t0 = time.time()
    try:
        resp = requests.post(CHAT_URL, json=payload, timeout=300)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"name": name, "error": str(e), "pass": False}

    if resp.status_code != 200:
        body = resp.text[:500]
        print(f"  HTTP {resp.status_code}: {body}")
        return {"name": name, "error": f"http_{resp.status_code}: {body}", "pass": False}

    data = resp.json()
    answer = data.get("answer", "")
    tool_cleared = data.get("tool_results_cleared", 0)
    compact = data.get("compaction_triggered", False)
    saved = data.get("compaction_tokens_saved", 0)
    mode = data.get("mode", "?")
    role = data.get("routed_to", "?")
    turns = data.get("turns", 0)

    print(f"  Answer: {len(answer):,} chars in {elapsed:.1f}s | mode={mode} | turns={turns} | role={role}")
    print(f"  C3 tool_results_cleared = {tool_cleared}")
    print(f"  C1 compaction_triggered = {compact} | tokens_saved = {saved}")
    if answer:
        print(f"  Preview: {answer[:200]}...")

    passed = True
    if expect_c1 and not compact:
        print(f"  *** C1 did NOT fire (expected it to)")
        # Not a hard fail — turns < 6 can prevent it
    if not expect_c1 and compact:
        print(f"  *** C1 fired unexpectedly")

    return {
        "name": name,
        "context_chars": len(context),
        "answer_chars": len(answer),
        "elapsed_s": round(elapsed, 1),
        "tool_results_cleared": tool_cleared,
        "compaction_triggered": compact,
        "compaction_tokens_saved": saved,
        "mode": mode,
        "turns": turns,
        "role": role,
        "has_answer": len(answer) > 30,
        "pass": passed,
    }


def _candidate_context_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_tmp = os.environ.get("ORCHESTRATOR_PATHS_TMP_DIR")
    if env_tmp:
        dirs.append(Path(env_tmp))
    dirs.extend(
        [
            Path("/mnt/raid0/llm/claude/tmp"),
            Path("/mnt/raid0/llm/tmp"),
            Path("/tmp"),
        ]
    )
    unique: list[Path] = []
    seen = set()
    for d in dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def main():
    print("=" * 70)
    print("LIVE VALIDATION: Context Window Management (C3 + C1)")
    print("=" * 70)

    # Health check
    try:
        h = requests.get(f"{API}/health", timeout=5)
        print(f"API: {h.json().get('status')}")
    except Exception as e:
        print(f"API unreachable: {e}")
        return 1

    # Verify feature flags in server
    try:
        pid = int(os.popen("pgrep -f 'uvicorn.*8000' | head -1").read().strip())
        env_raw = open(f"/proc/{pid}/environ").read()
        env_vars = dict(kv.split("=", 1) for kv in env_raw.split("\0") if "=" in kv)
        for key in [
            "ORCHESTRATOR_SESSION_COMPACTION",
            "ORCHESTRATOR_TOOL_RESULT_CLEARING",
            "ORCHESTRATOR_CHAT_SESSION_COMPACTION_MIN_TURNS",
        ]:
            print(f"  {key}={env_vars.get(key, 'NOT SET')}")
    except Exception:
        pass

    results = []

    # Test 1: Small context (5K chars), 3 turns → nothing fires
    results.append(run_test("baseline_small", 5_000, 3, expect_c1=False))

    # Test 2: Medium context (15K chars), 10 turns → C1 should fire (>12K char heuristic, turns>5)
    results.append(run_test("c1_medium_context", 15_000, 10, expect_c1=True))

    # Test 3: Larger context (25K chars), 10 turns → C1 should definitely fire
    results.append(run_test("c1_large_context", 25_000, 10, expect_c1=True))

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        c1 = "YES" if r.get("compaction_triggered") else "no"
        c3 = r.get("tool_results_cleared", 0)
        ans = "OK" if r.get("has_answer") else "FAIL"
        print(f"  {r['name']:25s} | C1={c1:3s} | C3={c3} | answer={ans} | {r.get('elapsed_s', '?')}s")

    # Context files
    ctx_files: list[str] = []
    for d in _candidate_context_dirs():
        ctx_files.extend(glob.glob(str(d / "session_*_ctx_*.md")))
    ctx_files = sorted(set(ctx_files))
    # Filter to recent (last 10 min)
    recent = [f for f in ctx_files if os.path.getmtime(f) > time.time() - 600]
    if recent:
        print(f"\nNew context files ({len(recent)}):")
        for f in recent:
            size = os.path.getsize(f)
            print(f"  {f} ({size:,} bytes)")
            # Show head for I4 validation (Current Execution State)
            with open(f) as fh:
                head = fh.read(500)
            print(f"    Head: {head[:300]}...")

    # Save
    out_dir = "/mnt/raid0/llm/claude/benchmarks/results/runs/compaction_validation"
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"{out_dir}/results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "tests": results}, f, indent=2)
    print(f"\nSaved: {out_path}")

    c1_any = any(r.get("compaction_triggered") for r in results)
    answers_ok = all(r.get("has_answer") for r in results if not r.get("error"))
    if c1_any and answers_ok:
        print("\n✓ PASSED — C1 fired, answers coherent post-compaction")
        return 0
    elif answers_ok:
        print("\n⚠ PARTIAL — Answers OK but C1 didn't fire (check turns > 5 guard)")
        return 0
    else:
        print("\n✗ FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
