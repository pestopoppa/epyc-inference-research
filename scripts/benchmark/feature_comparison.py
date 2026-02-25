#!/usr/bin/env python3
"""Compare baseline (all new features OFF) vs candidate (all new features ON).

Usage:
    python3 scripts/benchmark/feature_comparison.py

Requires orchestrator stack running on localhost:8000.
"""
from __future__ import annotations

import json
import signal
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


class PromptTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise PromptTimeout("hard timeout")

ORCHESTRATOR = "http://localhost:8000"
RESULTS_DIR = Path("benchmarks/results/runs/feature_validation/comparison")

# The 15 features enabled in the validation battery
VALIDATED_FEATURES = [
    "specialist_routing", "plan_review", "architect_delegation", "parallel_execution",
    "react_mode", "output_formalizer", "input_formalizer", "unified_streaming",
    "model_fallback", "escalation_compression",
    "approval_gates", "cascading_tool_policy", "resume_tokens",
    "side_effect_tracking", "structured_tool_output",
]

# Load prompts
PROMPT_DIR = Path("benchmarks/prompts/v1/feature_validation")
PROMPTS = []
for manifest in ["general_5.json", "tool_compliance.json"]:
    with open(PROMPT_DIR / manifest) as f:
        PROMPTS.extend(json.load(f))


CLIENT = httpx.Client(timeout=httpx.Timeout(connect=10, read=180, write=10, pool=10))


def set_features(enabled: bool) -> None:
    """Toggle all validated features on or off."""
    config = {feat: enabled for feat in VALIDATED_FEATURES}
    r = CLIENT.post(f"{ORCHESTRATOR}/config", json=config)
    r.raise_for_status()
    state = "ON" if enabled else "OFF"
    print(f"  Features set to {state}")


def run_prompts(label: str) -> dict:
    """Run all prompts and collect metrics."""
    results = []
    for i, p in enumerate(PROMPTS):
        pid = p.get("id", f"p{i}")
        payload = {
            "prompt": p.get("prompt", ""),
            "role": p.get("role", "frontdoor"),
            "mode": p.get("mode", "direct"),
            "mock_mode": False,
            "real_mode": True,
        }
        t0 = time.monotonic()
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(180)  # Hard 180s deadline
            resp = CLIENT.post(f"{ORCHESTRATOR}/chat", json=payload)
            signal.alarm(0)  # Cancel alarm on success
            elapsed = time.monotonic() - t0
            if resp.status_code == 200:
                data = resp.json()
                tokens = data.get("tokens_generated", 0)
                server_tps = data.get("predicted_tps", 0)
                result = {
                    "prompt_id": pid,
                    "status": 200,
                    "elapsed_s": round(elapsed, 2),
                    "tokens_generated": tokens,
                    "client_tps": round(tokens / elapsed, 1) if elapsed > 0 and tokens > 0 else 0,
                    "server_tps": round(server_tps, 1) if server_tps else 0,
                    "routed_to": data.get("routed_to", ""),
                    "turns": data.get("turns", 0),
                    "answer": data.get("answer", "")[:500],
                }
            else:
                result = {
                    "prompt_id": pid,
                    "status": resp.status_code,
                    "elapsed_s": round(elapsed, 2),
                    "tokens_generated": 0,
                    "client_tps": 0,
                    "routed_to": "",
                    "turns": 0,
                    "answer": "",
                }
        except (Exception, PromptTimeout) as e:
            signal.alarm(0)
            elapsed = time.monotonic() - t0
            result = {
                "prompt_id": pid,
                "status": 0,
                "elapsed_s": round(elapsed, 2),
                "tokens_generated": 0,
                "client_tps": 0,
                "routed_to": "",
                "turns": 0,
                "answer": "",
                "error": str(e),
            }

        results.append(result)
        status_str = f"{result['status']}"
        if result["status"] != 200:
            status_str += f" ({result.get('error', 'timeout')})"
        tps_str = ""
        if result.get("server_tps", 0) > 0:
            tps_str = f" ({result['tokens_generated']} tok, {result['server_tps']} t/s)"
        print(f"  [{label}] {pid}: {result['elapsed_s']:.1f}s {status_str}{tps_str} → {result['routed_to']}")

        # Cooldown between prompts to avoid saturating backends
        if i < len(PROMPTS) - 1:
            time.sleep(5)

    # Compute summary stats
    elapsed_vals = [r["elapsed_s"] for r in results if r["status"] == 200]
    tps_vals = [r["client_tps"] for r in results if r["client_tps"] > 0]
    server_tps_vals = [r["server_tps"] for r in results if r.get("server_tps", 0) > 0]

    return {
        "label": label,
        "prompts_run": len(results),
        "success_count": len(elapsed_vals),
        "p50_s": round(statistics.median(elapsed_vals), 2) if elapsed_vals else 0,
        "mean_s": round(statistics.mean(elapsed_vals), 2) if elapsed_vals else 0,
        "total_s": round(sum(r["elapsed_s"] for r in results), 2),
        "avg_tps": round(statistics.mean(tps_vals), 1) if tps_vals else 0,
        "avg_server_tps": round(statistics.mean(server_tps_vals), 1) if server_tps_vals else 0,
        "responses": results,
    }


def main():
    # Health check
    r = CLIENT.get(f"{ORCHESTRATOR}/health")
    r.raise_for_status()
    print("Orchestrator healthy\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Baseline (all validated features OFF)
    print("=== BASELINE (15 features OFF) ===")
    set_features(False)
    time.sleep(2)  # Let config propagate
    baseline = run_prompts("baseline")

    # Cooldown between phases — let backends recover from baseline load
    print("\n--- Cooldown (60s) ---")
    time.sleep(60)

    # Phase 2: Candidate (all validated features ON)
    print("\n=== CANDIDATE (15 features ON) ===")
    set_features(True)
    time.sleep(5)
    candidate = run_prompts("candidate")

    # Restore features to ON (current production default)
    set_features(True)

    # Compute deltas
    lat_delta = candidate["p50_s"] - baseline["p50_s"]
    mean_delta = candidate["mean_s"] - baseline["mean_s"]
    total_delta = candidate["total_s"] - baseline["total_s"]

    # Compute success rates
    n = len(PROMPTS)
    b_rate = baseline["success_count"] / n if n else 0
    c_rate = candidate["success_count"] / n if n else 0
    rate_delta = c_rate - b_rate

    # Determine verdict: success rate trumps latency
    if c_rate > b_rate:
        verdict = "candidate"
        reason = f"higher success rate ({c_rate:.0%} vs {b_rate:.0%})"
    elif b_rate > c_rate:
        verdict = "baseline"
        reason = f"higher success rate ({b_rate:.0%} vs {c_rate:.0%})"
    elif total_delta < 0:
        verdict = "candidate"
        reason = f"same success rate, {abs(total_delta):.1f}s faster wall-clock"
    elif total_delta > 0:
        verdict = "baseline"
        reason = f"same success rate, {abs(total_delta):.1f}s faster wall-clock"
    else:
        verdict = "tie"
        reason = "no difference"

    # Write results
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features_tested": VALIDATED_FEATURES,
        "prompts_count": n,
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "success_rate": round(rate_delta, 2),
            "p50_s": round(lat_delta, 2),
            "mean_s": round(mean_delta, 2),
            "total_s": round(total_delta, 2),
        },
        "verdict": verdict,
        "verdict_reason": reason,
    }

    out_path = RESULTS_DIR / "comparison_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Prompts: {n}")
    print(f"")
    print(f"{'Metric':<20} {'Baseline':>12} {'Candidate':>12} {'Delta':>12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'success rate':<20} {b_rate:>11.0%} {c_rate:>11.0%} {rate_delta:>+11.0%}")
    print(f"{'p50 latency (s)':<20} {baseline['p50_s']:>12.2f} {candidate['p50_s']:>12.2f} {lat_delta:>+12.2f}")
    print(f"{'mean latency (s)':<20} {baseline['mean_s']:>12.2f} {candidate['mean_s']:>12.2f} {mean_delta:>+12.2f}")
    print(f"{'total wall-clock':<20} {baseline['total_s']:>12.2f} {candidate['total_s']:>12.2f} {total_delta:>+12.2f}")
    print(f"{'avg client TPS':<20} {baseline['avg_tps']:>12.1f} {candidate['avg_tps']:>12.1f} {candidate['avg_tps'] - baseline['avg_tps']:>+12.1f}")
    print(f"{'avg server TPS':<20} {baseline['avg_server_tps']:>12.1f} {candidate['avg_server_tps']:>12.1f} {candidate['avg_server_tps'] - baseline['avg_server_tps']:>+12.1f}")
    print(f"")

    print(f"Result: {verdict.upper()} WINS — {reason}" if verdict != "tie" else f"Result: {reason.upper()}")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
