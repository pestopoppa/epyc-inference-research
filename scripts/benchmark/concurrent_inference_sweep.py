#!/usr/bin/env python3
"""Concurrent inference sweep: benchmark optimal concurrency per model tier.

Measures per-request and aggregate throughput at varying concurrency levels
for each model server. Helps determine optimal -np (parallel slots) settings.

Usage:
    python scripts/benchmark/concurrent_inference_sweep.py --dry-run
    python scripts/benchmark/concurrent_inference_sweep.py --roles frontdoor,worker
    python scripts/benchmark/concurrent_inference_sweep.py --skip-architects
    python scripts/benchmark/concurrent_inference_sweep.py --concurrency 1,2,4
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install with: pip install httpx", file=sys.stderr)
    sys.exit(1)

# Fixed ~300-token prompt for consistent benchmarking
SWEEP_PROMPT = """You are reviewing the following codebase. Explain what each function does and suggest improvements:

```python
import hashlib
import os
from pathlib import Path
from typing import Iterator

def compute_hash(data: bytes, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()

def stream_file(path: Path, chunk_size: int = 8192) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk

def find_duplicates(directory: Path) -> dict[str, list[Path]]:
    hashes: dict[str, list[Path]] = {}
    for path in directory.rglob("*"):
        if path.is_file():
            file_hash = compute_hash(path.read_bytes())
            hashes.setdefault(file_hash, []).append(path)
    return {h: paths for h, paths in hashes.items() if len(paths) > 1}

class FileWatcher:
    def __init__(self, directory: Path):
        self.directory = directory
        self._cache: dict[Path, float] = {}

    def scan(self) -> list[Path]:
        changed = []
        for path in self.directory.rglob("*"):
            if path.is_file():
                mtime = path.stat().st_mtime
                if self._cache.get(path) != mtime:
                    self._cache[path] = mtime
                    changed.append(path)
        return changed
```"""

# Test matrix: role -> (port, current_np, concurrency_levels)
DEFAULT_TEST_MATRIX: dict[str, dict] = {
    "frontdoor": {
        "port": 8080,
        "current_np": 1,
        "concurrency": [1, 2, 3],
        "description": "Qwen3-Coder-30B-A3B (3B active MoE)",
    },
    "coder": {
        "port": 8081,
        "current_np": 1,
        "concurrency": [1, 2],
        "description": "Qwen2.5-Coder-32B (dense)",
    },
    "worker": {
        "port": 8082,
        "current_np": 2,
        "concurrency": [1, 2, 3, 4],
        "description": "Qwen2.5-7B-f16",
    },
    "architect_general": {
        "port": 8083,
        "current_np": 1,
        "concurrency": [1, 2],
        "description": "Qwen3-235B-A22B",
    },
    "fast_worker": {
        "port": 8102,
        "current_np": 4,
        "concurrency": [1, 2, 4, 6, 8],
        "description": "Qwen2.5-1.5B",
    },
}

ARCHITECT_ROLES = {"architect_general", "architect_coding"}
MAX_CONCURRENCY = 8


@dataclass
class SweepResult:
    """Result from a single (port, concurrency) measurement."""

    timestamp: str
    port: int
    role: str
    current_np: int
    concurrency: int
    np_warning: bool
    per_tps_mean: float
    per_tps_stdev: float
    agg_tps: float
    p50_latency_ms: float
    p95_latency_ms: float
    ttft_baseline_ms: float
    prompt_tokens: int
    n_predict: int
    success_count: int
    total_count: int
    error_rate: float


@dataclass
class SingleRequestResult:
    """Result from a single inference request."""

    per_tps: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    predicted_tokens: int = 0
    success: bool = False
    error: str = ""


async def _send_request(
    client: httpx.AsyncClient,
    port: int,
    prompt: str,
    n_predict: int,
) -> SingleRequestResult:
    """Send a single completion request and extract timings."""
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"http://localhost:{port}/completion",
            json=payload,
            timeout=120.0,
        )
        latency = (time.perf_counter() - start) * 1000
        if resp.status_code != 200:
            return SingleRequestResult(error=f"HTTP {resp.status_code}")
        data = resp.json()
        timings = data.get("timings", {})
        return SingleRequestResult(
            per_tps=timings.get("predicted_per_second", 0.0),
            latency_ms=latency,
            prompt_tokens=timings.get("prompt_n", 0),
            predicted_tokens=timings.get("predicted_n", 0),
            success=True,
        )
    except Exception as e:
        return SingleRequestResult(error=str(e))


async def _measure_ttft_baseline(
    client: httpx.AsyncClient,
    port: int,
    prompt: str,
    runs: int = 3,
) -> float:
    """Measure baseline TTFT via streaming probe (median of runs)."""
    ttfts = []
    payload = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": True,
    }
    for _ in range(runs):
        start = time.perf_counter()
        try:
            async with client.stream(
                "POST",
                f"http://localhost:{port}/completion",
                json=payload,
                timeout=60.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        ttft = (time.perf_counter() - start) * 1000
                        ttfts.append(ttft)
                        break
        except Exception:
            pass
    return statistics.median(ttfts) if ttfts else -1.0


async def _run_concurrent_batch(
    client: httpx.AsyncClient,
    port: int,
    concurrency: int,
    prompt: str,
    n_predict: int,
) -> list[SingleRequestResult]:
    """Send `concurrency` requests in parallel and collect results."""
    tasks = [
        _send_request(client, port, prompt, n_predict)
        for _ in range(concurrency)
    ]
    return await asyncio.gather(*tasks)


async def _check_server_health(port: int) -> bool:
    """Check if a server is responding on the given port."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://localhost:{port}/health",
                timeout=5.0,
            )
            return resp.status_code == 200
    except Exception:
        return False


async def sweep_role(
    role: str,
    config: dict,
    n_warmup: int,
    n_measured: int,
    n_predict: int,
    csv_writer,
    csv_file,
    dry_run: bool = False,
) -> list[SweepResult]:
    """Run the full sweep for a single role."""
    port = config["port"]
    current_np = config["current_np"]
    concurrencies = config["concurrency"]
    desc = config["description"]

    print(f"\n{'='*60}")
    print(f"Role: {role} | Port: {port} | {desc}")
    print(f"Current -np: {current_np} | Test concurrency: {concurrencies}")
    print(f"{'='*60}")

    if dry_run:
        for c in concurrencies:
            warn = c > current_np
            print(f"  [DRY-RUN] concurrency={c} {'⚠ >np' if warn else '✓'}")
        return []

    # Health check
    healthy = await _check_server_health(port)
    if not healthy:
        print(f"  ⚠ Server on port {port} not responding, skipping")
        return []

    results = []
    async with httpx.AsyncClient() as client:
        # TTFT baseline
        print(f"  Measuring TTFT baseline (3 runs)...")
        ttft_baseline = await _measure_ttft_baseline(client, port, SWEEP_PROMPT)
        print(f"  TTFT baseline: {ttft_baseline:.1f}ms")

        for concurrency in concurrencies:
            if concurrency > MAX_CONCURRENCY:
                print(f"  ⚠ Concurrency {concurrency} > max {MAX_CONCURRENCY}, skipping")
                continue

            np_warning = concurrency > current_np
            warn_str = " ⚠ concurrency>np (queuing)" if np_warning else ""
            print(f"\n  concurrency={concurrency}{warn_str}")

            # Warmup
            print(f"    Warmup: {n_warmup} batches...", end="", flush=True)
            for _ in range(n_warmup):
                await _run_concurrent_batch(client, port, concurrency, SWEEP_PROMPT, n_predict)
                print(".", end="", flush=True)
            print(" done")

            # Measured batches
            all_per_tps = []
            all_latencies = []
            success_count = 0
            total_count = 0
            print(f"    Measured: {n_measured} batches...", end="", flush=True)
            for _ in range(n_measured):
                batch = await _run_concurrent_batch(
                    client, port, concurrency, SWEEP_PROMPT, n_predict
                )
                total_count += len(batch)
                for r in batch:
                    if r.success:
                        all_per_tps.append(r.per_tps)
                        all_latencies.append(r.latency_ms)
                        success_count += 1
                print(".", end="", flush=True)
            print(" done")

            if not all_per_tps:
                print(f"    ✗ All requests failed")
                continue

            per_tps_mean = statistics.mean(all_per_tps)
            per_tps_stdev = statistics.stdev(all_per_tps) if len(all_per_tps) > 1 else 0.0
            agg_tps = per_tps_mean * concurrency
            latencies_sorted = sorted(all_latencies)
            p50 = latencies_sorted[len(latencies_sorted) // 2]
            p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)
            p95 = latencies_sorted[p95_idx]
            prompt_tokens = batch[0].prompt_tokens if batch[0].success else 0
            error_rate = (
                (total_count - success_count) / total_count if total_count > 0 else 1.0
            )

            result = SweepResult(
                timestamp=datetime.now().isoformat(),
                port=port,
                role=role,
                current_np=current_np,
                concurrency=concurrency,
                np_warning=np_warning,
                per_tps_mean=per_tps_mean,
                per_tps_stdev=per_tps_stdev,
                agg_tps=agg_tps,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                ttft_baseline_ms=ttft_baseline,
                prompt_tokens=prompt_tokens,
                n_predict=n_predict,
                success_count=success_count,
                total_count=total_count,
                error_rate=error_rate,
            )
            results.append(result)

            # Incremental CSV write
            csv_writer.writerow([
                result.timestamp, result.port, result.role, result.current_np,
                result.concurrency, result.np_warning,
                f"{result.per_tps_mean:.2f}", f"{result.per_tps_stdev:.2f}",
                f"{result.agg_tps:.2f}", f"{result.p50_latency_ms:.1f}",
                f"{result.p95_latency_ms:.1f}", f"{result.ttft_baseline_ms:.1f}",
                result.prompt_tokens, result.n_predict, result.success_count, result.total_count,
                f"{result.error_rate:.4f}",
            ])
            csv_file.flush()

            print(f"    per_tps: {per_tps_mean:.2f} ± {per_tps_stdev:.2f} t/s")
            print(f"    agg_tps: {agg_tps:.2f} t/s")
            print(f"    latency: p50={p50:.0f}ms p95={p95:.0f}ms")
            print(
                f"    success: {success_count}/{total_count} "
                f"(error_rate={error_rate * 100:.1f}%)"
            )

    return results


async def main(args: argparse.Namespace) -> None:
    """Run the full concurrent inference sweep."""
    # Filter test matrix
    matrix = dict(DEFAULT_TEST_MATRIX)
    if args.roles:
        roles = {r.strip() for r in args.roles.split(",")}
        matrix = {k: v for k, v in matrix.items() if k in roles}
    if args.skip_architects:
        matrix = {k: v for k, v in matrix.items() if k not in ARCHITECT_ROLES}

    # Override concurrency if specified
    if args.concurrency:
        concurrency_list = [int(c) for c in args.concurrency.split(",")]
        for config in matrix.values():
            config["concurrency"] = concurrency_list

    if not matrix:
        print("No roles selected. Available:", ", ".join(DEFAULT_TEST_MATRIX.keys()))
        return

    # Print test plan
    print("Concurrent Inference Sweep")
    print(f"  n_predict: {args.n_predict}")
    print(f"  warmup batches: {args.n_warmup}")
    print(f"  measured batches: {args.n_measured}")
    print(f"  roles: {', '.join(matrix.keys())}")

    if args.dry_run:
        print("\n  [DRY-RUN MODE]")
        for role, config in matrix.items():
            await sweep_role(role, config, 0, 0, 0, None, None, dry_run=True)
        return

    # Confirm architect ports
    architect_roles = {k for k in matrix if k in ARCHITECT_ROLES}
    if architect_roles and not args.yes:
        print(f"\n⚠ About to benchmark architect ports: {architect_roles}")
        print("  This sends concurrent requests to large models.")
        resp = input("  Continue? [y/N] ")
        if resp.lower() != "y":
            print("Aborted.")
            return

    # Setup CSV output
    results_dir = Path("benchmarks/results/eval")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"concurrent_sweep_{ts}.csv"

    csv_headers = [
        "timestamp", "port", "role", "current_np", "concurrency", "np_warning",
        "per_tps_mean", "per_tps_stdev", "agg_tps", "p50_latency_ms",
        "p95_latency_ms", "ttft_baseline_ms", "prompt_tokens", "n_predict",
        "success_count", "total_count", "error_rate",
    ]

    all_results = []
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)
        csv_file.flush()

        for role, config in matrix.items():
            results = await sweep_role(
                role, config,
                n_warmup=args.n_warmup,
                n_measured=args.n_measured,
                n_predict=args.n_predict,
                csv_writer=writer,
                csv_file=csv_file,
                dry_run=False,
            )
            all_results.extend(results)

    print(f"\n{'='*60}")
    print(f"Results written to: {csv_path}")
    print(f"Total measurements: {len(all_results)}")

    # Summary table
    if all_results:
        print(f"\nSummary:")
        print(f"{'Role':<20} {'Conc':>4} {'Per TPS':>8} {'Agg TPS':>8} {'P50ms':>7} {'P95ms':>7}")
        print("-" * 60)
        for r in all_results:
            warn = "⚠" if r.np_warning else " "
            print(f"{r.role:<20} {r.concurrency:>3}{warn} {r.per_tps_mean:>8.2f} "
                  f"{r.agg_tps:>8.2f} {r.p50_latency_ms:>7.0f} {r.p95_latency_ms:>7.0f}")

    summary = build_recommendations(
        all_results,
        min_agg_gain_pct=args.min_agg_gain_pct,
        max_p95_multiplier=args.max_p95_multiplier,
        max_error_rate=args.max_error_rate,
    )
    summary_path = csv_path.with_suffix(".summary.json")
    summary_payload = {
        "created_at": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "args": {
            "roles": args.roles,
            "concurrency": args.concurrency,
            "n_warmup": args.n_warmup,
            "n_measured": args.n_measured,
            "n_predict": args.n_predict,
            "skip_architects": args.skip_architects,
            "min_agg_gain_pct": args.min_agg_gain_pct,
            "max_p95_multiplier": args.max_p95_multiplier,
            "max_error_rate": args.max_error_rate,
        },
        "recommendations": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"Summary JSON written to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent inference sweep: benchmark optimal concurrency per model tier"
    )
    parser.add_argument("--roles", type=str, default="",
                       help="Comma-separated roles to test (default: all)")
    parser.add_argument("--concurrency", type=str, default="",
                       help="Override concurrency levels (e.g., '1,2,4')")
    parser.add_argument("--n-warmup", type=int, default=2,
                       help="Warmup batches per config (default: 2)")
    parser.add_argument("--n-measured", type=int, default=5,
                       help="Measured batches per config (default: 5)")
    parser.add_argument("--n-predict", type=int, default=128,
                       help="Tokens per request (default: 128)")
    parser.add_argument("--skip-architects", action="store_true",
                       help="Skip architect ports (8083/8084)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print test plan without running")
    parser.add_argument("--yes", "-y", action="store_true",
                       help="Skip confirmation prompts")
    parser.add_argument(
        "--min-agg-gain-pct",
        type=float,
        default=10.0,
        help="Minimum aggregate TPS gain vs c=1 to recommend higher concurrency (default: 10.0)",
    )
    parser.add_argument(
        "--max-p95-multiplier",
        type=float,
        default=1.5,
        help="Maximum allowed p95 multiplier vs c=1 (default: 1.5)",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.02,
        help="Maximum allowed error rate for recommended concurrency (default: 0.02)",
    )
    return parser.parse_args()


def build_recommendations(
    results: list[SweepResult],
    *,
    min_agg_gain_pct: float,
    max_p95_multiplier: float,
    max_error_rate: float,
) -> list[dict]:
    """Derive per-role concurrency recommendations from sweep output."""
    by_role: dict[str, list[SweepResult]] = {}
    for r in results:
        by_role.setdefault(r.role, []).append(r)

    recommendations: list[dict] = []
    for role, role_results in sorted(by_role.items()):
        ordered = sorted(role_results, key=lambda r: r.concurrency)
        baseline = next((r for r in ordered if r.concurrency == 1), None)
        if baseline is None:
            recommendations.append(
                {
                    "role": role,
                    "status": "insufficient_baseline",
                    "reason": "missing concurrency=1 baseline",
                    "recommended_concurrency": None,
                }
            )
            continue

        accepted: list[SweepResult] = []
        rejected: list[dict[str, object]] = []
        for r in ordered:
            if r.concurrency == 1:
                accepted.append(r)
                continue

            gain_pct = ((r.agg_tps - baseline.agg_tps) / baseline.agg_tps * 100) if baseline.agg_tps > 0 else 0.0
            p95_mult = (r.p95_latency_ms / baseline.p95_latency_ms) if baseline.p95_latency_ms > 0 else 999.0
            checks = {
                "agg_gain_ok": gain_pct >= min_agg_gain_pct,
                "p95_ok": p95_mult <= max_p95_multiplier,
                "error_ok": r.error_rate <= max_error_rate,
            }
            if all(checks.values()):
                accepted.append(r)
            else:
                rejected.append(
                    {
                        "concurrency": r.concurrency,
                        "agg_gain_pct": round(gain_pct, 2),
                        "p95_multiplier": round(p95_mult, 3),
                        "error_rate": round(r.error_rate, 4),
                        "checks": checks,
                    }
                )

        best = max(accepted, key=lambda r: r.agg_tps)
        recommendations.append(
            {
                "role": role,
                "status": "ok",
                "recommended_concurrency": best.concurrency,
                "baseline_concurrency": 1,
                "baseline_agg_tps": round(baseline.agg_tps, 2),
                "recommended_agg_tps": round(best.agg_tps, 2),
                "agg_gain_pct": round(
                    ((best.agg_tps - baseline.agg_tps) / baseline.agg_tps * 100) if baseline.agg_tps > 0 else 0.0,
                    2,
                ),
                "recommended_p95_ms": round(best.p95_latency_ms, 1),
                "baseline_p95_ms": round(baseline.p95_latency_ms, 1),
                "recommended_error_rate": round(best.error_rate, 4),
                "rejected_candidates": rejected,
            }
        )
    return recommendations


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
