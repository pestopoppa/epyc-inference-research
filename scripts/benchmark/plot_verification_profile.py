#!/usr/bin/env python3
"""SpecExec Verification Profiling — Plot Generator

Reads Phase 1-3 CSV data from data/specexec/ and produces publication-quality plots.

Usage:
    python scripts/benchmark/plot_verification_profile.py
    python scripts/benchmark/plot_verification_profile.py --phase 1
    python scripts/benchmark/plot_verification_profile.py --phase 2
    python scripts/benchmark/plot_verification_profile.py --phase 3
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "specexec"
PLOT_DIR = DATA_DIR / "plots"


def parse_llama_bench_csv(path: Path) -> list[dict]:
    """Parse llama-bench CSV output."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_phase1():
    """Latency vs batch size curves for each target model."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("phase1_*_distribute.csv"))
    if not files:
        print("No Phase 1 data found. Run profile_verification_cost.sh first.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, f in enumerate(files):
        name = f.stem.replace("phase1_", "").replace("_distribute", "")
        rows = parse_llama_bench_csv(f)
        if not rows:
            continue

        # Extract pp (prompt processing) data: batch_size → avg time
        batch_times: dict[int, list[float]] = {}
        for row in rows:
            # llama-bench CSV columns vary; look for n_prompt and t/s
            n_prompt = int(row.get("n_prompt", row.get("pp", 0)))
            if n_prompt == 0:
                continue
            # Time can be in different columns
            speed = float(row.get("avg_ts", row.get("pp_avg", row.get("t/s", 0))))
            if speed > 0:
                # Convert tokens/s to ms per batch: (n_prompt / speed) * 1000
                time_ms = (n_prompt / speed) * 1000
                batch_times.setdefault(n_prompt, []).append(time_ms)

        if not batch_times:
            print(f"  WARN: No valid data in {f.name}")
            continue

        sizes = sorted(batch_times.keys())
        avg_times = [sum(batch_times[s]) / len(batch_times[s]) for s in sizes]

        ax.plot(sizes, avg_times, "o-", color=colors[i % len(colors)], label=name, linewidth=2, markersize=5)

    ax.set_xlabel("Batch Size (tokens)", fontsize=12)
    ax.set_ylabel("Processing Time (ms)", fontsize=12)
    ax.set_title("Phase 1: Verification Latency vs Batch Size\n(EPYC 9655, --numa distribute)", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    outpath = PLOT_DIR / "phase1_verification_latency.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Phase 1 plot: {outpath}")

    # NUMA comparison plot
    for f_dist in files:
        name = f_dist.stem.replace("phase1_", "").replace("_distribute", "")
        f_isol = DATA_DIR / f"phase1_{name}_isolate.csv"
        if not f_isol.exists():
            continue

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for mode, fpath, style in [("distribute", f_dist, "o-"), ("isolate", f_isol, "s--")]:
            rows = parse_llama_bench_csv(fpath)
            batch_times: dict[int, list[float]] = {}
            for row in rows:
                n_prompt = int(row.get("n_prompt", row.get("pp", 0)))
                if n_prompt == 0:
                    continue
                speed = float(row.get("avg_ts", row.get("pp_avg", row.get("t/s", 0))))
                if speed > 0:
                    time_ms = (n_prompt / speed) * 1000
                    batch_times.setdefault(n_prompt, []).append(time_ms)
            if batch_times:
                sizes = sorted(batch_times.keys())
                avg_times = [sum(batch_times[s]) / len(batch_times[s]) for s in sizes]
                ax2.plot(sizes, avg_times, style, label=f"{mode}", linewidth=2, markersize=5)

        ax2.set_xlabel("Batch Size (tokens)", fontsize=12)
        ax2.set_ylabel("Processing Time (ms)", fontsize=12)
        ax2.set_title(f"NUMA Comparison: {name}", fontsize=13)
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        outpath = PLOT_DIR / f"phase1_numa_{name}.png"
        fig2.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"NUMA plot: {outpath}")


def plot_phase2():
    """Draft model cost comparison bar chart."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    csvpath = DATA_DIR / "phase2_draft_costs.csv"
    if not csvpath.exists():
        print("No Phase 2 data found. Run profile_verification_cost.sh phase2 first.")
        return

    rows = parse_llama_bench_csv(csvpath)
    if not rows:
        print("Phase 2 CSV is empty.")
        return

    # Extract model name → generation speed (tokens/s)
    model_speeds: dict[str, list[float]] = {}
    for row in rows:
        # Model name from the CSV
        model = row.get("model_filename", row.get("model", "unknown"))
        # Clean up path to just model name
        model = os.path.basename(model).replace(".gguf", "")
        speed = float(row.get("avg_ts", row.get("tg_avg", row.get("t/s", 0))))
        if speed > 0:
            model_speeds.setdefault(model, []).append(speed)

    if not model_speeds:
        print("  WARN: No valid generation data in Phase 2 CSV")
        return

    names = sorted(model_speeds.keys())
    avg_speeds = [sum(model_speeds[n]) / len(model_speeds[n]) for n in names]
    per_token_ms = [1000 / s for s in avg_speeds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart: tokens/s
    bars = ax1.barh(names, avg_speeds, color=plt.cm.viridis(
        [i / len(names) for i in range(len(names))]))
    ax1.set_xlabel("Generation Speed (tokens/s)", fontsize=12)
    ax1.set_title("Phase 2: Draft Model Generation Speed", fontsize=13)
    ax1.grid(True, alpha=0.3, axis="x")
    for bar, speed in zip(bars, avg_speeds):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{speed:.0f}", va="center", fontsize=9)

    # Bar chart: ms per token
    bars2 = ax2.barh(names, per_token_ms, color=plt.cm.magma(
        [i / len(names) for i in range(len(names))]))
    ax2.set_xlabel("Time per Token (ms)", fontsize=12)
    ax2.set_title("Phase 2: Draft Model Cost per Token", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="x")
    for bar, ms in zip(bars2, per_token_ms):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{ms:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    outpath = PLOT_DIR / "phase2_draft_costs.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Phase 2 plot: {outpath}")


def plot_phase3():
    """Large-K throughput vs K curves for each pair."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("phase3_*_k*.csv"))
    # Filter out server logs
    files = [f for f in files if "server" not in f.name]
    if not files:
        print("No Phase 3 data found. Run bench_largek_speculation.sh first.")
        return

    # Group by pair
    pair_data: dict[str, list[tuple[int, float, float]]] = {}
    for f in files:
        base = f.stem.replace("phase3_", "")
        parts = base.rsplit("_k", 1)
        if len(parts) != 2:
            continue
        pair, k = parts[0], int(parts[1])
        with open(f) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        speeds = [float(r["speed_tps"]) for r in rows if float(r.get("speed_tps", 0)) > 0]
        accepts = [float(r["acceptance_rate"]) for r in rows if float(r.get("acceptance_rate", 0)) > 0]
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            avg_accept = sum(accepts) / len(accepts) if accepts else 0
            pair_data.setdefault(pair, []).append((k, avg_speed, avg_accept))

    if not pair_data:
        print("  WARN: No valid Phase 3 data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors

    for i, (pair, data) in enumerate(sorted(pair_data.items())):
        data.sort()
        ks = [d[0] for d in data]
        speeds = [d[1] for d in data]
        accepts = [d[2] for d in data]

        color = colors[i % len(colors)]
        ax1.plot(ks, speeds, "o-", color=color, label=pair, linewidth=2, markersize=6)
        ax2.plot(ks, accepts, "s-", color=color, label=pair, linewidth=2, markersize=6)

    ax1.set_xlabel("Draft Max (K)", fontsize=12)
    ax1.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax1.set_title("Phase 3: Throughput vs Draft-Max K", fontsize=13)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Draft Max (K)", fontsize=12)
    ax2.set_ylabel("Acceptance Rate (%)", fontsize=12)
    ax2.set_title("Phase 3: Acceptance Rate vs Draft-Max K", fontsize=13)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = PLOT_DIR / "phase3_largek_throughput.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Phase 3 plot: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="SpecExec Verification Profiling — Plot Generator")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Generate plots for a specific phase (default: all)")
    args = parser.parse_args()

    phases = [args.phase] if args.phase else [1, 2, 3]
    for p in phases:
        print(f"\n=== Phase {p} ===")
        {1: plot_phase1, 2: plot_phase2, 3: plot_phase3}[p]()


if __name__ == "__main__":
    main()
