#!/usr/bin/env python3
"""Run a governed, diagnostic-only WGM/L2 locality sweep on gfx90a."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim

SOURCE = REPO_ROOT / "scripts/benchmark/autokernel_wgm_proxy.cpp"
PROFILER_ROOT = Path("/mnt/raid0/llm/tools/rocm-profilers-6.2")
PROFILER_PREFIX = PROFILER_ROOT / "opt/rocm-6.2.0"
PROFILER = PROFILER_PREFIX / "bin/rocprofv2"
FACTORS = (0, 2, 4, 8, 16, 32)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: Sequence[str], *, stdout: Path, stderr: Path,
                timeout_s: float, env: dict[str, str] | None = None) -> float:
    started = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            tuple(command), stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            env=env, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_s)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, 15)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                    process.wait(timeout=10)
            raise
    if returncode:
        tail = stderr.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command exited {returncode}: {tail}")
    return time.monotonic() - started


def logical_mapping(factor: int, linear: int, *, rows: int = 64,
                    cols: int = 64) -> tuple[int, int]:
    if factor == 0:
        return linear // cols, linear % cols
    if factor not in FACTORS:
        raise ValueError(f"unsupported factor {factor}")
    workgroups_per_group = factor * cols
    group_id = linear // workgroups_per_group
    first_m = group_id * factor
    group_size_m = min(rows - first_m, factor)
    in_group = linear % workgroups_per_group
    return first_m + (in_group % group_size_m), in_group // group_size_m


def parse_samples(path: Path, *, rounds: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    headers = [row for row in rows if row.get("type") == "header"]
    samples = [row for row in rows if row.get("type") == "sample"]
    if len(headers) != 1 or headers[0].get("correctness") != "bit_exact":
        raise RuntimeError("probe did not report one bit-exact correctness header")
    expected = rounds * len(FACTORS)
    if len(samples) != expected:
        raise RuntimeError(f"expected {expected} samples, found {len(samples)}")
    counts = {factor: sum(row.get("factor") == factor for row in samples) for factor in FACTORS}
    if any(count != rounds for count in counts.values()):
        raise RuntimeError(f"unbalanced factor samples: {counts}")
    return headers[0], samples


def summarize(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples = list(samples)
    grouped = {factor: [] for factor in FACTORS}
    for row in samples:
        factor = int(row["factor"])
        value = float(row["elapsed_ms"])
        if factor not in grouped or value <= 0:
            raise ValueError("invalid timing sample")
        grouped[factor].append(value)
    medians = {factor: statistics.median(values) for factor, values in grouped.items()}
    baseline = medians[0]
    factors = {
        str(factor): {
            "sample_count": len(grouped[factor]),
            "median_ms": medians[factor],
            "mad_ms": statistics.median(
                abs(value - medians[factor]) for value in grouped[factor]),
            "relative_to_none_pct": (baseline / medians[factor] - 1.0) * 100.0,
        }
        for factor in FACTORS
    }
    best = min(FACTORS, key=lambda factor: medians[factor])
    by_round: dict[int, dict[int, float]] = {}
    for row in samples:
        by_round.setdefault(int(row["round"]), {})[int(row["factor"])] = float(
            row["elapsed_ms"])
    if not by_round or any(set(panel) != set(FACTORS) for panel in by_round.values()):
        raise ValueError("timing samples do not form complete paired rounds")
    panels = list(by_round.values())
    generator = random.Random(20260811)
    draws = 10_000
    paired_improvements = {factor: [] for factor in FACTORS}
    best_counts = {factor: 0 for factor in FACTORS}
    for _ in range(draws):
        selected = [panels[generator.randrange(len(panels))] for _ in panels]
        draw_medians = {
            factor: statistics.median(panel[factor] for panel in selected)
            for factor in FACTORS
        }
        draw_best = min(FACTORS, key=lambda factor: draw_medians[factor])
        best_counts[draw_best] += 1
        for factor in FACTORS:
            paired = [
                (panel[0] / panel[factor] - 1.0) * 100.0 for panel in selected
            ]
            paired_improvements[factor].append(statistics.median(paired))
    for factor in FACTORS:
        ordered = sorted(paired_improvements[factor])
        factors[str(factor)]["paired_improvement_pct_bootstrap_95ci"] = [
            ordered[int(0.025 * draws)], ordered[int(0.975 * draws) - 1]
        ]
        factors[str(factor)]["bootstrap_best_frequency"] = best_counts[factor] / draws
    return {"metric": "elapsed_ms", "direction": "lower_is_better",
            "best_factor": best, "bootstrap_draws": draws, "factors": factors}


def profiler_environment(binary: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_PATH"] = "/opt/rocm"
    env["PATH"] = f"{PROFILER_PREFIX / 'bin'}:/opt/rocm/bin:{env.get('PATH', '')}"
    libs = f"{PROFILER_PREFIX / 'lib'}:{PROFILER_ROOT / 'usr/lib/x86_64-linux-gnu'}"
    env["LD_LIBRARY_PATH"] = f"{libs}:{binary.parent}:/opt/rocm/lib"
    env["ROCP_METRICS"] = str(PROFILER_PREFIX / "lib/rocprofiler/metrics.xml")
    return env


def source_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, check=True, text=True,
        capture_output=True).stdout.strip()
    return {
        "repo": str(REPO_ROOT), "base_commit": commit,
        "state": "uncommitted_experimental",
        "source_path": str(SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "runner_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def inventory(directory: Path) -> list[dict[str, Any]]:
    return [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
             "sha256": sha256_file(path)}
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "receipt.json"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not PROFILER.is_file():
        raise RuntimeError(f"rocprofv2 unavailable: {PROFILER}")
    directory = Path(storage.assert_not_scratch(
        args.output_dir, what="gfx90a WGM diagnostic evidence directory"))
    directory.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    started_mono = time.monotonic()
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="AutoKernel gfx90a WGM launch-order diagnostic",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_wgm_sweep.py", timeout_s=args.claim_timeout_s,
        max_hold_s=2 * args.timeout_s + 300.0)
    opened = claim.receipt().to_dict()
    sampling = None
    sampler = None
    released = None
    error: BaseException | None = None
    payload: dict[str, Any] | None = None
    try:
        binary = directory / "autokernel_wgm_proxy"
        build = ["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx90a",
                 "-std=c++17", str(SOURCE), "-o", str(binary)]
        build_s = run_command(build, stdout=directory / "build.stdout.txt",
                              stderr=directory / "build.stderr.txt", timeout_s=args.timeout_s)
        sweep = [str(binary), "--rounds", str(args.rounds),
                 "--elements", str(args.elements)]
        # The device trace brackets the timing process, not compilation or the
        # subsequent profiler verification pass.
        sampler = device_sampler.RocmSmiSampler(
            device_index=0, interval_s=0.250).start()
        sweep_s = run_command(sweep, stdout=directory / "sweep.jsonl",
                              stderr=directory / "sweep.stderr.txt", timeout_s=args.timeout_s)
        sampling = sampler.stop().to_dict()
        sampler = None
        header, samples = parse_samples(directory / "sweep.jsonl", rounds=args.rounds)

        profile_dir = directory / "profile.raw"
        profile_dir.mkdir()
        profile = [str(PROFILER), "--kernel-trace", "--plugin", "file",
                   "--plugin-version", "2", "-d", str(profile_dir), "-o", "wgm",
                   str(binary), "--rounds", "1", "--elements", str(args.elements),
                   "--profile-once"]
        profile_s = run_command(
            profile, env=profiler_environment(binary),
            stdout=directory / "profile.stdout.txt",
            stderr=directory / "profile.stderr.txt", timeout_s=args.timeout_s)
        profile_files = [path for path in profile_dir.rglob("*") if path.is_file()]
        if not profile_files:
            raise RuntimeError("rocprofv2 produced no kernel-trace artifact")
        profile_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in profile_files)
        if "wgm_l2_proxy" not in profile_text:
            raise RuntimeError("kernel trace does not name wgm_l2_proxy dispatches")
        traced_factors = {
            int(value) for value in re.findall(r"wgm_l2_proxy<(\d+)>", profile_text)
        }
        if traced_factors != set(FACTORS):
            raise RuntimeError(
                f"kernel trace factors {sorted(traced_factors)} != {list(FACTORS)}")

        payload = {
            "schema": "epyc.autokernel.wgm_proxy_sweep.v1",
            "status": "pass", "authority": "diagnostic_only",
            "surface": "standalone_l2_tile_reuse_proxy_not_mmq",
            "campaign_id": args.campaign_id, "started_at": started_at,
            "ended_at": utc_now(), "duration_s": time.monotonic() - started_mono,
            "protocol": {
                "factors": list(FACTORS), "rounds": args.rounds,
                "balanced_order": "six-position cyclic rotation",
                "rows": 64, "cols": 64, "elements_per_row": args.elements,
                "kernel_body_change_between_factors": False,
                "chiplet_transform": "omitted_single_gcd_mi210",
                "correctness": header["correctness"],
            },
            "source": source_identity(),
            "toolchain": {"hipcc": "/opt/rocm/bin/hipcc",
                          "hipcc_sha256": sha256_file("/opt/rocm/bin/hipcc"),
                          "rocprofv2": str(PROFILER),
                          "rocprofv2_sha256": sha256_file(PROFILER)},
            "commands": {"build": build, "sweep": sweep, "profile": profile},
            "timing_s": {"build": build_s, "sweep": sweep_s, "profile": profile_s},
            "binary_sha256": sha256_file(binary),
            "raw_samples": {"path": "sweep.jsonl",
                            "sha256": sha256_file(directory / "sweep.jsonl")},
            "profile_artifacts": [{"path": str(path.relative_to(directory)),
                                   "sha256": sha256_file(path)} for path in profile_files],
            "profiled_factors": sorted(traced_factors),
            "result": summarize(samples), "device_claim": {"opened": opened},
        }
    except BaseException as exc:
        error = exc
    finally:
        try:
            if sampler is not None:
                sampling = sampler.stop().to_dict()
        except BaseException as exc:
            error = error or exc
        try:
            released = claim.release().to_dict()
        except BaseException as exc:
            error = error or exc
    if error is not None:
        raise error
    assert payload is not None and sampling is not None and released is not None
    payload["device_sampling"] = sampling
    payload["device_claim"]["released"] = released
    if not released.get("released_at"):
        raise RuntimeError("device claim release receipt lacks released_at")
    payload["artifacts"] = inventory(directory)
    receipt = directory / "receipt.json"
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--campaign-id", default="inf36-wgm-gfx90a-20260811")
    parser.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    parser.add_argument("--claim-timeout-s", type=float, default=300.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--rounds", type=int, default=48)
    parser.add_argument("--elements", type=int, default=131072)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
