#!/usr/bin/env python3
"""Run the HipKittens LDS bank/phase method on gfx90a for AutoKernel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.autokernel_claimed_sampling import (
    error_payload,
    stop_sampler_and_release,
)
from scripts.kernel_rnd.autokernel import hipkittens_lds as lds
from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


PROFILER_ROOT = Path("/mnt/raid0/llm/tools/rocm-profilers-6.2")
PROFILER_PREFIX = PROFILER_ROOT / "opt" / "rocm-6.2.0"
PROFILER = PROFILER_PREFIX / "bin" / "rocprofv2"
PROBE_SOURCE = REPO_ROOT / "scripts/benchmark/autokernel_lds_probe.cpp"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def assert_source_identity(expected_commit: str | None) -> str:
    observed = git_output("rev-parse", "HEAD")
    if expected_commit is not None and observed != expected_commit:
        raise RuntimeError(
            f"source commit mismatch: expected {expected_commit}, observed {observed}")
    dirty = git_output("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"research source tree is dirty: {dirty.splitlines()[:8]}")
    return observed


def profiler_environment(binary: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_PATH"] = "/opt/rocm"
    env["PATH"] = f"{PROFILER_PREFIX / 'bin'}:/opt/rocm/bin:{env.get('PATH', '')}"
    profiler_libs = (
        f"{PROFILER_PREFIX / 'lib'}:"
        f"{PROFILER_ROOT / 'usr/lib/x86_64-linux-gnu'}")
    env["LD_LIBRARY_PATH"] = f"{profiler_libs}:{binary.parent}:/opt/rocm/lib"
    env["ROCP_METRICS"] = str(PROFILER_PREFIX / "lib/rocprofiler/metrics.xml")
    return env


def run_command(command: Sequence[str], *, env: dict[str, str] | None,
                stdout: Path, stderr: Path, timeout_s: float) -> float:
    started = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            tuple(command), env=env, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, start_new_session=True)
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
    if returncode != 0:
        tail = stderr.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"command exited {returncode}: {tail!r}")
    return time.monotonic() - started


def build_probe(output_dir: Path, *, timeout_s: float) -> tuple[Path, list[str], float]:
    binary = output_dir / "autokernel_lds_probe"
    command = [
        "/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx90a",
        "-std=c++17", str(PROBE_SOURCE), "-o", str(binary),
    ]
    duration = run_command(
        command, env=None, stdout=output_dir / "build.stdout.txt",
        stderr=output_dir / "build.stderr.txt", timeout_s=timeout_s)
    if not binary.is_file():
        raise RuntimeError("hipcc succeeded without producing the LDS probe")
    return binary, command, duration


def select_counter_csv(raw_dir: Path) -> Path:
    matches = []
    for candidate in sorted(raw_dir.rglob("*.csv")):
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if lds.TARGET_KERNEL in text and "SQ_LDS_BANK_CONFLICT" in text:
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one counter CSV below {raw_dir}, found {matches}")
    return matches[0]


def profile(binary: Path, *, kind: str, probe_args: Sequence[str],
            output_dir: Path, timeout_s: float
            ) -> tuple[Path, tuple[lds.CounterSample, ...], list[str], float]:
    raw_dir = output_dir / f"{kind}.raw"
    raw_dir.mkdir()
    counters = output_dir / f"{kind}.counters.txt"
    counters.write_text(
        "pmc: SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT\n", encoding="utf-8")
    command = [
        str(PROFILER), "-i", str(counters), "--plugin", "file",
        "--plugin-version", "2", "-d", str(raw_dir), "-o", kind,
        str(binary), kind, *probe_args,
    ]
    duration = run_command(
        command, env=profiler_environment(binary),
        stdout=output_dir / f"{kind}.stdout.txt",
        stderr=output_dir / f"{kind}.stderr.txt", timeout_s=timeout_s)
    counter_csv = select_counter_csv(raw_dir)
    digest = lds.sha256_file(counter_csv)
    samples = lds.load_counter_samples(counter_csv, expected_sha256=digest)
    return counter_csv, samples, command, duration


def inventory(output_dir: Path) -> list[dict[str, Any]]:
    return [{
        "path": str(path.relative_to(output_dir)),
        "bytes": path.stat().st_size,
        "sha256": lds.sha256_file(path),
    } for path in sorted(output_dir.rglob("*"))
      if path.is_file() and path.name != "receipt.json"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_commit = assert_source_identity(args.source_commit)
    if not PROFILER.is_file():
        raise RuntimeError(f"rocprofv2 is unavailable: {PROFILER}")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="gfx90a LDS topology evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    started_mono = time.monotonic()
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="AutoKernel gfx90a LDS bank/phase solver",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_lds_solver.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=2 * args.profile_timeout_s + args.build_timeout_s + 300.0)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling = None
    released = None
    captured_error: BaseException | None = None
    payload: dict[str, Any] | None = None
    try:
        sampler = device_sampler.RocmSmiSampler(
            device_index=0, interval_s=0.250).start()
        binary, build_command, build_duration = build_probe(
            output_dir, timeout_s=args.build_timeout_s)
        bank_lanes = (0, 1)
        bank_csv, bank_samples, bank_command, bank_duration = profile(
            binary, kind="bank", probe_args=(
                str(args.max_bank), str(args.bank_repetitions),
                str(bank_lanes[0] + 1), str(bank_lanes[1] + 1)),
            output_dir=output_dir, timeout_s=args.profile_timeout_s)
        bank_plan = lds.bank_cases(
            max_bank=args.max_bank, repetitions=args.bank_repetitions)
        bank_solution = lds.solve_bank_count(bank_plan, bank_samples)
        phase_csv, phase_samples, phase_command, phase_duration = profile(
            binary, kind="phase", probe_args=(
                str(bank_solution.bank_count), str(args.phase_repetitions)),
            output_dir=output_dir, timeout_s=args.profile_timeout_s)
        phase_plan = lds.phase_cases(repetitions=args.phase_repetitions)
        phase_solution = lds.solve_phases(phase_plan, phase_samples)
        if not any(set(bank_lanes).issubset(row) for row in phase_solution.groups):
            raise RuntimeError(
                "bank-solver lanes did not validate as same-phase in all-pairs capture")
        matches_cdna3 = (
            bank_solution.bank_count == 64
            and len(phase_solution.groups) == 2
            and sorted(len(row) for row in phase_solution.groups) == [32, 32]
        )
        payload = {
            "schema": lds.SCHEMA,
            "status": "pass",
            "authority": "diagnostic_only",
            "campaign_id": args.campaign_id,
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_s": time.monotonic() - started_mono,
            "target_arch": lds.ARCH,
            "instruction": "ds_read_b128",
            "source": {
                "repo": str(REPO_ROOT), "commit": source_commit,
                "probe_path": str(PROBE_SOURCE.relative_to(REPO_ROOT)),
                "probe_sha256": lds.sha256_file(PROBE_SOURCE),
                "upstream_repo": lds.UPSTREAM_REPO,
                "upstream_commit": lds.UPSTREAM_COMMIT,
                "upstream_method": lds.UPSTREAM_METHOD,
                "adaptation": "batched standalone HIP probe; no framework vendoring",
            },
            "toolchain": {
                "hipcc": "/opt/rocm/bin/hipcc",
                "hipcc_sha256": lds.sha256_file("/opt/rocm/bin/hipcc"),
                "rocprofv2": str(PROFILER),
                "rocprofv2_sha256": lds.sha256_file(PROFILER),
                "counter_file": "pmc: SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT",
            },
            "commands": {
                "build": build_command, "bank": bank_command, "phase": phase_command,
            },
            "timing_s": {
                "build": build_duration, "bank": bank_duration, "phase": phase_duration,
            },
            "binary": str(binary),
            "binary_sha256": lds.sha256_file(binary),
            "bank_capture": {
                "path": str(bank_csv), "sha256": lds.sha256_file(bank_csv),
                "summary": lds.summarize_samples(bank_samples),
                "repetitions": args.bank_repetitions,
            },
            "bank_solution": {
                "bank_count": bank_solution.bank_count,
                "same_phase_probe_lanes": list(bank_lanes),
                "tested_bases": list(bank_solution.tested_bases),
                "conflict_bases": list(bank_solution.conflict_bases),
                "candidate_mismatches": {
                    str(key): value
                    for key, value in bank_solution.candidate_mismatches.items()
                },
            },
            "phase_capture": {
                "path": str(phase_csv), "sha256": lds.sha256_file(phase_csv),
                "summary": lds.summarize_samples(phase_samples),
                "repetitions": args.phase_repetitions,
            },
            "phase_solution": {
                "phase_count": phase_solution.phase_count,
                "groups": [list(row) for row in phase_solution.groups],
                "tested_pairs": phase_solution.tested_pairs,
            },
            "swizzle_transfer_class": (
                "topology_matches_cdna3" if matches_cdna3 else "retune_required"),
            "interpretation": (
                "Topology compatibility is a design prior only; it does not establish "
                "correctness or performance of a HipKittens swizzle in llama.cpp."),
            "device_claim": {"opened": opened},
        }
    except BaseException as exc:
        captured_error = exc
    finally:
        sampling, released_receipt, teardown_errors = stop_sampler_and_release(
            sampler=sampler, claim=claim)
        released = released_receipt.to_dict() if released_receipt is not None else None
    if teardown_errors and captured_error is None:
        captured_error = teardown_errors[0]
    if payload is None:
        payload = {
            "schema": lds.SCHEMA, "status": "failed",
            "authority": "diagnostic_only", "campaign_id": args.campaign_id,
            "started_at": started_at, "ended_at": utc_now(),
            "duration_s": time.monotonic() - started_mono,
            "target_arch": lds.ARCH,
            "source": {"repo": str(REPO_ROOT), "commit": source_commit},
            "failure": repr(captured_error),
            "device_claim": {"opened": opened},
        }
    if sampling is not None:
        payload["device_sampling"] = sampling.to_dict()
    payload["device_claim"]["released"] = released
    payload["teardown_errors"] = error_payload(teardown_errors)
    payload["artifacts"] = inventory(output_dir)
    write_json_atomic(output_dir / "receipt.json", payload)
    if captured_error is not None:
        raise captured_error
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Empirically solve gfx90a LDS banks/phases with rocprofv2")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", required=True)
    result.add_argument("--source-commit")
    result.add_argument("--max-bank", type=int, default=127)
    result.add_argument("--bank-repetitions", type=int, default=3)
    result.add_argument("--phase-repetitions", type=int, default=3)
    result.add_argument("--build-timeout-s", type=float, default=300.0)
    result.add_argument("--profile-timeout-s", type=float, default=3600.0)
    result.add_argument("--claim-timeout-s", type=float, default=300.0)
    result.add_argument(
        "--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "status": payload["status"],
        "receipt": str(Path(args.output_dir).resolve() / "receipt.json"),
        "bank_count": payload["bank_solution"]["bank_count"],
        "phase_count": payload["phase_solution"]["phase_count"],
        "swizzle_transfer_class": payload["swizzle_transfer_class"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
