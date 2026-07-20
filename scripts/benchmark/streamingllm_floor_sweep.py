#!/usr/bin/env python3
"""Run a bounded StreamingLLM KV-retention floor sweep.

This is a research runner for the experimental v7 llama.cpp worktree. It runs
llama-completion directly, hides accelerator devices for CPU-only sweeps, and
records observation-grade artifacts for deciding whether any sink/window cluster
is worth admitting to a larger quality gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
LLAMA_EXPERIMENTAL = Path("/mnt/raid0/llm/llama.cpp-experimental")
DEFAULT_BINARY = LLAMA_EXPERIMENTAL / "build-hip" / "bin" / "llama-completion"
DEFAULT_MODEL = Path("/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf")
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "data" / "streamingllm_floor"
MAX_CAPTURE_BYTES = 2 * 1024 * 1024

PROMPT = """You are running a deterministic long-output KV-retention check.

Keep these anchors alive throughout the answer:
- ANCHOR_ALPHA_17
- ANCHOR_BRAVO_29
- ANCHOR_CHARLIE_41

Write numbered audit lines from 001 upward. Every line must contain
ANCHOR_ALPHA_17 and the word retained. Every tenth line must also contain
ANCHOR_BRAVO_29. The final line must be exactly:
STREAMINGLLM_DONE ANCHOR_ALPHA_17 ANCHOR_BRAVO_29 ANCHOR_CHARLIE_41

Begin now.
"""


@dataclass(frozen=True)
class ArmSpec:
    name: str
    sink: int
    window: int

    @property
    def streaming_enabled(self) -> bool:
        return self.sink > 0 or self.window > 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run live commands")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--context", type=int, default=384)
    parser.add_argument("--tokens", type=int, default=640)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--cluster",
        action="append",
        default=[],
        metavar="SINK:WINDOW",
        help="StreamingLLM sink/window pair. May be repeated.",
    )
    args = parser.parse_args(argv)
    if args.context < 96:
        raise ValueError("--context must be at least 96")
    if args.tokens <= args.context:
        raise ValueError("--tokens must exceed --context so context shift is exercised")
    if str(args.binary).startswith("/mnt/raid0/llm/llama.cpp/"):
        raise ValueError("refusing production v6 binary; use llama.cpp-experimental")
    return args


def parse_cluster(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+):(\d+)", value.strip())
    if not match:
        raise ValueError(f"invalid cluster {value!r}; expected SINK:WINDOW")
    sink = int(match.group(1))
    window = int(match.group(2))
    if sink == 0 and window == 0:
        raise ValueError("streaming clusters must have non-zero sink or window")
    return sink, window


def build_arm_specs(args: argparse.Namespace) -> list[ArmSpec]:
    clusters = [parse_cluster(value) for value in args.cluster]
    if not clusters:
        clusters = [(8, 128), (16, 192), (32, 256)]
    arms = [ArmSpec("baseline_context_shift", 0, 0)]
    for sink, window in clusters:
        arms.append(ArmSpec(f"streaming_sink{sink}_window{window}", sink, window))
    return arms


def command_for(args: argparse.Namespace, arm: ArmSpec, prompt_path: Path) -> list[str]:
    cmd = [
        str(args.binary),
        "-m",
        str(args.model),
        "-f",
        str(prompt_path.resolve()),
        "-c",
        str(args.context),
        "-n",
        str(args.tokens),
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads),
        "-ngl",
        "0",
        "--device",
        "none",
        "--no-kv-offload",
        "--no-op-offload",
        "--keep",
        "0",
        "--context-shift",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--seed",
        str(args.seed),
        "--no-warmup",
        "--no-display-prompt",
        "--color",
        "off",
        "--verbose-prompt",
        "--perf",
        "-no-cnv",
        "--reasoning",
        "off",
        "--reasoning-format",
        "none",
    ]
    if arm.streaming_enabled:
        cmd += ["--kv-streaming-sink", str(arm.sink), "--kv-streaming-window", str(arm.window)]
    return cmd


def run_env() -> dict[str, str]:
    env = os.environ.copy()
    lib_path = str(LLAMA_EXPERIMENTAL / "build-hip" / "bin")
    env["LD_LIBRARY_PATH"] = f"{lib_path}:/opt/rocm/lib"
    env["HIP_VISIBLE_DEVICES"] = "-1"
    env["ROCR_VISIBLE_DEVICES"] = "-1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_capture(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def parse_perf(stderr: str) -> dict[str, Any]:
    def find(pattern: str, cast: Any = float) -> Any:
        match = re.search(pattern, stderr)
        if not match:
            return None
        return cast(match.group(1))

    return {
        "prompt_tokens": find(r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens", int),
        "prompt_tps": find(r"prompt eval time =.*?,\s+([\d.]+) tokens per second\)"),
        "decode_runs": find(r"common_perf_print:\s+eval time =\s+[\d.]+ ms /\s+(\d+) runs", int),
        "decode_tps": find(r"common_perf_print:\s+eval time =.*?,\s+([\d.]+) tokens per second\)"),
        "total_tokens": find(r"total time =\s+[\d.]+ ms /\s+(\d+) tokens", int),
        "max_rss_kib": find(r"Maximum resident set size \(kbytes\):\s+(\d+)", int),
        "exit_status_reported": find(r"Exit status:\s+(\d+)", int),
    }


def score_output(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    final_marker = "STREAMINGLLM_DONE ANCHOR_ALPHA_17 ANCHOR_BRAVO_29 ANCHOR_CHARLIE_41"
    alpha_count = stdout.count("ANCHOR_ALPHA_17")
    bravo_count = stdout.count("ANCHOR_BRAVO_29")
    charlie_count = stdout.count("ANCHOR_CHARLIE_41")
    numbered = sum(1 for line in lines if re.match(r"^\d{1,4}(?:[.)]|\s)", line))
    return {
        "pass": final_marker in stdout and alpha_count >= 20 and numbered >= 20,
        "final_marker_present": final_marker in stdout,
        "alpha_count": alpha_count,
        "bravo_count": bravo_count,
        "charlie_count": charlie_count,
        "numbered_line_count": numbered,
        "line_count": len(lines),
    }


def run_arm(args: argparse.Namespace, arm: ArmSpec, prompt_path: Path, output_dir: Path) -> dict[str, Any]:
    cmd = command_for(args, arm, prompt_path)
    arm_dir = output_dir / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(LLAMA_EXPERIMENTAL),
            env=run_env(),
            text=True,
            capture_output=True,
            timeout=args.timeout_sec,
            check=False,
        )
        timed_out = False
        stdout = proc.stdout
        stderr = proc.stderr
        return_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return_code = 124
    wall = time.monotonic() - started
    if len(stdout.encode("utf-8", errors="replace")) > MAX_CAPTURE_BYTES:
        stdout = stdout[:MAX_CAPTURE_BYTES] + "\n[stdout truncated by runner]\n"
    if len(stderr.encode("utf-8", errors="replace")) > MAX_CAPTURE_BYTES:
        stderr = stderr[:MAX_CAPTURE_BYTES] + "\n[stderr truncated by runner]\n"
    stdout = normalize_capture(stdout)
    stderr = normalize_capture(stderr)
    (arm_dir / "stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
    (arm_dir / "stderr.txt").write_text(stderr, encoding="utf-8", errors="replace")
    result = {
        "arm": arm.name,
        "sink": arm.sink,
        "window": arm.window,
        "streaming_enabled": arm.streaming_enabled,
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_time_s": wall,
        "stdout_sha256": sha256_text(stdout),
        "stderr_sha256": sha256_text(stderr),
        "perf": parse_perf(stderr),
        "quality": score_output(stdout),
    }
    (arm_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_summary(output_dir: Path, args: argparse.Namespace, arms: list[ArmSpec], results: list[dict[str, Any]]) -> None:
    baseline = next((row for row in results if row["arm"] == "baseline_context_shift"), None)
    baseline_tps = None
    if baseline:
        baseline_tps = baseline.get("perf", {}).get("decode_tps")
    rows = []
    for result in results:
        decode_tps = result.get("perf", {}).get("decode_tps")
        ratio = None
        if baseline_tps and decode_tps:
            ratio = decode_tps / baseline_tps
        rows.append(
            {
                "arm": result["arm"],
                "sink": result["sink"],
                "window": result["window"],
                "return_code": result["return_code"],
                "decode_tps": decode_tps,
                "decode_ratio_vs_baseline": ratio,
                "quality_pass": result["quality"]["pass"],
                "final_marker_present": result["quality"]["final_marker_present"],
                "alpha_count": result["quality"]["alpha_count"],
                "numbered_line_count": result["quality"]["numbered_line_count"],
            }
        )

    candidate_rows = [
        row
        for row in rows
        if row["arm"] != "baseline_context_shift"
        and row["return_code"] == 0
        and row["quality_pass"]
        and (row["decode_ratio_vs_baseline"] or 0.0) >= 0.98
    ]
    summary = {
        "mode": "execute" if results else "dry_run",
        "decision_grade": False,
        "observation_grade": bool(results),
        "admission_decision": {
            "admit_cluster": bool(candidate_rows),
            "reason": "quality_and_floor_candidate_found" if candidate_rows else "no_streaming_cluster_passed_quality_and_speed_floor",
            "candidate_rows": candidate_rows,
        },
        "binary": str(args.binary),
        "model": str(args.model),
        "context": args.context,
        "tokens": args.tokens,
        "threads": args.threads,
        "seed": args.seed,
        "arms": [arm.__dict__ for arm in arms],
        "rows": rows,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# StreamingLLM Floor Sweep",
        "",
        f"Mode: `{summary['mode']}`",
        f"Model: `{args.model}`",
        f"Binary: `{args.binary}`",
        f"Context/tokens: `{args.context}` / `{args.tokens}`",
        "",
        "| Arm | sink | window | rc | decode t/s | ratio | quality | marker | alpha | numbered |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        ratio = "" if row["decode_ratio_vs_baseline"] is None else f"{row['decode_ratio_vs_baseline']:.3f}"
        tps = "" if row["decode_tps"] is None else f"{row['decode_tps']:.2f}"
        lines.append(
            "| {arm} | {sink} | {window} | {return_code} | {tps} | {ratio} | {quality} | {marker} | {alpha} | {numbered} |".format(
                arm=row["arm"],
                sink=row["sink"],
                window=row["window"],
                return_code=row["return_code"],
                tps=tps,
                ratio=ratio,
                quality="pass" if row["quality_pass"] else "fail",
                marker="yes" if row["final_marker_present"] else "no",
                alpha=row["alpha_count"],
                numbered=row["numbered_line_count"],
            )
        )
    lines += [
        "",
        "Admission decision:",
        json.dumps(summary["admission_decision"], indent=2),
        "",
        "This artifact is observation-grade only. It is not a production serving or P-GPU-1 claim.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"streamingllm_floor_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(PROMPT, encoding="utf-8")

    arms = build_arm_specs(args)
    manifest = {
        "binary": str(args.binary),
        "model": str(args.model),
        "context": args.context,
        "tokens": args.tokens,
        "threads": args.threads,
        "timeout_sec": args.timeout_sec,
        "arms": [arm.__dict__ for arm in arms],
        "commands": {arm.name: command_for(args, arm, prompt_path) for arm in arms},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        write_summary(output_dir, args, arms, [])
        print(output_dir)
        return 0

    if not args.binary.exists():
        raise FileNotFoundError(args.binary)
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    results = [run_arm(args, arm, prompt_path, output_dir) for arm in arms]
    write_summary(output_dir, args, arms, results)
    print(output_dir)
    return 0 if all(result["return_code"] == 0 for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
