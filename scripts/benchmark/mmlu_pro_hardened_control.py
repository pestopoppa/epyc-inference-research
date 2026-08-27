#!/usr/bin/env python3
"""Run one pinned MMLU-Pro GPU arm under the exclusive MI210 claim."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "kernel_rnd"))

from autokernel.resource.device_claim import ClaimJournal, acquire_device_claim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--belief-category", choices=["BASELINE", "CANDIDATE"],
                        default=None,
                        help="SC32 belief category for THIS arm: BASELINE for the "
                             "anchor arm, CANDIDATE for controls. When set, the "
                             "runner emits producer-authored belief_measurements "
                             "rows at result-finalize; absent = zero rows "
                             "(pre-hook behavior).")
    parser.add_argument("--belief-config", default="",
                        help="Optional JSON with server-side facts the runner cannot "
                             "observe (template, quant detail); merged into the "
                             "belief row's extra.arm_config")
    args = parser.parse_args()

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    journal = ClaimJournal(args.artifact_root / "device_claim_journal.jsonl")
    claim = acquire_device_claim(
        "mi210_0",
        purpose=f"MMLU-Pro hardened control {args.arm}",
        campaign_id="mmlu-pro-hardened-control-20260812",
        journal=journal,
        holder_label=f"mmlu-pro-{args.arm}",
        timeout_s=None,
        max_hold_s=3 * 60 * 60,
    )
    (args.artifact_root / f"{args.arm}.claim_open.json").write_text(
        json.dumps(claim.receipt().to_dict(), sort_keys=True) + "\n"
    )

    env = os.environ.copy()
    env.update({
        "GPU_BENCH_ART": str(args.artifact_root),
        "GPU_BENCH_RES": str(REPO),
        "GPU_BENCH_PORT": "18072",
        "KERNEL_LABEL": "production-consolidated-v9",
        "GPU_BENCH_BIN": "/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server",
        "HF_HOME": "/mnt/raid0/llm/cache/huggingface",
        "HF_DATASETS_OFFLINE": "1",
        "GPU_BENCH_BELIEF_CATEGORY": args.belief_category or "",
        "GPU_BENCH_BELIEF_CONFIG": args.belief_config,
    })
    cmd = [
        str(HERE / "architect_bench_gpu_arm.sh"), args.arm, args.model,
        args.spec, "mmlu_pro", "150", str(args.max_tokens), "1",
    ]
    child: subprocess.Popen[str] | None = None
    pending_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal pending_signal
        pending_signal = signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        child = subprocess.Popen(cmd, env=env, start_new_session=True, text=True)
        rc = child.wait()
        return 128 + pending_signal if pending_signal is not None else rc
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        released = claim.release()
        (args.artifact_root / f"{args.arm}.claim_released.json").write_text(
            json.dumps(released.to_dict(), sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
