#!/usr/bin/env python3
"""Phase-3 shadow bake-off runner (P3-1): orchestrate, capture, defer scoring.

PLAN-ONLY BY DEFAULT.  Execution requires ``--execute --i-have-operator-grant``
(repo convention) and happens only in operator bench windows.

What this runner does:
- Verifies every manifest-pinned source hash (fail-closed).
- Emits/executes the exact ``v7_quality_gate_runner.py`` invocations per
  (arm, duty, suite) with the manifest's pinned questions + sampling params.
- Captures raw responses per-question incrementally via the existing
  ``v7_quality_gate_capture.v4`` path (the child runner persists per-row JSONL
  and a live-status sidecar); scoring is DEFERRED to the existing
  deterministic replay machinery (SWE converter+harness, critic replay
  scorer).  No new scorer is introduced for the coder duty.

What this runner NEVER does:
- Launch, stop, or manage servers.  The GPU shadow lane (role
  ``coder_escalation_shadow``, port 18100) is assumed already up via the
  operator-gated Steps 0-7 choreography in gpu-shadow-lane.md.  The A4
  control arm's server is likewise operator-window managed.
- Touch live /chat routing (D3: eval path only).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_bakeoff_common import CRITIC_SUITE, write_json  # noqa: E402
from p3_bakeoff_manifest import verify_manifest  # noqa: E402

RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
DEFAULT_OUT_ROOT = RESEARCH_ROOT / "artifacts/p3-shadow-bakeoff-20260728/runs"
DUTIES = ("coder", "cocritic")


def suite_jobs(manifest: dict, duty: str) -> list[dict]:
    """Expand a duty into per-suite job specs from the pinned manifest."""
    jobs = []
    if duty == "coder":
        for suite, spec in manifest["duties"]["coder"]["suites"].items():
            jobs.append({
                "suite": suite,
                "questions_file": spec["questions_file"]["path"],
                "n": spec["n"],
                "max_tokens": spec["max_tokens"],
                "request_timeout_s": spec["request_timeout_s"],
                "sampling": spec["sampling"],
            })
    elif duty == "cocritic":
        spec = manifest["duties"]["cocritic"]
        jobs.append({
            "suite": spec["suite"],
            "questions_file": spec["tasks_file"]["path"],
            "n": spec["n"],
            "max_tokens": spec["max_tokens"],
            "request_timeout_s": spec["request_timeout_s"],
            "sampling": spec["sampling"],
        })
    else:
        raise ValueError(f"unknown duty: {duty}")
    return jobs


def build_command(job: dict, *, arm_key: str, arm: dict, duty: str,
                  host: str, port: int, out_dir: Path,
                  runner_path: str) -> dict:
    s = job["sampling"]
    argv = [
        sys.executable, runner_path,
        "--host", host, "--port", str(port),
        "--suites", job["suite"],
        "--n", str(job["n"]), "--limit", str(job["n"]),
        "--seed", str(s["seed"]),
        "--max-tokens", str(job["max_tokens"]),
        "--repeats", str(s["repeats"]),
        "--concurrency", str(s["concurrency"]),
        "--temperature", str(s["temperature"]),
        "--top-p", str(s["top_p"]),
        "--top-k", str(s["top_k"]),
        "--endpoint", s["endpoint"],
        "--arm", f"p3_{arm_key}_{duty}_{job['suite']}",
        "--binary", "gpu-shadow-lane-v8-10107",
        "--models", arm["model_path"],
        "--questions-in", job["questions_file"],
        "--per-question-out", str(out_dir / "pq.jsonl"),
        "--output", str(out_dir / "r.json"),
    ]
    if not s["enable_thinking"]:
        argv.append("--no-enable-thinking")
    return {
        "duty": duty,
        "suite": job["suite"],
        "arm": arm_key,
        "out_dir": str(out_dir),
        "env": {
            "RUNNER_REQUEST_TIMEOUT_S": str(job["request_timeout_s"]),
        },
        "argv": argv,
        "shell": (
            f"RUNNER_REQUEST_TIMEOUT_S={job['request_timeout_s']} "
            + " ".join(shlex.quote(a) for a in argv)
        ),
        "watchdog": (
            f"{sys.executable} scripts/benchmark/capture_integrity_watchdog.py "
            f"--watch --stale-timeout-s 1800 {out_dir / 'pq.live-status.json'}"
        ),
    }


def post_run_pointers(manifest: dict, plan: list[dict]) -> list[str]:
    """Deterministic follow-up commands (scoring/reporting), never auto-run."""
    pointers = []
    for cmd in plan:
        d = Path(cmd["out_dir"])
        if cmd["suite"] == "swebench_oracle":
            conv = manifest["duties"]["coder"]["suites"]["swebench_oracle"]["scorer"]["converter"]["path"]
            pointers.append(
                f"python3 {conv} {d / 'pq.jsonl'} p3_{cmd['arm']} "
                f"{d / 'predictions.json'}  # then the pinned swebench harness"
            )
        elif cmd["suite"] == CRITIC_SUITE:
            tasks = manifest["duties"]["cocritic"]["tasks_file"]["path"]
            pointers.append(
                f"{sys.executable} scripts/benchmark/p3_bakeoff_critic_score.py "
                f"--tasks {tasks} --capture {d / 'pq.jsonl'} "
                f"--arm {cmd['arm']} --output {d / 'critic_score.json'}"
            )
    pointers.append(
        f"{sys.executable} scripts/benchmark/p3_bakeoff_report.py --help  "
        "# paired report once both arms of a duty are captured"
    )
    return pointers


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--arm", required=True,
                   help="Arm key from the manifest (stock27b|ff27b|a4_control)")
    p.add_argument("--duty", choices=[*DUTIES, "all"], default="all")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18100,
                   help="Assumed-up eval endpoint (shadow lane default 18100)")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run-id", default=None,
                   help="Run directory name (default: UTC timestamp)")
    p.add_argument("--execute", action="store_true",
                   help="Actually run the captures (operator window only)")
    p.add_argument("--i-have-operator-grant", action="store_true",
                   help="Attest an explicit operator grant for this window")
    args = p.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    if args.arm not in manifest["arms"]:
        print(f"[bakeoff] unknown arm {args.arm!r}; manifest arms: "
              f"{sorted(manifest['arms'])}", file=sys.stderr)
        return 2
    arm = manifest["arms"][args.arm]

    failures = verify_manifest(manifest)
    if failures:
        for f in failures:
            print(f"[bakeoff] PIN-FAIL {f}", file=sys.stderr)
        print("[bakeoff] refusing: pinned sources drifted (pairing discipline)",
              file=sys.stderr)
        return 1

    duties = list(DUTIES) if args.duty == "all" else [args.duty]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan: list[dict] = []
    runner_path = manifest["capture"]["runner"]["path"]
    for duty in duties:
        for job in suite_jobs(manifest, duty):
            out_dir = args.out_root / run_id / args.arm / duty / job["suite"]
            plan.append(build_command(
                job, arm_key=args.arm, arm=arm, duty=duty,
                host=args.host, port=args.port, out_dir=out_dir,
                runner_path=runner_path,
            ))

    plan_doc = {
        "mode": "execute" if args.execute else "plan_only",
        "run_id": run_id,
        "arm": args.arm,
        "manifest": str(args.manifest),
        "invariants": manifest["invariants"],
        "server_assumption": (
            f"eval endpoint {args.host}:{args.port} already up via the "
            "operator-gated lane choreography; this runner NEVER launches or "
            "manages servers"
        ),
        "commands": plan,
        "post_run": post_run_pointers(manifest, plan),
    }

    if not args.execute:
        print(json.dumps(plan_doc, indent=2))
        print(f"\n[bakeoff] PLAN ONLY ({len(plan)} capture job(s)). To execute "
              "in an operator window: add --execute --i-have-operator-grant",
              file=sys.stderr)
        return 0

    if not args.i_have_operator_grant:
        print("[bakeoff] refusing --execute without --i-have-operator-grant "
              "(inference requires an explicit operator window grant)",
              file=sys.stderr)
        return 1
    if not arm.get("sha256"):
        print(f"[bakeoff] refusing: arm {args.arm} has no pinned model sha256 "
              "in the manifest", file=sys.stderr)
        return 1

    run_dir = args.out_root / run_id / args.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "plan.json", plan_doc, sort_keys=False)
    rc_all = 0
    for cmd in plan:
        Path(cmd["out_dir"]).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(cmd["env"])
        print(f"[bakeoff] RUN {cmd['arm']}/{cmd['duty']}/{cmd['suite']}",
              file=sys.stderr)
        rc = subprocess.run(cmd["argv"], cwd=RESEARCH_ROOT, env=env).returncode
        if rc != 0:
            # Per-question captures are already durable; continue to the next
            # suite so a partial window still banks evidence.
            print(f"[bakeoff] job failed rc={rc}: {cmd['suite']} "
                  "(captures persisted; resume is idempotent)", file=sys.stderr)
            rc_all = rc
    print("[bakeoff] window complete. Next: post_run commands in plan.json",
          file=sys.stderr)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
