#!/usr/bin/env python3
"""DR-3d frontdoor opportunity-cost gate.

This dry-run-first harness measures the serving cost of temporarily leasing the
MI210 away from the resident frontdoor lane for the DR-3 quant-asymmetric K2
lane. It does not add a serving route, NumericSwarm surface, or production-stack
configuration.
"""

from __future__ import annotations

import argparse
import json
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from scripts.benchmark import dr0_quant_asym_self_spec_runner as dr0
from scripts.benchmark import dr3_quant_asym_k2_admission_runner as dr3

import k35_stack_context_matrix_runner as k35


SCHEMA = "epyc.dr3_frontdoor_opportunity_cost_gate.v1"
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "dr3_frontdoor_opportunity_cost"
    / f"dr3_frontdoor_opportunity_cost_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
)
DEFAULT_FRONTDOOR_SCENARIO = "frontdoor_gpu_resident_no_spec"
DEFAULT_CONTEXTS = (8192,)
DEFAULT_PORT_BASE = 22420
DEFAULT_FRONTDOOR_MAX_TOKENS = 512
DEFAULT_MIN_COMPLETION_TOKENS = 128
DEFAULT_DR3_TASK_CLASS = "long_repetitive_output"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def selected_contexts(values: list[int] | None) -> list[int]:
    return values or list(DEFAULT_CONTEXTS)


def launch_argv(argv: list[str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
    finally:
        stderr.close()


def terminate_process(proc: subprocess.Popen[str], *, port: int | None = None, timeout_s: int = 20) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": proc.pid, "terminated": False, "sigkill_sent": False}
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
            result["terminated"] = True
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)
            result["sigkill_sent"] = True
    result["returncode"] = proc.returncode
    ps = k35.run_capture(["ps", "-p", str(proc.pid), "-o", "pid=,comm=,args="], timeout=10)
    result["ps_after"] = ps
    result["dead"] = str(proc.pid) not in ps.get("stdout", "")
    result["port_open_after"] = dr0.port_is_open(port) if port is not None else None
    result["status"] = "ok" if result["dead"] and not result["port_open_after"] else "fail"
    return result


def frontdoor_argv(args: argparse.Namespace, port: int) -> list[str]:
    scenario = k35.scenario_by_name(args.frontdoor_scenario)
    return k35.build_server_argv(
        scenario,
        binary=args.binary,
        port=port,
        nominal_context=max(args.contexts),
        max_tokens=args.frontdoor_max_tokens,
    )


def build_dr3_arm_spec(args: argparse.Namespace, port: int) -> dr3.ArmSpec:
    compat = argparse.Namespace(
        binary=args.binary,
        cpu_verifier_model=args.cpu_verifier_model,
        mi210_drafter_model=args.mi210_drafter_model,
        context=args.dr3_context,
        threads=args.dr3_threads,
        ubatch=args.dr3_ubatch,
        spec_draft_n_max=dr3.K_VALUE,
    )
    arm = dr3.dr0_arm_by_id("quant_asymmetric_combined")
    return dr3.ArmSpec(
        id=f"dr3_combined_k2_ctx{args.dr3_context}",
        base_arm_id=arm.id,
        role=arm.role,
        context=args.dr3_context,
        k=dr3.K_VALUE,
        port=port,
        env=dr0.arm_env(arm),
        argv=dr0.arm_argv(compat, arm, port, spec_draft_n_max=dr3.K_VALUE),
    )


def dr3_task_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    task_args = argparse.Namespace(
        rows_per_class=1,
        max_tokens=args.dr3_max_tokens,
        seed=args.seed,
        context_fill_chars_per_token=args.context_fill_chars_per_token,
        max_context_fill_chars=args.max_context_fill_chars,
    )
    rows = dr3.materialize_task_rows(task_args, args.dr3_context)
    selected = [row for row in rows if row["class_id"] == args.dr3_task_class]
    if not selected:
        raise ValueError(f"unknown DR-3 task class: {args.dr3_task_class}")
    return selected


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    frontdoor_before_port = args.port_base
    dr3_port = args.port_base + 1
    frontdoor_after_port = args.port_base + 2
    dr3_spec = build_dr3_arm_spec(args, dr3_port)
    return {
        "schema": f"{SCHEMA}.plan",
        "created_at": utc_now(),
        "execute": args.execute,
        "binary": str(args.binary),
        "frontdoor_scenario": args.frontdoor_scenario,
        "contexts": args.contexts,
        "frontdoor_max_tokens": args.frontdoor_max_tokens,
        "min_completion_tokens": args.min_completion_tokens,
        "reps": args.reps,
        "fixed_k": dr3.K_VALUE,
        "serving_route_allowed": False,
        "numeric_swarm_surface_allowed": False,
        "arms": {
            "frontdoor_alone_before_eviction": {
                "port": frontdoor_before_port,
                "argv": frontdoor_argv(args, frontdoor_before_port),
            },
            "dr3_lane_active": {
                "port": dr3_port,
                "env": dr3_spec.env,
                "argv": dr3_spec.argv,
                "context": dr3_spec.context,
                "task_class": args.dr3_task_class,
            },
            "frontdoor_after_eviction_reload": {
                "port": frontdoor_after_port,
                "argv": frontdoor_argv(args, frontdoor_after_port),
            },
        },
        "required_evidence": [
            "frontdoor_alone_before_eviction decode and load wall time",
            "DR-3 combined K2 lane active quality, speed, alpha, and lease cleanup",
            "frontdoor_after_eviction_reload decode and load wall time",
            "post-run no llama-family process leak and no KFD PID leak",
        ],
    }


def render_commands(plan: dict[str, Any]) -> str:
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    before = plan["arms"]["frontdoor_alone_before_eviction"]
    dr3_arm = plan["arms"]["dr3_lane_active"]
    after = plan["arms"]["frontdoor_after_eviction_reload"]
    lines.extend(["# frontdoor alone before DR-3 lease", shlex.join(before["argv"]), ""])
    lines.extend(
        [
            "# DR-3 K2 lane active",
            dr0.render_shell(dr3_arm["argv"], dr3_arm["env"]),
            "",
        ]
    )
    lines.extend(["# frontdoor after DR-3 eviction/reload", shlex.join(after["argv"]), ""])
    return "\n".join(lines)


def run_frontdoor_request(
    args: argparse.Namespace,
    port: int,
    context: int,
) -> dict[str, Any]:
    scenario = k35.scenario_by_name(args.frontdoor_scenario)
    prompt = k35.prompt_for_context(context, args.frontdoor_max_tokens)
    body = k35.build_chat_request_body(
        scenario,
        prompt,
        max_tokens=args.frontdoor_max_tokens,
    )
    started = time.perf_counter()
    response = k35.query_chat(port, body, timeout_s=args.request_timeout)
    elapsed_s = time.perf_counter() - started
    summary = k35.summarize_response(
        scenario,
        context,
        args.frontdoor_max_tokens,
        response,
        elapsed_s,
        args.min_completion_tokens,
    )
    summary["raw_response"] = response
    summary["prompt_sha256"] = dr0.sha256_text(prompt)
    summary["request_sha256"] = dr0.sha256_text(canonical_json(body))
    return summary


def write_raw_response(path: Path, result: dict[str, Any]) -> None:
    raw = result.pop("raw_response", None)
    if raw is not None:
        write_json(path, raw)


def run_frontdoor_phase(
    args: argparse.Namespace,
    plan: dict[str, Any],
    *,
    arm_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    arm = plan["arms"][arm_id]
    arm_dir = output_dir / arm_id
    proc: subprocess.Popen[str] | None = None
    memory_samples: list[dict[str, Any]] = []
    load_wall_clock_s: float | None = None
    results: list[dict[str, Any]] = []
    status = "error"
    error: str | None = None
    try:
        started = time.perf_counter()
        proc = launch_argv(arm["argv"], arm_dir / "frontdoor.server.log")
        k35.wait_for_health(arm["port"], args.startup_timeout)
        load_wall_clock_s = time.perf_counter() - started
        memory_samples.append(k35.collect_resident_memory_sample(proc.pid, "after_health"))
        for rep in range(args.reps):
            for context in args.contexts:
                result = run_frontdoor_request(args, arm["port"], context)
                result["rep"] = rep
                result["phase"] = arm_id
                response_path = arm_dir / f"{arm_id}_rep{rep}_ctx{context}.response.json"
                write_raw_response(response_path, result)
                result["response_path"] = str(response_path)
                results.append(result)
        memory_samples.append(k35.collect_resident_memory_sample(proc.pid, "after_requests"))
        status = "ok" if results and all(row.get("passed_min_completion") for row in results) else "quality_fail"
    except Exception as exc:  # noqa: BLE001 - preserve failures in artifact
        error = repr(exc)
    finally:
        cleanup = terminate_process(proc, port=int(arm["port"])) if proc is not None else {"status": "not_started"}
    phase = {
        "arm": arm_id,
        "status": status,
        "error": error,
        "load_wall_clock_s": load_wall_clock_s,
        "results": results,
        "memory_samples": memory_samples,
        "cleanup": cleanup,
    }
    write_json(arm_dir / "result.json", phase)
    return phase


def run_dr3_lane_active(
    args: argparse.Namespace,
    plan: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    arm_plan = plan["arms"]["dr3_lane_active"]
    spec = build_dr3_arm_spec(args, int(arm_plan["port"]))
    arm_dir = output_dir / "dr3_lane_active"
    log_path = arm_dir / "dr3.server.log"
    proc: subprocess.Popen[str] | None = None
    memory_samples: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    load_wall_clock_s: float | None = None
    status = "error"
    error: str | None = None
    try:
        started = time.perf_counter()
        proc = dr0.launch_server(spec.argv, spec.env, log_path)
        dr0.wait_for_health(spec.port, args.startup_timeout, pid=proc.pid)
        load_wall_clock_s = time.perf_counter() - started
        memory_samples.append(k35.collect_resident_memory_sample(proc.pid, "after_health"))
        for task_row in dr3_task_rows(args):
            raw_path = arm_dir / f"{task_row['row_id']}.raw.json"
            response, raw_response, wall_s = dr0.query_chat(
                port=spec.port,
                prompt=task_row["prompt"],
                max_tokens=int(task_row.get("max_tokens") or args.dr3_max_tokens),
                temperature=0.0,
                seed=task_row["seed"],
                timeout_s=args.request_timeout,
            )
            raw_path.write_text(raw_response, encoding="utf-8")
            rows.append(dr3.row_from_response(spec, task_row, response, raw_path, wall_s))
        metrics = dr0.fetch_metrics(spec.port)
        write_json(arm_dir / "metrics.json", metrics)
        memory_samples.append(k35.collect_resident_memory_sample(proc.pid, "after_requests"))
        status = (
            "ok"
            if rows
            and all(row.get("status") == "ok" for row in rows)
            and all(row.get("quality", {}).get("pass") is True for row in rows)
            else "quality_fail"
        )
    except Exception as exc:  # noqa: BLE001 - preserve failures in artifact
        error = repr(exc)
    finally:
        cleanup = dr0.terminate_server(proc) if proc is not None else {"status": "not_started"}
        if proc is not None:
            cleanup["port_open_after"] = dr0.port_is_open(spec.port)
            cleanup["status"] = "ok" if cleanup.get("terminated") and not cleanup["port_open_after"] else "fail"
    write_json(arm_dir / "responses.json", rows)
    aggregate = dr0.aggregate_arm_rows(dr3.fake_variant(spec), rows, load_wall_clock_s)
    aggregate["context_band"] = spec.context
    aggregate["row_count"] = len(rows)
    result = {
        "arm": "dr3_lane_active",
        "status": status,
        "error": error,
        "spec": {
            "id": spec.id,
            "base_arm": spec.base_arm_id,
            "context": spec.context,
            "k": spec.k,
            "port": spec.port,
            "env": spec.env,
            "argv": spec.argv,
            "shell": dr0.render_shell(spec.argv, spec.env),
        },
        "load_wall_clock_s": load_wall_clock_s,
        "aggregate": aggregate,
        "rows": rows,
        "memory_samples": memory_samples,
        "cleanup": cleanup,
    }
    write_json(arm_dir / "result.json", result)
    return result


def mean_numeric(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def summarize_frontdoor_phase(phase: dict[str, Any]) -> dict[str, Any]:
    rows = phase.get("results") or []
    return {
        "status": phase.get("status"),
        "load_wall_clock_s": phase.get("load_wall_clock_s"),
        "decode_tps_mean": mean_numeric([row.get("decode_tps") for row in rows]),
        "prompt_tps_mean": mean_numeric([row.get("prompt_tps") for row in rows]),
        "completion_tokens_total": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "passed_min_completion": bool(rows) and all(row.get("passed_min_completion") for row in rows),
        "cleanup_status": phase.get("cleanup", {}).get("status"),
    }


def cleanup_proof(pre_process: dict[str, Any], post_process: dict[str, Any], pre_rocm: dict[str, Any], post_rocm: dict[str, Any]) -> dict[str, Any]:
    pre_pids = dr0.snapshot_pid_set(pre_process)
    post_pids = dr0.snapshot_pid_set(post_process)
    new_post_pids = sorted(post_pids - pre_pids)
    no_llama_leak = not post_process.get("lines")
    no_kfd_leak = not post_rocm.get("kfd_pids_observed")
    status = "pass" if no_llama_leak and no_kfd_leak else "fail"
    return {
        "status": status,
        "pre_process_snapshot": pre_process,
        "post_process_snapshot": post_process,
        "pre_rocm_smi_showpids": pre_rocm,
        "post_rocm_smi_showpids": post_rocm,
        "new_post_process_pids": new_post_pids,
        "no_llama_process_leak": no_llama_leak,
        "no_kfd_pid_leak": no_kfd_leak,
    }


def build_summary(
    args: argparse.Namespace,
    plan: dict[str, Any],
    *,
    pre_process: dict[str, Any] | None = None,
    post_process: dict[str, Any] | None = None,
    pre_rocm: dict[str, Any] | None = None,
    post_rocm: dict[str, Any] | None = None,
    results: dict[str, Any] | None = None,
    status: str | None = None,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    results = results or {}
    before = summarize_frontdoor_phase(results.get("frontdoor_alone_before_eviction", {}))
    after = summarize_frontdoor_phase(results.get("frontdoor_after_eviction_reload", {}))
    dr3_result = results.get("dr3_lane_active", {})
    dr3_aggregate = dr3_result.get("aggregate") or {}
    cleanup = (
        cleanup_proof(
            pre_process or {"lines": []},
            post_process or {"lines": []},
            pre_rocm or {"kfd_pids_observed": False},
            post_rocm or {"kfd_pids_observed": False},
        )
        if args.execute
        else {"status": "not_run"}
    )
    after_ratio = (
        after["decode_tps_mean"] / before["decode_tps_mean"]
        if isinstance(after.get("decode_tps_mean"), (int, float))
        and isinstance(before.get("decode_tps_mean"), (int, float))
        and before["decode_tps_mean"] > 0
        else None
    )
    gate_pass = (
        args.execute
        and before.get("status") == "ok"
        and after.get("status") == "ok"
        and dr3_result.get("status") == "ok"
        and cleanup.get("status") == "pass"
    )
    return {
        "schema": f"{SCHEMA}.summary",
        "created_at": utc_now(),
        "mode": "execute" if args.execute else "dry_run",
        "artifact_dir": str(args.output_dir),
        "status": status or ("pass" if gate_pass else ("not_run" if not args.execute else "fail")),
        "blockers": blockers or [],
        "decision_grade": False,
        "observation_grade": bool(gate_pass),
        "serving_route_allowed": False,
        "numeric_swarm_surface_allowed": False,
        "plan": plan,
        "results": results,
        "frontdoor_opportunity_cost_gate": {
            "status": "pass" if gate_pass else ("not_run" if not args.execute else "fail"),
            "serving_blocker": not gate_pass,
            "frontdoor_before": before,
            "frontdoor_after_eviction_reload": after,
            "after_vs_before_decode_ratio": after_ratio,
            "dr3_lane_active": {
                "status": dr3_result.get("status"),
                "decode_tps": dr3_aggregate.get("decode_tps"),
                "prompt_tps": dr3_aggregate.get("prompt_tps"),
                "alpha": dr3_aggregate.get("alpha"),
                "draft_tokens": dr3_aggregate.get("draft_tokens"),
                "accepted_draft_tokens": dr3_aggregate.get("accepted_draft_tokens"),
                "load_wall_clock_s": dr3_result.get("load_wall_clock_s"),
                "cleanup_status": dr3_result.get("cleanup", {}).get("status"),
            },
            "requirement": (
                "measure resident frontdoor alone, frontdoor after eviction/reload, "
                "and DR-3 lane active before routing policy rollout"
            ),
        },
        "cleanup_proof": cleanup,
        "p_gpu_1_gate": {
            "status": "not_applicable_to_experimental_observation",
            "serving_blocker": True,
            "requirement": "decision-grade production GPU claims require production-consolidated-v7 or later",
        },
    }


def validate_inputs(args: argparse.Namespace) -> None:
    dr0.validate_experimental_binary(args.binary)
    missing = [
        str(path)
        for path in (args.binary, args.cpu_verifier_model, args.mi210_drafter_model)
        if not path.exists()
    ]
    scenario = k35.scenario_by_name(args.frontdoor_scenario)
    if not scenario.model.exists():
        missing.append(str(scenario.model))
    if missing:
        raise FileNotFoundError("missing DR-3d input(s): " + ", ".join(missing))


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(args)
    pre_process = dr0.process_snapshot()
    pre_rocm = dr0.rocm_smi_showpids()
    if not args.allow_existing_processes:
        blockers = []
        if pre_process.get("lines"):
            blockers.append("existing llama-family process(es): " + "; ".join(pre_process["lines"][:5]))
        if pre_rocm.get("kfd_pids_observed"):
            blockers.append("existing ROCm KFD process(es)")
        if blockers:
            summary = build_summary(
                args,
                plan,
                pre_process=pre_process,
                pre_rocm=pre_rocm,
                status="blocked",
                blockers=blockers,
            )
            write_json(args.output_dir / "summary.json", summary)
            return summary
    results = {
        "frontdoor_alone_before_eviction": run_frontdoor_phase(
            args,
            plan,
            arm_id="frontdoor_alone_before_eviction",
            output_dir=args.output_dir,
        ),
        "dr3_lane_active": run_dr3_lane_active(args, plan, args.output_dir),
        "frontdoor_after_eviction_reload": run_frontdoor_phase(
            args,
            plan,
            arm_id="frontdoor_after_eviction_reload",
            output_dir=args.output_dir,
        ),
    }
    post_process = dr0.process_snapshot()
    post_rocm = dr0.rocm_smi_showpids()
    summary = build_summary(
        args,
        plan,
        pre_process=pre_process,
        post_process=post_process,
        pre_rocm=pre_rocm,
        post_rocm=post_rocm,
        results=results,
    )
    write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DR-3d frontdoor opportunity-cost gate")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=k35.DEFAULT_BINARY)
    parser.add_argument("--frontdoor-scenario", default=DEFAULT_FRONTDOOR_SCENARIO)
    parser.add_argument("--context", action="append", type=int)
    parser.add_argument("--frontdoor-max-tokens", type=int, default=DEFAULT_FRONTDOOR_MAX_TOKENS)
    parser.add_argument("--min-completion-tokens", type=int, default=DEFAULT_MIN_COMPLETION_TOKENS)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--dr3-context", type=int, default=8192)
    parser.add_argument("--dr3-task-class", default=DEFAULT_DR3_TASK_CLASS)
    parser.add_argument("--dr3-max-tokens", type=int, default=512)
    parser.add_argument("--cpu-verifier-model", type=Path, default=dr0.DEFAULT_CPU_VERIFIER_MODEL)
    parser.add_argument("--mi210-drafter-model", type=Path, default=dr0.DEFAULT_MI210_DRAFTER_MODEL)
    parser.add_argument("--dr3-threads", type=int, default=dr0.DEFAULT_THREADS)
    parser.add_argument("--dr3-ubatch", type=int, default=dr0.DEFAULT_UBATCH)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument("--seed", type=int, default=dr0.DEFAULT_SEED)
    parser.add_argument("--context-fill-chars-per-token", type=float, default=dr3.DEFAULT_CONTEXT_FILL_CHARS_PER_TOKEN)
    parser.add_argument("--max-context-fill-chars", type=int, default=dr3.DEFAULT_MAX_CONTEXT_FILL_CHARS)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--allow-existing-processes", action="store_true")
    args = parser.parse_args(argv)
    args.contexts = selected_contexts(args.context)
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if args.frontdoor_max_tokens <= 0 or args.dr3_max_tokens <= 0:
        raise ValueError("max token values must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    write_json(args.output_dir / "plan.json", plan)
    (args.output_dir / "commands.sh").write_text(render_commands(plan), encoding="utf-8")
    if not args.execute:
        summary = build_summary(args, plan)
        write_json(args.output_dir / "summary.json", summary)
        print(f"dry-run plan written to {args.output_dir}")
        print("arms: frontdoor_alone_before_eviction, dr3_lane_active, frontdoor_after_eviction_reload")
        return 0
    summary = execute(args, plan)
    print(canonical_json(summary))
    return 0 if summary.get("status") in {"pass", "blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
