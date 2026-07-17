#!/usr/bin/env python3
"""K35 MiniCPM-o/frontdoor service-policy matrix.

This runner measures the realistic serving question left open by the K35
vision work: whether the fast MiniCPM-o MI210 vision lane can remain resident
beside the MI210 frontdoor lane, and what active overlap costs frontdoor decode.

It is intentionally separate from the isolated K35 vision matrix. Baseline rows
are included only as controls for service-tax attribution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import signal
import subprocess
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import k35_stack_context_matrix_runner as k35
import k35_vision_matrix_runner as k35v


RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "k35_minicpm_service_matrix"
    / f"k35_minicpm_service_matrix_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
)
DEFAULT_CONTEXTS = (2048, 8192)
DEFAULT_FRONTDOOR_SCENARIO = "frontdoor_gpu_resident_no_spec"
DEFAULT_MINICPM_SCENARIO = "vision_candidate_mi210_minicpm_o45_q4"
DEFAULT_BASE_PORT = 19320


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_contexts(values: list[int] | None) -> list[int]:
    return values or list(DEFAULT_CONTEXTS)


def selected_fixtures(ids: list[str] | None) -> list[k35v.VisionFixture]:
    if not ids:
        return list(k35v.FIXTURES)
    return [k35v.fixture_by_id(fixture_id) for fixture_id in ids]


def frontdoor_server_context(contexts: list[int], max_tokens: int) -> int:
    scenario = k35.scenario_by_name(DEFAULT_FRONTDOOR_SCENARIO)
    return max(k35.server_context(scenario, context, max_tokens) for context in contexts)


def build_frontdoor_argv(*, binary: Path, port: int, contexts: list[int], max_tokens: int) -> list[str]:
    scenario = k35.scenario_by_name(DEFAULT_FRONTDOOR_SCENARIO)
    return k35.build_server_argv(
        scenario,
        binary=binary,
        port=port,
        nominal_context=max(contexts),
        max_tokens=max_tokens,
    )


def build_minicpm_argv(*, binary: Path, port: int) -> list[str]:
    scenario = k35v.scenario_by_name(DEFAULT_MINICPM_SCENARIO)
    return k35v.build_server_argv(scenario, binary=binary, port=port)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    contexts = selected_contexts(args.context)
    fixtures = selected_fixtures(args.fixture)
    frontdoor_port = k35.pick_port(args.port_base)
    minicpm_port = k35.pick_port(frontdoor_port + 1)
    return {
        "schema": "epyc.k35_minicpm_service_matrix.plan.v1",
        "created_at": utc_now(),
        "execute": args.execute,
        "binary": str(args.binary),
        "frontdoor": {
            "scenario": DEFAULT_FRONTDOOR_SCENARIO,
            "port": frontdoor_port,
            "contexts": contexts,
            "max_tokens": args.frontdoor_max_tokens,
            "min_completion_tokens": args.min_completion_tokens,
            "server_context": frontdoor_server_context(contexts, args.frontdoor_max_tokens),
            "server_argv": build_frontdoor_argv(
                binary=args.binary,
                port=frontdoor_port,
                contexts=contexts,
                max_tokens=args.frontdoor_max_tokens,
            ),
        },
        "minicpm": {
            "scenario": DEFAULT_MINICPM_SCENARIO,
            "port": minicpm_port,
            "fixtures": [fixture.fixture_id for fixture in fixtures],
            "max_tokens": args.vision_max_tokens,
            "server_argv": build_minicpm_argv(binary=args.binary, port=minicpm_port),
        },
        "reps": args.reps,
        "active_overlap_reps": args.active_overlap_reps,
        "arms": [
            "frontdoor_alone_control",
            "frontdoor_with_minicpm_resident_idle",
            "frontdoor_with_minicpm_active_overlap",
        ],
    }


def render_commands(plan: dict[str, Any]) -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            "# frontdoor",
            k35.shlex.join(plan["frontdoor"]["server_argv"]),
            "",
            "# minicpm",
            k35.shlex.join(plan["minicpm"]["server_argv"]),
            "",
        ]
    )


def launch_server(argv: list[str], log_path: Path) -> subprocess.Popen[str]:
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


def terminate_process(proc: subprocess.Popen[str], *, timeout_s: int = 20) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": proc.pid, "terminated": False, "killed": False}
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
            result["terminated"] = True
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)
            result["killed"] = True
    result["returncode"] = proc.returncode
    ps = k35.run_capture(["ps", "-p", str(proc.pid), "-o", "pid=,comm=,args="], timeout=10)
    result["ps_after"] = ps
    result["dead"] = str(proc.pid) not in ps.get("stdout", "")
    return result


def run_frontdoor_request(
    port: int,
    context: int,
    *,
    max_tokens: int,
    min_completion_tokens: int,
    request_timeout: int,
) -> dict[str, Any]:
    scenario = k35.scenario_by_name(DEFAULT_FRONTDOOR_SCENARIO)
    prompt = k35.prompt_for_context(context, max_tokens)
    started = time.monotonic()
    response = k35.query_chat(
        scenario,
        port,
        prompt,
        max_tokens=max_tokens,
        timeout_s=request_timeout,
    )
    elapsed_s = time.monotonic() - started
    summary = k35.summarize_response(
        scenario,
        context,
        max_tokens,
        response,
        elapsed_s,
        min_completion_tokens,
    )
    summary["raw_response"] = response
    return summary


def run_vision_request(
    port: int,
    fixture: k35v.VisionFixture,
    *,
    max_tokens: int,
    request_timeout: int,
) -> dict[str, Any]:
    scenario = k35v.scenario_by_name(DEFAULT_MINICPM_SCENARIO)
    started = time.monotonic()
    response = k35v.query_vision(
        port,
        fixture,
        max_tokens=max_tokens,
        timeout_s=request_timeout,
    )
    elapsed_s = time.monotonic() - started
    summary = k35v.summarize_fixture(scenario, fixture, response, elapsed_s)
    summary["raw_response"] = response
    return summary


def write_raw_response(path: Path, result: dict[str, Any]) -> None:
    raw = result.pop("raw_response", None)
    if raw is not None:
        write_json(path, raw)


def run_frontdoor_series(
    port: int,
    contexts: list[int],
    *,
    reps: int,
    max_tokens: int,
    min_completion_tokens: int,
    request_timeout: int,
    output_dir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rep in range(reps):
        for context in contexts:
            result = run_frontdoor_request(
                port,
                context,
                max_tokens=max_tokens,
                min_completion_tokens=min_completion_tokens,
                request_timeout=request_timeout,
            )
            response_path = output_dir / f"{prefix}_rep{rep}_ctx{context}.response.json"
            write_raw_response(response_path, result)
            result["response_path"] = str(response_path)
            result["rep"] = rep
            result["phase"] = prefix
            results.append(result)
    return results


def launch_pair(plan: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> tuple[subprocess.Popen[str], subprocess.Popen[str], list[dict[str, Any]]]:
    frontdoor_proc = launch_server(plan["frontdoor"]["server_argv"], output_dir / "frontdoor.stderr")
    minicpm_proc = launch_server(plan["minicpm"]["server_argv"], output_dir / "minicpm.stderr")
    memory_samples: list[dict[str, Any]] = []
    try:
        k35.wait_for_health(plan["frontdoor"]["port"], args.startup_timeout)
        k35.wait_for_health(plan["minicpm"]["port"], args.startup_timeout)
        memory_samples.append(
            {
                "phase": "after_pair_health",
                "frontdoor": k35.collect_resident_memory_sample(frontdoor_proc.pid, "frontdoor_after_health"),
                "minicpm": k35.collect_resident_memory_sample(minicpm_proc.pid, "minicpm_after_health"),
            }
        )
    except Exception:
        terminate_process(frontdoor_proc)
        terminate_process(minicpm_proc)
        raise
    return frontdoor_proc, minicpm_proc, memory_samples


def run_frontdoor_alone(plan: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    arm_dir = output_dir / "frontdoor_alone_control"
    proc = launch_server(plan["frontdoor"]["server_argv"], arm_dir / "frontdoor.stderr")
    memory_samples: list[dict[str, Any]] = []
    try:
        k35.wait_for_health(plan["frontdoor"]["port"], args.startup_timeout)
        memory_samples.append(k35.collect_resident_memory_sample(proc.pid, "after_health"))
        frontdoor_results = run_frontdoor_series(
            plan["frontdoor"]["port"],
            plan["frontdoor"]["contexts"],
            reps=args.reps,
            max_tokens=args.frontdoor_max_tokens,
            min_completion_tokens=args.min_completion_tokens,
            request_timeout=args.request_timeout,
            output_dir=arm_dir,
            prefix="frontdoor_alone",
        )
        memory_samples.append(k35.collect_resident_memory_sample(proc.pid, "after_requests"))
        status = "ok" if all(result.get("passed_min_completion") for result in frontdoor_results) else "quality_fail"
        result: dict[str, Any] = {
            "arm": "frontdoor_alone_control",
            "status": status,
            "frontdoor_results": frontdoor_results,
            "memory_samples": memory_samples,
        }
    except Exception as exc:  # noqa: BLE001 - artifact must preserve failures
        result = {"arm": "frontdoor_alone_control", "status": "error", "error": repr(exc)}
    finally:
        result["cleanup"] = terminate_process(proc)
    write_json(arm_dir / "result.json", result)
    return result


def run_pair_arms(plan: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    arm_dir = output_dir / "frontdoor_minicpm_pair"
    memory_samples: list[dict[str, Any]] = []
    frontdoor_proc: subprocess.Popen[str] | None = None
    minicpm_proc: subprocess.Popen[str] | None = None
    try:
        frontdoor_proc, minicpm_proc, memory_samples = launch_pair(plan, arm_dir, args)
        idle_results = run_frontdoor_series(
            plan["frontdoor"]["port"],
            plan["frontdoor"]["contexts"],
            reps=args.reps,
            max_tokens=args.frontdoor_max_tokens,
            min_completion_tokens=args.min_completion_tokens,
            request_timeout=args.request_timeout,
            output_dir=arm_dir,
            prefix="minicpm_idle",
        )
        memory_samples.append(
            {
                "phase": "after_idle_requests",
                "frontdoor": k35.collect_resident_memory_sample(frontdoor_proc.pid, "frontdoor_after_idle"),
                "minicpm": k35.collect_resident_memory_sample(minicpm_proc.pid, "minicpm_after_idle"),
            }
        )
        active_results: list[dict[str, Any]] = []
        fixtures = selected_fixtures(plan["minicpm"]["fixtures"])
        for rep in range(args.active_overlap_reps):
            for context in plan["frontdoor"]["contexts"]:
                for fixture in fixtures:
                    pair_started = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        frontdoor_future = pool.submit(
                            run_frontdoor_request,
                            plan["frontdoor"]["port"],
                            context,
                            max_tokens=args.frontdoor_max_tokens,
                            min_completion_tokens=args.min_completion_tokens,
                            request_timeout=args.request_timeout,
                        )
                        vision_future = pool.submit(
                            run_vision_request,
                            plan["minicpm"]["port"],
                            fixture,
                            max_tokens=args.vision_max_tokens,
                            request_timeout=args.request_timeout,
                        )
                        frontdoor_result = frontdoor_future.result()
                        vision_result = vision_future.result()
                    pair_elapsed_s = time.monotonic() - pair_started
                    frontdoor_response_path = (
                        arm_dir / f"active_rep{rep}_ctx{context}_{fixture.fixture_id}.frontdoor.response.json"
                    )
                    vision_response_path = (
                        arm_dir / f"active_rep{rep}_ctx{context}_{fixture.fixture_id}.vision.response.json"
                    )
                    write_raw_response(frontdoor_response_path, frontdoor_result)
                    write_raw_response(vision_response_path, vision_result)
                    active_results.append(
                        {
                            "rep": rep,
                            "context": context,
                            "fixture_id": fixture.fixture_id,
                            "pair_elapsed_s": pair_elapsed_s,
                            "frontdoor": {
                                **frontdoor_result,
                                "response_path": str(frontdoor_response_path),
                            },
                            "vision": {
                                **vision_result,
                                "response_path": str(vision_response_path),
                            },
                        }
                    )
                    memory_samples.append(
                        {
                            "phase": f"after_active:{rep}:{context}:{fixture.fixture_id}",
                            "frontdoor": k35.collect_resident_memory_sample(
                                frontdoor_proc.pid,
                                "frontdoor_after_active",
                            ),
                            "minicpm": k35.collect_resident_memory_sample(
                                minicpm_proc.pid,
                                "minicpm_after_active",
                            ),
                        }
                    )
        status = (
            "ok"
            if all(result.get("passed_min_completion") for result in idle_results)
            and all(item["frontdoor"].get("passed_min_completion") for item in active_results)
            and all(item["vision"]["score"].get("pass") for item in active_results)
            else "quality_fail"
        )
        result: dict[str, Any] = {
            "arm": "frontdoor_minicpm_pair",
            "status": status,
            "idle_results": idle_results,
            "active_results": active_results,
            "memory_samples": memory_samples,
        }
    except Exception as exc:  # noqa: BLE001 - artifact must preserve failures
        result = {
            "arm": "frontdoor_minicpm_pair",
            "status": "error",
            "error": repr(exc),
            "memory_samples": memory_samples,
        }
    finally:
        cleanup: dict[str, Any] = {}
        if frontdoor_proc is not None:
            cleanup["frontdoor"] = terminate_process(frontdoor_proc)
        if minicpm_proc is not None:
            cleanup["minicpm"] = terminate_process(minicpm_proc)
        result["cleanup"] = cleanup
    write_json(arm_dir / "result.json", result)
    return result


def summarize_decode(values: list[float | None]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"n": 0, "min": None, "max": None, "mean": None}
    return {
        "n": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": sum(numeric) / len(numeric),
    }


def compact_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    frontdoor_alone = next((item for item in results if item.get("arm") == "frontdoor_alone_control"), {})
    pair = next((item for item in results if item.get("arm") == "frontdoor_minicpm_pair"), {})
    alone_tps = [item.get("decode_tps") for item in frontdoor_alone.get("frontdoor_results", [])]
    idle_tps = [item.get("decode_tps") for item in pair.get("idle_results", [])]
    active_frontdoor_tps = [
        item.get("frontdoor", {}).get("decode_tps") for item in pair.get("active_results", [])
    ]
    active_vision_tps = [
        item.get("vision", {}).get("decode_tps") for item in pair.get("active_results", [])
    ]
    return {
        "frontdoor_alone_decode_tps": summarize_decode(alone_tps),
        "frontdoor_idle_resident_decode_tps": summarize_decode(idle_tps),
        "frontdoor_active_overlap_decode_tps": summarize_decode(active_frontdoor_tps),
        "minicpm_active_overlap_decode_tps": summarize_decode(active_vision_tps),
        "active_overlap_passed": all(
            item.get("frontdoor", {}).get("passed_min_completion")
            and item.get("vision", {}).get("score", {}).get("pass")
            for item in pair.get("active_results", [])
        ),
    }


def execute_plan(plan: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    guard = k35.collect_guard_state(args.binary)
    write_json(output_dir / "guard_state.json", guard)
    blockers = guard.get("process_blockers") or []
    if blockers and not args.allow_dirty_host:
        summary = {
            "schema": "epyc.k35_minicpm_service_matrix.summary.v1",
            "created_at": utc_now(),
            "status": "blocked",
            "reason": "process blockers present",
            "blockers": blockers,
            "results": [],
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    results = [
        run_frontdoor_alone(plan, args, output_dir),
        run_pair_arms(plan, args, output_dir),
    ]
    cleanup_guard = k35.collect_process_blockers()
    summary = {
        "schema": "epyc.k35_minicpm_service_matrix.summary.v1",
        "created_at": utc_now(),
        "status": "ok" if all(result.get("status") == "ok" for result in results) else "partial",
        "results": results,
        "compact": compact_summary(results),
        "cleanup_process_blockers": cleanup_guard,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute the service matrix")
    parser.add_argument("--context", action="append", type=int, help="Frontdoor nominal context. May repeat.")
    parser.add_argument(
        "--fixture",
        action="append",
        choices=[fixture.fixture_id for fixture in k35v.FIXTURES],
        help="MiniCPM-o fixture. May repeat. Defaults to all K35 fixtures.",
    )
    parser.add_argument("--reps", type=int, default=2, help="Frontdoor alone/idle reps per context")
    parser.add_argument("--active-overlap-reps", type=int, default=1)
    parser.add_argument("--frontdoor-max-tokens", type=int, default=512)
    parser.add_argument("--vision-max-tokens", type=int, default=96)
    parser.add_argument("--min-completion-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=k35.DEFAULT_BINARY)
    parser.add_argument("--port-base", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--allow-dirty-host", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    write_json(args.output_dir / "plan.json", plan)
    (args.output_dir / "commands.sh").write_text(render_commands(plan), encoding="utf-8")
    if not args.execute:
        print(f"dry-run plan written to {args.output_dir}")
        print(f"arms: {', '.join(plan['arms'])}")
        return 0
    summary = execute_plan(plan, args, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") in {"ok", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
