#!/usr/bin/env python3
"""Bounded CPU configuration discovery for the Laguna Q4 v8 bench arm.

This is deliberately a throughput-only, observation-grade instrument.  It
never changes a serving lineup and the selection it writes is only the exact
recipe for the subsequent Q4 quality arm, not a model or role verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laguna_q4_cpu_bench_runner as base


OUTPUT_ROOT = base.RESEARCH_ROOT / "artifacts/laguna-q4-cpu-v8-20260726/config-discovery"
PROMPT_FILE = base.SWE_QUESTIONS
PROMPT_ID = "django__django-10999"
PORT = 18096
REPS = 3
BENCH_CPUS = frozenset(range(96))
GPU_CPUS = frozenset(range(184, 192))
SIDE_CGROUP = "/epyc-v8-gpu-sidecar"


@dataclass(frozen=True)
class Cell:
    name: str
    threads: int
    batch_threads: int
    flash_attention: bool


CELLS = (
    Cell("baseline_96t_96tb_fa_on", 96, 96, True),
    Cell("candidate_72t_72tb_fa_on", 72, 72, True),
    Cell("candidate_48t_48tb_fa_on", 48, 48, True),
    Cell("candidate_96t_96tb_fa_off", 96, 96, False),
)


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_swap() -> dict[str, int]:
    rows = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, *_ = line.split()
        if key in {"SwapTotal:", "SwapFree:"}:
            rows[key[:-1]] = int(value)
    if set(rows) != {"SwapTotal", "SwapFree"}:
        raise RuntimeError("cannot read swap counters")
    return rows


def process_cgroup(pid: int, proc_root: Path = Path("/proc")) -> str:
    paths = []
    for row in (proc_root / str(pid) / "cgroup").read_text().splitlines():
        fields = row.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            paths.append(fields[2])
    if len(paths) != 1:
        raise RuntimeError(f"cannot establish unified cgroup for PID {pid}")
    return paths[0]


def exact_thread_affinity(
    pid: int,
    expected: frozenset[int],
    role: str,
    proc_root: Path = Path("/proc"),
) -> list[dict[str, Any]]:
    rows = base.proc_thread_cpu_allowed_lists(pid, proc_root)
    for row in rows:
        actual = frozenset(base.cpu_list(str(row["cpus_allowed_list"])))
        if actual != expected:
            raise RuntimeError(
                f"{role} PID {pid} TID {row['tid']} affinity {row['cpus_allowed_list']} != "
                f"{min(expected)}-{max(expected)}"
            )
    return rows


def hip_sidecars(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Capture all live HIP llama-server sidecars, not merely the known queue."""
    accelerated = str((base.LLAMA_ROOT / "build-hip/bin/llama-server").resolve())
    rows = []
    for row in base.live_llama_rows():
        if row.get("exe") != accelerated:
            continue
        pid = int(row["pid"])
        threads = exact_thread_affinity(pid, GPU_CPUS, "GPU sidecar", proc_root)
        cgroup = process_cgroup(pid, proc_root)
        if cgroup != SIDE_CGROUP:
            raise RuntimeError(f"GPU sidecar PID {pid} cgroup {cgroup!r} != {SIDE_CGROUP!r}")
        rows.append({**row, "cgroup": cgroup, "thread_cpu_allowed_lists": threads})
    return rows


def kfd_client_pids() -> list[int]:
    """Find GPU clients without trusting a scheduler or a predeclared queue."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for fd in (entry / "fd").iterdir():
                if os.readlink(fd) == "/dev/kfd":
                    found.append(int(entry.name))
                    break
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return sorted(found)


def continuity_sample(candidate_pid: int | None) -> dict[str, Any]:
    sidecars = hip_sidecars()
    kfd_pids = kfd_client_pids()
    sidecar_pids = {int(row["pid"]) for row in sidecars}
    unknown_gpu = sorted(set(kfd_pids) - sidecar_pids)
    if unknown_gpu:
        raise RuntimeError(f"unattested /dev/kfd client(s): {unknown_gpu}")
    candidate = None
    if candidate_pid is not None:
        candidate = {
            "pid": candidate_pid,
            "thread_cpu_allowed_lists": exact_thread_affinity(candidate_pid, BENCH_CPUS, "CPU candidate"),
        }
    return {
        "at": now(),
        "swap": read_swap(),
        "sidecar_detection_scope": "all live llama-server processes whose executable is llama.cpp/build-hip/bin/llama-server plus every /dev/kfd client",
        "gpu_workload_status": "present" if sidecars else "absent_no_hip_llama_server_detected",
        "kfd_client_pids": kfd_pids,
        "sidecars": sidecars,
        "candidate": candidate,
    }


class ContinuityMonitor:
    """A continuous witness whose lifetime strictly encloses the server."""

    def __init__(
        self,
        output: Path,
        *,
        interval_s: float = 1.0,
        window_samples: int = 5,
    ) -> None:
        self.output = output
        self.candidate_pid: int | None = None
        self.interval_s = interval_s
        self.window_samples = window_samples
        self.stop_event = threading.Event()
        self.failure: str | None = None
        self.lock = threading.RLock()
        self.started = False
        self.initial_swap: dict[str, int] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self, phase: str, *, allow_prior_failure: bool = False) -> None:
        with self.lock:
            prior_failure = self.failure
            if prior_failure and not allow_prior_failure:
                raise RuntimeError("continuity failure: " + self.failure)
            try:
                record = {"phase": phase, **continuity_sample(self.candidate_pid)}
                if record["swap"] != self.initial_swap:
                    raise RuntimeError("swap counters moved during discovery")
                if prior_failure:
                    record["prior_failure"] = prior_failure
            except Exception as exc:
                record = {"at": now(), "phase": phase, "error": repr(exc)}
                if self.failure is None:
                    self.failure = repr(exc)
            with self.output.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if self.failure and not allow_prior_failure:
                raise RuntimeError("continuity failure: " + self.failure)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            try:
                self.sample("during")
            except RuntimeError:
                return

    def start_prelaunch(self) -> None:
        if not self.started:
            self.initial_swap = read_swap()
        for _ in range(self.window_samples):
            self.sample("prelaunch")
            time.sleep(self.interval_s)
        # The background witness is live before subprocess.Popen.
        self.thread.start()
        self.started = True

    def attach_candidate(self, candidate_pid: int) -> None:
        with self.lock:
            self.candidate_pid = candidate_pid
            self.sample("candidate-attached")

    def detach_candidate_before_teardown(self) -> int | None:
        """Atomically sample the live PID, then prevent any stale-PID sample."""
        with self.lock:
            pid = self.candidate_pid
            error: Exception | None = None
            try:
                if pid is not None:
                    self.sample("pre-teardown")
            except Exception as exc:
                error = exc
            finally:
                self.candidate_pid = None
            if error is not None:
                raise error
            return pid

    def post_cleanup_window(self) -> None:
        with self.lock:
            if self.candidate_pid is not None:
                raise RuntimeError("candidate must be detached before post-cleanup window")
        for _ in range(self.window_samples):
            self.sample("post-cleanup", allow_prior_failure=True)
            time.sleep(self.interval_s)

    def ensure_healthy(self) -> None:
        with self.lock:
            if self.failure:
                raise RuntimeError("continuity failure: " + self.failure)

    def close(self) -> None:
        """Idempotent and safe even when prelaunch failed before thread start."""
        self.stop_event.set()
        if self.started:
            self.thread.join(self.interval_s + 1)


def server_argv(cell: Cell) -> list[str]:
    return [
        str(base.TASKSET), "-c", "0-95", str(base.NUMACTL), "--interleave=all", str(base.BINARY),
        "-m", str(base.MODEL), "--host", "127.0.0.1", "--port", str(PORT), "-c", "49152",
        "-t", str(cell.threads), "-tb", str(cell.batch_threads), "-b", "2048", "-ub", "2048",
        "-np", "1", "-ctk", "f16", "-ctv", "f16", "-fa", "on" if cell.flash_attention else "off",
        "-ngl", "0", "-dev", "none", "--no-op-offload", "--no-mmap", "--jinja", "--metrics",
        "--slots", "--reasoning", "off",
    ]


def server_env(cell: Cell) -> dict[str, str]:
    environment = base.clean_env()
    environment["OMP_NUM_THREADS"] = str(cell.threads)
    return environment


def stable_model_stat() -> dict[str, Any]:
    return base.path_stat_identity(base.MODEL)


def assert_model_stat(expected: dict[str, Any]) -> dict[str, Any]:
    actual = stable_model_stat()
    if actual != expected:
        raise RuntimeError("Laguna Q4 model stat identity drifted within the campaign")
    return actual


def prompt_identity() -> dict[str, Any]:
    rows = json.loads(PROMPT_FILE.read_text())
    selected = [row for row in rows if row.get("id") == PROMPT_ID]
    if len(selected) != 1 or not isinstance(selected[0].get("prompt"), str):
        raise RuntimeError("pinned representative SWE prompt is missing or ambiguous")
    row = selected[0]
    return {"path": str(PROMPT_FILE), "sha256": base.sha256(PROMPT_FILE), "id": PROMPT_ID,
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(), "prompt": row["prompt"]}


def request_body(prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_measurement(
    payload: dict[str, Any], elapsed: float, *, min_completion_tokens: int = 0
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("completion response is not a JSON object")
    usage = payload.get("usage") or {}
    timings = payload.get("timings") or {}
    completion = usage.get("completion_tokens", timings.get("predicted_n"))
    if not isinstance(completion, int):
        raise RuntimeError("response lacks integer completion token count")
    if completion < min_completion_tokens:
        raise RuntimeError(
            f"measurement completion token floor ({completion} < {min_completion_tokens}) failed"
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("completion response choice cardinality is not one")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content:
        raise RuntimeError("completion response has no assistant content")
    decode = timings.get("predicted_per_second")
    if not isinstance(decode, (int, float)) or decode <= 0:
        raise RuntimeError("measurement lacks positive server decode throughput")
    return {
        "wall_s": elapsed,
        "completion_tokens": completion,
        "prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n")),
        "decode_tok_s": decode,
        "timings": timings,
        "finish_reason": choices[0].get("finish_reason"),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def request(
    port: int,
    prompt: str,
    max_tokens: int,
    *,
    min_completion_tokens: int = 0,
) -> dict[str, Any]:
    body = json.dumps(request_body(prompt, max_tokens)).encode()
    started = time.monotonic()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"completion returned HTTP {response.status}")
        payload = json.loads(response.read())
    return validate_measurement(
        payload,
        time.monotonic() - started,
        min_completion_tokens=min_completion_tokens,
    )


def iqk_engagement(log_text: str) -> dict[str, Any]:
    exact = "[iqk] ACTIVE: ik_llama GEMM kernels engaged"
    lines = [line for line in log_text.splitlines() if exact in line]
    evidence = {"required_exact_text": exact, "matching_lines": lines, "engaged": bool(lines)}
    if not evidence["engaged"]:
        raise RuntimeError("server stderr lacks exact IQK ACTIVE engagement evidence")
    return evidence


def pid_exists(pid: int) -> bool:
    return (Path("/proc") / str(pid)).exists()


def terminate(process: subprocess.Popen[str]) -> dict[str, Any]:
    signals = []
    if process.poll() is None:
        signals.append("SIGTERM")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            signals.append("SIGKILL")
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(30)
    pid_absent = not pid_exists(process.pid)
    port_free = base.port_free(PORT)
    expected_exit = process.returncode in {0, -signal.SIGTERM, -signal.SIGKILL}
    if process.poll() is None or not pid_absent or not port_free or not expected_exit:
        raise RuntimeError("fresh discovery server cleanup failed")
    return {
        "pid": process.pid,
        "signals": signals,
        "exit_code": process.returncode,
        "expected_exit": expected_exit,
        "pid_absent": pid_absent,
        "port_free": port_free,
    }


def run_rep(
    cell: Cell,
    rep: int,
    output: Path,
    prompt: dict[str, Any],
    static: dict[str, Any],
    campaign_model_stat: dict[str, Any],
) -> dict[str, Any]:
    rep_dir = output / "runs" / f"{cell.name}_rep{rep}"
    rep_dir.mkdir(parents=True)
    argv = server_argv(cell)
    environment = server_env(cell)
    write_json(rep_dir / "server_argv.json", argv)
    write_json(rep_dir / "server_env.json", environment)
    started = now()
    monitor = ContinuityMonitor(rep_dir / "continuity.jsonl")
    process: subprocess.Popen[str] | None = None
    stderr = None
    result: dict[str, Any] = {
        "cell": cell.name,
        "rep": rep,
        "started_at": started,
        "status": "error",
        "argv": argv,
        "environment": environment,
        "campaign_model_stat": campaign_model_stat,
    }
    try:
        # The witness thread is already live when the server is created.
        monitor.start_prelaunch()
        prewarm = base.numa_prewarm(rep_dir, {"name": "config_discovery", "port": PORT})
        result["numa_prewarm"] = prewarm
        monitor.ensure_healthy()
        result["model_prelaunch_stat"] = assert_model_stat(campaign_model_stat)
        stderr = (rep_dir / "server.stderr").open("w")
        launch = time.monotonic()
        process = subprocess.Popen(
            argv,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        monitor.attach_candidate(process.pid)
        readiness = base.wait_ready(process, {"port": PORT}, static)
        readiness["startup_wall_s"] = time.monotonic() - launch
        exact_thread_affinity(process.pid, BENCH_CPUS, "CPU candidate")
        warmup = request(PORT, prompt["prompt"], 64)
        measured = request(
            PORT,
            prompt["prompt"],
            512,
            min_completion_tokens=256,
        )
        monitor.ensure_healthy()
        stderr.flush()
        iqk = iqk_engagement((rep_dir / "server.stderr").read_text(errors="replace"))
        result.update(
            status="ok",
            server_timing=readiness,
            warmup=warmup,
            measurement=measured,
            iqk_engagement=iqk,
            model_postrun_stat=assert_model_stat(campaign_model_stat),
        )
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        cleanup_errors: list[str] = []
        try:
            monitor.detach_candidate_before_teardown()
        except Exception as exc:
            cleanup_errors.append("pre-teardown witness: " + repr(exc))
        try:
            if process is not None:
                result["cleanup"] = terminate(process)
        except Exception as exc:
            cleanup_errors.append("server cleanup: " + repr(exc))
        try:
            if stderr is not None:
                stderr.close()
            monitor.post_cleanup_window()
        except Exception as exc:
            cleanup_errors.append("post-cleanup witness: " + repr(exc))
        finally:
            monitor.close()
        if monitor.failure:
            cleanup_errors.append("continuity: " + monitor.failure)
        if cleanup_errors:
            result["status"] = "error"
            result["cleanup_errors"] = cleanup_errors
        result["finished_at"] = now()
        write_json(rep_dir / "result.json", result)
    return result


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries = {}
    baseline = CELLS[0].name
    for cell in CELLS:
        records = [row for row in rows if row["cell"] == cell.name]
        values = [float(row["measurement"]["decode_tok_s"]) for row in records if row.get("status") == "ok"]
        summaries[cell.name] = {"reps": len(records), "ok_reps": len(values), "all_ok": len(values) == REPS,
                                "decode_tok_s": {"values": values, "median": statistics.median(values) if values else None,
                                                 "minimum": min(values) if values else None}}
    base_summary = summaries[baseline]
    selection = {
        "status": "baseline_retained",
        "selected_cell": baseline,
        "rule": "candidate median >= 1.03 * baseline median AND candidate minimum > baseline median",
    }
    incomplete = [name for name, summary in summaries.items() if not summary["all_ok"]]
    if incomplete:
        selection.update(
            status="invalid",
            selected_cell=None,
            reason=f"incomplete or invalid cells: {incomplete}",
        )
    else:
        eligible = []
        for cell in CELLS[1:]:
            candidate = summaries[cell.name]
            if candidate["all_ok"] and candidate["decode_tok_s"]["median"] >= 1.03 * base_summary["decode_tok_s"]["median"] and candidate["decode_tok_s"]["minimum"] > base_summary["decode_tok_s"]["median"]:
                eligible.append(cell.name)
        if eligible:
            selection.update(status="candidate_selected", selected_cell=max(eligible, key=lambda name: summaries[name]["decode_tok_s"]["median"]), eligible_candidates=eligible)
    return summaries, selection


def campaign_state(
    status: str,
    schedule: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    fatal_error: str | None = None,
) -> dict[str, Any]:
    completed = {(row["cell"], row["rep"]) for row in rows}
    remaining = [
        item for item in schedule if (item["cell"], item["rep"]) not in completed
    ]
    return {
        "schema": "epyc.laguna_q4_cpu_config_discovery.state.v1",
        "updated_at": now(),
        "status": status,
        "fatal_error": fatal_error,
        "completed": rows,
        "remaining_not_attempted": remaining,
        "decision_valid": status == "complete" and not remaining
        and all(row.get("status") == "ok" for row in rows),
    }


def bind_selection_to_campaign(
    selection: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    """A terminal identity or continuity failure overrides cell statistics."""
    if state.get("decision_valid") is True:
        return selection
    return {
        **selection,
        "status": "invalid",
        "selected_cell": None,
        "reason": (
            "campaign terminal state is not decision-valid: "
            f"{state.get('fatal_error') or state.get('status')}"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        summaries, selection = summarize([
            {"cell": cell.name, "rep": rep, "status": "ok", "measurement": {"decode_tok_s": value}}
            for cell, value in ((CELLS[0], 10), (CELLS[1], 10.3), (CELLS[2], 12), (CELLS[3], 9)) for rep in range(1, 4)
        ])
        assert selection["selected_cell"] == CELLS[2].name and summaries[CELLS[0].name]["all_ok"]
        print("LAGUNA_Q4_CPU_CONFIG_DISCOVERY_SELF_TEST_OK")
        return 0
    if not args.execute:
        plan = {"schema": "epyc.laguna_q4_cpu_config_discovery.plan.v1", "observation_only": True,
                "execute": False, "cells": [cell.__dict__ for cell in CELLS], "reps_per_cell": REPS,
                "fresh_server_per_rep": True, "fixed_recipe": {"context": 49152, "batch": 2048, "ubatch": 2048, "kv": "f16", "iqk": "1", "cpu_only": True, "no_mmap": True, "dflash": "off", "taskset": "0-95", "numa": "interleave=all"},
                "selection_rule": "baseline retained unless candidate median >=3% over baseline and candidate min > baseline median"}
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.output_dir is None or args.output_dir.exists():
        raise RuntimeError("--execute requires a fresh --output-dir")
    args.output_dir.mkdir(parents=True)
    prompt = prompt_identity()
    static = base.validate_static(verify_model_sha=True)
    model_stat = stable_model_stat()
    plan = {"schema": "epyc.laguna_q4_cpu_config_discovery.plan.v1", "created_at": now(), "runner": str(Path(__file__).resolve()),
            "runner_sha256": base.sha256(Path(__file__).resolve()), "static_identity": static,
            "campaign_model_identity": {
                "path": str(base.MODEL.resolve(strict=True)),
                "sha256": static["model_sha256"],
                "stat": model_stat,
                "hash_policy": "full SHA-256 before campaign and after campaign; stable stat identity before/after every rep",
            },
            "prompt": prompt,
            "cells": [cell.__dict__ for cell in CELLS], "reps_per_cell": REPS, "fresh_server_per_rep": True,
            "selection_rule": "baseline retained unless candidate median >=3% over baseline and candidate min > baseline median"}
    write_json(args.output_dir / "plan.json", plan)
    schedule = [
        {"cell": cell.name, "rep": rep}
        for cell in CELLS
        for rep in range(1, REPS + 1)
    ]
    rows: list[dict[str, Any]] = []
    write_json(args.output_dir / "state.json", campaign_state("running", schedule, rows))
    fatal_error = None
    for cell in CELLS:
        for rep in range(1, REPS + 1):
            result = run_rep(cell, rep, args.output_dir, prompt, static, model_stat)
            rows.append(result)
            if result.get("status") != "ok":
                fatal_error = (
                    f"{cell.name} rep {rep} failed; campaign stopped fail-closed: "
                    f"{result.get('error') or result.get('cleanup_errors')}"
                )
                write_json(
                    args.output_dir / "state.json",
                    campaign_state(
                        "invalid_fatal",
                        schedule,
                        rows,
                        fatal_error=fatal_error,
                    ),
                )
                break
            write_json(
                args.output_dir / "state.json",
                campaign_state("running", schedule, rows),
            )
        if fatal_error:
            break
    post_model = None
    try:
        post_model = base.verify_model_identity()
        assert_model_stat(model_stat)
    except Exception as exc:
        fatal_error = fatal_error or f"post-campaign model identity failed: {exc!r}"
    terminal_status = (
        "complete"
        if fatal_error is None and len(rows) == len(schedule)
        else "invalid_fatal"
    )
    state = campaign_state(
        terminal_status,
        schedule,
        rows,
        fatal_error=fatal_error,
    )
    write_json(args.output_dir / "state.json", state)
    summaries, selection = summarize(rows)
    selection = bind_selection_to_campaign(selection, state)
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "ok" if state["decision_valid"] else "failed",
            "campaign_state": state,
            "rows": rows,
            "cells": summaries,
            "selection": selection,
            "post_campaign_model_identity": post_model,
        },
    )
    return 0 if state["decision_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
