#!/usr/bin/env python3
"""Observation-only MI210 Laguna IQ2 K/V-cache and Flash Attention sweep.

This intentionally does not compare models or make a promotion decision.  It
holds the experimental candidate, target model, device placement, sampling, and
semantic prompt pack fixed while varying only K/V cache types and Flash
Attention.  Dry-run is the default; ``--execute`` runs fresh servers only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laguna_pgpu1_dflash_runner as common


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = Path("/mnt/raid0/llm/llama.cpp-experimental/build-v8-hip/bin/llama-server")
DEFAULT_SOURCE_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
DEFAULT_TARGET_MODEL = Path("/mnt/raid0/llm/models/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-IQ2_M.gguf")
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "data/gpu-mi210/laguna-iq2-kv-sweep"
EXPECTED_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
EXPECTED_SERVER_SHA256 = "094b395244d71f0d30f82999e53d261f9d4daeea0b651b3e95b51cb6712888ac"
EXPECTED_SERVER_VERSION = "10107"
EXPECTED_LOCAL_LIBRARY_SHA256 = {
    "libllama-server-impl.so": "62ca8b042e8422a3a895b40c00d242cd437e5924c8b89e97f3f7519054b9574a",
    "libllama-common.so.0": "859d51f639f8c19eb2a61cd6e2fc882bb5618c698edd6472ee94189b8ce7bad2",
    "libllama.so.0": "384e58029acdd89fc6031497d04e7940bdd57e7f7de71c3c44716d270029f79d",
    "libggml.so.0": "ddbd5883138c75ffd3aaaa0d00b9ed3364a8cc3331431dbb71b42f268f84fe25",
    "libggml-base.so.0": "f47cc4ad6ab59ea39de7e5fd302f79ba62626cdf84b48332144dfdfa34af0cde",
    "libggml-cpu.so.0": "147b56c811a0ceeb0a59e6cb62a06b42d03cdbf43ab06dd5e84a6c48a650ba34",
    "libggml-hip.so.0": "3bb701d74e5d75cb8d514e7cfb410a0ecdb6b39b9e225d5eb92bc3c75e7aeb45",
}
TARGET_MODEL_BYTES = 37_268_665_376
TARGET_MODEL_SHA256 = "1a0d44795f71044de1a9671bf70def4655f4ab7294b002263dfc8046820bfd2c"
REPS = 5
CONTEXT = 4096
MAX_TOKENS = common.DEFAULT_MAX_TOKENS
MIN_COMPLETION_TOKENS = common.DEFAULT_MIN_COMPLETION_TOKENS
SEED = common.DEFAULT_SEED
PORT_BASE = 19920
SETTLEMENT_TIMEOUT_S = 30.0
SETTLEMENT_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True)
class Cell:
    name: str
    cache_k: str
    cache_v: str
    flash_attention: bool


CELLS = (
    Cell("A_q8_kv_fa_on", "q8_0", "q8_0", True),
    Cell("B_f16_kv_fa_on", "f16", "f16", True),
    Cell("C_f16_kv_fa_off", "f16", "f16", False),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_state_is_clean(git: dict[str, Any]) -> bool:
    """`git diff --name-only` succeeds with rc=0 whether or not it has output."""
    return all(
        (git.get(key) or {}).get("returncode") == 0 and not str((git.get(key) or {}).get("stdout") or "").strip()
        for key in ("tracked_diff", "index_diff", "untracked")
    )


def source_identity() -> dict[str, Any]:
    git = common.git_state(DEFAULT_SOURCE_ROOT)
    commit = (git.get("commit") or {}).get("stdout", "").strip()
    clean = git_state_is_clean(git)
    return {"source_root": str(DEFAULT_SOURCE_ROOT), "expected_head": EXPECTED_HEAD, "head": commit, "head_matches": commit == EXPECTED_HEAD, "clean": clean, "git": git}


def identity_command(argv: list[str], env: dict[str, str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, check=False, env=env, timeout=30)
        return {"argv": argv, "environment": env, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "environment": env, "returncode": None, "stdout": "", "stderr": "", "exec_error": repr(exc)}


def local_library_identities(ldd: dict[str, Any]) -> list[dict[str, Any]]:
    if ldd.get("returncode") != 0:
        return []
    identities: list[dict[str, Any]] = []
    for line in str(ldd.get("stdout") or "").splitlines():
        match = re.match(r"^\s*(lib(?:llama|ggml)[^\s]*)\s+=>\s+(/[^\s]+)\s+\(", line)
        if match is None:
            continue
        resolved = Path(match.group(2)).resolve(strict=True)
        identity = common.stable_file_identity(resolved)
        identity["soname"] = match.group(1)
        identities.append(identity)
    return sorted(identities, key=lambda item: (str(item["soname"]), str(item["resolved_path"])))


def binary_identity() -> dict[str, Any]:
    environment = common.runtime_env(DEFAULT_BINARY)
    artifact = common.stable_file_identity(DEFAULT_BINARY)
    libraries = local_library_identities(
        identity_command(["ldd", str(DEFAULT_BINARY)], environment)
    )
    return {"binary": str(DEFAULT_BINARY), "binary_sha256": artifact.get("sha256"), "artifact": artifact,
            "server_version": identity_command([str(DEFAULT_BINARY), "--version"], environment),
            "local_llama_ggml_libraries": libraries, "environment": environment}


def harness_identity() -> dict[str, Any]:
    return common.stable_file_identity(Path(__file__).resolve())


def execution_binding(binary: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    return {
        "server": {"path": binary.get("binary"), "sha256": binary.get("binary_sha256"), "artifact": binary.get("artifact"),
                   "local_llama_ggml_libraries": binary.get("local_llama_ggml_libraries")},
        "models": {"target": model, "drafter": {"path": "/__laguna_dflash_forbidden__", "resolved_path": "/__laguna_dflash_forbidden__"}},
    }


def fixed_identities_valid(source: dict[str, Any], binary: dict[str, Any], model: dict[str, Any]) -> tuple[bool, str]:
    if not source.get("head_matches") or not source.get("clean"):
        return False, "candidate source HEAD is not pinned and clean"
    artifact = binary.get("artifact") or {}
    version = binary.get("server_version") or {}
    version_text = f"{version.get('stdout') or ''}\n{version.get('stderr') or ''}"
    libraries = binary.get("local_llama_ggml_libraries")
    if Path(str(binary.get("binary") or "")).resolve() != DEFAULT_BINARY.resolve() or artifact.get("stable") is not True or binary.get("binary_sha256") != EXPECTED_SERVER_SHA256:
        return False, "candidate server binary or local llama/ggml libraries are not immutably identified"
    if version.get("returncode") != 0 or f"version: {EXPECTED_SERVER_VERSION}" not in version_text:
        return False, "candidate server version does not match the pinned build"
    if not isinstance(libraries, list) or any(item.get("stable") is not True for item in libraries):
        return False, "candidate local shared-library identity is unstable"
    if {str(item.get("soname")): item.get("sha256") for item in libraries} != EXPECTED_LOCAL_LIBRARY_SHA256:
        return False, "candidate local llama/ggml library SHA256 pins differ"
    if Path(str(model.get("path") or "")).resolve() != DEFAULT_TARGET_MODEL.resolve() or model.get("stable") is not True or model.get("bytes") != TARGET_MODEL_BYTES or model.get("sha256") != TARGET_MODEL_SHA256:
        return False, "Laguna-S-2.1-UD-IQ2_M target identity does not match fixed pin"
    return True, "ok"


def ordered_cells(reps: int) -> list[dict[str, Any]]:
    """Latin-style cyclic ordering avoids always placing a setting first/last."""
    result: list[dict[str, Any]] = []
    for rep in range(1, reps + 1):
        rotation = (rep - 1) % len(CELLS)
        for cell in CELLS[rotation:] + CELLS[:rotation]:
            result.append({"cell": cell.name, "rep": rep, "port": PORT_BASE + len(result)})
    return result


def server_argv(args: argparse.Namespace, cell: Cell, port: int) -> list[str]:
    return [
        str(args.binary), "-m", str(args.target_model), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(args.context), "-ngl", "all", "-dev", "ROCm0", "-ot", "token_embd.weight=ROCm0",
        "-fa", "on" if cell.flash_attention else "off", "--cache-type-k", cell.cache_k,
        "--cache-type-v", cell.cache_v, "--seed", str(args.seed), "--temp", "0", "--top-k", "1",
        "--top-p", "1", "--jinja", "--reasoning", "off", "--reasoning-budget", "0", "-v",
    ]


def build_plan(args: argparse.Namespace, model_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "epyc.laguna_iq2_mi210_kv_sweep.plan.v1", "created_at": utc_now(), "execute": args.execute,
        "observation_only": True, "promotion_gate": False, "global_optimum_claim": False,
        "candidate": {"source": source_identity(), "binary": binary_identity(), "harness": harness_identity()}, "target_model": model_identity,
        "fixed_recipe": {"device": "ROCm0", "ngl": "all", "token_embedding": "ROCm0", "context": CONTEXT,
                         "seed": SEED, "temperature": 0, "top_k": 1, "top_p": 1, "reasoning": "off", "dflash": "forbidden"},
        "fixed_prompt_pack": [{"id": key, "text": value} for key, value in common.PROMPT_SPECS],
        "validators": {"primes_sum": 129, "nested_flatten": [1, 2, 3, 4, 5], "normalize_sum": 1.0},
        "reps_per_cell": REPS, "fresh_server_per_replicate": True,
        "counterbalance": "cyclic A,B,C; B,C,A; C,A,B; A,B,C; B,C,A",
        "cells": [{"name": cell.name, "cache_k": cell.cache_k, "cache_v": cell.cache_v, "flash_attention": cell.flash_attention} for cell in CELLS],
        "runs": ordered_cells(args.reps),
    }


def parse_log_residency(log_text: str, cell: Cell) -> dict[str, Any]:
    target = re.search(rf"loading model '{re.escape(str(DEFAULT_TARGET_MODEL))}'", log_text) is not None
    offload = re.search(r"offloaded 49/49 layers to GPU", log_text) is not None
    models = [float(value) for value in re.findall(r"ROCm0 model buffer size =\s*([0-9.]+) MiB", log_text)]
    kv_buffers = [(float(k), float(v)) for k, v in re.findall(rf"K \({re.escape(cell.cache_k)}\):\s*([0-9.]+) MiB, V \({re.escape(cell.cache_v)}\):\s*([0-9.]+) MiB", log_text)]
    positive_models = [value for value in models if value > 0]
    positive_kv = [pair for pair in kv_buffers if pair[0] > 0 and pair[1] > 0]
    return {"passed": target and offload and bool(positive_models) and bool(positive_kv), "target_model_load_exact": target,
            "full_target_offload": offload, "rocm0_model_buffers_mib": models, "positive_rocm0_model": bool(positive_models),
            "kv_buffers_mib": kv_buffers, "kv_types_and_positive_buffers": bool(positive_kv),
            "contract": "exact target, 49/49 GPU layers, positive ROCm0 model and requested K/V buffers"}


def record_from_response(response: dict[str, Any], prompt_id: str, prompt_index: int, lifecycle: dict[str, Any]) -> dict[str, Any]:
    finish = common.finish_reason_from_response(response)
    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("response lacks nonempty assistant content")
    sanity = common.response_sanity(content)
    semantic = common.semantic_validation(prompt_id, content)
    if not sanity["passed"] or not semantic["passed"]:
        raise RuntimeError(f"sanity/semantic gate failed: {sanity}; {semantic}")
    timings = common.timings_from_response(response, speculative=False)
    if timings["completion_tokens"] < MIN_COMPLETION_TOKENS:
        raise RuntimeError("completion token floor failed")
    return {"prompt_index": prompt_index, "prompt_id": prompt_id, "finish_reason": finish, "assistant_content_sha256": common.sha256_text(content),
            "response_sanity": sanity, "semantic_validation": semantic, "request_lifecycle": lifecycle, **timings}


def summarize_cell(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    def stats(metric: str) -> dict[str, Any]:
        values = [float(row[metric]) for row in ok]
        if not values or any(not math.isfinite(value) for value in values):
            return {"n": len(values), "median": None, "mad": None}
        median = float(statistics.median(values))
        return {"n": len(values), "median": median, "mad": float(statistics.median([abs(value - median) for value in values]))}
    return {"cell": name, "replicates": len(rows), "ok_replicates": len(ok), "all_ok": len(rows) == REPS and len(ok) == REPS,
            "prompt_ms": stats("prompt_ms"), "decode_ms": stats("decode_ms"), "prompt_tps": stats("prompt_tps"), "decode_tps": stats("decode_tps")}


def a_b_comparison(summaries: dict[str, dict[str, Any]], post_execution_identity_valid: bool) -> dict[str, Any]:
    """Compare only supported A/B settings; C is deliberately not part of this result."""
    unavailable = {
        "status": "unavailable",
        "a_cell": "A_q8_kv_fa_on",
        "b_cell": "B_f16_kv_fa_on",
        "decode_tps": {"a_median": None, "b_median": None, "b_over_a_ratio": None, "b_vs_a_percent": None},
        "prompt_tps": {"a_median": None, "b_median": None, "b_over_a_ratio": None, "b_vs_a_percent": None},
    }
    if not post_execution_identity_valid:
        return {**unavailable, "reason": "post-execution identity witness is invalid"}
    a, b = summaries.get("A_q8_kv_fa_on"), summaries.get("B_f16_kv_fa_on")
    if not isinstance(a, dict) or not isinstance(b, dict) or not a.get("all_ok") or not b.get("all_ok"):
        return {**unavailable, "reason": "A and B must each complete all five replicates"}

    def metric(name: str) -> dict[str, float] | None:
        a_value, b_value = (a.get(name) or {}).get("median"), (b.get(name) or {}).get("median")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0 for value in (a_value, b_value)):
            return None
        ratio = float(b_value) / float(a_value)
        return {"a_median": float(a_value), "b_median": float(b_value), "b_over_a_ratio": ratio, "b_vs_a_percent": (ratio - 1.0) * 100.0}

    decode, prompt = metric("decode_tps"), metric("prompt_tps")
    if decode is None or prompt is None:
        return {**unavailable, "reason": "A/B median throughput is absent, non-finite, or non-positive"}
    return {"status": "observed", "a_cell": "A_q8_kv_fa_on", "b_cell": "B_f16_kv_fa_on", "decode_tps": decode, "prompt_tps": prompt}


def bounded_best_observed(summaries: dict[str, dict[str, Any]], post_execution_identity_valid: bool = True) -> dict[str, Any]:
    if not post_execution_identity_valid:
        return {"status": "unavailable", "reason": "post-execution identity witness is invalid"}
    eligible = [
        summary
        for summary in summaries.values()
        if summary.get("all_ok")
        and isinstance((summary.get("decode_tps") or {}).get("median"), (int, float))
    ]
    if not eligible:
        return {
            "status": "unavailable",
            "reason": "no cell completed all five semantic, residency, and cleanup replicates",
        }
    winner = max(eligible, key=lambda summary: float(summary["decode_tps"]["median"]))
    return {
        "status": "observed",
        "scope": "bounded_to_the_three_predeclared_cells_not_a_global_optimum",
        "cell": winner["cell"],
        "decode_tps_median": winner["decode_tps"]["median"],
        "eligible_cells": [summary["cell"] for summary in eligible],
    }


def identity_witness_matches(pre: dict[str, Any], post: dict[str, Any]) -> tuple[bool, str]:
    """Identity drift invalidates all observations, even after a complete matrix."""
    for key in ("source", "binary", "target_model", "harness", "harness_snapshot", "binding"):
        if pre.get(key) != post.get(key):
            return False, f"post-execution {key} differs from pre-execution witness"
    return True, "ok"


def matrix_valid(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(rows) != len(CELLS) * REPS:
        return False, "incomplete matrix"
    for cell in CELLS:
        selected = [row for row in rows if row.get("cell") == cell.name]
        if sorted(row.get("rep") for row in selected) != list(range(1, REPS + 1)) or any(row.get("status") != "ok" for row in selected):
            return False, f"{cell.name} is incomplete or failed"
        for row in selected:
            records = row.get("records") or []
            if len(records) != len(common.PROMPT_SPECS) or [record.get("prompt_index") for record in records] != list(range(1, len(common.PROMPT_SPECS) + 1)) or [record.get("prompt_id") for record in records] != [prompt_id for prompt_id, _ in common.PROMPT_SPECS] or any(record.get("finish_reason") != "stop" or not record.get("semantic_validation", {}).get("passed") or not record.get("response_sanity", {}).get("passed") or not record.get("request_lifecycle", {}).get("fully_contained_valid") or not isinstance(record.get("request_lifecycle", {}).get("fully_contained_sample_count"), int) or record["request_lifecycle"]["fully_contained_sample_count"] < 1 for record in records):
                return False, f"{cell.name} failed prompt/residency gates"
            if not row.get("residency", {}).get("passed") or not row.get("cleanup", {}).get("dead") or not row.get("post_cleanup_clean") or not row.get("post_cleanup_vram_settled"):
                return False, f"{cell.name} lacks residency or cleanup proof"
    return True, "ok"


def poll_vram_settlement(
    before: dict[str, Any],
    port: int,
    *,
    timeout_s: float = SETTLEMENT_TIMEOUT_S,
    interval_s: float = SETTLEMENT_POLL_INTERVAL_S,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Require a clean process guard and settled valid ROCm evidence by deadline."""
    deadline = time.monotonic() + timeout_s
    samples: list[dict[str, Any]] = []
    while True:
        processes = common.process_snapshot()
        clean, reason = common.process_guard_clean(processes, port)
        rocm = common.collect_rocm_snapshot()
        valid = common.snapshot_is_valid(rocm)
        settled = valid and common.vram_settled(before, rocm)
        samples.append(
            {
                "attempt": len(samples) + 1,
                "captured_at": utc_now(),
                "processes": processes,
                "process_guard_clean": clean,
                "process_guard_reason": reason,
                "rocm": rocm,
                "rocm_valid": valid,
                "vram_settled": settled,
            }
        )
        if not clean:
            return False, reason, samples
        if settled:
            return True, "ok", samples
        if time.monotonic() >= deadline:
            return False, "ROCm VRAM did not settle before deadline", samples
        time.sleep(min(interval_s, max(0.0, deadline - time.monotonic())))


def run_replicate(args: argparse.Namespace, cell: Cell, rep: int, port: int, output_dir: Path, expected_binding: dict[str, Any]) -> dict[str, Any]:
    """Run one fail-closed cell while preserving evidence for later cells."""
    rep_dir = output_dir / "runs" / f"{cell.name}_rep{rep}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    argv = server_argv(args, cell, port)
    write_json(rep_dir / "server_argv.json", argv)
    write_json(rep_dir / "environment.json", {"exact_server_environment": common.runtime_env(args.binary), "scrubbed_parent_env_keys": common.scrubbed_parent_env_keys()})
    before = common.collect_rocm_snapshot()
    proc: subprocess.Popen[str] | None = None
    log: Any = None
    cleanup: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    residency: dict[str, Any] | None = None
    interrupted: KeyboardInterrupt | None = None
    result: dict[str, Any] = {"cell": cell.name, "rep": rep, "status": "error", "records": records, "residency": residency}
    try:
        clean, reason = common.process_guard_clean(common.process_snapshot(), port)
        if not clean or not common.snapshot_is_valid(before):
            raise RuntimeError(reason if not clean else "pre-launch ROCm evidence failed")
        log = (rep_dir / "server.stderr").open("w", encoding="utf-8")
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=log, text=True, start_new_session=True, env=common.runtime_env(args.binary))
        common.wait_for_health(port, args.startup_timeout)
        for index, (prompt_id, prompt) in enumerate(common.PROMPT_SPECS, 1):
            body = common.request_body(prompt, args)
            response, _elapsed, lifecycle = common.query_with_live_samples(port, body, args.request_timeout, proc.pid, args.binary, index, expected_binding=expected_binding, require_drafter=False)
            records.append(record_from_response(response, prompt_id, index, lifecycle))
        residency = parse_log_residency((rep_dir / "server.stderr").read_text(encoding="utf-8", errors="replace"), cell)
        if not residency["passed"]:
            raise RuntimeError(f"residency proof failed: {residency}")
        prompt_ms, decode_ms = sum(row["prompt_ms"] for row in records), sum(row["decode_ms"] for row in records)
        result = {"cell": cell.name, "rep": rep, "status": "ok", "records": records, "residency": residency,
                  "prompt_ms": prompt_ms, "decode_ms": decode_ms, "prompt_tps": sum(row["prompt_tokens"] for row in records) / (prompt_ms / 1000),
                  "decode_tps": sum(row["completion_tokens"] for row in records) / (decode_ms / 1000)}
    except Exception as exc:  # preserve exact failure evidence
        result = {
            "cell": cell.name,
            "rep": rep,
            "status": "error",
            "error": repr(exc),
            "records": records,
            "residency": residency,
        }
    except KeyboardInterrupt as exc:
        interrupted = exc
        result = {
            "cell": cell.name,
            "rep": rep,
            "status": "interrupted",
            "error": repr(exc),
            "records": records,
            "residency": residency,
        }
    finally:
        cleanup_interrupt: KeyboardInterrupt | None = None
        previous_handler = signal.getsignal(signal.SIGINT)

        def defer_sigint(_signum: int, _frame: Any) -> None:
            nonlocal cleanup_interrupt
            cleanup_interrupt = KeyboardInterrupt()

        signal.signal(signal.SIGINT, defer_sigint)
        try:
            if proc is not None:
                cleanup = common.terminate(proc)
            if log is not None:
                log.close()
            result["cleanup"] = cleanup
            result["rocm_before"] = before
            if cleanup and cleanup.get("dead"):
                settled, settlement_reason, settlement_samples = poll_vram_settlement(before, port)
                result["settlement_samples"] = settlement_samples
                result["post_cleanup_clean"] = settlement_samples[-1]["process_guard_clean"] if settlement_samples else False
                result["post_cleanup_reason"] = settlement_reason
                result["post_cleanup_vram_settled"] = settled
                result["post_cleanup_processes"] = settlement_samples[-1]["processes"] if settlement_samples else None
                result["rocm_after"] = settlement_samples[-1]["rocm"] if settlement_samples else None
            else:
                result["settlement_samples"] = []
                result["post_cleanup_clean"] = False
                result["post_cleanup_reason"] = "server death proof failed"
                result["post_cleanup_vram_settled"] = False
                result["post_cleanup_processes"] = None
                result["rocm_after"] = None
            if not cleanup or not cleanup.get("dead") or not result["post_cleanup_clean"] or not result["post_cleanup_vram_settled"]:
                result["status"] = "cleanup_failed"
        except KeyboardInterrupt:
            cleanup_interrupt = KeyboardInterrupt()
            result["status"] = "cleanup_failed"
            result["cleanup_evidence_error"] = "KeyboardInterrupt while collecting cleanup evidence"
        except Exception as exc:  # cleanup evidence must not prevent durable failure output
            result["status"] = "cleanup_failed"
            result["cleanup_evidence_error"] = repr(exc)
        finally:
            write_json(rep_dir / "result.json", result)
            signal.signal(signal.SIGINT, previous_handler)
        if cleanup_interrupt is not None and interrupted is None:
            interrupted = cleanup_interrupt
    if interrupted is not None:
        raise interrupted
    return result


def execute(args: argparse.Namespace, output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    source, binary, model, harness = source_identity(), binary_identity(), common.immutable_model_identity(args.target_model), harness_identity()
    identities_valid, identity_reason = fixed_identities_valid(source, binary, model)
    harness_snapshot = output_dir / "harness_source.py"
    shutil.copy2(Path(__file__).resolve(), harness_snapshot)
    captured_harness = common.stable_file_identity(harness_snapshot)
    harness_valid = harness.get("stable") is True and captured_harness.get("stable") is True and harness.get("sha256") == captured_harness.get("sha256")
    expected_binding = execution_binding(binary, model)
    pre_execution_identity = {"source": source, "binary": binary, "target_model": model, "harness": harness, "harness_snapshot": captured_harness, "binding": expected_binding}
    write_json(output_dir / "identities.json", {**pre_execution_identity, "harness_valid": harness_valid, "valid": identities_valid and harness_valid, "reason": identity_reason})
    results: list[dict[str, Any]] = []
    if identities_valid and harness_valid:
        cells = {cell.name: cell for cell in CELLS}
        for run in plan["runs"]:
            result = run_replicate(args, cells[run["cell"]], int(run["rep"]), int(run["port"]), output_dir, expected_binding)
            results.append(result)
    else:
        results.append({"status": "error", "error": identity_reason})
    complete, reason = matrix_valid(results)
    summaries = {cell.name: summarize_cell([row for row in results if row.get("cell") == cell.name], cell.name) for cell in CELLS}
    post_source, post_binary = source_identity(), binary_identity()
    post_model, post_harness = common.immutable_model_identity(args.target_model), harness_identity()
    post_snapshot = common.stable_file_identity(harness_snapshot)
    post_binding = execution_binding(post_binary, post_model)
    post_execution_identity = {"source": post_source, "binary": post_binary, "target_model": post_model, "harness": post_harness, "harness_snapshot": post_snapshot, "binding": post_binding}
    post_pins_valid, post_pins_reason = fixed_identities_valid(post_source, post_binary, post_model)
    post_harness_valid = post_harness.get("stable") is True and post_snapshot.get("stable") is True and post_harness.get("sha256") == post_snapshot.get("sha256")
    witness_matches, witness_reason = identity_witness_matches(pre_execution_identity, post_execution_identity)
    post_execution_identity_valid = post_pins_valid and post_harness_valid and witness_matches
    post_execution_identity_reason = "ok" if post_execution_identity_valid else (post_pins_reason if not post_pins_valid else ("post-execution harness snapshot is invalid" if not post_harness_valid else witness_reason))
    return {"schema": "epyc.laguna_iq2_mi210_kv_sweep.summary.v1", "created_at": utc_now(), "status": "ok" if identities_valid and harness_valid and complete and post_execution_identity_valid else "failed",
            "observation_only": True, "promotion_gate": False, "global_optimum_claim": False, "candidate": {"source": source, "binary": binary, "harness": harness, "harness_snapshot": captured_harness},
            "target_model": model, "exact_plan": plan, "results": results, "matrix_cardinality_valid": complete,
            "matrix_cardinality_reason": reason, "cell_summaries": summaries,
            "a_vs_b": a_b_comparison(summaries, post_execution_identity_valid),
            "post_execution_identity": post_execution_identity,
            "post_execution_identity_valid": post_execution_identity_valid,
            "post_execution_identity_reason": post_execution_identity_reason,
            "bounded_best_observed": bounded_best_observed(summaries, post_execution_identity_valid),
            "cleanup_contract": "PID, process group, port, KFD ownership, and VRAM settlement are required per cell"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--startup-timeout", type=int, default=common.DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--request-timeout", type=int, default=common.DEFAULT_REQUEST_TIMEOUT_S)
    args = parser.parse_args(argv)
    if (args.binary, args.source_root, args.target_model, args.reps, args.context, args.max_tokens, args.seed) != (DEFAULT_BINARY, DEFAULT_SOURCE_ROOT, DEFAULT_TARGET_MODEL, REPS, CONTEXT, MAX_TOKENS, SEED):
        parser.error("candidate, target identity, five reps, context, and replay seed are fixed")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute:
        args.output_dir = args.output_dir / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = common.immutable_model_identity(args.target_model) if args.execute else {"path": str(args.target_model), "bytes": TARGET_MODEL_BYTES, "sha256": TARGET_MODEL_SHA256}
    plan = build_plan(args, model)
    write_json(args.output_dir / "plan.json", plan)
    if not args.execute:
        write_json(args.output_dir / "summary.json", {"schema": "epyc.laguna_iq2_mi210_kv_sweep.summary.v1", "status": "prepared_no_inference", "observation_only": True, "promotion_gate": False, "global_optimum_claim": False, "exact_plan": plan})
        return 0
    summary = execute(args, args.output_dir, plan)
    write_json(args.output_dir / "summary.json", summary)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
