#!/usr/bin/env python3
from __future__ import annotations

"""Generate a no-inference clean-window benchmark plan.

The output groups pending benchmark work by physical model path so a live
window can keep each GGUF resident while draining all compatible tasks.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BENCHMARK_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from dataset_adapters import get_adapter
from lib.registry import load_registry
from suites import get_suites_for_role

RUN_BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "run_benchmark.py"
ROPE_PROBE_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "rope_position_probe.py"
SHORT_MK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "short_mk_voting.py"
DS_E1_KV_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "ds_e1_kv_measurements.sh"
SERVER_NP_SWEEP_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "server_np_sweep.py"
E2_EVAL_DRIVER_AB_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "e2_eval_driver_ab.py"
ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
XMAS_LIVE_AB_SCRIPT = ORCHESTRATOR_ROOT / "scripts" / "benchmark" / "xmas_live_ab.py"
XMAS_HELDOUT_PROMPTS = (
    ORCHESTRATOR_ROOT
    / "benchmarks"
    / "results"
    / "runs"
    / "xmas_live_ab"
    / "20260618-heldout-resilient"
    / "prompts.jsonl"
)
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "docs" / "data" / "clean_window_measurement_manifest.json"
DEFAULT_COMMANDS_PATH = PROJECT_ROOT / "docs" / "data" / "clean_window_measurement_commands.sh"
DEFAULT_LIVE_REGISTRY_PATH = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml")
SOURCE_HANDOFF = "/mnt/raid0/llm/epyc-root/handoffs/active/bulk-inference-campaign.md"

AA_TARGETS = {
    "G10": ["architect_general"],
    "G11": ["frontdoor", "worker_general"],
}
K_MEM_TARGETS = ["ingest_long_context"]
K_ROPE_TARGETS = ["frontdoor", "worker_general", "architect_general", "ingest_long_context"]
G5_TARGETS = ["frontdoor", "worker_general", "architect_general"]
ROPE_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_role_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_port_overrides(values: list[str]) -> dict[str, int]:
    ports: dict[str, int] = {}
    for item in values:
        role, sep, raw_port = item.partition("=")
        if not sep:
            raise ValueError(f"expected ROLE=PORT, got {item!r}")
        ports[role.strip()] = int(raw_port)
    return ports


def _parse_context_overrides(values: list[str]) -> dict[str, int]:
    contexts: dict[str, int] = {}
    for item in values:
        role, sep, raw_context = item.partition("=")
        if not sep:
            raise ValueError(f"expected ROLE=CONTEXT, got {item!r}")
        contexts[role.strip()] = int(raw_context)
    return contexts


def _load_live_registry(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_model_path(registry_data: dict[str, Any], role: str) -> str | None:
    role_config = registry_data.get("roles", {}).get(role, {})
    model = role_config.get("model", {})
    raw_path = model.get("path")
    if not raw_path:
        return None
    if str(raw_path).startswith("/"):
        return str(raw_path)
    model_base = registry_data.get("runtime_defaults", {}).get("model_base_path", "/mnt/raid0/llm/lmstudio/models")
    return str(Path(model_base) / str(raw_path))


def _role_config_from_data(registry_data: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    if not registry_data:
        return None
    return registry_data.get("roles", {}).get(role)


def _dataset_total(suite: str) -> tuple[int | None, str, str | None]:
    try:
        adapter = get_adapter(suite)
        total = adapter.total_available
    except Exception as exc:  # noqa: BLE001
        return None, "blocked", str(exc)
    return int(total), "ready" if total else "blocked", None if total else "no questions available"


def _role_metadata(registry, role: str, live_registry: dict[str, Any] | None = None) -> dict[str, Any]:
    role_config = registry.get_role_config(role) or {}
    benchmark_model_path = registry.get_model_path(role)
    live_role_config = _role_config_from_data(live_registry, role)
    live_model_path = _resolve_model_path(live_registry, role) if live_role_config and live_registry else None
    model_path = live_model_path or benchmark_model_path
    model_exists = bool(model_path and Path(model_path).exists())
    server = _server_for_role(live_registry, role) or _server_for_role(registry.data, role)
    live_max_context = None
    if live_role_config:
        live_model = live_role_config.get("model", {})
        live_max_context = live_model.get("max_context") or live_model.get("ctx_max")
    benchmark_max_context = registry.get_max_context(role) if hasattr(registry, "get_max_context") else None
    registry_mismatch = bool(live_model_path and benchmark_model_path and live_model_path != benchmark_model_path)
    return {
        "role": role,
        "tier": live_role_config.get("tier") if live_role_config else registry.get_tier(role) if hasattr(registry, "get_tier") else role_config.get("tier"),
        "architecture": (
            live_role_config.get("model", {}).get("architecture")
            if live_role_config
            else registry.get_architecture(role) if hasattr(registry, "get_architecture") else None
        ),
        "model_path": model_path,
        "benchmark_model_path": benchmark_model_path,
        "live_model_path": live_model_path,
        "model_path_source": "live_registry" if live_model_path else "benchmark_registry",
        "benchmark_registry_mismatch": registry_mismatch,
        "model_exists": model_exists,
        "max_context": live_max_context or benchmark_max_context,
        "server": server,
    }


def _server_for_role(registry_data: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    server_mode = registry_data.get("server_mode", {}) if registry_data else {}
    for name, config in server_mode.items():
        if not isinstance(config, dict):
            continue
        if name == role or config.get("model_role") == role or role in config.get("shared_with", []):
            return {
                "name": name,
                "url": config.get("url"),
                "port": config.get("port"),
                "source": "registry.server_mode",
            }
    return None


def _benchmark_command(role: str, suite: str, *, server_mode: bool = False) -> str:
    argv = [
        sys.executable,
        str(RUN_BENCHMARK_SCRIPT),
        "--model",
        role,
        "--suite",
        suite,
        "--new-run",
        "--server-mode",
        "--skip-speed-tests",
    ]
    return shlex.join(argv)


def _rope_command(role: str, context_length: int, port: int, output_root: Path) -> str:
    out = output_root / "rope_probe" / role / f"ctx_{context_length}.json"
    return shlex.join([
        sys.executable,
        str(ROPE_PROBE_SCRIPT),
        "--api",
        "chat",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--context-length",
        str(context_length),
        "--n-samples",
        "100",
        "--seed",
        "42",
        "--out",
        str(out),
    ])


def _short_mk_command(role: str, port: int, output_root: Path) -> str:
    out = output_root / "short_mk_voting" / f"{role}.json"
    return shlex.join([
        sys.executable,
        str(SHORT_MK_SCRIPT),
        "--role",
        role,
        "--host",
        "127.0.0.1",
        "--model-port",
        str(port),
        "--suites",
        "gpqa",
        "math",
        "--sample-per-suite",
        "20",
        "--k",
        "3",
        "--m",
        "3",
        "--sequential",
        "--output",
        str(out),
    ])


def _ds_e1_kv_command() -> str:
    return shlex.join(["bash", str(DS_E1_KV_SCRIPT), "--execute"])


def _e1_batched_decode_command() -> str:
    argv = [
        "uv",
        "run",
        "--extra",
        "benchmark",
        "python",
        "scripts/benchmark/server_np_sweep.py",
        "--run-id",
        "__RUN_ID__",
        "--prompt-limit",
        "43",
        "--prompt-seed",
        "42",
        "--tier",
        "1",
        "--np-levels",
        "1,2,4,8,16",
    ]
    command = shlex.join(argv).replace("__RUN_ID__", '"$run_id"')
    return f"cd {shlex.quote(str(PROJECT_ROOT))} && run_id=e1-pbench3-$(date -u +%Y%m%dT%H%M%SZ) && {command}"


def _e2_eval_driver_plan_command() -> str:
    argv = [
        "uv",
        "run",
        "--extra",
        "benchmark",
        "python",
        "scripts/benchmark/e2_eval_driver_ab.py",
        "--run-id",
        "__RUN_ID__",
        "--prompt-limit",
        "43",
        "--prompt-seed",
        "42",
        "--tier",
        "1",
        "--batch-np",
        "8",
        "--current-concurrency",
        "3",
    ]
    command = shlex.join(argv).replace("__RUN_ID__", '"$run_id"')
    return f"cd {shlex.quote(str(PROJECT_ROOT))} && run_id=e2-pbench3-$(date -u +%Y%m%dT%H%M%SZ) && {command}"


def _xmas_live_ab_command() -> str:
    argv = [
        "uv",
        "run",
        "python",
        "scripts/benchmark/xmas_live_ab.py",
        "--prompts",
        "benchmarks/results/runs/xmas_live_ab/20260618-heldout-resilient/prompts.jsonl",
        "--reps",
        "2",
        "--host-quiet-confirmed",
        "--output",
    ]
    output = "benchmarks/results/runs/xmas_live_ab/$(date -u +%Y%m%dT%H%M%SZ)-constrained-policy"
    return f"cd {shlex.quote(str(ORCHESTRATOR_ROOT))} && {shlex.join(argv)} {output}"


def _suite_entry(
    registry,
    *,
    package: str,
    role: str,
    suite: str,
    server_mode: bool = False,
    live_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total, dataset_status, dataset_note = _dataset_total(suite)
    meta = _role_metadata(registry, role, live_registry)
    suites = get_suites_for_role(role, registry)
    notes: list[str] = []
    if suite not in suites:
        notes.append(f"{suite} is not mapped to role {role}")
    if not meta["model_exists"]:
        notes.append("model path missing")
    if meta["benchmark_registry_mismatch"]:
        notes.append(
            "benchmark registry model path differs from live registry; run_benchmark.py would not measure the live role"
        )
    if dataset_note:
        notes.append(dataset_note)
    status = (
        "ready"
        if meta["model_exists"] and dataset_status == "ready" and suite in suites and not meta["benchmark_registry_mismatch"]
        else "blocked"
    )
    return {
        "package": package,
        "kind": "run_benchmark_suite",
        "role": role,
        "suite": suite,
        "question_count": total,
        "status": status,
        "command": _benchmark_command(role, suite, server_mode=server_mode),
        "notes": notes,
        "model": meta,
    }


def _rope_entries(
    registry,
    role: str,
    ports: dict[str, int],
    server_contexts: dict[str, int],
    output_root: Path,
    live_registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    meta = _role_metadata(registry, role, live_registry)
    port = ports.get(role)
    port_source = "override"
    if port is None and meta["server"]:
        port = meta["server"].get("port")
        port_source = meta["server"].get("source", "registry")

    entries: list[dict[str, Any]] = []
    for context_length in ROPE_CONTEXT_LENGTHS:
        notes: list[str] = []
        if not meta["model_exists"]:
            notes.append("model path missing")
        if port is None:
            notes.append("server port unavailable; pass --server-port ROLE=PORT after verifying live topology")
        if role in server_contexts:
            if context_length >= server_contexts[role]:
                notes.append(
                    f"context {context_length} needs chat-template headroom below live server context {server_contexts[role]}"
                )
        elif meta["max_context"] and context_length > meta["max_context"]:
            notes.append(f"context {context_length} exceeds registered max_context {meta['max_context']}")
        status = "ready" if not notes else "blocked"
        entries.append({
            "package": "K-ROPE-1",
            "kind": "rope_position_probe",
            "role": role,
            "context_length": context_length,
            "status": status,
            "command": _rope_command(role, context_length, int(port), output_root) if port is not None else None,
            "notes": notes,
            "port_source": port_source if port is not None else None,
            "model": meta,
        })
    return entries


def _g5_entry(
    registry,
    role: str,
    live_registry: dict[str, Any] | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    meta = _role_metadata(registry, role, live_registry)
    runner_candidates = [
        SHORT_MK_SCRIPT,
        PROJECT_ROOT / "scripts" / "benchmark" / "short_m_at_k.py",
    ]
    existing = [str(path) for path in runner_candidates if path.exists()]
    output_root = output_root or (PROJECT_ROOT / "data" / "benchmarks" / "clean_window")
    port = meta["server"].get("port") if meta["server"] else None
    notes = []
    if not existing:
        notes.append("no short-m@k voting runner found; G5 needs runner wiring before clean-window execution")
    if not meta["model_exists"]:
        notes.append("model path missing")
    if port is None:
        notes.append("server port unavailable; pass --server-port ROLE=PORT after verifying live topology")
    status = "ready" if not notes else "blocked"
    return {
        "package": "G5",
        "kind": "short_mk_voting",
        "role": role,
        "suite_candidates": ["gpqa", "math"],
        "grouping": {
            "k": 3,
            "m": 3,
            "vote_rule": "majority",
            "run_style": "sequential clean-window default; pass --parallel only for isolated capacity windows",
        },
        "status": status,
        "command": _short_mk_command(role, int(port), output_root) if port is not None and existing else None,
        "notes": notes,
        "runner_candidates": [str(path) for path in runner_candidates],
        "model": meta,
    }


def _ds_e1_entry() -> dict[str, Any]:
    notes = [
        "execute mode fails closed if AutoPilot, live llama-server processes, or port 8194 are active",
        "writes data/dynamic_stack/ds_e1_kv_measurements_<timestamp>/kv_measurements.csv for the DS-E1 packet",
    ]
    if not DS_E1_KV_SCRIPT.exists():
        notes.append("DS-E1 KV measurement harness missing")
    status = "ready" if DS_E1_KV_SCRIPT.exists() else "blocked"
    return {
        "package": "DS-E1",
        "kind": "production_kv_measurements",
        "role": "dynamic_stack",
        "status": status,
        "command": _ds_e1_kv_command() if DS_E1_KV_SCRIPT.exists() else None,
        "notes": notes,
        "model": {
            "role": "dynamic_stack",
            "tier": "clean_window",
            "architecture": "production_stack_kv_probe",
            "model_path": "clean-window-harness:ds-e1-kv",
            "benchmark_model_path": None,
            "live_model_path": None,
            "model_path_source": "harness",
            "benchmark_registry_mismatch": False,
            "model_exists": DS_E1_KV_SCRIPT.exists(),
            "max_context": None,
            "server": {
                "name": "ds_e1_kv_measurements",
                "url": "http://127.0.0.1:8194",
                "port": 8194,
                "source": "scripts/benchmark/ds_e1_kv_measurements.sh",
            },
        },
    }


def _e1_batched_decode_entry() -> dict[str, Any]:
    notes = [
        "P-BENCH-3 direct serving sweep over qwen36_q8_0 and qwen36_27b_q8 with -np 1,2,4,8,16",
        "server_np_sweep.py refuses decision-grade execution when AutoPilot, live llama-server processes, or host-health warnings are present",
        "writes data/batched_decode/<run-id>/{manifest.json,selected_prompts.jsonl,summary.csv,recommendations.json,cells.jsonl,events.jsonl}",
    ]
    if not SERVER_NP_SWEEP_SCRIPT.exists():
        notes.append("E1 server -np sweep harness missing")
    status = "ready" if SERVER_NP_SWEEP_SCRIPT.exists() else "blocked"
    return {
        "package": "E1",
        "kind": "batched_decode_np_sweep",
        "role": "eval_serving",
        "status": status,
        "command": _e1_batched_decode_command() if status == "ready" else None,
        "notes": notes,
        "model": {
            "role": "eval_serving",
            "tier": "clean_window",
            "architecture": "continuous_batching_serving_sweep",
            "model_path": "clean-window-harness:server-np-sweep",
            "benchmark_model_path": None,
            "live_model_path": None,
            "model_path_source": "harness",
            "benchmark_registry_mismatch": False,
            "model_exists": SERVER_NP_SWEEP_SCRIPT.exists(),
            "max_context": 32768,
            "server": {
                "name": "server_np_sweep",
                "url": "http://127.0.0.1:<dynamic>",
                "port": None,
                "source": str(SERVER_NP_SWEEP_SCRIPT),
            },
        },
    }


def _e2_eval_driver_ab_entry() -> dict[str, Any]:
    notes = [
        "Queue-2 coordinator for current EvalTower fan-out versus one full continuous-batching server at -np 8",
        "command writes an E2 run manifest plus commands.sh; execute that generated commands.sh only in the same clean window",
        "generated E2 plan marks decision_grade=false and comments arm commands when host-health warnings are present",
        "summary step after arms complete: uv run --extra benchmark python scripts/benchmark/e2_eval_driver_ab.py --summarize-run <run-dir>",
    ]
    if not E2_EVAL_DRIVER_AB_SCRIPT.exists():
        notes.append("E2 eval-driver A/B planner missing")
    status = "ready" if E2_EVAL_DRIVER_AB_SCRIPT.exists() else "blocked"
    return {
        "package": "E2",
        "kind": "eval_driver_ab_plan",
        "role": "eval_serving",
        "status": status,
        "command": _e2_eval_driver_plan_command() if status == "ready" else None,
        "notes": notes,
        "model": {
            "role": "eval_serving",
            "tier": "clean_window",
            "architecture": "eval_driver_ab_coordinator",
            "model_path": "clean-window-harness:e2-eval-driver-ab",
            "benchmark_model_path": None,
            "live_model_path": None,
            "model_path_source": "harness",
            "benchmark_registry_mismatch": False,
            "model_exists": E2_EVAL_DRIVER_AB_SCRIPT.exists(),
            "max_context": None,
            "server": {
                "name": "e2_eval_driver_ab",
                "url": None,
                "port": None,
                "source": str(E2_EVAL_DRIVER_AB_SCRIPT),
            },
        },
    }


def _xmas_live_ab_entry() -> dict[str, Any]:
    notes = [
        "requires attested quiet window; runner refuses AutoPilot and competing benchmark coordinators",
        "summary.json must carry xmas_policy=incumbent_constrained_cheapfirst_v2 and decision.status=promote_candidate",
        "writes benchmarks/results/runs/xmas_live_ab/<timestamp>-constrained-policy/{meta.json,results.jsonl,summary.json,report.md}",
    ]
    if not XMAS_LIVE_AB_SCRIPT.exists():
        notes.append("X-MAS held-out A/B runner missing")
    if not XMAS_HELDOUT_PROMPTS.exists():
        notes.append("X-MAS held-out prompt manifest missing")
    status = "ready" if XMAS_LIVE_AB_SCRIPT.exists() and XMAS_HELDOUT_PROMPTS.exists() else "blocked"
    return {
        "package": "X-MAS",
        "kind": "constrained_policy_heldout_ab",
        "role": "xmas_routing",
        "status": status,
        "command": _xmas_live_ab_command() if status == "ready" else None,
        "notes": notes,
        "model": {
            "role": "xmas_routing",
            "tier": "clean_window",
            "architecture": "function_axis_routing_ab",
            "model_path": "clean-window-harness:xmas-constrained-policy",
            "benchmark_model_path": None,
            "live_model_path": None,
            "model_path_source": "harness",
            "benchmark_registry_mismatch": False,
            "model_exists": status == "ready",
            "max_context": None,
            "server": {
                "name": "xmas_live_ab",
                "url": None,
                "port": None,
                "source": str(XMAS_LIVE_AB_SCRIPT),
            },
        },
    }


def build_manifest(
    *,
    aa_roles: list[str] | None = None,
    k_mem_roles: list[str] | None = None,
    k_rope_roles: list[str] | None = None,
    g5_roles: list[str] | None = None,
    server_ports: dict[str, int] | None = None,
    server_contexts: dict[str, int] | None = None,
    output_root: Path | None = None,
    live_registry_path: Path | None = DEFAULT_LIVE_REGISTRY_PATH,
) -> dict[str, Any]:
    registry = load_registry()
    live_registry = _load_live_registry(live_registry_path)
    server_ports = server_ports or {}
    server_contexts = server_contexts or {}
    output_root = output_root or (PROJECT_ROOT / "benchmarks" / "results" / "clean_window")
    entries: list[dict[str, Any]] = []

    aa_role_set = aa_roles if aa_roles is not None else [role for roles in AA_TARGETS.values() for role in roles]
    for role in aa_role_set:
        package = next((name for name, roles in AA_TARGETS.items() if role in roles), "G10/G11")
        entries.append(_suite_entry(registry, package=package, role=role, suite="omniscience", live_registry=live_registry))

    for role in k_mem_roles if k_mem_roles is not None else K_MEM_TARGETS:
        entries.append(
            _suite_entry(
                registry,
                package="K-MEM-1",
                role=role,
                suite="tulving_episodic",
                server_mode=True,
                live_registry=live_registry,
            )
        )

    for role in k_rope_roles if k_rope_roles is not None else K_ROPE_TARGETS:
        entries.extend(_rope_entries(registry, role, server_ports, server_contexts, output_root, live_registry=live_registry))

    for role in g5_roles if g5_roles is not None else G5_TARGETS:
        entries.append(_g5_entry(registry, role, live_registry=live_registry, output_root=output_root))

    entries.append(_e2_eval_driver_ab_entry())
    entries.append(_e1_batched_decode_entry())
    entries.append(_ds_e1_entry())
    entries.append(_xmas_live_ab_entry())

    groups: dict[str, dict[str, Any]] = {}
    for entry in entries:
        model_path = entry["model"].get("model_path") or f"missing:{entry['role']}"
        group = groups.setdefault(model_path, {
            "model_path": model_path,
            "roles": sorted(set()),
            "entries": [],
        })
        group["roles"] = sorted(set(group["roles"]) | {entry["role"]})
        group["entries"].append(entry)

    ordered_groups = sorted(groups.values(), key=lambda item: item["model_path"])
    for group in ordered_groups:
        group["summary"] = {
            "entries": len(group["entries"]),
            "ready": sum(1 for item in group["entries"] if item["status"] == "ready"),
            "blocked": sum(1 for item in group["entries"] if item["status"] != "ready"),
        }

    generated_at = _utc_now()
    registry_path = Path(getattr(registry, "registry_path", PROJECT_ROOT / "orchestration" / "model_registry.yaml"))
    return {
        "generated_at": generated_at,
        "run_id": f"clean-window-{generated_at.replace(':', '').replace('+', 'Z')}",
        "kind": "model_batched_clean_window",
        "purpose": "model-batched clean-window plan for G5/G10/G11/K-MEM-1/K-ROPE-1 plus Queue-2 E1/E2, DS-E1 KV measurements, and X-MAS constrained-policy A/B",
        "source_handoff": SOURCE_HANDOFF,
        "output_root": str(output_root),
        "topology": {
            "required_topology_hash": _file_sha256(registry_path),
            "topology_artifact": str(registry_path),
            "live_registry_artifact": str(live_registry_path) if live_registry_path else None,
            "live_registry_hash": _file_sha256(live_registry_path) if live_registry_path else None,
            "live_affinity_verified": False,
            "affinity_artifact": None,
            "matrix_status": "not_required",
        },
        "attestation": {
            "research_head": _git_head(),
            "generated_by": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "inference_started": False,
        },
        "pass_fail_gate": "All ready commands require a clean inference window, successful preflight, and per-task result aggregation; blocked entries must remain non-executable.",
        "next_action": "After AutoPilot quiesces, verify live topology/affinity, then run ready commands model group by model group.",
        "journal_quarantine_rule": "Do not mix these results into calibration journals if another benchmark or AutoPilot run is concurrently mutating the live stack.",
        "entries": entries,
        "groups": ordered_groups,
        "summary": {
            "entries_total": len(entries),
            "entries_ready": sum(1 for item in entries if item["status"] == "ready"),
            "entries_blocked": sum(1 for item in entries if item["status"] != "ready"),
            "models_total": len(ordered_groups),
        },
    }


def _write_outputs(manifest: dict[str, Any], manifest_path: Path, commands_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# Generated at {manifest['generated_at']}",
        "# Review live topology before running commands with direct --port values.",
        "",
    ]
    for group in manifest["groups"]:
        lines.append(f"# model_path: {group['model_path']}")
        lines.append(f"# roles: {', '.join(group['roles'])}")
        for entry in group["entries"]:
            label = f"{entry['package']} {entry['role']} {entry['kind']}"
            if entry.get("suite"):
                label += f" {entry['suite']}"
            if entry.get("context_length"):
                label += f" ctx={entry['context_length']}"
            lines.append(f"# {label} [{entry['status']}]")
            if entry["status"] == "ready" and entry.get("command"):
                lines.append(entry["command"])
            else:
                if entry.get("command"):
                    lines.append(f"# blocked: {entry['command']}")
                for note in entry.get("notes", []):
                    lines.append(f"# note: {note}")
            lines.append("")
    commands_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _print_summary(manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    print(f"Generated: {manifest['generated_at']}")
    print(f"Entries: {summary['entries_total']} ready={summary['entries_ready']} blocked={summary['entries_blocked']}")
    print(f"Model groups: {summary['models_total']}")
    for group in manifest["groups"]:
        gs = group["summary"]
        print(f"  - {', '.join(group['roles'])}: {gs['ready']}/{gs['entries']} ready")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a no-inference clean-window benchmark plan")
    parser.add_argument("--aa-roles", help="Comma-separated AA-Omniscience role override")
    parser.add_argument("--k-mem-roles", help="Comma-separated K-MEM role override")
    parser.add_argument("--k-rope-roles", help="Comma-separated K-ROPE role override")
    parser.add_argument("--g5-roles", help="Comma-separated G5 role override")
    parser.add_argument("--server-port", action="append", default=[], help="Direct probe port override as ROLE=PORT")
    parser.add_argument("--server-context", action="append", default=[], help="Live server context override as ROLE=CONTEXT")
    parser.add_argument("--live-registry-path", type=Path, default=DEFAULT_LIVE_REGISTRY_PATH)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "benchmarks" / "results" / "clean_window")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--commands-path", type=Path, default=DEFAULT_COMMANDS_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing files")
    parser.add_argument("--validate", action="store_true", help="Exit non-zero if any entry is blocked")
    args = parser.parse_args()

    manifest = build_manifest(
        aa_roles=_parse_role_list(args.aa_roles, [role for roles in AA_TARGETS.values() for role in roles]),
        k_mem_roles=_parse_role_list(args.k_mem_roles, K_MEM_TARGETS),
        k_rope_roles=_parse_role_list(args.k_rope_roles, K_ROPE_TARGETS),
        g5_roles=_parse_role_list(args.g5_roles, G5_TARGETS),
        server_ports=_parse_port_overrides(args.server_port),
        server_contexts=_parse_context_overrides(args.server_context),
        output_root=args.output_root,
        live_registry_path=args.live_registry_path,
    )
    _print_summary(manifest)

    if not args.dry_run:
        _write_outputs(manifest, args.manifest_path, args.commands_path)
        print(f"\nWrote manifest: {args.manifest_path}")
        print(f"Wrote commands: {args.commands_path}")

    if args.validate and manifest["summary"]["entries_blocked"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
