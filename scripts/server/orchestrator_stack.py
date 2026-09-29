#!/usr/bin/env python3
"""Unified orchestrator stack launcher.

Launches all models + orchestrator with granular reload support.

Usage:
    orchestrator_stack.py start [--hot-only] [--include-warm ROLE...] [--only ROLE...] [--dev]
    orchestrator_stack.py stop [--all | COMPONENT...]
    orchestrator_stack.py reload COMPONENT...
    orchestrator_stack.py status

Examples:
    # Start ONLY specific roles (skip everything else, preserve what's running)
    ./orchestrator_stack.py start --only worker_vision vision_escalation

    # Start HOT models only
    ./orchestrator_stack.py start --hot-only

    # Start with warm architect
    ./orchestrator_stack.py start --include-warm architect_general

    # Dev mode (single 0.5B model)
    ./orchestrator_stack.py start --dev

    # Reload orchestrator API after code changes
    ./orchestrator_stack.py reload orchestrator

    # Check status
    ./orchestrator_stack.py status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, MutableMapping

_ORIGINAL_SUBPROCESS_POPEN = subprocess.Popen

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Re-exec under the project venv so heavy deps (yaml, faiss, numpy, ...)
# resolve regardless of the caller's PATH or active interpreter. System
# Python in a fresh devcontainer typically lacks PyYAML — this guard
# means `python3 scripts/server/orchestrator_stack.py ...` always works.
#
# The re-exec MUST stay confined to the __main__ entry path.  os.execv replaces
# the running process with `orchestrator_stack.py <caller's argv[1:]>`, so doing
# it at import time hijacks every *importer*: the importing tool vanishes and the
# replacement parses the importer's flags against this module's subparsers.  That
# is what broke `autokernel_enrollment.py --master-registry ...`, which imports
# this module only to pin `__file__` and STACK_PRIORS_PATH.
_PROJECT_VENV_PY = Path(__file__).resolve().parents[2] / ".venv/bin/python"


def _reexec_under_project_venv() -> None:
    """Re-exec this SCRIPT under the project venv.  Entry-point use only."""
    if (
        _PROJECT_VENV_PY.exists()
        and Path(sys.executable).resolve() != _PROJECT_VENV_PY.resolve()
        and os.environ.get("ORCHESTRATOR_STACK_REEXEC") != "1"
    ):
        os.environ["ORCHESTRATOR_STACK_REEXEC"] = "1"
        os.execv(str(_PROJECT_VENV_PY), [str(_PROJECT_VENV_PY), __file__, *sys.argv[1:]])


if __name__ == "__main__":
    # Runs BEFORE the heavy third-party imports below, exactly as the old
    # module-level guard did — system Python may lack PyYAML.
    _reexec_under_project_venv()

from scripts.server import stack_processes as _stack_processes
from scripts.server.stack_env import (
    _role_env_overrides,
    build_launch_env,
    build_service_env,
    compose_ld_library_path,
    strip_ambient_ggml,
)
from scripts.server.stack_health import wait_for_health as _wait_for_health
from scripts.server.fleet_markers import (
    LAUNCH_SOURCE_STACK_COMMANDS as _FLEET_SRC_STACK,
    write_llama_marker as _write_llama_marker,
    write_orchestrator_marker as _write_orchestrator_marker,
)
from scripts.server.stack_runtime import (
    runtime_requirements_for_role as _runtime_requirements_for_role_impl,
)
from scripts.server.stack_manifest import (
    AUX_SERVICES,
    DEV_MODEL,
    DEV_MODEL_PATH,
    DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
    DEFAULT_UBATCH_TOKENS,
    EMBEDDER_PORTS,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_PLACEMENT,
    EMBEDDING_SERVER_RECIPES,
    EXPLORE_DRAFT_MODEL,
    GPU_SHADOW_LANE_DEVICE,
    GPU_SHADOW_LANE_FALLBACK_CONTEXT_TOKENS,
    GPU_SHADOW_LANE_FALLBACK_SLOTS,
    GPU_SHADOW_LANE_REASONING,
    GPU_SHADOW_LANE_TENANT_ROLE,
    LAUNCH_CONTEXT_TOKENS,
    LAUNCH_KV_QUANT_CONFIGS,
    NO_SPEC_DECODE_ROLES,
    ORCHESTRATOR_PROFILES,
    VISION_ESCALATION_MMPROJ,
    VISION_ESCALATION_MODEL,
    VISION_ESCALATION_DEVICE,
    VISION_ESCALATION_REASONING,
    VISION_WORKER_MMPROJ,
    VISION_WORKER_MODEL,
    WORKER_POOL_MODELS,
    resolve_slots,
)
from scripts.server.stack_paths import (
    _HEALTH_SERVER_STARTUP,
    _HEALTH_VISION_SERVER,
    _HEALTH_WORKER_SERVER,
    _PATHS,
    _V2_ROLES,
    LLAMA_SERVER,
    LLAMA_SERVER_V2,
    LOG_DIR,
    SLOT_SAVE_DIR,
    STATE_FILE,
)
from scripts.server.stack_numa_evict import (
    placement_summary,
    pre_evict_gib_for_role,
    pre_evict_nodes,
)
from scripts.server.stack_numa import (
    MLOCK_ROLES,
    NUMA_CONFIG,
    _numa_prefix,
)
from scripts.server.bench_core_claim import (
    BenchPlacementRefusal,
    detect_running_cpu_bench,
    enforce_placement,
)
from scripts.server.runtime_facts_manifest import read_runtime_stack_numa_mode
from scripts.server.stack_numa_mode import normalize_stack_numa_mode
from scripts.server.stack_state import (
    ProcessInfo,
    load_state_file as _load_state_file,
    save_state_file as _save_state_file,
)
from src.config import _registry_timeout
from src.registry_loader import RegistryLoader

# =============================================================================

STACK_PRIORS_PATH = _PATHS["project_root"] / "orchestration/derived/stack_priors.yaml"
_WORKER_GENERAL_DEGRADED_FALLBACK = {
    "spec_type": "draft-mtp",
    "draft_max": 2,
    "draft_p_min": 0.0,
    "threads_draft": 16,
    "ubatch": 512,
    "kv_type_k": "q8_0",
    "kv_type_v": "q8_0",
    "context_tokens": 16384,
}

def _kernel_server_binary(backend: str) -> Path:
    """Resolve a production server binary by BACKEND, or RAISE.

    Fails closed on purpose. This used to catch every exception and substitute
    ``/mnt/raid0/llm/llama.cpp/build{-hip}/bin/llama-server``, which turned the one
    condition the kernel store exists to detect — a dangling or mis-pointed
    ``production/<backend>`` symlink after a promotion — into a silent launch of
    whatever that literal happened to contain. A CPU build serving a lane that
    declared the GPU does not crash and does not look unhealthy; it just answers
    slowly and wrongly. `kernel_paths` raises `KernelPathError` naming the backend
    and the path it tried, and that message must reach the operator unmodified.

    Callers must resolve LAZILY (only when no explicit binary_path is declared):
    an unusable store must not fail a launch that never consults the store.
    """
    try:
        from src.registry.kernel_paths import server_binary
    except ImportError as exc:  # in-repo module; absence is a packaging defect
        raise RuntimeError(
            f"cannot resolve the production kernel binary for backend {backend!r}: "
            f"src.registry.kernel_paths is unavailable ({exc}). Refusing to substitute "
            "a hardcoded build path — that is how a GPU lane silently launches a CPU build."
        ) from exc

    return server_binary(backend)


_CPU_ONLY_DEVICE_FLAGS = ("--device", "-dev")
_CPU_ONLY_DRAFT_DEVICE_FLAGS = ("--device-draft", "-devd")


def _has_any_flag(cmd: list[str], flags: tuple[str, ...]) -> bool:
    return any(flag in cmd for flag in flags)


def _append_device_args(cmd: list[str], flags: dict[str, Any] | None = None) -> None:
    """Pin a role to its DECLARED device, defaulting to CPU when nothing is declared.

    Was ``_append_cpu_only_device_args``, whose premise — "the production stack's
    text roles are CPU roles" — was true when it landed and stopped being true at
    the 2026-07-31 W1 cutover. It ASSUMED CPU rather than reading the declaration,
    and because it runs last and unconditionally over every builder's argv, it
    overwrote the answer for any builder that had not emitted a device itself.
    ``_build_role_command`` was such a builder, so architect_general — declared
    ``device: ROCm0``, ``ngl: all``, 36.7 GiB VRAM, and correctly resolved to the
    HIP binary — launched as `build-hip/bin/llama-server ... --device none`: the
    right binary with every device switched off, serving a 27B on 24 CPU threads
    while reporting healthy (PID 1935263, rocm-smi VRAM 0.02%, observed
    2026-08-01).

    The CPU default is KEPT, not deleted: a role that declares no device is a CPU
    role and pinning it to ``none`` is deliberate — a HIP-capable binary otherwise
    auto-selects ROCm0 for host op offload / draft sampling and regresses
    worker_general's ngram+MTP throughput. What changes is that "no device" now
    means "none was declared" instead of "none was emitted".

    The draft device follows the target's device unless separately declared.
    Forcing ``--device-draft none`` under a GPU-resident target would strand a
    NEXTN self-draft (whose draft head lives in the target's own GGUF) on the CPU.
    """
    declared = flags.get("device") if isinstance(flags, dict) else None
    declared = declared.strip() if isinstance(declared, str) and declared.strip() else None
    declared_draft = flags.get("device_draft") if isinstance(flags, dict) else None
    declared_draft = (
        declared_draft.strip()
        if isinstance(declared_draft, str) and declared_draft.strip()
        else None
    )

    if not _has_any_flag(cmd, _CPU_ONLY_DEVICE_FLAGS):
        cmd.extend(["--device", declared or "none"])
    if (
        ("--spec-type" in cmd or "-md" in cmd)
        and not _has_any_flag(cmd, _CPU_ONLY_DRAFT_DEVICE_FLAGS)
    ):
        cmd.extend(["--device-draft", declared_draft or declared or "none"])


# Pre-2026-08-01 name. Kept as an alias because the old name asserts a CPU-only
# premise that is no longer true; new call sites should use _append_device_args.
_append_cpu_only_device_args = _append_device_args


def _repo_short_sha(path: Path | None = None) -> str | None:
    repo = path or _PATHS["project_root"]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _stack_prior_launch(role_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return generated launch requirements/runtime for a live role, if usable.

    gpu-serving-tie-in P2-6 (P0-1b): falls back to LAUNCHER-TENANT records
    (``deployment_status: launcher_tenant`` — registry roles named by a
    launcher-only entry's ``tenant_role`` meta key) so an explicitly-requested
    launcher-only start (e.g. ``start --only gpu_shadow_lane``) can resolve its
    launch record WITHOUT the role being classified live. Live records always
    win on a name collision; routing consumers keep using
    ``live_stack_role_records`` and never see tenant records.
    """
    from src.registry.stack_priors import (
        launcher_tenant_role_records,
        live_stack_role_records,
        stack_prior_serving,
    )

    record = live_stack_role_records(STACK_PRIORS_PATH).get(role_name)
    if not isinstance(record, dict):
        record = launcher_tenant_role_records(STACK_PRIORS_PATH).get(role_name)
    if not isinstance(record, dict):
        return {}, {}
    serving = stack_prior_serving(record)
    launch = serving.get("launch")
    if not isinstance(launch, dict):
        return {}, {}
    requirements = launch.get("requirements")
    runtime = launch.get("runtime")
    return (
        requirements if isinstance(requirements, dict) else {},
        runtime if isinstance(runtime, dict) else {},
    )


def _runtime_cache(runtime: dict[str, Any]) -> dict[str, Any]:
    cache = runtime.get("cache")
    return cache if isinstance(cache, dict) else {}


def _runtime_flags(runtime: dict[str, Any]) -> dict[str, Any]:
    flags = runtime.get("flags")
    return flags if isinstance(flags, dict) else {}


def _resolve_parallel_slots(
    cache: dict[str, Any],
    role_name: str,
    port: int,
    numa_instance: int,
    fallback_mode: str = "default",
    *,
    worker_type: str | None = None,
    vision_type: str | None = None,
) -> str:
    """`-np` for the instance being launched, not for the role.

    2026-08-02. `-np` used to be a per-ROLE number: `runtime.cache.slots`, one
    value for every instance a role runs. That cannot express the ratified shape
    — frontdoor's 96-core full takes 16 slots while each of its 48-core halves
    takes 4 — so the compiled priors now carry `runtime.cache.slots_by_port`, one
    entry per launch instance, joined by the compiler from the model's
    `slots_by_shape` and the instance's declared `cpu_shape`.

    Resolution order, most specific first:
      1. the compiled per-PORT value (what production uses),
      2. the compiled role-level `slots` (a record compiled before this field),
      3. `stack_manifest.resolve_slots(role, numa_instance=...)`, which re-derives
         the same join from the ambient registry when no prior is available.
    Port, not numa_instance, is the key: it is the one identifier the launcher is
    certain of at this point, and it is what admission keys on, so a mismatch
    between `-np` and the admission limit is not expressible.
    """
    by_port = cache.get("slots_by_port")
    if isinstance(by_port, dict):
        value = by_port.get(port)
        if value is None:
            value = by_port.get(str(port))
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return str(value)
    return _runtime_positive_int(
        cache,
        "slots",
        resolve_slots(
            role_name,
            fallback_mode,
            worker_type=worker_type,
            vision_type=vision_type,
            numa_instance=numa_instance,
        ).slots,
    )


def _runtime_positive_int(
    container: dict[str, Any],
    key: str,
    fallback: int | str,
) -> str:
    value = container.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return value
    return str(fallback)


def _warn_if_ubatch_exceeds_batch(
    role_name: str, ubatch: int | str, batch: int | str | None = None
) -> None:
    """K4 guard: llama.cpp clamps cparams.n_ubatch = min(n_batch, n_ubatch).

    Every command builder here emits ``-ub`` without an accompanying ``-b``, so
    the effective micro-batch is whatever ``-b`` would default to (2048) unless
    ``batch`` names an explicit value this builder actually emits. A declared
    ubatch above that ceiling is silently inert — warn at build time so the
    next case of this (handoffs/active/dynamic-stack-concurrency.md K4) is
    visible instead of requiring a fresh audit to rediscover.
    """
    try:
        ubatch_int = int(ubatch)
    except (TypeError, ValueError):
        return
    try:
        batch_int = int(batch) if batch is not None else 2048
    except (TypeError, ValueError):
        batch_int = 2048
    if ubatch_int > batch_int:
        print(
            f"    [WARN] {role_name}: -ub {ubatch_int} exceeds the effective "
            f"-b {batch_int} (llama.cpp clamps n_ubatch to n_batch; no -b is "
            "emitted here) -- the declared ubatch is INERT. K4."
        )


def _runtime_nonnegative_int(
    container: dict[str, Any],
    key: str,
    fallback: int | str,
) -> str:
    value = container.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    if isinstance(value, str) and value.isdigit() and int(value) >= 0:
        return value
    return str(fallback)


def _runtime_number_string(
    container: dict[str, Any],
    key: str,
    fallback: int | float | str,
) -> str:
    value = container.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and value:
        try:
            float(value)
        except ValueError:
            return str(fallback)
        return value
    return str(fallback)


def _runtime_string(container: dict[str, Any], key: str, fallback: str) -> str:
    value = container.get(key)
    return value if isinstance(value, str) and value else fallback


def _same_real_model_path(left: str, right: str) -> bool:
    return os.path.realpath(left) == os.path.realpath(right)


def _append_spec_decode_args(
    cmd: list[str],
    *,
    model_path: str,
    draft_model_path: str | None,
    spec_type: str | None,
    draft_max: str | None,
    draft_min: str | None = None,
    draft_p_min: str | None = None,
    draft_p_split: str | None = None,
    threads_draft: str | None = None,
    ngram_mod_n_min: str | None = None,
    ngram_mod_n_max: str | None = None,
    ngram_mod_n_match: str | None = None,
) -> None:
    if draft_model_path and not _same_real_model_path(model_path, draft_model_path):
        cmd.extend(["-md", draft_model_path])
    if spec_type:
        cmd.extend(["--spec-type", spec_type])
    if draft_max:
        cmd.extend(["--spec-draft-n-max", draft_max])
    if draft_min:
        cmd.extend(["--spec-draft-n-min", draft_min])
    if draft_p_min not in (None, ""):
        cmd.extend(["--draft-p-min", draft_p_min])
    if draft_p_split not in (None, ""):
        cmd.extend(["--draft-p-split", draft_p_split])
    if threads_draft:
        cmd.extend(["--threads-draft", threads_draft])
    if ngram_mod_n_min:
        cmd.extend(["--spec-ngram-mod-n-min", ngram_mod_n_min])
    if ngram_mod_n_max:
        cmd.extend(["--spec-ngram-mod-n-max", ngram_mod_n_max])
    if ngram_mod_n_match:
        cmd.extend(["--spec-ngram-mod-n-match", ngram_mod_n_match])


# Compiled flag name -> (llama-server argument, allow a non-numeric token,
# token must exist as a filesystem path).
# Declared in the registry's serving block, compiled into runtime.flags by
# src/registry/stack_priors.py, emitted here. Kept as DATA so adding a declared
# serving knob is a registry + compiler change, not a new branch in every builder.
#
# n_gpu_layers allows a string because the registry declares it both ways:
# worker_vision as `n_gpu_layers: 999`, the 27B (architect_critic since the
# 2026-09-27 ARCHITECT SWAP; architect_general before) as `ngl: all`. Both are
# valid llama-server values on this kernel.
#
# chat_template_file is a filesystem PATH, so it is emitted only when the file
# exists: a missing template must refuse at launch-build time (same posture as
# validate_model_paths() for a missing model — fail fast, never launch a server
# that would fall back to the GGUF-embedded template while the registry claims
# otherwise).
_RUNTIME_SERVING_FLAG_ARGS: tuple[tuple[str, str, bool, bool], ...] = (
    ("n_gpu_layers", "-ngl", True, False),
    ("image_min_tokens", "--image-min-tokens", False, False),
    ("cache_ram", "--cache-ram", False, False),
    ("chat_template_file", "--chat-template-file", True, True),
)


def _append_runtime_serving_flags(cmd: list[str], flags: dict[str, Any]) -> None:
    """Emit the declared serving flags that the compiled priors carry.

    Emitted ONLY when the compiled record declares them. Nothing here has a
    fallback: these came from ``roles.<role>.serving`` / ``server_mode.<role>``, and
    a launcher-side default would put a GPU-shaped flag on every CPU role — which is
    precisely how a "sensible default" becomes an undeclared production setting.

    ``0`` is a DECLARED value (``cache_ram: 0`` disables the prompt cache), so the
    guard is ``is not None`` / ``>= 0``, never truthiness.
    """
    for key, arg, allow_str, must_exist in _RUNTIME_SERVING_FLAG_ARGS:
        value = flags.get(key)
        if isinstance(value, bool):
            continue
        token: str | None = None
        if isinstance(value, int) and value >= 0:
            token = str(value)
        elif allow_str and isinstance(value, str) and value.strip():
            token = value.strip()
        if token is None:
            continue
        if must_exist and not Path(token).exists():
            raise ValueError(
                f"declared {key} path does not exist: {token} "
                f"(refusing to emit {arg}; fix the registry declaration or "
                f"restore the file)"
            )
        cmd.extend([arg, token])


def _append_runtime_kv_args(cmd: list[str], cache: dict[str, Any]) -> None:
    kv_type_k = cache.get("kv_type_k")
    kv_type_v = cache.get("kv_type_v")
    if isinstance(kv_type_k, str) and isinstance(kv_type_v, str):
        cmd.extend(["-ctk", kv_type_k, "-ctv", kv_type_v])
    # Unified KV pool. Emitted in BOTH directions when declared, never when not:
    # llama-server's own default is "unified only if -np is auto"
    # (tools/server/server.cpp:145-150), and this launcher always passes -np, so an
    # undeclared role gets split KV -- which is what :8083 silently ran while the
    # registry asserted the opposite (stack-change-kvu, 2026-09-24).
    draft_k = cache.get("draft_kv_type_k")
    draft_v = cache.get("draft_kv_type_v")
    if isinstance(draft_k, str) and draft_k and isinstance(draft_v, str) and draft_v:
        cmd.extend(["-ctkd", draft_k, "-ctvd", draft_v])
    kv_unified = cache.get("kv_unified")
    if kv_unified is True:
        cmd.append("--kv-unified")
    elif kv_unified is False:
        cmd.append("--no-kv-unified")
    if cache.get("kv_hadamard") is True:
        # 2026-06-26 v6 cutover: --kv-hadamard removed in v6 (no role sets kv_hadamard true today)
        # cmd.append("--kv-hadamard")
        pass


def _append_runtime_spec_args(cmd: list[str], runtime: dict[str, Any], model_path: str) -> None:
    spec = _runtime_flags(runtime).get("spec")
    if not isinstance(spec, dict) or spec.get("enabled") is not True:
        return
    draft_model_path = spec.get("draft_model_path")
    if not isinstance(draft_model_path, str) or not draft_model_path:
        return
    spec_type = spec.get("type")
    draft_max = spec.get("draft_max")
    draft_min = spec.get("draft_min")
    draft_p_min = spec.get("draft_p_min")
    draft_p_split = spec.get("draft_p_split")
    threads_draft = spec.get("threads_draft")
    _append_spec_decode_args(
        cmd,
        model_path=model_path,
        draft_model_path=draft_model_path,
        spec_type=spec_type if isinstance(spec_type, str) and spec_type else None,
        # 2026-06-26 v6 cutover: --draft-max renamed to --spec-draft-n-max (same value)
        draft_max=(
            str(draft_max)
            if isinstance(draft_max, int) and not isinstance(draft_max, bool) and draft_max > 0
            else None
        ),
        draft_min=(
            str(draft_min)
            if isinstance(draft_min, int) and not isinstance(draft_min, bool) and draft_min >= 0
            else None
        ),
        draft_p_min=(
            str(float(draft_p_min))
            if isinstance(draft_p_min, (int, float)) and not isinstance(draft_p_min, bool)
            else None
        ),
        draft_p_split=(
            str(float(draft_p_split))
            if isinstance(draft_p_split, (int, float)) and not isinstance(draft_p_split, bool)
            else None
        ),
        threads_draft=(
            str(threads_draft)
            if isinstance(threads_draft, int)
            and not isinstance(threads_draft, bool)
            and threads_draft > 0
            else None
        ),
        ngram_mod_n_min=_runtime_positive_int(spec, "ngram_mod_n_min", "")
        if "ngram_mod_n_min" in spec
        else None,
        ngram_mod_n_max=_runtime_positive_int(spec, "ngram_mod_n_max", "")
        if "ngram_mod_n_max" in spec
        else None,
        ngram_mod_n_match=_runtime_positive_int(spec, "ngram_mod_n_match", "")
        if "ngram_mod_n_match" in spec
        else None,
    )


# Path/binary constants moved to scripts/server/stack_paths.py (2026-05-22).
# Manifest (PORT_MAP, ROLE_LAUNCH_META, model paths, classification helpers,
# validate_model_paths, validate_against_registry, etc.) moved to
# scripts/server/stack_manifest.py. All names are re-exported via the imports
# at the top of this file so `from orchestrator_stack import X` works for the
# registry compiler fallback path (src/registry/registry_compiler.py:266).
#
# Port topology note: this launcher intentionally no longer owns the static
# port tables. Full/primary role ports live in stack_manifest.PORT_MAP; NUMA
# quarter/replica ports and their CPU pinning live in stack_numa.NUMA_CONFIG.
# Avoid documenting fixed 808x/818x ranges here because the current topology is
# role-specific (for example 8070 full + 8080/8180/8280/8380 quarters for
# frontdoor, 8072 full + 8082/8182/8282/8382 quarters for worker_general, and
# no 8084 architect_coding server).
# =============================================================================


# =============================================================================
# State Management
# =============================================================================


def load_state() -> dict[str, ProcessInfo]:
    """Load state from file."""
    return _load_state_file(STATE_FILE)


def save_state(state: dict[str, ProcessInfo]) -> None:
    """Save state to file."""
    _save_state_file(STATE_FILE, state)


# =============================================================================
# Process Management
# =============================================================================


def check_free_memory() -> int:
    """Return free memory in GB (delegates to stack_processes.free_memory_gb)."""
    return _stack_processes.free_memory_gb()


# =============================================================================
# Host prerequisites moved to scripts/server/stack_host.py (2026-05-21 refactor).
# Per-role env blocks moved to scripts/server/stack_env.py (same session).
# Re-exported above via the module-level imports so existing call sites
# (start_server, cmd_start etc.) keep resolving these unqualified names.
# =============================================================================


def _runtime_requirements_for_role(
    registry: "RegistryLoader", role_name: str
) -> tuple[str | None, list[str] | None]:
    """Return (binary_dir, ld_library_paths) for a role (delegates to stack_runtime)."""
    return _runtime_requirements_for_role_impl(registry, role_name)


def _stack_prior_runtime_overrides(role_name: str) -> tuple[str | None, list[str] | None]:
    """Return runtime binary override + LD paths from the compiled stack prior."""
    _requirements, runtime = _stack_prior_launch(role_name)
    binary_dir = runtime.get("binary_dir") if isinstance(runtime.get("binary_dir"), str) else None
    raw_ld_paths = runtime.get("ld_library_path")
    ld_paths = (
        [str(path) for path in raw_ld_paths if isinstance(path, str)]
        if isinstance(raw_ld_paths, list)
        else None
    )
    binary_override = str(Path(binary_dir) / "llama-server") if binary_dir else None
    return binary_override, ld_paths


def _strip_ambient_ggml(
    env: MutableMapping[str, str], preserve: Iterable[str] = (), *, label: str
) -> list[str]:
    """stack_env.strip_ambient_ggml plus a launch-log line. Used by every llama-server branch."""
    stripped = strip_ambient_ggml(env, preserve)
    if stripped:
        print(f"    [{label}] stripped ambient GGML_* env: {stripped}")
    return stripped


def _apply_runtime_requirements_env(
    env: MutableMapping[str, str],
    *,
    binary_override: str | None,
    ld_paths: list[str] | None,
    ld_path_mode: str = "prepend",
    preserve: Iterable[str] = (),
) -> None:
    """Apply role runtime overrides to a llama-server launch environment.

    `preserve` names GGML_* keys the ROLE itself declares (stack_env._ROLE_ENV_BLOCKS).
    The binary-override strip exists to keep AMBIENT GGML_* knobs out of a launch whose
    binary may not honour them. Until 2026-09-26 it also stripped the role's own declared
    block. Since b060dd56 (2026-07-31) every role's compiled prior carries a binary_dir, so
    the strip fired on every launch and dropped every declared GGML knob. The only one it
    ever dropped was architect_critic's dead GGML_NUMA_REPACK_INTERLEAVE (live environ
    readback, 2026-09-26, DAR-LAT-3h). The declared values come from build_launch_env,
    which has already overwritten any ambient value of the same key, so preserving them
    still keeps ambient values out.

    `ld_path_mode` defaults to "prepend", which routes through
    `compose_ld_library_path`'s pure-concatenation branch and is therefore
    byte-identical to the pre-2026-08-02 inline expression for every role. A role
    whose runtime requirements declare `ld_library_path_mode: replace` gets the
    ambient path dropped — available to llama-server roles for the same reason aux
    services need it (an experimental tree with its own ggml generation), but no
    role declares it today, so no live role's env changes.
    """
    # The strip no longer keys on `binary_override`. That was the misread signal: the
    # compiler marks a backend-derived binary_dir `env_policy: canonical` precisely so it
    # does NOT change env policy (stack_priors.py, "A backend-derived default must not, or
    # every role silently changes env policy"), but this function read binary_dir presence
    # and stripped anyway. Ambient GGML_* is now dropped on every launch and declared keys
    # are always kept, so there is no policy left for the flag to select.
    _strip_ambient_ggml(env, preserve, label="launch")
    if binary_override:
        env["KMP_BLOCKTIME"] = "10"
    if ld_paths:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = compose_ld_library_path(ld_paths, existing, ld_path_mode)
        if ld_path_mode == "replace":
            print(f"    LD_LIBRARY_PATH := {ld_paths}  (ambient dropped)")
        else:
            print(f"    LD_LIBRARY_PATH += {ld_paths}")


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    return _stack_processes.is_port_in_use(port)


def _pids_on_port(port: int) -> list[int]:
    """Best-effort discovery of LISTEN pids on a TCP port.

    NB: WITHOUT the `-sTCP:LISTEN` filter, `lsof -i :PORT` also returns the
    PIDs of any client process with an ESTABLISHED connection to that port —
    e.g. a Firefox tab open to localhost:8000/dashboard, or autopilot mid-
    HTTP request. Killing those clients was the destructive bug observed
    2026-05-21 where `reload orchestrator` would kill Firefox + autopilot.
    The filter ensures we only target actual listeners.
    """
    return _stack_processes.pids_on_port(port)


def _pid_alive(pid: int) -> bool:
    """Return True when a pid currently exists."""
    return _stack_processes.pid_alive(pid)


def _child_pids(pid: int) -> list[int]:
    """Return direct child pids for a process."""
    return _stack_processes.child_pids(pid)


def _collect_descendants(root_pid: int) -> list[int]:
    """Collect all descendants of root_pid (breadth-first)."""
    return _stack_processes.collect_descendants(root_pid)


def _renice_all_threads(pid: int, nice: int) -> None:
    """Renice every thread of `pid` to `nice`."""
    _stack_processes.renice_all_threads(pid, nice)


def _set_oom_protection(pids: list[int], adj: int = -1000) -> int:
    """Protect control-plane pids from earlyoom via oom_score_adj=-1000.

    See stack_processes.set_oom_score_adj — best-effort (`sudo -n`); the durable
    replacement for the manual one-shot `choom` that did not survive an API restart.
    """
    return _stack_processes.set_oom_score_adj(pids, adj)


def kill_process(pid: int, timeout: int = 5) -> bool:
    """Kill a process tree gracefully, then forcefully."""
    return _stack_processes.kill_process_tree(pid, timeout=timeout)


# =============================================================================
# Docker container management moved to scripts/server/stack_docker.py
# (2026-05-21 refactor). Re-exported via the module-level import above so
# existing call sites (cmd_start, cmd_stop, cmd_reload, cmd_status) keep
# resolving these names unqualified.
# =============================================================================


def wait_for_health(
    port: int, timeout: int = _HEALTH_SERVER_STARTUP, path: str = "/health"
) -> bool:
    """Wait for server health endpoint (delegates to stack_health.wait_for_health).

    Wrapper preserves the registry-driven _HEALTH_SERVER_STARTUP default for
    callers that don't pass timeout explicitly. (Currently every production
    caller does pass one — see the 14 call sites in this file — but the
    default is part of the public API.)
    """
    return _wait_for_health(port, timeout, path)


# =============================================================================
# Server Launching
# =============================================================================


# -----------------------------------------------------------------------------
# Mode-specific command builders (called by build_server_command dispatcher).
# Each returns a fully formed llama-server argv list for one launch shape.
# Kept private (_build_*) because the public API is build_server_command.
# -----------------------------------------------------------------------------


def _build_vision_command(port: int, vision_type: str | None, numa_instance: int = 0) -> list[str]:
    """VL launch: production worker or escalation multimodal server.

    Thread count comes from NUMA_CONFIG per (role, numa_instance) — added
    2026-05-24 along with the per-instance fix for `_build_role_command`, so
    that the newly-quartered vision roles get the correct -t per instance.
    """
    if vision_type == "escalation":
        role_name = "vision_escalation"
        requirements, runtime = _stack_prior_launch(role_name)
        cache = _runtime_cache(runtime)
        flags = _runtime_flags(runtime)
        thread_count = _resolve_thread_count(role_name, numa_instance)
        cmd = [
            _runtime_string(runtime, "binary_path", str(LLAMA_SERVER)),
            "-m",
            _runtime_string(requirements, "model_path", VISION_ESCALATION_MODEL),
            "--mmproj",
            _runtime_string(requirements, "mmproj_path", VISION_ESCALATION_MMPROJ),
        ]
        for override in flags.get("override_kv") or []:
            if isinstance(override, str) and override:
                cmd.extend(["--override-kv", override])
        cmd.extend(
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-np",
                _resolve_parallel_slots(
                    cache, role_name, port, numa_instance, "vision",
                    vision_type="escalation",
                ),
                "-c",
                _runtime_positive_int(cache, "context_tokens", LAUNCH_CONTEXT_TOKENS[role_name]),
                "-t",
                thread_count,
            ]
        )
        if flags.get("flash_attn", True) is True:
            cmd.extend(["--flash-attn", "on"])
        # KV quant, emitted exactly as the general builder emits it (compiled
        # runtime first, derived table as the no-prior fallback). This branch had
        # no KV emit at all, which was inert only while the VL roles declared no
        # kv_quant. The 2026-08-02 master change gave worker_vision (and its
        # vision_escalation alias) q8_0/q8_0 on measured evidence, so from that
        # point a silent f16 launch contradicted the declaration.
        if runtime:
            _append_runtime_kv_args(cmd, cache)
        else:
            _append_kv_quant_args(cmd, role_name)
        reasoning = flags.get("reasoning") or VISION_ESCALATION_REASONING
        if isinstance(reasoning, str) and reasoning:
            cmd.extend(["--reasoning", reasoning])
        device = flags.get("device") or VISION_ESCALATION_DEVICE
        if isinstance(device, str) and device:
            cmd.extend(["--device", device])
        _append_runtime_serving_flags(cmd, flags)
        if cache.get("no_mmap", False) is True:
            cmd.append("--no-mmap")
        return cmd
    # Qwen3-VL-30B-A3B-Instruct Q4_K_M on MI210 (2026-07-31 cutover from Qwen2.5-VL-7B).
    role_name = "worker_vision"
    requirements, runtime = _stack_prior_launch(role_name)
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    thread_count = _resolve_thread_count(role_name, numa_instance)
    cmd = [
        _runtime_string(runtime, "binary_path", str(LLAMA_SERVER)),
        "-m",
        _runtime_string(requirements, "model_path", VISION_WORKER_MODEL),
        "--mmproj",
        _runtime_string(requirements, "mmproj_path", VISION_WORKER_MMPROJ),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        # This literal was `2`. `server_mode.worker_vision.slots` declares 1, and
        # this branch fires for any non-escalation VL server — so the declaration
        # was shadowed and the VL role launched with twice the slots it declared.
        # The compiled-priors side of that defect was fixed 2026-08-01; this is
        # the launcher side of the same literal.
        "-np",
        _resolve_parallel_slots(
            cache, role_name, port, numa_instance, "vision", vision_type="worker"
        ),
        "-c",
        _runtime_positive_int(cache, "context_tokens", LAUNCH_CONTEXT_TOKENS[role_name]),
        "-t",
        thread_count,
    ]
    if flags.get("flash_attn", True) is True:
        cmd.extend(["--flash-attn", "on"])
    # KV quant — see the escalation branch above. `server_mode.worker_vision
    # .serving_shape.kv_quant` became q8_0/q8_0 on 2026-08-02 (MMMU-250 paired
    # A/B, non-inferior at a pre-registered 3 pp margin); without this the
    # launcher started the :8086 VL process on f16 while the compiled prior,
    # the derived KV table and the VRAM budget all said q8_0.
    if runtime:
        _append_runtime_kv_args(cmd, cache)
    else:
        _append_kv_quant_args(cmd, role_name)
    reasoning = flags.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        cmd.extend(["--reasoning", reasoning])
    device = flags.get("device")
    if isinstance(device, str) and device:
        cmd.extend(["--device", device])
    # -ngl / --image-min-tokens / --cache-ram: declared in the registry's serving
    # block, compiled into runtime.flags, emitted here. Without -ngl the server takes
    # `--device ROCm0` and then offloads nothing, which is a GPU launch in name only.
    _append_runtime_serving_flags(cmd, flags)
    if cache.get("no_mmap", False) is True:
        cmd.append("--no-mmap")
    return cmd


def _build_embedding_command(port: int) -> list[str]:
    """Embedding server command for production BGE ports or warm eval candidates."""
    recipe = EMBEDDING_SERVER_RECIPES.get(port)
    if recipe is None:
        recipe = {
            "model_path": EMBEDDING_MODEL_PATH,
            "context_tokens": 512,
            "threads": 4,
            "slots": 4,
            "pooling": "cls",
            "flash_attn": True,
        }

    cmd = [
        str(LLAMA_SERVER),
        "-m",
        str(recipe["model_path"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        str(recipe["slots"]),
        "-c",
        str(recipe["context_tokens"]),
        "-t",
        str(recipe["threads"]),
        "--embeddings",
        "--pooling",
        str(recipe["pooling"]),
    ]
    if recipe.get("flash_attn"):
        cmd.extend(["--flash-attn", "on"])
    # UFH-12 Phase 0: private weights, so the placement's membind is real
    # (a membind over shared-mmap pages is accepted and does nothing).
    if recipe.get("no_mmap"):
        cmd.append("--no-mmap")
    return cmd


def _build_worker_fast_command(port: int, model_path: str) -> list[str]:
    """Fast worker: 1.5B model, 4 slots for parallel burst capacity."""
    return [
        str(LLAMA_SERVER),
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        # Master has no server_mode row for worker_fast, so this is the manifest's
        # declared worker_pool_fast fallback (4) rather than a literal here.
        str(resolve_slots("worker_fast", "worker_pool", worker_type="fast").slots),
        "-c",
        str(LAUNCH_CONTEXT_TOKENS["worker_fast"]),  # 4K per slot
        "-t",
        "16",  # 16 threads for small model
        "--flash-attn",
        "on",
    ]


def _build_worker_general_command(
    port: int, model_path: str, binary_override: str | None, numa_instance: int = 0
) -> list[str]:
    """Worker-general MTP path: gemma4-26B-A4B Q4_K_M via ik_llama.cpp PR #1744.

    Swapped 2026-05-08 from Qwen3-Coder-30B-A3B Q4_K_M. Tool_compliance 96% vs 78%
    prior, +36% tps. `binary_override` comes from `server_mode.worker.runtime_requirements.binary_dir`
    in the registry — required because gemma4 MTP needs ik_llama.cpp PR #1744 build,
    not the production llama.cpp build.
    """
    requirements, runtime = _stack_prior_launch("worker_general")
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    spec = flags.get("spec") if isinstance(flags.get("spec"), dict) else {}
    binary = (
        binary_override
        if binary_override
        else _runtime_string(runtime, "binary_path", str(LLAMA_SERVER))
    )
    model_path = _runtime_string(requirements, "model_path", model_path)
    draft_model_path = _runtime_string(
        spec,
        "draft_model_path",
        _runtime_string(requirements, "draft_model_path", EXPLORE_DRAFT_MODEL),
    )
    # 2026-05-24: now uses generic _resolve_thread_count(role, numa_instance).
    # Pre-2026-05-24 used a port-matching workaround here because the generic
    # _resolve_thread_count ignored numa_instance — that bug is now fixed at
    # the source, so the workaround is no longer needed.
    numa_thread_count = int(_resolve_thread_count("worker_general", numa_instance))
    cmd = [
        binary,
        "-m",
        model_path,
    ]
    _append_spec_decode_args(
        cmd,
        model_path=model_path,
        draft_model_path=draft_model_path,
        spec_type=_runtime_string(
            spec, "type", str(_WORKER_GENERAL_DEGRADED_FALLBACK["spec_type"])
        ),
        draft_max=_runtime_positive_int(
            spec,
            "draft_max",
            _WORKER_GENERAL_DEGRADED_FALLBACK["draft_max"],
        ),
        draft_min=_runtime_nonnegative_int(spec, "draft_min", "")
        if "draft_min" in spec
        else None,
        draft_p_min=_runtime_number_string(
            spec,
            "draft_p_min",
            _WORKER_GENERAL_DEGRADED_FALLBACK["draft_p_min"],
        ),
        draft_p_split=_runtime_number_string(spec, "draft_p_split", "")
        if "draft_p_split" in spec
        else None,
        threads_draft=_runtime_positive_int(
            spec,
            "threads_draft",
            _WORKER_GENERAL_DEGRADED_FALLBACK["threads_draft"],
        ),
        ngram_mod_n_min=_runtime_nonnegative_int(spec, "ngram_mod_n_min", "")
        if "ngram_mod_n_min" in spec
        else None,
        ngram_mod_n_max=_runtime_nonnegative_int(spec, "ngram_mod_n_max", "")
        if "ngram_mod_n_max" in spec
        else None,
        ngram_mod_n_match=_runtime_positive_int(spec, "ngram_mod_n_match", "")
        if "ngram_mod_n_match" in spec
        else None,
    )
    # K4 (handoffs/active/dynamic-stack-concurrency.md): no -b is emitted on
    # this path either, so a declared ubatch above the 2048 n_batch default
    # would be silently clamped exactly as on the default-mode builder.
    worker_general_ubatch = _runtime_positive_int(
        cache, "ubatch", _WORKER_GENERAL_DEGRADED_FALLBACK["ubatch"]
    )
    _warn_if_ubatch_exceeds_batch("worker_general", worker_general_ubatch)
    cmd.extend(
        [
            "-ub",
            worker_general_ubatch,
            *(
                ["--no-mmap"] if cache.get("no_mmap", True) is True else []
            ),  # canonical recipe: bulk-read on EPYC NUMA cold-cache decode
            "--reasoning",
            str(
                flags.get("reasoning") or "off"
            ),  # disable gemma4 thinking-channel (output otherwise lands in
            # reasoning_content not content; registry: gemma4_26b reasoning=off)
            *(
                ["--jinja"] if flags.get("jinja", True) is True else []
            ),  # gemma4 ships a custom chat template embedded in the gguf;
            # without --jinja, llama.cpp rejects /v1/chat/completions
            # with "this custom template is not supported"
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            # ⚠ HISTORICAL -np 1 RATIONALE, SUPERSEDED 2026-08-02 BUT KEPT: MTP was
            # reported to share state with the target across slots in a way the
            # ik_llama.cpp PR #1744 build asserted on with -np 2 ("tensor buffer not
            # set" at ggml-backend.cpp:236). That build is NOT what this role runs any
            # more — the fleet is on production-consolidated-v8 with native draft-mtp
            # — and the operator has ratified full 16 / half 4 for this role. If the
            # assertion reappears, THIS is the note that explains it, and the fix is
            # to lower `serving_shape.slots_by_shape` in the master registry, not to
            # reintroduce a literal here.
            "-np",
            _resolve_parallel_slots(
                cache, "worker_general", port, numa_instance,
                "worker_pool", worker_type="explore",
            ),
            "-c",
            _runtime_positive_int(
                cache, "context_tokens", _WORKER_GENERAL_DEGRADED_FALLBACK["context_tokens"]
            ),
            # Per-instance thread count (full=96, quarters=48). Pre-2026-05-08 was
            # hardcoded -t 24 (Qwen3-Coder tolerated it); gemma4 + MTP under
            # ik_llama.cpp PR #1744 must match the bench recipe to avoid the
            # "tensor buffer not set" MTP assertion.
            "-t",
            str(numa_thread_count),
            # KV cache q8_0/q8_0 — registry-declared and required for stable MTP buffer
            # allocation. f16 default left some MTP tensor buffers uninitialized.
            "-ctk",
            _runtime_string(
                cache, "kv_type_k", str(_WORKER_GENERAL_DEGRADED_FALLBACK["kv_type_k"])
            ),
            "-ctv",
            _runtime_string(
                cache, "kv_type_v", str(_WORKER_GENERAL_DEGRADED_FALLBACK["kv_type_v"])
            ),
            *(["--flash-attn", "on"] if flags.get("flash_attn", True) is True else []),
        ]
    )
    return cmd


def _build_worker_explore_command(
    port: int, model_path: str, binary_override: str | None, numa_instance: int = 0
) -> list[str]:
    """Compatibility wrapper for the retired worker_explore name."""
    return _build_worker_general_command(port, model_path, binary_override, numa_instance)


def _build_dev_command(port: int) -> list[str]:
    """Dev mode: single 0.5B Qwen2.5-Coder model for fast iteration."""
    return [
        str(LLAMA_SERVER),
        "-m",
        DEV_MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        "4",
        "-c",
        "4096",
        "-t",
        "16",
        "--flash-attn",
        "on",
    ]


def _build_eval_batch_frontdoor_command(port: int, numa_instance: int = 0) -> list[str]:
    """Warm eval-batch frontdoor lane derived from the certified fleet shape.

    This intentionally reuses the frontdoor model/runtime priors but overrides
    the serving shape to a single `-np 2` process on a dedicated high port. The
    role is launcher-only; normal routing still speaks in frontdoor aliases.
    """
    source_role = "frontdoor"
    requirements, runtime = _stack_prior_launch(source_role)
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    model_path = _runtime_string(requirements, "model_path", "")
    if not model_path:
        source_config = RegistryLoader().get_role(source_role)
        if source_config is not None:
            model_path = str(source_config.model.full_path)
    binary = _runtime_string(runtime, "binary_path", str(_resolve_binary_for_role(source_role)))
    # K4 (handoffs/active/dynamic-stack-concurrency.md): this lane never emits
    # -b either, so the same n_batch=2048 default clamp applies to its -ub as
    # to the main role builder's. The degraded-fallback literal below tracks
    # DEFAULT_UBATCH_TOKENS (now the clamp value) rather than a stale 8192.
    eval_batch_frontdoor_ubatch = _runtime_positive_int(
        cache, "ubatch", DEFAULT_UBATCH_TOKENS
    )
    _warn_if_ubatch_exceeds_batch("eval_batch_frontdoor", eval_batch_frontdoor_ubatch)
    cmd = [
        binary,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        # The lane's defining shape (P-BENCH-3/E2), declared as
        # launch_shape.fallback_slots.eval_batch_frontdoor rather than as a
        # literal. It deliberately does NOT read the source role's slot prior:
        # this lane exists to serve a separately certified batch shape without
        # mutating the interactive frontdoor process.
        "-np",
        str(resolve_slots("eval_batch_frontdoor", "eval_batch_frontdoor").slots),
        # 2026-08-02: reads the LANE'S declared context, not `cache` — i.e. not
        # frontdoor's. Same argument as the `-np` above it: this lane borrows the
        # source role's model and runtime priors, never its serving shape. While
        # it read `cache["context_tokens"]` it tracked frontdoor, so frontdoor's
        # 32768 -> 262144 move would have silently taken this lane with it.
        "-c",
        str(LAUNCH_CONTEXT_TOKENS["eval_batch_frontdoor"]),
        "-t",
        _resolve_thread_count("eval_batch_frontdoor", numa_instance),
        "-ub",
        eval_batch_frontdoor_ubatch,
        "-ctk",
        _runtime_string(cache, "kv_type_k", "q8_0"),
        "-ctv",
        _runtime_string(cache, "kv_type_v", "q8_0"),
        "--log-colors",
        "off",
    ]
    if flags.get("flash_attn", True) is True:
        cmd.extend(["--flash-attn", "on"])
    if flags.get("jinja", True) is True:
        cmd.append("--jinja")
    if cache.get("mlock", True) is True:
        cmd.append("--mlock")
    if cache.get("no_mmap", False) is True:
        cmd.append("--no-mmap")
    _append_runtime_spec_args(cmd, runtime, model_path)
    reasoning = flags.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        cmd.extend(["--reasoning", reasoning])
    return cmd


def _build_gpu_shadow_lane_command(port: int, numa_instance: int = 0) -> list[str]:
    """Role-agnostic GPU shadow lane (docs/gpu-shadow-lane.md; P2-6/P0-2).

    INERT today: only reachable via the ``gpu_shadow_lane`` launch mode, which
    no ROLE_LAUNCH_META entry carries until the registry proposal
    (docs/proposals/gpu-shadow-lane-registry-proposal.md) is applied.

    Tenant priors come from the registry role named by
    GPU_SHADOW_LANE_TENANT_ROLE (tenancy as data; resolved through the
    launcher-tenant stack-prior record). Serving shape (-np / -c) flows from
    orchestration/gpu_shadow_lane_np_ceiling.yaml's serving_shape block via the
    compiled priors cache — never from the CPU-mode launcher defaults. The
    fallback literals mirror that same phase2 shape and exist only for the
    degraded no-priors case; the preflight probe
    (scripts/server/gpu_shadow_lane_preflight.py) verifies -np/-c against the
    np_ceiling policy before any activation. MTP OFF per program decision D6:
    no speculative args.
    """
    source_role = GPU_SHADOW_LANE_TENANT_ROLE
    requirements, runtime = _stack_prior_launch(source_role)
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    # Backend, not a build path. A literal here is how a GPU lane silently acquires
    # whatever that directory happens to contain. Resolved LAZILY: a declared
    # binary_path from the compiled priors is authoritative and must not be made to
    # depend on the store, but with nothing declared an unresolvable `gpu` backend is
    # fatal — `_kernel_server_binary` raises rather than substituting a CPU build.
    declared_binary = _runtime_string(runtime, "binary_path", "")
    cmd = [
        declared_binary or str(_kernel_server_binary("gpu")),
        "-m",
        _runtime_string(requirements, "model_path", ""),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--metrics",
        "--slots",
        "--jinja",
        "--device",
        str(flags.get("device") or GPU_SHADOW_LANE_DEVICE),
        "-ngl",
        "all",
        "-fa",
        "on",
        "-np",
        _runtime_positive_int(cache, "slots", GPU_SHADOW_LANE_FALLBACK_SLOTS),
        "-c",
        _runtime_positive_int(
            cache, "context_tokens", GPU_SHADOW_LANE_FALLBACK_CONTEXT_TOKENS
        ),
        "-t",
        _resolve_thread_count("gpu_shadow_lane", numa_instance),
        "-tb",
        _resolve_thread_count("gpu_shadow_lane", numa_instance),
        "-b",
        "2048",
        "-ub",
        "2048",
        "-ctk",
        _runtime_string(cache, "kv_type_k", "f16"),
        "-ctv",
        _runtime_string(cache, "kv_type_v", "f16"),
        "--log-colors",
        "off",
    ]
    reasoning = flags.get("reasoning") or GPU_SHADOW_LANE_REASONING
    if isinstance(reasoning, str) and reasoning:
        cmd.extend(["--reasoning", reasoning])
    return cmd


# -----------------------------------------------------------------------------
# Default-role builder sub-helpers (called by _build_role_command).
# -----------------------------------------------------------------------------

# Use v2 binary for roles with v3 spec decode bug (Qwen2.5 architecture).
# Currently empty (was {"coder_escalation"}); kept here for clarity of intent.
_NO_SPEC_DECODE = NO_SPEC_DECODE_ROLES

# Role-specific KV cache budgets and quantization.
# Phase 0 benchmarks (2026-03-25): generation speed neutral, memory savings significant at 65K+.
# CRITICAL (2026-03-28): V=q4_0 causes 71% prefill regression on pure-attention models.
# V=f16 has ZERO prefill regression (actually 1% faster due to K bandwidth savings).
# q4_0 K / f16 V = quality-neutral (PPL +0.017 with Hadamard), 37% KV savings, zero speed cost.
# q4_0 / q4_0 = 71% KV savings but 71% prefill regression on pure-attn. OK for hybrid (SSM amortizes).
# --kv-hadamard: production binary rebuilt with Hadamard support (commit b51c905ec, 2026-03-28).
# 2026-09-22 lineup change: built from the roles that HAVE their own launch
# context, instead of indexing two fixed names. ingest_long_context became an
# ALIAS on architect_general's :8083 server (architect_critic's since the
# 2026-09-27 swap) and no longer has an entry here, so
# the old literal `LAUNCH_CONTEXT_TOKENS["ingest_long_context"]` raised KeyError
# at import and took down every entry point that imports this module -- the
# stack launcher included. An alias has no context of its own by definition; it
# inherits its host's, and a table keyed on "roles that launch a server" is the
# thing that stays true across lineups.
# 2026-09-27 ARCHITECT SWAP: the :8083 27B host (and ingest's host) is
# architect_critic; architect_general is the :8074 Flash-Next launch row. Both
# launch a server, so both carry their own launch context fallback.
_KV_CONTEXT_ROLES = ("architect_general", "architect_critic", "ingest_long_context")
_KV_CONTEXT_SIZES = {
    role: str(LAUNCH_CONTEXT_TOKENS[role])
    for role in _KV_CONTEXT_ROLES
    if role in LAUNCH_CONTEXT_TOKENS
}
_KV_QUANT_CONFIGS = LAUNCH_KV_QUANT_CONFIGS


def _resolve_thread_count(role_name: str, numa_instance: int = 0) -> str:
    """NUMA-aware thread count for the given role + instance index (fallback: 96).

    Pre-2026-05-24 this function always returned `instances[0][2]` regardless of
    which instance was being launched. The launcher therefore always passed
    `-t 96` to every frontdoor quarter (which intends `-t 48`), `-t 48` to
    every worker_general quarter (which intends `-t 48` correctly only because
    worker_general had a manual workaround in the worker-general MTP helper),
    and so on. Threading `numa_instance` through lets each instance get the
    thread count its NUMA_CONFIG entry actually specifies.
    """
    numa_cfg = NUMA_CONFIG.get(role_name)
    if numa_cfg and numa_cfg["instances"]:
        # Defensive: out-of-range instance falls back to the first instance's
        # thread count rather than crashing — the wrong number is better than
        # a launcher abort during stack startup.
        idx = numa_instance if 0 <= numa_instance < len(numa_cfg["instances"]) else 0
        return str(numa_cfg["instances"][idx][2])
    return "96"


def _resolve_binary_for_role(role_name: str) -> Path:
    """Pick LLAMA_SERVER_V2 if the role is on the v2 binary allow-list and the binary exists."""
    if role_name in _V2_ROLES and LLAMA_SERVER_V2.exists():
        return LLAMA_SERVER_V2
    return LLAMA_SERVER


def _append_kv_quant_args(cmd: list[str], role_name: str) -> None:
    """Emit -ctk/-ctv (and --kv-hadamard for v2 binary) for roles with a KV quant config."""
    kv_quant = _KV_QUANT_CONFIGS.get(role_name)
    if not kv_quant:
        return
    cmd.extend(["-ctk", kv_quant[0], "-ctv", kv_quant[1]])
    # --kv-hadamard: v3 auto-enables (upstream #21038), v2 needs explicit flag
    if role_name in _V2_ROLES and LLAMA_SERVER_V2.exists():
        # 2026-06-26 v6 cutover: --kv-hadamard removed in v6 (would crash); no role on v2 today
        # cmd.append("--kv-hadamard")
        pass


def _append_acceleration_args(cmd: list[str], role_name: str, accel: Any, model_path: str) -> None:
    """Emit acceleration-mode-specific args (MoE expert reduction / spec decode / self-spec).

    2026-05-09: architect_general is gated out of speculative_decoding because
    Qwen3.5-122B M-RoPE refuses position rollback when speculative draft tokens
    are rejected — see `_NO_SPEC_DECODE` and the comment block above its
    definition for the full incident trace.
    """
    if accel.type == "moe_expert_reduction" and accel.experts:
        cmd.extend(["--override-kv", f"{accel.override_key}=int:{accel.experts}"])
    elif (
        accel.type == "speculative_decoding"
        and accel.draft_role
        and role_name not in _NO_SPEC_DECODE
    ):
        registry = RegistryLoader()
        draft_config = registry.get_role(accel.draft_role)
        if draft_config:
            cmd.extend(
                [
                    "-md",
                    draft_config.model.full_path,
                    "--spec-draft-n-max",  # 2026-06-26 v6 cutover: renamed from --draft-max
                    str(accel.k or 16),
                ]
            )

    # MoE + spec decode combo (e.g., 480B with jukofyork draft + expert reduction)
    if (
        accel.type == "moe_expert_reduction"
        and accel.draft_role
        and role_name not in _NO_SPEC_DECODE
    ):
        registry = RegistryLoader()
        draft_config = registry.get_role(accel.draft_role)
        if draft_config:
            cmd.extend(
                [
                    "-md",
                    draft_config.model.full_path,
                    "--spec-draft-n-max",  # 2026-06-26 v6 cutover: renamed from --draft-max
                    str(accel.k or 16),
                ]
            )

    # Self-speculation: same model as target and draft, draft exits early
    elif accel.type == "self_speculation" and accel.n_layer_exit_draft:
        cmd.extend(
            [
                "--n-layer-exit-draft",
                str(accel.n_layer_exit_draft),
                "--spec-draft-n-max",  # 2026-06-26 v6 cutover: renamed from --draft-max
                str(accel.k or 16),
            ]
        )

    # Hierarchical speculation: self-spec with intermediate verification
    elif accel.type == "hierarchical_speculation":
        cmd.extend(
            [
                "--n-layer-exit-draft",
                str(accel.n_layer_exit_draft or 0),
                "--hierarchical-spec",
                "--spec-draft-n-max",  # 2026-06-26 v6 cutover: renamed from --draft-max
                str(accel.k or 16),
            ]
        )
        if accel.n_layer_exit_intermediate:
            cmd.extend(["--n-layer-exit-intermediate", str(accel.n_layer_exit_intermediate)])


def _apply_numa_spec_overrides(cmd: list[str], numa_cfg: dict | None) -> None:
    """In-place rewrite of --draft-max based on NUMA spec_overrides.

    When NUMA thread count differs from 192t, the optimal draft_max may differ.
    Overrides come from bench_sweep_spec_params.sh results stored in NUMA_CONFIG.
    """
    if not numa_cfg or "spec_overrides" not in numa_cfg:
        return
    overrides = numa_cfg["spec_overrides"]
    if "draft_max" in overrides:
        for i, arg in enumerate(cmd):
            # 2026-06-26 v6 cutover: match renamed --spec-draft-n-max (was --draft-max)
            if arg == "--spec-draft-n-max" and i + 1 < len(cmd):
                cmd[i + 1] = str(overrides["draft_max"])
                break
    # Tree-spec (--draft-p-split) and bare --lookup flag both stripped in v5 binary;
    # see git history for re-enable conditions if/when a future binary restores them.


def _build_role_command(
    role_config: Any,
    port: int,
    numa_instance: int = 0,
    prepare_runtime_dirs: bool = True,
) -> list[str]:
    """Build llama-server command for a registry-backed role (default path).

    `numa_instance` selects which entry in NUMA_CONFIG[role]["instances"] this
    invocation refers to. 0 = the primary/full instance; 1..N = quarter
    instances. The thread count comes from that specific instance's tuple so
    quarters get `-t 48` and the full instance gets its declared `-t 96`.
    """
    if not isinstance(prepare_runtime_dirs, bool):
        raise TypeError("prepare_runtime_dirs must be bool")
    model_path = role_config.model.full_path
    accel = role_config.acceleration
    role_name = role_config.name
    _requirements, runtime = _stack_prior_launch(role_name)
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    # `-np` comes from the DECLARED server_mode.<role>.slots, not from
    # SERIAL_ROLES. The old formula here (`1 if role_name in SERIAL_ROLES else 2`)
    # answered an admission question with a serving number and disagreed with the
    # declaration for 6 of the 11 roles master declares slots for.
    # 2026-08-02: per-INSTANCE, not per-role. frontdoor's full runs -np 16 and each
    # of its halves -np 4 off the same compiled record.
    parallel_slots = _resolve_parallel_slots(cache, role_name, port, numa_instance)
    thread_count = _resolve_thread_count(role_name, numa_instance)
    context_size = _runtime_positive_int(
        cache,
        "context_tokens",
        _KV_CONTEXT_SIZES.get(role_name, str(DEFAULT_EFFECTIVE_CONTEXT_TOKENS)),
    )
    binary = _runtime_string(runtime, "binary_path", str(_resolve_binary_for_role(role_name)))
    ubatch = _runtime_positive_int(cache, "ubatch", DEFAULT_UBATCH_TOKENS)

    # K4 (handoffs/active/dynamic-stack-concurrency.md): this builder never
    # emits -b, and llama.cpp clamps cparams.n_ubatch = min(n_batch, n_ubatch)
    # against the DEFAULT n_batch (2048) whenever -b is absent. So the -ub
    # value below is a ceiling that is only ever real up to 2048 — raising it
    # further requires ALSO passing -b, plus a measured window (this is not a
    # config-only change). See _warn_if_ubatch_exceeds_batch just below.
    _warn_if_ubatch_exceeds_batch(role_name, ubatch)
    cmd = [
        binary,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        parallel_slots,
        "-c",
        context_size,
        "-t",
        thread_count,
    ]
    cmd.extend(["-ub", ubatch])
    if flags.get("flash_attn", True) is True:
        cmd.extend(["--flash-attn", "on"])

    # --jinja: model's native chat template (enables thinking on Qwen3/3.5).
    # 2026-08-21: the architect_general exclusion that used to live here was
    # REVERSED upstream on 2026-06-26 (commit f4a8a3ca; gated on the J12
    # think-loop probe — 0 leaks/loops over n=15,
    # orchestration/reports/j12_think_loop_probe_20260706T143621Z). The compiled
    # priors (stack_priors.py -> derived jinja: true) are authoritative; this
    # default only fires when a role has no compiled prior, and the old
    # `role_name != "architect_general"` default silently dropped --jinja for
    # exactly that role on the no-priors path — the reversed policy surviving
    # as a live fallback. Audit: per-request-reasoning-budget.md PRB-T1.
    if flags.get("jinja", True) is True:
        cmd.append("--jinja")

    if runtime:
        _append_runtime_kv_args(cmd, cache)
    else:
        _append_kv_quant_args(cmd, role_name)

    # mlock: lock model weights in RAM to prevent page cache eviction.
    # Validated in S2: 30x latency improvement under memory pressure.
    if cache.get("mlock", role_name in MLOCK_ROLES) is True:
        cmd.append("--mlock")

    # 2026-06-26 v6 cutover: honor the per-role no_mmap prior (enables N12 topology).
    # Mirrors _build_worker_general_command's cache.get("no_mmap", ...) emit. Default
    # False here so generic-role behavior is unchanged when no_mmap is absent/False.
    if cache.get("no_mmap", False) is True:
        cmd.append("--no-mmap")

    if runtime:
        for override in flags.get("override_kv") or []:
            if isinstance(override, str) and override:
                cmd.extend(["--override-kv", override])
        # Declared serving flags (-ngl / --image-min-tokens / --cache-ram). Inert for
        # every role that does not declare them, which today is every role on this
        # branch — but the vision defect existed precisely because the declaration
        # had nowhere to land, so the general builder gets the same door.
        _append_runtime_serving_flags(cmd, flags)
        _append_runtime_spec_args(cmd, runtime, model_path)
        reasoning = flags.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            cmd.extend(["--reasoning", reasoning])
        # `device`, emitted exactly as _build_vision_command emits it. This builder
        # read flash_attn, jinja, reasoning, override_kv and the whole spec block
        # but never `device`, so a role's declared processor was the one compiled
        # field with nowhere to land on the default path — and _append_device_args
        # then filled the silence with "none".
        device = flags.get("device")
        if isinstance(device, str) and device:
            cmd.extend(["--device", device])
        device_draft = flags.get("device_draft")
        if isinstance(device_draft, str) and device_draft:
            cmd.extend(["--device-draft", device_draft])
    else:
        _append_acceleration_args(cmd, role_name, accel, model_path)
    _apply_numa_spec_overrides(cmd, NUMA_CONFIG.get(role_name))

    # DS-3: KV state save/restore — per-role subdir to avoid slot ID collisions.
    slot_save_path = cache.get("slot_save_path")
    slot_dir = (
        Path(slot_save_path)
        if isinstance(slot_save_path, str) and slot_save_path
        else SLOT_SAVE_DIR / role_name
    )
    if prepare_runtime_dirs:
        slot_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--slot-save-path", str(slot_dir)])

    return cmd


def _dispatch_prior_role(
    role_config: Any,
    *,
    dev_mode: bool,
    embedding_mode: bool,
    worker_pool_mode: bool,
    worker_type: str | None,
    vision_mode: bool,
    vision_type: str | None,
    eval_batch_frontdoor_mode: bool,
    gpu_shadow_lane_mode: bool,
) -> str | None:
    """Which role's compiled priors describe the command build_server_command emits.

    Mirrors the dispatcher's own branch order, and returns the SAME role name each
    builder already looks its priors up under — so `--device` is resolved from the
    identical record that supplied the binary, context and slots. Returns None for
    the shapes that have no registry role at all (dev, embedders, worker_fast);
    those get the CPU default, which is what they had before.
    """
    if vision_mode:
        return "vision_escalation" if vision_type == "escalation" else "worker_vision"
    if embedding_mode:
        return None
    if eval_batch_frontdoor_mode:
        return "frontdoor"
    if gpu_shadow_lane_mode:
        return GPU_SHADOW_LANE_TENANT_ROLE
    if worker_pool_mode and worker_type:
        # _build_worker_general_command reads worker_general's priors for BOTH the
        # general and explore worker types; worker_fast reads none.
        return None if worker_type == "fast" else "worker_general"
    if dev_mode:
        return None
    name = getattr(role_config, "name", None)
    return str(name) if isinstance(name, str) and name else None


def build_server_command(
    role_config: Any,
    port: int,
    dev_mode: bool = False,
    embedding_mode: bool = False,
    worker_pool_mode: bool = False,
    worker_type: str = None,
    vision_mode: bool = False,
    vision_type: str = None,
    eval_batch_frontdoor_mode: bool = False,
    gpu_shadow_lane_mode: bool = False,
    binary_override: str | None = None,
    numa_instance: int = 0,
    prepare_runtime_dirs: bool = True,
) -> list[str]:
    """Dispatch to the per-mode command builder.

    `binary_override`: when set, replaces `LLAMA_SERVER` for the worker_pool
    general branch (used by worker_general / gemma4 MTP to launch ik_llama.cpp
    PR #1744). Other branches ignore this argument today.

    `numa_instance`: which instance in NUMA_CONFIG[role]["instances"] is being
    launched (0 = full/primary, 1..N = quarters). Used by `_build_role_command`
    to pick per-instance thread count. Defaults to 0 so callers that don't
    care about quarters (vision, embedding, dev, worker_pool) are unaffected.
    """
    if not isinstance(prepare_runtime_dirs, bool):
        raise TypeError("prepare_runtime_dirs must be bool")
    prior_role = _dispatch_prior_role(
        role_config,
        dev_mode=dev_mode,
        embedding_mode=embedding_mode,
        worker_pool_mode=worker_pool_mode,
        worker_type=worker_type,
        vision_mode=vision_mode,
        vision_type=vision_type,
        eval_batch_frontdoor_mode=eval_batch_frontdoor_mode,
        gpu_shadow_lane_mode=gpu_shadow_lane_mode,
    )
    if vision_mode:
        cmd = _build_vision_command(port, vision_type, numa_instance)
    elif embedding_mode:
        cmd = _build_embedding_command(port)
    elif eval_batch_frontdoor_mode:
        cmd = _build_eval_batch_frontdoor_command(port, numa_instance)
    elif gpu_shadow_lane_mode:
        # P2-6/P0-2: fires only for a server entry carrying the gpu_shadow_lane
        # mode flag — absent from every entry until the registry proposal is
        # applied (State-A inertness witness in tests/unit/test_gpu_shadow_lane.py).
        cmd = _build_gpu_shadow_lane_command(port, numa_instance)
    elif worker_pool_mode and worker_type:
        model_path = WORKER_POOL_MODELS.get(worker_type)
        if not model_path:
            raise ValueError(f"Unknown worker type: {worker_type}")
        if worker_type == "fast":
            cmd = _build_worker_fast_command(port, model_path)
        else:
            cmd = _build_worker_general_command(port, model_path, binary_override, numa_instance)
    elif dev_mode:
        cmd = _build_dev_command(port)
    else:
        if prepare_runtime_dirs:
            cmd = _build_role_command(role_config, port, numa_instance)
        else:
            cmd = _build_role_command(
                role_config, port, numa_instance, prepare_runtime_dirs=False)
    # Pass the role's DECLARED device down. Every builder above is covered, not just
    # the two that emit `--device` themselves — a declaration must not depend on
    # which launch shape a role happens to use.
    prior_flags = _runtime_flags(_stack_prior_launch(prior_role)[1]) if prior_role else None
    _append_device_args(cmd, prior_flags)
    return cmd


def start_server(
    port: int,
    roles: list[str],
    registry: RegistryLoader,
    dev_mode: bool = False,
    embedding_mode: bool = False,
    worker_pool_mode: bool = False,
    worker_type: str = None,
    vision_mode: bool = False,
    vision_type: str = None,
    eval_batch_frontdoor_mode: bool = False,
    gpu_shadow_lane_mode: bool = False,
    numa_instance: int = 0,
    bench_force: bool = False,
    env_override: dict[str, str] | None = None,
) -> ProcessInfo | None:
    """Start a llama-server for the given roles.

    ``env_override`` is the one-shot experiment env (UFH-12 arm A3,
    scripts/server/embedder_env_override.py). It is honoured ONLY for embedders
    and applied last, over the declared env; every other branch refuses it.
    """
    if env_override and not embedding_mode:
        raise ValueError("env_override is only honoured for embedders (embedding_mode=True)")
    detached_stdio = {
        "stdin": subprocess.DEVNULL,
        "start_new_session": True,
        "close_fds": True,
    }

    # SS-BENCH-GATE-b: ONE guarded placement decision per spawn, before any
    # branch. Every branch below pins with `_numa_prefix(roles[0], numa_instance)`
    # (primary_role is assigned roles[0] in each), so one prefix serves them
    # all; a live bench's real core claim refuses overlapping placements or
    # pins default-affinity spawns off the claim. `bench_force` mirrors
    # --allow-during-bench.
    declared_embedding = EMBEDDING_PLACEMENT.get(port) if embedding_mode else None
    if declared_embedding is not None:
        # UFH-12 Phase 0: an embedder with a DECLARED placement is pinned to
        # exactly that cpuset (refuse-never-re-pin, like a declared aux cpuset).
        spawn_prefix = _embedding_spawn_prefix(port, declared_embedding, bench_force=bench_force)
        if spawn_prefix is None:
            return None
    else:
        spawn_prefix = _bench_guarded_numa_prefix(
            roles[0] if roles else None,
            numa_instance,
            bench_force=bench_force,
            label=f"llama-server for {'/'.join(roles) or '?'}",
        )

    # P-BENCH-3/A7 warm eval-batch lane: dedicated frontdoor-model server
    # used only when explicitly started and when EVAL_BATCH_SERVING routes
    # eval-batch traffic to it.
    if eval_batch_frontdoor_mode:
        primary_role = roles[0]
        source_role = "frontdoor"
        log_file = LOG_DIR / f"eval-batch-frontdoor-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        cmd = build_server_command(
            None,
            port,
            eval_batch_frontdoor_mode=True,
            numa_instance=numa_instance,
        )
        requirements, _runtime = _stack_prior_launch(source_role)
        model_path = _runtime_string(requirements, "model_path", "")
        model_name = Path(model_path).name if model_path else "frontdoor model"

        print(f"  Starting eval-batch frontdoor on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")

        with open(log_file, "a") as log:
            env = build_launch_env(source_role, os.environ.copy())
            # No binary override on this branch either; strip ambient GGML_* like every
            # other branch (2026-09-26).
            _strip_ambient_ggml(env, _role_env_overrides(source_role), label="eval_batch")
            proc = subprocess.Popen(
                spawn_prefix + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        if wait_for_health(port, timeout=max(_HEALTH_SERVER_STARTUP, 180)):
            print("    [OK] Eval-batch frontdoor ready")
            return ProcessInfo(
                role=primary_role,
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )

        print("    [FAIL] Eval-batch frontdoor did not become healthy")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None

    # gpu-serving-tie-in P2-6/P0-2: GPU shadow lane (docs/gpu-shadow-lane.md).
    # INERT today — reachable only when a server entry carries the
    # gpu_shadow_lane mode flag, which no entry does until the registry
    # proposal is applied. Explicit-only warm lane; never part of a normal
    # `start` (same contract as eval_batch_frontdoor).
    if gpu_shadow_lane_mode:
        primary_role = roles[0]
        source_role = GPU_SHADOW_LANE_TENANT_ROLE
        log_file = LOG_DIR / f"gpu-shadow-lane-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        cmd = build_server_command(
            None,
            port,
            gpu_shadow_lane_mode=True,
            numa_instance=numa_instance,
        )
        requirements, _runtime = _stack_prior_launch(source_role)
        model_path = _runtime_string(requirements, "model_path", "")
        model_name = Path(model_path).name if model_path else "gpu shadow lane tenant"
        binary_override, ld_paths = _stack_prior_runtime_overrides(source_role)

        print(f"  Starting GPU shadow lane on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)} (tenant priors: {source_role})")
        print(f"    Command: {' '.join(cmd[:6])}...")

        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")

        with open(log_file, "a") as log:
            env = build_launch_env(source_role, os.environ.copy())
            # HIP tree binary + LD paths come from the tenant's compiled priors
            # (binary_dir/ld_library_path -> env_policy binary_override_strip_ggml).
            _apply_runtime_requirements_env(
                env,
                binary_override=binary_override,
                ld_paths=ld_paths,
                preserve=_role_env_overrides(source_role),
            )
            proc = subprocess.Popen(
                spawn_prefix + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        if wait_for_health(port, timeout=max(_HEALTH_SERVER_STARTUP, 300)):
            print("    [OK] GPU shadow lane ready")
            return ProcessInfo(
                role=primary_role,
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )

        print("    [FAIL] GPU shadow lane did not become healthy")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None

    # Vision mode - VL models with multimodal projector
    if vision_mode:
        log_file = LOG_DIR / f"vision-{vision_type or 'worker'}-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        binary_override, ld_paths = _stack_prior_runtime_overrides(roles[0])

        if vision_type == "escalation":
            model_path = VISION_ESCALATION_MODEL
            model_name = "Qwen2.5-VL-7B Q4_K_M (temporary vision escalation alias)"
        else:
            model_path = VISION_WORKER_MODEL
            model_name = "Qwen2.5-VL-7B (vision worker)"

        cmd = build_server_command(
            None,
            port,
            vision_mode=True,
            vision_type=vision_type,
            numa_instance=numa_instance,  # fix: forward so quarters get NUMA_CONFIG -t (was always -t 96)
        )

        print(f"  Starting vision server [{vision_type or 'worker'}] on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        if binary_override:
            print(f"    Binary override: {binary_override}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        # Fleet marker: written BEFORE Popen so subsequent watcher polls
        # see the new startup timestamp immediately + can resolve role→port.
        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")

        with open(log_file, "a") as log:
            env = build_launch_env(roles[0], os.environ.copy())
            _apply_runtime_requirements_env(
                env,
                binary_override=binary_override,
                ld_paths=ld_paths,
                preserve=_role_env_overrides(roles[0]),
            )
            proc = subprocess.Popen(
                spawn_prefix + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        # VL models take longer to load (mmproj + main model)
        timeout = _HEALTH_VISION_SERVER if vision_type == "escalation" else _HEALTH_WORKER_SERVER
        if wait_for_health(port, timeout=timeout):
            print(f"    [OK] Vision server {vision_type or 'worker'} ready")
            return ProcessInfo(
                role=roles[0],
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Vision server {vision_type or 'worker'} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Embedding mode uses dedicated config, no registry lookup needed
    if embedding_mode:
        log_file = LOG_DIR / f"embedder-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        cmd = build_server_command(None, port, dev_mode=False, embedding_mode=True)
        recipe = EMBEDDING_SERVER_RECIPES.get(port, {})
        model_name = str(recipe.get("model_name", "embedding model"))
        instance_idx = port - 8090 if port in EMBEDDER_PORTS else port

        print(f"  Starting embedder #{instance_idx} on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        # Fleet marker: written BEFORE Popen so the watcher can resolve
        # role→port and detect operator-initiated reloads.
        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")

        with open(log_file, "a") as log:
            env = build_launch_env(roles[0], os.environ.copy())
            # The embedding branch has no binary override, so it used to pass ambient GGML_*
            # straight through. Strip it like every other branch (2026-09-26).
            _strip_ambient_ggml(env, _role_env_overrides(roles[0]), label="embedding")
            if env_override:
                # One-shot experiment env (UFH-12 A3): applied last, recorded by the caller in
                # logs/embedder_env_override.json, and never read back by any later launch.
                env.update(env_override)
                print(f"    [EXPERIMENT] env override: {env_override}")
            # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
            proc = subprocess.Popen(
                spawn_prefix + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        if wait_for_health(port, timeout=60):  # Faster timeout for small model
            print(f"    [OK] Embedder #{instance_idx} ready")
            return ProcessInfo(
                role=roles[0],  # Use actual role name (embedder, embedder_1, etc.)
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=str(recipe.get("model_path", EMBEDDING_MODEL_PATH)),
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Embedder #{instance_idx} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Worker pool mode - heterogeneous workers
    if worker_pool_mode and worker_type:
        log_file = LOG_DIR / f"worker-{worker_type}-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        model_path = WORKER_POOL_MODELS.get(worker_type)
        if not model_path:
            print(f"  [!] Unknown worker type: {worker_type}")
            return None

        # Per-role binary + LD_LIBRARY_PATH override. Lookup is keyed on the
        # primary role (e.g. "worker_general"), not worker_type.
        binary_dir, ld_paths = _runtime_requirements_for_role(registry, roles[0])
        binary_override = str(Path(binary_dir) / "llama-server") if binary_dir else None

        cmd = build_server_command(
            None,
            port,
            worker_pool_mode=True,
            worker_type=worker_type,
            binary_override=binary_override,
            numa_instance=numa_instance,  # fix: forward so gemma4 quarters get NUMA_CONFIG -t 48 (was always -t 96)
        )
        model_name = Path(model_path).stem

        print(f"  Starting worker pool [{worker_type}] on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        if binary_override:
            print(f"    Binary override: {binary_override}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        with open(log_file, "a") as log:
            # Worker pool roles map their worker_type to the canonical "worker" role for env.
            env = build_launch_env("worker", os.environ.copy())
            _apply_runtime_requirements_env(
                env,
                binary_override=binary_override,
                ld_paths=ld_paths,
                preserve=_role_env_overrides("worker"),
            )
            # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
            # Fleet marker: written BEFORE Popen so the watcher can resolve
            # role→port and detect operator-initiated reloads.
            try:
                _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
            except Exception as exc:
                print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")
            proc = subprocess.Popen(
                spawn_prefix + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        # Faster timeout for smaller models (quick_check for fast workers)
        timeout = (
            int(_registry_timeout("health", "quick_check", 10)) * 6
            if worker_type == "fast"
            else _HEALTH_WORKER_SERVER
        )
        if wait_for_health(port, timeout=timeout):
            print(f"    [OK] Worker {worker_type} ready")
            # 2026-05-09: per-thread renice to nice=19 for binary_override roles
            # (ik_llama.cpp PR #1744 / gemma4 MTP). Reason: PR #1744's OMP_WAIT_POLICY=active
            # busy-loops 96 cores when idle, contaminating other-role measurements by
            # 30-69% throughput. CLI `renice -p PID` only affects the lead thread, so we
            # iterate /proc/<pid>/task/<tid> after model load completes (all OMP team
            # threads are spawned by health-check time).
            # Verified 2026-05-09: post-renice, frontdoor 4.55→7.11 t/s (+56%),
            # coder 4.02→12.34 (+207%), ingest 10.46→28.99 (+177%).
            # No sudo needed — increasing nice (lower priority) is permitted for owner.
            if binary_override:
                _renice_all_threads(proc.pid, 19)
            return ProcessInfo(
                role=f"worker_{worker_type}",
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Worker {worker_type} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Use first role's config for the server
    primary_role = roles[0]
    role_config = registry.get_role(primary_role)

    if not role_config and not dev_mode:
        print(f"  [!] Role {primary_role} not found in registry")
        return None

    log_file = LOG_DIR / f"llama-server-{port}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build command. `numa_instance` flows through so per-quarter thread counts
    # come from the role's NUMA_CONFIG entry rather than always defaulting to
    # the first instance's count.
    cmd = build_server_command(role_config, port, dev_mode, numa_instance=numa_instance)

    model_name = DEV_MODEL if dev_mode else role_config.model.name
    numa_cfg = NUMA_CONFIG.get(primary_role)
    numa_label = ""
    if numa_cfg and numa_instance < len(numa_cfg["instances"]):
        cpu_list = numa_cfg["instances"][numa_instance][0]
        numa_label = f" [NUMA {numa_instance}: cpus {cpu_list}]"

    print(f"  Starting port {port}: {model_name}{numa_label}")
    print(f"    Roles: {', '.join(roles)}")
    print(f"    Command: {' '.join(cmd[:5])}...")

    # INF-70/C7 (ENABLED 2026-09-03 for every CPU llama-server role): force
    # free pages onto every NUMA node before the load. `--interleave=all` in
    # spawn_prefix is a per-allocation HINT the kernel silently abandons for
    # any node that has none, so a large model on a long-lived box can land
    # 61%-on-one-node with an identical argv (measured 2026-09-02: decode
    # -25%). Sized the FORCING way (TARGET+2 on any node below TARGET, verify,
    # 2 passes) — see stack_numa_evict.py. Per role via `numa_pre_evict_gib`
    # in orchestration/stack_topology.yaml; a gpu_host_lane role resolves to 0
    # here whatever the YAML says, so no GPU launch ever evicts against node 3.
    _pre_evict_gib = pre_evict_gib_for_role(NUMA_CONFIG.get(primary_role))
    if _pre_evict_gib > 0:
        print(f"    [numa-pre-evict] forcing {_pre_evict_gib} GiB free per node...")
        _ok, _msg = pre_evict_nodes(_pre_evict_gib)
        print(f"    [numa-pre-evict] {'[OK]' if _ok else '[WARN]'} {_msg}")

    # Start process — taskset CPU-pinned per NUMA config + canonical OMP env + per-role GGML
    with open(log_file, "a") as log:
        env = build_launch_env(primary_role, os.environ.copy())
        binary_override, ld_paths = _stack_prior_runtime_overrides(primary_role)
        _apply_runtime_requirements_env(
            env,
            binary_override=binary_override,
            ld_paths=ld_paths,
            preserve=_role_env_overrides(primary_role),
        )
        # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
        # Fleet marker: written BEFORE Popen so the watcher can resolve
        # role→port and detect operator-initiated reloads.
        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")
        proc = subprocess.Popen(
            spawn_prefix + cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            **detached_stdio,
        )

    print(f"    PID: {proc.pid}")

    # Wait for health
    print("    Waiting for health...")
    if wait_for_health(port, timeout=180):
        print("    [OK] Server ready")
        # INF-70/C7: placement observation, not a gate. The weights are faulted
        # in by the time health passes, so this is the one moment the operator
        # can see whether --interleave=all was actually honoured. A share far
        # above 100/nodes % means the model is skewed and decode is paying for
        # it; the remedy is `numa_pre_evict_gib` above.
        print(f"    [numa-placement] pid={proc.pid} {placement_summary(proc.pid)}")
        return ProcessInfo(
            role=primary_role,
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path=DEV_MODEL_PATH if dev_mode else role_config.model.full_path,
            log_file=str(log_file),
        )
    else:
        print("    [FAIL] Server did not become healthy")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def _apply_orchestrator_profile(env: dict[str, str], profile: str | None) -> None:
    """Apply optional orchestrator profile env vars without overriding explicit env."""
    if not profile:
        return
    profile_vars = ORCHESTRATOR_PROFILES.get(profile)
    if not profile_vars:
        print(f"    [WARN] Unknown orchestrator profile '{profile}' (ignored)")
        return
    print(f"    Using orchestrator profile: {profile}")
    for key, value in profile_vars.items():
        env.setdefault(key, value)


PRODUCTION_FEATURE_WAVE_OVERRIDES: dict[str, bool] = {
    # Fable5 routing-truth W1: conservative wave-1 production intent.
    "specialist_routing": True,
    "model_fallback": True,
    # Wave-2 paths have been dormant for months; keep them declared-off until
    # each has its own observation window.
    "plan_review": False,
    "architect_delegation": False,
    "parallel_execution": False,
    "unified_streaming": False,
    # Weights have been absent since the 2026-05-25 memory reset.
    "routing_classifier": False,
    # OP-51 operator 2026-09-24: enable REPL /chat output_schema validation
    # (TD-21.1 FINAL() repair + 422-on-exhaustion, orch d7202dc9, evidence-
    # guarded since 2ffc3f51). Before: /chat force_mode=repl with an
    # output_schema returned the bare, unvalidated FINAL() value. After: the
    # value is validated against the caller's JSON Schema, repaired once on
    # failure, and the response is flagged 422 when still invalid after
    # exhausting the repair attempt.
    # OP-51 RE-ENABLED 2026-09-24 after an on/off A/B (15:35Z, region free): 3/3 on ->
    # HTTP 200 schema-valid {"result": 51}; 3/3 off -> HTTP 200 bare "51". The suspended
    # 13:41Z failures were a held CPU region (AutoKernel calibration), not this flag.
    "final_schema_validation": True,
    # HS-19a stage 1 ("Linked"), operator-approved 2026-09-27 for production: record
    # harness subagent parent->child session links (x-parent-session-id / body key) in
    # the inference tap and session log. Metadata only: no routing, admission or
    # response change (orch f58db8be; 45 unit tests). Enables the HS-19a live run.
    "v1_subagent_link": True,
    # UFH-13 TE-1, operator-approved 2026-09-28 for production: /v1 frontdoor
    # escalation. Opt-in per request (x_escalation=auto|architect_general); a request
    # without the key is served exactly as with the flag off (orch 280059cc). Enables
    # the thesis experiment's A2 arm.
    "v1_escalation": True,
    # RI-23 / OP-69 option (a), operator-approved 2026-09-29 for production: thinking-on roles
    # (architect_critic, coder_escalation, ingest_long_context) use the llama-server chat lane so
    # their template + chat_template_kwargs apply and reasoning is server-split; the review verdict
    # runs thinking-off with skip_suffix (fixes the RI-22 no-op gate). RI-23b A/B 2026-09-29:
    # verdicts parse (0/2 -> 2/2), code/ingest/REPL probes correct and faster.
    "thinking_roles_chat_lane": True,
}

LANGGRAPH_PHASE3_LIVE_ENV_VARS: tuple[str, ...] = (
    "ORCHESTRATOR_LANGGRAPH_INGEST",
    "ORCHESTRATOR_LANGGRAPH_WORKER",
    "ORCHESTRATOR_LANGGRAPH_FRONTDOOR",
    "ORCHESTRATOR_LANGGRAPH_CODER",
    "ORCHESTRATOR_LANGGRAPH_CODER_ESCALATION",
    "ORCHESTRATOR_LANGGRAPH_ARCHITECT",
)


def _production_feature_env() -> dict[str, str]:
    """Complete, attestable ORCHESTRATOR_FEATURE_* block for API workers."""
    from src.features import FEATURE_ENV_PREFIX, _FEATURE_REGISTRY, _REGISTRY_BY_NAME

    block = {
        f"{FEATURE_ENV_PREFIX}{spec.env_var}": "1" if spec.default_prod else "0"
        for spec in _FEATURE_REGISTRY
    }
    for name, enabled in PRODUCTION_FEATURE_WAVE_OVERRIDES.items():
        spec = _REGISTRY_BY_NAME[name]
        block[f"{FEATURE_ENV_PREFIX}{spec.env_var}"] = "1" if enabled else "0"
    return block


def _apply_production_feature_env(env: MutableMapping[str, str]) -> None:
    """Fill stack-managed production feature defaults without clobbering overrides."""
    for key, value in _production_feature_env().items():
        env.setdefault(key, value)


def start_orchestrator(
    profile: str | None = None,
    stack_numa_mode: str | None = None,
    bench_force: bool = False,
) -> ProcessInfo | None:
    """Start the orchestrator API."""
    # A unit test that misses one lifecycle monkeypatch must fail inside pytest,
    # never escape as a long-lived production listener.  On 2026-08-05
    # test_cmd_start_infers_missing_numa_mode_from_realized_fleet launched the
    # real six-worker API on :8000; it inherited PYTEST_CURRENT_TEST, TMPDIR and
    # ORCHESTRATOR_TMP_DIR, then served production traffic against pytest lock
    # files for 21 hours.  Tests that intentionally exercise this function may
    # proceed only after replacing Popen with a fake. Comparing against the
    # captured real callable makes the guard structural; there is no ambient
    # allow flag a leaking test process could accidentally inherit.
    if os.environ.get("PYTEST_CURRENT_TEST") and subprocess.Popen is _ORIGINAL_SUBPROCESS_POPEN:
        raise RuntimeError(
            "refusing to start the orchestrator API from pytest; mock the lifecycle "
            "boundary with a fake Popen"
        )
    # SS-BENCH-GATE-b: the API is the incident's sidecar shape — no explicit
    # placement, so default affinity covers every core. Under a live bench,
    # pin it to a non-overlapping subset of host cores instead of spawning
    # onto the bench's claim (refuse only when no such subset exists).
    spawn_prefix = _bench_guarded_numa_prefix(
        None, 0, bench_force=bench_force, label="orchestrator API (uvicorn)"
    )
    log_file = LOG_DIR / "orchestrator.log"

    print("  Starting orchestrator API on port 8000")
    stale_pids = _pids_on_port(8000)
    if stale_pids:
        print(f"    Clearing stale listeners on :8000 ({', '.join(str(p) for p in stale_pids)})")
        for stale_pid in stale_pids:
            kill_process(stale_pid)
        time.sleep(1)

    # Set environment — enable production feature flags
    env = os.environ.copy()
    env["HF_HOME"] = str(_PATHS["cache_dir"] / "huggingface")
    env["TMPDIR"] = str(_PATHS["tmp_dir"])
    # Resolve the intended mode (arg > runtime-facts manifest > shell env),
    # tracking provenance for the alignment log below.
    manifest_mode = read_runtime_stack_numa_mode()
    env_mode = env.get("ORCHESTRATOR_STACK_NUMA_MODE")
    runtime_numa_mode: str | None = None
    runtime_numa_source = "unset"
    if stack_numa_mode:
        runtime_numa_mode, runtime_numa_source = stack_numa_mode, "arg"
    elif manifest_mode:
        runtime_numa_mode, runtime_numa_source = manifest_mode, "runtime-facts-manifest"
    elif env_mode:
        runtime_numa_mode, runtime_numa_source = env_mode, "shell-env"
    resolved_mode = normalize_stack_numa_mode(runtime_numa_mode) if runtime_numa_mode else None
    # ESC-8 Fix 3: verify the resolved mode against the REALIZED fleet (bare TCP
    # connect on NUMA_CONFIG full vs quarter ports; no HTTP to llama-servers).
    # The API must never be launched with a mode the live fleet contradicts —
    # e.g. a poisoned manifest/env says "full" while only quarters are listening.
    realized_mode: str | None = None
    try:
        from scripts.server.realized_fleet import derive_realized_numa_mode

        realized_mode = derive_realized_numa_mode()
    except Exception as exc:  # noqa: BLE001
        print(f"    [numa-align] WARN: realized-fleet probe failed ({exc})")
    if realized_mode is not None and realized_mode != resolved_mode:
        print(
            f"    [numa-align] realized fleet is '{realized_mode}' but resolved mode was "
            f"{resolved_mode!r} (source: {runtime_numa_source}); "
            f"correcting to '{realized_mode}'."
        )
        resolved_mode = realized_mode
    if resolved_mode:
        env["ORCHESTRATOR_STACK_NUMA_MODE"] = resolved_mode
    # Feature flags: make every registry flag explicit in /proc/<pid>/environ.
    # Explicit launch-time env values are activation intent and must survive.
    _apply_production_feature_env(env)
    # 2026-05-22 Phase 5: per-CPU-region cross-process locks enabled by
    # default. Replaces the single global heavy_model.lock with
    # per-(role, atomic-region) fcntl locks so frontdoor full (0-47)
    # can run concurrently with frontdoor.q2 / q3 (48-71 / 72-95) on
    # disjoint cores. ConcurrencyAwareBackend's dispatch path uses these
    # for cross-process safe instance selection. Override with
    # ORCHESTRATOR_PER_REGION_LOCKS=0 to fall back to the legacy global
    # heavy lock if a regression surfaces.
    env.setdefault("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    # Q-TD-write (operator-granted 2026-07-23): routing observations TD-update
    # the (objective, action) row in place instead of blind-appending (the
    # 99.7% append-only defect, dar_write_path_audit.py). Pair with the
    # consolidate_q_append_only.py migration.
    env.setdefault("ORCHESTRATOR_Q_TD_WRITE", "1")
    # 2026-05-31: default-on cross-role physical exclusion. Adds a
    # role-agnostic cpu_region.GLOBAL.{qN}.lock layer so different roles cannot
    # decode on the same atomic CPU region.
    #
    # 2026-07-06: enable shape-aware contention by default now that
    # ConcurrencyAwareBackend dispatch threads the actual candidate_topology_idx
    # through the admission gate. Physical disjointness alone is insufficient:
    # the measured contention matrix still blocks some role pairs whose CPU
    # regions do not literally overlap.
    env.setdefault("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    env.setdefault("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", "1")
    # WP-7/J6 (2026-05-26): within-role placement rollout. These were previously
    # shell-env-only, so ANY API restart (autopilot config-apply, watcher relaunch,
    # manual) silently reverted the placement state machine to OFF — J6 ran without it
    # after a 19:11 restart until this was caught. Make them durable defaults (still
    # overridable via the env). PLACEMENT_STATE_MACHINE = WP-2 topology-safe placement;
    # REVERSE_MIGRATION = WP-4 quarter→full on load drop; URE shadow = J10 (no behavior change).
    env.setdefault("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    env.setdefault("ORCHESTRATOR_REVERSE_MIGRATION", "1")
    env.setdefault("ORCHESTRATOR_URE_UNCERTAINTY_SHADOW_LOG", "1")
    # WP-12 flip boundary (operator-directed 2026-07-23): fleet layer ON —
    # roles bound to a registry server_mode fleet share ONE backend per
    # physical fleet (one breaker/lock fact per endpoint, same-fleet fallback
    # compiled out). Durable default so API restarts don't silently revert
    # (the WP-7/J6 lesson above). INSTANT ROLLBACK: relaunch/reload with
    # ORCHESTRATOR_FLEET_LAYER=0 in the shell env (setdefault yields) — the
    # legacy per-role build (Fix-A delegations + priors alias records) is
    # retained as the rollback substrate until the post-soak §5 cleanup.
    env.setdefault("ORCHESTRATOR_FLEET_LAYER", "1")
    # P6.2-A2 (2026-05-21): frontdoor-specialist verifier loaded by the API
    # service when the gate flag is on. Defaults below put it in SHADOW MODE —
    # the verifier runs and logs P(success) to last_decision_meta but never
    # blocks a fast-path route. To enforce: launch with
    #   FRONTDOOR_VERIFIER_SHADOW=0 ./scripts/server/orchestrator_stack.py start
    # Both can be disabled by setting ORCHESTRATOR_FRONTDOOR_VERIFIER_GATE=0.
    # See handoffs/active/learned-routing-controller.md Phase 6 rollout.
    env.setdefault("ORCHESTRATOR_FRONTDOOR_VERIFIER_GATE", "1")
    env.setdefault("FRONTDOOR_VERIFIER_SHADOW", "1")
    env.setdefault("FRONTDOOR_VERIFIER_THRESHOLD", "0.5")
    env["ORCHESTRATOR_TOOLS"] = "1"
    env["ORCHESTRATOR_SCRIPTS"] = "1"
    # NOTE: Do NOT set ORCHESTRATOR_REPL here — it collides with
    # OrchestratorSettings.repl (REPLSettings model) in config.py.
    # The repl feature flag defaults to True in features.py already.
    env["ORCHESTRATOR_CACHING"] = "1"
    env["ORCHESTRATOR_STREAMING"] = "1"
    env["ORCHESTRATOR_MOCK_MODE"] = "0"
    env["ORCHESTRATOR_GENERATION_MONITOR"] = "1"
    env["ORCHESTRATOR_REACT_MODE"] = "1"
    env["ORCHESTRATOR_CASCADING_TOOL_POLICY"] = "1"
    env["ORCHESTRATOR_WORKER_CALL_BUDGET"] = "1"
    env["ORCHESTRATOR_TASK_TOKEN_BUDGET"] = "1"
    env.setdefault("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", "30")
    env.setdefault("ORCHESTRATOR_TASK_TOKEN_BUDGET_CAP", "200000")
    env["ORCHESTRATOR_SESSION_SCRATCHPAD"] = "1"
    env["ORCHESTRATOR_SESSION_LOG"] = "1"
    env["ORCHESTRATOR_APPROVAL_GATES"] = "1"
    env["ORCHESTRATOR_RESUME_TOKENS"] = "1"
    env["ORCHESTRATOR_SIDE_EFFECT_TRACKING"] = "1"
    env["ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT"] = "1"
    # Gate-3 tool telemetry must survive API reloads too; the authority daemon
    # already carries AUTOPILOT_TOOL_SENTINELS=1, but the orchestrator API was
    # previously restarted without it and quietly lost tool-use activation.
    env["AUTOPILOT_TOOL_SENTINELS"] = "1"
    # LangGraph Phase 3: per-node migration for live roles only.
    # The retired architect_coding role is intentionally not enabled here.
    for key in LANGGRAPH_PHASE3_LIVE_ENV_VARS:
        env[key] = "1"
    _apply_orchestrator_profile(env, profile)
    # Bound inference-lock waits by default to avoid multi-minute silent stalls
    # during iterative debugging / seeding runs. Bumped from 45s → 180s 2026-05-21
    # because GEPA-driven autopilot workloads were aborting worker_explore /
    # ingest requests after 45s while frontdoor held the exclusive lock for
    # 60-105s during reasoning-heavy spec-decode bursts. 180s rides out typical
    # spec-decode + multi-token-generation while still surfacing true deadlocks
    # within ~3 min. Tune via env override if your workload changes.
    env.setdefault("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_EXCLUSIVE_S", "180")
    env.setdefault("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_SHARED_S", "180")

    # Write the fleet-startup marker BEFORE Popen. uvicorn forks workers
    # AFTER its own startup (each worker imports the app independently),
    # and dashboard.py reads this marker at module-import time to serve
    # the same `server_started_at` across all six workers via
    # /dashboard/api/version. Without this marker, each worker would
    # carry its own `time.time()` and consumers (OrchestratorWatcher)
    # would see spurious restart signals depending on which worker the
    # load balancer routed each request to.
    try:
        marker_path = _write_orchestrator_marker(
            tmp_dir=_PATHS["tmp_dir"],
            git_sha=_repo_short_sha(),
        )
        print(f"    Fleet marker: {marker_path}")
    except Exception as exc:
        print(f"    [WARN] Failed to write orchestrator fleet marker: {exc}")

    with open(log_file, "a") as log:
        workers = int(env.get("ORCHESTRATOR_UVICORN_WORKERS", "6"))
        # 2026-06-13: bumped default from 16 to 64 (env-overridable). The
        # lower cap still allowed dashboard/SSE traffic plus AutoPilot reward
        # writes to saturate uvicorn before model-side locks could serialize
        # real inference, yielding HTTP 503 on /chat/reward during seeding.
        # This is an accept-queue headroom limit, not a model concurrency
        # limit; inference-lock/contention gates still provide serialization.
        concurrency = int(env.get("ORCHESTRATOR_UVICORN_LIMIT_CONCURRENCY", "64"))
        proc = subprocess.Popen(
            spawn_prefix
            + [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--workers",
                str(workers),
                "--limit-concurrency",
                str(concurrency),
            ],
            cwd=str(_PATHS["project_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print("    Waiting for health...")

    if wait_for_health(8000, timeout=60):
        print("    [OK] Orchestrator ready")
        # Durable earlyoom control-plane protection. The API master + its uvicorn
        # workers are comm=python and cannot be earlyoom --ignore'd by name (they
        # collide with runaway python evals), so set oom_score_adj=-1000 (earlyoom
        # skips exactly -1000 in both oom_score and --sort-by-rss modes). Workers
        # exist by health-check time. Best-effort; replaces the manual one-shot
        # `choom` that did not survive an API restart.
        _set_oom_protection([proc.pid, *_child_pids(proc.pid)])
        return ProcessInfo(
            role="orchestrator",
            pid=proc.pid,
            port=8000,
            started_at=datetime.now().isoformat(),
            model_path="uvicorn",
            log_file=str(log_file),
        )
    # Health probe can fail transiently (port permissions / local sandbox),
    # while the process is actually alive. Avoid killing a healthy API due to
    # a false-negative probe; only hard-fail when process already exited.
    if proc.poll() is None:
        print("    [WARN] Health probe timed out, but API process is still running")
        print(f"    Check log: {log_file}")
        _set_oom_protection([proc.pid, *_child_pids(proc.pid)])
        return ProcessInfo(
            role="orchestrator",
            pid=proc.pid,
            port=8000,
            started_at=datetime.now().isoformat(),
            model_path="uvicorn",
            log_file=str(log_file),
        )

    print("    [FAIL] Orchestrator did not start")
    print(f"    Check log: {log_file}")
    kill_process(proc.pid)
    return None


# =============================================================================
# Aux services — ONE launcher, driven by the declared AUX_SERVICES registry
# =============================================================================
#
# Replaces four near-identical `start_<name>()` functions (2026-08-02, W5/W4).
# They differed only in argv, cwd, env, health path and labels — all data — yet
# each one restated the Popen boilerplate, and `cmd_reload`'s dispatch chain had
# to be edited separately to know a service existed. Two of the four never were,
# which is exactly how `reload whisper` and `reload sd_server` came to kill a
# service without restarting it. The registry now feeds start, reload and status
# from one place, so a service cannot be startable-but-not-reloadable again.

_VERIFY_GGML_LINKAGE_SCRIPT = Path(
    "/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
)


def _resolve_aux_launch(service) -> tuple[list[str], list[str], Path | None]:
    """Resolve a service's argv, LD_LIBRARY_PATH entries and expected ggml tree.

    Backend resolution happens HERE — at launch — deliberately. Resolving it at
    import would make a single dangling kernel symlink fail the orchestrator's
    import rather than the one service that depends on it, and `kernel_paths`
    raises rather than falling back precisely so the failure is not silent.
    """
    argv = [token.replace("{python}", sys.executable) for token in service.argv]
    ld_paths = list(service.ld_library_path)
    tree: Path | None = None
    if service.backend:
        from src.registry.kernel_paths import (
            backend_dir,
            backend_ld_library_path,
            server_binary,
        )

        argv[0] = str(server_binary(service.backend))
        ld_paths = backend_ld_library_path(service.backend)
        tree = backend_dir(service.backend)
    return argv, ld_paths, tree


def _build_aux_env(service, ld_paths: list[str]) -> dict[str, str]:
    """Compose the launch environment for one aux service."""
    spec = service._replace(ld_library_path=tuple(ld_paths))
    env = build_service_env(spec, os.environ.copy())
    if service.pythonpath:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(service.pythonpath) + (
            f"{os.pathsep}{existing}" if existing else ""
        )
    return env


def _verify_aux_ggml_linkage(binary: str, tree: Path, env: dict[str, str]) -> bool:
    """Prove the binary resolves its ggml inside `tree` under the LAUNCH env.

    The check is run with `env` — the environment the service is about to be
    launched with — not the caller's. Verifying under any other environment
    proves nothing: LD_LIBRARY_PATH is the variable under test, so a check that
    does not carry it is a check of a different process.

    Failure is fatal to the launch, not a warning. A wrong-tree resolution does
    not crash and does not degrade visibly — on 2026-07-31 a HIP whisper-cli
    loaded the production CPU-only ggml, printed `use gpu = 1`, and produced
    well-formed transcripts at CPU speed. Serving those answers is worse than
    not serving.
    """
    if not _VERIFY_GGML_LINKAGE_SCRIPT.exists():
        print(f"    [FAIL] linkage verifier missing: {_VERIFY_GGML_LINKAGE_SCRIPT}")
        return False
    try:
        result = subprocess.run(
            ["bash", str(_VERIFY_GGML_LINKAGE_SCRIPT), binary, str(tree)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"    [FAIL] linkage verifier did not run: {exc}")
        return False
    for line in result.stdout.splitlines():
        if line.strip().startswith(("OK ", "BAD ", "FAIL", "PASS")):
            print(f"      {line.strip()}")
    if result.returncode != 0:
        print(f"    [FAIL] ggml linkage check failed for {binary}")
        print(f"           expected every ggml lib under {tree}")
        return False
    return True


def _run_aux_smoke(service) -> str | None:
    """Prove an aux service can actually WORK. Returns an error string, or None.

    Declared per service as `smoke: {path, method, json, min_bytes, timeout}`.
    A service with no `smoke` block is not checked, so this is additive.

    `min_bytes` is the load-bearing part: the failure this exists to catch is a
    200 response with an empty body, which every status-code check passes.
    """
    smoke = getattr(service, "smoke", None)
    if not isinstance(smoke, dict) or not smoke.get("path"):
        return None

    url = f"http://127.0.0.1:{service.port}{smoke['path']}"
    min_bytes = int(smoke.get("min_bytes", 1))
    timeout = int(smoke.get("timeout", 120))
    payload = smoke.get("json")
    print(f"    smoke: {smoke.get('method', 'POST')} {smoke['path']} (expect >= {min_bytes} bytes)")
    import urllib.error
    import urllib.request

    # Build the request OUTSIDE the try. A NameError or a malformed declaration
    # is a bug in this file, not a sick service, and must not be reported as
    # "smoke request failed" — that misattribution would send someone debugging
    # a healthy server. Only the network call is caught below.
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=str(smoke.get("method", "POST" if data else "GET")),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            code = resp.status
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return f"smoke request failed: {type(exc).__name__}: {exc}"

    if code != 200:
        return f"smoke returned HTTP {code}"
    if len(body) < min_bytes:
        return (
            f"smoke returned HTTP 200 but only {len(body)} bytes "
            f"(need >= {min_bytes}) — the service is up but not producing output"
        )
    print(f"    smoke: OK ({len(body)} bytes)")
    return None


_AUX_CPUSET_RE = re.compile(r"\d+(-\d+)?(,\d+(-\d+)?)*")


def _aux_spawn_prefix(service: Any, *, bench_force: bool) -> list[str] | None:
    """Spawn prefix for one aux service; None means "refuse this launch".

    Undeclared placement (every service but a CPU-pinned one): SS-BENCH-GATE-b
    as before -- default affinity, or a non-overlapping subset under a live bench.

    Declared `cpuset`: that exact list, or nothing. The bench guard may REFUSE it
    (overlap with a live bench's claim; `--allow-during-bench` bypasses), but it
    may never RE-PIN it: the declared list is a measured-safe layout, and
    whisper.cpp hangs on unmeasured ones (speech_cpu_realtime_20260924).
    """
    label = f"aux service {service.name}"
    cpuset = getattr(service, "cpuset", None)
    if not cpuset:
        return _bench_guarded_numa_prefix(None, 0, bench_force=bench_force, label=label)
    if not _AUX_CPUSET_RE.fullmatch(cpuset):
        print(f"    [FAIL] {service.name}: declared cpuset {cpuset!r} is not a cpu list")
        return None
    pinned = enforce_placement(cpuset, force=bench_force, label=label)
    if pinned is not None and pinned != cpuset:
        print(
            f"    [FAIL] {service.name}: bench guard wanted to re-pin the declared "
            f"cpuset {cpuset} to {pinned}; refusing (declared layouts are measured-safe)"
        )
        return None
    return ["taskset", "-c", cpuset]


def _embedding_spawn_prefix(
    port: int, placement: Any, *, bench_force: bool
) -> list[str] | None:
    """Spawn prefix for an embedder with a declared placement; None = refuse.

    `numactl --membind=<node> -- taskset -c <cpuset>`: the node is the single
    NPS4 node stack_manifest derived from the cpuset. The bench guard may
    REFUSE the declared cpuset (overlap with a live bench's claim;
    `--allow-during-bench` bypasses) but may never RE-PIN it: a re-pinned
    embedder would land back on cores other instances or the fleet own, which
    is the defect the declaration exists to remove.
    """
    label = f"embedder :{port}"
    pinned = enforce_placement(placement.cpuset, force=bench_force, label=label)
    if pinned is not None and pinned != placement.cpuset:
        print(
            f"    [FAIL] {label}: bench guard wanted to re-pin the declared cpuset "
            f"{placement.cpuset} to {pinned}; refusing (declared placements are not re-pinned)"
        )
        return None
    return [
        "numactl",
        f"--membind={placement.numa_node}",
        "--",
        "taskset",
        "-c",
        placement.cpuset,
    ]


def start_aux_service(name: str, bench_force: bool = False) -> ProcessInfo | None:
    """Start one declared auxiliary service. Returns None on any failure."""
    service = AUX_SERVICES.get(name)
    if service is None:
        print(f"    [FAIL] unknown aux service {name!r}")
        return None

    log_file = LOG_DIR / service.log
    label = service.description or service.name
    print(f"  Starting {service.name} ({label}) on port {service.port}")

    spawn_prefix = _aux_spawn_prefix(service, bench_force=bench_force)
    if spawn_prefix is None:
        return None

    try:
        argv, ld_paths, tree = _resolve_aux_launch(service)
    except Exception as exc:  # KernelPathError and anything else path-shaped
        print(f"    [FAIL] {service.name}: {exc}")
        return None

    executable = Path(argv[0])
    if executable.is_absolute() and not executable.exists():
        print(f"    [FAIL] launcher not found: {executable}")
        return None
    if not Path(service.cwd).is_dir():
        print(f"    [FAIL] working directory not found: {service.cwd}")
        return None

    env = _build_aux_env(service, ld_paths)
    # A missing LD_PRELOAD object is NOT fatal to the dynamic loader: glibc prints a
    # warning and runs the binary without it. For the CPU TTS that silently means
    # 96 spinning threads on a 16-core cpuset instead of 16 (see nprocs_shim.c), so
    # a declared preload that does not exist fails THIS service's launch.
    for preload in (service.env.get("LD_PRELOAD") or "").replace(":", " ").split():
        if not Path(preload).is_file():
            print(
                f"    [FAIL] {service.name}: declared LD_PRELOAD {preload} does not exist "
                f"(build it: scripts/voice/build_nprocs_shim.sh)"
            )
            return None
    if ld_paths:
        verb = ":=" if service.ld_library_path_mode == "replace" else "+="
        print(f"    LD_LIBRARY_PATH {verb} {ld_paths}")
        if service.ld_library_path_mode == "replace":
            print("      (ambient LD_LIBRARY_PATH dropped — foreign ggml trees unreachable)")

    if service.verify_ggml_linkage:
        if tree is None:
            print(f"    [FAIL] {service.name} requests a linkage check but declares no backend")
            return None
        if not _verify_aux_ggml_linkage(argv[0], tree, env):
            return None

    with open(log_file, "a") as log:
        proc = subprocess.Popen(
            spawn_prefix + argv,
            cwd=service.cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print(
        f"    Waiting for health (path={service.health_path}, "
        f"timeout={service.health_timeout}s)..."
    )

    if wait_for_health(service.port, timeout=service.health_timeout, path=service.health_path):
        # `/health` proves the PORT answers, not that the service can do its job.
        # The TTS server returns HTTP 200 with ZERO audio bytes for a request it
        # cannot satisfy (e.g. an unknown `voice`), and whisper will answer
        # /health while a model failed to load. A service that reports ready and
        # then silently produces nothing is worse than one that fails to start,
        # because the failure surfaces at first real use, far from the cause.
        #
        # `smoke:` in launch_manifest.yaml declares an OPTIONAL request whose
        # response must exceed `min_bytes`. Absent -> no smoke check, which is
        # the previous behaviour for every service that does not declare one.
        smoke_error = _run_aux_smoke(service)
        if smoke_error:
            print(f"    [FAIL] {service.name} answered /health but failed its smoke check")
            print(f"           {smoke_error}")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None
        print(f"    [OK] {service.name} ready")
        return ProcessInfo(
            role=service.name,
            pid=proc.pid,
            port=service.port,
            started_at=datetime.now().isoformat(),
            model_path=service.model_label,
            log_file=str(log_file),
        )

    print(f"    [FAIL] {service.name} did not start")
    print(f"    Check log: {log_file}")
    kill_process(proc.pid)
    return None


# Named wrappers kept so existing call sites and their tests keep working
# unchanged. They carry no logic — the declaration in launch_manifest.yaml is
# the whole definition of each service.
def start_document_formalizer(bench_force: bool = False) -> ProcessInfo | None:
    """Start the document formalizer (LightOnOCR-2) server."""
    return start_aux_service("document_formalizer", bench_force=bench_force)


def start_sd_server(bench_force: bool = False) -> ProcessInfo | None:
    """Start the stack-managed image service (legacy role name: sd_server).

    The manifest selects Qwen-Image-2.1's isolated CPU Diffusers runner while
    preserving the existing role name and sdapi-compatible client contract.
    """
    return start_aux_service("sd_server", bench_force=bench_force)


def start_whisper(bench_force: bool = False) -> ProcessInfo | None:
    """Start the STT server on :9000.

    2026-08-02 (W4): this is whisper.cpp @ production-speech-v1 on the MI210, NOT
    the faster-whisper CTranslate2 service it replaced. That service hardcoded
    device="cpu" and could not have been otherwise — CTranslate2 4.7.2 ships no
    ROCm backend. See the `whisper` entry in launch_manifest.yaml for the API
    delta this swap carries.
    """
    return start_aux_service("whisper", bench_force=bench_force)


def start_tts(bench_force: bool = False) -> ProcessInfo | None:
    """Start the qwentts.cpp TTS server on :9002 (first registered 2026-08-02)."""
    return start_aux_service("tts", bench_force=bench_force)


def start_handoff_dashboard(bench_force: bool = False) -> ProcessInfo | None:
    """Start the epyc-root handoff progress dashboard hub (port 8100).

    Project-wide, file/artifact-backed progress board owned by the governance
    repo. Deliberately dependency-free (stdlib only), so it runs under any
    interpreter — the orchestrator venv is not required.
    """
    return start_aux_service("handoff_dashboard", bench_force=bench_force)


# =============================================================================
# Commands
# =============================================================================


# =============================================================================
# CLI commands (cmd_start, cmd_stop, cmd_reload, cmd_status) moved to
# scripts/server/stack_commands.py (2026-05-22 refactor). main() below imports
# them lazily to avoid a module-import cycle with stack_commands, which itself
# pulls helpers from this module.
# Module-level __getattr__ exposes them as orchestrator_stack.cmd_* for
# backward compatibility (e.g. tests that call stack.cmd_reload(...) after
# `from scripts.server import orchestrator_stack as stack`).
# =============================================================================

_STACK_MANIFEST_EXPORTS = {
    "PORT_MAP",
    "ROLE_LAUNCH_META",
    "HOT_ROLES",
    # 2026-08-02: `-np` no longer derives from SERIAL_ROLES, so this module no
    # longer imports it directly. It stays re-exported because SERIAL_ROLES is
    # still the admission policy and callers/tests read it off this module.
    "SERIAL_ROLES",
    "DECLARED_SLOTS",
    "FALLBACK_SLOTS",
    "validate_declaration_parity",
    "NUMA_REPLICA_PORTS",
    "HOT_SERVERS",
    "WARM_SERVERS",
    "DOCKER_SERVICES",
    "validate_model_paths",
    "validate_against_registry",
    "_build_servers_from_classification",
    "_validate_role_classification",
    "_filter_by_numa_mode",
}

_STACK_PATH_EXPORTS = {
    "_get_paths",
    "LLAMA_MATH_TOOLS",
}


def __getattr__(name: str):
    if name in ("cmd_start", "cmd_stop", "cmd_reload", "cmd_status"):
        from scripts.server import stack_commands

        return getattr(stack_commands, name)
    if name in _STACK_MANIFEST_EXPORTS:
        from scripts.server import stack_manifest

        return getattr(stack_manifest, name)
    if name in _STACK_PATH_EXPORTS:
        from scripts.server import stack_paths

        return getattr(stack_paths, name)
    raise AttributeError(f"module 'scripts.server.orchestrator_stack' has no attribute {name!r}")



# --------------------------------------------------------------------------- #
# SS-BENCH-GATE: refuse a lifecycle action while a pinned CPU bench is running.
# --------------------------------------------------------------------------- #
#
# 2026-07-27 incident: an operator-authorized `reload orchestrator` killed a
# Laguna Q4 CPU bench 1h09m into a decision-gating run. The reload spawned an
# accelerated sidecar on cores the bench had pinned, and the bench's own
# campaign-continuity gate invalidated the run:
#
#   campaign continuity gate failed: production stack continuity invalid:
#   accelerated sidecar 1202069 overlaps CPU bench cores: [0, 1, 2, ...]
#
# The precondition that WAS checked ("autopilot is down") is not the relevant
# one — the gate keys on CORE OVERLAP with a pinned bench, an entirely separate
# condition. This guard makes that check impossible to forget.
#
# The bench-driver identity list and `detect_running_cpu_bench` live in
# scripts/server/bench_core_claim.py (single source of truth): the placement
# guard (SS-BENCH-GATE-b) must detect the exact same drivers this guard does,
# and a second copy of the name list would drift.


def guard_against_running_bench(command: str, force: bool) -> bool:
    """False when the caller should abort. Prints the reason."""
    running = detect_running_cpu_bench()
    if not running:
        return True
    print(f"REFUSING to `{command}`: a CPU benchmark is running.")
    for pid, cmd in running:
        print(f"    PID {pid}: {cmd}")
    print(
        "  A lifecycle action spawns stack processes that can overlap the bench's\n"
        "  pinned cores, and the bench's campaign-continuity gate will invalidate\n"
        "  the run (this destroyed 1h09m of decision-gating measurement on\n"
        "  2026-07-27). Wait for the bench, or pass --allow-during-bench if the\n"
        "  operator has accepted that the run may be invalidated."
    )
    return bool(force)


def _placement_from_prefix(prefix: list[str]) -> str | None:
    """The cpu list a spawn prefix pins to, or None when it does not pin.

    `_numa_prefix` emits `["taskset", "-c", <cpu_list>]` (optionally preceded
    by a `numactl --<policy> --` pair); a prefix without `taskset` leaves the
    spawn on its default affinity, which is the incident's sidecar shape.
    """
    if "taskset" not in prefix:
        return None
    idx = prefix.index("taskset")
    if len(prefix) > idx + 2 and prefix[idx + 1] == "-c":
        return prefix[idx + 2]
    raise BenchPlacementRefusal(
        f"taskset prefix without a `-c <cpu_list>` argument: {prefix!r}"
    )


def _bench_guarded_numa_prefix(
    role: str | None,
    numa_instance: int,
    *,
    bench_force: bool,
    label: str | None = None,
) -> list[str]:
    """`_numa_prefix` plus SS-BENCH-GATE-b placement guarding.

    Reads the bench's REAL core claim from the live processes and:
      * refuses the spawn when the requested placement overlaps it
        (BenchPlacementRefusal -> exit 2; `bench_force` mirrors
        --allow-during-bench and bypasses the refusal), and
      * pins default-affinity spawns to a non-overlapping subset of host cores
        instead of refusing — the stack stays functional during a bench.
    No bench live -> the plain `_numa_prefix` result, unchanged.
    """
    prefix = _numa_prefix(role, numa_instance)
    placement = _placement_from_prefix(prefix)
    pinned = enforce_placement(
        placement,
        force=bench_force,
        label=label or (f"server for {role}" if role else "stack process"),
    )
    if pinned is None:
        return prefix
    return ["taskset", "-c", pinned]



def _cmd_validate_only(args) -> int:
    """Validate a stack template and exit WITHOUT launching anything.

    Implements the long-declared `start --validate-only`. Deliberately narrow:
    it loads and validates, prints, and returns. It must never acquire a lock,
    write runtime facts, or start a process — the whole point of the flag is
    that a caller can run it on a busy host and be certain nothing happens.

    Exit codes: 0 valid (warnings allowed), 1 invalid or unloadable.
    """
    from src.config.stack_templates import get_active_profile, load_template, validate_template

    profile = getattr(args, "stack_profile", None) or get_active_profile()
    try:
        template = load_template(profile)
    except Exception as exc:  # noqa: BLE001 — surface the reason, do not launch
        print(f"validate-only: FAILED to load stack template {profile!r}: {exc}")
        return 1

    result = validate_template(template)
    print(f"validate-only: stack template {profile!r} — {result.summary}")
    for err in result.errors:
        print(f"  ERROR   {err}")
    for warn in result.warnings:
        print(f"  warning {warn}")
    print("validate-only: nothing was launched.")
    return 0 if result.valid else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator stack manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the stack")
    start_parser.add_argument("--hot-only", action="store_true", help="Start HOT models only")
    start_parser.add_argument(
        "--include-warm", nargs="+", metavar="ROLE", help="Include WARM models"
    )
    start_parser.add_argument(
        "--only",
        nargs="+",
        metavar="ROLE",
        help="Start ONLY these roles (skip everything else). "
        "Searches both HOT and WARM server lists.",
    )
    start_parser.add_argument("--dev", action="store_true", help="Dev mode (single 0.5B model)")
    start_parser.add_argument(
        "--numa-mode",
        choices=["full", "quarter", "both"],
        default=None,
        help=(
            "For roles with both a full-NUMA-node instance and sub-full siblings "
            "(currently frontdoor + worker_general + ingest_long_context — see "
            "NUMA_CONFIG[role]['full_instance_idx']), pick one mode. "
            "When OMITTED the launcher INFERS the mode from the running fleet (ESC-8); "
            "with no live fleet it falls back to ORCHESTRATOR_STACK_NUMA_MODE, then to "
            "the ratified production mode 'both'. Production is 'both' — full + half "
            "instances (quarters were retired 2026-07-30), so an unflagged cold start "
            "defaults to 'both'. "
            "'both' = full + 2x48t half siblings (ratified production mode). "
            "'quarter' = the sub-full shape only; since the 2026-07-30 half-fleet "
            "cutover this token means HALVES (2x48t), not quarters (4x24t). "
            "'full' = single full instance (max single-stream tps; only when a full "
            "fleet is deliberately brought up). Single-instance roles "
            "(architect_general, embedders) are unaffected by this flag."
        ),
    )
    start_parser.add_argument(
        "--skip-host-prereqs",
        action="store_true",
        help="Skip host_prereq audit/apply (numa_balancing, THP, governor). NOT recommended for benchmarks.",
    )
    start_parser.add_argument(
        "--skip-stack-change-gate",
        action="store_true",
        help=(
            "Skip the canonical stack-change promotion gate before production launch. "
            "Use only for emergency diagnostics; benchmarks and AutoPilot resumes should not bypass it."
        ),
    )
    start_parser.add_argument(
        "--skip-page-cache-prewarm",
        action="store_true",
        help="Skip the [1.5] numactl --interleave=all GGUF prewarm step. "
        "NOT recommended after a cold cache / container rebuild — sequential "
        "mlock will pin all shared-GGUF pages to one NUMA node and quarters "
        "will fetch cross-socket. See handoffs/active/numa-page-cache-prewarm.md. "
        "Equivalent: set ORCHESTRATOR_SKIP_PAGE_CACHE_PREWARM=1.",
    )
    start_parser.add_argument(
        "--repair-embeddings",
        action="store_true",
        help="If [0.7] embedding health check finds orphans, run repair before launch "
        "(re-embeds via the configured parallel BGE servers, rebuilds FAISS index, ~5-15 min). "
        "Default behavior is read-only — just print warning and continue. "
        "See scripts/maintenance/repair_episodic_embeddings.py for the manual workflow.",
    )
    start_parser.add_argument(
        "--profile",
        choices=sorted(ORCHESTRATOR_PROFILES.keys()),
        help="Optional orchestrator API env profile",
    )
    start_parser.add_argument(
        "--stack-profile",
        metavar="NAME",
        help="Load stack template from stack_templates/<NAME>.yaml (DS-7). "
        "Use --validate-only to check without launching.",
    )
    start_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate stack template and exit (use with --stack-profile)",
    )
    start_parser.add_argument(
        "--migrate-to",
        metavar="NAME",
        help="Migrate running stack to stack_templates/<NAME>.yaml via full "
        "restart (DS-7 / NIB2-19). Use with --dry-run to plan only.",
    )
    start_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --migrate-to, plan the migration without stopping any servers.",
    )
    start_parser.add_argument(
        "--compile-registry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompile orchestration/model_registry.yaml from the master "
        "registry at epyc-inference-research before starting (default: on). "
        "Cache-aware: no-op if neither master nor active role set has changed. "
        "Use --no-compile-registry or set ORCHESTRATOR_REGISTRY_NO_COMPILE=1 "
        "to disable.",
    )
    start_parser.add_argument(
        "--compile-descriptors",
        action="store_true",
        help="Compile orchestration/model_descriptors.yaml from the active "
        "lean/research registries before starting. Strict by default: refuses "
        "missing load-bearing descriptor fields.",
    )
    start_parser.add_argument(
        "--allow-incomplete-descriptors",
        action="store_true",
        help="With --compile-descriptors, emit descriptor records with known_gaps "
        "instead of failing on incomplete evidence. Intended for diagnostics only.",
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop components")
    stop_parser.add_argument("--all", action="store_true", help="Stop all components")
    stop_parser.add_argument("components", nargs="*", help="Components to stop")

    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload components")
    reload_parser.add_argument("components", nargs="+", help="Components to reload")
    reload_parser.add_argument(
        "--profile",
        choices=sorted(ORCHESTRATOR_PROFILES.keys()),
        help="Optional orchestrator API env profile (used when reloading orchestrator)",
    )
    reload_parser.add_argument(
        "--embedder-env-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "EXPERIMENT ONLY (UFH-12 A3): relaunch the embedders with this OpenMP env over the "
            "declared one. Only with the single component `embedders`, only OMP_WAIT_POLICY / "
            "KMP_BLOCKTIME / KMP_LIBRARY, and only with --experiment-id. Recorded in "
            "logs/embedder_env_override.json; a plain `reload embedders` restores and clears it."
        ),
    )
    reload_parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment id recorded with --embedder-env-override (required with it)",
    )
    reload_parser.add_argument(
        "--override-ttl-s",
        type=float,
        default=5400.0,
        help="Expiry of the recorded override; past it env_attestation reports the deviation "
        "as an ERROR again (default 5400 s, max 14400 s)",
    )

    # Status command
    for _lifecycle_parser in (start_parser, stop_parser, reload_parser):
        _lifecycle_parser.add_argument(
            "--allow-during-bench",
            action="store_true",
            help=(
                "proceed even if a CPU benchmark is running — the bench's continuity "
                "gate may invalidate its run (see SS-BENCH-GATE)"
            ),
        )

    subparsers.add_parser("status", help="Show status")

    args = parser.parse_args()

    # Lazy import — stack_commands imports back into this module's helpers
    # (start_server, init_memrl_and_tools, the thin process/state wrappers).
    # Loading it only inside main() avoids the circular-import problem.
    from scripts.server.stack_commands import (
        cmd_start,
        cmd_stop,
        cmd_reload,
        cmd_status,
    )

    # --validate-only was DECLARED and never READ (found by `mainA`, 2026-08-12).
    # argparse accepted it, main() discarded it, and dispatch fell straight
    # through to cmd_start — so anyone who trusted the help text "Validate stack
    # template and exit" / "check without launching" LAUNCHED THE PRODUCTION
    # STACK instead. A dry-run flag that is not wired is worse than no dry-run
    # flag: it manufactures the confidence to run the command.
    #
    # Handled ABOVE `guard_against_running_bench` deliberately (hoisted 2026-08-12
    # after `mainA` caught the first placement). The guard covers start/stop/reload
    # because those MUTATE the host, and it exists because a lifecycle action
    # destroyed 1h09m of decision-gating measurement on 2026-07-27. Validation
    # mutates nothing — it loads a YAML, checks it, prints, returns. Leaving it
    # below the guard meant `start --validate-only` returned 2 whenever a bench was
    # detectable, i.e. pure config validation was unavailable exactly when the host
    # is busy, which is when you most want to check config WITHOUT touching it.
    # Fail-closed, so it was never dangerous — just useless at the moment of need.
    #
    # This branch must stay above the guard AND above dispatch: nothing between it
    # and `cmd_start` may start a server.
    if args.command == "start" and getattr(args, "validate_only", False):
        return _cmd_validate_only(args)

    if args.command in ("start", "stop", "reload"):
        if not guard_against_running_bench(
            args.command, getattr(args, "allow_during_bench", False)
        ):
            return 2

    # SS-BENCH-GATE-b: the per-spawn placement guard refuses overlapping
    # placements with the same exit code as the CLI-level guard above. The
    # incident context was already printed where the refusal was raised; a
    # refusal raised without one (e.g. a malformed taskset prefix) prints here.
    try:
        if args.command == "start":
            return cmd_start(args)
        elif args.command == "stop":
            return cmd_stop(args)
        elif args.command == "reload":
            return cmd_reload(args)
        elif args.command == "status":
            return cmd_status(args)
    except BenchPlacementRefusal as exc:
        if str(exc):
            print(str(exc))
        return 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
