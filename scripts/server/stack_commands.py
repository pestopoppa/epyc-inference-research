"""CLI command handlers for orchestrator_stack — start/stop/reload/status.

Extracted from orchestrator_stack.py during the 2026-05-22 Tranche-7 refactor.
These functions are the public CLI surface invoked by `orchestrator_stack.py main()`;
they live here so the CLI shim stays focused on argument parsing + dispatch.

Imports from orchestrator_stack (and helper modules) at module load. This is safe
because orchestrator_stack.py loads stack_commands lazily inside main() — no
circular import at module-import time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any

# Helper modules (extracted earlier in the refactor)
from scripts.server import stack_checkpoint as _stack_checkpoint
from scripts.server import stack_processes as _stack_processes
from scripts.server.stack_docker import (
    _docker_available,
    docker_container_running,
    start_docker_container,
    stop_docker_container,
)
from scripts.server.stack_health import wait_for_health as _wait_for_health
from scripts.server.stack_host import apply_host_prerequisites
from scripts.server.stack_manifest import (
    AUX_SERVICES,
    DOCKER_SERVICES,
    EMBEDDER_PORTS,
    HOT_SERVERS,
    NUMA_REPLICA_PORTS,
    OPTIONAL_AUXILIARY_ROLES,
    PORT_MAP,
    ROLE_LAUNCH_META,
    WARM_SERVERS,
    _filter_by_numa_mode,
    validate_against_registry,
    validate_model_paths,
)
from scripts.server.stack_paths import (
    _HEALTH_SERVER_STARTUP,
    LLAMA_MATH_TOOLS,
    _PATHS,
    LOG_DIR,
    STATE_FILE,
)
from scripts.server.stack_prewarm import prewarm_all as _prewarm_all
from scripts.server.runtime_facts_manifest import (
    read_runtime_stack_numa_mode,
    realized_stack_numa_mode_from_state,
    write_runtime_facts_manifest,
)
from scripts.server.stack_state import ProcessInfo
from src.roles import Role
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_serving,
    stack_prior_serving_ports,
)
from src.registry_loader import RegistryLoader


STACK_CHANGE_LAUNCH_GATE_COMMAND = (
    "uv",
    "run",
    "python",
    "scripts/registry/stack_change_pipeline.py",
    "check",
    "--run-promotion-gate",
)


def _descriptor_active_roles() -> set[str]:
    """Return canonical roles for descriptor compilation.

    `write_model_descriptors()` expands shared aliases from registry state on its
    own, so the launch helper only needs the canonical launch-role keys here.

    gpu-serving-tie-in P2-6 (P0-1): launcher-only entries stay excluded, but a
    registry TENANT role they name via the optional ``tenant_role`` meta key
    compiles through (empty set today — no entry carries the key).
    """
    from src.registry.registry_compiler import launcher_tenant_roles

    roles = {
        role
        for role, meta in ROLE_LAUNCH_META.items()
        if not (isinstance(meta, dict) and meta.get("launcher_only") is True)
    }
    roles.update(launcher_tenant_roles(ROLE_LAUNCH_META))
    return roles


def wait_for_health(
    port: int, timeout: int = _HEALTH_SERVER_STARTUP, path: str = "/health"
) -> bool:
    """Preserve orchestrator_stack.wait_for_health default-timeout semantics."""
    return _wait_for_health(port, timeout, path)


def _orchestrator_stack():
    """Lazy import of orchestrator_stack to access functions defined there
    (start_server, start_orchestrator, start_*, the thin process/state wrappers).
    orchestrator_stack imports this module lazily inside main() so no module-load
    cycle exists.
    """
    from scripts.server import orchestrator_stack

    return orchestrator_stack


# Convenience accessors via the lazy proxy — keep cmd_* bodies readable.
def start_server(*a, **kw):
    return _orchestrator_stack().start_server(*a, **kw)


def start_orchestrator(*a, **kw):
    return _orchestrator_stack().start_orchestrator(*a, **kw)


def start_document_formalizer(*a, **kw):
    return _orchestrator_stack().start_document_formalizer(*a, **kw)


def start_sd_server(*a, **kw):
    return _orchestrator_stack().start_sd_server(*a, **kw)


def start_whisper(*a, **kw):
    return _orchestrator_stack().start_whisper(*a, **kw)


def start_handoff_dashboard(*a, **kw):
    return _orchestrator_stack().start_handoff_dashboard(*a, **kw)


def start_tts(*a, **kw):
    return _orchestrator_stack().start_tts(*a, **kw)


def start_aux_service(*a, **kw):
    return _orchestrator_stack().start_aux_service(*a, **kw)


def load_state(*a, **kw):
    return _orchestrator_stack().load_state(*a, **kw)


def save_state(*a, **kw):
    return _orchestrator_stack().save_state(*a, **kw)


def kill_process(*a, **kw):
    return _orchestrator_stack().kill_process(*a, **kw)


def is_port_in_use(*a, **kw):
    return _orchestrator_stack().is_port_in_use(*a, **kw)


def _pids_on_port(*a, **kw):
    return _orchestrator_stack()._pids_on_port(*a, **kw)


def _collect_descendants(*a, **kw):
    return _orchestrator_stack()._collect_descendants(*a, **kw)


def _renice_all_threads(*a, **kw):
    return _orchestrator_stack()._renice_all_threads(*a, **kw)


def check_free_memory(*a, **kw):
    return _orchestrator_stack().check_free_memory(*a, **kw)


def _apply_orchestrator_profile(*a, **kw):
    return _orchestrator_stack()._apply_orchestrator_profile(*a, **kw)


def _clip_gate_output(text: str, *, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _should_skip_stack_change_launch_gate(args: argparse.Namespace) -> str | None:
    if getattr(args, "dev", False):
        return "dev mode"
    if getattr(args, "validate_only", False):
        return "validate-only"
    if getattr(args, "migrate_to", None) and getattr(args, "dry_run", False):
        return "migration dry-run"
    if getattr(args, "skip_stack_change_gate", False):
        return "--skip-stack-change-gate"
    if os.environ.get("ORCHESTRATOR_SKIP_STACK_CHANGE_GATE") == "1":
        return "ORCHESTRATOR_SKIP_STACK_CHANGE_GATE=1"
    return None


def _run_stack_change_launch_gate(args: argparse.Namespace) -> bool:
    """Run the canonical stack-change promotion gate before production launch."""
    skip_reason = _should_skip_stack_change_launch_gate(args)
    if skip_reason is not None:
        print(f"[stack-change-gate] SKIPPED ({skip_reason})")
        return True

    command = list(STACK_CHANGE_LAUNCH_GATE_COMMAND)
    # Thread an EXPLICIT --numa-mode through to the gate subprocess as a FLAG.
    # Since NIB2-69 (2026-09-15) the pipeline evaluates the declared production
    # mode (orchestration/stack_topology.yaml) for both the priors compile and
    # the guard's launch view, and never reads ORCHESTRATOR_STACK_NUMA_MODE — so
    # the old env-threading (the 37-error cold-start class, 2026-07-25) would now
    # be silently ignored. Omitted flag (None) => the gate checks production.
    gate_env = None
    requested_mode = getattr(args, "numa_mode", None)
    if requested_mode:
        command.extend(["--numa-mode", str(requested_mode)])
        print(f"[stack-change-gate] threading --numa-mode {requested_mode} into the gate")
    print("[stack-change-gate] Running canonical launch gate...")
    print("  " + " ".join(command))
    try:
        result = subprocess.run(
            command,
            cwd=str(_PATHS["project_root"]),
            env=gate_env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        print(f"[stack-change-gate] FATAL: could not launch gate: {exc}")
        return False

    if result.stdout:
        print(_clip_gate_output(result.stdout.rstrip()))
    if result.stderr:
        print(_clip_gate_output(result.stderr.rstrip()))
    if result.returncode == 0:
        print("[stack-change-gate] OK")
        return True

    print(f"[stack-change-gate] FATAL: gate exited {result.returncode}; refusing launch")
    # THE CATCH-22, NAMED. Documented 2026-08-11 (autopilot-continuous-optimization.md).
    # This gate refuses a launch while live != config, and the only cure for live !=
    # config is a restart — which requires the launch it just refused. The obvious
    # operator reaction, retrying `start`, therefore cannot work, and until now
    # nothing said so: the message stated the refusal and not the way out. The escape
    # is to stop first, so there is no live process left to drift against.
    print("[stack-change-gate] If this is a live-vs-config drift, retrying `start` CANNOT")
    print("[stack-change-gate]   work: the cure for drift is a restart, and the gate refuses")
    print("[stack-change-gate]   the launch that would perform it. Stop first, then start:")
    print("[stack-change-gate]     orchestrator_stack.py stop --all && orchestrator_stack.py start")
    print("[stack-change-gate]   With nothing live there is no drift to detect, so the gate passes.")
    return False


def _find_pids_on_port(port: int) -> list[int]:
    """Find PIDs listening on a port via lsof (fallback for stale state)."""
    return _stack_processes.pids_on_port(port, timeout=5)


def _stack_prior_serving_ports(path: Path | None = None) -> set[int]:
    """Return live model-serving ports from generated stack priors."""
    priors_path = path or _PATHS["project_root"] / "orchestration/derived/stack_priors.yaml"
    ports: set[int] = set()
    for record in live_stack_role_records(priors_path).values():
        ports.update(stack_prior_serving_ports(stack_prior_serving(record)))
    return ports


def _scan_known_ports() -> dict[int, list[int]]:
    """Scan all known orchestrator ports for running processes."""
    managed_server_ports = {s["port"] for s in HOT_SERVERS + WARM_SERVERS}
    docker_ports = {int(svc["port"]) for svc in DOCKER_SERVICES if "port" in svc}
    manifest_ports = {int(port) for port in PORT_MAP.values()}
    stack_prior_ports = _stack_prior_serving_ports()
    known_ports = sorted(
        managed_server_ports
        | NUMA_REPLICA_PORTS
        | docker_ports
        | manifest_ports
        | stack_prior_ports
    )
    return _stack_processes.scan_known_ports(known_ports)


def _attestable_model_path(model_path: str) -> bool:
    """Return True for concrete GGUF paths whose launch cmdline can be checked."""
    return model_path.endswith(".gguf") and Path(model_path).is_absolute()


def _cmdline_flag_values(cmdline: list[str], *flags: str) -> list[str]:
    values: list[str] = []
    flag_set = set(flags)
    for idx, token in enumerate(cmdline):
        if token in flag_set and idx + 1 < len(cmdline):
            values.append(cmdline[idx + 1])
            continue
        for flag in flags:
            prefix = flag + "="
            if token.startswith(prefix):
                values.append(token[len(prefix):])
    return values


def _cmdline_has_path(cmdline: list[str], expected_path: str) -> bool:
    expected_name = Path(expected_path).name
    return expected_path in cmdline or any(Path(token).name == expected_name for token in cmdline)


def _same_real_model_path(left: str, right: str) -> bool:
    return os.path.realpath(left) == os.path.realpath(right)


def _stack_prior_launch_requirements(path: Path | None = None) -> dict[str, dict[str, str]]:
    contracts = _stack_prior_launch_contracts(path)
    return {
        role: contract["requirements"]
        for role, contract in contracts.items()
        if contract.get("requirements")
    }


def _stack_prior_launch_contracts(path: Path | None = None) -> dict[str, dict[str, Any]]:
    priors_path = path or _PATHS["project_root"] / "orchestration/derived/stack_priors.yaml"

    contracts_by_role: dict[str, dict[str, Any]] = {}
    for role, record in live_stack_role_records(priors_path).items():
        serving = stack_prior_serving(record)
        launch = serving.get("launch") if isinstance(serving, dict) else None
        requirements = launch.get("requirements") if isinstance(launch, dict) else None
        runtime = launch.get("runtime") if isinstance(launch, dict) else None
        cleaned_requirements = {
            str(key): str(value)
            for key, value in requirements.items()
            if value is not None
        } if isinstance(requirements, dict) else {}
        contracts_by_role[str(role)] = {
            "requirements": cleaned_requirements,
            "runtime": runtime if isinstance(runtime, dict) else {},
            "ports": stack_prior_serving_ports(serving),
        }
    return contracts_by_role


def _refresh_runtime_facts_manifest(
    source: str,
    state: dict[str, ProcessInfo],
    *,
    stack_numa_mode: str | None = None,
) -> Path | None:
    """Best-effort derived runtime facts cache for operators and autopilot."""
    stack_priors_path = _PATHS["project_root"] / "orchestration/derived/stack_priors.yaml"
    try:
        path = write_runtime_facts_manifest(
            state=state,
            launch_contracts=_stack_prior_launch_contracts(stack_priors_path),
            stack_priors_path=stack_priors_path,
            stack_numa_mode=stack_numa_mode,
            tmp_dir=_PATHS["tmp_dir"],
            repo_short_sha=_orchestrator_stack()._repo_short_sha(),
            source=source,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[runtime-facts] WARN: failed to write runtime facts manifest: {exc}")
        return None
    print(f"[runtime-facts] Wrote {path}")
    return path


def _merge_persisted_state_for_facts(
    persisted: Mapping[str, ProcessInfo],
    in_memory: Mapping[str, ProcessInfo],
) -> dict[str, ProcessInfo]:
    """Merge persisted state rows UNDER the in-memory (freshly-(re)started) rows.

    ESC-8 Fix 2-addendum: the runtime-facts refresh must see the FULL realized
    fleet, not just this invocation's newly-started rows. A subset/``--only``
    start (or a start that leaves already-healthy llama-servers untouched) keeps
    those llama rows solely in the persisted state file; passing the in-memory
    ``state`` alone yields ``selected_servers: []`` even though the fleet is up
    (the 09:14 manifest defect). In-memory rows win on key collision (freshest
    pid/port). ``write_runtime_facts_manifest``'s pid-liveness filter drops any
    stale persisted rows, so a merged-in dead row is harmless.
    """
    merged: dict[str, ProcessInfo] = dict(persisted)
    merged.update(in_memory)
    return merged


def _model_path_attestation(info: ProcessInfo, alive: bool, cmdline: list[str]) -> str:
    if info.pid == -1:
        return "docker"
    if not alive:
        return "dead"
    if info.model_path == "uvicorn":
        if not cmdline:
            return "unknown"
        return "ok" if any("uvicorn" in token for token in cmdline) else "unknown"
    if not _attestable_model_path(info.model_path):
        return "n/a"
    if not cmdline:
        return "unknown"
    if info.model_path in cmdline:
        return "ok"
    expected_name = Path(info.model_path).name
    if any(Path(token).name == expected_name for token in cmdline):
        return "ok"
    if any(token.endswith(".gguf") for token in cmdline):
        return "model-drift"
    return "unknown"


def _status_attestation(
    info: ProcessInfo,
    alive: bool,
    cmdline: list[str],
    launch_requirements: dict[str, str] | None = None,
) -> str:
    """Classify whether live process args match persisted/generated stack state."""
    base = _model_path_attestation(info, alive, cmdline)
    if base in {"docker", "dead", "model-drift"}:
        return base

    expected_mmproj = (launch_requirements or {}).get("mmproj_path")
    if not expected_mmproj:
        return base
    if not cmdline:
        return "unknown"
    if _cmdline_has_path(cmdline, expected_mmproj):
        return "ok" if base == "n/a" else base
    return "mmproj-drift"


def _attestation_warning(
    name: str,
    info: ProcessInfo,
    attestation: str,
    cmdline: list[str],
    launch_requirements: dict[str, str],
) -> str | None:
    if attestation == "model-drift":
        expected = Path(info.model_path).name
        actual_models = ", ".join(Path(token).name for token in cmdline if token.endswith(".gguf"))
        return f"{name} pid {info.pid} expected {expected}; live cmdline has {actual_models}"
    if attestation == "mmproj-drift":
        expected = Path(str(launch_requirements.get("mmproj_path") or "")).name
        actual_mmproj = ", ".join(
            Path(token).name for token in _cmdline_flag_values(cmdline, "--mmproj")
        ) or "no --mmproj"
        return (
            f"{name} pid {info.pid} expected mmproj {expected}; "
            f"live cmdline has {actual_mmproj}"
        )
    return None


def _last_cmdline_flag_value(cmdline: list[str], *flags: str) -> str | None:
    values = _cmdline_flag_values(cmdline, *flags)
    return values[-1] if values else None


def _runtime_value_warning(
    name: str,
    info: ProcessInfo,
    label: str,
    expected: Any,
    actual: Any,
) -> str:
    return (
        f"{name} pid {info.pid} runtime {label} expected {expected}; "
        f"live cmdline has {actual}"
    )


# ---------------------------------------------------------------------------
# Derived attestation coverage
# ---------------------------------------------------------------------------
#
# 2026-08-02: the attestation checklist below used to be HAND-WRITTEN — a fixed
# sequence of field names read out of the declared launch record. Anything the
# compiler emitted but nobody had listed was never compared, and the omission was
# invisible because both halves were individually correct and the output was green.
# `device` was the field that fell through: a 27B declared on ROCm0 ran on 24 CPU
# threads with the GPU at 0% while reporting `healthy / attest ok`.
#
# Adding one more name by hand does not close that class, so the checklist is now
# DERIVED: `_declared_runtime_field_paths` walks the keys the producer actually
# emitted for this role, and every one of them must resolve to an entry in
# `_RUNTIME_FIELD_CHECKS`. A declared field with no entry is itself REPORTED —
# "this role declares X and nothing verifies it" — which inverts the failure mode.
# Before, an unknown field was silent; now it is a finding.
#
# `_ATTESTED_RUNTIME_FIELDS` is the old hand list, kept as a FLOOR (same shape as
# `REQUIRED_SOURCE_ARTIFACTS` in scripts/validate/stack_change_guard.py): iteration
# is over `sorted(set(FLOOR) | set(declared))`, so deleting a mapping that used to
# exist is caught even if the producer also stops emitting the field.
#
# Modes:
#   dedicated   verified above by purpose-written code (path/semantics too specific
#               for the table); listed here so the field still counts as covered.
#   container   a nested block whose CHILDREN carry the facts (walked recursively).
#   int_flag              `<flag> <int>`, emitted for int >= 0 (mirrors the launcher).
#   int_or_token_flag     as int_flag, but a non-empty string token is also valid
#                         (`-ngl all`).
#   positive_int_flag     emitted only when > 0.
#   float_flag            emitted as str(float(value)).
#   not_emitted           declared, but this kernel's launcher cannot emit it —
#                         warn if a role ever declares it truthy, because then the
#                         declaration is a promise the launch cannot keep.
#   implied               fully determined by another field that IS checked.
#   env_prefix            not a CLI flag: verified against /proc/<pid>/environ.
#   env_value             not a CLI flag: exact environment variable match.
#   cross_check           compared against another declared field, not the cmdline.
#   metadata              provenance only; no runtime consequence to attest.
_RUNTIME_FIELD_CHECKS: dict[str, tuple[str, tuple[str, ...], str]] = {
    # section: requirements
    "requirements.model_path": ("dedicated", (), ""),
    "requirements.draft_model_path": ("dedicated", (), ""),
    "requirements.mmproj_path": ("dedicated", (), ""),
    # section: runtime (top level)
    "runtime.binary_path": ("dedicated", (), ""),
    "runtime.binary_dir": ("implied", (), "binary_path (full path) is compared verbatim"),
    "runtime.binary_family": ("metadata", (), "provenance label; binary_path is the fact"),
    "runtime.env_policy": (
        "metadata",
        (),
        "names the env recipe; its observable effects are ld_library_path/kmp_blocktime",
    ),
    "runtime.ld_library_path": ("env_prefix", ("LD_LIBRARY_PATH",), ""),
    "runtime.kmp_blocktime": ("env_value", ("KMP_BLOCKTIME",), ""),
    "runtime.cache": ("container", (), ""),
    "runtime.flags": ("container", (), ""),
    # section: runtime.cache
    "runtime.cache.context_tokens": ("dedicated", (), ""),
    "runtime.cache.slots": ("dedicated", (), ""),
    # 2026-08-02 per-instance `-np`. A MAP, port -> slots, not a container of
    # named fields: its keys are ports, so they cannot be enumerated in this
    # table and the walker is told not to descend into it (see
    # _RUNTIME_FIELD_WALK_LEAVES). The dedicated check picks the entry matching
    # the live process's own --port, which is what makes attestation
    # instance-accurate — the role-level `slots` cannot be right for all three
    # instances of a role whose full and halves differ.
    "runtime.cache.slots_by_port": ("dedicated", (), ""),
    "runtime.cache.ubatch": ("dedicated", (), ""),
    "runtime.cache.kv_type_k": ("dedicated", (), ""),
    "runtime.cache.kv_type_v": ("dedicated", (), ""),
    "runtime.cache.no_mmap": ("dedicated", (), ""),
    "runtime.cache.mlock": ("dedicated", (), ""),
    "runtime.cache.slot_save_path": ("dedicated", (), ""),
    # Unified KV pool: a BOOLEAN whose live truth depends on TWO flags (-kvu and
    # -np), so it cannot be a token check -- see the dedicated block below.
    "runtime.cache.kv_unified": ("dedicated", (), ""),
    "runtime.cache.draft_kv_type_k": (
        "int_or_token_flag",
        ("-ctkd", "--spec-draft-type-k", "--cache-type-k-draft"),
        "",
    ),
    "runtime.cache.draft_kv_type_v": (
        "int_or_token_flag",
        ("-ctvd", "--spec-draft-type-v", "--cache-type-v-draft"),
        "",
    ),
    # --kv-hadamard was removed in the v6 binary (it would crash), so the launcher's
    # emission site is commented out. A role declaring it true would therefore get a
    # silent no-op, which is exactly the kind of unkept declaration worth reporting.
    "runtime.cache.kv_hadamard": (
        "not_emitted",
        (),
        "--kv-hadamard removed in the v6 binary; the launcher cannot emit it",
    ),
    # section: runtime.flags
    "runtime.flags.flash_attn": ("dedicated", (), ""),
    "runtime.flags.jinja": ("dedicated", (), ""),
    "runtime.flags.reasoning": ("dedicated", (), ""),
    "runtime.flags.override_kv": ("dedicated", (), ""),
    "runtime.flags.device": ("dedicated", (), ""),
    "runtime.flags.device_draft": ("dedicated", (), ""),
    "runtime.flags.spec": ("container", (), ""),
    # The declared serving flags. orchestrator_stack.py keeps these as DATA
    # (`_RUNTIME_SERVING_FLAG_ARGS`) and emits them; nothing verified them until now.
    # -ngl is the sharpest of the three and sits on the same failure path as `device`:
    # without it a server takes `--device ROCm0`, offloads nothing, and is a GPU launch
    # in name only — indistinguishable, to the old checklist, from a healthy one.
    "runtime.flags.n_gpu_layers": ("int_or_token_flag", ("-ngl", "--n-gpu-layers", "--gpu-layers"), ""),
    "runtime.flags.image_min_tokens": ("int_flag", ("--image-min-tokens",), ""),
    "runtime.flags.cache_ram": ("int_flag", ("--cache-ram",), ""),
    # Per-role chat template file: a string token on the cmdline, same emission
    # rule as `-ngl all` (the launcher additionally refuses at build time when the
    # declared path is missing, so a live server always matches its declaration).
    "runtime.flags.chat_template_file": (
        "int_or_token_flag",
        ("--chat-template-file",),
        "",
    ),
    # section: runtime.flags.spec
    "runtime.flags.spec.enabled": ("dedicated", (), ""),
    "runtime.flags.spec.type": ("dedicated", (), ""),
    "runtime.flags.spec.draft_max": ("dedicated", (), ""),
    "runtime.flags.spec.draft_p_min": ("dedicated", (), ""),
    "runtime.flags.spec.threads_draft": ("dedicated", (), ""),
    "runtime.flags.spec.draft_min": ("int_flag", ("--spec-draft-n-min",), ""),
    "runtime.flags.spec.draft_p_split": ("float_flag", ("--draft-p-split",), ""),
    "runtime.flags.spec.ngram_mod_n_min": ("positive_int_flag", ("--spec-ngram-mod-n-min",), ""),
    "runtime.flags.spec.ngram_mod_n_max": ("positive_int_flag", ("--spec-ngram-mod-n-max",), ""),
    "runtime.flags.spec.ngram_mod_n_match": ("positive_int_flag", ("--spec-ngram-mod-n-match",), ""),
    # The launcher takes `-md` from spec.draft_model_path while the dedicated draft
    # check above reads requirements.draft_model_path. Two declarations of one fact:
    # compare them, so a divergence is a finding rather than a check aimed at the
    # value the launcher did not use.
    "runtime.flags.spec.draft_model_path": (
        "cross_check",
        ("requirements.draft_model_path",),
        "",
    ),
    "runtime.flags.spec.disabled_by": (
        "metadata",
        (),
        "records why speculation is off; spec.enabled is the attested fact",
    ),
}

# FLOOR — the hand-written checklist as it stood before this was derived. These
# paths MUST stay mapped; losing one is reported even when the producer no longer
# emits the field. Do not prune this list to match the current record.
_ATTESTED_RUNTIME_FIELDS: tuple[str, ...] = (
    "requirements.model_path",
    "requirements.draft_model_path",
    "requirements.mmproj_path",
    "runtime.binary_path",
    "runtime.cache.context_tokens",
    "runtime.cache.slots",
    "runtime.cache.ubatch",
    "runtime.cache.kv_type_k",
    "runtime.cache.kv_type_v",
    "runtime.cache.no_mmap",
    "runtime.cache.mlock",
    "runtime.cache.slot_save_path",
    "runtime.flags.device",
    "runtime.flags.device_draft",
    "runtime.flags.flash_attn",
    "runtime.flags.jinja",
    "runtime.flags.reasoning",
    "runtime.flags.override_kv",
    "runtime.flags.spec.enabled",
    "runtime.flags.spec.type",
    "runtime.flags.spec.draft_max",
    "runtime.flags.spec.draft_p_min",
    "runtime.flags.spec.threads_draft",
)

_RUNTIME_FIELD_WALK_MAX_DEPTH = 4

# Declared blocks that are MAPS OVER DATA, not blocks of named fields. The walker
# stops at these: their keys are values (ports), so descending would demand a
# `_RUNTIME_FIELD_CHECKS` entry per port and report every instance as unattested.
# The block itself must still be mapped — it is, as `dedicated` — so adding a new
# data-keyed map without a checker is still a finding.
_RUNTIME_FIELD_WALK_LEAVES: frozenset[str] = frozenset({"runtime.cache.slots_by_port"})


def _declared_runtime_field_paths(
    requirements: dict[str, Any],
    runtime: dict[str, Any],
) -> set[str]:
    """Return every field path the PRODUCER emitted for this role.

    The checklist is derived from this, not from a literal, so a field added to
    the compiled record is verified (or reported as unverifiable) without anyone
    having to remember to extend a list here.
    """

    paths: set[str] = set()

    def walk(prefix: str, node: Any, depth: int) -> None:
        if not isinstance(node, dict) or depth > _RUNTIME_FIELD_WALK_MAX_DEPTH:
            return
        for key, value in node.items():
            path = f"{prefix}.{key}"
            paths.add(path)
            if path in _RUNTIME_FIELD_WALK_LEAVES:
                continue
            walk(path, value, depth + 1)

    walk("requirements", requirements, 1)
    walk("runtime", runtime, 1)
    return paths


def _declared_runtime_field_value(
    requirements: dict[str, Any],
    runtime: dict[str, Any],
    path: str,
) -> Any:
    section, _, rest = path.partition(".")
    node: Any = requirements if section == "requirements" else runtime
    for key in rest.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _process_environ(pid: int) -> dict[str, str] | None:
    """Read a live process environment, or None when it cannot be read.

    Unreadable is NOT treated as a mismatch: a contract fixture or a process owned
    by another user has no environ to compare, and inventing a warning there would
    make the check noisy exactly where it has no evidence.
    """
    if pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/environ").read_text(errors="replace")
    except OSError:
        return None
    return dict(
        entry.split("=", 1) for entry in raw.split("\0") if "=" in entry
    )


def _derived_runtime_field_warnings(
    name: str,
    info: ProcessInfo,
    cmdline: list[str],
    requirements: dict[str, Any],
    runtime: dict[str, Any],
) -> list[str]:
    """Verify every DECLARED runtime field, and report the ones nothing verifies.

    Iterates ``sorted(set(FLOOR) | set(declared))`` — the producer's own keys, with
    the historical hand list as a floor. Fields handled by the dedicated checks above
    are skipped here (their warning text is the established one); everything else is
    compared generically, and an unmapped declared field becomes a finding.
    """
    warnings: list[str] = []
    declared = _declared_runtime_field_paths(requirements, runtime)
    environ: dict[str, str] | None | bool = False  # False = not yet read

    for path in sorted(set(_ATTESTED_RUNTIME_FIELDS) | declared):
        entry = _RUNTIME_FIELD_CHECKS.get(path)
        if entry is None:
            # THE inversion: a field with no mapping is REPORTED, not skipped. Either
            # the producer emitted something new that nothing verifies, or a mapping
            # the floor requires was deleted.
            if path in declared:
                warnings.append(
                    f"{name} pid {info.pid} declares runtime field {path} but nothing "
                    f"attests it (no entry in _RUNTIME_FIELD_CHECKS)"
                )
            else:
                warnings.append(
                    f"{name} pid {info.pid} attestation coverage lost: {path} is a "
                    f"required attested field with no entry in _RUNTIME_FIELD_CHECKS"
                )
            continue
        if path not in declared:
            # Floor entry the producer did not emit for this role: the dedicated
            # checks already no-op on an absent value. Nothing to compare.
            continue

        mode, flags, _note = entry
        value = _declared_runtime_field_value(requirements, runtime, path)
        label = path.split(".", 1)[1] if path.startswith("requirements.") else path
        label = label.removeprefix("runtime.").removeprefix("cache.").removeprefix("flags.")

        if mode in ("dedicated", "container", "metadata", "implied"):
            continue

        if mode == "not_emitted":
            if value:
                warnings.append(
                    f"{name} pid {info.pid} declares {label}={value!r} but the launcher "
                    f"cannot emit it ({_note or 'not emitted by this kernel'}); the "
                    f"declaration has no runtime effect"
                )
            continue

        if mode == "cross_check":
            other_path = flags[0] if flags else ""
            other = _declared_runtime_field_value(requirements, runtime, other_path)
            if value is not None and other is not None and str(value) != str(other):
                warnings.append(
                    f"{name} pid {info.pid} runtime {label} is {value}, but "
                    f"{other_path} is {other}; one fact is declared twice and the "
                    f"two declarations disagree"
                )
            continue

        if mode in ("env_prefix", "env_value"):
            if value in (None, "", [], {}):
                continue
            if environ is False:
                environ = _process_environ(info.pid)
            if not environ:
                continue
            var = flags[0]
            live = environ.get(var)
            if mode == "env_value":
                if live != str(value):
                    warnings.append(
                        f"{name} pid {info.pid} runtime {label} expected {value}; "
                        f"live {var} is {live or 'unset'}"
                    )
                continue
            expected_entries = [str(entry) for entry in value] if isinstance(value, list) else [str(value)]
            live_entries = (live or "").split(":")
            if live_entries[: len(expected_entries)] != expected_entries:
                warnings.append(
                    f"{name} pid {info.pid} runtime {label} expected leading entries "
                    f"{':'.join(expected_entries)}; live {var} is {live or 'unset'}"
                )
            continue

        # Remaining modes compare a value token on the live cmdline. The expected
        # token mirrors the launcher's own formatting rule for that flag, so a
        # declaration the launcher would silently drop is reported rather than
        # compared against a flag that was never going to appear.
        expected_token: str | None = None
        if isinstance(value, bool) or value is None:
            expected_token = None
        elif mode == "float_flag":
            if isinstance(value, (int, float)):
                expected_token = str(float(value))
        elif mode == "positive_int_flag":
            if isinstance(value, int) and value > 0:
                expected_token = str(value)
            elif isinstance(value, str) and value.isdigit() and int(value) > 0:
                expected_token = value
        elif mode in ("int_flag", "int_or_token_flag"):
            if isinstance(value, int) and value >= 0:
                expected_token = str(value)
            elif mode == "int_or_token_flag" and isinstance(value, str) and value.strip():
                expected_token = value.strip()

        if expected_token is None:
            if value not in (None, "", [], {}) and not isinstance(value, bool):
                warnings.append(
                    f"{name} pid {info.pid} declares {label}={value!r}, which the "
                    f"launcher cannot express as {flags[0]}; the declaration has no "
                    f"runtime effect"
                )
            continue

        actual = _last_cmdline_flag_value(cmdline, *flags)
        if actual != expected_token:
            warnings.append(_runtime_value_warning(
                name, info, label, expected_token, actual or f"no {flags[0]}"
            ))

    return warnings


_KV_UNIFIED_ON_FLAGS: frozenset[str] = frozenset({"-kvu", "--kv-unified"})
_KV_UNIFIED_OFF_FLAGS: frozenset[str] = frozenset({"-no-kvu", "--no-kv-unified"})


def _live_kv_unified(cmdline: list[str]) -> bool:
    """What llama-server resolved ``kv_unified`` to for this cmdline.

    Last explicit flag wins (argparse semantics). With neither flag, the server
    default applies: unified iff ``-np``/``--parallel`` is absent (auto slots).
    """
    for token in reversed(cmdline):
        if token in _KV_UNIFIED_ON_FLAGS:
            return True
        if token in _KV_UNIFIED_OFF_FLAGS:
            return False
    return _last_cmdline_flag_value(cmdline, "-np", "--parallel") is None


def _runtime_attestation_warnings(
    name: str,
    info: ProcessInfo,
    cmdline: list[str],
    launch_contract: dict[str, Any],
) -> list[str]:
    if not cmdline or info.pid == -1 or not launch_contract:
        return []
    requirements = launch_contract.get("requirements")
    runtime = launch_contract.get("runtime")
    if not isinstance(requirements, dict):
        requirements = {}
    if not isinstance(runtime, dict):
        runtime = {}

    warnings: list[str] = []
    binary_path = runtime.get("binary_path")
    if isinstance(binary_path, str) and binary_path:
        actual_binary = cmdline[0]
        if actual_binary != binary_path:
            warnings.append(_runtime_value_warning(
                name, info, "binary_path", Path(binary_path).name, Path(actual_binary).name
            ))

    model_path = requirements.get("model_path")
    if isinstance(model_path, str) and model_path:
        model_values = _cmdline_flag_values(cmdline, "-m", "--model")
        if not any(_cmdline_has_path([value], model_path) for value in model_values):
            actual = ", ".join(Path(value).name for value in model_values) or "no -m"
            warnings.append(_runtime_value_warning(
                name, info, "model_path", Path(model_path).name, actual
            ))

    draft_model_path = requirements.get("draft_model_path")
    if isinstance(draft_model_path, str) and draft_model_path:
        draft_values = _cmdline_flag_values(cmdline, "-md")
        has_explicit_draft = any(
            _cmdline_has_path([value], draft_model_path) for value in draft_values
        )
        has_embedded_nextn_spec = (
            not draft_values
            and isinstance(model_path, str)
            and bool(_cmdline_flag_values(cmdline, "--spec-type"))
            and _same_real_model_path(model_path, draft_model_path)
        )
        if not has_explicit_draft and not has_embedded_nextn_spec:
            actual = ", ".join(Path(value).name for value in draft_values) or "no -md"
            warnings.append(_runtime_value_warning(
                name, info, "draft_model_path", Path(draft_model_path).name, actual
            ))

    mmproj_path = requirements.get("mmproj_path")
    if isinstance(mmproj_path, str) and mmproj_path:
        mmproj_values = _cmdline_flag_values(cmdline, "--mmproj")
        if not any(_cmdline_has_path([value], mmproj_path) for value in mmproj_values):
            actual = ", ".join(Path(value).name for value in mmproj_values) or "no --mmproj"
            warnings.append(_runtime_value_warning(
                name, info, "mmproj_path", Path(mmproj_path).name, actual
            ))

    cache = runtime.get("cache")
    flags = runtime.get("flags")
    cache = cache if isinstance(cache, dict) else {}
    flags = flags if isinstance(flags, dict) else {}
    scalar_flags = {
        "context_tokens": ("-c", "--ctx-size"),
        "ubatch": ("-ub", "--ubatch-size"),
        "kv_type_k": ("-ctk",),
        "kv_type_v": ("-ctv",),
    }
    for key, flag_names in scalar_flags.items():
        expected = cache.get(key)
        if expected is None:
            continue
        actual = _last_cmdline_flag_value(cmdline, *flag_names)
        if actual != str(expected):
            warnings.append(_runtime_value_warning(
                name, info, key, expected, actual or f"no {flag_names[0]}"
            ))

    # `-np`, PER INSTANCE. 2026-08-02: this used to sit in `scalar_flags` above and
    # compare the live `-np` against the ROLE-level `runtime.cache.slots`. Once a
    # role's full and half instances legitimately run different slot counts
    # (frontdoor 16 / 4 / 4), that comparison is wrong for two of its three
    # processes — it would report drift on correctly-launched servers, which is the
    # fastest way to teach a reader to ignore this output. The expectation is now
    # selected by the live process's OWN --port, so each instance is attested
    # against the number the compiler resolved for it. The role-level value stays
    # as the fallback for a single-instance role or a record compiled before
    # `slots_by_port` existed.
    expected_slots = cache.get("slots")
    slots_by_port = cache.get("slots_by_port")
    live_port = _last_cmdline_flag_value(cmdline, "--port")
    if isinstance(slots_by_port, dict) and live_port is not None:
        for key, value in slots_by_port.items():
            if str(key) == str(live_port):
                expected_slots = value
                break
    if expected_slots is not None:
        actual = _last_cmdline_flag_value(cmdline, "-np", "--parallel")
        if actual != str(expected_slots):
            warnings.append(_runtime_value_warning(
                name, info, "slots", expected_slots, actual or "no -np"
            ))

    for key, flag_name in {"no_mmap": "--no-mmap", "mlock": "--mlock"}.items():
        expected = cache.get(key)
        if not isinstance(expected, bool):
            continue
        actual = flag_name in cmdline
        if actual != expected:
            warnings.append(_runtime_value_warning(name, info, key, expected, actual))

    # kv_unified: the LIVE value is whatever llama-server resolved, which is the last
    # -kvu/--kv-unified/-no-kvu/--no-kv-unified on the cmdline, else "auto" -- and
    # auto means unified ONLY when -np is absent (tools/server/server.cpp:145-150).
    # Checking flag presence alone would pass a process that runs split KV while its
    # declaration says unified, which is the exact defect this field was added for.
    expected_kvu = cache.get("kv_unified")
    if isinstance(expected_kvu, bool):
        actual_kvu = _live_kv_unified(cmdline)
        if actual_kvu != expected_kvu:
            warnings.append(_runtime_value_warning(
                name, info, "kv_unified", expected_kvu, actual_kvu
            ))

    slot_save_path = cache.get("slot_save_path")
    if isinstance(slot_save_path, str) and slot_save_path:
        actual = _last_cmdline_flag_value(cmdline, "--slot-save-path")
        if actual is None or not _cmdline_has_path([actual], slot_save_path):
            warnings.append(_runtime_value_warning(
                name, info, "slot_save_path", Path(slot_save_path).name,
                Path(actual).name if actual else "no --slot-save-path",
            ))

    # 2026-08-01: `device` was the ONE declared field this table never checked. That
    # omission is why a role could run on the wrong processor while attesting clean:
    # architect_general was verified against binary_path, model, context, slots,
    # ubatch, kv types, mmap, mlock, slot-save-path, flash_attn, jinja, reasoning,
    # override_kv and the entire spec block — and passed all of them — while serving
    # a GPU-declared 27B on 24 CPU threads under `--device none`. Attestation that
    # covers every field except which processor is executing is not attestation.
    #
    # BOTH directions are covered. A declared device must appear verbatim. An
    # UNDECLARED device means "CPU role", so a live device that is anything other
    # than `none` is drift too — that is the check that would catch the mirror-image
    # accident of a CPU role landing on the GPU. A cmdline carrying no device flag at
    # all is not asserted against: that is the shape of a contract fixture, not of a
    # stack-launched process (the launcher always emits one).
    declared_device = flags.get("device")
    declared_device = (
        declared_device.strip()
        if isinstance(declared_device, str) and declared_device.strip()
        else None
    )
    actual_device = _last_cmdline_flag_value(cmdline, "--device", "-dev")
    if declared_device is not None:
        if actual_device != declared_device:
            warnings.append(_runtime_value_warning(
                name, info, "device", declared_device, actual_device or "no --device"
            ))
    elif actual_device is not None and actual_device != "none":
        warnings.append(_runtime_value_warning(
            name, info, "device", "none", actual_device
        ))

    # The draft device follows the target's device unless a role declares its own.
    declared_draft_device = flags.get("device_draft")
    declared_draft_device = (
        declared_draft_device.strip()
        if isinstance(declared_draft_device, str) and declared_draft_device.strip()
        else declared_device
    )
    if _cmdline_flag_values(cmdline, "--spec-type") or _cmdline_flag_values(cmdline, "-md"):
        actual_draft_device = _last_cmdline_flag_value(cmdline, "--device-draft", "-devd")
        if declared_draft_device is not None:
            if actual_draft_device != declared_draft_device:
                warnings.append(_runtime_value_warning(
                    name, info, "device_draft", declared_draft_device,
                    actual_draft_device or "no --device-draft",
                ))
        elif actual_draft_device is not None and actual_draft_device != "none":
            warnings.append(_runtime_value_warning(
                name, info, "device_draft", "none", actual_draft_device
            ))

    flash_attn = flags.get("flash_attn")
    if isinstance(flash_attn, bool):
        actual = _last_cmdline_flag_value(cmdline, "--flash-attn") == "on"
        if actual != flash_attn:
            warnings.append(_runtime_value_warning(
                name, info, "flash_attn", flash_attn, actual
            ))
    jinja = flags.get("jinja")
    if isinstance(jinja, bool):
        actual = "--jinja" in cmdline
        if actual != jinja:
            warnings.append(_runtime_value_warning(name, info, "jinja", jinja, actual))
    reasoning = flags.get("reasoning")
    if reasoning is not None:
        actual = _last_cmdline_flag_value(cmdline, "--reasoning")
        if actual != str(reasoning):
            warnings.append(_runtime_value_warning(
                name, info, "reasoning", reasoning, actual or "no --reasoning"
            ))

    override_kv = flags.get("override_kv", [])
    expected_overrides = sorted(str(value) for value in override_kv) if isinstance(
        override_kv, list
    ) else []
    actual_overrides = sorted(_cmdline_flag_values(cmdline, "--override-kv"))
    if actual_overrides != expected_overrides:
        warnings.append(_runtime_value_warning(
            name, info, "override_kv", expected_overrides, actual_overrides
        ))

    spec = flags.get("spec")
    spec = spec if isinstance(spec, dict) else {}
    expected_spec_enabled = spec.get("enabled") is True
    actual_spec_enabled = bool(
        _cmdline_flag_values(cmdline, "-md")
        or _cmdline_flag_values(cmdline, "--spec-type")
    )
    if actual_spec_enabled != expected_spec_enabled:
        warnings.append(_runtime_value_warning(
            name, info, "spec.enabled", expected_spec_enabled, actual_spec_enabled
        ))
    if expected_spec_enabled:
        for key, flag_names in {
            "type": ("--spec-type",),
            # 2026-06-26 v6 cutover: v6 removed --draft-max (arg_removed -> exit);
            # n-max is now emitted as --spec-draft-n-max (same draft_max value).
            "draft_max": ("--spec-draft-n-max",),
            "draft_p_min": ("--draft-p-min",),
            "threads_draft": ("--threads-draft",),
        }.items():
            expected = spec.get(key)
            if expected is None:
                continue
            # 2026-06-26 v6 cutover: the only valid MTP spec-type token in v6 is
            # 'draft-mtp'; normalize any legacy 'mtp' contract value so the
            # reader expects 'draft-mtp' (not the removed bare 'mtp' token).
            if key == "type" and str(expected) == "mtp":
                expected = "draft-mtp"
            actual = _last_cmdline_flag_value(cmdline, *flag_names)
            if actual != str(expected):
                warnings.append(_runtime_value_warning(
                    name, info, f"spec.{key}", expected, actual or f"no {flag_names[0]}"
                ))

    # Everything above is the checks that were written by hand. This closes the loop
    # over the fields the PRODUCER declared: it verifies the ones the table can map
    # and REPORTS the ones nothing maps. Appended last so the established warnings
    # keep their existing order and text.
    warnings.extend(
        _derived_runtime_field_warnings(name, info, cmdline, requirements, runtime)
    )
    return warnings


def _launch_contract_for_process(
    name: str,
    info: ProcessInfo,
    contracts_by_role: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    direct = contracts_by_role.get(info.role) or contracts_by_role.get(name)
    if direct:
        return direct
    canonical_role = str(Role.from_string(info.role) or info.role)
    canonical_name = str(Role.from_string(name) or name)
    direct = contracts_by_role.get(canonical_role) or contracts_by_role.get(canonical_name)
    if direct:
        return direct
    for contract in contracts_by_role.values():
        ports = contract.get("ports")
        if isinstance(ports, list) and info.port in ports:
            return contract
    return {}


def _episodic_embedding_status_line() -> str:
    """Return a concise read-only health line for the episodic FAISS mirror."""
    try:
        from scripts.maintenance.repair_episodic_embeddings import (
            DEFAULT_DB_PATH,
            DEFAULT_FAISS_PATH,
            DEFAULT_REEMBEDDED_PATH,
            diagnose as _diagnose_embeddings,
        )

        report = _diagnose_embeddings(
            DEFAULT_DB_PATH,
            DEFAULT_FAISS_PATH,
            DEFAULT_REEMBEDDED_PATH,
        )
    except Exception as exc:  # noqa: BLE001 - status should be diagnostic-only
        return f"Episodic FAISS: unknown ({exc})"

    status = "healthy" if report.healthy else "ORPHANED"
    indexed_count = getattr(report, "n_db_indexed", 0) or report.n_db_routing
    return (
        f"Episodic FAISS: {status} — "
        f"{report.n_faiss_vectors:,}/{indexed_count:,} indexed vectors "
        f"({report.faiss_coverage:.1%}), "
        f"id_map {report.n_id_map:,} ids / overlap {report.id_map_overlap_live:.1%} "
        f"missing {getattr(report, 'missing_id_count', 0):,} "
        f"stale {getattr(report, 'stale_id_count', 0):,}, "
        f"{report.orphan_count:,} live repairable lag/stale; "
        f"training artifact reembedded overlap {report.overlap_live:.1%} "
        f"(non-blocking, {getattr(report, 'reembedded_stale_count', 0):,} stale)"
    )


def runtime_attestation_warnings(
    state: dict[str, ProcessInfo] | None = None,
) -> list[str]:
    """Return concrete live process drift warnings without mutating stack state."""
    state = load_state() if state is None else dict(state)

    warnings: list[str] = []
    seen_pids: set[int] = set()
    contracts_by_role = _stack_prior_launch_contracts()
    state_ports = {info.port for info in state.values()}
    for port, pids in sorted(_scan_known_ports().items()):
        if port not in state_ports:
            warnings.append(
                f"known stack port {port} has unmanaged listener pid(s) "
                + ",".join(str(pid) for pid in pids)
            )
    for name, info in sorted(state.items()):
        if info.pid != -1 and info.pid in seen_pids:
            continue
        seen_pids.add(info.pid)
        cmdline: list[str] = []

        if info.pid == -1:
            alive = docker_container_running(info.role)
        else:
            try:
                os.kill(info.pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            if not alive and is_port_in_use(info.port):
                replacement_pids = _pids_on_port(info.port)
                if replacement_pids:
                    info = ProcessInfo(
                        role=info.role,
                        pid=replacement_pids[0],
                        port=info.port,
                        started_at=info.started_at,
                        model_path=info.model_path,
                        log_file=info.log_file,
                    )
                    alive = True
            cmdline = _stack_processes.process_cmdline(info.pid) if alive else []

        launch_contract = _launch_contract_for_process(name, info, contracts_by_role)
        launch_requirements = launch_contract.get("requirements")
        launch_requirements = launch_requirements if isinstance(launch_requirements, dict) else {}
        attestation = _status_attestation(info, alive, cmdline, launch_requirements)
        warning = _attestation_warning(
            name,
            info,
            attestation,
            cmdline,
            launch_requirements,
        )
        if warning:
            warnings.append(warning)
        warnings.extend(_runtime_attestation_warnings(name, info, cmdline, launch_contract))
    return warnings


def _preserved_process_info(
    role: str,
    port: int,
    model_path: str,
    log_file: str = "",
) -> ProcessInfo | None:
    """Build a durable state record for an already-healthy listener."""
    pids = _pids_on_port(port)
    if not pids:
        print(f"  [WARN] {role} on port {port} is healthy but listener PID was not found")
        return None
    return ProcessInfo(
        role=role,
        pid=pids[0],
        port=port,
        started_at=datetime.now().isoformat(),
        model_path=model_path,
        log_file=log_file,
    )


def _only_mode_transition_allowed(numa_mode: str, realized_mode: str) -> bool:
    """Whether a `--only` start may proceed with `numa_mode` != realized mode.

    Additive mode promotion (2026-07-23 lineup restoration): an EXPLICIT
    `--numa-mode both` over a realized single-mode fleet only ADDS the missing
    complementary instances — skip-healthy leaves every running server
    untouched, so it is the deterministic no-outage path to the ratified
    big+quarters lineup. The arg is the deliberate-authority signal (ESC-8's
    threat model is env/manifest ACCIDENTAL resurrection; an explicit arg is
    neither). Every other mismatch — especially a NARROWER requested mode,
    which would imply stopping live servers — refuses.
    """
    return numa_mode == "both" and realized_mode in ("quarter", "full")


def cmd_start(args: argparse.Namespace) -> int:
    """Start the orchestrator stack."""
    # SS-BENCH-GATE-b: --allow-during-bench flows into the per-spawn placement
    # guard (the CLI-level guard already exited 2 without it when a bench runs).
    bench_force = bool(getattr(args, "allow_during_bench", False))
    _registry_yaml = Path(__file__).parent.parent.parent / "orchestration" / "model_registry.yaml"
    _descriptor_yaml = (
        Path(__file__).parent.parent.parent / "orchestration" / "model_descriptors.yaml"
    )
    _master_registry = Path(
        "/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml"
    )
    _cache_key_path = _registry_yaml.parent / ".lean_cache_key"

    # 2026-06-27: lean-registry compile is default-on for normal starts.
    # Regenerates orchestration/model_registry.yaml from
    # the master at epyc-inference-research/orchestration/model_registry.yaml,
    # filtered to active roles per ROLE_LAUNCH_META + their transitive
    # draft/alias deps. Cache-keyed by SHA-256 of (master content + active
    # role names) — re-runs without changes are no-ops.
    #
    # Escape hatches: pass --no-compile-registry or set
    # ORCHESTRATOR_REGISTRY_NO_COMPILE=1 to bypass.
    if getattr(args, "compile_registry", True):
        try:
            from src.registry.registry_compiler import (
                active_roles_from_launch_meta,
                load_or_compile,
            )

            active_roles = active_roles_from_launch_meta(ROLE_LAUNCH_META)
            print(f"[registry-compile] master={_master_registry}")
            print(f"[registry-compile] active_roles={sorted(active_roles)}")
            load_or_compile(
                master_path=_master_registry,
                active_roles=active_roles,
                output_path=_registry_yaml,
                cache_key_path=_cache_key_path,
            )
            print(f"[registry-compile] OK — wrote {_registry_yaml}")
        except Exception as exc:  # noqa: BLE001
            print(f"[registry-compile] FATAL: {exc}")
            return 2

    # 2026-05-09: registry consistency gate — runs first, fails fast on cross-section
    # acceleration disagreements, GGUF/port inconsistencies, or duplicate YAML keys.
    # Catches the failure modes that wasted ~2 hours debugging architect_general on
    # 2026-05-09 (server_mode.X.acceleration silently overridden by roles.X.acceleration).
    try:
        from src.registry.registry_validator import validate_or_raise, RegistryValidationError

        try:
            validate_or_raise(_registry_yaml)
        except RegistryValidationError as exc:
            print("[registry-validator] FATAL — refusing to start a stack on a broken registry:")
            print(f"  {exc}")
            print("  Fix the registry, then re-run.")
            return 2
    except ImportError:
        # Validator module not present (older deployment) — proceed without gate.
        # Once landed everywhere, drop this fallback.
        pass

    if getattr(args, "compile_descriptors", False):
        try:
            from src.registry.model_descriptors import write_model_descriptors

            active_roles = _descriptor_active_roles()
            allow_incomplete = bool(getattr(args, "allow_incomplete_descriptors", False))
            print(f"[descriptor-compile] lean={_registry_yaml}")
            print(f"[descriptor-compile] research={_master_registry}")
            print(f"[descriptor-compile] active_roles={sorted(active_roles)}")
            write_model_descriptors(
                _descriptor_yaml,
                lean_registry_path=_registry_yaml,
                research_registry_path=_master_registry,
                active_roles=active_roles,
                allow_incomplete=allow_incomplete,
            )
            print(f"[descriptor-compile] OK — wrote {_descriptor_yaml}")
        except Exception as exc:  # noqa: BLE001
            print(f"[descriptor-compile] FATAL: {exc}")
            return 2

    if not _run_stack_change_launch_gate(args):
        return 2

    # DS-7 / NIB2-19: --migrate-to handler (runs before any start path)
    migrate_to = getattr(args, "migrate_to", None)
    if migrate_to:
        dry_run = getattr(args, "dry_run", False)
        try:
            from src.config.stack_migration import migrate_to_template
        except Exception as exc:  # noqa: BLE001
            print(f"[DS-7] Migration module unavailable: {exc}")
            return 1
        print(
            f"[DS-7] Migrating stack → template '{migrate_to}' ({'DRY-RUN' if dry_run else 'LIVE'})"
        )
        registry_path = (
            Path(_PATHS.get("model_registry", "")) if _PATHS.get("model_registry") else None
        )
        result = migrate_to_template(migrate_to, dry_run=dry_run, registry_path=registry_path)
        print(result.summary())
        return 0 if result.ok else 1

    # DS-7: Stack template validation (before any other work)
    stack_profile = getattr(args, "stack_profile", None)
    validate_only = getattr(args, "validate_only", False)
    if stack_profile:
        try:
            from src.config.stack_templates import (
                load_template,
                validate_template,
            )

            print(f"[DS-7] Loading stack template: {stack_profile}")
            template = load_template(stack_profile)
            print(f"  Name: {template.name}")
            print(f"  Description: {template.description}")
            print(f"  Roles: {len(template.roles)} ({', '.join(template.role_names())})")
            print(f"  Instances: {template.total_instances}")
            print(f"  RAM: {template.total_ram_gb:.0f} GB")
            print()

            registry_path = (
                Path(_PATHS.get("model_registry", "")) if _PATHS.get("model_registry") else None
            )
            result = validate_template(template, registry_path)
            if result.errors:
                print(f"  [FAIL] {len(result.errors)} validation errors:")
                for err in result.errors:
                    print(f"    ERROR: {err}")
            if result.warnings:
                for warn in result.warnings:
                    print(f"    WARN: {warn}")
            if result.valid:
                print("  [OK] Template valid")
            else:
                print("\n  Template validation failed. Fix errors and retry.")
                return 1

            if validate_only:
                print("\n--validate-only: exiting after validation.")
                return 0

            print(
                "  (Template loaded but not yet used for server launch — "
                "integration pending DS-7 Phase 2)"
            )
            print()
        except FileNotFoundError as exc:
            print(f"[DS-7] ERROR: {exc}")
            return 1
        except Exception as exc:
            print(f"[DS-7] Template load error: {exc}")
            return 1

    print("=" * 60)
    print("ORCHESTRATOR STACK STARTUP")
    print("=" * 60)
    print()

    # Host prerequisites — applied before any llama-server launch.
    # See cpu-kernel-env-flags-inventory.md §211 + model-registry-v5-deployment-draft.yaml.
    skip_host_prereqs = getattr(args, "skip_host_prereqs", False)
    if skip_host_prereqs:
        print("[host_prereq] SKIPPED (--skip-host-prereqs). Canonical state NOT enforced.")
    else:
        if not apply_host_prerequisites(auto_fix=True):
            print("[!] Host prerequisites could not be applied. Refusing to launch.")
            print("    Override with --skip-host-prereqs (NOT recommended for benchmarks).")
            return 1
    print()

    # Check memory
    free_gb = check_free_memory()
    print(f"[i] Free memory: {free_gb} GB")
    if free_gb < 100 and not args.dev:
        print("[!] WARNING: Less than 100GB free. Consider --dev mode.")
        if input("Continue? (y/N) ").lower() != "y":
            return 1
    print()

    # Load registry
    registry = RegistryLoader()
    # In --only mode we start a SUBSET of roles, so seed `state` from the
    # existing on-disk state and merge — otherwise save_state() at the end
    # clobbers every other still-running role's entry, leaving live processes
    # untracked (stop/reload can no longer find them). A full start (no --only)
    # rebuilds from scratch, which is correct since it (re)launches everything.
    state: dict[str, ProcessInfo] = load_state() if getattr(args, "only", None) else {}

    # Validate model paths (prevents hallucinations about missing models)
    if not args.dev:
        print("[0.5] Validating model paths...")
        errors = validate_model_paths()
        if errors:
            print("[!] MODEL VALIDATION FAILED:")
            for err in errors:
                print(f"    - {err}")
            print("\nFix missing models or update paths in orchestrator_stack.py")
            print(f"Check {_PATHS['models_dir']} and {_PATHS['model_base']}")
            return 1
        print("  [OK] All model paths validated")
        print()

    # Cross-check launcher classification vs registry process_layout / server_mode.
    # Non-fatal: prints warnings but does not abort. Useful for catching drift
    # between the launcher's ROLE_LAUNCH_META and the registry's source-of-truth
    # process_layout section.
    if not args.dev:
        registry_warnings = validate_against_registry()
        if registry_warnings:
            print("[0] Registry classification warnings:")
            for w in registry_warnings:
                print(f"  ⚠ {w}")
            print()

    # [0.7] Episodic embedding health check (A3, 2026-05-21).
    # Detects the orphan-FAISS condition where the live episodic.db has many more
    # routing memories than FAISS vectors — e.g. after a FAISS reset or a BGE
    # outage during write. KNN fallback (DAR-2 contrastive sharpening, low-confidence
    # routing fallback) silently degrades only when this live FAISS/id_map contract fails.
    #
    # Read-only by default — does not block startup. To repair, run:
    #   python3 scripts/maintenance/repair_episodic_embeddings.py --repair
    # Or pass --repair-embeddings to this command for auto-repair before launch.
    if not args.dev:
        try:
            from scripts.maintenance.repair_episodic_embeddings import (
                diagnose as _diagnose_embeddings,
                print_report as _print_embedding_report,
                run_repair as _run_embedding_repair,
                DEFAULT_DB_PATH,
                DEFAULT_EMBEDDER_BASE_PORT,
                DEFAULT_EMBEDDER_SERVERS,
                DEFAULT_FAISS_PATH,
                DEFAULT_ID_MAP_PATH,
                DEFAULT_MAX_DB_GROWTH,
                DEFAULT_REEMBEDDED_PATH,
            )

            print("[0.7] Episodic embedding health check...")
            _report = _diagnose_embeddings(
                DEFAULT_DB_PATH,
                DEFAULT_FAISS_PATH,
                DEFAULT_REEMBEDDED_PATH,
            )
            _print_embedding_report(_report)
            if not _report.healthy:
                if getattr(args, "repair_embeddings", False):
                    print("\n[0.7] --repair-embeddings: starting bulk re-embed + FAISS rebuild...")
                    print(
                        "      "
                        f"(uses {DEFAULT_EMBEDDER_SERVERS} configured BGE server(s) "
                        f"starting at port {DEFAULT_EMBEDDER_BASE_PORT}; expected 5-15 min)"
                    )
                    _run_embedding_repair(
                        db_path=DEFAULT_DB_PATH,
                        faiss_path=DEFAULT_FAISS_PATH,
                        id_map_path=DEFAULT_ID_MAP_PATH,
                        reembedded_path=DEFAULT_REEMBEDDED_PATH,
                        servers=DEFAULT_EMBEDDER_SERVERS,
                        base_port=DEFAULT_EMBEDDER_BASE_PORT,
                        max_db_growth=DEFAULT_MAX_DB_GROWTH,
                    )
                    print("[0.7] Re-running diagnostic post-repair:")
                    _report2 = _diagnose_embeddings(
                        DEFAULT_DB_PATH,
                        DEFAULT_FAISS_PATH,
                        DEFAULT_REEMBEDDED_PATH,
                    )
                    _print_embedding_report(_report2)
                    if not _report2.healthy:
                        print("[!] WARNING: repair did not restore health — proceeding anyway.")
                else:
                    print(
                        "[!] Live episodic FAISS/id_map store is ORPHANED. KNN fallback may degrade."
                    )
                    print(
                        "    Repair: python3 scripts/maintenance/repair_episodic_embeddings.py --repair"
                    )
                    print("    Or re-run with --repair-embeddings to auto-repair.")
            print()
        except ImportError as exc:
            # Maintenance script not present (older deployment) — proceed without check.
            print(f"[0.7] Skipping embedding health check (maintenance module unavailable: {exc})")
            print()
        except Exception as exc:
            # Read-only diagnostic should never raise. If it does, log and continue.
            print(f"[0.7] Embedding health check failed: {exc} (proceeding anyway)")
            print()

    # Determine which servers to start
    servers_to_start = []

    if args.dev:
        print("[1] Starting in DEV mode (single 0.5B model)...")
        servers_to_start = [{"port": 8080, "roles": ["dev"]}]
    elif args.only:
        # --only: start ONLY the specified roles, nothing else
        requested = set(args.only)
        print(f"[1] Selective start: {', '.join(sorted(requested))}")
        for server in HOT_SERVERS + WARM_SERVERS:
            if requested & set(server["roles"]):
                servers_to_start.append(server)
                print(f"  Including: port {server['port']} ({', '.join(server['roles'])})")
        if not servers_to_start:
            print(f"  [!] No servers matched roles: {', '.join(sorted(requested))}")
            print(
                f"  Available roles: {', '.join(sorted({r for s in HOT_SERVERS + WARM_SERVERS for r in s['roles']}))}"
            )
            return 1
    else:
        print("[1] Starting HOT servers...")
        servers_to_start = HOT_SERVERS.copy()

        # Add warm servers if requested
        if args.include_warm:
            for warm_server in WARM_SERVERS:
                for role in warm_server["roles"]:
                    if role in args.include_warm:
                        servers_to_start.append(warm_server)
                        print(f"  Including WARM server: port {warm_server['port']} ({role})")
                        break

    # Apply --numa-mode filter. Picks full XOR sub-full siblings for any role
    # with full_instance_idx + multiple instances (currently frontdoor +
    # worker_general + ingest_long_context); single-instance roles pass through.
    # See launcher-numa-mode-gating handoff.
    #
    # ESC-8 Fix 4: default is None (no hardcoded 'full'). When unset, INFER the
    # mode from the running fleet; with no live fleet (cold start), fall back to
    # ORCHESTRATOR_STACK_NUMA_MODE, then to the ratified production mode 'both'
    # (full + half instances; quarters were retired 2026-07-30). A
    # `start --only <role>` that conflicts with the live fleet is refused so it
    # cannot stamp overlapping full instances next to live halves (kill chain A2).
    numa_mode = getattr(args, "numa_mode", None)
    only = getattr(args, "only", None)
    realized_mode: str | None = None
    if numa_mode is None or only:
        try:
            from scripts.server.realized_fleet import derive_realized_numa_mode

            realized_mode = derive_realized_numa_mode()
        except Exception as exc:  # noqa: BLE001
            print(f"  [--numa-mode] WARN: realized-fleet probe failed ({exc})")
    if numa_mode is None:
        if realized_mode:
            numa_mode = realized_mode
            print(f"  [--numa-mode] not specified; inferred '{numa_mode}' from the running fleet")
        else:
            from scripts.server.stack_numa_mode import env_stack_numa_mode

            numa_mode = env_stack_numa_mode(default="both")
            print(
                f"  [--numa-mode] not specified and no live fleet detected; "
                f"defaulting to production mode '{numa_mode}'"
            )
    if only and realized_mode is not None and numa_mode != realized_mode:
        if _only_mode_transition_allowed(numa_mode, realized_mode):
            print(
                f"  [--numa-mode both] additive promotion over realized "
                f"'{realized_mode}' fleet: healthy servers are kept; only the "
                f"missing complementary instances will be started."
            )
        else:
            print(
                f"  [!] --only refused: requested --numa-mode '{numa_mode}' conflicts with the "
                f"running fleet ('{realized_mode}'). Pass --numa-mode {realized_mode} explicitly "
                f"to add roles to the live stack, use an explicit '--numa-mode both' to "
                f"additively promote a single-mode fleet, or stop the fleet first."
            )
            return 1
    if numa_mode == "both":
        # Light advisory only — 'both' has been working for frontdoor/coder_escalation since
        # 2026-03 (Qwen3.6-35B Q8 quarters tuned to coexist with the full instance). The
        # gemma4-MTP exception is the one that needs --numa-mode full per role. We don't
        # spam at every start since most roles are fine.
        if any("worker_general" in s.get("roles", []) for s in servers_to_start):
            print(
                "  [advisory] worker_general (gemma4-MTP) runs at -t 96; if its full + 4 quarters "
                "are all kept (--numa-mode both), expect 1.5x CPU oversubscription. "
                "Use '--numa-mode full' (single instance) or '--numa-mode quarter' (4 concurrent) "
                "for that role specifically. See launcher-numa-mode-gating.md."
            )
    pre_filter_count = len(servers_to_start)
    servers_to_start = _filter_by_numa_mode(servers_to_start, numa_mode)
    if numa_mode != "both" and len(servers_to_start) != pre_filter_count:
        dropped = pre_filter_count - len(servers_to_start)
        print(
            f"  [--numa-mode={numa_mode}] dropped {dropped} overlapping instance(s); "
            f"{len(servers_to_start)} server(s) to start"
        )

    print()

    # [1.5] Page-cache prewarm — distribute shared-GGUF pages across NUMA nodes
    # under `numactl --interleave=all` BEFORE the per-instance launches mlock
    # them. Otherwise sequential mlock pins every page of a shared model onto
    # whichever node the first launcher's --membind targeted, and remote-node
    # quarters cross-socket-fetch for the whole stack lifetime (~50-65% t/s drop
    # observed 2026-05-28 after a container rebuild evicted the page cache).
    # See handoffs/active/numa-page-cache-prewarm.md.
    _prewarm_all(
        servers_to_start,
        _orchestrator_stack().build_server_command,
        registry,
        args=args,
    )
    print()

    # Check target ports — skip healthy, clean up unhealthy
    print("[2] Checking target ports...")
    already_healthy_ports: set[int] = set()
    for server in servers_to_start:
        port = server["port"]
        if is_port_in_use(port):
            if wait_for_health(port, timeout=3):
                print(f"  Port {port} already healthy, skipping")
                already_healthy_ports.add(port)
                continue
            print(f"  Port {port} in use but unhealthy, cleaning up...")
            try:
                for pid in _pids_on_port(port):
                    kill_process(pid)
            except Exception as e:
                print(f"  [!] Error cleaning port {port}: {e}")
    if already_healthy_ports:
        print(f"  Preserved {len(already_healthy_ports)} healthy server(s)")

    print()

    # Start servers sequentially (skip already-healthy ports)
    print("[3] Starting llama-servers...")
    for i, server in enumerate(servers_to_start):
        port = server["port"]
        roles = server["roles"]

        if port in already_healthy_ports:
            role_label = roles[0] if roles else str(port)
            print(f"  Skipping port {port}: {role_label} (already healthy)")
            # Record existing server in state so status reporting works
            preserved = _preserved_process_info(
                role_label,
                port,
                f"preserved:{role_label}",
                str(LOG_DIR / f"llama-server-{port}.log"),
            )
            if preserved:
                state[f"server_{port}"] = preserved
                for role in roles:
                    if role not in state:
                        state[role] = preserved
            continue

        embedding_mode = server.get("embedding", False)
        worker_pool_mode = server.get("worker_pool", False)
        worker_type = server.get("worker_type")
        vision_mode = server.get("vision", False)
        vision_type = server.get("vision_type")
        eval_batch_frontdoor_mode = server.get("eval_batch_frontdoor", False)
        # P2-6/P0-2: flag set only by a ROLE_LAUNCH_META entry with mode
        # "gpu_shadow_lane" — absent until the lane proposal is applied.
        gpu_shadow_lane_mode = server.get("gpu_shadow_lane", False)
        numa_instance = server.get("numa_instance", 0)

        info = start_server(
            port,
            roles,
            registry,
            args.dev,
            embedding_mode=embedding_mode,
            worker_pool_mode=worker_pool_mode,
            worker_type=worker_type,
            vision_mode=vision_mode,
            vision_type=vision_type,
            eval_batch_frontdoor_mode=eval_batch_frontdoor_mode,
            gpu_shadow_lane_mode=gpu_shadow_lane_mode,
            numa_instance=numa_instance,
            bench_force=bench_force,
        )
        if info:
            state[f"server_{port}"] = info
            # Also map all roles to this server
            for role in roles:
                if role not in state:
                    state[role] = info
        else:
            print(f"  [!] Failed to start server on port {port}")
            # Embedding/worker_pool/vision server failure is non-fatal (fallback available)
            is_optional = (
                embedding_mode
                or worker_pool_mode
                or vision_mode
                or eval_batch_frontdoor_mode
                or gpu_shadow_lane_mode
            )
            if not args.dev and not is_optional:
                return 1

        # Sequential loading: wait for this server to be healthy before launching
        # the next one. Concurrent mlock on large models causes crashes even when
        # total RAM is sufficient (race condition during page fault + lock).
        is_small_model = (
            embedding_mode
            or (worker_pool_mode and worker_type == "fast")
            or (vision_mode and vision_type != "escalation")
        )
        if i < len(servers_to_start) - 1 and not args.dev and not is_small_model:
            if not wait_for_health(port, timeout=300):
                print(f"  [!] Server on port {port} did not become healthy within 300s")
            else:
                print(f"  Server on port {port} healthy, proceeding to next")

    print()

    # Start orchestrator (skip if already healthy, or if --only was used for model servers)
    if args.only:
        print("[4] Skipping orchestrator API (--only mode)")
        if wait_for_health(8000, timeout=2):
            print("  Orchestrator already healthy")
            preserved = _preserved_process_info(
                "orchestrator", 8000, "uvicorn", str(LOG_DIR / "orchestrator.log")
            )
            if preserved:
                state["orchestrator"] = preserved
        else:
            print("  [i] Orchestrator not running — start separately if needed")
    elif 8000 in already_healthy_ports:
        print("[4] Starting orchestrator API...")
        print("  Orchestrator already healthy, skipping")
        preserved = _preserved_process_info(
            "orchestrator", 8000, "uvicorn", str(LOG_DIR / "orchestrator.log")
        )
        if preserved:
            state["orchestrator"] = preserved
    else:
        info = start_orchestrator(
            getattr(args, "profile", None),
            stack_numa_mode=numa_mode,
            bench_force=bench_force,
        )
        if info:
            state["orchestrator"] = info
        else:
            print("  [!] Failed to start orchestrator")
            return 1

    print()

    # Start every declared auxiliary service (optional, non-fatal).
    #
    # 2026-08-02 (W5/W4): was four hand-written blocks, one per service, each
    # restating the same already-healthy-skip / start / record shape with its own
    # literal port, health path and label. That duplication is why registering a
    # service meant editing several places and why two of them were missed in the
    # reload path. The loop below iterates AUX_SERVICES, so declaring a service in
    # launch_manifest.yaml is now the ONLY step required to have it started,
    # reloaded and reported.
    if not args.dev and not args.only:
        for index, service in enumerate(AUX_SERVICES.values()):
            step = f"[5{'' if index == 0 else chr(ord('a') + index - 1)}]"
            print(f"{step} Starting {service.name} ({service.description or service.name})...")
            if is_port_in_use(service.port) and wait_for_health(
                service.port, timeout=3, path=service.health_path
            ):
                print("  Already healthy, skipping")
                preserved = _preserved_process_info(
                    service.name,
                    service.port,
                    service.model_label,
                    str(LOG_DIR / service.log),
                )
                if preserved:
                    state[service.name] = preserved
            else:
                starter = globals().get(f"start_{service.name}")
                info = (
                    starter(bench_force=bench_force)
                    if callable(starter)
                    else start_aux_service(service.name, bench_force=bench_force)
                )
                if info:
                    state[service.name] = info
                else:
                    print(f"  [!] {service.name} failed (non-fatal, continuing)")
            print()

        # Start Docker services (NextPLAID retrieval + SearXNG metasearch)
        if _docker_available():
            print("[5.5] Starting Docker services (NextPLAID retrieval + SearXNG metasearch)...")
            for service in DOCKER_SERVICES:
                info = start_docker_container(service)
                if info:
                    state[service["name"]] = info
                else:
                    svc_name = service["name"]
                    if svc_name == "searxng":
                        print(
                            f"  [!] {svc_name} failed (non-fatal, web_search falls back to DDG HTML scraping)"
                        )
                    else:
                        print(
                            f"  [!] {svc_name} failed (non-fatal, code_search degrades gracefully)"
                        )
            print()
        else:
            print("[5.5] Docker not available, skipping Docker containers")
            print("  code_search/doc_search will be unavailable")
            print("  web_search will use DDG HTML scraping fallback")
            print()

        # Initialize MemRL databases and tool registry
        init_memrl_and_tools()

        print()

    # Save state
    # ESC-8 Fix 2-addendum: capture the persisted fleet BEFORE save_state()
    # overwrites the file, then feed the runtime-facts refresh the MERGED view so
    # llama rows this invocation did not re-touch (a subset/--only start) are not
    # dropped, which otherwise emits selected_servers: [] (the 09:14 defect).
    _persisted_state_before_save = load_state()
    save_state(state)
    print(f"[i] State saved to {STATE_FILE}")
    _refresh_runtime_facts_manifest(
        "stack_start",
        _merge_persisted_state_for_facts(_persisted_state_before_save, state),
        stack_numa_mode=numa_mode,
    )
    print()

    # Final status
    print("=" * 60)
    print("STACK READY")
    print("=" * 60)
    cmd_status(args)

    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop components."""
    state = load_state()

    if not state and args.all:
        # State file empty — fall back to port scanning
        found = _scan_known_ports()
        if not found:
            print("No running components found")
            return 0

        print(f"State file empty but found processes on {len(found)} ports (port scan fallback)")
        killed = 0
        for port, pids in sorted(found.items()):
            for pid in pids:
                print(f"  Stopping PID {pid} on port {port}...")
                if kill_process(pid):
                    print("    [OK] Stopped")
                    killed += 1
                else:
                    print("    [!] Failed to stop")
        print(f"Stopped {killed} orphaned processes")
        save_state({})
        # ESC-8 Fix 2: pass the realized mode explicitly (empty fleet ⇒ None ⇒
        # serialized null, fail-safe) rather than letting the writer coerce a
        # missing mode to a poisoned "full".
        _refresh_runtime_facts_manifest(
            "stack_stop", {}, stack_numa_mode=realized_stack_numa_mode_from_state({})
        )
        return 0

    if not state:
        print("No running components found")
        return 0

    targets = []
    if args.all:
        targets = list(state.keys())
    elif args.components:
        targets = args.components
    else:
        print("Specify --all or component names")
        return 1

    for name in targets:
        if name in state:
            info = state[name]
            if info.pid == -1:
                # Docker-managed container
                print(f"Stopping Docker container {name}...")
                if stop_docker_container(info.role):
                    del state[name]
                    print("  [OK] Stopped")
                else:
                    print(f"  [!] Failed to stop container {name}")
            else:
                print(f"Stopping {name} (PID {info.pid})...")
                if kill_process(info.pid):
                    del state[name]
                    print("  [OK] Stopped")
                else:
                    print("  [!] Failed to stop")
        else:
            print(f"  [?] {name} not found in state")

    save_state(state)

    # After state-based stop, scan for orphans that survived
    if args.all:
        orphans = _scan_known_ports()
        if orphans:
            print(
                f"\nFound {sum(len(p) for p in orphans.values())} orphaned processes on {len(orphans)} ports"
            )
            for port, pids in sorted(orphans.items()):
                for pid in pids:
                    print(f"  Stopping orphan PID {pid} on port {port}...")
                    if kill_process(pid):
                        print("    [OK] Stopped")
                    else:
                        print("    [!] Failed to stop")

    # ESC-8 Fix 2: record the realized mode derived from the surviving fleet.
    _refresh_runtime_facts_manifest(
        "stack_stop", state, stack_numa_mode=realized_stack_numa_mode_from_state(state)
    )
    return 0


def _reload_server_port(component: str) -> int | None:
    """Resolve a reload component to a managed llama-server port.

    Two spellings resolve: a role name from PORT_MAP, and the state-file key
    form ``server_<port>`` for any port carried by a HOT_SERVERS/WARM_SERVERS
    entry. The second form is the ONLY one that can address a sub-full
    instance: PORT_MAP maps a role to its primary port (frontdoor -> 8070), so
    ``reload server_8080`` used to fall through every branch to "[?] Unknown
    component", restart nothing, and still exit 0 — while the stale half kept
    answering /health on its old cmdline. A declared serving-flag change
    (e.g. chat_template_file) was then undeployable to sub-full instances.
    Returns None when the component is not a managed llama-server; the caller's
    later branches (Docker services, unknown) then apply.
    """
    if component in PORT_MAP:
        return PORT_MAP[component]
    if component.startswith("server_"):
        suffix = component[len("server_"):]
        if suffix.isdigit():
            port = int(suffix)
            if any(server["port"] == port for server in HOT_SERVERS + WARM_SERVERS):
                return port
    return None


def cmd_reload(args: argparse.Namespace) -> int:
    """Reload components."""
    # SS-BENCH-GATE-b: same force plumbing as cmd_start — the reload that
    # spawned the incident's sidecar must honor --allow-during-bench per-spawn.
    bench_force = bool(getattr(args, "allow_during_bench", False))
    state = load_state()
    registry: RegistryLoader | None = None

    def get_registry() -> RegistryLoader:
        nonlocal registry
        if registry is None:
            registry = RegistryLoader()
        return registry

    # UFH-12 arm A3: a one-shot, recorded embedder env override (embedder_env_override.py).
    from scripts.server import embedder_env_override as _emb_override

    override_items = list(getattr(args, "embedder_env_override", None) or [])
    experiment_id = getattr(args, "experiment_id", None)
    try:
        env_override = _emb_override.parse_overrides(override_items)
    except _emb_override.OverrideError as exc:
        print(f"[REFUSED] {exc}")
        return 2
    if env_override:
        if list(args.components) != ["embedders"]:
            print("[REFUSED] --embedder-env-override applies only to `reload embedders` on its own "
                  f"(got components {list(args.components)})")
            return 2
        if not (experiment_id and str(experiment_id).strip()):
            print("[REFUSED] --embedder-env-override needs --experiment-id")
            return 2
        ttl_s = float(getattr(args, "override_ttl_s", _emb_override.DEFAULT_TTL_S))
        if not 0 < ttl_s <= _emb_override.MAX_TTL_S:
            print(f"[REFUSED] --override-ttl-s must be in (0, {_emb_override.MAX_TTL_S:.0f}]")
            return 2
    elif experiment_id:
        print("[REFUSED] --experiment-id without --embedder-env-override records nothing")
        return 2

    for component in args.components:
        print(f"Reloading {component}...")

        # Special case: reload all embedders at once
        if component == "embedders":
            print("  Reloading all 6 BGE embedder instances...")
            if env_override:
                print(f"  [EXPERIMENT {experiment_id}] env override {env_override} "
                      f"(expires in {ttl_s:.0f}s; restore = plain `reload embedders`)")

            # Kill by state file entries
            for port in EMBEDDER_PORTS:
                key = f"server_{port}"
                role = "embedder" if port == 8090 else f"embedder_{port - 8090}"
                if key in state:
                    kill_process(state[key].pid)
                    del state[key]
                if role in state:
                    del state[role]

            # Also kill by port (in case state is stale)
            for port in EMBEDDER_PORTS:
                if is_port_in_use(port):
                    try:
                        for pid in _pids_on_port(port):
                            kill_process(pid)
                            print(f"    Killed stale process on port {port}")
                    except (subprocess.TimeoutExpired, OSError, ValueError):
                        pass  # Best-effort stale process cleanup

            time.sleep(2)  # Wait for ports to free

            # Start all embedders
            success_count = 0
            launched_pids: dict[int, int] = {}
            for port in EMBEDDER_PORTS:
                role = "embedder" if port == 8090 else f"embedder_{port - 8090}"
                info = start_server(
                    port,
                    [role],
                    get_registry(),
                    dev_mode=False,
                    embedding_mode=True,
                    bench_force=bench_force,
                    **({"env_override": env_override} if env_override else {}),
                )
                if info:
                    state[f"server_{port}"] = info
                    state[role] = info
                    success_count += 1
                    launched_pids[int(port)] = int(info.pid)

            print(f"  [OK] {success_count}/{len(EMBEDDER_PORTS)} embedders restarted")
            if env_override:
                # Record even a partial launch: every pid that got the override deviates.
                if launched_pids:
                    rec = _emb_override.write_active(
                        experiment_id=str(experiment_id), env=env_override, pids=launched_pids,
                        ttl_s=ttl_s, argv=list(sys.argv))
                    print(f"  [EXPERIMENT] recorded {_emb_override.record_path()} "
                          f"(expires_at {rec['expires_at']})")
            else:
                # A plain reload IS the restore: archive any recorded override with a readback.
                cleared = _emb_override.clear(
                    reason="restored by plain `reload embedders`",
                    readback_after=_emb_override.readback(launched_pids))
                if cleared is not None:
                    print("  [EXPERIMENT] override record cleared "
                          f"({(cleared.get('record') or {}).get('experiment_id')}); "
                          f"archived to {_emb_override.history_path()}")
            if success_count == 0:
                return 1
            continue

        elif component == "orchestrator":
            # Stop by authoritative listener port only.
            # State-file PIDs can go stale and be reused by unrelated processes.
            for pid in _pids_on_port(8000):
                kill_process(pid)
            time.sleep(1)

            # Start new
            info = start_orchestrator(
                getattr(args, "profile", None),
                stack_numa_mode=read_runtime_stack_numa_mode(),
                bench_force=bench_force,
            )
            if info:
                state["orchestrator"] = info
            else:
                print("  [!] Failed to restart orchestrator")
                return 1

        elif component in AUX_SERVICES:
            # Auxiliary services: do NOT route through start_server()/RegistryLoader.
            # None of them is a llama-server registry role.
            #
            # 2026-08-02 (W5) — THIS BRANCH MUST PRECEDE THE `component in PORT_MAP`
            # BRANCH BELOW, and that ordering is the whole fix.
            #
            # Before today this was a hand-written chain that named only
            # document_formalizer and handoff_dashboard. whisper (9000) and
            # sd_server (8190) are in PORT_MAP but in neither HOT_SERVERS nor
            # WARM_SERVERS, so `reload whisper` fell through to the PORT_MAP branch,
            # which killed every listener on the port and then called start_server()
            # — the llama-server path — with roles=["whisper"]. There is no such
            # registry role, so the start could only fail. Net effect: the service
            # was killed, never restarted, and the command returned 1 having left
            # STT down. Same for sd_server.
            #
            # Dispatching off the declared registry (rather than a chain a human
            # must remember to extend) is what stops the next registered service
            # from inheriting the defect.
            service = AUX_SERVICES[component]
            for pid in _pids_on_port(service.port):
                kill_process(pid)
            state.pop(component, None)
            time.sleep(1)

            # Prefer the service's named wrapper when one exists, else the generic
            # launcher. Derived from the name, not a second hand-kept table — a
            # newly declared service works with no wrapper at all, while the
            # existing public `start_<name>()` entry points (and every caller and
            # test bound to them) keep their exact meaning.
            starter = globals().get(f"start_{component}")
            info = (
                starter(bench_force=bench_force)
                if callable(starter)
                else start_aux_service(component, bench_force=bench_force)
            )
            if info:
                state[component] = info
            else:
                print(f"  [!] Failed to restart {component}")
                return 1

        elif (server_port := _reload_server_port(component)) is not None:
            port = server_port
            key = f"server_{port}"

            # Find roles and config for this port
            roles = [component]
            worker_pool_mode = False
            worker_type = None
            embedding_mode = False
            vision_mode = False
            vision_type = None
            eval_batch_frontdoor_mode = False
            # P2-6/P0-2: set only by a gpu_shadow_lane-mode server entry —
            # absent until the lane proposal is applied.
            gpu_shadow_lane_mode = False

            numa_instance = 0
            for server in HOT_SERVERS + WARM_SERVERS:
                if server["port"] == port:
                    roles = server["roles"]
                    worker_pool_mode = server.get("worker_pool", False)
                    worker_type = server.get("worker_type")
                    embedding_mode = server.get("embedding", False)
                    vision_mode = server.get("vision", False)
                    vision_type = server.get("vision_type")
                    eval_batch_frontdoor_mode = server.get("eval_batch_frontdoor", False)
                    gpu_shadow_lane_mode = server.get("gpu_shadow_lane", False)
                    numa_instance = server.get(
                        "numa_instance", 0
                    )  # fix: reload must preserve per-quarter -t
                    break

            # Stop existing
            # Stop by authoritative listener port only.
            # State-file PIDs can go stale and be reused by unrelated processes.
            for pid in _pids_on_port(port):
                kill_process(pid)
            time.sleep(1)

            # Start new
            info = start_server(
                port,
                roles,
                get_registry(),
                dev_mode=False,
                embedding_mode=embedding_mode,
                worker_pool_mode=worker_pool_mode,
                worker_type=worker_type,
                vision_mode=vision_mode,
                vision_type=vision_type,
                eval_batch_frontdoor_mode=eval_batch_frontdoor_mode,
                gpu_shadow_lane_mode=gpu_shadow_lane_mode,
                numa_instance=numa_instance,
                bench_force=bench_force,
            )
            if info:
                state[key] = info
                for role in roles:
                    state[role] = info
            else:
                print(f"  [!] Failed to restart {component}")
                return 1

        else:
            # Check if it's a Docker service
            docker_service = None
            for svc in DOCKER_SERVICES:
                if component == svc["name"]:
                    docker_service = svc
                    break

            if docker_service:
                print(f"  Reloading Docker service {component}...")
                stop_docker_container(component)
                time.sleep(2)
                info = start_docker_container(docker_service)
                if info:
                    state[component] = info
                else:
                    print(f"  [!] Failed to restart {component}")
                    return 1
            else:
                # Fail loudly: exiting 0 here is how a no-op reload passes for a
                # deploy — the old process keeps answering /health while the
                # caller believes it relaunched with the new configuration.
                print(f"  [?] Unknown component: {component}")
                return 1

    save_state(state)
    # ESC-8 Fix 2: record the realized mode so the next reload reads a manifest
    # that matches the live fleet instead of carrying a poisoned "full" forward.
    _refresh_runtime_facts_manifest(
        "stack_reload", state, stack_numa_mode=realized_stack_numa_mode_from_state(state)
    )
    return 0


def _declared_non_optional_services() -> list[tuple[str, int, str]]:
    """Return active manifest services that must be visible even without state.

    Stack priors define the active serving topology.  Docker services are also
    manifest-declared, and unlike the explicit optional auxiliary set they must
    be surfaced when absent.  Group aliases by port so shared servers produce
    one status row.
    """
    names_by_port: dict[int, set[str]] = {}
    health_paths: dict[int, str] = {}

    for role, record in live_stack_role_records().items():
        for port in stack_prior_serving_ports(stack_prior_serving(record)):
            names_by_port.setdefault(port, set()).add(role)
            health_paths.setdefault(port, "/health")

    for service in DOCKER_SERVICES:
        name = service.get("name")
        port = service.get("port")
        if (
            not isinstance(name, str)
            or not isinstance(port, int)
            or name in OPTIONAL_AUXILIARY_ROLES
        ):
            continue
        names_by_port.setdefault(port, set()).add(name)
        health_path = service.get("health_path", "/health")
        health_paths[port] = health_path if isinstance(health_path, str) else "/health"

    return [
        ("/".join(sorted(names)), port, health_paths[port])
        for port, names in sorted(names_by_port.items())
    ]


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of all components."""
    state = load_state()
    state_roles = {info.role for info in state.values()}
    unavailable_optional_roles = sorted(OPTIONAL_AUXILIARY_ROLES - state_roles)
    declared_services = _declared_non_optional_services()
    state_ports = {info.port for info in state.values()}
    missing_declared_services = [
        service for service in declared_services if service[1] not in state_ports
    ]

    if not state and not unavailable_optional_roles and not missing_declared_services:
        print("No components running")
        return 0

    print()
    print(f"{'COMPONENT':<25} {'PORT':<8} {'PID':<10} {'STATUS':<10} {'ATTEST':<12} {'MODEL'}")
    print("-" * 96)

    seen_pids = set()
    attestation_warnings: list[str] = []
    contracts_by_role = _stack_prior_launch_contracts()
    for name, info in sorted(state.items()):
        if info.pid != -1 and info.pid in seen_pids:
            continue  # Skip duplicates (roles sharing servers)
        seen_pids.add(info.pid)
        cmdline: list[str] = []

        if info.pid == -1:
            # Docker-managed container
            alive = docker_container_running(info.role)
            # Look up health_path for this service (SearXNG uses /, others use /health)
            health_path = "/health"
            for svc in DOCKER_SERVICES:
                if svc["name"] == info.role:
                    health_path = svc.get("health_path", "/health")
                    break
            healthy = wait_for_health(info.port, timeout=3, path=health_path) if alive else False
            status = "healthy" if healthy else ("running" if alive else "stopped")
            pid_str = "docker"
        else:
            # Native process
            try:
                os.kill(info.pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            healthy = wait_for_health(info.port, timeout=3) if alive else False
            if not alive and is_port_in_use(info.port):
                # PID drift can happen if the original launcher PID exits while
                # a listener remains healthy on the same port.
                replacement_pids = _pids_on_port(info.port)
                if replacement_pids:
                    replacement_pid = replacement_pids[0]
                    info.pid = replacement_pid
                    state[name] = info
                    alive = True
                    healthy = wait_for_health(info.port, timeout=3)
            status = "healthy" if healthy else ("running" if alive else "dead")
            pid_str = str(info.pid)
            cmdline = _stack_processes.process_cmdline(info.pid) if alive else []

        model = Path(info.model_path).stem if info.model_path != "uvicorn" else "uvicorn"
        launch_contract = _launch_contract_for_process(name, info, contracts_by_role)
        launch_requirements = launch_contract.get("requirements")
        launch_requirements = launch_requirements if isinstance(launch_requirements, dict) else {}
        attestation = _status_attestation(info, alive, cmdline, launch_requirements)
        warning = _attestation_warning(name, info, attestation, cmdline, launch_requirements)
        if warning:
            attestation_warnings.append(warning)
        attestation_warnings.extend(
            _runtime_attestation_warnings(name, info, cmdline, launch_contract)
        )

        print(
            f"{name:<25} {info.port:<8} {pid_str:<10} {status:<10} "
            f"{attestation:<12} {model[:30]}"
        )

    for role in unavailable_optional_roles:
        print(
            f"{role:<25} {PORT_MAP[role]:<8} {'-':<10} "
            f"{'unavailable_optional':<10} {'n/a':<12} configured optional"
        )

    # The state file is an observation of a launch, not the launch contract.
    # A crash between process launch and state persistence previously made an
    # active-priors service disappear from `status` entirely. Reconcile only
    # the currently declared live roles (plus non-optional Docker services):
    # inactive warm roles deliberately do not appear here.
    for name, port, health_path in missing_declared_services:
        listening = is_port_in_use(port)
        healthy = wait_for_health(port, timeout=3, path=health_path) if listening else False
        status = "healthy" if healthy else "unavailable"
        print(
            f"{name:<25} {port:<8} {'-':<10} {status:<10} "
            f"{'state-missing':<12} manifest-declared; no state row"
        )

    print()
    if attestation_warnings:
        print("Attestation warnings:")
        for warning in attestation_warnings:
            print(f"  [!] {warning}")
        print()
    print(_episodic_embedding_status_line())
    print()
    print(f"State file: {STATE_FILE}")
    # ESC-8 Fix 2: `status` is a READ command — it no longer persists state.
    # The prior save_state(state) bumped the state-file mtime without refreshing
    # the runtime-facts manifest, making the manifest look stale to
    # read_runtime_stack_numa_mode(); a reload run right after a status then fell
    # through to the shell env (the order-dependence hazard, audit §1a). The only
    # thing lost is opportunistic PID-drift persistence, which reload/stop
    # re-derive via port scans — lower blast radius than giving a read command a
    # manifest-write side effect.
    return 0


# =============================================================================
# MemRL and Tool Registry Initialization
# =============================================================================


def init_memrl_and_tools() -> bool:
    """Initialize MemRL databases and tool registry for the session.

    This ensures all deterministic tools (41 total) are ready and
    the REPL memory system is initialized with seed examples.
    """
    success = True

    # [6] REPL Memory Initialization
    print("[6] Initializing MemRL databases...")

    # Initialize REPL seed examples
    seed_loader_path = _PATHS["project_root"] / "orchestration/repl_memory/seed_loader.py"
    if seed_loader_path.exists():
        result = subprocess.run(
            [sys.executable, str(seed_loader_path), "--init"],
            capture_output=True,
            text=True,
            cwd=str(_PATHS["project_root"]),
        )
        if result.returncode == 0:
            print("  [OK] REPL seed examples loaded")
        else:
            print(
                f"  [WARN] Seed loader failed: {result.stderr[:100] if result.stderr else 'no output'}"
            )

    # Warm up all embedding servers with test query
    try:
        import urllib.request
        import urllib.error

        test_payload = json.dumps({"content": "test embedding warmup"}).encode()
        healthy_count = 0
        for port in EMBEDDER_PORTS:
            try:
                req = urllib.request.Request(
                    f"http://localhost:{port}/embedding",
                    data=test_payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        healthy_count += 1
            except (urllib.error.URLError, TimeoutError, OSError):
                pass  # Expected during warmup — server may still be starting
        print(f"  [OK] Embedding servers warmed up: {healthy_count}/{len(EMBEDDER_PORTS)} healthy")
    except Exception as e:
        print(f"  [WARN] Embedding warmup failed: {e}")

    # [7] Tool Registry Initialization
    print("[7] Initializing deterministic tool registry...")

    # Validate tool registry exists
    tool_registry_path = _PATHS["project_root"] / "orchestration/tool_registry.yaml"
    if not tool_registry_path.exists():
        print(f"  [!] Tool registry not found: {tool_registry_path}")
        success = False
    else:
        # Load and validate tool executor
        try:
            # Add src to path for imports
            import sys as _sys

            _sys.path.insert(0, str(_PATHS["project_root"]))
            from orchestration.tools.executor import get_executor

            executor = get_executor()
            tools = executor.list_tools()
            print(f"  [OK] Tool registry loaded: {len(tools)} tools")

            # Categorize tools
            categories: dict[str, int] = {}
            for t in tools:
                cat = t.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1
            for cat, count in sorted(categories.items()):
                print(f"      {cat}: {count}")
        except Exception as e:
            print(f"  [WARN] Tool executor init failed: {e}")

    # Verify C++ math tools binary
    cpp_binary = LLAMA_MATH_TOOLS
    if cpp_binary.exists():
        print(f"  [OK] C++ math tools binary found: {cpp_binary}")
    else:
        print(f"  [WARN] C++ math tools not built: {cpp_binary}")
        print(
            f"        Run: cd {_PATHS['llm_root']}/llama.cpp/tools/math-tools "
            "&& cmake -B build && cmake --build build"
        )

    return success


# =============================================================================
# Checkpoint Hooks for Self-Management Procedures
# =============================================================================

CHECKPOINT_DIR = _PATHS["project_root"] / "orchestration/checkpoints"
_REGISTRY_PATH_FOR_CHECKPOINT = _PATHS["project_root"] / "orchestration/model_registry.yaml"


def checkpoint_create(name: str, include_state: bool = True) -> dict[str, Any]:
    """Create a checkpoint (wrapper supplying CHECKPOINT_DIR + STATE_FILE)."""
    return _stack_checkpoint.checkpoint_create(
        name,
        CHECKPOINT_DIR,
        STATE_FILE,
        include_state=include_state,
        registry_path=_REGISTRY_PATH_FOR_CHECKPOINT,
    )


def checkpoint_restore(checkpoint_id: str) -> dict[str, Any]:
    """Restore from a checkpoint (wrapper supplying CHECKPOINT_DIR + STATE_FILE)."""
    return _stack_checkpoint.checkpoint_restore(checkpoint_id, CHECKPOINT_DIR, STATE_FILE)


def checkpoint_list(limit: int = 10) -> list[dict[str, Any]]:
    """List available checkpoints (wrapper supplying CHECKPOINT_DIR)."""
    return _stack_checkpoint.checkpoint_list(CHECKPOINT_DIR, limit=limit)


def checkpoint_delete(checkpoint_id: str) -> bool:
    """Delete a checkpoint (wrapper supplying CHECKPOINT_DIR)."""
    return _stack_checkpoint.checkpoint_delete(checkpoint_id, CHECKPOINT_DIR)


# Export hooks for use by procedure_registry
__checkpoint_hooks__ = {
    "create": checkpoint_create,
    "restore": checkpoint_restore,
    "list": checkpoint_list,
    "delete": checkpoint_delete,
}
