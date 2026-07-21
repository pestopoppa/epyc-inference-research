#!/usr/bin/env python3
from __future__ import annotations

"""Clean-window / benchmark execution bridge (inference-batch item B2).

Consumed by the operator's long-horizon loop (and, downstream, by B5's verdict
layer). Given ONE inference-batch manifest entry — either a ``clean_window_entry``
reference or a raw ``command`` — this bridge resolves it to a runnable command
and runs it under canonical discipline, but ALWAYS behind a MANDATORY dry-run
preflight gate. It PREPARES and validates; it never runs live inference unless
the operator passes the explicit, default-OFF ``--execute`` flag.

Flow (per entry)
----------------
    resolve  ->  topology-hash gate  ->  canonical dry-run  ->  [gated execute]
      |               |                        |                      |
      |               |                        |                      +-- ONLY when
      |               |                        |                          --execute is
      |               |                        |                          set AND the
      |               |                        |                          gate passed.
      |               |                        +-- llama-bench  : bench_canonical.sh --dry-run
      |               |                            server-suite : run_benchmark.py --dry-run
      |               |                            resolved cmd : resolution-only (self-guarded runner)
      |               +-- compare entry.required_topology_hash vs the live
      |                   registry-derived hash (reuse clean_window_manifest's
      |                   _file_sha256). Consume B4's attestation JSON if present;
      |                   otherwise the window is UNVERIFIED and execute is refused.
      +-- clean_window_entry : regenerate (or load) the clean-window manifest,
          look the entry up by selector (package/kind/role/suite/context_length),
          extract its `command` + model metadata.
          command             : take the entry's literal argv / shell string.

Ownership / boundaries
----------------------
* This module NEVER writes the batch ledger (B1 owns execution_manifest.jsonl).
  It RETURNS a structured result dict for the caller to record.
* It does not import from epyc-orchestrator ``scripts/lab`` — the timeout,
  output-artifact-validation, and --continue-on-error patterns are re-implemented
  here (borrowed, not imported).
* It does not edit clean_window_manifest.py / bench_canonical.sh /
  canonical_recipe.py / run_benchmark.py — it consumes them.

Result dict (JSONL-appendable; the B5 verdict layer consumes it)
----------------------------------------------------------------
    {
      "entry_id":         str,
      "phase":            "preflight" | "execute" | "skipped",
      "dry_run_ok":       bool,
      "blocking_reasons": list[str],
      "command_resolved": str | None,     # shell-ready EXECUTE command
      "artifacts":        list[str],       # predicted (preflight) or produced (execute)
      "wall_clock_s":     float,
      "exit_code":        int | None,      # None in preflight/skipped
      # --- superset (extra keys are fine for JSONL consumers) ---
      "schema_version":   "batch_entry_result.v1",
      "driver":           "clean_window_entry" | "command",
      "exec_path":        "llama_bench" | "server_suite" | "resolved_command",
      "dry_run_mode":     "canonical_subprocess" | "resolution_only" | None,
      "dry_run_command":  str | None,
      "dry_run_exit_code": int | None,
      "topology":         {...},
      "model_path":       str | None,
      "notes":            list[str],
      "generated_at":     iso8601 str,
    }
"""

import argparse
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

BENCHMARK_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = BENCHMARK_DIR.parents[1]
SCRIPTS_DIR = RESEARCH_ROOT / "scripts"
ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
ORCHESTRATOR_PYTHON = ORCHESTRATOR_ROOT / ".venv" / "bin" / "python"
EPYC_ROOT = Path("/mnt/raid0/llm/epyc-root")
EPYC_ROOT_COORDINATION = EPYC_ROOT / "scripts" / "coordination"

# clean_window_manifest.py performs its own sys.path.insert for dataset_adapters /
# lib.registry / suites at import time, so it must be importable first.
for _p in (str(RESEARCH_ROOT), str(SCRIPTS_DIR), str(BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clean_window_manifest as cwm  # noqa: E402  (path set up above)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULT_SCHEMA_VERSION = "batch_entry_result.v1"

BENCH_CANONICAL_SH = BENCHMARK_DIR / "bench_canonical.sh"
RUN_BENCHMARK_PY = BENCHMARK_DIR / "run_benchmark.py"

# B4 (preflight_gate.py) writes its attestation JSON here; B7's quiet-window
# detector shares the same coordination tree. If the directory is absent or holds
# no matching attestation, the topology is treated as UNVERIFIED and execute is
# refused (dry-run/preflight still proceeds — it runs no inference).
DEFAULT_ATTESTATION_DIR = Path(
    "/mnt/raid0/llm/epyc-root/coordination/inference-batch/attestations"
)

DEFAULT_DRY_RUN_TIMEOUT_S = 180
DEFAULT_EXECUTE_TIMEOUT_S = 4 * 3600  # 4h ceiling for a single clean-window task

DRIVER_CLEAN_WINDOW = "clean_window_entry"
DRIVER_COMMAND = "command"

PATH_LLAMA_BENCH = "llama_bench"
PATH_SERVER_SUITE = "server_suite"
PATH_RESOLVED_COMMAND = "resolved_command"

# clean-window entry kinds that map to the server-path quality-suite runner.
SERVER_SUITE_KINDS = {"run_benchmark_suite"}
# clean-window entry kinds that map to the raw llama-bench recipe.
LLAMA_BENCH_KINDS = {"llama_bench", "llama_bench_speed"}

_SELECTOR_KEYS = ("package", "kind", "role", "suite", "context_length")


class BatchEntryError(RuntimeError):
    """Raised for operator-facing bridge failures (bad selector, missing model, ...)."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Entry resolution
# ---------------------------------------------------------------------------


@dataclass
class ResolvedEntry:
    entry_id: str
    driver: str
    exec_path: str
    command_resolved: str            # shell-ready EXECUTE command
    command_argv: Optional[list[str]]  # structured argv when available, else None
    model_path: Optional[str]
    required_topology_hash: Optional[str]
    topology_artifact: Optional[str]
    baseline_run: Optional[str]
    bench: dict[str, Any] = field(default_factory=dict)
    expected_artifacts: list[str] = field(default_factory=list)
    source_entry: Optional[dict[str, Any]] = None
    preconditions: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    cwd: Optional[str] = None
    requires_live_stack_contract: bool = False
    notes: list[str] = field(default_factory=list)


def derive_entry_id(spec: dict[str, Any]) -> str:
    """Build a stable entry id from selector/entry fields."""
    parts = [str(spec[k]) for k in _SELECTOR_KEYS if spec.get(k) not in (None, "")]
    return ":".join(parts) if parts else "entry"


def load_manifest(
    manifest_json: Optional[Path] = None,
    *,
    regenerate: bool = True,
    builder: Callable[[], dict[str, Any]] = cwm.build_manifest,
) -> dict[str, Any]:
    """Load a clean-window manifest.

    Prefers a pre-generated JSON (fast, hermetic); otherwise regenerates via
    clean_window_manifest.build_manifest() (read-only against the live registry —
    a plan emitter, no model calls). ``builder`` is injectable for tests.
    """
    if manifest_json is not None:
        p = Path(manifest_json)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        if not regenerate:
            raise BatchEntryError(f"manifest json not found: {p}")
    if not regenerate:
        raise BatchEntryError("no manifest json provided and regenerate=False")
    return builder()


def find_clean_window_entry(manifest: dict[str, Any], selector: dict[str, Any]) -> dict[str, Any]:
    """Return the single clean-window entry matching every provided selector key.

    Selector keys are drawn from package/kind/role/suite/context_length. Raises
    BatchEntryError on zero matches (unknown) or more than one (ambiguous).
    """
    active = {k: selector[k] for k in _SELECTOR_KEYS if selector.get(k) not in (None, "")}
    if not active:
        raise BatchEntryError("clean_window_entry selector is empty")
    matches = [
        e
        for e in manifest.get("entries", [])
        if all(e.get(k) == v for k, v in active.items())
    ]
    if not matches:
        raise BatchEntryError(f"no clean-window entry matches selector {active}")
    if len(matches) > 1:
        ids = [derive_entry_id(m) for m in matches]
        raise BatchEntryError(
            f"ambiguous selector {active} matched {len(matches)} entries: {ids}; "
            "add kind/suite/context_length to disambiguate"
        )
    return matches[0]


def classify_exec_path(
    *,
    driver: str,
    kind: Optional[str],
    command: Optional[str],
    model_path: Optional[str],
    override: Optional[str] = None,
) -> str:
    """Pick the resolution/dry-run path for an entry.

    Precedence: explicit override > kind mapping > command sniffing.
    """
    if override in (PATH_LLAMA_BENCH, PATH_SERVER_SUITE, PATH_RESOLVED_COMMAND):
        return override
    if kind in LLAMA_BENCH_KINDS:
        return PATH_LLAMA_BENCH
    if kind in SERVER_SUITE_KINDS:
        return PATH_SERVER_SUITE
    if command and "run_benchmark.py" in command:
        return PATH_SERVER_SUITE
    # A bare model with no recognised suite kind reads as a raw llama-bench request.
    if kind is None and model_path and driver == DRIVER_COMMAND and not command:
        return PATH_LLAMA_BENCH
    return PATH_RESOLVED_COMMAND


def _command_to_str(command: Any) -> str:
    if command is None:
        return ""
    if isinstance(command, (list, tuple)):
        return shlex.join(str(c) for c in command)
    return str(command)


def _nested_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _requires_live_stack_contract(
    *,
    command: Optional[str],
    exec_path: str,
    preconditions: dict[str, Any],
    execution: dict[str, Any],
) -> bool:
    """Return whether an entry measures the live production stack.

    Direct llama-bench rows validate their own binary/recipe through the
    canonical dry-run path. Live server/API rows need the stronger launch
    contract check so the loop cannot spend inference on a drifted or
    partially-optimized stack.
    """
    if exec_path == PATH_LLAMA_BENCH:
        return False
    if str(execution.get("concurrency_mode") or "") == "serial_noninference":
        return False
    cmd = command or ""
    live_markers = (
        "localhost:8000",
        "127.0.0.1:8000",
        "--server-mode",
        "run_benchmark.py",
        "eval_batch_serving_evaltower_window.py",
        "reviewer_",
        "scripts/analysis/",
        "scripts/autopilot/",
    )
    if any(marker in cmd for marker in live_markers):
        return True
    models = preconditions.get("models")
    return isinstance(models, list) and bool(models)


def resolve_entry(
    batch_entry: dict[str, Any],
    *,
    manifest: Optional[dict[str, Any]] = None,
    manifest_json: Optional[Path] = None,
    regenerate: bool = True,
    builder: Callable[[], dict[str, Any]] = cwm.build_manifest,
) -> ResolvedEntry:
    """Resolve an inference-batch entry to a runnable command + metadata.

    ``batch_entry`` (the consumer contract, kept permissive until B1's
    inference_batch.schema.json lands):
        driver:                 "clean_window_entry" | "command" (inferred if absent)
        selector:               {package,kind,role,suite,context_length}  (clean_window_entry)
        command:                str | argv-list                            (command driver)
        required_topology_hash: sha256 recorded at batch-compile time       (optional)
        baseline_run:           run-id for server-suite --baseline-run resume (optional)
        model_path:             override / raw-llama-bench model            (optional)
        bench:                  {n_gen,n_prompt,reps,extra_flags}           (optional)
        exec_path:              force "llama_bench"|"server_suite"|"resolved_command"
        expected_artifacts:     list[str]                                   (optional)
        artifacts.outputs:      list[str]                                   (optional)
    """
    driver = batch_entry.get("driver")
    preconditions = _nested_dict(batch_entry.get("preconditions"))
    execution = _nested_dict(batch_entry.get("execution"))
    if driver is None:
        driver = DRIVER_CLEAN_WINDOW if batch_entry.get("selector") else DRIVER_COMMAND

    source_entry: Optional[dict[str, Any]] = None
    notes: list[str] = []
    kind: Optional[str] = None
    command: Optional[str] = None
    command_argv: Optional[list[str]] = None
    model_path = batch_entry.get("model_path")
    required_hash = batch_entry.get("required_topology_hash")
    topology_artifact = batch_entry.get("topology_artifact")
    if isinstance(preconditions.get("topology"), dict):
        required_hash = required_hash or preconditions["topology"].get("required_topology_hash")
        topology_artifact = topology_artifact or preconditions["topology"].get("topology_artifact")

    if driver == DRIVER_CLEAN_WINDOW:
        selector = batch_entry.get("selector") or {}
        if manifest is None:
            manifest = load_manifest(manifest_json, regenerate=regenerate, builder=builder)
        source_entry = find_clean_window_entry(manifest, selector)
        kind = source_entry.get("kind")
        command = source_entry.get("command")
        model_meta = source_entry.get("model") or {}
        model_path = model_path or model_meta.get("model_path")
        # Manifest-level topology hash is the batch's compile-time expectation
        # unless the batch entry overrode it explicitly.
        topo = manifest.get("topology") or {}
        required_hash = required_hash or topo.get("required_topology_hash")
        topology_artifact = topology_artifact or topo.get("topology_artifact")
        for n in source_entry.get("notes", []) or []:
            notes.append(f"clean_window_note: {n}")
        entry_id = batch_entry.get("entry_id") or derive_entry_id(source_entry)
    elif driver == DRIVER_COMMAND:
        raw = batch_entry.get("command")
        if raw is None:
            raw = execution.get("command")
        if raw is None and not model_path:
            raise BatchEntryError("command driver requires a 'command' or 'model_path'")
        if isinstance(raw, (list, tuple)):
            command_argv = [str(c) for c in raw]
        command = _command_to_str(raw) or None
        entry_id = (
            batch_entry.get("entry_id")
            or batch_entry.get("task_id")
            or derive_entry_id(batch_entry)
            or "command-entry"
        )
    else:
        raise BatchEntryError(f"unknown driver: {driver!r}")

    exec_path = classify_exec_path(
        driver=driver,
        kind=kind,
        command=command,
        model_path=model_path,
        override=batch_entry.get("exec_path"),
    )

    if command is None and exec_path != PATH_LLAMA_BENCH:
        raise BatchEntryError(
            f"entry {entry_id} resolved to no command (exec_path={exec_path}); "
            "clean-window entry may be blocked/non-executable"
        )

    if command_argv is None and command is not None:
        # Only attempt argv splitting for simple (non-compound) commands.
        if "&&" not in command and "|" not in command and "$(" not in command:
            try:
                command_argv = shlex.split(command)
            except ValueError:
                command_argv = None

    # command_resolved holds the EXECUTE command (shell-ready). For llama_bench
    # with only a model_path we synthesise the canonical execute command now.
    command_resolved = command
    bench = dict(batch_entry.get("bench") or {})
    if exec_path == PATH_LLAMA_BENCH:
        if not model_path:
            raise BatchEntryError(f"llama_bench entry {entry_id} has no model_path")
        exec_argv = _llama_bench_argv(model_path, bench, dry_run=False)
        command_argv = exec_argv
        command_resolved = shlex.join(exec_argv)
    elif exec_path == PATH_SERVER_SUITE and command_argv is not None:
        exec_argv = _server_suite_execute_argv(command_argv, batch_entry.get("baseline_run"))
        command_argv = exec_argv
        command_resolved = shlex.join(exec_argv)

    expected_artifacts = [str(a) for a in (batch_entry.get("expected_artifacts") or [])]
    artifacts_meta = _nested_dict(batch_entry.get("artifacts"))
    for item in artifacts_meta.get("outputs") or []:
        expected_artifacts.append(str(item))

    return ResolvedEntry(
        entry_id=entry_id,
        driver=driver,
        exec_path=exec_path,
        command_resolved=command_resolved or "",
        command_argv=command_argv,
        model_path=model_path,
        required_topology_hash=required_hash,
        topology_artifact=topology_artifact,
        baseline_run=batch_entry.get("baseline_run"),
        bench=bench,
        expected_artifacts=expected_artifacts,
        source_entry=source_entry,
        preconditions=preconditions,
        execution=execution,
        cwd=str(execution.get("cwd")) if execution.get("cwd") else None,
        requires_live_stack_contract=_requires_live_stack_contract(
            command=command_resolved,
            exec_path=exec_path,
            preconditions=preconditions,
            execution=execution,
        ),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Canonical command construction (execute + dry-run)
# ---------------------------------------------------------------------------


def _llama_bench_argv(model_path: str, bench: dict[str, Any], *, dry_run: bool) -> list[str]:
    """Build the bench_canonical.sh argv. ``--dry-run`` validates env/topology
    drift and prints the canonical command WITHOUT running llama-bench."""
    argv: list[str] = [
        "bash",
        str(BENCH_CANONICAL_SH),
        "-m",
        str(model_path),
        "-n",
        str(bench.get("n_gen", 512)),
        "-p",
        str(bench.get("n_prompt", 0)),
        "-r",
        str(bench.get("reps", 2)),
    ]
    extra = bench.get("extra_flags") or []
    if extra:
        argv += ["--", *[str(x) for x in extra]]
    if dry_run:
        # place --dry-run before the `--` passthrough so bench_canonical parses it
        if "--" in argv:
            i = argv.index("--")
            argv = argv[:i] + ["--dry-run"] + argv[i:]
        else:
            argv.append("--dry-run")
    return argv


def _server_suite_execute_argv(base_argv: list[str], baseline_run: Optional[str]) -> list[str]:
    """Model the server-path quality-suite EXECUTE invocation.

    The clean-window command is a run_benchmark.py server-mode / skip-speed
    invocation. For resume we swap ``--new-run`` for ``--baseline-run <id>``
    (per the plan's '--baseline-run resume'); otherwise it is passed through
    unchanged. No --dry-run here — this is the (gated) execute form.
    """
    argv = list(base_argv)
    if baseline_run:
        argv = [a for a in argv if a != "--new-run"]
        if "--baseline-run" not in argv:
            argv += ["--baseline-run", str(baseline_run)]
    return argv


def build_dry_run_command(resolved: ResolvedEntry) -> Optional[list[str]]:
    """Return the canonical dry-run argv for this entry, or None when the entry
    has no canonical dry-run wrapper (resolution-only path)."""
    if resolved.exec_path == PATH_LLAMA_BENCH:
        if not resolved.model_path:
            return None
        return _llama_bench_argv(resolved.model_path, resolved.bench, dry_run=True)
    if resolved.exec_path == PATH_SERVER_SUITE and resolved.command_argv is not None:
        argv = list(resolved.command_argv)
        if "--dry-run" not in argv:
            argv.append("--dry-run")
        return argv
    return None


# ---------------------------------------------------------------------------
# Topology-hash gate + B4 attestation
# ---------------------------------------------------------------------------


@dataclass
class TopologyGateResult:
    required_hash: Optional[str]
    live_hash: Optional[str]
    topology_artifact: Optional[str]
    hash_match: bool
    attestation_path: Optional[str]
    verified: bool           # safe to EXECUTE only when True
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_hash": self.required_hash,
            "live_hash": self.live_hash,
            "topology_artifact": self.topology_artifact,
            "hash_match": self.hash_match,
            "attestation_path": self.attestation_path,
            "verified": self.verified,
            "reasons": list(self.reasons),
        }


@dataclass
class StackContractGateResult:
    required: bool
    ok: bool
    warnings: list[str]
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "ok": self.ok,
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
        }


@dataclass
class ContentionMatrixGateResult:
    required: bool
    ok: bool
    command: Optional[str]
    exit_code: Optional[int]
    reasons: list[str]
    stdout_tail: str = ""
    stderr_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "ok": self.ok,
            "command": self.command,
            "exit_code": self.exit_code,
            "reasons": list(self.reasons),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


@dataclass
class AutopilotPreconditionGateResult:
    required: str
    ok: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "ok": self.ok,
            "reason": self.reason,
        }


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ok", "pass", "verified"}
    return bool(value)


def load_attestation(
    attestation_dir: Path,
    *,
    expected_hash: Optional[str],
) -> Optional[dict[str, Any]]:
    """Return B4's attestation (dict + _path) whose recorded topology hash matches
    ``expected_hash`` and which reports live affinity verified. None otherwise.

    Consumed contract (permissive — B4/preflight_gate.py owns the producer):
        topology_hash | required_topology_hash | registry_hash : sha256 str
        live_affinity_verified | affinity_verified            : bool-ish
        status (optional)                                     : ok/pass/verified
    """
    if not attestation_dir.exists():
        return None
    best: Optional[dict[str, Any]] = None
    for path in sorted(attestation_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        att_hash = (
            data.get("topology_hash")
            or data.get("required_topology_hash")
            or data.get("registry_hash")
        )
        affinity_ok = _truthy(
            data.get("live_affinity_verified", data.get("affinity_verified"))
        )
        status_ok = _truthy(data.get("status", "ok")) if "status" in data else True
        if expected_hash is not None and att_hash != expected_hash:
            continue
        if affinity_ok and status_ok:
            data = dict(data)
            data["_path"] = str(path)
            best = data  # last (newest by sorted name) wins
    return best


def topology_gate(
    resolved: ResolvedEntry,
    *,
    attestation_dir: Path = DEFAULT_ATTESTATION_DIR,
    hasher: Callable[[Path], Optional[str]] = cwm._file_sha256,
    live_hash_override: Optional[str] = None,
) -> TopologyGateResult:
    """Compare the entry's required_topology_hash against the live registry-derived
    hash (reusing clean_window_manifest's hashing) and fold in B4's attestation.

    EXECUTE is permitted only when hashes match AND a matching B4 attestation is
    present. Any drift / missing attestation yields blocking reasons (preflight
    and dry-run still run — they touch no inference)."""
    reasons: list[str] = []
    required = resolved.required_topology_hash
    artifact = resolved.topology_artifact

    if live_hash_override is not None:
        live = live_hash_override
    elif artifact:
        live = hasher(Path(artifact))
    elif required is None:
        live = None
    else:
        # Full inference-batch command entries often carry a required topology
        # hash without a clean-window artifact. In that case the B4 attestation
        # is the live-hash evidence; leave `live` unset until the attestation
        # lookup below and do not fabricate a hash.
        live = None

    if required is None:
        return TopologyGateResult(
            required_hash=None,
            live_hash=None,
            topology_artifact=artifact,
            hash_match=True,
            attestation_path=None,
            verified=True,
            reasons=[],
        )
    if artifact is not None and live is None:
        reasons.append(
            f"could not compute live topology hash from {artifact} (missing/unreadable registry)"
        )

    hash_match = live is None or required == live
    if required is not None and live is not None and not hash_match:
        reasons.append(
            f"topology hash mismatch: required={required} live={live} "
            f"(registry drifted since batch compile; re-generate the batch)"
        )

    attestation = load_attestation(attestation_dir, expected_hash=live or required)
    attestation_path = attestation.get("_path") if attestation else None
    if live is None and attestation is not None:
        live = (
            attestation.get("topology_hash")
            or attestation.get("required_topology_hash")
            or attestation.get("registry_hash")
        )
        hash_match = live == required
    if attestation is None:
        reasons.append(
            f"no matching B4 attestation under {attestation_dir}; "
            "live TCP/affinity unverified (B4 preflight_gate not run) — execute refused"
        )

    verified = hash_match and attestation is not None
    return TopologyGateResult(
        required_hash=required,
        live_hash=live,
        topology_artifact=artifact,
        hash_match=hash_match,
        attestation_path=attestation_path,
        verified=verified,
        reasons=reasons,
    )


def live_stack_contract_gate(
    resolved: ResolvedEntry,
    *,
    runner: Callable[..., tuple[int, str, str]],
) -> StackContractGateResult:
    """Check the live production stack against generated launch contracts.

    The orchestrator owns this attestation logic because it has the runtime
    state file and stack_priors contract. We call it as a subprocess so the
    research bridge can remain a thin execution boundary while still refusing to
    run live-stack entries on stale/non-optimized servers.
    """
    if not resolved.requires_live_stack_contract:
        return StackContractGateResult(required=False, ok=True, warnings=[], reasons=[])

    if not ORCHESTRATOR_ROOT.exists():
        return StackContractGateResult(
            required=True,
            ok=False,
            warnings=[],
            reasons=[f"orchestrator root missing: {ORCHESTRATOR_ROOT}"],
        )
    python = ORCHESTRATOR_PYTHON if ORCHESTRATOR_PYTHON.exists() else Path(sys.executable)
    code = (
        "import sys\n"
        "import json\n"
        f"sys.path.insert(0, {str(ORCHESTRATOR_ROOT)!r})\n"
        "from scripts.server.stack_commands import runtime_attestation_warnings\n"
        "warnings = runtime_attestation_warnings()\n"
        "print(json.dumps({'warnings': warnings}, sort_keys=True))\n"
    )
    argv = [str(python), "-c", code]
    rc, out, err = runner(argv, timeout_s=60)
    if rc != 0:
        tail = (err or out or "").strip().splitlines()[-8:]
        return StackContractGateResult(
            required=True,
            ok=False,
            warnings=[],
            reasons=[
                "live stack launch-contract checker failed "
                f"(exit {rc}): {' | '.join(tail)[:600]}"
            ],
        )
    try:
        payload = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return StackContractGateResult(
            required=True,
            ok=False,
            warnings=[],
            reasons=[f"live stack launch-contract checker emitted invalid JSON: {exc}"],
        )
    warnings = payload.get("warnings")
    warnings = [str(item) for item in warnings] if isinstance(warnings, list) else []
    reasons = (
        []
        if not warnings
        else [
            f"live stack launch contract has {len(warnings)} warning(s); "
            "refusing to measure a drifted/non-optimized stack"
        ]
    )
    return StackContractGateResult(
        required=True,
        ok=not warnings,
        warnings=warnings,
        reasons=reasons,
    )


def _is_eval_fanout_entry(resolved: ResolvedEntry) -> bool:
    mode = str((resolved.execution or {}).get("concurrency_mode") or "")
    return "eval_fanout" in mode


def _tail(text: str, limit: int = 1200) -> str:
    return text[-limit:] if len(text) > limit else text


def contention_matrix_gate(
    resolved: ResolvedEntry,
    *,
    runner: Callable[..., tuple[int, str, str]],
) -> ContentionMatrixGateResult:
    """Require fresh contention-matrix evidence for every eval fanout entry."""
    if not _is_eval_fanout_entry(resolved):
        return ContentionMatrixGateResult(
            required=False,
            ok=True,
            command=None,
            exit_code=None,
            reasons=[],
        )

    reasons: list[str] = []
    topology = (resolved.preconditions or {}).get("topology") or {}
    if isinstance(topology, dict) and topology.get("contention_matrix") == "not_required":
        reasons.append(
            "eval_fanout entry declares contention_matrix:not_required; "
            "compile must pin a fresh matrix or mark recert required"
        )

    if not ORCHESTRATOR_ROOT.exists():
        reasons.append(f"orchestrator root missing: {ORCHESTRATOR_ROOT}")
        return ContentionMatrixGateResult(
            required=True,
            ok=False,
            command=None,
            exit_code=None,
            reasons=reasons,
        )

    python = ORCHESTRATOR_PYTHON if ORCHESTRATOR_PYTHON.exists() else Path(sys.executable)
    argv = [
        str(python),
        "scripts/validate/check_contention_matrix_fresh.py",
    ]
    rc, out, err = _run_with_cwd(runner, argv, timeout_s=60, cwd=ORCHESTRATOR_ROOT)
    if rc != 0:
        tail = " | ".join((err or out or "").strip().splitlines()[-8:])
        reasons.append(
            f"contention matrix freshness gate failed (exit {rc}): {tail[:600]}"
        )

    return ContentionMatrixGateResult(
        required=True,
        ok=rc == 0 and not reasons,
        command=shlex.join(argv),
        exit_code=rc,
        reasons=reasons,
        stdout_tail=_tail(out),
        stderr_tail=_tail(err),
    )


def autopilot_precondition_gate(
    resolved: ResolvedEntry,
    *,
    load_signals: Optional[dict[str, Any]] = None,
) -> AutopilotPreconditionGateResult:
    """Require the live AutoPilot state to match ``preconditions.autopilot``.

    This is a read-only, no-inference gate. It imports the root-owned pure
    checker so the batch runner and status tooling share the same semantics.
    Entries that omit the precondition default to ``any`` and do not probe the
    live system.
    """
    required = str((resolved.preconditions or {}).get("autopilot") or "any")
    if required == "any":
        return AutopilotPreconditionGateResult(
            required=required,
            ok=True,
            reason="precondition 'any' imposes no autopilot constraint",
        )

    if str(EPYC_ROOT_COORDINATION) not in sys.path:
        sys.path.insert(0, str(EPYC_ROOT_COORDINATION))
    try:
        from autopilot_precondition_gate import check_autopilot_precondition
    except Exception as exc:  # noqa: BLE001
        return AutopilotPreconditionGateResult(
            required=required,
            ok=False,
            reason=f"autopilot precondition checker unavailable: {exc}",
        )

    if load_signals is None:
        try:
            import inference_load_check as ic  # type: ignore[import-not-found]

            load_signals = ic.classify_load()
        except Exception as exc:  # noqa: BLE001
            return AutopilotPreconditionGateResult(
                required=required,
                ok=False,
                reason=f"autopilot state signal unavailable: {exc}",
            )

    ok, reason = check_autopilot_precondition(
        {"preconditions": {"autopilot": required}},
        load_signals,
    )
    return AutopilotPreconditionGateResult(
        required=required,
        ok=bool(ok),
        reason=str(reason),
    )


# ---------------------------------------------------------------------------
# Artifact prediction / validation (borrowed from run_job.py patterns)
# ---------------------------------------------------------------------------

_ARTIFACT_FLAGS = {"--out", "--output", "--output-root", "--out-dir"}


def predict_artifacts(resolved: ResolvedEntry) -> list[str]:
    """Best-effort: pull output paths from the resolved command's known flags.
    In preflight these are PREDICTED (not yet existing)."""
    out: list[str] = list(resolved.expected_artifacts)
    tokens: list[str]
    if resolved.command_argv is not None:
        tokens = resolved.command_argv
    else:
        try:
            tokens = shlex.split(resolved.command_resolved)
        except ValueError:
            tokens = []
    for i, tok in enumerate(tokens[:-1]):
        if tok in _ARTIFACT_FLAGS:
            out.append(tokens[i + 1])
    # dedupe, keep order
    seen: set[str] = set()
    deduped = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def validate_output_artifacts(artifacts: list[str], *, cwd: str | Path | None = None) -> list[str]:
    """Return the subset of predicted artifacts that DO NOT exist after execute.
    An empty list means every expected artifact was produced."""
    base = Path(cwd) if cwd else None
    missing: list[str] = []
    for artifact in artifacts:
        path = Path(artifact)
        if not path.is_absolute() and base is not None:
            path = base / path
        if not path.exists():
            missing.append(artifact)
    return missing


# ---------------------------------------------------------------------------
# Preflight (mandatory) + gated execute
# ---------------------------------------------------------------------------


def _default_runner(
    argv: list[str], *, timeout_s: float, cwd: str | Path | None = None
) -> tuple[int, str, str]:
    """Run a child process with timeout and interrupt-safe process-group cleanup."""
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd or RESEARCH_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        out, err = _terminate_child_group(proc, signal.SIGTERM)
        return 124, out, f"TIMEOUT after {timeout_s}s: {exc}\n{err}".strip()
    except KeyboardInterrupt:
        out, err = _terminate_child_group(proc, signal.SIGINT)
        return 130, out, f"INTERRUPTED by KeyboardInterrupt; child terminated\n{err}".strip()
    except OSError as exc:
        return 127, "", f"spawn error: {exc}"
    return proc.returncode or 0, out, err


def _terminate_child_group(
    proc: subprocess.Popen[str] | None,
    sig: signal.Signals,
    *,
    grace_s: float = 10.0,
) -> tuple[str, str]:
    """Terminate a child process group and collect any remaining output."""
    if proc is None:
        return "", ""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            return proc.communicate(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return proc.communicate()


def _run_with_cwd(
    runner: Callable[..., tuple[int, str, str]],
    argv: list[str],
    *,
    timeout_s: float,
    cwd: str | Path | None,
) -> tuple[int, str, str]:
    try:
        return runner(argv, timeout_s=timeout_s, cwd=cwd)
    except TypeError:
        return runner(argv, timeout_s=timeout_s)


# Test-visible sentinel: proves the gated execute path is never entered when
# --execute is default-off. Tests assert this stays False.
_EXECUTE_INVOKED = False


def _blank_result(resolved: ResolvedEntry) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "entry_id": resolved.entry_id,
        "driver": resolved.driver,
        "exec_path": resolved.exec_path,
        "phase": "preflight",
        "dry_run_ok": False,
        "dry_run_mode": None,
        "dry_run_command": None,
        "dry_run_exit_code": None,
        "blocking_reasons": [],
        "command_resolved": resolved.command_resolved or None,
        "cwd": resolved.cwd,
        "artifacts": [],
        "wall_clock_s": 0.0,
        "exit_code": None,
        "model_path": resolved.model_path,
        "topology": {},
        "stack_contract": {},
        "contention_matrix": {},
        "autopilot_precondition": {},
        "notes": list(resolved.notes),
        "generated_at": _utc_now(),
    }


def run_preflight(
    resolved: ResolvedEntry,
    *,
    attestation_dir: Path = DEFAULT_ATTESTATION_DIR,
    dry_run_timeout_s: float = DEFAULT_DRY_RUN_TIMEOUT_S,
    runner: Callable[..., tuple[int, str, str]] = _default_runner,
    stack_contract_checker: Callable[..., StackContractGateResult] = live_stack_contract_gate,
    contention_matrix_checker: Callable[..., ContentionMatrixGateResult] = contention_matrix_gate,
    autopilot_precondition_checker: Callable[
        ..., AutopilotPreconditionGateResult
    ] = autopilot_precondition_gate,
    live_hash_override: Optional[str] = None,
) -> dict[str, Any]:
    """Run the MANDATORY preflight: topology gate + canonical dry-run. Runs NO
    inference. Returns a result dict with phase='preflight'."""
    t0 = time.perf_counter()
    result = _blank_result(resolved)

    gate = topology_gate(
        resolved,
        attestation_dir=attestation_dir,
        live_hash_override=live_hash_override,
    )
    result["topology"] = gate.as_dict()
    blocking = list(gate.reasons)
    stack_gate = stack_contract_checker(resolved, runner=runner)
    result["stack_contract"] = stack_gate.as_dict()
    blocking.extend(stack_gate.reasons)
    matrix_gate = contention_matrix_checker(resolved, runner=runner)
    result["contention_matrix"] = matrix_gate.as_dict()
    blocking.extend(matrix_gate.reasons)
    autopilot_gate = autopilot_precondition_checker(resolved)
    result["autopilot_precondition"] = autopilot_gate.as_dict()
    if not autopilot_gate.ok:
        blocking.append(autopilot_gate.reason)

    dry_cmd = build_dry_run_command(resolved)
    result["artifacts"] = predict_artifacts(resolved)

    if dry_cmd is None:
        # resolution-only path: the resolved command self-guards at execute time.
        result["dry_run_mode"] = "resolution_only"
        result["dry_run_ok"] = bool(resolved.command_resolved)
        if not resolved.command_resolved:
            blocking.append("entry resolved to an empty command")
    else:
        result["dry_run_mode"] = "canonical_subprocess"
        result["dry_run_command"] = shlex.join(dry_cmd)
        rc, out, err = runner(dry_cmd, timeout_s=dry_run_timeout_s)
        result["dry_run_exit_code"] = rc
        result["dry_run_ok"] = rc == 0
        if rc != 0:
            tail = (err or out or "").strip().splitlines()[-8:]
            blocking.append(
                f"canonical dry-run failed (exit {rc}): {' | '.join(tail)[:600]}"
            )

    result["blocking_reasons"] = blocking
    result["wall_clock_s"] = round(time.perf_counter() - t0, 3)
    return result


def _execute_resolved(
    resolved: ResolvedEntry,
    *,
    preflight: dict[str, Any],
    execute_timeout_s: float,
    runner: Callable[..., tuple[int, str, str]],
) -> dict[str, Any]:
    """GATED live execution. Reached ONLY when the operator passes --execute AND
    the preflight gate passed (dry_run_ok + topology verified). Not exercised by
    tests. Runs the resolved EXECUTE command under a timeout and validates that
    the predicted output artifacts were produced."""
    global _EXECUTE_INVOKED
    _EXECUTE_INVOKED = True

    result = dict(preflight)
    result["phase"] = "execute"
    result["generated_at"] = _utc_now()
    t0 = time.perf_counter()

    argv = resolved.command_argv
    if argv is None:
        # compound shell string (e.g. `cd X && uv run ...`) — run via a shell.
        argv = ["bash", "-lc", resolved.command_resolved]
    try:
        rc, out, err = _run_with_cwd(
            runner,
            argv,
            timeout_s=execute_timeout_s,
            cwd=resolved.cwd,
        )
    except KeyboardInterrupt:
        rc, out, err = 130, "", "INTERRUPTED by KeyboardInterrupt before runner returned"
    result["exit_code"] = rc

    missing = validate_output_artifacts(result.get("artifacts", []), cwd=resolved.cwd)
    reasons = list(result.get("blocking_reasons", []))
    if rc != 0:
        tail = (err or out or "").strip().splitlines()[-8:]
        reasons.append(f"execute failed (exit {rc}): {' | '.join(tail)[:600]}")
    if missing:
        reasons.append(f"expected artifacts missing after execute: {missing}")
    result["blocking_reasons"] = reasons
    result["wall_clock_s"] = round(time.perf_counter() - t0, 3)
    return result


def run_batch_entry(
    batch_entry: dict[str, Any],
    *,
    manifest: Optional[dict[str, Any]] = None,
    manifest_json: Optional[Path] = None,
    regenerate: bool = True,
    builder: Callable[[], dict[str, Any]] = cwm.build_manifest,
    attestation_dir: Path = DEFAULT_ATTESTATION_DIR,
    dry_run_timeout_s: float = DEFAULT_DRY_RUN_TIMEOUT_S,
    execute_timeout_s: float = DEFAULT_EXECUTE_TIMEOUT_S,
    execute: bool = False,               # DEFAULT OFF — never set by tests
    continue_on_error: bool = False,
    runner: Callable[..., tuple[int, str, str]] = _default_runner,
    stack_contract_checker: Callable[..., StackContractGateResult] = live_stack_contract_gate,
    contention_matrix_checker: Callable[..., ContentionMatrixGateResult] = contention_matrix_gate,
    autopilot_precondition_checker: Callable[
        ..., AutopilotPreconditionGateResult
    ] = autopilot_precondition_gate,
    live_hash_override: Optional[str] = None,
) -> dict[str, Any]:
    """Bridge one inference-batch entry through preflight (always) and, only when
    ``execute=True`` AND the gate passed, live execution.

    Returns the structured result dict for the caller (B1 ledger / B5 verdict);
    this function never writes the ledger itself.
    """
    try:
        resolved = resolve_entry(
            batch_entry,
            manifest=manifest,
            manifest_json=manifest_json,
            regenerate=regenerate,
            builder=builder,
        )
    except BatchEntryError as exc:
        if not continue_on_error:
            raise
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "entry_id": batch_entry.get("entry_id") or "unresolved",
            "driver": batch_entry.get("driver"),
            "exec_path": None,
            "phase": "skipped",
            "dry_run_ok": False,
            "dry_run_mode": None,
            "dry_run_command": None,
            "dry_run_exit_code": None,
            "blocking_reasons": [f"resolution failed: {exc}"],
            "command_resolved": None,
            "artifacts": [],
            "wall_clock_s": 0.0,
            "exit_code": None,
            "model_path": None,
            "topology": {},
            "stack_contract": {},
            "contention_matrix": {},
            "autopilot_precondition": {},
            "notes": [],
            "generated_at": _utc_now(),
        }

    preflight = run_preflight(
        resolved,
        attestation_dir=attestation_dir,
        dry_run_timeout_s=dry_run_timeout_s,
        runner=runner,
        stack_contract_checker=stack_contract_checker,
        contention_matrix_checker=contention_matrix_checker,
        autopilot_precondition_checker=autopilot_precondition_checker,
        live_hash_override=live_hash_override,
    )

    if not execute:
        return preflight

    gate = preflight.get("topology", {})
    if preflight.get("blocking_reasons") or not preflight["dry_run_ok"] or not gate.get("verified"):
        # Refuse execute: preflight failed or topology unverified. Keep phase
        # 'preflight' so the caller records a blocked, non-executed entry.
        return preflight

    return _execute_resolved(
        resolved,
        preflight=preflight,
        execute_timeout_s=execute_timeout_s,
        runner=runner,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "entries" in data and isinstance(data["entries"], list):
        return data["entries"]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise BatchEntryError(f"batch-entry file must be an object or list: {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean-window / benchmark execution bridge (B2)")
    p.add_argument("--batch-entry", type=Path, required=True,
                   help="JSON file: a single batch entry, a list, or {entries:[...]}")
    p.add_argument("--manifest-json", type=Path, default=None,
                   help="Pre-generated clean-window manifest JSON (else regenerate)")
    p.add_argument("--no-regenerate", dest="regenerate", action="store_false", default=True,
                   help="Do not regenerate the clean-window manifest if --manifest-json is absent")
    p.add_argument("--attestation-dir", type=Path, default=DEFAULT_ATTESTATION_DIR,
                   help="Directory of B4 attestation JSONs (topology/affinity verification)")
    p.add_argument("--dry-run-timeout-s", type=float, default=DEFAULT_DRY_RUN_TIMEOUT_S)
    p.add_argument("--execute-timeout-s", type=float, default=DEFAULT_EXECUTE_TIMEOUT_S)
    p.add_argument("--continue-on-error", action="store_true",
                   help="Capture per-entry failures into results and keep going")
    p.add_argument("--result-out", type=Path, default=None,
                   help="Append result dicts as JSONL here (NOT the batch ledger; B1 owns that)")
    # DEFAULT OFF. Without this flag the bridge only PREPARES + dry-run-validates.
    p.add_argument("--execute", action="store_true", default=False,
                   help="Actually run the resolved command (gated: requires a passing "
                        "preflight + verified topology). DEFAULT OFF — omit to prepare only.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        entries = _load_entries(args.batch_entry)
    except (OSError, json.JSONDecodeError, BatchEntryError) as exc:
        print(f"run_batch_entry: {exc}", file=sys.stderr)
        return 2

    exit_code = 0
    results: list[dict[str, Any]] = []
    for entry in entries:
        try:
            result = run_batch_entry(
                entry,
                manifest_json=args.manifest_json,
                regenerate=args.regenerate,
                attestation_dir=args.attestation_dir,
                dry_run_timeout_s=args.dry_run_timeout_s,
                execute_timeout_s=args.execute_timeout_s,
                execute=args.execute,
                continue_on_error=args.continue_on_error,
            )
        except BatchEntryError as exc:
            print(f"run_batch_entry: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2
            exit_code = 2
            continue
        results.append(result)
        print(json.dumps(result, sort_keys=True))
        if result["blocking_reasons"] and not args.continue_on_error and args.execute:
            exit_code = 1

    if args.result_out is not None:
        args.result_out.parent.mkdir(parents=True, exist_ok=True)
        with args.result_out.open("a", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, sort_keys=True) + "\n")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
