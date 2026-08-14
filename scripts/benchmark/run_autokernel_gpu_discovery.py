#!/usr/bin/env python3
"""Fast non-promotable MI210 discovery over one factor and workload frame."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_rnd.autokernel import schemas, storage
from scripts.kernel_rnd.autokernel.execution import (
    cpu_region_claim, device_sampler, inference_window)
from scripts.kernel_rnd.autokernel.resource import device_claim
from scripts.benchmark import autokernel_gpu_discovery_beliefs as gpu_beliefs
from scripts.benchmark import autokernel_progression
from scripts.kernel_rnd.autokernel.controller import split_runtime_verifier
from scripts.kernel_rnd.autokernel.controller import gpu_load_admission


SCHEMA_BANK = "epyc.autokernel.gpu_screening_baseline.v2"
SCHEMA_RESULT = "epyc.autokernel.gpu_candidate_only_screen.v2"
SCHEMA_LIVE_GOVERNANCE = "epyc.autokernel.gpu_discovery_live_governance.v1"
SOURCE_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
READY_CONTINUE_INSTRUMENT_COMMIT = "81bf32f11b4a421880e8f25faec3e4ba872363f0"
READY_CONTINUE_CONTRACT_SHA256 = "1411f5e81c1b0b3db6952523922c672d88a78aaff5945865c9ccc2b4fc5fd99f"
CPU_LIST = "184-191"
DEVICE_ID = "mi210_0"
DEFAULT_HOST_BANDWIDTH_BYTES_S = 400 * 1000 * 1000 * 1000
DEFAULT_HOST_TRANSFER_FRACTION = 0.01
VRAM_USED = Path("/sys/class/drm/card2/device/mem_info_vram_used")
KFD_PROCS = Path("/sys/class/kfd/kfd/proc")
MODEL_CALL_WINDOW = inference_window.InferenceCallWindow(timeout_s=600.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cache_bool(build: Path, name: str) -> bool:
    prefix = f"{name}:BOOL="
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):]
            if value in {"ON", "OFF"}:
                return value == "ON"
    raise RuntimeError(f"{build} does not declare {name}:BOOL=ON|OFF")


def artifacts(build: Path) -> dict:
    binary = build / "bin" / "llama-bench"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"llama-bench is not executable: {binary}")
    libraries = {
        path.name: sha256_file(path)
        for path in sorted((build / "bin").glob("*.so*")) if path.is_file()
    }
    return {"binary": str(binary.resolve()), "binary_sha256": sha256_file(binary),
            "libraries": libraries}


def build_identity(build: Path) -> dict:
    source_commit = SOURCE_COMMIT
    try:
        resolved = subprocess.run(
            ["git", "-C", str(build), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        if len(resolved) == 40:
            source_commit = resolved
    except (OSError, subprocess.CalledProcessError):
        pass
    identity = {
        "source_commit": source_commit,
        "hip_graphs": cache_bool(build, "GGML_HIP_GRAPHS"),
        "rocwmma_fattn": cache_bool(build, "GGML_HIP_ROCWMMA_FATTN"),
        "mmq_mfma": cache_bool(build, "GGML_HIP_MMQ_MFMA"),
        "artifacts": artifacts(build),
    }
    return identity


def _kfd_pids() -> tuple[int, ...]:
    try:
        return tuple(sorted(int(path.name) for path in KFD_PROCS.iterdir()
                            if path.name.isdigit()))
    except OSError as exc:
        raise RuntimeError(f"KFD process inventory unreadable: {exc}") from exc


def _start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    try:
        tail = (proc_root / str(pid) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(tail[19])  # proc stat field 22 after pid+comm
    except (OSError, IndexError, ValueError) as exc:
        raise RuntimeError("captured GPU child start ticks are unavailable") from exc


def _runtime_maps_identity(*, runtime_root: Path, arm: str, model: Path,
                           kfd_pid: int, proc_root: Path = Path("/proc"),
                           boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id")) -> dict:
    """Prove actual loader mapping while the governed arm is resident."""
    try:
        manifest = split_runtime_verifier.verify_split_runtime(runtime_root)
        maps = (proc_root / str(kfd_pid) / "maps").read_text(encoding="utf-8")
        identity = split_runtime_verifier.verify_runtime_maps(
            manifest, arm=arm, maps_text=maps, model_path=model,
            model_sha256=sha256_file(model), device_id=DEVICE_ID, kfd_pid=kfd_pid,
            boot_id=boot_id_path.read_text(encoding="utf-8").strip(),
            process_start_ticks=_start_ticks(kfd_pid, proc_root=proc_root))
    except (OSError, split_runtime_verifier.SplitRuntimeError) as exc:
        raise RuntimeError(f"runtime loader-map proof refused: {exc}") from exc
    return identity.to_dict()


@dataclass(frozen=True)
class LoadReadinessPolicy:
    """Typed authority for releasing a serialized cold-load window.

    A cold serialized arm may not release the shared CPU window merely because
    a child started or because VRAM happened to rise.  This policy binds the
    exact split-runtime closure, model and arm that must be witnessed in the
    child's maps while it owns KFD residency.  There is intentionally no
    fallback for a normal build without a split-runtime maps authority.
    """

    schema: str
    runtime_root: Path
    runtime_manifest_sha256: str
    runtime_arm: str
    model_path: Path
    model_sha256: str
    device_id: str
    policy_sha256: str

    @classmethod
    def from_split_runtime(cls, *, runtime_root: Path, runtime_arm: str,
                           model: Path, device_id: str = DEVICE_ID
                           ) -> "LoadReadinessPolicy":
        if runtime_arm not in {"anchor", "candidate"}:
            raise RuntimeError("serialized cold load requires an exact runtime arm")
        root = runtime_root.resolve(strict=True)
        model_path = model.resolve(strict=True)
        manifest = split_runtime_verifier.verify_split_runtime(root)
        body = {
            "schema": "epyc.autokernel.gpu_load_readiness_policy.v1",
            "runtime_root": str(root),
            "runtime_manifest_sha256": manifest.manifest_sha256,
            "runtime_arm": runtime_arm,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "device_id": device_id,
        }
        return cls(
            schema=body["schema"], runtime_root=root,
            runtime_manifest_sha256=body["runtime_manifest_sha256"],
            runtime_arm=runtime_arm, model_path=model_path,
            model_sha256=body["model_sha256"], device_id=device_id,
            policy_sha256=schemas.content_hash(body))

    def __post_init__(self) -> None:
        body = self.to_dict(include_hash=False)
        if (self.schema != "epyc.autokernel.gpu_load_readiness_policy.v1"
                or self.runtime_arm not in {"anchor", "candidate"}
                or self.device_id != DEVICE_ID
                or not self.runtime_root.is_absolute() or not self.model_path.is_absolute()
                or len(self.runtime_manifest_sha256) != 64
                or len(self.model_sha256) != 64
                or self.policy_sha256 != schemas.content_hash(body)):
            raise RuntimeError("serialized load readiness policy is malformed")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, str]:
        body = {
            "schema": self.schema,
            "runtime_root": str(self.runtime_root),
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "runtime_arm": self.runtime_arm,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "device_id": self.device_id,
        }
        if include_hash:
            body["policy_sha256"] = self.policy_sha256
        return body

    def validate_witness(self, witness: Mapping[str, Any]) -> None:
        if not isinstance(witness, Mapping):
            raise RuntimeError("serialized load readiness witness is absent")
        expected = {
            "schema": split_runtime_verifier.MAPS_SCHEMA,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "arm": self.runtime_arm,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "device_id": self.device_id,
        }
        observed = {key: witness.get(key) for key in expected}
        if observed != expected or not isinstance(witness.get("identity_sha256"), str):
            raise RuntimeError("serialized load readiness witness does not bind the sealed runtime/model")


@dataclass(frozen=True)
class ReadyContinueHandshake:
    """One sealed, opt-in pre-measurement barrier for the governed instrument."""

    schema: str
    decision_sha256: str
    readiness_policy_sha256: str
    arm: str
    seed: int
    repetitions: int
    token: str
    ready_path: Path
    continue_path: Path

    @classmethod
    def create(cls, *, root: Path, decision: Mapping[str, Any],
               policy: LoadReadinessPolicy, arm: str, seed: int,
               repetitions: int) -> "ReadyContinueHandshake":
        if (not isinstance(decision.get("decision_sha256"), str)
                or len(decision["decision_sha256"]) != 64
                or arm != policy.runtime_arm or repetitions < 1):
            raise RuntimeError("ready/continue handshake lacks sealed decision authority")
        target = root.resolve()
        if not target.is_absolute() or target.is_symlink():
            raise RuntimeError("ready/continue handshake root is unsafe")
        target.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(target, 0o700)
        root_stat = target.lstat()
        if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.geteuid()
                or stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise RuntimeError("ready/continue handshake root ownership is unsafe")
        marker = {
            "schema": "epyc.autokernel.ready_continue.v1",
            "decision_sha256": decision["decision_sha256"],
            "readiness_policy_sha256": policy.policy_sha256,
            "arm": arm, "seed": seed, "repetitions": repetitions,
        }
        token = schemas.content_hash(marker)
        return cls(schema=marker["schema"], decision_sha256=marker["decision_sha256"],
                   readiness_policy_sha256=marker["readiness_policy_sha256"],
                   arm=arm, seed=seed, repetitions=repetitions, token=token,
                   ready_path=target / "ready", continue_path=target / "continue")

    def __post_init__(self) -> None:
        if (self.schema != "epyc.autokernel.ready_continue.v1" or self.arm not in {"anchor", "candidate"}
                or self.seed < 0 or self.repetitions < 1 or len(self.token) != 64
                or len(self.decision_sha256) != 64 or len(self.readiness_policy_sha256) != 64
                or not self.ready_path.is_absolute() or not self.continue_path.is_absolute()
                or self.ready_path.parent != self.continue_path.parent):
            raise RuntimeError("ready/continue handshake is malformed")

    def argv(self) -> tuple[str, ...]:
        return ("--autokernel-ready-file", str(self.ready_path),
                "--autokernel-continue-file", str(self.continue_path),
                "--autokernel-ready-token", self.token,
                "--autokernel-ready-timeout-ms", "600000")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "decision_sha256": self.decision_sha256,
                "readiness_policy_sha256": self.readiness_policy_sha256,
                "arm": self.arm, "seed": self.seed, "repetitions": self.repetitions,
                "token": self.token, "ready_path": str(self.ready_path),
                "continue_path": str(self.continue_path)}

    def validate_ready(self, *, pid: int) -> dict[str, Any]:
        try:
            file_stat = self.ready_path.lstat()
            raw = self.ready_path.read_text(encoding="ascii")
        except OSError as exc:
            raise RuntimeError("governed instrument ready receipt is unavailable") from exc
        if (self.ready_path.is_symlink() or not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_uid != os.geteuid() or file_stat.st_nlink != 1
                or stat.S_IMODE(file_stat.st_mode) != 0o600 or file_stat.st_size > 512):
            raise RuntimeError("governed instrument ready receipt is unsafe")
        fields = raw.split()
        expected = [self.schema, str(pid), str(self.seed), str(self.repetitions), self.token]
        if fields != expected or raw != " ".join(expected) + "\n":
            raise RuntimeError("governed instrument ready receipt does not bind PID/seed/repetitions/token")
        return {"schema": self.schema, "pid": pid, "seed": self.seed,
                "repetitions": self.repetitions, "token": self.token,
                "ready_path": str(self.ready_path), "continue_path": str(self.continue_path)}

    def continue_after_release(self) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.continue_path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError("cannot create governed instrument continue receipt") from exc
        try:
            payload = (self.token + "\n").encode("ascii")
            if os.write(fd, payload) != len(payload):
                raise RuntimeError("governed instrument continue receipt write was incomplete")
            os.fsync(fd)
            file_stat = os.fstat(fd)
            if (not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.geteuid()
                    or file_stat.st_nlink != 1 or stat.S_IMODE(file_stat.st_mode) != 0o600):
                raise RuntimeError("governed instrument continue receipt ownership is unsafe")
        finally:
            os.close(fd)

    def cleanup(self) -> dict[str, bool]:
        result = {"ready_removed": False, "continue_removed": False}
        for key, path in (("ready_removed", self.ready_path),
                          ("continue_removed", self.continue_path)):
            try:
                if path.exists():
                    file_stat = path.lstat()
                    if (path.is_symlink() or not stat.S_ISREG(file_stat.st_mode)
                            or file_stat.st_uid != os.geteuid() or file_stat.st_nlink != 1):
                        raise RuntimeError("governed instrument handshake marker changed ownership")
                    path.unlink()
                    result[key] = True
            except OSError as exc:
                raise RuntimeError(f"governed instrument handshake cleanup failed: {path}") from exc
        return result


class BandwidthDutyCycleBudget:
    """Sealed cold-load host-transfer budget; size alone never decides overlap."""
    def __init__(self, *, host_bandwidth_bytes_per_s: float,
                 rolling_interval_s: float, budget_fraction: float) -> None:
        if (host_bandwidth_bytes_per_s <= 0 or rolling_interval_s <= 0
                or not 0 < budget_fraction <= 1):
            raise RuntimeError("host-transfer duty-cycle budget is invalid")
        self.host_bandwidth_bytes_per_s = float(host_bandwidth_bytes_per_s)
        self.rolling_interval_s = float(rolling_interval_s)
        self.budget_fraction = float(budget_fraction)

    def admit(self, *, cold_load_host_bytes: int, observed_at_s: float,
              prior_cold_load_bytes: int = 0) -> dict:
        if (isinstance(cold_load_host_bytes, bool) or cold_load_host_bytes < 1
                or isinstance(prior_cold_load_bytes, bool) or prior_cold_load_bytes < 0
                or observed_at_s < 0):
            raise RuntimeError("host-transfer load observation is invalid")
        budget = self.host_bandwidth_bytes_per_s * self.rolling_interval_s * self.budget_fraction
        rolling = prior_cold_load_bytes + cold_load_host_bytes
        return {"schema": "epyc.autokernel.host_transfer_budget.v1",
                "cold_load_host_bytes": cold_load_host_bytes,
                "rolling_interval_s": self.rolling_interval_s,
                "host_bandwidth_bytes_per_s": self.host_bandwidth_bytes_per_s,
                "budget_fraction": self.budget_fraction,
                "rolling_cold_load_bytes": rolling, "budget_bytes": budget,
                "transfer_ratio": rolling / (self.host_bandwidth_bytes_per_s * self.rolling_interval_s),
                "observed_at_s": observed_at_s, "admitted": rolling <= budget}


@dataclass(frozen=True)
class SiteLoadProfile:
    """Reviewed, workload-specific cold-overlap authority; never planner input."""
    policy_version: str
    model_sha256: str
    model_path: str
    model_bytes: int
    workload: str
    calls_per_arm: int
    device_id: str
    worst_case_cold_loads: int
    budget: BandwidthDutyCycleBudget

    def decide(self, *, model: Path, workload: str, calls: int, device_id: str,
               observed_headroom: bool) -> dict:
        actual_sha = sha256_file(model)
        exact = (actual_sha == self.model_sha256 and str(model.resolve()) == self.model_path
                 and model.stat().st_size == self.model_bytes and workload == self.workload
                 and calls == self.calls_per_arm and device_id == self.device_id)
        transfer = self.budget.admit(cold_load_host_bytes=model.stat().st_size,
            observed_at_s=time.monotonic(), prior_cold_load_bytes=model.stat().st_size
            * max(0, self.worst_case_cold_loads - 1))
        if exact and observed_headroom and transfer["admitted"]:
            return {**transfer, "policy_version": self.policy_version, "mode": "cold_overlap",
                    "reason": "exact reviewed site load profile", "lock_interval": None,
                    "residency_transition": "cold_load_required"}
        return {**transfer, "policy_version": self.policy_version, "mode": "cold_serialized",
                "reason": "profile mismatch/missing headroom/transfer budget", "lock_interval": "load_only",
                "residency_transition": "cold_load_required"}


SITE_LOAD_PROFILES = {
    "mi210-qwen05b-tg128-18-v1": SiteLoadProfile(
        policy_version="mi210-qwen05b-tg128-18-v1",
        model_sha256="f175ecace8c24336cbf9e22bd71ea032a16492bd264a3caab6dfa4cafe80ddd3",
        model_path="/mnt/raid0/llm/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q4_K_M.gguf",
        model_bytes=397807840, workload="decode_tg128", calls_per_arm=9,
        device_id=DEVICE_ID, worst_case_cold_loads=18,
        budget=BandwidthDutyCycleBudget(host_bandwidth_bytes_per_s=DEFAULT_HOST_BANDWIDTH_BYTES_S,
                                        rolling_interval_s=60.0, budget_fraction=.01)),
}


def decide_load_mode(*, hot_resident: bool, residency_identity_matches: bool,
                     host_observation_available: bool, transfer: dict,
                     dedicated_window_available: bool,
                     policy_version: str = "site-host-transfer-v1") -> dict:
    """Fail-closed three-mode load admission; planner text cannot select it."""
    if not policy_version or not isinstance(transfer, dict):
        raise RuntimeError("load admission policy input is malformed")
    if hot_resident:
        if not residency_identity_matches:
            raise RuntimeError("hot resident declaration lacks exact model/runtime/residency identity")
        return {"mode": "hot_resident", "policy_version": policy_version,
                "reason": "exact resident model/runtime identity", "cpu_window_required": False}
    if host_observation_available and transfer.get("admitted") is True:
        return {"mode": "cold_overlap", "policy_version": policy_version,
                "reason": "sealed host-transfer policy admitted declared cold load", "cpu_window_required": False}
    if dedicated_window_available:
        return {"mode": "cold_serialized", "policy_version": policy_version,
                "reason": "unknown/over-budget host transfer serialized for load only", "cpu_window_required": True}
    raise RuntimeError("cold GPU load cannot be admitted: no host observation budget or dedicated window")


def host_transfer_admission(*, bytes_per_cold_load: int, cold_loads: int,
                            interval_s: float, host_bandwidth_bytes_s: float,
                            conservative_fraction: float,
                            site_policy_allows_overlap: bool = True,
                            observed_headroom: bool = True,
                            hot_resident: bool = False,
                            resident_identity: str | None = None,
                            expected_identity: str | None = None) -> dict:
    """Compatibility/public policy entry point with an explicit three-mode outcome."""
    budget = BandwidthDutyCycleBudget(host_bandwidth_bytes_per_s=host_bandwidth_bytes_s,
                                      rolling_interval_s=interval_s,
                                      budget_fraction=conservative_fraction)
    transfer = budget.admit(cold_load_host_bytes=bytes_per_cold_load, observed_at_s=time.monotonic(),
                            prior_cold_load_bytes=bytes_per_cold_load * max(0, cold_loads - 1))
    exact_hot = hot_resident and resident_identity is not None and resident_identity == expected_identity
    if exact_hot:
        decision = {"mode": "hot_resident", "reason": "exact resident identity", "lock_interval": None,
                    "residency_transition": "reused"}
    elif site_policy_allows_overlap and observed_headroom and transfer["admitted"]:
        decision = {"mode": "cold_overlap", "reason": "site policy/headroom/duty-cycle admitted",
                    "lock_interval": None, "residency_transition": "cold_load_required"}
    else:
        decision = {"mode": "cold_serialized", "reason": "missing/over-budget/unsafe overlap observation",
                    "lock_interval": "load_only", "residency_transition": "cold_load_required"}
    return {**transfer, "policy_version": "site-host-transfer-v1", "inputs": {
        "site_policy_allows_overlap": site_policy_allows_overlap,
        "observed_headroom": observed_headroom, "hot_resident": hot_resident,
        "resident_identity": resident_identity, "expected_identity": expected_identity}, **decision}


def invoke(*, build: Path, model: Path, seed: int, baseline_vram: int,
           flash_attention: bool, campaign_id: str,
           expected_source_commit: str = SOURCE_COMMIT,
           prompt_tokens: int = 512, generation_tokens: int = 0,
           cpu_journal: cpu_region_claim.RegionClaimJournal,
           threads: int = 8, batch: int = 512, ubatch: int = 512,
           mmap: bool = True, no_op_offload: bool = False,
           split_mode: str = "layer", no_kv_offload: bool = False,
           poll: int = 50, inference_window_lock: Path | None = None,
           reward_binary: Path | None = None, hip_library_dir: Path | None = None,
           common_loader_dir: Path | None = None,
           runtime_arm: str | None = None,
           host_transfer_interval_s: float = 60.0,
           host_bandwidth_bytes_s: float = DEFAULT_HOST_BANDWIDTH_BYTES_S,
           host_transfer_fraction: float = DEFAULT_HOST_TRANSFER_FRACTION,
           cold_loads_in_interval: int = 1,
           sealed_load_decision: dict | None = None,
           repetitions: int = 1,
           load_readiness_policy: LoadReadinessPolicy | None = None,
           ready_continue_handshake: ReadyContinueHandshake | None = None,
           process_factory: Callable[..., Any] | None = None,
           kfd_pid_provider: Callable[[], tuple[int, ...]] | None = None,
           vram_reader: Callable[[], int] | None = None,
           pgid_provider: Callable[[int], int] | None = None,
           sleep: Callable[[float], None] | None = None,
           supervisor_root: Path | None = None) -> dict:
    """Run one cold load and all sealed repetitions for one discovery arm."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise RuntimeError("GPU discovery repetitions must be a positive integer")
    if (not isinstance(sealed_load_decision, dict)
            or sealed_load_decision.get("mode") not in {"cold_overlap", "cold_serialized"}):
        raise RuntimeError("nonpersistent runner requires a sealed cold load decision")
    if sealed_load_decision["mode"] == "cold_overlap":
        model_bytes = model.stat().st_size
        if sealed_load_decision is None:
            raise RuntimeError("GPU overlap requires a preflight-sealed site load decision")
        transfer = sealed_load_decision
        if transfer["mode"] != "cold_overlap":
            raise RuntimeError("GPU cold load was not admitted for overlap; use serialized load mode")
        claims = cpu_region_claim.inspect_region_claims()
        concurrent = []
        for region, entries in (claims.get("regions") or {}).items():
            for entry in entries:
                if not entry.get("held"):
                    continue
                concurrent.append({
                    "region": region, "role": entry.get("role"),
                    "holder_pids": entry.get("holder_pids") or [],
                    "attribution": entry.get("attribution"),
                })
        result = _invoke_locked(
            build=build, model=model, seed=seed, baseline_vram=baseline_vram,
            expected_source_commit=expected_source_commit,
            flash_attention=flash_attention, prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens, threads=threads, ubatch=ubatch,
            batch=batch, mmap=mmap, no_op_offload=no_op_offload,
            split_mode=split_mode, no_kv_offload=no_kv_offload, poll=poll,
            reward_binary=reward_binary, hip_library_dir=hip_library_dir, common_loader_dir=common_loader_dir,
            runtime_arm=runtime_arm, repetitions=repetitions,
            process_factory=process_factory, kfd_pid_provider=kfd_pid_provider,
            vram_reader=vram_reader, pgid_provider=pgid_provider, sleep=sleep,
            supervisor_root=supervisor_root)
        # Deliberately no shared CPU inference-window lock here.  The sealed
        # admission receipt, not model size or caller flags, admits this noise.
        result["inference_call_window"] = None
        result["cpu_coverage"] = {
            "schema": "epyc.autokernel.discovery_cpu_overlap.v1",
            "cpu_overlap_policy": "allowed_discovery_noise",
            "cpu_exclusivity": False, "borrowed": False,
            "model_size_bytes": model_bytes,
            "host_transfer": transfer,
            "load_mode": "cold",
            "concurrent_claims": concurrent,
            "promotion_claim": False,
        }
        result["load_admission_decision"] = sealed_load_decision
        result["load_readiness_transition"] = {
            "schema": "epyc.autokernel.gpu_load_readiness_transition.v1",
            "status": "not_required_cold_overlap",
            "lock_released_before_measurement": True,
        }
        return result
    # JSONL is emitted only after llama-bench completes its repetitions.  It
    # cannot prove a point *before* the first timed sample, so absent an
    # explicit instrument ready/continue barrier we conservatively retain the
    # serialized lock for the complete one-load batched process.
    if load_readiness_policy is not None and (
            load_readiness_policy.model_path != model.resolve()
            or load_readiness_policy.model_sha256 != sha256_file(model)
            or load_readiness_policy.runtime_arm != runtime_arm):
        raise RuntimeError("serialized load readiness policy does not bind this arm/model")
    if ready_continue_handshake is not None and (
            load_readiness_policy is None
            or ready_continue_handshake.arm != runtime_arm
            or ready_continue_handshake.seed != seed
            or ready_continue_handshake.repetitions != repetitions
            or ready_continue_handshake.readiness_policy_sha256 != load_readiness_policy.policy_sha256):
        raise RuntimeError("ready/continue handshake does not bind this serialized arm")
    window = (inference_window.InferenceCallWindow(inference_window_lock, timeout_s=600.0)
              if inference_window_lock is not None else MODEL_CALL_WINDOW)
    configured_lease = window.acquire()
    owned_claim = None
    coverage: Any = None
    coverage_receipt: dict[str, Any] | None = None
    transition: dict[str, Any] = {
        "schema": "epyc.autokernel.gpu_load_readiness_transition.v1",
        "status": "instrument_barrier_unavailable_held_through_process",
        "lock_released_before_measurement": False,
        "required_instrument_capability": "autokernel-ready-continue-v1",
        "readiness_policy": (None if load_readiness_policy is None
                             else load_readiness_policy.to_dict()),
    }

    def release_for_ready(witness: Mapping[str, Any]) -> None:
        nonlocal owned_claim, transition
        if getattr(coverage, "borrowed", False):
            coverage.validate()
        elif owned_claim is not None:
            owned_claim.release()
            owned_claim = None
        configured_lease.release()
        transition = {
            "schema": "epyc.autokernel.gpu_load_readiness_transition.v1",
            "status": "ready_witnessed_lock_released_before_continue",
            "lock_released_before_measurement": True,
            "lock_path": str(configured_lease.path),
            "waited_s": configured_lease.waited_s,
            "witness": dict(witness),
            "readiness_policy": load_readiness_policy.to_dict(),
            "handshake": ready_continue_handshake.to_dict(),
        }

    try:
        try:
            owned_claim = cpu_region_claim.acquire_cpu_region_claim(
                CPU_LIST, purpose="AutoKernel GPU cold-load helper window",
                campaign_id=campaign_id, journal=cpu_journal,
                role="autokernel-gpu-discovery", timeout_s=0, max_hold_s=300)
            coverage = owned_claim
            coverage_receipt = {
                "schema": "epyc.autokernel.owned_cpu_coverage.v1",
                "borrowed": False,
                "claim": owned_claim.receipt().to_dict(),
            }
        except cpu_region_claim.CpuRegionClaimTimeout:
            coverage = inference_window.borrow_windowed_cpu_coverage(CPU_LIST)
            coverage_receipt = coverage.to_dict()
        result = _invoke_locked(
            build=build, model=model, seed=seed, baseline_vram=baseline_vram,
            expected_source_commit=expected_source_commit,
            flash_attention=flash_attention, prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens, threads=threads, ubatch=ubatch,
            batch=batch, mmap=mmap, no_op_offload=no_op_offload,
            split_mode=split_mode, no_kv_offload=no_kv_offload, poll=poll,
            reward_binary=reward_binary, hip_library_dir=hip_library_dir,
            common_loader_dir=common_loader_dir, runtime_arm=runtime_arm,
            repetitions=repetitions, readiness_policy=load_readiness_policy,
            ready_continue_handshake=ready_continue_handshake,
            on_load_ready=(release_for_ready if ready_continue_handshake is not None else None),
            process_factory=process_factory,
            kfd_pid_provider=kfd_pid_provider, vram_reader=vram_reader,
            pgid_provider=pgid_provider, sleep=sleep,
            supervisor_root=supervisor_root)
        if getattr(coverage, "borrowed", False) and configured_lease.held:
            coverage.validate()
    finally:
        if owned_claim is not None:
            try:
                owned_claim.release()
            finally:
                owned_claim = None
        configured_lease.release()
        if ready_continue_handshake is not None:
            cleanup = ready_continue_handshake.cleanup()
            transition = {**transition, "handshake_cleanup": cleanup}
    result["inference_call_window"] = {
        "schema": "epyc.autokernel.inference_call_window.v1",
        "lock_path": str(configured_lease.path),
        "waited_s": configured_lease.waited_s,
        "scope": "one_load_and_all_batched_measurements_no_ready_barrier",
        "released": configured_lease.held is False,
    }
    result["cpu_coverage"] = coverage_receipt or {}
    result["load_mode"] = "cold_serialized"
    result["site_load_decision"] = sealed_load_decision
    result["load_admission_decision"] = sealed_load_decision
    result["load_readiness_transition"] = transition
    return result


def _invoke_locked(*, build: Path, model: Path, seed: int, baseline_vram: int,
                   flash_attention: bool, prompt_tokens: int = 512,
                   expected_source_commit: str = SOURCE_COMMIT,
                   generation_tokens: int = 0, threads: int = 8, ubatch: int = 512,
                   batch: int = 512, mmap: bool = True,
                   no_op_offload: bool = False, split_mode: str = "layer",
                   no_kv_offload: bool = False, poll: int = 50,
                   reward_binary: Path | None = None, hip_library_dir: Path | None = None,
                   common_loader_dir: Path | None = None, runtime_arm: str | None = None,
                   repetitions: int = 1,
                   readiness_policy: LoadReadinessPolicy | None = None,
                   ready_continue_handshake: ReadyContinueHandshake | None = None,
                   on_load_ready: Callable[[Mapping[str, Any]], None] | None = None,
                   process_factory: Callable[..., Any] | None = None,
                   kfd_pid_provider: Callable[[], tuple[int, ...]] | None = None,
                   vram_reader: Callable[[], int] | None = None,
                   pgid_provider: Callable[[int], int] | None = None,
                   sleep: Callable[[float], None] | None = None,
                   max_runtime_s: float = 1800.0,
                   supervisor_root: Path | None = None) -> dict:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise RuntimeError("GPU discovery repetitions must be a positive integer")
    if (isinstance(max_runtime_s, bool) or not isinstance(max_runtime_s, (int, float))
            or not math.isfinite(float(max_runtime_s)) or not 1 <= max_runtime_s <= 3600):
        raise RuntimeError("GPU discovery supervisor deadline is outside reviewed bounds")
    if readiness_policy is not None and (
            runtime_arm != readiness_policy.runtime_arm
            or model.resolve() != readiness_policy.model_path
            or sha256_file(model) != readiness_policy.model_sha256
            or common_loader_dir is None or hip_library_dir is None):
        raise RuntimeError("serialized readiness policy lacks its exact runtime/model closure")
    if (ready_continue_handshake is None) != (on_load_ready is None):
        raise RuntimeError("ready/continue handshake and release callback must be paired")
    if ready_continue_handshake is not None and readiness_policy is None:
        raise RuntimeError("ready/continue handshake requires a typed readiness policy")
    binary = (reward_binary or build / "bin" / "llama-bench").resolve()
    loader_dir = (hip_library_dir or build / "bin").resolve()
    common_dir = (common_loader_dir or binary.parent).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("sealed reward executable is not executable")
    if not loader_dir.is_dir() or not (loader_dir / "libggml-hip.so").is_file():
        raise RuntimeError("sealed HIP loader directory lacks libggml-hip.so")
    argv = ("/usr/bin/taskset", "-c", CPU_LIST, "/usr/bin/numactl", "--interleave=all", str(binary),
            "-m", str(model), "-p", str(prompt_tokens), "-n", str(generation_tokens),
            "-r", str(repetitions), "-ngl", "99",
            "-fa", "on" if flash_attention else "off",
            "-t", str(threads), "-b", str(batch), "-ub", str(ubatch),
            "-mmp", "1" if mmap else "0",
            "-nopo", "1" if no_op_offload else "0", "-sm", split_mode,
            "-nkvo", "1" if no_kv_offload else "0",
            "--poll", str(poll),
            "--autokernel-harden", str(seed),
            *(ready_continue_handshake.argv() if ready_continue_handshake else ()),
            "-o", "jsonl")
    if not common_dir.is_dir():
        raise RuntimeError("sealed common reward loader directory is absent")
    env = {"PATH": "/usr/bin:/bin",
           "LD_LIBRARY_PATH": f"{loader_dir}:{common_dir}:/opt/rocm/lib"}
    factory = subprocess.Popen if process_factory is None else process_factory
    kfd_provider = _kfd_pids if kfd_pid_provider is None else kfd_pid_provider
    pgid = os.getpgid if pgid_provider is None else pgid_provider
    pause = time.sleep if sleep is None else sleep
    def read_vram() -> int:
        if vram_reader is not None:
            return vram_reader()
        try:
            return int(VRAM_USED.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"VRAM residency counter unreadable: {exc}") from exc

    # Real children write to regular files, not PIPEs: llama-bench may emit
    # enough diagnostics to fill a pipe while the supervisor is sampling KFD,
    # which would deadlock the process and retain the shared inference lock.
    if supervisor_root is not None:
        supervisor_root = supervisor_root.resolve()
        supervisor_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_stat = supervisor_root.lstat()
        if (supervisor_root.is_symlink() or root_stat.st_uid != os.getuid()
                or stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise RuntimeError("GPU supervisor output root is not private operation authority")
    output_context = tempfile.TemporaryDirectory(
        prefix="arm-", dir=None if supervisor_root is None else supervisor_root)
    output_root = Path(output_context.name)
    stdout_path, stderr_path = output_root / "stdout", output_root / "stderr"
    stdout_handle = stdout_path.open("w+", encoding="utf-8")
    stderr_handle = stderr_path.open("w+", encoding="utf-8")
    os.chmod(stdout_path, 0o600); os.chmod(stderr_path, 0o600)
    real_process = process_factory is None
    try:
        process = factory(argv, env=env, stdin=subprocess.DEVNULL,
                          stdout=(stdout_handle if real_process else subprocess.PIPE),
                          stderr=(stderr_handle if real_process else subprocess.PIPE),
                          text=True, start_new_session=True)
    except BaseException:
        stdout_handle.close(); stderr_handle.close(); output_context.cleanup()
        raise
    samples = []
    maps_identity = None
    readiness_witness = None
    supervisor_started = time.monotonic()
    teardown: dict[str, Any] = {"required": False, "term_sent": False,
                                "kill_sent": False, "death_proved": False}
    stdout = stderr = ""
    captured_owned: set[int] = set()
    def stop_child() -> None:
        teardown["required"] = True
        if process.poll() is None:
            try:
                if real_process:
                    if os.getpgid(process.pid) != process.pid:
                        raise RuntimeError("GPU discovery child does not own its sealed process group")
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                teardown["term_sent"] = True
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if real_process:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                teardown["kill_sent"] = True
                process.wait(timeout=10)
        if process.returncode is None:
            raise RuntimeError("GPU discovery child remained alive after TERM/KILL teardown")
        remaining = captured_owned.intersection(kfd_provider())
        if remaining:
            raise RuntimeError(f"GPU discovery owned KFD descendants survived teardown: {sorted(remaining)}")
        if real_process:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("GPU discovery process group survived teardown")
        teardown["death_proved"] = True
    try:
      while process.poll() is None:
        if time.monotonic() - supervisor_started > max_runtime_s:
            raise RuntimeError("GPU discovery supervisor deadline exceeded")
        kfd = kfd_provider()
        try:
            vram = read_vram()
        except BaseException:
            raise
        owned = []
        foreign = []
        for pid in kfd:
            try:
                (owned if pgid(pid) == process.pid else foreign).append(pid)
            except (ProcessLookupError, PermissionError):
                continue
        if foreign:
            raise RuntimeError(f"foreign KFD inference overlapped discovery: {foreign}")
        captured_owned.update(owned)
        samples.append({"offset_s": time.monotonic(), "kfd_pids": list(kfd),
                        "owned_kfd_pids": owned, "vram_used_bytes": vram})
        if (maps_identity is None and runtime_arm is not None and owned
                and vram > baseline_vram and common_loader_dir is not None
                and hip_library_dir is not None):
            maps_identity = _runtime_maps_identity(runtime_root=common_loader_dir.parent,
                arm=runtime_arm, model=model, kfd_pid=owned[0])
        if ready_continue_handshake is not None and readiness_witness is None:
            if len(owned) == 1 and vram > baseline_vram and maps_identity is not None:
                assert readiness_policy is not None and on_load_ready is not None
                readiness_policy.validate_witness(maps_identity)
                ready = ready_continue_handshake.validate_ready(pid=process.pid)
                readiness_witness = {
                    "ready": ready, "owned_kfd_pids": list(owned),
                    "vram_used_bytes": vram, "baseline_vram_bytes": baseline_vram,
                    "runtime_maps_identity": maps_identity,
                    "sample_offset_s": samples[-1]["offset_s"],
                }
                # Ordering is the contract: lock/claim release is complete
                # before the token that permits the first timed sample exists.
                on_load_ready(readiness_witness)
                ready_continue_handshake.continue_after_release()
        pause(0.05)
      if real_process:
          stdout_handle.flush(); stderr_handle.flush()
          for artifact in (stdout_path, stderr_path):
              output_stat = artifact.lstat()
              if (artifact.is_symlink() or output_stat.st_uid != os.getuid()
                      or output_stat.st_nlink != 1
                      or stat.S_IMODE(output_stat.st_mode) & 0o077):
                  raise RuntimeError("GPU supervisor output carrier is unsafe")
          stdout_handle.seek(0); stderr_handle.seek(0)
          stdout, stderr = stdout_handle.read(), stderr_handle.read()
      else:
          stdout, stderr = process.communicate(timeout=10)
    except BaseException as original:
        try:
            stop_child()
        except BaseException as teardown_error:
            raise RuntimeError(
                f"GPU discovery teardown failed after {type(original).__name__}: {teardown_error}") from original
        raise
    finally:
        stdout_handle.close(); stderr_handle.close()
        output_context.cleanup()
    if process.returncode != 0:
        raise RuntimeError(f"GPU discovery invocation exited {process.returncode}: {stderr[-2000:]}")
    rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"GPU discovery invocation emitted {len(rows)} rows")
    row = rows[0]
    raw_samples = row.get("samples_ts")
    if (not isinstance(raw_samples, list) or len(raw_samples) != repetitions
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in raw_samples)):
        raise RuntimeError(
            f"GPU discovery invocation requires exactly {repetitions} finite raw samples")
    metric = row.get("avg_ts")
    if (isinstance(metric, bool) or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))):
        raise RuntimeError("GPU discovery invocation emitted a non-finite reward metric")
    if row.get("backends") != "ROCm" or row.get("gpu_info") != "AMD Instinct MI210":
        raise RuntimeError("GPU discovery invocation did not report MI210 ROCm execution")
    reported_commit = str(row.get("build_commit", ""))
    if expected_source_commit is not None and (len(reported_commit) < 7 or not expected_source_commit.startswith(reported_commit)):
        raise RuntimeError("GPU discovery binary does not report the sealed source commit")
    expected_flash = 1 if flash_attention else 0
    if (row.get("n_prompt") != prompt_tokens or row.get("n_gen") != generation_tokens
            or row.get("flash_attn") != expected_flash):
        raise RuntimeError("GPU discovery result differs from the sealed workload frame")
    expected = {
        "n_threads": threads, "n_batch": batch, "n_ubatch": ubatch,
        "use_mmap": mmap, "no_op_offload": 1 if no_op_offload else 0,
        "split_mode": split_mode,
        "no_kv_offload": no_kv_offload,
        "poll": poll,
    }
    mismatched = {key: (row.get(key), value) for key, value in expected.items()
                  if row.get(key) != value}
    if mismatched:
        raise RuntimeError(f"GPU discovery result differs from sealed runtime config: {mismatched}")
    if not any(sample["owned_kfd_pids"] for sample in samples):
        raise RuntimeError("GPU discovery window has no owned KFD residency sample")
    if max(sample["vram_used_bytes"] for sample in samples) <= baseline_vram:
        raise RuntimeError("GPU discovery window has no positive VRAM residency delta")
    if runtime_arm is not None and maps_identity is None:
        raise RuntimeError("GPU discovery window lacks sealed runtime loader-map identity")
    if ready_continue_handshake is not None and readiness_witness is None:
        raise RuntimeError("governed instrument exited without ready before measurement")
    return {"argv": list(argv), "env": {"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]},
            "reward_binary": str(binary), "reward_binary_sha256": sha256_file(binary),
            "hip_library": str(loader_dir / "libggml-hip.so"),
            "hip_library_sha256": sha256_file(loader_dir / "libggml-hip.so"),
            "common_loader_dir": str(common_dir),
            "metric": float(metric), "samples": [float(value) for value in raw_samples],
            "sample_count": repetitions, "seed": seed, "raw_row": row,
            "stderr_tail": stderr[-2000:], "residency": samples,
            "supervisor": {"deadline_s": float(max_runtime_s),
                           "elapsed_s": time.monotonic() - supervisor_started,
                           "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                           "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                           "teardown": teardown, "temporary_output_cleaned": True},
            "runtime_maps_identity": maps_identity,
            "load_readiness_witness": readiness_witness,
            "hip_residency_proved": True}


def factor_spec(*, factor: str, anchor_build: Path, candidate_build: Path,
                anchor_identity: dict, candidate_identity: dict) -> dict:
    """Validate and describe the only difference admitted by this screen."""
    if factor == "mmq_mfma":
        if not anchor_identity["rocwmma_fattn"] or not candidate_identity["rocwmma_fattn"]:
            raise RuntimeError("both MMQ arms must keep ROCWMMA_FATTN=ON")
        if anchor_identity["mmq_mfma"] is not True or candidate_identity["mmq_mfma"] is not False:
            raise RuntimeError("sole factor must be GGML_HIP_MMQ_MFMA ON->OFF")
        return {
            "name": "GGML_HIP_MMQ_MFMA", "anchor": "ON", "candidate": "OFF",
            "anchor_flash_attention": True, "candidate_flash_attention": True,
        }
    if factor == "flash_attention":
        if anchor_build != candidate_build:
            raise RuntimeError("flash_attention screen requires one identical build path for both arms")
        if anchor_identity != candidate_identity:
            raise RuntimeError("flash_attention screen requires identical sealed build identities")
        if not anchor_identity["rocwmma_fattn"] or anchor_identity["mmq_mfma"]:
            raise RuntimeError("flash_attention screen requires the r1m0 build (ROCWMMA ON, MMQ MFMA OFF)")
        return {
            "name": "flash_attention", "anchor": "OFF", "candidate": "ON",
            "anchor_flash_attention": False, "candidate_flash_attention": True,
        }
    if factor == "rocwmma_fattn":
        if anchor_identity["mmq_mfma"] or candidate_identity["mmq_mfma"]:
            raise RuntimeError("both ROCWMMA arms must keep MMQ_MFMA=OFF")
        if anchor_identity["rocwmma_fattn"] is not False or candidate_identity["rocwmma_fattn"] is not True:
            raise RuntimeError("sole factor must be GGML_HIP_ROCWMMA_FATTN OFF->ON")
        return {
            "name": "GGML_HIP_ROCWMMA_FATTN", "anchor": "OFF", "candidate": "ON",
            "anchor_flash_attention": True, "candidate_flash_attention": True,
        }
    if factor == "hip_graphs":
        if anchor_identity["rocwmma_fattn"] != candidate_identity["rocwmma_fattn"]:
            raise RuntimeError("HIP graphs arms must keep ROCWMMA_FATTN identical")
        if anchor_identity["mmq_mfma"] != candidate_identity["mmq_mfma"]:
            raise RuntimeError("HIP graphs arms must keep MMQ_MFMA identical")
        if anchor_identity["hip_graphs"] is not True or candidate_identity["hip_graphs"] is not False:
            raise RuntimeError("sole factor must be GGML_HIP_GRAPHS ON->OFF")
        return {
            "name": "GGML_HIP_GRAPHS", "anchor": "ON", "candidate": "OFF",
            "anchor_flash_attention": True, "candidate_flash_attention": True,
        }
    if factor == "source_patch":
        for key in ("hip_graphs", "rocwmma_fattn", "mmq_mfma"):
            if anchor_identity[key] != candidate_identity[key]:
                raise RuntimeError(
                    f"source patch arms must keep {key} compile setting identical")
        if anchor_identity["source_commit"] == candidate_identity["source_commit"]:
            raise RuntimeError("source patch arms must have distinct source commits")
        return {
            "name": "source_patch",
            "anchor": anchor_identity["source_commit"][:12],
            "candidate": candidate_identity["source_commit"][:12],
            "anchor_flash_attention": True,
            "candidate_flash_attention": True,
        }
    if factor in {"helper_threads", "helper_threads_12", "helper_threads_16",
                  "helper_threads_24", "batch", "batch_up", "ubatch", "ubatch_up",
                  "mmap", "op_offload", "split_row", "kv_offload", "poll_zero"}:
        if anchor_build != candidate_build or anchor_identity != candidate_identity:
            raise RuntimeError(f"{factor} screen requires one identical sealed build")
        configs = {
            "helper_threads": ("gpu_helper_threads", 8, 4),
            "helper_threads_12": ("gpu_helper_threads", 8, 12),
            "helper_threads_16": ("gpu_helper_threads", 8, 16),
            "helper_threads_24": ("gpu_helper_threads", 8, 24),
            "batch": ("batch_size", 512, 256),
            "batch_up": ("batch_size", 512, 1024),
            "ubatch": ("ubatch_size", 512, 256),
            "ubatch_up": ("ubatch_size", 512, 1024),
            "mmap": ("mmap", "ON", "OFF"),
            "op_offload": ("op_offload", "ON", "OFF"),
            "split_row": ("split_mode", "layer", "row"),
            "kv_offload": ("kv_offload", "ON", "OFF"),
            "poll_zero": ("gpu_poll", 50, 0),
        }
        name, anchor, candidate = configs[factor]
        result = {"name": name, "anchor": anchor, "candidate": candidate,
                "anchor_flash_attention": True, "candidate_flash_attention": True,
                "anchor_threads": 8,
                "candidate_threads": (4 if factor == "helper_threads" else
                                      12 if factor == "helper_threads_12" else
                                      16 if factor == "helper_threads_16" else 8),
                "anchor_batch": 512,
                "candidate_batch": (256 if factor == "batch" else
                                    1024 if factor == "batch_up" else 512),
                "anchor_ubatch": 512,
                "candidate_ubatch": (256 if factor == "ubatch" else
                                     1024 if factor == "ubatch_up" else 512),
                "anchor_mmap": True, "candidate_mmap": False if factor == "mmap" else True}
        if factor == "op_offload":
            result["anchor_no_op_offload"] = False
            result["candidate_no_op_offload"] = True
        if factor == "split_row":
            result["anchor_split_mode"] = "layer"
            result["candidate_split_mode"] = "row"
        if factor == "kv_offload":
            result["anchor_no_kv_offload"] = False
            result["candidate_no_kv_offload"] = True
        if factor == "helper_threads_24":
            result["candidate_threads"] = 24
        if factor == "poll_zero":
            result["anchor_poll"] = 50
            result["candidate_poll"] = 0
        return result
    raise RuntimeError(f"unsupported GPU discovery factor: {factor}")


def preflight(args: argparse.Namespace) -> dict:
    model = Path(args.model).resolve()
    anchor_build = Path(args.anchor_build).resolve()
    candidate_build = Path(args.candidate_build).resolve()
    order = tuple(getattr(args, "arm_order_schedule", "anchor,candidate").split(","))
    order_seed = getattr(args, "arm_order_seed_sha256", "0" * 64)
    if (set(order) != {"anchor", "candidate"} or len(order) != 2
            or not isinstance(order_seed, str) or len(order_seed) != 64
            or any(ch not in "0123456789abcdef" for ch in order_seed)):
        raise RuntimeError("GPU discovery arm-order authority is malformed")
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    model_size_bytes = model.stat().st_size
    # Admission is computed once by the sealed deployment lease.  This runner
    # is deliberately only a consumer: it never accepts a CLI profile, a size
    # threshold, or fabricated host headroom as an overlap authority.
    transfer = getattr(args, "load_admission_decision", None)
    if not isinstance(transfer, dict):
        raise RuntimeError("GPU discovery runner requires a sealed load-admission decision")
    try:
        gpu_load_admission.validate_decision_receipt(
            transfer,
            expected_policy_version=getattr(args, "load_admission_policy_version"),
            expected_policy_sha256=getattr(args, "load_admission_policy_sha256"),
            expected_policy_file_sha256=getattr(args, "load_admission_policy_file_sha256"),
            expected_effective_context_sha256=getattr(args, "load_admission_effective_context_sha256"))
    except gpu_load_admission.AdmissionPolicyError as exc:
        raise RuntimeError(f"sealed load-admission decision refused: {exc}") from exc
    request = transfer.get("request")
    if (not isinstance(request, dict) or request.get("model_path") != str(model)
            or request.get("model_sha256") != sha256_file(model)
            or request.get("model_bytes") != model_size_bytes
            or request.get("workload") != args.workload
            or request.get("calls_per_arm") != args.calls
            or request.get("device_id") != getattr(args, "device_id", DEVICE_ID)):
        raise RuntimeError("sealed load-admission decision does not bind this runner frame")
    if getattr(args, "device_id", DEVICE_ID) != DEVICE_ID:
        raise RuntimeError("GPU discovery device must be the admitted MI210")
    configured_lock = getattr(args, "inference_window_lock", None)
    lock = (Path(configured_lock) if configured_lock else MODEL_CALL_WINDOW.path).resolve()
    if lock.is_symlink() or not lock.parent.is_dir():
        raise RuntimeError("configured inference-window lock is unsafe")
    anchor_identity = build_identity(anchor_build)
    candidate_identity = build_identity(candidate_build)
    factor = factor_spec(
        factor=args.factor, anchor_build=anchor_build, candidate_build=candidate_build,
        anchor_identity=anchor_identity, candidate_identity=candidate_identity)
    prompt_tokens, generation_tokens, recipe, metric = (
        (512, 0, "pp512-ngl99", "prefill_tokens_per_s")
        if args.workload == "prefill_pp512"
        else (0, 128, "tg128-ngl99", "decode_tokens_per_s"))
    runtime_arms = None
    if args.factor == "source_patch":
        if not all(getattr(args, key, None) for key in
                   ("measurement_binary", "common_loader_dir", "anchor_loader_dir", "candidate_loader_dir")):
            raise RuntimeError("source patch requires a sealed shared reward runtime closure")
        measurement = Path(args.measurement_binary).resolve()
        anchor_loader = Path(args.anchor_loader_dir).resolve()
        candidate_loader = Path(args.candidate_loader_dir).resolve()
        common_loader = Path(args.common_loader_dir).resolve()
        def hip_object(path: Path) -> tuple[Path, str]:
            link = path / "libggml-hip.so.0"
            if (not (path / "libggml-hip.so").is_symlink() or not link.is_symlink()
                    or (path / "libggml-hip.so").resolve(strict=True) != link.resolve(strict=True)):
                raise RuntimeError("source patch HIP runtime lacks an exact .so/.so.0 topology")
            resolved = link.resolve(strict=True)
            if resolved.parent != path or resolved.is_symlink() or not resolved.is_file():
                raise RuntimeError("source patch HIP SONAME resolves outside its arm runtime")
            return resolved, sha256_file(resolved)
        if (not measurement.is_file() or not os.access(measurement, os.X_OK) or not common_loader.is_dir()
                or not all(path.is_dir() for path in (anchor_loader, candidate_loader))):
            raise RuntimeError("source patch runtime closure is incomplete")
        shared_sha = sha256_file(measurement)
        _anchor_hip_object, anchor_hip = hip_object(anchor_loader)
        _candidate_hip_object, candidate_hip = hip_object(candidate_loader)
        if anchor_hip == candidate_hip:
            raise RuntimeError("source patch runtime closure requires distinct HIP DSOs")
        runtime_arms = {"measurement_binary": str(measurement),
                        "measurement_binary_sha256": shared_sha,
                        "anchor_loader_dir": str(anchor_loader),
                        "candidate_loader_dir": str(candidate_loader),
                        "common_loader_dir": str(common_loader),
                        "anchor_hip_sha256": anchor_hip,
                        "candidate_hip_sha256": candidate_hip,
                        "reward_closure": "shared_anchor_binary_per_arm_hip_dso"}
    requested_handshake = getattr(args, "instrument_ready_continue_v1", False)
    instrument_commit = getattr(args, "instrument_ready_continue_commit", None)
    contract_sha256 = getattr(args, "instrument_ready_continue_contract_sha256", None)
    if requested_handshake and (
            not isinstance(instrument_commit, str)
            or instrument_commit != READY_CONTINUE_INSTRUMENT_COMMIT
            or contract_sha256 != READY_CONTINUE_CONTRACT_SHA256
            or runtime_arms is None
            or anchor_identity["source_commit"] != READY_CONTINUE_INSTRUMENT_COMMIT):
        raise RuntimeError(
            "ready/continue requires the sealed 81bf32f11 instrument, exact contract, "
            "and instrument-derived anchor")
    return {
        "schema": "epyc.autokernel.gpu_discovery_preflight.v1",
        "campaign_id": args.campaign_id,
        "authority": "nonpromotable_candidate_only_discovery",
        "model": str(model),
        "model_sha256": sha256_file(model),
        "model_size_bytes": model_size_bytes,
        "host_transfer": transfer,
        "device_id": DEVICE_ID,
        "inference_window_lock": str(lock),
        "cpu_overlap_policy": ("allowed_discovery_noise" if transfer["mode"] == "cold_overlap"
                               else "cold_serialized_load_window"),
        "promotion_claim": False,
        "non_promotable": True,
        "anchor_build": str(anchor_build),
        "candidate_build": str(candidate_build),
        "anchor_identity": anchor_identity,
        "candidate_identity": candidate_identity,
        "sole_factor": {key: factor[key] for key in ("name", "anchor", "candidate")},
        "anchor_flash_attention": factor["anchor_flash_attention"],
        "candidate_flash_attention": factor["candidate_flash_attention"],
        "anchor_threads": factor.get("anchor_threads", 8),
        "candidate_threads": factor.get("candidate_threads", 8),
        "anchor_batch": factor.get("anchor_batch", 512),
        "candidate_batch": factor.get("candidate_batch", 512),
        "anchor_ubatch": factor.get("anchor_ubatch", 512),
        "candidate_ubatch": factor.get("candidate_ubatch", 512),
        "anchor_mmap": factor.get("anchor_mmap", True),
        "candidate_mmap": factor.get("candidate_mmap", True),
        "anchor_no_op_offload": factor.get("anchor_no_op_offload", False),
        "candidate_no_op_offload": factor.get("candidate_no_op_offload", False),
        "anchor_split_mode": factor.get("anchor_split_mode", "layer"),
        "candidate_split_mode": factor.get("candidate_split_mode", "layer"),
        "anchor_no_kv_offload": factor.get("anchor_no_kv_offload", False),
        "candidate_no_kv_offload": factor.get("candidate_no_kv_offload", False),
        "anchor_poll": factor.get("anchor_poll", 50),
        "candidate_poll": factor.get("candidate_poll", 50),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "frame": recipe,
        "metric": metric,
        "invocations": {"anchor": args.calls, "candidate": args.calls},
        "arm_order_schedule": list(order),
        "arm_order_seed_sha256": order_seed,
        "inference_executed": False,
        "runtime_arms": runtime_arms,
        "serialized_readiness": {
            "required": transfer["mode"] == "cold_serialized",
            "proof": "owned_kfd+positive_vram+exact_split_runtime_maps",
            "available": runtime_arms is not None,
            "ready_continue": {"enabled": bool(requested_handshake),
                               "instrument_commit": instrument_commit,
                               "contract_source_sha256": contract_sha256},
        },
    }


def _readiness_policy_for_arm(*, sealed: Mapping[str, Any], arm: str,
                              model: Path) -> LoadReadinessPolicy | None:
    """Materialize the only authority permitted to release a cold-load lock."""
    runtime_arms = sealed.get("runtime_arms")
    if runtime_arms is None:
        return None
    if not isinstance(runtime_arms, Mapping):
        raise RuntimeError("sealed shared runtime closure is malformed")
    common = runtime_arms.get("common_loader_dir")
    if not isinstance(common, str):
        raise RuntimeError("sealed shared runtime closure lacks its common loader")
    return LoadReadinessPolicy.from_split_runtime(
        runtime_root=Path(common).resolve().parent, runtime_arm=arm, model=model)


def run(args: argparse.Namespace) -> dict:
    sealed = preflight(args)
    started_at = utc_now()
    out = Path(storage.assert_not_scratch(args.output_dir, what="GPU discovery output"))
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "preflight.json", sealed)
    model = Path(sealed["model"])
    anchor_build = Path(sealed["anchor_build"])
    candidate_build = Path(sealed["candidate_build"])
    anchor_identity = sealed["anchor_identity"]
    candidate_identity = sealed["candidate_identity"]
    sole_factor = sealed["sole_factor"]
    anchor_readiness = _readiness_policy_for_arm(
        sealed=sealed, arm="anchor", model=model)
    candidate_readiness = _readiness_policy_for_arm(
        sealed=sealed, arm="candidate", model=model)
    handshake_enabled = bool(sealed["serialized_readiness"]["ready_continue"]["enabled"])
    anchor_handshake = (ReadyContinueHandshake.create(
        root=out / "ready-continue-anchor", decision=sealed["host_transfer"],
        policy=anchor_readiness, arm="anchor", seed=args.seed, repetitions=args.calls)
        if handshake_enabled and anchor_readiness is not None else None)
    candidate_handshake = (ReadyContinueHandshake.create(
        root=out / "ready-continue-candidate", decision=sealed["host_transfer"],
        policy=candidate_readiness, arm="candidate", seed=args.seed + args.calls,
        repetitions=args.calls)
        if handshake_enabled and candidate_readiness is not None else None)
    if _kfd_pids():
        raise RuntimeError("MI210 already has KFD users")
    baseline_vram = int(VRAM_USED.read_text(encoding="utf-8").strip())
    purpose = ("AutoKernel GPU candidate-only discovery "
               f"{sole_factor['name']} {sole_factor['anchor']}->{sole_factor['candidate']}")
    cpu_journal = cpu_region_claim.RegionClaimJournal(args.cpu_claim_journal)
    gpu_journal = device_claim.ClaimJournal(args.device_claim_journal)
    claim_acquirer = getattr(args, "_device_claim_acquirer",
                             device_claim.acquire_device_claim)
    if not callable(claim_acquirer):
        raise RuntimeError("device claim acquirer is not callable")
    gpu = None
    sampler = None
    live_governance = None
    borrowed_phase_end = None
    live_governance_path = out / "live-governance.json"
    try:
        gpu = claim_acquirer(
            DEVICE_ID, purpose=purpose, campaign_id=args.campaign_id,
            journal=gpu_journal, timeout_s=0, max_hold_s=300)
        claim_mode = ("borrowed_outer_reservation"
                      if getattr(gpu, "borrowed_outer_reservation", False)
                      else "direct_device_claim")
        live_governance = {
            "schema": SCHEMA_LIVE_GOVERNANCE,
            "status": "active",
            "campaign_id": args.campaign_id,
            "runner_pid": os.getpid(),
            "authority": "nonpromotable_candidate_only_discovery",
            "cpu_overlap_policy": sealed["cpu_overlap_policy"],
            "model": sealed["model"],
            "model_sha256": sealed["model_sha256"],
            "model_size_bytes": sealed["model_size_bytes"],
            "site_load_decision": sealed["host_transfer"],
            "promotion_claim": False,
            "non_promotable": True,
            "preflight_sha256": schemas.content_hash(sealed),
            "device_claim_open": gpu.receipt().to_dict(),
            "device_claim_mode": claim_mode,
            "started_at": started_at,
        }
        atomic_json(live_governance_path, live_governance)
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        arm_order = tuple(sealed.get("arm_order_schedule", [
            "anchor", "candidate"]))
        if set(arm_order) != {"anchor", "candidate"} or len(arm_order) != 2:
            raise RuntimeError("arm order must contain anchor and candidate exactly once")

        def run_arm(arm: str) -> list[dict]:
            anchor = arm == "anchor"
            prefix = "anchor" if anchor else "candidate"
            identity = anchor_identity if anchor else candidate_identity
            readiness = anchor_readiness if anchor else candidate_readiness
            handshake = anchor_handshake if anchor else candidate_handshake
            return [invoke(
                build=anchor_build if anchor else candidate_build,
                model=model, seed=args.seed if anchor else args.seed + args.calls,
                expected_source_commit=(None if sealed["runtime_arms"]
                                        else identity["source_commit"]),
                baseline_vram=baseline_vram,
                flash_attention=sealed[f"{prefix}_flash_attention"],
                prompt_tokens=sealed["prompt_tokens"],
                generation_tokens=sealed["generation_tokens"],
                threads=sealed[f"{prefix}_threads"],
                ubatch=sealed[f"{prefix}_ubatch"], batch=sealed[f"{prefix}_batch"],
                mmap=sealed[f"{prefix}_mmap"],
                no_op_offload=sealed[f"{prefix}_no_op_offload"],
                split_mode=sealed[f"{prefix}_split_mode"],
                no_kv_offload=sealed[f"{prefix}_no_kv_offload"],
                poll=sealed[f"{prefix}_poll"],
                campaign_id=args.campaign_id, cpu_journal=cpu_journal,
                sealed_load_decision=sealed["host_transfer"],
                inference_window_lock=Path(sealed["inference_window_lock"]),
                reward_binary=(Path(sealed["runtime_arms"]["measurement_binary"])
                               if sealed["runtime_arms"] else None),
                hip_library_dir=(Path(sealed["runtime_arms"][f"{prefix}_loader_dir"])
                                 if sealed["runtime_arms"] else None),
                common_loader_dir=(Path(sealed["runtime_arms"]["common_loader_dir"])
                                   if sealed["runtime_arms"] else None),
                runtime_arm=(arm if sealed["runtime_arms"] else None),
                repetitions=args.calls, load_readiness_policy=readiness,
                ready_continue_handshake=handshake,
                supervisor_root=out / f"supervisor-{arm}")]

        arm_runs = {arm: run_arm(arm) for arm in arm_order}
        anchor_runs = arm_runs["anchor"]
        candidate_runs = arm_runs["candidate"]
        bank_body = {
            "schema": SCHEMA_BANK, "campaign_id": args.campaign_id,
            "status": "complete", "started_at": started_at, "ended_at": utc_now(),
            "authority": "nonpromotable_candidate_only_discovery",
            "frame": {"backend": "llama_gpu", "recipe": sealed["frame"],
                      "metric": sealed["metric"], "metric_direction": "higher_better",
                      "n_prompt": sealed["prompt_tokens"],
                      "n_gen": sealed["generation_tokens"],
                      "model": str(model), "model_sha256": sha256_file(model),
                      "source_commit": candidate_identity["source_commit"], "cpu_list": CPU_LIST,
                      "device": "AMD Instinct MI210", "architecture": "gfx90a"},
            "sole_factor": sole_factor,
            "anchor_invocations": args.calls,
            "anchor_identity": anchor_identity,
            "candidate_identity": candidate_identity,
            "anchor_processes": 1,
            "arm_order_schedule": list(arm_order),
            "arm_order_seed_sha256": sealed.get("arm_order_seed_sha256", "0" * 64),
            "anchor_samples": [sample for run in anchor_runs for sample in run["samples"]],
            "anchor_runs": anchor_runs,
        }
        bank = gpu_beliefs.attach_baseline_beliefs(
            bank_body, producer_path=Path(__file__).resolve())
        atomic_json(out / "baseline-bank.json", bank)
        center = sum(bank["anchor_samples"]) / len(bank["anchor_samples"])
        values = [sample for run in candidate_runs for sample in run["samples"]]
        effects = [(value - center) / center for value in values]
        numeric = sampler.stop().to_dict()
        sampler = None
        if claim_mode == "borrowed_outer_reservation":
            borrowed_phase_end = gpu.release()
            if hasattr(borrowed_phase_end, "to_dict"):
                borrowed_phase_end = borrowed_phase_end.to_dict()
            if (not isinstance(borrowed_phase_end, Mapping)
                    or borrowed_phase_end.get("schema") !=
                    "epyc.autokernel.borrowed_device_claim_phase.v1"
                    or borrowed_phase_end.get("mode") != "borrowed_outer_reservation"
                    or borrowed_phase_end.get("outer_claim_id") !=
                    gpu.receipt().to_dict().get("claim_id")
                    or borrowed_phase_end.get("physical_release") is not False
                    or "released_at" in borrowed_phase_end):
                raise RuntimeError("borrowed throughput phase end is malformed")
        result_body = {
            "schema": SCHEMA_RESULT, "campaign_id": args.campaign_id,
            "status": "complete", "started_at": started_at, "ended_at": utc_now(),
            "authority": "nonpromotable_candidate_only_discovery",
            "state": "decided", "ok": True, "non_promotable": True,
            "nomination": "top_k_candidate_only_not_a_keep",
            "baseline_sha256": bank["baseline_sha256"],
            "anchor_invocations": args.calls, "candidate_invocations": args.calls,
            "anchor_processes": 1, "candidate_processes": 1,
            "arm_order_schedule": list(arm_order),
            "arm_order_seed_sha256": sealed.get("arm_order_seed_sha256", "0" * 64),
            "baseline_center": center, "candidate_samples": values,
            "relative_effects": effects, "median_relative": median(effects),
            "host_noise_policy": "ordinary_host_activity_recorded_not_blocking",
            "cpu_overlap_policy": sealed["cpu_overlap_policy"],
            "model_size_bytes": model.stat().st_size,
            "site_load_decision": sealed["host_transfer"],
            "promotion_claim": False,
            "frame": bank["frame"], "sole_factor": bank["sole_factor"],
            "candidate_identity": bank["candidate_identity"],
            "candidate_runs": candidate_runs, "device_sampling": numeric,
            "hip_residency_proved": all(run["hip_residency_proved"]
                                         for run in anchor_runs + candidate_runs),
            "cpu_coverage_windows": [
                run["cpu_coverage"] for run in anchor_runs + candidate_runs],
            "device_claim_open": gpu.receipt().to_dict(),
            "device_claim_mode": claim_mode,
            **({"device_claim_borrowed_phase_end": dict(borrowed_phase_end)}
               if borrowed_phase_end is not None else {}),
        }
        result = gpu_beliefs.attach_result_beliefs(
            result_body, bank=bank, producer_path=Path(__file__).resolve())
        atomic_json(out / "result.json", result)
        # A derived operator view, kept separate from the strict terminal
        # campaign contract.  The immutable result above is already durable, so
        # an export failure must not erase or reclassify the measurement.
        try:
            autokernel_progression.export_progression()
        except Exception as exc:
            print(f"WARNING: GPU result is durable but progression export failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return result
    finally:
        primary_active = sys.exc_info()[0] is not None
        sampler_error: BaseException | None = None
        if sampler is not None:
            try:
                sampler.stop()
            except BaseException as exc:
                sampler_error = exc
        if gpu is not None:
            ended = (borrowed_phase_end
                     if borrowed_phase_end is not None else gpu.release())
            if hasattr(ended, "to_dict"):
                ended = ended.to_dict()
            if not isinstance(ended, Mapping):
                raise RuntimeError("device claim end did not return a typed receipt")
            if live_governance is not None:
                terminal = {
                    **live_governance,
                    "ended_at": utc_now(),
                }
                if claim_mode == "borrowed_outer_reservation":
                    terminal.update(
                        status="borrowed_phase_ended",
                        device_claim_borrowed_phase_end=dict(ended))
                else:
                    terminal.update(
                        status="released", device_claim_released=dict(ended))
                atomic_json(live_governance_path, terminal)
        if sampler_error is not None and not primary_active:
            raise sampler_error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--anchor-build", required=True)
    result.add_argument("--candidate-build", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", required=True)
    result.add_argument("--factor", choices=("mmq_mfma", "flash_attention", "rocwmma_fattn",
                                             "source_patch",
                                             "hip_graphs", "helper_threads", "helper_threads_12",
                                             "helper_threads_16", "helper_threads_24", "batch",
                                             "batch_up", "ubatch", "ubatch_up", "mmap",
                                             "op_offload", "split_row", "kv_offload", "poll_zero"),
                        default="mmq_mfma")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--preflight-output")
    result.add_argument("--seed", type=int, default=8613)
    result.add_argument("--calls", type=int, choices=(3, 5, 9), default=3,
                        help="fresh invocations per arm (discovery evidence only)")
    result.add_argument("--arm-order-schedule",
                        choices=("anchor,candidate", "candidate,anchor"),
                        default="anchor,candidate")
    result.add_argument("--arm-order-seed-sha256", default="0" * 64)
    result.add_argument("--workload", choices=("prefill_pp512", "decode_tg128"),
                        default="prefill_pp512")
    result.add_argument("--inference-window-lock")
    result.add_argument("--device-id", default=DEVICE_ID)
    result.add_argument("--measurement-binary")
    result.add_argument("--common-loader-dir")
    result.add_argument("--anchor-loader-dir")
    result.add_argument("--candidate-loader-dir")
    result.add_argument("--instrument-ready-continue-v1", action="store_true")
    result.add_argument("--instrument-ready-continue-commit")
    result.add_argument("--instrument-ready-continue-contract-sha256")
    result.add_argument("--cpu-claim-journal", default="/mnt/raid0/llm/ak-claims/region.jsonl")
    result.add_argument("--device-claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        payload = preflight(args) if args.preflight_only else run(args)
        if args.preflight_output:
            atomic_json(Path(args.preflight_output), payload)
    except Exception as exc:
        print(f"GPU discovery REFUSED: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    if args.preflight_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps({key: payload[key] for key in (
            "state", "baseline_center", "candidate_samples", "median_relative",
            "hip_residency_proved", "result_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
