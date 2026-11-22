"""AKU-11: lifecycle provider bound to the host's EXISTING claim primitives.

CPU work holds the orchestrator's physical region owner (`claim.hold_cpu`, role
``bench``); GPU work holds the existing MI210 device flock (`claim.hold`). The
provider owns the open lock descriptions between ``authorize`` and ``release``;
grant identity and generation come from the acquired kernel flocks, never from a
request, argv or configuration.

OFF by default. The registered factory (`make_registry`) installs nothing unless
`ACTIVATION_ENV` names an operator activation record bound to this exact source
digest; without it the startup preflight keeps reporting the provider as
unavailable debt. Containment stays the existing cgroup-v2 `OwnedCgroup`.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Callable, Mapping, Protocol

PROVIDER_ID = "host_claim_v1"
ACTIVATION_ENV = "AUTOKERNEL_HOST_CLAIM_PROVIDER_ACTIVATION"
ACTIVATION_SCHEMA = "epyc.autokernel.host_claim_provider_activation.v1"
_ACTIVATION_FIELDS = {"schema", "provider_id", "source_sha256", "cpu_list", "cgroup_parent"}
GRANT_PREFIX = "hostclaim.v1"
CPU_ROLE = "bench"


def _loop(name: str):
    # The registered factory is exec'd without a package; the supervised
    # interpreter imports the loop under its absolute name.
    return importlib.import_module(f"{__package__ or 'scripts.kernel_rnd.autokernel.loop'}.{name}")


class HostClaimRefused(RuntimeError):
    """The host claim could not be acquired, proven or released as required."""


class LockBusy(HostClaimRefused):
    """Another holder owns the claim; the provider does not queue behind it."""


@dataclass(frozen=True)
class ClaimSpec:
    backend: str
    cpu_list: str = ""
    device_id: str = ""
    device_lock: str = ""
    role: str = ""  # AKU-11c: "" -> CPU_ROLE; "build" folds SMT siblings into regions
    jobs: int = 0  # AKU-11c: build-role parallel jobs cap (1..len(logical_cpus))
    gpu_quiet_exclusive: bool = False  # AKU-11c: GPU phase also holds gpu-quiet EXCLUSIVE

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "gpu"} or (self.backend == "cpu") != bool(self.cpu_list) \
                or (self.backend == "gpu") != bool(self.device_id and self.device_lock):
            raise HostClaimRefused("claim spec is not a closed CPU region or GPU device claim")
        build = self.role == "build"
        if (self.role not in {"", CPU_ROLE, "build"} or (build and self.backend != "cpu")
                or type(self.jobs) is not int or build != (self.jobs > 0)
                or (build and self.jobs > len(self.logical_cpus))
                or (self.gpu_quiet_exclusive and self.backend != "gpu")):
            raise HostClaimRefused("claim spec role/jobs/gpu-quiet phase is not closed")

    @property
    def cpu_role(self) -> str:
        return self.role or CPU_ROLE

    @property
    def logical_cpus(self) -> tuple[int, ...]:
        cpus: set[int] = set()
        for part in filter(None, self.cpu_list.split(",")):
            low, _, high = part.partition("-")
            cpus.update(range(int(low), int(high or low) + 1))
        return tuple(sorted(cpus))


@dataclass(frozen=True)
class LockHandle:
    """Facts of one acquired claim; `token` is the backend's private owner."""

    boot_id: str
    holder_pid: int
    holder_start_ticks: int
    claim_ids: tuple[str, ...]
    generation: int
    region_fraction: float
    gpu_device_ids: tuple[str, ...]
    affinity: tuple[str, ...]
    token: Any = None

    @property
    def grant_id(self) -> str:
        return "|".join((GRANT_PREFIX, self.boot_id, str(self.holder_pid),
                         str(self.holder_start_ticks), ",".join(self.claim_ids)))


def parse_grant_id(grant_id: str) -> tuple[str, int, int, tuple[str, ...]] | None:
    parts = grant_id.split("|") if isinstance(grant_id, str) else []
    try:
        if len(parts) != 5 or parts[0] != GRANT_PREFIX or not parts[4]:
            return None
        claims = tuple(parts[4].split(","))
        if any(len(item.split(":")) != 4 or item.split(":")[0] != parts[1] for item in claims):
            return None
        return parts[1], int(parts[2]), int(parts[3]), claims
    except ValueError:
        return None


class HostLockBackend(Protocol):
    def acquire(self, spec: ClaimSpec, timeout_s: float) -> LockHandle: ...
    def observe(self, handle: LockHandle) -> str: ...  # held | lost | unavailable
    def release(self, handle: LockHandle) -> bool: ...
    def owners(self, claim_ids: tuple[str, ...]) -> tuple[int, ...] | None: ...
    def holder_alive(self, pid: int, start_ticks: int) -> bool: ...
    def current_boot(self) -> str: ...


def flock_owners(raw: str, device: int, inode: int) -> tuple[int, ...]:
    """PIDs /proc/locks lists for an exclusive FLOCK on (device, inode)."""
    owners = []
    for fields in (line.split() for line in raw.splitlines()):
        if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        ident = fields[5].split(":")
        if len(ident) == 3 and (int(ident[0], 16), int(ident[1], 16), int(ident[2])) == (
                os.major(device), os.minor(device), inode):
            owners.append(int(fields[4]))
    return tuple(owners)


class OrchestratorHostLockBackend:
    """Production backend over `claim.hold_cpu` / `claim.hold`; not exercised by tests."""

    #: `hold_cpu` makes one region attempt bounded at 1.0 s; refuse shorter deadlines.
    min_acquire_s = 1.0

    def acquire(self, spec: ClaimSpec, timeout_s: float) -> LockHandle:
        if timeout_s < self.min_acquire_s:
            raise HostClaimRefused("deadline is shorter than the owner's bounded attempt")
        claim = _loop("claim")
        stack = ExitStack()
        try:
            receipt = stack.enter_context(
                claim.hold_cpu(spec.cpu_list, role=spec.cpu_role) if spec.backend == "cpu"
                else claim.hold(Path(spec.device_lock), device_id=spec.device_id))
            opened = receipt.observe()
            if opened["status"] != "held" or not opened["locks"]:
                raise HostClaimRefused(f"acquired claim is not provably held: {opened['error']}")
            locks = list(opened["locks"])
            if spec.gpu_quiet_exclusive:  # AKU-11c: exclusive GPU measurement phase
                quiet = stack.enter_context(claim.hold_gpu_quiet_measurement(
                    max(timeout_s, self.min_acquire_s), retain=True))
                held_quiet = quiet.observe()
                if held_quiet["status"] != "held" or not held_quiet["locks"]:
                    raise HostClaimRefused("gpu-quiet EXCLUSIVE phase is not provably held")
                locks.extend(held_quiet["locks"])
        except BaseException as exc:
            stack.close()
            if isinstance(exc, claim.ClaimRefused) or claim.region_lock_busy(exc):
                raise LockBusy(str(exc)) from exc
            raise
        boot = self.current_boot()
        cpu = spec.backend == "cpu"
        return LockHandle(
            boot, os.getpid(), claim._process_start_ticks(os.getpid()) or 0,
            tuple(f"{boot}:flock:{row['device']}:{row['inode']}" for row in locks),
            getattr(receipt, "_ownership_generation", 1),
            receipt._region_fraction if cpu else 0.0,
            () if cpu else (spec.device_id,), tuple(receipt._affinity), (stack, receipt))

    def observe(self, handle: LockHandle) -> str:
        return handle.token[1].observe()["status"]

    def release(self, handle: LockHandle) -> bool:
        try:
            handle.token[0].close()
        except Exception:
            return False  # e.g. ClaimRefused: the claim did not survive to its close
        return True

    def owners(self, claim_ids: tuple[str, ...]) -> tuple[int, ...] | None:
        try:
            with Path("/proc/locks").open("r", encoding="utf-8") as stream:
                raw = stream.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                return None
            found: list[int] = []
            for item in claim_ids:
                _boot, _kind, device, inode = item.split(":")
                found.extend(flock_owners(raw, int(device), int(inode)))
            return tuple(found)
        except (OSError, ValueError):
            return None

    def holder_alive(self, pid: int, start_ticks: int) -> bool:
        return _loop("claim")._process_start_ticks(pid) == start_ticks

    def current_boot(self) -> str:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


@dataclass
class _Held:
    handle: LockHandle
    authorization: Any
    acquired_at: float
    released_at: float | None = None
    spec: ClaimSpec | None = None


class HostClaimProvider:
    """`worker_lifecycle.TrustedGrantProvider` over a `HostLockBackend`.

    Every call checks its deadline before touching the backend, and backend
    acquisition receives min(remaining, max_acquire_s). Release is always
    attempted (flock release never blocks): refusing it would leak the claim.
    """

    def __init__(self, backend: HostLockBackend, *, specs: Mapping[str, ClaimSpec],
                 container_factory: Callable[[str], Any],
                 clock: Callable[[], float] = time.monotonic,
                 clock_domain: Callable[[], str] | None = None,
                 max_acquire_s: float = 5.0) -> None:
        self._wl = _loop("worker_lifecycle")
        self.backend, self.specs = backend, dict(specs)
        self.container_factory, self.clock = container_factory, clock
        self.clock_domain = clock_domain or self._wl.monotonic_clock_domain
        self.max_acquire_s = max_acquire_s
        self._held: dict[str, _Held] = {}
        self._closed: dict[str, _Held] = {}
        self._attempted: set[str] = set()
        self._released: set[tuple[str, int]] = set()

    def readiness(self) -> tuple[bool, str | None]:
        try:
            _loop("claim")._ensure_orchestrator_importable()
        except Exception as exc:
            return False, f"orchestrator claim owner unavailable: {exc}"
        return True, None

    def _exact(self, authorization: Any, table: Mapping[str, _Held]) -> _Held | None:
        held = table.get(getattr(authorization, "container_id", None))
        grant = getattr(authorization, "grant", None)
        if held is None or grant is None:
            return None
        mine = held.authorization.grant
        return held if (mine.grant_id, mine.generation) == (grant.grant_id, grant.generation) \
            else None

    def authorize(self, request: Any, container_id: str, deadline: float):
        wl = self._wl
        remaining = deadline - self.clock()
        binding = request.accounting_binding
        spec = None if binding is None else self.spec_for(binding)
        reason = ("provider deadline expired before acquisition" if remaining <= 0
                  else "no enrolled host claim for this accounting backend" if spec is None
                  else None)
        if reason is not None:
            return wl.AuthorizationDenied(request.request_id, container_id, reason)
        if container_id in self._attempted:
            raise HostClaimRefused("container identity was already used for an acquisition")
        self._attempted.add(container_id)
        try:
            handle = self.backend.acquire(spec, min(remaining, self.max_acquire_s))
        except LockBusy as exc:
            return wl.AuthorizationDenied(request.request_id, container_id,
                                          f"host claim busy: {exc}")
        acquired_at = self.clock()
        try:
            grant = wl.GrantReceipt(handle.grant_id, handle.generation, deadline,
                                    self.clock_domain())
            authorization = wl.AuthorizedLaunch(
                grant, container_id, self.container_factory(container_id))
        except BaseException:
            self.backend.release(handle)
            raise
        self._held[container_id] = _Held(handle, authorization, acquired_at, spec=spec)
        return authorization

    def spec_for(self, binding: Any) -> ClaimSpec | None:
        """AKU-11c: a stage-class spec ("cpu:build", "gpu:search") wins over the backend's."""
        return self.specs.get(f"{binding.backend}:{binding.scheduler_stage_class}",
                              self.specs.get(binding.backend))

    def build_jobs(self, authorization: Any) -> int:
        """The parallel-jobs cap of a held build-role claim; refused for any other claim."""
        held = self._exact(authorization, self._held)
        if held is None or held.spec is None or held.spec.role != "build":
            raise HostClaimRefused("no held build-role claim for this authorization")
        return held.spec.jobs

    def refresh(self, authorization: Any, deadline: float):
        grant = authorization.grant
        held = self._exact(authorization, self._held)
        status = ("unknown" if held is None else "late" if self.clock() >= deadline
                  else self.backend.observe(held.handle))
        if status == "held" and self.clock() > deadline:
            status = "late"
        return self._wl.GrantReceipt(grant.grant_id, grant.generation, grant.deadline,
                                     grant.clock_domain, status in {"unknown", "lost"},
                                     status == "held")

    def release(self, authorization: Any, deadline: float) -> bool:
        del deadline  # always attempted; see class docstring
        grant = authorization.grant
        held = self._exact(authorization, self._held)
        if held is None:
            return (grant.grant_id, grant.generation) in self._released
        if not self.backend.release(held.handle):
            return False
        held.released_at = self.clock()
        self._closed[authorization.container_id] = self._held.pop(authorization.container_id)
        self._released.add((grant.grant_id, grant.generation))
        return True

    def inspect_pending(self, identity: Any, deadline: float):
        wl = self._wl
        held = self._held.get(identity.container_id)
        if held is not None and self.clock() < deadline:
            return wl.PendingAcquisitionInspection(
                "exact", held.authorization, "provider holds this acquisition's claim")
        if identity.container_id in self._attempted and held is None:
            return wl.PendingAcquisitionInspection(
                "absent", None, "this provider's atomic acquisition holds no claim")
        return wl.PendingAcquisitionInspection(
            "unknown", None, "no in-process record; a prior claim cannot be proven released")

    def inspect(self, grant: Any, container_id: str, deadline: float):
        wl = self._wl

        def result(status: str, reason: str, authorization: Any = None):
            if status != "unknown" and self.clock() > deadline:
                status, reason, authorization = "unknown", "inspection exceeded deadline", None
            return wl.RecoveryInspection(status, authorization, reason)

        if self.clock() >= deadline:
            return result("unknown", "deadline expired before inspection")
        held = self._held.get(container_id)
        if held is not None and (held.authorization.grant.grant_id,
                                 held.authorization.grant.generation) \
                == (grant.grant_id, grant.generation):
            status = self.backend.observe(held.handle)
            return (result("exact", "provider still holds the exact claim", held.authorization)
                    if status == "held" else result("unknown", f"held claim observed {status}"))
        if (grant.grant_id, grant.generation) in self._released:
            return result("absent_released", "released by this provider")
        parsed = parse_grant_id(grant.grant_id)
        if parsed is None:
            return result("unknown", "grant identity is not a host claim identity")
        boot, pid, ticks, claim_ids = parsed
        try:
            if boot != self.backend.current_boot():
                return result("absent_released", "kernel flocks do not survive a reboot")
            if self.backend.holder_alive(pid, ticks):
                return result("unknown", "original holder is alive; ownership is not transferable")
            owners = self.backend.owners(claim_ids)
        except Exception as exc:
            return result("unknown", f"lock state unprovable: {type(exc).__name__}")
        if owners is None:
            return result("unknown", "kernel lock table is unreadable")
        if owners:
            return result("unknown", f"claimed inodes still held by pids {sorted(set(owners))}")
        return result("absent_released", "holder exited and no flock remains on the claim")

    def close_held_receipt(self, *, authorization: Any, request: Any, worker_id: str,
                           worker_generation: int, container_identity: Any,
                           lifecycle_started_at: float, released_at: float,
                           deadline: float):
        del container_identity
        wl, scheduling = self._wl, _loop("scheduling")
        held = self._exact(authorization, self._closed)
        binding = request.accounting_binding
        if self.clock() > deadline:
            raise wl.WaitingAuthority("held receipt deadline expired")
        if held is None or binding is None:
            raise wl.WaitingAuthority("no provider-held released claim with an accounting binding")
        # The lifecycle charges [its start, its release]; the real held interval
        # must lie inside that envelope or the receipt is refused.
        if not lifecycle_started_at <= held.acquired_at < held.released_at <= released_at:
            raise wl.WaitingAuthority("real held interval lies outside the charged interval")
        grant, handle = held.authorization.grant, held.handle
        receipt = scheduling.HeldClaimReceipt(
            schema=scheduling.RECEIPT_SCHEMA_V2,
            receipt_id="host-claim:" + hashlib.sha256(
                f"{grant.grant_id}#{grant.generation}#{worker_generation}".encode()).hexdigest(),
            proposal_id=binding.proposal_id, backend=binding.backend,
            stage_class=binding.scheduler_stage_class,
            started_at=lifecycle_started_at, ended_at=released_at,
            ownership_generation=worker_generation, allocation_generation=grant.generation,
            physical_claim_ids=handle.claim_ids, physical_region_fraction=handle.region_fraction,
            gpu_device_ids=handle.gpu_device_ids, memory_reservation_bytes=0,
            affinity_cores=handle.affinity, beneficiary_shares={binding.proposal_id: 1.0})
        del self._closed[authorization.container_id]
        return wl.TrustedHeldClaimReceipt(
            request.request_id, request.plan_digest, worker_id, worker_generation,
            grant.grant_id, grant.generation, authorization.container_id,
            grant.clock_domain, lifecycle_started_at, released_at, receipt)

    def describe_active_observation_claim(self, *, authorization: Any, request: Any,
                                          unit_id: str, process_generation_id: str,
                                          deadline: float) -> Mapping[str, Any]:
        wl = self._wl
        held = self._exact(authorization, self._held)
        binding = request.accounting_binding
        spec = None if binding is None else self.spec_for(binding)
        if (held is None or spec is None or not spec.logical_cpus or self.clock() >= deadline
                or self.backend.observe(held.handle) != "held"):
            raise wl.WaitingAuthority("no provable held CPU claim to describe")
        grant = held.authorization.grant
        row = {"schema": wl.ACTIVE_OBSERVATION_CLAIM_SCHEMA, "grant_id": grant.grant_id,
               "grant_generation": grant.generation,
               "container_id": authorization.container_id,
               "held_claim": {"logical_cpus": list(spec.logical_cpus),
                              "gpu_devices": sorted(held.handle.gpu_device_ids)},
               "active_claim_ref": f"{grant.grant_id}#{unit_id}#{process_generation_id}"}
        return {**row, "claim_digest": wl._digest(row)}


def activation(environ: Mapping[str, str] | None = None) -> tuple[dict | None, str]:
    """The operator activation record, or (None, reason). Absent by default."""
    raw = (os.environ if environ is None else environ).get(ACTIVATION_ENV)
    if not raw:
        return None, "activation_absent"
    path = Path(raw)
    try:
        info = path.lstat()
        if (not path.is_absolute() or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid() or info.st_mode & 0o022 or info.st_size > 65536):
            return None, "activation_record_untrusted"
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "activation_record_unreadable"
    if (not isinstance(record, dict) or set(record) != _ACTIVATION_FIELDS
            or record["schema"] != ACTIVATION_SCHEMA or record["provider_id"] != PROVIDER_ID
            or not all(isinstance(record[key], str) and record[key] for key in record)):
        return None, "activation_record_invalid"
    if record["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        return None, "activation_source_differs"
    return record, "active"


def cgroup2_parent(value: str) -> Path:
    """A uid-owned writable directory beneath the exact cgroup-v2 mount."""
    root = Path("/sys/fs/cgroup")
    mounts = [line.partition(" - ") for line in
              Path("/proc/self/mountinfo").read_text(encoding="ascii").splitlines()]
    if not any(left.split()[4:5] == [str(root)] and right.split()[:1] == ["cgroup2"]
               for left, _, right in mounts):
        raise HostClaimRefused("/sys/fs/cgroup is not the cgroup-v2 mount")
    parent = Path(value).resolve(strict=True)
    if (root not in parent.parents and parent != root) or not parent.is_dir() \
            or parent.stat().st_uid != os.getuid() or not os.access(parent, os.W_OK):
        raise HostClaimRefused("cgroup parent is not an owned writable cgroup-v2 directory")
    return parent


def make_registry():
    """Registered no-argument factory: empty registry unless explicitly activated."""
    standalone = _loop("standalone_inputs")
    record, _reason = activation()
    if record is None:
        return standalone.ProviderRegistry({})
    claim = _loop("claim")
    secure = importlib.import_module(
        claim.__name__.rsplit(".", 2)[0] + ".controller.discovery_supervisor_secure")
    provider = HostClaimProvider(
        OrchestratorHostLockBackend(),
        specs={"cpu": ClaimSpec("cpu", cpu_list=record["cpu_list"]),
               "gpu": ClaimSpec("gpu", device_id=claim.DEVICE_ID,
                                device_lock=str(claim.DEVICE_LOCK))},
        container_factory=functools.partial(
            secure.OwnedCgroup, base=cgroup2_parent(record["cgroup_parent"])))
    return standalone.ProviderRegistry({PROVIDER_ID: standalone.ProviderBinding(
        lifecycle_provider=provider, readiness_check=provider.readiness)})
