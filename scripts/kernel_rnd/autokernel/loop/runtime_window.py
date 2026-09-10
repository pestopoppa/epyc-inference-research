"""Original direct-serving window facts, using the installed host/claim predicates."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading

from .. import schemas
from ..execution import microbench
from ..execution.cpu_region_claim import parse_cpu_list
from ..resource import preflight
from . import observation_binding as ob, residency, search_window as sw

# Same installed reference as execution.live_controls.NOMINAL_KHZ. This is a
# declared reference, not a frequency measured off the current server or max_freq.
DEFAULT_NOMINAL_KHZ = 2_500_000
NOMINAL_REFERENCE = "autokernel.execution.live_controls:NOMINAL_KHZ=2500000"


def configuration(store, held, *, storage_floor_bytes_free, nominal_khz=DEFAULT_NOMINAL_KHZ):
    return sw.InstalledSearchWindowConfiguration(
        claim_root=str(preflight.default_region_lock_dir()), proc_root="/proc",
        sysfs_cpu_root="/sys/devices/system/cpu", storage_root=str(store.root),
        cpu_regions=tuple(held["regions"]),
        host_policy=microbench.HostStatePolicy(nominal_khz=nominal_khz),
        storage_floor_bytes_free=storage_floor_bytes_free, expected_source_digest=sw.source_digest(),
        limits=sw.WindowLimits(cadence_s=1.0, max_sample_duration_s=0.1,
            max_gap_s=2.0, max_samples=8192))


def snapshot(config, cpus, *, census, marker):
    original = sw.BoundedWindowSnapshot(config, cpus, census=census, marker=marker)
    while not original.complete:
        original.step()
    return original.body()


def boundary(config, held, *, marker):
    cpus = tuple(sorted(parse_cpu_list(held["cpu_list"])))
    raw = snapshot(config, cpus, census=True, marker=marker)
    result = sw.captured_preflight(raw, self_pid=os.getpid(), scope=preflight.PreflightScope(
        label="original direct CPU serving window", cpu_regions=frozenset(held["regions"]),
        protocol_id="P-AK-SEARCH-1"))
    return {"snapshot": raw, "preflight": result.to_dict(), "claim": held.observe()}


def artifacts(pair):
    """Actual executable/DSO identities at the two owning window boundaries.

    No model bytes are read. Model identity is the original enrolled subject;
    these reads establish that the runtime-only arms still use their exact build.
    """
    rows = []
    for arm in ("anchor", "candidate"):
        recipe = getattr(pair, arm)
        for item in (recipe.executable, *recipe.dsos):
            try:
                with open(item.path, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if before.st_size > 512 * 1024 * 1024:
                        raise ValueError("executable/DSO exceeds finite 512 MiB identity bound")
                    digest = hashlib.sha256()
                    remaining = before.st_size
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("executable/DSO shrank during bounded identity read")
                        digest.update(chunk)
                        remaining -= len(chunk)
                    if stream.read(1):
                        raise ValueError("executable/DSO grew during bounded identity read")
                    digest = digest.hexdigest()
                    after = os.fstat(stream.fileno())
                stable = all(getattr(before, key) == getattr(after, key)
                    for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
                rows.append({"arm": arm, "expected": item.to_dict(), "sha256": digest,
                             "stable": stable, "error": None})
            except (OSError, ValueError) as exc:
                rows.append({"arm": arm, "expected": item.to_dict(), "sha256": None,
                             "stable": False, "error": f"{type(exc).__name__}: {exc}"})
    return rows


class DirectHeldClaimAdapter:
    """Original direct context -> the existing microbench HeldClaim protocol.

    No RegionClaimReceipt is manufactured. Every attestation rereads the actual
    owner's kernel lock facts and retains them in the original artifact store.
    """
    def __init__(self, owner, *, cpu_list, store):
        from .claim import HeldCpuClaim
        if type(owner) is not HeldCpuClaim or owner["device_id"] != "cpu":
            raise TypeError("direct historical control requires its actual acquired CPU owner")
        if not parse_cpu_list(cpu_list).issubset(parse_cpu_list(owner["cpu_list"])):
            raise ValueError("historical control footprint is outside the original CPU claim")
        self.owner, self.cpu_list, self.store = owner, cpu_list, store
        self.claim_id = "direct-loop:" + owner._context_id
        self.observations = []

    def attest(self):
        if len(self.observations) >= 16384:
            raise ValueError("original direct claim attestation capacity exhausted")
        raw = self.owner.observe()
        reference = self.store.write("direct-held-observation", {
            "claim_id": self.claim_id, "cpu_list": self.cpu_list, "observation": raw})
        self.observations.append(reference.to_dict())
        outcome = (schemas.PASS if raw["status"] == "held" else schemas.FAIL
                   if raw["status"] == "lost" else schemas.COULD_NOT_CHECK)
        return microbench.ClaimAttestation(self.claim_id, f"pid:{raw['owner_pid']}",
            self.cpu_list, datetime.fromtimestamp(raw["observed_at"], timezone.utc).isoformat(),
            schemas.Check(outcome, (f"original kernel lock observation {reference.locator}#sha256={reference.sha256}",)))


class DuringWork:
    """Reuse original lifecycle hooks; sample only while the request is running.

    Endpoint reads never become under-load observations. CPU facts retain their
    original permissions-only meaning; no page placement is inferred here.
    """
    def __init__(self, config, recipe):
        self.config = config
        actual = recipe.template.cpu_list
        self.cpus = tuple(sorted(parse_cpu_list(actual))) if actual else tuple(sorted(os.sched_getaffinity(0)))
        self.lifecycle = residency.CpuLifecycleSampler()
        self._lock = threading.Lock()
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = None
        self._phase, self._generation = "setup", 0
        self._target = None
        self._bytes = 0
        self.samples, self.errors = [], []

    def start(self, phase="setup"):
        self.lifecycle.start(phase)
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def phase(self, phase):
        self.lifecycle.phase(phase)
        with self._lock:
            self._phase, self._generation = phase, self._generation + 1
        self._wake.set()

    def attach_target(self, pid):
        self.lifecycle.attach_target(pid)
        self._target = dict(self.lifecycle.observation["target"])

    def checkpoint(self, marker):
        self.phase(marker)

    def note_hook_failure(self, method, error):
        self.errors.append(f"{method}: {type(error).__name__}: {error}")
        self.lifecycle.note_hook_failure(method, error)

    def _collect(self):
        while not self._stop.is_set():
            with self._lock:
                phase, generation = self._phase, self._generation
            if phase == "measurement":
                try:
                    if len(self.samples) >= self.config.limits.max_samples:
                        self.errors.append("during-work host sample capacity exhausted")
                        return
                    raw = snapshot(self.config, self.cpus, census=False, marker=phase)
                    target = self._target
                    if target is not None:
                        process = Path(self.config.proc_root) / str(target["pid"])
                        before = self.lifecycle._identity(process, target["pid"])
                        mapped = sw.read_raw_file(process / "maps", self.config.limits.max_read_bytes)
                        after = self.lifecycle._identity(process, target["pid"])
                        raw["loaded_maps"] = {"target": target, "before_start_ticks": before,
                                              "after_start_ticks": after, "file": mapped}
                    with self._lock:
                        inside = (phase, generation) == (self._phase, self._generation)
                    self._bytes += len(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode())
                    if self._bytes > 32 * 1024 * 1024:
                        self.errors.append("during-work host byte capacity exhausted")
                        return
                    self.samples.append({"snapshot": raw, "phase_contained": inside})
                except Exception as exc:
                    self.errors.append(f"during-work host read: {type(exc).__name__}: {exc}")
            self._wake.wait(self.config.limits.cadence_s)
            self._wake.clear()

    def finish(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self.errors.append("during-work observer did not finish")
        self.lifecycle.finish()

    @property
    def shutdown_resolved(self):
        return ((self._thread is None or not self._thread.is_alive())
                and self.lifecycle.observation["shutdown_resolved"])

    def body(self):
        return {"nominal_reference": NOMINAL_REFERENCE,
            "host_policy": asdict(self.config.host_policy), "cpus": list(self.cpus),
            "samples": list(self.samples), "errors": list(self.errors),
            "cpu_lifecycle": self.lifecycle.observation}


def health(body):
    policy = microbench.HostStatePolicy(**body["host_policy"])
    cpus = tuple(body["cpus"])
    checks = []
    for row in body["samples"]:
        raw = row["snapshot"]
        if row["phase_contained"] and raw["complete"] and not raw["errors"]:
            _classification, check = policy.frequency_verdict(sw.host_state(raw, cpus), under_load=True)
            checks.append(check)
    if body["errors"] or len(checks) < 2:
        checks.append(schemas.Check(schemas.COULD_NOT_CHECK,
            ("fewer than two successful during-work host observations or capture error", *body["errors"])))
    return schemas.Check.worst_of(checks)


def launch_health(body, responses, *, max_gap_s):
    """Existing host predicate plus coverage of this original HTTP measurement."""
    checks = [health(body)]
    measured = [row for row in responses if row["raw"]["phase"] == "measurement"]
    if not measured:
        checks.append(schemas.Check(schemas.COULD_NOT_CHECK, ("original measured request interval absent",)))
        return schemas.Check.worst_of(checks)
    start = min(row["raw"]["started_monotonic_s"] for row in measured)
    end = max(row["raw"]["ended_monotonic_s"] for row in measured)
    samples = [row["snapshot"] for row in body["samples"] if row["phase_contained"]
        and start <= row["snapshot"]["started"] <= row["snapshot"]["ended"] <= end]
    covered = (len(samples) >= 2 and samples[0]["started"] - start <= max_gap_s
        and end - samples[-1]["ended"] <= max_gap_s
        and all(b["started"] - a["ended"] <= max_gap_s for a, b in zip(samples, samples[1:])))
    checks.append(schemas.Check(schemas.PASS if covered else schemas.COULD_NOT_CHECK,
                               ("original during-request host coverage",)))
    return schemas.Check.worst_of(checks)


def source_identity():
    from . import lifecycle_observation as lo
    return ob._freeze({"functions": [lo.callable_identity(function) for function in (
        configuration, snapshot, boundary, artifacts, DuringWork.__init__, DuringWork._collect,
        DirectHeldClaimAdapter.__init__, DirectHeldClaimAdapter.attest,
        DuringWork.start, DuringWork.attach_target, DuringWork.checkpoint,
        DuringWork.note_hook_failure, DuringWork.shutdown_resolved.fget,
        DuringWork.phase, DuringWork.finish, DuringWork.body, health, launch_health,
        microbench.HostStatePolicy.frequency_verdict)],
        "nominal_reference": NOMINAL_REFERENCE, "default_nominal_khz": DEFAULT_NOMINAL_KHZ,
        "snapshot_source": ob._plain(sw.source_identity())})
