"""CTC-1: derive per-stage resource footprints from recipes; keep each target's adapter.

A footprint is a DESCRIPTION of what one stage of one target will occupy. It is
not a claim, a grant or a receipt: the flock/provider claim stays the ownership
fact (claim.py), and admission is CTC-2. Everything here is read from records
that already exist -- the enrolled `TargetRevision`, its `CanonicalResolvedRecipe`
and the campaign `ResourceRequest` -- so there is no second model registry, and
no model, quant or recipe NAME takes part in any decision below. `conflicts()`
reads only physical regions, devices and the build slot.

Two things this module deliberately keeps:

* the target's existing execution adapter (CPU serving, GPU serving, or the
  historical GPU screen -> confirm -> serving-gate route), read from the child
  argv the roster already produced, never re-chosen for scheduling convenience;
* the recipe's explicit capability refusal: an unsupported mixed CPU/GPU draft or
  multi-device launch is refused here with its original reason codes rather than
  being relabelled GPU-only.

Logical SMT siblings fold onto physical cores through the existing topology
helper (`execution.cpu_region_claim`), so a GPU host helper on logical 184 and a
CPU worker on 88 resolve to the same physical core and the same region.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..execution import cpu_region_claim as topo
from . import claim, legacy_targets

#: Stage kinds from the CTC design §4 table. Remote planning/edits take no footprint.
STAGE_KINDS = ("build", "cold_load", "profile", "correctness", "calibration",
               "measure", "anchor", "maintenance")
RUNTIME_KINDS = frozenset(STAGE_KINDS) - {"build"}

ADAPTER_CPU_SERVING = "cpu_serving"
ADAPTER_GPU_SERVING = "gpu_serving"
ADAPTER_GPU_LEGACY_SCREEN = "gpu_legacy_screen"

#: `--max-active-targets` default. Values above 1 need stage admission (CTC-2/3).
DEFAULT_MAX_ACTIVE_TARGETS = 1


class FootprintRefused(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def target_identity(target) -> str:
    """Stable target identity: the same full-revision digest serial_roster scopes stores by."""
    return _digest(target.to_dict())


def _option(argv: Sequence[str], name: str) -> Optional[str]:
    found = None
    for index, arg in enumerate(argv):
        if arg == name and index + 1 < len(argv):
            found = argv[index + 1]
        elif arg.startswith(name + "="):
            found = arg[len(name) + 1:]
    return found


@dataclass(frozen=True)
class ExecutionAdapter:
    """The target's existing workflow, exactly as its child argv selects it."""

    route: str
    instrument: Optional[str]
    screen_scope: Optional[str]
    pairs: Optional[str]
    serving_pairs: Optional[str]

    def to_dict(self) -> dict:
        return {"route": self.route, "instrument": self.instrument,
                "screen_scope": self.screen_scope, "pairs": self.pairs,
                "serving_pairs": self.serving_pairs}


def execution_adapter(argv: Sequence[str]) -> ExecutionAdapter:
    """Read (never choose) the adapter. A GPU target without a serving launch keeps
    the historical screen route; it is not migrated to the unprofiled serving route."""
    argv = list(argv)
    cpu, gpu = _option(argv, "--cpu-serving-launch"), _option(argv, "--gpu-serving-launch")
    if cpu and gpu:
        raise FootprintRefused("child argv selects both CPU and GPU serving launches")
    if cpu:
        route, instrument = ADAPTER_CPU_SERVING, _option(argv, "--serving-instrument") or "legacy_v1"
    elif gpu:
        route, instrument = ADAPTER_GPU_SERVING, None
    else:
        if _option(argv, "--model") is None:
            raise FootprintRefused("legacy GPU screen argv names no --model")
        route, instrument = ADAPTER_GPU_LEGACY_SCREEN, None
    return ExecutionAdapter(route, instrument, _option(argv, "--cpu-screen-scope"),
                            _option(argv, "--pairs"), _option(argv, "--serving-pairs"))


@dataclass(frozen=True)
class StageFootprint:
    kind: str
    target: str
    attempt: str
    adapter: ExecutionAdapter
    source: Optional[str]
    build: Optional[str]
    instrument: Optional[str]
    recipe: Optional[str]
    cpus: frozenset
    cores: frozenset
    regions: tuple
    memory: Optional[str]
    devices: tuple
    build_jobs: Optional[int]

    def to_dict(self) -> dict:
        return {"schema": "epyc.autokernel.stage_footprint.v1", "kind": self.kind,
                "target": self.target, "attempt": self.attempt,
                "adapter": self.adapter.to_dict(), "source": self.source,
                "build": self.build, "instrument": self.instrument, "recipe": self.recipe,
                "cpus": topo.render_cpu_list(self.cpus) if self.cpus else "",
                "cores": sorted(self.cores), "regions": list(self.regions),
                "memory": self.memory, "devices": list(self.devices),
                "build_jobs": self.build_jobs}


def _fold(cpus: frozenset, sibling_map) -> tuple[frozenset, tuple]:
    cores = topo.physical_cores(cpus, sibling_map)
    return cores, topo.cores_to_regions(cores)


def _memory_policy(topology_prefix: Sequence[str]) -> Optional[str]:
    prefix = list(topology_prefix)
    return prefix[1] if len(prefix) >= 2 and prefix[0] == "numactl" else None


def _devices(launch) -> tuple:
    device = launch.template.device
    if device in (None, "none"):
        return ()
    # Only the installed ROCm0 route has a physical claim (legacy_targets enforces it).
    return (claim.DEVICE_ID,) if device == "ROCm0" else (device,)


def runtime_footprint(kind: str, target, launch, resources, *, adapter: ExecutionAdapter,
                      attempt: str, source: Optional[str] = None,
                      environment: Optional[Mapping[str, str]] = None,
                      sibling_map: Optional[Mapping] = None) -> StageFootprint:
    """Footprint of a stage that runs the recipe's instrument (load/profile/measure...).

    The recipe's taskset CPU list pins the main model, drafter and every helper
    thread the server spawns, so it is the stage's CPU footprint -- not the whole
    declared allocation. Containment in owned resources is validated first."""
    if kind not in RUNTIME_KINDS:
        raise FootprintRefused(f"{kind!r} is not a runtime stage kind")
    backend = target.execution.backend
    if launch.backend != backend:
        raise FootprintRefused("launch backend differs from the enrolled target")
    capability = launch.capability
    if not capability.supported:
        codes = ",".join(sorted(reason.code for reason in capability.reasons))
        raise FootprintRefused(f"recipe capability is unsupported: {codes}")
    try:
        legacy_targets.validate_resources(resources, launch, backend=backend,
                                          environment=environment)
    except legacy_targets.TargetSelectionRefused as exc:
        raise FootprintRefused(str(exc)) from exc
    if not launch.template.cpu_list:
        raise FootprintRefused("recipe declares no CPU affinity; footprint is unknowable")
    cpus = topo.parse_cpu_list(launch.template.cpu_list)
    cores, regions = _fold(cpus, sibling_map)
    return StageFootprint(kind, target_identity(target), attempt, adapter, source,
                          launch.build_dir, adapter.instrument, launch.execution_digest,
                          cpus, cores, regions, _memory_policy(launch.topology_prefix),
                          _devices(launch), None)


def build_footprint(target, resources, *, adapter: ExecutionAdapter, attempt: str,
                    build: str, source: Optional[str] = None,
                    sibling_map: Optional[Mapping] = None) -> StageFootprint:
    """Configure/compile/link: the declared build allocation and job bound."""
    cpus = frozenset(resources.cpu_logical)
    if not cpus:
        raise FootprintRefused("build requires declared host CPU resources")
    if resources.build_jobs < 1:
        raise FootprintRefused("build requires a positive build_jobs bound")
    cores, regions = _fold(cpus, sibling_map)
    return StageFootprint("build", target_identity(target), attempt, adapter, source, build,
                          None, None, cpus, cores, regions, None, (), resources.build_jobs)


def conflicts(a: StageFootprint, b: StageFootprint) -> tuple:
    """Why two stages cannot run together: shared regions, devices or the build slot.

    Identity-free on purpose: target ids, model paths and recipe names never enter."""
    reasons = [f"region:{r}" for r in sorted(set(a.regions) & set(b.regions))]
    reasons += [f"device:{d}" for d in sorted(set(a.devices) & set(b.devices))]
    if a.kind == "build" and b.kind == "build":
        reasons.append("build_slot")
    return tuple(reasons)


def add_max_active_targets(parser) -> None:
    parser.add_argument("--max-active-targets", type=int, default=DEFAULT_MAX_ACTIVE_TARGETS,
                        help="concurrent target children (default 1; >1 needs CTC stage admission)")


def check_max_active_targets(parser, args) -> None:
    value = getattr(args, "max_active_targets", DEFAULT_MAX_ACTIVE_TARGETS)
    if value < 1:
        parser.error("--max-active-targets must be >= 1")
    if value > 1:
        parser.error("--max-active-targets > 1 needs stage admission (CTC-2/CTC-3), "
                     "which is not installed yet")
