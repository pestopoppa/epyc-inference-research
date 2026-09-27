"""Cross-role contention matrix loader + policy decisions.

Phase A of `handoffs/active/cross-role-bw-aware-routing.md`.

Loads `orchestration/contention_matrix.yaml`, exposes per-pair throughput
ratios, and translates `(role_a, role_b, traffic_class)` into a `PairDecision`
that the admission gate (Phase B) acts on.

Design points:

- Pair keys are sorted to make `(A, B)` and `(B, A)` interchangeable.
- Missing matrix file or stale topology hash returns a `MatrixStatus` the
  caller surfaces to the dashboard; the gate fails open for foreground
  runtime and blocks background campaigns (per handoff line 113).
- The default contention floor is 0.85 (`CONTENTION_RATIO_FLOOR`) — values
  at or above this are concurrency-positive enough to allow.
- `topology_fingerprint(NUMA_CONFIG)` produces a deterministic sha256 hash
  of the (role, instances) topology so re-bench detection is automatic
  when NUMA_CONFIG changes.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.scheduling.device_model import (
    DeviceClass,
    DeviceResolutionError,
    resolve_role_device,
)

log = logging.getLogger("scheduling.contention")

# Default contention floor — pair ratios at or above this are "allow", below
# are "block" for background traffic. Foreground traffic has additional rules
# based on traffic class + SLO budget (see pair_policy).
CONTENTION_RATIO_FLOOR = 0.85

# Matrix is considered stale (independent of topology hash) after this many
# days. Re-bench workflow surfaces the warning even if topology hasn't changed.
MATRIX_STALENESS_DAYS = 30

DEFAULT_MATRIX_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "contention_matrix.yaml"
)


class MatrixStatus(str, enum.Enum):
    """High-level state of the on-disk contention matrix."""

    OK = "ok"
    MISSING = "missing"
    STALE = "stale"  # topology hash mismatch or age > MATRIX_STALENESS_DAYS
    INVALID = "invalid"  # parse error / schema violation


class TrafficClass(str, enum.Enum):
    """Request classification used by `pair_policy`."""

    FOREGROUND_INTERACTIVE = "foreground_interactive"
    FOREGROUND_SPECIALIST = "foreground_specialist"
    BACKGROUND = "background"
    MAINTENANCE = "maintenance"


class PairDecision(str, enum.Enum):
    """Output of `pair_policy(role_a, role_b, traffic_class)`."""

    ALLOW = "allow"
    QUEUE = "queue"  # block-then-retry; caller decides timeout
    DEGRADED_ALLOW = "degraded_allow"  # foreground SLO-override; metric-flagged
    BLOCK = "block"  # hard block (only used when nothing better is possible)


@dataclass(frozen=True)
class Pair:
    roles: tuple[str, str]
    ratio: float
    verdict: str  # raw verdict string from YAML; "allow"/"borderline"/"block"
    samples: int = 1
    note: str = ""


@dataclass(frozen=True)
class InstancePair:
    """One within-role instance-pair measurement (a co-runs with b).

    Labels match the YAML: typically "full" (the role's primary multi-region
    instance — could be a true full or a half depending on NUMA_CONFIG) or
    "q0".."q3" (single-quarter instances).
    """

    a: str
    b: str
    ratio: float = 0.0
    verdict: str = ""  # raw verdict from YAML
    cv: float = 0.0


@dataclass(frozen=True)
class SameRole:
    role: str
    verdict: str  # "allow" / "block" / "n/a"
    note: str = ""
    instance_pairs: tuple[InstancePair, ...] = ()


@dataclass(frozen=True)
class SameRoleCertification:
    """Freshness certificate for one same-role live-shape subset.

    This is intentionally narrower than the matrix-level `topology_hash`.
    A role-scoped certification can authorize a background EvalTower fan-out
    on the exact live ports it measured while the older full cross-role matrix
    remains stale/fail-closed for broader scheduling.
    """

    role: str
    topology_hash: str
    verdict: str
    measured_at: str = ""
    mode: str = ""
    live_ports: tuple[int, ...] = ()
    samples: int = 1
    min_ratio: float = 0.0
    max_cv: float = 0.0
    artifact: str = ""
    affinity_artifact: str = ""
    note: str = ""


@dataclass(frozen=True)
class Nway:
    """A measured N-way (>=2 role) cross-role active set (quarter-level)."""

    roles: tuple[str, ...]  # sorted
    ratio: float
    verdict: str  # "allow" / "borderline" / "block"
    cv: float = 0.0
    samples: int = 1
    contains_heavy: bool = False


@dataclass
class ContentionMatrix:
    """In-memory contention matrix. Use `load_contention_matrix` to construct."""

    version: int
    measured_at: str  # ISO 8601
    host: str
    topology_hash: str
    default_floor: float
    pairs: dict[tuple[str, str], Pair] = field(default_factory=dict)  # sorted-tuple keys
    same_role: dict[str, SameRole] = field(default_factory=dict)
    same_role_certifications: dict[str, SameRoleCertification] = field(default_factory=dict)
    unknown_pairs: list[tuple[str, str]] = field(default_factory=list)
    n_way: dict[tuple[str, ...], Nway] = field(default_factory=dict)  # sorted-role-tuple keys
    light_roles: frozenset[str] = field(default_factory=frozenset)
    heavy_roles: frozenset[str] = field(default_factory=frozenset)

    def get_pair(self, role_a: str, role_b: str) -> Pair | None:
        return self.pairs.get(_sorted_pair_key(role_a, role_b))

    def is_unknown_pair(self, role_a: str, role_b: str) -> bool:
        return _sorted_pair_key(role_a, role_b) in {tuple(p) for p in self.unknown_pairs}

    def get_same_role(self, role: str) -> SameRole | None:
        return self.same_role.get(role)

    def get_same_role_certification(self, role: str) -> SameRoleCertification | None:
        return self.same_role_certifications.get(role)

    def get_nway(self, roles) -> Nway | None:
        """Exact-match lookup for an N-way active set (order-independent)."""
        return self.n_way.get(tuple(sorted(set(roles))))


# ────────────────────────────────────────────────────────────────────
# Loading + validation
# ────────────────────────────────────────────────────────────────────


def _sorted_pair_key(role_a: str, role_b: str) -> tuple[str, str]:
    """Canonicalize (A, B) and (B, A) to the same key."""
    return tuple(sorted([role_a, role_b]))  # type: ignore[return-value]


def load_contention_matrix(path: Path | None = None) -> ContentionMatrix:
    """Parse the YAML matrix into a ContentionMatrix dataclass.

    Raises FileNotFoundError if missing, ValueError on schema problems.
    Callers should use `matrix_status()` first to handle missing/stale
    gracefully (fail-open per handoff line 113).
    """
    import yaml  # imported here so missing yaml doesn't import-break the module

    path = path or DEFAULT_MATRIX_PATH
    if not path.exists():
        raise FileNotFoundError(f"contention matrix not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"matrix YAML root is not a mapping: {path}")

    pairs: dict[tuple[str, str], Pair] = {}
    for entry in data.get("pairs", []) or []:
        roles = entry.get("roles")
        if not isinstance(roles, list) or len(roles) != 2:
            raise ValueError(f"pair entry missing roles: {entry}")
        key = _sorted_pair_key(roles[0], roles[1])
        pairs[key] = Pair(
            roles=key,
            ratio=float(entry.get("ratio", 0.0)),
            verdict=str(entry.get("verdict", "")),
            samples=int(entry.get("samples", 1)),
            note=str(entry.get("note", "")),
        )

    same_role: dict[str, SameRole] = {}
    for entry in data.get("same_role", []) or []:
        role = entry.get("role")
        if not role:
            raise ValueError(f"same_role entry missing role: {entry}")
        raw_pairs = entry.get("instance_pairs") or []
        pairs_parsed: list[InstancePair] = []
        for ip in raw_pairs:
            if not isinstance(ip, dict):
                continue
            a, b = ip.get("a"), ip.get("b")
            if not a or not b:
                continue
            pairs_parsed.append(
                InstancePair(
                    a=str(a),
                    b=str(b),
                    ratio=float(ip.get("ratio", 0.0)),
                    verdict=str(ip.get("verdict", "")),
                    cv=float(ip.get("cv", 0.0)),
                )
            )
        same_role[role] = SameRole(
            role=role,
            verdict=str(entry.get("verdict", "")),
            note=str(entry.get("note", "")),
            instance_pairs=tuple(pairs_parsed),
        )

    same_role_certifications: dict[str, SameRoleCertification] = {}
    for entry in data.get("same_role_certifications", []) or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if not role:
            raise ValueError(f"same_role_certification entry missing role: {entry}")
        live_ports_raw = entry.get("live_ports") or []
        live_ports: list[int] = []
        if isinstance(live_ports_raw, list):
            for port in live_ports_raw:
                try:
                    live_ports.append(int(port))
                except (TypeError, ValueError):
                    continue
        same_role_certifications[str(role)] = SameRoleCertification(
            role=str(role),
            topology_hash=str(entry.get("topology_hash", "")),
            verdict=str(entry.get("verdict", "")),
            measured_at=str(entry.get("measured_at", "")),
            mode=str(entry.get("mode", "")),
            live_ports=tuple(sorted(live_ports)),
            samples=int(entry.get("samples", 1)),
            min_ratio=float(entry.get("min_ratio", 0.0) or 0.0),
            max_cv=float(entry.get("max_cv", 0.0) or 0.0),
            artifact=str(entry.get("artifact", "")),
            affinity_artifact=str(entry.get("affinity_artifact", "")),
            note=str(entry.get("note", "")),
        )

    unknown_pairs: list[tuple[str, str]] = []
    for entry in data.get("unknown_pairs", []) or []:
        roles = entry.get("roles")
        if isinstance(roles, list) and len(roles) == 2:
            unknown_pairs.append(_sorted_pair_key(roles[0], roles[1]))

    n_way: dict[tuple[str, ...], Nway] = {}
    for entry in data.get("n_way", []) or []:
        roles = entry.get("roles")
        if not isinstance(roles, list) or len(roles) < 2:
            continue
        key = tuple(sorted(roles))
        n_way[key] = Nway(
            roles=key,
            ratio=float(entry.get("ratio", 0.0)),
            verdict=str(entry.get("verdict", "")),
            cv=float(entry.get("cv", 0.0)),
            samples=int(entry.get("samples", 1)),
            contains_heavy=bool(entry.get("contains_heavy", False)),
        )

    return ContentionMatrix(
        version=int(data.get("version", 1)),
        measured_at=str(data.get("measured_at", "")),
        host=str(data.get("host", "")),
        topology_hash=str(data.get("topology_hash", "")),
        default_floor=float(data.get("default_floor", CONTENTION_RATIO_FLOOR)),
        pairs=pairs,
        same_role=same_role,
        same_role_certifications=same_role_certifications,
        unknown_pairs=unknown_pairs,
        n_way=n_way,
        light_roles=frozenset(data.get("nway_light_roles", []) or []),
        heavy_roles=frozenset(data.get("nway_heavy_roles", []) or []),
    )


def matrix_status(
    path: Path | None = None,
    current_topology_hash: str | None = None,
    max_age_days: int = MATRIX_STALENESS_DAYS,
) -> MatrixStatus:
    """Cheap pre-load check the orchestrator startup can call.

    - MISSING: file doesn't exist
    - INVALID: file exists but can't be parsed
    - STALE: topology hash differs OR file mtime older than max_age_days
    - OK: parses, topology matches (if provided), within age window
    """
    path = path or DEFAULT_MATRIX_PATH
    if not path.exists():
        return MatrixStatus.MISSING
    try:
        m = load_contention_matrix(path)
    except (ValueError, FileNotFoundError):
        return MatrixStatus.INVALID
    except Exception as exc:  # noqa: BLE001 — YAML parse can throw various
        log.warning("matrix parse failed: %s", exc)
        return MatrixStatus.INVALID

    if current_topology_hash is not None and m.topology_hash != current_topology_hash:
        return MatrixStatus.STALE

    try:
        mtime = path.stat().st_mtime
        import time

        age_days = (time.time() - mtime) / 86400.0
        if age_days > max_age_days:
            return MatrixStatus.STALE
    except OSError:
        pass

    return MatrixStatus.OK


def contention_ratio(
    role_a: str, role_b: str, matrix: ContentionMatrix | None = None
) -> float | None:
    """Return the measured parallel/sequential ratio for the pair, or None
    if unmeasured.  Same-role queries return None — use `same_role_verdict`."""
    if role_a == role_b:
        return None
    if matrix is None:
        matrix = load_contention_matrix()
    pair = matrix.get_pair(role_a, role_b)
    return pair.ratio if pair else None


def pair_policy(
    role_a: str,
    role_b: str,
    traffic_class: TrafficClass | str = TrafficClass.FOREGROUND_INTERACTIVE,
    matrix: ContentionMatrix | None = None,
    floor: float | None = None,
) -> PairDecision:
    """Translate (role_a, role_b, traffic_class) into an admission decision.

    Rules (handoff lines 105-115):
      - ratio >= 1.0           → ALLOW
      - floor <= ratio < 1.0   → ALLOW for foreground; QUEUE for background
      - ratio < floor          → QUEUE for all; foreground may DEGRADED_ALLOW
      - unknown pair           → QUEUE for background; ALLOW (with warning) for foreground
      - same role              → consult `same_role` verdict in matrix
    """
    if isinstance(traffic_class, str):
        try:
            traffic_class = TrafficClass(traffic_class)
        except ValueError:
            log.warning("unknown traffic_class %r — defaulting to background", traffic_class)
            traffic_class = TrafficClass.BACKGROUND

    if matrix is None:
        try:
            matrix = load_contention_matrix()
        except FileNotFoundError:
            # Fail-open for foreground (per handoff); block background
            log.warning("contention matrix missing — fail-open for foreground")
            if traffic_class == TrafficClass.BACKGROUND:
                return PairDecision.QUEUE
            return PairDecision.ALLOW

    eff_floor = floor if floor is not None else matrix.default_floor

    if role_a == role_b:
        sr = matrix.get_same_role(role_a)
        if sr is None or sr.verdict in ("allow", "n/a", ""):
            return PairDecision.ALLOW
        if sr.verdict == "block":
            # Same-role explicitly blocked. Background queues; foreground gets
            # degraded-allow with a metric so the operator knows the gate had
            # to override.
            if traffic_class == TrafficClass.BACKGROUND:
                return PairDecision.QUEUE
            return PairDecision.DEGRADED_ALLOW
        return PairDecision.ALLOW

    pair = matrix.get_pair(role_a, role_b)
    if pair is None:
        if matrix.is_unknown_pair(role_a, role_b):
            # Explicitly acknowledged unknown — block background, allow foreground
            if traffic_class == TrafficClass.BACKGROUND:
                return PairDecision.QUEUE
            return PairDecision.ALLOW
        # Silently unknown pair (not even in unknown_pairs) — same policy
        log.debug("unknown pair %s+%s (not in matrix)", role_a, role_b)
        if traffic_class == TrafficClass.BACKGROUND:
            return PairDecision.QUEUE
        return PairDecision.ALLOW

    if pair.ratio >= 1.0:
        return PairDecision.ALLOW
    if pair.ratio >= eff_floor:
        # Borderline — ALLOW foreground, QUEUE background
        if traffic_class == TrafficClass.BACKGROUND:
            return PairDecision.QUEUE
        return PairDecision.ALLOW
    # Below floor — QUEUE for all, foreground can override to DEGRADED_ALLOW
    if traffic_class in (TrafficClass.FOREGROUND_INTERACTIVE, TrafficClass.FOREGROUND_SPECIALIST):
        # Caller decides whether to wait or call DEGRADED_ALLOW; default to QUEUE
        # so the gate's queueing logic gets a chance. The caller can promote
        # to DEGRADED_ALLOW based on SLO budget.
        return PairDecision.QUEUE
    return PairDecision.QUEUE


def nway_policy(
    roles,
    traffic_class: TrafficClass | str = TrafficClass.FOREGROUND_INTERACTIVE,
    matrix: ContentionMatrix | None = None,
    floor: float | None = None,
) -> PairDecision:
    """Admission decision for an N-way (>=2 distinct roles) cross-role active set.

    Pairwise `pair_policy` is necessary but NOT sufficient — an all-pairwise-allowed
    set could in principle be aggregate-negative, so this is a DEFENSIVE gate. (History:
    {frontdoor, ingest, vision} once read 0.847 BLOCK and was cited as proof of
    pairwise!=N-way — but that was a bad-affinity artifact; on certified disjoint quarters
    it is 1.731 ALLOW. Per the 2026-05-26 certified re-bench there is currently NO measured
    N-way block.) This consults the measured `n_way` matrix for the EXACT active set, then applies
    a policy for unmeasured sets:
      * measured allow      -> ALLOW
      * measured borderline -> ALLOW (foreground) / QUEUE (background)
      * measured block      -> QUEUE (serialize; aggregate-negative)
      * unmeasured, all roles BW-light + quartered -> ALLOW (anchored by the measured
        4-way 1.605x + within-role 1.88-2.86x; covers mixed multi-instance light sets)
      * unmeasured, contains a heavy full/half instance -> QUEUE (fail-closed)
    """
    # Launcher-only lanes are physical placements, not new model/contention
    # classes. ``eval_batch_frontdoor`` deliberately stays out of the measured
    # matrix, but its disjoint co-residency verdict must inherit the frontdoor
    # measurements rather than fail closed forever as an unknown heavy role.
    # Physical overlap is decided before this function from the auxiliary lane's
    # own regions, so this alias changes policy identity only—not lock ownership.
    policy_aliases = {"eval_batch_frontdoor": "frontdoor"}
    rs = tuple(sorted({policy_aliases.get(str(role), str(role)) for role in roles}))
    if len(rs) < 2:
        return PairDecision.ALLOW
    if isinstance(traffic_class, str):
        try:
            traffic_class = TrafficClass(traffic_class)
        except ValueError:
            traffic_class = TrafficClass.BACKGROUND
    if matrix is None:
        try:
            matrix = load_contention_matrix()
        except FileNotFoundError:
            return (
                PairDecision.QUEUE
                if traffic_class == TrafficClass.BACKGROUND
                else PairDecision.ALLOW
            )

    is_background = traffic_class == TrafficClass.BACKGROUND

    entry = matrix.get_nway(rs)
    if entry is not None:
        if entry.verdict == "allow":
            return PairDecision.ALLOW
        if entry.verdict == "borderline":
            return PairDecision.QUEUE if is_background else PairDecision.ALLOW
        # block: measured aggregate-negative -> serialize (foreground caller may
        # promote to DEGRADED_ALLOW on SLO budget, mirroring pair_policy below-floor).
        return PairDecision.QUEUE

    # Unmeasured active set. An all-light quartered set is allow-by-policy (anchored
    # by the measured 4-way 1.605x + within-role 1.88-2.86x; covers mixed light sets).
    # Everything else fails OPEN for foreground runtime, CLOSED for background/bulk —
    # matching pair_policy + the cross-role-bw handoff (fail-open foreground, block
    # background campaigns). The heavy-vs-light split only changes the background path.
    if matrix.light_roles and all(r in matrix.light_roles for r in rs):
        return PairDecision.ALLOW
    return PairDecision.QUEUE if is_background else PairDecision.ALLOW


# ────────────────────────────────────────────────────────────────────
# Part B (shape-keyed-contention-gating): placement-aware admission.
#
# SCAFFOLDING ONLY — `admit_set` is intentionally UNUSED by runtime. The
# gate (`contention_gate.py`) and seeder (`seeding_eval.py`) are NOT yet
# rewired to call it; that is a separate, explicitly-gated step. This block
# adds the pure decision function + its value types so the logic can be
# unit-proven against the existing matrix without touching live dispatch.
#
# Why this exists: `pair_policy`/`nway_policy` key on bare ROLE names, but the
# physics is per-instance-SHAPE. The same role pair has two true, opposite
# matrix entries — `frontdoor+ingest` overlapping node0-half primaries = 0.37
# block, the SAME pair on disjoint quarters = 1.716 allow. A role-keyed lookup
# cannot tell them apart. `admit_set` disambiguates by computing overlap from
# canonical CPU-region SETS (never a shape label), then delegating certified
# smallest-disjoint placements to the role-set `nway_policy`.
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Placement:
    """A proposed or in-flight role instance, keyed by the CANONICAL CPU
    regions it occupies — not by a human shape label like "full"/"q0".

    `regions` is a frozenset of atomic-region ids (e.g. {"q0","q1"} for a
    node0-half, {"q0","q1","q2","q3"} for a whole-machine full). Source of
    truth is `src.runtime.instance_topology.get_instance_regions()`.

    `device_class` is the DEVICE axis (rider §2). It decides WHICH resource the
    placement claims exclusively:

      * `DeviceClass.CPU` — claims its regions exclusively (today's rule).
      * `DeviceClass.GPU` — claims NO CPU regions. Its host lane is shareable
        light work (tokenise, sample, marshal) while the weights sit in VRAM
        under `-ngl`, so overlapping a CPU decode's cpuset is not a conflict.
      * `None` — UNRESOLVED. Treated conservatively as CPU-exclusive, which is
        exactly today's behaviour, so every existing caller is unchanged. This
        is not a "default to CPU": the RESOLVER (`device_model.resolve_role_device`)
        never guesses — it raises. `None` here means the caller did not ask.

    An empty region set means "placement unknown" for a CPU placement —
    admit_set fails closed for background and falls back to the legacy
    role-keyed `pair_policy` for foreground. For a RESOLVED GPU placement an
    empty region set is not unknown at all, it is the correct answer: the
    canonical GPU host lane `184-191` is SMT-siblings-only, so it maps to zero
    atomic regions. Conflating those two was the live-path half of the defect.
    """

    role: str
    regions: frozenset[str] = frozenset()
    device_class: DeviceClass | None = None


def claims_cpu_regions(p: Placement) -> bool:
    """Does this placement hold its CPU regions EXCLUSIVELY?

    True for CPU-device placements and for unresolved ones (conservative).
    False only for a placement positively resolved as GPU.
    """
    return p.device_class is not DeviceClass.GPU


def placements_overlap(a: Placement, b: Placement) -> bool:
    """True iff two placements share any physical region. Overlap is a set
    intersection over canonical regions — the shape's NAME is irrelevant
    (frontdoor "full" = {q0,q1} is disjoint from vision "full" = {q2,q3}).

    Physical overlap ONLY. Whether an overlap is a *conflict* is a device
    question — see `placements_conflict`."""
    return bool(a.regions & b.regions)


def placements_conflict(a: Placement, b: Placement) -> bool:
    """True iff the two placements cannot co-run on the same cores.

    Region overlap is necessary but not sufficient: if EITHER side is a
    resolved GPU placement, the overlap is a shared host lane rather than
    contended DRAM bandwidth, and the pair does not conflict. Two GPU roles
    sharing one host lane — which `architect_critic` (the 27B, since the
    2026-09-27 ARCHITECT SWAP; `architect_general` before) and `worker_vision` do
    right now on `184-191` — are likewise not in conflict here; what they
    actually contend for is VRAM, a capacity constraint that belongs to the
    feasibility/capacity artifact, not to this region-overlap predicate.
    """
    if not claims_cpu_regions(a) or not claims_cpu_regions(b):
        return False
    return placements_overlap(a, b)


def placement_for_instance(
    role: str,
    topology_idx: int,
    instance_regions: dict[tuple[str, int], frozenset[str]] | None = None,
    *,
    device_class: DeviceClass | None = None,
) -> Placement:
    """Build a `Placement` for (role, topology_idx) from the canonical
    instance→regions map. If `instance_regions` is None it is loaded live
    from `instance_topology.get_instance_regions()`. Unknown (role, idx)
    yields an empty-region Placement (→ admit_set treats it as unknown).

    `device_class` is passed through unresolved by default; callers that want
    the device axis resolve it themselves (see `seam_admit`) so this helper
    stays pure and cannot raise."""
    if instance_regions is None:
        try:
            from src.runtime.instance_topology import get_instance_regions

            instance_regions = get_instance_regions()
        except Exception:  # noqa: BLE001 — keep pure/import-safe for tests
            instance_regions = {}
    return Placement(
        role=role,
        regions=instance_regions.get((role, topology_idx), frozenset()),
        device_class=device_class,
    )


def admit_set(
    active_placements,
    candidate_placement: Placement,
    traffic_class: TrafficClass | str = TrafficClass.FOREGROUND_INTERACTIVE,
    matrix: ContentionMatrix | None = None,
    floor: float | None = None,
) -> PairDecision:
    """Placement-aware admission: may `candidate_placement` join the set of
    currently-active `active_placements`?

    SCAFFOLDING — not yet called by the gate or seeder.

    Decision (shape-keyed AND device-keyed):
      1. **CPU-region conflict** — if the candidate's regions intersect ANY
         active placement's regions AND both sides claim their regions
         exclusively (i.e. neither is a resolved GPU placement), the two cannot
         co-run on the same cores. Return QUEUE (serialize; the holder must
         release first). This is the case a role-keyed gate gets wrong by
         reading a stale primary-overlap ratio. A GPU candidate is NOT queued
         behind a CPU holder on cpuset overlap alone — its host lane is
         shareable light work, not contended DRAM bandwidth.
      2. **Disjoint** — delegate the role-set verdict to the authoritative
         `nway_policy` over the union of roles. This path assumes the caller has
         supplied A's certified smallest-disjoint placements (quarters where
         supported, otherwise the smallest disjoint shape); `nway_policy` is
         called UNCHANGED.
      3. **Unknown placement** — if the candidate (or any active placement)
         claims CPU regions but has no region info, overlap is undecidable;
         background fails closed immediately, while foreground uses the legacy
         role-keyed `pair_policy` fallback against each active role. A RESOLVED
         GPU placement with no regions is not undecidable — it is a correct
         "claims no CPU region" — so it does not trigger this branch.

    Returns a `PairDecision`; callers map QUEUE→wait/serialize as today.
    """
    if isinstance(traffic_class, str):
        try:
            traffic_class = TrafficClass(traffic_class)
        except ValueError:
            traffic_class = TrafficClass.BACKGROUND
    is_background = traffic_class == TrafficClass.BACKGROUND

    active = tuple(active_placements)
    if not active:
        return PairDecision.ALLOW

    # (3) Unknown placement → fail closed for background. Undecidable overlap
    # means we cannot trust the shape-keyed path; foreground keeps the legacy
    # role-keyed fallback for compatibility. A resolved GPU placement is exempt:
    # it legitimately holds no CPU region, which is an ANSWER, not an absence.
    if any(claims_cpu_regions(p) and not p.regions for p in (candidate_placement, *active)):
        if is_background:
            return PairDecision.QUEUE
        if matrix is None:
            try:
                matrix = load_contention_matrix()
            except FileNotFoundError:
                return PairDecision.ALLOW
        worst = PairDecision.ALLOW
        for ap in active:
            d = pair_policy(
                candidate_placement.role, ap.role, traffic_class, matrix=matrix, floor=floor
            )
            if d != PairDecision.ALLOW:
                worst = d
        return worst

    # (1) CPU-region conflict → serialize. A shape's label is never consulted;
    # only the region sets AND the device each side actually draws on. This is
    # the disambiguation the role-keyed gate cannot make.
    for ap in active:
        if placements_conflict(ap, candidate_placement):
            return PairDecision.QUEUE

    # (2) Disjoint → role-set N-way verdict over the role union. This branch is
    # valid for A's certified smallest-disjoint placements; nway_policy is
    # invoked unchanged (it handles measured allow/borderline/block + the
    # all-light / heavy-fail-closed fallbacks).
    roles = [p.role for p in active] + [candidate_placement.role]
    return nway_policy(roles, traffic_class, matrix=matrix, floor=floor)


_DECISION_PRECEDENCE = {
    PairDecision.ALLOW: 0,
    PairDecision.DEGRADED_ALLOW: 1,
    PairDecision.QUEUE: 2,
    PairDecision.BLOCK: 3,
}


def _worse_decision(a: PairDecision, b: PairDecision) -> PairDecision:
    """Return the more-restrictive of two decisions (ALLOW < DEGRADED_ALLOW <
    QUEUE < BLOCK), mirroring the gate's worst-of accumulation."""
    return a if _DECISION_PRECEDENCE[a] >= _DECISION_PRECEDENCE[b] else b


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def shape_aware_contention_enabled() -> bool:
    """B wiring seam gate (audit #1 — DUAL flag). Shape-aware admission requires
    BOTH `ORCHESTRATOR_SHAPE_AWARE_CONTENTION=1` AND
    `ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT=1` — the contention verdict is
    only trustworthy when the placement layer is also realizing disjoint
    placements (and the cross-role region mutex is active). Either flag alone →
    disabled → `seam_admit` returns None → callers keep their legacy
    pair_policy/nway_policy path. Default OFF; no live behavior change until an
    operator flips BOTH."""
    return _env_on("ORCHESTRATOR_SHAPE_AWARE_CONTENTION") and _env_on(
        "ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"
    )


def seam_admit(
    candidate_role: str,
    candidate_topology_idx: int,
    active_holders: dict[str, frozenset[str]] | None = None,
    *,
    traffic_class: TrafficClass | str = TrafficClass.FOREGROUND_INTERACTIVE,
    instance_regions: dict[tuple[str, int], frozenset[str]] | None = None,
    matrix: ContentionMatrix | None = None,
    floor: float | None = None,
) -> PairDecision | None:
    """B WIRING SEAM — placement-aware admission with same-role preservation.

    WIRED AND LIVE-REACHABLE (corrected 2026-08-01). This docstring previously
    said "SCAFFOLDING: not yet called by the gate/seeder". That stopped being
    true on 2026-05-31 when the dispatch-side caller landed, and a stale
    "dead code" comment on a live path is worse than no comment: it tells the
    next reader — and the next auditor — to skip the thing that actually runs.
    The real call chain today is
        src/backends/concurrency_aware.py:1334  (passes place.topology_idx)
          -> ContentionGate.admit(candidate_topology_idx=...)
          -> ContentionGate.evaluate()          (contention_gate.py:312)
          -> seam_admit()                       (here)

    What IS still true: this returns None when shape-aware contention is
    disabled, which is the default, so callers keep the legacy
    pair_policy/nway_policy verdict and production behaviour is unchanged until
    an operator arms BOTH flags. Inert-by-flag, not unreachable-by-wiring.

    DEVICE AXIS (2026-08-01, artifact 1 of the device/load-axes rider): this
    module previously had NO concept of `device`, so a role whose weights are
    VRAM-resident under `-ngl` and whose cpuset only supplies host threads for
    tokenising/sampling was accounted identically to a CPU decode holding the
    same cpuset — GPU-lane co-residency with a half/full CPU instance read as
    contention on cpuset overlap alone, when in physics it largely is not. It
    now resolves each placement's device from the compiled
    `orchestration/derived/stack_priors.yaml`
    (`roles.<r>.serving.launch.runtime.flags.device`), corroborated by
    `NUMA_CONFIG[<r>]["gpu_host_lane"]`, and a GPU placement no longer claims
    CPU regions exclusively.

    Resolution FAILS CLOSED, never silently: an undeclared role, or a
    disagreement between the two declarations, is treated like the unknown
    occupancy case below (background QUEUE, foreground legacy path) and logged —
    it is never resolved by picking one declaration. VRAM capacity is NOT gated
    here; it is a set-wide capacity constraint owned by the feasibility model
    (`scripts/server/contention_matrix.assess_feasibility`), because it cannot
    be decided per-pair. See
    handoffs/active/contention-model-device-and-load-axes-rider.md §2/§4.

    The on/off decision is made ONLY by `shape_aware_contention_enabled()` (the
    dual-flag gate). There is deliberately NO `enabled` override parameter
    (audit): a runtime caller must not be able to bypass the dual-flag safety
    contract. Tests enable the seam by setting both env flags (see the
    `shape_aware_on` fixture), exercising the exact runtime gate.

    When enabled:
      - **Same-role** contention (candidate's role already holds a region) is
        routed through `pair_policy(role, role)` so the `same_role` matrix
        verdict is honored. admit_set's disjoint branch delegates to
        `nway_policy`, which dedupes roles — a same-role pair collapses to one
        role (len < 2 -> ALLOW), bypassing `same_role`. The seam prevents that.
      - **Cross-role** contention delegates to `admit_set` (overlap -> QUEUE,
        disjoint -> nway_policy).
      - The worse of the same-role and cross-role verdicts is returned.

    `active_holders` is the EXACT {role: held-regions} view from
    `held_regions_by_role` (not the over-reporting `active_region_holders`);
    when None it is read live.
    """
    if not shape_aware_contention_enabled():
        return None

    if isinstance(traffic_class, str):
        try:
            traffic_class = TrafficClass(traffic_class)
        except ValueError:
            traffic_class = TrafficClass.BACKGROUND
    is_background = traffic_class == TrafficClass.BACKGROUND

    if active_holders is None:
        try:
            from src.runtime.cpu_region_lock import held_regions_by_role

            active_holders = held_regions_by_role(instance_regions)
        except Exception:
            # Audit #2: snapshot failure → occupancy UNKNOWN. Never silently
            # ALLOW under unknown occupancy. Background fails closed (QUEUE);
            # foreground returns None so the caller uses its legacy path rather
            # than a fabricated verdict. (Distinct from a successful empty
            # snapshot below, which legitimately means "no holders → ALLOW".)
            log.warning("seam_admit: held_regions_by_role failed — unknown occupancy")
            return PairDecision.QUEUE if is_background else None

    if not active_holders:
        return PairDecision.ALLOW

    if matrix is None:
        try:
            matrix = load_contention_matrix()
        except FileNotFoundError:
            return PairDecision.QUEUE if is_background else PairDecision.ALLOW

    # Device axis. Resolved for the candidate AND every holder, up front, so a
    # bad declaration cannot colour just one comparison. `resolve_role_device`
    # raises rather than guessing; here that is downgraded to the same
    # fail-closed treatment as an unknown occupancy snapshot — never to a
    # silent CPU default, which is what produced the false-exclusion bug.
    try:
        devices = {
            role: resolve_role_device(role).device_class
            for role in {candidate_role, *active_holders}
        }
    except DeviceResolutionError as exc:
        log.warning("seam_admit: device resolution failed (%s) — failing closed", exc)
        return PairDecision.QUEUE if is_background else None

    candidate = placement_for_instance(
        candidate_role,
        candidate_topology_idx,
        instance_regions,
        device_class=devices[candidate_role],
    )
    all_active = [
        Placement(role, regions, devices[role]) for role, regions in active_holders.items()
    ]

    # Unknown candidate placement -> defer to admit_set's unknown handling (bg
    # fail-closed; fg per-pair, which already routes same-role via pair_policy).
    # A resolved GPU candidate holds no CPU region BY CONSTRUCTION, so it is not
    # "unknown" and must not be routed down the fail-closed path.
    if claims_cpu_regions(candidate) and not candidate.regions:
        return admit_set(all_active, candidate, traffic_class, matrix=matrix, floor=floor)

    same_role_active = [p for p in all_active if p.role == candidate_role]
    cross_role_active = [p for p in all_active if p.role != candidate_role]

    worst = PairDecision.ALLOW

    if same_role_active:
        for p in same_role_active:
            if placements_conflict(p, candidate):
                return PairDecision.QUEUE
        worst = _worse_decision(
            worst,
            pair_policy(candidate_role, candidate_role, traffic_class, matrix=matrix, floor=floor),
        )

    if cross_role_active:
        worst = _worse_decision(
            worst,
            admit_set(cross_role_active, candidate, traffic_class, matrix=matrix, floor=floor),
        )

    return worst


def select_backfill_candidate(
    candidates,
    active_holders: dict[str, frozenset[str]],
    traffic_class: TrafficClass | str = TrafficClass.BACKGROUND,
    *,
    instance_regions: dict[tuple[str, int], frozenset[str]] | None = None,
    matrix: ContentionMatrix | None = None,
    floor: float | None = None,
):
    """C PREP (work-conserving backfill selection) — PURE, NOT yet called by
    runtime. Given dispatcher-priority `candidates` (list of (role,
    topology_idx)) and the EXACT `active_holders` ({role: held-regions}), return
    the FIRST candidate that is BOTH physically disjoint from all held regions
    AND admitted (ALLOW) by the shape-aware `admit_set`. Returns None if none
    qualify.

    This is the selection a future C backfill path will use to fill idle
    quarters beside a heavy node-half holder, instead of leaving them idle
    (the seeder's current non-work-conserving gap). It does NOT remove or alter
    the heavy-port veto, the all-heavy idle barrier, or the dispatch pressure
    skip — those stay until C is explicitly authorized; this only provides the
    pure "what could fill the gap" computation for tests + future wiring.

    `admit_set` enforces: overlap → QUEUE (skip), disjoint → nway verdict,
    unknown placement → bg fail-closed (skip). So a candidate is selected only
    when admit_set returns ALLOW for it against the active set.

    DEVICE-UNAWARE, deliberately: placements are built without a resolved
    `device_class`, so every candidate is treated as CPU-exclusive — identical
    to this function's behaviour before the device axis landed. Resolving the
    device here would let a GPU candidate backfill beside a whole-machine CPU
    holder, which is the RIGHT answer physically but a live-behaviour change in
    a path C has not authorized yet. Resolve devices here when C is wired.
    """
    active_placements = [Placement(role, regions) for role, regions in active_holders.items()]
    for role, topology_idx in candidates:
        cand = placement_for_instance(role, topology_idx, instance_regions)
        decision = admit_set(active_placements, cand, traffic_class, matrix=matrix, floor=floor)
        if decision == PairDecision.ALLOW:
            return (role, topology_idx)
    return None


def topology_fingerprint(numa_config: dict[str, Any]) -> str:
    """Deterministic sha256 of role topology — used to detect stale matrices.

    Hashes (role_name, [(cpu_list, port, threads), ...]) tuples in sorted
    order. NUMA_CONFIG metadata like mlock/numactl_policy is intentionally
    excluded — the matrix is about CPU/NUMA placement, not server tuning.
    """
    fingerprint_input: list[tuple[str, list[tuple[str, int, int]]]] = []
    for role in sorted(numa_config.keys()):
        cfg = numa_config[role]
        instances = cfg.get("instances", []) if isinstance(cfg, dict) else []
        instance_tuples = []
        for inst in instances:
            if isinstance(inst, (list, tuple)) and len(inst) >= 3:
                cpu_list = str(inst[0])
                port = int(inst[1])
                threads = int(inst[2])
                instance_tuples.append((cpu_list, port, threads))
        fingerprint_input.append((role, instance_tuples))
    canonical = json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def role_topology_fingerprint(
    numa_config: dict[str, Any],
    role: str,
    *,
    live_ports: set[int] | frozenset[int] | tuple[int, ...] | list[int] | None = None,
) -> str:
    """Deterministic hash for one role's certified live instance subset.

    `topology_fingerprint()` answers "does the entire matrix still describe
    this NUMA_CONFIG?". EvalTower same-role fan-out sometimes only needs a
    narrower answer: "do these live ports for this role still match the
    recertified live-shape measurement?". This helper hashes the role name and
    filtered `(cpu_list, port, threads)` tuples using the same canonical rules.
    """
    cfg = (numa_config or {}).get(role) if numa_config else None
    instances = cfg.get("instances", []) if isinstance(cfg, dict) else []
    port_filter: set[int] | None = None
    if live_ports is not None:
        port_filter = {int(port) for port in live_ports}
    instance_tuples: list[tuple[str, int, int]] = []
    for inst in instances:
        if isinstance(inst, (list, tuple)) and len(inst) >= 3:
            try:
                port = int(inst[1])
                threads = int(inst[2])
            except (TypeError, ValueError):
                continue
            if port_filter is not None and port not in port_filter:
                continue
            instance_tuples.append((str(inst[0]), port, threads))
    canonical = json.dumps([(role, instance_tuples)], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def matrix_measured_roles(matrix: ContentionMatrix | None) -> set[str]:
    """Return every NUMA role a contention matrix says anything about.

    This is the BROAD set (pairs + same_role + n_way + unknown_pairs) and is what
    reporting surfaces want. It is deliberately NOT the set the top-level
    ``topology_hash`` stamp covers — see ``matrix_stamped_roles``.
    """
    if matrix is None:
        return set()
    roles: set[str] = set(matrix.same_role)
    for pair in matrix.pairs:
        roles.update(pair)
    for pair in matrix.unknown_pairs:
        roles.update(pair)
    for roles_key in matrix.n_way:
        roles.update(roles_key)
    return roles


def matrix_stamped_roles(matrix: ContentionMatrix | None) -> set[str]:
    """Return the roles the matrix's top-level ``topology_hash`` was computed over.

    ``scripts/server/contention_matrix.py::cmd_run`` computes that stamp as
    ``topology_fingerprint({role: NUMA_CONFIG[role] for role in measured_roles})``
    where ``measured_roles`` are exactly the roles of the PAIRWISE run it is
    writing — the same run that emits ``pairs`` and ``unknown_pairs``. The other
    sections come from the separate ``bench-nway`` / ``bench-within-role``
    subcommands, are hand-merged as fragments, and carry their own provenance
    (``same_role_certifications`` entries even hold a per-entry ``topology_hash``).
    They never (re)stamp the top-level hash.

    Comparing the stamp against a role set WIDER than the one it was computed
    over cannot succeed: it is a hash of a different input. That mismatch is what
    made a matrix carrying a correctly re-benched ``pairs`` block read as
    uncertified merely because a preserved N-way fragment still names a role that
    has since been folded onto another server's process (``vision_escalation`` ->
    ``worker_vision``'s :8086 in the 2026-08-01 W1 cutover).
    """
    if matrix is None:
        return set()
    roles: set[str] = set()
    for pair in matrix.pairs:
        roles.update(pair)
    for pair in matrix.unknown_pairs:
        roles.update(pair)
    return roles


def topology_fingerprint_for_matrix(
    numa_config: dict[str, Any],
    matrix: ContentionMatrix | None,
) -> str:
    """Fingerprint the live topology subset comparable to ``matrix``.

    Explicit-only auxiliary roles can live in NUMA_CONFIG without being part of
    the production contention matrix. If the matrix clearly represents a role
    subset, compare only that subset. If the matrix is unavailable or references
    missing roles, fall back to the full topology so stale detection remains
    conservative.

    The subset is the STAMPED set (``matrix_stamped_roles``), because that is the
    input the committed ``topology_hash`` was computed from. Staleness detection
    is not weakened: if any pairwise-measured role has vanished from the live
    NUMA_CONFIG the fallback below still fires, and a topology CHANGE to any
    stamped role still changes the fingerprint.
    """
    roles = matrix_stamped_roles(matrix)
    if roles and roles.issubset(numa_config):
        return topology_fingerprint({role: numa_config[role] for role in sorted(roles)})
    return topology_fingerprint(numa_config)
