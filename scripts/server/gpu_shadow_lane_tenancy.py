"""Registry-driven tenancy for the GPU-resident shadow lane (P2-1).

THE ENGINEERING DELIVERABLE IS THE SLOT, NOT THE TENANT. This module turns
``orchestration/gpu_shadow_lane_tenancy.yaml`` into a validated object graph so
that swapping the resident model, its launch mode, or its duty bindings is a
DATA edit plus the State-A/State-B choreography in ``docs/gpu-shadow-lane.md``
— never a code change.

What this module refuses to do, and why:

- **It never applies anything to the registry.** The master registry is FROZEN
  (program decision D3). ``render_registry_proposal`` emits a PROPOSAL DIFF for
  human review; there is no apply path in this file, not even a guarded one.
- **It never starts a process.** Launch plans are argv lists for reports and
  proposals. Activation is the operator's Steps 0-7.
- **It never falls back.** A tenant with no policy row, an unattested artifact,
  or a mode with no measured frontier is REFUSED. Every fallback in a capacity
  path is a silent authorisation of an unmeasured operating point.

Gating: everything here is behind the default-off
``ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE`` flag, and no production module imports
this file (witnessed by the zero-coupling tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scripts.server.gpu_shadow_lane import (
    GpuShadowLaneDisabled,
    LANE_NAME,
    NpCeilingPolicy,
    build_tenant_launch_plan,
    estimated_dynamic_gib,
    lane_enabled,
    mode_for,
    np_ceiling,
)
from src.features import Features
from src.registry.kernel_paths import backend_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TENANCY_PATH = PROJECT_ROOT / "orchestration" / "gpu_shadow_lane_tenancy.yaml"

# State A = lane absent. It is the rollback state AND the only state a data file
# may declare: a YAML edit must not be able to assert that a server is running.
STATE_A = "state_a"
STATE_B = "state_b"

# Role names that exist in the production registry. A shadow binding that names
# one of these is refused — this is program decision D3 enforced at the data
# layer rather than by reviewer vigilance. Kept as a literal deny-list (not an
# import of the live registry) precisely so this module stays zero-coupled to
# the production launch path.
PRODUCTION_ROLE_NAMES = frozenset(
    {
        "frontdoor",
        "coder_escalation",
        "worker_general",
        "architect_general",
        # Added 2026-09-27 (ARCHITECT SWAP): a live production role (the :8083
        # 27B) since 2026-08-01 that this deny-list never named.
        "architect_critic",
        "ingest_long_context",
        "vision_escalation",
        "eval_batch_frontdoor",
    }
)

# docs/gpu-shadow-lane.md §3, program decision D1. Priority order, highest
# first. Declaring the order here reserves it so later implementation of the
# unbuilt classes is a policy fill-in, not a redesign.
ADMISSION_CLASSES = (
    "escalations",
    "distillation_backfill",
    "shed_batch",
    "degraded_frontdoor_overflow",
)
# Classes 3 and 4 are NAMED but unimplemented (D1: "design provision now;
# implementation not required until after lane hardening"). A binding may not
# claim them yet — reserving a name is not the same as having built it.
IMPLEMENTED_ADMISSION_CLASSES = frozenset({"escalations", "distillation_backfill"})


class TenancyError(ValueError):
    """Raised when the tenancy table violates a lane invariant."""


@dataclass(frozen=True)
class RoleBinding:
    """One duty this tenant would serve. Shadow-only until P3-3."""

    role: str
    shadow: bool
    duty: str
    admission_class: str
    note: str | None = None


@dataclass(frozen=True)
class LaunchMode:
    """Launch-bound mode properties (D6: MTP is global, not per-request)."""

    mtp: bool
    reasoning: bool
    draft_n_max: int | None = None

    @property
    def policy_mode(self) -> str:
        return mode_for(mtp=self.mtp)


@dataclass(frozen=True)
class TenantArtifact:
    path: str
    bytes: int
    sha256: str | None
    sha256_status: str

    @property
    def attested(self) -> bool:
        # "attested_by_instrument" = the measuring instrument's own pre-launch
        # identity contract refused to run against a mismatched file. That is
        # stronger evidence than an unverified re-hash, and it is recorded as a
        # distinct status so the provenance stays legible.
        return self.sha256 is not None and self.sha256_status in {
            "attested",
            "attested_by_instrument",
        }


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    status: str
    description: str
    np_policy_tenant: str
    artifact: TenantArtifact
    mode: LaunchMode
    role_bindings: tuple[RoleBinding, ...]
    co_residency: str = "allowed"
    registry_catalogue_id: str | None = None


@dataclass(frozen=True)
class LaneSlot:
    """The slot itself. Every field is a lane property, never a tenant one."""

    device: str
    port: int
    host_cpuset: str
    host_threads: int
    binary_dir: str
    binary_version: str
    binary_commit: str
    lock_role: str
    cpu_regions_from: str
    device_lock: bool

    @property
    def binary_path(self) -> Path:
        return Path(self.binary_dir) / "llama-server"


@dataclass(frozen=True)
class Tenancy:
    version: int
    lane: str
    slot: LaneSlot
    resident_state: str
    resident_tenant: str | None
    tenants: dict[str, Tenant]


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise TenancyError(f"tenancy: {label} must be a mapping")
    return value


def _parse_role_binding(raw: Any, label: str) -> RoleBinding:
    record = _require_mapping(raw, label)
    role = str(record["role"])
    shadow = bool(record.get("shadow", False))
    if not shadow:
        raise TenancyError(
            f"tenancy: {label} role {role!r} is not marked shadow. The lane serves "
            "no production traffic until the P3-3 three-gates sign-off (D3)."
        )
    if role in PRODUCTION_ROLE_NAMES:
        raise TenancyError(
            f"tenancy: {label} binds the PRODUCTION role {role!r}. Shadow roles must "
            "be distinct names (e.g. coder_escalation_shadow); rebinding a live role "
            "is the P3-3 operator gate, not a data edit."
        )
    admission_class = str(record.get("admission_class", ""))
    if admission_class not in ADMISSION_CLASSES:
        raise TenancyError(
            f"tenancy: {label} admission_class {admission_class!r} is not one of "
            f"{ADMISSION_CLASSES}"
        )
    if admission_class not in IMPLEMENTED_ADMISSION_CLASSES:
        raise TenancyError(
            f"tenancy: {label} claims admission_class {admission_class!r}, which is "
            "reserved by D1 but not implemented. Build it behind its own default-off "
            "flag before binding a duty to it."
        )
    note = record.get("note")
    return RoleBinding(
        role=role,
        shadow=shadow,
        duty=str(record["duty"]),
        admission_class=admission_class,
        note=str(note) if note is not None else None,
    )


def _parse_tenant(tenant_id: str, raw: Any) -> Tenant:
    record = _require_mapping(raw, f"tenants.{tenant_id}")
    model = _require_mapping(record.get("model"), f"tenants.{tenant_id}.model")
    mode_raw = _require_mapping(record.get("mode"), f"tenants.{tenant_id}.mode")
    mtp = bool(mode_raw.get("mtp", False))
    draft_n_max = mode_raw.get("draft_n_max")
    if mtp and (draft_n_max is None or int(draft_n_max) <= 0):
        raise TenancyError(
            f"tenancy: tenants.{tenant_id}.mode has mtp: true but no positive "
            "draft_n_max. The self-draft depth is part of the measured identity "
            "(FF ran n_max=1, the A4 bridge n_max=4) — an unstated depth is an "
            "unmeasured operating point."
        )
    raw_bindings = record.get("role_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise TenancyError(
            f"tenancy: tenants.{tenant_id}.role_bindings must be a non-empty list"
        )
    bindings = tuple(
        _parse_role_binding(item, f"tenants.{tenant_id}.role_bindings[{idx}]")
        for idx, item in enumerate(raw_bindings)
    )
    sha = model.get("sha256")
    return Tenant(
        tenant_id=tenant_id,
        status=str(record.get("status", "candidate")),
        description=str(record.get("description", "")).strip(),
        np_policy_tenant=str(record["np_policy_tenant"]),
        artifact=TenantArtifact(
            path=str(model["path"]),
            bytes=int(model["bytes"]),
            sha256=str(sha) if sha is not None else None,
            sha256_status=str(model.get("sha256_status", "unattested")),
        ),
        mode=LaunchMode(
            mtp=mtp,
            reasoning=bool(mode_raw.get("reasoning", False)),
            draft_n_max=int(draft_n_max) if draft_n_max is not None else None,
        ),
        role_bindings=bindings,
        co_residency=str(record.get("co_residency", "allowed")),
        registry_catalogue_id=(
            str(record["registry_catalogue_id"])
            if record.get("registry_catalogue_id") is not None
            else None
        ),
    )


def _resolve_slot_binary_dir(binary: dict[str, Any]) -> str:
    """Resolve ``slot.binary`` to a directory, preferring the kernel store.

    ``backend:`` names a CAPABILITY and is resolved through
    ``src/registry/kernel_paths.py``, so a kernel promotion that repoints
    ``/mnt/raid0/llm/kernels/production/<backend>`` moves the lane with it. An
    explicit ``dir:`` remains an override (the same distinction stack_priors
    draws for ``runtime_requirements.binary_dir``) and is used verbatim.

    Deliberately fatal when neither resolves: substituting a build path here is
    how a lane keeps serving off a retired kernel without anything noticing.
    """
    explicit = binary.get("dir")
    if explicit is not None:
        return str(explicit)
    backend = binary.get("backend")
    if backend is None:
        raise TenancyError("slot.binary must declare either 'backend' or 'dir'")
    return str(backend_dir(str(backend)))


def _parse_slot(raw: Any) -> LaneSlot:
    slot = _require_mapping(raw, "slot")
    binary = _require_mapping(slot.get("binary"), "slot.binary")
    claim = _require_mapping(slot.get("region_claim"), "slot.region_claim")
    return LaneSlot(
        device=str(slot["device"]),
        port=int(slot["port"]),
        host_cpuset=str(slot["host_cpuset"]),
        host_threads=int(slot["host_threads"]),
        binary_dir=_resolve_slot_binary_dir(binary),
        binary_version=str(binary["version"]),
        binary_commit=str(binary["commit"]),
        lock_role=str(claim["lock_role"]),
        cpu_regions_from=str(claim["cpu_regions_from"]),
        device_lock=bool(claim.get("device_lock", False)),
    )


def load_tenancy(
    path: Path | None = None,
    *,
    feats: Features | None = None,
) -> Tenancy:
    """Load + validate the tenancy table.

    Raises GpuShadowLaneDisabled unless the gpu_shadow_lane feature flag is on,
    so no production code path can consume tenancy data accidentally.
    """
    if not lane_enabled(feats):
        raise GpuShadowLaneDisabled(
            "gpu_shadow_lane feature flag is off (set ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE=1 "
            "or pass an explicit Features override)"
        )
    tenancy_path = path or DEFAULT_TENANCY_PATH
    payload = _require_mapping(
        yaml.safe_load(tenancy_path.read_text(encoding="utf-8")), "document"
    )
    version = int(payload.get("version", 0))
    if version != 1:
        raise TenancyError(f"tenancy: unsupported version {version}")
    lane = str(payload.get("lane", ""))
    if lane != LANE_NAME:
        raise TenancyError(f"tenancy: lane {lane!r} != {LANE_NAME!r}")

    resident_state = str(payload.get("resident_state", ""))
    if resident_state != STATE_A:
        raise TenancyError(
            f"tenancy: resident_state must be {STATE_A!r} (lane absent), got "
            f"{resident_state!r}. State B is reached by the operator running the "
            "activation choreography — a data file cannot declare the lane resident."
        )
    resident_tenant = payload.get("resident_tenant")
    if resident_tenant is not None:
        raise TenancyError(
            "tenancy: resident_tenant must be null in state_a — a resident tenant "
            "without a running lane is a lie the rest of the system would believe."
        )

    tenants = {
        str(tenant_id): _parse_tenant(str(tenant_id), raw)
        for tenant_id, raw in _require_mapping(payload.get("tenants"), "tenants").items()
    }
    if not tenants:
        raise TenancyError("tenancy: no tenants defined")
    return Tenancy(
        version=version,
        lane=lane,
        slot=_parse_slot(payload.get("slot")),
        resident_state=resident_state,
        resident_tenant=None,
        tenants=tenants,
    )


# ── Cross-validation against the np_ceiling policy ───────────────────────────


def cross_validate(tenancy: Tenancy, policy: NpCeilingPolicy) -> list[str]:
    """Return the list of tenancy<->policy inconsistencies (empty = consistent).

    This is the check that stops a tenant swap from silently inheriting another
    model's VRAM arithmetic — the P2-4 P1-4 hazard, where stock and FF differed
    by 1.04 GiB while sharing one policy row.
    """
    problems: list[str] = []
    for tenant in tenancy.tenants.values():
        row = policy.tenants.get(tenant.np_policy_tenant)
        if row is None:
            problems.append(
                f"{tenant.tenant_id}: np_policy_tenant {tenant.np_policy_tenant!r} has no "
                f"row in the np_ceiling policy (have: {sorted(policy.tenants)})"
            )
            continue
        if row.model_path != tenant.artifact.path:
            problems.append(
                f"{tenant.tenant_id}: model path disagrees with policy row "
                f"{tenant.np_policy_tenant!r} ({tenant.artifact.path!r} vs {row.model_path!r})"
            )
        if row.model_bytes is not None and row.model_bytes != tenant.artifact.bytes:
            problems.append(
                f"{tenant.tenant_id}: model bytes disagree with policy row "
                f"({tenant.artifact.bytes} vs {row.model_bytes})"
            )
        if (
            row.model_sha256 is not None
            and tenant.artifact.sha256 is not None
            and row.model_sha256 != tenant.artifact.sha256
        ):
            problems.append(
                f"{tenant.tenant_id}: model sha256 disagrees with policy row"
            )
        if row.mode_policy(tenant.mode.policy_mode) is None:
            problems.append(
                f"{tenant.tenant_id}: policy row {tenant.np_policy_tenant!r} has no "
                f"validated frontier for mode {tenant.mode.policy_mode!r}"
            )
        if (
            tenant.mode.mtp
            and row.draft_n_max is not None
            and row.draft_n_max != tenant.mode.draft_n_max
        ):
            problems.append(
                f"{tenant.tenant_id}: draft depth {tenant.mode.draft_n_max} was not the "
                f"measured depth {row.draft_n_max} for policy row "
                f"{tenant.np_policy_tenant!r} — a different depth is a different, "
                "unmeasured operating point"
            )
    return problems


# ── Lane plan resolution (role-agnostic) ─────────────────────────────────────


@dataclass(frozen=True)
class ResolvedLanePlan:
    """A fully data-derived plan for occupying the slot with one tenant."""

    tenant_id: str
    duties: tuple[str, ...]
    roles: tuple[str, ...]
    model_path: str
    model_vram_gib: float
    mode: str
    np_slots: int
    slot_context_tokens: int
    budget_profile: str
    dynamic_budget_gib: float
    np_ceiling: int | None
    estimated_dynamic_gib: float | None
    port: int
    host_cpuset: str
    launch_argv: tuple[str, ...]
    refusals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def admissible(self) -> bool:
        return not self.refusals


def resolve_lane_plan(
    tenancy: Tenancy,
    policy: NpCeilingPolicy,
    *,
    tenant_id: str,
    budget_profile: str,
    np_slots: int,
    slot_context_tokens: int,
) -> ResolvedLanePlan:
    """Resolve (tenancy + policy) into a launch plan, collecting REFUSALS.

    Nothing here launches. The plan is admissible only when ``refusals`` is
    empty; callers must treat a non-empty list as a hard stop, never as advice.
    """
    tenant = tenancy.tenants.get(tenant_id)
    if tenant is None:
        raise KeyError(f"unknown tenant {tenant_id!r}")
    row = policy.tenants.get(tenant.np_policy_tenant)
    if row is None:
        raise KeyError(
            f"tenant {tenant_id!r} names np_policy_tenant "
            f"{tenant.np_policy_tenant!r}, which has no policy row"
        )

    refusals: list[str] = []
    mode = tenant.mode.policy_mode
    mode_policy = row.mode_policy(mode)

    if not tenant.artifact.attested:
        refusals.append(
            f"tenant artifact is not attested (sha256_status="
            f"{tenant.artifact.sha256_status!r}); activation requires a pinned hash"
        )

    budget_gib = 0.0
    if mode_policy is None:
        refusals.append(
            f"no validated frontier for mode {mode!r} on policy row "
            f"{tenant.np_policy_tenant!r} — refuse, never fall back to the other mode"
        )
    else:
        rows = {budget_row.name: budget_row for budget_row in mode_policy.budgets}
        if budget_profile not in rows:
            refusals.append(
                f"budget profile {budget_profile!r} is not defined for mode {mode!r} "
                f"(have: {sorted(rows)})"
            )
        else:
            budget_gib = rows[budget_profile].dynamic_budget_gib

    ceiling: int | None = None
    if not refusals:
        ceiling = np_ceiling(
            policy,
            tenant.np_policy_tenant,
            dynamic_budget_gib=budget_gib,
            slot_context_tokens=slot_context_tokens,
            mode=mode,
        )
        if ceiling is None:
            refusals.append(
                f"np_ceiling has no validated operating point for "
                f"{tenant.np_policy_tenant!r} at slot_context={slot_context_tokens} "
                f"within budget {budget_gib} GiB (mode {mode})"
            )
        elif np_slots > ceiling:
            refusals.append(
                f"planned -np {np_slots} exceeds the validated ceiling {ceiling} at "
                f"slot_context={slot_context_tokens} (mode {mode})"
            )

    if tenant.co_residency == "forbidden" and budget_profile != "solo_resident":
        refusals.append(
            f"tenant {tenant_id!r} is co_residency: forbidden and may only be planned "
            "against the solo_resident profile (sequential bench windows only)"
        )

    argv = build_tenant_launch_plan(
        model_path=tenant.artifact.path,
        np_slots=np_slots,
        slot_context_tokens=slot_context_tokens,
        port=tenancy.slot.port,
        host_cpuset=tenancy.slot.host_cpuset,
        host_threads=tenancy.slot.host_threads,
        device=tenancy.slot.device,
        binary=tenancy.slot.binary_path,
        mtp=tenant.mode.mtp,
        draft_n_max=tenant.mode.draft_n_max,
        reasoning=tenant.mode.reasoning,
    )
    return ResolvedLanePlan(
        tenant_id=tenant_id,
        duties=tuple(binding.duty for binding in tenant.role_bindings),
        roles=tuple(binding.role for binding in tenant.role_bindings),
        model_path=tenant.artifact.path,
        model_vram_gib=row.model_vram_gib,
        mode=mode,
        np_slots=np_slots,
        slot_context_tokens=slot_context_tokens,
        budget_profile=budget_profile,
        dynamic_budget_gib=budget_gib,
        np_ceiling=ceiling,
        estimated_dynamic_gib=estimated_dynamic_gib(
            row, np_slots=np_slots, slot_context_tokens=slot_context_tokens
        ),
        port=tenancy.slot.port,
        host_cpuset=tenancy.slot.host_cpuset,
        launch_argv=tuple(argv),
        refusals=tuple(refusals),
    )


# ── Registry PROPOSAL rendering (never an apply) ─────────────────────────────


def render_registry_proposal(plan: ResolvedLanePlan, tenancy: Tenancy) -> str:
    """Render the master-registry role block for ``plan`` as a PROPOSAL.

    Returns markdown for human review. There is deliberately no function in
    this module that writes to a registry: the registry is frozen (D3), and the
    absence of an apply path is the enforcement, not a flag guarding one.
    """
    tenant = tenancy.tenants[plan.tenant_id]
    total_ctx = plan.np_slots * plan.slot_context_tokens
    binding_lines = "\n".join(
        f"#   - {binding.role}  (duty: {binding.duty}, admission: {binding.admission_class})"
        for binding in tenant.role_bindings
    )
    refusal_block = (
        "\n".join(f"  - {item}" for item in plan.refusals)
        if plan.refusals
        else "  (none — plan is admissible)"
    )
    return f"""# PROPOSAL — gpu_shadow_lane role block for `{plan.tenant_id}`

**NOT APPLIED.** The master registry is frozen (program decision D3). This is a
diff for human review; applying it is Step 1 of the activation choreography in
`docs/gpu-shadow-lane.md`, and it is the operator's to run.

- tenant: `{plan.tenant_id}` ({tenant.status})
- artifact: `{tenant.artifact.path}`
  - bytes `{tenant.artifact.bytes}`, sha256 `{tenant.artifact.sha256}` ({tenant.artifact.sha256_status})
- mode: `{plan.mode}`{f" (draft n_max {tenant.mode.draft_n_max})" if tenant.mode.mtp else ""}, reasoning `{'on' if tenant.mode.reasoning else 'off'}`
- plan: `-np {plan.np_slots}` x `{plan.slot_context_tokens}` ctx = `{total_ctx}` total
- np_ceiling: `{plan.np_ceiling}` at budget `{plan.dynamic_budget_gib}` GiB (`{plan.budget_profile}`)
- estimated dynamic VRAM: `{plan.estimated_dynamic_gib}` GiB over `{plan.model_vram_gib}` GiB weights

## Refusals

{refusal_block}

## Shadow role bindings (D3 — none is a production role)

```
{binding_lines}
```

## Planned launch argv (NOT executed)

```bash
{" ".join(plan.launch_argv)}
```
"""
