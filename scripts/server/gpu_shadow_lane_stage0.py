#!/usr/bin/env python3
"""Stage-0 hardening for the GPU shadow lane (P2-3).

Three subcommands, all ZERO-INFERENCE and all safe to run against a live fleet:

  ``smoke``    Deterministic self-check of the lane's data plane. Runs entirely
               against the policy/tenancy tables and recorded FIXTURES — it does
               not contact the GPU, does not launch a server, does not send a
               request. Same inputs always produce the same verdict, so it can
               gate a commit rather than a bench window.

  ``attest``   Parse-and-judge the three attestation surfaces (health, host
               affinity, VRAM residency) from JSON on disk or stdin. The
               PARSING and the JUDGING are pure functions, so the same code that
               scores a fixture today scores live output at activation Step 5 —
               there is no second, unreviewed implementation waiting to be
               written under time pressure at 3am.

  ``recert``   Emit the COMPLETE contention recert set for the lane's host-side
               cores, and the exact ``contention_matrix.py`` command to run it.
               Plan-only: it prints the command, it never benches.

Why the recert set needed rebuilding (P2-4 finding P1-2): the lane's host slice
is the SMT siblings 184-191, whose physical cores are 88-95. A raw string
overlap of "184-191" against ``architect_general``'s "0-95" finds NOTHING, so
the single role that occupies all 96 physical cores was missing from the recert
set — the one role guaranteed to contend with the lane. Folding SMT siblings
onto physical cores first is what makes the set complete.

Exit codes: 0 clean · 75 blocked/failed · 2 usage (argparse).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

from scripts.server.gpu_shadow_lane import (
    LANE_BINARY_COMMIT,
    LANE_BINARY_VERSION,
    LANE_DEVICE,
    LANE_HOST_CPUSET,
    LANE_PORT,
    MODE_MTP_OFF,
    NpCeilingPolicy,
    load_np_ceiling_policy,
    load_serving_shape,
)
from scripts.server.gpu_shadow_lane_lease import (
    cpuset_shares_physical_cores,
    fold_smt_to_physical,
    lane_host_regions,
)
from scripts.server.gpu_shadow_lane_tenancy import (
    Tenancy,
    cross_validate,
    load_tenancy,
    resolve_lane_plan,
)
from scripts.server.stack_numa import NUMA_CONFIG
from src.features import Features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "orchestration" / "reports"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "gpu_shadow_lane"

# Contention floor below which a pair is a rollback trigger
# (docs/gpu-shadow-lane.md §7 Step 7; matches contention_matrix's own default).
CONTENTION_FLOOR = 0.65


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ── Attestation: pure parsers + pure judges ──────────────────────────────────


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    @property
    def marker(self) -> str:
        return "PASS" if self.passed else "FAIL"


def judge_health(payload: Any, *, expect_slots: int | None = None) -> list[Check]:
    """Judge a llama-server ``/health`` + ``/slots`` snapshot.

    Accepts ``{"health": {...}, "slots": [...]}``. A missing key is a FAIL, not
    a skip: "the probe could not see it" and "the probe saw it and it was fine"
    must never collapse into the same verdict.
    """
    checks: list[Check] = []
    if not isinstance(payload, dict):
        return [Check("health.shape", False, "payload is not a JSON object")]

    health = payload.get("health")
    if not isinstance(health, dict):
        checks.append(Check("health.present", False, "no 'health' object in payload"))
    else:
        status = str(health.get("status", ""))
        checks.append(
            Check("health.status", status == "ok", f"status={status or '<missing>'}")
        )

    slots = payload.get("slots")
    if not isinstance(slots, list):
        checks.append(Check("slots.present", False, "no 'slots' array in payload"))
        return checks
    checks.append(Check("slots.nonempty", bool(slots), f"{len(slots)} slot(s)"))
    if expect_slots is not None:
        checks.append(
            Check(
                "slots.count",
                len(slots) == expect_slots,
                f"{len(slots)} slots, planned -np {expect_slots}",
            )
        )
    busy = [s for s in slots if isinstance(s, dict) and s.get("is_processing")]
    checks.append(
        Check("slots.idle", not busy, f"{len(busy)} slot(s) processing at attestation")
    )
    return checks


def judge_affinity(payload: Any, *, expect_cpuset: str = LANE_HOST_CPUSET) -> list[Check]:
    """Judge host-thread affinity for the lane process.

    Accepts ``{"pid": N, "cpus_allowed_list": "184-191"}``. The comparison is on
    the FOLDED physical-core set, not the string: "184-191" and "88-95" name the
    same silicon, and a check that only compares strings would pass a process
    pinned to the physical cores while the GPU host-thread rule says siblings.
    Both the string and the folded set are reported so a mismatch is legible.
    """
    checks: list[Check] = []
    if not isinstance(payload, dict):
        return [Check("affinity.shape", False, "payload is not a JSON object")]
    actual = payload.get("cpus_allowed_list")
    if not isinstance(actual, str) or not actual.strip():
        return [Check("affinity.present", False, "no cpus_allowed_list in payload")]

    checks.append(
        Check("affinity.exact", actual == expect_cpuset, f"{actual!r} vs {expect_cpuset!r}")
    )
    actual_cores = fold_smt_to_physical(actual)
    expect_cores = fold_smt_to_physical(expect_cpuset)
    checks.append(
        Check(
            "affinity.physical_cores",
            actual_cores == expect_cores,
            f"folded {sorted(actual_cores)} vs {sorted(expect_cores)}",
        )
    )
    expect_regions = lane_host_regions(expect_cpuset)
    checks.append(
        Check(
            "affinity.regions",
            bool(expect_regions),
            f"lane occupies regions {sorted(expect_regions)}",
        )
    )
    return checks


def judge_vram(
    payload: Any,
    *,
    model_vram_gib: float,
    estimated_dynamic_gib: float | None,
    device: str = LANE_DEVICE,
) -> list[Check]:
    """Judge GPU residency from a rocm-smi-shaped snapshot.

    Accepts ``{"total_gib": f, "used_gib": f, "compute_pids": [...]}``.
    """
    checks: list[Check] = []
    if not isinstance(payload, dict):
        return [Check("vram.shape", False, "payload is not a JSON object")]
    total = payload.get("total_gib")
    used = payload.get("used_gib")
    if not isinstance(total, (int, float)) or not isinstance(used, (int, float)):
        return [Check("vram.present", False, "total_gib/used_gib missing or non-numeric")]

    required = float(model_vram_gib) + float(estimated_dynamic_gib or 0.0)
    checks.append(
        Check(
            "vram.residency",
            float(used) >= float(model_vram_gib) * 0.9,
            f"used {used:.1f} GiB vs {model_vram_gib:.1f} GiB weights "
            "(a resident tenant must show its weights)",
        )
    )
    checks.append(
        Check(
            "vram.headroom",
            float(total) >= required,
            f"total {total:.1f} GiB vs required {required:.1f} GiB on {device}",
        )
    )
    pids = payload.get("compute_pids")
    if pids is None:
        checks.append(Check("vram.pids_reported", False, "compute_pids key absent"))
    else:
        checks.append(
            Check(
                "vram.pids_known",
                isinstance(pids, list),
                f"compute pids: {pids}",
            )
        )
    return checks


# ── Contention recert set (SMT-aware — the P1-2 correction) ──────────────────


@dataclass(frozen=True)
class RecertRole:
    role: str
    port: int
    cpu_list: str
    shared_cores: tuple[int, ...]
    basis: str  # "smt_sibling_overlap" | "physical_core_overlap"


def recert_roles(host_cpuset: str = LANE_HOST_CPUSET) -> list[RecertRole]:
    """Every production instance that physically shares cores with the lane.

    Comparison is on FOLDED physical cores. The two bases are distinguished so
    a reviewer can see which entries a string-overlap check would have found
    (``smt_sibling_overlap``: the instance literally lists 168-191) and which it
    would have MISSED (``physical_core_overlap``: the instance lists physical
    cores only, e.g. architect_general's "0-95" — the :8074 instance, which was
    architect_critic's between 2026-08-01 and the 2026-09-27 ARCHITECT SWAP).

    The example used to be architect_general's "0-95"; the 2026-08-01 W1 cutover
    moved that role ONTO the lane (184-191), so it is now a literal
    ``smt_sibling_overlap`` entry. The full-machine instances the fold catches
    today are architect_general:8074 (architect_critic:8074 until the 2026-09-27
    swap), frontdoor:8070, ingest_long_context:8085 and worker_general:8072.
    """
    lane_cores = fold_smt_to_physical(host_cpuset)
    found: list[RecertRole] = []
    for role, cfg in NUMA_CONFIG.items():
        for cpu_list, port, _threads in cfg.get("instances", []):
            shared = cpuset_shares_physical_cores(cpu_list, host_cpuset)
            if not shared:
                continue
            # Did the instance's own spec literally name any of the lane's
            # logical CPUs? If not, a raw string overlap would have missed it.
            literal = bool(
                {
                    cpu
                    for cpu in _parse_literal(cpu_list)
                    if cpu in _parse_literal(host_cpuset)
                }
            )
            found.append(
                RecertRole(
                    role=role,
                    port=port,
                    cpu_list=cpu_list,
                    shared_cores=tuple(sorted(shared)),
                    basis="smt_sibling_overlap" if literal else "physical_core_overlap",
                )
            )
    assert lane_cores, "lane cpuset folded to no physical cores"
    return sorted(found, key=lambda item: (item.role, item.port))


def _parse_literal(spec: str) -> set[int]:
    cpus: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(chunk))
    return cpus


def recert_command(roles: list[RecertRole], lane_role: str = "gpu_shadow_lane") -> list[str]:
    """The exact contention_matrix invocation for the recert set (plan only)."""
    names = sorted({item.role for item in roles} | {lane_role})
    return [
        "uv",
        "run",
        "python",
        "scripts/server/contention_matrix.py",
        "run",
        "--roles",
        *names,
    ]


# ── Deterministic smoke ──────────────────────────────────────────────────────


def smoke_checks(
    tenancy: Tenancy,
    policy: NpCeilingPolicy,
    *,
    fixture_dir: Path = FIXTURE_DIR,
) -> list[Check]:
    """The full deterministic Stage-0 self-check. No I/O beyond fixtures."""
    checks: list[Check] = []

    # 1. Tenancy is in the rollback state.
    checks.append(
        Check(
            "state.is_state_a",
            tenancy.resident_state == "state_a" and tenancy.resident_tenant is None,
            f"resident_state={tenancy.resident_state}, tenant={tenancy.resident_tenant}",
        )
    )

    # 2. Slot pins match the frozen production kernel + the host-thread rule.
    checks.append(
        Check(
            "slot.binary_pin",
            tenancy.slot.binary_version == LANE_BINARY_VERSION
            and tenancy.slot.binary_commit == LANE_BINARY_COMMIT,
            f"{tenancy.slot.binary_version} ({tenancy.slot.binary_commit})",
        )
    )
    checks.append(
        Check(
            "slot.host_threads_are_smt_siblings",
            all(cpu > 95 for cpu in _parse_literal(tenancy.slot.host_cpuset)),
            f"host_cpuset={tenancy.slot.host_cpuset} (GPU host threads pin to SMT "
            "siblings, never physical cores 88-95)",
        )
    )
    checks.append(
        Check("slot.port", tenancy.slot.port == LANE_PORT, f"port={tenancy.slot.port}")
    )

    # 3. Tenancy <-> policy consistency (the tenant-swap hazard).
    problems = cross_validate(tenancy, policy)
    checks.append(
        Check(
            "policy.cross_validate",
            not problems,
            "; ".join(problems) if problems else "every tenant has a matching policy row",
        )
    )

    # 4. Saturation cap holds across every tenant/mode/row. The loader enforces
    #    this, so a failure here means the loader was bypassed.
    over: list[str] = []
    for tenant_id, row in policy.tenants.items():
        for mode, mode_policy in row.modes.items():
            for budget in mode_policy.budgets:
                for bucket, ceiling in budget.ceilings.items():
                    if ceiling is not None and ceiling > row.np_throughput_saturation:
                        over.append(f"{tenant_id}/{mode}/{budget.name}/{bucket}={ceiling}")
    checks.append(
        Check("policy.saturation_cap", not over, "; ".join(over) if over else "all rows capped")
    )

    # 5. Every tenant resolves to a plan whose refusals are explainable, and the
    #    default planning cell is admissible for at least one tenant.
    admissible: list[str] = []
    for tenant_id, tenant in tenancy.tenants.items():
        profile = (
            "solo_resident" if tenant.co_residency == "forbidden" else "phase2_resident_set"
        )
        plan = resolve_lane_plan(
            tenancy,
            policy,
            tenant_id=tenant_id,
            budget_profile=profile,
            np_slots=8,
            slot_context_tokens=8192,
        )
        if plan.admissible:
            admissible.append(tenant_id)
    checks.append(
        Check(
            "plan.some_tenant_admissible",
            bool(admissible),
            f"admissible at np8 x 8192: {admissible or 'NONE'}",
        )
    )

    # 6. Refusal really refuses: an np above the ceiling must be rejected.
    guard = resolve_lane_plan(
        tenancy,
        policy,
        tenant_id="qwen36_27b_stock_q8",
        budget_profile="phase2_resident_set",
        np_slots=32,
        slot_context_tokens=32768,
    )
    checks.append(
        Check(
            "plan.refuses_over_ceiling",
            not guard.admissible,
            f"np32 x 32768 refusals: {list(guard.refusals) or 'NONE (BUG)'}",
        )
    )

    # 6a. The compiled SERVING SHAPE is inside the ceiling table.
    #
    # P2-6's load_serving_shape() feeds orchestration/gpu_shadow_lane_np_ceiling
    # .yaml's `serving_shape` block through stack-priors into the builder's real
    # -np/-c. It validates only that np_slots is a measured np LEVEL and that
    # the context is positive — it never consults the ceiling rows in the same
    # file. So `np_slots: 32, slot_context_tokens: 32768` would compile into a
    # live launch while every tenant's ceiling refuses it (32 also exceeds the
    # saturation cap of 16). The two blocks can drift silently.
    #
    # Closing that from the Stage-0 side, because the loader lives in a file
    # another session is actively editing. The right long-term home is the
    # loader itself; tracked as a follow-up.
    try:
        shape = load_serving_shape()
    except Exception as exc:  # noqa: BLE001 — any failure is a smoke failure
        checks.append(Check("serving_shape.loadable", False, f"{type(exc).__name__}: {exc}"))
    else:
        offenders: list[str] = []
        for tenant_id, tenant in tenancy.tenants.items():
            if tenant.status != "bake_off_arm":
                continue  # only arms that can actually become resident
            profile = (
                "solo_resident"
                if tenant.co_residency == "forbidden"
                else "phase2_resident_set"
            )
            plan = resolve_lane_plan(
                tenancy,
                policy,
                tenant_id=tenant_id,
                budget_profile=profile,
                np_slots=shape["np_slots"],
                slot_context_tokens=shape["slot_context_tokens"],
            )
            if not plan.admissible:
                offenders.append(f"{tenant_id}: {'; '.join(plan.refusals)}")
        checks.append(
            Check(
                "serving_shape.within_ceiling",
                not offenders,
                (
                    f"-np {shape['np_slots']} x {shape['slot_context_tokens']} "
                    f"(total -c {shape['context_tokens']})"
                    + (f" REFUSED BY: {offenders}" if offenders else " admissible for every arm")
                ),
            )
        )

    # 7. The recert set is complete — specifically, it contains at least one
    #    role a string-overlap check would have missed.
    roles = recert_roles(tenancy.slot.host_cpuset)
    missed = [item for item in roles if item.basis == "physical_core_overlap"]
    checks.append(
        Check(
            "recert.set_nonempty",
            bool(roles),
            f"{len(roles)} contending instance(s): "
            + ", ".join(f"{item.role}:{item.port}" for item in roles),
        )
    )
    checks.append(
        Check(
            "recert.catches_smt_blind_spot",
            bool(missed),
            "physical-core-only overlaps a string check would miss: "
            + (", ".join(f"{item.role}:{item.port}" for item in missed) or "none"),
        )
    )

    # 8. Attestation judges score their recorded fixtures as expected. This is
    #    what makes the Step-5 attestation code reviewable BEFORE activation
    #    rather than written live against a running server.
    checks.extend(_fixture_checks(fixture_dir))
    return checks


def _fixture_checks(fixture_dir: Path) -> list[Check]:
    checks: list[Check] = []
    cases = (
        ("health_ok.json", judge_health, {"expect_slots": 8}, True),
        ("health_degraded.json", judge_health, {"expect_slots": 8}, False),
        ("affinity_ok.json", judge_affinity, {}, True),
        ("affinity_wrong_cpuset.json", judge_affinity, {}, False),
        (
            "vram_ok.json",
            judge_vram,
            {"model_vram_gib": 26.7, "estimated_dynamic_gib": 20.4},
            True,
        ),
        (
            "vram_short.json",
            judge_vram,
            {"model_vram_gib": 26.7, "estimated_dynamic_gib": 20.4},
            False,
        ),
    )
    for filename, judge, kwargs, expect_pass in cases:
        path = fixture_dir / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            checks.append(Check(f"fixture.{filename}", False, f"unreadable: {exc}"))
            continue
        results = judge(payload, **kwargs)
        actually_passed = all(item.passed for item in results)
        failed = [item.name for item in results if not item.passed]
        checks.append(
            Check(
                f"fixture.{filename}",
                actually_passed == expect_pass,
                (
                    f"expected {'PASS' if expect_pass else 'FAIL'}, "
                    f"got {'PASS' if actually_passed else 'FAIL'}"
                    + (f" (failing: {failed})" if failed else "")
                ),
            )
        )
    return checks


# ── CLI ──────────────────────────────────────────────────────────────────────


def _load_both(args: argparse.Namespace) -> tuple[Tenancy, NpCeilingPolicy]:
    # Stage-0 is itself the guarded tool: it loads with an explicit override so
    # it can run before the flag is flipped anywhere else.
    feats = Features(gpu_shadow_lane=True)
    return (
        load_tenancy(args.tenancy_path, feats=feats),
        load_np_ceiling_policy(args.policy_path, feats=feats),
    )


def _emit(report: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "stage0.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if not args.summary_only:
        print(json.dumps(report, indent=2, sort_keys=True))


def cmd_smoke(args: argparse.Namespace) -> int:
    tenancy, policy = _load_both(args)
    checks = smoke_checks(tenancy, policy)
    failed = [check for check in checks if not check.passed]
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "smoke",
        "status": "fail" if failed else "pass",
        "checks": [asdict(check) for check in checks],
    }
    _emit(report, args)
    for check in checks:
        print(f"[{check.marker}] {check.name}: {check.detail}", file=sys.stderr)
    return 75 if failed else 0


def cmd_attest(args: argparse.Namespace) -> int:
    raw = (
        json.loads(args.input.read_text(encoding="utf-8"))
        if args.input
        else json.loads(sys.stdin.read())
    )
    tenancy, policy = _load_both(args)
    plan = resolve_lane_plan(
        tenancy,
        policy,
        tenant_id=args.tenant,
        budget_profile=args.budget_profile,
        np_slots=args.np,
        slot_context_tokens=args.slot_context,
    )
    checks: list[Check] = []
    checks.extend(judge_health(raw.get("health_snapshot"), expect_slots=args.np))
    checks.extend(judge_affinity(raw.get("affinity_snapshot")))
    checks.extend(
        judge_vram(
            raw.get("vram_snapshot"),
            model_vram_gib=plan.model_vram_gib,
            estimated_dynamic_gib=plan.estimated_dynamic_gib,
        )
    )
    failed = [check for check in checks if not check.passed]
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "attest",
        "status": "fail" if failed else "pass",
        "plan": {
            "tenant_id": plan.tenant_id,
            "mode": plan.mode,
            "np_slots": plan.np_slots,
            "slot_context_tokens": plan.slot_context_tokens,
            "refusals": list(plan.refusals),
        },
        "checks": [asdict(check) for check in checks],
    }
    _emit(report, args)
    for check in checks:
        print(f"[{check.marker}] {check.name}: {check.detail}", file=sys.stderr)
    return 75 if failed else 0


def cmd_recert(args: argparse.Namespace) -> int:
    tenancy, _policy = _load_both(args)
    roles = recert_roles(tenancy.slot.host_cpuset)
    command = recert_command(roles)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "recert",
        "status": "plan_only",
        "lane_host_cpuset": tenancy.slot.host_cpuset,
        "lane_physical_cores": sorted(fold_smt_to_physical(tenancy.slot.host_cpuset)),
        "lane_regions": sorted(lane_host_regions(tenancy.slot.host_cpuset)),
        "contention_floor": CONTENTION_FLOOR,
        "recert_set": [asdict(item) for item in roles],
        "planned_command": command,
        "note": (
            "PLAN ONLY — this prints the command, it does not bench. Running it "
            "requires an operator grant and a quiet window (MEASUREMENT policy). "
            "Entries with basis=physical_core_overlap are the ones a raw string "
            "overlap of the lane cpuset would have missed."
        ),
    }
    _emit(report, args)
    print(" ".join(command), file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenancy-path", type=Path, default=None)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke", help="deterministic fixture-driven Stage-0 self-check")

    p_attest = sub.add_parser("attest", help="judge health/affinity/VRAM snapshots")
    p_attest.add_argument("--input", type=Path, default=None, help="JSON file (default: stdin)")
    p_attest.add_argument("--tenant", default="qwen36_27b_stock_q8")
    p_attest.add_argument("--budget-profile", default="phase2_resident_set")
    p_attest.add_argument("--np", type=int, default=8)
    p_attest.add_argument("--slot-context", type=int, default=8192)

    sub.add_parser("recert", help="emit the complete contention recert set (plan only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    handlers = {"smoke": cmd_smoke, "attest": cmd_attest, "recert": cmd_recert}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
