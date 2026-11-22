"""AKX-P3b: leave-one-out restricted to TOUCHED workloads per keep, plus the AK-GH-1 count.

A keep whose blast-radius screen classes a workload INERT cannot change that workload, so it
needs no LOO arm there (a direct saving over n_keeps x n_surfaces). TOUCHED and UNKNOWN
workloads both owe an arm: UNKNOWN is not evidence of inertness. The AK-GH-1 count
(intake-1378#record) is then taken over TOUCHED-workload arms only and goes on the promotion
record together with the LOO receipts. Pure functions; no compute, no store access.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from . import goodhart_count

SCHEMA = "epyc.autokernel.loo_touched_record.v1"
REQUIRED_FIELDS = ("goodharted_count", "loo_receipts", "loo_plan")


class LOOTouchedRefusal(ValueError):
    """The LOO plan, count or promotion record is incomplete; the message names why."""


def loo_plan(screens: Mapping[str, Mapping]) -> dict:
    """screens: {keep_id: screen} (keep_screen record `screen`: touched_ids/inert_ids/unknown_ids).

    Returns {keep_id: {"required": [...], "skipped_inert": [...]}}.
    """
    if not screens:
        raise LOOTouchedRefusal("no keep screens: LOO scope cannot be derived")
    plan = {}
    for kid, screen in screens.items():
        try:
            touched, inert, unknown = (list(screen[k]) for k in ("touched_ids", "inert_ids", "unknown_ids"))
        except (KeyError, TypeError) as exc:
            raise LOOTouchedRefusal(f"keep {kid}: screen lacks touched/inert/unknown ids") from exc
        if set(touched) & set(inert) or set(touched) & set(unknown) or set(inert) & set(unknown):
            raise LOOTouchedRefusal(f"keep {kid}: a workload has more than one class")
        required = sorted(set(touched) | set(unknown))
        if not required:
            # fully INERT keep: nothing to leave out; recorded, owes no arm.
            pass
        plan[kid] = {"required": required, "skipped_inert": sorted(inert)}
    return plan


def owed_keeps(keep_ids: Sequence[str], plan: Mapping[str, Mapping], surface_id: str) -> tuple:
    """Split `keep_ids` into (owed, skipped_inert) for one LOO surface.

    A keep owes an arm on `surface_id` when the surface is in its plan's `required`
    (TOUCHED or UNKNOWN); it is skipped only when the plan names the surface INERT. A keep
    absent from the plan, or a surface the plan classes neither way, REFUSES: the caller may
    not pick surfaces itself.
    """
    if not isinstance(surface_id, str) or not surface_id:
        raise LOOTouchedRefusal("LOO surface id is required to apply the TOUCHED scope")
    owed, skipped = [], []
    for kid in keep_ids:
        entry = plan.get(kid)
        if entry is None:
            raise LOOTouchedRefusal(f"keep {kid}: absent from the TOUCHED-scoped LOO plan")
        if surface_id in entry.get("required", ()):
            owed.append(kid)
        elif surface_id in entry.get("skipped_inert", ()):
            skipped.append(kid)
        else:
            raise LOOTouchedRefusal(f"keep {kid}: surface {surface_id} is not classed by its screen")
    return owed, skipped


def count_touched(arms: Sequence[Mapping], *, full_bundle: Mapping[str, float],
                  plan: Mapping[str, Mapping]) -> dict:
    """AK-GH-1 count over TOUCHED-workload LOO arms only.

    Every keep in `plan` needs an arm and every required workload needs a measured LOO value;
    a missing one REFUSES (never counted as clean). Arm values on workloads outside the
    required set (INERT) are ignored and reported as `ignored_surfaces`.
    """
    by_id = {}
    for arm in arms:
        kid = arm.get("keep_id")
        if kid in by_id:
            raise LOOTouchedRefusal(f"duplicate LOO arm for keep {kid}")
        by_id[kid] = arm
    extra = sorted(k for k in by_id if k not in plan)
    if extra:
        raise LOOTouchedRefusal(f"LOO arm for keep(s) absent from the plan: {extra}")
    restricted, ignored = [], {}
    for kid, p in plan.items():
        required = p["required"]
        if not required:
            continue
        arm = by_id.get(kid)
        if arm is None:
            raise LOOTouchedRefusal(f"keep {kid}: no LOO arm but {required} are TOUCHED/UNKNOWN")
        loo = arm.get("loo") or {}
        missing = sorted(set(required) - set(loo))
        if missing:
            raise LOOTouchedRefusal(f"keep {kid}: missing LOO value for TOUCHED workload(s) {missing}")
        ignored[kid] = sorted(set(loo) - set(required))
        restricted.append({**arm, "loo": {s: loo[s] for s in required}})
    if not restricted:
        raise LOOTouchedRefusal("every keep is INERT everywhere: no TOUCHED LOO arm to count")
    try:
        result = goodhart_count.count_goodharted(restricted, full_bundle=full_bundle)
    except ValueError as exc:
        raise LOOTouchedRefusal(str(exc)) from exc
    result["ignored_surfaces"] = {k: v for k, v in ignored.items() if v}
    return result


def promotion_loo_record(*, plan: Mapping[str, Mapping], count: Mapping,
                         loo_receipts: Sequence[str]) -> dict:
    """Assemble the promotion-record LOO block; refuses without the count or receipts."""
    record = {"schema": SCHEMA, "loo_plan": dict(plan),
              "goodharted_count": count.get("goodharted_count"),
              "goodharted_keep_ids": list(count.get("goodharted_keep_ids", [])),
              "loo_receipts": list(loo_receipts)}
    require_fields(record)
    return record


def require_fields(record: Mapping) -> None:
    """The required-field rule: no count or no LOO receipts means the record refuses."""
    count = record.get("goodharted_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise LOOTouchedRefusal("promotion record lacks the AK-GH-1 goodharted_count")
    receipts = record.get("loo_receipts")
    if not isinstance(receipts, (list, tuple)) or not receipts \
            or any(not isinstance(r, str) or not r for r in receipts):
        raise LOOTouchedRefusal("promotion record lacks LOO receipts")
    if not record.get("loo_plan"):
        raise LOOTouchedRefusal("promotion record lacks the TOUCHED-scoped LOO plan")
    owed = sum(1 for p in record["loo_plan"].values() if p.get("required"))
    if len(receipts) < owed:
        raise LOOTouchedRefusal(f"{len(receipts)} LOO receipts < {owed} keeps owing an arm")
