"""U3-SEED: build the `epyc.autokernel.variance_input.v1` document `variance_keep` consumes.

Pure and hermetic: no I/O, no clock, no host. The only source is the already-recorded paired
RUNTIME_CONFIG serving comparison (`epyc.autokernel.serving_runtime_ab.v1`, which is what
`serving.compare` writes when a `runtime_pair` is supplied: separate-process anchor and candidate
launches in a matched, seeded order, `anchor_samples` / `candidate_samples` per launch,
`launch_membership` binding every launch to its (pair_index, arm), `recipe_hash` and
`candidate_recipe_hash`). Session unit i = pair i; the pairing is never broken or re-derived.

This module REFUSES (typed `VarianceInputRefused.code`) anything that is not an exactly paired,
recipe-consistent, session-unit record. It does NOT re-implement any decision rule: the verdict,
including `insufficient_n` below 14 pairs, is `variance_keep.evaluate`'s own, returned beside the
document. A short record still yields a document (so the refusal reason is the rule's, not ours).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from . import variance_keep as vk

SOURCE_SCHEMA = "epyc.autokernel.serving_runtime_ab.v1"
SESSION_UNIT = vk.SESSION_UNIT

C_SCHEMA = "unsupported_schema"
C_NOT_MATCHED = "pairing_unverifiable"
C_UNPAIRED = "unpaired_launches"
C_RECIPE = "recipe_hash_mismatch"
C_UNIT = "unit_mismatch"
C_FLOOR = "missing_floor"
C_VALUE = "bad_launch_value"
C_CANDIDATE = "bad_candidate_id"


class VarianceInputRefused(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass(frozen=True)
class ProducedVarianceInput:
    document: dict
    verdict: dict


def _launch_values(record: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    out = []
    for key in ("anchor_samples", "candidate_samples"):
        values = record.get(key)
        if not isinstance(values, list) or not values:
            raise VarianceInputRefused(C_UNPAIRED, f"{key} is missing or empty")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise VarianceInputRefused(C_VALUE, f"{key} must be finite positive throughputs")
        out.append([float(v) for v in values])
    return out[0], out[1]


def _check_pairing(record: Mapping[str, Any], n_anchor: int, n_candidate: int) -> list[int]:
    """Return the sample index of each arm per pair, proving every pair has both separate launches."""
    membership = record.get("launch_membership")
    if not isinstance(membership, list) or not membership:
        raise VarianceInputRefused(C_NOT_MATCHED, "record carries no launch_membership; "
                                   "pairing cannot be verified (matched instrument required)")
    if n_anchor != n_candidate:
        raise VarianceInputRefused(C_UNPAIRED, f"{n_anchor} anchor launches vs {n_candidate} candidate")
    n = n_anchor
    if len(membership) != 2 * n:
        raise VarianceInputRefused(C_UNPAIRED, "launch_membership does not cover every launch exactly once")
    ordinals = Counter(row.get("ordinal") for row in membership if isinstance(row, Mapping))
    if len(ordinals) != 2 * n or any(count != 1 for count in ordinals.values()):
        raise VarianceInputRefused(C_UNPAIRED, "duplicate or missing launch ordinals (one process per launch)")
    by_pair: dict[int, dict[str, int]] = {}
    for row in membership:
        if (not isinstance(row, Mapping) or row.get("arm") not in ("anchor", "candidate")
                or type(row.get("pair_index")) is not int or type(row.get("arm_sample_index")) is not int):
            raise VarianceInputRefused(C_UNPAIRED, "malformed launch_membership row")
        slot = by_pair.setdefault(row["pair_index"], {})
        if row["arm"] in slot:
            raise VarianceInputRefused(C_UNPAIRED, f"pair {row['pair_index']} has two {row['arm']} launches")
        slot[row["arm"]] = row["arm_sample_index"]
    if sorted(by_pair) != list(range(n)) or any(set(slot) != {"anchor", "candidate"} for slot in by_pair.values()):
        raise VarianceInputRefused(C_UNPAIRED, "every pair needs exactly one anchor and one candidate launch")
    for arm in ("anchor", "candidate"):
        if sorted(slot[arm] for slot in by_pair.values()) != list(range(n)):
            raise VarianceInputRefused(C_UNPAIRED, f"{arm} sample indices are not a permutation of the arm's launches")
    return [by_pair[i] for i in range(n)]


def produce(record: Mapping[str, Any], *, candidate_id: str, floor: Mapping[str, Any] | None,
            unit: str = SESSION_UNIT) -> ProducedVarianceInput:
    """Closed `variance_input.v1` document + variance_keep's own verdict for it."""
    if not isinstance(record, Mapping) or record.get("schema") != SOURCE_SCHEMA \
            or not isinstance(record.get("runtime_pair"), Mapping):
        raise VarianceInputRefused(C_SCHEMA, f"only a paired RUNTIME_CONFIG {SOURCE_SCHEMA} record is consumed")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise VarianceInputRefused(C_CANDIDATE, "candidate_id must be a non-empty string")
    if unit != SESSION_UNIT or record.get("effect_unit") not in (None, "process", SESSION_UNIT):
        raise VarianceInputRefused(C_UNIT, "the resampling unit must be the session; got "
                                   f"unit={unit!r} effect_unit={record.get('effect_unit')!r}")
    if not isinstance(floor, Mapping) or isinstance(floor.get("pct"), bool) \
            or not isinstance(floor.get("pct"), (int, float)) or not math.isfinite(floor["pct"]) \
            or floor["pct"] <= 0:
        raise VarianceInputRefused(C_FLOOR, "a positive session-unit floor percent is required")
    if floor.get("unit") != SESSION_UNIT:
        raise VarianceInputRefused(C_UNIT, f"floor unit {floor.get('unit')!r} is not the session")
    anchor_hash, candidate_hash = record.get("recipe_hash"), record.get("candidate_recipe_hash")
    if not isinstance(anchor_hash, str) or not isinstance(candidate_hash, str) \
            or floor.get("recipe_hash") != anchor_hash:
        raise VarianceInputRefused(C_RECIPE, "floor recipe_hash must equal the record's anchor recipe_hash "
                                   "and both recipe hashes must be present")
    anchor, candidate = _launch_values(record)
    _check_pairing(record, len(anchor), len(candidate))
    # Pair i is (anchor_samples[i], candidate_samples[i]): launch_membership proved each arm's
    # sample indices are exactly the pair indices' permutation; the record appends per arm in
    # pair order, so index == pair_index. Re-verify that rather than assume it.
    by_arm = {(row["arm"], row["pair_index"]): row["arm_sample_index"] for row in record["launch_membership"]}
    units = [{"anchor": anchor[by_arm[("anchor", i)]], "candidate": candidate[by_arm[("candidate", i)]]}
             for i in range(len(anchor))]
    document = {"schema": vk.INPUT_SCHEMA, "candidate_id": candidate_id, "floor_pct": float(floor["pct"]),
                "floor_unit": SESSION_UNIT, "units": units}
    verdict = vk.evaluate(units, floor_pct=document["floor_pct"], floor_unit=document["floor_unit"])
    return ProducedVarianceInput(document, verdict)


def dumps(document: Mapping[str, Any]) -> str:
    """Canonical text for `run.py --variance-input` (load_input reads exactly this)."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


__all__ = ["ProducedVarianceInput", "VarianceInputRefused", "produce", "dumps", "SOURCE_SCHEMA"]
