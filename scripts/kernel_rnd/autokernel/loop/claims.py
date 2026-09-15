"""Typed claims carried by AutoKernel experiment records.

An A/B establishes an effect of the measured tree.  It does not establish the
planner's causal explanation for that effect.  Keep those two facts separate so
retrieval cannot silently upgrade narrative into verified mechanism evidence.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA = "epyc.autokernel.keep_claims.v1"
VERIFIED = "verified"
HYPOTHESIS = "hypothesis"
UNVERIFIED = "unverified"


def _oracle_passed(gates: Sequence[Mapping[str, Any]]) -> bool:
    """Require a positive non-build gate; absence or ambiguity fails closed."""
    non_oracles = {"configure", "compile", "build"}
    oracle = [row for row in gates
              if isinstance(row.get("gate"), str)
              and row["gate"].lower() not in non_oracles]
    return bool(oracle) and all(row.get("passed") is True for row in oracle)


def keep_claims(*, status: str, mechanism_id: str, statement: str,
                comparison: Mapping[str, Any] | None,
                gates: Sequence[Mapping[str, Any]],
                ablation: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Build additive claims for a keep, with strict verification boundaries."""
    if status != "kept":
        return None
    paired_ab = (isinstance(comparison, Mapping)
                 and isinstance(comparison.get("pairs"), int)
                 and comparison["pairs"] > 0)
    effect_verified = paired_ab and _oracle_passed(gates)
    ablation_verified = (
        isinstance(ablation, Mapping)
        and ablation.get("status") == VERIFIED
        and isinstance(ablation.get("evidence"), Mapping)
        and bool(ablation["evidence"])
    )
    return {
        "schema": SCHEMA,
        "effect": {
            "status": VERIFIED if effect_verified else UNVERIFIED,
            "basis": ["oracle", "paired_ab"] if effect_verified else [],
        },
        "mechanism": {
            "status": VERIFIED if ablation_verified else HYPOTHESIS,
            "mechanism_id": mechanism_id,
            "statement": statement,
            "ablation": dict(ablation) if ablation_verified else None,
        },
    }


def mechanism_status(record: Mapping[str, Any]) -> str:
    """Return verified only for the exact typed claim shape; legacy is hypothesis."""
    claims = record.get("claims")
    if not isinstance(claims, Mapping) or claims.get("schema") != SCHEMA:
        return HYPOTHESIS
    mechanism = claims.get("mechanism")
    if not isinstance(mechanism, Mapping) or mechanism.get("status") != VERIFIED:
        return HYPOTHESIS
    ablation = mechanism.get("ablation")
    if (not isinstance(ablation, Mapping)
            or ablation.get("status") != VERIFIED
            or not isinstance(ablation.get("evidence"), Mapping)
            or not ablation["evidence"]):
        return HYPOTHESIS
    return VERIFIED


__all__ = ["HYPOTHESIS", "SCHEMA", "UNVERIFIED", "VERIFIED", "keep_claims",
           "mechanism_status"]
