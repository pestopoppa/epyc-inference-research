"""Pure native mutation pair features; selection and acceptance remain separate policies."""
from __future__ import annotations
from typing import Any


def overlap_reasons(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if a.get("subsystem") and a.get("subsystem") == b.get("subsystem"):
        reasons.append(f"same subsystem {a['subsystem']}")
    for key, label in (("files_touched", "file"), ("prompt_sections_touched", "prompt section")):
        shared = sorted(set(a.get(key) or []) & set(b.get(key) or []))
        if shared:
            reasons.append(f"shared {label}: {', '.join(shared[:4])}")
    shared_flags = sorted(set((a.get("feature_flags") or {}).keys()) &
                          set((b.get("feature_flags") or {}).keys()))
    if shared_flags:
        reasons.append(f"shared feature flag: {', '.join(shared_flags[:4])}")
    return reasons


def compare_mutation_features(new: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    """Project recorded pair facts only; does not assign conflict or donor severity."""
    nd = new.get("behavior_signature_delta") or {}
    od = prior.get("behavior_signature_delta") or {}
    ni, oi = set(nd.get("improved_sentinels") or []), set(od.get("improved_sentinels") or [])
    nr, ore = set(nd.get("regressed_sentinels") or []), set(od.get("regressed_sentinels") or [])
    nc, oc = set(nd.get("changed_fields") or []), set(od.get("changed_fields") or [])
    return {
        "overlap_reasons": overlap_reasons(new, prior),
        "delta_blocking": nd.get("severity") == "blocking" or od.get("severity") == "blocking",
        "different_surfaces": bool(nc and oc and nc != oc),
        "new_changed": sorted(nc), "prior_changed": sorted(oc),
        "disjoint_improvements": bool(ni and oi and ni.isdisjoint(oi)),
        "opposing_movement": bool(nr & oi or ore & ni),
        "has_regressions": bool(nr or ore),
        "new_improvements": sorted(ni), "prior_improvements": sorted(oi),
    }
