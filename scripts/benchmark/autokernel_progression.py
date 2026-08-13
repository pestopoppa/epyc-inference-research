"""Build the operator-facing AutoKernel discovery/progression snapshot.

This is a projection of immutable campaign receipts.  It never promotes a
candidate and never replaces the strict terminal campaign contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "epyc.autokernel.progression.v1"
DEFAULT_ROOT = Path("/mnt/raid0/llm/autokernel")
DEFAULT_OUTPUT = DEFAULT_ROOT / "surface" / "kernel_progression.json"


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": _sha(path),
        "receipt_sha256": receipt.get("result_sha256"),
        "campaign_id": receipt.get("campaign_id"),
    }


def _cpu_screen(path: Path, receipt: dict[str, Any]) -> dict[str, Any] | None:
    report = receipt.get("screening_report")
    spec = receipt.get("spec")
    if not (receipt.get("schema") == "epyc.autokernel.campaign_result.v1"
            and receipt.get("screening_only") is True
            and receipt.get("non_promotable") is True
            and receipt.get("ok") is True and receipt.get("state") == "decided"
            and not receipt.get("error") and isinstance(report, dict)
            and report.get("candidate_invocations") == 3
            and report.get("nomination") == "top_k_candidate_only_not_a_keep"
            and isinstance(spec, dict)):
        return None
    factor = report.get("sole_intended_factor") or {}
    workload = ("decode tg128" if str(spec.get("metric", "")).startswith("decode_")
                else f"prefill pp{spec.get('n_prompt', '—')}")
    effect = report.get("median_relative")
    if not isinstance(effect, (int, float)):
        return None
    return {
        "key": f"cpu:{factor.get('name')}:{workload}",
        "lane": "CPU", "candidate": str(factor.get("name") or receipt.get("candidate_id")),
        "transition": f"{factor.get('anchor', '—')} → {factor.get('candidate', '—')}",
        "workload": workload, "metric": spec.get("metric"),
        "metric_direction": "higher_better", "effect_fraction": effect,
        "evidence_tier": "screening", "stage": "candidate",
        "confidence": "directional / unquantified noise (3 candidate calls)",
        "current_gate": "strict matched confirmation",
        "next_action": "Run a calibrated paired confirmation; keep promotion authority false.",
        "promotable": False, "champion": False,
        "observed_at": spec.get("created_at"),
        "evidence": [_evidence(path, receipt)],
    }


def _gpu_screen(path: Path, receipt: dict[str, Any]) -> dict[str, Any] | None:
    factor = receipt.get("sole_factor") or receipt.get("sole_build_factor")
    candidate_calls = receipt.get("candidate_invocations")
    anchor_calls = receipt.get("anchor_invocations")
    if not (receipt.get("schema") in {
                "epyc.autokernel.gpu_candidate_only_screen.v1",
                "epyc.autokernel.gpu_candidate_only_screen.v2"}
            and receipt.get("non_promotable") is True
            and receipt.get("ok") is True and receipt.get("state") == "decided"
            and candidate_calls in {3, 5, 9}
            and anchor_calls == candidate_calls
            and receipt.get("hip_residency_proved") is True
            and isinstance(factor, dict)
            and isinstance(receipt.get("median_relative"), (int, float))):
        return None
    runs = receipt.get("candidate_runs") or []
    observed = None
    for run in runs:
        row = run.get("raw_row") if isinstance(run, dict) else None
        if isinstance(row, dict) and isinstance(row.get("test_time"), str):
            observed = max(observed or row["test_time"], row["test_time"])
    effects = receipt.get("relative_effects") or []
    numeric_effects = [float(value) for value in effects
                       if isinstance(value, (int, float))]
    sign_conflict = bool(numeric_effects
                         and min(numeric_effects) < 0 < max(numeric_effects))
    effect_spread = (max(numeric_effects) - min(numeric_effects)
                     if numeric_effects else None)
    overlap_policy = receipt.get("cpu_overlap_policy")
    noisy_overlap = bool(
        overlap_policy == "allowed_discovery_noise"
        and (sign_conflict or (effect_spread is not None and effect_spread >= 0.10))
    )
    stage = ("inconclusive" if noisy_overlap else
             "screened_out" if receipt["median_relative"] <= 0 else "candidate")
    confidence = (f"directional / unquantified noise ({anchor_calls}+{candidate_calls} "
                  "calls; HIP resident)")
    current_gate = "correctness + strict matched confirmation"
    next_action = "Confirm correctness, then run promotion-grade paired evidence."
    if noisy_overlap:
        conflict = "sign-conflicted" if sign_conflict else "high-dispersion"
        spread = f"; {effect_spread * 100:.2f} pp spread" if effect_spread is not None else ""
        confidence = (
            f"inconclusive: CPU-overlap discovery noise; {conflict} "
            f"{anchor_calls}+{candidate_calls} vector{spread}"
        )
        current_gate = "clean/windowed discovery retest"
        next_action = (
            "Retest under shared model-call windows; do not interpret the median as "
            "a negative conclusion."
        )
    elif stage == "screened_out":
        current_gate = "screened out by sign-consistent discovery result"
        next_action = "Retain as an abandoned strategy; revisit only with a new rationale."
    frame = receipt.get("frame") or {}
    recipe = frame.get("recipe", "pp512-ngl99")
    workload = ("MI210 decode tg128" if recipe == "tg128-ngl99"
                else "MI210 prefill pp512")
    metric = frame.get("metric", "prefill_tokens_per_s")
    short_workload = "decode tg128" if recipe == "tg128-ngl99" else "prefill pp512"
    return {
        "key": (f"gpu:{factor.get('name')}:{factor.get('anchor')}->"
                f"{factor.get('candidate')}:{short_workload}"),
        "lane": "GPU", "candidate": str(factor.get("name")),
        "transition": f"{factor.get('anchor', '—')} → {factor.get('candidate', '—')}",
        "workload": workload, "metric": metric,
        "metric_direction": "higher_better",
        "effect_fraction": receipt["median_relative"],
        "evidence_tier": "screening", "stage": stage,
        "confidence": confidence,
        "noise": {
            "cpu_overlap_policy": overlap_policy,
            "sign_conflict": sign_conflict,
            "effect_spread_fraction": effect_spread,
        },
        "current_gate": current_gate,
        "next_action": next_action,
        "promotable": False, "champion": False, "observed_at": observed,
        "evidence": [_evidence(path, receipt)],
    }


def _strict(path: Path, receipt: dict[str, Any]) -> dict[str, Any] | None:
    decision = receipt.get("decision")
    spec = receipt.get("spec")
    if not (receipt.get("schema") == "epyc.autokernel.campaign_result.v1"
            and receipt.get("screening_only") is False
            and receipt.get("ok") is True and receipt.get("state") == "decided"
            and not receipt.get("error") and isinstance(decision, dict)
            and isinstance(spec, dict) and "aa-control" not in str(receipt.get("campaign_id"))):
        return None
    effect = decision.get("median_relative")
    if not isinstance(effect, (int, float)):
        return None
    keep = decision.get("keep") is True
    return {
        "key": "cpu:GGML_IQK:prefill pp512", "lane": "CPU",
        "candidate": "GGML_IQK", "transition": "0 → 1",
        "workload": "prefill pp512", "metric": spec.get("metric"),
        "metric_direction": "higher_better", "effect_fraction": effect,
        "evidence_tier": "strict", "stage": "strict_keep" if keep else "rejected",
        "confidence": f"calibrated paired ({decision.get('blocks', '—')} blocks)",
        "current_gate": "held-out matched regime" if keep else "closed by strict decision",
        "next_action": ("Run the required distinct-regime held-out pair."
                        if keep else "Retain as rejected evidence; do not promote."),
        "promotable": False, "champion": False,
        "observed_at": spec.get("created_at"),
        "evidence": [_evidence(path, receipt)],
    }


def build_progression(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    screens = root / "screens"
    for path in sorted(screens.glob("*/result.json")):
        receipt = _load(path)
        if receipt is None:
            continue
        item = _cpu_screen(path, receipt) or _gpu_screen(path, receipt)
        if item is not None:
            candidates[item["key"]] = item
    for path in sorted(root.glob("r*-execute.json")):
        receipt = _load(path)
        if receipt is None:
            continue
        item = _strict(path, receipt)
        if item is not None:
            prior = candidates.get(item["key"])
            if prior:
                item["evidence"] = prior["evidence"] + item["evidence"]
            candidates[item["key"]] = item

    # A preflight without a terminal result is an unexplored/queued hypothesis,
    # not a candidate and never a performance observation.
    unexplored = []
    completed_ids = {e["campaign_id"] for item in candidates.values()
                     for e in item["evidence"]}
    completed_gpu_factors = {
        item["candidate"] for item in candidates.values() if item["lane"] == "GPU"
    }
    latest_preflight: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(screens.glob("*.preflight.json")):
        receipt = _load(path)
        factor = receipt.get("sole_factor") if receipt else None
        if not (isinstance(receipt, dict) and isinstance(factor, dict)
                and receipt.get("inference_executed") is False
                and receipt.get("campaign_id") not in completed_ids
                and str(factor.get("name")) not in completed_gpu_factors):
            continue
        key = str(factor.get("name"))
        prior = latest_preflight.get(key)
        if prior is None or path.stat().st_mtime_ns > prior[0].stat().st_mtime_ns:
            latest_preflight[key] = (path, receipt)
    for path, receipt in latest_preflight.values():
        factor = receipt["sole_factor"]
        unexplored.append({
            "lane": "GPU", "hypothesis": factor.get("name"),
            "workload": "MI210 prefill pp512", "state": "preflight_ready",
            "next_action": "Execute the sealed 3+3 nonpromotable screen.",
            "evidence": _evidence(path, receipt),
        })

    ordered = sorted(candidates.values(),
                     key=lambda row: (row["lane"], -row["effect_fraction"], row["candidate"]))
    funnel = {
        "candidate": sum(row["stage"] in {"candidate", "inconclusive"}
                         for row in ordered),
        "strict_keep": sum(row["stage"] == "strict_keep" for row in ordered),
        "champion": sum(row["champion"] for row in ordered),
        "promotable": sum(row["promotable"] for row in ordered),
    }
    observed = [row.get("observed_at") for row in ordered if row.get("observed_at")]
    return {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "observed_through": max(observed) if observed else None,
        "authority": "presentation_projection_only",
        "promotion_claim": False, "candidates": ordered, "funnel": funnel,
        "strategy": {
            "pursued": [row for row in ordered if row["stage"] == "candidate"],
            "accepted": [row for row in ordered if row["stage"] == "strict_keep"],
            "abandoned": [row for row in ordered
                          if row["stage"] in {"rejected", "screened_out", "inconclusive"}],
        },
        "unexplored": unexplored,
    }


def export_progression(*, root: Path = DEFAULT_ROOT,
                       output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    document = build_progression(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    return document


if __name__ == "__main__":
    print(json.dumps(export_progression(), indent=2, sort_keys=True))
