"""Exact original serving-gate rows for a propagated whole-source candidate.

This adapter adds no numerical policy.  The intended source target uses the loop's
existing ``accumulate.classify_serving`` verdict; every other target uses the
existing FOLD-2 non-regression verdict, where a within-floor result is not a
regression claim.  Missing calibration never passes a row.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import accumulate, fold2_gates

SCHEMA = "epyc.autokernel.whole_source_serving_validation.v1"
DEBT_SCHEMA = "epyc.autokernel.whole_source_serving_validation_debt.v1"
REFERENCE_SCHEMA = "epyc.autokernel.whole_source_serving_validation_reference.v1"
MAX_BYTES = 16 << 20


class SurfaceValidationRefused(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def original_anchor_commit(build: Path, repo: Path) -> str:
    """Recover the existing build owner's original source commit without relabelling it."""
    from . import surface_fold
    build, repo = Path(build).resolve(), Path(repo).resolve()
    provenance = build / "provenance.json"
    identity = build / "IDENTITY.json"
    try:
        if provenance.exists():
            body = json.loads(surface_fold.bounded_regular_bytes(provenance, 1 << 20))
            commit = body.get("champion_commit") if isinstance(body, dict) else None
        else:
            body = json.loads(surface_fold.bounded_regular_bytes(identity, 1 << 20))
            required = {"schema", "kind", "head", "source", "source_status", "build",
                        "build_dir", "files"}
            if (not isinstance(body, dict) or set(body) != required
                    or body.get("schema") != "epyc.champion-candidate-build.v1"
                    or Path(str(body.get("source"))).resolve() != repo):
                raise SurfaceValidationRefused("original experimental build identity differs")
            commit = body.get("head")
    except (OSError, json.JSONDecodeError) as exc:
        raise SurfaceValidationRefused("original anchor source identity is unavailable") from exc
    if (not isinstance(commit, str) or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)):
        raise SurfaceValidationRefused("original anchor source commit is invalid")
    return commit


def classify(comparison: Mapping[str, Any], *, intended_target: bool) -> str:
    """Return the existing owner's exact three-valued validation disposition."""
    if not isinstance(comparison, Mapping) or comparison.get("schema") != \
            "epyc.autokernel.serving_ab.v1":
        raise SurfaceValidationRefused("validation requires an original serving A/B row")
    effect = comparison.get("effect")
    effect_pct = comparison.get("effect_pct")
    floor = comparison.get("noise_floor_pct")
    decisive = comparison.get("decisive")
    if (not isinstance(effect, (int, float)) or isinstance(effect, bool)
            or not math.isfinite(float(effect))
            or not isinstance(effect_pct, (int, float)) or isinstance(effect_pct, bool)
            or not math.isfinite(float(effect_pct))
            or not math.isclose(float(effect_pct), float(effect) * 100.0,
                                rel_tol=1e-12, abs_tol=1e-12)
            or not isinstance(floor, (int, float)) or isinstance(floor, bool)
            or not math.isfinite(float(floor)) or float(floor) < 0
            or type(decisive) is not bool):
        return "pending"
    if intended_target:
        return ("passed" if accumulate.classify_serving(dict(comparison))
                is accumulate.Outcome.PROMOTE else "failed")
    return "passed" if fold2_gates.ab_verdict(dict(comparison)) == "PASS" else "failed"


def row(*, source_commit: str, source_tree: str, source_keep_ids: list[str],
        target: Mapping[str, Any], original_anchor: Mapping[str, str],
        candidate_anchor: Mapping[str, str], request_digest: str,
        recipe_execution_digest: str, comparison: Mapping[str, Any],
        intended_target: bool) -> dict[str, Any]:
    values = {
        "source_commit": source_commit, "source_tree": source_tree,
        "source_keep_ids": list(source_keep_ids), "target": dict(target),
        "original_anchor": dict(original_anchor), "candidate_anchor": dict(candidate_anchor),
        "request_digest": request_digest,
        "recipe_execution_digest": recipe_execution_digest,
        "comparison": dict(comparison), "intended_target": intended_target,
    }
    result = classify(comparison, intended_target=intended_target)
    body = {"schema": SCHEMA, **values, "disposition": result}
    body["row_digest"] = _digest(body)
    return validate(body)


def debt(*, source_commit: str, source_tree: str, source_keep_ids: list[str],
         target: Mapping[str, Any], original_anchor: Mapping[str, Any],
         candidate_anchor: Mapping[str, str], request_digest: str,
         recipe_execution_digest: str, reason: str, failure: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason or len(reason) > 1024 \
            or not isinstance(failure, Mapping):
        raise SurfaceValidationRefused("validation debt reason/evidence is malformed")
    body = {"schema": DEBT_SCHEMA, "source_commit": source_commit,
        "source_tree": source_tree, "source_keep_ids": list(source_keep_ids),
        "target": dict(target), "original_anchor": dict(original_anchor),
        "candidate_anchor": dict(candidate_anchor), "request_digest": request_digest,
        "recipe_execution_digest": recipe_execution_digest, "disposition": "pending",
        "reason": reason, "failure": dict(failure)}
    body["row_digest"] = _digest(body)
    return validate_debt(body)


def validate(value: Any) -> dict[str, Any]:
    fields = {"schema", "source_commit", "source_tree", "source_keep_ids", "target",
              "original_anchor", "candidate_anchor", "request_digest",
              "recipe_execution_digest", "comparison", "intended_target",
              "disposition", "row_digest"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != SCHEMA:
        raise SurfaceValidationRefused("whole-source validation row has an open schema")
    body = json.loads(json.dumps(value, allow_nan=False))
    for key in ("source_commit", "source_tree", "request_digest", "recipe_execution_digest"):
        raw = body[key]
        expected_length = 40 if key in {"source_commit", "source_tree"} else 64
        if not isinstance(raw, str) or len(raw) != expected_length:
            raise SurfaceValidationRefused(f"invalid {key}")
        if any(char not in "0123456789abcdef" for char in raw):
            raise SurfaceValidationRefused(f"invalid {key}")
    if (not isinstance(body["source_keep_ids"], list)
            or not body["source_keep_ids"]
            or len(body["source_keep_ids"]) > 64
            or any(not isinstance(item, str) or not item for item in body["source_keep_ids"])):
        raise SurfaceValidationRefused("source keep membership is invalid")
    if not isinstance(body["target"], dict) or not body["target"]:
        raise SurfaceValidationRefused("target identity is invalid")
    for key in ("original_anchor", "candidate_anchor"):
        anchor = body[key]
        if (not isinstance(anchor, dict) or set(anchor) != {"path", "commit"}
                or not isinstance(anchor["path"], str) or not Path(anchor["path"]).is_absolute()
                or not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40):
            raise SurfaceValidationRefused(f"{key} is invalid")
    if type(body["intended_target"]) is not bool:
        raise SurfaceValidationRefused("intended-target marker is invalid")
    comparison = body["comparison"]
    belief = comparison.get("belief_capture") if isinstance(comparison, dict) else None
    inputs = belief.get("inputs") if isinstance(belief, dict) else None
    resolved_arms = inputs.get("resolved_arms") if isinstance(inputs, dict) else None
    candidate_resolved = resolved_arms.get("candidate") \
        if isinstance(resolved_arms, dict) else None
    if (body["candidate_anchor"]["commit"] != body["source_commit"]
            or comparison.get("request_digest") != body["request_digest"]
            or not isinstance(candidate_resolved, dict)
            or candidate_resolved.get("execution_digest") != body["recipe_execution_digest"]):
        raise SurfaceValidationRefused("validation source/request/recipe join differs")
    expected = classify(body["comparison"], intended_target=body["intended_target"])
    unsigned = {key: item for key, item in body.items() if key != "row_digest"}
    if body["disposition"] != expected or body["row_digest"] != _digest(unsigned):
        raise SurfaceValidationRefused("validation disposition or digest changed")
    if len(json.dumps(body, sort_keys=True).encode()) > MAX_BYTES:
        raise SurfaceValidationRefused("whole-source validation row exceeds byte budget")
    return body


def validate_debt(value: Any) -> dict[str, Any]:
    fields = {"schema", "source_commit", "source_tree", "source_keep_ids", "target",
              "original_anchor", "candidate_anchor", "request_digest",
              "recipe_execution_digest", "disposition", "reason", "failure", "row_digest"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != DEBT_SCHEMA:
        raise SurfaceValidationRefused("whole-source validation debt has an open schema")
    body = json.loads(json.dumps(value, allow_nan=False))
    # Reuse the row's identity checks without inventing a numerical comparison.
    for key in ("source_commit", "source_tree", "request_digest", "recipe_execution_digest"):
        expected_length = 40 if key in {"source_commit", "source_tree"} else 64
        raw = body[key]
        if (not isinstance(raw, str) or len(raw) != expected_length
                or any(char not in "0123456789abcdef" for char in raw)):
            raise SurfaceValidationRefused(f"invalid debt {key}")
    if (not isinstance(body["source_keep_ids"], list) or not body["source_keep_ids"]
            or len(body["source_keep_ids"]) > 64
            or any(not isinstance(item, str) or not item for item in body["source_keep_ids"])
            or not isinstance(body["target"], dict) or not body["target"]):
        raise SurfaceValidationRefused("validation debt membership/target is invalid")
    for key in ("original_anchor", "candidate_anchor"):
        anchor = body[key]
        if (not isinstance(anchor, dict) or set(anchor) != {"path", "commit"}
                or not isinstance(anchor["path"], str) or not Path(anchor["path"]).is_absolute()
                or (key == "candidate_anchor" and
                    (not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40))
                or (key == "original_anchor" and anchor["commit"] is not None and
                    (not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40))):
            raise SurfaceValidationRefused(f"debt {key} is invalid")
    if (body["candidate_anchor"]["commit"] != body["source_commit"]
            or body["disposition"] != "pending" or not isinstance(body["reason"], str)
            or not body["reason"] or len(body["reason"]) > 1024
            or not isinstance(body["failure"], dict)):
        raise SurfaceValidationRefused("validation debt disposition is malformed")
    unsigned = {key: item for key, item in body.items() if key != "row_digest"}
    if body["row_digest"] != _digest(unsigned) or len(json.dumps(body).encode()) > MAX_BYTES:
        raise SurfaceValidationRefused("validation debt digest/size differs")
    return body


def retain(directory: Path, value: Any) -> dict[str, Any]:
    """Write one immutable batch-local row and return its exact routing reference."""
    from . import status, surface_fold
    body = validate_debt(value) if isinstance(value, Mapping) \
        and value.get("schema") == DEBT_SCHEMA else validate(value)
    directory = Path(directory)
    path = directory / "whole-source-validation.json"
    expected = json.dumps(body, indent=2, sort_keys=True).encode()
    if path.exists():
        actual = surface_fold.bounded_regular_bytes(path, MAX_BYTES)
        if actual != expected:
            raise SurfaceValidationRefused("validation output path already holds different bytes")
    else:
        status.write_json(directory, path.name, body, prefix=".whole-source-validation-")
        actual = surface_fold.bounded_regular_bytes(path, MAX_BYTES)
    if actual != expected or len(actual) > MAX_BYTES:
        raise SurfaceValidationRefused("validation output readback differs")
    return {"schema": REFERENCE_SCHEMA, "path": str(path.resolve()),
            "sha256": hashlib.sha256(actual).hexdigest()}


def reopen_reference(value: Any) -> dict[str, Any]:
    from . import surface_fold
    if (not isinstance(value, Mapping)
            or set(value) != {"schema", "path", "sha256"}
            or value.get("schema") != REFERENCE_SCHEMA
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64):
        raise SurfaceValidationRefused("validation reference is malformed")
    path = Path(value["path"])
    raw = surface_fold.bounded_regular_bytes(path, MAX_BYTES)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise SurfaceValidationRefused("validation reference bytes changed")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SurfaceValidationRefused("validation reference is not JSON") from exc
    return validate_debt(body) if body.get("schema") == DEBT_SCHEMA else validate(body)


__all__ = ["DEBT_SCHEMA", "MAX_BYTES", "REFERENCE_SCHEMA", "SCHEMA",
           "SurfaceValidationRefused", "classify", "debt", "original_anchor_commit",
           "reopen_reference", "retain", "row", "validate", "validate_debt"]
