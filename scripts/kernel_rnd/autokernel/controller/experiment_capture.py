"""Self-hashed `belief_capture` for one experiments row (belief redesign Phase 2, DESIGN §5.1/§5.2).

WHY. ROOT's `scripts/vidya/adapters/autokernel_experiments.py` projects every `experiments.db`
row into a loop belief. A row written before this hook carries no producer capture, so ROOT marks
it RETROFIT (`observation.retrofit` / `provenance.retrofit`): the fields were read back from a
record that never said, at write time, which of its facts it stands behind. A row written WITH
this capture is PROSPECTIVE: the writer names the projected facts and seals them with a hash at
the moment the row is inserted.

WHAT. `build(columns, payload)` returns

    {"schema": SCHEMA, "subject": {...}, "content_sha256": sha256(canonical(subject))}

`subject` holds the facts ROOT projects, each copied from the row it rides on by a fixed rule
(`subject`, below), plus `payload_sha256`, a digest of the whole payload WITHOUT the capture, so
every payload fact a projection may read (gate lines, integrity class, serving join id) is bound
too. ROOT re-derives the same subject from the stored row, refuses the row if the capture's hash
or any subject field disagrees, and only then projects it as prospective (retrofit false).

THE CONTRACT IS DUPLICATED ON PURPOSE. ROOT never imports research code, so ROOT's verifier
(`autokernel_experiments._capture_subject`) re-implements `subject` and `canonical`. Change one,
change both, and bump SCHEMA: a v1 reader refuses any other schema rather than guessing.

Write side only. The capture never changes the DB schema: it is one more top-level payload key.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

SCHEMA = "epyc.autokernel.experiment_belief_capture.v1"

#: The `experiments` columns other than `payload`, in table order (controller/experiments.py `_DDL`).
COLUMNS = ("attempt_id", "recorded_at", "campaign_id", "deployment", "epoch_sha256",
           "hypothesis_id", "mechanism_id", "target_surface", "target_symbol", "statement",
           "falsifier", "status", "effect_fraction", "exact_effect", "target_effect",
           "refusal_reason", "result_sha256", "spawn_parent", "branch_id", "width", "depth")

#: Comparison keys ROOT reads for a measured value, its anchor, effect, floor, n and the serving join.
MEASUREMENT_KEYS = ("metric", "surface", "anchor_tok_s", "anchor_samples", "candidate_tok_s",
                    "candidate_samples", "effect", "effect_pct", "noise_floor_pct", "floor_pct",
                    "pairs", "decisive", "np")


def finite(value: Any) -> Any:
    """`value` with every non-finite float replaced by None (SQLite stores NaN as NULL)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(finite(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Digest of the payload without its own `belief_capture`."""
    return hashlib.sha256(canonical({key: value for key, value in payload.items()
                                     if key != "belief_capture"})).hexdigest()


def _comparison(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The block ROOT reads numbers from: comparison, else anchor_guard(.comparison), else serving."""
    comparison = payload.get("comparison")
    if isinstance(comparison, Mapping):
        return comparison
    guard = payload.get("anchor_guard")
    if isinstance(guard, Mapping):
        inner = guard.get("comparison")
        return inner if isinstance(inner, Mapping) else guard
    serving = payload.get("serving")
    return serving if isinstance(serving, Mapping) else {}


def subject(columns: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """The projected facts of one row, copied by a fixed rule (ROOT re-derives the same dict)."""
    scope = payload.get("research_scope")
    research_scope = None
    if isinstance(scope, Mapping):
        model = scope.get("model")
        research_scope = {
            "model_path": model.get("path") if isinstance(model, Mapping) else model,
            "model_sha256": model.get("sha256") if isinstance(model, Mapping) else None,
            "quant": scope.get("quant"), "backend": scope.get("backend"),
            "measurement_surface": scope.get("measurement_surface")}
    comparison = _comparison(payload)
    capture = comparison.get("belief_capture")
    measurement = {key: comparison[key] for key in MEASUREMENT_KEYS if key in comparison}
    plan = comparison.get("measurement_plan")
    if isinstance(plan, Mapping) and "instrument" in plan:
        measurement["instrument"] = plan["instrument"]
    if isinstance(capture, Mapping) and "capture_id" in capture:
        measurement["serving_capture_id"] = capture["capture_id"]
    return finite({
        "row": {name: columns.get(name) for name in COLUMNS},
        "research_scope": research_scope,
        "measurement": measurement,
        "build": {"spawn_parent": columns.get("spawn_parent"),
                  "champion_head": payload.get("champion_head")},
        "payload_sha256": payload_sha256(payload)})


def build(columns: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """The capture to store as `payload["belief_capture"]`. Raises on an unencodable payload."""
    body = subject(columns, payload)
    return {"schema": SCHEMA, "subject": body,
            "content_sha256": hashlib.sha256(canonical(body)).hexdigest()}


def verify(capture: Any, columns: Mapping[str, Any], payload: Mapping[str, Any]) -> str | None:
    """None when `capture` seals exactly this row; else the refusal reason (ROOT's rule, for tests)."""
    if not isinstance(capture, Mapping) or capture.get("schema") != SCHEMA:
        return "schema"
    body = capture.get("subject")
    if not isinstance(body, Mapping):
        return "malformed"
    if hashlib.sha256(canonical(body)).hexdigest() != capture.get("content_sha256"):
        return "hash_mismatch"
    if canonical(body) != canonical(subject(columns, payload)):
        return "disagrees_with_row"
    return None


__all__ = ["COLUMNS", "MEASUREMENT_KEYS", "SCHEMA", "build", "canonical", "finite",
           "payload_sha256", "subject", "verify"]
