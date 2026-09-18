"""Request-distinct serving confirmation for integrity-flagged source candidates.

This is a per-candidate integrity gate, not a champion-of-record promotion gate.
The caller supplies a separately frozen request set and its independently
calibrated, request-bound floor. No measurement is inferred from the public screen.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from . import serving, status


def validate_requests(recipe, public, heldout):
    """Require valid, genuinely different serving requests at the same recipe."""
    public_digest = serving.request_digest(recipe, public)
    heldout_digest = serving.request_digest(recipe, heldout)
    if (public_digest is None or heldout_digest is None
            or tuple(body for _, body in public) == tuple(body for _, body in heldout)):
        raise ValueError("held-out serving prompts must differ from public screen request bytes")
    for (_, public_body), (_, heldout_body) in zip(public, heldout):
        try:
            original = json.loads(public_body)
            unseen = json.loads(heldout_body)
        except (TypeError, ValueError) as exc:
            raise ValueError("held-out serving request body is not JSON") from exc
        original_prompt = original.pop("prompt", None)
        unseen_prompt = unseen.pop("prompt", None)
        if (not isinstance(original_prompt, list) or not isinstance(unseen_prompt, list)
                or len(original_prompt) != len(unseen_prompt) or original != unseen):
            raise ValueError("held-out serving requests must preserve token count and all generation controls")
    return public_digest, heldout_digest


def decide(*, store: Path, mechanism_id: str, screen, heldout,
           public_digest: str, heldout_digest: str, floor_path: Path):
    """Persist both measurements before returning a conservative keep disposition."""
    row = heldout.to_dict()
    decisive = row.get("decisive")
    effect = row.get("effect")
    valid = (type(decisive) is bool and type(effect) in (int, float)
             and math.isfinite(effect) and type(row.get("noise_floor_pct")) in (int, float)
             and math.isfinite(row["noise_floor_pct"]) and row["noise_floor_pct"] > 0
             and row.get("request_digest") == heldout_digest
             and row.get("effect_unit") == serving.COMPARE_EFFECT_UNIT
             and row.get("floor_unit") == serving.COMPARE_EFFECT_UNIT)
    promoted = valid and not (decisive and effect < 0)
    reason = ("held-out serving comparison invalid or uncalibrated" if not valid else
              "held-out serving decisive regression" if decisive and effect < 0 else
              "held-out serving did not show a decisive regression")
    recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "schema": "epyc.autokernel.heldout_serving_confirm.v1",
        "mechanism_id": mechanism_id, "promoted": promoted, "reason": reason,
        "public_request_digest": public_digest,
        "heldout_request_digest": heldout_digest,
        "floor_path": str(floor_path), "screen": screen.to_dict(),
        "confirm": row, "recorded_at": recorded_at,
    }
    stamp = recorded_at.replace(":", "").replace("-", "")
    status.write_json(Path(store) / "confirm", f"{mechanism_id}.{stamp}.json", record,
                      prefix=".heldout-confirm-")
    return record
