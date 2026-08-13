"""Immutable amortized baseline bank for non-promotable discovery screens.

The bank deliberately carries only an exact-frame anchor vector.  It is never
accepted by strict T1 and cannot be converted into a candidate/archive record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .. import schemas

SCHEMA = "epyc.autokernel.screening_baseline_bank.v1"


class BaselineBankError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineBank:
    frame: Mapping[str, Any]
    anchor_samples: tuple[float, ...]
    sentinel_before: float
    sentinel_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": SCHEMA, "frame": dict(self.frame),
                "anchor_samples": list(self.anchor_samples),
                "sentinel_before": self.sentinel_before,
                "sentinel_after": self.sentinel_after}
        return {**body, "baseline_sha256": schemas.content_hash(body)}

    def admit(self, frame: Mapping[str, Any]) -> None:
        if dict(frame) != dict(self.frame):
            raise BaselineBankError("screening baseline frame differs from candidate frame")
        if self.sentinel_after is not None:
            raise BaselineBankError("screening baseline is closed; create a fresh bank")

    def nominate(self, candidate_samples: tuple[float, ...]) -> dict[str, Any]:
        """Noise-tolerant directional summary, never a pass/fail decision."""
        if not candidate_samples:
            raise BaselineBankError("screening candidate has no samples")
        center = sum(self.anchor_samples) / len(self.anchor_samples)
        values = tuple((x - center) / center for x in candidate_samples)
        return {"baseline_center": center, "candidate_samples": list(candidate_samples),
                "relative_effects": list(values),
                "median_relative": sorted(values)[len(values) // 2],
                "uncertainty": "screening_noise_unquantified_nonpromotable",
                "nomination": "top_k_candidate_only_not_a_keep"}


def load(path: str | Path) -> BaselineBank:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BaselineBankError("baseline bank must be an object")
    body = {key: raw.get(key) for key in ("schema", "frame", "anchor_samples",
                                          "sentinel_before", "sentinel_after")}
    if raw.get("baseline_sha256") != schemas.content_hash(body) or body["schema"] != SCHEMA:
        raise BaselineBankError("baseline bank schema/hash is invalid")
    values = body["anchor_samples"]
    if not isinstance(body["frame"], Mapping) or not isinstance(values, list) or len(values) < 2:
        raise BaselineBankError("baseline bank needs exact frame and >=2 anchor samples")
    return BaselineBank(dict(body["frame"]), tuple(float(x) for x in values),
                        float(body["sentinel_before"]),
                        None if body["sentinel_after"] is None else float(body["sentinel_after"]))
