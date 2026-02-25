from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
import uuid


FIXED_CONTROL_NAMES = ("baseline", "raw_subgraph", "category_coarse")


@dataclass
class AdaptiveVariant:
    """Single memory variant definition and observed metrics."""

    variant_id: str
    name: str
    knobs: dict[str, Any]
    parent_id: str | None = None
    provenance: str = "mutation"
    generation: int = 0
    score: float | None = None
    trials: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def signature(self) -> str:
        return json.dumps(
            {"name": self.name, "knobs": self.knobs},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def make(
        cls,
        *,
        name: str,
        knobs: dict[str, Any],
        parent_id: str | None = None,
        provenance: str = "mutation",
        generation: int = 0,
        score: float | None = None,
        trials: int = 0,
    ) -> "AdaptiveVariant":
        return cls(
            variant_id=str(uuid.uuid4()),
            name=name,
            knobs=knobs,
            parent_id=parent_id,
            provenance=provenance,
            generation=generation,
            score=score,
            trials=trials,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdaptiveVariant":
        score_val = payload.get("score")
        return cls(
            variant_id=str(payload.get("variant_id") or str(uuid.uuid4())),
            name=str(payload["name"]),
            knobs=dict(payload.get("knobs") or {}),
            parent_id=payload.get("parent_id"),
            provenance=str(payload.get("provenance") or "mutation"),
            generation=int(payload.get("generation") or 0),
            score=float(score_val) if score_val is not None else None,
            trials=int(payload.get("trials") or 0),
            created_at=str(payload.get("created_at") or datetime.now(UTC).isoformat()),
            updated_at=str(payload.get("updated_at") or datetime.now(UTC).isoformat()),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VariantRegistry:
    """Append-only JSONL registry for controls and adaptive variants."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._variants: dict[str, AdaptiveVariant] = {}
        self._signature_to_id: dict[str, str] = {}
        self._load()
        self.ensure_fixed_controls()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                variant = AdaptiveVariant.from_dict(payload)
                self._variants[variant.variant_id] = variant
                self._signature_to_id[variant.signature()] = variant.variant_id

    def _persist_variant(self, variant: AdaptiveVariant) -> None:
        variant.updated_at = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(variant.to_dict(), sort_keys=True) + "\n")

    def ensure_fixed_controls(self) -> list[AdaptiveVariant]:
        controls = [
            (
                "baseline",
                {
                    "memory_mode": "baseline",
                    "graph_variant": "none",
                    "retrieval_k": 0,
                    "edge_k": 0,
                    "motif_k": 0,
                    "memory_char_budget": 0,
                },
            ),
            (
                "raw_subgraph",
                {
                    "memory_mode": "subgraph",
                    "graph_variant": "raw_graph_latest",
                    "retrieval_k": 8,
                    "edge_k": 10,
                    "motif_variant": "level_category",
                    "failure_motif_k": 4,
                    "hypothesis_motif_k": 4,
                    "memory_char_budget": 1800,
                },
            ),
            (
                "category_coarse",
                {
                    "memory_mode": "subgraph",
                    "graph_variant": "coarse_graph_level_category",
                    "retrieval_k": 6,
                    "edge_k": 8,
                    "motif_variant": "level_category",
                    "failure_motif_k": 5,
                    "hypothesis_motif_k": 5,
                    "memory_char_budget": 1400,
                },
            ),
        ]
        ensured: list[AdaptiveVariant] = []
        for name, knobs in controls:
            existing = self.find_by_signature(name=name, knobs=knobs)
            if existing is not None:
                ensured.append(existing)
                continue
            variant = AdaptiveVariant.make(name=name, knobs=knobs, provenance="control")
            self.register(variant)
            ensured.append(variant)
        return ensured

    def all(self) -> list[AdaptiveVariant]:
        return list(self._variants.values())

    def get(self, variant_id: str) -> AdaptiveVariant | None:
        return self._variants.get(variant_id)

    def get_by_name(self, name: str) -> AdaptiveVariant | None:
        for v in self._variants.values():
            if v.name == name:
                return v
        return None

    def find_by_signature(self, *, name: str, knobs: dict[str, Any]) -> AdaptiveVariant | None:
        signature = json.dumps(
            {"name": name, "knobs": knobs},
            sort_keys=True,
            separators=(",", ":"),
        )
        existing_id = self._signature_to_id.get(signature)
        if existing_id is None:
            return None
        return self._variants.get(existing_id)

    def is_novel(self, *, name: str, knobs: dict[str, Any]) -> bool:
        return self.find_by_signature(name=name, knobs=knobs) is None

    def register(self, variant: AdaptiveVariant) -> AdaptiveVariant:
        existing = self.find_by_signature(name=variant.name, knobs=variant.knobs)
        if existing is not None:
            return existing
        self._variants[variant.variant_id] = variant
        self._signature_to_id[variant.signature()] = variant.variant_id
        self._persist_variant(variant)
        return variant

    def record_result(self, variant_id: str, score: float) -> AdaptiveVariant:
        variant = self._variants[variant_id]
        if variant.score is None or variant.trials == 0:
            variant.score = float(score)
            variant.trials = 1
        else:
            total = variant.score * variant.trials + float(score)
            variant.trials += 1
            variant.score = total / variant.trials
        self._persist_variant(variant)
        return variant

    def top_variants(
        self,
        *,
        limit: int = 5,
        include_controls: bool = False,
        min_trials: int = 1,
    ) -> list[AdaptiveVariant]:
        candidates = []
        for variant in self._variants.values():
            if variant.score is None:
                continue
            if variant.trials < min_trials:
                continue
            if not include_controls and variant.name in FIXED_CONTROL_NAMES:
                continue
            candidates.append(variant)
        candidates.sort(key=lambda v: (float(v.score or 0.0), v.trials), reverse=True)
        return candidates[: max(0, limit)]
