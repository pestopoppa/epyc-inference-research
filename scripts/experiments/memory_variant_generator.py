from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any

import httpx

from scripts.experiments.memory_variant_registry import AdaptiveVariant, VariantRegistry


DEFAULT_SEARCH_SPACE: dict[str, Any] = {
    "retrieval_k": [4, 6, 8, 10, 12, 16, 20],
    "edge_k": [4, 6, 8, 10, 12, 16],
    "failure_motif_k": [0, 2, 4, 6, 8],
    "hypothesis_motif_k": [0, 2, 4, 6, 8],
    "memory_char_budget": [800, 1200, 1600, 2000, 2400],
    "graph_variant": [
        "raw_graph_latest",
        "coarse_graph_level_category",
        "coarse_graph_category_only",
        "coarse_graph_category_top_payload",
        "coarse_graph_embedding_t075",
    ],
    "motif_variant": ["level_category", "category_only", "embedding_t075"],
}


@dataclass
class GeneratorConfig:
    top_k: int = 4
    mutation_tries: int = 24
    llm_timeout_s: float = 25.0
    orchestrator_url: str = "http://localhost:8000"
    use_llm: bool = False
    search_space: dict[str, Any] | None = None


class AdaptiveVariantGenerator:
    """Proposes adaptive variants with optional LLM helper and safe fallback."""

    def __init__(
        self,
        registry: VariantRegistry,
        config: GeneratorConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or GeneratorConfig()
        self._rng = random.Random(seed)
        self._space = self.config.search_space or DEFAULT_SEARCH_SPACE

    def select_parents(self, *, include_controls: bool = True) -> list[AdaptiveVariant]:
        top = self.registry.top_variants(
            limit=self.config.top_k,
            include_controls=include_controls,
            min_trials=1,
        )
        if top:
            return top
        return self.registry.ensure_fixed_controls()

    def propose_many(self, count: int, *, use_llm: bool | None = None) -> list[AdaptiveVariant]:
        out: list[AdaptiveVariant] = []
        should_use_llm = self.config.use_llm if use_llm is None else use_llm
        for _ in range(max(0, count)):
            out.append(self.propose_next(use_llm=should_use_llm))
        return out

    def propose_next(self, *, use_llm: bool = False) -> AdaptiveVariant:
        parents = self.select_parents(include_controls=True)
        if use_llm:
            llm_variant = self._propose_with_llm(parents)
            if llm_variant is not None and self._is_novel(llm_variant):
                return self.registry.register(llm_variant)
        return self.registry.register(self._propose_by_mutation(parents))

    def _is_novel(self, variant: AdaptiveVariant) -> bool:
        return self.registry.is_novel(name=variant.name, knobs=variant.knobs)

    def _propose_by_mutation(self, parents: list[AdaptiveVariant]) -> AdaptiveVariant:
        if not parents:
            parents = self.registry.ensure_fixed_controls()
        base = self._rng.choice(parents)

        for _ in range(self.config.mutation_tries):
            knobs = dict(base.knobs)
            self._mutate_knobs(knobs)
            if knobs == base.knobs:
                continue
            candidate = AdaptiveVariant.make(
                name="adaptive_mutation",
                knobs=knobs,
                parent_id=base.variant_id,
                provenance="mutation",
                generation=base.generation + 1,
            )
            if self._is_novel(candidate):
                return candidate

        knobs = dict(base.knobs)
        knobs["mutation_nonce"] = self._rng.randint(1, 10_000_000)
        return AdaptiveVariant.make(
            name="adaptive_mutation",
            knobs=knobs,
            parent_id=base.variant_id,
            provenance="mutation",
            generation=base.generation + 1,
        )

    def _mutate_knobs(self, knobs: dict[str, Any]) -> None:
        keys = list(self._space.keys())
        if not keys:
            knobs["retrieval_k"] = self._rng.choice([6, 8, 10])
            return

        n_changes = self._rng.choice([1, 2, 2, 3])
        for _ in range(n_changes):
            key = self._rng.choice(keys)
            choices = self._space.get(key)
            if isinstance(choices, list) and choices:
                knobs[key] = self._rng.choice(choices)

        if int(knobs.get("retrieval_k", 0) or 0) <= 0:
            knobs["memory_mode"] = "baseline"
            knobs["edge_k"] = 0
            knobs["failure_motif_k"] = 0
            knobs["hypothesis_motif_k"] = 0
            knobs["memory_char_budget"] = 0
        else:
            knobs["memory_mode"] = "subgraph"

    def _propose_with_llm(self, parents: list[AdaptiveVariant]) -> AdaptiveVariant | None:
        lines = [
            "Propose exactly one NEW memory viability variant.",
            "Return strict JSON only with keys: name, knobs, parent_id.",
            "Knobs should optimize quality uplift over baseline for small models.",
            "Parents:",
        ]
        for parent in parents[: self.config.top_k]:
            lines.append(
                json.dumps(
                    {
                        "variant_id": parent.variant_id,
                        "name": parent.name,
                        "knobs": parent.knobs,
                        "score": parent.score,
                        "trials": parent.trials,
                    },
                    sort_keys=True,
                )
            )

        payload = {
            "prompt": "\n".join(lines),
            "force_role": "frontdoor",
            "force_mode": "direct",
            "real_mode": True,
        }

        try:
            response = httpx.post(
                f"{self.config.orchestrator_url.rstrip('/')}/chat",
                json=payload,
                timeout=self.config.llm_timeout_s,
            )
            response.raise_for_status()
            body = response.json()
            answer = body.get("answer", "") if isinstance(body, dict) else ""
            proposed = json.loads(str(answer).strip())
            if not isinstance(proposed, dict):
                return None
            knobs = proposed.get("knobs")
            if not isinstance(knobs, dict) or not knobs:
                return None

            parent_id = proposed.get("parent_id")
            if parent_id is not None:
                parent_id = str(parent_id)
                if self.registry.get(parent_id) is None:
                    parent_id = None

            generation = 0
            if parent_id is not None:
                parent = self.registry.get(parent_id)
                generation = 0 if parent is None else parent.generation + 1

            return AdaptiveVariant.make(
                name=str(proposed.get("name") or "adaptive_llm"),
                knobs=knobs,
                parent_id=parent_id,
                provenance="llm",
                generation=generation,
            )
        except Exception:
            return None
