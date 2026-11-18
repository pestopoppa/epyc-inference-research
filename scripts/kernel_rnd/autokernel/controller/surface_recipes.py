#!/usr/bin/env python3
"""U3-DEFAULTS-b: a PER-SURFACE serving recipe on the champion record, and its gate.

R23-58 made the champion `ef81196d5` + a CPU recipe (THP shim ON) + a GPU recipe (shim
NOT set). One recipe per champion cannot say that. `SurfaceRecipes` keys the recipe
hash by measurement surface, is canonical-JSON serialisable so it can ride on the
champion record, and `require_match` is the refusal the measurement gate calls: a
measurement whose (surface, recipe_hash) pair is not the champion's is refused, and an
unknown surface or a missing pin is refused rather than defaulted (a defaulted recipe is
how an unset variable reached production screening).

Pure data and refusals: no launch, no hashing of files, no store access.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA = "epyc.autokernel.surface_recipes.v1"
SURFACES = ("cpu", "gpu")
_SHA = re.compile(r"[0-9a-f]{64}")


class SurfaceRecipeRefused(ValueError):
    """The (surface, recipe_hash) pair does not match the champion's declared pair."""


@dataclass(frozen=True)
class SurfaceRecipes:
    pins: tuple[tuple[str, str], ...]          # sorted (surface, recipe_hash)

    def __post_init__(self) -> None:
        if not self.pins:
            raise SurfaceRecipeRefused("a champion must pin at least one surface recipe")
        surfaces = [s for s, _ in self.pins]
        if len(set(surfaces)) != len(surfaces) or surfaces != sorted(surfaces):
            raise SurfaceRecipeRefused("pins must be unique and sorted by surface")
        for surface, digest in self.pins:
            if surface not in SURFACES:
                raise SurfaceRecipeRefused(f"unknown surface {surface!r}")
            if not isinstance(digest, str) or not _SHA.fullmatch(digest):
                raise SurfaceRecipeRefused(f"{surface}: recipe_hash must be lowercase sha256")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> "SurfaceRecipes":
        if not isinstance(mapping, Mapping):
            raise SurfaceRecipeRefused("surface recipes must be a mapping")
        return cls(tuple(sorted(mapping.items())))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SurfaceRecipes":
        block = record.get("surface_recipes") if isinstance(record, Mapping) else None
        if not isinstance(block, Mapping) or block.get("schema") != SCHEMA:
            raise SurfaceRecipeRefused(f"champion record carries no {SCHEMA} block")
        if set(block) != {"schema", "pins"}:
            raise SurfaceRecipeRefused("surface_recipes block has unknown or missing fields")
        return cls.from_mapping(block["pins"])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "pins": dict(self.pins)}

    def digest(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()

    def require_match(self, surface: str, recipe_hash: str) -> None:
        pinned = dict(self.pins)
        if surface not in pinned:
            raise SurfaceRecipeRefused(f"champion pins no recipe for surface {surface!r}")
        if pinned[surface] != recipe_hash:
            raise SurfaceRecipeRefused(
                f"{surface}: measured recipe_hash {str(recipe_hash)[:12]} != champion "
                f"pin {pinned[surface][:12]}")
