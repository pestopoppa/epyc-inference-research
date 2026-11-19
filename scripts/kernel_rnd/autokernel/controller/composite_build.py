#!/usr/bin/env python3
"""Recipe pinning for composite champion GPU store builds (C14/C15/C16 and later).

Composite store build jobs (the per-row ``job.py`` that runs cmake configure/build and writes
``BUILD-IDENTITY.json`` / ``BUILD-OK.json``) used to take their flags from a pinned
``configure_argv-gpu.json`` with no named recipe, so the C14 recipe had to be derived after the
build (AKX-ALL-14).  A job now calls :func:`pin_identity` on the identity record it is about to
write: it refuses when ``configure_argv`` drifts from the named recipe and stamps the recipe name
and digest into the record.  Default recipe: ``GFX90A_ROCM62_C14``.

Only the recipe's own flags are checked.  Flags the recipe does not name (compiler paths,
file-prefix-map, UI toggles, ...) are free to vary, and a composite that adds a new feature flag
must register a new recipe first -- that is the point.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import build_recipe

DEFAULT_COMPOSITE_RECIPE = build_recipe.GFX90A_ROCM62_C14.name


class RecipeDriftError(ValueError):
    """configure_argv does not realise the named recipe."""


def cmake_defs(configure_argv: Sequence[str]) -> dict[str, str]:
    """``-DNAME[:TYPE]=VALUE`` arguments as {NAME: VALUE}; the last occurrence wins, as in cmake."""
    defs: dict[str, str] = {}
    for arg in configure_argv:
        if arg.startswith("-D") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            defs[key.split(":", 1)[0]] = value
    return defs


def drift(configure_argv: Sequence[str], recipe: build_recipe.BuildRecipe) -> list[str]:
    defs = cmake_defs(configure_argv)
    return [f"{f.name}: recipe={f.value} argv={defs.get(f.name)}"
            for f in recipe.flags if defs.get(f.name) != f.value]


def pin_identity(identity: Mapping[str, Any], recipe_name: str = DEFAULT_COMPOSITE_RECIPE) -> dict[str, Any]:
    """Return ``identity`` plus build_recipe / build_recipe_digest, or raise on drift.

    ``identity`` must carry ``configure_argv`` (the argv actually run).  Call this BEFORE writing
    BUILD-IDENTITY.json and BUILD-OK.json so a drifted build never gets an OK marker.
    """
    recipe = build_recipe.recipe_for(recipe_name)
    argv = identity.get("configure_argv")
    if not argv:
        raise RecipeDriftError("identity has no configure_argv; cannot pin a recipe")
    bad = drift(argv, recipe)
    if bad:
        raise RecipeDriftError(f"configure_argv drifts from recipe {recipe.name}: " + "; ".join(bad))
    return dict(identity, build_recipe=recipe.name, build_recipe_digest=recipe.sha256())


def verify_identity(identity: Mapping[str, Any]) -> list[str]:
    """Problems with an already-written identity's recipe stamp (empty list = pinned and clean)."""
    name = identity.get("build_recipe")
    if not name:
        return ["no build_recipe in identity"]
    try:
        recipe = build_recipe.recipe_for(name)
    except build_recipe.BuildRecipeError as exc:
        return [str(exc)]
    problems = []
    if identity.get("build_recipe_digest") != recipe.sha256():
        problems.append("build_recipe_digest does not match the registered recipe")
    problems += drift(identity.get("configure_argv") or [], recipe)
    return problems


__all__ = ["DEFAULT_COMPOSITE_RECIPE", "RecipeDriftError", "cmake_defs", "drift",
           "pin_identity", "verify_identity"]
