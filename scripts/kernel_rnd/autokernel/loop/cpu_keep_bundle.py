"""CPU keeps present in a fold, recorded as ``accumulator-bundle.cpu.<recipe>.json`` (schema v1).

The GPU surface persists one ``accumulator-bundle-<surface>.json`` (``accumulate.Bundle``).
CPU keeps arrive from INF-70-lineage folds measured under several recipes, with magnitudes
that must never be laundered into settled numbers, so each recipe gets its own bundle and
every keep carries the facts that qualify its magnitude:

* ``contention_label`` -- every INF-70 arm before 2026-09-07 carries a WRONG label (the
  sampler read 184-191 as disjoint); those are ``pre-hook`` and their magnitude is
  ``provisional``.
* ``contention_label`` ``unproven`` -- the serving residency record could not prove the host
  quiet (every DS41 serving A/B); like ``pre-hook`` it can never back a settled magnitude.
* ``statistic`` -- which test produced the magnitude. A sign test over pairings
  double-counts the shared treatment arm and is inflated; only an arm-level permutation
  test may back a ``settled`` magnitude.
* ``floor_pct`` -- an effect at or below its instrument's floor has a solid sign, not a
  solid magnitude, so it is ``provisional``.

Pure data validation plus one atomic writer; no measurement, no store scan.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import status

SCHEMA = "epyc.autokernel.cpu_accumulator_bundle.v1"
SURFACE = "cpu"
FILENAME_PREFIX = "accumulator-bundle.cpu."
MAGNITUDES = ("provisional", "settled")
CONTENTION_LABELS = ("pre-hook", "unproven", "quiet", "co-tenant")
SETTLED_STATISTICS = frozenset({"arm_level_permutation"})
_SHA = re.compile(r"^[0-9a-f]{40,64}$")
_KEY_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_FIELDS = {"schema", "surface", "recipe", "recipe_hash", "champion_of_record", "tip",
           "keeps", "compounded_bench_pct"}
_KEEP_FIELDS = {"keep_id", "effect_pct", "magnitude", "contention_label", "statistic",
                "floor_pct", "floor_unit"}


class CpuBundleError(ValueError):
    """A CPU bundle record is malformed or claims more than its evidence supports."""


def recipe_key(recipe: str) -> str:
    """Filesystem-safe key; unsafe names get a digest so two recipes never share a file."""
    import hashlib
    if not isinstance(recipe, str) or not recipe.strip():
        raise CpuBundleError("recipe name is empty")
    if len(recipe) <= 96 and not recipe.startswith(".") and not _KEY_SAFE.search(recipe):
        return recipe
    digest = hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:8]
    return f"{(_KEY_SAFE.sub('_', recipe).lstrip('.') or 'recipe')[:80]}-{digest}"


def filename(recipe: str) -> str:
    return f"{FILENAME_PREFIX}{recipe_key(recipe)}.json"


def _num(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CpuBundleError(f"{label} must be a finite number")
    return float(value)


def validate_keep(keep: Any) -> dict:
    if not isinstance(keep, Mapping) or not {"keep_id", "effect_pct", "magnitude",
                                            "contention_label", "statistic"} <= set(keep) \
            or set(keep) - _KEEP_FIELDS:
        raise CpuBundleError("keep entry has missing or unknown fields")
    kid = keep["keep_id"]
    if not isinstance(kid, str) or not kid.strip():
        raise CpuBundleError("keep_id must be non-empty text")
    effect = _num(keep["effect_pct"], f"{kid}.effect_pct")
    if keep["magnitude"] not in MAGNITUDES:
        raise CpuBundleError(f"{kid}: magnitude must be one of {MAGNITUDES}")
    if keep["contention_label"] not in CONTENTION_LABELS:
        raise CpuBundleError(f"{kid}: contention_label must be one of {CONTENTION_LABELS}")
    stat = keep["statistic"]
    if not isinstance(stat, str) or not stat.strip():
        raise CpuBundleError(f"{kid}: the statistic that produced the magnitude is required")
    floor = keep.get("floor_pct")
    if floor is not None:
        floor = _num(floor, f"{kid}.floor_pct")
        if keep.get("floor_unit") not in ("arm", "session", "process"):
            raise CpuBundleError(f"{kid}: a floor must state its unit")
    if keep["magnitude"] == "settled":
        if keep["contention_label"] in ("pre-hook", "unproven"):
            raise CpuBundleError(f"{kid}: {keep['contention_label']} contention cannot back a settled magnitude")
        if stat not in SETTLED_STATISTICS:
            raise CpuBundleError(f"{kid}: statistic {stat!r} cannot back a settled magnitude "
                                 f"(only {sorted(SETTLED_STATISTICS)})")
        if floor is None or abs(effect) <= floor:
            raise CpuBundleError(f"{kid}: effect at or below its instrument floor (or no floor) "
                                 "cannot be settled")
    out = {"keep_id": kid, "effect_pct": effect, "magnitude": keep["magnitude"],
           "contention_label": keep["contention_label"], "statistic": stat}
    if floor is not None:
        out.update(floor_pct=floor, floor_unit=keep["floor_unit"])
    return out


def build(*, recipe: str, recipe_hash: str, champion_of_record: str, tip: str,
          keeps: Sequence[Mapping]) -> dict:
    """Assemble and validate a bundle. ``compounded_bench_pct`` is derived, never supplied."""
    recipe_key(recipe)
    for label, value in (("recipe_hash", recipe_hash), ("champion_of_record", champion_of_record),
                         ("tip", tip)):
        if not isinstance(value, str) or not value.strip():
            raise CpuBundleError(f"{label} must be non-empty text")
    rows = [validate_keep(k) for k in keeps]
    if len({r["keep_id"] for r in rows}) != len(rows):
        raise CpuBundleError("duplicate keep_id")
    gain = 1.0
    for r in rows:
        if r["effect_pct"] <= -100.0:
            raise CpuBundleError(f"{r['keep_id']}: effect_pct <= -100")
        gain *= 1.0 + r["effect_pct"] / 100.0
    return {"schema": SCHEMA, "surface": SURFACE, "recipe": recipe, "recipe_hash": recipe_hash,
            "champion_of_record": champion_of_record, "tip": tip, "keeps": rows,
            "compounded_bench_pct": (gain - 1.0) * 100.0}


def from_dict(raw: Any) -> dict:
    """Strict reader: unknown schema/fields refuse; re-derives and checks the compounded figure."""
    if not isinstance(raw, Mapping) or raw.get("schema") != SCHEMA:
        raise CpuBundleError(f"unknown cpu bundle schema {getattr(raw, 'get', lambda k: None)('schema')!r}")
    if set(raw) != _FIELDS or raw.get("surface") != SURFACE:
        raise CpuBundleError("cpu bundle has missing or unknown fields")
    rebuilt = build(recipe=raw["recipe"], recipe_hash=raw["recipe_hash"],
                    champion_of_record=raw["champion_of_record"], tip=raw["tip"],
                    keeps=raw["keeps"])
    if abs(rebuilt["compounded_bench_pct"] - _num(raw["compounded_bench_pct"], "compounded_bench_pct")) > 1e-9:
        raise CpuBundleError("compounded_bench_pct disagrees with the keeps")
    return rebuilt


def cpu_keeps_in_fold(fold_keeps: Sequence[Mapping]) -> list[Mapping]:
    """The CPU entries of a fold; GPU entries are not this bundle's business.

    Each entry needs an explicit ``surface``; one without it refuses rather than guessing.
    """
    out = []
    for keep in fold_keeps:
        surface = keep.get("surface")
        if surface not in ("cpu", "gpu"):
            raise CpuBundleError(f"fold keep {keep.get('keep_id')!r} does not state its surface")
        if surface == "cpu":
            out.append({k: v for k, v in keep.items() if k != "surface"})
    return out


def write(store: Path | str, bundle: Mapping) -> Path:
    """Validate, then publish atomically as ``accumulator-bundle.cpu.<recipe>.json``."""
    checked = from_dict(bundle)
    return status.write_json(Path(store), filename(checked["recipe"]), checked,
                             prefix=".accumulator-bundle-cpu-")


def read(path: Path | str) -> dict:
    import json
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
