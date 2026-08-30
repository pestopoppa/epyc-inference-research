#!/usr/bin/env python3
"""The GPU build recipe, versioned, with a stated reason for every divergence.

WHY THIS EXISTS
---------------
The screening surface was never specified. It accumulated from CMake defaults:
`discovery_deployment_factory` passed three defines and `GGML_HIP_ROCWMMA_FATTN`
was never named, so it fell to the CMake default OFF -- a path measured on
2026-08-27 to produce NON-FINITE VALUES at longer sequence lengths on gfx90a under
`-fa on` (all 12 pinned long prompts failed; a 25-character prompt passed on the
same binary, so prompt length is the discriminator and a short smoke test hides it).

Nobody decided that. It was an unset variable.

A screening surface that deliberately diverges from production is legitimate --
cheaper, more portable, more sensitive -- provided someone decided it and the
divergence is on the record. So this module makes the recipe a first-class,
versioned object in which **a flag that diverges from production without a stated
reason is refused at construction**. Divergence becomes a decision you can read,
never an omission you have to discover by measuring non-finite outputs.

WHAT THIS CANNOT DO, AND WHY IT MATTERS
---------------------------------------
Production's own GPU recipe is NOT recoverable from disk: `/mnt/raid0/llm/llama.cpp/
build-hip/` contains only `bin/`, with no `CMakeCache.txt`. The production column
below is therefore a DECLARED reference -- sourced from the CH-8 ruling recorded in
the research repo README -- not a value read back from the build that production
serves. `PRODUCTION_RECIPE_IS_VERIFIABLE` is False to say so out loud, because a
reference nobody can re-derive is exactly the kind of thing that silently rots.
Closing that gap means recording production's configure line at freeze time; it is
filed against the promotion runbook, not fixed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

RECIPE_SCHEMA = "epyc.autokernel.gpu_build_recipe.v1"

#: The production reference is declared, not read back. See the module docstring.
PRODUCTION_RECIPE_IS_VERIFIABLE = False


class BuildRecipeError(ValueError):
    """A flag diverges from production with no stated reason."""


@dataclass(frozen=True)
class NonAdoption:
    """A config win that was measured, settled, and deliberately NOT adopted."""

    setting: str
    rejected_value: str
    #: False for a runtime setting, which cannot be a CMake define at all.
    is_cmake_flag: bool
    finding: str


#: CH-6, settled 2026-08-28: NEITHER standing config win enters the recipe.
#:
#: They are recorded HERE, beside the mechanism that would carry them, because the
#: withdrawn numbers still circulate and a reader who meets only those will
#: re-adopt them. The mechanism is a capability; adopting these two is not part of
#: it.
SETTLED_NON_ADOPTIONS = (
    NonAdoption(
        "GGML_HIP_MMQ_MFMA", "OFF", True,
        "+23.09% on Qwen2.5-Coder-0.5B and +0.50% prefill / -0.28% decode on "
        "Qwen3.8-27B, the production model. The 0.5B has n_embd=896, which is not "
        "divisible by 256, so its K-quants fall back to Q5_0 and it never "
        "dispatches the kernels production uses -- the win was measured on the "
        "wrong model. Real where it was taken, worth nothing where the fleet runs."),
    NonAdoption(
        "n_ubatch", "1024", False,
        "A NULL ARM, and not a build flag at all. llama.cpp clamps "
        "n_ubatch = min(n_batch, n_ubatch), so the `-b 512 -ub 1024` screen ran ONE "
        "byte-identical binary in both arms; the +46.9% is a bimodal sample whose "
        "median landed on the fast mode. It could not have measured anything."),
)

#: (flag, value) -> why it was not adopted, for the defines a recipe can express.
_SETTLED_FLAG_VALUES = {
    (item.setting, item.rejected_value): item.finding
    for item in SETTLED_NON_ADOPTIONS if item.is_cmake_flag
}


@dataclass(frozen=True)
class Flag:
    """One CMake define, with where its value came from and why."""

    name: str
    value: str
    #: What production uses. None means production leaves it at the CMake default.
    production_value: str | None
    #: Required whenever `value != production_value`. Enforced, not advisory.
    reason: str = ""

    @property
    def diverges(self) -> bool:
        return self.value != self.production_value

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.value, str):
            raise BuildRecipeError("a build flag needs a name and a string value")
        settled = _SETTLED_FLAG_VALUES.get((self.name, self.value))
        if settled is not None and not self.reason.strip():
            # Checked BEFORE the divergence rule, and not folded into it, for two
            # reasons. It catches the one spelling divergence cannot -- declaring
            # `production_value` to BE the rejected value makes the flag
            # non-diverging, so no reason is ever demanded -- and where both apply
            # it is the message that carries the correcting number, which is the
            # thing a reader about to re-adopt from the withdrawn one needs.
            raise BuildRecipeError(
                f"{self.name}={self.value} was measured and NOT adopted (CH-6): "
                f"{settled} Re-adopting it requires a stated reason, not the "
                f"withdrawn number.")
        if self.diverges and not self.reason.strip():
            raise BuildRecipeError(
                f"{self.name}={self.value} diverges from production "
                f"({self.production_value}) with no stated reason. A screening "
                f"surface may diverge from production, but never by omission -- "
                f"state why, or match production.")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value,
                "production_value": self.production_value,
                "diverges": self.diverges, "reason": self.reason or None}


@dataclass(frozen=True)
class BuildRecipe:
    """A named, versioned set of flags with a content-addressed identity."""

    name: str
    flags: tuple[Flag, ...]
    notes: str = ""

    def cmake_defines(self) -> tuple[tuple[str, str], ...]:
        """The defines, in the shape `StaticGpuSourceBuilder` takes."""
        return tuple((flag.name, flag.value) for flag in self.flags)

    def divergences(self) -> tuple[Flag, ...]:
        return tuple(flag for flag in self.flags if flag.diverges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RECIPE_SCHEMA,
            "name": self.name,
            "production_reference_is_verifiable": PRODUCTION_RECIPE_IS_VERIFIABLE,
            "flags": [flag.to_dict() for flag in self.flags],
            "divergences": [flag.name for flag in self.divergences()],
            "notes": self.notes or None,
        }

    def sha256(self) -> str:
        """Recipe identity, emitted with every result.

        A number is only interpretable against the recipe that produced it. Carrying
        this digest on the record is what lets a later reader tell whether two
        measurements are even comparable -- and it is what feeds the epoch hash in
        `controller/experiments.py`.
        """
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).hexdigest()


#: The house GPU recipe. Every value here is a decision with a reason on the record.
HOUSE_GPU_RECIPE = BuildRecipe(
    name="gfx90a-house-v1",
    notes=("Matches production on every flag. The screening surface is deliberately "
           "identical to the serving surface, so a measured win transfers without "
           "an argument about the build."),
    flags=(
        Flag("GGML_HIP", "ON", "ON"),
        Flag("AMDGPU_TARGETS", "gfx90a", "gfx90a"),
        Flag(
            "GGML_HIP_ROCWMMA_FATTN", "ON", "ON",
            reason="",  # matches production; no divergence to justify
        ),
        Flag("GGML_NATIVE", "ON", "ON"),
    ),
)


def recipe_for(name: str) -> BuildRecipe:
    if name != HOUSE_GPU_RECIPE.name:
        raise BuildRecipeError(f"unknown build recipe {name!r}")
    return HOUSE_GPU_RECIPE


def from_flags(name: str, flags: Sequence[Mapping[str, Any]], *,
               notes: str = "") -> BuildRecipe:
    """Build a recipe from declarations, refusing any unjustified divergence."""
    return BuildRecipe(
        name=name,
        notes=notes,
        flags=tuple(Flag(name=str(item["name"]), value=str(item["value"]),
                         production_value=item.get("production_value"),
                         reason=str(item.get("reason") or ""))
                    for item in flags))


__all__ = ["BuildRecipe", "BuildRecipeError", "Flag", "HOUSE_GPU_RECIPE",
           "NonAdoption", "PRODUCTION_RECIPE_IS_VERIFIABLE", "RECIPE_SCHEMA",
           "SETTLED_NON_ADOPTIONS", "from_flags", "recipe_for"]
