#!/usr/bin/env python3
"""Shared scaffolding for the loop's HAND-RUN instruments.

WHY THIS EXISTS
---------------
The loop is not the only thing that measures. A fold window -- the loop dead, the GPU
claim free, an operator driving by hand -- runs the same gates the loop runs, on the same
store, and its verdicts are just as consequential: the 2026-09-08 window seeded the
accumulator bundle, recalibrated the serving floor under the pinned recipe (R23-49), ran
the FOLD-2 correctness/perf battery that cleared the current champion, and spent the
serving gate. Every one of those instruments lived in `/mnt/raid0/llm/tmp/`, which is
scratch: one `rm` from making the campaign's most consequential numbers unreproducible.

Worse, being outside the package they were outside its SAFETY LAYER. The hand-run serving
gate built `serving-floor.<name>.json` itself and read `["floor_pct"]` off it -- exactly
the bare name-keyed read that `run.py` was fixed to stop doing, so a hand-run gate could
still be judged against a floor calibrated under a different condition. The hand-run
recalibration wrote that file directly, missing both the atomic write and the identity
refusal. Promoting them into the package is what makes `serving.load_floor` /
`serving.write_floor` the ONLY paths to a floor, from every caller there is.

THE POSTURE. These instruments spend GPU time and write decision state, and they are run
by hand under time pressure. They are therefore DRY BY DEFAULT and escalate monotonically:

  * (no flag)   -- resolve every input, print exactly what would be measured, spend nothing.
  * ``--execute`` -- take the measurement. Writes evidence records; never advances state
    a later decision reads as settled (the champion of record, the floor file).
  * ``--apply``   -- implies ``--execute``, and additionally writes that decision state.

A flag that only wrote state without measuring would be meaningless here, so ``--apply``
implying ``--execute`` matches what the scratch originals did (they always measured) while
the new dry default is the safety they lacked.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from . import serving

#: `.../epyc-inference-research` -- this file is `scripts/kernel_rnd/autokernel/loop/`.
REPO_ROOT = Path(__file__).resolve().parents[4]

#: The loop's durable memory. Same value `run.py --store` is always given; a default so a
#: hand run cannot file its evidence into a store the loop does not read.
DEFAULT_STORE = Path("/mnt/raid0/llm/autokernel/loop-memory")

#: The champion's canonical serving recipe, IN THE REPO. The scratch originals pointed at
#: this same file through an absolute worktree path, which meant an instrument copied to a
#: second worktree silently measured the first worktree's recipe.
DEFAULT_RECIPE = REPO_ROOT / "artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json"

#: The 27B the serving recipe and the FOLD-2 battery are both about.
DEFAULT_MODEL = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")

#: Exit code for "the instrument refused: nothing was measured". Distinct from 1, which
#: means a gate ran and FAILED -- a caller must be able to tell "no reading" from "a bad
#: reading", because only one of them is evidence.
REFUSED = 2


class InstrumentRefusal(RuntimeError):
    """A precondition failed, so nothing was measured. Never a verdict."""


@dataclass(frozen=True)
class Posture:
    """How far this invocation is allowed to go. See the module docstring."""
    execute: bool
    apply: bool

    @property
    def dry_run(self) -> bool:
        return not self.execute

    def describe(self) -> str:
        if self.apply:
            return "APPLY (measure, and write decision state)"
        if self.execute:
            return "EXECUTE (measure; decision state NOT written)"
        return "DRY (default): nothing measured, nothing written"


def add_posture_args(parser: argparse.ArgumentParser, *,
                     apply_help: str | None = None) -> None:
    """The escalation flags, worded identically everywhere so the posture of one
    instrument can never be read as the posture of another.

    `apply_help` is None for an instrument that writes no decision state (a gate battery
    writes an evidence record and nothing else): it then gets `--execute` only, rather
    than an `--apply` that would imply there is state it could settle.
    """
    parser.add_argument("--execute", action="store_true",
                        help="actually take the measurement (spends GPU time). Without "
                             "this the instrument resolves its inputs and prints the plan.")
    if apply_help is not None:
        parser.add_argument("--apply", action="store_true",
                            help=apply_help + " Implies --execute.")


def resolve_posture(args: argparse.Namespace) -> Posture:
    applied = bool(getattr(args, "apply", False))
    return Posture(execute=bool(args.execute or applied), apply=applied)


def add_recipe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE,
                        help="loop durable memory (default: %(default)s)")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE,
                        help="canonical serving recipe (default: the repo's shipped recipe)")


def require_binary(build: Path, name: str) -> Path:
    """Refuse a build directory that cannot produce the measurement.

    `PROMOTION_TARGETS` builds carry `llama-server`; a screening build may not. Asserting
    this up front is the difference between a refusal and a half-hour of wasted GPU that
    dies at the first launch.
    """
    binary = Path(build) / "bin" / name
    if not binary.is_file():
        raise InstrumentRefusal(f"no {name} in {build} (a build carrying it is required)")
    return binary


def read_floor(store: Path | str, recipe: serving.Recipe, *,
               echo=print) -> serving.FloorReading:
    """The ONE way an instrument obtains a serving floor.

    Identical in grammar to `run.py`'s startup block, and for the identical reason: the
    floor is a property of the measured CONDITION, not of the recipe's NAME, so it is read
    through `serving.load_floor` -- which REFUSES a file carrying another recipe's hash
    rather than degrading to "no floor" -- and an unstamped, grandfathered floor is used
    but announced, so no record it touches can pass as a checked one.

    An ABSENT floor is refused here rather than passed on as `None`. `serving.compare`
    would return `decisive: None`, `accumulate.classify_serving` would read that as
    DIVERGED, and the run would spend the full gate to produce a verdict that was decided
    before it started.
    """
    reading = serving.load_floor(store, recipe)
    if reading.provenance == "absent":
        raise InstrumentRefusal(
            f"no serving floor at {reading.path}: uncalibrated. Every comparison would "
            f"come back decisive=None, i.e. a guaranteed non-promotion bought with a full "
            f"gate's GPU time. Calibrate first (autokernel.loop.recal_serving_floor).")
    if reading.provenance == "unverified":
        echo(f"floor     WARNING {reading.path.name} carries no recipe_hash: it predates "
             f"identity-stamped floors, so NOTHING proves it was calibrated under this "
             f"recipe. It is used, and every record it touches is stamped "
             f"floor_provenance=unverified. Recalibrate it.")
    return reading


__all__ = ["DEFAULT_MODEL", "DEFAULT_RECIPE", "DEFAULT_STORE", "REFUSED", "REPO_ROOT",
           "InstrumentRefusal", "Posture", "add_posture_args", "add_recipe_args",
           "read_floor", "require_binary", "resolve_posture"]
