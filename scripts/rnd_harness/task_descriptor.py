#!/usr/bin/env python3
"""task_descriptor.py — the domain-agnostic per-task descriptor for the EPYC
agent-collaboration R&D harness (handoff: agent-collab-rnd-harness).

WHAT THIS IS
------------
A single, uniform contract every R&D task must satisfy so the *same* search /
experience-bank / scoring machinery can drive **any** domain — a CPU-kernel
sweep, a routing-policy tune, a prompt-forge experiment, a NUMA-placement
study — without the loop knowing anything domain-specific. This is the "HyRA
cross-domain contract" referenced by the handoff:

    {task_id,
     seed_or_baseline_solution   (an ALWAYS-VALID fallback),
     run_script  -> solution.json (the run contract),
     objective_scorer            (ref/callable: solution.json -> float),
     is_correct                  (hard gate; correctness is lexicographic-first),
     target                      (compute / hardware the task runs on),
     dependencies}

RELATION TO THE KERNEL-SPECIFIC SPEC (do not duplicate it)
----------------------------------------------------------
The MI210/EPYC kernel-R&D loop already has a *specialized* instance of this
contract: the OBSERVATION JSONL record + ``runs`` table + lexicographic
``_is_correct`` gate in
``epyc-inference-research/scripts/kernel_rnd/kernel_store.py`` (SOL-ExecBench
lineage), and the ``KernelTaskSpec`` being authored alongside it in
``scripts/kernel_rnd/c6_reward_integrity.py``. Those bind the generic slots to
kernel semantics:

    seed_or_baseline_solution  <- the production-frozen kernel build (always valid)
    run_script -> solution.json <- kernel_eval.sh emitting the OBSERVATION record
    objective_scorer           <- single_tps / aggregate_tps delta
    is_correct                 <- status==OK AND full test-backend-ops pass AND
                                  coherent|byte-identical output
    target                     <- {compute: cpu|gpu, hardware: epyc-9655|mi210}

This module is the GENERAL version: the kernel spec is one specialization of it.
We intentionally live OUTSIDE ``scripts/kernel_rnd/`` and never edit that tree.

DISCIPLINE INHERITED FROM THE KERNEL LOOP
-----------------------------------------
- **Correctness is lexicographic-first.** ``is_correct`` is a HARD GATE: a
  fast-but-wrong candidate is never a search win, no matter its objective score.
  (Mirrors ``kernel_store._is_correct`` / the "speed cannot buy back
  correctness" rule.)
- **Baseline-commit-first.** ``fallback()`` guarantees a valid ``solution.json``
  exists *before any search begins*, so the loop always has a known-good anchor
  to fall back to and to diff against — the generic analogue of committing the
  frozen production kernel as ``sol_0000`` (cf. OpenHyra's seed_solution + the
  Hyra Experience Bank's always-present baseline record).
- **Every produced number is an OBSERVATION** (MEASUREMENT.md): this descriptor
  carries no protocol id and NEVER, by itself, gates a keep/deploy/promote
  decision — it structures evidence for the loop and the operator.

DEPENDENCIES
------------
Pure standard library. ``pydantic`` and ``jsonschema`` are used **only if
importable** (feature-detected) as an extra validation layer; the module is
fully functional without them (the research venv has neither).
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

# ── optional deps (never required) ────────────────────────────────────────────
try:  # pragma: no cover - availability is environment-dependent
    import pydantic as _pydantic  # noqa: F401
    _HAVE_PYDANTIC = True
except Exception:  # pragma: no cover
    _pydantic = None
    _HAVE_PYDANTIC = False

try:  # pragma: no cover
    import jsonschema as _jsonschema  # noqa: F401
    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _jsonschema = None
    _HAVE_JSONSCHEMA = False


DEFAULT_SOLUTION_FILENAME = "solution.json"
_VALID_DIRECTIONS = ("min", "max")
_VALID_COMPUTE = ("cpu", "gpu", "hybrid", "any")
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REF_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")  # "pkg.mod:function"


class TaskDescriptorError(ValueError):
    """Raised when a descriptor fails validation in strict mode."""


@dataclass
class TaskDescriptor:
    """A uniform, domain-agnostic R&D-task descriptor (the HyRA cross-domain
    contract). Generalizes the kernel-loop's ``KernelTaskSpec`` (SOL-ExecBench)
    to arbitrary domains — see module docstring.

    Fields
    ------
    task_id:
        Stable slug identifying the task (``[A-Za-z0-9][A-Za-z0-9_.-]*``).
    seed_or_baseline_solution:
        Path to an **always-valid** seed/baseline solution (a directory, or a
        single ``solution.json``). This is the fallback the search can never do
        worse than; ``fallback()`` materializes it as the committed baseline
        before any search runs.
    run_script:
        Path/command that, executed inside a solution workspace, produces
        ``<solution_filename>`` — the run contract. Domain-opaque to the loop;
        the loop only knows "run this, then read the solution file".
    objective_scorer:
        Reference to the fixed objective scorer that maps a solution.json to a
        float. Either a ``"module:function"`` import ref or a filesystem path to
        an evaluator script. Fixed per task so scores are comparable across
        rounds (cf. OpenHyra's trusted ``evaluator.py``).
    is_correct:
        Reference (``"module:function"`` or path) to the HARD correctness gate:
        solution.json -> bool. Lexicographic-first — a candidate that fails this
        is never a search win regardless of objective score.
    target:
        Compute/hardware the task runs on, e.g.
        ``{"compute": "cpu", "hardware": "epyc-9655"}``. ``compute`` is required.
    dependencies:
        Prerequisite task_ids or artifact identifiers that must exist first.
    objective_direction:
        ``"min"`` or ``"max"`` — whether lower or higher objective is better.
    solution_filename:
        Name of the file ``run_script`` must emit (default ``solution.json``).
    metric_name:
        Human-readable name of the objective metric (free-form).
    baseline_solution_json:
        Optional inline baseline solution payload. Used by ``fallback()`` when
        the seed path does not itself contain a solution file — guarantees a
        valid solution.json can always be materialized.
    metadata:
        Free-form, non-gating extra context (era labels, notes, provenance).
    """

    task_id: str
    seed_or_baseline_solution: str
    run_script: str
    objective_scorer: str
    is_correct: str
    target: dict[str, Any] = field(default_factory=lambda: {"compute": "any"})
    dependencies: list[str] = field(default_factory=list)
    objective_direction: str = "max"
    solution_filename: str = DEFAULT_SOLUTION_FILENAME
    metric_name: str = "score"
    baseline_solution_json: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── (de)serialization ────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskDescriptor":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise TaskDescriptorError(
                f"unknown descriptor field(s): {sorted(unknown)}"
            )
        return cls(**data)  # missing required fields -> TypeError, caught by load()

    # ── validation ───────────────────────────────────────────────────────────
    def validate(self, *, strict: bool = True) -> list[str]:
        """Return a list of human-readable validation errors (empty == valid).

        With ``strict=True`` (default) a non-empty list is raised as a
        :class:`TaskDescriptorError`. Validation is pure-stdlib; if
        ``jsonschema`` is importable it is additionally cross-checked against the
        emitted JSON Schema, but its absence never weakens validation.
        """
        errors: list[str] = []

        if not isinstance(self.task_id, str) or not _TASK_ID_RE.match(self.task_id or ""):
            errors.append(
                "task_id must be a non-empty slug matching "
                "[A-Za-z0-9][A-Za-z0-9_.-]*"
            )

        # The always-valid fallback is the load-bearing requirement of the whole
        # contract: without it there is no baseline-commit-first anchor.
        if not isinstance(self.seed_or_baseline_solution, str) or not self.seed_or_baseline_solution.strip():
            errors.append(
                "seed_or_baseline_solution is REQUIRED (the always-valid "
                "fallback / baseline-commit-first anchor) and must be a non-empty path"
            )

        for name in ("run_script", "objective_scorer", "is_correct"):
            val = getattr(self, name)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{name} is required and must be a non-empty string")

        # Scorer / gate refs, when written as "module:function", must be well-formed.
        for name in ("objective_scorer", "is_correct"):
            val = getattr(self, name)
            if isinstance(val, str) and ":" in val and "/" not in val and not _REF_RE.match(val):
                errors.append(
                    f"{name} looks like a module ref but is malformed "
                    f"(expected 'package.module:function'): {val!r}"
                )

        if self.objective_direction not in _VALID_DIRECTIONS:
            errors.append(
                f"objective_direction must be one of {_VALID_DIRECTIONS}, "
                f"got {self.objective_direction!r}"
            )

        if not isinstance(self.solution_filename, str) or not self.solution_filename.strip():
            errors.append("solution_filename must be a non-empty string")

        if not isinstance(self.target, dict) or "compute" not in self.target:
            errors.append("target must be a dict containing at least a 'compute' key")
        elif self.target.get("compute") not in _VALID_COMPUTE:
            errors.append(
                f"target.compute must be one of {_VALID_COMPUTE}, "
                f"got {self.target.get('compute')!r}"
            )

        if not isinstance(self.dependencies, list) or not all(
            isinstance(d, str) for d in self.dependencies
        ):
            errors.append("dependencies must be a list of task-id/artifact strings")

        if self.baseline_solution_json is not None and not isinstance(
            self.baseline_solution_json, dict
        ):
            errors.append("baseline_solution_json, when present, must be an object")

        if _HAVE_JSONSCHEMA and not errors:  # pragma: no cover - env-dependent
            try:
                _jsonschema.validate(self.to_dict(), emit_json_schema())
            except _jsonschema.ValidationError as exc:  # type: ignore[union-attr]
                errors.append(f"jsonschema: {exc.message}")

        if strict and errors:
            raise TaskDescriptorError(
                "invalid TaskDescriptor:\n  - " + "\n  - ".join(errors)
            )
        return errors

    # ── ref resolution (optional convenience) ────────────────────────────────
    def resolve_scorer(self) -> Callable[..., Any]:
        """Import and return the objective scorer callable (module-ref only)."""
        return _resolve_ref(self.objective_scorer, "objective_scorer")

    def resolve_is_correct(self) -> Callable[..., Any]:
        """Import and return the hard correctness-gate callable (module-ref only)."""
        return _resolve_ref(self.is_correct, "is_correct")

    # ── baseline-commit-first ────────────────────────────────────────────────
    def fallback(self, workspace: str, *, overwrite: bool = False) -> str:
        """Guarantee a valid ``solution.json`` exists in ``workspace`` BEFORE any
        search runs, and return its path.

        This is the generic "baseline-commit-first" rule: the search must always
        have a known-good anchor to fall back to and to diff candidates against
        (the domain-agnostic analogue of committing the frozen production kernel
        as ``sol_0000``, and of OpenHyra copying ``seed_solution/`` in as the
        first Experience-Bank record).

        Resolution order for the baseline solution file:
          1. If the seed path is itself a ``*.json`` file, copy it in.
          2. Else if the seed is a directory containing ``solution_filename``,
             copy that whole seed tree in.
          3. Else if ``baseline_solution_json`` was supplied inline, write it.
          4. Else raise — the descriptor cannot honour its always-valid promise.

        No scoring or inference is performed here; this only *materializes* the
        always-valid baseline. Raises :class:`TaskDescriptorError` if it cannot.
        """
        os.makedirs(workspace, exist_ok=True)
        dest = os.path.join(workspace, self.solution_filename)
        if os.path.exists(dest) and not overwrite:
            return dest

        seed = self.seed_or_baseline_solution

        # 1) seed is a solution.json-style file
        if os.path.isfile(seed) and seed.endswith(".json"):
            shutil.copyfile(seed, dest)
            return dest

        # 2) seed is a directory that already contains a solution file
        if os.path.isdir(seed):
            seed_sol = os.path.join(seed, self.solution_filename)
            if os.path.isfile(seed_sol):
                _copy_tree_into(seed, workspace)
                if os.path.isfile(dest):
                    return dest

        # 3) inline baseline payload
        if self.baseline_solution_json is not None:
            tmp = f"{dest}.tmp.{os.getpid()}"
            with open(tmp, "w") as fh:
                json.dump(self.baseline_solution_json, fh, indent=1)
            os.replace(tmp, dest)
            return dest

        raise TaskDescriptorError(
            f"fallback() cannot materialize a valid '{self.solution_filename}': "
            f"seed_or_baseline_solution={seed!r} is not a *.json file, is not a "
            f"directory containing '{self.solution_filename}', and no "
            f"baseline_solution_json was provided. The always-valid-fallback "
            f"guarantee is unmet — refusing to start a search without a baseline."
        )


# ── module-level helpers ──────────────────────────────────────────────────────
def _resolve_ref(ref: str, label: str) -> Callable[..., Any]:
    if not isinstance(ref, str) or ":" not in ref or "/" in ref:
        raise TaskDescriptorError(
            f"{label}={ref!r} is not an importable 'module:function' ref "
            f"(it looks like a path — load it yourself)"
        )
    mod_name, _, fn_name = ref.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        raise TaskDescriptorError(f"{label}: cannot import module {mod_name!r}: {exc}")
    try:
        obj = getattr(mod, fn_name)
    except AttributeError:
        raise TaskDescriptorError(f"{label}: {mod_name!r} has no attribute {fn_name!r}")
    if not callable(obj):
        raise TaskDescriptorError(f"{label}: {ref!r} resolved to a non-callable")
    return obj


def _copy_tree_into(src: str, dst: str) -> None:
    """Copy the contents of directory ``src`` into ``dst`` (dst may exist)."""
    for entry in os.listdir(src):
        s = os.path.join(src, entry)
        d = os.path.join(dst, entry)
        if os.path.isdir(s):
            shutil.copytree(
                s, d, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".venv", "__pycache__", ".git"),
            )
        else:
            shutil.copyfile(s, d)


def load(path: str, *, strict: bool = True) -> TaskDescriptor:
    """Load a descriptor from a JSON file and validate it.

    Raises :class:`TaskDescriptorError` on malformed JSON, missing required
    fields, unknown fields, or (in strict mode) any validation error.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise TaskDescriptorError(f"descriptor file not found: {path}")
    except json.JSONDecodeError as exc:
        raise TaskDescriptorError(f"descriptor {path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise TaskDescriptorError(f"descriptor {path} must be a JSON object")
    try:
        desc = TaskDescriptor.from_dict(data)
    except TypeError as exc:
        raise TaskDescriptorError(f"descriptor {path} is missing required field(s): {exc}")
    desc.validate(strict=strict)
    return desc


def emit_json_schema() -> dict[str, Any]:
    """Return a JSON Schema (Draft 2020-12) for the descriptor.

    Emitting requires no third-party library; ``jsonschema`` is only used to
    *validate against* this schema when it happens to be installed.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://epyc.local/schemas/rnd_harness/task_descriptor.json",
        "title": "TaskDescriptor",
        "description": (
            "Domain-agnostic per-task descriptor for the EPYC agent-collab R&D "
            "harness (HyRA cross-domain contract). Generalizes the kernel-loop "
            "KernelTaskSpec / SOL-ExecBench record to arbitrary domains."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "task_id",
            "seed_or_baseline_solution",
            "run_script",
            "objective_scorer",
            "is_correct",
        ],
        "properties": {
            "task_id": {
                "type": "string",
                "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
                "description": "Stable task slug.",
            },
            "seed_or_baseline_solution": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Path to the always-valid seed/baseline solution (the "
                    "fallback / baseline-commit-first anchor)."
                ),
            },
            "run_script": {
                "type": "string",
                "minLength": 1,
                "description": "Command/path that emits solution_filename (run contract).",
            },
            "objective_scorer": {
                "type": "string",
                "minLength": 1,
                "description": "Fixed objective scorer: 'module:function' ref or path.",
            },
            "is_correct": {
                "type": "string",
                "minLength": 1,
                "description": "Hard correctness gate (lexicographic-first): ref or path.",
            },
            "target": {
                "type": "object",
                "required": ["compute"],
                "properties": {
                    "compute": {"type": "string", "enum": list(_VALID_COMPUTE)},
                    "hardware": {"type": "string"},
                },
                "additionalProperties": True,
                "description": "Compute/hardware the task runs on.",
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Prerequisite task_ids / artifact identifiers.",
            },
            "objective_direction": {
                "type": "string",
                "enum": list(_VALID_DIRECTIONS),
                "default": "max",
            },
            "solution_filename": {"type": "string", "default": DEFAULT_SOLUTION_FILENAME},
            "metric_name": {"type": "string", "default": "score"},
            "baseline_solution_json": {
                "type": ["object", "null"],
                "description": "Optional inline baseline solution payload for fallback().",
            },
            "metadata": {"type": "object", "description": "Non-gating extra context."},
        },
    }


def pydantic_model():  # pragma: no cover - only exercised where pydantic exists
    """Return a pydantic ``BaseModel`` mirroring the descriptor, or ``None`` if
    pydantic is not installed. Optional convenience for callers in richer envs;
    the core validation path never depends on it.
    """
    if not _HAVE_PYDANTIC:
        return None
    from pydantic import BaseModel, Field  # local import; optional dep

    class TaskDescriptorModel(BaseModel):
        model_config = {"extra": "forbid"}
        task_id: str
        seed_or_baseline_solution: str
        run_script: str
        objective_scorer: str
        is_correct: str
        target: dict = Field(default_factory=lambda: {"compute": "any"})
        dependencies: list[str] = Field(default_factory=list)
        objective_direction: str = "max"
        solution_filename: str = DEFAULT_SOLUTION_FILENAME
        metric_name: str = "score"
        baseline_solution_json: Optional[dict] = None
        metadata: dict = Field(default_factory=dict)

    return TaskDescriptorModel


if __name__ == "__main__":  # pragma: no cover - CLI smoke path
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "schema":
        print(json.dumps(emit_json_schema(), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "validate":
        d = load(sys.argv[2])
        print(f"OK: {d.task_id} is a valid TaskDescriptor "
              f"(pydantic={_HAVE_PYDANTIC}, jsonschema={_HAVE_JSONSCHEMA})")
    else:
        print("usage: task_descriptor.py [schema | validate <path.json>]")
