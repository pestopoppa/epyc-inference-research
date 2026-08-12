"""Observation-only AK-LE-1/2/3 controller experiment contracts.

This module plans and reduces matched authoring experiments.  It deliberately
has no model client, process runner, evaluator import, campaign hook, ranking
API, champion API, or release API.  Callers may execute the predeclared cells
through the governed Arena boundary and bring the completed observations back
here; importing this module can never execute a cell.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable

from . import authoring_contract


CONTRACT_SCHEMA = "epyc.autokernel.loop_engineering_experiment.v1"
RECEIPT_SCHEMA = "epyc.autokernel.loop_engineering_receipt.v1"
AUTHORITY = "observe_only_no_campaign_ranking_or_release_authority"
TARGET_ABSENT = "absent"
TARGET_RENDERED = "rendered_context_line"
SCAFFOLD_DIRECT = "direct_implement"
SCAFFOLD_SPLIT = "implement_then_exploit"
TERMINATIONS = frozenset({
    "already_optimized", "budget_exhausted", "search_exhausted",
})
DIRECTIONS = frozenset({
    "higher_effort_increases_search_persistence",
    "higher_effort_decreases_search_persistence",
    "no_directional_change",
})
_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,95}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class LoopExperimentError(ValueError):
    """The experiment is confounded, mutable, incomplete, or authority-seeking."""


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise LoopExperimentError(f"{label} must be non-empty text without NUL")
    return value.strip()


def _sha(value: object, label: str) -> str:
    value = _text(value, label)
    if not _SHA_RE.fullmatch(value):
        raise LoopExperimentError(f"{label} must be a lowercase SHA-256")
    return value


def _positive(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise LoopExperimentError(f"{label} must be positive and finite")
    return float(value)


def _id(value: object, label: str) -> str:
    value = _text(value, label)
    if not _ID_RE.fullmatch(value):
        raise LoopExperimentError(f"{label} is not a stable identifier")
    return value


@dataclass(frozen=True)
class ArtifactPin:
    ref: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.ref, "artifact ref")
        _sha(self.sha256, "artifact sha256")

    def to_dict(self) -> dict[str, str]:
        return {"ref": self.ref, "sha256": self.sha256}


@dataclass(frozen=True)
class SelectedTaskArtifact:
    """Exact selected hypothesis/task bytes supplied to every scaffold arm."""

    ref: str
    task: str
    task_sha256: str

    def __post_init__(self) -> None:
        _text(self.ref, "selected task ref")
        task = _text(self.task, "selected task")
        if task != self.task:
            raise LoopExperimentError(
                "selected task cannot have unbound surrounding whitespace")
        _sha(self.task_sha256, "selected task SHA-256")
        if hashlib.sha256(self.task.encode("utf-8")).hexdigest() != self.task_sha256:
            raise LoopExperimentError(
                "task_sha256 does not bind the exact selected task bytes")

    def to_dict(self) -> dict[str, str]:
        return {
            "ref": self.ref,
            "task": self.task,
            "task_sha256": self.task_sha256,
        }


@dataclass(frozen=True)
class FixedPromptFrame:
    """Immutable planner and scaffold inputs, each bound independently."""

    champion: ArtifactPin
    retrieval_context_sha256: str
    propose_prompt: str
    propose_prompt_sha256: str
    selected_task: SelectedTaskArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.champion, ArtifactPin):
            raise TypeError("champion must be an ArtifactPin")
        if not isinstance(self.selected_task, SelectedTaskArtifact):
            raise TypeError("selected_task must be a SelectedTaskArtifact")
        _sha(self.retrieval_context_sha256, "retrieval_context_sha256")
        prompt = _text(self.propose_prompt, "propose_prompt")
        _sha(self.propose_prompt_sha256, "propose_prompt_sha256")
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != self.propose_prompt_sha256:
            raise LoopExperimentError("propose_prompt_sha256 does not bind PROPOSE text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion": self.champion.to_dict(),
            "retrieval_context_sha256": self.retrieval_context_sha256,
            "propose_prompt": self.propose_prompt,
            "propose_prompt_sha256": self.propose_prompt_sha256,
            "selected_task": self.selected_task.to_dict(),
        }


@dataclass(frozen=True)
class PlannerArm:
    cell_id: str
    model_id: str
    quant_id: str
    effort: str
    target_context_mode: str

    def __post_init__(self) -> None:
        _id(self.cell_id, "planner cell_id")
        for name in ("model_id", "quant_id", "effort"):
            _text(getattr(self, name), f"planner {name}")
        if self.target_context_mode not in {TARGET_ABSENT, TARGET_RENDERED}:
            raise LoopExperimentError("target_context_mode is unsupported")

    @property
    def model_quant(self) -> tuple[str, str]:
        return self.model_id, self.quant_id

    def to_dict(self) -> dict[str, str]:
        # The target VALUE is intentionally absent.  Only the matched arm shape is
        # predeclared; its value can exist solely in the rendered prompt.
        return {
            "cell_id": self.cell_id, "model_id": self.model_id,
            "quant_id": self.quant_id, "effort": self.effort,
            "target_context_mode": self.target_context_mode,
        }


@dataclass(frozen=True)
class DirectionPrediction:
    model_id: str
    quant_id: str
    direction: str
    rationale: str

    def __post_init__(self) -> None:
        _text(self.model_id, "prediction model_id")
        _text(self.quant_id, "prediction quant_id")
        if self.direction not in DIRECTIONS:
            raise LoopExperimentError("direction prediction is unsupported")
        _text(self.rationale, "direction rationale")

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id, "quant_id": self.quant_id,
            "direction": self.direction, "rationale": self.rationale,
        }


@dataclass(frozen=True)
class RoleBudget:
    role: str
    wall_seconds: float
    instruction: str
    instruction_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"implement", "exploit"}:
            raise LoopExperimentError("role must be implement or exploit")
        _positive(self.wall_seconds, "role wall_seconds")
        instruction = _text(self.instruction, "role instruction")
        _sha(self.instruction_sha256, "role instruction_sha256")
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != self.instruction_sha256:
            raise LoopExperimentError("instruction_sha256 does not bind role instruction")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "wall_seconds": float(self.wall_seconds),
            "instruction": self.instruction,
            "instruction_sha256": self.instruction_sha256,
        }


@dataclass(frozen=True)
class ScaffoldArm:
    cell_id: str
    model_id: str
    quant_id: str
    effort: str
    scaffold: str
    roles: tuple[RoleBudget, ...]

    def __post_init__(self) -> None:
        _id(self.cell_id, "scaffold cell_id")
        for name in ("model_id", "quant_id", "effort"):
            _text(getattr(self, name), f"scaffold {name}")
        if self.scaffold not in {SCAFFOLD_DIRECT, SCAFFOLD_SPLIT}:
            raise LoopExperimentError("unknown scaffold")
        if not isinstance(self.roles, tuple):
            raise LoopExperimentError("roles must be an immutable tuple")
        names = tuple(role.role for role in self.roles)
        expected = (("implement",) if self.scaffold == SCAFFOLD_DIRECT
                    else ("implement", "exploit"))
        if names != expected:
            raise LoopExperimentError(
                f"{self.scaffold} roles must be ordered {expected}")

    @property
    def model_quant_effort(self) -> tuple[str, str, str]:
        return self.model_id, self.quant_id, self.effort

    @property
    def wall_seconds(self) -> float:
        return sum(role.wall_seconds for role in self.roles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id, "model_id": self.model_id,
            "quant_id": self.quant_id, "effort": self.effort,
            "scaffold": self.scaffold,
            "roles": [role.to_dict() for role in self.roles],
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True)
class ExperimentContract:
    experiment_id: str
    fixed: FixedPromptFrame
    planner_arms: tuple[PlannerArm, ...]
    predictions: tuple[DirectionPrediction, ...]
    scaffold_arms: tuple[ScaffoldArm, ...]
    prior_hypothesis_sha256: tuple[str, ...]
    prefilter: ArtifactPin

    def __post_init__(self) -> None:
        _id(self.experiment_id, "experiment_id")
        if not isinstance(self.fixed, FixedPromptFrame):
            raise TypeError("fixed must be a FixedPromptFrame")
        if not isinstance(self.prefilter, ArtifactPin):
            raise TypeError("prefilter must be an ArtifactPin")
        ids = [arm.cell_id for arm in (*self.planner_arms, *self.scaffold_arms)]
        if len(ids) != len(set(ids)):
            raise LoopExperimentError("experiment cell ids must be unique")
        prior = tuple(_sha(value, "prior hypothesis SHA-256")
                      for value in self.prior_hypothesis_sha256)
        if prior != tuple(sorted(set(prior))):
            raise LoopExperimentError(
                "prior_hypothesis_sha256 must be sorted and duplicate-free")

        planner_groups: dict[tuple[str, str], set[str]] = {}
        target_pairs: dict[tuple[str, str, str], set[str]] = {}
        for arm in self.planner_arms:
            planner_groups.setdefault(arm.model_quant, set()).add(arm.effort)
            target_pairs.setdefault((*arm.model_quant, arm.effort), set()).add(
                arm.target_context_mode)
        if not planner_groups or any(len(efforts) < 2 for efforts in planner_groups.values()):
            raise LoopExperimentError(
                "AK-LE-1 requires at least two effort levels for every model/quant")
        if any(modes != {TARGET_ABSENT, TARGET_RENDERED}
               for modes in target_pairs.values()):
            raise LoopExperimentError(
                "AK-LE-2 requires matched absent/rendered-context cells at every effort")
        prediction_keys = [(row.model_id, row.quant_id) for row in self.predictions]
        if (len(prediction_keys) != len(set(prediction_keys))
                or set(prediction_keys) != set(planner_groups)):
            raise LoopExperimentError(
                "each model/quant effort sweep needs exactly one predeclared prediction")

        scaffold_groups: dict[tuple[str, str, str], dict[str, ScaffoldArm]] = {}
        for arm in self.scaffold_arms:
            scaffold_groups.setdefault(arm.model_quant_effort, {})[arm.scaffold] = arm
        if len({key[0] for key in scaffold_groups}) < 2:
            raise LoopExperimentError(
                "AK-LE-3 requires at least two models so model and "
                "scaffold effects are independently estimable")
        totals: set[float] = set()
        for arms in scaffold_groups.values():
            if set(arms) != {SCAFFOLD_DIRECT, SCAFFOLD_SPLIT}:
                raise LoopExperimentError(
                    "every model cell needs both direct and implement-then-exploit scaffolds")
            totals.update(round(arm.wall_seconds, 9) for arm in arms.values())
        if len(totals) != 1:
            raise LoopExperimentError("every scaffold arm must have the same wall-time budget")

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "schema": CONTRACT_SCHEMA,
            "experiment_id": self.experiment_id,
            "authority": AUTHORITY,
            "fixed": self.fixed.to_dict(),
            "planner_arms": [row.to_dict() for row in self.planner_arms],
            "direction_predictions": [row.to_dict() for row in self.predictions],
            "scaffold_arms": [row.to_dict() for row in self.scaffold_arms],
            "prior_hypothesis_sha256": list(self.prior_hypothesis_sha256),
            "prefilter": self.prefilter.to_dict(),
            "constraints": {
                "campaign_1_authority": False, "ranking_authority": False,
                "champion_authority": False, "release_authority": False,
                "model_or_kernel_invoked_by_contract": False,
                "target_value_is_manifest_field": False,
            },
        }
        payload["contract_sha256"] = _digest(payload)
        return payload


def context_sha256(context: authoring_contract.PricedContext) -> str:
    if not isinstance(context, authoring_contract.PricedContext):
        raise TypeError("context must be a PricedContext")
    return _digest({
        "round_id": context.round_id,
        "budget": {
            "max_total_tokens": context.budget.max_total_tokens,
            "max_item_tokens": context.budget.max_item_tokens,
            "max_items": context.budget.max_items,
        },
        "items": [{
            "source_ref": row.source_ref, "purpose": row.purpose,
            "content": row.content, "bulk_read": row.bulk_read,
        } for row in context.items],
    })


def render_planner_prompt(contract: ExperimentContract, cell_id: str, *,
                          context: authoring_contract.PricedContext,
                          target_line: str | None = None) -> str:
    """Render one planner cell; a proximate value exists only in this return value."""
    arms = {arm.cell_id: arm for arm in contract.planner_arms}
    if cell_id not in arms:
        raise LoopExperimentError("unknown planner cell")
    if context_sha256(context) != contract.fixed.retrieval_context_sha256:
        raise LoopExperimentError("retrieval context differs from the predeclared frame")
    arm = arms[cell_id]
    if arm.target_context_mode == TARGET_ABSENT:
        if target_line is not None:
            raise LoopExperimentError("control cell cannot receive a target context line")
        suffix = ""
    else:
        line = _text(target_line, "target context line")
        if "\n" in line or not line.startswith("PROXIMATE AUTHORING TARGET: "):
            raise LoopExperimentError(
                "target must be one rendered planner-context line with the reviewed prefix")
        suffix = f"\n{line}"
    task = (
        f"{contract.fixed.propose_prompt}\n"
        f"FIXED CHAMPION: {contract.fixed.champion.ref} "
        f"sha256={contract.fixed.champion.sha256}{suffix}"
    )
    return authoring_contract.assemble_authoring_prompt(
        role="planner", task=task, context=context)


def render_scaffold_prompt(contract: ExperimentContract, cell_id: str, role: str, *,
                           context: authoring_contract.PricedContext) -> str:
    arms = {arm.cell_id: arm for arm in contract.scaffold_arms}
    if cell_id not in arms:
        raise LoopExperimentError("unknown scaffold cell")
    if context_sha256(context) != contract.fixed.retrieval_context_sha256:
        raise LoopExperimentError("retrieval context differs from the predeclared frame")
    arm = arms[cell_id]
    stages = {stage.role: stage for stage in arm.roles}
    if role not in stages:
        raise LoopExperimentError("role is not predeclared for this scaffold cell")
    stage = stages[role]
    selected = contract.fixed.selected_task
    task = (
        f"SELECTED TASK ARTIFACT: {selected.ref} "
        f"sha256={selected.task_sha256}\n"
        "BEGIN EXACT SELECTED TASK\n"
        f"{selected.task}\n"
        "END EXACT SELECTED TASK\n"
        f"FIXED CHAMPION: {contract.fixed.champion.ref} "
        f"sha256={contract.fixed.champion.sha256}\n"
        f"STAGE INSTRUCTION: {stage.instruction}"
    )
    return authoring_contract.assemble_authoring_prompt(
        role=role, task=task, context=context)


def _semantic_text(value: object, label: str) -> str:
    return " ".join(_text(value, label).casefold().split())


@dataclass(frozen=True)
class HypothesisObservation:
    mechanism: str
    target_surface: str
    falsifiable_counter: str
    predicted_direction: str
    survived_prefilter: bool

    def __post_init__(self) -> None:
        for name in ("mechanism", "target_surface", "falsifiable_counter",
                     "predicted_direction"):
            _semantic_text(getattr(self, name), name)
        if not isinstance(self.survived_prefilter, bool):
            raise LoopExperimentError("survived_prefilter must be boolean")

    @property
    def fingerprint(self) -> str:
        return _digest({
            "mechanism": _semantic_text(self.mechanism, "mechanism"),
            "target_surface": _semantic_text(self.target_surface, "target_surface"),
            "falsifiable_counter": _semantic_text(
                self.falsifiable_counter, "falsifiable_counter"),
            "predicted_direction": _semantic_text(
                self.predicted_direction, "predicted_direction"),
        })


@dataclass(frozen=True)
class PlannerObservation:
    cell_id: str
    termination: str
    hypotheses: tuple[HypothesisObservation, ...]
    elapsed_wall_seconds: float
    evidence_sha256: str

    def __post_init__(self) -> None:
        _id(self.cell_id, "planner observation cell_id")
        if self.termination not in TERMINATIONS:
            raise LoopExperimentError("planner termination is not explicit")
        if not isinstance(self.hypotheses, tuple):
            raise LoopExperimentError("hypotheses must be an immutable tuple")
        _positive(self.elapsed_wall_seconds, "planner elapsed_wall_seconds")
        _sha(self.evidence_sha256, "planner evidence_sha256")


@dataclass(frozen=True)
class RoleObservation:
    role: str
    elapsed_wall_seconds: float
    candidates_emitted: int
    candidates_survived_prefilter: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"implement", "exploit"}:
            raise LoopExperimentError("observed role must be implement or exploit")
        _positive(self.elapsed_wall_seconds, "role elapsed_wall_seconds")
        for name in ("candidates_emitted", "candidates_survived_prefilter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LoopExperimentError(f"{name} must be a nonnegative integer")
        if self.candidates_survived_prefilter > self.candidates_emitted:
            raise LoopExperimentError("prefilter survivors cannot exceed emitted candidates")
        _sha(self.evidence_sha256, "role evidence_sha256")


@dataclass(frozen=True)
class ScaffoldObservation:
    cell_id: str
    roles: tuple[RoleObservation, ...]

    def __post_init__(self) -> None:
        _id(self.cell_id, "scaffold observation cell_id")
        if not isinstance(self.roles, tuple):
            raise LoopExperimentError("observed roles must be an immutable tuple")


def reduce_receipt(contract: ExperimentContract, *,
                   planner_observations: Iterable[PlannerObservation],
                   scaffold_observations: Iterable[ScaffoldObservation],
                   capture_mode: str) -> dict[str, Any]:
    """Reduce a complete matched panel without selecting or ranking a kernel."""
    if capture_mode not in {"measured", "fixture"}:
        raise LoopExperimentError("capture_mode must be measured or fixture")
    planners = tuple(planner_observations)
    scaffolds = tuple(scaffold_observations)
    planner_map = {row.cell_id: row for row in planners}
    scaffold_map = {row.cell_id: row for row in scaffolds}
    if len(planner_map) != len(planners) or set(planner_map) != {
            arm.cell_id for arm in contract.planner_arms}:
        raise LoopExperimentError("planner observations must cover every cell exactly once")
    if len(scaffold_map) != len(scaffolds) or set(scaffold_map) != {
            arm.cell_id for arm in contract.scaffold_arms}:
        raise LoopExperimentError("scaffold observations must cover every cell exactly once")

    prior = set(contract.prior_hypothesis_sha256)
    search_rows = []
    for arm in contract.planner_arms:
        observation = planner_map[arm.cell_id]
        fingerprints = [row.fingerprint for row in observation.hypotheses]
        unique = set(fingerprints)
        search_rows.append({
            **arm.to_dict(),
            "hypotheses_total": len(fingerprints),
            "hypotheses_unique": len(unique),
            "duplicate_count": len(fingerprints) - len(unique),
            "novel_nonduplicate_count": len(unique - prior),
            "already_optimized_termination_count": int(
                observation.termination == "already_optimized"),
            "prefilter_survival_count": sum(
                row.survived_prefilter for row in observation.hypotheses),
            "termination": observation.termination,
            "elapsed_wall_seconds": observation.elapsed_wall_seconds,
            "evidence_sha256": observation.evidence_sha256,
        })

    scaffold_rows = []
    arm_map = {arm.cell_id: arm for arm in contract.scaffold_arms}
    for cell_id in [arm.cell_id for arm in contract.scaffold_arms]:
        arm = arm_map[cell_id]
        observation = scaffold_map[cell_id]
        expected_roles = tuple(row.role for row in arm.roles)
        observed_roles = tuple(row.role for row in observation.roles)
        if observed_roles != expected_roles:
            raise LoopExperimentError("observed roles differ from predeclared role order")
        for planned, observed in zip(arm.roles, observation.roles):
            if observed.elapsed_wall_seconds > planned.wall_seconds:
                raise LoopExperimentError("role exceeded its predeclared wall-time budget")
        elapsed = sum(row.elapsed_wall_seconds for row in observation.roles)
        survived = sum(row.candidates_survived_prefilter for row in observation.roles)
        scaffold_rows.append({
            **arm.to_dict(),
            "observed_wall_seconds": elapsed,
            "candidates_emitted": sum(row.candidates_emitted for row in observation.roles),
            "candidates_survived_prefilter": survived,
            "survivors_per_wall_hour": survived * 3600.0 / elapsed,
            "role_observations": [{
                "role": row.role,
                "elapsed_wall_seconds": row.elapsed_wall_seconds,
                "candidates_emitted": row.candidates_emitted,
                "candidates_survived_prefilter": row.candidates_survived_prefilter,
                "evidence_sha256": row.evidence_sha256,
            } for row in observation.roles],
        })

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "experiment_id": contract.experiment_id,
        "contract_sha256": contract.to_manifest()["contract_sha256"],
        "capture_mode": capture_mode,
        "authority": AUTHORITY,
        "search_persistence_observations": search_rows,
        "scaffold_throughput_observations": scaffold_rows,
        "objective": {
            "metric": "prefilter_survivors_per_wall_hour",
            "direction": "higher_is_better",
            "wall_time_budget_matched": True,
        },
        "constraints": {
            "empirical_claim": capture_mode == "measured",
            "campaign_1_authority": False, "ranking_authority": False,
            "champion_authority": False, "release_authority": False,
            "controller_ab_authority": False,
        },
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


__all__ = [
    "AUTHORITY", "CONTRACT_SCHEMA", "DIRECTIONS", "ExperimentContract",
    "FixedPromptFrame", "ArtifactPin", "SelectedTaskArtifact", "PlannerArm",
    "DirectionPrediction",
    "RoleBudget", "ScaffoldArm", "HypothesisObservation", "PlannerObservation",
    "RoleObservation", "ScaffoldObservation", "LoopExperimentError",
    "TARGET_ABSENT", "TARGET_RENDERED", "SCAFFOLD_DIRECT", "SCAFFOLD_SPLIT",
    "context_sha256", "render_planner_prompt", "render_scaffold_prompt",
    "reduce_receipt",
]
