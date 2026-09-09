"""Prospective inputs for serving calibration through the existing worker.

These records are preparation advice, not a resource grant, a control panel, or
an assertion that raw observations qualify for search. In particular, an accepted
numerical CalibrationSolve is not an OriginalCalibrationEvidenceReference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from .. import schemas
from ..evaluator import api, statistics as st
from . import experiment_plan as ep, planned_serving as ps, scheduling
from .resolved_recipe import CanonicalResolvedRecipe, resolved_recipe_from_dict

DECLARATION_SCHEMA = "epyc.autokernel.serving_preparation_declaration.v1"
STATISTICS_SCHEMA = "epyc.autokernel.serving_statistics_declaration.v1"
PAIR_SCHEMA = "epyc.autokernel.preparation_arm_pair.v1"
REQUEST_SCHEMA = "epyc.autokernel.calibration_preparation_request.v1"
DISPATCH_SCHEMA = "epyc.autokernel.calibration_preparation_dispatch.v1"
MAX_CONFIGURATION_BYTES = 4 * 1024 * 1024  # Existing driver config-reader bound.


class PreparationRefused(ValueError):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _bytes(value: Any) -> bytes:
    try:
        return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (ValueError, TypeError) as exc:
        raise PreparationRefused("preparation requires finite canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PreparationRefused(f"{label} fields differ")
    _bytes(value)
    return _plain(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparationRefused(f"{label} must be nonempty text")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value):
        raise PreparationRefused(f"{label} must be lowercase SHA-256")
    return value


def _instant(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_text(value, label).replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("timezone missing")
        return parsed
    except ValueError as exc:
        raise PreparationRefused(f"{label} must be an offset ISO-8601 instant") from exc


def _positive(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise PreparationRefused(f"{label} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class PreparationArmPair:
    """A/A or explicitly supplied byte-identical executable-copy material.

The neutral reference records the supplied material's provenance; it is not a
control-2 PASS. The existing recipe normalization establishes equal execution
semantics, and the existing worker must still reopen/hash the actual binaries.
"""

    kind: str
    anchor: CanonicalResolvedRecipe
    candidate: CanonicalResolvedRecipe
    neutral_material_ref: str | None
    schema: str = PAIR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIR_SCHEMA or self.kind not in {"aa", "neutral"}:
            raise PreparationRefused("preparation pair schema/kind is unsupported")
        for name in ("anchor", "candidate"):
            value = getattr(self, name)
            if type(value) is not CanonicalResolvedRecipe:
                raise PreparationRefused("preparation requires canonical resolved recipes")
            value.validate_launch(value.template, value.build_dir, value.port)
        if self.kind == "aa":
            if (self.anchor.to_dict() != self.candidate.to_dict()
                    or self.neutral_material_ref is not None):
                raise PreparationRefused("A/A requires the exact anchor and no neutral label")
        else:
            _text(self.neutral_material_ref, "neutral_material_ref")
            if (self.anchor.execution_digest != self.candidate.execution_digest
                    or self.anchor.executable.path == self.candidate.executable.path
                    or self.anchor.executable.sha256 != self.candidate.executable.sha256
                    or self.anchor.port != self.candidate.port):
                raise PreparationRefused(
                    "neutral requires an explicit byte-identical executable copy, "
                    "not a runtime intervention or relabelled A/A")

    @classmethod
    def from_dict(cls, value: Any) -> PreparationArmPair:
        row = _exact(value, {"schema", "kind", "anchor", "candidate",
                             "neutral_material_ref"}, "preparation pair")
        return cls(row["kind"], resolved_recipe_from_dict(row["anchor"]),
                   resolved_recipe_from_dict(row["candidate"]),
                   row["neutral_material_ref"], row["schema"])

    def to_dict(self) -> dict:
        return {"schema": self.schema, "kind": self.kind,
                "anchor": self.anchor.to_dict(), "candidate": self.candidate.to_dict(),
                "neutral_material_ref": self.neutral_material_ref}


@dataclass(frozen=True)
class ServingStatisticsDeclaration:
    """Full original statistical choices, never reconstructed from a floor."""

    campaign_seed: str
    controls: api.CampaignControls
    stopping_rule: st.StoppingRule
    commitment: st.StoppingRuleCommitment
    split_rule: st.StratumSplitRule
    construction_id: str
    effect_scale: str
    hypothesis: str
    margin: float
    owning_rep_rule: st.OwningProtocolRepRule
    schema: str = STATISTICS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATISTICS_SCHEMA:
            raise PreparationRefused("statistics declaration schema is unsupported")
        _text(self.campaign_seed, "campaign_seed")
        for name, kind in (("controls", api.CampaignControls),
                           ("stopping_rule", st.StoppingRule),
                           ("commitment", st.StoppingRuleCommitment),
                           ("split_rule", st.StratumSplitRule),
                           ("owning_rep_rule", st.OwningProtocolRepRule)):
            if type(getattr(self, name)) is not kind:
                raise PreparationRefused(f"statistics {name} requires its exact owning type")
        if self.commitment.verify(self.stopping_rule).outcome != schemas.PASS:
            raise PreparationRefused("stopping rule differs from original commitment")
        _instant(self.commitment.committed_at, "committed_at")
        if self.split_rule.campaign_seed != self.campaign_seed:
            raise PreparationRefused("split and statistical campaign seeds differ")
        if self.stopping_rule.max_blocks_per_candidate != self.controls.max_blocks_per_candidate:
            raise PreparationRefused("stopping rule and controls block ceilings differ")
        st.select_construction(self.construction_id)
        if self.effect_scale not in st.EFFECT_SCALES:
            raise PreparationRefused("statistical effect scale is unsupported")
        st.null_boundary_for(self.hypothesis, self.margin)
        _bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> ServingStatisticsDeclaration:
        row = _exact(value, {"schema", "campaign_seed", "controls", "stopping_rule",
            "commitment", "split_rule", "construction_id", "effect_scale", "hypothesis",
            "margin", "owning_rep_rule"}, "statistics declaration")
        control = _exact(row["controls"], set(api.CampaignControls.__dataclass_fields__),
                         "campaign controls")
        rule = _exact(row["stopping_rule"], set(st.StoppingRule.__dataclass_fields__),
                      "stopping rule")
        rule["decisions"] = tuple(tuple(pair) for pair in rule["decisions"])
        rule["extension"] = st.BoundedExtension(**_exact(rule["extension"],
            set(st.BoundedExtension.__dataclass_fields__), "bounded extension"))
        if rule["futility"] is not None:
            rule["futility"] = st.FutilityRule(**_exact(rule["futility"], {"kind"}, "futility"))
        split = _exact(row["split_rule"], {"rule_id", "campaign_seed",
            "confirmation_fraction", "rotation", "campaign_ordinal"}, "split rule")
        split["rotation"] = st.RotationSchedule(**_exact(split["rotation"],
            set(st.RotationSchedule.__dataclass_fields__), "rotation"))
        return cls(row["campaign_seed"], api.CampaignControls(**control),
            st.StoppingRule(**rule), st.StoppingRuleCommitment(**_exact(row["commitment"],
                set(st.StoppingRuleCommitment.__dataclass_fields__), "commitment")),
            st.StratumSplitRule(**split), row["construction_id"], row["effect_scale"],
            row["hypothesis"], row["margin"], st.OwningProtocolRepRule(**_exact(
                row["owning_rep_rule"], set(st.OwningProtocolRepRule.__dataclass_fields__),
                "owning rep rule")), row["schema"])

    def to_dict(self) -> dict:
        return {"schema": self.schema, "campaign_seed": self.campaign_seed,
                "controls": asdict(self.controls), "stopping_rule": self.stopping_rule.to_dict(),
                "commitment": self.commitment.to_dict(), "split_rule": asdict(self.split_rule),
                "construction_id": self.construction_id, "effect_scale": self.effect_scale,
                "hypothesis": self.hypothesis, "margin": self.margin,
                "owning_rep_rule": self.owning_rep_rule.to_dict()}


@dataclass(frozen=True)
class PreparationRetryPolicy:
    """Prospective finite retries of an invalid independent pair, never values."""

    max_attempts: int
    retry_on: tuple[str, ...]
    schema: str = "epyc.autokernel.preparation_retry_policy.v1"

    def __post_init__(self) -> None:
        if (self.schema != "epyc.autokernel.preparation_retry_policy.v1"
                or type(self.max_attempts) is not int or self.max_attempts < 1
                or tuple(self.retry_on) != ("failed", "contaminated")):
            raise PreparationRefused("retry policy requires finite attempts and explicit failure dispositions")
        object.__setattr__(self, "retry_on", tuple(self.retry_on))

    @classmethod
    def from_dict(cls, value: Any) -> PreparationRetryPolicy:
        return cls(**_exact(value, set(cls.__dataclass_fields__), "retry policy"))

    def to_dict(self) -> dict:
        return _plain(asdict(self))


@dataclass(frozen=True)
class ServingPreparationDeclaration:
    declaration_id: str
    campaign_id: str
    target_revision: str
    issued_at: str
    frame: Mapping[str, Any]
    statistics: ServingStatisticsDeclaration
    aa_pair: PreparationArmPair
    neutral_pair: PreparationArmPair | None
    prompt_manifest_digest: str
    resources: scheduling.ResourceVector
    max_stage_seconds: float
    teardown_seconds: float
    source_identities: Mapping[str, str]
    retry_policy: PreparationRetryPolicy
    schema: str = DECLARATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DECLARATION_SCHEMA:
            raise PreparationRefused("preparation declaration schema is unsupported")
        _text(self.declaration_id, "declaration_id")
        _text(self.campaign_id, "campaign_id")
        _sha(self.target_revision, "target_revision")
        issued = _instant(self.issued_at, "issued_at")
        if type(self.statistics) is not ServingStatisticsDeclaration:
            raise PreparationRefused("preparation requires original statistics declaration")
        if type(self.retry_policy) is not PreparationRetryPolicy:
            raise PreparationRefused("preparation requires its original finite retry policy")
        if (self.statistics.commitment.campaign_id != self.campaign_id
                or _instant(self.statistics.commitment.committed_at, "committed_at") > issued):
            raise PreparationRefused("commitment is foreign or later than preparation issue")
        frame = _exact(self.frame, {"backend", "phase", "cell_class", "model_sha256",
            "quant", "metric", "metric_direction", "estimator_id", "epoch"}, "frame")
        for key, value in frame.items():
            _text(value, f"frame.{key}")
        _sha(frame["model_sha256"], "frame.model_sha256")
        st.orient(0.0, frame["metric_direction"])
        if type(self.aa_pair) is not PreparationArmPair or self.aa_pair.kind != "aa":
            raise PreparationRefused("declaration needs its original exact A/A pair")
        if (self.aa_pair.anchor.model.sha256 != frame["model_sha256"]
                or self.aa_pair.anchor.backend != frame["backend"]):
            raise PreparationRefused("frame model/backend differs from canonical anchor")
        if self.neutral_pair is not None and (type(self.neutral_pair) is not PreparationArmPair
                or self.neutral_pair.kind != "neutral"
                or self.neutral_pair.anchor.to_dict() != self.aa_pair.anchor.to_dict()):
            raise PreparationRefused("neutral does not use the original calibration anchor")
        _sha(self.prompt_manifest_digest, "prompt_manifest_digest")
        if type(self.resources) is not scheduling.ResourceVector:
            raise PreparationRefused("resources require an explicit scheduler ResourceVector")
        if self.resources.physical_region_fraction <= 0:
            raise PreparationRefused("calibration needs a positive declared CPU fraction")
        object.__setattr__(self, "max_stage_seconds", _positive(
            self.max_stage_seconds, "max_stage_seconds"))
        object.__setattr__(self, "teardown_seconds", _positive(
            self.teardown_seconds, "teardown_seconds"))
        if not isinstance(self.source_identities, Mapping) or not self.source_identities:
            raise PreparationRefused("preparation must pin original source identities")
        sources = {_text(key, "source name"): _sha(value, "source identity")
                   for key, value in self.source_identities.items()}
        object.__setattr__(self, "frame", _freeze(frame))
        object.__setattr__(self, "source_identities", _freeze(sources))

    @classmethod
    def from_dict(cls, value: Any) -> ServingPreparationDeclaration:
        row = _exact(value, set(cls.__dataclass_fields__), "preparation declaration")
        row["statistics"] = ServingStatisticsDeclaration.from_dict(row["statistics"])
        row["aa_pair"] = PreparationArmPair.from_dict(row["aa_pair"])
        if row["neutral_pair"] is not None:
            row["neutral_pair"] = PreparationArmPair.from_dict(row["neutral_pair"])
        row["resources"] = scheduling.ResourceVector.from_dict(row["resources"])
        row["retry_policy"] = PreparationRetryPolicy.from_dict(row["retry_policy"])
        return cls(**row)

    def to_dict(self) -> dict:
        return {"schema": self.schema, "declaration_id": self.declaration_id,
                "campaign_id": self.campaign_id, "target_revision": self.target_revision,
                "issued_at": self.issued_at, "frame": _plain(self.frame),
                "statistics": self.statistics.to_dict(), "aa_pair": self.aa_pair.to_dict(),
                "neutral_pair": (None if self.neutral_pair is None else
                                 self.neutral_pair.to_dict()),
                "prompt_manifest_digest": self.prompt_manifest_digest,
                "resources": self.resources.to_dict(), "max_stage_seconds": self.max_stage_seconds,
                "teardown_seconds": self.teardown_seconds,
                "source_identities": _plain(self.source_identities),
                "retry_policy": self.retry_policy.to_dict()}

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def preparation_debt(self) -> tuple[str, ...]:
        return (() if self.neutral_pair is not None else ("neutral_material_unavailable",))


@dataclass(frozen=True)
class CalibrationPreparationRequest:
    """One bounded process-pair chunk of the prospectively declared raw pool.

Each statistical block has precisely one independent native process per arm.
Prompt observations within that process are not extra statistical repetitions.
The pool's block/material names are explicit inputs; order is rederived through
the existing owning OrderSchedule, never chosen after seeing measurements.
"""

    declaration: ServingPreparationDeclaration
    kind: str
    plan: ep.ExperimentPlan
    prompts: ps.FrozenPromptManifest
    block_membership: tuple[Mapping[str, Any], ...]
    stage_proposal: scheduling.StageProposal
    attempt_ordinal: int = 0
    schema: str = REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != REQUEST_SCHEMA
                or type(self.declaration) is not ServingPreparationDeclaration
                or self.kind not in {"aa", "neutral"}):
            raise PreparationRefused("calibration request schema/declaration/kind differs")
        declaration = self.declaration
        if (type(self.attempt_ordinal) is not int
                or not 0 <= self.attempt_ordinal < declaration.retry_policy.max_attempts):
            raise PreparationRefused("attempt is outside the original finite retry policy")
        pair = self.pair
        if pair is None:
            raise PreparationRefused("requested neutral material is unavailable")
        if type(self.plan) is not ep.ExperimentPlan:
            raise PreparationRefused("calibration requires the concrete native ExperimentPlan")
        plan = ep.ExperimentPlan.from_dict(self.plan.to_dict())
        object.__setattr__(self, "plan", plan)
        if type(self.prompts) is not ps.FrozenPromptManifest:
            raise PreparationRefused("calibration requires the original frozen prompts")
        prompts = ps.FrozenPromptManifest.from_dict(self.prompts.to_dict())
        object.__setattr__(self, "prompts", prompts)
        if _digest(prompts.to_dict()) != declaration.prompt_manifest_digest:
            raise PreparationRefused("calibration prompt manifest differs from declaration")
        # These are raw preparation observations, not an A2 intervention screen
        # or a strict-search verdict. No new protocol/record enum is invented.
        if (plan.schema != ep.PLAN_SCHEMA_V2 or plan.unit != "process"
                or plan.phase != "observation" or plan.record_class != "observation"
                or plan.intended_use != "explore" or plan.calibration_ref is not None
                or plan.changed_factors or not plan.stopping["paired"]
                or plan.campaign_id != declaration.campaign_id
                or plan.target_revision != declaration.target_revision
                or plan.epoch != declaration.frame["epoch"]
                or plan.metric != declaration.frame["metric"]
                or plan.estimator_id != declaration.frame["estimator_id"]
                or plan.metric_direction != {
                    "higher_better": "higher", "lower_better": "lower"
                }[declaration.frame["metric_direction"]]):
            raise PreparationRefused("native calibration plan differs from original raw frame")
        loaded = _plain(plan.loaded_instrument)
        if (dict(plan.anchor_identity) != ps.arm_identity(pair.anchor.template, pair.anchor,
                                                         loaded_instrument=loaded)
                or dict(plan.candidate_identity) != ps.arm_identity(pair.candidate.template,
                    pair.candidate, loaded_instrument=loaded)):
            raise PreparationRefused("calibration arm identities differ from original recipes")
        if not isinstance(self.block_membership, (tuple, list)) or not self.block_membership:
            raise PreparationRefused("calibration requires explicit independent block membership")
        if declaration.retry_policy.max_attempts > 1 and len(self.block_membership) != 1:
            raise PreparationRefused("retry-enabled requests must isolate one independent pair")
        n = declaration.statistics.controls.calibration_block_count
        schedule = st.OrderSchedule.derive(
            campaign_seed=declaration.statistics.campaign_seed,
            candidate_id=f"{declaration.declaration_id}:{self.kind}", base_blocks=n,
            attempt=self.attempt_ordinal)
        rows = []
        seen_units: set[str] = set()
        seen_material: set[str] = set()
        previous_index = None
        by_id = {unit.unit_id: unit for unit in plan.expected_units}
        for local_index, value in enumerate(self.block_membership):
            row = _exact(value, {"block_index", "material_unit_id", "stratum",
                                 "anchor_unit_id", "candidate_unit_id"}, "block membership")
            block_index = row["block_index"]
            if (type(block_index) is not int or not 0 <= block_index < n
                    or (previous_index is not None and block_index != previous_index + 1)):
                raise PreparationRefused("calibration block indices must be an original bounded chunk")
            previous_index = block_index
            material = _text(row["material_unit_id"], "material_unit_id")
            if material in seen_material:
                raise PreparationRefused("calibration material unit is reused")
            seen_material.add(material)
            if row["stratum"] != declaration.statistics.split_rule.assign(material):
                raise PreparationRefused("calibration material stratum differs from original split")
            expected_order = ("anchor", "candidate") if schedule.order_for(block_index) == \
                st.ORDER_ANCHOR_FIRST else ("candidate", "anchor")
            for arm in ("anchor", "candidate"):
                unit_id = _text(row[f"{arm}_unit_id"], f"{arm}_unit_id")
                unit = by_id.get(unit_id)
                if (unit is None or unit.arm != arm or unit_id in seen_units
                        or unit.pair_id != local_index
                        or unit.order_index != 2 * local_index + expected_order.index(arm)):
                    raise PreparationRefused("native process/block membership or original order differs")
                seen_units.add(unit_id)
            rows.append(_freeze(row))
        if seen_units != set(by_id):
            raise PreparationRefused("block membership must cover every original native process once")
        if type(self.stage_proposal) is not scheduling.StageProposal:
            raise PreparationRefused("calibration requires an explicit scheduler StageProposal")
        stage = scheduling.StageProposal.from_dict(self.stage_proposal.to_dict())
        if (stage.stage_class != "calibration" or stage.target_revision != declaration.target_revision
                or stage.eligibility_ref != declaration.digest
                or stage.backend != declaration.frame["backend"]
                or stage.estimated_claims != declaration.resources
                or stage.estimated_duration_seconds > declaration.max_stage_seconds
                or stage.reservation_kind not in (None, "calibration")):
            raise PreparationRefused("calibration scheduler proposal differs from original resource budget")
        object.__setattr__(self, "stage_proposal", stage)
        object.__setattr__(self, "block_membership", tuple(rows))

    @property
    def pair(self) -> PreparationArmPair | None:
        return self.declaration.aa_pair if self.kind == "aa" else self.declaration.neutral_pair

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def chunk_identity(self) -> str:
        """Original logical membership, intentionally independent of retry process IDs."""
        return _digest({"declaration": self.declaration.digest, "kind": self.kind,
                        "membership": self.logical_membership})

    @property
    def logical_membership(self) -> list[dict]:
        return [{key: row[key] for key in ("block_index", "material_unit_id", "stratum")}
                for row in self.block_membership]

    @classmethod
    def from_dict(cls, value: Any) -> CalibrationPreparationRequest:
        row = _exact(value, {"schema", "declaration", "kind", "plan", "prompt_manifest",
                             "block_membership", "stage_proposal", "attempt_ordinal"},
                     "calibration request")
        return cls(ServingPreparationDeclaration.from_dict(row["declaration"]), row["kind"],
                   ep.ExperimentPlan.from_dict(row["plan"]),
                   ps.FrozenPromptManifest.from_dict(row["prompt_manifest"]),
                   tuple(row["block_membership"]),
                   scheduling.StageProposal.from_dict(row["stage_proposal"]),
                   row["attempt_ordinal"], row["schema"])

    def to_dict(self) -> dict:
        return {"schema": self.schema, "declaration": self.declaration.to_dict(),
                "kind": self.kind, "plan": self.plan.to_dict(),
                "attempt_ordinal": self.attempt_ordinal,
                "prompt_manifest": self.prompts.to_dict(),
                "block_membership": _plain(self.block_membership),
                "stage_proposal": self.stage_proposal.to_dict()}


@dataclass(frozen=True)
class CalibrationPreparationDispatch:
    request: CalibrationPreparationRequest
    selection: scheduling.Selection
    execution_authorized: bool = False
    schema: str = DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != DISPATCH_SCHEMA or self.execution_authorized is not False
                or type(self.request) is not CalibrationPreparationRequest
                or type(self.selection) is not scheduling.Selection
                or self.selection.status != "selected"
                or self.selection.proposal != self.request.stage_proposal):
            raise PreparationRefused("calibration dispatch is not exact selected preparation advice")

    @classmethod
    def from_dict(cls, value: Any) -> CalibrationPreparationDispatch:
        row = _exact(value, {"schema", "request", "selection", "execution_authorized"},
                     "calibration dispatch")
        return cls(CalibrationPreparationRequest.from_dict(row["request"]),
                   scheduling.Selection.from_dict(row["selection"]),
                   row["execution_authorized"], row["schema"])

    def to_dict(self) -> dict:
        return {"schema": self.schema, "request": self.request.to_dict(),
                "selection": self.selection.to_dict(), "execution_authorized": False}


def source_identity() -> Mapping[str, Any]:
    """Loaded preparation admission implementations, pinned before native issue."""
    from . import lifecycle_observation as lo, unified_worker as worker, unified_driver as driver
    from . import serving_preparation_startup as startup
    from . import standalone_inputs
    from ..evaluator import controls
    return _freeze({"schema": "epyc.autokernel.serving_preparation_source.v1",
        "schemas": [DECLARATION_SCHEMA, STATISTICS_SCHEMA, PAIR_SCHEMA,
                    REQUEST_SCHEMA, DISPATCH_SCHEMA, worker.PREPARED_SCHEMA_V3,
                    startup.SCHEMA, standalone_inputs.PREPARATION_MANIFEST_SCHEMA],
        "startup_protocol_fields": sorted(startup.PROTOCOL_FIELDS),
        "max_configuration_bytes": MAX_CONFIGURATION_BYTES,
        "callables": [lo.callable_identity(item) for item in (
            _plain, _bytes, _digest, _freeze, _exact, _text, _sha, _instant, _positive,
            PreparationArmPair.__post_init__, PreparationArmPair.from_dict.__func__,
            PreparationArmPair.to_dict, ServingStatisticsDeclaration.__post_init__,
            ServingStatisticsDeclaration.from_dict.__func__, ServingStatisticsDeclaration.to_dict,
            PreparationRetryPolicy.__post_init__, PreparationRetryPolicy.from_dict.__func__,
            PreparationRetryPolicy.to_dict,
            ServingPreparationDeclaration.__post_init__,
            ServingPreparationDeclaration.from_dict.__func__, ServingPreparationDeclaration.to_dict,
            CalibrationPreparationRequest.__post_init__, CalibrationPreparationRequest.from_dict.__func__,
            CalibrationPreparationRequest.to_dict, CalibrationPreparationDispatch.__post_init__,
            CalibrationPreparationRequest.chunk_identity.fget,
            CalibrationPreparationRequest.logical_membership.fget,
            CalibrationPreparationDispatch.from_dict.__func__, CalibrationPreparationDispatch.to_dict,
            worker.PreparedPlannedServingStage.from_dict.__func__,
            worker.PreparedPlannedServingStage.native_observed.fget,
            driver.UnifiedCampaignDriver.materialize_calibration, bounded_requests,
            validate_pool_membership, preparation_disposition,
            startup.PreparationStartupConfiguration.from_dict.__func__,
            startup.PreparationStartupConfiguration.to_dict, startup.materialize,
            standalone_inputs._shared_native_settings,
            controls.run_calibration_block, st.solve_calibration,
            CollectedCalibrationReference.__post_init__,
            CollectedCalibrationReference.from_dict.__func__, CollectedCalibrationReference.to_dict,
            InstalledServingPreparationOwner.__init__, InstalledServingPreparationOwner._own,
            InstalledServingPreparationOwner.recover, InstalledServingPreparationOwner.pending_requests,
            InstalledServingPreparationOwner.disposition, InstalledServingPreparationOwner.accept,
            InstalledServingPreparationOwner._collect_original,
            InstalledServingPreparationOwner.reopen_chunk,
            InstalledServingPreparationOwner.refresh_settled,
            InstalledServingPreparationOwner.collection_outcome,
            InstalledServingPreparationOwner._materialize_pool,
            InstalledServingPreparationOwner._solve_body,
            InstalledServingPreparationOwner.solve_collected, InstalledServingPreparationOwner.reopen_solve,
            InstalledServingPreparationOwner.close,
            source_identity)]})


def bounded_requests(values: Any, *, max_requests: int) -> tuple[CalibrationPreparationRequest, ...]:
    """Reject campaign-sized excess before parsing any individual request."""
    if type(max_requests) is not int or max_requests <= 0:
        raise PreparationRefused("request bound requires the positive campaign attempt cap")
    if not isinstance(values, (tuple, list)) or len(values) > max_requests:
        raise PreparationRefused("calibration request count exceeds campaign attempt budget")
    rows = []
    size = 0
    for value in values:
        row = value.to_dict() if type(value) is CalibrationPreparationRequest else value
        size += len(_bytes(row))
        if size > MAX_CONFIGURATION_BYTES:
            raise PreparationRefused("calibration request bytes exceed driver configuration bound")
        rows.append(row)
    return tuple(CalibrationPreparationRequest.from_dict(row) for row in rows)


def validate_pool_membership(requests: tuple[CalibrationPreparationRequest, ...]) -> None:
    """Check original pool coverage globally, including every future chunk."""
    declarations = {}
    indices, materials, processes, groups, stage_ids = {}, set(), set(), {}, set()
    for request in requests:
        declaration = request.declaration
        prior = declarations.setdefault(declaration.declaration_id, declaration)
        if prior.digest != declaration.digest:
            raise PreparationRefused("one calibration declaration has conflicting original bytes")
        key = (declaration.digest, request.kind)
        slots = indices.setdefault(key, set())
        attempts = groups.setdefault(request.chunk_identity, {})
        if request.attempt_ordinal in attempts:
            raise PreparationRefused("one original chunk repeats an attempt ordinal")
        attempts[request.attempt_ordinal] = request
        if request.attempt_ordinal == 0:
            for member in request.block_membership:
                if member["block_index"] in slots:
                    raise PreparationRefused("original calibration block is repeated across chunks")
                slots.add(member["block_index"])
                material = (declaration.digest, member["material_unit_id"])
                if material in materials:
                    raise PreparationRefused("original material is reused across calibration pools/chunks")
                materials.add(material)
        if request.stage_proposal.proposal_id in stage_ids:
            raise PreparationRefused("original attempt scheduler identity is reused")
        stage_ids.add(request.stage_proposal.proposal_id)
        for unit in request.plan.expected_units:
            if unit.process_id in processes:
                raise PreparationRefused("independent process is reused across calibration chunks")
            processes.add(unit.process_id)
    for attempts in groups.values():
        original = attempts.get(0)
        if original is None or set(attempts) != set(range(original.declaration.retry_policy.max_attempts)):
            raise PreparationRefused("all bounded retry attempts must be issued prospectively")
        original_stage = original.stage_proposal.to_dict()
        original_stage.pop("proposal_id")
        for attempt in attempts.values():
            stage = attempt.stage_proposal.to_dict()
            stage.pop("proposal_id")
            if stage != original_stage:
                raise PreparationRefused("retry changed the original scheduling budget or eligibility")
    for declaration in declarations.values():
        kinds = ("aa", "neutral") if declaration.neutral_pair is not None else ("aa",)
        for kind in kinds:
            if indices.get((declaration.digest, kind)) != set(range(
                    declaration.statistics.controls.calibration_block_count)):
                raise PreparationRefused("original declaration pool has missing calibration blocks")


def preparation_disposition(requests: tuple[CalibrationPreparationRequest, ...],
                            settlements: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Pure finite retry reduction; the installed owner supplies verified Journal facts.

    These digests are advice, not execution or publication authority. Only a
    successfully settled original block is retained; unissued retries cannot
    replace it. The request constructors already verify each retry's order.
    """
    groups = {}
    for request in requests:
        groups.setdefault(request.chunk_identity, []).append(request)
    pending, collected, exhausted = [], [], []
    for chunk_id, group in groups.items():
        finished = False
        for request in sorted(group, key=lambda item: item.attempt_ordinal):
            settlement = settlements.get(request.digest)
            if finished:
                if settlement is not None:
                    raise PreparationRefused("later retry was settled without an eligible predecessor")
                continue
            if settlement is None:
                pending.append(request.digest)
                finished = True
            elif settlement["outcome"] == "calibration":
                collected.append(request.digest)
                finished = True
            elif settlement["outcome"] not in ("failed", "invalid"):
                raise PreparationRefused("calibration has an unsupported original settlement")
        if not finished:
            exhausted.append(chunk_id)
    return _freeze({"pending": pending, "collected": collected, "exhausted": exhausted})


@dataclass(frozen=True)
class CollectedCalibrationReference:
    """A raw collection/diagnostic solve reference, explicitly not qualification."""

    declaration_digest: str
    locator: str
    sha256: str
    schema: str = "epyc.autokernel.collected_calibration_reference.v1"

    def __post_init__(self) -> None:
        if self.schema != "epyc.autokernel.collected_calibration_reference.v1":
            raise PreparationRefused("collected calibration reference schema differs")
        _sha(self.declaration_digest, "declaration_digest")
        _sha(self.sha256, "collected artifact digest")
        if (not isinstance(self.locator, str) or not self.locator.endswith(".json")
                or self.locator.startswith(".") or "/" in self.locator):
            raise PreparationRefused("collected reference requires a private artifact leaf")

    @classmethod
    def from_dict(cls, value: Any) -> CollectedCalibrationReference:
        return cls(**_exact(value, set(cls.__dataclass_fields__), "collected reference"))

    def to_dict(self) -> dict:
        return asdict(self)


class InstalledServingPreparationOwner:
    """Same-runtime-thread raw-pool owner; Journal settlement is its only ledger.

    It never upgrades unknown native units, mints a control panel, or issues an
    original qualified calibration. The existing numeric owner may be invoked
    diagnostically once all declared uncontaminated raw blocks were collected.
    """

    def __init__(self, *, controller: Any, requests: tuple[CalibrationPreparationRequest, ...]):
        from .campaign_control import CampaignController
        if type(controller) is not CampaignController:
            raise PreparationRefused("preparation requires its actual campaign controller")
        self.controller = controller
        self.requests = bounded_requests(requests,
            max_requests=controller._scheduler_engine.config.campaign_attempt_cap)
        validate_pool_membership(self.requests)
        self._by_digest = {request.digest: request for request in self.requests}
        self._settled: dict[str, Mapping[str, Any]] = {}
        self._chunks: dict[str, Mapping[str, Any]] = {}
        self._solves: dict[str, CollectedCalibrationReference] = {}
        self._thread: int | None = None
        self._store = None
        self._recovered = False
        self._closed = False

    def _own(self) -> None:
        import threading
        from .measurement_capture import ArtifactStore
        if self._closed:
            raise PreparationRefused("preparation owner is closed")
        current = threading.get_ident()
        if self._thread is None:
            store = ArtifactStore(self.controller.store / "unified-native-artifacts")
            self._store, self._thread = store, current
        if self._thread != current:
            raise PreparationRefused("preparation store must stay on its actual runtime thread")

    def recover(self) -> None:
        self._own()
        if self._recovered:
            return
        history = self.controller.unified_driver_preparation_history(self.requests)
        settled, chunks = {}, {}
        for row in history["records"]:
            request_digest = row["request_digest"]
            settlement = row["settlement"]
            if settlement is None:
                continue
            if request_digest in settled:
                raise PreparationRefused("one original calibration chunk was settled more than once")
            settled[request_digest] = settlement
            references = [item for item in settlement["terminal_refs"]
                          if item.startswith("calibration-collected:")]
            if settlement["outcome"] in ("calibration", "invalid"):
                if len(references) != 1:
                    raise PreparationRefused("raw calibration settlement lacks its exact collection artifact")
                reference = CollectedCalibrationReference.from_dict(json.loads(
                    references[0].removeprefix("calibration-collected:")))
                chunk = self.reopen_chunk(reference,
                    request=self._by_digest[request_digest])
                expected = ("collected_unqualified" if settlement["outcome"] == "calibration"
                            else "contaminated")
                if chunk["raw_status"] != expected:
                    raise PreparationRefused("collection disposition differs from exact settlement")
                chunks[request_digest] = chunk
            elif settlement["outcome"] != "failed" or references:
                raise PreparationRefused("failed preparation settlement has unsupported collection evidence")
        preparation_disposition(self.requests, settled)
        self._settled, self._chunks = settled, chunks
        self._recovered = True

    def pending_requests(self) -> tuple[CalibrationPreparationRequest, ...]:
        self.recover()
        pending = set(preparation_disposition(self.requests, self._settled)["pending"])
        return tuple(item for item in self.requests if item.digest in pending)

    def disposition(self) -> Mapping[str, Any]:
        self.refresh_settled()
        return preparation_disposition(self.requests, self._settled)

    def accept(self, *, prepared: Any, start: Any, terminal: Any, reference: Any,
               fence: Any) -> CollectedCalibrationReference:
        """Reopen original native output AFTER its real controller capture commit."""
        from . import unified_worker as worker
        self.recover()
        prepared = worker.PreparedPlannedServingStage.from_dict(prepared.to_dict())
        if prepared.schema != worker.PREPARED_SCHEMA_V3:
            raise PreparationRefused("raw preparation owner requires exact prepared v3")
        request = CalibrationPreparationDispatch.from_dict(prepared.dispatch).request
        if request.digest not in self._by_digest:
            raise PreparationRefused("completed preparation was not prospectively installed")
        result, _captures = worker.reopen_deferred_result(reference, prepared=prepared,
            start=start, terminal=terminal, fence=fence)
        body = self._collect_original(request=request, result=result, reference=reference,
                                      prepared_digest=prepared.prepared_digest)
        stored = self._store.write(f"calibration-chunk:{request.digest}", body)
        # Publication can fail after this write. Only recover() may admit a
        # chunk to the pool, after the exact successful Journal settlement.
        return CollectedCalibrationReference(request.declaration.digest, stored.locator, stored.sha256)

    def _collect_original(self, *, request: CalibrationPreparationRequest, result: Any,
                          reference: Any, prepared_digest: str) -> dict:
        if (result.body["plan_digest"] != request.plan.digest
                or result.body["prepared_digest"] != prepared_digest
                or reference.prepared_digest != prepared_digest
                or result.result_digest != reference.result_digest):
            raise PreparationRefused("original raw result differs from installed preparation")
        documents = {}
        capture_refs = []
        for capture in result.body["captures"]:
            measurement_id, payload = capture["measurement_id"], capture["payload"]
            committed = self.controller.native_capture(measurement_id)
            if committed is None or _plain(committed.payload) != _plain(payload):
                raise PreparationRefused("raw collection must follow actual native publication")
            carrier = payload["carrier"]
            self._store.verify(f"carrier:{measurement_id}", _plain(carrier))
            capture_refs.append({"measurement_id": measurement_id,
                                 "artifact": _plain(payload["artifact"])})
            for item in carrier["raw_artifacts"]:
                document, stored = item["document"], item["stored"]
                if document.get("kind") != "completed_attempt":
                    continue
                reopened = self._store.read(stored["locator"], stored["sha256"])
                if _plain(reopened) != _plain(document) or document["unit_id"] in documents:
                    raise PreparationRefused("original completed-attempt artifact changed or repeats")
                documents[document["unit_id"]] = _plain(document)
        raws = tuple(ep.RawUnit.from_dict(item) for item in result.body["run"]["raw_units"])
        expected = {unit.unit_id: unit for unit in request.plan.expected_units}
        if set(documents) != set(expected) or {raw.unit_id for raw in raws} != set(expected):
            raise PreparationRefused("raw calibration chunk is incomplete; no failed block is replaced")
        observations = []
        for raw in raws:
            unit, document = expected[raw.unit_id], documents[raw.unit_id]
            if (not raw.terminal or raw.arm != unit.arm or raw.process_id != unit.process_id
                    or raw.observed_order_index != unit.order_index
                    or raw.artifact_digest != document["artifact_digest"]
                    or raw.value != document["value"]):
                raise PreparationRefused("raw calibration value/order differs from original process artifact")
            started = _instant(document["observed_started_at"], "observed_started_at")
            ended = _instant(document["observed_ended_at"], "observed_ended_at")
            if started < _instant(request.declaration.issued_at, "declaration issue") or ended < started:
                raise PreparationRefused("raw calibration predates declaration or has reversed time")
            observations.append({"raw_unit": raw.to_dict(),
                "observed_started_at": document["observed_started_at"],
                "observed_ended_at": document["observed_ended_at"]})
        contaminated = any(witness.status == "fail" for raw in raws for witness in raw.witnesses.values())
        body = {"schema": "epyc.autokernel.collected_calibration_chunk.v1",
            "declaration_digest": request.declaration.digest, "request_digest": request.digest,
            "prepared_digest": prepared_digest, "plan_digest": request.plan.digest,
            "result_reference": reference.to_dict(), "capture_refs": capture_refs,
            "block_membership": _plain(request.block_membership), "observations": observations,
            "raw_status": "contaminated" if contaminated else "collected_unqualified",
            "qualification": "unavailable", "qualification_debt": [
                "original_window_admissibility_unavailable", "original_control_panel_unavailable",
                "frame_phase_cell_scope_unverified"],
            "ranking_authorized": False}
        return body

    def reopen_chunk(self, reference: CollectedCalibrationReference, *,
                     request: CalibrationPreparationRequest) -> Mapping[str, Any]:
        from . import unified_worker as worker
        self._own()
        reference = CollectedCalibrationReference.from_dict(reference.to_dict())
        body = self._store.read(reference.locator, reference.sha256)
        if (body.get("schema") != "epyc.autokernel.collected_calibration_chunk.v1"
                or reference.declaration_digest != request.declaration.digest
                or body.get("declaration_digest") != request.declaration.digest
                or body.get("request_digest") != request.digest
                or body.get("plan_digest") != request.plan.digest
                or _plain(body.get("block_membership")) != _plain(request.block_membership)
                or body.get("qualification") != "unavailable"
                or body.get("ranking_authorized") is not False):
            raise PreparationRefused("retained raw calibration differs from original declaration")
        self._store.verify(f"calibration-chunk:{request.digest}", _plain(body))
        original_ref = worker.PlannedWorkerResultReference.from_dict(body["result_reference"])
        original = worker.PlannedWorkerResult.from_dict(self._store.read(
            original_ref.result_locator, original_ref.result_sha256))
        self._store.verify(f"planned-worker-result:{original_ref.prepared_digest}:{original_ref.nonce}",
                           original.to_dict())
        rederived = self._collect_original(request=request, result=original, reference=original_ref,
                                          prepared_digest=body["prepared_digest"])
        if _plain(body) != rederived:
            raise PreparationRefused("retained raw chunk differs from its original producer facts")
        return _freeze(body)

    def refresh_settled(self) -> None:
        self._own()
        self._recovered = False
        self.recover()

    def collection_outcome(self, reference: CollectedCalibrationReference, *,
                           request: CalibrationPreparationRequest) -> str:
        chunk = self.reopen_chunk(reference, request=request)
        return "calibration" if chunk["raw_status"] == "collected_unqualified" else "invalid"

    def _materialize_pool(self, declaration: ServingPreparationDeclaration) -> tuple | None:
        """Reconstruct the exact winning original blocks, not stored copied vectors."""
        originals = tuple(item for item in self.requests if item.declaration.digest == declaration.digest)
        disposition = preparation_disposition(originals, self._settled)
        if (not originals or declaration.neutral_pair is None or disposition["pending"]
                or disposition["exhausted"]):
            return None
        requests = tuple(self._by_digest[digest] for digest in disposition["collected"])
        pools: dict[str, list[st.PairedBlock]] = {"aa": [], "neutral": []}
        sources = []
        for request in requests:
            chunk = self._chunks[request.digest]
            if chunk["raw_status"] != "collected_unqualified":
                raise PreparationRefused("successful settlement contains a contaminated raw block")
            stored = self._store.verify(f"calibration-chunk:{request.digest}", _plain(chunk))
            reference = CollectedCalibrationReference(declaration.digest, stored.locator, stored.sha256)
            chunk = self.reopen_chunk(reference, request=request)
            sources.append({"request_digest": request.digest, "reference": reference.to_dict(),
                            "settlement_digest": _digest(self._settled[request.digest])})
            observations = {row["raw_unit"]["unit_id"]: row for row in chunk["observations"]}
            schedule = st.OrderSchedule.derive(campaign_seed=declaration.statistics.campaign_seed,
                candidate_id=f"{declaration.declaration_id}:{request.kind}",
                base_blocks=declaration.statistics.controls.calibration_block_count,
                attempt=request.attempt_ordinal)
            for member in request.block_membership:
                anchor = observations[member["anchor_unit_id"]]
                candidate = observations[member["candidate_unit_id"]]
                measured_at = max((anchor["observed_ended_at"], candidate["observed_ended_at"]),
                                  key=lambda value: _instant(value, "original observed end"))
                pools[request.kind].append(st.PairedBlock(
                    block_index=member["block_index"], unit_id=member["material_unit_id"],
                    stratum=member["stratum"], order=schedule.order_for(member["block_index"]),
                    anchor_samples=(anchor["raw_unit"]["value"],),
                    candidate_samples=(candidate["raw_unit"]["value"],),
                    measured_at=measured_at))
        for blocks in pools.values():
            blocks.sort(key=lambda block: block.block_index)
        material = {"schema": "epyc.autokernel.original_calibration_raw_pool.v1",
            "declaration": declaration.to_dict(), "chunks": sources,
            "aa_blocks": [block.to_list() for block in pools["aa"]],
            "neutral_blocks": [block.to_list() for block in pools["neutral"]]}
        return material, pools

    def solve_collected(self, declaration: ServingPreparationDeclaration) \
            -> CollectedCalibrationReference | None:
        """Run the owning numeric solve after exact successful durable settlements.

        A restart deterministically rederives the same diagnostic from retained
        original blocks; it neither recollects samples nor creates a second
        accounting event. This is never qualified calibration/control authority.
        """
        self.refresh_settled()
        prior = self._solves.get(declaration.digest)
        if prior is not None:
            self.reopen_solve(prior, declaration=declaration)
            return prior
        reconstructed = self._materialize_pool(declaration)
        if reconstructed is None:
            return None
        material, pools = reconstructed
        raw = self._store.write(f"calibration-raw-pool:{declaration.digest}", material)
        body = self._solve_body(declaration, raw, pools)
        sealed = self._store.write(f"calibration-solve:{declaration.digest}", body)
        reference = CollectedCalibrationReference(declaration.digest, sealed.locator, sealed.sha256)
        self._solves[declaration.digest] = reference
        return reference

    @staticmethod
    def _solve_body(declaration: ServingPreparationDeclaration, raw: Any,
                    pools: Mapping[str, list[st.PairedBlock]]) -> dict:
        """Pure owning numeric derivation shared by first solve and retained replay."""
        from ..evaluator import controls
        statistical = declaration.statistics
        try:
            inputs = st.CalibrationInputs(
                backend=declaration.frame["backend"], phase=declaration.frame["phase"],
                cell_class=declaration.frame["cell_class"], campaign_seed=statistical.campaign_seed,
                controls=statistical.controls, stopping_rule=statistical.stopping_rule,
                construction=st.select_construction(statistical.construction_id),
                effect_scale=statistical.effect_scale, metric_direction=declaration.frame["metric_direction"],
                hypothesis=statistical.hypothesis, margin=statistical.margin,
                aa_blocks=tuple(pools["aa"]), neutral_blocks=tuple(pools["neutral"]),
                anchor_calibration_values=tuple(block.anchor_samples[0] for block in pools["aa"]),
                samples_ref=f"{raw.locator}#sha256={raw.sha256}", owning_rep_rule=statistical.owning_rep_rule)
            solve = controls.run_calibration_block(inputs)
            numeric, reasons = solve.to_dict(), []
        except st.StatisticsError as exc:
            numeric, reasons = None, [f"{type(exc).__name__}: {exc}"]
        return {"schema": "epyc.autokernel.collected_calibration_solve.v1",
            "declaration_digest": declaration.digest, "original_material": raw.to_dict(),
            "numeric_solve": numeric, "numeric_reasons": reasons, "qualification": "unavailable",
            "qualification_debt": ["original_window_admissibility_unavailable",
                                   "original_control_panel_unavailable", "frame_phase_cell_scope_unverified"],
            "ranking_authorized": False}

    def reopen_solve(self, reference: CollectedCalibrationReference, *,
                     declaration: ServingPreparationDeclaration) -> Mapping[str, Any]:
        self.refresh_settled()
        reference = CollectedCalibrationReference.from_dict(reference.to_dict())
        body = self._store.read(reference.locator, reference.sha256)
        if (reference.declaration_digest != declaration.digest
                or body.get("schema") != "epyc.autokernel.collected_calibration_solve.v1"
                or body.get("declaration_digest") != declaration.digest
                or body.get("qualification") != "unavailable"
                or body.get("ranking_authorized") is not False):
            raise PreparationRefused("diagnostic solve cannot become qualified calibration authority")
        self._store.verify(f"calibration-solve:{declaration.digest}", _plain(body))
        raw = body["original_material"]
        material = self._store.read(raw["locator"], raw["sha256"])
        expected = self._materialize_pool(declaration)
        if expected is None or _plain(material) != expected[0]:
            raise PreparationRefused("diagnostic solve differs from successful original settled blocks")
        raw = self._store.verify(f"calibration-raw-pool:{declaration.digest}", _plain(material))
        if _plain(body) != self._solve_body(declaration, raw, expected[1]):
            raise PreparationRefused("retained numeric solve differs from the original owning derivation")
        return _freeze(body)

    def close(self) -> None:
        if self._store is not None:
            self._own()
            self._store.close()
            self._store = None
        self._closed = True
