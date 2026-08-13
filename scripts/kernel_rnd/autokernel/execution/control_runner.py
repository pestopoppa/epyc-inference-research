#!/usr/bin/env python3
"""control_runner.py — the implementation of `evaluator.controls.ControlRunner`.

WHY THIS MODULE EXISTS
----------------------
`evaluator/controls.py` evaluates the five fixed controls of P-AK-SEARCH-1 §15.2
and says so plainly: *"It runs NO control."* `ControlRunner` was the Protocol it
observed them through, and it had no implementation anywhere in the tree. Every
control result in every test was a `ControlObservation` handed over by a fixture
object. The harness was complete and the panel was fiction.

The five controls are what make the evaluator trustworthy at all — four test its
ability to REJECT, the fifth its ability to ACCEPT — so an unimplemented runner
means the gate has never been checked against anything. This module runs them.

WHAT IT GUARANTEES, AND HOW
---------------------------
1. **One code path.** *"A control that runs down a different code path proves
   nothing about the path that matters."* There is exactly ONE thing a control
   can be handed to — `CandidatePipeline.evaluate(CandidateSubmission)` — and it
   is the same call a candidate makes with the same argument type.
   `CandidateSubmission` carries no field naming a control, no field marking one,
   and no `EffectEstimate`: the pipeline reduces the raw blocks itself, so a
   control cannot supply the number it is about to be scored on any more than a
   candidate can. `audit_single_evaluation_path()` proves from this module's own
   AST that `ExecutedControlRunner` reaches the evaluator through that one call
   and no other.

2. **Seed rotation, wired.** `derive_control_seed`, `ControlBundle.seed_for` and
   `SeedRotationSchedule.check_rotation` were declared, hashed into the campaign
   digest, and had NO CALLER — `run_all` handed all five controls one seed and a
   campaign could run for its whole life on one holdout. `ControlSweep` is the
   caller: it checks the rotation schedule BEFORE the sweep and refuses to
   produce a panel when the schedule says rotate and the recorded epoch has not
   moved, and the derived per-control seed reaches the measurement material as
   `PairedBlock.unit_id`.

   Read the limit of that honestly. A rotated seed changes the unit id a block is
   LABELLED with; with a static fixture it does not change one sample. The two
   epochs are byte-identical measurements under different labels, and
   `calibration_material` refuses to pool them (`CalibrationMaterialRelabelled`)
   for that reason. Rotation becomes real when the material behind the fixture is
   re-measured per epoch — which is the first real control run's job, not this
   module's.

3. **A panel that carries proof it was run.** Closed in `controls.py` itself
   (`ControlPanelForged`), because that is where the object lives: the result
   re-derives its outcomes from the observations it carries, and an observation
   that ran carries an `api.Verdict`, which only `api.compute_verdict()` mints.

4. **The A/A arm reaches the calibration solve.** `calibration_material()` and
   `build_calibration_inputs()` emit exactly the `statistics.PairedBlock` tuples
   `statistics.CalibrationInputs` consumes, taken from the blocks the A/A and
   neutral controls were ACTUALLY measured on — not a parallel set assembled for
   the solver. φ is *"estimated from the A/A control"*, so a φ estimated from
   material the A/A control did not run on is a φ estimated from nothing.

WHAT IT DOES NOT DO
-------------------
It launches no process and writes no file. It is the layer BELOW the evaluator
and ABOVE the executors: it assembles submissions from recorded or freshly
measured material and hands them to a pipeline. Building a candidate and running
a benchmark are `execution/`'s other modules, under a held claim; this module
never calls them directly, which is why it can be unit-tested tonight against
fixtures on a contended host and run unchanged tomorrow against real material.

`P-AK-SEARCH-1` denial 8 binds everything here: *"no inference run OUTSIDE A HELD
CLAIM."* This module holds no claim and acquires none — it cannot, because it
runs nothing. The pipeline it is handed is what must hold one.
"""
from __future__ import annotations

MODULE_ID = "autokernel.execution.control_runner/v1"

import ast
import json
from dataclasses import dataclass, fields as dataclass_fields, replace
from math import isclose, isfinite
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Optional, Protocol, Sequence

from .. import schemas
from ..evaluator import api, controls
from ..evaluator import statistics as ak_statistics

__all__ = [
    "CONTROL_RUNNER_ID", "SUBMISSION_KIND",
    # errors
    "ControlExecutionError", "PipelineNotWired", "FixtureNotDeclared",
    "FixtureBundleDrift", "CalibrationMaterialMissing",
    "CalibrationMaterialRelabelled", "SweepNotLicensed", "WindowBindingStale",
    "RotationLedgerViolation",
    # the one submission type and the one seam
    "CandidateSubmission", "CandidatePipeline", "DispatchPipeline",
    # campaign bindings and fixtures
    "CampaignBinding", "ControlFixture", "ControlFixtureSet", "resolve_fixture_set",
    # the runner and the sweep
    "ExecutedControlRunner", "SeedAssignment", "SweepResult", "ControlSweep",
    # the calibration join
    "calibration_material", "pool_calibration_material", "build_calibration_inputs",
    # prospective live-campaign evaluation records
    "LiveEvaluationAuthority", "load_live_evaluation_authority",
    "reduce_live_blocks", "attach_belief_capture",
    # audit
    "audit_single_evaluation_path",
    "audit_submission_carries_no_control_marker",
]


@dataclass(frozen=True)
class LiveEvaluationAuthority:
    """Typed, verified authority projected from one accepted live-control bundle.

    The campaign driver consumes this object; it never copies calibration or
    control values into its own constants.  The bundle remains the evidence
    source and all five control outcomes remain three-valued ``Check`` objects.
    """

    campaign_controls: api.CampaignControls
    calibration: api.CalibrationOutputs
    controls: api.ControlPanel
    aa_cadence: schemas.Check
    control_definitions_immutable: schemas.Check
    construction_id: str
    stopping_rule_id: str
    mde: float
    runtime_source_label_ref: str
    evidence_ref: str
    #: Stable arm/recipe facts for which the A/A dispersion was actually
    #: measured.  Per-run seeds are deliberately absent: they identify a run,
    #: not the cell whose noise the calibration licenses.
    calibration_frame: Optional[Mapping[str, Any]] = None


def _recorded_check(value: Any, label: str) -> schemas.Check:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a recorded check mapping")
    outcome = value.get("outcome")
    reasons = value.get("reasons", ())
    if not isinstance(reasons, list):
        raise ValueError(f"{label}.reasons must be a list")
    return schemas.Check(outcome, tuple(str(reason) for reason in reasons))


def load_live_evaluation_authority(path: str | Path) -> LiveEvaluationAuthority:
    """Read the accepted calibration/control records used by the live writer.

    This is deliberately stricter than a display loader: any missing control,
    rejected calibration, or mismatched construction refuses the campaign.
    """
    root = Path(path).resolve()

    def read(name: str) -> Mapping[str, Any]:
        try:
            value = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"live evaluation authority {name}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"live evaluation authority {name} must be an object")
        return value

    declaration = read("campaign_declaration.json")
    calibration_record = read("calibration.json")
    sweep = read("control_sweep.json")
    source = read("runtime-source-label.json")
    inputs = calibration_record.get("inputs")
    outputs = calibration_record.get("outputs")
    attempts = calibration_record.get("attempts")
    panel_result = sweep.get("panel_result")
    if not calibration_record.get("accepted") or not isinstance(inputs, Mapping) \
            or not isinstance(outputs, Mapping) or not outputs.get("accepted"):
        raise ValueError("live evaluation authority carries no accepted calibration")
    if not isinstance(attempts, list):
        raise ValueError("live evaluation authority carries no calibration attempts")
    accepted = [item for item in attempts
                if isinstance(item, Mapping) and item.get("accepted")]
    if len(accepted) != 1 or not isinstance(accepted[0].get("mde"), Mapping) \
            or not accepted[0]["mde"].get("found"):
        raise ValueError("live evaluation authority needs exactly one solved MDE")
    if not isinstance(panel_result, Mapping) or not panel_result.get("may_rank"):
        raise ValueError("live evaluation authority control panel does not authorize ranking")
    panel = panel_result.get("panel")
    if not isinstance(panel, Mapping) or panel.get("marker") != "5/5":
        raise ValueError("live evaluation authority requires the recorded five-control panel")

    frame = declaration.get("calibration_frame")
    expected_frame_keys = {
        "recipe_id", "prompt_tokens", "reps", "candidate_ggml_iqk",
        "anchor_ggml_iqk",
    }
    if not isinstance(frame, Mapping) or set(frame) != expected_frame_keys:
        raise ValueError(
            "live evaluation authority lacks the exact calibration_frame required "
            "to bind its A/A noise to a prospective campaign")
    if frame.get("recipe_id") != declaration.get("recipe_id") \
            or not isinstance(frame.get("prompt_tokens"), int) \
            or not isinstance(frame.get("reps"), int) \
            or frame["prompt_tokens"] < 1 or frame["reps"] < 1 \
            or frame.get("candidate_ggml_iqk") not in ("0", "1") \
            or frame.get("anchor_ggml_iqk") not in ("0", "1"):
        raise ValueError("live evaluation authority has an invalid calibration_frame")

    controls_obj = inputs.get("controls")
    if not isinstance(controls_obj, Mapping):
        raise ValueError("live evaluation authority has no campaign controls")
    campaign_controls = api.CampaignControls(**{
        name: controls_obj[name] for name in (
            "calibration_block_count", "contribution_floor", "max_candidates",
            "confirmation_admission_count", "max_blocks_per_candidate",
            "storage_floor_bytes_free")
    })
    calibration_values = {
        name: outputs[name] for name in (
            "backend", "phase", "cell_class", "noise_floor_phi", "b_min_blocks",
            "alpha_sel", "alpha_conf", "accepted", "samples_ref",
            "e_process_construction_id")
    }
    calibration = api.CalibrationOutputs(
        **calibration_values,
        # JSON arrays are transport shapes; the typed API owns tuple identity.
        anchor_gate_band=tuple(outputs["anchor_gate_band"]),
        solve_order_recorded=tuple(outputs["solve_order_recorded"]),
    )
    recorded_panel = api.ControlPanel(
        positive=schemas.Check(panel["positive"]),
        neutral=schemas.Check(panel["neutral"]),
        degraded_negative=schemas.Check(panel["degraded_negative"]),
        aa=schemas.Check(panel["aa"]),
        historical_replay=schemas.Check(panel["historical_replay"]),
        historical_replay_unavailable_reason=panel.get(
            "historical_replay_unavailable_reason"),
        operator_escalation_ref=panel.get("operator_escalation_ref"),
    )
    construction = inputs.get("construction")
    stopping = inputs.get("stopping_rule")
    if not isinstance(construction, Mapping) or not isinstance(stopping, Mapping):
        raise ValueError("live evaluation authority lacks construction/stopping rule")
    construction_id = construction.get("construction_id")
    # Selection is also a content check: a stale or invented id is refused by
    # the evaluator bundle here, before a candidate's samples are read.
    ak_statistics.select_construction(construction_id)
    source_body = dict(source)
    source_sha = source_body.pop("source_sha256", None)
    if source_sha != schemas.content_hash(source_body):
        raise ValueError("runtime source label hash does not verify")
    return LiveEvaluationAuthority(
        campaign_controls=campaign_controls,
        calibration=calibration,
        controls=recorded_panel,
        aa_cadence=_recorded_check(panel_result.get("aa_cadence"), "aa_cadence"),
        control_definitions_immutable=_recorded_check(
            panel_result.get("definitions_check"), "definitions_check"),
        construction_id=construction_id,
        stopping_rule_id=str(stopping.get("rule_id")),
        mde=float(accepted[0]["mde"]["value"]),
        runtime_source_label_ref=f"{root / 'runtime-source-label.json'}#sha256:{source_sha}",
        evidence_ref=str(root),
        calibration_frame=dict(frame),
    )


def reduce_live_blocks(request: api.EvaluationRequest, blocks: Sequence[Any],
                       authority: LiveEvaluationAuthority) -> api.EffectEstimate:
    """Reduce actual paired blocks with the bundle's fixed e-process.

    Raw per-arm samples are retained in the returned estimate.  No median-only
    campaign ``Pair`` can enter this function, so a writer cannot accidentally
    manufacture reproducibility from the accept rule's display values.
    """
    material = tuple(blocks)
    if not material or any(not isinstance(item, ak_statistics.PairedBlock)
                           for item in material):
        raise ValueError("live evaluation reduction requires real PairedBlock material")
    effects = tuple(ak_statistics.block_effect(
        item, scale=ak_statistics.EFFECT_SCALE_RELATIVE)
                    for item in material)
    oriented = tuple(ak_statistics.orient(value, request.metric_direction)
                     for value in effects)
    construction = ak_statistics.select_construction(authority.construction_id)
    threshold = authority.calibration.threshold_for(material[0].stratum)
    e_run = ak_statistics.run_e_process(
        oriented, construction=construction,
        hypothesis=ak_statistics.HYPOTHESIS_IMPROVEMENT, margin=0.0,
        threshold=threshold)
    raw = tuple(item.to_tuple() for item in material)
    raw_ref = "sha256:" + schemas.content_hash([item.to_list() for item in material])
    return api.EffectEstimate(
        metric=request.metric, metric_direction=request.metric_direction,
        value=median(effects), e_value=e_run.e_running_max, threshold=threshold,
        mde=authority.mde, noise_floor=authority.calibration.noise_floor_phi,
        paired_blocks=len(material), stratum=material[0].stratum,
        raw_samples=raw, raw_samples_ref=raw_ref)


def attach_belief_capture(event: Mapping[str, Any], *, effect_scale: str,
                          model_id: str, model_sha256: str,
                          producer_sha256: str) -> dict:
    """Attach the prospective Vidya capture with an identity-bound reduction.

    This mirrors the read-side contract in epyc-root's
    ``vidya.adapters.autokernel_evaluation_event/v1``.  It is intentionally a
    small producer helper in the research repository: runtime ingestion must
    not depend on whichever root-repository branch happens to be mounted.
    """
    record = json.loads(schemas.canonical_json(event))
    performance = record["performance"]
    raw = performance["raw_samples"]
    raw_sha = schemas.content_hash(raw)
    raw_ref = f"sha256:{raw_sha}"
    if performance.get("raw_samples_ref") != raw_ref:
        raise ValueError("belief capture raw sample hash disagrees with evaluation event")
    reps = record["claim_grammar"]["reps"]
    if performance.get("paired_blocks") != len(raw):
        raise ValueError("belief capture paired_blocks disagrees with raw block count")
    for index, block in enumerate(raw):
        if not isinstance(block, list) or len(block) != 9:
            raise ValueError(f"belief capture raw block {index} is not the 9-field shape")
        if not isinstance(block[7], list) or not isinstance(block[8], list) \
                or len(block[7]) != reps or len(block[8]) != reps:
            raise ValueError(
                f"belief capture raw block {index} arm vectors must each contain "
                f"claim reps={reps} scored repetitions")
    effects = []
    seen = set()
    for index, block in enumerate(raw):
        block_index, unit, stratum, order, segment, extension, measured_at, anchors, candidates = block
        if (isinstance(block_index, bool) or not isinstance(block_index, int)
                or block_index < 0 or block_index in seen
                or not isinstance(unit, str) or not unit.strip()
                or not isinstance(stratum, str) or not stratum.strip()
                or order not in {"anchor_first", "candidate_first"}
                or segment not in {"base", "extension"}
                or (segment == "base" and extension is not None)
                or (segment == "extension" and (
                    isinstance(extension, bool) or not isinstance(extension, int)
                    or extension < 1))):
            raise ValueError(f"belief capture raw block {index} has invalid identity/order")
        seen.add(block_index)
        anchor_value, candidate_value = median(anchors), median(candidates)
        if effect_scale == "relative":
            if anchor_value <= 0:
                raise ValueError("belief capture relative effect has non-positive anchor")
            effects.append((candidate_value - anchor_value) / anchor_value)
        elif effect_scale == "absolute":
            effects.append(candidate_value - anchor_value)
        else:
            raise ValueError("belief capture effect_scale must be relative or absolute")
    derived = median(effects)
    estimate = performance.get("estimate")
    if not isinstance(estimate, (int, float)) or isinstance(estimate, bool) \
            or not isfinite(float(estimate)) or not isclose(
                derived, float(estimate), rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("belief capture event estimate disagrees with raw block reduction")
    capture = {
        "schema": "epyc.vidya.autokernel_evaluation_event_capture.v1",
        "effect_scale": effect_scale,
        "model_id": model_id,
        "model_sha256": model_sha256,
        "source_sha256": record["artifact"]["source_sha256"],
        "binary_sha256": record["artifact"]["binary_sha256"],
        "resource_claim_receipt": record["resource_claim_receipt"],
        "producer_sha256": producer_sha256,
        "raw_samples_sha256": raw_sha,
    }
    binding = {
        "schema": capture["schema"],
        "event_id": record["event_id"],
        "campaign_id": record["campaign_id"],
        "candidate_id": record["candidate_id"],
        "category": record["claim_grammar"]["category"],
        "protocol_id": record["claim_grammar"]["protocol_id"],
        "metric": record["claim_grammar"]["metric"],
        "metric_direction": record["claim_grammar"]["metric_direction"],
        "reps": record["claim_grammar"]["reps"],
        **capture,
    }
    capture["identity_binding_sha256"] = schemas.content_hash(binding)
    performance["search_discipline"]["belief_capture"] = capture
    return record

#: Versioned, because a runner id with no version cannot fail closed on drift.
CONTROL_RUNNER_ID = "ak3-executed-control-runner/v1"

#: There is one kind of submission. Named so the absence of a second one is
#: visible in the module's own vocabulary rather than only in its type list.
SUBMISSION_KIND = "candidate"


# =============================================================================
# Errors — refusals, never degraded results
# =============================================================================

class ControlExecutionError(controls.ControlsError):
    """Base class for every refusal this module makes.

    Subclasses `controls.ControlsError` (and so `api.EvaluatorError`) on purpose:
    a control that could not be executed and a control that could not be
    evaluated are the same fact from the campaign's point of view, and a caller
    that catches one must not silently miss the other.
    """


class PipelineNotWired(ControlExecutionError):
    """No pipeline, no fixtures, or no campaign statistics were supplied.

    There is no default for any of the three. `api.TierDispatcher` refuses a tier
    with no runner because *"an unrun tier with no gate results would derive to
    PASS"*; a control runner with a default pipeline is the same fail-open one
    layer down — it would report five controls as having produced no failures.
    """


class FixtureNotDeclared(ControlExecutionError):
    """A control definition names a `fixture_id` the fixture set does not carry.

    A wiring defect, so it raises. *"A campaign that cannot run controls 1-4 MUST
    NOT rank any candidate"* — and a fixture set missing one is a bundle that
    cannot run them, which is a fact about the evaluator rather than about the
    gate.
    """


class FixtureBundleDrift(ControlExecutionError):
    """The fixtures do not hash to the value the campaign pinned.

    *"Control definitions, FIXTURES, expected directions, and seeds live inside
    the evaluator bundle under the measurement trust boundary and MUST NOT be
    modified by any process inside the loop."* Fixtures are named in that clause
    alongside the definitions, and until this class existed only the definitions
    had a pin. A control whose material was swapped has been modified exactly as
    surely as one whose `requirement` string was edited.
    """


class CalibrationMaterialRelabelled(ControlExecutionError):
    """Two calibration blocks with different `unit_id`s carry identical samples.

    `PairedBlock.unit_id` is *"the measurement-material unit (shape, seed) the
    block used"*, and `estimate_noise_floor` refuses when
    `len(aa_effects) < calibration_block_count` because *"phi is estimated over AT
    LEAST the declared count"*. Nothing downstream compares the SAMPLES, so a
    producer that re-emits one recorded arm under a fresh unit id satisfies the
    declared count with material it never measured: twelve rotation epochs over a
    static ten-block fixture present as 120 units and 10 measurements, and the
    solve accepts, and phi is the 95th percentile of ten numbers wearing a
    hundred-and-twenty-number label.

    A relabelled block is not a second measurement. This refuses rather than
    letting the count be manufactured, because every layer above reads the count
    and none of them can see the duplication.
    """


class SweepNotLicensed(ControlExecutionError):
    """Calibration material was requested out of a sweep that produced no panel.

    A blocked sweep — an off-schedule rotation, a control that could not be
    disposed of — has `panel_result is None` and `may_rank False`. Its A/A arm is
    still sitting in the runner's submission list, and reading it out would
    calibrate phi from a window the campaign was not allowed to proceed in.
    """


class WindowBindingStale(ControlExecutionError):
    """The campaign binding describes a different window than the sweep is running.

    `api.WindowAttestations` carries the resource-claim receipt, the host receipt
    and the anchor identities *for one window*. A binding captured at window 1 and
    reused at window 12 submits every control under window 1's claim receipt —
    a claim that has since been released — and `resource_claim_same_holder`,
    `host_health` and `anchor_gate` all read as whatever they were when the object
    was built. Denial 8's *"no inference run OUTSIDE A HELD CLAIM"* is not
    satisfied by attesting a claim receipt copied from an earlier window.
    """


class RotationLedgerViolation(ControlExecutionError):
    """The sweep was asked to run at a window count it has already run at or past.

    `SeedRotationSchedule.check_rotation` compares two integers the CALLER
    supplies, and the per-control seeds are derived from one of them — so a caller
    that reports a matching pair always passes, and a caller that freezes
    `windows_completed` runs its whole campaign on one holdout while the check
    reads PASS every time. That is the very defect the check was written for.
    The sweep therefore keeps its own ledger of the window counts it has served
    and refuses a repeat or a rewind: a monotone clock is the one part of this
    the gated party cannot supply.
    """


class CalibrationMaterialMissing(ControlExecutionError):
    """The calibration block was asked for material the sweep did not produce.

    φ is *"estimated from the A/A control"*. A solve assembled without the A/A
    control's own blocks would calibrate the false-positive rate of a measurement
    that never happened, and — because `solve_calibration` returns an accepted
    solve either way — the substitution would never surface as an error. So this
    raises rather than falling back to whatever blocks are to hand.
    """


# =============================================================================
# The ONE thing that enters the evaluation pipeline
# =============================================================================

@dataclass(frozen=True)
class CandidateSubmission:
    """One evaluation's material. Candidates submit this; so do controls.

    Deliberately missing, and each absence is a guarantee:

      * **no `control_id`, no `is_control`, no `kind`** — the pipeline cannot
        branch on whether what it is scoring is a control, so a control cannot be
        scored down a path a candidate never takes. This is the structural form
        of *"a control that runs down a different code path proves nothing about
        the path that matters"*; the alternative (an honour-system label the
        pipeline promises not to read) is not checkable.
      * **no `effect`** — the submission carries RAW BLOCKS and the pipeline
        reduces them. A submission that carried an `api.EffectEstimate` would let
        a control hand over the number it is about to be scored on, and control 3
        exists precisely to catch a candidate that reports rather than measures.
      * **no gates** — those come from the tier gate runner inside the
        dispatcher, which is the candidate path's own.

    `blocks` may be empty, and that means exactly one thing: this record is not a
    rate comparison. `api.TierDispatcher` reads `effect is None` that way, and the
    pipeline maps an empty block tuple onto it rather than inventing an estimate.
    """

    request: api.EvaluationRequest
    window: api.WindowAttestations
    blocks: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request, api.EvaluationRequest):
            raise TypeError("submission.request must be an api.EvaluationRequest")
        if not isinstance(self.window, api.WindowAttestations):
            raise TypeError("submission.window must be an api.WindowAttestations")
        if not isinstance(self.blocks, tuple):
            raise TypeError("submission.blocks must be a tuple of statistics.PairedBlock")
        for block in self.blocks:
            if not isinstance(block, ak_statistics.PairedBlock):
                raise TypeError(
                    f"submission.blocks must all be statistics.PairedBlock, got "
                    f"{type(block).__name__}")

    def to_dict(self) -> dict:
        return {
            "kind": SUBMISSION_KIND,
            "event_id": self.request.event_id,
            "candidate_id": self.request.candidate_id,
            "tier": self.request.tier,
            "order_seed": self.window.order_seed,
            "block_count": len(self.blocks),
            "unit_ids": [b.unit_id for b in self.blocks],
        }


class CandidatePipeline(Protocol):
    """The single seam through which anything is evaluated. One method, on purpose.

    A second method would be a second code path, and the control panel's whole
    claim is that the controls went down the candidate's.
    """

    pipeline_id: str

    def evaluate(self, submission: CandidateSubmission) -> api.EvaluationOutcome:
        ...


class DispatchPipeline:
    """The real pipeline: reduce the blocks, then dispatch. Nothing else.

    The reducer is `statistics.PairedBlockReducer` and the dispatcher is
    `api.TierDispatcher`; both are the campaign's own instances, passed in. The
    reducer is what turns raw blocks into an `api.EffectEstimate`, and putting it
    HERE rather than in the runner is what stops a control from supplying one.

    `statistics.ReductionInadmissible` is deliberately NOT swallowed. Its
    docstring is explicit that answering `None` would make
    `api.TierDispatcher` treat the record as "not a rate comparison" and skip the
    rate-only void conditions — suppressing the very void a strata or
    order-control violation must raise. It propagates, and
    `ExecutedControlRunner` records the control as having not run, with the
    reduction's own reason.
    """

    def __init__(self, *, dispatcher: api.TierDispatcher,
                 reducer: Any,
                 pipeline_id: str = "ak3-dispatch-pipeline/v1") -> None:
        if not isinstance(dispatcher, api.TierDispatcher):
            raise PipelineNotWired("DispatchPipeline requires an api.TierDispatcher")
        if reducer is None or not hasattr(reducer, "reduce_blocks"):
            raise PipelineNotWired(
                "DispatchPipeline requires an api.EffectReducer exposing "
                "reduce_blocks(request, blocks); there is no default reducer, because a "
                "pipeline that could not reduce would evaluate every record as though it "
                "were not a rate comparison and skip the rate-only void conditions")
        if not isinstance(pipeline_id, str) or "/v" not in pipeline_id:
            raise PipelineNotWired(
                f"pipeline_id {pipeline_id!r} has no '/vN' suffix; an unversioned "
                "pipeline id cannot fail closed on drift")
        self._dispatcher = dispatcher
        self._reducer = reducer
        self.pipeline_id = pipeline_id

    def evaluate(self, submission: CandidateSubmission) -> api.EvaluationOutcome:
        if not isinstance(submission, CandidateSubmission):
            raise TypeError("pipeline.evaluate takes a CandidateSubmission")
        effect = None
        if submission.blocks:
            effect = self._reducer.reduce_blocks(submission.request, submission.blocks)
        return self._dispatcher.dispatch(submission.request, submission.window,
                                         effect=effect)


# =============================================================================
# Campaign bindings — what every submission in a window shares
# =============================================================================

@dataclass(frozen=True)
class CampaignBinding:
    """The parts of a submission that are the CAMPAIGN's, not the candidate's.

    Controls are bound to the same object candidates are. That is not a
    convenience: *"the calibration block runs on the campaign's own anchor …
    under the identical recipe, claim, interleaving and reduction discipline that
    candidate rounds will use"*, and the controls are what that discipline is
    calibrated from. A control window assembled from its own anchor, its own
    scope denominator or its own recipe receipt would be measuring a different
    machine and reporting on this one.

    It carries NO `api.WindowAttestations`, and that is the point of the class.
    A window's attestations are `resource_claim_receipt`, `host_receipt`,
    `anchor_at_open/close` and a dozen Checks — every one of them a fact about ONE
    window. Held on a campaign-scoped object they are captured once and reused
    forever: every control in every window of the campaign submits under the first
    window's claim receipt, long after that claim was released, with
    `resource_claim_same_holder` frozen at whatever it was when the object was
    built. Denial 8's *"no inference run OUTSIDE A HELD CLAIM"* is not satisfied by
    attesting a receipt copied from an earlier window. The live window is opened on
    the RUNNER, per sweep, by `ExecutedControlRunner.open_window()`.
    """

    campaign_id: str
    backend: str
    phase: str
    cell_class: str
    protocol_id: str
    evaluator: api.EvaluatorIdentity
    scope_denominator: api.ScopeDenominator
    scope_manifest_sha256: str
    co_residency: str
    metric: str
    metric_direction: str
    reps: int
    change_class: str
    anchor: Optional[api.AnchorIdentity]
    campaign_controls: Optional[api.CampaignControls]
    calibration: Optional[api.CalibrationOutputs]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.campaign_id, "binding.campaign_id")
        if not isinstance(self.evaluator, api.EvaluatorIdentity):
            raise TypeError("binding.evaluator must be an api.EvaluatorIdentity")
        if not isinstance(self.scope_denominator, api.ScopeDenominator):
            raise TypeError("binding.scope_denominator must be an api.ScopeDenominator")
        if self.anchor is not None and not isinstance(self.anchor, api.AnchorIdentity):
            raise TypeError("binding.anchor must be an api.AnchorIdentity or None")

    def request_for(self, *, event_id: str, candidate_id: str, tier: str,
                    artifact: api.ArtifactIdentity,
                    determinism: api.DeterminismReport,
                    created_at: str) -> api.EvaluationRequest:
        return api.EvaluationRequest(
            event_id=event_id, campaign_id=self.campaign_id, candidate_id=candidate_id,
            tier=tier, backend=self.backend, phase=self.phase, cell_class=self.cell_class,
            protocol_id=self.protocol_id, artifact=artifact, anchor=self.anchor,
            evaluator=self.evaluator, scope_denominator=self.scope_denominator,
            scope_manifest_sha256=self.scope_manifest_sha256,
            co_residency=self.co_residency, determinism=determinism,
            metric=self.metric, metric_direction=self.metric_direction, reps=self.reps,
            change_class=self.change_class, anchor_tier=tier, transfer_ratio_to=(),
            created_at=created_at, campaign_controls=self.campaign_controls,
            calibration=self.calibration)


# =============================================================================
# Fixtures — the controls' measurement material, hash-pinned like the definitions
# =============================================================================

@dataclass(frozen=True)
class ControlFixture:
    """One control's material, as data the actor cannot alter.

    Everything here is JSON-canonicalizable so the set's content hash covers it.
    What is deliberately NOT here:

      * **`promoted`** — control 5's *"it MUST promote"* is DERIVED by the runner
        from the verdict the gate produced. A fixture that could declare itself
        promoted would let the accept-side control pass by assertion, which is
        the exact failure (a gate that has quietly stopped passing anything, and
        a surface that reports "active, blockers: []") the control exists to
        catch.
      * **an effect value** — the blocks are the material and the reducer
        computes the estimate.
      * **an expected outcome** — the expected direction is the DEFINITION's
        (`ControlDefinition.expected_direction`), under the measurement trust
        boundary, and a fixture carrying a second copy would be a second source
        of truth for what the control is supposed to do.

    `anchor_samples` and `candidate_samples` are per-block sample vectors and must
    be the same length as each other: they are the two arms of a paired block, and
    a block with one arm shorter has not been interleaved, it has been trimmed.

    `measured_at` is REQUIRED and is stamped onto every block this fixture
    produces. `PairedBlock.measured_at` is optional and was being left `None`,
    which made a control block indistinguishable from one measured under the
    window's own claim — the record could not tell a number taken last month from
    a number taken under tonight's held claim. It is *"what orders confirmation
    evidence against lineage entry"*, so a control arm with no stamp is an arm
    that cannot be ordered against anything.
    """

    fixture_id: str
    control_id: str
    tier: str
    candidate_id: str
    artifact: api.ArtifactIdentity
    determinism: api.DeterminismReport
    created_at: str
    measured_at: str
    stratum: str
    anchor_samples: tuple
    candidate_samples: tuple
    available: bool = True
    unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.fixture_id, "fixture.fixture_id")
        if self.control_id not in controls.CONTROL_IDS:
            raise ValueError(f"fixture.control_id {self.control_id!r} is not one of "
                             f"{list(controls.CONTROL_IDS)}")
        api.admit_tier(self.tier)
        _require_nonempty_str(self.candidate_id, "fixture.candidate_id")
        if not self.candidate_id.startswith("akc-"):
            raise ValueError(
                f"fixture.candidate_id {self.candidate_id!r} must start with 'akc-'; a "
                "control is submitted as a candidate and carries a candidate's identity, "
                "because the pipeline must not be able to tell it apart from one")
        if not isinstance(self.artifact, api.ArtifactIdentity):
            raise TypeError("fixture.artifact must be an api.ArtifactIdentity")
        if not isinstance(self.determinism, api.DeterminismReport):
            raise TypeError("fixture.determinism must be an api.DeterminismReport")
        _require_nonempty_str(self.created_at, "fixture.created_at")
        _require_nonempty_str(self.measured_at, "fixture.measured_at")
        if ak_statistics._parse_instant(self.measured_at) is None:
            raise ValueError(
                f"fixture.measured_at {self.measured_at!r} is not an ISO-8601 timestamp "
                "with a UTC offset; a control arm that cannot be ordered against lineage "
                "entry cannot be shown to have been measured under this window's claim "
                "rather than recovered from an earlier one")
        if self.stratum not in api.STRATA:
            raise ValueError(f"fixture.stratum {self.stratum!r} is not one of "
                             f"{list(api.STRATA)}")
        if self.control_id == controls.CONTROL_HISTORICAL_WIN_REPLAY and self.tier != "T2":
            raise ValueError(
                "the historical-win replay is 'replayed end-to-end through T0-T2 under a "
                f"declared contract', so its fixture runs at T2, not {self.tier!r}; a "
                "replay that stopped short of T2 cannot have promoted through it")
        for name in ("anchor_samples", "candidate_samples"):
            arms = getattr(self, name)
            if not isinstance(arms, tuple):
                raise TypeError(f"fixture.{name} must be a tuple of per-block sample "
                                "tuples")
            for arm in arms:
                if not isinstance(arm, tuple) or not arm:
                    raise ValueError(f"fixture.{name} entries must be non-empty tuples of "
                                     "samples")
        if len(self.anchor_samples) != len(self.candidate_samples):
            raise ValueError(
                f"fixture {self.fixture_id!r} has {len(self.anchor_samples)} anchor blocks "
                f"and {len(self.candidate_samples)} candidate blocks; a paired block has "
                "two arms and an unpaired one is not a paired design")
        if self.available and not self.anchor_samples:
            raise ValueError(
                f"fixture {self.fixture_id!r} declares itself available with no blocks; "
                "'zero reps is not a measurement' applies to a control exactly as it "
                "applies to a candidate")
        if not self.available:
            _require_nonempty_str(self.unavailable_reason,
                                           "fixture.unavailable_reason")

    #: How many nonces `_unit_id` may try before it gives up looking for a unit
    #: the split rule puts in this fixture's declared stratum. A bound, not a
    #: guess: an unbounded search would hang instead of failing, and a search that
    #: cannot find a unit is a split rule this fixture cannot draw from — which is
    #: a wiring fact worth raising rather than looping on.
    _MAX_UNIT_DRAWS = 4096

    def _unit_id(self, *, seed: str, block_index: int, split_rule: Any) -> str:
        for nonce in range(self._MAX_UNIT_DRAWS):
            digest = schemas.content_hash({
                "derivation": "ak3-control-unit/v1",
                "fixture_id": self.fixture_id,
                "seed": seed,
                "block_index": block_index,
                "nonce": nonce,
            })
            unit = f"{self.fixture_id}#u{digest[:16]}"
            if split_rule.assign(unit) == self.stratum:
                return unit
        raise PipelineNotWired(
            f"fixture {self.fixture_id!r} could not draw a measurement unit in stratum "
            f"{self.stratum!r} in {self._MAX_UNIT_DRAWS} attempts under this campaign's "
            "split rule; the control cannot be measured on material the campaign's own "
            "partition does not admit")

    @staticmethod
    def _segment(index: int, *, base_blocks: int, blocks_per_round: int) -> tuple:
        """`(segment, extension_round)` for one block index.

        *"An extension that cannot say which declared round it belongs to is
        unstructured continuation."* The base segment is the first `B_min` blocks
        and everything past it belongs to a declared round, which is the shape
        `statistics._check_extension_structure` enforces on candidates. A control
        whose blocks were all labelled `base` would be reduced under a structure
        no candidate is allowed to use.
        """
        if index < base_blocks:
            return ak_statistics.SEGMENT_BASE, None
        beyond = index - base_blocks
        return ak_statistics.SEGMENT_EXTENSION, (beyond // blocks_per_round) + 1

    def blocks_for(self, *, seed: str, schedule: Any, split_rule: Any,
                   base_blocks: int, blocks_per_round: int) -> tuple:
        """Build this fixture's paired blocks for one rotation epoch's seed.

        The seed reaches `PairedBlock.unit_id`, which the protocol defines as
        *"the measurement-material unit (shape, seed) the block used"*. That is
        what makes rotation mean something: a rotated control seed measures
        different units, so a holdout cannot be silently reused across every
        window of a campaign. It is also why the seed cannot merely be recorded
        next to the run — a seed that changes nothing about what was measured is
        a rotation schedule with no subject.

        NOTE the seed does NOT change the samples a static fixture serves: the
        arms are the recorded ones whatever the epoch, so two epochs are the same
        measurement under two labels. `_refuse_relabelled_material` is what stops
        those being pooled as if they were two measurements.

        Two campaign disciplines the fixture does NOT get to choose, because a
        control measured under different ones is not measured under the
        candidates':

          * the block ORDER comes from the campaign's `OrderSchedule` —
            *"candidate and anchor are interleaved and order-randomized within
            every paired block"*; and
          * the unit must land in the fixture's declared stratum under the
            campaign's `StratumSplitRule`, so the selection/confirmation split
            partitions control material exactly as it partitions candidate
            material.
        """
        _require_nonempty_str(seed, "seed")
        if schedule is None or not hasattr(schedule, "order_for"):
            raise PipelineNotWired(
                "blocks_for() needs the campaign's OrderSchedule; a control that chose "
                "its own interleave order would be measured under a discipline the "
                "candidates are not")
        if split_rule is None or not hasattr(split_rule, "assign"):
            raise PipelineNotWired(
                "blocks_for() needs the campaign's StratumSplitRule; a control whose "
                "units were not drawn through the campaign's own partition would sit "
                "outside the selection/confirmation split that governs every candidate")
        for name, value in (("base_blocks", base_blocks),
                            ("blocks_per_round", blocks_per_round)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PipelineNotWired(
                    f"blocks_for() needs a positive {name}; it comes from the campaign's "
                    "calibrated B_min and its declared bounded extension, and a control "
                    "segmented by a guess is a control reduced under a structure the "
                    "candidates are not")
        blocks = []
        for index, (anchor_arm, candidate_arm) in enumerate(
                zip(self.anchor_samples, self.candidate_samples)):
            segment, extension_round = self._segment(
                index, base_blocks=base_blocks, blocks_per_round=blocks_per_round)
            blocks.append(ak_statistics.PairedBlock(
                block_index=index,
                unit_id=self._unit_id(seed=seed, block_index=index,
                                      split_rule=split_rule),
                stratum=self.stratum,
                order=schedule.order_for(index),
                anchor_samples=tuple(float(v) for v in anchor_arm),
                candidate_samples=tuple(float(v) for v in candidate_arm),
                segment=segment,
                extension_round=extension_round,
                measured_at=self.measured_at,
            ))
        return tuple(blocks)

    def to_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "control_id": self.control_id,
            "tier": self.tier,
            "candidate_id": self.candidate_id,
            "artifact": self.artifact.to_dict(),
            "determinism": self.determinism.to_dict(),
            "created_at": self.created_at,
            "measured_at": self.measured_at,
            "stratum": self.stratum,
            "anchor_samples": [list(arm) for arm in self.anchor_samples],
            "candidate_samples": [list(arm) for arm in self.candidate_samples],
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class ControlFixtureSet:
    """The five controls' fixtures, with a content hash the campaign pins.

    Same shape as `controls.ControlBundle`: `__post_init__` RE-DERIVES the digest
    from the set's own contents, so a set carrying a digest that does not describe
    it is refused at construction rather than trusted.
    """

    fixtures: tuple
    digest: str
    source_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.fixtures, tuple) or not self.fixtures:
            raise ValueError("fixture_set.fixtures must be a non-empty tuple")
        for fixture in self.fixtures:
            if not isinstance(fixture, ControlFixture):
                raise TypeError("fixture_set.fixtures must all be ControlFixture")
        by_control: dict = {}
        by_fixture_id: dict = {}
        for fixture in self.fixtures:
            if fixture.control_id in by_control:
                raise ValueError(
                    f"two fixtures for control {fixture.control_id!r}; the control "
                    "definitions name one fixture_id per control, and choosing between "
                    "two would be the runner selecting its own material")
            by_control[fixture.control_id] = fixture
            # Lookup is BY fixture_id (`for_definition`), so two fixtures sharing
            # one would make the answer depend on tuple order — and the losing
            # control would be measured on the winner's material with no digest
            # disturbed, because both sides still hash to themselves.
            if fixture.fixture_id in by_fixture_id:
                raise ValueError(
                    f"two fixtures carry the id {fixture.fixture_id!r} (controls "
                    f"{by_fixture_id[fixture.fixture_id].control_id!r} and "
                    f"{fixture.control_id!r}); fixtures are resolved by the definition's "
                    "fixture_id, so a duplicated id makes which material a control runs "
                    "on depend on declaration order")
            by_fixture_id[fixture.fixture_id] = fixture
        _require_nonempty_str(self.source_label, "fixture_set.source_label")
        derived = schemas.content_hash(_fixture_payload(self.fixtures))
        if self.digest != derived:
            raise FixtureBundleDrift(
                f"fixture_set.digest {self.digest!r} does not describe the fixtures this "
                f"set carries (derived {derived!r}); a digest is derived from the "
                "contents, never supplied alongside them")

    def for_definition(self, definition: controls.ControlDefinition) -> ControlFixture:
        """Look the fixture up by the DEFINITION's `fixture_id`, not by control id.

        The definitions are under the measurement trust boundary and the fixture
        ids live in them. Looking up by control id would let a fixture set answer
        for a control while carrying material the definition does not name, which
        is a fixture substitution that no digest would catch — both sides would
        still hash to themselves.
        """
        for fixture in self.fixtures:
            if fixture.fixture_id == definition.fixture_id:
                if fixture.control_id != definition.control_id:
                    raise FixtureNotDeclared(
                        f"fixture {fixture.fixture_id!r} is declared for control "
                        f"{fixture.control_id!r} but definition "
                        f"{definition.control_id!r} names it")
                return fixture
        raise FixtureNotDeclared(
            f"no fixture with id {definition.fixture_id!r} for control "
            f"{definition.control_id!r}; the fixture set carries "
            f"{[f.fixture_id for f in self.fixtures]}. A campaign that cannot run "
            "controls 1-4 MUST NOT rank any candidate.")

    def to_dict(self) -> dict:
        return {"fixtures": [f.to_dict() for f in self.fixtures],
                "digest": self.digest, "source_label": self.source_label}


def _fixture_payload(fixtures: Sequence[ControlFixture]) -> dict:
    return {
        "payload_id": "ak3-control-fixtures/v1",
        "fixtures": [f.to_dict() for f in sorted(fixtures, key=lambda f: f.fixture_id)],
    }


def resolve_fixture_set(*, fixtures: Sequence[ControlFixture],
                        pinned_digest: str,
                        source_label: str) -> ControlFixtureSet:
    """Resolve the fixture set against the campaign's pin, or raise.

    The pin is REQUIRED and there is no unpinned resolve, for the reason
    `resolve_control_bundle` gives: an unpinned resolve is a resolve that cannot
    detect the thing the pin exists to detect. *"Control definitions, fixtures,
    expected directions, and seeds … MUST NOT be modified by any process inside
    the loop"* names fixtures explicitly.
    """
    _require_nonempty_str(pinned_digest, "pinned_digest")
    resolved = tuple(fixtures)
    derived = schemas.content_hash(_fixture_payload(resolved))
    if derived != pinned_digest:
        raise FixtureBundleDrift(
            f"the control fixtures do not match the campaign pin: pinned "
            f"{pinned_digest[:12]}, resolved {derived[:12]}. The fixtures live inside the "
            "evaluator bundle under the measurement trust boundary and MUST NOT be "
            "modified by any process inside the loop.")
    return ControlFixtureSet(fixtures=resolved, digest=derived, source_label=source_label)


# =============================================================================
# The runner
# =============================================================================

def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _opposite_direction(direction: str) -> str:
    return "lower_better" if direction == "higher_better" else "higher_better"


def _rank_key_or_none(verdict: api.Verdict):
    """Call the exact method a ranking loop calls. Returns the key, or None.

    The same construction `controls._rank_key_or_none` uses, for the same reason:
    *"MUST receive no speed rank at all"* is a statement about what
    `Verdict.rank_key()` does, so promotion is decided by calling it rather than
    by reading a flag that is supposed to agree with it.
    """
    try:
        return verdict.rank_key()
    except api.SpeedRankUnavailable:
        return None


class ExecutedControlRunner:
    """`controls.ControlRunner`, implemented: runs one control through the pipeline.

    Everything it produces is DERIVED from the `api.EvaluationOutcome` the
    pipeline returned. It stamps no status, no promotion and no magnitude. The
    fixture supplies material; the gate supplies the verdict; this class only
    projects one onto `controls.ControlObservation`.
    """

    runner_id = CONTROL_RUNNER_ID

    def __init__(self, *, pipeline: Any, fixtures: ControlFixtureSet,
                 binding: CampaignBinding,
                 campaign_statistics: Any) -> None:
        if pipeline is None or not hasattr(pipeline, "evaluate"):
            raise PipelineNotWired(
                "ExecutedControlRunner requires a CandidatePipeline exposing "
                "evaluate(submission); there is no default, because a control with no "
                "verdict would be evaluated as a control that found no failures")
        if not isinstance(fixtures, ControlFixtureSet):
            raise PipelineNotWired("ExecutedControlRunner requires a ControlFixtureSet")
        if not isinstance(binding, CampaignBinding):
            raise PipelineNotWired("ExecutedControlRunner requires a CampaignBinding")
        if campaign_statistics is None or not hasattr(campaign_statistics,
                                                      "order_schedule") \
                or not hasattr(campaign_statistics, "split_rule"):
            raise PipelineNotWired(
                "ExecutedControlRunner requires the campaign's statistics.CampaignStatistics: "
                "the block interleave order comes from the campaign's OrderSchedule, and a "
                "control that ordered its own blocks would be measured under a discipline "
                "the candidates are not")
        self._pipeline = pipeline
        self._fixtures = fixtures
        self._binding = binding
        self._statistics = campaign_statistics
        #: Every submission this runner made, for the record. Written here and
        #: never read by the scoring path — an audit trail, not an input.
        self.submissions: list = []
        #: The subset whose control ACTUALLY RAN: the pipeline returned an
        #: `api.EvaluationOutcome` and the observation carries its verdict. A
        #: submission whose reduction was refused as not search-grade is in
        #: `submissions` (it was made) and NOT here (it measured nothing), because
        #: `calibration_material` reads this one. Before it existed, blocks the
        #: reducer had just rejected — below B_min, undeclared extension — were
        #: pooled into phi, and phi is the floor every candidate is judged against.
        self.admitted_submissions: list = []
        #: The LIVE window. `None` between sweeps, and a submission cannot be
        #: assembled while it is None: a control measured outside an open window
        #: has no claim receipt of its own to attest.
        self._window_id: Optional[str] = None
        self._window: Optional[api.WindowAttestations] = None
        #: Every window id this runner has opened. A window is opened ONCE; a
        #: second open under the same id with different receipts is receipt
        #: shopping — re-attesting a completed window with a healthier host.
        self._windows_opened: list = []

    # -- the live window --------------------------------------------------------

    def open_window(self, *, window_id: str,
                    window: api.WindowAttestations) -> None:
        """Bind the attestations of the window that is open RIGHT NOW.

        Per sweep, never per campaign. The claim receipt, the host receipt and the
        anchor identities in `window` are facts about this window and are stale
        the moment it closes.
        """
        _require_nonempty_str(window_id, "window_id")
        if not isinstance(window, api.WindowAttestations):
            raise WindowBindingStale(
                "open_window() requires the live api.WindowAttestations; there is no "
                "template and no default, because a default window is a claim receipt "
                "nobody checked")
        if window.resource_claim_open.outcome != schemas.PASS:
            raise WindowBindingStale(
                f"window {window_id!r} attests resource_claim_open="
                f"{window.resource_claim_open.outcome} "
                f"({'; '.join(window.resource_claim_open.reasons)}); denial 8 is "
                "'no inference run OUTSIDE A HELD CLAIM', and a sweep opened on a window "
                "whose claim did not open is a sweep with no claim. The dispatcher would "
                "void the records afterwards; refusing to run is the point at which that "
                "is still free.")
        if window_id in self._windows_opened:
            raise WindowBindingStale(
                f"window {window_id!r} has already been opened on this runner; a window is "
                "opened once, and re-opening one with fresh receipts would let a completed "
                "window be re-attested under a healthier host state")
        self._windows_opened.append(window_id)
        self._window_id = window_id
        self._window = window

    def close_window(self) -> None:
        """Drop the live window. Any further submission raises until one is open."""
        self._window_id = None
        self._window = None

    @property
    def fixtures(self) -> ControlFixtureSet:
        """The pinned fixture set this runner is bound to. Read-only."""
        return self._fixtures

    @property
    def effect_scale(self) -> str:
        """The campaign's effect scale — the reducer's, not a second declaration."""
        return self._statistics.effect_scale

    # -- the ControlRunner seam -------------------------------------------------

    def run_control(self, definition: controls.ControlDefinition,
                    context: controls.ControlRunContext) -> controls.ControlObservation:
        if not isinstance(definition, controls.ControlDefinition):
            raise TypeError("run_control takes a controls.ControlDefinition")
        if not isinstance(context, controls.ControlRunContext):
            raise TypeError("run_control takes a controls.ControlRunContext")

        fixture = self._fixtures.for_definition(definition)
        if not fixture.available:
            return controls.ControlObservation(
                control_id=definition.control_id, ran=False,
                could_not_run_reason=(
                    f"the {definition.fixture_id!r} fixture is not available: "
                    f"{fixture.unavailable_reason}"))

        submission = self._submission_for(fixture, context)
        self.submissions.append(submission)
        try:
            outcome = self._pipeline.evaluate(submission)
        except ak_statistics.ReductionInadmissible as exc:
            # The reduction refused, carrying its own reasons. A control whose
            # measurement was not search-grade did not run — it is COULD_NOT_CHECK
            # with a reason, never a silent PASS and never a FAIL against the gate,
            # because nothing about the GATE was learned here.
            return controls.ControlObservation(
                control_id=definition.control_id, ran=False,
                could_not_run_reason=(
                    f"the {definition.control_id} control's blocks did not reduce to a "
                    f"search-grade estimate: {exc}"))
        if not isinstance(outcome, api.EvaluationOutcome):
            raise PipelineNotWired(
                f"the pipeline returned {type(outcome).__name__} for control "
                f"{definition.control_id!r}; expected an api.EvaluationOutcome")

        observation = self._observation_from(definition, fixture, submission, outcome)
        if observation.ran:
            self.admitted_submissions.append(submission)
        return observation

    # -- assembly ---------------------------------------------------------------

    def _submission_for(self, fixture: ControlFixture,
                        context: controls.ControlRunContext) -> CandidateSubmission:
        if self._window is None:
            raise WindowBindingStale(
                "no window is open on this runner; call open_window() with the live "
                "api.WindowAttestations before running a control. A control assembled "
                "without one would carry no resource-claim receipt of its own, and "
                "'no inference run OUTSIDE A HELD CLAIM' is not satisfied by a receipt "
                "copied from some earlier window")
        if context.window_id != self._window_id:
            raise WindowBindingStale(
                f"the open window is {self._window_id!r} and the sweep is running window "
                f"{context.window_id!r}. The open window's attestations — claim receipt "
                f"{self._window.resource_claim_receipt!r}, host receipt "
                f"{self._window.host_receipt!r} — describe {self._window_id!r} and nothing "
                "else; submitting another window's controls under them attests a claim "
                "that window never held")
        if context.campaign_id != self._binding.campaign_id:
            raise WindowBindingStale(
                f"this runner is bound to campaign {self._binding.campaign_id!r} and the "
                f"sweep is running campaign {context.campaign_id!r}")
        schedule = self._statistics.order_schedule(fixture.candidate_id)
        blocks = fixture.blocks_for(
            seed=context.seed, schedule=schedule,
            split_rule=self._statistics.split_rule,
            base_blocks=self._statistics.b_min,
            blocks_per_round=self._statistics.stopping_rule.extension.blocks_per_round)
        event_id = "ake-" + schemas.content_hash({
            "derivation": "ak3-control-event/v1",
            "campaign_id": context.campaign_id,
            "window_id": context.window_id,
            "fixture_id": fixture.fixture_id,
            "seed": context.seed,
        })[:32]
        request = self._binding.request_for(
            event_id=event_id, candidate_id=fixture.candidate_id, tier=fixture.tier,
            artifact=fixture.artifact, determinism=fixture.determinism,
            created_at=fixture.created_at)
        return CandidateSubmission(
            request=request,
            # The LIVE window, carrying this control's derived order seed. Only
            # the seed moves: everything else is the attestation of the window
            # that is open right now, and it was opened by the sweep.
            window=replace(self._window, order_seed=context.seed),
            blocks=blocks)

    def _observation_from(self, definition: controls.ControlDefinition,
                          fixture: ControlFixture,
                          submission: CandidateSubmission,
                          outcome: api.EvaluationOutcome) -> controls.ControlObservation:
        verdict = outcome.verdict
        effect = verdict.effect

        # The verdict must describe THIS submission's material. `EffectEstimate`
        # carries the raw samples it was reduced from, and nothing else ties an
        # outcome back to the submission that produced it — `Verdict` names no
        # candidate and no event. Without this the runner projects whatever
        # verdict the pipeline hands back onto whichever control it happens to be
        # running, which is a control reporting a measurement it did not take.
        if effect is not None:
            submitted = tuple(block.to_tuple() for block in submission.blocks)
            if tuple(effect.raw_samples) != submitted:
                raise PipelineNotWired(
                    f"the verdict returned for control {definition.control_id!r} was "
                    f"reduced from {len(effect.raw_samples)} block(s) that are not the "
                    f"{len(submitted)} this runner submitted. A control observes the "
                    "verdict the gate gave IT; projecting one computed over other "
                    "material onto this control reports a measurement it did not take.")

        abs_effects = tuple(
            abs(ak_statistics.block_effect(block, scale=self._statistics.effect_scale))
            for block in submission.blocks)

        promoted = None
        observed_magnitude = None
        observed_direction = None
        if definition.control_id == controls.CONTROL_HISTORICAL_WIN_REPLAY:
            # "It MUST promote", DERIVED. `rank_key()` is called rather than
            # `speed_rank_admissible` read, so promotion means the same thing here
            # as it does to the ranking loop that would have banked the win.
            promoted = bool(
                _rank_key_or_none(verdict) is not None
                and verdict.effect_resolution == api.EFFECT_IMPROVEMENT)
            if effect is not None:
                observed_magnitude = abs(effect.value)
                improved = (effect.value > 0) == (effect.metric_direction
                                                  == "higher_better")
                observed_direction = (effect.metric_direction if improved
                                      else _opposite_direction(effect.metric_direction))

        return controls.ControlObservation(
            control_id=definition.control_id,
            ran=True,
            verdict=verdict,
            abs_effects=abs_effects,
            promoted=promoted,
            observed_magnitude=observed_magnitude,
            observed_direction=observed_direction,
            evidence_ref=(None if effect is None else effect.raw_samples_ref),
            notes=(f"runner={self.runner_id}",
                   f"fixture={fixture.fixture_id}",
                   f"order_seed={submission.window.order_seed[:16]}",
                   f"blocks={len(submission.blocks)}"),
        )


# =============================================================================
# The sweep — the caller that closes the seed-rotation gap
# =============================================================================

@dataclass(frozen=True)
class SeedAssignment:
    """One control's seed for one rotation epoch. Derived, recorded, never chosen."""

    control_id: str
    epoch: int
    seed: str

    def to_dict(self) -> dict:
        return {"control_id": self.control_id, "epoch": self.epoch, "seed": self.seed}


@dataclass(frozen=True)
class SweepResult:
    """Everything one sweep produced, including the reasons it produced nothing.

    `panel_result` is `None` when the sweep did not run — and the ONLY way to a
    ranking licence is a `controls.ControlPanelResult`, so a blocked sweep cannot
    be mistaken for a clean one by a caller that forgot to look at
    `rotation_check`.
    """

    seeds: tuple
    rotation_check: schemas.Check
    observations: tuple
    panel_result: Optional[controls.ControlPanelResult]
    blocked_reason: Optional[str] = None

    @property
    def may_rank(self) -> bool:
        """False unless a panel exists AND it says so. Never inferred."""
        return self.panel_result is not None and self.panel_result.may_rank

    def seed_for(self, control_id: str) -> str:
        for assignment in self.seeds:
            if assignment.control_id == control_id:
                return assignment.seed
        raise KeyError(control_id)

    def to_dict(self) -> dict:
        return {
            "seeds": [s.to_dict() for s in self.seeds],
            "rotation_check": {"outcome": self.rotation_check.outcome,
                               "reasons": list(self.rotation_check.reasons)},
            "observation_count": len(self.observations),
            "panel_result": (None if self.panel_result is None
                             else self.panel_result.to_dict()),
            "blocked_reason": self.blocked_reason,
            "may_rank": self.may_rank,
        }


class ControlSweep:
    """Runs one control sweep, in the order the protocol's own clauses require.

    1. **Check the rotation schedule first.** *"Confirmation shapes and control
       seeds rotate on the schedule declared in the evaluator bundle"*, and design
       §12 calls a never-rotated holdout *"an evaluator coverage defect, not a
       tolerable simplification"*. `SeedRotationSchedule.check_rotation()` had no
       caller in the tree; this is it, and it runs BEFORE the sweep because a
       sweep on a stale holdout is not evidence that becomes valid once the epoch
       is bumped afterwards.
    2. **Derive one seed per control** from `ControlBundle.seed_for()` — also
       previously uncalled — and record the assignment.
    3. Run the five through `ControlHarness.run_all`, which now takes the campaign
       seed and window count and derives the same per-control seeds.
    4. Evaluate through `ControlHarness.evaluate`, which is the only thing that
       can mint a `ControlPanelResult`.

    **What `check_rotation` can and cannot see.** It compares `windows_completed`
    with `last_rotation_epoch`, and BOTH are supplied by the caller — which is the
    party being gated. Worse, the per-control seeds are derived from
    `windows_completed` too, so a caller who reports the matching pair
    (`last_rotation_epoch == windows_completed // rotate_every_windows`) passes by
    construction, every time. The failure the check exists to catch — *"a campaign
    could run for its whole life on one holdout"* — is reached by freezing
    `windows_completed`, and the check reads PASS throughout, because a frozen
    clock is consistent with itself. So the sweep keeps the one piece of state the
    gated party cannot supply: a ledger of the window counts it has actually
    served. A repeat or a rewind is refused. See `RotationLedgerViolation`.
    """

    def __init__(self, *, harness: controls.ControlHarness, campaign_seed: str) -> None:
        if not isinstance(harness, controls.ControlHarness):
            raise PipelineNotWired("ControlSweep requires a controls.ControlHarness")
        if not hasattr(harness.runner, "open_window"):
            raise PipelineNotWired(
                "ControlSweep requires a runner that can have a window opened on it "
                "(ExecutedControlRunner.open_window); a runner that cannot be told which "
                "window is live has no claim receipt to attest, and skipping the call for "
                "runners that lack it would be exactly the fail-open this refuses")
        _require_nonempty_str(campaign_seed, "campaign_seed")
        self.harness = harness
        self.campaign_seed = campaign_seed
        #: Window counts this sweep has served, in the order it served them. The
        #: rotation check's only caller-independent input.
        self._windows_served: list = []

    def seed_ledger(self, *, windows_completed: int) -> tuple:
        epoch = self.harness.bundle.seed_rotation.epoch_for(windows_completed)
        plan = self.harness.seed_plan(campaign_seed=self.campaign_seed,
                                      windows_completed=windows_completed)
        return tuple(
            SeedAssignment(control_id=cid, epoch=epoch, seed=plan[cid])
            for cid in controls.CONTROL_IDS)

    def run(self, *, run_context: controls.ControlRunContext,
            context: controls.ControlContext,
            window: api.WindowAttestations,
            aa_cadence: schemas.Check,
            windows_completed: int,
            last_rotation_epoch: int,
            escalation: Optional[controls.OperatorEscalation] = None,
            pinned_definitions_digest: Optional[str] = None,
            pinned_campaign_digest: Optional[str] = None) -> SweepResult:
        # The ledger first: a repeated or rewound window count is refused before
        # anything derives a seed from it. Raising rather than returning a blocked
        # SweepResult is deliberate — a caller that reuses a window count is not a
        # campaign whose holdout is stale, it is a campaign whose clock is wrong,
        # and a blocked result would be journaled as an ordinary rotation problem.
        if isinstance(windows_completed, bool) or not isinstance(windows_completed, int) \
                or windows_completed < 0:
            raise ValueError("windows_completed must be a non-negative int")
        if self._windows_served and windows_completed <= self._windows_served[-1]:
            raise RotationLedgerViolation(
                f"this sweep has already served window count {self._windows_served[-1]} and "
                f"was asked to run at {windows_completed}; the window count must advance. "
                "check_rotation() compares two numbers the caller supplies and the control "
                "seeds are derived from one of them, so a frozen counter passes the "
                "rotation check forever while every window reuses one holdout.")
        self._windows_served.append(windows_completed)

        seeds = self.seed_ledger(windows_completed=windows_completed)
        rotation = self.harness.bundle.seed_rotation.check_rotation(
            windows_completed=windows_completed,
            last_rotation_epoch=last_rotation_epoch)
        if rotation.outcome != schemas.PASS:
            return SweepResult(
                seeds=seeds, rotation_check=rotation, observations=(),
                panel_result=None,
                blocked_reason=(
                    "the control seeds are not on their declared rotation schedule, so "
                    "this sweep was not run: " + "; ".join(rotation.reasons)))

        # The live window is opened HERE and closed unconditionally, so no control
        # can be assembled between sweeps under a receipt from a closed one.
        self.harness.runner.open_window(window_id=run_context.window_id, window=window)
        try:
            observations = self.harness.run_all(
                run_context=run_context, historical=context.historical,
                campaign_seed=self.campaign_seed, windows_completed=windows_completed)
        finally:
            self.harness.runner.close_window()
        panel_result = self.harness.evaluate(
            observations=observations, context=context, aa_cadence=aa_cadence,
            escalation=escalation,
            pinned_definitions_digest=pinned_definitions_digest,
            pinned_campaign_digest=pinned_campaign_digest)
        return SweepResult(
            seeds=seeds, rotation_check=rotation, observations=observations,
            panel_result=panel_result, blocked_reason=panel_result.blocked_reason)


# =============================================================================
# The calibration join — the A/A arm reaching statistics.CalibrationSolve
# =============================================================================

def calibration_material(runner: ExecutedControlRunner,
                         result: SweepResult) -> dict:
    """`{"aa_blocks": (...), "neutral_blocks": (...)}` from the sweeps that ran.

    Taken from the submissions the A/A and neutral controls were ACTUALLY
    measured on, matched back by the fixtures' candidate ids. φ is *"estimated
    from the A/A control"* and the neutral control's dispersion is compared
    against φ; assembling either from blocks the controls did not run on would
    calibrate a measurement that never happened, and `solve_calibration` would
    return an accepted solve regardless — so the substitution would never surface
    as an error. It raises instead.

    It pools every ADMITTED submission the RUNNER has made, not only the last
    sweep's, and that is deliberate: the A/A control *"runs periodically on its
    declared cadence, not once per campaign"*, and one window's arm is far shorter
    than a declared `calibration_block_count`. φ is calibrated over the
    accumulated A/A history, which is what makes a cadence a calibration mechanism
    rather than a formality.

    Three things it refuses, each of which used to pass:

      * **A sweep that produced no panel.** `result` was type-checked and then
        never read, so the docstring's claim that it stopped material being taken
        out of a BLOCKED sweep was false: an off-schedule rotation returned
        `panel_result=None` and the A/A blocks came out anyway. It is read now.
      * **Submissions whose control did not run.** A reduction refused as not
        search-grade — below `B_min`, undeclared extension — left its blocks in
        `runner.submissions`, and they were pooled into φ. `admitted_submissions`
        holds only the submissions whose observation carries a verdict.
      * **Relabelled material.** See `CalibrationMaterialRelabelled`: twelve
        rotation epochs over a static fixture are 120 unit ids and 10
        measurements, and nothing downstream compares the samples.
    """
    if not isinstance(runner, ExecutedControlRunner):
        raise TypeError("calibration_material takes an ExecutedControlRunner")
    if not isinstance(result, SweepResult):
        raise TypeError("calibration_material takes a SweepResult")
    if result.panel_result is None:
        raise SweepNotLicensed(
            "this sweep produced no control panel and therefore no ranking licence "
            f"({result.blocked_reason}); its A/A arm is still in the runner's submission "
            "list, and calibrating phi out of a window the campaign was not allowed to "
            "proceed in would launder a blocked sweep into the noise floor every later "
            "candidate is judged against.")
    material = {}
    for key, control_id in (("aa_blocks", controls.CONTROL_AA),
                            ("neutral_blocks", controls.CONTROL_NEUTRAL)):
        fixture = runner.fixtures.for_definition(_definition_for(control_id))
        blocks = tuple(
            block
            for submission in runner.admitted_submissions
            if submission.request.candidate_id == fixture.candidate_id
            for block in submission.blocks)
        if not blocks:
            raise CalibrationMaterialMissing(
                f"the {control_id} control produced no paired blocks that ran, so "
                f"there is no {key} to calibrate from. The calibration block's phi is "
                "estimated from the A/A control and its consistency check is a property "
                "of the neutral control; a solve assembled without them calibrates a "
                "measurement that never happened.")
        _refuse_relabelled_material(blocks, label=key, control_id=control_id)
        material[key] = blocks
    return material


def _refuse_relabelled_material(blocks: Sequence[ak_statistics.PairedBlock], *,
                                label: str, control_id: str) -> None:
    """Raise when two distinct `unit_id`s carry byte-identical sample arms.

    The count of calibration blocks is load-bearing — `estimate_noise_floor`
    refuses below the declared `calibration_block_count` — and it is the ONLY
    thing checked. Nothing downstream compares two blocks' samples, so a producer
    that re-serves one recorded arm under a fresh unit id manufactures the count
    out of nothing. This is the check that makes the unit id mean what
    `PairedBlock` says it means.
    """
    seen: dict = {}
    for block in blocks:
        material = (block.anchor_samples, block.candidate_samples)
        first = seen.get(material)
        if first is None:
            seen[material] = block.unit_id
            continue
        how = (f"under two unit ids ({first!r} and {block.unit_id!r})"
               if first != block.unit_id
               else f"twice under the one unit id {first!r}")
        raise CalibrationMaterialRelabelled(
            f"the {control_id} control's {label} carry the same measured arms {how}. A "
            "unit id is 'the measurement-material unit (shape, seed) the block used'; "
            "re-serving one recorded arm — under a rotated seed, or by pooling a window "
            "with itself — relabels a measurement, it does not take a second one, and phi "
            "is 'estimated over AT LEAST the declared count' of measurements, not of "
            "labels. (Two independent timing arms are not byte-identical; the same clause "
            "that refuses a zero noise floor because 'the instrument did not vary' refuses "
            "this.)")


def pool_calibration_material(materials: Sequence[dict]) -> dict:
    """Pool per-window calibration material into one A/A history.

    This is how a declared `calibration_block_count` is legitimately reached. The
    A/A control *"runs periodically on its declared cadence, not once per
    campaign"*, and one window's arm is a handful of blocks — but the windows'
    arms have to be DIFFERENT MEASUREMENTS, which means each window resolves its
    own freshly measured fixture set under its own claim. One runner cannot supply
    them, because its fixture set is pinned; so the pooling happens here, over the
    materials several windows produced, and the relabelling refusal is applied
    across the whole pool rather than within each window.

    A pool assembled from one window's material repeated is refused for the same
    reason it is refused inside a window: the count is of measurements, not of
    labels.
    """
    materials = tuple(materials)
    if not materials:
        raise CalibrationMaterialMissing(
            "pool_calibration_material() over zero windows: phi is estimated from the A/A "
            "control, and no windows is no A/A control")
    pooled = {}
    for key, control_id in (("aa_blocks", controls.CONTROL_AA),
                            ("neutral_blocks", controls.CONTROL_NEUTRAL)):
        blocks = []
        for index, material in enumerate(materials):
            if not isinstance(material, dict) or key not in material:
                raise CalibrationMaterialMissing(
                    f"window {index} contributed no {key!r}; pool the dicts "
                    "calibration_material() returns, not something shaped like them")
            blocks.extend(material[key])
        blocks = tuple(blocks)
        _refuse_relabelled_material(blocks, label=key, control_id=control_id)
        pooled[key] = blocks
    return pooled


def _definition_for(control_id: str) -> controls.ControlDefinition:
    for definition in controls.CONTROL_DEFINITIONS:
        if definition.control_id == control_id:
            return definition
    raise KeyError(control_id)  # pragma: no cover - CONTROL_IDS is closed


def build_calibration_inputs(*, runner: ExecutedControlRunner,
                             result: SweepResult,
                             binding: CampaignBinding,
                             campaign_seed: str,
                             campaign_controls: api.CampaignControls,
                             stopping_rule: Any,
                             construction: Any,
                             effect_scale: str,
                             hypothesis: str,
                             margin: float,
                             anchor_calibration_values: tuple,
                             samples_ref: str,
                             owning_rep_rule: Any = None,
                             material: Optional[dict] = None):
    """Assemble the `statistics.CalibrationInputs` the solve consumes.

    This module supplies the MATERIAL and nothing else: the solve order, φ,
    `B_min`, the alpha budgets and the anchor-gate band are `statistics.py`'s, and
    `controls.run_calibration_block()` is the one door into them. A second solve
    here would be the failure this project keeps paying for — both
    implementations would keep returning an accepted calibration, so the drift
    between them would never surface as an error.

    The cell (`backend`, `phase`, `cell_class`) comes from the CampaignBinding the
    controls were submitted under, not from a parameter: *"values calibrated under
    a different host state, backend, phase or cell class MUST NOT be reused"*, and
    a cell supplied separately from the material is a cell that can disagree with
    it.
    """
    # `material=None` means "this window's own", and it is still the strict path:
    # `calibration_material` refuses a blocked sweep, a control that did not run,
    # and relabelled arms. A supplied `material` comes from
    # `pool_calibration_material`, which applies the same refusals across the
    # pool. Neither branch is a fallback — there is no lenient one to fall back to.
    if material is None:
        material = calibration_material(runner, result)
    else:
        if result.panel_result is None:
            raise SweepNotLicensed(
                "the sweep this calibration is being built for produced no control panel "
                f"({result.blocked_reason}); pooled material does not license a window "
                "that was blocked")
        for key, control_id in (("aa_blocks", controls.CONTROL_AA),
                                ("neutral_blocks", controls.CONTROL_NEUTRAL)):
            if not material.get(key):
                raise CalibrationMaterialMissing(
                    f"the supplied material carries no {key!r}; phi is estimated from the "
                    f"{control_id} control")
            _refuse_relabelled_material(tuple(material[key]), label=key,
                                        control_id=control_id)
    return ak_statistics.CalibrationInputs(
        backend=binding.backend,
        phase=binding.phase,
        cell_class=binding.cell_class,
        campaign_seed=campaign_seed,
        controls=campaign_controls,
        stopping_rule=stopping_rule,
        construction=construction,
        effect_scale=effect_scale,
        metric_direction=binding.metric_direction,
        hypothesis=hypothesis,
        margin=margin,
        aa_blocks=material["aa_blocks"],
        neutral_blocks=material["neutral_blocks"],
        anchor_calibration_values=tuple(anchor_calibration_values),
        samples_ref=samples_ref,
        owning_rep_rule=owning_rep_rule,
    )


# =============================================================================
# Self-audit — "one code path", proved from the AST rather than promised
# =============================================================================

#: Evaluator entry points that would constitute a SECOND way to obtain a verdict.
#: `ExecutedControlRunner` must reach the gate only through
#: `CandidatePipeline.evaluate`, so none of these may appear inside its body.
_SECOND_PATH_CALLS = frozenset({
    "dispatch", "compute_verdict", "run_gates", "reduce_blocks", "reduce",
    "check_void_conditions", "check_preconditions", "evaluate_search_grade",
    "build_evaluation_event", "rank_candidates", "_derive",
})

#: The one call. Exactly one call site, because two call sites are two code paths
#: even when both call the same method — the second one is where a special case
#: for controls would go.
_SINGLE_PATH_CALL = "evaluate"


def audit_single_evaluation_path(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's AST that a control has ONE way to be evaluated.

    Structural rather than documented. The class that runs controls may not call
    the dispatcher, the reducer, the verdict computer or the search-grade
    evaluator; it may call `.evaluate(...)` and it may do so exactly once. If a
    later edit adds `if definition.control_id == ...: <other path>`, the second
    call site is what this catches — the branch itself is invisible to any test
    that only checks the observations came out right.

    COULD_NOT_CHECK when the source cannot be read or parsed: an unaudited module
    is not an audited one.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (f"could not parse the module source: {exc}",))

    target = None
    module_functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ExecutedControlRunner":
            target = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_functions.setdefault(node.name, node)
    if target is None:
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("ExecutedControlRunner was not found in the parsed source; an audit that "
             "cannot find its subject has not audited it",))

    # The class body is not the whole reachable surface. A second route to a
    # verdict placed in a module-level helper and called from `run_control` is
    # invisible to an audit that walks only the ClassDef — which is precisely
    # where someone avoiding this check would put it. So the audit follows the
    # module-level functions the class calls, transitively, and says so when it
    # cannot resolve one.
    reasons = []
    evaluate_calls = 0
    scanned: set = set()
    frontier = [("ExecutedControlRunner", target)]
    while frontier:
        owner, node_root = frontier.pop()
        if owner in scanned:
            continue
        scanned.add(owner)
        for node in ast.walk(node_root):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name in _SECOND_PATH_CALLS:
                reasons.append(
                    f"{owner} calls {name!r} on line {node.lineno}: that is a second route "
                    "to a verdict, and a control that runs down a different code path "
                    "proves nothing about the path that matters")
            if name == _SINGLE_PATH_CALL:
                evaluate_calls += 1
            # A bare `helper(...)` naming a module-level def is reachable code.
            if isinstance(func, ast.Name) and func.id in module_functions \
                    and func.id not in scanned:
                frontier.append((func.id, module_functions[func.id]))
    if evaluate_calls != 1:
        reasons.append(
            f"ExecutedControlRunner contains {evaluate_calls} call(s) to "
            f"{_SINGLE_PATH_CALL!r}; there must be exactly one, because a second call site "
            "is where a special case for controls would live even when both sites call the "
            "same method")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


#: Field-name tokens that would let a submission tell the pipeline what it is, or
#: hand it the number it is supposed to compute.
_SUBMISSION_BANNED_TOKENS = ("control", "kind", "fixture", "effect", "estimate",
                             "synthetic", "replay")


def audit_submission_carries_no_control_marker(
        submission_type: Any = CandidateSubmission) -> schemas.Check:
    """Prove a submission type cannot tell the pipeline it is a control.

    Read off the dataclass's own fields rather than asserted in prose, so adding
    `is_control: bool = False` — the natural, well-meant next edit — fails this
    instead of quietly reintroducing the branch the whole design forbids.

    `submission_type` is a parameter so the guard has a failing case to be tested
    against: a checker with no reachable FAIL branch is a checker nobody has
    shown to work.
    """
    try:
        declared = dataclass_fields(submission_type)
    except TypeError:
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"{submission_type!r} is not a dataclass, so its fields cannot be read; an "
             "unreadable submission type is not an audited one",))
    offending = [
        f.name for f in declared
        if any(token in f.name.lower() for token in _SUBMISSION_BANNED_TOKENS)
    ]
    if offending:
        return schemas.Check(
            schemas.FAIL,
            (f"{getattr(submission_type, '__name__', submission_type)!r} carries "
             f"{offending}; the pipeline must not be able to tell a control from a "
             "candidate, and must not be handed an effect estimate it is supposed to "
             "compute",))
    return schemas.Check(schemas.PASS)
