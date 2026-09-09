"""Compact prospective preparation inputs; no artifact or resource I/O.

The original statistical declaration and protocol reference remain explicit
inputs. Pair membership, retry process identities and scheduler enrollment are
derived before selection, never reconstructed from measured outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import experiment_plan as ep, planned_serving as ps, scheduling
from . import serving_preparation as prep

SCHEMA = "epyc.autokernel.serving_preparation_startup.v1"
PROTOCOL_FIELDS = {"protocol_ref", "protocol_status", "policy_snapshot",
                   "required_witnesses", "comparison_kind", "estimand"}


@dataclass(frozen=True)
class PreparationStartupConfiguration:
    entries: tuple[Mapping[str, Any], ...]
    schema: str = SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> PreparationStartupConfiguration:
        row = prep._exact(value, {"schema", "entries"}, "preparation startup")
        if row["schema"] != SCHEMA or not isinstance(row["entries"], (list, tuple)):
            raise prep.PreparationRefused("preparation startup schema/entries differ")
        if not row["entries"] or len(prep._bytes(row)) > prep.MAX_CONFIGURATION_BYTES:
            raise prep.PreparationRefused("compact preparation configuration is empty or oversized")
        entries = []
        for value in row["entries"]:
            item = prep._exact(value, {"declaration", "protocol", "submitted_at",
                "estimated_duration_seconds"}, "preparation startup entry")
            declaration = prep.ServingPreparationDeclaration.from_dict(item["declaration"])
            protocol = prep._exact(item["protocol"], PROTOCOL_FIELDS, "preparation protocol")
            # ExperimentPlan owns protocol grammar. This is not ratification or
            # permission to bypass its original unknown/unsupported witnesses.
            prep._positive(item["estimated_duration_seconds"], "estimated_duration_seconds")
            item.update(declaration=declaration.to_dict(), protocol=protocol)
            entries.append(prep._freeze(item))
        return cls(tuple(entries))

    def to_dict(self) -> dict:
        return {"schema": self.schema, "entries": prep._plain(self.entries)}


def materialize(configuration: PreparationStartupConfiguration, *, resolved: Any,
                anchors: Any, execution_inputs: Mapping[str, Any], profiles: Mapping,
                scheduler_config: scheduling.SchedulerConfig, loaded_instrument: Mapping,
                expected_epoch: str) -> tuple[prep.CalibrationPreparationRequest, ...]:
    """Construct the complete fixed pool and all declared retries without I/O."""
    if type(configuration) is not PreparationStartupConfiguration:
        raise prep.PreparationRefused("preparation needs its concrete startup configuration")
    configuration = PreparationStartupConfiguration.from_dict(configuration.to_dict())
    declarations = [prep.ServingPreparationDeclaration.from_dict(item["declaration"])
                    for item in configuration.entries]
    total = sum(item.statistics.controls.calibration_block_count *
                (2 if item.neutral_pair is not None else 1) * item.retry_policy.max_attempts
                for item in declarations)
    if total > scheduler_config.campaign_attempt_cap:
        raise prep.PreparationRefused("preparation expansion exceeds campaign attempt budget")
    if len({item.declaration_id for item in declarations}) != len(declarations):
        raise prep.PreparationRefused("preparation startup repeats a declaration")
    targets = {prep._digest(target.to_dict()): target for target in resolved.targets}
    requests = []
    for item, declaration in zip(configuration.entries, declarations):
        target = targets.get(declaration.target_revision)
        anchor = anchors.recipes.get(declaration.target_revision)
        execution = execution_inputs.get(declaration.target_revision)
        if (target is None or target.status != "ready" or anchor is None
                or declaration.campaign_id != resolved.campaign_id
                or declaration.aa_pair.anchor.to_dict() != anchor.to_dict()
                or declaration.frame["metric"] != target.execution.metric
                or declaration.frame["metric_direction"] != {
                    "higher": "higher_better", "lower": "lower_better"
                }[target.execution.metric_direction]
                or declaration.frame["epoch"] != expected_epoch
                or execution is None
                or len(execution.prompt_manifest.prompts) != anchor.template.np
                or execution.instrument_id != loaded_instrument["identity_sha256"]
                or declaration.prompt_manifest_digest != prep._digest(execution.prompt_manifest.to_dict())
                or declaration.max_stage_seconds != execution.max_stage_seconds
                or declaration.teardown_seconds != execution.teardown_seconds):
            raise prep.PreparationRefused("preparation differs from enrolled execution inputs")
        profile = profiles.get(declaration.target_revision)
        if profile is not None and profile.quant != declaration.frame["quant"]:
            raise prep.PreparationRefused("preparation quant differs from enrolled profile")
        claims, capacity = declaration.resources, scheduler_config.capacity
        if (claims.physical_region_fraction > capacity.physical_region_fraction
                or not set(claims.gpu_devices) <= set(capacity.gpu_devices)
                or claims.memory_reservation_bytes > capacity.memory_reservation_bytes
                or (target.execution.backend == "cpu" and claims.gpu_devices)
                or (target.execution.backend == "gpu" and not claims.gpu_devices)
                or declaration.max_stage_seconds > scheduler_config.max_stage_seconds
                or item["estimated_duration_seconds"] > declaration.max_stage_seconds):
            raise prep.PreparationRefused("preparation exceeds installed capacity or stage budget")
        for kind in ("aa", "neutral"):
            pair = declaration.aa_pair if kind == "aa" else declaration.neutral_pair
            if pair is None:
                continue
            n = declaration.statistics.controls.calibration_block_count
            for attempt in range(declaration.retry_policy.max_attempts):
                order = prep.st.OrderSchedule.derive(
                    campaign_seed=declaration.statistics.campaign_seed,
                    candidate_id=f"{declaration.declaration_id}:{kind}",
                    base_blocks=n, attempt=attempt)
                for index in range(n):
                    material = "calibration-material:" + prep._digest({
                        "declaration": declaration.digest, "kind": kind, "block": index})
                    identity = prep._digest({"material": material, "attempt": attempt})
                    arm_order = (("anchor", "candidate") if order.order_for(index) ==
                                 prep.st.ORDER_ANCHOR_FIRST else ("candidate", "anchor"))
                    units = [{"unit_id": f"{identity}:{arm}", "arm": arm,
                              "process_id": f"process:{identity}:{arm}",
                              "expected_prompt_ids": [prompt.prompt_id for prompt in
                                                      execution.prompt_manifest.prompts],
                              "order_index": position, "pair_id": 0}
                             for position, arm in enumerate(arm_order)]
                    plan = ep.ExperimentPlan.from_dict({
                        "schema": ep.PLAN_SCHEMA_V2, "plan_id": "calibration:" + identity,
                        "campaign_id": declaration.campaign_id,
                        "target_revision": declaration.target_revision, "epoch": expected_epoch,
                        "instrument_class": "serving", "category": "CANDIDATE",
                        "phase": "observation", "record_class": "observation",
                        "intended_use": "explore", **prep._plain(item["protocol"]),
                        "metric": declaration.frame["metric"], "metric_direction": {
                            "higher_better": "higher", "lower_better": "lower"
                        }[declaration.frame["metric_direction"]],
                        "estimator_id": declaration.frame["estimator_id"], "unit": "process",
                        "changed_factors": [], "loaded_instrument": prep._plain(loaded_instrument),
                        "anchor_identity": ps.arm_identity(pair.anchor.template, pair.anchor,
                                                           loaded_instrument=loaded_instrument),
                        "candidate_identity": ps.arm_identity(pair.candidate.template, pair.candidate,
                                                              loaded_instrument=loaded_instrument),
                        "expected_units": units,
                        "stopping": {"kind": "fixed_n", "n_per_arm": 1, "paired": True},
                        "calibration_ref": None, "continuation_allowed": False})
                    production = "production" in target.enrolled_as
                    stage = scheduling.StageProposal(
                        "calibration:" + identity, item["submitted_at"], target.execution.backend,
                        declaration.target_revision, declaration.target_revision,
                        "production:" + declaration.target_revision if production else None,
                        production, "seed:" + declaration.target_revision
                        if "seed" in target.enrolled_as else None,
                        "calibration", item["estimated_duration_seconds"], claims, True,
                        declaration.digest, "calibration", claims.physical_region_fraction == 1.0,
                        (), False)
                    requests.append(prep.CalibrationPreparationRequest(
                        declaration, kind, plan, execution.prompt_manifest, ({
                            "block_index": index, "material_unit_id": material,
                            "stratum": declaration.statistics.split_rule.assign(material),
                            "anchor_unit_id": f"{identity}:anchor",
                            "candidate_unit_id": f"{identity}:candidate"},), stage, attempt))
    result = prep.bounded_requests(requests, max_requests=scheduler_config.campaign_attempt_cap)
    prep.validate_pool_membership(result)
    return result
