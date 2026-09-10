"""Closed scheduling inputs for the existing serial loop owner."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import scheduling

MANIFEST_SCHEMA = "epyc.autokernel.serial_scheduler_manifest.v1"
VALID_COMPARISONS = frozenset({"kept", "keep_candidate", "measured_null",
                               "source_validation_passed"})
INVALID_OUTCOMES = frozenset({
    "measurement_invalid", "runtime_observed", "runtime_refused", "refused_at_formation",
    "stopped_before_reschedule", "stopped_mid_formation", "superseded",
    "source_validation_failed", "source_validation_pending",
})
FAILED_OUTCOMES = frozenset({"bench_failed", "planner_transient"})
INTERVAL_SCHEMA = "epyc.autokernel.direct_held_intervals.v1"
REFERENCE_SCHEMA = "epyc.autokernel.direct_held_reference.v1"
COST_FORECAST_POLICY = "original-held-stage-p75-last8.v1"
COST_SAMPLE_LIMIT = 8


class SerialSchedulingRefused(ValueError):
    pass


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SerialSchedulerManifest:
    scheduler_id: str
    config: scheduling.SchedulerConfig
    proposals: Mapping[str, scheduling.StageProposal]
    schema: str = MANIFEST_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "SerialSchedulerManifest":
        if not isinstance(value, Mapping) or set(value) != {
                "schema", "scheduler_id", "config", "targets"}:
            raise SerialSchedulingRefused("serial scheduler manifest shape is invalid")
        if value["schema"] != MANIFEST_SCHEMA:
            raise SerialSchedulingRefused("serial scheduler manifest schema is unsupported")
        scheduler_id = value["scheduler_id"]
        targets = value["targets"]
        if not isinstance(scheduler_id, str) or not scheduler_id or "\0" in scheduler_id:
            raise SerialSchedulingRefused("serial scheduler_id must be nonempty text")
        if not isinstance(targets, Mapping) or not 0 < len(targets) <= 64:
            raise SerialSchedulingRefused("serial scheduler targets must contain 1-64 entries")
        proposals: dict[str, scheduling.StageProposal] = {}
        proposal_ids: set[str] = set()
        for selected_id, proposal in targets.items():
            if not isinstance(selected_id, str) or not selected_id or "\0" in selected_id:
                raise SerialSchedulingRefused("scheduled selected_id must be nonempty text")
            parsed = scheduling.StageProposal.from_dict(proposal)
            if parsed.proposal_id in proposal_ids:
                raise SerialSchedulingRefused("serial scheduler proposal IDs must be unique")
            proposal_ids.add(parsed.proposal_id)
            proposals[selected_id] = parsed
        return cls(scheduler_id=scheduler_id,
                   config=scheduling.SchedulerConfig.from_dict(value["config"]),
                   proposals=MappingProxyType(proposals))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "scheduler_id": self.scheduler_id,
                "config": self.config.to_dict(),
                "targets": {key: value.to_dict() for key, value in self.proposals.items()}}

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


def validate_target_bindings(manifest: SerialSchedulerManifest,
                             bindings: Mapping[str, Mapping[str, str]]) -> None:
    """Bind policy proposals to target facts rederived by the serial owner."""
    if set(bindings) != set(manifest.proposals):
        raise SerialSchedulingRefused("scheduled targets differ from executable target roster")
    for selected_id, expected in bindings.items():
        if not isinstance(expected, Mapping) or set(expected) != {
                "target_revision", "alias_identity", "backend", "eligibility_ref"}:
            raise SerialSchedulingRefused("serial target binding shape is invalid")
        proposal = manifest.proposals[selected_id]
        if (proposal.target_revision != expected["target_revision"]
                or proposal.alias_identity != expected["alias_identity"]
                or proposal.backend != expected["backend"]
                or proposal.eligibility_ref != expected["eligibility_ref"]
                or not proposal.eligible or proposal.stage_class != "search"):
            raise SerialSchedulingRefused(
                f"scheduled proposal differs from enrolled target {selected_id}")


def select_target(manifest: SerialSchedulerManifest, state: scheduling.SchedulerState,
                  selected_ids: Sequence[str], *, now: float, stage_number: int,
                  scope_previews: Mapping[str, dict] | None = None,
                  validation_ids: frozenset[str] = frozenset(),
                  duration_forecasts: Mapping[str, dict] | None = None
                  ) -> tuple[scheduling.SchedulerState, scheduling.Selection, int]:
    if type(stage_number) is not int or stage_number < 0:
        raise SerialSchedulingRefused("serial stage number must be nonnegative")
    if scope_previews is not None and set(scope_previews) != set(selected_ids):
        raise SerialSchedulingRefused("CPU scope previews differ from available selected targets")
    if not validation_ids <= set(selected_ids):
        raise SerialSchedulingRefused("validation targets differ from available selected targets")
    indexed = []
    for index, selected_id in enumerate(selected_ids):
        proposal = manifest.proposals[selected_id]
        identity = {"manifest": manifest.digest, "selected_id": selected_id,
                    "stage_number": stage_number}
        if scope_previews is not None:
            from .cpu_screen import scoped_proposal
            preview = scope_previews[selected_id]
            proposal = scoped_proposal(proposal, preview)
            identity["cpu_scope"] = _digest(preview)
        if selected_id in validation_ids:
            proposal = replace(proposal, stage_class="validation",
                               reservation_kind="validation")
            identity["source_validation"] = True
        forecast = (duration_forecasts or {}).get(selected_id)
        if forecast is not None:
            # A serial-owner estimate, never a new bound, grant, weight or grade.
            # Another stage (e.g. source validation) must retain its own estimate.
            if forecast["stage_class"] == proposal.stage_class:
                duration = _number(forecast["estimated_duration_seconds"], "duration forecast")
                if not 0 < duration <= manifest.config.max_stage_seconds:
                    raise SerialSchedulingRefused("duration forecast is outside the original stage bound")
                proposal = replace(proposal, estimated_duration_seconds=duration)
                identity["duration_forecast"] = _digest(forecast)
        indexed.append((index, replace(proposal, proposal_id=_digest(identity), submitted_at=now)))
    proposals = tuple(proposal for _index, proposal in indexed)
    next_state, selection = scheduling.select_stage(
        manifest.config, state, proposals, now=now)
    if selection.status == "complete":
        return next_state, selection, -1
    if selection.status != "selected" or selection.proposal is None:
        raise SerialSchedulingRefused(
            f"scheduler did not select executable work: {selection.status}: "
            f"{'; '.join(selection.reasons)}")
    by_proposal = {proposal.proposal_id: index for index, proposal in indexed}
    return next_state, selection, by_proposal[selection.proposal.proposal_id]


def cost_scope(*, binding, anchor, cor_anchor, runtime_recipe, geometry,
               preparation, proposal, state):
    """Compatibility for an operational whole-stage forecast, not scientific reuse.

    The caller supplies original continuation/input facts. No model reads, guessed
    initial source identity, cross-recipe statistics, or cross-scope gain ranking.
    """
    return _digest({"binding": binding, "anchor": anchor, "cor_anchor": cor_anchor,
        "runtime_recipe": runtime_recipe, "geometry": geometry, "preparation": preparation,
        "target_revision": proposal.target_revision, "alias_identity": proposal.alias_identity,
        "backend": proposal.backend, "stage_class": proposal.stage_class,
        "reservation_kind": proposal.reservation_kind, "claims": proposal.estimated_claims.to_dict(),
        "accounting_epoch": state.accounting_epoch, "capacity_digest": state.capacity_digest})


def _cost_samples(history, selected_id, scope_digest):
    if history is None:
        return []
    _closed(history, {"policy", "targets"}, "serial cost history")
    if history["policy"] != COST_FORECAST_POLICY or not isinstance(history["targets"], dict) \
            or len(history["targets"]) > 64:
        raise SerialSchedulingRefused("serial cost history policy/size differs")
    row = history["targets"].get(selected_id)
    if row is None:
        return []
    _closed(row, {"scope_digest", "samples"}, "serial cost target")
    if row["scope_digest"] != scope_digest:
        return []
    samples = row["samples"]
    if not isinstance(samples, list) or not 0 < len(samples) <= COST_SAMPLE_LIMIT:
        raise SerialSchedulingRefused("serial cost samples exceed their bounded window")
    ids = set()
    for sample in samples:
        _closed(sample, {"selection_digest", "held_seconds"}, "serial cost sample")
        digest = sample["selection_digest"]
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(char not in "0123456789abcdef" for char in digest) or digest in ids \
                or _number(sample["held_seconds"], "original held duration") <= 0:
            raise SerialSchedulingRefused("serial cost sample identity/duration differs")
        ids.add(digest)
    return [dict(sample) for sample in samples]


def retain_cost_sample(history, selected_id, scope_digest, selection, receipts):
    """After NEW owning settlement only; keep at most eight samples/current target scope."""
    samples = _cost_samples(history, selected_id, scope_digest)
    sample = {"selection_digest": selection.digest,
              "held_seconds": receipts[-1].ended_at - receipts[0].started_at}
    if sample["held_seconds"] <= 0:
        raise SerialSchedulingRefused("original held duration must be positive")
    prior = next((row for row in samples if row["selection_digest"] == selection.digest), None)
    if prior is not None and prior != sample:
        raise SerialSchedulingRefused("original settled cost changed")
    if prior is None:
        samples.append(sample)
    targets = json.loads(json.dumps(history["targets"], allow_nan=False)) if history is not None else {}
    targets[selected_id] = {"scope_digest": scope_digest, "samples": samples[-COST_SAMPLE_LIMIT:]}
    return {"policy": COST_FORECAST_POLICY, "targets": targets}


def duration_forecast(history, selected_id, scope_digest, *, proposal, max_stage_seconds):
    samples = _cost_samples(history, selected_id, scope_digest)
    if not samples:
        return None
    durations = sorted(row["held_seconds"] for row in samples)
    # Nearest-rank empirical p75. This is a versioned scheduling heuristic, NOT a
    # confidence limit, future service guarantee, or measurement precision claim.
    duration = min(max_stage_seconds, durations[math.ceil(.75 * len(durations)) - 1])
    return {"policy": COST_FORECAST_POLICY, "scope_digest": scope_digest,
            "stage_class": proposal.stage_class, "estimated_duration_seconds": duration,
            "samples": samples}


def one_iteration_outcome(terminal: str, outcome_counts: Mapping[str, int]) -> str:
    if terminal == "stopped":
        return "failed"
    if terminal != "complete" or not isinstance(outcome_counts, Mapping) \
            or len(outcome_counts) != 1:
        raise SerialSchedulingRefused("scheduled child lacks one exact terminal outcome")
    ((status, count),) = outcome_counts.items()
    if count != 1:
        raise SerialSchedulingRefused("scheduled child must cover exactly one research iteration")
    if status in VALID_COMPARISONS:
        return "valid_comparison"
    if status in INVALID_OUTCOMES:
        return "invalid"
    if status in FAILED_OUTCOMES:
        return "failed"
    raise SerialSchedulingRefused(f"scheduled child outcome is unsupported: {status}")


def _closed(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SerialSchedulingRefused(f"{label} shape is invalid")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise SerialSchedulingRefused(f"{label} must be a finite number")
    return float(value)


def _component(value: Any) -> dict[str, Any]:
    row = _closed(value, {"context_id", "domain", "ownership_generation",
        "allocation_generation", "started_at", "ended_at", "device_id",
        "physical_claim_ids", "physical_region_fraction", "gpu_device_ids",
        "memory_reservation_bytes", "affinity_cores", "open", "close", "released"},
        "held component")
    domain = _closed(row["domain"], {"kind", "clock", "pid", "boot_id",
        "process_start_ticks", "error"}, "held component domain")
    if domain["kind"] != "direct_loop" or domain["clock"] != "monotonic" \
            or domain["error"] is not None or type(domain["pid"]) is not int \
            or type(domain["process_start_ticks"]) is not int \
            or not isinstance(domain["boot_id"], str) or not domain["boot_id"]:
        raise SerialSchedulingRefused("held component domain is unavailable")
    observations = []
    for observation_name in ("open", "close"):
        observation = _closed(row[observation_name], {"observed_at", "started_monotonic_s",
            "ended_monotonic_s", "owner_pid", "locks", "error", "status"},
            f"held component {observation_name}")
        if observation["status"] != "held" or observation["error"] is not None \
                or observation["owner_pid"] != domain["pid"] \
                or not isinstance(observation["locks"], list) or not observation["locks"]:
            raise SerialSchedulingRefused("held component lacks exact same-owner observations")
        for lock in observation["locks"]:
            lock = _closed(lock, {"path", "device", "inode", "path_unchanged", "owners",
                                  "same_holder"}, "held component lock")
            owner = (_closed(lock["owners"][0], {"pid", "kernel_row"},
                             "held component lock owner")
                     if isinstance(lock["owners"], list) and len(lock["owners"]) == 1
                     else {})
            if lock["path_unchanged"] is not True or lock["same_holder"] is not True \
                    or not isinstance(lock["owners"], list) or len(lock["owners"]) != 1 \
                    or owner.get("pid") != domain["pid"]:
                raise SerialSchedulingRefused("held component lock was not continuously owned")
        observations.append(observation)
    start = _number(row["started_at"], "held component start")
    end = _number(row["ended_at"], "held component end")
    if end <= start or row["released"] is not True \
            or row["ownership_generation"] != 1 or row["allocation_generation"] != 1:
        raise SerialSchedulingRefused("held component interval/release is invalid")
    for name in ("physical_claim_ids", "gpu_device_ids", "affinity_cores"):
        if not isinstance(row[name], list) or not all(
                isinstance(item, str) and item for item in row[name]):
            raise SerialSchedulingRefused(f"held component {name} is invalid")
    opened = [(item["path"], item["device"], item["inode"])
              for item in observations[0]["locks"]]
    closed = [(item["path"], item["device"], item["inode"])
              for item in observations[1]["locks"]]
    expected_claims = [f"{domain['boot_id']}:flock:{item[1]}:{item[2]}" for item in opened]
    expected_context = _digest({"domain": dict(domain), "started_at": start,
                                "locks": observations[0]["locks"]})
    if opened != closed or row["physical_claim_ids"] != expected_claims \
            or row["context_id"] != expected_context:
        raise SerialSchedulingRefused("held component original lock identity differs")
    fraction = _number(row["physical_region_fraction"], "held physical fraction")
    if not 0 <= fraction <= 1 or type(row["memory_reservation_bytes"]) is not int \
            or row["memory_reservation_bytes"] < 0:
        raise SerialSchedulingRefused("held component resource vector is invalid")
    return dict(row, started_at=start, ended_at=end, physical_region_fraction=fraction)


def reopen_held_receipts(batch_dir: Path, reference: Mapping[str, Any], *,
                         selection: scheduling.Selection, target: Mapping[str, Any]
                         ) -> tuple[scheduling.HeldClaimReceipt, ...]:
    """Reopen original direct-owner intervals and partition one selected stage."""
    reference = _closed(reference, {"schema", "selection_digest", "evidence"},
                        "held reference")
    if reference["schema"] != REFERENCE_SCHEMA \
            or reference["selection_digest"] != selection.digest:
        raise SerialSchedulingRefused("held reference belongs to another selection")
    artifact = _closed(reference["evidence"], {"locator", "sha256", "verified"},
                       "held artifact reference")
    if artifact["verified"] is not True:
        raise SerialSchedulingRefused("held artifact reference is not verified")
    from .measurement_capture import ArtifactStore, _plain
    store = ArtifactStore(Path(batch_dir).resolve() / "held-claim-artifacts")
    try:
        body = _plain(store.read(artifact["locator"], artifact["sha256"]))
    finally:
        store.close()
    body = _closed(body, {"schema", "selection", "selection_digest", "target", "components"},
                   "held interval bundle")
    if body["schema"] != INTERVAL_SCHEMA or body["selection_digest"] != selection.digest \
            or body["selection"] != selection.to_dict() or body["target"] != dict(target):
        raise SerialSchedulingRefused("held interval bundle identity differs")
    if not isinstance(body["components"], list) or not 1 <= len(body["components"]) <= 2:
        raise SerialSchedulingRefused("held interval bundle component count is invalid")
    components = tuple(_component(item) for item in body["components"])
    cpu = [item for item in components if item["device_id"] == "cpu"]
    gpu = [item for item in components if item["device_id"] != "cpu"]
    if len(cpu) != 1 or len(gpu) > 1:
        raise SerialSchedulingRefused("held intervals lack one original CPU context")
    proposal = selection.proposal
    if proposal is None:
        raise SerialSchedulingRefused("held interval selection has no proposal")
    host = cpu[0]
    segments: list[tuple[float, float, tuple[str, ...], tuple[str, ...], float,
                        tuple[str, ...], int]] = []
    if gpu:
        device = gpu[0]
        if proposal.backend != "gpu" or tuple(device["gpu_device_ids"]) \
                != proposal.estimated_claims.gpu_devices:
            raise SerialSchedulingRefused("held GPU identity differs from selected resources")
        if not (host["started_at"] <= device["started_at"] < device["ended_at"]
                <= host["ended_at"]):
            raise SerialSchedulingRefused("GPU interval is outside its original host claim")
        if host["started_at"] < device["started_at"]:
            segments.append((host["started_at"], device["started_at"],
                tuple(host["physical_claim_ids"]), (), host["physical_region_fraction"],
                tuple(host["affinity_cores"]), host["memory_reservation_bytes"]))
        segments.append((device["started_at"], device["ended_at"],
            tuple(host["physical_claim_ids"] + device["physical_claim_ids"]),
            tuple(device["gpu_device_ids"]), host["physical_region_fraction"],
            tuple(host["affinity_cores"]),
            max(host["memory_reservation_bytes"], device["memory_reservation_bytes"])))
        if device["ended_at"] < host["ended_at"]:
            segments.append((device["ended_at"], host["ended_at"],
                tuple(host["physical_claim_ids"]), (), host["physical_region_fraction"],
                tuple(host["affinity_cores"]), host["memory_reservation_bytes"]))
    else:
        if proposal.backend != "cpu" or proposal.estimated_claims.gpu_devices:
            raise SerialSchedulingRefused("held CPU identity differs from selected resources")
        segments.append((host["started_at"], host["ended_at"],
            tuple(host["physical_claim_ids"]), (), host["physical_region_fraction"],
            tuple(host["affinity_cores"]), host["memory_reservation_bytes"]))
    if host["physical_region_fraction"] != proposal.estimated_claims.physical_region_fraction \
            or any(item["memory_reservation_bytes"]
                   != proposal.estimated_claims.memory_reservation_bytes for item in components):
        raise SerialSchedulingRefused("held resources differ from selected resource vector")
    context_ids = [item["context_id"] for item in components]
    return tuple(scheduling.HeldClaimReceipt(
        receipt_id=_digest({"selection_digest": selection.digest,
                            "context_ids": context_ids, "segment": index}),
        proposal_id=proposal.proposal_id, backend=proposal.backend,
        stage_class=proposal.stage_class, started_at=start, ended_at=end,
        ownership_generation=1, allocation_generation=1,
        physical_claim_ids=physical, physical_region_fraction=fraction,
        gpu_device_ids=devices, memory_reservation_bytes=memory,
        affinity_cores=affinity, beneficiary_shares={proposal.proposal_id: 1.0})
        for index, (start, end, physical, devices, fraction, affinity, memory)
        in enumerate(segments))


__all__ = [
    "MANIFEST_SCHEMA", "SerialSchedulerManifest", "SerialSchedulingRefused",
    "one_iteration_outcome", "reopen_held_receipts", "select_target",
    "validate_target_bindings",
]
