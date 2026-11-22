"""Contention and GPU-residency witnesses derived from ORIGINAL in-window parent evidence.

Box text (AKU-07o): use existing protocols with original in-window parent evidence; no generic
success, late sample, synthetic telemetry or reduced required-witness list.

Everything here is derived from the already-sealed, reopened lifecycle observation (the bytes the
parent captured during the unit's own window). Nothing samples a process, reads a clock or the
host. Owning protocols reused, not re-implemented:
  - residency: `lifecycle_observation._residency` (exact mapped-DSO + target KFD allocation + FD);
    the recorded `residency` block must equal an independent replay of it from the retained samples.
  - contention: `lifecycle_observation._intervals` (process census, physical-claim overlap, foreign
    verification); the recorded `intervals` must equal an independent replay from the samples.
Policy (from `residency.cpu_lifecycle_invalidity`): load, pressure and ordinary foreign processes
are diagnostic, never a veto; only a VERIFIED foreign model-inference process with CPU activity
overlapping the held claim contradicts a clean contention reading. Any gap, unverified overlap or
incomplete census stays `unknown`. A witness is never `pass` from absence of evidence.

Foreign GPU consumers (GPU arm): v2 lifecycle samples carry the in-window KFD per-process VRAM
census of the held devices; `derive_foreign_gpu_consumers` reads none/present/unknown from it (owner
excluded, late samples excluded, owner must be visible or the census is not trusted). `present`
makes contention FAIL; anything but `none` keeps it from PASS (legacy v1 samples -> unknown).
Still uncaptured (see `UNCAPTURED`): node bandwidth/LLC, and purpose, a policy witness; the plan's
declared category/intended use is exposed by `declared_purpose` as a declared input, never sealed.
"""
from __future__ import annotations

from typing import Any, Mapping
import threading

from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_scientific_witness as scientific
from . import observation_binding as ob
from . import worker_lifecycle as wl

ADAPTER_ID_CONTENTION = "epyc.autokernel.observation_contention_witness.v1"
ADAPTER_ID_RESIDENCY = "epyc.autokernel.observation_residency_witness.v1"
REPLAY_COVERAGE = "original_in_window_lifecycle_observation_replay_by_owning_protocol"
ADAPTER_ROLES = ("__init__", "evaluate", "reopen", "source_identity")
HELPER_ROLES = ("derive_contention", "derive_residency", "_replayed_observation_blocks",
                "derive_foreign_gpu_consumers")  # historical sources carry the first three
WITNESSES = {"contention": ADAPTER_ID_CONTENTION, "residency": ADAPTER_ID_RESIDENCY}
UNCAPTURED = {
    "purpose": "intended-use authority is a validation_semantic_adapter policy decision; no sealed record carries it",
    "contention_gpu": "foreign GPU consumers are not captured: no in-window sample carries a KFD census (sample v2)",
    "contention_numa_bandwidth": "no per-node memory-bandwidth or LLC counters are captured in the observation; "
        "needs a sealed per-marker uncore read (amd_df/amd_umc DRAM + amd_l3 PMUs, not exposed in the "
        "container's event_source list) with multiplex/reset state and a declared threshold policy",
}


def _replayed_observation_blocks(observation: Mapping[str, Any]) -> tuple[Any, Any]:
    """Independently recompute `intervals` (+ their issues) from the retained samples."""
    claimed = set(observation["held_claim"].get("physical_cpus", []))
    intervals, issues = lo._intervals(observation["samples"], claimed, observation["gap_limit_s"])
    return intervals, issues


def _measurement_window(observation: Mapping[str, Any]) -> tuple[float, float] | None:
    bounds = sorted(observation["phase_boundaries"], key=lambda row: row["monotonic_s"])
    starts = [row["monotonic_s"] for row in bounds if row["phase"] == "measurement"]
    ends = [row["monotonic_s"] for row in bounds if starts and row["monotonic_s"] > starts[0]
            and row["phase"] != "measurement"] + [observation["ended_monotonic_s"]]
    ends = [value for value in ends if isinstance(value, (int, float))]
    return (starts[0], min(ends)) if len(starts) == 1 and ends else None


def derive_foreign_gpu_consumers(observation: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    """none/present/unknown from in-window v2 KFD censuses; the held-allocation owner is excluded."""
    if backend != "gpu":
        return {"status": "not_applicable", "reason": "CPU arm", "facts": {}}
    window, devices = _measurement_window(observation), set(observation["held_claim"]["gpu_devices"])
    owner = (observation["target_binding"] or {}).get("pid")
    rows = [row for row in observation["samples"] if row["phase"] == "measurement"]
    timely = [window is not None and window[0] <= row["read_started_monotonic_s"]
              <= row["read_ended_monotonic_s"] <= window[1] for row in rows]
    censuses = [(row["gpu"] or {}).get("kfd_consumers") if row["schema"] == lo.SAMPLE_SCHEMA_V2 else None
                for row, ok in zip(rows, timely) if ok]
    foreign, idle, owner_seen = [], [], 0
    for index, census in enumerate(censuses):
        for item in census["processes"] if census and census["status"] == "observed" else ():
            held = sum(value for device, value in item["vram_bytes"].items() if device in devices)
            if item["pid"] == owner:
                owner_seen += int(held > 0)
            else:
                (foreign if held > 0 else idle).append(
                    {"sample": index, "pid": item["pid"], "held_device_vram_bytes": held})
    captured = sum(bool(census) and census["status"] == "observed" for census in censuses)
    facts = {"measurement_window": list(window) if window else None, "in_window_samples": len(censuses),
             "late_samples_excluded": timely.count(False), "captured_samples": captured,
             "owner_pid": owner, "owner_visible_samples": owner_seen, "foreign_consumers": foreign,
             "foreign_kfd_without_held_vram": idle}
    if foreign:
        return {"status": "present", "facts": facts,
                "reason": "a foreign process held VRAM on the held GPU device during the measurement window"}
    if owner is None or len(censuses) < 2 or captured != len(censuses) or owner_seen != captured:
        return {"status": "unknown", "facts": facts, "reason": "in-window KFD census missing, unreadable, "
                "too short, or does not show the owner's own allocation"}
    return {"status": "none", "facts": facts,
            "reason": "only the held allocation's owner held VRAM on the held device in every in-window census"}


def declared_purpose(plan: Any) -> dict[str, Any]:
    """Purpose is a policy witness: expose the plan's DECLARED intent, never sealed, never concrete."""
    return {"status": "unknown", "reason": UNCAPTURED["purpose"], "facts": {
        "basis": "declared_plan_input_not_sealed", "declared": {"category": plan.category,
            "intended_use": plan.intended_use, "record_class": plan.record_class}}}


def derive_contention(observation: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    """Facts-only contention finding from the recorded in-window intervals."""
    replayed, _issues = _replayed_observation_blocks(observation)
    if replayed != observation["intervals"]:
        return {"status": "unknown", "reason": "recorded intervals differ from owning-protocol replay",
                "facts": {"measurement_intervals": None}}
    intervals = [row for row in replayed if row["phase"] == "measurement"]
    gpu = derive_foreign_gpu_consumers(observation, backend=backend) if backend == "gpu" else None
    facts: dict[str, Any] = {"measurement_intervals": len(intervals),
        "observation_completeness": observation["completeness"], "uncaptured": [
            UNCAPTURED["contention_numa_bandwidth"]] + (
            [UNCAPTURED["contention_gpu"]] if gpu and not gpu["facts"]["captured_samples"] else [])}
    if gpu:
        facts["foreign_gpu_consumers"] = gpu
    if not intervals:
        return {"status": "unknown", "reason": "no measurement-phase interval was observed", "facts": facts}
    contradictions, unverified, diagnostic = [], [], []
    for index, row in enumerate(intervals):
        for overlap in row["potential_foreign_overlap"]:
            verdict = overlap["verification"] or {"status": "unknown", "kind": None}
            entry = {"interval": index, "pid": overlap["pid"],
                     "cpu_tick_delta": overlap["process_total_cpu_tick_delta"],
                     "overlap": overlap["potential_physical_claim_overlap"],
                     "verification": verdict}
            if verdict["status"] != "verified":
                unverified.append(entry)
            elif verdict["kind"] == "model_inference":
                contradictions.append(entry)
            else:
                diagnostic.append(entry)
    gaps = [index for index, row in enumerate(intervals) if row["status"] != "observed"]
    facts |= {"verified_foreign_inference_overlap": contradictions, "unverified_overlap": unverified,
              "diagnostic_only_overlap": diagnostic, "unobserved_intervals": gaps,
              "issues": observation["issues"]}
    if contradictions:
        return {"status": "fail", "reason": "verified foreign model inference overlapped the held claim "
                "during the measurement window", "facts": facts}
    if gpu and gpu["status"] == "present":
        return {"status": "fail", "reason": gpu["reason"], "facts": facts}
    if (gaps or unverified or observation["completeness"] != "complete" or observation["issues"]
            or gpu and gpu["status"] != "none"):
        return {"status": "unknown", "reason": "measurement window coverage or foreign verification is incomplete",
                "facts": facts}
    return {"status": "pass", "reason": "no verified foreign inference overlapped the held claim in a "
            "fully observed measurement window; ordinary load is diagnostic only", "facts": facts}


def derive_residency(observation: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    """GPU residency from the owning `_residency` protocol replayed over retained samples."""
    replay = lo._residency(backend, observation["samples"])
    recorded = observation.get("residency")
    facts = {"replayed": replay, "recorded": recorded}
    if replay != recorded:
        return {"status": "unknown", "reason": "recorded residency differs from owning-protocol replay",
                "facts": facts}
    status = {"observed": "pass", "not_observed": "fail"}.get(replay["status"], "unknown")
    reason = {"pass": "target KFD allocation, FD and exact mapped DSO observed during the measurement window",
              "fail": "measurement window sampled with no GPU residency",
              "unknown": ("CPU arm: GPU residency is not applicable" if replay["status"] == "not_applicable"
                          else "GPU residency attribution or DSO identity is unknown")}[status]
    return {"status": status, "reason": reason, "facts": facts}


class ObservationWitnessAdapter:
    """Concrete parent-side adapter for the `contention` or `residency` registry slot."""

    def __init__(self, witness: str) -> None:
        if witness not in WITNESSES:
            raise scientific.ScientificWitnessRefused("unsupported observation witness")
        self.witness = witness
        self._lock = threading.RLock()

    def evaluate(self, context, native, lifecycle_link, runtime_readbacks, *, store):
        from .native_parent_evidence import NativeUnitEvidenceProducer
        if type(store) is not mc.ArtifactStore:
            raise scientific.ScientificWitnessRefused("concrete original artifact store required")
        verifier = NativeUnitEvidenceProducer(store=store, context=context)
        verifier._join_native(native)
        reference = lifecycle_link.reference
        observation = lo.validate_observation(ob._plain(store.read(
            reference.artifact.locator, reference.artifact.sha256)))
        verifier._join_observation(native, observation, reference)
        derive = derive_contention if self.witness == "contention" else derive_residency
        finding = derive(observation, backend=context.recipe.backend)
        body = {"schema": scientific.WITNESS_SCHEMA, "witness": self.witness,
            "adapter_id": WITNESSES[self.witness], "frame": ob._plain(scientific._frame(context)),
            "native_artifact_digest": native["artifact_digest"],
            "lifecycle_reference": reference.to_dict(),
            "runtime_readbacks": [ref.to_dict() for ref in runtime_readbacks],
            "source": self.source_identity(),
            "report": {"reason": finding["reason"], "facts": ob._plain(finding["facts"])},
            "scope_defects": [], "status": finding["status"], "replay_coverage": REPLAY_COVERAGE}
        digest = wl._digest(body)
        return scientific.WitnessEvidenceReference(
            store.write(f"parent-scientific-witness:{digest}", body), digest, self.witness)

    def reopen(self, reference, *, context, native, lifecycle_link, runtime_readbacks, store):
        if type(reference) is not scientific.WitnessEvidenceReference or reference.witness != self.witness:
            raise scientific.ScientificWitnessRefused("concrete witness reference for this slot required")
        body = store.read(reference.artifact.locator, reference.artifact.sha256)
        scientific._same(wl._digest(ob._plain(body)), reference.digest, "observation witness digest")
        store.verify(f"parent-scientific-witness:{reference.digest}", body)
        expected = self.evaluate(context, native, lifecycle_link, runtime_readbacks, store=store)
        scientific._same(reference.to_dict(), expected.to_dict(), "independently replayed observation witness")
        return ob._freeze(ob._plain(body))

    def source_identity(self) -> Mapping[str, Any]:
        cls = type(self)
        return {"adapter_id": WITNESSES[self.witness], "witness": self.witness,
            "callables": [{"role": name, "identity": lo.callable_identity(getattr(cls, name))}
                          for name in ADAPTER_ROLES],
            "helpers": [{"role": name, "identity": lo.callable_identity(globals()[name])}
                        for name in HELPER_ROLES],
            "owning_protocols": [lo.callable_identity(lo._residency), lo.callable_identity(lo._intervals)]}


def validate_observation_source(value: Any, witness: str) -> Mapping[str, Any]:
    """Closed historical grammar for one slot; no comparison with today's code."""
    from .native_producer_source import _callables, _closed, _identity
    row = _closed(value, ("adapter_id", "witness", "callables", "helpers", "owning_protocols"),
                  "observation witness source")
    if witness not in WITNESSES or row["adapter_id"] != WITNESSES[witness] or row["witness"] != witness:
        raise scientific.ScientificWitnessRefused("unsupported observation witness adapter")
    row["callables"] = _callables(row["callables"], ADAPTER_ROLES)
    roles = HELPER_ROLES if type(row["helpers"]) in (list, tuple) and len(row["helpers"]) == 4 else HELPER_ROLES[:3]
    row["helpers"] = _callables(row["helpers"], roles)
    if type(row["owning_protocols"]) not in (list, tuple) or len(row["owning_protocols"]) != 2:
        raise scientific.ScientificWitnessRefused("observation witness owning protocol set differs")
    row["owning_protocols"] = [_identity(item) for item in row["owning_protocols"]]
    return ob._freeze(ob._plain(row))


def observation_source_identities(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return (tuple(item["identity"] for item in (*row["callables"], *row["helpers"]))
            + tuple(row["owning_protocols"]))
