"""Generic Annex K A2 runtime screening with immutable three-sample banks.

This module emits advisory discovery records only.  It neither grades claims nor
creates keep, validation, release, or production authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import median
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import experiment_plan as ep
from . import planned_serving
from . import unified_planner

BANK_SCHEMA = "epyc.autokernel.a2_runtime_baseline_bank.v1"
CONTEXT_SCHEMA = "epyc.autokernel.a2_runtime_frame_context.v1"
EVENT_SCHEMA = "epyc.autokernel.a2_runtime_phase_event.v1"
PROOF_SCHEMA = "epyc.autokernel.a2_runtime_invocation_proof.v1"
RECEIPT_SCHEMA = "epyc.autokernel.a2_runtime_screen_receipt.v1"
A2_PROTOCOL = "P-AK-SEARCH-1-A2"
MANDATORY_WITNESSES = frozenset({
    "correctness", "identity", "linkage", "frequency_power_envelope",
    "resource_claim_open_close", "inference_exclusion",
})
_VERIFIER_TOKEN = object()


class DiscoveryScreenRefused(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise DiscoveryScreenRefused(f"value is not finite canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DiscoveryScreenRefused("mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiscoveryScreenRefused(f"{label} must be nonempty text")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DiscoveryScreenRefused(f"{label} must be lowercase SHA-256")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        qualifier = "positive " if positive else ""
        raise DiscoveryScreenRefused(f"{label} must be {qualifier}finite numeric")
    return float(value)


@dataclass(frozen=True)
class RuntimeFrameContext:
    evaluator_identity: Mapping[str, Any]
    runtime_source_identity: Mapping[str, Any]
    linkage_identity: Mapping[str, Any]
    frequency_power_envelope: Mapping[str, Any]
    resource_claim: Mapping[str, Any]
    policy_digest: str
    host_epoch: str
    schema: str = CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONTEXT_SCHEMA:
            raise DiscoveryScreenRefused("runtime frame context schema is unsupported")
        for name in ("evaluator_identity", "runtime_source_identity", "linkage_identity",
                     "frequency_power_envelope", "resource_claim"):
            item = getattr(self, name)
            if not isinstance(item, Mapping) or not item:
                raise DiscoveryScreenRefused(f"{name} must be a nonempty identity object")
            _canonical(_plain(item))
            object.__setattr__(self, name, _freeze(item))
        object.__setattr__(self, "policy_digest", _sha(self.policy_digest, "policy_digest"))
        object.__setattr__(self, "host_epoch", _sha(self.host_epoch, "host_epoch"))

    @classmethod
    def from_dict(cls, value: Any) -> "RuntimeFrameContext":
        fields = {"schema", "evaluator_identity", "runtime_source_identity",
                  "linkage_identity", "frequency_power_envelope", "resource_claim",
                  "policy_digest", "host_epoch"}
        if not isinstance(value, Mapping) or set(value) != fields \
                or value["schema"] != CONTEXT_SCHEMA:
            raise DiscoveryScreenRefused("runtime frame context fields/schema differ")
        rows = []
        for name in ("evaluator_identity", "runtime_source_identity", "linkage_identity",
                     "frequency_power_envelope", "resource_claim"):
            item = value[name]
            if not isinstance(item, Mapping) or not item:
                raise DiscoveryScreenRefused(f"{name} must be a nonempty identity object")
            rows.append(item)
        return cls(*rows, value["policy_digest"], value["host_epoch"])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "evaluator_identity": _plain(self.evaluator_identity),
                "runtime_source_identity": _plain(self.runtime_source_identity),
                "linkage_identity": _plain(self.linkage_identity),
                "frequency_power_envelope": _plain(self.frequency_power_envelope),
                "resource_claim": _plain(self.resource_claim),
                "policy_digest": self.policy_digest, "host_epoch": self.host_epoch}


@dataclass(frozen=True)
class InvocationProof:
    producer_identity: Mapping[str, Any]
    started_at: float
    ended_at: float
    witness_refs: Mapping[str, str]
    overlapping_competing_inference: bool
    ordinary_load: Mapping[str, Any]
    invocation_identity: Mapping[str, Any]
    observations_digest: str
    schema: str = PROOF_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "InvocationProof":
        fields = {"schema", "producer_identity", "started_at", "ended_at",
                  "witness_refs", "overlapping_competing_inference", "ordinary_load",
                  "invocation_identity", "observations_digest"}
        if not isinstance(value, Mapping) or set(value) != fields \
                or value["schema"] != PROOF_SCHEMA:
            raise DiscoveryScreenRefused("invocation proof fields/schema differ")
        producer = value["producer_identity"]
        load = value["ordinary_load"]
        refs = value["witness_refs"]
        invocation = value["invocation_identity"]
        if not isinstance(producer, Mapping) or not producer \
                or not isinstance(load, Mapping) or not isinstance(refs, Mapping) \
                or set(refs) != MANDATORY_WITNESSES or not isinstance(invocation, Mapping) \
                or set(invocation) != {"unit_id", "process_id", "launch_id"}:
            raise DiscoveryScreenRefused("invocation proof identity/witness set is incomplete")
        for name in invocation:
            _text(invocation[name], f"invocation_identity.{name}")
        normalized_refs = {name: _text(ref, f"witness_refs.{name}")
                           for name, ref in refs.items()}
        start = _finite(value["started_at"], "started_at")
        end = _finite(value["ended_at"], "ended_at")
        if end <= start:
            raise DiscoveryScreenRefused("invocation proof interval must be positive")
        if type(value["overlapping_competing_inference"]) is not bool:
            raise DiscoveryScreenRefused("inference-overlap result must be boolean")
        _canonical(producer)
        _canonical(load)
        return cls(_freeze(producer), start, end, _freeze(normalized_refs),
                   value["overlapping_competing_inference"], _freeze(load),
                   _freeze(invocation), _sha(value["observations_digest"],
                                             "observations_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "producer_identity": _plain(self.producer_identity),
                "started_at": self.started_at, "ended_at": self.ended_at,
                "witness_refs": _plain(self.witness_refs),
                "overlapping_competing_inference": self.overlapping_competing_inference,
                "ordinary_load": _plain(self.ordinary_load),
                "invocation_identity": _plain(self.invocation_identity),
                "observations_digest": self.observations_digest}


@dataclass(frozen=True)
class InvocationResult:
    raw_unit: ep.RawUnit
    proof: InvocationProof
    observations: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        try:
            raw = ep.RawUnit.from_dict(self.raw_unit.to_dict())
            proof = InvocationProof.from_dict(self.proof.to_dict())
        except Exception as exc:
            raise DiscoveryScreenRefused(f"invocation typed data is invalid: {exc}") from exc
        if not isinstance(self.observations, (list, tuple)) or not self.observations:
            raise DiscoveryScreenRefused("invocation raw observations must be nonempty")
        rows = []
        for item in self.observations:
            if not isinstance(item, Mapping) or not item:
                raise DiscoveryScreenRefused("raw observation must be a nonempty object")
            _canonical(_plain(item))
            rows.append(_freeze(item))
        object.__setattr__(self, "raw_unit", raw)
        object.__setattr__(self, "proof", proof)
        object.__setattr__(self, "observations", tuple(rows))
        if proof.observations_digest != _digest([_plain(item) for item in rows]):
            raise DiscoveryScreenRefused("invocation observation digest differs")

    @classmethod
    def from_dict(cls, value: Any) -> "InvocationResult":
        if not isinstance(value, Mapping) or set(value) != {
                "raw_unit", "proof", "observations"}:
            raise DiscoveryScreenRefused("invocation result fields differ")
        try:
            return cls(ep.RawUnit.from_dict(value["raw_unit"]),
                       InvocationProof.from_dict(value["proof"]),
                       tuple(value["observations"]))
        except ep.PlanValidationError as exc:
            raise DiscoveryScreenRefused(f"invocation raw unit is invalid: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"raw_unit": self.raw_unit.to_dict(), "proof": self.proof.to_dict(),
                "observations": [_plain(item) for item in self.observations]}


class TrustedInvoker(Protocol):
    def invoke(self, unit: ep.UnitSpec, recipe: Any) -> InvocationResult: ...


PhaseSink = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class BaselineBank:
    frame: Mapping[str, Any]
    anchor_results: tuple[InvocationResult, ...]
    schema: str = BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BANK_SCHEMA or len(self.anchor_results) != 3:
            raise DiscoveryScreenRefused("A2 bank requires exactly three anchor results")
        object.__setattr__(self, "frame", _freeze(self.frame))
        object.__setattr__(self, "anchor_results", tuple(
            InvocationResult.from_dict(item.to_dict()) for item in self.anchor_results))
        identities = [item.proof.invocation_identity["launch_id"]
                      for item in self.anchor_results]
        if len(identities) != len(set(identities)):
            raise DiscoveryScreenRefused("baseline bank repeats an invocation identity")

    @property
    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "frame": _plain(self.frame),
                "anchor_results": [item.to_dict() for item in self.anchor_results]}

    @property
    def bank_digest(self) -> str:
        return _digest(self.body)

    def to_dict(self) -> dict[str, Any]:
        return {**self.body, "bank_digest": self.bank_digest}

    @classmethod
    def from_dict(cls, value: Any) -> "BaselineBank":
        if not isinstance(value, Mapping) or set(value) != {
                "schema", "frame", "anchor_results", "bank_digest"}:
            raise DiscoveryScreenRefused("baseline bank fields differ")
        rows = value["anchor_results"]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise DiscoveryScreenRefused("baseline bank results must be an array")
        result = cls(_freeze(value["frame"]), tuple(InvocationResult.from_dict(x) for x in rows),
                     value["schema"])
        if _sha(value["bank_digest"], "bank_digest") != result.bank_digest:
            raise DiscoveryScreenRefused("baseline bank digest differs")
        return result


@dataclass(frozen=True)
class ScreenReceipt:
    plan_digest: str
    frame_digest: str
    bank_digest: str
    candidate_results: tuple[InvocationResult, ...]
    candidate_invocations: int
    anchor_invocations: int
    metric_direction: str
    baseline_median: float
    advisory_median: float
    epoch: str
    non_promotable: bool = True
    schema: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("plan_digest", "frame_digest", "bank_digest", "epoch"):
            _sha(getattr(self, name), name)
        if (self.schema != RECEIPT_SCHEMA
                or type(self.candidate_invocations) is not int
                or type(self.anchor_invocations) is not int
                or self.candidate_invocations != 3 or self.anchor_invocations != 0
                or len(self.candidate_results) != 3
                or self.metric_direction not in {"higher", "lower"}
                or self.non_promotable is not True):
            raise DiscoveryScreenRefused("A2 screen receipt cardinality/authority differs")
        object.__setattr__(self, "candidate_results", tuple(
            InvocationResult.from_dict(item.to_dict()) for item in self.candidate_results))
        identities = [item.proof.invocation_identity["launch_id"]
                      for item in self.candidate_results]
        if len(identities) != len(set(identities)):
            raise DiscoveryScreenRefused("screen receipt repeats an invocation identity")
        object.__setattr__(self, "baseline_median", _finite(
            self.baseline_median, "baseline_median"))
        object.__setattr__(self, "advisory_median", _finite(
            self.advisory_median, "advisory_median"))
        actual = median(item.raw_unit.value for item in self.candidate_results)
        if self.advisory_median != actual:
            raise DiscoveryScreenRefused("advisory_median differs from candidate samples")

    @property
    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "plan_digest": self.plan_digest,
                "frame_digest": self.frame_digest, "bank_digest": self.bank_digest,
                "candidate_results": [item.to_dict() for item in self.candidate_results],
                "candidate_invocations": self.candidate_invocations,
                "anchor_invocations": self.anchor_invocations,
                "metric_direction": self.metric_direction,
                "baseline_median": self.baseline_median,
                "advisory_median": self.advisory_median, "epoch": self.epoch,
                "non_promotable": self.non_promotable}

    @property
    def receipt_digest(self) -> str:
        return _digest(self.body)

    def to_dict(self) -> dict[str, Any]:
        return {**self.body, "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Any) -> "ScreenReceipt":
        if not isinstance(value, Mapping) or set(value) != {
                "schema", "plan_digest", "frame_digest", "bank_digest",
                "candidate_results", "candidate_invocations", "anchor_invocations",
                "metric_direction", "baseline_median", "advisory_median", "epoch",
                "non_promotable",
                "receipt_digest"}:
            raise DiscoveryScreenRefused("screen receipt fields differ")
        result = cls(value["plan_digest"], value["frame_digest"], value["bank_digest"],
                     tuple(InvocationResult.from_dict(x) for x in value["candidate_results"]),
                     value["candidate_invocations"], value["anchor_invocations"],
                     value["metric_direction"], value["baseline_median"],
                     value["advisory_median"], value["epoch"],
                     value["non_promotable"], value["schema"])
        if _sha(value["receipt_digest"], "receipt_digest") != result.receipt_digest:
            raise DiscoveryScreenRefused("screen receipt digest differs")
        return result


def _normalized_plan(value: ep.ExperimentPlan) -> ep.ExperimentPlan:
    try:
        plan = ep.ExperimentPlan.from_dict(value.to_dict())
    except Exception as exc:
        raise DiscoveryScreenRefused(f"A2 ExperimentPlan is invalid: {exc}") from exc
    units = sorted(plan.expected_units, key=lambda item: item.order_index)
    if (plan.record_class != "discovery_screen" or plan.category != "CANDIDATE"
            or plan.phase != "discovery" or plan.protocol_ref != A2_PROTOCOL
            or plan.protocol_status != "ratified" or plan.intended_use != "nominate"
            or plan.stopping["n_per_arm"] != 3 or plan.stopping["paired"]
            or len(plan.changed_factors) != 1
            or [item.arm for item in units] != ["anchor"] * 3 + ["candidate"] * 3):
        raise DiscoveryScreenRefused(
            "plan is not a fixed three-anchor then three-candidate A2 nomination")
    return plan


def _frame(plan: ep.ExperimentPlan, pair: unified_planner.RuntimeArmPair,
           context: RuntimeFrameContext) -> dict[str, Any]:
    pair = unified_planner.RuntimeArmPair.from_dict(pair.to_dict())
    plan = _normalized_plan(plan)
    anchor_identity = planned_serving.arm_identity(pair.anchor.template, pair.anchor)
    candidate_identity = planned_serving.arm_identity(pair.candidate.template, pair.candidate)
    if (dict(plan.anchor_identity) != anchor_identity
            or dict(plan.candidate_identity) != candidate_identity
            or tuple(plan.changed_factors) != (pair.dimension.kind,)
            or context.host_epoch != plan.epoch
            or context.policy_digest != _digest(_plain(plan.policy_snapshot))):
        raise DiscoveryScreenRefused("plan arm/factor identity differs from runtime pair")
    ordered = tuple(sorted(plan.expected_units, key=lambda item: item.order_index))
    prompt_membership = {
        arm: [list(item.expected_prompt_ids) for item in ordered if item.arm == arm]
        for arm in ("anchor", "candidate")}
    return {"protocol_ref": A2_PROTOCOL, "instrument_class": plan.instrument_class,
            "metric": plan.metric, "metric_direction": plan.metric_direction,
            "unit": plan.unit, "phase": plan.phase, "target_revision": plan.target_revision,
            "estimand": plan.estimand, "estimator_id": plan.estimator_id,
            "required_witnesses": list(plan.required_witnesses),
            "policy_snapshot": _plain(plan.policy_snapshot),
            "prompt_membership": prompt_membership,
            "shape": {"n_per_arm": 3, "paired": False,
                      "order": ["anchor"] * 3 + ["candidate"] * 3},
            "factor": {"field": pair.dimension.kind,
                       "anchor": _plain(pair.dimension.anchor),
                       "authority_ref": pair.dimension.authority_ref},
            "anchor_recipe": pair.anchor.to_dict(), "context": context.to_dict()}


def _event(phase: str, state: str, index: int | None, plan_digest: str,
           frame_digest: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {"schema": EVENT_SCHEMA, "phase": phase, "state": state, "index": index,
            "plan_digest": plan_digest, "frame_digest": frame_digest,
            "payload": _plain(payload)}
    return {**body, "event_digest": _digest(body)}


def _emit(sink: PhaseSink, event: Mapping[str, Any]) -> None:
    try:
        committed = sink(_plain(event))
    except BaseException:
        raise
    if not isinstance(committed, Mapping) or dict(committed) != _plain(event):
        raise DiscoveryScreenRefused("phase sink did not commit the exact event")


def _validated_event(value: Any, *, plan_digest: str,
                     frame_digest: str) -> dict[str, Any]:
    fields = {"schema", "phase", "state", "index", "plan_digest", "frame_digest",
              "payload", "event_digest"}
    if not isinstance(value, Mapping) or set(value) != fields \
            or value["schema"] != EVENT_SCHEMA:
        raise DiscoveryScreenRefused("phase event fields/schema differ")
    body = {name: _plain(value[name]) for name in fields - {"event_digest"}}
    if (_sha(value["event_digest"], "event_digest") != _digest(body)
            or value["plan_digest"] != plan_digest
            or value["frame_digest"] != frame_digest
            or value["phase"] not in {"anchor_bank", "candidate_screen"}
            or value["state"] not in {"INTENT", "TERMINAL", "SEALED"}):
        raise DiscoveryScreenRefused("phase event identity/state differs")
    index = value["index"]
    if value["state"] == "SEALED":
        if index is not None:
            raise DiscoveryScreenRefused("sealed phase event cannot name an index")
    elif isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 3:
        raise DiscoveryScreenRefused("invocation phase index must be 0..2")
    return _plain(value)


def _validate_result(result: InvocationResult, plan: ep.ExperimentPlan,
                     unit: ep.UnitSpec, *, producer: Mapping[str, Any]) -> InvocationResult:
    result = InvocationResult.from_dict(result.to_dict())
    raw = result.raw_unit
    if (raw.plan_digest != plan.digest or raw.unit_id != unit.unit_id
            or raw.arm != unit.arm or raw.process_id != unit.process_id
            or raw.observed_order_index != unit.order_index
            or tuple(raw.prompt_ids) != tuple(unit.expected_prompt_ids)
            or not raw.terminal or raw.value is None or raw.value == 0
            or raw.recorded_screen not in {"clean", "flagged_but_retained"}
            or (raw.recorded_screen == "clean") != (raw.reason is None)
            or set(raw.witnesses) != MANDATORY_WITNESSES
            or any(item.status != "pass" for item in raw.witnesses.values())
            or any(raw.witnesses[name].ref != result.proof.witness_refs[name]
                   for name in MANDATORY_WITNESSES)
            or _plain(result.proof.producer_identity) != _plain(producer)
            or _plain(result.proof.invocation_identity) != {
                "unit_id": unit.unit_id, "process_id": unit.process_id,
                "launch_id": result.proof.invocation_identity.get("launch_id")}
            or any(set(item) != {"metric", "value", "unit_id"}
                   or item["metric"] != plan.metric or item["unit_id"] != unit.unit_id
                   or _finite(item["value"], "raw observation value") != raw.value
                   for item in result.observations)):
        raise DiscoveryScreenRefused("invocation result is incomplete or identity-mismatched")
    if result.proof.overlapping_competing_inference:
        raise DiscoveryScreenRefused(
            "competing model inference overlaps the held discovery claim")
    return result


class RegisteredPhaseVerifier:
    """Exact in-process authority for one already trusted phase history."""

    def __init__(self, histories: frozenset[str], token: object) -> None:
        if token is not _VERIFIER_TOKEN:
            raise DiscoveryScreenRefused("registered phase verifier requires trusted live state")
        self._histories = histories

    def verify(self, plan_digest: str, frame_digest: str,
               events: Sequence[Mapping[str, Any]]) -> bool:
        return _digest({"plan_digest": plan_digest, "frame_digest": frame_digest,
                        "events": [_plain(item) for item in events]}) in self._histories


class RegisteredNominationVerifier:
    """Concrete verifier minted only from completed consumer state."""

    def __init__(self, receipts: Mapping[str, ScreenReceipt], token: object) -> None:
        if token is not _VERIFIER_TOKEN:
            raise DiscoveryScreenRefused("registered verifier requires trusted consumer state")
        self._receipts = dict(receipts)

    def verify(self, plan: ep.ExperimentPlan, view: ep.AdmissibleUnitView,
               receipt_value: Any) -> bool:
        receipt = ScreenReceipt.from_dict(receipt_value)
        stored = self._receipts.get(receipt.receipt_digest)
        if stored is None or stored.to_dict() != receipt.to_dict() \
                or receipt.plan_digest != plan.digest:
            raise DiscoveryScreenRefused("nomination receipt is not in trusted completed state")
        candidate_rows = tuple(item.raw_unit for item in receipt.candidate_results)
        if (tuple(row.to_dict() for row in view.selected_rows)
                != tuple(row.to_dict() for row in candidate_rows)
                or dict(view.independent_n) != {"anchor": 0, "candidate": 3}):
            raise DiscoveryScreenRefused("nomination unit view differs from candidate-only receipt")
        return True


class RegisteredBankVerifier:
    """Capability for reusing only an actually sealed immutable bank."""

    def __init__(self, banks: Mapping[str, BaselineBank], token: object) -> None:
        if token is not _VERIFIER_TOKEN:
            raise DiscoveryScreenRefused("registered bank verifier requires trusted phase state")
        self._banks = dict(banks)

    def verify(self, bank: BaselineBank) -> bool:
        stored = self._banks.get(bank.bank_digest)
        return stored is not None and stored.to_dict() == bank.to_dict()


class A2RuntimeScreen:
    """One validated A2 plan/pair consumer with a serialized phase-event sink."""

    def __init__(self, plan: ep.ExperimentPlan, pair: unified_planner.RuntimeArmPair,
                 context: RuntimeFrameContext | Mapping[str, Any], *, invoker: TrustedInvoker,
                 phase_sink: PhaseSink, events: Sequence[Mapping[str, Any]] = (),
                 replay_verifier: RegisteredPhaseVerifier | None = None) -> None:
        if not isinstance(pair, unified_planner.RuntimeArmPair):
            raise DiscoveryScreenRefused(
                "generic A2 v1 supports validated runtime arm pairs only; source mode is unsupported")
        self.plan = _normalized_plan(plan)
        self.pair = unified_planner.RuntimeArmPair.from_dict(pair.to_dict())
        self.context = (RuntimeFrameContext.from_dict(context.to_dict())
                        if isinstance(context, RuntimeFrameContext)
                        else RuntimeFrameContext.from_dict(context))
        self.frame = _freeze(_frame(self.plan, self.pair, self.context))
        self.frame_digest = _digest(_plain(self.frame))
        if not callable(getattr(invoker, "invoke", None)) or not callable(phase_sink):
            raise DiscoveryScreenRefused("trusted invoker and serialized phase sink are required")
        self.invoker = invoker
        self.phase_sink = phase_sink
        self._receipts: dict[str, ScreenReceipt] = {}
        self._events: list[dict[str, Any]] = []
        self._results: dict[tuple[str, int], tuple[InvocationResult, str, str | None]] = {}
        self._intents: dict[tuple[str, int], Mapping[str, Any]] = {}
        self._pending: set[tuple[str, int]] = set()
        self._sealed: dict[str, Mapping[str, Any]] = {}
        self._banks: dict[str, BaselineBank] = {}
        self._poisoned = False
        supplied_events = tuple(_plain(item) for item in events)
        self._history_authoritative = not supplied_events or (
            isinstance(replay_verifier, RegisteredPhaseVerifier)
            and replay_verifier.verify(self.plan.digest, self.frame_digest, supplied_events))
        for value in supplied_events:
            self._apply_event(_validated_event(
                value, plan_digest=self.plan.digest, frame_digest=self.frame_digest))

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        phase, state, index = event["phase"], event["state"], event["index"]
        key = None if index is None else (phase, index)
        if state == "INTENT":
            if key in self._pending or key in self._results or phase in self._sealed:
                raise DiscoveryScreenRefused("phase history repeats an invocation intent")
            self._pending.add(key)
            self._intents[key] = _freeze(event["payload"])
        elif state == "TERMINAL":
            if key not in self._pending or key in self._results or phase in self._sealed:
                raise DiscoveryScreenRefused("terminal phase lacks its exact pending intent")
            terminal = event["payload"]
            if not isinstance(terminal, Mapping) or set(terminal) != {
                    "status", "reason", "result"} or terminal["status"] not in {
                        "valid", "invalid"}:
                raise DiscoveryScreenRefused("terminal disposition is malformed")
            result = InvocationResult.from_dict(terminal["result"])
            unit = ep.UnitSpec.from_dict(self._intents[key]["unit"])
            producer = self._intents[key]["producer_identity"]
            try:
                _validate_result(result, self.plan, unit, producer=producer)
                actual_status, actual_reason = "valid", None
            except DiscoveryScreenRefused as exc:
                actual_status, actual_reason = "invalid", str(exc)
            if terminal["status"] != actual_status or terminal["reason"] != actual_reason:
                raise DiscoveryScreenRefused("terminal disposition differs from result")
            self._pending.remove(key)
            self._results[key] = (result, actual_status, actual_reason)
        else:
            if phase in self._sealed or any(item[0] == phase for item in self._pending) \
                    or {item[1] for item in self._results if item[0] == phase} != {0, 1, 2}:
                raise DiscoveryScreenRefused("phase sealed before three exact terminals")
            self._sealed[phase] = _freeze(event["payload"])
        self._events.append(_plain(event))

    def _commit(self, event: Mapping[str, Any]) -> None:
        if self._poisoned:
            raise DiscoveryScreenRefused(
                "phase append outcome is uncertain; replay trusted events before continuing")
        try:
            _emit(self.phase_sink, event)
        except BaseException:
            self._poisoned = True
            raise
        self._apply_event(_validated_event(
            event, plan_digest=self.plan.digest, frame_digest=self.frame_digest))

    def _run(self, phase: str, units: Sequence[ep.UnitSpec], recipe: Any) \
            -> tuple[InvocationResult, ...]:
        if self._poisoned:
            raise DiscoveryScreenRefused(
                "phase append outcome is uncertain; replay trusted events before continuing")
        producer = {"frame_digest": self.frame_digest,
                    "recipe_snapshot_digest": recipe.snapshot_digest,
                    "recipe_execution_digest": recipe.execution_digest}
        results = []
        for index, unit in enumerate(units):
            key = (phase, index)
            expected_intent = {"unit": unit.to_dict(), "producer_identity": producer}
            if key in self._intents and _plain(self._intents[key]) != expected_intent:
                raise DiscoveryScreenRefused("phase intent differs from fixed unit/producer")
            existing = self._results.get(key)
            if existing is not None:
                result, status, reason = existing
                if status != "valid":
                    raise DiscoveryScreenRefused(
                        f"completed invocation is non-admissible and cannot be rerun: {reason}")
                results.append(_validate_result(result, self.plan, unit, producer=producer))
                continue
            if key in self._pending:
                raise DiscoveryScreenRefused(
                    "incomplete invocation requires owned reconciliation; it is not rerun")
            self._commit(_event(
                phase, "INTENT", index, self.plan.digest, self.frame_digest,
                {"unit": unit.to_dict(), "producer_identity": producer}))
            result = InvocationResult.from_dict(self.invoker.invoke(unit, recipe).to_dict())
            try:
                _validate_result(result, self.plan, unit, producer=producer)
                status, reason = "valid", None
            except DiscoveryScreenRefused as exc:
                status, reason = "invalid", str(exc)
            self._commit(_event(
                phase, "TERMINAL", index, self.plan.digest, self.frame_digest,
                {"status": status, "reason": reason, "result": result.to_dict()}))
            if status != "valid":
                raise DiscoveryScreenRefused(
                    f"completed invocation is non-admissible and cannot be rerun: {reason}")
            results.append(result)
        identities = [item.proof.invocation_identity["launch_id"] for item in results]
        if len(identities) != len(set(identities)):
            raise DiscoveryScreenRefused("A2 samples do not have independent invocation identities")
        return tuple(results)

    def create_bank(self) -> BaselineBank:
        units = tuple(item for item in sorted(
            self.plan.expected_units, key=lambda row: row.order_index) if item.arm == "anchor")
        results = self._run("anchor_bank", units, self.pair.anchor)
        bank = BaselineBank(self.frame, results)
        sealed = self._sealed.get("anchor_bank")
        if sealed is not None:
            restored = BaselineBank.from_dict(_plain(sealed))
            if restored.to_dict() != bank.to_dict():
                raise DiscoveryScreenRefused("sealed bank differs from terminal phase results")
            if self._history_authoritative:
                self._banks[restored.bank_digest] = restored
            return restored
        self._commit(_event("anchor_bank", "SEALED", None, self.plan.digest,
                            self.frame_digest, bank.to_dict()))
        if self._history_authoritative:
            self._banks[bank.bank_digest] = bank
        return bank

    def screen(self, bank: BaselineBank | Mapping[str, Any], *,
               bank_verifier: RegisteredBankVerifier | None = None) -> ScreenReceipt:
        bank = (BaselineBank.from_dict(bank.to_dict()) if isinstance(bank, BaselineBank)
                else BaselineBank.from_dict(bank))
        if _plain(bank.frame) != _plain(self.frame):
            raise DiscoveryScreenRefused("baseline bank common frame differs")
        trusted = self._banks.get(bank.bank_digest)
        if ((trusted is None or trusted.to_dict() != bank.to_dict())
                and (not isinstance(bank_verifier, RegisteredBankVerifier)
                     or not bank_verifier.verify(bank))):
            raise DiscoveryScreenRefused("baseline bank lacks trusted sealed phase authority")
        bank_launches = {item.proof.invocation_identity["launch_id"]
                         for item in bank.anchor_results}
        units = tuple(item for item in sorted(
            self.plan.expected_units, key=lambda row: row.order_index) if item.arm == "candidate")
        results = self._run("candidate_screen", units, self.pair.candidate)
        if bank_launches & {item.proof.invocation_identity["launch_id"]
                            for item in results}:
            raise DiscoveryScreenRefused(
                "candidate samples reuse a baseline invocation identity")
        values = tuple(item.raw_unit.value for item in results)
        anchors = tuple(item.raw_unit.value for item in bank.anchor_results)
        receipt = ScreenReceipt(
            self.plan.digest, self.frame_digest, bank.bank_digest, results, 3, 0,
            self.plan.metric_direction, median(anchors), median(values), self.plan.epoch)
        sealed = self._sealed.get("candidate_screen")
        if sealed is not None:
            restored = ScreenReceipt.from_dict(_plain(sealed))
            if restored.to_dict() != receipt.to_dict():
                raise DiscoveryScreenRefused(
                    "sealed screen differs from terminal phase results")
            receipt = restored
        else:
            self._commit(_event("candidate_screen", "SEALED", None, self.plan.digest,
                                self.frame_digest, receipt.to_dict()))
        if self._history_authoritative:
            self._receipts[receipt.receipt_digest] = receipt
        return receipt

    def nomination_verifier(self) -> RegisteredNominationVerifier:
        return RegisteredNominationVerifier(self._receipts, _VERIFIER_TOKEN)

    def bank_verifier(self) -> RegisteredBankVerifier:
        return RegisteredBankVerifier(self._banks, _VERIFIER_TOKEN)

    def phase_verifier(self) -> RegisteredPhaseVerifier:
        if self._poisoned or self._pending or not self._history_authoritative:
            raise DiscoveryScreenRefused(
                "uncertain or unattested phase history cannot mint replay authority")
        identity = _digest({"plan_digest": self.plan.digest,
                            "frame_digest": self.frame_digest,
                            "events": self._events})
        return RegisteredPhaseVerifier(frozenset({identity}), _VERIFIER_TOKEN)


def advisory_history(receipts: Sequence[ScreenReceipt | Mapping[str, Any]], *,
                     current_epoch: str, metric_direction: str) -> tuple[dict[str, Any], ...]:
    """A3 ordering: same-epoch magnitudes only; stale records retain conclusions only."""
    current_epoch = _sha(current_epoch, "current_epoch")
    if metric_direction not in {"higher", "lower"}:
        raise DiscoveryScreenRefused("metric_direction is unsupported")
    rows = []
    current_frame_digest = None
    for value in receipts:
        receipt = (ScreenReceipt.from_dict(value.to_dict())
                   if isinstance(value, ScreenReceipt) else ScreenReceipt.from_dict(value))
        if receipt.metric_direction != metric_direction:
            raise DiscoveryScreenRefused("advisory history mixes metric directions")
        stale = receipt.epoch != current_epoch
        if not stale:
            if current_frame_digest is None:
                current_frame_digest = receipt.frame_digest
            elif receipt.frame_digest != current_frame_digest:
                raise DiscoveryScreenRefused(
                    "advisory history mixes incomparable common frames")
        rows.append({"receipt_digest": receipt.receipt_digest, "stale_epoch": stale,
                     "comparable_measurement": not stale,
                     "advisory_median": None if stale else receipt.advisory_median,
                     "conclusion": "runtime_screen_completed_nonpromotable"})
    comparable = sorted((row for row in rows if not row["stale_epoch"]),
                        key=lambda row: row["advisory_median"],
                        reverse=metric_direction == "higher")
    stale_rows = sorted((row for row in rows if row["stale_epoch"]),
                        key=lambda row: row["receipt_digest"])
    return tuple(comparable + stale_rows)


__all__ = ["A2RuntimeScreen", "BaselineBank", "DiscoveryScreenRefused",
           "InvocationProof", "InvocationResult", "RegisteredNominationVerifier",
           "RegisteredBankVerifier", "RegisteredPhaseVerifier", "RuntimeFrameContext", "ScreenReceipt",
           "advisory_history"]
