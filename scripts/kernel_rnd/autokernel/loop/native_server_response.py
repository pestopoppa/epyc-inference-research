"""Prospective raw HTTP evidence from the already-owned native serving lifecycle.

This module neither launches work nor supplies a correctness verdict. Original request
and response bytes are retained after the request window, before owned teardown.
Reopening checks exact frozen identity/order and parses only the retained bytes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any

from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import observation_binding as ob


RESPONSE_SCHEMA = "epyc.autokernel.native_server_response.v1"
UNIT_SCHEMA = "epyc.autokernel.native_server_response_unit.v1"
DIRECT_FRAME_SCHEMA = "epyc.autokernel.direct_server_response_frame.v1"
DIRECT_RESPONSE_SCHEMA = "epyc.autokernel.direct_server_response.v1"
DIRECT_UNIT_SCHEMA = "epyc.autokernel.direct_server_response_unit.v1"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_TOTAL_RAW_BYTES = 64 * 1024 * 1024
MAX_SLOTS = 1024
PHASES = ("warmup", "measurement")


class ServerResponseRefused(RuntimeError):
    """Original byte evidence is absent, unsupported, or inconsistently bound."""


def _same(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ServerResponseRefused(f"server response {name} differs")


def _bytes(value: Any, maximum: int, label: str) -> bytes:
    if type(value) is not bytes or len(value) > maximum:
        raise ServerResponseRefused(f"{label} is not bounded immutable bytes")
    return value


def _clock(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ServerResponseRefused("server response clock is invalid")
    return float(value)


def source_identity() -> Mapping[str, Any]:
    from . import serving
    return ob._freeze({"schema": RESPONSE_SCHEMA, "max_response_bytes": MAX_RESPONSE_BYTES,
        "max_request_bytes": MAX_REQUEST_BYTES, "max_slots": MAX_SLOTS,
        "max_total_raw_bytes": MAX_TOTAL_RAW_BYTES,
        "callables": [lo.callable_identity(item) for item in (
            serving._measure_once, RawServerResponse.__init__, RawServerResponse.__post_init__,
            ServerResponseCapture.__init__, json.loads,
            ServerResponseCapture.seal, ServerResponseCapture.validate_launch,
            reopen_unit, _bytes, _clock, _same, _frame, _instrument_source,
            source_identity, ServerResponseCapture.for_direct.__func__, _direct_frame,
            reopen_direct_unit, _reopen_rows)],
        "direct_schemas": [DIRECT_FRAME_SCHEMA, DIRECT_RESPONSE_SCHEMA, DIRECT_UNIT_SCHEMA]})


def _direct_frame(context, recipe, prompts):
    """Original working-loop identity, never a fabricated native plan or fence."""
    from .planned_serving import FrozenPromptManifest
    from .resolved_recipe import CanonicalResolvedRecipe
    fields = {"campaign_id", "epoch", "comparison_id", "arm", "launch_index"}
    if not isinstance(context, Mapping) or set(context) != fields:
        raise ServerResponseRefused("direct response context fields differ")
    if any(type(context[key]) is not str or not context[key].strip()
           for key in ("campaign_id", "epoch", "comparison_id")) \
            or context["arm"] not in {"anchor", "candidate"} \
            or type(context["launch_index"]) is not int or context["launch_index"] < 0:
        raise ServerResponseRefused("direct response identity is invalid")
    if type(recipe) is not CanonicalResolvedRecipe or type(prompts) is not FrozenPromptManifest:
        raise ServerResponseRefused("direct capture requires original canonical recipe and prompts")
    recipe = CanonicalResolvedRecipe.from_dict(recipe.to_dict())
    prompts = FrozenPromptManifest.from_dict(prompts.to_dict())
    recipe.validate_launch(recipe.template, recipe.build_dir, recipe.port)
    requests = prompts.requests(tuple(item.prompt_id for item in prompts.prompts), recipe.template)
    if len(requests) != recipe.template.np:
        raise ServerResponseRefused("direct capture requests differ from original slot count")
    return ob._freeze({"schema": DIRECT_FRAME_SCHEMA, "context": dict(context),
        "recipe": recipe.to_dict(), "prompt_manifest_digest": prompts.digest,
        "prompt_ids": [name for name, _ in requests]}), requests


def _frame(plan: Any, unit: Any, fence: Any, recipe: Any, prompts: Any) -> Mapping[str, Any]:
    from . import experiment_plan as ep
    from . import planned_serving as ps
    from . import resolved_recipe as rr
    if (type(plan) is not ep.ExperimentPlan or type(unit) is not ep.UnitSpec
            or type(fence) is not ps.StageFence
            or type(recipe) not in (rr.ResolvedRecipe, rr.CanonicalResolvedRecipe)
            or type(prompts) is not ps.FrozenPromptManifest):
        raise ServerResponseRefused("concrete original serving identities required")
    if (unit not in plan.expected_units or unit.unit_id != fence.unit_id
            or unit.process_id != fence.process_generation_id):
        raise ServerResponseRefused("server response frozen unit identity differs")
    return ob._freeze({"plan_digest": plan.digest, "unit_id": unit.unit_id,
        "arm": unit.arm, "process_generation_id": unit.process_id,
        "fence": asdict(fence), "recipe": recipe.to_dict(),
        "prompt_manifest_digest": prompts.digest,
        "prompt_ids": list(unit.expected_prompt_ids),
        "loaded_instrument": ob._plain(plan.loaded_instrument)})


def _instrument_source(store: mc.ArtifactStore, frame: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(frame.get("loaded_instrument"), Mapping):
        raise ServerResponseRefused("original response-source-bound instrument is unavailable")
    reference = ob.LoadedInstrumentReference.from_dict(ob._plain(frame["loaded_instrument"]))
    identity = lo.validate_instrument_identity(ob._plain(store.read(
        reference.artifact.locator, reference.artifact.sha256)))
    _same(identity["sha256"], reference.identity_sha256, "original instrument identity")
    if not reference.configuration_complete or not identity["configuration_complete"]:
        raise ServerResponseRefused("original response instrument configuration is incomplete")
    recorded = ob._freeze(identity["used_constants"].get("server_response_source"))
    _same(recorded, source_identity(), "original write-time response producer source")
    return recorded


@dataclass(frozen=True, init=False)
class RawServerResponse:
    phase: str
    slot_index: int
    prompt_id: str
    request: bytes
    response: bytes | None
    started_monotonic_s: float
    ended_monotonic_s: float
    error: str | None

    def __init__(self, phase: str, slot_index: int, prompt_id: str, request: bytes,
                 response: bytes | None, started_monotonic_s: float,
                 ended_monotonic_s: float, error: str | None) -> None:
        # An explicit immutable constructor has a closed loaded identity. Python's
        # generated frozen initializer captures a builtin object, which the existing
        # identity machinery correctly leaves unproven; do not relax that machinery.
        for name, value in (("phase", phase), ("slot_index", slot_index), ("prompt_id", prompt_id),
                ("request", request), ("response", response), ("started_monotonic_s", started_monotonic_s),
                ("ended_monotonic_s", ended_monotonic_s), ("error", error)):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (self.phase not in PHASES or type(self.slot_index) is not int
                or not 0 <= self.slot_index < MAX_SLOTS
                or type(self.prompt_id) is not str or not self.prompt_id
                or self.error is not None and type(self.error) is not str):
            raise ServerResponseRefused("server response row has unsupported fields")
        _bytes(self.request, MAX_REQUEST_BYTES, "request")
        if self.response is not None:
            _bytes(self.response, MAX_RESPONSE_BYTES, "response")
        if _clock(self.started_monotonic_s) > _clock(self.ended_monotonic_s):
            raise ServerResponseRefused("server response interval is reversed")


class ServerResponseCapture:
    """One-use concrete sink; no callback, process launch, or deserialized authority."""

    def __init__(self, *, store: mc.ArtifactStore, plan: Any, unit: Any, fence: Any,
                 recipe: Any, prompts: Any, frozen_requests: Sequence[tuple[str, bytes]]) -> None:
        if type(store) is not mc.ArtifactStore:
            raise ServerResponseRefused("server capture requires the concrete artifact store")
        self.store = store
        self.frame = _frame(plan, unit, fence, recipe, prompts)
        self.requests = tuple((name, _bytes(body, MAX_REQUEST_BYTES, "request"))
                              for name, body in frozen_requests)
        if (not 1 <= len(self.requests) <= MAX_SLOTS
                or tuple(name for name, _ in self.requests) != unit.expected_prompt_ids):
            raise ServerResponseRefused("server capture frozen request membership differs")
        by_id = {item.prompt_id: item for item in prompts.prompts}
        _same(self.requests, tuple((name, by_id[name].body) for name in unit.expected_prompt_ids),
              "original request bytes")
        # Reserve the worst case deterministically before any worker request. No
        # racing shared counter, partial omission, or optimistic average response size.
        if len(PHASES) * sum(len(body) + MAX_RESPONSE_BYTES for _, body in self.requests) > MAX_TOTAL_RAW_BYTES:
            raise ServerResponseRefused("server response aggregate raw-byte budget cannot admit all slots")
        self.pins = source_identity()
        if any(item["implementation_status"] != "pinned" or item["configuration_status"] != "pinned"
               for item in self.pins["callables"]):
            raise ServerResponseRefused("server response loaded source identity is incomplete")
        _same(_instrument_source(store, self.frame), self.pins, "original instrument source")
        self._sealed = False

    @classmethod
    def for_direct(cls, *, store, context, recipe, prompts):
        """Use this same recorder in the direct owner; no native issuance is claimed."""
        if cls is not ServerResponseCapture or type(store) is not mc.ArtifactStore:
            raise ServerResponseRefused("direct capture requires the concrete recorder/store")
        frame, requests = _direct_frame(context, recipe, prompts)
        if len(PHASES) * sum(len(body) + MAX_RESPONSE_BYTES for _, body in requests) > MAX_TOTAL_RAW_BYTES:
            raise ServerResponseRefused("direct response aggregate raw-byte budget exceeded")
        result = object.__new__(cls)
        result.store, result.frame, result.requests = store, frame, requests
        result.pins = source_identity()
        if any(item["implementation_status"] != "pinned" or item["configuration_status"] != "pinned"
               for item in result.pins["callables"]):
            raise ServerResponseRefused("direct response loaded source identity is incomplete")
        result._sealed = False
        return result

    def validate_launch(self, recipe: Any, requests: Sequence[tuple[str, bytes]]) -> None:
        _same(self.frame["recipe"], ob._freeze(recipe.to_dict()), "selected recipe")
        _same(self.requests, tuple(requests), "selected requests")

    def seal(self, rows: Sequence[RawServerResponse], *, process_pid: int,
             request_started_monotonic_s: float, request_ended_monotonic_s: float
             ) -> Mapping[str, Any]:
        if self._sealed:
            raise ServerResponseRefused("server response capture already sealed")
        self._sealed = True  # a partial store failure cannot be retried as a fresh observation
        if type(process_pid) is not int or process_pid <= 0:
            raise ServerResponseRefused("server response has no live target PID")
        start, end = _clock(request_started_monotonic_s), _clock(request_ended_monotonic_s)
        retained_at = time.monotonic()
        if not start <= end <= retained_at:
            raise ServerResponseRefused("server response persistence overlaps request window")
        expected = tuple((phase, index, name, body) for phase in PHASES
                         for index, (name, body) in enumerate(self.requests))
        if len(rows) != len(expected) or any(type(row) is not RawServerResponse for row in rows):
            raise ServerResponseRefused("server response phase/slot cardinality differs")
        refs = []
        direct = self.frame.get("schema") == DIRECT_FRAME_SCHEMA
        for sequence, (row, want) in enumerate(zip(rows, expected)):
            _same((row.phase, row.slot_index, row.prompt_id, row.request), want,
                  "phase/slot/request order")
            if not start <= row.started_monotonic_s <= row.ended_monotonic_s <= end:
                raise ServerResponseRefused("server response lies outside original request interval")
            body = {"schema": DIRECT_RESPONSE_SCHEMA if direct else RESPONSE_SCHEMA, "frame": ob._plain(self.frame),
                "process_pid": process_pid, "sequence": sequence,
                "phase": row.phase, "slot_index": row.slot_index, "prompt_id": row.prompt_id,
                "request_hex": row.request.hex(), "request_sha256": hashlib.sha256(row.request).hexdigest(),
                "response_hex": None if row.response is None else row.response.hex(),
                "response_sha256": None if row.response is None else hashlib.sha256(row.response).hexdigest(),
                "started_monotonic_s": row.started_monotonic_s,
                "ended_monotonic_s": row.ended_monotonic_s, "error": row.error,
                "source_identity": ob._plain(self.pins)}
            refs.append(self.store.write(f"server-response:{sequence}", body).to_dict())
        _same(source_identity(), self.pins, "loaded producer source")
        return {"schema": DIRECT_UNIT_SCHEMA if direct else UNIT_SCHEMA, "frame": ob._plain(self.frame),
            "process_pid": process_pid, "request_started_monotonic_s": start,
            "request_ended_monotonic_s": end, "retention_started_monotonic_s": retained_at,
            "retention_ended_monotonic_s": time.monotonic(), "responses": refs,
            "source_identity": ob._plain(self.pins)}


def reopen_unit(value: Mapping[str, Any], *, store: mc.ArtifactStore,
                expected_frame: Mapping[str, Any], expected_requests: Sequence[tuple[str, bytes]],
                expected_pid: int) -> tuple[Mapping[str, Any], ...]:
    """Reparse every retained response, checking bytes; never infer missing token/seed data."""
    if not isinstance(value, Mapping) or value.get("schema") != UNIT_SCHEMA:
        raise ServerResponseRefused("native server unit schema differs")
    return _reopen_rows(value, store=store, expected_frame=expected_frame,
        expected_requests=expected_requests, expected_pid=expected_pid,
        unit_schema=UNIT_SCHEMA, response_schema=RESPONSE_SCHEMA)


def reopen_direct_unit(value, *, store, context, recipe, prompts, expected_pid):
    frame, requests = _direct_frame(context, recipe, prompts)
    return _reopen_rows(value, store=store, expected_frame=frame,
        expected_requests=requests, expected_pid=expected_pid,
        unit_schema=DIRECT_UNIT_SCHEMA, response_schema=DIRECT_RESPONSE_SCHEMA)


def _reopen_rows(value, *, store, expected_frame, expected_requests, expected_pid,
                 unit_schema, response_schema):
    fields = {"schema", "frame", "process_pid", "request_started_monotonic_s",
        "request_ended_monotonic_s", "retention_started_monotonic_s",
        "retention_ended_monotonic_s", "responses", "source_identity"}
    if not isinstance(value, Mapping) or set(value) != fields or value["schema"] != unit_schema:
        raise ServerResponseRefused("server unit receipt has missing or unknown fields")
    value = ob._freeze(ob._plain(value))
    _same(value["frame"], expected_frame, "parent frame")
    _same(value["process_pid"], expected_pid, "parent target PID")
    _same(value["source_identity"], source_identity(), "installed source identity")
    if unit_schema == UNIT_SCHEMA:
        _same(_instrument_source(store, expected_frame), value["source_identity"], "original source closure")
    start, end, retained, closed = (_clock(value[name]) for name in (
        "request_started_monotonic_s", "request_ended_monotonic_s",
        "retention_started_monotonic_s", "retention_ended_monotonic_s"))
    if not start <= end <= retained <= closed:
        raise ServerResponseRefused("server receipt intervals overlap or reverse")
    expected = tuple((phase, index, name, body) for phase in PHASES
                     for index, (name, body) in enumerate(expected_requests))
    refs = value["responses"]
    if not isinstance(refs, (list, tuple)) or len(refs) != len(expected):
        raise ServerResponseRefused("server response reference cardinality differs")
    if (not 1 <= len(expected_requests) <= MAX_SLOTS or len(PHASES) * sum(
            len(_bytes(body, MAX_REQUEST_BYTES, "request")) + MAX_RESPONSE_BYTES
            for _, body in expected_requests) > MAX_TOTAL_RAW_BYTES):
        raise ServerResponseRefused("reopened server response aggregate byte budget exceeded")
    result = []
    for sequence, (ref, want) in enumerate(zip(refs, expected)):
        if not isinstance(ref, Mapping) or set(ref) != {"locator", "sha256", "verified"} \
                or ref["verified"] is not True:
            raise ServerResponseRefused("server response artifact reference is invalid")
        row = store.read(ref["locator"], ref["sha256"])
        if set(row) != {"schema", "frame", "process_pid", "sequence", "phase", "slot_index",
                "prompt_id", "request_hex", "request_sha256", "response_hex", "response_sha256",
                "started_monotonic_s", "ended_monotonic_s", "error", "source_identity"}:
            raise ServerResponseRefused("server response artifact fields differ")
        _same(store.verify(f"server-response:{sequence}", row).to_dict(), dict(ref), "artifact namespace")
        _same((row["schema"], row["frame"], row["process_pid"], row["sequence"], row["source_identity"]),
              (response_schema, expected_frame, expected_pid, sequence, value["source_identity"]), "original identity")
        request = _bytes(bytes.fromhex(row["request_hex"]), MAX_REQUEST_BYTES, "request")
        response = None if row["response_hex"] is None else _bytes(
            bytes.fromhex(row["response_hex"]), MAX_RESPONSE_BYTES, "response")
        _same((row["phase"], row["slot_index"], row["prompt_id"], request), want, "complete request order")
        _same(hashlib.sha256(request).hexdigest(), row["request_sha256"], "request bytes")
        _same(None if response is None else hashlib.sha256(response).hexdigest(),
              row["response_sha256"], "response bytes")
        captured = RawServerResponse(row["phase"], row["slot_index"], row["prompt_id"],
            request, response, row["started_monotonic_s"], row["ended_monotonic_s"], row["error"])
        if not start <= captured.started_monotonic_s <= captured.ended_monotonic_s <= end:
            raise ServerResponseRefused("reopened response interval differs")
        try:
            parsed = None if response is None else json.loads(response)
        except (ValueError, UnicodeDecodeError):
            parsed = None
        result.append(ob._freeze({"artifact": dict(ref), "raw": ob._plain(row),
                                  "request": json.loads(request), "response": parsed}))
    return tuple(result)
