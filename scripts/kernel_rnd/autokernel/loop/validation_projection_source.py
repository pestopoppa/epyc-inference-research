"""Prospective canonical receipt provenance, never live or scientific authority."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .. import schemas
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_producer_source as nps
from . import observation_binding as ob

SOURCE_SCHEMA = "epyc.autokernel.canonical_projection_source.v2"
RECEIPT_SCHEMA = "epyc.autokernel.canonical_claim_grade_receipt.v2"
PRODUCER_ID = "autokernel.loop.validation_semantic_adapter/v2"
CAPTURE_SCHEMA = "epyc.autokernel.unified_arm_capture.v3"
PROJECTOR_NAME = "autokernel-unified-arm-measurement"
SOURCE_FIELDS = ("schema", "root_commit", "root_source_schema", "root_source_sha256",
    "adapter_id", "projector_name", "capture_schema", "expected_native_producer_source",
    "receipt_producer")
PRODUCER_ROLES = ("build_receipt", "source_identity", "validate_source_identity",
    "validate_current_receipt", "reopen_native", "projection_source.__post_init__",
    "projection.produce_receipt", "projection.reopen_receipt",
    "projection._current_receipt_body", "receipt.validate_receipt_body",
    "root_loader.load", "root_loader.source_schema", "root_loader.source_bytes", "native_sources_bound")


def _closed(value, fields, label):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label} has missing or unknown fields")
    return ob._plain(value)


@dataclass(frozen=True)
class ProjectionSourceClosure:
    root_commit: str
    root_source_sha256: Mapping[str, str]
    expected_native_producer_source: Mapping[str, Any]

    def __post_init__(self):
        from . import feed_runtime as feed
        if (not isinstance(self.root_commit, str) or len(self.root_commit) != 40
                or any(c not in "0123456789abcdef" for c in self.root_commit)):
            raise ValueError("ROOT commit must be an exact lowercase commit identity")
        if not isinstance(self.root_source_sha256, Mapping):
            raise ValueError("ROOT source pins must be a mapping")
        pins = dict(self.root_source_sha256)
        if feed._root_source_schema(pins) != feed.ROOT_PROJECTION_SCHEMA_V2:
            raise ValueError("current semantic receipts require the seven-file ROOT closure")
        expected = nps.validate_producer_source_closure(self.expected_native_producer_source)
        object.__setattr__(self, "root_source_sha256", ob._freeze(pins))
        object.__setattr__(self, "expected_native_producer_source", expected)


def source_identity(projection):
    from . import feed_runtime as feed
    from . import validation_claim_receipt as cr
    from . import validation_semantic_adapter as sa
    selected = projection.source_closure
    functions = (build_receipt, source_identity, validate_source_identity,
        validate_current_receipt, reopen_native, ProjectionSourceClosure.__post_init__,
        sa.PinnedRootProjection.produce_receipt, sa.PinnedRootProjection.reopen_receipt,
        sa.PinnedRootProjection._current_receipt_body, cr.validate_receipt_body,
        feed.LoadedFeedProjection.load.__func__, feed._root_source_schema, feed._source_bytes,
        native_sources_bound)
    row = {"schema": SOURCE_SCHEMA, "root_commit": selected.root_commit,
        "root_source_schema": projection.loaded_projection.source_schema,
        "root_source_sha256": dict(projection.loaded_projection.source_sha256),
        "adapter_id": projection.arm_adapter.ADAPTER_ID, "projector_name": PROJECTOR_NAME,
        "capture_schema": CAPTURE_SCHEMA,
        "expected_native_producer_source": ob._plain(selected.expected_native_producer_source),
        "receipt_producer": {"producer_id": PRODUCER_ID, "callables": [
            {"role": role, "identity": lo.callable_identity(function)}
            for role, function in zip(PRODUCER_ROLES, functions)]}}
    result = validate_source_identity(row)
    installed = getattr(projection, "_installed_source_identity", None)
    if installed is not None and result != installed:
        raise ValueError("installed semantic receipt implementation/configuration changed")
    return result if installed is None else installed


def validate_source_identity(value):
    from . import feed_runtime as feed
    row = _closed(value, SOURCE_FIELDS, "current projection source")
    if (row["schema"] != SOURCE_SCHEMA
            or row["root_source_schema"] != feed.ROOT_PROJECTION_SCHEMA_V2
            or row["projector_name"] != PROJECTOR_NAME or row["capture_schema"] != CAPTURE_SCHEMA
            or row["adapter_id"] != "vidya.adapters.autokernel_unified_arm/v1"):
        raise ValueError("current projection source contract differs")
    selected = ProjectionSourceClosure(row["root_commit"], row["root_source_sha256"],
                                       row["expected_native_producer_source"])
    producer = _closed(row["receipt_producer"], ("producer_id", "callables"), "receipt producer")
    if producer["producer_id"] != PRODUCER_ID:
        raise ValueError("receipt producer identity differs")
    producer["callables"] = nps._callables(producer["callables"], PRODUCER_ROLES)
    row.update(root_source_sha256=dict(selected.root_source_sha256),
        expected_native_producer_source=ob._plain(selected.expected_native_producer_source),
        receipt_producer=producer)
    return ob._freeze(row)


def reopen_native(projection, event, store):
    """Reopen original artifacts; never invoke the original live issuer again."""
    from . import experiment_plan as ep
    payload = event.get("payload") if isinstance(event, Mapping) else None
    if (not isinstance(payload, Mapping) or payload.get("schema") != CAPTURE_SCHEMA
            or not isinstance(payload.get("carrier"), Mapping)
            or payload["carrier"].get("schema") != CAPTURE_SCHEMA):
        raise ValueError("current canonical receipt requires a final-v3 source event")
    # This owning readback validates the full original/final artifact graph even
    # when there is no measurement tuple. No reconstructed native capability.
    projected = projection.arm_adapter.project_journal_event(ob._plain(event), corpus_root=store.root)
    carrier = ob._plain(payload["carrier"])
    final_ref = carrier["parent_final_trial"]
    final = ob._plain(store.read(final_ref["artifact"]["locator"], final_ref["artifact"]["sha256"]))
    if final_ref["digest"] != schemas.content_hash(final):
        raise ValueError("original final trial digest differs")
    plan = ep.ExperimentPlan.from_dict(carrier["plan"])
    rows = tuple(ep.RawUnit.from_dict(item["row"]) for item in final["final_rows"])
    view = ep.admissible_units(plan, rows)
    if (view.to_dict() != final["final_admissible_view"]
            or view.to_dict() != carrier["admissible_view"]):
        raise ValueError("original full final view differs from owning reducer")
    instrument_ref = ob.LoadedInstrumentReference.from_dict(carrier["loaded_instrument"])
    instrument = lo.validate_instrument_identity(ob._plain(store.read(
        instrument_ref.artifact.locator, instrument_ref.artifact.sha256)))
    if instrument["sha256"] != instrument_ref.identity_sha256:
        raise ValueError("original instrument source identity differs")
    closure = instrument["used_constants"].get("producer_source_closure")
    if closure is not None:
        closure = ob._plain(nps.validate_producer_source_closure(closure))
    provenance = {"loaded_instrument": carrier["loaded_instrument"],
        "producer_source_closure": closure, "parent_final_source_identity": final["source_identity"]}
    return projected, carrier, plan, view, provenance


def native_sources_bound(provenance, expected):
    from .native_final_trial import source_identities
    from .native_server_t0_witness import server_source_identities
    original = provenance["producer_source_closure"]
    if original is None or original != expected or not nps.producer_source_closure_complete(original):
        return False
    selected = expected.get("scientific_adapters", {}).get("correctness")
    final = provenance["parent_final_source_identity"]
    if (not isinstance(selected, Mapping)
            or final.get("selected_scientific_adapter") != selected
            or final.get("finalizer") != selected.get("owning_source_pins", {}).get("final_trial_source")):
        return False
    identities = source_identities(final["finalizer"]) + server_source_identities(selected)
    return all(item["implementation_status"] == item["configuration_status"] == "pinned"
               for item in identities)


def build_receipt(projection, event, *, source_store, source_reference):
    projected, carrier, _plan, view, provenance = reopen_native(projection, event, source_store)
    source = ob._plain(source_identity(projection))
    if projected is None:
        result = {"status": "diagnostic", "reason": projection.arm_adapter.diagnostic_reason(
            ob._plain(event), corpus_root=source_store.root)}
    else:
        claim = mc._plain(asdict(projected))
        grade, trace, reasons = projection.claim_tuple.grade(projected)
        result = {"status": "measurement", "claim_tuple": claim,
            "claim_tuple_digest": schemas.content_hash(claim), "source_grade": grade,
            "trace_grade": trace, "reasons": list(reasons)}
    identities = [item["identity"] for item in source["receipt_producer"]["callables"]]
    source_bound = (native_sources_bound(provenance, source["expected_native_producer_source"])
        and all(item["implementation_status"] == item["configuration_status"] == "pinned"
                for item in identities))
    binding = {"measurement_id": event["record_id"], "arm": carrier["arm"],
        "plan_digest": schemas.content_hash(carrier["plan"]), "lineage_id": carrier["lineage_id"],
        "comparison_identities_digest": schemas.content_hash(carrier["comparison_identities"]),
        "instrument_identity_sha256": carrier["loaded_instrument"]["identity_sha256"],
        "capture_schema": CAPTURE_SCHEMA, "parent_final_trial": carrier["parent_final_trial"],
        "original_arm_capture": carrier["original_arm_capture"], "final_view_digest": view.view_digest}
    body = {"schema": RECEIPT_SCHEMA, "producer": PRODUCER_ID, "source_identity": source,
        "source_event": dict(source_reference), "projection": result, "native_binding": binding,
        "native_provenance": provenance,
        "authority_scope": "final_pinned_source" if source_bound else "compatibility_only"}
    body["receipt_id"] = "claim-grade-" + schemas.content_hash(body)[:24]
    return validate_current_receipt(body)


def validate_current_receipt(value):
    from . import validation_claim_receipt as cr
    row = _closed(value, ("schema", "producer", "receipt_id", "source_identity", "source_event",
        "projection", "native_binding", "native_provenance", "authority_scope"), "current claim receipt")
    if row["schema"] != RECEIPT_SCHEMA or row["producer"] != PRODUCER_ID:
        raise ValueError("current claim receipt schema/producer differs")
    row["source_identity"] = ob._plain(validate_source_identity(row["source_identity"]))
    event = _closed(row["source_event"], ("locator", "sha256"), "source event")
    cr._text(event["locator"], "source locator")
    cr._sha(event["sha256"], "source digest")
    result = row["projection"]
    if isinstance(result, Mapping) and result.get("status") == "diagnostic":
        result = _closed(result, ("status", "reason"), "diagnostic projection")
        cr._text(result["reason"], "diagnostic reason")
    else:
        result = _closed(result, ("status", "claim_tuple", "claim_tuple_digest", "source_grade",
            "trace_grade", "reasons"), "measurement projection")
        if result["status"] != "measurement" or not isinstance(result["claim_tuple"], Mapping):
            raise ValueError("measurement projection is malformed")
        if result["claim_tuple_digest"] != schemas.content_hash(result["claim_tuple"]):
            raise ValueError("canonical ClaimTuple digest differs")
        for name in ("source_grade", "trace_grade"):
            cr._text(result[name], name)
        if not isinstance(result["reasons"], list) or any(not isinstance(item, str) or not item
                                                       for item in result["reasons"]):
            raise ValueError("grade reasons must be text array")
    binding = _closed(row["native_binding"], ("measurement_id", "arm", "plan_digest", "lineage_id",
        "comparison_identities_digest", "instrument_identity_sha256", "capture_schema",
        "parent_final_trial", "original_arm_capture", "final_view_digest"), "current native binding")
    for name in ("measurement_id", "plan_digest", "comparison_identities_digest",
                 "instrument_identity_sha256", "final_view_digest"):
        cr._sha(binding[name], name)
    cr._text(binding["lineage_id"], "lineage")
    if binding["arm"] not in ("anchor", "candidate") or binding["capture_schema"] != CAPTURE_SCHEMA:
        raise ValueError("current native arm/schema differs")
    final = _closed(binding["parent_final_trial"], ("schema", "artifact", "digest"), "final reference")
    if final["schema"] != "epyc.autokernel.parent_final_trial_reference.v1":
        raise ValueError("parent final reference schema differs")
    cr._sha(final["digest"], "final digest")
    original = _closed(binding["original_arm_capture"], ("measurement_id", "carrier_digest", "artifact"),
                       "original arm reference")
    cr._sha(original["measurement_id"], "original id")
    cr._sha(original["carrier_digest"], "original digest")
    for artifact in (final["artifact"], original["artifact"]):
        item = _closed(artifact, ("locator", "sha256", "verified"), "source artifact")
        cr._text(item["locator"], "artifact locator")
        cr._sha(item["sha256"], "artifact digest")
        if item["verified"] is not True:
            raise ValueError("original artifact is not producer verified")
    provenance = _closed(row["native_provenance"], ("loaded_instrument", "producer_source_closure",
        "parent_final_source_identity"), "native provenance")
    instrument = ob.LoadedInstrumentReference.from_dict(provenance["loaded_instrument"])
    if instrument.identity_sha256 != binding["instrument_identity_sha256"]:
        raise ValueError("native source instrument differs")
    if provenance["producer_source_closure"] is not None:
        nps.validate_producer_source_closure(provenance["producer_source_closure"])
    sources = _closed(provenance["parent_final_source_identity"], ("finalizer", "selected_scientific_adapter"),
                      "parent final source identity")
    # Its complete original schema is verified by captured ROOT during produce
    # and replay; this grammar cannot manufacture a currently installed issuer.
    if any(not isinstance(item, Mapping) for item in sources.values()):
        raise ValueError("parent final source identity must contain objects")
    if row["authority_scope"] not in ("compatibility_only", "final_pinned_source"):
        raise ValueError("current receipt authority scope differs")
    expected = schemas.content_hash({key: item for key, item in row.items() if key != "receipt_id"})
    if row["receipt_id"] != "claim-grade-" + expected[:24]:
        raise ValueError("current receipt id does not rederive")
    return ob._freeze(row)
