"""Prospective loaded identities for the enumerated native v2 producer path.

This is an explicit named-method scope, not an inferred transitive call graph or
scientific warrant.  Incomplete loaded identities remain incomplete.  Historical
instruments without this field cannot be reconstructed from today's installation.
"""
from __future__ import annotations

from typing import Any, Mapping

from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import observation_binding as ob


PRODUCER_SOURCE_SCHEMA = "epyc.autokernel.native_capture_producer_source.v1"
PRODUCER_SOURCE_SCHEMA_V2 = "epyc.autokernel.native_capture_producer_source.v2"
CAPTURE_ROLES = ("deferred_sink.__call__", "deferred_sink._build_payloads",
                 "deferred_sink.finalize_run")
OBSERVATION_ROLES = ("seal_observation", "validate_reopened_observation",
    "parent_status", "validate_observation", "validate_instrument_identity",
    "artifact_store.read", "artifact_store.verify")
VALIDATOR_ROLES = ("native_validator.prevalidate", "native_validator.validate_prevalidated",
    "native_validator._validate_carrier", "native_validator._validate_binding",
    "native_validator._verify_artifacts", "native_validator._verify_observations",
    "native_validator._validate_supplied_fence")
CALLABLE_FIELDS = ("module", "qualname", "kind", "implementation_status",
                   "implementation_sha256", "configuration_status", "configuration_sha256")
SCHEMA_FIELDS = ("loaded_instrument_reference", "observation_unit_binding",
                 "lifecycle_observation_reference", "lifecycle_observation_link")
VERIFIER_FIELDS = ("observation", "purpose", "runtime", "gpu")


class ProducerSourceRefused(ob.ObservationBindingError):
    pass


def _closed(value: Any, fields: tuple[str, ...], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ProducerSourceRefused(f"{label} has missing or unknown fields")
    return dict(value)


def _identity(value: Any) -> dict[str, Any]:
    identity = _closed(value, CALLABLE_FIELDS, "producer loaded identity")
    for name in ("module", "qualname"):
        ob._text(identity[name], name)
    if identity["kind"] not in ("python", "builtin_or_extension"):
        raise ProducerSourceRefused("producer callable kind is unsupported")
    for name in ("implementation", "configuration"):
        status, digest = identity[f"{name}_status"], identity[f"{name}_sha256"]
        if status not in ("pinned", "unproven"):
            raise ProducerSourceRefused("producer identity status is unsupported")
        if status == "pinned" or digest is not None:
            ob._sha(digest, f"producer {name} digest")
    return ob._plain(identity)


def _callables(value: Any, roles: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or len(value) != len(roles):
        raise ProducerSourceRefused("producer callable scope is incomplete or oversized")
    result = []
    for supplied, role in zip(value, roles):
        row = _closed(supplied, ("role", "identity"), "producer role")
        if row["role"] != role:
            raise ProducerSourceRefused("producer roles must occur exactly once in fixed order")
        result.append({"role": role, "identity": _identity(row["identity"])})
    return result


def validate_producer_source_closure(value: Any) -> Mapping[str, Any]:
    """Validate a closed original record without comparing it to installed code."""
    v2 = isinstance(value, Mapping) and value.get("schema") == PRODUCER_SOURCE_SCHEMA_V2
    fields = ("schema", "measurement_capture", "observation_binding", "native_validator")
    row = _closed(value, fields + (("scientific_adapters",) if v2 else ()), "producer source closure")
    if row["schema"] not in (PRODUCER_SOURCE_SCHEMA, PRODUCER_SOURCE_SCHEMA_V2):
        raise ProducerSourceRefused("producer source schema is unsupported")
    capture = _closed(row["measurement_capture"], ("producer_id", "capture_schema", "callables"),
                      "measurement capture source")
    ob._text(capture["producer_id"], "producer id")
    ob._text(capture["capture_schema"], "capture schema")
    capture["callables"] = _callables(capture["callables"], CAPTURE_ROLES)
    observation = _closed(row["observation_binding"], ("schemas", "callables"),
                          "observation binding source")
    observation["schemas"] = _closed(observation["schemas"], SCHEMA_FIELDS, "observation schemas")
    for name, schema in observation["schemas"].items():
        ob._text(schema, name)
    observation["callables"] = _callables(observation["callables"], OBSERVATION_ROLES)
    validator = _closed(row["native_validator"], ("callables", "observation_verifiers"),
                        "native validator source")
    validator["callables"] = _callables(validator["callables"], VALIDATOR_ROLES)
    verifiers = _closed(validator["observation_verifiers"], VERIFIER_FIELDS,
                        "selected parent observation verifiers")
    validator["observation_verifiers"] = {
        name: None if item is None else _identity(item) for name, item in verifiers.items()}
    result = {"schema": row["schema"], "measurement_capture": capture,
              "observation_binding": observation, "native_validator": validator}
    if v2:
        from .native_scientific_witness import validate_scientific_source
        result["scientific_adapters"] = validate_scientific_source(row["scientific_adapters"])
    return ob._freeze(result)


def producer_source_closure_complete(value: Any) -> bool:
    """Report loaded-code/config pin completeness, never acceptance or eligibility."""
    row = validate_producer_source_closure(value)
    identities = [item["identity"]
        for group in ("measurement_capture", "observation_binding", "native_validator")
        for item in row[group]["callables"]]
    identities.extend(item for item in row["native_validator"]["observation_verifiers"].values()
                      if item is not None)
    if row["schema"] == PRODUCER_SOURCE_SCHEMA_V2:
        from .native_scientific_witness import scientific_source_identities
        identities.extend(scientific_source_identities(row["scientific_adapters"]))
    return all(item["implementation_status"] == "pinned"
               and item["configuration_status"] == "pinned" for item in identities)


def loaded_producer_source_closure(*, capture_type: type | None = None,
                                  validator_type: type | None = None,
                                  observation_verifiers: ob.ParentObservationVerifiers | None = None,
                                  scientific_adapters: Any = None
                                  ) -> Mapping[str, Any]:
    """Resolve actual selected methods, including inherited deferred-sink methods."""
    from . import native_capture_control as nc
    capture_type = mc.DeferredNativeMeasurementSink if capture_type is None else capture_type
    validator_type = nc.NativeCaptureValidator if validator_type is None else validator_type
    if observation_verifiers is None:
        observation_verifiers = ob.ParentObservationVerifiers()
    if type(observation_verifiers) is not ob.ParentObservationVerifiers:
        raise ProducerSourceRefused("selected observation verifier configuration must be concrete")
    verifier_identities = {name: None if getattr(observation_verifiers, name) is None
        else lo.callable_identity(getattr(observation_verifiers, name)) for name in VERIFIER_FIELDS}
    capture_functions = (capture_type.__call__, capture_type._build_payloads,
                         capture_type.finalize_run)
    observation_functions = (ob.seal_observation, ob.validate_reopened_observation,
        ob._parent_status, lo.validate_observation, lo.validate_instrument_identity,
        mc.ArtifactStore.read, mc.ArtifactStore.verify)
    validator_functions = (validator_type.prevalidate, validator_type.validate_prevalidated,
        validator_type._validate_carrier, validator_type._validate_binding,
        validator_type._verify_artifacts, validator_type._verify_observations,
        validator_type._validate_supplied_fence)
    def records(roles, functions):
        return [{"role": role, "identity": lo.callable_identity(function)}
                for role, function in zip(roles, functions)]
    result = {"schema": PRODUCER_SOURCE_SCHEMA,
        "measurement_capture": {"producer_id": mc.PRODUCER_ID_V2,
            "capture_schema": mc.CAPTURE_SCHEMA_V2,
            "callables": records(CAPTURE_ROLES, capture_functions)},
        "observation_binding": {"schemas": {
            "loaded_instrument_reference": ob.INSTRUMENT_REFERENCE_SCHEMA,
            "observation_unit_binding": ob.UNIT_BINDING_SCHEMA,
            "lifecycle_observation_reference": ob.OBSERVATION_REFERENCE_SCHEMA,
            "lifecycle_observation_link": ob.OBSERVATION_LINK_SCHEMA},
            "callables": records(OBSERVATION_ROLES, observation_functions)},
        "native_validator": {"callables": records(VALIDATOR_ROLES, validator_functions),
                             "observation_verifiers": verifier_identities}}
    if scientific_adapters is not None:
        from .native_scientific_witness import ParentScientificWitnessAdapters
        if type(scientific_adapters) is not ParentScientificWitnessAdapters:
            raise ProducerSourceRefused("scientific producer configuration must be concrete")
        result.update(schema=PRODUCER_SOURCE_SCHEMA_V2,
                      scientific_adapters=scientific_adapters.source_identity())
    return validate_producer_source_closure(result)


def verify_capture_producer_source(carrier: Mapping[str, Any], *, store: mc.ArtifactStore,
                                   validator: Any = None,
                                   validator_type: type | None = None) -> bool:
    """Check selected code against original prospective bytes; old absence stays absent."""
    if validator is not None and validator_type is not None:
        raise ProducerSourceRefused("select a validator instance or type, not both")
    if validator is not None:
        validator_type = type(validator)
    selected_verifiers = None if validator is None else validator.observation_verifiers
    instrument = ob.LoadedInstrumentReference.from_dict(carrier["loaded_instrument"])
    identity = lo.validate_instrument_identity(ob._plain(store.read(
        instrument.artifact.locator, instrument.artifact.sha256)))
    store.verify(f"loaded-instrument:{identity['sha256']}", identity)
    if identity["sha256"] != instrument.identity_sha256:
        raise ProducerSourceRefused("producer source instrument identity differs")
    closure = identity["used_constants"].get("producer_source_closure")
    if closure is None:
        return False
    expected = validate_producer_source_closure(closure)
    scientific_adapters = None
    if expected["schema"] == PRODUCER_SOURCE_SCHEMA_V2:
        if validator is None:
            from .native_scientific_witness import installed_scientific_adapters
            scientific_adapters = installed_scientific_adapters(expected["scientific_adapters"])
        else:
            replayer = validator.parent_receipt_replayer
            if replayer is None:
                raise ProducerSourceRefused("original selected scientific authority is unavailable")
            scientific_adapters = replayer.selected_scientific_adapters()
    actual = loaded_producer_source_closure(validator_type=validator_type,
        observation_verifiers=selected_verifiers, scientific_adapters=scientific_adapters)
    if expected != actual:
        raise ProducerSourceRefused("selected native producer source differs from issued instrument")
    return producer_source_closure_complete(expected)


__all__ = ["PRODUCER_SOURCE_SCHEMA", "PRODUCER_SOURCE_SCHEMA_V2", "ProducerSourceRefused",
    "loaded_producer_source_closure", "validate_producer_source_closure",
    "producer_source_closure_complete", "verify_capture_producer_source"]
