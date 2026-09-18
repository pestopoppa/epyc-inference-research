"""Prospective loaded producer identity, not historical on-read source invention."""
from __future__ import annotations

import time
from contextlib import closing
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import native_producer_source as source
from . import observation_binding as ob
from . import serving


def _validator(cls=nc.NativeCaptureValidator):
    validator = object.__new__(cls)
    validator.observation_verifiers = ob.ParentObservationVerifiers()
    return validator


def test_actual_deferred_methods_and_actual_v2_validator_are_selected():
    row = source.loaded_producer_source_closure()
    capture = row["measurement_capture"]
    assert capture["producer_id"] == mc.PRODUCER_ID_V2
    assert capture["capture_schema"] == mc.CAPTURE_SCHEMA_V2
    for actual, expected in zip(capture["callables"], (
            mc.DeferredNativeMeasurementSink.__call__, mc.DeferredNativeMeasurementSink._build_payloads,
            mc.DeferredNativeMeasurementSink.finalize_run)):
        assert actual["identity"] == lo.callable_identity(expected)
    assert capture["callables"][-1]["identity"] != lo.callable_identity(mc.NativeMeasurementSink.finalize_run)
    assert row["native_validator"]["callables"][0]["identity"] == lo.callable_identity(nc.NativeCaptureValidator.prevalidate)
    assert row["native_validator"]["callables"][1]["identity"] == lo.callable_identity(nc.NativeCaptureValidator.validate_prevalidated)
    assert all(item["identity"]["qualname"] != "NativeCaptureValidator.validate"
               for item in row["native_validator"]["callables"])
    with pytest.raises(TypeError):
        capture["producer_id"] = "mutated"


@pytest.mark.parametrize("mutation", ["extra", "schema", "missing_group", "missing_role",
    "duplicate", "reorder", "unknown_role", "extra_identity", "invalid_status", "invalid_digest"])
def test_closed_source_one_fact_mutations_refused(mutation):
    row = ob._plain(source.loaded_producer_source_closure())
    calls = row["measurement_capture"]["callables"]
    if mutation == "extra":
        row["extra"] = True
    elif mutation == "schema":
        row["schema"] += ".invented"
    elif mutation == "missing_group":
        del row["native_validator"]
    elif mutation == "missing_role":
        calls.pop()
    elif mutation == "duplicate":
        calls[1] = calls[0]
    elif mutation == "reorder":
        calls.reverse()
    elif mutation == "unknown_role":
        calls[0]["role"] = "generic.success"
    elif mutation == "extra_identity":
        calls[0]["identity"]["source_path"] = "not-a-loaded-warrant"
    elif mutation == "invalid_status":
        calls[0]["identity"]["implementation_status"] = "verified"
    else:
        calls[0]["identity"]["implementation_sha256"] = "not-a-digest"
    with pytest.raises(ob.ObservationBindingError):
        source.validate_producer_source_closure(row)


def test_unproven_identity_is_preserved_not_promoted():
    row = ob._plain(source.loaded_producer_source_closure())
    for group in ("measurement_capture", "observation_binding", "native_validator"):
        for item in row[group]["callables"]:
            item["identity"].update(implementation_status="pinned", implementation_sha256="a" * 64,
                                    configuration_status="pinned", configuration_sha256="b" * 64)
    assert source.producer_source_closure_complete(row)
    row["measurement_capture"]["callables"][0]["identity"].update(
        implementation_status="unproven", implementation_sha256="c" * 64)
    validated = source.validate_producer_source_closure(row)
    assert validated["measurement_capture"]["callables"][0]["identity"]["implementation_status"] == "unproven"
    assert not source.producer_source_closure_complete(validated)


def _sealed(store, *, omit=False):
    identity = ob._plain(ob.loaded_planned_serving_identity(
        measurement_callable=serving._measure_once, fence_clock=time.monotonic, serving_timer=time.time))
    if omit:
        identity["used_constants"].pop("producer_source_closure")
        identity.pop("sha256")
        identity["sha256"] = lo._digest(identity)
    artifact = store.write(f"loaded-instrument:{identity['sha256']}", identity)
    return {"loaded_instrument": ob.LoadedInstrumentReference(
        identity["sha256"], identity["configuration_complete"], artifact).to_dict()}, identity


def test_source_is_frozen_inside_instrument_and_reopened_at_capture(tmp_path):
    with closing(mc.ArtifactStore(tmp_path)) as store:
        carrier, identity = _sealed(store)
        assert identity["used_constants"]["producer_source_closure"] == ob._plain(source.loaded_producer_source_closure())
        validator = _validator()
        assert source.verify_capture_producer_source(carrier, store=store, validator=validator) is source.producer_source_closure_complete(
            identity["used_constants"]["producer_source_closure"])
        assert source.verify_capture_producer_source(carrier, store=store,
            validator_type=nc.NativeCaptureValidator) is source.verify_capture_producer_source(carrier, store=store)
        with pytest.raises(source.ProducerSourceRefused, match="not both"):
            source.verify_capture_producer_source(carrier, store=store,
                validator=validator, validator_type=nc.NativeCaptureValidator)


def test_historical_absence_is_not_reconstructed(tmp_path):
    with closing(mc.ArtifactStore(tmp_path)) as store:
        carrier, identity = _sealed(store, omit=True)
        assert not source.verify_capture_producer_source(
            carrier, store=store, validator=_validator())
        assert "producer_source_closure" not in identity["used_constants"]


@pytest.mark.parametrize("target", ["capture", "observation", "validator"])
def test_loaded_implementation_change_invalidates_original_pin(tmp_path, monkeypatch, target):
    with closing(mc.ArtifactStore(tmp_path)) as store:
        carrier, original = _sealed(store)
        def changed(*args, **kwargs):
            return None
        if target == "capture":
            monkeypatch.setattr(mc.DeferredNativeMeasurementSink, "finalize_run", changed)
        elif target == "observation":
            monkeypatch.setattr(ob, "seal_observation", changed)
        else:
            monkeypatch.setattr(nc.NativeCaptureValidator, "_verify_artifacts", changed)
        current = ob.loaded_planned_serving_identity(measurement_callable=serving._measure_once,
            fence_clock=time.monotonic, serving_timer=time.time)
        assert current["sha256"] != original["sha256"]
        with pytest.raises(source.ProducerSourceRefused, match="selected native producer"):
            source.verify_capture_producer_source(carrier, store=store,
                validator=_validator())


def test_selected_validator_subclass_is_not_hidden_by_base_name(tmp_path):
    class ChangedValidator(nc.NativeCaptureValidator):
        def _verify_artifacts(self, *args):
            return None
    with closing(mc.ArtifactStore(tmp_path)) as store:
        carrier, _ = _sealed(store)
        with pytest.raises(source.ProducerSourceRefused, match="selected native producer"):
            source.verify_capture_producer_source(carrier, store=store,
                validator=_validator(ChangedValidator))


def test_actual_loaded_default_producer_scope_is_complete():
    assert source.producer_source_closure_complete(source.loaded_producer_source_closure())


def test_actual_selected_verifier_cannot_hide_behind_unbound_class(tmp_path):
    with closing(mc.ArtifactStore(tmp_path)) as store:
        carrier, _ = _sealed(store)
        validator = _validator()
        def configured_verifier(*args, **kwargs):
            return {"status": "unknown", "kind": None, "evidence_ref": None}
        validator.observation_verifiers = ob.ParentObservationVerifiers(purpose=configured_verifier)
        configured = source.loaded_producer_source_closure(
            observation_verifiers=validator.observation_verifiers)
        assert configured["native_validator"]["observation_verifiers"]["purpose"] == lo.callable_identity(configured_verifier)
        with pytest.raises(source.ProducerSourceRefused, match="selected native producer"):
            source.verify_capture_producer_source(carrier, store=store, validator=validator)


def _constant_function(value):
    def function():
        return None
    code = function.__code__.replace(co_consts=(None, value))
    return types.FunctionType(code, globals(), "constant_function")


def test_exact_string_frozenset_is_order_independent_and_member_sensitive():
    first = lo.callable_identity(_constant_function(frozenset(("a", "b"))))
    reordered = lo.callable_identity(_constant_function(frozenset(("b", "a"))))
    changed = lo.callable_identity(_constant_function(frozenset(("a", "c"))))
    assert first["implementation_status"] == "pinned"
    assert first == reordered
    assert first["implementation_sha256"] != changed["implementation_sha256"]
    assert lo._stable_json_value(frozenset(("a", "b"))) == ("unproven", None)


@pytest.mark.parametrize("kind", ["non_string", "subclass", "member_subclass", "mutable", "oversized"])
def test_other_constant_types_remain_unproven(kind):
    class FrozenSubclass(frozenset):
        pass
    class TextSubclass(str):
        pass
    value = {"non_string": frozenset((1, 2)), "subclass": FrozenSubclass(("a",)),
        "member_subclass": frozenset((TextSubclass("a"),)), "mutable": {"a", "b"},
        "oversized": frozenset(str(i) for i in range(4097))}[kind]
    assert lo.callable_identity(_constant_function(value))["implementation_status"] == "unproven"


def test_compiler_string_set_identity_is_cross_hash_seed_reproducible():
    script = """from autokernel.loop import lifecycle_observation as lo
def member(value):
    return value in {'alpha', 'beta', 'gamma'}
print(lo.callable_identity(member)['implementation_sha256'])
"""
    package_root = str(Path(lo.__file__).resolve().parents[2])
    digests = [subprocess.check_output([sys.executable, "-c", script], text=True,
        env={**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": package_root}, timeout=5).strip()
        for seed in (1, 42, 999)]
    assert len(set(digests)) == 1
