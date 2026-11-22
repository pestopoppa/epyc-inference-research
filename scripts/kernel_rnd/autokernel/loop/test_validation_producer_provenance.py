"""AKU-08g: typed original-producer provenance warrant at canonical reopening.

Hermetic fixtures only: closures are resolved from the code loaded in this test
process (and in hash-seeded child interpreters); no receipt, run or measurement
is fabricated, and a warrant here is provenance only, never a grade or use.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import native_producer_source as nps
from . import native_scientific_witness as scientific
from . import native_server_t0_witness as server
from . import observation_binding as ob
from . import validation_projection_source as current


def _installed(**kwargs):
    adapters = scientific.ParentScientificWitnessAdapters(
        correctness=server.NativeServerT0WitnessAdapter(max_units=kwargs.pop("max_units", 4)))
    return ob._plain(nps.loaded_producer_source_closure(scientific_adapters=adapters, **kwargs))


def _provenance(closure, expected=None):
    selected = (expected or closure or {}).get("scientific_adapters", {}).get("correctness") or {}
    return {"loaded_instrument": None, "producer_source_closure": closure,
            "parent_final_source_identity": {
                "finalizer": selected.get("owning_source_pins", {}).get("final_trial_source"),
                "selected_scientific_adapter": selected}}


def test_matching_captured_closure_carries_provenance_warrant():
    expected = _installed()
    warrant = current.native_sources_bound(_provenance(_installed(), expected), expected, typed=True)
    assert warrant == current.ProducerProvenance(True, ())
    assert current.native_sources_bound(_provenance(_installed(), expected), expected) is True


def _changed_bytes(closure):
    closure["measurement_capture"]["callables"][0]["identity"]["implementation_sha256"] = "e" * 64


def _changed_slot(closure):
    def configured_verifier(*_args, **_kwargs):
        return {"status": "unknown", "kind": None, "evidence_ref": None}
    closure["native_validator"]["observation_verifiers"]["purpose"] = \
        ob._plain(nps.loaded_producer_source_closure(observation_verifiers=ob.ParentObservationVerifiers(
            purpose=configured_verifier)))["native_validator"]["observation_verifiers"]["purpose"]


@pytest.mark.parametrize("mutate,code", [
    (_changed_bytes, "producer_implementation_differs"),
    (_changed_slot, "configured_verifier_slot_differs"),
    (lambda closure: closure.update(_installed(max_units=3)), "scientific_adapter_source_differs"),
    (lambda closure: closure.clear(), "producer_source_closure_absent"),
])
def test_changed_or_missing_closure_is_typed_compatibility_only(mutate, code):
    expected = _installed()
    captured = _installed()
    mutate(captured)
    provenance = _provenance(captured or None, expected)
    warrant = current.native_sources_bound(provenance, expected, typed=True)
    assert not warrant.warranted and code in warrant.reasons
    assert current.native_sources_bound(provenance, expected) is False  # Untyped result unchanged.
    # The installed side changing is symmetric: the original run is never relabelled.
    if captured:
        assert not current.native_sources_bound(_provenance(expected, captured), captured, typed=True).warranted


def test_matching_closure_without_bound_final_source_is_not_warranted():
    expected = _installed()
    provenance = _provenance(expected)
    provenance["parent_final_source_identity"]["finalizer"] = None
    assert current.native_sources_bound(provenance, expected, typed=True).reasons == (
        "parent_final_source_unbound",)


def test_legacy_v1_closure_without_scientific_adapters_never_is_warranted():
    legacy = ob._plain(nps.loaded_producer_source_closure())
    warrant = current.native_sources_bound(_provenance(legacy), legacy, typed=True)
    assert warrant.reasons == ("parent_final_source_unbound",)


def test_closure_captured_under_other_hash_seeds_matches_installed_closure():
    script = """import json
from autokernel.loop import native_producer_source as nps, native_scientific_witness as sw
from autokernel.loop import native_server_t0_witness as st, observation_binding as ob
from autokernel.loop import validation_projection_source as current
from autokernel.loop import lifecycle_observation as lo
adapters = sw.ParentScientificWitnessAdapters(correctness=st.NativeServerT0WitnessAdapter(max_units=4))
print(json.dumps({"closure": ob._plain(nps.loaded_producer_source_closure(scientific_adapters=adapters)),
    "bound": lo.callable_identity(current.native_sources_bound)}, sort_keys=True))
"""
    package_root = str(Path(nps.__file__).resolve().parents[2])
    rows = [json.loads(subprocess.check_output([sys.executable, "-c", script], text=True, timeout=60,
        env={**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": package_root}))
        for seed in (1, 977)]
    expected = _installed()
    from . import lifecycle_observation as lo
    for row in rows:
        assert row["closure"] == expected
        assert row["bound"] == lo.callable_identity(current.native_sources_bound)
        assert row["bound"]["implementation_status"] == row["bound"]["configuration_status"] == "pinned"
        assert current.native_sources_bound(_provenance(row["closure"], expected), expected,
                                            typed=True).warranted
