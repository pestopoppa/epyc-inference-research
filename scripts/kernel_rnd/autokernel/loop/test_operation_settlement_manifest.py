"""AKU-06l3: operation_settlement is configurable from the startup manifest."""
from __future__ import annotations

import copy

import pytest

from . import standalone_inputs as inputs
from .test_standalone_inputs import FullHeldProvider, _management_document, _verifier


def _with(document, value):
    document = copy.deepcopy(document)
    document["operation_settlement"] = value
    document["manifest_digest"] = inputs._digest({
        key: item for key, item in document.items() if key != "manifest_digest"})
    return document


def test_manifest_round_trips_operation_settlement_and_absent_stays_legacy(tmp_path):
    legacy = _management_document(tmp_path)
    assert inputs.StartupManifest.from_dict(legacy).operation_settlement is None
    assert "operation_settlement" not in inputs.StartupManifest.from_dict(legacy).to_dict()
    manifest = inputs.StartupManifest.from_dict(_with(legacy, True))
    assert manifest.operation_settlement is True
    assert inputs.StartupManifest.from_dict(manifest.to_dict()) == manifest
    with pytest.raises(inputs.StandaloneInputsRefused, match="bool"):
        inputs.StartupManifest.from_dict(_with(legacy, "yes"))


@pytest.mark.parametrize("value", [True, False])
def test_runtime_factory_installs_the_manifest_choice(tmp_path, value):
    document = _with(_management_document(tmp_path), value)
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    containers = tmp_path / "containers"
    containers.mkdir(mode=0o700)
    registry = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(lifecycle_provider=FullHeldProvider(containers)),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_verifiers={"fixture-evidence": _verifier()})

    class Args:
        store = document["driver_config"]["store_path"]
        config_generation = 1
        snapshot_version = 3

    controller, runtime = inputs.runtime_factory(materialized, registry)(materialized.resolved, Args())
    try:
        assert runtime.config.operation_settlement is value
        assert runtime.executor._operation_settlement is value
    finally:
        runtime.close()
        controller.close()
