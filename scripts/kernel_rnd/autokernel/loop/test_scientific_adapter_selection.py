"""AKU-07h5: a startup manifest carries native-v2 scientific adapter configuration."""
from __future__ import annotations

from pathlib import Path

import pytest

from . import native_observation_witnesses as now
from . import native_scientific_witness as nsw
from . import native_server_t0_witness as server
from . import scientific_adapter_selection as sel
from . import standalone_inputs as inputs
from .test_standalone_inputs import FullHeldProvider
from .test_standalone_native_inputs import feed_binding, native_document

V2 = {"schema": sel.SCHEMA_V2, "correctness": {"adapter_id": server.ADAPTER_ID, "max_units": 8},
      "purpose": None, "contention": {"adapter_id": now.ADAPTER_ID_CONTENTION},
      "residency": {"adapter_id": now.ADAPTER_ID_RESIDENCY}}


def test_selection_v2_builds_the_closed_v3_registry():
    adapters = sel.installed(V2)
    assert adapters.schema == nsw.ADAPTERS_SCHEMA_V3
    assert type(adapters.correctness) is server.NativeServerT0WitnessAdapter
    assert adapters.contention.witness == "contention" and adapters.residency.witness == "residency"
    for bad in ({**V2, "purpose": {"adapter_id": "x"}},
                {**V2, "contention": {"adapter_id": now.ADAPTER_ID_RESIDENCY}},
                {**V2, "correctness": {"adapter_id": "foreign", "max_units": 8}},
                {**V2, "extra": 1}):
        with pytest.raises(sel.SelectionRefused):
            sel.installed(bad)


def test_manifest_round_trips_and_runtime_installs_v2_adapters(tmp_path):
    document = native_document(tmp_path, selection=V2)
    manifest = inputs.StartupManifest.from_dict(document)
    assert inputs.StartupManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.native_evidence["scientific_adapters"]["schema"] == sel.SCHEMA_V2
    materialized = inputs.materialize(manifest)
    feed = materialized.manifest.evidence_feed
    for path in (Path(feed.corpus_root), Path(feed.store_root), Path(feed.ledger_path).parent,
                 tmp_path / "containers"):
        path.mkdir(parents=True, exist_ok=True)
    registry = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(
            lifecycle_provider=FullHeldProvider(tmp_path / "containers")),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_feeds={feed.binding_id: feed_binding()})

    class Args:
        store = document["driver_config"]["store_path"]
        config_generation = 1
        snapshot_version = 3

    controller, runtime = inputs.runtime_factory(materialized, registry)(materialized.resolved, Args())
    try:
        installed = runtime.executor._native_evidence_configuration.scientific_adapters
        assert installed.schema == nsw.ADAPTERS_SCHEMA_V3
        assert installed.source_identity() == sel.installed(V2).source_identity()
    finally:
        runtime.close()
        controller.close()
