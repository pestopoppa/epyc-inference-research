"""AKU-07o2: device -> KFD gpu_id map from a fake topology tree, carried to the worker."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from . import kfd_topology as kt
from . import unified_driver


def _tree(root, nodes):
    for index, (gpu_id, minor) in enumerate(nodes):
        node = root / str(index)
        node.mkdir(parents=True)
        (node / "gpu_id").write_text(f"{gpu_id}\n")
        (node / "properties").write_text(f"cpus 0\ndrm_render_minor {minor}\nlocation_id 1\n")
    return root


def test_topology_orders_gpu_nodes_by_render_minor_and_skips_cpu_nodes(tmp_path):
    root = _tree(tmp_path / "nodes", [(0, 0), (900, 129), (25466, 128), (0, 0)])
    assert kt.topology_gpu_ids(root) == ["25466", "900"]
    assert kt.gpu_id_map(["ROCm0", "1"], root=root) == {"1": "900", "ROCm0": "25466"}


def test_manifest_fallback_and_refusal(tmp_path):
    root = _tree(tmp_path / "nodes", [(25466, 128)])
    assert kt.gpu_id_map(["ROCm0", "gpu-x"], root=root, manifest={"gpu-x": "77"}) \
        == {"ROCm0": "25466", "gpu-x": "77"}
    absent = tmp_path / "absent"
    assert kt.gpu_id_map(["ROCm0"], root=absent, manifest={"ROCm0": "5"}) == {"ROCm0": "5"}
    assert kt.gpu_id_map(["ROCm0"], root=absent) is None  # no KFD, no manifest: no reader
    with pytest.raises(kt.KfdTopologyRefused, match="no KFD gpu_id"):
        kt.gpu_id_map(["ROCm3"], root=root, manifest={"other": "1"})
    with pytest.raises(kt.KfdTopologyRefused):
        kt.validate_map({"ROCm0": "not-digits"})
    assert kt.gpu_id_map([], root=root) is None


def test_reader_for_installs_the_kfd_consumer_reader(tmp_path, monkeypatch):
    assert kt.reader_for(SimpleNamespace(kfd_gpu_ids=None)) is None
    seen = {}
    from . import lifecycle_observation as lo
    monkeypatch.setattr(lo, "kfd_consumer_reader", lambda mapping: seen.setdefault("m", mapping))
    assert kt.reader_for(SimpleNamespace(kfd_gpu_ids={"ROCm0": "25466"})) == {"ROCm0": "25466"}


def test_driver_config_round_trips_the_map_and_legacy_rows_stay_unchanged(tmp_path):
    row = {"schema": unified_driver.CONFIG_SCHEMA, "resolved_campaign_path": str(tmp_path / "r"),
           "store_path": str(tmp_path / "s"), "scheduler_config": {}, "scheduler_state": {},
           "runtime_anchors": {}, "runtime_dimensions": {}, "profiles": {},
           "experiment_plans": {}, "profile_requests": {}, "execution_inputs": {},
           "native_artifact_sink_ref": "sink", "config_generation": 1}
    assert unified_driver.DriverConfig.from_dict(row).kfd_gpu_ids is None
    config = unified_driver.DriverConfig.from_dict(row | {"kfd_gpu_ids": {"ROCm0": "25466"}})
    assert config.kfd_gpu_ids == {"ROCm0": "25466"}
    with pytest.raises(kt.KfdTopologyRefused):
        unified_driver.DriverConfig.from_dict(row | {"kfd_gpu_ids": {}})


def test_prepared_stage_carries_the_map_and_legacy_digest_is_unchanged(tmp_path):
    from . import unified_worker as uw
    from .test_unified_worker import _prepared
    legacy = _prepared(tmp_path, backend="gpu")
    assert legacy.kfd_gpu_ids is None and "kfd_gpu_ids" not in legacy.to_dict()
    body = legacy.body() | {"kfd_gpu_ids": {"ROCm0": "25466"}}
    carried = uw.PreparedPlannedServingStage.from_dict(body | {"prepared_digest": uw._digest(body)})
    assert dict(carried.kfd_gpu_ids) == {"ROCm0": "25466"}
    assert carried.prepared_digest != legacy.prepared_digest  # the map is digest-bound
    assert uw.PreparedPlannedServingStage.from_dict(carried.to_dict()) == carried
    with pytest.raises(uw.WorkerBridgeRefused):
        uw.PreparedPlannedServingStage.from_dict(body | {"prepared_digest": legacy.prepared_digest})
