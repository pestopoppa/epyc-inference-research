"""Durable CPU/GPU accumulator-surface isolation and shared-COR invariants."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest import mock

import pytest

from . import accumulate as A
from . import new_epoch


def _ancestor(_older: str, _newer: str) -> bool:
    return True


def test_legacy_bundle_paths_and_schema_remain_unchanged(tmp_path):
    bundle = A.Bundle(champion_of_record="c0", tip="c1", keeps=["k1"],
                      compounded_bench_pct=1.25)
    path = bundle.save(tmp_path)
    assert path.name == "accumulator-bundle.json"
    assert (tmp_path / "journal").is_dir()
    saved = json.loads(path.read_text())
    assert saved["schema"] == A.BUNDLE_SCHEMA_V2
    assert "surface" not in saved
    restored, _ = A.load_bundle(tmp_path, anchor_commit="c1", is_ancestor=_ancestor)
    assert restored.surface is None
    assert restored.keeps == ["k1"]


def test_explicit_surface_read_only_miss_does_not_create_paths(tmp_path):
    store = tmp_path / "store"
    with pytest.raises(A.BundleRecoveryRequired, match="no accumulator state"):
        A.load_bundle(store, anchor_commit="c0", is_ancestor=_ancestor,
                      read_only=True, surface="cpu")
    assert not store.exists()


def test_explicit_surfaces_replay_independent_journals(tmp_path):
    cpu, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                           surface="cpu")
    gpu, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                           surface="gpu")
    cpu.add_keep("cpu-keep", "c1", 2.0)
    gpu.add_keep("gpu-keep", "c2", 3.0)
    cpu.save(tmp_path)
    gpu.save(tmp_path)

    (tmp_path / "accumulator-bundle-cpu.json").unlink()
    cpu_replayed, _ = A.load_bundle(tmp_path, anchor_commit="c1", is_ancestor=_ancestor,
                                    surface="cpu")
    gpu_replayed, _ = A.load_bundle(tmp_path, anchor_commit="c2", is_ancestor=_ancestor,
                                    surface="gpu")
    assert cpu_replayed.keeps == ["cpu-keep"]
    assert gpu_replayed.keeps == ["gpu-keep"]
    assert (tmp_path / "accumulator-bundle-cpu.json").exists()


def test_explicit_surface_refuses_ambiguous_legacy_state(tmp_path):
    A.Bundle(champion_of_record="c0", tip="c0").save(tmp_path)
    with pytest.raises(A.BundleRecoveryRequired, match="cannot adopt ambiguous legacy"):
        A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                      surface="cpu")


def test_explicit_save_cannot_claim_legacy_store_without_authority(tmp_path):
    A.Bundle(champion_of_record="c0", tip="c0").save(tmp_path)
    with pytest.raises(A.BundleRecoveryRequired, match="ambiguous|without shared COR"):
        A.Bundle(champion_of_record="c0", tip="c0", surface="cpu").save(tmp_path)
    assert not (tmp_path / A.SHARED_COR_FILENAME).exists()


def test_untyped_legacy_bundle_cannot_be_relabelled_with_explicit_save(tmp_path):
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="cpu")
    before = (tmp_path / "accumulator-bundle-cpu.json").read_bytes()
    legacy = A.Bundle(champion_of_record="c0", tip="c0", keeps=["old-keep"],
                      compounded_bench_pct=8.0)
    with pytest.raises(A.BundleRecoveryRequired, match="already typed"):
        legacy.save(tmp_path, surface="cpu")
    assert (tmp_path / "accumulator-bundle-cpu.json").read_bytes() == before


def test_read_only_shared_cor_check_refuses_before_seed_measurement(tmp_path):
    A._write_shared_cor(tmp_path, "c0")
    with pytest.raises(A.BundleRecoveryRequired, match="shared COR mismatch"):
        A.require_shared_cor(tmp_path, "c1", "cpu")
    assert not (tmp_path / "journal-cpu").exists()


def test_shared_cor_mismatch_refuses_restore_and_save_without_mutation(tmp_path):
    cpu, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                           surface="cpu")
    authority = tmp_path / A.SHARED_COR_FILENAME
    before_journal = (tmp_path / "journal-cpu" / "events.jsonl").read_bytes()
    authority.write_text(json.dumps({"schema": "epyc.autokernel.shared_cor.v1",
                                     "champion_of_record": "foreign"}))
    with pytest.raises(A.BundleRecoveryRequired, match="shared COR mismatch"):
        A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                      surface="cpu")
    cpu.tip = "c1"
    with pytest.raises(A.BundleRecoveryRequired, match="shared COR mismatch"):
        cpu.save(tmp_path)
    assert (tmp_path / "journal-cpu" / "events.jsonl").read_bytes() == before_journal


def test_surface_promotion_cas_advances_only_after_own_snapshot_and_stales_peer(tmp_path):
    _cpu, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                           surface="cpu")
    _gpu, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                           surface="gpu")
    promoted = A.Bundle(champion_of_record="c1", tip="c1", surface="cpu")
    promoted.promote_shared_cor(tmp_path, expected_cor="c0")
    assert A._read_shared_cor(tmp_path) == "c1"
    cpu_now, _ = A.load_bundle(tmp_path, anchor_commit="c1", is_ancestor=_ancestor,
                               surface="cpu")
    assert cpu_now.champion_of_record == "c1"
    with pytest.raises(A.BundleRecoveryRequired, match="shared COR mismatch"):
        A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                      read_only=True, surface="gpu")
    stale_bytes = (tmp_path / "journal-gpu" / "events.jsonl").read_bytes()
    with pytest.raises(A.BundleRecoveryRequired, match="compare-and-swap failed"):
        A.Bundle(champion_of_record="c2", tip="c2", surface="gpu").promote_shared_cor(
            tmp_path, expected_cor="c0")
    assert (tmp_path / "journal-gpu" / "events.jsonl").read_bytes() == stale_bytes


def test_concurrent_surface_writes_do_not_cross_overwrite(tmp_path):
    bundles = {
        surface: A.Bundle(champion_of_record="c0", tip=f"{surface}-tip",
                          keeps=[f"{surface}-keep"], surface=surface)
        for surface in ("cpu", "gpu")
    }
    A._write_shared_cor(tmp_path, "c0")
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda item: item[1].save(tmp_path), bundles.items()))
    assert {path.name for path in paths} == {
        "accumulator-bundle-cpu.json", "accumulator-bundle-gpu.json"}
    for surface in ("cpu", "gpu"):
        restored, _ = A.load_bundle(tmp_path, anchor_commit=f"{surface}-tip",
                                    is_ancestor=_ancestor, surface=surface)
        assert restored.keeps == [f"{surface}-keep"]


def test_explicit_anchor_epoch_archives_only_its_surface(tmp_path):
    old = A.Bundle(champion_of_record="c0", tip="c1", keeps=["old"],
                   compounded_bench_pct=4.0, surface="cpu")
    old.save(tmp_path)
    gpu, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                           surface="gpu")
    gpu.add_keep("gpu-keep", "c1", 1.0)
    gpu.save(tmp_path)

    fresh, note = new_epoch.start_new_anchor_epoch(
        tmp_path, anchor_commit="c2",
        is_ancestor=lambda older, newer: older == "c0" or older == newer,
        stamp="epoch-cpu", surface="cpu")
    assert "accumulator-bundle-cpu.json" in note
    assert fresh.surface == "cpu"
    assert fresh.champion_of_record == "c0"
    assert fresh.measurement_validity == A.MEASUREMENT_STALE_TIP_ADVANCE
    cpu_recovered, _ = A.load_bundle(tmp_path, anchor_commit="c2",
                                     is_ancestor=_ancestor, surface="cpu")
    gpu_recovered, _ = A.load_bundle(tmp_path, anchor_commit="c1",
                                     is_ancestor=_ancestor, surface="gpu")
    assert cpu_recovered.keeps == []
    assert gpu_recovered.keeps == ["gpu-keep"]
    archive = tmp_path / "archived-epochs" / "epoch-cpu"
    assert (archive / "accumulator-bundle-cpu.json").exists()
    assert (archive / "journal-cpu").exists()
    assert not (archive / "accumulator-bundle-gpu.json").exists()


def test_epoch_and_other_surface_save_use_shared_then_surface_lock_order(tmp_path):
    cpu = A.Bundle(champion_of_record="c0", tip="c1", keeps=["cpu-old"],
                   compounded_bench_pct=1.0, surface="cpu")
    gpu = A.Bundle(champion_of_record="c0", tip="c0", surface="gpu")
    cpu.save(tmp_path)
    gpu.save(tmp_path)
    gpu_writer = A.Bundle(champion_of_record="c0", tip="c0", keeps=["gpu-new"],
                          compounded_bench_pct=2.0, surface="gpu")

    real_lock = A._shared_cor_lock
    local = threading.local()
    epoch_has_shared_lock = threading.Event()
    writer_waiting_for_shared_lock = threading.Event()

    @contextmanager
    def ordered_lock(store):
        if getattr(local, "epoch", False):
            with real_lock(store):
                epoch_has_shared_lock.set()
                assert writer_waiting_for_shared_lock.wait(timeout=2)
                yield
        else:
            writer_waiting_for_shared_lock.set()
            with real_lock(store):
                yield

    def archive_epoch():
        local.epoch = True
        return new_epoch.start_new_anchor_epoch(
            tmp_path, anchor_commit="c2",
            is_ancestor=lambda older, newer: older == "c0" or older == newer,
            stamp="epoch-interleaved", surface="cpu")

    with (mock.patch.object(A, "_shared_cor_lock", ordered_lock),
          ThreadPoolExecutor(max_workers=2) as pool):
        epoch = pool.submit(archive_epoch)
        assert epoch_has_shared_lock.wait(timeout=2)
        writer = pool.submit(gpu_writer.save, tmp_path)
        _, note = epoch.result(timeout=4)
        assert "epoch-interleaved" in note
        writer.result(timeout=4)

    cpu_after, _ = A.load_bundle(tmp_path, anchor_commit="c2", is_ancestor=_ancestor,
                                 surface="cpu")
    gpu_after, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                                 surface="gpu")
    assert cpu_after.keeps == []
    assert gpu_after.keeps == ["gpu-new"]


def test_explicit_initialization_refuses_symlink_and_unowned_history(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked-store"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(A.BundleRecoveryRequired, match="symlink"):
        A.load_bundle(symlink, anchor_commit="c0", is_ancestor=_ancestor,
                      surface="cpu")
    assert not list(target.iterdir())

    populated = tmp_path / "populated"
    populated.mkdir()
    from ..controller import experiments
    with experiments.ExperimentStore(populated) as memory:
        memory.record({"mechanism_id": "historic", "status": "measured_null",
                       "hypothesis": "fixture history"},
                      epoch="epoch", recorded_at="2026-11-01T00:00:00Z",
                      campaign_id="fixture")
    with pytest.raises(A.BundleRecoveryRequired, match="experiment history"):
        A.load_bundle(populated, anchor_commit="c0", is_ancestor=_ancestor,
                      surface="cpu")
    assert not (populated / A.SHARED_COR_FILENAME).exists()
    assert not (populated / "journal-cpu").exists()
