"""Durable CPU/GPU accumulator-surface isolation and shared-COR invariants."""
from __future__ import annotations

import json
from pathlib import Path
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


def test_control_only_loaded_surface_is_fenced_when_shared_cor_advances(tmp_path):
    # CONTROL_ONLY: temporary journal, no production promotion or measurements.
    gpu, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                           surface="gpu")
    gpu.add_keep("gpu-keep", "c1", 42.0)
    gpu.save(tmp_path)
    A.Bundle(champion_of_record="c1", tip="c1", surface="cpu").promote_shared_cor(
        tmp_path, expected_cor="c0")
    assert A.effective_measurement_validity(gpu, tmp_path) == A.MEASUREMENT_STALE_BASELINE
    assert A.gate_trigger(gpu, 1.0, A.AccumulatorPolicy(), store=tmp_path) is None
    with pytest.raises(A.BundleRecoveryRequired, match="stale_baseline"):
        A.resolve(gpu, {"decisive": True, "effect": 0.2},
                  A.AccumulatorPolicy(), store=tmp_path)
    recovered, _ = A.load_stale_rebaseline(tmp_path, anchor_commit="c1",
                                            is_ancestor=_ancestor, surface="gpu")
    assert recovered.measurement_validity == A.MEASUREMENT_STALE_BASELINE
    assert recovered.champion_of_record == "c1"
    assert recovered.keeps == ["gpu-keep"]
    with pytest.raises(A.BundleRecoveryRequired, match="not an ancestor"):
        A.load_stale_rebaseline(tmp_path, anchor_commit="c1", surface="gpu",
                                is_ancestor=lambda older, newer: older == "c0")
    with pytest.raises(A.BundleRecoveryRequired, match="shared COR mismatch"):
        A.load_bundle(tmp_path, anchor_commit="c1", is_ancestor=_ancestor, surface="gpu")


def test_control_only_late_surface_starts_stale_even_at_cor(tmp_path):
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="cpu")
    gpu, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                           surface="gpu")
    assert A.effective_measurement_validity(gpu, tmp_path) == A.MEASUREMENT_STALE_BASELINE
    pending, _ = A.load_stale_rebaseline(tmp_path, anchor_commit="c0",
                                          is_ancestor=_ancestor, surface="gpu")
    assert pending.tip == pending.champion_of_record == "c0"
    assert pending.measurement_validity == A.MEASUREMENT_STALE_BASELINE


def _control_comparison(run, effect=0.25):
    return run.bench.Comparison(surface="tg128", anchor_samples=[100.0],
                                candidate_samples=[100.0 * (1 + effect)], effect=effect,
                                estimator="median_over_median", pairs=1,
                                noise_floor_pct=None, residency={})


def test_control_only_native_rebaseline_refuses_cor_change_during_measurement(tmp_path):
    from . import run
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="gpu")
    A.Bundle(champion_of_record="c1", tip="c1", surface="cpu").promote_shared_cor(
        tmp_path, expected_cor="c0")
    pending, _ = A.load_stale_rebaseline(tmp_path, anchor_commit="c1",
                                          is_ancestor=_ancestor, surface="gpu")
    journal = tmp_path / "journal-gpu" / "events.jsonl"
    before = journal.read_bytes()

    def measure():
        intents = list((tmp_path / "accumulator-comparisons").glob("intent-*.json"))
        assert len(intents) == 1
        assert json.loads(intents[0].read_text())["surface"] == "gpu"
        with A._shared_cor_lock(tmp_path):
            A._cas_shared_cor_locked(tmp_path, "c1", "c2")
        return _control_comparison(run)

    with mock.patch.object(run, "_git", return_value="c1"):
        with pytest.raises(run.champion.StartupRefused, match="changed during rebaseline"):
            run._native_stale_rebaseline(tmp_path, tmp_path, pending, "CONTROL_ONLY", measure)
    assert journal.read_bytes() == before
    assert pending.measurement_validity == A.MEASUREMENT_STALE_BASELINE
    receipts = list((tmp_path / "accumulator-comparisons").glob("rebaseline-*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["warrant"] == "aborted_stale"
    assert receipt["surface"] == "gpu" and receipt["harness"] == "CONTROL_ONLY"


def test_control_only_native_rebaseline_saves_measured_surface(tmp_path):
    from . import run
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="cpu")
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="gpu")
    pending, _ = A.load_stale_rebaseline(tmp_path, anchor_commit="c0",
                                          is_ancestor=_ancestor, surface="gpu")
    with mock.patch.object(run, "_git", return_value="c0"):
        run._native_stale_rebaseline(tmp_path, tmp_path, pending, "CONTROL_ONLY",
                                     lambda: _control_comparison(run, 0.125))
    loaded, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                               surface="gpu")
    assert loaded.measurement_validity == A.MEASUREMENT_CURRENT
    assert loaded.compounded_bench_pct == 12.5
    assert json.loads(Path(loaded.comparison_evidence["path"]).read_text())["surface"] == "gpu"


def test_control_only_rebaseline_refuses_incomplete_native_result(tmp_path):
    from . import run
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="cpu")
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="gpu")
    pending, _ = A.load_stale_rebaseline(tmp_path, anchor_commit="c0",
                                          is_ancestor=_ancestor, surface="gpu")
    journal = tmp_path / "journal-gpu" / "events.jsonl"
    before = journal.read_bytes()
    with mock.patch.object(run, "_git", return_value="c0"):
        with pytest.raises(ValueError, match="completed result type"):
            run._native_stale_rebaseline(tmp_path, tmp_path, pending, "CONTROL_ONLY",
                                         lambda: {"effect": 1.0})
    assert journal.read_bytes() == before
    assert pending.measurement_validity == A.MEASUREMENT_STALE_BASELINE
    assert list((tmp_path / "accumulator-comparisons").glob("failure-*.json"))
    assert list((tmp_path / "accumulator-comparisons").glob("intent-*.json"))


def test_control_only_cpu_native_serving_result_and_incomplete_refusal(tmp_path):
    from . import run, serving
    from .test_serving import RECIPE
    # CONTROL_ONLY: invoke the real serving reducer with mocked launch samples.
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="gpu")
    A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor, surface="cpu")
    pending, _ = A.load_stale_rebaseline(tmp_path, anchor_commit="c0",
                                          is_ancestor=_ancestor, surface="cpu")
    with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 120.0]):
        comparison = run._serving_comparison(
            lambda: serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=None),
            "CONTROL_ONLY")
    assert isinstance(comparison, run.ServingComparison)
    incomplete = run.ServingComparison({**comparison.row, "candidate_samples": []}, "CONTROL_ONLY")
    with mock.patch.object(run, "_git", return_value="c0"):
        with pytest.raises(ValueError, match="both arm sample vectors"):
            run._native_stale_rebaseline(tmp_path, tmp_path, pending, "CONTROL_ONLY_CPU",
                                         lambda: incomplete)
        assert pending.measurement_validity == A.MEASUREMENT_STALE_BASELINE
        failure = RuntimeError("CONTROL_ONLY native failure")
        failure.record = {"status": "measurement_invalid", "anchor_raw_samples": [100.0]}
        with pytest.raises(RuntimeError, match="CONTROL_ONLY native failure"):
            run._native_stale_rebaseline(tmp_path, tmp_path, pending, "CONTROL_ONLY_CPU",
                                         mock.Mock(side_effect=failure))
        failure_receipts = [json.loads(path.read_text()) for path in
                            (tmp_path / "accumulator-comparisons").glob("failure-*.json")]
        assert any(receipt["native_record"] == failure.record for receipt in failure_receipts)
        assert pending.measurement_validity == A.MEASUREMENT_STALE_BASELINE
        run._native_stale_rebaseline(tmp_path, tmp_path, pending, "CONTROL_ONLY_CPU",
                                     lambda: comparison)
    restored, _ = A.load_bundle(tmp_path, anchor_commit="c0", is_ancestor=_ancestor,
                                 surface="cpu")
    assert restored.measurement_validity == A.MEASUREMENT_CURRENT
    record = json.loads(Path(restored.comparison_evidence["path"]).read_text())
    assert record["surface"] == "cpu"
    assert record["comparison"]["schema"] == "epyc.autokernel.serving_ab.v1"
    assert len(record["comparison"]["anchor_samples"]) == 1
    assert len(record["comparison"]["candidate_samples"]) == 1


def test_control_only_rebaseline_and_anchor_epoch_are_exclusive(tmp_path, capsys):
    from . import run
    with pytest.raises(SystemExit) as error:
        run._main(["--worktree", str(tmp_path), "--anchor-build", str(tmp_path),
                   "--store", str(tmp_path), "--bundle-surface", "cpu",
                   "--rebaseline-stale", "--new-anchor-epoch", "c0"])
    assert error.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_control_only_rebaseline_flag_requires_surface(tmp_path, capsys):
    from . import run
    with pytest.raises(SystemExit) as error:
        run._main(["--worktree", str(tmp_path), "--anchor-build", str(tmp_path),
                   "--store", str(tmp_path), "--rebaseline-stale"])
    assert error.value.code == 2
    assert "--rebaseline-stale requires --bundle-surface" in capsys.readouterr().err


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
