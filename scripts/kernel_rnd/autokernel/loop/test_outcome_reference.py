"""Last-outcome pointers through original owners/readers, not new evidence."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import run, serial_run as sr, status


def _root():
    root = Path(os.environ.get("EPYC_ROOT_REPO", "/workspace"))
    if not (root / "scripts/vidya/adapters/autokernel_legacy_serving.py").is_file():
        pytest.skip("EPYC_ROOT_REPO must select the existing serving reader")
    return root


def test_actual_source_owner_last_null_reopens_without_full_result_or_restart_dependency(monkeypatch):
    from .test_existing_cpu_run import (
        test_existing_main_cpu_five_iterations_preserves_canonical_champion as execute,
    )
    original_write, original_read = status.write_json, sr._read
    seen = []

    def write(root, name, value, **kwargs):
        result = original_write(root, name, value, **kwargs)
        if name != "loop-continuation.json":
            return result
        reference = value["last_outcome_reference"]
        assert reference["iteration_index"] == 4 and reference["status"] == "measured_null"
        assert reference["runtime_result"] is None
        assert len(json.dumps(reference).encode()) <= 12 * 1024
        # Existing load_completed may stat the full result, never parse it.
        def read(path, **kw):
            assert Path(path).name != "loop-run.json"
            return original_read(path, **kw)
        with monkeypatch.context() as bounded:
            bounded.setattr(sr, "_read", read)
            with monkeypatch.context() as lazy:
                lazy.setattr(sr, "read_last_outcome_reference", lambda *a, **kw:
                             pytest.fail("continuation loading eagerly reopened diagnostic evidence"))
                row, _ = sr.load_completed(Path(root) / name)
            observation = sr.read_last_outcome_reference(row)
            assert observation["status"] == "available", observation
            assert not observation["scientific_eligibility"]
            receipt = Path(reference["serving_receipt"]["path"])
            before = receipt.read_bytes()
            assert sr.read_last_outcome_reference(row) == observation
            assert receipt.read_bytes() == before
            # Missing optional evidence cannot erase the original completed run.
            moved = receipt.with_suffix(".retained")
            receipt.rename(moved)
            try:
                reopened, _ = sr.load_completed(Path(root) / name)
                unavailable = sr.read_last_outcome_reference(reopened)
                assert unavailable["status"] == "unavailable" and unavailable["errors"]
                assert reopened["last_outcome_reference"] == reference
            finally:
                moved.rename(receipt)
            for change in ({"iteration_index": 0}, {"status": "not_an_original_outcome"},
                           {"mechanism_id": "another-mechanism"}):
                changed = copy.deepcopy(row)
                changed["last_outcome_reference"].update(change)
                assert sr.read_last_outcome_reference(changed)["status"] == "unavailable"
            old = {key: item for key, item in row.items() if key != "last_outcome_reference"}
            original_write(root, name, old, **kwargs)
            assert "last_outcome_reference" not in sr.load_completed(Path(root) / name)[0]
            original_write(root, name, row, **kwargs)
        seen.append(observation)
        return result

    monkeypatch.setattr(status, "write_json", write)
    execute(False, feedback_root=_root())
    assert len(seen) == 1


def test_original_nonadmitted_runtime_result_transports_after_owner_closed(tmp_path, monkeypatch):
    from . import measurement_capture as mc, runtime_admission as admission
    from .test_runtime_admission import (
        test_measured_negative_panel_blocks_retention_despite_internal_bootstrap as execute,
    )
    original_store_init, original_compare = mc.ArtifactStore.__init__, admission.RuntimeAdmission.compare
    observed = []

    def store_init(self, root):
        # Configure the existing fixture owner at the original run's actual slot
        # before capture; do not relocate/reseal an already captured artifact.
        root = Path(root)
        original_store_init(self, root.with_name("runtime-preparation") if root.name == "captures" else root)

    def compare(owner, pair):
        row = original_compare(owner, pair)
        observed.append((owner, pair, row))
        return row

    monkeypatch.setenv("EPYC_ROOT_REPO", str(_root()))
    monkeypatch.setattr(mc.ArtifactStore, "__init__", store_init)
    monkeypatch.setattr(admission.RuntimeAdmission, "compare", compare)
    execute(tmp_path, monkeypatch)
    owner, pair, native = observed.pop()
    assert not observed and native["qualified"] is False and native["decisive"] is None
    assert owner.store._runtime.fd == -1
    monkeypatch.setattr(admission.RuntimeAdmission, "__init__", lambda *a, **k:
                        pytest.fail("transport reconstructed a live admission owner"))
    comparison = run.ServingComparison(native)
    outcome = SimpleNamespace(status="runtime_observed", comparison=comparison,
        hypothesis=SimpleNamespace(mechanism_id=pair.dimension.dimension_id))
    reference = sr.last_outcome_reference([outcome], store=tmp_path)
    assert reference["runtime_result"] == native["runtime_admission"]
    launch, prompts, out = tmp_path / "launch.json", tmp_path / "prompts.json", tmp_path / "out"
    status.write_json(tmp_path, launch.name, pair.anchor.to_dict())
    status.write_json(tmp_path, prompts.name, owner.prompts.to_dict())
    argv = ["--worktree", str(owner.worktree), "--model", pair.anchor.model.path,
            "--anchor-build", pair.anchor.build_dir, "--store", str(tmp_path),
            "--experimental-branch", "ak/experimental/outcome-fixture",
            "--cpu-serving-launch", str(launch), "--frozen-prompts", str(prompts),
            "--iterations", "1", "--out", str(out)]
    row = sr.continuation(argv=argv, binding=sr.input_binding(argv), terminal="complete",
        worktree=owner.worktree, branch="ak/experimental/outcome-fixture", model=pair.anchor.model.path,
        selected_target=None, anchor_build=pair.anchor.build_dir, anchor_commit=owner.source_commit,
        iterations_requested=1, outcomes=[outcome], last_outcome_reference=reference)
    status.write_json(out, "loop-run.json", {"fixture": "not parsed as evidence"})
    status.write_json(out, "loop-continuation.json", row)
    reopened, _ = sr.load_completed(out / "loop-continuation.json")
    result = sr.read_last_outcome_reference(reopened)
    assert result["status"] == "available", result
    assert result["runtime"]["recorded_admitted"] is False
    assert not result["scientific_eligibility"]
    assert result["runtime"]["reference"] == native["runtime_admission"]
    original = owner.store.root / native["runtime_admission"]["locator"]
    moved = original.with_suffix(".retained")
    original.rename(moved)
    try:
        resumed, _ = sr.load_completed(out / "loop-continuation.json")
        partial = sr.read_last_outcome_reference(resumed)
        assert partial["status"] == "partial" and partial["runtime"] is None
        assert partial["serving"] is not None
    finally:
        moved.rename(original)


def test_last_reference_bound_does_not_grow_with_batch_length():
    outcomes = [SimpleNamespace(status="bench_failed", comparison=None)] * 1000
    assert sr.last_outcome_reference(outcomes, store="/no-store") is None
