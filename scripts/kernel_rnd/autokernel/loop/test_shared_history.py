"""Shared recall is advice from original stores, never cross-target admission."""
import json
from pathlib import Path
import sqlite3

import pytest

from ..controller import experiments
from . import actors, archive, run, serial_roster, serial_run
from .test_serial_roster import _inputs


def _record(root, index=0, **changes):
    attempt = {"mechanism_id": f"mechanism-{index}", "status": "measured_null",
        "statement": "reduce redundant unpacking", "falsifier": "no reduction in loads",
        "target_surface": "ggml/src/ggml-cpu/ops.cpp", "target_symbol": "unpack",
        "reason": "original window was cold; do not treat as settled refutation",
        "effect_fraction": 0.123456789, "exact_attribution_effect_fraction": 0.234567891,
        "target_runtime_effect_fraction": 0.345678912,
        "research_scope": {"model": {"path": "/original/model.gguf", "sha256": "a" * 64},
            "quant": "Q4_K", "backend": "cpu", "measurement_surface": "serving-np1",
            "recipe": {"original_threads": 96}, "request_digest": "b" * 64}}
    attempt.update(changes)
    with experiments.ExperimentStore(root) as store:
        store.record(attempt, epoch="original-epoch", recorded_at=f"2026-09-10T01:{index:02}:00Z",
                     campaign_id="original-campaign")
    return attempt


def test_shared_read_preserves_qualitative_scope_but_redacts_all_structured_magnitudes(tmp_path):
    source, current = tmp_path / "source", tmp_path / "current"
    originals = [_record(source, i, status=status) for i, status in enumerate(
        ("measured_null", "kept", "confirm_vetoed"))]
    _record(current, 9)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()}
    view = archive.SharedHistory([source, current, alias], current_store=current).recall()
    assert view["queried_roots"] == [str(source)] and not view["errors"]
    assert len(view["rows"]) == 3
    for row in view["rows"]:
        original = next(a for a in originals if a["mechanism_id"] == row["mechanism_id"])
        assert row["research_scope"] == original["research_scope"]
        assert row["original_epoch"] == "original-epoch"
        assert row["status"] == original["status"]
        assert row["refusal_reason"] == original["reason"]
        assert row["statement"] == original["statement"] and row["falsifier"] == original["falsifier"]
        assert not row["comparable_measurement"] and not row["same_epoch"]
        assert row["transfer"] == "unproven_not_local_gain_or_refutation"
        assert all(row[field] is None for field in experiments._MAGNITUDE_FIELDS)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()}
    # The optional shared projection did not change the existing owner's recall.
    with experiments.ExperimentStore(source) as store:
        local = store.recall(epoch="original-epoch")
        assert local[0]["effect_fraction"] == 0.123456789
        assert local[0]["comparable_measurement"] and "research_scope" not in local[0]
    with experiments.ExperimentStore(source, read_only=True) as store:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.record({"status": "must_not_write"}, epoch="e", recorded_at="now", campaign_id="new")


def test_missing_corrupt_and_nonregular_history_are_diagnostic_only(tmp_path):
    missing, corrupt, fifo = (tmp_path / name for name in ("missing", "corrupt", "fifo"))
    corrupt.mkdir()
    (corrupt / "experiments.db").write_bytes(b"not a database")
    fifo.mkdir()
    import os
    os.mkfifo(fifo / "experiments.db")
    view = archive.SharedHistory([missing, corrupt, fifo], current_store=tmp_path / "current").recall()
    assert len(view["errors"]) == 3 and not view["rows"]
    assert not missing.exists()
    assert (corrupt / "experiments.db").read_bytes() == b"not a database"


def test_seventeen_roots_rotate_without_silent_coverage_or_local_relabeling(tmp_path):
    roots = [tmp_path / f"source-{i}" for i in range(17)]
    for i, root in enumerate(roots):
        _record(root, i)
    history = archive.SharedHistory(roots, current_store=tmp_path / "current")
    queried = set()
    for _ in range(3):
        view = history.recall()
        assert len(view["queried_roots"]) == 8 and len(view["omitted_roots"]) == 9
        assert len(view["rows"]) <= 40 and not view["errors"]
        queried.update(view["queried_roots"])
    assert queried == {str(root) for root in roots}
    # A fresh one-iteration child for the same target every 17 serial batches
    # cannot reset coverage to roots 0..7 each time.
    queried = set()
    for batch in (0, 17, 34):
        view = archive.SharedHistory(roots, current_store=tmp_path / "current",
            batch_directory=tmp_path / f"batch-{batch:06d}").recall()
        queried.update(view["queried_roots"])
    assert queried == {str(root) for root in roots}


def test_useful_mechanisms_and_scope_relevance_survive_transient_flood(tmp_path):
    source = tmp_path / "source"
    _record(source, 0, status="kept", mechanism_id="retained-keep")
    _record(source, 1, mechanism_id="relevant-null")
    for i in range(2, 12):
        _record(source, i, research_scope={"model": "/other/model", "quant": "Q8_0"})
    for i in range(12, 30):
        _record(source, i, status="planner_transient")
    view = archive.SharedHistory([source], current_store=tmp_path / "current").recall(
        scope={"model": "/original/model.gguf", "quant": "Q4_K"})
    assert not view["errors"] and len(view["rows"]) == 5
    assert [row["mechanism_id"] for row in view["rows"][:2]] == ["retained-keep", "relevant-null"]
    assert any(row["status"] == "planner_transient" for row in view["rows"])
    assert all(not row["comparable_measurement"] for row in view["rows"])


def test_legacy_original_fields_and_unknowns_not_current_run_inferences(tmp_path):
    source = tmp_path / "old"
    _record(source, 0, research_scope=None, comparison={"surface": "tg128", "model": "old-model"})
    _record(source, 1, research_scope=None, comparison={"surface": "serving-np4", "recipe": "old-recipe",
        "recipe_hash": "c" * 64, "recipe_env": {"GGML_IQK": "1"},
        "request_digest": "d" * 64,
        "belief_capture": {"inputs": {"recipe": {"model": "/old/other.gguf"},
                                       "resolved_arms": {"anchor": {"execution_digest": "e" * 64}}}}})
    _record(source, 2, research_scope=None)
    _record(source, 3, research_scope=None, irrelevant_large_payload="x" * (2 * 1024 * 1024))
    view = archive.SharedHistory([source], current_store=tmp_path / "new").recall()
    assert not view["errors"]
    rows = {r["mechanism_id"]: r["research_scope"] for r in view["rows"]}
    assert rows["mechanism-0"]["model"] == "old-model"
    assert rows["mechanism-0"]["measurement_surface"] == "tg128"
    assert rows["mechanism-1"]["model"] == "/old/other.gguf"
    assert rows["mechanism-1"]["recipe"]["hash"] == "c" * 64
    assert rows["mechanism-1"]["recipe"]["original_serving_arms"]["anchor"]["execution_digest"] == "e" * 64
    assert rows["mechanism-2"]["model"] is None and rows["mechanism-2"]["quant"] is None
    assert "bounded" in rows["mechanism-3"]["unknown_reason"]


def test_actual_planner_prompt_keeps_shared_records_out_of_characterized_pool(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for i in range(4):
        _record(source, i, mechanism_id="shared-mechanism")
    context = {"shared_prior_experiments": archive.SharedHistory(
        [source], current_store=tmp_path / "local").recall()}
    prompts = []

    def reply(prompt, **_kwargs):
        prompts.append(prompt)
        return json.dumps({"mechanism_id": "fresh", "statement": "test here", "falsifier": "local null",
                           "target_surface": "ggml/src/ggml-cuda/vecdotq.cuh", "target_symbol": "unpack"})

    monkeypatch.setattr(actors, "_run_agent", reply)
    actors.AgentPlanner(workspace=tmp_path).propose(context)
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "Shared historical mechanisms" in prompt and "transfer NOT established" in prompt
    assert "shared-mechanism" in prompt and "original window was cold" in prompt
    assert "/original/model.gguf" in prompt and "Q4_K" in prompt
    assert "0.123456789" not in prompt and "0.234567891" not in prompt
    assert "Characterised" not in prompt and "measured 4x" not in prompt


def test_roster_automatically_forwards_canonical_sibling_and_explicit_shared_history(tmp_path):
    _resolved, _owners, argv = _inputs(tmp_path, backends=("cpu", "gpu", "cpu"))
    common = tmp_path / "common.json"
    extra = str(tmp_path / "additional-history")
    common.write_text(json.dumps(["--shared-history-root", extra]))
    targets, _skipped, _cpus = serial_roster.build_targets(
        Path(serial_run.option(argv, "--resolved-campaign")),
        Path(serial_run.option(argv, "--owned-targets")),
        target_root=tmp_path / "targets", common_path=common)
    stores = {serial_run.option(row, "--store") for row in targets}
    for row in targets:
        roots = {row[i + 1] for i, flag in enumerate(row) if flag == "--shared-history-root"}
        assert roots == {str(archive.CANONICAL_HISTORY_ROOT), extra} | (
            stores - {serial_run.option(row, "--store")})
        serial_run._validate_target_args(row)


def test_actual_run_reads_shared_history_and_records_original_scope_through_keep(tmp_path, monkeypatch):
    from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion

    source = tmp_path / "history"
    _record(source)
    before = (source / "experiments.db").read_bytes()
    real_main, real_recall, real_record = run.main, archive.SharedHistory.recall, archive.record
    reads, writes = [], []

    def recall(self, **kwargs):
        result = real_recall(self, **kwargs)
        reads.append(result)
        return result

    def record(root, attempt, **kwargs):
        if "research_scope" in attempt:
            writes.append((root, attempt))
        return real_record(root, attempt, **kwargs)

    monkeypatch.setattr(run, "main", lambda argv: real_main([*argv, "--shared-history-root", str(source)]))
    monkeypatch.setattr(archive.SharedHistory, "recall", recall)
    monkeypatch.setattr(archive, "record", record)
    test_existing_main_cpu_five_iterations_preserves_canonical_champion(False)
    assert len(reads) >= 5 and all(r["rows"][0]["mechanism_id"] == "mechanism-0" for r in reads)
    assert len(writes) == 5 and all(root != source for root, _ in writes)
    for _root, attempt in writes:
        original = attempt["comparison"]["belief_capture"]["inputs"]["resolved_arms"]
        scope = attempt["research_scope"]
        assert scope["recipe"]["original_serving_arms"] == original
        assert scope["model"]["sha256"] == original["anchor"]["model"]["sha256"]
        assert scope["request_digest"] == attempt["comparison"]["request_digest"]
        assert scope["backend"] == "cpu" and scope["quant"] == "Q4_K"
    # The keep's scope remains its measured OLD anchor, not the anchor built after keeping.
    assert writes[3][1]["research_scope"]["recipe"] != writes[4][1]["research_scope"]["recipe"]
    assert (source / "experiments.db").read_bytes() == before
