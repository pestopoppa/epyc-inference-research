"""The persisted analysis envelopes bind the exact input bytes actually parsed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_mf_vbs_manifest_hashes_the_same_bytes_it_parsed(tmp_path):
    mod = _load("mf_vbs1_provenance_test", "scripts/analysis/mf_vbs1_verify_before_stop.py")
    source = tmp_path / "data" / "results.jsonl"
    raw = b'{"mode":"real","task":"t1","arm":"off","block":0}\n'
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    manifest = []
    parsed = mod._load_jsonl(str(source), input_manifest=manifest, repo_root=str(tmp_path))
    assert parsed == [{"mode": "real", "task": "t1", "arm": "off", "block": 0}]
    assert manifest == [{"path": "data/results.jsonl", "sha256": _sha(raw),
                         "byte_count": len(raw), "row_count": 1,
                         "normalized_row_count": 1}]
    report = {"corpus": {"n_total_real_trajectories": 1}, "sample_metric": 0.25}
    sealed = mod._seal_report(report, str(tmp_path), manifest)
    assert sealed["sample_metric"] == report["sample_metric"]
    assert sealed["native_provenance"]["schema"] == mod.REPORT_SCHEMA
    assert sealed["native_provenance"]["inputs"] == manifest
    assert sealed["native_provenance"]["producer_sha256"] == _sha(
        (ROOT / "scripts/analysis/mf_vbs1_verify_before_stop.py").read_bytes())


def test_eval_discriminability_manifest_hashes_the_same_bytes_it_parsed(tmp_path):
    mod = _load("eval_discriminability_provenance_test",
                "scripts/analysis/eval_suite_discriminability.py")
    source = tmp_path / "question_ledger.jsonl"
    raw = b'{"suite":"tiny","qid":"q1","correct":true,"error":false}\n'
    source.write_bytes(raw)
    manifest = []
    parsed, warnings = mod.load_rows([source], input_manifest=manifest)
    assert warnings == []
    assert len(parsed) == 1 and parsed[0]["qid"] == "q1"
    assert manifest == [{"path": source.resolve().as_posix(), "sha256": _sha(raw),
                         "byte_count": len(raw), "row_count": 1,
                         "normalized_row_count": 1}]
    original = {
        "schema_version": mod.REPORT_SCHEMA, "generated_at": "2026-10-06T12:00:00+00:00",
        "measurement_class": "OBSERVATION", "config": {"target_effect": 0.15},
        "inputs": [str(source)], "warnings": [], "summary": {}, "suites": [], "task_classes": [],
    }
    sealed = mod._seal_report(dict(original), manifest)
    assert {k: v for k, v in sealed.items() if k != "native_provenance"} == original
    assert sealed["native_provenance"]["inputs"] == manifest
    assert sealed["native_provenance"]["analysis_config_sha256"] == mod._canonical_sha256(
        original["config"])


def test_mf_vbs_edited_call_bearing_numerator_uses_intersection():
    mod = _load("mf_vbs1_intersection_test", "scripts/analysis/mf_vbs1_verify_before_stop.py")
    edited_without_call = mod.Trajectory(
        resdir="fixture", task="edit", arm="a", block=0, turns=1, quality_pass=True,
        touched_files=["changed.py"], answer_preview="done", trace_path="",
    )
    no_edit_with_call = mod.Trajectory(
        resdir="fixture", task="no-edit", arm="a", block=1, turns=1, quality_pass=True,
        touched_files=[], answer_preview="done", trace_path="",
        trace_turns=[{"raw_output": "run_shell('true')"}],
    )
    mod.classify(edited_without_call)
    mod.classify(no_edit_with_call)
    report = mod.summarize([edited_without_call, no_edit_with_call])
    rate = report["rates"]["no_execution_after_edit_corpuswide_forced_and_voluntary"]
    assert rate["denominator"] == 1
    assert rate["call_bearing_edited_trajectories"] == 0
    assert rate["numerator"] == 1
    assert rate["point"] == 1.0


def test_mf_vbs_zero_denominators_are_unknown_json_values():
    mod = _load("mf_vbs1_zero_denominator_test", "scripts/analysis/mf_vbs1_verify_before_stop.py")
    report = mod.summarize([])
    for rate in report["rates"].values():
        assert rate["denominator"] == 0
        assert rate["point"] is None
