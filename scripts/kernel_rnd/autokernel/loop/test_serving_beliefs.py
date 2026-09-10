"""Actual compare/archive with synthetic observations, never inference or new grading."""
import copy
import hashlib
import json
import sqlite3
import time

from . import archive, serving, serving_beliefs as beliefs
from .test_resolved_recipe import BUILD, _resolve


def comparison(monkeypatch, *, resolved=True):
    recipe = serving.Recipe(name="fixture", model="/fixture-model", device="none", ngl=0, np=1)
    launch = _resolve(recipe, backend="cpu") if resolved else None
    requests = (("fixture-prompt", b'{"prompt":[1,2],"n_predict":64}'),)
    samples = iter([10.0, 20.0, 12.0, 18.0])
    calls = []

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append((build, kwargs))
        started = time.time()
        evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                         "status": "not_applicable", "samples": 1, "vram_reads": 1,
                         "peak_vram_bytes": 0, "median_vram_bytes": 0,
                         "peak_kfd_processes": 0, "resident_floor_bytes": 1 << 30,
                         "window_start": started, "window_end": time.time(),
                         "window_s": 0.001, "request_start": started, "request_end": time.time(),
                         "covers_request_phase": True, "sampled": True,
                         "cpu_placement": "unproven", "contention": "unproven",
                         "cpu_lifecycle": {"scope": "synthetic observation boundary",
                                           "status": "unavailable", "samples": [],
                                           "errors": ["fixture missing affinity"], "truncated": False}})
        return next(samples)

    monkeypatch.setattr(serving, "_measure_once", measure)
    row = serving.compare(recipe, BUILD, BUILD, pairs=2, floor_pct=5.0,
                          anchor_resolved_recipe=launch, candidate_resolved_recipe=launch,
                          frozen_requests=requests,
                          floor_request_digest=serving.request_digest(recipe, requests))
    return row, calls


def archived(tmp_path, monkeypatch, *, resolved=True):
    row, calls = comparison(monkeypatch, resolved=resolved)
    attempt = {"status": "measured_null", "mechanism_id": "fixture-source",
               "comparison": row}
    assert archive.record(tmp_path, attempt, epoch="original-epoch",
                          recorded_at="2026-09-10T01:00:00Z", campaign_id="original-campaign")
    path = tmp_path / "serving-beliefs" / f"{row['belief_capture']['capture_id']}.json"
    return row, calls, attempt, path


def test_original_compare_archive_and_atomic_export_bytes(tmp_path, monkeypatch):
    row, calls, attempt, path = archived(tmp_path, monkeypatch)
    assert len(calls) == 4
    assert [item["reps"] for item in row["belief_capture"]["belief_measurements"]] == [2, 2]
    assert [item["value"] for item in row["belief_capture"]["belief_measurements"]] == [11.0, 19.0]
    assert all(item["protocol_id"] == "" for item in row["belief_capture"]["belief_measurements"])
    receipt = json.loads(path.read_bytes())
    native_bytes = (path.parent / receipt["native_reference"]["path"]).read_bytes()
    native = json.loads(native_bytes)
    assert len(native_bytes) == receipt["native_reference"]["size"]
    assert hashlib.sha256(native_bytes).hexdigest() == receipt["native_reference"]["sha256"]
    assert native_bytes == json.dumps(native, indent=2, sort_keys=True).encode()
    assert not native_bytes.endswith(b"\n")  # matches existing status.write_json exactly
    assert native["comparison"] == row
    with sqlite3.connect(tmp_path / "experiments.db") as db:
        retained = json.loads(db.execute("SELECT payload FROM experiments").fetchone()[0])
    assert retained == attempt
    assert not archive.record(tmp_path, attempt, epoch="original-epoch",
                              recorded_at="2026-09-10T01:00:00Z", campaign_id="original-campaign")
    assert len(calls) == 4


def test_legacy_paths_are_preserved_without_implying_binary_identity(monkeypatch):
    row, _ = comparison(monkeypatch, resolved=False)
    inputs = row["belief_capture"]["inputs"]
    assert inputs["build_paths"] == {"anchor": str(BUILD), "candidate": str(BUILD)}
    assert inputs["resolved_arms"] == {"anchor": None, "candidate": None}
    assert inputs["loaded_instrument_attestation"] == "not_recorded"


def test_pre_hook_archive_has_no_export(tmp_path):
    attempt = {"status": "measured_null", "comparison": {"schema": "epyc.autokernel.serving_ab.v1"}}
    assert archive.record(tmp_path, attempt, epoch="e", recorded_at="2026-09-10T01:00:00Z", campaign_id="c")
    assert not (tmp_path / "serving-beliefs").exists()


def test_capture_fault_preserves_result_and_is_retained(monkeypatch):
    monkeypatch.setattr(beliefs, "prepare", lambda *a, **kw: (_ for _ in ()).throw(OSError("capture fault")))
    row, calls = comparison(monkeypatch)
    assert len(calls) == 4 and row["anchor_tok_s"] == 11.0 and row["candidate_tok_s"] == 19.0
    assert "belief_capture" not in row
    assert row["belief_capture_error"] == "OSError: capture fault"


def test_export_fault_preserves_durable_result_and_is_visible(tmp_path, monkeypatch, capsys):
    row, calls = comparison(monkeypatch)
    original = copy.deepcopy(row)
    monkeypatch.setattr(beliefs, "_write_exact", lambda *a, **kw: (_ for _ in ()).throw(OSError("export fault")))
    attempt = {"status": "measured_null", "comparison": row}
    assert archive.record(tmp_path, attempt, epoch="e", recorded_at="2026-09-10T01:00:00Z", campaign_id="c")
    assert "serving belief export failed after durable archive: OSError: export fault" in capsys.readouterr().err
    assert row == original and len(calls) == 4
    with sqlite3.connect(tmp_path / "experiments.db") as db:
        assert json.loads(db.execute("SELECT payload FROM experiments").fetchone()[0]) == attempt
