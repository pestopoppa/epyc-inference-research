"""Actual compare/archive with synthetic observations, never inference or new grading."""
import copy
import hashlib
import json
import sqlite3
import time

import pytest

from . import archive, serving, serving_beliefs as beliefs
from ..controller import experiment_capture
from .test_resolved_recipe import BUILD, _resolve


def test_missing_root_is_a_lazy_diagnostic_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("EPYC_ROOT_REPO", raising=False)
    monkeypatch.delenv("EPYC_ROOT", raising=False)
    bridge = beliefs.PlannerFeedback(tmp_path)
    result = bridge.context({"epoch": "fixture"})
    assert result["status"] == "unavailable"
    assert result["errors"] and "EPYC_ROOT is not set" in result["errors"][0]


def comparison(monkeypatch, *, resolved=True, floor_pct=5.0):
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
    row = serving.compare(recipe, BUILD, BUILD, pairs=2, floor_pct=floor_pct,
                          floor_unit=None if floor_pct is None else serving.COMPARE_EFFECT_UNIT,
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
    assert native_bytes == json.dumps(native, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()
    assert not native_bytes.endswith(b"\n")
    assert native["comparison"] == row
    with sqlite3.connect(tmp_path / "experiments.db") as db:
        retained = json.loads(db.execute("SELECT payload FROM experiments").fetchone()[0])
    # archive.record seals the row (belief redesign Phase 2); the attempt itself is unchanged.
    assert retained.pop("belief_capture")["schema"] == experiment_capture.SCHEMA
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
        stored = json.loads(db.execute("SELECT payload FROM experiments").fetchone()[0])
    assert stored.pop("belief_capture")["schema"] == experiment_capture.SCHEMA
    assert stored == attempt


def test_compact_export_fits_when_pretty_encoding_would_exceed_budget(tmp_path, monkeypatch):
    body = {"observations": [{"allowed_cpus": list(range(64))}] * 8}
    compact = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    pretty = json.dumps(body, indent=2, sort_keys=True).encode()
    assert len(compact) < len(pretty)
    monkeypatch.setattr(beliefs, "MAX_BYTES", len(compact))

    reference = beliefs._write_exact(tmp_path, "capture.json", body)

    assert reference["size"] == len(compact)
    assert (tmp_path / "capture.json").read_bytes() == compact
    assert not list(tmp_path.glob(".patch-*"))


def test_compact_export_still_refuses_a_true_over_budget_body(tmp_path, monkeypatch):
    body = {"observations": ["retained-native-observation"]}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    monkeypatch.setattr(beliefs, "MAX_BYTES", len(encoded) - 1)

    with pytest.raises(ValueError, match="belief export byte budget exceeded"):
        beliefs._write_exact(tmp_path / "beliefs", "capture.json", body)

    assert not (tmp_path / "beliefs").exists()


def _strict_reader(monkeypatch, *, required=True):
    import importlib
    import os
    from pathlib import Path
    root = os.environ.get("EPYC_ROOT_REPO") or os.environ.get("EPYC_ROOT")
    if not root:
        if not required:
            return None
        pytest.skip("named ROOT required for producer/strict-reader integration")
    expected = Path(root).resolve() / "scripts/vidya/adapters/autokernel_legacy_serving.py"
    monkeypatch.syspath_prepend(str(expected.parents[1]))
    reader = importlib.import_module("adapters.autokernel_legacy_serving")
    assert Path(reader.__file__).resolve() == expected
    return reader


def _read_final(reader, path):
    raw = path.read_bytes()
    return reader.native_rows_v2(json.loads(raw), receipt_locator=f"autokernel:{path}",
                                 receipt_sha256=hashlib.sha256(raw).hexdigest())


def test_tripwire_final_boundary_exports_and_ingests_original_values(tmp_path, monkeypatch):
    from . import run, same_code_tripwire
    reader = _strict_reader(monkeypatch)
    row, calls = comparison(monkeypatch)
    original = copy.deepcopy(row)
    observed = same_code_tripwire.inspect(tmp_path, run.ServingComparison(row),
        anchor_build=tmp_path / "missing-a", candidate_build=tmp_path / "missing-c",
        library="libggml-cpu.so", measurement_epoch="original-epoch", launch_identity={})
    final = observed.to_dict()
    assert final["same_code_tripwire"]["status"] == "unavailable"
    assert row == original and len(calls) == 4
    assert {k: final[k] for k in original if k != "belief_capture"} == {
        k: v for k, v in original.items() if k != "belief_capture"}
    capture = final["belief_capture"]
    assert capture["inputs"] == original["belief_capture"]["inputs"]
    assert capture["capture_id"] != original["belief_capture"]["capture_id"]
    # Snapshot owns evidence and native bytes; to_dict cannot reseal or expose them.
    observed.evidence["status"] = "changed-after-construction"
    row["anchor_samples"][0] = 999.0
    exposed = observed.to_dict()
    exposed["same_code_tripwire"]["status"] = "changed-by-caller"
    with monkeypatch.context() as patch:
        patch.setattr(beliefs, "finish", lambda *a: pytest.fail("serialization resealed"))
        assert observed.to_dict() == final
    assert archive.record(tmp_path, {"status": "measured_null", "comparison": final},
        epoch="original-epoch", recorded_at="2026-10-31T01:00:00Z", campaign_id="original-campaign")
    path = tmp_path / "serving-beliefs" / f"{capture['capture_id']}.json"
    projected, = _read_final(reader, path)
    claim = reader.project_v2(projected)
    from claim_tuple import grade
    assert claim.value == 19.0 and claim.observation["anchor_value"] == 11.0
    assert grade(claim)[:2] == ("Judged", "Located")
    assert claim.measurement_id == f"legacy-serving-v2:{capture['capture_id']}"
    feedback = reader.PlannerFeedback(tmp_path)
    assert feedback.notes == []
    frames = list(feedback.ledger.read_all())
    assert len(frames) == 6
    assert feedback.ingest(path) == 0
    assert [x.frame for x in feedback.ledger.read_all()] == [x.frame for x in frames]
    assert not (path.parent / f"{original['belief_capture']['capture_id']}.json").exists()


@pytest.mark.parametrize("field", ["same_code_tripwire", "candidate_serving_profile", "longctx"])
@pytest.mark.parametrize("fault", ["value", "input", "capture", "missing"])
def test_diagnostic_boundary_never_repairs_bad_original(tmp_path, monkeypatch, capsys, field, fault):
    row, calls = comparison(monkeypatch)
    if fault == "value":
        row["candidate_tok_s"] += 1
    elif fault == "input":
        row["belief_capture"]["inputs"]["recipe"]["model"] = "/tampered"
    elif fault == "capture":
        row["belief_capture"] = {"malformed": True}
    else:
        del row["belief_capture"]
    original = copy.deepcopy(row)
    final = beliefs.annotate(row, {field: {"status": "unavailable"}})
    assert row == original and len(calls) == 4
    assert final.get("belief_capture") == original.get("belief_capture")
    assert "diagnostic annotation refused" in final["belief_capture_error"]
    assert final["candidate_tok_s"] == original["candidate_tok_s"]
    assert archive.record(tmp_path, {"status": "measured_null", "comparison": final},
                          epoch="e", recorded_at="2026-10-31T01:00:00Z", campaign_id="c")
    if fault != "missing":
        assert "serving belief export failed after durable archive" in capsys.readouterr().err
    assert not (tmp_path / "serving-beliefs").exists()


@pytest.mark.parametrize("diagnostic", ["same_code_tripwire", "candidate_serving_profile", "longctx"])
@pytest.mark.parametrize("field", ["candidate_tok_s", "recipe_hash", "diagnostic"])
def test_final_native_and_diagnostic_tamper_refused_even_with_rehashed_file_reference(
        tmp_path, monkeypatch, field, diagnostic):
    reader = _strict_reader(monkeypatch)
    row, _ = comparison(monkeypatch)
    final = beliefs.annotate(row, {diagnostic: {"status": "unavailable"}})
    path = beliefs.export(tmp_path, {"comparison": final}, campaign_id="c", epoch="e",
                          recorded_at="2026-10-31T01:00:00Z")
    assert len(_read_final(reader, path)) == 1
    receipt = json.loads(path.read_bytes())
    source_path = path.parent / receipt["native_reference"]["path"]
    source = json.loads(source_path.read_bytes())
    source["comparison"][diagnostic if field == "diagnostic" else field] = (20.0 if field == "candidate_tok_s" else
                                    "changed" if field == "recipe_hash" else {"status": "within_floor"})
    raw = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    source_path.write_bytes(raw)
    receipt["native_reference"].update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(receipt))
    from claim_tuple import ProjectionError
    with pytest.raises(ProjectionError, match="native/input binding differs"):
        _read_final(reader, path)
    with pytest.raises(ValueError, match="does not rederive"):
        beliefs.export(tmp_path / "tampered", {"comparison": source["comparison"]},
                       campaign_id="c", epoch="e", recorded_at="2026-10-31T01:00:00Z")


@pytest.mark.parametrize("annotations", [{"candidate_tok_s": 999.0},
    {"longctx": float("nan")}, {"belief_capture": {}}, {}])
def test_diagnostic_fault_cannot_overwrite_or_mint_a_capture(monkeypatch, annotations):
    row, _ = comparison(monkeypatch)
    original = copy.deepcopy(row)
    final = beliefs.annotate(row, annotations)
    assert row == original
    assert final["belief_capture"] == original["belief_capture"]
    assert {k: v for k, v in final.items() if k != "belief_capture_error"} == original
    assert "diagnostic annotation refused" in final["belief_capture_error"]
    with pytest.raises(ValueError, match="does not rederive"):
        beliefs.export(None, {"comparison": final}, campaign_id="c", epoch="e", recorded_at="date")


def test_nonwrapped_capture_keeps_original_id_and_strict_reader_contract(tmp_path, monkeypatch):
    reader = _strict_reader(monkeypatch)
    row, _ = comparison(monkeypatch)
    original = copy.deepcopy(row)
    path = beliefs.export(tmp_path, {"comparison": row}, campaign_id="c", epoch="e",
                          recorded_at="2026-10-31T01:00:00Z")
    projected, = _read_final(reader, path)
    assert reader.project_v2(projected).measurement_id == (
        "legacy-serving-v2:" + original["belief_capture"]["capture_id"])
    assert row == original
    assert beliefs.finish(row, row["belief_capture"]["inputs"]) == original["belief_capture"]


def test_actual_tripwire_retains_tampered_capture_as_nonfatal_refusal(tmp_path, monkeypatch):
    from . import run, same_code_tripwire
    row, calls = comparison(monkeypatch)
    row["candidate_samples"][0] = 500.0
    original = copy.deepcopy(row)
    observed = same_code_tripwire.inspect(tmp_path, run.ServingComparison(row),
        anchor_build=tmp_path / "missing-a", candidate_build=tmp_path / "missing-c",
        library="libggml-cpu.so", measurement_epoch="original-epoch", launch_identity={})
    final = observed.to_dict()
    assert final["belief_capture"] == original["belief_capture"]
    assert "diagnostic annotation refused" in final["belief_capture_error"]
    assert observed.effect == original["effect"] and observed.decisive == original["decisive"]
    assert len(calls) == 4
    with pytest.raises(ValueError, match="does not rederive"):
        beliefs.export(tmp_path, {"comparison": final}, campaign_id="c", epoch="e", recorded_at="date")


def test_final_annotation_seal_fault_keeps_original_warrant_and_numeric_result(monkeypatch):
    row, _ = comparison(monkeypatch)
    original = copy.deepcopy(row)
    finish = beliefs.finish
    calls = []

    def fail_final(body, inputs):
        calls.append(body)
        if len(calls) == 2:
            raise OSError("synthetic final seal fault")
        return finish(body, inputs)

    monkeypatch.setattr(beliefs, "finish", fail_final)
    final = beliefs.annotate(row, {"longctx": {"passed": False}})
    assert len(calls) == 2
    assert row == original and final["belief_capture"] == original["belief_capture"]
    assert final["candidate_tok_s"] == original["candidate_tok_s"]
    assert final["longctx"] == {"passed": False}
    assert "synthetic final seal fault" in final["belief_capture_error"]
    monkeypatch.setattr(beliefs, "finish", finish)
    with pytest.raises(ValueError, match="does not rederive"):
        beliefs.export(None, {"comparison": final}, campaign_id="c", epoch="e", recorded_at="date")
