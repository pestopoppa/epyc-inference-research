import importlib.util
import json
from pathlib import Path
import sys


MODULE = Path(__file__).with_name("aggregate_np_context_v8.py")
SPEC = importlib.util.spec_from_file_location("aggregate_np_context_v8", MODULE)
aggregate = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = aggregate
SPEC.loader.exec_module(aggregate)


def make_surface(tmp_path: Path, grid="a4_bridge", label="bridge") -> Path:
    surface = tmp_path / label
    surface.mkdir()
    (surface / "provenance.txt").write_text(f"mode=throughput_only grid={grid} mtp_depth=4 thinking=false\n")
    return surface


def result(label: str, np: int, length: int, *, errors=0) -> dict:
    return {"meta": {"kernel": "production-consolidated-v8", "arm": f"{label}_np{np}_L{length}",
            "max_tokens": length, "questions_pinned": aggregate.CANONICAL_THROUGHPUT_PROMPTS,
            "enable_thinking": False}, "suites": [{"suite": "olympiadbench_hard", "n": np,
            "n_questions": np, "errors": errors, "throughput": {
        "concurrency": np, "wall_s": 2.0, "completion_tokens": 20,
        "prompt_tokens": 10, "aggregate_decode_tok_s": 10.0, "aggregate_total_tok_s": 15.0,
    }}]}


def fill(surface: Path) -> None:
    for np, length in aggregate.required_cells("a4_bridge"):
        cell = surface / f"np{np}_L{length}"
        cell.mkdir()
        (cell / "results.json").write_text(json.dumps(result(surface.name, np, length)))


def test_incomplete_surface_does_not_publish(tmp_path):
    surface = make_surface(tmp_path)
    report = aggregate.make_report(tmp_path, [surface.name])
    assert report["final_publishable"] is False
    assert report["surfaces"][0]["missing_cells"] == 11


def test_terminal_surface_aggregates_measured_cells(tmp_path):
    surface = make_surface(tmp_path)
    fill(surface)
    (surface / "complete.txt").write_text("COMPLETE 2026-07-27T00:00:00Z\n")
    report = aggregate.make_report(tmp_path, [surface.name])
    row = report["surfaces"][0]
    assert report["final_publishable"] is True
    assert row["measured_cells"] == 11
    assert row["invalid_cells"] == 0


def test_skip_is_terminal_but_errors_and_duplicate_dispositions_are_invalid(tmp_path):
    surface = make_surface(tmp_path)
    fill(surface)
    first = surface / "np1_L2048"
    (first / "results.json").unlink()
    (first / "skip.txt").write_text("SKIP n_ctx_slot=1024 vram=62G requested_L=2048\n")
    bad = surface / "np2_L2048"
    bad.joinpath("results.json").write_text(json.dumps(result(surface.name, 2, 2048, errors=1)))
    (surface / "complete.txt").write_text("COMPLETE 2026-07-27T00:00:00Z\n")
    row = aggregate.make_report(tmp_path, [surface.name])["surfaces"][0]
    assert row["state"] == "incomplete"
    assert row["skipped_cells"] == 1
    assert row["invalid_cells"] == 1


def test_write_refuses_incomplete_surface(tmp_path, monkeypatch):
    surface = make_surface(tmp_path)
    destination = tmp_path / "final.json"
    monkeypatch.setattr("sys.argv", ["aggregate", "--root", str(tmp_path), "--label", surface.name, "--write", str(destination)])
    assert aggregate.main() == 2
    assert not destination.exists()


def test_default_labels_exclude_quarantined_surface_names():
    assert aggregate.CANONICAL_LABELS == (
        "A3_tc_thinkingcap_q8", "A3_ff_fable_non_mtp_q8", "A3_ff_fable_mtp_q8",
        "Laguna_ud_iq2_gpu_dflash_off", "A4_35b_a3b_v8_bridge",
    )


def test_result_requires_full_evidence_binding_and_skip_requires_canonical_reason(tmp_path):
    surface = make_surface(tmp_path)
    fill(surface)
    target = surface / "np1_L2048"
    payload = result(surface.name, 1, 2048)
    payload["meta"]["kernel"] = "v7"
    target.joinpath("results.json").write_text(json.dumps(payload))
    other = surface / "np2_L2048"
    other.joinpath("results.json").unlink()
    other.joinpath("skip.txt").write_text("SKIP arbitrary capacity pressure\n")
    (surface / "complete.txt").write_text("COMPLETE 2026-07-27T00:00:00Z\n")
    row = aggregate.make_report(tmp_path, [surface.name])["surfaces"][0]
    assert row["invalid_cells"] == 2


def test_write_is_create_only_or_exact_idempotent(tmp_path, monkeypatch):
    surface = make_surface(tmp_path)
    fill(surface)
    (surface / "complete.txt").write_text("COMPLETE 2026-07-27T00:00:00Z\n")
    destination = tmp_path / "final.json"
    argv = ["aggregate", "--root", str(tmp_path), "--label", surface.name, "--write", str(destination)]
    monkeypatch.setattr("sys.argv", argv)
    assert aggregate.main() == 0
    original = destination.read_text()
    assert aggregate.main() == 0
    destination.write_text("different\n")
    assert aggregate.main() == 3
    assert destination.read_text() == "different\n"


def test_complete_marker_requires_exact_utc_grammar(tmp_path):
    surface = make_surface(tmp_path)
    fill(surface)
    marker = surface / "complete.txt"
    marker.write_text("COMPLETE someday\n")
    assert aggregate.make_report(tmp_path, [surface.name])["final_publishable"] is False
    marker.write_text("COMPLETE 2026-07-27T00:00:00Z\n")
    assert aggregate.make_report(tmp_path, [surface.name])["final_publishable"] is True


def test_publish_cleans_temp_if_link_fails(tmp_path, monkeypatch):
    destination = tmp_path / "final.json"
    monkeypatch.setattr(aggregate.os, "link", lambda *_: (_ for _ in ()).throw(OSError("simulated crash")))
    try:
        aggregate.publish_final(destination, "payload\n")
    except OSError:
        pass
    else:
        raise AssertionError("link failure must propagate")
    assert not destination.exists()
    assert not list(tmp_path.glob(".final.json.*.tmp"))


def quality_capture(directory: Path, *, n=2, rows=2, errors=0):
    directory.mkdir()
    payload = {
        "meta": {"kernel": "production-consolidated-v8", "arm": "arm_rb1024_suite", "models": "model.gguf",
                 "questions_pinned": "questions.json", "max_tokens": 1024, "enable_thinking": True,
                 "endpoint": "chat", "repeats": 1, "n_per_suite": n},
        "suites": [{"suite": "suite", "n": n, "n_questions": n, "errors": errors}],
    }
    (directory / "summary.json").write_text(json.dumps(payload))
    (directory / "per_question.jsonl").write_text("".join(
        json.dumps({"id": f"q{i}", "request_error": ""}) + "\n" for i in range(rows)
    ))


def validate_quality(directory: Path):
    return aggregate.validate_quality(directory, label="arm", model="model.gguf", suite_name="suite", expected_n=2,
                                      max_tokens=1024, questions="questions.json", thinking=True)


def test_quality_requires_full_zero_error_saved_capture(tmp_path):
    directory = tmp_path / "quality"
    quality_capture(directory)
    assert validate_quality(directory) is None
    quality_capture(tmp_path / "partial", rows=1)
    assert "per_question rows" in validate_quality(tmp_path / "partial")
    quality_capture(tmp_path / "error", errors=1)
    assert "request error" in validate_quality(tmp_path / "error")


def test_validate_cell_rejects_false_complete_dispositions(tmp_path):
    surface = make_surface(tmp_path)
    cell = surface / "np1_L2048"
    cell.mkdir()
    (cell / "results.json").write_text(json.dumps(result(surface.name, 1, 2048)))
    (cell / "skip.txt").write_text("SKIP n_ctx_slot=1024 vram=62G requested_L=2048\n")
    assert aggregate.validate_cell(surface, 1, 2048) == "both results.json and skip.txt exist"


def test_canonical_label_binds_mode_grid_model_and_binary(tmp_path):
    surface = make_surface(tmp_path, grid="full", label="A4_35b_a3b_v8_bridge")
    try:
        aggregate.load_spec(surface)
    except ValueError as exc:
        assert "requires mode/grid" in str(exc)
    else:
        raise AssertionError("canonical A4 accepted full grid")

    second = tmp_path / "second"; second.mkdir()
    surface = make_surface(second, label="A4_35b_a3b_v8_bridge")
    payload = result(surface.name, 1, 2048)
    payload["meta"].update(models="stale.gguf", binary="stale-server")
    cell = surface / "np1_L2048"; cell.mkdir()
    (cell / "results.json").write_text(json.dumps(payload))
    reason = aggregate.validate_cell(surface, 1, 2048)
    assert "canonical model" in reason
    payload["meta"].update(models=aggregate.CANONICAL_SURFACE_BINDINGS[surface.name][2])
    (cell / "results.json").write_text(json.dumps(payload))
    assert "frozen v8 binary" in aggregate.validate_cell(surface, 1, 2048)
