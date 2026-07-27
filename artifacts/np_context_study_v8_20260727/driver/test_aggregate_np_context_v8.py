import importlib.util
import hashlib
import json
from pathlib import Path
import sys


MODULE = Path(__file__).with_name("aggregate_np_context_v8.py")
SPEC = importlib.util.spec_from_file_location("aggregate_np_context_v8", MODULE)
aggregate = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = aggregate
SPEC.loader.exec_module(aggregate)


def make_surface(tmp_path: Path, grid="a4_bridge", label="bridge", mode="throughput_only", mtp_depth=None,
                 include_contract=True) -> Path:
    surface = tmp_path / label
    surface.mkdir(parents=True)
    mtp_depth = aggregate.CANONICAL_MTP_DEPTHS.get(label, 4) if mtp_depth is None else mtp_depth
    contract = f"grid_contract={aggregate.GRID_CONTRACTS[grid]}\n" if include_contract else ""
    (surface / "provenance.txt").write_text(
        f"mode={mode} grid={grid} mtp_depth={mtp_depth} thinking=false\n{contract}"
    )
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


def capacity_start_skip(cell: Path, np: int, length: int, *, signature="hip_error_out_of_memory") -> None:
    stderr = "ggml backend error: hipErrorOutOfMemory while allocating KV buffer\n"
    (cell / "server.stderr").write_text(stderr)
    model = aggregate.CANONICAL_SURFACE_BINDINGS.get(cell.parent.name, (None, None, "test-model.gguf"))[2]
    mtp_depth = aggregate.CANONICAL_MTP_DEPTHS.get(cell.parent.name, 0)
    spec = "" if mtp_depth == 0 else f" --spec-type draft-mtp --spec-draft-n-max {mtp_depth}"
    argv = (
        "env GGML_IQK=1 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin taskset -c 184-191 "
        f"{aggregate.V8_BINARY} -m {model} --host 127.0.0.1 --port 18072 --metrics --slots --jinja "
        f"--device ROCm0 -ngl all -fa on -np {np} -c {np * length} -t 8 -tb 8 -b 2048 -ub 2048 "
        f"-ctk f16 -ctv f16 --reasoning off{spec}\n"
    )
    (cell / "server.argv").write_text(argv)
    stderr_digest = hashlib.sha256(stderr.encode()).hexdigest()
    argv_digest = hashlib.sha256(argv.encode()).hexdigest()
    cell.joinpath("skip.txt").write_text(
        f"SKIP capacity_start signature={signature} stderr_sha256={stderr_digest} server_argv_sha256={argv_digest} "
        f"requested_np={np} requested_L={length} requested_ctx={np * length}\n"
    )


def canonical_full_surface(tmp_path: Path) -> Path:
    return make_surface(tmp_path, grid="full", label="A3_ff_fable_non_mtp_q8", mode="full")


def rewrite_argv_and_hash(cell: Path, changed: str) -> None:
    cell.joinpath("server.argv").write_text(changed)
    skip = cell / "skip.txt"
    old_hash = skip.read_text().split("server_argv_sha256=", 1)[1].split(" ", 1)[0]
    skip.write_text(skip.read_text().replace(old_hash, hashlib.sha256(changed.encode()).hexdigest()))


def test_incomplete_surface_does_not_publish(tmp_path):
    surface = make_surface(tmp_path)
    report = aggregate.make_report(tmp_path, [surface.name])
    assert report["final_publishable"] is False
    assert report["surfaces"][0]["missing_cells"] == 11


def test_full_grid_is_literal_cartesian_and_legacy_triangular_surface_is_incomplete(tmp_path):
    assert len(aggregate.required_cells("full")) == 24
    assert len(aggregate.required_cells("a4_bridge")) == 11
    assert set(aggregate.required_cells("full")) == {
        (np, length) for np in (1, 2, 4, 8, 16, 32) for length in (2048, 8192, 16384, 32768)
    }
    surface = make_surface(tmp_path, grid="full")
    legacy = {
        2048: (1, 2, 4, 8, 16, 32),
        8192: (1, 2, 4, 8, 16),
        16384: (1, 2, 4, 8),
        32768: (1, 2, 4),
    }
    for length, nps in legacy.items():
        for np in nps:
            cell = surface / f"np{np}_L{length}"
            cell.mkdir()
            cell.joinpath("results.json").write_text(json.dumps(result(surface.name, np, length)))
    report = aggregate.make_report(tmp_path, [surface.name])["surfaces"][0]
    assert report["missing_cells"] == 6
    assert report["state"] == "incomplete"


def test_grid_contract_rejects_legacy_missing_duplicate_or_conflicting_lines(tmp_path):
    legacy = make_surface(tmp_path, grid="full", include_contract=False)
    try:
        aggregate.load_spec(legacy)
    except ValueError as exc:
        assert "exactly one grid_contract" in str(exc)
    else:
        raise AssertionError("legacy full provenance unexpectedly accepted")

    duplicate = make_surface(tmp_path / "duplicate", grid="full")
    provenance = duplicate / "provenance.txt"
    provenance.write_text(provenance.read_text() + "grid_contract=cartesian24_v2\n")
    try:
        aggregate.load_spec(duplicate)
    except ValueError as exc:
        assert "exactly one grid_contract" in str(exc)
    else:
        raise AssertionError("duplicate grid contract unexpectedly accepted")

    conflicting = make_surface(tmp_path / "conflicting", grid="full")
    provenance = conflicting / "provenance.txt"
    provenance.write_text(provenance.read_text().replace("grid_contract=cartesian24_v2", "grid_contract=triangular18_v1"))
    try:
        aggregate.load_spec(conflicting)
    except ValueError as exc:
        assert "not canonical" in str(exc)
    else:
        raise AssertionError("conflicting grid contract unexpectedly accepted")


def test_new_grid_contract_provenance_passes_and_driver_self_test_exercises_append_only_migration(tmp_path):
    import subprocess

    assert aggregate.load_spec(make_surface(tmp_path, grid="full")).cells == aggregate.required_cells("full")
    surface_root = Path(__file__).parent
    completed = subprocess.run(
        ["bash", str(surface_root / "run_model_block.sh"), "--self-test"],
        check=True, capture_output=True, text=True,
    )
    assert completed.stdout == "RUN_MODEL_BLOCK_SELF_TEST_OK\n"


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


def test_legacy_healthy_server_resource_skip_remains_valid(tmp_path):
    surface = make_surface(tmp_path)
    cell = surface / "np1_L2048"
    cell.mkdir()
    cell.joinpath("skip.txt").write_text("SKIP n_ctx_slot=1024 vram=62G requested_L=2048\n")
    assert aggregate.validate_cell(surface, 1, 2048) is None


def test_successful_result_validation_is_unchanged(tmp_path):
    surface = make_surface(tmp_path, grid="full")
    cell = surface / "np32_L32768"
    cell.mkdir()
    cell.joinpath("results.json").write_text(json.dumps(result(surface.name, 32, 32768)))
    assert aggregate.validate_cell(surface, 32, 32768) is None


def test_capacity_start_skip_is_terminal_only_with_bound_oom_evidence(tmp_path):
    surface = canonical_full_surface(tmp_path)
    cell = surface / "np32_L8192"
    cell.mkdir()
    capacity_start_skip(cell, 32, 8192)
    assert aggregate.validate_cell(surface, 32, 8192) is None


def test_capacity_start_skip_rejects_tampered_hash_signature_or_request(tmp_path):
    surface = canonical_full_surface(tmp_path)
    cell = surface / "np32_L8192"
    cell.mkdir()
    capacity_start_skip(cell, 32, 8192)
    skip = cell / "skip.txt"
    original = skip.read_text()
    digest = original.split("stderr_sha256=", 1)[1].split(" ", 1)[0]
    tampered_digest = ("0" if digest[-1] != "0" else "1") + digest[1:]
    skip.write_text(original.replace(digest, tampered_digest))
    assert "SHA-256 mismatch" in aggregate.validate_cell(surface, 32, 8192)
    skip.write_text(original.replace("hip_error_out_of_memory", "allocation_failure"))
    assert "signature is absent" in aggregate.validate_cell(surface, 32, 8192)
    skip.write_text(original.replace("requested_np=32", "requested_np=16"))
    assert "does not bind" in aggregate.validate_cell(surface, 32, 8192)


def test_capacity_start_skip_rejects_generic_crash_and_non_new_cell(tmp_path):
    surface = canonical_full_surface(tmp_path)
    cell = surface / "np32_L8192"
    cell.mkdir()
    stderr = "fatal: unable to bind HTTP port\n"
    cell.joinpath("server.stderr").write_text(stderr)
    digest = hashlib.sha256(stderr.encode()).hexdigest()
    argv = "env GGML_IQK=1 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin taskset -c 184-191 " \
        f"{aggregate.V8_BINARY} -m {aggregate.CANONICAL_SURFACE_BINDINGS[surface.name][2]} " \
        "--host 127.0.0.1 --port 18072 --metrics --slots --jinja --device ROCm0 -ngl all -fa on " \
        "-np 32 -c 262144 -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 --reasoning off\n"
    cell.joinpath("server.argv").write_text(argv)
    argv_digest = hashlib.sha256(argv.encode()).hexdigest()
    cell.joinpath("skip.txt").write_text(
        f"SKIP capacity_start signature=hip_error_out_of_memory stderr_sha256={digest} server_argv_sha256={argv_digest} "
        "requested_np=32 requested_L=8192 requested_ctx=262144\n"
    )
    assert "signature is absent" in aggregate.validate_cell(surface, 32, 8192)
    cell = surface / "np1_L2048"
    cell.mkdir()
    capacity_start_skip(cell, 1, 2048)
    assert ">=262144" in aggregate.validate_cell(surface, 1, 2048)


def test_capacity_start_skip_rejects_tampered_or_wrong_argv(tmp_path):
    surface = canonical_full_surface(tmp_path)
    cell = surface / "np32_L8192"
    cell.mkdir()
    capacity_start_skip(cell, 32, 8192)
    cell.joinpath("server.argv").write_text("tampered argv\n")
    assert "server.argv SHA-256 mismatch" in aggregate.validate_cell(surface, 32, 8192)

    capacity_start_skip(cell, 32, 8192)
    argv = cell.joinpath("server.argv")
    changed = argv.read_text().replace("-np 32", "-np 16")
    argv.write_text(changed)
    skip = cell.joinpath("skip.txt")
    old_hash = skip.read_text().split("server_argv_sha256=", 1)[1].split(" ", 1)[0]
    skip.write_text(skip.read_text().replace(old_hash, hashlib.sha256(changed.encode()).hexdigest()))
    assert "-np is not '32'" in aggregate.validate_cell(surface, 32, 8192)

    capacity_start_skip(cell, 32, 8192)
    argv = cell.joinpath("server.argv")
    changed = argv.read_text().replace("-c 262144", "-c 131072")
    argv.write_text(changed)
    skip = cell.joinpath("skip.txt")
    old_hash = skip.read_text().split("server_argv_sha256=", 1)[1].split(" ", 1)[0]
    skip.write_text(skip.read_text().replace(old_hash, hashlib.sha256(changed.encode()).hexdigest()))
    assert "-c is not '262144'" in aggregate.validate_cell(surface, 32, 8192)

    capacity_start_skip(cell, 32, 8192)
    argv = cell.joinpath("server.argv")
    changed = argv.read_text().replace(aggregate.V8_BINARY, "/tmp/wrong-llama-server")
    argv.write_text(changed)
    skip = cell.joinpath("skip.txt")
    old_hash = skip.read_text().split("server_argv_sha256=", 1)[1].split(" ", 1)[0]
    skip.write_text(skip.read_text().replace(old_hash, hashlib.sha256(changed.encode()).hexdigest()))
    assert "canonical v8 launch prefix" in aggregate.validate_cell(surface, 32, 8192)

    canonical = canonical_full_surface(tmp_path / "canonical")
    cell = canonical / "np32_L8192"
    cell.mkdir()
    capacity_start_skip(cell, 32, 8192)
    argv = cell.joinpath("server.argv")
    changed = argv.read_text().replace("-m " + aggregate.CANONICAL_SURFACE_BINDINGS[canonical.name][2], "-m wrong.gguf")
    argv.write_text(changed)
    skip = cell.joinpath("skip.txt")
    old_hash = skip.read_text().split("server_argv_sha256=", 1)[1].split(" ", 1)[0]
    skip.write_text(skip.read_text().replace(old_hash, hashlib.sha256(changed.encode()).hexdigest()))
    assert "-m is not canonical" in aggregate.validate_cell(canonical, 32, 8192)


def test_capacity_start_skip_rejects_memory_flags_and_mtp_tampering(tmp_path):
    surface = canonical_full_surface(tmp_path)
    cell = surface / "np32_L8192"
    cell.mkdir()
    for old, new, expected in (
        ("-ctk f16", "-ctk f32", "-ctk is not 'f16'"),
        ("-ctv f16", "-ctv f32", "-ctv is not 'f16'"),
        ("-b 2048", "-b 1024", "-b is not '2048'"),
    ):
        capacity_start_skip(cell, 32, 8192)
        changed = cell.joinpath("server.argv").read_text().replace(old, new)
        rewrite_argv_and_hash(cell, changed)
        assert expected in aggregate.validate_cell(surface, 32, 8192)

    capacity_start_skip(cell, 32, 8192)
    changed = cell.joinpath("server.argv").read_text().replace("--reasoning off", "--reasoning off --spec-type draft-mtp --spec-draft-n-max 1")
    rewrite_argv_and_hash(cell, changed)
    assert "MTP flags for mtp_depth 0" in aggregate.validate_cell(surface, 32, 8192)

    thinkingcap = make_surface(tmp_path / "tc", grid="full", label="A3_tc_thinkingcap_q8", mode="full")
    cell = thinkingcap / "np32_L8192"
    cell.mkdir()
    capacity_start_skip(cell, 32, 8192)
    changed = cell.joinpath("server.argv").read_text().replace(" --spec-type draft-mtp", "")
    rewrite_argv_and_hash(cell, changed)
    assert "missing or duplicate --spec-type" in aggregate.validate_cell(thinkingcap, 32, 8192)

    capacity_start_skip(cell, 32, 8192)
    changed = cell.joinpath("server.argv").read_text().replace("--spec-draft-n-max 4", "--spec-draft-n-max 1")
    rewrite_argv_and_hash(cell, changed)
    assert "--spec-draft-n-max is not '4'" in aggregate.validate_cell(thinkingcap, 32, 8192)

    capacity_start_skip(cell, 32, 8192)
    changed = cell.joinpath("server.argv").read_text().replace("--spec-type draft-mtp", "--spec-type draft-mtp --spec-type draft-mtp", 1)
    rewrite_argv_and_hash(cell, changed)
    assert "missing or duplicate --spec-type" in aggregate.validate_cell(thinkingcap, 32, 8192)

    capacity_start_skip(cell, 32, 8192)
    changed = cell.joinpath("server.argv").read_text().replace("--slots", "--slots --slots", 1)
    rewrite_argv_and_hash(cell, changed)
    assert "missing or duplicate --slots" in aggregate.validate_cell(thinkingcap, 32, 8192)

    (thinkingcap / "provenance.txt").write_text(
        "mode=full grid=full mtp_depth=1 thinking=false\ngrid_contract=cartesian24_v2\n"
    )
    assert "requires mtp_depth 4" in aggregate.validate_cell(thinkingcap, 32, 8192)


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


def test_fable_mtp_canonical_surface_is_throughput_only(tmp_path):
    surface = make_surface(
        tmp_path,
        grid="full",
        label="A3_ff_fable_mtp_q8",
        mode="throughput_only",
        mtp_depth=1,
    )
    spec = aggregate.load_spec(surface)
    assert (spec.mode, spec.grid) == ("throughput_only", "full")


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

    second = tmp_path / "second"
    second.mkdir()
    surface = make_surface(second, label="A4_35b_a3b_v8_bridge")
    payload = result(surface.name, 1, 2048)
    payload["meta"].update(models="stale.gguf", binary="stale-server")
    cell = surface / "np1_L2048"
    cell.mkdir()
    (cell / "results.json").write_text(json.dumps(payload))
    reason = aggregate.validate_cell(surface, 1, 2048)
    assert "canonical model" in reason
    payload["meta"].update(models=aggregate.CANONICAL_SURFACE_BINDINGS[surface.name][2])
    (cell / "results.json").write_text(json.dumps(payload))
    assert "frozen v8 binary" in aggregate.validate_cell(surface, 1, 2048)
