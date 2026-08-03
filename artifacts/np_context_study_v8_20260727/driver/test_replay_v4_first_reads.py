import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import mock


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("replay_v4_first_reads", HERE / "replay_v4_first_reads.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_static_authorities_are_pinned_and_loadable():
    swe_ids, lcb_questions, converter, scorer, harness = replay.static_authorities()
    assert len(swe_ids) == 40
    assert len(lcb_questions) == 53
    assert hasattr(converter, "apply_blocks")
    assert hasattr(scorer, "score_code")
    assert harness.is_file()
    assert replay.CONVERTER_SHA256.startswith("6bd2302d")
    assert "expanded-six-arm-v4-tail-replay" in str(replay.CONVERTER)


def test_partial_or_missing_future_capture_fails_closed():
    future_arm = {"name": "future", "label": "no_such_capture", "mtp_depth": 0}
    try:
        replay.complete_rows(future_arm, "swebench_oracle", ["one"])
    except RuntimeError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("missing capture was accepted")


def test_incomplete_inputs_do_not_allocate_a_sealed_run(tmp_path):
    with mock.patch.object(replay, "ART", tmp_path):
        with mock.patch.object(replay, "static_authorities", return_value=(["one"], {"two": {}}, None, None, Path("/tmp/harness"))):
            with mock.patch.object(replay, "complete_rows", side_effect=RuntimeError("incomplete")):
                try:
                    replay.seal()
                except RuntimeError as exc:
                    assert "incomplete" in str(exc)
                else:
                    raise AssertionError("partial capture was sealed")
    assert not (tmp_path / "v4_first_read_replays").exists()


def test_hash_ledger_tamper_is_rejected(tmp_path):
    payload = tmp_path / "value.txt"
    payload.write_text("original")
    replay.write_digest(tmp_path)
    replay.verify_digest(tmp_path)
    payload.write_text("changed")
    try:
        replay.verify_digest(tmp_path)
    except RuntimeError as exc:
        assert "hash ledger" in str(exc)
    else:
        raise AssertionError("tampered sealed package was accepted")


def test_q4_terminal_contract_requires_real_runner_receipts(tmp_path):
    summary = {"schema": "epyc.laguna_q4_cpu_v8.summary.v2", "status": "ok"}
    clean = {"members_after_kill": [], "port_free": True}
    docker = {"execution_error": "", "postflight_errors": [], "residual_unproven_ids": [], "cleanup_failed_ids": []}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "swe_oracle.cleanup.json").write_text(json.dumps(clean))
    (tmp_path / "lcb_hard.cleanup.json").write_text(json.dumps(clean))
    (tmp_path / "swe_docker_terminal.json").write_text(json.dumps(docker))
    replay.require_q4_terminal(tmp_path)
    (tmp_path / "lcb_hard.cleanup.json").write_text(json.dumps({"members_after_kill": [7], "port_free": True}))
    try:
        replay.require_q4_terminal(tmp_path)
    except RuntimeError as exc:
        assert "cleanup" in str(exc)
    else:
        raise AssertionError("unclean Q4 terminal receipt was accepted")


def test_reviewed_official_report_validator_rejects_harness_error(tmp_path):
    q4 = replay.load_q4_authority()
    ids = [f"id-{index}" for index in range(40)]
    report = {
        "schema_version": 2, "total_instances": 40, "submitted_instances": 40,
        "completed_instances": 39, "resolved_instances": 1, "unresolved_instances": 38,
        "empty_patch_instances": 1, "error_instances": 1,
        "submitted_ids": ids, "completed_ids": ids[:-1], "empty_patch_ids": [ids[-1]],
        "resolved_ids": [ids[0]], "unresolved_ids": ids[1:-1], "error_ids": [ids[0]], "incomplete_ids": [],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    try:
        q4.validate_official_swe_report(path, {"empty_patch_ids": [ids[-1]]})
    except RuntimeError as exc:
        assert "harness" in str(exc) or "denominator" in str(exc)
    else:
        raise AssertionError("official harness error was accepted")
