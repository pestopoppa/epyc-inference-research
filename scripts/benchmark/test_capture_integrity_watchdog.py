#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import capture_integrity_watchdog as watchdog


def _status(**overrides):
    status = {
        "schema_version": watchdog.CANONICAL_CAPTURE_SCHEMA,
        "runner_source_sha256": "a" * 64,
        "suite": "swebench_oracle",
        "completed_draws": 1,
        "expected_draws": 1,
        "complete": True,
        "request_error_rows": 0,
        "length_cap_rows": 0,
        "artifact_integrity_fail_closed": False,
        "swebench_search_replace": {"state_counts": {}},
    }
    status.update(overrides)
    return status


def _write(path, **overrides):
    path.write_text(json.dumps(_status(**overrides)))


def test_integrity_failure_fails_closed(tmp_path):
    path = tmp_path / "bad.live-status.json"
    _write(path, artifact_integrity_fail_closed=True)

    assert watchdog.watch_paths([path], watch=False, poll_interval_s=1, startup_grace_s=0,
                                stale_timeout_s=1, request_error_threshold=1) == 1


def test_request_errors_fail_at_configurable_threshold(tmp_path):
    path = tmp_path / "error.live-status.json"
    _write(path, request_error_rows=2)

    assert watchdog.watch_paths([path], watch=False, poll_interval_s=1, startup_grace_s=0,
                                stale_timeout_s=1, request_error_threshold=2) == 1


def test_length_only_is_a_warning_not_a_failure(tmp_path, capsys):
    path = tmp_path / "length.live-status.json"
    _write(path, length_cap_rows=1, swebench_search_replace={
        "state_counts": {"model_truncation_no_patch": 1},
    })

    assert watchdog.watch_paths([path], watch=False, poll_interval_s=1, startup_grace_s=0,
                                stale_timeout_s=1, request_error_threshold=1) == 0
    assert "WARNING" in capsys.readouterr().err


def test_malformed_status_fails_after_startup_grace(tmp_path):
    path = tmp_path / "malformed.live-status.json"
    path.write_text("{")

    assert watchdog.watch_paths([path], watch=False, poll_interval_s=1, startup_grace_s=0,
                                stale_timeout_s=1, request_error_threshold=1) == 1


def test_stale_incomplete_status_fails_in_watch_mode(tmp_path):
    path = tmp_path / "stale.live-status.json"
    _write(path, completed_draws=0, expected_draws=2, complete=False)
    ticks = iter((0.0, 0.0, 2.0, 2.0))

    assert watchdog.watch_paths(
        [path], watch=True, poll_interval_s=1, startup_grace_s=0,
        stale_timeout_s=1, request_error_threshold=1, monotonic=lambda: next(ticks), sleep=lambda _: None,
    ) == 1


def test_successful_completion_passes(tmp_path):
    path = tmp_path / "complete.live-status.json"
    _write(path)

    assert watchdog.watch_paths([path], watch=True, poll_interval_s=1, startup_grace_s=0,
                                stale_timeout_s=1, request_error_threshold=1) == 0


def test_one_shot_fails_incomplete_but_observe_once_allows_inspection(tmp_path):
    path = tmp_path / "incomplete.live-status.json"
    _write(path, completed_draws=0, expected_draws=1, complete=False)

    assert watchdog.watch_paths([path], watch=False, poll_interval_s=1, startup_grace_s=0,
                                stale_timeout_s=1, request_error_threshold=1) == 1
    assert watchdog.watch_paths([path], watch=False, poll_interval_s=1, startup_grace_s=0,
                                stale_timeout_s=1, request_error_threshold=1,
                                require_complete=False) == 0
