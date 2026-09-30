"""RI-16: the routing-stage latency reader, on synthetic progress JSONL."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.analysis import routing_stage_latency as reader
from src.runtime.routing_stage_timing import STAGE_KEYS


def _event(event_type: str, task_id: str, ts: str, path: str, **stage_ms) -> dict:
    stages = dict.fromkeys(STAGE_KEYS)
    stages.update(stage_ms)
    return {
        "event_type": event_type,
        "task_id": task_id,
        "timestamp": ts,
        "data": {"routing_path": path, "stage_ms": stages},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture(log_dir: Path) -> None:
    rows = [
        # t1: decision snapshot, then the final completion (preferred)
        _event("routing_decision", "t1", "2026-09-30T10:00:00+00:00", "chat",
               priors=1.0, route=10.0, route_total=12.0, total=12.0),
        _event("task_completed", "t1", "2026-09-30T10:00:05+00:00", "chat",
               priors=1.0, route=10.0, route_total=12.0, mode=3.0, review_gate=2.0,
               review_verdict=900.0, total=17.0),
        # t2: only a decision snapshot (no completion logged)
        _event("routing_decision", "t2", "2026-09-30T11:00:00+00:00", "chat",
               priors=2.0, route=20.0, route_total=25.0, total=25.0),
        # t3: unified stream, failed
        _event("task_failed", "t3", "2026-09-30T12:00:00+00:00", "unified_stream",
               priors=3.0, route=30.0, route_total=40.0, routing_context=5.0, total=45.0),
        # t4: legacy stream, outside the window used below
        _event("task_completed", "t4", "2026-09-29T09:00:00+00:00", "legacy_stream",
               routing_context=7.0, review_gate=1.0, total=8.0),
        # noise: no stage_ms / unrelated event
        {"event_type": "task_completed", "task_id": "t5",
         "timestamp": "2026-09-30T10:00:00+00:00", "data": {}},
        {"event_type": "gate_passed", "task_id": "t1",
         "timestamp": "2026-09-30T10:00:00+00:00", "data": {"stage_ms": {"route": 1e9}}},
    ]
    _write(log_dir / "2026-09-30.jsonl", rows[:4] + rows[5:])
    with gzip.open(log_dir / "2026-09-29.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(rows[4]) + "\n")


def test_percentile_nearest_rank() -> None:
    values = sorted(float(v) for v in range(1, 101))
    assert reader.percentile(values, 50) == 50.0
    assert reader.percentile(values, 95) == 95.0
    assert reader.percentile([4.0], 95) == 4.0


def test_completion_record_preferred_and_none_is_not_counted(tmp_path: Path) -> None:
    _fixture(tmp_path)
    requests = reader.collect_requests(reader.iter_events(tmp_path))
    assert set(requests) == {"t1", "t2", "t3", "t4"}
    assert requests["t1"]["final"] is True
    assert requests["t1"]["stage_ms"]["mode"] == 3.0
    assert requests["t2"]["final"] is False

    summary = reader.summarize(requests)
    stages = summary["stages"]
    assert summary["requests"] == 4
    assert summary["final_records"] == 3
    assert summary["by_path"] == {"chat": 2, "legacy_stream": 1, "unified_stream": 1}
    assert stages["mode"]["n"] == 1  # only t1 ran mode selection
    assert stages["route"] == {"n": 3, "p50": 20.0, "p95": 30.0, "max": 30.0}
    assert stages["total"]["n"] == 4
    assert stages["total"]["max"] == 45.0
    assert list(stages)[: 3] == ["priors", "route", "route_total"]  # pipeline order
    assert "memrl_init" not in stages  # never ran anywhere


def test_time_window_and_path_filters(tmp_path: Path) -> None:
    _fixture(tmp_path)
    since = reader.parse_time("2026-09-30T00:00")
    until = reader.parse_time("2026-09-30T11:30:00Z")
    requests = reader.collect_requests(
        reader.iter_events(tmp_path, since, until), since=since, until=until, path="chat"
    )
    assert set(requests) == {"t1", "t2"}

    stream = reader.collect_requests(reader.iter_events(tmp_path), path="unified_stream")
    assert set(stream) == {"t3"}


def test_relative_time() -> None:
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    assert reader.parse_time("24h", now=now) == datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
    assert reader.parse_time("90m", now=now) == datetime(2026, 9, 30, 10, 30, tzinfo=timezone.utc)
    assert reader.parse_time(None) is None


def test_main_prints_table_and_json(tmp_path: Path, capsys) -> None:
    _fixture(tmp_path)
    assert reader.main(["--log-dir", str(tmp_path), "--path", "chat"]) == 0
    out = capsys.readouterr().out
    assert "requests=2" in out
    assert "route_total" in out

    assert reader.main(["--log-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["requests"] == 4
    assert payload["stages"]["review_gate"]["n"] == 2

    assert reader.main(["--log-dir", str(tmp_path / "missing")]) == 2


def test_empty_window_reports_nothing(tmp_path: Path, capsys) -> None:
    _fixture(tmp_path)
    assert reader.main(["--log-dir", str(tmp_path), "--since", "2027-01-01"]) == 0
    assert "no stage_ms records" in capsys.readouterr().out


def test_reader_stage_order_matches_emitter() -> None:
    assert reader.STAGE_ORDER == STAGE_KEYS
