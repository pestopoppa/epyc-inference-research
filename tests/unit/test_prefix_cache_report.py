"""UFH14-B4: the before/after prefix-cache metric over serving-call records."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.analysis import prefix_cache_report as rpt
from src.backends.serving_calls import prefix_fingerprints
from src.model_server import InferenceRequest

T0 = datetime(2026, 10, 3, 6, 0, tzinfo=timezone.utc)


def _rec(minute: float, *, port=8083, role="architect_critic", cache_n=0, prompt_n=100,
         prompt_ms=100.0, prompt: str | None = None, dur_s: float = 10.0, dispatched=True):
    start = T0 + timedelta(minutes=minute)
    rec = {
        "schema": "epyc.orchestrator.serving_call.v1",
        "dispatched": dispatched,
        "ts_start": start.isoformat(),
        "ts_end": (start + timedelta(seconds=dur_s)).isoformat(),
        "role": role,
        "server": {"port": port},
        "timings": {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": prompt_ms},
        "request": {},
    }
    if prompt is not None:
        rec["request"]["prefix_fp"] = prefix_fingerprints(InferenceRequest(role=role, prompt=prompt))
    return rec


def _write(path: Path, recs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


HEAD = "H" * 40_000  # shares c2048, c8192, c32768 -> ~10.9k implied tokens


def test_hit_ratio_and_cold_large(tmp_path):
    log = _write(tmp_path / "s.jsonl", [
        _rec(0, cache_n=0, prompt_n=20_000, prompt_ms=40_000),     # cold large
        _rec(1, cache_n=19_000, prompt_n=1_000, prompt_ms=1_000),  # warm
    ])
    out = rpt.report(rpt.load_calls([log], None, None, None), 3600)
    port = out["ports"]["8083"]
    assert port["calls"] == 2
    assert port["hit_tok"] == pytest.approx(19_000 / 40_000)
    assert port["cold_large"] == 1 and port["cold_large_s"] == 40.0
    assert port["cold_large_share"] == pytest.approx(40 / 41, abs=1e-4)


def test_missed_reuse_is_attributed_to_the_shared_prefix(tmp_path):
    log = _write(tmp_path / "s.jsonl", [
        _rec(0, cache_n=0, prompt_n=14_000, prompt_ms=20_000, prompt=HEAD + "A" * 2000),
        # 5 min later, same 40k-char head, but nothing reused: a missed reuse
        _rec(5, cache_n=10, prompt_n=14_000, prompt_ms=20_000, prompt=HEAD + "B" * 2000),
        # 6 min later, same head, reused properly: not missed
        _rec(6, cache_n=13_000, prompt_n=1_000, prompt_ms=1_500, prompt=HEAD + "C" * 2000),
    ])
    port = rpt.report(rpt.load_calls([log], None, None, None), 3600)["ports"]["8083"]
    assert port["missed_reuse"] == 1
    implied = 32768 / rpt.CHARS_PER_TOKEN
    assert port["missed_tokens"] == int(implied - 10)
    assert port["missed_prefill_s"] == pytest.approx(20.0 * (implied - 10) / 14_000, rel=1e-3)
    assert port["fp_coverage"] == 1.0


def test_window_and_port_isolation(tmp_path):
    log = _write(tmp_path / "s.jsonl", [
        _rec(0, prompt=HEAD, prompt_n=14_000),
        _rec(120, prompt=HEAD, prompt_n=14_000),            # outside a 60-min window
        _rec(121, port=8070, prompt=HEAD, prompt_n=14_000),  # other server: never matches
    ])
    out = rpt.report(rpt.load_calls([log], None, None, None), 3600)
    assert out["ports"]["8083"]["missed_reuse"] == 0
    assert out["ports"]["8070"]["missed_reuse"] == 0


def test_concurrent_duplicates_are_counted(tmp_path):
    log = _write(tmp_path / "s.jsonl", [
        _rec(0, prompt=HEAD + "x", prompt_n=14_000, dur_s=600),
        _rec(1, prompt=HEAD + "y", prompt_n=14_000),  # starts while the first is in flight
    ])
    port = rpt.report(rpt.load_calls([log], None, None, None), 3600)["ports"]["8083"]
    assert port["concurrent_dupes"] == 1 and port["missed_reuse"] == 1


def test_records_without_timings_or_undispatched_are_skipped(tmp_path):
    bad = _rec(0)
    bad["timings"] = None
    log = _write(tmp_path / "s.jsonl", [bad, _rec(1, dispatched=False), _rec(2)])
    assert len(rpt.load_calls([log], None, None, None)) == 1


def test_pre_b4_records_have_zero_fp_coverage(tmp_path):
    log = _write(tmp_path / "s.jsonl", [_rec(0), _rec(1)])
    port = rpt.report(rpt.load_calls([log], None, None, None), 3600)["ports"]["8083"]
    assert port["fp_coverage"] == 0.0 and port["missed_reuse"] == 0


def test_rotated_shards_read_oldest_first(tmp_path):
    base = tmp_path / "serving_calls.jsonl"
    _write(tmp_path / "serving_calls.jsonl.2", [_rec(0)])
    _write(tmp_path / "serving_calls.jsonl.1", [_rec(1)])
    _write(base, [_rec(2)])
    assert [p.name for p in rpt.shard_paths(base)] == [
        "serving_calls.jsonl.2", "serving_calls.jsonl.1", "serving_calls.jsonl"]


def test_split_at_cli_reports_before_after_and_delta(tmp_path, capsys):
    log = _write(tmp_path / "s.jsonl", [
        _rec(0, cache_n=0, prompt_n=20_000, prompt_ms=40_000),
        _rec(30, cache_n=18_000, prompt_n=2_000, prompt_ms=3_000),
    ])
    out_json = tmp_path / "r.json"
    rc = rpt.main(["--log", str(log), "--split-at", (T0 + timedelta(minutes=10)).isoformat(),
                   "--json", str(out_json)])
    assert rc == 0
    out = json.loads(out_json.read_text())
    assert out["before"]["ports"]["8083"]["hit_tok"] == 0.0
    assert out["after"]["ports"]["8083"]["hit_tok"] == pytest.approx(0.9)
    assert out["delta_after_minus_before"]["8083"]["hit_tok"] == pytest.approx(0.9)
    assert out["directions"]["missed_prefill_share"] == "lower_better"


def test_missing_log_is_an_error(tmp_path):
    assert rpt.main(["--log", str(tmp_path / "nope.jsonl")]) == 2
