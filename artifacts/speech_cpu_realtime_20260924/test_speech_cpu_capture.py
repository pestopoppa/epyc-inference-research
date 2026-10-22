"""Tests for the VB-SPEECH-CPU-1/1a write side (speech_cpu_capture.py). No server, no inference.

run: cd <research root> && /usr/bin/python3 -m pytest -q artifacts/speech_cpu_realtime_20260924/test_speech_cpu_capture.py
"""
import json
import os
import py_compile
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import speech_cpu_capture as C  # noqa: E402


@pytest.fixture
def fake_bin(tmp_path):
    p = tmp_path / "whisper-server"
    p.write_bytes(b"\x7fELF fake binary for digest tests")
    return {"stt": C.binary_digest(str(p)), "tts": C.binary_digest(str(p))}


def _cot(t0, t1, generating=False):
    return {"port": 8070, "samples": [{"t": (t0 + t1) / 2, "pid": 1, "resolved": True}],
            "n_samples": 1, "generating": generating, "generating_basis": {}}


def _stt(rtf=0.19, **kw):
    t0 = 1000.0
    d = {"kind": "stt", "clip": "short_11s.wav", "audio_s": 11.0, "wall_s": 2.1, "rtf": rtf,
         "status": 200, "t_start_epoch": t0, "t_end_epoch": t0 + 2.1}
    d.update(kw)
    return d


def _rows(native, fake_bin, **kw):
    args = dict(harness="speech_cpu_bench", run_id="r1", row_index=0, binaries=fake_bin,
                cores="0-23", threads=24, shim_nprocs=None,
                cotenant=_cot(native.get("t_start_epoch", 0), native.get("t_end_epoch", 0)),
                cap_s=600.0, mode="solo", emitted_at="2026-10-22T12:00:00Z")
    args.update(kw)
    return C.make_rows(native, **args)


def test_stt_row_is_valid_and_self_hashed(fake_bin):
    (row,) = _rows(_stt(), fake_bin)
    assert C.validate_row(row) == []
    assert row["schema"] == C.CAPTURE_SCHEMA and row["metric"] == "stt_rtf"
    assert row["metric_direction"] == "lower_better" and row["value"] == 0.19
    assert row["category"] == "BASELINE" and row["request"]["completed"] is True
    assert row["row_sha256"] == C.row_digest(row)


def test_tampered_row_fails_hash(fake_bin):
    (row,) = _rows(_stt(), fake_bin)
    row["value"] = 0.01
    assert any("row_sha256" in p for p in C.validate_row(row))


def test_tts_row_needs_shim_nprocs(fake_bin):
    t0 = 1000.0
    tts = {"kind": "tts", "fmt": "pcm", "status": 200, "first_packet_s": 0.11, "rtf": 0.6,
           "t_start_epoch": t0, "t_end_epoch": t0 + 3}
    rows = _rows(tts, fake_bin, cores="24-39", threads=16, shim_nprocs=32)
    assert [r["metric"] for r in rows] == ["tts_rtf", "tts_first_packet_s"]
    with pytest.raises(C.CaptureRefused, match="SHIM_NPROCS"):
        _rows(tts, fake_bin, cores="24-39", threads=16, shim_nprocs=None)


def test_layout_is_required(fake_bin):
    with pytest.raises(C.CaptureRefused, match="core list"):
        _rows(_stt(), fake_bin, cores="", threads=24)


def test_missing_binary_digest_refused(fake_bin):
    with pytest.raises(C.CaptureRefused, match="sha256"):
        _rows(_stt(), {"stt": {"path": "/x", "sha256": "abc"}})


def test_cotenant_must_be_sampled_during_row(fake_bin):
    native = _stt()
    late = {"port": 8070, "samples": [{"t": native["t_end_epoch"] + 5}], "generating": False}
    with pytest.raises(C.CaptureRefused, match="inside the row's window"):
        _rows(native, fake_bin, cotenant=late)
    with pytest.raises(C.CaptureRefused, match="sampled, not assumed"):
        _rows(native, fake_bin, cotenant={})


def test_timed_out_row_never_reads_as_completed(fake_bin):
    native = _stt(rtf=None, wall_s=None, timed_out_after_s=90.0, rtf_lower_bound=8.2)
    native.pop("status")
    (row,) = _rows(native, fake_bin, harness="live_llm_contention", cap_s=90.0)
    assert row["request"]["completed"] is False and row["request"]["timed_out"] is True
    assert row["value"] is None and row["reps_basis"] == "attempted"
    assert C.validate_row(row) == []


def test_guard_aborted_row_is_not_completed_even_with_a_value(fake_bin):
    (row,) = _rows(_stt(), fake_bin, guard_aborted=True)
    assert row["request"]["guard_aborted"] is True and row["value"] is None


def test_forged_completed_capped_row_is_refused(fake_bin):
    (row,) = _rows(_stt(), fake_bin)
    row["request"]["timed_out"] = True
    row["row_sha256"] = C.row_digest(row)
    assert any("cannot be completed" in p for p in C.validate_row(row))


def test_contended_row_is_candidate(fake_bin):
    native = _stt()
    (row,) = _rows(native, fake_bin, mode="stt+tts+llm")
    assert row["category"] == "CANDIDATE"


def test_llm_rows_bench_and_live_shapes(fake_bin):
    t0 = 1000.0
    bench = {"kind": "llm", "llm": {"status": 200, "capped_at_s": None, "t_start_epoch": t0,
                                    "t_end_epoch": t0 + 9},
             "llm_timings": {"predicted_per_second": 43.1}, "t_start_epoch": t0, "t_end_epoch": t0 + 9}
    (row,) = _rows(bench, fake_bin, mode="solo-llm")
    assert row["metric"] == "cotenant_llm_tok_s" and row["value"] == 43.1
    live = {"kind": "llm", "capped": True, "tok_s": 0.3, "t_start_epoch": t0, "t_end_epoch": t0 + 180}
    (row,) = _rows(live, fake_bin, harness="live_llm_contention", cap_s=180.0)
    assert row["request"]["completed"] is False and row["value"] is None


def test_legacy_rows_validate_as_refused():
    legacy = json.loads((HERE / "raw" / "live_contention.jsonl").read_text().splitlines()[0])
    assert C.validate_row(legacy)  # no schema/hash -> refused, zero tuples, never backfilled


def test_refuses_writing_into_retrospective_raw(tmp_path):
    with pytest.raises(C.CaptureRefused, match="retrospective"):
        C.open_capture_sink(str(C.RETROSPECTIVE_RAW))
    with pytest.raises(C.CaptureRefused):
        C.open_capture_sink(str(C.RETROSPECTIVE_RAW / "sub"))
    sink = C.open_capture_sink(str(tmp_path / "new"))
    sink.close()


def test_sample_process_reads_own_pid():
    s = C.sample_process(os.getpid(), interval_s=0.01)
    assert s["resolved"] and s["argv"] and s["cpus_allowed_list"]
    assert C.sample_process(None)["resolved"] is False


def test_harness_capture_wrap_emit_flush(tmp_path):
    b = tmp_path / "bin"
    b.write_bytes(b"binary")
    cap = C.HarnessCapture("speech_cpu_bench", str(tmp_path / "out"), cotenant_port=None,
                           binaries={"stt": str(b)}, run_id="t1", sample_interval_s=0.01)

    def fake_request():
        time.sleep(0.05)
        return {"kind": "stt", "rtf": 0.2, "status": 200}

    res = cap.wrap(fake_request)()
    assert "t_start_epoch" in res and res["_cotenant"]["n_samples"] >= 1
    assert "_cotenant" not in C.strip_private(res)
    cap.emit(res, cores="0-23", threads=24, shim_nprocs=None, cap_s=600.0, mode="solo")
    assert cap.flush() == 1
    lines = (tmp_path / "out" / "speech_cpu_capture.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    assert C.validate_row(row) == [] and "_cotenant" not in row["native"]


def test_harness_capture_own_load_marks_generating(tmp_path):
    b = tmp_path / "bin"
    b.write_bytes(b"binary")
    cap = C.HarnessCapture("speech_cpu_bench", str(tmp_path / "o"), cotenant_port=None,
                           binaries={"stt": str(b)}, sample_interval_s=0.01)

    def llm(results, key):
        time.sleep(0.05)
        results[key] = {"status": 200}

    d = {}
    cap.wrap(llm, own_load=True)(d, "llm")
    assert d["llm"]["_cotenant"]["generating"] is True
    assert cap.own_in_flight == 0


def test_harnesses_still_compile():
    for name in ("speech_cpu_bench.py", "live_llm_contention.py", "live_llm_contention_capture.py"):
        py_compile.compile(str(HERE / name), doraise=True)


def test_bench_import_without_capture_is_legacy(monkeypatch):
    monkeypatch.delenv("SPEECH_CPU_CAPTURE_DIR", raising=False)
    sys.modules.pop("speech_cpu_bench", None)
    import speech_cpu_bench as B
    assert B.CAPTURE is None and B.RAW.endswith("/raw")


def test_pinned_retrospective_harness_is_byte_identical():
    import hashlib
    sums = dict(reversed(l.split("  ", 1)) for l in (HERE / "SHA256SUMS").read_text().splitlines() if l)
    got = hashlib.sha256((HERE / "live_llm_contention.py").read_bytes()).hexdigest()
    assert sums["./live_llm_contention.py"] == got
