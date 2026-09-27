"""VB-UFH12-PLACEMENT: the placement gate's belief sidecar (offline, no live server).

The snapshot function is injected, so nothing here reads another process or sends a request.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.server import embedder_placement_capture as cap
from scripts.server import embedder_placement_gate as gate

GATE = Path(gate.__file__).resolve()
T0 = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)
FD = [8070, 8080]


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _proc(port: int, cpus: str = "0-3") -> dict:
    argv = ["/mnt/raid0/llm/kernels/production/cpu/bin/llama-server", "--port", str(port)]
    return {"pid": 1000 + port, "argv": argv, "argv_sha256": cap.content_hash(argv),
            "cpus_allowed_list": cpus, "exe": argv[0], "exe_sha256": "a" * 64,
            "ld_library_path": "/mnt/raid0/llm/kernels/production/cpu/lib",
            "ggml_libs": ["/mnt/raid0/llm/kernels/production/cpu/lib/libggml.so"]}


def _snapshots(*, drift: bool = False, missing: int | None = None, topo: str = "4893e37e00000000"):
    calls = {"n": 0}

    def snap(process_ports, props_ports):
        calls["n"] += 1
        procs = {str(p): (None if p == missing else _proc(p)) for p in process_ports}
        if drift and calls["n"] == 2:
            procs[str(gate.POOL_PORTS[0])] = _proc(gate.POOL_PORTS[0], cpus="0-95")
        return {"processes": procs,
                "frontdoor_props": {str(p): {"build_info": "b10303-ffc1bac82", "model_path": "/m.gguf"}
                                    for p in props_ports},
                "topology_hash": topo}
    return snap


def _record(label: str = "post-cap", load_mode: str = "scheduler", start: datetime = T0) -> dict:
    g1 = {}
    for port in FD:
        q = [50.0, 50.5, 49.8, 50.2]
        s = [47.0, 47.5, 46.9]
        ratios = [si / ((qa + qb) / 2) for si, qa, qb in zip(s, q, q[1:])]
        g1[str(port)] = {"unit": "decode tok/s (timings.predicted_per_second)",
                         "q_samples": q, "s_samples": s, "s_over_q_all": ratios,
                         "s_over_q_median": sorted(ratios)[1],
                         "aa_noise_floor_rel_median": cap.aa_floor(q)}
        if load_mode == "scheduler":
            g1[str(port)]["s_embedding_load"] = [
                {"texts_per_s": v, "counters_delta": {"granted": 10}, "deferred": 0}
                for v in (30.0, 31.0, 29.0)]
    return {
        "schema": cap.RECORD_SCHEMA, "label": label, "load_mode": load_mode,
        "started_at": _iso(start),
        "params": {"fd_ports": FD, "pool_ports": gate.POOL_PORTS, "per_port_in_flight": 4,
                   "load_warmup_s": 3.0, "pairs": 3, "n_predict": 256, "g2_window_s": 60.0,
                   "sched_admission_wait_s": 30.0, "fd_prompt_sha256": "b" * 64,
                   "emb_text_sha256": "c" * 64},
        "scheduler_policy": {"neighbour_cap": {"max_in_flight": 1}},
        "g2_pool_scaling": {"one_port": {"texts": 551, "seconds": 60.0, "texts_per_s": 551 / 60},
                            "whole_pool": {"texts": 2734, "seconds": 60.0, "texts_per_s": 2734 / 60},
                            "scaling_ratio": 2734 / 551},
        "g1_frontdoor_decode": g1,
        "finished_at": _iso(start + timedelta(minutes=20)),
    }


def _window(snap=None, ports=None):
    w = cap.CaptureWindow(ports or gate.POOL_PORTS + FD, FD, snapshot_fn=snap or _snapshots())
    w.begin()
    w.mark("g2", _iso(T0), _iso(T0 + timedelta(minutes=2)))
    w.mark("g1", _iso(T0 + timedelta(minutes=2)), _iso(T0 + timedelta(minutes=20)))
    w.finish()
    return w


def _write(tmp_path, record, window=None, emitted=None):
    out = tmp_path / "post-cap-20260927.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    return out, cap.write_belief_measurements(
        out, record, window=window or _window(), producer="test", gate_path=GATE,
        emitted_at=emitted or _iso(T0 + timedelta(minutes=21)))


def test_sidecar_rows_are_self_hashed_and_rederive(tmp_path):
    out, sidecar = _write(tmp_path, _record())
    assert sidecar.name == "post-cap-20260927.belief_measurements.jsonl"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    metrics = sorted((r["gate"], r["port"], r["metric"]) for r in rows)
    assert ("G2", None, "g2_scaling_ratio") in metrics
    assert sum(1 for m in metrics if m[0] == "G1") == 3 * len(FD)
    assert {r["gate"] for r in rows} == {"G1", "G2"}  # G3 is its own record, never mixed
    for row in rows:
        assert cap.validate_row(row) == []
        assert row["record_sha256"] == cap.file_sha256(out)
        assert row["category"] == "CANDIDATE" and row["protocol_id"] == ""
        prov = row["extra"]["provenance"]
        assert prov["topology_hash"] and prov["load"]["in_flight_total"] == 24
        assert prov["instrument"]["gate_sha256"] == cap.file_sha256(GATE)
        assert set(prov["window"]["gates"]) == {"g1", "g2"}
    g1 = next(r for r in rows if r["metric"] == "g1_s_over_q_median")
    assert "A/A noise floor" in g1["claim"]
    assert g1["value"] == pytest.approx(_record()["g1_frontdoor_decode"]["8070"]["s_over_q_median"])


@pytest.mark.parametrize("field,mutate", [
    ("value", lambda r: r.__setitem__("value", r["value"] + 0.01)),
    ("floor", lambda r: r["samples"].__setitem__("aa_noise_floor_rel_median", 0.0)),
    ("unit", lambda r: r.__setitem__("unit", "tok/s")),
    ("protocol", lambda r: r.__setitem__("protocol_id", "P-CPU-1")),
    ("cpuset", lambda r: r["extra"]["provenance"]["processes"]["8090"].__setitem__("cpus_allowed_list", "0-191")),
])
def test_validate_row_refuses_tampering_even_when_rehashed(tmp_path, field, mutate):
    _, sidecar = _write(tmp_path, _record())
    row = json.loads(next(line for line in sidecar.read_text().splitlines()
                          if '"g1_s_over_q_median"' in line))
    mutate(row)
    row["extra"]["provenance_sha256"] = cap.content_hash(row["extra"]["provenance"])
    row["row_sha256"] = cap.row_digest(row)
    assert cap.validate_row(row)


def test_refusals_write_nothing(tmp_path):
    with pytest.raises(cap.CaptureError, match="changed during the window"):
        _write(tmp_path, _record(), window=_window(_snapshots(drift=True)))
    with pytest.raises(cap.CaptureError, match="no process"):
        _write(tmp_path, _record(), window=_window(_snapshots(missing=8093)))
    with pytest.raises(cap.CaptureError, match="topology"):
        _write(tmp_path, _record(), window=_window(_snapshots(topo="")))
    with pytest.raises(cap.CaptureError, match="retrospective"):
        _write(tmp_path, _record(start=datetime(2026, 9, 26, 18, 0, tzinfo=UTC)))
    with pytest.raises(cap.CaptureError, match="backfill"):
        _write(tmp_path, _record(), emitted=_iso(T0 + timedelta(days=2)))
    assert not list(tmp_path.glob("*.belief_measurements.jsonl"))


def test_phase0_record_shape_is_refused(tmp_path):
    """The 2026-09-26 files (no params, pre-hook) produce zero rows."""
    rec = _record(label="post", load_mode="raw")
    rec.pop("params")
    rec["started_at"] = "2026-09-26T18:12:14+00:00"
    with pytest.raises(cap.CaptureError):
        _write(tmp_path, rec)


def test_g3_record_projects_separately(tmp_path):
    rec = {
        "schema": cap.G3_RECORD_SCHEMA, "gate": "G3", "label": "post-cap",
        "started_at": _iso(T0), "finished_at": _iso(T0 + timedelta(minutes=5)),
        "method": {"warmup_discarded": True, "reps": 5},
        "params": {"pool_ports": gate.POOL_PORTS, "per_port_in_flight": 4, "clip_sha256": "d" * 64},
        "quiet": {"stt_rtf": [0.2, 0.19], "stt_status": [200, 200],
                  "tts_first_packet_s": [0.09, 0.08], "tts_status": [200, 200]},
        "saturated": {"stt_rtf": [0.22, 0.24, 0.21], "stt_status": [200, 200, 200],
                      "tts_first_packet_s": [0.11, 0.12, 0.10], "tts_status": [200, 200, 200]},
        "verdict": "PASS",
    }
    ports = gate.POOL_PORTS + [9000, 9002]
    w = cap.CaptureWindow(ports, [], snapshot_fn=_snapshots())
    w.begin()
    w.finish()
    out = tmp_path / "g3-post-cap-20260927.json"
    out.write_text(json.dumps(rec))
    sidecar = cap.write_belief_measurements(out, rec, window=w, producer="test", gate_path=GATE,
                                            emitted_at=_iso(T0 + timedelta(minutes=6)))
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert {r["gate"] for r in rows} == {"G3"} and len(rows) == 4
    worst = next(r for r in rows if r["metric"] == "g3_saturated_stt_rtf_worst")
    assert worst["value"] == 0.24 and worst["metric_direction"] == "lower_better"
    assert all(cap.validate_row(r) == [] for r in rows)
    bad = copy.deepcopy(rec)
    bad["saturated"]["stt_status"][0] = 500
    out.write_text(json.dumps(bad))
    with pytest.raises(cap.CaptureError, match="non-200"):
        cap.write_belief_measurements(out, bad, window=w, producer="test", gate_path=GATE,
                                      emitted_at=_iso(T0 + timedelta(minutes=6)))


def test_gate_main_writes_record_then_sidecar(tmp_path, monkeypatch):
    """The gate's measurement path is untouched: g1/g2 are called as before, the record keeps its
    keys, and the capture runs only after the record is on disk."""
    base = _record(load_mode="raw")
    monkeypatch.setattr(gate, "_healthy", lambda port: True)
    monkeypatch.setattr(gate, "embedder_env_readback", lambda ports: {})  # no live /proc in unit tests
    monkeypatch.setattr(gate, "g2", lambda window_s, load_factory: base["g2_pool_scaling"])
    monkeypatch.setattr(gate, "g1", lambda fd, pairs, n, load_factory: base["g1_frontdoor_decode"])
    monkeypatch.setattr(gate, "_capture_window",
                        lambda fd: cap.CaptureWindow(gate.POOL_PORTS + fd, fd, snapshot_fn=_snapshots()))
    out = tmp_path / "post-20260927.json"
    monkeypatch.setattr(sys, "argv", ["gate", "--label", "post", "--out", str(out),
                                      "--fd-ports", "8070,8080", "--pairs", "3"])
    assert gate.main() == 0
    record = json.loads(out.read_text())
    assert {"g1_frontdoor_decode", "g2_pool_scaling", "load_mode", "params", "gate_windows"} <= set(record)
    rows = [json.loads(line) for line in cap.sidecar_path(out).read_text().splitlines()]
    assert rows and all(cap.validate_row(r) == [] for r in rows)
    assert rows[0]["record_sha256"] == cap.file_sha256(out)


def test_gate_main_refused_capture_keeps_record_and_exits_4(tmp_path, monkeypatch):
    base = _record(load_mode="raw")
    monkeypatch.setattr(gate, "_healthy", lambda port: True)
    monkeypatch.setattr(gate, "embedder_env_readback", lambda ports: {})  # no live /proc in unit tests
    monkeypatch.setattr(gate, "g2", lambda window_s, load_factory: base["g2_pool_scaling"])
    monkeypatch.setattr(gate, "g1", lambda fd, pairs, n, load_factory: base["g1_frontdoor_decode"])
    monkeypatch.setattr(gate, "_capture_window", lambda fd: cap.CaptureWindow(
        gate.POOL_PORTS + fd, fd, snapshot_fn=_snapshots(drift=True)))
    out = tmp_path / "post-20260927.json"
    monkeypatch.setattr(sys, "argv", ["gate", "--label", "post", "--out", str(out),
                                      "--fd-ports", "8070,8080"])
    assert gate.main() == gate.CAPTURE_EXIT
    assert out.is_file() and not cap.sidecar_path(out).exists()
