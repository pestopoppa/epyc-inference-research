"""G1: AK's half of the MI210 window contract (executor-owned window file, compute-request)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from . import gpu_window as gw

NOW = datetime(2026, 10, 5, 2, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.isoformat()


def _schedule(tmp_path, entries):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"schema": gw.SCHEDULE_SCHEMA, "entries": entries}))
    return path


def _entry(entry_id="sched-4", start=NOW, minutes=60, **extra):
    row = {"id": entry_id, "device_id": "mi210_0", "start": _iso(start),
           "end": _iso(start + timedelta(minutes=minutes)), "holder": "autokernel",
           "campaign_id": "ak-27b", "purpose": "rocprof", "operator_approved": False}
    row.update(extra)
    return row


def test_request_is_a_compute_request_to_the_coordinator_daemon(tmp_path):
    schedule = _schedule(tmp_path, [_entry()])
    request = gw.WindowRequest("ak-27b", "rocprof serving profile", 1800, schedule_ref="sched-4")
    msg = request.to_bus_payload(now=NOW, schedule_path=schedule)
    assert msg["kind"] == "compute-request" and msg["from"] == "inference"
    # coordinator-daemon has no roster row, so it is the transport `to`, never a
    # `needs_routing_to` entry (session_bus.py append refuses non-roster routing ids).
    assert msg["to"] == "coordinator-daemon" and "needs_routing_to" not in msg
    assert msg["requires_ack"] is True
    payload = msg["payload"]
    assert payload["device"] == "gpu" and payload["device_id"] == "mi210_0"
    assert payload["device_region"] == "gpu/mi210_0"
    assert isinstance(payload["window"], str) and "1800 s" in payload["window"]
    assert payload["window_detail"] == {"duration_s": 1800, "start": _iso(NOW),
                                        "expected_end": _iso(NOW + timedelta(seconds=1800))}
    assert payload["est_h"] == 0.5 and payload["release_condition"] and payload["task"]
    assert payload["parked_ports"] == [8083] and payload["parked_roles"] == ["architect_critic"]
    assert payload["gpu_peak_ceiling_bytes"] == 62 << 30
    assert payload["schedule_ref"] == "sched-4" and payload["campaign_id"] == "ak-27b"
    assert "workspace-ec" not in json.dumps(msg) and "gpu-window-request" not in json.dumps(msg)
    with pytest.raises(ValueError, match="operator"):
        gw.WindowRequest("ak", "p", 3601)
    with pytest.raises(ValueError, match="62 GiB"):
        gw.WindowRequest("ak", "p", 60, gpu_peak_ceiling_bytes=63 << 30)


def test_request_refuses_without_a_covering_schedule_entry(tmp_path):
    schedule = _schedule(tmp_path, [_entry(minutes=20)])
    with pytest.raises(gw.ScheduleInvalid, match="schedule_ref"):
        gw.WindowRequest("ak-27b", "p", 600).to_bus_payload(now=NOW, schedule_path=schedule)
    with pytest.raises(gw.ScheduleInvalid, match="not in"):
        gw.WindowRequest("ak-27b", "p", 600, schedule_ref="nope").to_bus_payload(
            now=NOW, schedule_path=schedule)
    with pytest.raises(gw.ScheduleInvalid, match="does not cover"):
        gw.WindowRequest("ak-27b", "p", 1800, schedule_ref="sched-4").to_bus_payload(
            now=NOW, schedule_path=schedule)
    with pytest.raises(gw.ScheduleInvalid, match="does not cover"):
        gw.WindowRequest("ak-27b", "p", 60, schedule_ref="sched-4").to_bus_payload(
            now=NOW - timedelta(minutes=5), schedule_path=schedule)
    with pytest.raises(gw.ScheduleInvalid, match="scheduled for"):
        gw.WindowRequest("other", "p", 60, schedule_ref="sched-4").to_bus_payload(
            now=NOW, schedule_path=schedule)
    with pytest.raises(gw.ScheduleInvalid, match="unreadable"):
        gw.WindowRequest("ak-27b", "p", 60, schedule_ref="sched-4").to_bus_payload(
            now=NOW, schedule_path=tmp_path / "missing.json")


def test_schedule_validator_rules(tmp_path):
    assert gw.validate_schedule({"schema": gw.SCHEDULE_SCHEMA, "entries": [_entry()]}) == []
    assert gw.validate_schedule({"schema": "x", "entries": []})
    dup = gw.validate_schedule({"schema": gw.SCHEDULE_SCHEMA,
                                "entries": [_entry(), _entry(start=NOW + timedelta(hours=2))]})
    assert any("duplicate" in e for e in dup)
    backwards = _entry()
    backwards["end"] = backwards["start"]
    assert any("after start" in e for e in gw.validate_schedule(
        {"schema": gw.SCHEDULE_SCHEMA, "entries": [backwards]}))
    long = gw.validate_schedule({"schema": gw.SCHEDULE_SCHEMA, "entries": [_entry(minutes=61)]})
    assert any("operator_approved" in e for e in long)
    assert gw.validate_schedule({"schema": gw.SCHEDULE_SCHEMA,
                                 "entries": [_entry(minutes=120, operator_approved=True)]}) == []
    overlap = gw.validate_schedule({"schema": gw.SCHEDULE_SCHEMA, "entries": [
        _entry("a"), _entry("b", start=NOW + timedelta(minutes=30))]})
    assert any("overlaps" in e for e in overlap)
    with pytest.raises(gw.ScheduleInvalid, match="overlaps"):
        gw.read_schedule(_schedule(tmp_path, [_entry("a"), _entry("b", start=NOW + timedelta(minutes=59))]))


def _window(**over):
    body = {"holder": "autokernel", "since": _iso(datetime.now(timezone.utc)),
            "expected_end": _iso(datetime.now(timezone.utc) + timedelta(minutes=30)),
            "preempt_requested_at": None, "preempt_reason": None,
            "parked_roles": ["architect_critic"], "parked_ports": [8083],
            "window_id": "w-1", "device_id": "mi210_0", "written_by": "gpu_window_executor",
            "grant_state": "granted", "schedule_ref": "sched-4"}
    body.update(over)
    return body


@pytest.mark.parametrize("override, reason", [
    ({}, None),
    ({"holder": "production"}, "holder"),
    ({"holder": "released"}, "holder"),
    ({"preempt_requested_at": "2026-10-04T12:00:00+00:00"}, "preempt"),
    ({"written_by": "stack_owner"}, "written by"),
    ({"window_id": None}, "window_id"),
    ({"device_id": "mi100_0"}, "mi210_0"),
    ({"grant_state": "draining"}, "grant_state"),
    ({"grant_state": "restoring"}, "grant_state"),
    ({"grant_state": None}, "grant_state"),
    ({"expected_end": _iso(datetime.now(timezone.utc) - timedelta(minutes=1))}, "expired"),
    ({"expected_end": None}, "expected_end"),
    ({"parked_ports": [8084]}, "parked ports"),
])
def test_grant_is_active_only_for_a_live_executor_window(tmp_path, override, reason):
    path = tmp_path / "mi210.json"
    path.write_text(json.dumps(_window(**override)))
    grant = gw.read_grant(path)
    assert grant.active is (reason is None)
    if reason is None:
        assert gw.assert_window_active(path).window_id == "w-1"
    else:
        assert reason in grant.inactive_reason()
        with pytest.raises(gw.GpuWindowPreempted, match=reason):
            gw.assert_window_active(path)


def test_missing_or_garbled_window_grants_nothing(tmp_path):
    path = tmp_path / "mi210.json"
    assert gw.read_grant(path).active is False
    path.write_text("{garbled")
    assert gw.read_grant(path).active is False
    path.write_text("[1, 2]")
    assert gw.read_grant(path).active is False


def test_ak_never_writes_the_window_file(tmp_path):
    path = tmp_path / "mi210.json"
    path.write_text(json.dumps(_window()))
    before = path.read_bytes()
    grant = gw.read_grant(path)
    msg = gw.release_message(grant, campaign_id="ak-27b")
    assert msg["payload"]["window_id"] == "w-1" and msg["to"] == "coordinator-daemon"
    assert path.read_bytes() == before
    with open(gw.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert "write_text" not in text and ".open(" not in text


def test_the_handshake_refuses_until_the_executor_ships():
    stub = gw.UnagreedStackOwnerWindow()
    with pytest.raises(gw.WindowNotAgreed, match="executor"):
        stub.request(gw.WindowRequest("ak", "p", 60))


def test_serial_router_refuses_a_gpu_batch_outside_a_granted_window(tmp_path):
    from . import serial_run
    path = tmp_path / "mi210.json"
    path.write_text(json.dumps(_window()))
    assert serial_run.gpu_batch_window_refusal(path) is None
    path.write_text(json.dumps(_window(grant_state="draining")))
    assert "grant_state" in serial_run.gpu_batch_window_refusal(path)
    path.write_text(json.dumps(_window(preempt_requested_at="2026-10-04T12:00:00Z")))
    assert "preempt" in serial_run.gpu_batch_window_refusal(path)
    import inspect
    params = inspect.signature(serial_run._drive).parameters
    assert params["gpu_window"].default == "off" and "gpu_window_path" in params
    source = open(serial_run.__file__, encoding="utf-8").read()
    assert "gpu_window=args.gpu_window, gpu_window_path=args.gpu_window_path" in source
    boundary = source.index('gpu_batch = (gpu_window == "required"')
    assert boundary < source.index("process = subprocess.Popen(", boundary)


def _fake_residency(monkeypatch, residency, *, device, own):
    monkeypatch.setattr(residency, "vram_bytes", lambda: device)
    monkeypatch.setattr(residency, "kfd_pid_vram", lambda _pid: own)
    monkeypatch.setattr(residency, "kfd_processes", lambda: 1)
    monkeypatch.setattr(residency, "sclk_mhz", lambda: 0)
    monkeypatch.setattr(residency, "pid_peak_rss", lambda _pid: None)


def test_residency_sampler_aborts_its_own_launch_above_the_62_gib_ceiling(monkeypatch):
    from . import residency
    _fake_residency(monkeypatch, residency, device=63 << 30, own=(62 << 30) + 1)
    killed = []
    sampler = residency.Sampler(interval=0.01)
    sampler.watch_pid(12345)
    sampler.abort_above(gw.GPU_PEAK_CEILING_BYTES, lambda: killed.append("terminate"))
    sampler._check_ceiling(residency.vram_bytes(), sampler._watch_once())
    sampler._check_ceiling(residency.vram_bytes(), sampler._watch_once())
    assert killed == ["terminate"]  # once, not per sample
    breach = sampler.proof["vram_ceiling_breach"]
    assert breach["source"] == "own_pid_kfd_vram" and breach["terminated"] is True
    with pytest.raises(residency.VramCeilingExceeded, match="62.00 GiB ceiling"):
        sampler.raise_if_ceiling_breached()


def test_residency_sampler_ceiling_is_inert_at_or_below_the_ceiling_and_when_unarmed(monkeypatch):
    from . import residency
    # Production co-resident: the device total is high, the watched launch's own VRAM is not.
    _fake_residency(monkeypatch, residency, device=63 << 30, own=62 << 30)
    sampler = residency.Sampler(interval=0.01)
    sampler.watch_pid(1)
    sampler.abort_above(gw.GPU_PEAK_CEILING_BYTES, lambda: pytest.fail("terminated"))
    sampler._check_ceiling(residency.vram_bytes(), sampler._watch_once())
    sampler.raise_if_ceiling_breached()
    assert "vram_ceiling_breach" not in sampler.proof
    unarmed = residency.Sampler(interval=0.01)
    unarmed._check_ceiling(70 << 30, None)
    assert unarmed.ceiling_breach is None


def test_residency_sampler_thread_runs_the_ceiling_hook(monkeypatch):
    import time as _time
    from . import residency
    _fake_residency(monkeypatch, residency, device=63 << 30, own=None)
    killed = []
    with residency.Sampler(interval=0.01) as sampler:
        sampler.abort_above(gw.GPU_PEAK_CEILING_BYTES, lambda: killed.append(1))
        deadline = _time.monotonic() + 5
        while not killed and _time.monotonic() < deadline:
            _time.sleep(0.01)
    assert killed == [1] and sampler.ceiling_breach["source"] == "device_vram"


def test_run_gpu_measurement_boundary_refuses_and_stops_on_a_preempted_window(tmp_path, capsys):
    from . import loop, run
    stops = []
    path = tmp_path / "mi210.json"
    path.write_text(json.dumps(_window(preempt_requested_at="2026-10-04T12:00:00Z")))
    before = path.read_bytes()
    with pytest.raises(loop.TailRefused, match="preempt"):
        with run._gpu_window_boundary(True, path, on_preempt=lambda: stops.append(1)):
            pytest.fail("measured in a preempted window")
    assert stops == [1]
    with run._gpu_window_boundary(False, path, on_preempt=lambda: pytest.fail("checked")):
        pass
    path.write_text(json.dumps(_window()))
    with run._gpu_window_boundary(True, path, on_preempt=lambda: pytest.fail("preempted")):
        pass
    assert json.loads(path.read_text())["holder"] == "autokernel" and before
    with pytest.raises(SystemExit):
        run.main(["--help"])
    assert "--gpu-window {off,required}" in " ".join(capsys.readouterr().out.split())
    source = open(run.__file__, encoding="utf-8").read()
    assert "quiet = _stacked(_gpu_window_boundary(" in source
