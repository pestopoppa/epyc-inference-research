"""G1 design stub: the MI210 window handover is a design until the stack owner agrees."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from . import gpu_window as gw


def test_request_respects_the_stack_owner_constraints():
    request = gw.WindowRequest("ak-27b", "rocprof serving profile", 1800, schedule_ref="sched#4")
    payload = request.to_bus_payload()
    assert payload["needs_routing_to"] == ["workspace-ec"] and payload["requires_ack"] is True
    assert payload["payload"]["parked_ports"] == [8083]
    assert payload["payload"]["gpu_peak_ceiling_bytes"] == 62 << 30
    with pytest.raises(ValueError, match="operator"):
        gw.WindowRequest("ak", "p", 3601)
    with pytest.raises(ValueError, match="62 GiB"):
        gw.WindowRequest("ak", "p", 60, gpu_peak_ceiling_bytes=63 << 30)


def test_grant_is_read_from_the_window_file(tmp_path):
    path = tmp_path / "mi210.json"
    end = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    path.write_text(json.dumps({"holder": "autokernel", "expected_end": end, "parked_ports": [8083]}))
    assert gw.read_grant(path).active is True
    path.write_text(json.dumps({"holder": "production", "expected_end": end, "parked_ports": [8083]}))
    assert gw.read_grant(path).active is False
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    path.write_text(json.dumps({"holder": "autokernel", "expected_end": past, "parked_ports": [8083]}))
    assert gw.read_grant(path).active is False
    path.write_text("{garbled")
    assert gw.read_grant(path).active is False


def test_the_handshake_refuses_until_agreed():
    stub = gw.UnagreedStackOwnerWindow()
    with pytest.raises(gw.WindowNotAgreed, match="workspace-ec"):
        stub.request(gw.WindowRequest("ak", "p", 60))
