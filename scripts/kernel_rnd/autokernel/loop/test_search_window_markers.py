"""Versioned window transport; synthetic markers, no scientific PASS fixture."""
from __future__ import annotations

import pytest

from . import unified_worker as uw
from . import test_native_parent_integration as parent_fixtures

# Pytest discovers these original fixtures; no receipt cache is seeded here.
phase_invocation = parent_fixtures.phase_invocation
issued = parent_fixtures.issued


def _marker(packet, name, offset=0.0):
    body = {key: value for key, value in packet.items() if key != "request_digest"}
    body.update(schema=(uw.OBSERVATION_PHASE_REQUEST_SCHEMA if name == "health"
                        else uw.OBSERVATION_PHASE_REQUEST_SCHEMA_V2), phase=name,
                boundary_monotonic_s=packet["boundary_monotonic_s"] + offset)
    return body | {"request_digest": uw._digest(body)}


def _accept(invocation, authority, packet):
    invocation.handle_observation_phase(packet)
    notice = authority.next_notice(timeout=0.1)
    authority.publish_observation_phase(notice["key"], {"outcome": "unavailable"})
    invocation.poll_evidence()


def test_closed_markers_and_legacy_health_retry_keep_exact_cache(phase_invocation):
    invocation, authority, packet, _case = phase_invocation
    for offset, name in enumerate(uw.OBSERVATION_WINDOW_MARKERS):
        _accept(invocation, authority, _marker(packet, name, offset))
    original_keys = authority._retained_keys()
    invocation.handle_observation_phase(packet)
    assert authority._retained_keys() == original_keys
    assert tuple(invocation._active_phase_requests) == uw.OBSERVATION_WINDOW_MARKERS
    assert len(original_keys) == len(uw.OBSERVATION_WINDOW_MARKERS)


@pytest.mark.parametrize("name", ["unknown", "setup", "load", "placement", "", None])
def test_unknown_v2_marker_is_refused(phase_invocation, name):
    invocation, _authority, packet, _case = phase_invocation
    with pytest.raises(uw.WorkerBridgeRefused):
        invocation.handle_observation_phase(_marker(packet, name))


@pytest.mark.parametrize("name", ["warmup", "measurement", "measurement_end"])
def test_marker_without_prior_boundary_refused(phase_invocation, name):
    invocation, _authority, packet, _case = phase_invocation
    with pytest.raises(uw.WorkerBridgeRefused, match="prior boundary"):
        invocation.handle_observation_phase(_marker(packet, name))


def test_conflicting_retry_and_backwards_time_refused(phase_invocation):
    invocation, authority, packet, _case = phase_invocation
    _accept(invocation, authority, packet)
    with pytest.raises(uw.WorkerBridgeRefused, match="retry conflicts"):
        invocation.handle_observation_phase(_marker(packet, "health", 0.1))
    with pytest.raises(uw.WorkerBridgeRefused, match="backwards"):
        invocation.handle_observation_phase(_marker(packet, "warmup", -0.1))


def test_exception_teardown_can_arrive_without_scientific_window(phase_invocation):
    invocation, authority, packet, _case = phase_invocation
    _accept(invocation, authority, _marker(packet, "teardown"))
    assert tuple(invocation._active_phase_requests) == ("teardown",)

