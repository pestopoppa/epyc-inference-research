#!/usr/bin/env python3
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from .. import storage
from . import preflight as P


NOW = "2026-08-12T12:00:00Z"
DEADLINE = "2026-08-13T12:00:00Z"


def host(**overrides) -> P.HostHealth:
    fields = {
        "uptime_seconds": 3600,
        "observed_at": NOW,
        "receipt": "host-receipt",
    }
    fields.update(overrides)
    return P.HostHealth(**fields)


def storage_observation(*, free: int, floor: int,
                        pressured: bool, backlog: int = 0) -> P.StorageObservation:
    state = storage.StorageState(
        state=storage.DISK_PRESSURE if pressured else storage.STORAGE_OK,
        free_bytes=free, total_bytes=max(free, floor) + floor,
        floor_bytes=floor)
    return P.StorageObservation(
        path="/mnt/raid0", state=state,
        expirable_backlog_bytes=backlog, receipt="storage-receipt")


class ReleaseLocalPreflightTests(unittest.TestCase):
    def test_host_below_ceiling_continues(self):
        decision = P.guard_host_uptime(
            host(), owner="operator", escalation_deadline=DEADLINE, now=NOW)
        self.assertEqual(decision.outcome, P.CONTINUE)

    def test_host_at_ceiling_stops_and_cannot_loosen_the_ceiling(self):
        decision = P.guard_host_uptime(
            host(uptime_seconds=P.HOST_UPTIME_CEILING_SECONDS),
            owner="operator", escalation_deadline=DEADLINE, now=NOW)
        self.assertEqual(decision.outcome, P.STOP)
        with self.assertRaises(P.PreflightInputError):
            host(ceiling_seconds=P.HOST_UPTIME_CEILING_SECONDS + 1)

    def test_unobservable_host_is_not_a_pass(self):
        decision = P.guard_host_uptime(
            host(observable=False), owner="operator",
            escalation_deadline=DEADLINE, now=NOW)
        self.assertEqual(decision.outcome, P.COULD_NOT_EVALUATE)

    def test_resource_requires_an_acquired_claim_receipt(self):
        held = P.ResourceClaimObservation(
            resource="llama_gpu", claim_kind="gpu_device", acquired=True,
            observed_at=NOW, receipt="claim-mi210", held_by="akt3-v9")
        self.assertEqual(P.guard_resource_available(held).outcome, P.CONTINUE)
        missing = P.ResourceClaimObservation(
            resource="llama_gpu", claim_kind="gpu_device", acquired=False,
            observed_at=NOW, unavailable_reason="claim held by another session")
        self.assertEqual(P.guard_resource_available(missing).outcome, P.STOP)

    def test_storage_pressure_never_reclaims_or_continues(self):
        reclaimable = storage_observation(
            free=40, floor=50, pressured=True, backlog=20)
        decision = P.guard_storage_headroom(reclaimable)
        self.assertEqual(decision.outcome, P.STOP)
        self.assertIn("cannot perform reclamation", decision.reason)

    def test_consistent_storage_headroom_continues(self):
        decision = P.guard_storage_headroom(
            storage_observation(free=60, floor=50, pressured=False))
        self.assertEqual(decision.outcome, P.CONTINUE)

    def test_module_has_no_observation_process_or_write_capability(self):
        tree = ast.parse(Path(P.__file__).read_text(encoding="utf-8"))
        imports = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        self.assertTrue(imports.isdisjoint(
            {"os", "subprocess", "multiprocessing", "signal", "socket"}))
        self.assertTrue(calls.isdisjoint({
            "open", "write", "write_text", "write_bytes", "unlink", "remove",
            "rename", "Popen", "run", "system", "kill", "send_signal",
        }))


if __name__ == "__main__":
    unittest.main()
