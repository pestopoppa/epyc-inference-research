#!/usr/bin/env python3
"""The compact terminal dashboard producer; no inference or process launch."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import campaign, dashboard, journal, schemas

NOW = "2026-08-05T08:00:00+00:00"
LATER = "2026-08-05T09:00:00+00:00"


def result(*, state="decided", error=None, keep=True) -> dict:
    return {
        "schema": "epyc.autokernel.campaign_result.v1",
        "state": state,
        "campaign_id": "ak-dashboard-test",
        "candidate_id": "akc-dashboard-test",
        "spec": {"backend": "llama_cpu", "recipe_id":
                 "t1b.llama_cpu.llama_bench_decode.v1"},
        "decision": None if keep is None else {"keep": keep, "reason": "fixture"},
        "preflight": None,
        "releases": [],
        "error": error,
        "executed": True,
    }


def entry(**over) -> journal.JournalEntry:
    fields = {
        "event_id": "akj-000000000007-deadbeef0000",
        "seq": 7,
        "kind": journal.KIND_STOP_STATE,
        "campaign_id": "ak-dashboard-test",
        "record_id": None,
        "written_at": NOW,
        "payload": {"state": "decided", "campaign_id": "ak-dashboard-test",
                    "result": result()},
    }
    fields.update(over)
    return journal.JournalEntry(**fields)


class TerminalContractTest(unittest.TestCase):
    def test_terminal_record_builds_the_existing_v2_contract(self):
        doc = dashboard.build_terminal_contract(entry(), exported_at=LATER)
        self.assertEqual(schemas.validate_kernel_dashboard_v2(doc), [])
        self.assertEqual(doc["produced_at"], NOW)
        self.assertEqual(doc["exported_at"], LATER)
        self.assertEqual(doc["producer"]["run"]["ledger_receipt"],
                         entry().event_id)
        self.assertEqual(doc["sections"]["backend_standing"]["backends"]
                         ["llama_cpu"]["standing"], "keep")
        self.assertTrue(doc["sections"]["campaign"]["stopped"])

    def test_reexporting_old_evidence_does_not_manufacture_freshness(self):
        doc = dashboard.build_terminal_contract(entry(), exported_at=LATER)
        self.assertEqual(doc["generated_at"], NOW)
        self.assertNotEqual(doc["generated_at"], doc["exported_at"])

    def test_unowned_planes_are_explicitly_unreported(self):
        doc = dashboard.build_terminal_contract(entry(), exported_at=LATER)
        self.assertEqual(doc["unreported_sections"],
                         ["champion", "headroom", "release_package"])
        for name in doc["unreported_sections"]:
            self.assertEqual(doc["sections"][name]["status"],
                             schemas.SECTION_NOT_REPORTED)
            self.assertTrue(doc["sections"][name]["reason"])

    def test_error_and_preflight_refusal_are_visible_blocking_conditions(self):
        error_result = result(state="error", error="boom", keep=None)
        error_entry = entry(payload={"state": "error", "campaign_id":
                           "ak-dashboard-test", "result": error_result})
        conditions = dashboard.build_terminal_contract(
            error_entry, exported_at=LATER)["sections"]["blocking_conditions"]["open"]
        self.assertEqual(conditions[0]["kind"], "CAMPAIGN_ERROR")
        self.assertIn("boom", conditions[0]["detail"])

        refused = result(state="preflight_refused", keep=None)
        refused["preflight"] = {"outcome": "FAIL", "reasons": ["co-tenant active"]}
        refused_entry = entry(payload={"state": "preflight_refused", "campaign_id":
                             "ak-dashboard-test", "result": refused})
        conditions = dashboard.build_terminal_contract(
            refused_entry, exported_at=LATER)["sections"]["blocking_conditions"]["open"]
        self.assertEqual(conditions[0]["kind"], "PREFLIGHT_REFUSED")
        self.assertIn("co-tenant", conditions[0]["detail"])

    def test_wrong_kind_and_campaign_mismatch_are_refused(self):
        with self.assertRaisesRegex(dashboard.DashboardError, "STOP_STATE"):
            dashboard.build_terminal_contract(entry(kind=journal.KIND_VIEW_REBASED))
        with self.assertRaisesRegex(dashboard.DashboardError, "identity disagrees"):
            dashboard.build_terminal_contract(entry(campaign_id="ak-other"))


class ExportTest(unittest.TestCase):
    def test_export_is_atomic_json_at_a_durable_non_checkout_path(self):
        with tempfile.TemporaryDirectory(dir="/mnt/raid0/llm") as td:
            path = Path(td) / "surface" / "kernel_dashboard.json"
            dashboard.export_terminal_entry(entry(), path=path, exported_at=LATER)
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(schemas.validate_kernel_dashboard_v2(doc), [])
            self.assertEqual(list(path.parent.glob("*.tmp.*")), [])

    def test_scratch_and_checkout_destinations_are_refused(self):
        with self.assertRaises(Exception):
            dashboard.export_terminal_entry(entry(), path="/tmp/kernel_dashboard.json")
        with self.assertRaisesRegex(dashboard.DashboardError, "checkout"):
            dashboard.export_terminal_entry(
                entry(), path=Path(__file__).parent / "kernel_dashboard.json")


class CampaignHookTest(unittest.TestCase):
    def test_export_failure_does_not_relabel_a_fsynced_terminal_record_as_lost(self):
        with tempfile.TemporaryDirectory(dir="/mnt/raid0/llm") as td:
            spec = campaign.CampaignSpec(
                campaign_id="ak-dashboard-test", candidate_id="akc-dashboard-test",
                candidate_ref="fixture", journal_root=td,
                model="/mnt/raid0/llm/models/dashboard-fixture.gguf")
            ops = campaign.HostOps()
            payload = {"state": "decided", "campaign_id": "ak-dashboard-test",
                       "result": result()}
            with mock.patch.object(dashboard, "export_terminal_entry",
                                   side_effect=dashboard.DashboardError("fixture")):
                event_id = ops.journal(spec, payload)
            events = journal.Journal(td, campaign_id=spec.campaign_id).read_all()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, event_id)
        self.assertEqual(events[0].kind, journal.KIND_STOP_STATE)


if __name__ == "__main__":
    unittest.main()
