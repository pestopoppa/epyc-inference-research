"""No-inference journal-to-sequencer banking adapter tests."""

from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

from .. import journal as J
from .. import schemas as S
from ..test_journal import _candidate, _event
from ..test_schemas import _campaign, _proposal
from . import champion
from . import completed_campaign_adapter as A
from . import hypotheses as H


class _EmptyDoNotRepeat:
    def matches_for(self, regime, statement):
        return ()


class CompletedCampaignAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = (Path(self.temp.name) / "live").resolve()
        self.campaign_id = "ak-iqk-adapter-20260812"
        self.proposal_id = "akp-20260812-7001"
        self.candidate_id = "akc-20260812-7001"
        proposal = _proposal()
        proposal.update({
            "campaign_id": self.campaign_id, "proposal_id": self.proposal_id,
            "campaign_kind": "config", "change_class": "parameter",
        })
        proposal["change"]["parameter_surface"] = {
            "candidate": {"ggml_iqk": "1"}, "anchor": {"ggml_iqk": "0"}}
        candidate = _candidate("7001", status="evaluating")
        candidate.update({
            "campaign_id": self.campaign_id, "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "evaluation_event_ids": ["ake-iqk-t0", "ake-iqk-t1"],
        })
        t0 = _event("7001")
        t0.update({
            "event_id": "ake-iqk-t0", "campaign_id": self.campaign_id,
            "candidate_id": self.candidate_id, "tier": "T0", "anchor_tier": "T0",
            "correctness": {"test_backend_ops": S.PASS},
            "stability": {"no_fallback_dispatch_trace": S.PASS},
            "mechanism": {},
        })
        t1 = copy.deepcopy(t0)
        t1.update({
            "event_id": "ake-iqk-t1", "tier": "T1", "anchor_tier": "T1",
            "mechanism": {"t1.parameter_intervention_explained": S.PASS},
        })
        t1["performance"]["search_discipline"] = {
            "speed_rank_admissible": True,
        }
        for event in (t0, t1):
            event["device_state"]["source"] = "rocm-smi"
            event["device_state"]["receipt_ref"] = "rcpt-device-state-adapter"
        self.campaign = _campaign()
        self.campaign.update({
            "campaign_id": self.campaign_id, "backend": "llama_cpu",
            "source_tree": "llama.cpp",
        })
        book = J.Journal(str(self.root), campaign_id=self.campaign_id)
        book.initialize()
        book.append(J.KIND_PROPOSAL_RECORDED, proposal)
        self.hypothesis_id = "akh-iqk-adapter-20260812"
        tracker = H.HypothesisTracker(
            journal_=book, root=str(self.root), campaign_id=self.campaign_id)
        tracker.open_hypothesis(H.Hypothesis(
            hypothesis_id=self.hypothesis_id,
            statement=proposal["hypothesis"],
            falsifier="The accepted paired run misses its effect floor.",
            origin=H.ORIGIN_CONTROLLER,
            author="completed-campaign-adapter-test",
            regime={"recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1"},
        ))
        authorization = tracker.authorize_claim(
            self.hypothesis_id,
            purpose="exercise completed-campaign sequencer admission",
            authorized_by="completed-campaign-adapter-test",
            ledger=_EmptyDoNotRepeat(),
        )
        book.append(J.KIND_EVALUATION_EVENT, t0)
        book.append(J.KIND_EVALUATION_EVENT, t1)
        book.append(J.KIND_CANDIDATE_RECORDED, candidate)
        self.terminal = book.append(J.KIND_STOP_STATE, {
            "state": "decided",
            "result": {
                "state": "decided", "campaign_id": self.campaign_id,
                "candidate_id": self.candidate_id, "executed": True, "ok": True,
                "spec": {
                    "recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1",
                    "hypothesis": {
                        "bound": True,
                        "hypothesis_id": self.hypothesis_id,
                        "authorization": authorization.to_dict(),
                    },
                    "proposal": {"proposal_id": self.proposal_id},
                    "calibration": {
                        "contribution_floor": 0.03, "mde": 0.027,
                    },
                },
                "decision": {"keep": True, "median_relative": 0.05},
                "production_unchanged": {"outcome": S.PASS},
                "releases": [{"claim": "cpu", "released": True}],
                "pairs": [{"block_index": 0, "candidate": 1.05, "anchor": 1.0}],
            },
        })

    def tearDown(self):
        self.temp.cleanup()

    def project(self):
        return A.project(
            campaign_record=self.campaign, journal_root=str(self.root),
            campaign_id=self.campaign_id, proposal_id=self.proposal_id,
            completion_event_id=self.terminal.event_id)

    def test_clean_keep_becomes_event_bound_banked_campaign_run(self):
        projected = self.project()
        candidate = projected.run.candidate_records[0]
        self.assertEqual(candidate["status"], "banked")
        verdict = champion.BankingVerdict.from_candidate(
            candidate, projected.run.evaluation_events)
        self.assertEqual(verdict.qualifying_axis, "throughput")

    def test_missing_mechanism_gate_cannot_become_a_champion_candidate(self):
        book = J.Journal(str(self.root), campaign_id=self.campaign_id)
        entries = book.read_all()
        # Immutable journal: build another journal with the T1 mechanism removed.
        other = self.root.parent / "no-mechanism"
        rewritten = J.Journal(str(other), campaign_id=self.campaign_id)
        rewritten.initialize()
        shutil.copy2(self.root / H.LEDGER_FILENAME, other / H.LEDGER_FILENAME)
        terminal = None
        for entry in entries:
            payload = copy.deepcopy(entry.payload)
            if entry.kind == J.KIND_EVALUATION_EVENT and payload.get("tier") == "T1":
                payload["mechanism"] = {}
            appended = rewritten.append(entry.kind, payload, record_id=entry.record_id)
            if entry.kind == J.KIND_STOP_STATE:
                terminal = appended
        assert terminal is not None
        with self.assertRaisesRegex(A.CompletedCampaignAdapterError, "mechanism"):
            A.project(
                campaign_record=self.campaign, journal_root=str(other),
                campaign_id=self.campaign_id, proposal_id=self.proposal_id,
                completion_event_id=terminal.event_id)

    def test_keep_with_an_unrankable_final_t1_cannot_become_banked(self):
        """The accept rule and final evaluator can diverge at window close.

        The fixture keeps every value the campaign rule consumed: it remains a
        terminal KEEP with passing T0/T1 gates and a gain above both MDE and
        floor.  Only the T1 evaluator's own final rank admission changes.
        Banking it would be a fail-open path around the evaluator.
        """
        book = J.Journal(str(self.root), campaign_id=self.campaign_id)
        entries = book.read_all()
        other = self.root.parent / "unrankable-final-t1"
        rewritten = J.Journal(str(other), campaign_id=self.campaign_id)
        rewritten.initialize()
        shutil.copy2(self.root / H.LEDGER_FILENAME, other / H.LEDGER_FILENAME)
        terminal = None
        for entry in entries:
            payload = copy.deepcopy(entry.payload)
            if entry.kind == J.KIND_EVALUATION_EVENT and payload.get("tier") == "T1":
                payload["performance"]["search_discipline"][
                    "speed_rank_admissible"] = False
            appended = rewritten.append(entry.kind, payload, record_id=entry.record_id)
            if entry.kind == J.KIND_STOP_STATE:
                terminal = appended
        assert terminal is not None
        with self.assertRaisesRegex(A.CompletedCampaignAdapterError,
                                    "not speed-rank-admissible"):
            A.project(
                campaign_record=self.campaign, journal_root=str(other),
                campaign_id=self.campaign_id, proposal_id=self.proposal_id,
                completion_event_id=terminal.event_id)


if __name__ == "__main__":
    unittest.main()
