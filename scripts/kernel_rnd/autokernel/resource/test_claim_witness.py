#!/usr/bin/env python3
"""Unit tests for claim_witness.py — the device-claim / preflight / event seam.

NO GPU, NO inference, NO model, NO benchmark, and no process is started or
signalled. The "devices" are made-up ids whose lock files live in a per-test
temp directory, so nothing here touches the real `/mnt/raid0/llm/tmp` lock root.
Every claim acquired here is acquired and released by this process.

The suite is organised around the two ways the seam was broken:

  * a GPU-scoped preflight could not witness a claim that WAS held, because no
    conforming `GpuClaimReader` existed — and the obvious hand-rolled bridge
    produced a finding whose `whose` read the literal string "None";
  * an `evaluation_event.resource_claim_receipt` was an opaque string that
    nothing downstream could resolve, so an invented receipt and a real one were
    the same object to every reader.

And around the one shape that must never appear: an unreadable claim reported as
a free device.

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/resource/test_claim_witness.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/resource/test_claim_witness.py
    python3 scripts/kernel_rnd/autokernel/resource/test_claim_witness.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE, never by putting this directory on sys.path:
# `autokernel/resource/` would shadow the stdlib `resource` module for anything
# imported afterwards (AutoPilot scar item 12, §2.5 — ambient import identity).
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel.resource import claim_witness as CW  # noqa: E402
from autokernel.resource import device_claim as dc  # noqa: E402
from autokernel.resource import preflight as P  # noqa: E402

DEVICE = "testdev0"
OTHER_DEVICE = "testdev1"
CAMPAIGN = "ak-llama_gpu-decode-20260803"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


class _SeamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ak-claim-witness-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.lock_root = os.path.join(self.tmp, "lockroot")
        os.makedirs(self.lock_root, exist_ok=True)
        self.journal = dc.ClaimJournal(os.path.join(self.tmp, "claims.jsonl"))

    def acquire(self, device_id=DEVICE, *, campaign_id=CAMPAIGN, label=None,
                purpose="integration seam test"):
        claim = dc.acquire_device_claim(
            device_id, purpose=purpose, campaign_id=campaign_id,
            journal=self.journal, holder_label=label, lock_root=self.lock_root,
            timeout_s=5.0,
        )
        self.addCleanup(self._release_quietly, claim)
        return claim

    @staticmethod
    def _release_quietly(claim):
        try:
            if claim.held:
                claim.release()
        except dc.DeviceClaimError:
            pass


# =========================================================================
# 1. The reader conforms to preflight's contract
# =========================================================================

class WitnessReaderTest(_SeamTest):
    def test_a_free_device_yields_no_witness(self):
        self.assertEqual(
            CW.device_claim_witnesses([DEVICE], lock_root=self.lock_root), []
        )

    def test_a_held_device_yields_an_attributable_witness(self):
        claim = self.acquire(label="ak-evaluator")
        witnesses = CW.device_claim_witnesses([DEVICE], lock_root=self.lock_root)
        self.assertEqual(len(witnesses), 1)
        witness = witnesses[0]
        self.assertEqual(witness.device_id, DEVICE)
        self.assertEqual(witness.holder_pid, os.getpid())
        self.assertEqual(witness.holder_label, "ak-evaluator")
        self.assertEqual(witness.source, dc.RECEIPT_SCHEMA)
        self.assertEqual(witness.acquired_at, claim.receipt().acquired_at)

    def test_an_unlabelled_claim_still_gets_a_non_empty_label(self):
        """The defect the bridge exists to prevent, at its source.

        `ClaimReceipt.holder_label` is Optional; `GpuClaimWitness.holder_label`
        is not. The hand-rolled bridge passed the None straight through and a
        FAIL finding's `whose` rendered as "None (pid ..., via ...)".
        """
        self.acquire(label=None, purpose="unlabelled hold")
        witness = CW.device_claim_witnesses([DEVICE], lock_root=self.lock_root)[0]
        self.assertIsInstance(witness.holder_label, str)
        self.assertTrue(witness.holder_label.strip())
        self.assertIn(CAMPAIGN, witness.holder_label)

    def test_the_witness_type_refuses_a_none_label_outright(self):
        with self.assertRaises(ValueError):
            P.GpuClaimWitness(device_id=DEVICE, holder_pid=1234,
                              holder_label=None, source="x")
        with self.assertRaises(ValueError):
            P.GpuClaimWitness(device_id=DEVICE, holder_pid=1234,
                              holder_label="   ", source="x")

    def test_an_empty_device_list_is_refused_not_silently_empty(self):
        with self.assertRaises(ValueError):
            CW.device_claim_witnesses([], lock_root=self.lock_root)

    def test_a_malformed_device_id_raises_rather_than_reading_as_unclaimed(self):
        with self.assertRaises(ValueError):
            CW.device_claim_witnesses(["not/a/device"], lock_root=self.lock_root)

    def test_an_unreadable_claim_is_unavailable_never_free(self):
        """Silence is not freedom — the `gpu_idle()` failure, structurally."""
        self.acquire()
        lock_path = dc.device_lock_path(DEVICE, self.lock_root)
        with open(lock_path, "w", encoding="utf-8") as fh:
            fh.write("{ this is not json\n")
        with self.assertRaises(P.PreflightUnavailable):
            CW.device_claim_witnesses([DEVICE], lock_root=self.lock_root)

    def test_a_stale_dead_holders_claim_is_still_a_witness(self):
        """A dead holder's claim occupies the device until it is reclaimed."""
        self.acquire()
        payload = json.loads(
            dc.device_lock_path(DEVICE, self.lock_root).read_text(encoding="utf-8")
        )
        # Rewrite the holder as a long-dead pid on this boot; the flock is still
        # ours, so `inspect_device_claim` reports it held rather than stale — the
        # assertion that matters is that it is NOT reported free.
        payload["holder"]["pid"] = 999_999_999
        with open(dc.device_lock_path(DEVICE, self.lock_root), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        witnesses = CW.device_claim_witnesses([DEVICE], lock_root=self.lock_root)
        self.assertEqual([w.device_id for w in witnesses], [DEVICE])
        self.assertEqual(witnesses[0].holder_pid, 999_999_999)


# =========================================================================
# 2. The bridge, through the real preflight
# =========================================================================

class PreflightThroughTheBridgeTest(_SeamTest):
    def test_gpu_scope_without_a_reader_is_could_not_check(self):
        scope = P.PreflightScope.gpu("gpu-bench", [DEVICE])
        sources = P.ClaimSources(region_lock_dir=Path(self.lock_root))
        result = P.claim_witness_preflight(scope, sources)
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertTrue(any("device-claim reader" in r for r in result.reasons))

    def test_our_own_claim_passes_through_the_real_bridge(self):
        self.acquire(label="ak-evaluator")
        scope = P.PreflightScope.gpu("gpu-bench", [DEVICE])
        sources = CW.gpu_claim_sources([DEVICE], lock_root=self.lock_root)
        result = P.claim_witness_preflight(scope, sources)
        self.assertEqual(result.verdict, P.PASS)

    def test_a_free_device_passes(self):
        scope = P.PreflightScope.gpu("gpu-bench", [DEVICE])
        sources = CW.gpu_claim_sources([DEVICE], lock_root=self.lock_root)
        self.assertEqual(
            P.claim_witness_preflight(scope, sources).verdict, P.PASS
        )

    def test_a_foreign_claim_fails_with_an_attributable_finding(self):
        self.acquire(label="another-session")
        scope = P.PreflightScope.gpu("gpu-bench", [DEVICE])
        # An owned scope that does NOT contain us models a foreign holder
        # without spawning a second process.
        owned = P.OwnedScope(self_pid=os.getpid(), cgroup=None,
                             pids=frozenset({-1}), reasons={})
        sources = CW.gpu_claim_sources([DEVICE], lock_root=self.lock_root)
        result = P.claim_witness_preflight(scope, sources, owned=owned)
        self.assertEqual(result.verdict, P.FAIL)
        self.assertEqual(len(result.findings), 1)
        self.assertIn(DEVICE, result.findings[0].what)
        self.assertIn("another-session", result.findings[0].whose)
        self.assertNotIn("None", result.findings[0].whose)

    def test_an_unreadable_claim_downgrades_the_verdict_not_the_opposite(self):
        self.acquire()
        with open(dc.device_lock_path(DEVICE, self.lock_root), "w",
                  encoding="utf-8") as fh:
            fh.write("garbage\n")
        scope = P.PreflightScope.gpu("gpu-bench", [DEVICE])
        sources = CW.gpu_claim_sources([DEVICE], lock_root=self.lock_root)
        result = P.claim_witness_preflight(scope, sources)
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertFalse(result.passed)

    def test_gpu_claim_sources_keeps_one_lock_root_for_both_planes(self):
        """Two roots is how two repositories stop excluding each other."""
        sources = CW.gpu_claim_sources([DEVICE], lock_root=self.lock_root)
        self.assertEqual(str(sources.region_lock_dir), self.lock_root)

    def test_the_attestation_is_journalable_verbatim(self):
        """`require_no_concurrent_inference` says to journal `exc.result`."""
        from autokernel import journal as J  # noqa: PLC0415

        root = os.path.join(self.tmp, "journal")
        jr = J.Journal(root, campaign_id=CAMPAIGN)
        jr.initialize()
        scope = P.PreflightScope.gpu("gpu-bench", [DEVICE])
        result = P.claim_witness_preflight(
            scope, P.ClaimSources(region_lock_dir=Path(self.lock_root))
        )
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        entry = jr.append_preflight_attestation(result)
        self.assertEqual(entry.kind, J.KIND_PREFLIGHT_ATTESTATION)
        self.assertEqual(entry.payload["verdict"], P.COULD_NOT_CHECK)
        # canonical_json is what the journal hashed it as; it must round-trip.
        self.assertEqual(
            json.loads(S.canonical_json(entry.payload))["scope"]["label"],
            "gpu-bench",
        )


# =========================================================================
# 3. Resolving an evaluation event's receipt
# =========================================================================

def _event(receipt_id: str, *, campaign_id: str = CAMPAIGN) -> dict:
    return {
        "schema": S.SCHEMA_EVALUATION_EVENT,
        "event_id": "ake-20260803-0001",
        "campaign_id": campaign_id,
        "candidate_id": "akc-20260803-0001",
        "tier": "T1",
        "claim_grammar": {
            "category": "CANDIDATE",
            "protocol_id": "P-AK-SEARCH-1/v1",
            "metric": "decode_tokens_per_s",
            "metric_direction": "higher_better",
            "reps": 5,
            "attestation_ref": "rcpt-host-20260803T101500Z",
        },
        "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": _sha("bundle")},
        "artifact": {
            "source_sha256": _sha("snapshot"),
            "binary_sha256": _sha("candidate-binary"),
            "linkage_sha256": _sha("candidate-linkage"),
        },
        "anchor": {
            "source_commit": "67a433bf45a8a091d83b4ea0b32ff0735fd51800",
            "binary_sha256": _sha("anchor-binary"),
            "linkage_sha256": _sha("anchor-linkage"),
            "measurement_event_ids": ["ake-20260801-0009"],
        },
        "scope_manifest_sha256": _sha("scope-manifest"),
        "host_receipt": "rcpt-host-20260803T101500Z",
        "resource_claim_receipt": receipt_id,
        "co_residency": "single",
        "correctness": {"test_backend_ops": "pass"},
        "quality": {},
        "stability": {},
        "scope_denominator": {
            "machine_subset": "partial",
            "numa_nodes": [0],
            "devices": ["gfx90a:0"],
            "cores": 8,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "performance": {
            "raw_samples": [51.2, 51.4, 51.1],
            "paired_blocks": 3,
            "estimate": 51.23,
            "uncertainty": {"e_process_value": 12.4},
        },
        "mechanism": {},
        "integrity_flags": [],
        "status": "pass",
        "supersedes": [],
        "created_at": "2026-08-03T10:45:00+00:00",
    }


class ReceiptResolutionTest(_SeamTest):
    def test_a_real_receipt_resolves_and_passes(self):
        claim = self.acquire()
        event = _event(claim.claim_id)
        self.assertEqual(S.validate_evaluation_event(event), [])
        check = CW.check_event_claim_receipt(event, self.journal)
        self.assertEqual(check.outcome, S.PASS)
        receipt = CW.resolve_claim_receipt(claim.claim_id, self.journal)
        self.assertEqual(receipt.device_id, DEVICE)
        self.assertEqual(receipt.campaign_id, CAMPAIGN)

    def test_an_invented_receipt_is_schema_valid_and_still_fails(self):
        """The whole point: the schema cannot tell these two apart.

        `validate_evaluation_event` requires a non-empty string, and an invented
        receipt is a non-empty string. Only the claim record can say which one
        names an exclusivity that actually happened.
        """
        self.acquire()
        event = _event("akd-0000000000000000")
        self.assertEqual(S.validate_evaluation_event(event), [])
        check = CW.check_event_claim_receipt(event, self.journal)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("names no acquisition", check.reasons[0])

    def test_a_receipt_from_another_campaign_fails(self):
        claim = self.acquire(campaign_id="ak-someone-elses-20260801")
        event = _event(claim.claim_id, campaign_id=CAMPAIGN)
        check = CW.check_event_claim_receipt(event, self.journal)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("ak-someone-elses-20260801", check.reasons[0])

    def test_an_unreadable_claim_journal_is_could_not_check(self):
        claim = self.acquire()
        with open(self.journal.path, "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        check = CW.check_event_claim_receipt(_event(claim.claim_id), self.journal)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertNotEqual(check.outcome, S.FAIL)

    def test_a_missing_receipt_field_is_could_not_check_not_fail(self):
        event = _event("x")
        del event["resource_claim_receipt"]
        check = CW.check_event_claim_receipt(event, self.journal)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_resolution_refuses_a_sink_without_read_all(self):
        with self.assertRaises(TypeError):
            CW.resolve_claim_receipt("akd-1", object())

    def test_resolution_of_an_unknown_id_is_none_not_an_exception(self):
        self.acquire()
        self.assertIsNone(
            CW.resolve_claim_receipt("akd-ffffffffffffffff", self.journal)
        )

    def test_a_truncated_receipt_record_raises_rather_than_resolving(self):
        """A receipt that round-trips into a DIFFERENT receipt is not a receipt."""
        claim = self.acquire()
        lines = self.journal.path.read_text(encoding="utf-8").splitlines()
        rewritten = []
        for line in lines:
            record = json.loads(line)
            if record.get("kind") == dc.KIND_ACQUIRED:
                record["detail"]["receipt"].pop("device_id")
            rewritten.append(json.dumps(record))
        self.journal.path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            CW.resolve_claim_receipt(claim.claim_id, self.journal)
        # And the three-outcome wrapper reports that as COULD_NOT_CHECK, never
        # as a receipt that failed.
        check = CW.check_event_claim_receipt(_event(claim.claim_id), self.journal)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
