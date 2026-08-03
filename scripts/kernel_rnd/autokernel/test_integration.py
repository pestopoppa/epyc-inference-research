#!/usr/bin/env python3
"""End-to-end integration test for the AutoKernel substrate (AK1 + AK2).

WHY THIS FILE EXISTS
--------------------
`schemas.py`, `journal.py`, `storage.py`, `resource/device_claim.py` and
`resource/preflight.py` were each written and red-teamed on their own, and each
one passes its own suite. Every defect this file was written to catch lived
*between* two of them, where each module was individually correct and the two
descriptions of the same object did not match:

  * `storage.expire_artifact` writes its tombstone THROUGH `journal.Journal`.
    The two modules disagreed about what identifies a reclamation — storage's
    `tombstone_id` covers (campaign, path, sha256, kind, rule); the journal's
    receipt view was keyed by the content hash ALONE. Two byte-identical build
    trees at two paths were two reclamations there and ONE receipt here, and
    `check_view_consistency` returned PASS because it recounted by the same key.
  * `preflight.require_no_concurrent_inference` hands its attestation back on
    the exception so the caller can journal it — invariant 7 — and the journal's
    closed kind vocabulary had no kind to journal it AS.
  * `preflight` needs a `GpuClaimReader`; `device_claim` produces `ClaimReceipt`
    objects; nothing bridged them, so every GPU-scoped preflight was
    COULD_NOT_CHECK even with the claim held.
  * `evaluation_event.resource_claim_receipt` is an opaque string that
    `schemas.py` can only check for being non-empty. An invented receipt and a
    real one were indistinguishable to every reader.
  * `storage.py` bound a SECOND copy of `schemas.py` via a `sys.path` insert, so
    `storage.Check is schemas.Check` was False across the seam and `import
    resource` resolved to this package instead of the stdlib.

WHAT THIS SUITE DOES NOT DO
---------------------------
NO inference, NO server, NO model, NO GPU, NO benchmark, NO sqlite. No process
is started, stopped, or signalled — the only claim acquired is acquired and
released by this process, in a temp directory, on a made-up device id. Every
path it writes is under a `tempfile.mkdtemp()` tree it removes again.

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/test_integration.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/test_integration.py
    python3 scripts/kernel_rnd/autokernel/test_integration.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Import through the PACKAGE. Every module is bound exactly once, so the
# `schemas` the journal validates with IS the `schemas` storage checks against —
# which is the property half this file exists to assert.
_KERNEL_RND = str(Path(__file__).resolve().parents[1])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel import storage as ST  # noqa: E402
from autokernel.resource import claim_witness as CW  # noqa: E402
from autokernel.resource import device_claim as DC  # noqa: E402
from autokernel.resource import preflight as PF  # noqa: E402

CAMPAIGN = "ak-llama_gpu-decode-20260803"
DEVICE = "akintegdev0"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


# =============================================================================
# §7 record fixtures — minimal records that schemas.py accepts
# =============================================================================

def _campaign() -> dict:
    return {
        "schema": S.SCHEMA_CAMPAIGN,
        "campaign_id": CAMPAIGN,
        "backend": "llama_gpu",
        "source_tree": "llama.cpp",
        "production_anchor": {
            "repo": "/mnt/raid0/llm/llama.cpp",
            "branch": "production-consolidated-v8",
            "commit": V8_COMMIT,
        },
        "objective": {
            "rule": "per_phase_non_inferiority_plus_improvement",
            "phases": ["prefill", "decode"],
            "protocol_by_phase": {"prefill": "P-BENCH-PREFILL-1", "decode": "P-BENCH-1"},
            "recipe_class": "production_optimal",
            "phase_trade_exception": None,
            "target_regimes": [],
        },
        "scope": {
            "affected_ops": [],
            "affected_arch_classes": [],
            "derived_role_manifest_sha256": _sha("role-manifest"),
        },
        "policy_ref": {
            "search_protocol": "P-AK-SEARCH-1/v1",
            "release_protocol": "P-KERNEL-FREEZE-1/v1",
            "policy_bundle_sha256": _sha("policy-bundle"),
        },
        "budgets": {
            "max_wall_hours": 48.0,
            "max_gpu_hours": 12.0,
            "max_cpu_region_hours": 0.0,
            "max_candidates": 40,
            "max_controller_tokens": 4_000_000,
            "max_storage_gb": 60.0,
        },
        "readiness_reporting": {"reference_point_gain": 0.25, "reference_lcb_gain": 0.20},
        "stop_policy": {
            "plateau_rounds": 6,
            "max_consecutive_integrity_failures": 2,
            "max_consecutive_build_failures": 3,
            "max_command_retries": 3,
        },
    }


def _candidate(suffix: str, *, status: str, build_dir: str,
               receipt_id: str) -> dict:
    return {
        "schema": S.SCHEMA_CANDIDATE,
        "candidate_id": f"akc-20260803-{suffix}",
        "campaign_id": CAMPAIGN,
        "proposal_id": "akp-20260803-0001",
        "parent_candidate_id": None,
        "worktree": {
            "path": f"/mnt/raid0/llm/llama.cpp-{CAMPAIGN}",
            "branch": f"ak/{CAMPAIGN}/akp-{suffix}",
            "source_commit": V7_COMMIT,
            "clean": True,
        },
        "source_snapshot": {
            "snapshot_sha256": _sha(f"snapshot-{suffix}"),
            "patch_bundle_sha256": _sha(f"patch-{suffix}"),
        },
        "ancestry": {
            "production_base_commit": V8_COMMIT,
            "is_descendant_of_production_base": True,
            "proof": "git merge-base --is-ancestor 67a433bf.. HEAD -> 0",
        },
        "build": {
            "toolchain": "rocm-6.2",
            "compiler": "hipcc 6.2.0",
            "command": "cmake --build build -j 96",
            "build_dir": build_dir,
            "log_path": f"data/{CAMPAIGN}/build/akc-{suffix}.log",
            "log_sha256": _sha(f"build-log-{suffix}"),
        },
        "artifacts": {
            "binary_sha256": _sha(f"binary-{suffix}"),
            "linkage_sha256": _sha(f"linkage-{suffix}"),
            "library_sha256s": {"libggml.so": _sha("libggml")},
        },
        "dispatch": {"feature_flags": ["GGML_AK_WIDE_TILE"],
                     "dispatch_predicate": "K >= 4096"},
        "affected_surface": {
            "derived_sha256": _sha("derived-surface"),
            "traced_sha256": None,
            "reconciled": False,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "evaluator": {"id": "P-AK-SEARCH-1/v1",
                      "bundle_sha256": _sha("evaluator-bundle")},
        "receipts": {
            "host_receipt": "rcpt-host-20260803T101500Z",
            "resource_claim_receipt": receipt_id,
        },
        "storage": {"footprint_gb": 3.4,
                    "durability_class": "hash_and_provenance_only"},
        "evaluation_event_ids": [],
        "derived_verdicts": {},
        "controller": {
            "provider": "local",
            "model_id": "architect-a4",
            "effort": "high",
            "prompt_bundle_sha256": _sha("prompt-bundle"),
        },
        "champion_status": "none",
        "status": status,
        "supersession_reason": None,
        "created_at": "2026-08-03T10:15:00+00:00",
    }


def _event(suffix: str, candidate_suffix: str, receipt_id: str, *,
           narrative: str) -> dict:
    return {
        "schema": S.SCHEMA_EVALUATION_EVENT,
        "event_id": f"ake-20260803-{suffix}",
        "campaign_id": CAMPAIGN,
        "candidate_id": f"akc-20260803-{candidate_suffix}",
        "tier": "T1",
        "claim_grammar": {
            "category": "CANDIDATE",
            "protocol_id": "P-AK-SEARCH-1/v1",
            "metric": "decode_tokens_per_s",
            "metric_direction": "higher_better",
            "reps": 5,
            "attestation_ref": "rcpt-host-20260803T101500Z",
        },
        "evaluator": {"id": "P-AK-SEARCH-1/v1",
                      "bundle_sha256": _sha("evaluator-bundle")},
        "artifact": {
            "source_sha256": _sha(f"snapshot-{candidate_suffix}"),
            "binary_sha256": _sha(f"binary-{candidate_suffix}"),
            "linkage_sha256": _sha(f"linkage-{candidate_suffix}"),
        },
        "anchor": {
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
        "narrative": narrative,
        "narrative_retrievable": False,
        "created_at": "2026-08-03T10:45:00+00:00",
    }


# =============================================================================
# The scenario
# =============================================================================

class AutoKernelSubstrateEndToEndTest(unittest.TestCase):
    """One campaign, start to reclamation, across all five modules.

    `setUpClass` runs the scenario ONCE and the tests assert against its
    results. That ordering is deliberate: the scenario is the fixture, and each
    test names one property of it, so a failure names the property that broke
    rather than "the integration test failed".
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        # Not under /tmp: /tmp is in `storage.EPHEMERAL_ROOTS`, so every guard
        # this scenario exercises would refuse a fixture built there and the
        # "normal path" assertions would pass for the wrong reason.
        cls.tmp = tempfile.mkdtemp(prefix="ak-integration-",
                                   dir=os.path.dirname(os.path.abspath(__file__)))
        try:
            cls._run_scenario()
        except BaseException:
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _run_scenario(cls):
        tmp = cls.tmp
        cls.lock_root = os.path.join(tmp, "lockroot")
        os.makedirs(cls.lock_root, exist_ok=True)
        repo_root = os.path.join(tmp, "repo")
        os.makedirs(repo_root, exist_ok=True)

        # --- 1. campaign evidence root (MEASUREMENT.md:146-156) --------------
        cls.evidence = ST.ensure_campaign_evidence_root(
            CAMPAIGN, repo_root=repo_root)
        cls.stub_layout = ST.check_evidence_root_layout(cls.evidence.path)
        # A human fills in what/when/which-claim; the stub must NOT pass until
        # then, so the scenario writes a real README to reach a PASS.
        with open(cls.evidence.readme_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"# {CAMPAIGN}\n\n"
                "| | |\n|---|---|\n"
                "| what was measured | llama_gpu decode tokens/s, np=1 |\n"
                "| when | 2026-08-03T10:45:00Z |\n"
                "| which claim it backs | AK integration scenario |\n"
            )
        with open(cls.evidence.sha256sums_path, "w", encoding="utf-8") as fh:
            fh.write(f"{_sha('artifact')}  raw_samples.json\n")
        with open(os.path.join(cls.evidence.path, "raw_samples.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"raw_samples": [51.2, 51.4, 51.1]}, fh)
        cls.filled_layout = ST.check_evidence_root_layout(cls.evidence.path)

        # --- 2. journals ------------------------------------------------------
        cls.journal = J.Journal(os.path.join(tmp, "journal"), campaign_id=CAMPAIGN)
        cls.journal.initialize()
        cls.claim_journal = DC.ClaimJournal(os.path.join(tmp, "claims.jsonl"))

        # --- 3. acquire the device claim (invariant 9: acquired, not observed)
        claim = DC.acquire_device_claim(
            DEVICE, purpose="AK integration scenario", campaign_id=CAMPAIGN,
            journal=cls.claim_journal, holder_label="ak-integration",
            lock_root=cls.lock_root, timeout_s=5.0,
        )
        cls.receipt_id = claim.claim_id

        # --- 4. preflight, WITH the claim held --------------------------------
        scope = PF.PreflightScope.gpu("ak-integration-decode", [DEVICE])
        sources = CW.gpu_claim_sources([DEVICE], lock_root=cls.lock_root)
        cls.preflight_result = PF.require_no_concurrent_inference(scope, sources)
        cls.preflight_entry = cls.journal.append_preflight_attestation(
            cls.preflight_result)

        # A preflight WITHOUT a device-claim reader must not manufacture a PASS.
        cls.blind_preflight = PF.claim_witness_preflight(
            scope, PF.ClaimSources(region_lock_dir=Path(cls.lock_root)))
        cls.blind_entry = cls.journal.append_preflight_attestation(
            cls.blind_preflight)

        # The claim is held for the duration of the "measurement".
        cls.held_check = DC.check_device_claim_held(
            claim.receipt(), lock_root=cls.lock_root)

        # --- 5. append the campaign, two candidates, two events ---------------
        cls.journal.append(J.KIND_CAMPAIGN_OPENED, _campaign())

        cls.rejected_build_dir = os.path.join(tmp, "build", "akc-0001")
        cls.banked_build_dir = os.path.join(tmp, "build", "akc-0002")
        for build_dir, marker in ((cls.rejected_build_dir, b"REJECTED CANDIDATE\n"),
                                  (cls.banked_build_dir, b"BANKED CANDIDATE\n")):
            os.makedirs(build_dir, exist_ok=True)
            with open(os.path.join(build_dir, "ggml-cuda.o"), "wb") as fh:
                fh.write(marker * 512)

        cls.journal.append(J.KIND_CANDIDATE_RECORDED, _candidate(
            "0001", status="rejected", build_dir=cls.rejected_build_dir,
            receipt_id=cls.receipt_id))
        cls.journal.append(J.KIND_CANDIDATE_RECORDED, _candidate(
            "0002", status="banked", build_dir=cls.banked_build_dir,
            receipt_id=cls.receipt_id))

        cls.stale_event = cls.journal.append(J.KIND_EVALUATION_EVENT, _event(
            "0001", "0001", cls.receipt_id,
            narrative="the wide-tile dispatch is the win; press this direction"))
        cls.live_event = cls.journal.append(J.KIND_EVALUATION_EVENT, _event(
            "0002", "0002", cls.receipt_id,
            narrative="the gain is real and reproduces at np=1"))

        # The receipt every event cites resolves to the claim that was held.
        cls.receipt_check = CW.check_event_claim_receipt(
            cls.live_event.payload, cls.claim_journal)
        cls.invented_receipt_check = CW.check_event_claim_receipt(
            {**cls.live_event.payload,
             "resource_claim_receipt": "akd-0000000000000000"},
            cls.claim_journal)

        # --- 6. stop believing one belief, without forgetting it -------------
        cls.journal.append_retrieval_superseded(
            cls.stale_event.event_id,
            reason="the mechanism claim was traced to a warm-context artifact, "
                   "not the dispatch change",
            receipt=f"retraction receipt bound to {V8_COMMIT}",
        )

        # --- 7. release the claim --------------------------------------------
        cls.release_receipt = claim.release()

        # --- 8. reclaim the rejected candidate's build tree -------------------
        cls.rejected_hash = ST.hash_tree_manifest(cls.rejected_build_dir)
        cls.banked_hash = ST.hash_tree_manifest(cls.banked_build_dir)
        policy = ST.StoragePolicy(campaign_quota_gb=60.0, owned_roots=(tmp,))
        artifact = ST.ExpirableArtifact(
            path=cls.rejected_build_dir,
            campaign_id=CAMPAIGN,
            sha256=cls.rejected_hash,
            durability_class="hash_and_provenance_only",
            expirable_kind="rejected_candidate_build_tree",
            reason="candidate rejected; every outcome it produced is journalled",
            rule_id="R-AK-5.8-rejected-build-tree",
            actor="ak-integration",
            retention_class="expirable",
            preconditions={
                "candidate_id": "akc-20260803-0001",
                "candidate_status": "rejected",
                "champion_status": "none",
                "evaluation_events_journaled": True,
            },
        )
        cls.dry_run = ST.expire_artifact(artifact, policy, now=NOW)
        cls.expiry = ST.expire_artifact(
            artifact, policy, journal=ST.JournalTombstoneSink(cls.journal),
            force=True, now=NOW,
        )

        # A banked candidate's tree is NOT reclaimable — the same call, refused.
        cls.banked_refusal = None
        try:
            ST.expire_artifact(
                ST.ExpirableArtifact(
                    path=cls.banked_build_dir, campaign_id=CAMPAIGN,
                    sha256=cls.banked_hash,
                    durability_class="hash_and_provenance_only",
                    expirable_kind="rejected_candidate_build_tree",
                    reason="disk pressure", rule_id="R-AK-5.8-rejected-build-tree",
                    actor="ak-integration", retention_class="expirable",
                    preconditions={
                        "candidate_id": "akc-20260803-0002",
                        "candidate_status": "banked",
                        "champion_status": "none",
                        "evaluation_events_journaled": True,
                    },
                ),
                policy, journal=ST.JournalTombstoneSink(cls.journal), force=True,
                now=NOW,
            )
        except ST.ExpiryRefused as exc:
            cls.banked_refusal = exc

        # --- 9. read the record back and rebuild the derived views ------------
        cls.events = cls.journal.read_all()
        cls.views = J.rebuild_views(cls.events)
        cls.consistency = J.check_view_consistency(cls.events, cls.views)
        cls.record_rows = cls.events
        cls.retrieval_rows = cls.journal.retrieve()

    # -- one module identity ------------------------------------------------

    def test_every_module_binds_exactly_one_schemas(self):
        """One source of truth that exists twice is not one source of truth.

        `storage.py` used to `sys.path.insert` its own directory and import a
        FLAT `schemas`, so `storage.Check is schemas.Check` was False and every
        `isinstance(verdict, schemas.Check)` across the storage seam said no.
        """
        self.assertIs(ST.Check, S.Check)
        self.assertIs(PF.Check, S.Check)
        self.assertIs(DC.Check, S.Check)
        self.assertIs(CW._pf.Check, S.Check)
        self.assertIsInstance(self.filled_layout, S.Check)
        self.assertIsInstance(self.held_check, S.Check)
        self.assertIsInstance(self.consistency, S.Check)
        # The identity that actually broke: `autokernel.storage` bound a module
        # object that was not `autokernel.schemas`, even though both came from
        # the same file. Asserted against `sys.modules` so it cannot be satisfied
        # by two copies that happen to compare equal.
        canonical = sys.modules["autokernel.schemas"]
        self.assertIs(canonical, S)
        for name in ("autokernel.storage", "autokernel.journal",
                     "autokernel.resource.preflight"):
            module = sys.modules[name]
            bound = getattr(module, "schemas", None) or getattr(module, "_schemas")
            self.assertIs(bound, canonical, f"{name} bound a second schemas")

    def test_no_module_mutates_sys_path_unconditionally_at_import(self):
        """`autokernel/resource/` is a stdlib module name; the package says so.

        `storage.py`'s preamble was an unconditional module-level
        `sys.path.insert(0, <autokernel dir>)`, which put this package ahead of
        the stdlib for the rest of the process: `import resource` then resolved
        to `autokernel/resource/__init__.py`, which has no `getrusage`.

        Asserted STRUCTURALLY, from each module's AST, rather than by observing
        `sys.modules` — the test files themselves insert paths, so an
        observational check would report whichever suite ran first rather than
        the property of the modules under test. A `sys.path` mutation is legal
        only inside an `except ImportError` handler, i.e. only on the genuine
        flat-import fallback.
        """
        import ast  # noqa: PLC0415

        here = Path(__file__).resolve().parent
        modules = [
            here / "schemas.py", here / "journal.py", here / "storage.py",
            here / "resource" / "preflight.py",
            here / "resource" / "device_claim.py",
            here / "resource" / "claim_witness.py",
        ]
        offenders = []
        for path in modules:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            guarded = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for handler in node.handlers:
                        for inner in ast.walk(handler):
                            guarded.add(id(inner))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr != "insert":
                    continue
                target = func.value
                if not (isinstance(target, ast.Attribute) and target.attr == "path"):
                    continue
                if id(node) not in guarded:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], f"unguarded sys.path mutation: {offenders}")

    # -- 1. evidence root ---------------------------------------------------

    def test_the_generated_evidence_root_stub_does_not_pass_its_own_check(self):
        """Creating the layout satisfies the SHAPE, never the CONTENT."""
        self.assertEqual(self.stub_layout.outcome, S.FAIL)
        self.assertTrue(self.evidence.created)
        self.assertEqual(self.filled_layout.outcome, S.PASS)

    # -- 2/3. claim + preflight ---------------------------------------------

    def test_the_preflight_passed_over_our_own_held_claim(self):
        self.assertEqual(self.preflight_result.verdict, PF.PASS)
        self.assertEqual(self.preflight_result.basis, PF.BASIS_CLAIM_WITNESS)
        self.assertTrue(self.preflight_result.passed)
        self.assertEqual([w.device_id for w in self.preflight_result.gpu_claims],
                         [DEVICE])
        self.assertEqual(self.preflight_result.gpu_claims[0].holder_label,
                         "ak-integration")

    def test_a_preflight_that_inspected_no_claim_plane_is_not_a_pass(self):
        """The one shape that would fabricate a P-GPU-1 precondition."""
        self.assertEqual(self.blind_preflight.verdict, PF.COULD_NOT_CHECK)
        self.assertFalse(self.blind_preflight.passed)
        with self.assertRaises(TypeError):
            bool(self.blind_preflight)

    def test_the_claim_was_actually_held_during_the_measurement(self):
        self.assertEqual(self.held_check.outcome, S.PASS)

    def test_both_attestations_are_in_the_record(self):
        """Invariant 7: a precondition checked but not recorded is a skipped one.

        `require_no_concurrent_inference` instructs the caller to journal the
        attestation and the journal's closed vocabulary had no kind for it, so
        the instruction could not be followed at all.
        """
        attestations = [
            e for e in self.events if e.kind == J.KIND_PREFLIGHT_ATTESTATION
        ]
        self.assertEqual(len(attestations), 2)
        self.assertEqual(
            sorted(e.payload["verdict"] for e in attestations),
            sorted([S.PASS, S.COULD_NOT_CHECK]),
        )
        for entry in attestations:
            # The record must survive canonical encoding — that is the form the
            # journal hashed it as.
            self.assertEqual(
                json.loads(S.canonical_json(entry.payload))["basis"],
                entry.payload["basis"],
            )

    def test_a_could_not_check_attestation_may_not_be_journalled_bare(self):
        """The record must not be allowed to say less than the object it came from."""
        with self.assertRaises(ValueError) as ctx:
            self.journal.append(J.KIND_PREFLIGHT_ATTESTATION, {
                "verdict": S.COULD_NOT_CHECK,
                "basis": PF.BASIS_CLAIM_WITNESS,
                "scope": {"label": "x"},
                "observed_at": "2026-08-03T12:00:00Z",
                "reasons": [],
            })
        self.assertIn("could not check", str(ctx.exception))

    # -- 4. receipt binding --------------------------------------------------

    def test_the_events_receipt_resolves_to_the_claim_that_was_held(self):
        self.assertEqual(self.receipt_check.outcome, S.PASS)
        resolved = CW.resolve_claim_receipt(self.receipt_id, self.claim_journal)
        self.assertEqual(resolved.device_id, DEVICE)
        self.assertEqual(resolved.campaign_id, CAMPAIGN)

    def test_an_invented_receipt_is_schema_valid_and_fails_resolution(self):
        """`schemas.py` structurally cannot tell these apart: both are strings."""
        invented = {**self.live_event.payload,
                    "resource_claim_receipt": "akd-0000000000000000"}
        self.assertEqual(S.validate_evaluation_event(invented), [])
        self.assertEqual(self.invented_receipt_check.outcome, S.FAIL)

    # -- 5. views ------------------------------------------------------------

    def test_the_rebuilt_views_describe_the_journal(self):
        self.assertEqual(self.consistency.outcome, S.PASS, self.consistency.reasons)
        J.assert_views_consistent(self.events, self.views)

    def test_the_views_hold_the_campaign_candidates_and_events(self):
        self.assertEqual(list(self.views.campaigns), [CAMPAIGN])
        self.assertEqual(sorted(self.views.candidates),
                         ["akc-20260803-0001", "akc-20260803-0002"])
        self.assertEqual(sorted(self.views.evaluations),
                         ["ake-20260803-0001", "ake-20260803-0002"])
        self.assertEqual(self.views.frontier, ("akc-20260803-0002",))

    def test_a_gutted_view_is_caught_rather_than_certified(self):
        import dataclasses  # noqa: PLC0415

        gutted = dataclasses.replace(self.views, candidates={}, frontier=())
        self.assertEqual(
            J.check_view_consistency(self.events, gutted).outcome, S.FAIL)
        with self.assertRaises(J.ViewConsistencyError):
            J.assert_views_consistent(self.events, gutted)

    # -- 6. expiry -----------------------------------------------------------

    def test_the_dry_run_wrote_nothing_at_all(self):
        self.assertEqual(self.dry_run.state, "DRY_RUN")
        self.assertFalse(self.dry_run.deleted)
        self.assertEqual(self.dry_run.journal_event_ids, ())

    def test_the_artifact_is_gone_and_its_tombstone_is_in_the_record(self):
        """§5.8: the bytes may go; the record of why they went may not."""
        self.assertEqual(self.expiry.state, "RECLAIMED")
        self.assertTrue(self.expiry.deleted)
        self.assertFalse(os.path.lexists(self.rejected_build_dir))

        tombstones = [e for e in self.events if e.kind == J.KIND_TOMBSTONE]
        self.assertEqual([e.payload["reclamation_state"] for e in tombstones],
                         ["intent", "reclaimed"])
        self.assertEqual(self.expiry.journal_event_ids,
                         tuple(e.event_id for e in tombstones))
        for entry in tombstones:
            self.assertEqual(entry.payload["artifact_sha256"], self.rejected_hash)
            self.assertEqual(entry.payload["path"], self.rejected_build_dir)
            self.assertEqual(entry.payload["storage_class"], "expirable")
            self.assertEqual(entry.payload["campaign_id"], CAMPAIGN)
            # The record the journal holds is still a valid artifact tombstone
            # by storage's own validator, not merely by the journal's weaker
            # native check.
            self.assertEqual(ST.validate_artifact_tombstone(entry.payload), [])

        key = J.tombstone_view_key(tombstones[-1].payload)
        self.assertIn(key, self.views.tombstones)
        self.assertEqual(self.views.tombstones[key]["reclamation_state"],
                         "reclaimed")

    def test_a_banked_candidates_tree_is_refused_and_survives(self):
        self.assertIsInstance(self.banked_refusal, ST.ExpiryRefused)
        self.assertTrue(os.path.isdir(self.banked_build_dir))
        self.assertNotIn(
            self.banked_hash,
            [p["artifact_sha256"] for p in self.views.tombstones.values()],
        )

    def test_two_reclamations_of_identical_bytes_stay_two_receipts(self):
        """The journal/storage identity disagreement, at the seam that produced it.

        storage's `tombstone_id` covers (campaign, path, sha, kind, rule); the
        journal's receipt view was keyed by the hash alone. Two byte-identical
        trees at two paths were two `tombstone_id`s and ONE slot, and the
        consistency checker agreed because it recounted by the same key.
        """
        tmp = tempfile.mkdtemp(prefix="ak-identical-", dir=self.tmp)
        journal = J.Journal(os.path.join(tmp, "journal"), campaign_id=CAMPAIGN)
        journal.initialize()
        policy = ST.StoragePolicy(campaign_quota_gb=10.0, owned_roots=(tmp,))
        sink = ST.JournalTombstoneSink(journal)
        ids = set()
        for name in ("treeA", "treeB"):
            path = os.path.join(tmp, name)
            os.makedirs(path)
            with open(os.path.join(path, "obj.o"), "wb") as fh:
                fh.write(b"BYTE IDENTICAL\n" * 64)
            outcome = ST.expire_artifact(
                ST.ExpirableArtifact(
                    path=path, campaign_id=CAMPAIGN,
                    sha256=ST.hash_tree_manifest(path),
                    durability_class="hash_and_provenance_only",
                    expirable_kind="rejected_candidate_build_tree",
                    reason="rejected", rule_id="R-AK-5.8-rejected-build-tree",
                    actor="ak-integration", retention_class="expirable",
                    preconditions={
                        "candidate_id": f"akc-{name}",
                        "candidate_status": "rejected",
                        "champion_status": "none",
                        "evaluation_events_journaled": True,
                    },
                ),
                policy, journal=sink, force=True, now=NOW,
            )
            ids.add(outcome.tombstone["tombstone_id"])
        self.assertEqual(len(ids), 2, "storage produced one id for two artifacts")

        events = journal.read_all()
        views = J.rebuild_views(events)
        self.assertEqual(len(views.tombstones), 2)
        self.assertEqual(
            sorted(p["path"] for p in views.tombstones.values()),
            sorted([os.path.join(tmp, "treeA"), os.path.join(tmp, "treeB")]),
        )
        self.assertEqual(J.check_view_consistency(events, views).outcome, S.PASS)

    # -- 7. record vs retrieval ---------------------------------------------

    def test_the_record_api_still_shows_the_withdrawn_belief(self):
        """Invariant 8: the loop may stop believing, never forget."""
        record_ids = {e.event_id for e in self.record_rows}
        self.assertIn(self.stale_event.event_id, record_ids)
        withdrawn = next(e for e in self.record_rows
                         if e.event_id == self.stale_event.event_id)
        self.assertIn("narrative", withdrawn.payload)
        self.assertIn("press this direction", withdrawn.payload["narrative"])
        # And the view keeps it too — only retrieval withholds it.
        self.assertIn("ake-20260803-0001", self.views.evaluations)
        self.assertIn(self.stale_event.event_id,
                      self.views.retrieval_superseded_event_ids)

    def test_the_retrieval_api_hides_the_withdrawn_belief(self):
        retrieval_ids = {row["event_id"] for row in self.retrieval_rows}
        self.assertNotIn(self.stale_event.event_id, retrieval_ids)
        self.assertIn(self.live_event.event_id, retrieval_ids)

    def test_retrieval_strips_prose_from_the_beliefs_it_does_return(self):
        """§5.5 item 6, invariant 20 — the 81-trials-on-a-false-story scar."""
        live = next(row for row in self.retrieval_rows
                    if row["event_id"] == self.live_event.event_id)
        self.assertNotIn("narrative", live["payload"])
        blob = json.dumps(self.retrieval_rows)
        self.assertNotIn("press this direction", blob)
        self.assertNotIn("the gain is real", blob)

    def test_citing_the_withdrawn_belief_back_in_raises(self):
        with self.assertRaises(J.RetrievalCitationError):
            self.journal.retrieve(cite_event_ids=[self.stale_event.event_id])
        # Citing a LIVE event is how prose legitimately comes back.
        cited = self.journal.retrieve(cite_event_ids=[self.live_event.event_id])
        live = next(row for row in cited
                    if row["event_id"] == self.live_event.event_id)
        self.assertIn("the gain is real", live["payload"]["narrative"])

    def test_a_narrowed_slice_cannot_readmit_a_withdrawn_belief(self):
        """The basis is the whole journal, not whatever list was handed in."""
        evaluations_only = [e for e in self.events
                            if e.kind == J.KIND_EVALUATION_EVENT]
        rows = J.retrieval_filter(evaluations_only, supersession_basis=self.events)
        self.assertNotIn(self.stale_event.event_id,
                         {row["event_id"] for row in rows})

    # -- 8. the claim is released and the whole record survives a reread -----

    def test_the_claim_was_released_and_journalled(self):
        self.assertIsNotNone(self.release_receipt.released_at)
        kinds = [r["kind"] for r in self.claim_journal.read_all()]
        self.assertIn(DC.KIND_ACQUIRED, kinds)
        self.assertIn(DC.KIND_RELEASED, kinds)
        self.assertEqual(
            DC.check_device_claim_held(self.release_receipt,
                                       lock_root=self.lock_root).outcome,
            S.FAIL,
            "a released claim must not read as held",
        )

    def test_a_fresh_reader_rebuilds_the_same_record(self):
        """Deterministic reconstruction from the journal alone (AK1 exit)."""
        reread = J.Journal(os.path.join(self.tmp, "journal"),
                           campaign_id=CAMPAIGN).read_all()
        self.assertEqual(J.events_digest(reread), J.events_digest(self.events))
        views = J.rebuild_views(reread)
        self.assertEqual(views.cardinalities(), self.views.cardinalities())
        self.assertEqual(J.check_view_consistency(reread, views).outcome, S.PASS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
