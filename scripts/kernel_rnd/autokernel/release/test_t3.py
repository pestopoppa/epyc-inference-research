#!/usr/bin/env python3
"""test_t3.py — the T3 kernel-freeze gate and the release bundle (design §10).

Run standalone (no pytest needed):

    python3 -m unittest scripts.kernel_rnd.autokernel.release.test_t3 -v
    python3 -W error::ResourceWarning -m unittest \
        scripts.kernel_rnd.autokernel.release.test_t3

WHAT THIS SUITE IS FOR
----------------------
Three things beyond ordinary coverage:

  * **The §10.4 calibration.** `CalibrationV8` runs the compiler and validator
    against the PRESERVED v8 freeze artifacts and asserts the dry run FAILs without
    the waiver. §10.4: *"the T3 dry-run against preserved v8 artifacts should predict
    a FAIL without the waiver. If it passes, the compiler is wrong."* The test that
    would be comfortable to write is the one that passes; this one asserts the
    failure and names the cells.
  * **The negative space.** A waiver that does not verify, a linkage receipt that
    exits 0 having checked nothing, a rebuilt quality baseline, a rerun on an
    unchanged fingerprint — each has a test asserting the gate REFUSES, because
    every one of them is a way a release could look green while being wrong.
  * **The boundary.** `audit_no_write_or_process_paths` and the authority-key scan
    are asserted here so "T3 never freezes" stops being a docstring.

The fixtures are hand-built rather than compiled from live priors on purpose: this
suite runs no inference, no benchmark, no build, and reads no production tree.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path

from .. import schemas, storage
from ..controller import guards
from ..evaluator import api, integrity
from . import t3

# The preserved operator artifacts §10.4 names. Read when present; the embedded
# fixtures below are cross-checked against them so a fixture cannot drift silently.
_OPERATOR_ARTIFACTS = Path("/workspace/artifacts/operator")
V8_RATIFICATION_PATH = _OPERATOR_ARTIFACTS / "ratify_v8_final_freeze_20260725.json"
V8_WAIVER_PATH = _OPERATOR_ARTIFACTS / "waive_q8_cpu_prefill_v8_20260725.json"
SPEECH_RATIFICATION_PATH = (_OPERATOR_ARTIFACTS /
                            "ratify_speech_kernel_freeze_20260731.json")

V8_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_HEAD = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
V8_CPU_BINARY = "a4b667163022aa166ade7c0e00fa4e775b37662e02c10da7642c8c23a4d6b414"
V8_HIP_BINARY = "112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882"
V8_WAIVER_SHA = "fcd52b61610fcc2782e11f41ffac359343233924805f83d872eeceffbb7522d7"
V7_BASELINE_BINARY = "/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/llama-server"

#: The v8 final-freeze attestation, reduced to the fields T3 reads. Cross-checked
#: against the real artifact in `TestPreservedArtifactFixtures`.
V8_RATIFICATION = {
    "schema": "epyc.operator_v8_final_freeze_attestation.v1",
    "production_branch": "production-consolidated-v8",
    "production_head": V8_HEAD,
    "production_binary_sha256": {"cpu": V8_CPU_BINARY, "hip": V8_HIP_BINARY},
    "rollback": {"branch": "production-consolidated-v7", "head": V7_HEAD},
    "promotion_decision": False,
    "promotion_decision_interpretation": (
        "The CPU campaign's promotion_decision=false is preserved as a non-automatic "
        "matrix verdict; this final freeze is an operator-attested release decision."),
    "q8_claim": ("none; campaign-scoped WAIVE-Q8 remains binding and v8 makes no Q8 "
                 "non-regression claim"),
    "evidence_sha256": {"waive_q8": V8_WAIVER_SHA},
    "production_lineup_gate": {
        "quality_contract": {
            "baseline_binary": V7_BASELINE_BINARY,
            "baseline_kernel": "production-consolidated-v7",
            "candidate_kernel": "production-consolidated-v8",
        },
    },
}

#: The preserved WAIVE-Q8 attestation, verbatim in every field the gate's predicate
#: reads. `epyc.cpu_prefill_v8.operator_waiver.v1` predates the AutoKernel schema and
#: is READ as it is, never rewritten to validate.
V8_WAIVER = {
    "schema": "epyc.cpu_prefill_v8.operator_waiver.v1",
    "decision": "WAIVE-Q8",
    "ratified_at": "2026-07-25T14:04:16Z",
    "protocol": "P-BENCH-PREFILL-1",
    "protocol_changed": False,
    "candidate_head": V8_HEAD,
    "production_head": V7_HEAD,
    "scope": {
        "excluded_model": "qwen36_q8",
        "excluded_pairs": ["qwen36_q8-tg128-iqk1", "qwen36_q8-pp2048-iqk1"],
        "excluded_arm_runs": 4,
        "remaining_matched_pairs": 14,
    },
    "reason": ("The Qwen3.6 Q8 workload naturally sustains about 50-55 target "
               "core-equivalents and cannot satisfy the ratified 72-core eligibility "
               "floor."),
    "consequences": [
        "No v8 Q8 non-regression claim may be made from this campaign.",
        "The ratified 72-core eligibility floor remains unchanged for every remaining arm.",
        "The Gemma Q4 non-IQ B4 pairs remain mandatory.",
        "All retained IQ B3 pairs remain mandatory.",
        "Pre-waiver artifacts remain ineligible and cannot be retro-certified.",
    ],
}

SPEECH_RATIFICATION = {
    "ratification": "speech-kernel-freeze-v1",
    "kernels": {
        "whisper_cpp": {
            "branch": "production-speech-v1",
            "commit": "b307379226d93d9c5ed790d7cea0626613c0ef4b",
            "ggml": "0.18.0",
            "binary_sha256":
                "82aa8b569b7c8ee031f7a8bba6b21425b760654ea05e1d99991067d5d9bd9c7b",
            "load_bearing_patch": ("ggml/src/ggml-cuda/vendors/hip.h — FP8 guard "
                                   "60200000 -> 60300000 for ROCm 6.2"),
        },
        "qwentts_cpp": {
            "branch": "production-speech-v1",
            "commit": "2c1b5182e7e9f1acaa04405ff21747d8a7acf4d5",
            "ggml": "0.17.0",
            "binary_sha256":
                "369fc2f1de88e41f4459e1f56c0e962035e20984acf0ea2d4678f602232ff654",
            "load_bearing_patch": ("ggml/src/ggml-cuda/argsort.{cu,cuh} — thread-strided "
                                   "bitonic sort"),
        },
    },
}


def digest(label: str) -> str:
    """A well-formed, non-placeholder digest derived from a label."""
    return schemas.content_hash({"test_fixture": label})


NOW = "2026-08-03T12:00:00Z"
CAMPAIGN_START = "2026-08-01T00:00:00Z"
LLAMA_BACKENDS = ("llama_cpu", "llama_gpu")
CANDIDATE_COMMIT = "b" * 40
BASE_COMMIT = "a" * 40
BUILD_ROOT = "/mnt/raid0/llm/llama.cpp-experimental/build"


def linkage_receipt(backend: str, **overrides) -> t3.LinkageReceipt:
    fields = {
        "backend": backend,
        "binary_path": f"{BUILD_ROOT}/bin/llama-server",
        "expected_tree_root": BUILD_ROOT,
        "verifier_path":
            f"/mnt/raid0/llm/epyc-inference-research/{t3.LINKAGE_VERIFIER_RELPATH}",
        "verifier_sha256": digest("verify_ggml_linkage.sh"),
        "exit_code": 0,
        "stdout": (f"binary : {BUILD_ROOT}/bin/llama-server\n"
                   f"  OK   libggml-base.so.0            -> {BUILD_ROOT}/bin/libggml-base.so.0\n"
                   f"PASS: all linked ggml libraries resolve inside {BUILD_ROOT}\n"),
        "ld_library_path": (f"{BUILD_ROOT}/bin", "/opt/rocm/lib"),
        "observed_at": NOW,
    }
    fields.update(overrides)
    return t3.LinkageReceipt(**fields)


def matrix_cells() -> list:
    """The gating matrix: prefill + decode per backend, plus the four evidence
    classes §10.2 phases 3, 5, 6 and 7 own."""
    cells: list = []
    for backend in LLAMA_BACKENDS:
        for workload_phase, protocol_id in (("prefill", "P-BENCH-PREFILL-1"),
                                            ("decode", "P-BENCH-1")):
            cells.append(t3.Cell(
                cell_id=f"{backend}.{workload_phase}",
                backend=backend, release_phase=t3.PHASE_PERFORMANCE_MATRIX,
                protocol_id=protocol_id,
                recipe_class=t3.RECIPE_PRODUCTION_OPTIMAL,
                metric="tokens_per_s", metric_direction="higher_better",
                workload_phase=workload_phase,
                claim=f"{backend} {workload_phase} non-regression vs v8",
                roles_protected=("worker_general",),
                co_resident=(backend == "llama_cpu"),
                reps=10))
        for phase_id in (t3.PHASE_BACKEND_CORRECTNESS, t3.PHASE_QUALITY,
                         t3.PHASE_STABILITY, t3.PHASE_CAPACITY_UTILITY):
            cells.append(t3.Cell(
                cell_id=f"{backend}.{phase_id}", backend=backend,
                release_phase=phase_id, protocol_id="P-KERNEL-FREEZE-1",
                recipe_class=t3.RECIPE_PRODUCTION_OPTIMAL, metric="pass_fail",
                metric_direction="higher_better",
                claim=f"{backend} {phase_id} parity", reps=1))
    return cells


def cell_results(cells) -> list:
    return [t3.CellResult(cell=cell, check=schemas.Check(schemas.PASS),
                          raw_samples_ref=f"data/ak/{cell.cell_id}.jsonl",
                          reducer_id="median_mad/v1")
            for cell in cells]


def archive(**overrides) -> t3.IncumbentArchive:
    entry_fields = {
        "generation": t3.ARCHIVE_GENERATION_N1,
        "branch": "production-consolidated-v8",
        "commit": V8_HEAD,
        "archive_root": "/mnt/raid0/llm/kernels/archive/v8",
        "binaries": (("/mnt/raid0/llm/kernels/archive/v8/cpu/llama-server",
                      V8_CPU_BINARY),
                     ("/mnt/raid0/llm/kernels/archive/v8/gpu/llama-server",
                      V8_HIP_BINARY)),
        "libraries": (("/mnt/raid0/llm/kernels/archive/v8/cpu/libggml-base.so.0",
                       digest("v8-libggml-base")),),
        "rebuilt": False,
    }
    entry_fields.update(overrides)
    return t3.IncumbentArchive(entries=(t3.ArchivedBuild(**entry_fields),))


def transaction(**overrides) -> t3.TransactionPlan:
    fields = {
        "next_branch": "production-consolidated-v9",
        "next_version_number": 9,
        "next_tag": "production-consolidated-v9",
        "install_path": "/mnt/raid0/llm/kernels/production",
        "symlink_diff": (
            ("/mnt/raid0/llm/kernels/production/cpu",
             "/mnt/raid0/llm/llama.cpp/build/bin",
             "/mnt/raid0/llm/llama.cpp-v9/build/bin"),
            ("/mnt/raid0/llm/kernels/production/gpu",
             "/mnt/raid0/llm/llama.cpp/build-hip/bin",
             "/mnt/raid0/llm/llama.cpp-v9/build-hip/bin"),
        ),
        "service_impact": ("llama-server restart at the inference owner's boundary",),
        "era_actions": ({"draft": True, "action": "kernel_era_row",
                         "branch": "production-consolidated-v9"},),
        "receipt_paths": ("artifacts/operator/v9-freeze/",),
        "rollback_branch": "production-consolidated-v8",
        "rollback_head": V8_HEAD,
    }
    fields.update(overrides)
    return t3.TransactionPlan(**fields)


def request(**overrides) -> t3.T3Request:
    """A complete request that PASSes. Every test perturbs exactly one thing."""
    cells = overrides.pop("_cells", None) or matrix_cells()
    results = overrides.pop("_results", None)
    if results is None:
        results = cell_results(cells)
    plan = t3.ReleasePlanView(
        plan_id="akplan-v9", plan_sha256=digest("plan-v9"), source_tree="llama.cpp",
        backends=LLAMA_BACKENDS, cells=tuple(cells),
        incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
        incumbent_version_number=8)
    storage_state = storage.StorageState(
        state=storage.STORAGE_OK, free_bytes=200 * 1024 ** 3,
        total_bytes=3700 * 1024 ** 3, floor_bytes=50 * 1024 ** 3)
    fields = {
        "run_id": "akt3-v9-001",
        "campaign_id": "ak-v9",
        "mode": t3.MODE_DRY_RUN,
        "now": NOW,
        "protocol": t3.ProtocolBinding(
            protocol_id=t3.RELEASE_PROTOCOL_ID,
            document_sha256=digest("P-KERNEL-FREEZE-1-draft"), ratified=False),
        "sealed": t3.SealedCandidate(
            candidate_id="akc-v9", source_tree="llama.cpp",
            candidate_branch="llama.cpp-experimental/v9",
            production_base_commit=BASE_COMMIT, candidate_commit=CANDIDATE_COMMIT,
            seal_sha256=digest("seal"), evaluator_bundle_sha256=digest("evaluator"),
            scope_manifest_sha256=digest("scope"),
            evidence_tree_sha256=digest("evidence"),
            binary_sha256={b: digest(f"bin:{b}") for b in LLAMA_BACKENDS},
            linkage_sha256={b: digest(f"link:{b}") for b in LLAMA_BACKENDS},
            build_dirs={b: BUILD_ROOT for b in LLAMA_BACKENDS},
            overlay_present=True, tree_clean=True, ancestry_clean=True),
        "plan": plan,
        "backend_unchanged": {
            b: t3.UnchangedView(backend=b, may_drop_cells=False,
                                unchanged_outcome=schemas.FAIL,
                                agreement_outcome=schemas.PASS, stage2_ran=True,
                                reasons=("the closure changed",))
            for b in LLAMA_BACKENDS},
        "host": guards.HostHealth(uptime_seconds=3600, observed_at=NOW,
                                  receipt="host-receipt-1"),
        "host_owner": "operator",
        "host_escalation_deadline": "2026-08-04T12:00:00Z",
        "resource_claims": tuple(
            guards.ResourceClaimObservation(
                resource=b, claim_kind="cpu_region" if b.endswith("cpu") else "gpu_device",
                acquired=True, observed_at=NOW, receipt=f"claim-{b}", held_by="akt3-v9-001")
            for b in LLAMA_BACKENDS),
        "storage_observation": guards.StorageObservation(
            path="/mnt/raid0", state=storage_state, expirable_backlog_bytes=0,
            receipt="storage-receipt-1"),
        "transaction": transaction(),
        "archive": archive(),
        "supplied_components": {name: digest(f"component:{name}")
                                for name in t3.SUPPLIED_COMPONENTS},
        "cooldown_seconds": 86400,
        "release_reps_by_protocol": {"P-BENCH-1": 10, "P-BENCH-PREFILL-1": 5,
                                     "P-KERNEL-FREEZE-1": 1},
        "phase_protocols": {b: {"prefill": "P-BENCH-PREFILL-1", "decode": "P-BENCH-1"}
                            for b in LLAMA_BACKENDS},
        "linkage_receipts": tuple(linkage_receipt(b) for b in LLAMA_BACKENDS),
        "backend_inventories": tuple(
            t3.BackendInventory(
                backend=b, entries=("CPU",) + (("HIP",) if b.endswith("gpu") else ()),
                device_entries=(("AMD Instinct MI210",) if b.endswith("gpu") else ()),
                source_ref=f"startup-log:{b}")
            for b in LLAMA_BACKENDS),
        "determinism": tuple(
            t3.DeterminismDeclaration(
                backend=b, anchor_class="bitwise_stable",
                candidate_class="bitwise_stable", evidence_ref=f"det:{b}")
            for b in LLAMA_BACKENDS),
        "cell_results": tuple(results),
        "standings": tuple(
            t3.PhaseStanding(backend=b, workload_phase=phase, protocol_id=protocol,
                             standing=standing, cell_ids=(f"{b}.{phase}",),
                             evidence_ref=f"standing:{b}.{phase}")
            for b in LLAMA_BACKENDS
            for phase, protocol, standing in (
                ("prefill", "P-BENCH-PREFILL-1", t3.STANDING_IMPROVED),
                ("decode", "P-BENCH-1", t3.STANDING_NON_INFERIOR))),
        "quality_evidence": tuple(
            t3.QualityEvidence(
                backend=b, mode=t3.QUALITY_MEASURED_PARITY,
                baseline_binary_path="/mnt/raid0/llm/kernels/archive/v8/cpu/llama-server",
                baseline_binary_sha256=V8_CPU_BINARY,
                baseline_kernel="production-consolidated-v8",
                baseline_is_rebuild=False,
                evidence_refs=(f"data/ak/quality/{b}.json",),
                suites=("mmlu_pro", "gpqa"), shared_question_identity=True)
            for b in LLAMA_BACKENDS),
        "stability_evidence": tuple(
            t3.StabilityEvidence(
                backend=b, load_unload_cycles=5, memory_growth_bytes=0,
                memory_growth_allowance_bytes=1024, profiler_or_runtime_errors=0,
                cleanup_verified=True, mixed_prefill_decode_exercised=True,
                evidence_ref=f"stability:{b}")
            for b in LLAMA_BACKENDS),
        "stability_min_cycles": 5,
        "complexity": {
            b: integrity.ComplexityAssessment(
                requires_human_code_review=False, reasons=(), first_page_notice=None,
                measured={"total_changed_lines": 40, "files_touched": 2})
            for b in LLAMA_BACKENDS},
        "campaign_start_at": CAMPAIGN_START,
    }
    fields.update(overrides)
    return t3.T3Request(**fields)


def reasons_of(result: t3.T3Result) -> str:
    return " | ".join(result.verdict_computation.blocking_reasons)


# =============================================================================
# Vocabulary, structure, and the boundary
# =============================================================================

class TestStructure(unittest.TestCase):

    def test_nine_phases_in_order(self):
        self.assertEqual(len(t3.PHASES), 9)
        self.assertEqual(t3.PHASES[0], t3.PHASE_IDENTITY_PREFLIGHT)
        self.assertEqual(t3.PHASES[-1], t3.PHASE_SEAL)

    def test_phase_coverage_is_total(self):
        self.assertEqual(t3.audit_phase_coverage_totality().outcome, schemas.PASS)

    def test_module_cannot_write_spawn_or_signal(self):
        check = t3.audit_no_write_or_process_paths()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_audit_reports_a_write_path_when_one_exists(self):
        # The audit must be able to FAIL, or its PASS means nothing.
        check = t3.audit_no_write_or_process_paths(
            "import subprocess\nsubprocess.run(['x'])\n")
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_audit_could_not_check_on_unparseable_source(self):
        self.assertEqual(
            t3.audit_no_write_or_process_paths("def (").outcome,
            schemas.COULD_NOT_CHECK)

    def test_search_evaluator_refuses_this_tier_and_names_the_owner(self):
        self.assertIn(t3.TIER, api.RELEASE_TIERS)
        with self.assertRaises(api.TierNotOwned):
            api.admit_tier(t3.TIER)

    def test_bundle_components_are_the_seven_of_10_2(self):
        self.assertEqual(len(t3.BUNDLE_COMPONENTS), 7)
        self.assertEqual(
            set(t3.SUPPLIED_COMPONENTS) | set(t3.COMPUTED_COMPONENTS),
            set(t3.BUNDLE_COMPONENTS))
        self.assertFalse(set(t3.SUPPLIED_COMPONENTS) & set(t3.COMPUTED_COMPONENTS))

    def test_verdict_vocabulary_matches_the_schema(self):
        self.assertEqual(schemas.T3_VERDICTS, {"PASS", "FAIL", "PASS_WITH_WAIVER"})


# =============================================================================
# Input refusals — wiring defects raise, they are not candidate failures
# =============================================================================

class TestInputRefusals(unittest.TestCase):

    def test_serving_runtime_is_refused_at_the_kernel_freeze_path(self):
        with self.assertRaises(t3.StackChangePathRequired) as ctx:
            t3.ReleasePlanView(
                plan_id="p", plan_sha256=digest("p"), source_tree="llama.cpp",
                backends=("serving_runtime",), cells=matrix_cells(),
                incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
                incumbent_version_number=8)
        self.assertIn("stack-change", str(ctx.exception))

    def test_plan_may_not_omit_a_backend_the_tree_serves(self):
        cells = [c for c in matrix_cells() if c.backend == "llama_cpu"]
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.ReleasePlanView(
                plan_id="p", plan_sha256=digest("p"), source_tree="llama.cpp",
                backends=("llama_cpu",), cells=cells,
                incumbent_branch="production-consolidated-v8", incumbent_commit=V8_HEAD,
                incumbent_version_number=8)
        self.assertIn("union of backends", str(ctx.exception))

    def test_candidate_branch_may_not_be_a_frozen_production_branch(self):
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.SealedCandidate(
                candidate_id="akc-x", source_tree="llama.cpp",
                candidate_branch="production-consolidated-v9",
                production_base_commit=BASE_COMMIT, candidate_commit=CANDIDATE_COMMIT,
                seal_sha256=digest("s"), evaluator_bundle_sha256=digest("e"),
                scope_manifest_sha256=digest("m"), evidence_tree_sha256=digest("t"))
        self.assertIn("version PAST production", str(ctx.exception))

    def test_candidate_equal_to_base_is_refused(self):
        with self.assertRaises(t3.T3InputError):
            t3.SealedCandidate(
                candidate_id="akc-x", source_tree="llama.cpp",
                candidate_branch="exp", production_base_commit=BASE_COMMIT,
                candidate_commit=BASE_COMMIT, seal_sha256=digest("s"),
                evaluator_bundle_sha256=digest("e"), scope_manifest_sha256=digest("m"),
                evidence_tree_sha256=digest("t"))

    def test_placeholder_digest_is_refused(self):
        empty = schemas.content_hash  # noqa: F841 — documents where the digest is from
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.ProtocolBinding(
                protocol_id="P", ratified=False,
                document_sha256=(
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        self.assertIn("no bytes at all", str(ctx.exception))

    def test_executed_transaction_is_refused_outright(self):
        with self.assertRaises(t3.ProductionWriteRefused) as ctx:
            transaction(executed=True)
        self.assertIn("DRY RUN", str(ctx.exception))

    def test_era_action_must_be_a_draft(self):
        with self.assertRaises(t3.T3InputError) as ctx:
            transaction(era_actions=({"action": "kernel_era_row"},))
        self.assertIn("draft=True", str(ctx.exception))

    def test_diagnostic_cell_may_not_carry_a_claim(self):
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.Cell(cell_id="d", backend="llama_cpu",
                    release_phase=t3.PHASE_PERFORMANCE_MATRIX, protocol_id="P-BENCH-1",
                    recipe_class=t3.RECIPE_DIAGNOSTIC, metric="tokens_per_s",
                    metric_direction="higher_better", claim="baseline is fine")
        self.assertIn("invariant 15", str(ctx.exception).lower())

    def test_release_mode_under_an_unratified_protocol_is_refused(self):
        with self.assertRaises(t3.ReleaseProtocolNotRatified) as ctx:
            t3.run_t3(request(mode=t3.MODE_RELEASE))
        self.assertIn("AK-D20", str(ctx.exception))

    def test_result_for_a_cell_outside_the_plan_is_refused(self):
        cells = matrix_cells()
        stray = t3.Cell(cell_id="not.in.plan", backend="llama_cpu",
                        release_phase=t3.PHASE_PERFORMANCE_MATRIX,
                        protocol_id="P-BENCH-1",
                        recipe_class=t3.RECIPE_PRODUCTION_OPTIMAL,
                        metric="tokens_per_s", metric_direction="higher_better",
                        workload_phase="decode", reps=10)
        with self.assertRaises(t3.T3InputError) as ctx:
            request(_cells=cells,
                    _results=cell_results(cells) + cell_results([stray]))
        self.assertIn("widening its own matrix", str(ctx.exception))

    def test_phase_result_verdict_cannot_be_stamped(self):
        with self.assertRaises(t3.T3Error):
            t3.PhaseResult(phase_id=t3.PHASE_SEAL, check=schemas.Check(schemas.PASS),
                           blocking_reasons=("something broke",))


# =============================================================================
# The happy path
# =============================================================================

class TestPassingRun(unittest.TestCase):

    def setUp(self):
        self.result = t3.run_t3(request())

    def test_verdict_is_pass(self):
        self.assertEqual(self.result.verdict, "PASS", reasons_of(self.result))

    def test_every_phase_ran_in_order(self):
        self.assertEqual([p.phase_id for p in self.result.phase_results], list(t3.PHASES))

    def test_bundle_seals_and_rehashes(self):
        bundle = self.result.bundle
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.bundle_sha256, schemas.content_hash(bundle.payload))

    def test_bundle_hashes_all_seven_components(self):
        digests = self.result.bundle.payload["component_digests"]
        self.assertEqual(set(digests), set(t3.BUNDLE_COMPONENTS))

    def test_bundle_carries_no_authority_flavoured_key(self):
        self.assertEqual(
            schemas.find_authority_flavoured_keys(self.result.bundle.payload), [])

    def test_bundle_records_the_mode_so_a_dry_run_cannot_pose_as_a_release(self):
        self.assertEqual(self.result.bundle.payload["mode"], t3.MODE_DRY_RUN)
        self.assertFalse(self.result.bundle.payload["release_protocol"]["ratified"])

    def test_receipt_enumerates_the_claims_the_release_licenses(self):
        claims = self.result.receipt.claims
        self.assertIn("llama_cpu decode non-regression vs v8", claims)
        self.assertEqual(self.result.receipt.suppressed_claims, ())

    def test_computed_components_are_not_caller_supplied(self):
        """A caller cannot hand T3 a digest of validation results T3 did not produce."""
        result = t3.run_t3(request(supplied_components={
            **{n: digest(n) for n in t3.SUPPLIED_COMPONENTS},
            t3.COMPONENT_VALIDATION_RESULTS: digest("forged")}))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIsNone(result.bundle)
        self.assertIn("second source of truth for the seal", reasons_of(result))

    def test_validation_results_digest_tracks_the_phases(self):
        """The computed digest must change when the phase results change."""
        passing = t3.run_t3(request())
        cells, results = failing_matrix()
        failing = t3.run_t3(request(_cells=cells, _results=results))
        self.assertNotEqual(
            passing.bundle.payload["component_digests"][t3.COMPONENT_VALIDATION_RESULTS],
            failing.bundle.payload["component_digests"][t3.COMPONENT_VALIDATION_RESULTS])

    def test_missing_supplied_component_blocks_the_seal(self):
        partial = {n: digest(n) for n in t3.SUPPLIED_COMPONENTS
                   if n != t3.COMPONENT_RAW_EVIDENCE}
        result = t3.run_t3(request(supplied_components=partial))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIsNone(result.bundle)
        self.assertIn("raw_evidence", reasons_of(result))


# =============================================================================
# Operator waivers (§10.4)
# =============================================================================

def failing_matrix():
    """A matrix with one FAILing gating performance cell."""
    cells = matrix_cells()
    results = []
    for cell in cells:
        check = schemas.Check(schemas.PASS)
        if cell.cell_id == "llama_cpu.prefill":
            check = t3._fail("qwen36_q8 cannot satisfy the 72-core eligibility floor")
        results.append(t3.CellResult(cell=cell, check=check,
                                     raw_samples_ref=f"data/ak/{cell.cell_id}.jsonl",
                                     reducer_id="median_mad/v1"))
    return cells, results


def autokernel_waiver(**overrides) -> dict:
    doc = {
        "schema": schemas.SCHEMA_OPERATOR_WAIVER,
        "waiver_id": "WAIVE-Q8-V9",
        "campaign_id": "ak-v9",
        "decision": "WAIVE",
        "protocol": "P-BENCH-PREFILL-1",
        "protocol_changed": False,
        "candidate_head": CANDIDATE_COMMIT,
        "production_head": BASE_COMMIT,
        "scope": {"excluded_models": ["qwen36_q8"],
                  "excluded_pairs": ["llama_cpu.prefill"],
                  "remaining_matched_pairs": 14},
        "reason": "the Q8 workload cannot satisfy the ratified core-equivalent floor",
        "consequences": ["No v9 Q8 prefill non-regression claim may be made."],
        "authorized_by": "operator",
        "expiry": {"expires_at": None, "reopen_predicate": "the floor is re-derived"},
        "created_at": "2026-08-02T00:00:00Z",
        "narrative": None,
    }
    doc.update(overrides)
    return doc


def waiver_binding(document=None, **overrides) -> t3.WaiverBinding:
    document = document if document is not None else autokernel_waiver()
    pinned = digest(json.dumps(document, sort_keys=True))
    fields = {
        "waiver_id": document.get("waiver_id", "W"),
        "pinned_sha256": pinned,
        "observed_sha256": pinned,
        "document": document,
        "document_path": "artifacts/operator/waive-q8-v9.json",
        "covers_cell_ids": ("llama_cpu.prefill",),
    }
    fields.update(overrides)
    return t3.WaiverBinding(**fields)


class TestWaivers(unittest.TestCase):

    def test_failing_cell_without_a_waiver_fails(self):
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("llama_cpu.prefill", result.verdict_computation.failed_cells)

    def test_verified_waiver_yields_pass_with_waiver_and_suppresses_the_claim(self):
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(waiver_binding(),)))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER", reasons_of(result))
        self.assertEqual(result.verdict_computation.failed_cells, ())
        suppressed = [s["claim"] for s in result.receipt.suppressed_claims]
        self.assertIn("llama_cpu prefill non-regression vs v8", suppressed)
        self.assertNotIn("llama_cpu prefill non-regression vs v8", result.receipt.claims)

    def test_the_forfeited_claim_is_named_in_the_receipt(self):
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(waiver_binding(),)))
        self.assertEqual(
            result.receipt.forfeited_claims,
            ("No v9 Q8 prefill non-regression claim may be made.",))

    def test_a_pass_run_that_pins_a_waiver_stays_pass_with_waiver_only_if_used(self):
        # A waiver over a cell that passed suppresses nothing, so the verdict is PASS.
        result = t3.run_t3(request(waivers=(waiver_binding(),)))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))
        self.assertEqual(result.receipt.suppressed_claims, ())

    def test_hash_mismatch_blocks(self):
        cells, results = failing_matrix()
        binding = waiver_binding(observed_sha256=digest("something else"))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is not the waiver that is here", reasons_of(result))

    def test_unread_waiver_is_could_not_check_and_still_blocks(self):
        cells, results = failing_matrix()
        binding = waiver_binding(observed_sha256=None)
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("A quoted waiver is not a read one", reasons_of(result))

    def test_waiver_naming_another_candidate_head_blocks(self):
        cells, results = failing_matrix()
        binding = waiver_binding(autokernel_waiver(candidate_head="c" * 40))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("names candidate head", reasons_of(result))

    def test_waiver_whose_protocol_moved_blocks(self):
        cells, results = failing_matrix()
        binding = waiver_binding(autokernel_waiver(protocol_changed=True))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("protocol that moved underneath it", reasons_of(result))

    def test_expired_waiver_blocks(self):
        cells, results = failing_matrix()
        binding = waiver_binding(autokernel_waiver(
            expiry={"expires_at": "2026-07-01T00:00:00Z"}))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("expired", reasons_of(result))

    def test_waiver_covering_a_cell_outside_the_matrix_blocks(self):
        cells, results = failing_matrix()
        binding = waiver_binding(covers_cell_ids=("no.such.cell",))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("not in this release matrix", reasons_of(result))

    def test_waiver_forfeiting_nothing_is_an_approval_and_blocks(self):
        cells, results = failing_matrix()
        binding = waiver_binding(autokernel_waiver(consequences=[]))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("forfeits no claim", reasons_of(result))

    def test_machine_authored_waiver_without_a_human_attestation_blocks(self):
        cells, results = failing_matrix()
        doc = autokernel_waiver()
        doc.pop("authorized_by")
        binding = waiver_binding(doc)
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("human-authored", reasons_of(result))

    def test_the_evaluator_does_not_judge_the_waivers_merits(self):
        """§10.4: it verifies the hash and the predicate, never the merits.

        Two waivers identical but for the operator's stated REASON must produce the
        same predicate results — otherwise the gate is grading the argument.
        """
        cells, results = failing_matrix()
        base = t3.verify_waiver(
            waiver_binding(autokernel_waiver(reason="a well-argued reason")),
            candidate_commit=CANDIDATE_COMMIT, production_base_commit=BASE_COMMIT,
            campaign_id="ak-v9", known_cell_ids=[c.cell_id for c in cells],
            failing_cell_ids=[r.cell.cell_id for r in results
                              if r.check.outcome != schemas.PASS], now=NOW)
        weak = t3.verify_waiver(
            waiver_binding(autokernel_waiver(reason="because")),
            candidate_commit=CANDIDATE_COMMIT, production_base_commit=BASE_COMMIT,
            campaign_id="ak-v9", known_cell_ids=[c.cell_id for c in cells],
            failing_cell_ids=[r.cell.cell_id for r in results
                              if r.check.outcome != schemas.PASS], now=NOW)
        self.assertEqual(base.predicate_results, weak.predicate_results)
        self.assertTrue(base.verified and weak.verified)

    def test_a_waiver_cannot_waive_a_phase_blocker(self):
        """A linkage failure is not a scoped, claim-forfeiting exclusion."""
        cells, results = failing_matrix()
        receipts = (linkage_receipt("llama_cpu", exit_code=1,
                                    stdout="  BAD  libggml-base.so.0 -> /elsewhere/lib\n"),
                    linkage_receipt("llama_gpu"))
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(waiver_binding(),),
                                   linkage_receipts=receipts))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("build_linkage", reasons_of(result))

    def test_unknown_waiver_schema_blocks(self):
        cells, results = failing_matrix()
        binding = waiver_binding(autokernel_waiver(schema="epyc.some.other.v1"))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("not a waiver schema this gate reads", reasons_of(result))


# =============================================================================
# Phase 2 — linkage (INC-20260731-ggml-linkage-silent-cpu-fallback)
# =============================================================================

class TestLinkage(unittest.TestCase):

    def _run(self, **receipt_overrides):
        receipts = (linkage_receipt("llama_cpu", **receipt_overrides),
                    linkage_receipt("llama_gpu"))
        return t3.run_t3(request(linkage_receipts=receipts))

    def test_bad_library_line_fails(self):
        result = self._run(stdout="  BAD  libggml-base.so.0 -> /mnt/raid0/llm/other/lib\n",
                           exit_code=1)
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("runs silently wrong", reasons_of(result))

    def test_the_scripts_own_fail_open_is_not_inherited(self):
        """`verify_ggml_linkage.sh` exits 0 when ldd finds no ggml libraries at all."""
        result = self._run(
            stdout="  (no ggml/whisper/llama libs in ldd output — statically linked, "
                   "or ldd failed)\n",
            exit_code=0)
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is not 'everything resolved correctly'", reasons_of(result))

    def test_verifier_must_be_the_research_repo_copy(self):
        result = self._run(
            verifier_path=f"/mnt/raid0/llm/epyc-root/{t3.LINKAGE_VERIFIER_RELPATH}")
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("epyc-inference-research", reasons_of(result))

    def test_reintroducing_a_frozen_production_library_dir_fails(self):
        result = self._run(ld_library_path=(f"{BUILD_ROOT}/bin",
                                            "/mnt/raid0/llm/llama.cpp/build/bin"))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("removed from the global environment", reasons_of(result))

    def test_ld_library_path_must_lead_with_the_candidates_own_tree(self):
        result = self._run(ld_library_path=("/opt/rocm/lib", f"{BUILD_ROOT}/bin"))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("must set its OWN", reasons_of(result))

    def test_a_proof_about_another_tree_proves_nothing(self):
        result = self._run(expected_tree_root="/mnt/raid0/llm/some-other-tree/build",
                           stdout="PASS: all linked ggml libraries resolve inside "
                                  "/mnt/raid0/llm/some-other-tree/build\n",
                           ld_library_path=("/mnt/raid0/llm/some-other-tree/build/bin",))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("proves nothing about this binary", reasons_of(result))

    def test_missing_receipt_fails(self):
        result = t3.run_t3(request(linkage_receipts=(linkage_receipt("llama_cpu"),)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("llama_gpu: no", reasons_of(result))

    def test_gpu_inventory_without_a_device_fails(self):
        inventories = (
            t3.BackendInventory(backend="llama_cpu", entries=("CPU",),
                                source_ref="log"),
            t3.BackendInventory(backend="llama_gpu", entries=("CPU", "HIP"),
                                device_entries=(), source_ref="log"))
        result = t3.run_t3(request(backend_inventories=inventories))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("runs on the CPU while still reporting", reasons_of(result))

    def test_building_inside_a_production_tree_is_refused(self):
        sealed = request().sealed
        result = t3.run_t3(request(sealed=t3.SealedCandidate(
            **{**sealed.to_dict(),
               "build_dirs": {b: "/mnt/raid0/llm/llama.cpp/build" for b in LLAMA_BACKENDS}})))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("inside a FROZEN production tree", reasons_of(result))


# =============================================================================
# Phase 3 — correctness and determinism
# =============================================================================

class TestCorrectness(unittest.TestCase):

    def test_undeclared_determinism_class_change_fails(self):
        declarations = (
            t3.DeterminismDeclaration(backend="llama_cpu", anchor_class="bitwise_stable",
                                      candidate_class="bitwise_unstable",
                                      evidence_ref="det"),
            t3.DeterminismDeclaration(backend="llama_gpu", anchor_class="bitwise_stable",
                                      candidate_class="bitwise_stable",
                                      evidence_ref="det"))
        result = t3.run_t3(request(determinism=declarations))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("without being declared", reasons_of(result))

    def test_declared_determinism_class_change_passes(self):
        declarations = (
            t3.DeterminismDeclaration(backend="llama_cpu", anchor_class="bitwise_stable",
                                      candidate_class="bitwise_unstable",
                                      change_declared=True, evidence_ref="det"),
            t3.DeterminismDeclaration(backend="llama_gpu", anchor_class="bitwise_stable",
                                      candidate_class="bitwise_stable",
                                      evidence_ref="det"))
        result = t3.run_t3(request(determinism=declarations))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_unmeasured_determinism_is_not_unchanged(self):
        declarations = tuple(
            t3.DeterminismDeclaration(backend=b, anchor_class="bitwise_stable",
                                      candidate_class="not_measured", evidence_ref="d")
            for b in LLAMA_BACKENDS)
        result = t3.run_t3(request(determinism=declarations))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("not an unchanged one", reasons_of(result))

    def test_missing_correctness_cell_is_a_skip_not_a_pass(self):
        cells = [c for c in matrix_cells()
                 if not (c.backend == "llama_gpu"
                         and c.release_phase == t3.PHASE_BACKEND_CORRECTNESS)]
        result = t3.run_t3(request(_cells=cells, _results=cell_results(cells)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("it has skipped it", reasons_of(result))

    def test_gating_cell_without_raw_samples_is_invalid(self):
        cells = matrix_cells()
        results = cell_results(cells)
        results[0] = t3.CellResult(cell=results[0].cell,
                                   check=schemas.Check(schemas.PASS),
                                   reducer_id="median_mad/v1")
        result = t3.run_t3(request(_cells=cells, _results=results))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("cannot be recomputed from its raw samples", reasons_of(result))


# =============================================================================
# Phase 4 — the performance matrix
# =============================================================================

class TestPerformanceMatrix(unittest.TestCase):

    def test_diagnostic_cell_failure_never_vetoes(self):
        cells = matrix_cells()
        diagnostic = t3.Cell(
            cell_id="llama_cpu.baseline", backend="llama_cpu",
            release_phase=t3.PHASE_PERFORMANCE_MATRIX, protocol_id="P-BENCH-1",
            recipe_class=t3.RECIPE_DIAGNOSTIC, metric="tokens_per_s",
            metric_direction="higher_better", workload_phase="decode")
        cells.append(diagnostic)
        results = cell_results([c for c in cells if c is not diagnostic])
        results.append(t3.CellResult(cell=diagnostic,
                                     check=t3._fail("the off-recipe baseline regressed")))
        result = t3.run_t3(request(_cells=cells, _results=results))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_reps_below_the_release_rule_fail(self):
        cells = [c if c.cell_id != "llama_cpu.decode" else t3.Cell(
            cell_id=c.cell_id, backend=c.backend, release_phase=c.release_phase,
            protocol_id=c.protocol_id, recipe_class=c.recipe_class, metric=c.metric,
            metric_direction=c.metric_direction, workload_phase=c.workload_phase,
            claim=c.claim, roles_protected=c.roles_protected,
            co_resident=c.co_resident, reps=3) for c in matrix_cells()]
        result = t3.run_t3(request(_cells=cells, _results=cell_results(cells)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("below the 10 release reps", reasons_of(result))

    def test_protocol_with_no_declared_release_reps_fails(self):
        result = t3.run_t3(request(release_reps_by_protocol={"P-BENCH-1": 10}))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("nobody ratified", reasons_of(result))

    def test_cell_measured_under_the_wrong_protocol_fails(self):
        cells = [
            t3.Cell(cell_id=c.cell_id, backend=c.backend, release_phase=c.release_phase,
                    protocol_id=("P-BENCH-1" if c.cell_id.endswith("prefill")
                                 else c.protocol_id),
                    recipe_class=c.recipe_class, metric=c.metric,
                    metric_direction=c.metric_direction,
                    workload_phase=c.workload_phase, claim=c.claim,
                    roles_protected=c.roles_protected, co_resident=c.co_resident,
                    reps=c.reps)
            for c in matrix_cells()]
        result = t3.run_t3(request(_cells=cells, _results=cell_results(cells)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("one protocol and one instrument version", reasons_of(result))

    def test_missing_co_resident_cell_for_llama_cpu_fails(self):
        cells = [
            t3.Cell(cell_id=c.cell_id, backend=c.backend, release_phase=c.release_phase,
                    protocol_id=c.protocol_id, recipe_class=c.recipe_class,
                    metric=c.metric, metric_direction=c.metric_direction,
                    workload_phase=c.workload_phase, claim=c.claim,
                    roles_protected=c.roles_protected, co_resident=False, reps=c.reps)
            for c in matrix_cells()]
        result = t3.run_t3(request(_cells=cells, _results=cell_results(cells)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("no gating co-resident cell", reasons_of(result))

    def test_task_rate_on_a_kernel_backend_is_not_commensurable(self):
        cells = [
            t3.Cell(cell_id=c.cell_id, backend=c.backend, release_phase=c.release_phase,
                    protocol_id=c.protocol_id, recipe_class=c.recipe_class,
                    metric=("task_rate" if c.cell_id == "llama_cpu.decode" else c.metric),
                    metric_direction=c.metric_direction,
                    workload_phase=c.workload_phase, claim=c.claim,
                    roles_protected=c.roles_protected, co_resident=c.co_resident,
                    reps=c.reps)
            for c in matrix_cells()]
        result = t3.run_t3(request(_cells=cells, _results=cell_results(cells)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("task_rate belongs", reasons_of(result))

    def test_full_machine_gate_on_a_partial_machine_cell_is_refused(self):
        cells = [
            t3.Cell(cell_id=c.cell_id, backend=c.backend, release_phase=c.release_phase,
                    protocol_id=c.protocol_id, recipe_class=c.recipe_class,
                    metric=c.metric, metric_direction=c.metric_direction,
                    workload_phase=c.workload_phase, claim=c.claim,
                    roles_protected=c.roles_protected, co_resident=c.co_resident,
                    reps=c.reps,
                    scope_denominator=({"machine_subset": "partial",
                                        "numa_nodes": [0], "devices": [], "cores": 48}
                                       if c.cell_id == "llama_cpu.decode" else None))
            for c in matrix_cells()]
        result = t3.run_t3(request(
            _cells=cells, _results=cell_results(cells),
            gate_scope={"machine_subset": "full", "cores": 96}))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("partial machine", reasons_of(result))


# =============================================================================
# Phase 5 — quality, and the §10.5 preserved-incumbent rule
# =============================================================================

class TestQuality(unittest.TestCase):

    def _quality(self, **overrides):
        base = {
            "mode": t3.QUALITY_MEASURED_PARITY,
            "baseline_binary_path": "/mnt/raid0/llm/kernels/archive/v8/cpu/llama-server",
            "baseline_binary_sha256": V8_CPU_BINARY,
            "baseline_kernel": "production-consolidated-v8",
            "baseline_is_rebuild": False,
            "evidence_refs": ("data/ak/quality.json",),
            "suites": ("mmlu_pro",),
            "shared_question_identity": True,
        }
        base.update(overrides)
        return tuple(t3.QualityEvidence(backend=b, **base) for b in LLAMA_BACKENDS)

    def test_a_rebuilt_baseline_is_not_the_incumbent(self):
        result = t3.run_t3(request(quality_evidence=self._quality(
            baseline_is_rebuild=True)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("does not reproduce it", reasons_of(result))

    def test_an_unarchived_baseline_cannot_be_rerun(self):
        result = t3.run_t3(request(quality_evidence=self._quality(
            baseline_binary_path=V7_BASELINE_BINARY)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("not among the archived incumbent binaries", reasons_of(result))

    def test_a_moved_baseline_hash_fails(self):
        result = t3.run_t3(request(quality_evidence=self._quality(
            baseline_binary_sha256=digest("a different build"))))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("moved under the evidence", reasons_of(result))

    def test_transferred_quality_needs_the_paired_parity_receipt(self):
        result = t3.run_t3(request(quality_evidence=self._quality(
            mode=t3.QUALITY_TRANSFERRED)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("assumption wearing a receipt", reasons_of(result))

    def test_transferred_quality_with_a_receipt_passes(self):
        result = t3.run_t3(request(quality_evidence=self._quality(
            mode=t3.QUALITY_TRANSFERRED,
            paired_parity_receipt="data/ak/quality/paired-parity.json")))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_measured_parity_needs_shared_question_identity(self):
        result = t3.run_t3(request(quality_evidence=self._quality(
            shared_question_identity=False)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("answered different questions", reasons_of(result))

    def test_missing_quality_evidence_is_not_a_third_route(self):
        result = t3.run_t3(request(quality_evidence=()))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("silence is not one of them", reasons_of(result))


# =============================================================================
# Phase 6 — stability
# =============================================================================

class TestStability(unittest.TestCase):

    def _stability(self, **overrides):
        base = {"load_unload_cycles": 5, "memory_growth_bytes": 0,
                "memory_growth_allowance_bytes": 1024,
                "profiler_or_runtime_errors": 0, "cleanup_verified": True,
                "mixed_prefill_decode_exercised": True, "evidence_ref": "s"}
        base.update(overrides)
        return tuple(t3.StabilityEvidence(backend=b, **base) for b in LLAMA_BACKENDS)

    def test_memory_growth_beyond_the_allowance_fails(self):
        result = t3.run_t3(request(stability_evidence=self._stability(
            memory_growth_bytes=4096)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("memory grew", reasons_of(result))

    def test_unverified_cleanup_fails_regardless_of_throughput(self):
        result = t3.run_t3(request(stability_evidence=self._stability(
            cleanup_verified=False)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("cleanup failure a FAIL regardless of throughput",
                      reasons_of(result))

    def test_unrecorded_concurrency_is_could_not_check(self):
        result = t3.run_t3(request(stability_evidence=self._stability(
            mixed_prefill_decode_exercised=None)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is unrecorded", reasons_of(result))


# =============================================================================
# Phase 7 — capacity and the §1.6 per-phase objective
# =============================================================================

def standings(overrides=None) -> tuple:
    table = {("llama_cpu", "prefill"): t3.STANDING_IMPROVED,
             ("llama_cpu", "decode"): t3.STANDING_NON_INFERIOR,
             ("llama_gpu", "prefill"): t3.STANDING_IMPROVED,
             ("llama_gpu", "decode"): t3.STANDING_NON_INFERIOR}
    table.update(overrides or {})
    protocols = {"prefill": "P-BENCH-PREFILL-1", "decode": "P-BENCH-1"}
    return tuple(
        t3.PhaseStanding(backend=backend, workload_phase=phase,
                         protocol_id=protocols[phase], standing=standing,
                         cell_ids=(f"{backend}.{phase}",),
                         evidence_ref=f"standing:{backend}.{phase}")
        for (backend, phase), standing in sorted(table.items()))


class TestObjective(unittest.TestCase):

    def test_regression_without_an_exception_fails(self):
        result = t3.run_t3(request(standings=standings(
            {("llama_cpu", "decode"): t3.STANDING_REGRESSED})))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("no operator-", reasons_of(result))

    def test_regression_with_a_predeclared_approved_exception_is_admitted(self):
        trade = t3.PhaseTradeException(
            backend="llama_cpu", regressing_phase="decode",
            regression_band=(-0.03, -0.01), gaining_phase="prefill",
            expected_gain=0.18, roles_affected=("worker_general",),
            declared_at=CAMPAIGN_START, campaign_start_at=CAMPAIGN_START,
            operator_approved=True, approved_by="operator")
        result = t3.run_t3(request(
            standings=standings({("llama_cpu", "decode"): t3.STANDING_REGRESSED}),
            phase_trades=(trade,)))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_an_exception_declared_after_the_campaign_started_is_not_predeclared(self):
        trade = t3.PhaseTradeException(
            backend="llama_cpu", regressing_phase="decode",
            regression_band=(-0.03, -0.01), gaining_phase="prefill",
            expected_gain=0.18, roles_affected=("worker_general",),
            declared_at="2026-08-02T00:00:00Z", campaign_start_at=CAMPAIGN_START,
            operator_approved=True, approved_by="operator")
        result = t3.run_t3(request(
            standings=standings({("llama_cpu", "decode"): t3.STANDING_REGRESSED}),
            phase_trades=(trade,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("rationalisation", reasons_of(result))

    def test_an_unapproved_exception_is_a_controller_decision_and_fails(self):
        trade = t3.PhaseTradeException(
            backend="llama_cpu", regressing_phase="decode",
            regression_band=(-0.03, -0.01), gaining_phase="prefill",
            expected_gain=0.18, roles_affected=("worker_general",),
            declared_at=CAMPAIGN_START, campaign_start_at=CAMPAIGN_START,
            operator_approved=False, approved_by="controller")
        result = t3.run_t3(request(
            standings=standings({("llama_cpu", "decode"): t3.STANDING_REGRESSED}),
            phase_trades=(trade,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("not a controller one", reasons_of(result))

    def test_an_unbounded_regression_band_is_refused(self):
        with self.assertRaises(t3.T3InputError):
            t3.PhaseTradeException(
                backend="llama_cpu", regressing_phase="decode",
                regression_band=(float("-inf"), 0.0), gaining_phase="prefill",
                expected_gain=0.1, roles_affected=("r",), declared_at=CAMPAIGN_START,
                campaign_start_at=CAMPAIGN_START, operator_approved=True,
                approved_by="operator")

    def test_no_phase_improving_fails(self):
        result = t3.run_t3(request(standings=standings(
            {("llama_cpu", "prefill"): t3.STANDING_NON_INFERIOR,
               ("llama_gpu", "prefill"): t3.STANDING_NON_INFERIOR})))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("no phase improved", reasons_of(result))

    def test_indeterminate_is_not_non_inferior(self):
        result = t3.run_t3(request(standings=standings(
            {("llama_cpu", "decode"): t3.STANDING_INDETERMINATE})))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is not \"not worse\"", reasons_of(result))

    def test_a_missing_phase_standing_is_not_a_non_inferior_one(self):
        keep = [s for s in standings() if s.cell_ids != ("llama_gpu.decode",)]
        result = t3.run_t3(request(standings=tuple(keep)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("an unmeasured phase is not a non-inferior one", reasons_of(result))

    def test_capacity_floor_breach_fails(self):
        floor = t3.CapacityFloor(cell_id="llama_gpu.capacity_utility",
                                 quantity="context_tokens", floor=32768.0,
                                 observed=16384.0, direction="higher_better",
                                 unit=" tok")
        result = t3.run_t3(request(capacity_floors=(floor,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("outside its fixed floor", reasons_of(result))


# =============================================================================
# Phase 1 — the §3.2 per-backend unchanged test
# =============================================================================

class TestBackendUnchanged(unittest.TestCase):

    def _drop_gpu(self, receipt=True):
        views = {
            "llama_cpu": t3.UnchangedView(
                backend="llama_cpu", may_drop_cells=False,
                unchanged_outcome=schemas.FAIL, agreement_outcome=schemas.PASS,
                stage2_ran=True, reasons=("the closure changed",)),
            "llama_gpu": t3.UnchangedView(
                backend="llama_gpu", may_drop_cells=True,
                unchanged_outcome=schemas.PASS, agreement_outcome=schemas.PASS,
                stage2_ran=True),
        }
        receipts = {}
        if receipt:
            receipts["llama_gpu"] = t3.TransferReceipt(
                backend="llama_gpu",
                incumbent_artifacts=(("data/ak/v8/gpu-matrix.json", digest("gpu-matrix")),),
                incumbent_evidence_refs=("P-GPU-1 v8 production matrix",),
                unchanged_digest=digest("unchanged:llama_gpu"),
                incumbent_commit=V8_HEAD)
        return views, receipts

    def test_unchanged_backend_drops_its_cells_with_a_receipt(self):
        views, receipts = self._drop_gpu()
        # Remove every llama_gpu evidence input: a dropped backend owes none.
        base = request()
        result = t3.run_t3(request(
            backend_unchanged=views, transfer_receipts=receipts,
            linkage_receipts=(linkage_receipt("llama_cpu"),),
            backend_inventories=(base.backend_inventories[0],),
            determinism=(base.determinism[0],),
            quality_evidence=(base.quality_evidence[0],),
            stability_evidence=(base.stability_evidence[0],),
            standings=standings()[:2]))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))
        self.assertEqual(result.products.dropped_backends, ("llama_gpu",))

    def test_dropping_without_a_receipt_is_an_unaudited_hole(self):
        views, _ = self._drop_gpu(receipt=False)
        result = t3.run_t3(request(backend_unchanged=views))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("unaudited hole in the matrix", reasons_of(result))

    def test_a_stage_disagreement_is_a_hard_finding(self):
        views = {
            "llama_cpu": t3.UnchangedView(
                backend="llama_cpu", may_drop_cells=False,
                unchanged_outcome=schemas.FAIL, agreement_outcome=schemas.FAIL,
                stage2_ran=True,
                findings=({"code": "STAGE_DISAGREEMENT_SOURCE_CLEAN_BINARY_DIFFERS",
                           "detail": "the closure is wrong or the build is "
                                     "non-deterministic"},)),
            "llama_gpu": request().backend_unchanged["llama_gpu"],
        }
        result = t3.run_t3(request(backend_unchanged=views))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("build-identity finding", reasons_of(result))

    def test_a_missing_unchanged_result_means_full_evidence_is_owed(self):
        views = {"llama_cpu": request().backend_unchanged["llama_cpu"]}
        result = t3.run_t3(request(backend_unchanged=views))
        self.assertIn("llama_gpu", result.products.evidence_owed_backends)
        self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_may_drop_cells_cannot_be_asserted_over_a_failing_unchanged_verdict(self):
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.UnchangedView(backend="llama_cpu", may_drop_cells=True,
                             unchanged_outcome=schemas.FAIL,
                             agreement_outcome=schemas.PASS, stage2_ran=True)
        self.assertIn("asserting the conclusion", str(ctx.exception))

    def test_a_transfer_receipt_from_another_incumbent_is_not_a_transfer(self):
        views, receipts = self._drop_gpu()
        receipts["llama_gpu"] = t3.TransferReceipt(
            backend="llama_gpu",
            incumbent_artifacts=(("data/ak/v7/gpu.json", digest("v7-gpu")),),
            incumbent_evidence_refs=("P-GPU-1 v7 matrix",),
            unchanged_digest=digest("u"), incumbent_commit=V7_HEAD)
        result = t3.run_t3(request(backend_unchanged=views, transfer_receipts=receipts))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("not transferred evidence", reasons_of(result))

    def test_a_scratch_cited_transfer_receipt_is_refused(self):
        views, receipts = self._drop_gpu()
        receipts["llama_gpu"] = t3.TransferReceipt(
            backend="llama_gpu",
            incumbent_artifacts=(("/mnt/raid0/llm/tmp/gpu-matrix.json", digest("t")),),
            incumbent_evidence_refs=("scratch",), unchanged_digest=digest("u"),
            incumbent_commit=V8_HEAD)
        result = t3.run_t3(request(backend_unchanged=views, transfer_receipts=receipts))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("scratch citation", reasons_of(result))


# =============================================================================
# Phase 8 — the transaction dry run and the §10.5 incumbent archive
# =============================================================================

class TestTransaction(unittest.TestCase):

    def test_version_must_move_past_production(self):
        result = t3.run_t3(request(transaction=transaction(
            next_branch="production-consolidated-v8", next_version_number=8,
            next_tag="production-consolidated-v8")))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("never patched in place", reasons_of(result))

    def test_a_lower_version_number_is_refused(self):
        result = t3.run_t3(request(transaction=transaction(
            next_branch="production-consolidated-v7", next_version_number=7,
            next_tag="production-consolidated-v7")))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("does not exceed the incumbent", reasons_of(result))

    def test_branch_and_version_number_must_agree(self):
        result = t3.run_t3(request(transaction=transaction(next_version_number=11)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("disagree", reasons_of(result))

    def test_a_transaction_that_installs_nothing_is_not_a_transaction(self):
        result = t3.run_t3(request(transaction=transaction(symlink_diff=())))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("installs nothing", reasons_of(result))

    def test_a_no_op_symlink_entry_hides_which_links_move(self):
        result = t3.run_t3(request(transaction=transaction(symlink_diff=(
            ("/mnt/raid0/llm/kernels/production/cpu", "/a/b", "/a/b"),))))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("where it\nalready points".replace("\n", " "),
                      reasons_of(result).replace("\n", " "))

    def test_rollback_anchor_must_match_the_archive(self):
        result = t3.run_t3(request(transaction=transaction(rollback_head=V7_HEAD)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is not the\narchived N-1 commit".replace("\n", " "),
                      reasons_of(result).replace("\n", " "))

    def test_scratch_receipt_paths_are_refused(self):
        result = t3.run_t3(request(transaction=transaction(
            receipt_paths=("/mnt/raid0/llm/tmp/v9-freeze/",))))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("scratch citation", reasons_of(result))

    def test_a_transaction_that_drafts_no_era_action_fails(self):
        result = t3.run_t3(request(transaction=transaction(era_actions=())))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("drafts no era action", reasons_of(result))


class TestIncumbentArchive(unittest.TestCase):
    """§10.5 — *"Incumbent builds are archived, not merely rebuildable."*"""

    def test_no_archive_and_no_reason_fails(self):
        result = t3.run_t3(request(archive=t3.IncumbentArchive()))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("/mnt/raid0/llm/kernels/archive/ is empty", reasons_of(result))

    def test_a_rebuild_is_not_an_archive(self):
        result = t3.run_t3(request(archive=archive(rebuilt=True)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is a REBUILD, not an archive", reasons_of(result))

    def test_binaries_without_libraries_are_refused(self):
        with self.assertRaises(t3.T3InputError):
            archive(libraries=())

    def test_a_scratch_archive_root_is_refused(self):
        result = t3.run_t3(request(archive=archive(
            archive_root="/mnt/raid0/llm/tmp/archive/v8")))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("one sweep away", reasons_of(result))

    def test_an_archive_inside_the_frozen_tree_is_not_a_rollback_target(self):
        result = t3.run_t3(request(archive=archive(
            archive_root="/mnt/raid0/llm/llama.cpp/archive")))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("lives in the thing it protects against", reasons_of(result))

    def test_absent_n2_is_a_note_not_a_gate(self):
        result = t3.run_t3(request())
        seal_notes = result.phase(t3.PHASE_TRANSACTION_DRY_RUN).notes
        self.assertTrue(any("N-2 is not archived" in n for n in seal_notes))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_a_first_freeze_with_no_incumbent_has_no_rollback_anchor(self):
        result = t3.run_t3(request(
            archive=t3.IncumbentArchive(
                no_incumbent_reason="first freeze of this tree"),
            transaction=transaction(rollback_branch=None, rollback_head=None)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("no rollback target", reasons_of(result))


# =============================================================================
# §10.6 — the diff-complexity ceiling
# =============================================================================

class TestComplexityCeiling(unittest.TestCase):

    def test_an_unassessed_diff_has_not_cleared_a_ceiling(self):
        result = t3.run_t3(request(complexity={}))
        self.assertTrue(result.requires_human_code_review)
        self.assertIn(integrity.REQUIRES_HUMAN_CODE_REVIEW, result.first_page_notice)

    def test_marking_does_not_fail_the_gate(self):
        assessment = integrity.ComplexityAssessment(
            requires_human_code_review=True,
            reasons=("change_class is 'core_header'",),
            first_page_notice="REQUIRES_HUMAN_CODE_REVIEW — core_header",
            measured={"total_changed_lines": 12})
        result = t3.run_t3(request(
            complexity={b: assessment for b in LLAMA_BACKENDS}))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))
        self.assertTrue(result.requires_human_code_review)
        self.assertTrue(result.bundle.payload["requires_human_code_review"])
        self.assertIn("core_header", result.bundle.payload["first_page_notice"])


# =============================================================================
# §9.1 / §12 — sealed-fingerprint idempotence and failed-gate cooldown
# =============================================================================

class TestFingerprintAndRerun(unittest.TestCase):

    def test_fingerprint_facets_are_enumerated_not_implied(self):
        self.assertEqual(len(t3.FINGERPRINT_FACETS), 15)
        self.assertIn("active_waiver_sha256", t3.FINGERPRINT_FACETS)

    def test_run_id_and_timestamp_do_not_perturb_the_fingerprint(self):
        first = request().fingerprint()
        second = request(run_id="akt3-v9-002", now="2026-08-03T18:00:00Z").fingerprint()
        self.assertEqual(first, second)

    def test_adding_a_waiver_is_evidence_affecting(self):
        cells, results = failing_matrix()
        without = request(_cells=cells, _results=results).fingerprint()
        with_waiver = request(_cells=cells, _results=results,
                              waivers=(waiver_binding(),)).fingerprint()
        self.assertNotEqual(without, with_waiver)

    def test_a_sealed_pass_cannot_be_recomputed(self):
        base = request()
        ledger = (t3.T3Attempt(fingerprint=base.fingerprint(), verdict="PASS",
                               completed_at="2026-08-02T00:00:00Z",
                               bundle_sha256=digest("bundle")),)
        with self.assertRaises(t3.RerunRefused) as ctx:
            t3.run_t3(request(attempt_ledger=ledger))
        self.assertIn(t3.RERUN_REFUSED_ALREADY_SEALED, str(ctx.exception))

    def test_a_failed_gate_cannot_be_re_entered_unchanged(self):
        base = request()
        ledger = (t3.T3Attempt(
            fingerprint=base.fingerprint(), verdict="FAIL",
            completed_at="2026-08-02T00:00:00Z", bundle_sha256=digest("bundle"),
            failed_phases=(t3.PHASE_PERFORMANCE_MATRIX,)),)
        with self.assertRaises(t3.RerunRefused) as ctx:
            t3.run_t3(request(attempt_ledger=ledger))
        self.assertIn(t3.RERUN_REFUSED_UNCHANGED_FINGERPRINT, str(ctx.exception))

    def test_cooldown_blocks_a_repair_that_is_too_soon(self):
        base = request()
        fingerprint = base.fingerprint()
        ledger = (t3.T3Attempt(
            fingerprint=fingerprint, verdict="FAIL",
            completed_at="2026-08-03T11:00:00Z", bundle_sha256=digest("bundle"),
            failed_phases=(t3.PHASE_PERFORMANCE_MATRIX,)),)
        repair = t3.StageRepair(prior_fingerprint=fingerprint,
                                repaired_phase=t3.PHASE_PERFORMANCE_MATRIX,
                                deterministic_replay=True, repair_ref="replay-1")
        disposition = t3.check_rerun(fingerprint, ledger, now=NOW,
                                     cooldown_seconds=86400, repair=repair)
        self.assertFalse(disposition.admissible)
        self.assertEqual(disposition.code, t3.RERUN_REFUSED_COOLDOWN)

    def test_a_deterministic_repair_after_the_cooldown_is_admitted(self):
        base = request()
        fingerprint = base.fingerprint()
        ledger = (t3.T3Attempt(
            fingerprint=fingerprint, verdict="FAIL",
            completed_at="2026-08-01T00:00:00Z", bundle_sha256=digest("bundle"),
            failed_phases=(t3.PHASE_PERFORMANCE_MATRIX,)),)
        repair = t3.StageRepair(prior_fingerprint=fingerprint,
                                repaired_phase=t3.PHASE_PERFORMANCE_MATRIX,
                                deterministic_replay=True, repair_ref="replay-1")
        result = t3.run_t3(request(attempt_ledger=ledger, stage_repair=repair))
        self.assertEqual(result.rerun.code, t3.RERUN_ADMITTED_AFTER_REPAIR)
        self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_a_repair_that_re_measures_is_a_new_run(self):
        fingerprint = request().fingerprint()
        ledger = (t3.T3Attempt(fingerprint=fingerprint, verdict="FAIL",
                               completed_at="2026-08-01T00:00:00Z",
                               bundle_sha256=digest("b"),
                               failed_phases=(t3.PHASE_PERFORMANCE_MATRIX,)),)
        repair = t3.StageRepair(prior_fingerprint=fingerprint,
                                repaired_phase=t3.PHASE_PERFORMANCE_MATRIX,
                                deterministic_replay=False, repair_ref="rerun")
        disposition = t3.check_rerun(fingerprint, ledger, now=NOW,
                                     cooldown_seconds=3600, repair=repair)
        self.assertFalse(disposition.admissible)
        self.assertIn("must present itself as one", disposition.reason)

    def test_repairing_a_stage_that_did_not_fail_is_refused(self):
        fingerprint = request().fingerprint()
        ledger = (t3.T3Attempt(fingerprint=fingerprint, verdict="FAIL",
                               completed_at="2026-08-01T00:00:00Z",
                               bundle_sha256=digest("b"),
                               failed_phases=(t3.PHASE_QUALITY,)),)
        repair = t3.StageRepair(prior_fingerprint=fingerprint,
                                repaired_phase=t3.PHASE_STABILITY,
                                deterministic_replay=True, repair_ref="r")
        disposition = t3.check_rerun(fingerprint, ledger, now=NOW,
                                     cooldown_seconds=3600, repair=repair)
        self.assertFalse(disposition.admissible)
        self.assertIn("leaves the one that did untouched", disposition.reason)

    def test_a_new_fingerprint_is_admitted(self):
        ledger = (t3.T3Attempt(fingerprint=digest("some other run"), verdict="FAIL",
                               completed_at="2026-08-01T00:00:00Z",
                               bundle_sha256=digest("b")),)
        result = t3.run_t3(request(attempt_ledger=ledger))
        self.assertEqual(result.rerun.code, t3.RERUN_ADMITTED_NEW_FINGERPRINT)

    def test_there_is_no_default_cooldown(self):
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.check_rerun(digest("f"), (), now=NOW, cooldown_seconds=0)
        self.assertIn("a policy nobody declared", str(ctx.exception))

    def test_there_is_no_force_argument(self):
        import inspect
        params = set(inspect.signature(t3.check_rerun).parameters)
        self.assertFalse({"force", "override", "skip"} & params)


# =============================================================================
# The `plan.py` seam
# =============================================================================

class TestPlanSeam(unittest.TestCase):
    """T3 builds against the release-plan compiler; it never reimplements it."""

    def test_unchanged_results_are_read_from_the_compiled_plan(self):
        from . import plan as plan_module

        backend_plan = plan_module.BackendPlan(
            backend="llama_gpu", binding_ref={}, cells=(), transfer_receipt=None,
            co_residency_group=None, affected_ops=(), uncovered_ops=(),
            canary_roles=(), findings=(), checks={},
            unchanged_ref={"backend": "llama_gpu", "may_drop_cells": True,
                           "unchanged": {"outcome": schemas.PASS, "reasons": []},
                           "agreement": {"outcome": schemas.PASS, "reasons": []},
                           "stage2": {"stage": "normalized_binary_identity"},
                           "findings": [], "blocking_reasons": []})
        compiled = type("FakePlan", (), {"backends": (backend_plan,)})()
        views = t3.unchanged_results_from_plan(compiled)
        self.assertEqual(views["llama_gpu"].may_drop_cells, True)
        self.assertTrue(views["llama_gpu"].stage2_ran)

    def test_a_plan_with_no_unchanged_result_must_say_so(self):
        from . import plan as plan_module

        backend_plan = plan_module.BackendPlan(
            backend="llama_cpu", binding_ref={}, cells=(), transfer_receipt=None,
            co_residency_group=None, affected_ops=(), uncovered_ops=(),
            canary_roles=(), findings=(), checks={})
        compiled = type("FakePlan", (), {"backends": (backend_plan,)})()
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.unchanged_results_from_plan(compiled)
        self.assertIn("inferring it from an absence", str(ctx.exception))

    def test_transfer_receipts_carry_the_incumbent_artifacts_across(self):
        from . import plan as plan_module

        incumbent = plan_module.IncumbentEvidence(
            backend="llama_gpu", era_id="E8-gpu",
            artifacts=(("data/ak/v8/gpu-matrix.json", digest("gpu-matrix")),),
            protocol_ids=("P-GPU-1",),
            archive_path="/mnt/raid0/llm/kernels/archive/v8")
        receipt = plan_module.TransferReceipt(
            backend="llama_gpu", production_base_commit=BASE_COMMIT,
            candidate_commit=CANDIDATE_COMMIT,
            unchanged_result={"backend": "llama_gpu"}, incumbent=incumbent.to_dict(),
            dropped_cell_ids=("llama_gpu.decode",), dropped_cell_count=1)
        backend_plan = plan_module.BackendPlan(
            backend="llama_gpu", binding_ref={}, cells=(), transfer_receipt=receipt,
            co_residency_group=None, affected_ops=(), uncovered_ops=(),
            canary_roles=(), findings=(), checks={})
        compiled = type("FakePlan", (), {"backends": (backend_plan,)})()
        adapted = t3.transfer_receipts_from_plan(compiled, incumbent_commit=V8_HEAD)
        self.assertEqual(adapted["llama_gpu"].incumbent_artifacts,
                         (("data/ak/v8/gpu-matrix.json", digest("gpu-matrix")),))
        self.assertEqual(adapted["llama_gpu"].check().outcome, schemas.PASS)

    def test_release_plan_view_refuses_a_shape_it_cannot_read(self):
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.release_plan_view(object())
        self.assertIn("not a release plan T3 can read", str(ctx.exception))

    def test_release_plan_view_accepts_a_mapping(self):
        view = request().plan
        rebuilt = t3.release_plan_view({
            "plan_id": view.plan_id, "plan_sha256": view.plan_sha256,
            "source_tree": view.source_tree, "backends": view.backends,
            "cells": view.cells, "incumbent_branch": view.incumbent_branch,
            "incumbent_commit": view.incumbent_commit,
            "incumbent_version_number": view.incumbent_version_number})
        self.assertEqual(rebuilt.plan_sha256, view.plan_sha256)


# =============================================================================
# §10.4 CALIBRATION — the preserved v8 and speech freeze artifacts
# =============================================================================

class TestPreservedArtifactFixtures(unittest.TestCase):
    """The embedded fixtures must not drift from the artifacts they quote."""

    def _load(self, path):
        if not path.is_file():
            self.skipTest(f"{path} is not present on this host")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_v8_fixture_matches_the_preserved_attestation(self):
        real = self._load(V8_RATIFICATION_PATH)
        self.assertEqual(real["production_head"], V8_HEAD)
        self.assertEqual(real["rollback"]["head"], V7_HEAD)
        self.assertIs(real["promotion_decision"], False)
        self.assertEqual(real["evidence_sha256"]["waive_q8"], V8_WAIVER_SHA)
        self.assertEqual(real["production_binary_sha256"]["cpu"], V8_CPU_BINARY)
        self.assertEqual(
            real["production_lineup_gate"]["quality_contract"]["baseline_binary"],
            V7_BASELINE_BINARY)

    def test_v8_waiver_fixture_matches_the_preserved_waiver(self):
        real = self._load(V8_WAIVER_PATH)
        self.assertEqual(real["schema"], t3.WAIVER_SCHEMA_V8_CPU_PREFILL)
        self.assertEqual(real["scope"]["excluded_pairs"],
                         V8_WAIVER["scope"]["excluded_pairs"])
        self.assertEqual(real["consequences"][0], V8_WAIVER["consequences"][0])
        self.assertIs(real["protocol_changed"], False)

    def test_speech_fixture_matches_the_preserved_ratification(self):
        real = self._load(SPEECH_RATIFICATION_PATH)
        self.assertEqual(real["kernels"]["whisper_cpp"]["commit"],
                         SPEECH_RATIFICATION["kernels"]["whisper_cpp"]["commit"])
        self.assertEqual(real["kernels"]["qwentts_cpp"]["binary_sha256"],
                         SPEECH_RATIFICATION["kernels"]["qwentts_cpp"]["binary_sha256"])

    def test_the_preserved_attestation_records_no_baseline_binary_hash(self):
        """The §10.5 hole, asserted against the artifact rather than described."""
        real = self._load(V8_RATIFICATION_PATH)
        contract = real["production_lineup_gate"]["quality_contract"]
        self.assertNotIn("baseline_binary_sha256", contract)


class TestCalibrationV8(unittest.TestCase):
    """§10.4: *"the T3 dry-run against preserved v8 artifacts should predict a FAIL
    without the waiver. If it passes, the compiler is wrong."*"""

    def setUp(self):
        self.freeze = t3.preserved_freeze_from_v8_artifacts(V8_RATIFICATION, V8_WAIVER)

    def test_the_reader_extracts_the_freeze_identity(self):
        self.assertEqual(self.freeze.source_tree, "llama.cpp")
        self.assertEqual(self.freeze.backends, ("llama_cpu", "llama_gpu"))
        self.assertEqual(self.freeze.production_head, V8_HEAD)
        self.assertEqual(self.freeze.rollback_head, V7_HEAD)
        self.assertIs(self.freeze.promotion_decision, False)
        self.assertEqual(self.freeze.excluded_pairs,
                         ("qwen36_q8-tg128-iqk1", "qwen36_q8-pp2048-iqk1"))

    def test_the_reader_refuses_a_document_that_is_not_the_attestation(self):
        with self.assertRaises(t3.T3InputError):
            t3.preserved_freeze_from_v8_artifacts({"schema": "something.else"})

    def test_the_dry_run_FAILS_without_the_waiver(self):
        result = t3.run_t3(t3.calibration_request(
            self.freeze, now=NOW, include_waiver=False))
        self.assertEqual(result.verdict, "FAIL")
        failed = set(result.verdict_computation.failed_cells)
        self.assertIn("llama_cpu.pair.qwen36_q8-tg128-iqk1", failed)
        self.assertIn("llama_cpu.pair.qwen36_q8-pp2048-iqk1", failed)

    def test_the_failure_names_the_eligibility_floor_and_the_archive_hole(self):
        result = t3.run_t3(t3.calibration_request(
            self.freeze, now=NOW, include_waiver=False))
        text = " | ".join(result.verdict_computation.blocking_reasons)
        self.assertIn("records NO sha256 for it", text)
        self.assertIn("no rollback target", text)
        matrix = result.phase(t3.PHASE_PERFORMANCE_MATRIX)
        floor_reasons = [r for c in matrix.cell_results for r in c.check.reasons
                         if "eligibility floor" in r]
        self.assertEqual(len(floor_reasons), 2, floor_reasons)

    def test_the_waiver_alone_does_not_clear_the_archive_hole(self):
        """A waiver covers cells; it never covers the integrity spine."""
        result = t3.run_t3(t3.calibration_request(
            self.freeze, now=NOW, include_waiver=True))
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(result.verdict_computation.failed_cells, ())
        self.assertIn("no rollback target",
                      " | ".join(result.verdict_computation.blocking_reasons))

    def test_with_the_waiver_and_an_archive_the_run_is_pass_with_waiver(self):
        preserved_v7 = t3.IncumbentArchive(entries=(t3.ArchivedBuild(
            generation=t3.ARCHIVE_GENERATION_N1,
            branch="production-consolidated-v7", commit=V7_HEAD,
            archive_root="/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff",
            binaries=((V7_BASELINE_BINARY, digest("v7-llama-server")),),
            libraries=(("/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/"
                        "libggml-base.so.0", digest("v7-libggml-base")),)),))
        request_with_archive = t3.calibration_request(
            self.freeze, now=NOW, include_waiver=True, archive=preserved_v7)
        # The quality baseline must be the archived preserved binary, as v8's was.
        quality = tuple(
            t3.QualityEvidence(
                backend=b, mode=t3.QUALITY_MEASURED_PARITY,
                baseline_binary_path=V7_BASELINE_BINARY,
                baseline_binary_sha256=digest("v7-llama-server"),
                baseline_kernel="production-consolidated-v7",
                baseline_is_rebuild=False,
                evidence_refs=("data/kernel-v8-candidate/quality-gate/",),
                suites=("mmlu_pro", "gpqa"), shared_question_identity=True)
            for b in self.freeze.backends)
        result = t3.run_t3(t3.T3Request(**{
            **{f.name: getattr(request_with_archive, f.name)
               for f in request_with_archive.__dataclass_fields__.values()},
            "quality_evidence": quality}))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER",
                         " | ".join(result.verdict_computation.blocking_reasons))
        self.assertEqual(len(result.receipt.suppressed_claims), 2)
        self.assertIn("No v8 Q8 non-regression claim may be made from this campaign.",
                      result.receipt.forfeited_claims)

    def test_the_suppressed_claims_are_the_q8_pairs(self):
        preserved_v7 = t3.IncumbentArchive(entries=(t3.ArchivedBuild(
            generation=t3.ARCHIVE_GENERATION_N1,
            branch="production-consolidated-v7", commit=V7_HEAD,
            archive_root="/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff",
            binaries=((V7_BASELINE_BINARY, digest("v7-llama-server")),),
            libraries=(("/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/"
                        "libggml-base.so.0", digest("v7-libggml-base")),)),))
        base = t3.calibration_request(self.freeze, now=NOW, include_waiver=True,
                                      archive=preserved_v7)
        quality = tuple(
            t3.QualityEvidence(
                backend=b, mode=t3.QUALITY_MEASURED_PARITY,
                baseline_binary_path=V7_BASELINE_BINARY,
                baseline_binary_sha256=digest("v7-llama-server"),
                baseline_kernel="production-consolidated-v7",
                baseline_is_rebuild=False, evidence_refs=("data/",),
                shared_question_identity=True)
            for b in self.freeze.backends)
        result = t3.run_t3(t3.T3Request(**{
            **{f.name: getattr(base, f.name)
               for f in base.__dataclass_fields__.values()},
            "quality_evidence": quality}))
        claims = {s["claim"] for s in result.receipt.suppressed_claims}
        self.assertEqual(claims, {"qwen36_q8-tg128-iqk1 non-regression",
                                  "qwen36_q8-pp2048-iqk1 non-regression"})
        for claim in claims:
            self.assertNotIn(claim, result.receipt.claims)

    def test_the_calibration_run_is_never_a_release(self):
        result = t3.run_t3(t3.calibration_request(self.freeze, now=NOW))
        self.assertEqual(result.mode, t3.MODE_DRY_RUN)


class TestCalibrationSpeech(unittest.TestCase):
    """The 2026-07-31 speech freeze, replayed through the gate."""

    def setUp(self):
        self.freezes = t3.preserved_freeze_from_speech_artifact(SPEECH_RATIFICATION)

    def test_one_freeze_per_source_tree(self):
        self.assertEqual(set(self.freezes), {"whisper.cpp", "qwentts.cpp"})

    def test_the_dry_run_fails_on_the_uncommitted_load_bearing_patch(self):
        result = t3.run_t3(t3.calibration_request(
            self.freezes["whisper.cpp"], now=NOW))
        self.assertEqual(result.verdict, "FAIL")
        text = " | ".join(result.verdict_computation.blocking_reasons)
        self.assertIn("the candidate tree is not clean", text)
        self.assertIn("uncommitted load-bearing patch", text)

    def test_the_dry_run_fails_on_the_absent_rollback_anchor(self):
        result = t3.run_t3(t3.calibration_request(
            self.freezes["qwentts.cpp"], now=NOW))
        text = " | ".join(result.verdict_computation.blocking_reasons)
        self.assertIn("no rollback target", text)

    def test_the_speech_family_branch_is_accepted(self):
        result = t3.run_t3(t3.calibration_request(
            self.freezes["whisper.cpp"], now=NOW))
        text = " | ".join(result.verdict_computation.blocking_reasons)
        self.assertNotIn("family", text)

    def test_the_notes_record_what_the_artifact_recorded(self):
        freeze = self.freezes["whisper.cpp"]
        self.assertTrue(any("load-bearing patch" in n for n in freeze.notes))
        self.assertTrue(any("no incumbent" in n for n in freeze.notes))


# =============================================================================
# Red-team regressions — one per defect an adversarial pass actually landed
#
# Every test below FAILED against the first implementation, each by making a
# green release out of evidence that does not support one. They are grouped
# rather than scattered so the class reads as what it is: the list of ways this
# gate has already been got past once.
# =============================================================================

class TestWaiverScopeIsTheOperators(unittest.TestCase):
    """A hash-pinned waiver must cover the cells the OPERATOR named, and no others.

    The original `verify_waiver` pinned the document's digest and then took
    `covers_cell_ids` from the BINDING — i.e. from the party being gated — without
    ever consulting the scope block inside the document it had just verified. The
    genuine, byte-identical WAIVE-Q8 attestation, whose scope names two Q8
    model/shape pairs, therefore suppressed anything at all it was pointed at.
    """

    def _run(self, covers, doc=None):
        cells = matrix_cells()
        results = [
            t3.CellResult(
                cell=c,
                check=(t3._fail("the GPU produced NaNs on unseen op shapes")
                       if c.cell_id == "llama_gpu.backend_correctness"
                       else schemas.Check(schemas.PASS)),
                raw_samples_ref=f"data/ak/{c.cell_id}.jsonl",
                reducer_id="median_mad/v1")
            for c in cells]
        binding = waiver_binding(doc, covers_cell_ids=covers)
        return t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))

    def test_a_waiver_cannot_be_pointed_at_a_cell_its_scope_never_named(self):
        """The whole exploit: a real waiver, a real hash, someone else's failure."""
        result = self._run(("llama_gpu.backend_correctness",))
        self.assertEqual(result.verdict, "FAIL", reasons_of(result))
        self.assertIn("llama_gpu.backend_correctness",
                      result.verdict_computation.failed_cells)
        self.assertEqual(result.verdict_computation.waived_cells, ())
        self.assertIn("its own attested scope does not name", reasons_of(result))

    def test_a_waiver_still_covers_the_cell_its_scope_does_name(self):
        """The fix must not forbid the compliant path (`llama_cpu.prefill`)."""
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(waiver_binding(),)))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER", reasons_of(result))

    def test_a_scope_naming_only_the_model_still_resolves_its_cells(self):
        """Operators write model/pair names, not cell ids; both must resolve."""
        doc = autokernel_waiver(scope={"excluded_models": ["llama_cpu"],
                                       "excluded_pairs": [],
                                       "remaining_matched_pairs": 14})
        cells, results = failing_matrix()
        result = t3.run_t3(request(
            _cells=cells, _results=results,
            waivers=(waiver_binding(doc, covers_cell_ids=("llama_cpu.prefill",)),)))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER", reasons_of(result))

    def test_a_waiver_with_no_declared_scope_covers_nothing(self):
        doc = autokernel_waiver()
        doc.pop("scope")
        result = self._run(("llama_gpu.backend_correctness",), doc)
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("declares no resolvable scope", reasons_of(result))

    def test_the_scope_predicate_is_reported_like_every_other_predicate(self):
        cells, results = failing_matrix()
        verification = t3.verify_waiver(
            waiver_binding(covers_cell_ids=("llama_gpu.quality",)),
            candidate_commit=CANDIDATE_COMMIT, production_base_commit=BASE_COMMIT,
            campaign_id="ak-v9", known_cell_ids=[c.cell_id for c in cells],
            failing_cell_ids=[], now=NOW)
        self.assertEqual(verification.predicate_results["scope"], schemas.FAIL)
        self.assertFalse(verification.verified)

    def test_the_scope_predicate_does_not_judge_the_waivers_merits(self):
        """Two waivers differing only in the operator's REASON stay identical."""
        cells = matrix_cells()
        kwargs = dict(candidate_commit=CANDIDATE_COMMIT,
                      production_base_commit=BASE_COMMIT, campaign_id="ak-v9",
                      known_cell_ids=[c.cell_id for c in cells],
                      failing_cell_ids=["llama_cpu.prefill"], now=NOW)
        strong = t3.verify_waiver(
            waiver_binding(autokernel_waiver(reason="a well-argued reason")), **kwargs)
        weak = t3.verify_waiver(
            waiver_binding(autokernel_waiver(reason="because")), **kwargs)
        self.assertEqual(strong.predicate_results, weak.predicate_results)
        self.assertTrue(strong.verified and weak.verified)


class TestLinkageProofIsAnchored(unittest.TestCase):
    """The receipt's tree root must be the candidate's build dir, or inside it.

    Containment was checked the wrong way round: an ANCESTOR was accepted, so
    `expected_tree_root="/"` satisfied it — and `"/"` also satisfies the
    LD_LIBRARY_PATH lead-entry test, which reuses the same root. A llama binary
    whose loader led with `qwentts.cpp/build/bin` (ggml 0.17.0, not 0.16.0) sealed
    a clean PASS: INC-20260731-ggml-linkage-silent-cpu-fallback, verbatim.
    """

    FOREIGN = "/mnt/raid0/llm/qwentts.cpp/build/bin"

    def _receipt(self, root, **over):
        return linkage_receipt(
            "llama_cpu", expected_tree_root=root,
            stdout=f"PASS: all linked ggml libraries resolve inside {root}\n",
            ld_library_path=(self.FOREIGN, "/opt/rocm/lib"), **over)

    def test_a_root_that_merely_contains_the_build_dir_proves_nothing(self):
        for root in ("/", "/mnt", "/mnt/raid0", "/mnt/raid0/llm"):
            with self.subTest(root=root):
                check = self._receipt(root).check(expected_build_dir=BUILD_ROOT)
                self.assertEqual(check.outcome, schemas.FAIL)
                self.assertIn("merely CONTAINS this one", " ".join(check.reasons))

    def test_the_widened_root_no_longer_seals_a_release(self):
        receipts = (self._receipt("/"), linkage_receipt("llama_gpu"))
        result = t3.run_t3(request(linkage_receipts=receipts))
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(result.phase(t3.PHASE_BUILD_LINKAGE).check.outcome, schemas.FAIL)

    def test_a_root_inside_the_build_dir_is_a_narrower_proof_and_is_admitted(self):
        check = linkage_receipt(
            "llama_cpu", expected_tree_root=f"{BUILD_ROOT}/bin",
            stdout=f"PASS: all linked ggml libraries resolve inside {BUILD_ROOT}/bin\n",
            ld_library_path=(f"{BUILD_ROOT}/bin",)).check(expected_build_dir=BUILD_ROOT)
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))

    def test_deleting_the_build_dir_does_not_delete_the_check(self):
        """The check took `expected_build_dir=None` as 'nothing to compare against'."""
        sealed = t3.SealedCandidate(
            candidate_id="akc-v9", source_tree="llama.cpp",
            candidate_branch="llama.cpp-experimental/v9",
            production_base_commit=BASE_COMMIT, candidate_commit=CANDIDATE_COMMIT,
            seal_sha256=digest("seal"), evaluator_bundle_sha256=digest("evaluator"),
            scope_manifest_sha256=digest("scope"),
            evidence_tree_sha256=digest("evidence"),
            binary_sha256={b: digest(f"bin:{b}") for b in LLAMA_BACKENDS},
            linkage_sha256={b: digest(f"link:{b}") for b in LLAMA_BACKENDS},
            build_dirs={}, overlay_present=True, tree_clean=True, ancestry_clean=True)
        receipts = tuple(self._receipt(self.FOREIGN) if b == "llama_cpu"
                         else linkage_receipt(b) for b in LLAMA_BACKENDS)
        result = t3.run_t3(request(sealed=sealed, linkage_receipts=receipts))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("records no build directory", reasons_of(result))


class TestScopeDenominatorRefusesRatherThanShrugs(unittest.TestCase):
    """§7.4: an unreadable scope is not a matching scope.

    Only `FAIL` was acted on, so the HONEST partial-machine declaration failed
    against a full-machine gate while the same cell with `machine_subset` deleted,
    misspelled, or emptied returned COULD_NOT_CHECK and passed. The check was
    defeated by removing what it inspects.
    """

    GATE = {"machine_subset": "full", "cores": 96}

    def _run(self, scope):
        cells = [t3.Cell(**{**{f.name: getattr(c, f.name)
                               for f in dataclasses.fields(c)},
                            "scope_denominator": scope})
                 if c.cell_id == "llama_cpu.decode" else c for c in matrix_cells()]
        return t3.run_t3(request(_cells=cells, _results=cell_results(cells),
                                 gate_scope=self.GATE))

    def test_an_unreadable_scope_denominator_blocks(self):
        cases = {
            "machine_subset omitted": {"numa_nodes": [0], "cores": 48},
            "machine_subset misspelled": {"machine_subset": "PARTIAL", "cores": 48},
            "numa_nodes not a list": {"machine_subset": "partial",
                                      "numa_nodes": "0", "devices": [], "cores": 48},
            "partial naming nothing": {"machine_subset": "partial", "numa_nodes": [],
                                       "devices": [], "cores": 48},
            "cores not an int": {"machine_subset": "full", "cores": "96"},
        }
        for label, scope in cases.items():
            with self.subTest(label):
                raw = schemas.check_scope_denominator_admits_gate(
                    {"scope_denominator": scope}, self.GATE)
                self.assertEqual(raw.outcome, schemas.COULD_NOT_CHECK)
                self.assertEqual(self._run(scope).verdict, "FAIL")

    def test_a_commensurate_scope_still_passes(self):
        result = self._run({"machine_subset": "full", "cores": 96})
        self.assertEqual(result.verdict, "PASS", reasons_of(result))


class TestPerformanceMatrixCannotBeEmptied(unittest.TestCase):
    """Phase 3 refuses a backend with no gating cell; phase 4 has to as well.

    Invariant 15 correctly makes a diagnostic cell inert in both directions — so
    relabelling a backend's whole performance row `diagnostic` removed it from the
    gate along with the rep rule, the protocol-ownership rule and the co-residency
    rule. A `llama_gpu` that regressed 40% on both phases sealed PASS.
    """

    def _relabel(self, cell):
        if (cell.release_phase == t3.PHASE_PERFORMANCE_MATRIX
                and cell.backend == "llama_gpu"):
            fields = {f.name: getattr(cell, f.name) for f in dataclasses.fields(cell)}
            fields.update(recipe_class=t3.RECIPE_DIAGNOSTIC, claim=None, reps=1,
                          protocol_id="P-MADE-UP-9")
            return t3.Cell(**fields)
        return cell

    def test_a_backend_measured_only_diagnostically_has_skipped_the_comparison(self):
        cells = [self._relabel(c) for c in matrix_cells()]
        results = [
            t3.CellResult(
                cell=c,
                check=(t3._fail("llama_gpu decode regressed 40%")
                       if (c.backend == "llama_gpu"
                           and c.release_phase == t3.PHASE_PERFORMANCE_MATRIX)
                       else schemas.Check(schemas.PASS)),
                raw_samples_ref=(None if c.recipe_class == t3.RECIPE_DIAGNOSTIC
                                 else f"data/ak/{c.cell_id}.jsonl"),
                reducer_id=(None if c.recipe_class == t3.RECIPE_DIAGNOSTIC
                            else "median_mad/v1"))
            for c in cells]
        result = t3.run_t3(request(_cells=cells, _results=results))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("no gating performance cell", reasons_of(result))
        self.assertNotIn("llama_gpu decode non-regression vs v8", result.receipt.claims)

    def test_a_diagnostic_cell_beside_a_gating_one_is_still_inert(self):
        """The fix must not turn invariant 15 into 'diagnostic cells now gate'."""
        cells = matrix_cells()
        noise = t3.Cell(
            cell_id="llama_gpu.decode.baseline", backend="llama_gpu",
            release_phase=t3.PHASE_PERFORMANCE_MATRIX, protocol_id="P-BENCH-1",
            recipe_class=t3.RECIPE_DIAGNOSTIC, metric="tokens_per_s",
            metric_direction="higher_better", workload_phase="decode", reps=1)
        cells.append(noise)
        results = cell_results(cells[:-1]) + [
            t3.CellResult(cell=noise, check=t3._fail("the off-recipe baseline regressed"))]
        result = t3.run_t3(request(_cells=cells, _results=results))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_a_dropped_backend_is_not_asked_for_a_performance_cell(self):
        """A §3.2 transfer removes the backend; it must not then owe a cell."""
        cells = [c for c in matrix_cells() if c.backend != "llama_gpu"]
        unchanged = {
            "llama_cpu": t3.UnchangedView(
                backend="llama_cpu", may_drop_cells=False,
                unchanged_outcome=schemas.FAIL, agreement_outcome=schemas.PASS,
                stage2_ran=True, reasons=("the closure changed",)),
            "llama_gpu": t3.UnchangedView(
                backend="llama_gpu", may_drop_cells=True,
                unchanged_outcome=schemas.PASS, agreement_outcome=schemas.PASS,
                stage2_ran=True),
        }
        receipt = t3.TransferReceipt(
            backend="llama_gpu",
            incumbent_artifacts=(("data/ak/v8/llama_gpu.jsonl", digest("v8-gpu")),),
            incumbent_evidence_refs=("v8 gpu matrix",),
            unchanged_digest=digest("unchanged:llama_gpu"), incumbent_commit=V8_HEAD)
        result = t3.run_t3(request(
            _cells=cells, _results=cell_results(cells), backend_unchanged=unchanged,
            transfer_receipts={"llama_gpu": receipt},
            quality_evidence=tuple(q for q in request().quality_evidence
                                   if q.backend == "llama_cpu"),
            stability_evidence=tuple(s for s in request().stability_evidence
                                     if s.backend == "llama_cpu")))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))


class TestTheWriteAuditActuallyAudits(unittest.TestCase):
    """`audit_no_write_or_process_paths` is the module's proof of its cardinal rule.

    `pathlib` is allowed so the module can read its own source — and the four most
    direct pathlib routes to a production write were all unlisted. Each source
    below returned PASS from the audit that exists to forbid exactly it.
    """

    BYPASSES = {
        "Path().open('w') then .write()":
            'from pathlib import Path\n'
            'def cut_over(p, tree):\n'
            '    with Path(p).open("w") as fh:\n'
            '        fh.write(tree)\n',
        "Path().replace() moves a stable kernel symlink":
            'from pathlib import Path\n'
            'def cut_over(link, new):\n'
            '    Path(new).replace(link)\n',
        "Path().hardlink_to() is symlink_to under another name":
            'from pathlib import Path\n'
            'def cut_over(link, target):\n'
            '    Path(link).hardlink_to(target)\n',
        "Path().unlink() then Path().symlink_to()":
            'from pathlib import Path\n'
            'def cut_over(link, target):\n'
            '    Path(link).unlink()\n'
            '    Path(link).symlink_to(target)\n',
        "getattr routes around the attribute denylist":
            'from pathlib import Path\n'
            'def cut_over(p, tree):\n'
            '    getattr(Path(p), "write_text")(tree)\n',
    }

    def test_every_known_write_path_is_refused(self):
        for label, source in self.BYPASSES.items():
            with self.subTest(label):
                check = t3.audit_no_write_or_process_paths(source)
                self.assertEqual(check.outcome, schemas.FAIL, label)

    def test_the_module_still_passes_its_own_audit(self):
        """The guard must not be satisfiable only by exempting its own call sites."""
        check = t3.audit_no_write_or_process_paths()
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))

    def test_reading_is_still_allowed(self):
        check = t3.audit_no_write_or_process_paths(
            'from pathlib import Path\n'
            'def read(p):\n'
            '    return Path(p).read_text(encoding="utf-8")\n')
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))

    def test_duck_typed_getattr_is_not_collateral_damage(self):
        """T3's plan adapters live on `getattr`; only denied NAMES are refused."""
        check = t3.audit_no_write_or_process_paths(
            'def adapt(cell):\n'
            '    return getattr(cell, "protocol", None)\n')
        self.assertEqual(check.outcome, schemas.PASS, list(check.reasons))


class TestNoUndeclaredReleaseThreshold(unittest.TestCase):
    """`stability_min_cycles` defaulted to 1, so one load/unload cleared a floor
    described in the failure text as "the declared release minimum" that nobody had
    declared. `check_rerun` refuses a defaulted cooldown for this exact reason."""

    def test_the_stability_floor_must_be_declared(self):
        fields = {f.name: getattr(request(), f.name)
                  for f in dataclasses.fields(t3.T3Request)
                  if f.name != "stability_min_cycles"}
        with self.assertRaises(t3.T3InputError) as caught:
            t3.T3Request(**fields)
        self.assertIn("no default", str(caught.exception))

    def test_a_declared_floor_still_gates(self):
        stability = tuple(
            t3.StabilityEvidence(
                backend=b, load_unload_cycles=1, memory_growth_bytes=0,
                memory_growth_allowance_bytes=0, profiler_or_runtime_errors=0,
                cleanup_verified=True, mixed_prefill_decode_exercised=True,
                evidence_ref=f"stability:{b}")
            for b in LLAMA_BACKENDS)
        result = t3.run_t3(request(stability_evidence=stability,
                                   stability_min_cycles=5))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("below the declared release minimum of 5", reasons_of(result))


if __name__ == "__main__":
    unittest.main()

