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

import atexit
import dataclasses
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from .. import schemas, storage
from ..adapters import whisper_stt
from . import preflight as guards
from ..evaluator import api, integrity
from . import packager, t3

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

#: §1.6's phase vocabulary for the llama pair, and the Annex B protocol that owns
#: each phase. Both are RATIFIED, so the default fixture binds them as such — a
#: fixture that left them as bare ids would be exercising the UNBOUND path in every
#: test at once and hiding the one it is supposed to exercise.
LLAMA_PHASE_PROTOCOLS = {"prefill": "P-BENCH-PREFILL-1", "decode": "P-BENCH-1"}


def ratified_protocol(protocol_id: str, **overrides) -> t3.ProtocolBinding:
    fields = {"protocol_id": protocol_id,
              "document_sha256": digest(f"protocol:{protocol_id}"),
              "ratified": True, "ratified_at": "2026-05-01T00:00:00Z", "annex": "B"}
    fields.update(overrides)
    return t3.ProtocolBinding(**fields)


def draft_protocol(protocol_id: str) -> t3.ProtocolBinding:
    return t3.ProtocolBinding(
        protocol_id=protocol_id, document_sha256=digest(f"draft:{protocol_id}"),
        ratified=False)


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
        # One `libggml-base.so.0` genuinely serves both llama backends of one tree,
        # which is why the attribution is a SET rather than a name.
        "libraries": ((LLAMA_BACKENDS,
                       "/mnt/raid0/llm/kernels/archive/v8/cpu/libggml-base.so.0",
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
        # This suite cannot write to `/workspace/artifacts/operator/` — that is the
        # operator's tree, and the reader's DEFAULT root — so every read waiver it
        # builds comes from `_TEST_ATTESTATION_ROOT`, an `artifacts/operator/`
        # directory inside this checkout that the suite creates with `mkdir -p`.
        # That is exactly the location `verify_waiver` now refuses by default, and
        # declaring it here is the point: the widening is stated by the RUN, in the
        # request, where the fingerprint hashes it, rather than being handed to the
        # reader as a keyword nobody downstream ever sees. Production sets nothing
        # and gets the real root. `TestTheGateHoldsItsOwnAttestationRoots` in
        # `test_t3_waiver_reader_redteam.py` is the control that this default is a
        # test seam and not a hole.
        "attestation_roots": (str(_TEST_ATTESTATION_ROOT),),
        "supplied_components": {name: digest(f"component:{name}")
                                for name in t3.SUPPLIED_COMPONENTS},
        "cooldown_seconds": 86400,
        "release_reps_by_protocol": {"P-BENCH-1": 10, "P-BENCH-PREFILL-1": 5,
                                     "P-KERNEL-FREEZE-1": 1},
        "phase_protocols": {b: {phase: ratified_protocol(protocol)
                                for phase, protocol in LLAMA_PHASE_PROTOCOLS.items()}
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


def quoted_waiver(document=None, **overrides) -> t3.WaiverBinding:
    """A waiver as somebody QUOTED it. No file exists behind it, and that is the point.

    Every fixture built here is the defect in miniature: a document the caller
    invented, a digest over that document, and a path nothing checks. It must never
    suppress anything.
    """
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


#: Where these tests put a waiver they intend to be READ. It must be a real
#: operator-SHAPED citation (`artifacts/operator/…` under a checkout root) or
#: `schemas.operator_owned_path_check` refuses it before any I/O — which is the
#: correct behaviour and the reason the fixture cannot use `/tmp`, a scratch root the
#: reader refuses twice over. The reader's DEFAULT attestation root is
#: `/workspace/artifacts/operator`, which is operator-owned and which these tests
#: therefore never write to; they declare this root explicitly instead, and
#: `TestTheDeclaredAttestationRootNarrowsTheCitationCheck` proves the default refuses
#: this very directory.
_TEST_ATTESTATION_ROOT = storage.REPO_ROOT / "artifacts" / "operator"
_WAIVER_FILE_DIR = None
_LIVE_HUMAN_ONLY_BOUNDARY = t3.human_only_boundary


def _fixture_boundary() -> schemas.TrustBoundary:
    """Test-only authority over this suite's disposable fixture directory.

    The 2026-08-10 checkout-name hardening correctly stopped treating arbitrary
    worktrees as repositories.  Tests therefore declare their disposable root in
    an injected boundary instead of relying on the old ``artifacts/operator``
    spelling shortcut.  Production continues to read only the live manifest.
    """
    live = _LIVE_HUMAN_ONLY_BOUNDARY()
    forms = schemas.repo_relative_forms(str(_TEST_ATTESTATION_ROOT))
    if len(forms) != 1:
        raise AssertionError(f"test attestation root has unexpected forms: {forms}")
    root = forms[0]
    return schemas.TrustBoundary(
        globs=tuple(dict.fromkeys((*live.globs, root, root + "/*"))),
        branches=live.branches, source="test-only:release-waiver-fixtures")


_TEST_BOUNDARY = _fixture_boundary()


def _test_human_only_boundary(manifest=None):
    return (_TEST_BOUNDARY if manifest is None
            else _LIVE_HUMAN_ONLY_BOUNDARY(manifest))


def setUpModule():  # noqa: N802 - unittest protocol
    t3.human_only_boundary = _test_human_only_boundary


def _waiver_file_dir() -> Path:
    global _WAIVER_FILE_DIR
    if _WAIVER_FILE_DIR is None:
        _TEST_ATTESTATION_ROOT.mkdir(parents=True, exist_ok=True)
        _WAIVER_FILE_DIR = Path(tempfile.mkdtemp(prefix="_ak_waiver_",
                                                 dir=_TEST_ATTESTATION_ROOT))
        # Cleanup registered by the CREATOR, not by a module teardown, because the
        # creator is not always this module. `test_t3_waiver_authority_redteam`
        # imports `read_waiver` from here and defines its own `tearDownModule` for
        # its own directory, so running that file alone — which `make test` now
        # does, since it is in PYTEST_SMOKE — built one `_ak_waiver_*` here and
        # cleaned a different one, orphaning a directory per run. Measured: eight
        # had accumulated, then nine, then ten.
        #
        # Litter anywhere else would be cosmetic. Here it is not: this path is
        # `<checkout>/artifacts/operator/`, the directory whose SPELLING makes
        # `schemas.operator_owned_path_check` PASS, and the whole reason
        # `DEFAULT_ATTESTATION_ROOTS` exists is that agents in this repo can create
        # it with `mkdir -p`. A test suite that leaves operator-shaped JSON lying in
        # it is manufacturing the exploit fixture the reader was hardened against.
        atexit.register(_remove_waiver_file_dir)
    return _WAIVER_FILE_DIR


def _remove_waiver_file_dir() -> None:
    """Idempotent: safe to call from `tearDownModule` AND from `atexit`."""
    global _WAIVER_FILE_DIR
    if _WAIVER_FILE_DIR is not None:
        shutil.rmtree(_WAIVER_FILE_DIR, ignore_errors=True)
        _WAIVER_FILE_DIR = None


def tearDownModule():
    t3.human_only_boundary = _LIVE_HUMAN_ONLY_BOUNDARY
    _remove_waiver_file_dir()


def write_waiver_file(document, *, name=None, raw=None) -> tuple:
    """Write a waiver to a real operator-shaped path. Returns `(path, raw_sha256)`.

    The digest is over the BYTES WRITTEN — `schemas.raw_bytes_digest`, never
    `content_hash`: the v8 ratification pins `sha256(<file>)`, and a fixture that
    pinned a canonical re-encoding would be testing the reader against a digest no
    operator record uses.
    """
    payload = raw if raw is not None else json.dumps(
        document, indent=1, sort_keys=True).encode("utf-8")
    sha = schemas.raw_bytes_digest(payload)
    target = _waiver_file_dir() / (name or f"waive-{sha[:16]}.json")
    target.write_bytes(payload)
    return (str(target), sha)


def read_waiver(document=None, **overrides) -> t3.ReadWaiver:
    """The same waiver, WRITTEN to disk and READ back through the mandatory reader."""
    document = document if document is not None else autokernel_waiver()
    path, sha = write_waiver_file(document)
    fields = {
        "document_path": path,
        "pinned_sha256": sha,
        "waiver_id": document.get("waiver_id", "W"),
        "covers_cell_ids": ("llama_cpu.prefill",),
        "attestation_roots": (str(_TEST_ATTESTATION_ROOT),),
    }
    fields.update(overrides)
    return t3.waiver_binding_from_path(fields.pop("document_path"), **fields)


class TestWaivers(unittest.TestCase):

    def test_failing_cell_without_a_waiver_fails(self):
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("llama_cpu.prefill", result.verdict_computation.failed_cells)

    def test_verified_waiver_yields_pass_with_waiver_and_suppresses_the_claim(self):
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(read_waiver(),)))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER", reasons_of(result))
        self.assertEqual(result.verdict_computation.failed_cells, ())
        suppressed = [s["claim"] for s in result.receipt.suppressed_claims]
        self.assertIn("llama_cpu prefill non-regression vs v8", suppressed)
        self.assertNotIn("llama_cpu prefill non-regression vs v8", result.receipt.claims)

    def test_the_forfeited_claim_is_named_in_the_receipt(self):
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(read_waiver(),)))
        self.assertEqual(
            result.receipt.forfeited_claims,
            ("No v9 Q8 prefill non-regression claim may be made.",))

    def test_a_pass_run_that_pins_a_waiver_stays_pass_with_waiver_only_if_used(self):
        # A waiver over a cell that passed suppresses nothing, so the verdict is PASS.
        result = t3.run_t3(request(waivers=(read_waiver(),)))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))
        self.assertEqual(result.receipt.suppressed_claims, ())

    def test_hash_mismatch_blocks(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(observed_sha256=digest("something else"))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is not the waiver that is here", reasons_of(result))

    def test_unread_waiver_is_could_not_check_and_still_blocks(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(observed_sha256=None)
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("A quoted waiver is not a read one", reasons_of(result))

    def test_waiver_naming_another_candidate_head_blocks(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(autokernel_waiver(candidate_head="c" * 40))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("names candidate head", reasons_of(result))

    def test_waiver_whose_protocol_moved_blocks(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(autokernel_waiver(protocol_changed=True))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("protocol that moved underneath it", reasons_of(result))

    def test_expired_waiver_blocks(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(autokernel_waiver(
            expiry={"expires_at": "2026-07-01T00:00:00Z"}))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("expired", reasons_of(result))

    def test_waiver_covering_a_cell_outside_the_matrix_blocks(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(covers_cell_ids=("no.such.cell",))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("not in this release matrix", reasons_of(result))

    def test_waiver_forfeiting_nothing_is_an_approval_and_blocks(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(autokernel_waiver(consequences=[]))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("forfeits no claim", reasons_of(result))

    def test_machine_authored_waiver_without_a_human_attestation_blocks(self):
        cells, results = failing_matrix()
        doc = autokernel_waiver()
        doc.pop("authorized_by")
        binding = quoted_waiver(doc)
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
            read_waiver(autokernel_waiver(reason="a well-argued reason")),
            candidate_commit=CANDIDATE_COMMIT, production_base_commit=BASE_COMMIT,
            campaign_id="ak-v9", known_cell_ids=[c.cell_id for c in cells],
            failing_cell_ids=[r.cell.cell_id for r in results
                              if r.check.outcome != schemas.PASS], now=NOW,
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        weak = t3.verify_waiver(
            read_waiver(autokernel_waiver(reason="because")),
            candidate_commit=CANDIDATE_COMMIT, production_base_commit=BASE_COMMIT,
            campaign_id="ak-v9", known_cell_ids=[c.cell_id for c in cells],
            failing_cell_ids=[r.cell.cell_id for r in results
                              if r.check.outcome != schemas.PASS], now=NOW,
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertEqual(base.predicate_results, weak.predicate_results)
        self.assertTrue(base.verified and weak.verified)

    def test_a_waiver_cannot_waive_a_phase_blocker(self):
        """A linkage failure is not a scoped, claim-forfeiting exclusion."""
        cells, results = failing_matrix()
        receipts = (linkage_receipt("llama_cpu", exit_code=1,
                                    stdout="  BAD  libggml-base.so.0 -> /elsewhere/lib\n"),
                    linkage_receipt("llama_gpu"))
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(read_waiver(),),
                                   linkage_receipts=receipts))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("build_linkage", reasons_of(result))

    def test_unknown_waiver_schema_blocks(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(autokernel_waiver(schema="epyc.some.other.v1"))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("not a waiver schema this gate reads", reasons_of(result))


# =============================================================================
# §10.4 — WHO wrote the waiver, and where it lives
#
# The two authorship holes, closed 2026-08-03. Both were invisible from inside
# this file before: `verify_waiver` accepted any non-empty `authorized_by`, and
# `document_path` was free text nothing ever read. The machine-actor refusal
# existed only in `packager.py`, one layer up, so T3's own verdict read
# PASS_WITH_WAIVER and any caller reaching T3 directly bypassed the refusal.
#
# Every guard below is paired with a COMPLIANT-PATH control asserting it does not
# forbid its own legitimate idiom — a self-refusing guard is the recurring defect
# in this plane (`serving_runtime`'s `kernels/production` pattern).
# =============================================================================

#: A boundary in the manifest's own shape, for tests that must not depend on the
#: live file. Every glob here is read from the DOCUMENT, never from a list in the
#: gate — that is the property under test.
BOUNDARY_YAML = """
schema_version: session_bus.human_only_paths.v1
paths:
  - repo: epyc-root
    glob: "MEASUREMENT.md"
    why: "instrument constitution"
  - repo: epyc-root
    glob: "measurement/protocols/*.md"
    why: "protocol annexes"
branches:
  - repo: epyc-llama
    glob: "production-consolidated-*"
    why: "frozen production kernels"
"""

UNREADABLE_BOUNDARY = schemas.TrustBoundary(source="<absent>")


def verify(binding, *, boundary=None, cells=None, failing=("llama_cpu.prefill",),
           attestation_roots=(str(_TEST_ATTESTATION_ROOT),)):
    """`verify_waiver` on one binding, with everything else compliant.

    `attestation_roots` defaults to the suite's own root for the same reason
    `request()` declares it: the fixtures read from a checkout directory, and the
    gate now refuses a read it was not told to expect. Pass `None` to judge against
    the real operator root, which is what production does.
    """
    known = [c.cell_id for c in (cells if cells is not None else matrix_cells())]
    return t3.verify_waiver(
        binding, candidate_commit=CANDIDATE_COMMIT,
        production_base_commit=BASE_COMMIT, campaign_id="ak-v9",
        known_cell_ids=known, failing_cell_ids=list(failing), now=NOW,
        boundary=boundary, attestation_roots=attestation_roots)


class TestWaiverAuthorship(unittest.TestCase):
    """§10.4: a waiver is human-authored BY DEFINITION."""

    def test_a_waiver_attributed_to_the_loop_does_not_verify_at_t3(self):
        binding = read_waiver(autokernel_waiver(authorized_by="autokernel"))
        verification = verify(binding, boundary=UNREADABLE_BOUNDARY)
        self.assertFalse(verification.verified)
        self.assertEqual(verification.predicate_results["human_attested"], schemas.FAIL)
        self.assertIn("machine actor", " ".join(verification.check.reasons))

    def test_the_verdict_itself_is_fail_not_pass_with_waiver(self):
        # The hole this closes: the packager refused the package, but T3's OWN
        # verdict still read PASS_WITH_WAIVER, so any caller reaching T3 directly
        # got a waived pass.
        cells, results = failing_matrix()
        binding = read_waiver(autokernel_waiver(authorized_by="autokernel"))
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("llama_cpu.prefill", result.verdict_computation.failed_cells)
        self.assertIn("machine actor", reasons_of(result))

    def test_every_attribution_field_is_scanned_not_just_authorized_by(self):
        # A guard that reads `authorized_by` and not `approved_by` has a
        # rename-shaped hole, and a waiver is a document a human hand-writes.
        for field_name in schemas.ACTOR_ATTRIBUTION_FIELDS:
            with self.subTest(field=field_name):
                # `authorized_by` stays a human name except where it IS the field
                # under test, so each subtest proves that one field alone refuses.
                doc = autokernel_waiver(**{field_name: "autokernel-daemon"})
                verification = verify(read_waiver(doc),
                                      boundary=UNREADABLE_BOUNDARY)
                self.assertFalse(verification.verified, field_name)
                self.assertIn("machine actor", " ".join(verification.check.reasons))

    def test_the_token_vocabulary_is_the_one_the_packager_reads(self):
        # One vocabulary, in `schemas.py`. Two copies drift, and the copy that
        # drifts is the one that stops catching things.
        self.assertIs(packager.MACHINE_ACTOR_TOKENS, schemas.MACHINE_ACTOR_TOKENS)
        for token in ("autokernel", "daemon", "bot", "cron", "runner"):
            with self.subTest(token=token):
                self.assertTrue(schemas.machine_actor_tokens(f"the-{token}-1"))

    def test_a_human_named_after_no_machine_still_verifies(self):
        # COMPLIANT-PATH CONTROL. The guard matches whole tokens, so a human name
        # is unaffected — including one that merely CONTAINS a token's letters.
        for name in ("daniele", "operator", "Daniele Pinna", "scriptor"):
            with self.subTest(name=name):
                binding = read_waiver(autokernel_waiver(authorized_by=name))
                verification = verify(binding, boundary=_TEST_BOUNDARY)
                self.assertTrue(verification.verified, verification.check.reasons)
                self.assertEqual(verification.predicate_results["human_attested"],
                                 schemas.PASS)

    def test_a_human_attributed_waiver_still_turns_fail_into_pass_with_waiver(self):
        # COMPLIANT-PATH CONTROL, end to end: the refusal must not have closed the
        # door §10.4 exists to open. v8 shipped through this door.
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(read_waiver(),)))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER", reasons_of(result))

    def test_schemas_refuses_a_machine_attributed_waiver_document(self):
        violations = schemas.validate_operator_waiver(
            autokernel_waiver(authorized_by="autokernel"))
        self.assertTrue(any("machine actor" in v for v in violations))
        self.assertEqual(schemas.validate_operator_waiver(autokernel_waiver()), [])


class TestWaiverProvenance(unittest.TestCase):
    """§10.4: a waiver is *"stored under the trust-boundary path set"*.

    A hash proves the bytes did not change after somebody quoted them. It says
    nothing about who could have written them, which is the whole question.
    """

    def test_a_waiver_in_the_loops_own_scratch_does_not_verify(self):
        binding = quoted_waiver(document_path="/mnt/raid0/llm/tmp/waive-q8.json")
        verification = verify(binding,
                              boundary=schemas.parse_trust_boundary(BOUNDARY_YAML))
        self.assertFalse(verification.verified)
        self.assertEqual(verification.predicate_results["document_path"], schemas.FAIL)

    def test_a_scratch_waiver_blocks_the_whole_run(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(
            document_path="scripts/kernel_rnd/autokernel/waive-q8.json")
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("llama_cpu.prefill", result.verdict_computation.failed_cells)

    def test_a_path_outside_every_checkout_root_resolves_to_nothing(self):
        # Containment is tested on the RESOLVED root, never on a substring, so a
        # directory that merely spells the operator root does not inherit it.
        check = schemas.operator_owned_path_check(
            "/tmp/artifacts/operator/waive-q8.json",
            boundary=schemas.parse_trust_boundary(BOUNDARY_YAML))
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_traversal_out_of_the_operator_root_is_not_operator_owned(self):
        check = schemas.operator_owned_path_check(
            "artifacts/operator/../../tmp/waive-q8.json",
            boundary=schemas.parse_trust_boundary(BOUNDARY_YAML))
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_an_unreadable_boundary_is_could_not_check_never_pass(self):
        # THE PROPERTY: a guarantee obtainable by deleting what it inspects is not
        # one. Emptying the manifest must not widen what counts as operator-owned.
        for boundary in (UNREADABLE_BOUNDARY,
                         schemas.parse_trust_boundary(""),
                         schemas.parse_trust_boundary("schema_version: something.else\n"
                                                      "paths:\n  - glob: \"**\"\n")):
            with self.subTest(source=boundary.source):
                self.assertFalse(boundary.readable)
                check = schemas.operator_owned_path_check("/mnt/raid0/llm/tmp/w.json",
                                                          boundary=boundary)
                self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_could_not_check_suppresses_nothing(self):
        binding = quoted_waiver(document_path="/mnt/raid0/llm/tmp/waive-q8.json")
        verification = verify(binding, boundary=UNREADABLE_BOUNDARY)
        self.assertEqual(verification.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertFalse(verification.verified)

    def test_a_foreign_manifest_cannot_widen_the_boundary(self):
        # A document that is not this schema is not this boundary, however
        # generous its globs are.
        foreign = schemas.parse_trust_boundary(
            "schema_version: some.other.v1\npaths:\n  - glob: \"**\"\n",
            source="<foreign>")
        self.assertFalse(foreign.readable)
        self.assertEqual(foreign.globs, ())

    def test_the_manifest_widens_the_boundary_rather_than_a_list_in_the_gate(self):
        # Reuse, not restatement: a path nothing in this module names is
        # operator-owned because the MANIFEST says so.
        boundary = schemas.parse_trust_boundary(BOUNDARY_YAML, source="<fixture>")
        self.assertEqual(
            schemas.operator_owned_path_check("measurement/protocols/kernel-research.md",
                                              boundary=boundary).outcome, schemas.PASS)
        self.assertEqual(
            schemas.operator_owned_path_check("/workspace/MEASUREMENT.md",
                                              boundary=boundary).outcome, schemas.PASS)

    def test_the_live_manifest_is_readable_and_is_the_one_schemas_names(self):
        boundary = t3.human_only_boundary()
        self.assertTrue(str(t3.TRUST_BOUNDARY_MANIFEST).endswith(
            schemas.HUMAN_ONLY_PATHS_MANIFEST))
        if not Path(t3.TRUST_BOUNDARY_MANIFEST).exists():
            self.skipTest("epyc-root is not checked out beside this repo")
        self.assertTrue(boundary.readable, boundary.to_dict())
        self.assertIn("MEASUREMENT.md", boundary.globs)

    def test_an_absent_manifest_reads_as_unreadable_not_as_empty(self):
        boundary = t3.human_only_boundary(
            Path(t3.TRUST_BOUNDARY_MANIFEST).parent / "no-such-manifest.yaml")
        self.assertFalse(boundary.readable)
        self.assertEqual(boundary.globs, ())

    def test_the_operator_attestation_root_needs_no_manifest(self):
        # COMPLIANT-PATH CONTROL. The idiom this plane already writes —
        # `artifacts/operator/<label>/waiver.json`, the path `calibration_request`
        # itself emits — must pass, and must pass even with no manifest at all, or
        # the guard forbids its own output.
        for path in ("artifacts/operator/waive-q8-v9.json",
                     "artifacts/operator/v9-freeze/waiver.json",
                     "/workspace/artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json"):
            for boundary in (UNREADABLE_BOUNDARY,
                             schemas.parse_trust_boundary(BOUNDARY_YAML)):
                with self.subTest(path=path, readable=boundary.readable):
                    self.assertEqual(
                        schemas.operator_owned_path_check(path,
                                                          boundary=boundary).outcome,
                        schemas.PASS)

    def test_the_default_boundary_admits_the_normal_waiver_path(self):
        # COMPLIANT-PATH CONTROL against the LIVE boundary, through `run_t3`'s own
        # default: no explicit boundary, the fixture's ordinary operator path.
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(read_waiver(),)))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER", reasons_of(result))

    def test_schemas_checks_the_path_only_when_the_caller_states_one(self):
        document = autokernel_waiver()
        self.assertEqual(schemas.validate_operator_waiver(document), [])
        self.assertEqual(
            schemas.validate_operator_waiver(
                document, document_path="artifacts/operator/w.json"), [])
        self.assertTrue(schemas.validate_operator_waiver(
            document, document_path="/mnt/raid0/llm/tmp/w.json",
            boundary=schemas.parse_trust_boundary(BOUNDARY_YAML)))


# =============================================================================
# Waiver authority — the RED-TEAM pass over the 2026-08-03 closure.
#
# Three of the four defects below are the same shape the closure was written to
# answer, one layer further in: a guarantee obtainable by DELETING what it
# inspects, and a guard walked around by spelling.
# =============================================================================

#: A boundary that declares the era registry, so the glob path can be attacked as
#: well as the attestation root.
ERA_BOUNDARY = schemas.parse_trust_boundary("""
schema_version: session_bus.human_only_paths.v1
paths:
  - repo: epyc-orchestrator
    glob: "orchestration/instrument_eras.yaml"
    why: "era registry rows"
""", source="<era-fixture>")


def v8_shaped_waiver(**overrides) -> dict:
    """The PRESERVED v8 waiver's shape: no attribution field anywhere.

    `artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json` carries
    `ratified_at` and nothing else — no `authorized_by`, `ratified_by`,
    `approved_by`, `attested_by` or `granted_by`. That is a fact about the genuine
    ratified record, so it is a fixture here rather than something to be fixed
    there.
    """
    doc = {
        "schema": t3.WAIVER_SCHEMA_V8_CPU_PREFILL,
        "decision": "WAIVE",
        "protocol": "P-BENCH-PREFILL-1",
        "protocol_changed": False,
        "candidate_head": CANDIDATE_COMMIT,
        "production_head": BASE_COMMIT,
        "scope": {"excluded_pairs": ["llama_cpu.prefill"]},
        "reason": "the Q8 workload cannot satisfy the ratified core-equivalent floor",
        "consequences": ["No v9 Q8 prefill non-regression claim may be made."],
        "ratified_at": "2026-08-02T00:00:00Z",
    }
    doc.update(overrides)
    return doc


class TestNothingReadsTheWaiverFile(unittest.TestCase):
    """THE DEFECT, reproduced and then made structurally unreachable.

    `WaiverBinding` carried `document`, `document_path` and `observed_sha256` as three
    INDEPENDENT caller assertions and nothing opened the file. A document the caller
    invented, pinned to its own digest, at a path that does not exist, verified — and
    took its AUTHORSHIP from `attribution_source="operator_owned_path"`, borrowing the
    standing of a directory it was not in. §10.4 turns FAIL into PASS_WITH_WAIVER, so
    that was the authority path of the whole freeze gate resting on the honesty of the
    party being gated.
    """

    ABSENT = "/workspace/artifacts/operator/this-file-does-not-exist.json"

    def test_the_defect_shape_is_still_constructible_and_now_suppresses_nothing(self):
        # The EXACT reproduction: an invented document, its own digest, a path with
        # no file behind it. Constructible — a quotation is a legitimate object — and
        # inert.
        self.assertFalse(Path(self.ABSENT).exists())
        self.assertEqual(schemas.operator_owned_path_check(self.ABSENT).outcome,
                         schemas.PASS,
                         "the citation must still SPELL an operator-owned location, "
                         "or this test is proving the wrong refusal")
        binding = quoted_waiver(document_path=self.ABSENT)
        verification = verify(binding)
        self.assertFalse(verification.verified)
        self.assertEqual(verification.predicate_results["read"], schemas.COULD_NOT_CHECK)
        self.assertEqual(verification.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("was never read from disk", " ".join(verification.check.reasons))

    def test_a_quotation_covers_no_cell_even_though_its_scope_names_one(self):
        # The second lock, and a separate defect: `covered_cell_ids` used to be
        # populated regardless of the outcome, so a refused waiver landed in the
        # durable bundle carrying a waived-LOOKING coverage list.
        verification = verify(quoted_waiver(document_path=self.ABSENT))
        self.assertEqual(verification.covered_cell_ids, ())
        # ... while the same document, READ, does cover it.
        self.assertEqual(verify(read_waiver()).covered_cell_ids,
                         ("llama_cpu.prefill",))

    def test_the_whole_run_blocks_rather_than_merely_not_suppressing(self):
        # Fail-CLOSED, not merely inert: `phase_identity_preflight` turns
        # COULD_NOT_CHECK into a blocking reason, so an unread waiver stops the run.
        cells, results = failing_matrix()
        result = t3.run_t3(request(_cells=cells, _results=results,
                                   waivers=(quoted_waiver(document_path=self.ABSENT),)))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("was never read from disk", reasons_of(result))

    def test_unread_stays_expressible_and_is_still_the_honest_answer(self):
        # COMPLIANT-PATH CONTROL for the OTHER half of the constraint. A design that
        # made "I did not read the file" inexpressible would push callers into
        # asserting a digest they never computed, which is worse than the defect.
        binding = quoted_waiver(observed_sha256=None)
        self.assertIsInstance(binding, t3.WaiverBinding)
        self.assertFalse(binding.was_read)
        self.assertIsNone(binding.observed_sha256)
        verification = verify(binding)
        self.assertEqual(verification.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(verification.covered_cell_ids, ())

    def test_a_refusal_to_read_is_not_an_unread_waiver(self):
        # A degrade would turn "the reader refused these bytes" into "nobody looked",
        # which is the fail-open shape this project keeps getting bitten by.
        with self.assertRaises(t3.WaiverNotReadable):
            t3.waiver_binding_from_path(
                self.ABSENT, pinned_sha256=digest("anything"), waiver_id="W",
                covers_cell_ids=("llama_cpu.prefill",))

    def test_only_the_reader_can_produce_the_type_that_verifies(self):
        # The capability, not a flag: a receipt cannot be constructed by hand, so a
        # `ReadWaiver` cannot be constructed by hand either.
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.WaiverReadReceipt(
                resolved_path="/workspace/artifacts/operator/w.json",
                citation="/workspace/artifacts/operator/w.json", st_dev=1, st_ino=2,
                st_size=3, st_mtime_ns=4, byte_length=3,
                bytes_sha256=digest("x"), document={"schema": "x"})
        self.assertIn("MINTED", str(ctx.exception))

    def test_a_stolen_receipt_cannot_be_re_pointed_at_another_document(self):
        # The identity assertion, which is what makes "the document returned is the
        # one whose bytes were hashed" a fact rather than a hope. An EQUAL mapping is
        # not the same object, and equality is exactly what a substitution preserves.
        real = read_waiver()
        clone = json.loads(json.dumps(dict(real.document)))
        self.assertEqual(clone, dict(real.document))
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.ReadWaiver(
                waiver_id=real.waiver_id, pinned_sha256=real.pinned_sha256,
                document=clone, document_path=real.document_path,
                covers_cell_ids=real.covers_cell_ids,
                observed_sha256=real.observed_sha256,
                read_receipt=real.read_receipt)
        self.assertIn("is not the object the receipt hashed", str(ctx.exception))

    def test_a_read_waiver_cannot_restate_a_path_it_did_not_read(self):
        real = read_waiver()
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.ReadWaiver(
                waiver_id=real.waiver_id, pinned_sha256=real.pinned_sha256,
                document=real.document,
                document_path="artifacts/operator/somewhere-else.json",
                covers_cell_ids=real.covers_cell_ids,
                observed_sha256=real.observed_sha256,
                read_receipt=real.read_receipt)
        self.assertIn("must be the citation the receipt read", str(ctx.exception))


class TestTheGenuineV8WaiverStillVerifies(unittest.TestCase):
    """COMPLIANT-PATH CONTROL, and the one that decides whether the reader is right.

    The real ratified v8 waiver is in the LEGACY schema
    `epyc.cpu_prefill_v8.operator_waiver.v1`, carries NONE of the five attribution
    fields, and its whole attestation is a `ratified_at` timestamp plus where it
    lives. A reader that cannot ingest it is WRONG, not strict: invalidating a
    genuine ratified record to make a new rule tidy is the failure mode here.
    """

    @classmethod
    def setUpClass(cls):
        if not (V8_WAIVER_PATH.is_file() and V8_RATIFICATION_PATH.is_file()):
            raise AssertionError(
                f"{V8_WAIVER_PATH} / {V8_RATIFICATION_PATH} are not present. This is a "
                "FAILURE and never a skip: the preserved v8 pair is the only real "
                "operator record the reader grades itself against.")

    def _read(self, **overrides):
        fields = {"pinned_sha256": V8_WAIVER_SHA, "waiver_id": "WAIVE-Q8",
                  "covers_cell_ids": ("llama_cpu.pair.qwen36_q8-pp2048-iqk1",)}
        fields.update(overrides)
        return t3.waiver_binding_from_path(str(V8_WAIVER_PATH), **fields)

    def test_the_real_waiver_reads_at_the_default_attestation_root(self):
        # No `attestation_roots` override, no `boundary` override: the live defaults.
        binding = self._read()
        self.assertIsInstance(binding, t3.ReadWaiver)
        self.assertEqual(binding.observed_sha256, V8_WAIVER_SHA)
        self.assertEqual(binding.read_receipt.byte_length, 1267)
        self.assertEqual(binding.read_receipt.attestation_root,
                         t3.DEFAULT_ATTESTATION_ROOTS[0])
        self.assertEqual(binding.document["schema"], t3.WAIVER_SCHEMA_V8_CPU_PREFILL)
        # Its keys are EXACTLY the legacy set — none of the five attribution fields.
        self.assertEqual(
            sorted(binding.document),
            ["candidate_head", "consequences", "decision", "production_head",
             "protocol", "protocol_changed", "ratified_at", "reason",
             "runner_sha256_before_waiver_implementation", "schema", "scope"])
        for field_name in schemas.ACTOR_ATTRIBUTION_FIELDS:
            self.assertNotIn(field_name, binding.document)

    def test_the_digest_is_the_one_the_ratification_pins(self):
        ratification = json.loads(V8_RATIFICATION_PATH.read_text(encoding="utf-8"))
        self.assertEqual(ratification["evidence_sha256"]["waive_q8"], V8_WAIVER_SHA)
        binding = self._read(ratification_pin=(str(V8_RATIFICATION_PATH), "waive_q8"))
        self.assertEqual(binding.read_receipt.ratification_pin, V8_WAIVER_SHA)

    def test_the_pin_is_over_raw_bytes_and_content_hash_would_not_match(self):
        # The trap: `schemas.content_hash` digests a canonical RE-ENCODING, so a
        # reader that used it could not verify a single real operator record.
        raw = V8_WAIVER_PATH.read_bytes()
        self.assertEqual(schemas.raw_bytes_digest(raw), V8_WAIVER_SHA)
        self.assertNotEqual(schemas.content_hash(json.loads(raw)), V8_WAIVER_SHA)
        with self.assertRaises(t3.WaiverNotReadable):
            self._read(pinned_sha256=schemas.content_hash(json.loads(raw)))

    def test_the_real_waiver_verifies_through_verify_waiver(self):
        # End to end at the gate: the legacy schema, no author, attribution carried by
        # the path — and now by a path that was actually opened.
        binding = self._read(
            covers_cell_ids=("llama_cpu.pair.qwen36_q8-pp2048-iqk1",
                             "llama_cpu.pair.qwen36_q8-tg128-iqk1"))
        cells = [
            t3.Cell(cell_id=f"llama_cpu.pair.{pair}", backend="llama_cpu",
                    release_phase=t3.PHASE_PERFORMANCE_MATRIX,
                    protocol_id="P-BENCH-PREFILL-1",
                    recipe_class=t3.RECIPE_PRODUCTION_OPTIMAL, metric="tokens_per_s",
                    metric_direction="higher_better", workload_phase="prefill",
                    claim=f"{pair} non-regression", reps=10)
            for pair in ("qwen36_q8-pp2048-iqk1", "qwen36_q8-tg128-iqk1")]
        verification = t3.verify_waiver(
            binding,
            candidate_commit="67a433bf45a8a091d83b4ea0b32ff0735fd51800",
            production_base_commit="6ad45fa3ff6718c07c000061dbc6e29c1771f6e3",
            campaign_id="ak-calibration",
            known_cell_ids=[c.cell_id for c in cells],
            failing_cell_ids=[c.cell_id for c in cells], now=NOW)
        self.assertTrue(verification.verified, verification.check.reasons)
        self.assertEqual(verification.predicate_results["read"], schemas.PASS)
        self.assertEqual(verification.predicate_results["attribution_source"],
                         "operator_owned_path")
        self.assertEqual(set(verification.covered_cell_ids),
                         {c.cell_id for c in cells})

    def test_a_byte_identical_copy_at_another_declared_root_also_reads(self):
        # The reader is about BYTES and LOCATION, never about one hard-coded file.
        path, sha = write_waiver_file(None, raw=V8_WAIVER_PATH.read_bytes(),
                                      name="waive_q8_copy.json")
        self.assertEqual(sha, V8_WAIVER_SHA)
        binding = t3.waiver_binding_from_path(
            path, pinned_sha256=V8_WAIVER_SHA, waiver_id="WAIVE-Q8",
            covers_cell_ids=("llama_cpu.prefill",), attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertEqual(binding.observed_sha256, V8_WAIVER_SHA)

    def test_the_default_root_refuses_a_checkouts_own_artifacts_operator(self):
        # THE NARROWING, and the named residual it closes. `operator_owned_path_check`
        # answers PASS for the suite's disposable root only under the explicit
        # synthetic boundary. The live boundary correctly refuses this arbitrary
        # worktree; the reader must still narrow a passing citation to its own root.
        path, sha = write_waiver_file(autokernel_waiver())
        self.assertEqual(
            schemas.operator_owned_path_check(path, boundary=_TEST_BOUNDARY).outcome,
            schemas.PASS)
        self.assertNotEqual(
            schemas.operator_owned_path_check(
                path, boundary=_LIVE_HUMAN_ONLY_BOUNDARY()).outcome,
            schemas.PASS)
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            t3.waiver_binding_from_path(path, pinned_sha256=sha, waiver_id="W",
                                        covers_cell_ids=("llama_cpu.prefill",))
        self.assertIn("declared attestation roots", str(ctx.exception))


class TestTheReaderRefusesEveryFilesystemHazard(unittest.TestCase):
    """One test per row of the hazard table. Each writes a real file and reads it."""

    def _path(self, name: str) -> Path:
        target = _waiver_file_dir() / name
        self.addCleanup(lambda: target.unlink(missing_ok=True))
        return target

    def _read(self, target, **overrides):
        # lstat FIRST, exactly as the reader does: reading a FIFO here would block
        # this test process in the open syscall, which is the hazard under test.
        try:
            mode = target.lstat().st_mode
            raw = target.read_bytes() if stat.S_ISREG(mode) else b"{}"
        except OSError:
            raw = b"{}"
        fields = {"pinned_sha256": schemas.raw_bytes_digest(raw), "waiver_id": "W",
                  "covers_cell_ids": ("llama_cpu.prefill",),
                  "attestation_roots": (str(_TEST_ATTESTATION_ROOT),)}
        fields.update(overrides)
        return t3.waiver_binding_from_path(str(target), **fields)

    def test_a_final_component_symlink_is_refused(self):
        real, sha = write_waiver_file(autokernel_waiver())
        link = self._path("link-to-a-waiver.json")
        link.symlink_to(real)
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(link, pinned_sha256=sha)
        self.assertIn("symbolic link", str(ctx.exception))

    def test_a_fifo_is_refused_and_never_opened(self):
        # Without the type check BEFORE the read, `read_bytes` on a FIFO blocks in the
        # open syscall and the test never returns. That this test terminates IS the
        # assertion.
        fifo = self._path("waiver.fifo.json")
        os.mkfifo(fifo)
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(fifo, pinned_sha256=digest("x"))
        self.assertIn("not a regular file", str(ctx.exception))

    def test_a_directory_is_refused(self):
        d = _waiver_file_dir() / "a-directory.json"
        d.mkdir(exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(d, pinned_sha256=digest("x"))
        self.assertIn("not a regular file", str(ctx.exception))

    def test_a_hardlinked_file_is_refused(self):
        real, sha = write_waiver_file(autokernel_waiver())
        second = self._path("second-name.json")
        os.link(real, second)
        self.addCleanup(lambda: Path(real).unlink(missing_ok=True))
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(second, pinned_sha256=sha)
        self.assertIn("hard links", str(ctx.exception))

    def test_an_oversized_file_is_refused_without_being_read(self):
        target = self._path("oversized.json")
        target.write_bytes(b"{" + b" " * 4096 + b"}")
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(target, max_bytes=64)
        self.assertIn("ceiling for an operator attestation", str(ctx.exception))

    def test_a_hash_mismatch_is_refused(self):
        target, _ = write_waiver_file(autokernel_waiver())
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(Path(target), pinned_sha256=digest("some other bytes"))
        self.assertIn("is not the waiver that is here", str(ctx.exception))

    def test_bytes_that_are_not_a_json_object_are_refused(self):
        target = self._path("not-an-object.json")
        target.write_bytes(b"[1, 2, 3]")
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(target)
        self.assertIn("not a JSON object", str(ctx.exception))
        target.write_bytes(b"not json at all")
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(target)
        self.assertIn("is not JSON", str(ctx.exception))

    def test_a_parent_directory_symlink_is_refused_by_the_resolved_check(self):
        # THE HAZARD THE CITATION CHECK STRUCTURALLY CANNOT SEE: the citation is
        # spelled `artifacts/operator/…` and every string check passes; only
        # `realpath` finds that the directory is a link into the loop's own scratch.
        scratch = Path(tempfile.mkdtemp(prefix="_ak_scratch_", dir="/tmp"))
        self.addCleanup(lambda: shutil.rmtree(scratch, ignore_errors=True))
        raw = json.dumps(autokernel_waiver(), sort_keys=True).encode("utf-8")
        (scratch / "w.json").write_bytes(raw)
        link_dir = self._path("via-a-linked-parent")
        link_dir.symlink_to(scratch, target_is_directory=True)
        citation = str(link_dir / "w.json")
        self.assertEqual(schemas.operator_owned_path_check(
            citation, boundary=_TEST_BOUNDARY).outcome,
                         schemas.PASS, "the citation must LOOK operator-owned")
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            t3.waiver_binding_from_path(
                citation, pinned_sha256=schemas.raw_bytes_digest(raw), waiver_id="W",
                covers_cell_ids=("llama_cpu.prefill",),
                attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertIn("scratch root", str(ctx.exception))

    def test_a_citation_outside_the_boundary_is_refused_before_any_io(self):
        # Order is load-bearing: the citation check is first and cheapest, so a path
        # outside the boundary is never stat'd, let alone opened.
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            t3.waiver_binding_from_path(
                "/mnt/raid0/llm/tmp/artifacts/operator/w.json",
                pinned_sha256=digest("x"), waiver_id="W", covers_cell_ids=("llama_cpu.prefill",))
        self.assertIn("not established as an operator-owned citation",
                      str(ctx.exception))

    def test_a_production_tree_can_never_be_a_waiver_source(self):
        for tree in storage.PRODUCTION_TREES:
            with self.subTest(tree=tree):
                with self.assertRaises(t3.WaiverNotReadable):
                    t3.waiver_binding_from_path(
                        f"{tree}/artifacts/operator/w.json",
                        pinned_sha256=digest("x"), waiver_id="W", covers_cell_ids=("llama_cpu.prefill",))

    def test_an_empty_root_set_is_refused_rather_than_read_as_anywhere(self):
        target, sha = write_waiver_file(autokernel_waiver())
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.waiver_binding_from_path(target, pinned_sha256=sha, waiver_id="W",
                                        covers_cell_ids=("llama_cpu.prefill",), attestation_roots=())
        self.assertIn("must not be empty", str(ctx.exception))

    def test_the_ratification_cross_check_refuses_a_key_that_is_not_there(self):
        target, sha = write_waiver_file(autokernel_waiver())
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(Path(target), pinned_sha256=sha,
                       ratification_pin=(str(V8_RATIFICATION_PATH), "no_such_key"),
                       attestation_roots=(str(_TEST_ATTESTATION_ROOT),
                                          t3.DEFAULT_ATTESTATION_ROOTS[0]))
        self.assertIn("pins no digest at", str(ctx.exception))

    def test_a_waiver_the_ratification_does_not_hash_is_refused(self):
        target, sha = write_waiver_file(autokernel_waiver())
        with self.assertRaises(t3.WaiverNotReadable) as ctx:
            self._read(Path(target), pinned_sha256=sha,
                       ratification_pin=(str(V8_RATIFICATION_PATH), "waive_q8"),
                       attestation_roots=(str(_TEST_ATTESTATION_ROOT),
                                          t3.DEFAULT_ATTESTATION_ROOTS[0]))
        self.assertIn("is not the ratified waiver", str(ctx.exception))

    def test_the_compliant_path_still_reads(self):
        """COMPLIANT-PATH CONTROL for the whole hazard table: an ordinary regular
        file, one link, small, at an operator-shaped path, still reads."""
        target, sha = write_waiver_file(autokernel_waiver())
        binding = self._read(Path(target), pinned_sha256=sha)
        self.assertIsInstance(binding, t3.ReadWaiver)
        self.assertEqual(binding.read_receipt.bytes_sha256, sha)
        self.assertIs(binding.document, binding.read_receipt.document)


class TestTheReaderReadsExactlyOnce(unittest.TestCase):
    """The single-`bytes`-object guarantee, enforced mechanically.

    Hashing a file and separately parsing it verifies nothing about the parsed
    object. What makes "the document returned is the one whose bytes were hashed" a
    fact is that there is exactly ONE read and one `bytes` object.
    """

    def test_one_read_call_per_reader_invocation(self):
        target, sha = write_waiver_file(autokernel_waiver())
        calls = []
        original = Path.read_bytes

        def counting(self, *a, **kw):
            calls.append(str(self))
            return original(self, *a, **kw)

        Path.read_bytes = counting
        try:
            t3.waiver_binding_from_path(
                target, pinned_sha256=sha, waiver_id="W", covers_cell_ids=("llama_cpu.prefill",),
                attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        finally:
            Path.read_bytes = original
        self.assertEqual(calls, [target])

    def test_the_ast_audit_names_every_reader_and_passes(self):
        check = t3.audit_waiver_reader_is_the_only_reader()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_audit_bites_on_a_reintroduced_read_text(self):
        # THE BITE. This is what stops a future edit from putting
        # `Path(document_path).read_text()` back into the module.
        source = Path(t3.__file__).read_text(encoding="utf-8")
        doctored = source + (
            "\n\ndef _reread_the_waiver(path):\n"
            "    return Path(path).read_text(encoding='utf-8')\n")
        check = t3.audit_waiver_reader_is_the_only_reader(doctored)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("_reread_the_waiver", " ".join(check.reasons))

    def test_the_audit_bites_on_a_second_read_inside_the_reader(self):
        source = Path(t3.__file__).read_text(encoding="utf-8")
        # Anchored on the try-block form. `read_bytes()` is wrapped so an EACCES or a
        # NUL byte becomes a `WaiverNotReadable` instead of a bare OSError, and a
        # doctoring string that no longer matches its target silently produces
        # unparsable source, which the audit answers COULD_NOT_CHECK — a bite test
        # that stops biting without ever failing loudly.
        anchor = "        raw = target.read_bytes()\n"
        self.assertIn(anchor, source)
        doctored = source.replace(
            anchor, anchor + "        _again = target.read_bytes()\n", 1)
        self.assertNotEqual(doctored, source)
        check = t3.audit_waiver_reader_is_the_only_reader(doctored)
        self.assertEqual(check.outcome, schemas.FAIL, check.reasons)
        self.assertIn("performs 2 read calls", " ".join(check.reasons))

    def test_the_audit_bites_on_a_second_mention_of_the_mint_token(self):
        source = Path(t3.__file__).read_text(encoding="utf-8")
        doctored = source + (
            "\n\ndef _forge_a_receipt(**kw):\n"
            "    return WaiverReadReceipt(_minted=_READER_TOKEN, **kw)\n")
        check = t3.audit_waiver_reader_is_the_only_reader(doctored)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("_READER_TOKEN is named", " ".join(check.reasons))

    def test_the_module_still_cannot_write_or_spawn(self):
        # The reader added filesystem READS to a module whose whole standing is that
        # it cannot mutate the host. That property must survive the addition — it is
        # also the reason the fd-based `os.open` design is not implementable here.
        check = t3.audit_no_write_or_process_paths()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)


#: Real digests. `_sha256` refuses a repeated-character digest outright ("the
#: digest of no bytes at all"), so a forgery test must forge with a plausible
#: one — the guard being probed is the READ capability, not digest well-formedness.
_FORGED_SHA = "628ab7c6a2531dac93e8b1c080b6614f5d8bd8686665a06f316cfb1335588703"
_LIAR_SHA = "a7626b263866e2f9fffb622bd4be88b092670d1e6a4993c737fec38dd688c312"
_PROBE_SHA = "cd76258368e3ff53dc0e539152d695705caec6218f7e499a39e2d70383137b95"


class TestTheReadPredicateIsACapabilityNotAType(unittest.TestCase):
    """A capability that INHERITANCE confers is a flag the caller sets.

    The reader's central claim was that `isinstance(binding, t3.ReadWaiver)` is "a
    capability test rather than a flag the caller sets", because only
    `waiver_binding_from_path` can mint the receipt `ReadWaiver.__post_init__`
    demands. A filesystem red-team refuted it in three lines: subclass `ReadWaiver`,
    make `__post_init__` do nothing, and the constructor that demands the receipt
    never runs. The resulting object needed no receipt, no token and no filesystem,
    and it took `read=PASS`, `attribution_source='operator_owned_path'`,
    `verified=True` and coverage of a failing gating cell — the whole §10.4 defect,
    restored inside its own fix.

    `t3.waiver_read_violations` is the answer: the capability is the TOKEN OBJECT on
    the receipt, so the gate looks at the token itself and re-asserts every invariant
    rather than trusting a constructor a subclass may decline to run.
    """

    #: The forged type, written exactly as an attacker would. `frozen=True` because
    #: the base is; nothing else is needed.
    @staticmethod
    def _sneaky_cls():
        @dataclasses.dataclass(frozen=True)
        class Sneaky(t3.ReadWaiver):
            def __post_init__(self):  # noqa: D105 - the entire attack
                pass
        return Sneaky

    def _verify(self, binding, cell="llama_cpu.pair.qwen36_q8-pp2048-iqk1"):
        # The fixture root, for the same reason the module-level `verify()` helper
        # declares it: these bindings are read from a checkout directory, and the gate
        # holds its own root set rather than the reader's.
        return t3.verify_waiver(
            binding, candidate_commit=V8_HEAD, production_base_commit=V7_HEAD,
            campaign_id="C", known_cell_ids=(cell,), failing_cell_ids=(cell,),
            now="2026-08-03T00:00:00Z",
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),
                               t3.DEFAULT_ATTESTATION_ROOTS[0]))

    def test_a_readwaiver_subclass_with_no_receipt_suppresses_nothing(self):
        # THE BITE. Reverting `verify_waiver`'s read predicate to
        # `if isinstance(binding, t3.ReadWaiver)` makes every assertion below fail:
        # verified becomes True and covered_cell_ids becomes the failing cell.
        missing = "/workspace/artifacts/operator/this-file-does-not-exist.json"
        self.assertFalse(os.path.exists(missing), "the attack path must not exist")
        forged = self._sneaky_cls()(
            waiver_id="FORGED", pinned_sha256=_FORGED_SHA, document=V8_WAIVER,
            document_path=missing,
            covers_cell_ids=("llama_cpu.pair.qwen36_q8-pp2048-iqk1",),
            observed_sha256=_FORGED_SHA, read_receipt=None)
        self.assertIsInstance(forged, t3.ReadWaiver)  # the type test it defeats
        verification = self._verify(forged)
        self.assertEqual(verification.predicate_results["read"], schemas.FAIL)
        self.assertFalse(verification.verified)
        self.assertEqual(verification.covered_cell_ids, ())

    def test_a_hand_built_receipt_subclass_is_not_a_mint(self):
        # `WaiverReadReceipt.__post_init__` demands the token — and is skippable by
        # exactly the same move, so the gate must not accept "is a WaiverReadReceipt"
        # as evidence of a mint either.
        @dataclasses.dataclass(frozen=True)
        class SneakyReceipt(t3.WaiverReadReceipt):
            def __post_init__(self):
                pass

        receipt = SneakyReceipt(
            resolved_path="/workspace/artifacts/operator/nope.json",
            citation="/workspace/artifacts/operator/nope.json", st_dev=1, st_ino=1,
            st_size=1, st_mtime_ns=1, byte_length=1, bytes_sha256=_FORGED_SHA,
            document=V8_WAIVER, attestation_root="/workspace/artifacts/operator")
        self.assertIsInstance(receipt, t3.WaiverReadReceipt)
        forged = self._sneaky_cls()(
            waiver_id="FORGED", pinned_sha256=_FORGED_SHA, document=V8_WAIVER,
            document_path="/workspace/artifacts/operator/nope.json",
            covers_cell_ids=("llama_cpu.pair.qwen36_q8-pp2048-iqk1",),
            observed_sha256=_FORGED_SHA, read_receipt=receipt)
        verification = self._verify(forged)
        self.assertEqual(verification.predicate_results["read"], schemas.FAIL)
        self.assertEqual(verification.covered_cell_ids, ())
        self.assertIn("did not mint", " ".join(verification.check.reasons))

    def test_a_stolen_genuine_receipt_cannot_carry_another_document(self):
        # The receipt IS a mint, but it attests to one `bytes` object and the document
        # parsed from it. Skipping the constructor does not detach them, because the
        # gate re-checks object identity.
        genuine = read_waiver(V8_WAIVER)
        swapped = dict(V8_WAIVER)
        swapped["scope"] = {"excluded_pairs": ["anything-at-all"]}
        forged = self._sneaky_cls()(
            waiver_id="W", pinned_sha256=genuine.pinned_sha256, document=swapped,
            document_path=genuine.document_path,
            covers_cell_ids=("llama_cpu.pair.qwen36_q8-pp2048-iqk1",),
            observed_sha256=genuine.observed_sha256,
            read_receipt=genuine.read_receipt)
        verification = self._verify(forged)
        self.assertEqual(verification.predicate_results["read"], schemas.FAIL)
        self.assertEqual(verification.covered_cell_ids, ())
        self.assertIn("not the object the receipt hashed",
                      " ".join(verification.check.reasons))

    def test_a_forgery_FAILS_while_an_honest_quotation_only_COULD_NOT_CHECK(self):
        # The two states must stay distinguishable. "I did not read it" invites
        # somebody to go and look; "I read it" from something that did not is a lie,
        # and a lie is not rehabilitated by later evidence.
        quotation = quoted_waiver(V8_WAIVER, covers_cell_ids=(
            "llama_cpu.pair.qwen36_q8-pp2048-iqk1",))
        self.assertEqual(self._verify(quotation).predicate_results["read"],
                         schemas.COULD_NOT_CHECK)
        forged = self._sneaky_cls()(
            waiver_id="F", pinned_sha256=_FORGED_SHA, document=V8_WAIVER,
            document_path="/workspace/artifacts/operator/nope.json",
            covers_cell_ids=("llama_cpu.prefill",), observed_sha256=_FORGED_SHA, read_receipt=None)
        self.assertEqual(self._verify(forged).predicate_results["read"], schemas.FAIL)

    def test_was_read_is_computed_from_the_token_not_from_the_class(self):
        self.assertFalse(quoted_waiver().was_read)
        self.assertTrue(read_waiver().was_read)
        # A `WaiverBinding` subclass that simply declares itself read still lies to
        # anything reading the property — which is exactly why nothing load-bearing
        # reads it. The function is the authority and it is not fooled.
        class Liar(t3.WaiverBinding):
            @property
            def was_read(self):
                return True

        liar = Liar(waiver_id="W", pinned_sha256=_LIAR_SHA, document=V8_WAIVER,
                    document_path="artifacts/operator/w.json",
                    covers_cell_ids=("llama_cpu.prefill",), observed_sha256=_LIAR_SHA)
        self.assertTrue(liar.was_read)
        self.assertTrue(t3.waiver_read_violations(liar))
        # COULD_NOT_CHECK rather than FAIL, and correctly so: the object carries the
        # quotation TYPE and no receipt, so what it asserts to the gate is "nobody
        # looked" — the property override lies only to whatever reads the property,
        # and nothing load-bearing does. Either way it suppresses nothing.
        verification = self._verify(liar)
        self.assertEqual(verification.predicate_results["read"],
                         schemas.COULD_NOT_CHECK)
        self.assertFalse(verification.verified)
        self.assertEqual(verification.covered_cell_ids, ())

    def test_the_package_records_the_capability_not_the_property(self):
        # The package is the durable record: the one place a lie outlives the run.
        liar_doc = dict(V8_WAIVER)

        class Liar(t3.WaiverBinding):
            @property
            def was_read(self):
                return True

        liar = Liar(waiver_id="W", pinned_sha256=_LIAR_SHA, document=liar_doc,
                    document_path="artifacts/operator/w.json",
                    covers_cell_ids=("llama_cpu.prefill",), observed_sha256=_LIAR_SHA)
        self.assertIs(liar.to_dict()["read"], False)

    def test_a_genuinely_read_waiver_still_passes_the_predicate(self):
        # THE COMPLIANT-PATH CONTROL for this guard: the capability check must not
        # forbid its own legitimate idiom.
        binding = read_waiver(V8_WAIVER)
        self.assertEqual(t3.waiver_read_violations(binding), ())
        self.assertEqual(
            self._verify(binding).predicate_results["read"], schemas.PASS)


class TestTheReaderGuardsCanOnlyBeNarrowed(unittest.TestCase):
    """Every knob on the reader is a GUARD, and a guard the caller can relax is decor.

    `attestation_roots` is documented as a NARROWING of
    `schemas.operator_owned_path_check` — the citation check is about the SPELLING of
    a path (`artifacts/operator/…` inside a known repository checkout, which the loop can create with
    `mkdir -p`), and the reader additionally requires the bytes to be where the
    operator actually keeps them. It was a free SUBSTITUTION instead:
    `attestation_roots=("/mnt/raid0/llm/epyc-inference-research",)` admitted the whole
    research checkout, so the composite guard was weaker than the check it claims to
    narrow, undone by one keyword argument from the party being gated. Likewise
    `max_bytes=10**9` read a 1 MiB file the ceiling exists to refuse.
    """

    def _read(self, path, sha, **kw):
        kw.setdefault("attestation_roots", (str(_TEST_ATTESTATION_ROOT),))
        return t3.waiver_binding_from_path(
            path, pinned_sha256=sha, waiver_id="W",
            covers_cell_ids=("llama_cpu.prefill",), **kw)

    def test_a_declared_root_must_itself_be_operator_owned(self):
        target, sha = write_waiver_file(autokernel_waiver())
        for widened in (str(storage.REPO_ROOT), "/mnt/raid0/llm", "/workspace",
                        "/mnt/raid0/llm/tmp", str(storage.REPO_ROOT / "artifacts")):
            with self.subTest(root=widened):
                with self.assertRaises(t3.T3InputError) as caught:
                    self._read(target, sha, attestation_roots=(widened,))
                self.assertIn("not itself operator-owned", str(caught.exception))

    def test_one_bad_root_poisons_a_set_that_also_names_a_good_one(self):
        # A root set is a disjunction; a caller that appends a wide root to a narrow
        # one has widened the whole set, so ANY bad member must refuse.
        target, sha = write_waiver_file(autokernel_waiver())
        with self.assertRaises(t3.T3InputError):
            self._read(target, sha, attestation_roots=(
                str(_TEST_ATTESTATION_ROOT), str(storage.REPO_ROOT)))

    def test_max_bytes_may_narrow_but_never_raise_the_ceiling(self):
        target, sha = write_waiver_file(autokernel_waiver())
        with self.assertRaises(t3.T3InputError) as caught:
            self._read(target, sha,
                       max_bytes=schemas.MAX_OPERATOR_WAIVER_BYTES + 1)
        self.assertIn("exceeds", str(caught.exception))
        # ... and narrowing still works, so the parameter keeps its stated purpose.
        self.assertEqual(self._read(target, sha, max_bytes=4096)
                         .read_receipt.bytes_sha256, sha)

    def test_the_declared_roots_the_fixtures_use_are_legitimate(self):
        # THE COMPLIANT-PATH CONTROL. Both roots this suite and the shipping gate use
        # are operator-owned citations, so the narrowing rule does not forbid its own
        # idiom: the fixture root is `<checkout>/artifacts/operator`, the default is
        # `/workspace/artifacts/operator`.
        target, sha = write_waiver_file(autokernel_waiver())
        self.assertEqual(self._read(target, sha).read_receipt.bytes_sha256, sha)
        self.assertEqual(
            t3.waiver_binding_from_path(
                str(V8_WAIVER_PATH), pinned_sha256=V8_WAIVER_SHA, waiver_id="WAIVE-Q8",
                covers_cell_ids=("llama_cpu.prefill",),
                attestation_roots=t3.DEFAULT_ATTESTATION_ROOTS).read_receipt.byte_length,
            1267)

    def test_no_shipping_module_reaches_for_a_relaxable_guard(self):
        check = t3.audit_reader_narrowing_is_never_widened()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_narrowing_audit_bites_on_a_module_that_widens(self):
        # THE BITE for the static half. A runtime clamp bounds how far a caller may
        # go; only this says no shipping caller goes there at all.
        scratch = Path(tempfile.mkdtemp(dir=_waiver_file_dir()))
        (scratch / "widener.py").write_text(
            "from .t3 import waiver_binding_from_path\n"
            "def go(p):\n"
            "    return waiver_binding_from_path(p, pinned_sha256='x', waiver_id='w',\n"
            "                                    covers_cell_ids=(),\n"
            "                                    attestation_roots=('/',))\n",
            encoding="utf-8")
        check = t3.audit_reader_narrowing_is_never_widened(package_root=scratch)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("attestation_roots", " ".join(check.reasons))

    def test_the_narrowing_audit_ignores_the_test_files_that_legitimately_widen(self):
        # THE COMPLIANT-PATH CONTROL for the audit: this very file passes
        # `attestation_roots=` dozens of times and must not be a finding, or the guard
        # forbids the fixture idiom it exists to permit.
        scratch = Path(tempfile.mkdtemp(dir=_waiver_file_dir()))
        (scratch / "test_widener.py").write_text(
            "waiver_binding_from_path('p', attestation_roots=('/',), max_bytes=1)\n",
            encoding="utf-8")
        (scratch / "ok.py").write_text("waiver_binding_from_path('p')\n",
                                       encoding="utf-8")
        check = t3.audit_reader_narrowing_is_never_widened(package_root=scratch)
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)


class TestTheConstructorDisciplineOutlivesThisSession(unittest.TestCase):
    """`waiver_binding_from_path` is the ONLY trusted constructor — as a checked fact.

    Migrating the call sites that existed on 2026-08-03 fixed 2026-08-03. The defect
    comes back the first time somebody writes `WaiverBinding(document=...,
    document_path=..., observed_sha256=...)` in a new module, and that edit reads as
    entirely ordinary in review: it is a frozen dataclass with public fields being
    used as one. Nothing about the type's SHAPE stops it — `verify_waiver` refuses to
    trust the result, which is the fix, but the second unhardened construction site
    is still a thing a future session will write without noticing.

    So the discipline is an audit over the AST, not a convention in a docstring.
    """

    #: A corpus that DEFINES the three types, so the audit's identity check is
    #: satisfied and its answer is about waiver code rather than about an empty
    #: directory. Everything below writes this file plus the module under test.
    _DEFINITIONS = (
        "class WaiverBinding:\n    pass\n\n\n"
        "class ReadWaiver(WaiverBinding):\n    pass\n\n\n"
        "class WaiverReadReceipt:\n    pass\n"
    )

    def _corpus(self, **modules) -> Path:
        scratch = Path(tempfile.mkdtemp(dir=_waiver_file_dir()))
        (scratch / "types_stub.py").write_text(self._DEFINITIONS, encoding="utf-8")
        for name, text in modules.items():
            (scratch / f"{name}.py").write_text(text, encoding="utf-8")
        return scratch

    def _audit(self, **modules):
        return t3.audit_waiver_binding_is_constructed_only_by_the_reader(
            package_root=self._corpus(**modules))

    def test_the_shipping_package_builds_no_waiver_outside_the_allowlist(self):
        """THE COMPLIANT PATH, over the real tree: the package as it stands passes."""
        check = t3.audit_waiver_binding_is_constructed_only_by_the_reader()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_audit_actually_sees_the_construction_sites_that_exist(self):
        """ANTI-VACUITY, against the REAL package rather than a fixture.

        A guard that finds nothing and a guard that looks for nothing are the same
        green. Emptying the allowlist must turn up exactly the three real sites —
        which is also the proof that the allowlist is not carrying a fourth entry
        nobody noticed.
        """
        saved = t3._WAIVER_CONSTRUCTOR_SITES
        try:
            t3._WAIVER_CONSTRUCTOR_SITES = {name: {} for name in saved}
            check = t3.audit_waiver_binding_is_constructed_only_by_the_reader()
        finally:
            t3._WAIVER_CONSTRUCTOR_SITES = saved
        self.assertEqual(
            t3.audit_waiver_binding_is_constructed_only_by_the_reader().outcome,
            schemas.PASS, "the allowlist must be restored")
        self.assertEqual(check.outcome, schemas.FAIL)
        found = {r.split(": ", 1)[0].split(":")[0] + "::" +
                 r.split("constructs ")[1].split("(")[0] for r in check.reasons}
        self.assertEqual(found, {"release/t3.py::ReadWaiver",
                                 "release/t3.py::WaiverReadReceipt",
                                 "release/t3.py::WaiverBinding"})
        self.assertEqual(len(check.reasons), 3, check.reasons)

    def test_the_allowlist_is_exactly_the_three_sites_it_documents(self):
        """Widening the allowlist must be a VISIBLE edit, not a silent one.

        Without this, the cheapest way to make the audit green after adding a fourth
        construction is to add a fourth allowlist entry — which is a one-line diff
        that looks like configuration and is in fact a re-opening of §10.4.
        """
        sites = t3._WAIVER_CONSTRUCTOR_SITES
        self.assertEqual(set(sites), {"WaiverBinding", "ReadWaiver",
                                      "WaiverReadReceipt"})
        self.assertEqual(set(sites["ReadWaiver"]),
                         {("release/t3.py", "waiver_binding_from_path")})
        self.assertEqual(set(sites["WaiverReadReceipt"]),
                         {("release/t3.py", "waiver_binding_from_path")})
        self.assertEqual(set(sites["WaiverBinding"]),
                         {("release/t3.py", "calibration_request")})

    def test_the_guard_bites_on_a_new_direct_construction(self):
        """THE BITE. A new non-test module builds a quotation; the audit must FAIL."""
        check = self._audit(promoter=(
            "from .t3 import WaiverBinding\n\n\n"
            "def promote(doc, sha):\n"
            "    return WaiverBinding(waiver_id='W', pinned_sha256=sha,\n"
            "                         document=doc, document_path='artifacts/"
            "operator/w.json',\n"
            "                         covers_cell_ids=(), observed_sha256=sha)\n"))
        self.assertEqual(check.outcome, schemas.FAIL, check.reasons)
        self.assertIn("promoter.py", " ".join(check.reasons))
        self.assertIn("waiver_binding_from_path", " ".join(check.reasons))

    def test_the_guard_bites_on_an_attribute_qualified_construction(self):
        """`t3.WaiverBinding(...)` is the same construction spelled differently."""
        check = self._audit(promoter=(
            "from . import t3\n\n\n"
            "def promote(doc):\n"
            "    return t3.WaiverBinding(document=doc)\n"))
        self.assertEqual(check.outcome, schemas.FAIL, check.reasons)

    def test_the_guard_bites_on_a_module_level_construction(self):
        """Not every construction is inside a function; `<module>` is not allowlisted."""
        check = self._audit(promoter="from .t3 import ReadWaiver\nDEFAULT = ReadWaiver()\n")
        self.assertEqual(check.outcome, schemas.FAIL, check.reasons)
        self.assertIn("<module>()", " ".join(check.reasons))

    def test_the_guard_bites_on_an_alias_that_would_walk_past_the_call_scan(self):
        """The obvious evasion, and the reason check 2 exists.

        `_WB = t3.WaiverBinding` followed by `_WB(...)` constructs exactly the same
        object while naming something the call scan does not look for. Found here
        because it is a rebinding, not because anyone predicted the alias's name.
        """
        check = self._audit(sneaky=(
            "from . import t3\n\n"
            "_WB = t3.WaiverBinding\n\n\n"
            "def go(doc):\n"
            "    return _WB(document=doc)\n"))
        self.assertEqual(check.outcome, schemas.FAIL, check.reasons)
        self.assertIn("aliases WaiverBinding", " ".join(check.reasons))

    def test_a_nested_helper_does_not_inherit_the_allowlist(self):
        """The allowlist is keyed by the NEAREST enclosing function.

        Otherwise the cheapest evasion is to define the new constructor inside the
        allowlisted one, where an `any enclosing function` rule would bless it.
        """
        check = self._audit(t3=(
            "def waiver_binding_from_path(p):\n"
            "    def _inner(doc):\n"
            "        return ReadWaiver(document=doc)\n"
            "    return _inner\n"))
        self.assertEqual(check.outcome, schemas.FAIL, check.reasons)
        self.assertIn("_inner()", " ".join(check.reasons))

    def test_the_guard_does_not_forbid_passing_the_class_as_a_value(self):
        """COMPLIANT-PATH CONTROL 1 — the package's own legitimate idioms.

        `isinstance(x, WaiverBinding)` and `_typed_tuple(..., WaiverBinding)` are how
        the packager and the request validator already work, and a guard that
        forbade them would forbid the type system it is defending.
        """
        check = self._audit(consumer=(
            "from .t3 import WaiverBinding, ReadWaiver\n\n\n"
            "def use(x, xs):\n"
            "    assert isinstance(x, WaiverBinding)\n"
            "    assert not isinstance(x, ReadWaiver)\n"
            "    return _typed_tuple(xs, 'waivers', WaiverBinding)\n"))
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_guard_does_not_forbid_the_test_suites_own_idiom(self):
        """COMPLIANT-PATH CONTROL 2 — and the most important one.

        Most of the evidence that this fix WORKS is tests that build a quotation and
        prove it suppresses nothing. A guard scoped over test modules would forbid
        the compliant path of the fix it is guarding; this very file constructs
        `t3.WaiverBinding(**fields)` in `quoted_waiver()` and must never be a finding.
        """
        check = self._audit(test_promoter=(
            "from .t3 import WaiverBinding, ReadWaiver, WaiverReadReceipt\n"
            "q = WaiverBinding(document={})\n"
            "r = ReadWaiver(document={})\n"
            "rec = WaiverReadReceipt()\n"))
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)
        # ...and the live proof, over the real corpus: this suite's own quotation
        # helper is a direct construction in a test module, and the package passes.
        self.assertNotIsInstance(quoted_waiver(), t3.ReadWaiver)
        self.assertEqual(
            t3.audit_waiver_binding_is_constructed_only_by_the_reader().outcome,
            schemas.PASS)

    def test_a_corpus_that_does_not_define_the_types_is_could_not_check(self):
        """The anti-vacuity guard's OWN bite.

        A root holding no waiver code satisfies "no unlisted construction" perfectly.
        That must not read as PASS, or the audit quietly stops covering anything the
        day someone moves `t3.py` — the exact failure mode it exists to prevent.
        """
        scratch = Path(tempfile.mkdtemp(dir=_waiver_file_dir()))
        (scratch / "unrelated.py").write_text("X = 1\n", encoding="utf-8")
        check = t3.audit_waiver_binding_is_constructed_only_by_the_reader(
            package_root=scratch)
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK, check.reasons)

    def test_an_empty_or_unparseable_corpus_is_could_not_check(self):
        scratch = Path(tempfile.mkdtemp(dir=_waiver_file_dir()))
        self.assertEqual(
            t3.audit_waiver_binding_is_constructed_only_by_the_reader(
                package_root=scratch).outcome, schemas.COULD_NOT_CHECK)
        (scratch / "broken.py").write_text("def (\n", encoding="utf-8")
        self.assertEqual(
            t3.audit_waiver_binding_is_constructed_only_by_the_reader(
                package_root=scratch).outcome, schemas.COULD_NOT_CHECK)

    def test_the_allowlisted_quotation_fallback_still_blocks_the_run(self):
        """The one allowlisted `WaiverBinding(` is not a loophole.

        `calibration_request` builds a quotation when a preserved freeze has a waiver
        document but no waiver PATH. It is allowlisted because refusing to build it
        would delete the honest answer — but it must remain worthless as authority,
        or the allowlist entry is a hole with a comment on it.
        """
        pathless = t3.preserved_freeze_from_v8_artifacts(V8_RATIFICATION, V8_WAIVER)
        request = t3.calibration_request(pathless, now=NOW, include_waiver=True)
        binding = request.waivers[0]
        self.assertNotIsInstance(binding, t3.ReadWaiver)
        self.assertTrue(t3.waiver_read_violations(binding))
        result = t3.run_t3(request)
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(result.receipt.suppressed_claims, ())


class TestTheFixturesDoNotManufactureTheExploitTheyTestFor(unittest.TestCase):
    """The suite must not leave operator-SHAPED files in `<checkout>/artifacts/operator`.

    Not cosmetic. `schemas.operator_owned_path_check` PASSes anything spelled
    `artifacts/operator/…` inside a known repository checkout, and the entire reason
    `t3.DEFAULT_ATTESTATION_ROOTS` exists is that this repo's own agents can create
    that directory with one `mkdir -p`. A suite that leaves waiver JSON lying there
    is standing up the exploit fixture the reader was hardened against — and this
    one did: `test_t3_waiver_authority_redteam` imports `read_waiver` from this
    module but defines its own `tearDownModule`, so running that file ALONE (which
    `make test` does, now that it is in PYTEST_SMOKE) created a directory here and
    cleaned a different one. Ten had accumulated before it was noticed.
    """

    def test_the_cleanup_is_registered_by_whoever_creates_the_directory(self):
        """THE FIX, as a property: creation registers cleanup, in the same call.

        A module-teardown cleanup is a fact about which module unittest happened to
        run. `atexit` makes it a fact about the PROCESS that created the directory,
        which is the only entity that always exists.
        """
        # Observed at the registration itself rather than through
        # `atexit._ncallbacks()`, which does NOT decrease on `unregister` in this
        # CPython — a counter that cannot go down cannot witness a registration.
        saved = globals()["_WAIVER_FILE_DIR"]
        real_register = atexit.register
        seen: list = []

        def spy(func, *args, **kwargs):
            seen.append(func)
            return real_register(func, *args, **kwargs)

        try:
            globals()["_WAIVER_FILE_DIR"] = None          # force the creating branch
            atexit.register = spy
            created = _waiver_file_dir()
        finally:
            atexit.register = real_register
            globals()["_WAIVER_FILE_DIR"] = saved
        shutil.rmtree(created, ignore_errors=True)        # leave nothing behind
        self.assertIn(
            _remove_waiver_file_dir, seen,
            "_waiver_file_dir() must atexit-register its own cleanup; without it a "
            "test module that imports this one's helpers orphans a directory inside "
            "artifacts/operator/ on every isolated run")

    def test_the_cleanup_is_idempotent_so_both_entry_points_are_safe(self):
        """`tearDownModule` and `atexit` both fire in a normal run; the second must
        not explode on an already-removed directory, or the fix trades a leak for a
        teardown error."""
        target = _waiver_file_dir()
        self.assertTrue(target.is_dir())
        _remove_waiver_file_dir()
        self.assertFalse(target.exists())
        _remove_waiver_file_dir()                          # must be a no-op
        # Re-create so the rest of the module still has somewhere to write, and so
        # this test leaves no state behind either.
        self.assertTrue(_waiver_file_dir().is_dir())

    def test_the_fixture_root_is_the_directory_the_reader_refuses_by_default(self):
        """Anti-vacuity, and the reason the two facts belong in one test.

        If `_TEST_ATTESTATION_ROOT` ever stopped being an operator-SHAPED path, the
        hygiene above would be pointless — and so would the tests that use it to
        prove the default root set refuses a checkout's own `artifacts/operator`.
        """
        self.assertEqual(_TEST_ATTESTATION_ROOT.name, "operator")
        self.assertEqual(_TEST_ATTESTATION_ROOT.parent.name, "artifacts")
        self.assertNotIn(str(_TEST_ATTESTATION_ROOT.resolve()),
                         t3.DEFAULT_ATTESTATION_ROOTS)
        # ...and it really is operator-SHAPED, which is the whole trap.
        self.assertEqual(
            schemas.operator_owned_path_check(
                str(_waiver_file_dir()), boundary=t3.human_only_boundary()).outcome,
            schemas.PASS)


class TestTheReaderRefusesADocumentThatMeansTwoThings(unittest.TestCase):
    """A digest is honest about BYTES and silent about how they are READ.

    Two decoding permissivenesses let one file mean one thing to the operator who
    ratified it and another to the gate. Neither is a hash failure: the pin matches
    perfectly in both.
    """

    def _read(self, raw, **kw):
        target, sha = write_waiver_file(None, raw=raw)
        kw.setdefault("attestation_roots", (str(_TEST_ATTESTATION_ROOT),))
        return t3.waiver_binding_from_path(
            target, pinned_sha256=sha, waiver_id="W",
            covers_cell_ids=("llama_cpu.prefill",), **kw)

    def test_a_repeated_key_is_a_refusal_not_a_last_wins(self):
        # MEASURED: `{"protocol_changed": true, ..., "protocol_changed": false}`
        # parsed as False, took `protocol_stable: PASS`, verified, and suppressed its
        # cell, while the operator scrolling the file they signed reads `true`.
        document = dict(V8_WAIVER)
        raw = json.dumps(document).encode("utf-8")
        doctored = raw.replace(
            b'"protocol_changed": false',
            b'"protocol_changed": true, "protocol_changed": false', 1)
        self.assertNotEqual(doctored, raw)
        with self.assertRaises(t3.WaiverNotReadable) as caught:
            self._read(doctored)
        self.assertIn("duplicate key 'protocol_changed'", str(caught.exception))

    def test_a_repeated_key_nested_inside_scope_is_also_refused(self):
        raw = (b'{"schema": "epyc.autokernel.operator_waiver.v1", '
               b'"scope": {"excluded_pairs": ["a"], "excluded_pairs": ["b"]}}')
        with self.assertRaises(t3.WaiverNotReadable) as caught:
            self._read(raw)
        self.assertIn("duplicate key 'excluded_pairs'", str(caught.exception))

    def test_a_byte_order_mark_is_refused_in_every_encoding_json_sniffs(self):
        # MEASURED: a UTF-16-LE waiver verified END TO END here — read PASS, hash
        # PASS, human_attested PASS, cell covered — while `json.loads(raw.decode(
        # "utf-8"))`, which is what the v8 freeze script and `jq` do to the same
        # bytes, raised UnicodeDecodeError on byte 0. An authority document one of
        # whose two readings is "this file is unreadable" is the parser differential
        # §10.4 can least afford.
        body = json.dumps(dict(V8_WAIVER))
        for encoding, name in (("utf-8-sig", "UTF-8"), ("utf-16", "UTF-16"),
                               ("utf-16-be", None), ("utf-32", "UTF-32")):
            raw = (body.encode(encoding) if name != "UTF-16-BE"
                   else b"\xfe\xff" + body.encode("utf-16-be"))
            if encoding == "utf-16-be":
                raw = b"\xfe\xff" + body.encode("utf-16-be")
            with self.subTest(encoding=encoding):
                with self.assertRaises(t3.WaiverNotReadable) as caught:
                    self._read(raw)
                self.assertIn("byte-order mark", str(caught.exception))

    def test_bytes_that_are_not_utf8_at_all_are_refused(self):
        raw = json.dumps(dict(V8_WAIVER)).encode("utf-8").replace(b"Q8", b"\xff\xfe")
        with self.assertRaises(t3.WaiverNotReadable) as caught:
            self._read(raw)
        self.assertIn("not UTF-8", str(caught.exception))

    def test_the_same_discipline_applies_to_the_ratification_document(self):
        # The authenticity cross-check reads a second operator file, and a permissive
        # parse there is the same defect one level out.
        target, sha = write_waiver_file(autokernel_waiver())
        rat = _waiver_file_dir() / "ratify_doctored.json"
        rat.write_bytes(b'{"evidence_sha256": {"waive_q8": "' + sha.encode() + b'"}, '
                        b'"evidence_sha256": {"waive_q8": "' + ("0" * 64).encode()
                        + b'"}}')
        with self.assertRaises(t3.WaiverNotReadable) as caught:
            t3.waiver_binding_from_path(
                target, pinned_sha256=sha, waiver_id="W",
                covers_cell_ids=("llama_cpu.prefill",),
                ratification_pin=(str(rat), "waive_q8"),
                attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertIn("duplicate key", str(caught.exception))

    def test_the_genuine_v8_record_is_plain_utf8_with_no_repeated_key(self):
        # THE COMPLIANT-PATH CONTROL, and the one that matters most: the ratified
        # record must survive both rules. It does, at the DEFAULT root, with the
        # ratification cross-check on.
        binding = t3.waiver_binding_from_path(
            str(V8_WAIVER_PATH), pinned_sha256=V8_WAIVER_SHA, waiver_id="WAIVE-Q8",
            covers_cell_ids=("llama_cpu.pair.qwen36_q8-pp2048-iqk1",),
            ratification_pin=(str(V8_RATIFICATION_PATH), "waive_q8"))
        self.assertEqual(binding.observed_sha256, V8_WAIVER_SHA)
        raw = V8_WAIVER_PATH.read_bytes()
        self.assertEqual(raw.decode("utf-8"), raw.decode("utf-8"))  # strict-UTF-8
        self.assertFalse(any(raw.startswith(m) for m, _ in t3._BYTE_ORDER_MARKS))


class TestEveryRefusalIsAWaiverNotReadable(unittest.TestCase):
    """`WaiverNotReadable` documents itself as covering every refusal. It did not.

    An embedded NUL byte made `Path.lstat` raise `ValueError: embedded null character
    in path`, and a permission failure made `read_bytes` raise a bare `OSError` —
    both escaping the exception type this module tells its callers to expect, so a
    driver catching `T3Error` to RECORD a refusal crashed instead of recording one.
    """

    def _read(self, path, sha=_PROBE_SHA, **kw):
        kw.setdefault("attestation_roots", (str(_TEST_ATTESTATION_ROOT),))
        return t3.waiver_binding_from_path(
            path, pinned_sha256=sha, waiver_id="W",
            covers_cell_ids=("llama_cpu.prefill",), **kw)

    def test_a_nul_byte_in_the_citation_is_a_refusal(self):
        for citation in ("artifacts/operator/w\x00.json",
                         str(_TEST_ATTESTATION_ROOT) + "/w\x00.json"):
            with self.subTest(citation=citation):
                with self.assertRaises(t3.WaiverNotReadable):
                    self._read(citation)

    def test_an_unreadable_file_is_a_refusal_not_a_bare_oserror(self):
        target, sha = write_waiver_file(autokernel_waiver())
        os.chmod(target, 0o000)
        try:
            with self.assertRaises(t3.WaiverNotReadable) as caught:
                self._read(target, sha)
            self.assertIn("could not be read", str(caught.exception))
        finally:
            os.chmod(target, 0o644)

    def test_every_refusal_is_catchable_as_T3Error(self):
        # The contract a driver actually writes: one `except t3.T3Error`.
        self.assertTrue(issubclass(t3.WaiverNotReadable, t3.T3Error))
        cases = ("artifacts/operator/w\x00.json",
                 "artifacts/operator/definitely-absent-12345.json",
                 str(_waiver_file_dir()))
        for citation in cases:
            with self.subTest(citation=citation):
                with self.assertRaises(t3.T3Error):
                    self._read(citation)

    def test_a_readable_file_is_still_read(self):
        # THE COMPLIANT-PATH CONTROL: widening the caught exception set must not turn
        # a successful read into a refusal.
        target, sha = write_waiver_file(autokernel_waiver())
        self.assertEqual(self._read(target, sha).observed_sha256, sha)


class TestTheFingerprintIsBlindToTheRead(unittest.TestCase):
    """§9.1 idempotence is over the EVIDENCE, not over the filesystem metadata.

    A rerun that re-reads the same bytes from the same path at a different inode or
    mtime is the SAME run. Hashing the receipt would send it into
    REFUSED_UNCHANGED_FINGERPRINT — fail-closed and still wrong.
    """

    def test_a_read_and_a_quoted_waiver_over_the_same_digest_agree(self):
        document = autokernel_waiver()
        read = read_waiver(document)
        quoted = quoted_waiver(document, pinned_sha256=read.pinned_sha256,
                               observed_sha256=read.observed_sha256,
                               document_path=read.document_path)
        self.assertEqual(request(waivers=(read,)).fingerprint(),
                         request(waivers=(quoted,)).fingerprint())

    def test_two_reads_of_different_bytes_do_not(self):
        # Anti-vacuity: the fingerprint must still move when the DOCUMENT moves.
        first = read_waiver(autokernel_waiver())
        second = read_waiver(autokernel_waiver(reason="a different stated reason"))
        self.assertNotEqual(first.pinned_sha256, second.pinned_sha256)
        self.assertNotEqual(request(waivers=(first,)).fingerprint(),
                            request(waivers=(second,)).fingerprint())

    def test_the_receipt_is_not_a_fingerprint_facet(self):
        self.assertNotIn("waiver_read_receipt", t3.FINGERPRINT_FACETS)
        self.assertIn("active_waiver_sha256", t3.FINGERPRINT_FACETS)


class TestCitationCanonicalisation(unittest.TestCase):
    """A consumer that checks one spelling and opens another has no guarantee.

    Measured, not assumed: `posixpath.normpath('//x')` returns `'//x'` (POSIX leaves
    a leading double slash implementation-defined) while `'///x'` collapses to
    `'/x'`. So the same location had two answers, decided by a slash.
    """

    def test_double_and_triple_slash_now_agree_with_the_single_form(self):
        single = "/workspace/artifacts/operator/w.json"
        for spelling in (single, "//workspace/artifacts/operator/w.json",
                         "///workspace/artifacts/operator/w.json",
                         "/workspace/artifacts/./operator/w.json",
                         "/workspace/artifacts/operator//w.json"):
            with self.subTest(spelling=spelling):
                self.assertEqual(schemas.canonical_citation(spelling), single)
                self.assertEqual(
                    schemas.operator_owned_path_check(spelling).outcome, schemas.PASS)

    def test_collapsing_slashes_never_widens_the_boundary(self):
        # ANTI-VACUITY: the normalisation must not turn a refusal into a PASS.
        for spelling in ("//mnt/raid0/llm/tmp/artifacts/operator/w.json",
                         "///mnt/raid0/llm/tmp/artifacts/operator/w.json",
                         "//tmp/artifacts/operator/w.json"):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    schemas.operator_owned_path_check(
                        spelling,
                        boundary=schemas.parse_trust_boundary(BOUNDARY_YAML)).outcome,
                    schemas.FAIL)

    def test_the_receipt_records_the_canonical_citation_that_was_opened(self):
        target, sha = write_waiver_file(autokernel_waiver())
        binding = t3.waiver_binding_from_path(
            "/" + target, pinned_sha256=sha, waiver_id="W", covers_cell_ids=("llama_cpu.prefill",),
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertEqual(binding.document_path, target)
        self.assertEqual(binding.read_receipt.citation, target)

    def test_containment_is_on_segments_never_on_a_substring(self):
        self.assertFalse(schemas.under_any_root("/a/bcd", ("/a/bc",)))
        self.assertTrue(schemas.under_any_root("/a/bc/d", ("/a/bc",)))
        self.assertFalse(schemas.under_any_root("/a/b", ()))


class TestOperatorRootCannotBeManufactured(unittest.TestCase):
    """The repo-name strip must REDUCE a citation, never invent one.

    `repo_relative_forms` dropped the first path segment of any absolute citation
    under a checkout root, so `/mnt/raid0/llm/tmp/artifacts/operator/w.json`
    reduced to `artifacts/operator/w.json`. `/mnt/raid0/llm/tmp/` is the loop's own
    scratch root — `resource/device_claim.py` puts its lock files there — so the
    trust-boundary PASS the provenance check exists to withhold was obtainable with
    `mkdir -p`.
    """

    def test_the_loops_scratch_root_cannot_spell_the_operator_root(self):
        for path in ("/mnt/raid0/llm/tmp/artifacts/operator/waive-q8.json",
                     "/workspace/tmp/artifacts/operator/waive-q8.json",
                     "/mnt/raid0/llm/scratch/ak/artifacts/operator/w.json"):
            with self.subTest(path=path):
                check = schemas.operator_owned_path_check(
                    path, boundary=schemas.parse_trust_boundary(BOUNDARY_YAML))
                self.assertEqual(check.outcome, schemas.FAIL, check.reasons)

    def test_a_scratch_waiver_that_spells_the_operator_root_blocks_the_run(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(
            document_path="/mnt/raid0/llm/tmp/artifacts/operator/waive-q8.json")
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL", reasons_of(result))
        self.assertIn("llama_cpu.prefill", result.verdict_computation.failed_cells)

    def test_a_scratch_dir_cannot_wear_a_human_only_glob_either(self):
        # The same strip made `/mnt/raid0/llm/tmp/orchestration/instrument_eras.yaml`
        # match the era-registry glob, i.e. any manifest entry, not just the root.
        self.assertEqual(
            schemas.operator_owned_path_check(
                "/mnt/raid0/llm/tmp/orchestration/instrument_eras.yaml",
                boundary=ERA_BOUNDARY).outcome,
            schemas.FAIL)

    def test_a_real_checkout_still_reduces_to_its_repo_relative_form(self):
        # COMPLIANT-PATH CONTROL. The strip exists so an absolute citation inside a
        # checkout matches a manifest glob written repo-relative. That must keep
        # working, or the fix has forbidden the idiom it was protecting.
        self.assertEqual(
            schemas.operator_owned_path_check(
                "/workspace/repos/epyc-orchestrator/orchestration/instrument_eras.yaml",
                boundary=ERA_BOUNDARY).outcome,
            schemas.PASS)
        for path in ("/workspace/artifacts/operator/waive-q8.json",
                     "/mnt/raid0/llm/epyc-inference-research/artifacts/operator/w.json",
                     "/workspace/repos/epyc-orchestrator/artifacts/operator/w.json",
                     "artifacts/operator/waive-q8-v9.json"):
            with self.subTest(path=path):
                self.assertEqual(
                    schemas.operator_owned_path_check(
                        path,
                        boundary=schemas.parse_trust_boundary(BOUNDARY_YAML)).outcome,
                    schemas.PASS)

    def test_the_checkout_names_are_derived_from_the_backend_source_trees(self):
        # Not a hand-list: a backend whose tree this set does not know about would
        # silently stop reducing, so the kernel trees come from the SSOT.
        self.assertTrue(schemas.SOURCE_TREES <= schemas.REPO_CHECKOUT_NAMES)
        self.assertNotIn("tmp", schemas.REPO_CHECKOUT_NAMES)


class TestAuthorshipIsNotSatisfiedByOmission(unittest.TestCase):
    """A five-field scan is satisfiable by naming none of the five fields."""

    def test_a_legacy_waiver_with_no_author_and_no_provenance_is_refused(self):
        binding = quoted_waiver(
            v8_shaped_waiver(),
            document_path="/mnt/raid0/llm/tmp/ak/waive-q8.json")
        verification = verify(binding,
                              boundary=schemas.parse_trust_boundary(BOUNDARY_YAML))
        self.assertFalse(verification.verified)
        self.assertEqual(verification.predicate_results["human_attested"], schemas.FAIL)
        self.assertIn("a timestamp is not an author",
                      " ".join(verification.check.reasons))

    def test_an_unauthored_waiver_at_an_unknown_path_does_not_suppress_a_cell(self):
        cells, results = failing_matrix()
        binding = quoted_waiver(
            v8_shaped_waiver(),
            document_path="/mnt/raid0/llm/tmp/ak/waive-q8.json")
        result = t3.run_t3(request(_cells=cells, _results=results, waivers=(binding,)))
        self.assertEqual(result.verdict, "FAIL", reasons_of(result))
        self.assertIn("llama_cpu.prefill", result.verdict_computation.failed_cells)

    def test_an_unknown_provenance_does_not_stand_in_for_an_author(self):
        # COULD_NOT_CHECK is not PASS here either: an origin nobody can establish
        # establishes no authorship.
        binding = quoted_waiver(v8_shaped_waiver(),
                                 document_path="/mnt/raid0/llm/tmp/ak/waive-q8.json")
        verification = verify(binding, boundary=_TEST_BOUNDARY)
        self.assertEqual(verification.predicate_results["human_attested"], schemas.FAIL)

    def test_the_preserved_v8_shape_still_verifies_from_its_own_home(self):
        # COMPLIANT-PATH CONTROL. The genuine v8 waiver names no author at all, so
        # the refusal must not close the door v8 itself walked through: READ from an
        # operator-owned path it verifies, and the bundle records WHICH fact carried
        # the attribution.
        binding = read_waiver(v8_shaped_waiver())
        verification = verify(binding, boundary=_TEST_BOUNDARY)
        self.assertTrue(verification.verified, verification.check.reasons)
        self.assertEqual(verification.predicate_results["attribution_source"],
                         "operator_owned_path")
        self.assertEqual(verification.predicate_results["read"], schemas.PASS)

    def test_a_named_human_is_still_attributed_by_name(self):
        # COMPLIANT-PATH CONTROL for the other branch: the ordinary schema names an
        # author, and that is what carries it — not the path.
        verification = verify(read_waiver(), boundary=_TEST_BOUNDARY)
        self.assertTrue(verification.verified, verification.check.reasons)
        self.assertEqual(verification.predicate_results["attribution_source"],
                         "named_actor")

    def test_a_machine_name_is_refused_even_from_the_operator_root(self):
        # Provenance never launders a machine attribution: the two conditions are
        # AND-ed on the refusal side, not OR-ed.
        binding = quoted_waiver(autokernel_waiver(authorized_by="autokernel"),
                                 document_path="artifacts/operator/waive-q8-v9.json")
        verification = verify(binding,
                              boundary=schemas.parse_trust_boundary(BOUNDARY_YAML))
        self.assertFalse(verification.verified)
        self.assertEqual(verification.predicate_results["human_attested"], schemas.FAIL)


class TestSeparatorsDoNotLaunderAMachineName(unittest.TestCase):
    """The scan split on every non-alphanumeric, so a hyphen walked around it."""

    def test_a_separator_spelling_is_still_a_machine_name(self):
        for identity in ("auto-kernel", "auto_kernel", "auto.kernel", "auto pilot",
                         "Auto Kernel", "autokernel2", "sub agent", "auto-pilot"):
            with self.subTest(identity=identity):
                self.assertTrue(schemas.machine_actor_tokens(identity), identity)
                verification = verify(
                    read_waiver(autokernel_waiver(authorized_by=identity)),
                    boundary=_TEST_BOUNDARY)
                self.assertFalse(verification.verified, identity)
                self.assertIn("machine actor", " ".join(verification.check.reasons))

    def test_a_human_name_is_still_a_human_name(self):
        # COMPLIANT-PATH CONTROL. Re-joining adjacent runs must not become substring
        # matching: "scriptor" contains "script" and is a fine handle, and a
        # two-part human name must not become a token by concatenation.
        for identity in ("scriptor", "Daniele Pinna", "operator", "daniele",
                         "Anna-Maria", "d.pinna", "Jean-Luc Picard", "ops-daniele"):
            with self.subTest(identity=identity):
                self.assertEqual(schemas.machine_actor_tokens(identity), (), identity)
                verification = verify(
                    read_waiver(autokernel_waiver(authorized_by=identity)),
                    boundary=_TEST_BOUNDARY)
                self.assertTrue(verification.verified, verification.check.reasons)


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

    def _trade(self, **overrides):
        fields = {"backend": "llama_cpu", "regressing_phase": "decode",
                  "regression_band": (-0.03, -0.01), "gaining_phase": "prefill",
                  "expected_gain": 0.18, "roles_affected": ("worker_general",),
                  "declared_at": CAMPAIGN_START, "campaign_start_at": CAMPAIGN_START,
                  "operator_approved": True, "approved_by": "operator"}
        fields.update(overrides)
        return t3.PhaseTradeException(**fields)

    def _traded_run(self, gaining_standing, **trade_overrides):
        """A run where `decode` regressed under an approved trade, and `prefill`
        — the phase the trade was priced on — carries `gaining_standing`."""
        table = {("llama_cpu", "decode"): t3.STANDING_REGRESSED}
        if gaining_standing is not None:
            table[("llama_cpu", "prefill")] = gaining_standing
        keep = standings(table)
        if gaining_standing is None:
            keep = tuple(s for s in keep
                         if (s.backend, s.workload_phase) != ("llama_cpu", "prefill"))
        return t3.run_t3(request(standings=keep,
                                 phase_trades=(self._trade(**trade_overrides),)))

    def test_a_trade_whose_gaining_phase_also_regressed_is_a_finding(self):
        # `expected_gain` was validated for structure and compared to nothing, so
        # a trade could pay for a regression with a second one and pass silently.
        result = self._traded_run(t3.STANDING_REGRESSED)
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("contradicts the pre-declared expected gain", reasons_of(result))

    def test_a_trade_whose_gain_was_never_established_is_a_finding(self):
        for standing in (t3.STANDING_NON_INFERIOR, t3.STANDING_INDETERMINATE,
                         t3.STANDING_NOT_MEASURED):
            with self.subTest(standing=standing):
                result = self._traded_run(standing)
                self.assertEqual(result.verdict, "FAIL")
                self.assertIn("was not established", reasons_of(result))

    def test_a_trade_priced_on_a_phase_nobody_measured_is_a_finding(self):
        result = self._traded_run(None)
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("a regression with a story attached", reasons_of(result))

    def test_a_trade_priced_on_a_phase_that_is_not_the_backends_is_a_finding(self):
        result = self._traded_run(t3.STANDING_IMPROVED, gaining_phase="throughput")
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("no standing for that phase", reasons_of(result))

    def test_a_trade_with_no_expected_gain_is_refused_at_declaration(self):
        for gain in (0.0, -0.2, float("nan"), float("inf")):
            with self.subTest(gain=gain):
                with self.assertRaises(t3.T3InputError):
                    self._trade(expected_gain=gain)

    def test_a_realised_trade_is_admitted_and_the_note_names_the_standing(self):
        # COMPLIANT-PATH CONTROL: the comparison must not forbid the thing §1.6
        # allows. The gain happened, so the trade stands — and the receipt now
        # records the REALISED standing beside the pre-declared gain.
        result = self._traded_run(t3.STANDING_IMPROVED)
        self.assertEqual(result.verdict, "PASS", reasons_of(result))
        notes = " ".join(n for r in result.phase_results for n in r.notes)
        self.assertIn("realised standing 'improved'", notes)

    def test_the_realisation_is_recorded_in_the_phase_detail(self):
        result = self._traded_run(t3.STANDING_IMPROVED)
        detail = [r.detail for r in result.phase_results
                  if r.phase_id == t3.PHASE_CAPACITY_UTILITY][0]
        realisation = detail["phase_trade.llama_cpu.decode.realisation"]
        self.assertTrue(realisation["realised"])
        self.assertEqual(realisation["expected_gain"], 0.18)
        self.assertEqual(realisation["gaining_standing"], t3.STANDING_IMPROVED)

    def test_a_trade_the_loop_approved_is_not_an_operator_decision(self):
        # §1.6 makes the trade an operator decision. `approved_by` was the only
        # place the approver is NAMED and it was unguarded, while the
        # identically-shaped `authorized_by` on a §10.4 waiver is refused a machine
        # name — so the loop could approve its own regression by setting a boolean
        # and typing its own name.
        for approver in ("autokernel", "the controller", "ak-runner", "auto-pilot"):
            with self.subTest(approver=approver):
                result = self._traded_run(t3.STANDING_IMPROVED, approved_by=approver)
                self.assertEqual(result.verdict, "FAIL", reasons_of(result))
                self.assertIn("machine actor", reasons_of(result))

    def test_an_operator_approved_trade_is_still_admitted(self):
        # COMPLIANT-PATH CONTROL: the approver vocabulary must not refuse the people
        # who actually approve these.
        for approver in ("operator", "Daniele Pinna", "daniele"):
            with self.subTest(approver=approver):
                result = self._traded_run(t3.STANDING_IMPROVED, approved_by=approver)
                self.assertEqual(result.verdict, "PASS", reasons_of(result))

    def test_a_regression_band_that_describes_a_gain_is_refused(self):
        # `readiness.PhaseTradeException` refuses `high > 0` at declaration; T3
        # mirrored the `expected_gain` half of that refusal and not this one, so a
        # "regression band" of (0.01, 0.05) was an admissible exception here and
        # inadmissible one module away.
        for band in ((0.01, 0.05), (-0.03, 0.02)):
            with self.subTest(band=band):
                with self.assertRaises(t3.T3InputError):
                    self._trade(regression_band=band)

    def test_an_oriented_regression_band_is_still_accepted(self):
        # COMPLIANT-PATH CONTROL.
        for band in ((-0.03, -0.01), (-0.03, 0.0)):
            with self.subTest(band=band):
                self.assertEqual(self._trade(regression_band=band).check().outcome,
                                 schemas.PASS)

    def test_two_standings_for_one_phase_are_a_contradiction_not_a_preference(self):
        # Every per-phase consumer in this gate is a dict keyed on
        # (backend, workload_phase) — `owned`, and the trade-realisation lookup — so
        # a second standing for one phase WINS silently, and the party supplying it
        # is the party being gated. Here the regression is real and a second,
        # later standing overwrites it with `non_inferior`.
        regressed = standings({("llama_cpu", "decode"): t3.STANDING_REGRESSED})
        overwrite = t3.PhaseStanding(
            backend="llama_cpu", workload_phase="decode", protocol_id="P-BENCH-1",
            standing=t3.STANDING_NON_INFERIOR, cell_ids=("llama_cpu.decode",),
            evidence_ref="standing:llama_cpu.decode.second")
        result = t3.run_t3(request(standings=regressed + (overwrite,)))
        self.assertEqual(result.verdict, "FAIL", reasons_of(result))
        self.assertIn("more than one standing for the phase", reasons_of(result))

    def test_one_standing_per_phase_is_not_a_duplicate(self):
        # COMPLIANT-PATH CONTROL: four standings over four distinct (backend, phase)
        # keys is the normal shape and must stay silent.
        result = t3.run_t3(request(standings=standings()))
        self.assertEqual(result.verdict, "PASS", reasons_of(result))
        self.assertNotIn("more than one standing", reasons_of(result))

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
        self.assertEqual(len(t3.FINGERPRINT_FACETS), 18)
        self.assertIn("active_waiver_sha256", t3.FINGERPRINT_FACETS)
        self.assertIn("active_waiver_coverage", t3.FINGERPRINT_FACETS)
        self.assertIn("phase_protocol_standing", t3.FINGERPRINT_FACETS)
        self.assertIn("protocol_registry_standing", t3.FINGERPRINT_FACETS)

    def test_two_runs_whose_waivers_cover_different_cells_are_different_runs(self):
        # The digest is a fact about the DOCUMENT; the coverage is a fact about the
        # RUN. Hashing only the digest gave both runs one fingerprint, so §9.1
        # would have refused the second as "already sealed" — a rerun that
        # suppresses a different cell is not the run that was already graded.
        cells, results = failing_matrix()
        prefill = request(_cells=cells, _results=results,
                          waivers=(quoted_waiver(),)).fingerprint()
        decode = request(_cells=cells, _results=results,
                         waivers=(quoted_waiver(
                             covers_cell_ids=("llama_cpu.decode",)),)).fingerprint()
        self.assertNotEqual(prefill, decode)

    def test_the_same_waiver_over_the_same_cells_is_the_same_run(self):
        # COMPLIANT-PATH CONTROL: the coverage facet must not make an unchanged
        # rerun look new, or §9.1's idempotence is unreachable.
        cells, results = failing_matrix()
        first = request(_cells=cells, _results=results,
                        waivers=(quoted_waiver(),)).fingerprint()
        second = request(_cells=cells, _results=results, run_id="akt3-v9-002",
                         now="2026-08-03T18:00:00Z",
                         waivers=(quoted_waiver(),)).fingerprint()
        self.assertEqual(first, second)

    def test_run_id_and_timestamp_do_not_perturb_the_fingerprint(self):
        first = request().fingerprint()
        second = request(run_id="akt3-v9-002", now="2026-08-03T18:00:00Z").fingerprint()
        self.assertEqual(first, second)

    def test_adding_a_waiver_is_evidence_affecting(self):
        cells, results = failing_matrix()
        without = request(_cells=cells, _results=results).fingerprint()
        with_waiver = request(_cells=cells, _results=results,
                              waivers=(quoted_waiver(),)).fingerprint()
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
        # The PATHS as well as the documents. `V8_WAIVER` below is a REDUCTION of the
        # real record and does not hash to `V8_WAIVER_SHA`; it is kept because the
        # freeze reader consumes the mapping, but the §10.4 authority document is now
        # READ from `/workspace/artifacts/operator/` and cross-checked against
        # `evidence_sha256.waive_q8` in the ratification beside it.
        for path in (V8_WAIVER_PATH, V8_RATIFICATION_PATH):
            if not path.is_file():
                self.skipTest(f"{path} is not present on this host")
        self.freeze = t3.preserved_freeze_from_v8_artifacts(
            V8_RATIFICATION, V8_WAIVER, waiver_path=str(V8_WAIVER_PATH),
            ratification_path=str(V8_RATIFICATION_PATH))

    def test_the_calibration_waiver_is_read_not_quoted(self):
        """THE HEADLINE. The §10.4 calibration's authority document used to be a
        caller-supplied mapping pinned to a digest asserted equal to itself, at
        `artifacts/operator/<label>/waiver.json` — a path that has never existed.
        """
        request = t3.calibration_request(self.freeze, now=NOW, include_waiver=True)
        self.assertEqual(len(request.waivers), 1)
        binding = request.waivers[0]
        self.assertIsInstance(binding, t3.ReadWaiver)
        self.assertTrue(binding.was_read)
        self.assertEqual(binding.observed_sha256, V8_WAIVER_SHA)
        self.assertEqual(binding.read_receipt.resolved_path, str(V8_WAIVER_PATH))
        # The authenticity fact: the ratification hashes its own waiver.
        self.assertEqual(binding.read_receipt.ratification_pin, V8_WAIVER_SHA)
        self.assertEqual(binding.document["schema"], t3.WAIVER_SCHEMA_V8_CPU_PREFILL)

    def test_the_calibration_without_a_path_blocks_instead_of_trusting(self):
        """A behaviour CHANGE, and the correct one. A freeze built from the mapping
        alone can still be calibrated, but its waiver is a quotation, so the run
        blocks rather than reading PASS_WITH_WAIVER off a document nobody opened.
        """
        pathless = t3.preserved_freeze_from_v8_artifacts(V8_RATIFICATION, V8_WAIVER)
        request = t3.calibration_request(pathless, now=NOW, include_waiver=True)
        self.assertEqual(len(request.waivers), 1)
        self.assertNotIsInstance(request.waivers[0], t3.ReadWaiver)
        result = t3.run_t3(request)
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("was never read from disk",
                      " | ".join(result.verdict_computation.blocking_reasons))

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
            libraries=((LLAMA_BACKENDS,
                        "/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/"
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
            libraries=((LLAMA_BACKENDS,
                        "/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/"
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
        binding = read_waiver(doc, covers_cell_ids=covers)
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
                                   waivers=(read_waiver(),)))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER", reasons_of(result))

    def test_a_scope_naming_only_the_model_still_resolves_its_cells(self):
        """Operators write model/pair names, not cell ids; both must resolve."""
        doc = autokernel_waiver(scope={"excluded_models": ["llama_cpu"],
                                       "excluded_pairs": [],
                                       "remaining_matched_pairs": 14})
        cells, results = failing_matrix()
        result = t3.run_t3(request(
            _cells=cells, _results=results,
            waivers=(read_waiver(doc, covers_cell_ids=("llama_cpu.prefill",)),)))
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
            read_waiver(covers_cell_ids=("llama_gpu.quality",)),
            candidate_commit=CANDIDATE_COMMIT, production_base_commit=BASE_COMMIT,
            campaign_id="ak-v9", known_cell_ids=[c.cell_id for c in cells],
            failing_cell_ids=[], now=NOW,
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertEqual(verification.predicate_results["scope"], schemas.FAIL)
        self.assertFalse(verification.verified)

    def test_the_scope_predicate_does_not_judge_the_waivers_merits(self):
        """Two waivers differing only in the operator's REASON stay identical."""
        cells = matrix_cells()
        kwargs = dict(candidate_commit=CANDIDATE_COMMIT,
                      production_base_commit=BASE_COMMIT, campaign_id="ak-v9",
                      known_cell_ids=[c.cell_id for c in cells],
                      failing_cell_ids=["llama_cpu.prefill"], now=NOW,
                      attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        strong = t3.verify_waiver(
            read_waiver(autokernel_waiver(reason="a well-argued reason")), **kwargs)
        weak = t3.verify_waiver(
            read_waiver(autokernel_waiver(reason="because")), **kwargs)
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


# =============================================================================
# Per-phase protocol ratification (AK5 carried-forward item)
#
# `ProtocolBinding` proved the FREEZE protocol was ratified. The protocols the
# matrix cells were GRADED UNDER arrived as bare ids, so a cell measured under a
# DRAFT `P-STT-*` was indistinguishable from one measured under Annex B's
# `P-BENCH-1`, and the gate licensed claims for both.
# =============================================================================

class TestPerPhaseProtocolRatification(unittest.TestCase):

    def test_a_bare_id_is_could_not_check_never_pass(self):
        bound = t3.phase_protocol_binding(
            "P-BENCH-1", backend="llama_cpu", workload_phase="decode")
        self.assertIsNone(bound.ratified)
        self.assertEqual(bound.check().outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("BARE ID", " ".join(bound.check().reasons))

    def test_a_bare_id_blocks_the_run(self):
        """The bite: this run PASSed before the per-phase binding existed."""
        result = t3.run_t3(request(phase_protocols={
            b: dict(LLAMA_PHASE_PROTOCOLS) for b in LLAMA_BACKENDS}))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("arrives as a BARE ID", reasons_of(result))

    def test_a_draft_protocol_blocks_the_run(self):
        result = t3.run_t3(request(phase_protocols={
            b: {"prefill": draft_protocol("P-BENCH-PREFILL-1"),
                "decode": ratified_protocol("P-BENCH-1")}
            for b in LLAMA_BACKENDS}))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("declared NOT ratified", reasons_of(result))

    def test_ratified_bindings_are_the_compliant_path_and_pass(self):
        """The control: the guard must not forbid its own legitimate idiom."""
        result = t3.run_t3(request())
        self.assertEqual(result.verdict, "PASS", reasons_of(result))
        self.assertNotIn("BARE ID", reasons_of(result))
        self.assertNotIn("declared NOT ratified", reasons_of(result))

    def test_a_ratification_receipt_for_another_protocol_is_refused(self):
        with self.assertRaises(t3.T3InputError) as caught:
            t3.PhaseProtocolBinding(
                backend="llama_cpu", workload_phase="decode",
                protocol_id="P-BENCH-1", binding=ratified_protocol("P-GPU-1"))
        self.assertIn("A ratification receipt for a DIFFERENT protocol",
                      str(caught.exception))

    def test_a_binding_filed_under_another_phase_is_refused(self):
        decode = t3.phase_protocol_binding(
            ratified_protocol("P-BENCH-1"), backend="llama_cpu",
            workload_phase="decode")
        with self.assertRaises(t3.T3InputError):
            t3.phase_protocol_binding(
                decode, backend="llama_cpu", workload_phase="prefill")

    def test_the_map_still_reads_as_a_protocol_owner_in_phase_four(self):
        """Normalising the map must not delete the §1.6 ownership check it feeds."""
        crossed = {b: {"prefill": ratified_protocol("P-BENCH-PREFILL-1"),
                       "decode": ratified_protocol("P-GPU-1")}
                   for b in LLAMA_BACKENDS}
        result = t3.run_t3(request(phase_protocols=crossed))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is owned by 'P-GPU-1'", reasons_of(result))

    def test_an_unknown_backend_key_is_refused_rather_than_ignored(self):
        with self.assertRaises(t3.T3InputError):
            request(phase_protocols={"not_a_backend": {"decode": "P-BENCH-1"}})

    def test_ratifying_a_phase_protocol_moves_the_fingerprint(self):
        """§9.1's idempotence is over the evidence GRADED, and the standing of the
        instrument is part of it. A fingerprint blind to ratification would send the
        post-ratification rerun into REFUSED_UNCHANGED_FINGERPRINT."""
        self.assertIn("phase_protocol_standing", t3.FINGERPRINT_FACETS)
        draft = request(phase_protocols={
            b: {"prefill": draft_protocol("P-BENCH-PREFILL-1"),
                "decode": ratified_protocol("P-BENCH-1")}
            for b in LLAMA_BACKENDS})
        self.assertNotEqual(draft.fingerprint(), request().fingerprint())
        bare = request(phase_protocols={b: dict(LLAMA_PHASE_PROTOCOLS)
                                        for b in LLAMA_BACKENDS})
        self.assertNotEqual(bare.fingerprint(), draft.fingerprint())

    def test_an_unchanged_binding_keeps_the_fingerprint_stable(self):
        """Control: the facet must not make the key perturbable for free."""
        self.assertEqual(request().fingerprint(), request().fingerprint())


class TestDeclaredRatifiedProtocolIds(unittest.TestCase):
    """The set handed to the adapters is DERIVED from hashed bindings. A constant,
    a flag, or an adapter edit must not be able to produce a member of it."""

    def test_only_ratified_bindings_are_in_the_set(self):
        req = request(protocol_registry=(ratified_protocol("P-AK-SEARCH-1", annex="K"),
                                         draft_protocol("P-STT-1")))
        ids = t3.declared_ratified_protocol_ids(req)
        self.assertIn("P-AK-SEARCH-1", ids)
        self.assertIn("P-BENCH-1", ids)
        self.assertNotIn("P-STT-1", ids)
        # The freeze protocol is a draft in the default fixture and must not appear.
        self.assertNotIn(t3.RELEASE_PROTOCOL_ID, ids)

    def test_bare_ids_contribute_nothing(self):
        req = request(phase_protocols={b: dict(LLAMA_PHASE_PROTOCOLS)
                                       for b in LLAMA_BACKENDS})
        self.assertEqual(t3.declared_ratified_protocol_ids(req), ())


class TestSpeechAdapterReadinessIsConsulted(unittest.TestCase):
    """AK5/AK9: *"the adapters know … and nothing in the release plane calls it."*

    Driven through the PRESERVED 2026-07-31 speech freeze rather than a synthetic
    plan, so the seam is exercised on the artifact it exists for.
    """

    def setUp(self):
        self.freezes = t3.preserved_freeze_from_speech_artifact(SPEECH_RATIFICATION)
        self.freeze = self.freezes["whisper.cpp"]

    def _reasons(self, **overrides):
        base = t3.calibration_request(self.freeze, now=NOW)
        fields = {f.name: getattr(base, f.name)
                  for f in dataclasses.fields(t3.T3Request)}
        fields.update(overrides)
        return reasons_of(t3.run_t3(t3.T3Request(**fields)))

    def test_the_registry_names_both_speech_backends(self):
        self.assertEqual(sorted(t3.RELEASE_READINESS_BY_BACKEND),
                         ["qwentts_tts", "whisper_stt"])

    def test_a_draft_family_blocks_through_the_adapters_own_verdict(self):
        """The bite: nothing called `release_gate_readiness()` before this."""
        text = self._reasons()
        self.assertIn("release_gate_readiness() returns COULD_NOT_CHECK", text)

    def test_search_authority_alone_still_blocks_release(self):
        text = self._reasons(
            protocol_registry=(ratified_protocol("P-AK-SEARCH-1", annex="K"),))
        self.assertIn("release_gate_readiness() returns COULD_NOT_CHECK", text)
        self.assertIn("are absent from the supplied ratified registry", text)

    def test_a_fully_ratified_family_clears_the_readiness_seam(self):
        """The control: the seam must not forbid the operator's compliant path.

        Ratifying `P-AK-SEARCH-1` and every `whisper_stt` release protocol — each as
        a hashed `ProtocolBinding`, which is the only route there is — must make the
        adapter answer PASS and remove the readiness blocker. (The run still FAILs on
        the artifact's own holes: an unclean tree and no archived incumbent. Those
        are facts about the 2026-07-31 freeze, not about this seam.)
        """
        registry = tuple(
            ratified_protocol(pid, annex="K")
            for pid in (whisper_stt.SEARCH_PROTOCOL_ID,)
            + tuple(whisper_stt.RELEASE_PROTOCOL_IDS))
        text = self._reasons(protocol_registry=registry)
        self.assertNotIn("release_gate_readiness()", text)
        # …and the adapter itself agrees, asked with the same derived set.
        base = t3.calibration_request(self.freeze, now=NOW)
        fields = {f.name: getattr(base, f.name)
                  for f in dataclasses.fields(t3.T3Request)}
        fields["protocol_registry"] = registry
        ids = t3.declared_ratified_protocol_ids(t3.T3Request(**fields))
        self.assertEqual(whisper_stt.release_gate_readiness(ids).outcome, schemas.PASS)

    def test_a_llama_only_release_is_not_blocked_by_a_seam_it_has_no_adapter_for(self):
        """Control: a backend absent from the registry is not swept up by it."""
        result = t3.run_t3(request())
        self.assertEqual(result.verdict, "PASS", reasons_of(result))
        self.assertNotIn("release_gate_readiness()", reasons_of(result))


class TestTheReadinessAuditActuallyAudits(unittest.TestCase):
    """A registry nothing calls is the defect it was added to close. The call is a
    checked property of this module's AST, not a convention."""

    def test_the_live_module_consults_the_registry(self):
        check = t3.audit_backend_readiness_is_consulted()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_a_module_that_looks_up_and_never_calls_fails(self):
        source = (
            "RELEASE_READINESS_BY_BACKEND = {}\n"
            "def declared_ratified_protocol_ids(request):\n    return ()\n"
            "def phase_identity_preflight(request):\n"
            "    for backend in request.plan.backends:\n"
            "        readiness_of = RELEASE_READINESS_BY_BACKEND.get(backend)\n"
            "    return None\n"
        )
        check = t3.audit_backend_readiness_is_consulted(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("never calls it", " ".join(check.reasons))

    def test_a_module_that_never_reads_the_registry_fails(self):
        source = (
            "RELEASE_READINESS_BY_BACKEND = {}\n"
            "def declared_ratified_protocol_ids(request):\n    return ()\n"
            "def phase_identity_preflight(request):\n    return None\n"
        )
        check = t3.audit_backend_readiness_is_consulted(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("does not read RELEASE_READINESS_BY_BACKEND",
                      " ".join(check.reasons))

    def test_empty_source_is_could_not_check_never_pass(self):
        check = t3.audit_backend_readiness_is_consulted("")
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_foreign_source_is_could_not_check_never_pass(self):
        check = t3.audit_backend_readiness_is_consulted("def unrelated():\n    pass\n")
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_unparseable_source_is_could_not_check(self):
        self.assertEqual(
            t3.audit_backend_readiness_is_consulted("def (").outcome,
            schemas.COULD_NOT_CHECK)

    def test_the_compliant_shape_passes(self):
        """Control: the audit must accept the idiom it is written to require.

        The shape below is the LIVE module's: look the predicate up, call it, bind
        the verdict, and block on a non-PASS. This control previously ran on a
        preflight that called the predicate and threw the answer away, returning
        `None` and never touching `blocking` — which is not the compliant shape, it
        is the fail-open one, and asserting PASS on it made the control certify the
        defect. See `test_t3_protocol_binding_redteam.py` for the FAILing cases.
        """
        source = (
            "RELEASE_READINESS_BY_BACKEND = {}\n"
            "def declared_ratified_protocol_ids(request):\n    return ()\n"
            "def phase_identity_preflight(request):\n"
            "    blocking = []\n"
            "    ratified_ids = declared_ratified_protocol_ids(request)\n"
            "    for backend in request.plan.backends:\n"
            "        readiness_of = RELEASE_READINESS_BY_BACKEND.get(backend)\n"
            "        if readiness_of is None:\n            continue\n"
            "        readiness = readiness_of(ratified_ids)\n"
            "        if readiness.outcome != schemas.PASS:\n"
            "            blocking.append(f'{backend}: {readiness.reasons}')\n"
            "    return blocking\n"
        )
        self.assertEqual(
            t3.audit_backend_readiness_is_consulted(source).outcome, schemas.PASS)


# =============================================================================
# §10.5 — the archived library's BACKEND ATTRIBUTION, recorded at the source
# =============================================================================

class TestArchivedLibraryAttribution(unittest.TestCase):

    def test_an_unattributed_pair_is_refused(self):
        """The bite: `(path, sha256)` was the whole shape before this."""
        with self.assertRaises(t3.T3InputError) as caught:
            archive(libraries=(("/mnt/raid0/llm/kernels/archive/v8/cpu/libggml-base.so.0",
                                digest("v8-libggml-base")),))
        self.assertIn("UNATTRIBUTED shape", str(caught.exception))

    def test_an_empty_backend_set_is_refused(self):
        with self.assertRaises(t3.T3InputError):
            archive(libraries=(((), "/mnt/raid0/llm/kernels/archive/v8/cpu/lib.so",
                                digest("lib")),))

    def test_a_bare_string_is_not_a_backend_set(self):
        with self.assertRaises(t3.T3InputError) as caught:
            archive(libraries=(("llama_cpu",
                                "/mnt/raid0/llm/kernels/archive/v8/cpu/lib.so",
                                digest("lib")),))
        self.assertIn("not a backend SET", str(caught.exception))

    def test_an_unknown_backend_is_refused(self):
        with self.assertRaises(t3.T3InputError):
            archive(libraries=((("not_a_backend",),
                                "/mnt/raid0/llm/kernels/archive/v8/cpu/lib.so",
                                digest("lib")),))

    def test_a_shared_library_may_name_both_backends(self):
        """Control: one ggml runtime serving two backends is the normal case, and
        the guard must not force a false single attribution onto it."""
        entry = archive().entry(t3.ARCHIVE_GENERATION_N1)
        self.assertEqual(entry.attributed_backends, ("llama_cpu", "llama_gpu"))
        for backend in LLAMA_BACKENDS:
            self.assertEqual(len(entry.libraries_for(backend)), 1)
        self.assertEqual(t3.run_t3(request()).verdict, "PASS")

    def test_the_attribution_reaches_the_serialised_record(self):
        entry = archive().to_dict()["entries"][0]
        self.assertEqual(entry["libraries"][0]["backends"],
                         ["llama_cpu", "llama_gpu"])


if __name__ == "__main__":
    unittest.main()
