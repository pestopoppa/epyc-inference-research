#!/usr/bin/env python3
"""test_loop_integration.py — AK4 walks the loop, against fakes, with no inference.

WHY THIS FILE EXISTS
--------------------
The eight controller modules were built in parallel against one state machine and
each is green on its own. Nothing exercised them TOGETHER, and a suite of
individually-green modules is exactly the shape in which a seam defect survives:
every module is consistent with itself, and the disagreement lives in the gap.

The AK4 integration pass found six such disagreements, and this file is where
they stay fixed:

  1. **Two `proposal_fingerprint` implementations, one journal field.** The
     planner adapter hashed prose (`change.conceptual_change`) and the screener
     did not; both wrote `PROPOSAL_SKIPPED.payload["fingerprint"]` and
     `read_skip_history()` counted them in ONE dict against a threshold of two.
     Two skips of one concept counted 1 + 1, so §8.4's auto-blacklist never fired
     and §8.10's degradation run was computed over a key the record did not use.
  2. **Two §6.5 oracle registries sharing ONE id out of nineteen.** The compiler
     rendered `upstream llama.cpp / ggml` into the planner brief; the critic
     gated on `llama.cpp_upstream` and rejected the citation as *"not in the
     declared registry"*.
  3. **Two harvest-class vocabularies.** The critic had no `conditional`, so
     §6.5's own FlashAttention row was inexpressible in the plane that gates it.
  4. **Two hypothesis-origin vocabularies.** `hypotheses` opens at `controller`
     and `import`; `context` accepted neither and offered a `record` origin the
     store cannot produce — so a controller-opened hypothesis raised on its way
     into the brief §8.4.0 requires it to appear in.
  5. **A reserved closure word the DISPOSER did not know.** `guards` refused
     "exhausted"; `state_machine.check_stop_evidence` — the authority, and
     reachable directly through the public `stop()` — did not.
  6. **Three budget units and no converter.** The manifest declares hours, a
     proposal declares minutes, the journal ledger accumulates seconds. The
     obvious wiring makes the budget gate 60x too permissive.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO MODEL CALL, NO PROCESS. The planner and
critic providers are scripted fakes that serve dicts; the evaluator verdicts are
constructed, not measured; every file written lives under a per-test temporary
directory. A test here that would reach a real model is a defect in the test.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_loop_integration.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_loop_integration.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel import storage as ST  # noqa: E402
from autokernel.controller import composition as CP  # noqa: E402
from autokernel.controller import context as C  # noqa: E402
from autokernel.controller import critic as CR  # noqa: E402
from autokernel.controller import fingerprint as FP  # noqa: E402
from autokernel.controller import guards as G  # noqa: E402
from autokernel.controller import hypotheses as H  # noqa: E402
from autokernel.controller import oracles as O  # noqa: E402
from autokernel.controller import planner as PL  # noqa: E402
from autokernel.controller import selection as SEL  # noqa: E402
from autokernel.controller import state_machine as SM  # noqa: E402
from autokernel.evaluator import api as EV  # noqa: E402
from autokernel.evaluator import surface as SF  # noqa: E402

CAMPAIGN = "ak-llama_gpu-decode-20260803"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
V9_COMMIT = "1122334455667788990011223344556677889900"
TS = "2026-08-03T10:00:00+00:00"
BACKEND = "llama_gpu"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _loc(tag: str) -> C.SourceLocator:
    return C.SourceLocator(repo="epyc-root", path=f"docs/{tag}.md", locator="L1")


# =============================================================================
# Fakes. Neither of these knows what a network is.
# =============================================================================

class ScriptedProvider:
    """Serves pre-built `Completion`s in order and records what it was asked.

    The whole model plane of AK4 is behind `planner.Provider`, which is why a
    no-inference integration test is possible at all: swap the provider and the
    controller is unchanged. `requests` is kept so a test can assert what the
    controller sent, which is the only way to check that a brief the compiler
    built is the brief the adapter used.
    """

    def __init__(self, completions):
        self._queue = list(completions)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("ScriptedProvider ran out of scripted completions")
        data, usage = self._queue.pop(0)
        return PL.Completion(data=data, usage=usage, binding=request.binding)


def _usage(tokens=(400, 200)):
    return PL.TokenUsage(input_tokens=tokens[0], output_tokens=tokens[1])


def _planner_binding():
    return PL.ModelBinding(provider="local", model_id="planner-A", effort="high",
                           sampling_params={"temperature": 0.0, "seed": 42})


def _critic_binding():
    return PL.ModelBinding(provider="local", model_id="critic-B", effort="high",
                           sampling_params={"temperature": 0.0, "seed": 42})


# =============================================================================
# Records. Shapes are §7's, trimmed to what this file actually asserts on.
# =============================================================================

def _anchor(*, commit=V8_COMMIT, backends=(BACKEND,), binary="anchor-binary",
            linkage="anchor-linkage") -> SM.AnchorIdentity:
    return SM.AnchorIdentity(
        source_tree="llama.cpp",
        branch="production-consolidated-v8",
        commit=commit,
        binary_sha256={b: _sha(f"{binary}-{b}") for b in backends},
        linkage_sha256={b: _sha(f"{linkage}-{b}") for b in backends},
    )


def _campaign() -> dict:
    return {
        "schema": S.SCHEMA_CAMPAIGN,
        "campaign_id": CAMPAIGN,
        "backend": BACKEND,
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
            "max_wall_hours": 40.0,
            "max_gpu_hours": 10.0,
            "max_cpu_region_hours": 10.0,
            "max_candidates": 50,
            "max_controller_tokens": 1_000_000,
            "max_storage_gb": 100.0,
        },
        "readiness_reporting": {"reference_point_gain": 0.25, "reference_lcb_gain": 0.20},
        # `max_consecutive_proposal_skips` is REQUIRED by
        # `selection.planner_health_stop_request` and is NOT named by
        # `schemas.validate_campaign`'s `stop_policy` block — see
        # `test_ak4_conformance.TestDeclaredCampaignControls`. A §7.1-conforming
        # manifest can therefore omit the one input PLANNER_DEGRADED needs.
        "stop_policy": {
            "plateau_rounds": 5,
            "max_consecutive_integrity_failures": 2,
            "max_consecutive_build_failures": 3,
            "max_command_retries": 3,
            "max_consecutive_proposal_skips": 3,
        },
        "created_at": TS,
    }


def _reconciliation(candidate_id: str, backends=(BACKEND,)):
    backends = tuple(sorted(backends))
    derived = SF.AffectedSurface(
        candidate_id=candidate_id,
        backends=backends,
        link_targets=("libggml-hip.so",),
        objects=("ggml-hip/mmvq.o",),
        touched_files=("ggml/src/ggml-hip/mmvq.hip",),
        symbols=("ggml_hip_mul_mat_vec_q",),
        op_registrations=tuple(
            SF.OpRegistration(op_name="MUL_MAT", backend=b, dispatch_predicate="K>=4096")
            for b in backends),
        dispatch_predicates=("K>=4096",),
        over_approximations=(),
        axes_derived=SF.SURFACE_AXES,
        coverage=S.Check(S.PASS),
        full_tree=False,
        inputs={"diff_ref": f"diff-{candidate_id}"},
    )
    traced = SF.TracedSurface(
        candidate_id=candidate_id,
        trace_ref=f"trace-{candidate_id}",
        events=tuple(
            SF.DispatchEvent(op_name="MUL_MAT", backend=b,
                             kernel_symbol="ggml_hip_mul_mat_vec_q",
                             link_target="libggml-hip.so",
                             dispatch_predicate="K>=4096")
            for b in backends),
        truncated=False,
        completeness=S.Check(S.PASS),
        no_fallback=S.Check(S.PASS),
    )
    return SF.reconcile_surface(derived, traced)


def _candidate(candidate_id: str, reconciliation, *, status="banked",
               champion_status="frontier", base=V8_COMMIT) -> dict:
    tag = candidate_id.rsplit("-", 1)[-1]
    return {
        "schema": S.SCHEMA_CANDIDATE,
        "candidate_id": candidate_id,
        "campaign_id": CAMPAIGN,
        "proposal_id": f"akp-20260803-{tag}",
        "parent_candidate_id": None,
        "worktree": {
            "path": f"/mnt/raid0/llm/llama.cpp-{CAMPAIGN}",
            "branch": f"ak/{CAMPAIGN}/akp-{tag}",
            "source_commit": V7_COMMIT,
            "clean": True,
        },
        "source_snapshot": {
            "snapshot_sha256": _sha(f"snapshot-{candidate_id}"),
            "patch_bundle_sha256": _sha(f"patch-{candidate_id}"),
        },
        "ancestry": {
            "production_base_commit": base,
            "is_descendant_of_production_base": True,
            "proof": "git merge-base --is-ancestor -> 0",
        },
        "build": {
            "toolchain": "rocm-6.2",
            "compiler": "hipcc 6.2.0",
            "command": "cmake --build build-hip -j 96",
            "build_dir": f"/mnt/raid0/llm/tmp/ak-build/{candidate_id}",
            "log_path": f"data/{CAMPAIGN}/build/{candidate_id}.log",
            "log_sha256": _sha(f"build-log-{candidate_id}"),
        },
        "artifacts": {
            "binary_sha256": _sha(f"binary-{candidate_id}"),
            "linkage_sha256": _sha(f"linkage-{candidate_id}"),
            "library_sha256s": {"libggml.so": _sha(f"libggml-{candidate_id}")},
        },
        "dispatch": {
            "feature_flags": ["GGML_AK_WIDE_TILE"],
            "dispatch_predicate": "K >= 4096",
        },
        "affected_surface": SF.candidate_affected_surface_block(reconciliation),
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": _sha("evaluator-bundle")},
        "receipts": {
            "host_receipt": "rcpt-host-20260803T101500Z",
            "resource_claim_receipt": "rcpt-gpu-claim-0042",
        },
        "storage": {"footprint_gb": 3.0, "durability_class": "durable_untracked"},
        "evaluation_event_ids": [],
        "derived_verdicts": {},
        "controller": {
            "provider": "local", "model_id": "planner-A", "effort": "high",
            "prompt_bundle_sha256": _sha("prompt-bundle"),
        },
        "champion_status": champion_status,
        "status": status,
        "supersession_reason": None,
        "created_at": TS,
    }


def _event(event_id: str, *, candidate_id: str, tier: str, anchor: SM.AnchorIdentity,
           status="pass", backend=BACKEND, measurement_ids=None,
           created_at="2026-08-03T11:00:00+00:00") -> dict:
    record = {
        "schema": S.SCHEMA_EVALUATION_EVENT,
        "event_id": event_id,
        "campaign_id": CAMPAIGN,
        "candidate_id": candidate_id,
        "tier": tier,
        "claim_grammar": {
            "category": "CANDIDATE",
            "protocol_id": "P-AK-SEARCH-1/v1",
            "metric": "decode_tokens_per_s",
            "metric_direction": "higher_better",
            "reps": 5,
            "attestation_ref": "rcpt-host-20260803T101500Z",
        },
        "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": _sha("evaluator-bundle")},
        "artifact": {
            "source_sha256": _sha(f"snapshot-{candidate_id}"),
            "binary_sha256": _sha(f"binary-{candidate_id}"),
            "linkage_sha256": _sha(f"linkage-{candidate_id}"),
        },
        "anchor": {
            "source_commit": anchor.commit,
            "binary_sha256": anchor.binary_sha256[backend],
            "linkage_sha256": anchor.linkage_sha256[backend],
            "measurement_event_ids": (["ake-anchor-0001"] if measurement_ids is None
                                      else list(measurement_ids)),
        },
        "scope_manifest_sha256": _sha(f"scope-{candidate_id}"),
        "host_receipt": "rcpt-host-20260803T101500Z",
        "resource_claim_receipt": "rcpt-gpu-claim-0042",
        "co_residency": "single",
        "correctness": {"test_backend_ops": "pass"},
        "quality": {},
        "stability": {},
        "scope_denominator": {
            "machine_subset": "partial", "numa_nodes": [0],
            "devices": ["gfx90a:0"], "cores": 8,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "mechanism": {},
        "integrity_flags": [],
        "status": status,
        "supersedes": [],
        "created_at": created_at,
        "performance": {
            "raw_samples": [51.2, 51.4, 51.1],
            "paired_blocks": 3,
            "estimate": 51.23,
            "uncertainty": {"e_process_value": 12.4},
        },
    }
    if tier == "T0":
        record["anchor"]["measurement_event_ids"] = []
    return record


def _anchor_measurement(event_id: str, *, anchor: SM.AnchorIdentity,
                        backend=BACKEND) -> dict:
    record = _event(event_id, candidate_id="akc-20260803-base", tier="T1",
                    anchor=anchor, backend=backend, measurement_ids=[event_id])
    record["claim_grammar"]["category"] = "BASELINE"
    record["artifact"] = {
        "source_sha256": _sha("anchor-source"),
        "binary_sha256": anchor.binary_sha256[backend],
        "linkage_sha256": anchor.linkage_sha256[backend],
    }
    return record


def _draft(**over) -> dict:
    draft = {
        "hypothesis": ("Splitting the Q4_K MMVQ dispatch by row-block residency "
                       "lifts decode on gfx90a at B=1"),
        "narrative": "Reasoning about the dispatch predicate and its history.",
        "falsifier": ("If the per-op wall-share map shows mul_mat_vec_q under 12% of "
                      "decode wall time at B=1, the hypothesis is wrong"),
        "change_class": "dispatcher",
        "declared_symbol_deltas": {"added": ["ggml_cuda_mmvq_split"], "removed": [],
                                   "arity_changed": []},
        "campaign_kind": "dispatch",
        "novelty_basis": {"prior_event_ids": [],
                          "source_receipts": [f"{V8_COMMIT}:ggml/src/mmvq.cu:538"]},
        "expected_information_gain": 0.62,
        "target": {"regimes": ["decode_b1"], "ops": ["mul_mat_vec_q"],
                   "shapes": ["4096x4096xq4_K"], "models": ["gemma4-26B-A4B"]},
        "non_target": {"regimes": ["prefill_b1"], "shapes": []},
        "mechanism_prediction": {
            "bottleneck_before": "memory_bandwidth",
            "expected_counter_changes": {"MemUnitStalled": "down"},
            "expected_wall_share_ceiling": 0.30,
            "expected_end_to_end_gain": 0.05,
            "wall_share_receipt_id": "wsr-1",
        },
        "change": {
            "predicted_affected_surface": ["ggml-cuda/mmvq"],
            "files_and_symbols": ["ggml-cuda/mmvq.cu:mul_mat_vec_q"],
            "conceptual_change": "split the MMVQ dispatch predicate by row residency",
            "parameter_surface": {"rows_per_block": [4, 8]},
            "estimated_diff_size": 120,
        },
        "risks": {"correctness": [], "numerical": [], "state_or_rollback": [],
                  "resource": [], "integrity": []},
        "fallback": {"dispatch_guard": "GGML_CUDA_MMVQ_SPLIT=0",
                     "kill_switch": "compile-time flag"},
        "evaluation_plan": {
            "required_t0": ["t0.correctness.op_suite", "t0.integrity.symbol_table"],
            "required_t1": ["t1a.mul_mat_vec_q.paired"],
            "conditional_t2": [], "profiler_questions": ["MemUnitStalled"],
        },
        "resource_request": {"lane": "gpu", "expected_minutes": 40,
                             "expected_storage_gb": 3.0},
        "stop_condition": "two consecutive inconclusive T1 windows",
    }
    draft.update(over)
    return draft


def _layer_skip_receipt(layer="placement_and_launch_config", *, ceiling=0.01,
                        gap=0.20) -> dict:
    """§8.3: a cheaper layer is skipped only on ARITHMETIC — `layer_ceiling <
    measured_gap` — whose gap operand resolves to a journaled receipt."""
    return {
        "layer": layer,
        "measured_gap": gap,
        "layer_ceiling": ceiling,
        "gap_receipt_id": "ake-profile-1",
        "evidence_event_ids": ["ake-profile-1"],
        "anchor_commit": V8_COMMIT,
        "basis": "a measured launch-config sweep moves decode by at most 1% here",
    }


def _selection_block(**over) -> dict:
    block = {
        "mechanism": "mmvq-dispatch-threshold",
        "hierarchy_layer": "dispatcher",
        "conceptual_change_count": 1,
        "expected_end_to_end_gain": 0.05,
        "domains": ["llama.cpp/ggml-cuda"],
        "regime_identity": {"backend": [BACKEND], "phase": ["decode"],
                            "quant": ["Q4_K"], "batch": [1]},
        "layer_skip_receipts": [_layer_skip_receipt()],
    }
    block.update(over)
    return block


def _selection_context(**over) -> SEL.SelectionContext:
    base = dict(
        campaign_id=CAMPAIGN,
        backend=BACKEND,
        source_tree="llama.cpp",
        anchor_commit=V8_COMMIT,
        phase=SEL.PHASE_HARVEST,
        owned_domains=frozenset({"llama.cpp/ggml-cuda"}),
        correctness_oracles={"mul_mat_vec_q": "oracle.ops.mmvq"},
        real_graph_shape_digests=frozenset({S.content_hash("4096x4096xq4_K")}),
        confirmation_shape_digests=frozenset({S.content_hash("8192x8192xq4_K")}),
        wall_share_receipts={"wsr-1": 0.30},
        measured_profile={"gemm": 0.55, "elementwise_norm": 0.30, "attention": 0.15},
        evaluator_steps=frozenset({"t0.correctness.op_suite", "t0.integrity.symbol_table",
                                   "t1a.mul_mat_vec_q.paired"}),
        budget_remaining={"wall_minutes": 600.0, "gpu_minutes": 300.0,
                          "cpu_region_minutes": 300.0, "storage_gb": 50.0,
                          "candidates": 20.0},
        known_event_ids=frozenset({"ake-profile-1"}),
    )
    base.update(over)
    return SEL.SelectionContext(**base)


def _proposal_facts(**over) -> CR.ProposalFacts:
    kwargs = dict(
        derived_affected_surface=("ggml-cuda/mmvq",),
        correctness_oracles_by_surface={"ggml-cuda/mmvq": ("test-backend-ops",)},
        real_graph_shapes=frozenset({"4096x4096xq4_K"}),
        confirmation_shapes=frozenset({"8192x8192xq4_K"}),
        wall_share_receipts=frozenset({"wsr-1"}),
        backend_owned_domains=frozenset({"llama.cpp"}),
        proposal_domains=frozenset({"llama.cpp"}),
        budget=CR.BudgetEnvelope(minutes_remaining=600.0, storage_gb_remaining=50.0,
                                 candidates_remaining=12,
                                 controller_tokens_remaining=1_000_000),
        surface_reconciled=S.Check(S.PASS),
        roofline_utilisation={"basis": "achievable", "value": 0.715},
    )
    kwargs.update(over)
    return CR.ProposalFacts(**kwargs)


# =============================================================================
# The context-compiler fixture. Trimmed to what a round actually needs.
# =============================================================================

class _ContextFixture:
    """A journal plus one bootstrap-knowledge event per compiler-supplied fact,
    so that every citation the compiler emits resolves in the record."""

    FACTS = ("profile", "roofline", "constraints", "dispatch", "oracles",
             "coverage", "host", "surface", "ledger", "hypothesis")

    def __init__(self, root: str) -> None:
        self.journal = J.Journal(os.path.join(root, "journal"), campaign_id=CAMPAIGN)
        self.journal.initialize()
        self.campaign_entry = self.journal.append(J.KIND_CAMPAIGN_OPENED, _campaign())
        self.fact_ids = {}
        for name in self.FACTS:
            entry = self.journal.append(
                "PRIOR_SOURCE_VERIFIED",
                {"fact": name, "campaign_id": CAMPAIGN, "verified_against": V8_COMMIT})
            self.fact_ids[name] = entry.event_id

    def target(self) -> C.TargetScope:
        return C.TargetScope(
            backend=BACKEND, phase="decode", regime="batch_one_q4_k",
            architecture_class="dense", quant="q4_k", batch_band="batch_one",
            mechanism_classes=("bandwidth",), ops=("mul_mat_vec_q",),
            families=("quant_gemv",))

    def inputs(self, **over) -> C.ContextInputs:
        base = dict(
            campaign=_campaign(),
            journal_=self.journal,
            current_state=SM.PROPOSE,
            round_index=1,
            anchor=_anchor(),
            target=self.target(),
            role_exposure=(C.RoleExposure(
                role="worker", model_id="gemma4-26B-A4B", quant="q4_k_m",
                phase="decode", weight=1.0, event_id=self.fact_ids["profile"],
                locator=_loc("roles")),),
            wall_share=(C.WallShareRow(
                op="mul_mat_vec_q", phase="decode", regime="batch_one_q4_k",
                wall_share=0.30, mechanism_class="bandwidth", receipt_id="wsr-1",
                event_id=self.fact_ids["profile"], locator=_loc("wall"),
                shape="4096x4096"),),
            roofline=(C.RooflineUtilisation(
                regime="batch_one_q4_k", backend=BACKEND, phase="decode",
                architecture_class="dense", weight_basis=C.WEIGHT_BASIS_WHOLE_MODEL,
                bytes_per_token=1.6e10, measured_tps=64.0,
                datasheet_peak_bytes_per_s=1.638e12,
                achievable_bytes_per_s=1.4333e12,
                achievable_probe_receipt="rcpt-stream-20260803",
                event_id=self.fact_ids["roofline"], locator=_loc("roofline")),),
            compiler_constraints=(C.CompilerConstraint(
                constraint_id="gfx90a-no-async-dma", backend=BACKEND,
                statement=("gfx90a has direct global->LDS but no async DMA engine "
                           "and no SMEM-operand matrix instruction"),
                event_id=self.fact_ids["constraints"], locator=_loc("constraints")),),
            dispatch_behaviour=(C.DispatchBehaviour(
                path_id="mmvq", op="mul_mat_vec_q", predicate="ncols_y <= 4",
                fallback="dequantize_mul_mat_vec", backend=BACKEND,
                event_id=self.fact_ids["dispatch"], locator=_loc("dispatch")),),
            surfaces=(C.SurfaceRecord(
                candidate_id="akc-20260803-0001",
                derived_surface=("mul_mat_vec_q",), reconciled=True,
                event_id=self.fact_ids["surface"], locator=_loc("surface")),),
            suppressions=(
                C.SuppressionEntry(
                    entry_id="mfma-decode-kernels-are-worth-zero",
                    entry_class="HARD_CONSTRAINT",
                    content=("at batch-1 arithmetic intensity the matrix units cannot "
                             "exceed ~1.7-3.2% busy at any bandwidth"),
                    match_dimensions={"backend": BACKEND, "phase": "decode",
                                      "batch_band": "batch_one"},
                    reopen_when="batch size at or above B*",
                    evidence_grade="source_verified", breadth="family",
                    receipt=C.SourceLocator(repo="epyc-root", path="docs/roofline.md",
                                            locator="L41",
                                            content_sha256=_sha("roofline-doc")),
                    verified_against_commit=V8_COMMIT,
                    event_id=self.fact_ids["ledger"], locator=_loc("ledger")),
                C.SuppressionEntry(
                    entry_id="generic-q8-dequant-premise",
                    entry_class="MATCHED_NEGATIVE",
                    content="the generic Q8 dequant premise is falsified",
                    match_dimensions={"backend": BACKEND, "phase": "decode"},
                    reopen_when="a new dequant path lands in mmq.cu",
                    evidence_grade="protocol_bound", breadth="cell",
                    receipt=C.SourceLocator(repo="epyc-llama", path="ggml-cuda/mmq.cu",
                                            locator="L512"),
                    verified_against_commit=V8_COMMIT,
                    event_id=self.fact_ids["ledger"], locator=_loc("ledger")),
            ),
            evaluator_coverage=C.EvaluatorCoverage(
                bundle_sha256=_sha("evaluator-bundle"),
                covered_gate_classes=tuple(
                    g for g in EV.GATE_CLASSES if g != EV.GATE_QUALITY),
                gaps=(C.CoverageGap(
                    missing_class=EV.GATE_QUALITY,
                    blocked_lineage="ak/champion/llama-20260802",
                    owner="operator", deadline="2026-08-17",
                    drafted_amendment_ref="handoffs/active/measurement-debt/"),),
                event_id=self.fact_ids["coverage"], locator=_loc("coverage")),
            budget_state=C.BudgetState(
                wall_hours_used=3.5, storage_state=ST.STORAGE_OK,
                bytes_free=200 * 1024 ** 3,
                event_id=self.fact_ids["host"], locator=_loc("host")),
            oracle_registry_event_id=self.fact_ids["oracles"],
            compiled_at=TS,
        )
        base.update(over)
        return C.ContextInputs(**base)

    def open_hypothesis(self, *, origin="operator", hypothesis_id="akh-g15") -> C.OpenHypothesis:
        return C.OpenHypothesis(
            hypothesis_id=hypothesis_id,
            statement="G15's elementwise/norm cluster holds the B=128 decode time",
            falsifier="a current wall-share map showing the cluster under 20%",
            origin=origin,
            evidence_grade=H.ENTRY_GRADE,
            event_id=self.fact_ids["hypothesis"],
            locator=_loc("hypothesis"),
            opened_round=1)


class _LoopCase(unittest.TestCase):
    """Per-test temporary root. Nothing here writes outside it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def fixture(self) -> _ContextFixture:
        return _ContextFixture(self.root)

    def machine(self, journal_, *, name="controller") -> SM.ControllerStateMachine:
        return SM.ControllerStateMachine(
            journal_=journal_, root=os.path.join(self.root, name),
            campaign_id=CAMPAIGN)


# =============================================================================
# THE WALK — BOOTSTRAP to a stop, one state at a time, nothing measured
# =============================================================================

class TestLoopWalksEndToEnd(_LoopCase):

    def test_the_controller_walks_a_full_round_and_stops(self):
        fx = self.fixture()
        journal_ = fx.journal
        machine = self.machine(journal_)
        anchor = _anchor()

        # ---- BOOTSTRAP (§8.2): consistency asserted, anchor recorded ---------
        self.assertEqual(machine.state, SM.BOOTSTRAP)
        self.assertTrue(machine.begin_iteration().proceed)
        report = machine.bootstrap(anchor=anchor,
                                   views=J.rebuild_views(journal_.read_all()))
        self.assertEqual(report.view_check.outcome, S.PASS)
        self.assertEqual(machine.state, SM.DISCOVER)
        self.assertEqual(machine.anchor_store.read(), anchor)

        # ---- DISCOVER (§8.3, §8.3.1): an operator hypothesis enters context --
        tracker = H.HypothesisTracker(
            journal_=journal_, root=os.path.join(self.root, "hyp"),
            campaign_id=CAMPAIGN)
        hypothesis = H.Hypothesis(
            hypothesis_id="akh-g15-elementwise",
            statement="G15's elementwise/norm cluster holds the B=128 decode time",
            falsifier="a current wall-share map showing the cluster under 20%",
            origin=H.ORIGIN_OPERATOR,
            author="operator",
            regime={"backend": BACKEND, "phase": "decode"},
            source={"channel": "operator", "ref": "session-2026-08-03"})
        tracker.open_hypothesis(hypothesis)
        # AK-D38: whoever states it, it enters at design_prior and no higher.
        self.assertEqual(hypothesis.evidence_grade, H.GRADE_DESIGN_PRIOR)
        self.assertEqual(H.entry_grade(H.ORIGIN_OPERATOR), H.GRADE_DESIGN_PRIOR)

        open_set = tracker.planner_round_block(round_id="r1")
        self.assertEqual(open_set["open_count"], 1)
        entry = open_set["still_open"][0]
        # The seam: the store's entry becomes a compiler item without translation.
        surfaced = C.OpenHypothesis(
            hypothesis_id=entry["hypothesis_id"],
            statement=entry["statement"],
            falsifier=entry["falsifier"],
            origin=entry["origin"],
            evidence_grade=entry["entry_evidence_grade"],
            event_id=fx.fact_ids["hypothesis"],
            locator=_loc("hypothesis"),
            opened_round=1)

        bundle = C.compile_context(fx.inputs(open_hypotheses=(surfaced,)))
        self.assertIn("akh-g15-elementwise", bundle.planner_text)
        self.assertIn("akh-g15-elementwise", bundle.critic_text)
        self.assertEqual(C.audit_every_item_cited(bundle).outcome, S.PASS)
        self.assertEqual(C.audit_no_confirmation_stratum(bundle).outcome, S.PASS)
        self.assertEqual(C.audit_no_uncited_narrative(bundle).outcome, S.PASS)
        # §6.5's retired row reaches the planner rather than being filtered out.
        self.assertIn("AMD AITER", bundle.planner_text)

        machine.transition(SM.SELECT_TARGET, trigger="discover_complete",
                           reason="wall-share and roofline compiled for decode_b1")

        # ---- SELECT_TARGET -> PROPOSE ---------------------------------------
        machine.transition(SM.PROPOSE, trigger="target_selected",
                           reason="mul_mat_vec_q at decode_b1 holds 30% of wall time")

        # ---- PROPOSE (§8.4): a FAKE planner drafts against the compiled brief -
        provider = ScriptedProvider([(_draft(), _usage())])
        prompt = PL.PromptBundle(role=PL.ROLE_PLANNER, sections=(
            PL.PromptSection("task", PL.SECTION_INSTRUCTION,
                             "Propose one conceptual kernel change."),
            PL.PromptSection("context", PL.SECTION_CONTEXT, bundle.planner_text),
        ))
        drafted = PL.draft_proposal(
            provider=provider, binding=_planner_binding(), bundle=prompt,
            context=bundle, campaign_id=CAMPAIGN, proposal_id="akp-20260803-0001")
        manifest = dict(drafted.manifest)
        # The compiled brief is bound INTO the manifest, so "the planner drafted
        # this against that context" is checkable rather than asserted.
        self.assertEqual(manifest["controller"]["context_manifest_sha256"],
                         bundle.manifest_sha256)
        self.assertEqual(S.validate_proposal(manifest), [])
        manifest[SEL.SELECTION_BLOCK_KEY] = _selection_block()

        machine.transition(SM.PRE_RUN_CRITIC, trigger="proposal_drafted",
                           reason="one conceptual change, one mechanism")

        # ---- PRE_RUN_CRITIC (§6.3, §8.4): one proposal is REJECTED and skipped
        screener = SEL.ProposalScreener(journal_, campaign_id=CAMPAIGN)
        doomed = dict(manifest)
        doomed["proposal_id"] = "akp-20260803-0002"
        doomed["target"] = dict(manifest["target"], shapes=["8192x8192xq4_K"])
        rejected = screener.screen(doomed, _selection_context())
        self.assertFalse(rejected.admitted)
        self.assertIn(SEL.REJECT_TARGETS_CONFIRMATION_SHAPE,
                      [r.code for r in rejected.rejections])
        skips = [e for e in journal_.read_all() if e.kind == J.KIND_PROPOSAL_SKIPPED]
        self.assertEqual(len(skips), 1)
        # §8.4 forbids a bare discard: the skip is journaled, WITH its fingerprint,
        # and that fingerprint is the one the planner adapter would have written.
        self.assertEqual(skips[0].payload["fingerprint"],
                         PL.proposal_fingerprint(doomed))

        admitted = screener.screen(manifest, _selection_context())
        self.assertTrue(admitted.admitted, admitted.rejections)

        facts = _proposal_facts()
        gates = CR.evaluate_pre_run_gates(manifest, facts)
        blocking = [g.gate_id for g in gates
                    if g.blocking and g.check.outcome != S.PASS]
        self.assertEqual(blocking, [])
        critique = CR.critique_proposal(
            manifest=manifest, facts=facts,
            provider=ScriptedProvider([({
                "answers": {q.qid: {"outcome": S.PASS, "reasons": []}
                            for q in CR.PRE_RUN_QUESTIONS},
                "disposition": CR.DISPOSITION_ACCEPT,
                "revisions": {},
                "reasons": ["mechanism is named and the ceiling is receipted"],
            }, _usage())]),
            binding=_critic_binding(), bundle=PL.PromptBundle(
                role=PL.ROLE_PRE_RUN_CRITIC, sections=(
                    PL.PromptSection("task", PL.SECTION_INSTRUCTION, "Critique."),
                    PL.PromptSection("context", PL.SECTION_CONTEXT, bundle.critic_text),
                )),
            planner_binding=_planner_binding(),
            shared_model_reason="single-model host; independence recorded as degraded")
        self.assertEqual(critique.disposition, CR.DISPOSITION_ACCEPT)
        verified = CR.apply_pre_run_verdict(manifest, critique)
        self.assertEqual(verified["critic_verdict"]["status"], "pass")
        journal_.append(J.KIND_PROPOSAL_RECORDED, verified, campaign_id=CAMPAIGN)

        # ---- MUTATE -> BUILD -> T0 -> T1 (evaluator FAKES) -------------------
        machine.transition(SM.MUTATE, trigger="critic_accepted",
                           reason="proposal admitted by the pre-run gates")
        machine.transition(SM.BUILD, trigger="worktree_ready",
                           reason="patch applied to a clean parent")
        machine.transition(SM.T0_GATE, trigger="build_succeeded",
                           reason="candidate binary and linkage recorded")

        candidate_id = "akc-20260803-0001"
        reconciliation = _reconciliation(candidate_id)
        journal_.append(J.KIND_EVALUATION_EVENT,
                        _event("ake-t0-0001", candidate_id=candidate_id, tier="T0",
                               anchor=anchor))
        machine.transition(SM.T1_SEARCH_EVAL, trigger="t0_passed",
                           reason="correctness and integrity gates green")
        journal_.append(J.KIND_EVALUATION_EVENT,
                        _anchor_measurement("ake-anchor-0001", anchor=anchor))
        journal_.append(J.KIND_EVALUATION_EVENT,
                        _event("ake-t1-0001", candidate_id=candidate_id, tier="T1",
                               anchor=anchor))
        machine.transition(SM.POST_RUN_CRITIC, trigger="t1_complete",
                           reason="paired blocks reduced against the calibrated floor")

        # ---- POST_RUN_CRITIC (§8.8): the model INTERPRETS, gates DISPOSE -----
        verdict = EV.compute_verdict(
            tier="T1",
            gates=(EV.GateResult(gate_id="t1a.mul_mat_vec_q.paired",
                                 gate_class=EV.GATE_PERFORMANCE,
                                 check=S.Check(S.PASS)),
                   # Without a MECHANISM gate the only admissible mechanism
                   # status is `unavailable`: the critic may not report a
                   # mechanism the evaluator never observed.
                   EV.GateResult(gate_id="t1c.mechanism.memunitstalled",
                                 gate_class=EV.GATE_MECHANISM,
                                 check=S.Check(S.PASS)),),
            void_scan=EV.VoidScan(findings=(), evaluated=(), not_applicable=()),
            search_grade=EV.SearchGradeResult(
                satisfied=True, evaluated=("protocol_ratified",), failed=(),
                not_applicable=(), reasons=()),
            anchor=EV.AnchorIdentity(
                source_commit=anchor.commit,
                binary_sha256=anchor.binary_sha256[BACKEND],
                linkage_sha256=anchor.linkage_sha256[BACKEND],
                measurement_event_ids=("ake-anchor-0001",)),
            effect=EV.EffectEstimate(
                metric="decode_tokens_per_s", metric_direction="higher_better",
                value=0.052, e_value=40.0, threshold=20.0, mde=0.02,
                noise_floor=0.011, paired_blocks=12, stratum=EV.STRATUM_SELECTION,
                raw_samples=(51.2, 51.4), raw_samples_ref="raw-ake-t1-0001"))
        self.assertTrue(verdict.speed_rank_admissible)

        post = CR.classify_run(
            provider=ScriptedProvider([({
                "hypothesis_kind": "rate",
                "hypothesis_status": "confirmed",
                "mechanism_status": "confirmed",
                "wall_share": {
                    "op_share_before": 0.30, "op_delta_observed": -0.18,
                    "graph_delta_claimed": 0.052, "receipt_id": "wsr-1",
                    "explanation": "op share times op delta, cross-checked end to end"},
                "target_behaviour": {"decode_b1": "improved"},
                "non_target_behaviour": {"prefill_b1": "unchanged"},
                "signal_class": CR.SIGNAL_SIGNAL,
                "champion_interaction": "compatible",
                "champion_reason": "the split touches no path the champion holds",
                "next_experiment": {
                    "question": "row residency or occupancy?",
                    "distinguishes": ["row_residency", "occupancy"],
                    "observation": "MemUnitStalled falls while MfmaUtil stays flat",
                    "tier": "T1c", "estimated_cost_class": "small"},
                "durable_lesson": {
                    "entry_id": "dnr-mmvq-split-memory-bound",
                    "ledger_class": "CONDITIONAL_NEGATIVE",
                    "statement": ("row-residency splitting helps only while the "
                                  "quantized path is memory bound"),
                    "match_dimensions": {"backend": BACKEND, "phase": "decode"},
                    "reopen_when": "the path stops being memory bound",
                    "evidence_grade": "observation", "scope": "cell",
                    "derived_from_event_ids": ["ake-t1-0001"]},
            }, _usage())]),
            binding=_critic_binding(),
            bundle=PL.PromptBundle(role=PL.ROLE_POST_RUN_CRITIC, sections=(
                PL.PromptSection("task", PL.SECTION_INSTRUCTION, "Classify the run."),
                PL.PromptSection("context", PL.SECTION_CONTEXT, bundle.critic_text),
            )),
            verdict=verdict, manifest=verified, facts=facts, candidate_id=candidate_id,
            planner_binding=_planner_binding(),
            shared_model_reason="single-model host; independence recorded as degraded")
        self.assertEqual(
            CR.reconcile_classification(post.classification, verdict,
                                        manifest=verified, facts=facts).outcome,
            S.PASS)

        # ---- BANK_EVENT -> UPDATE_SEARCH_STATE -> CHAMPION_GUARD -------------
        machine.transition(SM.BANK_EVENT, trigger="post_run_classified",
                           reason="mechanism confirmed and reconciled with the gates")
        journal_.append(J.KIND_CANDIDATE_RECORDED,
                        _candidate(candidate_id, reconciliation))
        machine.transition(SM.UPDATE_SEARCH_STATE, trigger="candidate_banked",
                           reason="frontier and ledger updated from the record")
        machine.transition(SM.CHAMPION_GUARD, trigger="search_state_updated",
                           reason="one banked candidate is eligible for composition")

        # ---- CHAMPION_GUARD (§8.9): compose, re-measure the COMBINED id ------
        record = next(e.payload for e in journal_.read_all()
                      if e.kind == J.KIND_CANDIDATE_RECORDED)
        frontier = CP.admit_to_frontier(record, reconciliation,
                                        mechanism_class="dispatcher")
        lineage = CP.propose_lineage([frontier], anchor_commit=V8_COMMIT)
        combined_id = "akc-20260803-comb"
        combined_reconciliation = _reconciliation(combined_id)
        journal_.append(J.KIND_CANDIDATE_RECORDED,
                        _candidate(combined_id, combined_reconciliation,
                                   champion_status="composed_champion"))
        for tier, event_id in (("T0", "ake-t0-comb"), ("T1", "ake-t1-comb")):
            journal_.append(J.KIND_EVALUATION_EVENT,
                            _event(event_id, candidate_id=combined_id, tier=tier,
                                   anchor=anchor,
                                   created_at="2026-08-03T12:00:00+00:00"))
        champion = CP.compose_champion(
            lineage, combined_candidate_id=combined_id,
            combined_reconciliation=combined_reconciliation,
            views=J.rebuild_views(journal_.read_all()),
            recorded_anchor=anchor, observed_anchor=anchor, storage_gb=3.0)
        self.assertEqual(S.validate_champion(champion), [])
        self.assertEqual(champion["member_candidates"], [candidate_id])
        CP.record_champion(journal_, champion)

        # ---- a STOP the machine agrees with ---------------------------------
        machine.stop(
            SM.PLATEAU_STOP,
            reason=("closed for the mmvq dispatch sub-scope; the fusion sub-scope "
                    "is deferred with its T1 un-run"),
            detail={
                "closed": [{"sub_scope": "gpu/decode/mmvq-dispatch",
                            "gates_met": ["T0", "T1"]}],
                "deferred": [{"sub_scope": "gpu/decode/fusion",
                              "gates_unrun": ["T1", "T2"]}],
                "planner_health": {"proposal_skipped_count": 1,
                                   "repeated_fingerprint_count": 0,
                                   "degraded_ruled_out": True},
            })
        self.assertEqual(machine.state, SM.PLATEAU_STOP)
        self.assertTrue(machine.is_stopped())
        self.assertFalse(machine.begin_iteration().proceed)

        # The whole walk is reconstructible from the record alone.
        replayed = self.machine(journal_)
        self.assertEqual(replayed.state, SM.PLATEAU_STOP)
        states = [t.to_state for t in replayed.ledger.read().transitions]
        self.assertEqual(states, [
            SM.DISCOVER, SM.SELECT_TARGET, SM.PROPOSE, SM.PRE_RUN_CRITIC, SM.MUTATE,
            SM.BUILD, SM.T0_GATE, SM.T1_SEARCH_EVAL, SM.POST_RUN_CRITIC, SM.BANK_EVENT,
            SM.UPDATE_SEARCH_STATE, SM.CHAMPION_GUARD, SM.PLATEAU_STOP,
        ])

    def test_bootstrap_refuses_an_empty_view_over_a_non_empty_journal(self):
        """§8.2 step 10 — the failure that cost 232 trials and ~16 days."""
        fx = self.fixture()
        fx.journal.append(J.KIND_CANDIDATE_RECORDED,
                          _candidate("akc-20260803-0001",
                                     _reconciliation("akc-20260803-0001")))
        machine = self.machine(fx.journal)
        emptied = dataclasses.replace(
            J.rebuild_views(fx.journal.read_all()), candidates={}, frontier=())
        with self.assertRaises(SM.BootstrapRefused):
            machine.bootstrap(anchor=_anchor(), views=emptied)
        self.assertEqual(machine.state, SM.BOOTSTRAP)

    def test_a_deliberate_rebase_states_its_reason_on_the_record(self):
        fx = self.fixture()
        fx.journal.append(J.KIND_CANDIDATE_RECORDED,
                          _candidate("akc-20260803-0001",
                                     _reconciliation("akc-20260803-0001")))
        machine = self.machine(fx.journal)
        emptied = dataclasses.replace(
            J.rebuild_views(fx.journal.read_all()), candidates={}, frontier=())
        report = machine.bootstrap(
            anchor=_anchor(), views=emptied, deliberate_rebase=True,
            rebase_reason="derived store rebuilt after the 08-03 schema migration")
        self.assertTrue(report.deliberate_rebase)
        self.assertEqual(machine.state, SM.DISCOVER)
        rebased = [e for e in fx.journal.read_all() if e.kind == J.KIND_VIEW_REBASED]
        self.assertEqual(len(rebased), 1)
        self.assertIn("schema migration", rebased[0].payload["rebase_reason"])


# =============================================================================
# NEGATIVE PATHS
# =============================================================================

class TestLatchedHaltSurvivesRestart(_LoopCase):
    """§4 invariant 19 — AutoPilot's pause was a silent no-op for months."""

    def test_a_latched_halt_stops_a_freshly_constructed_machine(self):
        fx = self.fixture()
        machine = self.machine(fx.journal)
        machine.bootstrap(anchor=_anchor(),
                          views=J.rebuild_views(fx.journal.read_all()))
        machine.submit_control("pause", control_id="ctl-0001",
                               requested_by="operator",
                               reason="host maintenance window")

        decision = machine.begin_iteration()
        self.assertFalse(decision.proceed)
        self.assertEqual(machine.state, SM.OPERATOR_STOP_REQUESTED)

        # A NEW process over the same root: nothing is carried in memory.
        restarted = self.machine(fx.journal)
        self.assertEqual(restarted.state, SM.OPERATOR_STOP_REQUESTED)
        self.assertTrue(restarted.restore_report.latch_present)
        again = restarted.begin_iteration()
        self.assertFalse(again.proceed)
        self.assertEqual(again.control, "pause")

        # It stays stopped until an OPERATOR resumes; the loop cannot.
        with self.assertRaises(SM.ControlLatchError):
            restarted.reopen(reason="carry on", authorized_by="controller")
        restarted.resume_control("ctl-0001", requested_by="operator",
                                 reason="maintenance finished")
        restarted.reopen(reason="resuming after maintenance",
                         authorized_by="operator")
        self.assertEqual(restarted.state, SM.BOOTSTRAP)

    def test_no_machine_holds_the_latch_as_state(self):
        fx = self.fixture()
        machine = self.machine(fx.journal)
        self.assertEqual(SM.audit_no_cached_control_state(machine).outcome, S.PASS)


class TestAnchorMovePreservesWorkAndKillsComparisons(_LoopCase):
    """§8.9 items 2-3 — *"only the comparisons died, not the work"*."""

    def test_anchor_move_supersedes_comparisons_and_preserves_source_and_t0(self):
        fx = self.fixture()
        journal_ = fx.journal
        old, new = _anchor(), _anchor(commit=V9_COMMIT, binary="v9-binary",
                                      linkage="v9-linkage")
        candidate_id = "akc-20260803-0001"
        reconciliation = _reconciliation(candidate_id)
        journal_.append(J.KIND_CANDIDATE_RECORDED, _candidate(candidate_id, reconciliation))
        t0 = journal_.append(J.KIND_EVALUATION_EVENT,
                             _event("ake-t0-0001", candidate_id=candidate_id,
                                    tier="T0", anchor=old))
        t1 = journal_.append(J.KIND_EVALUATION_EVENT,
                             _event("ake-t1-0001", candidate_id=candidate_id,
                                    tier="T1", anchor=old))

        response = CP.respond_to_anchor_move(
            recorded_anchor=old, observed_anchor=new,
            entries=journal_.read_all(),
            backends_by_candidate={candidate_id: [BACKEND]})

        superseded = set(response.sweep.superseded_entry_ids)
        preserved = set(response.sweep.preserved_entry_ids)
        self.assertIn(t1.event_id, superseded)
        self.assertIn(t0.event_id, preserved)
        # The candidate record itself — source, patch, build — is never touched.
        candidate_entry = next(e for e in journal_.read_all()
                               if e.kind == J.KIND_CANDIDATE_RECORDED)
        self.assertNotIn(candidate_entry.event_id, superseded)
        self.assertIn(candidate_id, response.sweep.preserved_candidate_ids)

        CP.apply_anchor_move_supersession(journal_, response.sweep)
        views = J.rebuild_views(journal_.read_all())
        self.assertNotIn("ake-t1-0001", views.evaluations)
        self.assertIn("ake-t0-0001", views.evaluations)
        self.assertIn(candidate_id, views.candidates)

        # And the machine's own ANCHOR_MOVED is the SAME transition: the response's
        # stop package is what `stop()` is handed, not a second opinion.
        machine = self.machine(journal_)
        machine.bootstrap(anchor=old, views=J.rebuild_views(journal_.read_all()))
        machine.anchor_store.record(old)
        # An anchor NOBODY LOOKED AT is not an anchor that did not move: a
        # partially observed identity is COULD_NOT_CHECK, and continuing past an
        # uncheckable denominator is the fail-open shape the check exists for.
        two_backend = _anchor(backends=(BACKEND, "llama_cpu"))
        machine.anchor_store.record(two_backend)
        with self.assertRaises(SM.AnchorUncheckable):
            machine.campaign_boundary(observed_anchor=_anchor(backends=(BACKEND,)))
        machine.anchor_store.record(old)
        check = machine.campaign_boundary(observed_anchor=new)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertEqual(machine.state, SM.ANCHOR_MOVED)
        self.assertEqual(SM.STOP_RECOVERY[SM.ANCHOR_MOVED], SM.RECOVERY_REANCHOR)

    def test_the_composition_stop_request_is_one_the_machine_accepts(self):
        old, new = _anchor(), _anchor(commit=V9_COMMIT, binary="v9-binary",
                                      linkage="v9-linkage")
        fx = self.fixture()
        response = CP.respond_to_anchor_move(
            recorded_anchor=old, observed_anchor=new, entries=fx.journal.read_all())
        request = response.to_stop_request()
        self.assertEqual(request.state, SM.ANCHOR_MOVED)
        self.assertEqual(
            SM.check_stop_evidence(request.state, request.reason, request.detail).outcome,
            S.PASS)

    def test_an_unverifiable_anchor_is_refused_not_assumed_good(self):
        old = _anchor()
        fx = self.fixture()
        with self.assertRaises(SM.AnchorUncheckable):
            CP.respond_to_anchor_move(recorded_anchor=old, observed_anchor=None,
                                      entries=fx.journal.read_all())


class TestPlannerDegradedIsNotPlateau(_LoopCase):
    """§8.10 — *"conflating them once cost this project months of paid no-ops"*."""

    def _skip(self, journal_, screener, shape):
        proposal = dict(PL.assemble_proposal(
            draft=_draft(), campaign_id=CAMPAIGN, proposal_id=f"akp-{shape}",
            parent_candidate_id=None, binding=_planner_binding(),
            prompt_bundle_sha256=_sha("bundle"), context_manifest_sha256=_sha("context"),
            do_not_repeat_matches=(), realized_cost=PL.RealizedCost(controller_tokens=10),
            created_at="2026-08-03T10:05:00Z"))
        proposal[SEL.SELECTION_BLOCK_KEY] = _selection_block()
        proposal["target"] = dict(proposal["target"], shapes=["8192x8192xq4_K"])
        return screener.screen(proposal, _selection_context())

    def test_a_repeated_skip_fingerprint_raises_planner_degraded(self):
        fx = self.fixture()
        journal_ = fx.journal
        screener = SEL.ProposalScreener(journal_, campaign_id=CAMPAIGN)
        for round_id in range(4):
            result = self._skip(journal_, screener, f"{round_id:04d}")
            self.assertFalse(result.admitted)

        history = SEL.read_skip_history(journal_, campaign_id=CAMPAIGN)
        # ONE fingerprint, because the concept never changed — which is the whole
        # point of a prose-free fingerprint, and was not true when the planner
        # adapter and this screener each had their own.
        self.assertEqual(len(history.counts), 1)
        self.assertEqual(len(history.blacklisted), 1)

        request = SEL.planner_health_stop_request(
            history, stop_policy=_campaign()["stop_policy"])
        self.assertIsNotNone(request)
        self.assertEqual(request.state, SM.PLANNER_DEGRADED)
        self.assertNotEqual(request.state, SM.PLATEAU_STOP)
        self.assertEqual(
            SM.check_stop_evidence(request.state, request.reason, request.detail).outcome,
            S.PASS)

        machine = self.machine(journal_)
        machine.bootstrap(anchor=_anchor(),
                          views=J.rebuild_views(journal_.read_all()))
        machine.dispose_stop_request(request)
        self.assertEqual(machine.state, SM.PLANNER_DEGRADED)
        # A broken searcher goes to an operator, a finished search starts a new
        # campaign. The recovery classes are what keep the two apart.
        self.assertEqual(SM.STOP_RECOVERY[SM.PLANNER_DEGRADED],
                         SM.RECOVERY_OPERATOR_REVIEW)
        self.assertEqual(SM.STOP_RECOVERY[SM.PLATEAU_STOP], SM.RECOVERY_NEW_CAMPAIGN)

    def test_a_plateau_that_never_ruled_out_a_broken_searcher_is_refused(self):
        fx = self.fixture()
        machine = self.machine(fx.journal)
        machine.bootstrap(anchor=_anchor(),
                          views=J.rebuild_views(fx.journal.read_all()))
        with self.assertRaises(SM.StopEvidenceMissing) as ctx:
            machine.stop(SM.PLATEAU_STOP,
                         reason="marginal yield sat under the floor for five rounds",
                         detail={
                             "closed": [{"sub_scope": "gpu/decode",
                                         "gates_met": ["T0", "T1"]}],
                             "deferred": [],
                         })
        self.assertIn("planner_health", str(ctx.exception))


class TestExhaustedSurfaceNeedsItsEnumeration(_LoopCase):

    def test_exhausted_surface_is_refused_without_the_enumeration(self):
        fx = self.fixture()
        machine = self.machine(fx.journal)
        machine.bootstrap(anchor=_anchor(),
                          views=J.rebuild_views(fx.journal.read_all()))
        for detail in ({}, {"closed": []}, {"closed": [{"sub_scope": "gpu/decode",
                                                        "gates_met": ["T0"]}]}):
            with self.subTest(detail=detail):
                with self.assertRaises(SM.StopEvidenceMissing):
                    machine.stop(SM.EXHAUSTED_SURFACE,
                                 reason="no eligible layer remains at this target",
                                 detail=detail)
        self.assertEqual(machine.state, SM.DISCOVER)

    def test_the_reserved_words_are_refused_by_the_disposer_not_only_by_a_guard(self):
        """The seam: `stop()` is public, so a stop that never met a guard must
        still meet the vocabulary."""
        fx = self.fixture()
        machine = self.machine(fx.journal)
        machine.bootstrap(anchor=_anchor(),
                          views=J.rebuild_views(fx.journal.read_all()))
        detail = {"closed": [{"sub_scope": "gpu/decode", "gates_met": ["T0", "T1"]}],
                  "deferred": []}
        for reason in ("the surface is exhausted", "we tried all paths",
                       "nothing left to try"):
            with self.subTest(reason=reason):
                with self.assertRaises(SM.StopEvidenceMissing):
                    machine.stop(SM.EXHAUSTED_SURFACE, reason=reason, detail=detail)

    def test_a_reserved_word_hiding_in_the_enumeration_is_refused(self):
        detail = {"closed": [{"sub_scope": "all paths through iqk",
                              "gates_met": ["T0", "T1"]}],
                  "deferred": []}
        check = SM.check_stop_evidence(
            SM.EXHAUSTED_SURFACE, "closed for the iqk sub-scope", detail)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("closed[0].sub_scope" in r for r in check.reasons))


# =============================================================================
# SEAM REGRESSIONS — one per disagreement the integration pass found
# =============================================================================

class TestSeamOracleRegistry(unittest.TestCase):
    """Seam 2/3: the compiler renders ids the critic must be able to gate on."""

    def test_every_id_the_planner_can_read_resolves_for_the_critic(self):
        for row in C.ORACLE_REGISTRY:
            with self.subTest(oracle=row.oracle_id):
                gated = CR.oracle_row(row.oracle_id)
                self.assertIsNotNone(
                    gated, f"{row.oracle_id!r} is rendered into the planner brief but "
                           "the critic cannot resolve it")
                self.assertEqual(gated.harvest_class, row.harvest_class)
                self.assertEqual(gated.retired, row.status == C.ORACLE_RETIRED)

    def test_every_id_the_critic_gates_on_is_a_declared_oracle(self):
        for row in CR.ORACLE_REGISTRY:
            with self.subTest(oracle=row.oracle_id):
                self.assertIsNotNone(O.resolve(row.oracle_id))

    def test_both_consumers_pass_the_registry_audit(self):
        self.assertEqual(
            O.audit_consumer_registry(
                C.ORACLE_REGISTRY, id_of=lambda r: r.oracle_id,
                harvest_class_of_row=lambda r: r.harvest_class,
                retired_of=lambda r: r.status == C.ORACLE_RETIRED,
                what="context").outcome,
            S.PASS)
        self.assertEqual(
            O.audit_consumer_registry(
                CR.ORACLE_REGISTRY, id_of=lambda r: r.oracle_id,
                harvest_class_of_row=lambda r: r.harvest_class,
                retired_of=lambda r: r.retired, what="critic").outcome,
            S.PASS)

    def test_the_harvest_class_vocabularies_are_one_vocabulary(self):
        self.assertEqual(C.HARVEST_CLASSES, CR.HARVEST_CLASSES)
        self.assertEqual(C.HARVEST_CLASSES, O.HARVEST_CLASSES)
        self.assertIn(O.HARVEST_CONDITIONAL, CR.HARVEST_CLASSES)

    def test_the_retired_row_reaches_both_planes(self):
        """§6.5 keeps a wrong row VISIBLE. A plane that dropped it would let the
        same wrong entry be re-derived at zero cost."""
        self.assertTrue(any(r.status == C.ORACLE_RETIRED for r in C.ORACLE_REGISTRY))
        self.assertTrue(CR.oracle_row("AITER").retired)
        self.assertTrue(CR.oracle_row("AMD AITER").retired)


class TestSeamFingerprint(unittest.TestCase):
    """Seam 1: two fingerprints, one journal field, one blacklist count."""

    def _manifest(self):
        manifest = dict(PL.assemble_proposal(
            draft=_draft(), campaign_id=CAMPAIGN, proposal_id="akp-20260803-0001",
            parent_candidate_id=None, binding=_planner_binding(),
            prompt_bundle_sha256=_sha("bundle"), context_manifest_sha256=_sha("context"),
            do_not_repeat_matches=(), realized_cost=PL.RealizedCost(controller_tokens=10),
            created_at="2026-08-03T10:05:00Z"))
        manifest[SEL.SELECTION_BLOCK_KEY] = _selection_block()
        return manifest

    def test_the_two_modules_compute_one_digest(self):
        manifest = self._manifest()
        self.assertEqual(PL.proposal_fingerprint(manifest),
                         SEL.proposal_fingerprint(manifest))
        self.assertEqual(PL.proposal_fingerprint(manifest),
                         FP.proposal_fingerprint(manifest))

    def test_rewording_prose_does_not_mint_a_new_concept(self):
        base = self._manifest()
        reworded = dict(base)
        reworded["hypothesis"] = "entirely different words for the same change"
        reworded["narrative"] = "different prose"
        reworded["change"] = dict(base["change"],
                                  conceptual_change="reworded conceptual change")
        self.assertEqual(SEL.proposal_fingerprint(base),
                         SEL.proposal_fingerprint(reworded))
        self.assertEqual(PL.proposal_fingerprint(base),
                         PL.proposal_fingerprint(reworded))

    def test_a_structural_change_does_mint_a_new_concept(self):
        base = self._manifest()
        moved = dict(base)
        moved["target"] = dict(base["target"], ops=["mul_mat"])
        self.assertNotEqual(SEL.proposal_fingerprint(base),
                            SEL.proposal_fingerprint(moved))


class TestSeamHypothesisOrigins(unittest.TestCase):
    """Seam 4: the store's origins and the compiler's were different sets."""

    def test_every_origin_the_store_can_open_is_renderable(self):
        for origin in sorted(H.ORIGINS):
            with self.subTest(origin=origin):
                self.assertIn(origin, C.HYPOTHESIS_ORIGINS)
                item = C.OpenHypothesis(
                    hypothesis_id=f"akh-{origin}",
                    statement="a statement long enough to be a statement",
                    falsifier="a measurement that would refute it",
                    origin=origin, evidence_grade=H.ENTRY_GRADE,
                    event_id="ake-0001", locator=_loc("h"))
                self.assertEqual(item.origin, origin)

    def test_the_compiler_offers_no_origin_the_store_cannot_produce(self):
        self.assertEqual(set(C.HYPOTHESIS_ORIGINS), set(H.ORIGINS))

    def test_the_entry_grade_is_the_same_grade_on_both_sides(self):
        self.assertIn(H.ENTRY_GRADE, C.EVIDENCE_GRADES)
        self.assertEqual(set(H.EVIDENCE_GRADES), set(C.EVIDENCE_GRADES))
        self.assertEqual(set(H.EVIDENCE_GRADES), set(PL.EVIDENCE_GRADES))


class TestSeamStopVocabulary(unittest.TestCase):
    """Seam 5, plus the vocabulary agreement the guard plane rests on."""

    def test_guards_covers_every_stop_the_machine_declares(self):
        covered = set(G.GUARD_BY_STOP) | set(G.NON_GUARD_STOPS)
        self.assertEqual(covered, set(SM.STOP_STATES))
        self.assertEqual(G.audit_stop_coverage_totality().outcome, S.PASS)

    def test_the_precedence_order_names_only_declared_stops(self):
        self.assertEqual(set(G.STOP_PRECEDENCE) - set(SM.STOP_STATES), set())

    def test_the_reserved_words_guards_names_are_the_machines_words(self):
        for word in G.RESERVED_CLOSURE_WORDS:
            with self.subTest(word=word):
                self.assertIn(word, SM.RESERVED_CLOSURE_PHRASES)
                self.assertEqual(SM.reserved_closure_findings(f"we were {word}")[0],
                                 word)

    def test_a_state_name_in_a_reason_is_not_a_closure_claim(self):
        """`EXHAUSTED_SURFACE` is an identifier; "exhausted" is the claim."""
        self.assertEqual(SM.reserved_closure_findings("reached EXHAUSTED_SURFACE"), ())
        self.assertEqual(SM.reserved_closure_findings("the surface is exhausted"),
                         ("exhausted",))

    def test_every_guard_stop_carries_evidence_the_machine_accepts(self):
        self.assertEqual(set(G.ESCALATING_STOPS) - set(SM.STOP_STATES), set())


class TestSeamBudgetUnits(unittest.TestCase):
    """Seam 6: hours in the manifest, minutes in a proposal, seconds in the ledger."""

    def test_the_converter_crosses_all_three_units(self):
        remaining = SEL.budget_remaining_from_caps(
            _campaign()["budgets"],
            wall_hours_used=1.0,
            gpu_seconds_used=3600.0,
            cpu_region_seconds_used=1800.0,
            storage_gb_used=10.0,
            candidates_used=5)
        self.assertEqual(sorted(remaining), sorted(SEL.BUDGET_KEYS))
        self.assertAlmostEqual(remaining["wall_minutes"], 40 * 60 - 60)
        self.assertAlmostEqual(remaining["gpu_minutes"], 10 * 60 - 60)
        self.assertAlmostEqual(remaining["cpu_region_minutes"], 10 * 60 - 30)
        self.assertAlmostEqual(remaining["storage_gb"], 90.0)
        self.assertAlmostEqual(remaining["candidates"], 45.0)

    def test_the_result_is_a_valid_selection_context_budget(self):
        context = _selection_context(budget_remaining=SEL.budget_remaining_from_caps(
            _campaign()["budgets"], wall_hours_used=0.0, gpu_seconds_used=0.0,
            cpu_region_seconds_used=0.0, storage_gb_used=0.0, candidates_used=0))
        self.assertEqual(sorted(context.budget_remaining), sorted(SEL.BUDGET_KEYS))

    def test_a_missing_cap_raises_rather_than_defaulting(self):
        caps = dict(_campaign()["budgets"])
        caps.pop("max_gpu_hours")
        with self.assertRaises(ValueError) as ctx:
            SEL.budget_remaining_from_caps(
                caps, wall_hours_used=0.0, gpu_seconds_used=0.0,
                cpu_region_seconds_used=0.0, storage_gb_used=0.0, candidates_used=0)
        self.assertIn("max_gpu_hours", str(ctx.exception))

    def test_overspend_floors_at_zero_rather_than_going_negative(self):
        remaining = SEL.budget_remaining_from_caps(
            _campaign()["budgets"], wall_hours_used=1000.0,
            gpu_seconds_used=10 ** 9, cpu_region_seconds_used=10 ** 9,
            storage_gb_used=10 ** 6, candidates_used=10 ** 6)
        self.assertEqual(set(remaining.values()), {0.0})

    def test_the_guard_plane_owns_the_dimension_this_screen_cannot_see(self):
        """A proposal declares no token cost, so the screen cannot gate one."""
        for name in SEL.BUDGET_KEYS_NOT_SCREENED:
            with self.subTest(name=name):
                self.assertIn(name, G.BUDGET_DIMENSIONS)
                self.assertNotIn(name, SEL.BUDGET_KEYS)


class TestSeamSuppressionClasses(unittest.TestCase):
    """A DELIBERATE asymmetry, pinned so it is not "fixed" in the wrong direction."""

    def test_the_six_classes_are_one_vocabulary_in_four_modules(self):
        self.assertEqual(tuple(SEL.LEDGER_CLASSES), tuple(CR.LEDGER_CLASSES))
        self.assertEqual(tuple(SEL.LEDGER_CLASSES), tuple(C.SUPPRESSION_CLASSES))
        self.assertEqual(set(SEL.LEDGER_CLASSES), set(H.MATCH_CLASSES))

    def test_the_three_family_closing_classes_agree(self):
        self.assertEqual(set(SEL.RECEIPT_REQUIRED_CLASSES),
                         set(C.RECEIPT_REQUIRING_SUPPRESSION_CLASSES))
        self.assertEqual(set(SEL.REJECTING_LEDGER_CLASSES),
                         set(CR.SUPPRESSING_LEDGER_CLASSES))

    def test_superseded_fact_rejects_a_proposal_but_not_a_question(self):
        """§19.2: `SUPERSEDED_FACT` = *"do not execute the stale PROPOSAL;
        regenerate from current source"*. It closes the proposal, not the
        question — so `selection` rejects and `hypotheses` does not, and making
        the two equal would close research the design keeps open. Asserted rather
        than left to look like an oversight.
        """
        self.assertIn("SUPERSEDED_FACT", SEL.REJECTING_LEDGER_CLASSES)
        self.assertNotIn("SUPERSEDED_FACT", H.REJECTING_MATCH_CLASSES)
        self.assertEqual(set(H.REJECTING_MATCH_CLASSES),
                         {"HARD_CONSTRAINT", "MATCHED_NEGATIVE"})


class TestSeamPackageSurface(unittest.TestCase):

    def test_the_package_exports_every_module_of_its_own_plane(self):
        from autokernel import controller
        for name in ("composition", "context", "critic", "fingerprint", "guards",
                     "hypotheses", "oracles", "planner", "selection", "state_machine"):
            with self.subTest(module=name):
                self.assertIn(name, controller.__all__)
                self.assertTrue(hasattr(controller, name))

    def test_the_compiled_context_binds_directly_to_the_planner_adapter(self):
        """The compiler's bundle IS a planner context: no round-trip, because a
        round-trip would change the bytes the manifest hash was taken over."""
        with tempfile.TemporaryDirectory() as tmp:
            fx = _ContextFixture(tmp)
            bundle = C.compile_context(fx.inputs())
            self.assertEqual(
                PL.resolve_context_binding(bundle, CAMPAIGN), bundle.manifest_sha256)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
