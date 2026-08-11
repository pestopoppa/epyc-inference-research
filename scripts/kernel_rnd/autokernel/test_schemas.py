"""Unit tests for autokernel/schemas.py — the AutoKernel versioned data contracts.

Pure data. NO inference, NO server, NO model, NO GPU, NO filesystem, NO sqlite —
every case here is a dict in memory, which is why this suite is safe to run on
the shared host at any time.

The suite is organised around the failures the contracts exist to prevent
(handoff `autokernel-research-loop.md`):

  * a record missing any required field is REJECTED, per schema, field by field —
    an unlabelled measurement is not decision-grade (MEASUREMENT.md:85-95);
  * a campaign carrying an authority-flavoured key is REJECTED, and the
    legitimate lookalikes (`release_protocol`, `draft_autopilot_rebaseline_note`)
    are NOT (§1.3 — there is no freeze authority to carry);
  * canonical serialisation is stable under key reordering and REFUSES the
    encodings that would silently produce a wrong content hash;
  * `SCHEMA_REGISTRY` dispatch resolves each schema string to the validator that
    accepts its own record and rejects every other.

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/test_schemas.py
    python3 scripts/kernel_rnd/autokernel/test_schemas.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schemas as S  # noqa: E402


V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"


def _sha(tag: str) -> str:
    """A syntactically valid, distinct sha256 for fixtures."""
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _representation_contract(tag: str = "decode-kernels") -> dict:
    contract = {
        "vocabulary": {
            "regimes": ["decode", "prefill"],
            "surfaces": ["mul_mat"],
            "outcomes": ["throughput_gain", "non_inferiority"],
            "contradictions": ["counter_did_not_move", "heldout_regression"],
        },
        "vocabulary_source_receipts": [f"rcpt-vocabulary-{tag}"],
        "considered_alternatives": ["wide_tile", "dispatcher_only", "rewrite_backend"],
        "excluded_alternatives": [{
            "alternative_id": "rewrite_backend",
            "reason": "not a one-factor intervention",
            "source_receipt_id": f"rcpt-alternative-{tag}",
        }],
        "empirical_demand": {
            "receipt_id": f"rcpt-demand-{tag}",
            "weights_sha256": _sha(f"demand-{tag}"),
        },
        "abstraction_construction_cost": {
            "value": 3.0,
            "unit": "typed_facts",
            "receipt_id": f"rcpt-cost-{tag}",
        },
        "canonical_encoding": {
            "encoding_id": "ak-representation-json/v1",
            "schema_sha256": _sha("representation-schema-v1"),
        },
        "semantics_preserving_recoding_fixture_ids": [
            f"ak-recode-{tag}-renamed", f"ak-recode-{tag}-permuted",
        ],
    }
    contract["frame_sha256"] = S.representation_frame_sha256(contract)
    return contract


# =============================================================================
# Minimal VALID fixtures — one per schema, containing exactly the required keys
# so the omission sweep below can delete each in turn.
# =============================================================================

def _campaign() -> dict:
    return {
        "schema": S.SCHEMA_CAMPAIGN,
        "campaign_id": "ak-llama_gpu-decode-20260803",
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
            "protocol_by_phase": {
                "prefill": "P-BENCH-PREFILL-1",
                "decode": "P-BENCH-1",
            },
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
        "readiness_reporting": {
            "reference_point_gain": 0.25,
            "reference_lcb_gain": 0.20,
        },
        "stop_policy": {
            "plateau_rounds": 6,
            "max_consecutive_integrity_failures": 2,
            "max_consecutive_build_failures": 3,
            "max_command_retries": 3,
        },
    }


def _proposal() -> dict:
    return {
        "schema": S.SCHEMA_PROPOSAL,
        "proposal_id": "akp-20260803-0001",
        "campaign_id": "ak-llama_gpu-decode-20260803",
        "parent_candidate_id": None,
        "controller": {
            "provider": "local",
            "model_id": "architect-a4",
            "effort": "high",
            "prompt_bundle_sha256": _sha("prompt-bundle"),
            "sampling_params": {"temperature": 0.0, "seed": 42},
            "context_manifest_sha256": _sha("context-manifest"),
        },
        "realized_cost": {
            "controller_tokens": 18_500,
            "build_seconds": 412.0,
            "evaluator_wall_seconds": 900.0,
            "gpu_seconds": 240.0,
            "cpu_region_seconds": 0.0,
            "storage_gb": 1.5,
        },
        "hypothesis": "Selecting the wide-tile path for K>=4096 removes a launch stall.",
        "narrative": "planner prose that must never be retrieved as fact",
        "narrative_retrievable": False,
        "change_class": "dispatcher",
        "declared_symbol_deltas": {"added": [], "removed": [], "arity_changed": []},
        "campaign_kind": "source_change",
        "oracle_reference": {"oracle": None, "commit": None, "license_check": None},
        "novelty_basis": {
            "prior_event_ids": [],
            "source_receipts": [],
            "do_not_repeat_matches": [],
        },
        "expected_information_gain": 0.4,
        "representation_contract": _representation_contract(),
        "external_numbers": [],
        "target": {"regimes": ["decode"], "ops": ["mul_mat"], "shapes": [], "models": []},
        "non_target": {"regimes": ["prefill"], "shapes": []},
        "mechanism_prediction": {
            "bottleneck_before": "memory_latency",
            "expected_counter_changes": {"L2CacheHit": "increase"},
            "expected_wall_share_ceiling": 0.35,
            "wall_share_receipt_id": "rcpt-wall-share-0007",
        },
        "change": {
            "predicted_affected_surface": ["mul_mat"],
            "files_and_symbols": ["ggml-cuda/mmq.cu:mul_mat_q"],
            "conceptual_change": "widen the tile selection predicate",
            "parameter_surface": {},
            "estimated_diff_size": 40,
        },
        "risks": {
            "correctness": [],
            "numerical": [],
            "state_or_rollback": [],
            "resource": [],
            "integrity": [],
        },
        "fallback": {
            "dispatch_guard": "GGML_AK_WIDE_TILE=0",
            "kill_switch": "env GGML_AK_WIDE_TILE=0",
        },
        "evaluation_plan": {
            "required_t0": ["symbol_preservation", "clean_snapshot_build"],
            "required_t1": ["t1a_target_operator_discriminator"],
            "conditional_t2": [],
            "profiler_questions": [],
        },
        "resource_request": {
            "lane": "gpu",
            "expected_minutes": 25,
            "expected_storage_gb": 2.0,
        },
        "stop_condition": "reject if the discriminator shows no path change",
        "critic_verdict": {"status": "pending", "reasons": []},
    }


def _proposal_v2() -> dict:
    record = _proposal()
    record["schema"] = S.SCHEMA_PROPOSAL_V2
    del record["representation_contract"]
    del record["external_numbers"]
    return record


def _candidate() -> dict:
    return {
        "schema": S.SCHEMA_CANDIDATE,
        "candidate_id": "akc-20260803-0001",
        "campaign_id": "ak-llama_gpu-decode-20260803",
        "proposal_id": "akp-20260803-0001",
        "parent_candidate_id": None,
        "worktree": {
            "path": "/mnt/raid0/llm/llama.cpp-ak-llama_gpu-decode-20260803",
            "branch": "ak/ak-llama_gpu-decode-20260803/akp-0001",
            "source_commit": V7_COMMIT,
            "clean": True,
        },
        "source_snapshot": {
            "snapshot_sha256": _sha("snapshot"),
            "patch_bundle_sha256": _sha("patch-bundle"),
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
            "build_dir": "/mnt/raid0/llm/tmp/ak-build/akc-20260803-0001",
            "log_path": "data/ak-llama_gpu-decode-20260803/build/akc-0001.log",
            "log_sha256": _sha("build-log"),
        },
        "artifacts": {
            "binary_sha256": _sha("candidate-binary"),
            "linkage_sha256": _sha("candidate-linkage"),
            "library_sha256s": {"libggml.so": _sha("libggml")},
        },
        "dispatch": {
            "feature_flags": ["GGML_AK_WIDE_TILE"],
            "dispatch_predicate": "K >= 4096",
        },
        "affected_surface": {
            "derived_sha256": _sha("derived-surface"),
            "traced_sha256": None,
            "reconciled": False,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "evaluator": {
            "id": "P-AK-SEARCH-1/v1",
            "bundle_sha256": _sha("evaluator-bundle"),
        },
        "receipts": {
            "host_receipt": "rcpt-host-20260803T101500Z",
            "resource_claim_receipt": "rcpt-gpu-claim-0042",
        },
        "storage": {"footprint_gb": 3.4, "durability_class": "hash_and_provenance_only"},
        "evaluation_event_ids": [],
        "derived_verdicts": {},
        "controller": {
            "provider": "local",
            "model_id": "architect-a4",
            "effort": "high",
            "prompt_bundle_sha256": _sha("prompt-bundle"),
        },
        "champion_status": "none",
        "status": "built",
        "supersession_reason": None,
        "created_at": "2026-08-03T10:15:00+00:00",
    }


def _event() -> dict:
    """A v2 evaluation event — a record of the generation already in journals.

    Kept as v2 ON PURPOSE. v2 is not retired and is not rewritten, so the corpus
    has to contain one; if this fixture were quietly moved to v3 nothing would be
    left proving that yesterday's shard still reads.
    """
    return {
        "schema": S.SCHEMA_EVALUATION_EVENT_V2,
        "event_id": "ake-20260803-0001",
        "campaign_id": "ak-llama_gpu-decode-20260803",
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
        "evaluator": {
            "id": "P-AK-SEARCH-1/v1",
            "bundle_sha256": _sha("evaluator-bundle"),
        },
        "artifact": {
            "source_sha256": _sha("snapshot"),
            "binary_sha256": _sha("candidate-binary"),
            "linkage_sha256": _sha("candidate-linkage"),
        },
        "anchor": {
            "binary_sha256": _sha("anchor-binary"),
            "linkage_sha256": _sha("anchor-linkage"),
            "measurement_event_ids": ["ake-20260801-0009"],
        },
        "scope_manifest_sha256": _sha("scope-manifest"),
        "host_receipt": "rcpt-host-20260803T101500Z",
        "resource_claim_receipt": "rcpt-gpu-claim-0042",
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


def _event_v3(**overrides) -> dict:
    """A v3 evaluation event with a fully named anchor (all THREE components)."""
    record = _event()
    record["schema"] = S.SCHEMA_EVALUATION_EVENT_V3
    record["anchor"] = {
        "source_commit": V8_COMMIT,
        "binary_sha256": _sha("anchor-binary"),
        "linkage_sha256": _sha("anchor-linkage"),
        "measurement_event_ids": ["ake-20260801-0009"],
    }
    record.update(overrides)
    return record


def _event_v3_voided_without_an_anchor() -> dict:
    """The record v3 exists for: a run VOIDED because no anchor was resolvable.

    The protocol requires it to be journaled ("A voided run is journaled as
    `INVALID` with its reason, and is **never silently discarded**") and v2 could
    not express it — `anchor.binary_sha256` was unconditionally required, so the
    only way to produce a valid record was to invent a digest. Here the block is
    STRUCTURALLY ABSENT and the reason is in `integrity_flags`.

    It is the FIXTURES entry for v3 deliberately: it is the shape the version was
    introduced for, and it is also not a valid v2 record, which is what keeps
    `test_dispatch_rejects_a_record_under_the_wrong_schema_string` honest.
    """
    record = _event()
    record["schema"] = S.SCHEMA_EVALUATION_EVENT_V3
    del record["anchor"]
    record["status"] = "invalid"
    record["integrity_flags"] = ["VOID:ANCHOR_MISSING_OR_MUTATED:FAIL"]
    record["performance"] = {
        "raw_samples": [],
        "paired_blocks": 0,
        "estimate": None,
        "uncertainty": None,
    }
    return record


def _event_v4(**overrides) -> dict:
    record = _event_v3()
    record["schema"] = S.SCHEMA_EVALUATION_EVENT_V4
    record["change_class"] = "dispatcher"
    record["anchor_tier"] = "T2"
    record["transfer_ratio_to"] = [{
        "event_id": "ake-20260801-0099",
        "tier": "T2",
        "source_effect": 0.02,
        "target_effect": 0.025,
        "ratio": 0.8,
    }]
    record.update(overrides)
    return record


def _event_v5(**overrides) -> dict:
    record = _event_v4()
    record["schema"] = S.SCHEMA_EVALUATION_EVENT_V5
    record["backend"] = "llama_cpu"
    record["device_state"] = None
    record.update(overrides)
    return record


def _champion() -> dict:
    return {
        "schema": S.SCHEMA_CHAMPION,
        "source_tree": "llama.cpp",
        "anchor_commit": V8_COMMIT,
        "branch": "ak/champion/llama-20260802",
        "member_candidates": ["akc-20260803-0001"],
        "combined_candidate_id": "akc-20260803-0009",
        "last_t0": {"event_id": "ake-20260803-0002", "status": "pass"},
        "last_t1": {"event_id": "ake-20260803-0001", "status": "pass"},
        "last_t2": None,
        "readiness": {
            "by_backend": {"llama_gpu": {"prefill": {}, "decode": {}}},
            "reference_signal": "point +2.1% / LCB +0.8% versus anchor on 6 cells",
        },
        "affected_surface_union_sha256": _sha("surface-union"),
        "storage_gb": 12.0,
        "blocking_conditions": [],
    }


def _release_package() -> dict:
    return {
        "schema": S.SCHEMA_RELEASE_PACKAGE,
        "package_id": "akr-20260803-0001",
        "campaign_id": "ak-llama_gpu-decode-20260803",
        "source_tree": "llama.cpp",
        "sealed_candidate": {
            "candidate_id": "akc-20260803-0009",
            "seal_sha256": _sha("seal"),
            "binary_sha256": _sha("sealed-binary"),
            "linkage_sha256": _sha("sealed-linkage"),
            "build_receipt_sha256": _sha("build-receipt"),
        },
        "t3_verdict": {
            "verdict": "PASS",
            "bundle_sha256": _sha("t3-bundle"),
            "phase_results": {"identity_preflight": "PASS", "build_linkage": "PASS"},
        },
        "active_waivers": [],
        "release_plan": {"next_version": "production-consolidated-v9"},
        "transaction_plan": {"steps": ["archive incumbent", "create branch"]},
        "rollback_plan": {
            "incumbent_archive_path": "/mnt/raid0/llm/kernels/archive/v8",
            "incumbent_binary_sha256": _sha("incumbent-binary"),
        },
        "draft_era_registry_row": {"era_id": "E9-gpu-kernel", "status": "draft"},
        "draft_autopilot_rebaseline_note": "E9 rebaseline hold; operator ratification required",
        "linkage_verification": {"status": S.PASS, "receipt": "rcpt-linkage-0003"},
        "operator_command_sequence": [
            {
                "command": "scripts/freeze/freeze_v9.sh --dry-run",
                "validated": True,
                "validation_receipt": "rcpt-cmd-0001",
            }
        ],
        "change_classes": ["dispatcher"],
        "requires_human_code_review": False,
        "diff_complexity": {
            "diff_size": 40,
            "files_touched": 2,
            "touches_shared_core": False,
        },
        "created_at": "2026-08-03T12:00:00+00:00",
    }


def _waiver() -> dict:
    return {
        "schema": S.SCHEMA_OPERATOR_WAIVER,
        "waiver_id": "WAIVE-Q8-20260803",
        "campaign_id": "ak-llama_gpu-decode-20260803",
        "decision": "release without a Q8 non-regression claim",
        "protocol": "P-KERNEL-FREEZE-1/v1",
        "protocol_changed": False,
        "candidate_head": V7_COMMIT,
        "production_head": V8_COMMIT,
        "scope": {
            "excluded_models": ["qwen3-q8_0"],
            "excluded_pairs": [["qwen3-q8_0", "decode"]],
            "remaining_matched_pairs": 11,
        },
        "reason": "Q8 kernel path is out of campaign scope",
        "consequences": ["v9 makes no Q8 non-regression claim"],
        "authorized_by": "operator",
        "expiry": {"expires_at": None, "reopen_predicate": "Q8 iqk coverage lands"},
        "created_at": "2026-08-03T11:30:00+00:00",
    }


FIXTURES = {
    S.SCHEMA_CAMPAIGN: _campaign,
    S.SCHEMA_PROPOSAL_V2: _proposal_v2,
    S.SCHEMA_PROPOSAL: _proposal,
    S.SCHEMA_CANDIDATE: _candidate,
    S.SCHEMA_EVALUATION_EVENT_V2: _event,
    S.SCHEMA_EVALUATION_EVENT_V3: _event_v3_voided_without_an_anchor,
    S.SCHEMA_EVALUATION_EVENT_V4: _event_v4,
    S.SCHEMA_EVALUATION_EVENT_V5: _event_v5,
    S.SCHEMA_CHAMPION: _champion,
    S.SCHEMA_RELEASE_PACKAGE: _release_package,
    S.SCHEMA_OPERATOR_WAIVER: _waiver,
}


# =============================================================================
# Minimal valid objects + exhaustive required-field omission
# =============================================================================

class MinimalValidTest(unittest.TestCase):
    def test_every_fixture_validates(self):
        for schema, build in FIXTURES.items():
            with self.subTest(schema=schema):
                self.assertEqual(S.SCHEMA_REGISTRY[schema](build()), [])

    def test_every_fixture_declares_its_own_schema_string(self):
        for schema, build in FIXTURES.items():
            with self.subTest(schema=schema):
                self.assertEqual(build()["schema"], schema)

    def test_fixture_covers_every_registered_schema(self):
        self.assertEqual(set(FIXTURES), set(S.KNOWN_SCHEMAS))


class RequiredFieldOmissionTest(unittest.TestCase):
    """Deleting ANY top-level field of a valid record must produce a violation."""

    def test_top_level_omissions_are_rejected(self):
        for schema, build in FIXTURES.items():
            validator = S.SCHEMA_REGISTRY[schema]
            for key in list(build()):
                with self.subTest(schema=schema, omitted=key):
                    record = build()
                    del record[key]
                    violations = validator(record)
                    self.assertTrue(violations,
                                    f"omitting {key!r} from {schema} was accepted")
                    self.assertTrue(any(key in v for v in violations),
                                    f"violations for missing {key!r} do not name it: "
                                    f"{violations}")

    def _assert_nested_omissions_rejected(self, build, validator, block):
        for key in list(build()[block]):
            with self.subTest(block=block, omitted=key):
                record = build()
                del record[block][key]
                violations = validator(record)
                self.assertTrue(violations, f"omitting {block}.{key} was accepted")
                self.assertTrue(any(key in v for v in violations), violations)

    def test_claim_grammar_subfield_omissions_are_rejected(self):
        # MEASUREMENT.md:13 and :85-95 — a claim missing any of its parts is not
        # decision-grade, so no sub-field is optional.
        self._assert_nested_omissions_rejected(
            _event, S.validate_evaluation_event, "claim_grammar")

    def test_anchor_subfield_omissions_are_rejected(self):
        self._assert_nested_omissions_rejected(
            _event, S.validate_evaluation_event, "anchor")

    def test_scope_denominator_subfield_omissions_are_rejected(self):
        self._assert_nested_omissions_rejected(
            _event, S.validate_evaluation_event, "scope_denominator")

    def test_controller_subfield_omissions_are_rejected(self):
        self._assert_nested_omissions_rejected(
            _proposal, S.validate_proposal, "controller")

    def test_declared_symbol_delta_omissions_are_rejected(self):
        self._assert_nested_omissions_rejected(
            _proposal, S.validate_proposal, "declared_symbol_deltas")

    def test_realized_cost_omissions_are_rejected(self):
        self._assert_nested_omissions_rejected(
            _proposal, S.validate_proposal, "realized_cost")

    def test_representation_contract_omissions_are_rejected(self):
        self._assert_nested_omissions_rejected(
            _proposal, S.validate_proposal, "representation_contract")

    def test_budget_omissions_are_rejected(self):
        self._assert_nested_omissions_rejected(
            _campaign, S.validate_campaign, "budgets")

    def test_artifact_and_ancestry_omissions_are_rejected(self):
        for block in ("artifacts", "ancestry", "source_snapshot", "receipts"):
            self._assert_nested_omissions_rejected(
                _candidate, S.validate_candidate, block)

    def test_wrong_schema_string_is_rejected(self):
        record = _campaign()
        record["schema"] = S.SCHEMA_PROPOSAL
        self.assertTrue(S.validate_campaign(record))

    def test_missing_schema_string_is_rejected(self):
        for build in FIXTURES.values():
            record = build()
            del record["schema"]
            with self.subTest(schema=record.get("schema")):
                self.assertTrue(S.validate_record(record))

    def test_validators_never_raise_on_garbage(self):
        for schema, validator in S.SCHEMA_REGISTRY.items():
            for garbage in (None, [], "x", 7, {"schema": schema}):
                with self.subTest(schema=schema, garbage=garbage):
                    self.assertTrue(validator(garbage))


# =============================================================================
# §1.3 — no campaign carries freeze or cutover authority
# =============================================================================

class AuthorityFlagTest(unittest.TestCase):
    AUTH_KEYS = [
        "auto_freeze", "auto_cutover", "autoFreeze", "autofreeze",
        "may_cutover", "can_promote", "freeze_authority", "cutover_authority",
        "unattended_freeze", "self_freeze", "approved_cutover",
        "autonomous_promotion", "freeze", "cutover", "auto_release",
    ]

    def test_authority_keys_are_rejected_at_top_level(self):
        for key in self.AUTH_KEYS:
            with self.subTest(key=key):
                record = _campaign()
                record[key] = True
                violations = S.validate_campaign(record)
                self.assertTrue(any(key in v and "authority" in v for v in violations),
                                f"{key!r} was not rejected: {violations}")

    def test_authority_keys_are_rejected_when_nested(self):
        record = _campaign()
        record["stop_policy"]["auto_freeze_on_ready"] = True
        violations = S.validate_campaign(record)
        self.assertTrue(any("stop_policy.auto_freeze_on_ready" in v for v in violations),
                        violations)

    def test_authority_key_nested_in_a_list_is_found(self):
        record = _campaign()
        record["objective"]["target_regimes"] = [{"may_freeze": True}]
        self.assertTrue(S.find_authority_flavoured_keys(record))
        self.assertTrue(S.validate_campaign(record))

    def test_legitimate_lookalike_keys_are_not_rejected(self):
        # These are real field names in the design; flagging them would make the
        # scan unusable and would push authors to rename around it.
        benign = [
            "release_protocol", "release_plan", "draft_autopilot_rebaseline_note",
            "production_anchor", "parent_candidate_id", "reference_signal",
            "champion_status", "declared_symbol_deltas", "readiness_reporting",
            "license_check", "autopilot_state", "frozen_at",
        ]
        for key in benign:
            with self.subTest(key=key):
                self.assertEqual(S.find_authority_flavoured_keys({key: 1}), [],
                                 f"{key!r} was falsely flagged")

    def test_every_machine_authored_schema_rejects_authority_keys(self):
        for schema, build in FIXTURES.items():
            if schema == S.SCHEMA_OPERATOR_WAIVER:
                continue  # human-authored; an operator attestation lives there
            with self.subTest(schema=schema):
                record = build()
                record["auto_cutover"] = True
                violations = S.SCHEMA_REGISTRY[schema](record)
                self.assertTrue(any("auto_cutover" in v for v in violations), violations)

    def test_valid_fixtures_contain_no_authority_keys(self):
        for schema, build in FIXTURES.items():
            with self.subTest(schema=schema):
                self.assertEqual(S.find_authority_flavoured_keys(build()), [])


# =============================================================================
# Canonical serialisation and content hashing
# =============================================================================

class CanonicalSerialisationTest(unittest.TestCase):
    def test_key_order_does_not_change_the_encoding(self):
        record = _event()
        shuffled = dict(reversed(list(record.items())))
        self.assertEqual(S.canonical_json(record), S.canonical_json(shuffled))
        self.assertEqual(S.content_hash(record), S.content_hash(shuffled))

    def test_nested_key_order_does_not_change_the_hash(self):
        record = _event()
        other = copy.deepcopy(record)
        other["claim_grammar"] = dict(reversed(list(record["claim_grammar"].items())))
        self.assertEqual(S.content_hash(record), S.content_hash(other))

    def test_hash_is_stable_across_a_json_round_trip(self):
        for schema, build in FIXTURES.items():
            with self.subTest(schema=schema):
                record = build()
                reloaded = json.loads(S.canonical_json(record))
                self.assertEqual(S.content_hash(record), S.content_hash(reloaded))

    def test_hash_changes_when_any_value_changes(self):
        record = _event()
        before = S.content_hash(record)
        record["claim_grammar"]["reps"] = 6
        self.assertNotEqual(before, S.content_hash(record))

    def test_list_order_is_significant(self):
        a = {"x": [1, 2]}
        b = {"x": [2, 1]}
        self.assertNotEqual(S.content_hash(a), S.content_hash(b))

    def test_non_string_keys_raise_rather_than_colliding(self):
        # {1: "a"} and {"1": "a"} would otherwise hash identically.
        with self.assertRaises(TypeError):
            S.canonical_json({1: "a"})
        with self.assertRaises(TypeError):
            S.canonical_json({"outer": {2: "b"}})

    def test_non_finite_floats_raise(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    S.canonical_json({"x": bad})

    def test_unserialisable_types_raise(self):
        with self.assertRaises(TypeError):
            S.canonical_json({"x": {1, 2}})
        with self.assertRaises(TypeError):
            S.canonical_json({"x": (1, 2)})

    def test_canonical_bytes_is_utf8_of_canonical_json(self):
        record = {"b": 1, "a": "é"}
        self.assertEqual(S.canonical_bytes(record),
                         S.canonical_json(record).encode("utf-8"))
        self.assertEqual(S.content_hash(record),
                         hashlib.sha256(S.canonical_bytes(record)).hexdigest())


# =============================================================================
# §5.5 item 6 / invariant 20 — narrative is separate and non-retrievable
# =============================================================================

class NarrativeRetrievalTest(unittest.TestCase):
    def test_proposal_requires_the_non_retrievable_marking(self):
        record = _proposal()
        del record["narrative_retrievable"]
        self.assertTrue(any("narrative_retrievable" in v
                            for v in S.validate_proposal(record)))

    def test_narrative_may_not_declare_itself_retrievable(self):
        record = _proposal()
        record["narrative_retrievable"] = True
        self.assertTrue(any("narrative_retrievable" in v
                            for v in S.validate_proposal(record)))

    def test_optional_narrative_elsewhere_still_needs_the_marking(self):
        record = _event()
        record["narrative"] = "a story about why this run was fine"
        self.assertTrue(any("narrative_retrievable" in v
                            for v in S.validate_evaluation_event(record)))
        record["narrative_retrievable"] = False
        self.assertEqual(S.validate_evaluation_event(record), [])

    def test_retrievable_view_strips_prose(self):
        record = _proposal()
        view = S.retrievable_view(record)
        self.assertNotIn("narrative", view)
        self.assertIn("hypothesis", view)
        self.assertIn("narrative", record)  # the record itself is untouched

    def test_retrievable_view_refuses_an_unknown_schema(self):
        # Passing the record through unchanged would leak exactly the prose the
        # retrieval boundary exists to withhold.
        with self.assertRaises(ValueError):
            S.retrievable_view({"schema": "epyc.autokernel.proposal.v99"})


# =============================================================================
# §7.4 — evaluation event: claim grammar, anchor, scope denominator, statuses
# =============================================================================

class EvaluationEventRuleTest(unittest.TestCase):
    def test_inconclusive_is_a_distinct_accepted_status(self):
        self.assertIn("inconclusive", S.EVENT_STATUSES)
        self.assertIn("invalid", S.EVENT_STATUSES)
        for status in ("inconclusive", "invalid", "timeout", "crash", "rejected",
                       "fail", "pass"):
            with self.subTest(status=status):
                record = _event()
                record["status"] = status
                self.assertEqual(S.validate_evaluation_event(record), [])
        record = _event()
        record["status"] = "unknown"
        self.assertTrue(S.validate_evaluation_event(record))

    def test_claim_category_must_be_one_of_the_three(self):
        for category in sorted(S.CLAIM_CATEGORIES):
            record = _event()
            record["claim_grammar"]["category"] = category
            with self.subTest(category=category):
                self.assertEqual(S.validate_evaluation_event(record), [])
        record = _event()
        record["claim_grammar"]["category"] = "OBSERVATION"
        self.assertTrue(S.validate_evaluation_event(record))

    def test_zero_reps_is_not_a_claim(self):
        record = _event()
        record["claim_grammar"]["reps"] = 0
        self.assertTrue(any("reps" in v for v in S.validate_evaluation_event(record)))

    def test_metric_direction_is_constrained(self):
        record = _event()
        record["claim_grammar"]["metric_direction"] = "up"
        self.assertTrue(S.validate_evaluation_event(record))

    def test_anchor_without_measurement_events_is_rejected_above_t0(self):
        record = _event()
        record["anchor"]["measurement_event_ids"] = []
        self.assertTrue(any("measurement_event_ids" in v
                            for v in S.validate_evaluation_event(record)))

    def test_t0_may_compare_artifacts_without_anchor_measurements(self):
        record = _event()
        record["tier"] = "T0"
        record["anchor"]["measurement_event_ids"] = []
        self.assertEqual(S.validate_evaluation_event(record), [])

    def test_partial_scope_must_name_what_it_measured(self):
        record = _event()
        record["scope_denominator"]["numa_nodes"] = []
        record["scope_denominator"]["devices"] = []
        self.assertTrue(any("scope_denominator" in v
                            for v in S.validate_evaluation_event(record)))

    def test_estimate_without_raw_samples_is_a_self_reported_score(self):
        record = _event()
        record["performance"]["raw_samples"] = []
        self.assertTrue(any("estimate" in v
                            for v in S.validate_evaluation_event(record)))

    def test_null_estimate_needs_no_samples(self):
        record = _event()
        record["performance"]["raw_samples"] = []
        record["performance"]["estimate"] = None
        self.assertEqual(S.validate_evaluation_event(record), [])

    def test_pass_cannot_carry_integrity_flags(self):
        record = _event()
        record["integrity_flags"] = ["TRACED_NOT_SUBSET_OF_DERIVED"]
        self.assertTrue(any("integrity_flags" in v
                            for v in S.validate_evaluation_event(record)))
        record["status"] = "invalid"
        self.assertEqual(S.validate_evaluation_event(record), [])

    def test_mutable_evaluator_id_is_refused(self):
        record = _event()
        record["evaluator"]["id"] = "P-AK-SEARCH-1"
        self.assertTrue(any("evaluator.id" in v
                            for v in S.validate_evaluation_event(record)))

    def test_rolled_up_correctness_verdict_is_refused(self):
        record = _event()
        record["correctness"] = True
        self.assertTrue(any("correctness" in v
                            for v in S.validate_evaluation_event(record)))

    def test_co_residency_grammar(self):
        for value, ok in (("single", True), ("co_resident:lineup-quarters", True),
                          ("co_resident:", False), ("maybe", False)):
            with self.subTest(value=value):
                record = _event()
                record["co_residency"] = value
                self.assertEqual(not S.validate_evaluation_event(record), ok)

    def test_naive_timestamp_is_refused(self):
        record = _event()
        record["created_at"] = "2026-08-03T10:45:00"
        self.assertTrue(any("created_at" in v
                            for v in S.validate_evaluation_event(record)))

    def test_determinism_class_cannot_be_claimed_from_zero_repeats(self):
        record = _event()
        record["determinism"]["same_seed_repeat_runs"] = 0
        self.assertTrue(S.validate_evaluation_event(record))
        record["determinism"]["class"] = "not_measured"
        self.assertEqual(S.validate_evaluation_event(record), [])


# =============================================================================
# §7.4 — evaluation_event.v3: the anchor's third component, and the ONE case in
# which a record may carry no anchor at all
# =============================================================================

class EvaluationEventV3AnchorTest(unittest.TestCase):
    """The two defects v3 exists to close, each asserted from both sides.

    Both were RECORDED against the AK3 evaluator rather than patched around, and
    both are schema defects:

      1. `evaluation_event.v2` required `anchor.binary_sha256` unconditionally,
         so the ANCHOR-MISSING void — the one the protocol names explicitly and
         requires to be *"journaled as INVALID with its reason"* — could not
         produce a valid record, and `journal.Journal.append` refused it.
      2. Precondition 4 names the anchor by source commit AND binary SHA-256 AND
         linkage SHA-256; v2's `anchor` carried two of the three, so the commit
         travelled as an unchecked extra key.
    """

    # ---- the anchor is named by all THREE components ----------------------

    def test_source_commit_is_required_when_an_anchor_is_present(self):
        record = _event_v3()
        del record["anchor"]["source_commit"]
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("anchor.source_commit" in v for v in violations), violations)

    def test_source_commit_must_be_a_full_40_hex_commit(self):
        for bad in ("deadbeef", V8_COMMIT.upper(), V8_COMMIT + "0", 1234, None, ""):
            with self.subTest(value=bad):
                record = _event_v3()
                record["anchor"]["source_commit"] = bad
                self.assertTrue(any("anchor.source_commit" in v
                                    for v in S.validate_evaluation_event(record)))

    def test_a_fully_named_anchor_validates(self):
        self.assertEqual(S.validate_evaluation_event(_event_v3()), [])

    # ---- a fabricated anchor is refused, never accepted as "recorded" -----

    def test_a_placeholder_digest_is_rejected_rather_than_read_as_an_anchor(self):
        fillers = {
            "source_commit": ("0" * 40, "f" * 40),
            "binary_sha256": ("0" * 64, "f" * 64,
                              hashlib.sha256(b"").hexdigest()),
            "linkage_sha256": ("0" * 64, "a" * 64),
        }
        for field, values in fillers.items():
            for value in values:
                with self.subTest(field=field, value=value[:8]):
                    record = _event_v3()
                    record["anchor"][field] = value
                    violations = S.validate_evaluation_event(record)
                    self.assertTrue(any("placeholder digest" in v for v in violations),
                                    violations)

    def test_the_void_exemption_cannot_be_taken_by_supplying_a_placeholder(self):
        """The two escape routes are closed together or neither is closed."""
        record = _event_v3_voided_without_an_anchor()
        record["anchor"] = {
            "source_commit": "0" * 40,
            "binary_sha256": "0" * 64,
            "linkage_sha256": "0" * 64,
            "measurement_event_ids": ["ake-20260801-0009"],
        }
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("placeholder digest" in v for v in violations), violations)

    def test_a_real_digest_is_not_falsely_accused(self):
        self.assertFalse(S.is_placeholder_digest(_sha("anchor-binary")))
        self.assertFalse(S.is_placeholder_digest(V8_COMMIT))
        # Not hex, wrong length, or not a string at all: not this check's subject.
        for other in ("", "zz", "0" * 63, "0" * 41, None, 0, ["0" * 64]):
            with self.subTest(value=other):
                self.assertFalse(S.is_placeholder_digest(other))
        for filler in ("0" * 64, "f" * 64, "9" * 40, hashlib.sha256(b"").hexdigest()):
            with self.subTest(value=filler[:8]):
                self.assertTrue(S.is_placeholder_digest(filler))

    # ---- the conditional: absent ONLY for an anchor-voided INVALID record --

    def test_a_voided_anchorless_record_validates(self):
        self.assertEqual(
            S.validate_evaluation_event(_event_v3_voided_without_an_anchor()), [])

    def test_a_pass_record_without_an_anchor_is_rejected(self):
        """The loosest possible reading of the exemption, refused explicitly."""
        record = _event_v3_voided_without_an_anchor()
        record["status"] = "pass"
        record["integrity_flags"] = []
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("anchor" in v for v in violations), violations)

    def test_every_non_invalid_status_still_requires_an_anchor(self):
        for status in sorted(S.EVENT_STATUSES - {"invalid"}):
            with self.subTest(status=status):
                record = _event_v3_voided_without_an_anchor()
                record["status"] = status
                violations = S.validate_evaluation_event(record)
                self.assertTrue(any("anchor: required field is missing" in v
                                    for v in violations), violations)

    def test_invalid_for_some_other_reason_still_requires_an_anchor(self):
        """A voided run that HAD an anchor does not get to drop it."""
        record = _event_v3_voided_without_an_anchor()
        record["integrity_flags"] = ["VOID:HOST_HEALTH_TIER_VIOLATION:FAIL",
                                     "VOID:STORAGE_EXHAUSTED_MID_WINDOW:FAIL"]
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("anchor: required field is missing" in v
                            for v in violations), violations)

    def test_both_anchor_void_reasons_admit_the_omission(self):
        for reason in sorted(S.ANCHOR_VOID_REASONS):
            with self.subTest(reason=reason):
                record = _event_v3_voided_without_an_anchor()
                record["integrity_flags"] = [f"VOID:{reason}:{S.COULD_NOT_CHECK}"]
                self.assertEqual(S.validate_evaluation_event(record), [])

    def test_an_anchor_reason_outside_a_void_flag_does_not_admit_the_omission(self):
        """The flag has to be a VOID flag, not merely a string mentioning one."""
        for flag in ("ANCHOR_MISSING_OR_MUTATED",
                     "INTEGRITY:ANCHOR_MISSING_OR_MUTATED:FAIL",
                     "VOID:ANCHOR_MISSING_OR_MUTATED_LATER:FAIL"):
            with self.subTest(flag=flag):
                record = _event_v3_voided_without_an_anchor()
                record["integrity_flags"] = [flag]
                self.assertTrue(S.validate_evaluation_event(record))

    def test_a_null_anchor_is_refused_so_absence_has_one_representation(self):
        record = _event_v3_voided_without_an_anchor()
        record["anchor"] = None
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("null" in v for v in violations), violations)

    def test_an_anchor_present_but_malformed_still_fails_in_a_voided_record(self):
        """Being INVALID is not permission to carry an unreadable anchor."""
        record = _event_v3_voided_without_an_anchor()
        record["anchor"] = {"source_commit": V8_COMMIT,
                            "binary_sha256": "not-a-sha256",
                            "linkage_sha256": _sha("anchor-linkage"),
                            "measurement_event_ids": []}
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("anchor.binary_sha256" in v for v in violations), violations)

    # ---- v2 keeps its own rules, and is never rewritten -------------------

    def test_a_v2_record_still_validates_under_the_v2_validator(self):
        record = _event()
        self.assertEqual(record["schema"], S.SCHEMA_EVALUATION_EVENT_V2)
        self.assertNotIn("source_commit", record["anchor"])
        self.assertEqual(S.validate_evaluation_event_v2(record), [])
        self.assertEqual(S.validate_record(record), [])
        self.assertEqual(S.validate_evaluation_event(record), [])

    def test_v2_still_requires_its_anchor_unconditionally(self):
        """The v3 exemption is NOT back-ported: v2's rule is v2's rule."""
        record = _event()
        del record["anchor"]
        record["status"] = "invalid"
        record["integrity_flags"] = ["VOID:ANCHOR_MISSING_OR_MUTATED:FAIL"]
        self.assertTrue(any("anchor" in v
                            for v in S.validate_evaluation_event_v2(record)))

    def test_a_v2_record_is_not_a_v3_record(self):
        record = _event()
        record["schema"] = S.SCHEMA_EVALUATION_EVENT_V3
        self.assertTrue(any("anchor.source_commit" in v
                            for v in S.validate_evaluation_event_v3(record)))

    def test_a_v3_void_record_is_not_a_v2_record(self):
        record = _event_v3_voided_without_an_anchor()
        record["schema"] = S.SCHEMA_EVALUATION_EVENT_V2
        self.assertTrue(S.validate_evaluation_event_v2(record))

    def test_the_dispatcher_refuses_an_unknown_evaluation_event_version(self):
        record = _event_v3()
        record["schema"] = "epyc.autokernel.evaluation_event.v6"
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("not an AutoKernel evaluation_event schema" in v
                            for v in violations), violations)
        self.assertTrue(S.validate_evaluation_event(None))

    def test_the_current_schema_string_is_v5(self):
        self.assertEqual(S.SCHEMA_EVALUATION_EVENT, S.SCHEMA_EVALUATION_EVENT_V5)
        self.assertIn(S.SCHEMA_EVALUATION_EVENT_V2, S.KNOWN_SCHEMAS)
        self.assertIn(S.SCHEMA_EVALUATION_EVENT_V3, S.KNOWN_SCHEMAS)
        for schema in (S.SCHEMA_EVALUATION_EVENT_V2, S.SCHEMA_EVALUATION_EVENT_V3,
                       S.SCHEMA_EVALUATION_EVENT_V4,
                       S.SCHEMA_EVALUATION_EVENT_V5):
            self.assertEqual(S.NON_RETRIEVABLE_FIELDS[schema], frozenset({"narrative"}))

    def test_v4_transfer_ratio_is_recomputed_not_trusted(self):
        record = _event_v4()
        record["transfer_ratio_to"][0]["ratio"] = 0.7
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("source_effect / target_effect" in v for v in violations),
                        violations)

    def test_v4_transfer_target_must_match_the_declared_anchor_tier(self):
        record = _event_v4()
        record["transfer_ratio_to"][0]["tier"] = "T1"
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("does not match anchor_tier" in v for v in violations),
                        violations)

    def test_v5_gpu_device_state_rederives_throttle_observed(self):
        record = _event_v5(backend="llama_gpu", device_state={
            "device_id": "ROCm0", "source": "rocm-smi/v6.2",
            "nominal_sclk_mhz": 1700.0, "min_sclk_ratio": 0.9,
            "samples": [{"sclk_mhz": 800.0, "mclk_mhz": 1600.0,
                         "power_w": 180.0, "temperature_c": 61.0,
                         "under_measurement_load": True}],
            "throttle_observed": False, "receipt_ref": "akraw://state/1",
        })
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("does not equal" in v for v in violations), violations)

    def test_v5_gpu_device_state_is_parsed_not_a_text_blob(self):
        record = _event_v5(backend="llama_gpu", device_state="sclk 1700 power 180")
        violations = S.validate_evaluation_event(record)
        self.assertTrue(any("parsed mapping" in v for v in violations), violations)

    def test_the_narrative_is_stripped_from_a_v3_record_too(self):
        record = _event_v3(narrative="the planner's story", narrative_retrievable=False)
        self.assertEqual(S.validate_evaluation_event(record), [])
        self.assertNotIn("narrative", S.retrievable_view(record))


# =============================================================================
# §7.1 / §7.2 — campaign and proposal specific rules
# =============================================================================

class CampaignRuleTest(unittest.TestCase):
    def test_backend_and_source_tree_must_agree(self):
        record = _campaign()
        record["source_tree"] = "whisper.cpp"
        self.assertTrue(any("source_tree" in v for v in S.validate_campaign(record)))

    def test_unknown_backend_is_rejected(self):
        record = _campaign()
        record["backend"] = "vllm"
        self.assertTrue(S.validate_campaign(record))

    def test_phase_outside_the_llama_vocabulary_is_rejected(self):
        record = _campaign()
        record["objective"]["phases"] = ["prefill", "warmup"]
        record["objective"]["protocol_by_phase"] = {
            "prefill": "P-BENCH-PREFILL-1", "warmup": "P-BENCH-1"}
        self.assertTrue(any("warmup" in v for v in S.validate_campaign(record)))

    def test_every_declared_phase_needs_a_protocol(self):
        record = _campaign()
        del record["objective"]["protocol_by_phase"]["decode"]
        self.assertTrue(any("protocol_by_phase" in v
                            for v in S.validate_campaign(record)))

    def test_off_recipe_campaign_is_rejected(self):
        record = _campaign()
        record["objective"]["recipe_class"] = "baseline"
        self.assertTrue(S.validate_campaign(record))

    def test_unbounded_budget_is_rejected(self):
        for bad in (-1, float("inf"), None, "lots"):
            with self.subTest(bad=bad):
                record = _campaign()
                record["budgets"]["max_gpu_hours"] = bad
                self.assertTrue(any("max_gpu_hours" in v
                                    for v in S.validate_campaign(record)))

    def test_zero_budget_is_bounded_and_therefore_legal(self):
        record = _campaign()
        record["budgets"]["max_gpu_hours"] = 0
        self.assertEqual(S.validate_campaign(record), [])

    def test_retry_cap_is_three(self):
        record = _campaign()
        record["stop_policy"]["max_command_retries"] = 4
        self.assertTrue(any("max_command_retries" in v
                            for v in S.validate_campaign(record)))

    def test_scope_hash_must_be_present(self):
        record = _campaign()
        record["scope"]["derived_role_manifest_sha256"] = ""
        self.assertTrue(S.validate_campaign(record))

    def test_phase_trade_exception_must_be_predeclared_in_full(self):
        record = _campaign()
        record["objective"]["phase_trade_exception"] = {"regressing_phase": "prefill"}
        violations = S.validate_campaign(record)
        self.assertTrue(any("band" in v for v in violations), violations)


class ProposalRuleTest(unittest.TestCase):
    def test_core_header_is_an_accepted_change_class(self):
        self.assertIn("core_header", S.CHANGE_CLASSES)
        record = _proposal()
        record["change_class"] = "core_header"
        self.assertEqual(S.validate_proposal(record), [])

    def test_every_change_class_maps_to_a_cheap_suite(self):
        self.assertEqual(set(S.CHANGE_CLASS_CHEAP_SUITE), set(S.CHANGE_CLASSES))
        for change_class in sorted(S.CHANGE_CLASSES):
            with self.subTest(change_class=change_class):
                record = _proposal()
                record["change_class"] = change_class
                self.assertEqual(S.validate_proposal(record), [])

    def test_undeclared_change_class_is_rejected(self):
        record = _proposal()
        record["change_class"] = "vibes"
        self.assertTrue(any("change_class" in v for v in S.validate_proposal(record)))

    def test_symbol_deltas_must_be_declared_even_when_empty(self):
        record = _proposal()
        del record["declared_symbol_deltas"]["removed"]
        self.assertTrue(any("removed" in v for v in S.validate_proposal(record)))

    def test_missing_falsifiable_counter_is_rejected(self):
        record = _proposal()
        record["mechanism_prediction"]["expected_counter_changes"] = {}
        self.assertTrue(any("expected_counter_changes" in v
                            for v in S.validate_proposal(record)))

    def test_missing_wall_share_receipt_is_rejected(self):
        record = _proposal()
        record["mechanism_prediction"]["wall_share_receipt_id"] = ""
        self.assertTrue(S.validate_proposal(record))

    def test_missing_fallback_is_rejected(self):
        record = _proposal()
        record["fallback"]["kill_switch"] = ""
        self.assertTrue(any("kill_switch" in v for v in S.validate_proposal(record)))

    def test_structurally_inseparable_change_must_name_its_risk_class(self):
        record = _proposal()
        record["fallback"] = {"structurally_inseparable": True}
        self.assertTrue(any("risk_class_ack" in v for v in S.validate_proposal(record)))
        record["fallback"]["risk_class_ack"] = "campaign carries irreversible-path risk"
        self.assertEqual(S.validate_proposal(record), [])

    def test_oracle_port_requires_an_oracle_reference(self):
        record = _proposal()
        record["campaign_kind"] = "oracle_port"
        violations = S.validate_proposal(record)
        self.assertTrue(any("oracle_reference" in v for v in violations), violations)
        record["oracle_reference"] = {
            "oracle": "github.com/example/kernels",
            "commit": V7_COMMIT,
            "license_check": "MIT, verified 2026-08-03",
        }
        self.assertEqual(S.validate_proposal(record), [])

    def test_unbounded_resource_request_is_rejected(self):
        record = _proposal()
        record["resource_request"]["expected_storage_gb"] = -1
        self.assertTrue(S.validate_proposal(record))

    def test_unknown_lane_is_rejected(self):
        record = _proposal()
        record["resource_request"]["lane"] = "whatever"
        self.assertTrue(S.validate_proposal(record))

    def test_controller_provenance_is_required(self):
        record = _proposal()
        record["controller"]["prompt_bundle_sha256"] = "not-a-hash"
        self.assertTrue(any("prompt_bundle_sha256" in v
                            for v in S.validate_proposal(record)))

    def test_realized_cost_is_required(self):
        record = _proposal()
        record["realized_cost"]["gpu_seconds"] = None
        self.assertTrue(any("gpu_seconds" in v for v in S.validate_proposal(record)))


# =============================================================================
# §7.3 / §7.5 / §7.6 / §10.4 — candidate, champion, package, waiver rules
# =============================================================================

class CandidateRuleTest(unittest.TestCase):
    def test_production_branch_is_refused_as_a_worktree(self):
        record = _candidate()
        record["worktree"]["branch"] = "production-consolidated-v8"
        self.assertTrue(any("production branch" in v
                            for v in S.validate_candidate(record)))

    def test_stale_production_base_is_refused(self):
        record = _candidate()
        record["ancestry"]["is_descendant_of_production_base"] = False
        self.assertTrue(S.validate_candidate(record))

    def test_supersession_requires_a_reason(self):
        record = _candidate()
        record["status"] = "superseded"
        self.assertTrue(any("supersession_reason" in v
                            for v in S.validate_candidate(record)))
        record["supersession_reason"] = "anchor moved 2026-08-04"
        self.assertEqual(S.validate_candidate(record), [])

    def test_reconciled_surface_requires_a_traced_manifest(self):
        record = _candidate()
        record["affected_surface"]["reconciled"] = True
        self.assertTrue(any("reconciled" in v for v in S.validate_candidate(record)))

    def test_unknown_durability_class_is_rejected(self):
        record = _candidate()
        record["storage"]["durability_class"] = "somewhere"
        self.assertTrue(S.validate_candidate(record))

    def test_natural_key_is_content_addressed(self):
        record = _candidate()
        key = S.candidate_natural_key(record)
        self.assertEqual(key[0], record["campaign_id"])
        self.assertIn(record["artifacts"]["binary_sha256"], key)
        other = _candidate()
        other["artifacts"]["binary_sha256"] = _sha("rebuilt-binary")
        self.assertNotEqual(key, S.candidate_natural_key(other))

    def test_natural_key_raises_rather_than_returning_a_partial_tuple(self):
        record = _candidate()
        del record["artifacts"]["linkage_sha256"]
        with self.assertRaises(KeyError):
            S.candidate_natural_key(record)


class ChampionRuleTest(unittest.TestCase):
    def test_champion_branch_must_be_namespaced(self):
        record = _champion()
        record["branch"] = "main"
        self.assertTrue(S.validate_champion(record))

    def test_champion_branch_may_not_be_a_production_branch(self):
        record = _champion()
        record["branch"] = "production-speech-v1"
        self.assertTrue(any("production branch" in v
                            for v in S.validate_champion(record)))

    def test_non_empty_lineage_needs_a_composed_candidate(self):
        record = _champion()
        record["combined_candidate_id"] = None
        self.assertTrue(any("combined_candidate_id" in v
                            for v in S.validate_champion(record)))

    def test_non_green_champion_must_declare_a_blocking_condition(self):
        record = _champion()
        record["last_t1"] = {"event_id": "ake-x", "status": "fail"}
        self.assertTrue(any("blocking_conditions" in v
                            for v in S.validate_champion(record)))
        record["blocking_conditions"] = ["EVALUATOR_COVERAGE_GAP"]
        self.assertEqual(S.validate_champion(record), [])

    def test_unknown_backend_in_readiness_is_rejected(self):
        record = _champion()
        record["readiness"]["by_backend"]["sglang"] = {}
        self.assertTrue(S.validate_champion(record))


class ReleasePackageRuleTest(unittest.TestCase):
    def test_pass_with_waiver_requires_a_pinned_waiver(self):
        record = _release_package()
        record["t3_verdict"]["verdict"] = "PASS_WITH_WAIVER"
        self.assertTrue(any("active_waivers" in v
                            for v in S.validate_release_package(record)))
        record["active_waivers"] = [
            {"waiver_id": "WAIVE-Q8-20260803", "sha256": _sha("waiver")}]
        self.assertEqual(S.validate_release_package(record), [])

    def test_plain_pass_may_not_pin_waivers(self):
        record = _release_package()
        record["active_waivers"] = [
            {"waiver_id": "WAIVE-Q8-20260803", "sha256": _sha("waiver")}]
        self.assertTrue(any("PASS_WITH_WAIVER" in v
                            for v in S.validate_release_package(record)))

    def test_unverifiable_linkage_is_not_a_pass(self):
        record = _release_package()
        record["linkage_verification"]["status"] = S.COULD_NOT_CHECK
        self.assertTrue(any("linkage_verification.status" in v
                            for v in S.validate_release_package(record)))

    def test_core_header_forces_human_code_review(self):
        record = _release_package()
        record["change_classes"] = ["core_header"]
        self.assertTrue(any("requires_human_code_review" in v
                            for v in S.validate_release_package(record)))
        record["requires_human_code_review"] = True
        self.assertEqual(S.validate_release_package(record), [])

    def test_shared_core_diff_forces_human_code_review(self):
        record = _release_package()
        record["diff_complexity"]["touches_shared_core"] = True
        self.assertTrue(any("requires_human_code_review" in v
                            for v in S.validate_release_package(record)))

    def test_unvalidated_operator_command_is_rejected(self):
        record = _release_package()
        record["operator_command_sequence"][0]["validated"] = False
        self.assertTrue(any("validated" in v
                            for v in S.validate_release_package(record)))

    def test_rollback_plan_needs_an_archived_incumbent(self):
        record = _release_package()
        record["rollback_plan"]["incumbent_archive_path"] = ""
        self.assertTrue(S.validate_release_package(record))

    def test_empty_command_sequence_is_rejected(self):
        record = _release_package()
        record["operator_command_sequence"] = []
        self.assertTrue(S.validate_release_package(record))


class OperatorWaiverRuleTest(unittest.TestCase):
    def test_waiver_must_name_what_it_forfeits(self):
        record = _waiver()
        record["consequences"] = []
        self.assertTrue(any("consequences" in v
                            for v in S.validate_operator_waiver(record)))

    def test_waiver_needs_an_expiry_or_a_reopen_predicate(self):
        record = _waiver()
        record["expiry"] = {"expires_at": None, "reopen_predicate": ""}
        self.assertTrue(any("expiry" in v for v in S.validate_operator_waiver(record)))

    def test_expiry_timestamp_alone_is_sufficient(self):
        record = _waiver()
        record["expiry"] = {"expires_at": "2026-12-31T00:00:00+00:00",
                            "reopen_predicate": ""}
        self.assertEqual(S.validate_operator_waiver(record), [])

    def test_waiver_binds_a_campaign_and_a_protocol(self):
        record = _waiver()
        record["protocol"] = ""
        self.assertTrue(S.validate_operator_waiver(record))


class RepresentationContractTest(unittest.TestCase):
    def test_frame_hash_is_stable_under_key_reordering(self):
        contract = _representation_contract()
        reordered = dict(reversed(list(contract.items())))
        self.assertEqual(
            S.representation_frame_sha256(contract),
            S.representation_frame_sha256(reordered),
        )

    def test_same_frame_is_comparable(self):
        self.assertEqual(
            S.check_representation_comparable(
                _representation_contract(), _representation_contract()
            ).outcome,
            S.PASS,
        )

    def test_changed_demand_is_not_comparable(self):
        left = _representation_contract()
        right = _representation_contract()
        right["empirical_demand"]["receipt_id"] = "rcpt-demand-other"
        right["frame_sha256"] = S.representation_frame_sha256(right)
        check = S.check_representation_comparable(left, right)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("not comparable", check.reasons[0])

    def test_actor_cannot_forge_frame_digest(self):
        record = _proposal()
        record["representation_contract"]["considered_alternatives"].append("new")
        violations = S.validate_proposal(record)
        self.assertTrue(any("frame_sha256" in violation for violation in violations))

    def test_excluded_alternative_must_have_been_considered(self):
        record = _proposal()
        record["representation_contract"]["considered_alternatives"].remove(
            "rewrite_backend"
        )
        record["representation_contract"]["frame_sha256"] = (
            S.representation_frame_sha256(record["representation_contract"])
        )
        violations = S.validate_proposal(record)
        self.assertTrue(any("was not considered" in violation for violation in violations))


# =============================================================================
# Registry dispatch
# =============================================================================

class RegistryDispatchTest(unittest.TestCase):
    def test_registry_maps_each_schema_string_to_its_validator(self):
        expected = {
            S.SCHEMA_CAMPAIGN: S.validate_campaign,
            S.SCHEMA_PROPOSAL_V2: S.validate_proposal_v2,
            S.SCHEMA_PROPOSAL_V3: S.validate_proposal_v3,
            S.SCHEMA_CANDIDATE: S.validate_candidate,
            S.SCHEMA_EVALUATION_EVENT_V2: S.validate_evaluation_event_v2,
            S.SCHEMA_EVALUATION_EVENT_V3: S.validate_evaluation_event_v3,
            S.SCHEMA_EVALUATION_EVENT_V4: S.validate_evaluation_event_v4,
            S.SCHEMA_EVALUATION_EVENT_V5: S.validate_evaluation_event_v5,
            S.SCHEMA_CHAMPION: S.validate_champion,
            S.SCHEMA_RELEASE_PACKAGE: S.validate_release_package,
            S.SCHEMA_OPERATOR_WAIVER: S.validate_operator_waiver,
        }
        self.assertEqual(S.SCHEMA_REGISTRY, expected)

    def test_dispatch_accepts_every_valid_fixture(self):
        for schema, build in FIXTURES.items():
            with self.subTest(schema=schema):
                self.assertEqual(S.validate_record(build()), [])
                self.assertTrue(S.is_valid(build()))

    def test_dispatch_rejects_a_record_under_the_wrong_schema_string(self):
        for schema, build in FIXTURES.items():
            for other in FIXTURES:
                if other == schema:
                    continue
                with self.subTest(record=schema, labelled=other):
                    record = build()
                    record["schema"] = other
                    self.assertTrue(S.validate_record(record))

    def test_unknown_schema_is_a_violation_not_an_exception(self):
        violations = S.validate_record({"schema": "epyc.autokernel.campaign.v1"})
        self.assertTrue(violations)
        self.assertIn("not a known AutoKernel schema", violations[0])
        self.assertFalse(S.is_valid({"schema": "nope"}))

    def test_dispatch_on_non_mapping_and_missing_schema(self):
        for garbage in (None, [], "x", 7):
            with self.subTest(garbage=garbage):
                self.assertTrue(S.validate_record(garbage))
        self.assertTrue(S.validate_record({"campaign_id": "ak-x"}))

    def test_schema_version_is_part_of_the_identity(self):
        for schema in S.KNOWN_SCHEMAS:
            with self.subTest(schema=schema):
                self.assertRegex(schema, r"^epyc\.autokernel\.[a-z_]+\.v\d+$")


# =============================================================================
# Three-outcome checkers — inability to evaluate is never a pass
# =============================================================================

class ScopeDenominatorCheckerTest(unittest.TestCase):
    def test_full_machine_gate_refuses_a_partial_machine_cell(self):
        check = S.check_scope_denominator_admits_gate(
            _event(), {"machine_subset": "full", "cores": 96})
        self.assertEqual(check.outcome, S.FAIL)
        self.assertFalse(check.passed)

    def test_matching_scope_passes(self):
        check = S.check_scope_denominator_admits_gate(
            _event(),
            {"machine_subset": "partial", "numa_nodes": [0],
             "devices": ["gfx90a:0"], "cores": 8},
        )
        self.assertEqual(check.outcome, S.PASS)

    def test_gate_requiring_more_cores_than_the_cell_fails(self):
        check = S.check_scope_denominator_admits_gate(
            _event(), {"machine_subset": "partial", "cores": 96})
        self.assertEqual(check.outcome, S.FAIL)

    def test_missing_scope_is_could_not_check(self):
        record = _event()
        del record["scope_denominator"]
        check = S.check_scope_denominator_admits_gate(record, {"machine_subset": "full"})
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(check.passed)

    def test_undeclared_partial_cell_is_could_not_check(self):
        record = _event()
        record["scope_denominator"]["numa_nodes"] = []
        record["scope_denominator"]["devices"] = []
        check = S.check_scope_denominator_admits_gate(
            record, {"machine_subset": "partial", "numa_nodes": [0]})
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_invalid_outcome_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            S.Check("MAYBE")


class AnchorBindingCheckerTest(unittest.TestCase):
    def test_no_resolver_is_could_not_check(self):
        check = S.check_anchor_binding(_event())
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_missing_anchor_is_a_hard_fail(self):
        record = _event()
        del record["anchor"]
        self.assertEqual(S.check_anchor_binding(record).outcome, S.FAIL)

    def test_unresolvable_anchor_event_fails(self):
        check = S.check_anchor_binding(_event(), resolve_event=lambda _id: None)
        self.assertEqual(check.outcome, S.FAIL)

    def test_anchor_measuring_a_different_binary_fails(self):
        def resolve(_id):
            return {"artifact": {"binary_sha256": _sha("some-other-binary")}}

        self.assertEqual(S.check_anchor_binding(_event(), resolve).outcome, S.FAIL)

    def test_resolved_matching_anchor_passes(self):
        def resolve(_id):
            return {"artifact": {"binary_sha256": _sha("anchor-binary")}}

        self.assertEqual(S.check_anchor_binding(_event(), resolve).outcome, S.PASS)


class MetricCommensurabilityCheckerTest(unittest.TestCase):
    def test_serving_runtime_may_not_report_tokens_per_second(self):
        check = S.check_metric_commensurability(
            "serving_runtime", {"metric": "decode_tokens_per_s"})
        self.assertEqual(check.outcome, S.FAIL)

    def test_kernel_backend_may_not_report_task_rate(self):
        check = S.check_metric_commensurability("llama_gpu", {"metric": "task_rate"})
        self.assertEqual(check.outcome, S.FAIL)

    def test_matching_metric_passes(self):
        self.assertEqual(
            S.check_metric_commensurability(
                "llama_gpu", _event()["claim_grammar"]).outcome, S.PASS)
        self.assertEqual(
            S.check_metric_commensurability(
                "serving_runtime", {"metric": "task_rate"}).outcome, S.PASS)

    def test_unknown_backend_is_could_not_check(self):
        self.assertEqual(
            S.check_metric_commensurability(None, {"metric": "task_rate"}).outcome,
            S.COULD_NOT_CHECK)
        self.assertEqual(
            S.check_metric_commensurability("sglang", {"metric": "task_rate"}).outcome,
            S.COULD_NOT_CHECK)


class CheckWorstOfTest(unittest.TestCase):
    """`Check.worst_of` — the one reducer, and the empty case it exists for."""

    def test_an_empty_vector_is_could_not_check_and_never_pass(self):
        """The defect this classmethod exists to make unwritable.

        Zero evidence is not agreement. Nine of the package's eleven hand-written
        reducers answered PASS here, which is a fail-open verdict that reads as a
        clean one.
        """
        for empty in ([], (), iter([]), (c for c in ())):
            with self.subTest(kind=type(empty).__name__):
                combined = S.Check.worst_of(empty)
                self.assertEqual(combined.outcome, S.COULD_NOT_CHECK)
                self.assertFalse(combined.passed)
                self.assertEqual(combined.reasons, (S.EMPTY_CHECK_VECTOR_REASON,))

    def test_all_pass_is_pass_with_no_reasons(self):
        combined = S.Check.worst_of([S.Check(S.PASS), S.Check(S.PASS)])
        self.assertEqual(combined.outcome, S.PASS)
        self.assertEqual(combined.reasons, ())

    def test_a_pass_check_does_not_donate_its_prose_to_the_combined_record(self):
        """A PASS carries no finding, so its reasons must not ride along.

        Otherwise the combined reason list contains unattributable text and a
        reader cannot tell a finding from a note.
        """
        combined = S.Check.worst_of([S.Check(S.PASS, ("all tools agree",)),
                                     S.Check(S.FAIL, ("digest mismatch",))])
        self.assertEqual(combined.reasons, ("[FAIL] digest mismatch",))

    def test_could_not_check_dominates_pass(self):
        combined = S.Check.worst_of([S.Check(S.PASS), S.Check(S.COULD_NOT_CHECK, ("no fd",))])
        self.assertEqual(combined.outcome, S.COULD_NOT_CHECK)

    def test_fail_dominates_could_not_check_in_either_order(self):
        for vector in ([S.Check(S.COULD_NOT_CHECK, ("a",)), S.Check(S.FAIL, ("b",))],
                       [S.Check(S.FAIL, ("b",)), S.Check(S.COULD_NOT_CHECK, ("a",))]):
            with self.subTest(order=[c.outcome for c in vector]):
                self.assertEqual(S.Check.worst_of(vector).outcome, S.FAIL)

    def test_every_non_pass_reason_is_carried_prefixed_with_its_own_outcome(self):
        """FAIL dominating COULD_NOT_CHECK must not erase which was which."""
        combined = S.Check.worst_of([
            S.Check(S.COULD_NOT_CHECK, ("the attestation file was unreadable",)),
            S.Check(S.FAIL, ("the anchor digest does not match", "and neither does argv")),
        ])
        self.assertEqual(combined.outcome, S.FAIL)
        self.assertEqual(combined.reasons, (
            "[COULD_NOT_CHECK] the attestation file was unreadable",
            "[FAIL] the anchor digest does not match",
            "[FAIL] and neither does argv",
        ))

    def test_a_non_check_element_raises_rather_than_being_skipped(self):
        """A silently ignored foreign object yields PASS for a vector never reduced."""
        for bad in ("PASS", {"outcome": "PASS"}, None, 0):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(TypeError):
                    S.Check.worst_of([S.Check(S.PASS), bad])

    def test_a_generator_is_consumed_exactly_once(self):
        """Emptiness is detected by iteration, not by len() or truthiness.

        A generator is always truthy and has no length, so a reducer that tested
        `if not checks` would return PASS for an exhausted one.
        """
        exhausted = (c for c in [S.Check(S.PASS)])
        list(exhausted)
        self.assertEqual(S.Check.worst_of(exhausted).outcome, S.COULD_NOT_CHECK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
