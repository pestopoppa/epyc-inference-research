"""Hardware-free controller integration for cumulative source composition."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from .. import cumulative_composition as composition
from .. import source_candidate
from . import discovery_controller as controller
from . import gpu_source_adapter
from . import gpu_source_proofs


BASE = "1" * 40
INSTRUMENT = "2" * 40
RUNTIME = {
    "kind": "docker_workspace_bind_only", "docker_path": "/docker",
    "docker_sha256": "a" * 64, "image_id": "image",
    "codex_native_sha256": "a" * 64,
    "code_mode_host_sha256": "a" * 64,
    "ca_certificate_sha256": "a" * 64,
    "writable_host_binds": ["/workspace"],
    "host_network_mode": "docker_bridge",
}
CLAUDE_RUNTIME = {
    "kind": "claude_cli_structured_critic", "provider": "claude",
    "model": "claude-fable-5", "effort": "high",
    "wrapper_path": "/sealed/claude", "wrapper_sha256": "a" * 64,
    "argv_policy_sha256": "a" * 64,
    "auth_staging_policy":
        "ephemeral_0600_copy_atomic_oauth_rotation_sync_no_secret_receipt",
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _candidate() -> controller.PlannedCandidate:
    path = "ggml/src/ggml-cuda/composition-test.cu"
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n"
        "@@ -10,1 +10,1 @@ static int fresh_lever()\n"
        "-return 1;\n+return 2;\n"
    ).encode()
    manifest = source_candidate.SourcePatchManifest(
        campaign_id="ak-cumulative-controller-test",
        proposal_id="akp-cumulative-controller-test",
        candidate_id="akc-cumulative-controller-test",
        source_tree="llama.cpp", production_base_commit=BASE,
        instrument_commit=INSTRUMENT, change_class="arithmetic",
        declared_files=(path,), declared_symbols={path: ("fresh_lever",)},
        mechanism_id="fresh-lever",
        patch_sha256=hashlib.sha256(patch).hexdigest(), patch_bytes=patch)
    return controller.PlannedCandidate(
        hypothesis_id="akh-cumulative-controller-test",
        statement="the fresh lever reduces target runtime",
        falsifier="two isolated repetitions do not remain positive",
        regime={"backend": "gpu", "phase": "decode"},
        proposal={
            "proposal_id": manifest.proposal_id,
            "change_class": manifest.change_class,
            "change": {
                "files_and_symbols": [f"{path}:fresh_lever"],
                "estimated_diff_size": 2,
            },
        }, source_manifest=manifest,
        source_manifest_sha256=manifest.patch_bundle_sha256)


def _second_candidate() -> controller.PlannedCandidate:
    path = "ggml/src/ggml-cuda/composition-test.cu"
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n"
        "@@ -30,1 +30,1 @@ static int second_lever()\n"
        "-return 3;\n+return 4;\n"
    ).encode()
    manifest = source_candidate.SourcePatchManifest(
        campaign_id="ak-cumulative-controller-test",
        proposal_id="akp-cumulative-controller-second",
        candidate_id="akc-cumulative-controller-second",
        source_tree="llama.cpp", production_base_commit=BASE,
        instrument_commit=INSTRUMENT, change_class="arithmetic",
        declared_files=(path,), declared_symbols={path: ("second_lever",)},
        mechanism_id="second-lever",
        patch_sha256=hashlib.sha256(patch).hexdigest(), patch_bytes=patch)
    return controller.PlannedCandidate(
        hypothesis_id="akh-cumulative-controller-second",
        statement="the second lever reduces target runtime",
        falsifier="two isolated repetitions do not remain positive",
        regime={"backend": "gpu", "phase": "decode"},
        proposal={
            "proposal_id": manifest.proposal_id,
            "change_class": manifest.change_class,
            "change": {
                "files_and_symbols": [f"{path}:second_lever"],
                "estimated_diff_size": 2,
            },
        }, source_manifest=manifest,
        source_manifest_sha256=manifest.patch_bundle_sha256)


def _screen(repetition: int, classification: str) -> controller.SealedScreen:
    return controller.SealedScreen(
        receipt_path=f"/sealed/s{repetition}/result.json",
        result_sha256=_sha(f"result-{repetition}"),
        effect_fraction=.01 + repetition / 1000,
        classification=classification,
        baseline_sha256=_sha("baseline"),
        source_proof_sha256=_sha(f"source-{repetition}"),
        dispatch_proof_sha256=_sha(f"dispatch-{repetition}"),
        exact_attribution_effect_fraction=.02,
        target_runtime_effect_fraction=.01 + repetition / 1000,
        series_key=_sha("series"),
        series_effect_fraction=.0115,
        build_identity_sha256=_sha("same-build"),
        correctness_receipt_sha256=_sha(f"correctness-{repetition}"),
        attribution_receipt_sha256=_sha(f"attribution-{repetition}"),
        graphs_off_receipt_sha256=_sha(f"graphs-off-{repetition}"),
        graphs_on_receipt_sha256=_sha(f"graphs-on-{repetition}"))


def _second_screen(
        repetition: int, classification: str) -> controller.SealedScreen:
    return replace(
        _screen(repetition, classification),
        receipt_path=f"/sealed/second/s{repetition}/result.json",
        result_sha256=_sha(f"second-result-{repetition}"),
        source_proof_sha256=_sha(f"second-source-{repetition}"),
        dispatch_proof_sha256=_sha(f"second-dispatch-{repetition}"),
        series_key=_sha("second-series"),
        build_identity_sha256=_sha("second-same-build"),
        correctness_receipt_sha256=
            _sha(f"second-correctness-{repetition}"),
        attribution_receipt_sha256=
            _sha(f"second-attribution-{repetition}"),
        graphs_off_receipt_sha256=
            _sha(f"second-graphs-off-{repetition}"),
        graphs_on_receipt_sha256=
            _sha(f"second-graphs-on-{repetition}"))


def _identity(label: str) -> gpu_source_proofs.BuildIdentity:
    return gpu_source_proofs.BuildIdentity(
        source_commit=_sha(f"commit-{label}")[:40],
        source_sha256=_sha(f"source-{label}"),
        binary_sha256=_sha(f"binary-{label}"),
        hip_library_sha256=_sha(f"hip-{label}"),
        config_sha256=_sha(f"config-{label}"),
        linkage_sha256=_sha(f"linkage-{label}"))


class CumulativeControllerIntegrationTests(unittest.TestCase):
    def _scheduled(self, root: Path):
        item = _candidate()
        first = _screen(1, "candidate")
        second = _screen(2, "top_k_replicated_candidate")
        config = controller.ControllerConfig(
            root, max_iterations=10, dry_run=True,
            production_base_commit=BASE, instrument_commit=INSTRUMENT)
        common = {
            "hypothesis_id": item.hypothesis_id,
            "statement": item.statement, "falsifier": item.falsifier,
            "regime": dict(item.regime),
            "proposal_sha256": controller._sha(item.proposal),
            "source_manifest_sha256": item.source_manifest_sha256,
            "candidate_semantic_sha256":
                controller._candidate_semantic_identity(item),
        }
        state = {
            "iterations": [
                {**controller._screen_iteration_fields(first, repetition=1),
                 **common},
                {**controller._screen_iteration_fields(second, repetition=2),
                 **common},
            ],
            "next": 3, "scientific_attempts": 2,
            "attempted_candidate_identities": {},
        }
        controller._schedule_cumulative_composition(
            state, config=config, item=item, result=second)
        return config, state, controller._restore_pending(
            state["pending"], config)

    def test_replicated_positive_queues_exact_anchor_plus_one_without_actor(self):
        item = _candidate()
        first = _screen(1, "candidate")
        second = _screen(2, "top_k_replicated_candidate")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = controller.ControllerConfig(
                root, max_iterations=10, dry_run=True,
                production_base_commit=BASE,
                instrument_commit=INSTRUMENT)
            state = {
                "iterations": [
                    {**controller._screen_iteration_fields(first, repetition=1),
                     "hypothesis_id": item.hypothesis_id,
                     "statement": item.statement, "falsifier": item.falsifier,
                     "regime": dict(item.regime),
                     "proposal_sha256": controller._sha(item.proposal),
                     "source_manifest_sha256": item.source_manifest_sha256,
                     "candidate_semantic_sha256":
                         controller._candidate_semantic_identity(item)},
                    {**controller._screen_iteration_fields(second, repetition=2),
                     "hypothesis_id": item.hypothesis_id,
                     "statement": item.statement, "falsifier": item.falsifier,
                     "regime": dict(item.regime),
                     "proposal_sha256": controller._sha(item.proposal),
                     "source_manifest_sha256": item.source_manifest_sha256,
                     "candidate_semantic_sha256":
                         controller._candidate_semantic_identity(item)},
                ],
                "next": 3, "scientific_attempts": 2,
                "attempted_candidate_identities": {},
            }
            controller._schedule_cumulative_composition(
                state, config=config, item=item, result=second)
            pending = state["pending"]
            self.assertEqual(pending["phase"], "cumulative_ready")
            restored = controller._restore_pending(pending, config)
            self.assertIsNotNone(restored.composition_plan)
            plan = restored.composition_plan
            assert plan is not None
            self.assertEqual(plan.anchor.accepted, ())
            self.assertEqual(len(plan.candidate.accepted), 1)
            self.assertEqual(
                plan.candidate.accepted[0].manifest, item.source_manifest)
            self.assertEqual(
                plan.candidate.accepted[0].replications[0].result_sha256,
                first.result_sha256)
            self.assertEqual(
                plan.candidate.accepted[0].replications[1].result_sha256,
                second.result_sha256)
            ledger = composition.CompositionLedger(
                root / "cumulative-composition.json").load()
            self.assertIsNone(ledger["pending"])
            self.assertEqual(
                state["pending"]["candidate"]["composition_plan"],
                plan.to_dict())
            self.assertEqual(ledger["scientific_attempts"], 0)

    def test_current_correctness_failure_rolls_back_once_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            plan = item.composition_plan
            assert plan is not None
            pair = composition.CumulativeBuildPair.create(
                plan,
                anchor=composition.BuildBinding.create(
                    plan.anchor.ordered_patch_set_sha256,
                    _identity("failed-anchor"),
                    source_materialization_receipt_sha256=
                        _sha("failed-anchor-source")),
                candidate=composition.BuildBinding.create(
                    plan.candidate.ordered_patch_set_sha256,
                    _identity("failed-candidate"),
                    source_materialization_receipt_sha256=
                        _sha("failed-candidate-source")))
            correctness = composition.FullCorrectness.create(
                pair, suite_id="full-current-suite-v1",
                cases_sha256=_sha("failed-cases"),
                receipt_sha256=_sha("failed-correctness"), passed=False)
            exc = controller.CumulativeCorrectnessRefusal(
                "current stack correctness failed",
                receipt_path="/sealed/correctness-failed.json",
                receipt_sha256=correctness.receipt_sha256,
                build_pair=pair, correctness=correctness)
            terminal = controller._terminalize_cumulative_refusal(
                config, item, exc)
            row = dict(state["pending"]["row"])
            row["operation_key"] = _sha("controller-correctness-operation")
            state["inflight"] = {"operation_key": row["operation_key"]}
            controller._bind_cumulative_terminal_row(row, terminal)
            controller._record_governed_stage_refusal(state, row, exc)
            self.assertEqual(state["scientific_attempts"], 3)
            self.assertEqual(state["iterations"][-1]["status"],
                             "correctness_falsified")
            self.assertEqual(state["iterations"][-1][
                "composition_disposition"], "correctness_rollback")
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json")
            self.assertEqual(ledger.load()["scientific_attempts"], 1)
            self.assertEqual(
                controller._terminalize_cumulative_refusal(
                    config, item, exc), terminal)

    def test_positive_stack_admission_becomes_next_anchor_plus_one(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            first_plan = item.composition_plan
            assert first_plan is not None
            pair = composition.CumulativeBuildPair.create(
                first_plan,
                anchor=composition.BuildBinding.create(
                    first_plan.anchor.ordered_patch_set_sha256,
                    _identity("first-stack-anchor"),
                    source_materialization_receipt_sha256=
                        _sha("first-stack-anchor-source")),
                candidate=composition.BuildBinding.create(
                    first_plan.candidate.ordered_patch_set_sha256,
                    _identity("first-stack-candidate"),
                    source_materialization_receipt_sha256=
                        _sha("first-stack-candidate-source")))
            correctness = composition.FullCorrectness.create(
                pair, suite_id="full-current-suite-v1",
                cases_sha256=_sha("first-stack-cases"),
                receipt_sha256=_sha("first-stack-correctness"), passed=True)
            comparison = composition.IncrementalComparison.create(
                pair, correctness,
                exact_route_receipt_sha256=_sha("first-stack-route"),
                expected_route_set_sha256=_sha("first-stack-route-set"),
                graphs_off_receipt_sha256=_sha("first-stack-graphs-off"),
                graphs_on_receipt_sha256=_sha("first-stack-graphs-on"),
                target_runtime_frame_sha256=_sha("first-stack-frame"),
                exact_route_effect_fraction=.01,
                graphs_off_effect_fraction=.01,
                graphs_on_effect_fraction=.01)
            measured = replace(
                _screen(3, "candidate"),
                composition_build_pair=pair,
                composition_correctness=correctness,
                composition_comparison=comparison)
            terminal = controller._finalize_cumulative_screen(
                config, item, measured)
            self.assertEqual(terminal["disposition"], "admitted")
            admitted_row = dict(state["pending"]["row"])
            controller._bind_cumulative_terminal_row(admitted_row, terminal)
            state["iterations"].append(admitted_row)
            state.pop("pending")
            state["scientific_attempts"] = 3
            state["next"] = 4

            second = _second_candidate()
            first_replication = _second_screen(1, "candidate")
            second_replication = _second_screen(
                2, "top_k_replicated_candidate")
            common = {
                "hypothesis_id": second.hypothesis_id,
                "statement": second.statement,
                "falsifier": second.falsifier,
                "regime": dict(second.regime),
                "proposal_sha256": controller._sha(second.proposal),
                "source_manifest_sha256": second.source_manifest_sha256,
                "candidate_semantic_sha256":
                    controller._candidate_semantic_identity(second),
            }
            state["iterations"].extend((
                {**controller._screen_iteration_fields(
                    first_replication, repetition=1), **common},
                {**controller._screen_iteration_fields(
                    second_replication, repetition=2), **common},
            ))
            state["scientific_attempts"] = 5
            state["next"] = 6
            controller._schedule_cumulative_composition(
                state, config=config, item=second,
                result=second_replication)
            next_item = controller._restore_pending(state["pending"], config)
            next_plan = next_item.composition_plan
            assert next_plan is not None
            self.assertEqual(next_plan.anchor, first_plan.candidate)
            self.assertEqual(len(next_plan.anchor.accepted), 1)
            self.assertEqual(len(next_plan.candidate.accepted), 2)
            self.assertEqual(
                next_plan.candidate.accepted[-1].manifest,
                second.source_manifest)
            self.assertEqual(
                next_plan.anchor.ordered_patch_set_sha256,
                first_plan.candidate.ordered_patch_set_sha256)

    def test_precompute_failure_rolls_back_without_spending_science(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            exc = controller.SourceApplyRefusal(
                "stack application interrupted",
                receipt_path="/sealed/source-apply-failed.json",
                receipt_sha256=_sha("source-apply-failed"))
            terminal = controller._terminalize_cumulative_refusal(
                config, item, exc)
            row = dict(state["pending"]["row"])
            row["operation_key"] = _sha("controller-source-operation")
            controller._bind_cumulative_terminal_row(row, terminal)
            controller._record_governed_stage_refusal(state, row, exc)
            self.assertEqual(controller._derived_scientific_attempts(state), 2)
            self.assertFalse(terminal["scientific_budget_spent"])
            self.assertEqual(terminal["disposition"],
                             "infrastructure_rollback")
            self.assertEqual(
                composition.CompositionLedger(
                    config.output_root /
                    "cumulative-composition.json").load()[
                        "scientific_attempts"], 0)

    def test_attribution_refusal_rolls_back_scientifically_once(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            plan = item.composition_plan
            assert plan is not None
            pair = composition.CumulativeBuildPair.create(
                plan,
                anchor=composition.BuildBinding.create(
                    plan.anchor.ordered_patch_set_sha256,
                    _identity("attribution-anchor"),
                    source_materialization_receipt_sha256=
                        _sha("attribution-anchor-source")),
                candidate=composition.BuildBinding.create(
                    plan.candidate.ordered_patch_set_sha256,
                    _identity("attribution-candidate"),
                    source_materialization_receipt_sha256=
                        _sha("attribution-candidate-source")))
            correctness = composition.FullCorrectness.create(
                pair, suite_id="full-current-suite-v1",
                cases_sha256=_sha("attribution-cases"),
                receipt_sha256=_sha("attribution-correctness"), passed=True)
            receipt = _sha("attribution-refusal")
            exc = controller.CumulativeAttributionRefusal(
                "exact route authority failed",
                receipt_path="/sealed/attribution-failed.json",
                receipt_sha256=receipt,
                build_pair=pair, correctness=correctness)
            terminal = controller._terminalize_cumulative_refusal(
                config, item, exc)
            row = dict(state["pending"]["row"])
            row["operation_key"] = _sha("controller-attribution-operation")
            state["inflight"] = {"operation_key": row["operation_key"]}
            controller._bind_cumulative_terminal_row(row, terminal)
            controller._record_governed_stage_refusal(state, row, exc)
            self.assertEqual(state["scientific_attempts"], 3)
            self.assertEqual(state["iterations"][-1]["status"],
                             "attribution_route_falsified")
            self.assertEqual(terminal["disposition"],
                             "attribution_rollback")
            self.assertEqual(terminal["attribution_receipt_sha256"], receipt)
            self.assertEqual(
                controller._terminalize_cumulative_refusal(
                    config, item, exc), terminal)

    def test_infrastructure_ambiguity_rolls_back_and_requeues_fresh_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            prior_plan = item.composition_plan
            assert prior_plan is not None
            controller_operation = _sha("ambiguous-controller-operation")
            row = dict(state["pending"]["row"])
            state["inflight"] = {
                "operation_key": controller_operation,
                "row": row, "candidate": state["pending"]["candidate"],
                "authorization": {"fixture": "authorization"},
                "confirmation": False, "parent_authorization": None,
                "infrastructure_retry_epoch": 0,
            }
            state.pop("pending")
            exc = controller.ScreenInfrastructureAmbiguity(
                "within-arm output instability",
                receipt_path="/sealed/infrastructure.json",
                receipt_sha256=_sha("infrastructure-receipt"),
                operation_key=controller_operation)
            controller._record_cumulative_infrastructure_ambiguity(
                state, config=config, item=item, row=row, exc=exc)
            self.assertEqual(controller._derived_scientific_attempts(state), 2)
            retry = controller._restore_pending(state["pending"], config)
            self.assertIsNotNone(retry.composition_plan)
            self.assertNotEqual(retry.composition_plan.operation_key,
                                prior_plan.operation_key)
            self.assertEqual(retry.composition_plan.anchor,
                             prior_plan.anchor)
            self.assertEqual(retry.composition_plan.candidate,
                             prior_plan.candidate)
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json").load()
            self.assertEqual(ledger["scientific_attempts"], 0)
            self.assertEqual(ledger["terminals"][0]["disposition"],
                             "infrastructure_rollback")
            self.assertIsNone(ledger["pending"])

    def test_controller_ledger_join_allows_only_owned_crash_window(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            plan = item.composition_plan
            assert plan is not None
            # A controller checkpoint may exist before lazy ledger begin.
            controller._validate_cumulative_composition_state(config, state)
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json")
            ledger.begin(plan)
            # The exact pending plan is owned by the controller checkpoint.
            controller._validate_cumulative_composition_state(config, state)
            pair = composition.CumulativeBuildPair.create(
                plan,
                anchor=composition.BuildBinding.create(
                    plan.anchor.ordered_patch_set_sha256,
                    _identity("join-anchor"),
                    source_materialization_receipt_sha256=
                        _sha("join-anchor-source")),
                candidate=composition.BuildBinding.create(
                    plan.candidate.ordered_patch_set_sha256,
                    _identity("join-candidate"),
                    source_materialization_receipt_sha256=
                        _sha("join-candidate-source")))
            failed = composition.FullCorrectness.create(
                pair, suite_id="full-current-suite-v1",
                cases_sha256=_sha("join-cases"),
                receipt_sha256=_sha("join-correctness"), passed=False)
            ledger.record_build_pair(pair)
            terminal_state = ledger.record_correctness(failed)
            terminal = terminal_state["terminals"][0]
            # The ledger may be one terminal ahead while the exact controller
            # plan is still pending/inflight after a crash.
            controller._validate_cumulative_composition_state(config, state)
            row = dict(state["pending"]["row"])
            controller._bind_cumulative_terminal_row(row, terminal)
            state["iterations"].append(row)
            state.pop("pending")
            controller._validate_cumulative_composition_state(config, state)
            state["iterations"][-1]["composition_terminal_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                    controller.DiscoveryControllerError,
                    "differs from its ledger"):
                controller._validate_cumulative_composition_state(config, state)

    def test_unowned_ledger_pending_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            plan = item.composition_plan
            assert plan is not None
            composition.CompositionLedger(
                config.output_root / "cumulative-composition.json").begin(plan)
            state.pop("pending")
            with self.assertRaisesRegex(
                    controller.DiscoveryControllerError,
                    "lacks controller ownership"):
                controller._validate_cumulative_composition_state(config, state)

    def test_adapter_operation_identity_cryptographically_binds_stack_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            _config, state, item = self._scheduled(Path(directory).resolve())
            operation_key = _sha("adapter-composition-operation")
            authorization = {"claim": "sealed"}
            lease = {"operation_key": operation_key, "repetition": 1,
                     "mode": "hardware-free"}
            intent = gpu_source_adapter._intent_body(
                operation_key=operation_key, candidate=item,
                authorization=authorization, lease=lease)
            inflight = {
                "operation_key": operation_key,
                "candidate": state["pending"]["candidate"],
                "authorization": authorization, "lease": lease,
            }
            identity = gpu_source_adapter._inflight_identity(inflight)
            self.assertEqual(
                intent["composition_plan_sha256"],
                item.composition_plan.plan_sha256)
            self.assertEqual(
                identity["composition_plan_sha256"],
                intent["composition_plan_sha256"])
            altered = dict(inflight)
            altered["candidate"] = dict(inflight["candidate"])
            altered["candidate"]["composition_plan"] = dict(
                altered["candidate"]["composition_plan"])
            altered["candidate"]["composition_plan"]["plan_sha256"] = "0" * 64
            with self.assertRaises(gpu_source_adapter.GpuSourceAdapterError):
                gpu_source_adapter._inflight_identity(altered)

    def test_full_loop_executes_cumulative_third_science_and_rolls_back(self):
        item = _candidate()

        class Planner:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return {**controller.SOL, "runtime": RUNTIME}

            def plan(self, **_kwargs):
                self.calls += 1
                return item

        class Critic:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return {**controller.FABLE5_CRITIC,
                        "runtime": CLAUDE_RUNTIME}

            def review(self, *_args, **_kwargs):
                self.calls += 1
                return controller.Critique("accept", "bounded test acceptance")

        class Lease:
            def admit(self, _item, *, operation_key):
                return {"admitted": True, "operation_key": operation_key,
                        "mode": "hardware-free-fixture"}

        class Screener:
            def __init__(self):
                self.calls = 0

            def reconcile(self, _inflight):
                return controller.Recovery("safe_to_start")

            def screen(self, candidate, _authorization, _permit):
                self.calls += 1
                if self.calls <= 2:
                    return _screen(self.calls, "candidate")
                plan = candidate.composition_plan
                self_outer.assertIsNotNone(plan)
                pair = composition.CumulativeBuildPair.create(
                    plan,
                    anchor=composition.BuildBinding.create(
                        plan.anchor.ordered_patch_set_sha256,
                        _identity("anchor"),
                        source_materialization_receipt_sha256=
                            _sha("anchor-materialization")),
                    candidate=composition.BuildBinding.create(
                        plan.candidate.ordered_patch_set_sha256,
                        _identity("candidate"),
                        source_materialization_receipt_sha256=
                            _sha("candidate-materialization")))
                correctness = composition.FullCorrectness.create(
                    pair, suite_id="full-current-suite-v1",
                    cases_sha256=_sha("cases"),
                    receipt_sha256=_sha("correctness"), passed=True)
                comparison = composition.IncrementalComparison.create(
                    pair, correctness,
                    exact_route_receipt_sha256=_sha("exact-route"),
                    expected_route_set_sha256=_sha("route-set"),
                    graphs_off_receipt_sha256=_sha("graphs-off"),
                    graphs_on_receipt_sha256=_sha("graphs-on"),
                    target_runtime_frame_sha256=_sha("frame"),
                    exact_route_effect_fraction=-.01,
                    graphs_off_effect_fraction=-.01,
                    graphs_on_effect_fraction=-.01)
                return controller.SealedScreen(
                    receipt_path="/sealed/composition/result.json",
                    result_sha256=comparison.result_sha256,
                    effect_fraction=-.01, classification="screened_out",
                    baseline_sha256=_sha("composition-baseline"),
                    source_proof_sha256=_sha("composition-source"),
                    dispatch_proof_sha256=_sha("composition-dispatch"),
                    exact_attribution_effect_fraction=-.01,
                    target_runtime_effect_fraction=-.01,
                    series_key=plan.candidate.ordered_patch_set_sha256,
                    build_identity_sha256=
                        pair.candidate.build_identity_sha256,
                    correctness_receipt_sha256=
                        correctness.receipt_sha256,
                    attribution_receipt_sha256=_sha("exact-route"),
                    graphs_off_receipt_sha256=_sha("graphs-off"),
                    graphs_on_receipt_sha256=_sha("graphs-on"),
                    composition_build_pair=pair,
                    composition_correctness=correctness,
                    composition_comparison=comparison)

        self_outer = self
        planner, critic, screener = Planner(), Critic(), Screener()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                controller, "_write_projection"):
            config = controller.ControllerConfig(
                Path(directory).resolve(), max_iterations=3)
            state = controller.run_controller(
                config, planner=planner, critic=critic,
                screener=screener, lease=Lease())
            self.assertTrue(state["complete"])
            self.assertEqual(state["scientific_attempts"], 3)
            self.assertEqual(planner.calls, 1)
            self.assertEqual(critic.calls, 1)
            self.assertEqual(screener.calls, 3)
            self.assertEqual(
                [row["status"] for row in state["iterations"]],
                ["candidate", "top_k_replicated_candidate", "screened_out"])
            cumulative = state["iterations"][-1]
            self.assertEqual(
                cumulative["composition_disposition"],
                "incremental_rollback")
            self.assertTrue(
                cumulative["composition_scientific_budget_spent"])
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json").load()
            self.assertEqual(ledger["scientific_attempts"], 1)
            self.assertEqual(ledger["authority"]["accepted"], [])
            again = controller.run_controller(
                config, planner=planner, critic=critic,
                screener=screener, lease=Lease())
            self.assertEqual(again, state)
            self.assertEqual((planner.calls, critic.calls, screener.calls),
                             (1, 1, 3))


if __name__ == "__main__":
    unittest.main()
