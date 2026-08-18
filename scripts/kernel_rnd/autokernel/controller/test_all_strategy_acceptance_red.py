"""Hardware-free acceptance gate for every currently eligible GPU strategy.

The non-FA assertions describe seams that are already usable.  The two FA
assertions are intentionally red until every strategy can traverse the same
typed, resumable proof state machine.  This file is test-only audit work; it
does not widen runtime authority.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.benchmark import test_run_autokernel_gpu_discovery as TR
from .. import hypothesis_portfolio
from . import discovery_controller as C
from . import discovery_deployment_factory as F
from . import gpu_source_evidence as E
from . import test_discovery_controller as TD
from . import test_discovery_static_registry as TSR
from . import test_gpu_source_adapter as TA
from .test_gpu_source_evidence import ClaimFactory, FakeExecutors, plan


ELIGIBLE = (
    ("akh-v2-q5-type-specific-dequant", "cuda-vecdotq-v1", "MUL_MAT", 1139),
    ("akh-v2-q8-quantizer-new-mechanism", "cuda-quantize-q8-v1", "MUL_MAT", 1139),
    ("akh-v2-fa-gqa7-pair-tail", "cuda-fattn-tile-v1", "FLASH_ATTN_EXT", 2868),
    ("akh-v2-rms-direct-load-reduction", "cuda-norm-v2", "RMS_NORM", 21),
)


class AllStrategyAcceptanceRedGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.portfolio = hypothesis_portfolio.load(
            hypothesis_portfolio.DEFAULT_PORTFOLIO)
        cls.registry = F._template_registry()
        cls.dispatch = F._portfolio_dispatch_authority(
            cls.registry, cls.portfolio)
        cls.surfaces = F._normalized_template_surfaces(
            cls.registry, cls.portfolio)

    def test_all_four_planner_source_and_current_dispatch_bindings_exist(self):
        records = sorted(
            self.portfolio.eligible_hypotheses(),
            key=lambda row: row["priority"]["rank"])
        self.assertEqual(
            [row["hypothesis_id"] for row in records],
            [row[0] for row in ELIGIBLE])
        for hypothesis_id, template_id, op, cases in ELIGIBLE:
            with self.subTest(hypothesis=hypothesis_id):
                record = next(row for row in records
                              if row["hypothesis_id"] == hypothesis_id)
                template = self.registry.templates[template_id]
                self.assertEqual(
                    tuple(record["target"]["source_files"]),
                    tuple(sorted(template.allowed_files)))
                self.assertTrue(set(record["target"]["source_symbols"]).issubset(
                    template.allowed_symbols[record["target"]["source_files"][0]]))
                self.assertIn(
                    record["mechanism"]["facets"]["change_class"],
                    self.surfaces[template_id]["change_classes"])
                self.assertEqual(template.semantics["correctness_op"], op)
                self.assertEqual(
                    template.semantics["expected_correctness_cases"], cases)
                self.assertGreaterEqual(len(self.dispatch[hypothesis_id]), 1)

    def test_portfolio_accounting_selects_each_next_hypothesis_after_budget(self):
        config = SimpleNamespace(
            hypothesis_portfolio=self.portfolio,
            hypothesis_portfolio_sha256=self.portfolio.sha256,
            planner_context={"portfolio_dispatch_authority": self.dispatch})
        state = {"iterations": [], "portfolio_terminals": {}}
        for hypothesis_id, _template_id, _op, _cases in ELIGIBLE:
            selected = C._select_portfolio_binding(state, config)
            self.assertIsNotNone(selected)
            self.assertEqual(selected["hypothesis_id"], hypothesis_id)
            policy = selected["decision_policy"]
            # A measured terminal is authoritative regardless of whether its
            # budget was 2 or 3 distinct candidates.
            state["portfolio_terminals"][hypothesis_id] = {
                "disposition": policy["terminal_rule"], "policy": policy}
        self.assertIsNone(C._select_portfolio_binding(state, config))

    def _real_portfolio_config(self, root: Path, records, *, iterations: int):
        portfolio = hypothesis_portfolio.Portfolio(
            {"hypotheses": list(records), "frames": [], "do_not_repeat": []},
            "f" * 64)
        authority = {
            row["hypothesis_id"]: self.dispatch[row["hypothesis_id"]]
            for row in records}
        return C.ControllerConfig(
            root, iterations, dry_run=False,
            planner_context={"portfolio_dispatch_authority": authority},
            planner_context_sha256="e" * 64,
            production_base_commit="0" * 40,
            instrument_commit="1" * 40,
            experiment_template_registry_sha256=self.registry.registry_sha256,
            admission_corpus_sha256="c" * 64,
            admission_corpus_version="test-v1",
            deployment_identity_sha256="d" * 64,
            hypothesis_portfolio=portfolio,
            hypothesis_portfolio_sha256=portfolio.sha256)

    @staticmethod
    def _bound_candidate(fixture, binding, context, sequence):
        base = fixture.portfolio_candidate(binding)
        assignment = context["authoring_assignment"]
        patch = base.source_manifest.patch_bytes.replace(
            b"+y\n", f"+y{sequence}\n".encode())
        manifest = dataclasses.replace(
            base.source_manifest,
            campaign_id=assignment["campaign_id"],
            proposal_id=assignment["proposal_id"],
            candidate_id=assignment["candidate_id"],
            patch_bytes=patch,
            patch_sha256=__import__("hashlib").sha256(patch).hexdigest())
        proposal = {
            **base.proposal,
            "proposal_id": assignment["proposal_id"]}
        return dataclasses.replace(
            base, proposal=proposal, source_manifest=manifest,
            source_manifest_sha256=manifest.patch_bundle_sha256)

    def test_measured_terminal_automatically_selects_next_then_exhausts(self):
        """All four ranks must advance without an operator/controller restart."""
        fixture = TD.Tests(methodName="runTest")
        records = sorted(
            self.portfolio.eligible_hypotheses(),
            key=lambda row: row["priority"]["rank"])

        class Planner:
            def __init__(self):
                self.selected = []
                self.sequence = 0

            def attest(self):
                return {**C.SOL, "runtime": TD.RUNTIME}

            def plan(self, *, context, workspace):
                self.sequence += 1
                binding = context["authoring_assignment"]["portfolio_binding"]
                self.selected.append(binding["hypothesis_id"])
                return AllStrategyAcceptanceRedGate._bound_candidate(
                    fixture, binding, context, self.sequence)

        class NegativeScreen:
            def __init__(self):
                self.calls = 0

            def screen(self, *_args):
                self.calls += 1
                return C.SealedScreen(
                    "receipt", f"{self.calls:064x}", -0.01, "candidate",
                    "a" * 64, "b" * 64, "c" * 64)

            def reconcile(self, inflight):
                return C.Recovery("safe_to_start")

        expected = [
            row["hypothesis_id"]
            for row in records
            for _ in range(row["decision_policy"]["max_distinct_candidates"])]
        with tempfile.TemporaryDirectory() as directory:
            config = self._real_portfolio_config(
                Path(directory), records, iterations=len(expected) + 1)
            planner = Planner()
            result = C.run_controller(
                config, planner=planner,
                critic=TD.FakeCritic(["accept"] * len(expected)),
                screener=NegativeScreen(),
                lease=TD.Lease())
        self.assertEqual(planner.selected, expected)
        self.assertEqual(result["terminal_reason"], "portfolio_exhausted")
        self.assertEqual(set(result["portfolio_terminals"]),
                         {row["hypothesis_id"] for row in records})
        self.assertTrue(all(
            row["disposition"] == "retire"
            for row in result["portfolio_terminals"].values()))

    def test_each_strategy_positive_s1_runs_one_s2_then_nominates(self):
        """Replication reuses the candidate and never asks either actor twice."""
        fixture = TD.Tests(methodName="runTest")
        records = {
            row["hypothesis_id"]: row
            for row in self.portfolio.eligible_hypotheses()}
        for hypothesis_id, _template, _op, _cases in ELIGIBLE:
            with self.subTest(hypothesis=hypothesis_id), \
                    tempfile.TemporaryDirectory() as directory:
                record = records[hypothesis_id]

                class Planner:
                    def __init__(self):
                        self.calls = 0

                    def attest(self):
                        return {**C.SOL, "runtime": TD.RUNTIME}

                    def plan(self, *, context, workspace):
                        self.calls += 1
                        binding = context[
                            "authoring_assignment"]["portfolio_binding"]
                        return AllStrategyAcceptanceRedGate._bound_candidate(
                            fixture, binding, context, self.calls)

                class Critic(TD.FakeCritic):
                    def __init__(self):
                        super().__init__(["accept"])
                        self.calls = 0

                    def review(self, *args, **kwargs):
                        self.calls += 1
                        return super().review(*args, **kwargs)

                planner, critic = Planner(), Critic()
                screen = TD.FakeScreen([0.02, 0.02])
                result = C.run_controller(
                    self._real_portfolio_config(
                        Path(directory), [record], iterations=3),
                    planner=planner, critic=critic,
                    screener=screen, lease=TD.Lease())
                self.assertEqual((planner.calls, critic.calls, screen.calls),
                                 (1, 1, 2))
                self.assertEqual(
                    [row["status"] for row in result["iterations"]],
                    ["candidate", "top_k_replicated_candidate"])
                self.assertEqual(
                    result["portfolio_terminals"][hypothesis_id]["disposition"],
                    "nominated")
                self.assertEqual(result["terminal_reason"],
                                 "portfolio_exhausted")

    def test_authoring_refusals_do_not_spend_the_scientific_candidate_budget(self):
        """RED: no measurement cannot establish 'no gain after N candidates'."""
        records = {row["hypothesis_id"]: row
                   for row in self.portfolio.eligible_hypotheses()}
        for hypothesis_id, _template_id, _op, _cases in ELIGIBLE:
            policy = records[hypothesis_id]["decision_policy"]
            for refused_status in (
                    "critic_reject", "screen_refused", "authorization_refused"):
                with self.subTest(hypothesis=hypothesis_id,
                                  status=refused_status):
                    state = {"iterations": [], "portfolio_terminals": {}}
                    for number in range(policy["max_distinct_candidates"]):
                        row = {
                            "status": refused_status,
                            "portfolio_hypothesis_id": hypothesis_id,
                            "portfolio_decision_policy": dict(policy),
                            "source_manifest_sha256": f"{number + 1:064x}",
                        }
                        state["iterations"].append(row)
                        C._apply_portfolio_outcome(state, row)
                    self.assertNotIn(hypothesis_id,
                                     state["portfolio_terminals"])

    def test_fa_has_sealed_distinct_bulk_and_tail_candidate_routes(self):
        """RED: a pair+tail mutation cannot be forced through the anchor route."""
        template = self.registry.templates["cuda-fattn-tile-v1"]
        variants = template.semantics.get("candidate_dispatch_variants")
        self.assertIsInstance(variants, dict)
        self.assertEqual(set(variants), {"gqa7_bulk_pairs", "gqa7_scalar_tail"})
        for name, row in variants.items():
            with self.subTest(route=name):
                self.assertEqual(row["gqa_ratio"], 7)
                self.assertEqual(row["head_size"], 64)
                self.assertIn(row["ncols2"], {1, 2})
                self.assertIsInstance(row["kernel_name"], str)
                self.assertGreater(row["calls"], 0)
                self.assertGreater(row["grid"], 0)
                self.assertGreater(row["workgroup"], 0)
                self.assertGreaterEqual(row["lds_bytes"], 0)
        self.assertEqual(variants["gqa7_bulk_pairs"]["ncols2"], 2)
        self.assertEqual(variants["gqa7_scalar_tail"]["ncols2"], 1)

    def test_q5_exact_duration_compares_the_same_three_routes(self):
        """RED: 3 candidate routes divided by 8 anchor routes is fake gain."""
        hypothesis_id = "akh-v2-q5-type-specific-dequant"
        record = next(
            row for row in self.portfolio.eligible_hypotheses()
            if row["hypothesis_id"] == hypothesis_id)
        template = self.registry.templates["cuda-vecdotq-v1"]
        authority = self.dispatch[hypothesis_id]
        intent = C.GpuSourceExperimentIntent(
            template.template_id, template.target_surface,
            record["target"]["source_symbols"][0],
            template.correctness_id, template.dispatch_id,
            tuple(C.BoundedDispatchExpectation(**row) for row in authority))
        contract = template.bind_dispatch(intent)
        self.assertEqual(len(contract.candidate_exact), 3)
        self.assertEqual(len(contract.anchor_exact), 3)
        self.assertEqual(
            [row.signature for row in contract.anchor_exact],
            [row["route_id"] for row in authority])
        self.assertEqual(
            [(row.calls, row.grid, row.workgroup, row.lds_bytes)
             for row in contract.candidate_exact],
            [(row.calls, row.grid, row.workgroup, row.lds_bytes)
             for row in contract.anchor_exact])
        selected_patterns = {
            row.kernel_pattern for row in contract.anchor_exact}
        unselected_patterns = {
            row.kernel_pattern for row in template.dispatch.anchor_exact
            if row.kernel_pattern not in selected_patterns}
        self.assertEqual(
            {row.kernel_pattern for row in contract.invariants},
            unselected_patterns)

    def test_fa_correctness_requires_and_receipts_exact_odd_gqa7_cases(self):
        """RED: metadata alone cannot make the generic total a GQA7 proof."""
        template = self.registry.templates["cuda-fattn-tile-v1"]
        required = template.semantics.get("required_correctness_cases")
        self.assertIsInstance(required, list)
        self.assertGreaterEqual(len(required), 1)
        for row in required:
            self.assertEqual(row["op"], "FLASH_ATTN_EXT")
            self.assertEqual(row["hsk"], 64)
            self.assertEqual(row["hsv"], 64)
            self.assertEqual(row["gqa_ratio"], 7)
            self.assertEqual(row["query_tokens"], 1)
            self.assertGreaterEqual(row["expected_matches"], 1)
        invocations = template.semantics.get("correctness_invocations")
        self.assertIsInstance(invocations, list)
        self.assertEqual([row["invocation_id"] for row in invocations],
                         ["generic_flash_attn_ext", "odd_gqa7_d64_q1"])
        generic, dedicated = invocations
        self.assertEqual(generic["expected_cases"], 2868)
        self.assertNotIn(
            "AUTOKERNEL_CORRECTNESS_CASE_SET",
            dict(generic.get("environment_overrides", ())))
        self.assertEqual(dedicated["expected_cases"], len(required))
        self.assertEqual(dedicated["required_cases"], required)
        self.assertEqual(dedicated["case_set"], "odd_gqa7_d64_q1_v1")
        self.assertEqual(dict(dedicated["environment_overrides"]), {
            "AUTOKERNEL_CORRECTNESS_CASE_SET": "odd_gqa7_d64_q1_v1"})
        # The plan/policy/receipt API must carry both invocations.  A template
        # declaration that is never consumed is deliberately not acceptance.
        self.assertIn("correctness_invocations",
                      E.GpuSourceEvidencePlan.__dataclass_fields__)

    def test_each_strategy_has_a_separate_graphs_on_target_runtime_screen(self):
        """RED: serialized graphs-off attribution cannot discharge this gate."""
        records = {row["hypothesis_id"]: row
                   for row in self.portfolio.eligible_hypotheses()}
        for hypothesis_id, template_id, _op, _cases in ELIGIBLE:
            with self.subTest(hypothesis=hypothesis_id):
                record = records[hypothesis_id]
                self.assertIs(record["regime"]["target_runtime_graphs"], True)
                screen = self.registry.templates[template_id].semantics.get(
                    "target_runtime_screen")
                self.assertIsInstance(screen, dict)
                self.assertEqual(screen["workload"], "decode_tg128")
                self.assertIs(screen["hip_graphs"], True)
                self.assertIs(screen["paired"], True)
                self.assertIs(screen["decision_required"], True)
                self.assertEqual(screen["stage_id"],
                                 "target_runtime_graphs_on_screen")
                self.assertEqual(screen["exact_invocations"], 1)
                self.assertIs(screen["resume_without_repeat"], True)

    def test_each_strategy_declares_the_exactly_once_resumable_stage_fsm(self):
        """RED: stage receipts, not a fresh-root rule, own crash recovery."""
        expected = (
            "correctness",
            "candidate_attribution",
            "anchor_attribution",
            "measurement_graphs_off_screen",
            "target_runtime_graphs_on_screen",
        )
        for hypothesis_id, template_id, _op, _cases in ELIGIBLE:
            with self.subTest(hypothesis=hypothesis_id):
                fsm = self.registry.templates[template_id].semantics.get(
                    "stage_fsm")
                self.assertIsInstance(fsm, dict)
                self.assertEqual(tuple(fsm["stages"]), expected)
                self.assertEqual(tuple(fsm["crash_after_test_points"]),
                                 expected)
                self.assertEqual(fsm["completed_stage_policy"],
                                 "revalidate_receipt_and_reuse")
                self.assertEqual(fsm["first_incomplete_stage_policy"],
                                 "execute_once")
                self.assertIs(fsm["reject_identity_drift"], True)
                schedule = fsm["attribution_arm_order_schedule"]
                self.assertIs(schedule["counterbalanced"], True)
                self.assertEqual(tuple(schedule["s1"]),
                                 tuple(reversed(schedule["s2"])))
                self.assertEqual(set(schedule["s1"]),
                                 {"candidate", "anchor"})

    def test_each_strategy_requires_exact_attribution_and_graphs_on_gain(self):
        """RED: an opaque attribution hash cannot discharge a falsifier."""
        for hypothesis_id, template_id, _op, _cases in ELIGIBLE:
            with self.subTest(hypothesis=hypothesis_id):
                decision = self.registry.templates[template_id].semantics.get(
                    "decision_evidence")
                self.assertIsInstance(decision, dict)
                self.assertIs(decision["all_exact_routes_have_duration"], True)
                self.assertIs(decision["exact_attribution_gain_required"], True)
                self.assertIs(decision["target_runtime_graphs_on_gain_required"],
                              True)
                self.assertEqual(decision["combination"], "conjunction")
                self.assertEqual(decision["direction"],
                                 "lower_exact_duration_and_higher_throughput")
                self.assertIs(
                    decision["short_circuit_graphs_on_when_exact_nonpositive"],
                    True)

    def test_profile_reducer_preserves_exact_route_duration(self):
        """RED: BeginNs/EndNs are parsed today and then silently discarded."""
        rows = [
            {"kernel": "route", "grid": 128, "workgroup": 64, "lds": 0,
             "blocks_per_call": 2, "begin_ns": 100, "end_ns": 130},
            {"kernel": "route", "grid": 128, "workgroup": 64, "lds": 0,
             "blocks_per_call": 2, "begin_ns": 200, "end_ns": 250},
        ]
        exact = (E.ExactDispatch(
            "route", r"^route$", 2, 128, 64, 0, 2),)
        reduced = E._reduce_arm(
            rows, exact=exact, forbidden=(), invariants=())
        route = reduced["exact"]["route"]
        self.assertEqual(route["total_duration_ns"], 80)
        self.assertEqual(route["duration_ns"], [30, 50])
        self.assertEqual(route["median_duration_ns"], 40)

    def test_invariant_comparison_excludes_independent_arm_timing_noise(self):
        """RED: invariants compare topology; exact routes own time effects."""
        exact = (E.ExactDispatch(
            "target", r"^target$", 1, 128, 64, 0, 2),)
        invariants = (E.InvariantDispatch("unchanged", r"^unchanged$"),)
        candidate = E._reduce_arm([
            {"kernel": "target", "grid": 128, "workgroup": 64, "lds": 0,
             "blocks_per_call": 2, "begin_ns": 100, "end_ns": 120},
            {"kernel": "unchanged", "grid": 64, "workgroup": 64, "lds": 0,
             "blocks_per_call": 1, "begin_ns": 100, "end_ns": 150},
        ], exact=exact, forbidden=(), invariants=invariants)
        anchor = E._reduce_arm([
            {"kernel": "target", "grid": 128, "workgroup": 64, "lds": 0,
             "blocks_per_call": 2, "begin_ns": 100, "end_ns": 130},
            {"kernel": "unchanged", "grid": 64, "workgroup": 64, "lds": 0,
             "blocks_per_call": 1, "begin_ns": 100, "end_ns": 180},
        ], exact=exact, forbidden=(), invariants=invariants)
        self.assertEqual(candidate["invariants"], anchor["invariants"])
        self.assertNotEqual(
            candidate["exact"]["target"]["total_duration_ns"],
            anchor["exact"]["target"]["total_duration_ns"])

    def test_sealed_decision_carries_both_effects_and_requires_conjunction(self):
        """RED: one whole-model scalar cannot stand in for exact attribution."""
        fields = C.SealedScreen.__dataclass_fields__
        self.assertIn("exact_attribution_effect_fraction", fields)
        self.assertIn("target_runtime_effect_fraction", fields)
        result = C.SealedScreen(
            receipt_path="result.json", result_sha256="a" * 64,
            effect_fraction=.05, classification="candidate",
            baseline_sha256="b" * 64, source_proof_sha256="c" * 64,
            dispatch_proof_sha256="d" * 64,
            exact_attribution_effect_fraction=-.01,
            target_runtime_effect_fraction=.05)
        classified = C._classified_result(
            {"iterations": []},
            SimpleNamespace(source_manifest_sha256="e" * 64, regime={}),
            result)
        self.assertNotIn(classified.classification,
                         {"candidate", "top_k_replicated_candidate"})

    def test_controller_stage_row_preserves_dual_decision_evidence(self):
        result = C.SealedScreen(
            receipt_path="result.json", result_sha256="a" * 64,
            effect_fraction=.04, classification="candidate",
            baseline_sha256="b" * 64, source_proof_sha256="c" * 64,
            dispatch_proof_sha256="d" * 64,
            exact_attribution_effect_fraction=.03,
            target_runtime_effect_fraction=.04,
            stages=("materialized", "built", "correctness", "attribution",
                    "measurement_graphs_off_screen",
                    "target_runtime_graphs_on_screen"))
        row = C._screen_iteration_fields(result, repetition=2)
        self.assertEqual(
            (row["exact_attribution_effect_fraction"],
             row["target_runtime_effect_fraction"],
             row["target_runtime_executed"], row["target_runtime_reason"],
             row["repetition"]),
            (.03, .04, True, None, 2))
        self.assertEqual(row["stages"], list(result.stages))

    def test_nonpositive_exact_duration_is_measured_and_short_circuits_graphs_on(self):
        """A valid neutral/regression is evidence, not an execution refusal."""
        fixture = TD.Tests(methodName="runTest")
        record = fixture.portfolio_record(
            hypothesis_id="akh-exact-nonpositive", rank=1, budget=1)
        with tempfile.TemporaryDirectory() as directory:
            config = fixture.portfolio_config(Path(directory), [record])
            binding = C._select_portfolio_binding(
                {"iterations": [], "portfolio_terminals": {}}, config)
            item = fixture.portfolio_candidate(binding)
            result = C.SealedScreen(
                receipt_path="result.json", result_sha256="a" * 64,
                effect_fraction=0.0, classification="candidate",
                baseline_sha256="b" * 64, source_proof_sha256="c" * 64,
                dispatch_proof_sha256="d" * 64,
                exact_attribution_effect_fraction=0.0,
                target_runtime_effect_fraction=None)
            classified = C._classified_result(
                {"iterations": []}, item, result,
                binding["decision_policy"])
            self.assertIn(classified.classification,
                          {"inconclusive", "screened_out"})
            self.assertIsNone(classified.target_runtime_effect_fraction)
            row = {
                "status": classified.classification,
                "portfolio_hypothesis_id": binding["hypothesis_id"],
                "portfolio_decision_policy": binding["decision_policy"],
                "source_manifest_sha256": item.source_manifest_sha256,
                "result_sha256": classified.result_sha256,
                "evidence": {
                    "baseline": classified.baseline_sha256,
                    "source": classified.source_proof_sha256,
                    "dispatch": classified.dispatch_proof_sha256},
                "scientific_budget_spent": True,
            }
            state = {"iterations": [row], "portfolio_terminals": {}}
            C._apply_portfolio_outcome(state, row)
            self.assertIn(binding["hypothesis_id"],
                          state["portfolio_terminals"])

    def test_graphs_on_process_environment_omits_presence_only_disable_flag(self):
        """RED: `DISABLE_GRAPHS=0` still disables graphs in llama.cpp."""
        fixture = TR.TestGpuDiscoveryBatchedSubprocess(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = TR._build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"
            model.write_bytes(b"model")
            process = fixture._Process(fixture._row([100.0] * 3))
            seen = []
            result = C.gpu_discovery._invoke_locked(
                build=build, model=model, seed=8613, baseline_vram=0,
                flash_attention=True,
                expected_source_commit=C.gpu_discovery.SOURCE_COMMIT,
                repetitions=3, runtime_graphs="on",
                process_factory=lambda argv, **kwargs: (
                    seen.append((argv, kwargs)) or process),
                kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                pgid_provider=lambda _pid: process.pid, sleep=lambda _: None)
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", seen[0][1]["env"])
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", result["env"])
        self.assertEqual(result["metric_contract"]["graph_environment"],
                         {"GGML_CUDA_DISABLE_GRAPHS": None})
        self.assertIs(
            result["graphs_on_output_integrity"]["reward_admissible"], True)

    def test_graphs_on_reward_inspects_outputs_and_unique_inputs(self):
        """RED: phase/content replay must fail before its speed is scored."""
        fixture = TR.TestGpuDiscoveryBatchedSubprocess(methodName="runTest")
        for corruption, pattern in (
                ("output_mismatch", "output"),
                ("reused_input", "input")):
            with self.subTest(corruption=corruption), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                build = TR._build(root, rocwmma="ON", mfma="OFF")
                model = root / "model.gguf"
                model.write_bytes(b"model")
                row = __import__("json").loads(fixture._row([100.0] * 3))
                if corruption == "output_mismatch":
                    row["autokernel_output_hashes"] = (
                        "a0a0a0a0a0a0a065/f0f0f0f0f0f0f0ff,"
                        "a0a0a0a0a0a0a066/a0a0a0a0a0a0a066,"
                        "a0a0a0a0a0a0a067/a0a0a0a0a0a0a067")
                else:
                    row["autokernel_input_hashes"] = ",".join(
                        ["a1a1a1a1a1a1a1a1"] * 3)
                process = fixture._Process(
                    __import__("json").dumps(row) + "\n")
                with self.assertRaisesRegex(RuntimeError, pattern):
                    C.gpu_discovery._invoke_locked(
                        build=build, model=model, seed=8613,
                        baseline_vram=0, flash_attention=True,
                        expected_source_commit=C.gpu_discovery.SOURCE_COMMIT,
                        repetitions=3, runtime_graphs="on",
                        process_factory=lambda _argv, **_kwargs: process,
                        kfd_pid_provider=lambda: (123,),
                        vram_reader=lambda: 64,
                        pgid_provider=lambda _pid: process.pid,
                        sleep=lambda _: None)

    def test_nonpositive_exact_attribution_never_starts_a_model_screen(self):
        """RED: route-level falsification short-circuits both runner stages."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / "anchor"
            candidate = root / "candidate"
            anchor.mkdir()
            candidate.mkdir()
            source_file = root / "source.json"
            dispatch_file = root / "dispatch.json"
            source_file.write_text("source")
            dispatch_file.write_text("dispatch")
            source_hash = __import__("hashlib").sha256(
                source_file.read_bytes()).hexdigest()
            dispatch_hash = __import__("hashlib").sha256(
                dispatch_file.read_bytes()).hexdigest()
            build = C.GpuSourceBuild(
                anchor, candidate,
                C.gpu_source_proofs.BuildIdentity(
                    "commit-a", "a" * 64, "a" * 64, "a" * 64,
                    "a" * 64, "a" * 64),
                C.gpu_source_proofs.BuildIdentity(
                    "commit-b", "b" * 64, "b" * 64, "b" * 64,
                    "b" * 64, "b" * 64))
            material = {
                "manifest_sha256": "a" * 64,
                "candidate": build.candidate_identity,
                "anchor": build.anchor_identity,
                "workload_sha256": "a" * 64,
                "correctness": {
                    "file_sha256": source_hash, "native_sha256": "a" * 64},
                "attribution": {
                    "file_sha256": dispatch_hash, "native_sha256": "a" * 64,
                    "body": {"exact_duration_comparison": {
                        "relative_improvement_fraction": -0.01,
                        "candidate_routes": {
                            "candidate": {"total_duration_ns": 101}},
                        "anchor_routes": {
                            "anchor": {"total_duration_ns": 100}}}}},
            }
            hashed = {
                **material,
                "candidate": build.candidate_identity.__dict__,
                "anchor": build.anchor_identity.__dict__}
            bundle = C.gpu_source_proofs.GpuSourceProofBundle(
                **material,
                bundle_sha256=C.gpu_source_proofs._hash(hashed))
            item = SimpleNamespace(source_manifest_sha256="a" * 64)
            graphs_off = SimpleNamespace(
                factor="source_patch", anchor_build=str(anchor),
                candidate_build=str(candidate),
                output_dir=str(root / "graphs-off"))
            graphs_on = SimpleNamespace(
                factor="source_patch", anchor_build=str(anchor),
                candidate_build=str(candidate),
                output_dir=str(root / "graphs-on"))
            graphs_off._target_runtime_args = graphs_on
            screener = C.GpuSourceScreener(
                build_source=lambda *_args: build,
                proof_bundle=lambda *_args: bundle,
                args_factory=lambda *_args: graphs_off)
            with mock.patch.object(
                    C.gpu_discovery, "run",
                    side_effect=AssertionError("model screen executed")):
                result = screener.screen(item, object(), {})
        self.assertEqual(result.exact_attribution_effect_fraction, -0.01)
        self.assertIsNone(result.target_runtime_effect_fraction)
        self.assertNotIn(result.classification,
                         {"candidate", "top_k_replicated_candidate"})

    def test_graphs_off_and_on_receipts_resume_without_repeating_process(self):
        """A crash after either model stage reuses its durable result exactly once."""
        class CrashAfterDurableResult(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor, candidate = root / "anchor", root / "candidate"
            anchor.mkdir(); candidate.mkdir()
            correctness_file, attribution_file = root / "correctness.json", root / "dispatch.json"
            correctness_file.write_text("correctness")
            attribution_file.write_text("dispatch")
            correctness_sha = hashlib.sha256(
                correctness_file.read_bytes()).hexdigest()
            attribution_sha = hashlib.sha256(
                attribution_file.read_bytes()).hexdigest()
            build = C.GpuSourceBuild(
                anchor, candidate,
                C.gpu_source_proofs.BuildIdentity(
                    "commit-a", "a" * 64, "a" * 64, "a" * 64,
                    "a" * 64, "a" * 64),
                C.gpu_source_proofs.BuildIdentity(
                    "commit-b", "b" * 64, "b" * 64, "b" * 64,
                    "b" * 64, "b" * 64))
            material = {
                "manifest_sha256": "a" * 64,
                "candidate": build.candidate_identity,
                "anchor": build.anchor_identity,
                "workload_sha256": "a" * 64,
                "correctness": {
                    "file_sha256": correctness_sha,
                    "native_sha256": "a" * 64},
                "attribution": {
                    "file_sha256": attribution_sha,
                    "native_sha256": "a" * 64,
                    "body": {"exact_duration_comparison": {
                        "relative_improvement_fraction": .05,
                        "candidate_routes": {"candidate": {
                            "total_duration_ns": 95}},
                        "anchor_routes": {"anchor": {
                            "total_duration_ns": 100}}}}},
            }
            hashed = {
                **material,
                "candidate": build.candidate_identity.__dict__,
                "anchor": build.anchor_identity.__dict__}
            bundle = C.gpu_source_proofs.GpuSourceProofBundle(
                **material,
                bundle_sha256=C.gpu_source_proofs._hash(hashed))
            off = SimpleNamespace(
                factor="source_patch", anchor_build=str(anchor),
                candidate_build=str(candidate), output_dir=str(root / "off"),
                runtime_graphs="off")
            on = SimpleNamespace(
                factor="source_patch", anchor_build=str(anchor),
                candidate_build=str(candidate), output_dir=str(root / "on"),
                runtime_graphs="on")
            off._target_runtime_args = on
            item = SimpleNamespace(source_manifest_sha256="a" * 64)
            screener = C.GpuSourceScreener(
                build_source=lambda *_args: build,
                proof_bundle=lambda *_args: bundle,
                args_factory=lambda *_args: off)
            process_calls = []

            def durable_then_crash(current):
                graph_mode = current.runtime_graphs
                process_calls.append(graph_mode)
                body = {
                    "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
                    "non_promotable": True, "promotion_claim": False,
                    "hip_residency_proved": True,
                    "runtime_graphs": graph_mode,
                    "median_relative": .04,
                    "baseline_sha256": "c" * 64,
                }
                body["result_sha256"] = C.gpu_source_proofs._hash(body)
                output = Path(current.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "result.json").write_text(
                    __import__("json").dumps(body, sort_keys=True))
                raise CrashAfterDurableResult(graph_mode)

            with mock.patch.object(
                    C.gpu_discovery, "run", side_effect=durable_then_crash):
                with self.assertRaises(CrashAfterDurableResult):
                    screener.screen(item, object(), {})
                with self.assertRaises(CrashAfterDurableResult):
                    screener.screen(item, object(), {})
                with mock.patch.object(
                        C.autokernel_progression, "_gpu_screen",
                        return_value={"stage": "candidate"}):
                    result = screener.screen(item, object(), {})
            self.assertEqual(process_calls, ["off", "on"])
            self.assertEqual(result.target_runtime_effect_fraction, .04)

    def test_governed_adapter_resumes_each_dual_runner_terminal(self):
        """Public reconcile/re-entry must understand the two-output runner plan."""
        class CrashAfterDurableResult(BaseException):
            pass

        class EpochReservation(TA.ReservationManager):
            def __init__(self):
                super().__init__()
                self.claim_ids = []

            def reserve(self, operation_key):
                self.reserve_calls += 1
                epoch = self.reserve_calls
                self.outer = E.device_claim.ClaimReceipt(
                    claim_id=f"akd-outer-{epoch}", device_id="mi210_0",
                    lock_path=f"/claim-{epoch}", state="held",
                    holder_pid=epoch, holder_start_ticks=epoch,
                    holder_boot_id="boot", host="host",
                    holder_label="outer", purpose="outer reservation",
                    campaign_id="ak-gpu-source-evidence-test",
                    acquired_at=f"2026-08-14T00:00:0{epoch}Z")
                self.claim_ids.append(self.outer.claim_id)
                self.active = True
                return self.outer.to_dict()

        helper = TA.GpuSourceAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            values = helper.setup(directory)
            adapter, candidate, authorization, lease, inflight, _current, executors = values
            manager = EpochReservation()
            adapter.reservation_manager = manager
            operation_root = adapter._root(lease["operation_key"])
            off = SimpleNamespace(
                factor="source_patch",
                anchor_build=str(Path(directory) / "anchor-build"),
                candidate_build=str(Path(directory) / "candidate-build"),
                output_dir=str(operation_root / "runner" / "graphs-off"),
                runtime_graphs="off")
            on = SimpleNamespace(
                factor="source_patch",
                anchor_build=str(Path(directory) / "anchor-build"),
                candidate_build=str(Path(directory) / "candidate-build"),
                output_dir=str(operation_root / "runner" / "graphs-on"),
                runtime_graphs="on")
            off._target_runtime_args = on
            adapter.args_factory = lambda *_args: off
            original_rocprof = executors.rocprof

            def positive_exact(invocation):
                capture = original_rocprof(invocation)
                if invocation.arm == "anchor":
                    path = invocation.timestamp_csv_path
                    raw = path.read_text()
                    path.write_text(raw.replace(
                        ",10,20", ",10,30").replace(
                        ",21,30", ",21,50"))
                return capture

            adapter.rocprof_executor = positive_exact
            process_calls = []

            def durable_then_crash(current):
                mode = current.runtime_graphs
                process_calls.append(mode)
                opened = manager.outer.to_dict()
                phase_end = {
                    "schema": "epyc.autokernel.borrowed_device_claim_phase.v1",
                    "mode": "borrowed_outer_reservation",
                    "outer_claim_id": opened["claim_id"],
                    "device_id": opened["device_id"],
                    "campaign_id": opened["campaign_id"],
                    "phase_ended_at": "2026-08-14T00:00:30Z",
                    "physical_release": False,
                }
                body = {
                    "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
                    "non_promotable": True, "promotion_claim": False,
                    "hip_residency_proved": True, "runtime_graphs": mode,
                    "median_relative": .04,
                    "baseline_sha256": "c" * 64,
                    "device_claim_mode": "borrowed_outer_reservation",
                    "device_claim_open": opened,
                    "device_claim_borrowed_phase_end": phase_end,
                    "device_claim_released": None,
                }
                body["result_sha256"] = C.gpu_source_proofs._hash(body)
                output = Path(current.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "result.json").write_text(
                    __import__("json").dumps(body, sort_keys=True))
                (output / "live-governance.json").write_text(
                    __import__("json").dumps({
                        "status": "borrowed_phase_ended",
                        "device_claim_mode": "borrowed_outer_reservation",
                        "device_claim_open": opened,
                        "device_claim_borrowed_phase_end": phase_end,
                        "device_claim_released": None,
                    }, sort_keys=True))
                raise CrashAfterDurableResult(mode)

            with mock.patch.object(C.gpu_discovery, "run",
                                   side_effect=durable_then_crash), \
                    mock.patch.object(C.autokernel_progression, "_gpu_screen",
                                      return_value={"stage": "candidate"}):
                with self.assertRaises(CrashAfterDurableResult):
                    adapter.screen(candidate, authorization, lease)
                self.assertEqual((manager.reserve_calls, manager.release_calls),
                                 (1, 1))
                self.assertEqual(adapter.reconcile(inflight).status,
                                 "safe_to_start")
                with self.assertRaises(CrashAfterDurableResult):
                    adapter.screen(candidate, authorization, lease)
                self.assertEqual((manager.reserve_calls, manager.release_calls),
                                 (2, 2))
                recovered_after_on = adapter.reconcile(inflight)
                self.assertIn(recovered_after_on.status,
                              {"safe_to_start", "sealed_result"})
                result = (recovered_after_on.result
                          if recovered_after_on.status == "sealed_result"
                          else adapter.screen(candidate, authorization, lease))
            self.assertEqual(process_calls, ["off", "on"])
            self.assertEqual(
                [call[:2] for call in executors.calls],
                [("correctness", "candidate"),
                 ("rocprof", "candidate"), ("rocprof", "anchor")])
            self.assertGreater(result.exact_attribution_effect_fraction, 0)
            self.assertEqual(result.target_runtime_effect_fraction, .04)
            self.assertEqual(manager.reserve_calls, manager.release_calls)
            self.assertFalse(manager.active)
            self.assertEqual(len(set(manager.claim_ids)), len(manager.claim_ids))
            recovered = adapter.reconcile(inflight)
            self.assertEqual(recovered.status, "sealed_result")
            self.assertEqual(
                (recovered.result.exact_attribution_effect_fraction,
                 recovered.result.target_runtime_effect_fraction),
                (result.exact_attribution_effect_fraction, .04))

    def test_three_real_critic_rejections_skip_only_that_strategy(self):
        """RED: exercise the live critic_pending branch, not a helper directly."""
        fixture = TD.Tests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            records = [
                fixture.portfolio_record(
                    hypothesis_id="akh-first", rank=1, budget=2),
                fixture.portfolio_record(
                    hypothesis_id="akh-second", rank=2, budget=2),
            ]
            config = dataclasses.replace(
                fixture.portfolio_config(Path(directory), records),
                max_iterations=4)

            class BoundPlanner:
                def __init__(self):
                    self.selected = []

                def attest(self):
                    return {**C.SOL, "runtime": {
                        "kind": "docker_workspace_bind_only",
                        "docker_path": "/docker", "docker_sha256": "a" * 64,
                        "image_id": "image", "codex_native_sha256": "a" * 64,
                        "code_mode_host_sha256": "a" * 64,
                        "ca_certificate_sha256": "a" * 64,
                        "writable_host_binds": ["/workspace"],
                        "host_network_mode": "docker_bridge"}}

                def plan(self, *, context, workspace):
                    binding = context["authoring_assignment"]["portfolio_binding"]
                    self.selected.append(binding["hypothesis_id"])
                    return fixture.portfolio_candidate(binding)

            class Never:
                def __getattr__(self, _name):
                    def fail(*_args, **_kwargs):
                        raise AssertionError("dry authoring audit reached compute")
                    return fail

            planner = BoundPlanner()
            result = C.run_controller(
                config, planner=planner,
                critic=TD.FakeCritic(["reject", "reject", "reject", "accept"]),
                screener=Never(), lease=Never())
            self.assertEqual(planner.selected,
                             ["akh-first", "akh-first", "akh-first", "akh-second"])
            self.assertIn("akh-first", result["portfolio_skips"])
            self.assertNotIn("akh-first", result["portfolio_terminals"])

    def test_dry_run_advances_through_every_eligible_strategy_once(self):
        """RED: authorization-only validation must not spin on rank one."""
        fixture = TD.Tests(methodName="runTest")

        class BoundPlanner:
            def __init__(self):
                self.selected = []

            def attest(self):
                return {**C.SOL, "runtime": {
                    "kind": "docker_workspace_bind_only",
                    "docker_path": "/docker", "docker_sha256": "a" * 64,
                    "image_id": "image", "codex_native_sha256": "a" * 64,
                    "code_mode_host_sha256": "a" * 64,
                    "ca_certificate_sha256": "a" * 64,
                    "writable_host_binds": ["/workspace"],
                    "host_network_mode": "docker_bridge"}}

            def plan(self, *, context, workspace):
                binding = context["authoring_assignment"]["portfolio_binding"]
                self.selected.append(binding["hypothesis_id"])
                return fixture.portfolio_candidate(binding)

        class NeverCompute:
            def __getattr__(self, _name):
                def fail(*_args, **_kwargs):
                    raise AssertionError("dry-run reached a compute boundary")
                return fail

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = C.ControllerConfig(
                root, len(ELIGIBLE) + 1, dry_run=True,
                planner_context={
                    "portfolio_dispatch_authority": self.dispatch},
                planner_context_sha256="e" * 64,
                production_base_commit="0" * 40,
                instrument_commit="1" * 40,
                hypothesis_portfolio=self.portfolio,
                hypothesis_portfolio_sha256=self.portfolio.sha256)
            planner = BoundPlanner()
            result = C.run_controller(
                config, planner=planner,
                critic=TD.FakeCritic(["accept"] * (len(ELIGIBLE) + 1)),
                screener=NeverCompute(), lease=NeverCompute())
        expected = [row[0] for row in ELIGIBLE]
        self.assertEqual(planner.selected, expected)
        self.assertEqual([row["status"] for row in result["iterations"]],
                         ["dry_run_authorized"] * len(ELIGIBLE))
        self.assertEqual(result["terminal_reason"], "portfolio_exhausted")
        self.assertEqual(set(result["portfolio_validations"]), set(expected))
        self.assertTrue(all(
            row["disposition"] == "dry_run_validated"
            for row in result["portfolio_validations"].values()))
        self.assertEqual(result.get("portfolio_skips", {}), {})
        self.assertEqual(result["portfolio_terminals"], {})

    def test_planner_provider_transient_is_typed_and_retries_after_restart(self):
        """RED: an API reload cannot poison a durable planning operation."""
        fixture = TD.Tests(methodName="runTest")
        transient_type = getattr(C, "PlannerProviderTransient", None)
        self.assertIsInstance(transient_type, type)
        self.assertTrue(issubclass(transient_type, C.PlannerOutputRefusal))

        class StopAfterTransientCheckpoint(BaseException):
            pass

        class FlakyPlanner:
            def __init__(self):
                self.calls = 0
                self.selected = []

            def attest(self):
                return {**C.SOL, "runtime": {
                    "kind": "docker_workspace_bind_only",
                    "docker_path": "/docker", "docker_sha256": "a" * 64,
                    "image_id": "image", "codex_native_sha256": "a" * 64,
                    "code_mode_host_sha256": "a" * 64,
                    "ca_certificate_sha256": "a" * 64,
                    "writable_host_binds": ["/workspace"],
                    "host_network_mode": "docker_bridge"}}

            def plan(self, *, context, workspace):
                self.calls += 1
                binding = context["authoring_assignment"]["portfolio_binding"]
                self.selected.append(binding["hypothesis_id"])
                if self.calls == 1:
                    raise transient_type("provider unavailable during API reload")
                return fixture.portfolio_candidate(binding)

        class NeverCompute:
            def __getattr__(self, _name):
                def fail(*_args, **_kwargs):
                    raise AssertionError("transient dry-run reached compute")
                return fail

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = fixture.portfolio_record(
                hypothesis_id="akh-provider-retry", rank=1, budget=2)
            config = dataclasses.replace(
                fixture.portfolio_config(root, [record]), max_iterations=3)
            planner = FlakyPlanner()
            original_save = C.DurableState.save
            stopped = [False]

            def save_then_stop(store, state, phase):
                original_save(store, state, phase)
                if phase == "planner_transient" and not stopped[0]:
                    stopped[0] = True
                    raise StopAfterTransientCheckpoint(phase)

            with mock.patch.object(C.DurableState, "save", new=save_then_stop), \
                    self.assertRaises(StopAfterTransientCheckpoint):
                C.run_controller(
                    config, planner=planner,
                    critic=TD.FakeCritic(["accept"]),
                    screener=NeverCompute(), lease=NeverCompute())
            result = C.run_controller(
                config, planner=planner, critic=TD.FakeCritic(["accept"]),
                screener=NeverCompute(), lease=NeverCompute())
        self.assertEqual(planner.calls, 2)
        self.assertEqual(planner.selected,
                         ["akh-provider-retry", "akh-provider-retry"])
        self.assertEqual([row["status"] for row in result["iterations"]],
                         ["planner_transient", "dry_run_authorized"])
        self.assertEqual(result["terminal_reason"], "portfolio_exhausted")
        self.assertNotIn("akh-provider-retry", result["portfolio_terminals"])

    def test_repeated_planner_provider_transients_do_not_spend_turn_or_skip(self):
        """Provider/API availability is not an authored or scientific attempt."""
        fixture = TD.Tests(methodName="runTest")
        transient_type = getattr(C, "PlannerProviderTransient", None)
        self.assertIsInstance(transient_type, type)

        class Planner:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return {**C.SOL, "runtime": TD.RUNTIME}

            def plan(self, *, context, workspace):
                self.calls += 1
                if self.calls <= 4:
                    raise transient_type("provider unavailable")
                return fixture.portfolio_candidate(
                    context["authoring_assignment"]["portfolio_binding"])

        class NeverCompute:
            def __getattr__(self, _name):
                def fail(*_args, **_kwargs):
                    raise AssertionError("transient dry-run reached compute")
                return fail

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = fixture.portfolio_record(
                hypothesis_id="akh-provider-endurance", rank=1, budget=1)
            # Exactly one controller/scientific turn is available.  Four API
            # reloads must not consume it or become a bounded authoring skip.
            config = dataclasses.replace(
                fixture.portfolio_config(root, [record]), max_iterations=1)
            planner = Planner()
            result = C.run_controller(
                config, planner=planner,
                critic=TD.FakeCritic(["accept"]),
                screener=NeverCompute(), lease=NeverCompute())
        self.assertEqual(planner.calls, 5)
        self.assertEqual(
            [row["status"] for row in result["iterations"]],
            ["planner_transient"] * 4 + ["dry_run_authorized"])
        self.assertNotIn("akh-provider-endurance",
                         result.get("portfolio_authoring_failures", {}))
        self.assertNotIn("akh-provider-endurance",
                         result.get("portfolio_skips", {}))
        self.assertEqual(result["terminal_reason"], "portfolio_exhausted")

    def test_critic_provider_failure_retries_without_rerunning_planner(self):
        """An API interruption leaves critic_pending as the restart point."""
        fixture = TD.Tests(methodName="runTest")

        class CountingPlanner:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return {**C.SOL, "runtime": TD.RUNTIME}

            def plan(self, *, context, workspace):
                self.calls += 1
                return fixture.portfolio_candidate(
                    context["authoring_assignment"]["portfolio_binding"])

        class FlakyCritic:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return TD.FakeCritic(["accept"]).attest()

            def review(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("critic provider interrupted")
                return C.Critique("accept", "bounded gate")

        class NeverCompute:
            def __getattr__(self, _name):
                def fail(*_args, **_kwargs):
                    raise AssertionError("critic retry reached compute")
                return fail

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dataclasses.replace(
                fixture.portfolio_config(root, [fixture.portfolio_record()]),
                max_iterations=1)
            planner, critic = CountingPlanner(), FlakyCritic()
            with self.assertRaisesRegex(RuntimeError,
                                        "critic provider interrupted"):
                C.run_controller(
                    config, planner=planner, critic=critic,
                    screener=NeverCompute(), lease=NeverCompute())
            result = C.run_controller(
                config, planner=planner, critic=critic,
                screener=NeverCompute(), lease=NeverCompute())
        self.assertEqual((planner.calls, critic.calls), (1, 2))
        self.assertEqual([row["status"] for row in result["iterations"]],
                         ["dry_run_authorized"])
        self.assertEqual(result["portfolio_terminals"], {})

    def test_nonzero_codex_actor_exit_is_provider_transient_not_terminal(self):
        """RED: the concrete planner must map provider exits to retry policy."""
        fixture = TD.Tests(methodName="runTest")
        package = fixture.source_package()
        transient_type = getattr(C, "PlannerProviderTransient", None)
        self.assertIsInstance(transient_type, type)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "operation" / "workspace"
            workspace.mkdir(parents=True)
            wrapper = root / "codex"
            wrapper.write_bytes(b"codex")
            wrapper.chmod(0o700)
            planner = C.CodexPlanner(
                wrapper=wrapper, environment={"PATH": "/usr/bin"},
                reviewed_sources=package)
            assignment = C.AuthoringAssignment(
                "ak-test", "akp-test", "akc-test", "0" * 40, "1" * 40)
            context = {
                "authoring_assignment": assignment.to_dict(),
                "planner_context": {
                    "reviewed_source_package_sha256": package.package_sha256}}
            actor_result = SimpleNamespace(
                returncode=75, stdout="", stderr="provider unavailable")
            with mock.patch.object(
                    C.codex_container_actor, "runtime_identity",
                    return_value=TD.RUNTIME), mock.patch.object(
                    C.codex_container_actor, "run_actor",
                    return_value=actor_result), self.assertRaises(transient_type):
                planner.plan(
                    context=context, workspace=workspace,
                    checkpoint_path=root / "operation" / "actor-result.json")

    def test_typed_stage_refusals_have_exact_accounting_contract(self):
        """RED: known stage outcomes must never become generic ambiguity."""
        base = getattr(C, "GovernedStageRefusal", None)
        self.assertIsInstance(base, type)
        expected = (
            ("SourceApplyRefusal", "source_apply", "authoring_refused"),
            ("CompileRefusal", "compile", "authoring_refused"),
            ("CorrectnessRefusal", "correctness", "correctness_falsified"),
            ("DispatchAttributionRefusal", "dispatch_attribution",
             "attribution_route_falsified"),
        )
        for name, stage, disposition in expected:
            with self.subTest(refusal=name):
                refusal_type = getattr(C, name, None)
                self.assertIsInstance(refusal_type, type)
                self.assertTrue(issubclass(refusal_type, base))
                refusal = refusal_type(
                    "sealed stage outcome", receipt_path="/sealed/receipt.json",
                    receipt_sha256="a" * 64)
                self.assertEqual(refusal.stage, stage)
                self.assertEqual(refusal.disposition, disposition)
                self.assertEqual(refusal.receipt_path, "/sealed/receipt.json")
                self.assertEqual(refusal.receipt_sha256, "a" * 64)
                self.assertIs(refusal.scientific_budget_spent, False)

    def test_static_builder_emits_reusable_typed_source_apply_terminal(self):
        """A known rejected patch is durable authoring evidence, not ambiguity."""
        case = TSR.StaticBuildCacheTests(methodName="runTest")
        fixture = case.fixture()
        try:
            error = C.source_candidate.SourceCandidateError(
                "committed diff derives undeclared symbols")
            with mock.patch.object(
                    TSR.source_candidate, "apply_source_candidate",
                    side_effect=error), self.assertRaises(
                        C.SourceApplyRefusal) as first:
                fixture.builder.build(
                    fixture.candidate, object(), fixture.permit)
            with self.assertRaises(C.SourceApplyRefusal) as reopened:
                case.invoke(
                    fixture, {**fixture.permit, "operation_key": "6" * 64})
            self.assertEqual(first.exception.stage, "source_apply")
            self.assertEqual(first.exception.disposition, "authoring_refused")
            self.assertEqual(
                (reopened.exception.receipt_path,
                 reopened.exception.receipt_sha256),
                (first.exception.receipt_path,
                 first.exception.receipt_sha256))
            receipt = Path(first.exception.receipt_path)
            self.assertTrue(receipt.is_file())
            self.assertEqual(
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
                first.exception.receipt_sha256)
            self.assertEqual(fixture.calls, [])
        finally:
            case.doCleanups()

    def test_static_builder_emits_reusable_typed_compile_terminal(self):
        """A completed failed build is not retried or collapsed into generic error."""
        case = TSR.StaticBuildCacheTests(methodName="runTest")
        fixture = case.fixture()
        original_run_build = fixture.run_build

        def failed_build(*args, **kwargs):
            result = original_run_build(*args, **kwargs)
            result.succeeded = False
            return result

        fixture.run_build = failed_build
        try:
            with self.assertRaises(C.CompileRefusal) as first:
                case.invoke(fixture)
            calls_after_terminal = list(fixture.calls)
            with self.assertRaises(C.CompileRefusal) as reopened:
                case.invoke(
                    fixture, {**fixture.permit, "operation_key": "7" * 64})
            self.assertEqual(first.exception.stage, "compile")
            self.assertEqual(first.exception.disposition, "authoring_refused")
            self.assertEqual(fixture.calls, calls_after_terminal)
            self.assertEqual(
                (reopened.exception.receipt_path,
                 reopened.exception.receipt_sha256),
                (first.exception.receipt_path,
                 first.exception.receipt_sha256))
            receipt = Path(first.exception.receipt_path)
            self.assertEqual(
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
                first.exception.receipt_sha256)
        finally:
            case.doCleanups()

    def test_adapter_emits_reusable_typed_correctness_terminal(self):
        """Parsed target mismatch must be terminal without repeating the GPU call."""
        helper = TA.GpuSourceAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            values = helper.setup(directory)
            adapter, candidate, authorization, lease, _inflight, _current, executors = values
            executors.correctness_summary = "2/3 tests passed"
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(C.CorrectnessRefusal) as first:
                adapter.screen(candidate, authorization, lease)
            calls_after_terminal = list(executors.calls)
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(C.CorrectnessRefusal) as reopened:
                adapter.screen(candidate, authorization, lease)
            self.assertEqual(executors.calls, calls_after_terminal)
            self.assertEqual(len(executors.calls), 1)
            self.assertEqual(
                (reopened.exception.receipt_path,
                 reopened.exception.receipt_sha256),
                (first.exception.receipt_path,
                 first.exception.receipt_sha256))
            receipt = Path(first.exception.receipt_path)
            self.assertEqual(
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
                first.exception.receipt_sha256)

    def test_adapter_emits_reusable_typed_dispatch_terminal(self):
        """A measured forbidden/drifted route terminates without replay."""
        helper = TA.GpuSourceAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            values = helper.setup(directory)
            adapter, candidate, authorization, lease, _inflight, _current, executors = values
            executors.forbidden = True
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(C.DispatchAttributionRefusal) as first:
                adapter.screen(candidate, authorization, lease)
            calls_after_terminal = list(executors.calls)
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(C.DispatchAttributionRefusal) as reopened:
                adapter.screen(candidate, authorization, lease)
            self.assertEqual(executors.calls, calls_after_terminal)
            self.assertEqual(
                [call[:2] for call in executors.calls],
                [("correctness", "candidate"), ("rocprof", "candidate")])
            self.assertEqual(
                (reopened.exception.receipt_path,
                 reopened.exception.receipt_sha256),
                (first.exception.receipt_path,
                 first.exception.receipt_sha256))
            receipt = Path(first.exception.receipt_path)
            self.assertEqual(
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
                first.exception.receipt_sha256)

    def test_cross_arm_invariant_drift_is_reusable_dispatch_terminal(self):
        """A parsed candidate/anchor topology change is a measured falsifier."""
        helper = TA.GpuSourceAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            values = helper.setup(directory)
            adapter, candidate, authorization, lease, _inflight, _current, executors = values
            executors.invariant_changed = True
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(C.DispatchAttributionRefusal) as first:
                adapter.screen(candidate, authorization, lease)
            calls_after_terminal = list(executors.calls)
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(C.DispatchAttributionRefusal) as reopened:
                adapter.screen(candidate, authorization, lease)
            self.assertEqual(executors.calls, calls_after_terminal)
            self.assertEqual(
                [call[:2] for call in executors.calls],
                [("correctness", "candidate"),
                 ("rocprof", "candidate"), ("rocprof", "anchor")])
            self.assertEqual(
                (reopened.exception.receipt_path,
                 reopened.exception.receipt_sha256),
                (first.exception.receipt_path,
                 first.exception.receipt_sha256))
            receipt = Path(first.exception.receipt_path)
            self.assertEqual(
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
                first.exception.receipt_sha256)

    def test_malformed_profile_output_remains_ambiguous_not_falsified(self):
        """Only a parsed route mismatch may become a scientific route terminal."""
        helper = TA.GpuSourceAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            values = helper.setup(directory)
            adapter, candidate, authorization, lease, _inflight, _current, executors = values

            def malformed(invocation):
                executors.calls.append((
                    invocation.kind, invocation.arm, invocation.argv,
                    invocation.environment))
                invocation.stdout_path.write_text("profile complete\n")
                invocation.stderr_path.write_text("")
                invocation.timestamp_csv_path.write_text("malformed,csv\n")
                return executors._capture(invocation, 0)

            adapter.rocprof_executor = malformed
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(E.EvidenceProducerError) as caught:
                adapter.screen(candidate, authorization, lease)
            self.assertNotEqual(
                type(caught.exception).__name__,
                "DispatchAttributionParseRefusal")
            proof = adapter._root(lease["operation_key"]) / "proof"
            self.assertFalse(any(proof.glob("attribution-*/refusal.json")))

    def test_dispatch_refusal_rederives_reason_before_typed_reopen(self):
        """A rewritten self-hashed terminal cannot manufacture route falsification."""
        helper = TA.GpuSourceAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            values = helper.setup(directory)
            adapter, candidate, authorization, lease, _inflight, _current, executors = values
            executors.forbidden = True
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(C.DispatchAttributionRefusal) as first:
                adapter.screen(candidate, authorization, lease)
            calls_after_terminal = list(executors.calls)
            receipt = Path(first.exception.receipt_path)
            forged = __import__("json").loads(receipt.read_text())
            forged["reason"] = "forged but self-consistent route reason"
            forged.pop("receipt_sha256")
            forged["receipt_sha256"] = E.schemas.content_hash(forged)
            receipt.write_text(__import__("json").dumps(
                forged, sort_keys=True) + "\n")
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(E.EvidenceProducerError):
                adapter.screen(candidate, authorization, lease)
            self.assertEqual(executors.calls, calls_after_terminal)

    def test_dedicated_correctness_refusal_rederives_reason_on_reopen(self):
        """The FA second invocation has the same tamper/restart contract."""
        helper = TA.GpuSourceAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = helper.setup(directory)
            adapter, candidate, authorization, lease, _inflight, _current, executors = values
            evidence_plan = plan(root / "multi-correctness-inputs")
            base_argv = list(evidence_plan.correctness_argv)
            invocations = (
                {"invocation_id": "generic", "argv": base_argv,
                 "backend": evidence_plan.correctness_backend,
                 "op": evidence_plan.correctness_op,
                 "case_set": "generic", "expected_cases": 3,
                 "required_cases": [], "environment_overrides": []},
                {"invocation_id": "dedicated", "argv": base_argv,
                 "backend": evidence_plan.correctness_backend,
                 "op": evidence_plan.correctness_op,
                 "case_set": "odd_gqa7_d64_q1_v1", "expected_cases": 3,
                 "required_cases": [], "environment_overrides": [[
                     "AUTOKERNEL_CORRECTNESS_CASE_SET",
                     "odd_gqa7_d64_q1_v1"]]},
            )
            provisional = dataclasses.replace(
                evidence_plan, correctness_invocations=invocations)
            policy_path = provisional.policy.path
            policy_path.write_text(__import__("json").dumps(
                E._policy_payload(provisional), sort_keys=True))
            policy = E.BoundInputFile(
                "execution_policy", policy_path,
                hashlib.sha256(policy_path.read_bytes()).hexdigest())
            evidence_plan = dataclasses.replace(provisional, policy=policy)
            adapter.plan_factory = lambda *_args: evidence_plan
            original_correctness = executors.correctness

            def fail_dedicated(invocation):
                if sum(call[0] == "correctness" for call in executors.calls):
                    executors.correctness_summary = "2/3 tests passed"
                return original_correctness(invocation)

            adapter.correctness_executor = fail_dedicated
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(C.CorrectnessRefusal) as first:
                adapter.screen(candidate, authorization, lease)
            self.assertEqual(
                [call[:2] for call in executors.calls],
                [("correctness", "candidate"),
                 ("correctness", "candidate")])
            calls_after_terminal = list(executors.calls)
            receipt = Path(first.exception.receipt_path)
            forged = __import__("json").loads(receipt.read_text())
            forged["reason"] = "forged but self-consistent correctness reason"
            forged.pop("receipt_sha256")
            forged["receipt_sha256"] = E.schemas.content_hash(forged)
            receipt.write_text(__import__("json").dumps(
                forged, sort_keys=True) + "\n")
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    self.assertRaises(E.EvidenceProducerError):
                adapter.screen(candidate, authorization, lease)
            self.assertEqual(executors.calls, calls_after_terminal)

    def test_controller_accounts_each_typed_stage_refusal_without_ambiguity(self):
        """RED: exercise the public screen boundary and portfolio state."""
        fixture = TD.Tests(methodName="runTest")

        class BoundPlanner:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return {**C.SOL, "runtime": TD.RUNTIME}

            def plan(self, *, context, workspace):
                self.calls += 1
                return fixture.portfolio_candidate(
                    context["authoring_assignment"]["portfolio_binding"])

        expected = (
            ("SourceApplyRefusal", "authoring_refused", False, True),
            ("CompileRefusal", "authoring_refused", False, True),
            ("CorrectnessRefusal", "correctness_falsified", True, False),
            ("DispatchAttributionRefusal", "attribution_route_falsified", False, False),
        )
        for name, disposition, hypothesis_terminal, authoring_failure in expected:
            with self.subTest(refusal=name), tempfile.TemporaryDirectory() as directory:
                refusal_type = getattr(C, name, None)
                self.assertIsInstance(refusal_type, type)
                receipt = E._seal(
                    Path(directory) / "stage-receipt.json",
                    {"schema": "epyc.autokernel.stage_outcome_test.v1",
                     "stage": name})
                receipt_path = str(
                    (Path(directory) / "stage-receipt.json").resolve())
                receipt_sha256 = receipt["body"]["receipt_sha256"]
                refusal = refusal_type(
                    "sealed stage outcome", receipt_path=receipt_path,
                    receipt_sha256=receipt_sha256)

                class RefusingScreen:
                    def __init__(self):
                        self.calls = 0

                    def reconcile(self, inflight):
                        return C.Recovery("safe_to_start")

                    def screen(self, *_args):
                        self.calls += 1
                        raise refusal

                # Exercise the live path with the same complete, sealed
                # deployment authority required at launch.  Replacing only
                # ``dry_run`` on the legacy unit fixture would (correctly)
                # fail before the typed screen boundary under test.
                record = next(
                    row for row in self.portfolio.eligible_hypotheses()
                    if row["hypothesis_id"] ==
                    "akh-v2-q8-quantizer-new-mechanism")
                config = self._real_portfolio_config(
                    Path(directory), [record], iterations=1)
                planner = BoundPlanner()
                critic = TD.FakeCritic(["accept"])
                screener = RefusingScreen()
                original_save = C.DurableState.save
                stopped = [False]

                class StopAfterStageCheckpoint(BaseException):
                    pass

                def save_then_stop(store, state, phase):
                    original_save(store, state, phase)
                    rows = state.get("iterations", [])
                    if (rows and rows[-1].get("status") == disposition
                            and not stopped[0]):
                        stopped[0] = True
                        raise StopAfterStageCheckpoint(phase)

                with mock.patch.object(
                        C.DurableState, "save", new=save_then_stop), \
                        self.assertRaises(StopAfterStageCheckpoint):
                    C.run_controller(
                        config, planner=planner, critic=critic,
                        screener=screener, lease=TD.Lease())
                result = C.run_controller(
                    config, planner=planner, critic=critic,
                    screener=screener, lease=TD.Lease())
                row = result["iterations"][0]
                self.assertEqual(row["status"], disposition)
                self.assertEqual(row["stage_receipt_path"], receipt_path)
                self.assertEqual(row["stage_receipt_sha256"], receipt_sha256)
                self.assertIs(row["scientific_budget_spent"], False)
                self.assertEqual(
                    record["hypothesis_id"] in result["portfolio_terminals"],
                    hypothesis_terminal)
                self.assertEqual(
                    record["hypothesis_id"] in
                    result.get("portfolio_authoring_failures", {}),
                    authoring_failure)
                self.assertEqual((planner.calls, screener.calls), (1, 1))

    def test_controller_recovery_accounts_durable_typed_terminal(self):
        """Crash after producer terminal must not strand inflight forever."""
        fixture = TD.Tests(methodName="runTest")

        class CrashAfterProducerTerminal(BaseException):
            pass

        class Planner:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return {**C.SOL, "runtime": TD.RUNTIME}

            def plan(self, *, context, workspace):
                self.calls += 1
                return fixture.portfolio_candidate(
                    context["authoring_assignment"]["portfolio_binding"])

        for refusal_name, disposition in (
                ("SourceApplyRefusal", "authoring_refused"),
                ("CompileRefusal", "authoring_refused"),
                ("CorrectnessRefusal", "correctness_falsified"),
                ("DispatchAttributionRefusal", "attribution_route_falsified")):
            with self.subTest(refusal=refusal_name), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = root / "durable-stage-terminal.json"
                loaded = E._seal(receipt, {
                    "schema": "epyc.autokernel.stage_outcome_test.v1",
                    "stage": refusal_name})
                refusal_type = getattr(C, refusal_name)
                refusal = refusal_type(
                    "durable producer outcome",
                    receipt_path=str(receipt.resolve()),
                    receipt_sha256=loaded["file_sha256"])

                class Screen:
                    def __init__(self):
                        self.calls = 0
                        self.executor_calls = 0

                    def reconcile(self, _inflight):
                        return C.Recovery("safe_to_start")

                    def screen(self, *_args):
                        self.calls += 1
                        if self.calls == 1:
                            self.executor_calls += 1
                            raise CrashAfterProducerTerminal()
                        raise refusal

                record = next(
                    row for row in self.portfolio.eligible_hypotheses()
                    if row["hypothesis_id"] ==
                    "akh-v2-q8-quantizer-new-mechanism")
                config = self._real_portfolio_config(
                    root, [record], iterations=1)
                planner, screen = Planner(), Screen()
                with self.assertRaises(CrashAfterProducerTerminal):
                    C.run_controller(
                        config, planner=planner,
                        critic=TD.FakeCritic(["accept"]),
                        screener=screen, lease=TD.Lease())
                result = C.run_controller(
                    config, planner=planner,
                    critic=TD.FakeCritic(["accept"]),
                    screener=screen, lease=TD.Lease())
                self.assertEqual(result["iterations"][0]["status"], disposition)
                self.assertEqual((planner.calls, screen.executor_calls), (1, 1))
                self.assertEqual(screen.calls, 2)
                self.assertNotIn("inflight", result)

    def test_bounded_dispatch_falsifiers_advance_to_next_hypothesis(self):
        """Candidate route terminals cannot starve the rest of the portfolio."""
        fixture = TD.Tests(methodName="runTest")
        records_by_id = {
            row["hypothesis_id"]: row
            for row in self.portfolio.eligible_hypotheses()}
        first = records_by_id["akh-v2-q8-quantizer-new-mechanism"]
        second = records_by_id["akh-v2-rms-direct-load-reduction"]
        route_budget = first["decision_policy"]["max_distinct_candidates"]

        class Planner:
            def __init__(self):
                self.selected = []
                self.sequence = 0

            def attest(self):
                return {**C.SOL, "runtime": TD.RUNTIME}

            def plan(self, *, context, workspace):
                self.sequence += 1
                binding = context["authoring_assignment"]["portfolio_binding"]
                self.selected.append(binding["hypothesis_id"])
                return AllStrategyAcceptanceRedGate._bound_candidate(
                    fixture, binding, context, self.sequence)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_receipt = E._seal(root / "route-refusal.json", {
                "schema": "epyc.autokernel.stage_outcome_test.v1",
                "stage": "dispatch_attribution"})
            correctness_receipt = E._seal(root / "correctness-refusal.json", {
                "schema": "epyc.autokernel.stage_outcome_test.v1",
                "stage": "correctness"})

            class Screen:
                def reconcile(self, _inflight):
                    return C.Recovery("safe_to_start")

                def screen(self, candidate, *_args):
                    if candidate.hypothesis_id == first["hypothesis_id"]:
                        raise C.DispatchAttributionRefusal(
                            "reviewed route absent",
                            receipt_path=str((root / "route-refusal.json").resolve()),
                            receipt_sha256=route_receipt["file_sha256"])
                    raise C.CorrectnessRefusal(
                        "targeted correctness mismatch",
                        receipt_path=str(
                            (root / "correctness-refusal.json").resolve()),
                        receipt_sha256=correctness_receipt["file_sha256"])

            planner = Planner()
            result = C.run_controller(
                self._real_portfolio_config(
                    root / "state", [first, second],
                    iterations=route_budget + 2),
                planner=planner,
                critic=TD.FakeCritic(["accept"] * (route_budget + 1)),
                screener=Screen(), lease=TD.Lease())
        self.assertEqual(
            planner.selected,
            [first["hypothesis_id"]] * route_budget +
            [second["hypothesis_id"]])
        self.assertIn(first["hypothesis_id"], result["portfolio_skips"])
        self.assertNotIn(first["hypothesis_id"], result["portfolio_terminals"])
        self.assertEqual(
            result["portfolio_terminals"][second["hypothesis_id"]][
                "disposition"],
            "correctness_falsified")
        self.assertEqual(result["terminal_reason"], "portfolio_exhausted")


class EvidenceStageResumeRedGate(unittest.TestCase):
    """Acceptance for exact-once proof stages after a process/API restart.

    A completed receipt must be revalidated against the unchanged plan and
    reused.  The resumed call may acquire a claim only for the first incomplete
    GPU command.  These tests deliberately exercise the existing public
    producer so recovery cannot be hidden in an uncalled helper.
    """

    @staticmethod
    def _produce(root: Path, current: E.GpuSourceEvidencePlan,
                 executors: FakeExecutors, claims: ClaimFactory):
        return E.produce_gpu_source_evidence(
            output_root=root, plan=current,
            correctness_executor=executors.correctness,
            rocprof_executor=executors.rocprof,
            claim_journal=object(), claim_acquirer=claims,
            claim_verifier=lambda _receipt: True, claim_timeout_s=0)

    def test_resume_after_correctness_does_not_repeat_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            current = plan(base / "inputs")
            output = base / "evidence"
            first_exec, first_claims = FakeExecutors(), ClaimFactory()
            with mock.patch.object(
                    E, "_produce_attribution_arm",
                    side_effect=RuntimeError("crash after correctness")):
                with self.assertRaisesRegex(RuntimeError,
                                            "crash after correctness"):
                    self._produce(output, current, first_exec, first_claims)
            self.assertTrue((output / "correctness/receipt.json").is_file())
            self.assertEqual([row[:2] for row in first_exec.calls],
                             [("correctness", "candidate")])

            resumed_exec, resumed_claims = FakeExecutors(), ClaimFactory()
            bundle = self._produce(
                output, current, resumed_exec, resumed_claims)
            self.assertIsInstance(bundle, E.proofs.GpuSourceProofBundle)
            self.assertEqual([row[:2] for row in resumed_exec.calls], [
                ("rocprof", "candidate"), ("rocprof", "anchor")])
            self.assertEqual(len(resumed_claims.claims), 2)

    def test_resume_after_candidate_attribution_does_not_repeat_prior_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            current = plan(base / "inputs")
            output = base / "evidence"
            first_exec, first_claims = FakeExecutors(), ClaimFactory()
            original = E._produce_attribution_arm

            def crash_before_anchor(root, arm, plan_, executor, **kwargs):
                if arm == "anchor":
                    raise RuntimeError("crash after candidate attribution")
                return original(root, arm, plan_, executor, **kwargs)

            with mock.patch.object(E, "_produce_attribution_arm",
                                   side_effect=crash_before_anchor):
                with self.assertRaisesRegex(
                        RuntimeError, "crash after candidate attribution"):
                    self._produce(output, current, first_exec, first_claims)
            self.assertTrue(
                (output / "attribution-candidate/receipt.json").is_file())
            self.assertEqual([row[:2] for row in first_exec.calls], [
                ("correctness", "candidate"), ("rocprof", "candidate")])

            resumed_exec, resumed_claims = FakeExecutors(), ClaimFactory()
            bundle = self._produce(
                output, current, resumed_exec, resumed_claims)
            self.assertIsInstance(bundle, E.proofs.GpuSourceProofBundle)
            self.assertEqual([row[:2] for row in resumed_exec.calls],
                             [("rocprof", "anchor")])
            self.assertEqual(len(resumed_claims.claims), 1)

    def test_resume_after_anchor_attribution_runs_no_gpu_command_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            current = plan(base / "inputs")
            output = base / "evidence"
            first_exec, first_claims = FakeExecutors(), ClaimFactory()
            with mock.patch.object(
                    E, "_produce_pair",
                    side_effect=RuntimeError("crash after anchor attribution")):
                with self.assertRaisesRegex(
                        RuntimeError, "crash after anchor attribution"):
                    self._produce(output, current, first_exec, first_claims)
            self.assertTrue(
                (output / "attribution-anchor/receipt.json").is_file())
            self.assertEqual([row[:2] for row in first_exec.calls], [
                ("correctness", "candidate"), ("rocprof", "candidate"),
                ("rocprof", "anchor")])

            resumed_exec, resumed_claims = FakeExecutors(), ClaimFactory()
            bundle = self._produce(
                output, current, resumed_exec, resumed_claims)
            self.assertIsInstance(bundle, E.proofs.GpuSourceProofBundle)
            self.assertEqual(resumed_exec.calls, [])
            self.assertEqual(resumed_claims.claims, [])

    def test_pair_receipt_seals_exact_duration_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "evidence"
            current = plan(base / "inputs")
            bundle = self._produce(
                output, current, FakeExecutors(), ClaimFactory())
            pair = E.proofs.load_receipt(
                Path(bundle.attribution["path"]), schema=E.PAIR_SCHEMA)["body"]
            comparison = pair["exact_duration_comparison"]
            self.assertEqual(comparison["candidate_total_duration_ns"], 19)
            self.assertEqual(comparison["anchor_total_duration_ns"], 19)
            self.assertEqual(comparison["relative_improvement_fraction"], 0.0)
            self.assertIs(comparison["all_candidate_routes_present"], True)
            self.assertIs(comparison["all_anchor_routes_present"], True)

    def test_plan_owned_reverse_arm_order_executes_and_receipts_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            current = dataclasses.replace(
                plan(base / "inputs"),
                attribution_arm_order_seed_sha256="f" * 64,
                attribution_arm_order=("anchor", "candidate"))
            policy_raw = __import__("json").dumps(
                E._policy_payload(current), sort_keys=True,
                separators=(",", ":")).encode()
            current.policy.path.write_bytes(policy_raw)
            current = dataclasses.replace(
                current, policy=E.BoundInputFile(
                    current.policy.role, current.policy.path,
                    __import__("hashlib").sha256(policy_raw).hexdigest()))
            executors = FakeExecutors()
            bundle = self._produce(
                base / "evidence", current, executors, ClaimFactory())
            self.assertEqual([row[:2] for row in executors.calls], [
                ("correctness", "candidate"),
                ("rocprof", "anchor"),
                ("rocprof", "candidate"),
            ])
            pair = E.proofs.load_receipt(
                Path(bundle.attribution["path"]), schema=E.PAIR_SCHEMA)["body"]
            self.assertEqual(pair["attribution_arm_order"],
                             ["anchor", "candidate"])
            self.assertEqual(pair["attribution_arm_order_seed_sha256"],
                             "f" * 64)

    def test_s1_s2_attribution_orders_are_exact_reversals(self):
        first_seed, first = F._arm_order_schedule(
            deployment_config_sha256="a" * 64,
            source_manifest_sha256="b" * 64, repetition=1)
        second_seed, second = F._arm_order_schedule(
            deployment_config_sha256="a" * 64,
            source_manifest_sha256="b" * 64, repetition=2)
        self.assertEqual(first_seed, second_seed)
        self.assertEqual(tuple(second.split(",")),
                         tuple(reversed(first.split(","))))


class AdapterStageResumeRedGate(unittest.TestCase):
    """The controller-facing adapter must reopen partial proof receipts.

    Producer-level reuse is insufficient: on a real controller restart,
    ``reconcile`` runs before the producer can inspect those receipts.  Each
    case below crashes after one durable boundary, then calls the public
    adapter again and requires exactly one total invocation of correctness,
    candidate attribution, and anchor attribution.
    """

    def _exercise(self, crash_stage: str) -> None:
        fixture = TA.GpuSourceAdapterTests(methodName="runTest")
        with tempfile.TemporaryDirectory() as directory:
            values = fixture.setup(directory)
            (adapter, candidate, authorization, lease, inflight, current,
             executors) = values
            original_arm = E._produce_attribution_arm

            if crash_stage == "correctness":
                patcher = mock.patch.object(
                    E, "_produce_attribution_arm",
                    side_effect=RuntimeError("crash after correctness"))
            elif crash_stage == "candidate_attribution":
                def crash_before_anchor(root, arm, plan_, executor, **kwargs):
                    if arm == "anchor":
                        raise RuntimeError("crash after candidate attribution")
                    return original_arm(root, arm, plan_, executor, **kwargs)
                patcher = mock.patch.object(
                    E, "_produce_attribution_arm",
                    side_effect=crash_before_anchor)
            elif crash_stage == "anchor_attribution":
                patcher = mock.patch.object(
                    E, "_produce_pair",
                    side_effect=RuntimeError("crash after anchor attribution"))
            else:  # pragma: no cover - test authoring error
                self.fail(f"unknown crash stage {crash_stage}")

            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate), \
                    patcher, self.assertRaisesRegex(RuntimeError, "crash after"):
                adapter.screen(candidate, authorization, lease)

            # This is the decisive adapter seam.  "ambiguous" here prevents
            # the already resumable producer from ever being re-entered.
            self.assertEqual(adapter.reconcile(inflight).status,
                             "safe_to_start")
            with mock.patch.object(C, "GpuSourceScreener", TA.FakeDelegate):
                resumed = adapter.screen(candidate, authorization, lease)
            self.assertEqual(resumed.result_sha256, current.result_sha256)
            self.assertEqual([row[:2] for row in executors.calls], [
                ("correctness", "candidate"),
                ("rocprof", "candidate"),
                ("rocprof", "anchor"),
            ])

    def test_adapter_resume_after_correctness_is_exactly_once(self):
        self._exercise("correctness")

    def test_adapter_resume_after_candidate_attribution_is_exactly_once(self):
        self._exercise("candidate_attribution")

    def test_adapter_resume_after_anchor_attribution_is_exactly_once(self):
        self._exercise("anchor_attribution")


if __name__ == "__main__":
    unittest.main()
