"""Hardware-free acceptance gate for every currently eligible GPU strategy.

The non-FA assertions describe seams that are already usable.  The two FA
assertions are intentionally red until the paired-head bulk/tail dispatch and
odd-GQA7 correctness contracts are sealed.  This file is test-only audit work;
it does not widen runtime authority.
"""
from __future__ import annotations

import dataclasses
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
        self.assertEqual(dedicated["expected_cases"], len(required))
        self.assertEqual(dedicated["required_cases"], required)
        self.assertEqual(dedicated["case_set"], "odd_gqa7_d64_q1_v1")
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
                        "relative_improvement_fraction": -0.01}}},
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


if __name__ == "__main__":
    unittest.main()
