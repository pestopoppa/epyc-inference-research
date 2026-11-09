#!/usr/bin/env python3
"""CPU-only mutation barrier for the Hawkeye-derived evaluator substrate."""
from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.kernel_rnd.autokernel.evaluator import hawkeye_measurement as H
from scripts.kernel_rnd.autokernel.execution.physical_bounds import PhysicalEnvelope


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def manifest():
    return {
        "variants": 1,
        "tensors": [
            {"name": "x", "file": "x.bin", "dtype": "torch.float32",
             "shape": [2, 2], "numel": 4, "role": "input"},
            {"name": "w", "file": "w.bin", "dtype": "float32",
             "shape": [2, 2], "numel": 4, "role": "weight"},
        ],
        "outputs": [{"name": "y", "dtype": "float32", "numel": 4}],
    }


def timing():
    return {
        "rate_samples_per_s": 1000.0,
        "rate_samples_per_s_wall": 900.0,
        "ms_per_iter_gpu": 1.0,
        "ms_per_iter_wall": 1.1,
        "gpu_ms_min": 0.9,
        "gpu_ms_max": 1.1,
        "gpu_ms_std": 0.01,
        "cv_gpu": 0.01,
        "wall_ms_min": 1.0,
        "wall_ms_max": 1.2,
        "wall_ms_std": 0.02,
        "cv_wall": 0.02,
        "timing_redos": 0,
        "cv_gate_pass": True,
        "cv_gate_max": 0.05,
    }


def envelope():
    return PhysicalEnvelope(
        shape_id="gemm-m16n16k16", delivered_unit="kernel",
        flops_per_unit=2e6, bytes_per_unit=1e6,
        peak_compute_flops_s=20e12, peak_memory_bytes_s=1e12,
        measurement_frame_sha256=SHA_A,
        work_derivation_ref="operator-work/v1",
        hardware_peak_ref="operator-gfx90a-roofline/v1")


class SchemaTests(unittest.TestCase):
    def test_manifest_normalizes_dtype(self):
        self.assertEqual(H.validate_hawkeye_tensor_manifest(manifest())
                         ["tensors"][0]["dtype"], "float32")

    def test_manifest_rejects_parent_path(self):
        row = manifest()
        row["tensors"][0]["file"] = "../golden.bin"
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "relative leaf"):
            H.validate_hawkeye_tensor_manifest(row)

    def test_manifest_rejects_shape_laundering(self):
        row = manifest()
        row["tensors"][0]["numel"] = 3
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "shape product"):
            H.validate_hawkeye_tensor_manifest(row)

    def test_manifest_requires_perturbable_input(self):
        row = manifest()
        for tensor in row["tensors"]:
            tensor["role"] = "weight"
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "perturbable"):
            H.validate_hawkeye_tensor_manifest(row)

    def test_timing_schema_accepts_source_shape(self):
        self.assertTrue(H.validate_hawkeye_timing_result(timing())["cv_gate_pass"])

    def test_timing_rejects_claimed_cv_pass(self):
        row = timing()
        row["cv_gpu"] = 0.1
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "disagrees"):
            H.validate_hawkeye_timing_result(row)

    def test_timing_rejects_surplus_self_report(self):
        row = timing()
        row["claimed_speedup"] = 999
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "fields differ"):
            H.validate_hawkeye_timing_result(row)

    def test_only_two_schema_files_are_adopted(self):
        schema_dir = Path(H.__file__).with_name("schemas")
        adopted = sorted(path.name for path in schema_dir.glob("*.schema.json"))
        self.assertEqual(adopted, sorted(H.HAWKEYE_ADOPTED_SCHEMAS))
        self.assertNotIn("spec.json", " ".join(adopted))
        for name in adopted:
            json.loads((schema_dir / name).read_text())


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        self.seed = H.EvaluatorRunSeed(bytes(range(32)))

    def test_seed_repr_is_redacted(self):
        self.assertEqual(repr(self.seed), "EvaluatorRunSeed(<redacted>)")

    def test_short_seed_is_refused(self):
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "32 bytes"):
            H.EvaluatorRunSeed(b"weak")

    def test_each_iteration_is_distinct(self):
        source = [1.0, 2.0, 3.0, 4.0]
        rows = [H.perturb_values(source, dtype="float32", iteration=i,
                                 tensor_name="x", run_seed=self.seed)
                for i in range(4)]
        self.assertEqual(len(set(rows)), 4)
        self.assertTrue(all(row != tuple(source) for row in rows))

    def test_order_insensitive_sum_changes(self):
        source = [1, 2, 3, 4]
        changed = H.perturb_values(source, dtype="int32", iteration=0,
                                   tensor_name="x", run_seed=self.seed)
        self.assertNotEqual(sum(source), sum(changed))
        self.assertNotEqual(sorted(source), sorted(changed))

    def test_receipt_commits_without_secret(self):
        source = [1.0, 2.0]
        changed = H.perturb_values(source, dtype="float32", iteration=2,
                                   tensor_name="x", run_seed=self.seed)
        receipt = H.perturbation_receipt(
            source, changed, dtype="float32", iteration=2,
            tensor_name="x", run_seed=self.seed)
        self.assertNotIn(self.seed.secret.hex(), repr(receipt))
        self.assertNotEqual(receipt.pristine_sha256, receipt.perturbed_sha256)


class IsolationTests(unittest.TestCase):
    def plan(self):
        return H.CandidateIsolationPlan(
            Path("/sandbox/candidate"), Path("/sandbox/inputs"),
            Path("/oracle/pristine"), Path("/oracle/golden"))

    def receipt(self, plan, *, pid, start, cgroup, candidate=False):
        row = {
            "schema": H.candidate_sandbox.RECEIPT_SCHEMA,
            "sandbox_id": H.candidate_sandbox.SANDBOX_ID,
            "pid": pid,
            "process_start_ticks": start,
            "policy_sha256": ("d" if candidate else "e") * 64,
            "cgroup_path": cgroup,
            "read_allowlist_enforced": True,
        }
        row.update(plan.candidate_policy_projection() if candidate
                   else plan.oracle_policy_projection())
        row["schema"] = H.candidate_sandbox.RECEIPT_SCHEMA
        row["pid"] = pid
        row["process_start_ticks"] = start
        row["policy_sha256"] = ("d" if candidate else "e") * 64
        row["cgroup_path"] = cgroup
        return row

    def teardown(self, cgroup):
        return {"cgroup_path": cgroup, "verified_empty": True,
                "removed": True, "descendants_killed": False}

    def test_policy_projection_omits_private_paths(self):
        plan = H.CandidateIsolationPlan(
            Path("/sandbox/candidate"), Path("/sandbox/inputs"),
            Path("/oracle/pristine"), Path("/oracle/golden"))
        rendered = json.dumps(plan.candidate_policy_projection())
        self.assertNotIn("pristine", rendered)
        self.assertNotIn("golden", rendered)
        self.assertEqual(plan.sandbox_profile, H.candidate_sandbox.EVALUATOR_PROFILE)

    def test_overlapping_oracle_root_is_refused(self):
        with self.assertRaisesRegex(H.IsolationError, "overlap"):
            H.CandidateIsolationPlan(
                Path("/sandbox"), Path("/sandbox/inputs"),
                Path("/sandbox/pristine"), Path("/oracle/golden"))

    def test_shared_process_is_refused(self):
        plan = self.plan()
        with self.assertRaisesRegex(H.IsolationError, "distinct processes"):
            H.bind_process_isolation(
                plan,
                oracle_sandbox_receipt=self.receipt(
                    plan, pid=10, start=1, cgroup="/cg/o"),
                candidate_sandbox_receipt=self.receipt(
                    plan, pid=10, start=2, cgroup="/cg/c", candidate=True),
                oracle_teardown_receipt=self.teardown("/cg/o"),
                candidate_teardown_receipt=self.teardown("/cg/c"))

    def test_shared_cgroup_activation_is_refused(self):
        plan = self.plan()
        with self.assertRaisesRegex(H.IsolationError, "share a cgroup"):
            H.bind_process_isolation(
                plan,
                oracle_sandbox_receipt=self.receipt(
                    plan, pid=10, start=1, cgroup="/cg/same"),
                candidate_sandbox_receipt=self.receipt(
                    plan, pid=11, start=2, cgroup="/cg/same", candidate=True),
                oracle_teardown_receipt=self.teardown("/cg/same"),
                candidate_teardown_receipt=self.teardown("/cg/same"))

    def test_candidate_receipt_allowlist_drift_is_refused(self):
        plan = self.plan()
        candidate = self.receipt(
            plan, pid=11, start=2, cgroup="/cg/c", candidate=True)
        candidate["readable_roots"].append("/oracle/golden")
        with self.assertRaisesRegex(H.IsolationError, "readable_roots"):
            H.bind_process_isolation(
                plan,
                oracle_sandbox_receipt=self.receipt(
                    plan, pid=10, start=1, cgroup="/cg/o"),
                candidate_sandbox_receipt=candidate,
                oracle_teardown_receipt=self.teardown("/cg/o"),
                candidate_teardown_receipt=self.teardown("/cg/c"))

    def test_only_runtime_receipt_is_serializable(self):
        plan = self.plan()
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "not an exported"):
            H.serialize_carrier(plan)
        receipt = H.bind_process_isolation(
            plan,
            oracle_sandbox_receipt=self.receipt(
                plan, pid=10, start=1, cgroup="/cg/o"),
            candidate_sandbox_receipt=self.receipt(
                plan, pid=11, start=2, cgroup="/cg/c", candidate=True),
            oracle_teardown_receipt=self.teardown("/cg/o"),
            candidate_teardown_receipt=self.teardown("/cg/c"))
        payload = H.serialize_carrier(receipt)
        self.assertEqual(payload["schema"], "epyc.autokernel.process_isolation.v1")
        self.assertNotIn("golden", json.dumps(payload))
        self.assertRegex(payload["carrier_sha256"], r"^[0-9a-f]{64}$")


class SourceIntegrityTests(unittest.TestCase):
    def test_post_run_snapshot_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel.cpp"
            path.write_text("void kernel() {}\n")
            before = H.snapshot_compiled_sources(tmp, ["kernel.cpp"])
            after = H.snapshot_compiled_sources(tmp, ["kernel.cpp"])
            H.verify_post_run_source_integrity(before, after)

    def test_post_run_mutation_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel.cpp"
            path.write_text("void kernel() {}\n")
            before = H.snapshot_compiled_sources(tmp, ["kernel.cpp"])
            path.write_text("void kernel() { cheat(); }\n")
            after = H.snapshot_compiled_sources(tmp, ["kernel.cpp"])
            with self.assertRaisesRegex(H.SourceIntegrityError, "changed"):
                H.verify_post_run_source_integrity(before, after)

    def test_symlink_source_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "real.cpp").write_text("x")
            (root / "alias.cpp").symlink_to(root / "real.cpp")
            with self.assertRaisesRegex(H.SourceIntegrityError, "regular file"):
                H.snapshot_compiled_sources(root, ["alias.cpp"])


class PrecisionTests(unittest.TestCase):
    def policy(self, **updates):
        values = dict(
            operator_id="ggml_mul_mat", template_id="q5_0_m32_n1_k256",
            input_dtype="float32", required_output_dtype="float32",
            required_accumulator_dtype="float32", reduce_dimension=16,
            structural_evidence_sha256=SHA_A)
        values.update(updates)
        return H.PrecisionEquivalencePolicy(**values)

    def test_multiplier_cannot_loosen_operator_bound(self):
        with self.assertRaisesRegex(H.PrecisionEquivalenceError, "exactly 1.0"):
            self.policy(bound_multiplier=2.0)

    def test_policy_binds_operator_template_and_structural_proof(self):
        left = self.policy()
        right = self.policy(template_id="q5_0_m64_n1_k256")
        self.assertNotEqual(left.policy_sha256, right.policy_sha256)

    def test_bound_is_sqrt_dimension_not_linear(self):
        policy = self.policy()
        self.assertAlmostEqual(policy.normalized_error_bound,
                               math.sqrt(16) * 2**-23)
        self.assertNotEqual(policy.normalized_error_bound, 16 * 2**-23)

    def test_small_error_passes_against_float64(self):
        verdict = H.evaluate_precision_equivalence(
            [1.0, 2.0], [1.0 + 1e-7, 2.0 - 1e-7], policy=self.policy(),
            observed_output_dtype="float32", observed_accumulator_dtype="float32")
        self.assertTrue(verdict.correct)
        self.assertEqual(verdict.reference_dtype, "float64")

    def test_linear_scale_error_fails(self):
        verdict = H.evaluate_precision_equivalence(
            [1.0, 2.0], [1.0 + 1e-4, 2.0], policy=self.policy(),
            observed_output_dtype="float32", observed_accumulator_dtype="float32")
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.reason, "exceeds_sqrt_d_epsilon_bound")

    def test_dtype_downgrade_fails_before_tolerance(self):
        verdict = H.evaluate_precision_equivalence(
            [1.0], [1.0], policy=self.policy(), observed_output_dtype="float16",
            observed_accumulator_dtype="float32")
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.stage, "structural_precision")
        self.assertIsNone(verdict.normalized_rms_error)

    def test_accumulator_downgrade_fails_before_tolerance(self):
        verdict = H.evaluate_precision_equivalence(
            [1.0], [1.0], policy=self.policy(), observed_output_dtype="float32",
            observed_accumulator_dtype="float16")
        self.assertEqual(verdict.reason, "dtype_or_accumulator_mismatch")

    def test_non_float64_reference_is_refused(self):
        with self.assertRaisesRegex(H.PrecisionEquivalenceError, "float64"):
            H.evaluate_precision_equivalence(
                [1.0], [1.0], policy=self.policy(), reference_dtype="float32",
                observed_output_dtype="float32", observed_accumulator_dtype="float32")


class LibrarySubstitutionTests(unittest.TestCase):
    def generated(self):
        return [H.DispatchEvent("my_kernel", "candidate", "candidate_generated")]

    def test_generated_dispatch_passes(self):
        verdict = H.evaluate_library_substitution(
            "def run(x): return tl.load(x)", language="triton", action_tags=[],
            dispatch_events=self.generated())
        self.assertEqual(verdict.outcome, "PASS")

    def test_torch_alias_is_caught(self):
        verdict = H.evaluate_library_substitution(
            "import torch as t\ndef run(a,b): return t.mm(a,b)\n",
            language="python", action_tags=[], dispatch_events=self.generated())
        self.assertEqual(verdict.outcome, "FAIL")
        self.assertIn("torch.mm", verdict.forbidden_source_calls)

    def test_native_vendor_call_is_caught_but_comment_is_not(self):
        bad = H.evaluate_library_substitution(
            "void run(){ rocblas_sgemm(); }", language="hip", action_tags=[],
            dispatch_events=self.generated())
        good = H.evaluate_library_substitution(
            "// rocblas_sgemm()\nvoid run(){ my_kernel(); }", language="hip",
            action_tags=[], dispatch_events=self.generated())
        self.assertEqual(bad.outcome, "FAIL")
        self.assertEqual(good.outcome, "PASS")

    def test_published_action_tags_are_caught(self):
        for tag in ("CALL_LIBRARY", "SOTA_TRITON_API"):
            with self.subTest(tag=tag):
                verdict = H.evaluate_library_substitution(
                    "def run(): pass", language="python", action_tags=[tag],
                    dispatch_events=self.generated())
                self.assertEqual(verdict.outcome, "FAIL")

    def test_vendor_dispatch_is_caught(self):
        verdict = H.evaluate_library_substitution(
            "void run(){}", language="hip", action_tags=[], dispatch_events=[
                H.DispatchEvent("rocblas_gemm", "rocblas", "dynamic_library")])
        self.assertEqual(verdict.outcome, "FAIL")

    def test_absent_trace_never_passes(self):
        verdict = H.evaluate_library_substitution(
            "void run(){}", language="hip", action_tags=[], dispatch_events=[])
        self.assertEqual(verdict.outcome, "COULD_NOT_CHECK")


class RuntimeClosureTests(unittest.TestCase):
    def row(self, name, digest=SHA_A):
        return H.RuntimeLibraryIdentity(name, f"/runtime/{name}.so", digest, "ggml")

    def test_exact_full_join_passes(self):
        rows = [self.row("base"), self.row("cpu"), self.row("hip")]
        self.assertEqual(H.evaluate_runtime_library_closure(rows, list(rows)).outcome,
                         "PASS")

    def test_missing_base_library_fails(self):
        expected = [self.row("base"), self.row("cpu"), self.row("hip")]
        verdict = H.evaluate_runtime_library_closure(expected, expected[1:])
        self.assertEqual(verdict.outcome, "FAIL")
        self.assertEqual(verdict.missing, ("base",))

    def test_swapped_library_digest_fails(self):
        expected = [self.row("base"), self.row("hip")]
        observed = [self.row("base"), self.row("hip", SHA_B)]
        self.assertEqual(H.evaluate_runtime_library_closure(
            expected, observed).identity_mismatches, ("hip",))


class RooflineTests(unittest.TestCase):
    def test_cap_is_derived_from_workload_envelope(self):
        cap = H.derive_roofline_speedup_cap(
            gpu_part="gfx90a", envelope=envelope(),
            baseline_throughput_units_s=100_000)
        # Envelope is memory limited: 1e12 / 1e6 = 1e6 kernels/s.
        self.assertEqual(cap.roofline_throughput_ceiling_units_s, 1_000_000)
        self.assertEqual(cap.max_speedup, 10.0)
        self.assertEqual(H.check_implausible_speedup(10.1, cap), "NULL_IMPLAUSIBLE")
        self.assertEqual(H.check_implausible_speedup(9.9, cap), "ADMISSIBLE")

    def test_unknown_part_refuses_instead_of_estimating(self):
        with self.assertRaisesRegex(Exception, "unknown GPU part"):
            H.derive_roofline_speedup_cap(
                gpu_part="gfx9999", envelope=envelope(),
                baseline_throughput_units_s=100_000)

    def test_impossible_baseline_is_refused(self):
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "already exceeds"):
            H.derive_roofline_speedup_cap(
                gpu_part="gfx90a", envelope=envelope(),
                baseline_throughput_units_s=2_000_000)

    def test_tcc_and_l2_hit_rate_are_forbidden(self):
        for metric in ("tcc hit rate", "l2-hit-rate"):
            with self.subTest(metric=metric), self.assertRaisesRegex(
                    H.HawkeyeMeasurementError, "forbidden"):
                H.validate_reward_metric(metric)

    def test_pc_sampling_is_not_an_input_class(self):
        for input_class in ("pc_sampling", "gpu_busy_percent"):
            with self.subTest(input_class=input_class), self.assertRaisesRegex(
                    H.HawkeyeMeasurementError, "unavailable"):
                H.validate_measurement_input_class(input_class)


class GhostReplayTests(unittest.TestCase):
    def test_native_is_typed_not_applicable_not_relabelled(self):
        result = H.run_ghost_replay(
            candidate_language="hip", task_name="x", spec={},
            candidate_fn=None, device="cpu")
        self.assertEqual(result.outcome, "NOT_APPLICABLE")
        self.assertEqual(result.applicability, "not_applicable_native")

    def test_exact_validated_triton_helper_is_lifted(self):
        helper = H._load_exact_ghost_replay_helper()
        self.assertEqual(helper.__name__, "ghost_replay")

    def isolation(self, *, candidate_pid, start, cgroup):
        return H.ProcessIsolationReceipt(
            oracle_pid=100, candidate_pid=candidate_pid,
            oracle_process_start_ticks=1,
            candidate_process_start_ticks=start,
            oracle_sandbox_receipt_sha256="1" * 64,
            candidate_sandbox_receipt_sha256="2" * 64,
            oracle_teardown_receipt_sha256="a" * 64,
            candidate_teardown_receipt_sha256="b" * 64,
            oracle_policy_sha256="3" * 64,
            candidate_policy_sha256="4" * 64,
            oracle_cgroup_path="/cg/oracle",
            candidate_cgroup_path=cgroup,
            candidate_read_allowlist_sha256="5" * 64)

    def closure(self, names):
        rows = [H.RuntimeLibraryIdentity(
            name, f"/runtime/{name}.so", SHA_A, "ggml") for name in names]
        return H.evaluate_runtime_library_closure(rows, rows)

    def perturbation(self):
        seed = H.EvaluatorRunSeed(bytes(range(32)))
        source = [1.0, 2.0]
        changed = H.perturb_values(
            source, dtype="float32", iteration=3, tensor_name="x", run_seed=seed)
        return H.perturbation_receipt(
            source, changed, dtype="float32", iteration=3,
            tensor_name="x", run_seed=seed)

    def bundle(self, *, same_output=False, same_process=False):
        perturbation = self.perturbation()
        real_closure = self.closure(("base", "cpu", "hip"))
        noop_closure = self.closure(("base", "cpu", "hip", "interposer"))
        plan = H.NativeGhostReplayPlan(
            interposer_path="/trusted/libautokernel_hip_noop.so",
            interposer_sha256=SHA_A,
            interposer_source_sha256=H.NATIVE_GHOST_INTERPOSER_SOURCE_SHA256,
            intercepted_symbols=H.NATIVE_GHOST_INTERCEPT_SYMBOLS,
            candidate_build_sha256="6" * 64,
            candidate_source_snapshot_sha256="7" * 64,
            perturbation_carrier_sha256=H.serialize_carrier(
                perturbation)["carrier_sha256"],
            initialized_output_sha256="8" * 64,
            real_runtime_closure_sha256=real_closure.observed_closure_sha256,
            noop_runtime_closure_sha256=noop_closure.observed_closure_sha256)
        real = H.NativeReplayWitness(
            mode="real", output_sha256="9" * 64,
            initialized_output_sha256=plan.initialized_output_sha256,
            candidate_build_sha256=plan.candidate_build_sha256,
            candidate_source_snapshot_sha256=plan.candidate_source_snapshot_sha256,
            isolation=self.isolation(candidate_pid=101, start=2,
                                      cgroup="/cg/real"),
            runtime_closure=real_closure, loaded_interposer_sha256=None)
        noop = H.NativeReplayWitness(
            mode="noop", output_sha256=("9" if same_output else "0") * 64,
            initialized_output_sha256=plan.initialized_output_sha256,
            candidate_build_sha256=plan.candidate_build_sha256,
            candidate_source_snapshot_sha256=plan.candidate_source_snapshot_sha256,
            isolation=self.isolation(
                candidate_pid=101 if same_process else 102,
                start=2 if same_process else 3,
                cgroup="/cg/real" if same_process else "/cg/noop"),
            runtime_closure=noop_closure,
            loaded_interposer_sha256=plan.interposer_sha256)
        event = H.NATIVE_GHOST_EVENT_STRUCT.pack(
            H.NATIVE_GHOST_EVENT_MAGIC, 1, 0)
        return plan, perturbation, real, noop, event

    def test_native_noop_divergence_passes(self):
        plan, perturbation, real, noop, event = self.bundle()
        verdict = H.evaluate_native_ghost_replay(
            plan=plan, perturbation=perturbation, real=real, noop=noop,
            interposer_event_bytes=event)
        self.assertEqual(verdict.outcome, "PASS")
        self.assertEqual(verdict.launch_count, 1)

    def test_native_noop_identity_fails(self):
        plan, perturbation, real, noop, event = self.bundle(same_output=True)
        verdict = H.evaluate_native_ghost_replay(
            plan=plan, perturbation=perturbation, real=real, noop=noop,
            interposer_event_bytes=event)
        self.assertEqual(verdict.outcome, "FAIL")

    def test_native_zero_launch_never_passes(self):
        plan, perturbation, real, noop, _event = self.bundle()
        verdict = H.evaluate_native_ghost_replay(
            plan=plan, perturbation=perturbation, real=real, noop=noop,
            interposer_event_bytes=b"")
        self.assertEqual(verdict.outcome, "COULD_NOT_CHECK")

    def test_native_runtime_closure_mismatch_fails(self):
        plan, perturbation, real, noop, event = self.bundle()
        noop = dataclasses.replace(
            noop, runtime_closure=self.closure(("base", "cpu", "hip", "wrong")))
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "differs from plan"):
            H.evaluate_native_ghost_replay(
                plan=plan, perturbation=perturbation, real=real, noop=noop,
                interposer_event_bytes=event)

    def test_native_different_initial_output_is_refused(self):
        plan, perturbation, real, noop, event = self.bundle()
        noop = dataclasses.replace(noop, initialized_output_sha256="f" * 64)
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "initialization"):
            H.evaluate_native_ghost_replay(
                plan=plan, perturbation=perturbation, real=real, noop=noop,
                interposer_event_bytes=event)

    def test_native_different_build_is_refused(self):
        plan, perturbation, real, noop, event = self.bundle()
        noop = dataclasses.replace(noop, candidate_build_sha256="f" * 64)
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "candidate identity"):
            H.evaluate_native_ghost_replay(
                plan=plan, perturbation=perturbation, real=real, noop=noop,
                interposer_event_bytes=event)

    def test_native_reused_process_is_refused(self):
        plan, perturbation, real, noop, event = self.bundle(same_process=True)
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "distinct sandbox"):
            H.evaluate_native_ghost_replay(
                plan=plan, perturbation=perturbation, real=real, noop=noop,
                interposer_event_bytes=event)

    def test_native_partial_event_log_is_refused(self):
        plan, perturbation, real, noop, _event = self.bundle()
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "partial"):
            H.evaluate_native_ghost_replay(
                plan=plan, perturbation=perturbation, real=real, noop=noop,
                interposer_event_bytes=b"short")

    def test_partial_intercept_surface_is_refused(self):
        plan, _perturbation, _real, _noop, _event = self.bundle()
        with self.assertRaisesRegex(H.HawkeyeMeasurementError, "complete"):
            dataclasses.replace(plan, intercepted_symbols=("hipLaunchKernel",))


if __name__ == "__main__":
    unittest.main()
