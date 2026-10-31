from __future__ import annotations

from array import array
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from . import hip_decision_grade as H


class HipDecisionGradeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_suite_is_replayable_hostile_and_not_generated_until_called(self):
        first, expected_a = H._create_suite(self.work, "a" * 64)
        self.assertEqual(len(first["cases"]), 24)
        self.assertEqual({row["distribution"] for row in first["cases"]}, {
            "baseline", "alternating", "sparse_outlier", "cancellation"})
        self.assertTrue(first["suite_seed_generated_after_candidate_seal"])
        self.assertEqual(first["timing_blocks"], 20)
        self.assertEqual(first["repetitions_per_arm"], 30_000)
        self.assertEqual(len(expected_a), 24)
        replay_work = self.root / "replay"
        replay_work.mkdir()
        replay, expected_b = H._create_suite(replay_work, "a" * 64)
        self.assertEqual(
            [row["input_sha256"] for row in first["cases"]],
            [row["input_sha256"] for row in replay["cases"]])
        self.assertEqual(expected_a, expected_b)

    def test_host_double_silu_is_stable_at_extremes(self):
        self.assertEqual(H._silu_host_double(1000.0), 1000.0)
        self.assertEqual(H._silu_host_double(-1000.0), -0.0)
        self.assertAlmostEqual(H._silu_host_double(0.0), 0.0)

    def test_static_c6_scan_accepts_candidate_and_detects_environment_probe(self):
        source = self.root / "candidate.hip"
        source.write_text("void forward() { int value = 1; }\n")
        self.assertTrue(H._scan_candidate(source)["clean"])
        source.write_text("void forward() { auto p = getenv(\"MODE\"); }\n")
        scan = H._scan_candidate(source)
        self.assertFalse(scan["clean"])
        self.assertTrue(scan["environment_probe_findings"])

    def test_correctness_reducer_requires_both_poisons_bitwise_equal(self):
        inputs = self.work / "inputs"
        outputs = self.work / "outputs"
        inputs.mkdir()
        outputs.mkdir()
        values = [0.0, 1.0, -1.0]
        H._write_f32(inputs / "x.f32", values)
        oracle = tuple(H._silu_host_double(v) for v in H._read_f32(inputs / "x.f32"))
        H._write_f32(outputs / "x-a.f32", oracle)
        H._write_f32(outputs / "x-b.f32", oracle)
        specification = {"cases": [{"case_id": "x"}]}
        child = {"cases": [{
            "case_id": "x", "input_file_unchanged": True,
            "device_input_unchanged": True,
            "output_a": "outputs/x-a.f32", "output_b": "outputs/x-b.f32",
        }]}
        reduced = H._reduce_correctness(
            work=self.work, specification=specification,
            expected={"x": oracle}, child=child)
        self.assertTrue(reduced["all_passed"])
        H._write_f32(outputs / "x-b.f32", [0.0, 0.0, 0.0])
        reduced = H._reduce_correctness(
            work=self.work, specification=specification,
            expected={"x": oracle}, child=child)
        self.assertFalse(reduced["all_passed"])

    def test_timing_reducer_uses_anytime_valid_bundle_construction(self):
        child = {
            "provider": {"provider_id": "torch_rocm_compile"},
            "repetitions_per_arm": 30_000,
            "blocks": [{
                "block_index": index,
                "order": "candidate_first" if index % 2 else "anchor_first",
                "candidate_ns": 80.0, "anchor_ns": 100.0,
                "candidate_measured_duration_ns": 300_000_000.0,
                "anchor_measured_duration_ns": 300_000_000.0,
            } for index in range(20)],
        }
        result = H._reduce_timing(child)
        self.assertTrue(result["candidate_beats_exact_provider"])
        self.assertEqual(result["e_process"]["construction_id"], H.CONSTRUCTION_ID)
        self.assertEqual(result["e_process"]["threshold"], 20.0)

    def test_sub_floor_timing_cannot_be_rankable(self):
        child = {
            "provider": {"provider_id": "torch_rocm_compile"},
            "repetitions_per_arm": 200,
            "blocks": [{
                "block_index": index, "order": "anchor_first",
                "candidate_ns": 80.0, "anchor_ns": 100.0,
                "candidate_measured_duration_ns": 2_000_000.0,
                "anchor_measured_duration_ns": 2_000_000.0,
            } for index in range(20)],
        }
        result = H._reduce_timing(child)
        self.assertFalse(result["candidate_beats_exact_provider"])
        self.assertEqual(
            result["ranked_duration_admission"]["all_arms_passed"], False)
        self.assertEqual(
            result["ranked_duration_admission"]["checks"][0]["outcome"],
            "COULD_NOT_CHECK")

    def _counter_pair(self, *, anchor_value="10", candidate_value="0"):
        expectation = {
            "schema": H.COUNTER_EXPECTATION_SCHEMA,
            "counter_name": "SQ_LDS_BANK_CONFLICT",
            "expected_direction": "decrease", "unit": "cycles",
            "candidate_kernel_name": "candidate_kernel",
            "anchor_kernel_name": "anchor_kernel",
            "normalization": "sum_exact_dispatches",
        }
        workload_sha = "a" * 64
        candidate_source_sha = "b" * 64
        anchor_identity_sha = "c" * 64
        campaign_id = "ak-counter-public-test"

        def profile(name, run_id, source_sha, value):
            path = self.root / f"{name}.csv"
            path.write_text(
                "Dispatch_ID,Kernel_Name,Counter_Name,Counter_Value\n"
                f"7,{name}_kernel,SQ_LDS_BANK_CONFLICT,{value}\n",
                encoding="utf-8")
            return {
                "run_id": run_id, "campaign_id": campaign_id,
                "source_sha256": source_sha,
                "workload_sha256": workload_sha, "csv_path": str(path),
                "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kernel_name": f"{name}_kernel", "dispatch_ids": [7],
                "counter_name": "SQ_LDS_BANK_CONFLICT", "unit": "cycles",
                "normalization": "sum_exact_dispatches",
            }

        evidence = {
            "schema": H.COUNTER_EVIDENCE_SCHEMA, "task_id": "torch2hip/test/silu",
            "campaign_id": campaign_id,
            "candidate_source_sha256": candidate_source_sha,
            "anchor_identity_sha256": anchor_identity_sha,
            "workload_sha256": workload_sha,
            "expectation_sha256": H._canonical_sha256(expectation),
            "counter_name": "SQ_LDS_BANK_CONFLICT", "unit": "cycles",
            "normalization": "sum_exact_dispatches",
            "candidate": profile("candidate", "candidate-run", candidate_source_sha,
                                  candidate_value),
            "anchor": profile("anchor", "anchor-run", anchor_identity_sha, anchor_value),
        }
        context = {
            "candidate_source_sha256": candidate_source_sha,
            "task_id": "torch2hip/test/silu", "campaign_id": campaign_id,
            "workload_sha256": workload_sha,
            "anchor_identity_sha256": anchor_identity_sha,
        }
        return expectation, evidence, context

    def test_native_counter_receipt_replays_raw_rows_and_accepts_zero(self):
        expectation, evidence, context = self._counter_pair()
        result = H.evaluate_counter_gate(
            expectation=expectation, evidence=evidence, **context)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["anchor"]["normalized_value"], "10")
        self.assertEqual(result["candidate"]["normalized_value"], "0")
        self.assertTrue(result["direction_fired"])
        self.assertFalse(result["end_to_end_capture_proven"])
        self.assertEqual(result["capture_boundary"], "external_receipt_replay_only")

    def test_counter_gate_refuses_equal_or_wrong_direction(self):
        expectation, evidence, context = self._counter_pair(anchor_value="10",
                                                              candidate_value="10")
        result = H.evaluate_counter_gate(
            expectation=expectation, evidence=evidence, **context)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["speedup_claim_allowed"])

    def test_declared_counter_missing_or_forged_raw_receipt_is_unknown(self):
        expectation, evidence, context = self._counter_pair()
        missing = H.evaluate_counter_gate(
            expectation=expectation, evidence=None, **context)
        self.assertEqual(missing["status"], "UNKNOWN")
        evidence["candidate"]["csv_sha256"] = "0" * 64
        forged = H.evaluate_counter_gate(
            expectation=expectation, evidence=evidence, **context)
        self.assertEqual(forged["status"], "UNKNOWN")
        self.assertFalse(forged["speedup_claim_allowed"])

    def test_counter_raw_read_rejects_symlink_fifo_oversize_and_mutation(self):
        expectation, evidence, context = self._counter_pair()
        candidate_path = Path(evidence["candidate"]["csv_path"])
        symlink_path = self.root / "counter-symlink.csv"
        symlink_path.symlink_to(candidate_path)
        evidence["candidate"]["csv_path"] = str(symlink_path)
        self.assertEqual(H.evaluate_counter_gate(
            expectation=expectation, evidence=evidence, **context)["status"], "UNKNOWN")

        expectation, evidence, context = self._counter_pair()
        fifo_path = self.root / "counter.fifo"
        os.mkfifo(fifo_path)
        evidence["candidate"]["csv_path"] = str(fifo_path)
        self.assertEqual(H.evaluate_counter_gate(
            expectation=expectation, evidence=evidence, **context)["status"], "UNKNOWN")

        expectation, evidence, context = self._counter_pair()
        with mock.patch.object(H, "_MAX_COUNTER_CSV_BYTES", 8):
            oversized = H.evaluate_counter_gate(
                expectation=expectation, evidence=evidence, **context)
        self.assertEqual(oversized["status"], "UNKNOWN")
        self.assertIn("bounded read size", oversized["reason"])

        expectation, evidence, context = self._counter_pair()
        original_read = os.read
        mutated = False

        def mutate_after_read(descriptor, size):
            nonlocal mutated
            block = original_read(descriptor, size)
            if block and not mutated:
                mutated = True
                with open(evidence["candidate"]["csv_path"], "ab") as handle:
                    handle.write(b"\n")
            return block

        with mock.patch.object(H.os, "read", side_effect=mutate_after_read):
            changed = H.evaluate_counter_gate(
                expectation=expectation, evidence=evidence, **context)
        self.assertEqual(changed["status"], "UNKNOWN")
        self.assertIn("changed during bounded read", changed["reason"])

    def test_counter_raw_read_accepts_stable_hardlink(self):
        expectation, evidence, context = self._counter_pair()
        original = Path(evidence["candidate"]["csv_path"])
        hardlink = self.root / "counter-hardlink.csv"
        os.link(original, hardlink)
        evidence["candidate"]["csv_path"] = str(hardlink)
        result = H.evaluate_counter_gate(
            expectation=expectation, evidence=evidence, **context)
        self.assertEqual(result["status"], "PASS")

    def test_counter_raw_read_rejects_duplicate_casefold_and_empty_headers(self):
        for index, header in enumerate((
                "Dispatch_ID,Kernel_Name,Counter_Name,counter_name,Counter_Value\n",
                "Dispatch_ID,Kernel_Name,,Counter_Value\n")):
            expectation, evidence, context = self._counter_pair()
            path = self.root / f"ambiguous-{index}.csv"
            path.write_text(
                header + "7,candidate_kernel,SQ_LDS_BANK_CONFLICT,other,0\n",
                encoding="utf-8")
            evidence["candidate"]["csv_path"] = str(path)
            evidence["candidate"]["csv_sha256"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
            result = H.evaluate_counter_gate(
                expectation=expectation, evidence=evidence, **context)
            self.assertEqual(result["status"], "UNKNOWN")
            self.assertIn("headers", result["reason"])

    def test_counter_gate_refuses_unit_workload_and_normalization_mismatch(self):
        for field, bad_value in (("unit", "bytes"), ("workload_sha256", "d" * 64),
                                 ("normalization", "mean")):
            expectation, evidence, context = self._counter_pair()
            evidence["candidate"][field] = bad_value
            result = H.evaluate_counter_gate(
                expectation=expectation, evidence=evidence, **context)
            self.assertEqual(result["status"], "UNKNOWN", field)

    def test_counter_gate_refuses_cross_campaign_and_source_pair_mismatch(self):
        for field, bad_value in (("campaign_id", "other-campaign"),
                                 ("candidate_source_sha256", "d" * 64),
                                 ("anchor_identity_sha256", "e" * 64)):
            expectation, evidence, context = self._counter_pair()
            evidence[field] = bad_value
            result = H.evaluate_counter_gate(
                expectation=expectation, evidence=evidence, **context)
            self.assertEqual(result["status"], "UNKNOWN", field)

    def test_counter_gate_rejects_non_numeric_and_boolean_like_native_values(self):
        for raw in ("True", "NaN", "Infinity", "-1"):
            expectation, evidence, context = self._counter_pair(candidate_value=raw)
            result = H.evaluate_counter_gate(
                expectation=expectation, evidence=evidence, **context)
            self.assertEqual(result["status"], "UNKNOWN", raw)

    def test_correctness_and_speedup_controls_still_bound_rankability(self):
        expectation, evidence, context = self._counter_pair()
        counter = H.evaluate_counter_gate(
            expectation=expectation, evidence=evidence, **context)
        timing = {"candidate_beats_exact_provider": True}
        self.assertFalse(H.candidate_decision(
            correctness_passed=False, integrity_clean=True,
            timing=timing, counter_gate=counter)["speedup_claim_allowed"])
        failed_speed = H.candidate_decision(
            correctness_passed=True, integrity_clean=True,
            timing={"candidate_beats_exact_provider": False}, counter_gate=counter)
        self.assertFalse(failed_speed["rankable_against_exact_task_local_provider"])
        self.assertTrue(failed_speed["speedup_claim_allowed"])
        unknown = H.evaluate_counter_gate(
            expectation=expectation, evidence=None, **context)
        blocked = H.candidate_decision(
            correctness_passed=True, integrity_clean=True,
            timing=timing, counter_gate=unknown)
        self.assertFalse(blocked["speedup_claim_allowed"])
        self.assertFalse(blocked["candidate_beats_exact_provider"])

    def test_legacy_undeclared_counter_path_is_unchanged(self):
        gate = H.evaluate_counter_gate(
            expectation=None, evidence=None, candidate_source_sha256="b" * 64,
            task_id="torch2hip/test/silu", campaign_id="ak-counter-public-test",
            workload_sha256="a" * 64,
            anchor_identity_sha256="c" * 64)
        self.assertEqual(gate["status"], "NOT_REQUIRED")
        self.assertTrue(H.candidate_decision(
            correctness_passed=True, integrity_clean=True,
            timing={"candidate_beats_exact_provider": True},
            counter_gate=gate)["rankable_against_exact_task_local_provider"])
        legacy_incorrect = H.candidate_decision(
            correctness_passed=False, integrity_clean=False,
            timing={"candidate_beats_exact_provider": True}, counter_gate=gate)
        self.assertTrue(legacy_incorrect["speedup_claim_allowed"])
        self.assertFalse(legacy_incorrect["rankable_against_exact_task_local_provider"])

    def test_cli_passes_explicit_counter_inputs_and_suppresses_blocked_speedup(self):
        expectation_path = self.root / "expectation.json"
        evidence_path = self.root / "evidence.json"
        expectation_path.write_text("{}", encoding="utf-8")
        evidence_path.write_text("{}", encoding="utf-8")
        captured = {}
        fake_receipt = {
            "status": "complete", "campaign_id": "ak-counter-test",
            "receipt_sha256": "a" * 64,
            "decision": {"rankable_against_exact_task_local_provider": False,
                         "speedup_claim_allowed": False},
            "counter_gate": {"status": "UNKNOWN"},
            "timing": {"median_speedup": 1.5},
        }

        def fake_run(**kwargs):
            captured.update(kwargs)
            return fake_receipt

        with mock.patch.object(H, "run", side_effect=fake_run), \
                mock.patch("sys.stdout") as stdout:
            self.assertEqual(H.main([
                "--candidate-source", "candidate.hip", "--output-root", "out",
                "--campaign-id", "ak-counter-test",
                "--counter-expectation", str(expectation_path),
                "--counter-evidence", str(evidence_path)]), 0)
        self.assertEqual(captured["counter_expectation"], {})
        self.assertEqual(captured["counter_evidence"], {})
        printed = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
        self.assertIsNone(printed["median_speedup"])
        self.assertEqual(printed["counter_gate_status"], "UNKNOWN")

    def _run_public_panel(self, output_root, source, expectation, evidence):
        provider = {"provider_id": "torch_rocm_compile", "build_sha256": "f" * 64}
        suite = {"timing_input_sha256": "a" * 64, "cases": []}
        clean_sandbox = {"teardown": {"verified_empty": True}}

        def sandboxed(**kwargs):
            if kwargs["mode"] == "compile":
                return {"extension_has_forward_out": True}, clean_sandbox
            if kwargs["mode"] == "correctness":
                return {}, clean_sandbox
            return {"provider": provider, "blocks": []}, clean_sandbox

        def measurement_window(*, action, phase, **_kwargs):
            return action(), {"phase": phase}

        with (mock.patch.object(H.hip, "audit_task",
                                return_value=SimpleNamespace(to_dict=lambda: {"task_id": "torch2hip/test/silu"})),
              mock.patch.object(H.hip, "toolchain_identity", return_value={}),
              mock.patch.object(H.arena_adapter, "detect_gfx_arch",
                                return_value={"architectures": ["gfx90a"]}),
              mock.patch.object(H, "_scan_candidate", return_value={"clean": True}),
              mock.patch.object(H, "_create_suite", return_value=(suite, {})),
              mock.patch.object(H, "_run_sandboxed", side_effect=sandboxed),
              mock.patch.object(H.hip, "_measurement_window", side_effect=measurement_window),
              mock.patch.object(H, "_reduce_correctness", return_value={
                  "all_passed": True, "passed": 24, "total": 24}),
              mock.patch.object(H, "_reduce_timing", return_value={
                  "candidate_beats_exact_provider": True, "median_speedup": 1.2,
                  "block_count": 20})):
            return H.run(
                candidate_source=source, output_root=output_root,
                campaign_id="ak-counter-public-test", task_id="torch2hip/test/silu",
                counter_expectation=expectation, counter_evidence=evidence)

    def test_public_run_requires_and_consumes_counter_before_speedup_claim(self):
        source = self.root / "candidate.hip"
        source.write_text("void forward() { int value = 1; }\n", encoding="utf-8")
        expectation, evidence, context = self._counter_pair()
        evidence["candidate_source_sha256"] = H._sha256(source)
        evidence["candidate"]["source_sha256"] = H._sha256(source)
        evidence["workload_sha256"] = "a" * 64
        evidence["candidate"]["workload_sha256"] = "a" * 64
        evidence["anchor"]["workload_sha256"] = "a" * 64
        # The panel binds the exact provider object produced by its native timing arm.
        evidence["anchor_identity_sha256"] = H._canonical_sha256(
            {"provider_id": "torch_rocm_compile", "build_sha256": "f" * 64})
        evidence["anchor"]["source_sha256"] = evidence["anchor_identity_sha256"]

        blocked = self._run_public_panel(
            self.root / "missing-receipt", source, expectation, None)
        declaration = json.loads((self.root / "missing-receipt" /
                                  "counter-expectation.json").read_text(encoding="utf-8"))
        self.assertEqual(declaration["expectation"], expectation)
        self.assertEqual(declaration["candidate_source_sha256"], H._sha256(source))
        self.assertEqual(blocked["counter_gate"]["status"], "UNKNOWN")
        self.assertFalse(blocked["decision"]["speedup_claim_allowed"])
        self.assertFalse(blocked["decision"]["candidate_beats_exact_provider"])
        self.assertNotIn("hip_exact_provider_speedup", {
            row["measurement_id"] for row in blocked["belief_measurements"]})

        # Rebuild the expectation hash after binding all predeclared counter fields.
        evidence["expectation_sha256"] = H._canonical_sha256(expectation)
        passed = self._run_public_panel(
            self.root / "matched-receipt", source, expectation, evidence)
        self.assertEqual(passed["counter_gate"]["status"], "PASS")
        self.assertTrue(passed["decision"]["speedup_claim_allowed"])
        self.assertTrue(passed["decision"]["candidate_beats_exact_provider"])
        self.assertIn("hip_exact_provider_speedup", {
            row["measurement_id"] for row in passed["belief_measurements"]})


if __name__ == "__main__":
    unittest.main()
