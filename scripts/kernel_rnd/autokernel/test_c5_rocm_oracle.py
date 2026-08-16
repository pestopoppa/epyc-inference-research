#!/usr/bin/env python3
"""Hardware-free, fail-closed tests for the C5 gfx90a correctness provider."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import c5_rocm_oracle as C


class C5RocmOracleConfigTest(unittest.TestCase):
    def document(self):
        return json.loads(C._config_path().read_text(encoding="utf-8"))

    def test_checked_in_config_covers_all_eight_seeds_and_193_workloads(self):
        config = C.load()
        self.assertEqual(tuple(seed.seed_id for seed in config.seeds), tuple(C.EXPECTED_PROBLEMS))
        self.assertEqual(sum(seed.workload_count for seed in config.seeds), 193)
        self.assertEqual(
            [seed.workload_count for seed in config.seeds],
            [16, 16, 16, 16, 29, 15, 47, 38],
        )

    def test_oracle_dtype_evidence_is_not_conflated_with_hyra_candidate_metadata(self):
        config = C.load()
        by_id = {seed.seed_id: seed for seed in config.seeds}
        self.assertEqual(by_id["k145"].oracle_workload_dtypes, ("fp32",))
        self.assertEqual(by_id["k227"].oracle_workload_dtypes, ("bf16",))
        # The launch-tip HyRA registry separately describes implementation
        # metadata, not the port's tracked workload records.
        from . import c5_seed_corpus
        hyra = {seed.seed_id: seed for seed in c5_seed_corpus.load().seeds}
        self.assertEqual(hyra["k145"].dtypes, ("fp16",))
        self.assertEqual(hyra["k227"].dtypes, ("bf16", "fp16"))

    def test_gfx950_constants_cannot_be_relabelled_or_enabled_for_gfx90a(self):
        mutations = []
        document = self.document()
        document["scoring"]["enabled"] = True
        mutations.append(document)

        document = self.document()
        document["scoring"]["source_constant_architectures"].append("gfx90a")
        mutations.append(document)

        document = self.document()
        document["scoring"]["import_source_constants"] = True
        mutations.append(document)

        for document in mutations:
            with self.subTest(scoring=document["scoring"]), self.assertRaisesRegex(
                C.OracleRefusal, "scoring|constants"
            ):
                C._parse_config(copy.deepcopy(document))

    def test_rocm_72_source_provenance_is_mandatory_and_exact(self):
        document = self.document()
        del document["source_runtime_provenance"]["torch_hip_version"]
        with self.assertRaisesRegex(C.OracleRefusal, "source runtime fields"):
            C._parse_config(document)

        document = self.document()
        document["source_runtime_provenance"]["rocm_version"] = "6.2.0"
        with self.assertRaisesRegex(C.OracleRefusal, "ROCm 7.2 provenance"):
            C._parse_config(document)

    def test_problem_join_population_and_dtype_drift_refuse(self):
        document = self.document()
        document["seeds"][0]["problem_id"] = document["seeds"][1]["problem_id"]
        with self.assertRaisesRegex(C.OracleRefusal, "problem join"):
            C._parse_config(document)

        document = self.document()
        document["seeds"][1]["oracle_workload_dtypes"] = ["fp16"]
        with self.assertRaisesRegex(C.OracleRefusal, "dtype evidence"):
            C._parse_config(document)

        document = self.document()
        document["seeds"][7]["workload_count"] -= 1
        with self.assertRaisesRegex(C.OracleRefusal, "total 193"):
            C._parse_config(document)


class C5RocmOraclePlanTest(unittest.TestCase):
    @staticmethod
    def runtime():
        return {
            "rocm_version": "6.2.0",
            "torch_version": "2.3.0+rocm6.2",
            "torch_hip_version": "6.2.41133",
            "driver_version": "6.8.5",
        }

    @staticmethod
    def audit():
        return {
            "schema": C.AUDIT_SCHEMA,
            "provider_id": C.PROVIDER_ID,
            "source_commit": C.load().source["commit"],
            "seed_count": 8,
            "workload_count": 193,
            "hardware_accessed": False,
            "build_executed": False,
            "scoring_constants_imported": False,
            "authority": C.AUTHORITY,
            "receipt_sha256": "a" * 64,
        }

    def plan(self, seed_ids=None):
        with mock.patch.object(C, "audit_primary_artifacts", return_value=self.audit()), \
                mock.patch.object(C, "_render_correctness_driver", return_value="# oracle\n"):
            return C.compile_plan(
                "/tmp", runtime_provenance=self.runtime(), seed_ids=seed_ids)

    def test_plan_is_correctness_only_and_records_cross_version_without_claiming_compatibility(self):
        plan = self.plan()
        self.assertEqual(plan["target"]["architecture"], "gfx90a")
        self.assertEqual(plan["target"]["hardware"], "LOCAL")
        self.assertEqual(plan["corpus"]["workload_count"], 193)
        self.assertEqual(plan["operations"]["correctness_rounds"], 10)
        self.assertTrue(plan["operations"]["compile"])
        self.assertTrue(plan["operations"]["correctness"])
        self.assertFalse(plan["operations"]["timing"])
        self.assertFalse(plan["operations"]["profiling"])
        self.assertFalse(plan["operations"]["sol_scoring"])
        self.assertFalse(plan["execution_seam"]["timing_path_reachable"])
        self.assertFalse(plan["execution_seam"]["sol_scoring_path_reachable"])
        self.assertEqual(
            plan["execution_seam"]["required_environment"],
            {C._CORRECTNESS_ONLY_ENV: "1"},
        )
        self.assertFalse(plan["scoring"]["enabled"])
        self.assertFalse(plan["runtime_compatibility"]["rocm_exact_match"])
        self.assertFalse(plan["runtime_compatibility"]["compatibility_claimed"])
        self.assertNotIn("sol_score", json.dumps(plan).lower())

    def test_runtime_provenance_is_required_even_when_not_rocm72(self):
        runtime = self.runtime()
        del runtime["driver_version"]
        with mock.patch.object(C, "audit_primary_artifacts", return_value=self.audit()), \
                mock.patch.object(C, "_render_correctness_driver", return_value="# oracle\n"):
            with self.assertRaisesRegex(C.OracleRefusal, "runtime_provenance fields"):
                C.compile_plan("/tmp", runtime_provenance=runtime)

        runtime = self.runtime()
        runtime["rocm_version"] = "unknown"
        with mock.patch.object(C, "audit_primary_artifacts", return_value=self.audit()), \
                mock.patch.object(C, "_render_correctness_driver", return_value="# oracle\n"):
            with self.assertRaisesRegex(C.OracleRefusal, "exact version"):
                C.compile_plan("/tmp", runtime_provenance=runtime)

    def test_staged_driver_exits_after_correctness_before_port_timing(self):
        with tempfile.TemporaryDirectory(prefix="ak-c5-source-") as source_text, \
                tempfile.TemporaryDirectory(prefix="ak-c5-stage-") as stage_text:
            source = Path(source_text)
            template = source / "src/sol_execbench/driver/templates/eval_driver.py"
            template.parent.mkdir(parents=True)
            template.write_text("# fixture\n" + C._EVAL_ANCHOR + "# timing\n", encoding="utf-8")
            with mock.patch.object(C, "audit_primary_artifacts", return_value=self.audit()):
                receipt = C.stage_correctness_driver(source, stage_text)
            rendered = Path(stage_text, "eval_driver.py").read_text(encoding="utf-8")
            self.assertIn(C._CORRECTNESS_ONLY_ENV, rendered)
            self.assertIn("performance=None", rendered)
            self.assertIn("continue\n\n    # -- Monkey-patch defense before timing --", rendered)
            self.assertFalse(receipt["timing_path_reachable"])
            self.assertFalse(receipt["sol_scoring_path_reachable"])
            self.assertFalse(receipt["hardware_accessed"])

    def test_staging_refuses_to_overwrite_any_noncanonical_driver(self):
        with tempfile.TemporaryDirectory(prefix="ak-c5-source-") as source_text, \
                tempfile.TemporaryDirectory(prefix="ak-c5-stage-") as stage_text:
            source = Path(source_text)
            template = source / "src/sol_execbench/driver/templates/eval_driver.py"
            template.parent.mkdir(parents=True)
            template.write_text("# fixture\n" + C._EVAL_ANCHOR, encoding="utf-8")
            Path(stage_text, "eval_driver.py").write_text("foreign\n", encoding="utf-8")
            with mock.patch.object(C, "audit_primary_artifacts", return_value=self.audit()):
                with self.assertRaisesRegex(C.OracleRefusal, "differs from the audited"):
                    C.stage_correctness_driver(source, stage_text)

    def test_seed_selection_preserves_counts_and_separate_dtype_surfaces(self):
        plan = self.plan(("k145", "k227"))
        self.assertEqual(plan["corpus"]["workload_count"], 63)
        rows = {row["seed_id"]: row for row in plan["corpus"]["seeds"]}
        self.assertEqual(rows["k145"]["oracle_workload_dtypes"], ["fp32"])
        self.assertEqual(rows["k145"]["hyra_reference_dtypes"], ["fp16"])
        self.assertEqual(rows["k227"]["oracle_workload_dtypes"], ["bf16"])
        self.assertEqual(rows["k227"]["hyra_reference_dtypes"], ["bf16", "fp16"])

    def full_result(self, plan):
        return {
            "schema": C.RESULT_SCHEMA,
            "provider_id": C.PROVIDER_ID,
            "plan_sha256": plan["plan_sha256"],
            "target": {"hardware": "LOCAL", "architecture": "gfx90a"},
            "runtime_provenance": plan["runtime_provenance"],
            "authority": C.AUTHORITY,
            "seed_results": [{
                "seed_id": seed["seed_id"],
                "compile_status": "passed",
                "correctness_status": "passed",
                "correctness_rounds_run": 10,
                "workloads_checked": seed["workload_count"],
                "error": None,
            } for seed in plan["corpus"]["seeds"]],
        }

    def test_full_compile_and_correctness_result_is_admitted_without_score(self):
        plan = self.plan(("k138", "k215"))
        result = self.full_result(plan)
        self.assertEqual(C.validate_result(result, plan=plan), result)

    def test_false_sol_timing_and_performance_claims_refuse_at_any_depth(self):
        plan = self.plan(("k138",))
        for key in ("sol_score", "t_sol_ms", "t_b_ms", "latency_ms", "speedup"):
            result = self.full_result(plan)
            result["seed_results"][0][key] = 1.0
            with self.subTest(key=key), self.assertRaisesRegex(
                C.OracleRefusal, "correctness-only result"
            ):
                C.validate_result(result, plan=plan)

    def test_correctness_pass_requires_all_workloads_and_all_ten_rounds(self):
        plan = self.plan(("k138",))
        result = self.full_result(plan)
        result["seed_results"][0]["correctness_rounds_run"] = 9
        with self.assertRaisesRegex(C.OracleRefusal, "full ten-round coverage"):
            C.validate_result(result, plan=plan)

        result = self.full_result(plan)
        result["seed_results"][0]["workloads_checked"] = 15
        with self.assertRaisesRegex(C.OracleRefusal, "full ten-round coverage"):
            C.validate_result(result, plan=plan)


if __name__ == "__main__":
    unittest.main()
