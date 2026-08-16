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
        config = C.load()
        rows = [{
            "seed_id": seed.seed_id,
            "problem_id": seed.problem_id,
            "workload_count": seed.workload_count,
            "oracle_workload_dtypes": list(seed.oracle_workload_dtypes),
            "workload_sha256": seed.workload_sha256,
            "workload_uuids": [
                f"{seed.seed_id}-workload-{index:03d}"
                for index in range(seed.workload_count)
            ],
        } for seed in config.seeds]
        value = {
            "schema": C.AUDIT_SCHEMA,
            "provider_id": C.PROVIDER_ID,
            "source_root": "/tmp",
            "source_commit": C.load().source["commit"],
            "manifest_sha256": C.load().document["primary_artifacts"]["manifest_sha256"],
            "provider_code": {},
            "seed_count": 8,
            "workload_count": 193,
            "seeds": rows,
            "hardware_accessed": False,
            "build_executed": False,
            "scoring_constants_imported": False,
            "authority": C.AUTHORITY,
        }
        value["receipt_sha256"] = C._canonical_sha256(value)
        return value

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
            plan["execution_seam"]["correctness_stop"],
            "unconditional_before_timing",
        )
        self.assertFalse(plan["scoring"]["enabled"])
        self.assertFalse(plan["runtime_compatibility"]["rocm_exact_match"])
        self.assertFalse(plan["runtime_compatibility"]["compatibility_claimed"])
        self.assertNotIn("sol_score", json.dumps(plan.to_dict()).lower())
        with self.assertRaises(TypeError):
            plan["corpus"]["workload_count"] = 0
        self.assertEqual(
            plan["plan_sha256"],
            C._canonical_sha256({
                key: value for key, value in plan.to_dict().items()
                if key != "plan_sha256"
            }),
        )

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
            self.assertNotIn("EPYC_AUTOKERNEL_C5_CORRECTNESS_ONLY", rendered)
            self.assertNotIn("if os.environ", C._EVAL_REPLACEMENT)
            self.assertIn('"performance": None', rendered)
            self.assertIn("continue\n\n    # -- Monkey-patch defense before timing --", rendered)
            self.assertEqual(receipt["correctness_stop"], "unconditional_before_timing")
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
        self.assertEqual(rows["k145"]["oracle_workload_dtypes"], ("fp32",))
        self.assertEqual(rows["k145"]["hyra_reference_dtypes"], ("fp16",))
        self.assertEqual(rows["k227"]["oracle_workload_dtypes"], ("bf16",))
        self.assertEqual(rows["k227"]["hyra_reference_dtypes"], ("bf16", "fp16"))

    def stage_receipt(self, plan, root):
        destination = Path(root, "eval_driver.py")
        destination.write_text("# oracle\n", encoding="utf-8")
        stage = {
            "schema": C.STAGE_SCHEMA,
            "provider_id": C.PROVIDER_ID,
            "source_audit_sha256": plan["source_audit"]["receipt_sha256"],
            "destination": str(destination),
            "driver_sha256": C._file_sha256(destination),
            "correctness_stop": "unconditional_before_timing",
            "replaced_packager_template": True,
            "timing_path_reachable": False,
            "sol_scoring_path_reachable": False,
            "hardware_accessed": False,
            "build_executed": False,
            "authority": C.AUTHORITY,
        }
        stage["receipt_sha256"] = C._canonical_sha256(stage)
        return stage

    @staticmethod
    def raw_traces(plan):
        rows = []
        for seed in plan["corpus"]["seeds"]:
            for workload_uuid in seed["workload_uuids"]:
                rows.append({
                    "schema": C.RAW_TRACE_SCHEMA,
                    "provider_id": C.PROVIDER_ID,
                    "seed_id": seed["seed_id"],
                    "problem_id": seed["problem_id"],
                    "solution": "candidate",
                    "workload_uuid": workload_uuid,
                    "target": {"hardware": "AMD Instinct MI210", "architecture": "gfx90a"},
                    "evaluation": {
                        "status": "PASSED",
                        "correctness": {
                            "max_relative_error": 0.0,
                            "max_absolute_error": 0.0,
                            "has_nan": False,
                            "has_inf": False,
                            "extra": None,
                        },
                        "performance": None,
                        "rounds": 10,
                        "fresh_inputs_each_round": True,
                        "live_reference_each_round": True,
                        "message": (
                            "EPYC AutoKernel correctness oracle; timing and SOL scoring disabled"
                        ),
                    },
                    "authority": C.AUTHORITY,
                })
        return b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows)

    def reduce(self, raw, plan, stage):
        with mock.patch.object(C, "audit_primary_artifacts", return_value=self.audit()), \
                mock.patch.object(C, "_render_correctness_driver", return_value="# oracle\n"):
            return C.reduce_staged_result(raw, plan=plan, stage_receipt=stage)

    def validate(self, result, raw, plan, stage):
        with mock.patch.object(C, "audit_primary_artifacts", return_value=self.audit()), \
                mock.patch.object(C, "_render_correctness_driver", return_value="# oracle\n"):
            return C.validate_result(
                result, plan=plan, stage_receipt=stage, raw_traces=raw)

    def test_full_compile_and_correctness_result_is_reduced_from_raw_bytes(self):
        plan = self.plan(("k138", "k215"))
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.stage_receipt(plan, temporary)
            raw = self.raw_traces(plan)
            result = self.reduce(raw, plan, stage)
            self.assertEqual(result["raw_trace_count"], 45)
            self.assertEqual(result["raw_trace_sha256"], C.hashlib.sha256(raw).hexdigest())
            self.assertEqual(self.validate(result, raw, plan, stage), result)
            with self.assertRaises(TypeError):
                result["raw_trace_count"] = 0

    def test_original_fabricated_plan_zero_workload_poc_is_refused(self):
        fabricated = {
            "plan_sha256": "not-a-sha", "runtime_provenance": {"fabricated": True},
            "corpus": {"seeds": [{"seed_id": "k138", "workload_count": 0}]},
        }
        with self.assertRaisesRegex(C.OracleRefusal, "plan.*fields|plan_sha256"):
            C.reduce_staged_result(b"{}\n", plan=fabricated, stage_receipt={})

        plan = self.plan(("k138",))
        tampered = plan.to_dict()
        tampered["corpus"]["seeds"][0]["workload_count"] = 0
        tampered["corpus"]["seeds"][0]["workload_uuids"] = []
        tampered["corpus"]["workload_count"] = 0
        tampered["plan_sha256"] = C._canonical_sha256({
            key: value for key, value in tampered.items() if key != "plan_sha256"
        })
        with mock.patch.object(C, "audit_primary_artifacts", return_value=self.audit()), \
                mock.patch.object(C, "_render_correctness_driver", return_value="# oracle\n"), \
                self.assertRaisesRegex(C.OracleRefusal, "differs from primary evidence"):
            C._validate_plan(tampered)

    def test_raw_population_timing_claim_and_aggregate_forgery_refuse(self):
        plan = self.plan(("k138",))
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.stage_receipt(plan, temporary)
            raw = self.raw_traces(plan)
            result = self.reduce(raw, plan, stage)

            with self.assertRaisesRegex(C.OracleRefusal, "population is incomplete"):
                self.reduce(b"\n".join(raw.splitlines()[:-1]) + b"\n", plan, stage)

            rows = [json.loads(line) for line in raw.splitlines()]
            rows[0]["evaluation"]["performance"] = {"latency_ms": 1.0}
            timed = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
            with self.assertRaisesRegex(C.OracleRefusal, "correctness-only result"):
                self.reduce(timed, plan, stage)

            rows = [json.loads(line) for line in raw.splitlines()]
            rows[0]["target"]["architecture"] = "gfx950"
            foreign = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
            with self.assertRaisesRegex(C.OracleRefusal, "correctness-only authority"):
                self.reduce(foreign, plan, stage)

            forged = C._plain(result)
            forged["seed_results"][0]["workloads_checked"] = 0
            forged["result_sha256"] = C._canonical_sha256({
                key: value for key, value in forged.items() if key != "result_sha256"
            })
            with self.assertRaisesRegex(C.OracleRefusal, "strict raw-trace reduction"):
                self.validate(forged, raw, plan, stage)

    def test_staged_driver_byte_drift_refuses_before_reduction(self):
        plan = self.plan(("k138",))
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.stage_receipt(plan, temporary)
            Path(stage["destination"]).write_text("candidate changed driver\n", encoding="utf-8")
            with self.assertRaisesRegex(C.OracleRefusal, "stage receipt differs"):
                self.reduce(self.raw_traces(plan), plan, stage)


if __name__ == "__main__":
    unittest.main()
