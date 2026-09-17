"""Fixture-only tests for the retained-build diagnostic artifact."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from . import campaign, codegen_summary
from .loop import loop as loop_module, run as loop_run


class TestCodegenSummary(unittest.TestCase):
    def test_cpu_and_missing_hip_object_are_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cpu = codegen_summary.summarize_codegen("llama_cpu", directory)
            self.assertEqual(cpu["status"], "unavailable")
            self.assertIsNone(cpu["instruction_mix"])
            hip = codegen_summary.summarize_codegen("llama_gpu", directory)
            self.assertIn("embedded HIP fatbin", hip["reason"])
            self.assertIsNone(hip["occupancy"])
            self.assertIsNone(hip["register_spills"])

    def test_amd_code_object_summary_is_hash_bound_and_not_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            obj = Path(directory) / "kernel.hsaco"
            obj.write_bytes(b"fixture-object")
            disassembly = ("00000000: s_mov_b32 s0, 0\n"
                           "00000004: v_mfma_f32_16x16x16f16 v0, v1, v2, v3\n"
                           "00000008: global_load_dword v0, v1, off\n")
            with mock.patch.object(codegen_summary, "_disassemble",
                                   return_value=(disassembly, "ok")):
                summary = codegen_summary.summarize_codegen("llama_gpu", directory)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["objects"][0]["sha256"],
                             hashlib.sha256(b"fixture-object").hexdigest())
            self.assertEqual(summary["instruction_mix"]["matrix"], 1)
            self.assertEqual(summary["instruction_mix"]["memory"], 1)
            self.assertIn("non-CUDA", summary["ptx_sass_cubin"])
            self.assertIsNone(summary["occupancy"])

    def test_disassembler_timeout_is_a_real_wall_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "stall"
            executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(2)\n")
            executable.chmod(0o700)
            with mock.patch.object(codegen_summary, "LLVM_OBJDUMP", executable):
                start = time.monotonic()
                output, reason = codegen_summary._disassemble(
                    Path(directory) / "unused.hsaco", timeout_s=0.05)
                elapsed = time.monotonic() - start
            self.assertIsNone(output)
            self.assertEqual(reason, "disassembly timeout")
            self.assertLess(elapsed, 0.5)

    def test_total_disassembly_budget_is_shared_across_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                (Path(directory) / f"{index}.hsaco").write_bytes(b"fixture")
            with mock.patch.object(codegen_summary, "_disassemble",
                                   return_value=("", "ok")) as inspect:
                with mock.patch.object(codegen_summary.time, "monotonic",
                                       side_effect=[0.0, 1.0, 11.0]):
                    codegen_summary.summarize_codegen("llama_gpu", directory)
            self.assertEqual(inspect.call_args_list[0].kwargs["timeout_s"], 8.0)
            self.assertEqual(inspect.call_args_list[1].kwargs["timeout_s"], 1.0)

    def test_only_keep_attaches_and_summary_does_not_change_keep(self) -> None:
        spec = campaign.CampaignSpec(campaign_id="ak-codegen-test",
                                     candidate_id="akc-codegen-test",
                                     candidate_ref="fixture", backend="llama_cpu",
                                     model="/mnt/raid0/llm/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
        ops = campaign.HostOps()
        tree = object()
        keep = mock.Mock(keep=True)
        revert = mock.Mock(keep=False)
        with mock.patch.object(codegen_summary, "summarize_codegen",
                               return_value={"status": "unavailable"}) as inspect:
            self.assertFalse(ops.keep_or_revert(spec, tree, revert)["keep"])
            inspect.assert_not_called()
            self.assertTrue(ops.keep_or_revert(spec, tree, keep)["keep"])
            inspect.assert_called_once_with("llama_cpu", spec.build_dir)
        self.assertEqual(ops._codegen_summary, {"status": "unavailable"})

    def test_live_pool_sidecar_is_idempotent_and_commit_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            build.mkdir()
            head = "a" * 40
            first = codegen_summary.retain_summary(
                root / "store", head, backend="llama_gpu", build_dir=build)
            self.assertEqual(first["status"], "unavailable")
            sidecar = root / "store" / first["artifact_ref"]
            self.assertEqual(json.loads(sidecar.read_text()), first)
            self.assertEqual(codegen_summary.retain_summary(
                root / "store", head, backend="llama_gpu", build_dir=build), first)
            cpu = codegen_summary.retain_summary(
                root / "store", "b" * 40, backend="llama_cpu", build_dir=build)
            self.assertEqual(cpu["status"], "unavailable")
            self.assertEqual(cpu["backend"], "llama_cpu")
            self.assertNotEqual(first["artifact_ref"], cpu["artifact_ref"])
            changed_recipe = codegen_summary.retain_summary(
                root / "store", head, backend="llama_gpu", build_dir=build,
                recipe={"flags": "different"})
            self.assertNotEqual(first["artifact_ref"], changed_recipe["artifact_ref"])
            with self.assertRaises(ValueError):
                codegen_summary.retain_summary(
                    root / "store", "../not-a-commit", backend="llama_gpu",
                    build_dir=build)

    def test_write_side_tuple_binds_attempt_tree_and_unavailable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            build.mkdir()
            (build / "CMakeCache.txt").write_text(
                "CMAKE_CXX_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang++\n")
            kwargs = dict(backend="llama_gpu", build_dir=build,
                          recipe={"schema": "fixture-recipe", "GGML_HIP": "ON"},
                          attempt_identity="ak-attempt-fixture", source_tree_oid="d" * 40)
            summary = codegen_summary.retain_summary(
                root / "store", "e" * 40, **kwargs)
            tuple_row = summary["belief_claim_tuple"]
            self.assertEqual(tuple_row["protocol_id"], "")
            self.assertEqual(tuple_row["value"], 0)
            self.assertEqual(tuple_row["extra"]["attempt_identity"], "ak-attempt-fixture")
            self.assertEqual(tuple_row["extra"]["retained_source_tree_oid"], "d" * 40)
            self.assertEqual(tuple_row["extra"]["toolchain"]["status"], "captured")
            self.assertIsNone(tuple_row["extra"]["instruction_mix"])
            self.assertIn("occupancy", tuple_row["extra"]["unavailable_fields"])
            core = {key: value for key, value in summary.items()
                    if key not in {"summary_core_sha256", "belief_claim_tuple"}}
            digest = hashlib.sha256(json.dumps(
                core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(summary["summary_core_sha256"], digest)
            self.assertEqual(tuple_row["extra"]["summary_core_sha256"], digest)
            vidya = Path("/workspace/scripts/vidya")
            if (vidya / "claim_tuple.py").is_file():
                sys.path.insert(0, str(vidya))
                try:
                    from claim_tuple import ClaimTuple, grade
                    self.assertEqual(grade(ClaimTuple(**tuple_row))[:2],
                                     ("Judged", "Located"))
                finally:
                    sys.path.remove(str(vidya))
            self.assertEqual(codegen_summary.retain_summary(
                root / "store", "e" * 40, **kwargs), summary)
            with self.assertRaises(ValueError):
                codegen_summary.retain_summary(
                    root / "store", "e" * 40, **{**kwargs,
                        "attempt_identity": "different-attempt"})

    def test_live_attempt_embeds_diagnostic_only_for_keeps(self) -> None:
        head = "b" * 40
        summary = {"schema": codegen_summary.SCHEMA, "status": "unavailable"}
        kept = loop_module.Outcome("kept", champion_head=head)
        rejected = loop_module.Outcome("measured_null", champion_head=head)
        self.assertEqual(loop_run.attempt_with_codegen(
            kept, {head: summary})["codegen_summary"], summary)
        self.assertNotIn("codegen_summary", loop_run.attempt_with_codegen(
            rejected, {head: summary}))
        mismatch = {**summary, "attempt_identity": "other-attempt"}
        self.assertNotIn("attempt_identity", loop_run.attempt_with_codegen(
            kept, {head: mismatch})["codegen_summary"])
        runtime_hypothesis = mock.Mock(runtime_pair=object())
        runtime_hypothesis.to_dict.return_value = {"mechanism_id": "runtime-only"}
        runtime = loop_module.Outcome("kept", hypothesis=runtime_hypothesis)
        self.assertIn("no new codegen artifact", loop_run.attempt_with_codegen(
            runtime, {})["codegen_summary"]["reason"])

    def test_concurrent_sidecar_create_never_overwrites_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            build.mkdir()
            winner = b"other writer's immutable artifact\n"

            def rival(_source, destination):
                Path(destination).write_bytes(winner)
                raise FileExistsError(destination)

            with mock.patch.object(codegen_summary.os, "link", side_effect=rival):
                with self.assertRaisesRegex(ValueError, "existing codegen sidecar"):
                    codegen_summary.retain_summary(
                        root / "store", "c" * 40, backend="llama_gpu",
                        build_dir=build)
            sidecars = list((root / "store" / "codegen").glob("*.json"))
            self.assertEqual(len(sidecars), 1)
            self.assertEqual(sidecars[0].read_bytes(), winner)

    def test_live_commit_hook_covers_gpu_and_direct_cpu_source_keeps(self) -> None:
        """Static seam test; executing main would claim the shared GPU."""
        source = Path(loop_run.__file__).read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Attribute)
                            and call.func.attr == "retain_summary" for call in calls))
        self.assertIn('backend="llama_cpu" if cpu_launch else "llama_gpu"', source)
        self.assertIn("recipe=recipe.to_dict()", source)
        self.assertIn("attempt_with_codegen(outcome, codegen_by_head)", source)


if __name__ == "__main__":
    unittest.main()
