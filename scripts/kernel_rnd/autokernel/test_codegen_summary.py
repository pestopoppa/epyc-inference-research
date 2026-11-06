"""Fixture-only tests for the retained-build diagnostic artifact."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock

from . import codegen_summary
from .loop import loop as loop_module, run as loop_run


class TestCodegenSummary(unittest.TestCase):
    @staticmethod
    def _embedded_hip_dso(target: bytes = b"hipv4-amdgcn-amd-amdhsa--gfx90a",
                          object_bytes: bytes = b"\x7fELFfixture") -> tuple[bytes, int]:
        """Minimal ELF64 DSO with one clang offload bundle in .hip_fatbin."""
        names = b"\0.shstrtab\0.hip_fatbin\0"
        bundle = (codegen_summary._BUNDLE_MAGIC + struct.pack("<Q", 1)
                  + struct.pack("<QQQ", 256, len(object_bytes), len(target))
                  + target)
        fatbin = bundle + bytes(256 - len(bundle)) + object_bytes
        elf = bytearray(512 + len(fatbin))
        elf[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<Q", elf, 40, 64)
        struct.pack_into("<HHH", elf, 58, 64, 3, 1)
        struct.pack_into("<IIQQQQIIQQ", elf, 64 + 64,
                         1, 3, 0, 0, 256, len(names), 0, 0, 1, 0)
        struct.pack_into("<IIQQQQIIQQ", elf, 64 + 128,
                         11, 1, 0, 0, 512, len(fatbin), 0, 0, 1, 0)
        elf[256:256 + len(names)] = names
        elf[512:] = fatbin
        return bytes(elf), 512 + 256

    def test_embedded_gfx90a_summary_binds_container_and_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "bin").mkdir()
            raw, offset = self._embedded_hip_dso()
            (build / "bin" / "libggml-hip.so").write_bytes(raw)
            with mock.patch.object(codegen_summary, "_disassemble",
                                   return_value=(
                                       "\ts_nop 0 // ABCDEF: BF800000\n"
                                       "\tv_mfma_f32_16x16x16f16 v0, v1 // ABCDEF: DEADBEEF\n"
                                       "\tglobal_load_dword v0, v1 // ABCDEF: DEADBEEF\n",
                                       "ok")):
                summary = codegen_summary.summarize_codegen("llama_gpu", build)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["instruction_mix"]["matrix"], 1)
            self.assertEqual(summary["instruction_mix"]["memory"], 1)
            self.assertEqual(summary["instruction_mix"]["scalar"], 1)
            self.assertEqual(len(summary["objects"]), 1)
            row = summary["objects"][0]
            self.assertEqual(row["container_path"], "bin/libggml-hip.so")
            self.assertEqual(row["container_bytes"], len(raw))
            self.assertEqual(row["container_offset"], offset)
            self.assertEqual(row["container_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(row["sha256"], hashlib.sha256(b"\x7fELFfixture").hexdigest())
            self.assertTrue(row["relative_path"].endswith(f"#fatbin-{offset}.co"))
            self.assertIsNone(summary["occupancy"])
            self.assertIsNone(summary["register_spills"])

    def test_embedded_non_gfx_or_oversized_container_stays_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "bin").mkdir()
            library = build / "bin" / "libggml-hip.so"
            raw, _ = self._embedded_hip_dso(target=b"hipv4-amdgcn-amd-amdhsa--gfx1100")
            library.write_bytes(raw)
            self.assertEqual(codegen_summary.summarize_codegen(
                "llama_gpu", build)["status"], "unavailable")
            with mock.patch.object(codegen_summary, "MAX_HIP_LIBRARY_BYTES", len(raw) - 1):
                summary = codegen_summary.summarize_codegen("llama_gpu", build)
            self.assertEqual(summary["status"], "unavailable")
            self.assertIn("exceeds inspection bound", summary["reason"])

    def test_cpu_and_missing_hip_object_are_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cpu = codegen_summary.summarize_codegen("llama_cpu", directory)
            self.assertEqual(cpu["status"], "unavailable")
            self.assertIsNone(cpu["instruction_mix"])
            hip = codegen_summary.summarize_codegen("llama_gpu", directory)
            self.assertIn("candidate HIP library missing", hip["reason"])
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

    def test_cpu_symbol_summary_is_bounded_and_hashes_exact_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            bin_dir = build / "bin"
            bin_dir.mkdir(parents=True)
            elf = bin_dir / "libggml-cpu.so.0.16.0"
            elf.write_bytes(b"fixture-cpu-elf")
            (bin_dir / "libggml-cpu.so").symlink_to(elf.name)

            def disassembly(_path, symbol, *, timeout_s):
                self.assertGreater(timeout_s, 0)
                if symbol != "ggml_compute_forward_gated_delta_net":
                    return None, "CPU symbol absent from disassembly"
                return ("00000100 <ggml_compute_forward_gated_delta_net>:\n"
                        " 100: vmovaps %xmm0,%xmm1\n"
                        " 104: mov (%rax),%ebx\n"
                        " 108: add %ebx,%ecx\n"), "ok"

            with mock.patch.object(codegen_summary, "_disassemble_cpu",
                                   side_effect=disassembly):
                result = codegen_summary.summarize_codegen("llama_cpu", build)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["objects"][0]["relative_path"],
                             "bin/libggml-cpu.so.0.16.0")
            self.assertEqual(result["objects"][0]["sha256"],
                             hashlib.sha256(b"fixture-cpu-elf").hexdigest())
            self.assertEqual(result["objects"][0]["symbols"][0]["name"],
                             "ggml_compute_forward_gated_delta_net")
            self.assertEqual(result["instruction_mix"]["vector"], 1)
            self.assertEqual(result["instruction_mix"]["memory"], 1)
            self.assertEqual(result["instruction_mix"]["scalar"], 1)
            self.assertIsNone(result["occupancy"])
            self.assertIsNone(result["register_spills"])

    def test_cpu_library_outside_build_or_symbols_absent_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            bin_dir = build / "bin"
            bin_dir.mkdir(parents=True)
            outside = root / "external.so"
            outside.write_bytes(b"not this build")
            (bin_dir / "libggml-cpu.so").symlink_to(outside)
            self.assertEqual(codegen_summary.summarize_codegen(
                "llama_cpu", build)["status"], "unavailable")
            (bin_dir / "libggml-cpu.so").unlink()
            (bin_dir / "libggml-cpu.so").write_bytes(b"stripped-fixture")
            with mock.patch.object(codegen_summary, "_disassemble_cpu",
                                   return_value=(None, "CPU symbol absent from disassembly")):
                result = codegen_summary.summarize_codegen("llama_cpu", build)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["objects"][0]["symbols"], [])
            self.assertIsNone(result["instruction_mix"])

    def test_cpu_retained_tuple_binds_library_and_disassembled_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            bin_dir = build / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "libggml-cpu.so.0.16.0").write_bytes(b"fixture-cpu-elf")
            (bin_dir / "libggml-cpu.so").symlink_to("libggml-cpu.so.0.16.0")
            symbol = "ggml_compute_forward_gated_delta_net"
            disassembly = (f"00000100 <{symbol}>:\n"
                           " 100: add %ebx,%ecx\n")
            with mock.patch.object(codegen_summary, "_disassemble_cpu",
                                   side_effect=lambda _path, name, **_kw: (
                                       (disassembly, "ok") if name == symbol else
                                       (None, "CPU symbol absent from disassembly"))):
                summary = codegen_summary.retain_summary(
                    root / "store", "a" * 40, backend="llama_cpu",
                    build_dir=build, recipe={"kind": "fixture"},
                    attempt_identity="cpu-keep", source_tree_oid="b" * 40)
            self.assertEqual(summary["belief_claim_tuple"]["value"], 1)
            self.assertEqual(summary["belief_claim_tuple"]["extra"]["disassembled_symbols"],
                             [symbol])
            self.assertEqual(summary["belief_claim_tuple"]["extra"]["code_objects"], [
                {"relative_path": "bin/libggml-cpu.so.0.16.0",
                 "sha256": hashlib.sha256(b"fixture-cpu-elf").hexdigest()}])
            self.assertIn("cpu_objdump_stat", summary["build_frame"]["toolchain"])

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


H5_FA_CO_CANDIDATES = (
    Path(__file__).parent / "fixtures" / "h5_fa_hd128.gfx90a.co",
    Path("/mnt/raid0/llm/tmp/ak-asm-h5-fa-retarget-20261105/build/"
         "fwd_hd128_bf16_rtne.gfx90a.co"),
)


class TestGfx90aIsaSidecar(unittest.TestCase):
    NOTES = """\
amdhsa.kernels:
  - .name:           k0
    .symbol:         k0.kd
    .vgpr_count:     64
    .agpr_count:     32
    .sgpr_count:     40
    .vgpr_spill_count: 3
    .sgpr_spill_count: 0
    .private_segment_fixed_size: 16
    .group_segment_fixed_size: 8192
amdhsa.target:   amdgcn-amd-amdhsa--gfx90a
"""
    DIS = """\
1000 <k0>:
\ts_setprio 3                                                  // 1000: BF8F0003
\tv_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]       // 1004: D3D48000
\tglobal_load_dwordx4 v[4:7], v[8:9], off                       // 1008: DC380000
\ts_barrier                                                     // 100C: BF8A0000
\tv_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]       // 1010: D3D48000
\tv_add_u32 v0, v0, v1                                          // 1014: 68000101
\tds_read_b128 v[4:7], v8                                       // 1018: D9FE0000
\tv_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]       // 101C: D3D48000
\ts_endpgm                                                      // 1020: BF810000
"""

    def test_text_fixture_gap_histogram_and_barrier_counts(self):
        out = codegen_summary.gfx90a_isa_from_texts(self.NOTES, self.DIS)
        block = out["k0"]["gfx90a_isa"]
        self.assertEqual((block["vgpr"], block["agpr"], block["sgpr"]), (64, 32, 40))
        self.assertEqual(block["vgpr_spill_count"], 3)
        self.assertEqual(block["scratch_bytes"], 16)
        self.assertEqual(block["lds_bytes"], 8192)
        self.assertEqual(block["mfma"], 3)
        gap = block["mfma_to_load_gap"]
        # mfma#1 -> load immediately (gap 0); mfma#2 -> 2 insts to ds_read (gap 1);
        # mfma#3 has no later load.
        self.assertEqual((gap["0"], gap["1"], gap["none"]), (1, 1, 1))
        self.assertEqual(sum(gap.values()), block["mfma"])
        self.assertEqual(block["mfma_after_barrier"], 2)
        self.assertEqual(block["mfma_within_16_after_barrier"], 2)
        self.assertEqual(block["s_setprio"], 1)
        self.assertEqual(block["instruction_mix"]["matrix"], 3)

    def test_schedule_stats_pure(self):
        stats = codegen_summary.schedule_stats(
            ["v_mfma_x"] + ["v_nop"] * 40 + ["buffer_load_dword", "s_barrier", "v_mfma_y"])
        self.assertEqual(stats["mfma_to_load_gap"]["33+"], 1)
        self.assertEqual(stats["mfma_to_load_gap"]["none"], 1)
        self.assertEqual(stats["mfma_after_barrier"], 1)

    def test_sidecar_is_not_part_of_summary_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = codegen_summary.summarize_codegen("llama_gpu", tmp)
        self.assertNotIn("gfx90a_isa", json.dumps(summary))

    def test_h5_fa_hd128_code_object(self):
        co = next((p for p in H5_FA_CO_CANDIDATES if p.is_file()), None)
        if co is None:
            self.skipTest("SKIP-MARKER: H5 FA hd128 .co fixture absent")
        audit = codegen_summary._isa_audit()
        if not (audit.LLVM_BIN / "llvm-readelf").is_file():
            self.skipTest("SKIP-MARKER: ROCm llvm tools absent")
        data = co.read_bytes()
        doc = codegen_summary.gfx90a_isa_sidecar(data)
        self.assertEqual(doc["status"], "ok", doc["reason"])
        self.assertEqual(doc["code_object_sha256"], hashlib.sha256(data).hexdigest())
        blocks = [v["gfx90a_isa"] for v in doc["kernels"].values()]
        self.assertTrue(blocks)
        top = max(blocks, key=lambda b: b["mfma"])
        self.assertGreater(top["mfma"], 0)
        self.assertGreater(top["vgpr"], 0)
        self.assertEqual(sum(top["mfma_to_load_gap"].values()), top["mfma"])
        with tempfile.TemporaryDirectory() as tmp:
            ref = codegen_summary.retain_gfx90a_isa_sidecar(Path(tmp), data)
            again = codegen_summary.retain_gfx90a_isa_sidecar(Path(tmp), data)
            self.assertEqual(ref, again)
            payload = (Path(tmp) / ref["ref"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), ref["sha256"])


if __name__ == "__main__":
    unittest.main()
