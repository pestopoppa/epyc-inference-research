#!/usr/bin/env python3
"""Zero-device tests for the paired C4 capture launcher."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark import capture_autokernel_c4_profile as C


class CaptureAutokernelC4ProfileTest(unittest.TestCase):
    def args(self, **overrides) -> argparse.Namespace:
        values = {
            "workload_kind": "quant-op",
            "quant_type": "iq2_xxs",
            "op_m": 16,
            "op_n": 1,
            "op_k": 256,
            "suite_seed": 4711,
            "prompt_tokens": 512,
            "stage": "decode",
            "campaign_id": "c4-iq2",
            "workload_id": "iq2-op",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_parser_supports_generic_quant_ops(self):
        args = C.parser().parse_args([
            "--binary", "/tmp/test-backend-ops",
            "--source-root", "/tmp/source",
            "--output-dir", "/tmp/evidence",
            "--workload-kind", "quant-op",
            "--quant-type", "iq2_xxs",
        ])
        self.assertEqual(args.binary, "/tmp/test-backend-ops")
        self.assertEqual(args.quant_type, "iq2_xxs")

    def test_quant_op_command_is_exactly_shape_and_type_scoped(self):
        command = C.bench_command(
            Path("/bin/test-backend-ops"), None, repetitions=5,
            args=self.args())
        self.assertIn(
            r"^type_a=iq2_xxs,type_b=f32,m=16,n=1,k=256.*$", command)
        repeat_index = command.index("--repeat-suite")
        self.assertEqual(command[repeat_index + 1], "5")
        self.assertEqual(command[0], "/bin/test-backend-ops")

    def test_quant_type_is_not_a_shell_injection_surface(self):
        with self.assertRaisesRegex(RuntimeError, "letters, digits"):
            C.bench_command(
                Path("/bin/test-backend-ops"), None, repetitions=1,
                args=self.args(quant_type="q4_K; touch /tmp/no"))

    def test_op_manifest_carries_stage_and_quant_identity(self):
        receipt = type("Receipt", (), {
            "corpus_id": "c", "workload_id": "w", "profile_path": "p",
            "profile_sha256": "1" * 64, "source_commit": "deadbeef",
        })()
        capture = {
            "role": "mapping", "attribution_mode": "graphs_disabled",
            "warmup_steps": 10, "active_steps": 5, "receipt": receipt,
        }
        formal = dict(capture, role="formal",
                      attribution_mode="production_optimizations")
        manifest = C.manifest_for(
            capture, formal, args=self.args(), catalogue_hash="2" * 64)
        self.assertEqual(manifest["mapping"]["stage"], "decode")
        self.assertEqual(
            manifest["architecture_blocks"][0]["block_id"],
            "iq2_xxs-op-requantized-matvec")

    def test_artifact_inventory_hashes_files_and_excludes_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "capture.txt").write_text("trace\n", encoding="utf-8")
            (root / "receipt.json").write_text("{}\n", encoding="utf-8")
            rows = C.artifact_inventory(root)
        self.assertEqual([row["path"] for row in rows], ["capture.txt"])
        self.assertEqual(rows[0]["bytes"], 6)
        self.assertEqual(len(rows[0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
