#!/usr/bin/env python3
"""Tests for the fail-closed GEAK/AgentKernelArena gfx90a seam."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import arena_adapter as A


class ArenaAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def task(self, **overrides):
        payload = dict(
            task_id="hip2hip/silu",
            task_prompt="Optimize the supplied HIP kernel without changing semantics.",
            workspace=str(self.workspace),
            controller_id="geak_v1",
            round_id="arena-r1",
            actual_gfx_arch="gfx90a",
        )
        payload.update(overrides)
        return A.ArenaTask(**payload)

    def executable(self, name: str, body: str) -> Path:
        path = self.root / name
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_all_declared_controller_families_are_registered(self):
        self.assertEqual(
            set(A.CONTROLLERS),
            {"claude_codex_actor_critic", "evoengineer", "kernelfoundry",
             "k_search", "xe_forge", "geak_v1", "argus"},
        )
        self.assertTrue(all(row.evidence_scope == "whole_agent_task_only"
                            for row in A.CONTROLLERS.values()))

    def test_prepare_binds_gfx90a_and_runs_final_prompt_hygiene(self):
        prepared = A.prepare_task(self.task(), base_environment={"PATH": os.environ["PATH"]})
        self.assertIn("MI210, CDNA2 gfx90a", prepared.prompt)
        self.assertEqual(prepared.environment["PYTORCH_ROCM_ARCH"], "gfx90a")
        self.assertEqual(prepared.environment["AMDGPU_TARGETS"], "gfx90a")
        self.assertEqual(prepared.environment["GPU_TARGETS"], "gfx90a")
        self.assertEqual(
            prepared.prompt_sha256,
            hashlib.sha256(prepared.prompt.encode()).hexdigest(),
        )
        with self.assertRaisesRegex(Exception, "sealed evaluator internals"):
            A.prepare_task(self.task(task_prompt="inspect max_nmse_err"))

    def test_wrong_arch_and_spoof_override_refuse(self):
        with self.assertRaisesRegex(A.ArenaAdapterError, "requires gfx90a"):
            self.task(actual_gfx_arch="gfx942")
        with self.assertRaisesRegex(A.ArenaAdapterError, "falsify"):
            A.architecture_environment({"HSA_OVERRIDE_GFX_VERSION": "9.0.10"})
        with self.assertRaisesRegex(A.ArenaAdapterError, "conflicts"):
            A.architecture_environment({"AMDGPU_TARGETS": "gfx942"})

    def test_detect_arch_strips_feature_suffix_and_rejects_other_family(self):
        good = self.executable("good-enumerator", "#!/bin/sh\necho gfx90a:sramecc+:xnack-\n")
        observed = A.detect_gfx_arch(str(good))
        self.assertEqual(observed["architectures"], ["gfx90a"])
        bad = self.executable("bad-enumerator", "#!/bin/sh\necho gfx942\n")
        with self.assertRaisesRegex(A.ArenaAdapterError, "expected exactly one"):
            A.detect_gfx_arch(str(bad))

    def test_launch_uses_stdin_no_shell_and_refuses_failed_controller(self):
        prepared = A.prepare_task(self.task(), base_environment={"PATH": os.environ["PATH"]})
        output = A.launch(
            prepared,
            (sys.executable, "-c", "import sys; print(sys.stdin.read().splitlines()[0])"),
            timeout_seconds=5,
        )
        self.assertEqual(output.strip(), "AUTOKERNEL AUTHORING ROLE: actor")
        with self.assertRaisesRegex(A.ArenaAdapterError, "exited 7"):
            A.launch(prepared, (sys.executable, "-c", "raise SystemExit(7)"),
                     timeout_seconds=5)

    def test_vendor_shape_registration_reaches_the_hygienic_launcher(self):
        registry = {}

        def register(name):
            def decorate(function):
                registry[name] = function
                return function
            return decorate

        enumerator = self.executable(
            "arena-enumerator", "#!/bin/sh\necho gfx90a:sramecc+:xnack-\n")
        launcher = A.register_agentkernelarena_adapter(
            register,
            lambda config, task_dir, workspace: "Optimize this public HIP task.",
        )
        self.assertIs(registry["epyc_autokernel"], launcher)
        output = launcher({"epyc_autokernel": {
            "controller_id": "claude_codex_actor_critic",
            "argv": [sys.executable, "-c",
                     "import sys; print(sys.stdin.read().splitlines()[0])"],
            "timeout_seconds": 5,
            "enumerator": str(enumerator),
            "round_id": "arena-ab-r1",
        }}, str(self.root / "task-config"), str(self.workspace))
        self.assertEqual(output.strip(), "AUTOKERNEL AUTHORING ROLE: actor")

    def test_c4_context_is_hash_bound(self):
        report = self.root / "c4.json"
        receipt = {"source_commit": "deadbeef", "profile_sha256": "2" * 64}
        payload = {
            "schema": "epyc.autokernel.c4_profile_report.v1",
            "manifest_sha256": "1" * 64,
            "comparison_id": "c4-q4", "stage": "decode",
            "capture_protocol": {
                "mapping": {"receipt": dict(receipt)},
                "formal": {"receipt": dict(receipt)},
            },
            "kernel_table": [{"kernel_family": "mul_mat_vec_q", "dispatches": 1,
                              "duration_ns": 10, "gpu_time_share": 1.0}],
            "overlap_opportunity_table": [], "fuse_pattern_table": [],
            "architecture_shape_table": [], "coverage_gaps": [],
        }
        report.write_text(json.dumps(payload), encoding="utf-8")
        digest = hashlib.sha256(report.read_bytes()).hexdigest()
        prepared = A.prepare_task(self.task(
            c4_report_path=str(report), c4_report_sha256=digest))
        self.assertIn(f"c4-profile://{digest}", prepared.prompt)
        with self.assertRaisesRegex(Exception, "hash mismatch"):
            A.prepare_task(self.task(
                c4_report_path=str(report), c4_report_sha256="f" * 64))

    def test_c5_reference_seed_is_bound_into_the_priced_task_context(self):
        prepared = A.prepare_task(self.task(c5_seed_ids=("k175", "k225")))
        self.assertIn("hyra-c5://", prepared.prompt)
        self.assertIn("hyra-sol-execbench/k175", prepared.prompt)
        self.assertIn("hyra-sol-execbench/k225", prepared.prompt)
        self.assertIn("re_author_and_re_attest", prepared.prompt)
        self.assertNotIn("sol_score", prepared.prompt)
        with self.assertRaisesRegex(Exception, "unknown C5 seed"):
            A.prepare_task(self.task(c5_seed_ids=("k999",)))

    def test_vendor_source_identity_is_exact_and_clean(self):
        pin = A.VendorPin("fixture", "a" * 40, "LICENSE", ("required.py",))
        (self.root / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
        (self.root / "required.py").write_text("pass\n", encoding="utf-8")
        with mock.patch.object(A, "_git", side_effect=("a" * 40, "")):
            result = A.inspect_vendor_source(self.root, pin)
        self.assertTrue(result["clean"])
        self.assertEqual(result["commit"], "a" * 40)
        with mock.patch.object(A, "_git", side_effect=("b" * 40,)):
            with self.assertRaisesRegex(A.ArenaAdapterError, "expected commit"):
                A.inspect_vendor_source(self.root, pin)
        with mock.patch.object(A, "_git", side_effect=("a" * 40, " M required.py")):
            with self.assertRaisesRegex(A.ArenaAdapterError, "not clean"):
                A.inspect_vendor_source(self.root, pin)

    def test_preflight_receipt_write_is_atomic_and_exact(self):
        receipt = {"schema": A.PREFLIGHT_SCHEMA, "receipt_sha256": "a" * 64}
        output = self.root / "evidence" / "receipt.json"
        self.assertEqual(A.write_preflight_receipt(output, receipt), output.resolve())
        self.assertEqual(json.loads(output.read_text()), receipt)
        self.assertEqual(list(output.parent.glob(".receipt.json.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
