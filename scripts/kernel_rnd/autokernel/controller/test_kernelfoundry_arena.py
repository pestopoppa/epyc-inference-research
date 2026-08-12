#!/usr/bin/env python3
"""Tests for the licensed KernelFoundry gfx90a AgentKernelArena port."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest

from . import arena_upstream_common as common
from . import kernelfoundry_arena as K


BLOCK_POINTER = """import triton
import triton.language as tl
@triton.jit
def kernel(x, y, M: tl.constexpr, N: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    p = tl.make_block_ptr(base=x, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    a = tl.load(p, boundary_check=(0, 1))
    tl.store(y, a)
"""

AUTOTUNE = """import triton
import triton.language as tl
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 128}, num_warps=4)], key=["n"])
@triton.jit
def kernel(x, y, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + offsets, mask=offsets < n)
    tl.store(y + offsets, tl.exp(values), mask=offsets < n)
"""


class FixtureModel:
    def __init__(self, *, workspace, budget):
        self.workspace = workspace
        self.budget = budget
        self.artifact_root = workspace / common.ARTIFACT_DIRNAME
        self.artifact_root.mkdir()
        self.calls = 0

    def call(self, prompt):
        self.asserted_prompt = prompt
        code = (BLOCK_POINTER, AUTOTUNE)[self.calls % 2]
        self.calls += 1
        return f"```triton\n{code}\n```"

    def identity(self):
        return {"cli": "fixture", "model": common.MODEL_ID,
                "call_count": self.calls}


class FixtureEvaluator:
    def __init__(self, *, workspace, arena_root):
        self.workspace = workspace
        self.arena_root = arena_root
        self.source_paths = ("kernel.py",)
        self.config = {"target_kernel_functions": ["kernel"]}
        self.calls = 0

    def definition(self, prompt):
        return prompt

    def evaluate(self, files):
        self.calls += 1
        return common.EvaluationRecord(
            passed=True, latency_ms=1.0,
            speedup=1.1 + self.calls / 10,
            log_excerpt="Arena fixture accepted",
            raw={"pass_compilation": True, "pass_correctness": True})

    def materialize_best(self):
        return None

    def receipt_fields(self):
        return {"evaluation_count": self.calls, "best_score": 2.0}


class KernelFoundryArenaTest(unittest.TestCase):
    def test_shared_arena_workspace_serializes_parallel_branch_evaluation(self):
        class Vendor:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def evaluate_kernel(self, workspace, config, baseline, logger, device):
                del config, baseline, logger, device
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(0.03)
                    (workspace / "kernel.py").read_text()
                    return {
                        "pass_compilation": True,
                        "pass_correctness": True,
                        "valid_optimized_cases": 1,
                        "best_optimized_execution_time": 1.0,
                        "average_speedup": 0.5,
                    }
                finally:
                    with self.lock:
                        self.active -= 1

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "kernel.py").write_text("starting\n")
            evaluator = object.__new__(common.ArenaWorkspaceEvaluator)
            evaluator.workspace = workspace
            evaluator.source_paths = ("kernel.py",)
            evaluator.vendor = Vendor()
            evaluator.config = {}
            evaluator.baseline_cases = {}
            evaluator.logger = None
            evaluator.best_files = {"kernel.py": b"starting\n"}
            evaluator.best_score = 1.0
            evaluator.last_record = None
            evaluator.evaluation_count = 0
            evaluator.broker_receipts = []
            def brokered(ordinal, candidate):
                evaluator._materialize(candidate)
                return evaluator.vendor.evaluate_kernel(
                    evaluator.workspace, {}, {}, None, None)
            evaluator._brokered_evaluation = brokered
            evaluator._evaluation_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(
                    evaluator.evaluate,
                    ({"kernel.py": "branch-a\n"}, {"kernel.py": "branch-b\n"}),
                ))
            self.assertTrue(all(result.passed for result in results))
            self.assertEqual(evaluator.evaluation_count, 2)
            self.assertEqual(evaluator.vendor.max_active, 1)
            self.assertEqual((workspace / "kernel.py").read_text(), "starting\n")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()
        (self.workspace / "kernel.py").write_text(
            BLOCK_POINTER, encoding="utf-8")
        self.arena = Path(self.tmp.name) / "arena"
        self.arena.mkdir()

    def test_campaign_identity_and_validation_fail_closed(self):
        argv = K.campaign_argv("python-fixture")
        self.assertEqual(argv[0], "python-fixture")
        self.assertEqual(argv[argv.index("--checkpoint-hours") + 1], "32")
        self.assertEqual(argv[argv.index("--branches-per-iteration") + 1], "4")
        with self.assertRaisesRegex(K.KernelFoundryArenaError, r"\[1, 256\]"):
            K.run_controller(
                prompt="optimize", workspace=self.workspace,
                arena_root=self.arena, source_root=K.DEFAULT_SOURCE_ROOT,
                budget=common.ControllerBudget(2, 7200), max_iterations=0)
        with self.assertRaisesRegex(K.KernelFoundryArenaError, r"\[2, 16\]"):
            K.run_controller(
                prompt="optimize", workspace=self.workspace,
                arena_root=self.arena, source_root=K.DEFAULT_SOURCE_ROOT,
                budget=common.ControllerBudget(2, 7200), max_iterations=2,
                branches_per_iteration=1)

    def test_pinned_upstream_retains_map_elites_and_qd_transitions(self):
        marker = "AUTOKERNEL_KF_REAL_UPSTREAM_TEST"
        if os.environ.get(marker) != "1":
            if not K.RUNTIME_PYTHON.is_file() or not K.DEFAULT_SOURCE_ROOT.is_dir():
                self.skipTest("pinned KernelFoundry runtime is not installed")
            env = os.environ.copy()
            env[marker] = "1"
            completed = subprocess.run(
                [str(K.RUNTIME_PYTHON), "-m", "unittest",
                 f"{__name__}.{type(self).__name__}."
                 "test_pinned_upstream_retains_map_elites_and_qd_transitions"],
                cwd=Path(__file__).resolve().parents[2], env=env,
                text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0,
                             msg=completed.stdout + "\n" + completed.stderr)
            return

        receipt = K.run_controller(
            prompt="optimize", workspace=self.workspace,
            arena_root=self.arena, source_root=K.DEFAULT_SOURCE_ROOT,
            budget=common.ControllerBudget(2, 7200), max_iterations=2,
            branches_per_iteration=2, model_factory=FixtureModel,
            evaluator_factory=FixtureEvaluator)
        extra = receipt["extra"]
        self.assertEqual(extra["upstream_callable"], "Controller.run_single")
        self.assertGreaterEqual(extra["triton_patterns_added_to_cache"], 150)
        self.assertGreaterEqual(extra["map_elites_occupied_cells"], 2)
        self.assertEqual(extra["map_elites_program_count"], 4)
        self.assertTrue(extra["qd_gradient_enabled"])
        self.assertGreaterEqual(extra["qd_transition_count"], 2)
        self.assertGreaterEqual(extra["qd_unique_directions"], 1)
        self.assertEqual(receipt["model"]["call_count"], 4)


if __name__ == "__main__":
    unittest.main()
