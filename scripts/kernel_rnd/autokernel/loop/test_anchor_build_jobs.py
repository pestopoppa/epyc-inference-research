#!/usr/bin/env python3
"""C46: champion (anchor + guard) builds are serial for HIP only.

R23-40 made every champion build `-j1` because `-j64` hipcc builds were not
reproducible. C46 (2026-09-27) proved the CPU/gcc recipe IS reproducible at -j64
(object digest identical across -j64, -j64 and the serial anchor-gen-002 build;
evidence in artifacts/c46-cpu-j64-reproducibility/), so the CPU recipe takes the
run's `build_jobs` while HIP stays at 1 until R23-41 resolves.

Each assertion names what it would read if the split were broken: a CPU recipe
reading 1 is the pre-C46 cost regression; a HIP recipe reading >1 re-opens the
Run-18 digest-abort fault class.
"""
from __future__ import annotations

import ast
import inspect
import unittest

from autokernel.controller import build_recipe
from autokernel.controller.build_recipe import BuildRecipe, Flag
from autokernel.loop import run as run_mod


class AnchorBuildJobs(unittest.TestCase):
    def test_cpu_recipe_takes_normal_build_jobs(self):
        self.assertEqual(run_mod.anchor_build_jobs(build_recipe.NATIVE_CPU_RECIPE, 64), 64)
        self.assertEqual(run_mod.anchor_build_jobs(build_recipe.NATIVE_CPU_RECIPE, 48), 48)

    def test_cpu_recipe_never_below_one(self):
        self.assertEqual(run_mod.anchor_build_jobs(build_recipe.NATIVE_CPU_RECIPE, 0), 1)

    def test_hip_recipe_stays_serial(self):
        # R23-41 unresolved: -j64 hipcc builds differ in every code section.
        self.assertEqual(run_mod.anchor_build_jobs(build_recipe.HOUSE_GPU_RECIPE, 64), 1)

    def test_unknown_backend_is_treated_as_hip(self):
        bare = BuildRecipe(name="no-hip-flag", flags=(
            Flag("GGML_NATIVE", "ON", None, "test"),))
        self.assertEqual(run_mod.anchor_build_jobs(bare, 64), 1)

    def test_build_champion_uses_the_split(self):
        # build_champion is a closure inside main(); pin that it routes -j through
        # anchor_build_jobs rather than a hardcoded width.
        node = next(node for node in ast.walk(ast.parse(inspect.getsource(run_mod.main)))
                    if isinstance(node, ast.FunctionDef) and node.name == "build_champion")
        call = next(node.value for node in node.body if isinstance(node, ast.Return))
        self.assertEqual(ast.unparse(call.func), "local_compiles")
        jobs = next(keyword.value for keyword in call.keywords if keyword.arg == "jobs")
        self.assertEqual(ast.unparse(jobs), "anchor_build_jobs(recipe, build_jobs)")


if __name__ == "__main__":
    unittest.main()
