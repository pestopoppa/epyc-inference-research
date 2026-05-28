"""Unit tests for canonical_recipe.py validators.

Exercises each drift-trap with both negative paths (drifted state → violation
raised) and positive sanity checks (current state → passes). Run with:

    python3 -m pytest scripts/lib/test_canonical_recipe.py -v

Or standalone:

    python3 scripts/lib/test_canonical_recipe.py

These tests catch the drift scenarios that bit the project on 2026-05-02 and
2026-05-28 (each documented in canonical_recipe.py's module docstring).
"""

from __future__ import annotations

import os
import sys
import unittest

# Allow running directly OR via pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_recipe as r


class TestCanonicalCmd(unittest.TestCase):
    """Tests for assert_canonical_cmd — catches drift-traps 1+2."""

    def test_correct_cmd_passes(self):
        cmd = list(r.CANONICAL_PREFIX) + ["/path/to/llama-bench", "-mmp", "0", "-m", "foo"]
        r.assert_canonical_cmd(cmd)  # should not raise

    def test_no_mmap_flag_also_passes(self):
        cmd = list(r.CANONICAL_PREFIX) + ["/path/to/llama-bench", "--no-mmap", "-m", "foo"]
        r.assert_canonical_cmd(cmd)  # should not raise

    def test_missing_prefix_raises(self):
        cmd = ["/path/to/llama-bench", "-mmp", "0", "-m", "foo"]
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.assert_canonical_cmd(cmd)
        self.assertIn("canonical prefix", str(ctx.exception))
        self.assertIn("apply_canonical_prefix", str(ctx.exception))

    def test_wrong_prefix_order_raises(self):
        # numactl BEFORE taskset is wrong (taskset must wrap numactl)
        cmd = ["numactl", "--interleave=all", "taskset", "-c", "0-95", "x", "-mmp", "0"]
        with self.assertRaises(r.CanonicalRecipeViolation):
            r.assert_canonical_cmd(cmd)

    def test_missing_mmap_flag_raises(self):
        cmd = list(r.CANONICAL_PREFIX) + ["/path/to/llama-bench", "-m", "foo"]
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.assert_canonical_cmd(cmd)
        self.assertIn("--no-mmap", str(ctx.exception))


class TestCanonicalEnv(unittest.TestCase):
    """Tests for assert_canonical_env — catches drift-traps 3+4."""

    def _good_env(self) -> dict[str, str]:
        env = {"LD_LIBRARY_PATH": f"{r.LLVM20_LIBDIR}:/other/path"}
        env.update(r.CANONICAL_OMP_ENV)
        return env

    def test_correct_env_passes(self):
        r.assert_canonical_env(self._good_env())  # should not raise

    def test_missing_omp_dynamic_raises(self):
        env = self._good_env()
        del env["OMP_DYNAMIC"]
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.assert_canonical_env(env)
        self.assertIn("OMP_DYNAMIC", str(ctx.exception))

    def test_wrong_omp_value_raises(self):
        env = self._good_env()
        env["OMP_WAIT_POLICY"] = "passive"  # wrong
        with self.assertRaises(r.CanonicalRecipeViolation):
            r.assert_canonical_env(env)

    def test_missing_llvm20_on_ld_path_raises(self):
        env = self._good_env()
        env["LD_LIBRARY_PATH"] = "/some/other/path"  # no LLVM20
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.assert_canonical_env(env)
        self.assertIn(r.LLVM20_LIBDIR, str(ctx.exception))
        self.assertIn("AOCC", str(ctx.exception))


class TestBinaryResolution(unittest.TestCase):
    """Tests for assert_binary_resolves_correctly — catches drift-trap 5
    (the 2026-05-28 RUNPATH bug). These tests require the ik_llama build to
    exist on disk; skipped otherwise.
    """

    def test_ik_llama_bench_resolves_correctly(self):
        if not os.path.isfile(r.IK_LLAMA_BENCH):
            self.skipTest("ik_llama bench binary not built")
        # If this raises, the RUNPATH bug has regressed. Rebuild with
        # -Wl,--disable-new-dtags. The error message itself contains the fix.
        r.assert_binary_resolves_correctly(r.IK_LLAMA_BENCH, r.EXPECTED_LIBS_IK_LLAMA)

    def test_nonexistent_binary_raises(self):
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.assert_binary_resolves_correctly("/nonexistent/llama-bench", [])
        self.assertIn("not found", str(ctx.exception))

    def test_wrong_expected_libs_raises(self):
        if not os.path.isfile(r.IK_LLAMA_BENCH):
            self.skipTest("ik_llama bench binary not built")
        # Lie about the expected path — should report the actual resolution
        fake_expected = ["/totally/wrong/path/libllama.so", "/wrong/libggml.so"]
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.assert_binary_resolves_correctly(r.IK_LLAMA_BENCH, fake_expected)
        # The error message must point at the disable-new-dtags fix
        self.assertIn("disable-new-dtags", str(ctx.exception))


class TestBinaryDiscovery(unittest.TestCase):
    """Tests for discover_canonical_bench_binary + discover_v4_fork_bench."""

    def test_discovery_returns_ik_llama_by_default(self):
        if not os.path.isfile(r.IK_LLAMA_BENCH):
            self.skipTest("ik_llama bench binary not built")
        binary, libs = r.discover_canonical_bench_binary()
        self.assertEqual(binary, r.IK_LLAMA_BENCH)
        self.assertEqual(libs, r.EXPECTED_LIBS_IK_LLAMA)

    def test_v4_fork_discovery_when_built(self):
        if not os.path.isfile(r.V4_FORK_BENCH):
            self.skipTest("V4 fork bench binary not built")
        binary, libs = r.discover_v4_fork_bench()
        self.assertEqual(binary, r.V4_FORK_BENCH)
        self.assertEqual(libs, r.EXPECTED_LIBS_V4_FORK)

    def test_v4_fork_discovery_raises_when_unbuilt(self):
        # Negative case: if the binary doesn't exist, FileNotFoundError with
        # rebuild instructions
        if os.path.isfile(r.V4_FORK_BENCH):
            self.skipTest(
                "V4 fork is built; this test only meaningful pre-build."
            )
        with self.assertRaises(FileNotFoundError) as ctx:
            r.discover_v4_fork_bench()
        # Error message must include the rebuild command
        self.assertIn("--disable-new-dtags", str(ctx.exception))

    def test_v4_fork_does_not_appear_in_default_discovery(self):
        # discover_canonical_bench_binary must NOT return V4_FORK_BENCH because
        # the V4 fork binary doesn't support other archs.
        if not os.path.isfile(r.IK_LLAMA_BENCH) and not os.path.isfile(
            r.V5_CLEAN_BENCH
        ):
            self.skipTest("no default-discovery binary built")
        binary, _ = r.discover_canonical_bench_binary()
        self.assertNotEqual(binary, r.V4_FORK_BENCH)


class TestHostEnvironment(unittest.TestCase):
    """Tests for validate_host_environment — catches drift-traps 6-9."""

    def test_parse_thp_active(self):
        self.assertEqual(r._parse_thp_active("[always] madvise never"), "always")
        self.assertEqual(r._parse_thp_active("always [madvise] never"), "madvise")
        self.assertEqual(r._parse_thp_active("always madvise [never]"), "never")
        self.assertIsNone(r._parse_thp_active("garbage"))

    def test_validate_host_passes_or_reports_clearly(self):
        """Either the host is in canonical state (no exception), or the exception
        message names every drift with a fix line. This test is sensitive to
        live host state — what we're checking is that the FAILURE PATH produces
        actionable output.
        """
        try:
            r.validate_host_environment()
        except r.CanonicalRecipeViolation as e:
            msg = str(e)
            # Every drift block must include a 'fix:' line
            self.assertIn("fix:", msg)
            # The remediation pointer must be present
            self.assertIn("orchestrator_stack.py", msg)


class TestComposite(unittest.TestCase):
    """Tests for validate_canonical_env — the all-in-one composite."""

    def test_skip_all_passes(self):
        # No args + check_host=False → no-op, no exception
        r.validate_canonical_env(check_host=False)

    def test_binary_without_expected_libs_raises_value_error(self):
        with self.assertRaises(ValueError):
            r.validate_canonical_env(binary="/some/path", expected_libs=None, check_host=False)


class TestBuildCanonicalBenchCommand(unittest.TestCase):
    """End-to-end test of the high-level constructor."""

    def test_missing_model_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            r.build_canonical_bench_command(model="/nonexistent/model.gguf")

    def test_emitted_command_passes_validators(self):
        # Use any real GGUF; gemma4 if available
        candidates = [
            "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
        ]
        model = next((m for m in candidates if os.path.isfile(m)), None)
        if model is None:
            self.skipTest("no test model available")

        binary, cmd, env = r.build_canonical_bench_command(model=model, n_gen=8, reps=1)

        # The output must satisfy its own validators
        r.assert_canonical_cmd(cmd)
        r.assert_canonical_env(env)
        # cmd must include the binary as the executable
        self.assertIn(binary, cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
