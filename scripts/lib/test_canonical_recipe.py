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
import tempfile
import unittest
from unittest import mock

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
        """Baseline OMP-only canonical env — matches stack_env.py's
        build_launch_env() output for non-V4 production roles.
        """
        env = {"LD_LIBRARY_PATH": f"{r.LLVM20_LIBDIR}:/other/path"}
        env.update(r.CANONICAL_OMP_ENV)
        return env

    def _good_env_v4(self) -> dict[str, str]:
        """Baseline + V4 §Throughput gate extras — for V4-fork bench/runner."""
        env = self._good_env()
        env.update(r.V4_GATE_EXTRA_ENV)
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

    def test_default_assert_does_not_require_v4_extras(self):
        """Critical alignment with orchestrator stack_env.py: the default
        assert_canonical_env must NOT require V4_GATE_EXTRA_ENV. stack_env.py
        deliberately excludes GGML_NUMA_WEIGHTS for the worker role and only
        applies KMP_BLOCKTIME via a separate worker_pool launch branch.
        Requiring these here would falsely flag every non-V4 production env
        as drifted.
        """
        env = self._good_env()  # OMP only, no V4 extras
        r.assert_canonical_env(env)  # should not raise

    def test_v4_gate_missing_kmp_blocktime_raises(self):
        # V4 §Throughput gate requires KMP_BLOCKTIME=10 — opt in via require_v4_gate_extras
        env = self._good_env_v4()
        del env["KMP_BLOCKTIME"]
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.assert_canonical_env(env, require_v4_gate_extras=True)
        self.assertIn("KMP_BLOCKTIME", str(ctx.exception))

    def test_v4_gate_missing_ggml_numa_weights_raises(self):
        # V4 §Throughput gate requires GGML_NUMA_WEIGHTS=1
        env = self._good_env_v4()
        del env["GGML_NUMA_WEIGHTS"]
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.assert_canonical_env(env, require_v4_gate_extras=True)
        self.assertIn("GGML_NUMA_WEIGHTS", str(ctx.exception))

    def test_build_canonical_env_default_omits_v4_extras(self):
        """Default build_canonical_env() must NOT emit V4 gate extras —
        preserves comparability with the documented raw Coder-30B tripwire
        baseline AND with stack_env.py's non-V4 launch env.
        """
        env = r.build_canonical_env()
        for k in r.V4_GATE_EXTRA_ENV:
            self.assertNotIn(k, env,
                             f"default build_canonical_env() must not set {k}")

    def test_build_canonical_env_v4_opt_in_emits_extras(self):
        env = r.build_canonical_env(use_v4_gate_extras=True)
        for k, v in r.V4_GATE_EXTRA_ENV.items():
            self.assertEqual(env.get(k), v,
                             f"V4 opt-in build_canonical_env() must set {k}={v}")

    def test_explicit_library_path_is_first_and_iqk_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_lib = os.path.join(tmp, "build", "bin")
            os.makedirs(candidate_lib)
            with mock.patch.dict(
                os.environ,
                {"LD_LIBRARY_PATH": "/mnt/raid0/llm/llama.cpp/build/bin:/opt/rocm/lib"},
            ):
                env = r.build_canonical_env(
                    extra_vars={"GGML_IQK": "0"},
                    library_path=candidate_lib,
                )

        self.assertEqual(
            env["LD_LIBRARY_PATH"].split(":")[0],
            os.path.realpath(candidate_lib),
        )
        self.assertIn(r.LLVM20_LIBDIR, env["LD_LIBRARY_PATH"].split(":"))
        self.assertEqual(env["GGML_IQK"], "0")


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

    def test_not_found_dep_fails_closed(self):
        """If ldd reports `<libname> => not found`, validation must FAIL — the
        pre-fix code skipped on no-regex-match which let truly-broken binaries
        pass. Build a tiny fake binary whose ldd output we mock via a wrapper
        script, then point assert_binary_resolves_correctly at it.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Create a fake binary that's executable but whose ldd output is
            # mocked via the LD_TRACE_LOADED_OBJECTS pattern. Simpler: build a
            # tiny C program that has an unresolved dep, OR mock ldd via PATH.
            # Simplest: write a shell script that mimics ldd's output for a
            # binary we point at, by intercepting ldd via PATH.
            fake_binary = os.path.join(tmp, "fake-bench")
            # Create an actual executable file so the os.path.isfile check passes.
            with open(fake_binary, "w") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(fake_binary, 0o755)
            # Create a fake `ldd` in tmp that emits "=> not found" for our libs.
            fake_ldd = os.path.join(tmp, "ldd")
            with open(fake_ldd, "w") as f:
                f.write(
                    "#!/bin/sh\n"
                    "echo '\tlibllama.so => not found'\n"
                    "echo '\tlibggml.so => not found'\n"
                )
            os.chmod(fake_ldd, 0o755)
            # Prepend tmp to PATH so our fake ldd wins
            old_path = os.environ.get("PATH", "")
            try:
                os.environ["PATH"] = f"{tmp}:{old_path}"
                with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
                    r.assert_binary_resolves_correctly(
                        fake_binary,
                        ["/expected/libllama.so", "/expected/libggml.so"],
                    )
                self.assertIn("not found", str(ctx.exception))
                self.assertIn("UNRESOLVED", str(ctx.exception))
            finally:
                os.environ["PATH"] = old_path


class TestExplicitBinaryIdentity(unittest.TestCase):
    def _make_candidate(self, tmp: str) -> tuple[str, str, str]:
        source_root = os.path.join(tmp, "llama.cpp-experimental")
        library_path = os.path.join(source_root, "build", "bin")
        binary = os.path.join(library_path, "llama-bench")
        os.makedirs(library_path)
        with open(binary, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(binary, 0o755)
        return source_root, library_path, binary

    def test_candidate_identity_accepts_libraries_from_selected_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root, library_path, binary = self._make_candidate(tmp)
            env = {"LD_LIBRARY_PATH": f"{library_path}:{r.LLVM20_LIBDIR}"}
            ldd_output = "\n".join(
                [
                    f"libllama-bench-impl.so => {library_path}/libllama-bench-impl.so",
                    f"libllama.so.0 => {library_path}/libllama.so.0",
                    f"libggml-cpu.so.0 => {library_path}/libggml-cpu.so.0",
                ]
            )

            def fake_check_output(argv, **kwargs):
                if argv[0] == "git":
                    return f"{source_root}\n"
                if argv[0] == "ldd":
                    self.assertEqual(kwargs["env"], env)
                    return ldd_output
                self.fail(f"unexpected command: {argv}")

            with mock.patch.object(
                r.subprocess, "check_output", side_effect=fake_check_output
            ):
                r.assert_explicit_bench_identity(
                    binary, source_root, library_path, env
                )

    def test_candidate_identity_rejects_production_library_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root, library_path, binary = self._make_candidate(tmp)
            env = {"LD_LIBRARY_PATH": f"{library_path}:{r.LLVM20_LIBDIR}"}
            production_lib = "/mnt/raid0/llm/llama.cpp/build/bin/libggml-cpu.so.0"

            def fake_check_output(argv, **kwargs):
                if argv[0] == "git":
                    return f"{source_root}\n"
                if argv[0] == "ldd":
                    return f"libggml-cpu.so.0 => {production_lib} (0x1234)\n"
                self.fail(f"unexpected command: {argv}")

            with mock.patch.object(
                r.subprocess, "check_output", side_effect=fake_check_output
            ):
                with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
                    r.assert_explicit_bench_identity(
                        binary, source_root, library_path, env
                    )

        self.assertIn("outside --library-path", str(ctx.exception))
        self.assertIn("mixed candidate/production", str(ctx.exception))


class TestBinaryDiscovery(unittest.TestCase):
    """Tests for discover_canonical_bench_binary + discover_v4_fork_bench."""

    def test_discovery_returns_production_consolidated_by_default(self):
        if not os.path.isfile(r.V6_IQK_BENCH):
            self.skipTest("production consolidated bench binary not built")
        binary, libs = r.discover_canonical_bench_binary()
        self.assertEqual(binary, r.V6_IQK_BENCH)
        self.assertEqual(libs, r.EXPECTED_LIBS_V6_IQK)

    def test_v4_fork_discovery_is_retired(self):
        with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
            r.discover_v4_fork_bench()
        self.assertIn("--v4-fork is retired", str(ctx.exception))
        self.assertIn("--binary/--source-root/--library-path", str(ctx.exception))

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

    def test_explicit_arm_requires_complete_identity_tuple(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model:
            with self.assertRaises(r.CanonicalRecipeViolation) as ctx:
                r.build_canonical_bench_command(
                    model=model.name,
                    binary="/tmp/build/bin/llama-bench",
                )
        self.assertIn("must be supplied together", str(ctx.exception))

    def test_explicit_arm_threads_identity_and_iqk_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root = os.path.join(tmp, "llama.cpp-experimental")
            library_path = os.path.join(source_root, "build", "bin")
            binary_path = os.path.join(library_path, "llama-bench")
            model_path = os.path.join(tmp, "model.gguf")
            os.makedirs(library_path)
            for path in (binary_path, model_path):
                with open(path, "w") as f:
                    f.write("")
            os.chmod(binary_path, 0o755)

            with mock.patch.object(
                r, "assert_explicit_bench_identity"
            ) as identity_check:
                binary, cmd, env = r.build_canonical_bench_command(
                    model=model_path,
                    binary=binary_path,
                    source_root=source_root,
                    library_path=library_path,
                    ggml_iqk="0",
                    n_gen=8,
                    reps=1,
                )

        self.assertEqual(binary, os.path.realpath(binary_path))
        self.assertIn(binary, cmd)
        self.assertEqual(env["GGML_IQK"], "0")
        self.assertEqual(
            env["LD_LIBRARY_PATH"].split(":")[0],
            os.path.realpath(library_path),
        )
        identity_check.assert_called_once_with(
            os.path.realpath(binary_path),
            os.path.realpath(source_root),
            os.path.realpath(library_path),
            env,
        )


class TestEmitBenchCommandCli(unittest.TestCase):
    def test_explicit_options_are_parsed_and_forwarded(self):
        captured = {}

        def fake_emit(args):
            captured.update(vars(args))
            return 0

        argv = [
            "canonical_recipe.py",
            "emit-bench-command",
            "--model",
            "/tmp/model.gguf",
            "--binary",
            "/candidate/build/bin/llama-bench",
            "--source-root",
            "/candidate",
            "--library-path",
            "/candidate/build/bin",
            "--ggml-iqk",
            "0",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            r, "_emit_bench_command_json", side_effect=fake_emit
        ):
            self.assertEqual(r._main(), 0)

        self.assertEqual(captured["binary"], "/candidate/build/bin/llama-bench")
        self.assertEqual(captured["source_root"], "/candidate")
        self.assertEqual(captured["library_path"], "/candidate/build/bin")
        self.assertEqual(captured["ggml_iqk"], "0")


class TestQ8Subgate(unittest.TestCase):
    def test_q8_subgate_is_explicit_and_rejects_other_values(self):
        with tempfile.NamedTemporaryFile() as model:
            with mock.patch.object(r, "discover_canonical_bench_binary", return_value=("/bench", [])):
                _binary, _cmd, env = r.build_canonical_bench_command(
                    model=model.name, ggml_iqk_q8_0="1"
                )
        self.assertEqual(env["GGML_IQK_Q8_0"], "1")
        with tempfile.NamedTemporaryFile() as model:
            with self.assertRaises(r.CanonicalRecipeViolation):
                r.build_canonical_bench_command(model=model.name, ggml_iqk_q8_0="0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
