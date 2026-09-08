"""Anti-drift tests for the Qwen3.8-Flash-Next canonical serving recipe.

INTENDED REPO PATH: epyc-inference-research/scripts/lib/test_qwen38_flash_next_recipe.py
Register in the Makefile's PYTEST_SMOKE list alongside test_recipes.py.

These tests are pure-Python and need no model, no binary and no region lock, EXCEPT
the two marked `needs_binary` / `needs_artifacts`, which skip when the champion tree
is not on this host.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_flash_next_recipe as R  # noqa: E402

HAVE_BIN = Path(R.CHAMPION_BINDIR, "llama-server").is_file()
HAVE_ART = Path(R.TRUNK_GGUF).is_file() and Path(R.MTP_HEAD_GGUF).is_file()


class TestTheFlagDefectThisRecipeExistsToPrevent(unittest.TestCase):
    """SYNC-10 lost seven MTP arms to `--fa 1`. These are the guards."""

    def test_recipe_never_emits_the_long_fa_form(self):
        cmd = R.build_serve_command()
        self.assertNotIn("--fa", cmd)
        self.assertIn("-fa", cmd)
        self.assertEqual(cmd[cmd.index("-fa") + 1], "on")

    def test_bad_flag_form_is_rejected_by_the_validator(self):
        with self.assertRaises(R.RecipeViolation):
            R.assert_no_bad_flag_forms(["llama-server", "--fa", "1"])

    def test_every_known_bad_form_has_a_stated_reason(self):
        for form, why in R.KNOWN_BAD_FLAG_FORMS.items():
            self.assertTrue(form.startswith("-"))
            self.assertGreater(len(why), 20, f"{form} needs a real explanation")

    @unittest.skipUnless(HAVE_BIN, "champion binary not on this host")
    def test_dry_run_accepts_every_flag_the_recipe_emits(self):
        R.assert_flag_forms_exist(str(Path(R.CHAMPION_BINDIR) / "llama-server"))

    @unittest.skipUnless(HAVE_BIN, "champion binary not on this host")
    def test_dry_run_actually_detects_a_bad_flag(self):
        """Mutation test: the checker must FAIL on a flag that does not exist,
        otherwise it is a check that passes for the wrong reason."""
        with self.assertRaises(R.RecipeViolation):
            R.assert_flag_forms_exist(
                str(Path(R.CHAMPION_BINDIR) / "llama-server"), forms=[["--fa", "1"]]
            )


class TestMtpIsNotOptional(unittest.TestCase):
    """OP-35: the MTP head is part of the MODEL. Every serving path carries it."""

    def test_default_command_carries_the_head(self):
        R.assert_mtp_present(R.build_serve_command())

    def test_head_is_the_shared_variant(self):
        self.assertIn("shared-Q8_0", R.MTP_HEAD_GGUF)
        self.assertEqual(R.MTP_HEAD_KIND, "shared-Q8_0")

    def test_rejected_heads_are_named_so_nobody_re_derives_them(self):
        self.assertIn("mtp-Qwen3.8-Flash-Next-Q8_0.gguf", R.MTP_HEAD_REJECTED)

    def test_missing_head_fails_closed(self):
        with self.assertRaises(R.RecipeViolation):
            R.assert_mtp_present(R.build_serve_command(mtp=False))

    def test_spec_parameters_are_the_measured_optimum(self):
        cmd = R.build_serve_command()
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "4")
        self.assertEqual(cmd[cmd.index("--spec-draft-p-min") + 1], "0.5")
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "draft-mtp")


class TestKvCache(unittest.TestCase):
    def test_kv_is_f16(self):
        R.assert_kv_f16(R.build_serve_command())

    def test_quantised_kv_is_rejected(self):
        with self.assertRaises(R.RecipeViolation):
            R.assert_kv_f16(["llama-server", "-ctk", "q8_0", "-ctv", "q8_0"])

    def test_the_reason_is_carried_with_the_decision(self):
        self.assertLess(R.MTP_ALPHA["b9_quantised_kv"],
                        R.MTP_ALPHA["speed_claim_b1_3_build10221_kvf16"])


class TestProcessWrapping(unittest.TestCase):
    def test_taskset_precedes_numactl(self):
        self.assertEqual(R.SERVE_PREFIX[0], "taskset")
        self.assertEqual(R.SERVE_PREFIX[3], "numactl")
        R.assert_canonical_prefix(R.build_serve_command())

    def test_reversed_prefix_is_rejected(self):
        with self.assertRaises(R.RecipeViolation):
            R.assert_canonical_prefix(
                ["numactl", "--interleave=all", "taskset", "-c", "0-95", "llama-server"]
            )

    def test_no_mmap_is_present(self):
        self.assertIn("--no-mmap", R.build_serve_command())

    def test_threads_are_48_not_96(self):
        self.assertEqual(R.THREADS, 48)


class TestOmpAndGgmlEnv(unittest.TestCase):
    def test_full_omp_stack_present(self):
        for k in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_DYNAMIC"):
            self.assertIn(k, R.CANONICAL_OMP_ENV)
        self.assertEqual(R.CANONICAL_OMP_ENV["OMP_WAIT_POLICY"], "active")

    def test_ggml_iqk_must_be_exported(self):
        """The kernels are compiled in but runtime-gated OFF. Omitting this measures
        a different kernel, silently."""
        self.assertEqual(R.CHAMPION_GGML_ENV["GGML_IQK"], "1")
        self.assertEqual(R.CHAMPION_KNOBS["GGML_IQK"][1], "export")

    def test_partial_env_is_rejected(self):
        env = dict(R.CANONICAL_OMP_ENV)
        env.update(R.CHAMPION_GGML_ENV)
        del env["OMP_DYNAMIC"]
        with self.assertRaises(R.RecipeViolation):
            R.assert_canonical_env(env)

    def test_setting_a_leave_unset_knob_is_rejected(self):
        env = {**R.CANONICAL_OMP_ENV, **R.CHAMPION_GGML_ENV, "GGML_VEC_Q8K": "0"}
        with self.assertRaises(R.RecipeViolation):
            R.assert_canonical_env(env)

    def test_ld_library_path_is_prepended_with_the_bindir(self):
        env = R.build_serve_env(bindir="/x/bin", base_env={"LD_LIBRARY_PATH": "/other"})
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/x/bin:"))

    def test_no_intel_openmp_vars(self):
        """The build is libgomp. KMP_* belongs to the ik_llama path and does not apply."""
        for k in {**R.CANONICAL_OMP_ENV, **R.CHAMPION_GGML_ENV}:
            self.assertFalse(k.startswith("KMP_"))
            self.assertFalse(k.startswith("GOMP_"))


class TestThpKnobsAreNotConflated(unittest.TestCase):
    """Conflating GGML_NOHUGEPAGE with GGML_NOHUGEPAGE_PROCESS discards 86.4% of the
    champion. They must remain two entries with opposite intended states."""

    def test_both_knobs_present_and_distinct(self):
        self.assertEqual(R.CHAMPION_KNOBS["GGML_NOHUGEPAGE"][0], "on")
        # CHAMP-2 ADOPTED 2026-09-08: the process shim is now EXPORTED =1.
        self.assertEqual(R.CHAMPION_KNOBS["GGML_NOHUGEPAGE_PROCESS"][0], "1")
        self.assertEqual(R.CHAMPION_KNOBS["GGML_NOHUGEPAGE_PROCESS"][1], "export")
        # ...and it must actually be in the env the recipe emits, not merely documented.
        self.assertEqual(R.CHAMPION_GGML_ENV["GGML_NOHUGEPAGE_PROCESS"], "1")
        # The two knobs must never collapse into one entry.
        self.assertNotIn("GGML_NOHUGEPAGE", R.CHAMPION_GGML_ENV)

    def test_thp_shim_carries_its_unit(self):
        """★ A floor/knob record without its UNIT is the 1200-fold error.

        CHAMP-2 is a SESSION-unit knob. The arm floor does not transfer to it.
        """
        self.assertEqual(R.THP_SHIM["knob"], "GGML_NOHUGEPAGE_PROCESS")
        self.assertIn("SESSION", R.THP_SHIM["unit"])
        self.assertIn("LAUNCH", R.THP_SHIM["set_at"])
        # Direction is claimed; magnitude is NOT.
        self.assertFalse(R.THP_SHIM["magnitude_claimed"])
        # The exact alpha was enumerated, not union-bounded.
        self.assertLess(R.THP_SHIM["alpha_exact_two_sided"], 0.05)
        self.assertGreater(R.THP_SHIM["alpha_union_bound_would_have_said"], 0.05)
        # Recipe change, not kernel change.
        self.assertEqual(R.THP_SHIM["binary_unchanged"], "ef81196d5")

    def test_shim_is_not_verified_by_the_invalid_discriminator(self):
        """★ Vacuous-instrument guard.

        AnonHugePages/Rss reads 0.06% at load and ~6% minutes later on the SAME
        process, so it cannot decide the shim's state. The authoritative check is
        THP_enabled in /proc/PID/status, read once per LAUNCH, fail-closed.
        """
        shim_check = R.PRECONDITIONS["thp_process_shim"]
        self.assertIn("THP_enabled", shim_check)
        self.assertIn("fail-closed", shim_check)
        self.assertIn("LAUNCH", shim_check)
        self.assertNotIn("AnonHugePages", shim_check)
        # The time-varying read must still exist, but named for what it can decide.
        self.assertIn("AnonHugePages", R.PRECONDITIONS["thp_readback_madvise_only"])
        self.assertNotIn("thp_readback", R.PRECONDITIONS)

    def test_every_floor_carries_its_unit(self):
        """★ A bare number is not a floor. 1200-fold error guard."""
        for name, (unit, sd, governs) in R.FLOORS.items():
            self.assertTrue(unit and isinstance(unit, str), name)
            self.assertTrue(any(u in unit for u in ("arm", "session", "launch")), name)
            self.assertIsInstance(sd, float, name)
            self.assertTrue(governs, name)
        # The arm floor and the session floor must not be the same number.
        self.assertNotEqual(R.FLOORS["arm_campaign"][1], R.FLOORS["session"][1])
        self.assertGreater(R.FLOORS["session"][1], R.FLOORS["arm_campaign"][1] * 5)


class TestArtifactIdentity(unittest.TestCase):
    def test_uniform_is_not_uniform(self):
        self.assertNotEqual(R.TRUNK_EFFECTIVE_BPW, 4.0)
        self.assertAlmostEqual(R.TRUNK_EFFECTIVE_BPW, 4.995, places=3)
        self.assertGreater(len(R.TRUNK_QUANT_CENSUS), 3)

    def test_bytes_per_token_matches_the_bpw_and_param_count(self):
        derived = R.TRUNK_ACTIVE_PARAMS_B * R.TRUNK_EFFECTIVE_BPW / 8
        self.assertAlmostEqual(derived, R.TRUNK_BYTES_PER_TOKEN_GB, places=2)

    def test_digests_are_real_sha256(self):
        for h in [R.TRUNK_SHA256, R.MTP_HEAD_SHA256, *R.CHAMPION_SHA256.values()]:
            self.assertEqual(len(h), 64)
            int(h, 16)

    @unittest.skipUnless(HAVE_ART, "artifacts not on this host")
    def test_artifacts_present_at_pinned_sizes(self):
        R.assert_artifacts_exist()


class TestKernelIdentity(unittest.TestCase):
    def test_champion_pinned_by_commit_and_build(self):
        self.assertEqual(len(R.CHAMPION_COMMIT), 40)
        self.assertEqual(R.CHAMPION_BUILD_NUMBER, 10241)

    @unittest.skipUnless(HAVE_BIN, "champion binary not on this host")
    def test_binary_reports_the_pinned_build(self):
        out = subprocess.run(
            [str(Path(R.CHAMPION_BINDIR) / "llama-server"), "--version"],
            capture_output=True, text=True,
        )
        blob = out.stdout + out.stderr
        self.assertIn(str(R.CHAMPION_BUILD_NUMBER), blob)
        self.assertIn(R.CHAMPION_COMMIT[:9], blob)

    @unittest.skipUnless(HAVE_BIN, "champion binary not on this host")
    def test_claimed_knob_defaults_are_the_compiled_ones(self):
        R.assert_knob_markers()

    @unittest.skipUnless(HAVE_BIN, "champion binary not on this host")
    def test_binary_digests_match(self):
        R.assert_binary_identity()


class TestHeadlineHygiene(unittest.TestCase):
    def test_both_headlines_name_their_binary_and_window(self):
        for name, h in R.HEADLINES.items():
            self.assertIn("binary", h, name)
            self.assertIn("window", h, name)

    def test_the_23_16_entry_is_not_attributed_to_the_champion(self):
        self.assertNotIn("10241", R.HEADLINES["speed_claim_aba"]["binary"])

    def test_canonical_headline_is_the_champion(self):
        """★ Was `assertEqual(CANONICAL_HEADLINE, "champion3")` -- a literal that
        became WRONG the moment the champion moved (fold, 2026-09-08) while still
        passing every other test. Assert the PROPERTY, not the string: the canonical
        headline must name the CURRENT champion and must not be a superseded entry.
        The standing rule is `the champion is always current`.
        """
        h = R.HEADLINES[R.CANONICAL_HEADLINE]
        self.assertNotIn("status", h,
                         "the canonical headline must not be a superseded entry")
        self.assertIn(R.CURRENT_CHAMPION["commit"], h["binary"])
        # And every superseded entry must SAY it is superseded, not just be unused.
        for name, entry in R.HEADLINES.items():
            if name == R.CANONICAL_HEADLINE:
                continue
            self.assertIn("status", entry,
                          f"{name} is not canonical and carries no supersession note")

    def test_the_pin_gap_is_declared_not_hidden(self):
        """★ The module names champion3's digests but ef81196d5 is the champion.
        That gap must be DECLARED and fail-closed, never papered over."""
        self.assertFalse(R.CHAMPION_PIN_RESOLVED)
        self.assertTrue(R.CHAMPION_PIN_GAP)
        self.assertIsNone(R.CURRENT_CHAMPION["build_number"])
        self.assertIsNone(R.CURRENT_CHAMPION["sha256"])
        # The prior champion is an ANCESTOR of the current one, not the champion.
        self.assertEqual(R.CURRENT_CHAMPION["lineages_by_ancestry"]["cpu_lineage"],
                         R.CHAMPION_COMMIT[:9])

    def test_do_not_fold_list_is_carried(self):
        """★ Folding a superseded decision is a failure ancestry cannot see."""
        self.assertIn("feature/tree-draft-v6", R.CURRENT_CHAMPION["do_not_fold"])
        joined = " ".join(R.CURRENT_CHAMPION["do_not_fold"])
        self.assertIn("sync17-fix2", joined)
        self.assertIn("retest1-fix1", joined)

    def test_workload_is_pinned(self):
        for k in ("prompts", "max_tokens", "temperature", "cache_prompt"):
            self.assertIn(k, R.WORKLOAD)


class TestProvenance(unittest.TestCase):
    def test_recipe_hashes_itself(self):
        h = R.recipe_sha256()
        self.assertEqual(len(h), 64)

    def test_edit_changes_the_hash(self):
        """A recipe whose provenance cannot be hashed is not a codified recipe."""
        import hashlib
        raw = Path(R.__file__).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), R.recipe_sha256())
        self.assertNotEqual(hashlib.sha256(raw + b"#").hexdigest(), R.recipe_sha256())


if __name__ == "__main__":
    unittest.main(verbosity=2)
