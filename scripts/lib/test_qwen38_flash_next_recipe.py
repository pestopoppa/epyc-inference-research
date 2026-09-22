"""Anti-drift tests for the Qwen3.8-Flash-Next canonical serving recipe.

REPO PATH: epyc-inference-research/scripts/lib/test_qwen38_flash_next_recipe.py
Registered in the Makefile's PYTEST_SMOKE list (landed 2026-09-14 by WRAP-10).

These tests are pure-Python and need no model, no binary and no region lock, EXCEPT
the two marked `needs_binary` / `needs_artifacts`, which skip when the champion tree
is not on this host.

RUN IT THROUGH THE REPO RUNNER: `make test`, or
`uv run --with pytest --with pyyaml pytest -q scripts/lib/test_qwen38_flash_next_recipe.py`.
A bare `python3 -m pytest` from the repo root fails at COLLECTION, not in any test:
`scripts/lib/` is a package whose `__init__.py` pulls in `requests`, so pytest's default
prepend import mode executes that first. `uv run` provides the project's declared
dependencies and collection succeeds.
"""
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


class TestTheRecipeIsPerSurface(unittest.TestCase):
    """★ 2026-09-08 operator ruling: the champion is commit + a PER-SURFACE recipe.

    CPU decode exports GGML_NOHUGEPAGE_PROCESS=1; GPU serving must NOT (R23-58, 48
    launches, bounded null). Carrying one surface's recipe onto the other is the defect.
    """

    def test_this_module_declares_its_surface(self):
        self.assertEqual(R.SURFACE, "cpu-decode")
        self.assertIn(R.SURFACE, R.SURFACE_RECIPES)
        self.assertEqual(R.CURRENT_CHAMPION["recipe_surface"], R.SURFACE)

    def test_the_shim_is_on_for_this_surface_and_absent_for_the_other(self):
        self.assertEqual(
            R.SURFACE_RECIPES["cpu-decode"]["GGML_NOHUGEPAGE_PROCESS"], "1")
        self.assertIsNone(
            R.SURFACE_RECIPES["gpu-serving"]["GGML_NOHUGEPAGE_PROCESS"])

    def test_the_emitted_env_matches_this_surface_not_the_other(self):
        want = R.SURFACE_RECIPES[R.SURFACE]["GGML_NOHUGEPAGE_PROCESS"]
        self.assertEqual(R.CHAMPION_GGML_ENV["GGML_NOHUGEPAGE_PROCESS"], want)

    def test_the_gpu_null_is_recorded_with_its_evidence_not_just_asserted(self):
        status = R.SURFACE_RECIPES["gpu-serving"]["status"]
        self.assertIn("R23-58", status)
        self.assertIn("48 launches", status)
        self.assertIn("NULL", status.upper())


class TestKernelIdentity(unittest.TestCase):
    def test_champion_pinned_by_commit_and_build(self):
        self.assertEqual(len(R.CHAMPION_COMMIT), 40)
        self.assertEqual(R.CHAMPION_BUILD_NUMBER, 10241)

    def test_the_champion_pin_is_labelled_as_a_pin_not_as_the_champion(self):
        """★ The CHAMPION_* block is the MEASUREMENT PIN (champion3), an ANCESTOR.

        Relabelling a measured digest onto a later commit is the failure this module
        exists to prevent, so the pin must say out loud that it is not the champion.
        """
        pin = R.CHAMPION_PIN_MEASURED_AT
        self.assertFalse(pin["is_current_champion"])
        self.assertEqual(pin["measured_at_commit"], R.CHAMPION_COMMIT)
        self.assertEqual(pin["measured_at_branch"], R.CHAMPION_BRANCH)
        self.assertEqual(pin["measured_at_build_number"], R.CHAMPION_BUILD_NUMBER)
        self.assertNotEqual(pin["measured_at_commit_short"],
                            R.CURRENT_CHAMPION["commit"])

    def test_the_current_champion_names_the_consolidated_branch(self):
        """★ Reconciled 2026-09-14 against CURRENT-CAMPAIGN.md's banner. The draft
        carried the FOLD CANDIDATE branch, which is where the merge was staged, not
        where the consolidated champion lives."""
        self.assertEqual(R.CURRENT_CHAMPION["branch"],
                         "ak/champion/llama-cpp-ffc1bac82eec")
        self.assertEqual(R.CURRENT_CHAMPION["fold_candidate_branch"],
                         "inf70/fold-candidate-20260908")
        # ef81196d5 = GPU tip bff30cebe + CPU champion3 9c4f73e29.
        lin = R.CURRENT_CHAMPION["lineages_by_ancestry"]
        self.assertEqual(lin["gpu_tip"], "bff30cebe")
        self.assertEqual(lin["cpu_lineage"], "9c4f73e29")
        for part in (lin["gpu_tip"], lin["cpu_lineage"]):
            self.assertIn(part, R.CURRENT_CHAMPION["identity"])

    def test_the_pin_refuses_to_certify_the_current_champion(self):
        """★ The loud refusal WRAP-10 landed instead of a silent relabel."""
        with self.assertRaises(R.RecipeViolation):
            R.assert_current_champion_identity()
        self.assertIn(R.CURRENT_CHAMPION["commit"], R.PIN_GAP_BANNER)
        self.assertIn(R.CHAMPION_COMMIT[:9], R.PIN_GAP_BANNER)

    def test_measured_numbers_keep_the_commit_they_were_measured_at(self):
        """★ Provenance, not relabelling: the champion3 headline still says 9c4f73e29."""
        self.assertEqual(R.HEADLINES["champion3"]["measured_at_commit"], "9c4f73e29")
        self.assertIn("10241", R.HEADLINES["champion3"]["binary"])

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
        # ★ These stay None even though digested builds of ef81196d5 now exist: they
        # describe the HEADLINE's binary, which is a different (unidentified) build.
        # Filling them in from CURRENT_CHAMPION["builds"] is the exact silent relabel
        # the flag exists to prevent.
        self.assertIsNone(R.CURRENT_CHAMPION["build_number"])
        self.assertIsNone(R.CURRENT_CHAMPION["sha256"])
        # The prior champion is an ANCESTOR of the current one, not the champion.
        self.assertEqual(R.CURRENT_CHAMPION["lineages_by_ancestry"]["cpu_lineage"],
                         R.CHAMPION_COMMIT[:9])

    def test_the_full_sha_is_recorded_and_the_short_form_is_its_prefix(self):
        full = R.CURRENT_CHAMPION["commit_full"]
        self.assertEqual(len(full), 40)
        int(full, 16)
        self.assertTrue(full.startswith(R.CURRENT_CHAMPION["commit"]))
        self.assertTrue(R.CURRENT_CHAMPION["source_tree_clean"])

    def test_both_surfaces_have_a_digested_build_labelled_by_surface(self):
        """★ One entry per surface, each saying whether it is a HIP build. A CPU recipe
        pinned to a HIP build's bundled libggml-cpu.so is a wrong pin that looks right."""
        builds = R.CURRENT_CHAMPION["builds"]
        self.assertEqual(set(builds), {"gpu-serving", "cpu-decode"})
        self.assertTrue(builds["gpu-serving"]["is_hip_build"])
        self.assertFalse(builds["cpu-decode"]["is_hip_build"])
        for surface, spec in builds.items():
            self.assertEqual(spec["build_number"], 10301, surface)
            self.assertEqual(spec["commit_in_build_info"], "ef81196d5", surface)
            for name, digest in spec["digests"].items():
                self.assertEqual(len(digest), 64, f"{surface}/{name}")
                int(digest, 16)
        # The two builds' CPU backends are DIFFERENT objects: same commit, different
        # configuration, so the digests must not be equal.
        self.assertNotEqual(
            builds["gpu-serving"]["digests"]["libggml-cpu.so.0.16.0"],
            builds["cpu-decode"]["digests"]["libggml-cpu.so.0.16.0"],
        )
        # The GPU build's two doc-recorded digests were reproduced from the tree.
        self.assertTrue(builds["gpu-serving"]["digests_confirmed_against_doc"])
        # And the CPU build's cmake is the recipe's own cmake, not a lookalike.
        self.assertTrue(builds["cpu-decode"]["cmake_matches_champion_cmake_args"])

    def test_the_gap_names_the_measurement_not_a_missing_digest(self):
        """★ The gap was NARROWED, not closed. Digests exist now; the headline's binary
        is still unidentified (build 10303 vs the builds' 10301), and THAT is why the
        flag stays False. A gap text still claiming 'no digests exist' would be stale in
        a way every reader would act on."""
        gap = R.CHAMPION_PIN_GAP
        self.assertIn("10303", gap)
        self.assertIn("10301", gap)
        self.assertIn("build-champion-ef81196d5-cpu-20260909", gap)
        self.assertIn("build-fold-ef81196d5", gap)
        headline = R.HEADLINES[R.CANONICAL_HEADLINE]
        self.assertEqual(headline["binary_build_number"], 10303)
        self.assertNotEqual(
            headline["binary_build_number"],
            R.CURRENT_CHAMPION["builds"]["cpu-decode"]["build_number"],
        )
        # The instrument branch the numbers actually came off is named, and it is the
        # same one do_not_fold already refuses to merge.
        self.assertIn("2516c9807", headline["binary_gap"])
        self.assertIn("2516c9807",
                      " ".join(R.CURRENT_CHAMPION["do_not_fold"]))

    def test_the_headline_binary_is_identified_and_matches_the_headline(self):
        """★ THE GUARD THAT TIES THE NUMBERS TO AN ARTIFACT.

        The canonical headline's build number and HEADLINE_BINARY's must be the SAME
        number, or the module is quoting numbers against a binary record that is not
        theirs. Captured from scratch on 2026-09-14; if that tree is collected these
        digests become the headline's entire identity.
        """
        hb = R.HEADLINE_BINARY
        headline = R.HEADLINES[R.CANONICAL_HEADLINE]
        self.assertEqual(headline["binary_build_number"], hb["build_number"])
        self.assertEqual(headline["binary_record"], "HEADLINE_BINARY")
        self.assertEqual(hb["headline"], R.CANONICAL_HEADLINE)
        # It is NOT the champion commit, and NOT either digested champion build.
        self.assertNotEqual(hb["commit"], R.CURRENT_CHAMPION["commit"])
        for surface, spec in R.CURRENT_CHAMPION["builds"].items():
            self.assertNotEqual(hb["build_number"], spec["build_number"], surface)
        # ...but it DOES descend from the champion, and the delta is named rather than
        # left for a reader to assume either way.
        self.assertEqual(hb["descends_from_champion"], R.CURRENT_CHAMPION["commit"])
        self.assertTrue(hb["delta_from_champion"])
        self.assertTrue(hb["commit_full"].startswith(hb["commit"]))
        self.assertEqual(len(hb["commit_full"]), 40)
        for name, digest in hb["digests"].items():
            self.assertEqual(len(digest), 64, name)
            int(digest, 16)
        # The role must say what this binary is NOT, not only what it is.
        self.assertIn("NOT the champion", hb["role"])
        # And the gap must point at the record, so a reader who only reads the gap
        # still finds the digests.
        self.assertIn("HEADLINE_BINARY", R.CHAMPION_PIN_GAP)

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
