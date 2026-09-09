"""A serving recipe that carries ENVIRONMENT, tested WITHOUT launching a server.

WHY THIS EXISTS. `GGML_NOHUGEPAGE_PROCESS=1` -- the `prctl(PR_SET_THP_DISABLE)` shim, set at
LAUNCH and distinct from the already-on madvise `GGML_NOHUGEPAGE` -- cut between-launch
variance 25.3x on the CPU surface (sd 2.510% OFF vs 0.481% ON) for a +5.23% median, and the
operator has adopted it into the canonical CPU launch recipe. The GPU serving floor (4.581%
p95 at n=10) is the binding constraint on every GPU keep, so the same question has to be
ASKABLE on the serving path -- and it was not, because `serving.Recipe` had no way to carry
an environment variable at all.

What must hold before that arm can be trusted:
  * the env reaches the launched process, and the recipe wins over the inherited env;
  * a recipe can NEVER take `LD_LIBRARY_PATH` from the loader -- three ggml generations live
    on this host and a silent override runs wrong with a zero exit code;
  * the env is visible in `describe()` and in the persisted row, so a record cannot claim an
    arm it did not run;
  * the recipe has a content-addressed identity that MOVES when the env moves (R23-59);
  * the readback proves the knob took effect in BOTH directions -- a control arm that is
    secretly the treatment is unrecoverable after the fact;
  * the comparator reports SPREAD, because the effect under test is a compressed tail and a
    means-only comparator returns "no effect" on exactly that shape.
"""
import dataclasses
import json
from pathlib import Path
import unittest
from unittest import mock

from autokernel.loop import serving


BASE = serving.Recipe(name="t", model="/m/target.gguf", np=4, ctx=16384)
#: The paired arm this whole change exists to make expressible.
THP = {"GGML_NOHUGEPAGE_PROCESS": "1"}
#: One declaration, both directions: shim ON => THP_enabled 0, shim absent => THP_enabled 1.
THP_READBACK = ({"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS",
                 "expect": {"1": "0", "unset": "1"}},)


def _status(**fields: str) -> str:
    """A synthetic `/proc/<pid>/status` blob. No process is ever launched here."""
    return "".join(f"{k}:\t{v}\n" for k, v in fields.items())


class EnvReachesTheProcess(unittest.TestCase):
    def test_default_recipe_carries_no_env_and_launches_exactly_as_before(self):
        self.assertIsNone(BASE.env)
        env = BASE.server_env(Path("/B"), base={"HOME": "/h"})
        self.assertEqual(env["LD_LIBRARY_PATH"], "/B/bin")
        self.assertEqual(env["HOME"], "/h")
        self.assertNotIn("GGML_NOHUGEPAGE_PROCESS", env)

    def test_recipe_env_is_applied_to_the_launched_environment(self):
        r = dataclasses.replace(BASE, env=dict(THP))
        env = r.server_env(Path("/B"), base={"HOME": "/h"})
        self.assertEqual(env["GGML_NOHUGEPAGE_PROCESS"], "1")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/B/bin")  # loader pin survives

    def test_recipe_env_overrides_the_inherited_env_for_the_same_key(self):
        r = dataclasses.replace(BASE, env=dict(THP))
        env = r.server_env(Path("/B"), base={"GGML_NOHUGEPAGE_PROCESS": "0"})
        self.assertEqual(env["GGML_NOHUGEPAGE_PROCESS"], "1")

    def test_explicit_unset_removes_an_inherited_parent_value(self):
        control = BASE.with_env(GGML_NOHUGEPAGE_PROCESS=None)
        env = control.server_env(
            Path("/B"), base={"GGML_NOHUGEPAGE_PROCESS": "1", "HOME": "/h"})
        self.assertNotIn("GGML_NOHUGEPAGE_PROCESS", env)
        self.assertEqual(env["HOME"], "/h")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/B/bin")

    def test_the_launch_path_uses_server_env_and_not_a_hand_built_dict(self):
        """The env must reach `Popen`, not merely exist on the recipe."""
        r = dataclasses.replace(BASE, env=dict(THP))
        seen = {}

        class _Proc:
            pid = 4321
            returncode = 0

            def poll(self):
                return None

            def terminate(self):
                seen["terminated"] = True

            def wait(self, _t=None):
                return 0

        def _popen(argv, **kw):
            seen["argv"], seen["env"] = argv, kw["env"]
            return _Proc()

        with mock.patch.object(serving.subprocess, "Popen", _popen), \
             mock.patch.object(serving.urllib.request, "urlopen",
                               side_effect=RuntimeError("no server in a unit test")):
            with self.assertRaises(serving.ServerDied):
                serving._measure_once(r, Path("/B"), 18311, boot_timeout_s=2)
        self.assertEqual(seen["env"]["GGML_NOHUGEPAGE_PROCESS"], "1")
        self.assertEqual(seen["env"]["LD_LIBRARY_PATH"], "/B/bin")
        self.assertTrue(seen["terminated"])


class LoaderVariablesAreRefused(unittest.TestCase):
    """A silent override of the linkage pin is the one failure with no exit code."""

    def test_setting_ld_library_path_is_a_hard_error(self):
        with self.assertRaises(serving.RecipeError) as ctx:
            dataclasses.replace(BASE, env={"LD_LIBRARY_PATH": "/somewhere/else"})
        self.assertIn("LD_LIBRARY_PATH", str(ctx.exception))

    def test_setting_hsa_override_is_a_hard_error_too(self):
        # The loader env deliberately UNSETS it; a recipe re-setting it would defeat that
        # just as silently as overriding the library path.
        with self.assertRaises(serving.RecipeError):
            dataclasses.replace(BASE, env={"HSA_OVERRIDE_GFX_VERSION": "9.0.10"})

    def test_every_loader_owned_name_is_refused(self):
        for name in serving.LOADER_OWNED_ENV:
            with self.subTest(name=name), self.assertRaises(serving.RecipeError):
                dataclasses.replace(BASE, env={name: "x"})

    def test_every_loader_owned_name_is_also_refused_as_an_explicit_unset(self):
        for name in serving.LOADER_OWNED_ENV:
            with self.subTest(name=name), self.assertRaises(serving.RecipeError):
                dataclasses.replace(BASE, explicit_unsets=(name,))

    def test_a_non_string_env_value_is_refused_rather_than_coerced(self):
        with self.assertRaises(serving.RecipeError):
            dataclasses.replace(BASE, env={"GGML_NOHUGEPAGE_PROCESS": 1})

    def test_invalid_set_and_unset_keys_are_refused(self):
        for key in ("", "A=B", "A\0B", 1):
            with self.subTest(key=key), self.assertRaises(serving.RecipeError):
                dataclasses.replace(BASE, env={key: "x"})
            with self.subTest(key=key), self.assertRaises(serving.RecipeError):
                dataclasses.replace(BASE, explicit_unsets=(key,))

    def test_a_key_cannot_be_both_set_and_unset(self):
        with self.assertRaises(serving.RecipeError):
            dataclasses.replace(BASE, env={"KNOB": "1"}, explicit_unsets=("KNOB",))

    def test_duplicate_unset_keys_are_refused(self):
        with self.assertRaises(serving.RecipeError):
            dataclasses.replace(BASE, explicit_unsets=("A", "A"))

    def test_a_string_is_not_misread_as_a_sequence_of_unset_keys(self):
        with self.assertRaises(serving.RecipeError):
            serving.Recipe.from_dict({**BASE.to_dict(), "explicit_unsets": "KNOB"})


class DescribeStatesTheArm(unittest.TestCase):
    def test_describe_renders_the_env_pairs(self):
        r = dataclasses.replace(BASE, env=dict(THP))
        self.assertIn("env=GGML_NOHUGEPAGE_PROCESS=1", r.describe())

    def test_describe_says_env_none_when_there_is_none(self):
        self.assertIn("env=none", BASE.describe())

    def test_describe_renders_explicit_unsets(self):
        self.assertIn("env=KNOB=<unset>", BASE.with_env(KNOB=None).describe())

    def test_describe_renders_the_resolved_readback(self):
        r = dataclasses.replace(BASE, env=dict(THP), env_readback=THP_READBACK)
        self.assertIn("readback=THP_enabled=0", r.describe())

    def test_describe_carries_the_recipe_hash_prefix(self):
        self.assertIn(BASE.recipe_hash[:12], BASE.describe())


class RecipeIdentity(unittest.TestCase):
    """R23-59: a champion identified only by a commit hash is under-specified once the
    launch recipe is part of the artifact."""

    def test_the_hash_moves_when_the_env_moves(self):
        self.assertNotEqual(BASE.recipe_hash,
                            dataclasses.replace(BASE, env=dict(THP)).recipe_hash)

    def test_no_env_and_empty_env_are_the_same_condition(self):
        self.assertEqual(BASE.recipe_hash, dataclasses.replace(BASE, env={}).recipe_hash)

    def test_legacy_serialization_and_hash_are_exactly_unchanged(self):
        self.assertNotIn("explicit_unsets", BASE.to_dict())
        self.assertEqual(
            BASE.recipe_hash,
            "36f0b19a0ba9c0d6c82b69a06851673d806512b7431dba89ebac41fa3865de98")

    def test_explicit_unset_is_distinct_from_ordinary_inherited_state(self):
        control = BASE.with_env(name=BASE.name, GGML_NOHUGEPAGE_PROCESS=None)
        self.assertNotEqual(control.recipe_hash, BASE.recipe_hash)
        self.assertEqual(control.env, {})
        self.assertEqual(control.explicit_unsets, ("GGML_NOHUGEPAGE_PROCESS",))

    def test_explicit_unsets_round_trip_in_canonical_sorted_order(self):
        r = dataclasses.replace(BASE, explicit_unsets=("Z_KNOB", "A_KNOB"))
        self.assertEqual(r.explicit_unsets, ("A_KNOB", "Z_KNOB"))
        self.assertEqual(r.to_dict()["explicit_unsets"], ["A_KNOB", "Z_KNOB"])
        back = serving.Recipe.from_dict(json.loads(json.dumps(r.to_dict())))
        self.assertEqual(back.explicit_unsets, r.explicit_unsets)
        self.assertEqual(back.recipe_hash, r.recipe_hash)

    def test_the_hash_is_stable_across_reserialization(self):
        r = dataclasses.replace(BASE, env=dict(THP), env_readback=THP_READBACK)
        back = serving.Recipe.from_dict(json.loads(json.dumps(r.to_dict())))
        self.assertEqual(r.recipe_hash, back.recipe_hash)
        self.assertEqual(r, back)

    def test_the_hash_still_covers_the_pre_existing_fields(self):
        for change in ({"np": 8}, {"cpu_list": "184-191"}, {"ctx": 8192},
                       {"model": "/m/other.gguf"}, {"extra_flags": ("--foo",)}):
            with self.subTest(change=change):
                self.assertNotEqual(BASE.recipe_hash,
                                    dataclasses.replace(BASE, **change).recipe_hash)

    def test_the_shipped_recipe_round_trips_and_still_carries_no_env(self):
        r = serving.Recipe.load(Path(__file__).resolve().parents[4]
                                / "artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json")
        self.assertEqual(r.env, None)          # UNCHANGED: the arm is expressible, not adopted
        self.assertEqual(r.env_readback, ())
        self.assertEqual(r.recipe_hash,
                         serving.Recipe.from_dict(r.to_dict()).recipe_hash)


class WithEnv(unittest.TestCase):
    def test_with_env_returns_an_independent_recipe(self):
        on = BASE.with_env(GGML_NOHUGEPAGE_PROCESS="1")
        self.assertIsNone(BASE.env)                       # the original is untouched
        self.assertEqual(on.env, THP)
        self.assertNotEqual(on.recipe_hash, BASE.recipe_hash)

    def test_with_env_names_the_arm_so_it_cannot_reuse_the_other_arms_floor(self):
        # `serving-floor.<name>.json` is keyed by NAME; a different env is a different
        # measured condition and must not be judged against the old arm's floor.
        self.assertEqual(BASE.with_env(GGML_NOHUGEPAGE_PROCESS="1").name,
                         "t+GGML_NOHUGEPAGE_PROCESS=1")

    def test_a_none_override_removes_the_variable_giving_the_control_arm(self):
        on = BASE.with_env(GGML_NOHUGEPAGE_PROCESS="1")
        off = on.with_env(GGML_NOHUGEPAGE_PROCESS=None)
        self.assertEqual(off.env, {})
        self.assertEqual(off.explicit_unsets, ("GGML_NOHUGEPAGE_PROCESS",))
        # Explicit removal is distinct from permitted inheritance even after normalising
        # the name, because the parent process may carry the knob.
        self.assertNotEqual(dataclasses.replace(off, name=BASE.name).recipe_hash,
                            BASE.recipe_hash)
        self.assertNotEqual(off.recipe_hash, BASE.recipe_hash)
        self.assertNotEqual(off.name, BASE.name)

    def test_setting_an_explicitly_unset_key_replaces_the_marker(self):
        control = BASE.with_env(KNOB=None)
        treatment = control.with_env(KNOB="1")
        self.assertEqual(treatment.env, {"KNOB": "1"})
        self.assertEqual(treatment.explicit_unsets, ())

    def test_unsetting_a_set_key_deletes_the_override_and_records_the_marker(self):
        treatment = BASE.with_env(KNOB="1")
        control = treatment.with_env(KNOB=None)
        self.assertEqual(control.env, {})
        self.assertEqual(control.explicit_unsets, ("KNOB",))

    def test_derivation_never_mutates_the_base_recipe(self):
        base = dataclasses.replace(BASE, env={"KEEP": "yes"},
                                   explicit_unsets=("OLD",))
        derived = base.with_env(KEEP=None, OLD="new", ADDED=None)
        self.assertEqual(base.env, {"KEEP": "yes"})
        self.assertEqual(base.explicit_unsets, ("OLD",))
        self.assertEqual(derived.env, {"OLD": "new"})
        self.assertEqual(derived.explicit_unsets, ("ADDED", "KEEP"))

    def test_an_explicit_name_is_honoured(self):
        self.assertEqual(BASE.with_env(name="thp-on", GGML_NOHUGEPAGE_PROCESS="1").name,
                         "thp-on")


class EnvReadback(unittest.TestCase):
    """"I set the knob" is not evidence the knob took effect."""

    def test_the_treatment_arm_passes_when_thp_is_actually_disabled(self):
        r = dataclasses.replace(BASE, env=dict(THP), env_readback=THP_READBACK)
        got = serving.verify_env_readback(r, 1, status_text=_status(Name="llama-server",
                                                                    THP_enabled="0"))
        self.assertEqual(got, {"THP_enabled": "0"})

    def test_the_treatment_arm_refuses_when_the_shim_did_not_take(self):
        r = dataclasses.replace(BASE, env=dict(THP), env_readback=THP_READBACK)
        with self.assertRaises(serving.EnvReadbackFailed):
            serving.verify_env_readback(r, 1, status_text=_status(THP_enabled="1"))

    def test_the_control_arm_refuses_when_it_is_secretly_the_treatment(self):
        # The direction that a one-sided check misses, and the one that cannot be
        # recovered after the fact.
        control = dataclasses.replace(BASE, env_readback=THP_READBACK)
        self.assertEqual(serving.verify_env_readback(
            control, 1, status_text=_status(THP_enabled="1")), {"THP_enabled": "1"})
        with self.assertRaises(serving.EnvReadbackFailed):
            serving.verify_env_readback(control, 1, status_text=_status(THP_enabled="0"))

    def test_explicit_control_unset_preserves_bidirectional_readback(self):
        treatment = dataclasses.replace(BASE, env=dict(THP), env_readback=THP_READBACK)
        control = treatment.with_env(GGML_NOHUGEPAGE_PROCESS=None)
        self.assertEqual(treatment.readback_expectations(), (("THP_enabled", "0"),))
        self.assertEqual(control.readback_expectations(), (("THP_enabled", "1"),))
        self.assertNotIn("GGML_NOHUGEPAGE_PROCESS", control.server_env(
            Path("/B"), base={"GGML_NOHUGEPAGE_PROCESS": "1"}))

    def test_a_missing_status_field_refuses_rather_than_passing(self):
        r = dataclasses.replace(BASE, env=dict(THP), env_readback=THP_READBACK)
        with self.assertRaises(serving.EnvReadbackFailed):
            serving.verify_env_readback(r, 1, status_text=_status(Name="llama-server"))

    def test_an_unreadable_proc_status_refuses(self):
        r = dataclasses.replace(BASE, env=dict(THP), env_readback=THP_READBACK)
        with self.assertRaises(serving.EnvReadbackFailed):
            serving.verify_env_readback(r, -1)

    def test_no_declaration_means_no_check_and_no_proc_read(self):
        self.assertEqual(serving.verify_env_readback(BASE, -1), {})

    def test_an_env_state_the_declaration_does_not_cover_is_refused_at_construction(self):
        one_sided = ({"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS",
                      "expect": {"1": "0"}},)
        dataclasses.replace(BASE, env=dict(THP), env_readback=one_sided)   # ON arm is fine
        with self.assertRaises(serving.RecipeError):
            dataclasses.replace(BASE, env_readback=one_sided)              # control is not

    def test_with_env_cannot_produce_an_undeclared_control_arm(self):
        one_sided = ({"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS",
                      "expect": {"1": "0"}},)
        on = dataclasses.replace(BASE, env=dict(THP), env_readback=one_sided)
        with self.assertRaises(serving.RecipeError):
            on.with_env(GGML_NOHUGEPAGE_PROCESS=None)

    def test_an_unconditional_expectation_is_allowed(self):
        r = dataclasses.replace(BASE, env_readback=({"field": "THP_enabled",
                                                     "expect": "1"},))
        self.assertEqual(r.readback_expectations(), (("THP_enabled", "1"),))

    def test_a_malformed_declaration_is_refused(self):
        with self.assertRaises(serving.RecipeError):
            dataclasses.replace(BASE, env_readback=({"expect": "1"},))
        with self.assertRaises(serving.RecipeError):
            dataclasses.replace(BASE, env_readback=({"field": "F", "expect": {"1": "0"}},))

    def test_the_launch_path_verifies_before_measuring(self):
        src = Path(serving.__file__).read_text()
        body = src.split("def _measure_once(", 1)[1]
        self.assertLess(body.index("verify_env_readback(recipe, srv.pid)"),
                        body.index("def one("), "readback must precede the workload")


class SpreadReporting(unittest.TestCase):
    """The THP effect was a COMPRESSED DOWNSIDE TAIL, not a shifted mean. A means-only
    comparator returns "no effect" on exactly that shape, so the row must carry spread."""

    #: Same median, radically different tails -- the shape a means-only read cannot see.
    NOISY = [100.0, 100.0, 90.0, 110.0, 80.0]
    TIGHT = [100.0, 100.0, 99.0, 101.0, 99.5]

    def test_spread_fields_are_present_per_arm_with_the_per_launch_values(self):
        with mock.patch.object(serving, "_measure_once",
                               side_effect=[v for pair in zip(self.NOISY, self.TIGHT)
                                            for v in pair]):
            out = serving.compare(BASE, Path("/a"), Path("/c"), pairs=5, floor_pct=1.0)
        self.assertEqual(out["anchor_spread"]["runs"], self.NOISY)
        self.assertEqual(out["candidate_spread"]["runs"], self.TIGHT)
        for key in ("n", "median", "mean", "sd", "cv_pct", "min", "max",
                    "range_pct", "p95_dev_pct", "max_dev_pct"):
            self.assertIn(key, out["anchor_spread"], key)

    def test_the_spread_numbers_are_right_on_a_synthetic_vector(self):
        sp = serving._spread(self.NOISY)
        self.assertEqual((sp["n"], sp["median"], sp["min"], sp["max"]), (5, 100.0, 80.0, 110.0))
        self.assertAlmostEqual(sp["mean"], 96.0, places=6)
        self.assertAlmostEqual(sp["sd"], 10.19803902718557, places=9)
        self.assertEqual(sp["cv_pct"], 10.198)
        self.assertEqual(sp["range_pct"], 30.0)
        self.assertEqual(sp["max_dev_pct"], 20.0)   # the 80.0 launch
        self.assertEqual(sp["p95_dev_pct"], 20.0)

    def test_a_tail_compression_at_an_equal_median_is_visible_in_the_row(self):
        with mock.patch.object(serving, "_measure_once",
                               side_effect=[v for pair in zip(self.NOISY, self.TIGHT)
                                            for v in pair]):
            out = serving.compare(BASE, Path("/a"), Path("/c"), pairs=5, floor_pct=1.0)
        self.assertAlmostEqual(out["effect_pct"], 0.0, places=9)   # medians identical
        self.assertFalse(out["decisive"])                          # verdict UNCHANGED
        self.assertGreater(out["anchor_spread"]["sd"], out["candidate_spread"]["sd"] * 5)
        self.assertGreater(out["anchor_spread"]["p95_dev_pct"],
                           out["candidate_spread"]["p95_dev_pct"])
        self.assertLess(out["anchor_spread"]["min"], out["candidate_spread"]["min"])

    def test_the_floor_is_the_same_p95_deviation_the_spread_reports(self):
        with mock.patch.object(serving, "_measure_once", side_effect=list(self.NOISY)):
            floor = serving.calibrate_floor(BASE, Path("/b"), samples=5)
        self.assertEqual(floor["floor_pct"], floor["spread"]["p95_dev_pct"])
        self.assertEqual(floor["cv_pct"], floor["spread"]["cv_pct"])
        self.assertEqual(floor["median_tok_s"], floor["spread"]["median"])
        self.assertEqual(floor["runs"], self.NOISY)

    def test_a_single_pair_still_produces_a_well_formed_spread(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 120.0]):
            out = serving.compare(BASE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)
        self.assertEqual(out["anchor_spread"]["n"], 1)
        self.assertEqual(out["anchor_spread"]["sd"], 0.0)
        self.assertEqual(out["anchor_spread"]["p95_dev_pct"], 0.0)


class TheRecordStatesTheArm(unittest.TestCase):
    """A record must never be able to claim an arm it did not run."""

    def test_the_ab_row_carries_the_env_and_the_recipe_hash(self):
        r = dataclasses.replace(BASE, env=dict(THP))
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 120.0]):
            out = serving.compare(r, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)
        self.assertEqual(out["recipe_env"], THP)
        self.assertEqual(out["recipe_hash"], r.recipe_hash)
        self.assertIn("GGML_NOHUGEPAGE_PROCESS=1", out["recipe_describe"])

    def test_the_floor_row_carries_the_env_and_the_recipe_hash(self):
        r = dataclasses.replace(BASE, env=dict(THP))
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0] * 5):
            out = serving.calibrate_floor(r, Path("/b"), samples=5)
        self.assertEqual(out["recipe_env"], THP)
        self.assertEqual(out["recipe_hash"], r.recipe_hash)

    def test_two_arms_of_one_ab_are_distinguishable_in_their_rows(self):
        off, on = BASE, BASE.with_env(GGML_NOHUGEPAGE_PROCESS="1")
        rows = []
        for r in (off, on):
            with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 100.0]):
                rows.append(serving.compare(r, Path("/a"), Path("/c"), pairs=1,
                                            floor_pct=1.0))
        self.assertNotEqual(rows[0]["recipe_hash"], rows[1]["recipe_hash"])
        self.assertNotEqual(rows[0]["recipe_env"], rows[1]["recipe_env"])
        self.assertNotEqual(rows[0]["recipe"], rows[1]["recipe"])


if __name__ == "__main__":
    unittest.main()
