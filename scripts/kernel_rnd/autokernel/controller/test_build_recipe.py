"""A build flag may diverge from production, but never by omission."""
import unittest

from autokernel.controller import build_recipe as br
from autokernel.controller import discovery_deployment_factory as factory


class RecipeContract(unittest.TestCase):

    def test_the_house_recipe_matches_production_on_every_flag(self):
        self.assertEqual(br.HOUSE_GPU_RECIPE.divergences(), ())

    def test_rocwmma_fattn_is_named_explicitly(self):
        """The flag whose ABSENCE was the defect.

        It was never in `cmake_defines`, so it fell to the CMake default OFF -- a
        path measured to produce non-finite values at longer sequence lengths on
        gfx90a under `-fa on`. A recipe that cannot express "this flag was
        considered" cannot prevent that recurring.
        """
        names = {name for name, _ in br.HOUSE_GPU_RECIPE.cmake_defines()}
        self.assertIn("GGML_HIP_ROCWMMA_FATTN", names)
        flag = next(f for f in br.HOUSE_GPU_RECIPE.flags
                    if f.name == "GGML_HIP_ROCWMMA_FATTN")
        self.assertEqual(flag.value, "ON")
        self.assertEqual(flag.production_value, "ON")

    def test_a_divergence_without_a_reason_is_refused(self):
        with self.assertRaises(br.BuildRecipeError) as caught:
            br.Flag("GGML_NATIVE", "OFF", "ON")
        self.assertIn("no stated reason", str(caught.exception))

    def test_a_divergence_with_a_reason_is_allowed(self):
        flag = br.Flag("GGML_NATIVE", "OFF", "ON",
                       reason="portable build for a fork on other hardware")
        self.assertTrue(flag.diverges)
        self.assertEqual(flag.to_dict()["reason"],
                         "portable build for a fork on other hardware")

    def test_a_whitespace_reason_does_not_count_as_a_reason(self):
        with self.assertRaises(br.BuildRecipeError):
            br.Flag("GGML_NATIVE", "OFF", "ON", reason="   ")

    def test_recipe_identity_changes_with_any_flag(self):
        """The digest is emitted with results; it must move when the build moves."""
        baseline = br.HOUSE_GPU_RECIPE.sha256()
        altered = br.from_flags("gfx90a-house-v1", [
            {"name": "GGML_HIP", "value": "ON", "production_value": "ON"},
            {"name": "AMDGPU_TARGETS", "value": "gfx90a", "production_value": "gfx90a"},
            {"name": "GGML_HIP_ROCWMMA_FATTN", "value": "ON", "production_value": "ON"},
            {"name": "GGML_NATIVE", "value": "OFF", "production_value": "ON",
             "reason": "portability trade for a fork"},
        ]).sha256()
        self.assertNotEqual(baseline, altered)

    def test_identity_is_stable_for_the_same_recipe(self):
        self.assertEqual(br.HOUSE_GPU_RECIPE.sha256(), br.HOUSE_GPU_RECIPE.sha256())

    def test_the_production_reference_is_declared_not_verified(self):
        """`build-hip/` holds only `bin/` -- there is no CMakeCache to read back.

        The production column is sourced from the CH-8 ruling, not from the build
        production actually serves. Saying so in a constant is the difference
        between a known gap and a silent one.
        """
        self.assertFalse(br.PRODUCTION_RECIPE_IS_VERIFIABLE)
        self.assertIs(
            br.HOUSE_GPU_RECIPE.to_dict()["production_reference_is_verifiable"],
            False)

    def test_unknown_recipe_names_are_refused(self):
        with self.assertRaises(br.BuildRecipeError):
            br.recipe_for("something-else")

    def test_a_recipe_round_trips_through_its_record_with_the_same_identity(self):
        """`to_dict` is how a recipe reaches a champion record; a reader rebuilds it.

        If the round trip lost the name, the notes or a reason, the rebuilt recipe
        would carry a DIFFERENT digest, and two records of one build would compare
        unequal — which is the comparison the champion plane runs on.
        """
        record = br.HOUSE_GPU_RECIPE.to_dict()
        rebuilt = br.from_flags(record["name"], record["flags"],
                                notes=record["notes"] or "")
        self.assertEqual(rebuilt.to_dict(), record)
        self.assertEqual(rebuilt.sha256(), br.HOUSE_GPU_RECIPE.sha256())


class SettledNonAdoptions(unittest.TestCase):
    """CH-6 settled both standing config wins as NOT adopted. Recorded, not baked in."""

    def test_neither_settled_non_adoption_is_in_the_house_recipe(self):
        values = {(flag.name, flag.value) for flag in br.HOUSE_GPU_RECIPE.flags}
        for item in br.SETTLED_NON_ADOPTIONS:
            self.assertNotIn((item.setting, item.rejected_value), values)

    def test_the_withdrawn_numbers_are_recorded_next_to_the_mechanism(self):
        """The +23.09% and +46.9% still circulate; the correction travels with them.

        Not a prose assertion for its own sake: this is the only place a reader
        deciding whether to adopt these flags will be standing, and a rejection
        recorded without the number it corrects does not stop a re-adoption.
        """
        findings = {item.setting: item.finding for item in br.SETTLED_NON_ADOPTIONS}
        self.assertIn("+0.50%", findings["GGML_HIP_MMQ_MFMA"])
        self.assertIn("n_embd=896", findings["GGML_HIP_MMQ_MFMA"])
        self.assertIn("min(n_batch, n_ubatch)", findings["n_ubatch"])

    def test_readopting_a_settled_flag_without_a_reason_is_refused(self):
        with self.assertRaises(br.BuildRecipeError) as caught:
            br.Flag("GGML_HIP_MMQ_MFMA", "OFF", "ON")
        self.assertIn("+0.50%", str(caught.exception))

    def test_the_refusal_survives_a_misdeclared_production_value(self):
        """The spelling the divergence rule alone cannot catch.

        Claiming production builds `MMQ_MFMA=OFF` makes the flag non-diverging, so
        no reason would ever be demanded and the withdrawn +23.09% would enter the
        recipe silently.
        """
        with self.assertRaises(br.BuildRecipeError):
            br.Flag("GGML_HIP_MMQ_MFMA", "OFF", "OFF")

    def test_readopting_it_with_a_stated_reason_is_allowed(self):
        flag = br.Flag("GGML_HIP_MMQ_MFMA", "OFF", "ON",
                       reason="re-testing at -np 8 on a real model, where the "
                              "0.5B pp512 surface says nothing")
        self.assertTrue(flag.diverges)

    def test_a_settled_runtime_setting_is_recorded_but_not_a_cmake_flag(self):
        """`n_ubatch` is a serving flag; a build recipe cannot express it at all.

        Recording it here anyway is the point — it is the second half of the same
        decision, and a reader who finds only one will re-run the other.
        """
        ubatch = next(item for item in br.SETTLED_NON_ADOPTIONS
                      if item.setting == "n_ubatch")
        self.assertFalse(ubatch.is_cmake_flag)
        # Not enforced as a flag refusal, because it is not a flag.
        br.Flag("n_ubatch", "1024", "1024")

    def test_the_house_recipe_identity_is_pinned(self):
        """Changing this digest re-epochs every recorded measurement.

        `loop/run.py` feeds `HOUSE_GPU_RECIPE.to_dict()` into `epoch_sha256`, so a
        flag added or removed here marks every prior record `stale_epoch`. That may
        well be correct — but it is a decision, and this line is where it gets made
        rather than noticed later in a recall that returned nothing comparable.
        """
        self.assertEqual(
            br.HOUSE_GPU_RECIPE.sha256(),
            "27a65ce389e7e053ab44cc64f5bae30579c81f7d6344b3595e927a72728e8404")


class FactoryUsesTheRecipe(unittest.TestCase):

    def test_the_factory_source_no_longer_carries_a_literal_tuple(self):
        """One place defines the build contract, and it is the recipe."""
        import inspect
        source = inspect.getsource(factory)
        self.assertIn("build_recipe.HOUSE_GPU_RECIPE.cmake_defines()", source)
        self.assertNotIn('("GGML_HIP", "ON"), ("AMDGPU_TARGETS", "gfx90a")', source)


if __name__ == "__main__":
    unittest.main()
