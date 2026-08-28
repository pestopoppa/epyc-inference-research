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


class FactoryUsesTheRecipe(unittest.TestCase):

    def test_the_factory_source_no_longer_carries_a_literal_tuple(self):
        """One place defines the build contract, and it is the recipe."""
        import inspect
        source = inspect.getsource(factory)
        self.assertIn("build_recipe.HOUSE_GPU_RECIPE.cmake_defines()", source)
        self.assertNotIn('("GGML_HIP", "ON"), ("AMDGPU_TARGETS", "gfx90a")', source)


if __name__ == "__main__":
    unittest.main()
