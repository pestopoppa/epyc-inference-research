"""AK-TEMP-AXIS-3: temperature is a required per-surface field and the 0.6 default is gone."""
import dataclasses
import unittest

import pytest

from autokernel.loop import production_surfaces as PS, resolved_recipe as RR, sampling_axis as S
from autokernel.loop import served_shape_cases as SC, serving

REGISTRY = {"roles": {
    "critic": {"model": {"name": PS.SERVED_MODEL}, "generation_defaults": {"temperature": 0.1}},
    "coder": {"model": {"name": PS.SERVED_MODEL}, "generation_defaults": {"temperature": 0.2}},
    "front": {"model": {"name": "Other"}, "generation_defaults": {"temperature": 0.3}},
    "math": {"model": {"name": "Other"}, "generation_defaults": {"temperature": 0.0}}}}


class SurfaceTemperature(unittest.TestCase):
    def test_a_surface_without_temperature_cannot_be_built(self):
        with self.assertRaises(TypeError):
            PS.Surface("n8", "uniform", (8,), anchor=True)

    def test_every_declared_surface_carries_the_8083_direct_temperature(self):
        self.assertTrue(all(s.temperature == S.DIRECT_8083_TEMPERATURE == 1.0 for s in PS.SURFACES))

    def test_served_temperatures_come_from_the_registry_roles_of_the_served_model(self):
        self.assertEqual(PS.served_temperatures(REGISTRY), (0.1, 0.2, 1.0))

    def test_census_is_keyed_by_surface_and_temperature(self):
        keys = PS.census_keys(REGISTRY)
        self.assertEqual(len(keys), len(PS.production_surfaces()) * 3)
        self.assertIn(("n16", 0.1), keys)
        self.assertNotIn(("n8", 1.0), keys)  # the anchor is not a production surface

    def test_assess_by_temperature_needs_every_surface_at_every_served_temperature(self):
        full = {k: 1.0 for k in PS.census_keys(REGISTRY)}
        self.assertTrue(PS.assess_by_temperature(full, floor_pct=1.0, registry=REGISTRY).promotable)
        partial = {k: v for k, v in full.items() if k[1] == 1.0}
        got = PS.assess_by_temperature(partial, floor_pct=1.0, registry=REGISTRY)
        self.assertEqual(got.label, PS.NOT_YET_TRANSFERABLE)
        self.assertIn("n16@T0.1", got.missing)
        worse = {**full, ("n32", 0.2): -5.0}
        self.assertEqual(PS.assess_by_temperature(worse, floor_pct=1.0, registry=REGISTRY).regressed,
                         ("n32@T0.2",))
        with self.assertRaises(ValueError):
            PS.assess_by_temperature({("n16", 0.6): 1.0}, floor_pct=1.0, registry=REGISTRY)

    def test_legacy_assess_is_unchanged(self):
        self.assertEqual(PS.assess({}, floor_pct=1.0).label, PS.NOT_YET_TRANSFERABLE)


class LiveRegistry(unittest.TestCase):
    def setUp(self):
        if not S.REGISTRY_DEFAULT.exists():
            self.skipTest("orchestrator registry not on this host")

    def test_surface_family_temperatures_equal_the_registry(self):
        want = sorted({*S.role_temperatures(model_name=PS.SERVED_MODEL).values(), 1.0})
        self.assertEqual(list(PS.served_temperatures()), want)

    def test_lane_temperatures_equal_the_registry(self):
        roles = S.role_temperatures()
        self.assertEqual(SC.LANE_PROFILES["q38fn"].served_temperature, roles["architect_general"])
        self.assertIsNone(SC.LANE_PROFILES["ds41"].served_temperature)
        self.assertFalse([r for r in roles if "ds41" in r.lower() or "deepseek" in r.lower()])


class LaneTemperature(unittest.TestCase):
    def test_the_field_is_required(self):
        fields = {f.name: f for f in dataclasses.fields(SC.MoeProfile)}
        self.assertIs(fields["served_temperature"].default, dataclasses.MISSING)
        self.assertTrue(all(hasattr(p, "served_temperature") for p in SC.LANE_PROFILES.values()))


class NoDefaultTemperature(unittest.TestCase):
    def recipe(self, **kw):
        return serving.Recipe(name="t", model="/m.gguf", **kw)

    def test_a_new_recipe_has_no_temperature(self):
        self.assertIsNone(self.recipe().temperature)

    def test_resolution_refuses_a_missing_temperature(self):
        with self.assertRaisesRegex(RR.ResolutionError, "declared"):
            RR._validate_workload(RR.WorkloadSpec(256, None, 0.95, 20, "aggregate_tok_s"))
        RR._validate_workload(RR.WorkloadSpec(256, 0.2, 0.95, 20, "aggregate_tok_s"))

    def test_projection_does_not_inherit_a_default(self):
        import inspect
        self.assertIsNone(inspect.signature(RR.canonical_recipe_projection).parameters["temperature"].default)

    def test_is_greedy_needs_an_explicit_zero_or_top_k_1(self):
        self.assertFalse(RR._is_greedy(self.recipe()))
        self.assertTrue(RR._is_greedy(self.recipe(temperature=0.0)))
        self.assertTrue(RR._is_greedy(self.recipe(top_k=1)))
        self.assertFalse(RR._is_greedy(self.recipe(temperature=0.2)))

    def test_legacy_serialisations_keep_their_hash(self):
        declared = self.recipe(temperature=serving.LEGACY_UNDECLARED_TEMPERATURE)
        legacy = declared.to_dict()
        del legacy["temperature"]
        loaded = serving.Recipe.from_dict(legacy)
        self.assertEqual(loaded.temperature, 0.6)
        self.assertEqual(loaded.recipe_hash, declared.recipe_hash)
        self.assertEqual(serving.Recipe.from_dict(declared.to_dict()).recipe_hash, declared.recipe_hash)

    def test_a_declared_null_is_not_rewritten_to_the_legacy_value(self):
        data = self.recipe().to_dict()
        self.assertIsNone(serving.Recipe.from_dict(data).temperature)


@pytest.mark.parametrize("source", ["serving.py", "resolved_recipe.py"])
def test_no_0_6_default_remains(source):
    import re
    from pathlib import Path
    text = (Path(__file__).parent / source).read_text(encoding="utf-8")
    assert not re.search(r"temperature\s*:\s*float\s*=\s*0\.6", text)
