"""Focused source fixture for R24-9; no benchmark or device needed."""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# The private pre-promotion fixture loads the exact origin/main candidate files.
# Normal repository tests import the promoted modules directly.
if os.environ.get('R24_CAPSULE_SOURCE'):
    CAPSULE = Path(os.environ['R24_CAPSULE_SOURCE']) / 'scripts/kernel_rnd/autokernel'
    sys.path.insert(0, str(Path(os.environ['R24_RESEARCH_SOURCE']) / 'scripts/kernel_rnd'))

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    from autokernel import controller
    controller.experiment_capture = load('autokernel.controller.experiment_capture',
                                         CAPSULE / 'controller/experiment_capture.py')
    controller.anchor_integrity = load('autokernel.controller.anchor_integrity',
                                       CAPSULE / 'controller/anchor_integrity.py')
    ex = load('autokernel.controller.r24_experiments_fixture',
              CAPSULE / 'controller/experiments.py')
    bench_fixture = load('autokernel.loop.r24_bench_fixture',
                         CAPSULE / 'loop/bench.py')
    st = load('autokernel.loop.r24_tripwire_fixture',
              CAPSULE / 'loop/same_code_tripwire.py')
    st.experiments = ex
    st.anchor_integrity = controller.anchor_integrity
else:
    from autokernel.controller import experiments as ex
    from autokernel.loop import bench as bench_fixture
    from autokernel.loop import same_code_tripwire as st

class Comparison:
    effect = 0.01
    pairs = 5
    decisive = True
    drifting = False
    noise_floor_pct = 0.5
    surface = 'tg128'

    def __init__(self, effect=0.01, floor=0.5):
        self.effect = effect
        self.noise_floor_pct = floor

    def to_dict(self):
        return {'surface': self.surface, 'model': 'model-A', 'effect': self.effect,
                'effect_unit': 'launch', 'noise_floor_pct': self.noise_floor_pct,
                'decisive': self.decisive}


class Tripwire(unittest.TestCase):
    def test_plain_cpu_bench_uses_cpu_code_manifest(self):
        recipe = type("Recipe", (), {"cmake_defines": lambda self: {"GGML_HIP": "OFF"}})()
        self.assertEqual(st.library_for(recipe), "libggml-cpu.so")
        self.assertEqual(st.library_for(recipe, type("Launch", (), {"backend": "gpu"})()),
                         "libggml-hip.so")
        recipe.cmake_defines = lambda: {"GGML_HIP": "ON"}
        self.assertEqual(st.library_for(recipe), "libggml-hip.so")
        self.assertEqual(st.library_for(recipe, type("Launch", (), {"backend": "cpu"})()),
                         "libggml-cpu.so")

    def test_partial_source_imports_have_no_canonical_dependency(self):
        root = os.environ.get('R24_PARTIAL_SOURCE_ROOT')
        if root is None:
            self.skipTest('only the pre-promotion source-bound fixture checks custody')
        for module in (ex, st, bench_fixture, st.loop, st.anchor_integrity):
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(Path(root)),
                            module.__file__)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for stem in ('anchor', 'candidate'):
            bin_dir = self.root / stem / 'bin'
            bin_dir.mkdir(parents=True)
            (bin_dir / 'libggml-hip.so').write_bytes(b'lib')
            (bin_dir / 'llama-bench').write_bytes(b'exe')
            (bin_dir / 'llama-server').write_bytes(b'server')
        self.la = {'build_recipe' : {'GGML_HIP': 'ON'}, 'model': 'model-A',
                   'surface': 'tg128', 'pp': 0, 'tg': 128, 'ubatch': 512, 'declared_pairs': 5}
        self.patches = [patch.object(st.anchor_integrity, 'section_spans',
                                     return_value=[('.text', 0, 1), ('.hip_fatbin', 1, 1)]),
                        patch.object(st.anchor_integrity, 'code_digest', side_effect=lambda p:
                                     'a' * 64 if 'anchor' in str(p) else 'c' * 64)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def observe(self, effect=0.01, launch=None, floor=0.5):
        return st.inspect(self.root, Comparison(effect, floor),
                          anchor_build=self.root / 'anchor', candidate_build=self.root / 'candidate',
                          library='libggml-hip.so', measurement_epoch='e' * 64,
                          launch_identity=self.la if launch is None else launch)

    def record(self, observed, identity='first'):
        with ex.ExperimentStore(self.root) as store:
            return store.record({'attempt_id': identity, 'status': 'measured_null',
                                 'effect_fraction': observed.effect,
                                 'comparison': observed.to_dict()}, epoch='e' * 64,
                                recorded_at='2026-10-30T00:00:00Z', campaign_id='ak-loop')

    def test_real_comparisons_different_samples_same_work_hit_excursion(self):
        first = bench_fixture.Comparison('tg128', [100.0] * 5, [101.0] * 5,
                                 0.01, 'median-ratio', 5, 0.5, {}, model='model-A')
        second = bench_fixture.Comparison('tg128', [98.0] * 5, [101.0] * 5,
                                  0.03, 'median-ratio', 5, 0.5, {}, model='model-A')
        def inspect(native):
            return st.inspect(self.root, native, anchor_build=self.root / 'anchor',
                              candidate_build=self.root / 'candidate',
                              library='libggml-hip.so', measurement_epoch='e' * 64,
                              launch_identity=self.la)
        observed = inspect(first)
        self.assertTrue(self.record(observed))
        with self.assertRaises(st.loop.MeasurementInvalid) as raised:
            inspect(second)
        self.assertEqual(raised.exception.record['same_code_tripwire']['status'],
                         'same_code_excursion')
        self.assertEqual(raised.exception.record['comparison']['anchor_samples'], [98.0] * 5)

    def test_same_work_divergence_is_archived_invalid_before_keep(self):
        first = self.observe()
        self.assertEqual(first.to_dict()['same_code_tripwire']['status'], 'first_observation')
        self.assertTrue(self.record(first))
        with self.assertRaises(st.loop.MeasurementInvalid) as raised:
            self.observe(0.03)
        evidence = raised.exception.record['same_code_tripwire']
        self.assertEqual(evidence['status'], 'same_code_excursion')
        self.assertAlmostEqual(evidence['delta_pct'], 2.0)
        self.assertEqual(evidence['prior_effect_fraction'], 0.01)
        self.assertEqual(raised.exception.record['comparison']['effect'], 0.03)

    def test_within_floor_does_not_change_native_decision(self):
        self.record(self.observe())
        again = self.observe(0.012)
        self.assertEqual(again.to_dict()['same_code_tripwire']['status'], 'within_floor')
        self.assertTrue(again.decisive)
        self.assertEqual(again.effect, 0.012)

    def test_different_launch_or_baseline_is_not_misidentified(self):
        self.record(self.observe())
        changed = dict(self.la, tg=256)
        self.assertEqual(self.observe(0.03, changed).evidence['status'], 'first_observation')
        with patch.object(st.anchor_integrity, 'code_digest', side_effect=lambda p:
                          'b' * 64 if 'anchor' in str(p) else 'c' * 64):
            self.assertEqual(self.observe(0.03).evidence['status'], 'first_observation')

    def test_real_serving_comparison_requires_same_metric_and_recipe(self):
        from autokernel.loop import run
        def native(effect, *, metric='tok_s', recipe_hash='r' * 64):
            return run.ServingComparison({
                'recipe': 'r', 'recipe_hash': recipe_hash, 'metric': metric,
                'request_digest': 'q' * 64, 'effect_unit': 'process',
                'effect': effect, 'noise_floor_pct': 0.5,
                'decisive': True, 'pairs': 5,
                'anchor_samples': [100.0] * 5,
                'candidate_samples': [100.0 * (1 + effect)] * 5})
        def inspect(compared):
            return st.inspect(self.root, compared, anchor_build=self.root / 'anchor',
                              candidate_build=self.root / 'candidate',
                              library='libggml-hip.so', measurement_epoch='e' * 64,
                              launch_identity=self.la)
        self.record(inspect(native(0.01)))
        self.assertEqual(inspect(native(0.03, metric='latency_ms')).evidence['status'],
                         'first_observation')
        self.assertEqual(inspect(native(0.03, recipe_hash='s' * 64)).evidence['status'],
                         'first_observation')
        with self.assertRaises(st.loop.MeasurementInvalid):
            inspect(native(0.03))

    def test_signed_effect_direction_is_preserved(self):
        self.record(self.observe(0.01))
        with self.assertRaises(st.loop.MeasurementInvalid) as raised:
            self.observe(-0.01)
        self.assertAlmostEqual(raised.exception.record['same_code_tripwire']['delta_pct'], -2.0)

    def test_missing_device_code_section_is_unknown(self):
        with patch.object(st.anchor_integrity, 'section_spans',
                          return_value=[('.text', 0, 1)]):
            result = self.observe()
        self.assertEqual(result.evidence['status'], 'unavailable')
        self.assertIsNone(result.evidence['work_key'])

    def test_missing_floor_never_claims_excursion(self):
        self.record(self.observe(floor=None))
        self.assertEqual(self.observe(0.03).evidence['status'], 'floor_unavailable')

    def test_unmeasured_row_has_no_tripwire_projection(self):
        with ex.ExperimentStore(self.root) as store:
            store.record({'attempt_id': 'refusal', 'status': 'gate_refused'},
                         epoch='e' * 64, recorded_at='2026-10-30T00:00:00Z',
                         campaign_id='ak-loop')
            self.assertEqual(store._connection.execute(
                'SELECT count(*) FROM same_code_measurements').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
