import importlib.util
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
spec = importlib.util.spec_from_file_location('kpf50_analyze', Path(__file__).parents[1] / 'analyze_profile.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample():
    h = 'a' * 64
    return {'schema': 'epyc.kpf50.profile.v1', 'classification': 'prospective-diagnostic',
            **{k: h for k in ('model_sha256', 'server_sha256', 'hip_library_sha256', 'prompt_sha256',
                              'fork_witness_sha256', 'profiler_sha256')},
            'gpu_resident': True, 'exclusive_gpu_claim': True,
            'target': {'sequences': 4, 'context_tokens': 80000, 'trunk_tokens': 60000, 'drafter': 'DFlash2'},
            'fork_witnesses': 3, 'clock_domain': 'hsa-ns', 'rocprof_hsa_alignment_verified': True, 'max_clock_uncertainty_ns': 10,
            'attention_kernel_patterns': ['fattn'],
            'steps': [{'id': 's1', 'phase': 'decode', 'start_ns': 100, 'end_ns': 200,
                       'complete': True, 'synchronized': True}]}


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.m = sample()
        self.rows = [{'kernel': 'fattn', 'begin_ns': 110, 'end_ns': 150},
                     {'kernel': 'fattn', 'begin_ns': 130, 'end_ns': 160},
                     {'kernel': 'matmul', 'begin_ns': 160, 'end_ns': 180}]

    def test_union_and_no_trunk_or_gate_claim(self):
        r = m.analyze_profile(self.m, self.rows, (0, 0))
        self.assertEqual(r['steps'][0]['attention_busy_ns'], 50)
        self.assertEqual(r['steps'][0]['gpu_busy_ns'], 70)
        self.assertEqual(r['mean_attention_share'], .5)
        self.assertIsNone(r['trunk_attention_ns'])
        self.assertEqual(r['gate_verdict'], 'UNRESOLVED')

    def test_unwitnessed_or_uncalibrated_refuses(self):
        for key, value in [('fork_witnesses', 2), ('rocprof_hsa_alignment_verified', False), ('gpu_resident', False)]:
            with self.subTest(key=key):
                q = sample(); q[key] = value
                with self.assertRaises(ValueError): m.analyze_profile(q, self.rows, (0, 0))

    def test_crossing_kernel_and_prefill_refuse(self):
        with self.assertRaises(ValueError): m.analyze_profile(self.m, self.rows + [{'kernel':'fattn','begin_ns':90,'end_ns':110}], (0, 0))
        self.m['steps'][0]['phase'] = 'prefill'
        with self.assertRaises(ValueError): m.analyze_profile(self.m, self.rows, (0, 0))

    def test_mismatched_counterfactual_refuses(self):
        self.m['counterfactual_pairs'] = [{'complete':True,'same_query_mask_layout':True,'same_kernel':False,
                                           'alternate_order':True,'full_ns':100,'private_only_ns':30}]
        with self.assertRaises(ValueError): m.analyze_profile(self.m, self.rows, (0, 0))

    def test_counterfactual_is_not_trunk_verdict(self):
        self.m['counterfactual_pairs'] = [{'complete':True,'same_query_mask_layout':True,'same_kernel':True,
                                           'alternate_order':True,'full_ns':100,'private_only_ns':30}]
        r = m.analyze_profile(self.m, self.rows, (0, 0))
        self.assertEqual(r['counterfactual_marginal_ns'], [70])
        self.assertIsNone(r['trunk_attention_ns'])
        self.assertEqual(r['gate_verdict'], 'UNRESOLVED')


if __name__ == '__main__': unittest.main()
