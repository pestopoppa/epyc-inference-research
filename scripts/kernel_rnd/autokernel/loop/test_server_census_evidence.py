"""Independent prompt/API/tensor widths: a whole prompt is not its ubatch."""
import unittest
from types import SimpleNamespace
from autokernel.loop.server_census_evidence import shape_evidence,tensor_shape_observed

class PrefillTests(unittest.TestCase):
    def test_checkpoint_suffix_requires_exact_native_ubatch(self):
        shape=SimpleNamespace(phase='prefill',n_tokens=2048,n_seq=1)
        requests=[{'prompt':[1]*2052}]
        startup='n_seq_max = 1\nn_ubatch = 2048\n'
        log='slot operator(): id 0 | task 1 | new prompt, task.n_tokens = 2052\n'
        responses=[{'stop':True,'timings':{'predicted_n':1}}]
        samples=[{'phase':'requests','slots':[{'id':0,'is_processing':True}]}]
        self.assertTrue(shape_evidence(shape,startup,log,requests,responses,samples)['observed'])
        self.assertFalse(shape_evidence(shape,startup,log.replace('2052','2051'),requests,responses,samples)['observed'])
        self.assertFalse(shape_evidence(shape,'',log,requests,responses,samples)['observed'])
        def graph(api,tensor):
            return {'graphs':[{'api_batch':{'tokens':api,'tokens_per_seq':{'0':api}},'dense_matmul_ne11':[tensor]}]}
        self.assertTrue(tensor_shape_observed(shape,graph(2048,2048)))
        self.assertFalse(tensor_shape_observed(shape,graph(2052,2048)))
        self.assertFalse(tensor_shape_observed(shape,graph(2048,2052)))

if __name__=='__main__':unittest.main()
