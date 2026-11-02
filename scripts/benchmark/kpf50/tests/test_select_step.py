import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from select_step import first_step,compare

def log(masked):
    return (['KPF50_ITER_BEGIN iter=1 host_ns=80 draft_context=1'] + [f'KPF30_PHASE call=1 index={i} seq={i} pos=80000 phase=decode' for i in range(4)] +
            [f'KPF50_INPUT call=1 index={i} token={100+i} pos=80000' for i in range(4)] +
            ['KPF50_STEP_BEGIN call=1 host_ns=100',f'KPF50_TRUNK_MASK verified=1 masked={masked} rows=4 n_kv=80000 expected=60000 layout_fnv=0123',
             'KPF50_STEP_END call=1 host_ns=200 rc=0', 'KPF50_ITER_END iter=1 host_ns=250'])
class Tests(unittest.TestCase):
    def test_match(self):
        x=compare(first_step(log(0),0),first_step(log(1),1))
        self.assertEqual(x['gate_verdict'],'UNRESOLVED'); self.assertEqual(x['kv_value_identity'],'unproved')
    def test_changed_token_refuses(self):
        a=log(1); a[5]=a[5].replace('token=100','token=999')
        with self.assertRaises(ValueError): compare(first_step(log(0),0),first_step(a,1))
    def test_layout_mismatch_refuses(self):
        a=log(1); a[10]=a[10].replace('layout_fnv=0123','layout_fnv=9999')
        with self.assertRaises(ValueError): compare(first_step(log(0),0),first_step(a,1))
    def test_missing_token_refuses(self):
        a=log(1); del a[5]
        with self.assertRaises(ValueError): first_step(a,1)
    def test_missing_draft_refuses(self):
        a=log(1); a[0]=a[0].replace('draft_context=1','draft_context=0')
        with self.assertRaises(ValueError): first_step(a,1)
    def test_nonwitness_refuses(self):
        a=log(1); a[10]=a[10].replace('verified=1','verified=0')
        with self.assertRaises(ValueError): first_step(a,1)
    def test_mixed_phase_refuses(self):
        a=log(1); a[1]=a[1].replace('decode','prefill')
        with self.assertRaises(ValueError): first_step(a,1)
if __name__=='__main__': unittest.main()
