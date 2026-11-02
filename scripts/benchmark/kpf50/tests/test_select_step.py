import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from select_step import first_step,compare

def log(masked):
    return (['KPF50_ITER_BEGIN iter=1 host_ns=80 draft_context=1','KPF50_DRAFT_BEGIN host_ns=81 slots=4','KPF50_DRAFT_END host_ns=90 slots=4'] + [f'KPF30_PHASE call=1 index={i} seq={i} pos=80000 phase=decode' for i in range(4)] +
            [f'KPF50_INPUT call=1 index={i} token={100+i} pos=80000' for i in range(4)] +
            ['KPF50_STEP_BEGIN call=1 host_ns=100',f'KPF50_TRUNK_MASK verified=1 masked={masked} rows=4 n_kv=80000 expected=60000 layout_fnv=0123',
             'KPF50_STEP_END call=1 host_ns=200 rc=0', 'KPF50_ITER_END iter=1 host_ns=250'])
class Tests(unittest.TestCase):
    def test_match(self):
        x=compare(first_step(log(0),0),first_step(log(1),1))
        self.assertEqual(x['gate_verdict'],'UNRESOLVED'); self.assertEqual(x['kv_value_identity'],'unproved')
    def test_changed_token_refuses(self):
        a=log(1); a[7]=a[7].replace('token=100','token=999')
        with self.assertRaises(ValueError): compare(first_step(log(0),0),first_step(a,1))
    def test_layout_mismatch_refuses(self):
        a=log(1); a[12]=a[12].replace('layout_fnv=0123','layout_fnv=9999')
        with self.assertRaises(ValueError): compare(first_step(log(0),0),first_step(a,1))
    def test_missing_token_refuses(self):
        a=log(1); del a[7]
        with self.assertRaises(ValueError): first_step(a,1)
    def test_missing_draft_refuses(self):
        a=log(1); a[0]=a[0].replace('draft_context=1','draft_context=0')
        with self.assertRaises(ValueError): first_step(a,1)
    def test_first_witness_is_not_overwritten(self):
        a=log(1)
        second=[x.replace('call=1','call=2').replace('host_ns=100','host_ns=201').replace('host_ns=200','host_ns=210')
                for x in log(1)[3:-1]]
        a[-1:-1]=second
        self.assertEqual(first_step(a,1)['call'],1)
    def test_inactive_draft_refuses(self):
        a=log(1); a[1]=a[1].replace('slots=4','slots=0'); a[2]=a[2].replace('slots=4','slots=0')
        with self.assertRaises(ValueError): first_step(a,1)
    def test_nonwitness_refuses(self):
        a=log(1); a[12]=a[12].replace('verified=1','verified=0')
        with self.assertRaises(ValueError): first_step(a,1)
    def test_mixed_phase_refuses(self):
        a=log(1); a[3]=a[3].replace('decode','prefill')
        with self.assertRaises(ValueError): first_step(a,1)
if __name__=='__main__': unittest.main()
