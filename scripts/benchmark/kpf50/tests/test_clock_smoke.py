import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from verify_clock_smoke import verify

def cap(t,b,e): return {'schema':'epyc.kpf50.hsa-clock.v1','frequency_hz':1000000000,'samples':[{'hsa_ticks':t,'host_before_ns':b,'host_after_ns':e}]}
class Test(unittest.TestCase):
    def test_valid(self):
        r=verify(cap(100,110,114),cap(300,310,314),'KPF50_CLOCK_WITNESS begin_ns=150 end_ns=250',[{'KernelName':'kpf50_clock_witness','BeginNs':'150','EndNs':'190'}])
        self.assertTrue(r['verified'])
    def test_bad_epoch(self):
        with self.assertRaises(ValueError): verify(cap(100,110,114),cap(300,310,314),'KPF50_CLOCK_WITNESS begin_ns=150 end_ns=250',[{'KernelName':'kpf50_clock_witness','BeginNs':'1','EndNs':'2'}])
    def test_not_bracketed(self):
        with self.assertRaises(ValueError): verify(cap(200,210,214),cap(300,310,314),'KPF50_CLOCK_WITNESS begin_ns=150 end_ns=250',[{'KernelName':'kpf50_clock_witness','BeginNs':'150','EndNs':'190'}])
if __name__=='__main__': unittest.main()
