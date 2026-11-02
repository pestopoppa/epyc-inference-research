import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('hsa_clock', Path(__file__).parents[1] / 'hsa_clock.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def cap(t, before, after, f=1000000000):
    return {'schema':'epyc.kpf50.hsa-clock.v1', 'frequency_hz':f,
            'samples':[{'host_before_ns':before,'hsa_ticks':t,'host_after_ns':after}]}

class ClockTest(unittest.TestCase):
    def test_exact_intersection(self):
        self.assertEqual(m.offset_interval([cap(100, 110, 114), cap(200, 211, 213)]), (11, 13))
    def test_drift_fails_closed(self):
        with self.assertRaises(ValueError): m.offset_interval([cap(100, 110, 111), cap(200, 220, 221)])
    def test_missing_after_fails_closed(self):
        with self.assertRaises(ValueError): m.offset_interval([cap(100,110,111)])
    def test_frequency_change_fails_closed(self):
        with self.assertRaises(ValueError): m.offset_interval([cap(100,110,111),cap(200,220,221,200)])
    def test_tick_rounding_brackets(self):
        self.assertEqual(m.offset_interval([cap(1,334,335,3000000),cap(2,667,668,3000000)]), (0,2))

if __name__ == '__main__': unittest.main()
