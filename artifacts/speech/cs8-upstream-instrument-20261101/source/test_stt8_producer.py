import tempfile, unittest
from pathlib import Path
from stt8_producer import REQUIRED, read_case, run_case
BIND={k:'pinned' for k in REQUIRED}
class ProducerTests(unittest.TestCase):
    def test_normal_empty_and_failure(self):
        with tempfile.TemporaryDirectory() as d:
            good=lambda _: {'reaped':True,'interrupted':False,'rc':0,'stdout':b'noise','stderr':b'', 'sidecar':b'CS8TRACE 1 chat\nT 1 0\n\nE complete eog\n'}
            r=run_case(d,'a',BIND,good)
            self.assertEqual((r['status'],r['text']),('complete',''))
            self.assertEqual(read_case(d,'a',BIND)['text'],'')
            self.assertIsNone(run_case(d,'a',BIND,good))
            (Path(d)/'a.stdout').write_bytes(b'tampered')
            with self.assertRaises(ValueError): read_case(d,'a',BIND)
            r=run_case(d,'b',BIND,lambda _: {'reaped':True,'interrupted':False,'rc':1,'stdout':b'', 'stderr':b'bad'})
            self.assertEqual(r['status'],'failure')
    def test_cutoff_and_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            def cutoff(_): raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt): run_case(d,'a',BIND,cutoff)
            self.assertTrue((Path(d)/'a.exposure.json').exists()); self.assertFalse((Path(d)/'a.result.json').exists())
            self.assertIsNone(run_case(d,'a',BIND,lambda _:None))
            r=run_case(d,'b',BIND,lambda _: {'reaped':True,'rc':0,'stdout':b'', 'stderr':b''})
            self.assertEqual(r['status'],'unknown')
            r=run_case(d,'c',BIND,lambda _: {'reaped':True,'rc':True,'stdout':b'', 'stderr':b''})
            self.assertEqual(r['status'],'unknown')
            r=run_case(d,'d',BIND,lambda _: {'reaped':True,'interrupted':True,'rc':-15,'stdout':b'partial', 'stderr':b''})
            self.assertEqual(r['status'],'unknown')
            strict={**BIND,'require_runtime_mapping':True}
            r=run_case(d,'e',strict,lambda _: {'reaped':True,'interrupted':False,'rc':0,'stdout':b'', 'stderr':b'',
                                               'sidecar':b'CS8TRACE 1 chat\nE complete eog\n'})
            self.assertEqual(r['status'],'unknown')
    def test_binding_required(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): run_case(d,'a',{},lambda _:None)
if __name__=='__main__':unittest.main()
