import json, tempfile, unittest
from pathlib import Path
from exposure_journal import begin, finish

class JournalTests(unittest.TestCase):
    def test_normal_then_no_redraw(self):
        with tempfile.TemporaryDirectory() as d:
            token=begin(d,'clip_1',{'pcm':'abc'})
            finish(d,'clip_1',token,{'status':'complete','text':''})
            self.assertEqual(json.loads((Path(d)/'clip_1.result.json').read_text())['result']['text'],'')
            with self.assertRaises(FileExistsError): begin(d,'clip_1',{'pcm':'abc'})
            with self.assertRaises(FileExistsError): finish(d,'clip_1',token,{'status':'changed'})
    def test_cutoff_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            token=begin(d,'clip_2',{'pcm':'abc'})
            with self.assertRaises(FileExistsError): begin(d,'clip_2',{'pcm':'abc'})
            (Path(d)/'clip_2.exposure.json').write_text('{}')
            with self.assertRaises(ValueError): finish(d,'clip_2',token,{'status':'complete'})
            self.assertFalse((Path(d)/'clip_2.result.json').exists())
    def test_bad_key(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): begin(d,'../x',{})

if __name__=='__main__': unittest.main()
