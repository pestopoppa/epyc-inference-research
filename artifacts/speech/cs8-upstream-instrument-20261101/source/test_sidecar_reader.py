import unittest
from sidecar_reader import read
class SidecarTests(unittest.TestCase):
    def test_text_and_empty(self):
        x=read(b'CS8TRACE 1 stream\nT 3 3\nabc\nT 4 0\n\nE complete stream_end\n')
        self.assertEqual(x['text'],'abc'); self.assertEqual(x['token_ids'],[3,4])
        self.assertEqual(read(b'CS8TRACE 1 chat\nE complete eog\n')['text'],'')
    def test_refuse_mutations(self):
        for x in (b'',b'CS8TRACE 1 stream\nT 3 4\nabc\nE complete eog\n',b'CS8TRACE 1 chat\nE unknown token_limit\n',b'CS8TRACE 1 chat\nE complete eog\nmore',b'CS8TRACE 1 chat\nE complete stream_end\n',b'CS8TRACE 1 chat\nT 5 1\n\xff\nE complete eog\n'):
            with self.assertRaises((ValueError,UnicodeDecodeError)): read(x)
if __name__=='__main__': unittest.main()
