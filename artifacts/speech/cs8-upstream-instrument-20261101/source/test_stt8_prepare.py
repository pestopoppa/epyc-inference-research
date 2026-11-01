import unittest
from stt8_prepare import command, provisional_voxtral

class TestSTT8(unittest.TestCase):
    def test_commands_pin_language_and_projector(self):
        c=command('lfm3','/tmp/clip.wav','it')
        self.assertIn('Perform ASR in Italian.',c); self.assertIn('--mmproj',c); self.assertIn('<__media__>',c)
        v=command('voxtral','/tmp/clip.wav','en'); self.assertIn('--audio',v); self.assertNotIn('--system-prompt',v)
        with self.assertRaises(ValueError): command('lfm3','x','de')
    def test_framed_completion_empty_and_mutations(self):
        before=b'init\nTranscribing 1.0 s of audio, fed in chunks of 80 ms\n\n'
        after=b'\n\nrun_audio_stream: 4 tokens, 1.00 s audio in 2.00 s (0.50x realtime)\n'
        self.assertEqual(provisional_voxtral(before+b'hello world'+after,0),'hello world')
        self.assertEqual(provisional_voxtral(before+after,0),'')
        capped=after.replace(b'4 tokens',b'1024 tokens')
        for bad, rc in [(before+b'hello',0),(before+b'hello'+after,1),(before+before+after,0),(before+capped,0)]:
            with self.assertRaises(ValueError): provisional_voxtral(bad,rc)

if __name__=='__main__': unittest.main()
