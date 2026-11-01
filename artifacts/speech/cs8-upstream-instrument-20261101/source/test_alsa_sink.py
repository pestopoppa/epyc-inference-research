import asyncio, unittest
from alsa_digital_sink import ALSADigitalSink

class ALSATest(unittest.IsolatedAsyncioTestCase):
    async def test_real_alsa_null_pcm_drop(self):
        # Linux ALSA null plugin is a digital software control, not a speaker.
        s=ALSADigitalSink('null',24000,clock=lambda:1234)
        try:
            s.push(b'\0\0'*80,1000)
            self.assertEqual(await s.cancel_and_drain(1100),1234)
        finally: s.close()
    async def test_empty_refused(self):
        s=ALSADigitalSink('null',24000)
        try:
            with self.assertRaises(ValueError): s.push(b'',0)
        finally: s.close()
if __name__=='__main__': unittest.main()
