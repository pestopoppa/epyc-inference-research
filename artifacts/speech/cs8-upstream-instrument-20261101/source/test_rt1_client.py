import asyncio
import base64
import tempfile
import unittest
from pathlib import Path
from rt1_client import UnknownSink, configure, journaled, run_turn

class Clock:
    def __init__(self): self.t = 0
    def __call__(self): self.t += 10; return self.t

class Conn:
    def __init__(self, events): self.events = iter(events); self.sent = []
    async def send(self, event): self.sent.append(event['type'])
    async def recv(self): return next(self.events)

class Sink:
    def __init__(self): self.pushed = []
    def push(self, audio, ns): self.pushed.append((audio, ns))
    async def cancel_and_drain(self, ns): return ns + 17

async def no_sleep(_): pass
AUDIO = {'type':'response.output_audio.delta','delta':base64.b64encode(b'\x01\x00').decode()}
DONE = {'type':'response.done','response':{'status':'completed'}}
CANCELLED = {'type':'response.done','response':{'status':'cancelled'}}

class TestClient(unittest.IsolatedAsyncioTestCase):
    async def test_text_before_audio_and_order(self):
        c = Conn([{'type':'response.output_text.delta','delta':'hi'}, AUDIO, DONE]); clock = Clock()
        r = await run_turn(c, b'\x00\x00', 1, Sink(), chunk_ms=1000, clock=clock, sleep=no_sleep)
        self.assertEqual(c.sent, ['input_audio_buffer.append','input_audio_buffer.commit','response.create'])
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['first_audio_ns'], next(e['time_ns'] for e in r['events'] if e['type']=='response.output_audio.delta'))
        self.assertGreater(r['first_audio_ns'], next(e['time_ns'] for e in r['events'] if e['type']=='response.output_text.delta'))
    async def test_cancel_measured_sink_and_unknown_sink(self):
        c = Conn([AUDIO, CANCELLED])
        r = await run_turn(c, b'\0\0', 1, Sink(), cancel=True, chunk_ms=1000, clock=Clock(), sleep=no_sleep)
        self.assertEqual(c.sent[-1], 'response.cancel'); self.assertEqual(r['cancel_to_silence_ns'], 17); self.assertEqual(r['status'],'complete')
        c = Conn([AUDIO, CANCELLED])
        r = await run_turn(c, b'\0\0', 1, UnknownSink(), cancel=True, chunk_ms=1000, clock=Clock(), sleep=no_sleep)
        self.assertIsNone(r['cancel_to_silence_ns']); self.assertEqual(r['status'],'unknown')
    async def test_null_detection_refusal(self):
        c = Conn([{'type':'session.created','session':{'audio':{'input':{'turn_detection':{'type':'server_vad'}}}}}])
        with self.assertRaises(ValueError): await configure(c)
    async def test_error_does_not_pass(self):
        c = Conn([{'type':'error','error':'broken'}])
        r = await run_turn(c, b'\0\0', 1, Sink(), chunk_ms=1000, clock=Clock(), sleep=no_sleep)
        self.assertEqual(r['status'],'failure'); self.assertIsNone(r['first_audio_latency_ns'])
    async def test_audio_after_claimed_silence_is_unknown(self):
        class FastSink(Sink):
            async def cancel_and_drain(self, ns): return ns + 5
        c = Conn([AUDIO, AUDIO, CANCELLED])
        r = await run_turn(c, b'\0\0', 1, FastSink(), cancel=True, chunk_ms=1000, clock=Clock(), sleep=no_sleep)
        self.assertTrue(r['audio_after_silence'])
        self.assertEqual(r['status'], 'unknown')
    async def test_null_detection_acknowledged(self):
        session = {'audio':{'input':{'turn_detection':None,'format':{'rate':16000}}}}
        c = Conn([{'type':'session.created','session':session}, {'type':'session.updated','session':session}])
        self.assertEqual(await configure(c), session)
        self.assertEqual(c.sent, ['session.update'])
    async def test_journaled_normal_and_cutoff_no_redraw(self):
        with tempfile.TemporaryDirectory() as d:
            called = []
            async def normal():
                called.append('normal')
                return await run_turn(Conn([AUDIO, DONE]), b'\0\0', 1, Sink(),
                                      chunk_ms=1000, clock=Clock(), sleep=no_sleep)
            self.assertEqual((await journaled(d,'ok',{'pcm':'x'},normal))['status'],'complete')
            self.assertIsNone(await journaled(d,'ok',{'pcm':'x'},normal))
            self.assertEqual(called, ['normal'])
            async def cutoff():
                raise asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):
                await journaled(d,'cut',{'pcm':'y'},cutoff)
            self.assertTrue((Path(d)/'cut.exposure.json').exists())
            self.assertFalse((Path(d)/'cut.result.json').exists())
            self.assertIsNone(await journaled(d,'cut',{'pcm':'y'},normal))

if __name__ == '__main__': unittest.main()
