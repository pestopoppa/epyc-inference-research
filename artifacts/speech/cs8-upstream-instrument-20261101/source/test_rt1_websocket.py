import asyncio, base64, json, tempfile, unittest
from pathlib import Path
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
from rt1_client import configure, journaled, run_turn
from alsa_digital_sink import ALSADigitalSink

class Conn:
    def __init__(self,ws): self.ws=ws
    async def send(self,event): await self.ws.send(json.dumps(event))
    async def recv(self): return json.loads(await self.ws.recv())

class WebsocketControl(unittest.IsolatedAsyncioTestCase):
    async def test_same_entry_real_websocket_and_null_pcm(self):
        received=[]
        session={'audio':{'input':{'turn_detection':None,'format':{'rate':16000}},'output':{'format':{'rate':24000}}}}
        async def server(ws):
            await ws.send(json.dumps({'type':'session.created','session':session}))
            async for msg in ws:
                ev=json.loads(msg); received.append(ev['type'])
                if ev['type']=='session.update': await ws.send(json.dumps({'type':'session.updated','session':session}))
                if ev['type']=='response.create':
                    await ws.send(json.dumps({'type':'response.output_text.delta','delta':'words'}))
                    await ws.send(json.dumps({'type':'response.output_audio.delta','delta':base64.b64encode(b'\0\0'*80).decode()}))
                if ev['type']=='response.cancel':
                    await ws.send(json.dumps({'type':'response.done','response':{'status':'cancelled'}})); return
        async def no_sleep(_): pass
        async with serve(server,'127.0.0.1',0) as srv:
            port=srv.sockets[0].getsockname()[1]
            with tempfile.TemporaryDirectory() as d:
                async def action():
                    async with connect(f'ws://127.0.0.1:{port}/v1/realtime') as ws:
                        conn=Conn(ws); await configure(conn)
                        sink=ALSADigitalSink('null',24000)
                        try: return await run_turn(conn,b'\0\0'*80,16000,sink,cancel=True,sleep=no_sleep,timeout=5)
                        finally: sink.close()
                result=await journaled(d,'item',{'fixture':True},action)
                self.assertEqual(result['status'],'complete')
                self.assertIsNotNone(result['first_digital_write_ns'])
                self.assertGreaterEqual(result['cancel_to_silence_ns'],0)
                self.assertEqual(received[-3:],['input_audio_buffer.commit','response.create','response.cancel'])
                self.assertTrue((Path(d)/'item.result.json').exists())
                self.assertIsNone(await journaled(d,'item',{'fixture':True},action))
    async def test_same_entry_websocket_cutoff_unknown_no_redraw(self):
        session={'audio':{'input':{'turn_detection':None,'format':{'rate':16000}},'output':{'format':{'rate':24000}}}}
        async def server(ws):
            await ws.send(json.dumps({'type':'session.created','session':session}))
            async for msg in ws:
                ev=json.loads(msg)
                if ev['type']=='session.update': await ws.send(json.dumps({'type':'session.updated','session':session}))
                if ev['type']=='response.create': await asyncio.sleep(.3)
        async def no_sleep(_): pass
        async with serve(server,'127.0.0.1',0) as srv:
            port=srv.sockets[0].getsockname()[1]
            with tempfile.TemporaryDirectory() as d:
                async def action():
                    async with connect(f'ws://127.0.0.1:{port}/v1/realtime') as ws:
                        c=Conn(ws); await configure(c)
                        sink=ALSADigitalSink('null',24000)
                        try: return await run_turn(c,b'\0\0'*80,16000,sink,sleep=no_sleep,timeout=.05)
                        finally: sink.close()
                r=await journaled(d,'cut',{'fixture':True},action)
                self.assertEqual(r['status'],'unknown')
                self.assertIsNone(r['first_audio_ns'])
                self.assertIsNone(await journaled(d,'cut',{'fixture':True},action))

if __name__=='__main__': unittest.main()

class SDKWebsocketControl(unittest.IsolatedAsyncioTestCase):
    async def test_openai_sdk_real_websocket_contract(self):
        from openai import AsyncOpenAI
        session={'id':'fixture','object':'realtime.session','type':'realtime','audio':{'input':{'turn_detection':None,'format':{'type':'audio/pcm','rate':16000}},'output':{'format':{'type':'audio/pcm','rate':24000}}}}
        received=[]
        async def server(ws):
            await ws.send(json.dumps({'event_id':'e0','type':'session.created','session':session}))
            async for msg in ws:
                ev=json.loads(msg); received.append(ev['type'])
                if ev['type']=='session.update': await ws.send(json.dumps({'event_id':'e1','type':'session.updated','session':session}))
                if ev['type']=='response.create':
                    await ws.send(json.dumps({'event_id':'e2','type':'response.output_audio.delta','response_id':'r','item_id':'i','output_index':0,'content_index':0,'delta':base64.b64encode(b'\0\0'*80).decode()}))
                if ev['type']=='response.cancel':
                    await ws.send(json.dumps({'event_id':'e3','type':'response.done','response':{'id':'r','object':'realtime.response','status':'cancelled','output':[]}})); return
        async def no_sleep(_): pass
        async with serve(server,'127.0.0.1',0) as srv:
            port=srv.sockets[0].getsockname()[1]; url=f'http://127.0.0.1:{port}/v1'
            client=AsyncOpenAI(api_key='test',base_url=url,websocket_base_url=url.replace('http','ws',1))
            async with client.realtime.connect(model='fixture') as c:
                await configure(c)
                sink=ALSADigitalSink('null',24000)
                try: r=await run_turn(c,b'\0\0'*80,16000,sink,cancel=True,sleep=no_sleep,timeout=5)
                finally: sink.close()
            self.assertEqual(r['status'],'complete')
            self.assertEqual(received[-3:],['input_audio_buffer.commit','response.create','response.cancel'])
