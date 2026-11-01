#!/usr/bin/env python3
"""RT1 prospective client core; playback sink must provide measured drain evidence."""
import asyncio
import base64
import hashlib
import json
import time
import wave
from pathlib import Path
from exposure_journal import begin, finish


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_cohort(path):
    raw = Path(path).read_bytes()
    obj = json.loads(raw)
    rows = obj['utterances']
    if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
        raise ValueError('cohort must contain 20 distinct IDs')
    for row in rows:
        p = Path(row['path'])
        if sha(p.read_bytes()) != row['sha256']:
            raise ValueError('input hash mismatch: ' + row['id'])
    return obj, sha(raw)


def read_pcm(path, expected_rate):
    with wave.open(str(path), 'rb') as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != expected_rate:
            raise ValueError('input must be mono PCM16 at negotiated input rate')
        return w.readframes(w.getnframes())


class UnknownSink:
    """Do not equate websocket delivery or response.done with audible silence."""
    def push(self, pcm, ns):
        pass
    async def cancel_and_drain(self, ns):
        return None


async def run_turn(conn, pcm, rate, sink, *, cancel=False, chunk_ms=80,
                   clock=time.monotonic_ns, sleep=asyncio.sleep, timeout=300):
    if not pcm or len(pcm) % 2 or rate <= 0 or chunk_ms <= 0:
        raise ValueError('invalid PCM or chunk parameters')
    deadline = time.monotonic() + timeout
    step = rate * chunk_ms // 1000 * 2
    if not step:
        raise ValueError('empty chunk')
    events = []
    record = {'schema':'cs8-rt1-turn/v1', 'input_pcm_sha256':sha(pcm),
              'input_rate':rate, 'chunk_ms':chunk_ms, 'cancel_requested':cancel,
              'input_end_ns':None, 'first_audio_ns':None, 'first_digital_write_ns':None, 'cancel_ns':None,
              'silence_ns':None, 'status':'unknown', 'events':events, 'audio_sha256':[]}
    async def send(payload):
        await conn.send(payload)
        events.append({'direction':'out', 'type':payload['type'], 'time_ns':clock()})
    try:
        for i in range(0, len(pcm), step):
            if time.monotonic() >= deadline:
                raise TimeoutError('whole-turn deadline')
            part = pcm[i:i+step]
            await send({'type':'input_audio_buffer.append', 'audio':base64.b64encode(part).decode('ascii')})
            await sleep(len(part) / (2 * rate))
        record['input_end_ns'] = clock()
        await send({'type':'input_audio_buffer.commit'})
        await send({'type':'response.create'})
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('whole-turn deadline')
            ev = await asyncio.wait_for(conn.recv(), timeout=remaining)
            if hasattr(ev, 'model_dump'):
                ev = ev.model_dump(warnings=False)
            t = clock(); kind = ev['type']
            item = {'direction':'in', 'type':kind, 'time_ns':t}
            item['event'] = {k:v for k,v in ev.items() if k != 'delta' or kind != 'response.output_audio.delta'}
            events.append(item)
            if kind == 'error':
                record['error'] = ev.get('error'); record['status'] = 'failure'; break
            if kind == 'response.output_audio.delta':
                audio = base64.b64decode(ev['delta'], validate=True)
                if not audio or len(audio) % 2:
                    raise ValueError('malformed output audio')
                item['audio_sha256'] = sha(audio); item['bytes'] = len(audio)
                record['audio_sha256'].append(item['audio_sha256'])
                if record['silence_ns'] is not None and t >= record['silence_ns']:
                    record['audio_after_silence'] = True
                if record['first_audio_ns'] is None: record['first_audio_ns'] = t
                written_at = sink.push(audio, t)
                if written_at is not None and record['first_digital_write_ns'] is None:
                    record['first_digital_write_ns'] = written_at
                if cancel and record['cancel_ns'] is None:
                    await send({'type':'response.cancel'})
                    record['cancel_ns'] = events[-1]['time_ns']
                    record['silence_ns'] = await sink.cancel_and_drain(record['cancel_ns'])
            if kind == 'response.done':
                record['server_response_status'] = ev.get('response', {}).get('status')
                if record['first_audio_ns'] is None:
                    record['status'] = 'failure'
                elif cancel and (record['server_response_status'] != 'cancelled' or
                                 record['silence_ns'] is None or
                                 record['silence_ns'] < record['cancel_ns'] or
                                 record.get('audio_after_silence', False)):
                    record['status'] = 'unknown'
                else:
                    record['status'] = 'complete'
                break
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        record['status'] = 'unknown'
    record['first_audio_latency_ns'] = (None if record['first_audio_ns'] is None or record['input_end_ns'] is None
                                        else record['first_audio_ns'] - record['input_end_ns'])
    record['cancel_to_silence_ns'] = (None if record['silence_ns'] is None or record['cancel_ns'] is None
                                      else record['silence_ns'] - record['cancel_ns'])
    record['first_digital_write_latency_ns'] = (None if record['first_digital_write_ns'] is None or record['input_end_ns'] is None
                                               else record['first_digital_write_ns'] - record['input_end_ns'])
    return record


async def configure(conn, events=None):
    ev = await conn.recv()
    if hasattr(ev, 'model_dump'): ev = ev.model_dump(warnings=False)
    if events is not None: events.append({'direction':'in','time_ns':time.monotonic_ns(),'event':ev})
    session = ev.get('session', {})
    input_audio = session.get('audio', {}).get('input', {})
    if ev.get('type') != 'session.created' or input_audio.get('turn_detection', 'missing') is not None:
        raise ValueError('server did not declare turn_detection null')
    await conn.send({'type':'session.update','session':{'type':'realtime',
                    'instructions':'Respond with interleaved text and audio.',
                    'output_modalities':['text','audio'],
                    'audio':{'input':{'turn_detection':None}}}})
    if events is not None: events.append({'direction':'out','time_ns':time.monotonic_ns(),'type':'session.update'})
    ev = await conn.recv()
    if hasattr(ev, 'model_dump'): ev = ev.model_dump(warnings=False)
    if events is not None: events.append({'direction':'in','time_ns':time.monotonic_ns(),'event':ev})
    if ev.get('type') != 'session.updated' or ev.get('session', {}).get('audio', {}).get('input', {}).get('turn_detection', 'missing') is not None:
        raise ValueError('server did not acknowledge turn_detection null')
    return session


async def journaled(directory, key, binding, action):
    try:
        exposure_sha = begin(directory, key, binding)
    except FileExistsError:
        return None  # Existing or interrupted exposure: no redraw.
    try:
        result = await action()
    except Exception as exc:
        result = {'status':'unknown','error':f'{type(exc).__name__}: {exc}'}
    finish(directory, key, exposure_sha, result)
    return result


async def run_cohort(args):
    from openai import AsyncOpenAI
    cohort, cohort_hash = load_cohort(args.manifest)
    binding = json.loads(Path(args.binding).read_bytes())
    required = ('server_binary_sha256','model_sha256','projector_sha256','library_set_sha256','plan_sha256','admission_id')
    if any(not isinstance(binding.get(k), str) or not binding[k] for k in required):
        raise ValueError('missing execution binding')
    client = AsyncOpenAI(base_url=args.url, api_key=args.api_key,
                         websocket_base_url=args.url.replace('http','ws',1))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for row in cohort['utterances']:
        if not row['id'].replace('-','').replace('_','').isalnum():
            raise ValueError('unsafe cohort ID')
        result = {'schema':'cs8-rt1-exposure/v1','id':row['id'],
                  'cohort_sha256':cohort_hash,'input_file_sha256':row['sha256'],
                  'status':'unknown'}
        exposure_binding = {**binding, 'id':row['id'], 'cohort_sha256':cohort_hash,
                            'input_file_sha256':row['sha256'], 'cancel':args.cancel,
                            'clock':'CLOCK_MONOTONIC', 'monotonic_anchor_ns':time.monotonic_ns(),
                            'realtime_anchor_ns':time.time_ns()}
        async def action():
            from alsa_digital_sink import ALSADigitalSink
            try:
                async with client.realtime.connect(model=args.model) as conn:
                    result['session_events'] = []
                    session = await asyncio.wait_for(configure(conn,result['session_events']), timeout=args.timeout)
                    rate = session['audio']['input']['format']['rate']
                    output_rate = session['audio']['output']['format']['rate']
                    pcm = read_pcm(row['path'], rate)
                    sink = ALSADigitalSink(args.alsa_device, output_rate)
                    try:
                        result['turn'] = await run_turn(conn, pcm, rate, sink, cancel=args.cancel, timeout=args.timeout)
                    finally:
                        sink.close()
                    result['sink'] = {'type':'ALSA PCM','device':args.alsa_device,'rate':output_rate,
                                      'silence_scope':'local ALSA PCM queue dropped; downstream acoustic output unobserved'}
                    result['status'] = result['turn']['status']
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
            return result
        await journaled(output, row['id'], exposure_binding, action)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--binding', required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--api-key', default='none')
    p.add_argument('--alsa-device', required=True)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--cancel', action='store_true')
    args = p.parse_args()
    if not 0 < args.timeout <= 300:
        p.error('timeout must be in (0, 300]')
    p.error('native execution is disabled until canonical qualification and a real playback sink are wired')
