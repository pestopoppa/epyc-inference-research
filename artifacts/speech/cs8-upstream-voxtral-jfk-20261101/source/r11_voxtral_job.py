#!/usr/bin/env python3
"""One public noncohort JFK Voxtral streaming ASR engineering control; never a READY job."""
import hashlib, json, os, time, wave
from pathlib import Path
from stt8_prepare import command, MODELS, PREFIX
from stt8_engine import engine_for, run_process
from stt8_producer import run_case
ROOT=Path(__file__).resolve().parent
PIN=ROOT/'r11-voxtral'/'PINS.json'
OUT=ROOT/'r11-voxtral'/'jfk-control'
WAV=Path('/mnt/raid0/llm/whisper.cpp/samples/jfk.wav')
EXPECTED_WAV='59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
EXPECTED_PCM='a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def main():
    expected_pin=os.environ.get('CS8_CONTROL_PIN_SHA','')
    if len(expected_pin)!=64 or sha(PIN)!=expected_pin: raise RuntimeError('control pin catalog mismatch')
    pins=json.loads(PIN.read_text())
    for path,want in pins['files'].items():
        if sha(path)!=want: raise RuntimeError('pinned input drift: '+path)
    if sha(WAV)!=EXPECTED_WAV: raise RuntimeError('JFK WAV drift')
    with wave.open(str(WAV),'rb') as w:
        if (w.getnchannels(),w.getsampwidth(),w.getframerate())!=(1,2,16000): raise RuntimeError('JFK format drift')
        pcm=w.readframes(w.getnframes())
    if hashlib.sha256(pcm).hexdigest()!=EXPECTED_PCM: raise RuntimeError('JFK PCM drift')
    OUT.mkdir(parents=True,exist_ok=True)
    cmd=command('voxtral',WAV,'en')+['--ctx-size','4096','--no-mmproj-offload']
    prefix=str(PREFIX.parent.resolve()); trace=OUT/'jfk.native-trace'
    binding={'pcm_sha256':EXPECTED_PCM,'input_sha256':EXPECTED_WAV,'plan_sha256':sha(PIN),
             'model_sha256':pins['files'][MODELS['voxtral'][0]],'projector_sha256':pins['files'][MODELS['voxtral'][1]],
             'binary_sha256':pins['files'][str(PREFIX)],'library_set_sha256':pins['library_set_sha256'],
             'source_sha256':pins['source_sha256'],'admission_id':'standing-r11-noncohort-diagnostic',
             'arm':'voxtral-jfk-diagnostic','clip_id':'public-jfk','command':cmd,'require_runtime_mapping':True,
             'cpu_only':True,'model_cohort_disjointness':'LEX2 QUALIFICATION-INTEGRATION.md exact WAV and PCM hash',
             'runtime_clock':'CLOCK_MONOTONIC','recipe':'upstream realtime transcription native stream, default stream step'}
    def runner(command_,env,bind):
        raw=run_process(command_,env,bind,deadline=time.monotonic()+215,stop_path=ROOT/'r11-voxtral'/'STOP')
        lines=raw.get('mappings') or []
        paths={line.split()[-1] for line in lines if '/' in line}
        needed=('libggml-base','libggml-cpu','libllama','libmtmd')
        raw['mapping_verified']=(all(any(n in p for p in paths) for n in needed)
            and all(os.path.realpath(p).startswith(prefix+'/') for p in paths)
            and not raw.get('kfd_fd_seen') and not any('libggml-hip' in p for p in paths))
        return raw
    engine=engine_for(cmd,prefix,OUT/'jfk.native-trace',runner)
    result=run_case(OUT,'jfk',binding,engine)
    if result is None: raise RuntimeError('existing exposure: no redraw')
    print(json.dumps({'diagnostic_only':True,'status':result['status'],'reaped':result.get('reaped'),
                      'mapping_verified':result.get('mapping_verified'),'kfd_fd_seen':result.get('kfd_fd_seen'),
                      'sidecar_present':'sidecar' in result.get('files',{})},sort_keys=True),flush=True)
    return 0 if result['status']=='complete' else 2
if __name__=='__main__': raise SystemExit(main())
