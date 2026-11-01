#!/usr/bin/env python3
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/mnt/raid0/llm/tmp/codex3-cs8-stt8-20261101');sys.path.insert(0,str(ROOT))
from stt8_producer import read_case
D=ROOT/'r11-control/jfk-control'; raw=(D/'jfk.exposure.json').read_bytes(); binding=json.loads(raw)['binding']
assert binding['plan_sha256']=='77a08274d2840b8f0bde7d1bef122f369d37347f3b8450a25136b5f5ef32e0c4'
assert binding['input_sha256']=='59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
assert binding['pcm_sha256']=='a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9'
r=read_case(D,'jfk',binding)
assert r['status']=='complete' and r['mapping_verified'] is True and r['kfd_fd_seen'] is False
assert r['process']['uid']==1000 and r['reaped'] is True
assert r['termination']=='eog' and isinstance(r['text'],str) and len(r['token_ids'])>0
paths={line.split()[-1] for line in r['mappings'] if '/' in line}; prefix=str((ROOT/'prefix-cpu/bin').resolve())+'/'
assert all(os.path.realpath(p).startswith(prefix) for p in paths)
assert all(any(n in p for p in paths) for n in ('libggml-base','libggml-cpu','libllama','libmtmd'))
summary={'diagnostic_only':True,'scientific_score':None,'status':r['status'],'termination':r['termination'],'nonempty_text':bool(r['text']),'token_count':len(r['token_ids']),'mapping_paths':sorted(paths),'kfd_fd_seen':r['kfd_fd_seen'],'process':r['process'],'exposure_sha256':hashlib.sha256(raw).hexdigest(),'result_sha256':hashlib.sha256((D/'jfk.result.json').read_bytes()).hexdigest()}
print(json.dumps(summary,sort_keys=True,indent=2))
