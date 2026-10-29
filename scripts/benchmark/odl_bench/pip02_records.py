"""Validate append-only PIP-02 native producer checkpoints before extraction."""
import hashlib
import json
from pathlib import Path

def sha(data): return hashlib.sha256(data).hexdigest()
def canon(v): return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(path): return json.loads(path.read_bytes())
def digest(path): return sha(path.read_bytes())

def odl(directory, page, plan_sha, protocol_sha, dependency_sha):
    d=Path(directory)
    if not d.exists(): return None
    if not (d/'DONE').is_file(): return None
    done=read(d/'DONE'); result=read(d/'result.json'); start=read(d/'start.json'); identity=read(d/'input-identity.json')
    if done.get('result_sha256')!=digest(d/'result.json') or result.get('rc')!=0 or result.get('partial') is not False or result.get('children_drained') is not True: raise ValueError('ODL completion integrity')
    if start.get('page')!=page or start.get('plan_sha256')!=plan_sha or start.get('protocol_sha256')!=protocol_sha or start.get('dependency_manifest_sha256')!=dependency_sha or start.get('mode')!='odl-2.5.0-docling-fast-full' or start.get('cohort_id')!='pip02-english-fresh-20261029': raise ValueError('ODL source binding')
    if identity.get('source_sha256')!=page['sha256'] or identity.get('pdf_sha256')!=start.get('input_sha256') or digest(d/'input.pdf')!=start['input_sha256']: raise ValueError('ODL input binding')
    raw=(d/'out/input.md').read_bytes()
    if sha(raw)!=result.get('prediction_sha256'): raise ValueError('ODL output hash')
    return raw.decode('utf-8')

def paddle(directory,page,campaign):
    d=Path(directory); campaign_sha=sha(canon(campaign)); key=sha(canon({'campaign':campaign_sha,'page':page}))
    start=d/(key+'.start'); result=d/(key+'.result')
    if not start.exists() and not result.exists(): return None
    if not start.exists(): raise ValueError('orphan Paddle result')
    if not result.exists(): return None
    a=read(start); b=read(result)
    if set(a)!={'campaign','page','started_monotonic_ns'} or a['campaign']!=campaign_sha or a['page']!=page or type(a['started_monotonic_ns']) is not int or a['started_monotonic_ns']<=0: raise ValueError('Paddle source binding')
    if b.get('start_sha256')!=digest(start) or not isinstance(b.get('markdown'),str) or sha(b['markdown'].encode())!=b.get('markdown_sha256'): raise ValueError('Paddle result hash')
    return b['markdown']
