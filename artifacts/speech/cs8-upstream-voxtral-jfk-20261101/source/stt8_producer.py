#!/usr/bin/env python3
"""Prospective STT8 raw producer interface; native engine is not yet enrolled."""
import json
from pathlib import Path
from exposure_journal import begin, digest, finish, write_once
from sidecar_reader import read as read_sidecar

REQUIRED=('pcm_sha256','input_sha256','plan_sha256','model_sha256','projector_sha256',
          'binary_sha256','library_set_sha256','source_sha256','admission_id','arm','clip_id')

def run_case(directory,key,binding,engine):
    if any(not isinstance(binding.get(x),str) or not binding[x] for x in REQUIRED):
        raise ValueError('missing pre-exposure binding')
    d=Path(directory)
    try: exposure=begin(d,key,binding)
    except FileExistsError: return None
    try:
        raw=engine(binding)
    except Exception as exc:
        result={'status':'unknown','error':f'{type(exc).__name__}: {exc}'}
        finish(d,key,exposure,result); return result
    result={'status':'unknown','reaped':raw.get('reaped'), 'rc':raw.get('rc'),
            'interrupted':raw.get('interrupted'), 'process':raw.get('process'),
            'mappings':raw.get('mappings'), 'mapping_verified':raw.get('mapping_verified'),
            'kfd_fd_seen':raw.get('kfd_fd_seen'), 'wall_s':raw.get('wall_s'), 'files':{}}
    for field in ('stdout','stderr','sidecar'):
        data=raw.get(field)
        if data is not None:
            if not isinstance(data,bytes): raise TypeError('native bytes required')
            write_once(d/(key+'.'+field),data); result['files'][field]=digest(data)
    mapping_ok = binding.get('require_runtime_mapping') is not True or result['mapping_verified'] is True
    if result['reaped'] is True and type(result['rc']) is int and result['interrupted'] is False and mapping_ok:
        if result['rc'] != 0:
            result['status']='failure'
        elif raw.get('sidecar') is not None:
            try:
                parsed=read_sidecar(raw['sidecar'])
                result.update({'status':'complete','text':parsed['text'],'token_ids':parsed['token_ids'],'termination':parsed['reason']})
            except (ValueError,UnicodeDecodeError) as exc:
                result['error']=str(exc)
    finish(d,key,exposure,result)
    return result

def read_case(directory,key,expected_binding):
    d=Path(directory)
    exp=(d/(key+'.exposure.json')).read_bytes()
    e=json.loads(exp); wrapped=json.loads((d/(key+'.result.json')).read_bytes())
    if e.get('binding')!=expected_binding or wrapped.get('exposure_sha256')!=digest(exp):
        raise ValueError('exposure binding mismatch')
    result=wrapped['result']; files=result.get('files',{})
    if not isinstance(files,dict) or not {'stdout','stderr'} <= files.keys():
        raise ValueError('missing native stdout/stderr')
    for field,h in files.items():
        if field not in ('stdout','stderr','sidecar') or digest((d/(key+'.'+field)).read_bytes())!=h:
            raise ValueError('native file hash mismatch')
    if result.get('status')=='complete':
        if expected_binding.get('require_runtime_mapping') is True and result.get('mapping_verified') is not True:
            raise ValueError('runtime mappings not verified')
        if result.get('reaped') is not True or result.get('interrupted') is not False or type(result.get('rc')) is not int or result['rc']!=0 or 'sidecar' not in files:
            raise ValueError('invalid completion witness')
        parsed=read_sidecar((d/(key+'.sidecar')).read_bytes())
        if parsed['text']!=result.get('text') or parsed['token_ids']!=result.get('token_ids') or parsed['reason']!=result.get('termination'):
            raise ValueError('sidecar/result mismatch')
    elif result.get('status')=='failure':
        if expected_binding.get('require_runtime_mapping') is True and result.get('mapping_verified') is not True:
            raise ValueError('runtime mappings not verified')
        if result.get('reaped') is not True or result.get('interrupted') is not False or type(result.get('rc')) is not int or result['rc']==0:
            raise ValueError('invalid failure witness')
    else:
        raise ValueError('unknown or incomplete result')
    return result
