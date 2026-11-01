#!/usr/bin/env python3
"""Write-once exposure/result records; dangling exposure is never redrawn."""
import hashlib
import json
import os
from pathlib import Path

def canonical(obj): return (json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode()
def digest(data): return hashlib.sha256(data).hexdigest()
def write_once(path, data):
    path=Path(path)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
    except BaseException:
        # A failed write is retained: exposure cannot be silently redrawn.
        raise

def begin(directory, key, binding):
    if not key or not all(c.isalnum() or c in '-_' for c in key): raise ValueError('invalid key')
    d=Path(directory); d.mkdir(parents=True,exist_ok=True)
    exposure=d/(key+'.exposure.json'); result=d/(key+'.result.json')
    if exposure.exists() or result.exists(): raise FileExistsError('existing or interrupted exposure; no redraw')
    data=canonical({'schema':'cs8-exposure/v1','binding':binding})
    write_once(exposure,data)
    return digest(data)

def finish(directory,key,exposure_sha256,result):
    d=Path(directory); e=d/(key+'.exposure.json'); raw=e.read_bytes()
    if digest(raw)!=exposure_sha256: raise ValueError('exposure tampered')
    write_once(d/(key+'.result.json'),canonical({'schema':'cs8-result/v1','exposure_sha256':exposure_sha256,'result':result}))
