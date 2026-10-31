"""Prospective score-invocation receipts for PIP-02; no retrospective enrollment."""
import datetime, hashlib, json, os, uuid
from pathlib import Path

SCHEMA = 'epyc.pip02.incremental_teds.v1'
SCOPE = 'selected-warrant-and-metric-inputs'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def digest(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write_new(path, obj):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def begin(path, arm, shard, files, argv):
    """Called before the scorer subprocess starts; files are fixed call-level inputs."""
    if arm not in ('odl','paddle') or type(shard) is not int or shard < 0 or not files:
        raise ValueError('invalid invocation identity')
    files={str(Path(p).absolute()):digest(p) for p in files}
    start={'schema':SCHEMA,'id':str(uuid.uuid4()),'started_utc':now(),'arm':arm,'shard':shard,
           'inputs_scope':SCOPE,'argv':list(argv),'inputs':files}
    write_new(path,start)
    return start

def finish(start_path, result_path, receipt_path):
    """Bind native result; recheck fixed and per-page selected inputs after evaluation."""
    raw=Path(start_path).read_bytes(); start=json.loads(raw); result_path=Path(result_path).absolute()
    if start.get('schema')!=SCHEMA or start.get('inputs_scope')!=SCOPE: raise ValueError('bad start')
    result=json.loads(result_path.read_bytes())
    if result['arm']!=start['arm'] or result['shard']!=start['shard']: raise ValueError('identity mismatch')
    paths=dict(start['inputs'])
    for page in result['pages']:
        lineage=page.get('source_lineage',{})
        sources=[page.get('input_sha256',{}),lineage.get('selected_captures',{}),lineage.get('selected_registries',{})]
        sources.extend(lineage.get('registered_source_pins',{}).values())
        for source in sources:
            for path, expected in source.items():
                if path in paths and paths[path]!=expected: raise ValueError('conflicting input digest')
                paths[path]=expected
    for path, expected in paths.items():
        if digest(path)!=expected: raise ValueError('input changed during scoring: '+path)
    body={'schema':SCHEMA,'id':start['id'],'start_path':str(Path(start_path).absolute()),
          'start_sha256':hashlib.sha256(raw).hexdigest(),'ended_utc':now(),
          'inputs_scope':SCOPE,'excluded':'unselected producer catalog reads',
          'selected_inputs':paths,'result_path':str(result_path),'result_sha256':digest(result_path)}
    write_new(receipt_path, {'payload':body,'payload_sha256':hashlib.sha256(canonical(body)).hexdigest()})
