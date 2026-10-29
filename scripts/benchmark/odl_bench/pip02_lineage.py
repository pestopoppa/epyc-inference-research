"""Infer lineage only when original packet or capture evidence supports it."""
import datetime, hashlib, json
from pathlib import Path
from .pip02_records import canon, sha
BASE=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029')
GRANTS=Path('/mnt/raid0/llm/tmp/compute-flags/grants')

def pinned(path,pins):
    path=Path(path).resolve(); expected=pins.get(str(path))
    if not expected: raise ValueError('packet evidence is not pinned: '+str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError('packet evidence changed: '+str(path))
    return path

def odl_lineage(page_dir, shard, pins):
    packet=BASE/('packets-final/odl-s0/draft.json' if shard==0 else f'packets-quad/odl-s{shard}/draft.json')
    if shard>4: raise ValueError('no supported armed packet')
    draft=json.loads(pinned(packet,pins).read_bytes()); start=json.loads((page_dir/'start.json').read_bytes())
    if packet.stat().st_mtime > start['started']: raise ValueError('packet is newer than page start')
    grant=GRANTS/(draft['id']+'.grant')
    if not grant.is_file() or grant.read_text()!=start.get('grant'): raise ValueError('original grant cannot be matched')
    same=[x for x in GRANTS.glob('*.grant') if x.is_file() and x.read_bytes()==grant.read_bytes()]
    if len(same)!=1: raise ValueError('grant bytes do not uniquely identify the packet')
    for field in ('launcher','job','source_manifest'):
        x=draft[field]; p=pinned(x['path'],pins)
        if sha(p.read_bytes())!=x['sha256']: raise ValueError('armed packet source mismatch')
    return {'verified':True,'basis':'inferred from recorded grant bytes and pinned armed packet; source hash is not recorded on the page','packet_sha256':sha(packet.read_bytes()),'grant_sha256':sha(grant.read_bytes())}

def paddle_lineage(directory,page,campaign,shard,pins):
    campaign_sha=sha(canon(campaign));key=sha(canon({'campaign':campaign_sha,'page':page}));finish=Path(directory)/(key+'.result')
    finish_ns=finish.stat().st_mtime_ns; candidates=[]
    for capture_path in (BASE/'attempts').glob('*/capture.json'):
        try:
            cap=json.loads(capture_path.read_bytes()); result_path=capture_path.parent/'result.json'
            if not result_path.is_file() or cap.get('smoke') is not False or cap.get('shard')!=shard: continue
            started=datetime.datetime.fromisoformat(cap['started_utc'].replace('Z','+00:00')).timestamp()
            if not (int(started*1e9)<=finish_ns<=result_path.stat().st_mtime_ns): continue
            result=json.loads(result_path.read_bytes())
            if result.get('population_sha256')!=campaign['population_sha256'] or result.get('scope')!=campaign['scope'] or result.get('states',{}).get(str(page['index']))!='complete': continue
            if cap.get('protocol_sha256')!='dc7d8e167edc79a07b9db10849940cb84a7f04c34f67274afbe067531d226ae3' or not cap.get('grant') or sha(cap['grant'].encode())!=cap.get('grant_sha256'): continue
            source=cap.get('source_pins')
            if not isinstance(source,dict) or not source: continue
            for name,expected in source.items():
                path=pinned(BASE/name,pins)
                if sha(path.read_bytes())!=expected: raise ValueError('captured source mismatch')
            candidates.append((capture_path,cap))
        except (KeyError,ValueError,OSError,TypeError): continue
    if len(candidates)!=1: raise ValueError('finishing capture is not uniquely identifiable')
    capture,cap=candidates[0]
    return {'verified':True,'basis':'inferred from unique overlapping capture and complete native state; mtime is not an original per-page capture','capture_sha256':sha(capture.read_bytes()),'grant_sha256':cap['grant_sha256']}
