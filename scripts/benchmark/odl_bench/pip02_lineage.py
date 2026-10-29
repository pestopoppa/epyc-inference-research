"""Infer lineage only when original packet or capture evidence supports it."""
import datetime, hashlib, json, re
from pathlib import Path
from .pip02_records import canon, sha
BASE=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029')
ARMED=Path('/mnt/raid0/llm/tmp/compute-flags/ready/armed')

def pinned(path,pins):
    path=Path(path).resolve(); expected=pins.get(str(path))
    if not expected: raise ValueError('packet evidence is not pinned: '+str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError('packet evidence changed: '+str(path))
    return path

def odl_lineage(page_dir, shard, pins):
    if shard>4: raise ValueError('no supported armed packet')
    ident='codex3-pip02-fresh-odl-s0-20261029' if shard==0 else f'codex3-pip02-fresh-odl-quad-s{shard}-20261029'
    armed=ARMED/(ident+'.json')
    packet=pinned(armed,pins); draft=json.loads(packet.read_bytes()); start=json.loads((page_dir/'start.json').read_bytes())
    if draft.get('id')!=ident or packet.stat().st_mtime>start['started']: raise ValueError('armed packet identity/timing')
    for field in ('launcher','job','source_manifest'):
        x=draft[field]; p=pinned(x['path'],pins)
        if sha(p.read_bytes())!=x['sha256']: raise ValueError('armed packet source mismatch')
    grant=start.get('grant',''); match=re.fullmatch(r'regions=([^ ]+) start=([^ ]+) end=([^\s]+)\s*',grant)
    if not match: raise ValueError('unknown grant schema')
    region,begin,end=match.groups(); begin_dt=datetime.datetime.fromisoformat(begin.replace('Z','+00:00')); end_dt=datetime.datetime.fromisoformat(end.replace('Z','+00:00'))
    if region!=draft['regions'] or not begin_dt.timestamp()<=start['started']<=end_dt.timestamp(): raise ValueError('page outside recorded grant')
    log=Path(draft['launcher_log']); raw=log.read_bytes(); lines=raw.decode().splitlines()
    starts=[x for x in lines if f'[{ident}] START regions={region} ' in x and f'end={end}' in x]
    commands=[x for x in lines if '--inner --id '+ident+' ' in x and draft['job']['path'] in x and f'--shard {shard} ' in x]
    if len(starts)!=1 or len(commands)!=1: raise ValueError('original launcher record is missing or ambiguous')
    return {'verified':True,'basis':'inferred from recorded page grant, pre-start armed packet, launcher START and actual command; source hash is not recorded on the page','packet_sha256':sha(packet.read_bytes()),'launcher_log_sha256':sha(raw),'grant_sha256':sha(grant.encode())}

def paddle_lineage(directory,page,campaign,shard,pins):
    campaign_sha=sha(canon(campaign));key=sha(canon({'campaign':campaign_sha,'page':page}));finish=Path(directory)/(key+'.result')
    finish_ns=finish.stat().st_mtime_ns; candidates=[]
    prefix='region-'+sha(canon(page))[:24]+'-'
    region_starts=sorted(Path(directory).glob(prefix+'*.start.json'))
    if not region_starts: raise ValueError('no request-origin evidence for finishing capture')
    origins=set()
    for region_start in region_starts:
        a=json.loads(region_start.read_bytes()); response_path=region_start.with_name(region_start.name.replace('.start.json','.result.json')); b=json.loads(response_path.read_bytes())
        provenance=a['provenance']; identity=provenance['identity']
        if identity['campaign_sha256']!=campaign_sha or sha(bytes.fromhex(a['body_hex']))!=a['body_sha256'] or b['request_sha256']!=a['body_sha256'] or b['start_sha256']!=sha(region_start.read_bytes()) or sha(bytes.fromhex(b['body_hex']))!=b['response_sha256'] or not 200<=b['status']<300: raise ValueError('region request/response identity')
        origins.add(provenance['origin_capture_sha256'])
    validated_origins=set()
    for capture_path in (BASE/'attempts').glob('*/capture.json'):
        try:
            cap=json.loads(capture_path.read_bytes()); result_path=capture_path.parent/'result.json'
            if cap.get('smoke') is not False or cap.get('shard')!=shard: continue
            if cap.get('protocol_sha256')!='dc7d8e167edc79a07b9db10849940cb84a7f04c34f67274afbe067531d226ae3' or not cap.get('grant') or sha(cap['grant'].encode())!=cap.get('grant_sha256'): continue
            source=cap.get('source_pins')
            if not isinstance(source,dict) or not source: continue
            for name,expected in source.items():
                path=pinned(BASE/name,pins)
                if sha(path.read_bytes())!=expected: raise ValueError('captured source mismatch')
            if sha(capture_path.read_bytes()) in origins: validated_origins.add(sha(capture_path.read_bytes()))
            if not result_path.is_file(): continue
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
    if origins != validated_origins: raise ValueError('region origins are not all supported by captured source/grant')
    if len(candidates)!=1: raise ValueError('finishing capture is not uniquely identifiable')
    capture,cap=candidates[0]
    return {'verified':True,'basis':'inferred from unique overlapping capture and complete native state; mtime is not an original per-page capture','capture_sha256':sha(capture.read_bytes()),'grant_sha256':cap['grant_sha256']}
