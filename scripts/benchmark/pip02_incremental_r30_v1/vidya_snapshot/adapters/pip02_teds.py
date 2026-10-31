"""Strict prospective PIP-02 score receipt projection; historic scores are declined."""
import datetime, hashlib, json, math, re, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claim_tuple import ClaimTuple, ProjectionError, register

ADAPTER_ID='vidya.adapters.pip02_teds/v1'
AUTHORITY='measurement'
SCHEMA='epyc.pip02.incremental_teds.v1'
SCOPE='selected-warrant-and-metric-inputs'
HASH=re.compile(r'^[0-9a-f]{64}$')

def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(path): return Path(path).read_bytes()
def stamp(v):
    x=datetime.datetime.fromisoformat(v)
    if x.tzinfo is None: raise ValueError('timestamp lacks zone')
    return x

def native_rows(path):
    path=Path(path).absolute()
    if not path.exists(): return ()
    try:
        raw=read(path); env=json.loads(raw); body=env['payload']
        if set(env)!={'payload','payload_sha256'} or sha(canonical(body))!=env['payload_sha256']: raise ValueError('envelope mismatch')
        if set(body)!={'schema','id','start_path','start_sha256','ended_utc','inputs_scope','excluded','selected_inputs','result_path','result_sha256'} or body['schema']!=SCHEMA or body['inputs_scope']!=SCOPE or body['excluded']!='unselected producer catalog reads': raise ValueError('receipt shape/scope')
        st_raw=read(body['start_path']); start=json.loads(st_raw)
        if sha(st_raw)!=body['start_sha256'] or set(start)!={'schema','id','started_utc','arm','shard','inputs_scope','argv','inputs'} or start['schema']!=SCHEMA or start['id']!=body['id'] or start['inputs_scope']!=SCOPE or stamp(start['started_utc'])>stamp(body['ended_utc']): raise ValueError('start binding')
        if start['arm'] not in ('odl','paddle') or type(start['shard']) is not int or start['shard']<0 or not isinstance(start['argv'],list) or not start['argv']: raise ValueError('invocation identity')
        selected=body['selected_inputs']
        if not isinstance(selected,dict) or not selected or not isinstance(start['inputs'],dict) or any(selected.get(k)!=v for k,v in start['inputs'].items()): raise ValueError('input inventory')
        for name,d in selected.items():
            if not isinstance(name,str) or not Path(name).is_absolute() or not isinstance(d,str) or not HASH.fullmatch(d) or sha(read(name))!=d: raise ValueError('selected input mismatch')
        result_raw=read(body['result_path'])
        if sha(result_raw)!=body['result_sha256']: raise ValueError('result digest')
        result=json.loads(result_raw)
        if result['arm']!=start['arm'] or result['shard']!=start['shard'] or not isinstance(result['cohort'],str) or not result['cohort'] or len(result['cohort'])>80: raise ValueError('result identity')
        for field in ('plan_sha256','gt_sha256'):
            if result.get(field) not in selected.values(): raise ValueError('plan or GT not bound')
        if start['arm']=='paddle':
            if result.get('config_sha256') not in selected.values(): raise ValueError('config not bound')
            if any(d not in selected.values() for d in result['scoring_sources_sha256'].values()): raise ValueError('scoring source not bound')
        elif result.get('scorer_sha256') not in selected.values() or result.get('pins_sha256') not in selected.values(): raise ValueError('ODL scorer or pins not bound')
        plan_paths=[k for k,v in selected.items() if v==result['plan_sha256']]
        plan=json.loads(read(plan_paths[0])); plan_pages=plan['pages']
        pages=result['pages']; seen=set(); values=[]; total=0; complete=0
        if not isinstance(pages,list) or not pages: raise ValueError('no pages')
        for page in pages:
            pos=page['position']; count=page['gt_tables']; status=page['status']; tables=page['tables']
            if type(pos) is not int or pos in seen or type(count) is not int or count<0 or not isinstance(page['image'],str) or not page['image'] or type(page['gt_index']) is not int: raise ValueError('page identity')
            width=5 if start['arm']=='paddle' else 25
            if not start['shard']*width<=pos<min((start['shard']+1)*width,len(plan_pages)): raise ValueError('page outside shard')
            planned=plan_pages[pos]
            if planned['image']!=page['image'] or planned['index']!=page['gt_index'] or planned['table_count']!=count: raise ValueError('page differs from plan')
            seen.add(pos); total+=count
            if status=='complete':
                if planned.get('sha256') not in selected.values(): raise ValueError('source image not bound')
                if not isinstance(tables,list) or len(tables)!=count: raise ValueError('table denominator')
                if start['arm']=='paddle':
                    lin=page.get('source_lineage',{})
                    if lin.get('verified') is not True or not lin.get('selected_captures') or not lin.get('registered_source_pins') or not lin.get('selected_registries'): raise ValueError('completed page lacks selected lineage')
                    capture_hashes=set(lin['selected_captures'].values())
                    claimed=set(lin.get('origin_capture_sha256',[]))|{lin.get('finish_capture_sha256')}
                    if None in claimed or capture_hashes!=claimed or set(lin['registered_source_pins'])!=claimed: raise ValueError('origin/finish capture inventory mismatch')
                    for capture,source_pins in lin['registered_source_pins'].items():
                        if not isinstance(source_pins,dict) or not source_pins or any(selected.get(k)!=v for k,v in source_pins.items()): raise ValueError('producer source not bound')
                else:
                    if result.get('source_lineage',{}).get(str(pos),{}).get('verified') is not True: raise ValueError('ODL lineage absent')
                complete+=1
                gt_seen=set();pred_seen=set()
                for table in tables:
                    if set(table)!={'gt_idx','pred_idx','teds','teds_structure_only'} or not isinstance(table['gt_idx'],list) or not isinstance(table['pred_idx'],list): raise ValueError('matching')
                    if len(table['gt_idx'])!=1 or type(table['gt_idx'][0]) is not int or table['gt_idx'][0] not in range(count) or table['gt_idx'][0] in gt_seen: raise ValueError('duplicate or invalid GT match')
                    pred=table['pred_idx']
                    if len(pred)!=1 or not (pred[0]=='' or (type(pred[0]) is int and pred[0]>=0)): raise ValueError('invalid prediction match index')
                    if pred[0]!='':
                        if pred[0] in pred_seen: raise ValueError('duplicate prediction match')
                        pred_seen.add(pred[0])
                    gt_seen.add(table['gt_idx'][0])
                    if any(type(table[k]) not in (float,int) or not math.isfinite(table[k]) or not 0<=table[k]<=1 for k in ('teds','teds_structure_only')): raise ValueError('score')
                    values.append(table)
            elif status in ('unknown','error','unverified'):
                if tables is not None or not isinstance(page.get('reason'),str) or not page['reason']: raise ValueError('unknown must have reason and no scores')
            else: raise ValueError('unknown status')
            for mapping in (page.get('input_sha256',{}),page.get('source_lineage',{}).get('selected_captures',{}),page.get('source_lineage',{}).get('selected_registries',{})):
                if any(selected.get(k)!=v for k,v in mapping.items()): raise ValueError('page input missing')
        if seen!=set(range(start['shard']*width,min((start['shard']+1)*width,len(plan_pages)))): raise ValueError('shard pages missing')
        coverage={'pages_complete':complete,'pages_total':len(pages),'gt_tables_scored':len(values),'gt_tables_total':total}
        if result['coverage']!=coverage: raise ValueError('coverage mismatch')
        for k in ('teds','teds_structure_only'):
            expected=sum(t[k] for t in values)/len(values) if values else None
            if result['means'][k]!=expected: raise ValueError('mean mismatch')
        if start['arm']=='odl': return ()  # No validated model applicability; receipt remains retained.
        return tuple({'path':str(path),'sha256':sha(raw),'body':body,'start':start,'result':result,'metric':k} for k in ('teds','teds_structure_only') if values)
    except (OSError,ValueError,TypeError,KeyError,IndexError) as exc:
        raise ProjectionError('PIP-02 score receipt refused: '+str(exc)) from exc

@register('pip02-incremental-teds',source_class='measurement')
def project(native):
    rows=native_rows(native['path'])
    if native not in rows: raise ProjectionError('receipt changed after read')
    start=native['start']; result=native['result']; metric=native['metric']; cov=result['coverage']
    applicability={}
    if start['arm']=='paddle':
        config_path=next(k for k,v in native['body']['selected_inputs'].items() if v==result['config_sha256'])
        server=json.loads(read(config_path))['server']
        applicability={'model_file':Path(server['model']).name,'model_sha256':server['pins']['model']}
    complete=cov['pages_complete']==cov['pages_total'] and cov['gt_tables_scored']==cov['gt_tables_total']
    mechanism=f"pip02-score:{start['arm']}:{result['cohort']}:shard-{start['shard']}:sha256-{native['body']['result_sha256']}:metric-{metric}"
    return ClaimTuple(applicability=applicability,measurement_id='pip02-teds:'+start['id']+':'+metric,metric=metric,value=result['means'][metric],date=start['started_utc'],category='CANDIDATE',metric_direction='higher_better',protocol_id='',reps=1,reps_basis='one scoring invocation; table denominator is coverage, not repeats',unit='score',claim=f"{result['cohort']} {start['arm']} shard {start['shard']} ({5 if start['arm']=='paddle' else 25}-page shard convention) {metric} {result['means'][metric]}; {cov['gt_tables_scored']}/{cov['gt_tables_total']} GT tables scored, {cov['pages_complete']}/{cov['pages_total']} pages complete; model scope is request-config identity, not proof of loaded runtime bytes; partial coverage remains explicit.",attestation_path=native['path'],attestation_sha256=native['sha256'],attestation_present=True,attestation_verified=True,source_kind='pip02-incremental-teds',observation={'outcome_native':'complete shard coverage' if complete else 'partial shard coverage','mechanism':{'id':mechanism}},extra={'coverage':cov,'cohort':result['cohort'],'arm':start['arm'],'shard':start['shard'],'inputs_scope':SCOPE})
