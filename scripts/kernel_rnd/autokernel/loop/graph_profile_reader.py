"""SC55 prospective native decoder. No registration, tuple construction or grading."""
import hashlib,json,math,re
from pathlib import Path
from . import graph_profile_capture as c
NODE_COLUMNS = "idx op name src0_type src0_ne0 src0_ne1 src0_ne2 dst_ne0 dst_ne1 dst_ne2 compute_us wall_us evals wall_max_us wall_max_ev spikes thr_max_us thr_mean_us thr_min_us".split()
PATH_COLUMNS = "path calls_per_eval compute_ms wall_ms bytes_per_eval GBs_on_compute GBs_on_wall".split()
SOURCE_SHA256 = "1b30ebae2f08253ee74aca42cf9458ed6d0126c39cae19d38cd5291945db405c"

def json_bytes(raw):
    def pairs(items):
        result={}
        for key,value in items:
            c.require(key not in result,"duplicate field");result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda value:c.require(False,"nonfinite JSON"))

def bound_bytes(ref):
    c.exact(ref,{"path","size","sha256","device","inode","mtime_ns"})
    # One descriptor snapshot binds the exact decoded bytes and their receipt.
    import os,stat
    p=Path(ref['path']);c.require(p.is_absolute(),"absolute retained locator required")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);c.require(stat.S_ISREG(before.st_mode),"regular retained original required")
        c.require(before.st_size<=64*1024*1024,"oversized native file")
        chunks=[]
        while True:
            part=os.read(fd,1024*1024)
            if not part:break
            chunks.append(part)
        raw=b''.join(chunks);after=os.fstat(fd);loc=os.stat(p,follow_symlinks=False)
        c.require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns) and (after.st_dev,after.st_ino)==(loc.st_dev,loc.st_ino),"retained original changed")
        actual={"path":str(p),"size":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"device":after.st_dev,"inode":after.st_ino,"mtime_ns":after.st_mtime_ns}
        c.require(actual==ref,"retained custody mismatch")
        return raw
    finally:os.close(fd)

def uint(token):
    c.require(type(token) is str and re.fullmatch(r'0|[1-9][0-9]*',token) is not None,"invalid native integer")
    return int(token)

def number(token):
    try:value=float(token)
    except (ValueError,TypeError) as exc:raise c.Refusal("invalid native number") from exc
    c.require(math.isfinite(value) and value>=0,"negative/nonfinite native number")
    return value

def parse_nodes(raw,threads):
    lines=raw.decode('utf-8').splitlines();c.require(lines and lines[0].split('\t')==NODE_COLUMNS,"wrong 19-column SYNC1 header")
    rows=[];identities=set()
    for line in lines[1:]:
        cells=line.split('\t');c.require(len(cells)==19,"wrong node width")
        row=dict(zip(NODE_COLUMNS,cells))
        for field in ('idx','src0_ne0','src0_ne1','src0_ne2','dst_ne0','dst_ne1','dst_ne2','evals','wall_max_ev','spikes'):row[field]=uint(row[field])
        for field in ('compute_us','wall_us','wall_max_us','thr_max_us','thr_mean_us','thr_min_us'):row[field]=number(row[field])
        c.require(row['idx'] not in identities and row['op'] and row['src0_type'],"ambiguous node identity")
        identities.add(row['idx']);c.require(row['evals']>0 and row['spikes']<=row['evals'],"invalid eval/spike count")
        c.require(row['thr_min_us']<=row['thr_mean_us']<=row['thr_max_us'],"invalid aggregate thread ordering")
        if threads=='unavailable':c.require(all(row[field]==0 for field in ('thr_min_us','thr_mean_us','thr_max_us')),"thread counters without gate")
        rows.append(row)
    c.require(rows,"empty native node capture")
    return rows

def parse_paths(raw):
    lines=raw.decode('utf-8').splitlines();headers=[line for line in lines if line.startswith('[cpu_prof] PATHTABLE\t')]
    c.require(headers==['[cpu_prof] PATHTABLE\t'+'\t'.join(PATH_COLUMNS)],"missing/duplicate/wrong seven-column PATHROW header")
    rows=[];seen=set()
    for line in lines:
        if not line.startswith('[cpu_prof] PATHROW\t'):continue
        cells=line.split('\t')[1:];c.require(len(cells)==7,"wrong PATHROW width")
        c.require(cells[0] in ('dense_mul_mat','expert_mul_mat_id','lm_head') and cells[0] not in seen,"unknown/duplicate path")
        seen.add(cells[0]);rows.append(dict(zip(PATH_COLUMNS,[cells[0]]+[number(value) for value in cells[1:]])))
    c.require(rows,"empty native path capture")
    return rows

def interpret(graph,run,p,rows):
    # Prospective receipt contracts must be emitted at capture time; opaque historical bytes refuse.
    c.exact(graph,{'schema','source','nodes','external_tensors'})
    c.require(graph['schema']=='epyc.graph_structural_receipt.v1' and graph['source']==p['source'],"unknown graph/source receipt")
    c.require(type(graph['nodes']) is list and graph['nodes'],"missing graph topology")
    external={}
    for tensor in graph['external_tensors']:
        c.exact(tensor,{'id','type','ne','strides'})
        c.require(type(tensor['id']) is str and tensor['id'] not in external and tensor['type'],"ambiguous external tensor")
        for field in ('ne','strides'):
            c.require(type(tensor[field]) is list and len(tensor[field])==4,"unknown tensor geometry")
            for value in tensor[field]:c.integer(value)
        external[tensor['id']]=tensor
    nodes={}
    for node in graph['nodes']:
        c.exact(node,{'idx','op','name','src0_type','src0_ne','dst_ne','src_ids','op_params_hex'})
        c.integer(node['idx']);c.require(node['idx']==len(nodes),"graph indices not contiguous")
        c.require(type(node['op_params_hex']) is str and re.fullmatch(r'(?:[0-9a-f]{2})*',node['op_params_hex']) is not None,"unknown op parameters")
        for field in ('src0_ne','dst_ne'):
            c.require(type(node[field]) is list and len(node[field])==3,"unknown graph dimensions")
            for value in node[field]:c.integer(value)
        c.require(type(node['src_ids']) is list,"unknown graph edges")
        for edge in node['src_ids']:
            c.require(type(edge) is int and edge in nodes or type(edge) is str and edge in external,"unresolved topology edge")
        nodes[node['idx']]=node
    c.exact(run,{'schema','capture_id','owner','source','graph_identity_sha256','counter_reset','eval_events','expected_pernode_evals'})
    c.require(run['schema']=='epyc.graph_profile_run_receipt.v1' and run['capture_id']==p['capture_id'] and run['owner']==p['owner'] and run['source']==p['source'] and run['graph_identity_sha256']==p['graph']['identity_sha256'] and run['counter_reset'] is False,"unknown run identity/reset semantics")
    c.require(type(run['eval_events']) is list and run['eval_events'],"missing native eval history")
    accumulated=0
    for index,event in enumerate(run['eval_events']):
        c.exact(event,{'global_idx','n_nodes','structural_sha256','phase','accumulated'})
        c.integer(event['global_idx']);c.integer(event['n_nodes']);c.require(event['global_idx']==index,"incomplete global eval sequence")
        c.digest(event['structural_sha256']);c.require(event['phase'] in ('warmup','prefill','decode'),'unknown eval phase')
        selected=index>=p['eval']['skip']
        for suffix,compare in (('EQ',lambda x,y:x==y),('MIN',lambda x,y:x>=y),('MAX',lambda x,y:x<=y)):
            value=p['knobs']['GGML_CPU_PROF_NNODES_'+suffix]
            if value is not None:selected=selected and compare(event['n_nodes'],int(value))
        c.require(type(event['accumulated']) is bool and event['accumulated']==selected,"accumulation/filter mismatch")
        if selected:
            c.require(event['structural_sha256']==p['graph']['identity_sha256'] and event['n_nodes']==len(nodes) and event['phase']=='decode',"mixed graph/phase evidence")
            accumulated+=1
    c.require(accumulated>0,"no accumulated decode graph")
    c.require(type(run['expected_pernode_evals']) is list,'unknown pernode cardinality')
    expected={}
    for item in run['expected_pernode_evals']:
        c.exact(item,{'idx','evals'});c.integer(item['idx']);c.integer(item['evals'],1)
        c.require(item['idx'] in nodes and item['idx'] not in expected and item['evals']<=accumulated,'ambiguous expected pernode counts')
        expected[item['idx']]=item['evals']
    c.require({row['idx']:row['evals'] for row in rows}==expected,'native node cardinality/count mismatch')
    for row in rows:
        c.require(row['idx'] in nodes,"native node outside graph")
        node=nodes[row['idx']]
        c.require(all(row[field]==node[field] for field in ('op','name','src0_type')) and [row['src0_ne'+str(i)] for i in range(3)]==node['src0_ne'] and [row['dst_ne'+str(i)] for i in range(3)]==node['dst_ne'],"native structural metadata mismatch")
        c.require(row['evals']<=accumulated and row['wall_max_ev']<=accumulated,"node counts outside eval window")
    return {'accumulated_evals':accumulated,'structural_nodes':len(nodes)}

def read(seal_ref):
    envelope=json_bytes(bound_bytes(seal_ref))
    c.exact(envelope,{'schema','phase','pre','during','provenance','observation','closure','raw','observed_custody','writer_observed_at'})
    c.require(envelope['schema']==c.SCHEMA and envelope['phase']=='sealed',"unknown carrier")
    pre=json_bytes(bound_bytes(envelope['pre']));during=json_bytes(bound_bytes(envelope['during']))
    c.exact(pre,{'schema','phase','provenance','observed_custody','writer_observed_at'})
    c.exact(during,{'schema','phase','pre','observation','writer_observed_at'})
    p=envelope['provenance'];c.validate_pre(p);c.validate_observation(p,envelope['observation'])
    c.require(p['source']['profiler_source_sha256']==SOURCE_SHA256,"unsupported profiler grammar/source")
    c.require(pre['schema']==c.SCHEMA and pre['phase']=='pre' and during['schema']==c.SCHEMA and during['phase']=='during' and pre['provenance']==p and pre['observed_custody']==envelope['observed_custody'] and during['pre']==envelope['pre'] and during['observation']==envelope['observation'],"phase linkage mismatch")
    closed=envelope['closure'];c.exact(closed,{'ended_at','closed_at','producer_exited','all_handles_closed','owner','capture_id'})
    c.require(closed['producer_exited'] is True and closed['all_handles_closed'] is True and closed['owner']==p['owner'] and closed['capture_id']==p['capture_id'],"unknown closure")
    times=[p['recorded_at'],pre['writer_observed_at'],envelope['observation']['started_at'],envelope['observation']['observed_at'],during['writer_observed_at'],closed['ended_at'],closed['closed_at'],envelope['writer_observed_at']]
    parsed=[c.stamp(t) for t in times];c.require(parsed==sorted(parsed),"invalid phase window")
    c.exact(envelope['raw'],c.ROLES);raw={}
    for role,ref in envelope['raw'].items():
        c.require((ref is None)==(p['raw_paths'][role] is None),"unknown raw availability")
        if ref is not None:
            c.require(ref['path']==p['raw_paths'][role],"raw locator mismatch");raw[role]=bound_bytes(ref)
    custody=envelope['observed_custody'];c.exact(custody,{'binary','graph_identity','compiled_strings'})
    binary=bound_bytes(custody['binary'])
    c.require(hashlib.sha256(binary).hexdigest()==p['binary']['sha256'] and custody['binary']['mtime_ns']==p['binary']['mtime_ns'] and all(knob.encode() in binary for knob in c.KNOBS),"binary proof mismatch")
    for role in ('graph_identity','compiled_strings'):c.require(envelope['raw'][role]==custody[role],"pre/raw custody mismatch")
    c.require(hashlib.sha256(raw['compiled_strings']).hexdigest()==p['binary']['compiled_strings_sha256'] and all(knob.encode() in raw['compiled_strings'] for knob in c.KNOBS),"compiled strings mismatch")
    c.require(hashlib.sha256(raw['graph_identity']).hexdigest()==p['graph']['identity_sha256'],"graph identity mismatch")
    rows=parse_nodes(raw['pernode'],envelope['observation']['thread_availability']);paths=parse_paths(raw['log'])
    semantics=interpret(json_bytes(raw['graph_identity']),json_bytes(raw['run_receipt']),p,rows)
    for metric in p['measurement_metadata']['metrics']:
        selector=metric['selector']
        matches=[row for row in (rows if selector['kind']=='node' else paths) if row['idx' if selector['kind']=='node' else 'path']==selector['identity']]
        c.require(len(matches)==1,'native measurement selector absent/ambiguous')
        c.require(not (selector['field'].startswith('thr_') and envelope['observation']['thread_availability']=='unavailable'),'unavailable thread measurement')
    return {'schema':'epyc.graph_profile_decoded.v1','measurement_metadata':p['measurement_metadata'],'native_artifacts':envelope['raw'],'nodes':rows,'paths':paths,'semantics':semantics,'thread_availability':envelope['observation']['thread_availability'],'limitations':['owner topology/run assertions are not independently verified','thread0 timing and empty-node catchup do not establish imbalance','no measurement tuple or grade assigned']}
