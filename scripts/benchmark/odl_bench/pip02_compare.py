"""Offline PIP-02 common native scorer and compare-existing persistence."""
import argparse, hashlib, importlib.metadata, json, sys
from pathlib import Path
from . import pip02_records as records
from . import pip02_lineage as lineage
from .pip02_native_score import evaluate
from .comparison import build_existing_comparison, write_existing_comparison

PLAN='63028e74950ba7250f33cbce13f4f0bdd165bba2a075242db4475c26f645ed68'
PROTOCOL='dc7d8e167edc79a07b9db10849940cb84a7f04c34f67274afbe067531d226ae3'
DEPENDENCY='38d2d27b681bbe398107470b84447ac45f426aa5d0fc0ecfbdcce34daa64ab9f'
CONFIG='e424bd8aea788ad3662e1d5a695b99f4e0fdae13ed484160b909ac22681bbbc3'
LAYOUT={'inference.json':'2b68367c5b312a03de5a6e1642c597c8f95165a7e40cd59c6700cf4a5042f4fd','inference.pdiparams':'70bd316b0582769ec968829fd1feb1a6a58b7c941b938327e551b6b12b45c137','inference.yml':'506fcfac13b3b546ae40d7886b44126420f392adb694e3f8bb6a6286a1f90fdc'}
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def loaded_files():
    paths=set()
    for m in list(sys.modules.values()):
        path=getattr(m,'__file__',None)
        if not path: continue
        p=Path(path)
        if p.suffix=='.pyc': p=Path(importlib.util.source_from_cache(str(p)))
        if p.is_file(): paths.add(str(p.resolve()))
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)==6 and parts[5].startswith('/'):
            p=Path(parts[5])
            if p.is_file(): paths.add(str(p.resolve()))
    paths.add(str(Path(sys.executable).resolve()))
    return sorted(paths)
def pins_verify(pins):
    if sys.version!=pins['python']: raise ValueError('python identity changed')
    for path, expected in pins['files'].items():
        if digest(path)!=expected: raise ValueError('pinned bytes changed: '+path)
    missing=set(loaded_files())-set(pins['files'])
    if missing: raise ValueError('unlocked loaded modules: '+str(sorted(missing)))

def campaign(config):
    return {'scope':'pip02-english-fresh-20261029-Paddle-full-4096','population_sha256':PLAN,'layout_sha256':LAYOUT,'server_pins':config['server']['pins']}

def run(plan,gt,odl_root,paddle_root,config,image_root=None,pins=None):
    if len(plan['pages'])!=plan['total'] or len({p['image'] for p in plan['pages']})!=len(plan['pages']): raise ValueError('plan duplicate/size')
    output={arm:{'pages':[],'scores':[]} for arm in ('odl','paddle')}; total=0
    from src.dataset.end2end_dataset import End2EndDataset
    for pos,page in enumerate(plan['pages']):
        g=gt[page['index']]
        if g['page_info']['image_path']!=page['image'] or g['page_info']['page_attribute']['language']!='english': raise ValueError('GT identity')
        count=len(End2EndDataset.get_page_elements(None,g).get('table',[])); total+=count
        if count!=page['table_count']: raise ValueError('GT table count')
        sources={'odl':lambda:records.odl(Path(odl_root)/f'shard-{pos//25:02d}'/f'page-{pos:04d}',page,PLAN,PROTOCOL,DEPENDENCY), 'paddle':lambda:records.paddle(Path(paddle_root)/f'shard-{pos//5:02d}'/f'shard-{pos//5}',page,campaign(config))}
        for arm,reader in sources.items():
            try:
                text=reader()
                if text is not None and image_root is not None and digest(Path(image_root)/page['image'])!=page['sha256']: raise ValueError('source image hash mismatch')
                status='complete' if text is not None else 'unknown'; error=None
            except (OSError,ValueError,KeyError,TypeError,UnicodeError) as exc: text=None;status='invalid';error=str(exc)
            lineage_record={'verified':False,'basis':'source lineage not evaluated'}
            if text is not None and pins is not None:
                try:
                    lineage_record=(lineage.odl_lineage(Path(odl_root)/f'shard-{pos//25:02d}'/f'page-{pos:04d}',pos//25,pins) if arm=='odl' else lineage.paddle_lineage(Path(paddle_root)/f'shard-{pos//5:02d}'/f'shard-{pos//5}',page,campaign(config),pos//5,pins))
                except (OSError,ValueError,KeyError,TypeError) as exc: lineage_record={'verified':False,'basis':str(exc)}
            values=evaluate(g,text) if text is not None else None
            if values is not None and len(values)!=count: raise ValueError('matcher denominator mismatch')
            if values: output[arm]['scores'].extend(values)
            output[arm]['pages'].append({'image':page['image'],'status':status,'gt_tables':count,'error':error,'source_lineage':lineage_record,'prediction_sha256':hashlib.sha256(text.encode()).hexdigest() if text is not None else None,'tables':values})
    if total!=plan['table_regions']: raise ValueError('population GT total')
    for arm in output:
        v=output[arm]; all_scores=v.pop('scores'); n=len(all_scores); complete=sum(x['status']=='complete' for x in v['pages'])
        v['coverage']={'pages_complete':complete,'pages_total':len(plan['pages']),'gt_tables_scored':n,'gt_tables_total':total}
        v['full_population_defined']=complete==len(plan['pages']) and n==total
        v['source_lineage_verified']=all(x['source_lineage']['verified'] for x in v['pages'])
        v['gate_eligible']=v['full_population_defined'] and v['source_lineage_verified']
        v['table_weighted_means']={k:sum(x[k] for x in all_scores)/n if n else None for k in ('teds','teds_structure_only')}
    return output

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',required=True); ap.add_argument('--gt',required=True); ap.add_argument('--config',required=True); ap.add_argument('--image-root',required=True); ap.add_argument('--odl-root',required=True); ap.add_argument('--paddle-root',required=True); ap.add_argument('--pins',required=True); ap.add_argument('--pins-sha256',required=True); ap.add_argument('--out-dir',required=True); a=ap.parse_args()
    if digest(a.pins)!=a.pins_sha256: raise ValueError('manifest identity')
    pins=json.loads(Path(a.pins).read_bytes()); pins_verify(pins)
    if digest(a.plan)!=PLAN or digest(a.config)!=CONFIG or digest(a.gt)!=pins['gt_sha256']: raise ValueError('population/config/GT identity')
    output=run(json.loads(Path(a.plan).read_bytes()),json.loads(Path(a.gt).read_bytes()),a.odl_root,a.paddle_root,json.loads(Path(a.config).read_bytes()),a.image_root,pins['files'])
    pins_verify(pins)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=False)
    report={'schema':'pip02.common-teds.v1','pins_sha256':a.pins_sha256,'plan_sha256':PLAN,'gt_sha256':pins['gt_sha256'],'arms':output,'joint_full_population_defined':all(v['full_population_defined'] for v in output.values()),'gate_eligible':all(v['gate_eligible'] for v in output.values())}
    (out/'coverage-and-scores.json').write_text(json.dumps(report,indent=2)+'\n')
    metric_rows=[]
    for arm,v in output.items():
        for key in ('teds','teds_structure_only'):
            metric_rows.append({'engine':arm,'metric_family':'table_fidelity','metric_name':key,'value':v['table_weighted_means'][key] if report['gate_eligible'] else None,'n':v['coverage']['gt_tables_scored'],'detail':f"higher better; table-weighted; full population defined={v['full_population_defined']}; coverage={v['coverage']}"})
    rowset=out/'row_set.json';rowset.write_text(json.dumps({'gt_json':a.gt,'metric_rows':metric_rows},indent=2)+'\n')
    payload=build_existing_comparison(artifacts=[(None,rowset)],prediction_dirs=[],result_dir=None,gt_json=a.gt,match_method='native-simple-table')
    write_existing_comparison(payload,out)
    print(json.dumps({'out_dir':str(out),'joint_full_population_defined':report['joint_full_population_defined'],'coverage':{k:v['coverage'] for k,v in output.items()}}))
if __name__=='__main__':main()
