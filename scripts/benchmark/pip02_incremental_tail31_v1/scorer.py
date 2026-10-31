import sys,json,hashlib,importlib.metadata,os
from pathlib import Path
SRC=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from odl_bench import pip02_records as rec,pip02_lineage as lin,pip02_compare as compare
from odl_bench.pip02_native_score import evaluate
from src.dataset.end2end_dataset import End2EndDataset
R=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029');O=Path(os.environ['PIP_PADDLE_STAGE'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=json.loads((R/'scoring-prep/pins_v1.json').read_text());assert sys.version==pin['python'];assert sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions())==pin['packages']
for p,h in pin['files'].items():assert sha(p.replace('/home/node/','/home/daniele/'))==h,p
source=SRC/'odl_bench'; hashes={n:sha(source/n) for n in ['pip02_records.py','pip02_lineage.py','pip02_compare.py','pip02_native_score.py']}
assert hashes=={'pip02_records.py':'2d364d7a7ef1d47b21fb8f3352ff4ffe98e93c682da3996f6e1270c8193c81cd','pip02_lineage.py':'a42ea9e671cb6a1956b592a38ee87258bb8f0552e1de2ed84db357d92e25ebe0','pip02_compare.py':'37e9d673ba46ac16c5f8169db97290933b3c63fb48ef133b825d5db3a9ba995e','pip02_native_score.py':'27f0b630b6812aecad35d1c65329841a5011e421326e81f9f14c8c66a6fdbe5e'}
config=Path('/mnt/raid0/llm/tmp/pip02-pipeline-prep-20261029/paddle-cpu-campaign-config-v3.json');assert sha(config)==compare.CONFIG
plan=json.loads(Path(pin['plan']).read_text());gt=json.loads(Path(pin['gt']).read_text());campaign=compare.campaign(json.loads(config.read_text()));pins={}
for version,(root,h) in lin.PADDLE_ROOTS.items():
 p=root/'source-pins.json';assert sha(p)==h;pins[str(p.resolve())]=h
 for k,v in json.loads(p.read_text()).items():pins[str(Path(k).resolve())]=v
pins['/mnt/raid0/llm/tmp/codex3-filing-pip-dtap-c6-20261031-0230/repair/head/launch-p0.sh']='ae394dd5e671ff82ef472218c527675810f0d586b2a8ae6a68c3133a9e8e6e2f'
pins['/mnt/raid0/llm/tmp/codex3-filing-pip-dtap-c6-20261031-0230/repair/head/pool.py']='71eee5884d5c5bc92f195ebfad7291eb3deb5adb462d0f98bd6b1604e4161828'
r29_catalog=Path('/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031/source-pins.json')
assert sha(r29_catalog)=='63f15f638c88e3de96c584f621a71fb3ac778855b58cd06bef27bb64eaf2cec3'
pins[str(r29_catalog)]=sha(r29_catalog)
pins.update(json.loads(r29_catalog.read_bytes()))
pins['/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031/launch-p0.sh']='dbe28a4f43d0890a14a80037a80ebc2b52d49ab8ac3fee13b960f62e1b20fbbd'
pins['/mnt/raid0/llm/tmp/codex3-pip-pool2-r30-v2-20261031/launch-p2.sh']='2dbe8265199aabaae505f17520e4a84b2cae67fc7b7f218392d201353061160c'
recovery=Path('/mnt/raid0/llm/tmp/codex3-pip-head-assembly-20261031-0530/recovery-1/output');recovery_start=recovery/'RECOVERY-START.json';assert sha(recovery_start)=='5bb8f7ccf46655decd5fe8c5fb21f36406f7432206cb44a7f58171bcbec089c6'
pins[str(recovery_start.resolve())]='5bb8f7ccf46655decd5fe8c5fb21f36406f7432206cb44a7f58171bcbec089c6'
for n,h in json.loads(recovery_start.read_bytes())['inputs_sha256'].items():pins[str(Path(n).resolve())]=h
for n,h in {'RECOVERY-FINISH.json':'335777dd25717190398f7d65026821dc8b6dcdcbbb77f55d35f497925775eb20','page-0000.recovery.json':'8ee6a158fa5a4c3b3b472ad670a412307ce0e133b4b8a6e7b79b71a2a4801af3','page-0034.recovery.json':'3291129c94de5008687337e53d1366d8b87aed33828664c4f2a1ed98b3ee4ce9'}.items():pins[str((recovery/n).resolve())]=h
pins['/mnt/raid0/llm/tmp/codex3-pip-pool3-recovery-20261031-0606/launch-p3.sh']='5f10c0b6d2965f0de22164b7c38ddc2ce21317350fe700ceaa2eb13c49d76047'
fresh_manifest=R/'PADDLE-FRESH-SOURCES.sha256'
assert sha(fresh_manifest)=='8c8508862d3e9be31ce184da53f5e977eb3935f80f9d23856eb3a845a9175ddb'
pins[str(fresh_manifest)]=sha(fresh_manifest)
for line in fresh_manifest.read_text().splitlines():
 digest,name=line.split('  ',1);pins[name]=digest
roots=[R/'campaigns-fresh',Path('/mnt/micron/codex3-pip-split-r22-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-relocatable-r25-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-head-r26-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool2-r30-v2-20261031/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool3-tail-r31-20261031/campaigns-fresh')]
for sh in map(int,sys.argv[1:]):
 dest=O/f's{sh:03}.json'
 if dest.exists():raise ValueError('refuse overwrite: '+str(dest))
 d=rec.LayeredPaths([p/f'shard-{sh:02}'/f'shard-{sh}' for p in roots]);pages=[]
 for pos in range(sh*5,min(sh*5+5,len(plan['pages']))):
  row=plan['pages'][pos];g=gt[row['index']];assert g['page_info']['image_path']==row['image'] and g['page_info']['page_attribute']['language']=='english';assert len(End2EndDataset.get_page_elements(None,g).get('table',[]))==row['table_count']
  x={'position':pos,'image':row['image'],'gt_index':row['index'],'gt_tables':row['table_count'],'status':'unknown','reason':None,'tables':None}
  selected_sources={}
  try:
   key=rec.sha(rec.canon({'campaign':rec.sha(rec.canon(campaign)),'page':row}));prefix='region-'+rec.sha(rec.canon(row))[:24]+'-'
   def snapshot():
    paths=d.glob(key+'.*')+d.glob(prefix+'*')
    return {str(p):sha(p) for p in paths}
   before=snapshot()
   text=rec.paddle(d,row,campaign)
   if text is None:x['reason']='no durable completed page'
   else:
    v=lin.paddle_lineage(d,row,campaign,sh,pins);x['source_lineage']=v;x['prediction_sha256']=hashlib.sha256(text.encode()).hexdigest()
    for path,digest in v.get('selected_captures',{}).items():
     if sha(path)!=digest:raise ValueError('selected capture changed before evaluation')
    for sources in v.get('registered_source_pins',{}).values():
     for path,digest in sources.items():
      if sha(path)!=digest:raise ValueError('selected producer source changed before evaluation')
      selected_sources[path]=digest
    if not v.get('verified'):x.update(status='unverified',reason=v['basis'])
    else:
     image_path='/mnt/raid0/llm/datasets/omnidocbench/images/'+row['image'];assert sha(image_path)==row['sha256'];selected_sources[image_path]=row['sha256'];vals=evaluate(g,text);assert len(vals)==row['table_count'];x.update(status='complete',reason='native finish and request origins verified',tables=vals)
  except Exception as e:x.update(status='error',reason=f'{type(e).__name__}: {e}')
  else:
   try:
    if snapshot()!=before:raise ValueError('captured page input changed during scoring')
    x['input_sha256']={**before,**selected_sources}
   except Exception as e:x.update(status='error',reason=f'{type(e).__name__}: {e}',tables=None)
  pages.append(x)
 vals=[t for p in pages if p['tables'] is not None for t in p['tables']];out={'arm':'paddle','cohort':'pip02-english-fresh-20261029','shard':sh,'plan_sha256':sha(pin['plan']),'gt_sha256':sha(pin['gt']),'config_sha256':sha(config),'scoring_sources_sha256':hashes,'pages':pages,'coverage':{'pages_complete':sum(p['status']=='complete' for p in pages),'pages_total':len(pages),'gt_tables_scored':len(vals),'gt_tables_total':sum(p['gt_tables'] for p in pages)},'means':{k:sum(t[k] for t in vals)/len(vals) if vals else None for k in ['teds','teds_structure_only']}}
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(sh,out['coverage'],out['means'],flush=True)
