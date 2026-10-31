#!/usr/bin/env python3
"""Read-only per-shard input epochs and write-once generation setup."""
import hashlib,json,os,sys,subprocess
from pathlib import Path
S=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029/scoring-shards-20261030');R=S.parent
SRC=Path(__file__).resolve().parent
RESEARCH=SRC;sys.path.insert(0,str(RESEARCH))
from odl_bench import pip02_records as rec,pip02_lineage as lin
sha=lambda data:hashlib.sha256(data).hexdigest()
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(p):return sha(Path(p).read_bytes())
def filehash(p):
 try:return digest(p)
 except FileNotFoundError:return None
plan=json.loads((R/'english755-fresh-plan.json').read_bytes())
config=json.loads(Path('/mnt/raid0/llm/tmp/pip02-pipeline-prep-20261029/paddle-cpu-campaign-config-v3.json').read_bytes())
layout={'inference.json':'2b68367c5b312a03de5a6e1642c597c8f95165a7e40cd59c6700cf4a5042f4fd','inference.pdiparams':'70bd316b0582769ec968829fd1feb1a6a58b7c941b938327e551b6b12b45c137','inference.yml':'506fcfac13b3b546ae40d7886b44126420f392adb694e3f8bb6a6286a1f90fdc'}
campaign={'scope':'pip02-english-fresh-20261029-Paddle-full-4096','population_sha256':'63028e74950ba7250f33cbce13f4f0bdd165bba2a075242db4475c26f645ed68','layout_sha256':layout,'server_pins':config['server']['pins']}
roots=[R/'campaigns-fresh',Path('/mnt/micron/codex3-pip-split-r22-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-relocatable-r25-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-head-r26-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool2-r30-v2-20261031/campaigns-fresh')]
def capture_index():
 found={}
 for root,_ in lin.PADDLE_ROOTS.values():
  for path in (root/'attempts').glob('*/capture.json'):
   raw=path.read_bytes(); found.setdefault(sha(raw),[]).append((str(path.resolve()),str((root/'source-pins.json').resolve())))
 fresh=R/'attempts'
 for path in fresh.glob('codex3-pip02-fresh-s1-20261029-*/capture.json'):
  raw=path.read_bytes();found.setdefault(sha(raw),[]).append((str(path.resolve()),str((R/'PADDLE-FRESH-SOURCES.sha256').resolve())))
 return found

def paddle_epoch(n,captures):
 files={};refs=set()
 for page in plan['pages'][n*5:min(n*5+5,len(plan['pages']))]:
  image=Path('/mnt/raid0/llm/datasets/omnidocbench/images')/page['image'];files[str(image)]=filehash(image)
  key=sha(canon({'campaign':sha(canon(campaign)),'page':page}));prefix='region-'+sha(canon(page))[:24]+'-'
  for root in roots:
   directory=root/f'shard-{n:02}'/f'shard-{n}'
   if not directory.exists():continue
   for path in directory.iterdir():
    if not (path.name.startswith(key+'.') or path.name.startswith(prefix)):continue
    files[str(path)]=filehash(path)
    if path.name.endswith('.start.json') or path.name.endswith('.finish.json'):
     try:
      d=json.loads(path.read_bytes());h=(d.get('provenance',{}).get('origin_capture_sha256') if path.name.endswith('.start.json') else d.get('capture_sha256'))
      if isinstance(h,str):refs.add(h)
     except (ValueError,OSError,TypeError):pass
 selected={h:sorted(captures.get(h,[])) for h in sorted(refs)}
 for rows in selected.values():
  for p,reg in rows:
   files[p]=filehash(p);files[reg]=filehash(reg)
   try:
    cap=json.loads(Path(p).read_bytes())
    if cap.get('id')=='codex3-pip02-fresh-s1-20261029':
     for name in cap.get('source_pins',{}):
      if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe captured source')
      source_path=str(R/name);files[source_path]=filehash(source_path)
   except (OSError,ValueError,TypeError):pass
   try:
    for source_path in json.loads(Path(reg).read_bytes()):
     if source_path not in files:files[source_path]=filehash(source_path)
   except (OSError,ValueError,TypeError):pass
 source=RESEARCH/'odl_bench'; sources={str(source/x):filehash(source/x) for x in ('pip02_records.py','pip02_lineage.py','pip02_compare.py','pip02_native_score.py','pip02_score_capture.py')}
 for module in source.glob('*.py'):sources[str(module)]=filehash(module)
 for extra in (Path('/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031/source-pins.json'),Path('/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031/source.sha256'),Path('/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031/launch-p0.sh'),R/'PADDLE-FRESH-SOURCES.sha256',Path('/mnt/raid0/llm/tmp/codex3-filing-pip-dtap-c6-20261031-0230/repair/head/launch-p0.sh'),Path('/mnt/raid0/llm/tmp/codex3-filing-pip-dtap-c6-20261031-0230/repair/head/pool.py')):sources[str(extra)]=filehash(extra)
 try:
  for name in json.loads(Path('/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031/source-pins.json').read_bytes()):sources[name]=filehash(name)
 except (OSError,ValueError,TypeError):pass
 sources['/mnt/raid0/llm/tmp/codex3-pip-pool2-r30-v2-20261031/launch-p2.sh']=filehash('/mnt/raid0/llm/tmp/codex3-pip-pool2-r30-v2-20261031/launch-p2.sh')
 for x in (SRC/'run_one.py',SRC/'scorer.py',SRC/'ready_shards.py',SRC/'epoch_watch.py',R/'english755-fresh-plan.json',R/'scoring-prep/pins_v1.json',Path('/mnt/raid0/llm/tmp/pip02-pipeline-prep-20261029/paddle-cpu-campaign-config-v3.json')):sources[str(x)]=filehash(x)
 return {'arm':'paddle','shard':n,'case_files':files,'capture_refs':selected,'source':sources}
def odl_epoch(n):
 files={}
 for pos in range(n*25,min(n*25+25,len(plan['pages']))):
  d=R/f'odl-batch-results/shard-{n:02}/page-{pos:04}'
  for name in ('DONE','result.json','start.json','input-identity.json','input.pdf','out/input.md'):files[str(d/name)]=filehash(d/name)
  image=Path('/mnt/raid0/llm/datasets/omnidocbench/images')/plan['pages'][pos]['image'];files[str(image)]=filehash(image)
  try:
   a=json.loads((d/'start.json').read_bytes());m=Path(a['producer_source_manifest_path']);files[str(m)]=filehash(m)
   for line in m.read_text().splitlines():
    _,name=line.split('  ',1);files[name]=filehash(name)
  except (KeyError,TypeError,OSError,ValueError):pass
 source_paths=[RESEARCH/'odl_bench/pip02_lineage.py',RESEARCH/'odl_bench/pip02_score_capture.py',S/'watch/run_one.py',S/'watch/scorer.py',R/'english755-fresh-plan.json',R/'scoring-prep/pins_v1.json',Path('/mnt/raid0/llm/worktrees/codex-pip02-common-scorer/scripts/benchmark/odl_bench/pip02_native_score.py')]
 return {'arm':'odl','shard':n,'case_files':files,'source':{str(p):filehash(p) for p in source_paths}}
def epoch_id(o):return sha(canon(o))
def legacy(arm,n):
 roots=('paddle-watch','paddle-split-v3','paddle-split') if arm=='paddle' else ('watch','recheck','.')
 name=f's{n:03}.json' if arm=='paddle' else f'odl-s{n:02}.json'
 result=[]
 for root in roots:
  p=S/root/name
  if p.exists():
   try:result.append((p,json.loads(p.read_bytes())))
   except (ValueError,OSError):pass
 if arm=='paddle':
  sys.path.insert(0,str(SRC/'vidya_snapshot'))
  from adapters.pip02_teds import native_rows
  for p in sorted(p for root in ('paddle-watch','paddle-watch-r29','paddle-watch-r30') for p in (S/root/'epochs'/f's{n:03}').glob(f'*/s{n:03}.json')):
   receipt=p.with_name(p.stem+'.score-receipt.json')
   try:
    if not receipt.is_file():continue
    native_rows(receipt);result.append((p,json.loads(p.read_bytes())))
   except (OSError,ValueError,TypeError):continue
 return result
def complete_snapshot(rows):
 return any(d.get('coverage',{}).get('pages_complete')==d.get('coverage',{}).get('pages_total') and d.get('coverage',{}).get('pages_total',0)>0 for _,d in rows)
def changed_complete(o,rows):
 # Existing full Paddle snapshots retain each matched page input inventory. Detect any
 # changed/removed original case file and newly selected origin/finish capture digest.
 if o['arm']=='odl':
  for _,d in rows:
   if d.get('coverage',{}).get('pages_complete')!=d.get('coverage',{}).get('pages_total'):continue
   for page in d.get('pages',[]):
    pos=page['position'];base=R/f"odl-batch-results/shard-{o['shard']:02}/page-{pos:04}"
    try:
     if json.loads((base/'start.json').read_bytes())!=page.get('start') or filehash(base/'out/input.md')!=page.get('prediction_sha256') or filehash(Path('/mnt/raid0/llm/datasets/omnidocbench/images')/page['image'])!=plan['pages'][pos]['sha256']:return True
     st=page.get('start',{});manifest=st.get('producer_source_manifest_path')
     if manifest:
      if filehash(manifest)!=st.get('producer_source_manifest_sha256'):return True
      for line in Path(manifest).read_text().splitlines():
       h,path=line.split('  ',1)
       if filehash(path)!=h:return True
     prediction=rec.odl(base,plan['pages'][pos],'63028e74950ba7250f33cbce13f4f0bdd165bba2a075242db4475c26f645ed68',lin.PROTOCOL,'38d2d27b681bbe398107470b84447ac45f426aa5d0fc0ecfbdcce34daa64ab9f')
     if prediction is None or sha(prediction.encode())!=page.get('prediction_sha256'):return True
    except (OSError,ValueError,TypeError,KeyError):return True
   return False
  return False
 for _,d in rows:
  if d.get('coverage',{}).get('pages_complete')!=d.get('coverage',{}).get('pages_total'):continue
  for page in d.get('pages',[]):
   for path,h in page.get('input_sha256',{}).items():
    if path in o['case_files'] and o['case_files'][path]!=h:return True
    if path not in o['case_files'] and ('/region-' in path or path.endswith(('.start','.result','.finish.json'))):return True
   pos=page['position'];row=plan['pages'][pos];key=sha(canon({'campaign':sha(canon(campaign)),'page':row}));prefix='region-'+sha(canon(row))[:24]+'-'
   actual={path for path in o['case_files'] if Path(path).name.startswith(key+'.') or Path(path).name.startswith(prefix)}
   recorded={path for path in page.get('input_sha256',{}) if Path(path).name.startswith(key+'.') or Path(path).name.startswith(prefix)}
   if actual!=recorded:return True
   lineage=page.get('source_lineage',{});old=set(lineage.get('origin_capture_sha256',[]))|{lineage.get('finish_capture_sha256')}
   if not old-{None} <= set(o['capture_refs']):return True
  return False
 return False

def need(o):
 arm=o['arm'];n=o['shard'];eid=epoch_id(o);W=S/('paddle-watch-r30' if arm=='paddle' else 'watch');state=W/'epochs'/f's{n:03}'/eid
 if state.exists():return False
 shard_dir=state.parent
 prior_epochs=shard_dir.exists() and any(p.is_dir() for p in shard_dir.iterdir())
 rows=legacy(arm,n)
 if prior_epochs and not complete_snapshot(rows):return True
 if not rows:
  if arm=='odl' and (W/f'refused-s{n:02}').exists():
   baselines=W/'epochs'/'legacy-refusal-baselines';baselines.mkdir(parents=True,exist_ok=True);baseline=baselines/f's{n:03}.json'
   if not baseline.exists():
    try:
     with baseline.open('x') as f:json.dump({'epoch':eid},f)
    except FileExistsError:pass
   return json.loads(baseline.read_text())['epoch']!=eid
  return True
 if complete_snapshot(rows):return changed_complete(o,rows)
 # Legacy partial/refusal is reconsidered only if the original recorded source differs.
 if arm=='paddle':
  current=o['source'].get(str(RESEARCH/'odl_bench/pip02_lineage.py'))
  return any(d.get('scoring_sources_sha256',{}).get('pip02_lineage.py') not in (None,current) for _,d in rows)
 return False

def ready_paddle():
 out=subprocess.check_output(['/usr/bin/python3',str(SRC/'ready_shards.py')],text=True);return sorted({int(x) for x in out.split()})
def candidates(arm):
 caps=capture_index() if arm=='paddle' else None
 nums=ready_paddle() if arm=='paddle' else [n for n in range(31) if (R/f'ODL-BATCH-DONE-{n}').is_file()]
 for n in nums:
  o=paddle_epoch(n,caps) if arm=='paddle' else odl_epoch(n)
  if need(o):print(n,epoch_id(o))
def prepare(arm,n,expected):
 o=paddle_epoch(n,capture_index()) if arm=='paddle' else odl_epoch(n)
 if epoch_id(o)!=expected or not need(o):raise ValueError('epoch changed or already handled')
 W=S/('paddle-watch-r30' if arm=='paddle' else 'watch');d=W/'epochs'/f's{n:03}'/expected;d.mkdir(parents=True,exist_ok=False)
 (d/'epoch.json').write_bytes(canon(o)+b'\n')
 print(d)
if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='list':candidates(sys.argv[2])
 elif len(sys.argv)==5 and sys.argv[1]=='prepare':prepare(sys.argv[2],int(sys.argv[3]),sys.argv[4])
 else:raise SystemExit('usage: epoch_watch.py list ARM | prepare ARM SHARD DIGEST')
