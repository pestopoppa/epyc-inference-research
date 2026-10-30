import argparse,json,sys
from pathlib import Path
from odl_bench import pip02_compare as m
from odl_bench.pip02_native_score import evaluate
from odl_bench.comparison import build_existing_comparison
p={'page_info':{'image_path':'x'},'extra':{'relation':[]},'layout_dets':[{'category_type':'table','order':1,'anno_id':'x','html':'<table><tr><td>x</td></tr></table>'}]}
evaluate(p,'<table><tr><td>x</td></tr></table>');evaluate(p,'| x | y |\n| --- | --- |\n| a | b |')
root=Path('/mnt/raid0/llm/omnidocbench'); research=Path(__file__).resolve().parent;gt='/mnt/raid0/llm/datasets/omnidocbench/OmniDocBench.json'
paths=set(m.loaded_files())|{str(x.resolve()) for x in (root/'src').rglob('*.py')}|{str(x.resolve()) for x in research.glob('*.py')}
for list_name in ('ODL-FINAL-SOURCES.sha256','ODL-QUAD-SOURCES.sha256','PADDLE-SAMEQUARTER-SOURCES.sha256','packets-quad-review/paddle-s2/PADDLE-QUAD-SOURCES.sha256','packets-quad-review/paddle-s3/PADDLE-QUAD-SOURCES.sha256'):
 for line in (Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029')/list_name).read_text().splitlines():
  expected, source=line.split(maxsplit=1); path=Path(source.strip())
  if not path.is_file() or m.digest(path)!=expected: raise ValueError('producer source mismatch: '+str(path))
  paths.add(str(path.resolve()))
for name, expected in {'odl_quad_job.py':'051c5bfa92dd77053b287241c9598e3ffb3ae10e4cb181ba36b9062101a33c39','odl_quad_batchpage.py':'e4d2e44476b7bda7359de3f383eac309466752caa705e6b3fcb8142c325592e6'}.items():
 path=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029')/name
 if m.digest(path)!=expected: raise ValueError('quad source mismatch: '+str(path))
 paths.add(str(path.resolve()))
# Capture-time source registries are explicitly allowlisted, never inferred from mtime.
from odl_bench import pip02_lineage as lineage
registries=[(lineage.ODL,lineage.ODL_SHA),(lineage.ODL_R3,lineage.ODL_R3_SHA),(lineage.RECOVERY/'SOURCES.sha256',lineage.RECOVERY_SHA)]+[(root/'source-pins.json',expected) for root,expected in lineage.PADDLE_ROOTS.values()]
for registry,expected in registries:
 if m.digest(registry)!=expected: raise ValueError('registered producer manifest changed: '+str(registry))
 paths.add(str(registry.resolve()))
 if registry.suffix=='.json': entries=json.loads(registry.read_bytes()).items()
 else: entries=((name,digest) for digest,name in (line.split('  ',1) for line in registry.read_text().splitlines()))
 for name,digest in entries:
  p=Path(name)
  if not p.is_file() or m.digest(p)!=digest: raise ValueError('registered producer source mismatch: '+str(p))
  paths.add(str(p.resolve()))
# Original pre-client resolution evidence is part of prospective scoring pins.
for rr in json.loads((lineage.RECOVERY/'RESOLUTIONS.json').read_bytes()).values():
 for field in ('producer_source','original_launcher_log'):
  p=Path(rr[field])
  if m.digest(p)!=rr[field+'_sha256']: raise ValueError('recovery eligibility source changed: '+str(p))
  paths.add(str(p.resolve()))
 for name,expected in rr['evidence'].items():
  p=Path(rr['original_path'])/name
  if m.digest(p)!=expected: raise ValueError('original pre-client input changed: '+str(p))
  paths.add(str(p.resolve()))
for i in range(5):
 ident='codex3-pip02-fresh-odl-s0-20261029' if i==0 else f'codex3-pip02-fresh-odl-quad-s{i}-20261029'
 packet=Path('/mnt/raid0/llm/tmp/compute-flags/ready/armed')/(ident+'.json')
 if not packet.exists(): continue
 data=json.loads(packet.read_bytes()); paths.add(str(packet.resolve()))
 for field in ('launcher','job','source_manifest'):
  v=data[field]; p=Path(v['path']);
  if m.digest(p)!=v['sha256']: raise ValueError('packet source mismatch: '+str(p))
  paths.add(str(p.resolve()))
manifest={'python':sys.version,'gt_sha256':m.digest(gt),'files':{p:m.digest(p) for p in sorted(paths)}}
parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); args=parser.parse_args()
with Path(args.out).open('x',encoding='utf-8') as fh: fh.write(json.dumps(manifest,indent=2)+'\n')
print('files',len(paths))
