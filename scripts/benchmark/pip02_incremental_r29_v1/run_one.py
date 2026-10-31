import hashlib,json,os,sys,subprocess,tempfile
from pathlib import Path
SRC=Path(__file__).resolve().parent
W=Path(os.environ['PIP_SCORE_OUTPUT']).resolve(strict=True)
if not W.is_dir() or W==SRC or SRC in W.parents:raise ValueError('output must be an existing directory outside source')
n=int(sys.argv[1]);dst=W/f's{n:03}.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(SRC))
from odl_bench.pip02_score_capture import begin, finish

if dst.exists():raise ValueError('refuse overwrite')
source=SRC/'odl_bench';names=['pip02_records.py','pip02_lineage.py','pip02_compare.py','pip02_native_score.py'];before={n:sha(source/n) for n in names};before['watch_scorer']=sha(SRC/'scorer.py');before['watch_runner']=sha(__file__)
package_sources=sorted(source.glob('*.py'));package_before={str(p):sha(p) for p in package_sources}
pinpath=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029/scoring-prep/pins_v1.json')
pin=json.loads(pinpath.read_bytes())
fixed=[pinpath,Path(pin['plan']),Path(pin['gt']),Path('/mnt/raid0/llm/tmp/pip02-pipeline-prep-20261029/paddle-cpu-campaign-config-v3.json'),SRC/'scorer.py',Path(__file__),source/'pip02_score_capture.py']+[source/n for n in names]
fixed += [Path(p.replace('/home/node/','/home/daniele/')) for p in pin['files']]
fixed += package_sources
start_path=W/f's{n:03}.score-start.json'; receipt_path=W/f's{n:03}.score-receipt.json'
begin(start_path,'paddle',n,fixed,sys.argv)
with tempfile.TemporaryDirectory(dir=W) as tmp:
 env=os.environ.copy();env['PIP_PADDLE_STAGE']=tmp
 child=subprocess.run([sys.executable,'-S',str(SRC/'scorer.py'),str(n)],env=env)
 if child.returncode:raise SystemExit(child.returncode)
 for name,digest in before.items():
  p=(SRC/'scorer.py' if name=='watch_scorer' else Path(__file__) if name=='watch_runner' else source/name)
  if sha(p)!=digest:raise ValueError('scorer source changed during execution')
 if any(sha(name)!=digest for name,digest in package_before.items()):raise ValueError('package source changed during execution')
 pinpath=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029/scoring-prep/pins_v1.json')
 if sha(pinpath)!='262462c58402db48aa88063cdb1817915dcdef0c1933e72869682dd905c34cf0':raise ValueError('score pin changed')
 pin=json.loads(pinpath.read_bytes())
 for path,digest in pin['files'].items():
  if sha(path.replace('/home/node/','/home/daniele/'))!=digest:raise ValueError('score input changed: '+path)
 if sha(pin['plan'])!='63028e74950ba7250f33cbce13f4f0bdd165bba2a075242db4475c26f645ed68' or sha(pin['gt'])!='a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496':raise ValueError('plan or GT changed')
 result=json.loads((Path(tmp)/f's{n:03}.json').read_bytes());result['watch_source_sha256']=before
 staged=Path(tmp)/'final.json';staged.write_text(json.dumps(result,indent=2)+'\n')
 with staged.open('rb') as f:os.fsync(f.fileno())
 os.link(staged,dst)
 finish(start_path,dst,receipt_path)
