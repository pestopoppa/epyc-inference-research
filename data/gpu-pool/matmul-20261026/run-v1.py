#!/usr/bin/python3
"""Grant-only finite artifact correctness; no timing or release inference."""
import ctypes, datetime, glob, hashlib, json, os, pathlib, re, signal, subprocess, time
ROOT=pathlib.Path('/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-pool-matmul')
BUILD=pathlib.Path('/mnt/raid0/llm/tmp/build-champion-c463f601b-hip-20260909')
BIN=BUILD/'bin/test-backend-ops'
FLAGS=pathlib.Path('/mnt/raid0/llm/tmp/compute-flags')
ID='ni-gpu-pool-matmul-20261026-v1'
STARTED=time.monotonic()
ALLOWED=set(range(184,192))
TYPES='(f32|f16|q4_0|q8_0|q4_K|q6_K)'
CASES=[('MUL_MAT', '^type_a='+TYPES+',type_b=f32,m=16,n=([1-9]|16),k=256,bs=\\[(1,1|3,1|3,2|2,3)\\],nr=\\[[12],[12]\\],per=\\[(0,1,2,3|0,2,1,3|0,1,3,2|0,3,2,1)\\],k_v=0,o=1$',201),
 ('MUL_MAT_ID','^type_a='+TYPES+',type_b=f32,((n_mats=4,n_used=2,b=0,m=64,n=16,k=(3|96|768))|(n_mats=(4|8),n_used=(1|2|4),b=[01],m=512,n=(1|4|5|17|32|129),k=256))$',370)]
PIN={ 'test-backend-ops':'373b44b7476e7d34cae8e1b03e1bee40c31e515ece133667d440ce4128eb745f', 'libggml-base.so':'23735e6f21028356fa81657393f9c4ffeda70477e3377a4fb49fb2eda55b002a', 'libggml-cpu.so':'a19eb802decd2c76acd7403239ecf0ec054ae4519e2b991d6bd4cfff8edb6b5e', 'libggml-hip.so':'45bca96a11d4dbaa30470e79b24012f32504d8c0edd00710c130ad755ca0bb7c', 'libggml.so':'9c1bcacfe8fd67b0a1ac8dab56ff6c21ca6825a5771adfd8dbee1606461fa742'}
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def safe():
 assert time.monotonic()-STARTED < 1400,'finite workload cap1400s'
 assert not list(FLAGS.glob('HOLD_*')),'HOLD active'
 assert not (ROOT/'STOP').exists(),'STOP active'
 now=time.time();grant=dict(s.split('=',1) for s in (FLAGS/'grants'/f'{ID}.grant').read_text().split())
 parse=lambda s:datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
 assert parse(grant['start'])<=now<parse(grant['end']),'grant time'
 assert grant['regions']=='mi210_0,184-191','exact GPU grant shape'
 assert now < min(float(os.environ['FILLER_DEADLINE']),parse(grant['end'])-90),'deadline'
 assert os.sched_getaffinity(0)<=ALLOWED,'driver affinity'
def expand(s):
 out=set()
 for word in s.split(','):
  pair=word.split('-');out.update(range(int(pair[0]),int(pair[-1])+1))
 return out
child=None
samples=[]
def make_record(artifacts,shim_hash):
 return {'schema':'epyc.model_free_backend_correctness.v1','row_id':ID,
  'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'model_free':True,'scope':'synthetic seeded tensor correctness for this exact executable and DSOs',
  'proposition':'Selected supported MUL_MAT and MUL_MAT_ID cases agree with the native CPU reference under the existing binary NMSE tolerance 5e-4; each selected type must have an actual pass, and unsupported cases remain explicit.',
  'claim_limits':['No performance warrant','No release or current-clean-champion assertion','No new grading rule'],
  'source':{'commit':'c463f601bd39d0e313b744c214b8c22f9455bcd3','removed_worktree_dirty_state':'UNKNOWN','snapshot':str(ROOT/'test-backend-ops.c463f601b.cpp')},
  'artifacts':artifacts,'shim':{'path':'/mnt/raid0/llm/cache/shims/nprocs_shim.so','sha256':shim_hash,'SHIM_NPROCS':2,'original_source_digest_receipt':'51080dff80b0b8a72b7a41ce878f093eff0f13e739fee93d3490421c8d1041e6','later_recreated_source_matches_receipt':False},
  'seed':20261026,'workload_cap_seconds':1400,
  'evidence':{key:str(ROOT/name) for key,name in [('residency','residency.json'),('linkage','linkage.log'),('shim_probe','shim-probe.json'),('death_proof','death-proof.json')]},
  'runs':[]}
def stop_child():
 global child
 if child is None:return
 if child.poll() is None:
  child.terminate()
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
 assert child.poll() is not None,'owned child remains'
 (ROOT/'death-proof.json').write_text(json.dumps({'pid':child.pid,'returncode':child.returncode,'proc_exists':pathlib.Path(f'/proc/{child.pid}').exists()}))
def sample(pid):
 masks={}
 for p in pathlib.Path(f'/proc/{pid}/task').glob('*/status'):
  try:
   s=re.search(r'^Cpus_allowed_list:\s*(.*)$',p.read_text(),re.M).group(1)
  except FileNotFoundError:continue
  masks[p.parent.name]=s
  assert expand(s)<=ALLOWED,f'escaped task {p}: {s}'
 kfd=pathlib.Path(f'/sys/class/kfd/kfd/proc/{pid}')
 vram={}
 for p in kfd.glob('vram*'):
  try:vram[p.name]=int(p.read_text().strip())
  except FileNotFoundError:pass
 maps=pathlib.Path(f'/proc/{pid}/maps')
 own_maps=[]
 try:
  for line in maps.read_text().splitlines():
   if '/libggml' in line:
    path=line.split()[-1];assert pathlib.Path(path).resolve().is_relative_to((BUILD/'bin').resolve()),path
    own_maps.append(path)
 except FileNotFoundError:pass
 return {'time':time.time(),'pid':pid,'tasks':len(masks),'masks':masks,'kfd':kfd.exists(),'vram':vram,'ggml_maps':sorted(set(own_maps))}
def main():
 global child
 safe()
 artifacts=[]
 for name,sha in PIN.items():
  actual=digest(BUILD/'bin'/name);assert actual==sha,name+' changed'
  artifacts.append({'path':str(BUILD/'bin'/name),'sha256':actual})
 shim=pathlib.Path('/mnt/raid0/llm/cache/shims/nprocs_shim.so')
 assert digest(shim)=='d3103d7ab45e0a5fe9473d7dd401e5462f39c06174b4a64109af6e387e3b3eb4'
 receipt=make_record(artifacts,digest(shim))
 (ROOT/'results.json').write_text(json.dumps(receipt,indent=2))
 assert os.environ['SHIM_NPROCS']=='2'
 lib=ctypes.CDLL('libstdc++.so.6');fn=getattr(lib,'_ZNSt6thread20hardware_concurrencyEv');fn.restype=ctypes.c_uint
 cap=fn();(ROOT/'shim-probe.json').write_text(json.dumps({'hardware_concurrency':cap,'shim':str(shim),'source_state':'UNKNOWN; recreated source differs'}));assert cap==2,'shim cannot cap hardware_concurrency; build prerequisite'
 subprocess.run(['/bin/bash','/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh',str(BIN),str(BUILD)],check=True,stdout=open(ROOT/'linkage.log','w'),stderr=subprocess.STDOUT)
 # Capture strings before matching: never grep -q a live pipe under pipefail.
 knob=subprocess.run(['/usr/bin/strings',str(BUILD/'bin/libggml-hip.so')],check=True,capture_output=True,text=True).stdout
 assert 'GGML_CUDA_DISABLE_GRAPHS' in knob,'graphs-off knob absent'
 (ROOT/'compiled-knob.txt').write_text('GGML_CUDA_DISABLE_GRAPHS present; set=1\n')
 for op,params,expected in CASES:
  safe();argv=[str(BIN),'test','-b','ROCm0','-o',op,'-p',params,'-j','1','--suite-seed','20261026']
  log=ROOT/(op+'.log');start=time.time();local=[]
  result={'op':op,'argv':argv,'seed':20261026,'log':str(log),'generated_expected':expected,'status':'native_launch_pending'}
  receipt['runs'].append(result)
  (ROOT/'results.json').write_text(json.dumps(receipt,indent=2))
  with log.open('w') as f:
   f.write('ARGV '+json.dumps(argv)+'\n');f.flush()
   child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT)
   (ROOT/'owned-pids.txt').open('a').write(str(child.pid)+'\n')
   while child.poll() is None:
    safe();record=sample(child.pid);local.append(record);samples.append(record)
    if sum(record['vram'].values())>60*1024**3:raise RuntimeError('own-PID VRAM guard60GiB')
    time.sleep(.02)
   rc=child.wait();stop_child()
  text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text())
  lines=[line for line in text.splitlines() if line.startswith('  '+op+'(')]
  passed=sum(line.endswith('OK') for line in lines)
  unsupported=sum('not supported' in line for line in lines)
  failed=[line for line in lines if not line.endswith('OK') and 'not supported' not in line]
  resident=[s for s in local if s['kfd'] and sum(s['vram'].values())>0 and any('libggml-hip' in p for p in s['ggml_maps'])]
  result.update({'status':'native_exit_recorded','rc':rc,'observed':len(lines),'passed':passed,'unsupported':unsupported,'failed':failed,'resident_samples':len(resident),'seconds':time.time()-start,'max_tasks':max((s['tasks'] for s in local),default=0)})
  (ROOT/'results.json').write_text(json.dumps(receipt,indent=2))
  assert rc==0 and len(lines)==expected and passed>0 and not failed,result
  assert re.search(r'Backend [0-9]+/[0-9]+: ROCm0\n  Device description:',text) and len(resident)>=2,'native HIP/owned-PID DURING residency not proven'
  # Every selected type must actually pass; unsupported receipt alone cannot close coverage.
  for ty in ['f32','f16','q4_0','q8_0','q4_K','q6_K']:
   assert any(('type_a='+ty+',') in line and line.endswith('OK') for line in lines),'no actual pass '+ty
 safe();(ROOT/'SUCCESS').write_text('finite artifact-specific correctness completed; no performance/release claim\n')
if __name__=='__main__':
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda a,b:(_ for _ in ()).throw(InterruptedError('signalled')))
 try:main()
 except BaseException as exc:
  (ROOT/'FAILED').write_text(repr(exc)+'\n');raise
 finally:
  stop_child();(ROOT/'residency.json').write_text(json.dumps(samples));(ROOT/'JOB_ENDED').touch()
