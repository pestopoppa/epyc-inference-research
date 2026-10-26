#!/usr/bin/python3
"""Grant-only finite artifact correctness; no timing or release inference."""
import ctypes, datetime, glob, hashlib, json, os, pathlib, re, signal, subprocess, time
ROOT=pathlib.Path('/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/elementwise')
BUILD=pathlib.Path('/mnt/raid0/llm/tmp/build-champion-c463f601b-hip-20260909')
BIN=BUILD/'bin/test-backend-ops'
FLAGS=pathlib.Path('/mnt/raid0/llm/tmp/compute-flags')
ID='ni-gpu-sampler-elementwise-20261026-v3'
STARTED=time.monotonic()
ALLOWED=set(range(184,192))
BCAST=r'^type=f32,ne=\[10,5,4,3\],nr=\[[12],[12],[12],[12]\],nf=1,perm1=[01],src_overlap=0$'
CASES=[('ADD',BCAST,16),('MUL',BCAST,16),
 ('SCALE',r'^type=f32,ne=\[(10|100),10,10,10\],',4),
 ('NORM',r'^type=f32,ne=\[(64|1025),5,4,3\],',30),
 ('RMS_NORM',r'^type=f32,ne=\[(64|1025),5,4,3\],',21),
 ('SUM',r'^type=f32,ne=\[(10,5,4,3|11,5,6,3|33,(1|256|1024),1,1)\]',8),
 ('SUM_ROWS',r'^type=f32,ne=\[(10,5,4,3|11,5,6,3|16,5,6,3|33,(1|256|1024),1,1)\]',10)]
SEEDS=list(range(2026102600,2026102632))
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
  'proposition':'Selected supported ADD/MUL/SCALE/NORM/RMS_NORM/SUM/SUM_ROWS cases agree with the native CPU reference under their existing source-native tolerances; every selected operator must actually pass supported cases. Unsupported cases remain explicit.',
  'claim_limits':['No performance warrant','No release or current-clean-champion assertion','No new grading rule'],
  'source':{'commit':'c463f601bd39d0e313b744c214b8c22f9455bcd3','removed_worktree_dirty_state':'UNKNOWN','snapshot':'/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-pool-elementwise/test-backend-ops.c463f601b.cpp'},
  'artifacts':artifacts,'shim':{'path':'/mnt/raid0/llm/cache/shims/nprocs_shim.so','sha256':shim_hash,'SHIM_NPROCS':2,'original_source_digest_receipt':'51080dff80b0b8a72b7a41ce878f093eff0f13e739fee93d3490421c8d1041e6','later_recreated_source_matches_receipt':False},
  'seeds':SEEDS,'seed_rationale':'32 distinct deterministic tensor initializations exercise value-dependent numerical behavior; no duration padding','workload_cap_seconds':1400,
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
 identity=pathlib.Path(f'/proc/{pid}/stat');missing=[]
 try:starttime=identity.read_text().rsplit(')',1)[1].split()[19]
 except (FileNotFoundError,ProcessLookupError) as exc:return {'time':time.time(),'pid':pid,'starttime':None,'start_ticks':None,'tasks':0,'masks':{},'skipped_tasks':[],'missing_observations':[{'path':str(identity),'errno':exc.errno}],'partial':True,'kfd':False,'kfd_detail':{},'vram':{},'ggml_maps':[],'overlap':False}
 masks={};skipped=[]
 for p in pathlib.Path(f'/proc/{pid}/task').glob('*/status'):
  try:
   s=re.search(r'^Cpus_allowed_list:\s*(.*)$',p.read_text(),re.M).group(1)
  except (FileNotFoundError,ProcessLookupError) as exc:
   skipped.append({'tid':p.parent.name,'errno':exc.errno,'reason':'task vanished during status read'});continue
  masks[p.parent.name]=s
  assert expand(s)<=ALLOWED,f'escaped task {p}: {s}'
 kfd=pathlib.Path(f'/sys/class/kfd/kfd/proc/{pid}')
 vram={}
 for p in kfd.glob('vram*'):
  try:vram[p.name]=int(p.read_text().strip())
  except FileNotFoundError:pass
 maps=pathlib.Path(f'/proc/{pid}/maps');own_maps=[]
 try:
  for line in maps.read_text().splitlines():
   if '/libggml' in line:
    path=line.split()[-1];assert pathlib.Path(path).resolve().is_relative_to((BUILD/'bin').resolve()),path
    own_maps.append(path)
 except (FileNotFoundError,ProcessLookupError) as exc:missing.append({'path':str(maps),'errno':exc.errno})
 kfd_detail={}
 for name in ['pasid','queues']:
  try:kfd_detail[name]=(kfd/name).read_text()
  except (FileNotFoundError,IsADirectoryError):pass
 kfd_present=kfd.exists()
 try:overlap=identity.read_text().rsplit(')',1)[1].split()[19]==starttime and child.poll() is None
 except (FileNotFoundError,ProcessLookupError) as exc:overlap=False;missing.append({'path':str(identity),'errno':exc.errno})
 return {'time':time.time(),'pid':pid,'starttime':starttime,'start_ticks':starttime,'overlap':overlap,'tasks':len(masks),'masks':masks,'skipped_tasks':skipped,'missing_observations':missing,'partial':bool(skipped or missing) or not overlap or not masks,'kfd':kfd_present,'kfd_detail':kfd_detail,'vram':vram,'ggml_maps':sorted(set(own_maps))}
def main():
 global child
 safe()
 artifacts=[]
 for name,sha in PIN.items():
  actual=digest(BUILD/'bin'/name);assert actual==sha,name+' changed'
  assert (BUILD/'bin'/name).stat().st_mtime > 1788936811,'artifact predates pinned source commit'
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
 cli_strings=subprocess.run(['/usr/bin/strings',str(BIN)],check=True,capture_output=True,text=True).stdout
 assert '--suite-seed' in cli_strings and '-p <params regex>' in cli_strings,'source-selected CLI not compiled'
 (ROOT/'compiled-knob.txt').write_text('GGML_CUDA_DISABLE_GRAPHS present; set=1\n')
 for seed,op,params,expected in [(seed,*case) for seed in SEEDS for case in CASES]:
  safe();argv=[str(BIN),'test','-b','ROCm0','-o',op,'-p',params,'-j','1','--suite-seed',str(seed)]
  log=ROOT/(op+'-'+str(seed)+'.log');start=time.time();local=[]
  result={'op':op,'argv':argv,'seed':seed,'log':str(log),'generated_expected':expected,'status':'native_launch_pending'}
  receipt['runs'].append(result)
  (ROOT/'results.json').write_text(json.dumps(receipt,indent=2))
  with log.open('w') as f:
   f.write('ARGV '+json.dumps(argv)+'\n');f.flush()
   child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT)
   (ROOT/'owned-pids.txt').open('a').write(str(child.pid)+'\n')
   while child.poll() is None:
    safe();record=sample(child.pid);local.append(record);samples.append(record)
    if sum(record['vram'].values())>8*1024**3:raise RuntimeError('own-PID VRAM guard8GiB')
    time.sleep(.02)
   rc=child.wait();stop_child()
  text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text())
  lines=[line for line in text.splitlines() if line.startswith('  '+op+'(')]
  passed=sum(line.endswith('OK') for line in lines)
  unsupported=sum('not supported' in line for line in lines)
  failed=[line for line in lines if not line.endswith('OK') and 'not supported' not in line]
  resident=[s for s in local if s['overlap'] and not s['partial'] and s['tasks']>0 and s['kfd'] and sum(s['vram'].values())>0 and all(any(lib in p for p in s['ggml_maps']) for lib in ['libggml-hip','libggml-base','libggml-cpu'])]
  result.update({'status':'native_exit_recorded','rc':rc,'observed':len(lines),'passed':passed,'unsupported':unsupported,'failed':failed,'resident_samples':len(resident),'seconds':time.time()-start,'max_tasks':max((s['tasks'] for s in local),default=0)})
  (ROOT/'results.json').write_text(json.dumps(receipt,indent=2))
  assert rc==0 and len(lines)==expected and passed>0 and not failed,result
  assert re.search(r'Backend [0-9]+/[0-9]+: ROCm0\n  Device description:',text) and len(resident)>=2,'native HIP/owned-PID DURING residency not proven'
  assert all('type=f32,' in line for line in lines),'scope escaped f32'
 safe();receipt['status']='PASS';(ROOT/'results.json').write_text(json.dumps(receipt,indent=2));(ROOT/'SUCCESS').write_text('finite artifact-specific correctness completed; no performance/release claim\n')
if __name__=='__main__':
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda a,b:(_ for _ in ()).throw(InterruptedError('signalled')))
 try:main()
 except BaseException as exc:
  (ROOT/'FAILED').write_text(repr(exc)+'\n');raise
 finally:
  stop_child();(ROOT/'residency.json').write_text(json.dumps(samples));(ROOT/'JOB_ENDED').touch()
