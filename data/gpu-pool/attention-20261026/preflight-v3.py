#!/usr/bin/python3
"""Owner-granted two-CPU dry run; never invokes native test-backend-ops."""
import datetime, hashlib, json, os, pathlib, re, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parent
family=ROOT.name
row_id=f'ni-gpu-sampler-{family}-20261026-v3'
pf_id=f'ni-gpu-sampler-{family}-pf-20261026-v3'
cpus=set(map(int,os.environ['PREFLIGHT_CPUS'].split(',')))
assert cpus=={26,27} and os.sched_getaffinity(0)==cpus,'owner authority is exactly CPUs26,27'
FLAGS=pathlib.Path('/mnt/raid0/llm/tmp/compute-flags')
owner_receipt=ROOT/'OWNER-REPLY-1038.raw.json'
assert hashlib.sha256(owner_receipt.read_bytes()).hexdigest()=='bd92629e3a0f8eebb5bb3b05bf4c39788cc95c1ed14dac6038ff6705dd40e5d3','owner receipt pin'
owner=json.loads(owner_receipt.read_text())
entry=next(item for item in owner['updates'] if item.get('ts')=='2026-10-26T10:38Z')
assert entry['from']=='workspace-58' and 'PREFLIGHT/DRYRUN ADMISSION (owner-permitted, no row needed, so not circular)' in entry['text']
assert 'taskset -c 26,27' in entry['text'] and 'timeout 300, one at a time, no model/server/GPU device open' in entry['text']
checks=[]
def expand(region):
 result=set()
 for part in region.split(','):
  if part=='mi210_0':continue
  assert re.fullmatch(r'\d+(?:-\d+)?',part),'unrecognized grant region; cannot prove q1 disjoint'
  nums=part.split('-');result.update(range(int(nums[0]),int(nums[-1])+1))
 return result
def authority():
 assert os.sched_getaffinity(0)=={26,27},'lost owner preflight affinity'
 holds=list(FLAGS.glob('HOLD_*'))
 assert not holds and not (ROOT/'STOP').exists(),'real abort active'
 grants={}
 for path in sorted((FLAGS/'grants').glob('*.grant')):
  raw=path.read_text();grants[str(path)]=raw
  grant=dict(item.split('=',1) for item in raw.split())
  assert not (expand(grant['regions']) & set(range(24,48))),'q1 or whole-host grant now open: '+str(path)
 record={'authority':'owner-special-PREFLIGHT-only','owner_entry':entry,'owner_receipt_sha256':'bd92629e3a0f8eebb5bb3b05bf4c39788cc95c1ed14dac6038ff6705dd40e5d3','cpus':[26,27],'pid':os.getpid(),'time':time.time(),'grants':grants,'holds':[]}
 checks.append(record)
 (ROOT/'PREFLIGHT-AUTHORITY.json').write_text(json.dumps(checks,indent=2))
 return record
grant=authority()
def run(argv,limit=30):
 authority();subprocess.run(argv,check=True,timeout=limit)
run(['/usr/bin/sha256sum','-c','SEAL.sha256'])
namespace={'__name__':'dryrun_source'}
source=ROOT/'run-v3.py'
exec(compile(source.read_text(),str(source),'exec'),namespace)
assert namespace['ID']==row_id and namespace['ROOT']==ROOT
assert namespace['ALLOWED']==set(range(184,192)) and namespace['expand']('184-191')==set(range(184,192))
row=next(line for line in (ROOT/"ROWS-PROPOSED-NOT-READY.md").read_text().splitlines() if line.startswith("| "+row_id+" |"))
cells=[cell.strip() for cell in row.split("|")][1:-1]
assert len(cells)==18 and cells[3]=="mi210_0,184-191" and cells[16]=="gpu"
estimated_host_ram_gib=8
cap=int(re.search(r"time.monotonic\(\)-STARTED\s*<\s*(\d+)",source.read_text()).group(1))
assert estimated_host_ram_gib<=float(cells[5]) and int(cells[9])>=cap+300 and int(cells[10])==1500
for script in ['launch-v3.sh','job-v3.sh','watch-v3.sh','dryrun-v3.sh']:
 run(['/bin/bash','-n',str(ROOT/script)],10)
compile((ROOT/'fixtures-v3.py').read_text(),str(ROOT/'fixtures-v3.py'),'exec')
for name,expected in namespace['PIN'].items():
 path=namespace['BUILD']/'bin'/name
 assert path.is_file() and namespace['digest'](path)==expected
 assert path.stat().st_mtime>1788936811,'native artifact predates native source commit'
shim=pathlib.Path('/mnt/raid0/llm/cache/shims/nprocs_shim.so')
assert namespace['digest'](shim)=='d3103d7ab45e0a5fe9473d7dd401e5462f39c06174b4a64109af6e387e3b3eb4'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS']:assert os.environ[key]=='1'
assert os.environ['SHIM_NPROCS']=='2' and os.environ['LD_PRELOAD']==str(shim)
assert os.environ['LD_LIBRARY_PATH']==str(namespace['BUILD']/'bin')+':/opt/rocm/lib'
assert os.environ['GGML_CUDA_DISABLE_GRAPHS']=='1' and os.environ['HSA_OVERRIDE_CPU_AFFINITY_DEBUG']=='0'
assert os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']=='0'
for key in ['AUTOKERNEL_CORRECTNESS_CASE_SET','GGML_BACKEND_PATH','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS']:assert key not in os.environ
knobs=subprocess.run(['/usr/bin/strings',str(namespace['BUILD']/'bin/libggml-hip.so')],check=True,capture_output=True,text=True,timeout=30).stdout
assert 'GGML_CUDA_DISABLE_GRAPHS' in knobs
cli=subprocess.run(['/usr/bin/strings',str(namespace['BIN'])],check=True,capture_output=True,text=True,timeout=30).stdout
assert '--suite-seed' in cli and '-p <params regex>' in cli
snapshot=pathlib.Path('/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-pool-elementwise/audit/test-backend-ops-c463f601b.cpp')
assert namespace['digest'](snapshot)=='9a017871d6354015151630af6d75fa88d77637d599a40421a38746e0f67e7239'
assert b'ProcessLookupError' in source.read_bytes() and source.stat().st_mtime>0
print('Native source c463f601b/artifacts pinned; Python sampler repair is interpreted source sealed now; native binary need not postdate Python repair.')
linkage=pathlib.Path('/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh')
assert linkage.is_file()
run(['/bin/bash',str(linkage),str(namespace['BIN']),str(namespace['BUILD'])])
for name in ['SUCCESS','FAILED','JOB_ENDED','STOP','residency.json','results.json','linkage.log','shim-probe.json','death-proof.json','PREFLIGHT-AUTHORITY.json','DRYRUN.log']:
 assert os.access(ROOT/name,os.W_OK) if (ROOT/name).exists() else os.access(ROOT,os.W_OK),name
for directory in [namespace["BUILD"]/"bin",pathlib.Path("/opt/rocm/lib"),FLAGS]:assert directory.is_dir(),str(directory)
state=ROOT/"state";state.mkdir(exist_ok=True);assert os.access(state,os.W_OK),"owned state directory unwritable"
probe=ROOT/'DRYRUN-WRITE-PROBE'
probe.write_text('owned output write probe\n');probe.unlink()
run(['/usr/bin/python3',str(ROOT/'fixtures-v3.py'),str(source),'--isolated'])
authority()
print('DRYRUN PASS: pinned owner10:38Z special preflight authority, one thread/two CPUs, sealed paths/pins/source/knobs/linkage/env,184-191 mask,8GiB host request,1800>=1400+300,ports=none,owned flag paths and fake abort guards+25 sampler controls. No native GPU benchmark/model/server executed.')
