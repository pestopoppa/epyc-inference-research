#!/usr/bin/env python3
"""Provider-qualified, lease-bound standalone HIP verifier/observation launcher.

This launcher never enables a provider, grants a lease, reloads a serving process,
or steals a claim. The owning session supplies an already ACTIVE GPU lease.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import time

HERE=Path(__file__).resolve().parent
class Refusal(RuntimeError): pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def preflight(root: Path, lease_id: str, holder: str) -> dict:
    import yaml
    config=yaml.safe_load((root/'coordination/session-bus/config.yaml').read_text())
    gpu=(config.get('resource_claims') or {}).get('gpu') or {}
    if not gpu.get('enabled') or not gpu.get('provider'):
        raise Refusal('resource_claims.gpu is disabled or provider=null; physical flock alone is insufficient authority')
    # Provider selection is owned by root governance. Do not guess compatibility.
    if gpu['provider'] not in {'autokernel-device-claim','gpu-device-claim'}:
        raise Refusal(f'provider {gpu["provider"]!r} has no qualified adapter in this launcher')
    sys.path.insert(0,str(root/'scripts/coordination'))
    import session_bus
    lease=session_bus.fold_resource_leases(root/'coordination/session-bus').get(lease_id,{})
    if lease.get('state')!='ACTIVE' or lease.get('holder')!=holder:
        raise Refusal('GPU resource lease must be ACTIVE and owned by the caller')
    if 'mi210_0' not in (lease.get('resources') or {}).get('gpu_devices',[]):
        raise Refusal('lease does not cover mi210_0')
    expires=lease.get('expires_ts')
    if not expires or datetime.fromisoformat(expires.replace('Z','+00:00'))<=datetime.now(timezone.utc):
        raise Refusal('lease expired or has no bounded expiry')
    return {'provider':gpu,'lease':lease}

def residency_sample(pid: int) -> dict:
    proc=Path('/sys/class/kfd/kfd/proc')
    pids=sorted(p.name for p in proc.iterdir()) if proc.is_dir() else None
    vram={str(p):int(p.read_text()) for p in Path('/sys/class/drm').glob('card*/device/mem_info_vram_used')}
    return {'time':datetime.now(timezone.utc).isoformat(),'pid':pid,
            'pid_alive':Path(f'/proc/{pid}').exists(),'kfd_pids':pids,'vram_used_bytes':vram}

def seal_verifier(e,out,binary,build,receipt,rows,returncode,reads,fixture):
    def identity(name,content):return {'id':name,'sha256':hashlib.sha256(content).hexdigest()}
    proposition='The claimed standalone gfx90a harness passed its declared operator assertions'
    fields=dict(schema=e.VERIFIER,run_id=out.name,date=datetime.now(timezone.utc).isoformat(),category='CANDIDATE',
        protocol_id='',protocol_eligible=False,arm='standalone_gfx90a',comparator='portable_fp32_oracle',
        backend='gfx90a_rocm62',fixture=fixture,path='standalone_operators',decided_proposition=proposition,
        claim=proposition,verdict='pass' if returncode==0 else 'fail',
        checker={'id':'exl3_gfx90a_harness/v1','path':str(binary),'sha256':sha(binary)},
        fixture_sha256=reads[0]['sha256'],read_set=reads,read_set_sha256=e.digest(reads),
        identities={
            'model':identity('standalone_no_model_inference',fixture.encode()),
            'artifact':identity(fixture,e.canonical(reads)),
            'source':identity('exl3_gfx90a',e.canonical(build['source_sha256'])),
            'binary':{'id':str(binary),'sha256':sha(binary)},
            'library':identity('HIP_and_host_dependencies',(out/'libraries.txt').read_bytes()),
            'toolchain':identity('rocm62',build['compiler'].encode()),
            'hardware':identity('gfx90a',e.canonical(rows[:1])),
            'residency':identity('in_window_kfd_vram',(out/'residency.json').read_bytes())})
    row=e.write(out/'native',fields);e.project(row)
    return row

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace'))
    p.add_argument('--contract-root',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lease-id',required=True);p.add_argument('--holder',required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--microbench',action='store_true')
    p.add_argument('--fixture',type=Path,help='Canonical manifest.json; transport derived and verified before GPU allocation')
    a=p.parse_args()
    if a.fixture and a.microbench:raise Refusal('fixture correctness and microbench modes are separate')
    authority=preflight(a.root,a.lease_id,a.holder)
    if a.preflight_only:print(json.dumps(authority));return
    sys.path.insert(0,str(a.contract_root.resolve()))
    from scripts.kernel_rnd.exl3 import evidence as e
    from scripts.kernel_rnd.autokernel.resource.device_claim import gpu_device_claim,ClaimJournal
    build=json.loads((a.build/'build.json').read_text());binary=(a.build/'test_runtime').resolve()
    if sha(binary)!=build['artifact_sha256']['test_runtime']:raise Refusal('binary digest differs from recorded build')
    for name,digest in build['source_sha256'].items():
        if sha(HERE/name)!=digest:raise Refusal(f'source drift since build: {name}')
    out=a.output.resolve()
    if not str(out).startswith('/mnt/raid0/'):raise Refusal('run artifacts must live under /mnt/raid0')
    out.mkdir(parents=True,exist_ok=False)
    (out/'authority.json').write_text(json.dumps(authority,sort_keys=True))
    (out/'build.json').write_text(json.dumps(build,sort_keys=True))
    snapshot=out/'snapshot';snapshot.mkdir()
    shutil.copy2(binary,snapshot/'test_runtime');binary=snapshot/'test_runtime'
    for name in build['source_sha256']:shutil.copy2(HERE/name,snapshot/name)
    command=[str(binary)];read_paths=[snapshot/'test_runtime.hip',snapshot/'kernels.hip',snapshot/'contract.hpp',out/'build.json']
    fixture='synthetic_k1_k8_mixed_routing'
    if a.fixture:
        from fixtures import export
        transport=out/'fixture.bin';record=export(a.fixture,transport)
        fixture=record['canonical_artifact_sha256']
        # Reopen all transport and canonical bytes at the actual launch boundary.
        for entry in [record['manifest'],record['fixture'],record['transport'],*record['tensor_reads'],*record['provenance_reads']]:
            if sha(entry['path'])!=entry['sha256']:raise Refusal('fixture bytes drifted before launch')
            read_paths.append(Path(entry['path']))
        command+=['--fixture',str(transport),fixture,str(out/'operator-output')]
    elif a.microbench:command+=['--microbench']
    # Read dependencies under the same deliberately isolated runtime search path.
    env=dict(os.environ,LD_LIBRARY_PATH='/opt/rocm/lib:/opt/rocm/lib64',HIP_VISIBLE_DEVICES='0',ROCR_VISIBLE_DEVICES='0')
    (out/'libraries.txt').write_bytes(subprocess.check_output(['ldd',str(binary)],env=env))
    samples=[]
    with gpu_device_claim('mi210_0',purpose='EXL3 standalone operator validation',campaign_id=a.lease_id,
                          holder_label=a.holder,journal=ClaimJournal(out/'claim-journal.jsonl'),timeout_s=0) as claim:
        receipt=asdict(claim.receipt());(out/'physical-claim.json').write_text(json.dumps(receipt,sort_keys=True))
        if not claim.held:raise Refusal('physical claim did not remain held')
        preflight(a.root,a.lease_id,a.holder)
        # Conservative co-residency: no foreign KFD process may share a verifier.
        before=residency_sample(os.getpid())
        if before['kfd_pids'] is None:raise Refusal('KFD process census unavailable')
        if before['kfd_pids']:raise Refusal('foreign KFD process present; no measured co-residency policy for this operator')
        env['EXL3_PHYSICAL_CLAIM_FD']=str(claim._fd)
        with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
            proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,pass_fds=(claim._fd,))
            started=datetime.now(timezone.utc).isoformat()
            while proc.poll() is None:
                samples.append(residency_sample(proc.pid));time.sleep(0.02)
            code=proc.wait()
            ended=datetime.now(timezone.utc).isoformat()
        if claim.revocation():claim.acknowledge_revocation()
    teardown=[]
    for _ in range(2):
        teardown.append(residency_sample(proc.pid));time.sleep(0.02)
    if any(str(proc.pid) in (sample['kfd_pids'] or []) for sample in teardown):code=code or 2
    residency={'started':started,'ended':ended,'samples':samples,'teardown_samples':teardown,'physical_claim':receipt}
    (out/'residency.json').write_text(json.dumps(residency,sort_keys=True))
    rows=[json.loads(line) for line in (out/'stdout.jsonl').read_text().splitlines() if line.strip().startswith('{')]
    in_window=any(str(proc.pid) in (s['kfd_pids'] or []) and any(v>0 for v in s['vram_used_bytes'].values()) for s in samples)
    if not in_window:code=code or 2
    read_paths += [out/'stdout.jsonl',out/'stderr.txt',out/'residency.json',out/'physical-claim.json']
    reads=[{'path':str(path.resolve()),'sha256':sha(path)} for path in dict.fromkeys(read_paths)]
    row=seal_verifier(e,out,binary,build,receipt,rows,code,reads,fixture)
    measurement_ids=[]
    if a.microbench and code==0:
        for record in rows:
            if record.get('proposition')!='routing_microbenchmark':continue
            arm='/'.join([record['policy'],record['routing'],record['compute'],record['reduction']])
            fields={key:row[key] for key in ('run_id','date','category','protocol_id','protocol_eligible','comparator','backend','identities')}
            # The artifact identity binds this row's route histogram and raw timing vector.
            fields['identities']=dict(fields['identities'])
            fields['identities']['artifact']={'id':'routing_histogram_and_vector','sha256':e.digest(record)}
            raw=record['raw_vector']
            fields.update(schema=e.MEASUREMENT,arm=arm,operator=record['compute'],
                shape=[record['tokens'],record['topk'],record['inputs'],record['outputs']],metric='operator_latency',
                value=sum(raw)/len(raw),unit='ms',metric_direction='lower_better',repetitions=len(raw),
                reps_basis='7 device-event intervals, each 20 complete operator calls; route histogram bound to artifact identity',
                raw_vector=raw,aggregation='arithmetic_mean',
                claim='Observed standalone operator latency for '+arm+' under the recorded route histogram')
            measured=e.write(out/'native',fields);e.project(measured);measurement_ids.append(measured['row_id'])
    print(json.dumps({'returncode':code,'verifier':row['row_id'],'in_window_residency':in_window,'output':str(out),'measurements':measurement_ids}))
    raise SystemExit(code)

if __name__=='__main__':
    try:main()
    except Refusal as exc:
        print(json.dumps({'status':'refused','reason':str(exc)}),file=sys.stderr);raise SystemExit(2)
