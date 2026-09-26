#!/usr/bin/env python3
"""Run host dispatch checks and seal a prospective native EXL3 verifier row."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import shutil

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--contract-root',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixture',type=Path,action='append',default=[])
    a=p.parse_args();sys.path.insert(0,str(a.contract_root.resolve()))
    from scripts.kernel_rnd.exl3 import evidence as e
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    build=json.loads((a.build/'build.json').read_text())
    built=a.build/'test_contract'
    if e.file_hash(built)!=build['artifact_sha256']['test_contract']:
        raise SystemExit('build binary digest drift')
    snapshot=out/'snapshot';snapshot.mkdir(exist_ok=False)
    for source in (HERE/'contract.hpp',HERE/'test_contract.cpp',built,a.build/'build.json'):
        shutil.copy2(source,snapshot/source.name)
    binary=snapshot/'test_contract'
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    (out/'stdout.jsonl').write_text(result.stdout)
    (out/'stderr.txt').write_text(result.stderr)
    summary=json.loads(result.stdout) if result.returncode==0 else {'status':'fail'}
    checked='The 5120 host capability truth-table cells and 256-expert dispatch boundary assertions execute their declared outcomes'
    def ident(name,data):return {'id':name,'sha256':hashlib.sha256(data).hexdigest()}
    read_paths=[snapshot/'contract.hpp',snapshot/'test_contract.cpp',binary,out/'stdout.jsonl',out/'stderr.txt',snapshot/'build.json']
    reads=[{'path':str(path.resolve()),'sha256':e.file_hash(path)} for path in read_paths]
    fields=dict(schema=e.VERIFIER,run_id=out.name,date=datetime.now(timezone.utc).isoformat(),category='CANDIDATE',
                protocol_id='',protocol_eligible=False,arm='gfx90a_dispatch_host',comparator='declared_capability_table',
                backend='host_cpp17',fixture='dispatch_truth_table_and_domain_boundaries',path='plan_schedule',
                decided_proposition=checked,claim=checked,verdict='pass' if result.returncode==0 else 'fail',
                checker={'id':'exl3_gfx90a_host_checks/v1','path':str(binary.resolve()),'sha256':e.file_hash(binary)},
                fixture_sha256=e.file_hash(snapshot/'test_contract.cpp'),read_set=reads,read_set_sha256=e.digest(reads),
                identities={
                    'model':ident('not_applicable_no_model',b'no-model'),
                    'artifact':ident('synthetic_dispatch_fixture',(HERE/'test_contract.cpp').read_bytes()),
                    'source':ident('contract.hpp',(HERE/'contract.hpp').read_bytes()),
                    'binary':{'id':str(binary.resolve()),'sha256':e.file_hash(binary)},
                    'library':ident('host_standard_library',subprocess.check_output(['ldd',str(binary)])),
                    'toolchain':ident('g++',subprocess.check_output(['g++','--version'])),
                    'hardware':ident('host',platform.uname()._asdict().__repr__().encode()),
                    'residency':ident('not_applicable_host_only',b'no-device-execution')})
    row=e.write(out/'native',fields);e.project(row)
    fixture_rows=[]
    if a.fixture:
        from fixtures import export
        runtime=snapshot/'test_runtime';shutil.copy2(a.build/'test_runtime',runtime)
        shutil.copy2(HERE/'test_runtime.hip',snapshot/'test_runtime.hip')
        if e.file_hash(runtime)!=build['artifact_sha256']['test_runtime']:
            raise SystemExit('runtime harness build digest drift')
        for manifest in a.fixture:
            copied=out/'fixtures'/manifest.parent.name
            shutil.copytree(manifest.parent,copied)
            transport=copied/'gpu-fixture.bin';record=export(copied/'manifest.json',transport)
            command=[str(runtime),'--host-fixture',str(transport),record['canonical_artifact_sha256']]
            checked_run=subprocess.run(command,capture_output=True,text=True)
            (copied/'stdout.jsonl').write_text(checked_run.stdout);(copied/'stderr.txt').write_text(checked_run.stderr)
            fixture_read_paths=[snapshot/'test_runtime.hip',runtime,*sorted(path for path in copied.iterdir() if path.is_file())]
            fixture_reads=[{'path':str(path.resolve()),'sha256':e.file_hash(path)} for path in fixture_read_paths]
            exact='Independent C++ ascending FP32 materialized-weight output equals the canonical '+manifest.parent.name+' golden exactly'
            f=dict(fields);f.update(fixture=manifest.parent.name,path='materialized_weight_fp32_fma_v1',
                claim=exact,decided_proposition=exact,verdict='pass' if checked_run.returncode==0 else 'fail',
                checker={'id':'exl3_gfx90a_portable_reference/v1','path':str(runtime),'sha256':e.file_hash(runtime)},
                fixture_sha256=e.file_hash(copied/'fixture.json'),read_set=fixture_reads,read_set_sha256=e.digest(fixture_reads))
            f['identities']=dict(fields['identities']);f['identities'].update(
                model=ident(record['source']['repository']+'@'+record['source']['revision'],e.canonical(record['source'])),
                artifact={'id':record['canonical_artifact_sha256'],'sha256':record['canonical_artifact_sha256']},
                source={'id':'native_packed_trellis','sha256':record['packed_tensor_sha256']},
                binary={'id':str(runtime),'sha256':e.file_hash(runtime)},
                library=ident('HIP_linked_host_reference_dependencies',subprocess.check_output(['ldd',str(runtime)])),
                toolchain=ident('rocm62_hipcc',build['compiler'].encode()))
            frow=e.write(out/'native',f);e.project(frow);fixture_rows.append(frow['row_id'])
            if checked_run.returncode:raise SystemExit(checked_run.stderr)
    print(json.dumps({'status':summary['status'],'checks':summary.get('checks'),
                      'verifier_row':row['row_id'],'fixture_rows':fixture_rows,'native_directory':str(out/'native')}))
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
