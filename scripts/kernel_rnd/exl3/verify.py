"""Prospective per-proposition verifier receipts for every required EXL3 fixture."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import struct
import sys
from . import contract, oracle, fixtures, evidence


def _bytes(matrix):
    return b''.join(struct.pack('<e', value) for row in matrix for value in row)


def verify_suite(output, fixture_root=None):
    fixture_root=Path(fixture_root or Path(__file__).parent/'fixtures')
    folders=sorted(p for p in fixture_root.iterdir() if p.is_dir())
    fixtures.require_real_suite([(p/'manifest.json',p/'fixture.json') for p in folders if p.name.startswith('real-')])
    expected={f'{cb}-k{k}' for cb in ('mul1','mcg') for k in range(1,9)}
    if not expected <= {p.name for p in folders}:
        raise contract.Refusal('mandatory synthetic fixture is missing')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    date=datetime.now(timezone.utc).isoformat();run_id='exl3-portable-'+date
    host={'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
          'pid':os.getpid(),'backend':'portable_cpu','inference':False}
    host_path=output/'execution-identity.json'
    with host_path.open('xb') as stream:stream.write(contract.canonical(host))
    checker=Path(__file__).resolve()
    code=[checker,Path(contract.__file__),Path(oracle.__file__),Path(fixtures.__file__),Path(evidence.__file__)]
    source_hash=evidence.digest({str(p.name):evidence.file_hash(p) for p in code})
    total,failed=0,0
    for folder in folders:
        manifest,payload=contract.load(folder/'manifest.json')
        fixture=contract.read_json(folder/'fixture.json');ref=fixture['reference'];m=manifest['matrices'][0]
        packed=payload[m['tensors']['trellis']['path']];k=m['K']
        raw,weight,vectors=oracle.reconstruct(manifest,payload,fixture['matrix_id'])
        results={
            'packed_states':oracle.windows(packed[:32*k],k)==ref['packed_states'],
            'reconstructed_tiles':oracle.tile(packed[:32*k],k,m['codebook'])==ref['reconstructed_tile'] and contract.sha(_bytes(raw))==ref['raw_fp16_sha256'],
            'transforms':vectors['suh']==ref['suh'] and vectors['svh']==ref['svh'] and contract.sha(_bytes(weight))==ref['transformed_fp16_sha256'],
            'full_operator':oracle.gemm(manifest,payload,fixture['matrix_id'],fixture['activations'])==ref['operator_outputs'],
        }
        inputs=[folder/'manifest.json',folder/'fixture.json',host_path,*code]
        inputs += [folder/d['path'] for d in m['tensors'].values() if d is not None]
        inputs += [folder/d['path'] for d in fixture.get('provenance',[])]
        read_set=[{'path':str(p.resolve()),'sha256':evidence.file_hash(p)} for p in inputs]
        identities={
            'model':{'id':manifest['source']['repository']+'@'+manifest['source']['revision'],'sha256':evidence.digest(manifest['source'])},
            'artifact':{'id':m['id'],'sha256':manifest['artifact_sha256']},
            'source':{'id':'epyc.exl3.portable.v1','sha256':source_hash},
            'binary':{'id':sys.executable,'sha256':evidence.file_hash(sys.executable)},
            'library':{'id':'Python stdlib struct IEEE arithmetic','sha256':evidence.digest(sys.version)},
            'toolchain':{'id':platform.python_implementation()+' '+platform.python_version(),'sha256':evidence.digest(sys.version)},
            'hardware':{'id':host['platform'],'sha256':evidence.digest(host)},
            'residency':{'id':'portable CPU process '+str(os.getpid()),'sha256':evidence.file_hash(host_path)},
        }
        for stage,passed in results.items():
            proposition=f'{folder.name} {stage} equals the independent fixture reference under materialized_weight_fp32_fma_v1'
            row=dict(schema=evidence.VERIFIER,run_id=run_id,date=date,category='CANDIDATE',protocol_id='',protocol_eligible=False,
                     arm='portable-reference',comparator=fixture['oracle'],identities=identities,claim=proposition,backend='portable_cpu',
                     fixture=folder.name,path='materialized_weight_fp32_fma_v1',decided_proposition=proposition,verdict='pass' if passed else 'fail',
                     checker={'id':'epyc.exl3.verify/v1','path':str(checker),'sha256':evidence.file_hash(checker)},
                     fixture_sha256=evidence.file_hash(folder/'fixture.json'),read_set=read_set,read_set_sha256=evidence.digest(read_set))
            evidence.write(output,row);total+=1;failed+=not passed
    return {'fixtures':len(folders),'verifier_rows':total,'failed':failed,'inference':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fixtures',type=Path)
    args=p.parse_args();result=verify_suite(args.output,args.fixtures);print(json.dumps(result,sort_keys=True))
    raise SystemExit(bool(result['failed']))
