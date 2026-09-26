"""Second-oracle check against a revision-pinned Apache donor; no source vendoring."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from . import oracle, evidence, contract

REVISION='3753c33b0b70737a60a8859f4c4ad0b136a0ae22'


def check(repo, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    sources={}
    for source,name in [('LICENSE','donor.LICENSE'),('src/vllm_exl3/dequant.py','donor.py')]:
        data=subprocess.check_output(['git','-C',str(repo),'show',REVISION+':'+source])
        path=output/name
        with path.open('xb') as stream:stream.write(data)
        sources[name]=path
    if b'Apache License' not in sources['donor.LICENSE'].read_bytes()[:200]:
        raise contract.Refusal('second oracle is not the pinned Apache revision')
    spec=importlib.util.spec_from_file_location('_exl3_apache_donor',sources['donor.py'])
    donor=importlib.util.module_from_spec(spec);spec.loader.exec_module(donor)
    fixture=output/'domain.json';fixture.write_bytes(contract.canonical({'states':[0,65535],'count':65536,'revision':REVISION}))
    checker=Path(__file__).resolve()
    read_set=[{'path':str(p),'sha256':evidence.file_hash(p)} for p in [checker,Path(oracle.__file__),fixture,*sources.values()]]
    result={}
    for cb,number in [('mcg',1),('mul1',2)]:
        mismatches=sum(oracle.decode(window,cb)!=donor.half_bits_to_float_py(donor.decode_codebook_bits_py(window,number)) for window in range(65536))
        proposition=f'Portable {cb} reconstructs all 65536 uint16 states exactly as Apache donor {REVISION}'
        identities={k:{'id':'not_applicable:'+k,'sha256':evidence.digest({'not_applicable':k,'scope':'codebook-only'})} for k in evidence.IDENTITIES}
        identities.update(source={'id':REVISION,'sha256':evidence.file_hash(sources['donor.py'])},
                          binary={'id':sys.executable,'sha256':evidence.file_hash(sys.executable)},
                          artifact={'id':'all_uint16_states','sha256':evidence.file_hash(fixture)})
        row=dict(schema=evidence.VERIFIER,run_id='exl3-donor-'+REVISION,date=datetime.now(timezone.utc).isoformat(),category='CANDIDATE',protocol_id='',protocol_eligible=False,
                 arm='portable',comparator='apache-donor',identities=identities,claim=proposition,backend='portable_cpu',fixture='uint16-'+cb,path='codebook-half-rne',
                 decided_proposition=proposition,verdict='pass' if mismatches==0 else 'fail',
                 checker={'id':'epyc.exl3.check_donor/v1','path':str(checker),'sha256':evidence.file_hash(checker)},
                 fixture_sha256=evidence.file_hash(fixture),read_set=read_set,read_set_sha256=evidence.digest(read_set))
        evidence.write(output,row);result[cb]=mismatches
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=check(a.repo,a.output);print(json.dumps(result));raise SystemExit(any(result.values()))
