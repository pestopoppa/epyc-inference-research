"""Import retained CPU donor captures; mutable revisions never clear G1."""
import argparse
import json
from pathlib import Path
import struct
from .contract import canonical, digest, sha, Refusal
from .fixtures import generate, golden


def import_capture(donor_manifest, output, revisions=None):
    revisions=revisions or {};donor_manifest=Path(donor_manifest)
    index=json.loads(donor_manifest.read_text());result=[]
    for item in index['fixtures']:
        name=Path(item['file']).stem;folder=Path(output)/('real-'+name.replace('_','-'))
        cb='mcg' if name.startswith('mcg') else 'mul1'
        raw_capture=(donor_manifest.parent/item['file']).read_bytes()
        if sha(raw_capture)!=item['sha256']:raise Refusal('donor fixture digest mismatch')
        magic,k,cb_id,dim=struct.unpack_from('<4I',raw_capture)
        if magic!=0x334C5845 or dim!=128 or cb_id!=int(cb=='mcg'):raise Refusal('unknown donor capture header')
        remote = revisions.get(name)
        if remote is not None:
            comparisons = remote.get('comparisons', [])
            if len(comparisons) < 3 or any(x.get('equal_retained_capture') is not True for x in comparisons):
                raise Refusal('revision verification must contain successful byte comparisons')
            for comparison in comparisons:
                suffix = comparison['tensor'].split('.')[-1]
                source = Path(item['source'])
                separator = '.' if cb == 'mcg' else '_'
                source = Path(str(source).removesuffix(separator+'trellis.bin')+separator+suffix+'.bin')
                with source.open('rb') as stream:
                    captured = stream.read(comparison['length'])
                if sha(captured) != comparison['sha256']:
                    raise Refusal('revision receipt disagrees with retained bytes')
        generate(folder,k,cb,make_golden=False)
        manifest=json.loads((folder/'manifest.json').read_text());matrix=manifest['matrices'][0]
        size=128*128*k//8;packed=raw_capture[16:16+size]
        suh=struct.unpack_from('<128f',raw_capture,16+size);svh=struct.unpack_from('<128f',raw_capture,16+size+512)
        payload={'trellis.bin':packed,'suh.bin':struct.pack('<128e',*suh),'svh.bin':struct.pack('<128e',*svh)}
        for key,data in payload.items():
            (folder/key).write_bytes(data);desc=matrix['tensors'][key.split('.')[0]];desc.update(sha256=sha(data),nbytes=len(data))
        matrix['tensors']['bias']=None;(folder/'bias.bin').unlink()
        matrix['tensor_role']='gate';matrix['shape']=[128,128];matrix['source_sha256']=sha(packed)
        if cb=='mcg':
            matrix['expert_domain']={'kind':'expert','local':0,'global':0,'count':1}
            repository='https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Q4';revision='99cccdf0e8741715662c383828a9ea601990c125';kind='real_weight'
        elif name in revisions:
            repository='https://huggingface.co/'+item['model'];revision=revisions[name]['revision'];kind='real_weight'
        else:
            repository='local-retained-capture:'+item['model'];revision=sha(raw_capture);kind='legacy_real_capture'
        manifest['source']={'repository':repository,'revision':revision,'kind':kind}
        manifest['artifact_sha256']=digest({k:v for k,v in manifest.items() if k!='artifact_sha256'})
        activations=[[(i*7+r*3)%17/16-.5 for i in range(128)] for r in range(3)]
        reference=golden(manifest,payload,activations)
        donor_raw=struct.unpack_from('<16384f',raw_capture,16+size+1024)
        donor_weight=struct.unpack_from('<16384f',raw_capture,16+size+1024+65536)
        if reference['raw_fp16_sha256']!=sha(struct.pack('<16384e',*donor_raw)) or reference['transformed_fp16_sha256']!=sha(struct.pack('<16384e',*donor_weight)):
            raise Refusal('independent CPU donor and portable golden reconstruction disagree')
        provenance={'source_capture':item,'source_revision_status':'pinned' if kind=='real_weight' else 'upstream_revision_unavailable',
                    'capture_sha256':sha(raw_capture),'independent_generator_sha256':sha((donor_manifest.parent.parent/'make_fixtures.py').read_bytes()),
                    'canonical_generator_sha256':sha(Path(__file__).read_bytes()),'model_revision_verification':revisions.get(name),
                    'canonical_native_sha256':{key:sha(value) for key,value in payload.items()}}
        provenance_path=folder/'source-receipt.json';provenance_path.write_bytes(canonical(provenance))
        fixture=dict(schema='epyc.exl3.fixture.v1',kind=kind,artifact_sha256=manifest['artifact_sha256'],source_revision=revision,matrix_id='projection',
                     activations=activations,reference=reference,operator_path='materialized_weight_fp32_fma_v1',oracle='independent-cpu-capture-plus-portable-direct-v1',
                     provenance=[{'path':'source-receipt.json','sha256':sha(provenance_path.read_bytes())}])
        (folder/'manifest.json').write_bytes(canonical(manifest));(folder/'fixture.json').write_bytes(canonical(fixture))
        result.append({'fixture':name,'canonical_sha256':manifest['artifact_sha256'],'kind':kind})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('donor_manifest');p.add_argument('output');p.add_argument('--revisions')
    a=p.parse_args();revisions=json.loads(Path(a.revisions).read_text()) if a.revisions else None
    print(json.dumps(import_capture(a.donor_manifest,a.output,revisions),indent=2))
