"""Focused EXL3-1 correctness and hostile metadata controls (no inference)."""
import copy
import math
from pathlib import Path
import struct
import shutil
import tempfile
import unittest
from . import contract as c, oracle as o, fixtures as f, evidence as e
FIXTURES = Path(__file__).parent/'fixtures'


def resign(m):
    m['artifact_sha256'] = c.digest({k:v for k,v in m.items() if k != 'artifact_sha256'})
    return m


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.payload = c.load(FIXTURES/'mul1-k4'/'manifest.json')

    def test_unknown_conflicting_and_capacity(self):
        changes = [lambda m:m.update(extra=True), lambda m:m['matrices'][0].update(codebook='lcg'),
                   lambda m:m['matrices'][0].update(rate_x2=7), lambda m:m['matrices'][0].update(K=True),
                   lambda m:m['matrices'][0].update(shape=[129,128]), lambda m:m['matrices'][0].update(padded_shape=[1048576,128]),
                   lambda m:m['matrices'][0].update(scaling='global_scale'), lambda m:m['matrices'][0].update(tensor_role='fused_qkv'),
                   lambda m:m['matrices'][0]['expert_domain'].update(kind='unknown'),
                   lambda m:m['matrices'][0]['tensors']['suh'].update(path='../escape'),
                   lambda m:m['matrices'][0]['tensors']['trellis'].update(nbytes=3)]
        for change in changes:
            m=copy.deepcopy(self.manifest);change(m)
            with self.assertRaises(c.Refusal):c.validate(resign(m))

    def test_duplicates_digest_and_payload(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.json';p.write_text('{"schema":1,"schema":2}')
            with self.assertRaises(c.Refusal):c.read_json(p)
        m=copy.deepcopy(self.manifest);m['artifact_sha256']='0'*64
        with self.assertRaises(c.Refusal):c.validate(m)
        for data in (b'',b'x'*len(self.payload['trellis.bin'])):
            with self.assertRaises(c.Refusal):o.reconstruct(self.manifest,{**self.payload,'trellis.bin':data},'projection')

    def test_preallocation_refusal_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);malformed=copy.deepcopy(self.manifest)
            malformed['matrices'][0]['codebook']='unknown'
            (d/'manifest.json').write_bytes(c.canonical(resign(malformed)))
            with self.assertRaisesRegex(c.Refusal,'codebook'):c.load(d/'manifest.json')
            (d/'manifest.json').write_bytes(c.canonical(self.manifest))
            for name,data in self.payload.items():(d/name).write_bytes(data)
            (d/'suh.bin').unlink();(d/'suh.bin').symlink_to(FIXTURES/'mul1-k4'/'suh.bin')
            with self.assertRaises(c.Refusal):c.load(d/'manifest.json')

    def test_nonfinite_scales_and_bias(self):
        for name in ('suh','svh','bias'):
            m,payload=copy.deepcopy(self.manifest),dict(self.payload)
            data=struct.pack('<e',math.nan)*128;payload[name+'.bin']=data
            m['matrices'][0]['tensors'][name]=dict(path=name+'.bin',sha256=c.sha(data),nbytes=256,dtype='float16_le',shape=[128])
            with self.assertRaises(c.Refusal):o.reconstruct(resign(m),payload,'projection')

    def test_repack_binding(self):
        row=dict(schema='epyc.exl3.repack.v1',canonical_sha256=self.manifest['artifact_sha256'],backend='cpu',layout='band-v1',payload_sha256='a'*64)
        c.bind_repack(row,self.manifest);row['canonical_sha256']='b'*64
        with self.assertRaises(c.Refusal):c.bind_repack(row,self.manifest)

    def test_activation_refusals(self):
        for x in ([],[['bad']*125],[[math.inf]*125],[[0]*125]*257,[None]):
            with self.assertRaises(c.Refusal):o.gemm(self.manifest,self.payload,'projection',x)
        malformed=copy.deepcopy(self.manifest);malformed['matrices']=None
        with self.assertRaises(c.Refusal):o.gemm(malformed,self.payload,'projection',[[1]])


class OracleTests(unittest.TestCase):
    def test_known_codebook_values(self):
        values={'mul1':[-3.453125,0.64111328125,-0.6513671875,-0.5771484375,-0.8681640625],
                'mcg':[1.84375,0.134521484375,0.85009765625,1.5146484375,-0.63525390625]}
        for cb,expected in values.items():self.assertEqual([o.decode(w,cb) for w in (0,1,65535,0x1234,0x8000)],expected)
        with self.assertRaises(c.Refusal):o.decode(0,'unknown')

    def test_exhaustive_independent_codebooks(self):
        for cb in ('mul1','mcg'):
            for w in range(65536):self.assertEqual(o.decode(w,cb),f.golden_decode(w,cb))

    def test_known_windows_and_lane_map(self):
        for k in (1,8):
            packed=struct.pack('<I',0x80000000)+b'\0'*(32*k-4);states=o.windows(packed,k)
            self.assertEqual(states,f.golden_windows(packed,k))
            self.assertEqual(states[:16] if k==1 else states[:3],[1<<i for i in range(16)] if k==1 else [128,32768,0])
        self.assertEqual([o.state_index(r,c) for r,c in [(0,0),(1,0),(8,0),(0,8),(0,1),(0,2),(15,15)]],[0,1,2,4,32,64,255])
        self.assertEqual(len({o.state_index(r,c) for r in range(16) for c in range(16)}),256)

    def test_transform(self):
        impulse=[0.0]*128;impulse[0]=1.0
        self.assertEqual(o.hadamard(impulse),[o.half(o.f32(1/math.sqrt(128)))]*128)
        values=[(i-64)/32 for i in range(128)]
        self.assertEqual(o.hadamard(values),f.golden_h(values))

    def test_all_synthetic_stages_and_full_gemv_gemm(self):
        for folder in sorted(FIXTURES.iterdir()):
            with self.subTest(fixture=folder.name):
                m,p=c.load(folder/'manifest.json')
                cb,k=m['matrices'][0]['codebook'],m['matrices'][0]['K']
                fixture=c.read_json(folder/'fixture.json');ref=fixture['reference']
                self.assertEqual(o.windows(p['trellis.bin'][:32*k],k),ref['packed_states'])
                self.assertEqual(o.tile(p['trellis.bin'][:32*k],k,cb),ref['reconstructed_tile'])
                raw,weight,vectors=o.reconstruct(m,p,'projection')
                self.assertEqual(c.sha(f._pack(raw,'<e')),ref['raw_fp16_sha256'])
                self.assertEqual(c.sha(f._pack(weight,'<e')),ref['transformed_fp16_sha256'])
                self.assertEqual(vectors['suh'],ref['suh']);self.assertEqual(vectors['svh'],ref['svh'])
                self.assertEqual(o.gemm(m,p,'projection',fixture['activations']),ref['operator_outputs'])
                self.assertEqual(o.gemv(m,p,'projection',fixture['activations'][0]),ref['operator_outputs'][0])

    def test_real_suite_present(self):
        paths=[(p/'manifest.json',p/'fixture.json') for p in FIXTURES.glob('real-*')]
        self.assertEqual(len(paths),3)
        f.require_real_suite(paths)

    def test_real_revision_cannot_be_asserted_without_byte_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'real'
            shutil.copytree(FIXTURES/'real-mcg-k4',folder)
            receipt=c.read_json(folder/'source-receipt.json')
            receipt['model_revision_verification']=None
            (folder/'source-receipt.json').write_bytes(c.canonical(receipt))
            fixture=c.read_json(folder/'fixture.json')
            fixture['provenance'][0]['sha256']=c.sha((folder/'source-receipt.json').read_bytes())
            (folder/'fixture.json').write_bytes(c.canonical(fixture))
            with self.assertRaisesRegex(c.Refusal,'byte-verified'):
                f.admit_real(folder/'manifest.json',folder/'fixture.json')

    def test_real_gate_cannot_silently_skip(self):
        with self.assertRaisesRegex(c.Refusal,'real MUL1'):f.require_real_suite([])
        with self.assertRaisesRegex(c.Refusal,'real-weight'):f.admit_real(FIXTURES/'mcg-k4'/'manifest.json',FIXTURES/'mcg-k4'/'fixture.json')


def evidence_fields(schema=e.MEASUREMENT):
    row=dict(schema=schema,run_id='exl3-unit-20260926',date='2026-09-26T00:00:00Z',category='CANDIDATE',protocol_id='',protocol_eligible=False,
             arm='portable',comparator='independent-scalar',identities={k:{'id':'test-only:'+k,'sha256':e.digest(k)} for k in e.IDENTITIES},
             claim='test latency observation',backend='portable')
    if schema==e.MEASUREMENT:row.update(operator='gemv',shape=[1,128,128],metric='latency',value=2.0,unit='ms',metric_direction='lower_better',
                                      repetitions=2,reps_basis='scored independent invocations',raw_vector=[1.0,3.0],aggregation='arithmetic_mean')
    return row


class EvidenceTests(unittest.TestCase):
    def test_measurement_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            row=e.write(d,evidence_fields());self.assertEqual(e.project(row)['value'],2)
            self.assertFalse(e.project(row)['extra']['promotion_authority'])
            for key,value in [('value',3),('schema','old'),('authority','production'),('metric_direction','unknown'),('raw_vector',[2]),('protocol_id','invented')]:
                bad=copy.deepcopy(row);bad[key]=value
                with self.assertRaises(e.EvidenceRefusal):e.project(bad)
            Path(row['attestation_path']).write_text('{}')
            with self.assertRaises(e.EvidenceRefusal):e.project(row)

    def test_verifier_roundtrip_fail_and_readset_drift(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);fixture=d/'fixture';fixture.write_text('actual fixture bytes');checker=Path(__file__).resolve()
            fields=evidence_fields(e.VERIFIER)
            fields.update(fixture='native-k4',path='materialized',decided_proposition='Native K4 reconstruction equals independent golden',
                          claim='Native K4 reconstruction equals independent golden',verdict='fail',
                          checker={'id':'test_exl3/v1','path':str(checker),'sha256':e.file_hash(checker)},fixture_sha256=e.file_hash(fixture),
                          read_set=[{'path':str(fixture),'sha256':e.file_hash(fixture)}])
            fields['read_set_sha256']=e.digest(fields['read_set']);row=e.write(d,fields);projection=e.project(row)
            self.assertEqual(projection['value'],'fail');self.assertEqual(projection['source_class'],'verifier');self.assertEqual(projection['binding_kind'],'identity')
            fixture.write_text('changed')
            with self.assertRaises(e.EvidenceRefusal):e.project(row)

    def test_bad_writer_rows_refused_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            row=evidence_fields();row['repetitions']=True
            with self.assertRaises(e.EvidenceRefusal):e.write(d,row)
            self.assertEqual(list(Path(d).iterdir()),[])


if __name__=='__main__':unittest.main()
