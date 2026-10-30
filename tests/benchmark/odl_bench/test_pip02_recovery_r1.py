import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import odl_bench.pip02_lineage as l
sha=lambda b:hashlib.sha256(b).hexdigest()
def put(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))
class Recovery(unittest.TestCase):
 def test_first_client_binding_and_refusals(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);rec=base/'recovery';old=base/'odl-batch-results/shard-02/page-0050';old.mkdir(parents=True);(old/'input.pdf').write_bytes(b'pdf');(old/'input-identity.json').write_bytes(b'identity');src=base/'producer.py';src.write_text('producer');log=base/'producer.log';log.write_text('failure before client');rec.mkdir();r=rec/'RESOLUTIONS.json';put(r,{'50':{'ordinal':50,'shard':2,'original_path':str(old),'evidence':{x:sha((old/x).read_bytes()) for x in ('input.pdf','input-identity.json')},'expected_absent':['start.json','client.log'],'producer_source':str(src),'producer_source_sha256':sha(src.read_bytes()),'original_launcher_log':str(log),'original_launcher_log_sha256':sha(log.read_bytes())}});manifest=rec/'SOURCES.sha256';manifest.write_text(f'{sha(r.read_bytes())}  {r}\n{sha(src.read_bytes())}  {src}\n{sha(log.read_bytes())}  {log}\n');ms=sha(manifest.read_bytes());pins={str(p):sha(p.read_bytes()) for p in (manifest,r,src,log)};out=rec/'first-client/page-0050';out.mkdir(parents=True);(out/'out').mkdir();(out/'out/input.md').write_text('prediction');grant='regions=48-71 start=2026-10-29T23:00Z end=2026-10-30T00:00Z\n';start={'producer_id':l.RECOVERY_ID,'producer_source_manifest_path':str(manifest),'producer_source_manifest_sha256':ms,'resolution_sha256':sha(r.read_bytes()),'grant':grant,'started':1793315700};put(out/'start.json',start);put(out/'result.json',{'rc':0});put(out/'DONE',{'producer_id':l.RECOVERY_ID,'producer_source_manifest_sha256':ms,'start_sha256':sha((out/'start.json').read_bytes()),'result_sha256':sha((out/'result.json').read_bytes()),'prediction_sha256':sha((out/'out/input.md').read_bytes())})
   with patch.object(l,'RECOVERY',rec),patch.object(l,'RECOVERY_SHA',ms),patch.object(l,'BASE',base):
    self.assertEqual(l.recovery_directory(50,old,pins),out);self.assertTrue(l.odl_lineage(out,2,pins)['verified'])
    (out/'out/input.md').write_text('tamper')
    with self.assertRaises(ValueError):l.odl_lineage(out,2,pins)
    (old/'client.log').write_text('exposed')
    with self.assertRaises(ValueError):l.recovery_directory(50,old,pins)
    self.assertEqual(l.recovery_directory(51,old,pins),old)
if __name__=='__main__':unittest.main()
