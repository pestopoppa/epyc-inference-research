import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from odl_bench import pip02_lineage as l
from odl_bench.pip02_records import canon,sha

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canon(x))
def d(p):return sha(p.read_bytes())
class StrictLineage(unittest.TestCase):
 def test_odl_captured_identity_and_legacy_refusal(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);source=t/'odl_page.py';source.write_text('producer');manifest=t/'SOURCES.sha256';manifest.write_text(f'{d(source)}  {source}\n');pins={str(source.resolve()):d(source),str(manifest.resolve()):d(manifest)};page=t/'page';grant='regions=48-71 start=2026-10-29T23:00Z end=2026-10-30T00:00Z\n';start={'producer_id':'codex3-pip02-odl-relocatable-r2-s20-20261029','producer_source_manifest_path':str(manifest),'producer_source_manifest_sha256':d(manifest),'grant':grant,'started':1793315700};dump(page/'start.json',start)
   with patch.object(l,'ODL',manifest),patch.object(l,'ODL_SHA',d(manifest)):
    self.assertTrue(l.odl_lineage(page,20,pins)['verified'])
    with self.assertRaises(ValueError):l.odl_lineage(page,21,pins)
    start['producer_source_manifest_sha256']='a'*64;dump(page/'start.json',start)
    with self.assertRaises(ValueError):l.odl_lineage(page,20,pins)
    dump(page/'start.json',{'grant':grant,'started':1793315700});self.assertFalse(l.odl_lineage(page,20,pins)['verified'])
 def test_r3_producer_scope_and_source_tampering(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);source=t/'odl_page.py';source.write_text('r3 native');manifest=t/'SOURCES.sha256';manifest.write_text(f'{d(source)}  {source}\n');pins={str(source):d(source),str(manifest):d(manifest)};page=t/'page';grant='regions=48-71 start=2026-10-29T23:00Z end=2026-10-30T00:00Z\n';start={'producer_id':'codex3-pip02-odl-relocatable-r3-s24-20261030','producer_source_manifest_path':str(manifest),'producer_source_manifest_sha256':d(manifest),'grant':grant,'started':1793315700};dump(page/'start.json',start)
   with patch.object(l,'ODL_R3',manifest),patch.object(l,'ODL_R3_SHA',d(manifest)):
    self.assertTrue(l.odl_lineage(page,24,pins)['verified'])
    with self.assertRaises(ValueError):l.odl_lineage(page,25,pins)
    start['producer_id']='codex3-pip02-odl-relocatable-r3-s23-20261030';dump(page/'start.json',start)
    with self.assertRaises(ValueError):l.odl_lineage(page,23,pins)
    start['producer_id']='codex3-pip02-odl-relocatable-r3-s24-20261030';dump(page/'start.json',start);source.write_text('tampered')
    with self.assertRaises(ValueError):l.odl_lineage(page,24,pins)
 def test_paddle_origins_cannot_imply_finisher(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);root=t/'r8';root.mkdir();source=t/'worker.py';source.write_text('native');registry=root/'source-pins.json';dump(registry,{str(source):d(source)});pins={str(registry.resolve()):d(registry),str(source.resolve()):d(source)};campaign={'server_pins':{'model':'m'}};page={'image':'x','index':7};grant='regions=48-71 start=2026-10-29T23:00Z end=2026-10-30T00:00Z\n';cap={'id':'codex3-pip02-paddle-relocatable-r8-g5-20261029','shards':[5,6,7],'smoke':False,'source_pins':{str(source):d(source)},'protocol_sha256':l.PROTOCOL,'grant':grant,'grant_sha256':sha(grant.encode())};capfile=root/'attempts/a/capture.json';dump(capfile,cap);prefix='region-'+sha(canon(page))[:24]+'-000000';start=t/'journal'/f'{prefix}.start.json';response=t/'journal'/f'{prefix}.result.json';body=b'request';output=b'response';origin={'identity':{'campaign_sha256':sha(canon(campaign)),'model_pins':campaign['server_pins'],'source_pins':cap['source_pins']},'origin_grant':grant,'origin_capture_sha256':d(capfile)};dump(start,{'provenance':origin,'body_hex':body.hex(),'body_sha256':sha(body),'originated_at':'2026-10-29T23:10:00+00:00'});dump(response,{'request_sha256':sha(body),'start_sha256':d(start),'body_hex':output.hex(),'response_sha256':sha(output),'status':200})
   with patch.object(l,'PADDLE_ROOTS',{'r8':(root,d(registry))}):
    result=l.paddle_lineage(start.parent,page,campaign,5,pins);self.assertTrue(result['request_origins_verified']);self.assertFalse(result['verified'])
    response.unlink()
    with self.assertRaisesRegex(ValueError,'unknown'):l.paddle_lineage(start.parent,page,campaign,5,pins)
    dump(response,{'request_sha256':sha(body),'start_sha256':d(start),'body_hex':output.hex(),'response_sha256':'0'*64,'status':200})
    with self.assertRaisesRegex(ValueError,'hash'):l.paddle_lineage(start.parent,page,campaign,5,pins)
    source.write_text('tampered')
    with self.assertRaisesRegex(ValueError,'changed'):l.paddle_lineage(start.parent,page,campaign,5,pins)
 def test_r9_finish_receipt_verifies_exact_capture(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);root=t/'r9';root.mkdir();source=t/'worker.py';source.write_text('native');reg=root/'source-pins.json';dump(reg,{str(source):d(source)});pins={str(reg.resolve()):d(reg),str(source.resolve()):d(source)};campaign={'server_pins':{'model':'m'}};page={'image':'x','index':7};grant='regions=48-71 start=2026-10-29T23:00Z end=2026-10-30T00:00Z\n';cap={'id':'codex3-pip02-paddle-relocatable-r9-g5-20261029','shards':[5,6,7],'smoke':False,'source_pins':{str(source):d(source)},'protocol_sha256':l.PROTOCOL,'grant':grant,'grant_sha256':sha(grant.encode())};cp=root/'attempts/a/capture.json';dump(cp,cap);j=t/'journal';prefix='region-'+sha(canon(page))[:24]+'-000000';start=j/f'{prefix}.start.json';response=j/f'{prefix}.result.json';raw=b'req';body=b'resp';identity={'campaign_sha256':sha(canon(campaign)),'model_pins':campaign['server_pins'],'source_pins':cap['source_pins']};dump(start,{'provenance':{'identity':identity,'origin_grant':grant,'origin_capture_sha256':d(cp)},'body_hex':raw.hex(),'body_sha256':sha(raw),'originated_at':'2026-10-29T23:10:00+00:00'});dump(response,{'request_sha256':sha(raw),'start_sha256':d(start),'body_hex':body.hex(),'response_sha256':sha(body),'status':200});key=sha(canon({'campaign':sha(canon(campaign)),'page':page}));result=j/(key+'.result');dump(result,{'markdown':'x','markdown_sha256':sha(b'x')});finish=j/(key+'.finish.json');receipt={'schema':'pip02-paddle-page-finish/v1','page':page,'campaign_sha256':sha(canon(campaign)),'result_sha256':d(result),'markdown_sha256':sha(b'x'),'capture_sha256':d(cp),'grant_sha256':cap['grant_sha256'],'source_pins_sha256':sha(canon(cap['source_pins']))};dump(finish,receipt)
   with patch.object(l,'PADDLE_ROOTS',{'r9':(root,d(reg))}):
    self.assertTrue(l.paddle_lineage(j,page,campaign,5,pins)['verified'])
    receipt['capture_sha256']='0'*64;dump(finish,receipt)
    with self.assertRaisesRegex(ValueError,'capture'):l.paddle_lineage(j,page,campaign,5,pins)
 def test_r10_finish_receipt_verifies_exact_capture(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);root=t/'r10';root.mkdir();source=t/'worker.py';source.write_text('native');reg=root/'source-pins.json';dump(reg,{str(source):d(source)});pins={str(reg.resolve()):d(reg),str(source.resolve()):d(source)};campaign={'server_pins':{'model':'m'}};page={'image':'x','index':7};grant='regions=48-71 start=2026-10-29T23:00Z end=2026-10-30T00:00Z\n';cap={'id':'codex3-pip02-paddle-relocatable-r10-g5-20261029','shards':[5,6,7],'smoke':False,'source_pins':{str(source):d(source)},'protocol_sha256':l.PROTOCOL,'grant':grant,'grant_sha256':sha(grant.encode())};cp=root/'attempts/a/capture.json';dump(cp,cap);j=t/'journal';prefix='region-'+sha(canon(page))[:24]+'-000000';start=j/f'{prefix}.start.json';response=j/f'{prefix}.result.json';raw=b'req';body=b'resp';identity={'campaign_sha256':sha(canon(campaign)),'model_pins':campaign['server_pins'],'source_pins':cap['source_pins']};dump(start,{'provenance':{'identity':identity,'origin_grant':grant,'origin_capture_sha256':d(cp)},'body_hex':raw.hex(),'body_sha256':sha(raw),'originated_at':'2026-10-29T23:10:00+00:00'});dump(response,{'request_sha256':sha(raw),'start_sha256':d(start),'body_hex':body.hex(),'response_sha256':sha(body),'status':200});key=sha(canon({'campaign':sha(canon(campaign)),'page':page}));result=j/(key+'.result');dump(result,{'markdown':'x','markdown_sha256':sha(b'x')});finish=j/(key+'.finish.json');receipt={'schema':'pip02-paddle-page-finish/v1','page':page,'campaign_sha256':sha(canon(campaign)),'result_sha256':d(result),'markdown_sha256':sha(b'x'),'capture_sha256':d(cp),'grant_sha256':cap['grant_sha256'],'source_pins_sha256':sha(canon(cap['source_pins']))};dump(finish,receipt)
   with patch.object(l,'PADDLE_ROOTS',{'r10':(root,d(reg))}):
    self.assertTrue(l.paddle_lineage(j,page,campaign,5,pins)['verified'])
    receipt['capture_sha256']='0'*64;dump(finish,receipt)
    with self.assertRaisesRegex(ValueError,'capture'):l.paddle_lineage(j,page,campaign,5,pins)
 def test_r11_finish_receipt_verifies_exact_capture(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);root=t/'r11';root.mkdir();source=t/'worker.py';source.write_text('native');reg=root/'source-pins.json';dump(reg,{str(source):d(source)});pins={str(reg.resolve()):d(reg),str(source.resolve()):d(source)};campaign={'server_pins':{'model':'m'}};page={'image':'x','index':7};grant='regions=48-71 start=2026-10-29T23:00Z end=2026-10-30T00:00Z\n';cap={'id':'codex3-pip02-paddle-relocatable-r11-g5-20261029','shards':[5,6,7],'smoke':False,'source_pins':{str(source):d(source)},'protocol_sha256':l.PROTOCOL,'grant':grant,'grant_sha256':sha(grant.encode())};cp=root/'attempts/a/capture.json';dump(cp,cap);j=t/'journal';prefix='region-'+sha(canon(page))[:24]+'-000000';start=j/f'{prefix}.start.json';response=j/f'{prefix}.result.json';raw=b'req';body=b'resp';identity={'campaign_sha256':sha(canon(campaign)),'model_pins':campaign['server_pins'],'source_pins':cap['source_pins']};dump(start,{'provenance':{'identity':identity,'origin_grant':grant,'origin_capture_sha256':d(cp)},'body_hex':raw.hex(),'body_sha256':sha(raw),'originated_at':'2026-10-29T23:10:00+00:00'});dump(response,{'request_sha256':sha(raw),'start_sha256':d(start),'body_hex':body.hex(),'response_sha256':sha(body),'status':200});key=sha(canon({'campaign':sha(canon(campaign)),'page':page}));result=j/(key+'.result');dump(result,{'markdown':'x','markdown_sha256':sha(b'x')});finish=j/(key+'.finish.json');receipt={'schema':'pip02-paddle-page-finish/v1','page':page,'campaign_sha256':sha(canon(campaign)),'result_sha256':d(result),'markdown_sha256':sha(b'x'),'capture_sha256':d(cp),'grant_sha256':cap['grant_sha256'],'source_pins_sha256':sha(canon(cap['source_pins']))};dump(finish,receipt)
   with patch.object(l,'PADDLE_ROOTS',{'r11':(root,d(reg))}):
    self.assertTrue(l.paddle_lineage(j,page,campaign,5,pins)['verified'])
    receipt['capture_sha256']='0'*64;dump(finish,receipt)
    with self.assertRaisesRegex(ValueError,'capture'):l.paddle_lineage(j,page,campaign,5,pins)
if __name__=='__main__':unittest.main()
