import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from odl_bench import pip02_lineage as l
from odl_bench.pip02_records import canon,sha

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canon(x))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(t,first,shard):
 root=t/'r15';root.mkdir();source=t/'worker.py';source.write_text('native');reg=root/'source-pins.json';dump(reg,{str(source):digest(source)});pins={str(reg.resolve()):digest(reg),str(source.resolve()):digest(source)};campaign={'server_pins':{'model':'m'}};page={'image':'x','index':shard};grant='regions=48-71 start=2026-10-29T23:00Z end=2026-10-30T00:00Z\n';cap={'id':f'codex3-pip02-paddle-relocatable-r15-pool{((first-5)//3)%4}-20261030','shards':list(range(first,min(first+3,151))),'smoke':False,'source_pins':{str(source):digest(source)},'protocol_sha256':l.PROTOCOL,'grant':grant,'grant_sha256':sha(grant.encode())};cp=root/'attempts/a/capture.json';dump(cp,cap);j=t/'journal';prefix='region-'+sha(canon(page))[:24]+'-000000';start=j/f'{prefix}.start.json';response=j/f'{prefix}.result.json';raw=b'req';body=b'resp';identity={'campaign_sha256':sha(canon(campaign)),'model_pins':campaign['server_pins'],'source_pins':cap['source_pins']};dump(start,{'provenance':{'identity':identity,'origin_grant':grant,'origin_capture_sha256':digest(cp)},'body_hex':raw.hex(),'body_sha256':sha(raw),'originated_at':'2026-10-29T23:10:00+00:00'});dump(response,{'request_sha256':sha(raw),'start_sha256':digest(start),'body_hex':body.hex(),'response_sha256':sha(body),'status':200});key=sha(canon({'campaign':sha(canon(campaign)),'page':page}));result=j/(key+'.result');dump(result,{'markdown':'x','markdown_sha256':sha(b'x')});finish=j/(key+'.finish.json');dump(finish,{'schema':'pip02-paddle-page-finish/v1','page':page,'campaign_sha256':sha(canon(campaign)),'result_sha256':digest(result),'markdown_sha256':sha(b'x'),'capture_sha256':digest(cp),'grant_sha256':cap['grant_sha256'],'source_pins_sha256':sha(canon(cap['source_pins']))});return root,reg,source,pins,j,page,campaign,cap,cp

class R15Lineage(unittest.TestCase):
 def test_all_49_groups_and_tail(self):
  for first in range(5,150,3):
   for shard in range(first,min(first+3,151)):
    with self.subTest(first=first,shard=shard),tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
     root,reg,source,pins,j,page,campaign,cap,cp=fixture(Path(tmp),first,shard)
     with patch.object(l,'PADDLE_ROOTS',{'r15':(root,digest(reg))}):self.assertTrue(l.paddle_lineage(j,page,campaign,shard,pins)['verified'])
 def test_wrong_pool_group_source_and_grant_refused(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
   root,reg,source,pins,j,page,campaign,cap,cp=fixture(Path(tmp),5,5)
   with patch.object(l,'PADDLE_ROOTS',{'r15':(root,digest(reg))}):
    source.write_text('tampered')
    with self.assertRaisesRegex(ValueError,'changed'):l.paddle_lineage(j,page,campaign,5,pins)
    source.write_text('native');cap['id']='codex3-pip02-paddle-relocatable-r15-pool1-20261030';dump(cp,cap)
    with self.assertRaisesRegex(ValueError,'pool/group'):l.paddle_lineage(j,page,campaign,5,pins)
    cap['id']='codex3-pip02-paddle-relocatable-r15-pool0-20261030';cap['shards']=[5,6,8];dump(cp,cap)
    with self.assertRaisesRegex(ValueError,'pool/group'):l.paddle_lineage(j,page,campaign,5,pins)
    cap['shards']=[5,6,7];cap['grant_sha256']='0'*64;dump(cp,cap)
    with self.assertRaisesRegex(ValueError,'grant hash'):l.paddle_lineage(j,page,campaign,5,pins)
 def test_legacy_r4_without_warrants_remains_unverified(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
   root,reg,source,pins,j,page,campaign,cap,cp=fixture(Path(tmp),11,11)
   cap['id']='codex3-pip02-paddle-parallel-owner-r4-g11-20261029';cap.pop('protocol_sha256');cap.pop('grant_sha256');dump(cp,cap)
   start=next(j.glob('*.start.json'));record=json.loads(start.read_text());record['provenance']['origin_capture_sha256']=digest(cp);dump(start,record)
   response=next(j.glob('*.result.json'));reply=json.loads(response.read_text());reply['start_sha256']=digest(start);dump(response,reply)
   with patch.object(l,'PADDLE_ROOTS',{'r15':(root,digest(reg))}):
    with self.assertRaisesRegex(ValueError,'unregistered original capture'):l.paddle_lineage(j,page,campaign,11,pins)
if __name__=='__main__':unittest.main()
