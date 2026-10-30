import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from odl_bench import pip02_lineage as l
from odl_bench.pip02_records import LayeredPaths,sha,paddle,canon
from test_pip02_lineage_r15 import fixture,dump,digest

class SplitLineage(unittest.TestCase):
 def setup_split(self,t,version,producer,first,shard):
  root,reg,src,pins,j,page,campaign,cap,cp=fixture(t,first,shard)
  cap['id']=producer;dump(cp,cap)
  start=next(j.glob('region-*.start.json'));a=json.loads(start.read_text());a['provenance']['origin_capture_sha256']=digest(cp);dump(start,a)
  response=next(j.glob('region-*.result.json'));b=json.loads(response.read_text());b['start_sha256']=digest(start);dump(response,b)
  finish=next(j.glob('*.finish.json'));f=json.loads(finish.read_text());f['capture_sha256']=digest(cp);dump(finish,f)
  old=t/'old';new=t/'new';old.mkdir();new.mkdir()
  for p in j.iterdir():
   target=old if p.name.endswith('.start.json') else new
   p.rename(target/p.name)
  for p in new.iterdir():
   if p.name.endswith('.result') or p.name.endswith('.result.json') or p.name.endswith('.finish.json'):
    (new/(p.name+'.durable')).write_text(digest(p))
  layered=LayeredPaths([old,new],durable_roots=[new],finish_durable_roots=[new])
  return root,reg,pins,layered,page,campaign,old,new,cp
 def test_r22_pool_and_r27_runtime(self):
  for version,producer,first,shard in [('r22','codex3-pip02-paddle-relocatable-r22-pool0-20261030',5,5),('r25','codex3-pip02-paddle-relocatable-r27-pool0-20261030',5,5),('r26','codex3-pip02-paddle-relocatable-r27-head-20261030',3,3)]:
   with self.subTest(version=version),tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
    root,reg,pins,j,page,campaign,old,new,cp=self.setup_split(Path(tmp),version,producer,first,shard)
    if version=='r26':
     cap=json.loads(cp.read_text());cap['shards']=[3,4];dump(cp,cap)
     start=next(old.glob('*.start.json'));a=json.loads(start.read_text());a['provenance']['origin_capture_sha256']=digest(cp);dump(start,a)
     response=next(new.glob('*.result.json'));b=json.loads(response.read_text());b['start_sha256']=digest(start);dump(response,b);(new/(response.name+'.durable')).write_text(digest(response))
     finish=next(new.glob('*.finish.json'));f=json.loads(finish.read_text());f['capture_sha256']=digest(cp);dump(finish,f);(new/(finish.name+'.durable')).write_text(digest(finish))
    with patch.object(l,'PADDLE_ROOTS',{version:(root,digest(reg))}):self.assertTrue(l.paddle_lineage(j,page,campaign,shard,pins)['verified'])
 def test_orphan_duplicate_marker_and_wrong_epoch(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
   root,reg,pins,j,page,campaign,old,new,cp=self.setup_split(Path(tmp),'r22','codex3-pip02-paddle-relocatable-r22-pool0-20261030',5,5)
   with patch.object(l,'PADDLE_ROOTS',{'r22':(root,digest(reg))}):
    response=next(new.glob('*.result.json')); marker=new/(response.name+'.durable');marker.unlink()
    with self.assertRaisesRegex(ValueError,'missing durable'):l.paddle_lineage(j,page,campaign,5,pins)
    marker.write_text(digest(response));(old/response.name).write_bytes(response.read_bytes())
    with self.assertRaisesRegex(ValueError,'duplicate checkpoint'):l.paddle_lineage(j,page,campaign,5,pins)
    (old/response.name).unlink();cap=json.loads(cp.read_text());cap['id']='codex3-pip02-paddle-relocatable-r27-pool0-20261030';dump(cp,cap)
    with self.assertRaisesRegex(ValueError,'unregistered original capture'):l.paddle_lineage(j,page,campaign,5,pins)
 def test_orphan_page_result_refused(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
   root,reg,pins,j,page,campaign,old,new,cp=self.setup_split(Path(tmp),'r22','codex3-pip02-paddle-relocatable-r22-pool0-20261030',5,5)
   key=sha(canon({'campaign':sha(canon(campaign)),'page':page})); self.assertFalse((old/(key+'.start')).exists())
   with self.assertRaisesRegex(ValueError,'orphan'):paddle(j,page,campaign)
 def test_r14_origin_and_r22_finish_are_both_required(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
   t=Path(tmp);root,reg,pins,j,page,campaign,old,new,cp=self.setup_split(t,'r22','codex3-pip02-paddle-relocatable-r22-pool0-20261030',5,5)
   prior=t/'r14';prior.mkdir();source=t/'prior.py';source.write_text('r14 native');preg=prior/'source-pins.json';dump(preg,{str(source):digest(source)});pins.update({str(preg.resolve()):digest(preg),str(source.resolve()):digest(source)})
   cap=json.loads(cp.read_text());cap['id']='codex3-pip02-paddle-relocatable-r14-pool0-20261030';cap['source_pins']={str(source):digest(source)};pc=prior/'attempts/b/capture.json';dump(pc,cap)
   start=next(old.glob('*.start.json'));a=json.loads(start.read_text());a['provenance']['origin_capture_sha256']=digest(pc);a['provenance']['identity']['source_pins']=cap['source_pins'];prior_start=old/start.name.replace('000000','000001');dump(prior_start,a)
   response=next(new.glob('*.result.json'));b=json.loads(response.read_text());b['start_sha256']=digest(prior_start);prior_response=old/response.name.replace('000000','000001');dump(prior_response,b)
   with patch.object(l,'PADDLE_ROOTS',{'r14':(prior,digest(preg)),'r22':(root,digest(reg))}):
    self.assertTrue(l.paddle_lineage(j,page,campaign,5,pins)['verified'])
    prior_response.unlink()
    with self.assertRaisesRegex(ValueError,'unknown native request exposure'):l.paddle_lineage(j,page,campaign,5,pins)
 def test_refinish_and_body_hash_refused(self):
  with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
   root,reg,pins,j,page,campaign,old,new,cp=self.setup_split(Path(tmp),'r22','codex3-pip02-paddle-relocatable-r22-pool0-20261030',5,5)
   with patch.object(l,'PADDLE_ROOTS',{'r22':(root,digest(reg))}):
    finish=next(new.glob('*.finish.json'));f=json.loads(finish.read_text());f['result_sha256']='0'*64;dump(finish,f);(new/(finish.name+'.durable')).write_text(digest(finish))
    with self.assertRaisesRegex(ValueError,'finish receipt'):l.paddle_lineage(j,page,campaign,5,pins)
    f['result_sha256']=digest(next(new.glob('*.result')));dump(finish,f);(new/(finish.name+'.durable')).write_text(digest(finish))
    start=next(old.glob('*.start.json'));a=json.loads(start.read_text());a['body_hex']='00';dump(start,a)
    with self.assertRaisesRegex(ValueError,'hash'):l.paddle_lineage(j,page,campaign,5,pins)
if __name__=='__main__':unittest.main()
