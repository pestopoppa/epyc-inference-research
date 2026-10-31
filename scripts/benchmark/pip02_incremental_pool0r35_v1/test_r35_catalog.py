import json,hashlib,tempfile,sys
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from odl_bench import pip02_lineage as lin,pip02_records as rec
H=Path('/mnt/micron/codex3-pip-pool0-merged-r35-20261031');L='/mnt/raid0/llm/tmp/codex3-pip-pool0-merged-r35-20261031-0650/launch-p0.sh';LS='19359800db1b0a88f21a23947836b91efc0a5ed080f02163aeab417cb236fab8';registry=H/'source-pins.json';pins=json.loads(registry.read_bytes());pins[str(registry.resolve())]=rec.sha(registry.read_bytes());pins[L]=LS
plan=json.loads(Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029/english755-fresh-plan.json').read_bytes());page=plan['pages'][205];config=json.loads(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r11-20261030/attempts/codex3-pip02-paddle-relocatable-r11-g71-20261029-a94a922f363f40babd31928ab6f26005/worker-config.json').read_bytes());campaign=config['campaign'];ch=rec.sha(rec.canon(campaign));grant='regions=24-47 start=2026-10-31T05:00Z end=2026-10-31T06:00Z\n'
with tempfile.TemporaryDirectory() as td:
 d=Path(td);capfile=d/'capture.json';source=json.loads(registry.read_bytes());cap={'id':'codex3-pip02-paddle-relocatable-r35-pool0-20261031','smoke':False,'shards':[41,42,43],'source_pins':source,'protocol_sha256':'6dc0b9cae7a1240c46abac3603e44a7635aa7b613aa1cca4a9e2af4703cf3d4a','launcher_path':L,'launcher_sha256':LS,'grant':grant,'grant_sha256':rec.sha(grant.encode())}
 prefix='region-'+rec.sha(rec.canon(page))[:24]+'-';sp=d/(prefix+'000000.start.json');rp=d/(prefix+'000000.result.json');key=rec.sha(rec.canon({'campaign':ch,'page':page}));result=d/(key+'.result');finish=d/(key+'.finish.json')
 def write(v):
  capfile.write_bytes(rec.canon(v));capt=rec.sha(capfile.read_bytes());body=b'test request';reply=b'test response';a={'ordinal':0,'body_sha256':rec.sha(body),'body_hex':body.hex(),'originated_at':'2026-10-31T05:10:00Z','provenance':{'origin_capture_sha256':capt,'origin_grant':grant,'identity':{'campaign_sha256':ch,'model_pins':campaign['server_pins'],'source_pins':source}}};sp.write_bytes(rec.canon(a));rp.write_bytes(rec.canon({'request_sha256':a['body_sha256'],'start_sha256':rec.sha(sp.read_bytes()),'body_hex':reply.hex(),'response_sha256':rec.sha(reply),'status':200}));out={'markdown':'fixture','markdown_sha256':rec.sha(b'fixture')};result.write_bytes(rec.canon(out));finish.write_bytes(rec.canon({'schema':'pip02-paddle-page-finish/v1','page':page,'campaign_sha256':ch,'result_sha256':rec.sha(result.read_bytes()),'markdown_sha256':out['markdown_sha256'],'capture_sha256':capt,'grant_sha256':rec.sha(grant.encode()),'source_pins_sha256':rec.sha(rec.canon(source))}))
 old=Path.glob
 def glob(self,pattern):
  if self==H/'attempts':return iter([capfile])
  return old(self,pattern)
 with patch.object(lin,'PADDLE_ROOTS',{'r35':(H,rec.sha(registry.read_bytes()))}),patch.object(Path,'glob',glob):
  write(cap);assert lin.paddle_lineage(d,page,campaign,41,pins)['verified']
  for changes in ({'id':'codex3-pip02-paddle-relocatable-r35-pool2-20261031'},{'shards':[42,43,44]},{'launcher_sha256':'0'*64},{'protocol_sha256':'0'*64},{'grant_sha256':'0'*64},{'source_pins':{}},{'smoke':True}):
   write({**cap,**changes})
   try:lin.paddle_lineage(d,page,campaign,41,pins);raise AssertionError(str(changes))
   except ValueError:pass
  write(cap);bad={**pins,L:'0'*64}
  try:lin.paddle_lineage(d,page,campaign,41,bad);raise AssertionError('bad launcher pin')
  except ValueError:pass
print('synthetic R35 pool0 control: exact capture/finish positive and wrong ID/group/launcher/protocol/grant/source/smoke negatives passed; not production evidence')
