"""Strict source-bound lineage; no file-mtime or post-hoc producer inference."""
import datetime,hashlib,json,re
from pathlib import Path
from .pip02_records import canon,sha
BASE=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029')
ODL=BASE/'odl-relocatable-r2'/'SOURCES.sha256'
ODL_SHA='a17e08f66ffcd3c49e980110e444db2841ae5367b5f6dfff410db18423897cb9'
PADDLE_ROOTS={
 'r7':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-parallel-owner-r7-20261029'),'fdae650ccb36ace00838e5be695f93b196d5ae13deae0b47510ed17208ac12a8'),
 'r8':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r8-20261029'),'809585fee03923bb58c769d33aac53a3afa73eb08a4b4283f19f10c28d5837d0'),
 'r9':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r9-20261029'),'f553ebd2c1239866c35bb93e7a5b5794bc6ecaebc90b7a01ef1fc34e2f5a0b23'),
 'r10':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r10-20261030'),'6acc1d15c2bb3a2804cead02b1df916557147bff52637a78c472d2fefa0b2180'),
 'r11':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r11-20261030'),'20b60b7ac3fb6fcd4db6d49163eea7d3158a4be0c6b4a94e04a2fab850226ebb'),
}
PROTOCOL='dc7d8e167edc79a07b9db10849940cb84a7f04c34f67274afbe067531d226ae3'
def pinned(path,pins):
 p=Path(path).resolve();expected=pins.get(str(p))
 if not expected:raise ValueError('source evidence is not pinned: '+str(p))
 if sha(p.read_bytes())!=expected:raise ValueError('source evidence changed: '+str(p))
 return p

def grant_window(grant,epoch=None):
 m=re.fullmatch(r'regions=(0-23|24-47|48-71|72-95) start=([^ ]+) end=([^\s]+)\s*',grant or '')
 if not m:raise ValueError('grant schema')
 region,start,end=m.groups();a=datetime.datetime.fromisoformat(start.replace('Z','+00:00'));b=datetime.datetime.fromisoformat(end.replace('Z','+00:00'))
 if a.tzinfo is None or b.tzinfo is None or a>=b:raise ValueError('grant interval')
 if epoch is not None and not a.timestamp()<=epoch<=b.timestamp():raise ValueError('exposure outside captured grant')
 return region

def odl_lineage(page_dir,shard,pins):
 start=json.loads((Path(page_dir)/'start.json').read_bytes());producer=start.get('producer_id');manifest=start.get('producer_source_manifest_path');digest=start.get('producer_source_manifest_sha256')
 if not producer and not manifest and not digest:
  return {'verified':False,'basis':'legacy ODL start does not capture producer source; cannot infer from packet time or mtime'}
 if producer!=f'codex3-pip02-odl-relocatable-r2-s{shard}-20261029' or manifest!=str(ODL) or digest!=ODL_SHA:raise ValueError('unregistered ODL captured producer')
 p=pinned(ODL,pins)
 if sha(p.read_bytes())!=ODL_SHA:raise ValueError('ODL source manifest mismatch')
 for line in p.read_text().splitlines():
  expected,name=line.split('  ',1);source=pinned(name,pins)
  if sha(source.read_bytes())!=expected:raise ValueError('ODL registered source mismatch')
 started=start.get('started')
 if type(started) not in (int,float) or started<=0:raise ValueError('missing native start timestamp')
 grant_window(start.get('grant'),started)
 return {'verified':True,'basis':'pre-client producer id and registered source-manifest digest captured in native start; grant interval verified','source_manifest_sha256':digest,'grant_sha256':sha(start['grant'].encode())}

def paddle_lineage(directory,page,campaign,shard,pins):
 campaign_sha=sha(canon(campaign));prefix='region-'+sha(canon(page))[:24]+'-';starts=sorted(Path(directory).glob(prefix+'*.start.json'))
 if not starts:raise ValueError('no native request-origin receipts')
 captures={}
 for version,(root,registered_sha) in PADDLE_ROOTS.items():
  registry=pinned(root/'source-pins.json',pins)
  if sha(registry.read_bytes())!=registered_sha:raise ValueError('unregistered Paddle source registry')
  registered=json.loads(registry.read_bytes())
  for name,digest in registered.items():
   if sha(pinned(name,pins).read_bytes())!=digest:raise ValueError('Paddle registered source mismatch')
  for path in (root/'attempts').glob('*/capture.json'):
   raw=path.read_bytes();cap=json.loads(raw)
   if cap.get('smoke') is not False or shard not in cap.get('shards',[]):continue
   if cap.get('source_pins')!=registered or cap.get('protocol_sha256')!=PROTOCOL:continue
   ident=cap.get('id','');pattern={'r7':r'codex3-pip02-paddle-parallel-owner-r7-g\d+-20261029','r8':r'codex3-pip02-paddle-relocatable-r8-g\d+-20261029','r9':r'codex3-pip02-paddle-relocatable-r9-g\d+-20261029','r10':r'codex3-pip02-paddle-relocatable-r10-g\d+-20261029','r11':r'codex3-pip02-paddle-relocatable-r11-g\d+-20261029'}[version]
   if not re.fullmatch(pattern,ident):continue
   grant=cap.get('grant');grant_window(grant)
   if sha(grant.encode())!=cap.get('grant_sha256'):raise ValueError('Paddle captured grant hash')
   captures[sha(raw)]=cap
 origins=set()
 for p in starts:
  a=json.loads(p.read_bytes());response=p.with_name(p.name.replace('.start.json','.result.json'))
  if not response.is_file():raise ValueError('unknown native request exposure')
  b=json.loads(response.read_bytes());prov=a['provenance'];identity=prov['identity'];cap=captures.get(prov['origin_capture_sha256'])
  if cap is None:raise ValueError('unregistered original capture')
  if identity.get('campaign_sha256')!=campaign_sha or identity.get('model_pins')!=campaign['server_pins'] or identity.get('source_pins')!=cap['source_pins'] or prov.get('origin_grant')!=cap['grant']:raise ValueError('request origin binding')
  raw=bytes.fromhex(a['body_hex']);body=bytes.fromhex(b['body_hex'])
  if sha(raw)!=a['body_sha256'] or b['request_sha256']!=a['body_sha256'] or b['start_sha256']!=sha(p.read_bytes()) or sha(body)!=b['response_sha256'] or not 200<=b['status']<300:raise ValueError('request/response hash')
  stamp=datetime.datetime.fromisoformat(a['originated_at'].replace('Z','+00:00'))
  if stamp.tzinfo is None:raise ValueError('request timestamp timezone')
  grant_window(cap['grant'],stamp.timestamp());origins.add(prov['origin_capture_sha256'])
 key=sha(canon({'campaign':campaign_sha,'page':page}));finish=Path(directory)/(key+'.finish.json')
 if not finish.exists():return {'verified':False,'request_origins_verified':True,'origin_capture_sha256':sorted(origins),'basis':'native request origins verified; page-result record has no assembler/finishing-capture binding, so page lineage remains unverified'}
 receipt=json.loads(finish.read_bytes());cap=captures.get(receipt.get('capture_sha256'));result_path=Path(directory)/(key+'.result');result=json.loads(result_path.read_bytes())
 if cap is None or not re.fullmatch(r'codex3-pip02-paddle-relocatable-r(?:9|10|11)-g\d+-20261029',cap['id']):raise ValueError('finish capture is not registered R9/R10/R11')
 source_sha=sha(canon(cap['source_pins']))
 if receipt.get('schema')!='pip02-paddle-page-finish/v1' or receipt.get('page')!=page or receipt.get('campaign_sha256')!=campaign_sha or receipt.get('result_sha256')!=sha(result_path.read_bytes()) or receipt.get('markdown_sha256')!=result.get('markdown_sha256') or sha(result['markdown'].encode())!=receipt.get('markdown_sha256') or receipt.get('source_pins_sha256')!=source_sha or receipt.get('grant_sha256')!=cap['grant_sha256']:raise ValueError('native finish receipt mismatch')
 return {'verified':True,'request_origins_verified':True,'origin_capture_sha256':sorted(origins),'finish_capture_sha256':receipt['capture_sha256'],'basis':'native R9/R10/R11 finish receipt binds page result and markdown to captured grant and registered source'}
