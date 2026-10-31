"""Strict source-bound lineage; no file-mtime or post-hoc producer inference."""
import datetime,hashlib,json,re
from pathlib import Path
from .pip02_records import canon,sha,LayeredPaths
BASE=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029')
ODL=BASE/'odl-relocatable-r2'/'SOURCES.sha256'
ODL_SHA='a17e08f66ffcd3c49e980110e444db2841ae5367b5f6dfff410db18423897cb9'
RECOVERY=BASE/'odl-recover-50-118-310-r1-20261030'
RECOVERY_SHA='b2ce8fc980bba8d6aaca2bfc73508e24db33e798baa799e55d782af78ef449cd'
RECOVERY_ID='codex3-pip02-odl-recover-50-118-310-r1-20261030'
ODL_R3=BASE/'odl-relocatable-r3-20261030'/'SOURCES.sha256'
ODL_R3_SHA='2cbd60601aff752f709a5e7bf47b6f6f207dfcfbcad383a06ef300cec99f4961'
ODL_R4=BASE/'odl-relocatable-r4-20261030'/'SOURCES.sha256'
ODL_R4_SHA='af6262aaf380b3a69031fc5b7f5b11ec0f8b697f314c410c159568bccf4cc62a'
PADDLE_ROOTS={
 'r7':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-parallel-owner-r7-20261029'),'fdae650ccb36ace00838e5be695f93b196d5ae13deae0b47510ed17208ac12a8'),
 'r8':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r8-20261029'),'809585fee03923bb58c769d33aac53a3afa73eb08a4b4283f19f10c28d5837d0'),
 'r9':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r9-20261029'),'f553ebd2c1239866c35bb93e7a5b5794bc6ecaebc90b7a01ef1fc34e2f5a0b23'),
 'r10':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r10-20261030'),'6acc1d15c2bb3a2804cead02b1df916557147bff52637a78c472d2fefa0b2180'),
 'r11':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r11-20261030'),'20b60b7ac3fb6fcd4db6d49163eea7d3158a4be0c6b4a94e04a2fab850226ebb'),
 'r13':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-relocatable-r13-20261030'),'9e206417a8654fbd79f5d73238d152dbca25282a09be796bf3e0e56b0e8b3a58'),
 'r14':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-pool-r14-20261030/smoke-native'),'fe9ea7d7c98ff6b0280cfd7955fcd227fb4e8c028895b0d00e1d18b877ffd879'),
 'r15':(Path('/mnt/raid0/llm/tmp/codex3-pip02-paddle-pool-r15-20261030/smoke-native'),'07e23ed4e08b3a3accd46c5529e9ee992d02982ce1026d23c53377bdef65f926'),
 'r22':(Path('/mnt/micron/codex3-pip-split-r22-20261030'),'2019dcee79c82de51497a77605062bf534572119e0aa104d8cd3c308af4529f5'),
 'r25':(Path('/mnt/micron/codex3-pip-relocatable-r25-20261030'),'df6c355fa89f5182f28a3d25ba8b86f5524b31db682cb99ffa560070144cdbcd'),
 'r30':(Path('/mnt/micron/codex3-pip-pool2-r30-v2-20261031'),'da1b844dfe244a93a0ffcfe8f85303a2a3d3afe24021d228c698526720583e36'),
 'r26':(Path('/mnt/micron/codex3-pip-head-r26-20261030'),'11d496e2b52ea9e5034df15feeeae7602a06587f8ef14a47d07017022ca430ac'),
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

def recovery_directory(ordinal,original,pins):
 if ordinal not in (50,118,310):return Path(original)
 recovered=RECOVERY/'first-client'/f'page-{ordinal:04d}'
 if not recovered.exists():return Path(original)
 if pins is None:raise ValueError('recovery requires source pins')
 manifest=pinned(RECOVERY/'SOURCES.sha256',pins)
 if sha(manifest.read_bytes())!=RECOVERY_SHA:raise ValueError('recovery source manifest mismatch')
 for line in manifest.read_text().splitlines():
  expected,name=line.split('  ',1)
  if sha(pinned(name,pins).read_bytes())!=expected:raise ValueError('recovery source mismatch')
 resolution=pinned(RECOVERY/'RESOLUTIONS.json',pins);rr=json.loads(resolution.read_bytes())[str(ordinal)];old=Path(original)
 if rr['ordinal']!=ordinal or rr['shard']!=ordinal//25 or Path(rr['original_path'])!=old:raise ValueError('recovery original identity mismatch')
 if any((old/name).exists() for name in rr['expected_absent']):raise ValueError('original pre-client evidence changed')
 for name,expected in rr['evidence'].items():
  if sha((old/name).read_bytes())!=expected:raise ValueError('original input evidence changed')
 if sha(pinned(rr['producer_source'],pins).read_bytes())!=rr['producer_source_sha256']:raise ValueError('original producer changed')
 if sha(pinned(rr['original_launcher_log'],pins).read_bytes())!=rr['original_launcher_log_sha256']:raise ValueError('original failure log changed')
 return recovered

def odl_lineage(page_dir,shard,pins):
 start=json.loads((Path(page_dir)/'start.json').read_bytes());producer=start.get('producer_id');manifest=start.get('producer_source_manifest_path');digest=start.get('producer_source_manifest_sha256')
 if not producer and not manifest and not digest:
  return {'verified':False,'basis':'legacy ODL start does not capture producer source; cannot infer from packet time or mtime'}
 if producer==RECOVERY_ID and manifest==str(RECOVERY/'SOURCES.sha256') and digest==RECOVERY_SHA:
  m=re.fullmatch(r'page-(\d{4})',Path(page_dir).name)
  if not m:raise ValueError('recovery page path')
  ordinal=int(m.group(1)); original=BASE/f'odl-batch-results/shard-{ordinal//25:02d}/page-{ordinal:04d}'
  if ordinal//25!=shard or recovery_directory(ordinal,original,pins)!=Path(page_dir):raise ValueError('recovery scope mismatch')
  p=pinned(RECOVERY/'SOURCES.sha256',pins)
  done=json.loads((Path(page_dir)/'DONE').read_bytes()); result=Path(page_dir)/'result.json'; prediction=Path(page_dir)/'out/input.md'
  if start.get('resolution_sha256')!=sha(pinned(RECOVERY/'RESOLUTIONS.json',pins).read_bytes()) or done.get('producer_id')!=producer or done.get('producer_source_manifest_sha256')!=digest or done.get('start_sha256')!=sha((Path(page_dir)/'start.json').read_bytes()) or done.get('result_sha256')!=sha(result.read_bytes()) or done.get('prediction_sha256')!=sha(prediction.read_bytes()):raise ValueError('recovery completion identity mismatch')
 elif producer==f'codex3-pip02-odl-relocatable-r2-s{shard}-20261029' and manifest==str(ODL) and digest==ODL_SHA:p=pinned(ODL,pins)
 elif shard in (24,25,28,29) and producer==f'codex3-pip02-odl-relocatable-r3-s{shard}-20261030' and manifest==str(ODL_R3) and digest==ODL_R3_SHA:p=pinned(ODL_R3,pins)
 elif shard in (24,25,28,29) and producer==f'codex3-pip02-odl-relocatable-r4-s{shard}-20261030' and manifest==str(ODL_R4) and digest==ODL_R4_SHA:p=pinned(ODL_R4,pins)
 else:raise ValueError('unregistered ODL captured producer')
 if sha(p.read_bytes())!=digest:raise ValueError('ODL source manifest mismatch')
 for line in p.read_text().splitlines():
  expected,name=line.split('  ',1);source=pinned(name,pins)
  if sha(source.read_bytes())!=expected:raise ValueError('ODL registered source mismatch')
 started=start.get('started')
 if type(started) not in (int,float) or started<=0:raise ValueError('missing native start timestamp')
 grant_window(start.get('grant'),started)
 return {'verified':True,'basis':'pre-client producer id and registered source-manifest digest captured in native start; grant interval verified','source_manifest_sha256':digest,'grant_sha256':sha(start['grant'].encode())}

def paddle_lineage(directory,page,campaign,shard,pins):
 directory=directory if isinstance(directory,LayeredPaths) else Path(directory)
 campaign_sha=sha(canon(campaign));prefix='region-'+sha(canon(page))[:24]+'-';starts=sorted(directory.glob(prefix+'*.start.json'))
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
   ident=cap.get('id','')
   r28_head=(version=='r26' and ident=='codex3-pip02-paddle-relocatable-r28-head-20261031')
   r29_head=(version=='r26' and ident=='codex3-pip02-paddle-relocatable-r29-head-20261031')
   r29_sources={}
   if r29_head:
    r29_catalog='/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031/source-pins.json'
    catalog=pinned(r29_catalog,pins)
    if sha(catalog.read_bytes())!='63f15f638c88e3de96c584f621a71fb3ac778855b58cd06bef27bb64eaf2cec3':raise ValueError('R29 catalog mismatch')
    r29_sources=json.loads(catalog.read_bytes())
    for name,digest in r29_sources.items():
     if sha(pinned(name,pins).read_bytes())!=digest:raise ValueError('R29 source mismatch')
   if cap.get('source_pins')!={**registered,**r29_sources}:continue
   expected_protocol=registered.get(str(root/'PROTOCOL.md')) if r28_head or r29_head else PROTOCOL
   if (r28_head or r29_head) and expected_protocol!='be9974933703d273f65cdb174f1bdb462a2fd39debb88bf71a520e8477fba595':raise ValueError('head registered protocol mismatch')
   if version=='r30':expected_protocol='6dc0b9cae7a1240c46abac3603e44a7635aa7b613aa1cca4a9e2af4703cf3d4a'
   if cap.get('protocol_sha256')!=expected_protocol:continue
   ident=cap.get('id','');pattern={'r7':r'codex3-pip02-paddle-parallel-owner-r7-g\d+-20261029','r8':r'codex3-pip02-paddle-relocatable-r8-g\d+-20261029','r9':r'codex3-pip02-paddle-relocatable-r9-g\d+-20261029','r10':r'codex3-pip02-paddle-relocatable-r10-g\d+-20261029','r11':r'codex3-pip02-paddle-relocatable-r11-g\d+-20261029','r13':r'codex3-pip02-paddle-relocatable-r13-g(?:5|8|11|17|26|32|35|65|128)-20261030','r14':r'codex3-pip02-paddle-relocatable-r14-pool[0-3]-20261030','r15':r'codex3-pip02-paddle-relocatable-r15-pool[0-3]-20261030','r22':r'codex3-pip02-paddle-relocatable-r22-pool[0-3]-20261030','r25':r'codex3-pip02-paddle-relocatable-r(?:25|27)-pool[0-3]-20261030','r30':r'codex3-pip02-paddle-relocatable-r30-pool2-20261031','r26':r'(?:codex3-pip02-paddle-relocatable-r(?:26|27)-head-20261030|codex3-pip02-paddle-relocatable-r28-head-20261031|codex3-pip02-paddle-relocatable-r29-head-20261031)'}[version]
   if not re.fullmatch(pattern,ident):continue
   extra={}
   if ident=='codex3-pip02-paddle-relocatable-r28-head-20261031':
    launcher='/mnt/raid0/llm/tmp/codex3-filing-pip-dtap-c6-20261031-0230/repair/head/launch-p0.sh'
    pool='/mnt/raid0/llm/tmp/codex3-filing-pip-dtap-c6-20261031-0230/repair/head/pool.py'
    extra={launcher:'ae394dd5e671ff82ef472218c527675810f0d586b2a8ae6a68c3133a9e8e6e2f',pool:'71eee5884d5c5bc92f195ebfad7291eb3deb5adb462d0f98bd6b1604e4161828'}
    if cap.get('launcher_path')!=launcher or cap.get('launcher_sha256')!=extra[launcher]:raise ValueError('R28 launcher identity mismatch')
    for name,digest in extra.items():
     if sha(pinned(name,pins).read_bytes())!=digest:raise ValueError('R28 launcher or pool source mismatch')
   if r29_head:
    launcher='/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031/launch-p0.sh'
    launcher_sha='dbe28a4f43d0890a14a80037a80ebc2b52d49ab8ac3fee13b960f62e1b20fbbd'
    if cap.get('launcher_path')!=launcher or cap.get('launcher_sha256')!=launcher_sha:raise ValueError('R29 launcher identity mismatch')
    if sha(pinned(launcher,pins).read_bytes())!=launcher_sha:raise ValueError('R29 launcher changed')
    extra={**r29_sources,launcher:launcher_sha,r29_catalog:'63f15f638c88e3de96c584f621a71fb3ac778855b58cd06bef27bb64eaf2cec3'}
   if version=='r30':
    launcher='/mnt/raid0/llm/tmp/codex3-pip-pool2-r30-v2-20261031/launch-p2.sh';launcher_sha='2dbe8265199aabaae505f17520e4a84b2cae67fc7b7f218392d201353061160c'
    if cap.get('launcher_path')!=launcher or cap.get('launcher_sha256')!=launcher_sha or sha(pinned(launcher,pins).read_bytes())!=launcher_sha:raise ValueError('R30 launcher identity mismatch')
    group=cap.get('shards')
    if not isinstance(group,list) or not group or type(group[0]) is not int:raise ValueError('R30 group shape')
    first=group[0]
    if first not in [11,23,35,47,59,71,83,95,107,119,131,143] or group!=list(range(first,min(first+3,151))):raise ValueError('R30 group mismatch')
    extra={launcher:launcher_sha}
   if version=='r13':
    first=int(re.search(r'-g(\d+)-20261030$',ident).group(1))
    if cap.get('shards')!=list(range(first,first+3)):raise ValueError('R13 captured shard group mismatch')
   if version in ('r14','r15','r22','r25'):
    group=cap.get('shards')
    if not isinstance(group,list) or not group or type(group[0]) is not int:raise ValueError('R15 captured group shape')
    first=group[0];pool=int(re.search(r'-pool([0-3])-20261030$',ident).group(1))
    if not 5<=first<=149 or (first-5)%3 or ((first-5)//3)%4!=pool or group!=list(range(first,min(first+3,151))):raise ValueError('R15 captured pool/group mismatch')
   if version=='r26' and cap.get('shards') not in ([0,1,2],[3,4]):raise ValueError('head captured shard group mismatch')
   grant=cap.get('grant');grant_window(grant)
   if sha(grant.encode())!=cap.get('grant_sha256'):raise ValueError('Paddle captured grant hash')
   captures[sha(raw)]=(cap, str(path.resolve()), str(registry), registered_sha, extra)
 # The exact historical shard-1 producer captured its native source map
 # with relative names; resolve only this pinned manifest and map, never infer.
 if shard in (0,1,2):
  fresh=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029')
  manifest=pinned(fresh/'PADDLE-FRESH-SOURCES.sha256',pins)
  fresh_sha='8c8508862d3e9be31ce184da53f5e977eb3935f80f9d23856eb3a845a9175ddb'
  if sha(manifest.read_bytes())!=fresh_sha:raise ValueError('fresh-s1 manifest mismatch')
  fresh_manifest_sources={}
  for line in manifest.read_text().splitlines():
   digest,name=line.split('  ',1)
   fresh_manifest_sources[name]=digest
  for path in (fresh/'attempts').glob(f'codex3-pip02-fresh-s{shard}-20261029-*/capture.json'):
   raw=path.read_bytes();cap=json.loads(raw)
   if cap.get('id')!=f'codex3-pip02-fresh-s{shard}-20261029' or cap.get('smoke') is not False or cap.get('shard')!=shard or cap.get('protocol_sha256')!=PROTOCOL:continue
   source=cap.get('source_pins')
   if not isinstance(source,dict) or sha(canon(source))!='ed0731221fa5f91308174573c418611724db548ced92ca93db0c06ed982e2046':raise ValueError('fresh-s1 captured source map mismatch')
   if any(Path(name).is_absolute() or '..' in Path(name).parts or fresh_manifest_sources.get(str(fresh/name))!=digest for name,digest in source.items()):raise ValueError('fresh-s1 captured source binding')
   fresh_sources={str(fresh/name):digest for name,digest in source.items()}
   for name,digest in fresh_sources.items():
    if sha(pinned(name,pins).read_bytes())!=digest:raise ValueError('fresh-s1 selected source changed')
   grant=cap.get('grant');grant_window(grant)
   if sha(grant.encode())!=cap.get('grant_sha256'):raise ValueError('fresh-s1 grant hash')
   captures[sha(raw)]=(cap,str(path.resolve()),str(manifest),fresh_sha,fresh_sources)
 origins=set(); selected={}; selected_pins={}; selected_registries={}
 for p in starts:
  a=json.loads(p.read_bytes());response=directory/(p.name.replace('.start.json','.result.json'))
  if not response.is_file():raise ValueError('unknown native request exposure')
  b=json.loads(response.read_bytes());prov=a['provenance'];identity=prov['identity'];entry=captures.get(prov['origin_capture_sha256']);cap=entry[0] if entry else None
  if cap is None:raise ValueError('unregistered original capture')
  if identity.get('campaign_sha256')!=campaign_sha or identity.get('model_pins')!=campaign['server_pins'] or identity.get('source_pins')!=cap['source_pins'] or prov.get('origin_grant')!=cap['grant']:raise ValueError('request origin binding')
  raw=bytes.fromhex(a['body_hex']);body=bytes.fromhex(b['body_hex'])
  if sha(raw)!=a['body_sha256'] or b['request_sha256']!=a['body_sha256'] or b['start_sha256']!=sha(p.read_bytes()) or sha(body)!=b['response_sha256'] or not 200<=b['status']<300:raise ValueError('request/response hash')
  stamp=datetime.datetime.fromisoformat(a['originated_at'].replace('Z','+00:00'))
  if stamp.tzinfo is None:raise ValueError('request timestamp timezone')
  grant_window(cap['grant'],stamp.timestamp());origins.add(prov['origin_capture_sha256']);selected[entry[1]]=prov['origin_capture_sha256'];selected_pins[prov['origin_capture_sha256']]=entry[4] if cap['id'] in ('codex3-pip02-fresh-s0-20261029','codex3-pip02-fresh-s1-20261029','codex3-pip02-fresh-s2-20261029') else {**cap['source_pins'],**entry[4]};selected_registries[entry[2]]=entry[3]
 key=sha(canon({'campaign':campaign_sha,'page':page}));finish=directory/(key+'.finish.json')
 if not finish.exists():
  if page.get('index') not in (0,34):return {'verified':False,'request_origins_verified':True,'origin_capture_sha256':sorted(origins),'selected_captures':selected,'registered_source_pins':selected_pins,'selected_registries':selected_registries,'basis':'native request origins verified; page-result record has no assembler/finishing-capture binding, so page lineage remains unverified'}
  # A prospective offline assembly is its own producer, never an invented historical finish.
  root=Path('/mnt/raid0/llm/tmp/codex3-pip-head-assembly-20261031-0530/recovery-1/output');start_path=pinned(root/'RECOVERY-START.json',pins);finish_path=pinned(root/'RECOVERY-FINISH.json',pins)
  if sha(start_path.read_bytes())!='5bb8f7ccf46655decd5fe8c5fb21f36406f7432206cb44a7f58171bcbec089c6' or sha(finish_path.read_bytes())!='335777dd25717190398f7d65026821dc8b6dcdcbbb77f55d35f497925775eb20':raise ValueError('recovery start or finish identity')
  st=json.loads(start_path.read_bytes());done=json.loads(finish_path.read_bytes());idx=page['index'];expected={0:'8ee6a158fa5a4c3b3b472ad670a412307ce0e133b4b8a6e7b79b71a2a4801af3',34:'3291129c94de5008687337e53d1366d8b87aed33828664c4f2a1ed98b3ee4ce9'}[idx];rp=pinned(root/f'page-{idx:04}.recovery.json',pins)
  if sha(rp.read_bytes())!=expected:raise ValueError('recovery page identity')
  rec=json.loads(rp.read_bytes());sources=st.get('inputs_sha256');script=st.get('script_path')
  if st.get('schema')!='pip02-offline-assembly-recovery-start/v1' or st.get('id')!='pip02-head-offline-assembly-20261031-r1' or st.get('indices')!=[0,34] or st.get('original_missing_finish_preserved') is not True or not isinstance(sources,dict) or st.get('script_sha256')!='193494a47cda9bfe25f45531da4487f6fedd14d6b4c0f8739a39477947b5ab6d' or sources.get(script)!=st['script_sha256']:raise ValueError('recovery source binding')
  for name,digest in sources.items():
   if sha(pinned(name,pins).read_bytes())!=digest:raise ValueError('recovery input changed')
  if done.get('schema')!='pip02-offline-assembly-recovery-finish/v1' or done.get('start_path')!=str(start_path) or done.get('start_sha256')!='5bb8f7ccf46655decd5fe8c5fb21f36406f7432206cb44a7f58171bcbec089c6' or done.get('network_calls')!=0 or done.get('new_model_requests')!=0:raise ValueError('recovery completion binding')
  inventory={0:('/mnt/raid0/llm/tmp/codex3-pip-head-assembly-20261031-0530/recovery-1/output/page-0000.recovery.json','8ee6a158fa5a4c3b3b472ad670a412307ce0e133b4b8a6e7b79b71a2a4801af3'),34:('/mnt/raid0/llm/tmp/codex3-pip-head-assembly-20261031-0530/recovery-1/output/page-0034.recovery.json','3291129c94de5008687337e53d1366d8b87aed33828664c4f2a1ed98b3ee4ce9')}
  if done.get('pages')!=[{'index':i,'path':p,'sha256':h} for i,(p,h) in inventory.items()]:raise ValueError('recovery page inventory')
  orig=directory/(key+'.result');old=json.loads(orig.read_bytes());markdown=old.get('markdown')
  if rec.get('schema')!='pip02-prospective-recovery-page/v1' or rec.get('index')!=idx or rec.get('recovery_start_sha256')!='5bb8f7ccf46655decd5fe8c5fb21f36406f7432206cb44a7f58171bcbec089c6' or rec.get('network_calls')!=0 or rec.get('replayed_region_count')!={0:15,34:13}[idx] or len(starts)!=rec['replayed_region_count'] or not isinstance(markdown,str) or rec.get('markdown')!=markdown or rec.get('original_markdown_sha256')!=sha(markdown.encode()) or rec.get('reconstructed_markdown_sha256')!=sha(markdown.encode()) or old.get('markdown_sha256')!=sha(markdown.encode()) or sources.get(str(orig))!=sha(orig.read_bytes()):raise ValueError('recovery output does not reproduce original')
  a=datetime.datetime.fromisoformat(st['started_utc']);b=datetime.datetime.fromisoformat(rec['completed_utc']);c=datetime.datetime.fromisoformat(done['ended_utc'])
  if not (a.tzinfo and b.tzinfo and c.tzinfo and a<=b<=c):raise ValueError('recovery timestamp order')
  selected[str(start_path)]='5bb8f7ccf46655decd5fe8c5fb21f36406f7432206cb44a7f58171bcbec089c6';selected_pins['5bb8f7ccf46655decd5fe8c5fb21f36406f7432206cb44a7f58171bcbec089c6']={**sources,str(finish_path):'335777dd25717190398f7d65026821dc8b6dcdcbbb77f55d35f497925775eb20',str(rp):expected};selected_registries[str(finish_path)]='335777dd25717190398f7d65026821dc8b6dcdcbbb77f55d35f497925775eb20'
  return {'verified':True,'request_origins_verified':True,'origin_capture_sha256':sorted(origins),'finish_capture_sha256':'5bb8f7ccf46655decd5fe8c5fb21f36406f7432206cb44a7f58171bcbec089c6','selected_captures':selected,'registered_source_pins':selected_pins,'selected_registries':selected_registries,'basis':'prospective no-model offline assembly replay binds verified original regions and unchanged original markdown; no historical finish invented','recovery_kind':'pip02-offline-native-assembly/v1'}
 receipt=json.loads(finish.read_bytes());entry=captures.get(receipt.get('capture_sha256'));cap=entry[0] if entry else None;result_path=directory/(key+'.result');result=json.loads(result_path.read_bytes())
 if cap is None or not (re.fullmatch(r'codex3-pip02-paddle-relocatable-r(?:9|10|11)-g\d+-20261029',cap['id']) or re.fullmatch(r'codex3-pip02-paddle-relocatable-r13-g(?:5|8|11|17|26|32|35|65|128)-20261030',cap['id']) or re.fullmatch(r'codex3-pip02-paddle-relocatable-r(?:14|15)-pool[0-3]-20261030',cap['id']) or re.fullmatch(r'codex3-pip02-paddle-relocatable-r(?:22|25|27)-pool[0-3]-20261030',cap['id']) or cap['id']=='codex3-pip02-paddle-relocatable-r30-pool2-20261031' or re.fullmatch(r'(?:codex3-pip02-paddle-relocatable-r(?:26|27)-head-20261030|codex3-pip02-paddle-relocatable-r28-head-20261031|codex3-pip02-paddle-relocatable-r29-head-20261031)',cap['id'])):raise ValueError('finish capture is not registered')
 source_sha=sha(canon(cap['source_pins']))
 if receipt.get('schema')!='pip02-paddle-page-finish/v1' or receipt.get('page')!=page or receipt.get('campaign_sha256')!=campaign_sha or receipt.get('result_sha256')!=sha(result_path.read_bytes()) or receipt.get('markdown_sha256')!=result.get('markdown_sha256') or sha(result['markdown'].encode())!=receipt.get('markdown_sha256') or receipt.get('source_pins_sha256')!=source_sha or receipt.get('grant_sha256')!=cap['grant_sha256']:raise ValueError('native finish receipt mismatch')
 selected[entry[1]]=receipt['capture_sha256'];selected_pins[receipt['capture_sha256']]={**cap['source_pins'],**entry[4]};selected_registries[entry[2]]=entry[3]
 return {'verified':True,'request_origins_verified':True,'origin_capture_sha256':sorted(origins),'finish_capture_sha256':receipt['capture_sha256'],'selected_captures':selected,'registered_source_pins':selected_pins,'selected_registries':selected_registries,'basis':'native registered finish receipt binds page result and markdown to captured grant and registered source'}
