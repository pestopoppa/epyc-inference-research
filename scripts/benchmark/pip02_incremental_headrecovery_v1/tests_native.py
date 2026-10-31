import sys,json,hashlib,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));from odl_bench import pip02_lineage as lin,pip02_records as rec
H=Path('/mnt/micron/codex3-pip-head-r26-20261030');base=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029');plan=json.loads((base/'english755-fresh-plan.json').read_bytes());capture=next((H/'attempts').glob('codex3-pip02-paddle-relocatable-r28-head-20261031-600ca45e42ff4763ba0dd5a27f2f690b/capture.json'));campaign=json.loads((capture.parent/'worker-config.json').read_bytes())['campaign'];r29=Path('/mnt/raid0/llm/tmp/codex3-pip-head-r29-repair-20261031');lp='/mnt/raid0/llm/tmp/codex3-filing-pip-dtap-c6-20261031-0230/repair/head/launch-p0.sh';pp=lp.replace('launch-p0.sh','pool.py')
def pins():
 d={}
 for root,h in lin.PADDLE_ROOTS.values():
  p=root/'source-pins.json';d[str(p.resolve())]=h;d.update({str(Path(k).resolve()):v for k,v in json.loads(p.read_bytes()).items()})
 d[lp]='ae394dd5e671ff82ef472218c527675810f0d586b2a8ae6a68c3133a9e8e6e2f';d[pp]='71eee5884d5c5bc92f195ebfad7291eb3deb5adb462d0f98bd6b1604e4161828';d[str(r29/'source-pins.json')]='63f15f638c88e3de96c584f621a71fb3ac778855b58cd06bef27bb64eaf2cec3';d.update(json.loads((r29/'source-pins.json').read_bytes()));d[str(r29/'launch-p0.sh')]='dbe28a4f43d0890a14a80037a80ebc2b52d49ab8ac3fee13b960f62e1b20fbbd';mf=base/'PADDLE-FRESH-SOURCES.sha256';d[str(mf)]='8c8508862d3e9be31ce184da53f5e977eb3935f80f9d23856eb3a845a9175ddb';d.update({n:h for h,n in (line.split('  ',1) for line in mf.read_text().splitlines())});return d
def layers(sh):
 roots=[base/'campaigns-fresh',Path('/mnt/micron/codex3-pip-split-r22-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-relocatable-r25-20261030/campaigns-fresh'),H/'campaigns-fresh'];return rec.LayeredPaths([p/f'shard-{sh:02}'/f'shard-{sh}' for p in roots])
class NativeCatalog(unittest.TestCase):
 def test_actual_r28_and_fresh_positive(self):
  for pos in (5,6):
   x=lin.paddle_lineage(layers(1),plan['pages'][pos],campaign,1,pins());self.assertTrue(x['verified']);self.assertTrue(x['request_origins_verified'])
 def test_actual_r29_region_and_bad_catalog_refuse(self):
  page=plan['pages'][15];d=layers(3);prefix='region-'+rec.sha(rec.canon(page))[:24]+'-';r29_hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in (H/'attempts').glob('codex3-pip02-paddle-relocatable-r29-head-20261031-*/capture.json')};chosen=None
  for item in d.glob(prefix+'*.start.json'):
   a=json.loads(item.read_bytes());result=d/(item.name.replace('.start.json','.result.json'))
   if a['provenance']['origin_capture_sha256'] in r29_hashes and result.is_file():chosen=item;break
  self.assertIsNotNone(chosen,'actual completed R29 region required')
  class One(rec.LayeredPaths):
   def __init__(self):pass
   def glob(self,pattern):return [chosen]
   def __truediv__(self,name):
    if name.endswith('.finish.json'):return R/'nonexistent-finish-marker'
    return d/name
  lineage=lin.paddle_lineage(One(),page,campaign,3,pins());self.assertFalse(lineage['verified']);self.assertTrue(lineage['request_origins_verified']);self.assertTrue(set(lineage['origin_capture_sha256']) & r29_hashes)
  captured_sha=json.loads(chosen.read_bytes())['provenance']['origin_capture_sha256'];target=next(x for x in (H/'attempts').glob('codex3-pip02-paddle-relocatable-r29-head-20261031-*/capture.json') if hashlib.sha256(x.read_bytes()).hexdigest()==captured_sha);original=Path.read_bytes;badcap=json.loads(original(target));badcap['shards']=[3,5]
  def patched(path):return json.dumps(badcap,sort_keys=True).encode() if path==target else original(path)
  with patch.object(Path,'read_bytes',patched):
   with self.assertRaisesRegex(ValueError,'head captured shard group mismatch'):lin.paddle_lineage(One(),page,campaign,3,pins())
  for name in (str(r29/'source-pins.json'),str(r29/'launch-p0.sh')):
   bad=pins();bad[name]='0'*64
   with self.assertRaises(ValueError):lin.paddle_lineage(One(),page,campaign,3,bad)
 def test_fresh_manifest_and_source_map_refuse(self):
  name=str(base/'PADDLE-FRESH-SOURCES.sha256');bad=pins();bad[name]='0'*64
  with self.assertRaises(ValueError):lin.paddle_lineage(layers(1),plan['pages'][5],campaign,1,bad)
  fresh=next((base/'attempts').glob('codex3-pip02-fresh-s1-20261029-*/capture.json'));original=Path.read_bytes;obj=json.loads(original(fresh));obj['source_pins']['NEW-COHORT.md']='0'*64
  def patched(path):return json.dumps(obj,sort_keys=True).encode() if path==fresh else original(path)
  with patch.object(Path,'read_bytes',patched):
   with self.assertRaises(ValueError):lin.paddle_lineage(layers(1),plan['pages'][5],campaign,1,pins())
 def test_wrong_launcher_and_pool_refuse(self):
  for name in (lp,pp):
   d=pins();d[name]='0'*64
   with self.assertRaises(ValueError):lin.paddle_lineage(layers(1),plan['pages'][6],campaign,1,d)
 def test_changed_capture_id_or_group_or_launcher_refuse(self):
  original=Path.read_bytes;raw=json.loads(original(capture));variants=[dict(raw,id='codex3-pip02-paddle-relocatable-r99-head-20261031'),dict(raw,shards=[1,4]),dict(raw,launcher_sha256='0'*64)]
  for variant in variants:
   def patched(path):return json.dumps(variant,sort_keys=True).encode() if path==capture else original(path)
   with patch.object(Path,'read_bytes',patched):
    with self.assertRaises(ValueError):lin.paddle_lineage(layers(1),plan['pages'][6],campaign,1,pins())
 def test_wrong_region_capture_hash_refuses(self):
  d=layers(1);page=plan['pages'][6];prefix='region-'+rec.sha(rec.canon(page))[:24]+'-';target=next(p for p in d.glob(prefix+'*.start.json') if json.loads(p.read_bytes())['provenance']['origin_capture_sha256']==hashlib.sha256(capture.read_bytes()).hexdigest());original=Path.read_bytes;v=json.loads(original(target));v['provenance']['origin_capture_sha256']='0'*64
  def patched(path):return json.dumps(v,sort_keys=True).encode() if path==target else original(path)
  with patch.object(Path,'read_bytes',patched):
   with self.assertRaises(ValueError):lin.paddle_lineage(d,page,campaign,1,pins())
if __name__=='__main__':unittest.main(verbosity=2)
