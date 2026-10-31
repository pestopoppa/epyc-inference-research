import sys,json,hashlib,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import tests_native as t
from odl_bench import pip02_lineage as l
O=Path('/mnt/raid0/llm/tmp/codex3-pip-head-assembly-20261031-0530/recovery-1/output')
def pins():
 p=t.pins()
 for name,digest in json.loads((O/'RECOVERY-START.json').read_bytes())['inputs_sha256'].items():p[str(Path(name).resolve())]=digest
 for n in ('RECOVERY-START.json','RECOVERY-FINISH.json','page-0000.recovery.json','page-0034.recovery.json'):p[str((O/n).resolve())]=hashlib.sha256((O/n).read_bytes()).hexdigest()
 p['/mnt/raid0/llm/tmp/codex3-pip-pool2-r30-v2-20261031/launch-p2.sh']='2dbe8265199aabaae505f17520e4a84b2cae67fc7b7f218392d201353061160c'
 return p
def run(pos, p=None):return l.paddle_lineage(t.layers(pos//5),t.plan['pages'][pos],t.campaign,pos//5,p or pins())
class Recovery(unittest.TestCase):
 def test_actual_recovered_pages(self):
  for pos,count in ((0,15),(10,13)):
   v=run(pos);self.assertTrue(v['verified']);self.assertEqual(v['recovery_kind'],'pip02-offline-native-assembly/v1');self.assertEqual(len(v['origin_capture_sha256']),count)
 def test_source_start_finish_page_tampering(self):
  for target in (O/'RECOVERY-START.json',O/'RECOVERY-FINISH.json',O/'page-0000.recovery.json',Path('/mnt/raid0/llm/tmp/codex3-pip-head-assembly-20261031-0530/recovery-1/assemble.py')):
   orig=Path.read_bytes
   def altered(path):
    b=orig(path);return b+b' ' if path==target else b
   with self.subTest(target=str(target)),patch.object(Path,'read_bytes',altered):
    with self.assertRaises((ValueError,KeyError)):run(0)
 def test_unmatched_region_and_changed_layout_refuse(self):
  page=t.plan['pages'][0];d=t.layers(0);prefix='region-'+t.rec.sha(t.rec.canon(page))[:24]+'-';st=next(iter(d.glob(prefix+'*.start.json')));rp=d/(st.name.replace('.start.json','.result.json'));orig=Path.is_file
  def absent(path):return False if path==rp else orig(path)
  with patch.object(Path,'is_file',absent):
   with self.assertRaisesRegex(ValueError,'unknown native request exposure'):run(0)
  layout=d/(prefix+'layout.json');origread=Path.read_bytes
  def altered(path):return origread(path)+b' ' if path==layout else origread(path)
  with patch.object(Path,'read_bytes',altered):
   with self.assertRaises((ValueError,KeyError)):run(0)
if __name__=='__main__':unittest.main(verbosity=2)
