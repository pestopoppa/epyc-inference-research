import hashlib,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from odl_bench import pip02_records as r
from odl_bench.pip02_compare import run
from odl_bench.pip02_native_score import evaluate

H=lambda s:hashlib.sha256(s).hexdigest()
HTML='<table><tr><td>A</td><td>B</td></tr><tr><td>x</td><td>y</td></tr></table>'
def dump(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.canon(v))
def fixture(i=0):
 p={'image':f'{i}.png','index':i,'language':'english','sha256':'a'*64,'bytes':1,'table_count':1}
 g={'page_info':{'image_path':p['image'],'page_attribute':{'language':'english'}},'extra':{'relation':[]},'layout_dets':[{'category_type':'table','anno_id':'t','order':1,'html':HTML,'attribute':{}}]};return p,g
class Native(unittest.TestCase):
 def test_scorer(self):
  _,g=fixture(); a=evaluate(g,HTML)[0]; self.assertEqual((a['teds'],a['teds_structure_only']),(1,1))
  b=evaluate(g,'')[0];self.assertEqual((b['teds'],b['teds_structure_only']),(0,0))
  c=evaluate(g,HTML.replace('>x<','>z<'))[0];self.assertLess(c['teds'],1);self.assertEqual(c['teds_structure_only'],1)
  self.assertEqual(evaluate(g,'| A | B |\n| --- | --- |\n| x | y |')[0]['teds'],1)
  g['layout_dets'].append(dict(g['layout_dets'][0],anno_id='t2',order=2,html=HTML.replace('>x<','>z<')))
  self.assertTrue(all(x['teds']==1 for x in evaluate(g,HTML.replace('>x<','>z<')+HTML)))
 def test_records(self):
  page,_=fixture(); campaign={'a':1}
  with tempfile.TemporaryDirectory() as t:
   d=Path(t); self.assertIsNone(r.odl(d/'odl',page,'plan','protocol','dependency')); self.assertIsNone(r.paddle(d/'paddle',page,campaign))
   o=d/'odl';(o/'out').mkdir(parents=True);(o/'input.pdf').write_bytes(b'pdf');(o/'out/input.md').write_text(HTML)
   dump(o/'input-identity.json',{'source_sha256':page['sha256'],'pdf_sha256':H(b'pdf')});dump(o/'start.json',{'page':page,'plan_sha256':'plan','protocol_sha256':'protocol','dependency_manifest_sha256':'dependency','mode':'odl-2.5.0-docling-fast-full','cohort_id':'pip02-english-fresh-20261029','input_sha256':H(b'pdf')});dump(o/'result.json',{'rc':0,'partial':False,'children_drained':True,'prediction_sha256':H(HTML.encode())});dump(o/'DONE',{'result_sha256':r.digest(o/'result.json')})
   self.assertEqual(r.odl(o,page,'plan','protocol','dependency'),HTML);(o/'out/input.md').write_text('tampered')
   with self.assertRaises(ValueError):r.odl(o,page,'plan','protocol','dependency')
   c=r.sha(r.canon(campaign));key=r.sha(r.canon({'campaign':c,'page':page}));s=d/'paddle'/f'{key}.start';z=d/'paddle'/f'{key}.result';dump(s,{'campaign':c,'page':page,'started_monotonic_ns':1});self.assertIsNone(r.paddle(s.parent,page,campaign));dump(z,{'start_sha256':r.digest(s),'markdown':HTML,'markdown_sha256':H(HTML.encode())});self.assertEqual(r.paddle(s.parent,page,campaign),HTML);dump(z,{'start_sha256':r.digest(s),'markdown':'bad','markdown_sha256':H(HTML.encode())})
   with self.assertRaises(ValueError):r.paddle(s.parent,page,campaign)
 def test_unknown_population(self):
  page,g=fixture();plan={'pages':[page],'total':1,'table_regions':1}
  with tempfile.TemporaryDirectory() as t:
   result=run(plan,[g],t,t,{'server':{'pins':{}}})
   for v in result.values():self.assertFalse(v['full_population_defined']);self.assertIsNone(v['table_weighted_means']['teds']);self.assertEqual(v['coverage']['gt_tables_scored'],0)
 def test_full_population_and_partial(self):
  p,g=fixture(); plan={'pages':[p],'total':1,'table_regions':1}; cfg={'server':{'pins':{}}}
  with patch('odl_bench.pip02_records.odl',return_value=HTML),patch('odl_bench.pip02_records.paddle',return_value=HTML),patch('odl_bench.pip02_lineage.odl_lineage',return_value={'verified':True,'basis':'synthetic'}),patch('odl_bench.pip02_lineage.paddle_lineage',return_value={'verified':True,'basis':'synthetic'}):
   values=run(plan,[g],'/none','/none',cfg,pins={})
  self.assertTrue(all(v['full_population_defined'] and v['gate_eligible'] and v['table_weighted_means']['teds']==1 for v in values.values()))
  with patch('odl_bench.pip02_records.odl',return_value=''),patch('odl_bench.pip02_records.paddle',return_value=None):
   values=run(plan,[g],'/none','/none',cfg)
  self.assertEqual(values['odl']['table_weighted_means']['teds'],0);self.assertIsNone(values['paddle']['table_weighted_means']['teds'])
if __name__=='__main__':unittest.main()

class Lineage(unittest.TestCase):
 def test_odl_unique_grant_and_tamper(self):
  import time,datetime
  from odl_bench import pip02_lineage as l
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp); armed=base/'armed'; armed.mkdir(); ident='codex3-pip02-fresh-odl-s0-20261029'; packet=armed/(ident+'.json'); producer=base/'producer';producer.write_text('source'); log=base/'log'; now=time.time(); begin=datetime.datetime.fromtimestamp(now-10,datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'); end=datetime.datetime.fromtimestamp(now+3600,datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'); grant=f'regions=24-47 start={begin} end={end}\n'
   draft={'id':ident,'regions':'24-47','launcher_log':str(log),'launcher':{'path':str(producer),'sha256':r.digest(producer)},'job':{'path':str(producer),'sha256':r.digest(producer)},'source_manifest':{'path':str(producer),'sha256':r.digest(producer)}};dump(packet,draft)
   log.write_text(f'[{ident}] START regions=24-47 end={end}\n --inner --id {ident} {producer} --shard 0 \n');page=base/'page';dump(page/'start.json',{'started':now+1,'grant':grant}); pins={str(packet.resolve()):r.digest(packet),str(producer.resolve()):r.digest(producer)}
   with patch.object(l,'ARMED',armed):
    self.assertTrue(l.odl_lineage(page,0,pins)['verified']);log.write_text(log.read_text()*2)
    with self.assertRaises(ValueError):l.odl_lineage(page,0,pins)
 def test_paddle_unique_capture_and_ambiguity(self):
  import datetime,time,os
  from odl_bench import pip02_lineage as l
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp); page,_=fixture(); campaign={'population_sha256':'p','scope':'scope'};d=base/'journal';d.mkdir();key=r.sha(r.canon({'campaign':r.sha(r.canon(campaign)),'page':page})); finish=d/(key+'.result'); finish.write_text('{}')
   now=time.time();os.utime(finish,(now,now));producer=base/'producer.py';producer.write_text('source'); pins={str(producer.resolve()):r.digest(producer)}
   cap={'started_utc':datetime.datetime.fromtimestamp(now-5,datetime.timezone.utc).isoformat(),'smoke':False,'shard':0,'protocol_sha256':'dc7d8e167edc79a07b9db10849940cb84a7f04c34f67274afbe067531d226ae3','grant':'grant','grant_sha256':H(b'grant'),'source_pins':{'producer.py':r.digest(producer)}}
   result={'population_sha256':'p','scope':'scope','states':{'0':'complete'}}
   capture=base/'attempts/a/capture.json';dump(capture,cap);dump(base/'attempts/a/result.json',result)
   prefix='region-'+r.sha(r.canon(page))[:24]+'-000000'; start=d/(prefix+'.start.json'); end=d/(prefix+'.result.json'); body=b'input'; response=b'output'
   dump(start,{'body_hex':body.hex(),'body_sha256':H(body),'provenance':{'identity':{'campaign_sha256':r.sha(r.canon(campaign))},'origin_capture_sha256':r.digest(capture)}})
   dump(end,{'request_sha256':H(body),'start_sha256':r.digest(start),'body_hex':response.hex(),'response_sha256':H(response),'status':200})
   with patch.object(l,'BASE',base):
    self.assertTrue(l.paddle_lineage(d,page,campaign,0,pins)['verified'])
    dump(base/'attempts/b/capture.json',cap);dump(base/'attempts/b/result.json',result)
    with self.assertRaises(ValueError):l.paddle_lineage(d,page,campaign,0,pins)
