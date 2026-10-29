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
  with patch('odl_bench.pip02_records.odl',return_value=HTML),patch('odl_bench.pip02_records.paddle',return_value=HTML):
   values=run(plan,[g],'/none','/none',cfg)
  self.assertTrue(all(v['full_population_defined'] and v['table_weighted_means']['teds']==1 for v in values.values()))
  with patch('odl_bench.pip02_records.odl',return_value=''),patch('odl_bench.pip02_records.paddle',return_value=None):
   values=run(plan,[g],'/none','/none',cfg)
  self.assertEqual(values['odl']['table_weighted_means']['teds'],0);self.assertIsNone(values['paddle']['table_weighted_means']['teds'])
if __name__=='__main__':unittest.main()
