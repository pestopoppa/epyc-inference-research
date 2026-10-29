#!/usr/bin/env python3
"""Common OmniDocBench table evaluation; input is explicit complete/unknown page records."""
import argparse, hashlib, json, sys, importlib.metadata
from pathlib import Path
ROOT=Path('/mnt/raid0/llm/omnidocbench')
sys.path.insert(0,str(ROOT))
from src.dataset.end2end_dataset import End2EndDataset
from src.core.preprocess.extract import md_tex_filter
from src.core.preprocess.data_preprocess import normalized_table
from src.core.matching.match import match_gt2pred_simple
from src.metrics.table_metric import TEDS

def evaluate(page, text):
    gt = End2EndDataset.get_page_elements(None,page).get('table',[])
    gt = End2EndDataset.get_sorted_text_list(None,gt)
    if not gt: return []
    pred=md_tex_filter(text)
    candidates=list(pred['html_table'])
    for item in pred['latex_table']:
        x=dict(item); x['content']=normalized_table(item['content'],'latex'); x['category_type']='html_table'; x['fine_category_type']=x.get('fine_category_type','latex2html_table'); candidates.append(x)
    matches,_=match_gt2pred_simple(gt,candidates,'html_table',page['page_info']['image_path'])
    result=[]
    for m in matches:
        if m['gt_idx']==['']: continue
        g=normalized_table(m['gt']); p=normalized_table(m['pred'],'html')
        result.append({'gt_idx':m['gt_idx'], 'pred_idx':m['pred_idx'], 'teds':TEDS().evaluate(p,g), 'teds_structure_only':TEDS(structure_only=True).evaluate(p,g)})
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',required=True); ap.add_argument('--gt',required=True); ap.add_argument('--records',required=True); ap.add_argument('--pins',required=True); a=ap.parse_args()
    pins=json.load(open(a.pins));
    packages=sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions())
    if sys.version != pins['python'] or packages != pins['packages']: raise ValueError('runtime differs from pinned environment')
    for path,digest in pins['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise ValueError('pin mismatch: '+path)
    if str(Path(a.plan).resolve()) != pins['plan'] or str(Path(a.gt).resolve()) != pins['gt']: raise ValueError('plan or GT path not pinned')
    plan=json.load(open(a.plan)); gt=json.load(open(a.gt)); records=json.load(open(a.records));
    if len({r['image'] for r in records})!=len(records): raise ValueError('duplicate records')
    by={r['image']:r for r in records}; expected={p['image'] for p in plan['pages']}
    if set(by)-expected: raise ValueError('unplanned records')
    output=[]; known=0; total=0; completed_pages=0
    for row in plan['pages']:
        page=gt[row['index']]; assert page['page_info']['image_path']==row['image']; assert page['page_info']['page_attribute']['language']=='english'
        n=len(End2EndDataset.get_page_elements(None,page).get('table',[])); assert n==row['table_count']; total+=n
        rec=by.get(row['image']); status=rec.get('status') if rec else 'unknown'
        if status not in ('complete','unknown'): raise ValueError('invalid status')
        if status=='unknown': output.append({'image':row['image'],'status':'unknown','gt_tables':n,'tables':None}); continue
        if not isinstance(rec.get('text'),str): raise ValueError('complete requires text, possibly empty')
        scored=evaluate(page,rec['text']); assert len(scored)==n
        completed_pages+=1; known+=n; output.append({'image':row['image'],'status':'complete','gt_tables':n,'tables':scored})
    assert len(plan['pages'])==plan['total'] and total==plan['table_regions']
    vals=[x for page in output if page['tables'] is not None for x in page['tables']]
    result={'pages':output,'coverage':{'pages_complete':completed_pages,'pages_total':len(output),'gt_tables_scored':known,'gt_tables_total':total},'full_population_defined':known==total and completed_pages==len(output),'table_weighted_means':{k:sum(x[k] for x in vals)/len(vals) if vals else None for k in ('teds','teds_structure_only')}}
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
