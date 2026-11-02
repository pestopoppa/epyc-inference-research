#!/usr/bin/env python3
"""Select first witnessed four-sequence KPF50 step; refuse mismatched pairs."""
import argparse
import json
import re
from pathlib import Path

PHASE=re.compile(r'KPF30_PHASE call=(\d+) index=(\d+) seq=(-?\d+) pos=(-?\d+) phase=(\w+)')
INPUT=re.compile(r'KPF50_INPUT call=(\d+) index=(\d+) token=(-?\d+) pos=(-?\d+)')
BEGIN=re.compile(r'KPF50_STEP_BEGIN call=(\d+) host_ns=(\d+)')
END=re.compile(r'KPF50_STEP_END call=(\d+) host_ns=(\d+) rc=(-?\d+)')
ITER_BEGIN=re.compile(r'KPF50_ITER_BEGIN iter=(\d+) host_ns=(\d+) draft_context=(\d)')
DRAFT_BEGIN=re.compile(r'KPF50_DRAFT_BEGIN host_ns=(\d+) slots=(\d+)')
DRAFT_END=re.compile(r'KPF50_DRAFT_END host_ns=(\d+) slots=(\d+)')
ITER_END=re.compile(r'KPF50_ITER_END iter=(\d+) host_ns=(\d+)')
MASK=re.compile(r'KPF50_TRUNK_MASK verified=(\d) masked=(\d) rows=(\d+) n_kv=(\d+) expected=(\d+) layout_fnv=([0-9a-f]+)')


def first_step(lines, expected_mask):
    calls={}; active=None; iteration=None; selected=None
    def get(i): return calls.setdefault(i, {'phases':{},'inputs':{},'masks':[]})
    for line in lines:
        if m:=ITER_BEGIN.search(line):
            if iteration is not None: raise ValueError('nested iteration')
            iteration={'id':int(m[1]),'start_ns':int(m[2]),'draft_context':int(m[3])}
        if m:=DRAFT_BEGIN.search(line):
            if iteration is None or 'draft_start_ns' in iteration: raise ValueError('unmatched draft begin')
            iteration['draft_start_ns']=int(m[1]); iteration['draft_slots']=int(m[2])
        if m:=DRAFT_END.search(line):
            if iteration is None or 'draft_start_ns' not in iteration or int(m[2]) != iteration['draft_slots']:
                raise ValueError('unmatched draft end')
            iteration['draft_end_ns']=int(m[1])
        if m:=ITER_END.search(line):
            if iteration is None or iteration['id'] != int(m[1]): raise ValueError('unmatched iteration end')
            iteration['end_ns']=int(m[2])
            if iteration['end_ns'] <= iteration['start_ns']: raise ValueError('invalid iteration')
            if selected is not None:
                if not iteration['draft_context'] or iteration.get('draft_slots',0) <= 0: raise ValueError('missing active draft context')
                if not (iteration['start_ns'] <= iteration.get('draft_start_ns',-1) <= iteration.get('draft_end_ns',-1) <= iteration['end_ns']):
                    raise ValueError('draft span missing or outside iteration')
                if not (iteration['start_ns'] <= selected['start_ns'] < selected['end_ns'] <= iteration['end_ns']):
                    raise ValueError('step outside iteration')
                selected['iteration']=iteration
                return selected
            iteration=None
        if m:=PHASE.search(line):
            call,i,seq,pos,phase=m.groups(); get(int(call))['phases'][int(i)]=(int(seq),int(pos),phase)
        if m:=INPUT.search(line):
            call,i,tok,pos=m.groups(); get(int(call))['inputs'][int(i)]=(int(tok),int(pos))
        if m:=BEGIN.search(line):
            call,start=map(int,m.groups())
            if active is not None: raise ValueError('nested step')
            active=call; get(call)['start_ns']=start
        if m:=MASK.search(line):
            if active is None: raise ValueError('mask outside step')
            get(active)['masks'].append((int(m[1]),int(m[2]),int(m[3]),int(m[4]),int(m[5]),m[6]))
        if m:=END.search(line):
            call,end,rc=map(int,m.groups())
            if active != call: raise ValueError('unmatched end')
            s=get(call);active=None
            if not s['masks'] or selected is not None: continue
            if rc != 0 or end <= s['start_ns']: raise ValueError('invalid candidate step')
            if any(v!=1 or flag!=expected_mask or exp!=60000 for v,flag,n,kv,exp,sig in s['masks']):
                raise ValueError('mask witness failed')
            if not s['phases'] or s['phases'].keys()!=s['inputs'].keys(): raise ValueError('missing phase or input')
            indexes=sorted(s['phases'])
            if indexes != list(range(len(indexes))): raise ValueError('noncontiguous inputs')
            if any(s['inputs'][i][0] < 0 or s['inputs'][i][1] != s['phases'][i][1] for i in indexes):
                raise ValueError('missing token or input position mismatch')
            if len({s['phases'][i][0] for i in indexes}) != 4: raise ValueError('not four sequences')
            phases={s['phases'][i][2] for i in indexes}
            if not phases <= {'decode','verify'}: raise ValueError('prefill or unknown phase')
            if any(n!=len(indexes) for _,_,n,_,_,_ in s['masks']): raise ValueError('mask row count mismatch')
            if iteration is None: raise ValueError('step has no enclosing iteration')
            selected={'call':call,'start_ns':s['start_ns'],'end_ns':end, 'phase':next(iter(phases)) if len(phases)==1 else 'mixed_decode_verify',
                    'inputs':[(s['phases'][i][0],*s['inputs'][i],s['phases'][i][2]) for i in indexes],
                    'masks':[(n,kv,exp,sig) for _,_,n,kv,exp,sig in s['masks']]}
    raise ValueError('no complete eligible step')


def compare(full, masked):
    for key in ('phase','inputs','masks'):
        if full[key]!=masked[key]: raise ValueError('paired '+key+' mismatch')
    if full['iteration']['draft_slots'] != masked['iteration']['draft_slots']:
        raise ValueError('paired draft slot count mismatch')
    return {'scope':'first qualifying step only','full':full,'masked':masked,
            'kv_value_identity':'unproved','gate_verdict':'UNRESOLVED'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full-log',required=True);ap.add_argument('--mask-log',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    try:
        x=first_step(Path(a.full_log).read_text().splitlines(),0)
        y=first_step(Path(a.mask_log).read_text().splitlines(),1)
        Path(a.out).write_text(json.dumps(compare(x,y),indent=2)+'\n')
    except (ValueError,OSError) as e: ap.exit(2, f'KPF50 step pair refused: {e}\n')
if __name__=='__main__': main()
