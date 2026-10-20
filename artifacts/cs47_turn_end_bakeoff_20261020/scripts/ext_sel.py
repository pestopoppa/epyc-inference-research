import json,glob,numpy as np
from pathlib import Path
D=Path("/mnt/raid0/llm/worktrees/cs47-research-20261020/data/turn-taking-cs45-v1/manifest")
clips={c["clip_id"]:c for c in map(json.loads,(D/"clips.jsonl").read_text().splitlines())}
st={}
for fn in glob.glob("/mnt/raid0/llm/tmp/cs-47-20261020/out/st_*.jsonl"):
    for l in open(fn): r=json.loads(l); st[r["clip"]]=r["calls"]
print("SELECTION ONLY, extended Smart Turn theta grid (no confirmation touched): gate theta recall EOT-FPR INT-FPR n")
for g in (0.2,0.4,0.6,0.8,1.0):
  for th in (0.95,0.98,0.99,0.995,0.999,0.9999):
    rc=[];fp=[];ip=[]
    for cid,c in clips.items():
        if c["split"]!="selection": continue
        gd=c["gold"]; fs=sorted({r["t_fire_s"] for r in st[cid] if abs(r["gate_s"]-g)<1e-9 and r["prob"]>=th and r["t_fire_s"]>=gd["user_speech_start_s"]})
        if c["label"]=="complete":
            eg=gd["eot_gold_s"]; rc.append(bool([x for x in fs if x>=eg]) and not [x for x in fs if x<eg])
        elif c["label"]=="incomplete": fp.append(bool(fs))
        else: ip.append(bool(fs))
    print(g,th,round(np.mean(rc),3),round(np.mean(fp),3),round(np.mean(ip),3),len(rc),len(fp),len(ip))
