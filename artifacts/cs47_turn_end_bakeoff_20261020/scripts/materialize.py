import json,sys
from pathlib import Path
W=Path("/mnt/raid0/llm/worktrees/cs47-research-20261020")
sys.path.insert(0,str(W/"scripts/voice/turn_taking"))
import tt_audio as A, tt_corpus as C
D=W/"data/turn-taking-cs45-v1"
recs={r["recording_id"]:r for r in map(json.loads,(D/"manifest/recordings.jsonl").read_text().splitlines())}
clips=[json.loads(x) for x in (D/"manifest/clips.jsonl").read_text().splitlines()]
out=Path("/mnt/raid0/llm/tmp/cs-47-20261020/clips"); out.mkdir(exist_ok=True)
cache={}
def rec(r):
    if r not in cache: cache[r]=A.load_audio(D,recs[r]["path"])[0].samples[:,0]
    return cache[r]
bad=0
for c in clips:
    arr,lay=C.assemble_clip(rec(c["user"]["recording_id"]),tuple(c["user"]["trim"]),c["user"]["cut"],rec(c["assistant"]["recording_id"]),c["assistant"]["mode"],c["layout"].get("tail"))
    pc=A.Pcm(arr,A.SR)
    if pc.pcm_sha256()!=c["pcm"]["pcm_sha256"]: bad+=1; print("MISMATCH",c["clip_id"]); continue
    A.write_wav(out/f"{c['clip_id']}.wav",pc)
print("clips",len(clips),"bad",bad)
