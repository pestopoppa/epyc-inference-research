"""Read-only shard-ready hints. Scorer independently verifies every record."""
import json,hashlib
from pathlib import Path
R=Path('/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029');plan=json.loads((R/'english755-fresh-plan.json').read_text());cfg=json.loads(Path('/mnt/raid0/llm/tmp/pip02-pipeline-prep-20261029/paddle-cpu-campaign-config-v3.json').read_text())
canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode();sha=lambda b:hashlib.sha256(b).hexdigest()
campaign={'scope':'pip02-english-fresh-20261029-Paddle-full-4096','population_sha256':'63028e74950ba7250f33cbce13f4f0bdd165bba2a075242db4475c26f645ed68','layout_sha256':{'inference.json':'2b68367c5b312a03de5a6e1642c597c8f95165a7e40cd59c6700cf4a5042f4fd','inference.pdiparams':'70bd316b0582769ec968829fd1feb1a6a58b7c941b938327e551b6b12b45c137','inference.yml':'506fcfac13b3b546ae40d7886b44126420f392adb694e3f8bb6a6286a1f90fdc'},'server_pins':cfg['server']['pins']}
roots=[R/'campaigns-fresh',Path('/mnt/micron/codex3-pip-split-r22-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-relocatable-r25-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-head-r26-20261030/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool2-r30-v2-20261031/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool3-tail-r31-20261031/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool1-r32-20261031/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool0-tail-r33-20261031/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool0-partial-r34-20261031/campaigns-fresh'),Path('/mnt/micron/codex3-pip-pool0-merged-r35-20261031/campaigns-fresh')]
for sh in range(151):
 dirs=[r/f'shard-{sh:02}'/f'shard-{sh}' for r in roots];ready=True
 for row in plan['pages'][sh*5:min(sh*5+5,len(plan['pages']))]:
  key=sha(canon({'campaign':sha(canon(campaign)),'page':row}))
  if any(sum((d/(key+'.'+suffix)).is_file() for d in dirs)!=1 for suffix in ('start','result')):ready=False;break
  finish=sum((d/(key+'.finish.json')).is_file() for d in dirs)
  recovery=Path('/mnt/raid0/llm/tmp/codex3-pip-head-assembly-20261031-0530/recovery-1/output')
  if finish!=1 and not (finish==0 and row['index'] in (0,34) and (recovery/'RECOVERY-FINISH.json').is_file() and (recovery/f"page-{row['index']:04}.recovery.json").is_file()):ready=False;break
 if ready:print(sh)
