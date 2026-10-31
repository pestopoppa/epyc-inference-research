import json
import os
from pathlib import Path
import subprocess
import sys
OWNER=Path('/mnt/raid0/llm/worktrees/mains/refactor-research-code-01a16fc3')
PACKET=Path(__file__).parent
runner=OWNER/'artifacts/engineering/ak-exact-flag-20261031/run_retained.py'
cases=[('relative','relative-not-allowed','fresh absolute'),
       ('existing',str(PACKET/'repository-green-retained-r1'),'already exists'),
       ('git',str(OWNER/'artifacts/engineering/ak-exact-flag-20261031/rejected-retain-root-r1'),'outside every Git')]
results=[]
for name,root,expected in cases:
    env=dict(os.environ,AK_TEST_OWNER=str(OWNER),AK_TEST_RETAIN_ROOT=root)
    env.pop('AK_TEST_MUTATE_STARTUP',None)
    done=subprocess.run([sys.executable,str(runner)],env=env,capture_output=True,text=True,timeout=30)
    assert done.returncode!=0 and expected in done.stderr,(name,done.returncode,done.stdout,done.stderr)
    if name=='git':assert not Path(root).exists()
    results.append({'case':name,'exit':done.returncode,'stderr':done.stderr.strip(),'rejected_before_fixture':True})
print(json.dumps(results,indent=2))
