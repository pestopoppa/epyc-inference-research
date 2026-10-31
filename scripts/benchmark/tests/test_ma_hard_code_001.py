"""The authored edge-bounded shortest-path item has a usable stdin oracle."""
from pathlib import Path
import sys

import yaml

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from answer_scoring import score_response, score_response_or_error  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GOOD = '''
import sys
def solve():
    values=list(map(int,sys.stdin.read().split()))
    n,m,k=values[:3]
    edges=[tuple(values[i:i+3]) for i in range(3,len(values),3)]
    distance=[float('inf')]*(n+1)
    distance[1]=0
    for _ in range(k):
        updated=distance[:]
        for u,v,w in edges:
            updated[v]=min(updated[v],distance[u]+w)
        distance=updated
    print(-1 if distance[n]==float('inf') else distance[n])
if __name__ == '__main__':
    solve()
'''


def test_task_prompt_and_native_scorer():
    data=yaml.safe_load((ROOT / 'benchmarks/prompts/debug/mode_advantage_hard.yaml').read_text())
    task=next(t for t in data['questions'] if t['id']=='ma_hard_code_001')
    assert 'Output: 6' in task['prompt'] and 'Output: 3' not in task['prompt']
    assert 'Actually output' not in task['prompt']
    cfg=task['scoring_config']
    assert len(cfg['test_cases'])==3 and 'test_code' not in cfg
    assert score_response(GOOD,task['expected'],task) is True
    assert score_response('def solve():\n    print(6)\nsolve()',task['expected'],task) is False
    assert score_response('def solve():\n    print(3)\nsolve()',task['expected'],task) is False
    assert score_response_or_error(GOOD,task['expected'],task) == (True, None)
