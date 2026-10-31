"""Native artifacts retain each arm's observed identity through sealing."""
import json
import sys
from unittest.mock import patch

import emit_operator_gate_bundle as emitter


def test_real_artifacts_through_gates_and_bundle(tmp_path):
    anchor_path, cells_path, parity_path, out = [tmp_path / f'{name}.json' for name in ('anchor','cells','parity','out')]
    recipe = {'anchor': ['a'], 'champion': ['c']}
    anchor_path.write_text(json.dumps({'results': [
        {'surface':'prefill','model':'M','recipe':recipe,'anchor_median':10,'champion_median':11,'delta_pct':10},
        {'surface':'decode','anchor_median':5,'champion_median':5,'delta_pct':0}]}))
    records = [
        {'arm':'mtp','concurrency':1,'kv_unified':False,'aggregate_decode_tok_s':10,'model':'M','workload':'W','recipe':{'argv':['base']}},
        {'arm':'dflash2','concurrency':1,'kv_unified':False,'aggregate_decode_tok_s':12,'model':'M','workload':'W','recipe':{'argv':['candidate']}},
        {'arm':'mtp','concurrency':2,'kv_unified':False,'aggregate_decode_tok_s':10},
        {'arm':'dflash2','concurrency':2,'kv_unified':False,'aggregate_decode_tok_s':11}]
    cells_path.write_text(json.dumps(records)); parity_path.write_text(json.dumps({'arms':{}}))
    anchor = emitter._anchor_validation(anchor_path)
    points = emitter._concurrency_gate(cells_path)['points']
    assert anchor['surfaces'][0]['recipe'] == recipe and anchor['surfaces'][0]['model'] == 'M'
    assert anchor['surfaces'][1]['recipe'] is None and anchor['surfaces'][1]['model'] is None
    assert points[0]['recipe'] == {'anchor': {'argv':['base']}, 'champion': {'argv':['candidate']}}
    assert points[0]['model'] == 'M' and points[0]['workload'] == 'W'
    assert points[1]['recipe'] is None and points[1]['model'] is None and points[1]['workload'] is None
    argv=['emit','--out',str(out),'--anchor-artifact',str(anchor_path),'--concurrency-artifact',str(cells_path),'--parity-artifact',str(parity_path)]
    with patch.object(sys, 'argv', argv), patch.object(emitter, '_git', return_value='a'*40):
        assert emitter.main() == 0
    bundle=json.loads(out.read_text()); vector=bundle['workload_vector']
    assert len(vector) == 4 and vector[0]['recipe'] == recipe
    assert vector[2]['recipe'] == points[0]['recipe'] and vector[2]['workload'] == 'W'
    assert vector[3]['recipe'] is None and vector[3]['workload'] is None
    assert all(v['scope']=='workload_cell' and v['whole_champion'] is False for v in vector)
    assert bundle['headline']['scope']=='reference_cell_only' and bundle['headline']['whole_champion'] is False
    assert bundle['promotion_claim'] is False


def test_mismatched_arm_identities_remain_separate(tmp_path):
    p=tmp_path/'cells.json'; p.write_text(json.dumps([
      {'arm':'mtp','concurrency':1,'kv_unified':False,'aggregate_decode_tok_s':10,'model':'A','workload':'W1','recipe':{'argv':['a']}},
      {'arm':'dflash2','concurrency':1,'kv_unified':False,'aggregate_decode_tok_s':11,'model':'B','workload':'W2'}]))
    point=emitter._concurrency_gate(p)['points'][0]
    assert point['model'] is None and point['workload'] is None
    assert point['models']=={'anchor':'A','champion':'B'}
    assert point['workloads']=={'anchor':'W1','champion':'W2'}
    assert point['recipe']=={'anchor':{'argv':['a']},'champion':None}


def test_historical_boundary_report_uses_cells_without_inventing_recipe():
    import boundary_20260831_helper as boundary
    cells = boundary._workload_cells({'gates': [{'gate': 'anchor', 'surfaces': [
        {'surface': 'pp512', 'delta_pct': 1}]}, {'gate': 'spec', 'points': [
        {'in_flight': 2, 'delta_pct': 9}]}]})
    assert len(cells) == 2 and cells[0]['workload'] == 'pp512'
    assert cells[0]['recipe'] is None and cells[1]['workload'] is None
    assert cells[1]['recipe'] is None and cells[1]['in_flight'] == 2


def test_grid_captures_exact_commands_before_request(tmp_path, monkeypatch):
    import g2_df25_concurrency_grid as grid
    from types import SimpleNamespace
    seen={}
    class FakeProcess:
        def poll(self): return None
    monkeypatch.setattr(grid.subprocess,'Popen',lambda argv, **kw: (seen.update(server=list(argv)) or FakeProcess()))
    monkeypatch.setattr(grid,'wait_for_ready',lambda *args,**kwargs: None)
    monkeypatch.setattr(grid,'stop_server',lambda *args: None)
    monkeypatch.setattr(grid,'read_vram',lambda: grid.VRAM_RESIDENT_FLOOR)
    def fake_run(argv, **kwargs):
        record=json.loads((tmp_path/'mtp_c1_kvu0'/'cell_recipe.json').read_text())
        assert record['recipe']['server_argv']==seen['server']
        assert record['recipe']['runner_argv']==argv
        seen['before_request']=True
        out=argv[argv.index('--output')+1]
        from pathlib import Path
        Path(out).write_text(json.dumps({'suites':[{'throughput':{'aggregate_decode_tok_s':5}}]}))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(grid.subprocess,'run',fake_run)
    cell=grid.run_cell(tmp_path/'build',tmp_path,'mtp',1,False,'run',128,12)
    assert seen['before_request'] and cell['recipe']['server_argv']==seen['server']
    assert cell['model']==seen['server'][seen['server'].index('-m')+1]
    assert cell['workload']==cell['recipe']['runner_argv'][cell['recipe']['runner_argv'].index('--suites')+1]


def test_anchor_records_both_actual_argv_before_bench(tmp_path, monkeypatch):
    import champion_anchor_validation as anchor
    from types import SimpleNamespace
    baseline=tmp_path/'baseline'; candidate=tmp_path/'candidate'; out=tmp_path/'out'
    baseline.mkdir(); candidate.mkdir()
    (baseline/'llama-bench').touch(); (candidate/'llama-bench').touch()
    anchor.BENCH_INVOCATIONS.clear()
    def fake_run(argv, env, timeout, label):
        assert anchor.BENCH_INVOCATIONS[-1]['argv']==argv
        pp=int(argv[argv.index('-p')+1]); tg=int(argv[argv.index('-n')+1])
        return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps([{'n_prompt':pp,'n_gen':tg,'avg_ts':10.0}]))
    monkeypatch.setattr(anchor,'run_pinned',fake_run)
    monkeypatch.setattr(anchor,'read_vram',lambda: 1)
    monkeypatch.setattr(sys,'argv',['anchor','--anchor-bin',str(baseline),'--champion-bin',str(candidate),'--out',str(out),'--reps','1'])
    assert anchor.main()==0
    results=json.loads((out/'champion_anchor_validation.json').read_text())['results']
    assert len(results)==2
    for row in results:
        assert row['recipe']['anchor'][0]['build_bin']==str(baseline)
        assert row['recipe']['champion'][0]['build_bin']==str(candidate)
        argv=row['recipe']['anchor'][0]['argv']
        assert row['model']==argv[argv.index('-m')+1]
