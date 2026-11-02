"""Model-free checks for the target build completion observation."""
import ast
import datetime
from pathlib import Path
from types import SimpleNamespace
import time


def _target_build(passed):
    tree = ast.parse((Path(__file__).parent / 'run.py').read_text())
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'target_build']
    assert len(funcs) == 1
    stamp = [None]
    events = []
    def compile_stub(*args, **kwargs):
        events.append('build-return')
        return SimpleNamespace(passed=passed)
    def gmtime_stub():
        events.append('clock')
        return time.gmtime(0)
    scope = {'local_compiles': compile_stub, 'checked_source': object(),
             'validation_candidate_build': object(), 'recipe': SimpleNamespace(cmake_defines=lambda: ()),
             'build_jobs': 1, 'build_cpu_list': '0',
             'gates': SimpleNamespace(PROMOTION_TARGETS=()), 'build_finished_at': stamp,
             'time': SimpleNamespace(strftime=time.strftime, gmtime=gmtime_stub)}
    module = ast.fix_missing_locations(ast.Module(body=[funcs[0]], type_ignores=[]))
    exec(compile(module, 'run.py', 'exec'), scope)
    assert scope['target_build']().passed is passed
    return stamp[0], events


def test_success_records_after_build_before_later_gates():
    stamp, events = _target_build(True)
    assert events == ['build-return', 'clock']
    assert datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ').replace(
        tzinfo=datetime.timezone.utc).timestamp() == 0


def test_failed_build_has_no_completion_timestamp():
    stamp, events = _target_build(False)
    assert stamp is None
    assert events == ['build-return']


def test_reused_build_has_unknown_built_at():
    tree = ast.parse((Path(__file__).parent / 'run.py').read_text())
    fields = [(k, v) for node in ast.walk(tree) if isinstance(node, ast.Dict)
              for k, v in zip(node.keys, node.values)
              if isinstance(k, ast.Constant) and k.value == 'built_at']
    assert len(fields) == 1
    expr = ast.fix_missing_locations(ast.Expression(fields[0][1]))
    code = compile(expr, 'run.py', 'eval')
    assert eval(code, {'build_finished_at': ['2026-11-02T00:00:00Z'],
                       'build_dir_was_absent': False}) is None
    assert eval(code, {'build_finished_at': ['2026-11-02T00:00:00Z'],
                       'build_dir_was_absent': True}) == '2026-11-02T00:00:00Z'
