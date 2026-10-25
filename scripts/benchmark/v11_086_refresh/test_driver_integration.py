"""Exercise actual proposed cpu_bench_launch with synthetic completed child output."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

base = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('bench_protocol_086', base / 'bench_protocol_086.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
tree = ast.parse((base / 'v11_driver_086v2.py').read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('cpu_bench_launch', 'parse_bench', 'cell_name')]
assert len(functions) == 3
raw = ''
class Process:
    pid = 123456
    returncode = 0
    def __init__(self, argv, env, stdout, stderr): stdout.write(raw); stdout.flush()
    def wait(self, timeout): return 0
class Proof:
    def __init__(self, *args): pass
    def stop(self): return {'own': True}
class Ctx:
    def __init__(self, d): self.dir = d; self.log = None; self.messages = []
    def time_left(self): return 10000
    def register(self, pid): pass
    def say(self, msg): self.messages.append(msg)
ns = dict(json=json, Path=Path, Q38_PREFIX=[], Q38_TRUNK=Path('model'), Q38_BENCH_ARGS=['-n', '32'], Q38_GGML_ENV={},
          lib_env=lambda *args: {}, subprocess=types.SimpleNamespace(Popen=Process, TimeoutExpired=TimeoutError),
          OwnLibProof=Proof, BenchProtocolError=m.BenchProtocolError, validate_bench_protocol=m.validate_bench_protocol)
exec(compile(ast.Module(body=functions, type_ignores=[]), 'actual-driver-functions', 'exec'), ns)
with tempfile.TemporaryDirectory() as tmp:
    ctx = Ctx(Path(tmp))
    def run(rows):
        global raw
        raw = json.dumps(rows)
        return ns['cpu_bench_launch'](ctx, 'fixture', Path('/synthetic/build'))
    cell = {'n_prompt': 0, 'n_gen': 32, 'avg_ts': 10, 'autokernel_hardened': False}
    accepted = run([cell])
    assert len(accepted['rows']) == 1 and accepted['protocol']['autokernel_hardened'] is False
    for bad in ([cell | {'autokernel_hardened': True}], [cell, cell | {'autokernel_hardened': True}], [cell | {'autokernel_hardened': 0}], [{k: v for k, v in cell.items() if k != 'autokernel_hardened'}]):
        result = run(bad)
        assert 'error' in result and 'rows' not in result, result
    ns['Q38_BENCH_ARGS'] += ['--autokernel-harden', '1']
    assert 'rows' in run([cell | {'autokernel_hardened': True}])
    assert 'error' in run([cell])
print('PASS actual cpu_bench_launch: ordinary acceptance; unrequested/mixed/missing/nonboolean rejected before aggregation; explicit request semantics')
