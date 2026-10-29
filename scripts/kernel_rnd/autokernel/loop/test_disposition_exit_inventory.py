"""Review fence for new exits; this inventory does not prove artifact custody."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
TARGETS = {'loop.py': {'iterate', '_iterate'}, 'pipeline.py': {'run_pool'}}


def exits(source, names):
    tree = ast.parse(source)
    counts = Counter()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in names:
            for node in ast.walk(fn):
                if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    item = ast.dump(node, annotate_fields=True, include_attributes=False)
                    counts[f'{fn.name}:{type(node).__name__}:{hashlib.sha256(item.encode()).hexdigest()}'] += 1
    return dict(sorted(counts.items()))


def test_new_control_exits_require_disposition_review():
    expected = json.loads((HERE / 'disposition-exit-inventory.json').read_text())
    actual = {name: exits((HERE / name).read_text(), names) for name, names in TARGETS.items()}
    assert actual == expected, 'control exit changed: review artifact disposition and update inventory with behavior test'


def test_inventory_detects_new_unrecorded_return():
    source = (HERE / 'loop.py').read_text()
    changed = source.replace('    def mark_search_changed(decision: str) -> None:',
                             '    if working.get("new_drop"): return None\n\n    def mark_search_changed(decision: str) -> None:', 1)
    assert changed != source
    assert exits(changed, TARGETS['loop.py']) != exits(source, TARGETS['loop.py'])
