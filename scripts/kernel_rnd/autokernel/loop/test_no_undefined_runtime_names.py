"""Every name the autokernel package READS at runtime must be bound somewhere in its module.

2026-10-07: run.longctx_keep_gate called isinstance(measured_row, Mapping) without importing
Mapping. `from __future__ import annotations` hid the missing import in every annotation, and no
test drove measured_row, so the NameError only surfaced in the live Q38FN long-context lane, which
lost the lane twice. This is a cheap module-level check (an over-approximation of the bound names, so it
never false-alarms on locals) that catches that whole class before a lane does.
"""
import ast
import builtins
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__spec__", "__package__", "__path__"}


def runtime_undefined_names(source: str) -> list:
    tree = ast.parse(source)
    bound = set(dir(builtins)) | MODULE_DUNDERS
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.MatchAs) and node.name:
            bound.add(node.name)
    annotation_nodes = set()
    for node in ast.walk(tree):
        for field in ("annotation", "returns"):
            sub = getattr(node, field, None)
            if isinstance(sub, ast.AST):
                annotation_nodes.update(id(m) for m in ast.walk(sub))
    return sorted({(n.lineno, n.id) for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                   and id(n) not in annotation_nodes and n.id not in bound})


class NoUndefinedRuntimeNames(unittest.TestCase):
    def test_the_detector_catches_an_unimported_isinstance_target(self):
        src = "from __future__ import annotations\ndef f(x: Mapping):\n    return isinstance(x, Mapping)\n"
        self.assertEqual(runtime_undefined_names(src), [(3, "Mapping")])

    def test_every_autokernel_module_binds_what_it_reads(self):
        offenders = []
        for path in sorted(PACKAGE.rglob("*.py")):
            if path.name.startswith("test_"):
                continue
            for line, name in runtime_undefined_names(path.read_text()):
                offenders.append(f"{path.relative_to(PACKAGE)}:{line}: {name}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
