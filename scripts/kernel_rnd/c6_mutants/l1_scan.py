#!/usr/bin/env python3
"""L1 — AST blacklist scan, implemented to the specified import contract
(KernelGenBench L1 per intake-1227, plus TritonRL rule 1 per intake-1241).

Checks known forbidden symbols on the declared candidate computation path.
It cannot prove arbitrary dispatch absence or what the kernel computes — that
is the whole point of RVP-C6-20.

Scan classes:
  delegation   torch.ops.aten.*, torch.matmul / Tensor @ Tensor, torch.softmax,
               torch.nn.functional.* compute calls, torch.layer_norm, nn.Module
               compute layers, torch.compile, torch.einsum, torch.bmm/mm/addmm
  escape       import vllm, ctypes, subprocess, eval/exec on strings, os.system
  laundering   returning the reference / input unchanged is NOT detectable
               statically in general — recorded as out-of-scope, that is L2/judge

Allowed torch surface (allocation & launch plumbing, per the spec): empty_like,
empty, zeros, ones, randn, manual_seed, tensor metadata (.shape/.dtype/.device),
.to/.contiguous, and the reference implementations OUTSIDE candidate functions.

The scanner is itself mutation-tested by test_static.py against planted-dirty
samples — a scanner that flags nothing must fail that test, so it cannot go
vacuous silently.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

FORBIDDEN_CALL_ROOTS = {
    ("torch", "matmul"), ("torch", "softmax"), ("torch", "layer_norm"),
    ("torch", "einsum"), ("torch", "bmm"), ("torch", "mm"), ("torch", "addmm"),
    ("torch", "compile"),
}
FORBIDDEN_ATTR_CHAINS = (
    ("torch", "ops", "aten"),
    ("torch", "nn", "functional"),
)
FORBIDDEN_IMPORTS = {"vllm", "ctypes", "subprocess"}
FORBIDDEN_NAMES = {"eval", "exec"}


def _attr_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _library_dispatch(tree, candidate_functions):
    """Resolve finite Python aliases on the candidate's local computation path.

    This is an AST check, never execution or timing. Reflection, callback
    parameters and dynamic subscription dispatch stay UNKNOWN. It is not a
    whole-program proof about native extensions or arbitrary Python objects.
    """
    by_name, module_functions = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            by_name.setdefault(node.name, []).append(node)

    def collect_module_functions(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_functions.add(id(node))
            return
        if isinstance(node, ast.ClassDef):
            return
        for child in ast.iter_child_nodes(node):
            collect_module_functions(child)

    collect_module_functions(tree)
    # A name shared by definitions, or defined only in a nested scope, cannot
    # safely resolve as a module helper. Never choose the last ast.walk result.
    definitions = {name: nodes[0] for name, nodes in by_name.items()
                   if len(nodes) == 1 and id(nodes[0]) in module_functions}
    aliases = {}
    findings, unknown, visited = [], [], set()

    def resolve(node, names):
        if isinstance(node, ast.Name):
            return names.get(node.id, (node.id,))
        if isinstance(node, ast.Attribute):
            root = resolve(node.value, names)
            return (*root, node.attr) if root else ()
        return ()

    def bind(node, names):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split('.')[0]] = tuple(
                    alias.name.split('.') if alias.asname else [alias.name.split('.')[0]])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names[alias.asname or alias.name] = tuple(
                    (node.module or '').split('.') + [alias.name])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = resolve(node.value, names)
            for target in targets:
                if isinstance(target, ast.Name):
                    names[target.id] = value

    def merge_branches(names, arms):
        for key in arms[0].keys() | arms[1].keys():
            names[key] = (arms[0].get(key, (key,))
                          if arms[0].get(key, (key,)) == arms[1].get(key, (key,))
                          else ())

    def module_bind(nodes, names):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If):
                arms = [dict(names), dict(names)]
                module_bind(node.body, arms[0])
                module_bind(node.orelse, arms[1])
                merge_branches(names, arms)
                continue
            bind(node, names)
            for _field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    module_bind([child for child in value if isinstance(child, ast.stmt)], names)

    module_bind(tree.body, aliases)

    def visit_function(name):
        if name in visited:
            return
        visited.add(name)
        if name not in definitions:
            unknown.append({'line': by_name[name][0].lineno,
                            'class': 'unknown_dispatch',
                            'symbol': f'ambiguous or nested function: {name}'})
            return
        fn = definitions[name]
        names = dict(aliases)
        # Parameters shadow module imports, including an import named torch.
        for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
            names[arg.arg] = ()
        walk(fn.body, names)

    def walk(nodes, names):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If):
                walk([node.test], names)
                arms = [dict(names), dict(names)]
                walk(node.body, arms[0])
                walk(node.orelse, arms[1])
                merge_branches(names, arms)
                continue
            # Calls on RHS use the binding before assignment.
            if isinstance(node, ast.Call):
                chain = resolve(node.func, names)
                symbol = '.'.join(chain)
                vendor = (len(chain) >= 2 and chain[0].lower() in
                          {'rocblas', 'hipblas', 'hipblaslt'} and
                          any(token in chain[-1].lower() for token in
                              ('gemm', 'gemv', 'matmul', 'trsm', 'syrk')))
                torch_compute = (chain in FORBIDDEN_CALL_ROOTS or any(
                    chain[:len(root)] == root for root in FORBIDDEN_ATTR_CHAINS))
                if vendor or torch_compute:
                    findings.append({'line': node.lineno, 'class': 'library_substitution',
                                     'symbol': symbol})
                elif len(chain) == 1 and chain[0] in by_name:
                    visit_function(chain[0])
                elif ((not chain and not (isinstance(node.func, ast.Attribute)
                                          and node.func.attr in {'to', 'contiguous'}))
                      or (chain and chain[0] in {'getattr', '__import__'})):
                    # Triton's indexed launch is a known generated-kernel entry.
                    launch = (isinstance(node.func, ast.Subscript) and
                              isinstance(node.func.value, ast.Name) and
                              node.func.value.id in definitions)
                    if not launch:
                        unknown.append({'line': node.lineno, 'class': 'unknown_dispatch',
                                        'symbol': ast.unparse(node.func)})
            for _field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    walk([child for child in value if isinstance(child, ast.AST)], names)
                elif isinstance(value, ast.AST):
                    walk([value], names)
            bind(node, names)

    if candidate_functions is None:
        walk(tree.body, dict(aliases))
        for name in by_name:
            visit_function(name)
    else:
        for name in candidate_functions:
            if name in by_name:
                visit_function(name)
    return findings, unknown, sorted(visited)


def scan_source(src: str, *, candidate_functions: list[str] | None = None) -> dict:
    """Scan candidate functions (or the whole module if None). Returns a dict
    with per-finding rows; unresolved dynamic dispatch is UNKNOWN, not PASS."""
    tree = ast.parse(src)
    findings = []

    def in_scope(lineno, scopes):
        if candidate_functions is None:
            return True
        return any(a <= lineno <= b for a, b in scopes)

    scopes = []
    if candidate_functions is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name in candidate_functions:
                scopes.append((node.lineno, node.end_lineno))
        if not scopes:
            raise ValueError(f"none of {candidate_functions} found in source")

    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if line is None or not in_scope(line, scopes):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) \
                else [(node.module or "").split(".")[0]]
            for m in mods:
                if m in FORBIDDEN_IMPORTS:
                    findings.append({"line": line, "class": "escape", "symbol": f"import {m}"})
        elif isinstance(node, ast.Attribute):
            chain = _attr_chain(node)
            for bad in FORBIDDEN_ATTR_CHAINS:
                if chain[:len(bad)] == bad:
                    findings.append({"line": line, "class": "delegation",
                                     "symbol": ".".join(chain)})
        elif isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            if len(chain) >= 2 and (chain[0], chain[-1]) in FORBIDDEN_CALL_ROOTS \
                    and chain[0] == "torch" and len(chain) == 2:
                findings.append({"line": line, "class": "delegation",
                                 "symbol": ".".join(chain)})
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                findings.append({"line": line, "class": "escape", "symbol": node.func.id})
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            findings.append({"line": line, "class": "delegation", "symbol": "@ (MatMult)"})

    library, unknown, reachable = _library_dispatch(tree, candidate_functions)
    findings.extend(library)
    # dedupe (an Attribute inside a Call is visited twice)
    seen, uniq = set(), []
    for f in findings:
        k = (f["line"], f["symbol"])
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return {"findings": uniq,
            "verdict": "FAIL" if uniq else ("UNKNOWN" if unknown else "PASS"),
            "scanned_functions": candidate_functions or ["<module>"],
            "library_substitution_findings": library,
            "unknown_dispatch": unknown, "reachable_functions": reachable}


# The candidate computation path in mutants.py: every candidate + launch helper.
# References (layernorm_reference etc.) are deliberately OUT of scope — they are
# the oracle's ground truth and legitimately use torch operators.
CANDIDATE_FUNCTIONS = [
    "_layernorm_launch", "layernorm_candidate_honest", "layernorm_candidate_mutant",
    "_softmax_launch", "softmax_candidate_honest", "softmax_candidate_mutant",
    "_matmul_launch", "matmul_t_candidate_honest", "matmul_t_candidate_mutant",
]


def main():
    src_path = Path(__file__).parent / "mutants.py"
    res = scan_source(src_path.read_text(), candidate_functions=CANDIDATE_FUNCTIONS)
    print(json.dumps({"tier": "L1", "file": str(src_path), **res}, indent=2))
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
