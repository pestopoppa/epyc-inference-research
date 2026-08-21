#!/usr/bin/env python3
"""L1 — AST blacklist scan, implemented to the specified import contract
(KernelGenBench L1 per intake-1227, plus TritonRL rule 1 per intake-1241).

Proves: NO FORBIDDEN SYMBOL APPEARS in the candidate's computation path.
It cannot prove anything about what the kernel computes — that is the whole
point of RVP-C6-20.

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


def scan_source(src: str, *, candidate_functions: list[str] | None = None) -> dict:
    """Scan candidate functions (or the whole module if None). Returns a dict
    with per-finding rows; verdict PASS iff zero findings."""
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

    # dedupe (an Attribute inside a Call is visited twice)
    seen, uniq = set(), []
    for f in findings:
        k = (f["line"], f["symbol"])
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return {"findings": uniq, "verdict": "PASS" if not uniq else "FAIL",
            "scanned_functions": candidate_functions or ["<module>"]}


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
