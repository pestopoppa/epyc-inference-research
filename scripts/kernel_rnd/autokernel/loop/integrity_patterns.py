"""Pure screening patterns for integrity.py's special-case screen, split verbatim from integrity.py (code-budget split)."""
from __future__ import annotations

import re

_BRANCH = re.compile(r"\b(?:if|else\s+if|switch|while)\s*\(|\?")
# ggml uses both indexed dimensions (`src->ne[0]`) and flattened locals
# (`ne00`, `ne01`, `ne10`, `ne11`).  Keep the subject and literal tests separate:
# doing so catches both `ne00 == 4096` and its equally suspicious reversed spelling
# `4096 == ne00`.  The old one-direction regex silently missed the latter and all
# two-digit flattened names.
_TENSOR_SUBJECT = re.compile(
    r"(?:\bne(?:\d{1,2})?\b|(?:->|\.)ne\s*\[[^\]]+\]|"
    r"\b(?:type|op|op_id)\b|(?:->|\.)(?:type|op)\b)")
_LITERAL = re.compile(
    r"(?:\b(?:0[xX][0-9A-Fa-f]+|\d+)\b|"
    r"\bGGML_(?:TYPE|OP)_[A-Z0-9_]+\b)")
_COMPARISON = re.compile(r"==|!=|<=|>=|<|>")
# DS41-C90 (2026-09-30): the three tests above were only required to hold SOMEWHERE in
# the statement, and `>` matched inside `->`. So `f(ne01, ..., src0->nb[1])` (the `>` of
# `->`, the `1` of `nb[1]`) and `slab_gran <= 0 || ne01 % slab_gran != 0` (a literal
# compared with a tuning variable) were "literal shape predicates": a +2.285% MUL_MAT_ID
# keep was parked as needs-confirm with no confirm rung configured. The literal must be
# an OPERAND of a comparison whose other side is the tensor subject (optionally cast, or
# reduced `% LIT` / `& LIT` -- `ne01 % 32 == 0` is still a shape special case).
_CMP_OP = r"(?:==|!=|<=|>=|(?<![-<>])>(?![>=])|(?<![<])<(?![<=]))"
_SUBJECT_TERM = (r"(?:\(\s*\w+\s*\)\s*)?" + _TENSOR_SUBJECT.pattern
                 + r"(?:\s*[%&]\s*" + _LITERAL.pattern + r")?")
_TIED_PREDICATE = re.compile(
    _SUBJECT_TERM + r"\s*" + _CMP_OP + r"\s*(?:\(\s*\w+\s*\)\s*)?" + _LITERAL.pattern
    + r"|" + _LITERAL.pattern + r"\s*" + _CMP_OP + r"\s*" + _SUBJECT_TERM)


def _literal_tensor_branch(statement: str) -> bool:
    """Whether an added branch compares tensor identity/shape with a literal."""
    return bool(_BRANCH.search(statement) and _TIED_PREDICATE.search(statement))


_MUTABLE_STATE = re.compile(
    r"\b(?:static\s+(?!const\b|constexpr\b)|(?:std::)?atomic\s*<|thread_local\b)")
_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_DECLARATION = re.compile(
    r"\b(?:static\s+|thread_local\s+)?(?:std::atomic\s*<[^>]+>|"
    r"(?:unsigned\s+|signed\s+)?(?:bool|char|short|int|long|float|double|size_t|"
    r"uint\d+_t|int\d+_t|auto))\s+(?:\*\s*)?([A-Za-z_]\w*)")


def _shape_tokens(shape: dict | None) -> set[str]:
    if not shape:
        return set()
    tokens: set[str] = set()
    for key in ("dims", "types", "op_ids"):
        for value in shape.get(key, ()):
            tokens.add(str(value))
    return tokens


def _matches(statement: str, shape: dict | None) -> tuple[str, ...]:
    tokens = _shape_tokens(shape)
    present = set(_IDENTIFIER.findall(statement)) | set(re.findall(r"\b\d+\b", statement))
    return tuple(sorted(tokens & present))
