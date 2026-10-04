"""coherence_gate: the shared, deterministic, PAIRED output-coherence gate.

Operator-approved 2026-10-04 to replace the copy-pasted INF-70 classifier (classify.py, its
lib_gpublock re-implementation and ~15 client copies). No inference, no processes: it reads outputs
that a harness already captured and returns one verdict per item, naming the tier that decided it.

    tier 0  paired byte-identity vs the base/anchor arm (greedy)      -> PASS
    tier 1  ground-truth answer checks + degeneracy.v2 on REAL token ids, compared PAIRED
            -> REGRESSION (candidate worse) | BOTH_BAD (reported, not failed) | PASS
    tier 2  injected judge_fn(item_pair), only for diverged items with no decisive ground truth;
            default: NEEDS_REVIEW ("unjudged")
    gate    FAIL (any REGRESSION) | INCOMPLETE (any NEEDS_REVIEW, or no items) | PASS

Import with scripts/lib on sys.path (the repo convention for scripts/lib modules):

    sys.path.insert(0, "<repo>/scripts/lib")
    import coherence_gate as CG
    report = CG.evaluate(base_rows, cand_rows, truth_rows,
                         tokenizer=CG.HFTokenizer(".../tokenizer.json"))   # or rows carry token_ids
    report["aggregate"]["gate"]

CLI:  PYTHONPATH=scripts/lib python3 -m coherence_gate --base b.jsonl --cand c.jsonl --truth t.jsonl
Rows: {"id", "text" | "content"[+"reasoning"] | "text_path", "token_ids"?, "finish"?, "http_ok"?, "sha256"?}
Truth: question_pool.jsonl rows ({id, expected, scoring_method, scoring_config}) or {id, expected, grader}.
Schema of the report: `SCHEMA_ID` (epyc.coherence_gate.v1); see gate.py for the item/aggregate fields.
"""
from __future__ import annotations

from . import answers, degeneracy, tokens
from .answers import GRADERS, GRADERS_VERSION, grade, grader_for, register_grader
from .degeneracy import V1_ID, V2_ID, classify, classify_stats, classify_v1, severity
from .gate import (LIBRARY_VERSION, SCHEMA_ID, VERDICTS, GateInputError, aggregate, evaluate,
                   evaluate_pair, load_jsonl, normalize_row, sha256_text)
from .tokens import (FnTokenizer, HFTokenizer, ServerTokenizer, TokenProvenanceError, fake_reason,
                     surrogate_ids, validate_ids)

__all__ = [
    "answers", "degeneracy", "tokens",
    "GRADERS", "GRADERS_VERSION", "grade", "grader_for", "register_grader",
    "V1_ID", "V2_ID", "classify", "classify_stats", "classify_v1", "severity",
    "LIBRARY_VERSION", "SCHEMA_ID", "VERDICTS", "GateInputError", "aggregate", "evaluate",
    "evaluate_pair", "load_jsonl", "normalize_row", "sha256_text",
    "FnTokenizer", "HFTokenizer", "ServerTokenizer", "TokenProvenanceError", "fake_reason",
    "surrogate_ids", "validate_ids",
]
