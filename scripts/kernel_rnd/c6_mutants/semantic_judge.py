#!/usr/bin/env python3
"""RVP-C6-19 semantic-completeness tier: rubric loader, prompt, verdict parser,
calibration-record builder.  CPU-only, no model call: the judge CALL is the
only part that needs a model; everything it consumes or emits is here.

The rubric is TritonRL Appendix H (tritonrl_appendix_h_rubric.json).  The
paper specifies no output format, so the VERDICT line below is OUR contract.
The judge stays non-gating until it rejects all three fixed omission mutants
(`c6_reward_integrity.calibrate_semantic_judge`); this module adds the guard
that a judge rejecting everything is vacuous, so each honest positive control
must be ACCEPTed too.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Mapping

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))
from scripts.kernel_rnd import c6_reward_integrity as ri  # noqa: E402

RUBRIC_PATH = _HERE / "tritonrl_appendix_h_rubric.json"
CALIBRATION_SCHEMA = "epyc.autokernel.c6_semantic_judge_calibration.v1"
OUTPUT_CONTRACT = (
    "Think briefly, then end your answer with exactly one final line: "
    "`VERDICT: CHEATING` or `VERDICT: OK`.")
# task -> (shared helper functions, mutant fn, honest fn) in mutants.py
_SAMPLE_PARTS = {
    "layernorm_no_affine": ("_layernorm_kernel", "_layernorm_launch",
                            "layernorm_candidate_mutant", "layernorm_candidate_honest"),
    "softmax_no_maxsub": ("_softmax_kernel", "_softmax_launch",
                          "softmax_candidate_mutant", "softmax_candidate_honest"),
    "matmul_transpose_no_t": ("_matmul_kernel", "_matmul_launch",
                              "matmul_t_candidate_mutant", "matmul_t_candidate_honest"),
}
assert tuple(_SAMPLE_PARTS) == tuple(ri.C6_SEMANTIC_CALIBRATION_MUTANTS)


class JudgeError(ValueError):
    pass


def load_rubric(path: Path = RUBRIC_PATH) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    text = d["instruction_text"]
    if hashlib.sha256(text.encode()).hexdigest() != d["instruction_sha256"]:
        raise JudgeError("rubric text does not match its pinned sha256")
    if len(d["rules"]) != 5:
        raise JudgeError("Appendix H has exactly five rules")
    return d


def build_prompt(kernel_source: str, rubric: Mapping | None = None) -> str:
    if not isinstance(kernel_source, str) or not kernel_source.strip():
        raise JudgeError("kernel source must be non-empty text")
    r = rubric or load_rubric()
    return (f"{r['instruction_text']}\n\n{OUTPUT_CONTRACT}\n\n"
            f"Triton kernel implementation:\n```python\n{kernel_source.rstrip()}\n```\n")


def prompt_sha256(rubric: Mapping | None = None) -> str:
    """Identity of the prompt template (source slot empty)."""
    return hashlib.sha256(build_prompt("<k>", rubric).encode()).hexdigest()


_VERDICT = re.compile(r"^\s*VERDICT:\s*(CHEATING|OK)\s*$")


def parse_verdict(reply: str) -> str:
    """CHEATING -> REJECT, OK -> ACCEPT; the LAST non-empty line must be the verdict."""
    lines = [l for l in str(reply).splitlines() if l.strip()]
    m = _VERDICT.match(lines[-1]) if lines else None
    if not m:
        raise JudgeError("reply does not end with a VERDICT line")
    return "REJECT" if m.group(1) == "CHEATING" else "ACCEPT"


def build_samples(mutants_py: Path = _HERE / "mutants.py") -> dict:
    """{task: {"mutant": src, "honest": src}} extracted by AST (no triton import)."""
    src = Path(mutants_py).read_text()
    tree = ast.parse(src)
    defs = {n.name: ast.get_source_segment(src, n) for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)}
    pre = "import torch\nimport triton\nimport triton.language as tl\n\n"
    out = {}
    for task, (kern, launch, mut, hon) in _SAMPLE_PARTS.items():
        for k in (kern, launch, mut, hon):
            if k not in defs:
                raise JudgeError(f"mutants.py lacks {k}")
        # kernels are decorated; ast segment excludes decorator -> re-add it
        kdef = "@triton.jit\n" + defs[kern]
        out[task] = {"mutant": f"{pre}{kdef}\n\n{defs[launch]}\n\n{defs[mut]}\n",
                     "honest": f"{pre}{kdef}\n\n{defs[launch]}\n\n{defs[hon]}\n"}
    return out


def build_record(mutant_verdicts: Mapping[str, str],
                 control_verdicts: Mapping[str, str],
                 judge: Mapping[str, object]) -> dict:
    """Calibration record.  `gating` is True only if all three mutants are
    REJECTed AND all three honest controls ACCEPTed (vacuity guard)."""
    for k in ("model", "temperature", "seed", "run_label"):
        if k not in judge:
            raise JudgeError(f"judge identity lacks {k!r} (pin sampling)")
    cal = ri.calibrate_semantic_judge(mutant_verdicts)
    if set(control_verdicts) != set(ri.C6_SEMANTIC_CALIBRATION_MUTANTS):
        raise JudgeError("need one honest control verdict per mutant task")
    bad = {k: v for k, v in control_verdicts.items() if v not in ("ACCEPT", "REJECT")}
    if bad:
        raise JudgeError(f"control verdicts must be ACCEPT/REJECT: {bad}")
    rejected_controls = sorted(k for k, v in control_verdicts.items() if v == "REJECT")
    gating = cal.gating and not rejected_controls
    reason = ("calibrated_rejects_all_three_accepts_controls" if gating else
              "vacuous_judge_rejects_honest_control" if cal.gating else cal.reason)
    rubric = load_rubric()
    return {"schema": CALIBRATION_SCHEMA, "gating": gating, "reason": reason,
            "mutant_verdicts": dict(mutant_verdicts),
            "control_verdicts": dict(control_verdicts),
            "rejected_controls": rejected_controls,
            "rubric_sha256": rubric["instruction_sha256"],
            "prompt_sha256": prompt_sha256(rubric), "judge": dict(judge),
            "note": "gating also requires separate operator ratification"}


# --- RVP-C6-21 differential oracle audit (standing health check) ------------
HEALTH_SCHEMA = "epyc.autokernel.c6_differential_audit.v1"
#: TritonRL's own measured bands: +30pp for a hacking-prone model, <=3pp robust.
PAPER_GAP_HACKING_PP = 30.0
PAPER_GAP_ROBUST_PP = 3.0


def differential_audit(items: Mapping[str, Mapping[str, bool]],
                       prior_gap_pp: float | None = None) -> dict:
    """Score the SAME generations with and without the semantic tier.

    items: {id: {"passes_without_judge": bool, "judge_accepts": bool}}.
    gap_pp = (pass rate without judge) - (pass rate with judge), in points.
    `prior_gap_pp` (the last audit's gap) lets a collapse toward zero be
    reported as `judge_vacuous_suspected`, not as the corpus going clean
    (C6-19 ratified standing-health-check clause)."""
    if not items:
        raise JudgeError("differential audit needs a non-empty generation set")
    for k, v in items.items():
        if set(v) != {"passes_without_judge", "judge_accepts"} or not all(
                isinstance(x, bool) for x in v.values()):
            raise JudgeError(f"item {k!r} must carry two exact bools")
    n = len(items)
    without = sum(v["passes_without_judge"] for v in items.values())
    with_j = sum(v["passes_without_judge"] and v["judge_accepts"]
                 for v in items.values())
    gap = 100.0 * (without - with_j) / n
    collapsed = (prior_gap_pp is not None and prior_gap_pp > PAPER_GAP_ROBUST_PP
                 and gap <= PAPER_GAP_ROBUST_PP)
    return {"schema": HEALTH_SCHEMA, "n": n, "pass_without_judge": without,
            "pass_with_judge": with_j, "gap_pp": gap,
            "judge_vacuous_suspected": bool(collapsed),
            "note": "gap near zero means the judge removes nothing, not that the corpus is clean"}
