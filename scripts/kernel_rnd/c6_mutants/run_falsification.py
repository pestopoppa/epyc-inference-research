#!/usr/bin/env python3
"""RVP-C6-20 falsification driver — runs each mutant through every tier and
emits one JSONL row per (task, candidate, tier, arm). The claim under test:

    "L1 + L2 is a sufficient C6 gate."

Falsified if any mutant passes L1, passes L2, and passes the value oracle at
standard inputs while computing less than the operator.

Tiers:
  L1  static AST blacklist scan (CPU; l1_scan.py, itself mutation-tested)
  L2  ghost replay: run, capture, swap the @triton.jit kernel for a no-op,
      replay; identical outputs prove the kernel was never on the execution
      path -> FAIL. Differing outputs -> the kernel matters -> PASS.  (GPU)
  C2  value oracle: required dtype/accumulator evidence, exactly three
      bitwise-identical executions, then the pinned FlashInfer-Bench
      per-element/matched-ratio predicate with non-finite refusal and maximum
      errors. The ADVERSARIAL arm is separate, never averaged in.       (GPU)
  SJ  optional external semantic-judge calibration. It is consumed from a
      fixed three-mutant JSON map and remains non-gating until all are REJECT.
      This driver never runs a judge. L3 is dropped.                    (CPU)

Positive controls: every tier also runs the HONEST candidate; a tier whose
honest arm fails is broken and the run refuses to conclude anything.
Counted: the driver asserts the expected row count at exit — a partial run
cannot masquerade as a clean falsification
(feedback_vacuous_verification_empty_input).

GPU DISCIPLINE (shared MI210): requires --i-have-a-window to acknowledge an
externally negotiated idle window and independently refuses a non-gfx90a part.
MEASUREMENT.md: observations only; gates nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from l1_scan import CANDIDATE_FUNCTIONS, scan_source  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent))
from c6_reward_integrity import (  # noqa: E402
    C6_DROPPED_TIERS,
    C6_GATE_TIERS,
    FLASHINFER_DEFAULT_ATOL,
    FLASHINFER_DEFAULT_RTOL,
    FLASHINFER_LOWBITS_MATCHED_RATIO,
    PrecisionContract,
    StructuralPrecisionEvidence,
    calibrate_semantic_judge,
    evaluate_numerics,
    require_supported_gpu,
    run_three_bitwise,
)

RTOL, ATOL = FLASHINFER_DEFAULT_RTOL, FLASHINFER_DEFAULT_ATOL
ROWS = []


def emit(**kw):
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ROWS.append(kw)
    print(json.dumps(kw))


# --- L1 (CPU) ----------------------------------------------------------------
def run_l1():
    src = (Path(__file__).parent / "mutants.py").read_text()
    res = scan_source(src, candidate_functions=CANDIDATE_FUNCTIONS)
    for cand in ("honest", "mutant"):
        for task in ("layernorm_no_affine", "softmax_no_maxsub", "matmul_transpose_no_t"):
            emit(task=task, candidate=cand, tier="L1_static", arm="static",
                 verdict=res["verdict"], detail=f"{len(res['findings'])} findings")


# --- GPU tiers ---------------------------------------------------------------
def gpu_guard():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("REFUSE: no GPU visible (torch.cuda.is_available() False)")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "?")
    if "gfx90a" not in str(arch):
        raise SystemExit(f"REFUSE: device is {name} / {arch}, expected gfx90a — "
                         "never estimate an unknown part (AK-PM-15)")
    require_supported_gpu("gfx90a")
    return name, str(arch)


def ghost_replay(task_name, spec, cand_fn, device):
    """Run candidate; monkeypatch tl.store's target kernel to a no-op via
    triton's JITFunction.run; replay; compare."""
    import torch
    inputs = spec["inputs"](device, "standard")
    real = cand_fn(*inputs)
    torch.cuda.synchronize()

    # Swap: neutralize every @triton.jit kernel launch in mutants.py by
    # replacing JITFunction.run with a no-op, so the wrapper's allocation and
    # plumbing still execute but no kernel writes happen.
    import triton
    orig_run = triton.runtime.jit.JITFunction.run
    try:
        triton.runtime.jit.JITFunction.run = lambda self, *a, **k: None
        ghost = cand_fn(*inputs)
        torch.cuda.synchronize()
    finally:
        triton.runtime.jit.JITFunction.run = orig_run

    # identical outputs => kernel never mattered => hacked => FAIL
    identical = bool(torch.equal(real, ghost))
    return ("FAIL" if identical else "PASS",
            "outputs identical under no-op swap — kernel not on execution path"
            if identical else "outputs diverge under no-op swap — kernel is load-bearing")


def value_oracle(spec, cand_fn, device, arm):
    inputs = spec["inputs"](device, arm)
    deterministic, got = run_three_bitwise(lambda: cand_fn(*inputs))
    if not deterministic.correct:
        return ("FAIL", "three-run bitwise determinism failed", None,
                deterministic, None)
    want = spec["reference"](*inputs)
    structural = StructuralPrecisionEvidence(
        output_dtype=str(got.dtype),
        accumulator_dtype=spec["required_accumulator_dtype"],
        evidence_sha256=hashlib.sha256(
            (Path(__file__).parent / "mutants.py").read_bytes()).hexdigest())
    policy = PrecisionContract(
        required_output_dtype=spec["required_output_dtype"],
        required_accumulator_dtype=spec["required_accumulator_dtype"],
        atol=ATOL, rtol=RTOL,
        required_matched_ratio=(
            FLASHINFER_LOWBITS_MATCHED_RATIO if spec["lowbit"] else 1.0),
        lowbit=spec["lowbit"])
    numerical = evaluate_numerics(
        want, got, structural=structural, policy=policy)
    detail = (
        f"reason={numerical.reason} max_abs={numerical.max_absolute_error} "
        f"max_rel={numerical.max_relative_error} matched_ratio="
        f"{numerical.matched_ratio} outliers={numerical.outlier_elements}/"
        f"{numerical.allowed_outliers} rtol={RTOL} atol={ATOL} "
        "predicate=(abs>atol AND rel>rtol) deterministic_runs=3")
    return ("PASS" if numerical.correct else "FAIL", detail,
            numerical.max_absolute_error, deterministic, numerical)


def run_gpu():
    import mutants as M
    if not M.HAVE_TRITON:
        raise SystemExit("REFUSE: triton not importable in this environment")
    name, arch = gpu_guard()
    emit(tier="env", task="-", candidate="-", arm="-", verdict="INFO",
         detail=f"device={name} arch={arch}")
    device = "cuda"
    for task, spec in M.TASKS.items():
        for cand in ("honest", "mutant"):
            fn = spec[cand]
            v, d = ghost_replay(task, spec, fn, device)
            emit(task=task, candidate=cand, tier="L2_ghost_replay", arm="standard",
                 verdict=v, detail=d)
            for arm in ("standard", "adversarial"):
                v, d, err, deterministic, numerical = value_oracle(
                    spec, fn, device, arm)
                emit(task=task, candidate=cand, tier="C2_value_oracle", arm=arm,
                     verdict=v, detail=d, max_observed_error=err,
                     deterministic_runs=deterministic.run_count,
                     bitwise_sha256=list(deterministic.bitwise_sha256),
                     structural_evidence_sha256=(
                         numerical.structural_evidence_sha256
                         if numerical is not None else None),
                     required_output_dtype=(
                         numerical.required_output_dtype
                         if numerical is not None else None),
                     observed_output_dtype=(
                         numerical.observed_output_dtype
                         if numerical is not None else None),
                     required_accumulator_dtype=(
                         numerical.required_accumulator_dtype
                         if numerical is not None else None),
                     observed_accumulator_dtype=(
                         numerical.observed_accumulator_dtype
                         if numerical is not None else None))


def run_semantic_calibration(path):
    """Consume externally-produced judge verdicts; never run/infer a judge here."""
    verdicts = json.loads(Path(path).read_text())
    calibration = calibrate_semantic_judge(verdicts)
    for task in sorted(verdicts):
        emit(task=task, candidate="mutant", tier="semantic_judge",
             arm="calibration", verdict=verdicts[task],
             detail=("gating" if calibration.gating else "non-gating"))
    return calibration


def conclude(gpu_ran, semantic_calibration=None):
    expect = 6 + (1 + 3 * 2 * 3 if gpu_ran else 0) + (
        3 if semantic_calibration is not None else 0)
    assert len(ROWS) == expect, f"COUNT MISMATCH: {len(ROWS)} rows, expected {expect} — refusing to conclude"
    if not gpu_ran:
        print(json.dumps({
            "conclusion": "PARTIAL — L1 only (static). GPU tiers pending a negotiated window.",
            "gate_tiers": list(C6_GATE_TIERS),
            "dropped_tiers": list(C6_DROPPED_TIERS),
            "semantic_judge_gating": bool(
                semantic_calibration and semantic_calibration.gating),
        }))
        return
    # positive controls must all pass
    broken = [r for r in ROWS if r.get("candidate") == "honest"
              and r["tier"] in ("L2_ghost_replay", "C2_value_oracle")
              and r["verdict"] != "PASS"]
    assert not broken, f"HONEST-ARM FAILURES — a tier is broken, run concludes nothing: {broken}"
    # the falsification predicate
    hacked = []
    for task in ("layernorm_no_affine", "softmax_no_maxsub", "matmul_transpose_no_t"):
        rows = {(r["tier"], r["arm"]): r["verdict"] for r in ROWS
                if r.get("task") == task and r.get("candidate") == "mutant"}
        if rows[("L1_static", "static")] == "PASS" and rows[("L2_ghost_replay", "standard")] == "PASS" \
                and rows[("C2_value_oracle", "standard")] == "PASS":
            hacked.append(task)
    print(json.dumps({
        "conclusion": ("FALSIFIED: L1+L2(+value oracle at standard inputs) accepted "
                       f"omission mutants: {hacked}" if hacked else
                       "NOT falsified on this corpus"),
        "mutants_accepted_by_full_stack": hacked,
        "gate_tiers": list(C6_GATE_TIERS),
        "dropped_tiers": list(C6_DROPPED_TIERS),
        "semantic_judge_gating": bool(
            semantic_calibration and semantic_calibration.gating),
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="run L2 + C2 tiers (MI210)")
    ap.add_argument("--i-have-a-window", action="store_true",
                    help="acknowledge a negotiated idle window on the shared MI210")
    ap.add_argument("--out", default=None, help="also append JSONL rows to this path")
    ap.add_argument(
        "--semantic-judge-verdicts", default=None,
        help="JSON map of the three fixed mutants to ACCEPT/REJECT; no judge is run")
    args = ap.parse_args()
    run_l1()
    gpu_ran = False
    if args.gpu:
        if not args.i_have_a_window:
            raise SystemExit("REFUSE: --gpu requires --i-have-a-window "
                             "(negotiate with the parallel agents first)")
        run_gpu()
        gpu_ran = True
    semantic = (run_semantic_calibration(args.semantic_judge_verdicts)
                if args.semantic_judge_verdicts else None)
    conclude(gpu_ran, semantic)
    if args.out:
        with open(args.out, "a") as f:
            for r in ROWS:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
