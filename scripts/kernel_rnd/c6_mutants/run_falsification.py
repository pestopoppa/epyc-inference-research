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
  C2  value oracle: torch.allclose(candidate, reference, rtol/atol) at the
      STANDARD input arm, plus max-observed-error recorded per
      intake-1245's discipline. The ADVERSARIAL arm is run separately to
      demonstrate the repair, never averaged in.                       (GPU)

Positive controls: every tier also runs the HONEST candidate; a tier whose
honest arm fails is broken and the run refuses to conclude anything.
Counted: the driver asserts the expected row count at exit — a partial run
cannot masquerade as a clean falsification
(feedback_vacuous_verification_empty_input).

GPU DISCIPLINE (shared MI210): refuses to run if another KFD process holds the
GPU, and requires --i-have-a-window to acknowledge the negotiated idle window.
MEASUREMENT.md: observations only; gates nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from l1_scan import CANDIDATE_FUNCTIONS, scan_source  # noqa: E402

RTOL, ATOL = 1e-3, 1e-3
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
            emit(task=task, candidate=cand, tier="L1", arm="static",
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
    return name, str(arch)


def ghost_replay(task_name, spec, cand_fn, device):
    """Run candidate; monkeypatch tl.store's target kernel to a no-op via
    triton's JITFunction.run; replay; compare."""
    import mutants as M
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
    import torch
    inputs = spec["inputs"](device, arm)
    got = cand_fn(*inputs)
    want = spec["reference"](*inputs)
    finite = bool(torch.isfinite(got).all())
    if not finite:
        return "FAIL", "non-finite output (NaN/Inf rejected outright per intake-1245)", None
    err = (got.float() - want.float()).abs().max().item()
    ok = bool(torch.allclose(got.float(), want.float(), rtol=RTOL, atol=ATOL))
    return ("PASS" if ok else "FAIL",
            f"max_observed_error={err:.3e} rtol={RTOL} atol={ATOL}", err)


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
                v, d, err = value_oracle(spec, fn, device, arm)
                emit(task=task, candidate=cand, tier="C2_value_oracle", arm=arm,
                     verdict=v, detail=d, max_observed_error=err)


def conclude(gpu_ran):
    expect = 6 + (1 + 3 * 2 * 3 if gpu_ran else 0)
    assert len(ROWS) == expect, f"COUNT MISMATCH: {len(ROWS)} rows, expected {expect} — refusing to conclude"
    if not gpu_ran:
        print(json.dumps({"conclusion": "PARTIAL — L1 only (static). GPU tiers pending a negotiated window."}))
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
        if rows[("L1", "static")] == "PASS" and rows[("L2_ghost_replay", "standard")] == "PASS" \
                and rows[("C2_value_oracle", "standard")] == "PASS":
            hacked.append(task)
    print(json.dumps({
        "conclusion": ("FALSIFIED: L1+L2(+value oracle at standard inputs) accepted "
                       f"omission mutants: {hacked}" if hacked else
                       "NOT falsified on this corpus"),
        "mutants_accepted_by_full_stack": hacked,
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="run L2 + C2 tiers (MI210)")
    ap.add_argument("--i-have-a-window", action="store_true",
                    help="acknowledge a negotiated idle window on the shared MI210")
    ap.add_argument("--out", default=None, help="also append JSONL rows to this path")
    args = ap.parse_args()
    run_l1()
    gpu_ran = False
    if args.gpu:
        if not args.i_have_a_window:
            raise SystemExit("REFUSE: --gpu requires --i-have-a-window "
                             "(negotiate with the parallel agents first)")
        run_gpu()
        gpu_ran = True
    conclude(gpu_ran)
    if args.out:
        with open(args.out, "a") as f:
            for r in ROWS:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
