"""AK-BLOCKED-VS-FAILED: a gate refusal caused by the HOST is blocked, never a candidate failure.

A run with no GPU, a residency failure, an out-of-memory kill or a missing host resource is a
host fact. Recording it as a compile/correctness failure spends the candidate's authoring
budget and teaches the planner that a correct idea was wrong (intake-2746). The loop asks
`classify_verdict` before treating a failed gate verdict as a code fact.

Case table (source paths: `gates.py` Verdict producers -> `loop.iterate` gate branch):

    GPU absent       hipErrorNoDevice / "no ROCm-capable device" / "no usable GPU"   -> blocked
    residency fail   "residency" failure text / VRAM not resident                      -> blocked
    OOM              "out of memory" / std::bad_alloc / cc1plus "Killed signal" /
                     rc -9 (SIGKILL) / hipErrorOutOfMemory                             -> blocked
    host resource    "No space left on device" / "Too many open files" / ENOSPC        -> blocked
    candidate wrong  anything else (compile error, wrong output, op_scope)              -> failed

Only `compile`, `configure`, `correctness`, `reference_comparison` and `oracle_unavailable`
verdicts can be host-blocked; rule gates (`op_scope`) never are.
"""
from __future__ import annotations

import re
from typing import Any

BLOCKED = "host_blocked"
FAILED = "candidate_failed"

_HOST_GATES = frozenset({"compile", "configure", "correctness", "reference_comparison",
                         "oracle_unavailable"})

CASES = (
    ("gpu_absent", re.compile(
        r"hipErrorNoDevice|no ROCm-capable device|no usable GPU|no GPU (?:device )?(?:found|available)"
        r"|ggml_cuda_init: failed|hipGetDeviceCount\(\) returned 0|No devices found", re.I)),
    ("residency_failed", re.compile(
        r"residency (?:check |proof )?fail|not resident|VRAM residency", re.I)),
    ("oom", re.compile(
        r"out of memory|std::bad_alloc|hipErrorOutOfMemory|Killed signal terminated program"
        r"|\brc=-9\b|returncode -9|SIGKILL|Cannot allocate memory", re.I)),
    ("host_resource", re.compile(
        r"No space left on device|ENOSPC|Too many open files|EMFILE|Resource temporarily unavailable",
        re.I)),
)


def classify_verdict(verdict: Any) -> dict:
    """Return {"class": host_blocked|candidate_failed, "case": str|None, "gate": str}."""
    gate = str(getattr(verdict, "gate", "") or "")
    if getattr(verdict, "passed", False):
        raise ValueError("classify_verdict takes a FAILED verdict")
    if gate in _HOST_GATES:
        text = f"{getattr(verdict, 'reason', '') or ''}\n{getattr(verdict, 'detail', '') or ''}"
        for case, pattern in CASES:
            if pattern.search(text):
                return {"class": BLOCKED, "case": case, "gate": gate}
    return {"class": FAILED, "case": None, "gate": gate}


__all__ = ["BLOCKED", "CASES", "FAILED", "classify_verdict"]
