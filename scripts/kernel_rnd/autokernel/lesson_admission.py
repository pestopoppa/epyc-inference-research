"""Execution-gated lesson admission; the caller supplies the native rerun.

An ADMITTED receipt is only an admission to the prospective lesson library.
It is not a measurement warrant or a retrieval/planning canary.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

KIND = "LESSON_ADMISSION_EVALUATED"
_KEYS = frozenset({"schema", "task_id", "candidate_id", "commit", "lesson", "reopen_when",
                   "r1", "upper_cap", "verification", "decision", "reason", "attempt_ref"})


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def violations(p):
    if not isinstance(p, Mapping) or set(p) != _KEYS:
        return ["invalid receipt fields"]
    errors = []
    if p["schema"] != "autokernel.lesson_admission.v1":
        errors.append("invalid schema")
    for name in ("task_id", "candidate_id", "lesson", "reopen_when"):
        if not isinstance(p[name], str) or not p[name].strip():
            errors.append(f"{name} is required")
    if not isinstance(p["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", p["commit"]):
        errors.append("commit must be a full lowercase Git SHA")
    if not _number(p["r1"]) and not (p["r1"] is None and p["verification"] is None):
        errors.append("r1 must be finite and positive, or null when verification is unavailable")
    if p["attempt_ref"] is not None and (not isinstance(p["attempt_ref"], str) or not re.fullmatch(r"/[^#]+#[0-9a-f]{64}", p["attempt_ref"])):
        errors.append("invalid attempt reference")
    cap = p["upper_cap"]
    if cap is not None and not _number(cap):
        errors.append("upper_cap must be finite and positive or null")
    verify = p["verification"]
    if verify is None:
        expected = ("UNKNOWN", "cap_missing" if cap is None else "verification_unavailable")
    elif not isinstance(verify, Mapping) or set(verify) != {"task_id", "commit", "r_verify", "run_ref"}:
        errors.append("invalid verification fields")
        expected = None
    elif (verify["task_id"] != p["task_id"] or verify["commit"] != p["commit"]
          or not isinstance(verify["run_ref"], str) or not re.fullmatch(r'/[^#]+#manifest=[0-9a-f]{64};baseline=[0-9a-f]{64};verification=[0-9a-f]{64}', verify["run_ref"])
          or not _number(verify["r_verify"])):
        errors.append("verification must bind the same task and commit with a run reference and finite positive result")
        expected = None
    elif cap is None:
        errors.append("verification cannot run without a configured cap")
        expected = None
    elif _number(p["r1"]) and _number(cap):
        r = verify["r_verify"]
        if r > cap or p["r1"] > cap:
            expected = ("REJECTED", "upper_cap_exceeded")
        elif r < max(1.2, 1.2 * p["r1"]):
            expected = ("REJECTED", "verification_below_threshold")
        else:
            expected = ("ADMITTED", "execution_verified")
    else:
        expected = None
    if expected is not None and (p["decision"], p["reason"]) != expected:
        errors.append("decision does not match evidence and configured cap")
    return errors


def _evaluate_and_record(journal, *, task_id, candidate_id, commit, lesson, reopen_when,
                        r1, upper_cap, rerun, attempt_ref=None):
    """Call the native injected rerun on this task, then fsync its decision.

    The caller owns the rerun implementation and immutable run reference. No
    rerun occurs if the explicit upper cap is missing or the input is invalid.
    """
    p = {"schema": "autokernel.lesson_admission.v1", "task_id": task_id,
         "candidate_id": candidate_id, "commit": commit, "lesson": lesson,
         "reopen_when": reopen_when, "r1": r1, "upper_cap": upper_cap,
         "verification": None, "attempt_ref": None, "decision": "UNKNOWN",
         "reason": "cap_missing" if upper_cap is None else "verification_unavailable"}
    issues = violations(p)
    if issues:
        raise ValueError("; ".join(issues))
    if upper_cap is not None:
        try:
            result = rerun(task_id=task_id, commit=commit, lesson=lesson)
        except Exception:  # noqa: BLE001 - a failed native execution must remain UNKNOWN
            result = None
        if result is not None:
            p["verification"] = result
            if isinstance(result, Mapping) and _number(result.get("r_verify")):
                if result["r_verify"] > upper_cap or r1 > upper_cap:
                    p["decision"], p["reason"] = "REJECTED", "upper_cap_exceeded"
                elif result["r_verify"] < max(1.2, 1.2 * r1):
                    p["decision"], p["reason"] = "REJECTED", "verification_below_threshold"
                else:
                    p["decision"], p["reason"] = "ADMITTED", "execution_verified"
            issues = violations(p)
            if issues:
                # Invalid or mismatched native evidence never enters as verified.
                p["verification"] = None
                p["decision"], p["reason"] = "UNKNOWN", "verification_unavailable"
    if p["decision"] == "UNKNOWN" and attempt_ref is not None:
        p["attempt_ref"] = attempt_ref()
    return journal.append(KIND, p)
