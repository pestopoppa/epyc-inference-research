"""C5-ISA-2 (GCN-Gym): host-side harness for LLM-edited gfx90a AMDGCN.

Scope of this module (no GPU): the edit -> reassemble -> static-gate half of the loop
and the pre-registered decision rule. The GPU half (hipModuleLoad + served-shape
launch replay through the RVP-RACE-2 launch-capture shim, AK oracle diff-test, ABBA
timing) runs in a GPU row and hands back the numbers ``decide`` consumes.

Pipeline per agent edit:

1. ``apply_patch(source_s, unified_diff)`` -- strict unified-diff application (context
   must match exactly; fuzz is refused) to the hipcc ``--save-temps`` gfx90a ``.s`` of
   one translation unit (the reassemblable form; ``llvm-objdump`` text is not).
2. ``assemble_argv`` / ``link_argv`` -- ``clang -target amdgcn-amd-amdhsa
   -mcpu=gfx90a -c`` then ``ld.lld -shared`` into a ``.hsaco`` code object.
3. ``static_gate`` -- ``gfx90a_isa_audit.py diff`` of the candidate code object vs the
   hipcc incumbent must exit 0 (and, once RVP-RACE-2 M2 lands, its verdict must not be
   a finding).
4. ``isa_claims_gate`` (C5-ISA-1) -- beside the audit diff, the candidate's touched
   kernels are archived and every mechanism it claims (``isa_claims``) is checked in the
   ISA. ``accept_gate`` combines the two. Default ENFORCE, fail-closed only on a REFUTED
   claim (``ABSENT`` in the ISA, or ``NOT_EXECUTED`` per the counter witness); a
   ``STATIC_ONLY`` claim (present, execution unproven) is allowed, no claim is a pass, and
   a checker error is recorded as ``UNCHECKED`` without refusing.
   ``AK_ISA_CLAIMS_GATE=advisory`` (or ``0``/``off``) records the verdict without refusing.

Decision rule (handoff C5-ISA-2, per target, 3 runs at a fixed token budget): open an
ISA-rung arm if >=1 run reaches >=1.05x kernel speed at the served prefill shape,
passes the oracle, and its ABBA interval (>=2 pairs) does not overlap the incumbent's.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Callable, Mapping, Sequence

ROCM_LLVM = Path("/opt/rocm/llvm/bin")
SPEEDUP_FLOOR = 1.05
MIN_ABBA_PAIRS = 2
RUNS_PER_TARGET = 3
TARGETS = ("MMQ Q8_0 J=64 prefill tile", "MMA-FA DKQ>=192")


class PatchRefused(ValueError):
    pass


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_patch(source: str, diff: str) -> str:
    """Apply a unified diff exactly (no fuzz, no offset search)."""
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    pos = 0
    hunks = 0
    it = iter(diff.splitlines(keepends=True))
    for line in it:
        m = _HUNK.match(line)
        if not m:
            if line.startswith(("---", "+++", "diff ", "index ")) or not line.strip():
                continue
            raise PatchRefused(f"stray diff line: {line!r}")
        hunks += 1
        start = int(m.group(1)) - 1 if int(m.group(2) or 1) else int(m.group(1))
        old_n = int(m.group(2) if m.group(2) is not None else 1)
        new_n = int(m.group(4) if m.group(4) is not None else 1)
        if start < pos:
            raise PatchRefused("overlapping or unordered hunks")
        out.extend(lines[pos:start])
        pos = start
        seen_old = seen_new = 0
        while seen_old < old_n or seen_new < new_n:
            try:
                body = next(it)
            except StopIteration:
                raise PatchRefused("truncated hunk") from None
            tag, text = body[:1], body[1:]
            if tag in (" ", "-"):
                if pos >= len(lines) or lines[pos].rstrip("\n") != text.rstrip("\n"):
                    raise PatchRefused(f"context mismatch at line {pos + 1}")
                pos += 1
                seen_old += 1
                if tag == " ":
                    out.append(lines[pos - 1])
                    seen_new += 1
            elif tag == "+":
                out.append(text if text.endswith("\n") else text + "\n")
                seen_new += 1
            elif body.startswith("\\"):
                continue
            else:
                raise PatchRefused(f"bad hunk line {body!r}")
    if not hunks:
        raise PatchRefused("diff has no hunks")
    out.extend(lines[pos:])
    return "".join(out)


def assemble_argv(source_s: Path, object_o: Path, *, llvm: Path = ROCM_LLVM) -> list[str]:
    return [str(llvm / "clang"), "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx90a",
            "-c", str(source_s), "-o", str(object_o)]


def link_argv(object_o: Path, hsaco: Path, *, llvm: Path = ROCM_LLVM) -> list[str]:
    return [str(llvm / "ld.lld"), "-shared", str(object_o), "-o", str(hsaco)]


# Pre-timing ISA gate family set: the audit DEFAULT_FAMILIES, which include the served
# decode/verify attention kernels (fattn_vec, fattn_tile, gqa_verify, qsa_sparse) so a VGPR
# increase, spill or occupancy drop there fails before any timing (AK-ISA-FAMILIES).
ISA_GATE_FAMILIES = "mmq,mmf,fattn_mma,fattn_wmma,fattn_vec,fattn_tile,gqa_verify,qsa_sparse"


def static_gate_argv(incumbent_json: Path, candidate_json: Path, *,
                     families: str = ISA_GATE_FAMILIES,
                     audit: Path | None = None) -> list[str]:
    audit = audit or Path(__file__).resolve().parents[2] / "gfx90a_isa_audit.py"
    return ["python3", str(audit), "diff", str(incumbent_json), str(candidate_json),
            "--families", families]


ISA_CLAIMS_ENV = "AK_ISA_CLAIMS_GATE"
_ADVISORY = {"0", "off", "advisory", "false", "no"}
REFUTED = ("ABSENT", "NOT_EXECUTED")


def isa_claims_mode(environ: Mapping[str, str] | None = None) -> str:
    v = (os.environ if environ is None else environ).get(ISA_CLAIMS_ENV, "").strip().lower()
    return "advisory" if v in _ADVISORY else "enforce"


def isa_claims_gate(candidate_binary: Path, kernel_regex: str, claims: Sequence[str], *,
                    archive: Path | None = None, observation: Mapping[str, Any] | None = None,
                    checker: Callable[..., Mapping[str, Any]] | None = None,
                    environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """C5-ISA-1 at the candidate build: archive the touched kernels' ISA and check the claims.

    Refuses (``passed`` False, enforce mode) only when a claimed mechanism is refuted:
    ``ABSENT`` from every touched kernel or ``NOT_EXECUTED`` per the counter witness. A
    kernel regex matching nothing makes every claim ABSENT, so it refuses too."""
    from . import isa_claims  # noqa: PLC0415 -- keeps gcn_gym importable without the audit
    mode = isa_claims_mode(environ)
    claims = sorted(set(claims))
    if not claims:
        return {"gate": "isa_claims", "mode": mode, "status": "NO_CLAIMS", "passed": True,
                "refuted": [], "doc": None}
    check = checker or isa_claims.archive_and_check
    try:
        doc = dict(check(candidate_binary, kernel_regex, claims, archive=archive,
                         observation=observation))
    except Exception as exc:  # noqa: BLE001 -- subprocess failures from llvm-objdump; never a refutation
        return {"gate": "isa_claims", "mode": mode, "status": "UNCHECKED", "passed": True,
                "refuted": [], "error": f"{type(exc).__name__}: {exc}", "doc": None}
    statuses = {m: (doc.get("claims") or {}).get(m, {}).get("status", "ABSENT") for m in claims}
    refuted = sorted(m for m, st in statuses.items() if st in REFUTED)
    status = "REFUTED" if refuted else "ACCEPTED"
    return {"gate": "isa_claims", "mode": mode, "status": status,
            "passed": not (refuted and mode == "enforce"), "refuted": refuted,
            "statuses": statuses, "doc": doc}


def accept_gate(static_gate_rc: int, isa_claims_result: Mapping[str, Any]) -> dict[str, Any]:
    """The candidate build's accept decision: the audit diff (``static_gate_argv``) must exit 0
    AND the C5-ISA-1 claims gate must pass."""
    reasons = []
    if static_gate_rc != 0:
        reasons.append(f"isa_audit_diff_rc={static_gate_rc}")
    if not isa_claims_result.get("passed", False):
        reasons.append("isa_claims_refuted:" + ",".join(isa_claims_result.get("refuted") or []))
    return {"accepted": not reasons, "reasons": reasons,
            "isa_audit_diff_rc": static_gate_rc, "isa_claims": isa_claims_result.get("status")}


@dataclass(frozen=True)
class RunResult:
    """One agent run on one target, measured on the GPU (served prefill shape)."""
    oracle_pass: bool
    candidate_ms: Sequence[float]   # one per ABBA pair (mean of its two B legs)
    incumbent_ms: Sequence[float]   # one per ABBA pair (mean of its two A legs)


def _interval(xs: Sequence[float]) -> tuple[float, float]:
    return (min(xs), max(xs))


def run_qualifies(run: RunResult) -> dict:
    pairs = min(len(run.candidate_ms), len(run.incumbent_ms))
    if not run.oracle_pass:
        return {"qualifies": False, "reason": "oracle_fail"}
    if pairs < MIN_ABBA_PAIRS:
        return {"qualifies": False, "reason": "too_few_abba_pairs"}
    speedup = statistics.median(run.incumbent_ms) / statistics.median(run.candidate_ms)
    c_lo, c_hi = _interval(run.candidate_ms)
    i_lo, _i_hi = _interval(run.incumbent_ms)
    overlap = c_hi >= i_lo
    if speedup < SPEEDUP_FLOOR:
        return {"qualifies": False, "reason": "below_1.05x", "speedup": speedup}
    if overlap:
        return {"qualifies": False, "reason": "abba_overlap", "speedup": speedup}
    return {"qualifies": True, "reason": "ok", "speedup": speedup}


def decide(runs: Sequence[RunResult]) -> dict:
    if len(runs) != RUNS_PER_TARGET:
        return {"decision": "INVALID", "reason": f"need {RUNS_PER_TARGET} runs, got {len(runs)}"}
    verdicts = [run_qualifies(r) for r in runs]
    if any(v["qualifies"] for v in verdicts):
        return {"decision": "OPEN_ISA_RUNG_ARM", "runs": verdicts}
    return {"decision": "NULL_STOP_ISA_RUNG_FOR_KERNEL_CLASS", "runs": verdicts}
