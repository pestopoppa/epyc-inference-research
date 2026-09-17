"""Opt-in, non-ranked fresh-input correctness receipt for one benchmark arm.

The caller supplies an already-authorized process runner.  Nothing in the live
loop imports this module: collecting a receipt neither changes a ranked request
nor gives a result keep/promotion authority.  In particular, a CPU backend's
candidate-local CPU reference is *not* an independent reference for CPU edits.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping

from .. import schemas
from ..execution import instrument_integrity, t0_provider
from .integrity import CandidateIntegrity

SCHEMA = "epyc.autokernel.fresh_correctness.v1"
MAX_OUTPUT_BYTES = 256 * 1024


@dataclass(frozen=True)
class Plan:
    binary: Path
    expected_binary_sha256: str
    candidate_root: Path
    anchor_root: Path
    arm_id: str
    recipe_hash: str
    backend: str
    op: str
    suite_seed: int
    candidate_integrity: CandidateIntegrity | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
                self.arm_id, self.recipe_hash, self.backend, self.op)):
            raise ValueError("arm, recipe, backend and op identities are required")
        if isinstance(self.suite_seed, bool) or not isinstance(self.suite_seed, int) \
                or not 0 <= self.suite_seed < 2**64:
            raise ValueError("suite_seed must be a uint64 fixed before the arm")
        if not self.binary.is_absolute() or not self.candidate_root.is_absolute() \
                or not self.anchor_root.is_absolute():
            raise ValueError("binary and source roots must be absolute")
        if self.binary.name != "test-backend-ops":
            raise ValueError("fresh op receipt requires the selected test-backend-ops binary")
        if (len(self.expected_binary_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in self.expected_binary_sha256)):
            raise ValueError("expected binary SHA-256 must come from the build identity")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _receipt(plan: Plan, *, binary_sha256: str | None, status: str,
             verdict: str = "unavailable", reason: str,
             **facts: object) -> dict:
    return {"schema": SCHEMA, "authority": "report_only",
            "ranked_sample": False, "arm_id": plan.arm_id,
            "recipe_hash": plan.recipe_hash, "backend": plan.backend,
            "op": plan.op, "suite_seed": plan.suite_seed,
            "binary_sha256": binary_sha256, "status": status,
            "verdict": verdict, "reason": reason, **facts}


def _raw_facts(stdout: str, stderr: str) -> dict:
    out_bytes, err_bytes = stdout.encode(), stderr.encode()
    bound = (len(out_bytes).to_bytes(8, "big") + out_bytes
             + len(err_bytes).to_bytes(8, "big") + err_bytes)
    size = len(out_bytes) + len(err_bytes)
    return {"raw_output_sha256": hashlib.sha256(bound).hexdigest(),
            "raw_output_bytes": size,
            **({"raw_stdout": stdout, "raw_stderr": stderr}
               if size <= MAX_OUTPUT_BYTES else {})}


def collect(plan: Plan, *, invoke: Callable, env: Mapping[str, str],
            timeout_s: float = 1800.0) -> dict:
    """Run a new seeded op suite outside ranked timing and retain exact output.

    ``invoke(argv, env, timeout_s)`` is supplied by the resource-owning caller;
    this helper has no subprocess fallback and cannot acquire compute itself.
    ``reference_valid`` is restricted to a GPU-only kernel diff with a reviewed
    identical test instrument and a structured reference on every compared case.
    For CPU source work, independent host-double properties can support only
    ``property_only``; the candidate's own CPU reference cannot certify itself.
    """
    try:
        binary_sha = _sha256(plan.binary)
    except OSError as exc:
        return _receipt(plan, binary_sha256=None, status="oracle_unavailable",
                        reason=f"binary unreadable: {type(exc).__name__}: {exc}")
    if binary_sha != plan.expected_binary_sha256:
        return _receipt(plan, binary_sha256=binary_sha,
                        status="oracle_unavailable",
                        reason="selected binary differs from its predeclared build identity",
                        expected_binary_sha256=plan.expected_binary_sha256)
    source_check = instrument_integrity.compare_to_anchor(
        tool="test-backend-ops", candidate_root=str(plan.candidate_root),
        anchor_root=str(plan.anchor_root))
    if source_check.outcome != schemas.PASS:
        return _receipt(plan, binary_sha256=binary_sha,
                        status="oracle_unavailable",
                        reason="test instrument source is not the reviewed anchor",
                        instrument_source_outcome=source_check.outcome,
                        instrument_source_reasons=list(source_check.reasons))
    launch_env = dict(env)
    # Help capability detection must resolve the same build-local ggml libraries
    # as the suite; ambient LD_LIBRARY_PATH can silently pick frozen production.
    launch_env["LD_LIBRARY_PATH"] = str(plan.binary.parent)
    help_text = ""
    try:
        help_run = invoke((str(plan.binary), "--help"), launch_env, timeout_s)
        # This tool conventionally exits 1 after printing --help. A strict
        # Usage banner and the named flags, not that exit code, prove support.
        help_text = str(help_run.stdout) + str(help_run.stderr)
        if len(help_text.encode()) > MAX_OUTPUT_BYTES:
            raise ValueError("selected binary help exceeds the bounded evidence carrier")
        capabilities = t0_provider.parse_backend_ops_help(help_text)
        capabilities.require(("--suite-seed", "--autokernel-properties"))
    except (OSError, ValueError, TypeError, AttributeError,
            subprocess.TimeoutExpired, t0_provider.InstrumentCapabilityError) as exc:
        return _receipt(plan, binary_sha256=binary_sha,
                        status="oracle_unavailable",
                        reason=f"selected binary cannot prove seeded suite capability: {exc}",
                        help_sha256=hashlib.sha256(help_text.encode()).hexdigest(),
                        **({"help_text": help_text}
                           if len(help_text.encode()) <= MAX_OUTPUT_BYTES else {}))

    stdout = stderr = ""
    try:
        constructed = t0_provider.build_backend_ops_invocation(
            binary=str(plan.binary), library_path=str(plan.binary.parent),
            backend_filter=plan.backend, ops=(plan.op,),
            base_env=tuple(sorted(launch_env.items())), suite_seed=plan.suite_seed,
            parallel_workers=1, cpu_prefix=False, capabilities=capabilities)
        completed = invoke(constructed.argv, dict(constructed.env), timeout_s)
        stdout = str(completed.stdout)
        stderr = str(completed.stderr)
        if len((stdout + stderr).encode()) > MAX_OUTPUT_BYTES:
            raise ValueError("seeded suite output exceeds the bounded evidence carrier")
        parsed = t0_provider.parse_backend_ops_console(stdout + stderr)
        parsed.reconcile()
    except (OSError, ValueError, TypeError, AttributeError,
            subprocess.TimeoutExpired, t0_provider.OutputParseError) as exc:
        return _receipt(plan, binary_sha256=binary_sha,
                        status="oracle_unavailable",
                        reason=f"seeded suite unavailable or unreadable: {exc}",
                        **_raw_facts(stdout, stderr))

    frames = [frame for frame in parsed.backends
              if frame.name == plan.backend and not frame.skipped]
    cases = [case for frame in frames for case in frame.cases
             if case.op == plan.op and case.status != "not_supported"]
    refs = [case for case in cases if case.reference is not None]
    props = [prop for case in cases for prop in case.properties
             if prop.suite_seed == plan.suite_seed]
    tool_ok = (completed.returncode == 0 and parsed.overall == "OK"
               and bool(cases) and all(case.passed for case in cases))
    paths = (() if plan.candidate_integrity is None
             else plan.candidate_integrity.paths)
    independent_reference = (
        re.fullmatch(r"ROCm[0-9]+", plan.backend) is not None and bool(paths)
        and all(path.startswith("ggml/src/ggml-cuda/") for path in paths)
        and source_check.outcome == schemas.PASS)
    if independent_reference and cases and len(refs) == len(cases):
        status = "reference_valid"
    elif props:
        status = "property_only"
    else:
        status = "oracle_unavailable"
    residuals_ok = all(ref.reference.observed <= ref.reference.tolerance
                       for ref in refs)
    property_ok = all(prop.passed for prop in props)
    verdict = ("passed" if tool_ok and residuals_ok and property_ok else
               "failed" if cases else "unavailable")
    if status == "oracle_unavailable":
        # A failed process without independent reference/property evidence is a
        # setup/suite fact, not a proved candidate correctness failure.
        verdict = "unavailable"
    return _receipt(
        plan, binary_sha256=binary_sha, status=status, verdict=verdict,
        reason=("fresh seeded operation cases captured; not model-token parity"
                if status != "oracle_unavailable" else
                "no independent structured reference or seeded host property was captured"),
        instrument_source_outcome=source_check.outcome,
        candidate_tree=(None if plan.candidate_integrity is None
                        else plan.candidate_integrity.tree),
        compared_cases=len(cases), reference_cases=len(refs),
        property_checks=len(props), suite_exit_code=completed.returncode,
        suite_overall=parsed.overall,
        help_sha256=hashlib.sha256(help_text.encode()).hexdigest(),
        help_text=help_text,
        argv=list(constructed.argv),
        invocation_receipt=constructed.receipt.to_dict(),
        **_raw_facts(stdout, stderr))
