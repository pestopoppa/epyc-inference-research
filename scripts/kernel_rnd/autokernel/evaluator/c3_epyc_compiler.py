#!/usr/bin/env python3
"""JSON compiler for the prospective INF-48 C3/EPYC evaluator seam.

``plan`` emits the exact three-case contract and runner bindings. ``receipt``
accepts hash-bound observations and reduces them through :mod:`c3_epyc_suite`.
The module executes no benchmark, inference, build, profiler, or hot patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import schemas
from . import c3_epyc_suite as c3


PLAN_SCHEMA = "epyc.autokernel.c3_epyc_plan.v1"
INPUT_SCHEMA = "epyc.autokernel.c3_epyc_observations.v1"
RECEIPT_SCHEMA = "epyc.autokernel.c3_epyc_receipt.v1"
COMPILER_ID = "autokernel.evaluator.c3_epyc_compiler/v1"

RUNNER_BINDINGS = {
    c3.APEX_PYTHON_OVERLAY: {
        "source": "https://github.com/AMD-AGI/Apex",
        "revision": c3.PINNED_APEX_REVISION,
        "applies_to": [
            "epyc.attention.mla_paged_prefill.k228",
            "epyc.moe.sparse_expert_dispatch.k175",
        ],
        "capture": {
            "cli": "python3 workload_optimizer.py trace-kernel",
            "python_entrypoint": "pipeline.kernel_tracing.runner.run_trace_kernel",
            "outputs": ["trace_result.json", "workload_ranges.json",
                        "patched_files/patch_manifest.json"],
        },
        "integration": {
            "cli": "python3 workload_optimizer.py integrate",
            "python_entrypoint": "workload_optimizer.cmd_integrate->_reinject_kernel",
        },
        "whole_model": {
            "cli": "python3 workload_optimizer.py benchmark-final",
            "python_entrypoint": (
                "workload_optimizer.cmd_benchmark_final->_run_final_benchmark"
            ),
        },
        "boundary": "Python/Triton overlays only; no system C++ or monolithic binary patch",
    },
    c3.EPYC_EXPERIMENTAL_BINARY: {
        "source": "epyc-inference-research",
        "revision": "receipt_supplied_source_commit",
        "applies_to": ["epyc.dequant.q4_k_decode_gemv"],
        "capture": {
            "cli": "runner_supplied_captured_tensor_manifest",
            "python_entrypoint": "receipt_input_only",
            "outputs": ["captured_tensor_manifest.json"],
        },
        "integration": {
            "cli": "select immutable experimental candidate binary",
            "python_entrypoint": "autokernel.evaluator.recipes.construct",
        },
        "whole_model": {
            "cli": "t1b.llama_gpu.llama_bench_decode.v1",
            "python_entrypoint": "autokernel.execution.microbench",
        },
        "boundary": "experimental tree/binary only; never patches frozen production",
    },
}


class C3CompilerError(c3.C3ContractError):
    """An input document cannot be compiled into a governed receipt."""


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise C3CompilerError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise C3CompilerError(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if not schemas.SHA256_RE.fullmatch(value) or schemas.is_placeholder_digest(value):
        raise C3CompilerError(f"{label} must be a non-placeholder lowercase SHA-256")
    return value


def _exact_keys(payload: Mapping[str, Any], *, required: set[str],
                optional: set[str] = frozenset(), label: str) -> None:
    missing = required - set(payload)
    unknown = set(payload) - required - optional
    if missing or unknown:
        raise C3CompilerError(
            f"{label} fields differ from schema; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")


def _policy(fast_p_threshold: float, whole_model_minimum_speedup: float) -> dict[str, float]:
    for value, label in ((fast_p_threshold, "fast_p_threshold"),
                         (whole_model_minimum_speedup,
                          "whole_model_minimum_speedup")):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 1.0):
            raise C3CompilerError(f"{label} must be numeric and at least 1.0")
    return {
        "fast_p_threshold": float(fast_p_threshold),
        "whole_model_minimum_speedup": float(whole_model_minimum_speedup),
    }


def compile_plan(*, fast_p_threshold: float = 1.0,
                 whole_model_minimum_speedup: float = 1.0) -> dict[str, Any]:
    """Emit the canonical non-numeric plan callable by a controller/backend."""
    policy = _policy(fast_p_threshold, whole_model_minimum_speedup)
    cases = c3.epyc_op_suite()
    case_rows = []
    for case in cases:
        row = case.to_dict()
        row["runner_binding_id"] = (
            c3.EPYC_EXPERIMENTAL_BINARY
            if case.case_id == "epyc.dequant.q4_k_decode_gemv"
            else c3.APEX_PYTHON_OVERLAY)
        case_rows.append(row)
    document: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "compiler_id": COMPILER_ID,
        "compiler_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract_source_sha256": hashlib.sha256(
            Path(c3.__file__).read_bytes()).hexdigest(),
        "authority": c3.SEARCH_EXIT_AUTHORITY,
        "promotion_authorized": False,
        "policy": policy,
        "cases": case_rows,
        "runner_bindings": json.loads(json.dumps(RUNNER_BINDINGS)),
        "external_artifacts": [item.__dict__
                               for item in c3.external_artifact_requirements()],
        "receipt_input_schema": INPUT_SCHEMA,
    }
    document["plan_sha256"] = _digest(document)
    return document


@dataclass(frozen=True)
class EvidenceCheck:
    check: schemas.Check
    evidence_ref: str
    evidence_sha256: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> "EvidenceCheck":
        payload = _mapping(payload, label)
        _exact_keys(payload, required={"outcome", "reasons", "evidence_ref",
                                       "evidence_sha256"}, label=label)
        outcome = _text(payload.get("outcome"), f"{label}.outcome")
        reasons = payload.get("reasons")
        if (not isinstance(reasons, list)
                or any(not isinstance(item, str) or not item for item in reasons)):
            raise C3CompilerError(f"{label}.reasons must be a string list")
        if outcome != schemas.PASS and not reasons:
            raise C3CompilerError(f"{label} non-PASS outcome requires a reason")
        return cls(
            schemas.Check(outcome, tuple(reasons)),
            _text(payload.get("evidence_ref"), f"{label}.evidence_ref"),
            _sha(payload.get("evidence_sha256"), f"{label}.evidence_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.check.outcome,
            "reasons": list(self.check.reasons),
            "evidence_ref": self.evidence_ref,
            "evidence_sha256": self.evidence_sha256,
        }


def _surface(payload: Mapping[str, Any]) -> c3.ExactOpSurface:
    payload = _mapping(payload, "surface")
    required = {"case_id", "device_id", "model_sha256", "quant", "operation",
                "shape", "dtype", "tensor_manifest_sha256", "recipe_id",
                "recipe_sha256", "harness_build_sha256", "factors"}
    _exact_keys(payload, required=required, label="surface")
    factors = _mapping(payload["factors"], "surface.factors")
    return c3.ExactOpSurface.create(
        case_id=payload["case_id"], device_id=payload["device_id"],
        model_sha256=payload["model_sha256"], quant=payload["quant"],
        operation=payload["operation"], shape=payload["shape"], dtype=payload["dtype"],
        tensor_manifest_sha256=payload["tensor_manifest_sha256"],
        recipe_id=payload["recipe_id"], recipe_sha256=payload["recipe_sha256"],
        harness_build_sha256=payload["harness_build_sha256"], factors=factors)


def _timing(payload: Mapping[str, Any], surface: c3.ExactOpSurface,
            label: str) -> c3.TimingObservation:
    payload = _mapping(payload, label)
    _exact_keys(payload, required={"provider", "implementation_sha256", "samples_ns",
                                   "evidence_ref", "evidence_sha256"}, label=label)
    return c3.TimingObservation(
        provider=payload["provider"], surface=surface,
        implementation_sha256=payload["implementation_sha256"],
        samples_ns=payload["samples_ns"], evidence_ref=payload["evidence_ref"],
        evidence_sha256=payload["evidence_sha256"])


def _gate_dict(gate: c3.FastPGate) -> dict[str, Any]:
    return {
        "case_id": gate.case_id, "p": gate.p, "speedup": gate.speedup,
        "check": {"outcome": gate.check.outcome, "reasons": list(gate.check.reasons)},
        "baseline_provider": gate.baseline_provider,
        "baseline_evidence_ref": gate.baseline_evidence_ref,
        "candidate_evidence_ref": gate.candidate_evidence_ref,
        "candidate_implementation_sha256": gate.candidate_implementation_sha256,
    }


def _compile_case(case: c3.EpycOpCase, payload: Mapping[str, Any], *,
                  p: float) -> tuple[c3.FastPGate, dict[str, Any]]:
    payload = _mapping(payload, f"case {case.case_id}")
    case_id = _text(payload.get("case_id"), "case.case_id")
    if case_id != case.case_id:
        raise C3CompilerError("case observation order/identity differs from the plan")
    state = _text(payload.get("state"), f"{case_id}.state")
    if state == "unavailable":
        _exact_keys(payload, required={"case_id", "state", "reason"}, label=case_id)
        reason = _text(payload["reason"], f"{case_id}.reason")
        gate = c3.FastPGate(
            case_id, p, None, schemas.Check(schemas.COULD_NOT_CHECK, (reason,)),
            None, None, None, None)
        return gate, {"case_id": case_id, "state": state, "reason": reason,
                      "gate": _gate_dict(gate)}
    if state != "observed":
        raise C3CompilerError(f"{case_id}.state must be observed or unavailable")
    required = {"case_id", "state", "surface", "vendor_observations",
                "candidate_observation", "correctness", "integrity"}
    _exact_keys(payload, required=required, label=case_id)
    surface = _surface(payload["surface"])
    if surface.case_id != case_id:
        raise c3.IdentityMismatch("case row and surface name different tasks")
    vendor_rows = payload["vendor_observations"]
    if not isinstance(vendor_rows, list):
        raise C3CompilerError("vendor_observations must be a list")
    vendor = tuple(_timing(row, surface, "vendor observation") for row in vendor_rows)
    candidate = _timing(payload["candidate_observation"], surface,
                        "candidate observation")
    correctness = EvidenceCheck.from_dict(payload["correctness"], "correctness")
    integrity = EvidenceCheck.from_dict(payload["integrity"], "integrity")
    floor = c3.select_vendor_floor(case, surface, vendor)
    gate = c3.score_fast_p(
        floor=floor, candidate=candidate, p=p,
        correctness=correctness.check, integrity=integrity.check)
    return gate, {
        "case_id": case_id, "state": state, "gate": _gate_dict(gate),
        "evidence": {
            "vendor": [{"provider": row.provider, "evidence_ref": row.evidence_ref,
                        "evidence_sha256": row.evidence_sha256} for row in vendor],
            "candidate": {"evidence_ref": candidate.evidence_ref,
                          "evidence_sha256": candidate.evidence_sha256},
            "correctness": correctness.to_dict(), "integrity": integrity.to_dict(),
        },
    }


def _whole_surface(payload: Mapping[str, Any]) -> c3.WholeModelSurface:
    payload = _mapping(payload, "whole_model.surface")
    _exact_keys(payload, required={"workload", "device_id", "quant", "recipe_id",
                                   "recipe_sha256", "factors"},
                label="whole_model.surface")
    workload = _mapping(payload["workload"], "whole_model.surface.workload")
    _exact_keys(workload, required={"workload_id", "model_sha256",
                                    "tensor_manifest_sha256", "capture_receipt_ref",
                                    "capture_receipt_sha256"},
                label="whole_model.surface.workload")
    captured = c3.CapturedWorkload(**workload)
    return c3.WholeModelSurface.create(
        workload=captured, device_id=payload["device_id"], quant=payload["quant"],
        recipe_id=payload["recipe_id"], recipe_sha256=payload["recipe_sha256"],
        factors=_mapping(payload["factors"], "whole_model.surface.factors"))


def _integration(payload: Mapping[str, Any]) -> c3.CandidateIntegrationBinding:
    payload = _mapping(payload, "whole_model.integration")
    required = {"runner_id", "runner_revision", "patch_bundle_sha256",
                "candidate_source_sha256", "candidate_build_sha256",
                "candidate_binary_sha256", "receipt_ref", "receipt_sha256"}
    _exact_keys(payload, required=required, label="whole_model.integration")
    return c3.CandidateIntegrationBinding(**payload)


def _whole_observation(payload: Mapping[str, Any], surface: c3.WholeModelSurface,
                       label: str) -> c3.WholeModelObservation:
    payload = _mapping(payload, label)
    _exact_keys(payload, required={"arm", "build_sha256", "binary_sha256",
                                   "samples_ns", "evidence_ref", "evidence_sha256"},
                label=label)
    return c3.WholeModelObservation(surface=surface, **payload)


def _compile_whole(payload: Mapping[str, Any], gates: Mapping[str, c3.FastPGate], *,
                   minimum_speedup: float) -> dict[str, Any]:
    payload = _mapping(payload, "whole_model")
    state = _text(payload.get("state"), "whole_model.state")
    if state == "unavailable":
        _exact_keys(payload, required={"state", "reason"}, label="whole_model")
        reason = _text(payload["reason"], "whole_model.reason")
        report = c3.WholeModelExitReport(
            None, minimum_speedup,
            schemas.Check(schemas.COULD_NOT_CHECK, (reason,)))
        return _whole_report(report, state=state, target_case_id=None)
    if state != "observed":
        raise C3CompilerError("whole_model.state must be observed or unavailable")
    required = {"state", "target_case_id", "surface", "integration", "anchor",
                "candidate", "correctness", "integrity"}
    _exact_keys(payload, required=required, label="whole_model")
    target = _text(payload["target_case_id"], "whole_model.target_case_id")
    if target not in gates:
        raise C3CompilerError("whole-model target is not one of the exact suite cases")
    surface = _whole_surface(payload["surface"])
    integration = _integration(payload["integration"])
    expected_runner = (c3.EPYC_EXPERIMENTAL_BINARY
                       if target == "epyc.dequant.q4_k_decode_gemv"
                       else c3.APEX_PYTHON_OVERLAY)
    if integration.runner_id != expected_runner:
        raise C3CompilerError("whole-model integration runner conflicts with target case")
    anchor = _whole_observation(payload["anchor"], surface, "whole_model.anchor")
    candidate = _whole_observation(
        payload["candidate"], surface, "whole_model.candidate")
    correctness = EvidenceCheck.from_dict(
        payload["correctness"], "whole_model.correctness")
    integrity = EvidenceCheck.from_dict(payload["integrity"], "whole_model.integrity")
    report = c3.evaluate_whole_model_exit(
        operator_gate=gates[target], integration=integration, anchor=anchor,
        candidate=candidate, correctness=correctness.check, integrity=integrity.check,
        minimum_speedup=minimum_speedup)
    row = _whole_report(report, state=state, target_case_id=target)
    row["evidence"] = {
        "integration": {"receipt_ref": integration.receipt_ref,
                        "receipt_sha256": integration.receipt_sha256},
        "anchor": {"evidence_ref": anchor.evidence_ref,
                   "evidence_sha256": anchor.evidence_sha256},
        "candidate": {"evidence_ref": candidate.evidence_ref,
                      "evidence_sha256": candidate.evidence_sha256},
        "correctness": correctness.to_dict(), "integrity": integrity.to_dict(),
    }
    return row


def _whole_report(report: c3.WholeModelExitReport, *, state: str,
                  target_case_id: str | None) -> dict[str, Any]:
    return {
        "state": state, "target_case_id": target_case_id,
        "speedup": report.speedup, "minimum_speedup": report.minimum_speedup,
        "check": {"outcome": report.check.outcome,
                  "reasons": list(report.check.reasons)},
        "authority": report.authority,
        "promotion_authorized": report.promotion_authorized,
        "authority_boundary": report.authority_boundary,
    }


def compile_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one JSON observation document and derive its offline receipt."""
    payload = _mapping(payload, "receipt input")
    required = {"schema", "plan_sha256", "policy", "cases", "whole_model"}
    _exact_keys(payload, required=required, label="receipt input")
    if payload.get("schema") != INPUT_SCHEMA:
        raise C3CompilerError(f"receipt input schema must be {INPUT_SCHEMA}")
    policy_doc = _mapping(payload["policy"], "policy")
    _exact_keys(policy_doc, required={"fast_p_threshold",
                                      "whole_model_minimum_speedup"}, label="policy")
    policy = _policy(policy_doc["fast_p_threshold"],
                     policy_doc["whole_model_minimum_speedup"])
    plan = compile_plan(**policy)
    if _sha(payload["plan_sha256"], "plan_sha256") != plan["plan_sha256"]:
        raise C3CompilerError("receipt input is bound to a different plan")
    case_payloads = payload["cases"]
    cases = c3.epyc_op_suite()
    if not isinstance(case_payloads, list) or len(case_payloads) != len(cases):
        raise C3CompilerError("receipt input requires the exact three-case suite")
    gates: list[c3.FastPGate] = []
    case_rows: list[dict[str, Any]] = []
    for case, case_payload in zip(cases, case_payloads, strict=True):
        gate, row = _compile_case(
            case, case_payload, p=policy["fast_p_threshold"])
        gates.append(gate)
        case_rows.append(row)
    suite = c3.aggregate_fast_p(cases, gates, p=policy["fast_p_threshold"])
    by_id = {gate.case_id: gate for gate in gates}
    whole = _compile_whole(
        payload["whole_model"], by_id,
        minimum_speedup=policy["whole_model_minimum_speedup"])
    input_sha = _digest(dict(payload))
    document: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "compiler_id": COMPILER_ID,
        "plan_sha256": plan["plan_sha256"], "input_sha256": input_sha,
        "authority": c3.SEARCH_EXIT_AUTHORITY, "promotion_authorized": False,
        "policy": policy, "cases": case_rows,
        "fast_p": {"value": suite.fast_p, "p": suite.p,
                   "admitted_cases": suite.admitted_cases,
                   "scored_cases": suite.scored_cases,
                   "total_cases": suite.total_cases},
        "whole_model_exit": whole,
    }
    document["receipt_sha256"] = _digest(document)
    return document


class C3EpycBackend:
    """Small direct-call seam for a controller or external backend adapter."""

    def compile_plan(self, *, fast_p_threshold: float = 1.0,
                     whole_model_minimum_speedup: float = 1.0) -> dict[str, Any]:
        return compile_plan(
            fast_p_threshold=fast_p_threshold,
            whole_model_minimum_speedup=whole_model_minimum_speedup)

    def compile_receipt(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return compile_receipt(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="emit the exact non-numeric suite plan")
    plan_parser.add_argument("--fast-p", type=float, default=1.0)
    plan_parser.add_argument("--whole-model-min-speedup", type=float, default=1.0)
    receipt_parser = sub.add_parser("receipt", help="compile completed observation JSON")
    receipt_parser.add_argument("--input", default="-", help="JSON path or - for stdin")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = compile_plan(
            fast_p_threshold=args.fast_p,
            whole_model_minimum_speedup=args.whole_model_min_speedup)
    else:
        text = (sys.stdin.read() if args.input == "-" else
                Path(args.input).read_text(encoding="utf-8"))
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise C3CompilerError("receipt input is not valid JSON") from exc
        result = compile_receipt(payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPILER_ID", "INPUT_SCHEMA", "PLAN_SCHEMA", "RECEIPT_SCHEMA",
    "RUNNER_BINDINGS", "C3CompilerError", "EvidenceCheck", "C3EpycBackend",
    "compile_plan", "compile_receipt", "main",
]
