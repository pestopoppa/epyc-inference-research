"""Typed pre-model admissibility enforcement (AK-ADM-1).

The six planner-memory limitations of the v2 portfolio (exact kernel
instantiation, command phase, static-tool execution, Q4_K MMQ baseline
correctness, Q8 native-path premise, IQ serving residency) used to live only as
prose the planner/critic models were asked to respect.  This module turns them
into a versioned, self-hashed manifest plus deterministic predicates the
controller evaluates *before* any model judgment (the Fable critic).  A
violation is a typed refusal, never a model opinion.

Planner proposals may carry ``proposal["admissibility_facts"]`` (schema in
``FACTS_SCHEMA``).  A policy applies when the candidate's patch/target touches
its surface; when it applies, the required facts must be present and valid or
the candidate is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .. import schemas

SCHEMA = "epyc.autokernel.admissibility_policy.v1"
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "discovery_admissibility_policy_v1.json"

EXACT_INSTANTIATION = "ak-admit-exact-kernel-instantiation-v1"
COMMAND_PHASE = "ak-admit-command-phase-consistency-v1"
STATIC_TOOL = "ak-admit-static-tool-execution-v1"
Q4K_BASELINE = "ak-admit-q4k-mmq-baseline-correctness-v1"
Q8_PREMISE = "ak-admit-q8-native-source-premise-v1"
IQ_RESIDENCY = "ak-trigger-iq-serving-residency-v1"
POLICY_IDS = frozenset({EXACT_INSTANTIATION, COMMAND_PHASE, STATIC_TOOL,
                        Q4K_BASELINE, Q8_PREMISE, IQ_RESIDENCY})

FACTS_SCHEMA: dict[str, Any] = {
    "location": "proposal.admissibility_facts (optional mapping; required when a policy applies)",
    "occupancy_evidence": {"keys": ["instantiation", "production_instantiation"],
                           "rule": "both strings; must be equal and must not be synthetic"},
    "phase_claim": {"keys": ["phase", "command_argv"],
                    "rule": "phase in {prefill, decode}; decode requires -n/--n-predict > 0"},
    "static_scans": {"items": ["tool", "tool_resolved", "argv", "exit_code", "stderr_suppressed", "negative"],
                     "rule": "a negative scan needs tool_resolved true, exit_code 0, stderr_suppressed false"},
    "q4k_mmq_baseline": {"keys": ["forced_mmq_pass", "forced_mmq_total"],
                         "rule": "required for Q4_K MMQ targets; must be fully correctness-clean"},
    "q8_premise": {"values": ["integer_native_acknowledged"],
                   "rule": "required for Q8_0 MMVQ targets; FP-dequant-hiding premises are refused"},
    "iq_claim": {"values": ["latent_diagnostic", "live_payoff"],
                 "rule": "required for IQ-format targets; live_payoff refused while no IQ role is serving-resident"},
}


class AdmissibilityError(RuntimeError):
    """The manifest itself is malformed or unbound."""


@dataclass(frozen=True)
class Refusal:
    policy_id: str
    evidence_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"policy_id": self.policy_id, "evidence_id": self.evidence_id,
                "reason": self.reason}


@dataclass(frozen=True)
class PolicyManifest:
    body: Mapping[str, Any]
    sha256: str

    @property
    def policies(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.body["admissibility_policies"])

    @property
    def iq_serving_resident(self) -> bool:
        return bool(self.body["iq_serving_resident"])

    def evidence_for(self, policy_id: str) -> str:
        for row in self.policies:
            if row["policy_id"] == policy_id:
                return row["evidence_id"]
        raise AdmissibilityError(f"unknown policy {policy_id}")

    def context_projection(self) -> dict[str, Any]:
        return {"admissibility_policies": json.loads(json.dumps(self.body)),
                "admissibility_policy_sha256": self.sha256}


def validate(body: object, *, portfolio_evidence: Mapping[str, str] | None = None) -> PolicyManifest:
    if not isinstance(body, Mapping) or set(body) != {
            "schema", "version", "iq_serving_resident", "admissibility_policies",
            "manifest_sha256"}:
        raise AdmissibilityError("admissibility policy manifest has an unknown schema")
    if body["schema"] != SCHEMA or body["version"] != 1:
        raise AdmissibilityError("admissibility policy schema/version mismatch")
    if not isinstance(body["iq_serving_resident"], bool):
        raise AdmissibilityError("iq_serving_resident must be boolean")
    expected = schemas.content_hash({k: v for k, v in body.items() if k != "manifest_sha256"})
    if body["manifest_sha256"] != expected:
        raise AdmissibilityError("admissibility policy self-hash mismatch")
    rows = body["admissibility_policies"]
    if not isinstance(rows, list) or any(
            not isinstance(row, Mapping)
            or set(row) != {"policy_id", "kind", "evidence_id", "evidence_sha256", "rule"}
            or row["kind"] not in {"admit", "trigger"} for row in rows):
        raise AdmissibilityError("admissibility policy rows are malformed")
    ids = [row["policy_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != POLICY_IDS:
        raise AdmissibilityError("admissibility policy set is not exactly the six typed policies")
    if portfolio_evidence is not None:
        for row in rows:
            if portfolio_evidence.get(row["evidence_id"]) != row["evidence_sha256"]:
                raise AdmissibilityError(
                    f"policy {row['policy_id']} is not bound to sealed portfolio evidence")
    return PolicyManifest(body=body, sha256=expected)


def load(path: Path = DEFAULT_POLICY_PATH, *,
         portfolio_evidence: Mapping[str, str] | None = None) -> PolicyManifest:
    if path.is_symlink() or not path.is_file():
        raise AdmissibilityError("admissibility policy must be a regular non-symlink file")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissibilityError("admissibility policy is not JSON") from exc
    return validate(body, portfolio_evidence=portfolio_evidence)


_Q4K_MMQ = re.compile(r"q4_?k", re.I)
_MMQ = re.compile(r"\bmmq\b|mul_mat_q\b|mmq\.cuh|mmq\.cu", re.I)
_Q8_MMVQ = re.compile(r"vec_dot_q8_0_q8_1|q8_0.*mmvq|mmvq.*q8_0", re.I | re.S)
_IQ = re.compile(r"\biq[1-4]_?(?:xxs|xs|s|m|nl)?\b|iq[1-4]_", re.I)
_OCCUPANCY = re.compile(r"\bvgpr\b|occupancy|waves? per simd", re.I)
_N_FLAG = {"-n", "--n-predict", "--predict", "-n-gen", "--n-gen"}


def _surface_text(*, hypothesis_id: str, target_symbol: str | None,
                  declared_files: Iterable[str], patch_text: str, statement: str) -> str:
    return "\n".join([hypothesis_id, target_symbol or "", *declared_files, patch_text, statement])


def _generated_tokens(argv: object) -> int | None:
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return None
    for index, item in enumerate(argv):
        if item in _N_FLAG and index + 1 < len(argv):
            try:
                return int(argv[index + 1])
            except ValueError:
                return None
        for flag in _N_FLAG:
            if item.startswith(flag + "="):
                try:
                    return int(item.split("=", 1)[1])
                except ValueError:
                    return None
    return None


def enforce(manifest: PolicyManifest, *, hypothesis_id: str, statement: str,
            proposal: Mapping[str, Any], target_symbol: str | None,
            declared_files: Iterable[str], patch_text: str) -> list[Refusal]:
    """Return every typed refusal for one candidate; empty list means admissible."""
    files = list(declared_files)
    text = _surface_text(hypothesis_id=hypothesis_id, target_symbol=target_symbol,
                         declared_files=files, patch_text=patch_text, statement=statement)
    facts = proposal.get("admissibility_facts", {})
    refusals: list[Refusal] = []

    def refuse(policy_id: str, reason: str) -> None:
        # Recorded "refuse" before any model judgment; see discovery_controller.
        refusals.append(Refusal(policy_id, manifest.evidence_for(policy_id), reason))

    if not isinstance(facts, Mapping):
        refuse(EXACT_INSTANTIATION, "admissibility_facts must be a mapping")
        return refusals
    unknown = set(facts) - {key for key in FACTS_SCHEMA if key != "location"}
    if unknown:
        refuse(EXACT_INSTANTIATION, f"unknown admissibility_facts keys {sorted(unknown)}")

    # 1. Exact kernel instantiation.
    occupancy = facts.get("occupancy_evidence")
    if occupancy is not None or _OCCUPANCY.search(statement):
        if (not isinstance(occupancy, Mapping)
                or not isinstance(occupancy.get("instantiation"), str)
                or not isinstance(occupancy.get("production_instantiation"), str)):
            refuse(EXACT_INSTANTIATION,
                   "occupancy/VGPR claim lacks exact instantiation binding")
        elif ("synthetic" in occupancy["instantiation"].lower()
              or occupancy["instantiation"] != occupancy["production_instantiation"]):
            refuse(EXACT_INSTANTIATION,
                   "evidence instantiation differs from the production instantiation")

    # 2. Command phase.
    phase = facts.get("phase_claim")
    if phase is not None:
        if not isinstance(phase, Mapping) or phase.get("phase") not in {"prefill", "decode"}:
            refuse(COMMAND_PHASE, "phase_claim is malformed")
        elif phase["phase"] == "decode":
            generated = _generated_tokens(phase.get("command_argv"))
            if generated is None or generated <= 0:
                refuse(COMMAND_PHASE,
                       "decode claim requires an exact command with generated tokens > 0")

    # 3. Static-tool execution.
    scans = facts.get("static_scans", [])
    if not isinstance(scans, list):
        refuse(STATIC_TOOL, "static_scans must be a list")
        scans = []
    for scan in scans:
        if not isinstance(scan, Mapping) or not isinstance(scan.get("argv"), list):
            refuse(STATIC_TOOL, "static scan lacks exact argv")
            continue
        if scan.get("negative") is True and (
                scan.get("tool_resolved") is not True or scan.get("exit_code") != 0
                or scan.get("stderr_suppressed") is not False):
            refuse(STATIC_TOOL,
                   f"negative scan by {scan.get('tool')!r} is tool silence, not evidence")

    # 4. Q4_K MMQ baseline correctness.
    if _Q4K_MMQ.search(text) and _MMQ.search(text):
        baseline = facts.get("q4k_mmq_baseline")
        if (not isinstance(baseline, Mapping)
                or not isinstance(baseline.get("forced_mmq_total"), int)
                or baseline.get("forced_mmq_total", 0) <= 0
                or baseline.get("forced_mmq_pass") != baseline.get("forced_mmq_total")):
            refuse(Q4K_BASELINE,
                   "Q4_K MMQ target without a correctness-clean forced-MMQ baseline")

    # 5. Q8 native-path premise.
    if _Q8_MMVQ.search(text):
        if facts.get("q8_premise") != "integer_native_acknowledged":
            refuse(Q8_PREMISE,
                   "Q8_0 MMVQ target must acknowledge the integer-native dp4a premise")

    # 6. IQ serving residency.
    if _IQ.search(text):
        claim = facts.get("iq_claim")
        if claim not in {"latent_diagnostic", "live_payoff"}:
            refuse(IQ_RESIDENCY, "IQ-format target must declare iq_claim")
        elif claim == "live_payoff" and not manifest.iq_serving_resident:
            refuse(IQ_RESIDENCY,
                   "live-payoff IQ claim refused: no IQ role is ROCm0 serving-resident")
    return refusals
