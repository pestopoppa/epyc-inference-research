#!/usr/bin/env python3
"""t3.py — the T3 kernel-freeze gate runner and the release bundle (design §10).

WHY THIS MODULE EXISTS
----------------------
T3 is the one instrument that decides whether a sealed champion is a releasable
new kernel version. Everything upstream of it is a *search* record under
`P-AK-SEARCH-1`, which says in terms that it *"does NOT apply to T3 or any
release gate"*; `evaluator.api.admit_tier()` refuses T3 by name so that a
release-shaped decision can never be produced under a search protocol. This
module is the other side of that refusal.

It prevents four specific failures, each of which this project has already paid
for at least once:

  * **A binary PASS/FAIL gate would have blocked v8.** The v8 ratification
    records `promotion_decision: false`, *preserved as "a non-automatic matrix
    verdict"*, released as *"an operator-attested release decision"*, with
    `q8_claim: "none; campaign-scoped WAIVE-Q8 remains binding and v8 makes no Q8
    non-regression claim"*. So T3 emits **PASS / FAIL / PASS_WITH_WAIVER**, a
    waiver is a first-class hash-pinned input (§10.4), and a waived cell
    **suppresses its claim in the receipt** exactly as v8 suppressed its Q8
    claim. The evaluator verifies a waiver's HASH and PREDICATE. It never judges
    its merits — that judgement is the operator's, and it is the whole reason the
    waiver exists.

  * **A rebuilt incumbent is not the incumbent.** The v8 quality gate compared
    against a PRESERVED binary at
    `/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/llama-server`.
    Rebuilding an old commit under a drifted toolchain does not reproduce it, and
    `/mnt/raid0/llm/kernels/archive/` is empty. §10.5: the freeze transaction
    ARCHIVES the incumbent's built binaries **and linked libraries** for N-1 and
    ideally N-2, and a quality baseline that names a rebuild rather than an
    archived build FAILS here rather than being quietly accepted.

  * **A binary that inherits another tree's ggml runs silently wrong.** Three
    ggml generations coexist on this host (llama 0.16.0, qwentts 0.17.0, whisper
    0.18.0). On 2026-07-31 a HIP-built `whisper-cli` loaded the production
    CPU-only ggml, found no GPU, and ran full-CPU while printing `use gpu = 1`
    (INC-20260731-ggml-linkage-silent-cpu-fallback). Phase 2 therefore requires a
    receipt from the research repo's `scripts/utils/verify_ggml_linkage.sh` — and
    does **not** inherit that script's one fail-open: it exits 0 when `ldd`
    reports no ggml libraries at all, which this module treats as
    COULD_NOT_CHECK, never PASS.

  * **An expensive gate re-run on evidence that did not change.** §9.1: *"Run T3
    once per sealed fingerprint. A retry requires a new evidence-affecting
    fingerprint or a deterministic replay/repair of the failed stage."* §12 lists
    *"full release evaluation loops repeatedly"* against *"sealed-fingerprint
    idempotence and failed-gate cooldown"*. Both live in `check_rerun`.

THE CARDINAL RULE
-----------------
**T3 never freezes and never cuts over.** It computes a verdict and seals a
bundle; a human executes the transaction. A freeze crosses four human-only trust
boundaries (`MEASUREMENT.md:140-142`). This module writes no file, starts no
process, signals no process, runs no inference, builds nothing, and executes
nothing it drafted — `audit_no_write_or_process_paths()` proves the first three
from this module's own AST, and `TransactionPlan` REFUSES to be constructed with
`executed=True` because a transaction that has already run is not a dry run.

WHAT IS AN INPUT AND WHAT IS COMPUTED
-------------------------------------
Every measurement, hash, receipt and rebuild is an INPUT. This module reduces
them to a verdict. That split is deliberate: the trusted evaluator times and
reduces, the release gate adjudicates, and a module that did both could grade its
own arithmetic. Two of the seven bundle components — `validation_results` and
`active_waivers` — are computed HERE from this run's own products rather than
supplied, so the seal hashes what T3 actually decided rather than what a caller
said it decided.

THE THIRD OUTCOME
-----------------
`schemas.Check` has three values and all three are used. A cell that could not be
evaluated is not a passing cell. The T3 verdict vocabulary is closed at three
(`schemas.T3_VERDICTS`), so an unevaluable gating cell lands in FAIL — but the
bundle keeps `unevaluable_cells` separate from `failed_cells`, because "we could
not tell" and "it is worse" are different facts and only one of them is about the
candidate.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md` §1.3,
§1.5, §1.6, §3.2, §9.1, §10 (all of it), §11, §12, §13, §15.4, §17.
"""
from __future__ import annotations

import ast
import json
import re
import stat as stat_module
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .. import schemas, storage
from ..adapters import qwentts_tts, whisper_stt
from . import preflight as guards
from ..evaluator import api, integrity, surface

__all__ = [
    # identity
    "TIER", "RELEASE_PROTOCOL_ID", "BUNDLE_SCHEMA", "RECORD_CLASS",
    # vocabularies
    "PHASES", "PHASE_IDENTITY_PREFLIGHT", "PHASE_BUILD_LINKAGE",
    "PHASE_BACKEND_CORRECTNESS", "PHASE_PERFORMANCE_MATRIX", "PHASE_QUALITY",
    "PHASE_STABILITY", "PHASE_CAPACITY_UTILITY", "PHASE_TRANSACTION_DRY_RUN",
    "PHASE_SEAL", "CELL_PHASES", "MODES", "MODE_DRY_RUN", "MODE_RELEASE",
    "RECIPE_CLASSES", "RECIPE_PRODUCTION_OPTIMAL", "RECIPE_DIAGNOSTIC",
    "STANDINGS", "STANDING_IMPROVED", "STANDING_NON_INFERIOR",
    "STANDING_REGRESSED", "STANDING_INDETERMINATE", "STANDING_NOT_MEASURED",
    "OBJECTIVE_SATISFYING_STANDINGS", "QUALITY_MODES", "BUNDLE_COMPONENTS",
    "SUPPLIED_COMPONENTS", "COMPUTED_COMPONENTS", "KNOWN_WAIVER_SCHEMAS",
    "LINKAGE_VERIFIER_RELPATH", "RERUN_CODES", "ARCHIVE_GENERATIONS",
    "EPYC_ROOT", "TRUST_BOUNDARY_MANIFEST", "DEFAULT_ATTESTATION_ROOTS",
    "FINGERPRINT_FACETS",
    "RELEASE_READINESS_BY_BACKEND",
    # errors
    "T3Error", "T3InputError", "StackChangePathRequired",
    "ReleaseProtocolNotRatified", "ProductionWriteRefused", "RerunRefused",
    "WaiverNotReadable",
    # inputs
    "ProtocolBinding", "PhaseProtocolBinding", "phase_protocol_binding",
    "declared_ratified_protocol_ids",
    "SealedCandidate", "ReleasePlanView", "UnchangedView",
    "TransferReceipt", "LinkageReceipt", "BackendInventory", "DeterminismDeclaration",
    "Cell", "CellResult", "PhaseStanding", "PhaseTradeException", "CapacityFloor",
    "QualityEvidence", "StabilityEvidence", "ArchivedBuild", "IncumbentArchive",
    "TransactionPlan", "WaiverBinding", "ReadWaiver", "WaiverReadReceipt",
    "T3Attempt", "StageRepair", "T3Request",
    # outputs
    "PhaseResult", "WaiverVerification", "ReleaseReceipt", "ReleaseBundle",
    "RerunDisposition", "T3Result",
    # functions
    "release_plan_view", "release_plan_view_from_compiled", "unchanged_view",
    "unchanged_results_from_plan", "transfer_receipts_from_plan",
    "human_only_boundary", "waiver_binding_from_path", "waiver_read_violations",
    "verify_waiver",
    "sealed_fingerprint", "check_rerun",
    "phase_identity_preflight", "phase_build_linkage", "phase_backend_correctness",
    "phase_performance_matrix", "phase_quality", "phase_stability",
    "phase_capacity_utility", "phase_transaction_dry_run", "phase_seal",
    "compute_verdict", "run_t3", "T3Runner",
    "audit_no_write_or_process_paths", "audit_phase_coverage_totality",
    "audit_backend_readiness_is_consulted", "audit_waiver_reader_is_the_only_reader",
    "audit_reader_narrowing_is_never_widened",
    "audit_waiver_binding_is_constructed_only_by_the_reader",
    # calibration (§10.4 "expect the v8 dry-run to FAIL without its waiver")
    "PreservedFreeze", "preserved_freeze_from_v8_artifacts",
    "preserved_freeze_from_speech_artifact", "calibration_request",
]


# =============================================================================
# Identity
# =============================================================================

TIER = "T3"

if TIER not in api.RELEASE_TIERS:
    # Raised at IMPORT, not in a test somebody can forget to run.
    # `evaluator.api.admit_tier()` refuses this tier BY NAME and names AK5 as its
    # owner. If the two vocabularies ever drift, the refusal points at nothing and
    # a release-shaped decision becomes producible under a search protocol.
    raise RuntimeError(
        f"t3.TIER={TIER!r} is not among evaluator.api.RELEASE_TIERS "
        f"{list(api.RELEASE_TIERS)}; the search evaluator's refusal and the release "
        "gate's identity have drifted apart"
    )

#: The release protocol this gate is written against. It is NOT ratified: Annex K
#: (`measurement/protocols/kernel-research.md`) contains `P-AK-SEARCH-1` and
#: nothing else, and design §3.6 lists `P-KERNEL-FREEZE-1` among the documents
#: that still have to be ADDED to `human_only_paths.yaml`. AK-D20 splits the two
#: attestations for exactly this reason: "search authorization" after AK3,
#: "release authorization" before the first freeze. So the id is recorded, the
#: ratification state is an INPUT (`ProtocolBinding.ratified`), and a run in
#: `release` mode against an unratified protocol is refused rather than quietly
#: treated as authorised.
RELEASE_PROTOCOL_ID = "P-KERNEL-FREEZE-1"

BUNDLE_SCHEMA = "epyc.autokernel.t3_release_bundle.v1"

#: Annex K requires every protocol to state the class of record it emits. A T3
#: bundle is a release VERDICT over claims; the verdict itself is not a claim, and
#: the claims it licenses are named individually in the receipt.
RECORD_CLASS = "RELEASE VERDICT — the claims it licenses are enumerated, not implied"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PRODUCTION_BRANCH_RE = re.compile(r"^production-(consolidated|speech)-v(\d+)$")


# =============================================================================
# Vocabularies — the nine phases of §10.2, in order
# =============================================================================

PHASE_IDENTITY_PREFLIGHT = "identity_preflight"
PHASE_BUILD_LINKAGE = "build_linkage"
PHASE_BACKEND_CORRECTNESS = "backend_correctness"
PHASE_PERFORMANCE_MATRIX = "performance_matrix"
PHASE_QUALITY = "quality"
PHASE_STABILITY = "stability"
PHASE_CAPACITY_UTILITY = "capacity_utility"
PHASE_TRANSACTION_DRY_RUN = "transaction_dry_run"
PHASE_SEAL = "seal"

#: Exactly the nine of §10.2, in the order §10.2 states them. The order is not
#: cosmetic: identity gates everything, and the seal can only hash results that
#: already exist.
PHASES = (
    PHASE_IDENTITY_PREFLIGHT, PHASE_BUILD_LINKAGE, PHASE_BACKEND_CORRECTNESS,
    PHASE_PERFORMANCE_MATRIX, PHASE_QUALITY, PHASE_STABILITY,
    PHASE_CAPACITY_UTILITY, PHASE_TRANSACTION_DRY_RUN, PHASE_SEAL,
)

#: The five phases that admit measured CELLS. Identity, linkage, the transaction
#: dry-run and the seal are adjudicated from receipts, not from a matrix, and a
#: cell claiming one of them is a wiring error rather than a soft mismatch.
CELL_PHASES = (
    PHASE_BACKEND_CORRECTNESS, PHASE_PERFORMANCE_MATRIX, PHASE_QUALITY,
    PHASE_STABILITY, PHASE_CAPACITY_UTILITY,
)

#: `dry_run` is what §10.4's calibration note asks for — run the compiler and the
#: validator against preserved freeze artifacts and see what they say. It computes
#: the identical verdict; what it may not do is stand in for a release
#: authorisation, which is why the bundle carries the mode.
MODE_DRY_RUN = "dry_run"
MODE_RELEASE = "release"
MODES = (MODE_DRY_RUN, MODE_RELEASE)

#: Invariant 15: *"Baseline/off-recipe cells are diagnostic and never veto or
#: justify a release."* Two classes, and only one of them gates. `schemas` pins a
#: CAMPAIGN to `production_optimal`; a release matrix legitimately carries
#: diagnostic cells beside the gating ones, so the release vocabulary is wider by
#: exactly one member and the extra member is inert.
RECIPE_PRODUCTION_OPTIMAL = "production_optimal"
RECIPE_DIAGNOSTIC = "diagnostic"
RECIPE_CLASSES = frozenset({RECIPE_PRODUCTION_OPTIMAL, RECIPE_DIAGNOSTIC})

#: §1.6 per-phase standing. `indeterminate` and `not_measured` are separate on
#: purpose: "the e-process did not resolve" is a result, "nobody ran it" is a hole.
STANDING_IMPROVED = "improved"
STANDING_NON_INFERIOR = "non_inferior"
STANDING_REGRESSED = "regressed"
STANDING_INDETERMINATE = "indeterminate"
STANDING_NOT_MEASURED = "not_measured"
STANDINGS = (
    STANDING_IMPROVED, STANDING_NON_INFERIOR, STANDING_REGRESSED,
    STANDING_INDETERMINATE, STANDING_NOT_MEASURED,
)

#: Only these two SATISFY non-inferiority. `indeterminate` deliberately does not:
#: non-inferiority is an e-process decision (`MEASUREMENT.md:30-32`), and a window
#: that failed to resolve has not made it. Treating "no detectable difference" as
#: "not worse" is how a gate silently loosens.
OBJECTIVE_SATISFYING_STANDINGS = frozenset({STANDING_IMPROVED, STANDING_NON_INFERIOR})

#: §10.2 phase 5. `transferred` is the cheap path the constitution explicitly
#: allows — *"transfer banked quality across kernel eras once paired parity proves
#: transfer"* — and it is the path that most needs the §10.5 archive check.
QUALITY_DETERMINISTIC_PARITY = "deterministic_parity"
QUALITY_MEASURED_PARITY = "measured_parity"
QUALITY_TRANSFERRED = "transferred"
QUALITY_MODES = (
    QUALITY_DETERMINISTIC_PARITY, QUALITY_MEASURED_PARITY, QUALITY_TRANSFERRED,
)

#: §10.2 phase 9: *"hash the protocol, plan, raw evidence, reducers, validation
#: results, active waivers, and the exact transaction into one release bundle."*
#: Seven components, named individually so a missing one is a named absence.
COMPONENT_PROTOCOL = "protocol"
COMPONENT_PLAN = "plan"
COMPONENT_RAW_EVIDENCE = "raw_evidence"
COMPONENT_REDUCERS = "reducers"
COMPONENT_VALIDATION_RESULTS = "validation_results"
COMPONENT_ACTIVE_WAIVERS = "active_waivers"
COMPONENT_TRANSACTION = "transaction"
BUNDLE_COMPONENTS = (
    COMPONENT_PROTOCOL, COMPONENT_PLAN, COMPONENT_RAW_EVIDENCE,
    COMPONENT_REDUCERS, COMPONENT_VALIDATION_RESULTS, COMPONENT_ACTIVE_WAIVERS,
    COMPONENT_TRANSACTION,
)
#: Supplied by the caller — they hash artifacts this module deliberately never
#: reads.
SUPPLIED_COMPONENTS = (
    COMPONENT_PROTOCOL, COMPONENT_PLAN, COMPONENT_RAW_EVIDENCE,
    COMPONENT_REDUCERS, COMPONENT_TRANSACTION,
)
#: Computed here from this run's own products. A caller cannot hand T3 a digest
#: of validation results T3 did not produce.
COMPUTED_COMPONENTS = (COMPONENT_VALIDATION_RESULTS, COMPONENT_ACTIVE_WAIVERS)

#: Waiver schemas this gate can read. The v8 one is here because the calibration
#: dry-run of §10.4 runs against the real preserved artifact, and rewriting a
#: ratified operator attestation into a newer schema to make it validate would be
#: forging the very record the check exists to verify.
WAIVER_SCHEMA_AUTOKERNEL = schemas.SCHEMA_OPERATOR_WAIVER
WAIVER_SCHEMA_V8_CPU_PREFILL = "epyc.cpu_prefill_v8.operator_waiver.v1"
KNOWN_WAIVER_SCHEMAS = frozenset({
    WAIVER_SCHEMA_AUTOKERNEL, WAIVER_SCHEMA_V8_CPU_PREFILL,
})

#: §10.2 phase 2 names the verifier and §10.2 says where it lives: *"it lives in
#: epyc-inference-research, not epyc-root — CLAUDE.md cites it unqualified, same
#: defect class as the durability validator's path in MEASUREMENT.md:155"*. A
#: receipt naming an epyc-root path is refused rather than accepted as equivalent.
LINKAGE_VERIFIER_RELPATH = "scripts/utils/verify_ggml_linkage.sh"

#: §10.5. N-2 is "ideally"; N-1 is not optional.
ARCHIVE_GENERATION_N1 = "N-1"
ARCHIVE_GENERATION_N2 = "N-2"
ARCHIVE_GENERATIONS = (ARCHIVE_GENERATION_N1, ARCHIVE_GENERATION_N2)

#: backend -> the ADAPTER's own answer to *"is this backend's release path legally
#: runnable yet?"*. Both speech adapters have exposed `release_gate_readiness()`
#: since AK9 and nothing in the release plane called it, so the gate graded cells
#: under protocol families the adapters already knew were drafts. This mapping is
#: the missing adapter->gate edge, and `audit_backend_readiness_is_consulted()`
#: proves from this module's AST that phase 1 still consults it — a registry that
#: nothing reads is the defect it was added to close, wearing a table.
#:
#: A backend ABSENT from this mapping is not "ready": it is a backend whose adapter
#: states no release-readiness predicate, which is the llama pair's honest position
#: (their protocols are Annex B/G protocols the per-phase `ProtocolBinding` already
#: carries). Absence therefore delegates to the per-phase bindings and nothing else.
RELEASE_READINESS_BY_BACKEND = {
    whisper_stt.BACKEND: whisper_stt.release_gate_readiness,
    qwentts_tts.BACKEND: qwentts_tts.release_gate_readiness,
}

RERUN_ADMITTED_FIRST_ATTEMPT = "ADMITTED_FIRST_ATTEMPT"
RERUN_ADMITTED_NEW_FINGERPRINT = "ADMITTED_NEW_FINGERPRINT"
RERUN_ADMITTED_AFTER_REPAIR = "ADMITTED_AFTER_DETERMINISTIC_REPAIR"
RERUN_REFUSED_ALREADY_SEALED = "REFUSED_ALREADY_SEALED"
RERUN_REFUSED_UNCHANGED_FINGERPRINT = "REFUSED_UNCHANGED_FINGERPRINT"
RERUN_REFUSED_COOLDOWN = "REFUSED_FAILED_GATE_COOLDOWN"
RERUN_CODES = (
    RERUN_ADMITTED_FIRST_ATTEMPT, RERUN_ADMITTED_NEW_FINGERPRINT,
    RERUN_ADMITTED_AFTER_REPAIR, RERUN_REFUSED_ALREADY_SEALED,
    RERUN_REFUSED_UNCHANGED_FINGERPRINT, RERUN_REFUSED_COOLDOWN,
)


# =============================================================================
# Errors — every one is a refusal, none is a degraded result
# =============================================================================

class T3Error(Exception):
    """Base class for every refusal this module makes."""


class T3InputError(T3Error):
    """A wiring defect: the request cannot be adjudicated as given.

    Distinct from a FAIL verdict on purpose. A FAIL is a fact about the
    candidate; this is a fact about the harness, and reporting it as a candidate
    failure would put a defect in the wrong ledger.
    """


class StackChangePathRequired(T3Error):
    """`serving_runtime` was routed at the kernel-freeze path.

    §13.5: the adapter *"MUST refuse the kernel-freeze path outright rather than
    degrading to it"*, and §11.6 gives it a different release path with a
    different metric (`task_rate`, not tokens/s) and a different workload
    (variable-arrival replay). Degrading would spend a kernel-freeze transaction
    on a scheduler property.
    """


class ReleaseProtocolNotRatified(T3Error):
    """A `release`-mode run was requested under an unratified release protocol.

    AK-D20 splits "search authorization" from "release authorization" precisely
    so that release bindings are not ratified against schema sketches. Until the
    release protocol is ratified, the only honest T3 is a dry run.
    """


class ProductionWriteRefused(T3Error):
    """An input asserted that a production write had already happened.

    Invariant 5 and §11.2: the packager *"may not ... execute any command it
    drafted"*. A transaction handed to T3 with `executed=True` has crossed the
    boundary before the gate ran, so the gate refuses to grade it rather than
    producing a verdict that would look like retrospective authorisation.
    """


class RerunRefused(T3Error):
    """The expensive gate was asked to run again on evidence that did not change."""


# =============================================================================
# Small helpers
# =============================================================================

_SEVERITY = {schemas.PASS: 0, schemas.COULD_NOT_CHECK: 1, schemas.FAIL: 2}


def _worst(checks: Iterable[schemas.Check]) -> schemas.Check:
    """FAIL beats COULD_NOT_CHECK beats PASS, carrying every reason forward."""
    worst = schemas.PASS
    reasons: list = []
    for check in checks:
        if not isinstance(check, schemas.Check):
            raise T3InputError(f"expected a schemas.Check, got {type(check).__name__}")
        if _SEVERITY[check.outcome] > _SEVERITY[worst]:
            worst = check.outcome
        reasons.extend(check.reasons)
    return schemas.Check(worst, tuple(reasons))


def _fail(*reasons: str) -> schemas.Check:
    return schemas.Check(schemas.FAIL, tuple(reasons))


def _cnc(*reasons: str) -> schemas.Check:
    return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise T3InputError(f"{label}: required, a non-empty string")
    return value


def _opt_text(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _text(value, label)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise T3InputError(f"{label}: required, a bool")
    return value


def _sha256(value: Any, label: str) -> str:
    _text(value, label)
    if not _SHA256_RE.match(value):
        raise T3InputError(f"{label}: {value!r} is not a lowercase sha256 hex digest")
    if schemas.is_placeholder_digest(value):
        raise T3InputError(
            f"{label}: {value!r} is the digest of no bytes at all. A caller that had "
            "nothing to hash produced it; it is well-formed and it means the artifact "
            "was never read."
        )
    return value


def _commit(value: Any, label: str) -> str:
    _text(value, label)
    if not _COMMIT_RE.match(value):
        raise T3InputError(f"{label}: {value!r} is not a full 40-hex commit id")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    _text(value, label)
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise T3InputError(f"{label}: {value!r} is not an ISO-8601 timestamp ({exc})")
    if parsed.tzinfo:
        return parsed
    # `datetime.combine` rather than `.replace(tzinfo=…)`: `replace` is denied by
    # `audit_no_write_or_process_paths` because `Path.replace` moves a symlink, and
    # an AST audit cannot tell the two apart. The guard must not be satisfiable only
    # by exempting its own call sites, so the benign homographs are rewritten
    # instead. `astimezone` is NOT the substitute — on a naive datetime it assumes
    # local time, which would silently reinterpret every timestamp T3 reads.
    return datetime.combine(parsed.date(), parsed.time(), tzinfo=timezone.utc)


def _str_tuple(value: Any, label: str, *, non_empty: bool = True) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise T3InputError(f"{label}: required, a list/tuple of strings")
    out = tuple(value)
    for item in out:
        _text(item, f"{label}[]")
    if non_empty and not out:
        raise T3InputError(f"{label}: must not be empty")
    return out


def _typed_tuple(value: Any, label: str, klass: type, *, non_empty: bool = False) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise T3InputError(f"{label}: required, a list/tuple of {klass.__name__}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, klass):
            raise T3InputError(
                f"{label}: expected {klass.__name__}, got {type(item).__name__}")
    if non_empty and not out:
        raise T3InputError(f"{label}: must not be empty")
    return out


def _hashed_pairs(value: Any, label: str, *, non_empty: bool = True) -> tuple:
    """A tuple of `(path, sha256)` pairs, every path non-scratch."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise T3InputError(f"{label}: required, a list/tuple of (path, sha256) pairs")
    out: list = []
    for i, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise T3InputError(f"{label}[{i}]: expected a (path, sha256) pair")
        path = _text(item[0], f"{label}[{i}].path")
        digest = _sha256(item[1], f"{label}[{i}].sha256")
        out.append((path, digest))
    if non_empty and not out:
        raise T3InputError(f"{label}: must not be empty")
    return tuple(out)


def _attributed_hashed_triples(value: Any, label: str, *,
                               non_empty: bool = True) -> tuple:
    """A tuple of `(backends, path, sha256)`, where `backends` is NON-EMPTY.

    The attribution is required rather than optional because the fact it records is
    the one this host has already been bitten by: three ggml generations live here,
    and *which build a preserved `libggml-base.so.0` belongs to* is not recoverable
    from its path or its digest once the tree it came from is gone. An empty
    attribution is refused for the same reason `RollbackPlan.incumbent_binaries`
    refuses to be empty — an unattributed rollback library is a library that will be
    resolved against whatever is on the path at rollback time.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise T3InputError(
            f"{label}: required, a list/tuple of (backends, path, sha256) triples")
    out: list = []
    for i, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise T3InputError(
                f"{label}[{i}]: expected a (backends, path, sha256) triple. A bare "
                "(path, sha256) pair is the UNATTRIBUTED shape this field exists to "
                "replace: on a three-ggml-generation host, an archived library with no "
                "backend attribution is the 2026-07-31 linkage incident with a longer "
                "fuse.")
        backends = item[0]
        if isinstance(backends, str):
            raise T3InputError(
                f"{label}[{i}].backends: a single string is not a backend SET; a shared "
                "library legitimately serves more than one backend, and collapsing that "
                "to one name attributes it to the wrong build half the time")
        backends = _str_tuple(backends, f"{label}[{i}].backends")
        for backend in backends:
            if backend not in schemas.BACKENDS:
                raise T3InputError(
                    f"{label}[{i}].backends: {backend!r} is not a known backend")
        if len(set(backends)) != len(backends):
            raise T3InputError(f"{label}[{i}].backends: duplicate backend names")
        path = _text(item[1], f"{label}[{i}].path")
        digest = _sha256(item[2], f"{label}[{i}].sha256")
        out.append((tuple(sorted(set(backends))), path, digest))
    if non_empty and not out:
        raise T3InputError(f"{label}: must not be empty")
    return tuple(out)


def _check_dict(check: schemas.Check) -> dict:
    return {"outcome": check.outcome, "reasons": list(check.reasons)}


def _production_version_number(branch: str) -> Optional[int]:
    match = _PRODUCTION_BRANCH_RE.match(branch)
    return int(match.group(2)) if match else None


def _under_production_tree(path: str) -> bool:
    """True when a path resolves inside one of the FROZEN production trees.

    Invariant 3: *"No actor builds in or modifies any production tree."* The forms
    come from `storage.production_tree_forms()` rather than a literal list here, so
    a fourth production tree is added in one place.
    """
    resolved = path.rstrip("/") + "/"
    for root in storage.production_tree_forms():
        root = root.rstrip("/") + "/"
        if resolved == root or resolved.startswith(root):
            return True
    return False


# =============================================================================
# Inputs — identity
# =============================================================================

@dataclass(frozen=True)
class ProtocolBinding:
    """Which release protocol this run was adjudicated under, and whether it exists.

    `ratified` is an input rather than a constant because it is a fact about
    `epyc-root/measurement/protocols/`, which this repository may not read and must
    never assume. AK-D20 batches the release attestation separately from the search
    one; until it is signed, `ratified` is False and only a dry run is honest.
    """

    protocol_id: str
    document_sha256: str
    ratified: bool
    ratified_at: Optional[str] = None
    annex: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.protocol_id, "ProtocolBinding.protocol_id")
        _sha256(self.document_sha256, "ProtocolBinding.document_sha256")
        _bool(self.ratified, "ProtocolBinding.ratified")
        _opt_text(self.annex, "ProtocolBinding.annex")
        if self.ratified:
            _timestamp(_text(self.ratified_at, "ProtocolBinding.ratified_at"),
                       "ProtocolBinding.ratified_at")
        elif self.ratified_at is not None:
            raise T3InputError(
                "ProtocolBinding.ratified_at: present while ratified is False. A "
                "ratification timestamp on an unratified protocol is the shape of a "
                "half-applied amendment, not a partial record."
            )

    def to_dict(self) -> dict:
        return {"protocol_id": self.protocol_id, "document_sha256": self.document_sha256,
                "ratified": self.ratified, "ratified_at": self.ratified_at,
                "annex": self.annex}


@dataclass(frozen=True)
class PhaseProtocolBinding:
    """The protocol ONE (backend, workload phase) is GRADED UNDER, and its standing.

    `ProtocolBinding` proves the FREEZE protocol is ratified — the authority to run
    the gate at all. It says nothing about the protocols the matrix cells are
    *graded* under, which used to arrive in `T3Request.phase_protocols` as bare
    strings: `{"llama_cpu": {"decode": "P-BENCH-1"}}`. A bare id is a name, and a
    name is not a ratification. `P-STT-*` and `P-TTS-*` are drafts today
    (`kernel-research.md:54-56` — an unowned number cannot become a claim), and the
    old shape could not tell them apart from `P-BENCH-1`.

    So the value carries the binding. `binding is None` is the UNBOUND state and it
    is `COULD_NOT_CHECK`, never PASS: the run does not know what it was graded
    under, which is exactly what an unbound id means and exactly what deleting the
    binding would otherwise buy.
    """

    backend: str
    workload_phase: str
    protocol_id: str
    binding: Optional[ProtocolBinding] = None

    def __post_init__(self) -> None:
        _text(self.backend, "PhaseProtocolBinding.backend")
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(
                f"PhaseProtocolBinding.backend: {self.backend!r} is not a known backend")
        _text(self.workload_phase, "PhaseProtocolBinding.workload_phase")
        _text(self.protocol_id, "PhaseProtocolBinding.protocol_id")
        if self.binding is None:
            return
        if not isinstance(self.binding, ProtocolBinding):
            raise T3InputError(
                "PhaseProtocolBinding.binding: must be a ProtocolBinding or None")
        if self.binding.protocol_id != self.protocol_id:
            raise T3InputError(
                f"PhaseProtocolBinding({self.backend}/{self.workload_phase}): the phase "
                f"is declared under {self.protocol_id!r} but the binding attests "
                f"{self.binding.protocol_id!r}. A ratification receipt for a DIFFERENT "
                "protocol is the cheapest possible way to make a draft look ratified."
            )

    @property
    def ratified(self) -> Optional[bool]:
        """True / False / None, where None means *nobody said* — never False."""
        return None if self.binding is None else self.binding.ratified

    def check(self) -> schemas.Check:
        if self.binding is None:
            return _cnc(
                f"{self.backend}/{self.workload_phase}: protocol {self.protocol_id!r} "
                "arrives as a BARE ID with no ratification binding, so the gate cannot "
                "tell a ratified instrument from a draft. An unknown standing is not a "
                "ratified one (kernel-research.md:54-56)."
            )
        if not self.binding.ratified:
            return _cnc(
                f"{self.backend}/{self.workload_phase}: protocol {self.protocol_id!r} is "
                "declared NOT ratified, so the cells graded under it have no owning "
                "protocol and their numbers cannot become claims "
                "(kernel-research.md:54-56). Search under P-AK-SEARCH-1 stays legal; "
                "release eligibility for this phase is blocked until the operator "
                "ratifies or declines the family."
            )
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "workload_phase": self.workload_phase,
                "protocol_id": self.protocol_id, "ratified": self.ratified,
                "binding": None if self.binding is None else self.binding.to_dict(),
                "check": _check_dict(self.check())}


def phase_protocol_binding(value: Any, *, backend: str,
                           workload_phase: str) -> PhaseProtocolBinding:
    """Normalise one `phase_protocols[backend][phase]` entry.

    Accepts a `PhaseProtocolBinding`, a `ProtocolBinding`, or the legacy bare id.
    The bare id is accepted rather than refused ON PURPOSE: refusing it would turn
    every caller that has not yet been rewired into an exception, and an exception
    is not a verdict about the candidate. It is instead carried as the UNBOUND state
    and reported as `COULD_NOT_CHECK` by `check()`, so the gap is journaled in the
    bundle rather than either crashing the run or passing silently.
    """
    if isinstance(value, PhaseProtocolBinding):
        if value.backend != backend or value.workload_phase != workload_phase:
            raise T3InputError(
                f"phase_protocols[{backend!r}][{workload_phase!r}]: the binding names "
                f"({value.backend}, {value.workload_phase}); a binding filed under a "
                "different phase is a mislabelled instrument, not a relabelled one")
        return value
    if isinstance(value, ProtocolBinding):
        return PhaseProtocolBinding(
            backend=backend, workload_phase=workload_phase,
            protocol_id=value.protocol_id, binding=value)
    if isinstance(value, str):
        return PhaseProtocolBinding(
            backend=backend, workload_phase=workload_phase,
            protocol_id=_text(value, f"phase_protocols[{backend!r}][{workload_phase!r}]"))
    raise T3InputError(
        f"phase_protocols[{backend!r}][{workload_phase!r}]: expected a protocol id, a "
        f"ProtocolBinding, or a PhaseProtocolBinding, got {type(value).__name__}")


def declared_ratified_protocol_ids(request: "T3Request") -> tuple:
    """The ratified set the adapters are asked about — DERIVED, never a constant.

    `whisper_stt.release_gate_readiness()` documents why it takes the set as an
    argument: *"the source of truth is the protocol registry in `MEASUREMENT.md` §2,
    and a constant here would go stale silently"*. The same argument forbids a
    constant HERE, so the set is read off this request's own bindings and nothing
    else. A protocol is in the set only if some `ProtocolBinding` in the request
    carries a document hash and `ratified=True` — which means the only way to
    satisfy a speech adapter is to declare each of its release protocols, hashed,
    as ratified. There is no shorter route, and in particular no flag.
    """
    if not isinstance(request, T3Request):
        raise T3InputError(
            "declared_ratified_protocol_ids: request must be a T3Request")
    ids: set = set()
    if request.protocol.ratified:
        ids.add(request.protocol.protocol_id)
    for by_phase in request.phase_protocols.values():
        for binding in by_phase.values():
            if binding.ratified:
                ids.add(binding.protocol_id)
    for binding in request.protocol_registry:
        if binding.ratified:
            ids.add(binding.protocol_id)
    return tuple(sorted(ids))


@dataclass(frozen=True)
class SealedCandidate:
    """§11.1: *"Sealed release candidate: immutable full build plus evidence target."*

    §3.2 enumerates what a sealed candidate binds: production base commit; clean
    full candidate commit; complete source tree and agent-file overlay; toolchain
    identity; binary and linked-library hashes; immutable evaluator hash; derived
    scope manifest; and evidence directory hash tree. The CPU-side bindings restate
    `bench-cpu.md:38-44` and are cited rather than duplicated.
    """

    candidate_id: str
    source_tree: str
    candidate_branch: str
    production_base_commit: str
    candidate_commit: str
    seal_sha256: str
    evaluator_bundle_sha256: str
    scope_manifest_sha256: str
    evidence_tree_sha256: str
    #: backend -> sha256 of the built binary that backend runs.
    binary_sha256: Mapping[str, str] = field(default_factory=dict)
    #: backend -> sha256 over the linked-library set (the `ldd` closure).
    linkage_sha256: Mapping[str, str] = field(default_factory=dict)
    #: backend -> the build directory the candidate was built in.
    build_dirs: Mapping[str, str] = field(default_factory=dict)
    overlay_present: bool = False
    tree_clean: bool = False
    ancestry_clean: bool = False

    def __post_init__(self) -> None:
        _text(self.candidate_id, "SealedCandidate.candidate_id")
        if not self.candidate_id.startswith("akc-"):
            raise T3InputError("SealedCandidate.candidate_id: must start with 'akc-'")
        if self.source_tree not in schemas.SOURCE_TREES:
            raise T3InputError(
                f"SealedCandidate.source_tree: {self.source_tree!r} is not one of "
                f"{sorted(schemas.SOURCE_TREES)}")
        _text(self.candidate_branch, "SealedCandidate.candidate_branch")
        if _PRODUCTION_BRANCH_RE.match(self.candidate_branch):
            raise T3InputError(
                f"SealedCandidate.candidate_branch: {self.candidate_branch!r} names a "
                "FROZEN production branch. Invariant 3 — no actor builds in or modifies "
                "a production tree; we version PAST production, never patch it in place."
            )
        _commit(self.production_base_commit, "SealedCandidate.production_base_commit")
        _commit(self.candidate_commit, "SealedCandidate.candidate_commit")
        if self.production_base_commit == self.candidate_commit:
            raise T3InputError(
                "SealedCandidate: the candidate commit equals the production base. A "
                "release of the incumbent is not a release; there is nothing to gate."
            )
        for name in ("seal_sha256", "evaluator_bundle_sha256", "scope_manifest_sha256",
                     "evidence_tree_sha256"):
            _sha256(getattr(self, name), f"SealedCandidate.{name}")
        for label, mapping in (("binary_sha256", self.binary_sha256),
                               ("linkage_sha256", self.linkage_sha256)):
            if not isinstance(mapping, Mapping):
                raise T3InputError(f"SealedCandidate.{label}: must be a mapping")
            for backend, digest in mapping.items():
                if backend not in schemas.BACKENDS:
                    raise T3InputError(
                        f"SealedCandidate.{label}: {backend!r} is not a known backend")
                _sha256(digest, f"SealedCandidate.{label}[{backend}]")
        if not isinstance(self.build_dirs, Mapping):
            raise T3InputError("SealedCandidate.build_dirs: must be a mapping")
        for backend, path in self.build_dirs.items():
            if backend not in schemas.BACKENDS:
                raise T3InputError(
                    f"SealedCandidate.build_dirs: {backend!r} is not a known backend")
            _text(path, f"SealedCandidate.build_dirs[{backend}]")
        for name in ("overlay_present", "tree_clean", "ancestry_clean"):
            _bool(getattr(self, name), f"SealedCandidate.{name}")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "source_tree": self.source_tree,
            "candidate_branch": self.candidate_branch,
            "production_base_commit": self.production_base_commit,
            "candidate_commit": self.candidate_commit,
            "seal_sha256": self.seal_sha256,
            "evaluator_bundle_sha256": self.evaluator_bundle_sha256,
            "scope_manifest_sha256": self.scope_manifest_sha256,
            "evidence_tree_sha256": self.evidence_tree_sha256,
            "binary_sha256": dict(self.binary_sha256),
            "linkage_sha256": dict(self.linkage_sha256),
            "build_dirs": dict(self.build_dirs),
            "overlay_present": self.overlay_present, "tree_clean": self.tree_clean,
            "ancestry_clean": self.ancestry_clean,
        }


# =============================================================================
# Inputs — the derived release plan (the `plan.py` seam)
# =============================================================================

@dataclass(frozen=True)
class ReleasePlanView:
    """Exactly what T3 needs from the §10.1 release-plan compiler, and no more.

    The compiler lives in `release/plan.py`; this is the narrow view T3 adjudicates
    against, so the gate depends on the plan's SHAPE rather than on its internals.
    `release_plan_view()` adapts a compiler product, a mapping, or an object with
    the same attributes, and REFUSES anything else — a plan the gate had to guess
    at is a scope T3 set for itself, which is exactly what §12 forbids.
    """

    plan_id: str
    plan_sha256: str
    source_tree: str
    backends: tuple
    cells: tuple
    incumbent_branch: str
    incumbent_commit: str
    incumbent_version_number: int

    def __post_init__(self) -> None:
        _text(self.plan_id, "ReleasePlanView.plan_id")
        _sha256(self.plan_sha256, "ReleasePlanView.plan_sha256")
        if self.source_tree not in schemas.SOURCE_TREES:
            raise T3InputError(
                f"ReleasePlanView.source_tree: {self.source_tree!r} is not one of "
                f"{sorted(schemas.SOURCE_TREES)}")
        backends = _str_tuple(self.backends, "ReleasePlanView.backends")
        for backend in backends:
            if backend not in schemas.BACKENDS:
                raise T3InputError(
                    f"ReleasePlanView.backends: {backend!r} is not a known backend")
            if backend == "serving_runtime":
                raise StackChangePathRequired(
                    "the release plan names `serving_runtime`, which has no source tree "
                    "and no kernel freeze. Its release path is the three-gate "
                    "stack-change path of §11.6 on `stack_change_guard.py`, measured in "
                    "task_rate under variable arrival; §13.5 requires the adapter to "
                    "REFUSE the kernel-freeze path rather than degrade to it."
                )
            if schemas.SOURCE_TREE_BY_BACKEND[backend] != self.source_tree:
                raise T3InputError(
                    f"ReleasePlanView.backends: {backend!r} is served by "
                    f"{schemas.SOURCE_TREE_BY_BACKEND[backend]!r}, not by "
                    f"{self.source_tree!r}. Freeze scope is the union of backends served "
                    "by ONE tree (§1.5)."
                )
        served = frozenset(b for b, tree in schemas.SOURCE_TREE_BY_BACKEND.items()
                           if tree == self.source_tree)
        if frozenset(backends) != served:
            # The cheapest possible scope exploit: leave a backend out of the plan and
            # the freeze ships its binary unmeasured with nothing recording the
            # omission. §1.5 makes scope the union of backends the tree serves,
            # narrowed by exactly one sanctioned route — the §3.2 test, which leaves a
            # receipt. The compiler enforces this too; the gate is where it must not be
            # possible to have been forgotten.
            raise T3InputError(
                f"source tree {self.source_tree!r} serves {sorted(served)} but the plan "
                f"names {sorted(backends)}. Freeze scope is the union of backends served "
                "by the tree (§1.5); it is narrowed by the §3.2 backend-unchanged test "
                "with a transfer receipt, never by omitting a backend from the plan."
            )
        object.__setattr__(self, "backends", backends)
        cells = _typed_tuple(self.cells, "ReleasePlanView.cells", Cell, non_empty=True)
        seen: set = set()
        for cell in cells:
            if cell.cell_id in seen:
                raise T3InputError(
                    f"ReleasePlanView.cells: duplicate cell_id {cell.cell_id!r}. §10.1 "
                    "deduplicates equivalent cells WITHOUT losing which roles they "
                    "protect; two rows with one id lose exactly that."
                )
            seen.add(cell.cell_id)
            if cell.backend not in backends:
                raise T3InputError(
                    f"ReleasePlanView.cells: cell {cell.cell_id!r} names backend "
                    f"{cell.backend!r}, which the plan does not declare")
        object.__setattr__(self, "cells", cells)
        _text(self.incumbent_branch, "ReleasePlanView.incumbent_branch")
        _commit(self.incumbent_commit, "ReleasePlanView.incumbent_commit")
        if not isinstance(self.incumbent_version_number, int) or \
                isinstance(self.incumbent_version_number, bool) or \
                self.incumbent_version_number < 0:
            raise T3InputError(
                "ReleasePlanView.incumbent_version_number: required, a non-negative int")

    def cells_for(self, backend: str) -> tuple:
        return tuple(c for c in self.cells if c.backend == backend)

    def to_dict(self) -> dict:
        return {"plan_id": self.plan_id, "plan_sha256": self.plan_sha256,
                "source_tree": self.source_tree, "backends": list(self.backends),
                "cell_ids": [c.cell_id for c in self.cells],
                "incumbent_branch": self.incumbent_branch,
                "incumbent_commit": self.incumbent_commit,
                "incumbent_version_number": self.incumbent_version_number}


_PLAN_VIEW_FIELDS = (
    "plan_id", "plan_sha256", "source_tree", "backends", "cells",
    "incumbent_branch", "incumbent_commit", "incumbent_version_number",
)


def release_plan_view(plan: Any) -> ReleasePlanView:
    """Adapt a `plan.py` product into the view T3 adjudicates.

    Accepts a `ReleasePlanView`, an object exposing `.release_plan_view()` or
    `.to_view()`, an object carrying the view's own attributes, or a mapping with
    the view's keys. Anything else RAISES: a gate that guessed at its own scope
    would be setting it, and §12's named defence is that scope is derived, never
    declared by the party being measured.
    """
    if isinstance(plan, ReleasePlanView):
        return plan
    for attr in ("release_plan_view", "to_view"):
        hook = getattr(plan, attr, None)
        if callable(hook):
            produced = hook()
            if not isinstance(produced, ReleasePlanView):
                raise T3InputError(
                    f"{type(plan).__name__}.{attr}() returned "
                    f"{type(produced).__name__}, not a ReleasePlanView")
            return produced
    if isinstance(plan, Mapping):
        missing = [k for k in _PLAN_VIEW_FIELDS if k not in plan]
        if missing:
            raise T3InputError(
                f"release plan mapping is missing {missing}; T3 will not infer a "
                "release scope from a partial plan")
        return ReleasePlanView(**{k: plan[k] for k in _PLAN_VIEW_FIELDS})
    missing = [k for k in _PLAN_VIEW_FIELDS if not hasattr(plan, k)]
    if missing:
        raise T3InputError(
            f"{type(plan).__name__} is not a release plan T3 can read: missing "
            f"{missing}. Supply a ReleasePlanView, or expose release_plan_view()."
        )
    return ReleasePlanView(**{k: getattr(plan, k) for k in _PLAN_VIEW_FIELDS})


# =============================================================================
# Inputs — the §3.2 per-backend unchanged test, as `plan.py` produces it
# =============================================================================

@dataclass(frozen=True)
class UnchangedView:
    """The §3.2 two-stage backend-unchanged result, normalised.

    `evaluator.surface.backend_unchanged()` already implements the test — the
    source-closure diff as the gate, the normalized comparison against an anchor
    rebuild as the confirmation, the hard finding when the two disagree. This is
    the shape T3 reads, so a plan that has been round-tripped through JSON and a
    plan holding live `BackendUnchangedResult` objects adjudicate identically.
    """

    backend: str
    may_drop_cells: bool
    unchanged_outcome: str
    agreement_outcome: str
    stage2_ran: bool
    reasons: tuple = ()
    findings: tuple = ()
    blocking_reasons: tuple = ()

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(f"UnchangedView.backend: {self.backend!r} is not a backend")
        _bool(self.may_drop_cells, "UnchangedView.may_drop_cells")
        for name in ("unchanged_outcome", "agreement_outcome"):
            value = getattr(self, name)
            if value not in (schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK):
                raise T3InputError(
                    f"UnchangedView.{name}: {value!r} is not a Check outcome")
        _bool(self.stage2_ran, "UnchangedView.stage2_ran")
        object.__setattr__(self, "reasons", _str_tuple(
            self.reasons, "UnchangedView.reasons", non_empty=False))
        object.__setattr__(self, "blocking_reasons", _str_tuple(
            self.blocking_reasons, "UnchangedView.blocking_reasons", non_empty=False))
        findings = tuple(self.findings)
        for finding in findings:
            if not isinstance(finding, Mapping) or "code" not in finding:
                raise T3InputError(
                    "UnchangedView.findings: each finding must be a mapping with a 'code'")
        object.__setattr__(self, "findings", findings)
        contradictions = self.drop_contradictions()
        if contradictions:
            raise T3InputError(
                "UnchangedView: may_drop_cells is True while "
                + "; ".join(contradictions)
                + ". `surface.backend_unchanged()` never produces that pair, and "
                "`plan.drop_verdict_contradictions()` refuses the same shape one door "
                "up. A hand-built view that does is asserting the conclusion, and that "
                "boolean deletes a backend's whole release matrix."
            )

    def drop_contradictions(self) -> tuple:
        """Reasons `may_drop_cells` contradicts the evidence recorded beside it.

        The same conditions `plan.drop_verdict_contradictions()` re-derives, restated
        over the NORMALISED view — because the plan compiler is not the only door.
        `unchanged_view()` accepts a hand-built `UnchangedView` and a bare mapping by
        design (a JSON-round-tripped plan must adjudicate identically), and both of
        those routes bypass the compiler entirely. `may_drop_cells` is a plain field,
        not a property, and it drops an entire backend's cells; a drop verdict the two
        modules disagree about is a drop verdict neither of them derived.

        Empty when the view agrees with itself, so a genuine `backend_unchanged()`
        result — stage 1 PASS, stage 2 ran and PASSed, the stages agree, nothing filed
        — passes through untouched.
        """
        if not self.may_drop_cells:
            return ()
        reasons: list = []
        if self.unchanged_outcome != schemas.PASS:
            reasons.append(f"the unchanged verdict is {self.unchanged_outcome}, not PASS")
        if self.agreement_outcome != schemas.PASS:
            reasons.append(
                f"stage agreement is {self.agreement_outcome}, not PASS — §3.2 files a "
                "disagreement against the build-identity machinery rather than "
                "preferring the cheaper stage")
        if not self.stage2_ran:
            reasons.append(
                "stage 2 (the normalized-binary confirmation against an anchor rebuild) "
                "did not run; the source-closure gate alone never drops cells")
        if self.findings:
            reasons.append(
                "build-identity findings are filed: "
                + ", ".join(str(f.get("code")) for f in self.findings))
        if self.blocking_reasons:
            reasons.append("blocking reasons are recorded: "
                           + "; ".join(self.blocking_reasons))
        return tuple(reasons)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "may_drop_cells": self.may_drop_cells,
                "unchanged_outcome": self.unchanged_outcome,
                "agreement_outcome": self.agreement_outcome,
                "stage2_ran": self.stage2_ran, "reasons": list(self.reasons),
                "findings": [dict(f) for f in self.findings],
                "blocking_reasons": list(self.blocking_reasons)}


def unchanged_view(obj: Any) -> UnchangedView:
    """Normalise a `surface.BackendUnchangedResult`, its `to_dict()`, or a view."""
    if isinstance(obj, UnchangedView):
        return obj
    if isinstance(obj, surface.BackendUnchangedResult):
        return UnchangedView(
            backend=obj.backend,
            may_drop_cells=obj.may_drop_cells,
            unchanged_outcome=obj.unchanged.outcome,
            agreement_outcome=obj.agreement.outcome,
            stage2_ran=obj.stage2 is not None,
            reasons=tuple(obj.unchanged.reasons),
            findings=tuple(f.to_dict() for f in obj.findings),
            blocking_reasons=tuple(obj.blocking_reasons),
        )
    if isinstance(obj, Mapping):
        unchanged = obj.get("unchanged")
        agreement = obj.get("agreement")
        if not isinstance(unchanged, Mapping) or not isinstance(agreement, Mapping):
            raise T3InputError(
                "backend-unchanged mapping must carry 'unchanged' and 'agreement' "
                "blocks as `surface.BackendUnchangedResult.to_dict()` writes them")
        return UnchangedView(
            backend=obj.get("backend"),
            may_drop_cells=obj.get("may_drop_cells"),
            unchanged_outcome=unchanged.get("outcome"),
            agreement_outcome=agreement.get("outcome"),
            stage2_ran=obj.get("stage2") is not None,
            reasons=tuple(unchanged.get("reasons") or ()),
            findings=tuple(obj.get("findings") or ()),
            blocking_reasons=tuple(obj.get("blocking_reasons") or ()),
        )
    raise T3InputError(
        f"cannot read a §3.2 backend-unchanged result from {type(obj).__name__}. "
        "Supply a surface.BackendUnchangedResult, its to_dict(), or an UnchangedView."
    )


def _compiled_backend_plans(compiled: Any) -> Optional[list]:
    """The `plan.BackendPlan` list of a compiled `plan.ReleasePlan`, or None.

    Duck-typed rather than isinstance-checked so that importing `t3` never requires
    importing `plan`: they are siblings under one package and a hard edge between
    them would make either unusable without the other.
    """
    backends = getattr(compiled, "backends", None)
    if not isinstance(backends, (list, tuple)) or not backends:
        return None
    entries = list(backends)
    if all(hasattr(e, "backend") and hasattr(e, "cells") for e in entries):
        return entries
    return None


def unchanged_results_from_plan(plan: Any) -> dict:
    """Pull the per-backend §3.2 unchanged results off a `plan.py` product.

    §10.2 phase 1 *includes* the per-backend unchanged test, and the plan compiler
    is where it runs (§14 AK5): `plan.BackendPlan.unchanged_ref` carries the §3.2
    result verbatim, as a record rather than as a gate. This READS it. It never
    recomputes the test — two implementations of one identity test is how the two
    stages come to disagree for reasons that are about neither the source nor the
    binary, which §3.2 classifies as a build-identity defect precisely because it
    must never happen by accident.
    """
    compiled = _compiled_backend_plans(plan)
    if compiled is not None:
        out: dict = {}
        for backend_plan in compiled:
            ref = getattr(backend_plan, "unchanged_ref", None)
            if ref is None:
                continue
            out[backend_plan.backend] = unchanged_view(ref)
        if not out:
            raise T3InputError(
                "the compiled release plan carries no §3.2 unchanged result on any "
                "backend. Without it every backend the tree serves owes full "
                "candidate-grade evidence, and the gate must be TOLD that rather than "
                "inferring it from an absence."
            )
        return out

    raw = None
    if isinstance(plan, Mapping):
        raw = plan.get("backend_unchanged")
    if raw is None:
        raw = getattr(plan, "backend_unchanged", None)
    if raw is None:
        raise T3InputError(
            f"{type(plan).__name__} carries no `backend_unchanged` map. §10.2 phase 1 "
            "includes the per-backend unchanged test; without it every backend the tree "
            "serves owes full candidate-grade evidence and the gate must be told so "
            "explicitly rather than inferring it."
        )
    if not isinstance(raw, Mapping):
        raise T3InputError("plan.backend_unchanged must be a mapping keyed by backend")
    out = {}
    for backend, entry in raw.items():
        view = unchanged_view(entry)
        if view.backend != backend:
            raise T3InputError(
                f"plan.backend_unchanged[{backend!r}] holds a result for "
                f"{view.backend!r}; the key and the record disagree")
        out[backend] = view
    return out


def transfer_receipts_from_plan(plan: Any, *, incumbent_commit: str) -> dict:
    """Adapt `plan.BackendPlan.transfer_receipt` into the receipts phase 1 checks.

    The compiler already refuses a receipt that names no artifacts
    (`plan.IncumbentEvidence` raises on an empty set), so this adapter's job is to
    carry the artifact/hash pairs across without loss — and to fail loudly if a
    receipt arrives in a shape it cannot read, rather than producing an empty
    receipt that would look like a dropped cell nobody had to justify.
    """
    _commit(incumbent_commit, "transfer_receipts_from_plan: incumbent_commit")
    compiled = _compiled_backend_plans(plan)
    if compiled is None:
        raise T3InputError(
            f"{type(plan).__name__} is not a compiled release plan with BackendPlan "
            "entries; supply TransferReceipt objects directly")
    out: dict = {}
    for backend_plan in compiled:
        receipt = getattr(backend_plan, "transfer_receipt", None)
        if receipt is None:
            continue
        incumbent = getattr(receipt, "incumbent", None)
        if isinstance(incumbent, Mapping):
            raw_artifacts = incumbent.get("artifacts") or ()
        else:
            raw_artifacts = getattr(incumbent, "artifacts", ()) or ()
        artifacts: list = []
        for entry in raw_artifacts:
            if isinstance(entry, Mapping):
                artifacts.append((entry.get("ref"), entry.get("sha256")))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                artifacts.append((entry[0], entry[1]))
            else:
                raise T3InputError(
                    f"transfer receipt for {backend_plan.backend!r} carries an artifact "
                    f"entry T3 cannot read: {entry!r}")
        evidence_refs = tuple(getattr(receipt, "dropped_cell_ids", ())) or (
            f"{backend_plan.backend}: plan-declared transfer",)
        unchanged_result = getattr(receipt, "unchanged_result", {}) or {}
        out[backend_plan.backend] = TransferReceipt(
            backend=backend_plan.backend,
            incumbent_artifacts=tuple(artifacts),
            incumbent_evidence_refs=evidence_refs,
            unchanged_digest=schemas.content_hash(dict(unchanged_result)),
            incumbent_commit=incumbent_commit,
        )
    return out


def release_plan_view_from_compiled(compiled: Any, *, incumbent_branch: str,
                                    incumbent_commit: str,
                                    incumbent_version_number: int,
                                    extra_cells: Sequence[Cell] = ()) -> ReleasePlanView:
    """Adapt a compiled `plan.ReleasePlan` into the view T3 adjudicates.

    The incumbent identity is supplied rather than read off the plan on purpose:
    `plan.ReleaseTarget` names the production BASE the candidate was cut from, which
    is not the same fact as which production VERSION is currently installed behind
    the stable symlinks. Conflating them is how a transaction ends up computing its
    next version from the wrong side of a rollback.

    A compiled cell becomes a `performance_matrix` cell: that is the phase the
    compiler's cells describe (model, quant, context, KV, speculation, concurrency,
    placement, co-residency at the production-optimal recipe). Correctness, quality,
    stability and capacity cells are supplied through `extra_cells`, because those
    are evidence classes the compiler declares requirements for rather than rows it
    enumerates.
    """
    target = getattr(compiled, "target", None)
    backend_plans = _compiled_backend_plans(compiled)
    if target is None or backend_plans is None:
        raise T3InputError(
            f"{type(compiled).__name__} is not a compiled release plan "
            "(no `target` / `backends`)")
    sha_hook = getattr(compiled, "sha256", None)
    if not callable(sha_hook):
        raise T3InputError("the compiled plan exposes no sha256(); it cannot be sealed")

    cells: list = []
    for backend_plan in backend_plans:
        for cell in backend_plan.cells:
            protocol = getattr(cell, "protocol", None)
            if protocol is None:
                raise T3InputError(
                    f"compiled cell {cell.cell_id!r} names no owning protocol; §1.6 "
                    "judges each phase under its own protocol and T3 will not guess one")
            co_residency = getattr(cell, "co_residency", "single")
            model = getattr(cell, "model", None)
            cells.append(Cell(
                cell_id=cell.cell_id, backend=cell.backend,
                release_phase=PHASE_PERFORMANCE_MATRIX,
                protocol_id=protocol.protocol_id,
                recipe_class=(RECIPE_PRODUCTION_OPTIMAL
                              if cell.recipe_class == RECIPE_PRODUCTION_OPTIMAL
                              else RECIPE_DIAGNOSTIC),
                metric=protocol.metric, metric_direction=protocol.direction,
                workload_phase=cell.phase,
                claim=(f"{cell.backend} {cell.phase} non-inferiority at the "
                       f"production-optimal recipe for {cell.cell_id}"),
                roles_protected=tuple(getattr(cell, "protected_roles", ())),
                co_resident=co_residency != "single",
                model=getattr(model, "model_path", None),
            ))
    cells.extend(_typed_tuple(extra_cells, "extra_cells", Cell))

    return ReleasePlanView(
        plan_id=str(getattr(compiled, "compiler_id", "akplan")),
        plan_sha256=sha_hook(),
        source_tree=target.source_tree,
        backends=tuple(b.backend for b in backend_plans),
        cells=tuple(cells),
        incumbent_branch=incumbent_branch,
        incumbent_commit=incumbent_commit,
        incumbent_version_number=incumbent_version_number,
    )


@dataclass(frozen=True)
class TransferReceipt:
    """§10.2 phase 1: dropped cells leave a receipt naming the incumbent artifacts.

    *"Confirmed unchanged and incumbent evidence still in scope ⇒ that backend's
    cells drop with a transfer receipt naming the incumbent artifacts and their
    hashes."* A receipt naming nothing is the shape of a dropped cell nobody can
    audit, so `incumbent_artifacts` is required non-empty.
    """

    backend: str
    incumbent_artifacts: tuple
    incumbent_evidence_refs: tuple
    unchanged_digest: str
    incumbent_commit: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(f"TransferReceipt.backend: {self.backend!r} is not a backend")
        object.__setattr__(self, "incumbent_artifacts", _hashed_pairs(
            self.incumbent_artifacts, "TransferReceipt.incumbent_artifacts"))
        object.__setattr__(self, "incumbent_evidence_refs", _str_tuple(
            self.incumbent_evidence_refs, "TransferReceipt.incumbent_evidence_refs"))
        _sha256(self.unchanged_digest, "TransferReceipt.unchanged_digest")
        _commit(self.incumbent_commit, "TransferReceipt.incumbent_commit")

    def check(self) -> schemas.Check:
        reasons = [
            f"{path} is a scratch citation and cannot be the evidence of record "
            "(MEASUREMENT.md:146-156)"
            for path, _ in self.incumbent_artifacts if storage.is_scratch_path(path)
        ]
        return _fail(*reasons) if reasons else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"backend": self.backend,
                "incumbent_artifacts": [list(p) for p in self.incumbent_artifacts],
                "incumbent_evidence_refs": list(self.incumbent_evidence_refs),
                "unchanged_digest": self.unchanged_digest,
                "incumbent_commit": self.incumbent_commit}


# =============================================================================
# Inputs — build and linkage (§10.2 phase 2)
# =============================================================================

#: The two entries that were removed from the global `LD_LIBRARY_PATH` on
#: 2026-07-31 and that `verify_ggml_linkage.sh` exists to stop coming back. A
#: candidate whose environment reintroduces them is loading the FROZEN production
#: kernel's ggml, which is the whole incident.
_FROZEN_LLAMA_LIB_DIRS = (
    "/mnt/raid0/llm/llama.cpp/build/bin",
    "/mnt/raid0/llm/llama.cpp-dflash/build/bin",
)

_LINKAGE_PASS_RE = re.compile(r"^PASS: all linked ggml libraries resolve inside (.+)$",
                              re.MULTILINE)
_LINKAGE_BAD_RE = re.compile(r"^\s*BAD\s+\S+\s+->\s+(\S+)", re.MULTILINE)
_LINKAGE_NO_LIBS = "no ggml/whisper/llama libs in ldd output"


@dataclass(frozen=True)
class LinkageReceipt:
    """A captured run of the research repo's `scripts/utils/verify_ggml_linkage.sh`.

    T3 does not run the verifier — it runs no process at all. It reads the receipt
    the build stage captured, and it checks four things the script itself does not:
    that the verifier invoked was the one in **epyc-inference-research** and not a
    same-named copy elsewhere; that the expected tree root is the candidate's own
    build directory rather than any directory that happens to be consistent; that
    the loader's `LD_LIBRARY_PATH` puts the candidate's own tree first and does not
    reintroduce a frozen-production llama directory; and that `ldd` actually found
    ggml libraries — the script exits **0** when it finds none, which is the one
    fail-open in an otherwise fail-closed instrument.
    """

    backend: str
    binary_path: str
    expected_tree_root: str
    verifier_path: str
    verifier_sha256: str
    exit_code: int
    stdout: str
    ld_library_path: tuple
    observed_at: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(f"LinkageReceipt.backend: {self.backend!r} is not a backend")
        _text(self.binary_path, "LinkageReceipt.binary_path")
        _text(self.expected_tree_root, "LinkageReceipt.expected_tree_root")
        _text(self.verifier_path, "LinkageReceipt.verifier_path")
        _sha256(self.verifier_sha256, "LinkageReceipt.verifier_sha256")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise T3InputError("LinkageReceipt.exit_code: required, an int")
        if not isinstance(self.stdout, str):
            raise T3InputError("LinkageReceipt.stdout: required, a string")
        object.__setattr__(self, "ld_library_path", _str_tuple(
            self.ld_library_path, "LinkageReceipt.ld_library_path", non_empty=False))
        _timestamp(self.observed_at, "LinkageReceipt.observed_at")

    def check(self, *, expected_build_dir: Optional[str] = None) -> schemas.Check:
        reasons: list = []
        unknown: list = []

        if not self.verifier_path.endswith(LINKAGE_VERIFIER_RELPATH):
            reasons.append(
                f"the linkage receipt names {self.verifier_path!r}; §10.2 phase 2 "
                f"requires {LINKAGE_VERIFIER_RELPATH} from the research repo. CLAUDE.md "
                "cites it unqualified, which is the same defect class as the durability "
                "validator's path in MEASUREMENT.md:155, so the path is checked here."
            )
        if "epyc-root" in self.verifier_path:
            reasons.append(
                f"the verifier was invoked from {self.verifier_path!r}: the script lives "
                "in epyc-inference-research, and an epyc-root copy is a fork of a "
                "measurement instrument"
            )

        if expected_build_dir is None:
            unknown.append(
                f"{self.backend}: the sealed candidate records no build directory, so "
                f"the receipt's tree root {self.expected_tree_root!r} cannot be shown to "
                "be the candidate's own. An unanchored linkage proof is a proof about "
                "some tree, and three ggml generations coexist on this host."
            )
        else:
            want = expected_build_dir.rstrip("/")
            got = self.expected_tree_root.rstrip("/")
            # Containment runs ONE way. The receipt's root must be the build directory
            # or something INSIDE it: "all ggml resolves inside <build>/bin" is a
            # narrower claim about this binary and is admissible. An ANCESTOR is the
            # opposite — "all ggml resolves inside /mnt/raid0/llm" is satisfied by
            # loading the FROZEN production tree's ggml, and "inside /" is satisfied by
            # anything at all. The earlier form accepted ancestors, which made the
            # root, the LD_LIBRARY_PATH lead-entry test that reuses it, and therefore
            # the whole of INC-20260731 defeasible by widening one string.
            if got != want and not got.startswith(want + "/"):
                reasons.append(
                    f"the receipt proved linkage against {self.expected_tree_root!r} but "
                    f"the candidate was built in {expected_build_dir!r}; a linkage proof "
                    "about another tree — or about a directory that merely CONTAINS this "
                    "one — proves nothing about this binary"
                )

        if _LINKAGE_BAD_RE.search(self.stdout):
            bad = _LINKAGE_BAD_RE.findall(self.stdout)
            reasons.append(
                f"the verifier reported libraries resolving outside the tree: {bad}. "
                "A binary that inherits another tree's ggml runs silently wrong "
                "(INC-20260731-ggml-linkage-silent-cpu-fallback)."
            )
        if self.exit_code != 0:
            reasons.append(f"the verifier exited {self.exit_code}")

        if _LINKAGE_NO_LIBS in self.stdout:
            unknown.append(
                "ldd reported no ggml/whisper/llama libraries at all. The verifier exits "
                "0 in that case, which is its one fail-open: 'nothing was checked' is not "
                "'everything resolved correctly'."
            )
        elif not _LINKAGE_PASS_RE.search(self.stdout):
            unknown.append(
                "the receipt carries no `PASS: all linked ggml libraries resolve inside "
                "…` line; an exit code without the statement it summarises is not a proof"
            )

        if not self.ld_library_path:
            unknown.append(
                "the receipt records no LD_LIBRARY_PATH; per-tree isolation cannot be "
                "confirmed and three ggml generations coexist on this host")
        else:
            first = self.ld_library_path[0].rstrip("/")
            root = self.expected_tree_root.rstrip("/")
            if not (first == root or first.startswith(root + "/")):
                reasons.append(
                    f"LD_LIBRARY_PATH begins with {self.ld_library_path[0]!r}, outside "
                    f"{self.expected_tree_root!r}. Every launcher must set its OWN "
                    "LD_LIBRARY_PATH and the loader honours it before the binary's own "
                    "directory (CLAUDE.md, 2026-07-31 speech-kernel freeze)."
                )
            for entry in self.ld_library_path:
                normalised = entry.rstrip("/")
                if normalised in _FROZEN_LLAMA_LIB_DIRS and not root.startswith(normalised):
                    reasons.append(
                        f"LD_LIBRARY_PATH reintroduces {entry!r}, a FROZEN production "
                        "kernel library directory removed from the global environment on "
                        "2026-07-31; the verifier exists to stop exactly this"
                    )

        if reasons:
            return _fail(*reasons)
        if unknown:
            return _cnc(*unknown)
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "binary_path": self.binary_path,
                "expected_tree_root": self.expected_tree_root,
                "verifier_path": self.verifier_path,
                "verifier_sha256": self.verifier_sha256, "exit_code": self.exit_code,
                "ld_library_path": list(self.ld_library_path),
                "observed_at": self.observed_at}


@dataclass(frozen=True)
class BackendInventory:
    """The ABI/backend inventory the built binary reports for one backend.

    A GPU backend whose inventory carries no device entry is the 2026-07-31
    whisper failure in inventory form: `use gpu = 1` reports what was REQUESTED,
    never what was loaded.
    """

    backend: str
    #: e.g. ("CPU", "HIP") — whatever the binary enumerates at startup.
    entries: tuple
    device_entries: tuple = ()
    source_ref: str = ""

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(f"BackendInventory.backend: {self.backend!r} is not a backend")
        object.__setattr__(self, "entries", _str_tuple(
            self.entries, "BackendInventory.entries", non_empty=False))
        object.__setattr__(self, "device_entries", _str_tuple(
            self.device_entries, "BackendInventory.device_entries", non_empty=False))
        _text(self.source_ref, "BackendInventory.source_ref")

    def check(self) -> schemas.Check:
        if not self.entries:
            return _cnc(f"{self.backend}: the binary enumerated no backends at all; the "
                        "ABI/backend inventory is empty, not confirmed")
        if self.backend == "llama_gpu" and not self.device_entries:
            return _fail(
                "llama_gpu: the inventory names no device. A HIP build that enumerates "
                "no device runs on the CPU while still reporting a GPU was requested "
                "(INC-20260731-ggml-linkage-silent-cpu-fallback)."
            )
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "entries": list(self.entries),
                "device_entries": list(self.device_entries),
                "source_ref": self.source_ref}


@dataclass(frozen=True)
class DeterminismDeclaration:
    """Invariant 12: *"A determinism class is an interface."*

    A candidate may not SILENTLY change same-seed run-to-run bitwise stability; a
    change of class is a declared, release-relevant property. So an undeclared
    change FAILs, a declared one passes and is recorded, and `not_measured` on
    either side is COULD_NOT_CHECK rather than "unchanged".
    """

    backend: str
    anchor_class: str
    candidate_class: str
    change_declared: bool = False
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(
                f"DeterminismDeclaration.backend: {self.backend!r} is not a backend")
        for name in ("anchor_class", "candidate_class"):
            value = getattr(self, name)
            if value not in schemas.DETERMINISM_CLASSES:
                raise T3InputError(
                    f"DeterminismDeclaration.{name}: {value!r} is not one of "
                    f"{sorted(schemas.DETERMINISM_CLASSES)}")
        _bool(self.change_declared, "DeterminismDeclaration.change_declared")
        _text(self.evidence_ref, "DeterminismDeclaration.evidence_ref")

    def check(self) -> schemas.Check:
        if "not_measured" in (self.anchor_class, self.candidate_class):
            return _cnc(
                f"{self.backend}: determinism class is {self.anchor_class} -> "
                f"{self.candidate_class}; an unmeasured class is not an unchanged one")
        if self.anchor_class == self.candidate_class:
            return schemas.Check(schemas.PASS)
        if self.change_declared:
            return schemas.Check(schemas.PASS)
        return _fail(
            f"{self.backend}: determinism class changed {self.anchor_class} -> "
            f"{self.candidate_class} without being declared. Invariant 12 makes the "
            "class an interface, and an undeclared interface change is the definition "
            "of a silent one."
        )

    def to_dict(self) -> dict:
        return {"backend": self.backend, "anchor_class": self.anchor_class,
                "candidate_class": self.candidate_class,
                "change_declared": self.change_declared,
                "evidence_ref": self.evidence_ref,
                "check": _check_dict(self.check())}


# =============================================================================
# Inputs — the matrix
# =============================================================================

@dataclass(frozen=True)
class Cell:
    """One row of the derived release matrix (§10.1).

    `claim` is what the cell LICENSES in the release receipt. A waived cell
    suppresses exactly that string — which is how v8 shipped with
    `q8_claim: "none"` rather than with a quietly weaker Q8 claim.
    """

    cell_id: str
    backend: str
    release_phase: str
    protocol_id: str
    recipe_class: str
    metric: str
    metric_direction: str
    #: `prefill` / `decode` for llama; backend-specific elsewhere; None when the
    #: cell is not a throughput cell at all (correctness, capacity, stability).
    workload_phase: Optional[str] = None
    claim: Optional[str] = None
    roles_protected: tuple = ()
    co_resident: bool = False
    reps: Optional[int] = None
    scope_denominator: Optional[Mapping] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.cell_id, "Cell.cell_id")
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(f"Cell.backend: {self.backend!r} is not a known backend")
        if self.release_phase not in CELL_PHASES:
            raise T3InputError(
                f"Cell.release_phase: {self.release_phase!r} is not one of "
                f"{list(CELL_PHASES)}. Identity, linkage, the transaction dry-run and "
                "the seal are adjudicated from receipts, not from matrix cells."
            )
        _text(self.protocol_id, "Cell.protocol_id")
        if self.recipe_class not in RECIPE_CLASSES:
            raise T3InputError(
                f"Cell.recipe_class: {self.recipe_class!r} is not one of "
                f"{sorted(RECIPE_CLASSES)}")
        _text(self.metric, "Cell.metric")
        if self.metric_direction not in schemas.METRIC_DIRECTIONS:
            raise T3InputError(
                f"Cell.metric_direction: {self.metric_direction!r} is not one of "
                f"{sorted(schemas.METRIC_DIRECTIONS)}")
        _opt_text(self.workload_phase, "Cell.workload_phase")
        _opt_text(self.claim, "Cell.claim")
        _opt_text(self.model, "Cell.model")
        object.__setattr__(self, "roles_protected", _str_tuple(
            self.roles_protected, "Cell.roles_protected", non_empty=False))
        _bool(self.co_resident, "Cell.co_resident")
        if self.reps is not None:
            if not isinstance(self.reps, int) or isinstance(self.reps, bool) or self.reps < 1:
                raise T3InputError("Cell.reps: must be a positive int when present")
        if self.scope_denominator is not None and not isinstance(self.scope_denominator,
                                                                 Mapping):
            raise T3InputError("Cell.scope_denominator: must be a mapping when present")
        if self.recipe_class == RECIPE_DIAGNOSTIC and self.claim:
            raise T3InputError(
                f"Cell {self.cell_id!r}: a diagnostic cell carries a claim "
                f"({self.claim!r}). Invariant 15 — baseline/off-recipe cells never "
                "justify a release, so they license nothing."
            )

    @property
    def gating(self) -> bool:
        """Invariant 15: only production-optimal cells gate."""
        return self.recipe_class == RECIPE_PRODUCTION_OPTIMAL

    def to_dict(self) -> dict:
        return {"cell_id": self.cell_id, "backend": self.backend,
                "release_phase": self.release_phase, "protocol_id": self.protocol_id,
                "recipe_class": self.recipe_class, "metric": self.metric,
                "metric_direction": self.metric_direction,
                "workload_phase": self.workload_phase, "claim": self.claim,
                "roles_protected": list(self.roles_protected),
                "co_resident": self.co_resident, "reps": self.reps,
                "model": self.model, "gating": self.gating}


#: The facets of a cell that the release PLAN derives (§12) and that this gate then
#: reads back off the result. `reps`, `scope_denominator` and the evidence refs are
#: deliberately absent: those are facts about the measurement that the compiler,
#: running before anything was measured, cannot know.
_CELL_SCOPE_FACETS = (
    "backend", "release_phase", "protocol_id", "recipe_class", "metric",
    "metric_direction", "workload_phase", "claim", "co_resident", "roles_protected",
    "model",
)


def _cell_scope_drift(planned: "Cell", measured: "Cell") -> tuple:
    """Scope facets on which a measured cell contradicts the planned cell of that id."""
    drift: list = []
    for facet in _CELL_SCOPE_FACETS:
        want = getattr(planned, facet)
        got = getattr(measured, facet)
        if want != got:
            drift.append(f"{facet} (plan {want!r}, result {got!r})")
    return tuple(drift)


@dataclass(frozen=True)
class CellResult:
    """What the trusted evaluator produced for one cell, plus its evidence binding.

    `raw_samples_ref` and `reducer_id` are required for a gating cell because *"a
    record whose reduction cannot be recomputed from its raw samples is INVALID"*
    (P-AK-SEARCH-1 record grammar). A release cell with no raw samples is a number
    with no way back to what produced it.
    """

    cell: Cell
    check: schemas.Check
    raw_samples_ref: Optional[str] = None
    reducer_id: Optional[str] = None
    evidence_ref: Optional[str] = None
    notes: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cell, Cell):
            raise T3InputError("CellResult.cell: must be a Cell")
        if not isinstance(self.check, schemas.Check):
            raise T3InputError("CellResult.check: must be a schemas.Check")
        _opt_text(self.raw_samples_ref, "CellResult.raw_samples_ref")
        _opt_text(self.reducer_id, "CellResult.reducer_id")
        _opt_text(self.evidence_ref, "CellResult.evidence_ref")
        object.__setattr__(self, "notes", _str_tuple(
            self.notes, "CellResult.notes", non_empty=False))

    def evidence_check(self) -> schemas.Check:
        if not self.cell.gating:
            return schemas.Check(schemas.PASS)
        missing = [name for name in ("raw_samples_ref", "reducer_id")
                   if getattr(self, name) is None]
        if missing:
            return _cnc(
                f"{self.cell.cell_id}: gating cell does not bind {missing}; its "
                "reduction cannot be recomputed from its raw samples")
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"cell": self.cell.to_dict(), "check": _check_dict(self.check),
                "raw_samples_ref": self.raw_samples_ref, "reducer_id": self.reducer_id,
                "evidence_ref": self.evidence_ref, "notes": list(self.notes)}


@dataclass(frozen=True)
class PhaseStanding:
    """The §1.6 standing of one (backend, workload phase) at production-optimal recipe.

    T3 does NOT re-derive it. `evaluator.statistics` owns the e-process; this gate
    adjudicates the RULE over standings that instrument produced, which is why a
    standing carries its own protocol id and its evidence reference.
    """

    backend: str
    workload_phase: str
    protocol_id: str
    standing: str
    cell_ids: tuple
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(f"PhaseStanding.backend: {self.backend!r} is not a backend")
        _text(self.workload_phase, "PhaseStanding.workload_phase")
        _text(self.protocol_id, "PhaseStanding.protocol_id")
        if self.standing not in STANDINGS:
            raise T3InputError(
                f"PhaseStanding.standing: {self.standing!r} is not one of {list(STANDINGS)}")
        object.__setattr__(self, "cell_ids", _str_tuple(
            self.cell_ids, "PhaseStanding.cell_ids", non_empty=False))
        _text(self.evidence_ref, "PhaseStanding.evidence_ref")

    def to_dict(self) -> dict:
        return {"backend": self.backend, "workload_phase": self.workload_phase,
                "protocol_id": self.protocol_id, "standing": self.standing,
                "cell_ids": list(self.cell_ids), "evidence_ref": self.evidence_ref}


@dataclass(frozen=True)
class PhaseTradeException:
    """§1.6: a phase trade is *"a pre-declared campaign exception"*, and an operator
    decision at freeze time — never a controller decision.

    It must name the exact regression band, the exact expected gain, and the roles
    affected. Declared after the campaign started, it is not a pre-declaration; the
    gate says so rather than honouring it.
    """

    backend: str
    regressing_phase: str
    regression_band: tuple
    gaining_phase: str
    expected_gain: float
    roles_affected: tuple
    declared_at: str
    campaign_start_at: str
    operator_approved: bool
    approved_by: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(
                f"PhaseTradeException.backend: {self.backend!r} is not a backend")
        _text(self.regressing_phase, "PhaseTradeException.regressing_phase")
        _text(self.gaining_phase, "PhaseTradeException.gaining_phase")
        if self.regressing_phase == self.gaining_phase:
            raise T3InputError(
                "PhaseTradeException: the regressing and gaining phases are the same; a "
                "trade between a phase and itself is not a trade")
        band = tuple(self.regression_band)
        if len(band) != 2:
            raise T3InputError("PhaseTradeException.regression_band: expected (lo, hi)")
        for value in band:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise T3InputError("PhaseTradeException.regression_band: numbers only")
            if value != value or value in (float("inf"), float("-inf")):
                raise T3InputError(
                    "PhaseTradeException.regression_band: an unbounded band is not an "
                    "exact band, and §1.6 requires the exact regression band")
        if band[0] > band[1]:
            raise T3InputError("PhaseTradeException.regression_band: lo exceeds hi")
        if band[1] > 0:
            # The other half of the refusal `readiness.PhaseTradeException` makes at
            # declaration, and it was the half left behind. The band is oriented, so
            # a regression band's bounds are at or below zero; a band whose upper
            # bound is positive describes a GAIN, and a trade that names a gain as
            # the thing it is buying permission for is admitting a regression the
            # gate can no longer recognise as one.
            raise T3InputError(
                f"PhaseTradeException.regression_band: {list(band)} has a positive "
                "upper bound, so it does not describe a regression. §1.6 names the "
                "exact REGRESSION band, in the phase's own oriented scale.")
        object.__setattr__(self, "regression_band", band)
        if not isinstance(self.expected_gain, (int, float)) or \
                isinstance(self.expected_gain, bool):
            raise T3InputError("PhaseTradeException.expected_gain: required, a number")
        if self.expected_gain != self.expected_gain or \
                self.expected_gain in (float("inf"), float("-inf")):
            raise T3InputError(
                "PhaseTradeException.expected_gain: must be finite; §1.6 requires the "
                "EXACT expected gain, and an unbounded one names no quantity")
        if self.expected_gain <= 0:
            # Same refusal `readiness.PhaseTradeException` already makes at
            # declaration. A trade buys a regression with a gain; one that expects no
            # gain is an unpriced regression, and the gate would then be comparing a
            # realised effect against nothing.
            raise T3InputError(
                "PhaseTradeException.expected_gain: must be strictly positive; a trade "
                "with no expected gain is a regression, not a trade")
        object.__setattr__(self, "roles_affected", _str_tuple(
            self.roles_affected, "PhaseTradeException.roles_affected"))
        _timestamp(self.declared_at, "PhaseTradeException.declared_at")
        _timestamp(self.campaign_start_at, "PhaseTradeException.campaign_start_at")
        _bool(self.operator_approved, "PhaseTradeException.operator_approved")
        _text(self.approved_by, "PhaseTradeException.approved_by")

    def check(self) -> schemas.Check:
        reasons: list = []
        if _timestamp(self.declared_at, "declared_at") > \
                _timestamp(self.campaign_start_at, "campaign_start_at"):
            reasons.append(
                f"the {self.backend} phase-trade exception was declared at "
                f"{self.declared_at}, after the campaign started at "
                f"{self.campaign_start_at}. §1.6 requires a PRE-declared exception; one "
                "written after the regression was observed is a rationalisation."
            )
        if not self.operator_approved:
            reasons.append(
                f"the {self.backend} phase-trade exception is not operator-approved; "
                "§1.6 makes it an operator decision at freeze time, not a controller one")
        # `operator_approved` is a BOOLEAN the caller sets, and `approved_by` was the
        # only place the approver is named — unguarded, while the identically-shaped
        # `authorized_by` on a §10.4 waiver is refused a machine name. A phase trade
        # admits a REGRESSION into a release; that is the same authority a waiver
        # carries, so it meets the same vocabulary. A FINDING rather than a
        # constructor refusal, because "the controller approved it" must stay
        # expressible for the gate to be able to say no to it.
        approver_tokens = schemas.machine_actor_tokens(self.approved_by)
        if approver_tokens:
            reasons.append(
                f"the {self.backend} phase-trade exception is approved by "
                f"{self.approved_by!r}, a machine actor "
                f"({', '.join(approver_tokens)}). §1.6 makes the trade an operator "
                "decision at freeze time, not a controller one; a loop that approves "
                "its own regression has not been granted an exception, it has taken "
                "one (MEASUREMENT.md:140-142).")
        return _fail(*reasons) if reasons else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "regressing_phase": self.regressing_phase,
                "regression_band": list(self.regression_band),
                "gaining_phase": self.gaining_phase, "expected_gain": self.expected_gain,
                "roles_affected": list(self.roles_affected),
                "declared_at": self.declared_at,
                "campaign_start_at": self.campaign_start_at,
                "operator_approved": self.operator_approved,
                "approved_by": self.approved_by,
                "check": _check_dict(self.check())}


@dataclass(frozen=True)
class CapacityFloor:
    """§10.2 phase 7: *"every protected cell within its fixed floor"*."""

    cell_id: str
    quantity: str
    floor: float
    observed: float
    direction: str
    unit: str = ""

    def __post_init__(self) -> None:
        _text(self.cell_id, "CapacityFloor.cell_id")
        _text(self.quantity, "CapacityFloor.quantity")
        for name in ("floor", "observed"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or value != value:
                raise T3InputError(f"CapacityFloor.{name}: required, a finite number")
        if self.direction not in schemas.METRIC_DIRECTIONS:
            raise T3InputError(
                f"CapacityFloor.direction: {self.direction!r} is not one of "
                f"{sorted(schemas.METRIC_DIRECTIONS)}")
        if not isinstance(self.unit, str):
            raise T3InputError("CapacityFloor.unit: must be a string")

    def check(self) -> schemas.Check:
        if self.direction == "higher_better":
            ok = self.observed >= self.floor
        else:
            ok = self.observed <= self.floor
        if ok:
            return schemas.Check(schemas.PASS)
        return _fail(
            f"{self.cell_id}: {self.quantity} is {self.observed}{self.unit}, outside its "
            f"fixed floor of {self.floor}{self.unit} ({self.direction})")

    def to_dict(self) -> dict:
        return {"cell_id": self.cell_id, "quantity": self.quantity, "floor": self.floor,
                "observed": self.observed, "direction": self.direction,
                "unit": self.unit, "check": _check_dict(self.check())}


@dataclass(frozen=True)
class QualityEvidence:
    """§10.2 phase 5, and the §10.5 lesson in the place it actually bit.

    The v8 quality gate compared against
    `/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/llama-server` — a
    PRESERVED binary. Rebuilding v7 under a drifted toolchain would not have
    reproduced it, so a quality baseline that names a rebuild rather than an
    archived build FAILs here: the comparison would be against a binary nobody has.
    """

    backend: str
    mode: str
    baseline_binary_path: str
    baseline_binary_sha256: str
    baseline_kernel: str
    baseline_is_rebuild: bool
    evidence_refs: tuple
    #: Required for `transferred`: the paired-parity receipt that PROVED transfer.
    paired_parity_receipt: Optional[str] = None
    suites: tuple = ()
    shared_question_identity: bool = False

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(f"QualityEvidence.backend: {self.backend!r} is not a backend")
        if self.mode not in QUALITY_MODES:
            raise T3InputError(
                f"QualityEvidence.mode: {self.mode!r} is not one of {list(QUALITY_MODES)}")
        _text(self.baseline_binary_path, "QualityEvidence.baseline_binary_path")
        _sha256(self.baseline_binary_sha256, "QualityEvidence.baseline_binary_sha256")
        _text(self.baseline_kernel, "QualityEvidence.baseline_kernel")
        _bool(self.baseline_is_rebuild, "QualityEvidence.baseline_is_rebuild")
        object.__setattr__(self, "evidence_refs", _str_tuple(
            self.evidence_refs, "QualityEvidence.evidence_refs"))
        _opt_text(self.paired_parity_receipt, "QualityEvidence.paired_parity_receipt")
        object.__setattr__(self, "suites", _str_tuple(
            self.suites, "QualityEvidence.suites", non_empty=False))
        _bool(self.shared_question_identity, "QualityEvidence.shared_question_identity")

    def to_dict(self) -> dict:
        return {"backend": self.backend, "mode": self.mode,
                "baseline_binary_path": self.baseline_binary_path,
                "baseline_binary_sha256": self.baseline_binary_sha256,
                "baseline_kernel": self.baseline_kernel,
                "baseline_is_rebuild": self.baseline_is_rebuild,
                "evidence_refs": list(self.evidence_refs),
                "paired_parity_receipt": self.paired_parity_receipt,
                "suites": list(self.suites),
                "shared_question_identity": self.shared_question_identity}


@dataclass(frozen=True)
class StabilityEvidence:
    """§10.2 phase 6 — repeated load/unload, concurrency, memory growth, cleanup."""

    backend: str
    load_unload_cycles: int
    memory_growth_bytes: int
    memory_growth_allowance_bytes: int
    profiler_or_runtime_errors: int
    cleanup_verified: bool
    mixed_prefill_decode_exercised: Optional[bool]
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise T3InputError(
                f"StabilityEvidence.backend: {self.backend!r} is not a backend")
        for name in ("load_unload_cycles", "memory_growth_bytes",
                     "memory_growth_allowance_bytes", "profiler_or_runtime_errors"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise T3InputError(f"StabilityEvidence.{name}: required, a non-negative int")
        _bool(self.cleanup_verified, "StabilityEvidence.cleanup_verified")
        if self.mixed_prefill_decode_exercised is not None:
            _bool(self.mixed_prefill_decode_exercised,
                  "StabilityEvidence.mixed_prefill_decode_exercised")
        _text(self.evidence_ref, "StabilityEvidence.evidence_ref")

    def check(self, *, min_cycles: int) -> schemas.Check:
        reasons: list = []
        unknown: list = []
        if self.load_unload_cycles < min_cycles:
            reasons.append(
                f"{self.backend}: {self.load_unload_cycles} load/unload cycles is below "
                f"the declared release minimum of {min_cycles}")
        if self.memory_growth_bytes > self.memory_growth_allowance_bytes:
            reasons.append(
                f"{self.backend}: memory grew {self.memory_growth_bytes} bytes over the "
                f"soak, above the declared allowance of "
                f"{self.memory_growth_allowance_bytes}")
        if self.profiler_or_runtime_errors:
            reasons.append(
                f"{self.backend}: {self.profiler_or_runtime_errors} profiler/runtime "
                "errors during the stability soak")
        if not self.cleanup_verified:
            reasons.append(
                f"{self.backend}: teardown/cleanup was not verified. "
                "`bench-cpu.md:89-90` makes a cleanup failure a FAIL regardless of "
                "throughput.")
        if self.mixed_prefill_decode_exercised is None:
            unknown.append(
                f"{self.backend}: whether mixed prefill/decode concurrency was exercised "
                "is unrecorded, so §10.2 phase 6's concurrency clause is unchecked")
        if reasons:
            return _fail(*reasons)
        if unknown:
            return _cnc(*unknown)
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "load_unload_cycles": self.load_unload_cycles,
                "memory_growth_bytes": self.memory_growth_bytes,
                "memory_growth_allowance_bytes": self.memory_growth_allowance_bytes,
                "profiler_or_runtime_errors": self.profiler_or_runtime_errors,
                "cleanup_verified": self.cleanup_verified,
                "mixed_prefill_decode_exercised": self.mixed_prefill_decode_exercised,
                "evidence_ref": self.evidence_ref}


# =============================================================================
# Inputs — the transaction (§10.2 phase 8) and the §10.5 incumbent archive
# =============================================================================

@dataclass(frozen=True)
class ArchivedBuild:
    """One preserved incumbent build — binaries AND linked libraries (§10.5).

    Libraries are not optional, and neither is their BACKEND ATTRIBUTION. The three
    ggml generations on this host are precisely why: a preserved binary whose
    libraries were not preserved with it resolves against whatever is on the path at
    rollback time, which is the 2026-07-31 incident with a longer fuse — and a
    preserved library nobody attributed cannot be shown to be the one the backend
    being rolled back actually linked. `libraries` is therefore
    `((backends, path, sha256), …)`, and the attribution is recorded HERE, at the
    source, where the archiving actor knows it. `packager.RollbackPlan` carries it
    forward unchanged; it must never mint one, because a rollback plan that invented
    an attribution would put a fact in the operator's package that nothing measured.
    """

    generation: str
    branch: str
    commit: str
    archive_root: str
    binaries: tuple
    #: ((backends, path, sha256), …) — `backends` is a non-empty tuple of backend
    #: names, plural because one `libggml-base.so.0` legitimately serves both llama
    #: backends of one tree.
    libraries: tuple
    rebuilt: bool = False

    def __post_init__(self) -> None:
        if self.generation not in ARCHIVE_GENERATIONS:
            raise T3InputError(
                f"ArchivedBuild.generation: {self.generation!r} is not one of "
                f"{list(ARCHIVE_GENERATIONS)}")
        _text(self.branch, "ArchivedBuild.branch")
        _commit(self.commit, "ArchivedBuild.commit")
        _text(self.archive_root, "ArchivedBuild.archive_root")
        object.__setattr__(self, "binaries", _hashed_pairs(
            self.binaries, "ArchivedBuild.binaries"))
        object.__setattr__(self, "libraries", _attributed_hashed_triples(
            self.libraries, "ArchivedBuild.libraries"))
        _bool(self.rebuilt, "ArchivedBuild.rebuilt")

    def libraries_for(self, backend: str) -> tuple:
        """The archived libraries this entry attributes to `backend`."""
        return tuple((path, digest) for backends, path, digest in self.libraries
                     if backend in backends)

    @property
    def attributed_backends(self) -> tuple:
        return tuple(sorted({b for backends, _p, _d in self.libraries for b in backends}))

    def check(self) -> schemas.Check:
        reasons: list = []
        if self.rebuilt:
            reasons.append(
                f"{self.generation} ({self.branch}) is a REBUILD, not an archive. §10.5: "
                "the v8 quality gate compared against a preserved binary and "
                "\"rebuilding an old commit under a drifted toolchain does not reproduce "
                "it\"."
            )
        if storage.is_scratch_path(self.archive_root):
            reasons.append(
                f"{self.generation}: the archive root {self.archive_root!r} is a scratch "
                "path, one sweep away from being unverifiable "
                "(MEASUREMENT.md:146-156)")
        if _under_production_tree(self.archive_root):
            reasons.append(
                f"{self.generation}: the archive root {self.archive_root!r} is inside a "
                "FROZEN production tree; an archive that lives in the thing it protects "
                "against is not a rollback target")
        return _fail(*reasons) if reasons else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"generation": self.generation, "branch": self.branch,
                "commit": self.commit, "archive_root": self.archive_root,
                "binaries": [list(p) for p in self.binaries],
                "libraries": [{"backends": list(backends), "path": path,
                               "sha256": digest}
                              for backends, path, digest in self.libraries],
                "rebuilt": self.rebuilt}


@dataclass(frozen=True)
class IncumbentArchive:
    """§10.5: *"Incumbent builds are archived, not merely rebuildable."*

    `/mnt/raid0/llm/kernels/archive/` exists and is EMPTY (design §2), so this is
    the check most likely to fire on a real first run. N-1 is required; N-2 is
    "ideally" and its absence is a note, not a failure. A release with no incumbent
    at all — the speech-v1 shape — must SAY so, because the alternative is a
    rollback plan with no anchor and nobody noticing.
    """

    entries: tuple = ()
    no_incumbent_reason: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", _typed_tuple(
            self.entries, "IncumbentArchive.entries", ArchivedBuild))
        _opt_text(self.no_incumbent_reason, "IncumbentArchive.no_incumbent_reason")
        generations = [e.generation for e in self.entries]
        if len(set(generations)) != len(generations):
            raise T3InputError(
                f"IncumbentArchive.entries: duplicate generations {generations}")
        if self.entries and self.no_incumbent_reason is not None:
            raise T3InputError(
                "IncumbentArchive: entries are present alongside a no_incumbent_reason; "
                "one of the two statements is false")

    def entry(self, generation: str) -> Optional[ArchivedBuild]:
        for candidate in self.entries:
            if candidate.generation == generation:
                return candidate
        return None

    def check(self) -> tuple:
        """Returns (check, notes)."""
        notes: list = []
        n1 = self.entry(ARCHIVE_GENERATION_N1)
        if n1 is None:
            if self.no_incumbent_reason is None:
                return (_fail(
                    "no N-1 incumbent build is archived and no reason is recorded for "
                    "its absence. §10.5: the freeze transaction must archive the "
                    "incumbent's built binaries and linked libraries, because the v8 "
                    "quality gate compared against a PRESERVED v7 binary and "
                    "/mnt/raid0/llm/kernels/archive/ is empty."
                ), tuple(notes))
            return (_cnc(
                f"no incumbent is archived; the recorded reason is "
                f"{self.no_incumbent_reason!r}. A release with no archived incumbent has "
                "no binary-level rollback target, which is a release-relevant fact rather "
                "than a passing one."
            ), tuple(notes))
        checks = [n1.check()]
        n2 = self.entry(ARCHIVE_GENERATION_N2)
        if n2 is None:
            notes.append(
                "N-2 is not archived. §10.5 asks for N-1 \"and ideally N-2\"; the second "
                "generation is a note, not a gate.")
        else:
            checks.append(n2.check())
        return (_worst(checks), tuple(notes))

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries],
                "no_incumbent_reason": self.no_incumbent_reason}


@dataclass(frozen=True)
class TransactionPlan:
    """§10.2 phase 8 — the exact transaction, as a DRY RUN.

    `executed` exists only so it can be refused. §11.2: the packager *"may not …
    execute any command it drafted"*, and a transaction handed to the gate with
    `executed=True` has already crossed the boundary the gate exists to hold.
    """

    next_branch: str
    next_version_number: int
    next_tag: str
    install_path: str
    #: (link_path, current_target, next_target)
    symlink_diff: tuple
    service_impact: tuple
    #: Drafts for the operator. Every entry must be marked `draft: True` — §1.3
    #: item 2: the era row is an OUTPUT of the package, never a write.
    era_actions: tuple
    receipt_paths: tuple
    rollback_branch: Optional[str] = None
    rollback_head: Optional[str] = None
    executed: bool = False

    def __post_init__(self) -> None:
        _text(self.next_branch, "TransactionPlan.next_branch")
        if not isinstance(self.next_version_number, int) or \
                isinstance(self.next_version_number, bool) or self.next_version_number < 1:
            raise T3InputError(
                "TransactionPlan.next_version_number: required, a positive int")
        _text(self.next_tag, "TransactionPlan.next_tag")
        _text(self.install_path, "TransactionPlan.install_path")
        symlinks: list = []
        for i, entry in enumerate(self.symlink_diff or ()):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise T3InputError(
                    f"TransactionPlan.symlink_diff[{i}]: expected "
                    "(link_path, current_target, next_target)")
            symlinks.append(tuple(_text(v, f"TransactionPlan.symlink_diff[{i}]")
                                  for v in entry))
        object.__setattr__(self, "symlink_diff", tuple(symlinks))
        object.__setattr__(self, "service_impact", _str_tuple(
            self.service_impact, "TransactionPlan.service_impact", non_empty=False))
        era_actions = tuple(self.era_actions or ())
        for i, action in enumerate(era_actions):
            if not isinstance(action, Mapping):
                raise T3InputError(f"TransactionPlan.era_actions[{i}]: expected a mapping")
            if action.get("draft") is not True:
                raise T3InputError(
                    f"TransactionPlan.era_actions[{i}]: must carry draft=True. §1.3 item "
                    "2 makes the era-registry row an output of the package FOR the "
                    "operator, never a write the loop performs."
                )
        object.__setattr__(self, "era_actions", era_actions)
        object.__setattr__(self, "receipt_paths", _str_tuple(
            self.receipt_paths, "TransactionPlan.receipt_paths"))
        _opt_text(self.rollback_branch, "TransactionPlan.rollback_branch")
        if self.rollback_head is not None:
            _commit(self.rollback_head, "TransactionPlan.rollback_head")
        _bool(self.executed, "TransactionPlan.executed")
        if self.executed:
            raise ProductionWriteRefused(
                "TransactionPlan.executed is True. T3 grades a DRY RUN. A transaction "
                "that has already been applied cannot be gated, and grading it after the "
                "fact would read as retrospective authorisation for a write that crossed "
                "four human-only trust boundaries (MEASUREMENT.md:140-142)."
            )

    def to_dict(self) -> dict:
        return {"next_branch": self.next_branch,
                "next_version_number": self.next_version_number,
                "next_tag": self.next_tag, "install_path": self.install_path,
                "symlink_diff": [list(s) for s in self.symlink_diff],
                "service_impact": list(self.service_impact),
                "era_actions": [dict(a) for a in self.era_actions],
                "receipt_paths": list(self.receipt_paths),
                "rollback_branch": self.rollback_branch,
                "rollback_head": self.rollback_head,
                "executed": self.executed}


# =============================================================================
# Inputs — operator waivers (§10.4), the human-authored first-class input
# =============================================================================

#: epyc-root's working path (CLAUDE.md "Repository Map"). The trust-boundary
#: manifest lives in epyc-root, this module lives in epyc-inference-research, and
#: there is exactly one edge between them: a READ of one file.
EPYC_ROOT = Path("/workspace")

#: The manifest itself. `schemas.HUMAN_ONLY_PATHS_MANIFEST` owns the relative path
#: so the two planes cannot disagree about which file is the boundary.
TRUST_BOUNDARY_MANIFEST = EPYC_ROOT / schemas.HUMAN_ONLY_PATHS_MANIFEST

#: Where the operator's attestations ACTUALLY live, as absolute, symlink-resolved
#: roots. `waiver_binding_from_path` requires the resolved document to sit under one
#: of these IN ADDITION to passing `schemas.operator_owned_path_check`.
#:
#: The two answer different questions and both are needed. The schemas check is the
#: CITATION-SHAPE authority: it deals in the repo-relative vocabulary the manifest is
#: written in, and so "operator-owned" there means *spelled `artifacts/operator/…`
#: inside a known repository checkout* — which today includes
#: `/mnt/raid0/llm/epyc-inference-research/artifacts/operator/`, a directory this
#: repository's own agents can create with `mkdir -p`. That is correct for a citation
#: check and wrong for a READ, so the narrowing lives here, in the reader, and
#: `operator_owned_path_check` is left exactly as the hardened session left it: a
#: check whose answer cannot be widened by any caller.
DEFAULT_ATTESTATION_ROOTS = (
    str(Path(EPYC_ROOT / schemas.OPERATOR_ATTESTATION_ROOT).resolve()),
)


def human_only_boundary(manifest: Optional[Any] = None) -> schemas.TrustBoundary:
    """Read `human_only_paths.yaml` — the trust boundary, as data (§1.3 item 4).

    The ONLY filesystem edge in the waiver path, and it is a read. `schemas.py`
    performs no I/O and therefore cannot load its own boundary; it parses, this
    reads. An absent, unreadable, empty or foreign manifest yields an UNREADABLE
    boundary, and `schemas.operator_owned_path_check` answers COULD_NOT_CHECK
    rather than PASS on one — so a caller cannot widen what counts as
    operator-owned by deleting the file this check inspects.

    Not cached: the manifest is human-amendment-only and tiny, and a cache would
    make the gate's answer depend on which run warmed it.
    """
    target = Path(manifest) if manifest is not None else TRUST_BOUNDARY_MANIFEST
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return schemas.TrustBoundary(source=str(target))
    return schemas.parse_trust_boundary(text, source=str(target))


@dataclass(frozen=True)
class WaiverBinding:
    """A waiver as SOMEBODY QUOTED IT: pinned hash, quoted hash, document, coverage.

    This is the QUOTATION type, and every one of its five facts is a caller
    assertion. `document`, `document_path` and `observed_sha256` are three
    INDEPENDENT assertions in particular: nothing here reads a file, so
    `WaiverBinding(document_path="artifacts/operator/w.json", document={...},
    observed_sha256=<sha of that dict>)` is constructible with no filesystem at all,
    and the digest it pins is a digest of bytes the party being gated handed over.

    That is not a defect in this type — it is what a quotation IS. It is a defect
    only if a quotation can suppress something, so it cannot:

      * `verify_waiver` evaluates a `read` predicate FIRST, and only a `ReadWaiver`
        (the return type of `waiver_binding_from_path`, the sole constructor that
        opens the file) can satisfy it. A quotation is at best COULD_NOT_CHECK.
      * `run_t3` turns COULD_NOT_CHECK into a BLOCKING identity-phase reason, so an
        unread waiver does not merely suppress nothing — it stops the run.
      * `WaiverVerification.covered_cell_ids` is `()` unless the check PASSed, so a
        refused waiver cannot even land in the durable bundle looking like coverage.

    `observed_sha256` stays Optional, and stays a caller assertion, ON PURPOSE. The
    honest answer to *"did you read the file?"* is sometimes no, and that answer
    must be expressible: a design that made "unread" INEXPRESSIBLE would push
    callers into asserting a digest they never computed, which is strictly worse
    than a quotation that is labelled as one. Unread is available, and it fails
    CLOSED.

    A REFUSAL to read is a different thing from an unread waiver: when
    `waiver_binding_from_path` is asked to read and cannot — absent, symlinked,
    oversized, hash-mismatched — it RAISES `WaiverNotReadable`. It never degrades
    into one of these, because "the reader refused these bytes" must never be
    recorded as "nobody looked".
    """

    waiver_id: str
    pinned_sha256: str
    document: Mapping
    document_path: str
    covers_cell_ids: tuple
    observed_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.waiver_id, "WaiverBinding.waiver_id")
        _sha256(self.pinned_sha256, "WaiverBinding.pinned_sha256")
        if not isinstance(self.document, Mapping):
            raise T3InputError("WaiverBinding.document: must be a mapping")
        _text(self.document_path, "WaiverBinding.document_path")
        object.__setattr__(self, "covers_cell_ids", _str_tuple(
            self.covers_cell_ids, "WaiverBinding.covers_cell_ids"))
        if self.observed_sha256 is not None:
            _sha256(self.observed_sha256, "WaiverBinding.observed_sha256")

    @property
    def was_read(self) -> bool:
        """Did THIS PROCESS read these bytes? Computed, never declared.

        Deliberately not a `read: bool` field: a flag makes ONE type carry TWO
        meanings, so every downstream consumer must remember to ask, and forgetting
        is silent.

        Equally deliberately, this is not `return False` on the quotation type and
        `return True` on `ReadWaiver`. A per-class constant is a fact about the
        class, and a class is something a caller writes — a three-line subclass
        answered True with no receipt at all. It delegates to
        `waiver_read_violations`, which inspects the mint token on the receipt, so
        the answer is a fact about the OBJECT. Overriding this property still lies to
        anything that reads it, which is why the gate (`verify_waiver`) and the
        package call the function and never the property.
        """
        return not waiver_read_violations(self)

    def to_dict(self) -> dict:
        return {"waiver_id": self.waiver_id, "pinned_sha256": self.pinned_sha256,
                "document_path": self.document_path,
                "covers_cell_ids": list(self.covers_cell_ids),
                "observed_sha256": self.observed_sha256,
                "read": not waiver_read_violations(self), "read_receipt": None,
                "schema": self.document.get("schema")}


#: Minted at import, named by exactly ONE function in this package
#: (`waiver_binding_from_path`), and absent from `__all__`. A receipt cannot be
#: constructed without it.
#:
#: Python has no real privacy, so a sentinel is unforgeable only by convention —
#: which is why it sits on the RECEIPT, an object with no other reason to exist,
#: rather than on the binding, which has many. Forging a read therefore requires
#: writing the literal token name into source, which is greppable, and
#: `audit_reader_token_is_named_once` greps for it.
_READER_TOKEN = object()


class WaiverNotReadable(T3Error):
    """The reader was asked to read a waiver and refused.

    NOT a subclass of `T3InputError` and NOT convertible into an unread
    `WaiverBinding`: a caller that asked to read and got a refusal must not be able
    to record that as "nobody looked". Every refusal below is a fact about the
    bytes or the location, never about the caller's typing.
    """


@dataclass(frozen=True)
class WaiverReadReceipt:
    """What the reader OBSERVED, minted only by `waiver_binding_from_path`.

    Every field is a measurement, not an assertion. `document` is the object
    `json.loads` produced from the very `bytes` object `bytes_sha256` digests —
    `ReadWaiver.__post_init__` asserts that by OBJECT IDENTITY, which is what makes
    "the document returned is the one whose bytes were hashed" a fact rather than a
    hope. Hashing a file and separately parsing it proves nothing about the parsed
    object.
    """

    resolved_path: str
    citation: str
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    byte_length: int
    bytes_sha256: str
    document: Mapping
    boundary_source: str = ""
    attestation_root: str = ""
    ratification_pin: Optional[str] = None
    _minted: Any = None

    def __post_init__(self) -> None:
        if self._minted is not _READER_TOKEN:
            raise T3InputError(
                "WaiverReadReceipt: a read receipt is MINTED by "
                "`waiver_binding_from_path`, never constructed. Building one by hand "
                "would be asserting a read that did not happen, which is the exact "
                "defect the reader exists to close (§10.4).")
        _text(self.resolved_path, "WaiverReadReceipt.resolved_path")
        _text(self.citation, "WaiverReadReceipt.citation")
        _sha256(self.bytes_sha256, "WaiverReadReceipt.bytes_sha256")
        if not isinstance(self.document, Mapping):
            raise T3InputError("WaiverReadReceipt.document: must be a mapping")

    def to_dict(self) -> dict:
        return {"resolved_path": self.resolved_path, "citation": self.citation,
                "st_dev": self.st_dev, "st_ino": self.st_ino,
                "st_size": self.st_size, "st_mtime_ns": self.st_mtime_ns,
                "byte_length": self.byte_length, "bytes_sha256": self.bytes_sha256,
                "boundary_source": self.boundary_source,
                "attestation_root": self.attestation_root,
                "ratification_pin": self.ratification_pin}


@dataclass(frozen=True)
class ReadWaiver(WaiverBinding):
    """A waiver whose document was READ from an operator-owned path by this process.

    The only type `verify_waiver` can return a PASS for. A SUBCLASS rather than a
    separate type on purpose: `_typed_tuple(..., WaiverBinding)`,
    `isinstance(binding, WaiverBinding)` and the packager all keep working unchanged,
    and the one exact test — `isinstance(x, ReadWaiver)` — is in the safe direction,
    because a read waiver arriving where a quotation is accepted is a strengthening.
    """

    read_receipt: WaiverReadReceipt = field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        violations = waiver_read_violations(self)
        if violations:
            raise T3InputError("ReadWaiver: " + " ".join(violations))

    def to_dict(self) -> dict:
        out = super().to_dict()
        out["read_receipt"] = self.read_receipt.to_dict()
        return out


def waiver_read_violations(binding: Any) -> tuple:
    """Why `binding` is NOT a document this process read. Empty ⇒ it is one.

    THE authority on "was this read", and a module FUNCTION over a duck-typed object
    rather than `isinstance(binding, ReadWaiver)` or a `was_read` property, because
    both of those are satisfied by a three-line subclass:

        @dataclass(frozen=True)
        class Sneaky(ReadWaiver):
            def __post_init__(self): pass

        Sneaky(waiver_id="FORGED", pinned_sha256="a"*64, document={...},
               document_path="/workspace/artifacts/operator/does-not-exist.json",
               covers_cell_ids=(...), observed_sha256="a"*64, read_receipt=None)

    MEASURED, not hypothesised: that object satisfied `isinstance(x, ReadWaiver)`,
    took `read=PASS` and `attribution_source="operator_owned_path"`, verified, and
    covered its failing cell — the entire §10.4 defect restored with no filesystem,
    no receipt and no token, by declining to run a constructor. A capability test a
    subclass can inherit is not a capability test. The capability is the TOKEN
    OBJECT, so only something that looks at the token can test for it, and every
    invariant `ReadWaiver.__post_init__` asserts is re-asserted HERE, at the gate,
    where skipping a constructor cannot skip it.

    Pure: no I/O, no re-read. It answers "does this object carry a receipt this
    process minted, for these exact bytes", never "is the file still there" — a
    second read would be a second set of bytes (see `_read_operator_file`).
    """
    receipt = getattr(binding, "read_receipt", None)
    if receipt is None:
        return ("was never read from disk: it carries no read receipt, so its "
                "document, its path and its digest are three independent assertions "
                "by the party being gated (use `waiver_binding_from_path()`).",)
    out: list = []
    # The token, FIRST and unconditionally. `isinstance(receipt, WaiverReadReceipt)`
    # is not asked at all: a receipt subclass that skips its own `__post_init__`
    # passes that test while never having been minted, and a non-receipt object that
    # somehow held the token would be indistinguishable from one anyway. The token is
    # the whole check; the rest is consistency.
    if getattr(receipt, "_minted", None) is not _READER_TOKEN:
        out.append(
            "carries a read receipt that `waiver_binding_from_path` did not mint. A "
            "receipt is the reader's own record of opening a file; one built by hand, "
            "or by a subclass that declines to run the constructor, asserts a read "
            "that did not happen.")
        return tuple(out)
    # OBJECT IDENTITY, not equality. An equal-but-distinct mapping is a document
    # parsed from some OTHER bytes that happens to compare equal; the whole point of
    # the receipt is that these are one object.
    if getattr(binding, "document", None) is not receipt.document:
        out.append(
            "document is not the object the receipt hashed. The receipt attests to "
            "the bytes it digested and the document it parsed from those same bytes; "
            "substituting an equal mapping re-opens the gap between 'hashed' and "
            "'parsed' the reader exists to close.")
    if getattr(binding, "observed_sha256", None) != receipt.bytes_sha256:
        out.append(
            f"observed_sha256 must be the digest the receipt records "
            f"({receipt.bytes_sha256[:12]}), not "
            f"{getattr(binding, 'observed_sha256', None)!r}.")
    if schemas.canonical_citation(
            getattr(binding, "document_path", None)) != receipt.citation:
        out.append(
            f"document_path must be the citation the receipt read "
            f"({receipt.citation!r}), not "
            f"{getattr(binding, 'document_path', None)!r}. A binding whose stated "
            "path is not the path that was opened is the "
            "three-independent-assertions defect wearing a receipt.")
    return tuple(out)


def _read_operator_file(citation: str, *, boundary: schemas.TrustBoundary,
                        attestation_roots: tuple, max_bytes: int,
                        what: str) -> tuple:
    """Read one operator-owned document. Returns `(resolved_path, raw, stat_after)`.

    Order is load-bearing — most-refusing and cheapest FIRST, so a path outside the
    boundary is never opened at all.

    DEVIATION FROM THE `os.open(O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)` DESIGN,
    stated here rather than discovered later. `t3.audit_no_write_or_process_paths`
    (and rule 1 over the whole `release/` plane) forbid this module from importing
    `os` and from calling `open` in ANY spelling — `open()`, `.open()`, or
    `getattr(x, "open")()` — because `open` is the one call that takes a mode, so
    allowing it for reading allows `open(path, "w")` by omission. That audit is a
    ratified property of the release plane and is asserted PASS by this module's own
    suite. So the fd-based discipline is not available here, and the closest honest
    equivalent is used instead:

      * `lstat()` BEFORE the read refuses a symlink final component, a non-regular
        file (FIFO, device, socket, directory), a hardlinked file, and an oversized
        one — so a FIFO is never opened and `read_bytes` cannot block in the syscall;
      * `read_bytes()` produces exactly ONE `bytes` object;
      * `lstat()` AFTER the read requires `(dev, ino, size, mtime_ns, ctime_ns)`
        unchanged, so a file swapped or rewritten under the read is refused.

    What is NOT closed by this, stated at its real strength rather than its
    comfortable one. A race that replaces the regular file strictly between the
    pre-`lstat` and the `read_bytes` has TWO outcomes, not one:

      * replaced by another REGULAR file — caught after the fact by the post-`lstat`,
        whose `(dev, ino, size, mtime_ns, ctime_ns)` comparison refuses it. Measured.
      * replaced by a FIFO — NOT caught at all. `read_bytes` blocks in `open()`
        waiting for a writer and the gate hangs indefinitely; there is no "after the
        fact" because control never returns. Measured, by injecting the swap into
        that exact window: the process wedged until the probe's alarm fired.
        `O_NOFOLLOW|O_NONBLOCK` would have refused it in the syscall.

    Closing it needs an fd, and an fd needs `open`, which `audit_no_write_or_process_
    paths` forbids across the whole `release/` plane because `open` is the one call
    that takes a mode. The trade is deliberate and the residual is an availability
    failure, not an authenticity one: an attacker who can create a FIFO inside
    `/workspace/artifacts/operator/` can already write the waiver itself, so what
    they gain by racing is a hang rather than a forged PASS. It is recorded here
    because a hang in a release gate is a real outcome that a reader of this module
    must be able to predict.
    """
    located = schemas.operator_owned_path_check(citation, boundary=boundary)
    if located.outcome != schemas.PASS:
        raise WaiverNotReadable(
            f"{what}: {citation!r} is not established as an operator-owned citation "
            f"({located.outcome}): {'; '.join(located.reasons) or 'no reason given'}. "
            "§10.4 stores a waiver under the trust-boundary path set; the reader "
            "refuses to open anything else.")

    target = Path(citation)
    try:
        before = target.lstat()
    except (OSError, ValueError) as exc:
        # ValueError as well as OSError: an embedded NUL byte makes `lstat` raise
        # `ValueError: embedded null character in path`, which escaped this reader as
        # an uncaught exception of a type no caller is told to expect. `WaiverNotReadable`
        # documents itself as covering every refusal, and a driver catching `T3Error`
        # to RECORD a refusal instead crashed on `artifacts/operator/w\x00.json`.
        raise WaiverNotReadable(
            f"{what}: {citation!r} could not be stat'd: {exc}. An absent or "
            "unreadable attestation is a REFUSAL, never an unread quotation.") from exc

    if stat_module.S_ISLNK(before.st_mode):
        raise WaiverNotReadable(
            f"{what}: {citation!r} is a symbolic link. The citation is what §10.4 "
            "speaks about, and a link at the operator's path is a document the "
            "operator did not write sitting where one they did would be.")
    if not stat_module.S_ISREG(before.st_mode):
        raise WaiverNotReadable(
            f"{what}: {citation!r} is not a regular file (mode "
            f"{stat_module.filemode(before.st_mode)}). A FIFO, device, socket or "
            "directory is not an attestation, and a FIFO would block the read.")
    if before.st_nlink != 1:
        raise WaiverNotReadable(
            f"{what}: {citation!r} has {before.st_nlink} hard links. A second name "
            "for these bytes is a second door into them, and the loop may own it.")
    if before.st_size > max_bytes:
        raise WaiverNotReadable(
            f"{what}: {citation!r} is {before.st_size} bytes, over the {max_bytes}-byte "
            "ceiling for an operator attestation (the preserved v8 waiver is 1,267).")

    # The RESOLVED location, checked as well as the citation — never instead of it.
    # The citation check is what §10.4 speaks about and is written in repo-relative
    # vocabulary; resolving first would launder a symlinked PARENT directory whose
    # target happens to be operator-owned into a citation the operator never wrote.
    # Checking only the citation is today's behaviour and cannot see that parent at all.
    try:
        resolved = schemas.canonical_citation(str(target.resolve()))
    except (OSError, ValueError) as exc:  # symlink loop, ELOOP, NUL byte
        raise WaiverNotReadable(
            f"{what}: {citation!r} could not be resolved: {exc}") from exc
    # Most SPECIFIC refusal first, so the reason names the actual hazard rather than
    # the general one that also covers it.
    if storage.is_scratch_path(resolved):
        raise WaiverNotReadable(
            f"{what}: {citation!r} resolves into the loop's own scratch root "
            f"({resolved!r}). A path the loop can write is a path the loop can author.")
    if schemas.under_any_root(resolved, storage.production_tree_forms()):
        raise WaiverNotReadable(
            f"{what}: {citation!r} resolves inside a FROZEN production kernel tree "
            f"({resolved!r}). Those trees are records, never a waiver source.")
    resolved_check = schemas.operator_owned_path_check(resolved, boundary=boundary)
    if resolved_check.outcome != schemas.PASS:
        raise WaiverNotReadable(
            f"{what}: {citation!r} resolves to {resolved!r}, which is not "
            f"operator-owned ({resolved_check.outcome}). A parent-directory symlink "
            "is invisible to the citation check by construction.")
    if not schemas.under_any_root(resolved, attestation_roots):
        raise WaiverNotReadable(
            f"{what}: {citation!r} resolves to {resolved!r}, which is under none of "
            f"the declared attestation roots {list(attestation_roots)}. "
            "`operator_owned_path_check` answers a question about the SPELLING of a "
            "citation (`artifacts/operator/…` inside a known repository checkout); "
            "the reader "
            "additionally requires the bytes to be where the operator actually keeps "
            "them.")

    try:
        raw = target.read_bytes()
    except (OSError, ValueError) as exc:
        # A permission failure, an EISDIR from a directory that appeared under the
        # check, a NUL byte — all of them are the reader REFUSING these bytes, and all
        # of them escaped as raw OSError before, outside the exception type this
        # module tells its callers to expect.
        raise WaiverNotReadable(
            f"{what}: {citation!r} could not be read: {exc}") from exc

    try:
        after = target.lstat()
    except (OSError, ValueError) as exc:
        raise WaiverNotReadable(
            f"{what}: {citation!r} could not be re-stat'd after the read: {exc}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size,
                       before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after:
        raise WaiverNotReadable(
            f"{what}: {citation!r} changed underneath the read "
            f"({identity_before} -> {identity_after}). The bytes that were hashed are "
            "not the bytes that are there.")
    if len(raw) != before.st_size:
        raise WaiverNotReadable(
            f"{what}: {citation!r} yielded {len(raw)} bytes for a stat'd size of "
            f"{before.st_size}.")
    return (resolved, raw, before)


#: Byte-order marks `json.loads(bytes)` silently honours. `json.detect_encoding`
#: sniffs these and decodes UTF-16/UTF-32 for you, so a UTF-16-LE waiver parsed and
#: VERIFIED end to end here while `json.loads(raw.decode("utf-8"))` — what the v8
#: freeze script, `jq`, and every other consumer of an operator attestation on this
#: host do — raised `UnicodeDecodeError` on byte 0. One authority document with two
#: readings, one of which is "this file is unreadable", is the parser differential
#: §10.4 can least afford.
_BYTE_ORDER_MARKS = (
    (b"\xef\xbb\xbf", "UTF-8"), (b"\x00\x00\xfe\xff", "UTF-32-BE"),
    (b"\xff\xfe\x00\x00", "UTF-32-LE"), (b"\xfe\xff", "UTF-16-BE"),
    (b"\xff\xfe", "UTF-16-LE"),
)


def _no_duplicate_keys(pairs: Any) -> dict:
    """`object_pairs_hook` that REFUSES a JSON object with a repeated key.

    `json` keeps the LAST value for a duplicate key and says nothing. On an operator
    attestation that is a document that reads one way to the human who ratified it
    and another to the gate: a waiver whose bytes contain `"protocol_changed": true`
    followed by `"protocol_changed": false` parsed as False, took `protocol_stable:
    PASS`, verified, and suppressed its cell — while the operator scrolling the file
    they signed sees `true` at the top. The digest is honest about the bytes and
    silent about which of the two readings the gate took.

    Raises `ValueError`, which is what `json.loads` already raises for malformed
    input, so the caller's existing refusal path carries it. Applied to every nested
    object, not just the top level.
    """
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(
                f"duplicate key {key!r} in a JSON object: the file states {key!r} more "
                "than once, so the bytes a human ratified and the value this gate "
                "reads are two different facts")
        seen[key] = value
    return seen


def _json_object_from_bytes(raw: bytes, *, what: str, citation: str) -> Mapping:
    """The ONE way this module turns operator bytes into a document.

    Strict UTF-8, no byte-order mark, no duplicate keys, and a JSON object at the top
    level. Every one of those was permissive before and each admitted a document that
    verified while meaning something other than what it appears to mean.
    """
    for mark, name in _BYTE_ORDER_MARKS:
        if raw.startswith(mark):
            raise WaiverNotReadable(
                f"{what}: {citation!r} begins with a {name} byte-order mark. An "
                "operator attestation on this host is UTF-8 without a BOM; `json` "
                "would sniff this and decode it while a strict-UTF-8 consumer of the "
                "same bytes cannot read the file at all.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WaiverNotReadable(
            f"{what}: {citation!r} is not UTF-8: {exc}") from exc
    try:
        document = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except ValueError as exc:
        raise WaiverNotReadable(f"{what}: {citation!r} is not JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise WaiverNotReadable(
            f"{what}: {citation!r} decodes to {type(document).__name__}, not a JSON "
            "object.")
    return document


def waiver_binding_from_path(
        document_path: Any, *, pinned_sha256: str, waiver_id: str,
        covers_cell_ids: Iterable[str],
        boundary: Optional[schemas.TrustBoundary] = None,
        attestation_roots: Optional[Sequence[str]] = None,
        ratification_pin: Optional[tuple] = None,
        max_bytes: int = schemas.MAX_OPERATOR_WAIVER_BYTES) -> ReadWaiver:
    """READ a §10.4 waiver from an operator-owned path. T3's only trusted constructor.

    THE DEFECT THIS CLOSES. `WaiverBinding` carries `document`, `document_path` and
    `observed_sha256` as three independent caller assertions and nothing reads the
    file, so a document the caller invented, pinned to its own digest, at a path that
    does not exist, verified — and took its AUTHORSHIP from
    `attribution_source="operator_owned_path"`, borrowing the standing of a directory
    it was not in. §10.4 turns a FAIL into PASS_WITH_WAIVER, so that is the authority
    path of the whole freeze gate resting on the honesty of the party being gated.

    WHAT IS RETURNED. A `ReadWaiver` carrying a `WaiverReadReceipt`, which is minted
    here and nowhere else. There is exactly ONE `bytes` object in this function:
    `bytes_sha256` digests it, `document` is `json.loads` of it, and
    `ReadWaiver.__post_init__` asserts the two are the same object by identity.

    WHAT RAISES. Everything: a citation outside the trust boundary, an absent file, a
    symlink, a non-regular file, a hardlink, an oversized file, a resolved location
    outside the declared attestation roots or inside scratch or a production tree, a
    file that changed under the read, a digest that is not the pin, or bytes that are
    not a JSON object. A refusal is NEVER downgraded to an unread `WaiverBinding`.

    `pinned_sha256` is over RAW FILE BYTES (`schemas.raw_bytes_digest`), not
    `schemas.content_hash`: the v8 ratification pins
    `sha256(waive_q8_cpu_prefill_v8_20260725.json)`, and `content_hash` of the same
    parsed document matches nothing anybody ratified.

    `ratification_pin=(ratification_path, key)` is the only AUTHENTICITY fact
    available anywhere in this system: when a preserved attestation hashes the waiver
    — v8's `evidence_sha256.waive_q8` — the digest read here must equal the digest
    that record pins. Optional because a brand-new waiver has nothing ratified
    pinning it yet.
    """
    _text(waiver_id, "waiver_binding_from_path: waiver_id")
    _sha256(pinned_sha256, "waiver_binding_from_path: pinned_sha256")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise T3InputError("waiver_binding_from_path: max_bytes must be a positive int")
    if max_bytes > schemas.MAX_OPERATOR_WAIVER_BYTES:
        # `max_bytes` may only ever NARROW. A parameter that raises a ceiling is not a
        # ceiling, it is a default: `max_bytes=10**9` read a 1 MiB file that the
        # constant exists to refuse. Every knob on this reader is a guard, and a guard
        # the party being gated can turn off is decoration.
        raise T3InputError(
            f"waiver_binding_from_path: max_bytes={max_bytes} exceeds "
            f"schemas.MAX_OPERATOR_WAIVER_BYTES ({schemas.MAX_OPERATOR_WAIVER_BYTES}). "
            "The parameter may narrow the ceiling for a caller that wants a tighter "
            "one; it may not raise it.")
    citation = schemas.canonical_citation(document_path)
    if not citation:
        raise T3InputError(
            "waiver_binding_from_path: document_path must be a non-empty string")
    if boundary is None:
        boundary = human_only_boundary()
    elif not isinstance(boundary, schemas.TrustBoundary):
        raise T3InputError(
            "waiver_binding_from_path: boundary must be a schemas.TrustBoundary")
    roots = tuple(DEFAULT_ATTESTATION_ROOTS if attestation_roots is None
                  else _str_tuple(attestation_roots,
                                  "waiver_binding_from_path: attestation_roots"))
    if not roots:
        raise T3InputError(
            "waiver_binding_from_path: attestation_roots must name at least one root; "
            "an empty root set would read as 'anywhere'.")
    # A declared root must ITSELF be operator-owned. The root set is documented as a
    # NARROWING of `operator_owned_path_check` — the reader additionally requires the
    # bytes to sit where the operator actually keeps them — but it was a free
    # substitution: `attestation_roots=("/mnt/raid0/llm/epyc-inference-research",)`
    # admitted the whole research checkout, so the composite guard was weaker than the
    # citation check it claims to narrow, and the narrowing was undone by one keyword
    # argument supplied by the party being gated. Requiring each root to pass the
    # citation check makes the root set a SUBSET by construction: the reader's answer
    # can be tightened by a caller and never widened, which is the property
    # `operator_owned_path_check` already has and the reason this parameter is safe to
    # expose at all.
    for root in roots:
        root_check = schemas.operator_owned_path_check(root, boundary=boundary)
        if root_check.outcome != schemas.PASS:
            raise T3InputError(
                f"waiver_binding_from_path: attestation root {root!r} is not itself "
                f"operator-owned ({root_check.outcome}: "
                f"{'; '.join(root_check.reasons) or 'no reason given'}). The root set "
                "narrows where an attestation may live; a root outside the trust "
                "boundary would widen it instead.")

    resolved, raw, before = _read_operator_file(
        citation, boundary=boundary, attestation_roots=roots, max_bytes=max_bytes,
        what=f"waiver {waiver_id}")

    observed = schemas.raw_bytes_digest(raw)
    if observed != pinned_sha256:
        raise WaiverNotReadable(
            f"waiver {waiver_id}: {citation!r} hashes to {observed[:12]} but the "
            f"caller pinned {pinned_sha256[:12]}. The waiver that was authorised is "
            "not the waiver that is here.")

    pinned_by_ratification = None
    if ratification_pin is not None:
        pinned_by_ratification = _ratification_pinned_digest(
            ratification_pin, boundary=boundary, attestation_roots=roots,
            max_bytes=max_bytes, what=f"waiver {waiver_id}")
        if pinned_by_ratification != observed:
            raise WaiverNotReadable(
                f"waiver {waiver_id}: the preserved ratification pins "
                f"{str(pinned_by_ratification)[:12]} for this waiver, and "
                f"{citation!r} hashes to {observed[:12]}. A waiver that its own "
                "ratification does not hash to is not the ratified waiver.")

    document = _json_object_from_bytes(raw, what=f"waiver {waiver_id}",
                                       citation=citation)

    return ReadWaiver(
        waiver_id=waiver_id, pinned_sha256=pinned_sha256, document=document,
        document_path=citation, covers_cell_ids=tuple(covers_cell_ids),
        observed_sha256=observed,
        read_receipt=WaiverReadReceipt(
            resolved_path=resolved, citation=citation, st_dev=before.st_dev,
            st_ino=before.st_ino, st_size=before.st_size,
            st_mtime_ns=before.st_mtime_ns, byte_length=len(raw),
            bytes_sha256=observed, document=document,
            boundary_source=boundary.source,
            attestation_root=next(
                (r for r in roots if schemas.under_any_root(resolved, (r,))), ""),
            ratification_pin=pinned_by_ratification,
            _minted=_READER_TOKEN))


def _ratification_pinned_digest(ratification_pin: Any, *, boundary, attestation_roots,
                                max_bytes: int, what: str) -> str:
    """The digest a preserved ratification pins for a waiver, read from ITS file.

    Read with the same discipline as the waiver, from the same declared roots: an
    authenticity cross-check sourced from a document the caller supplied would be
    the same defect one level out.
    """
    if not (isinstance(ratification_pin, (tuple, list)) and len(ratification_pin) == 2):
        raise T3InputError(
            "waiver_binding_from_path: ratification_pin must be "
            "(ratification_path, evidence_key)")
    ratification_path, key = ratification_pin
    ratification_citation = schemas.canonical_citation(ratification_path)
    if not ratification_citation or not isinstance(key, str) or not key.strip():
        raise T3InputError(
            "waiver_binding_from_path: ratification_pin needs a path and a key")
    _, raw, _ = _read_operator_file(
        ratification_citation, boundary=boundary, attestation_roots=attestation_roots,
        max_bytes=max_bytes, what=f"{what} ratification")
    ratification = _json_object_from_bytes(
        raw, what=what, citation=ratification_citation)
    evidence = ratification.get("evidence_sha256") if isinstance(
        ratification, Mapping) else None
    pinned = evidence.get(key) if isinstance(evidence, Mapping) else None
    if not isinstance(pinned, str) or not pinned.strip():
        raise WaiverNotReadable(
            f"{what}: {ratification_citation!r} pins no digest at "
            f"evidence_sha256.{key}. A cross-check against a key that is not there "
            "is not a weaker check, it is no check.")
    return pinned


@dataclass(frozen=True)
class WaiverVerification:
    """What the evaluator may say about a waiver: whether it verifies, never whether
    it is wise.

    §10.4: *"The evaluator verifies the waiver's hash and predicate; it never
    judges its merits."* So there is no field here for the reason's adequacy, the
    size of the excluded set, or whether the trade was worth it.
    """

    waiver_id: str
    check: schemas.Check
    covered_cell_ids: tuple
    forfeited_claims: tuple
    predicate_results: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.check, schemas.Check):
            raise T3InputError("WaiverVerification.check: must be a schemas.Check")
        object.__setattr__(self, "covered_cell_ids", tuple(self.covered_cell_ids))
        object.__setattr__(self, "forfeited_claims", tuple(self.forfeited_claims))

    @property
    def verified(self) -> bool:
        return self.check.outcome == schemas.PASS

    def to_dict(self) -> dict:
        return {"waiver_id": self.waiver_id, "check": _check_dict(self.check),
                "covered_cell_ids": list(self.covered_cell_ids),
                "forfeited_claims": list(self.forfeited_claims),
                "predicate_results": dict(self.predicate_results)}


def _waiver_structural_violations(document: Mapping) -> list:
    """Structural validation only — never a judgement about the waiver's content."""
    schema = document.get("schema")
    if schema == WAIVER_SCHEMA_AUTOKERNEL:
        return list(schemas.validate_operator_waiver(document))
    if schema == WAIVER_SCHEMA_V8_CPU_PREFILL:
        # The preserved v8 attestation predates `epyc.autokernel.operator_waiver.v1`.
        # It is checked against the fields the v8 freeze script itself gated on
        # (`freeze_v8_production_20260725.sh:248`), not rewritten into the newer
        # schema: editing a ratified operator attestation to make it validate would
        # forge the record this check exists to verify.
        out: list = []
        for key in ("decision", "protocol", "candidate_head", "production_head",
                    "reason", "ratified_at"):
            if not isinstance(document.get(key), str) or not document.get(key):
                out.append(f"{key}: required, a non-empty string")
        if not isinstance(document.get("protocol_changed"), bool):
            out.append("protocol_changed: required, a bool")
        scope = document.get("scope")
        if not isinstance(scope, Mapping):
            out.append("scope: required, a mapping")
        consequences = document.get("consequences")
        if not isinstance(consequences, list) or not consequences:
            out.append("consequences: required, a non-empty list of forfeited claims")
        return out
    return [f"schema: {schema!r} is not a waiver schema this gate reads "
            f"({sorted(KNOWN_WAIVER_SCHEMAS)})"]


#: Scope keys an operator attestation may use to say WHAT a waiver covers, most
#: precise first. `covers_cell_ids`/`excluded_cell_ids` name matrix cells outright
#: and are matched exactly; `excluded_pairs`/`excluded_model(s)` name the
#: model/shape vocabulary the operator actually writes in (the preserved v8 record
#: names `qwen36_q8-pp2048-iqk1`, not a cell id) and are matched as tokens.
_WAIVER_EXACT_SCOPE_KEYS = ("covers_cell_ids", "excluded_cell_ids")
#: SPECIFIC scope: the operator named a model/shape arm, which resolves to a cell.
_WAIVER_PAIR_SCOPE_KEYS = ("excluded_pairs",)
#: BROAD scope: the operator named a model. Every cell of that model in every phase,
#: on every backend, in every source tree, carries the model's name.
_WAIVER_MODEL_SCOPE_KEYS = ("excluded_models",)
_WAIVER_MODEL_SCOPE_SCALARS = ("excluded_model",)


def _cell_id_components(cell_id: str) -> tuple:
    """A cell id's dot-separated components. The matrix vocabulary is dotted
    (`llama_cpu.prefill`, `llama_cpu.pair.qwen36_q8-pp2048-iqk1`,
    `llama_gpu.backend_correctness`), so a component is the smallest unit an
    operator's token can name without naming part of a word."""
    return tuple(part for part in str(cell_id).split(".") if part)


def _scope_token_matches(token: str, cell_id: str) -> bool:
    """Does an operator scope token NAME this cell?

    Whole cell id, or one whole dotted component of it — never a raw substring.
    `token in cell_id` matched `qwen36_q8` inside `x_qwen36_q8_y`, so a token the
    operator wrote to name a model authorised any cell whose id merely CONTAINED
    those characters. §10.4 scopes a waiver to the cells the operator named;
    resolving a name is mechanical, matching a fragment of one is inventing coverage.
    """
    return token == cell_id or token in _cell_id_components(cell_id)


def _waiver_scope(document: Mapping) -> tuple:
    """`(exact_cell_ids, scope_tokens)` as the OPERATOR wrote them.

    Returns empty tuples when the document declares no resolvable scope. Nothing is
    inferred and nothing is normalised: this reads a ratified attestation.

    MOST SPECIFIC DECLARATION WINS. A document that names cells or arms has said
    which ones; the model name beside them is CONTEXT for that list, not a second,
    wider grant. Flattening the two together made the model name the operative
    scope, and the genuine v8 WAIVE-Q8 attestation — a CPU prefill eligibility-floor
    exclusion of two arms, forfeiting only *"No v8 Q8 non-regression claim"* — then
    covered `llama_gpu.qwen36_q8.backend_correctness` and
    `llama_gpu.qwen36_q8.quality` as well, because both carry `qwen36_q8`. A
    correctness FAIL suppressed by a prefill waiver is a claim nobody forfeited.

    Demotion is fail-closed and it does not touch the ratified record: v8 names its
    two arms in `excluded_pairs`, both of its real cells resolve through those, and a
    document that names ONLY a model keeps the model as its scope.
    """
    scope = document.get("scope")
    if not isinstance(scope, Mapping):
        return ((), ())
    exact: list = []
    pairs: list = []
    models: list = []
    for key in _WAIVER_EXACT_SCOPE_KEYS:
        value = scope.get(key)
        if isinstance(value, (list, tuple)):
            exact.extend(v for v in value if isinstance(v, str) and v.strip())
    for key in _WAIVER_PAIR_SCOPE_KEYS:
        value = scope.get(key)
        if isinstance(value, (list, tuple)):
            pairs.extend(v for v in value if isinstance(v, str) and v.strip())
    for key in _WAIVER_MODEL_SCOPE_KEYS:
        value = scope.get(key)
        if isinstance(value, (list, tuple)):
            models.extend(v for v in value if isinstance(v, str) and v.strip())
    for key in _WAIVER_MODEL_SCOPE_SCALARS:
        value = scope.get(key)
        if isinstance(value, str) and value.strip():
            models.append(value)
    tokens = pairs if (pairs or exact) else models
    return (tuple(dict.fromkeys(exact)), tuple(dict.fromkeys(tokens)))


def verify_waiver(binding: WaiverBinding, *, candidate_commit: str,
                  production_base_commit: str, campaign_id: str,
                  known_cell_ids: Iterable[str], failing_cell_ids: Iterable[str],
                  now: str,
                  boundary: Optional[schemas.TrustBoundary] = None,
                  attestation_roots: Optional[Sequence[str]] = None
                  ) -> WaiverVerification:
    """Verify a waiver's HASH and PREDICATE. Never its merits (§10.4).

    The predicate is exactly what `freeze_v8_production_20260725.sh:248` gated on,
    generalised: the waiver names this candidate head and this production head, its
    protocol did not move underneath it, it has not expired, it names the claims it
    forfeits, and every cell it covers exists in this run's matrix.

    Two of the predicates are about AUTHORSHIP, and both were holes until
    2026-08-03:

      * The attestation may not name a machine actor. Accepting any non-empty
        `authorized_by` meant a document attributed to `autokernel` verified as
        human-attested, suppressed a failing gating cell, and produced a
        PASS_WITH_WAIVER verdict; the refusal existed only in the packager, one
        layer up, so every caller reaching T3 directly bypassed it.
      * The document must LIVE somewhere an operator owns. A hash proves the bytes
        did not change after somebody quoted them; it says nothing about who could
        have written them in the first place.

    `boundary` is the parsed trust-boundary manifest. It defaults to reading the
    live one; pass an explicit `schemas.TrustBoundary` to verify against a stated
    boundary instead. An unreadable boundary yields COULD_NOT_CHECK on any path
    outside the operator-attestation root, which suppresses nothing.

    `attestation_roots` is where THIS GATE holds that operator attestations live,
    and it defaults to the real one. It is not the same argument as the reader's:
    `waiver_binding_from_path` takes its roots from its caller, so a caller could
    read a document out of a directory it had just written and get back the trusted
    type. The narrowing has to be re-stated by the party that acts on the waiver, or
    it is not a narrowing at all — it is a preference the party being gated sets.
    """
    if not isinstance(binding, WaiverBinding):
        raise T3InputError("verify_waiver: binding must be a WaiverBinding")
    if boundary is None:
        boundary = human_only_boundary()
    elif not isinstance(boundary, schemas.TrustBoundary):
        raise T3InputError("verify_waiver: boundary must be a schemas.TrustBoundary")
    roots = tuple(DEFAULT_ATTESTATION_ROOTS if attestation_roots is None
                  else _str_tuple(attestation_roots,
                                  "verify_waiver: attestation_roots"))
    if not roots:
        raise T3InputError(
            "verify_waiver: attestation_roots must name at least one root; an empty "
            "root set would read as 'anywhere', which is the state this argument "
            "exists to refuse.")
    known = set(known_cell_ids)
    failing = set(failing_cell_ids)
    document = binding.document
    reasons: list = []
    unknown: list = []
    results: dict = {}

    # --- read: was this document read from disk, or quoted? ------------------
    # FIRST, because every predicate below it is a statement about `document`, and
    # on a quotation `document` is whatever the party being gated handed over. The
    # hash predicate in particular compares two numbers the SAME caller supplied.
    #
    # `waiver_read_violations` and NOT `isinstance(binding, ReadWaiver)`. The type was
    # the test until a red-team pass built `class Sneaky(ReadWaiver)` with a `__post_
    # init__` that does nothing, handed it `read_receipt=None` and a path that does not
    # exist, and watched it take `read=PASS`, `attribution_source="operator_owned_path"`
    # and coverage of a failing gating cell. Inheritance is a thing the caller writes,
    # so a capability that inheritance confers is a flag the caller sets. The capability
    # is the mint token on the receipt, and the invariants tying the binding to that
    # receipt are re-asserted here rather than trusted from a constructor that a
    # subclass may decline to run.
    read_violations = waiver_read_violations(binding)
    if not read_violations and not schemas.under_any_root(
            binding.read_receipt.resolved_path, roots):
        # WHICH operator root, and judged HERE rather than only at the read. The
        # reader narrows to `DEFAULT_ATTESTATION_ROOTS` because
        # `operator_owned_path_check` answers a question about the SPELLING of a
        # citation — `artifacts/operator/…` inside a known repository checkout, including
        # `/mnt/raid0/llm/epyc-inference-research/artifacts/operator/`, which this
        # repository's own agents create with `mkdir -p`. But that narrowing was a
        # KEYWORD ARGUMENT of the reader, so the loop could hand itself the root it
        # had just written to and get back the trusted type: MEASURED, a document the
        # loop authored in its own checkout, naming `autokernel` as its author, read
        # with `attestation_roots=(<that directory>,)`, verified PASS with
        # `read=PASS` and `attribution_source="operator_owned_path"` and suppressed a
        # GPU correctness FAIL. The entire defect, restored by an argument the reader
        # offers.
        #
        # A reader cannot defend a boundary the caller supplies to it. So the party
        # that decides a waiver SUPPRESSES a failing gate states, itself, where
        # operator attestations live — defaulting to the real root, never to
        # whatever the reader was told. A read from anywhere else is COULD_NOT_CHECK:
        # the bytes were genuinely read, and nothing establishes that an operator
        # wrote them.
        unknown.append(
            f"waiver {binding.waiver_id}: was read from "
            f"{binding.read_receipt.resolved_path!r}, which is under none of the "
            f"attestation roots this gate holds ({list(roots)}). The reader was told "
            "where to look by its caller; the gate is not. An `artifacts/operator/` "
            "directory inside a checkout the loop can write is operator-SHAPED, not "
            "operator-owned.")
        results["read"] = schemas.COULD_NOT_CHECK
        results["attestation_root"] = schemas.COULD_NOT_CHECK
    elif not read_violations:
        # PASS/FAIL/COULD_NOT_CHECK only. Deliberately NOT the resolved path: two
        # waivers differing solely in where their bytes sit must produce identical
        # predicate results, or the gate is grading something §10.4 does not ask
        # about. The path is on the receipt, which is where a fact about the read
        # belongs.
        results["read"] = schemas.PASS
        results["attestation_root"] = schemas.PASS
    elif isinstance(binding, ReadWaiver) \
            or getattr(binding, "read_receipt", None) is not None:
        # An AFFIRMATIVE claim of a read that did not happen is not the same state as
        # "nobody looked", and must not be recorded as one. A quotation is honest and
        # answers COULD_NOT_CHECK; a binding wearing the read TYPE, or a receipt it was
        # not given, is a forgery and answers FAIL — which no later evidence
        # rehabilitates, where COULD_NOT_CHECK invites somebody to go and look.
        reasons.extend(f"waiver {binding.waiver_id}: {v}" for v in read_violations)
        results["read"] = schemas.FAIL
    else:
        unknown.extend(f"waiver {binding.waiver_id}: {v}" for v in read_violations)
        results["read"] = schemas.COULD_NOT_CHECK

    # --- hash ---------------------------------------------------------------
    if binding.observed_sha256 is None:
        unknown.append(
            f"waiver {binding.waiver_id}: no observed sha256 was supplied, so the pinned "
            f"digest {binding.pinned_sha256[:12]} is unverified. A quoted waiver is not a "
            "read one.")
        results["hash"] = schemas.COULD_NOT_CHECK
    elif binding.observed_sha256 != binding.pinned_sha256:
        reasons.append(
            f"waiver {binding.waiver_id}: pinned {binding.pinned_sha256[:12]} but the "
            f"document on disk hashes to {binding.observed_sha256[:12]}. The waiver that "
            "was authorised is not the waiver that is here.")
        results["hash"] = schemas.FAIL
    else:
        results["hash"] = schemas.PASS

    # --- structure ----------------------------------------------------------
    violations = _waiver_structural_violations(document)
    if violations:
        reasons.extend(f"waiver {binding.waiver_id}: {v}" for v in violations)
        results["structure"] = schemas.FAIL
    else:
        results["structure"] = schemas.PASS

    # --- provenance: where the document LIVES --------------------------------
    # §10.4 stores a waiver "under the trust-boundary path set". Without this the
    # hash pins bytes and nothing pins their origin: a document the loop wrote to
    # its own scratch directory verifies exactly as well as a ratified attestation.
    # Evaluated BEFORE authorship because authorship depends on it: see below.
    located = schemas.operator_owned_path_check(binding.document_path,
                                                boundary=boundary)
    results["document_path"] = located.outcome
    if located.outcome == schemas.FAIL:
        reasons.extend(f"waiver {binding.waiver_id}: {r}" for r in located.reasons)
    elif located.outcome == schemas.COULD_NOT_CHECK:
        unknown.extend(f"waiver {binding.waiver_id}: {r}" for r in located.reasons)

    # --- human authorship ----------------------------------------------------
    # §10.4: waivers are human-only. A machine-authored record in this repository
    # carries no authority-flavoured key (schemas `_reject_authority_keys`), so the
    # ONLY thing that distinguishes a waiver is a named human attestation.
    #
    # THREE ways a document can be attributed, and the third is why the second is
    # not enough on its own. A scan of the five attribution fields is satisfiable
    # by DELETING them: the preserved v8 waiver
    # (`artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json`) carries none of
    # them — its whole attestation is a `ratified_at` timestamp — so under the
    # older `epyc.cpu_prefill_v8.operator_waiver.v1` schema, which does not require
    # one either, any document that simply named no author read as human-attested
    # and the machine-actor refusal never fired. That schema cannot be tightened
    # without invalidating the genuine ratified record, so the missing fact is
    # supplied by the one thing the v8 record HAS and a forged one does not: it
    # lives under an operator-owned path. Authorship is therefore
    #   (a named non-machine actor) OR (no attribution at all AND provenance PASS),
    # and never a machine name in either case. COULD_NOT_CHECK provenance does not
    # substitute for an author — an unknown origin establishes nothing.
    attested_by = document.get("authorized_by") or document.get("ratified_at")
    attributed_to_machine = schemas.machine_attributions(document)
    named_actors = [
        field_name for field_name in schemas.ACTOR_ATTRIBUTION_FIELDS
        if isinstance(document.get(field_name), str) and document[field_name].strip()]
    if not attested_by:
        reasons.append(
            f"waiver {binding.waiver_id}: carries neither `authorized_by` nor "
            "`ratified_at`. A waiver is human-authored by definition; an unattributed "
            "one is a machine granting itself an exception.")
        results["human_attested"] = schemas.FAIL
    elif attributed_to_machine:
        # An attribution is not a formality that a non-empty string satisfies. §10.4
        # makes the waiver human-authored BY DEFINITION, so a machine-attributed
        # document is not a waiver with a bad author — it is the loop excusing its
        # own failing cell, which is the single thing this gate exists to refuse.
        for field_name, identity, tokens in attributed_to_machine:
            reasons.append(
                f"waiver {binding.waiver_id}: names {identity!r} in {field_name!r}, a "
                f"machine actor ({', '.join(tokens)}). §10.4 waivers are human-only "
                "(MEASUREMENT.md:140-142); a waiver the loop attributed to itself is "
                f"the loop excusing {list(binding.covers_cell_ids)}.")
        results["human_attested"] = schemas.FAIL
    elif named_actors:
        results["human_attested"] = schemas.PASS
        results["attribution_source"] = "named_actor"
    elif located.outcome == schemas.PASS:
        # The preserved v8 shape: no author field anywhere, and the record's
        # standing rests on where it lives. Recorded as a DIFFERENT source of the
        # same verdict so the bundle says which one carried it.
        results["human_attested"] = schemas.PASS
        results["attribution_source"] = "operator_owned_path"
    else:
        reasons.append(
            f"waiver {binding.waiver_id}: names no actor in any of "
            f"{list(schemas.ACTOR_ATTRIBUTION_FIELDS)} — a timestamp is not an author — "
            f"and its path {binding.document_path!r} is not established as "
            f"operator-owned ({located.outcome}). The machine-actor refusal scans "
            "those fields, so a document that simply omits them would otherwise pass "
            "it by deleting what it inspects; the only other thing that distinguishes "
            "the preserved v8 attestation from a forgery is where it lives.")
        results["human_attested"] = schemas.FAIL
        results["attribution_source"] = "none"

    # --- predicate: heads ----------------------------------------------------
    doc_candidate = document.get("candidate_head")
    doc_production = document.get("production_head")
    if doc_candidate != candidate_commit:
        reasons.append(
            f"waiver {binding.waiver_id}: names candidate head {doc_candidate!r}, this "
            f"run seals {candidate_commit!r}")
    if doc_production != production_base_commit:
        reasons.append(
            f"waiver {binding.waiver_id}: names production head {doc_production!r}, this "
            f"run's production base is {production_base_commit!r}")
    results["heads"] = schemas.FAIL if (doc_candidate != candidate_commit or
                                        doc_production != production_base_commit) \
        else schemas.PASS

    # --- predicate: campaign binding ----------------------------------------
    doc_campaign = document.get("campaign_id")
    if doc_campaign is not None and doc_campaign != campaign_id:
        reasons.append(
            f"waiver {binding.waiver_id}: is bound to campaign {doc_campaign!r}, this run "
            f"is {campaign_id!r}")
        results["campaign"] = schemas.FAIL
    else:
        results["campaign"] = schemas.PASS

    # --- predicate: the protocol did not move --------------------------------
    if document.get("protocol_changed") is True:
        reasons.append(
            f"waiver {binding.waiver_id}: `protocol_changed` is true. A waiver bound to a "
            "protocol that moved underneath it is not the waiver that was granted, and "
            "re-authorisation is an operator act, not an inference.")
        results["protocol_stable"] = schemas.FAIL
    else:
        results["protocol_stable"] = schemas.PASS

    # --- predicate: expiry / reopen -----------------------------------------
    expiry = document.get("expiry")
    if isinstance(expiry, Mapping):
        expires_at = expiry.get("expires_at")
        if isinstance(expires_at, str) and expires_at:
            if _timestamp(now, "now") > _timestamp(expires_at, "expiry.expires_at"):
                reasons.append(
                    f"waiver {binding.waiver_id}: expired at {expires_at}, this run is at "
                    f"{now}")
                results["expiry"] = schemas.FAIL
            else:
                results["expiry"] = schemas.PASS
        else:
            results["expiry"] = schemas.PASS
    else:
        results["expiry"] = schemas.PASS

    # --- predicate: coverage -------------------------------------------------
    unknown_cells = [c for c in binding.covers_cell_ids if c not in known]
    if unknown_cells:
        reasons.append(
            f"waiver {binding.waiver_id}: covers cells that are not in this release "
            f"matrix: {unknown_cells}. A waiver whose scope does not resolve is not a "
            "narrower gate, it is an unread one.")
        results["coverage"] = schemas.FAIL
    else:
        results["coverage"] = schemas.PASS

    # --- predicate: the scope is the OPERATOR'S, not the caller's ------------
    # `covers_cell_ids` arrives on the BINDING — that is, from the party being
    # gated — while the hash above only proves the document was not edited. Without
    # this step the gate pins a digest and then ignores what the digest is over: the
    # genuine, unmodified WAIVE-Q8 attestation, whose scope names two Q8 model/shape
    # pairs, could be pointed at a GPU correctness FAIL and would suppress it, claim
    # and all. §10.4 scopes a waiver to the cells the operator named; RESOLVING those
    # names against this matrix is mechanical, INVENTING coverage the attestation does
    # not carry is the machine granting itself the exception.
    exact_scope, scope_tokens = _waiver_scope(document)
    if not exact_scope and not scope_tokens:
        if binding.covers_cell_ids:
            reasons.append(
                f"waiver {binding.waiver_id}: is bound to {list(binding.covers_cell_ids)} "
                "but its document declares no resolvable scope (no excluded cell ids, "
                "pairs or models). The coverage would then be the caller's assertion "
                "rather than the operator's, which is what pinning the hash exists to "
                "prevent.")
        results["scope"] = schemas.FAIL
    else:
        unauthorised = [
            cell_id for cell_id in binding.covers_cell_ids
            if cell_id not in exact_scope
            and not any(_scope_token_matches(token, cell_id) for token in scope_tokens)
        ]
        if unauthorised:
            reasons.append(
                f"waiver {binding.waiver_id}: is bound to {unauthorised}, which its own "
                f"attested scope does not name (cell ids {list(exact_scope)}, tokens "
                f"{list(scope_tokens)}). A waiver covers the cells the OPERATOR excluded; "
                "a hash-verified document pointed at other cells is a verified document "
                "and an unverified exception.")
            results["scope"] = schemas.FAIL
        else:
            results["scope"] = schemas.PASS

    # --- forfeited claims ----------------------------------------------------
    consequences = document.get("consequences")
    forfeited = tuple(c for c in (consequences or []) if isinstance(c, str))
    if not forfeited:
        reasons.append(
            f"waiver {binding.waiver_id}: forfeits no claim. The v8 precedent names the "
            "forfeited claim explicitly (\"No v8 Q8 non-regression claim may be made from "
            "this campaign.\"); a waiver that forfeits nothing is an approval.")

    if reasons:
        check = _fail(*reasons)
    elif unknown:
        check = _cnc(*unknown)
    else:
        check = schemas.Check(schemas.PASS)
    # Coverage is stated ONLY by a verification that PASSed. Populated regardless,
    # a refused waiver landed in the durable bundle carrying a waived-LOOKING
    # coverage list — the same defect this function exists to refuse (a record
    # asserting coverage nobody verified), one layer out. Inert for `compute_verdict`,
    # which guards on `.verified`; not inert for anything that reads the bundle.
    covered = (tuple(c for c in binding.covers_cell_ids if c in failing)
               if check.outcome == schemas.PASS else ())
    return WaiverVerification(
        waiver_id=binding.waiver_id, check=check, covered_cell_ids=covered,
        forfeited_claims=forfeited, predicate_results=results,
    )


# =============================================================================
# Sealed-fingerprint idempotence and failed-gate cooldown (§9.1, §12)
# =============================================================================

@dataclass(frozen=True)
class T3Attempt:
    """One prior T3 run, as the caller's ledger records it."""

    fingerprint: str
    verdict: str
    completed_at: str
    bundle_sha256: str
    failed_phases: tuple = ()

    def __post_init__(self) -> None:
        _sha256(self.fingerprint, "T3Attempt.fingerprint")
        if self.verdict not in schemas.T3_VERDICTS:
            raise T3InputError(
                f"T3Attempt.verdict: {self.verdict!r} is not one of "
                f"{sorted(schemas.T3_VERDICTS)}")
        _timestamp(self.completed_at, "T3Attempt.completed_at")
        _sha256(self.bundle_sha256, "T3Attempt.bundle_sha256")
        phases = _str_tuple(self.failed_phases, "T3Attempt.failed_phases", non_empty=False)
        for phase in phases:
            if phase not in PHASES:
                raise T3InputError(
                    f"T3Attempt.failed_phases: {phase!r} is not one of {list(PHASES)}")
        object.__setattr__(self, "failed_phases", phases)

    def to_dict(self) -> dict:
        return {"fingerprint": self.fingerprint, "verdict": self.verdict,
                "completed_at": self.completed_at, "bundle_sha256": self.bundle_sha256,
                "failed_phases": list(self.failed_phases)}


@dataclass(frozen=True)
class StageRepair:
    """§9.1's second admission route: *"a deterministic replay/repair of the failed
    stage."*

    `deterministic_replay=False` means fresh inference was run, which changes the
    evidence and therefore the fingerprint — so it is not a repair, it is a new
    run, and this object cannot license it.
    """

    prior_fingerprint: str
    repaired_phase: str
    deterministic_replay: bool
    repair_ref: str

    def __post_init__(self) -> None:
        _sha256(self.prior_fingerprint, "StageRepair.prior_fingerprint")
        if self.repaired_phase not in PHASES:
            raise T3InputError(
                f"StageRepair.repaired_phase: {self.repaired_phase!r} is not one of "
                f"{list(PHASES)}")
        _bool(self.deterministic_replay, "StageRepair.deterministic_replay")
        _text(self.repair_ref, "StageRepair.repair_ref")

    def to_dict(self) -> dict:
        return {"prior_fingerprint": self.prior_fingerprint,
                "repaired_phase": self.repaired_phase,
                "deterministic_replay": self.deterministic_replay,
                "repair_ref": self.repair_ref}


@dataclass(frozen=True)
class RerunDisposition:
    admissible: bool
    code: str
    reason: str
    prior: Optional[T3Attempt] = None

    def __post_init__(self) -> None:
        if self.code not in RERUN_CODES:
            raise T3InputError(f"RerunDisposition.code: {self.code!r} is not a rerun code")
        _text(self.reason, "RerunDisposition.reason")

    def to_dict(self) -> dict:
        return {"admissible": self.admissible, "code": self.code, "reason": self.reason,
                "prior": None if self.prior is None else self.prior.to_dict()}


#: Everything the fingerprint covers. Enumerated so a test can assert the SET
#: rather than a digest: a digest test tells you the algorithm changed, this tells
#: you what it now covers. Active waiver hashes ARE evidence-affecting — adding a
#: waiver is precisely how a FAILed v8-shaped run becomes PASS_WITH_WAIVER, and a
#: fingerprint that ignored them could never admit that rerun.
#:
#: `active_waiver_coverage` is separate from `active_waiver_sha256` because the
#: digest is a fact about the DOCUMENT and the coverage is a fact about the RUN.
#: The same hash-verified attestation can be bound to different cells — that is
#: exactly the misdirection the scope predicate refuses — and hashing only the
#: digest gave two runs that suppress different cells one fingerprint, so §9.1's
#: idempotence would have refused the second as "already sealed".
#:
#: The `WaiverReadReceipt` is DELIBERATELY ABSENT, and the naive move here is wrong.
#: A rerun that re-reads the same bytes from the same path gets a new inode number
#: and a new mtime whenever the operator touched the file; hashing the receipt would
#: make that a different fingerprint and send an identical rerun into
#: REFUSED_UNCHANGED_FINGERPRINT. §9.1's identity is over the EVIDENCE, and the
#: evidence is the digest (already here as `active_waiver_sha256`) plus the coverage,
#: not the filesystem metadata of the read. What DOES change the fingerprint is a
#: quotation becoming a read waiver of DIFFERENT bytes — because the digest changes.
FINGERPRINT_FACETS = (
    "candidate_id", "source_tree", "candidate_branch", "production_base_commit",
    "candidate_commit", "seal_sha256", "binary_sha256", "linkage_sha256",
    "evaluator_bundle_sha256", "scope_manifest_sha256", "evidence_tree_sha256",
    "plan_sha256", "protocol_document_sha256", "supplied_components",
    "active_waiver_sha256", "active_waiver_coverage", "phase_protocol_standing",
    "protocol_registry_standing",
)


def sealed_fingerprint(*, sealed: SealedCandidate, plan: ReleasePlanView,
                       protocol: ProtocolBinding, waivers: Sequence[WaiverBinding] = (),
                       supplied_components: Optional[Mapping] = None,
                       phase_protocols: Optional[Mapping] = None,
                       protocol_registry: Sequence[ProtocolBinding] = ()) -> str:
    """The identity §9.1's *"once per sealed fingerprint"* is keyed on.

    Deliberately excludes the run id, the timestamps, the operator, and every
    narrative field: a rerun that differs only in when it was asked for is the same
    run, and a fingerprint a caller can perturb for free is not an idempotence key.
    """
    if not isinstance(sealed, SealedCandidate):
        raise T3InputError("sealed_fingerprint: sealed must be a SealedCandidate")
    if not isinstance(plan, ReleasePlanView):
        raise T3InputError("sealed_fingerprint: plan must be a ReleasePlanView")
    if not isinstance(protocol, ProtocolBinding):
        raise T3InputError("sealed_fingerprint: protocol must be a ProtocolBinding")
    supplied = dict(supplied_components or {})
    facets = {
        "candidate_id": sealed.candidate_id,
        "source_tree": sealed.source_tree,
        "candidate_branch": sealed.candidate_branch,
        "production_base_commit": sealed.production_base_commit,
        "candidate_commit": sealed.candidate_commit,
        "seal_sha256": sealed.seal_sha256,
        "binary_sha256": {k: sealed.binary_sha256[k] for k in sorted(sealed.binary_sha256)},
        "linkage_sha256": {k: sealed.linkage_sha256[k]
                           for k in sorted(sealed.linkage_sha256)},
        "evaluator_bundle_sha256": sealed.evaluator_bundle_sha256,
        "scope_manifest_sha256": sealed.scope_manifest_sha256,
        "evidence_tree_sha256": sealed.evidence_tree_sha256,
        "plan_sha256": plan.plan_sha256,
        "protocol_document_sha256": protocol.document_sha256,
        "supplied_components": {k: supplied[k] for k in sorted(supplied)},
        "active_waiver_sha256": sorted(w.pinned_sha256 for w in waivers),
        "active_waiver_coverage": sorted(
            [w.waiver_id, w.pinned_sha256, sorted(w.covers_cell_ids)]
            for w in waivers),
        # Evidence-affecting for the same reason the waiver hashes are: ratifying
        # `P-STT-1` turns a run this gate BLOCKED into one it can adjudicate,
        # without touching the candidate, the plan, or a single measurement. A
        # fingerprint blind to it would send that rerun into
        # REFUSED_UNCHANGED_FINGERPRINT — fail-closed, and still wrong, because
        # §9.1's idempotence is over the evidence graded and the standing of the
        # instrument is part of it.
        "phase_protocol_standing": sorted(
            [backend, workload_phase, bound.protocol_id,
             "unbound" if bound.ratified is None
             else ("ratified" if bound.ratified else "draft"),
             "" if bound.binding is None else bound.binding.document_sha256]
            for backend, by_phase in dict(phase_protocols or {}).items()
            for workload_phase, bound in by_phase.items()),
        # The OTHER half of the same fact, and the half that actually moves the
        # speech backends. `declared_ratified_protocol_ids()` reads THREE sources —
        # `protocol`, `phase_protocols` and `protocol_registry` — and the adapter
        # seam is satisfied through the registry, because `P-AK-SEARCH-1` and
        # `P-STT-REL-1` are not per-phase grading instruments and have no phase to
        # be filed under. Covering only the per-phase half left the exact
        # post-ratification rerun this facet exists for — declare the five whisper
        # protocols as ratified `ProtocolBinding`s and run again — landing on an
        # UNCHANGED fingerprint and `REFUSED_UNCHANGED_FINGERPRINT`. A fingerprint
        # must cover every input the gate's own verdict is a function of, and
        # `phase_identity_preflight` blocks on this one.
        "protocol_registry_standing": sorted(
            [binding.protocol_id,
             "ratified" if binding.ratified else "draft",
             binding.document_sha256]
            for binding in _typed_tuple(
                protocol_registry, "sealed_fingerprint: protocol_registry",
                ProtocolBinding)),
    }
    if set(facets) != set(FINGERPRINT_FACETS):
        raise T3Error(
            "sealed_fingerprint: the computed facet set drifted from FINGERPRINT_FACETS. "
            f"computed={sorted(facets)} declared={sorted(FINGERPRINT_FACETS)}"
        )
    return schemas.content_hash(facets)


def check_rerun(fingerprint: str, ledger: Sequence[T3Attempt], *, now: str,
                cooldown_seconds: int,
                repair: Optional[StageRepair] = None) -> RerunDisposition:
    """§9.1 / §12: idempotence on an unchanged fingerprint, cooldown after a failure.

    Three admissions and three refusals, and no fourth of either. There is no
    "force" argument: a caller that could pass one would make the guard advisory,
    and §12's named defence against *"full release evaluation loops repeatedly"* is
    that the gate refuses, not that it warns.
    """
    _sha256(fingerprint, "check_rerun: fingerprint")
    attempts = _typed_tuple(ledger, "check_rerun: ledger", T3Attempt)
    if not isinstance(cooldown_seconds, int) or isinstance(cooldown_seconds, bool) \
            or cooldown_seconds <= 0:
        raise T3InputError(
            "check_rerun: cooldown_seconds must be a positive int. There is no default "
            "— a silent cooldown is a policy nobody declared."
        )
    now_dt = _timestamp(now, "check_rerun: now")
    if repair is not None and not isinstance(repair, StageRepair):
        raise T3InputError("check_rerun: repair must be a StageRepair or None")

    same = [a for a in attempts if a.fingerprint == fingerprint]
    if not same:
        code = RERUN_ADMITTED_FIRST_ATTEMPT if not attempts else \
            RERUN_ADMITTED_NEW_FINGERPRINT
        return RerunDisposition(
            admissible=True, code=code,
            reason=("no prior attempt carries this sealed fingerprint, so the evidence "
                    "that would be graded is not the evidence that was already graded"),
        )

    prior = max(same, key=lambda a: _timestamp(a.completed_at, "completed_at"))
    if prior.verdict in ("PASS", "PASS_WITH_WAIVER"):
        return RerunDisposition(
            admissible=False, code=RERUN_REFUSED_ALREADY_SEALED, prior=prior,
            reason=(
                f"this sealed fingerprint already produced a {prior.verdict} bundle "
                f"{prior.bundle_sha256[:12]} at {prior.completed_at}. §9.1: T3 runs once "
                "per sealed fingerprint; re-running would spend the full release matrix "
                "to recompute a decision that is already sealed. Reuse the bundle."
            ),
        )

    if repair is None:
        return RerunDisposition(
            admissible=False, code=RERUN_REFUSED_UNCHANGED_FINGERPRINT, prior=prior,
            reason=(
                f"this sealed fingerprint already FAILed at {prior.completed_at} "
                f"(phases: {list(prior.failed_phases) or 'unrecorded'}). §9.1: a retry "
                "requires a new evidence-affecting fingerprint or a deterministic "
                "replay/repair of the failed stage; neither is present."
            ),
        )
    if repair.prior_fingerprint != fingerprint:
        return RerunDisposition(
            admissible=False, code=RERUN_REFUSED_UNCHANGED_FINGERPRINT, prior=prior,
            reason=(
                f"the supplied repair is against fingerprint "
                f"{repair.prior_fingerprint[:12]}, not {fingerprint[:12]}; a repair of "
                "another run does not admit this one"
            ),
        )
    if not repair.deterministic_replay:
        return RerunDisposition(
            admissible=False, code=RERUN_REFUSED_UNCHANGED_FINGERPRINT, prior=prior,
            reason=(
                "the supplied repair is not a deterministic replay. Fresh inference "
                "changes the evidence, which changes the fingerprint; a repair that "
                "re-measures is a new run and must present itself as one."
            ),
        )
    if prior.failed_phases and repair.repaired_phase not in prior.failed_phases:
        return RerunDisposition(
            admissible=False, code=RERUN_REFUSED_UNCHANGED_FINGERPRINT, prior=prior,
            reason=(
                f"the repair names phase {repair.repaired_phase!r}, which is not among "
                f"the phases that failed {list(prior.failed_phases)}. Repairing a stage "
                "that did not fail leaves the one that did untouched."
            ),
        )
    elapsed = (now_dt - _timestamp(prior.completed_at, "completed_at")).total_seconds()
    if elapsed < cooldown_seconds:
        return RerunDisposition(
            admissible=False, code=RERUN_REFUSED_COOLDOWN, prior=prior,
            reason=(
                f"{int(elapsed)}s have passed since the failed gate at "
                f"{prior.completed_at}; the declared cooldown is {cooldown_seconds}s. "
                "The cooldown is not a second opinion on the repair — it is what stops a "
                "failed gate from being re-entered in a loop (§12)."
            ),
        )
    return RerunDisposition(
        admissible=True, code=RERUN_ADMITTED_AFTER_REPAIR, prior=prior,
        reason=(
            f"phase {repair.repaired_phase} was repaired by deterministic replay "
            f"({repair.repair_ref}) and the {cooldown_seconds}s cooldown has elapsed"
        ),
    )


# =============================================================================
# The request
# =============================================================================

@dataclass(frozen=True)
class T3Request:
    """Everything one T3 run adjudicates. Every measurement in it is an INPUT.

    The gate does not time, build, measure, or read a file. That separation is
    invariant 4 in request form: the actor produces, the trusted evaluator
    measures, and the release gate adjudicates — and a gate that also measured
    could grade its own arithmetic.
    """

    run_id: str
    campaign_id: str
    mode: str
    now: str
    protocol: ProtocolBinding
    sealed: SealedCandidate
    plan: ReleasePlanView
    backend_unchanged: Mapping
    host: guards.HostHealth
    host_owner: str
    host_escalation_deadline: str
    resource_claims: tuple
    storage_observation: guards.StorageObservation
    transaction: TransactionPlan
    archive: IncumbentArchive
    supplied_components: Mapping
    cooldown_seconds: int
    #: protocol id -> the release rep count that protocol requires. No default:
    #: §10.2 phase 4 says "release reps", and a made-up rep count is a made-up gate.
    release_reps_by_protocol: Mapping = field(default_factory=dict)
    #: backend -> workload phase -> the protocol that owns it (§1.6), as a
    #: `PhaseProtocolBinding`. A bare id or a `ProtocolBinding` is normalised into
    #: one on construction; a bare id normalises to the UNBOUND state, which reads
    #: COULD_NOT_CHECK and blocks — an id is a name, not a ratification.
    phase_protocols: Mapping = field(default_factory=dict)
    #: `ProtocolBinding`s for protocols this run depends on that are not per-phase
    #: and are not the freeze protocol — a speech family's `P-STT-REL-1`, or
    #: `P-AK-SEARCH-1` itself. Together with the per-phase bindings this is the ONLY
    #: source `declared_ratified_protocol_ids()` reads, so an adapter can never be
    #: told a family is ratified without a hashed binding saying so.
    protocol_registry: tuple = ()
    transfer_receipts: Mapping = field(default_factory=dict)
    linkage_receipts: tuple = ()
    backend_inventories: tuple = ()
    determinism: tuple = ()
    cell_results: tuple = ()
    standings: tuple = ()
    phase_trades: tuple = ()
    capacity_floors: tuple = ()
    quality_evidence: tuple = ()
    stability_evidence: tuple = ()
    #: §10.2 phase 6 gates on REPEATED load/unload. No default: this used to default
    #: to 1, so a soak of a single cycle cleared "the declared release minimum" that
    #: nobody had declared — the same defect `check_rerun` refuses by name for the
    #: cooldown ("there is no default — a silent cooldown is a policy nobody
    #: declared"). One load/unload is not repetition.
    stability_min_cycles: Optional[int] = None
    waivers: tuple = ()
    #: Where THIS RUN holds that operator attestations live. Empty means
    #: `DEFAULT_ATTESTATION_ROOTS`, which is the only correct answer in production;
    #: it is a request field so that a run reading attestations from anywhere else
    #: has to SAY SO, in the request, where the fingerprint hashes it and the
    #: package records it. `waiver_binding_from_path` takes its roots from its
    #: caller, so without this the gate inherited the party-being-gated's opinion of
    #: where an operator writes — see `verify_waiver`.
    attestation_roots: tuple = ()
    complexity: Mapping = field(default_factory=dict)
    attempt_ledger: tuple = ()
    stage_repair: Optional[StageRepair] = None
    campaign_start_at: Optional[str] = None
    #: The machine footprint the release thresholds were calibrated on, when one is
    #: declared. Checked against each cell's own `scope_denominator` (§7.4, §12).
    gate_scope: Optional[Mapping] = None

    def __post_init__(self) -> None:
        _text(self.run_id, "T3Request.run_id")
        _text(self.campaign_id, "T3Request.campaign_id")
        if not self.campaign_id.startswith("ak-"):
            raise T3InputError("T3Request.campaign_id: must start with 'ak-'")
        if self.mode not in MODES:
            raise T3InputError(f"T3Request.mode: {self.mode!r} is not one of {list(MODES)}")
        _timestamp(self.now, "T3Request.now")
        for name, klass in (("protocol", ProtocolBinding), ("sealed", SealedCandidate),
                            ("transaction", TransactionPlan),
                            ("archive", IncumbentArchive),
                            ("host", guards.HostHealth),
                            ("storage_observation", guards.StorageObservation)):
            if not isinstance(getattr(self, name), klass):
                raise T3InputError(f"T3Request.{name}: must be a {klass.__name__}")
        object.__setattr__(self, "plan", release_plan_view(self.plan))
        _text(self.host_owner, "T3Request.host_owner")
        _timestamp(self.host_escalation_deadline, "T3Request.host_escalation_deadline")
        object.__setattr__(self, "resource_claims", _typed_tuple(
            self.resource_claims, "T3Request.resource_claims",
            guards.ResourceClaimObservation))
        if not isinstance(self.supplied_components, Mapping):
            raise T3InputError("T3Request.supplied_components: must be a mapping")
        if not isinstance(self.cooldown_seconds, int) or \
                isinstance(self.cooldown_seconds, bool) or self.cooldown_seconds <= 0:
            raise T3InputError("T3Request.cooldown_seconds: required, a positive int")

        normalised = {}
        if not isinstance(self.backend_unchanged, Mapping):
            raise T3InputError("T3Request.backend_unchanged: must be a mapping")
        for backend, entry in self.backend_unchanged.items():
            normalised[backend] = unchanged_view(entry)
        object.__setattr__(self, "backend_unchanged", normalised)

        if not isinstance(self.transfer_receipts, Mapping):
            raise T3InputError("T3Request.transfer_receipts: must be a mapping")
        for backend, receipt in self.transfer_receipts.items():
            if not isinstance(receipt, TransferReceipt):
                raise T3InputError(
                    f"T3Request.transfer_receipts[{backend!r}]: must be a TransferReceipt")

        for name, klass in (("linkage_receipts", LinkageReceipt),
                            ("backend_inventories", BackendInventory),
                            ("determinism", DeterminismDeclaration),
                            ("cell_results", CellResult),
                            ("standings", PhaseStanding),
                            ("phase_trades", PhaseTradeException),
                            ("capacity_floors", CapacityFloor),
                            ("quality_evidence", QualityEvidence),
                            ("stability_evidence", StabilityEvidence),
                            ("waivers", WaiverBinding),
                            ("protocol_registry", ProtocolBinding),
                            ("attempt_ledger", T3Attempt)):
            object.__setattr__(self, name, _typed_tuple(
                getattr(self, name), f"T3Request.{name}", klass))
        if self.stage_repair is not None and not isinstance(self.stage_repair, StageRepair):
            raise T3InputError("T3Request.stage_repair: must be a StageRepair or None")
        object.__setattr__(self, "attestation_roots", _str_tuple(
            self.attestation_roots, "T3Request.attestation_roots", non_empty=False))
        for name in ("release_reps_by_protocol", "phase_protocols", "complexity"):
            if not isinstance(getattr(self, name), Mapping):
                raise T3InputError(f"T3Request.{name}: must be a mapping")
        # The per-phase protocol map is NORMALISED here rather than read as-typed
        # downstream, so there is exactly one shape in the gate and no reader has to
        # remember that a value might be a string. `phase_protocol_binding` accepts
        # the legacy bare id and records it as UNBOUND; it never invents a standing.
        bound_protocols: dict = {}
        for backend, by_phase in self.phase_protocols.items():
            if not isinstance(by_phase, Mapping):
                raise T3InputError(
                    f"T3Request.phase_protocols[{backend!r}]: must be a mapping of "
                    "workload phase -> protocol binding")
            bound_protocols[backend] = {
                phase: phase_protocol_binding(value, backend=backend,
                                              workload_phase=phase)
                for phase, value in by_phase.items()}
        object.__setattr__(self, "phase_protocols", bound_protocols)
        for backend, assessment in self.complexity.items():
            if not isinstance(assessment, integrity.ComplexityAssessment):
                raise T3InputError(
                    f"T3Request.complexity[{backend!r}]: must be an "
                    "evaluator.integrity.ComplexityAssessment")
        if not isinstance(self.stability_min_cycles, int) or \
                isinstance(self.stability_min_cycles, bool) or self.stability_min_cycles < 1:
            raise T3InputError(
                "T3Request.stability_min_cycles: required, a positive int, and there is "
                "no default. §10.2 phase 6 gates on REPEATED load/unload; a floor this "
                "module picked for itself is a release threshold nobody ratified.")
        _opt_text(self.campaign_start_at, "T3Request.campaign_start_at")
        if self.campaign_start_at is not None:
            _timestamp(self.campaign_start_at, "T3Request.campaign_start_at")
        if self.gate_scope is not None and not isinstance(self.gate_scope, Mapping):
            raise T3InputError("T3Request.gate_scope: must be a mapping or None")

        cell_ids = [r.cell.cell_id for r in self.cell_results]
        if len(set(cell_ids)) != len(cell_ids):
            raise T3InputError(
                f"T3Request.cell_results: duplicate cell ids {sorted(cell_ids)}; two "
                "results for one cell make the matrix unresolvable")
        planned_by_id = {c.cell_id: c for c in self.plan.cells}
        unplanned = sorted(set(cell_ids) - set(planned_by_id))
        if unplanned:
            raise T3InputError(
                f"T3Request.cell_results: {unplanned} are not in the derived release "
                "plan. §12: scope is mechanically derived; a result for a cell the plan "
                "does not contain is the measured party widening its own matrix."
            )
        for result in self.cell_results:
            drift = _cell_scope_drift(planned_by_id[result.cell.cell_id], result.cell)
            if drift:
                raise T3InputError(
                    f"T3Request.cell_results: the result for {result.cell.cell_id!r} "
                    "carries a cell that contradicts the planned cell of the same id on "
                    + ", ".join(drift) + ". §12 derives scope MECHANICALLY: matching the "
                    "id while relabelling the cell is the measured party redescribing "
                    "its own matrix, which the id check alone does not catch. "
                    "`co_resident` is the sharp edge — flipping it True satisfies the "
                    "§10.2 phase 4 llama_cpu co-residency requirement with a run that "
                    "was never co-resident, and flipping it False deletes the only cell "
                    "class that measures the machine the way production runs it."
                )

    # -- convenience views ---------------------------------------------------

    def results_for_phase(self, phase: str) -> tuple:
        return tuple(r for r in self.cell_results if r.cell.release_phase == phase)

    def fingerprint(self) -> str:
        return sealed_fingerprint(
            sealed=self.sealed, plan=self.plan, protocol=self.protocol,
            waivers=self.waivers, supplied_components=self.supplied_components,
            phase_protocols=self.phase_protocols,
            protocol_registry=self.protocol_registry)


# =============================================================================
# Phase results
# =============================================================================

@dataclass(frozen=True)
class PhaseResult:
    """One §10.2 phase. `check` is DERIVED here and re-derived on construction.

    A phase whose verdict can be stamped is a phase whose verdict can be wrong in
    a direction nobody notices; `evaluator.api.Verdict` uses the same shape for the
    same reason.
    """

    phase_id: str
    check: schemas.Check
    cell_results: tuple = ()
    blocking_reasons: tuple = ()
    notes: tuple = ()
    detail: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.phase_id not in PHASES:
            raise T3InputError(
                f"PhaseResult.phase_id: {self.phase_id!r} is not one of {list(PHASES)}")
        object.__setattr__(self, "cell_results", _typed_tuple(
            self.cell_results, "PhaseResult.cell_results", CellResult))
        object.__setattr__(self, "blocking_reasons", _str_tuple(
            self.blocking_reasons, "PhaseResult.blocking_reasons", non_empty=False))
        object.__setattr__(self, "notes", _str_tuple(
            self.notes, "PhaseResult.notes", non_empty=False))
        if not isinstance(self.detail, Mapping):
            raise T3InputError("PhaseResult.detail: must be a mapping")
        derived = _derive_phase_check(self.cell_results, self.blocking_reasons)
        if not isinstance(self.check, schemas.Check):
            raise T3InputError("PhaseResult.check: must be a schemas.Check")
        if self.check.outcome != derived.outcome:
            raise T3Error(
                f"PhaseResult({self.phase_id}): outcome {self.check.outcome} does not "
                f"follow from its own evidence, which yields {derived.outcome}"
            )

    @property
    def gating_cell_results(self) -> tuple:
        return tuple(r for r in self.cell_results if r.cell.gating)

    def to_dict(self) -> dict:
        return {"phase_id": self.phase_id, "check": _check_dict(self.check),
                "cell_results": [r.to_dict() for r in self.cell_results],
                "blocking_reasons": list(self.blocking_reasons),
                "notes": list(self.notes), "detail": dict(self.detail)}


def _derive_phase_check(cell_results: Sequence[CellResult],
                        blocking_reasons: Sequence[str]) -> schemas.Check:
    """FAIL on any blocking reason; otherwise the worst GATING cell.

    Invariant 15 lives here: a diagnostic cell's outcome is recorded and does not
    enter the phase verdict, in either direction. A failing baseline cell cannot
    veto a release and a passing one cannot justify it.
    """
    if blocking_reasons:
        return _fail(*blocking_reasons)
    gating = [r for r in cell_results if r.cell.gating]
    if not gating:
        return schemas.Check(schemas.PASS)
    return _worst([r.check for r in gating])


def _phase(phase_id: str, *, cell_results: Sequence[CellResult] = (),
           blocking: Sequence[str] = (), notes: Sequence[str] = (),
           detail: Optional[Mapping] = None) -> PhaseResult:
    results = tuple(cell_results)
    reasons = tuple(blocking)
    return PhaseResult(
        phase_id=phase_id, check=_derive_phase_check(results, reasons),
        cell_results=results, blocking_reasons=reasons, notes=tuple(notes),
        detail=dict(detail or {}),
    )


# =============================================================================
# Phase 1 — identity / preflight, INCLUDING the §3.2 per-backend unchanged test
# =============================================================================

@dataclass(frozen=True)
class PreflightProducts:
    """What phase 1 produces for the phases downstream of it."""

    dropped_backends: tuple
    evidence_owed_backends: tuple
    waiver_verifications: tuple
    build_identity_findings: tuple

    def to_dict(self) -> dict:
        return {"dropped_backends": list(self.dropped_backends),
                "evidence_owed_backends": list(self.evidence_owed_backends),
                "waiver_verifications": [v.to_dict() for v in self.waiver_verifications],
                "build_identity_findings": [dict(f) for f in self.build_identity_findings]}


def phase_identity_preflight(request: T3Request) -> tuple:
    """§10.2 phase 1. Returns `(PhaseResult, PreflightProducts)`.

    *"Immutable production and sealed-candidate identity, clean ancestry, evaluator
    hash, host health, resources, storage, rollback target, active waivers.
    Includes the per-backend unchanged test (§3.2)."*
    """
    if not isinstance(request, T3Request):
        raise T3InputError("phase_identity_preflight: request must be a T3Request")
    blocking: list = []
    notes: list = []
    detail: dict = {}

    # --- release authorisation ----------------------------------------------
    if request.mode == MODE_RELEASE and not request.protocol.ratified:
        raise ReleaseProtocolNotRatified(
            f"mode={MODE_RELEASE} was requested under protocol "
            f"{request.protocol.protocol_id!r}, which is not ratified. AK-D20 batches "
            "\"release authorization\" separately from \"search authorization\" for "
            "exactly this reason: ratifying release bindings against a schema sketch is "
            f"worse than not ratifying them. Run it as mode={MODE_DRY_RUN}."
        )
    if request.mode == MODE_DRY_RUN:
        notes.append(
            "dry run: this bundle computes the same verdict as a release run and is NOT "
            "a release authorisation. §10.4's calibration note asks for exactly this "
            "against the preserved v8 and speech freeze artifacts."
        )

    # --- the protocols the CELLS are graded under (§1.6) ---------------------
    # `request.protocol` above is the authority to run the gate. This is the other
    # half: the per-phase instruments the matrix is adjudicated with. They used to
    # arrive as bare ids, so a cell graded under a DRAFT family was indistinguishable
    # from one graded under a ratified Annex B protocol, and the gate licensed claims
    # for both. A non-PASS here blocks: `kernel-research.md:54-56` — a number with no
    # owning protocol cannot become a claim, and this bundle's whole output is a list
    # of claims it licenses.
    phase_protocol_detail: dict = {}
    for backend in sorted(request.phase_protocols):
        for workload_phase in sorted(request.phase_protocols[backend]):
            bound = request.phase_protocols[backend][workload_phase]
            phase_protocol_detail[f"{backend}.{workload_phase}"] = bound.to_dict()
            bound_check = bound.check()
            if bound_check.outcome != schemas.PASS:
                blocking.extend(bound_check.reasons)
    detail["phase_protocols"] = phase_protocol_detail

    # --- the adapters' own release-readiness verdict --------------------------
    # The edge AK5 shipped without. Both speech adapters have always known their
    # families are drafts and returned COULD_NOT_CHECK for it; nothing asked them.
    # The ratified set is DERIVED from this request's own hashed bindings, so this
    # cannot be satisfied by a constant, a flag, or an adapter edit.
    ratified_ids = declared_ratified_protocol_ids(request)
    detail["declared_ratified_protocol_ids"] = list(ratified_ids)
    readiness_detail: dict = {}
    for backend in request.plan.backends:
        readiness_of = RELEASE_READINESS_BY_BACKEND.get(backend)
        if readiness_of is None:
            continue
        readiness = readiness_of(ratified_ids)
        readiness_detail[backend] = _check_dict(readiness)
        if readiness.outcome != schemas.PASS:
            blocking.append(
                f"{backend}: the adapter's own release_gate_readiness() returns "
                f"{readiness.outcome} against the ratified set {list(ratified_ids)} — "
                + "; ".join(readiness.reasons)
            )
    detail["backend_release_readiness"] = readiness_detail

    # --- sealed candidate identity ------------------------------------------
    sealed = request.sealed
    if not sealed.tree_clean:
        blocking.append(
            f"{sealed.candidate_id}: the candidate tree is not clean. "
            "`bench-cpu.md:38-44` defines a candidate as \"a clean committed tree whose "
            "binary reports that commit\"; an uncommitted load-bearing patch is exactly "
            "what the 2026-07-31 speech freeze had to ratify after the fact."
        )
    if not sealed.ancestry_clean:
        blocking.append(
            f"{sealed.candidate_id}: ancestry is not clean — the candidate does not "
            f"descend from production base {sealed.production_base_commit[:12]}. "
            "Invariant 1: every campaign is anchored on the current production tip."
        )
    if not sealed.overlay_present:
        blocking.append(
            f"{sealed.candidate_id}: the project agent-file overlay is not baked into "
            "the candidate. CLAUDE.md's promotion checklist requires it so the new "
            "production tree ships freeze-aware agent files."
        )
    if sealed.source_tree != request.plan.source_tree:
        blocking.append(
            f"the sealed candidate is on {sealed.source_tree!r} but the release plan is "
            f"for {request.plan.source_tree!r}; freezes are per SOURCE TREE (§1.5)")
    for backend, build_dir in sealed.build_dirs.items():
        if _under_production_tree(build_dir):
            blocking.append(
                f"{backend}: the candidate was built in {build_dir!r}, inside a FROZEN "
                "production tree. Invariant 3 — no actor builds in or modifies a "
                "production tree; we version PAST production."
            )

    # --- host health (§10.7) -------------------------------------------------
    host_decision = guards.guard_host_uptime(
        request.host, owner=request.host_owner,
        escalation_deadline=request.host_escalation_deadline, now=request.now)
    detail["host_uptime"] = {"outcome": host_decision.outcome,
                             "reason": host_decision.reason}
    if host_decision.outcome == guards.STOP:
        blocking.append(f"host health: {host_decision.reason}")
    elif host_decision.outcome == guards.COULD_NOT_EVALUATE:
        notes.append(f"host health could not be evaluated: {host_decision.reason}")

    # --- resource claims -----------------------------------------------------
    if not request.resource_claims:
        blocking.append(
            "no resource claim was recorded for the release window. Invariant 9: "
            "resources are ACQUIRED, never observed, and idle sensing is never a claim.")
    for claim in request.resource_claims:
        decision = guards.guard_resource_available(claim)
        if decision.outcome != guards.CONTINUE:
            blocking.append(
                f"resource claim {claim.resource!r}: {decision.reason}")

    # --- storage -------------------------------------------------------------
    storage_decision = guards.guard_storage_headroom(request.storage_observation)
    detail["storage"] = {"outcome": storage_decision.outcome,
                         "reason": storage_decision.reason}
    if storage_decision.outcome == guards.STOP:
        blocking.append(f"storage: {storage_decision.reason}")
    elif storage_decision.outcome == guards.COULD_NOT_EVALUATE:
        notes.append(f"storage headroom could not be evaluated: {storage_decision.reason}")

    # --- rollback target -----------------------------------------------------
    archive_check, archive_notes = request.archive.check()
    notes.extend(archive_notes)
    if archive_check.outcome == schemas.FAIL:
        blocking.extend(archive_check.reasons)
    elif archive_check.outcome == schemas.COULD_NOT_CHECK:
        notes.extend(archive_check.reasons)
    detail["archive"] = _check_dict(archive_check)

    # --- the §3.2 per-backend unchanged test ---------------------------------
    dropped: list = []
    owed: list = []
    findings: list = []
    unchanged_detail: dict = {}
    for backend in request.plan.backends:
        view = request.backend_unchanged.get(backend)
        if view is None:
            owed.append(backend)
            notes.append(
                f"{backend}: no §3.2 unchanged result was supplied, so the backend owes "
                "full candidate-grade evidence under its own protocol. Absence of a "
                "comparison is not evidence of equivalence.")
            continue
        unchanged_detail[backend] = view.to_dict()
        for finding in view.findings:
            findings.append(dict(finding))
            blocking.append(
                f"{backend}: build-identity finding {finding.get('code')} — "
                f"{finding.get('detail')}. §3.2 files a stage disagreement against the "
                "build-identity machinery and the backend owes full evidence."
            )
        if view.may_drop_cells:
            receipt = request.transfer_receipts.get(backend)
            if receipt is None:
                blocking.append(
                    f"{backend}: the unchanged test permits dropping this backend's "
                    "cells, but no transfer receipt names the incumbent artifacts and "
                    "their hashes (§10.2 phase 1). A dropped cell with no receipt is an "
                    "unaudited hole in the matrix."
                )
                owed.append(backend)
                continue
            receipt_check = receipt.check()
            if receipt_check.outcome != schemas.PASS:
                blocking.extend(receipt_check.reasons)
                owed.append(backend)
                continue
            if receipt.incumbent_commit != request.plan.incumbent_commit:
                blocking.append(
                    f"{backend}: the transfer receipt names incumbent "
                    f"{receipt.incumbent_commit[:12]} but the plan's incumbent is "
                    f"{request.plan.incumbent_commit[:12]}; evidence transferred from a "
                    "different incumbent is not transferred evidence"
                )
                owed.append(backend)
                continue
            dropped.append(backend)
            notes.append(
                f"{backend}: cells dropped by the §3.2 unchanged test; the incumbent's "
                f"evidence transfers by identity under receipt over "
                f"{len(receipt.incumbent_artifacts)} artifact(s)."
            )
        else:
            owed.append(backend)
    detail["backend_unchanged"] = unchanged_detail

    # --- active waivers ------------------------------------------------------
    known_cell_ids = [c.cell_id for c in request.plan.cells]
    failing_cell_ids = [r.cell.cell_id for r in request.cell_results
                        if r.cell.gating and r.check.outcome != schemas.PASS]
    verifications: list = []
    # Read the trust boundary ONCE per run, not once per waiver: two waivers in the
    # same run must be judged against the same boundary, or a manifest edited
    # mid-run would give one document an answer the other never faced.
    boundary = human_only_boundary() if request.waivers else None
    for binding in request.waivers:
        verification = verify_waiver(
            binding, candidate_commit=sealed.candidate_commit,
            production_base_commit=sealed.production_base_commit,
            campaign_id=request.campaign_id, known_cell_ids=known_cell_ids,
            failing_cell_ids=failing_cell_ids, now=request.now, boundary=boundary,
            attestation_roots=request.attestation_roots or None)
        verifications.append(verification)
        if verification.check.outcome == schemas.FAIL:
            blocking.append(
                f"waiver {binding.waiver_id} does not verify: "
                f"{'; '.join(verification.check.reasons)}")
        elif verification.check.outcome == schemas.COULD_NOT_CHECK:
            blocking.append(
                f"waiver {binding.waiver_id} could not be verified: "
                f"{'; '.join(verification.check.reasons)}. An unverified waiver "
                "suppresses nothing and cannot be carried past the identity phase."
            )

    products = PreflightProducts(
        dropped_backends=tuple(dict.fromkeys(dropped)),
        evidence_owed_backends=tuple(dict.fromkeys(owed)),
        waiver_verifications=tuple(verifications),
        build_identity_findings=tuple(findings),
    )
    detail["preflight_products"] = products.to_dict()
    return (_phase(PHASE_IDENTITY_PREFLIGHT, blocking=blocking, notes=notes,
                   detail=detail), products)


# =============================================================================
# Phase 2 — build and linkage
# =============================================================================

def phase_build_linkage(request: T3Request, products: PreflightProducts) -> PhaseResult:
    """§10.2 phase 2 — full candidate build outside production, hashes, ABI/backend
    inventory, and per-tree `LD_LIBRARY_PATH` proven by `verify_ggml_linkage.sh`."""
    blocking: list = []
    notes: list = []
    detail: dict = {}
    receipts_by_backend = {r.backend: r for r in request.linkage_receipts}
    inventories = {i.backend: i for i in request.backend_inventories}

    for backend in request.plan.backends:
        if backend in products.dropped_backends:
            notes.append(
                f"{backend}: cells dropped by the §3.2 unchanged test; its linkage is the "
                "incumbent's, already proven at the incumbent's own freeze")
            continue
        if backend not in request.sealed.binary_sha256:
            blocking.append(
                f"{backend}: the sealed candidate records no binary sha256, so there is "
                "nothing to gate and nothing to install")
        if backend not in request.sealed.linkage_sha256:
            blocking.append(
                f"{backend}: the sealed candidate records no linkage sha256. Three ggml "
                "generations coexist on this host; an unrecorded library set cannot be "
                "reproduced at rollback."
            )
        if backend not in request.sealed.build_dirs:
            # Without it, `LinkageReceipt.check(expected_build_dir=None)` has nothing to
            # anchor the receipt's tree root against — i.e. the linkage proof can be
            # made to pass by DELETING the thing it is checked against, which is the
            # cheapest exploit in this phase and reproduces INC-20260731 exactly.
            blocking.append(
                f"{backend}: the sealed candidate records no build directory, so the "
                f"`{LINKAGE_VERIFIER_RELPATH}` receipt cannot be anchored to the tree "
                "this binary was actually built in (§3.2 binds the build directory as "
                "part of candidate identity)."
            )
        receipt = receipts_by_backend.get(backend)
        if receipt is None:
            blocking.append(
                f"{backend}: no `{LINKAGE_VERIFIER_RELPATH}` receipt. A binary that "
                "inherits another tree's ggml runs silently wrong and still prints "
                "plausible numbers (INC-20260731-ggml-linkage-silent-cpu-fallback)."
            )
        else:
            check = receipt.check(expected_build_dir=request.sealed.build_dirs.get(backend))
            detail[f"linkage.{backend}"] = _check_dict(check)
            if check.outcome == schemas.FAIL:
                blocking.extend(check.reasons)
            elif check.outcome == schemas.COULD_NOT_CHECK:
                blocking.extend(
                    f"{r} — an unproven linkage cannot accompany a release" for r in
                    check.reasons)
        inventory = inventories.get(backend)
        if inventory is None:
            blocking.append(
                f"{backend}: no ABI/backend inventory was recorded for the built binary")
        else:
            check = inventory.check()
            detail[f"inventory.{backend}"] = _check_dict(check)
            if check.outcome == schemas.FAIL:
                blocking.extend(check.reasons)
            elif check.outcome == schemas.COULD_NOT_CHECK:
                blocking.extend(check.reasons)

    return _phase(PHASE_BUILD_LINKAGE, blocking=blocking, notes=notes, detail=detail)


# =============================================================================
# Phase 3 — backend correctness
# =============================================================================

def phase_backend_correctness(request: T3Request,
                              products: PreflightProducts) -> PhaseResult:
    """§10.2 phase 3 — exact and unseen op shapes, no silent fallback, NaN/numerical
    bounds, state/rollback, teardown/race, real-model coherence, determinism class.

    Correctness is lexicographically prior to speed (invariant 6,
    `bench-cpu.md:89-90`), so a failure here is never traded against a throughput
    gain later: the verdict aggregator has no arithmetic that could.
    """
    blocking: list = []
    notes: list = []
    detail: dict = {}
    results = _live_results(request, PHASE_BACKEND_CORRECTNESS, products)

    covered = {r.cell.backend for r in results if r.cell.gating}
    for backend in request.plan.backends:
        if backend in products.dropped_backends:
            continue
        if backend not in covered:
            blocking.append(
                f"{backend}: the release matrix contains no gating correctness cell. A "
                "backend that owes candidate-grade evidence and produced none has not "
                "passed correctness, it has skipped it."
            )

    declared = {d.backend: d for d in request.determinism}
    for backend in request.plan.backends:
        if backend in products.dropped_backends:
            continue
        declaration = declared.get(backend)
        if declaration is None:
            blocking.append(
                f"{backend}: no determinism class was declared. Invariant 12 makes the "
                "class an interface; silence about an interface is not a guarantee "
                "about it."
            )
            continue
        check = declaration.check()
        detail[f"determinism.{backend}"] = declaration.to_dict()
        if check.outcome == schemas.FAIL:
            blocking.extend(check.reasons)
        elif check.outcome == schemas.COULD_NOT_CHECK:
            blocking.extend(check.reasons)

    blocking.extend(_evidence_binding_reasons(results))
    return _phase(PHASE_BACKEND_CORRECTNESS, cell_results=results, blocking=blocking,
                  notes=notes, detail=detail)


def _live_results(request: T3Request, phase: str,
                  products: PreflightProducts) -> tuple:
    """The phase's cell results, minus every backend the §3.2 test dropped.

    A dropped backend's cells do not become passing cells — they leave the matrix,
    with a receipt. Keeping them as PASS would let a transfer receipt manufacture
    evidence rather than cite it.
    """
    return tuple(r for r in request.results_for_phase(phase)
                 if r.cell.backend not in products.dropped_backends)


def _evidence_binding_reasons(results: Sequence[CellResult]) -> list:
    reasons: list = []
    for result in results:
        check = result.evidence_check()
        if check.outcome != schemas.PASS:
            reasons.extend(check.reasons)
    return reasons


# =============================================================================
# Phase 4 — the performance matrix
# =============================================================================

def phase_performance_matrix(request: T3Request,
                             products: PreflightProducts) -> PhaseResult:
    """§10.2 phase 4 — candidate versus production on derived production-optimal
    recipes, per phase, under the phase's own protocol and release reps, including
    co-resident cells.

    Four things are checked that a naive matrix would not: that each gating cell is
    at the production-optimal recipe (invariant 15); that its protocol is the one
    that OWNS its (backend, phase) — nothing crosses a protocol boundary (§1.6,
    `MEASUREMENT.md:83-84`); that its rep count meets the protocol's declared
    release reps; and that `llama_cpu` carries at least one co-resident cell, which
    is the only cell class that measures the machine the way production runs it.
    """
    blocking: list = []
    notes: list = []
    detail: dict = {}
    results = _live_results(request, PHASE_PERFORMANCE_MATRIX, products)

    # Invariant 15 makes a diagnostic cell inert in BOTH directions, which is right —
    # and which means reclassifying a backend's whole performance row as diagnostic
    # removes it from the gate for free, taking the rep rule, the protocol-ownership
    # rule and the co-residency rule with it. Phase 3 already refuses that shape ("has
    # not passed correctness, it has skipped it"); phase 4, the phase that decides
    # whether the kernel is actually faster, owes the same refusal.
    gating_backends = {r.cell.backend for r in results if r.cell.gating}
    for backend in request.plan.backends:
        if backend in products.dropped_backends:
            continue
        if backend not in gating_backends:
            blocking.append(
                f"{backend}: the release matrix contains no gating performance cell. A "
                "backend that owes candidate-grade evidence and produced only diagnostic "
                "rows has not been measured against production, it has been excused from "
                "the comparison — and a diagnostic cell can neither veto nor justify a "
                "release (invariant 15)."
            )

    diagnostic = [r for r in results if not r.cell.gating]
    if diagnostic:
        notes.append(
            f"{len(diagnostic)} diagnostic cell(s) recorded and excluded from the gate "
            "in BOTH directions (invariant 15): a baseline cell neither vetoes nor "
            "justifies a release."
        )
        detail["diagnostic_cells"] = [r.cell.cell_id for r in diagnostic]

    for result in results:
        cell = result.cell
        if not cell.gating:
            continue

        commensurability = schemas.check_metric_commensurability(
            cell.backend, {"metric": cell.metric})
        if commensurability.outcome == schemas.FAIL:
            blocking.extend(f"{cell.cell_id}: {r}" for r in commensurability.reasons)

        owning = None
        by_backend = request.phase_protocols.get(cell.backend)
        if isinstance(by_backend, Mapping) and cell.workload_phase is not None:
            bound = by_backend.get(cell.workload_phase)
            owning = None if bound is None else bound.protocol_id
        if owning is None:
            blocking.append(
                f"{cell.cell_id}: no protocol is declared for "
                f"({cell.backend}, {cell.workload_phase}). §1.6 judges each phase under "
                "its OWN protocol; an undeclared owner means the cell is being graded "
                "under whichever protocol it happened to name."
            )
        elif owning != cell.protocol_id:
            blocking.append(
                f"{cell.cell_id}: measured under {cell.protocol_id!r} but "
                f"({cell.backend}, {cell.workload_phase}) is owned by {owning!r}. "
                "Comparisons live within one protocol and one instrument version "
                "(MEASUREMENT.md:83-84)."
            )

        required = request.release_reps_by_protocol.get(cell.protocol_id)
        if required is None:
            blocking.append(
                f"{cell.cell_id}: no release rep count is declared for protocol "
                f"{cell.protocol_id!r}. §10.2 phase 4 says \"release reps\"; a default "
                "invented here would be a rep rule nobody ratified."
            )
        elif cell.reps is None:
            blocking.append(
                f"{cell.cell_id}: records no rep count, so it cannot be shown to meet "
                f"the {required} release reps {cell.protocol_id} requires")
        elif cell.reps < required:
            blocking.append(
                f"{cell.cell_id}: {cell.reps} reps is below the {required} release reps "
                f"{cell.protocol_id} requires")

        if cell.scope_denominator is not None and request.gate_scope is not None:
            # §7.4/§12: a full-machine threshold applied to a partial-machine cell
            # is a category error, and the required defence is that the gate
            # REFUSES rather than demoting the cell. 19 of 45 AutoPilot trials,
            # including a whole pre-registered family, would have been demoted
            # irreparably by exactly this mismatch.
            # COULD_NOT_CHECK blocks too. `check_scope_denominator_admits_gate` says
            # it itself — "an unreadable scope is not a matching scope" — and the
            # asymmetry is what makes it exploitable: an HONEST partial-machine cell
            # FAILs against a full-machine gate, while the same cell with
            # `machine_subset` omitted, misspelled, or declared partial while naming
            # no node returns COULD_NOT_CHECK and, if that were discarded, sailed
            # through. Passing a check by deleting what it inspects is the failure
            # mode §7.4 exists to prevent, not a route around it.
            scope_check = schemas.check_scope_denominator_admits_gate(
                {"scope_denominator": cell.scope_denominator}, request.gate_scope)
            if scope_check.outcome != schemas.PASS:
                blocking.extend(f"{cell.cell_id}: {r}" for r in scope_check.reasons)

    if "llama_cpu" in request.plan.backends and \
            "llama_cpu" not in products.dropped_backends:
        co_resident = [r for r in results
                       if r.cell.backend == "llama_cpu" and r.cell.gating
                       and r.cell.co_resident]
        if not co_resident:
            blocking.append(
                "llama_cpu: the release matrix contains no gating co-resident cell. "
                "§10.2 phase 4 requires co-resident cells and §13.2 makes the "
                "co-resident lineup cells the adapter's own responsibility; a "
                "single-instance matrix has not measured the machine production runs."
            )

    blocking.extend(_evidence_binding_reasons(results))
    return _phase(PHASE_PERFORMANCE_MATRIX, cell_results=results, blocking=blocking,
                  notes=notes, detail=detail)


# =============================================================================
# Phase 5 — quality
# =============================================================================

def phase_quality(request: T3Request, products: PreflightProducts) -> PhaseResult:
    """§10.2 phase 5 — deterministic parity where expected, otherwise PPL/numerical
    and focused quality parity; transfer banked quality once paired parity proves it.

    The §10.5 check lives here because this is where it bit: the v8 quality gate
    compared against a PRESERVED v7 binary, and a baseline that names a rebuild is
    comparing against a binary that no longer exists in the form it was measured in.
    """
    blocking: list = []
    notes: list = []
    detail: dict = {}
    results = _live_results(request, PHASE_QUALITY, products)
    evidence = {e.backend: e for e in request.quality_evidence}

    archived_binaries = {}
    for entry in request.archive.entries:
        for path, digest in entry.binaries:
            archived_binaries[path] = (entry, digest)

    for backend in request.plan.backends:
        if backend in products.dropped_backends:
            continue
        item = evidence.get(backend)
        if item is None:
            blocking.append(
                f"{backend}: no quality evidence. §10.2 phase 5 admits three routes — "
                "deterministic parity, measured parity, or proven transfer — and silence "
                "is not one of them."
            )
            continue
        detail[f"quality.{backend}"] = item.to_dict()

        if item.baseline_is_rebuild:
            blocking.append(
                f"{backend}: the quality baseline {item.baseline_binary_path!r} is a "
                "REBUILD of the incumbent. §10.5 — the v8 gate compared against a "
                "preserved binary, and \"rebuilding an old commit under a drifted "
                "toolchain does not reproduce it\"."
            )
        else:
            archived = archived_binaries.get(item.baseline_binary_path)
            if archived is None:
                blocking.append(
                    f"{backend}: the quality baseline {item.baseline_binary_path!r} is "
                    "not among the archived incumbent binaries. An unarchived baseline "
                    "cannot be re-run, so the comparison cannot be reproduced (§10.5)."
                )
            elif archived[1] != item.baseline_binary_sha256:
                blocking.append(
                    f"{backend}: the quality baseline hashes to "
                    f"{item.baseline_binary_sha256[:12]} but the archived binary at that "
                    f"path is {archived[1][:12]}; the baseline moved under the evidence"
                )

        if item.mode == QUALITY_TRANSFERRED and not item.paired_parity_receipt:
            blocking.append(
                f"{backend}: quality is claimed as TRANSFERRED with no paired-parity "
                "receipt. §10.2 phase 5 transfers banked quality \"once paired parity "
                "proves transfer\"; without the proof it is an assumption wearing a "
                "receipt's name."
            )
        if item.mode == QUALITY_MEASURED_PARITY and not item.shared_question_identity:
            blocking.append(
                f"{backend}: measured quality parity without shared question identity. "
                "The v8 quality contract recorded "
                "`shared_question_identity.baseline_and_candidate_rows_exact: true`; "
                "without it the two arms answered different questions."
            )

    blocking.extend(_evidence_binding_reasons(results))
    return _phase(PHASE_QUALITY, cell_results=results, blocking=blocking, notes=notes,
                  detail=detail)


# =============================================================================
# Phase 6 — stability
# =============================================================================

def phase_stability(request: T3Request, products: PreflightProducts) -> PhaseResult:
    """§10.2 phase 6 — repeated load/unload, concurrency and mixed prefill/decode
    where affected, memory growth, profiler/runtime errors, cleanup."""
    blocking: list = []
    notes: list = []
    detail: dict = {}
    results = _live_results(request, PHASE_STABILITY, products)
    evidence = {e.backend: e for e in request.stability_evidence}

    for backend in request.plan.backends:
        if backend in products.dropped_backends:
            continue
        item = evidence.get(backend)
        if item is None:
            blocking.append(
                f"{backend}: no stability evidence was recorded for the release window")
            continue
        check = item.check(min_cycles=request.stability_min_cycles)
        detail[f"stability.{backend}"] = {**item.to_dict(), "check": _check_dict(check)}
        if check.outcome == schemas.FAIL:
            blocking.extend(check.reasons)
        elif check.outcome == schemas.COULD_NOT_CHECK:
            blocking.extend(check.reasons)

    blocking.extend(_evidence_binding_reasons(results))
    return _phase(PHASE_STABILITY, cell_results=results, blocking=blocking, notes=notes,
                  detail=detail)


# =============================================================================
# Phase 7 — capacity, utility, and the §1.6 per-phase objective
# =============================================================================

def _standing_binding_reasons(request: T3Request, standings: Sequence[PhaseStanding],
                              products: PreflightProducts) -> list:
    """Reasons a §1.6 standing is not bound to the matrix cells it summarises.

    `PhaseStanding` is a caller-supplied record, and phase 7 decides the whole §1.6
    objective from those records. Phase 4 measures the matrix. Nothing joined them:
    a standing with `cell_ids=()`, or citing a correctness cell, or a diagnostic
    cell, or a cell of another phase, or a cell the run produced no result for,
    satisfied the conjunction on its own — and dropping every prefill CELL while
    keeping the prefill STANDING left the run at PASS.

    That is `readiness.py`'s own fix one plane down (a figure selected without
    consulting the cell's verdict), except here it decides a freeze rather than an
    advisory line. So each standing must name at least one planned, gating,
    production-optimal `performance_matrix` cell of its own (backend, phase) for
    which this run recorded a result. It does NOT re-derive the standing — that is
    `evaluator.statistics`' e-process and this gate adjudicates the rule over it.
    """
    reasons: list = []
    planned = {c.cell_id: c for c in request.plan.cells}
    measured = {r.cell.cell_id for r in request.cell_results}
    for standing in standings:
        if standing.backend in products.dropped_backends:
            continue
        label = f"{standing.backend}/{standing.workload_phase}"
        if not standing.cell_ids:
            reasons.append(
                f"{label}: the standing names no cells. §1.6 is a statement about "
                "measured cells at the production-optimal recipe; a standing with no "
                "cells behind it is a declaration, and invariant 14 lets the loop "
                "request but never declare."
            )
            continue
        supporting = 0
        for cell_id in standing.cell_ids:
            cell = planned.get(cell_id)
            if cell is None:
                reasons.append(
                    f"{label}: the standing cites {cell_id!r}, which is not in the "
                    "release plan. §12 derives scope mechanically; a standing over an "
                    "unplanned cell widens its own matrix.")
                continue
            if cell.backend != standing.backend or \
                    cell.workload_phase != standing.workload_phase:
                reasons.append(
                    f"{label}: the standing cites {cell_id!r}, which is "
                    f"({cell.backend}, {cell.workload_phase}). A standing assembled "
                    "across a phase or backend boundary is not that phase's standing "
                    "(§1.6, MEASUREMENT.md:83-84).")
                continue
            if cell.release_phase != PHASE_PERFORMANCE_MATRIX:
                reasons.append(
                    f"{label}: the standing cites {cell_id!r}, a "
                    f"{cell.release_phase!r} cell. Throughput standings come from the "
                    "performance matrix; a correctness or stability cell carries no "
                    "throughput comparison to be non-inferior about.")
                continue
            if not cell.gating:
                reasons.append(
                    f"{label}: the standing cites diagnostic cell {cell_id!r}. "
                    "Invariant 15 — a baseline/off-recipe cell neither vetoes nor "
                    "justifies a release, and supporting a standing IS justifying one.")
                continue
            if cell_id not in measured:
                reasons.append(
                    f"{label}: the standing cites {cell_id!r}, for which this run "
                    "recorded no result. An unmeasured cell cannot support a standing "
                    "about what was measured.")
                continue
            supporting += 1
        if supporting == 0:
            reasons.append(
                f"{label}: no cell the standing names survived the binding above, so "
                "the standing rests on nothing this run measured. §1.6's conjunction "
                "must not be satisfiable by supplying standings and no matrix."
            )
    return reasons


def _phase_trade_realisation(trade: PhaseTradeException,
                             standing_by_phase: Mapping) -> tuple:
    """`(reasons, detail)` for whether the trade's gain ACTUALLY happened.

    §1.6 makes a phase trade *"a pre-declared campaign exception"* naming the exact
    regression band, the exact expected gain, and the roles affected. The gate
    validated all three for STRUCTURE and then compared none of them to anything:
    `expected_gain` was a number that had to be present and never had to be true.
    So an operator-approved trade admitted its regression on the strength of a gain
    nobody checked for, and a run where the gaining phase ALSO regressed passed
    silently — the trade paying for itself with a loss.

    T3 does not re-derive standings (`evaluator.statistics` owns the e-process), so
    the comparison is over the standing VOCABULARY, which is the strongest
    statement available here: the phase the trade was granted for must carry an
    IMPROVED standing. `not worse` is not `+expected_gain`, and `we could not tell`
    is not either — both are reported as a finding rather than admitted, because a
    trade is the one place §1.6's conjunction is deliberately relaxed and a relaxed
    conjunct nobody checks is a deleted one.

    This never RE-GRADES the trade. It compares the pre-declaration to the
    standings this run produced and says whether they agree.
    """
    label = f"{trade.backend}/{trade.regressing_phase}"
    gaining = standing_by_phase.get((trade.backend, trade.gaining_phase))
    detail = {"expected_gain": trade.expected_gain,
              "gaining_phase": trade.gaining_phase,
              "gaining_standing": None if gaining is None else gaining.standing,
              "realised": False}
    if gaining is None:
        return ([
            f"{label}: the phase-trade exception is priced at a gain of "
            f"{trade.expected_gain} on {trade.backend}/{trade.gaining_phase}, and this "
            "run produced no standing for that phase. §1.6 admits a regression in "
            "exchange for a NAMED gain; a gain nobody measured is a regression with a "
            "story attached."
        ], detail)
    if gaining.standing == STANDING_IMPROVED:
        detail["realised"] = True
        return ([], detail)
    if gaining.standing == STANDING_REGRESSED:
        return ([
            f"{label}: the exception is priced at a gain of {trade.expected_gain} on "
            f"{trade.gaining_phase}, and {trade.gaining_phase} REGRESSED. The realised "
            "effect contradicts the pre-declared expected gain: the trade paid for a "
            "regression with a second one."
        ], detail)
    return ([
        f"{label}: the exception is priced at a gain of {trade.expected_gain} on "
        f"{trade.gaining_phase}, whose standing is {gaining.standing!r}. The gain the "
        "trade was granted for was not established, and §1.6 relaxes its conjunction "
        "for a trade whose gain HAPPENED, not for one that was declared."
    ], detail)


def phase_capacity_utility(request: T3Request,
                           products: PreflightProducts) -> PhaseResult:
    """§10.2 phase 7 — VRAM/RAM/context-capacity non-inferiority; every protected
    cell within its fixed floor; the §1.6 per-phase rule satisfied, or an
    operator-approved phase-trade exception present.

    §1.6, in full: *"At the production-optimal recipe for every protected cell, both
    prefill and decode throughput must be non-inferior to the production anchor, and
    at least one must improve."* Each phase is judged under its own protocol, so
    nothing here folds two protocols into a scalar — the composite objective was
    withdrawn on 2026-08-02 because `MEASUREMENT.md:83-84` and
    `gpu-cross-device.md:106-111` each forbid it independently.
    """
    blocking: list = []
    notes: list = []
    detail: dict = {}
    results = _live_results(request, PHASE_CAPACITY_UTILITY, products)

    planned_cells = {c.cell_id: c for c in request.plan.cells}
    for floor in request.capacity_floors:
        if floor.cell_id not in planned_cells:
            blocking.append(
                f"capacity floor names cell {floor.cell_id!r}, which is not in the "
                "release plan")
            continue
        if planned_cells[floor.cell_id].backend in products.dropped_backends:
            continue
        check = floor.check()
        detail[f"floor.{floor.cell_id}.{floor.quantity}"] = floor.to_dict()
        if check.outcome != schemas.PASS:
            blocking.extend(check.reasons)

    trades: dict = {}
    for trade in request.phase_trades:
        trade_check = trade.check()
        detail[f"phase_trade.{trade.backend}.{trade.regressing_phase}"] = trade.to_dict()
        if trade_check.outcome != schemas.PASS:
            blocking.extend(trade_check.reasons)
            continue
        if request.campaign_start_at is not None and \
                trade.campaign_start_at != request.campaign_start_at:
            blocking.append(
                f"the {trade.backend} phase-trade exception names campaign start "
                f"{trade.campaign_start_at}, the run's campaign started at "
                f"{request.campaign_start_at}")
            continue
        trades[(trade.backend, trade.regressing_phase)] = trade

    standings = [s for s in request.standings
                 if s.backend not in products.dropped_backends]
    by_backend: dict = {}
    standing_by_phase: dict = {}
    for standing in standings:
        by_backend.setdefault(standing.backend, []).append(standing)
        key = (standing.backend, standing.workload_phase)
        prior = standing_by_phase.get(key)
        if prior is not None:
            # Every consumer of a per-phase standing in this gate is a dict keyed on
            # (backend, phase) — here, `owned` below, and the trade-realisation
            # comparison — so a second standing for one phase silently WINS, and the
            # caller supplying it is the party being gated. Two standings that
            # disagree are a contradiction the run must state, not a preference the
            # gate resolves by insertion order.
            blocking.append(
                f"{standing.backend}/{standing.workload_phase}: this run supplies more "
                f"than one standing for the phase ({prior.standing!r} from "
                f"{prior.evidence_ref!r}, then {standing.standing!r} from "
                f"{standing.evidence_ref!r}). §1.6 adjudicates ONE standing per phase; "
                "a duplicate is resolved last-wins by every dict in this gate, which "
                "makes the adjudicated verdict a function of the order the caller "
                "listed them in.")
        standing_by_phase[key] = standing
    detail["standings"] = [s.to_dict() for s in standings]

    # A trade's expected_gain is a PRE-DECLARATION, and until now it was only ever
    # validated for structure. Compare it to what this run actually measured.
    for key, trade in sorted(trades.items()):
        if trade.backend in products.dropped_backends:
            continue
        trade_reasons, trade_detail = _phase_trade_realisation(trade, standing_by_phase)
        detail[f"phase_trade.{key[0]}.{key[1]}.realisation"] = trade_detail
        blocking.extend(trade_reasons)

    # A standing is a SUMMARY of measured cells, not a declaration that stands in for
    # them. Nothing else in this gate joins the two: phase 4 measures cells and this
    # phase adjudicates §1.6 over `request.standings`, so without this the whole
    # conjunction is satisfiable by supplying the standings and no matrix at all.
    blocking.extend(_standing_binding_reasons(request, standings, products))

    for backend in request.plan.backends:
        if backend in products.dropped_backends:
            continue
        declared_phases = request.phase_protocols.get(backend)
        if not isinstance(declared_phases, Mapping) or not declared_phases:
            blocking.append(
                f"{backend}: no per-phase protocol map is declared, so the §1.6 objective "
                "has no phases to be evaluated over")
            continue
        # §1.6's conjunction is over the backend's OWN phase vocabulary. A protocol
        # map naming only `decode` satisfies "every declared phase" by declaring one
        # — the conjunct is not failed, it is deleted. `schemas.PHASES_BY_BACKEND` is
        # the SSOT `plan.BackendBinding` and `readiness.compute_readiness` already
        # hold themselves to; the release gate owes the same reading of the same rule.
        owed_phases = schemas.PHASES_BY_BACKEND.get(backend)
        if owed_phases:
            undeclared = sorted(set(owed_phases) - set(declared_phases))
            if undeclared:
                blocking.append(
                    f"{backend}: the per-phase protocol map declares "
                    f"{sorted(declared_phases)} but §1.6 judges this backend over "
                    f"{sorted(owed_phases)}; {undeclared} is never asked about. A "
                    "conjunction satisfied by dropping a conjunct is not the "
                    "conjunction."
                )
        owned = {s.workload_phase: s for s in by_backend.get(backend, [])}
        improved = 0
        for phase_name, bound in declared_phases.items():
            protocol_id = bound.protocol_id
            standing = owned.get(phase_name)
            if standing is None:
                blocking.append(
                    f"{backend}/{phase_name}: no standing was produced. §1.6 requires "
                    "BOTH prefill and decode to be non-inferior; an unmeasured phase is "
                    "not a non-inferior one."
                )
                continue
            if standing.protocol_id != protocol_id:
                blocking.append(
                    f"{backend}/{phase_name}: standing was taken under "
                    f"{standing.protocol_id!r}, but the phase is owned by "
                    f"{protocol_id!r}")
                continue
            if standing.standing in OBJECTIVE_SATISFYING_STANDINGS:
                if standing.standing == STANDING_IMPROVED:
                    improved += 1
                continue
            if standing.standing == STANDING_REGRESSED:
                trade = trades.get((backend, phase_name))
                if trade is None:
                    blocking.append(
                        f"{backend}/{phase_name}: REGRESSED against the production "
                        "anchor at the production-optimal recipe, and no operator-"
                        "approved phase-trade exception covers it (§1.6)."
                    )
                else:
                    gaining = standing_by_phase.get((backend, trade.gaining_phase))
                    notes.append(
                        f"{backend}/{phase_name}: regression admitted by the pre-declared "
                        f"phase-trade exception approved by {trade.approved_by} "
                        f"(band {list(trade.regression_band)}, expected gain "
                        f"{trade.expected_gain} on {trade.gaining_phase}, realised "
                        f"standing {None if gaining is None else gaining.standing!r}, "
                        f"roles {list(trade.roles_affected)})."
                    )
            else:
                blocking.append(
                    f"{backend}/{phase_name}: standing is {standing.standing!r}. "
                    "Non-inferiority is an e-process decision "
                    "(MEASUREMENT.md:30-32); a window that did not resolve has not made "
                    "it, and \"we could not tell\" is not \"not worse\"."
                )
        if improved == 0:
            blocking.append(
                f"{backend}: no phase improved. §1.6 requires non-inferiority on every "
                "phase AND improvement on at least one; a release that improves nothing "
                "spends a freeze on churn."
            )

    blocking.extend(_evidence_binding_reasons(results))
    return _phase(PHASE_CAPACITY_UTILITY, cell_results=results, blocking=blocking,
                  notes=notes, detail=detail)


# =============================================================================
# Phase 8 — the transaction dry run
# =============================================================================

def phase_transaction_dry_run(request: T3Request) -> PhaseResult:
    """§10.2 phase 8 — exact next version, branch/tag, install path, archive/rollback
    link, symlink diff, service impact, era actions, receipt paths.

    Nothing here is executed. `TransactionPlan` refuses to be constructed with
    `executed=True`, so by the time this function runs the boundary is already
    held; what is left is checking that the plan is coherent, that the version
    genuinely moves PAST production rather than patching it, and that the archive
    §10.5 requires is actually there to roll back to.
    """
    blocking: list = []
    notes: list = []
    transaction = request.transaction
    plan = request.plan
    detail: dict = {"transaction": transaction.to_dict(),
                    "archive": request.archive.to_dict()}

    match = _PRODUCTION_BRANCH_RE.match(transaction.next_branch)
    if match is None:
        blocking.append(
            f"the next branch {transaction.next_branch!r} does not match "
            "`production-(consolidated|speech)-vN`; a release that does not name a "
            "production version cannot be the version past production")
    else:
        if int(match.group(2)) != transaction.next_version_number:
            blocking.append(
                f"the next branch {transaction.next_branch!r} and version number "
                f"{transaction.next_version_number} disagree")
        expected_family = "speech" if plan.source_tree in ("whisper.cpp", "qwentts.cpp") \
            else "consolidated"
        if match.group(1) != expected_family:
            blocking.append(
                f"the next branch {transaction.next_branch!r} names the "
                f"{match.group(1)!r} family, but {plan.source_tree} freezes on the "
                f"{expected_family!r} family (CLAUDE.md, §1.5)")
    if transaction.next_branch == plan.incumbent_branch:
        blocking.append(
            f"the next branch equals the incumbent {plan.incumbent_branch!r}. "
            "Production is FROZEN and is never patched in place — we version PAST it.")
    if transaction.next_version_number <= plan.incumbent_version_number:
        blocking.append(
            f"the next version {transaction.next_version_number} does not exceed the "
            f"incumbent {plan.incumbent_version_number}; a reused or lower version name "
            "makes the era ambiguous and the rollback anchor unresolvable")

    if not transaction.symlink_diff:
        blocking.append(
            "the transaction moves no stable kernel symlink, so nothing would become "
            "live. A release plan that installs nothing is not a transaction.")
    for link, current, nxt in transaction.symlink_diff:
        if current == nxt:
            blocking.append(
                f"symlink {link!r} is planned to point at {nxt!r}, which is where it "
                "already points; a no-op entry hides which links actually move")

    archive_check, archive_notes = request.archive.check()
    notes.extend(archive_notes)
    if archive_check.outcome == schemas.FAIL:
        blocking.extend(archive_check.reasons)
    elif archive_check.outcome == schemas.COULD_NOT_CHECK:
        blocking.extend(
            f"{reason} — a release cannot be gated PASS with no rollback target"
            for reason in archive_check.reasons)

    n1 = request.archive.entry(ARCHIVE_GENERATION_N1)
    if n1 is not None:
        if transaction.rollback_branch is None or transaction.rollback_head is None:
            blocking.append(
                "the transaction records no rollback branch/head even though an "
                "incumbent is archived; the archive and the plan must name the same "
                "anchor or the rollback is untested")
        else:
            if transaction.rollback_branch != n1.branch:
                blocking.append(
                    f"the rollback branch {transaction.rollback_branch!r} is not the "
                    f"archived N-1 branch {n1.branch!r}")
            if transaction.rollback_head != n1.commit:
                blocking.append(
                    f"the rollback head {transaction.rollback_head[:12]} is not the "
                    f"archived N-1 commit {n1.commit[:12]}")
            if n1.commit != plan.incumbent_commit:
                blocking.append(
                    f"the archived N-1 commit {n1.commit[:12]} is not the plan's "
                    f"incumbent {plan.incumbent_commit[:12]}; the thing archived is not "
                    "the thing being replaced")

    if not transaction.era_actions:
        blocking.append(
            "the transaction drafts no era action. §11.4: a new kernel changes the "
            "orchestrator's speed priors and AutoPilot's speed era even when model "
            "quality is identical, so the era row is part of the transaction — as a "
            "DRAFT for the operator (§1.3)."
        )
    for path in transaction.receipt_paths:
        if storage.is_scratch_path(path):
            blocking.append(
                f"receipt path {path!r} is a scratch citation; evidence of record lives "
                "in-repo (MEASUREMENT.md:146-156)")

    return _phase(PHASE_TRANSACTION_DRY_RUN, blocking=blocking, notes=notes, detail=detail)


# =============================================================================
# Phase 9 — the seal
# =============================================================================

@dataclass(frozen=True)
class ReleaseReceipt:
    """What the release may CLAIM, and what a waiver took off the table.

    v8's receipt read `q8_claim: "none; campaign-scoped WAIVE-Q8 remains binding and
    v8 makes no Q8 non-regression claim"`. That is the shape: a waived cell does not
    make a weaker claim, it makes NO claim, and the forfeit is named.
    """

    claims: tuple = ()
    suppressed_claims: tuple = ()
    forfeited_claims: tuple = ()
    #: Claims individual cells earned on a run whose VERDICT was FAIL. They are kept
    #: — losing them would hide which cells did pass — but they are not in `claims`,
    #: because a release that did not pass licenses nothing. AK6 reads `claims` off
    #: this object to render "claims licensed: N" onto the operator's first page and
    #: to write the package record; a FAILing run was putting a positive number
    #: there.
    withheld_claims: tuple = ()

    def to_dict(self) -> dict:
        return {"claims": list(self.claims),
                "suppressed_claims": [dict(s) for s in self.suppressed_claims],
                "forfeited_claims": list(self.forfeited_claims),
                "withheld_claims": list(self.withheld_claims)}


@dataclass(frozen=True)
class ReleaseBundle:
    """§10.2 phase 9 — one hash over protocol, plan, raw evidence, reducers,
    validation results, active waivers, and the exact transaction."""

    bundle_sha256: str
    payload: Mapping

    def __post_init__(self) -> None:
        _sha256(self.bundle_sha256, "ReleaseBundle.bundle_sha256")
        if not isinstance(self.payload, Mapping):
            raise T3InputError("ReleaseBundle.payload: must be a mapping")
        recomputed = schemas.content_hash(self.payload)
        if recomputed != self.bundle_sha256:
            raise T3Error(
                f"ReleaseBundle: the declared digest {self.bundle_sha256[:12]} is not the "
                f"hash of its own payload ({recomputed[:12]}). A bundle whose seal can be "
                "stamped independently of its contents seals nothing."
            )
        authority = schemas.find_authority_flavoured_keys(self.payload)
        if authority:
            raise T3Error(
                f"ReleaseBundle: payload carries authority-flavoured keys {authority}. "
                "§1.3: a freeze crosses four human-only trust boundaries, so there is no "
                "such authority for a machine-authored record to declare."
            )

    def to_dict(self) -> dict:
        return {"bundle_sha256": self.bundle_sha256, "payload": dict(self.payload)}


def _component_digests(request: T3Request, phase_results: Sequence[PhaseResult],
                       verifications: Sequence[WaiverVerification]) -> tuple:
    """The seven §10.2 component digests, and the reasons any of them is unusable.

    Five are supplied; two — `validation_results` and `active_waivers` — are
    computed here from this run's own products. A caller cannot hand T3 a digest
    of validation results T3 did not produce.
    """
    reasons: list = []
    digests: dict = {}
    for name in SUPPLIED_COMPONENTS:
        value = request.supplied_components.get(name)
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            reasons.append(
                f"bundle component {name!r} is absent or not a sha256; §10.2 phase 9 "
                "hashes seven named components and an unhashed one is an unsealed one")
            continue
        if schemas.is_placeholder_digest(value):
            reasons.append(
                f"bundle component {name!r} is the digest of no bytes at all — the "
                "artifact was never read")
            continue
        digests[name] = value
    extra = sorted(set(request.supplied_components) - set(SUPPLIED_COMPONENTS))
    if extra:
        reasons.append(
            f"supplied_components carries {extra}, which T3 computes for itself "
            f"({list(COMPUTED_COMPONENTS)}) or does not know. A caller-supplied digest "
            "for a computed component is a second source of truth for the seal.")
    digests[COMPONENT_VALIDATION_RESULTS] = schemas.content_hash(
        [p.to_dict() for p in phase_results])
    digests[COMPONENT_ACTIVE_WAIVERS] = schemas.content_hash(
        [v.to_dict() for v in verifications])
    return (digests, reasons)


def phase_seal(request: T3Request, phase_results: Sequence[PhaseResult], *,
               verdict: str, receipt: ReleaseReceipt,
               verifications: Sequence[WaiverVerification],
               products: PreflightProducts,
               fingerprint: str) -> tuple:
    """§10.2 phase 9. Returns `(PhaseResult, Optional[ReleaseBundle])`.

    The bundle is only built when every component hashes. A partial seal is worse
    than no seal: it looks like a sealed release and is one missing digest away
    from being unreproducible.
    """
    digests, reasons = _component_digests(request, phase_results, verifications)
    if reasons:
        return (_phase(PHASE_SEAL, blocking=reasons,
                       detail={"component_digests": digests}), None)

    review = _requires_human_code_review(request)
    payload = {
        "schema": BUNDLE_SCHEMA,
        "record_class": RECORD_CLASS,
        "run_id": request.run_id,
        "campaign_id": request.campaign_id,
        "mode": request.mode,
        "tier": TIER,
        "created_at": request.now,
        "release_protocol": request.protocol.to_dict(),
        "sealed_candidate": request.sealed.to_dict(),
        "release_plan": request.plan.to_dict(),
        "sealed_fingerprint": fingerprint,
        "component_digests": {k: digests[k] for k in sorted(digests)},
        "phase_results": [p.to_dict() for p in phase_results],
        "verdict": verdict,
        "receipt": receipt.to_dict(),
        "active_waivers": [v.to_dict() for v in verifications],
        "preflight_products": products.to_dict(),
        "transaction": request.transaction.to_dict(),
        "incumbent_archive": request.archive.to_dict(),
        "requires_human_code_review": review["requires_human_code_review"],
        "first_page_notice": review["first_page_notice"],
    }
    bundle = ReleaseBundle(bundle_sha256=schemas.content_hash(payload), payload=payload)
    return (_phase(PHASE_SEAL, detail={"bundle_sha256": bundle.bundle_sha256,
                                       "component_digests": payload["component_digests"]}),
            bundle)


def _requires_human_code_review(request: T3Request) -> dict:
    """§10.6 — mark, do not fail, and say so on the first page.

    `integrity.assess_complexity_ceiling()` already computes this per backend
    against the adapter's declared ceiling; this folds the per-backend assessments
    into the one flag the package's first page carries.
    """
    reasons: list = []
    for backend in sorted(request.complexity):
        assessment = request.complexity[backend]
        if assessment.requires_human_code_review:
            reasons.extend(f"{backend}: {r}" for r in assessment.reasons)
    missing = [b for b in request.plan.backends if b not in request.complexity]
    if missing:
        # An undeclared ceiling cannot clear a diff. §10.6 exists because
        # LLM-authored kernel C++/HIP should not reach a release package unreviewed
        # at arbitrary size, and "we did not measure the size" is not "the size was
        # small".
        reasons.append(
            f"no §10.6 complexity assessment was supplied for {missing}; an unassessed "
            "diff has not cleared a blast-radius ceiling")
    notice = None
    if reasons:
        notice = f"{integrity.REQUIRES_HUMAN_CODE_REVIEW} — " + "; ".join(reasons)
    return {"requires_human_code_review": bool(reasons), "first_page_notice": notice,
            "reasons": reasons}


# =============================================================================
# The verdict — PASS / FAIL / PASS_WITH_WAIVER
# =============================================================================

@dataclass(frozen=True)
class VerdictComputation:
    verdict: str
    failed_cells: tuple
    unevaluable_cells: tuple
    waived_cells: tuple
    blocking_reasons: tuple
    receipt: ReleaseReceipt

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "failed_cells": list(self.failed_cells),
                "unevaluable_cells": list(self.unevaluable_cells),
                "waived_cells": [dict(w) for w in self.waived_cells],
                "blocking_reasons": list(self.blocking_reasons),
                "receipt": self.receipt.to_dict()}


def compute_verdict(phase_results: Sequence[PhaseResult],
                    verifications: Sequence[WaiverVerification]) -> VerdictComputation:
    """Aggregate the phases into one of exactly three verdicts.

    Two rules do the work:

      * **A waiver covers CELLS, never a phase blocker.** A blocking reason is the
        integrity spine — identity, linkage, the transaction, an unverified waiver —
        and none of it is the kind of scoped, claim-forfeiting exclusion §10.4
        describes. v8's own waiver excluded two model/shape PAIRS from a matrix; it
        did not waive whether the binary linked correctly.
      * **The vocabulary is closed at three**, so an unevaluable gating cell lands
        in FAIL. The distinction survives in `unevaluable_cells`, separately from
        `failed_cells`, because "we could not tell" is not "it is worse" and only one
        of them is a fact about the candidate.
    """
    results = _typed_tuple(phase_results, "compute_verdict: phase_results", PhaseResult)
    verified = _typed_tuple(verifications, "compute_verdict: verifications",
                            WaiverVerification)
    covered: dict = {}
    forfeited: list = []
    for verification in verified:
        if not verification.verified:
            continue
        forfeited.extend(verification.forfeited_claims)
        for cell_id in verification.covered_cell_ids:
            covered.setdefault(cell_id, verification.waiver_id)

    failed: list = []
    unevaluable: list = []
    waived: list = []
    blocking: list = []
    claims: list = []
    suppressed: list = []

    for phase in results:
        blocking.extend(f"{phase.phase_id}: {reason}" for reason in phase.blocking_reasons)
        for result in phase.gating_cell_results:
            cell_id = result.cell.cell_id
            outcome = result.check.outcome
            if outcome == schemas.PASS:
                if result.cell.claim:
                    claims.append(result.cell.claim)
                continue
            waiver_id = covered.get(cell_id)
            if waiver_id is not None:
                waived.append({"cell_id": cell_id, "waiver_id": waiver_id,
                               "waived_outcome": outcome,
                               "claim": result.cell.claim})
                if result.cell.claim:
                    suppressed.append({"claim": result.cell.claim,
                                       "waiver_id": waiver_id, "cell_id": cell_id})
                continue
            if outcome == schemas.FAIL:
                failed.append(cell_id)
            else:
                unevaluable.append(cell_id)

    if failed or unevaluable or blocking:
        verdict = "FAIL"
    elif waived:
        verdict = "PASS_WITH_WAIVER"
    else:
        verdict = "PASS"

    # The verdict decides whether anything is licensed at all. `RECORD_CLASS` says
    # the claims a bundle licenses are ENUMERATED, not implied — and a FAILing run
    # enumerates none of them, however many individual cells passed. This is a seam
    # fact, not a cosmetic one: AK6 reads `T3Result.receipt.claims` to render the
    # count on the operator's first page and copies the receipt verbatim into the
    # durable package record, so a populated `claims` on a FAIL is a licensed-looking
    # claim list attached to a release that did not pass.
    licensed = tuple(dict.fromkeys(claims))
    receipt = ReleaseReceipt(
        claims=() if verdict == "FAIL" else licensed,
        suppressed_claims=tuple(suppressed),
        forfeited_claims=tuple(dict.fromkeys(forfeited)),
        withheld_claims=licensed if verdict == "FAIL" else (),
    )
    return VerdictComputation(
        verdict=verdict, failed_cells=tuple(failed),
        unevaluable_cells=tuple(unevaluable), waived_cells=tuple(waived),
        blocking_reasons=tuple(blocking), receipt=receipt,
    )


# =============================================================================
# The run
# =============================================================================

@dataclass(frozen=True)
class T3Result:
    """Everything one T3 run produced. `bundle` is None when the seal could not close."""

    run_id: str
    mode: str
    verdict: str
    fingerprint: str
    rerun: RerunDisposition
    phase_results: tuple
    verdict_computation: VerdictComputation
    products: PreflightProducts
    bundle: Optional[ReleaseBundle]
    requires_human_code_review: bool
    first_page_notice: Optional[str]

    @property
    def receipt(self) -> ReleaseReceipt:
        return self.verdict_computation.receipt

    def phase(self, phase_id: str) -> PhaseResult:
        for result in self.phase_results:
            if result.phase_id == phase_id:
                return result
        raise KeyError(phase_id)

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "mode": self.mode, "verdict": self.verdict,
                "fingerprint": self.fingerprint, "rerun": self.rerun.to_dict(),
                "phase_results": [p.to_dict() for p in self.phase_results],
                "verdict_computation": self.verdict_computation.to_dict(),
                "preflight_products": self.products.to_dict(),
                "bundle_sha256": None if self.bundle is None else self.bundle.bundle_sha256,
                "requires_human_code_review": self.requires_human_code_review,
                "first_page_notice": self.first_page_notice}


def run_t3(request: T3Request) -> T3Result:
    """Run all nine §10.2 phases in order and seal the bundle.

    The rerun guard runs FIRST and RAISES rather than returning a FAIL: refusing to
    spend the full release matrix is not a verdict about the candidate, and
    recording it as one would put "we declined to re-measure" in the same column as
    "the kernel regressed".
    """
    if not isinstance(request, T3Request):
        raise T3InputError("run_t3: request must be a T3Request")

    fingerprint = request.fingerprint()
    rerun = check_rerun(fingerprint, request.attempt_ledger, now=request.now,
                        cooldown_seconds=request.cooldown_seconds,
                        repair=request.stage_repair)
    if not rerun.admissible:
        raise RerunRefused(f"{rerun.code}: {rerun.reason}")

    preflight, products = phase_identity_preflight(request)
    phase_results = [
        preflight,
        phase_build_linkage(request, products),
        phase_backend_correctness(request, products),
        phase_performance_matrix(request, products),
        phase_quality(request, products),
        phase_stability(request, products),
        phase_capacity_utility(request, products),
        phase_transaction_dry_run(request),
    ]
    computation = compute_verdict(phase_results, products.waiver_verifications)
    seal, bundle = phase_seal(
        request, tuple(phase_results), verdict=computation.verdict,
        receipt=computation.receipt, verifications=products.waiver_verifications,
        products=products, fingerprint=fingerprint)
    phase_results.append(seal)

    # The seal is a phase like any other: if it could not close, the run did not
    # produce a releasable bundle and the verdict must say so rather than reporting
    # a PASS whose evidence nobody can rehash.
    if seal.check.outcome != schemas.PASS:
        computation = compute_verdict(tuple(phase_results), products.waiver_verifications)

    review = _requires_human_code_review(request)
    return T3Result(
        run_id=request.run_id, mode=request.mode, verdict=computation.verdict,
        fingerprint=fingerprint, rerun=rerun, phase_results=tuple(phase_results),
        verdict_computation=computation, products=products, bundle=bundle,
        requires_human_code_review=review["requires_human_code_review"],
        first_page_notice=review["first_page_notice"],
    )


class T3Runner:
    """The `evaluator.api.ReleaseTierEvaluator` seam, implemented.

    `api.admit_tier()` refuses T3 by name and points at this class. Registering it
    is what turns that refusal from a dead end into a boundary.
    """

    tier = TIER

    def evaluate_release(self, request: Any) -> T3Result:
        if not isinstance(request, T3Request):
            raise T3InputError(
                f"T3Runner.evaluate_release: expected a T3Request, got "
                f"{type(request).__name__}")
        return run_t3(request)


# =============================================================================
# Calibration — the §10.4 dry run against preserved freeze artifacts
# =============================================================================

@dataclass(frozen=True)
class PreservedFreeze:
    """The subset of a PRESERVED operator freeze artifact that T3 can be calibrated
    against.

    §10.4's calibration note: *"the T3 dry-run against preserved v8 artifacts should
    predict a FAIL without the waiver. If it passes, the compiler is wrong."* That
    only means something if the dry run is built from what the artifact actually
    records, so every field here is READ from a preserved JSON and nothing is
    invented. Where the artifact records nothing — the v8 attestation carries no
    sha256 for the preserved v7 baseline binary it compared against — the field is
    None and the calibration reports the hole rather than filling it.
    """

    label: str
    source_tree: str
    backends: tuple
    production_branch: str
    production_head: str
    production_binary_sha256: Mapping
    tree_clean_at_freeze: bool
    rollback_branch: Optional[str] = None
    rollback_head: Optional[str] = None
    quality_baseline_binary: Optional[str] = None
    quality_baseline_sha256: Optional[str] = None
    quality_baseline_kernel: Optional[str] = None
    promotion_decision: Optional[bool] = None
    waiver_document: Optional[Mapping] = None
    waiver_sha256: Optional[str] = None
    #: WHERE the waiver lives, when the caller knows. With it, `calibration_request`
    #: READS the attestation through `waiver_binding_from_path` and the calibration's
    #: authority document is a file on disk. Without it, the calibration falls back to
    #: a quotation, which now yields COULD_NOT_CHECK and BLOCKS — a behaviour change
    #: for that caller and the correct one: the §10.4 calibration is the one check
    #: that says the compiler is right, and it must not rest on a synthetic path.
    waiver_path: Optional[str] = None
    #: The preserved ratification that HASHES the waiver, so the reader can require
    #: the bytes to be the ones the freeze record pins (v8's `evidence_sha256.waive_q8`).
    ratification_path: Optional[str] = None
    excluded_pairs: tuple = ()
    notes: tuple = ()

    def __post_init__(self) -> None:
        _text(self.label, "PreservedFreeze.label")
        if self.source_tree not in schemas.SOURCE_TREES:
            raise T3InputError(
                f"PreservedFreeze.source_tree: {self.source_tree!r} is not one of "
                f"{sorted(schemas.SOURCE_TREES)}")
        object.__setattr__(self, "backends", _str_tuple(
            self.backends, "PreservedFreeze.backends"))
        _text(self.production_branch, "PreservedFreeze.production_branch")
        _commit(self.production_head, "PreservedFreeze.production_head")
        if not isinstance(self.production_binary_sha256, Mapping):
            raise T3InputError("PreservedFreeze.production_binary_sha256: a mapping")
        _bool(self.tree_clean_at_freeze, "PreservedFreeze.tree_clean_at_freeze")
        object.__setattr__(self, "excluded_pairs", _str_tuple(
            self.excluded_pairs, "PreservedFreeze.excluded_pairs", non_empty=False))
        object.__setattr__(self, "notes", _str_tuple(
            self.notes, "PreservedFreeze.notes", non_empty=False))

    def to_dict(self) -> dict:
        return {"label": self.label, "source_tree": self.source_tree,
                "backends": list(self.backends),
                "production_branch": self.production_branch,
                "production_head": self.production_head,
                "production_binary_sha256": dict(self.production_binary_sha256),
                "tree_clean_at_freeze": self.tree_clean_at_freeze,
                "rollback_branch": self.rollback_branch,
                "rollback_head": self.rollback_head,
                "quality_baseline_binary": self.quality_baseline_binary,
                "quality_baseline_sha256": self.quality_baseline_sha256,
                "quality_baseline_kernel": self.quality_baseline_kernel,
                "promotion_decision": self.promotion_decision,
                "waiver_sha256": self.waiver_sha256,
                "waiver_path": self.waiver_path,
                "ratification_path": self.ratification_path,
                "excluded_pairs": list(self.excluded_pairs),
                "notes": list(self.notes)}


def preserved_freeze_from_v8_artifacts(ratification: Mapping,
                                       waiver: Optional[Mapping] = None,
                                       *, waiver_path: Optional[str] = None,
                                       ratification_path: Optional[str] = None
                                       ) -> PreservedFreeze:
    """Read `artifacts/operator/ratify_v8_final_freeze_20260725.json` (+ its waiver).

    This is a READER. It resolves fields out of a preserved attestation and refuses
    a document that is not one; it never edits, normalises, or upgrades the
    attestation, because a ratified operator record that has been rewritten to fit a
    newer schema is no longer the record that was ratified.

    `waiver_path` / `ratification_path` are where those two documents LIVE. Supplying
    them is what lets `calibration_request` build its waiver through
    `waiver_binding_from_path` instead of quoting one. `waiver` (the mapping) stays
    positional and stays supported so a caller with only a document can still read
    the freeze identity, but a calibration built from it alone now BLOCKS on an
    unread waiver rather than trusting it.
    """
    if not isinstance(ratification, Mapping):
        raise T3InputError("preserved_freeze_from_v8_artifacts: ratification mapping")
    schema = ratification.get("schema")
    if schema != "epyc.operator_v8_final_freeze_attestation.v1":
        raise T3InputError(
            f"preserved_freeze_from_v8_artifacts: schema {schema!r} is not "
            "'epyc.operator_v8_final_freeze_attestation.v1'")
    gate = ratification.get("production_lineup_gate") or {}
    contract = gate.get("quality_contract") if isinstance(gate, Mapping) else {}
    contract = contract if isinstance(contract, Mapping) else {}
    rollback = ratification.get("rollback") or {}
    binaries = ratification.get("production_binary_sha256") or {}
    excluded: tuple = ()
    waiver_sha = None
    if waiver is not None:
        scope = waiver.get("scope") or {}
        excluded = tuple(scope.get("excluded_pairs") or ())
        evidence = ratification.get("evidence_sha256") or {}
        waiver_sha = evidence.get("waive_q8")
    notes = [
        "promotion_decision is preserved as a NON-AUTOMATIC matrix verdict; the "
        "freeze was an operator-attested release decision "
        f"({ratification.get('promotion_decision_interpretation')})",
        f"q8_claim recorded on the attestation: {ratification.get('q8_claim')!r}",
    ]
    if not contract.get("baseline_binary"):
        notes.append("the attestation records no quality baseline binary")
    return PreservedFreeze(
        label="v8-final-freeze-20260725",
        source_tree="llama.cpp",
        backends=("llama_cpu", "llama_gpu"),
        production_branch=ratification.get("production_branch"),
        production_head=ratification.get("production_head"),
        production_binary_sha256={"llama_cpu": binaries.get("cpu"),
                                  "llama_gpu": binaries.get("hip")},
        # `freeze_v8_production_20260725.sh:194-195` gates on `git diff --quiet` for
        # both the worktree and the index, so the tree was clean at freeze time.
        tree_clean_at_freeze=True,
        rollback_branch=rollback.get("branch"),
        rollback_head=rollback.get("head"),
        quality_baseline_binary=contract.get("baseline_binary"),
        # Deliberately None: the preserved attestation hashes the quality RESULTS
        # but never the v7 binary they were produced against. That absence is the
        # §10.5 hole in artifact form, and inventing a digest here would conceal it.
        quality_baseline_sha256=None,
        quality_baseline_kernel=contract.get("baseline_kernel"),
        promotion_decision=ratification.get("promotion_decision"),
        waiver_document=waiver,
        waiver_sha256=waiver_sha,
        waiver_path=waiver_path,
        ratification_path=ratification_path,
        excluded_pairs=excluded,
        notes=tuple(notes),
    )


def preserved_freeze_from_speech_artifact(ratification: Mapping) -> dict:
    """Read `artifacts/operator/ratify_speech_kernel_freeze_20260731.json`.

    Returns one `PreservedFreeze` per source tree, because a freeze is per tree
    (§1.5) and the speech ratification covers two independent ones. Both carry
    `tree_clean_at_freeze=False`: CLAUDE.md records that *"both speech kernels carry
    load-bearing gfx90a/ROCm-6.2 patches that were UNCOMMITTED until this
    ratification"*, which is the single most consequential thing the dry run has to
    say about them.
    """
    if not isinstance(ratification, Mapping):
        raise T3InputError("preserved_freeze_from_speech_artifact: mapping required")
    if ratification.get("ratification") != "speech-kernel-freeze-v1":
        raise T3InputError(
            "preserved_freeze_from_speech_artifact: not the speech-kernel-freeze-v1 "
            "ratification")
    kernels = ratification.get("kernels") or {}
    out: dict = {}
    for key, source_tree, backend in (("whisper_cpp", "whisper.cpp", "whisper_stt"),
                                      ("qwentts_cpp", "qwentts.cpp", "qwentts_tts")):
        entry = kernels.get(key)
        if not isinstance(entry, Mapping):
            continue
        out[source_tree] = PreservedFreeze(
            label=f"speech-kernel-freeze-v1:{source_tree}",
            source_tree=source_tree,
            backends=(backend,),
            production_branch=entry.get("branch"),
            production_head=entry.get("commit"),
            production_binary_sha256={backend: entry.get("binary_sha256")},
            tree_clean_at_freeze=False,
            rollback_branch=None,
            rollback_head=None,
            notes=(
                f"load-bearing patch carried at freeze time: "
                f"{entry.get('load_bearing_patch')!r}",
                f"ggml generation {entry.get('ggml')!r} — three generations coexist, "
                "which is why per-launcher LD_LIBRARY_PATH is load-bearing",
                "this is the FIRST freeze of this tree: there is no incumbent to "
                "archive and no binary-level rollback anchor",
            ),
        )
    if not out:
        raise T3InputError(
            "preserved_freeze_from_speech_artifact: the ratification names no readable "
            "kernel entries")
    return out


#: A synthetic digest is still a digest, and the calibration harness needs several
#: for artifacts it is not reading. They are derived from a labelled string so a
#: reader can tell at a glance that the bytes behind them are the label, not an
#: artifact — and `is_placeholder_digest` still rejects the empty-input digest.
def _calibration_digest(label: str) -> str:
    return schemas.content_hash({"calibration_synthetic_input": label})


def calibration_request(freeze: PreservedFreeze, *, now: str,
                        include_waiver: bool = True,
                        campaign_id: str = "ak-calibration",
                        run_id: Optional[str] = None,
                        archive: Optional[IncumbentArchive] = None,
                        cooldown_seconds: int = 24 * 60 * 60) -> T3Request:
    """Build the §10.4 dry-run request for a preserved freeze.

    What is READ from the artifact: identity, heads, binary hashes, the rollback
    anchor, the quality baseline, the waiver and its scope, and whether the tree was
    clean. What is SYNTHESISED: the host/storage/claim receipts and the component
    digests, because a preserved attestation records none of them — and each carries
    a label saying so.

    The excluded pairs become gating cells with a FAIL check. That is not a guess:
    the v8 waiver's own scope names `qwen36_q8-tg128-iqk1` and
    `qwen36_q8-pp2048-iqk1` as pairs that *"cannot satisfy the ratified 72-core
    eligibility floor"*, and `promotion_decision: false` is the matrix verdict that
    followed. Without the waiver those cells have nothing covering them, which is
    exactly the FAIL §10.4 predicts.
    """
    if not isinstance(freeze, PreservedFreeze):
        raise T3InputError("calibration_request: freeze must be a PreservedFreeze")
    _timestamp(now, "calibration_request: now")
    run_id = run_id or f"akt3-calibration-{freeze.label}"
    backend = freeze.backends[0]
    protocols = {}
    cells: list = []
    results: list = []
    standings: list = []

    for name in freeze.backends:
        # A backend with a DECLARED phase vocabulary (`schemas.PHASES_BY_BACKEND`,
        # i.e. the llama pair) was graded under Annex B protocols that are ratified,
        # so the calibration replays it with a synthetic-but-bound ratification
        # receipt — labelled as synthetic in the document digest's preimage, exactly
        # like the host and storage receipts above. A backend with NO declared phase
        # vocabulary is a speech backend whose family is a draft (§13.3, §13.4); its
        # per-phase protocols are left UNBOUND, because the honest answer for those
        # cells is "we do not know what they were graded under" and synthesising a
        # ratification there would forge the very fact §10.4 is calibrating.
        phases_declared = bool(schemas.PHASES_BY_BACKEND.get(name))
        protocols[name] = {}
        for workload_phase, protocol_id, standing in (
                ("prefill", "P-BENCH-PREFILL-1", STANDING_IMPROVED),
                ("decode", "P-BENCH-1", STANDING_NON_INFERIOR)):
            protocols[name][workload_phase] = (
                ProtocolBinding(
                    protocol_id=protocol_id,
                    document_sha256=_calibration_digest(f"protocol:{protocol_id}"),
                    ratified=True, ratified_at=now, annex="B")
                if phases_declared else protocol_id)
            cell = Cell(
                cell_id=f"{name}.{workload_phase}.production_optimal",
                backend=name, release_phase=PHASE_PERFORMANCE_MATRIX,
                protocol_id=protocol_id, recipe_class=RECIPE_PRODUCTION_OPTIMAL,
                metric="tokens_per_s", metric_direction="higher_better",
                workload_phase=workload_phase,
                claim=f"{name} {workload_phase} non-regression vs "
                      f"{freeze.rollback_branch or 'no incumbent'}",
                co_resident=(name == "llama_cpu"),
                reps=10,
            )
            cells.append(cell)
            results.append(CellResult(
                cell=cell, check=schemas.Check(schemas.PASS),
                raw_samples_ref=f"preserved:{freeze.label}:{cell.cell_id}",
                reducer_id="median_mad/v1"))
            standings.append(PhaseStanding(
                backend=name, workload_phase=workload_phase, protocol_id=protocol_id,
                standing=standing, cell_ids=(cell.cell_id,),
                evidence_ref=f"preserved:{freeze.label}"))
        for phase_id in (PHASE_BACKEND_CORRECTNESS, PHASE_QUALITY, PHASE_STABILITY,
                         PHASE_CAPACITY_UTILITY):
            cell = Cell(
                cell_id=f"{name}.{phase_id}", backend=name, release_phase=phase_id,
                protocol_id="P-KERNEL-FREEZE-1", recipe_class=RECIPE_PRODUCTION_OPTIMAL,
                metric="pass_fail", metric_direction="higher_better",
                claim=f"{name} {phase_id} parity", reps=1)
            cells.append(cell)
            results.append(CellResult(
                cell=cell, check=schemas.Check(schemas.PASS),
                raw_samples_ref=f"preserved:{freeze.label}:{cell.cell_id}",
                reducer_id="gate/v1"))

    for pair in freeze.excluded_pairs:
        cell = Cell(
            cell_id=f"{backend}.pair.{pair}", backend=backend,
            release_phase=PHASE_PERFORMANCE_MATRIX,
            protocol_id="P-BENCH-PREFILL-1" if "pp" in pair else "P-BENCH-1",
            recipe_class=RECIPE_PRODUCTION_OPTIMAL, metric="tokens_per_s",
            metric_direction="higher_better",
            workload_phase="prefill" if "pp" in pair else "decode",
            claim=f"{pair} non-regression", model=pair.split("-")[0], reps=10)
        cells.append(cell)
        results.append(CellResult(
            cell=cell,
            check=_fail(
                f"{pair}: the pair does not satisfy the ratified eligibility floor, so "
                "no non-regression result exists for it"),
            raw_samples_ref=f"preserved:{freeze.label}:{pair}",
            reducer_id="median_mad/v1"))

    incumbent_version = _production_version_number(freeze.rollback_branch or "") or 0
    next_version = (_production_version_number(freeze.production_branch or "")
                    or incumbent_version + 1)
    plan = ReleasePlanView(
        plan_id=f"akplan-calibration-{freeze.label}",
        plan_sha256=_calibration_digest(f"plan:{freeze.label}"),
        source_tree=freeze.source_tree, backends=freeze.backends, cells=tuple(cells),
        incumbent_branch=freeze.rollback_branch or "(none)",
        incumbent_commit=freeze.rollback_head or ("0" * 40),
        incumbent_version_number=incumbent_version,
    )

    if archive is None:
        if freeze.rollback_head is None:
            archive = IncumbentArchive(no_incumbent_reason=(
                f"{freeze.label} is the FIRST freeze of {freeze.source_tree}: the "
                "attestation names no rollback branch or head, so there is no incumbent "
                "build to archive and no binary-level rollback target (§10.5)"))
        else:
            archive = IncumbentArchive(no_incumbent_reason=(
                f"{freeze.label} names the preserved baseline binary "
                f"{freeze.quality_baseline_binary!r} but records NO sha256 for it, so no "
                "ArchivedBuild can be constructed from the preserved artifacts. §10.5's "
                "requirement is unverifiable from the record as it stands."))

    transaction = TransactionPlan(
        next_branch=freeze.production_branch, next_version_number=next_version,
        next_tag=freeze.production_branch,
        install_path="/mnt/raid0/llm/kernels/production",
        symlink_diff=tuple(
            (f"/mnt/raid0/llm/kernels/production/{suffix}",
             f"(incumbent {freeze.rollback_branch or 'none'})",
             f"(candidate {freeze.production_branch})")
            for suffix in ("cpu", "gpu") if freeze.source_tree == "llama.cpp"
        ) or ((f"/mnt/raid0/llm/kernels/production/{freeze.backends[0][-3:]}",
               f"(incumbent {freeze.rollback_branch or 'none'})",
               f"(candidate {freeze.production_branch})"),),
        service_impact=("llama-server restart at the inference owner's boundary",),
        era_actions=({"draft": True, "action": "kernel_era_row",
                      "branch": freeze.production_branch},),
        receipt_paths=(f"artifacts/operator/{freeze.label}/",),
        rollback_branch=freeze.rollback_branch,
        rollback_head=freeze.rollback_head,
    )

    # --- the §10.4 authority document ---------------------------------------
    # This was the single worst construction site in the tree: the calibration is the
    # one check that says the compiler is right, and its waiver was a caller-supplied
    # document, pinned to a digest asserted equal to itself, at a SYNTHETIC path
    # (`artifacts/operator/<label>/waiver.json`) that has never existed on this host.
    # With `freeze.waiver_path` the document is READ, and the read is additionally
    # cross-checked against the digest the preserved ratification pins for it — the
    # only authenticity fact available anywhere in this system.
    waivers: tuple = ()
    if include_waiver and freeze.waiver_path and freeze.waiver_sha256:
        covered = tuple(f"{backend}.pair.{p}" for p in freeze.excluded_pairs)
        pin = ((freeze.ratification_path, "waive_q8")
               if freeze.ratification_path else None)
        waivers = (waiver_binding_from_path(
            freeze.waiver_path, pinned_sha256=freeze.waiver_sha256,
            # The genuine v8 record carries NO `waiver_id` key — its keys are exactly
            # candidate_head, consequences, decision, production_head, protocol,
            # protocol_changed, ratified_at, reason,
            # runner_sha256_before_waiver_implementation, schema, scope — so `decision`
            # ("WAIVE-Q8") is the identifier the operator actually wrote.
            waiver_id=str(freeze.waiver_document.get("decision") or "waiver")
            if isinstance(freeze.waiver_document, Mapping) else "waiver",
            covers_cell_ids=covered, ratification_pin=pin),)
    elif include_waiver and freeze.waiver_document is not None and freeze.waiver_sha256:
        # No path: the honest shape is a QUOTATION, which `verify_waiver` answers
        # COULD_NOT_CHECK and `phase_identity_preflight` turns into a blocking reason.
        # Deliberately not silently dropped — the calibration must say that a waiver
        # was cited and that nobody read it, not pretend none was offered.
        waivers = (WaiverBinding(
            waiver_id=str(freeze.waiver_document.get("decision") or "waiver"),
            pinned_sha256=freeze.waiver_sha256,
            observed_sha256=freeze.waiver_sha256,
            document=freeze.waiver_document,
            document_path=f"artifacts/operator/{freeze.label}/waiver.json",
            covers_cell_ids=tuple(f"{backend}.pair.{p}" for p in freeze.excluded_pairs),
        ),)

    quality = tuple(
        QualityEvidence(
            backend=name, mode=QUALITY_MEASURED_PARITY,
            baseline_binary_path=freeze.quality_baseline_binary or "(unrecorded)",
            baseline_binary_sha256=(freeze.quality_baseline_sha256
                                    or _calibration_digest(f"baseline:{freeze.label}")),
            baseline_kernel=freeze.quality_baseline_kernel or "(unrecorded)",
            baseline_is_rebuild=False,
            evidence_refs=(f"preserved:{freeze.label}:quality",),
            suites=("mmlu_pro", "gpqa"), shared_question_identity=True)
        for name in freeze.backends)

    stability = tuple(
        StabilityEvidence(
            backend=name, load_unload_cycles=3, memory_growth_bytes=0,
            memory_growth_allowance_bytes=0, profiler_or_runtime_errors=0,
            cleanup_verified=True, mixed_prefill_decode_exercised=True,
            evidence_ref=f"preserved:{freeze.label}:stability")
        for name in freeze.backends)

    host = guards.HostHealth(
        uptime_seconds=3600, observed_at=now,
        receipt=f"calibration-synthetic:{freeze.label}:host")
    storage_state = storage.StorageState(
        state=storage.STORAGE_OK, free_bytes=200 * 1024 ** 3,
        total_bytes=3700 * 1024 ** 3, floor_bytes=50 * 1024 ** 3)
    storage_obs = guards.StorageObservation(
        path="/mnt/raid0", state=storage_state, expirable_backlog_bytes=0,
        receipt=f"calibration-synthetic:{freeze.label}:storage")
    claims = tuple(
        guards.ResourceClaimObservation(
            resource=name, claim_kind="gpu_device" if name.endswith("gpu") else
            "cpu_region", acquired=True, observed_at=now,
            receipt=f"calibration-synthetic:{freeze.label}:{name}",
            held_by=run_id)
        for name in freeze.backends)

    return T3Request(
        run_id=run_id, campaign_id=campaign_id, mode=MODE_DRY_RUN, now=now,
        protocol=ProtocolBinding(
            protocol_id=RELEASE_PROTOCOL_ID,
            document_sha256=_calibration_digest("release-protocol-draft"),
            ratified=False),
        sealed=SealedCandidate(
            candidate_id=f"akc-calibration-{freeze.label}",
            source_tree=freeze.source_tree,
            candidate_branch=f"{freeze.source_tree}-experimental/calibration",
            production_base_commit=freeze.rollback_head or ("0" * 39 + "1"),
            candidate_commit=freeze.production_head,
            seal_sha256=_calibration_digest(f"seal:{freeze.label}"),
            evaluator_bundle_sha256=_calibration_digest("evaluator-bundle"),
            scope_manifest_sha256=_calibration_digest(f"scope:{freeze.label}"),
            evidence_tree_sha256=_calibration_digest(f"evidence:{freeze.label}"),
            binary_sha256={k: v for k, v in freeze.production_binary_sha256.items() if v},
            linkage_sha256={k: _calibration_digest(f"linkage:{freeze.label}:{k}")
                            for k in freeze.backends},
            build_dirs={k: f"/mnt/raid0/llm/{freeze.source_tree}-experimental/build"
                        for k in freeze.backends},
            overlay_present=True,
            tree_clean=freeze.tree_clean_at_freeze,
            ancestry_clean=True),
        plan=plan,
        backend_unchanged={
            name: UnchangedView(backend=name, may_drop_cells=False,
                                unchanged_outcome=schemas.FAIL,
                                agreement_outcome=schemas.PASS, stage2_ran=True,
                                reasons=("the candidate changed this backend's source "
                                         "closure, so it owes full evidence",))
            for name in freeze.backends},
        host=host, host_owner="operator",
        host_escalation_deadline=_plus_hours(now, 24),
        resource_claims=claims, storage_observation=storage_obs,
        transaction=transaction, archive=archive,
        supplied_components={
            name: _calibration_digest(f"{name}:{freeze.label}")
            for name in SUPPLIED_COMPONENTS},
        cooldown_seconds=cooldown_seconds,
        release_reps_by_protocol={"P-BENCH-1": 10, "P-BENCH-PREFILL-1": 5,
                                  "P-KERNEL-FREEZE-1": 1},
        phase_protocols=protocols,
        linkage_receipts=tuple(
            LinkageReceipt(
                backend=name,
                binary_path=f"/mnt/raid0/llm/{freeze.source_tree}-experimental/build/bin",
                expected_tree_root=(
                    f"/mnt/raid0/llm/{freeze.source_tree}-experimental/build"),
                verifier_path=(
                    f"/mnt/raid0/llm/epyc-inference-research/{LINKAGE_VERIFIER_RELPATH}"),
                verifier_sha256=_calibration_digest("verify_ggml_linkage.sh"),
                exit_code=0,
                stdout=("PASS: all linked ggml libraries resolve inside "
                        f"/mnt/raid0/llm/{freeze.source_tree}-experimental/build"),
                ld_library_path=(
                    f"/mnt/raid0/llm/{freeze.source_tree}-experimental/build/bin",
                    "/opt/rocm/lib"),
                observed_at=now)
            for name in freeze.backends),
        backend_inventories=tuple(
            BackendInventory(
                backend=name, entries=("CPU",) + (("HIP",) if name.endswith("gpu") else ()),
                device_entries=(("AMD Instinct MI210",) if name.endswith("gpu") else ()),
                source_ref=f"preserved:{freeze.label}:startup-log")
            for name in freeze.backends),
        determinism=tuple(
            DeterminismDeclaration(
                backend=name, anchor_class="bitwise_stable",
                candidate_class="bitwise_stable",
                evidence_ref=f"preserved:{freeze.label}:determinism")
            for name in freeze.backends),
        cell_results=tuple(results), standings=tuple(standings),
        capacity_floors=(), quality_evidence=quality, stability_evidence=stability,
        stability_min_cycles=3, waivers=waivers,
        complexity={
            name: integrity.ComplexityAssessment(
                requires_human_code_review=False, reasons=(), first_page_notice=None,
                measured={"source": f"preserved:{freeze.label}"})
            for name in freeze.backends},
        campaign_start_at=now,
    )


def _plus_hours(now: str, hours: int) -> str:
    moment = _timestamp(now, "now")
    moment = moment - timedelta(microseconds=moment.microsecond) + timedelta(hours=hours)
    # `strftime` rather than `isoformat().replace("+00:00", "Z")`, for the reason
    # given in `_timestamp`; `astimezone` is correct here because `_timestamp`
    # always returns an aware datetime.
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Self-audits — properties proven from this module's own source
# =============================================================================

#: Kept byte-compatible in spirit with `evaluator.api._FORBIDDEN_IMPORTS`, plus the
#: three the release plane additionally must not reach for. `pathlib` is
#: deliberately ABSENT from both: reading is not writing, and this audit's subject
#: is the ability to MUTATE the host, not the ability to look at a file.
_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "tempfile", "sqlite3", "urllib", "http", "requests", "pty", "fcntl", "resource",
    "shlex", "asyncio", "threading", "time", "io",
})
#: `open` is forbidden even for reading: it is the one call that takes a mode, so
#: an audit that allowed it would be allowing `open(path, "w")` by omission. The
#: module reads its own source through `Path.read_text`, which cannot write.
_FORBIDDEN_CALL_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__",
                                   "input"})
#: Attribute calls. `pathlib` is allowed for READING, so every pathlib method that
#: mutates has to be named here or the allowance is a hole: `Path(p).open("w")` is
#: the same call as `open(p, "w")` written as an attribute, `.write()` is what one
#: does with the handle, `Path(new).replace(link)` MOVES A STABLE KERNEL SYMLINK,
#: and `hardlink_to` is `symlink_to` under a different name. The earlier list
#: forbade `open` only as a bare Name and `symlink_to` only in its `sym` spelling,
#: so all four passed — an audit that misses the four most direct routes to a
#: production write is a docstring with an AST parser attached.
_FORBIDDEN_CALL_ATTRS = frozenset({
    "write_text", "write_bytes", "write", "writelines", "truncate", "flush",
    "open", "mkdir", "makedirs", "unlink", "rmtree", "rmdir", "remove", "rename",
    "renames", "replace", "chmod", "chown", "symlink", "symlink_to", "link",
    "link_to", "hardlink_to", "touch", "copy", "copy2", "copyfile", "copytree",
    "move", "system", "popen", "run", "call", "check_call", "check_output",
    "Popen", "spawn", "fork", "kill", "killpg", "terminate", "send_signal",
    "sleep",
})
#: `getattr(obj, "write_text")()` is an `ast.Call` whose `func` is itself a Call,
#: so neither list above sees it. T3 uses `getattr` legitimately and constantly for
#: its duck-typed plan adapters, so the idiom is not forbidden — only a getattr
#: whose attribute name is a CONSTANT drawn from the denied sets is, which no
#: compliant call site in this module is (`getattr(cell, "protocol", None)`).


def audit_no_write_or_process_paths(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's own AST that T3 cannot write, spawn, or signal.

    Invariant 5 and the cardinal rule of the release plane are prose until something
    checks them. This parses the module and FAILs on a write-capable call, a process
    call, or an import that would grant either. `test_t3.py` asserts PASS, which
    turns the property into a regression barrier rather than an intention.

    COULD_NOT_CHECK when the source cannot be read or parsed: an unreadable module
    is not an audited one.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return _cnc(f"could not read {__file__}: {exc}")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _cnc(f"could not parse module: {exc}")

    findings: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root in _FORBIDDEN_IMPORTS:
                findings.append(f"line {node.lineno}: imports from {node.module!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
                findings.append(f"line {node.lineno}: calls {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CALL_ATTRS:
                findings.append(f"line {node.lineno}: calls .{func.attr}()")
            elif isinstance(func, ast.Name) and func.id in ("getattr", "setattr") \
                    and len(node.args) >= 2:
                named = node.args[1]
                if isinstance(named, ast.Constant) and isinstance(named.value, str) \
                        and (named.value in _FORBIDDEN_CALL_ATTRS
                             or named.value in _FORBIDDEN_CALL_NAMES):
                    findings.append(
                        f"line {node.lineno}: reaches {named.value!r} through "
                        f"{func.id}(), which routes around the attribute denylist")
    if findings:
        return _fail(*findings)
    return schemas.Check(schemas.PASS)


#: Every way this module could read a file. `waiver_binding_from_path` is allowed
#: exactly ONE of them, once; `human_only_boundary` reads the manifest with
#: `read_text` and `audit_no_write_or_process_paths` reads this file's own source.
_FILE_READ_ATTRS = frozenset({"read_bytes", "read_text", "readlines", "readline",
                              "read"})

#: The functions permitted to read from the filesystem AT ALL, and the read attribute
#: each is permitted. Anything else that reads is a second, unhardened door onto the
#: same evidence.
_PERMITTED_READERS = {
    "human_only_boundary": {"read_text"},
    "audit_no_write_or_process_paths": {"read_text"},
    "audit_waiver_reader_is_the_only_reader": {"read_text"},
    "audit_reader_narrowing_is_never_widened": {"read_text"},
    "audit_waiver_binding_is_constructed_only_by_the_reader": {"read_text"},
    "audit_backend_readiness_is_consulted": {"read_text"},
    "_read_operator_file": {"read_bytes"},
}


def audit_waiver_reader_is_the_only_reader(source: Optional[str] = None
                                           ) -> schemas.Check:
    """The §10.4 reader's discipline, enforced mechanically rather than by review.

    Three properties, each of which a future edit would otherwise silently undo:

      1. `_read_operator_file` performs EXACTLY ONE read call. Two reads is two
         different sets of bytes, and every guarantee in the reader is phrased over
         "the bytes that were hashed".
      2. No other function in this module reads a file except the three that
         legitimately do (the boundary manifest and the two source audits). A
         `Path(document_path).read_text()` added anywhere else re-opens the defect
         from a new direction.
      3. `_READER_TOKEN` is named in exactly THREE places — where the receipt's own
         constructor checks it, where the reader spends it, and where
         `waiver_read_violations` re-checks it at the gate — so "forge a read
         receipt" means writing the token's name into source, which is greppable,
         rather than passing a flag or inheriting from `ReadWaiver`.

    COULD_NOT_CHECK when the source cannot be read or parsed: an unreadable module is
    not an audited one.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return _cnc(f"could not read {__file__}: {exc}")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _cnc(f"could not parse module: {exc}")

    findings: list = []
    reads_by_function: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) \
                    and inner.func.attr in _FILE_READ_ATTRS:
                reads_by_function.setdefault(node.name, []).append(
                    (inner.lineno, inner.func.attr))

    for name, reads in sorted(reads_by_function.items()):
        permitted = _PERMITTED_READERS.get(name)
        if permitted is None:
            findings.append(
                f"{name}() reads the filesystem at line(s) "
                f"{[line for line, _ in reads]}; only {sorted(_PERMITTED_READERS)} may")
            continue
        for line, attr in reads:
            if attr not in permitted:
                findings.append(f"line {line}: {name}() calls .{attr}(), "
                                f"only {sorted(permitted)} is permitted there")
    operator_reads = reads_by_function.get("_read_operator_file", [])
    if len(operator_reads) != 1:
        findings.append(
            f"_read_operator_file() performs {len(operator_reads)} read calls; the "
            "single-bytes-object guarantee requires exactly one")

    # Loads only: the module-level `_READER_TOKEN = object()` is a Store and is the
    # mint itself, not a use of it.
    token_mentions = [n.lineno for n in ast.walk(tree)
                      if isinstance(n, ast.Name) and n.id == "_READER_TOKEN"
                      and isinstance(n.ctx, ast.Load)]
    if len(token_mentions) != 3:
        findings.append(
            f"_READER_TOKEN is named at lines {token_mentions}; it must be named "
            "exactly three times — checked in WaiverReadReceipt.__post_init__, spent "
            "in waiver_binding_from_path, and re-checked in waiver_read_violations "
            "(the gate must not take a constructor's word for a capability)")

    if findings:
        return _fail(*findings)
    return schemas.Check(schemas.PASS)


#: The reader's guards that are exposed as keyword arguments, and therefore the ones
#: a CALLER could relax. Each may only narrow at runtime; this audit additionally
#: proves no shipping module reaches for them at all.
_READER_NARROWING_KWARGS = ("attestation_roots", "max_bytes", "boundary")


def audit_reader_narrowing_is_never_widened(package_root: Optional[Any] = None
                                            ) -> schemas.Check:
    """No SHIPPING module relaxes `waiver_binding_from_path`'s guards.

    `attestation_roots`, `max_bytes` and `boundary` exist so a test can point the
    reader at a fixture root and so a caller can be STRICTER than the default. Each
    is now runtime-clamped — a declared root must itself pass
    `operator_owned_path_check`, `max_bytes` may not exceed
    `schemas.MAX_OPERATOR_WAIVER_BYTES` — but a clamp bounds how far a caller can go,
    not whether it goes. The strongest statement available about the SHIPPING gate is
    that no module outside the test files touches them, and that is a static fact,
    checkable over every future edit rather than over the call sites that exist today.

    Scans every non-test `.py` in the `autokernel` package. COULD_NOT_CHECK when a
    module cannot be read or parsed: an unaudited module is not an audited one.
    """
    root = Path(package_root) if package_root is not None \
        else Path(__file__).resolve().parents[1]
    findings: list = []
    scanned = 0
    for module in sorted(root.rglob("*.py")):
        if module.name.startswith("test_") or module.name == "conftest.py":
            continue
        try:
            tree = ast.parse(module.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            return _cnc(f"could not audit {module}: {exc}")
        scanned += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else \
                getattr(func, "id", None)
            if name != "waiver_binding_from_path":
                continue
            for kw in node.keywords:
                if kw.arg in _READER_NARROWING_KWARGS:
                    findings.append(
                        f"{module.name}:{node.lineno}: passes {kw.arg}= to "
                        "waiver_binding_from_path. That argument relaxes a guard the "
                        "party being gated should not be able to reach; the shipping "
                        "gate uses the defaults.")
    if not scanned:
        return _cnc(f"no modules found under {root}")
    if findings:
        return _fail(*findings)
    return schemas.Check(schemas.PASS)


#: THE CONSTRUCTOR ALLOWLIST — every place in the SHIPPING package where one of the
#: three waiver types may be built, keyed by `(package-relative module, enclosing
#: function)`. Anything not listed here is a finding.
#:
#: This exists because the §10.4 fix is a DISCIPLINE, not a patch. Migrating the call
#: sites that existed on 2026-08-03 fixed 2026-08-03; the defect returns the first
#: time somebody writes `WaiverBinding(document=..., observed_sha256=...)` in a new
#: module, and that edit looks entirely reasonable in review — it is a dataclass with
#: public fields being used as one. Nothing about the type's shape stops it, so the
#: stop has to be a test.
#:
#: Keyed by function, never by line number: a line-numbered allowlist is invalidated
#: by every unrelated edit above it, and an allowlist people routinely have to
#: re-bless stops being read. Keyed by EXACT function name, never a module prefix,
#: for the same reason `_RULE_ONE_EXEMPT_MODULES` uses exact paths — a
#: module-shaped entry would silently cover the next construction added beside it.
_WAIVER_CONSTRUCTOR_SITES = {
    "ReadWaiver": {
        ("release/t3.py", "waiver_binding_from_path"):
            "the reader itself. A `ReadWaiver` built anywhere else is a second, "
            "unhardened door onto the same authority: it would carry a receipt this "
            "module did not mint beside the bytes it attests to.",
    },
    "WaiverReadReceipt": {
        ("release/t3.py", "waiver_binding_from_path"):
            "minted beside the single `bytes` object it digests. A receipt built "
            "anywhere else asserts a read that some other code performed, which is "
            "the three-independent-assertions defect one level down.",
    },
    "WaiverBinding": {
        ("release/t3.py", "calibration_request"):
            "the deliberate QUOTATION fallback: a preserved freeze with a waiver "
            "DOCUMENT but no waiver PATH must still say that a waiver was cited and "
            "that nobody opened it. It suppresses nothing and blocks the run, which "
            "is the whole reason 'unread' stayed expressible.",
    },
}

#: Names the scanned corpus MUST define for this audit's answer to be ABOUT the
#: waiver types. Without it, "no unlisted construction found" is satisfied perfectly
#: by a root containing no waiver code at all — the same failure mode
#: `_READINESS_AUDIT_IDENTITY` exists to stop, and the one that would make this
#: guard rot silently the day somebody moves `t3.py`.
_WAIVER_CONSTRUCTOR_AUDIT_IDENTITY = ("WaiverBinding", "ReadWaiver",
                                      "WaiverReadReceipt")


def _enclosing_function_names(tree: ast.AST) -> dict:
    """`{id(Call node): nearest enclosing function name}` for one module AST.

    Nearest, not any: a helper nested inside an allowlisted function is its own
    function and gets its own entry, so the allowlist cannot be inherited by
    anything defined underneath it.
    """
    out: dict = {}

    def walk(node, current):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, child.name)
                continue
            if isinstance(child, ast.Call):
                out[id(child)] = current
            walk(child, current)

    walk(tree, "<module>")
    return out


def audit_waiver_binding_is_constructed_only_by_the_reader(
        package_root: Optional[Any] = None) -> schemas.Check:
    """No shipping module builds a waiver binding outside `_WAIVER_CONSTRUCTOR_SITES`.

    §10.4 turns a FAIL into PASS_WITH_WAIVER, so `WaiverBinding` is the authority
    type of the whole freeze gate — and it is an ordinary frozen dataclass with
    public fields. `waiver_binding_from_path` is only T3's *only* trusted constructor
    for as long as nobody writes the obvious thing somewhere else, and the obvious
    thing is one line that reads like normal code.

    AST, never grep. A textual scan cannot tell a construction from the word
    appearing in a docstring, an `isinstance` check or a comment — this module's own
    `WaiverBinding` docstring quotes the defective call verbatim, so a grep-based
    guard would either fail on its own documentation or be weakened until it stopped
    biting. Two things are checked:

      1. Every CALL to `WaiverBinding`, `ReadWaiver` or `WaiverReadReceipt` (bare or
         attribute-qualified) sits at an allowlisted `(module, function)`.
      2. No module ALIASES one of those names (`_WB = t3.WaiverBinding`), which would
         otherwise walk straight past check 1 while constructing exactly the same
         object. `isinstance(x, WaiverBinding)` and `_typed_tuple(..., WaiverBinding)`
         are untouched: passing the class as a value is not rebinding it to a name.

    Test modules are OUT of scope on purpose. Tests must be able to build a quotation
    — proving that a quotation suppresses nothing is most of the evidence that this
    fix works, and a guard that forbade it would forbid its own compliant path.

    COULD_NOT_CHECK, never PASS, when a module cannot be read or parsed, when the
    root holds no modules, or when the corpus does not define the three types.
    """
    root = Path(package_root) if package_root is not None \
        else Path(__file__).resolve().parents[1]
    guarded = set(_WAIVER_CONSTRUCTOR_SITES)
    findings: list = []
    defined: set = set()
    scanned = 0
    for module in sorted(root.rglob("*.py")):
        if module.name.startswith("test_") or module.name == "conftest.py":
            continue
        if "__pycache__" in module.parts:
            continue
        try:
            tree = ast.parse(module.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            return _cnc(f"could not audit {module}: {exc}")
        scanned += 1
        try:
            rel = module.relative_to(root).as_posix()
        except ValueError:                                    # pragma: no cover
            rel = module.name
        enclosing = _enclosing_function_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in guarded:
                defined.add(node.name)
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else \
                    getattr(func, "id", None)
                if name not in guarded:
                    continue
                where = enclosing.get(id(node), "<module>")
                allowed = _WAIVER_CONSTRUCTOR_SITES[name]
                if (rel, where) in allowed:
                    continue
                findings.append(
                    f"{rel}:{node.lineno}: {where}() constructs {name}(). The only "
                    f"permitted site(s) are "
                    f"{sorted(f'{m}::{fn}' for m, fn in allowed)}. Build it with "
                    "waiver_binding_from_path(), which READS the document from an "
                    "operator-owned path; a hand-built binding's document, path and "
                    "digest are three assertions by the party being gated (§10.4).")
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                alias = value.attr if isinstance(value, ast.Attribute) else (
                    value.id if isinstance(value, ast.Name) else None)
                if alias in guarded:
                    findings.append(
                        f"{rel}:{node.lineno}: aliases {alias} to another name. An "
                        "alias constructs the same object while naming something "
                        "this audit does not look for.")
    if not scanned:
        return _cnc(f"no modules found under {root}")
    missing = [n for n in _WAIVER_CONSTRUCTOR_AUDIT_IDENTITY if n not in defined]
    if missing:
        return _cnc(
            f"the corpus under {root} defines no {missing}; an audit of modules that "
            "do not contain the waiver types is not a passing audit of them")
    if findings:
        return _fail(*findings)
    return schemas.Check(schemas.PASS)


def audit_phase_coverage_totality() -> schemas.Check:
    """Every §10.2 phase has a runner, and every runner names a declared phase.

    A tenth phase nobody runs and a runner for a phase §10.2 does not declare are
    the same defect seen from two sides, and both are silent until the release that
    needed the missing one.
    """
    runners = {
        PHASE_IDENTITY_PREFLIGHT: phase_identity_preflight,
        PHASE_BUILD_LINKAGE: phase_build_linkage,
        PHASE_BACKEND_CORRECTNESS: phase_backend_correctness,
        PHASE_PERFORMANCE_MATRIX: phase_performance_matrix,
        PHASE_QUALITY: phase_quality,
        PHASE_STABILITY: phase_stability,
        PHASE_CAPACITY_UTILITY: phase_capacity_utility,
        PHASE_TRANSACTION_DRY_RUN: phase_transaction_dry_run,
        PHASE_SEAL: phase_seal,
    }
    reasons: list = []
    missing = [p for p in PHASES if p not in runners]
    extra = [p for p in runners if p not in PHASES]
    if missing:
        reasons.append(f"§10.2 phases with no runner: {missing}")
    if extra:
        reasons.append(f"runners for phases §10.2 does not declare: {extra}")
    if len(PHASES) != 9:
        reasons.append(f"§10.2 declares nine phases; PHASES has {len(PHASES)}")
    unknown_cell_phases = [p for p in CELL_PHASES if p not in PHASES]
    if unknown_cell_phases:
        reasons.append(f"CELL_PHASES names non-phases: {unknown_cell_phases}")
    return _fail(*reasons) if reasons else schemas.Check(schemas.PASS)


#: Names the audited source MUST bind for the result to be ABOUT this module. Same
#: device the adapters use: without an identity binding an AST audit is a property
#: of whatever string it was handed, and the empty string satisfies "contains no
#: defect" perfectly. Missing any of these is COULD_NOT_CHECK, never PASS.
_READINESS_AUDIT_IDENTITY = (
    "RELEASE_READINESS_BY_BACKEND", "declared_ratified_protocol_ids",
    "phase_identity_preflight",
)


def audit_backend_readiness_is_consulted(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's AST that phase 1 still ASKS the adapters.

    `RELEASE_READINESS_BY_BACKEND` closes a gap whose whole shape was *"the adapters
    knew and nothing called them"*. A registry is only worth as much as its call
    site, and a deleted call site is invisible: every existing test would still pass,
    because a release that consults nothing looks exactly like a release everything
    approved of. So the call is a checked property, not a convention.

    FAIL when `phase_identity_preflight` does not call a value taken out of the
    registry. COULD_NOT_CHECK — never PASS — when the source cannot be read or
    parsed, or when it does not bind the names above: an audit of a module that does
    not contain the mechanism is not a passing audit of the mechanism.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return _cnc(f"could not read {__file__}: {exc}")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _cnc(f"could not parse module: {exc}")

    bound: set = set()
    preflight = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            bound.add(node.name)
            if node.name == "phase_identity_preflight":
                preflight = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    absent = [name for name in _READINESS_AUDIT_IDENTITY if name not in bound]
    if absent:
        return _cnc(
            f"the audited source does not bind {absent}, so it is not this module and "
            "the result would be a property of whatever was handed in. An audit that "
            "an empty string can pass is not an audit.")
    if preflight is None:  # pragma: no cover - unreachable while the identity holds
        return _cnc("the audited source binds no `phase_identity_preflight` function")

    # Names inside phase 1 that were taken OUT of the registry, and then the proof
    # that one of them is actually called. Both halves matter: reading the registry
    # into a variable and never calling it is precisely the pre-AK5 state.
    taken: set = set()
    for node in ast.walk(preflight):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                    and func.value.id == "RELEASE_READINESS_BY_BACKEND":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        taken.add(target.id)
    if not taken:
        return _fail(
            "phase_identity_preflight does not read RELEASE_READINESS_BY_BACKEND. The "
            "adapters expose release_gate_readiness() and the release plane is back to "
            "never calling it, which is the AK5 defect this registry closed.")
    called = {node.func.id for node in ast.walk(preflight)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    if not (taken & called):
        return _fail(
            f"phase_identity_preflight reads RELEASE_READINESS_BY_BACKEND into {sorted(taken)} "
            "and never calls it. A readiness predicate that is looked up and not invoked "
            "is the same silence as not looking it up at all.")

    # ...and the third half, which the first two do not imply. Reading the registry
    # and calling what it returns still permits `readiness = readiness_of(ids)` with
    # the verdict then dropped on the floor — which is INDISTINGUISHABLE, in every
    # artifact this gate emits, from a release the adapters approved of. That is the
    # same shape as the pre-AK5 state, one step further in. So the verdict must be
    # shown to reach `blocking`.
    verdicts = {target.id
                for node in ast.walk(preflight)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id in taken
                for target in node.targets if isinstance(target, ast.Name)}
    if not verdicts:
        return _fail(
            "phase_identity_preflight calls the readiness predicate without binding its "
            "result, so the verdict cannot be adjudicated. A check whose answer is "
            "discarded is a check that was not made.")
    reached = _names_reaching_blocking(preflight)
    if not (verdicts & reached):
        return _fail(
            f"phase_identity_preflight computes the readiness verdict into "
            f"{sorted(verdicts)} and nothing it computes reaches `blocking`. A "
            "COULD_NOT_CHECK from an adapter that is never appended is a release the "
            "gate consulted and then ignored, which reads in every emitted artifact "
            "exactly like a release the adapters approved of.")
    return schemas.Check(schemas.PASS)


#: The two ways this gate turns a finding into a refusal. Both are accepted, because
#: an audit that recognised only the `if`-then-append shape would FAIL the moment
#: somebody factored the reasons out into a helper — a guard that forbids a
#: legitimate rewrite of the thing it is guarding.
_BLOCKING_SINKS = ("append", "extend")


def _blocking_calls(node: ast.AST) -> list:
    """Every `blocking.append(...)` / `blocking.extend(...)` under `node`."""
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in _BLOCKING_SINKS
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "blocking"]


def _names_reaching_blocking(func: ast.AST) -> set:
    """Names whose value demonstrably reaches `blocking` inside `func`.

    Two routes, and no attempt at a third: a name **read inside the arguments of** a
    `blocking.append/extend` call, and a name **tested by an `if` whose body**
    appends to `blocking`. Anything subtler than that is dataflow analysis, and an
    audit that guesses is worse than one that says what it recognises.
    """
    reaching: set = set()
    for call in _blocking_calls(func):
        for arg in list(call.args) + [kw.value for kw in call.keywords]:
            reaching.update(n.id for n in ast.walk(arg) if isinstance(n, ast.Name))
    for node in ast.walk(func):
        if isinstance(node, ast.If) and any(
                _blocking_calls(stmt) for stmt in node.body):
            reaching.update(n.id for n in ast.walk(node.test) if isinstance(n, ast.Name))
    return reaching
