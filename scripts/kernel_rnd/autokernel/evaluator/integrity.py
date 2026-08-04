#!/usr/bin/env python3
"""integrity.py — the §8.5.1 source-integrity gates, run BEFORE any behavioural check.

WHY THIS MODULE EXISTS
----------------------
AutoPilot's one venture into autonomous source mutation destroyed
`src/escalation.py` — 454 lines reduced to 3 — with an edit that passed
`ast.parse()` cleanly. The project's answer was four Python-specific defenses:
syntax validity, a >60% shrinkage reject, public-name preservation, and a live
`importlib` round-trip.

**None of those four transfer to compiled C++/HIP.** There is no import
round-trip, and *"it compiles"* is a far weaker claim than *"it imports"*: a
kernel edit that drops a template specialization, deletes a case from a dispatch
switch, or removes an op registration compiles cleanly and silently changes
behaviour for every shape nobody happened to test. Design §8.5.1 is the C++
analogue, and this module is it:

  1. **Symbol and registration preservation** — diff the exported symbol table,
     the op-registration tables and the dispatch predicates between the ANCHOR
     binary and the CANDIDATE binary. Any removal or arity change not declared in
     `proposal.declared_symbol_deltas` is a hard failure. This is the direct
     analogue of public-name preservation and the check that would have caught
     the class of edit that destroyed `escalation.py`.
  2. **Clean build from the recorded snapshot** — the artifact under test came
     from the content-addressed snapshot in a FRESH build directory, never from
     the actor's incremental tree. An incremental build links stale objects and
     hides the error the snapshot would surface, which would make the actor's
     build state part of the artifact.
  3. **Semantic diff conformance** — the diff touches only the declared surface,
     contains no unrelated deletions, and stays inside the change-class size
     envelope. Invariant 13 says one conceptual mutation; this is what ENFORCES
     it rather than trusting it. The shrinkage ceiling here is the direct port of
     AutoPilot's >60% reject.
  4. **`core_header` risk tier** — a change to shared ggml core or to a widely
     included header is not a large edit, it is a *different kind* of edit,
     because its reach is every op in both the CPU and the GPU build. It forces
     full-tree affected surface, forces per-backend binary comparison (§3.2), and
     marks the candidate `REQUIRES_HUMAN_CODE_REVIEW` regardless of diff size.
     The tier is derived MECHANICALLY from the diff, never taken from the actor's
     `change_class` (§6.4, invariant 18: the actor's declaration is a scored
     prediction, never a scope input).
  5. **Repair from a clean parent** — a bounded repair re-checks out the parent
     and re-applies, never continuing on the failed attempt's tree. Repairs are
     capped per proposal; exceeding the cap is a `PLANNER_DEGRADED` signal, not
     another retry. AutoPilot's scar here was a loop compounding edits onto an
     already-corrupted file.

Plus the **§10.6 diff-complexity ceiling**: LLM-authored kernel C++/HIP must not
reach a release package unreviewed at arbitrary size. Above the backend
adapter's declared ceiling the candidate is marked `REQUIRES_HUMAN_CODE_REVIEW`
and the marker is carried on the receipt's first page.

GOVERNING INSTRUMENTS
---------------------
`epyc-root/measurement/protocols/kernel-research.md` (Annex K, **P-AK-SEARCH-1**,
RATIFIED 2026-08-03). The clauses this module implements, by section name:

  * *Correctness precedence* — every gate here is `gate_class=integrity`, one of
    the five lexicographically prior classes, so a failure ends speed ranking
    entirely rather than penalising it (`api.Verdict.rank_key()` raises).
  * *Preconditions (all enforced or attested per run)* — precondition 4, the
    explicit immutable anchor: every symbol/registration gate carries
    `requires_anchor=True`, so a PASS produced without a bound anchor is demoted
    to `COULD_NOT_CHECK` by `api.compute_verdict`. Absence of a comparison is not
    evidence of equivalence.
  * *What voids a run* — nothing here votes on a void; a source-integrity
    failure is a CANDIDATE failure, not a voided window, and the two are kept
    distinct because *"a drifted anchor says nothing whatever about the
    candidate"* and the converse holds too.
  * *No self-amendment* — *"a controller that discovers a coverage gap in its
    evaluator RECORDS the gap … it does not patch the instrument."* Every
    coverage limit below (symbol versions, unmangled arity, unresolvable
    shrinkage) is reported as `COULD_NOT_CHECK` with the missing input NAMED,
    never as a PASS.

Owning design: `handoffs/active/autokernel-research-loop.md` §6.4, §7.2, §8.5,
§8.5.1, §8.6, §10.6, phase AK3.

WHAT THIS MODULE IS NOT
-----------------------
It runs NO build, NO benchmark and NO inference. It starts, stops and signals NO
process. It WRITES no file — it reads them: symbol tables out of ELF binaries and
content digests out of source trees, both with a pure-stdlib reader, so nothing
here shells out to `nm`, `readelf` or `git` either.
`audit_no_write_or_process_paths()` proves those properties from this module's
own AST (it permits `Path.open` only with a literal read mode) and
`test_integrity.py` asserts the audit PASSes, so the property is a regression
barrier rather than an intention.

Fail-open is the failure mode this file is shaped against. Two guards deserve
naming, because both would otherwise read as a clean PASS:

  * an **empty anchor symbol table** (or an empty anchor registration table)
    cannot evidence preservation — a diff of nothing against nothing has no
    removals. Both return `COULD_NOT_CHECK`, never PASS.
  * a **registration extractor with no declared patterns** finds no entries
    because it looked for none. `PatternRegistrationExtractor` refuses to be
    constructed without patterns.
"""
from __future__ import annotations

import ast
import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .. import schemas
from . import api
from . import surface as ak_surface

__all__ = [
    # identity
    "DESIGN_SECTION", "GATE_IDS", "FINDING_CODES",
    # errors
    "IntegrityError", "ElfFormatError", "DiffParseError", "EnvelopeNotDeclared",
    "IntegrityInputsMissing", "DeclarationMissing", "TreeReadError",
    "GATE_EVIDENCE_BINDING", "RUNNER_GATE_IDS",
    # ELF + names
    "ElfSymbol", "ElfSymbolTable", "extract_elf_symbols", "exported_surface",
    "ParsedName", "parse_mangled_name",
    # symbols
    "DeclaredSymbolDeltas", "SignatureChange", "SymbolDiff", "diff_symbol_tables",
    "check_symbol_preservation", "check_symbol_arity_coverage",
    # registrations
    "RegistrationEntry", "RegistrationTable", "PatternRegistrationExtractor",
    "diff_registration_tables", "check_registration_preservation",
    "KIND_OP_REGISTRATION", "KIND_DISPATCH_PREDICATE",
    # snapshot + build
    "TreeDigest", "EMPTY_TREE_SHA256", "hash_source_tree", "sha256_file",
    "BuildProvenance", "CleanBuildReceipt", "check_clean_build_from_snapshot",
    # diff
    "FileDiff", "SourceDiff", "parse_unified_diff",
    "DeclaredSurface", "ChangeClassEnvelope", "ComplexityCeiling",
    "ComplexityAssessment", "assess_complexity_ceiling",
    "check_semantic_diff_conformance",
    # core_header
    "CoreHeaderPolicy", "RiskTierDecision", "assess_risk_tier",
    "REQUIRES_HUMAN_CODE_REVIEW", "SURFACE_FULL_TREE", "SURFACE_PARTIAL",
    # the seam into surface.py's DERIVED affected-surface manifest
    "surface_scope_for", "check_declared_surface_scope", "GATE_SURFACE_SCOPE_BINDING",
    # repair
    "PLANNER_DEGRADED", "RepairPolicy", "RepairAttempt", "RepairDecision",
    "RepairLedger", "check_repair_from_clean_parent",
    # orchestration
    "SourceIntegrityInputs", "SourceIntegrityReport", "run_source_integrity_gates",
    "SourceIntegrityGateRunner", "SourceIntegrityFirstRunner",
    "check_evidence_binding",
    # self-audit
    "audit_no_write_or_process_paths",
]

DESIGN_SECTION = "autokernel-research-loop.md §8.5.1 (+ §10.6 complexity ceiling)"

PASS = schemas.PASS
FAIL = schemas.FAIL
COULD_NOT_CHECK = schemas.COULD_NOT_CHECK


# =============================================================================
# Errors — every one of these is a DEFECT or a MISSING INPUT, never a finding
# about the candidate. Findings are `Check`s; defects raise.
# =============================================================================

class IntegrityError(api.EvaluatorError):
    """Base for source-integrity wiring and input errors."""


class ElfFormatError(IntegrityError):
    """The file is not an ELF this reader can extract a symbol table from.

    Raised, never degraded: a binary whose symbols cannot be read has not been
    shown to preserve them, and the caller must turn this into COULD_NOT_CHECK
    explicitly rather than inheriting an empty table that diffs clean.
    """


class TreeReadError(IntegrityError):
    """A source tree could not be hashed. An unreadable input is not an empty one."""


class DiffParseError(IntegrityError):
    """The unified diff is malformed. A diff we cannot parse is not a small diff."""


class EnvelopeNotDeclared(IntegrityError):
    """No change-class envelope or complexity ceiling was declared for this cell.

    §10.6: *"Each backend adapter declares a complexity/blast-radius ceiling."*
    There is deliberately no default envelope — a defaulted ceiling is a ceiling
    nobody chose, and it would silently admit whatever the default happened to
    allow.
    """


class DeclarationMissing(IntegrityError):
    """A proposal is missing a declaration this gate consumes.

    `declared_symbol_deltas` absent is NOT an empty declaration: it is an
    undeclared removal waiting to happen (§7.2).
    """


class IntegrityInputsMissing(api.EvaluatorNotWired):
    """No source-integrity inputs registered for this candidate.

    Subclasses `EvaluatorNotWired` for the same reason the dispatcher raises it:
    an unrun gate set with no results derives to PASS, which is a fail-open
    verdict.
    """


# =============================================================================
# Gate ids and finding codes — a closed vocabulary, so a journaled reason is
# greppable and a test can assert the exact code rather than prose.
# =============================================================================

GATE_SYMBOL_PRESERVATION = "integrity.symbol_preservation"
GATE_SYMBOL_ARITY_COVERAGE = "integrity.symbol_arity_coverage"
GATE_OP_REGISTRATION = "integrity.op_registration_preservation"
GATE_DISPATCH_PREDICATE = "integrity.dispatch_predicate_preservation"
GATE_CLEAN_BUILD = "integrity.clean_build_from_snapshot"
GATE_SEMANTIC_DIFF = "integrity.semantic_diff_conformance"
GATE_CORE_HEADER = "integrity.core_header_risk_tier"
GATE_REPAIR_CLEAN_PARENT = "integrity.repair_from_clean_parent"
GATE_BEHAVIOURAL_NOT_RUN = "integrity.behavioural_gates_not_run"
#: Not a §8.5.1 gate: it binds the §8.5.1 evidence to the identities the
#: EvaluationRequest names, so it can only be produced where a request exists.
GATE_EVIDENCE_BINDING = "integrity.evidence_binding"

#: Emitted by `SourceIntegrityGateRunner` only when the runner was WIRED with the
#: derived affected-surface manifests (`derived_surfaces=`). It is a declared
#: capability, not a silent skip: `SourceIntegrityGateRunner.surface_binding` says
#: whether the runner has it, `run_gates()` raises for a candidate the wiring
#: promised and did not supply, and this module's report records the answer.
GATE_SURFACE_SCOPE_BINDING = "integrity.declared_surface_scope_binding"

#: In execution order. The first four are anchor comparisons; the rest are not.
GATE_IDS = (
    GATE_SYMBOL_PRESERVATION,
    GATE_SYMBOL_ARITY_COVERAGE,
    GATE_OP_REGISTRATION,
    GATE_DISPATCH_PREDICATE,
    GATE_CLEAN_BUILD,
    GATE_SEMANTIC_DIFF,
    GATE_CORE_HEADER,
    GATE_REPAIR_CLEAN_PARENT,
)

#: What `SourceIntegrityGateRunner` emits: the §8.5.1 set plus the binding gate.
RUNNER_GATE_IDS = GATE_IDS + (GATE_EVIDENCE_BINDING,)

F_UNDECLARED_SYMBOL_REMOVAL = "UNDECLARED_SYMBOL_REMOVAL"
F_UNDECLARED_ARITY_CHANGE = "UNDECLARED_ARITY_CHANGE"
F_UNDECLARED_SIGNATURE_CHANGE = "UNDECLARED_SIGNATURE_CHANGE"
F_UNDECLARED_SYMBOL_ADDITION = "UNDECLARED_SYMBOL_ADDITION"
F_EMPTY_ANCHOR_SYMBOL_TABLE = "EMPTY_ANCHOR_SYMBOL_TABLE"
F_SYMBOL_VERSIONS_NOT_EXTRACTED = "SYMBOL_VERSIONS_NOT_EXTRACTED"
F_UNMANGLED_ARITY_NOT_DERIVABLE = "UNMANGLED_ARITY_NOT_DERIVABLE"
F_UNDECLARED_REGISTRATION_REMOVAL = "UNDECLARED_REGISTRATION_REMOVAL"
F_UNDECLARED_REGISTRATION_ARITY_CHANGE = "UNDECLARED_REGISTRATION_ARITY_CHANGE"
F_REGISTRATION_ARITY_NOT_DERIVABLE = "REGISTRATION_ARITY_NOT_DERIVABLE"
F_EMPTY_ANCHOR_REGISTRATION_TABLE = "EMPTY_ANCHOR_REGISTRATION_TABLE"
F_ARTIFACT_NOT_FROM_CLEAN_BUILD = "ARTIFACT_NOT_FROM_CLEAN_BUILD"
F_ARTIFACT_FROM_INCREMENTAL_TREE = "ARTIFACT_FROM_INCREMENTAL_TREE"
F_BUILD_DIR_NOT_FRESH = "BUILD_DIR_NOT_FRESH"
F_BUILD_DIR_INSIDE_ACTOR_WORKTREE = "BUILD_DIR_INSIDE_ACTOR_WORKTREE"
F_BUILD_IN_PRODUCTION_TREE = "BUILD_IN_PRODUCTION_TREE"
F_SNAPSHOT_DIGEST_MISMATCH = "SNAPSHOT_DIGEST_MISMATCH"
F_SNAPSHOT_NOT_VERIFIED = "SNAPSHOT_NOT_VERIFIED"
F_UNDECLARED_FILE_TOUCHED = "UNDECLARED_FILE_TOUCHED"
F_UNDECLARED_FILE_CREATED = "UNDECLARED_FILE_CREATED"
F_UNDECLARED_FILE_DELETED = "UNDECLARED_FILE_DELETED"
F_PURE_DELETION_HUNK = "PURE_DELETION_HUNK"
F_EXCESSIVE_SHRINKAGE = "EXCESSIVE_SHRINKAGE"
F_SHRINKAGE_NOT_DERIVABLE = "SHRINKAGE_NOT_DERIVABLE"
F_BINARY_FILE_IN_DIFF = "BINARY_FILE_IN_DIFF"
F_ENVELOPE_FILES_EXCEEDED = "ENVELOPE_FILES_EXCEEDED"
F_ENVELOPE_LINES_EXCEEDED = "ENVELOPE_LINES_EXCEEDED"
F_ENVELOPE_HUNKS_EXCEEDED = "ENVELOPE_HUNKS_EXCEEDED"
F_MISDECLARED_CORE_HEADER_CHANGE = "MISDECLARED_CORE_HEADER_CHANGE"
F_CORE_HEADER_SURFACE_UNDER_DECLARED = "CORE_HEADER_SURFACE_UNDER_DECLARED"
F_REPAIR_NOT_RECHECKED_OUT = "REPAIR_NOT_RECHECKED_OUT"
F_REPAIR_BASE_NOT_PARENT_SNAPSHOT = "REPAIR_BASE_NOT_PARENT_SNAPSHOT"
F_REPAIR_CONTINUED_ON_FAILED_TREE = "REPAIR_CONTINUED_ON_FAILED_TREE"
F_REPAIR_CAP_EXCEEDED = "REPAIR_CAP_EXCEEDED"
F_BEHAVIOURAL_GATES_NOT_RUN = "BEHAVIOURAL_GATES_NOT_RUN"
F_EMPTY_DIFF = "EMPTY_DIFF"
F_UNPARSEABLE_DIFF = "UNPARSEABLE_DIFF"
F_CANDIDATE_ID_MISMATCH = "CANDIDATE_ID_MISMATCH"
F_ARTIFACT_SHA256_MISMATCH = "ARTIFACT_SHA256_MISMATCH"
F_NO_ANCHOR_BOUND = "NO_ANCHOR_BOUND"
F_SYMBOL_TABLE_NOT_BOUND_TO_ANCHOR = "SYMBOL_TABLE_NOT_BOUND_TO_ANCHOR"
F_SYMBOL_TABLE_NOT_BOUND_TO_ARTIFACT = "SYMBOL_TABLE_NOT_BOUND_TO_ARTIFACT"
F_SURFACE_SCOPE_NOT_BOUND = "SURFACE_SCOPE_NOT_BOUND"
F_SURFACE_SCOPE_MISDECLARED = "SURFACE_SCOPE_MISDECLARED"

FINDING_CODES = (
    F_UNDECLARED_SYMBOL_REMOVAL, F_UNDECLARED_ARITY_CHANGE,
    F_UNDECLARED_SIGNATURE_CHANGE, F_UNDECLARED_SYMBOL_ADDITION,
    F_EMPTY_ANCHOR_SYMBOL_TABLE, F_SYMBOL_VERSIONS_NOT_EXTRACTED,
    F_UNMANGLED_ARITY_NOT_DERIVABLE, F_UNDECLARED_REGISTRATION_REMOVAL,
    F_UNDECLARED_REGISTRATION_ARITY_CHANGE, F_REGISTRATION_ARITY_NOT_DERIVABLE,
    F_EMPTY_ANCHOR_REGISTRATION_TABLE,
    F_ARTIFACT_NOT_FROM_CLEAN_BUILD, F_ARTIFACT_FROM_INCREMENTAL_TREE,
    F_BUILD_DIR_NOT_FRESH, F_BUILD_DIR_INSIDE_ACTOR_WORKTREE,
    F_BUILD_IN_PRODUCTION_TREE, F_SNAPSHOT_DIGEST_MISMATCH, F_SNAPSHOT_NOT_VERIFIED,
    F_UNDECLARED_FILE_TOUCHED, F_UNDECLARED_FILE_CREATED, F_UNDECLARED_FILE_DELETED,
    F_PURE_DELETION_HUNK, F_EXCESSIVE_SHRINKAGE, F_SHRINKAGE_NOT_DERIVABLE,
    F_BINARY_FILE_IN_DIFF, F_ENVELOPE_FILES_EXCEEDED, F_ENVELOPE_LINES_EXCEEDED,
    F_ENVELOPE_HUNKS_EXCEEDED, F_MISDECLARED_CORE_HEADER_CHANGE,
    F_CORE_HEADER_SURFACE_UNDER_DECLARED, F_REPAIR_NOT_RECHECKED_OUT,
    F_REPAIR_BASE_NOT_PARENT_SNAPSHOT, F_REPAIR_CONTINUED_ON_FAILED_TREE,
    F_REPAIR_CAP_EXCEEDED, F_BEHAVIOURAL_GATES_NOT_RUN,
    F_EMPTY_DIFF, F_UNPARSEABLE_DIFF, F_CANDIDATE_ID_MISMATCH,
    F_ARTIFACT_SHA256_MISMATCH, F_NO_ANCHOR_BOUND,
    F_SYMBOL_TABLE_NOT_BOUND_TO_ANCHOR, F_SYMBOL_TABLE_NOT_BOUND_TO_ARTIFACT,
    F_SURFACE_SCOPE_NOT_BOUND, F_SURFACE_SCOPE_MISDECLARED,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ValueError(f"{label}: expected a lowercase hex sha256, got {value!r}")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label}: expected a bool, got {type(value).__name__}")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label}: expected an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{label}: must be >= {minimum}, got {value}")
    return value


_SEVERITY = {PASS: 0, COULD_NOT_CHECK: 1, FAIL: 2}


def _worst(checks: Iterable[schemas.Check]) -> schemas.Check:
    """Combine sub-checks. FAIL dominates COULD_NOT_CHECK dominates PASS.

    FAIL dominating COULD_NOT_CHECK is deliberate and is NOT a conflation: the
    reasons of every non-PASS sub-check are carried through, each prefixed with
    its own outcome, so the record still says which was which.
    """
    outcome = PASS
    reasons: list = []
    for chk in checks:
        if chk.outcome == PASS:
            continue
        if _SEVERITY[chk.outcome] > _SEVERITY[outcome]:
            outcome = chk.outcome
        reasons.extend(f"[{chk.outcome}] {r}" for r in chk.reasons)
    return schemas.Check(outcome, tuple(reasons))


def _fail(code: str, detail: str) -> schemas.Check:
    return schemas.Check(FAIL, (f"{code}: {detail}",))


def _cnc(code: str, detail: str) -> schemas.Check:
    return schemas.Check(COULD_NOT_CHECK, (f"{code}: {detail}",))


# =============================================================================
# ELF symbol extraction — "nm/readelf-style", in stdlib, without a subprocess
# =============================================================================

_ELF_MAGIC = b"\x7fELF"

_SHT_SYMTAB = 2
_SHT_STRTAB = 3
_SHT_DYNSYM = 11
_SHN_UNDEF = 0
_SHN_XINDEX = 0xFFFF

_BIND_NAMES = {0: "LOCAL", 1: "GLOBAL", 2: "WEAK", 10: "GNU_UNIQUE"}
_TYPE_NAMES = {0: "NOTYPE", 1: "OBJECT", 2: "FUNC", 3: "SECTION", 4: "FILE",
               5: "COMMON", 6: "TLS", 10: "GNU_IFUNC"}
_VIS_NAMES = {0: "DEFAULT", 1: "INTERNAL", 2: "HIDDEN", 3: "PROTECTED"}

#: The ABI surface: what `nm -D --defined-only` would list. LOCAL symbols are a
#: translation-unit detail and are excluded on purpose; HIDDEN/INTERNAL ones are
#: not exported at all.
_EXPORTED_BINDS = frozenset({"GLOBAL", "WEAK", "GNU_UNIQUE"})
_EXPORTED_VIS = frozenset({"DEFAULT", "PROTECTED"})
_EXPORTED_TYPES = frozenset({"FUNC", "OBJECT", "GNU_IFUNC", "NOTYPE", "TLS"})


@dataclass(frozen=True)
class ElfSymbol:
    """One entry of an ELF symbol table, decoded but not interpreted."""

    name: str
    table: str            # "dynsym" | "symtab"
    bind: str
    type: str
    visibility: str
    defined: bool
    size: int
    section_index: int

    @property
    def exported(self) -> bool:
        return (self.defined
                and bool(self.name)
                and self.bind in _EXPORTED_BINDS
                and self.visibility in _EXPORTED_VIS
                and self.type in _EXPORTED_TYPES)

    def to_dict(self) -> dict:
        return {"name": self.name, "table": self.table, "bind": self.bind,
                "type": self.type, "visibility": self.visibility,
                "defined": self.defined, "size": self.size,
                "section_index": self.section_index}


@dataclass(frozen=True)
class ElfSymbolTable:
    """Symbols extracted from one binary, with the provenance of the extraction.

    `label` is the role in the comparison ("anchor" / "candidate"); `preferred`
    names which ELF table the exported surface was taken from. `.dynsym` is
    preferred when present because it IS the ABI; `.symtab` is a superset that
    includes translation-unit-local symbols.

    `coverage_notes` carries what this extractor does NOT see, so a caller can
    turn a coverage gap into COULD_NOT_CHECK instead of inheriting a clean diff
    (*"a controller that discovers a coverage gap RECORDS the gap"*).
    """

    label: str
    source_path: str
    file_sha256: str
    elf_class: int          # 32 or 64
    symbols: tuple
    preferred: str          # "dynsym" | "symtab"
    extractor_id: str
    coverage_notes: tuple

    def __post_init__(self) -> None:
        _require_str(self.label, "ElfSymbolTable.label")
        _require_sha256(self.file_sha256, "ElfSymbolTable.file_sha256")
        if self.preferred not in ("dynsym", "symtab"):
            raise ValueError(f"preferred: {self.preferred!r} must be 'dynsym' or 'symtab'")
        if not isinstance(self.symbols, tuple):
            raise TypeError("ElfSymbolTable.symbols must be a tuple")

    def exported_names(self) -> frozenset:
        return frozenset(s.name for s in self.symbols
                         if s.table == self.preferred and s.exported)

    def to_dict(self) -> dict:
        return {"label": self.label, "source_path": self.source_path,
                "file_sha256": self.file_sha256, "elf_class": self.elf_class,
                "preferred": self.preferred, "extractor_id": self.extractor_id,
                "symbol_count": len(self.symbols),
                "exported_count": len(self.exported_names()),
                "coverage_notes": list(self.coverage_notes)}


def sha256_file(path: Any, *, max_bytes: Optional[int] = None,
                chunk: int = 1 << 20) -> str:
    """Stream a file's SHA-256. Raises on an unreadable file; never returns "".

    `max_bytes` is an OPTIONAL explicit bound. When it is exceeded the read
    RAISES — it does not truncate and return a hash of a prefix, which would be
    a wrong hash that looks exactly like a right one.
    """
    p = Path(path)
    digest = hashlib.sha256()
    seen = 0
    try:
        with p.open("rb") as handle:
            while True:
                block = handle.read(chunk)
                if not block:
                    break
                seen += len(block)
                if max_bytes is not None and seen > max_bytes:
                    raise TreeReadError(
                        f"{p}: exceeds max_bytes={max_bytes} at {seen} bytes; refusing "
                        "to hash a truncated prefix")
                digest.update(block)
    except OSError as exc:
        raise TreeReadError(f"{p}: unreadable ({exc})") from exc
    return digest.hexdigest()


def _u(fmt: str, blob: bytes, offset: int, label: str) -> tuple:
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(blob):
        raise ElfFormatError(f"{label}: truncated at offset {offset} (need {size} bytes)")
    return struct.unpack_from(fmt, blob, offset)


def _cstr(blob: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(blob):
        raise ElfFormatError(f"string table offset {offset} out of range")
    end = blob.find(b"\x00", offset)
    if end < 0:
        raise ElfFormatError(f"unterminated string at offset {offset}")
    return blob[offset:end].decode("utf-8", "surrogateescape")


def extract_elf_symbols(path: Any, *, label: str,
                        max_bytes: Optional[int] = None) -> ElfSymbolTable:
    """Extract `.dynsym` and `.symtab` from an ELF file, in pure stdlib.

    Raises `ElfFormatError` rather than returning an empty table for: a non-ELF
    file, an unsupported class/endianness, a stripped section-header table, or a
    truncated file. An empty table diffs clean against anything, which is exactly
    the fail-open this whole module exists to prevent.
    """
    _require_str(label, "label")
    p = Path(path)
    file_sha = sha256_file(p, max_bytes=max_bytes)
    try:
        blob = p.read_bytes()
    except OSError as exc:
        raise ElfFormatError(f"{p}: unreadable ({exc})") from exc

    if len(blob) < 64 or blob[:4] != _ELF_MAGIC:
        raise ElfFormatError(f"{p}: not an ELF file (magic {blob[:4]!r})")
    ei_class, ei_data = blob[4], blob[5]
    if ei_class not in (1, 2):
        raise ElfFormatError(f"{p}: unknown EI_CLASS {ei_class}")
    if ei_data not in (1, 2):
        raise ElfFormatError(f"{p}: unknown EI_DATA {ei_data}")
    endian = "<" if ei_data == 1 else ">"
    is64 = ei_class == 2

    if is64:
        (_e_type, _e_machine, _e_version, _e_entry, _e_phoff, e_shoff, _e_flags,
         _e_ehsize, _e_phentsize, _e_phnum, e_shentsize, e_shnum,
         e_shstrndx) = _u(endian + "HHIQQQIHHHHHH", blob, 16, "ELF64 header")
    else:
        (_e_type, _e_machine, _e_version, _e_entry, _e_phoff, e_shoff, _e_flags,
         _e_ehsize, _e_phentsize, _e_phnum, e_shentsize, e_shnum,
         e_shstrndx) = _u(endian + "HHIIIIIHHHHHH", blob, 16, "ELF32 header")

    if e_shoff == 0 or e_shnum == 0 and e_shoff == 0:
        raise ElfFormatError(
            f"{p}: no section header table (stripped). Symbol preservation cannot be "
            "checked from this binary; supply an unstripped artifact or a recorded "
            "symbol table")
    expected_entsize = 64 if is64 else 40
    if e_shentsize != expected_entsize:
        raise ElfFormatError(
            f"{p}: e_shentsize={e_shentsize}, expected {expected_entsize}")

    def shdr(index: int) -> tuple:
        off = e_shoff + index * e_shentsize
        if is64:
            return _u(endian + "IIQQQQIIQQ", blob, off, f"section header {index}")
        return _u(endian + "IIIIIIIIII", blob, off, f"section header {index}")

    # e_shnum == 0 means the real count lives in section 0's sh_size (ELF spec).
    shnum = e_shnum
    if shnum == 0:
        shnum = shdr(0)[5]
        if shnum == 0:
            raise ElfFormatError(f"{p}: section header count is zero")
    if e_shstrndx == _SHN_XINDEX:
        e_shstrndx = shdr(0)[6]

    headers = [shdr(i) for i in range(shnum)]

    def section_bytes(index: int) -> bytes:
        if index < 0 or index >= len(headers):
            raise ElfFormatError(f"{p}: section index {index} out of range")
        hdr = headers[index]
        sh_type, sh_offset, sh_size = hdr[1], hdr[4], hdr[5]
        if sh_type == 8:  # SHT_NOBITS occupies no file space
            return b""
        if sh_offset + sh_size > len(blob):
            raise ElfFormatError(f"{p}: section {index} extends past end of file")
        return blob[sh_offset:sh_offset + sh_size]

    symbols: list = []
    found_tables: set = set()
    sym_entsize = 24 if is64 else 16
    for index, hdr in enumerate(headers):
        sh_type, sh_link, sh_entsize = hdr[1], hdr[6], hdr[9]
        if sh_type not in (_SHT_SYMTAB, _SHT_DYNSYM):
            continue
        table_name = "symtab" if sh_type == _SHT_SYMTAB else "dynsym"
        if sh_entsize not in (0, sym_entsize):
            raise ElfFormatError(
                f"{p}: {table_name} sh_entsize={sh_entsize}, expected {sym_entsize}")
        if sh_link >= len(headers) or headers[sh_link][1] != _SHT_STRTAB:
            raise ElfFormatError(
                f"{p}: {table_name} sh_link={sh_link} is not a string table")
        strtab = section_bytes(sh_link)
        data = section_bytes(index)
        if len(data) % sym_entsize:
            raise ElfFormatError(
                f"{p}: {table_name} size {len(data)} is not a multiple of {sym_entsize}")
        found_tables.add(table_name)
        for off in range(0, len(data), sym_entsize):
            if is64:
                st_name, st_info, st_other, st_shndx, _val, st_size = _u(
                    endian + "IBBHQQ", data, off, f"{table_name} entry")
            else:
                st_name, _val, st_size, st_info, st_other, st_shndx = _u(
                    endian + "IIIBBH", data, off, f"{table_name} entry")
            name = _cstr(strtab, st_name) if st_name else ""
            bind = _BIND_NAMES.get(st_info >> 4, f"BIND_{st_info >> 4}")
            styp = _TYPE_NAMES.get(st_info & 0xF, f"TYPE_{st_info & 0xF}")
            vis = _VIS_NAMES.get(st_other & 0x3, f"VIS_{st_other & 0x3}")
            symbols.append(ElfSymbol(
                name=name, table=table_name, bind=bind, type=styp, visibility=vis,
                defined=st_shndx != _SHN_UNDEF, size=st_size, section_index=st_shndx))

    if not found_tables:
        raise ElfFormatError(
            f"{p}: contains neither .dynsym nor .symtab. A binary with no symbol table "
            "cannot evidence symbol preservation")

    preferred = "dynsym" if "dynsym" in found_tables else "symtab"
    notes = [
        f"{F_SYMBOL_VERSIONS_NOT_EXTRACTED}: symbol versions (.gnu.version / "
        "'name@@VER') are not decoded by this extractor, so a version-node change "
        "on an otherwise-identical name is not detected here",
    ]
    if preferred == "symtab":
        notes.append(
            "exported surface taken from .symtab because the binary has no .dynsym; "
            ".symtab is a superset and may include statically-linked internals")
    return ElfSymbolTable(
        label=label, source_path=str(p), file_sha256=file_sha,
        elf_class=64 if is64 else 32, symbols=tuple(symbols), preferred=preferred,
        extractor_id="autokernel.evaluator.integrity.elf/v1",
        coverage_notes=tuple(notes))


def exported_surface(table: ElfSymbolTable) -> frozenset:
    """The exported ABI names of `table`. Never silently empty for the caller."""
    if not isinstance(table, ElfSymbolTable):
        raise TypeError("exported_surface expects an ElfSymbolTable")
    return table.exported_names()


# =============================================================================
# Itanium mangled-name parsing — enough to pair a removal with an addition
# =============================================================================

class _MangleError(Exception):
    """Internal: this parser met a construct it does not fully handle."""


_BUILTIN_ONE = set("vwbcahstijlmxynofedgz")
_BUILTIN_D = {"Dd", "De", "Df", "Dh", "Di", "Ds", "Da", "Dc", "Dn"}
_QUALIFIERS = set("PROCGrVK")
_CTOR_DTOR = {"C1", "C2", "C3", "C4", "C5", "D0", "D1", "D2", "D4", "D5"}
_OPERATORS = {
    "nw", "na", "dl", "da", "ps", "ng", "ad", "de", "co", "pl", "mi", "ml", "dv",
    "rm", "an", "or", "eo", "aS", "pL", "mI", "mL", "dV", "rM", "aN", "oR", "eO",
    "ls", "rs", "lS", "rS", "eq", "ne", "lt", "gt", "le", "ge", "nt", "aa", "oo",
    "pp", "mm", "cm", "pm", "pt", "cl", "ix", "qu", "cv", "li", "ss", "aw",
}
_STD_ABBREV = {"Sa", "Sb", "Ss", "Si", "So", "Sd"}
#: GCC/Clang clone suffixes. Stripped before parsing; the RAW name stays the identity.
_CLONE_SUFFIX_RE = re.compile(r"\.(?:isra|part|constprop|cold|lto_priv|localalias)"
                              r"(?:\.\d+)*$")


@dataclass(frozen=True)
class ParsedName:
    """What this parser could establish about a mangled name.

    `param_count is None` means "not derivable", NEVER "zero". The distinction is
    load-bearing: a zero would make an arity change from 0 to 1 invisible.
    """

    mangled: str
    qualified: str
    param_count: Optional[int]
    templated: bool

    def to_dict(self) -> dict:
        return {"mangled": self.mangled, "qualified": self.qualified,
                "param_count": self.param_count, "templated": self.templated}


class _MangleParser:
    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 2  # past "_Z"

    def peek(self) -> str:
        if self.i >= len(self.s):
            raise _MangleError("unexpected end of mangled name")
        return self.s[self.i]

    def at_end(self) -> bool:
        return self.i >= len(self.s)

    # -- names ------------------------------------------------------------
    def source_name(self) -> str:
        start = self.i
        while self.i < len(self.s) and self.s[self.i].isdigit():
            self.i += 1
        if start == self.i:
            raise _MangleError(f"expected a source-name length at {start}")
        length = int(self.s[start:self.i])
        if self.i + length > len(self.s):
            raise _MangleError("source-name runs past end of mangled name")
        out = self.s[self.i:self.i + length]
        self.i += length
        return out

    def substitution(self) -> str:
        start = self.i
        self.i += 1  # 'S'
        two = self.s[start:start + 2]
        if two == "St":
            self.i += 1
            return "std"
        if two in _STD_ABBREV:
            self.i += 1
            return two
        while self.i < len(self.s) and self.s[self.i] in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.i += 1
        if self.i >= len(self.s) or self.s[self.i] != "_":
            raise _MangleError(f"malformed substitution at {start}")
        self.i += 1
        return self.s[start:self.i]

    def template_param(self) -> str:
        start = self.i
        self.i += 1  # 'T'
        while self.i < len(self.s) and self.s[self.i] != "_":
            self.i += 1
        if self.i >= len(self.s):
            raise _MangleError(f"malformed template-param at {start}")
        self.i += 1
        return self.s[start:self.i]

    def maybe_template_args(self) -> bool:
        if self.i < len(self.s) and self.s[self.i] == "I":
            self.template_args()
            return True
        return False

    def template_args(self) -> None:
        self.i += 1  # 'I'
        while True:
            if self.at_end():
                raise _MangleError("unterminated template-args")
            c = self.peek()
            if c == "E":
                self.i += 1
                return
            if c == "L":               # literal
                self.i += 1
                self.type()
                while not self.at_end() and self.peek() != "E":
                    self.i += 1
                if self.at_end():
                    raise _MangleError("unterminated literal")
                self.i += 1
                continue
            if c in "XJ":              # expression / argument pack
                raise _MangleError(f"unsupported template-arg {c!r}")
            self.type()

    def component(self) -> str:
        c = self.peek()
        if c.isdigit():
            out = self.source_name()
        elif self.s[self.i:self.i + 2] in _CTOR_DTOR:
            out = self.s[self.i:self.i + 2]
            self.i += 2
        elif self.s[self.i:self.i + 2] in _OPERATORS:
            out = "operator" + self.s[self.i:self.i + 2]
            self.i += 2
        elif c == "S":
            out = self.substitution()
            if out == "std" and not self.at_end() and self.peek().isdigit():
                out = "std::" + self.source_name()
        elif c == "T":
            out = self.template_param()
        elif c == "L":
            raise _MangleError("local/internal-linkage name")
        else:
            raise _MangleError(f"unsupported name component {c!r} at {self.i}")
        return out

    def nested_name(self) -> tuple:
        self.i += 1  # 'N'
        while not self.at_end() and self.peek() in "rVKRO":
            self.i += 1
        parts: list = []
        templated = False
        while True:
            if self.at_end():
                raise _MangleError("unterminated nested-name")
            if self.peek() == "E":
                self.i += 1
                break
            parts.append(self.component())
            templated = self.maybe_template_args()
        if not parts:
            raise _MangleError("empty nested-name")
        return "::".join(parts), templated

    def name(self) -> tuple:
        if self.peek() == "N":
            return self.nested_name()
        part = self.component()
        templated = self.maybe_template_args()
        return part, templated

    # -- types ------------------------------------------------------------
    def type(self) -> None:
        c = self.peek()
        if c in _QUALIFIERS:
            self.i += 1
            self.type()
            return
        if c == "D":
            two = self.s[self.i:self.i + 2]
            if two in _BUILTIN_D:
                self.i += 2
                return
            raise _MangleError(f"unsupported D-type {two!r}")
        if c in _BUILTIN_ONE:
            self.i += 1
            return
        if c == "F":
            self.i += 1
            if not self.at_end() and self.peek() == "Y":
                self.i += 1
            while self.peek() != "E":
                self.type()
            self.i += 1
            return
        if c == "A":
            self.i += 1
            start = self.i
            while not self.at_end() and self.s[self.i].isdigit():
                self.i += 1
            if self.at_end() or self.peek() != "_":
                raise _MangleError(f"unsupported array bound at {start}")
            self.i += 1
            self.type()
            return
        if c == "M":
            self.i += 1
            self.type()
            self.type()
            return
        if c == "T":
            self.template_param()
            self.maybe_template_args()
            return
        if c == "S":
            out = self.substitution()
            if out == "std" and not self.at_end() and self.peek().isdigit():
                self.source_name()
            self.maybe_template_args()
            return
        if c == "N":
            self.nested_name()
            return
        if c == "u":
            self.i += 1
            self.source_name()
            return
        if c.isdigit():
            self.source_name()
            self.maybe_template_args()
            return
        raise _MangleError(f"unsupported type code {c!r} at {self.i}")


def parse_mangled_name(name: str) -> Optional[ParsedName]:
    """Parse an Itanium C++ mangled name into (qualified name, parameter count).

    Returns `None` when the name is not a `_Z` mangling, or when even the NAME
    portion could not be parsed. When the name parses but the parameter list does
    not, `param_count` is `None` — three outcomes, not two.

    Why this exists at all: a symbol table diff detects a dropped template
    specialization or a changed signature as a REMOVAL of the old mangled name
    plus an ADDITION of a new one. Pairing the two by qualified name is what lets
    the finding be LABELLED `arity_changed` (which `declared_symbol_deltas` names)
    instead of only `removed`. The label is a convenience; the hard failure
    already fires on the removal, so a parse failure here cannot fail open.
    """
    if not isinstance(name, str) or not name.startswith("_Z") or len(name) < 3:
        return None
    core = _CLONE_SUFFIX_RE.sub("", name)
    parser = _MangleParser(core)
    try:
        qualified, templated = parser.name()
    except (_MangleError, IndexError, ValueError):
        return None
    if parser.at_end():
        return ParsedName(name, qualified, None, templated)
    types: list = []
    try:
        while not parser.at_end():
            start = parser.i
            parser.type()
            types.append(core[start:parser.i])
    except (_MangleError, IndexError, ValueError):
        return ParsedName(name, qualified, None, templated)
    # Itanium encodes the return type only for TEMPLATE functions, and it comes
    # first. Dropping it here is what keeps `max<int>(int,int)` at arity 2.
    if templated and types:
        types = types[1:]
    if len(types) == 1 and types[0] == "v":
        count = 0
    else:
        count = len(types)
    return ParsedName(name, qualified, count, templated)


# =============================================================================
# §8.5.1 (1) — symbol and registration preservation
# =============================================================================

@dataclass(frozen=True)
class DeclaredSymbolDeltas:
    """`proposal.declared_symbol_deltas` (§7.2), as a checked value object.

    A declaration matches a symbol by EXACT mangled name or by its qualified
    name, so a proposal may declare `ggml::mul_mat` without predicting the exact
    mangling the compiler will emit. It may not declare a wildcard: there is no
    pattern syntax here, on purpose.
    """

    added: frozenset
    removed: frozenset
    arity_changed: frozenset

    def __post_init__(self) -> None:
        for field_name in ("added", "removed", "arity_changed"):
            value = getattr(self, field_name)
            if not isinstance(value, frozenset):
                raise TypeError(f"DeclaredSymbolDeltas.{field_name} must be a frozenset")
            for item in value:
                _require_str(item, f"declared_symbol_deltas.{field_name}[]")

    @classmethod
    def from_proposal(cls, proposal: Mapping[str, Any]) -> "DeclaredSymbolDeltas":
        if not isinstance(proposal, Mapping):
            raise TypeError("proposal must be a mapping")
        deltas = proposal.get("declared_symbol_deltas")
        if not isinstance(deltas, Mapping):
            raise DeclarationMissing(
                "proposal.declared_symbol_deltas is absent or not a mapping. An absent "
                "declaration is NOT an empty one — it is an undeclared removal waiting "
                "to happen (§7.2), so this raises instead of defaulting to empty")
        out = {}
        for key in ("added", "removed", "arity_changed"):
            value = deltas.get(key)
            if not isinstance(value, (list, tuple)):
                raise DeclarationMissing(
                    f"proposal.declared_symbol_deltas.{key} is absent or not a list; "
                    "all three keys are required even when empty")
            out[key] = frozenset(value)
        return cls(added=out["added"], removed=out["removed"],
                   arity_changed=out["arity_changed"])

    def covers(self, declared: frozenset, mangled: str,
               parsed: Optional[ParsedName]) -> bool:
        if mangled in declared:
            return True
        return bool(parsed and parsed.qualified in declared)

    def to_dict(self) -> dict:
        return {"added": sorted(self.added), "removed": sorted(self.removed),
                "arity_changed": sorted(self.arity_changed)}


@dataclass(frozen=True)
class SignatureChange:
    """A removal/addition pair sharing a qualified name — arity or signature."""

    qualified: str
    removed: tuple
    added: tuple
    kind: str            # "arity_changed" | "signature_changed"
    old_arity: Optional[int]
    new_arity: Optional[int]

    def to_dict(self) -> dict:
        return {"qualified": self.qualified, "removed": list(self.removed),
                "added": list(self.added), "kind": self.kind,
                "old_arity": self.old_arity, "new_arity": self.new_arity}


@dataclass(frozen=True)
class SymbolDiff:
    """The complete symbol comparison. No list here is truncated or capped."""

    removed: tuple
    added: tuple
    signature_changes: tuple
    anchor_count: int
    candidate_count: int
    unmangled_removed: tuple
    unmangled_added: tuple

    def to_dict(self) -> dict:
        return {"removed": list(self.removed), "added": list(self.added),
                "signature_changes": [c.to_dict() for c in self.signature_changes],
                "anchor_count": self.anchor_count,
                "candidate_count": self.candidate_count,
                "unmangled_removed": list(self.unmangled_removed),
                "unmangled_added": list(self.unmangled_added),
                "listing_is_complete": True}


def diff_symbol_tables(anchor: ElfSymbolTable,
                       candidate: ElfSymbolTable) -> SymbolDiff:
    """Diff two exported surfaces and pair removals with additions by name."""
    for name, table in (("anchor", anchor), ("candidate", candidate)):
        if not isinstance(table, ElfSymbolTable):
            raise TypeError(f"{name} must be an ElfSymbolTable")
    a_names = anchor.exported_names()
    c_names = candidate.exported_names()
    removed = tuple(sorted(a_names - c_names))
    added = tuple(sorted(c_names - a_names))

    def group(names: Sequence[str]) -> dict:
        out: dict = {}
        for n in names:
            parsed = parse_mangled_name(n)
            if parsed is None:
                continue
            out.setdefault(parsed.qualified, []).append(parsed)
        return out

    removed_groups, added_groups = group(removed), group(added)
    changes: list = []
    for qualified in sorted(set(removed_groups) & set(added_groups)):
        olds, news = removed_groups[qualified], added_groups[qualified]
        old_arities = {p.param_count for p in olds}
        new_arities = {p.param_count for p in news}
        derivable = None not in old_arities and None not in new_arities
        if derivable and old_arities != new_arities:
            kind = "arity_changed"
        else:
            kind = "signature_changed"
        changes.append(SignatureChange(
            qualified=qualified,
            removed=tuple(p.mangled for p in olds),
            added=tuple(p.mangled for p in news),
            kind=kind,
            old_arity=sorted(a for a in old_arities if a is not None)[0]
            if derivable and old_arities else None,
            new_arity=sorted(a for a in new_arities if a is not None)[0]
            if derivable and new_arities else None,
        ))
    return SymbolDiff(
        removed=removed, added=added, signature_changes=tuple(changes),
        anchor_count=len(a_names), candidate_count=len(c_names),
        unmangled_removed=tuple(n for n in removed if parse_mangled_name(n) is None),
        unmangled_added=tuple(n for n in added if parse_mangled_name(n) is None),
    )


def check_symbol_preservation(anchor: Optional[ElfSymbolTable],
                              candidate: Optional[ElfSymbolTable],
                              declared: DeclaredSymbolDeltas) -> api.GateResult:
    """§8.5.1 (1): any removal or arity change not declared is a HARD failure.

    Three outcomes, and the third is real:
      * either table missing (unreadable, stripped, no anchor) -> COULD_NOT_CHECK;
      * an EMPTY anchor exported surface -> COULD_NOT_CHECK, because a diff of
        nothing against nothing has no removals and would read as a clean PASS;
      * otherwise PASS/FAIL on the declared-delta comparison.

    Undeclared ADDITIONS are recorded as a note and do NOT fail: §8.5.1 makes
    only *"removal or arity change"* a hard failure, and §6.4 makes the actor's
    declaration a scored prediction rather than a scope input. The prediction
    miss is carried in `notes` for the critic to score.
    """
    if not isinstance(declared, DeclaredSymbolDeltas):
        raise TypeError("declared must be a DeclaredSymbolDeltas")
    if anchor is None or candidate is None:
        missing = "anchor" if anchor is None else "candidate"
        return api.GateResult(
            gate_id=GATE_SYMBOL_PRESERVATION, gate_class=api.GATE_INTEGRITY,
            check=_cnc("SYMBOL_TABLE_UNAVAILABLE",
                       f"no {missing} symbol table was extracted; symbol preservation "
                       "is unevaluated, which is not the same as preserved"),
            requires_anchor=True)

    diff = diff_symbol_tables(anchor, candidate)
    if diff.anchor_count == 0:
        return api.GateResult(
            gate_id=GATE_SYMBOL_PRESERVATION, gate_class=api.GATE_INTEGRITY,
            check=_cnc(F_EMPTY_ANCHOR_SYMBOL_TABLE,
                       f"the anchor binary {anchor.source_path} exports no symbols; an "
                       "empty anchor surface cannot evidence preservation"),
            requires_anchor=True,
            evidence_ref=anchor.file_sha256)

    # A removal that is part of a declared arity change is declared.
    change_by_removed: dict = {}
    for change in diff.signature_changes:
        for m in change.removed:
            change_by_removed[m] = change

    undeclared_removals: list = []
    undeclared_arity: list = []
    undeclared_signature: list = []
    for name in diff.removed:
        parsed = parse_mangled_name(name)
        change = change_by_removed.get(name)
        if change is not None:
            if declared.covers(declared.arity_changed, name, parsed):
                continue
            if declared.covers(declared.removed, name, parsed):
                continue
            if change.kind == "arity_changed":
                undeclared_arity.append(change)
            else:
                undeclared_signature.append(change)
            continue
        if declared.covers(declared.removed, name, parsed):
            continue
        undeclared_removals.append(name)

    checks: list = []
    for name in undeclared_removals:
        checks.append(_fail(
            F_UNDECLARED_SYMBOL_REMOVAL,
            f"{name!r} is exported by the anchor and absent from the candidate, and is "
            "not in proposal.declared_symbol_deltas.removed"))
    seen: set = set()
    for change in undeclared_arity:
        if change.qualified in seen:
            continue
        seen.add(change.qualified)
        checks.append(_fail(
            F_UNDECLARED_ARITY_CHANGE,
            f"{change.qualified!r} changed arity {change.old_arity} -> "
            f"{change.new_arity} ({list(change.removed)} -> {list(change.added)}) and is "
            "not in proposal.declared_symbol_deltas.arity_changed"))
    for change in undeclared_signature:
        if change.qualified in seen:
            continue
        seen.add(change.qualified)
        checks.append(_fail(
            F_UNDECLARED_SIGNATURE_CHANGE,
            f"{change.qualified!r} changed signature ({list(change.removed)} -> "
            f"{list(change.added)}); arity was not derivable from the mangling, so this "
            "is reported as the superset finding and is still undeclared"))

    notes: list = list(anchor.coverage_notes)
    undeclared_additions = [
        n for n in diff.added
        if not declared.covers(declared.added, n, parse_mangled_name(n))
        and n not in {m for c in diff.signature_changes for m in c.added}
    ]
    if undeclared_additions:
        notes.append(
            f"{F_UNDECLARED_SYMBOL_ADDITION}: {len(undeclared_additions)} exported "
            f"symbol(s) added but not declared: {sorted(undeclared_additions)}. Recorded "
            "as a scored prediction miss (§6.4), NOT as a hard failure — §8.5.1 makes "
            "only removal and arity change hard")
    notes.append(
        f"compared {diff.anchor_count} anchor and {diff.candidate_count} candidate "
        f"exported symbols from .{anchor.preferred}; the listing is complete, not capped")

    check = _worst(checks) if checks else schemas.Check(PASS)
    return api.GateResult(
        gate_id=GATE_SYMBOL_PRESERVATION, gate_class=api.GATE_INTEGRITY, check=check,
        requires_anchor=True,
        evidence_ref=f"{anchor.file_sha256[:12]}..{candidate.file_sha256[:12]}",
        notes=tuple(notes))


def check_symbol_arity_coverage(
        anchor: Optional[ElfSymbolTable],
        candidate: Optional[ElfSymbolTable],
        signature_index: Optional[Mapping[str, int]]) -> api.GateResult:
    """Report, rather than hide, the arity that an ELF symbol table cannot show.

    An `extern "C"` symbol carries NO signature in its name, so a change from
    `f(a)` to `f(a, b)` leaves the exported name byte-identical and is INVISIBLE
    to `check_symbol_preservation`. That is a genuine coverage gap in this
    instrument, and P-AK-SEARCH-1's *"No self-amendment"* clause says a controller
    that finds one RECORDS it — so this gate answers COULD_NOT_CHECK, naming the
    symbols and naming the input that would close it (`signature_index`), instead
    of letting a clean symbol diff imply an unchanged C ABI.

    PASS when there are no unmangled exported symbols in scope, or when
    `signature_index` covers every one of them and the arities agree.
    """
    if anchor is None or candidate is None:
        return api.GateResult(
            gate_id=GATE_SYMBOL_ARITY_COVERAGE, gate_class=api.GATE_INTEGRITY,
            check=_cnc("SYMBOL_TABLE_UNAVAILABLE",
                       "arity coverage cannot be assessed without both symbol tables"),
            requires_anchor=True)

    common = sorted(anchor.exported_names() & candidate.exported_names())
    unmangled = [n for n in common if parse_mangled_name(n) is None]
    if not unmangled:
        return api.GateResult(
            gate_id=GATE_SYMBOL_ARITY_COVERAGE, gate_class=api.GATE_INTEGRITY,
            check=schemas.Check(PASS), requires_anchor=True,
            notes=("no unmangled exported symbols in the common surface; every "
                   "signature change would alter a mangled name and is therefore "
                   "visible to the preservation gate",))

    if signature_index is None:
        return api.GateResult(
            gate_id=GATE_SYMBOL_ARITY_COVERAGE, gate_class=api.GATE_INTEGRITY,
            check=_cnc(F_UNMANGLED_ARITY_NOT_DERIVABLE,
                       f"{len(unmangled)} exported symbol(s) are unmangled (C linkage), "
                       "so their arity is not derivable from the ELF symbol table and no "
                       "signature_index was supplied. Supply a declared signature source "
                       "(headers, compile database or DWARF) for: "
                       f"{unmangled[:20]}{' …' if len(unmangled) > 20 else ''}"),
            requires_anchor=True,
            notes=(f"the reason list above names at most 20 of {len(unmangled)} symbols; "
                   "the full set is the unmangled intersection of both exported surfaces "
                   "and is not otherwise truncated",))

    if not isinstance(signature_index, Mapping):
        raise TypeError("signature_index must be a mapping of symbol name -> arity")

    def _side_arity(entry: Any, name: str, side: str) -> Optional[int]:
        """The declared arity for one side, or None for 'not declared'.

        `None` is NEVER coerced to a number, for the reason `ParsedName` gives:
        an absent arity that compares equal to another absent arity turns a
        missing declaration into evidence of an unchanged C ABI.
        """
        value = entry.get(side)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"signature_index[{name!r}][{side!r}] must be an int arity, got "
                f"{value!r}")
        return value

    uncovered: list = []
    mismatches: list = []
    for name in unmangled:
        if name not in signature_index:
            uncovered.append(f"{name} (no entry)")
            continue
        entry = signature_index[name]
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"signature_index[{name!r}] must be a mapping with 'anchor' and "
                "'candidate' arities")
        old = _side_arity(entry, name, "anchor")
        new = _side_arity(entry, name, "candidate")
        if old is None or new is None:
            absent = "/".join(side for side, value in (("anchor", old), ("candidate", new))
                              if value is None)
            uncovered.append(f"{name} (entry declares no {absent} arity)")
            continue
        if old != new:
            mismatches.append((name, old, new))
    if uncovered:
        return api.GateResult(
            gate_id=GATE_SYMBOL_ARITY_COVERAGE, gate_class=api.GATE_INTEGRITY,
            check=_cnc(F_UNMANGLED_ARITY_NOT_DERIVABLE,
                       f"signature_index does not cover {len(uncovered)} unmangled "
                       f"exported symbol(s): {uncovered[:20]}"
                       f"{' …' if len(uncovered) > 20 else ''}. An entry present but "
                       "missing either arity covers NOTHING: two absent arities compare "
                       "equal, which would read as an unchanged C ABI"),
            requires_anchor=True,
            notes=(f"the reason list above names at most 20 of {len(uncovered)} "
                   "uncovered symbols; the full set is every unmangled exported symbol "
                   "for which signature_index declares fewer than two arities, and is "
                   "not otherwise truncated",))

    if mismatches:
        return api.GateResult(
            gate_id=GATE_SYMBOL_ARITY_COVERAGE, gate_class=api.GATE_INTEGRITY,
            check=_worst([
                _fail(F_UNDECLARED_ARITY_CHANGE,
                      f"{name!r} keeps its exported name but its arity changed "
                      f"{old} -> {new}; an unmangled arity change is invisible to a "
                      "symbol-table diff and is caught only here")
                for name, old, new in mismatches]),
            requires_anchor=True)
    return api.GateResult(
        gate_id=GATE_SYMBOL_ARITY_COVERAGE, gate_class=api.GATE_INTEGRITY,
        check=schemas.Check(PASS), requires_anchor=True,
        notes=(f"signature_index covers all {len(unmangled)} unmangled exported "
               "symbols and their arities agree",))


# =============================================================================
# §8.5.1 (1, continued) — op-registration tables and dispatch predicates
# =============================================================================

KIND_OP_REGISTRATION = "op_registration"
KIND_DISPATCH_PREDICATE = "dispatch_predicate"
_REGISTRATION_KINDS = (KIND_OP_REGISTRATION, KIND_DISPATCH_PREDICATE)


@dataclass(frozen=True)
class RegistrationEntry:
    """One op registration or dispatch predicate, keyed for a set diff."""

    registry: str
    key: str
    arity: Optional[int]
    location: Optional[str]

    def __post_init__(self) -> None:
        _require_str(self.registry, "RegistrationEntry.registry")
        _require_str(self.key, "RegistrationEntry.key")
        if self.arity is not None:
            _require_int(self.arity, "RegistrationEntry.arity")

    @property
    def ident(self) -> tuple:
        return (self.registry, self.key)

    def to_dict(self) -> dict:
        return {"registry": self.registry, "key": self.key, "arity": self.arity,
                "location": self.location}


@dataclass(frozen=True)
class RegistrationTable:
    """A registration/dispatch table for one side of the comparison."""

    label: str
    kind: str
    entries: tuple
    extractor_id: str

    def __post_init__(self) -> None:
        _require_str(self.label, "RegistrationTable.label")
        if self.kind not in _REGISTRATION_KINDS:
            raise ValueError(f"kind: {self.kind!r} must be one of {list(_REGISTRATION_KINDS)}")
        if not isinstance(self.entries, tuple):
            raise TypeError("RegistrationTable.entries must be a tuple")
        for entry in self.entries:
            if not isinstance(entry, RegistrationEntry):
                raise TypeError("RegistrationTable.entries must hold RegistrationEntry")
        _require_str(self.extractor_id, "RegistrationTable.extractor_id")

    def by_ident(self) -> dict:
        return {e.ident: e for e in self.entries}

    def to_dict(self) -> dict:
        return {"label": self.label, "kind": self.kind,
                "extractor_id": self.extractor_id,
                "entries": [e.to_dict() for e in self.entries]}


class PatternRegistrationExtractor:
    """Extract registration/dispatch entries from source text by DECLARED patterns.

    The patterns are supplied by the backend adapter, never guessed here: this
    module has no idea what a given tree's registration macro looks like, and a
    guessed pattern that matches nothing produces an empty table that diffs
    clean.

    Refusing an empty pattern set is the point of the constructor. An extractor
    with no patterns finds no entries **because it looked for none**, and the
    resulting "no removals" is the fail-open this module exists to prevent.
    """

    extractor_id = "autokernel.evaluator.integrity.pattern/v1"

    def __init__(self, *, kind: str, patterns: Mapping[str, str],
                 declared_by: str) -> None:
        if kind not in _REGISTRATION_KINDS:
            raise ValueError(f"kind: {kind!r} must be one of {list(_REGISTRATION_KINDS)}")
        if not isinstance(patterns, Mapping) or not patterns:
            raise EnvelopeNotDeclared(
                "PatternRegistrationExtractor requires a non-empty {registry: regex} "
                "mapping declared by the backend adapter; an extractor with no patterns "
                "reports no removals because it looked for none")
        compiled = {}
        for registry, pattern in patterns.items():
            _require_str(registry, "patterns key")
            _require_str(pattern, f"patterns[{registry!r}]")
            rx = re.compile(pattern)
            if "key" not in rx.groupindex:
                raise ValueError(
                    f"patterns[{registry!r}]: the regex must define a named group "
                    "'key' (and may define 'arity')")
            compiled[registry] = rx
        self.kind = kind
        self.patterns = compiled
        self.declared_by = _require_str(declared_by, "declared_by")

    def extract_text(self, label: str, sources: Mapping[str, str]) -> RegistrationTable:
        """Extract from `{relative_path: text}`. Raises on a non-string body."""
        if not isinstance(sources, Mapping):
            raise TypeError("sources must be a mapping of path -> text")
        entries: list = []
        for path in sorted(sources):
            text = sources[path]
            if not isinstance(text, str):
                raise TypeError(f"sources[{path!r}] must be str, got {type(text).__name__}")
            for registry, rx in self.patterns.items():
                for match in rx.finditer(text):
                    arity_raw = (match.groupdict().get("arity")
                                 if "arity" in rx.groupindex else None)
                    line = text.count("\n", 0, match.start()) + 1
                    entries.append(RegistrationEntry(
                        registry=registry, key=match.group("key"),
                        arity=int(arity_raw) if arity_raw is not None else None,
                        location=f"{path}:{line}"))
        return RegistrationTable(
            label=label, kind=self.kind,
            entries=tuple(sorted(entries, key=lambda e: (e.registry, e.key, e.location or ""))),
            extractor_id=f"{self.extractor_id}({self.declared_by})")

    def extract_tree(self, label: str, root: Any, *,
                     suffixes: Sequence[str],
                     exclude_dir_names: Sequence[str] = ()) -> RegistrationTable:
        """Read `root` recursively and extract. Raises on an unreadable file."""
        base = Path(root)
        if not base.is_dir():
            raise TreeReadError(f"{base}: not a directory")
        if not suffixes:
            raise ValueError("suffixes must be non-empty; scanning nothing finds nothing")
        excluded = set(exclude_dir_names)
        sources: dict = {}
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if excluded & set(path.relative_to(base).parts[:-1]):
                continue
            if path.suffix not in suffixes:
                continue
            try:
                sources[path.relative_to(base).as_posix()] = path.read_text(
                    encoding="utf-8", errors="surrogateescape")
            except OSError as exc:
                raise TreeReadError(f"{path}: unreadable ({exc})") from exc
        return self.extract_text(label, sources)


@dataclass(frozen=True)
class RegistrationDiff:
    removed: tuple
    added: tuple
    arity_changed: tuple
    #: Entries present on both sides where exactly one side carries an arity. An
    #: arity that is None means "the declared pattern did not capture it", NEVER
    #: "unchanged" — comparing it as equal would hide the arity change this gate
    #: exists to catch.
    arity_not_comparable: tuple
    anchor_count: int
    candidate_count: int

    def to_dict(self) -> dict:
        return {"removed": [list(i) for i in self.removed],
                "added": [list(i) for i in self.added],
                "arity_changed": [list(i) for i in self.arity_changed],
                "arity_not_comparable": [list(i) for i in self.arity_not_comparable],
                "anchor_count": self.anchor_count,
                "candidate_count": self.candidate_count}


def diff_registration_tables(anchor: RegistrationTable,
                             candidate: RegistrationTable) -> RegistrationDiff:
    for name, table in (("anchor", anchor), ("candidate", candidate)):
        if not isinstance(table, RegistrationTable):
            raise TypeError(f"{name} must be a RegistrationTable")
    if anchor.kind != candidate.kind:
        raise ValueError(
            f"cannot diff a {anchor.kind!r} table against a {candidate.kind!r} one")
    a, c = anchor.by_ident(), candidate.by_ident()
    removed = tuple(sorted(set(a) - set(c)))
    added = tuple(sorted(set(c) - set(a)))
    common = set(a) & set(c)
    arity_changed = tuple(sorted(
        (ident[0], ident[1], a[ident].arity, c[ident].arity)
        for ident in common
        if a[ident].arity is not None and c[ident].arity is not None
        and a[ident].arity != c[ident].arity))
    arity_not_comparable = tuple(sorted(
        (ident[0], ident[1], a[ident].arity, c[ident].arity)
        for ident in common
        if (a[ident].arity is None) != (c[ident].arity is None)))
    return RegistrationDiff(removed=removed, added=added, arity_changed=arity_changed,
                            arity_not_comparable=arity_not_comparable,
                            anchor_count=len(a), candidate_count=len(c))


_GATE_ID_BY_REGISTRATION_KIND = {
    KIND_OP_REGISTRATION: GATE_OP_REGISTRATION,
    KIND_DISPATCH_PREDICATE: GATE_DISPATCH_PREDICATE,
}


def check_registration_preservation(anchor: Optional[RegistrationTable],
                                    candidate: Optional[RegistrationTable],
                                    declared: DeclaredSymbolDeltas,
                                    *, expected_kind: Optional[str] = None
                                    ) -> api.GateResult:
    """§8.5.1 (1): an op registration or dispatch case removed but not declared FAILS.

    *"A kernel edit that … deletes a case from a dispatch switch, or removes an op
    registration compiles cleanly and silently changes behaviour for every shape
    nobody happened to test."*

    `expected_kind` is what names the gate when BOTH tables are absent. Without it,
    two absent kinds would both fall through to the op-registration gate id and the
    record would carry one gate twice while the dispatch-predicate gate silently
    vanished — so the ambiguous case RAISES rather than guessing.
    """
    kinds = {t.kind for t in (anchor, candidate) if t is not None}
    if expected_kind is not None:
        if expected_kind not in _REGISTRATION_KINDS:
            raise ValueError(f"expected_kind: {expected_kind!r} must be one of "
                             f"{list(_REGISTRATION_KINDS)}")
        if kinds - {expected_kind}:
            raise ValueError(
                f"expected_kind={expected_kind!r} but the supplied table(s) are "
                f"{sorted(kinds)}")
        kind = expected_kind
    elif len(kinds) == 1:
        kind = kinds.pop()
    elif not kinds:
        raise ValueError(
            "both tables are None and no expected_kind was given, so this gate has no "
            "identity; pass expected_kind so the missing table is reported under its "
            "own gate id instead of collapsing onto another one")
    else:
        raise ValueError(f"anchor and candidate tables disagree on kind: {sorted(kinds)}")

    gate_id = _GATE_ID_BY_REGISTRATION_KIND[kind]
    if anchor is None or candidate is None:
        missing = "anchor" if anchor is None else "candidate"
        return api.GateResult(
            gate_id=gate_id, gate_class=api.GATE_INTEGRITY,
            check=_cnc("REGISTRATION_TABLE_UNAVAILABLE",
                       f"no {missing} {kind} table was extracted"),
            requires_anchor=True)

    diff = diff_registration_tables(anchor, candidate)
    if diff.anchor_count == 0:
        return api.GateResult(
            gate_id=gate_id, gate_class=api.GATE_INTEGRITY,
            check=_cnc(F_EMPTY_ANCHOR_REGISTRATION_TABLE,
                       f"the anchor {anchor.kind} table extracted by "
                       f"{anchor.extractor_id!r} is empty; an empty anchor table has no "
                       "removals to find and cannot evidence preservation"),
            requires_anchor=True)

    checks: list = []
    for registry, key in diff.removed:
        if key in declared.removed or f"{registry}:{key}" in declared.removed:
            continue
        checks.append(_fail(
            F_UNDECLARED_REGISTRATION_REMOVAL,
            f"{anchor.kind} {key!r} is registered in {registry!r} by the anchor and "
            "absent from the candidate, and is not in "
            "proposal.declared_symbol_deltas.removed"))
    for registry, key, old, new in diff.arity_changed:
        if key in declared.arity_changed or f"{registry}:{key}" in declared.arity_changed:
            continue
        checks.append(_fail(
            F_UNDECLARED_REGISTRATION_ARITY_CHANGE,
            f"{anchor.kind} {key!r} in {registry!r} changed arity {old} -> {new} and is "
            "not in proposal.declared_symbol_deltas.arity_changed"))
    for registry, key, old, new in diff.arity_not_comparable:
        if key in declared.arity_changed or f"{registry}:{key}" in declared.arity_changed:
            continue
        missing = "candidate" if new is None else "anchor"
        checks.append(_cnc(
            F_REGISTRATION_ARITY_NOT_DERIVABLE,
            f"{anchor.kind} {key!r} in {registry!r} has arity {old} on the anchor side "
            f"and {new} on the candidate side: the declared extraction pattern did not "
            f"capture an arity on the {missing} side, so an arity change here is NOT "
            "ruled out. A missing arity is not an unchanged one — supply a pattern whose "
            "'arity' group matches both forms, or declare the entry in "
            "proposal.declared_symbol_deltas.arity_changed"))

    notes = [f"compared {diff.anchor_count} anchor and {diff.candidate_count} candidate "
             f"{anchor.kind} entries via {anchor.extractor_id!r}"]
    if diff.added:
        notes.append(f"{len(diff.added)} entr(ies) added: "
                     f"{[f'{r}:{k}' for r, k in diff.added]} (recorded, not failed)")
    return api.GateResult(
        gate_id=gate_id, gate_class=api.GATE_INTEGRITY,
        check=_worst(checks) if checks else schemas.Check(PASS),
        requires_anchor=True, notes=tuple(notes))


# =============================================================================
# §8.5.1 (2) — clean build from the recorded snapshot
# =============================================================================

@dataclass(frozen=True)
class TreeDigest:
    """A content-addressed digest of a source tree, plus the manifest it hashed."""

    sha256: str
    file_count: int
    total_bytes: int
    entries: tuple      # ((mode, sha256, relpath), ...) — complete, never capped

    def to_dict(self) -> dict:
        return {"sha256": self.sha256, "file_count": self.file_count,
                "total_bytes": self.total_bytes,
                "entries": [list(e) for e in self.entries],
                "listing_is_complete": True}


def _manifest_digest(lines: Sequence[str]) -> str:
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


#: The digest of an EMPTY tree. A fresh build directory must hash to this.
EMPTY_TREE_SHA256 = _manifest_digest([])


def hash_source_tree(root: Any, *, exclude_dir_names: Sequence[str] = (),
                     max_file_bytes: Optional[int] = None) -> TreeDigest:
    """Content-address a source tree, deterministically and without a subprocess.

    Manifest line per entry: `<mode>\\t<sha256>\\t<relpath>\\n`, sorted by POSIX
    relative path; `mode` is `100755`/`100644` for files and `120000` for a
    symlink, whose "content" is its target string. The digest is SHA-256 over the
    concatenated manifest.

    Deliberate properties:
      * symlinks are RECORDED, never followed — following one would let a link
        out of the tree change the digest of the tree;
      * empty directories are not represented (git's rule), and that is stated
        here rather than discovered later;
      * an unreadable file RAISES (`TreeReadError`). Skipping it would produce a
        digest that matches a tree we never read.

    `exclude_dir_names` defaults to EXCLUDING NOTHING. A default `.git` exclusion
    would be a silent behaviour with a security-shaped consequence.
    """
    base = Path(root)
    if not base.is_dir():
        raise TreeReadError(f"{base}: not a directory (cannot hash a source tree)")
    excluded = set(exclude_dir_names)
    entries: list = []
    total = 0
    try:
        walk = sorted(base.rglob("*"))
    except OSError as exc:
        raise TreeReadError(f"{base}: unreadable ({exc})") from exc
    for path in walk:
        rel = path.relative_to(base)
        if excluded & set(rel.parts[:-1]):
            continue
        if path.is_symlink():
            if rel.name in excluded:
                continue
            try:
                target = str(path.readlink())
            except OSError as exc:
                raise TreeReadError(f"{path}: unreadable symlink ({exc})") from exc
            digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
            entries.append(("120000", digest, rel.as_posix()))
            total += len(target)
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise TreeReadError(
                f"{path}: not a regular file, directory or symlink; refusing to hash a "
                "tree containing an entry this digest cannot represent")
        if rel.name in excluded:
            continue
        try:
            size = path.stat().st_size
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        except OSError as exc:
            raise TreeReadError(f"{path}: unreadable ({exc})") from exc
        digest = sha256_file(path, max_bytes=max_file_bytes)
        entries.append((mode, digest, rel.as_posix()))
        total += size
    entries.sort(key=lambda e: e[2])
    lines = [f"{m}\t{d}\t{p}\n" for m, d, p in entries]
    return TreeDigest(sha256=_manifest_digest(lines), file_count=len(entries),
                      total_bytes=total, entries=tuple(entries))


def _lexically_normal_parts(path: Any) -> Optional[tuple]:
    """`path`'s parts with `.` dropped and `..` collapsed, or None if not absolute.

    Collapsing `..` lexically is deliberately the CONSERVATIVE direction for a
    containment test: where a symlink makes the lexical answer differ from the
    real one, the lexical answer keeps the path inside the tree it was written
    under, and this helper only ever feeds checks that FAIL on containment.
    Comparing raw `Path.parts` instead let `/prod/x/../llama.cpp/build` read as
    OUTSIDE `/prod/llama.cpp`, which is a build in a frozen production tree
    answering PASS.
    """
    try:
        p = Path(path)
    except TypeError:
        return None
    if not p.is_absolute():
        return None
    out: list = []
    for part in p.parts[1:]:
        if part == ".":
            continue
        if part == "..":
            if out:
                out.pop()
            continue
        out.append(part)
    return (p.parts[0],) + tuple(out)


def _refuse_unnormalized_path(value: str, label: str) -> str:
    """An attested path carrying `.` or `..` is refused at the door.

    Normalizing it here would be a guess about the caller's intent; refusing it
    is the fail-closed reading, and it removes the whole class of containment
    evasion rather than one instance of it.
    """
    if any(part in (".", "..") for part in Path(value).parts):
        raise ValueError(
            f"{label}: {value!r} contains a '.' or '..' segment. An unnormalized "
            "attested path cannot be tested for containment in the actor worktree or a "
            "production tree; record the resolved path")
    return value


def _is_within(child: str, parent: str) -> bool:
    """True when `child` is `parent` or is nested under it. Pure path algebra."""
    cp = _lexically_normal_parts(child)
    pp = _lexically_normal_parts(parent)
    if cp is None or pp is None:
        return False
    return len(cp) >= len(pp) and cp[:len(pp)] == pp


@dataclass(frozen=True)
class BuildProvenance:
    """Everything §8.5.1 (2) needs to decide the artifact came from the snapshot.

    No field has a default. `WindowAttestations` in `api.py` makes the same
    choice for the same reason: a defaulted attestation is an attestation nobody
    made, and this record's whole job is to be an attestation somebody made.
    """

    candidate_id: str
    snapshot_sha256: str
    source_root: str
    build_dir: str
    build_dir_created_for_this_build: bool
    build_dir_pre_build_digest: str
    actor_worktree: str
    production_tree_paths: tuple
    toolchain: str
    compiler: str
    command: str
    build_log_path: str
    build_log_sha256: str
    output_binary_sha256: str
    incremental_output_binary_sha256: Optional[str]

    def __post_init__(self) -> None:
        _require_str(self.candidate_id, "BuildProvenance.candidate_id")
        if not self.candidate_id.startswith("akc-"):
            raise ValueError("BuildProvenance.candidate_id must start with 'akc-'")
        _require_sha256(self.snapshot_sha256, "BuildProvenance.snapshot_sha256")
        _require_sha256(self.build_dir_pre_build_digest,
                        "BuildProvenance.build_dir_pre_build_digest")
        _require_sha256(self.build_log_sha256, "BuildProvenance.build_log_sha256")
        _require_sha256(self.output_binary_sha256, "BuildProvenance.output_binary_sha256")
        if self.incremental_output_binary_sha256 is not None:
            _require_sha256(self.incremental_output_binary_sha256,
                            "BuildProvenance.incremental_output_binary_sha256")
        _require_bool(self.build_dir_created_for_this_build,
                      "BuildProvenance.build_dir_created_for_this_build")
        for name in ("source_root", "build_dir", "actor_worktree"):
            value = _require_str(getattr(self, name), f"BuildProvenance.{name}")
            if not value.startswith("/"):
                raise ValueError(f"BuildProvenance.{name}: {value!r} must be absolute")
            _refuse_unnormalized_path(value, f"BuildProvenance.{name}")
        for name in ("toolchain", "compiler", "command", "build_log_path"):
            _require_str(getattr(self, name), f"BuildProvenance.{name}")
        if not isinstance(self.production_tree_paths, tuple):
            raise TypeError("BuildProvenance.production_tree_paths must be a tuple")
        for path in self.production_tree_paths:
            if not _require_str(path, "production_tree_paths[]").startswith("/"):
                raise ValueError(f"production_tree_paths: {path!r} must be absolute")
            _refuse_unnormalized_path(path, "production_tree_paths[]")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "snapshot_sha256": self.snapshot_sha256,
            "source_root": self.source_root,
            "build_dir": self.build_dir,
            "build_dir_created_for_this_build": self.build_dir_created_for_this_build,
            "build_dir_pre_build_digest": self.build_dir_pre_build_digest,
            "actor_worktree": self.actor_worktree,
            "production_tree_paths": list(self.production_tree_paths),
            "toolchain": self.toolchain, "compiler": self.compiler,
            "command": self.command, "build_log_path": self.build_log_path,
            "build_log_sha256": self.build_log_sha256,
            "output_binary_sha256": self.output_binary_sha256,
            "incremental_output_binary_sha256": self.incremental_output_binary_sha256,
        }


@dataclass(frozen=True)
class CleanBuildReceipt:
    """The receipt §8.5.1 (2) asks for, content-hashed so it can be journaled."""

    candidate_id: str
    snapshot_sha256: str
    snapshot_verification: str      # "recomputed" | "attested" | "unverified"
    snapshot_recomputed_sha256: Optional[str]
    build_dir: str
    fresh_build_dir: bool
    artifact_binary_sha256: str
    provenance: dict
    checks: tuple                   # ((name, outcome, (reason, ...)), ...)

    def to_dict(self) -> dict:
        return {
            "receipt": "autokernel.evaluator.integrity.clean_build/v1",
            "candidate_id": self.candidate_id,
            "snapshot_sha256": self.snapshot_sha256,
            "snapshot_verification": self.snapshot_verification,
            "snapshot_recomputed_sha256": self.snapshot_recomputed_sha256,
            "build_dir": self.build_dir,
            "fresh_build_dir": self.fresh_build_dir,
            "artifact_binary_sha256": self.artifact_binary_sha256,
            "provenance": self.provenance,
            "checks": [{"name": n, "outcome": o, "reasons": list(r)}
                       for n, o, r in self.checks],
        }

    @property
    def content_hash(self) -> str:
        return schemas.content_hash(self.to_dict())


def check_clean_build_from_snapshot(
        provenance: BuildProvenance,
        artifact_binary_sha256: str,
        *,
        recompute_root: Optional[Any],
        snapshot_attested_by: Optional[str]) -> tuple:
    """§8.5.1 (2). Returns `(GateResult, CleanBuildReceipt)`.

    *"T0 compiles from the content-addressed source snapshot in a fresh build
    directory — never from the actor's incremental tree. An incremental build can
    link stale objects and hide the error that the snapshot would surface, which
    would make the actor's build state part of the artifact."*

    Both `recompute_root` and `snapshot_attested_by` are REQUIRED parameters and
    may be `None`; passing `None` for both yields COULD_NOT_CHECK on the snapshot
    identity sub-check, naming what is missing. There is no third state in which
    the snapshot is assumed good.
    """
    if not isinstance(provenance, BuildProvenance):
        raise TypeError("provenance must be a BuildProvenance")
    _require_sha256(artifact_binary_sha256, "artifact_binary_sha256")

    named: list = []

    def record(name: str, chk: schemas.Check) -> schemas.Check:
        named.append((name, chk.outcome, tuple(chk.reasons)))
        return chk

    checks: list = []

    # (a) The artifact under test IS the clean build's output.
    if artifact_binary_sha256 != provenance.output_binary_sha256:
        checks.append(record("artifact_is_clean_build_output", _fail(
            F_ARTIFACT_NOT_FROM_CLEAN_BUILD,
            f"artifact {artifact_binary_sha256[:12]} != clean-build output "
            f"{provenance.output_binary_sha256[:12]}; the binary under test was not "
            "produced by the snapshot build")))
    else:
        checks.append(record("artifact_is_clean_build_output", schemas.Check(PASS)))

    # (b) …and is NOT the actor's incremental output. This is the red-team case
    #     "an incremental tree that compiles while its snapshot does not".
    inc = provenance.incremental_output_binary_sha256
    if inc is not None and inc == artifact_binary_sha256 and inc != provenance.output_binary_sha256:
        checks.append(record("artifact_not_incremental", _fail(
            F_ARTIFACT_FROM_INCREMENTAL_TREE,
            f"artifact {artifact_binary_sha256[:12]} equals the actor's INCREMENTAL "
            f"build output and differs from the clean-build output "
            f"{provenance.output_binary_sha256[:12]}")))
    elif inc is not None and inc != provenance.output_binary_sha256:
        checks.append(record("artifact_not_incremental", schemas.Check(PASS)))
        named[-1] = ("artifact_not_incremental", PASS, (
            "the incremental tree produced a different binary; the clean build's output "
            "is the one under test",))
    else:
        checks.append(record("artifact_not_incremental", schemas.Check(PASS)))

    # (c) The build directory was fresh.
    fresh = (provenance.build_dir_created_for_this_build
             and provenance.build_dir_pre_build_digest == EMPTY_TREE_SHA256)
    if not fresh:
        checks.append(record("fresh_build_dir", _fail(
            F_BUILD_DIR_NOT_FRESH,
            f"build_dir {provenance.build_dir!r} was not fresh: "
            f"created_for_this_build={provenance.build_dir_created_for_this_build}, "
            f"pre_build_digest={provenance.build_dir_pre_build_digest[:12]} "
            f"(empty tree is {EMPTY_TREE_SHA256[:12]})")))
    else:
        checks.append(record("fresh_build_dir", schemas.Check(PASS)))

    # (d) The build did not happen inside the actor's worktree.
    if _is_within(provenance.build_dir, provenance.actor_worktree):
        checks.append(record("build_dir_outside_actor_worktree", _fail(
            F_BUILD_DIR_INSIDE_ACTOR_WORKTREE,
            f"build_dir {provenance.build_dir!r} is inside the actor worktree "
            f"{provenance.actor_worktree!r}; the actor's build state would become part "
            "of the artifact")))
    else:
        checks.append(record("build_dir_outside_actor_worktree", schemas.Check(PASS)))

    # (e) Nothing was built in a production tree (invariant 3, "frozen means immutable").
    offending = [t for t in provenance.production_tree_paths
                 if _is_within(provenance.build_dir, t) or _is_within(provenance.source_root, t)]
    if offending:
        checks.append(record("no_production_tree_build", _fail(
            F_BUILD_IN_PRODUCTION_TREE,
            f"build_dir/source_root resolve inside production tree(s) {offending}; "
            "no actor builds in or modifies a production tree (invariant 3)")))
    else:
        checks.append(record("no_production_tree_build", schemas.Check(PASS)))

    # (f) The snapshot the build consumed IS the recorded snapshot.
    verification = "unverified"
    recomputed: Optional[str] = None
    if recompute_root is not None:
        try:
            recomputed = hash_source_tree(recompute_root).sha256
        except TreeReadError as exc:
            checks.append(record("snapshot_identity", _cnc(
                F_SNAPSHOT_NOT_VERIFIED,
                f"could not recompute the snapshot digest at {recompute_root!r}: {exc}")))
        else:
            verification = "recomputed"
            if recomputed != provenance.snapshot_sha256:
                checks.append(record("snapshot_identity", _fail(
                    F_SNAPSHOT_DIGEST_MISMATCH,
                    f"recomputed tree digest {recomputed[:12]} != recorded "
                    f"snapshot_sha256 {provenance.snapshot_sha256[:12]}; the tree that "
                    "built is not the tree that was recorded")))
            else:
                checks.append(record("snapshot_identity", schemas.Check(PASS)))
    elif snapshot_attested_by is not None:
        verification = "attested"
        _require_str(snapshot_attested_by, "snapshot_attested_by")
        checks.append(record("snapshot_identity", schemas.Check(PASS)))
        named[-1] = ("snapshot_identity", PASS,
                     (f"snapshot identity attested by {snapshot_attested_by!r} rather "
                      "than recomputed here",))
    else:
        checks.append(record("snapshot_identity", _cnc(
            F_SNAPSHOT_NOT_VERIFIED,
            "the snapshot digest was neither recomputed (recompute_root=None) nor "
            "attested (snapshot_attested_by=None); supply one of the two")))

    receipt = CleanBuildReceipt(
        candidate_id=provenance.candidate_id,
        snapshot_sha256=provenance.snapshot_sha256,
        snapshot_verification=verification,
        snapshot_recomputed_sha256=recomputed,
        build_dir=provenance.build_dir,
        fresh_build_dir=fresh,
        artifact_binary_sha256=artifact_binary_sha256,
        provenance=provenance.to_dict(),
        checks=tuple(named),
    )
    gate = api.GateResult(
        gate_id=GATE_CLEAN_BUILD, gate_class=api.GATE_INTEGRITY,
        check=_worst(checks), requires_anchor=False,
        evidence_ref=receipt.content_hash,
        notes=(f"build receipt {receipt.content_hash[:12]}; snapshot verification: "
               f"{verification}",))
    return gate, receipt


# =============================================================================
# §8.5.1 (3) — semantic diff conformance, and the §10.6 complexity ceiling
# =============================================================================

@dataclass(frozen=True)
class FileDiff:
    """One file's change, with the numbers the envelope checks are computed from."""

    path: str
    old_path: Optional[str]
    added_lines: int
    removed_lines: int
    hunks: int
    is_new_file: bool
    is_deleted_file: bool
    is_rename: bool
    is_binary: bool
    observed_old_extent: int   # highest old line number any hunk reached; a LOWER
                               # bound on the pre-change file length

    def __post_init__(self) -> None:
        _require_str(self.path, "FileDiff.path")
        for name in ("added_lines", "removed_lines", "hunks", "observed_old_extent"):
            _require_int(getattr(self, name), f"FileDiff.{name}")

    @property
    def changed_lines(self) -> int:
        return self.added_lines + self.removed_lines

    @property
    def is_pure_deletion(self) -> bool:
        return self.removed_lines > 0 and self.added_lines == 0

    def to_dict(self) -> dict:
        return {"path": self.path, "old_path": self.old_path,
                "added_lines": self.added_lines, "removed_lines": self.removed_lines,
                "hunks": self.hunks, "is_new_file": self.is_new_file,
                "is_deleted_file": self.is_deleted_file, "is_rename": self.is_rename,
                "is_binary": self.is_binary,
                "observed_old_extent": self.observed_old_extent}


@dataclass(frozen=True)
class SourceDiff:
    files: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple):
            raise TypeError("SourceDiff.files must be a tuple")
        for f in self.files:
            if not isinstance(f, FileDiff):
                raise TypeError("SourceDiff.files must hold FileDiff")

    @property
    def files_touched(self) -> int:
        return len(self.files)

    @property
    def total_added(self) -> int:
        return sum(f.added_lines for f in self.files)

    @property
    def total_removed(self) -> int:
        return sum(f.removed_lines for f in self.files)

    @property
    def total_changed(self) -> int:
        return self.total_added + self.total_removed

    @property
    def total_hunks(self) -> int:
        return sum(f.hunks for f in self.files)

    def paths(self) -> frozenset:
        out = set()
        for f in self.files:
            out.add(f.path)
            if f.old_path:
                out.add(f.old_path)
        return frozenset(out)

    def to_dict(self) -> dict:
        return {"files": [f.to_dict() for f in self.files],
                "files_touched": self.files_touched,
                "total_added": self.total_added, "total_removed": self.total_removed,
                "total_hunks": self.total_hunks}


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_GIT_RE = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$")


def parse_unified_diff(text: str) -> SourceDiff:
    """Parse a unified / `git diff` text into a `SourceDiff`.

    Raises `DiffParseError` on anything it cannot account for, including a hunk
    whose body does not match its header's line counts. A diff we cannot parse is
    not a small diff, and silently under-counting it is how a large edit sails
    through a size envelope.

    Note: this parses TEXT the caller supplies. It does not run `git`.
    """
    if not isinstance(text, str):
        raise TypeError("parse_unified_diff expects str")
    lines = text.splitlines()
    files: list = []

    cur: Optional[dict] = None

    def flush() -> None:
        if cur is None:
            return
        if cur["path"] is None:
            raise DiffParseError("a file section ended without a +++ / --- path")
        files.append(FileDiff(
            path=cur["path"], old_path=cur["old_path"],
            added_lines=cur["added"], removed_lines=cur["removed"],
            hunks=cur["hunks"], is_new_file=cur["new"], is_deleted_file=cur["deleted"],
            is_rename=cur["rename"], is_binary=cur["binary"],
            observed_old_extent=cur["extent"]))

    i = 0
    pending: Optional[dict] = None
    while i < len(lines):
        line = lines[i]
        m = _DIFF_GIT_RE.match(line)
        if m:
            flush()
            cur = {"path": None, "old_path": None, "added": 0, "removed": 0,
                   "hunks": 0, "new": False, "deleted": False, "rename": False,
                   "binary": False, "extent": 0,
                   "a": m.group("a"), "b": m.group("b")}
            pending = cur
            i += 1
            continue
        if line.startswith("new file mode"):
            if cur is None:
                raise DiffParseError(f"'{line}' outside a file section")
            cur["new"] = True
            i += 1
            continue
        if line.startswith("deleted file mode"):
            if cur is None:
                raise DiffParseError(f"'{line}' outside a file section")
            cur["deleted"] = True
            i += 1
            continue
        if line.startswith("rename from ") or line.startswith("rename to "):
            if cur is None:
                raise DiffParseError(f"'{line}' outside a file section")
            cur["rename"] = True
            i += 1
            continue
        if line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            if cur is None:
                raise DiffParseError(f"'{line}' outside a file section")
            cur["binary"] = True
            if cur["path"] is None:
                cur["path"] = cur.get("b") or cur.get("a")
                cur["old_path"] = cur.get("a") if cur.get("a") != cur.get("b") else None
            i += 1
            continue
        if line.startswith("--- "):
            if cur is None:
                flush()
                cur = {"path": None, "old_path": None, "added": 0, "removed": 0,
                       "hunks": 0, "new": False, "deleted": False, "rename": False,
                       "binary": False, "extent": 0, "a": None, "b": None}
                pending = cur
            old = line[4:].split("\t")[0]
            cur["old_path"] = None if old == "/dev/null" else _strip_prefix(old)
            i += 1
            continue
        if line.startswith("+++ "):
            if cur is None:
                raise DiffParseError("'+++' without a preceding '---'")
            new = line[4:].split("\t")[0]
            if new == "/dev/null":
                cur["deleted"] = True
                if cur["old_path"] is None:
                    raise DiffParseError("a deletion with no old path")
                cur["path"] = cur["old_path"]
                cur["old_path"] = None
            else:
                cur["path"] = _strip_prefix(new)
                if cur["old_path"] is None:
                    cur["new"] = True
                elif cur["old_path"] == cur["path"]:
                    cur["old_path"] = None
            i += 1
            continue
        hm = _HUNK_RE.match(line)
        if hm:
            if cur is None or cur["path"] is None:
                raise DiffParseError(f"hunk header outside a file section: {line!r}")
            old_start = int(hm.group(1))
            old_count = int(hm.group(2)) if hm.group(2) is not None else 1
            new_count = int(hm.group(4)) if hm.group(4) is not None else 1
            cur["hunks"] += 1
            cur["extent"] = max(cur["extent"], old_start + old_count - 1)
            i += 1
            seen_old = seen_new = 0
            while i < len(lines) and (seen_old < old_count or seen_new < new_count):
                body = lines[i]
                if body.startswith("\\"):
                    i += 1
                    continue
                if body.startswith("+"):
                    cur["added"] += 1
                    seen_new += 1
                elif body.startswith("-"):
                    cur["removed"] += 1
                    seen_old += 1
                elif body.startswith(" ") or body == "":
                    seen_old += 1
                    seen_new += 1
                else:
                    raise DiffParseError(
                        f"unexpected line inside a hunk of {cur['path']!r}: {body!r}")
                i += 1
            if seen_old != old_count or seen_new != new_count:
                raise DiffParseError(
                    f"hunk of {cur['path']!r} declared -{old_count}/+{new_count} but the "
                    f"body contained -{seen_old}/+{seen_new}")
            continue
        if line.startswith("index ") or line.startswith("similarity index") \
                or line.startswith("old mode") or line.startswith("new mode") \
                or line.startswith("copy from") or line.startswith("copy to"):
            i += 1
            continue
        if cur is None or pending is None:
            # Leading commentary before the first file section.
            i += 1
            continue
        i += 1
    flush()
    if not files and any(line.strip() for line in lines):
        # An EMPTY input is a legitimate "no changes" diff. Non-blank text that
        # yielded no file section is not: it is text this parser could not
        # account for, and returning an empty SourceDiff for it would report a
        # zero-file, zero-line, no-core-path change — the smallest diff there is.
        raise DiffParseError(
            f"{F_UNPARSEABLE_DIFF}: the input has {sum(1 for ln in lines if ln.strip())} "
            "non-blank line(s) but yielded no file section ('diff --git', or '---' "
            "followed by '+++'). A diff this parser cannot account for is not an empty "
            "diff, and an empty diff touches no declared surface and no core header")
    return SourceDiff(files=tuple(files))


def _strip_prefix(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


@dataclass(frozen=True)
class DeclaredSurface:
    """`proposal.change.files_and_symbols` split into files and symbols.

    An EMPTY declared surface is legal and self-punishing: every touched file is
    then undeclared and the gate FAILs. An ABSENT declaration raises, because the
    two are different facts.
    """

    files: frozenset
    symbols: frozenset

    def __post_init__(self) -> None:
        for name in ("files", "symbols"):
            if not isinstance(getattr(self, name), frozenset):
                raise TypeError(f"DeclaredSurface.{name} must be a frozenset")

    @classmethod
    def from_proposal(cls, proposal: Mapping[str, Any]) -> "DeclaredSurface":
        """Split by an explicit rule: `path::symbol` declares both; an entry with a
        `/` or a source suffix is a file; anything else is a symbol."""
        if not isinstance(proposal, Mapping):
            raise TypeError("proposal must be a mapping")
        change = proposal.get("change")
        if not isinstance(change, Mapping) or "files_and_symbols" not in change:
            raise DeclarationMissing(
                "proposal.change.files_and_symbols is absent; the declared surface is "
                "what the conformance gate compares the diff against, so its absence "
                "raises rather than defaulting to 'everything' or 'nothing'")
        raw = change["files_and_symbols"]
        if not isinstance(raw, (list, tuple)):
            raise DeclarationMissing(
                "proposal.change.files_and_symbols must be a list of strings")
        files: set = set()
        symbols: set = set()
        for entry in raw:
            _require_str(entry, "change.files_and_symbols[]")
            if "::" in entry:
                head, _, tail = entry.partition("::")
                files.add(head)
                symbols.add(tail)
            elif "/" in entry or Path(entry).suffix:
                files.add(entry)
            else:
                symbols.add(entry)
        return cls(files=frozenset(files), symbols=frozenset(symbols))

    def to_dict(self) -> dict:
        return {"files": sorted(self.files), "symbols": sorted(self.symbols)}


@dataclass(frozen=True)
class ChangeClassEnvelope:
    """The §8.5.1 (3) change-class size envelope, DECLARED by a backend adapter.

    `max_file_shrinkage_ratio` is the direct port of AutoPilot's >60% shrinkage
    reject — the defense that would have stopped `escalation.py` going from 454
    lines to 3. Exceeding the envelope is a conformance FAILURE, which is
    distinct from exceeding the §10.6 complexity ceiling (that marks
    `REQUIRES_HUMAN_CODE_REVIEW` and does not fail).
    """

    change_class: str
    max_files_touched: int
    max_changed_lines: int
    max_hunks: int
    max_file_shrinkage_ratio: float
    allows_file_creation: bool
    allows_file_deletion: bool
    allows_pure_deletion_hunks: bool
    declared_by: str

    def __post_init__(self) -> None:
        if self.change_class not in schemas.CHANGE_CLASSES:
            raise ValueError(f"change_class: {self.change_class!r} is not one of "
                             f"{sorted(schemas.CHANGE_CLASSES)}")
        for name in ("max_files_touched", "max_changed_lines", "max_hunks"):
            _require_int(getattr(self, name), f"ChangeClassEnvelope.{name}", minimum=1)
        ratio = self.max_file_shrinkage_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("max_file_shrinkage_ratio must be a number")
        if not 0.0 < float(ratio) <= 1.0:
            raise ValueError(
                f"max_file_shrinkage_ratio: {ratio!r} must be in (0, 1]; AutoPilot's "
                "port of this defense rejected above 0.60")
        for name in ("allows_file_creation", "allows_file_deletion",
                     "allows_pure_deletion_hunks"):
            _require_bool(getattr(self, name), f"ChangeClassEnvelope.{name}")
        _require_str(self.declared_by, "ChangeClassEnvelope.declared_by")

    def to_dict(self) -> dict:
        return {"change_class": self.change_class,
                "max_files_touched": self.max_files_touched,
                "max_changed_lines": self.max_changed_lines,
                "max_hunks": self.max_hunks,
                "max_file_shrinkage_ratio": float(self.max_file_shrinkage_ratio),
                "allows_file_creation": self.allows_file_creation,
                "allows_file_deletion": self.allows_file_deletion,
                "allows_pure_deletion_hunks": self.allows_pure_deletion_hunks,
                "declared_by": self.declared_by}


def envelope_for(envelopes: Mapping[str, ChangeClassEnvelope],
                 change_class: str) -> ChangeClassEnvelope:
    """Look up an envelope. Raises rather than defaulting — §10.6 says the adapter
    declares the ceiling, and a defaulted ceiling is one nobody chose."""
    if not isinstance(envelopes, Mapping):
        raise TypeError("envelopes must be a mapping of change_class -> envelope")
    try:
        env = envelopes[change_class]
    except KeyError as exc:
        raise EnvelopeNotDeclared(
            f"no change-class envelope declared for {change_class!r}; declared classes "
            f"are {sorted(envelopes)}") from exc
    if not isinstance(env, ChangeClassEnvelope):
        raise TypeError(f"envelopes[{change_class!r}] must be a ChangeClassEnvelope")
    return env


@dataclass(frozen=True)
class ComplexityCeiling:
    """§10.6, declared per backend adapter. Above it: `REQUIRES_HUMAN_CODE_REVIEW`."""

    backend: str
    max_diff_lines: int
    max_files_touched: int
    shared_core_modification_requires_review: bool
    declared_by: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise ValueError(f"backend: {self.backend!r} is not one of "
                             f"{sorted(schemas.BACKENDS)}")
        for name in ("max_diff_lines", "max_files_touched"):
            _require_int(getattr(self, name), f"ComplexityCeiling.{name}", minimum=1)
        _require_bool(self.shared_core_modification_requires_review,
                      "ComplexityCeiling.shared_core_modification_requires_review")
        _require_str(self.declared_by, "ComplexityCeiling.declared_by")

    def to_dict(self) -> dict:
        return {"backend": self.backend, "max_diff_lines": self.max_diff_lines,
                "max_files_touched": self.max_files_touched,
                "shared_core_modification_requires_review":
                    self.shared_core_modification_requires_review,
                "declared_by": self.declared_by}


REQUIRES_HUMAN_CODE_REVIEW = "REQUIRES_HUMAN_CODE_REVIEW"


@dataclass(frozen=True)
class ComplexityAssessment:
    requires_human_code_review: bool
    reasons: tuple
    first_page_notice: Optional[str]
    measured: dict

    def to_dict(self) -> dict:
        return {"requires_human_code_review": self.requires_human_code_review,
                "reasons": list(self.reasons),
                "first_page_notice": self.first_page_notice,
                "measured": self.measured}


def assess_complexity_ceiling(diff: SourceDiff, ceiling: ComplexityCeiling, *,
                              touches_shared_core: bool,
                              change_class: str) -> ComplexityAssessment:
    """§10.6: mark, do not fail. *"Above it, the package is marked
    `REQUIRES_HUMAN_CODE_REVIEW` and says so on its first page."*"""
    if not isinstance(diff, SourceDiff):
        raise TypeError("diff must be a SourceDiff")
    if not isinstance(ceiling, ComplexityCeiling):
        raise TypeError("ceiling must be a ComplexityCeiling")
    _require_bool(touches_shared_core, "touches_shared_core")
    reasons: list = []
    if diff.total_changed > ceiling.max_diff_lines:
        reasons.append(
            f"diff is {diff.total_changed} changed lines, above the {ceiling.backend} "
            f"ceiling of {ceiling.max_diff_lines} declared by {ceiling.declared_by!r}")
    if diff.files_touched > ceiling.max_files_touched:
        reasons.append(
            f"diff touches {diff.files_touched} files, above the {ceiling.backend} "
            f"ceiling of {ceiling.max_files_touched}")
    if touches_shared_core and ceiling.shared_core_modification_requires_review:
        reasons.append(
            "the diff modifies shared ggml core / a widely-included header, which the "
            f"{ceiling.backend} adapter declares review-requiring regardless of size")
    if change_class == "core_header":
        reasons.append(
            "change_class is 'core_header', which requires human code review "
            "regardless of the §10.6 complexity ceiling (§8.5.1 core-header risk tier)")
    notice = None
    if reasons:
        notice = (f"{REQUIRES_HUMAN_CODE_REVIEW} — " + "; ".join(reasons))
    return ComplexityAssessment(
        requires_human_code_review=bool(reasons), reasons=tuple(reasons),
        first_page_notice=notice,
        measured={"total_changed_lines": diff.total_changed,
                  "files_touched": diff.files_touched,
                  "touches_shared_core": touches_shared_core,
                  "change_class": change_class})


def check_semantic_diff_conformance(
        diff: SourceDiff,
        declared: DeclaredSurface,
        envelope: ChangeClassEnvelope,
        *,
        original_line_counts: Optional[Mapping[str, int]]) -> api.GateResult:
    """§8.5.1 (3): only the declared surface, no unrelated deletions, inside the
    change-class size envelope.

    *"Invariant 13 says one conceptual mutation; this is what enforces it rather
    than trusting it."*

    Shrinkage, and why it has three outcomes. AutoPilot's Python defense compared
    the removed lines against the KNOWN file length. A unified diff does not carry
    the file length, only the extent its hunks reached, which is a LOWER bound. So:
      * if `removed / observed_extent` is already inside the ceiling the result is
        conclusive (a longer true file only lowers the ratio) -> PASS;
      * if `original_line_counts` supplies the true length, the answer is exact ->
        PASS or FAIL;
      * otherwise the upper bound breaches the ceiling and nothing here can tell
        whether the true ratio does -> COULD_NOT_CHECK, naming the file and the
        input that would settle it.
    """
    if not isinstance(diff, SourceDiff):
        raise TypeError("diff must be a SourceDiff")
    if not isinstance(declared, DeclaredSurface):
        raise TypeError("declared must be a DeclaredSurface")
    if not isinstance(envelope, ChangeClassEnvelope):
        raise TypeError("envelope must be a ChangeClassEnvelope")
    if original_line_counts is not None and not isinstance(original_line_counts, Mapping):
        raise TypeError("original_line_counts must be a mapping of path -> int or None")

    if not diff.files:
        # The same doctrine as the empty anchor symbol table: a diff of nothing
        # touches no undeclared file, deletes nothing and shrinks nothing, so it
        # satisfies every clause of this gate by having no content to judge.
        return api.GateResult(
            gate_id=GATE_SEMANTIC_DIFF, gate_class=api.GATE_INTEGRITY,
            check=_cnc(F_EMPTY_DIFF,
                       "the supplied diff touches no file; an empty diff cannot evidence "
                       "that the change stayed inside the declared surface, and a "
                       "candidate whose binary differs from the anchor did not arise "
                       "from one"),
            requires_anchor=False,
            notes=(f"envelope {envelope.change_class!r} declared by "
                   f"{envelope.declared_by!r}",))

    checks: list = []
    notes: list = [f"envelope {envelope.change_class!r} declared by "
                   f"{envelope.declared_by!r}"]

    # (a) Only the declared surface.
    undeclared = sorted(p for p in diff.paths() if p not in declared.files)
    if undeclared:
        checks.append(_fail(
            F_UNDECLARED_FILE_TOUCHED,
            f"the diff touches {len(undeclared)} file(s) outside the declared surface: "
            f"{undeclared}. Declared files: {sorted(declared.files)}"))

    # (b) No unrelated deletions.
    for f in diff.files:
        if f.is_deleted_file and not envelope.allows_file_deletion:
            checks.append(_fail(
                F_UNDECLARED_FILE_DELETED,
                f"{f.path!r} is deleted, and the {envelope.change_class!r} envelope does "
                "not allow file deletion"))
        if f.is_new_file and not envelope.allows_file_creation:
            checks.append(_fail(
                F_UNDECLARED_FILE_CREATED,
                f"{f.path!r} is created, and the {envelope.change_class!r} envelope does "
                "not allow file creation"))
        if (f.is_pure_deletion and not f.is_deleted_file
                and not envelope.allows_pure_deletion_hunks):
            checks.append(_fail(
                F_PURE_DELETION_HUNK,
                f"{f.path!r} has {f.removed_lines} removed and 0 added lines — a pure "
                f"deletion — and the {envelope.change_class!r} envelope does not allow "
                "one"))
        if f.is_binary:
            checks.append(_cnc(
                F_BINARY_FILE_IN_DIFF,
                f"{f.path!r} is a binary file in the diff; its line-level conformance "
                "and shrinkage cannot be assessed from a unified diff"))

    # (c) Shrinkage — AutoPilot's >60% reject, ported.
    ratio_ceiling = float(envelope.max_file_shrinkage_ratio)
    for f in diff.files:
        if f.is_binary or f.removed_lines == 0 or f.is_new_file:
            continue
        true_length = None
        if original_line_counts is not None and f.path in original_line_counts:
            true_length = _require_int(original_line_counts[f.path],
                                       f"original_line_counts[{f.path!r}]", minimum=1)
        if f.is_deleted_file:
            # A whole-file deletion is 100% shrinkage by construction; it is already
            # judged by the deletion rule above, so it is not double-counted here.
            continue
        if true_length is not None:
            ratio = f.removed_lines / true_length
            if ratio > ratio_ceiling:
                checks.append(_fail(
                    F_EXCESSIVE_SHRINKAGE,
                    f"{f.path!r} removes {f.removed_lines} of {true_length} lines "
                    f"({ratio:.1%}), above the declared ceiling of {ratio_ceiling:.1%}. "
                    "This is the check that would have caught escalation.py going from "
                    "454 lines to 3"))
            continue
        extent = f.observed_old_extent
        if extent <= 0:
            checks.append(_cnc(
                F_SHRINKAGE_NOT_DERIVABLE,
                f"{f.path!r}: no hunk extent and no original_line_counts entry; the "
                "shrinkage ratio has neither an exact value nor a bound"))
            continue
        upper = f.removed_lines / extent
        if upper <= ratio_ceiling:
            continue  # conclusive: the true ratio can only be lower
        checks.append(_cnc(
            F_SHRINKAGE_NOT_DERIVABLE,
            f"{f.path!r} removes {f.removed_lines} lines and its hunks reach old line "
            f"{extent}, an UPPER bound of {upper:.1%} on shrinkage, above the "
            f"{ratio_ceiling:.1%} ceiling. The true ratio needs the pre-change file "
            f"length: supply original_line_counts[{f.path!r}]"))

    # (d) Inside the change-class size envelope.
    if diff.files_touched > envelope.max_files_touched:
        checks.append(_fail(
            F_ENVELOPE_FILES_EXCEEDED,
            f"{diff.files_touched} files touched, envelope allows "
            f"{envelope.max_files_touched}"))
    if diff.total_changed > envelope.max_changed_lines:
        checks.append(_fail(
            F_ENVELOPE_LINES_EXCEEDED,
            f"{diff.total_changed} changed lines, envelope allows "
            f"{envelope.max_changed_lines}"))
    if diff.total_hunks > envelope.max_hunks:
        checks.append(_fail(
            F_ENVELOPE_HUNKS_EXCEEDED,
            f"{diff.total_hunks} hunks, envelope allows {envelope.max_hunks}. Invariant "
            "13: one conceptual mutation per proposal"))

    notes.append(f"measured: {diff.files_touched} file(s), +{diff.total_added}/"
                 f"-{diff.total_removed} lines, {diff.total_hunks} hunk(s)")
    if original_line_counts is None:
        notes.append("original_line_counts was not supplied; shrinkage is bounded from "
                     "hunk extents where that is conclusive and COULD_NOT_CHECK where "
                     "it is not")
    return api.GateResult(
        gate_id=GATE_SEMANTIC_DIFF, gate_class=api.GATE_INTEGRITY,
        check=_worst(checks) if checks else schemas.Check(PASS),
        requires_anchor=False, notes=tuple(notes))


# =============================================================================
# §8.5.1 — the core_header risk tier
# =============================================================================

SURFACE_FULL_TREE = "full_tree"
SURFACE_PARTIAL = "partial"
_SURFACE_SCOPES = (SURFACE_FULL_TREE, SURFACE_PARTIAL)

#: `surface.py` DERIVES the affected-surface manifest and records whether the
#: derivation went full-tree; this module takes the scope as a caller-supplied
#: string and FAILs a core_header tier that under-declares it. Two names for one
#: fact, written by two authors, with nothing between them: a caller that derived
#: `full_tree=True` in `surface` and passed `SURFACE_PARTIAL` here got a PASS.
#: `surface_scope_for()` is the projection so the string cannot be hand-typed,
#: and `check_declared_surface_scope()` is the gate that says so when it was.
#:
#: The import-time assertion below makes the coupling loud: if `surface.py` ever
#: grows a third scope, this module refuses to load rather than silently mapping
#: it onto `partial`.
_SURFACE_FULL_TREE_CLASSES = frozenset(ak_surface.FULL_TREE_CHANGE_CLASSES)
if not _SURFACE_FULL_TREE_CLASSES:  # pragma: no cover - import-time contract assertion
    raise ImportError(
        "surface.FULL_TREE_CHANGE_CLASSES is empty; integrity's core_header risk tier "
        "derives its full-tree requirement from the same set, and an empty set would "
        "classify every core-header edit as partial-scope")


def surface_scope_for(derived: Any) -> str:
    """Project a `surface.AffectedSurface` into `declared_surface_scope`.

    RAISES on anything else. There is deliberately no default and no `None`
    branch: a caller with no derived manifest must say so by not calling this,
    and `check_declared_surface_scope` answers COULD_NOT_CHECK for that case.
    """
    if not isinstance(derived, ak_surface.AffectedSurface):
        raise TypeError(
            f"surface_scope_for() takes a surface.AffectedSurface, got "
            f"{type(derived).__name__}; the scope is DERIVED from the manifest and is "
            "not a string a caller supplies")
    return SURFACE_FULL_TREE if derived.full_tree else SURFACE_PARTIAL


def check_declared_surface_scope(declared_surface_scope: str,
                                 derived: Optional[Any]) -> schemas.Check:
    """Does the scope these inputs declare match the one `surface.py` derived?

    COULD_NOT_CHECK when no derived manifest is bound — an unchecked declaration
    is not a verified one, and this is exactly the state every run was in before
    the binding existed.
    """
    if declared_surface_scope not in _SURFACE_SCOPES:
        raise ValueError(f"declared_surface_scope: {declared_surface_scope!r} must be "
                         f"one of {list(_SURFACE_SCOPES)}")
    if derived is None:
        return _cnc(
            F_SURFACE_SCOPE_NOT_BOUND,
            f"declared_surface_scope={declared_surface_scope!r} is bound to no derived "
            "affected-surface manifest, so the declaration is unverified; §6.4 makes the "
            "surface a DERIVED manifest and invariant 18 makes a declaration a scored "
            "prediction, never a scope input")
    expected = surface_scope_for(derived)
    if expected != declared_surface_scope:
        return _fail(
            F_SURFACE_SCOPE_MISDECLARED,
            f"these inputs declare declared_surface_scope="
            f"{declared_surface_scope!r} but surface.derive_affected_surface() derived "
            f"{expected!r} for candidate {derived.candidate_id!r} "
            f"(full_tree={derived.full_tree}); the core-header tier's full-tree "
            "requirement is checked against the DECLARED string, so a mis-declared scope "
            "buys a PASS on a candidate whose surface is the whole tree")
    return schemas.Check(PASS, (
        f"declared_surface_scope={declared_surface_scope!r} matches the manifest derived "
        f"by surface.py ({derived.sha256()[:12]})",))


@dataclass(frozen=True)
class CoreHeaderPolicy:
    """Which paths count as shared ggml core / widely-included headers.

    Declared by the backend adapter, and matched MECHANICALLY against the diff.
    The actor's `change_class` is never the input that decides this: §6.4 and
    invariant 18 make the actor's declaration a scored prediction, and an actor
    that could opt out of the core-header tier by declaring `parameter` would
    control its own release scope.
    """

    core_path_prefixes: tuple
    core_path_globs: tuple
    backends_served: tuple
    declared_by: str

    def __post_init__(self) -> None:
        for name in ("core_path_prefixes", "core_path_globs", "backends_served"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"CoreHeaderPolicy.{name} must be a tuple")
        if not self.backends_served:
            raise ValueError(
                "backends_served must name every backend the source tree serves; "
                "core_header forces per-backend binary comparison for all of them (§3.2)")
        for backend in self.backends_served:
            if backend not in schemas.BACKENDS:
                raise ValueError(f"backends_served: {backend!r} is not a known backend")
        if not self.core_path_prefixes and not self.core_path_globs:
            raise EnvelopeNotDeclared(
                "CoreHeaderPolicy needs at least one core path prefix or glob; a policy "
                "that matches nothing would classify every core-header edit as ordinary")
        _require_str(self.declared_by, "CoreHeaderPolicy.declared_by")

    def matches(self, path: str) -> bool:
        if any(path == p or path.startswith(p.rstrip("/") + "/")
               for p in self.core_path_prefixes):
            return True
        return any(Path(path).match(g) for g in self.core_path_globs)

    def to_dict(self) -> dict:
        return {"core_path_prefixes": list(self.core_path_prefixes),
                "core_path_globs": list(self.core_path_globs),
                "backends_served": list(self.backends_served),
                "declared_by": self.declared_by}


@dataclass(frozen=True)
class RiskTierDecision:
    change_class: str
    tier: str                        # "core_header" | "standard"
    matched_core_paths: tuple
    full_tree_surface_required: bool
    per_backend_binary_comparison_required: tuple
    requires_human_code_review: bool
    misdeclared: bool
    reasons: tuple

    def to_dict(self) -> dict:
        return {"change_class": self.change_class, "tier": self.tier,
                "matched_core_paths": list(self.matched_core_paths),
                "full_tree_surface_required": self.full_tree_surface_required,
                "per_backend_binary_comparison_required":
                    list(self.per_backend_binary_comparison_required),
                "requires_human_code_review": self.requires_human_code_review,
                "misdeclared": self.misdeclared, "reasons": list(self.reasons)}


def assess_risk_tier(change_class: str, diff: SourceDiff, policy: CoreHeaderPolicy,
                     *, declared_surface_scope: str) -> tuple:
    """§8.5.1 core-header risk tier. Returns `(RiskTierDecision, GateResult)`.

    *"A change to shared ggml core or to a widely-included header is not a large
    edit — it is a different kind of edit, because its reach is every op in both
    the CPU and GPU builds."* It therefore forces full-tree affected surface,
    forces the §3.2 binary-comparison stage for every backend the tree serves, and
    marks the candidate `REQUIRES_HUMAN_CODE_REVIEW` regardless of the §10.6
    ceiling.
    """
    if change_class not in schemas.CHANGE_CLASSES:
        raise ValueError(f"change_class: {change_class!r} is not one of "
                         f"{sorted(schemas.CHANGE_CLASSES)}")
    if not isinstance(diff, SourceDiff):
        raise TypeError("diff must be a SourceDiff")
    if not isinstance(policy, CoreHeaderPolicy):
        raise TypeError("policy must be a CoreHeaderPolicy")
    if declared_surface_scope not in _SURFACE_SCOPES:
        raise ValueError(f"declared_surface_scope: {declared_surface_scope!r} must be "
                         f"one of {list(_SURFACE_SCOPES)}")

    matched = tuple(sorted(p for p in diff.paths() if policy.matches(p)))
    is_core = bool(matched) or change_class == "core_header"
    reasons: list = []
    checks: list = []
    misdeclared = False

    if not diff.files:
        # The tier is derived MECHANICALLY from the diff (§6.4, invariant 18). A
        # diff with no files matches no core path, so "standard" here would be a
        # tier derived from the absence of evidence rather than from the diff.
        checks.append(_cnc(
            F_EMPTY_DIFF,
            "the supplied diff touches no file, so no core path can match and the risk "
            "tier is not derivable from it; supply the candidate's diff. Absence of a "
            "matched core path is not evidence that none was touched"))

    if matched and change_class != "core_header":
        misdeclared = True
        checks.append(_fail(
            F_MISDECLARED_CORE_HEADER_CHANGE,
            f"the diff touches core/widely-included path(s) {list(matched)} but the "
            f"proposal declares change_class={change_class!r}. The tier is derived from "
            "the diff, not from the declaration (§6.4, invariant 18); the candidate is "
            "held to the core_header tier regardless"))
        reasons.append("tier derived from the diff, overriding the declared change_class")

    if is_core:
        reasons.append(
            "core_header risk tier: full-tree affected surface, per-backend binary "
            f"comparison for {list(policy.backends_served)} (§3.2), and "
            f"{REQUIRES_HUMAN_CODE_REVIEW} regardless of diff size")
        if declared_surface_scope != SURFACE_FULL_TREE:
            checks.append(_fail(
                F_CORE_HEADER_SURFACE_UNDER_DECLARED,
                f"declared affected-surface scope is {declared_surface_scope!r}; the "
                "core_header tier forces full-tree surface regardless of the textual "
                "diff size"))
    else:
        reasons.append("standard risk tier: no declared core path is touched")

    decision = RiskTierDecision(
        change_class=change_class,
        tier="core_header" if is_core else "standard",
        matched_core_paths=matched,
        full_tree_surface_required=is_core,
        per_backend_binary_comparison_required=(
            tuple(policy.backends_served) if is_core else ()),
        requires_human_code_review=is_core,
        misdeclared=misdeclared,
        reasons=tuple(reasons),
    )
    gate = api.GateResult(
        gate_id=GATE_CORE_HEADER, gate_class=api.GATE_INTEGRITY,
        check=_worst(checks) if checks else schemas.Check(PASS),
        requires_anchor=False, notes=tuple(reasons))
    return decision, gate


# =============================================================================
# §8.5.1 (4) — repair starts from a clean parent, capped per proposal
# =============================================================================

PLANNER_DEGRADED = "PLANNER_DEGRADED"


@dataclass(frozen=True)
class RepairPolicy:
    """The per-proposal repair cap. Declared, never defaulted."""

    max_repairs_per_proposal: int
    declared_by: str

    def __post_init__(self) -> None:
        _require_int(self.max_repairs_per_proposal,
                     "RepairPolicy.max_repairs_per_proposal", minimum=0)
        _require_str(self.declared_by, "RepairPolicy.declared_by")

    def to_dict(self) -> dict:
        return {"max_repairs_per_proposal": self.max_repairs_per_proposal,
                "declared_by": self.declared_by}


@dataclass(frozen=True)
class RepairAttempt:
    """One repair attempt's provenance. `base_tree_sha256` is what it STARTED from."""

    proposal_id: str
    attempt_index: int
    parent_candidate_id: str
    parent_snapshot_sha256: str
    base_tree_sha256: str
    failed_attempt_tree_sha256: str
    checked_out_fresh: bool
    worktree_path: str

    def __post_init__(self) -> None:
        _require_str(self.proposal_id, "RepairAttempt.proposal_id")
        if not self.proposal_id.startswith("akp-"):
            raise ValueError("RepairAttempt.proposal_id must start with 'akp-'")
        _require_int(self.attempt_index, "RepairAttempt.attempt_index", minimum=1)
        _require_str(self.parent_candidate_id, "RepairAttempt.parent_candidate_id")
        if not self.parent_candidate_id.startswith("akc-"):
            raise ValueError("RepairAttempt.parent_candidate_id must start with 'akc-'")
        for name in ("parent_snapshot_sha256", "base_tree_sha256",
                     "failed_attempt_tree_sha256"):
            _require_sha256(getattr(self, name), f"RepairAttempt.{name}")
        _require_bool(self.checked_out_fresh, "RepairAttempt.checked_out_fresh")
        if not _require_str(self.worktree_path,
                            "RepairAttempt.worktree_path").startswith("/"):
            raise ValueError("RepairAttempt.worktree_path must be absolute")

    def to_dict(self) -> dict:
        return {"proposal_id": self.proposal_id, "attempt_index": self.attempt_index,
                "parent_candidate_id": self.parent_candidate_id,
                "parent_snapshot_sha256": self.parent_snapshot_sha256,
                "base_tree_sha256": self.base_tree_sha256,
                "failed_attempt_tree_sha256": self.failed_attempt_tree_sha256,
                "checked_out_fresh": self.checked_out_fresh,
                "worktree_path": self.worktree_path}


def check_repair_from_clean_parent(attempt: Optional[RepairAttempt]) -> api.GateResult:
    """§8.5.1 (4): *"A bounded repair attempt re-checks out the parent candidate and
    re-applies, never continuing on the failed attempt's tree."*

    AutoPilot's scar here was a loop compounding edits onto an already-corrupted
    file. A repair whose base tree IS the failed attempt's tree is that loop.
    """
    if attempt is None:
        return api.GateResult(
            gate_id=GATE_REPAIR_CLEAN_PARENT, gate_class=api.GATE_INTEGRITY,
            check=schemas.Check(PASS), requires_anchor=False,
            notes=("this candidate is not a repair attempt; the clean-parent rule does "
                   "not apply",))
    if not isinstance(attempt, RepairAttempt):
        raise TypeError("attempt must be a RepairAttempt or None")

    checks: list = []
    if not attempt.checked_out_fresh:
        checks.append(_fail(
            F_REPAIR_NOT_RECHECKED_OUT,
            f"repair attempt {attempt.attempt_index} of {attempt.proposal_id} did not "
            "re-check out the parent; it continued on an existing tree"))
    if attempt.base_tree_sha256 == attempt.failed_attempt_tree_sha256:
        checks.append(_fail(
            F_REPAIR_CONTINUED_ON_FAILED_TREE,
            f"repair base tree {attempt.base_tree_sha256[:12]} IS the failed attempt's "
            "tree; this is the compounding-edits loop that corrupted escalation.py"))
    elif attempt.base_tree_sha256 != attempt.parent_snapshot_sha256:
        checks.append(_fail(
            F_REPAIR_BASE_NOT_PARENT_SNAPSHOT,
            f"repair base tree {attempt.base_tree_sha256[:12]} is neither the failed "
            f"attempt's tree nor the parent candidate {attempt.parent_candidate_id}'s "
            f"snapshot {attempt.parent_snapshot_sha256[:12]}"))
    return api.GateResult(
        gate_id=GATE_REPAIR_CLEAN_PARENT, gate_class=api.GATE_INTEGRITY,
        check=_worst(checks) if checks else schemas.Check(PASS),
        requires_anchor=False,
        notes=(f"repair attempt {attempt.attempt_index} of {attempt.proposal_id}, "
               f"re-checked out from {attempt.parent_candidate_id}",))


@dataclass(frozen=True)
class RepairDecision:
    """Granted, or refused with a signal. There is no third 'try again anyway'."""

    granted: bool
    attempt_index: Optional[int]
    signal: Optional[str]
    reason: str
    parent_candidate_id: str
    parent_snapshot_sha256: str

    def to_dict(self) -> dict:
        return {"granted": self.granted, "attempt_index": self.attempt_index,
                "signal": self.signal, "reason": self.reason,
                "parent_candidate_id": self.parent_candidate_id,
                "parent_snapshot_sha256": self.parent_snapshot_sha256}


@dataclass(frozen=True)
class RepairLedger:
    """Repairs used so far for one proposal. Immutable; `request` returns a new one.

    *"Repairs are capped per proposal; exceeding the cap is a `PLANNER_DEGRADED`
    signal, not another retry."* `request()` therefore never raises-to-retry and
    never returns a grant past the cap; it returns a refusal carrying the signal,
    which the controller journals.
    """

    proposal_id: str
    policy: RepairPolicy
    parent_candidate_id: str
    parent_snapshot_sha256: str
    used: int

    def __post_init__(self) -> None:
        _require_str(self.proposal_id, "RepairLedger.proposal_id")
        if not self.proposal_id.startswith("akp-"):
            raise ValueError("RepairLedger.proposal_id must start with 'akp-'")
        if not isinstance(self.policy, RepairPolicy):
            raise TypeError("RepairLedger.policy must be a RepairPolicy")
        _require_str(self.parent_candidate_id, "RepairLedger.parent_candidate_id")
        _require_sha256(self.parent_snapshot_sha256, "RepairLedger.parent_snapshot_sha256")
        _require_int(self.used, "RepairLedger.used", minimum=0)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.policy.max_repairs_per_proposal

    def request(self) -> tuple:
        """Return `(RepairDecision, next_ledger)`. The ledger only advances on a grant."""
        if self.exhausted:
            return (RepairDecision(
                granted=False, attempt_index=None, signal=PLANNER_DEGRADED,
                reason=(f"{F_REPAIR_CAP_EXCEEDED}: {self.used} repair(s) already used for "
                        f"{self.proposal_id}, cap is "
                        f"{self.policy.max_repairs_per_proposal} (declared by "
                        f"{self.policy.declared_by!r}). Exceeding the cap is a "
                        f"{PLANNER_DEGRADED} signal, not another retry"),
                parent_candidate_id=self.parent_candidate_id,
                parent_snapshot_sha256=self.parent_snapshot_sha256), self)
        nxt = RepairLedger(
            proposal_id=self.proposal_id, policy=self.policy,
            parent_candidate_id=self.parent_candidate_id,
            parent_snapshot_sha256=self.parent_snapshot_sha256, used=self.used + 1)
        return (RepairDecision(
            granted=True, attempt_index=nxt.used, signal=None,
            reason=(f"repair {nxt.used} of {self.policy.max_repairs_per_proposal}: "
                    f"re-check out {self.parent_candidate_id} at snapshot "
                    f"{self.parent_snapshot_sha256[:12]} and re-apply; never continue on "
                    "the failed attempt's tree"),
            parent_candidate_id=self.parent_candidate_id,
            parent_snapshot_sha256=self.parent_snapshot_sha256), nxt)

    def to_dict(self) -> dict:
        return {"proposal_id": self.proposal_id, "policy": self.policy.to_dict(),
                "parent_candidate_id": self.parent_candidate_id,
                "parent_snapshot_sha256": self.parent_snapshot_sha256,
                "used": self.used, "exhausted": self.exhausted}


# =============================================================================
# Orchestration — all of §8.5.1, in order, before any behavioural check
# =============================================================================

@dataclass(frozen=True)
class SourceIntegrityInputs:
    """Every input the §8.5.1 gates consume. No field has a default.

    `api.WindowAttestations` makes the same choice for the same reason: a
    defaulted attestation is one nobody made. `Optional` here means "None is a
    MEANINGFUL value" (no repair attempt; no signature index), never "you may
    leave it out".
    """

    candidate_id: str
    backend: str
    change_class: str
    artifact_binary_sha256: str
    anchor_symbols: Optional[ElfSymbolTable]
    candidate_symbols: Optional[ElfSymbolTable]
    signature_index: Optional[Mapping]
    anchor_registrations: tuple
    candidate_registrations: tuple
    declared_symbol_deltas: DeclaredSymbolDeltas
    declared_surface: DeclaredSurface
    declared_surface_scope: str
    diff: SourceDiff
    envelope: ChangeClassEnvelope
    complexity_ceiling: ComplexityCeiling
    core_header_policy: CoreHeaderPolicy
    original_line_counts: Optional[Mapping]
    build: BuildProvenance
    snapshot_recompute_root: Optional[str]
    snapshot_attested_by: Optional[str]
    repair: Optional[RepairAttempt]

    def __post_init__(self) -> None:
        _require_str(self.candidate_id, "SourceIntegrityInputs.candidate_id")
        if not self.candidate_id.startswith("akc-"):
            raise ValueError("candidate_id must start with 'akc-'")
        if self.backend not in schemas.BACKENDS:
            raise ValueError(f"backend: {self.backend!r} is not a known backend")
        if self.change_class not in schemas.CHANGE_CLASSES:
            raise ValueError(f"change_class: {self.change_class!r} is not a known class")
        if self.envelope.change_class != self.change_class:
            raise ValueError(
                f"envelope is for {self.envelope.change_class!r} but the proposal "
                f"declares {self.change_class!r}")
        if self.complexity_ceiling.backend != self.backend:
            raise ValueError(
                f"complexity ceiling is for {self.complexity_ceiling.backend!r} but the "
                f"cell is {self.backend!r}")
        _require_sha256(self.artifact_binary_sha256,
                        "SourceIntegrityInputs.artifact_binary_sha256")
        if self.build.candidate_id != self.candidate_id:
            raise ValueError(
                f"build provenance is for {self.build.candidate_id!r} but these inputs "
                f"are for {self.candidate_id!r}")
        if self.declared_surface_scope not in _SURFACE_SCOPES:
            raise ValueError(f"declared_surface_scope must be one of {list(_SURFACE_SCOPES)}")
        for name in ("anchor_registrations", "candidate_registrations"):
            tables = getattr(self, name)
            if not isinstance(tables, tuple):
                raise TypeError(f"SourceIntegrityInputs.{name} must be a tuple")
            for table in tables:
                if not isinstance(table, RegistrationTable):
                    raise TypeError(f"{name} must hold RegistrationTable")
            kinds = [t.kind for t in tables]
            if len(kinds) != len(set(kinds)):
                raise ValueError(
                    f"{name} holds more than one table of the same kind ({kinds}); the "
                    "second would silently shadow the first and its removals would go "
                    "unchecked")


def _requires_human_code_review(complexity: ComplexityAssessment,
                                risk_tier: RiskTierDecision) -> bool:
    """Either source of the §10.6 marker sets it. The OR is the whole point.

    A backend adapter that declares `shared_core_modification_requires_review`
    False can zero the complexity side while the derived core-header tier still
    demands review; the receipt must carry THIS answer, not the complexity
    block's, or a reader sees `requires_human_code_review: false` on a
    core-header candidate.
    """
    return bool(complexity.requires_human_code_review
                or risk_tier.requires_human_code_review)


def _first_page_notice(complexity: ComplexityAssessment,
                       risk_tier: RiskTierDecision) -> Optional[str]:
    if not _requires_human_code_review(complexity, risk_tier):
        return None
    if complexity.first_page_notice:
        return complexity.first_page_notice
    return f"{REQUIRES_HUMAN_CODE_REVIEW} — " + "; ".join(risk_tier.reasons)


@dataclass(frozen=True)
class SourceIntegrityReport:
    """The full §8.5.1 outcome for one candidate."""

    candidate_id: str
    gates: tuple
    risk_tier: RiskTierDecision
    complexity: ComplexityAssessment
    clean_build_receipt: CleanBuildReceipt
    receipt: dict

    @property
    def blocking(self) -> bool:
        """True when ANY gate is not PASS — including COULD_NOT_CHECK.

        Fail-closed on purpose: *"Any failure ends speed ranking for that
        candidate"*, and an unevaluated integrity gate is not a passed one.
        """
        return any(g.check.outcome != PASS for g in self.gates)

    @property
    def requires_human_code_review(self) -> bool:
        return _requires_human_code_review(self.complexity, self.risk_tier)

    @property
    def first_page_notice(self) -> Optional[str]:
        return _first_page_notice(self.complexity, self.risk_tier)

    @property
    def content_hash(self) -> str:
        return schemas.content_hash(self.receipt)


def run_source_integrity_gates(inputs: SourceIntegrityInputs) -> SourceIntegrityReport:
    """Run all of §8.5.1, in order, and build the receipt.

    Order is the design's: symbol/registration preservation, clean build from the
    recorded snapshot, semantic diff conformance, core-header risk tier, repair
    from a clean parent. §8.6 requires the whole set to run *"before any
    behavioural check"*; `SourceIntegrityFirstRunner` is what enforces that at the
    dispatch seam.
    """
    if not isinstance(inputs, SourceIntegrityInputs):
        raise TypeError("inputs must be a SourceIntegrityInputs")

    gates: list = [
        check_symbol_preservation(inputs.anchor_symbols, inputs.candidate_symbols,
                                  inputs.declared_symbol_deltas),
        check_symbol_arity_coverage(inputs.anchor_symbols, inputs.candidate_symbols,
                                    inputs.signature_index),
    ]

    anchor_by_kind = {t.kind: t for t in inputs.anchor_registrations}
    cand_by_kind = {t.kind: t for t in inputs.candidate_registrations}
    for kind in _REGISTRATION_KINDS:
        gates.append(check_registration_preservation(
            anchor_by_kind.get(kind), cand_by_kind.get(kind),
            inputs.declared_symbol_deltas, expected_kind=kind))

    build_gate, build_receipt = check_clean_build_from_snapshot(
        inputs.build, inputs.artifact_binary_sha256,
        recompute_root=inputs.snapshot_recompute_root,
        snapshot_attested_by=inputs.snapshot_attested_by)
    gates.append(build_gate)

    gates.append(check_semantic_diff_conformance(
        inputs.diff, inputs.declared_surface, inputs.envelope,
        original_line_counts=inputs.original_line_counts))

    risk_tier, tier_gate = assess_risk_tier(
        inputs.change_class, inputs.diff, inputs.core_header_policy,
        declared_surface_scope=inputs.declared_surface_scope)
    gates.append(tier_gate)

    gates.append(check_repair_from_clean_parent(inputs.repair))

    complexity = assess_complexity_ceiling(
        inputs.diff, inputs.complexity_ceiling,
        touches_shared_core=bool(risk_tier.matched_core_paths),
        change_class=inputs.change_class)

    receipt = {
        "receipt": "autokernel.evaluator.integrity.source_integrity/v1",
        "design_section": DESIGN_SECTION,
        "protocol_id": api.PROTOCOL_VERSIONED_ID,
        # §10.6: *"the package is marked REQUIRES_HUMAN_CODE_REVIEW and says so
        # on its first page."* The receipt IS the journaled page, so the marker
        # is a top-level key of it and not something a reader must recompute by
        # OR-ing the complexity block with the risk-tier block.
        "requires_human_code_review": _requires_human_code_review(complexity, risk_tier),
        "first_page_notice": _first_page_notice(complexity, risk_tier),
        "candidate_id": inputs.candidate_id,
        "backend": inputs.backend,
        "change_class": inputs.change_class,
        "artifact_binary_sha256": inputs.artifact_binary_sha256,
        "gates": [g.to_dict() for g in gates],
        "risk_tier": risk_tier.to_dict(),
        "complexity": complexity.to_dict(),
        "clean_build_receipt": build_receipt.to_dict(),
        "declared_symbol_deltas": inputs.declared_symbol_deltas.to_dict(),
        "declared_surface": inputs.declared_surface.to_dict(),
        "declared_surface_scope": inputs.declared_surface_scope,
        "envelope": inputs.envelope.to_dict(),
        "complexity_ceiling": inputs.complexity_ceiling.to_dict(),
        "core_header_policy": inputs.core_header_policy.to_dict(),
        "diff": inputs.diff.to_dict(),
        "repair": inputs.repair.to_dict() if inputs.repair is not None else None,
        "anchor_symbol_table": (inputs.anchor_symbols.to_dict()
                                if inputs.anchor_symbols is not None else None),
        "candidate_symbol_table": (inputs.candidate_symbols.to_dict()
                                   if inputs.candidate_symbols is not None else None),
        "registration_tables": {
            "anchor": [t.to_dict() for t in inputs.anchor_registrations],
            "candidate": [t.to_dict() for t in inputs.candidate_registrations],
        },
    }
    return SourceIntegrityReport(
        candidate_id=inputs.candidate_id, gates=tuple(gates), risk_tier=risk_tier,
        complexity=complexity, clean_build_receipt=build_receipt, receipt=receipt)


def check_evidence_binding(request: Any,
                           inputs: SourceIntegrityInputs) -> api.GateResult:
    """Bind the §8.5.1 evidence to the identities the request NAMES.

    Precondition 4: *"Every performance, coherence, correctness, capacity, or
    determinism comparison names its anchor by source commit, binary SHA-256, and
    linkage SHA-256 … A rebuilt anchor is a different anchor."*

    `requires_anchor=True` alone does not deliver that. It proves only that SOME
    anchor object is bound to the window; it says nothing about WHICH binary the
    symbol table on the anchor side of the diff was read out of. Without this
    gate, an `ElfSymbolTable` extracted from any file at all diffs clean against
    the candidate and the verdict is PASS — a comparison against an anchor that
    was never the named one. The gate answers:

      * FAIL when an identity the caller stated twice disagrees with itself (the
        candidate id, the artifact binary SHA-256) — that is a wiring defect;
      * COULD_NOT_CHECK when a symbol table cannot be shown to have come from the
        named anchor or artifact binary — the reading may be right, but it is
        unevidenced, and *"absence of a comparison is not evidence of
        equivalence."*
    """
    if not isinstance(inputs, SourceIntegrityInputs):
        raise TypeError("inputs must be a SourceIntegrityInputs")
    if not isinstance(request, api.EvaluationRequest):
        raise TypeError("request must be an api.EvaluationRequest")

    checks: list = []
    notes: list = []

    if request.candidate_id != inputs.candidate_id:
        checks.append(_fail(
            F_CANDIDATE_ID_MISMATCH,
            f"the request evaluates {request.candidate_id!r} but these integrity inputs "
            f"describe {inputs.candidate_id!r}; the gates would report on a different "
            "candidate than the one under test"))
    if request.artifact.binary_sha256 != inputs.artifact_binary_sha256:
        checks.append(_fail(
            F_ARTIFACT_SHA256_MISMATCH,
            f"request.artifact.binary_sha256 {request.artifact.binary_sha256[:12]} != "
            f"integrity inputs' artifact_binary_sha256 "
            f"{inputs.artifact_binary_sha256[:12]}; the clean-build gate judged a "
            "different binary than the one the request names"))

    if request.anchor is None:
        checks.append(_cnc(
            F_NO_ANCHOR_BOUND,
            "the request binds no anchor, so no symbol or registration comparison in "
            "this set can be shown to be against the campaign's immutable anchor "
            "(P-AK-SEARCH-1 precondition 4)"))
    elif inputs.anchor_symbols is not None:
        if inputs.anchor_symbols.file_sha256 != request.anchor.binary_sha256:
            checks.append(_cnc(
                F_SYMBOL_TABLE_NOT_BOUND_TO_ANCHOR,
                f"the anchor symbol table was extracted from "
                f"{inputs.anchor_symbols.source_path!r} whose SHA-256 is "
                f"{inputs.anchor_symbols.file_sha256[:12]}, but the request names anchor "
                f"binary {request.anchor.binary_sha256[:12]} "
                f"({request.anchor.short()}). A rebuilt anchor is a different anchor: "
                "extract the table from the named anchor binary, or record the mapping "
                "from that binary to the object this table came from"))
        else:
            notes.append(f"anchor symbol table bound to {request.anchor.short()}")

    if inputs.candidate_symbols is not None:
        if inputs.candidate_symbols.file_sha256 != request.artifact.binary_sha256:
            checks.append(_cnc(
                F_SYMBOL_TABLE_NOT_BOUND_TO_ARTIFACT,
                f"the candidate symbol table was extracted from "
                f"{inputs.candidate_symbols.source_path!r} whose SHA-256 is "
                f"{inputs.candidate_symbols.file_sha256[:12]}, but the artifact under "
                f"test is {request.artifact.binary_sha256[:12]}; the ABI that was diffed "
                "has not been shown to be the ABI of the artifact being ranked"))
        else:
            notes.append("candidate symbol table bound to the artifact under test")

    return api.GateResult(
        gate_id=GATE_EVIDENCE_BINDING, gate_class=api.GATE_INTEGRITY,
        check=_worst(checks) if checks else schemas.Check(PASS),
        requires_anchor=True,
        evidence_ref=request.artifact.binary_sha256,
        notes=tuple(notes))


class SourceIntegrityGateRunner:
    """An `api.TierGateRunner` that yields ONLY the §8.5.1 gates.

    Raises `IntegrityInputsMissing` for an unregistered candidate rather than
    returning an empty gate list, for the same reason `TierDispatcher` raises
    `EvaluatorNotWired`: an empty gate list derives to PASS.
    """

    def __init__(self, *, tier: str,
                 inputs_by_candidate: Mapping[str, SourceIntegrityInputs],
                 derived_surfaces: Optional[Mapping[str, Any]] = None) -> None:
        self.tier = api.admit_tier(tier)   # refuses T3/T4 at WIRING time
        if not isinstance(inputs_by_candidate, Mapping):
            raise TypeError("inputs_by_candidate must be a mapping")
        for cid, value in inputs_by_candidate.items():
            if not isinstance(value, SourceIntegrityInputs):
                raise TypeError(f"inputs_by_candidate[{cid!r}] must be a "
                                "SourceIntegrityInputs")
            if value.candidate_id != cid:
                raise ValueError(
                    f"inputs_by_candidate[{cid!r}] describes candidate "
                    f"{value.candidate_id!r}; a mis-keyed entry silently evaluates one "
                    "candidate's evidence under another candidate's name")
        self._inputs = dict(inputs_by_candidate)
        self._reports: dict = {}
        # `derived_surfaces` binds each candidate's `declared_surface_scope` to
        # the manifest `surface.derive_affected_surface()` actually derived. It
        # is optional because a runner may legitimately be wired without stage 1
        # having run, and `surface_binding` is how a caller reads which of the two
        # states this runner is in — the alternative, defaulting to "unbound and
        # silent", is what let a mis-declared full-tree surface PASS.
        if derived_surfaces is None:
            self._derived: Optional[dict] = None
        else:
            if not isinstance(derived_surfaces, Mapping):
                raise TypeError("derived_surfaces must be a mapping or None")
            for cid, manifest in derived_surfaces.items():
                if not isinstance(manifest, ak_surface.AffectedSurface):
                    raise TypeError(
                        f"derived_surfaces[{cid!r}] must be a surface.AffectedSurface")
                if manifest.candidate_id != cid:
                    raise ValueError(
                        f"derived_surfaces[{cid!r}] is the manifest for "
                        f"{manifest.candidate_id!r}; a mis-keyed entry would verify one "
                        "candidate's scope against another candidate's surface")
            self._derived = dict(derived_surfaces)

    @property
    def surface_binding(self) -> bool:
        """True when this runner verifies `declared_surface_scope` against stage 1."""
        return self._derived is not None

    def report_for(self, candidate_id: str) -> SourceIntegrityReport:
        try:
            inputs = self._inputs[candidate_id]
        except KeyError as exc:
            raise IntegrityInputsMissing(
                f"no source-integrity inputs registered for candidate "
                f"{candidate_id!r}; registered candidates are {sorted(self._inputs)}. "
                "There is no default: a candidate with no integrity inputs would "
                "produce an empty gate list, which derives to PASS") from exc
        report = run_source_integrity_gates(inputs)
        self._reports[candidate_id] = report
        return report

    def last_report(self, candidate_id: str) -> SourceIntegrityReport:
        try:
            return self._reports[candidate_id]
        except KeyError as exc:
            raise IntegrityInputsMissing(
                f"no source-integrity report has been produced for {candidate_id!r} "
                "yet; run_gates() must run before the report is available") from exc

    def run_gates(self, request: api.EvaluationRequest) -> tuple:
        """The §8.5.1 set, PLUS the gate that binds it to the request's identities.

        The binding gate is emitted only here because only here does a request
        exist. `run_source_integrity_gates` still returns exactly `GATE_IDS`;
        this returns `RUNNER_GATE_IDS`, plus `GATE_SURFACE_SCOPE_BINDING` when
        this runner was wired with the derived affected-surface manifests.
        """
        if not isinstance(request, api.EvaluationRequest):
            raise TypeError("request must be an api.EvaluationRequest")
        report = self.report_for(request.candidate_id)
        inputs = self._inputs[request.candidate_id]
        gates = report.gates + (check_evidence_binding(request, inputs),)
        if self._derived is None:
            return gates
        try:
            manifest = self._derived[request.candidate_id]
        except KeyError as exc:
            raise IntegrityInputsMissing(
                f"this runner was wired to verify declared_surface_scope against the "
                f"derived affected-surface manifest, but no manifest is registered for "
                f"{request.candidate_id!r}; registered manifests are "
                f"{sorted(self._derived)}. Falling back to the unbound branch would turn "
                "a declared capability into a silent skip") from exc
        return gates + (api.GateResult(
            gate_id=GATE_SURFACE_SCOPE_BINDING, gate_class=api.GATE_INTEGRITY,
            check=check_declared_surface_scope(inputs.declared_surface_scope, manifest),
            requires_anchor=False, evidence_ref=manifest.sha256(),
            notes=("§6.4: the affected surface is DERIVED; invariant 18 makes a "
                   "declaration a scored prediction, never a scope input",)),)


class SourceIntegrityFirstRunner:
    """Compose the §8.5.1 gates AHEAD of a behavioural runner, and short-circuit.

    §8.6 lists the source-integrity gates first and says they *"run before any
    behavioural check"*. Running them first is not enough on its own: if
    behavioural gates still ran and PASSed, the record would carry passing
    correctness evidence for a binary whose ABI was never verified. So when the
    integrity set is blocking, the behavioural runner is NOT invoked and a single
    `COULD_NOT_CHECK` gate records that it was not — never a PASS, and never
    silence.
    """

    def __init__(self, *, integrity: SourceIntegrityGateRunner,
                 behavioural: Any) -> None:
        if not isinstance(integrity, SourceIntegrityGateRunner):
            raise TypeError("integrity must be a SourceIntegrityGateRunner")
        if not hasattr(behavioural, "run_gates"):
            raise TypeError("behavioural runner has no run_gates(request)")
        behaviour_tier = getattr(behavioural, "tier", None)
        if behaviour_tier is not None and behaviour_tier != integrity.tier:
            raise ValueError(
                f"tier mismatch: integrity runner is {integrity.tier!r}, behavioural "
                f"runner is {behaviour_tier!r}")
        self.tier = integrity.tier
        self.integrity = integrity
        self.behavioural = behavioural

    def run_gates(self, request: api.EvaluationRequest) -> tuple:
        # Ask the integrity runner, not the report: the report is the §8.5.1 set
        # alone, and the evidence-binding gate must block the behavioural runner
        # for the same reason the §8.5.1 gates do — a correctness PASS measured
        # against an unbound anchor is worse than no result.
        gates = list(self.integrity.run_gates(request))
        blocked = sorted(f"{g.gate_id}={g.check.outcome}" for g in gates
                         if g.check.outcome != PASS)
        if blocked:
            gates.append(api.GateResult(
                gate_id=GATE_BEHAVIOURAL_NOT_RUN, gate_class=api.GATE_INTEGRITY,
                check=_cnc(F_BEHAVIOURAL_GATES_NOT_RUN,
                           "the §8.5.1 source-integrity gates did not all PASS, so the "
                           f"behavioural gates were not run: {blocked}"),
                requires_anchor=False,
                notes=("§8.6: the source-integrity gates run before any behavioural "
                       "check; a behavioural PASS on an unverified binary would be "
                       "worse than no result",)))
            return tuple(gates)
        behavioural = self.behavioural.run_gates(request)
        if not isinstance(behavioural, (list, tuple)):
            raise TypeError(
                f"behavioural runner returned {type(behavioural).__name__}; expected a "
                "sequence of GateResult")
        for gate in behavioural:
            if not isinstance(gate, api.GateResult):
                raise TypeError("behavioural runner must return api.GateResult objects")
        gates.extend(behavioural)
        return tuple(gates)


# =============================================================================
# Self-audit — this module READS files; it must never write one or run a process
# =============================================================================

_FORBIDDEN_CALL_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "input"})

#: An AST audit cannot tell `Path.replace` from `str.replace`, so a few entries
#: here are broader than the write path they exist to forbid — the same trade
#: `flush` and `communicate` already make. Broader is the correct direction: the
#: cost is that a maintainer must spell an innocent call differently, and the
#: alternative is a permitted `Path(p).replace(q)`, which overwrites a file.
_FORBIDDEN_CALL_ATTRS = frozenset({
    "write", "writelines", "write_text", "write_bytes", "truncate", "flush", "fsync",
    "mkdir", "makedirs", "remove", "unlink", "rmdir", "rmtree", "rename", "chmod",
    "chown", "utime", "symlink", "link", "touch", "move", "copy", "copyfile", "copytree",
    "system", "popen", "Popen", "spawnv", "fork", "kill", "killpg", "send_signal",
    "terminate", "check_call", "check_output", "communicate", "setxattr",
    "replace", "symlink_to", "hardlink_to", "mknod", "lchmod",
})

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "tempfile", "sqlite3", "urllib", "http", "requests", "pty", "fcntl", "resource",
    "shlex", "asyncio", "io", "runpy",
})

#: `Path.open` is the ONE opener this module uses, and only to stream a read.
#: The audit therefore permits `.open(...)` exactly when its mode argument is a
#: string literal that reads: no `w`, `a`, `x`, or `+`.
_READ_MODES = frozenset({"r", "rb", "rt", "br", "tr"})


#: Bound by `audit_no_write_or_process_paths` so the self-audit proves it read its
#: OWN module. Added 2026-08-04: this hole was found by the refactor mining pass,
#: recorded as CLOSED at five sites by commit 4e96fdc0, and left LIVE here — the
#: plan that scheduled the fix skipped this module because it was condemned to
#: deletion. It then SURVIVED the deletion, because `worktree.py` takes its
#: build-identity receipt from here and `microbench.py` hashes the benchmarked
#: binary with `sha256_file`. A fix skipped on the grounds that the code was going
#: away is a fix that never happens when the code does not go away.
MODULE_ID = "autokernel.evaluator.integrity/v1"


def _defines_this_module(tree: ast.AST) -> bool:
    """True when the parsed source assigns this module's own `MODULE_ID`.

    Mirrors `api._defines_this_module` deliberately rather than importing it:
    `api` does not import `integrity`, and reversing that edge to share nine lines
    would put a cycle between the evaluator's two audit surfaces.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODULE_ID" and \
                        isinstance(node.value, ast.Constant) and \
                        node.value.value == MODULE_ID:
                    return True
    return False


def audit_no_write_or_process_paths(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's own AST that it reads but cannot write or signal.

    Unlike `api.audit_no_write_or_process_paths`, this module legitimately opens
    files — ELF binaries and source trees. So the audit is not "no opener": it is
    "no opener that could write". `.open(...)` is permitted only with a literal
    read mode; a bare `open()`, a computed mode, or any write/process call FAILs.

    COULD_NOT_CHECK when the source cannot be read or parsed — an unreadable
    module is not an audited one.
    """
    supplied = source is not None
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(COULD_NOT_CHECK, (f"could not parse module: {exc}",))

    findings: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_IMPORTS:
                findings.append(f"line {node.lineno}: imports from {node.module!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
                findings.append(f"line {node.lineno}: calls {func.id}()")
            elif isinstance(func, ast.Attribute):
                if func.attr == "open":
                    mode = node.args[0] if node.args else None
                    literal = (mode.value if isinstance(mode, ast.Constant)
                               and isinstance(mode.value, str) else None)
                    if literal is None:
                        for kw in node.keywords:
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                literal = kw.value.value
                    if literal not in _READ_MODES:
                        findings.append(
                            f"line {node.lineno}: .open() with mode {literal!r}; only a "
                            f"literal read mode from {sorted(_READ_MODES)} is permitted")
                elif func.attr in _FORBIDDEN_CALL_ATTRS:
                    findings.append(f"line {node.lineno}: calls .{func.attr}()")

    if findings:
        # A forbidden construct is a finding ABOUT THE TEXT, so it is returned
        # before any identity question — FAIL never depends on whose module it is.
        return schemas.Check(FAIL, tuple(findings))
    if not _defines_this_module(tree):
        # A clean bill of health is a statement about THIS module. Without this
        # branch `audit_no_write_or_process_paths("")` returned PASS — the
        # guarantee obtained by deleting the thing it inspects. Bound here rather
        # than trusted, whether the source was supplied or read from `__file__`
        # (which a repointed `__file__` would otherwise let lie).
        return schemas.Check(COULD_NOT_CHECK, (
            "the audited source does not define this module's MODULE_ID "
            f"({MODULE_ID!r}), so a clean result says nothing about "
            f"{'the supplied source' if supplied else 'this module'}",))
    return schemas.Check(PASS)
