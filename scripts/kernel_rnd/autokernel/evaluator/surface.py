#!/usr/bin/env python3
"""surface.py — affected-surface derivation, dispatch-trace reconciliation, and the
§3.2 two-stage backend-unchanged test.

WHY THIS MODULE EXISTS
----------------------
Freeze scope, lineage composition and sentinel selection all key off the
affected-surface manifest. **If the ACTOR declares that manifest, the actor sets its
own release scope** — a cheaper and far likelier exploit than faking a score, because
it needs no lying about numbers at all: a candidate that touches shared ggml core and
declares `backends: [llama_cpu]` simply never has its GPU cells measured, and the
freeze ships an unmeasured HIP binary. Invariant 18 is the answer: *"Declared equals
traced. The affected-surface manifest is mechanically derived and dynamically
confirmed; the actor's declaration is a scored prediction, never a scope input."*

The three stages of design §6.4 are three functions here, and the trust boundary is
carried in the **types**, not in a comment:

  1. `derive_affected_surface()` — STATIC. diff -> touched files -> objects ->
     link targets -> symbols -> op registrations -> dispatch predicates -> backends.
     The build-target dependency closure is taken from **the build system's own
     generated dependency information** (`gcc -MD` / ninja `-t deps` depfiles, CMake
     `link.txt` link manifests). `BuildDependencyIndex` REFUSES any provenance other
     than `build_system_generated`, so a hand-maintained list or a directory-prefix
     guess cannot be substituted — it raises `UntrustedProvenance`.
     **Over-approximation is expected and CORRECT**: a shared-header change implies
     the whole tree until the depfiles prove otherwise, and every widening is a named
     `OverApproximation` in the manifest rather than a silent one.
  2. `parse_dispatch_trace()` — DYNAMIC. Parses the T0 dispatch trace recording which
     kernels actually executed, including the no-fallback proof (§8.6).
  3. `reconcile_surface()` — RECONCILIATION. `derived ⊇ traced` must hold, and
     `traced ⊄ derived` is a HARD candidate failure. `reconcile_surface()` takes
     exactly two arguments — the derived manifest and the trace. **There is no
     parameter through which a declaration could enter**, which is what makes
     "never a scope input" structural. `score_actor_declaration()` scores the
     declaration against the derived manifest and returns a critic payload that is
     explicitly `is_scope_input: false`.

Absence of a trace is `COULD_NOT_CHECK`, never `PASS`: an empty or truncated trace is
exactly what a candidate that escaped its declared surface also produces.

THE SECOND JOB — §3.2's two-stage backend-unchanged test
--------------------------------------------------------
The release-plan compiler calls `backend_unchanged()` to decide whether a backend's
cells may be dropped because its binary did not change. **Naive binary byte-identity
is WRONG and is refused by name here** (`compare_binaries_byte_identical()` raises
`NaiveByteIdentityRefused`): ROCm/llama.cpp builds embed build IDs, timestamps and
absolute paths, so a freshly built binary is essentially never byte-identical to one
built months earlier in another directory, and a test formulated that way would never
fire. The real test is:

  * **Stage 1 — source-closure identity (the gate).** Diff `production_base..candidate`
    restricted to the backend's build-system-derived closure; unchanged iff that diff
    is empty AND toolchain, flags and build environment are identical.
  * **Stage 2 — normalized binary confirmation (required before dropping cells).**
    The production base is rebuilt *in the candidate's build environment* so both share
    one non-determinism regime, then normalized digests of `.text`, `.rodata`,
    `.data.rel.ro` and the dynamic symbol table are compared, with `.comment`,
    `.note.gnu.build-id` and debug sections excluded. `NormalizedBinaryDigest` refuses
    to hold an excluded section or a whole-file digest.
  * **Disagreement is a HARD FINDING filed against build identity** — never a silent
    preference for the cheaper answer. Either the closure is wrong or the build is
    non-deterministic; in both directions `may_drop_cells` is False and a
    `BuildIdentityFinding` is emitted.

Transfer additionally requires the incumbent's evidence to still be in scope — same
models and recipes, same topology hash, no era boundary crossed — via
`EvidenceTransferScope`.

PROTOCOL CLAUSES
----------------
`measurement/protocols/kernel-research.md` (Annex K, P-AK-SEARCH-1, RATIFIED
2026-08-03), by section name:

  * *"Preconditions (all enforced or attested per run)"* precondition 4 — the anchor
    is named by source commit, binary SHA-256 and linkage SHA-256, and *"a rebuilt
    anchor is a different anchor"*. `RebuildAttestation` therefore never claims to
    reproduce the anchor: it records that the base commit was rebuilt in the
    candidate's environment for a **normalized section comparison**, and the
    backend-unchanged gate carries `requires_anchor=True` so that api.py demotes its
    PASS to COULD_NOT_CHECK when no anchor is bound.
  * *"Correctness precedence"* — every gate this module emits is
    `gate_class=integrity`, one of the five lexicographically-prior classes, so a
    surface escape ends speed ranking for the candidate rather than penalising it.
  * *"What voids a run"* — this module raises nothing into a void reason; a surface
    escape is a candidate FAIL, not a void, because it says something about the
    candidate. Conversely a missing depfile is `COULD_NOT_CHECK`, not a pass.
  * *"Search-grade requires ALL of"* — the derived and traced manifest hashes are part
    of the candidate record (`affected_surface.derived_sha256` / `traced_sha256` /
    `reconciled`, `schemas.validate_candidate`), which `candidate_affected_surface_block()`
    emits in exactly the schema's shape.

Design context: `epyc-root/handoffs/active/autokernel-research-loop.md` §6.4
(affected-surface derivation), §3.2 (backend-unchanged escape), §8.5.1 (source-integrity
gates and the `core_header` risk tier), §8.6 (T0_GATE: *"dispatch trace for
affected-surface confirmation (§6.4) and no-fallback proof"*), §7.3 (candidate record
carries *"derived and traced affected-surface manifests"*), invariant 18.

WHAT THIS MODULE IS NOT
-----------------------
It runs no build, no benchmark and no inference; it starts, stops and signals no
process; and it writes no file. It DOES read files — depfiles, link manifests, traces
and ELF binaries — because that is its input. `audit_surface_module_is_read_only()`
parses this module's own AST and FAILs on any write-capable call, any process call,
any forbidden import, and on any `open()` whose mode is not a read-only literal.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .. import schemas
from . import api

__all__ = [
    # provenance
    "PROVENANCE_BUILD_SYSTEM", "PROVENANCE_TOOL_EXTRACTED", "PROVENANCE_CAMPAIGN_MANIFEST",
    "PROVENANCE_ACTOR_DECLARED", "PROVENANCE_DIRECTORY_PREFIX_GUESS",
    "TRUSTED_CLOSURE_PROVENANCE",
    # errors
    "SurfaceError", "UntrustedProvenance", "SurfaceInputError", "DepfileParseError",
    "TraceParseError", "ElfFormatError", "NormalizationViolation", "NaiveByteIdentityRefused",
    # vocabularies
    "OBJECT_SUFFIXES", "SURFACE_AXES", "GATING_AXES", "OVER_APPROXIMATION_REASONS",
    "COMPARED_SECTIONS", "EXCLUDED_SECTIONS", "EXCLUDED_SECTION_PREFIXES",
    "DISPATCH_TRACE_SCHEMA", "FULL_TREE_CHANGE_CLASSES",
    # stage 1 inputs
    "DepEdge", "LinkEdge", "BuildDependencyIndex", "build_dependency_index",
    "parse_make_depfile", "parse_ninja_deps", "parse_cmake_link_txt",
    "parse_link_manifest_json", "load_make_depfile", "load_cmake_link_txt",
    "DiffEntry", "SourceDiff", "parse_git_name_status",
    "OpRegistration", "SymbolRegistrationIndex",
    # stage 1 output
    "OverApproximation", "AffectedSurface", "derive_affected_surface",
    # stage 2
    "DispatchEvent", "TracedSurface", "parse_dispatch_trace", "traced_surface_from_events",
    # stage 3
    "AxisReconciliation", "SurfaceReconciliation", "reconcile_surface",
    "candidate_affected_surface_block",
    # the declaration, scored not consumed
    "ActorDeclaration", "DeclarationScore", "score_actor_declaration",
    # §3.2
    "ToolchainIdentity", "NormalizedBinaryDigest", "read_normalized_binary_digest",
    "normalized_binary_digest_from_sections", "compare_binaries_byte_identical",
    "RebuildAttestation", "SourceClosureIdentity", "NormalizedBinaryIdentity",
    "EvidenceTransferScope", "BuildIdentityFinding", "BackendUnchangedResult",
    "backend_unchanged_stage1_source_closure", "backend_unchanged_stage2_normalized_binary",
    "backend_unchanged",
    # seam
    "SurfaceGateRunner", "audit_surface_module_is_read_only",
]


# =============================================================================
# Provenance — the trust boundary of stage 1
# =============================================================================

#: The build system emitted it (`gcc -MD` depfile, `ninja -t deps`, CMake `link.txt`).
PROVENANCE_BUILD_SYSTEM = "build_system_generated"
#: A tool extracted it from the artifact (`nm`, `readelf`, a clang index).
PROVENANCE_TOOL_EXTRACTED = "tool_extracted"
#: The campaign manifest declared it, under the measurement trust boundary.
PROVENANCE_CAMPAIGN_MANIFEST = "campaign_manifest"
#: The actor (the LLM writing the patch) declared it. Never a scope input.
PROVENANCE_ACTOR_DECLARED = "actor_declared"
#: A directory-prefix heuristic. Named so it can be REFUSED by name (§6.4).
PROVENANCE_DIRECTORY_PREFIX_GUESS = "directory_prefix_guess"

PROVENANCES = (
    PROVENANCE_BUILD_SYSTEM, PROVENANCE_TOOL_EXTRACTED, PROVENANCE_CAMPAIGN_MANIFEST,
    PROVENANCE_ACTOR_DECLARED, PROVENANCE_DIRECTORY_PREFIX_GUESS,
)

#: Only one provenance may supply a build-target dependency closure. §6.4: *"Take the
#: build-target dependency closure from the BUILD SYSTEM's own generated dependency
#: information, never a hand-maintained list or a directory-prefix guess."*
TRUSTED_CLOSURE_PROVENANCE = frozenset({PROVENANCE_BUILD_SYSTEM})

#: The symbol / op-registration / dispatch-predicate index may additionally come from a
#: tool run over the artifact, but never from the actor.
TRUSTED_SYMBOL_PROVENANCE = frozenset({PROVENANCE_BUILD_SYSTEM, PROVENANCE_TOOL_EXTRACTED})


# =============================================================================
# Errors — every one of them is a refusal, none is a degraded default
# =============================================================================

class SurfaceError(Exception):
    """Base class for every refusal in this module."""


class UntrustedProvenance(SurfaceError):
    """A scope input was offered from a source that may not supply scope."""


class SurfaceInputError(SurfaceError):
    """A structurally invalid input. Distinct from an input that cannot be evaluated:
    the latter is a `COULD_NOT_CHECK`, this is a wiring defect."""


class DepfileParseError(SurfaceError):
    """The build system's dependency output could not be parsed.

    It RAISES rather than returning the edges it managed to read, because a partially
    parsed depfile under-approximates the closure and an under-approximated closure is
    exactly the failure this module exists to prevent.
    """


class TraceParseError(SurfaceError):
    """The dispatch trace could not be parsed. A dropped trace line is an unobserved
    kernel execution, so this raises rather than skipping the line."""


class ElfFormatError(SurfaceError):
    """The binary is not an ELF64 little-endian object this reader understands."""


class NormalizationViolation(SurfaceError):
    """A normalized digest was asked to include a section §3.2 excludes."""


class NaiveByteIdentityRefused(SurfaceError):
    """Whole-binary byte comparison was requested. §3.2: *"The test is not naive
    byte-identity of the built binary. llama.cpp/ROCm builds embed build IDs,
    timestamps, and absolute paths … a test formulated that way would never fire."*"""


# =============================================================================
# Vocabularies
# =============================================================================

#: Suffixes that identify a compiled object in a depfile target or a link line.
OBJECT_SUFFIXES = (".o", ".obj")

#: The axes a surface manifest is compared on.
AXIS_BACKENDS = "backends"
AXIS_LINK_TARGETS = "link_targets"
AXIS_OP_NAMES = "op_names"
AXIS_KERNEL_SYMBOLS = "kernel_symbols"
AXIS_DISPATCH_PREDICATES = "dispatch_predicates"

SURFACE_AXES = (
    AXIS_BACKENDS, AXIS_LINK_TARGETS, AXIS_OP_NAMES, AXIS_KERNEL_SYMBOLS,
    AXIS_DISPATCH_PREDICATES,
)

#: Axes on which containment ALWAYS gates, whether or not a symbol index was supplied.
#: These two are what freeze scope is computed from: a traced backend or link target
#: outside the derived surface is the exploit invariant 18 names.
GATING_AXES = (AXIS_BACKENDS, AXIS_LINK_TARGETS)

# --- over-approximation reasons -------------------------------------------------
#: Mechanical: the build system's own closure fans a header out across link targets.
OA_SHARED_HEADER_FANOUT = "SHARED_HEADER_FANOUT"
#: Fail-closed: a touched file appears in no depfile in any supplied index.
OA_UNMAPPED_TOUCHED_FILE = "UNMAPPED_TOUCHED_FILE"
#: Fail-closed: an object in the closure appears in no link manifest.
OA_UNLINKED_OBJECT = "UNLINKED_OBJECT"
#: Fail-closed: a link target no backend claims.
OA_UNATTRIBUTED_LINK_TARGET = "UNATTRIBUTED_LINK_TARGET"
#: Fail-closed: ninja reported the target's recorded deps as STALE.
OA_STALE_DEPENDENCY_ENTRY = "STALE_DEPENDENCY_ENTRY"
#: Fail-closed: `change_class: core_header` forces full-tree surface (§8.5.1).
OA_CORE_HEADER_CHANGE_CLASS = "CORE_HEADER_CHANGE_CLASS"
#: Fail-closed: no symbol/registration index, so the symbol axes cannot be derived.
OA_NO_SYMBOL_INDEX = "NO_SYMBOL_INDEX"

OVER_APPROXIMATION_REASONS = (
    OA_SHARED_HEADER_FANOUT, OA_UNMAPPED_TOUCHED_FILE, OA_UNLINKED_OBJECT,
    OA_UNATTRIBUTED_LINK_TARGET, OA_STALE_DEPENDENCY_ENTRY, OA_CORE_HEADER_CHANGE_CLASS,
    OA_NO_SYMBOL_INDEX,
)

OA_KIND_MECHANICAL = "mechanical_closure"
OA_KIND_FAIL_CLOSED = "fail_closed"
OA_KINDS = (OA_KIND_MECHANICAL, OA_KIND_FAIL_CLOSED)

#: §8.5.1: *"A change to shared ggml core or to a widely-included header is not a large
#: edit — it is a different kind of edit … `change_class: core_header` forces full-tree
#: affected surface regardless of the textual diff size."*
FULL_TREE_CHANGE_CLASSES = frozenset({"core_header"})

# --- §3.2 normalized binary comparison ------------------------------------------
#: *"compare normalized hashes of `.text`, `.rodata`, `.data.rel.ro`, and the dynamic
#: symbol table"*. The dynamic symbol table is handled separately because hashing its
#: raw bytes would compare st_value addresses, not the symbol set.
COMPARED_SECTIONS = (".text", ".rodata", ".data.rel.ro")
#: *"excluding `.comment`, `.note.gnu.build-id`, and debug sections."*
EXCLUDED_SECTIONS = (".comment", ".note.gnu.build-id")
EXCLUDED_SECTION_PREFIXES = (".debug", ".zdebug", ".note.")
#: Sentinel written into a digest map when a compared section is not in the binary.
SECTION_ABSENT = "ABSENT"

#: Local schema id for the T0 dispatch trace. It has NO home in `schemas.py`, which is
#: the single source of truth and outside this task's write scope; the gap is recorded
#: in the module report rather than patched here (P-AK-SEARCH-1 denial 6: a controller
#: that discovers a coverage gap RECORDS the gap and does not patch the instrument).
DISPATCH_TRACE_SCHEMA = "epyc.autokernel.dispatch_trace.v1"


# =============================================================================
# Lexical path handling — no filesystem access, no `os` import
# =============================================================================

def _normalize_path(path: str) -> str:
    """Lexically normalize a POSIX path. Never touches the filesystem.

    Resolution is lexical on purpose: a depfile is read long after (and often
    elsewhere than) the build that produced it, and a `realpath()` here would silently
    resolve a symlink into a different tree.
    """
    if not isinstance(path, str) or not path.strip():
        raise SurfaceInputError(f"path must be a non-empty string, got {path!r}")
    absolute = path.startswith("/")
    parts: list = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not absolute:
                parts.append("..")
            continue
        parts.append(part)
    joined = "/".join(parts)
    if absolute:
        return "/" + joined
    return joined or "."


def _sorted_unique(values: Iterable[str]) -> tuple:
    return tuple(sorted({v for v in values}))


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SurfaceInputError(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _is_object(path: str) -> bool:
    return path.endswith(OBJECT_SUFFIXES)


def _combine_checks(*checks: schemas.Check) -> schemas.Check:
    """Worst-of over Checks: FAIL beats COULD_NOT_CHECK beats PASS."""
    outcome = schemas.PASS
    reasons: list = []
    for chk in checks:
        if chk.outcome == schemas.PASS:
            continue
        reasons.extend(chk.reasons)
        if chk.outcome == schemas.FAIL:
            outcome = schemas.FAIL
        elif outcome != schemas.FAIL:
            outcome = schemas.COULD_NOT_CHECK
    return schemas.Check(outcome, tuple(reasons))


# =============================================================================
# STAGE 1a — parsing the build system's own dependency output
# =============================================================================

@dataclass(frozen=True)
class DepEdge:
    """One build target and the prerequisites the BUILD SYSTEM recorded for it."""

    target: str
    prerequisites: tuple
    origin_ref: str
    valid: bool = True
    invalidity_reason: Optional[str] = None

    def __post_init__(self) -> None:
        _require_str(self.target, "DepEdge.target")
        _require_str(self.origin_ref, "DepEdge.origin_ref")
        if not isinstance(self.prerequisites, tuple):
            raise SurfaceInputError("DepEdge.prerequisites must be a tuple")
        if not self.valid and not self.invalidity_reason:
            raise SurfaceInputError(
                "DepEdge marked invalid must name why; an unexplained invalid edge is "
                "indistinguishable from a parse bug")


@dataclass(frozen=True)
class LinkEdge:
    """One link target, the objects the BUILD SYSTEM links into it, and the
    LIBRARIES it links.

    `library_inputs` exists because a link line is not flat. In llama.cpp the
    benchmark binary links `ggml/src/libggml.so`, not `ggml.c.o` — so a link edge
    that recorded only `.o` tokens would report `bin/llama-bench`'s closure as
    "llama-bench.cpp only" and a `ggml/src/ggml-cpu.c` edit would fall OUTSIDE it.
    `build_dependency_index()` folds a library input's objects into every target
    that links it, and records one that resolves to no known link edge as an
    unresolved input so the closure fails closed instead of under-approximating.
    """

    link_target: str
    objects: tuple
    origin_ref: str
    unresolved_inputs: tuple = ()
    library_inputs: tuple = ()

    def __post_init__(self) -> None:
        _require_str(self.link_target, "LinkEdge.link_target")
        _require_str(self.origin_ref, "LinkEdge.origin_ref")
        for name in ("objects", "unresolved_inputs", "library_inputs"):
            if not isinstance(getattr(self, name), tuple):
                raise SurfaceInputError(f"LinkEdge.{name} must be a tuple")


_TRAILING_BACKSLASHES = re.compile(r"(\\*)$")
_COLON = object()


def _logical_lines(text: str) -> list:
    """Join Make line continuations. An odd run of trailing backslashes continues."""
    out: list = []
    buf = ""
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        match = _TRAILING_BACKSLASHES.search(line)
        trailing = len(match.group(1)) if match else 0
        if trailing % 2 == 1:
            buf += line[:-1] + " "
            continue
        buf += line
        out.append(buf)
        buf = ""
    if buf.strip():
        out.append(buf)
    return out


def _tokenize_make(line: str) -> list:
    """Tokenize one logical Make line, honouring `\\ `, `\\:`, `\\\\` and `$$`."""
    out: list = []
    cur: list = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and i + 1 < n and line[i + 1] in " \t:\\#":
            cur.append(line[i + 1])
            i += 2
            continue
        if ch == "$" and i + 1 < n and line[i + 1] == "$":
            cur.append("$")
            i += 2
            continue
        if ch in " \t":
            if cur:
                out.append("".join(cur))
                cur = []
            i += 1
            continue
        if ch == ":":
            if cur:
                out.append("".join(cur))
                cur = []
            out.append(_COLON)
            i += 1
            continue
        cur.append(ch)
        i += 1
    if cur:
        out.append("".join(cur))
    return out


def parse_make_depfile(text: str, *, origin_ref: str) -> tuple:
    """Parse a Make-syntax depfile (`gcc -MD`, CMake `depend.make`).

    Returns every recorded edge, including gcc `-MP` phony header targets, which are
    marked `valid=False` with reason `PHONY_TARGET`: they carry no prerequisites and
    treating them as build targets would make every header look like an object with an
    empty closure.

    Raises `DepfileParseError` on a line with more than one unescaped `:` or with
    prerequisites but no target — a shape this parser does not understand. It does not
    guess, because a guess here silently shrinks the closure.
    """
    _require_str(origin_ref, "origin_ref")
    if not isinstance(text, str):
        raise SurfaceInputError(f"depfile text must be a string, got {type(text).__name__}")
    edges: list = []
    for lineno, line in enumerate(_logical_lines(text), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = _tokenize_make(line)
        colons = [i for i, t in enumerate(tokens) if t is _COLON]
        if not colons:
            raise DepfileParseError(
                f"{origin_ref}:{lineno}: no ':' separator in {stripped!r}; this parser "
                "does not guess at Make syntax it was not given")
        if len(colons) > 1:
            raise DepfileParseError(
                f"{origin_ref}:{lineno}: {len(colons)} unescaped ':' separators in "
                f"{stripped!r}; escape them as '\\:' or fix the generator")
        idx = colons[0]
        targets = [t for t in tokens[:idx] if t is not _COLON]
        prereqs = [t for t in tokens[idx + 1:] if t is not _COLON]
        if not targets:
            raise DepfileParseError(f"{origin_ref}:{lineno}: rule with no target: {stripped!r}")
        for target in targets:
            if not prereqs and not _is_object(target):
                edges.append(DepEdge(target=target, prerequisites=(), origin_ref=origin_ref,
                                     valid=False, invalidity_reason="PHONY_TARGET"))
            else:
                edges.append(DepEdge(target=target, prerequisites=tuple(prereqs),
                                     origin_ref=origin_ref))
    return tuple(edges)


_NINJA_HEADER = re.compile(r"^(?P<target>.+?):\s*#deps\s+(?P<count>\d+)"
                           r"(?:,\s*deps mtime\s+\d+)?\s*\((?P<state>[A-Z]+)\)\s*$")


def parse_ninja_deps(text: str, *, origin_ref: str) -> tuple:
    """Parse `ninja -t deps` output.

    A `(STALE)` entry is returned with `valid=False`: ninja is telling us its recorded
    dependency information no longer matches the build, and a stale entry is precisely
    an under-approximation of the closure. `build_dependency_index()` refuses to fold a
    stale entry into the closure and records it, so the derivation widens instead.
    """
    _require_str(origin_ref, "origin_ref")
    if not isinstance(text, str):
        raise SurfaceInputError(f"ninja deps text must be a string, got {type(text).__name__}")
    edges: list = []
    target: Optional[str] = None
    state: Optional[str] = None
    declared = 0
    prereqs: list = []

    def _flush(lineno: int) -> None:
        nonlocal target, state, declared, prereqs
        if target is None:
            return
        if len(prereqs) != declared:
            raise DepfileParseError(
                f"{origin_ref}:{lineno}: target {target!r} declared #deps {declared} but "
                f"{len(prereqs)} were listed; a truncated deps block silently shrinks the "
                "closure")
        if state == "VALID":
            edges.append(DepEdge(target=target, prerequisites=tuple(prereqs),
                                 origin_ref=origin_ref))
        else:
            edges.append(DepEdge(target=target, prerequisites=tuple(prereqs),
                                 origin_ref=origin_ref, valid=False,
                                 invalidity_reason=f"NINJA_DEPS_{state}"))
        target, state, declared, prereqs = None, None, 0, []

    for lineno, raw in enumerate(text.split("\n"), start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            _flush(lineno)
            continue
        if line[0] in " \t":
            if target is None:
                raise DepfileParseError(
                    f"{origin_ref}:{lineno}: indented prerequisite {line.strip()!r} with no "
                    "preceding target header")
            prereqs.append(line.strip())
            continue
        _flush(lineno)
        match = _NINJA_HEADER.match(line)
        if not match:
            raise DepfileParseError(
                f"{origin_ref}:{lineno}: not a `ninja -t deps` target header: {line!r}")
        target = match.group("target").strip()
        state = match.group("state")
        declared = int(match.group("count"))
        prereqs = []
    _flush(len(text.split("\n")) + 1)
    return tuple(edges)


_RSP_TOKEN = re.compile(r"^@(?P<path>.+)$")

#: A path-like library input on a link line: `libggml.so`, `libggml.so.0.4.2`,
#: `libggml.a`. `-lggml` and `-L<dir>` are NOT matched — they are name-based
#: references this parser cannot resolve to a build target, and they are reported
#: as a residual limitation rather than silently treated as "no library".
_LIBRARY_INPUT_RE = re.compile(r"\.(?:a|so)(?:\.\d+)*$")


def _is_library_input(token: str) -> bool:
    return bool(token) and not token.startswith("-") and bool(_LIBRARY_INPUT_RE.search(token))


def _shell_split(command: str) -> list:
    """Minimal quote-aware split for a CMake `link.txt` command line."""
    tokens: list = []
    cur: list = []
    quote: Optional[str] = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote:
            if ch == "\\" and i + 1 < n:
                cur.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
                i += 1
                continue
            cur.append(ch)
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            cur.append(command[i + 1])
            i += 2
            continue
        if ch.isspace():
            if cur:
                tokens.append("".join(cur))
                cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    if quote:
        raise DepfileParseError(f"unterminated {quote!r} quote in link command: {command!r}")
    if cur:
        tokens.append("".join(cur))
    return tokens


def parse_cmake_link_txt(text: str, *, origin_ref: str,
                         link_target: Optional[str] = None) -> LinkEdge:
    """Parse a CMake `CMakeFiles/<target>.dir/link.txt` link command.

    The link target is taken from the command's own `-o` argument. A `@response.rsp`
    input is recorded in `unresolved_inputs` — not ignored: an unexpanded response file
    hides every object it names, and `build_dependency_index()` propagates it so the
    derivation widens rather than silently under-approximating.

    A path-like library input (`libggml.so`, `libggml.a`) is recorded in
    `library_inputs` for the same reason: a real llama.cpp link line names the shared
    library, not the objects inside it, and dropping it would make the target's closure
    silently exclude every source compiled into that library.
    """
    _require_str(origin_ref, "origin_ref")
    tokens = _shell_split(text.strip())
    if not tokens:
        raise DepfileParseError(f"{origin_ref}: empty link command")
    objects: list = []
    unresolved: list = []
    libraries: list = []
    output: Optional[str] = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "-o" and i + 1 < len(tokens):
            output = tokens[i + 1]
            i += 2
            continue
        rsp = _RSP_TOKEN.match(token)
        if rsp:
            unresolved.append(rsp.group("path"))
            i += 1
            continue
        if _is_object(token):
            objects.append(token)
        elif _is_library_input(token):
            libraries.append(token)
        i += 1
    resolved_target = link_target or output
    if not resolved_target:
        raise DepfileParseError(
            f"{origin_ref}: link command names no output (`-o`) and no link_target was "
            "supplied; the link edge would have no identity")
    if link_target and output and link_target != output:
        raise DepfileParseError(
            f"{origin_ref}: link_target {link_target!r} disagrees with the command's "
            f"-o {output!r}")
    return LinkEdge(link_target=resolved_target, objects=tuple(objects),
                    origin_ref=origin_ref, unresolved_inputs=tuple(unresolved),
                    library_inputs=tuple(libraries))


def parse_link_manifest_json(text: str, *, origin_ref: str) -> tuple:
    """Parse a generated JSON link manifest: `{"links": [{"link_target", "objects"}]}`.

    This is the escape hatch for a build system that is neither CMake+Make nor
    CMake+Ninja. It is still build-system output: the caller must generate it from the
    build graph, and `build_dependency_index()` still stamps it
    `build_system_generated` only if the caller says so.
    """
    _require_str(origin_ref, "origin_ref")
    try:
        obj = json.loads(text)
    except ValueError as exc:
        raise DepfileParseError(f"{origin_ref}: not JSON: {exc}") from exc
    if not isinstance(obj, Mapping) or not isinstance(obj.get("links"), list):
        raise DepfileParseError(f"{origin_ref}: expected {{'links': [...]}}")
    edges: list = []
    for i, entry in enumerate(obj["links"]):
        if not isinstance(entry, Mapping):
            raise DepfileParseError(f"{origin_ref}: links[{i}] is not an object")
        target = entry.get("link_target")
        objects = entry.get("objects")
        if not isinstance(target, str) or not target:
            raise DepfileParseError(f"{origin_ref}: links[{i}].link_target missing")
        if not isinstance(objects, list) or not all(isinstance(o, str) for o in objects):
            raise DepfileParseError(f"{origin_ref}: links[{i}].objects must be a list of str")
        unresolved = entry.get("unresolved_inputs", [])
        if not isinstance(unresolved, list):
            raise DepfileParseError(f"{origin_ref}: links[{i}].unresolved_inputs must be a list")
        libraries = entry.get("library_inputs", [])
        if not isinstance(libraries, list) or not all(isinstance(o, str) for o in libraries):
            raise DepfileParseError(
                f"{origin_ref}: links[{i}].library_inputs must be a list of str")
        edges.append(LinkEdge(link_target=target, objects=tuple(objects),
                              origin_ref=f"{origin_ref}#links[{i}]",
                              unresolved_inputs=tuple(unresolved),
                              library_inputs=tuple(libraries)))
    return tuple(edges)


def _read_text(path: Any, label: str) -> str:
    """Read a UTF-8 text input. Raises on anything unreadable — never returns ''."""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SurfaceInputError(f"{label}: cannot read {p}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise SurfaceInputError(f"{label}: {p} is not UTF-8 text: {exc}") from exc


def load_make_depfile(path: Any) -> tuple:
    """Read and parse a Make-syntax depfile from disk."""
    return parse_make_depfile(_read_text(path, "depfile"), origin_ref=str(path))


def load_cmake_link_txt(path: Any, *, link_target: Optional[str] = None) -> LinkEdge:
    """Read and parse a CMake `link.txt` from disk."""
    return parse_cmake_link_txt(_read_text(path, "link.txt"), origin_ref=str(path),
                                link_target=link_target)


# =============================================================================
# STAGE 1b — the build dependency index
# =============================================================================

@dataclass(frozen=True)
class BuildDependencyIndex:
    """The build system's dependency graph, resolved to repo-relative paths.

    Constructed by `build_dependency_index()`. Its `__post_init__` REFUSES any
    provenance outside `TRUSTED_CLOSURE_PROVENANCE`, so §6.4's *"never a
    hand-maintained list or a directory-prefix guess"* is a type error rather than a
    review comment.
    """

    label: str
    build_dir: str
    source_root: str
    dep_edges: tuple
    link_edges: tuple
    backend_link_targets: tuple  # ((backend, (link_target, ...)), ...)
    provenance: str = PROVENANCE_BUILD_SYSTEM
    #: The backend<-link-target association comes from the campaign manifest, which
    #: lives under the measurement trust boundary — but every target it names must
    #: exist in the build system's own link edges, checked below.
    backend_map_provenance: str = PROVENANCE_CAMPAIGN_MANIFEST

    _objects_by_source: dict = field(default_factory=dict, repr=False, compare=False)
    _sources_by_object: dict = field(default_factory=dict, repr=False, compare=False)
    _links_by_object: dict = field(default_factory=dict, repr=False, compare=False)
    _objects_by_link: dict = field(default_factory=dict, repr=False, compare=False)
    _backends_by_link: dict = field(default_factory=dict, repr=False, compare=False)
    _stale_targets: tuple = field(default=(), repr=False, compare=False)
    _phony_targets: tuple = field(default=(), repr=False, compare=False)
    _external_prerequisites: tuple = field(default=(), repr=False, compare=False)
    _unresolved_link_inputs: tuple = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.provenance not in TRUSTED_CLOSURE_PROVENANCE:
            raise UntrustedProvenance(
                f"BuildDependencyIndex provenance {self.provenance!r} may not supply a "
                f"build-target dependency closure; only {sorted(TRUSTED_CLOSURE_PROVENANCE)} "
                "may. §6.4: take the closure from the build system's own generated "
                "dependency information, never a hand-maintained list or a "
                "directory-prefix guess.")
        if self.backend_map_provenance == PROVENANCE_ACTOR_DECLARED:
            raise UntrustedProvenance(
                "the backend<-link-target map may not be actor-declared: it is the "
                "mapping freeze scope is computed from (invariant 18)")

    # --- lookups ---------------------------------------------------------------
    def objects_for_source(self, source: str) -> tuple:
        return tuple(sorted(self._objects_by_source.get(source, ())))

    def sources_for_object(self, obj: str) -> tuple:
        return tuple(sorted(self._sources_by_object.get(obj, ())))

    def link_targets_for_object(self, obj: str) -> tuple:
        return tuple(sorted(self._links_by_object.get(obj, ())))

    def objects_for_link_target(self, target: str) -> tuple:
        return tuple(sorted(self._objects_by_link.get(target, ())))

    def backends_for_link_target(self, target: str) -> tuple:
        return tuple(sorted(self._backends_by_link.get(target, ())))

    def link_targets_for_backend(self, backend: str) -> tuple:
        for name, targets in self.backend_link_targets:
            if name == backend:
                return targets
        return ()

    def source_closure_for_backend(self, backend: str) -> tuple:
        """Every build input the backend's link targets transitively depend on."""
        out: set = set()
        for target in self.link_targets_for_backend(backend):
            for obj in self.objects_for_link_target(target):
                out.update(self.sources_for_object(obj))
        return tuple(sorted(out))

    @property
    def known_backends(self) -> tuple:
        return tuple(sorted(name for name, _ in self.backend_link_targets))

    @property
    def known_link_targets(self) -> tuple:
        return tuple(sorted({e.link_target for e in self.link_edges}))

    @property
    def known_sources(self) -> tuple:
        return tuple(sorted(self._objects_by_source))

    @property
    def stale_targets(self) -> tuple:
        return self._stale_targets

    @property
    def phony_targets(self) -> tuple:
        return self._phony_targets

    @property
    def external_prerequisites(self) -> tuple:
        return self._external_prerequisites

    @property
    def unresolved_link_inputs(self) -> tuple:
        return self._unresolved_link_inputs

    def coverage_check(self) -> schemas.Check:
        """COULD_NOT_CHECK when the build system's own output is incomplete."""
        reasons: list = []
        if self._stale_targets:
            reasons.append(
                f"{len(self._stale_targets)} dependency entries are STALE "
                f"(e.g. {self._stale_targets[0]}); ninja is reporting its recorded deps no "
                "longer match the build, so the closure under-approximates")
        if self._unresolved_link_inputs:
            reasons.append(
                f"{len(self._unresolved_link_inputs)} link inputs are unexpanded response "
                f"files (e.g. {self._unresolved_link_inputs[0]}); every object they name is "
                "invisible to the closure")
        if not self.link_edges:
            reasons.append("no link manifest was supplied; objects cannot be attributed to "
                           "link targets or backends")
        if reasons:
            return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "build_dir": self.build_dir,
            "source_root": self.source_root,
            "provenance": self.provenance,
            "backend_map_provenance": self.backend_map_provenance,
            "dep_edge_count": len(self.dep_edges),
            "link_edge_count": len(self.link_edges),
            "backends": list(self.known_backends),
            "stale_targets": list(self._stale_targets),
            "phony_targets": list(self._phony_targets),
            "unresolved_link_inputs": list(self._unresolved_link_inputs),
            "external_prerequisite_count": len(self._external_prerequisites),
        }


def _resolve_against(build_dir: str, source_root: str, raw: str) -> tuple:
    """Return `(kind, path)` where kind is 'internal' (repo-relative) or 'external'."""
    root = _normalize_path(source_root)
    if raw.startswith("/"):
        norm = _normalize_path(raw)
        if norm == root:
            return ("external", norm)
        if norm.startswith(root.rstrip("/") + "/"):
            return ("internal", norm[len(root.rstrip("/")) + 1:])
        return ("external", norm)
    joined = _normalize_path(f"{build_dir}/{raw}")
    if joined.startswith("../") or joined == "..":
        return ("external", joined)
    return ("internal", joined)


def build_dependency_index(*,
                           label: str,
                           build_dir: str,
                           source_root: str,
                           dep_edges: Sequence[DepEdge],
                           link_edges: Sequence[LinkEdge],
                           backend_link_targets: Mapping[str, Sequence[str]],
                           provenance: str = PROVENANCE_BUILD_SYSTEM,
                           backend_map_provenance: str = PROVENANCE_CAMPAIGN_MANIFEST,
                           ) -> BuildDependencyIndex:
    """Resolve build-system output into a repo-relative dependency index.

    Raises rather than degrading on: an untrusted provenance, a declared backend that
    is not a known backend, a declared link target the build system never emitted, and
    a `backend_link_targets` entry that is empty. Each of those would otherwise produce
    an index that answers "nothing is affected" for a backend that is.
    """
    _require_str(label, "label")
    _require_str(build_dir, "build_dir")
    _require_str(source_root, "source_root")
    if provenance not in PROVENANCES:
        raise SurfaceInputError(f"provenance {provenance!r} is not one of {list(PROVENANCES)}")
    if not isinstance(backend_link_targets, Mapping):
        raise SurfaceInputError("backend_link_targets must be a mapping of backend -> targets")
    if not backend_link_targets:
        raise SurfaceInputError(
            "backend_link_targets is empty; an index that attributes no link target to any "
            "backend answers 'nothing affected' for every change, which is the exact failure "
            "invariant 18 exists to prevent")

    build_dir_n = _normalize_path(build_dir)
    source_root_n = _normalize_path(source_root)

    objects_by_source: dict = {}
    sources_by_object: dict = {}
    stale: list = []
    phony: list = []
    external: set = set()

    for edge in dep_edges:
        if not isinstance(edge, DepEdge):
            raise SurfaceInputError(f"dep_edges must be DepEdge, got {type(edge).__name__}")
        kind, target = _resolve_against(build_dir_n, source_root_n, edge.target)
        if not edge.valid:
            if edge.invalidity_reason == "PHONY_TARGET":
                phony.append(target)
            else:
                stale.append(target)
            continue
        if kind == "external":
            external.add(target)
            continue
        if not _is_object(target):
            # A valid rule whose target is not an object is not a compile edge; it is
            # kept out of the object graph and surfaced, never silently dropped.
            phony.append(target)
            continue
        bucket = sources_by_object.setdefault(target, set())
        for raw in edge.prerequisites:
            pkind, prereq = _resolve_against(build_dir_n, source_root_n, raw)
            if pkind == "external":
                external.add(prereq)
                continue
            bucket.add(prereq)
            objects_by_source.setdefault(prereq, set()).add(target)

    links_by_object: dict = {}
    objects_by_link: dict = {}
    unresolved: list = []
    library_edges: dict = {}     # link target -> resolved library inputs
    for edge in link_edges:
        if not isinstance(edge, LinkEdge):
            raise SurfaceInputError(f"link_edges must be LinkEdge, got {type(edge).__name__}")
        _, target = _resolve_against(build_dir_n, source_root_n, edge.link_target)
        for raw in edge.unresolved_inputs:
            unresolved.append(f"{target}:{raw}")
        bucket = objects_by_link.setdefault(target, set())
        for raw_obj in edge.objects:
            okind, obj = _resolve_against(build_dir_n, source_root_n, raw_obj)
            if okind == "external":
                external.add(obj)
                continue
            bucket.add(obj)
        libs = library_edges.setdefault(target, set())
        for raw_lib in edge.library_inputs:
            _, lib = _resolve_against(build_dir_n, source_root_n, raw_lib)
            libs.add(lib)

    # --- transitive link closure ------------------------------------------------
    # A link line names libraries, not the objects inside them: llama.cpp's
    # `llama-bench` links `ggml/src/libggml.so`. Without this fold, the target's
    # closure would exclude every ggml source and `backend_unchanged_stage1_source_
    # closure()` would answer PASS ("unchanged") for a ggml kernel edit — silently,
    # because a dropped library input is none of the fail-closed classes.
    # A library input that resolves to NO known link edge is recorded as an
    # unresolved link input, which `coverage_check()` already reports, so an
    # incomplete link manifest fails closed instead of under-approximating.
    for target, libs in library_edges.items():
        for lib in sorted(libs):
            if lib not in objects_by_link:
                unresolved.append(
                    f"{target}:library input {lib} resolves to no link edge in this "
                    "index; every object linked into it is invisible to the closure")
    changed_any = True
    while changed_any:
        changed_any = False
        for target, libs in library_edges.items():
            bucket = objects_by_link.setdefault(target, set())
            for lib in libs:
                inner = objects_by_link.get(lib)
                if inner and not inner <= bucket:
                    bucket |= inner
                    changed_any = True
    for target, objs in objects_by_link.items():
        for obj in objs:
            links_by_object.setdefault(obj, set()).add(target)

    normalized_backends: list = []
    backends_by_link: dict = {}
    for backend, targets in backend_link_targets.items():
        if backend not in schemas.BACKENDS:
            raise SurfaceInputError(
                f"backend {backend!r} is not one of {sorted(schemas.BACKENDS)}")
        if not targets:
            raise SurfaceInputError(
                f"backend_link_targets[{backend!r}] is empty; a backend with no link target "
                "would be silently unaffectable by any change")
        resolved: list = []
        for raw_target in targets:
            _, target = _resolve_against(build_dir_n, source_root_n, raw_target)
            if target not in objects_by_link:
                raise SurfaceInputError(
                    f"backend_link_targets[{backend!r}] names {raw_target!r}, which the "
                    f"build system's link manifest does not contain (known: "
                    f"{sorted(objects_by_link)}). A declared target the build never emitted "
                    "produces an empty closure, which reads as 'unaffected'.")
            resolved.append(target)
            backends_by_link.setdefault(target, set()).add(backend)
        normalized_backends.append((backend, tuple(sorted(set(resolved)))))

    # A library linked into a backend's binary is part of that backend. Without
    # this, `libggml.so` would be a link target no backend claims, and every
    # candidate touching ggml would trip OA_UNATTRIBUTED_LINK_TARGET and widen to
    # the whole tree — a fail-closed answer, but one that makes the derivation
    # useless on the tree it was written for.
    changed_any = True
    while changed_any:
        changed_any = False
        for target, libs in library_edges.items():
            owners = backends_by_link.get(target)
            if not owners:
                continue
            for lib in libs:
                if lib not in objects_by_link:
                    continue
                bucket = backends_by_link.setdefault(lib, set())
                if not owners <= bucket:
                    bucket |= owners
                    changed_any = True

    return BuildDependencyIndex(
        label=label,
        build_dir=build_dir_n,
        source_root=source_root_n,
        dep_edges=tuple(dep_edges),
        link_edges=tuple(link_edges),
        backend_link_targets=tuple(sorted(normalized_backends)),
        provenance=provenance,
        backend_map_provenance=backend_map_provenance,
        _objects_by_source={k: frozenset(v) for k, v in objects_by_source.items()},
        _sources_by_object={k: frozenset(v) for k, v in sources_by_object.items()},
        _links_by_object={k: frozenset(v) for k, v in links_by_object.items()},
        _objects_by_link={k: frozenset(v) for k, v in objects_by_link.items()},
        _backends_by_link={k: frozenset(v) for k, v in backends_by_link.items()},
        _stale_targets=tuple(sorted(set(stale))),
        _phony_targets=tuple(sorted(set(phony))),
        _external_prerequisites=tuple(sorted(external)),
        _unresolved_link_inputs=tuple(sorted(set(unresolved))),
    )


# =============================================================================
# STAGE 1c — the diff
# =============================================================================

CHANGE_KINDS = ("added", "modified", "deleted", "renamed", "copied", "typechange")

_GIT_STATUS_LETTER = {
    "A": "added", "M": "modified", "D": "deleted", "R": "renamed",
    "C": "copied", "T": "typechange",
}


@dataclass(frozen=True)
class DiffEntry:
    """One path the diff touched. A rename touches BOTH paths."""

    path: str
    change_kind: str
    old_path: Optional[str] = None

    def __post_init__(self) -> None:
        _require_str(self.path, "DiffEntry.path")
        if self.change_kind not in CHANGE_KINDS:
            raise SurfaceInputError(
                f"DiffEntry.change_kind {self.change_kind!r} is not one of {list(CHANGE_KINDS)}")
        if self.change_kind in ("renamed", "copied") and not self.old_path:
            raise SurfaceInputError(
                f"a {self.change_kind} entry must carry old_path; the old path is a touched "
                "file too and dropping it under-approximates the surface")

    @property
    def touched_paths(self) -> tuple:
        if self.old_path:
            return (_normalize_path(self.old_path), _normalize_path(self.path))
        return (_normalize_path(self.path),)


@dataclass(frozen=True)
class SourceDiff:
    """`production_base..candidate`, as the version-control system reported it."""

    base_commit: str
    candidate_commit: str
    entries: tuple
    origin_ref: str
    provenance: str = PROVENANCE_BUILD_SYSTEM

    def __post_init__(self) -> None:
        for name in ("base_commit", "candidate_commit", "origin_ref"):
            _require_str(getattr(self, name), f"SourceDiff.{name}")
        if not isinstance(self.entries, tuple):
            raise SurfaceInputError("SourceDiff.entries must be a tuple")
        for entry in self.entries:
            if not isinstance(entry, DiffEntry):
                raise SurfaceInputError(
                    f"SourceDiff.entries must be DiffEntry, got {type(entry).__name__}")
        if self.provenance == PROVENANCE_ACTOR_DECLARED:
            raise UntrustedProvenance(
                "the diff may not be actor-declared: it is the input the whole derivation "
                "keys off, so an actor-supplied diff would set the actor's own scope")

    @property
    def touched_paths(self) -> tuple:
        out: set = set()
        for entry in self.entries:
            out.update(entry.touched_paths)
        return tuple(sorted(out))

    def to_dict(self) -> dict:
        return {
            "base_commit": self.base_commit,
            "candidate_commit": self.candidate_commit,
            "origin_ref": self.origin_ref,
            "provenance": self.provenance,
            "entries": [{"path": e.path, "change_kind": e.change_kind,
                         "old_path": e.old_path} for e in self.entries],
        }


def parse_git_name_status(text: str, *, base_commit: str, candidate_commit: str,
                          origin_ref: str) -> SourceDiff:
    """Parse `git diff --name-status <base>..<candidate>` output.

    Raises on an unrecognised status letter rather than skipping the line: a skipped
    line is a touched file that never enters the closure.
    """
    entries: list = []
    for lineno, raw in enumerate(text.split("\n"), start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0].strip()
        letter = status[:1]
        if letter not in _GIT_STATUS_LETTER:
            raise SurfaceInputError(
                f"{origin_ref}:{lineno}: unrecognised git status {status!r} in {line!r}")
        kind = _GIT_STATUS_LETTER[letter]
        if kind in ("renamed", "copied"):
            if len(parts) != 3:
                raise SurfaceInputError(
                    f"{origin_ref}:{lineno}: a {kind} line needs old and new paths: {line!r}")
            entries.append(DiffEntry(path=parts[2].strip(), change_kind=kind,
                                     old_path=parts[1].strip()))
            continue
        if len(parts) != 2:
            raise SurfaceInputError(f"{origin_ref}:{lineno}: expected 'STATUS\\tpath': {line!r}")
        entries.append(DiffEntry(path=parts[1].strip(), change_kind=kind))
    return SourceDiff(base_commit=base_commit, candidate_commit=candidate_commit,
                      entries=tuple(entries), origin_ref=origin_ref)


# =============================================================================
# STAGE 1d — symbols, op registrations, dispatch predicates
# =============================================================================

@dataclass(frozen=True)
class OpRegistration:
    """One op registration: the op name, the backend it registers on, its predicate."""

    op_name: str
    backend: str
    dispatch_predicate: Optional[str] = None

    def __post_init__(self) -> None:
        _require_str(self.op_name, "OpRegistration.op_name")
        if self.backend not in schemas.BACKENDS:
            raise SurfaceInputError(
                f"OpRegistration.backend {self.backend!r} is not one of "
                f"{sorted(schemas.BACKENDS)}")

    def to_dict(self) -> dict:
        return {"op_name": self.op_name, "backend": self.backend,
                "dispatch_predicate": self.dispatch_predicate}


@dataclass(frozen=True)
class SymbolRegistrationIndex:
    """source file -> symbols -> op registrations -> dispatch predicates.

    Tool-extracted (`nm`, a clang index) or build-system generated. **Never
    actor-declared** — refused in `__post_init__`, because the registration table is
    what turns "this file changed" into "this op's dispatch changed", and an actor that
    can edit it can hide a dispatch change.
    """

    label: str
    symbols_by_source: Mapping[str, tuple]
    registrations_by_symbol: Mapping[str, tuple]
    provenance: str = PROVENANCE_TOOL_EXTRACTED

    def __post_init__(self) -> None:
        _require_str(self.label, "SymbolRegistrationIndex.label")
        if self.provenance not in TRUSTED_SYMBOL_PROVENANCE:
            raise UntrustedProvenance(
                f"SymbolRegistrationIndex provenance {self.provenance!r} may not supply op "
                f"registrations; only {sorted(TRUSTED_SYMBOL_PROVENANCE)} may")
        for source, symbols in self.symbols_by_source.items():
            if not isinstance(symbols, tuple):
                raise SurfaceInputError(f"symbols_by_source[{source!r}] must be a tuple")
        for symbol, regs in self.registrations_by_symbol.items():
            if not isinstance(regs, tuple):
                raise SurfaceInputError(f"registrations_by_symbol[{symbol!r}] must be a tuple")
            for reg in regs:
                if not isinstance(reg, OpRegistration):
                    raise SurfaceInputError(
                        f"registrations_by_symbol[{symbol!r}] must hold OpRegistration, got "
                        f"{type(reg).__name__}")

    def symbols_for(self, source: str) -> tuple:
        return tuple(self.symbols_by_source.get(source, ()))

    def registrations_for(self, symbol: str) -> tuple:
        return tuple(self.registrations_by_symbol.get(symbol, ()))

    @property
    def all_symbols(self) -> tuple:
        out: set = set()
        for symbols in self.symbols_by_source.values():
            out.update(symbols)
        out.update(self.registrations_by_symbol)
        return tuple(sorted(out))

    @property
    def all_registrations(self) -> tuple:
        out: list = []
        for regs in self.registrations_by_symbol.values():
            out.extend(regs)
        return tuple(out)


# =============================================================================
# STAGE 1 — the derived affected surface
# =============================================================================

@dataclass(frozen=True)
class OverApproximation:
    """One widening of the surface, with its reason and what it widened to.

    §6.4: *"Over-approximation is expected and acceptable (a shared-header change
    implies the whole tree until proven otherwise)."* Widening is correct; widening
    SILENTLY is not, which is why every one of them is a record.
    """

    reason: str
    kind: str
    trigger: str
    widened_to: tuple

    def __post_init__(self) -> None:
        if self.reason not in OVER_APPROXIMATION_REASONS:
            raise SurfaceInputError(
                f"over-approximation reason {self.reason!r} is not one of "
                f"{list(OVER_APPROXIMATION_REASONS)}")
        if self.kind not in OA_KINDS:
            raise SurfaceInputError(f"over-approximation kind {self.kind!r} is not one of "
                                    f"{list(OA_KINDS)}")

    def to_dict(self) -> dict:
        return {"reason": self.reason, "kind": self.kind, "trigger": self.trigger,
                "widened_to": list(self.widened_to)}


@dataclass(frozen=True)
class AffectedSurface:
    """The DERIVED affected-surface manifest — stage 1's output.

    `coverage` is a three-outcome Check, not a boolean: PASS when every touched file
    resolved through the build system's own dependency information, COULD_NOT_CHECK
    when any did not. A COULD_NOT_CHECK coverage always comes with a fail-closed
    `OverApproximation`, so the manifest is still a valid superset — but the record
    says the superset was reached by widening rather than by derivation.
    """

    candidate_id: str
    backends: tuple
    link_targets: tuple
    objects: tuple
    touched_files: tuple
    symbols: tuple
    op_registrations: tuple  # tuple[OpRegistration, ...]
    dispatch_predicates: tuple
    over_approximations: tuple
    axes_derived: tuple
    coverage: schemas.Check
    full_tree: bool
    inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_str(self.candidate_id, "AffectedSurface.candidate_id")
        for axis in self.axes_derived:
            if axis not in SURFACE_AXES:
                raise SurfaceInputError(f"axis {axis!r} is not one of {list(SURFACE_AXES)}")

    @property
    def op_names(self) -> tuple:
        return _sorted_unique(r.op_name for r in self.op_registrations)

    @property
    def fail_closed_widenings(self) -> tuple:
        return tuple(o for o in self.over_approximations if o.kind == OA_KIND_FAIL_CLOSED)

    def axis_values(self, axis: str) -> tuple:
        if axis == AXIS_BACKENDS:
            return self.backends
        if axis == AXIS_LINK_TARGETS:
            return self.link_targets
        if axis == AXIS_OP_NAMES:
            return self.op_names
        if axis == AXIS_KERNEL_SYMBOLS:
            return self.symbols
        if axis == AXIS_DISPATCH_PREDICATES:
            return self.dispatch_predicates
        raise SurfaceInputError(f"unknown axis {axis!r}")

    def to_dict(self) -> dict:
        return {
            "manifest_kind": "derived",
            "candidate_id": self.candidate_id,
            "backends": list(self.backends),
            "link_targets": list(self.link_targets),
            "objects": list(self.objects),
            "touched_files": list(self.touched_files),
            "symbols": list(self.symbols),
            "op_registrations": [r.to_dict() for r in
                                 sorted(self.op_registrations,
                                        key=lambda r: (r.op_name, r.backend,
                                                       r.dispatch_predicate or ""))],
            "dispatch_predicates": list(self.dispatch_predicates),
            "over_approximations": [o.to_dict() for o in self.over_approximations],
            "axes_derived": list(self.axes_derived),
            "coverage": {"outcome": self.coverage.outcome, "reasons": list(self.coverage.reasons)},
            "full_tree": self.full_tree,
            "inputs": dict(self.inputs),
        }

    def sha256(self) -> str:
        return schemas.content_hash(self.to_dict())


def derive_affected_surface(*,
                            candidate_id: str,
                            diff: SourceDiff,
                            indexes: Sequence[BuildDependencyIndex],
                            registrations: Optional[SymbolRegistrationIndex] = None,
                            change_class: Optional[str] = None) -> AffectedSurface:
    """§6.4 stage 1 — STATIC DERIVATION from the diff.

    diff -> touched files -> objects -> link targets -> backends, and in parallel
    objects -> their whole source closure -> symbols -> op registrations -> dispatch
    predicates (which may add further backends), entirely through the build system's
    own generated dependency information.

    `indexes` is a sequence so that BOTH the candidate's and the production base's
    build trees can be consulted. A deleted file no longer appears in the candidate's
    depfiles; consulting only the candidate would classify every deletion as an
    unmapped file and widen to the whole tree on every delete. Consulting the base too
    keeps the derivation sharp while remaining a superset.

    Widening is fail-closed and named:
      * a touched file in no index at all -> every known backend and link target;
      * an object in no link manifest -> same;
      * a link target no backend claims -> same;
      * a stale dependency entry anywhere in an index -> same;
      * `change_class: core_header` -> full tree, per §8.5.1, regardless of diff size.

    Raises on wiring defects (no indexes, an actor-provenance input, a change class the
    schema does not know). Never returns a narrower surface because an input was
    missing.
    """
    _require_str(candidate_id, "candidate_id")
    if not isinstance(diff, SourceDiff):
        raise SurfaceInputError(f"diff must be a SourceDiff, got {type(diff).__name__}")
    indexes = tuple(indexes)
    if not indexes:
        raise SurfaceInputError(
            "derive_affected_surface needs at least one BuildDependencyIndex; with none "
            "there is no closure and an empty surface would read as 'nothing affected'")
    for index in indexes:
        if not isinstance(index, BuildDependencyIndex):
            raise UntrustedProvenance(
                f"indexes must be BuildDependencyIndex, got {type(index).__name__}; the "
                "closure may only come from the build system (§6.4)")
    if registrations is not None and not isinstance(registrations, SymbolRegistrationIndex):
        raise SurfaceInputError(
            f"registrations must be a SymbolRegistrationIndex, got "
            f"{type(registrations).__name__}")
    if change_class is not None and change_class not in schemas.CHANGE_CLASSES:
        raise SurfaceInputError(
            f"change_class {change_class!r} is not one of {sorted(schemas.CHANGE_CLASSES)}")

    all_backends = _sorted_unique(b for idx in indexes for b in idx.known_backends)
    all_link_targets = _sorted_unique(t for idx in indexes for t in idx.known_link_targets)
    all_objects = _sorted_unique(o for idx in indexes
                                 for t in idx.known_link_targets
                                 for o in idx.objects_for_link_target(t))

    widenings: list = []
    full_tree = False

    def _widen(reason: str, trigger: str) -> None:
        nonlocal full_tree
        full_tree = True
        widenings.append(OverApproximation(reason=reason, kind=OA_KIND_FAIL_CLOSED,
                                           trigger=trigger, widened_to=all_backends))

    # --- stale build-system evidence anywhere in an index --------------------
    for index in indexes:
        for stale in index.stale_targets:
            _widen(OA_STALE_DEPENDENCY_ENTRY, f"{index.label}:{stale}")
        for unresolved in index.unresolved_link_inputs:
            _widen(OA_UNLINKED_OBJECT,
                   f"{index.label}:unresolved link input {unresolved}")

    # --- touched files -> objects -------------------------------------------
    touched = diff.touched_paths
    objects: set = set()
    for path in touched:
        found: set = set()
        for index in indexes:
            found.update(index.objects_for_source(path))
        if not found:
            _widen(OA_UNMAPPED_TOUCHED_FILE,
                   f"{path} appears in no depfile in {[i.label for i in indexes]}")
            continue
        objects.update(found)

    # --- objects -> link targets ---------------------------------------------
    link_targets: set = set()
    for obj in sorted(objects):
        found = set()
        for index in indexes:
            found.update(index.link_targets_for_object(obj))
        if not found:
            _widen(OA_UNLINKED_OBJECT, f"object {obj} appears in no link manifest")
            continue
        link_targets.update(found)

    # --- link targets -> backends --------------------------------------------
    backends: set = set()
    for target in sorted(link_targets):
        found = set()
        for index in indexes:
            found.update(index.backends_for_link_target(target))
        if not found:
            _widen(OA_UNATTRIBUTED_LINK_TARGET, f"link target {target} is claimed by no backend")
            continue
        backends.update(found)

    # --- §8.5.1 core-header risk tier ----------------------------------------
    if change_class in FULL_TREE_CHANGE_CLASSES:
        full_tree = True
        widenings.append(OverApproximation(
            reason=OA_CORE_HEADER_CHANGE_CLASS, kind=OA_KIND_FAIL_CLOSED,
            trigger=f"change_class={change_class}: a change to shared ggml core or a "
                    "widely-included header forces full-tree affected surface regardless of "
                    "the textual diff size (§8.5.1)",
            widened_to=all_backends))

    # --- mechanical fan-out, recorded but not a defect ------------------------
    for path in touched:
        reach: set = set()
        for index in indexes:
            for obj in index.objects_for_source(path):
                reach.update(index.link_targets_for_object(obj))
        if len(reach) > 1:
            widenings.append(OverApproximation(
                reason=OA_SHARED_HEADER_FANOUT, kind=OA_KIND_MECHANICAL,
                trigger=path, widened_to=tuple(sorted(reach))))

    if full_tree:
        backends = set(all_backends)
        link_targets = set(all_link_targets)
        objects = set(all_objects)

    # --- symbols / op registrations / dispatch predicates ---------------------
    axes = [AXIS_BACKENDS, AXIS_LINK_TARGETS]
    symbols: set = set()
    regs: list = []
    predicates: set = set()
    # An index that names NO symbol at all is an extractor that produced nothing, not
    # a fact about the candidate. Treating it as a derivation would mark the three
    # symbol axes derived with an empty derived set, and reconciliation would then
    # report every traced op as a hard candidate FAIL — filing a gap in the instrument
    # as a finding about the candidate, which §6.4 reconciliation must never do.
    usable_registrations = registrations
    if registrations is not None and not registrations.all_symbols:
        usable_registrations = None
    if usable_registrations is None:
        widenings.append(OverApproximation(
            reason=OA_NO_SYMBOL_INDEX, kind=OA_KIND_FAIL_CLOSED,
            trigger=("no SymbolRegistrationIndex was supplied" if registrations is None
                     else f"SymbolRegistrationIndex {registrations.label!r} names no symbol "
                          "at all; an index that derived nothing is a coverage gap, not "
                          "evidence that the candidate registers no op"),
            widened_to=(AXIS_OP_NAMES, AXIS_KERNEL_SYMBOLS, AXIS_DISPATCH_PREDICATES)))
    else:
        registrations = usable_registrations
        axes.extend([AXIS_OP_NAMES, AXIS_KERNEL_SYMBOLS, AXIS_DISPATCH_PREDICATES])
        if full_tree:
            symbols.update(registrations.all_symbols)
            regs.extend(registrations.all_registrations)
        else:
            # Symbols come from the CLOSURE, not from the touched files alone. A header
            # defines no symbols, but every object that includes it may define one whose
            # behaviour the header change alters — so the symbol set is that of every
            # source compiled into every affected object. Keying on touched files only
            # would give a header change an EMPTY op set, and an empty derived op set
            # turns every traced op into a false escape.
            closure_sources: set = set(touched)
            for obj in sorted(objects):
                for index in indexes:
                    closure_sources.update(index.sources_for_object(obj))
            for path in sorted(closure_sources):
                symbols.update(registrations.symbols_for(path))
            for symbol in sorted(symbols):
                regs.extend(registrations.registrations_for(symbol))
        for reg in regs:
            if reg.dispatch_predicate:
                predicates.add(reg.dispatch_predicate)
            # An op registration names its own backend; a registration reaching a
            # backend the link closure did not is still an affected backend.
            backends.add(reg.backend)

    # --- coverage -------------------------------------------------------------
    coverage_reasons: list = [
        f"{o.reason}: {o.trigger}" for o in widenings if o.kind == OA_KIND_FAIL_CLOSED
    ]
    index_coverage = _combine_checks(*[i.coverage_check() for i in indexes])
    if coverage_reasons or index_coverage.outcome != schemas.PASS:
        coverage = schemas.Check(
            schemas.COULD_NOT_CHECK,
            tuple(coverage_reasons) + tuple(index_coverage.reasons))
    else:
        coverage = schemas.Check(schemas.PASS)

    inputs = {
        "diff": diff.to_dict(),
        "indexes": [i.to_dict() for i in indexes],
        "registrations": (None if registrations is None
                          else {"label": registrations.label,
                                "provenance": registrations.provenance,
                                "symbol_count": len(registrations.all_symbols)}),
        "change_class": change_class,
        "protocol_id": api.PROTOCOL_VERSIONED_ID,
    }

    return AffectedSurface(
        candidate_id=candidate_id,
        backends=_sorted_unique(backends),
        link_targets=_sorted_unique(link_targets),
        objects=_sorted_unique(objects),
        touched_files=touched,
        symbols=_sorted_unique(symbols),
        op_registrations=tuple(regs),
        dispatch_predicates=_sorted_unique(predicates),
        over_approximations=tuple(widenings),
        axes_derived=tuple(axes),
        coverage=coverage,
        full_tree=full_tree,
        inputs=inputs,
    )


# =============================================================================
# STAGE 2 — the dispatch trace
# =============================================================================

@dataclass(frozen=True)
class DispatchEvent:
    """One kernel dispatch the instrumented T0 run actually executed."""

    op_name: str
    backend: str
    kernel_symbol: str
    link_target: Optional[str] = None
    dispatch_predicate: Optional[str] = None
    fallback: bool = False
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("op_name", "kernel_symbol"):
            _require_str(getattr(self, name), f"DispatchEvent.{name}")
        if self.backend not in schemas.BACKENDS:
            raise TraceParseError(
                f"DispatchEvent.backend {self.backend!r} is not one of "
                f"{sorted(schemas.BACKENDS)}; an unmapped backend is never defaulted")
        if not isinstance(self.fallback, bool):
            raise TraceParseError("DispatchEvent.fallback must be a bool")
        if self.fallback and not self.fallback_reason:
            raise TraceParseError(
                "a fallback dispatch must name its reason; §8.6 requires a no-fallback "
                "PROOF, and an unexplained fallback is not proof of anything")

    def to_dict(self) -> dict:
        return {"op_name": self.op_name, "backend": self.backend,
                "kernel_symbol": self.kernel_symbol, "link_target": self.link_target,
                "dispatch_predicate": self.dispatch_predicate, "fallback": self.fallback,
                "fallback_reason": self.fallback_reason}


@dataclass(frozen=True)
class TracedSurface:
    """What the dispatch trace observed — stage 2's output.

    `completeness` is COULD_NOT_CHECK for an empty trace and for a trace the producer
    marked truncated. Both are indistinguishable from a candidate whose escaping
    dispatch simply was not recorded, so neither may reconcile to PASS.
    """

    candidate_id: str
    trace_ref: str
    events: tuple
    truncated: bool
    completeness: schemas.Check
    no_fallback: schemas.Check
    header: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_str(self.candidate_id, "TracedSurface.candidate_id")
        _require_str(self.trace_ref, "TracedSurface.trace_ref")
        if not isinstance(self.events, tuple):
            raise TraceParseError("TracedSurface.events must be a tuple")
        for event in self.events:
            if not isinstance(event, DispatchEvent):
                raise TraceParseError(
                    f"TracedSurface.events must hold DispatchEvent, got "
                    f"{type(event).__name__}")
        for name in ("completeness", "no_fallback"):
            if not isinstance(getattr(self, name), schemas.Check):
                raise TraceParseError(f"TracedSurface.{name} must be a schemas.Check")
        if not isinstance(self.truncated, bool):
            raise TraceParseError("TracedSurface.truncated must be a bool")
        # The invariant is carried by the TYPE, not only by the factory below.
        # `TracedSurface` is public API, so without this a caller could hand a
        # zero-event trace stamped `completeness=PASS` to `reconcile_surface()` and
        # obtain `reconciled: true` — the exact "absence is containment" answer this
        # module exists to refuse.
        if self.completeness.outcome == schemas.PASS:
            if not self.events:
                raise TraceParseError(
                    "TracedSurface.completeness is PASS with zero events; an empty trace "
                    "is indistinguishable from one that missed the escaping dispatch and "
                    "can never confirm `derived ⊇ traced`")
            if self.truncated:
                raise TraceParseError(
                    "TracedSurface.completeness is PASS on a trace marked truncated; a "
                    "truncated trace under-reports executed kernels")
        if self.no_fallback.outcome == schemas.PASS:
            if any(e.fallback for e in self.events):
                raise TraceParseError(
                    "TracedSurface.no_fallback is PASS while the trace contains a fallback "
                    "dispatch; §8.6 requires a no-fallback PROOF")
            if not self.events:
                raise TraceParseError(
                    "TracedSurface.no_fallback is PASS with zero observations; §8.6 "
                    "requires a PROOF, and zero observations are not one")

    @property
    def backends(self) -> tuple:
        return _sorted_unique(e.backend for e in self.events)

    @property
    def link_targets(self) -> tuple:
        return _sorted_unique(e.link_target for e in self.events if e.link_target)

    @property
    def op_names(self) -> tuple:
        return _sorted_unique(e.op_name for e in self.events)

    @property
    def kernel_symbols(self) -> tuple:
        return _sorted_unique(e.kernel_symbol for e in self.events)

    @property
    def dispatch_predicates(self) -> tuple:
        return _sorted_unique(e.dispatch_predicate for e in self.events if e.dispatch_predicate)

    @property
    def fallback_events(self) -> tuple:
        return tuple(e for e in self.events if e.fallback)

    def axis_values(self, axis: str) -> tuple:
        if axis == AXIS_BACKENDS:
            return self.backends
        if axis == AXIS_LINK_TARGETS:
            return self.link_targets
        if axis == AXIS_OP_NAMES:
            return self.op_names
        if axis == AXIS_KERNEL_SYMBOLS:
            return self.kernel_symbols
        if axis == AXIS_DISPATCH_PREDICATES:
            return self.dispatch_predicates
        raise SurfaceInputError(f"unknown axis {axis!r}")

    def to_dict(self) -> dict:
        return {
            "manifest_kind": "traced",
            "candidate_id": self.candidate_id,
            "trace_ref": self.trace_ref,
            "event_count": len(self.events),
            "truncated": self.truncated,
            "backends": list(self.backends),
            "link_targets": list(self.link_targets),
            "op_names": list(self.op_names),
            "kernel_symbols": list(self.kernel_symbols),
            "dispatch_predicates": list(self.dispatch_predicates),
            "fallback_events": [e.to_dict() for e in self.fallback_events],
            "completeness": {"outcome": self.completeness.outcome,
                             "reasons": list(self.completeness.reasons)},
            "no_fallback": {"outcome": self.no_fallback.outcome,
                            "reasons": list(self.no_fallback.reasons)},
            "header": dict(self.header),
        }

    def sha256(self) -> str:
        return schemas.content_hash(self.to_dict())


_TRACE_REQUIRED = ("op_name", "backend", "kernel_symbol")
_TRACE_OPTIONAL = ("link_target", "dispatch_predicate", "fallback", "fallback_reason")


def _event_from_mapping(obj: Mapping[str, Any], where: str) -> DispatchEvent:
    unknown = sorted(set(obj) - set(_TRACE_REQUIRED) - set(_TRACE_OPTIONAL))
    if unknown:
        raise TraceParseError(
            f"{where}: unknown dispatch-trace fields {unknown}; an unrecognised field may be "
            "the one carrying the escape, so it is refused rather than ignored")
    missing = [k for k in _TRACE_REQUIRED if k not in obj]
    if missing:
        raise TraceParseError(f"{where}: missing required trace fields {missing}")
    # `bool(...)` would coerce, and the coercion is not symmetric: `"false"` and
    # `"no"` become True while `[]`, `{}` and `null` become False. This parser is
    # documented as having no tolerant mode — a line it cannot read is a refusal,
    # not a guess, because guessing `fallback=False` erases a fallback dispatch.
    fallback = obj.get("fallback", False)
    if not isinstance(fallback, bool):
        raise TraceParseError(
            f"{where}: 'fallback' must be a JSON boolean, got "
            f"{type(fallback).__name__} {fallback!r}; a coerced fallback flag is how a "
            "fallback dispatch would be read as a clean one")
    for key in ("link_target", "dispatch_predicate", "fallback_reason"):
        value = obj.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise TraceParseError(
                f"{where}: {key!r} must be a non-empty string or absent, got "
                f"{type(value).__name__} {value!r}")
    return DispatchEvent(
        op_name=obj["op_name"], backend=obj["backend"], kernel_symbol=obj["kernel_symbol"],
        link_target=obj.get("link_target"), dispatch_predicate=obj.get("dispatch_predicate"),
        fallback=fallback, fallback_reason=obj.get("fallback_reason"),
    )


def traced_surface_from_events(*, candidate_id: str, trace_ref: str,
                               events: Sequence[DispatchEvent],
                               truncated: bool = False,
                               header: Optional[Mapping[str, Any]] = None) -> TracedSurface:
    """Assemble a `TracedSurface` from already-parsed events."""
    events = tuple(events)
    for event in events:
        if not isinstance(event, DispatchEvent):
            raise TraceParseError(f"events must be DispatchEvent, got {type(event).__name__}")

    if not events:
        completeness = schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("the dispatch trace recorded zero kernel executions; an empty trace is "
             "indistinguishable from a trace that missed the escaping dispatch, so it "
             "cannot confirm `derived ⊇ traced`",))
    elif truncated:
        completeness = schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"the dispatch trace is marked truncated after {len(events)} events; a "
             "truncated trace under-reports executed kernels and cannot confirm "
             "containment",))
    else:
        completeness = schemas.Check(schemas.PASS)

    fallbacks = tuple(e for e in events if e.fallback)
    if fallbacks:
        no_fallback = schemas.Check(
            schemas.FAIL,
            tuple(f"{e.op_name} on {e.backend} fell back to {e.kernel_symbol}: "
                  f"{e.fallback_reason}" for e in fallbacks))
    elif not events:
        no_fallback = schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("no dispatch was observed, so no-fallback cannot be proved; §8.6 requires a "
             "no-fallback PROOF, and zero observations are not one",))
    else:
        no_fallback = schemas.Check(schemas.PASS)

    return TracedSurface(candidate_id=candidate_id, trace_ref=trace_ref, events=events,
                         truncated=truncated, completeness=completeness,
                         no_fallback=no_fallback, header=dict(header or {}))


def parse_dispatch_trace(text: str, *, trace_ref: str,
                         candidate_id: Optional[str] = None) -> TracedSurface:
    """§6.4 stage 2 — parse the T0 dispatch trace (JSONL).

    The first line MAY be a header object carrying
    `{"schema": DISPATCH_TRACE_SCHEMA, "candidate_id": ..., "truncated": bool}`; every
    other line is one `DispatchEvent`.

    Every malformed line RAISES. A skipped line is an unobserved kernel execution, and
    an unobserved execution is exactly what a candidate escaping its declared surface
    produces — so this parser has no tolerant mode.
    """
    _require_str(trace_ref, "trace_ref")
    if not isinstance(text, str):
        raise TraceParseError(f"trace text must be a string, got {type(text).__name__}")

    header: dict = {}
    events: list = []
    truncated = False
    for lineno, raw in enumerate(text.split("\n"), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            raise TraceParseError(f"{trace_ref}:{lineno}: not JSON: {exc}") from exc
        if not isinstance(obj, Mapping):
            raise TraceParseError(f"{trace_ref}:{lineno}: expected a JSON object")
        if obj.get("schema") is not None:
            if header:
                raise TraceParseError(f"{trace_ref}:{lineno}: a second header line")
            if obj["schema"] != DISPATCH_TRACE_SCHEMA:
                raise TraceParseError(
                    f"{trace_ref}:{lineno}: unknown trace schema {obj['schema']!r}; expected "
                    f"{DISPATCH_TRACE_SCHEMA!r}")
            header = dict(obj)
            truncated = bool(obj.get("truncated", False))
            continue
        events.append(_event_from_mapping(obj, f"{trace_ref}:{lineno}"))

    resolved_id = candidate_id or header.get("candidate_id")
    if not resolved_id:
        raise TraceParseError(
            f"{trace_ref}: the trace names no candidate_id and none was supplied; an "
            "unattributed trace could confirm any candidate's surface")
    if candidate_id and header.get("candidate_id") and header["candidate_id"] != candidate_id:
        raise TraceParseError(
            f"{trace_ref}: trace header candidate_id {header['candidate_id']!r} disagrees "
            f"with the requested {candidate_id!r}")
    return traced_surface_from_events(candidate_id=resolved_id, trace_ref=trace_ref,
                                      events=events, truncated=truncated, header=header)


# =============================================================================
# STAGE 3 — reconciliation
# =============================================================================

@dataclass(frozen=True)
class AxisReconciliation:
    """Containment on one axis. `escaped` is `traced \\ derived` — the hard failure."""

    axis: str
    derived: tuple
    traced: tuple
    escaped: tuple
    gating: bool
    check: schemas.Check

    def to_dict(self) -> dict:
        return {"axis": self.axis, "derived": list(self.derived), "traced": list(self.traced),
                "escaped": list(self.escaped), "gating": self.gating,
                "check": {"outcome": self.check.outcome, "reasons": list(self.check.reasons)}}


@dataclass(frozen=True)
class SurfaceReconciliation:
    """§6.4 stage 3 — `derived ⊇ traced`, per axis, with the escapes named.

    Constructed only by `reconcile_surface()`, whose signature admits the derived
    manifest and the trace and NOTHING ELSE. That is the structural form of *"the
    actor's declaration is retained as a scored prediction and fed to the critic, never
    used as a scope input"*: there is no parameter for it.
    """

    derived: AffectedSurface
    traced: Optional[TracedSurface]
    axes: tuple
    check: schemas.Check
    hard_failure: bool

    @property
    def escaped(self) -> tuple:
        out: list = []
        for axis in self.axes:
            for value in axis.escaped:
                out.append((axis.axis, value))
        return tuple(out)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.derived.candidate_id,
            "derived_sha256": self.derived.sha256(),
            "traced_sha256": None if self.traced is None else self.traced.sha256(),
            "axes": [a.to_dict() for a in self.axes],
            "check": {"outcome": self.check.outcome, "reasons": list(self.check.reasons)},
            "hard_failure": self.hard_failure,
            "invariant": "18: declared equals traced",
        }

    def gate_results(self) -> tuple:
        """The §6.4 / §8.6 gates, as `api.GateResult`s a `TierGateRunner` returns.

        All four are `integrity` — one of the five lexicographically-prior gate classes
        — so a surface escape ends speed ranking for the candidate rather than
        penalising it ("Correctness precedence").
        """
        gates = [
            api.GateResult(
                gate_id="surface.derived_coverage",
                gate_class=api.GATE_INTEGRITY,
                check=self.derived.coverage,
                notes=(f"{len(self.derived.fail_closed_widenings)} fail-closed widenings; "
                       f"full_tree={self.derived.full_tree}",),
                evidence_ref=self.derived.sha256(),
            ),
            api.GateResult(
                gate_id="surface.reconciliation",
                gate_class=api.GATE_INTEGRITY,
                check=self.check,
                notes=("invariant 18: derived ⊇ traced; traced ⊄ derived is a hard candidate "
                       "failure",),
                evidence_ref=self.derived.sha256(),
            ),
        ]
        if self.traced is not None:
            gates.append(api.GateResult(
                gate_id="surface.trace_completeness",
                gate_class=api.GATE_INTEGRITY,
                check=self.traced.completeness,
                evidence_ref=self.traced.trace_ref,
            ))
            gates.append(api.GateResult(
                gate_id="surface.no_fallback",
                gate_class=api.GATE_INTEGRITY,
                check=self.traced.no_fallback,
                notes=("§8.6: dispatch trace for affected-surface confirmation and "
                       "no-fallback proof",),
                evidence_ref=self.traced.trace_ref,
            ))
        else:
            gates.append(api.GateResult(
                gate_id="surface.trace_completeness",
                gate_class=api.GATE_INTEGRITY,
                check=schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    ("no dispatch trace was supplied; the derived surface is unconfirmed "
                     "(§6.4 stage 2)",)),
            ))
            gates.append(api.GateResult(
                gate_id="surface.no_fallback",
                gate_class=api.GATE_INTEGRITY,
                check=schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    ("no dispatch trace was supplied, so no-fallback cannot be proved",)),
            ))
        return tuple(gates)


def reconcile_surface(derived: AffectedSurface,
                      traced: Optional[TracedSurface]) -> SurfaceReconciliation:
    """§6.4 stage 3 — RECONCILIATION. Two arguments, and neither is a declaration.

    Per axis:
      * `escaped = traced \\ derived`. Any escape on a GATING axis (`backends`,
        `link_targets`) is a **hard candidate failure** — FAIL.
      * An escape on a non-gating axis is FAIL only when that axis was actually
        derived; when it was not (no symbol index), it is COULD_NOT_CHECK naming the
        derivation gap, because a gap in the instrument is not a finding about the
        candidate.
      * An unusable trace (absent, empty, truncated) is COULD_NOT_CHECK on every axis.
        Absence of a comparison is never evidence of containment.

    The overall check is worst-of, so it is PASS only when the trace was usable AND
    every axis contained.
    """
    if not isinstance(derived, AffectedSurface):
        raise SurfaceInputError(f"derived must be an AffectedSurface, got {type(derived).__name__}")
    if traced is not None and not isinstance(traced, TracedSurface):
        raise SurfaceInputError(f"traced must be a TracedSurface or None, got "
                                f"{type(traced).__name__}")
    if traced is not None and traced.candidate_id != derived.candidate_id:
        raise SurfaceInputError(
            f"trace is for candidate {traced.candidate_id!r} but the derived manifest is for "
            f"{derived.candidate_id!r}; reconciling across candidates would confirm the wrong "
            "surface")

    axes: list = []
    for axis in SURFACE_AXES:
        gating = axis in GATING_AXES
        derived_values = derived.axis_values(axis)
        if traced is None:
            axes.append(AxisReconciliation(
                axis=axis, derived=derived_values, traced=(), escaped=(), gating=gating,
                check=schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    ("no dispatch trace: `derived ⊇ traced` is unconfirmed on this axis",))))
            continue
        traced_values = traced.axis_values(axis)
        escaped = tuple(v for v in traced_values if v not in set(derived_values))
        derived_this_axis = axis in derived.axes_derived
        if escaped:
            if gating or derived_this_axis:
                check = schemas.Check(
                    schemas.FAIL,
                    (f"traced ⊄ derived on {axis}: {list(escaped)} executed but are outside "
                     "the derived affected surface. Invariant 18 (declared equals traced): "
                     "this is a hard candidate failure.",))
            else:
                check = schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    (f"{axis} was not derived (no symbol/registration index), so {list(escaped)} "
                     "cannot be attributed to the candidate; this is a derivation gap, not a "
                     "finding about the candidate",))
        elif traced.completeness.outcome != schemas.PASS:
            check = schemas.Check(schemas.COULD_NOT_CHECK, traced.completeness.reasons)
        elif not derived_this_axis and traced_values:
            check = schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"{axis} was not derived, so containment on it is unproven",))
        elif gating and derived_values and not traced_values:
            # `link_target` is an OPTIONAL trace field. A trace that simply omits it
            # produces an empty traced set on a GATING axis, and comparing a
            # non-empty derived set against nothing is not containment — it is the
            # same "absence is not evidence" the no-trace branch above refuses.
            # Without this, dropping one field from every event flips a real
            # link-target escape from FAIL to PASS.
            check = schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"the trace recorded {len(traced.events)} dispatches but no {axis} at "
                 f"all, while {len(derived_values)} were derived; containment on a "
                 "gating axis cannot be confirmed against an empty observation "
                 "(invariant 18)",))
        else:
            check = schemas.Check(schemas.PASS)
        axes.append(AxisReconciliation(axis=axis, derived=derived_values, traced=traced_values,
                                       escaped=escaped, gating=gating, check=check))

    overall = _combine_checks(*[a.check for a in axes])
    hard_failure = any(a.check.outcome == schemas.FAIL for a in axes)
    return SurfaceReconciliation(derived=derived, traced=traced, axes=tuple(axes),
                                 check=overall, hard_failure=hard_failure)


def candidate_affected_surface_block(reconciliation: SurfaceReconciliation) -> dict:
    """The `affected_surface` block of an `epyc.autokernel.candidate.v1` record (§7.3).

    `reconciled` is True only on a PASS: a COULD_NOT_CHECK reconciliation is not a
    reconciled one, and `schemas.validate_candidate` already refuses `reconciled: true`
    with a null `traced_sha256`.
    """
    if not isinstance(reconciliation, SurfaceReconciliation):
        raise SurfaceInputError("expected a SurfaceReconciliation")
    traced = reconciliation.traced
    return {
        "derived_sha256": reconciliation.derived.sha256(),
        "traced_sha256": None if traced is None else traced.sha256(),
        "reconciled": reconciliation.check.outcome == schemas.PASS,
    }


# =============================================================================
# The actor's declaration — a SCORED PREDICTION, never a scope input
# =============================================================================

@dataclass(frozen=True)
class ActorDeclaration:
    """What the actor said it touched. Deliberately a different type from
    `AffectedSurface`, so it cannot be passed where a surface is expected."""

    candidate_id: str
    backends: tuple
    link_targets: tuple = ()
    op_names: tuple = ()
    rationale: Optional[str] = None
    provenance: str = PROVENANCE_ACTOR_DECLARED

    def __post_init__(self) -> None:
        _require_str(self.candidate_id, "ActorDeclaration.candidate_id")
        if self.provenance != PROVENANCE_ACTOR_DECLARED:
            raise UntrustedProvenance(
                "an ActorDeclaration is actor-declared by definition; relabelling its "
                "provenance is how it would become a scope input")
        for backend in self.backends:
            if backend not in schemas.BACKENDS:
                raise SurfaceInputError(
                    f"declared backend {backend!r} is not one of {sorted(schemas.BACKENDS)}")

    def axis_values(self, axis: str) -> tuple:
        if axis == AXIS_BACKENDS:
            return _sorted_unique(self.backends)
        if axis == AXIS_LINK_TARGETS:
            return _sorted_unique(self.link_targets)
        if axis == AXIS_OP_NAMES:
            return _sorted_unique(self.op_names)
        return ()


@dataclass(frozen=True)
class DeclarationScore:
    """How well the actor predicted the derived surface. Fed to the critic (§6.3).

    `under_declared` is the interesting half: it is exactly the set the actor would
    have removed from its own release scope. `is_scope_input` is False in the payload
    and there is no code path that reads this object back into a derivation.
    """

    candidate_id: str
    per_axis: tuple  # ((axis, {"declared","derived","missing","extra","precision","recall"}), ...)
    under_declared: tuple
    over_declared: tuple

    @property
    def under_declared_any(self) -> bool:
        return bool(self.under_declared)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "is_scope_input": False,
            "consumer": "critic",
            "note": ("invariant 18: the actor's declaration is retained as a scored "
                     "prediction and fed to the critic, never used as a scope input"),
            "per_axis": {axis: dict(stats) for axis, stats in self.per_axis},
            "under_declared": [list(pair) for pair in self.under_declared],
            "over_declared": [list(pair) for pair in self.over_declared],
        }


def score_actor_declaration(declaration: ActorDeclaration,
                            derived: AffectedSurface) -> DeclarationScore:
    """Score the actor's declaration against the derived manifest.

    Returns a critic payload. It changes no scope, gates nothing, and is not consumed
    by `reconcile_surface()`.
    """
    if not isinstance(declaration, ActorDeclaration):
        raise SurfaceInputError("declaration must be an ActorDeclaration")
    if not isinstance(derived, AffectedSurface):
        raise SurfaceInputError("derived must be an AffectedSurface")
    if declaration.candidate_id != derived.candidate_id:
        raise SurfaceInputError(
            f"declaration is for {declaration.candidate_id!r} but the derived manifest is for "
            f"{derived.candidate_id!r}")

    per_axis: list = []
    under: list = []
    over: list = []
    for axis in (AXIS_BACKENDS, AXIS_LINK_TARGETS, AXIS_OP_NAMES):
        declared = set(declaration.axis_values(axis))
        actual = set(derived.axis_values(axis))
        missing = tuple(sorted(actual - declared))
        extra = tuple(sorted(declared - actual))
        precision = (len(declared & actual) / len(declared)) if declared else None
        recall = (len(declared & actual) / len(actual)) if actual else None
        per_axis.append((axis, {
            "declared": list(sorted(declared)),
            "derived": list(sorted(actual)),
            "missing": list(missing),
            "extra": list(extra),
            "precision": precision,
            "recall": recall,
        }))
        under.extend((axis, v) for v in missing)
        over.extend((axis, v) for v in extra)
    return DeclarationScore(candidate_id=derived.candidate_id, per_axis=tuple(per_axis),
                            under_declared=tuple(under), over_declared=tuple(over))


# =============================================================================
# §3.2 — the two-stage backend-unchanged test
# =============================================================================

@dataclass(frozen=True)
class ToolchainIdentity:
    """Compiler, linker, flags and build environment — stage 1's other half.

    §3.2: *"Unchanged iff the diff is empty AND toolchain, flags, and build environment
    are identical."* Flags and environment are ordered tuples, not sets: `-O2 -O3` and
    `-O3 -O2` are different builds.
    """

    compiler_id: str
    compiler_version: str
    linker_id: str
    linker_version: str
    flags: tuple
    defines: tuple
    environment: tuple  # ((name, value), ...)
    sysroot: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("compiler_id", "compiler_version", "linker_id", "linker_version"):
            _require_str(getattr(self, name), f"ToolchainIdentity.{name}")
        for name in ("flags", "defines", "environment"):
            if not isinstance(getattr(self, name), tuple):
                raise SurfaceInputError(f"ToolchainIdentity.{name} must be a tuple")
        for pair in self.environment:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise SurfaceInputError(
                    "ToolchainIdentity.environment must be ((name, value), ...)")

    def to_dict(self) -> dict:
        return {
            "compiler_id": self.compiler_id, "compiler_version": self.compiler_version,
            "linker_id": self.linker_id, "linker_version": self.linker_version,
            "flags": list(self.flags), "defines": list(self.defines),
            "environment": [[k, v] for k, v in self.environment], "sysroot": self.sysroot,
        }

    def digest(self) -> str:
        return schemas.content_hash(self.to_dict())

    def differences(self, other: "ToolchainIdentity") -> tuple:
        if not isinstance(other, ToolchainIdentity):
            raise SurfaceInputError("differences() needs another ToolchainIdentity")
        out: list = []
        mine, theirs = self.to_dict(), other.to_dict()
        for key in sorted(mine):
            if mine[key] != theirs[key]:
                out.append(f"{key}: {mine[key]!r} != {theirs[key]!r}")
        return tuple(out)


# --- ELF section reading (read-only; no build, no subprocess) --------------------

_ELF_MAGIC = b"\x7fELF"
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_SHT_NOBITS = 8
_SHN_XINDEX = 0xFFFF
_ELF64_SYM_SIZE = 24


def _parse_elf_sections(data: bytes, ref: str) -> dict:
    """Return `{section_name: (offset, size, sh_type, sh_link, sh_entsize)}`.

    ELF64 little-endian only. Anything else RAISES: guessing at a header layout would
    produce a digest over the wrong bytes, which is worse than no digest.
    """
    if len(data) < 64 or data[:4] != _ELF_MAGIC:
        raise ElfFormatError(f"{ref}: not an ELF file (bad magic)")
    if data[4] != _ELFCLASS64:
        raise ElfFormatError(f"{ref}: not ELF64 (EI_CLASS={data[4]}); this reader does not "
                             "guess at other classes")
    if data[5] != _ELFDATA2LSB:
        raise ElfFormatError(f"{ref}: not little-endian (EI_DATA={data[5]})")
    # Elf64_Ehdr: e_shoff @40 (Q), then 10 bytes of e_flags/e_ehsize/e_phentsize/e_phnum,
    # then e_shentsize @58, e_shnum @60, e_shstrndx @62.
    (e_shoff, e_shentsize, e_shnum, e_shstrndx) = struct.unpack_from("<Q10xHHH", data, 40)
    if e_shoff == 0:
        raise ElfFormatError(f"{ref}: no section header table")
    if e_shentsize != 64:
        raise ElfFormatError(f"{ref}: unexpected section header size {e_shentsize}")

    def _shdr(i: int) -> tuple:
        off = e_shoff + i * e_shentsize
        if off + e_shentsize > len(data):
            raise ElfFormatError(f"{ref}: section header {i} is past end of file")
        return struct.unpack_from("<IIQQQQIIQQ", data, off)

    first = _shdr(0)
    if e_shnum == 0:
        e_shnum = first[5]  # sh_size of section 0 holds the real count
    if e_shstrndx == _SHN_XINDEX:
        e_shstrndx = first[6]  # sh_link of section 0 holds the real index
    if e_shstrndx >= e_shnum:
        raise ElfFormatError(f"{ref}: shstrndx {e_shstrndx} out of range")

    str_hdr = _shdr(e_shstrndx)
    str_off, str_size = str_hdr[4], str_hdr[5]
    strtab = data[str_off:str_off + str_size]

    sections: dict = {}
    for i in range(e_shnum):
        (sh_name, sh_type, _flags, _addr, sh_offset, sh_size, sh_link, _info,
         _align, sh_entsize) = _shdr(i)
        end = strtab.find(b"\x00", sh_name)
        if end < 0:
            raise ElfFormatError(f"{ref}: unterminated section name at {sh_name}")
        name = strtab[sh_name:end].decode("utf-8", errors="strict")
        sections[name] = (sh_offset, sh_size, sh_type, sh_link, sh_entsize)
    return sections


def _dynsym_canonical(data: bytes, sections: dict, ref: str) -> Optional[list]:
    """Canonical dynamic symbol table: sorted `[name, bind, type, defined]`.

    Raw `.dynsym` bytes are NOT hashed: they carry `st_value` addresses, which move for
    reasons that have nothing to do with the symbol set. §3.2 asks for "the dynamic
    symbol table", and the table's content is its symbols.
    """
    if ".dynsym" not in sections or ".dynstr" not in sections:
        return None
    sym_off, sym_size, _t, _link, entsize = sections[".dynsym"]
    if entsize not in (0, _ELF64_SYM_SIZE):
        raise ElfFormatError(f"{ref}: .dynsym entsize {entsize} is not {_ELF64_SYM_SIZE}")
    str_off, str_size, _t2, _l2, _e2 = sections[".dynstr"]
    strtab = data[str_off:str_off + str_size]
    out: list = []
    count = sym_size // _ELF64_SYM_SIZE
    for i in range(count):
        off = sym_off + i * _ELF64_SYM_SIZE
        st_name, st_info, _other, st_shndx = struct.unpack_from("<IBBH", data, off)
        end = strtab.find(b"\x00", st_name)
        if end < 0:
            raise ElfFormatError(f"{ref}: unterminated dynstr entry at {st_name}")
        name = strtab[st_name:end].decode("utf-8", errors="replace")
        if not name:
            continue
        out.append([name, st_info >> 4, st_info & 0xF, bool(st_shndx)])
    out.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return out


@dataclass(frozen=True)
class NormalizedBinaryDigest:
    """§3.2 stage 2's operand. Refuses to hold anything §3.2 excludes.

    `section_digests` maps every name in `COMPARED_SECTIONS` to a SHA-256 or to
    `SECTION_ABSENT`. A whole-file digest cannot be stored here at all — the key is
    refused with `NaiveByteIdentityRefused`, because *"llama.cpp/ROCm builds embed
    build IDs, timestamps, and absolute paths"* and a whole-file comparison would never
    fire.
    """

    ref: str
    section_digests: Mapping[str, str]
    dynsym_digest: str
    absent_sections: tuple = ()
    excluded_sections: tuple = EXCLUDED_SECTIONS
    residual_risks: tuple = (
        ".rodata can embed absolute build paths via __FILE__ and assert() text, and .text "
        "can hold PC-relative references to them; a path-only difference therefore shows up "
        "as a stage-2 disagreement. That is loud, not silent: it is filed against build "
        "identity (§3.2) rather than resolved by preferring stage 1.",
    )

    def __post_init__(self) -> None:
        _require_str(self.ref, "NormalizedBinaryDigest.ref")
        for name in self.section_digests:
            lowered = name.lower()
            if lowered in ("whole_file", "file_sha256", "sha256", "binary_sha256"):
                raise NaiveByteIdentityRefused(
                    f"{self.ref}: {name!r} is a whole-binary digest. §3.2: the test is NOT "
                    "naive byte-identity of the built binary; ROCm builds embed build IDs, "
                    "timestamps and paths, so such a test would never fire.")
            if name in EXCLUDED_SECTIONS or name.startswith(EXCLUDED_SECTION_PREFIXES):
                raise NormalizationViolation(
                    f"{self.ref}: section {name!r} is excluded by §3.2 and must not be part "
                    "of a normalized digest")
        missing = [s for s in COMPARED_SECTIONS if s not in self.section_digests]
        if missing:
            raise NormalizationViolation(
                f"{self.ref}: normalized digest is missing {missing}; every compared section "
                f"must be present or explicitly {SECTION_ABSENT!r}")
        # A key spelled ABSENT satisfies the clause above, so a digest in which
        # EVERY section is ABSENT would pass the shape check and then compare equal
        # to any other such digest — a failed extractor would read as "the binary is
        # unchanged". A linked binary always has `.text`
        # (`read_normalized_binary_digest()` refuses one without it), so `.text:
        # ABSENT` is an extraction gap, not a comparison.
        if self.section_digests.get(".text") == SECTION_ABSENT:
            raise NormalizationViolation(
                f"{self.ref}: .text is {SECTION_ABSENT!r}. A linked binary always has a "
                ".text section, so this digest records a failed extraction, not a "
                "binary. Two such digests compare identical, which would report an "
                "unmeasured backend as unchanged (§3.2).")

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "section_digests": {k: v for k, v in sorted(self.section_digests.items())},
            "dynsym_digest": self.dynsym_digest,
            "absent_sections": list(self.absent_sections),
            "excluded_sections": list(self.excluded_sections),
        }

    def digest(self) -> str:
        return schemas.content_hash(self.to_dict())

    def differences(self, other: "NormalizedBinaryDigest") -> tuple:
        if not isinstance(other, NormalizedBinaryDigest):
            raise SurfaceInputError("differences() needs another NormalizedBinaryDigest")
        out: list = []
        for section in COMPARED_SECTIONS:
            mine = self.section_digests.get(section, SECTION_ABSENT)
            theirs = other.section_digests.get(section, SECTION_ABSENT)
            if mine != theirs:
                out.append(f"{section}: {mine} != {theirs}")
        if self.dynsym_digest != other.dynsym_digest:
            out.append(f"dynamic symbol table: {self.dynsym_digest} != {other.dynsym_digest}")
        return tuple(out)


def normalized_binary_digest_from_sections(*, ref: str,
                                           section_digests: Mapping[str, str],
                                           dynsym_digest: str) -> NormalizedBinaryDigest:
    """Build a digest from already-extracted section hashes (fixtures, or a remote
    extractor). Absent sections must be spelled `SECTION_ABSENT`, never omitted."""
    absent = tuple(s for s in COMPARED_SECTIONS
                   if section_digests.get(s) == SECTION_ABSENT)
    return NormalizedBinaryDigest(ref=ref, section_digests=dict(section_digests),
                                  dynsym_digest=dynsym_digest, absent_sections=absent)


def read_normalized_binary_digest(path: Any, *, ref: Optional[str] = None
                                  ) -> NormalizedBinaryDigest:
    """Read an ELF binary and produce its §3.2 normalized digest.

    Reads only. `.text`, `.rodata` and `.data.rel.ro` are hashed from their file bytes;
    the dynamic symbol table is canonicalised to its symbol set first. `.comment`,
    `.note.*` (including `.note.gnu.build-id`) and debug sections are never read into
    the digest.
    """
    p = Path(path)
    label = ref or str(p)
    try:
        with open(p, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise SurfaceInputError(f"cannot read binary {p}: {exc}") from exc

    sections = _parse_elf_sections(data, label)
    if ".text" not in sections:
        raise ElfFormatError(f"{label}: no .text section; this is not a linked binary")

    digests: dict = {}
    absent: list = []
    for name in COMPARED_SECTIONS:
        if name not in sections:
            digests[name] = SECTION_ABSENT
            absent.append(name)
            continue
        offset, size, sh_type, _link, _ent = sections[name]
        if sh_type == _SHT_NOBITS:
            digests[name] = SECTION_ABSENT
            absent.append(name)
            continue
        blob = data[offset:offset + size]
        if len(blob) != size:
            raise ElfFormatError(f"{label}: section {name} is truncated "
                                 f"({len(blob)} of {size} bytes)")
        digests[name] = hashlib.sha256(blob).hexdigest()

    canonical = _dynsym_canonical(data, sections, label)
    if canonical is None:
        dynsym_digest = SECTION_ABSENT
        absent.append(".dynsym")
    else:
        dynsym_digest = schemas.content_hash(canonical)

    return NormalizedBinaryDigest(ref=label, section_digests=digests,
                                  dynsym_digest=dynsym_digest, absent_sections=tuple(absent))


def compare_binaries_byte_identical(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. Present so the wrong test has a name that refuses.

    §3.2: *"The test is not naive byte-identity of the built binary. llama.cpp/ROCm
    builds embed build IDs, timestamps, and absolute paths, so a freshly built binary
    is essentially never byte-identical to one built months earlier in a different
    directory — a test formulated that way would never fire."*
    """
    raise NaiveByteIdentityRefused(
        "naive whole-binary byte comparison is not the backend-unchanged test; use "
        "read_normalized_binary_digest() + backend_unchanged_stage2_normalized_binary()")


@dataclass(frozen=True)
class RebuildAttestation:
    """Proof that the production base was rebuilt in the CANDIDATE's build environment.

    Without it, stage 2 compares two non-determinism regimes and its answer means
    nothing. Note this is NOT an anchor rebuild in the protocol's sense — *"a rebuilt
    anchor is a different anchor"* (precondition 4). The rebuild exists only to put both
    sides of a NORMALIZED SECTION comparison under one regime; the anchor of record is
    still the archived one, and the gate carries `requires_anchor=True`.
    """

    rebuilt_commit: str
    build_dir: str
    toolchain: ToolchainIdentity
    build_log_sha256: str

    def __post_init__(self) -> None:
        for name in ("rebuilt_commit", "build_dir", "build_log_sha256"):
            _require_str(getattr(self, name), f"RebuildAttestation.{name}")
        if not isinstance(self.toolchain, ToolchainIdentity):
            raise SurfaceInputError("RebuildAttestation.toolchain must be a ToolchainIdentity")

    def to_dict(self) -> dict:
        return {"rebuilt_commit": self.rebuilt_commit, "build_dir": self.build_dir,
                "toolchain_digest": self.toolchain.digest(),
                "build_log_sha256": self.build_log_sha256}


@dataclass(frozen=True)
class SourceClosureIdentity:
    """§3.2 stage 1 result — the gate."""

    backend: str
    closure_size: int
    changed_in_closure: tuple
    unmapped_diff_paths: tuple
    toolchain_differences: tuple
    check: schemas.Check
    #: The commits the diff was taken over. Recorded so `backend_unchanged()` can
    #: refuse to combine two stages that are about different trees; `None` means the
    #: caller built this result by hand and the cross-check cannot run.
    base_commit: Optional[str] = None
    candidate_commit: Optional[str] = None

    def to_dict(self) -> dict:
        return {"stage": "source_closure_identity", "backend": self.backend,
                "closure_size": self.closure_size,
                "changed_in_closure": list(self.changed_in_closure),
                "unmapped_diff_paths": list(self.unmapped_diff_paths),
                "toolchain_differences": list(self.toolchain_differences),
                "base_commit": self.base_commit,
                "candidate_commit": self.candidate_commit,
                "check": {"outcome": self.check.outcome, "reasons": list(self.check.reasons)}}


@dataclass(frozen=True)
class NormalizedBinaryIdentity:
    """§3.2 stage 2 result — the confirmation."""

    backend: str
    candidate_ref: str
    base_ref: str
    differing: tuple
    rebuild_verified: bool
    check: schemas.Check
    #: The production base this comparison was against. See SourceClosureIdentity.
    base_commit: Optional[str] = None

    def to_dict(self) -> dict:
        return {"stage": "normalized_binary_identity", "backend": self.backend,
                "candidate_ref": self.candidate_ref, "base_ref": self.base_ref,
                "base_commit": self.base_commit,
                "differing": list(self.differing), "rebuild_verified": self.rebuild_verified,
                "compared_sections": list(COMPARED_SECTIONS),
                "excluded_sections": list(EXCLUDED_SECTIONS),
                "check": {"outcome": self.check.outcome, "reasons": list(self.check.reasons)}}


@dataclass(frozen=True)
class EvidenceTransferScope:
    """§3.2: *"Transfer additionally requires the incumbent's evidence to still be in
    scope — same models and recipes, same topology hash, no era boundary crossed."*

    Every field is `Optional[bool]`/`Optional[str]` and `None` means unknown, which
    yields COULD_NOT_CHECK. There is no default of True.
    """

    same_models: Optional[bool] = None
    same_recipes: Optional[bool] = None
    candidate_topology_hash: Optional[str] = None
    incumbent_topology_hash: Optional[str] = None
    era_boundary_crossed: Optional[bool] = None

    def check(self) -> schemas.Check:
        unknown: list = []
        failed: list = []
        if self.same_models is None:
            unknown.append("same_models is unknown")
        elif not self.same_models:
            failed.append("the incumbent's evidence was taken on different models")
        if self.same_recipes is None:
            unknown.append("same_recipes is unknown")
        elif not self.same_recipes:
            failed.append("the incumbent's evidence was taken under different recipes")
        if not self.candidate_topology_hash or not self.incumbent_topology_hash:
            unknown.append("a topology hash is missing")
        elif self.candidate_topology_hash != self.incumbent_topology_hash:
            failed.append(f"topology hash differs: {self.incumbent_topology_hash} -> "
                          f"{self.candidate_topology_hash}")
        if self.era_boundary_crossed is None:
            unknown.append("era_boundary_crossed is unknown")
        elif self.era_boundary_crossed:
            failed.append("an era boundary was crossed for this backend")
        if failed:
            return schemas.Check(schemas.FAIL, tuple(failed))
        if unknown:
            return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unknown))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        chk = self.check()
        return {"same_models": self.same_models, "same_recipes": self.same_recipes,
                "candidate_topology_hash": self.candidate_topology_hash,
                "incumbent_topology_hash": self.incumbent_topology_hash,
                "era_boundary_crossed": self.era_boundary_crossed,
                "check": {"outcome": chk.outcome, "reasons": list(chk.reasons)}}


FINDING_STAGE_DISAGREEMENT_SOURCE_CLEAN = "STAGE_DISAGREEMENT_SOURCE_CLEAN_BINARY_DIFFERS"
FINDING_STAGE_DISAGREEMENT_SOURCE_DIRTY = "STAGE_DISAGREEMENT_SOURCE_CHANGED_BINARY_IDENTICAL"
FINDING_STAGE2_NOT_RUN = "STAGE2_NOT_RUN_BEFORE_DROPPING_CELLS"
FINDING_CORE_HEADER_REQUIRES_STAGE2 = "CORE_HEADER_REQUIRES_BINARY_STAGE"

BUILD_IDENTITY_FINDINGS = (
    FINDING_STAGE_DISAGREEMENT_SOURCE_CLEAN, FINDING_STAGE_DISAGREEMENT_SOURCE_DIRTY,
    FINDING_STAGE2_NOT_RUN, FINDING_CORE_HEADER_REQUIRES_STAGE2,
)


@dataclass(frozen=True)
class BuildIdentityFinding:
    """A finding filed against the BUILD-IDENTITY machinery, not against the candidate.

    §3.2: *"Disagreement between the stages is a hard finding, never a silent
    preference for the cheaper answer: the closure is wrong or the build is
    non-deterministic, the backend owes full evidence, and the discrepancy is filed
    against the build-identity machinery."*
    """

    code: str
    severity: str
    detail: str
    filed_against: str = "build_identity"

    def __post_init__(self) -> None:
        if self.code not in BUILD_IDENTITY_FINDINGS:
            raise SurfaceInputError(
                f"finding code {self.code!r} is not one of {list(BUILD_IDENTITY_FINDINGS)}")
        if self.severity not in ("hard", "blocking"):
            raise SurfaceInputError("finding severity must be 'hard' or 'blocking'")

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "detail": self.detail,
                "filed_against": self.filed_against}


@dataclass(frozen=True)
class BackendUnchangedResult:
    """Whether a backend's cells may be dropped, and why not when they may not."""

    backend: str
    stage1: SourceClosureIdentity
    stage2: Optional[NormalizedBinaryIdentity]
    transfer_scope: EvidenceTransferScope
    agreement: schemas.Check
    unchanged: schemas.Check
    may_drop_cells: bool
    findings: tuple
    blocking_reasons: tuple

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "stage1": self.stage1.to_dict(),
            "stage2": None if self.stage2 is None else self.stage2.to_dict(),
            "transfer_scope": self.transfer_scope.to_dict(),
            "agreement": {"outcome": self.agreement.outcome,
                          "reasons": list(self.agreement.reasons)},
            "unchanged": {"outcome": self.unchanged.outcome,
                          "reasons": list(self.unchanged.reasons)},
            "may_drop_cells": self.may_drop_cells,
            "findings": [f.to_dict() for f in self.findings],
            "blocking_reasons": list(self.blocking_reasons),
        }

    def gate_result(self) -> api.GateResult:
        """`requires_anchor=True`: this is an identity claim against the anchor, and
        precondition 4 forbids a byte-identity label produced without a named anchor
        comparison. api.py demotes its PASS to COULD_NOT_CHECK when no anchor is bound."""
        return api.GateResult(
            gate_id=f"build_identity.backend_unchanged.{self.backend}",
            gate_class=api.GATE_INTEGRITY,
            check=self.agreement if self.agreement.outcome == schemas.FAIL else self.unchanged,
            requires_anchor=True,
            notes=(f"may_drop_cells={self.may_drop_cells}",) + self.blocking_reasons,
        )


def backend_unchanged_stage1_source_closure(*,
                                            backend: str,
                                            diff: SourceDiff,
                                            indexes: Sequence[BuildDependencyIndex],
                                            candidate_toolchain: ToolchainIdentity,
                                            base_toolchain: ToolchainIdentity,
                                            ) -> SourceClosureIdentity:
    """§3.2 stage 1 — source-closure identity, the gate.

    *"Obtain the backend's build-target dependency closure from the build system itself
    (CMake/Ninja depfiles), never a hand-maintained list or a directory-prefix guess.
    Diff `production_base..candidate` restricted to that closure. Unchanged iff the diff
    is empty AND toolchain, flags, and build environment are identical."*

    A diff path that resolves in NO index is `COULD_NOT_CHECK`: we cannot show it is
    outside the closure, and "not provably inside" is not "outside".
    """
    if backend not in schemas.BACKENDS:
        raise SurfaceInputError(f"backend {backend!r} is not one of {sorted(schemas.BACKENDS)}")
    if not isinstance(diff, SourceDiff):
        raise SurfaceInputError("diff must be a SourceDiff")
    indexes = tuple(indexes)
    if not indexes:
        raise SurfaceInputError("stage 1 needs at least one BuildDependencyIndex")
    for index in indexes:
        if not isinstance(index, BuildDependencyIndex):
            raise UntrustedProvenance(
                f"indexes must be BuildDependencyIndex, got {type(index).__name__}")
    for name, value in (("candidate_toolchain", candidate_toolchain),
                        ("base_toolchain", base_toolchain)):
        if not isinstance(value, ToolchainIdentity):
            raise SurfaceInputError(f"{name} must be a ToolchainIdentity")

    closure: set = set()
    have_backend = False
    for index in indexes:
        if index.link_targets_for_backend(backend):
            have_backend = True
            closure.update(index.source_closure_for_backend(backend))

    tool_diffs = candidate_toolchain.differences(base_toolchain)
    changed = tuple(p for p in diff.touched_paths if p in closure)
    unmapped = tuple(p for p in diff.touched_paths
                     if not any(index.objects_for_source(p) for index in indexes))

    index_coverage = _combine_checks(*[i.coverage_check() for i in indexes])

    if not have_backend:
        check = schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"no supplied build index declares link targets for backend {backend!r}; its "
             "closure is unknown, and an unknown closure cannot show a backend unchanged",))
    elif not closure:
        # An EMPTY closure contains no diff path, so every later branch would read
        # "unchanged" for any diff whatever. A backend whose link targets resolve to
        # no source is a hole in the build-system evidence, not a proof of identity.
        check = schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"backend {backend!r} has link targets "
             f"{[t for i in indexes for t in i.link_targets_for_backend(backend)]} but its "
             "build-system-derived closure is EMPTY; an empty closure contains no diff "
             "path, so it would answer 'unchanged' for every change",))
    elif changed:
        check = schemas.Check(
            schemas.FAIL,
            (f"{len(changed)} diff paths lie inside {backend}'s build-system-derived closure: "
             f"{list(changed[:10])}" + ("…" if len(changed) > 10 else ""),))
    elif tool_diffs:
        check = schemas.Check(
            schemas.FAIL,
            ("the source closure is unchanged but the build does not reproduce it: "
             + "; ".join(tool_diffs),))
    elif unmapped:
        check = schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"{len(unmapped)} diff paths resolve in no depfile ({list(unmapped[:10])}"
             + (f", and {len(unmapped) - 10} more" if len(unmapped) > 10 else "")
             + "); they cannot be shown to lie outside the closure",))
    elif index_coverage.outcome != schemas.PASS:
        check = schemas.Check(schemas.COULD_NOT_CHECK, index_coverage.reasons)
    else:
        check = schemas.Check(schemas.PASS)

    return SourceClosureIdentity(backend=backend, closure_size=len(closure),
                                 changed_in_closure=changed, unmapped_diff_paths=unmapped,
                                 toolchain_differences=tool_diffs, check=check,
                                 base_commit=diff.base_commit,
                                 candidate_commit=diff.candidate_commit)


def backend_unchanged_stage2_normalized_binary(*,
                                               backend: str,
                                               candidate_digest: NormalizedBinaryDigest,
                                               base_digest: NormalizedBinaryDigest,
                                               candidate_toolchain: ToolchainIdentity,
                                               base_commit: str,
                                               rebuild: Optional[RebuildAttestation],
                                               ) -> NormalizedBinaryIdentity:
    """§3.2 stage 2 — normalized binary confirmation, required before dropping cells.

    *"Rebuild the production base commit in the candidate's build environment so both
    share one non-determinism regime, then compare normalized hashes of `.text`,
    `.rodata`, `.data.rel.ro`, and the dynamic symbol table."*

    Without a `RebuildAttestation` proving that, the result is `COULD_NOT_CHECK`. A
    comparison against an archived incumbent binary crosses two non-determinism regimes
    and its FAIL would be meaningless — and its PASS would be luck.
    """
    if backend not in schemas.BACKENDS:
        raise SurfaceInputError(f"backend {backend!r} is not one of {sorted(schemas.BACKENDS)}")
    for name, value in (("candidate_digest", candidate_digest), ("base_digest", base_digest)):
        if not isinstance(value, NormalizedBinaryDigest):
            raise SurfaceInputError(f"{name} must be a NormalizedBinaryDigest")
    if not isinstance(candidate_toolchain, ToolchainIdentity):
        raise SurfaceInputError("candidate_toolchain must be a ToolchainIdentity")
    _require_str(base_commit, "base_commit")

    differing = candidate_digest.differences(base_digest)

    if rebuild is None:
        return NormalizedBinaryIdentity(
            backend=backend, candidate_ref=candidate_digest.ref, base_ref=base_digest.ref,
            differing=differing, rebuild_verified=False, base_commit=base_commit,
            check=schemas.Check(
                schemas.COULD_NOT_CHECK,
                ("no RebuildAttestation: §3.2 requires the production base to be rebuilt in "
                 "the candidate's build environment so both sides share one non-determinism "
                 "regime; comparing an archived incumbent binary compares two regimes",)))
    if not isinstance(rebuild, RebuildAttestation):
        raise SurfaceInputError("rebuild must be a RebuildAttestation or None")
    if rebuild.rebuilt_commit != base_commit:
        return NormalizedBinaryIdentity(
            backend=backend, candidate_ref=candidate_digest.ref, base_ref=base_digest.ref,
            differing=differing, rebuild_verified=False, base_commit=base_commit,
            check=schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"the rebuild attests commit {rebuild.rebuilt_commit} but the production base "
                 f"is {base_commit}; this comparison is against the wrong tree",)))
    tool_diffs = rebuild.toolchain.differences(candidate_toolchain)
    if tool_diffs:
        return NormalizedBinaryIdentity(
            backend=backend, candidate_ref=candidate_digest.ref, base_ref=base_digest.ref,
            differing=differing, rebuild_verified=False, base_commit=base_commit,
            check=schemas.Check(
                schemas.COULD_NOT_CHECK,
                ("the base was not rebuilt in the candidate's build environment: "
                 + "; ".join(tool_diffs),)))

    if differing:
        check = schemas.Check(
            schemas.FAIL,
            (f"normalized sections differ between {candidate_digest.ref} and "
             f"{base_digest.ref}: " + "; ".join(differing),))
    else:
        check = schemas.Check(schemas.PASS)
    return NormalizedBinaryIdentity(
        backend=backend, candidate_ref=candidate_digest.ref, base_ref=base_digest.ref,
        differing=differing, rebuild_verified=True, base_commit=base_commit, check=check)


def backend_unchanged(*,
                      stage1: SourceClosureIdentity,
                      stage2: Optional[NormalizedBinaryIdentity] = None,
                      transfer_scope: Optional[EvidenceTransferScope] = None,
                      change_class: Optional[str] = None) -> BackendUnchangedResult:
    """Combine the two §3.2 stages. Disagreement is a HARD FINDING, both directions.

    `may_drop_cells` is True only when: stage 1 PASSes, stage 2 ran and PASSes, the two
    agree, and the incumbent's evidence is still in scope. Every other combination
    returns False WITH the reason, so a caller cannot mistake "not proven unchanged"
    for "unchanged".
    """
    if not isinstance(stage1, SourceClosureIdentity):
        raise SurfaceInputError("stage1 must be a SourceClosureIdentity")
    if stage2 is not None and not isinstance(stage2, NormalizedBinaryIdentity):
        raise SurfaceInputError("stage2 must be a NormalizedBinaryIdentity or None")
    if stage2 is not None and stage2.backend != stage1.backend:
        raise SurfaceInputError(
            f"stage1 is for {stage1.backend!r} and stage2 for {stage2.backend!r}")
    # Matching backends are not enough: the two stages must also be about the same
    # production base. Combining a closure diff taken over one base with a binary
    # comparison taken against another produces a single "unchanged" verdict from two
    # unrelated facts. Like the candidate-id check in `reconcile_surface()`, this is a
    # wiring defect, not a finding about the candidate, so it RAISES.
    if (stage2 is not None and stage1.base_commit and stage2.base_commit
            and stage1.base_commit != stage2.base_commit):
        raise SurfaceInputError(
            f"stage 1 diffed against production base {stage1.base_commit!r} but stage 2 "
            f"compared against {stage2.base_commit!r}; the two stages are about different "
            "trees and their agreement would be meaningless")
    scope = transfer_scope if transfer_scope is not None else EvidenceTransferScope()
    if not isinstance(scope, EvidenceTransferScope):
        raise SurfaceInputError("transfer_scope must be an EvidenceTransferScope or None")
    if change_class is not None and change_class not in schemas.CHANGE_CLASSES:
        raise SurfaceInputError(
            f"change_class {change_class!r} is not one of {sorted(schemas.CHANGE_CLASSES)}")

    findings: list = []
    blocking: list = []

    s1, s2 = stage1.check.outcome, (None if stage2 is None else stage2.check.outcome)

    # --- agreement -----------------------------------------------------------
    if s2 is None:
        agreement = schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("stage 2 was not run, so the two stages cannot be compared",))
    elif s1 == schemas.PASS and s2 == schemas.FAIL:
        detail = (f"{stage1.backend}: stage 1 found the source closure unchanged but the "
                  f"normalized binary differs ({'; '.join(stage2.differing)}). Either the "
                  "closure is wrong or the build is non-deterministic.")
        findings.append(BuildIdentityFinding(
            code=FINDING_STAGE_DISAGREEMENT_SOURCE_CLEAN, severity="hard", detail=detail))
        agreement = schemas.Check(schemas.FAIL, (detail,))
    elif s1 == schemas.FAIL and s2 == schemas.PASS:
        detail = (f"{stage1.backend}: stage 1 found {len(stage1.changed_in_closure)} changes "
                  f"inside the closure ({list(stage1.changed_in_closure[:5])}"
                  + (f", and {len(stage1.changed_in_closure) - 5} more"
                     if len(stage1.changed_in_closure) > 5 else "")
                  + ") but the normalized binary is identical. Either the closure "
                    "over-reaches or the change is dead code the build eliminated; the "
                    "cheaper answer is not preferred either way.")
        findings.append(BuildIdentityFinding(
            code=FINDING_STAGE_DISAGREEMENT_SOURCE_DIRTY, severity="hard", detail=detail))
        agreement = schemas.Check(schemas.FAIL, (detail,))
    elif schemas.COULD_NOT_CHECK in (s1, s2):
        agreement = schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"stage 1 is {s1} and stage 2 is {s2}; agreement is undetermined",))
    else:
        agreement = schemas.Check(schemas.PASS)

    # --- unchanged verdict ----------------------------------------------------
    if agreement.outcome == schemas.FAIL:
        unchanged = schemas.Check(
            schemas.FAIL,
            ("the stages disagree, so the backend owes full evidence (§3.2)",))
    elif s1 == schemas.FAIL:
        unchanged = schemas.Check(schemas.FAIL, stage1.check.reasons)
    elif s1 == schemas.COULD_NOT_CHECK:
        unchanged = schemas.Check(schemas.COULD_NOT_CHECK, stage1.check.reasons)
    elif stage2 is None:
        unchanged = schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("stage 1 passed but stage 2 is required before dropping cells (§3.2)",))
    elif s2 == schemas.COULD_NOT_CHECK:
        unchanged = schemas.Check(schemas.COULD_NOT_CHECK, stage2.check.reasons)
    else:
        unchanged = schemas.Check(schemas.PASS)

    # --- blocking reasons -----------------------------------------------------
    if stage2 is None:
        if change_class in FULL_TREE_CHANGE_CLASSES:
            detail = (f"change_class={change_class} forces the binary-comparison stage of §3.2 "
                      f"for every backend the tree serves (§8.5.1); it was not run for "
                      f"{stage1.backend}")
            findings.append(BuildIdentityFinding(
                code=FINDING_CORE_HEADER_REQUIRES_STAGE2, severity="blocking", detail=detail))
            blocking.append(detail)
        elif s1 == schemas.PASS:
            detail = ("stage 2 (normalized binary confirmation) is required before dropping "
                      f"{stage1.backend}'s cells")
            findings.append(BuildIdentityFinding(
                code=FINDING_STAGE2_NOT_RUN, severity="blocking", detail=detail))
            blocking.append(detail)

    scope_check = scope.check()
    if unchanged.outcome == schemas.PASS and scope_check.outcome != schemas.PASS:
        blocking.append(
            "the binary is unchanged but the incumbent's evidence is not in scope: "
            + "; ".join(scope_check.reasons))
    if unchanged.outcome != schemas.PASS:
        blocking.extend(unchanged.reasons)

    may_drop = (unchanged.outcome == schemas.PASS
                and agreement.outcome == schemas.PASS
                and scope_check.outcome == schemas.PASS
                and not findings)

    return BackendUnchangedResult(
        backend=stage1.backend, stage1=stage1, stage2=stage2, transfer_scope=scope,
        agreement=agreement, unchanged=unchanged, may_drop_cells=may_drop,
        findings=tuple(findings), blocking_reasons=tuple(blocking))


# =============================================================================
# The seam into api.TierDispatcher
# =============================================================================

class SurfaceGateRunner:
    """An `api.TierGateRunner` returning the §6.4 and §3.2 gates.

    It launches nothing. The reconciliation and the backend-unchanged results are
    computed before it is constructed; this class only hands them to the dispatcher in
    the shape the dispatcher consumes.

    The actor's `DeclarationScore` is accepted but is **not** turned into a gate: it is
    available through `critic_payload()` only. A declaration that gated anything would
    be a scope input by another name.
    """

    def __init__(self, *,
                 tier: str,
                 reconciliation: SurfaceReconciliation,
                 backend_unchanged_results: Sequence[BackendUnchangedResult] = (),
                 declaration_score: Optional[DeclarationScore] = None) -> None:
        self.tier = api.admit_tier(tier)
        if not isinstance(reconciliation, SurfaceReconciliation):
            raise SurfaceInputError("reconciliation must be a SurfaceReconciliation")
        results = tuple(backend_unchanged_results)
        for result in results:
            if not isinstance(result, BackendUnchangedResult):
                raise SurfaceInputError(
                    f"backend_unchanged_results must hold BackendUnchangedResult, got "
                    f"{type(result).__name__}")
        if declaration_score is not None and not isinstance(declaration_score, DeclarationScore):
            raise SurfaceInputError("declaration_score must be a DeclarationScore or None")
        self._reconciliation = reconciliation
        self._backend_unchanged = results
        self._declaration_score = declaration_score

    @property
    def reconciliation(self) -> SurfaceReconciliation:
        return self._reconciliation

    def run_gates(self, request: api.EvaluationRequest) -> tuple:
        if not isinstance(request, api.EvaluationRequest):
            raise SurfaceInputError("request must be an api.EvaluationRequest")
        if request.candidate_id != self._reconciliation.derived.candidate_id:
            raise SurfaceInputError(
                f"this runner holds the surface for "
                f"{self._reconciliation.derived.candidate_id!r} but was dispatched for "
                f"{request.candidate_id!r}; returning the wrong candidate's gates is a wiring "
                "defect, not a finding")
        gates = list(self._reconciliation.gate_results())
        for result in self._backend_unchanged:
            gates.append(result.gate_result())
        return tuple(gates)

    def critic_payload(self) -> dict:
        """Everything the critic gets, including the declaration score (§6.3)."""
        return {
            "reconciliation": self._reconciliation.to_dict(),
            "declaration_score": (None if self._declaration_score is None
                                  else self._declaration_score.to_dict()),
            "backend_unchanged": [r.to_dict() for r in self._backend_unchanged],
        }


# =============================================================================
# Self-audit — read-only, no writes, no processes
# =============================================================================

_FORBIDDEN_CALL_NAMES = frozenset({"exec", "eval", "compile", "__import__", "input"})

_FORBIDDEN_CALL_ATTRS = frozenset({
    "write", "writelines", "write_text", "write_bytes", "truncate", "fsync",
    "mkdir", "makedirs", "remove", "unlink", "rmdir", "rmtree", "rename", "replace",
    "chmod", "chown", "utime", "symlink", "touch", "move", "copy", "copyfile", "copytree",
    "system", "popen", "Popen", "spawnv", "fork", "kill", "killpg", "send_signal",
    "terminate", "check_call", "check_output", "communicate", "setxattr",
})

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "tempfile", "sqlite3", "urllib", "http", "requests", "pty", "fcntl", "resource",
    "asyncio",
    # Every one of these exposes its own writable `open`, so leaving them out let
    # `io.open(p, "w")` and `codecs.open(p, "w")` through the audit.
    "io", "codecs", "gzip", "bz2", "lzma", "zipfile", "tarfile", "mmap", "pickle",
    "shelve", "dbm", "builtins", "runpy",
})

_READ_ONLY_MODES = frozenset({"r", "rb", "rt", "br", "tr"})


def audit_surface_module_is_read_only(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's own AST that it reads but never writes or signals.

    Unlike `api.audit_no_write_or_process_paths()`, `open()` is permitted here — this
    module's inputs are files — but only with a **literal read-only mode**. A
    non-literal mode is a FAIL, because a mode this audit cannot read is a mode it
    cannot clear.

    COULD_NOT_CHECK when the source cannot be read or parsed: an unreadable module is
    not an audited one.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (f"could not parse module: {exc}",))

    findings: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _FORBIDDEN_IMPORTS:
                findings.append(f"line {node.lineno}: imports from {node.module!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in _FORBIDDEN_CALL_NAMES:
                    findings.append(f"line {node.lineno}: calls {func.id}()")
                elif func.id == "open":
                    findings.extend(_audit_open_call(node, mode_index=1))
            elif isinstance(func, ast.Attribute):
                if func.attr in _FORBIDDEN_CALL_ATTRS:
                    findings.append(f"line {node.lineno}: calls .{func.attr}()")
                elif func.attr == "open":
                    # `Path(p).open("w")` is `open` in attribute form, and its mode is
                    # positional arg 0 — the receiver is the path. Auditing only the
                    # bare `open` Name, at arg index 1, left every `.open()` call
                    # unexamined, so a write-mode open through a Path object cleared
                    # an audit whose whole claim is that it does not.
                    findings.extend(_audit_open_call(node, mode_index=0))

    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    return schemas.Check(schemas.PASS)


def _audit_open_call(node: ast.Call, *, mode_index: int = 1) -> list:
    """`open()` is allowed only with a literal read-only mode.

    `mode_index` is 1 for the builtin `open(path, mode)` and 0 for the bound form
    `Path(path).open(mode)`, whose receiver carries the path.
    """
    mode_node: Optional[ast.AST] = None
    if len(node.args) > mode_index:
        mode_node = node.args[mode_index]
    for kw in node.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
    if mode_node is None:
        return []  # open(path) defaults to text read
    if not isinstance(mode_node, ast.Constant) or not isinstance(mode_node.value, str):
        return [f"line {node.lineno}: open() with a non-literal mode cannot be audited"]
    if mode_node.value not in _READ_ONLY_MODES:
        return [f"line {node.lineno}: open() with write-capable mode {mode_node.value!r}"]
    return []
