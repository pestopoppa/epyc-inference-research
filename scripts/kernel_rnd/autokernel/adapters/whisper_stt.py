#!/usr/bin/env python3
"""whisper_stt.py — the `whisper_stt` backend adapter (§13.3, phase AK9).

WHY THIS MODULE EXISTS
----------------------
`whisper.cpp` is one of three source trees this project freezes, and until AK9 it
had **no adapter and no measurement protocol of any kind** — `measurement/protocols/`
contains nothing for STT (AK-D24). Everything backend-specific about it therefore
lived nowhere: which binaries are cells, which tree its ggml comes from, what a
transcription match is, what its complexity ceiling is, and what a release owes.
This module is where those facts live, so the domain-agnostic controller never has
to know them (AK-D8: one core loop with backend adapters, never a cloned controller).

WHICH FAILURE IT PREVENTS
-------------------------
**A binary that silently runs against another tree's ggml.** Three ggml generations
coexist on this host — llama.cpp 0.16.0, qwentts.cpp 0.17.0, whisper.cpp 0.18.0 —
and the loader honours `LD_LIBRARY_PATH` before a binary's own directory. On
2026-07-31 a HIP-built `whisper-cli` resolved the production CPU-only ggml, found no
GPU, and ran full-CPU **while printing `use gpu = 1`**
(INC-20260731-ggml-linkage-silent-cpu-fallback). The run completed, the output was
well-formed, and only the throughput was quietly wrong — a measurement-integrity
failure, not a build annoyance. Every candidate build and every T3 phase-2 check
therefore goes through the research repo's `scripts/utils/verify_ggml_linkage.sh`
(§10.2 phase 2), and this module owns two clauses the raw script does not enforce:

  * a verifier `PASS` is **necessary and not sufficient** — ggml backends are
    `dlopen`ed at runtime and `ldd` cannot see them, so the engine's own startup
    device line is also required, and `use gpu = 1` reports what was *requested*,
    never what was *loaded*; and
  * a verifier run that resolved **no** libraries is `COULD_NOT_CHECK`, **never**
    `PASS`. The script prints `(no ggml/whisper/llama libs in ldd output — statically
    linked, or ldd failed)` and then `exit 0`; a consumer reading only the exit status
    converts "the check could not run" into "the check passed".

WHAT THIS MODULE IS NOT
-----------------------
**It executes nothing.** It declares facts, constructs the argv a runner must
execute, and interprets output a runner captured. It runs no inference, no benchmark
and no build; it starts, stops and signals no process; it writes no file; and it
reads no file. `audit_no_write_or_process_paths()` proves the write/process half from
this module's own AST (`test_whisper_stt.py` asserts PASS), which is the same device
`evaluator/api.py` and `evaluator/recipes.py` use.

It also **freezes nothing**. `whisper.cpp` is independently freezable (§1.5), but a
freeze, a cutover, an era-registry row, an AutoPilot baseline apply and a repoint of
`/mnt/raid0/llm/kernels/production/stt` are human-only writes (`MEASUREMENT.md:140-142`,
invariant 5). Nothing here offers any of them, and `release_gate_readiness()` returns
`COULD_NOT_CHECK` — not `PASS` — for as long as the STT protocol family is a draft.

GOVERNING INSTRUMENTS
---------------------
  * `measurement/protocols/kernel-research.md` — **P-AK-SEARCH-1** (RATIFIED
    2026-08-03), whose scope is *"Tiers T0, T1 and T2 … on every declared backend
    adapter"*. Search on this backend is already authorized; **release is not**.
  * `artifacts/operator/autokernel-policy-draft/P-STT-1.draft.md` — the STT family
    (`P-STT-1`, `P-STT-2`, `P-STT-3`, `P-STT-REL-1`). **DRAFT, not in force.**
  * `artifacts/operator/ratify_speech_kernel_freeze_20260731.json` — the operator
    receipt that froze this tree.

Design context: §1.5 (three trees, four binaries), §3.2 (sealed candidate and the
backend-unchanged test), §10.2 (release-gate phases), §10.5, §10.6 (diff-complexity
ceiling), §11 (champion to production), §13.3.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence

from .. import schemas
from ..evaluator import api, integrity

# =============================================================================
# Errors — every one is a refusal, never a degraded answer
# =============================================================================


class WhisperAdapterError(Exception):
    """Base for every refusal this adapter makes."""


class ProductionPathRefused(WhisperAdapterError):
    """A path resolves inside a frozen production tree (invariant 3, denial 2)."""


class UnknownBinary(WhisperAdapterError):
    """A binary name that is not in this backend's declared inventory."""


class UnknownPhase(WhisperAdapterError):
    """A phase name outside this backend's declared vocabulary."""


class UnknownMetric(WhisperAdapterError):
    """A metric name with no declared direction. A bare metric is unusable."""


class WrongReleasePath(WhisperAdapterError):
    """A release path this backend must refuse rather than degrade to (§13.5)."""


class DerivationImpossible(WhisperAdapterError):
    """A derived quantity has no inputs. It is refused, never defaulted."""


# =============================================================================
# Tree identity (§1.5, speech-freeze receipt 2026-07-31)
# =============================================================================

BACKEND = "whisper_stt"
SOURCE_TREE = "whisper.cpp"

#: The FROZEN production tree. Invariant 3: no actor builds in or modifies it.
PRODUCTION_TREE_ROOT = "/mnt/raid0/llm/whisper.cpp"

#: Every frozen production kernel tree on this host, mirrored from
#: `storage.PRODUCTION_TREES` / `correctness.PRODUCTION_TREE_ROOTS`. Duplicated
#: deliberately and named as a mirror rather than reaching into another module's
#: constant, the same thing `correctness.py` does; `test_whisper_stt.py` asserts
#: the three lists agree, so the duplication is checked rather than trusted.
PRODUCTION_TREE_ROOTS = (
    "/mnt/raid0/llm/llama.cpp",
    "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp",
    "/workspace/repos/epyc-llama",
)

#: Live ALIASES for the frozen trees, which `PRODUCTION_TREE_ROOTS` cannot express.
#: `/mnt/raid0/llm/kernels/production/<backend>` is a SYMLINK into a frozen build
#: directory, and `kernels/README.md` calls it *"the only path anything should
#: name"*; `archive/<backend>-<date>-<sha>` is the same device pointed at a
#: superseded target. A path reaching the frozen tree through either one is inside
#: production while comparing unequal to every lexical root. This module already
#: declares `STABLE_PATH`/`STABLE_TARGET` below, so failing to refuse them would be
#: knowing the alias and not guarding it.
PRODUCTION_PATH_ALIASES = (
    "/mnt/raid0/llm/kernels/production",
    "/mnt/raid0/llm/kernels/archive",
)

FROZEN_BRANCH = "production-speech-v1"
FROZEN_COMMIT = "b307379226d93d9c5ed790d7cea0626613c0ef4b"

#: whisper.cpp's ggml generation. Load-bearing: see the module docstring.
GGML_GENERATION = "0.18.0"

#: whisper.cpp vendors ggml **in-tree** (no `.gitmodules`); its production patch
#: edits `ggml/src/ggml-cuda/vendors/hip.h` as an ordinary file. The sibling TTS
#: tree is the opposite case, which is exactly why this is declared rather than
#: assumed — a shared assumption would be wrong for one of the two.
GGML_VENDORING = "in_tree"
SUBMODULE_PATHS: tuple = ()

#: The build directory, relative to the tree root, and the directory that holds
#: BOTH the binaries and their libraries. For this backend they are the same
#: directory; for `qwentts_tts` they are not. §1.5: *"Adapters must not assume
#: uniformity."*
BUILD_DIR_REL = "build/bin"
LIBRARY_DIR_REL = "build/bin"

#: The stable production path and what it points at (§1.5 table).
STABLE_PATH = "/mnt/raid0/llm/kernels/production/stt"
STABLE_TARGET = "/mnt/raid0/llm/whisper.cpp/build/bin"


@dataclass(frozen=True)
class TreeFacts:
    """Everything the controller needs to know about this backend's source tree."""

    backend: str
    source_tree: str
    production_tree_root: str
    frozen_branch: str
    frozen_commit: str
    ggml_generation: str
    ggml_vendoring: str
    submodule_paths: tuple
    build_dir_rel: str
    library_dir_rel: str
    stable_path: str
    stable_target: str

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "source_tree": self.source_tree,
            "production_tree_root": self.production_tree_root,
            "frozen_branch": self.frozen_branch,
            "frozen_commit": self.frozen_commit,
            "ggml_generation": self.ggml_generation,
            "ggml_vendoring": self.ggml_vendoring,
            "submodule_paths": list(self.submodule_paths),
            "build_dir_rel": self.build_dir_rel,
            "library_dir_rel": self.library_dir_rel,
            "stable_path": self.stable_path,
            "stable_target": self.stable_target,
        }


def tree_facts() -> TreeFacts:
    return TreeFacts(
        backend=BACKEND,
        source_tree=SOURCE_TREE,
        production_tree_root=PRODUCTION_TREE_ROOT,
        frozen_branch=FROZEN_BRANCH,
        frozen_commit=FROZEN_COMMIT,
        ggml_generation=GGML_GENERATION,
        ggml_vendoring=GGML_VENDORING,
        submodule_paths=SUBMODULE_PATHS,
        build_dir_rel=BUILD_DIR_REL,
        library_dir_rel=LIBRARY_DIR_REL,
        stable_path=STABLE_PATH,
        stable_target=STABLE_TARGET,
    )


# =============================================================================
# Freeze scope (§1.5, AK-D11)
# =============================================================================


@dataclass(frozen=True)
class FreezeScope:
    """Which backends a freeze of this tree necessarily covers.

    `llama.cpp` serves TWO backends and cannot be frozen for one of them
    (`llama_cpu` and `llama_gpu` share one frozen branch). `whisper.cpp` serves
    exactly one, so it is **independently freezable** — the property §1.5 states
    explicitly and the reason AK9 can ship a speech release path without touching
    the llama champion.
    """

    source_tree: str
    backends: tuple
    independently_freezable: bool
    shares_tree_with: tuple

    def to_dict(self) -> dict:
        return {"source_tree": self.source_tree, "backends": list(self.backends),
                "independently_freezable": self.independently_freezable,
                "shares_tree_with": list(self.shares_tree_with)}


def freeze_scope() -> FreezeScope:
    siblings = tuple(sorted(
        b for b, t in schemas.SOURCE_TREE_BY_BACKEND.items()
        if t == SOURCE_TREE and b != BACKEND
    ))
    return FreezeScope(source_tree=SOURCE_TREE, backends=(BACKEND,),
                       independently_freezable=not siblings,
                       shares_tree_with=siblings)


def refuse_llama_champion(champion_source_tree: str) -> None:
    """Raise if asked to join a champion lineage belonging to another tree.

    Champions are per SOURCE TREE (AK-D11). A `whisper.cpp` candidate composed
    into the llama champion would be composed with changes that cannot reach its
    binary and cannot be frozen with it.
    """
    _require_str(champion_source_tree, "champion_source_tree")
    if champion_source_tree != SOURCE_TREE:
        raise WrongReleasePath(
            f"{BACKEND} candidates belong to the {SOURCE_TREE!r} champion; "
            f"{champion_source_tree!r} is a different source tree and a different "
            f"freeze (§1.5, AK-D11)")


def refuse_stack_change_path() -> None:
    """Raise: this backend releases through a kernel freeze, not §11.6.

    The mirror of `serving_runtime`'s refusal (§13.5). A speech kernel travelling
    the three-gate stack-change path would ship a new binary with no era row, no
    correctness matrix and no sealed bundle.
    """
    raise WrongReleasePath(
        f"{BACKEND} releases through the kernel-freeze path (§10, §11), not the "
        f"three-gate stack-change path (§11.6), which is the `serving_runtime` lane")


# =============================================================================
# Small validators — local on purpose (the `correctness.py` convention), so this
# module does not depend on another module's privates
# =============================================================================

_ABS_PATH_RE = re.compile(r"^/[^\x00]*$")


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WhisperAdapterError(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _require_abs_path(value: Any, label: str) -> str:
    _require_str(value, label)
    if not _ABS_PATH_RE.match(value):
        raise WhisperAdapterError(f"{label}: expected an absolute POSIX path, got {value!r}")
    if ".." in PurePosixPath(value).parts:
        raise WhisperAdapterError(f"{label}: contains '..'; refusing to normalise a path "
                                  f"whose target depends on the filesystem: {value!r}")
    if PurePosixPath(value).parts[:1] == ("//",):
        # POSIX leaves a leading `//` implementation-defined and `PurePosixPath`
        # preserves it as a distinct root segment, so `//mnt/raid0/llm/whisper.cpp/x`
        # compares UNEQUAL to `/mnt/raid0/llm/whisper.cpp` segment-by-segment while
        # Linux opens the identical file. That is a one-character walk straight
        # through `check_not_production_path` (invariant 3). Refused rather than
        # normalised, for the same reason as `..`: the two forms are only equivalent
        # on some kernels, and a refusal is correct on all of them.
        raise WhisperAdapterError(
            f"{label}: begins with '//', which names the same file as '/' on Linux but "
            f"is a different path root to every segment-wise comparison, including this "
            f"module's production-tree refusal: {value!r}")
    return value


def _require_positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WhisperAdapterError(f"{label}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not number > 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise WhisperAdapterError(f"{label}: must be finite and strictly positive, got {value!r}")
    return number


def _is_within(path: str, root: str) -> bool:
    """True when `path` is `root` or lives under it, by path SEGMENTS.

    A `startswith` test would call `/mnt/raid0/llm/whisper.cpp-experimental` a
    production path, which would refuse exactly the tree candidates are supposed
    to be built in.
    """
    parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return parts[:len(root_parts)] == root_parts


def check_not_production_path(path: str, *, label: str = "path") -> None:
    """Raise `ProductionPathRefused` when `path` is inside any frozen production tree.

    Invariant 3 (*"No actor builds in or modifies any production tree"*) and
    P-AK-SEARCH-1 denial 2 (*"No production write of any kind"*). This is a
    CANDIDATE-side check: the ANCHOR arm legitimately lives in the production tree,
    and executing a frozen binary read-only is not a write, so anchor paths are
    checked with `expect_production_anchor()` instead.
    """
    _require_abs_path(path, label)
    for root in PRODUCTION_TREE_ROOTS:
        if _is_within(path, root):
            raise ProductionPathRefused(
                f"{label} {path!r} is inside the frozen production tree {root!r}. "
                f"Candidates are built and measured in experimental worktrees only "
                f"(invariant 3; P-AK-SEARCH-1 denial 2)")
    for alias in PRODUCTION_PATH_ALIASES:
        if _is_within(path, alias):
            raise ProductionPathRefused(
                f"{label} {path!r} reaches a frozen production tree THROUGH the stable "
                f"kernel alias {alias!r} ({STABLE_PATH} -> {STABLE_TARGET}). The lexical "
                f"root list cannot see it, and `kernels/README.md` makes this the only "
                f"path anything is supposed to name — so it is the path a caller is "
                f"most likely to hand us (invariant 3; P-AK-SEARCH-1 denial 2)")


def expect_production_anchor(path: str, *, label: str = "anchor_path") -> str:
    """The inverse check: an ANCHOR path MUST be inside this backend's production tree.

    An "anchor" that is not the frozen binary is not an anchor. P-AK-SEARCH-1
    precondition 4: *"A run without an explicit anchor is INVALID."*
    """
    _require_abs_path(path, label)
    if not _is_within(path, PRODUCTION_TREE_ROOT):
        raise WhisperAdapterError(
            f"{label} {path!r} is not inside {PRODUCTION_TREE_ROOT!r}; the {BACKEND} "
            f"anchor is the FROZEN production binary, and a rebuilt anchor is a "
            f"different anchor (P-AK-SEARCH-1 precondition 4)")
    return path


# =============================================================================
# Binary inventory and path construction
# =============================================================================


@dataclass(frozen=True)
class BinarySpec:
    """One binary this backend measures, and what it is for."""

    name: str
    rel_path: str
    role: str

    def to_dict(self) -> dict:
        return {"name": self.name, "rel_path": self.rel_path, "role": self.role}


#: Verified against the frozen tree's own `build/bin` on 2026-08-03. `role` is what
#: the loop uses it for, not what upstream calls it.
BINARY_INVENTORY = (
    BinarySpec("whisper-cli", "build/bin/whisper-cli", "transcription_cell"),
    BinarySpec("whisper-server", "build/bin/whisper-server", "service_smoke"),
    BinarySpec("whisper-bench", "build/bin/whisper-bench", "operator_microbench"),
    BinarySpec("whisper-quantize", "build/bin/whisper-quantize", "quantization_tool"),
    BinarySpec("test-vad", "build/bin/test-vad", "op_and_unit_test"),
    BinarySpec("test-vad-full", "build/bin/test-vad-full", "op_and_unit_test"),
)

_BINARIES_BY_NAME = {b.name: b for b in BINARY_INVENTORY}


def binary_inventory() -> tuple:
    return BINARY_INVENTORY


def binary_path(tree_root: str, name: str, *, allow_production: bool = False) -> str:
    """Absolute path of `name` inside `tree_root`, with the layout DECLARED.

    The `bin/` segment is part of `BinarySpec.rel_path`, never appended by a caller:
    the `qwentts_tts` sibling's binaries are NOT in a `bin/` subdirectory, so a
    shared "append bin/" convention silently produces a non-existent path for one
    of the two backends (§1.5).
    """
    _require_abs_path(tree_root, "tree_root")
    _require_str(name, "name")
    try:
        spec = _BINARIES_BY_NAME[name]
    except KeyError as exc:
        raise UnknownBinary(
            f"{name!r} is not in the {BACKEND} binary inventory; declared binaries are "
            f"{sorted(_BINARIES_BY_NAME)}") from exc
    path = str(PurePosixPath(tree_root) / spec.rel_path)
    if not allow_production:
        check_not_production_path(path, label=f"binary_path({name!r})")
    return path


def library_dir(tree_root: str, *, allow_production: bool = False) -> str:
    """Where this tree's own ggml/whisper shared objects live."""
    _require_abs_path(tree_root, "tree_root")
    path = str(PurePosixPath(tree_root) / LIBRARY_DIR_REL)
    if not allow_production:
        check_not_production_path(path, label="library_dir")
    return path


#: Shared objects a correctly linked whisper.cpp binary resolves from its OWN tree.
#: `libparakeet` is included deliberately: the verifier script's name filter is
#: `libggml*|libwhisper*|libllama*|libmtmd*`, so a `libparakeet` resolving from
#: somewhere else would not appear as a BAD line at all. `interpret_linkage_report`
#: treats an expected library missing from the report as COULD_NOT_CHECK rather than
#: as a pass, which is the whole point of declaring the set here.
EXPECTED_SHARED_LIBRARIES = frozenset({
    "libggml-base.so",
    "libggml-cpu.so",
    "libggml.so",
    "libwhisper.so",
})

#: Libraries present in this tree but only linked by some binaries. Their absence
#: from a report is not a finding; their presence from the WRONG tree is.
OPTIONAL_SHARED_LIBRARIES = frozenset({
    "libggml-hip.so",
    "libparakeet.so",
})


def expected_shared_libraries() -> frozenset:
    return EXPECTED_SHARED_LIBRARIES


# =============================================================================
# Linkage verification (§10.2 phase 2, INC-20260731-ggml-linkage-silent-cpu-fallback)
# =============================================================================

#: The verifier lives in **epyc-inference-research**, not epyc-root. CLAUDE.md cites
#: it unqualified, which is the same defect class as the durability validator's path
#: in `MEASUREMENT.md:155` (§10.2 phase 2 says so by name).
LINKAGE_VERIFIER = (
    "/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
)


@dataclass(frozen=True)
class LinkageInvocation:
    """The exact command a runner must execute, with a FULLY DECLARED environment.

    The environment is complete, not a prepend onto whatever the caller's shell
    happened to export. An ambient `LD_LIBRARY_PATH` is precisely the mechanism of
    the 2026-07-31 incident, and it would also make the invocation's identity a
    function of who invoked it.
    """

    argv: tuple
    env: dict
    binary: str
    expected_root: str

    def to_dict(self) -> dict:
        return {"argv": list(self.argv), "env": dict(self.env),
                "binary": self.binary, "expected_root": self.expected_root}


def linkage_command(binary: str, *, library_path_entries: Sequence[str],
                    expected_root: Optional[str] = None) -> LinkageInvocation:
    """Construct the `verify_ggml_linkage.sh` invocation for `binary`.

    `library_path_entries` is the COMPLETE ordered `LD_LIBRARY_PATH`, and its first
    entry MUST be the binary's own tree library directory — that ordering is the
    property being verified, so accepting an arbitrary order would verify nothing.
    """
    _require_abs_path(binary, "binary")
    entries = [_require_abs_path(e, f"library_path_entries[{i}]")
               for i, e in enumerate(library_path_entries)]
    if not entries:
        raise WhisperAdapterError(
            "library_path_entries is empty; the invocation must declare the complete "
            "LD_LIBRARY_PATH, because inheriting the ambient one is the 2026-07-31 "
            "failure mode itself")
    own_dir = str(PurePosixPath(binary).parent)
    if entries[0] != own_dir:
        raise WhisperAdapterError(
            f"library_path_entries[0] is {entries[0]!r} but must be the binary's own "
            f"directory {own_dir!r}: the loader honours LD_LIBRARY_PATH before a "
            f"binary's own directory, so anything else lets another tree's ggml win")
    root = own_dir if expected_root is None else _require_abs_path(expected_root,
                                                                  "expected_root")
    return LinkageInvocation(
        argv=(LINKAGE_VERIFIER, binary, root),
        env={"LD_LIBRARY_PATH": ":".join(entries)},
        binary=binary,
        expected_root=root,
    )


@dataclass(frozen=True)
class LinkageVerdict:
    """The interpretation of a captured verifier report.

    `check` is a `schemas.Check`, so COULD_NOT_CHECK is expressible and is NOT a
    soft pass — which is exactly the state the raw script cannot express, since it
    exits 0 both when every library resolved correctly and when `ldd` failed.
    """

    check: schemas.Check
    ok_libraries: tuple
    bad_libraries: tuple
    missing_expected: tuple
    resolved_count: int

    def to_dict(self) -> dict:
        return {"outcome": self.check.outcome, "reasons": list(self.check.reasons),
                "ok_libraries": list(self.ok_libraries),
                "bad_libraries": list(self.bad_libraries),
                "missing_expected": list(self.missing_expected),
                "resolved_count": self.resolved_count}


_OK_LINE_RE = re.compile(r"^\s{2}OK\s+(\S+)\s+->\s+(\S+)\s*$")
_BAD_LINE_RE = re.compile(r"^\s{2}BAD\s+(\S+)\s+->\s+(\S+)\s*$")
_NO_LIBS_MARKER = "no ggml/whisper/llama libs in ldd output"


def _soname_stem(name: str) -> str:
    """`libggml-base.so.0.18.0` -> `libggml-base.so`. Version suffixes are not identity."""
    head, sep, _ = name.partition(".so")
    return head + sep if sep else name


def interpret_linkage_report(stdout: str, exit_code: int) -> LinkageVerdict:
    """Turn a captured verifier report into a three-outcome verdict.

    Four rules, and the last two are the ones the raw script cannot express:

      1. any `BAD` line, or a non-zero exit  ->  FAIL;
      2. every expected library present and `OK`, exit 0  ->  PASS;
      3. **zero libraries resolved  ->  COULD_NOT_CHECK, never PASS** — the script
         prints its "statically linked, or ldd failed" marker and exits 0, and a
         consumer reading the exit status alone would record a pass for a check that
         did not run;
      4. an expected library **missing from the report**  ->  COULD_NOT_CHECK naming
         the script's `libggml*|libwhisper*|libllama*|libmtmd*` name filter, because a
         library outside that filter is not examined at all and its absence from the
         report is silence, not evidence.
    """
    if not isinstance(stdout, str):
        raise WhisperAdapterError(f"stdout must be a string, got {type(stdout).__name__}")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise WhisperAdapterError("exit_code must be an int")

    ok: list = []
    bad: list = []
    for line in stdout.splitlines():
        match = _OK_LINE_RE.match(line)
        if match:
            ok.append((match.group(1), match.group(2)))
            continue
        match = _BAD_LINE_RE.match(line)
        if match:
            bad.append((match.group(1), match.group(2)))

    resolved = len(ok) + len(bad)
    seen_stems = {_soname_stem(name) for name, _ in ok}
    missing = tuple(sorted(EXPECTED_SHARED_LIBRARIES - seen_stems))

    if bad:
        offenders = ", ".join(f"{n} -> {p}" for n, p in bad)
        return LinkageVerdict(
            check=schemas.Check(schemas.FAIL, (
                f"{len(bad)} library/libraries resolve outside the candidate's own tree: "
                f"{offenders}. Any performance number produced now is attributable to the "
                f"WRONG BUILD (INC-20260731-ggml-linkage-silent-cpu-fallback)",)),
            ok_libraries=tuple(sorted(ok)), bad_libraries=tuple(sorted(bad)),
            missing_expected=missing, resolved_count=resolved)

    if exit_code != 0:
        return LinkageVerdict(
            check=schemas.Check(schemas.FAIL, (
                f"verifier exited {exit_code} with no BAD line parsed; the report is "
                f"inconsistent with its own exit status and is not trusted",)),
            ok_libraries=tuple(sorted(ok)), bad_libraries=(),
            missing_expected=missing, resolved_count=resolved)

    if resolved == 0 or _NO_LIBS_MARKER in stdout:
        return LinkageVerdict(
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                "the verifier resolved no ggml/whisper libraries at all — statically "
                "linked, or `ldd` failed. It exits 0 in this state, so an exit-status "
                "consumer would record a PASS for a check that did not run",)),
            ok_libraries=(), bad_libraries=(), missing_expected=missing,
            resolved_count=0)

    if missing:
        return LinkageVerdict(
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                f"expected libraries absent from the report: {list(missing)}. The "
                f"verifier's name filter is libggml*/libwhisper*/libllama*/libmtmd*, so a "
                f"library outside it is never examined and its absence is silence, not "
                f"evidence",)),
            ok_libraries=tuple(sorted(ok)), bad_libraries=(),
            missing_expected=missing, resolved_count=resolved)

    return LinkageVerdict(check=schemas.Check(schemas.PASS),
                          ok_libraries=tuple(sorted(ok)), bad_libraries=(),
                          missing_expected=(), resolved_count=resolved)


#: What a whisper.cpp binary prints when a real device was LOADED. A device line
#: names the device; a request flag names an intention.
_DEVICE_LINE_RE = re.compile(r"Device\s+\d+\s*:\s*(?P<name>[^\n,]+)", re.IGNORECASE)
_REQUEST_FLAG_RE = re.compile(r"use\s+gpu\s*=\s*1", re.IGNORECASE)


def check_device_evidence(startup_log: str, *, expected_lane: str) -> schemas.Check:
    """Confirm from the engine's OWN startup log which device actually loaded.

    `verify_ggml_linkage.sh` says it in its own PASS message: ggml backends are
    `dlopen`ed at runtime and are not covered by `ldd`, so *"do not trust a
    `use gpu = 1` flag alone — that flag reports what was REQUESTED, not what was
    LOADED."* On 2026-07-31 that exact flag was printed by a binary running
    full-CPU.
    """
    if not isinstance(startup_log, str):
        raise WhisperAdapterError("startup_log must be a string")
    if expected_lane not in ("cpu", "gpu"):
        raise WhisperAdapterError(f"expected_lane must be 'cpu' or 'gpu', got "
                                  f"{expected_lane!r}")
    device = _DEVICE_LINE_RE.search(startup_log)
    requested = bool(_REQUEST_FLAG_RE.search(startup_log))

    if not startup_log.strip():
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("startup log is empty; no device evidence was captured",))
    if expected_lane == "gpu":
        if device is None:
            reasons = ["no `Device N: <name>` line in the startup log, so nothing "
                       "establishes which backend actually loaded"]
            if requested:
                reasons.append("the log carries `use gpu = 1`, which reports what was "
                               "REQUESTED, never what was LOADED — this is the exact "
                               "signature of the 2026-07-31 silent CPU fallback")
            return schemas.Check(schemas.FAIL, tuple(reasons))
        return schemas.Check(schemas.PASS)
    # cpu lane
    if device is not None:
        return schemas.Check(schemas.FAIL, (
            f"a CPU cell's log names a device ({device.group('name').strip()!r}); the "
            f"measured footprint is not the declared one",))
    if requested:
        return schemas.Check(schemas.FAIL, (
            "a CPU cell's log carries `use gpu = 1`; the request contradicts the "
            "declared lane",))
    return schemas.Check(schemas.PASS)


# =============================================================================
# Phases, metrics, and resource lane
# =============================================================================

#: `schemas.PHASES_BY_BACKEND` has no entry for the speech backends — their phase
#: vocabulary was explicitly "to be defined" (§13.3) and is defined here. The names
#: are the engine's own pipeline stages, not invented labels.
PHASES = ("encode", "decode", "end_to_end")

#: Metric name -> direction. Every metric carries its direction because a bare
#: speech number is unusable (`MEASUREMENT.md:39-41`), and because this project
#: currently carries the same TTS measurement in two reciprocal conventions.
METRIC_DIRECTIONS = {
    "corpus_wer_pct": "lower_better",
    "rtf": "lower_better",
    "xrt": "higher_better",
    "latency_s": "lower_better",
    "throughput_audio_s_per_wall_s": "higher_better",
    "rss_slope_mib_per_cycle": "lower_better",
}

#: Names that LOOK like a metric and carry no direction. Offered so the refusal can
#: name the ambiguity instead of raising a bare KeyError.
_AMBIGUOUS_METRIC_NAMES = frozenset({
    "real_time_factor", "realtime_factor", "rt_factor", "speed", "wer",
})


def check_phase(phase: str) -> str:
    _require_str(phase, "phase")
    if phase not in PHASES:
        raise UnknownPhase(f"{phase!r} is not a {BACKEND} phase; declared phases are "
                           f"{list(PHASES)}")
    return phase


def metric_direction(metric: str) -> str:
    """Return the declared direction, or refuse.

    `real_time_factor` is refused BY NAME: the project carries `rtf: 0.169`
    (wall/audio, lower-better) in the ratified speech-freeze receipt and
    `xRT 5.47x` (audio/wall, higher-better) in the owning handoff for the same
    engine. They are reciprocals, and a name that does not say which is not a
    metric.
    """
    _require_str(metric, "metric")
    if metric in _AMBIGUOUS_METRIC_NAMES:
        raise UnknownMetric(
            f"{metric!r} names no direction and no denominator. Use `rtf` "
            f"(wall_s/audio_s, lower-better) or `xrt` (audio_s/wall_s, higher-better); "
            f"they are reciprocals and this project carries both for one engine")
    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise UnknownMetric(f"{metric!r} is not a declared {BACKEND} metric; declared "
                            f"metrics are {sorted(METRIC_DIRECTIONS)}") from exc


def check_metric_commensurable(metric: str) -> schemas.Check:
    """This backend reports no `task_rate`. Delegates to the schema-level rule."""
    metric_direction(metric)
    return schemas.check_metric_commensurability(BACKEND, {"metric": metric})


def resource_lane(*, device: Optional[str]) -> str:
    """`gpu` for an MI210 cell, `cpu` otherwise. Never `stack` (§11.6 is not our lane)."""
    if device is None:
        return "cpu"
    _require_str(device, "device")
    return "gpu"


# =============================================================================
# Domain ownership (critic / selection reject unowned domains)
# =============================================================================

#: Source subtrees this adapter owns, relative to the tree root. A proposal touching
#: anything else is rejected before it consumes a window.
OWNED_DOMAINS = frozenset({"src", "include", "ggml", "examples", "tests", "cmake"})

#: `ggml/` is shared core for THIS tree: it reaches every op in the whisper binary.
#: It does not reach any other tree — whisper.cpp vendors its own copy — so this is
#: narrower than the llama case, where shared ggml core reaches two binaries.
SHARED_CORE_DOMAINS = frozenset({"ggml", "include"})


def owned_domains() -> frozenset:
    return OWNED_DOMAINS


def check_domains_owned(domains: Iterable[str]) -> schemas.Check:
    """FAIL when a proposal names a domain this adapter does not own."""
    names = list(domains)
    if not names:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("no domains declared; scope is unknown, which is not the "
                              "same as empty",))
    unowned = sorted({d for d in names if _require_str(d, "domain") not in OWNED_DOMAINS})
    if unowned:
        return schemas.Check(schemas.FAIL, (
            f"domains {unowned} are not owned by the {BACKEND} adapter; owned domains "
            f"are {sorted(OWNED_DOMAINS)}",))
    return schemas.Check(schemas.PASS)


def touches_shared_core(domains: Iterable[str]) -> bool:
    """DECLARED shared-core reach. See `shared_core_paths()` for the traced answer."""
    return any(d in SHARED_CORE_DOMAINS for d in domains)


def _top_domain(path: str) -> str:
    parts = PurePosixPath(_require_str(path, "diff path")).parts
    return parts[0] if parts else ""


def diff_domains(diff: integrity.SourceDiff) -> tuple:
    """The domains a diff ACTUALLY reaches, read off its own file paths."""
    return tuple(sorted({_top_domain(p) for p in diff.paths()}))


def shared_core_paths(diff: integrity.SourceDiff) -> tuple:
    """Files IN THE DIFF that live in this tree's shared core.

    The traced counterpart of `touches_shared_core()`. It exists because a proposal's
    domain list is something the proposal SAYS, and §10.6's review marking must not
    be a function of what a candidate says about itself when the diff is in hand
    (invariant 18: declared equals traced). `run_source_integrity_gates` already
    derives the same flag from `risk_tier.matched_core_paths`; this is that discipline
    at the adapter seam.
    """
    return tuple(sorted(p for p in diff.paths() if _top_domain(p) in SHARED_CORE_DOMAINS))


def check_declared_domains_cover_diff(diff: integrity.SourceDiff,
                                      domains: Iterable[str]) -> schemas.Check:
    """FAIL when the diff reaches a domain the proposal did not declare.

    An under-declared domain list is not a paperwork slip: `assess_complexity` and
    the critic both read it, so a diff that touches `ggml/` while declaring `src`
    understates its own blast radius.
    """
    declared = {_require_str(d, "domain") for d in domains}
    if not declared:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no domains declared; scope is unknown, which is not the same as empty",))
    traced = set(diff_domains(diff))
    if not traced:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the supplied diff touches no file, so nothing can be traced against the "
            "declared domains",))
    undeclared = sorted(traced - declared)
    if undeclared:
        return schemas.Check(schemas.FAIL, (
            f"the diff reaches domains {undeclared} that the proposal did not declare "
            f"(declared {sorted(declared)}); declared must equal traced (invariant 18), "
            f"and an under-declared list understates the change's blast radius to every "
            f"reader that consumes it",))
    return schemas.Check(schemas.PASS)


# =============================================================================
# Complexity ceiling (§10.6) and change-class envelopes (§8.5.1)
# =============================================================================

#: DERIVATION, stated so the ceiling is not a number somebody liked.
#:
#: §10.6 makes the ceiling a MARKING threshold, not a rejection threshold: above it
#: the release package is marked `REQUIRES_HUMAN_CODE_REVIEW` and says so on its
#: first page. The right calibration is therefore *"larger than anything this project
#: has ever put on this branch"*, measured from the branch itself:
#:
#:     git log production-speech-v1 --not <upstream base> --numstat
#:
#: Measured 2026-08-03 on the frozen tree: exactly ONE commit beyond upstream —
#: `b3073792 "freeze: gfx90a/ROCm 6.2 GPU enablement for production speech"` —
#: touching 1 file (`ggml/src/ggml-cuda/vendors/hip.h`) with 1 insertion and 1
#: deletion, i.e. **2 changed lines**.
#:
#: The ceiling is set AT that observed maximum. The consequence is deliberate and is
#: not softened: essentially every LLM-authored change to this tree will be marked
#: `REQUIRES_HUMAN_CODE_REVIEW`. whisper.cpp is a third-party tree this project does
#: not own and whose upstream it does not control, and inflating the ceiling to make
#: the loop convenient would be a downgrade dressed as a calibration. Recomputed at
#: every freeze, since the historical maximum moves when a larger reviewed change
#: lands.
CEILING_DERIVATION = (
    "max(changed_lines) and max(files_touched) over every commit on "
    "production-speech-v1 beyond its upstream base, measured 2026-08-03: one commit "
    "(b3073792), 1 file, 2 changed lines"
)
_OBSERVED_MAX_CHANGED_LINES = 2
_OBSERVED_MAX_FILES_TOUCHED = 1


def complexity_ceiling() -> integrity.ComplexityCeiling:
    """§10.6, declared by this adapter and derived per `CEILING_DERIVATION`."""
    return integrity.ComplexityCeiling(
        backend=BACKEND,
        max_diff_lines=_OBSERVED_MAX_CHANGED_LINES,
        max_files_touched=_OBSERVED_MAX_FILES_TOUCHED,
        # Every accepted production change to this tree so far is inside
        # `ggml/src/ggml-cuda/`, i.e. shared core for this binary.
        shared_core_modification_requires_review=True,
        declared_by=f"autokernel.adapters.{BACKEND}/v1 ({CEILING_DERIVATION})",
    )


#: §8.5.1 (3) size envelopes. `max_file_shrinkage_ratio` is AutoPilot's >60 %
#: shrinkage reject ported to C++ — the defense that would have stopped
#: `escalation.py` going from 454 lines to 3. Exceeding an envelope is a conformance
#: FAILURE, which is distinct from exceeding the §10.6 ceiling (that marks, and does
#: not fail).
def change_class_envelopes() -> dict:
    declared_by = f"autokernel.adapters.{BACKEND}/v1"

    def env(change_class: str, files: int, lines: int, hunks: int, *,
            creation: bool = False, deletion: bool = False,
            pure_deletion: bool = False) -> integrity.ChangeClassEnvelope:
        return integrity.ChangeClassEnvelope(
            change_class=change_class, max_files_touched=files, max_changed_lines=lines,
            max_hunks=hunks, max_file_shrinkage_ratio=0.60,
            allows_file_creation=creation, allows_file_deletion=deletion,
            allows_pure_deletion_hunks=pure_deletion, declared_by=declared_by)

    return {
        "parameter": env("parameter", 2, 20, 4),
        "dispatcher": env("dispatcher", 4, 200, 12),
        "arithmetic": env("arithmetic", 3, 300, 16),
        "layout": env("layout", 6, 600, 30, creation=True),
        "fusion": env("fusion", 5, 500, 24, creation=True),
        "oracle_port": env("oracle_port", 8, 900, 40, creation=True),
        "core_header": env("core_header", 3, 150, 10),
    }


def assess_complexity(diff: integrity.SourceDiff, *, change_class: str,
                      domains: Iterable[str]) -> integrity.ComplexityAssessment:
    """§10.6 marking, with `touches_shared_core` TRACED from the diff, then OR-ed
    with the declared domains.

    The declared list alone would make the marking a function of what the candidate
    says about itself: the frozen tree's own production change is 1 file and 2 lines
    inside `ggml/src/ggml-cuda/`, i.e. inside both size ceilings, so declaring
    `domains=("src",)` for it removed every reason to mark it and the package came
    out `requires_human_code_review: false` on a shared-core edit. The diff is in
    hand, so the traced answer is the one that must be used; the declared list can
    only ever ADD a reason, never subtract one.
    """
    traced = shared_core_paths(diff)
    return integrity.assess_complexity_ceiling(
        diff, complexity_ceiling(),
        touches_shared_core=bool(traced) or touches_shared_core(domains),
        change_class=change_class)


# =============================================================================
# The backend-unchanged test (§3.2), in its single-backend form
# =============================================================================


@dataclass(frozen=True)
class UnchangedTestPlan:
    """What §3.2's two stages mean for a tree that serves exactly one backend."""

    backend: str
    stage1_closure_source: str
    traverse_submodules: tuple
    stage2_required: bool
    transfer_available: bool
    rationale: str

    def to_dict(self) -> dict:
        return {"backend": self.backend,
                "stage1_closure_source": self.stage1_closure_source,
                "traverse_submodules": list(self.traverse_submodules),
                "stage2_required": self.stage2_required,
                "transfer_available": self.transfer_available,
                "rationale": self.rationale}


def unchanged_test_plan() -> UnchangedTestPlan:
    return UnchangedTestPlan(
        backend=BACKEND,
        stage1_closure_source="build_system_depfiles",
        traverse_submodules=SUBMODULE_PATHS,
        stage2_required=True,
        # There is no second backend whose cells could be dropped.
        transfer_available=False,
        rationale=(
            "whisper.cpp serves exactly one backend, so §3.2's cell-dropping transfer "
            "has no counterpart: there is no other backend to drop. Both stages still "
            "run, for the opposite purpose — to establish that the candidate binary "
            "differs from the incumbent AT ALL, so a no-op candidate is refused rather "
            "than passing every gate trivially."),
    )


def classify_unchanged_result(*, stage1_closure_empty: bool,
                              stage2_normalized_identical: Optional[bool]
                              ) -> schemas.Check:
    """PASS = the candidate genuinely differs. A no-op candidate FAILs.

    A disagreement between the stages is a HARD FINDING, never a silent preference
    for the cheaper answer (§3.2): either the closure is wrong or the build is
    non-deterministic, and both are defects in the build-identity machinery.
    """
    if not isinstance(stage1_closure_empty, bool):
        raise WhisperAdapterError("stage1_closure_empty must be a bool")
    if stage2_normalized_identical is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "stage 2 (normalized binary comparison against an anchor rebuild in the "
            "candidate's environment) was not run; stage 1 alone may not drop or admit "
            "a cell (§3.2)",))
    if not isinstance(stage2_normalized_identical, bool):
        raise WhisperAdapterError("stage2_normalized_identical must be a bool or None")

    if stage1_closure_empty != stage2_normalized_identical:
        return schemas.Check(schemas.FAIL, (
            f"stage 1 reports closure_empty={stage1_closure_empty} and stage 2 reports "
            f"normalized_identical={stage2_normalized_identical}. §3.2: a disagreement is "
            f"a hard finding filed against the build-identity machinery — the closure is "
            f"wrong or the build is non-deterministic — never a silent preference for the "
            f"cheaper answer",))
    if stage1_closure_empty:
        return schemas.Check(schemas.FAIL, (
            "the candidate's source closure is empty and its normalized binary is "
            "identical to the incumbent's: this is a NO-OP candidate and is refused "
            "before it consumes a release matrix",))
    return schemas.Check(schemas.PASS)


# =============================================================================
# P-STT-1 correctness contract — the normalizer, the taxonomy, the derivations
# =============================================================================

#: The ordered normalization pipeline of `P-STT-1` §1.3. Declared here so the
#: evaluator's normalizer can be checked against the contract rather than trusted,
#: and so a reordering is visible as a diff. The order is normative: several steps
#: are not commutative.
NORMALIZATION_STEPS = (
    "nfkc",
    "casefold",
    "remove_enumerated_nonlexical_markers",
    "punctuation_to_separator",
    "preserve_apostrophes",
    "hyphen_split",
    "numerals_hypothesis_to_reference_form",
    "collapse_whitespace",
)

#: Transforms that would each convert a genuine recognition error into a match.
#: Enumerated so a later "improvement" is visibly out of contract rather than a
#: plausible convenience.
FORBIDDEN_NORMALIZATION_TRANSFORMS = frozenset({
    "stemming", "lemmatization", "stopword_removal", "synonym_mapping",
    "homophone_mapping", "spell_correction", "fuzzy_token_match", "truncation",
})

#: Whisper emits event tags where other engines emit nothing. The list is CLOSED: a
#: general "delete anything in brackets" rule would silently truncate a reference
#: containing a genuine bracket, and would reward a candidate that learned to wrap
#: its errors in brackets.
NONLEXICAL_MARKERS = (
    "[BLANK_AUDIO]", "[MUSIC]", "[SOUND]", "[NOISE]", "[LAUGHTER]",
    "(silence)", "[ Silence ]", "[SILENCE]", "*",
)

#: §1.5 taxonomy. `ok` scores; two classes are excluded from the denominator and
#: counted; three are categorical correctness failures that MUST NOT be averaged
#: into a rate.
FAILURE_CLASSES = ("ok", "empty", "repetition_loop", "numeral_uncovered",
                   "unknown_marker", "decode_error")
SCORING_CLASSES = ("ok",)
EXCLUDED_CLASSES = ("numeral_uncovered", "unknown_marker")
FAILING_CLASSES = ("empty", "repetition_loop", "decode_error")


def check_normalizer_contract(*, steps: Sequence[str],
                              transforms_used: Collection[str]) -> schemas.Check:
    """FAIL when the evaluator's normalizer is not the contracted one.

    Order is checked, not just membership: NFKC before casefold, punctuation-to-
    separator before numeral conversion, whitespace collapse last.
    """
    declared = [_require_str(s, "step") for s in steps]
    if tuple(declared) != NORMALIZATION_STEPS:
        return schemas.Check(schemas.FAIL, (
            f"normalizer step sequence {declared} is not the P-STT-1 §1.3 pipeline "
            f"{list(NORMALIZATION_STEPS)}; the order is normative because several steps "
            f"do not commute",))
    used = sorted({_require_str(t, "transform") for t in transforms_used})
    forbidden = [t for t in used if t in FORBIDDEN_NORMALIZATION_TRANSFORMS]
    if forbidden:
        return schemas.Check(schemas.FAIL, (
            f"forbidden transforms present: {forbidden}. Each can convert a genuine "
            f"recognition error into a match; a normalizer performing one is a different "
            f"instrument and requires a new protocol id",))
    return schemas.Check(schemas.PASS)


def check_normalizer_properties(*, symmetric: Optional[bool],
                                idempotent: Optional[bool],
                                deterministic: Optional[bool]) -> schemas.Check:
    """The three in-run assertions of `P-STT-1` §1.3. `None` is COULD_NOT_CHECK."""
    unmeasured = [name for name, value in
                  (("symmetric", symmetric), ("idempotent", idempotent),
                   ("deterministic", deterministic)) if value is None]
    if unmeasured:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"normalizer properties not asserted in-run: {unmeasured}; an unasserted "
            f"property is not a satisfied one",))
    failed = [name for name, value in
              (("symmetric", symmetric), ("idempotent", idempotent),
               ("deterministic", deterministic)) if value is False]
    if failed:
        return schemas.Check(schemas.FAIL, (
            f"normalizer properties violated: {failed}. A non-symmetric normalizer "
            f"manufactures errors; a non-idempotent one makes the score depend on how "
            f"many times it ran",))
    return schemas.Check(schemas.PASS)


#: The audio-input identity fields of `P-STT-1` §1.2. `pcm_sha256` is the SHA-256 of
#: the DECODED samples, never of the container file: the same audio as FLAC and as
#: WAV has two file hashes and one content, and a resampler change alters what was
#: measured while leaving the file untouched.
AUDIO_IDENTITY_FIELDS = ("utterance_id", "pcm_sha256", "sample_rate_hz", "channels",
                         "sample_format", "sample_count")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def check_audio_identity(record: Mapping[str, Any]) -> schemas.Check:
    """Every field present and well-formed, or the cell has no valid denominator."""
    if not isinstance(record, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("audio identity record is not a mapping",))
    missing = [f for f in AUDIO_IDENTITY_FIELDS if f not in record]
    if missing:
        return schemas.Check(schemas.FAIL, (
            f"audio identity is missing {missing}; without it nothing below layer 0 "
            f"means anything (P-STT-1 §1.2)",))
    digest = record.get("pcm_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.match(digest):
        return schemas.Check(schemas.FAIL,
                             (f"pcm_sha256 is not a lowercase sha256: {digest!r}",))
    if schemas.is_placeholder_digest(digest):
        return schemas.Check(schemas.FAIL, (
            f"pcm_sha256 {digest!r} is a well-formed digest no measurement produced; a "
            f"fabricated hash is indistinguishable from a measured one to every "
            f"downstream reader",))
    for field in ("sample_rate_hz", "channels", "sample_count"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return schemas.Check(schemas.FAIL,
                                 (f"{field} must be a positive int, got {value!r}",))
    return schemas.Check(schemas.PASS)


def compare_corpus_identity(anchor: Mapping[str, str],
                            candidate: Mapping[str, str]) -> schemas.Check:
    """A corpus whose PCM hashes differ from the anchor run's VOIDS the window.

    It is journaled `INVALID` and is NEVER recorded as a candidate correctness
    failure — a different corpus says nothing whatever about the candidate, exactly
    as a drifted anchor does not (`kernel-research.md:302-305`).
    """
    if not isinstance(anchor, Mapping) or not isinstance(candidate, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("corpus manifests must be mappings of "
                              "utterance_id -> pcm_sha256",))
    if not anchor:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("anchor corpus manifest is empty",))
    missing = sorted(set(anchor) - set(candidate))
    extra = sorted(set(candidate) - set(anchor))
    differing = sorted(k for k in set(anchor) & set(candidate) if anchor[k] != candidate[k])
    if missing or extra or differing:
        # The counts lead and the samples are LABELLED as samples. A bare `missing[:8]`
        # renders 800 mismatches and 8 mismatches as the same-looking list, which is the
        # silent-truncation defect this module refuses everywhere else.
        return schemas.Check(schemas.FAIL, (
            f"corpus mismatch — {len(missing)} missing (first 8 of {len(missing)}: "
            f"{missing[:8]}), {len(extra)} extra (first 8 of {len(extra)}: {extra[:8]}), "
            f"{len(differing)} differing (first 8 of {len(differing)}: {differing[:8]}). "
            f"VOID the window and journal it INVALID; this is NOT a candidate "
            f"correctness failure",))
    return schemas.Check(schemas.PASS)


def pooled_corpus_wer(errors: Sequence[int], reference_tokens: Sequence[int]) -> float:
    """`Σ errors / Σ reference_tokens`, as a percentage. Lower-better.

    The POOLED estimator, which is the one `P-STT-1` §1.4 defines. The mean of
    per-utterance WERs is a DIFFERENT quantity — it weights a three-word utterance
    equally with a forty-word one and is dominated by short utterances, where one
    error is 33 %. This module deliberately offers no function that computes it, so
    the two cannot be confused at a call site.
    """
    errs = list(errors)
    refs = list(reference_tokens)
    if len(errs) != len(refs):
        raise WhisperAdapterError(
            f"errors and reference_tokens differ in length ({len(errs)} vs {len(refs)}); "
            f"the pooled ratio would silently drop utterances")
    if not errs:
        raise DerivationImpossible("no utterances; a corpus WER over nothing is not 0.0")
    for i, value in enumerate(errs):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WhisperAdapterError(f"errors[{i}] must be a non-negative int, got {value!r}")
    for i, value in enumerate(refs):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WhisperAdapterError(
                f"reference_tokens[{i}] must be a positive int, got {value!r}; an "
                f"utterance with no reference tokens has no defined error rate")
    return 100.0 * sum(errs) / sum(refs)


def derive_repetition_envelope(anchor_hyp_to_ref_ratios: Sequence[float]) -> float:
    """The repetition-loop detector's threshold, DERIVED from the anchor's own behaviour.

    `P-STT-1` §1.5: the envelope is the **maximum ratio of normalized hypothesis
    tokens to normalized reference tokens observed on the ANCHOR over the calibration
    corpus**. The detector therefore fires only outside the anchor's own observed
    behaviour, and its threshold is a measured property of the instrument under this
    host state rather than a number somebody chose.

    Raises rather than defaulting: a campaign with no anchor observations cannot
    derive this, and a defaulted envelope is one nobody measured
    (`kernel-research.md:263-268` — no value may be supplied as a literal).
    """
    ratios = [_require_positive_number(r, f"anchor_hyp_to_ref_ratios[{i}]")
              for i, r in enumerate(anchor_hyp_to_ref_ratios)]
    if not ratios:
        raise DerivationImpossible(
            "the repetition envelope is derived from the anchor's observed "
            "hypothesis/reference token ratios; with none observed it cannot be derived "
            "and MUST NOT be defaulted")
    return max(ratios)


def derive_corpus_size(*, observed_halfwidth_pp: float, observed_n: int,
                       contribution_floor_pp: float) -> int:
    """Smallest corpus size whose paired bootstrap half-width meets the campaign floor.

    `P-STT-1` §1.4: corpus size is DERIVED from the campaign's declared
    `contribution_floor` (`kernel-research.md:181-183`), never fixed by the protocol.
    Half-width scales as `n^(-1/2)`, so

        n_required = ceil(observed_n * (observed_halfwidth / floor)**2)

    Worked precedent, descriptive only: at `observed_n=100` the 2026-07-31 corpus
    gave a paired half-width of 0.67 pp, so a 0.30 pp floor needs ~500 utterances and
    a 0.10 pp floor needs ~4500 — the latter exceeding all of LibriSpeech test-clean
    (2620). A campaign confronts that at calibration time instead of discovering it
    after spending its budget.

    The result may be SMALLER than `observed_n` when the observed corpus already
    beats the floor; that is a true statement ("this many would have sufficed") and
    is returned rather than clamped, because clamping would hide the headroom.
    """
    halfwidth = _require_positive_number(observed_halfwidth_pp, "observed_halfwidth_pp")
    floor = _require_positive_number(contribution_floor_pp, "contribution_floor_pp")
    if isinstance(observed_n, bool) or not isinstance(observed_n, int) or observed_n <= 0:
        raise WhisperAdapterError(f"observed_n must be a positive int, got {observed_n!r}")
    ratio = halfwidth / floor
    required = observed_n * ratio * ratio
    ceiling = int(required)
    if ceiling < required:
        ceiling += 1
    return max(ceiling, 1)


def check_failure_taxonomy(counts: Mapping[str, int], *, n_utterances: int
                           ) -> schemas.Check:
    """Every utterance classified, and a categorical failure never averaged into a rate.

    The Qwen3-ASR precedent is why this is mandatory: 29.36 % corpus WER was **not** a
    scoring artifact but a degenerate repetition loop on 21 of 100 utterances carrying
    94.7 % of all errors, with the clean rows at 2.27 %. A rate that averages a
    repetition loop reports a uniformly mediocre model where the truth is an excellent
    model that occasionally fails catastrophically — different production risks, and a
    different release decision.
    """
    if not isinstance(counts, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK, ("taxonomy counts is not a mapping",))
    if isinstance(n_utterances, bool) or not isinstance(n_utterances, int) or n_utterances <= 0:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (f"n_utterances must be a positive int, got {n_utterances!r}",))
    unknown = sorted(set(counts) - set(FAILURE_CLASSES))
    if unknown:
        return schemas.Check(schemas.FAIL, (
            f"taxonomy carries classes outside the declared vocabulary: {unknown}; the "
            f"declared classes are {list(FAILURE_CLASSES)}",))
    missing = [c for c in FAILURE_CLASSES if c not in counts]
    if missing:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"taxonomy omits classes {missing}; an omitted class is not a zero count, "
            f"it is an unclassified utterance",))
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return schemas.Check(schemas.FAIL,
                                 (f"taxonomy[{name}] must be a non-negative int, got "
                                  f"{value!r}",))
    total = sum(counts.values())
    if total != n_utterances:
        return schemas.Check(schemas.FAIL, (
            f"taxonomy totals {total} over {n_utterances} utterances; every utterance "
            f"receives exactly one class",))
    failing = {c: counts[c] for c in FAILING_CLASSES if counts[c] > 0}
    if failing:
        return schemas.Check(schemas.FAIL, (
            f"categorical correctness failures present: {failing}. These are NOT WER "
            f"contributions; correctness is lexicographically prior to speed and such a "
            f"candidate receives no speed rank at all, not a penalised one "
            f"(kernel-research.md:355-360)",))
    return schemas.Check(schemas.PASS)


def check_exclusion_rate(*, candidate_excluded: int, anchor_excluded: int,
                         n_utterances: int, aa_dispersion: float) -> schemas.Check:
    """The numeral/marker exclusion cap, DERIVED from the anchor plus A/A dispersion.

    `P-STT-1` §1.3 step 6: the cap is the anchor's own exclusion rate on the same
    corpus plus the A/A dispersion of that rate. A candidate above it is emitting
    forms the anchor does not, and a scorer that quietly drops them would be hiding
    the very change under study.
    """
    for name, value in (("candidate_excluded", candidate_excluded),
                        ("anchor_excluded", anchor_excluded)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"{name} must be a non-negative int, got {value!r}",))
    if isinstance(n_utterances, bool) or not isinstance(n_utterances, int) or n_utterances <= 0:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (f"n_utterances must be a positive int, got {n_utterances!r}",))
    if isinstance(aa_dispersion, bool) or not isinstance(aa_dispersion, (int, float)):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("aa_dispersion must be a number derived from the A/A control",))
    if float(aa_dispersion) < 0.0:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("aa_dispersion must be non-negative",))
    cand_rate = candidate_excluded / n_utterances
    anchor_rate = anchor_excluded / n_utterances
    cap = anchor_rate + float(aa_dispersion)
    if cand_rate > cap:
        return schemas.Check(schemas.FAIL, (
            f"candidate exclusion rate {cand_rate:.4f} exceeds the derived cap {cap:.4f} "
            f"(anchor {anchor_rate:.4f} + A/A dispersion {float(aa_dispersion):.4f}); the "
            f"candidate is emitting forms the anchor does not and the scorer would be "
            f"dropping the change under study",))
    return schemas.Check(schemas.PASS)


def derive_correctness_margin(*, aa_noise_floor: float, contribution_floor: float,
                              determinism_class: str) -> float:
    """`margin = max(φ_corr, contribution_floor)`, and exactly 0 when bitwise stable.

    `P-STT-REL-1` §4.3: when the determinism class is `bitwise_stable`, `φ_corr` is
    exactly 0 and the rule collapses to transcript identity — the strong form, which
    MUST be used whenever it is available. A tolerance is never wider than the
    instrument's own measured noise.
    """
    if determinism_class not in schemas.DETERMINISM_CLASSES:
        raise WhisperAdapterError(
            f"determinism_class {determinism_class!r} is not one of "
            f"{sorted(schemas.DETERMINISM_CLASSES)}")
    for name, value in (("aa_noise_floor", aa_noise_floor),
                        ("contribution_floor", contribution_floor)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WhisperAdapterError(f"{name} must be a number, got {type(value).__name__}")
        if float(value) < 0.0:
            raise WhisperAdapterError(f"{name} must be non-negative, got {value!r}")
    if determinism_class == "bitwise_stable":
        if float(aa_noise_floor) != 0.0:
            raise WhisperAdapterError(
                f"determinism_class is 'bitwise_stable' but the A/A noise floor is "
                f"{aa_noise_floor!r}; a bitwise-stable instrument has a zero floor by "
                f"construction, so one of the two measurements is wrong and this is a "
                f"hard finding, not a margin to widen")
        return 0.0
    if determinism_class == "not_measured":
        raise DerivationImpossible(
            "determinism class is 'not_measured', so no correctness margin can be "
            "derived: an unmeasured stability is not a stable one (invariant 12)")
    return max(float(aa_noise_floor), float(contribution_floor))


# =============================================================================
# Protocol bindings — and the honest refusal while the family is a draft
# =============================================================================

#: Ratified today: search on this backend is legal, because `P-AK-SEARCH-1`'s scope
#: is *"Tiers T0, T1 and T2 … on every declared backend adapter"*.
SEARCH_PROTOCOL_ID = "P-AK-SEARCH-1"

#: NOT ratified: the STT family is a draft under
#: `artifacts/operator/autokernel-policy-draft/`. Until it is ratified, a whisper.cpp
#: number has NO owning protocol and cannot become a claim
#: (`kernel-research.md:54-56`).
RELEASE_PROTOCOL_IDS = ("P-STT-1", "P-STT-2", "P-STT-3", "P-STT-REL-1")
RELEASE_PROTOCOL_DRAFT_LOCATOR = (
    "artifacts/operator/autokernel-policy-draft/P-STT-1.draft.md"
)


def release_gate_readiness(ratified_protocol_ids: Collection[str]) -> schemas.Check:
    """Is this backend's release path legally runnable yet?

    `ratified_protocol_ids` is SUPPLIED, never baked in: the source of truth is the
    protocol registry in `MEASUREMENT.md` §2, and a constant here would go stale
    silently the moment the operator ratified (or declined) the family.

    Returns COULD_NOT_CHECK — **never PASS** — while any required protocol is
    missing. This is P-AK-SEARCH-1 denial 6 in its adapter form: *"a controller that
    discovers a coverage gap in its evaluator RECORDS the gap, blocks release
    eligibility for the affected lineage, continues unrelated research, and MAY draft
    an amendment for human review. It does not patch the instrument, and it does not
    route around it."*
    """
    ratified = {_require_str(p, "ratified_protocol_id") for p in ratified_protocol_ids}
    missing_search = SEARCH_PROTOCOL_ID not in ratified
    missing_release = [p for p in RELEASE_PROTOCOL_IDS if p not in ratified]
    if missing_search:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{SEARCH_PROTOCOL_ID} is not in the supplied ratified set, so not even T0-T2 "
            f"search is authorized on this backend",))
    if missing_release:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the {BACKEND} release protocols {missing_release} are not ratified; they "
            f"are a DRAFT at {RELEASE_PROTOCOL_DRAFT_LOCATOR}. Search under "
            f"{SEARCH_PROTOCOL_ID} remains legal and candidates may be banked; release "
            f"eligibility is BLOCKED for this lineage until the operator ratifies or "
            f"declines the family",))
    return schemas.Check(schemas.PASS)


# =============================================================================
# Self-audit — the no-write / no-process guarantee, proved from this module's AST
# =============================================================================


#: Module-level names the audited source MUST define for the result to be ABOUT this
#: module. Without a binding of this kind the audit is a property of whatever string
#: it was handed, and the empty string satisfies it perfectly.
_AUDIT_IDENTITY_FUNCTIONS = (
    "check_not_production_path", "interpret_linkage_report", "release_gate_readiness",
)


def _source_is_this_module(tree: Any) -> bool:
    """True when the parsed AST is recognisably THIS adapter's source."""
    backend = None
    defined = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == "BACKEND"
                        and isinstance(node.value, ast.Constant)):
                    backend = node.value.value
    return backend == BACKEND and set(_AUDIT_IDENTITY_FUNCTIONS) <= defined


def audit_no_write_or_process_paths(source: Optional[str] = None) -> schemas.Check:
    """Delegate to the evaluator's AST auditor, on THIS module's source.

    Reusing `api.audit_no_write_or_process_paths` rather than reimplementing it keeps
    one definition of "cannot write, cannot signal" for the whole package. Note it
    takes the source text, so the caller supplies it — this module reads no file.

    The supplied text is BOUND to this module before a non-FAIL result is returned.
    The evaluator's own auditor anchors itself with `Path(__file__).read_text()`; this
    one cannot (it reads no file), so without a binding `audit_no_write_or_process_paths("")`
    returns PASS — the guarantee obtained by deleting the thing it inspects. A FAIL is
    returned unbound, because a forbidden construct is a finding about the text
    whoever the text belongs to.
    """
    if source is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no source supplied; this module reads no file, so the caller passes the "
            "module text (test_whisper_stt.py does)",))
    if not isinstance(source, str):
        raise WhisperAdapterError("source must be a string")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (f"could not parse module: {exc}",))
    result = api.audit_no_write_or_process_paths(source)
    if result.outcome == schemas.FAIL:
        return result
    if not _source_is_this_module(tree):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the supplied source does not define this module's identity (BACKEND = "
            f"{BACKEND!r} plus {list(_AUDIT_IDENTITY_FUNCTIONS)}), so the AST audited is "
            f"not this adapter's. A clean audit of text nobody bound to the module — the "
            f"empty string passes every rule — is not evidence about the module",))
    return result
