#!/usr/bin/env python3
"""qwentts_tts.py — the `qwentts_tts` backend adapter (§13.4, phase AK9).

WHY THIS MODULE EXISTS
----------------------
`qwentts.cpp` is the third of three source trees this project freezes, and like its
STT sibling it had **no adapter and no measurement protocol of any kind** until AK9
(AK-D24). It is also the tree that violates every convenient assumption about the
other three, which is why §1.5 ends with *"Adapters must not assume uniformity."*

**Three asymmetries, declared here so nothing has to guess them:**

1. **The stable path points at `build`, not `build/bin`.** `cpu`, `gpu` and `stt` all
   resolve into a `bin/` subdirectory; `/mnt/raid0/llm/kernels/production/tts` →
   `/mnt/raid0/llm/qwentts.cpp/build`. A path constructor that appends `bin/` for
   every backend produces a non-existent path here (§1.5, §13.4).
2. **`ggml` is a git SUBMODULE, not vendored in-tree.** `whisper.cpp` edits
   `ggml/src/ggml-cuda/vendors/hip.h` as an ordinary file; this tree carries a
   gitlink. The frozen production commit `2c1b5182` shows **`ggml | 2 +-` — one file,
   one insertion, one deletion** in the superproject, while the submodule commit it
   points at (`b86f6602`) changes **4 files and 115 lines**. A source-closure diff
   (§3.2 stage 1) or a §10.6 complexity assessment computed on the superproject alone
   under-reports this change by two orders of magnitude.
3. **It runs ggml 0.17.0**, between llama.cpp's 0.16.0 and whisper.cpp's 0.18.0.

WHICH FAILURE IT PREVENTS
-------------------------
Two, and both have already happened on this host.

**A binary that silently runs against another tree's ggml.** Three ggml generations
coexist and the loader honours `LD_LIBRARY_PATH` before a binary's own directory
(INC-20260731-ggml-linkage-silent-cpu-fallback). Every candidate build and every T3
phase-2 check goes through the research repo's `scripts/utils/verify_ggml_linkage.sh`
(§10.2 phase 2), and this module owns the three clauses the raw script does not
enforce: a `PASS` is necessary and **not sufficient** (ggml backends are `dlopen`ed
and `ldd` cannot see them, so the engine's own device line is also required, and
`use gpu = 1` reports what was *requested*); a run that resolved **no** libraries is
`COULD_NOT_CHECK`, never `PASS` (the script prints its "statically linked, or ldd
failed" marker and then exits 0); and the script's name filter is
`libggml*|libwhisper*|libllama*|libmtmd*`, so any qwentts-specific shared object is
**not examined at all** and its absence from the report is silence, not evidence.

**A test suite that got smaller while its pass rate stayed at 100 %.** The gfx90a
`ARGSORT` defect — `ne0=2048` launching 2048 threads per block against gfx90a's
1024-thread cap, **705 times per utterance** — was invisible while `test-backend-ops`
reported `ARGSORT 46/46` and `TOP_K 170/170`, because the failing shapes were
**silently skipped**. After the fix the same suite reported `74/74` and `292/292`.
Both readings are "100 % pass" and only the *enumeration* distinguishes them, so
`check_op_coverage()` fails a candidate whose attempted-case count falls below the
anchor's at any pass rate.

WHAT THIS MODULE IS NOT
-----------------------
**It executes nothing.** It declares facts, constructs the argv a runner must
execute, and interprets output a runner captured. It runs no inference, no synthesis,
no benchmark and no build; it starts, stops and signals no process; it writes no
file; and it reads no file. `audit_no_write_or_process_paths()` proves the
write/process half from this module's own AST.

It **freezes nothing**. `qwentts.cpp` is independently freezable (§1.5), but a
freeze, a cutover, an era-registry row, an AutoPilot baseline apply and a repoint of
`/mnt/raid0/llm/kernels/production/tts` are human-only writes
(`MEASUREMENT.md:140-142`, invariant 5). `release_gate_readiness()` returns
`COULD_NOT_CHECK` whenever the caller's ratified registry omits an Annex S prerequisite.

GOVERNING INSTRUMENTS
---------------------
  * `measurement/protocols/kernel-research.md` — **P-AK-SEARCH-1** (RATIFIED
    2026-08-03). Search on this backend is already authorized; **release is not**.
  * `measurement/protocols/speech.md` — ratified Annex S, including the TTS family
    (`P-TTS-1`, `P-TTS-2`, `P-TTS-3`, `P-TTS-REL-1`).
  * `artifacts/operator/ratify_speech_kernel_freeze_20260731.json` — the operator
    receipt that froze this tree.

Design context: §1.5, §3.2, §10.2, §10.6, §11, §13.4.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence

from .. import schemas
from ..evaluator import api, devices, integrity
from ..release import plan as release_plan

# =============================================================================
# Errors — every one is a refusal, never a degraded answer
# =============================================================================


class QwenTtsAdapterError(Exception):
    """Base for every refusal this adapter makes."""


class ProductionPathRefused(QwenTtsAdapterError):
    """A path resolves inside a frozen production tree (invariant 3, denial 2)."""


class UnknownBinary(QwenTtsAdapterError):
    """A binary name that is not in this backend's declared inventory."""


class UnknownPhase(QwenTtsAdapterError):
    """A phase name outside this backend's declared vocabulary."""


class UnknownMetric(QwenTtsAdapterError):
    """A metric name with no declared direction. A bare metric is unusable."""


class WrongReleasePath(QwenTtsAdapterError):
    """A release path this backend must refuse rather than degrade to."""


class DerivationImpossible(QwenTtsAdapterError):
    """A derived quantity has no inputs. It is refused, never defaulted."""


class SubmoduleClosureMissing(QwenTtsAdapterError):
    """A source closure that did not traverse `ggml/` is not a source closure here."""


# =============================================================================
# Tree identity (§1.5, speech-freeze receipt 2026-07-31)
# =============================================================================

BACKEND = "qwentts_tts"
SOURCE_TREE = "qwentts.cpp"
MODULE_ID = "autokernel.adapters.qwentts_tts/v1"

PRODUCTION_TREE_ROOT = "/mnt/raid0/llm/qwentts.cpp"

#: Mirror of `storage.PRODUCTION_TREES` / `correctness.PRODUCTION_TREE_ROOTS`;
#: `test_qwentts_tts.py` asserts they agree, so the duplication is checked.
PRODUCTION_TREE_ROOTS = (
    "/mnt/raid0/llm/llama.cpp",
    "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp",
    "/workspace/repos/epyc-llama",
)

#: Live ALIASES for the frozen trees, which `PRODUCTION_TREE_ROOTS` cannot express.
#: `/mnt/raid0/llm/kernels/production/<backend>` is a SYMLINK into a frozen build
#: directory — `kernels/README.md` calls it *"the only path anything should name"* —
#: and `archive/<backend>-<date>-<sha>` is the same device pointed at a superseded
#: target. A path reaching the frozen tree through either one is inside production
#: while comparing unequal to every lexical root.
PRODUCTION_PATH_ALIASES = (
    "/mnt/raid0/llm/kernels/production",
    "/mnt/raid0/llm/kernels/archive",
)

FROZEN_BRANCH = "production-speech-v1"
FROZEN_COMMIT = "2c1b5182e7e9f1acaa04405ff21747d8a7acf4d5"

#: The submodule commit the frozen superproject commit points at. It is candidate
#: identity in its own right: the superproject commit alone does not determine the
#: source, because the gitlink can be repointed without the superproject diff
#: showing more than one changed line.
FROZEN_GGML_SUBMODULE_COMMIT = "b86f660238dcc1a83b7cbf5a72d355a965de9245"

GGML_GENERATION = "0.17.0"

#: Asymmetry 2. The sibling STT adapter declares `in_tree`; a shared assumption
#: would be wrong for one of them.
GGML_VENDORING = "submodule"
SUBMODULE_PATHS = ("ggml",)

#: Asymmetry 1: binaries AND libraries live directly in `build`, with no `bin/`.
BUILD_DIR_REL = "build"
LIBRARY_DIR_REL = "build"

STABLE_PATH = "/mnt/raid0/llm/kernels/production/tts"
STABLE_TARGET = "/mnt/raid0/llm/qwentts.cpp/build"

#: The three other production binaries DO resolve through a `bin/` subdirectory.
#: Kept as data so `check_stable_path_assumption()` can name the difference instead
#: of a caller rediscovering it at cutover.
SIBLING_STABLE_TARGETS = {
    "cpu": "/mnt/raid0/llm/llama.cpp/build/bin",
    "gpu": "/mnt/raid0/llm/llama.cpp/build-hip/bin",
    "stt": "/mnt/raid0/llm/whisper.cpp/build/bin",
}


@dataclass(frozen=True)
class TreeFacts:
    """Everything the controller needs to know about this backend's source tree."""

    backend: str
    source_tree: str
    production_tree_root: str
    frozen_branch: str
    frozen_commit: str
    frozen_ggml_submodule_commit: str
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
            "frozen_ggml_submodule_commit": self.frozen_ggml_submodule_commit,
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
        frozen_ggml_submodule_commit=FROZEN_GGML_SUBMODULE_COMMIT,
        ggml_generation=GGML_GENERATION,
        ggml_vendoring=GGML_VENDORING,
        submodule_paths=SUBMODULE_PATHS,
        build_dir_rel=BUILD_DIR_REL,
        library_dir_rel=LIBRARY_DIR_REL,
        stable_path=STABLE_PATH,
        stable_target=STABLE_TARGET,
    )


def check_stable_path_assumption(constructed_target: str) -> schemas.Check:
    """FAIL a transaction dry-run that appended `bin/` to this backend's install path.

    §10.2 phase 8 is where a release transaction states its exact install path,
    archive link and symlink diff. This is the check that catches asymmetry 1 there,
    rather than at cutover.
    """
    _require_str(constructed_target, "constructed_target")
    if constructed_target == STABLE_TARGET:
        return schemas.Check(schemas.PASS)
    if constructed_target.rstrip("/") == STABLE_TARGET.rstrip("/") + "/bin":
        return schemas.Check(schemas.FAIL, (
            f"the transaction constructed {constructed_target!r} by appending 'bin/'. "
            f"{BACKEND} is the exception: {STABLE_PATH} points at {STABLE_TARGET!r}, "
            f"while cpu/gpu/stt point into a bin/ subdirectory "
            f"({sorted(SIBLING_STABLE_TARGETS.values())}). §1.5: adapters must not "
            f"assume uniformity",))
    return schemas.Check(schemas.FAIL, (
        f"constructed install target {constructed_target!r} is not this backend's "
        f"stable target {STABLE_TARGET!r}",))


# =============================================================================
# Freeze scope (§1.5, AK-D11)
# =============================================================================


@dataclass(frozen=True)
class FreezeScope:
    """Which backends a freeze of this tree necessarily covers.

    `qwentts.cpp` serves exactly one backend, so it is **independently freezable** —
    unlike `llama_cpu`/`llama_gpu`, which share one tree and one frozen branch and
    cannot be frozen apart.
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
    """Raise if asked to join a champion lineage belonging to another tree (AK-D11)."""
    _require_str(champion_source_tree, "champion_source_tree")
    if champion_source_tree != SOURCE_TREE:
        raise WrongReleasePath(
            f"{BACKEND} candidates belong to the {SOURCE_TREE!r} champion; "
            f"{champion_source_tree!r} is a different source tree and a different "
            f"freeze (§1.5, AK-D11)")


def refuse_stack_change_path() -> None:
    """Raise: this backend releases through a kernel freeze, not §11.6."""
    raise WrongReleasePath(
        f"{BACKEND} releases through the kernel-freeze path (§10, §11), not the "
        f"three-gate stack-change path (§11.6), which is the `serving_runtime` lane")


# =============================================================================
# Small validators — local on purpose
# =============================================================================

_ABS_PATH_RE = re.compile(r"^/[^\x00]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QwenTtsAdapterError(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _require_abs_path(value: Any, label: str) -> str:
    _require_str(value, label)
    if not _ABS_PATH_RE.match(value):
        raise QwenTtsAdapterError(f"{label}: expected an absolute POSIX path, got {value!r}")
    if ".." in PurePosixPath(value).parts:
        raise QwenTtsAdapterError(f"{label}: contains '..'; refusing to normalise a path "
                                  f"whose target depends on the filesystem: {value!r}")
    if PurePosixPath(value).parts[:1] == ("//",):
        # POSIX leaves a leading `//` implementation-defined and `PurePosixPath` keeps
        # it as a distinct root segment, so `//mnt/raid0/llm/qwentts.cpp/x` compares
        # UNEQUAL to `/mnt/raid0/llm/qwentts.cpp` segment-by-segment while Linux opens
        # the identical file — a one-character walk through `check_not_production_path`
        # (invariant 3). Refused, not normalised, exactly as `..` is.
        raise QwenTtsAdapterError(
            f"{label}: begins with '//', which names the same file as '/' on Linux but "
            f"is a different path root to every segment-wise comparison, including this "
            f"module's production-tree refusal: {value!r}")
    return value


def _require_non_negative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QwenTtsAdapterError(f"{label}: expected a number, got {type(value).__name__}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise QwenTtsAdapterError(f"{label}: must be finite, got {value!r}")
    if number < 0.0:
        raise QwenTtsAdapterError(f"{label}: must be non-negative, got {value!r}")
    return number


def _is_within(path: str, root: str) -> bool:
    """Segment-wise containment. `startswith` would call `qwentts.cpp-experimental`
    a production path and refuse the very tree candidates are built in."""
    return PurePosixPath(path).parts[:len(PurePosixPath(root).parts)] == \
        PurePosixPath(root).parts


def check_not_production_path(path: str, *, label: str = "path") -> None:
    """Raise when `path` is inside any frozen production tree (invariant 3, denial 2).

    A CANDIDATE-side check. The ANCHOR arm legitimately lives in the production tree
    and executing a frozen binary read-only is not a write, so anchors go through
    `expect_production_anchor()`.
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
                f"path anything is supposed to name — so it is the path a caller is most "
                f"likely to hand us (invariant 3; P-AK-SEARCH-1 denial 2)")


def expect_production_anchor(path: str, *, label: str = "anchor_path") -> str:
    _require_abs_path(path, label)
    if not _is_within(path, PRODUCTION_TREE_ROOT):
        raise QwenTtsAdapterError(
            f"{label} {path!r} is not inside {PRODUCTION_TREE_ROOT!r}; the {BACKEND} "
            f"anchor is the FROZEN production binary, and a rebuilt anchor is a "
            f"different anchor (P-AK-SEARCH-1 precondition 4)")
    return path


# =============================================================================
# Binary inventory and path construction
# =============================================================================


#: The ggml core EVERY member of this inventory resolves from its OWN tree. This is
#: the freeze premise restated as a set: qwentts.cpp runs ggml 0.17.0 while
#: whisper.cpp runs 0.18.0 and llama.cpp 0.16.0, so a member that resolves any of
#: these three from another tree runs silently wrong
#: (INC-20260731-ggml-linkage-silent-cpu-fallback).
CORE_SHARED_LIBRARIES = frozenset({
    "libggml-base.so",
    "libggml-cpu.so",
    "libggml.so",
})

#: Tree-local libraries never REQUIRED of any member. `libggml-hip.so` is here
#: because the HIP backend is `dlopen`ed at runtime and `ldd` cannot see it, so its
#: absence from a report is silence rather than evidence; its presence FROM ANOTHER
#: TREE is still a `BAD` line and still FAILs.
OPTIONAL_SHARED_LIBRARIES = frozenset({
    "libggml-hip.so",
})

#: WHERE a member's declared library set comes from. It says "declared, not
#: attested" on purpose: attesting a member's linkage means running `ldd` against
#: the FROZEN tree, and this module executes nothing (invariant 3). The declaration
#: is made in the direction that cannot manufacture a PASS — a required library
#: absent from a report is COULD_NOT_CHECK, and an optional one resolving from
#: another tree is still a FAIL.
_GGML_ONLY_MEMBER_PROVENANCE = (
    "role-derived: this backend has no engine shared object at all — qwentts.cpp's own "
    "code links as the static archive libqwen-core.a, which never appears in ldd output — "
    "so every member's required set is the ggml core it must resolve from its own tree. "
    "Declared here, not attested against a live ldd on the frozen tree."
)


@dataclass(frozen=True)
class BinarySpec:
    """One binary this backend measures, what it is for, and what IT links.

    `required_libraries` is **per member**, with no inventory-wide default. One set
    for the whole inventory makes the §10.2 phase-2 gate unrunnable for any member
    that links a subset: every report for that member is missing a library it never
    linked, so the verdict is COULD_NOT_CHECK forever and the phase can never pass.
    A default here would reintroduce that silently, so there is none.
    """

    name: str
    rel_path: str
    role: str
    required_libraries: frozenset
    optional_libraries: frozenset
    linkage_provenance: str

    def __post_init__(self) -> None:
        for field in ("required_libraries", "optional_libraries"):
            value = getattr(self, field)
            if not isinstance(value, frozenset) or not all(
                    isinstance(v, str) and v.endswith(".so") for v in value):
                raise QwenTtsAdapterError(
                    f"{self.name}.{field} must be a frozenset of `.so` stems, got "
                    f"{value!r}")
        if not self.required_libraries:
            raise QwenTtsAdapterError(
                f"{self.name} declares no required library; an empty requirement is a "
                f"gate that passes on any report at all")
        if not CORE_SHARED_LIBRARIES <= self.required_libraries:
            raise QwenTtsAdapterError(
                f"{self.name} does not require the ggml core {sorted(CORE_SHARED_LIBRARIES)}; "
                f"resolving this tree's own ggml is the property the freeze exists for")
        overlap = self.required_libraries & self.optional_libraries
        if overlap:
            raise QwenTtsAdapterError(
                f"{self.name} lists {sorted(overlap)} as both required and optional; a "
                f"library cannot be both, and the overlap decides silently which rule wins")
        if not isinstance(self.linkage_provenance, str) or not self.linkage_provenance.strip():
            raise QwenTtsAdapterError(
                f"{self.name} carries no linkage provenance; a per-member library set "
                f"whose origin is unrecorded is a guess nobody can audit")

    def to_dict(self) -> dict:
        return {"name": self.name, "rel_path": self.rel_path, "role": self.role,
                "required_libraries": sorted(self.required_libraries),
                "optional_libraries": sorted(self.optional_libraries),
                "linkage_provenance": self.linkage_provenance}


#: Verified against the frozen tree's own `build/` on 2026-08-03. Note every
#: `rel_path` lacks a `bin/` segment — asymmetry 1, carried in the data rather than
#: reconstructed by a caller. The library sets are DECLARED per member: uniform
#: today because this tree ships no engine shared object, per-member BY
#: CONSTRUCTION so a member that links a subset stays gradeable.
BINARY_INVENTORY = (
    BinarySpec("qwen-tts", "build/qwen-tts", "synthesis_cell",
               required_libraries=CORE_SHARED_LIBRARIES,
               optional_libraries=OPTIONAL_SHARED_LIBRARIES,
               linkage_provenance=_GGML_ONLY_MEMBER_PROVENANCE),
    BinarySpec("tts-server", "build/tts-server", "service_smoke",
               required_libraries=CORE_SHARED_LIBRARIES,
               optional_libraries=OPTIONAL_SHARED_LIBRARIES,
               linkage_provenance=_GGML_ONLY_MEMBER_PROVENANCE),
    BinarySpec("qwen-codec", "build/qwen-codec", "codec_cell",
               required_libraries=CORE_SHARED_LIBRARIES,
               optional_libraries=OPTIONAL_SHARED_LIBRARIES,
               linkage_provenance=_GGML_ONLY_MEMBER_PROVENANCE),
    BinarySpec("quantize", "build/quantize", "quantization_tool",
               required_libraries=CORE_SHARED_LIBRARIES,
               optional_libraries=OPTIONAL_SHARED_LIBRARIES,
               linkage_provenance=_GGML_ONLY_MEMBER_PROVENANCE),
    BinarySpec("test-backend-ops", "build/test-backend-ops", "op_and_unit_test",
               required_libraries=CORE_SHARED_LIBRARIES,
               optional_libraries=OPTIONAL_SHARED_LIBRARIES,
               linkage_provenance=_GGML_ONLY_MEMBER_PROVENANCE),
)

_BINARIES_BY_NAME = {b.name: b for b in BINARY_INVENTORY}


def binary_inventory() -> tuple:
    return BINARY_INVENTORY


def binary_path(tree_root: str, name: str, *, allow_production: bool = False) -> str:
    """Absolute path of `name` inside `tree_root`, with the layout DECLARED.

    The absence of a `bin/` segment is part of `BinarySpec.rel_path`, never a
    caller's concern. A shared "append bin/" convention silently produces a
    non-existent path for this backend (§1.5).
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
    _require_abs_path(tree_root, "tree_root")
    path = str(PurePosixPath(tree_root) / LIBRARY_DIR_REL)
    if not allow_production:
        check_not_production_path(path, label="library_dir")
    return path


def expected_shared_libraries(binary: str) -> frozenset:
    """The libraries THIS member must resolve from its own tree.

    `binary` has no default. The inventory-wide union is not a gate input and is not
    reachable as one — `all_declared_shared_libraries()` exists to DESCRIBE the tree,
    and grading a member against it is the defect this signature closes.

    `libqwen-core` appears in no member's set, required or optional: it is a static
    archive (`libqwen-core.a`) and never appears in `ldd` output at all, so listing
    it would make every report look incomplete.
    """
    return _require_member(binary).required_libraries


def optional_shared_libraries(binary: str) -> frozenset:
    """Libraries THIS member may resolve. Absence is not a finding; wrong tree is."""
    return _require_member(binary).optional_libraries


def all_declared_shared_libraries() -> frozenset:
    """The union over the inventory. A DESCRIPTION of the tree, never a gate input."""
    union: frozenset = frozenset()
    for spec in BINARY_INVENTORY:
        union = union | spec.required_libraries | spec.optional_libraries
    return union


def _require_member(binary: str) -> BinarySpec:
    _require_str(binary, "binary")
    try:
        return _BINARIES_BY_NAME[binary]
    except KeyError as exc:
        raise UnknownBinary(
            f"{binary!r} is not in the {BACKEND} binary inventory; declared binaries "
            f"are {sorted(_BINARIES_BY_NAME)}") from exc


# =============================================================================
# Linkage verification (§10.2 phase 2)
# =============================================================================

LINKAGE_VERIFIER = (
    "/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
)

#: The verifier's own library-name filter, restated so the coverage gap can be
#: NAMED rather than discovered. Anything outside it is not examined, and its
#: absence from a report is silence rather than evidence.
VERIFIER_NAME_FILTER = ("libggml*", "libwhisper*", "libllama*", "libmtmd*")


@dataclass(frozen=True)
class LinkageInvocation:
    """The exact command a runner must execute, with a FULLY DECLARED environment."""

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

    `library_path_entries` is the COMPLETE ordered `LD_LIBRARY_PATH` and its first
    entry MUST be the binary's own directory: that ordering is the property being
    verified, and inheriting the ambient path is the 2026-07-31 failure mode itself.
    """
    _require_abs_path(binary, "binary")
    entries = [_require_abs_path(e, f"library_path_entries[{i}]")
               for i, e in enumerate(library_path_entries)]
    if not entries:
        raise QwenTtsAdapterError(
            "library_path_entries is empty; the invocation must declare the complete "
            "LD_LIBRARY_PATH, because inheriting the ambient one is the 2026-07-31 "
            "failure mode itself")
    own_dir = str(PurePosixPath(binary).parent)
    if entries[0] != own_dir:
        raise QwenTtsAdapterError(
            f"library_path_entries[0] is {entries[0]!r} but must be the binary's own "
            f"directory {own_dir!r}: the loader honours LD_LIBRARY_PATH before a "
            f"binary's own directory, so anything else lets another tree's ggml win")
    root = own_dir if expected_root is None else _require_abs_path(expected_root,
                                                                  "expected_root")
    return LinkageInvocation(argv=(LINKAGE_VERIFIER, binary, root),
                             env={"LD_LIBRARY_PATH": ":".join(entries)},
                             binary=binary, expected_root=root)


@dataclass(frozen=True)
class LinkageVerdict:
    check: schemas.Check
    ok_libraries: tuple
    bad_libraries: tuple
    missing_expected: tuple
    resolved_count: int
    binary: str
    required_libraries: tuple

    def to_dict(self) -> dict:
        return {"outcome": self.check.outcome, "reasons": list(self.check.reasons),
                "ok_libraries": list(self.ok_libraries),
                "bad_libraries": list(self.bad_libraries),
                "missing_expected": list(self.missing_expected),
                "resolved_count": self.resolved_count,
                "binary": self.binary,
                "required_libraries": list(self.required_libraries)}


_OK_LINE_RE = re.compile(r"^\s{2}OK\s+(\S+)\s+->\s+(\S+)\s*$")
_BAD_LINE_RE = re.compile(r"^\s{2}BAD\s+(\S+)\s+->\s+(\S+)\s*$")
_NO_LIBS_MARKER = "no ggml/whisper/llama libs in ldd output"

#: The verifier's own first line: `echo "binary : $BIN"`, printed before any ldd
#: output. It is the ONLY thing in the report that says which binary was inspected.
_REPORT_BINARY_RE = re.compile(r"^binary\s*:\s*(?P<path>\S.*?)\s*$", re.MULTILINE)


def report_binary_name(stdout: str) -> Optional[str]:
    """The basename the report itself says it inspected, or None if it says nothing."""
    if not isinstance(stdout, str):
        raise QwenTtsAdapterError(f"stdout must be a string, got {type(stdout).__name__}")
    match = _REPORT_BINARY_RE.search(stdout)
    if match is None:
        return None
    return PurePosixPath(match.group("path")).name


def _soname_stem(name: str) -> str:
    head, sep, _ = name.partition(".so")
    return head + sep if sep else name


def interpret_linkage_report(stdout: str, exit_code: int, *,
                             binary: str) -> LinkageVerdict:
    """Turn a captured verifier report into a three-outcome verdict, FOR ONE MEMBER.

    Identical rules to the STT sibling, and deliberately not shared with it: the two
    adapters declare different expected library sets, and a shared helper would make
    one backend's coverage gap invisible in the other's report.

    `binary` is a required keyword: the report is graded against **that member's**
    declared library set, never against the inventory union. A union grades every
    member by the strictest one, so a member linking a subset is permanently
    COULD_NOT_CHECK and §10.2 phase 2 can never pass for it.

    And `binary` is CHECKED against the report's own `binary :` header, not trusted.
    A name the caller supplies is a claim by the party being gated; the header is
    the evidence's own statement of what it is. A disagreement, or a report naming
    no binary at all, is COULD_NOT_CHECK — never PASS. Uniform required sets today
    make the STT sibling the one that can currently be exploited by relabelling, but
    the binding belongs to the signature, not to the current contents of the table.
    """
    if not isinstance(stdout, str):
        raise QwenTtsAdapterError(f"stdout must be a string, got {type(stdout).__name__}")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise QwenTtsAdapterError("exit_code must be an int")
    spec = _require_member(binary)
    required = frozenset(spec.required_libraries)

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
    seen = {_soname_stem(name) for name, _ in ok}
    missing = tuple(sorted(required - seen))
    member = tuple(sorted(required))

    if bad:
        offenders = ", ".join(f"{n} -> {p}" for n, p in bad)
        return LinkageVerdict(
            check=schemas.Check(schemas.FAIL, (
                f"{len(bad)} library/libraries resolve outside the candidate's own tree: "
                f"{offenders}. qwentts.cpp runs ggml {GGML_GENERATION} while whisper.cpp "
                f"runs 0.18.0 and llama.cpp 0.16.0 — a binary inheriting another tree's "
                f"ggml runs silently wrong",)),
            ok_libraries=tuple(sorted(ok)), bad_libraries=tuple(sorted(bad)),
            missing_expected=missing, resolved_count=resolved,
            binary=spec.name, required_libraries=member)

    if exit_code != 0:
        return LinkageVerdict(
            check=schemas.Check(schemas.FAIL, (
                f"verifier exited {exit_code} with no BAD line parsed; the report is "
                f"inconsistent with its own exit status and is not trusted",)),
            ok_libraries=tuple(sorted(ok)), bad_libraries=(),
            missing_expected=missing, resolved_count=resolved,
            binary=spec.name, required_libraries=member)

    named = report_binary_name(stdout)
    if named != spec.name:
        return LinkageVerdict(
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                (f"this report was captured against {named!r} and is being graded as "
                 f"{spec.name!r}. `binary=` is the CALLER's claim about the evidence; "
                 f"the report's own `binary :` header is the evidence's statement of "
                 f"what it is, and only the second is supplied by something other than "
                 f"the party being gated. Per-member sets make the member's identity "
                 f"load-bearing, so a report graded under the wrong member is graded "
                 f"against a set it was never captured for"
                 if named is not None else
                 f"the report carries no `binary : <path>` header, so nothing in it "
                 f"says which binary was inspected and it cannot be bound to "
                 f"{spec.name!r}. The verifier prints that header unconditionally "
                 f"before any ldd output; a report without one is not its output, or "
                 f"is a fragment of it, and grading it against a member's set would be "
                 f"grading the caller's claim instead of the evidence"),)),
            ok_libraries=tuple(sorted(ok)), bad_libraries=(),
            missing_expected=(), resolved_count=resolved,
            binary=spec.name, required_libraries=member)

    if resolved == 0 or _NO_LIBS_MARKER in stdout:
        return LinkageVerdict(
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                "the verifier resolved no ggml libraries at all — statically linked, or "
                "`ldd` failed. It exits 0 in this state, so an exit-status consumer would "
                "record a PASS for a check that did not run",)),
            ok_libraries=(), bad_libraries=(), missing_expected=missing,
            resolved_count=0, binary=spec.name, required_libraries=member)

    if missing:
        return LinkageVerdict(
            check=schemas.Check(schemas.COULD_NOT_CHECK, (
                f"libraries {spec.name!r} is declared to require are absent from the "
                f"report: {list(missing)} (its declared set is {list(member)}; "
                f"provenance: {spec.linkage_provenance}). The verifier's name filter is "
                f"{list(VERIFIER_NAME_FILTER)}, so a library outside it is never examined "
                f"and its absence is silence, not evidence",)),
            ok_libraries=tuple(sorted(ok)), bad_libraries=(),
            missing_expected=missing, resolved_count=resolved,
            binary=spec.name, required_libraries=member)

    return LinkageVerdict(check=schemas.Check(schemas.PASS),
                          ok_libraries=tuple(sorted(ok)), bad_libraries=(),
                          missing_expected=(), resolved_count=resolved,
                          binary=spec.name, required_libraries=member)


#: This engine's LOG GRAMMAR — the line shape a qwentts.cpp build prints when a
#: device was enumerated, and the flag it prints when one was merely requested. The
#: grammar is this adapter's; what a device NAME denotes is not, and lives in
#: `evaluator/devices.py` so the two speech adapters cannot diverge on it.
_DEVICE_LINE_RE = re.compile(r"Device\s+\d+\s*:\s*(?P<name>[^\n,]+)", re.IGNORECASE)
_REQUEST_FLAG_RE = re.compile(r"use\s+gpu\s*=\s*1", re.IGNORECASE)


def device_names_in_log(startup_log: str) -> tuple:
    """Every `Device N: <name>` the log enumerates, in order.

    `finditer`, not `search`: a ROCm build enumerates more than one device, and
    grading the FIRST line alone lets one entry decide a cell the others contradict.
    """
    if not isinstance(startup_log, str):
        raise QwenTtsAdapterError("startup_log must be a string")
    return tuple(m.group("name").strip() for m in _DEVICE_LINE_RE.finditer(startup_log))


def check_device_evidence(startup_log: str, *, expected_lane: str) -> schemas.Check:
    """Confirm from the engine's OWN startup log which device actually loaded.

    The verifier says it in its own PASS message: ggml backends are `dlopen`ed at
    runtime and are not covered by `ldd`, so a `use gpu = 1` flag reports what was
    REQUESTED, never what was LOADED.

    The presence of a device LINE is necessary and not sufficient, which is the
    carried-forward defect this now closes: `Device 0: CPU` is a device line, and it
    is exactly what a silently-fallen-back ggml prints. The NAME is graded against
    `evaluator/devices.py`'s vocabulary — one table for both speech adapters.
    """
    if not isinstance(startup_log, str):
        raise QwenTtsAdapterError("startup_log must be a string")
    if expected_lane not in ("cpu", "gpu"):
        raise QwenTtsAdapterError(f"expected_lane must be 'cpu' or 'gpu', got "
                                  f"{expected_lane!r}")
    if not startup_log.strip():
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("startup log is empty; no device evidence was captured",))
    names = device_names_in_log(startup_log)
    requested = bool(_REQUEST_FLAG_RE.search(startup_log))
    if expected_lane == "gpu":
        if not names:
            reasons = ["no `Device N: <name>` line in the startup log, so nothing "
                       "establishes which backend actually loaded"]
            if requested:
                reasons.append("the log carries `use gpu = 1`, which reports what was "
                               "REQUESTED, never what was LOADED")
            return schemas.Check(schemas.FAIL, tuple(reasons))
        return devices.check_device_names(names, expected_lane="gpu")
    if names:
        return devices.check_device_names(names, expected_lane="cpu")
    if requested:
        return schemas.Check(schemas.FAIL, (
            "a CPU cell's log carries `use gpu = 1`; the request contradicts the "
            "declared lane",))
    return schemas.Check(schemas.PASS)


# =============================================================================
# Phases, metrics, and resource lane
# =============================================================================

#: The engine's own pipeline stages, not invented labels. `schemas.PHASES_BY_BACKEND`
#: mirrors this tuple so every release consumer shares the same vocabulary.
PHASES = ("talker", "code_predictor", "codec_decode", "end_to_end")

#: Stage attribution is REQUIRED by `P-TTS-3`, not optional: on 2026-07-31 the
#: CPU->GPU transition moved the bottleneck from `CodecDecode` (64 % -> 10.4 % of
#: wall) to `CodePredictor` (-> 65.5 %). A campaign reading only end-to-end RTF would
#: have kept optimizing the vocoder after it stopped being the problem.
STAGE_PHASES = ("talker", "code_predictor", "codec_decode")

METRIC_DIRECTIONS = {
    "ttfa_ms": "lower_better",
    "rtf": "lower_better",
    "xrt": "higher_better",
    "throughput_audio_s_per_wall_s": "higher_better",
    "roundtrip_wer_pct": "lower_better",
    "spectral_distance": "lower_better",
    "clipping_fraction": "lower_better",
    "rss_slope_mib_per_cycle": "lower_better",
}

_AMBIGUOUS_METRIC_NAMES = frozenset({
    "real_time_factor", "realtime_factor", "rt_factor", "speed", "quality", "mos",
})


def check_phase(phase: str) -> str:
    _require_str(phase, "phase")
    if phase not in PHASES:
        raise UnknownPhase(f"{phase!r} is not a {BACKEND} phase; declared phases are "
                           f"{list(PHASES)}")
    return phase


def metric_direction(metric: str) -> str:
    """Return the declared direction, or refuse.

    `real_time_factor` is refused BY NAME. This project carries `rtf: 0.169`
    (wall/audio, lower-better) in the ratified speech-freeze receipt and
    `xRT 0.86x -> 5.47x` (audio/wall, higher-better) in the owning handoff, for the
    same engine. They are reciprocals; a name that does not say which is not a metric
    (`MEASUREMENT.md:39-41`, CLAUDE.md "Always confirm metric direction").

    `mos` is refused because neither available signal is a mean opinion score and
    neither may be described as one (`P-TTS-2`).
    """
    _require_str(metric, "metric")
    if metric in _AMBIGUOUS_METRIC_NAMES:
        raise UnknownMetric(
            f"{metric!r} names no direction and no denominator. Use `rtf` "
            f"(wall_s/audio_s, lower-better) or `xrt` (audio_s/wall_s, higher-better); "
            f"they are reciprocals and this project carries both for one engine. There "
            f"is no MOS-grade metric on this backend")
    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise UnknownMetric(f"{metric!r} is not a declared {BACKEND} metric; declared "
                            f"metrics are {sorted(METRIC_DIRECTIONS)}") from exc


def check_metric_commensurable(metric: str) -> schemas.Check:
    metric_direction(metric)
    return schemas.check_metric_commensurability(BACKEND, {"metric": metric})


def rtf_from_xrt(xrt: float) -> float:
    """`rtf = 1 / xrt`. Explicit, because the two coexist in this project's artifacts."""
    value = _require_non_negative(xrt, "xrt")
    if value == 0.0:
        raise QwenTtsAdapterError("xrt of 0 has no reciprocal; a zero audio-per-wall "
                                  "ratio means no audio was produced, which is a "
                                  "categorical failure, not a slow run")
    return 1.0 / value


def xrt_from_rtf(rtf: float) -> float:
    value = _require_non_negative(rtf, "rtf")
    if value == 0.0:
        raise QwenTtsAdapterError("rtf of 0 has no reciprocal; zero wall time per audio "
                                  "second is not a measurement")
    return 1.0 / value


def resource_lane(*, device: Optional[str]) -> str:
    if device is None:
        return "cpu"
    _require_str(device, "device")
    return "gpu"


# =============================================================================
# Domain ownership
# =============================================================================

OWNED_DOMAINS = frozenset({"src", "include", "ggml", "examples", "tests", "cmake"})

#: `ggml/` is shared core for this tree AND is a submodule, so a change there is
#: both maximally reaching and maximally easy to under-report (asymmetry 2).
SHARED_CORE_DOMAINS = frozenset({"ggml", "include"})


def owned_domains() -> frozenset:
    return OWNED_DOMAINS


def check_domains_owned(domains: Iterable[str]) -> schemas.Check:
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

    The traced counterpart of `touches_shared_core()`. §10.6's review marking must
    not be a function of what a candidate says about itself when the diff is in hand
    (invariant 18: declared equals traced), and `run_source_integrity_gates` already
    derives the same flag from `risk_tier.matched_core_paths`. Here it matters twice
    over: every accepted production change to this tree so far is inside
    `ggml/src/ggml-cuda/`, and in this tree `ggml` is a submodule whose expanded
    diff paths are exactly the ones a declared-domain list is most likely to omit.
    """
    return tuple(sorted(p for p in diff.paths() if _top_domain(p) in SHARED_CORE_DOMAINS))


def check_declared_domains_cover_diff(diff: integrity.SourceDiff,
                                      domains: Iterable[str]) -> schemas.Check:
    """FAIL when the diff reaches a domain the proposal did not declare."""
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
# Complexity ceiling (§10.6) — with the submodule EXPANDED
# =============================================================================

#: DERIVATION, stated so the ceiling is not a number somebody liked.
#:
#: §10.6 makes the ceiling a MARKING threshold: above it the release package is
#: marked `REQUIRES_HUMAN_CODE_REVIEW` and says so on its first page. The calibration
#: is *"larger than anything this project has ever put on this branch"*, measured
#: from the branch itself, **with the submodule expanded**:
#:
#:     git log production-speech-v1 --not <upstream base> --numstat        # superproject
#:     git -C ggml show --numstat <gitlink target>                         # submodule
#:
#: Measured 2026-08-03: the superproject commit `2c1b5182` reports `ggml | 2 +-` —
#: **1 file, 2 changed lines** — while the submodule commit `b86f6602` it points at
#: changes **4 files and 115 lines** (`argsort.cu` 70+/26-, `argsort.cuh` 3+/0-,
#: `ggml-cuda.cu` 3+/1-, `vendors/hip.h` 10+/2-). The expanded figures are the
#: derivation inputs; the superproject figures would under-report by two orders of
#: magnitude.
#:
#: The consequence is deliberate: most LLM-authored changes to this tree will be
#: marked `REQUIRES_HUMAN_CODE_REVIEW`. qwentts.cpp is a third-party fork this project
#: does not own and whose upstream it does not control, and inflating the ceiling to
#: make the loop convenient would be a downgrade dressed as a calibration. Recomputed
#: at every freeze.
CEILING_DERIVATION = (
    "max(changed_lines) and max(files_touched) over every commit on "
    "production-speech-v1 beyond its upstream base WITH the ggml submodule expanded, "
    "measured 2026-08-03: superproject 2c1b5182 (1 file, 2 lines) whose gitlink target "
    "b86f6602 changes 4 files and 115 lines"
)
_EXPANDED_MAX_CHANGED_LINES = 115
_EXPANDED_MAX_FILES_TOUCHED = 4

#: What the same history looks like WITHOUT traversal. Kept as data so
#: `check_closure_traversed_submodules()` can quantify the under-report instead of
#: merely asserting one exists.
_SUPERPROJECT_ONLY_MAX_CHANGED_LINES = 2
_SUPERPROJECT_ONLY_MAX_FILES_TOUCHED = 1


def complexity_ceiling() -> integrity.ComplexityCeiling:
    return integrity.ComplexityCeiling(
        backend=BACKEND,
        max_diff_lines=_EXPANDED_MAX_CHANGED_LINES,
        max_files_touched=_EXPANDED_MAX_FILES_TOUCHED,
        # Every accepted production change to this tree so far is inside
        # `ggml/src/ggml-cuda/`, i.e. shared core for this binary.
        shared_core_modification_requires_review=True,
        declared_by=f"autokernel.adapters.{BACKEND}/v1 ({CEILING_DERIVATION})",
    )


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
        "arithmetic": env("arithmetic", 4, 300, 16),
        "layout": env("layout", 6, 600, 30, creation=True),
        "fusion": env("fusion", 5, 500, 24, creation=True),
        "oracle_port": env("oracle_port", 8, 900, 40, creation=True),
        "core_header": env("core_header", 3, 150, 10),
    }


def assess_complexity(diff: integrity.SourceDiff, *, change_class: str,
                      domains: Iterable[str],
                      submodule_traversed: bool) -> integrity.ComplexityAssessment:
    """§10.6 marking. REFUSES a diff computed without traversing the submodule.

    A superproject-only diff of a `ggml/` change measures 2 lines where the truth is
    115. Assessing it would mark a genuine shared-core rewrite as a trivial patch,
    which is precisely the marking §10.6 exists to make.
    """
    if not isinstance(submodule_traversed, bool):
        raise QwenTtsAdapterError("submodule_traversed must be a bool")
    if not submodule_traversed:
        raise SubmoduleClosureMissing(
            f"the diff was computed without traversing {list(SUBMODULE_PATHS)}. In this "
            f"tree a gitlink change is ONE line in the superproject and up to "
            f"{_EXPANDED_MAX_CHANGED_LINES} lines across "
            f"{_EXPANDED_MAX_FILES_TOUCHED} files in the submodule (frozen precedent: "
            f"{_SUPERPROJECT_ONLY_MAX_FILES_TOUCHED} file / "
            f"{_SUPERPROJECT_ONLY_MAX_CHANGED_LINES} lines untraversed, versus "
            f"{_EXPANDED_MAX_FILES_TOUCHED} files / {_EXPANDED_MAX_CHANGED_LINES} lines "
            f"expanded). Refusing to assess complexity on a diff that under-reports by "
            f"two orders of magnitude")
    # TRACED from the diff, then OR-ed with the declared list. The declared list alone
    # would make the §10.6 marking a function of what the candidate says about itself:
    # this tree's own frozen change is 4 files / 115 lines inside `ggml/src/ggml-cuda/`,
    # and a proposal declaring `domains=("src",)` for a small one removed every reason
    # to mark it. The declared list can only ADD a reason, never subtract one.
    traced = shared_core_paths(diff)
    return integrity.assess_complexity_ceiling(
        diff, complexity_ceiling(),
        touches_shared_core=bool(traced) or touches_shared_core(domains),
        change_class=change_class)


# =============================================================================
# The backend-unchanged test (§3.2), single-backend + submodule form
# =============================================================================


@dataclass(frozen=True)
class UnchangedTestPlan:
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
        transfer_available=False,
        rationale=(
            "qwentts.cpp serves exactly one backend, so §3.2's cell-dropping transfer "
            "has no counterpart: there is no other backend to drop. Both stages still "
            "run, to establish that the candidate binary differs from the incumbent AT "
            "ALL, so a no-op candidate is refused. Stage 1 here MUST traverse the ggml "
            "submodule: a closure computed on the superproject alone reports a 115-line "
            "change as one line."),
    )


def check_closure_traversed_submodules(traversed: Collection[str]) -> schemas.Check:
    """A §3.2 stage-1 result computed without `ggml/` is not a stage-1 result."""
    seen = {_require_str(p, "traversed_path") for p in traversed}
    missing = sorted(set(SUBMODULE_PATHS) - seen)
    if missing:
        return schemas.Check(schemas.FAIL, (
            f"the source closure did not traverse {missing}; in this tree that hides up "
            f"to {_EXPANDED_MAX_CHANGED_LINES} changed lines across "
            f"{_EXPANDED_MAX_FILES_TOUCHED} files behind a "
            f"{_SUPERPROJECT_ONLY_MAX_CHANGED_LINES}-line gitlink diff (§3.2 stage 1)",))
    return schemas.Check(schemas.PASS)


def classify_unchanged_result(*, stage1_closure_empty: bool,
                              stage2_normalized_identical: Optional[bool],
                              submodule_traversed: bool) -> schemas.Check:
    """PASS = the candidate genuinely differs. A no-op candidate FAILs.

    A stage-1 result computed without submodule traversal cannot be classified at
    all: `closure_empty=True` from an untraversed closure is not evidence of
    anything.
    """
    if not isinstance(submodule_traversed, bool):
        raise QwenTtsAdapterError("submodule_traversed must be a bool")
    if not isinstance(stage1_closure_empty, bool):
        raise QwenTtsAdapterError("stage1_closure_empty must be a bool")
    if not submodule_traversed:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"stage 1 did not traverse {list(SUBMODULE_PATHS)}; an 'empty closure' from "
            f"an untraversed closure is silence, not evidence",))
    if stage2_normalized_identical is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "stage 2 (normalized binary comparison against an anchor rebuild in the "
            "candidate's environment) was not run; stage 1 alone may not drop or admit "
            "a cell (§3.2)",))
    if not isinstance(stage2_normalized_identical, bool):
        raise QwenTtsAdapterError("stage2_normalized_identical must be a bool or None")
    if stage1_closure_empty != stage2_normalized_identical:
        return schemas.Check(schemas.FAIL, (
            f"stage 1 reports closure_empty={stage1_closure_empty} and stage 2 reports "
            f"normalized_identical={stage2_normalized_identical}. §3.2: a disagreement is "
            f"a hard finding filed against the build-identity machinery, never a silent "
            f"preference for the cheaper answer",))
    if stage1_closure_empty:
        return schemas.Check(schemas.FAIL, (
            "the candidate's source closure is empty and its normalized binary is "
            "identical to the incumbent's: this is a NO-OP candidate and is refused "
            "before it consumes a release matrix",))
    return schemas.Check(schemas.PASS)


# =============================================================================
# P-TTS-1 — input identity, greedy arm, two-layer oracle, numerical safety
# =============================================================================

#: `P-TTS-1` §1.1. Every one of these determines the output, so a missing one makes
#: the run unreproducible and the cell uncomparable. A voice-clone reference is an
#: INPUT, not a setting.
INPUT_IDENTITY_FIELDS = (
    "prompt_text_sha256", "prompt_text_bytes", "tokenizer_sha256",
    "talker_weights_sha256", "code_predictor_weights_sha256",
    "speaker_conditioning_sha256", "sampling_policy", "cache_state",
)

#: Mirrors `correctness.CACHE_STATES`. `unknown` exists so "we did not record it" is
#: sayable; `served_from_cache` is control 3's shape and FAILs.
CACHE_STATES = ("cold", "warm_page_cache", "warm_kv_cache", "served_from_cache", "unknown")


def check_input_identity(record: Mapping[str, Any]) -> schemas.Check:
    """Every field present and well-formed, or the cell is unreproducible.

    `speaker_conditioning_sha256` may be `None`, meaning *no voice-clone reference was
    used*; it may NOT be absent, because absent and none are different facts.
    """
    if not isinstance(record, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("input identity record is not a mapping",))
    missing = [f for f in INPUT_IDENTITY_FIELDS if f not in record]
    if missing:
        return schemas.Check(schemas.FAIL, (
            f"input identity is missing {missing}; every one of them determines the "
            f"audio, so a missing field makes the run unreproducible (P-TTS-1 §1.1)",))
    for field in ("prompt_text_sha256", "tokenizer_sha256", "talker_weights_sha256",
                  "code_predictor_weights_sha256"):
        digest = record.get(field)
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            return schemas.Check(schemas.FAIL,
                                 (f"{field} is not a lowercase sha256: {digest!r}",))
        if schemas.is_placeholder_digest(digest):
            return schemas.Check(schemas.FAIL, (
                f"{field} {digest!r} is a well-formed digest no measurement produced; a "
                f"fabricated hash is indistinguishable from a measured one to every "
                f"downstream reader",))
    speaker = record.get("speaker_conditioning_sha256")
    if speaker is not None and (not isinstance(speaker, str)
                                or not _SHA256_RE.match(speaker)):
        return schemas.Check(schemas.FAIL, (
            f"speaker_conditioning_sha256 must be a sha256 or None (meaning no clone "
            f"reference was used), got {speaker!r}",))
    length = record.get("prompt_text_bytes")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        return schemas.Check(schemas.FAIL,
                             (f"prompt_text_bytes must be a positive int, got {length!r}",))
    cache = record.get("cache_state")
    if cache not in CACHE_STATES:
        return schemas.Check(schemas.FAIL,
                             (f"cache_state {cache!r} is not one of {list(CACHE_STATES)}",))
    if cache == "served_from_cache":
        return schemas.Check(schemas.FAIL, (
            "cache_state is 'served_from_cache': a cached waveform is control 3's shape "
            "(degraded-negative) and MUST receive no rank at all",))
    return schemas.Check(schemas.PASS)


def check_greedy_arm(*, greedy: Optional[bool], temperature: Optional[float]
                     ) -> schemas.Check:
    """`P-TTS-1` §1.2: the greedy arm is MANDATORY and is the arm the release rule reads.

    Under greedy decoding the codec token sequence is a deterministic function of the
    inputs and the weights, so identity is OBSERVABLE. Under sampling it is not, and
    every check degrades from an identity test to a distributional one needing orders
    of magnitude more samples for the same resolution. The 2026-07-31 measurement
    recorded that under `--greedy` the GPU and CPU transcripts were identical, which
    is what makes a cross-backend identity oracle available here at all.
    """
    if greedy is None:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("the arm does not declare whether greedy decoding was in "
                              "force; an undeclared sampling policy is not a greedy one",))
    if not isinstance(greedy, bool):
        raise QwenTtsAdapterError("greedy must be a bool or None")
    if not greedy:
        return schemas.Check(schemas.FAIL, (
            "this is a sampled arm. A sampled arm MAY be reported alongside as a "
            "diagnostic; it MUST NOT be the arm a release verdict is taken on, and it "
            "MUST NOT be substituted for the greedy arm when the greedy arm fails "
            "(P-TTS-1 §1.2)",))
    if temperature is not None:
        value = _require_non_negative(temperature, "temperature")
        if value != 0.0:
            return schemas.Check(schemas.FAIL, (
                f"greedy=True but temperature={value!r}; the two contradict each other "
                f"and the arm's determinism is not established",))
    return schemas.Check(schemas.PASS)


@dataclass(frozen=True)
class IdentityVerdict:
    """The two-layer oracle of `P-TTS-1` §1.3."""

    layer1_codes: schemas.Check
    layer2_waveform: schemas.Check
    combined: schemas.Check

    def to_dict(self) -> dict:
        return {"layer1_codes": {"outcome": self.layer1_codes.outcome,
                                 "reasons": list(self.layer1_codes.reasons)},
                "layer2_waveform": {"outcome": self.layer2_waveform.outcome,
                                    "reasons": list(self.layer2_waveform.reasons)},
                "combined": {"outcome": self.combined.outcome,
                             "reasons": list(self.combined.reasons)}}


_SEVERITY = {schemas.PASS: 0, schemas.COULD_NOT_CHECK: 1, schemas.FAIL: 2}


def _worst(*checks: schemas.Check) -> schemas.Check:
    worst = max(checks, key=lambda c: _SEVERITY[c.outcome])
    if worst.outcome == schemas.PASS:
        return schemas.Check(schemas.PASS)
    reasons = tuple(r for c in checks if c.outcome != schemas.PASS for r in c.reasons)
    return schemas.Check(worst.outcome, reasons)


def check_code_sequence_identity(*, candidate_sha256: Optional[str],
                                 anchor_sha256: Optional[str]) -> schemas.Check:
    """Layer 1 — the sharp oracle. `COULD_NOT_CHECK` when the build does not expose it.

    This layer isolates the LM half (Talker + CodePredictor) from the vocoder half. A
    candidate that changes only vocoder arithmetic MUST be identical here; a
    divergence means the change reached further than declared, which is an
    affected-surface finding (invariant 18: declared equals traced).
    """
    if candidate_sha256 is None or anchor_sha256 is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the build does not expose the emitted codec token sequence, so the sharp "
            "oracle is unavailable. This is not a pass, and the coverage gap is "
            "journaled rather than worked around (P-AK-SEARCH-1 denial 6)",))
    for label, value in (("candidate_sha256", candidate_sha256),
                         ("anchor_sha256", anchor_sha256)):
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            return schemas.Check(schemas.FAIL,
                                 (f"{label} is not a lowercase sha256: {value!r}",))
        if schemas.is_placeholder_digest(value):
            return schemas.Check(schemas.FAIL, (
                f"{label} {value!r} is a well-formed digest no measurement produced",))
    if candidate_sha256 != anchor_sha256:
        return schemas.Check(schemas.FAIL, (
            f"codec token sequences differ (candidate {candidate_sha256[:12]} vs anchor "
            f"{anchor_sha256[:12]}); under greedy decoding they are a deterministic "
            f"function of the inputs and weights, so the change reached the LM half",))
    return schemas.Check(schemas.PASS)


def derive_waveform_tolerance(*, anchor_aa_dispersion: float,
                              determinism_class: str) -> float:
    """The layer-2 tolerance, DERIVED from the anchor's own A/A dispersion.

    `P-TTS-1` §1.3: *"If the anchor is bitwise stable, the dispersion is exactly zero
    and layer 2 collapses to identity."* A tolerance is only ever as wide as the
    instrument's own measured noise; no value here may be supplied as a literal
    (`kernel-research.md:263-268`).
    """
    if determinism_class not in schemas.DETERMINISM_CLASSES:
        raise QwenTtsAdapterError(
            f"determinism_class {determinism_class!r} is not one of "
            f"{sorted(schemas.DETERMINISM_CLASSES)}")
    dispersion = _require_non_negative(anchor_aa_dispersion, "anchor_aa_dispersion")
    if determinism_class == "not_measured":
        raise DerivationImpossible(
            "determinism class is 'not_measured', so no waveform tolerance can be "
            "derived: an unmeasured stability is not a stable one (invariant 12)")
    if determinism_class == "bitwise_stable":
        if dispersion != 0.0:
            raise QwenTtsAdapterError(
                f"determinism_class is 'bitwise_stable' but the A/A dispersion is "
                f"{anchor_aa_dispersion!r}; a bitwise-stable instrument has zero "
                f"dispersion by construction, so one of the two measurements is wrong "
                f"and that is a hard finding, not a tolerance to widen")
        return 0.0
    return dispersion


def check_waveform_identity(*, candidate_sample_count: int, anchor_sample_count: int,
                            max_abs_difference: float, spectral_distance: float,
                            tolerance: float) -> schemas.Check:
    """Layer 2 — the tolerant oracle, with sample-count identity checked FIRST and exactly.

    A length change means the model stopped somewhere else. It is categorical, and
    averaging a distance over two waveforms of different lengths produces a number
    with no interpretation.
    """
    for label, value in (("candidate_sample_count", candidate_sample_count),
                         ("anchor_sample_count", anchor_sample_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return schemas.Check(schemas.FAIL,
                                 (f"{label} must be a positive int, got {value!r}; zero "
                                  f"samples is NO AUDIO PRODUCED, a categorical failure",))
    if candidate_sample_count != anchor_sample_count:
        return schemas.Check(schemas.FAIL, (
            f"sample counts differ ({candidate_sample_count} vs {anchor_sample_count}); "
            f"the model stopped somewhere else. This is categorical and a distance over "
            f"differing lengths has no interpretation (P-TTS-1 §1.3)",))
    maxabs = _require_non_negative(max_abs_difference, "max_abs_difference")
    spectral = _require_non_negative(spectral_distance, "spectral_distance")
    tol = _require_non_negative(tolerance, "tolerance")
    exceeded = []
    if maxabs > tol:
        exceeded.append(f"max_abs_difference {maxabs} exceeds the derived tolerance {tol}")
    if spectral > tol:
        exceeded.append(f"spectral_distance {spectral} exceeds the derived tolerance {tol}")
    if exceeded:
        return schemas.Check(schemas.FAIL, tuple(exceeded))
    return schemas.Check(schemas.PASS)


def check_numerical_safety(*, nan_count: Optional[int], inf_count: Optional[int],
                           clipping_fraction: Optional[float],
                           clipping_band: Optional[Sequence[float]],
                           dc_offset: Optional[float],
                           dc_band: Optional[Sequence[float]]) -> schemas.Check:
    """`P-TTS-1` §1.4. Any NaN or Inf FAILs regardless of how the audio sounds.

    A NaN that clips to silence is inaudible in a five-second clip and catastrophic
    in production, and it will not move a round-trip word-error rate at all.
    """
    if nan_count is None or inf_count is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the PCM was not scanned for NaN/Inf; an unscanned buffer is not a clean "
            "one",))
    for label, value in (("nan_count", nan_count), ("inf_count", inf_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return schemas.Check(schemas.FAIL,
                                 (f"{label} must be a non-negative int, got {value!r}",))
    if nan_count or inf_count:
        return schemas.Check(schemas.FAIL, (
            f"numerical failure: {nan_count} NaN and {inf_count} Inf samples. This FAILs "
            f"regardless of how the audio sounds",))
    reasons = []
    for name, value, band in (("clipping_fraction", clipping_fraction, clipping_band),
                              ("dc_offset", dc_offset, dc_band)):
        if value is None or band is None:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{name} or its anchor-calibrated band was not supplied; a bound with no "
                f"band is not a bound",))
        number = float(value) if not isinstance(value, bool) else None
        if number is None:
            return schemas.Check(schemas.FAIL, (f"{name} must be a number, got {value!r}",))
        pair = list(band)
        if len(pair) != 2:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"{name} band must be a [lo, hi] pair, got {pair!r}",))
        lo, hi = float(pair[0]), float(pair[1])
        if lo > hi:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"{name} band {pair!r} is inverted",))
        if not (lo <= number <= hi):
            reasons.append(f"{name} {number} is outside the anchor's calibrated band "
                           f"[{lo}, {hi}]")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def combine_identity(*, layer1: schemas.Check, layer2: schemas.Check) -> IdentityVerdict:
    return IdentityVerdict(layer1_codes=layer1, layer2_waveform=layer2,
                           combined=_worst(layer1, layer2))


# =============================================================================
# P-TTS-2 — the human-independent intelligibility floor
# =============================================================================

#: The round-trip scorer's identity. It is pinned to a FROZEN PRODUCTION STT binary,
#: never to the STT champion: the two backends are independently freezable and
#: independently researched, and an oracle that is itself under optimization
#: confounds every reading taken through it.
INTELLIGIBILITY_INSTRUMENT_FIELDS = (
    "stt_binary_sha256", "stt_model_sha256", "stt_decode_parameters",
    "stt_normalizer_id", "stt_normalizer_sha256", "stt_binary_is_frozen_production",
)


def check_intelligibility_instrument(record: Mapping[str, Any]) -> schemas.Check:
    """The pinned-scorer contract of `P-TTS-2` §2.1.

    Three consequences of the pin, all enforced here:
      1. a change of STT instrument is an instrument-version boundary for TTS records
         (`MEASUREMENT.md:83-84`), so the identity must be carried to make the boundary
         detectable;
      2. the oracle MUST be the frozen production STT kernel, never the champion; and
      3. a `roundtrip_wer` regression is ambiguous until the STT instrument is proven
         unchanged, which is only possible if it was recorded.
    """
    if not isinstance(record, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("intelligibility instrument record is not a mapping",))
    missing = [f for f in INTELLIGIBILITY_INSTRUMENT_FIELDS if f not in record]
    if missing:
        return schemas.Check(schemas.FAIL, (
            f"the round-trip scorer's identity is missing {missing}; without it a "
            f"roundtrip_wer change cannot be attributed to the TTS kernel rather than to "
            f"the scorer (P-TTS-2 §2.1)",))
    for field in ("stt_binary_sha256", "stt_model_sha256", "stt_normalizer_sha256"):
        digest = record.get(field)
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            return schemas.Check(schemas.FAIL,
                                 (f"{field} is not a lowercase sha256: {digest!r}",))
        if schemas.is_placeholder_digest(digest):
            return schemas.Check(schemas.FAIL, (
                f"{field} {digest!r} is a well-formed digest no measurement produced",))
    frozen = record.get("stt_binary_is_frozen_production")
    if frozen is not True:
        return schemas.Check(schemas.FAIL, (
            "the round-trip scorer is not a frozen production STT binary. A campaign "
            "MUST NOT simultaneously advance the whisper_stt champion and use it as the "
            "TTS oracle; the oracle is pinned to the frozen production kernel "
            "(P-TTS-2 §2.1 clause 2)",))
    return schemas.Check(schemas.PASS)


def derive_intelligibility_floor(*, anchor_roundtrip_wer_pct: float,
                                 aa_dispersion_pp: float) -> float:
    """`floor = anchor_roundtrip_wer + A/A dispersion`. Derived, never a round number.

    Where the anchor is saturated at 0.0 % — which the 2026-07-31 CPU Q8_0 pair was,
    word-perfect — the floor is the A/A dispersion alone, and the campaign records
    that the floor's resolution is bounded by the corpus size.
    """
    anchor = _require_non_negative(anchor_roundtrip_wer_pct, "anchor_roundtrip_wer_pct")
    dispersion = _require_non_negative(aa_dispersion_pp, "aa_dispersion_pp")
    return anchor + dispersion


def check_intelligibility(*, roundtrip_wer_pct: float, floor_pct: float,
                          anchor_roundtrip_wer_pct: float,
                          spectral_distance: Optional[float],
                          spectral_band: Optional[Sequence[float]]) -> schemas.Check:
    """The floor gate PLUS the non-saturated companion. Neither alone is sufficient.

    Round-trip WER saturates (the anchor measured 0.0 %) and is gameable in exactly
    the direction control 3 exists to catch — a cached waveform or a flat robotic
    monotone scores BETTER while being worse speech. So it gates and never ranks, and
    a second human-independent signal that does not saturate is REQUIRED alongside.
    """
    value = _require_non_negative(roundtrip_wer_pct, "roundtrip_wer_pct")
    floor = _require_non_negative(floor_pct, "floor_pct")
    anchor = _require_non_negative(anchor_roundtrip_wer_pct, "anchor_roundtrip_wer_pct")
    if spectral_distance is None or spectral_band is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the non-saturated companion (reference-waveform spectral distance) was not "
            "supplied. The round-trip floor alone leaves the campaign blind to exactly "
            "the vocoder degradations an ASR front end is built to be robust to "
            "(P-TTS-2 §2.2)",))
    reasons = []
    if value > floor:
        reasons.append(f"roundtrip_wer {value} % exceeds its derived floor {floor} % "
                       f"(anchor {anchor} %)")
    distance = _require_non_negative(spectral_distance, "spectral_distance")
    band = list(spectral_band)
    if len(band) != 2:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (f"spectral_band must be a [lo, hi] pair, got {band!r}",))
    lo, hi = float(band[0]), float(band[1])
    if lo > hi:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (f"spectral_band {band!r} is inverted",))
    if not (lo <= distance <= hi):
        reasons.append(f"spectral_distance {distance} is outside the anchor's derived "
                       f"band [{lo}, {hi}]")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def saturation_label(anchor_roundtrip_wer_pct: float, aa_dispersion_pp: float) -> str:
    """`saturated` when the anchor is at or inside its own noise of the floor.

    A metric at its ceiling cannot detect improvement and detects only large
    regressions (`feedback_eval_saturation_masks_model_gap`), so the record must SAY
    so rather than let a reader infer parity from a passing gate.
    """
    anchor = _require_non_negative(anchor_roundtrip_wer_pct, "anchor_roundtrip_wer_pct")
    dispersion = _require_non_negative(aa_dispersion_pp, "aa_dispersion_pp")
    return "saturated" if anchor <= dispersion else "unsaturated"


# =============================================================================
# P-TTS-3 — stage attribution; P-STT-3 — op-coverage integrity
# =============================================================================


#: A stage-attribution tolerance may not exceed this fraction of the total it is a
#: tolerance ON. The bound is what makes the check a check: at `tolerance >= total`,
#: "the parts sum to the whole" is satisfied by parts of ZERO, so an attribution
#: that accounts for none of the wall time passes. 1 % is the ceiling because the
#: 2026-07-31 CPU->GPU move left the SMALLEST declared stage at 10.4 % of wall
#: (codec_decode, down from 64 %), so a 1 % slack keeps every real stage an order of
#: magnitude above the noise it must be distinguished from.
MAX_TOLERANCE_FRACTION_OF_TOTAL = 0.01


@dataclass(frozen=True)
class StageTimingTolerance:
    """The harness's timer resolution, times the number of stages it timed.

    A tolerance is a property of the instrument, so it is CONSTRUCTED from the
    instrument's resolution rather than typed as a number — `derive_stage_tolerance`
    is the only sanctioned constructor and it takes the stage count from
    `STAGE_PHASES`, not from the caller, so the slack cannot be widened by claiming
    more stages than the protocol declares.
    """

    timer_resolution_ms: float
    stage_count: int

    def __post_init__(self) -> None:
        _require_non_negative(self.timer_resolution_ms, "timer_resolution_ms")
        if self.stage_count != len(STAGE_PHASES):
            raise QwenTtsAdapterError(
                f"stage_count is {self.stage_count!r} but this backend declares "
                f"{len(STAGE_PHASES)} stages {list(STAGE_PHASES)}; a tolerance scaled by "
                f"a stage count the protocol does not have is slack invented by the caller")

    @property
    def value_ms(self) -> float:
        return float(self.timer_resolution_ms) * self.stage_count

    def to_dict(self) -> dict:
        return {"timer_resolution_ms": self.timer_resolution_ms,
                "stage_count": self.stage_count, "value_ms": self.value_ms}


def derive_stage_tolerance(*, timer_resolution_ms: float) -> StageTimingTolerance:
    """`tolerance = timer resolution x declared stage count`. Derived, never typed."""
    return StageTimingTolerance(
        timer_resolution_ms=_require_non_negative(timer_resolution_ms,
                                                  "timer_resolution_ms"),
        stage_count=len(STAGE_PHASES))


def check_stage_attribution(stage_ms: Mapping[str, float], *, total_ms: float,
                            tolerance: StageTimingTolerance) -> schemas.Check:
    """Every declared stage present, and the parts summing to the whole.

    `tolerance` is a `StageTimingTolerance` — the harness's own timer resolution
    times the declared stage count — and it is BOUNDED against the measurement it is
    a tolerance on. An unbounded tolerance is not a loose check, it is no check: a
    tolerance at or above `total_ms` passes an attribution whose stages sum to zero,
    and one at half the total passes an attribution that loses half the wall clock.
    Over `MAX_TOLERANCE_FRACTION_OF_TOTAL` of `total_ms` the result is
    COULD_NOT_CHECK — the instrument is too coarse for this measurement — never PASS.

    An unaccounted remainder inside the bound is still a finding: it means wall time
    is being spent somewhere no stage names.
    """
    if not isinstance(stage_ms, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK, ("stage_ms is not a mapping",))
    missing = [s for s in STAGE_PHASES if s not in stage_ms]
    if missing:
        return schemas.Check(schemas.FAIL, (
            f"stage attribution omits {missing}; an end-to-end RTF hides which stage "
            f"moved, and on 2026-07-31 the bottleneck moved from codec_decode (64 % -> "
            f"10.4 %) to code_predictor (-> 65.5 %) (P-TTS-3)",))
    unknown = sorted(set(stage_ms) - set(STAGE_PHASES))
    if unknown:
        return schemas.Check(schemas.FAIL, (
            f"stage attribution names stages outside the declared vocabulary: {unknown}; "
            f"declared stages are {list(STAGE_PHASES)}",))
    total = _require_non_negative(total_ms, "total_ms")
    if not isinstance(tolerance, StageTimingTolerance):
        raise QwenTtsAdapterError(
            f"tolerance must be a StageTimingTolerance, got "
            f"{type(tolerance).__name__} ({tolerance!r}). A bare number is unbounded: a "
            f"tolerance at or above the total passes an attribution whose stages sum to "
            f"zero. Use derive_stage_tolerance(timer_resolution_ms=...)")
    tol = tolerance.value_ms
    if total <= 0.0:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"total_ms is {total}; a tolerance has nothing to be bounded against and the "
            f"parts-versus-whole comparison has no whole",))
    ceiling = total * MAX_TOLERANCE_FRACTION_OF_TOTAL
    if tol > ceiling:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the tolerance is {tol} ms ({tolerance.timer_resolution_ms} ms timer "
            f"resolution x {tolerance.stage_count} stages) against a total of {total} ms — "
            f"above the ceiling of {ceiling} ms "
            f"({MAX_TOLERANCE_FRACTION_OF_TOTAL:.0%} of the measurement). A tolerance that "
            f"large cannot discriminate: the smallest stage this protocol has ever "
            f"observed was 10.4 % of wall, and slack of this size would absorb it. The "
            f"instrument is too coarse for this measurement, which is not a PASS",))
    summed = 0.0
    for name in STAGE_PHASES:
        summed += _require_non_negative(stage_ms[name], f"stage_ms[{name!r}]")
    if summed > total + tol:
        return schemas.Check(schemas.FAIL, (
            f"stages sum to {summed} ms against a total of {total} ms; the parts exceed "
            f"the whole",))
    if summed < total - tol:
        return schemas.Check(schemas.FAIL, (
            f"stages sum to {summed} ms against a total of {total} ms; {total - summed} "
            f"ms is unaccounted for, so wall time is being spent where no stage names it",))
    return schemas.Check(schemas.PASS)


def check_op_coverage(*, candidate_attempted: int, candidate_skipped: int,
                      candidate_passed: int, anchor_attempted: int,
                      skip_reasons: Optional[Sequence[str]],
                      anchor_skipped: Optional[int] = None) -> schemas.Check:
    """A pass count is meaningless without its enumeration (`P-STT-3`, applied here).

    The gfx90a ARGSORT defect was invisible while `test-backend-ops` reported
    `ARGSORT 46/46` and `TOP_K 170/170` — because the failing shapes were **silently
    skipped**. After the fix the same suite reported `74/74` and `292/292`. Both are
    "100 % pass"; only the enumeration distinguishes them. A candidate whose attempted
    count falls below the anchor's therefore FAILS at any pass rate.

    The attempted count alone does not bound coverage, which is why `anchor_skipped`
    is an input rather than an afterthought. `passed >= attempted - skipped` stays
    true no matter how many cases move into the skip bucket, so a candidate holding
    `attempted` at the anchor's number while skipping all of them reported PASS with
    ZERO cases passed. A skip is not a pass; a skip the anchor did not take is a
    coverage regression, and an unknown anchor skip count is COULD_NOT_CHECK.
    """
    for label, value in (("candidate_attempted", candidate_attempted),
                         ("candidate_skipped", candidate_skipped),
                         ("candidate_passed", candidate_passed),
                         ("anchor_attempted", anchor_attempted)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"{label} must be a non-negative int, got {value!r}",))
    if candidate_passed > candidate_attempted:
        return schemas.Check(schemas.FAIL, (
            f"candidate_passed {candidate_passed} exceeds candidate_attempted "
            f"{candidate_attempted}",))
    if candidate_attempted < anchor_attempted:
        return schemas.Check(schemas.FAIL, (
            f"the candidate attempted {candidate_attempted} cases where the anchor "
            f"attempted {anchor_attempted}. A shrinking enumeration is the signature of a "
            f"shape becoming unsupported and being silently dropped, which is "
            f"indistinguishable from a fix if only the pass RATIO is read "
            f"(gfx90a ARGSORT precedent: 46/46 -> 74/74)",))
    # The missing-anchor gate sits BELOW the findings the supplied inputs already
    # determine: a COULD_NOT_CHECK that masks a decidable FAIL is the same defect
    # in the other direction.
    if anchor_skipped is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the anchor's skip count was not supplied, so the candidate's skipped set "
            "cannot be compared with the anchor's. Attempted-count parity does not "
            "bound coverage on its own: moving N previously-passing cases into the skip "
            "bucket keeps `passed >= attempted - skipped` true at any pass rate",))
    if isinstance(anchor_skipped, bool) or not isinstance(anchor_skipped, int) \
            or anchor_skipped < 0:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (f"anchor_skipped must be a non-negative int, got "
                              f"{anchor_skipped!r}",))
    if candidate_skipped > anchor_skipped:
        return schemas.Check(schemas.FAIL, (
            f"the candidate skipped {candidate_skipped} cases where the anchor skipped "
            f"{anchor_skipped}. Those {candidate_skipped - anchor_skipped} cases RAN on "
            f"the anchor and did not run here: a skip is not a pass, and this regression "
            f"is invisible to both the pass ratio and the attempted count, which is the "
            f"gfx90a ARGSORT shape exactly",))
    if candidate_passed < candidate_attempted - candidate_skipped:
        return schemas.Check(schemas.FAIL, (
            f"{candidate_attempted - candidate_skipped - candidate_passed} attempted, "
            f"non-skipped cases did not pass",))
    if candidate_skipped:
        reasons = list(skip_reasons or [])
        if len(reasons) != candidate_skipped:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{candidate_skipped} cases were skipped but {len(reasons)} skip reasons "
                f"were reported; a skip whose reason the harness does not report is "
                f"COULD_NOT_CHECK for that op, and the coverage gap is journaled",))
    return schemas.Check(schemas.PASS)


# =============================================================================
# Protocol and release-compiler bindings
# =============================================================================

SEARCH_PROTOCOL_ID = "P-AK-SEARCH-1"

#: `P-STT-3` is listed deliberately: TTS stability and op-coverage integrity are
#: governed by it rather than duplicated into a `P-TTS-4` (where a rule already
#: lives, the amendment goes — `kernel-research.md:22-23`). It is therefore a
#: RELEASE dependency of this backend.
RELEASE_PROTOCOL_IDS = ("P-TTS-1", "P-TTS-2", "P-TTS-3", "P-TTS-REL-1", "P-STT-3")
RELEASE_PROTOCOL_LOCATOR = "measurement/protocols/speech.md"


def release_gate_readiness(ratified_protocol_ids: Collection[str]) -> schemas.Check:
    """Is this backend's release path legally runnable yet?

    `ratified_protocol_ids` is SUPPLIED, never baked in: the source of truth is the
    registry in `MEASUREMENT.md` §2, and a constant here would go stale silently the
    moment the operator ratified — or declined — the family.

    Returns COULD_NOT_CHECK, **never PASS**, while any required protocol is missing.
    P-AK-SEARCH-1 denial 6 in adapter form: record the gap, block release eligibility,
    continue unrelated research, do not patch the instrument and do not route around
    it.
    """
    ratified = {_require_str(p, "ratified_protocol_id") for p in ratified_protocol_ids}
    if SEARCH_PROTOCOL_ID not in ratified:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{SEARCH_PROTOCOL_ID} is not in the supplied ratified set, so not even T0-T2 "
            f"search is authorized on this backend",))
    missing = [p for p in RELEASE_PROTOCOL_IDS if p not in ratified]
    if missing:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the {BACKEND} release protocols {missing} are absent from the supplied "
            f"ratified registry; Annex S is at {RELEASE_PROTOCOL_LOCATOR} (P-STT-3 is "
            f"the cross-family integrity dependency). Search under "
            f"{SEARCH_PROTOCOL_ID} remains legal and candidates may be banked; release "
            f"eligibility is BLOCKED for this lineage until the operator ratifies or "
            f"declines the family",))
    return schemas.Check(schemas.PASS)


def release_binding(*, protocols: Mapping[str, release_plan.PhaseProtocol],
                    ratified_protocol_ids: Collection[str],
                    stt_instrument: Optional[Mapping[str, Any]] = None,
                    traversed_submodules: Collection[str] = ()) -> release_plan.BackendBinding:
    """Build the TTS release binding; missing pinned evidence remains a failed check."""
    for phase, protocol in protocols.items():
        if not isinstance(protocol, release_plan.PhaseProtocol):
            raise QwenTtsAdapterError(f"protocol {phase!r} is not a PhaseProtocol")
        if protocol.protocol_id != "P-TTS-3":
            raise QwenTtsAdapterError(
                f"phase {phase!r} must be measured under P-TTS-3, got "
                f"{protocol.protocol_id!r}")
        expected_direction = metric_direction(protocol.metric)
        if protocol.direction != expected_direction:
            raise QwenTtsAdapterError(
                f"metric {protocol.metric!r} is {expected_direction}, not "
                f"{protocol.direction}")
    ceiling = complexity_ceiling()
    return release_plan.BackendBinding(
        backend=BACKEND,
        stable_production_path=STABLE_PATH,
        production_tree_path=PRODUCTION_TREE_ROOT,
        binary_roots=(STABLE_PATH, STABLE_TARGET),
        phases=PHASES,
        protocols=dict(protocols),
        prerequisites={
            "ratified_protocol_registry": release_gate_readiness(ratified_protocol_ids),
            "pinned_stt_instrument": check_intelligibility_instrument(
                {} if stt_instrument is None else stt_instrument),
            "source_closure_submodules": check_closure_traversed_submodules(
                traversed_submodules),
        },
        linkage=release_plan.LinkageRequirement(
            source_tree=SOURCE_TREE,
            ggml_generation=GGML_GENERATION,
            required_ld_library_path=(STABLE_TARGET,),
        ),
        ceiling=release_plan.ComplexityCeiling(
            max_diff_lines=ceiling.max_diff_lines,
            max_files_touched=ceiling.max_files_touched,
            shared_core_requires_review=ceiling.shared_core_modification_requires_review,
        ),
        co_residency_required=False,
        canary_required=True,
    )


# =============================================================================
# Self-audit
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

    The supplied text is BOUND to this module before a non-FAIL result is returned.
    The evaluator's own auditor anchors itself with `Path(__file__).read_text()`; this
    one cannot (it reads no file), so without a binding
    `audit_no_write_or_process_paths("")` returns PASS — the guarantee obtained by
    deleting the thing it inspects. A FAIL is returned unbound, because a forbidden
    construct is a finding about the text whoever the text belongs to.
    """
    if source is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no source supplied; this module reads no file, so the caller passes the "
            "module text (test_qwentts_tts.py does)",))
    if not isinstance(source, str):
        raise QwenTtsAdapterError("source must be a string")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (f"could not parse module: {exc}",))
    result = api.audit_no_write_or_process_paths(source, module_id=MODULE_ID)
    if result.outcome == schemas.FAIL:
        return result
    if not _source_is_this_module(tree):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the supplied source does not define this module's identity (BACKEND = "
            f"{BACKEND!r} plus {list(_AUDIT_IDENTITY_FUNCTIONS)}), so the AST audited is "
            f"not this adapter's. A clean audit of text nobody bound to the module — the "
            f"empty string passes every rule — is not evidence about the module",))
    return result


def audit_device_vocabulary_delegation(source: Optional[str] = None) -> schemas.Check:
    """Prove from THIS module's AST that it holds no device vocabulary of its own.

    Same shape and same reason as the audit above, delegating to
    `evaluator/devices.py` so the rule is stated once for both speech adapters. It
    returns COULD_NOT_CHECK on empty, unparsable or foreign source, and FAILs both on
    a local device-name table and on a `check_device_evidence` that decides what a
    device name denotes without asking the shared vocabulary.
    """
    return devices.audit_delegates_device_vocabulary(
        source, expected_backend=BACKEND,
        checker_name="check_device_evidence",
        identity_functions=_AUDIT_IDENTITY_FUNCTIONS)
