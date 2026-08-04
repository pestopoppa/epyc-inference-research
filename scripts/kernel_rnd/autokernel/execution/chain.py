#!/usr/bin/env python3
"""chain.py — the seams BETWEEN the five executors and the evaluator that reads them.

WHAT THIS MODULE IS FOR
-----------------------
Five agents built five executors against one evaluator, in parallel, in one
hour. Each is internally sound. What none of them could check is the place where
one hands its output to the next: `worktree.py` emits a build receipt,
`correctness.py` consumes a *different* record with the same name, and nothing in
either module has ever seen the other's object. That gap is not a style problem.
It is where a campaign driver ends up hand-writing an evidence record — and a
hand-written evidence record is exactly the artifact the whole T0 gate exists to
refuse.

So this module holds the projections, and only the projections. It launches
nothing, spawns nothing, and hashes only files it is told to hash.

THE FOUR SEAMS, AND WHY EACH ONE IS A REAL DEFECT AND NOT A CONVENIENCE
----------------------------------------------------------------------

**Seam 1 — two `BuildProvenance` classes.** `worktree.BuildIdentity` has a
`to_build_provenance()` that returns an `integrity.BuildProvenance`.
`correctness.T0Evidence.build` requires a `correctness.BuildProvenance`. They
share a name and not one field name:

    integrity                          correctness
    ---------                          -----------
    snapshot_sha256                    built_from_snapshot_sha256
    build_dir_created_for_this_build   build_dir_was_fresh
    (build_dir_pre_build_digest)       incremental_objects_present
    compiler                           compiler_id + compiler_version
    build_log_path                     build_log_ref
    production_tree_paths              production_tree_paths_touched   <-- !!
    output_binary_sha256               output_binary_sha256

The last row is the dangerous one, and it is why this projection is a function
with tests rather than a paragraph in a runbook. `BuildIdentity.production_tree_paths`
is the **denylist** — every frozen tree, always non-empty, and `build_identity()`
forces the frozen set into it precisely so the receipt cannot carry a shrunken
one. `correctness.BuildProvenance.production_tree_paths_touched` is the
**violation list**, and `check_clean_build_from_snapshot` reads it as:

    if touched: reasons.append("the build touched production tree path(s) …")

Copying one into the other by name — the obvious thing a driver does at 2am —
FAILS every T0 gate on every candidate forever, with a reason that reads like a
frozen-tree violation. Copying `()` in instead makes the sub-check vacuous. The
correct value is neither: it is the subset of the denylist that actually contains
one of this build's own paths, computed here, once.

`build_dir_was_fresh` is derived from `build_dir_pre_build_digest ==
integrity.EMPTY_TREE_SHA256`, NOT from `build_dir_created_for_this_build`. The
latter says "we called mkdir"; the former says "there was nothing in it". The
gate's stated concern is *"an incremental build can link stale objects"*, which
is a fact about contents.

**ccache.** `build_identity()` already notes that ccache being active means *"a
fresh build directory does not by itself make this a clean build"*. There is no
field in `correctness.BuildProvenance` that says "the objects came from a cache
outside this directory", so this projection folds it into
`incremental_objects_present`, which is the field whose gate reason is *"the
actor's build state is part of the artifact"* — true of a ccache hit, and the
verdict lands FAIL rather than PASS. `BuildEvidence.checks` then carries the
precise reason the dataclass cannot.

**Seam 2 — the artifact digests must NOT be projected from the receipt.**
`check_clean_build_from_snapshot` compares `evidence.built_from_snapshot_sha256`
against `request.artifact.source_sha256`, and `evidence.output_binary_sha256`
against `request.artifact.binary_sha256`. A helper that filled both sides from
`BuildIdentity` would turn two of that gate's four sub-checks into `x == x`.
This module therefore offers `measure_artifact_identity()`, which goes to the
FILESYSTEM: it re-hashes the source root and re-hashes the binary. The gate then
compares a receipt written at build time against a measurement taken at
evaluation time, which is a comparison that can fail — and `test_execution_chain`
shows it failing.

**Seam 3 — the anchor, and the field that cannot hold it.** T0 measures the
anchor into a `t0_provider.AnchorCapture`; T1 is planned against an
`api.AnchorIdentity` in `microbench.MicrobenchPlan.anchor`; the record carries a
third in `api.EvaluationRequest.anchor`. `AnchorCapture.identity()` projects one
way and nothing asserted the three agree — a campaign can produce a T0 report
about anchor A and a T1 effect against anchor B, and `api.compute_verdict` will
combine them, because it checks that an anchor is BOUND and not that two stages
bound the same one.

Chasing that down surfaced the harder half. **T0 and T1 do not run the same
binary.** `capture_anchor` hashes the anchor `llama-cli`; `microbench` compares
`plan.anchor.binary_sha256` against the digest of the anchor `llama-bench` it is
about to spawn. `api.AnchorIdentity.binary_sha256` is single-valued, so one
triple cannot name both — bind T0's and T1 refuses the digest (correctly, it is a
different file); bind T1's and the T0 evidence names a binary T0 never executed.
So `bind_anchor(capture, tool=…)` forces the tool onto the binding,
`check_anchor_matches` compares one tool's consumers, and
`check_anchor_build_is_one_build` enforces what genuinely must hold across tools:
the same `source_commit` and the same `linkage_sha256`.

`api.AnchorIdentity` now carries `tool` as well (2026-08-04), so the half of this
seam that lived only on the binding no longer evaporates at `.identity`: two
identities derived from ONE capture for two tools used to compare PASS, which
made the record's single-valued digest silently reusable across binaries. The
rule the field enforces — *`binary_sha256` is the digest of the tool the record's
`metric` was measured with* — is argued on `api.AnchorIdentity`, including why it
is a name and not the per-backend digest table
`controller.state_machine.AnchorIdentity` carries.

**Seam 4 — one claim, two Protocols.** `t0_provider.HeldClaim` wants
`claim_id` + one of `verify_held`/`is_held`/`held`. `microbench.HeldClaim` wants
`claim_id` + `attest()`, and `CpuRegionClaim` does not have `attest()` — it needs
`microbench.CpuRegionClaimAdapter`. A driver that passes the raw claim to the
microbench runner gets a `TypeError` an hour into a claim window.
`bind_claim()` returns both bindings from one acquisition and
`check_claim_satisfies_both_seams()` proves the real class satisfies both.

**Seams 5-8 — the four T0 surfaces whose PRODUCER already existed and was not
wired.** `t0_provider` accepts `symbols`, `diff`, `build` and a change surface as
pass-through inputs and produces none of them; `integrity.py` derives all of them
from real ELF tables and real diffs, and `surface.py` derives the affected
surface. Nothing joined the two, so a candidate came out of T0 with eight PASS
and nine COULD_NOT_CHECK — and four of those nine were unevaluated only because
no line of code carried `integrity`'s output across to `correctness`'s shape.
`symbol_evidence`, `diff_policy_evidence`, `anchor_toolchain_from_build_log` and
`change_surface_from` are those four lines, with the refusals that stop each one
from becoming a producer of clean-shaped nothing:

  * `symbol_evidence` refuses an EMPTY anchor exported surface (a diff of nothing
    against nothing has no removals and reads as a clean PASS) and requires the
    registration tables, because `SymbolTableDiff.removed_op_registrations = ()`
    from an extractor that was never run is the fail-open `integrity`'s own
    `PatternRegistrationExtractor` refuses at its constructor;
  * `diff_policy_evidence` DERIVES `commit_was_pathspec_limited` from the commit
    argv rather than taking it as a boolean;
  * `change_surface_from` emits `True` or `None` for the behavioural flags and
    **never `False`** — see its docstring;
  * `anchor_toolchain_from_build_log` MEASURES the anchor toolchain out of the
    anchor's own build log instead of letting a driver type it in.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No campaign driver. Composing these into a loop is the controller's job
(`controller/state_machine.py` owns the state walk) and putting a second walk
here would give the loop two spellings. This module is adapters and checks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .. import schemas
from ..evaluator import api, correctness, integrity, surface as surface_module
from . import t0_provider, worktree

__all__ = [
    "ChainSeamError",
    "BuildProvenanceUnprojectable",
    "AnchorNotOneAnchor",
    "ClaimSeamUnsatisfied",
    "EmptyAnchorSurface",
    "BuildEvidence",
    "build_evidence",
    "measure_artifact_identity",
    "candidate_build_for",
    "AnchorBinding",
    "bind_anchor",
    "check_anchor_matches",
    "check_anchor_build_is_one_build",
    "require_anchor_matches",
    "ClaimBinding",
    "bind_claim",
    "check_claim_satisfies_both_seams",
    "production_trees_touched_by",
    "split_compiler_identity",
    "build_log_ref",
    # seam 5 — symbols and registrations
    "SymbolEvidence",
    "symbol_evidence",
    # seam 6 — the diff
    "DiffEvidence",
    "diff_policy_evidence",
    "commit_was_pathspec_limited",
    # seam 7 — the anchor toolchain
    "AnchorToolchain",
    "anchor_toolchain_from_build_log",
    # seam 8 — the change surface
    "BEHAVIOURAL_TOKENS",
    "ChangeSurfaceEvidence",
    "change_surface_from",
    "classify_behavioural_surface",
    "SEAM_NOTES",
]


class ChainSeamError(Exception):
    """A composition between two executors that cannot be made without inventing a fact."""


class BuildProvenanceUnprojectable(ChainSeamError):
    """A build receipt does not carry what `correctness.BuildProvenance` requires."""


class AnchorNotOneAnchor(ChainSeamError):
    """Two stages of one campaign leg name different anchors."""


class ClaimSeamUnsatisfied(ChainSeamError):
    """A claim object does not satisfy a seam it is about to be passed through."""


class EmptyAnchorSurface(ChainSeamError):
    """The anchor binary exports no symbols, so a symbol diff cannot evidence anything.

    A raise and not an empty `SymbolTableDiff`: a diff of nothing against nothing
    has no removals, and the gate reads no removals as a PASS.
    `integrity.check_symbol_preservation` makes the same call and returns
    COULD_NOT_CHECK there; here the caller gets to decide, and passing
    `symbols=None` to the plan is how the gate reads COULD_NOT_CHECK.
    """


#: Recorded, not resolved: what a reader of this module needs to know that no
#: assertion here can carry.
SEAM_NOTES = (
    ("correctness.BuildProvenance has no field for 'the objects came from a compiler cache "
     "outside this build directory'. This module folds an active ccache into "
     "`incremental_objects_present`, which is the field whose gate reason is 'the actor's "
     "build state is part of the artifact'. That lands the right verdict with an imprecise "
     "reason. The precise reason is in BuildEvidence.checks and is lost at the dataclass "
     "boundary. REQUIRED FOLLOW-UP (correctness.py, another agent's file this hour): add "
     "`objects_from_external_cache: bool` and a FAIL branch naming it."),
    ("api.AnchorIdentity.binary_sha256 is SINGLE-VALUED, and one anchor build ships several "
     "binaries. T0 hashes the anchor `llama-cli`; `microbench` compares the plan's anchor "
     "digest against the anchor `llama-bench` it is about to spawn. One triple cannot name "
     "both truthfully, so a campaign leg that runs T0 and T1 needs TWO AnchorBindings and "
     "`check_anchor_build_is_one_build` to tie them together. CLOSED 2026-08-04 in api.py: of "
     "the two options recorded here, the second — `binary_sha256` names the tool the record's "
     "`metric` was measured with — is now ENFORCED rather than documented, by "
     "`AnchorIdentity.tool` and the tool branch of `identity_matches`. (Why not the per-tool "
     "table: `AnchorIdentity` is the denominator of ONE ratio; the table shape belongs to "
     "`controller.state_machine.AnchorIdentity`, which is the campaign-wide production "
     "identity. Argued on the class.) `AnchorBinding.identity` stamps the name with "
     "`for_tool`, so two bindings of one capture that differ only in their tool no longer "
     "compare PASS. The tie ACROSS tools is still this module's job and still this check."),
    ("execution/cpu_region_claim.py mints claim ids with the prefix 'akc-', which is the "
     "prefix `api.EvaluationRequest.__post_init__` requires of a CANDIDATE id. A claim id "
     "passed where a candidate id belongs therefore satisfies the one validator written to "
     "catch that class of mistake. CLOSED 2026-08-04: `_CLAIM_ID_PREFIX` is 'akclaim-' (the "
     "spelling t0_provider's own fixtures already used), and that module now refuses AT "
     "IMPORT TIME to mint claim ids in a namespace that overlaps the candidate one, so the "
     "two id kinds cannot be re-merged by a later edit without the package failing to load."),
    ("correctness.SymbolTableDiff carries ANCHOR-DERIVED material — `anchor_symbol_count` is "
     "a count of the ANCHOR binary's exported surface, and every removal is a removal "
     "RELATIVE TO IT — and it has NO anchor triple fields. Every other anchor-bearing "
     "evidence type in correctness.py names its anchor by "
     "`anchor_source_commit`/`anchor_binary_sha256`/`anchor_linkage_sha256` and is validated "
     "by `_validate_anchor_triple`; this one names nothing, so a symbol diff taken against "
     "anchor A reaches a gate holding anchor B with no way to notice — the exact defect "
     "`EvidenceAnchorMismatch` exists for on four other surfaces. `symbol_evidence` closes "
     "it from THIS side, structurally, by requiring an AnchorBinding and refusing to diff "
     "an anchor binary whose digest is not the one that binding measured; the record it "
     "produces still cannot say so. REQUIRED FOLLOW-UP (correctness.py): add the triple to "
     "SymbolTableDiff, validate it with `_validate_anchor_triple`, and add a "
     "`SymbolTableAnchorMismatch` branch to `check_symbol_and_registration_preservation`."),
    ("correctness.SymbolTableDiff had no field for a registration ARITY change, and "
     "`integrity.RegistrationDiff` produces two: `arity_changed` and `arity_not_comparable` "
     "(the latter meaning 'exactly one side's declared pattern captured an arity', which is "
     "NOT 'unchanged'). CLOSED 2026-08-04 for the first: `SymbolTableDiff."
     "arity_changed_op_registrations` exists, `check_symbol_and_registration_preservation` "
     "FAILs on any entry not in `declared_removals`, and `symbol_evidence` fills it from "
     "BOTH registries. STILL OPEN for the second: `arity_not_comparable` reaches no gate. It "
     "is a COULD_NOT_CHECK, not a FAIL — the declared extraction pattern captured an arity "
     "on one side only — and this record has no field that can carry a COULD_NOT_CHECK at "
     "all, so it lands in `SymbolEvidence.checks` and stops there."),
    ("NOTHING READS `SymbolEvidence.checks`, `DiffEvidence.checks` or "
     "`ChangeSurfaceEvidence.checks`, and nothing reads their `worst` property. The T0 plan "
     "takes the projected RECORD (`.diff`, `.policy`, `.surface`) and the wrapper is "
     "dropped, so every finding these projections make that `correctness.py` has no field "
     "for evaporates between here and the report. Today that is: registration "
     "`arity_not_comparable`; `elf_extraction_complete` (the ELF extractor reported a "
     "coverage gap, so the symbol diff is over an incompletely-read table, and the gate "
     "still says PASS); `diff_is_textual` (a binary blob in the diff contributes no changed "
     "lines, so the §10.6 change-class envelope does not bound it); and the three "
     "`derived_touches_*` UNDETERMINEDs. All four are COULD_NOT_CHECKs turning into "
     "silence, which is the direction that overstates. REQUIRED FOLLOW-UP: give the T0 plan "
     "a channel for projection-side checks — `_Collected.notes` is the existing one and it "
     "reaches the record — or make the plan take the wrappers."),
)


# =============================================================================
# Seam 1 — the build receipt -> the T0 build gate's evidence
# =============================================================================

def production_trees_touched_by(identity: worktree.BuildIdentity) -> tuple:
    """The frozen trees this build's OWN paths actually landed inside.

    The polarity fix, isolated so it can be tested on its own. `identity`
    carries a DENYLIST; this returns the subset of it that contains at least one
    of the build's paths, which is what
    `correctness.check_clean_build_from_snapshot` reads as a violation.

    Containment is `worktree._is_within` — component-wise over realpaths — so
    `/mnt/raid0/llm/llama.cpp-ak-0001` is not inside `/mnt/raid0/llm/llama.cpp`
    though its string starts with it, and a `..` traversal or a symlink into a
    frozen tree IS caught. A `str.startswith` test gets the first case wrong in
    the direction that blocks every legitimate campaign and the second case
    wrong in the direction that admits a frozen-tree build.

    The receipt's `library_sha256s` are `(name, DIGEST)` pairs and carry no
    paths, so the libraries are NOT re-tested here. They were tested once, at
    receipt time: `worktree.build_identity()` refuses a library outside the
    build directory. Saying so is the point — a reader of this call site would
    otherwise assume the libraries are covered by the list below.
    """
    if not isinstance(identity, worktree.BuildIdentity):
        raise TypeError("production_trees_touched_by takes a worktree.BuildIdentity")
    subjects = [identity.build_dir, identity.source_root, identity.actor_worktree,
                identity.output_binary_path, identity.build_log_path]
    touched: list = []
    for tree in identity.production_tree_paths:
        real_tree = worktree._real(tree, "production_tree_paths[]")
        for subject in subjects:
            if not subject:
                continue
            if worktree._is_within(worktree._real(subject, "build path"), real_tree):
                touched.append(tree)
                break
    return tuple(sorted(set(touched)))


_VERSION_RE = re.compile(r"^\d[\w.+~-]*$")


def split_compiler_identity(compiler: str) -> tuple:
    """`"CXX GNU 15.2.0"` -> `("CXX GNU", "15.2.0")`. RAISES when there is no version.

    `correctness.BuildProvenance` requires `compiler_id` AND `compiler_version`,
    both non-empty. `worktree.BuildIdentity.compiler` is one string. The split
    has to happen somewhere, and everywhere it is not a tested function it is a
    `.split()[-1]` that silently yields `("GNU", "GNU")` for the ASM row
    (`('ASM', 'GNU')` is real: it is in the recorded configure log in
    `testdata/`).

    There is no `"unknown"` fallback. `worktree.build_identity()` already
    refuses to write a receipt that cannot say what built it, for the stated
    reason that *"a receipt whose compiler field is 'unknown' passes every schema
    and answers nothing"*. Weakening that here would restore it one layer down.
    """
    text = str(compiler or "").strip()
    if not text:
        raise BuildProvenanceUnprojectable(
            "the build receipt carries no compiler identity, and "
            "correctness.BuildProvenance requires compiler_id and compiler_version")
    parts = text.split()
    if len(parts) >= 2 and _VERSION_RE.match(parts[-1]):
        return " ".join(parts[:-1]), parts[-1]
    raise BuildProvenanceUnprojectable(
        f"compiler identity {text!r} carries no version token, so compiler_version would "
        "have to be invented. cmake prints '-- The CXX compiler identification is GNU "
        "15.2.0' at CONFIGURE time only; a build log from a re-used cmake cache does not "
        "repeat it. Pass compiler_id= and compiler_version= explicitly, from the toolchain "
        "you actually invoked.")


def build_log_ref(identity: worktree.BuildIdentity) -> str:
    """A ref that names the log's CONTENT, not just its path.

    `correctness.BuildProvenance.build_log_ref` becomes the gate's
    `evidence_ref`, which is what a reader follows six weeks later. A bare path
    resolves to whatever is at that path then; the digest is what makes the
    dereference checkable.
    """
    return f"file://{identity.build_log_path}#sha256={identity.build_log_sha256}"


@dataclass(frozen=True)
class BuildEvidence:
    """The projected `correctness.BuildProvenance` plus what the dataclass cannot carry.

    `checks` is not decoration. Two facts the projection must act on have no
    field on the far side — an active ccache and a build directory that was not
    empty — and both are folded into one boolean there. A reader that wants to
    know WHICH of them fired reads `checks`.
    """

    provenance: correctness.BuildProvenance
    checks: tuple
    notes: tuple = ()

    @property
    def worst(self) -> schemas.Check:
        return _worst(check for _name, check in self.checks)

    def to_dict(self) -> dict:
        return {
            "provenance": {
                "built_from_snapshot_sha256":
                    self.provenance.built_from_snapshot_sha256,
                "build_dir": self.provenance.build_dir,
                "build_dir_was_fresh": self.provenance.build_dir_was_fresh,
                "incremental_objects_present":
                    self.provenance.incremental_objects_present,
                "compiler_id": self.provenance.compiler_id,
                "compiler_version": self.provenance.compiler_version,
                "build_log_ref": self.provenance.build_log_ref,
                "production_tree_paths_touched":
                    list(self.provenance.production_tree_paths_touched),
                "output_binary_sha256": self.provenance.output_binary_sha256,
            },
            "checks": [[name, {"outcome": c.outcome, "reasons": list(c.reasons)}]
                       for name, c in self.checks],
            "notes": list(self.notes),
        }


def _worst(checks) -> schemas.Check:
    """Delegates to `schemas.Check.worst_of`. Two behaviours change, both closing holes.

    An evidence record built with `checks=()` derived PASS — and `campaign.py`
    reads `build_ev.worst.outcome != schemas.PASS` to decide whether to abort the
    leg, so an evidence record that carried no checks licensed the run. It is now
    COULD_NOT_CHECK.

    Reasons attached to a PASS sub-check are no longer carried; every producer in
    this module emits bare `Check(schemas.PASS)`, so nothing is lost, and the
    reasons that remain are now prefixed with the outcome that raised them.
    """
    return schemas.Check.worst_of(checks)


def build_evidence(identity: worktree.BuildIdentity, *,
                   compiler_id: Optional[str] = None,
                   compiler_version: Optional[str] = None) -> BuildEvidence:
    """Project a `worktree.BuildIdentity` into the T0 build gate's evidence.

    Seam 1. Every derivation and every refusal is argued in the module
    docstring; the short version is that four of this record's nine fields are
    NOT a rename of a field on the other side, and one of them
    (`production_tree_paths_touched`) inverts.

    `compiler_id`/`compiler_version` override the split of
    `identity.compiler`, for the case where the build log did not carry a
    version. They are the caller's attestation and are recorded as such.
    """
    if not isinstance(identity, worktree.BuildIdentity):
        raise TypeError("build_evidence takes a worktree.BuildIdentity")

    checks: list = []
    notes: list = []

    fresh = identity.build_dir_pre_build_digest == integrity.EMPTY_TREE_SHA256
    if fresh:
        checks.append(("build_dir_empty_before_build", schemas.Check(schemas.PASS)))
    else:
        checks.append(("build_dir_empty_before_build", schemas.Check(
            schemas.FAIL,
            (f"{identity.build_dir!r} digested {identity.build_dir_pre_build_digest[:12]} "
             f"before the build, not the empty tree "
             f"({integrity.EMPTY_TREE_SHA256[:12]}); §8.5.1 (2) requires a FRESH build "
             "directory",))))

    ccache = bool(identity.log_facts.ccache_enabled)
    if ccache:
        checks.append(("no_external_object_cache", schemas.Check(
            schemas.FAIL,
            ("ccache was ACTIVE for this build: objects may have been served from a cache "
             "populated by another tree, so an empty build directory does not make this a "
             "clean build. correctness.BuildProvenance has no field for it, so it is "
             "reported through `incremental_objects_present` — see SEAM_NOTES[0]. Configure "
             "the candidate with -DGGML_CCACHE=OFF.",))))
    else:
        checks.append(("no_external_object_cache", schemas.Check(schemas.PASS)))

    if identity.log_facts.ggml_commit_dirty:
        checks.append(("snapshot_is_what_built", schemas.Check(
            schemas.FAIL,
            (f"the build reported ggml commit {identity.log_facts.ggml_commit!r}; the "
             "'-dirty' suffix means the tree that built had uncommitted changes, so the "
             "snapshot digest is not the thing that built",))))
    else:
        checks.append(("snapshot_is_what_built", schemas.Check(schemas.PASS)))

    # `BuildResult` computes this and `build_identity` puts it in `notes` as prose.
    # Re-derived here from the two facts rather than grepped out of the prose: a
    # check that reads a note is a check that a reworded note silently disables.
    succeeded_by_exit = identity.exit_code == 0
    if succeeded_by_exit != identity.log_facts.succeeded_by_log:
        checks.append(("exit_code_agrees_with_log", schemas.Check(
            schemas.FAIL,
            (f"exit code {identity.exit_code} says succeeded={succeeded_by_exit} but the "
             f"log says succeeded={identity.log_facts.succeeded_by_log}; the status may "
             "have come from a pipe, a wrapper or a '|| true' rather than the compiler "
             "(feedback_pipe_hazards)",))))
    else:
        checks.append(("exit_code_agrees_with_log", schemas.Check(schemas.PASS)))
    if not succeeded_by_exit:
        checks.append(("build_succeeded", schemas.Check(
            schemas.FAIL,
            (f"the build exited {identity.exit_code}; there is no candidate binary to "
             "evaluate. A compilation failure is a valuable outcome (§8.5) and is banked "
             "as one — it is never evidence about a kernel.",))))
    else:
        checks.append(("build_succeeded", schemas.Check(schemas.PASS)))

    touched = production_trees_touched_by(identity)
    checks.append(("no_production_tree_path", schemas.Check(schemas.PASS) if not touched
                   else schemas.Check(
                       schemas.FAIL,
                       (f"this build's own paths resolve inside frozen production tree(s) "
                        f"{list(touched)}",))))

    if compiler_id is not None or compiler_version is not None:
        if not (compiler_id and compiler_version):
            raise BuildProvenanceUnprojectable(
                "compiler_id and compiler_version are supplied together or not at all; "
                "half an override leaves the other half to be invented")
        cid, cver = str(compiler_id), str(compiler_version)
        notes.append(f"compiler identity supplied by the caller: {cid} {cver} "
                     f"(the receipt says {identity.compiler!r})")
    else:
        cid, cver = split_compiler_identity(identity.compiler)

    provenance = correctness.BuildProvenance(
        built_from_snapshot_sha256=identity.snapshot_sha256,
        build_dir=identity.build_dir,
        build_dir_was_fresh=fresh,
        # The disjunction, not the negation: see the module docstring.
        incremental_objects_present=(not fresh) or ccache,
        compiler_id=cid,
        compiler_version=cver,
        build_log_ref=build_log_ref(identity),
        production_tree_paths_touched=touched,
        output_binary_sha256=identity.output_binary_sha256,
        # Projected from a `worktree.BuildIdentity` this module read, not from
        # anything the candidate declared — the same attribution `symbol_evidence`
        # records above.
        produced_by="evaluator",
    )
    return BuildEvidence(provenance=provenance, checks=tuple(checks), notes=tuple(notes))


# =============================================================================
# Seam 2 — the record's artifact identity, MEASURED rather than copied
# =============================================================================

def measure_artifact_identity(*, source_root: Any, binary: Any,
                              linkage_sha256: str,
                              exclude_dir_names: Sequence[str] = (".git",)) -> api.ArtifactIdentity:
    """Re-measure the candidate's three digests FROM DISK. Copies nothing.

    Seam 2, and the reason it is a measurement and not a projection:
    `correctness.check_clean_build_from_snapshot` compares

        evidence.built_from_snapshot_sha256 == request.artifact.source_sha256
        evidence.output_binary_sha256       == request.artifact.binary_sha256

    Both left-hand sides come off the build receipt. If both right-hand sides
    came off the same receipt, those two sub-checks would read `x == x` and the
    gate would be two-thirds theatre. Here the right-hand sides are taken by
    walking the source root and hashing the binary at evaluation time, so the
    comparison answers the question it is written to ask: *is the binary under
    test the binary this source built?*

    `linkage_sha256` is NOT measured here. It comes from
    `epyc-inference-research/scripts/utils/verify_ggml_linkage.sh`, which
    `t0_provider.collect_linkage` runs; `worktree.BuildIdentity` records the same
    refusal to invent it. Three ggml generations live on this host and a binary
    that inherits another tree's ggml runs silently wrong, so a digest made up
    here would attest to a linkage nobody checked.
    """
    root = os.path.realpath(str(source_root))
    bin_path = os.path.realpath(str(binary))
    if not os.path.isdir(root):
        raise ChainSeamError(
            f"source_root {root!r} is not a directory; the artifact's source digest is "
            "measured by walking the tree, never taken on the receipt's word")
    if not os.path.isfile(bin_path):
        raise ChainSeamError(
            f"binary {bin_path!r} does not exist; refusing to name an artifact whose "
            "binary cannot be hashed")
    digest = integrity.hash_source_tree(root, exclude_dir_names=tuple(exclude_dir_names))
    return api.ArtifactIdentity(
        source_sha256=digest.sha256,
        binary_sha256=integrity.sha256_file(bin_path),
        linkage_sha256=linkage_sha256,
    )


def candidate_build_for(identity: worktree.BuildIdentity, *,
                        binary: Optional[str] = None,
                        test_backend_ops: Optional[str] = None,
                        build_log_ref_: Optional[str] = None) -> t0_provider.CandidateBuild:
    """`worktree.BuildIdentity` -> `t0_provider.CandidateBuild`, no field retyped.

    Safe as a pure re-spelling because no gate compares a `CandidateBuild` field
    against a `BuildIdentity` field — this record says WHERE to run, not WHAT
    was measured. `library_path` is forced to the binary's own directory, which
    is what `CandidateBuild` requires anyway and what stops the binary resolving
    another tree's libggml.

    `test_backend_ops` defaults to `<bindir>/test-backend-ops`, which is where
    the llama.cpp build puts it. Passing it explicitly is the escape hatch for a
    build that did not.
    """
    if not isinstance(identity, worktree.BuildIdentity):
        raise TypeError("candidate_build_for takes a worktree.BuildIdentity")
    bin_path = os.path.realpath(binary or identity.output_binary_path)
    bin_dir = os.path.dirname(bin_path)
    worktree_path = identity.worktree_record.get("path")
    if not worktree_path:
        raise ChainSeamError(
            "the build receipt's worktree record carries no path; a CandidateBuild names "
            "the tree the candidate was built from and there is nothing to name it with")
    return t0_provider.CandidateBuild(
        worktree=worktree_path,
        build_dir=identity.build_dir,
        source_commit=identity.worktree_record.get("source_commit"),
        source_sha256=identity.snapshot_sha256,
        binary=bin_path,
        library_path=bin_dir,
        test_backend_ops=os.path.realpath(
            test_backend_ops or os.path.join(bin_dir, "test-backend-ops")),
        build_log_ref=build_log_ref_ or build_log_ref(identity),
    )


# =============================================================================
# Seam 3 — one anchor, three consumers
# =============================================================================

@dataclass(frozen=True)
class AnchorBinding:
    """One anchor capture of ONE TOOL, and every consumer's copy derived from it.

    Three objects in this package carry an anchor triple and none of them
    imports the others:

      * `t0_provider.AnchorCapture` — MEASURED off the anchor binary;
      * `api.AnchorIdentity` in `api.EvaluationRequest.anchor` — what the record
        says the comparison was against;
      * `api.AnchorIdentity` in `microbench.MicrobenchPlan.anchor` — which the T1
        runner compares against the digest of the binary it is ABOUT TO RUN, and
        against the `build_commit` `llama-bench` itself prints.

    **`tool` exists because T0 and T1 do not run the same binary.** T0's anchor
    capture hashes the anchor `llama-cli`; T1's plan is checked against the
    anchor `llama-bench`. `api.AnchorIdentity.binary_sha256` is single-valued, so
    one triple cannot honestly name both: bind T0's and T1 refuses the digest
    (correctly — it is a different file); bind T1's and the T0 evidence names a
    binary T0 never executed. Recorded in `SEAM_NOTES`; made visible here by
    forcing the tool onto the binding, so the two are never silently the same
    object.

    What DOES have to agree across tools of one anchor build is the pair
    (`source_commit`, `linkage_sha256`) — same commit, same resolved ggml
    generation. `check_anchor_build_is_one_build` is that rule.
    """

    capture: t0_provider.AnchorCapture
    tool: str

    @property
    def identity(self) -> api.AnchorIdentity:
        """The `api.AnchorIdentity` for THIS tool. One expression, used by every consumer.

        `for_tool` is what keeps the tool from evaporating here. Before
        `api.AnchorIdentity` carried the name, this property returned a bare
        triple and two bindings that differed ONLY in their tool — the normal
        result of deriving T0's and T1's identity from one capture — produced
        identities that compared PASS. The binding knew which binary it named and
        the object every consumer actually reads did not.
        """
        return self.capture.identity().for_tool(self.tool)

    def check_against(self, other: Optional[api.AnchorIdentity], *,
                      label: str) -> schemas.Check:
        return self.identity.identity_matches(other) if other is not None else schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"{label} bound no anchor to compare against the captured one",))


def bind_anchor(capture: t0_provider.AnchorCapture, *, tool: str) -> AnchorBinding:
    """Wrap a measured capture so every consumer of THAT TOOL takes it from one place."""
    if not isinstance(capture, t0_provider.AnchorCapture):
        raise TypeError("bind_anchor takes a t0_provider.AnchorCapture — the MEASURED "
                        "triple. An api.AnchorIdentity is what a record SAYS; passing one "
                        "here would make the binding a restatement of the claim.")
    if not isinstance(tool, str) or not tool.strip():
        raise ValueError(
            "bind_anchor(tool=…) names the binary this capture hashed (e.g. 'llama-cli', "
            "'llama-bench'). It is required because the digest is tool-specific and the "
            "field it lands in is not.")
    return AnchorBinding(capture=capture, tool=tool.strip())


def check_anchor_matches(binding: AnchorBinding, *,
                         consumers: Mapping[str, Optional[api.AnchorIdentity]]) -> schemas.Check:
    """PASS only when every named consumer of ONE TOOL names that tool's anchor.

    A `None` on a consumer is COULD_NOT_CHECK, never agreement: an unbound anchor
    is precondition 4's own failure and must not be reported as "matches".
    """
    if not isinstance(binding, AnchorBinding):
        raise TypeError("check_anchor_matches takes an AnchorBinding")
    reasons: list = []
    outcome = schemas.PASS
    for name, other in sorted(consumers.items()):
        check = binding.check_against(other, label=name)
        if check.outcome == schemas.PASS:
            continue
        reasons.extend(f"{name}: {reason}" for reason in check.reasons)
        if check.outcome == schemas.FAIL:
            outcome = schemas.FAIL
        elif outcome != schemas.FAIL:
            outcome = schemas.COULD_NOT_CHECK
    if outcome == schemas.PASS:
        return schemas.Check(
            schemas.PASS,
            (f"every consumer of the {binding.tool} anchor names "
             f"{binding.identity.short()}",))
    return schemas.Check(outcome, tuple(reasons))


def check_anchor_build_is_one_build(bindings: Sequence[AnchorBinding]) -> schemas.Check:
    """Every tool's anchor came from ONE anchor build: same commit, same linkage.

    The binary digests are EXPECTED to differ — `llama-cli` and `llama-bench` are
    different files out of the same build — and this check does not compare them.
    What it refuses is the composition nobody would notice: a T0 anchor captured
    off `production-consolidated-v8` and a T1 anchor arm that happens to be a
    stale `llama-bench` from a previous build, which agree on nothing a
    single-valued digest field can express.
    """
    items = [b for b in bindings if isinstance(b, AnchorBinding)]
    if len(items) != len(tuple(bindings)):
        raise TypeError("check_anchor_build_is_one_build takes AnchorBindings")
    if not items:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("no anchor bindings were supplied to compare",))
    tools = [b.tool for b in items]
    if len(set(tools)) != len(tools):
        return schemas.Check(schemas.FAIL, (
            f"two bindings name the same tool {sorted(tools)}; a tool has one anchor "
            "binary and two captures of it are two answers to one question",))
    reasons: list = []
    first = items[0]
    for other in items[1:]:
        for field_name in ("source_commit", "linkage_sha256"):
            mine = getattr(first.capture, field_name)
            theirs = getattr(other.capture, field_name)
            if mine != theirs:
                reasons.append(
                    f"{first.tool}.{field_name}={mine[:12]} but "
                    f"{other.tool}.{field_name}={theirs[:12]}; tools of ONE anchor build "
                    "share a commit and a resolved ggml generation")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS, (
        f"{sorted(tools)} all come from anchor commit {first.capture.source_commit[:12]} "
        f"with linkage {first.capture.linkage_sha256[:12]}",))


def require_anchor_matches(binding: AnchorBinding, *,
                           consumers: Mapping[str, Optional[api.AnchorIdentity]]
                           ) -> api.AnchorIdentity:
    """`check_anchor_matches`, but a FAIL RAISES.

    Same rule as `correctness._refuse_replay_mismatch`, and for the same reason:
    two stages naming different anchors is a defect in the CAMPAIGN, not a
    property of the candidate. Recording it as a degraded gate would file the bug
    as a finding and leave it in place. COULD_NOT_CHECK does not raise — nothing
    disagrees — and the caller routes it wherever an unbound anchor goes.
    """
    check = check_anchor_matches(binding, consumers=consumers)
    if check.outcome == schemas.FAIL:
        raise AnchorNotOneAnchor(
            "this campaign leg names more than one anchor: " + "; ".join(check.reasons)
            + ". A T0 report about one anchor and a T1 effect against another compose into "
            "a verdict about neither.")
    return binding.identity


# =============================================================================
# Seam 4 — one claim object, two Protocols
# =============================================================================

@dataclass(frozen=True)
class ClaimBinding:
    """One acquired claim, bound for both consumers.

    `t0_for` is the raw claim: `t0_provider.require_claim` reads
    `verify_held()`/`is_held()`/`held` plus `covers()` itself.

    `microbench_for` is the ADAPTER: `microbench.MicrobenchRunner` calls
    `claim.attest()` before every invocation, `CpuRegionClaim` has no `attest`,
    and passing the raw claim there raises `TypeError` — an hour into a claim
    window, after the worktree is built.
    """

    claim: Any
    cpu_list: str
    microbench_claim: Any

    @property
    def t0_claim(self) -> Any:
        return self.claim

    @property
    def claim_id(self) -> str:
        return str(self.claim.claim_id)


def bind_claim(claim: Any, *, cpu_list: str) -> ClaimBinding:
    """Bind one acquired claim for both the T0 provider and the T1 runner.

    `cpu_list` must be the footprint the ARGV pins — read it off
    `recipes.CANONICAL_PREFIX` (or `recipes.ClaimFootprint.cpu_list`), never
    retyped. A claim over a smaller region answers `is_held()` exactly like a
    claim over the whole machine and the run underneath it is pinned either way.
    """
    from . import microbench  # noqa: PLC0415 - avoids a hard cycle; see module docstring

    if claim is None:
        raise ClaimSeamUnsatisfied(
            "bind_claim(None): denial 8 is 'no inference run OUTSIDE A HELD CLAIM'. There "
            "is no unclaimed binding, not even for a dry run — a dry run that can be "
            "upgraded to a real one by flipping a flag is the shape that runs unclaimed.")
    if not isinstance(cpu_list, str) or not cpu_list.strip():
        raise ClaimSeamUnsatisfied("bind_claim(cpu_list=…) must be the argv's own footprint")
    if hasattr(claim, "attest"):
        micro = claim
    else:
        micro = microbench.CpuRegionClaimAdapter(claim, cpu_list=cpu_list)
    return ClaimBinding(claim=claim, cpu_list=cpu_list, microbench_claim=micro)


def check_claim_satisfies_both_seams(claim: Any, *, cpu_list: str) -> schemas.Check:
    """Prove one claim object can serve both Protocols, without running anything.

    Structural: it interrogates shape and calls the two read-only predicates
    (`verify_held`/`is_held`/`held`, and `attest()` on the bound adapter). It
    does NOT launch a measurement, so it is safe to call in a preflight.
    """
    reasons: list = []
    outcome = schemas.PASS

    if not hasattr(claim, "claim_id"):
        return schemas.Check(schemas.FAIL, (
            f"{type(claim).__name__} has no claim_id; neither seam can name it",))

    try:
        held = t0_provider._claim_is_held(claim)
    except TypeError as exc:
        return schemas.Check(schemas.FAIL, (
            f"the T0 seam cannot ask this claim whether it is held: {exc}",))
    if not held:
        reasons.append(f"claim {claim.claim_id!r} reports itself NOT held")
        outcome = schemas.FAIL

    covers = getattr(claim, "covers", None)
    if not callable(covers):
        reasons.append(
            f"{type(claim).__name__} cannot state its footprint (`covers`), so precondition "
            f"1's 'covering the EXACT footprint measured' cannot be checked for {cpu_list!r}")
        if outcome != schemas.FAIL:
            outcome = schemas.COULD_NOT_CHECK
    elif not covers(cpu_list):
        reasons.append(f"claim {claim.claim_id!r} does not cover {cpu_list!r}")
        outcome = schemas.FAIL

    try:
        binding = bind_claim(claim, cpu_list=cpu_list)
        attestation = binding.microbench_claim.attest()
    except (ClaimSeamUnsatisfied, TypeError) as exc:
        reasons.append(f"the T1 seam cannot be bound: {exc}")
        return schemas.Check(schemas.FAIL, tuple(reasons))
    if attestation.check.outcome != schemas.PASS:
        reasons.append(
            f"the T1 attestation is {attestation.check.outcome}: "
            + "; ".join(attestation.check.reasons))
        outcome = schemas.FAIL if attestation.check.outcome == schemas.FAIL else (
            outcome if outcome == schemas.FAIL else schemas.COULD_NOT_CHECK)

    if outcome == schemas.PASS:
        reasons.append(
            f"claim {claim.claim_id!r} satisfies t0_provider.HeldClaim and, through "
            f"{type(binding.microbench_claim).__name__}, microbench.HeldClaim over "
            f"{cpu_list!r}")
    return schemas.Check(outcome, tuple(reasons))


# =============================================================================
# Seam 5 — integrity's ELF symbol table -> the T0 symbol/registration gate
# =============================================================================

@dataclass(frozen=True)
class SymbolEvidence:
    """The projected `correctness.SymbolTableDiff` plus what the dataclass cannot carry.

    Same shape as `BuildEvidence`, for the same reason: the far-side record has
    no field for several facts this projection establishes (a registration arity
    change, the extractor's own coverage notes, the anchor identity the diff was
    taken against), and a projection that drops them silently is how a coverage
    gap becomes a clean gate. `checks` is where they land; `SEAM_NOTES` records
    the schema follow-ups.
    """

    diff: correctness.SymbolTableDiff
    anchor_table: integrity.ElfSymbolTable
    candidate_table: integrity.ElfSymbolTable
    symbol_diff: integrity.SymbolDiff
    op_registration_diff: integrity.RegistrationDiff
    dispatch_predicate_diff: integrity.RegistrationDiff
    checks: tuple
    notes: tuple = ()

    @property
    def worst(self) -> schemas.Check:
        return _worst(check for _name, check in self.checks)


def _registration_ident(entry: Sequence[str]) -> str:
    """`(registry, key)` -> `"registry:key"`, the string form the record carries.

    `correctness.SymbolTableDiff.removed_op_registrations` is a tuple of strings
    and `integrity.RegistrationDiff.removed` is a tuple of `(registry, key)`
    pairs. Rendering it here, once, rather than at each call site, is the same
    reason `split_compiler_identity` is a function: the version of this that is
    not a function is a `[0]` somewhere that quietly drops the registry.
    """
    registry, key = entry[0], entry[1]
    return f"{registry}:{key}"


def _declared_covering(declared: integrity.DeclaredSymbolDeltas,
                       names: Sequence[str], *, which: frozenset) -> tuple:
    """The subset of `names` the declaration `which` actually covers.

    This exists because the two sides spell "declared" differently and the
    difference is silent. `integrity.DeclaredSymbolDeltas.covers` matches a
    removal by EXACT mangled name **or by its demangled qualified name**, so a
    proposal may declare `ggml::mul_mat` without predicting the mangling.
    `correctness.check_symbol_and_registration_preservation` does a plain
    `set(items) - set(evidence.declared_removals)` and has no demangler.

    Handing the raw `declared_symbol_deltas.removed` straight to the record
    would therefore FAIL every honestly-declared removal that was declared by
    qualified name — a gate that fails on correct input is a gate that gets
    switched off. Resolving the declaration against the names ACTUALLY EMITTED,
    here, keeps the exact-set-difference on the far side correct.

    ONLY `declared.removed` covers a removal, and that is the whole point of the
    `which` argument. Until the 2026-08-04 red team this function also accepted
    `declared.arity_changed` for a name in `SymbolDiff.removed`, and
    `SymbolDiff.removed` is by construction a removal with NO matching addition
    — `symbol_evidence` partitions the removal/addition pairs out into
    `signature_changes` before calling this. So a proposal that declared *"I will
    change the arity of `ggml::detail::kernel_dispatch`"* and whose candidate
    instead DROPPED that specialization outright came out of the projection with
    the name inside `declared_removals`, and
    `check_symbol_and_registration_preservation` PASSed it — indistinguishable
    from a candidate that honestly declared the removal. That is §8.5.1's own
    headline example ("a dropped template specialization ... compiles cleanly and
    silently changes behaviour for every shape nobody happened to test") arriving
    through the gate written to catch it.
    """
    covered: list = []
    for name in names:
        parsed = integrity.parse_mangled_name(name)
        if declared.covers(which, name, parsed):
            covered.append(name)
    return tuple(covered)


def symbol_evidence(*, anchor_binary: Any, candidate_binary: Any,
                    anchor: AnchorBinding,
                    declared: integrity.DeclaredSymbolDeltas,
                    anchor_op_registrations: integrity.RegistrationTable,
                    candidate_op_registrations: integrity.RegistrationTable,
                    anchor_dispatch_predicates: integrity.RegistrationTable,
                    candidate_dispatch_predicates: integrity.RegistrationTable,
                    max_bytes: Optional[int] = None) -> SymbolEvidence:
    """Seam 5: `integrity.extract_elf_symbols` -> `correctness.SymbolTableDiff`.

    The producer existed and was not wired. `t0_provider`'s docstring says why it
    is not wired THERE — *"producing them again here would create a second
    derivation of the §8.5.1 gates"* — which is right and leaves the join to this
    module, where every other cross-module projection lives.

    **Four inputs are mandatory and none of them has a safe default.**

    `anchor` is an `AnchorBinding`, not a path, and the anchor binary's digest
    must be the one that binding MEASURED. `SymbolTableDiff` has no anchor triple
    (SEAM_NOTES), so this is the only place the "which anchor is this diff
    against?" question can be answered at all; answering it by trusting the path
    would make it unanswerable. Diffing a different binary of the same anchor
    build (`libggml.so.0` rather than `llama-cli`) is legitimate and needs its own
    `bind_anchor(..., tool=…)` — the same rule seam 3 already enforces for T0 and
    T1.

    The four `RegistrationTable`s are mandatory because
    `removed_op_registrations=()` and `removed_dispatch_predicates=()` are what
    the record says when an extractor found nothing, and an extractor that was
    never constructed also finds nothing. `integrity.PatternRegistrationExtractor`
    refuses an empty pattern set at its own constructor for exactly this reason
    (*"an extractor with no patterns finds no entries because it looked for
    none"*); accepting `None` here would reintroduce it one layer up. A campaign
    that has not declared its registration patterns has no symbol evidence, and
    the gate reads COULD_NOT_CHECK — which is true.

    Raises `EmptyAnchorSurface` when the anchor exports nothing.
    """
    if not isinstance(anchor, AnchorBinding):
        raise TypeError(
            "symbol_evidence(anchor=…) takes an AnchorBinding — the MEASURED anchor "
            "capture for the tool being diffed. A path or an api.AnchorIdentity is what a "
            "record SAYS the anchor is; SymbolTableDiff carries no anchor triple, so a "
            "restatement here would be the only name this evidence ever gets.")
    if not isinstance(declared, integrity.DeclaredSymbolDeltas):
        raise TypeError(
            "symbol_evidence(declared=…) takes an integrity.DeclaredSymbolDeltas. "
            "`DeclaredSymbolDeltas.from_proposal` RAISES on an absent declaration rather "
            "than defaulting to empty, because an absent declaration is not an empty one "
            "— it is an undeclared removal waiting to happen (§7.2).")
    tables = {
        "anchor op-registration": (anchor_op_registrations, integrity.KIND_OP_REGISTRATION),
        "candidate op-registration": (candidate_op_registrations,
                                      integrity.KIND_OP_REGISTRATION),
        "anchor dispatch-predicate": (anchor_dispatch_predicates,
                                      integrity.KIND_DISPATCH_PREDICATE),
        "candidate dispatch-predicate": (candidate_dispatch_predicates,
                                         integrity.KIND_DISPATCH_PREDICATE),
    }
    for label, (table, kind) in tables.items():
        if not isinstance(table, integrity.RegistrationTable):
            raise TypeError(
                f"symbol_evidence needs the {label} table as an "
                f"integrity.RegistrationTable; `removed_{kind}s=()` from an extractor that "
                "was never run is indistinguishable from a clean one, which is the "
                "fail-open PatternRegistrationExtractor refuses at its own constructor")
        if table.kind != kind:
            raise ValueError(
                f"the {label} table is a {table.kind!r} table; a dispatch-predicate table "
                "diffed as an op-registration table compares two different registries and "
                "reports every entry of each as removed from the other")

    anchor_path = os.path.realpath(str(anchor_binary))
    candidate_path = os.path.realpath(str(candidate_binary))
    measured = integrity.sha256_file(anchor_path, max_bytes=max_bytes)
    if measured != anchor.capture.binary_sha256:
        raise AnchorNotOneAnchor(
            f"the symbol diff was asked to read anchor binary {anchor_path!r}, which hashes "
            f"to {measured[:12]}, but the {anchor.tool!r} anchor binding measured "
            f"{anchor.capture.binary_sha256[:12]}. SymbolTableDiff carries no anchor triple "
            "(SEAM_NOTES), so a diff against the wrong anchor binary is a finding about "
            "nothing and the record cannot say which anchor it meant. Bind the tool you are "
            "diffing: chain.bind_anchor(capture, tool=…).")

    anchor_table = integrity.extract_elf_symbols(anchor_path, label="anchor",
                                                 max_bytes=max_bytes)
    candidate_table = integrity.extract_elf_symbols(candidate_path, label="candidate",
                                                    max_bytes=max_bytes)
    diff = integrity.diff_symbol_tables(anchor_table, candidate_table)
    if diff.anchor_count == 0:
        raise EmptyAnchorSurface(
            f"the anchor binary {anchor_path!r} exports no symbols from .{anchor_table.preferred}. "
            "A diff of nothing against nothing has no removals and the gate reads no "
            "removals as a PASS, so this refuses instead. Pass symbols=None to the T0 plan "
            "if the surface genuinely cannot be read; the gate then says COULD_NOT_CHECK, "
            "which is what was established.")

    # A removal paired with an addition under the same qualified name is a
    # SIGNATURE change, not a removal, and reporting it as both would make one
    # edit two findings. `integrity.check_symbol_preservation` partitions it the
    # same way; this projection follows that partition rather than inventing a
    # second one.
    paired = {mangled for change in diff.signature_changes for mangled in change.removed}
    removed_symbols = tuple(name for name in diff.removed if name not in paired)
    arity_changed = tuple(sorted({change.qualified for change in diff.signature_changes}))

    op_diff = integrity.diff_registration_tables(anchor_op_registrations,
                                                 candidate_op_registrations)
    predicate_diff = integrity.diff_registration_tables(anchor_dispatch_predicates,
                                                        candidate_dispatch_predicates)
    removed_registrations = tuple(_registration_ident(i) for i in op_diff.removed)
    removed_predicates = tuple(_registration_ident(i) for i in predicate_diff.removed)

    # `declared_removals` is ONE flat set on the far side and the gate subtracts
    # it from three different item lists, so a name only ever excuses the
    # category it also appears in. That is what makes the per-category resolution
    # below both possible and necessary: a removal is excused by
    # `declared.removed` and by nothing else.
    arity_changed_registrations = tuple(sorted(
        {_registration_ident(i) for i in op_diff.arity_changed}
        | {_registration_ident(i) for i in predicate_diff.arity_changed}))

    declared_removals = tuple(sorted(set(
        _declared_covering(declared, removed_symbols, which=declared.removed)
        + _declared_covering(declared, arity_changed, which=declared.arity_changed)
        + tuple(name for name in removed_registrations
                if name in declared.removed)
        + tuple(name for name in removed_predicates
                if name in declared.removed)
        # A registration ident is not a mangled name, so there is nothing to
        # demangle and both declaration sets are accepted: a proposal may
        # reasonably spell "GGML_CPU_OP(MUL_MAT, …) changes" under either key.
        + tuple(name for name in arity_changed_registrations
                if name in declared.removed or name in declared.arity_changed))))

    checks: list = []
    notes: list = list(anchor_table.coverage_notes) + list(candidate_table.coverage_notes)

    for label, registration_diff in (("op registration", op_diff),
                                     ("dispatch predicate", predicate_diff)):
        if registration_diff.arity_changed:
            checks.append((f"{label}_arity_unchanged", schemas.Check(
                schemas.FAIL,
                (f"{len(registration_diff.arity_changed)} {label}(s) changed arity: "
                 f"{[list(i) for i in registration_diff.arity_changed]}. This now reaches "
                 "the T0 gate through SymbolTableDiff.arity_changed_op_registrations; it "
                 "is repeated here because `checks` is where the arity PAIR is legible and "
                 "the record carries only the ident.",))))
        else:
            checks.append((f"{label}_arity_unchanged", schemas.Check(schemas.PASS)))
        if registration_diff.arity_not_comparable:
            checks.append((f"{label}_arity_comparable", schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"{len(registration_diff.arity_not_comparable)} {label}(s) carry an arity "
                 "on exactly one side: "
                 f"{[list(i) for i in registration_diff.arity_not_comparable]}. An arity of "
                 "None means the declared pattern did not capture it, NEVER 'unchanged'.",))))
        else:
            checks.append((f"{label}_arity_comparable", schemas.Check(schemas.PASS)))

    if anchor_table.coverage_notes or candidate_table.coverage_notes:
        checks.append(("elf_extraction_complete", schemas.Check(
            schemas.COULD_NOT_CHECK,
            tuple(f"the ELF extractor reported a coverage gap: {note}"
                  for note in (tuple(anchor_table.coverage_notes)
                               + tuple(candidate_table.coverage_notes))))))
    else:
        checks.append(("elf_extraction_complete", schemas.Check(schemas.PASS)))

    notes.append(
        f"anchor {anchor.tool} {anchor_table.file_sha256[:12]} (.{anchor_table.preferred}, "
        f"{diff.anchor_count} exported) vs candidate {candidate_table.file_sha256[:12]} "
        f"(.{candidate_table.preferred}, {diff.candidate_count} exported); the listing is "
        "complete, not capped")
    notes.append(
        f"anchor identity for this diff (the record cannot carry it — SEAM_NOTES): "
        f"{anchor.identity.short()}")
    if diff.unmangled_removed:
        notes.append(
            f"{len(diff.unmangled_removed)} removed symbol(s) did not demangle and were "
            f"compared by exact name only: {list(diff.unmangled_removed)}")

    receipt_ref = "aksym:" + schemas.content_hash({
        "anchor": anchor_table.to_dict(),
        "candidate": candidate_table.to_dict(),
        "symbols": diff.to_dict(),
        "op_registrations": op_diff.to_dict(),
        "dispatch_predicates": predicate_diff.to_dict(),
        "anchor_identity": [anchor.capture.source_commit, anchor.capture.binary_sha256,
                            anchor.capture.linkage_sha256, anchor.tool],
    })[:32]

    record = correctness.SymbolTableDiff(
        removed_symbols=removed_symbols,
        arity_changed_symbols=arity_changed,
        added_symbols=diff.added,
        removed_op_registrations=removed_registrations,
        removed_dispatch_predicates=removed_predicates,
        declared_removals=declared_removals,
        anchor_symbol_count=diff.anchor_count,
        candidate_symbol_count=diff.candidate_count,
        # The tool id names EVERY extractor whose output reached this record, so
        # a reader of the gate's `tool=` note can tell which registries were
        # actually looked at. It rides out on the gate: `check_symbol_and_
        # registration_preservation` puts it in `notes`.
        tool_id=(f"{anchor_table.extractor_id}"
                 f"+{anchor_op_registrations.extractor_id}"
                 f"[{anchor_op_registrations.kind},{anchor_dispatch_predicates.kind}]"),
        receipt_ref=receipt_ref,
        produced_by="evaluator",
        # BOTH registries, because the far side subtracts `declared_removals`
        # from this list too and a dispatch predicate whose arity changed is the
        # same defect as an op registration whose arity changed.
        arity_changed_op_registrations=arity_changed_registrations,
    )
    return SymbolEvidence(
        diff=record, anchor_table=anchor_table, candidate_table=candidate_table,
        symbol_diff=diff, op_registration_diff=op_diff,
        dispatch_predicate_diff=predicate_diff,
        checks=tuple(checks), notes=tuple(notes))


# =============================================================================
# Seam 6 — integrity's parsed diff -> the two T0 diff-policy gates
# =============================================================================

#: `git commit` flags that widen the commit beyond its pathspec, whatever
#: pathspec follows. Two families, and both must be here:
#:
#:   * `-a`/`--all` (and `-a` inside any bundled short-flag cluster) stages every
#:     modified tracked file: `git commit -am msg -- file` commits the lot and
#:     the pathspec only narrows the message of the lie.
#:   * `-i`/`--include` is the one the 2026-08-04 red team found missing, and it
#:     is the exact shape of the hazard this field exists for. `git commit -i --
#:     <paths>` means *"stage these paths IN ADDITION TO whatever is already
#:     staged"* (git-commit(1)); the default for a pathspec commit is `--only`,
#:     which disregards the index. So `-i` is the spelling under which another
#:     session's staged files ride into the artifact WITH a pathspec present, and
#:     it read as `True` — pathspec-limited — until this entry existed.
#:
#: `-p`/`--patch` narrows rather than widens, and is here because what it commits
#: is decided interactively and is therefore not derivable from the argv at all.
_COMMIT_ALL_FLAGS = frozenset({"-a", "--all", "--patch", "-p", "-i", "--include"})

#: Short flags that widen, for the bundled-cluster scan. `-am`, `-ia`, `-im` and
#: `-ai` are all real spellings and none of them is in `_COMMIT_ALL_FLAGS`.
_COMMIT_ALL_SHORT_LETTERS = frozenset({"a", "i", "p"})


def commit_was_pathspec_limited(commit_argv: Sequence[str]) -> tuple:
    """`(bool, reason)` — was this commit limited to an explicit pathspec?

    DERIVED, not declared. `correctness.DiffPolicyEvidence.commit_was_pathspec_limited`
    is the field whose gate reason is *"in a shared clone an unrestricted commit
    sweeps another session's staged files into the artifact"*
    (`feedback_parallel_agent_staged_files_ride_along`), and it is the one field
    of that record whose truth the evaluator has to establish for itself
    (`check_schema_and_diff_policy` now FAILs a `produced_by != "evaluator"`
    record outright, which stops a CANDIDATE asserting it — it does not supply the
    value). A boolean parameter would leave it a belief held by whoever called
    this function. The commit ARGV is a fact the driver has, so this reads it.

    True requires BOTH: a `--` separator with at least one pathspec after it, and
    no flag that widens the commit past that pathspec (`_COMMIT_ALL_FLAGS` — read
    its comment for why `--include` is there). `git add -- <paths>` followed by a
    bare `git commit -m …` is False, and deliberately: the index it commits is
    shared, so what rides along is whatever anyone else staged.

    Only the argv HEAD — everything before `--` — is scanned for flags, because
    that is where git accepts them; a pathspec after the separator that happens
    to be spelled like a flag is a filename.
    """
    argv = [str(token) for token in commit_argv]
    if not argv:
        return False, "no commit argv was recorded, so the commit's scope is unknown"
    head = argv[:argv.index("--")] if "--" in argv else argv
    for token in head:
        if token in _COMMIT_ALL_FLAGS:
            return False, (f"the commit carried {token!r}, which commits more than the "
                           "pathspec that follows it")
        if token.startswith("-") and not token.startswith("--") \
                and (set(token[1:]) & _COMMIT_ALL_SHORT_LETTERS):
            widening = "".join(sorted(set(token[1:]) & _COMMIT_ALL_SHORT_LETTERS))
            return False, (f"the commit carried {token!r}; {widening!r} inside a bundled "
                           "short-flag cluster commits more than the pathspec")
    if "--" not in argv:
        return False, ("the commit carried no '--' pathspec separator, so it committed "
                       "whatever the shared index held (feedback_parallel_agent_staged_"
                       "files_ride_along)")
    tail = argv[argv.index("--") + 1:]
    if not tail:
        return False, "the commit's '--' separator was followed by no pathspec at all"
    return True, f"commit limited to pathspec(s) {tail}"


@dataclass(frozen=True)
class DiffEvidence:
    """The projected `correctness.DiffPolicyEvidence` plus the parse it came from."""

    policy: correctness.DiffPolicyEvidence
    source_diff: integrity.SourceDiff
    checks: tuple
    notes: tuple = ()

    @property
    def worst(self) -> schemas.Check:
        return _worst(check for _name, check in self.checks)


def diff_policy_evidence(*, diff_text: str,
                         worktree_root: Any,
                         declared_surface_files: Sequence[str],
                         envelope: correctness.ChangeClassEnvelope,
                         branch_name: str,
                         commit_argv: Sequence[str],
                         record_schema_violations: Sequence[str],
                         diff_ref: Optional[str] = None) -> DiffEvidence:
    """Seam 6: `integrity.parse_unified_diff` -> `correctness.DiffPolicyEvidence`.

    One record, two gates: `check_semantic_diff_conformance` (§8.5.1 item 3) and
    `check_schema_and_diff_policy` (§8.6 + §10.6). Both read COULD_NOT_CHECK
    today for one reason — nothing ever built the record.

    Every field is derived from the diff text, the worktree, or the commit argv,
    with three exceptions, and each of the three is a required argument with no
    default so that supplying it is a decision rather than an omission:

      * `declared_surface_files` is the PROPOSAL's declared surface. It is the
        thing the derived surface is scored against; it cannot be derived from
        the diff, because the diff IS the other side of that comparison.
      * `envelope` is the change class's size bound, which is campaign policy.
        `change_class` is taken off the envelope rather than accepted separately:
        `DiffPolicyEvidence.__post_init__` refuses a mismatch, and two arguments
        that must agree are one argument.
      * `record_schema_violations` is the result of validating the candidate's
        OWN records against `schemas.py`. `()` here asserts that validation ran
        and was clean; it is not a default, because an empty list from a
        validator nobody invoked reads exactly like a clean one.

    `production_tree_paths` is MEASURED: every path in the diff is resolved
    against `worktree_root` and tested with `t0_provider.under_production_tree`,
    which folds `..` and resolves symlinks. A diff path is repo-relative, so the
    gate's own `_under_production_tree(p)` over the raw path can never fire —
    `ggml/src/ggml.c` is not an absolute path inside a frozen tree, and
    `../../../mnt/raid0/llm/llama.cpp/ggml/src/ggml.c` is.
    """
    if not isinstance(diff_text, str):
        raise TypeError("diff_policy_evidence(diff_text=…) takes the diff TEXT")
    if not isinstance(envelope, correctness.ChangeClassEnvelope):
        raise TypeError(
            "diff_policy_evidence(envelope=…) takes a correctness.ChangeClassEnvelope. "
            "integrity.ChangeClassEnvelope is a DIFFERENT class with the same name and "
            "`DiffPolicyEvidence.__post_init__` refuses it — the seam-1 shape again.")
    _req_str_arg(branch_name, "branch_name")
    root = os.path.realpath(str(worktree_root))

    parsed = integrity.parse_unified_diff(diff_text)
    declared = tuple(dict.fromkeys(str(p) for p in declared_surface_files))
    files_touched = tuple(sorted(parsed.paths()))

    # A pure deletion INSIDE the declared surface is the mutation; one outside it
    # is the "unrelated deletion" §8.5.1 names. Both are reported to the gate,
    # which subtracts the declared surface for the first finding and reads this
    # tuple directly for the second — so filtering here is not a duplicate of
    # what the gate does, it is the definition of the field.
    unrelated_deletions = tuple(sorted(
        {f.path for f in parsed.files
         if (f.is_deleted_file or f.is_pure_deletion) and f.path not in set(declared)}))

    production_paths: list = []
    for path in files_touched:
        resolved = os.path.realpath(os.path.join(root, path))
        if t0_provider.under_production_tree(resolved):
            production_paths.append(f"{path} -> {resolved}")

    limited, limit_reason = commit_was_pathspec_limited(commit_argv)

    checks: list = []
    notes: list = [limit_reason]
    checks.append(("commit_pathspec_limited", schemas.Check(schemas.PASS) if limited
                   else schemas.Check(schemas.FAIL, (limit_reason,))))
    binary_files = tuple(sorted(f.path for f in parsed.files if f.is_binary))
    if binary_files:
        checks.append(("diff_is_textual", schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"{len(binary_files)} file(s) in the diff are binary and contribute no changed "
             f"line count: {list(binary_files)}. Their size is not bounded by the change-"
             "class envelope, because the envelope counts lines.",))))
    else:
        checks.append(("diff_is_textual", schemas.Check(schemas.PASS)))
    renames = tuple(sorted(f"{f.old_path} -> {f.path}" for f in parsed.files if f.is_rename))
    if renames:
        notes.append(f"rename(s) in the diff: {list(renames)}")

    ref = diff_ref or ("akdiff:" + schemas.content_hash({"diff": diff_text})[:32])
    policy = correctness.DiffPolicyEvidence(
        files_touched=files_touched,
        declared_surface_files=declared,
        unrelated_deletions=unrelated_deletions,
        changed_lines=parsed.total_changed,
        change_class=envelope.change_class,
        envelope=envelope,
        branch_name=str(branch_name),
        commit_was_pathspec_limited=limited,
        production_tree_paths=tuple(production_paths),
        record_schema_violations=tuple(str(v) for v in record_schema_violations),
        diff_ref=ref,
        # Derived here by parsing the diff text, not copied from a declaration.
        produced_by="evaluator",
    )
    return DiffEvidence(policy=policy, source_diff=parsed, checks=tuple(checks),
                        notes=tuple(notes))


def _req_str_arg(value: Any, label: str) -> str:
    """`schemas.require.str`, raising this seam's own error.

    The `error=` kwarg exists for exactly this: the PREDICATE is shared, the
    module's vocabulary is not. A caller of this seam catches `ChainSeamError`
    and must keep catching it.
    """
    return schemas.require.str(value, label, error=ChainSeamError)


# =============================================================================
# Seam 7 — the anchor's OWN build log -> the anchor toolchain the static gate reads
# =============================================================================

@dataclass(frozen=True)
class AnchorToolchain:
    """What `capture_anchor` needs to make `static_and_compile_checks` a real gate.

    `t0_provider.collect_static_analysis` returns `None` — and the gate reads
    COULD_NOT_CHECK — unless the anchor capture carries `compiler_id` AND
    `compiler_version`. `capture_anchor` has taken all three of these as
    parameters since it was written; nothing measured them, so nothing passed
    them.
    """

    compiler_id: str
    compiler_version: str
    warning_count: int
    #: The log these three were measured from. It is not decoration: it is the
    #: only thing that distinguishes an anchor toolchain from the candidate's
    #: own, and `anchor_toolchain_from_build_log` refuses to build one of these
    #: from the candidate's build log because of it. Before the 2026-08-04 red
    #: team this field had NO reader anywhere in the package.
    log_ref: str

    def as_capture_kwargs(self) -> dict:
        """Splat straight into `t0_provider.capture_anchor(**…)`."""
        return {"compiler_id": self.compiler_id,
                "compiler_version": self.compiler_version,
                "warning_count": self.warning_count}


def anchor_toolchain_from_build_log(log_text: str, *, log_ref: str,
                                    candidate_build: Any) -> AnchorToolchain:
    """Seam 7: MEASURE the anchor's toolchain out of the anchor's own build log.

    Three fields, three consumers in `check_static_and_compile`:

      * `compiler_id`/`compiler_version` are compared against the candidate's. A
        mismatch is *"a toolchain comparison wearing a kernel comparison's
        clothes"* — the confound the gate exists to catch. Typing them by hand is
        how the confound arrives THROUGH the gate.
      * `warning_count` is the baseline for the new-warning delta. Without it the
        gate reads COULD_NOT_CHECK with the reason *"no anchor warning count was
        recorded, so a new warning cannot be detected"* — so passing the first
        two and not the third buys a COULD_NOT_CHECK with a different sentence.

    The compiler identity comes from `worktree.parse_build_log`, which reads
    cmake's `-- The CXX compiler identification is GNU 15.2.0`, and is split by
    `split_compiler_identity` — which RAISES rather than inventing a version.
    cmake prints that line at CONFIGURE time only, so a log from a re-used cmake
    cache does not carry it and this raises; that is the correct outcome, and the
    caller then either configures fresh or has no static-analysis evidence.

    The warning count comes from `t0_provider.parse_compiler_diagnostics`, which
    counts DISTINCT diagnostic lines, and NOT from `BuildLogFacts.warning_count`:
    the two count different things (the latter counts occurrences), and the
    candidate side of this comparison is produced by `parse_compiler_diagnostics`
    in `collect_static_analysis`. Two counters on the two sides of a `>` make the
    delta a function of build parallelism rather than of the kernel.

    **`candidate_build` is required, and it is required because of what this
    function CANNOT see.** It takes the candidate's own
    `correctness.BuildProvenance` — `chain.build_evidence(...).provenance`, the
    record `evaluate_t0` reads — and REFUSES when `log_ref` resolves to the same
    file as `candidate_build.build_log_ref`. Nothing in a build log says whose
    build it was, so a caller with only one log in hand passes that one, and the
    2026-08-04 red team found exactly that in the reference composition itself:
    `ChainLeg.bind_anchor` measured the "anchor" toolchain off the CANDIDATE's
    build log, both sides of `check_static_and_compile`'s two comparisons were
    then the same bytes, and `static_and_compile_checks` reported PASS for a
    self-comparison. Both branches it exists for — the toolchain mismatch and the
    new-warning delta — are unreachable in that wiring, and the failure is
    invisible because a self-comparison always agrees.

    A `BuildProvenance` and not a string: a string can be typed to whatever
    dodges the comparison, whereas this object has to have been produced by an
    actual candidate build before there is anything to pass.
    """
    if not isinstance(log_text, str) or not log_text.strip():
        raise BuildProvenanceUnprojectable(
            "the anchor build log is empty. `anchor_warning_count` taken off an empty log "
            "is 0, which is the strongest possible baseline and would make every new "
            "candidate warning a FAIL.")
    _req_str_arg(log_ref, "log_ref")
    if not isinstance(candidate_build, correctness.BuildProvenance):
        raise TypeError(
            "anchor_toolchain_from_build_log(candidate_build=…) takes the CANDIDATE's "
            "correctness.BuildProvenance — `chain.build_evidence(identity).provenance`. It "
            "is how this function can tell the anchor's build log from the candidate's, "
            "which no build log says about itself; without it the anchor toolchain can be "
            "measured off the candidate's own log and every comparison in "
            "`check_static_and_compile` becomes a self-comparison that always agrees.")
    anchor_log = t0_provider.resolve_build_log_ref(log_ref)
    candidate_log = t0_provider.resolve_build_log_ref(candidate_build.build_log_ref)
    if (anchor_log is not None and candidate_log is not None
            and os.path.realpath(anchor_log) == os.path.realpath(candidate_log)):
        raise BuildProvenanceUnprojectable(
            f"the 'anchor' build log {log_ref!r} resolves to {os.path.realpath(anchor_log)!r}, "
            f"which is the CANDIDATE's own build log ({candidate_build.build_log_ref!r}). "
            "`check_static_and_compile` compares compiler_id/compiler_version and "
            "warning_count across the two arms; taken off one log both comparisons are "
            "identities, the toolchain-confound branch and the new-warning branch can never "
            "fire, and the gate PASSes on a self-comparison. Capture the ANCHOR's own "
            "configure+build output, or pass no anchor toolchain at all and let the gate read "
            "COULD_NOT_CHECK.")
    facts = worktree.parse_build_log(log_text)
    compiler = worktree._compiler_identity(facts)
    if not compiler:
        raise BuildProvenanceUnprojectable(
            f"the anchor build log {log_ref!r} carries no '-- The <LANG> compiler "
            "identification is …' line, so the anchor toolchain cannot be measured from it. "
            "cmake prints it at CONFIGURE time only; a build log from a re-used cmake cache "
            "does not repeat it. Capture the anchor's configure output, or pass no anchor "
            "toolchain at all — the static gate then reads COULD_NOT_CHECK, which is true.")
    compiler_id, compiler_version = split_compiler_identity(compiler)
    _errors, warnings, _findings = t0_provider.parse_compiler_diagnostics(log_text)
    return AnchorToolchain(compiler_id=compiler_id, compiler_version=compiler_version,
                           warning_count=warnings, log_ref=log_ref)


# =============================================================================
# Seam 8 — surface.py's derivation -> the ChangeSurface four gates read
# =============================================================================

#: The lexical evidence that a change TOUCHES a behavioural surface, by surface.
#:
#: Read the polarity before reading the tokens. `classify_behavioural_surface`
#: returns `True` on a match and `None` on no match. It NEVER returns `False`,
#: and that is the whole design:
#:
#:   * a `True` widens the required evidence (ASAN/UBSAN become mandatory, the
#:     state/rollback surface becomes relevant) and can only make T0 stricter;
#:   * a `None` is COULD_NOT_CHECK, which is what the surface already reads
#:     today, so a token this table is missing costs a gate that was already
#:     unevaluated;
#:   * a `False` would license `check_asan`'s *"ASAN/UBSAN is not mandatory for
#:     this change: the mechanical derivation finds it touches neither memory nor
#:     threading"* — a PASS on a fact this table is not strong enough to
#:     establish. Proving that no reachable path allocates or spawns needs a
#:     whole-program analysis, not a token list, and nothing in this package has
#:     one.
#:
#: So the quality of this table decides how OFTEN a real gate is produced, never
#: whether a produced gate is honest. Adding a token can only turn a
#: COULD_NOT_CHECK into a FAIL-or-real-PASS; it can never turn a FAIL into a PASS.
BEHAVIOURAL_TOKENS = {
    "memory": (
        r"\b(?:m|c|re)alloc\b", r"\bfree\b", r"\balloca\b", r"\bposix_memalign\b",
        r"\baligned_alloc\b", r"\b_mm_malloc\b", r"\b_mm_free\b",
        r"\bmem(?:cpy|move|set|cmp)\b", r"\bnew\b", r"\bdelete\b",
        r"\bggml_(?:new_buffer|backend_buffer|aligned_malloc|free)\w*",
        r"\bwork_size\b", r"\bwsize\b", r"\bwdata\b",
        r"\breinterpret_cast\b", r"\bstatic_cast\s*<\s*\w+\s*\*",
    ),
    "threading": (
        r"\b(?:std::)?thread\b", r"\bpthread_\w+", r"#\s*pragma\s+omp", r"\bomp_\w+",
        r"\batomic\w*", r"\bmutex\b", r"\bmemory_order\w*",
        r"\bn_threads\b", r"\bnth\b", r"\bith\b", r"\bbarrier\b",
        r"\b__sync_\w+", r"\b__atomic_\w+",
    ),
    "persistent_state": (
        r"\bstatic\s+(?!inline\b)(?!constexpr\b)\w+", r"\bthread_local\b",
        r"\bfopen\b", r"\bfwrite\b", r"\bmmap\b", r"\bmunmap\b",
        r"\bcache\w*", r"\bpersist\w*", r"\bggml_backend_sched\w*",
        r"\bsetenv\b", r"\bgetenv\b",
    ),
}

_DIFF_BODY_RE = re.compile(r"^[+-](?![+-])")


def _changed_lines(diff_text: str) -> tuple:
    """The added and removed BODY lines of a unified diff, without the file headers.

    `+++ b/x` and `--- a/x` start with the marker and are not content; including
    them would let a path token — `ggml/src/ggml-cpu/cache.c` — score as evidence
    that the change touches persistent state.
    """
    return tuple(line for line in diff_text.splitlines() if _DIFF_BODY_RE.match(line))


def classify_behavioural_surface(diff_text: str) -> dict:
    """`{surface: (True|None, (matched tokens, …))}` over a unified diff.

    Three-valued by construction and two-valued in practice: `True` or `None`.
    See `BEHAVIOURAL_TOKENS` for why there is no `False` and why that is not a
    weakness of this function but the reason it is allowed to exist.
    """
    body = _changed_lines(diff_text)
    out: dict = {}
    for name, patterns in BEHAVIOURAL_TOKENS.items():
        matched: list = []
        for pattern in patterns:
            compiled = re.compile(pattern)
            for line in body:
                if compiled.search(line):
                    matched.append(f"{pattern} @ {line.strip()[:80]}")
                    break
        out[name] = (True if matched else None, tuple(matched))
    return out


@dataclass(frozen=True)
class ChangeSurfaceEvidence:
    """The projected `correctness.ChangeSurface` plus the derivation it came from."""

    surface: correctness.ChangeSurface
    affected: Any
    checks: tuple
    notes: tuple = ()

    @property
    def worst(self) -> schemas.Check:
        return _worst(check for _name, check in self.checks)


def change_surface_from(affected: Any, *, diff_text: str,
                        declared_touches_memory: Optional[bool] = None,
                        declared_touches_threading: Optional[bool] = None,
                        declared_ops: Sequence[str] = ()) -> ChangeSurfaceEvidence:
    """Seam 8: `surface.derive_affected_surface` -> `correctness.ChangeSurface`.

    Four T0 surfaces — `sanitizer.asan`, `sanitizer.ubsan`,
    `unseen_boundary_shapes` and `state_rollback_teardown_race` — decide their
    own applicability from `ChangeSurface.derived_touches_*`, and
    `t0_provider._change_surface` emits a surface with all four `None` because no
    derivation was ever supplied to the plan. This is the supply.

    WHAT COMES STRAIGHT OFF THE DERIVATION:

      * `derived_ops` = `AffectedSurface.op_names` — the op registrations in the
        change's build-system closure. It is the set `check_backend_op_units`
        UNIONS with the policy's mandatory ops, so wiring it makes an op the
        change actually touches a REQUIRED op.
      * `derived_files` = `AffectedSurface.touched_files`.
      * `derived_touches_dispatch` = `True` when the closure contains a dispatch
        predicate, `None` otherwise. Never `False`: an empty dispatch-predicate
        tuple is also what `derive_affected_surface` produces when no
        `SymbolRegistrationIndex` was supplied, and it records that as an
        `OA_NO_SYMBOL_INDEX` widening rather than as an absence.
      * `touches_shared_core_header` = a shared-header or core-header
        over-approximation is present. This one IS two-valued because
        `ChangeSurface` gives it no third state; it only ADDS a human-review
        marker, and `change_class == "core_header"` forces that marker
        independently, so the weaker reading cannot license a PASS on its own.

    WHAT DOES NOT: `AffectedSurface` carries no memory / threading /
    persistent-state axis — `SURFACE_AXES` is backends, link targets, op names,
    kernel symbols and dispatch predicates. Those three flags are classified from
    the diff body by `classify_behavioural_surface`, which yields `True` or
    `None` and never `False`. The consequence is stated plainly because a reader
    will otherwise expect more: wiring this turns the sanitizer and state gates
    into REAL gates for a candidate that visibly touches memory, threading or
    persistent state, and leaves them COULD_NOT_CHECK for one that does not
    visibly touch them. It does not manufacture the PASS branch, and the PASS
    branch is the one that would be a lie.

    `declared_*` are the ACTOR's prediction. They drive nothing (invariant 18);
    `ChangeSurface.prediction_score()` scores them and no gate reads them.
    """
    if not isinstance(affected, surface_module.AffectedSurface):
        raise TypeError(
            "change_surface_from takes a surface.AffectedSurface — the output of "
            "`derive_affected_surface`, which takes its closure from the BUILD SYSTEM's own "
            "dependency information and refuses an actor-declared one. Anything else here "
            "would make the derived surface a declaration wearing a derivation's name.")
    if not isinstance(diff_text, str):
        raise TypeError("change_surface_from(diff_text=…) takes the diff TEXT")

    classified = classify_behavioural_surface(diff_text)
    checks: list = []
    notes: list = []
    for name, (flag, matched) in sorted(classified.items()):
        if flag is True:
            checks.append((f"derived_touches_{name}", schemas.Check(
                schemas.PASS,
                (f"the change body matches {len(matched)} {name} token(s), so the surface "
                 f"is DETERMINED to be touched: {list(matched)}",))))
        else:
            checks.append((f"derived_touches_{name}", schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"no {name} token matched the change body. This is recorded as UNDETERMINED "
                 "and not as 'does not touch it': proving a negative here needs a "
                 "whole-program analysis and BEHAVIOURAL_TOKENS is a token list.",))))

    core_header = tuple(
        widening for widening in affected.over_approximations
        if widening.reason in (surface_module.OA_SHARED_HEADER_FANOUT,
                               surface_module.OA_CORE_HEADER_CHANGE_CLASS))
    if affected.coverage.outcome != schemas.PASS:
        checks.append(("closure_coverage", schemas.Check(
            affected.coverage.outcome,
            ("the affected-surface derivation's own coverage check is "
             f"{affected.coverage.outcome}: ",) + tuple(affected.coverage.reasons))))
    else:
        checks.append(("closure_coverage", schemas.Check(schemas.PASS)))
    if affected.fail_closed_widenings:
        notes.append(
            f"{len(affected.fail_closed_widenings)} fail-closed widening(s): "
            f"{[w.reason for w in affected.fail_closed_widenings]}; the manifest is a "
            "superset reached by widening, not by derivation")
    if affected.full_tree:
        notes.append("the derivation widened to the FULL TREE; every derived_ops entry below "
                     "is the whole known surface, not the change's own")
    if not affected.dispatch_predicates:
        notes.append(
            "the closure contains no dispatch predicate, so derived_touches_dispatch is "
            "UNDETERMINED rather than False — an empty tuple is also what the derivation "
            "produces with no SymbolRegistrationIndex supplied")

    # The ref names BOTH halves and the CONTENT of each, because a reader who
    # sees `derived_touches_memory=True` has to be able to tell which token table
    # said so — the table is editable and the flag is not self-explaining.
    token_digest = schemas.content_hash(
        {"tokens": {name: list(patterns)
                    for name, patterns in sorted(BEHAVIOURAL_TOKENS.items())}})
    derivation_ref = (f"autokernel.evaluator.surface.derive_affected_surface"
                      f"@{affected.sha256()[:16]}"
                      f"+autokernel.execution.chain.classify_behavioural_surface/v1"
                      f"@{token_digest[:16]}")

    record = correctness.ChangeSurface(
        derived_touches_memory=classified["memory"][0],
        derived_touches_threading=classified["threading"][0],
        derived_touches_dispatch=True if affected.dispatch_predicates else None,
        derived_touches_persistent_state=classified["persistent_state"][0],
        derived_ops=tuple(affected.op_names),
        derived_files=tuple(affected.touched_files),
        declared_touches_memory=declared_touches_memory,
        declared_touches_threading=declared_touches_threading,
        declared_ops=tuple(declared_ops),
        touches_shared_core_header=bool(core_header),
        derivation_ref=derivation_ref,
    )
    return ChangeSurfaceEvidence(surface=record, affected=affected, checks=tuple(checks),
                                 notes=tuple(notes))
