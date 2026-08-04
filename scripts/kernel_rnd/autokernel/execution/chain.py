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
the same `source_commit` and the same `linkage_sha256`. The api.py change that
would make the digest tool-aware is in `SEAM_NOTES`.

**Seam 4 — one claim, two Protocols.** `t0_provider.HeldClaim` wants
`claim_id` + one of `verify_held`/`is_held`/`held`. `microbench.HeldClaim` wants
`claim_id` + `attest()`, and `CpuRegionClaim` does not have `attest()` — it needs
`microbench.CpuRegionClaimAdapter`. A driver that passes the raw claim to the
microbench runner gets a `TypeError` an hour into a claim window.
`bind_claim()` returns both bindings from one acquisition and
`check_claim_satisfies_both_seams()` proves the real class satisfies both.

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
from ..evaluator import api, correctness, integrity
from . import t0_provider, worktree

__all__ = [
    "ChainSeamError",
    "BuildProvenanceUnprojectable",
    "AnchorNotOneAnchor",
    "ClaimSeamUnsatisfied",
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
    ("correctness.BuildProvenance has no `produced_by`, so a projection made here by the "
     "evaluator's own collection half is indistinguishable from one an actor supplied. "
     "t0_provider.SCHEMA_FOLLOWUPS[0] records the same gap and the same remedy; this is a "
     "second consumer of it, not a second finding."),
    ("api.AnchorIdentity.binary_sha256 is SINGLE-VALUED, and one anchor build ships several "
     "binaries. T0 hashes the anchor `llama-cli`; `microbench` compares the plan's anchor "
     "digest against the anchor `llama-bench` it is about to spawn. One triple cannot name "
     "both truthfully, so a campaign leg that runs T0 and T1 needs TWO AnchorBindings and "
     "`check_anchor_build_is_one_build` to tie them together. REQUIRED FOLLOW-UP (api.py, "
     "forbidden this hour): either a per-tool digest table on AnchorIdentity, or a documented "
     "rule that binary_sha256 names the tool the record's `metric` is measured with."),
    ("execution/cpu_region_claim.py mints claim ids with the prefix 'akc-', which is the "
     "prefix `api.EvaluationRequest.__post_init__` requires of a CANDIDATE id. A claim id "
     "passed where a candidate id belongs therefore satisfies the one validator written to "
     "catch that class of mistake. Not fixed here: the prefix is asserted by that module's "
     "own tests and changing it is its owner's edit. REQUIRED FOLLOW-UP: change "
     "cpu_region_claim._CLAIM_ID_PREFIX to 'akclaim-' (the spelling t0_provider's own test "
     "fixtures already use) and update its assertions."),
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
    reasons: list = []
    outcome = schemas.PASS
    for check in checks:
        reasons.extend(check.reasons)
        if check.outcome == schemas.FAIL:
            outcome = schemas.FAIL
        elif check.outcome == schemas.COULD_NOT_CHECK and outcome != schemas.FAIL:
            outcome = schemas.COULD_NOT_CHECK
    return schemas.Check(outcome, tuple(reasons))


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
        """The `api.AnchorIdentity` for THIS tool. One expression, used by every consumer."""
        return self.capture.identity()

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
