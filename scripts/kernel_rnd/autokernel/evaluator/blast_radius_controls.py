#!/usr/bin/env python3
"""AKX-P1d control fixtures for the T0/T1 cross-workload keep screen.

Zero compute.  The controls run on *recorded* data only: synthetic unified diffs fed
through the real ``footprint`` producer, a census-shaped record per workload, and
recorded (synthetic, clearly labelled) T0 DSO digests / T1 profile hashes.  The
reference classifier below implements the AKX program section 7.3 rules over those
inputs so the controls pin the rules; rows it emits are validated by the production
``loop.keep_screen`` row contract.  Live parts (a real build proving the T0 digest
equality, a real T1 coverage forward) are follow-ups and are marked ``LIVE_FOLLOWUP``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from . import footprint as FP

ROW_SCHEMA = "epyc.autokernel.blast_radius_row.v1"

# AKX program section 6.1 fleet snapshot: id -> (backend, dominant weight quant set).
# W5 (UD-IQ4_XS, dynamic) is deliberately parameterised by its census tensor-type set.
W5_WITH_Q8 = frozenset({"Q4_K", "Q6_K", "IQ4_XS", "Q8_0", "F32"})
W5_WITHOUT_Q8 = frozenset({"Q4_K", "Q6_K", "IQ4_XS", "F32"})
FLEET = {
    "W1": {"backend": "rocm", "types": frozenset({"Q8_0", "F32"})},
    "W2": {"backend": "rocm", "types": frozenset({"Q4_K", "Q6_K", "F32", "F16"})},
    "W3": {"backend": "cpu", "types": frozenset({"Q8_0", "F32"})},
    "W4": {"backend": "cpu", "types": frozenset({"Q4_K", "Q6_K", "F32"})},
    "W6": {"backend": "cpu", "types": frozenset({"Q4_K", "Q6_K", "F32"})},
}


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def diff(path: str, body: str, context: str = "") -> str:
    lines = body.splitlines()
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -10,3 +10,{len(lines)} @@ {context}\n" + "\n".join(lines) + "\n")


def produce(patch: str, child: str = "candidate") -> FP.PatchFootprint:
    path = patch.split("+++ b/", 1)[1].splitlines()[0]
    obj = "build/" + path.rsplit("/", 1)[-1] + ".o"
    cc = [{"directory": "/src", "file": "/src/" + path, "output": obj,
           "arguments": ["cc", "-c", "/src/" + path, "-o", obj]}]
    deps = {"fixture.d": f"{obj}: /src/{path} /src/ggml/include/ggml.h\n"}
    return FP.produce_patch_footprint(source_tree="llama.cpp", parent="parent",
                                      child=child, diff_text=patch,
                                      compile_commands=cc, depfiles=deps)


# --- synthetic patches -------------------------------------------------------------
COMMENT_ONLY = diff("ggml/src/ggml-cuda/mmvq.cu", """+// NOTE: Q4_K path; revisit if (ne11 <= 8) crossover moves.
+// getenv("GGML_FAKE_KNOB") is documented here, not read.
+/* case GGML_TYPE_Q4_K: return true; */""", "bool ggml_cuda_should_use_mmvq(void) {")

Q8_0_DISPATCH = diff("ggml/src/ggml.c", """+case GGML_TYPE_Q8_0:
+    return ne11 <= 16;""", "bool ggml_type_dispatch_fixture(enum ggml_type type, int64_t ne11) {")

Q4_K_DISPATCH = diff("ggml/src/ggml.c", """+case GGML_TYPE_Q4_K:
+    return ne11 <= 16;""", "bool ggml_type_dispatch_fixture(enum ggml_type type, int64_t ne11) {")

KNOB_READ_NO_DEFAULT_CHANGE = diff("ggml/src/ggml-cpu/ggml-cpu.c", """ {
     const char * e = getenv("GGML_FIXTURE_KNOB");
     ggml_cpu_fixture_knob = (e == NULL || *e == '\\0') ? false : (atoi(e) != 0);
+    ggml_cpu_fixture_knob_seen = e != NULL;
 }""", "void ggml_cpu_init(void) {")

KNOB_DEFAULT_FLIP_ON = diff("ggml/src/ggml-cpu/ggml-cpu.c", """-    ggml_cpu_fixture_knob = (e == NULL || *e == '\\0') ? false : (atoi(e) != 0);
+    ggml_cpu_fixture_knob = (e == NULL || *e == '\\0') ? true : (atoi(e) != 0);""",
                            "void ggml_cpu_init(void) { const char * e = getenv(\"GGML_FIXTURE_KNOB\");")


# --- reference classifier (AKX program section 7.3) --------------------------------
@dataclass(frozen=True)
class WorkloadRecord:
    workload_id: str
    backend: str
    tensor_types: frozenset
    census_observed: bool = True
    census_stale: bool = False
    effective_env: Mapping[str, str] = field(default_factory=dict)
    is_author_target: bool = False
    # Recorded witnesses; None means "no such evidence recorded".
    t0_dso_anchor: Optional[Mapping[str, str]] = None
    t0_dso_candidate: Optional[Mapping[str, str]] = None
    t1_profiles: Optional[tuple] = None  # (anchor_sha, candidate_sha, envelope_sha)


def workload(wid: str, *, types=None, **kw: Any) -> WorkloadRecord:
    spec = FLEET.get(wid, {"backend": "cpu", "types": frozenset()})
    return WorkloadRecord(wid, spec["backend"],
                          frozenset(types if types is not None else spec["types"]), **kw)


def is_no_op(fp: FP.PatchFootprint) -> bool:
    """True iff every hunk is comment-only, so no compiled behaviour can differ."""
    return bool(fp.hunks) and all(h.kind == "comment" for h in fp.hunks)


def _live_hunks(fp: FP.PatchFootprint):
    return [h for h in fp.hunks if h.kind != "comment"]


def _static_types(fp: FP.PatchFootprint) -> frozenset:
    out: set = set()
    for site in fp.dispatch_sites:
        out.update(t.removeprefix("GGML_TYPE_") for t in site.discriminators["types"])
    return frozenset(out)


def exercised_knob_flips(fp: FP.PatchFootprint, wl: WorkloadRecord) -> list:
    """Knobs whose effective default changes for ``wl``.

    A knob that is read but whose default is unchanged is not an exercised change; a
    knob the workload sets explicitly pins its value, so a default flip cannot reach it.
    """
    out = []
    for k in fp.env_knobs:
        if k.default_before == k.default_after:
            continue
        if k.name in wl.effective_env:
            continue
        out.append(k.name)
    return out


def classify(fp: FP.PatchFootprint, wl: WorkloadRecord) -> dict:
    """Return a ``blast_radius_row`` shaped record (fixture subset of fields)."""
    row: dict[str, Any] = {"schema": ROW_SCHEMA, "keep_id": "keep-fixture",
                           "parent": fp.production_base_commit,
                           "child": fp.candidate_source_commit,
                           "workload_id": wl.workload_id,
                           "census_freshness_key": sha("census-" + wl.workload_id),
                           "footprint_sha256": fp.sha256(), "touched_detail": {},
                           "tool_versions": {}, "created_at": "2026-11-07T00:00:00Z",
                           "witnesses": None, "tier": None, "unknown_reason": None}

    def done(cls: str, **kw: Any) -> dict:
        row.update(classification=cls,
                   decision_class="INERT" if cls == "INERT" else "TOUCHED", **kw)
        return row

    if wl.is_author_target:
        return done("TOUCHED", touched_detail={"reason": "author_target"})
    if not fp.hunks:
        raise FP.FootprintError("empty footprint refuses the screen")
    if wl.census_stale or not wl.census_observed:
        return done("UNKNOWN", unknown_reason="census_stale_or_unobserved")
    if wl.t0_dso_anchor and wl.t0_dso_anchor == wl.t0_dso_candidate:
        return done("INERT", tier="T0", witnesses={
            "dso_digests_anchor": dict(wl.t0_dso_anchor),
            "dso_digests_candidate": dict(wl.t0_dso_candidate)})
    if fp.change_class == "build":
        return done("TOUCHED", touched_detail={"reason": "build_variant"})
    flips = exercised_knob_flips(fp, wl)
    if flips:
        return done("TOUCHED", touched_detail={"knob_reads": flips})
    if fp.opaque_reasons:
        return done("UNKNOWN", unknown_reason=";".join(fp.opaque_reasons))
    if is_no_op(fp):
        # No compiled change, but section 7.3 only grants INERT with a witness.
        return done("UNKNOWN", unknown_reason="no_op_without_t0_witness")
    if fp.env_knobs:
        # Read site without a default change.  Unset knob: not an exercised change, so
        # INERT only through a T1 witness (clause iii); a set knob reaches the new code.
        if any(k.name in wl.effective_env for k in fp.env_knobs):
            return done("TOUCHED", touched_detail={
                "knob_reads": [k.name for k in fp.env_knobs]})
        if wl.t1_profiles:
            a, c, env = wl.t1_profiles
            return done("INERT", tier="T1", witnesses={
                "coverage_profile_sha256": [a, c], "shape_envelope_hash": env})
        return done("UNKNOWN", unknown_reason="knob_unset_without_t1_witness")
    live = _live_hunks(fp)
    types = _static_types(fp)
    type_gated = bool(types) and all(h.kind in ("predicate", "table") for h in live) \

    if type_gated and not (types & wl.tensor_types):
        if wl.t1_profiles:
            a, c, env = wl.t1_profiles
            return done("INERT", tier="T1", witnesses={
                "coverage_profile_sha256": [a, c], "shape_envelope_hash": env})
        return done("UNKNOWN", unknown_reason="type_disjoint_without_t1_witness")
    return done("TOUCHED", touched_detail={"executed_hunks": [h.hunk_id for h in live]})
