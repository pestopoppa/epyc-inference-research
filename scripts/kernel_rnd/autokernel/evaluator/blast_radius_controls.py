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
from typing import Any

from . import footprint as FP

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


# --- fixture adapter over the production classifier (AKX-P1e) ------------------------
from ..loop.blast_radius import (WorkloadRecord, classify as _classify,  # noqa: E402
                                 _static_types, exercised_knob_flips, is_no_op)


def workload(wid: str, *, types=None, **kw: Any) -> WorkloadRecord:
    spec = FLEET.get(wid, {"backend": "cpu", "types": frozenset()})
    return WorkloadRecord(wid, spec["backend"],
                          frozenset(types if types is not None else spec["types"]), **kw)


def classify(fp: FP.PatchFootprint, wl: WorkloadRecord) -> dict:
    """The production classifier under the fixtures' labelled identities."""
    return _classify(fp, wl, keep_id="keep-fixture",
                     census_freshness_key=sha("census-" + wl.workload_id),
                     created_at="2026-11-07T00:00:00Z")
