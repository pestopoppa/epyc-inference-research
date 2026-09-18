#!/usr/bin/env python3
"""Zero-compute AKX-P1a fixtures, including the required historical keeps."""
import unittest

from . import footprint as FP
from . import surface as SU


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


class HistoricalFootprintFixtures(unittest.TestCase):
    def test_7d2ea88b_q4k_mmvq_crossover(self):
        patch = diff("ggml/src/ggml-cuda/mmvq.cu", """ case GGML_TYPE_Q3_K:
-case GGML_TYPE_Q4_K:
 case GGML_TYPE_Q5_K:
+case GGML_TYPE_Q4_K:
+    return log_decision(ne11 <= MMVQ_MAX_BATCH_SIZE);""",
                     "bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {")
        fp = produce(patch, "7d2ea88b")
        sites = fp.to_dict()["dispatch_sites"]
        self.assertTrue(any("GGML_TYPE_Q4_K" in s["discriminators"]["types"] for s in sites))
        self.assertTrue(any(s["discriminators"]["shape_terms"] for s in sites))
        self.assertIn("ggml/src/ggml-cuda/mmvq.cu:ggml_cuda_should_use_mmvq", fp.actual_symbols)

    def test_732389d6_q4k_weight_block_hoist(self):
        patch = diff("ggml/src/ggml-cuda/mmvq.cu", """+if constexpr (type == GGML_TYPE_Q4_K) {
+    const int weight_block = kbx / QK_K;
+    load_weights(weight_block);
+}""", "template <ggml_type type> __global__ void mul_mat_vec_q(void) {")
        fp = produce(patch, "732389d6")
        self.assertEqual(fp.change_class, "dispatch_predicate")
        self.assertTrue(any("GGML_TYPE_Q4_K" in s.discriminators["types"]
                            for s in fp.dispatch_sites))

    def test_db18f393_arch_and_shape_discriminators(self):
        patch = diff("ggml/src/ggml-cuda/fattn-wmma-f16.cu", """+#if defined(GGML_USE_HIP)
+if (ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_CDNA2 && Q->ne[0] == 128 && Q->ne[1] <= 8) {
+    ggml_cuda_flash_attn_ext_wmma_f16_case<128, 16, half, 8>(ctx, dst);
+    return;
+}
+#endif""", "void ggml_cuda_flash_attn_ext_wmma_f16(void) {")
        fp = produce(patch, "db18f393")
        merged = [s.discriminators for s in fp.dispatch_sites]
        self.assertTrue(any("GGML_USE_HIP" in d["backend"] for d in merged))
        self.assertTrue(any("GGML_CUDA_CC_CDNA2" in d["arch"] for d in merged))
        self.assertTrue(any(d["shape_terms"] for d in merged))

    def test_2516c9807_default_on_flips(self):
        # Historical sync17-fix2 initialization and its auditable marker.  The old
        # side has no knob/read, so UNKNOWN -> ON is the conservative default flip.
        patch = diff("ggml/src/ggml-cpu/ggml-cpu.c", """+__attribute__((used)) static const char marker[] =
+    \"DEFAULT_ON=GGML_SCALE_SPLIT,GGML_SOLO_YIELD_ROWCOL\";
+{
+    const char * e = getenv(\"GGML_SCALE_SPLIT\");
+    ggml_cpu_scale_split = (e == NULL || *e == '\\0') ? true : (atoi(e) != 0);
+}
+{
+    const char * e = getenv(\"GGML_SOLO_YIELD_ROWCOL\");
+    ggml_cpu_solo_yield_rowcol = (e == NULL || *e == '\\0') ? true : (atoi(e) != 0);
+}""", "void ggml_cpu_init(void) {")
        fp = produce(patch, "2516c9807")
        knobs = {k.name: (k.default_before, k.default_after) for k in fp.env_knobs}
        self.assertEqual(knobs["GGML_SCALE_SPLIT"], ("UNKNOWN", "ON"))
        self.assertEqual(knobs["GGML_SOLO_YIELD_ROWCOL"], ("UNKNOWN", "ON"))
        self.assertEqual(dict(fp.feature_flag_assignments)["GGML_SCALE_SPLIT"], "ON")


class ProducerContractTests(unittest.TestCase):
    def test_composition_evidence_field_names_and_build_closure(self):
        fp = produce(diff("src/a.c", "+return x + 1;", "int changed(int x) {"))
        record = fp.to_dict()
        for field in ("source_tree", "production_base_commit", "candidate_source_commit",
                      "patch_bundle_sha256", "actual_files", "actual_hunk_ids",
                      "actual_symbols", "feature_flag_assignments", "dispatch_predicates",
                      "mechanism_id", "change_class"):
            self.assertIn(field, record)
        self.assertEqual(record["schema"], FP.SCHEMA)
        self.assertEqual(record["objects_expected_changed"]["default"], ["build/a.c.o"])
        self.assertEqual(record["opaque_reasons"], [])

    def test_nonempty_unparseable_diff_refuses(self):
        with self.assertRaises(FP.FootprintError):
            FP.parse_unified_diff("not a unified diff")

    def test_unmapped_file_is_opaque(self):
        patch = diff("src/a.c", "+return 1;", "int changed(void) {")
        fp = FP.produce_patch_footprint(source_tree="tree", parent="a", child="b",
                                        diff_text=patch, compile_commands=[], depfiles={})
        self.assertEqual(fp.opaque_reasons, ("unmapped_touched_file:src/a.c",))

    def test_footprint_feeds_derive_affected_surface(self):
        fp = produce(diff("src/a.c", "+return x + 1;", "int changed(int x) {"))
        index = SU.build_dependency_index(
            label="fixture", build_dir=".", source_root="/src",
            dep_edges=SU.parse_make_depfile("build/a.c.o: /src/src/a.c", origin_ref="a.d"),
            link_edges=(SU.LinkEdge("lib/liba.so", ("build/a.c.o",), "link.txt"),),
            backend_link_targets={"llama_cpu": ("lib/liba.so",)})
        affected = FP.derive_affected_surface_from_footprint(fp, indexes=(index,))
        self.assertEqual(affected.touched_files, ("src/a.c",))
        self.assertEqual(affected.objects, ("build/a.c.o",))
        self.assertEqual(affected.backends, ("llama_cpu",))


if __name__ == "__main__":
    unittest.main()
