"""GPU-POOL-1: device memory kept alive across HIP-graph captures must not come from the
shared ctx.pool() / ggml_cuda_pool_alloc (v10 mmvq_q8_1_graph_cache, 2026-10-04)."""
from . import gates

OWNED_POOL_HOLDER = """\
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -10,0 +11,6 @@
+struct mmvq_q8_1_graph_cache_entry {
+    std::unique_ptr<ggml_cuda_pool_alloc<char>> q8_1;
+};
+struct mmvq_q8_1_graph_cache {
+    ggml_cuda_pool * pool = nullptr;
+    unsigned long long capture_id = 0;
"""

STATIC_ADDRESS = """\
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -10,0 +11,8 @@
+struct mmvq_q8_1_graph_cache { unsigned long long capture_id = 0; void * q8_1 = nullptr; };
+static thread_local mmvq_q8_1_graph_cache q8_1_graph_cache;
+    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
+            q8_1_graph_cache.q8_1 = src1_q8_1.get();
"""

PRIVATE_ARENA = """\
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -10,0 +11,9 @@
+// They used to be held as ggml_cuda_pool_alloc's from the shared ctx.pool() until the next capture.
+struct mmvq_q8_1_graph_arena {
+    void * alloc(size_t size);
+    void rewind();
+};
+    mmvq_q8_1_graph_arena arena; // freed only at context destruction
+        arena.rewind();  // new capture
+            entry.q8_1 = cache.arena.alloc(src1_q8_1_size);
"""

SCOPED_POOL_IN_GRAPH_CODE = """\
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -10,0 +11,3 @@
+    const bool capture_active = is_capturing(stream);
+    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes);
+    launch(src1_q8_1.get(), capture_active);
"""


def test_owned_pool_holder_across_captures_is_refused():
    reason = gates.gpu_graph_pool_hold_refusal(OWNED_POOL_HOLDER)
    assert reason.startswith("GPU-POOL-1") and "private per-context arena" in reason


def test_pool_address_retained_in_static_storage_is_refused():
    reason = gates.gpu_graph_pool_hold_refusal(STATIC_ADDRESS)
    assert reason is not None and "static storage" in reason


def test_private_arena_and_scoped_pool_allocations_are_admitted():
    assert gates.gpu_graph_pool_hold_refusal(PRIVATE_ARENA) is None  # comments ignored
    assert gates.gpu_graph_pool_hold_refusal(SCOPED_POOL_IN_GRAPH_CODE) is None
    assert gates.gpu_graph_pool_hold_refusal(None) is None


def test_pool_holder_without_graph_lifetime_is_not_this_rule():
    no_graph = OWNED_POOL_HOLDER.replace("capture_id", "count").replace("graph_", "")
    assert gates.gpu_graph_pool_hold_refusal(no_graph) is None


def test_scope_gate_refuses_the_cuda_patch_before_build():
    verdict = gates.affected_op_scope(("ggml/src/ggml-cuda/mmvq.cu",),
                                      target_surface="ggml/src/ggml-cuda/mmvq.cu",
                                      target_symbol="mul_mat_vec_q", patch_text=OWNED_POOL_HOLDER)
    assert isinstance(verdict, gates.Verdict) and not verdict.passed
    assert "GPU-POOL-1" in verdict.reason
    admitted = gates.affected_op_scope(("ggml/src/ggml-cuda/mmvq.cu",),
                                       target_surface="ggml/src/ggml-cuda/mmvq.cu",
                                       target_symbol="mul_mat_vec_q", patch_text=PRIVATE_ARENA)
    assert admitted == ("MUL_MAT", "MUL_MAT_ID")
