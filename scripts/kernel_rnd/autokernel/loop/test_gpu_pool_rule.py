"""GPU-POOL-1: device memory kept alive across HIP-graph captures must not come from the
shared ctx.pool() / ggml_cuda_pool_alloc (v10 mmvq_q8_1_graph_cache, 2026-10-04)."""
import pytest

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


def test_pool_holder_without_a_graph_token_is_still_refused():
    """GGML_HIP_GRAPHS=ON captures every op: the rule does not wait for a `graph` or
    `capture` token in the patch (integration audit 2026-10-04)."""
    no_graph = OWNED_POOL_HOLDER.replace("capture_id", "count").replace("graph_", "")
    assert gates.gpu_graph_pool_hold_refusal(no_graph).startswith("GPU-POOL-1")


@pytest.mark.parametrize("snippet", [
    "+thread_local mmvq_cache q8_1_cache;\n+    q8_1_cache.q8_1 = src1_q8_1.get();",
    "+static\n+thread_local mmvq_cache q8_1_cache;\n+    q8_1_cache.q8_1 = src1_q8_1.ptr;",
    "+    cache[0] = src1_q8_1.get();",
    "+    ctx.q8_1_keep = src1_q8_1.get();",
    "+    char * keep = src1_q8_1.get();\n+    cache.q8_1 = keep;",
    "+    static auto & pool_ref = ctx.pool();",
    "+    static char * q8_1_keep = src1_q8_1.get();",
])
def test_retention_variants_are_refused(snippet):
    patch = "+++ b/ggml/src/ggml-cuda/mmvq.cu\n@@ -10,0 +11,2 @@\n" + snippet + "\n"
    reason = gates.gpu_graph_pool_hold_refusal(patch)
    assert reason is not None and reason.startswith("GPU-POOL-1"), snippet


def test_scoped_pool_use_without_retention_is_admitted():
    patch = ("+++ b/ggml/src/ggml-cuda/mmvq.cu\n@@ -10,0 +11,3 @@\n"
             "+    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes);\n"
             "+    char * q8 = src1_q8_1.get();\n"
             "+    launch(q8, ne00, stream);\n")
    assert gates.gpu_graph_pool_hold_refusal(patch) is None


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


# ---- multi-row verify routes (27B GPU verify campaign, 2026-10-04) ----------------------

@pytest.mark.parametrize("path,symbol", [
    ("ggml/src/ggml-cuda/mmvq.cu", "ggml_cuda_should_use_mmvq"),
    ("ggml/src/ggml-cuda/mmvq.cu", "calc_rows_per_block"),
    ("ggml/src/ggml-cuda/mmvq.cuh", "MMVQ_MAX_BATCH_SIZE"),
    ("ggml/src/ggml-cuda/mmq.cuh", "launch_mul_mat_q"),
    ("ggml/src/ggml-cuda/mmq.cuh", "mul_mat_q_stream_k_fixup"),
    ("ggml/src/ggml-cuda/mmq-config-cdna.cuh", "mmq_get_nwarps_device"),
    ("ggml/src/ggml-cuda/mmq-load-tiles.cuh", "load_tiles_q8_0"),
])
def test_verify_path_matmul_routes_admit_the_mul_mat_suites(path, symbol):
    assert gates.affected_op_scope((path,), target_surface=path, target_symbol=symbol) \
        == ("MUL_MAT", "MUL_MAT_ID")


@pytest.mark.parametrize("path,symbol", [
    ("ggml/src/ggml-cuda/mmq.cuh", "unrelated_helper"),
    ("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml_cuda_mul_mat"),
    ("ggml/src/ggml-cuda/mmq-config-ampere.cuh", "mmq_get_nwarps_device"),
])
def test_verify_path_routes_stay_closed_elsewhere(path, symbol):
    verdict = gates.affected_op_scope((path,), target_surface=path, target_symbol=symbol)
    assert isinstance(verdict, gates.Verdict) and not verdict.passed


def test_verify_path_header_route_still_applies_gpu_pool_rule():
    path = "ggml/src/ggml-cuda/mmq.cuh"
    verdict = gates.affected_op_scope((path,), target_surface=path, target_symbol="launch_mul_mat_q",
                                      patch_text=OWNED_POOL_HOLDER)
    assert isinstance(verdict, gates.Verdict) and "GPU-POOL-1" in verdict.reason
