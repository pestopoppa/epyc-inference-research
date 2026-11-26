"""AK-BNSIG-CENSUS-KERNEL-KEYS: trace kernels carry (kernel, grid); signatures attach by kernel+grid, per-record idempotent."""
from autokernel.loop import census_update as cu
from autokernel.loop.census_extract import extract_trace_kernels


def _trace(row="ak-c18-27b-df2-v5i-trace-gpu-x"):
    top = [{"name": "void rms_norm_f32<1024, true, false, false>(float const*, float*, int, long", "grid_wg": "8192x1x1x1024",
            "mean_us": 6.0, "count_per_seg": 122.2, "ms_per_seg": 0.75},
           {"name": "void rms_norm_f32<256, true, false, false>(float const*, float*)", "grid_wg": "1024x8x1x256",
            "mean_us": 4.6, "count_per_seg": 15.0, "ms_per_seg": 0.07},
           {"name": "void unary_gated_op_kernel<&(op_silu(float)), float>(float const*)", "grid_wg": "139264x1x1x256",
            "mean_us": 4.5, "count_per_seg": 60.6, "ms_per_seg": 0.27}]
    by = {"verify": {"gpu_ms_per_seg": 10.0, "top": top}, "draft": {"gpu_ms_per_seg": 1.0, "top": top}}
    return {"row": row, "summary": {"trace": {"tr-V-r0": {"arm": "V", "by_class": by}}}}


def test_extract_verify_only_with_identity():
    o = extract_trace_kernels(_trace())
    assert [(x["kernel"], x["grid"], x["calls_per_round"]) for x in o] == [
        ("rms_norm_f32<1024, true, false, false>", 8192, 122), ("rms_norm_f32<256, true, false, false>", 8192, 15),
        ("unary_gated_op_kernel<&(op_silu(float)), float>", 139264, 61)]
    assert o[0]["bytes_per_call"] is None and o[0]["share_of_round"] == 0.075 and o[0]["n"] == 0
    assert extract_trace_kernels(_trace(row="ak-a3b-trace")) == []


def test_kernel_keyed_attach_and_backfill():
    c = cu.empty_census()
    cu.apply(c, extract_trace_kernels(_trace()), "r", "a" * 64, "t")
    assert len(c["time_only"]) == 3
    c["time_only"].pop(next(iter(c["time_only"])))
    lines = cu.apply(c, extract_trace_kernels(_trace()), "r", "a" * 64, "t")  # applied sha: backfills only the missing kernel record
    assert len(lines) == 1 and len(c["time_only"]) == 3
    sig = {"class": "latency_bound", "rule": "x"}
    grids = {"rms_norm_f32ILi1024ELb1E": 8192, "unary_gated_op_kernelIXadL_ZL7op_silufEEf": 139264, "rms_norm_f32ILi999E": 8192}
    lines = cu.apply_signature(c, dict.fromkeys(grids, sig), "s", "b" * 64, "t", "bld", grids=grids)
    assert sum("signature latency_bound" in x for x in lines) == 2 and sum("skipped" in x for x in lines) == 1
    assert c["time_only"][cu.rid(extract_trace_kernels(_trace())[0])]["signature"]["build"] == "bld"
    assert [x for x in cu.apply_signature(c, dict.fromkeys(grids, sig), "s", "b" * 64, "t", grids=grids) if "signature " in x
            and "skipped" not in x] == [] and c["signature_applied"] == [{"row": "s", "sha256": "b" * 64}]
