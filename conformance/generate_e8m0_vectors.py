#!/usr/bin/env python3
"""Generate cross-backend E8M0 conformance vectors (CONFORMANCE-VECTORS-1).

WHY THESE EXIST
    An audit on 2026-08-03 found THREE different answers for the same E8M0 byte
    across seven backend sites in our own tree. Nothing had compared them because
    nothing ran. These vectors are what "something running" looks like.

THE DUAL-CONTRACT DESIGN, which is the load-bearing part
    The same format appears as SEPARATE CONTRACTS -- one per documented behaviour --
    so a backend cannot satisfy one by breaking another, and so a legitimate
    divergence is recorded as documented-divergent rather than as a bug.

VALUES ARE DERIVED, NOT TRANSCRIBED
    Each contract is computed from its own stated rule (below), not copied out of
    the implementation it describes. If an implementation drifts, the vectors
    disagree with it -- which is the entire point. Transcribing the implementation
    would produce vectors that can never fail.

COVERAGE IS EDGE-WEIGHTED
    Codes 0, 1, 2, 126, 127, 128, 253, 254, 255: both boundaries, the identity
    point (127 -> 2^0), and one step either side of each.
"""
import json
import struct
from pathlib import Path

CODES = [0, 1, 2, 126, 127, 128, 253, 254, 255]
OUT = Path(__file__).resolve().parent / "vectors"


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fmt(bits: int) -> dict:
    f = bits_to_float(bits)
    if f != f:
        v = "NaN"
    elif f == float("inf"):
        v = "Infinity"
    elif f == float("-inf"):
        v = "-Infinity"
    else:
        v = repr(f)
    return {"bits": f"0x{bits:08x}", "value": v}


# ---------------------------------------------------------------- contracts
def mx_spec(code: int) -> int:
    """OCP MX spec: 0xFF is NaN; otherwise the value is 2^(code-127).

    code 0 is the smallest exponent and denotes 2^-127, which is subnormal in
    binary32 and therefore has no normalized encoding -- hence the explicit
    pattern rather than `code << 23`.
    """
    if code == 0xFF:
        return 0x7FC00000                      # quiet NaN
    if code == 0:
        return 0x00400000                      # 2^-127, subnormal
    return code << 23                          # 2^(code-127)


def ggml_full(code: int) -> int:
    """ggml `ggml_e8m0_to_fp32`: identical to the spec EXCEPT 0xFF is NOT special-cased.

    The NaN branch is present in source but commented out ("disabled as we don't
    need to handle NaNs"), so 0xFF falls through to `code << 23` = 0x7F800000 = +Inf.
    Metal, SYCL, Vulkan, OpenCL and the HIP fallback all agree with this.
    """
    if code == 0:
        return 0x00400000
    return code << 23


def ggml_half(code: int) -> int:
    """ggml `ggml_e8m0_to_fp32_half`: exactly half of the above, used by MXFP4
    because its E0M2 values are doubled.

    Rule: code < 2 -> 0x00200000 << code ; else (code-1) << 23.
    Consequence at the edge: 0xFF -> 254<<23 = 0x7F000000 = 2^127, which is FINITE.
    This is the path every CPU MXFP4 decode actually takes.
    """
    if code < 2:
        return 0x00200000 << code
    return (code - 1) << 23


CONTRACTS = {
    "e8m0_mx_spec": dict(
        fn=mx_spec,
        spec="OCP Microscaling Formats (MX) v1.0",
        note=("Normative spec behaviour: 0xFF is NaN. Kept as a SEPARATE contract from "
              "the ggml ones so a backend cannot satisfy one by breaking another."),
        checked_by="conformance/test_e8m0_vectors.py::test_contract_matches_reference",
    ),
    "e8m0_ggml_full": dict(
        fn=ggml_full,
        spec="ggml ggml_e8m0_to_fp32 (ggml/src/ggml-impl.h:439)",
        note=("DOCUMENTED DIVERGENCE FROM SPEC, not a bug: the 0xFF->NaN branch is present "
              "but commented out, so 0xFF decodes to +Inf. Legitimate because 0xFF is "
              "REJECTED AT LOAD by validate_e_e8m0 (ggml/src/ggml-quants.c:5366), so a GGUF "
              "carrying it is refused -- 0xFF is treated as reserved, which is what the MX "
              "spec intends. NOTE: this function has ZERO call sites in the tree."),
        checked_by="conformance/test_e8m0_vectors.py::test_contract_matches_reference",
    ),
    "e8m0_ggml_half": dict(
        fn=ggml_half,
        spec="ggml ggml_e8m0_to_fp32_half (ggml/src/ggml-impl.h:477)",
        note=("The path CPU MXFP4 decode actually takes. 0xFF -> 2^127, FINITE -- a third "
              "distinct answer for the same byte. Also legitimate under the load gate."),
        checked_by="conformance/test_e8m0_vectors.py::test_contract_matches_reference",
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, c in CONTRACTS.items():
        doc = {
            "format": "E8M0",
            "contract": name,
            "spec": c["spec"],
            "note": c["note"],
            "checked_by": c["checked_by"],
            "coverage": ("edge-weighted: both boundaries, the identity point (127 -> 2^0), "
                         "and one step either side of each"),
            "cases": [dict(code=code, **fmt(c["fn"](code))) for code in CODES],
        }
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {p.relative_to(OUT.parent.parent)}  ({len(doc['cases'])} cases)")

    # The divergence table is the reason the instrument exists; emit it as data so a
    # future reader does not have to re-derive it from three files.
    div = {
        "format": "E8M0",
        "byte": "0xFF",
        "why_it_matters": ("Three different answers for the same byte across seven backend sites, "
                           "undetected because nothing compared them."),
        "answers": {
            "mx_spec": fmt(mx_spec(0xFF)),
            "ggml_e8m0_to_fp32 (CPU/Metal/SYCL/Vulkan/OpenCL/HIP fallback)": fmt(ggml_full(0xFF)),
            "ggml_e8m0_to_fp32_half (CPU MXFP4, the live path)": fmt(ggml_half(0xFF)),
        },
        "cuda_note": ("CUDA >= 12.8 uses __nv_cvt_e8m0_to_bf16raw and matches the SPEC (NaN); "
                      "below that, and on our HIP build where CUDART_VERSION is undefined, the "
                      "fallback matches ggml_e8m0_to_fp32 (+Inf). "
                      "Site: ggml/src/ggml-cuda/common.cuh:814-822."),
        "legitimising_fact": ("validate_e_e8m0 (ggml/src/ggml-quants.c:5366) REJECTS 0xFF at load, "
                              "wired for MXFP4 and called from llama-model-loader.cpp. A GGUF "
                              "carrying 0xFF is refused, so the divergence is unreachable in "
                              "practice -- documented-divergent, not broken."),
    }
    p = OUT / "e8m0_divergence.json"
    p.write_text(json.dumps(div, indent=2) + "\n")
    print(f"wrote {p.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
