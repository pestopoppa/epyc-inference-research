"""AKX-AUTO-9: generated default routes (file family -> oracle class)."""
from __future__ import annotations

from . import gates, roofline_coverage as rc, route_defaults as rd

IQ3 = "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp"
IQ4NL = "ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp"
CPU = "ggml/src/ggml-cpu/"


def test_iq3_s_and_iq4_nl_kernels_get_a_scalar_reference_route_with_no_hand_list():
    for path in (IQ3, IQ4NL):
        # Hand list emptied: the default alone must still route the file.
        route = rd.default_route_uncovered_by_hand_list(path, ())
        assert route is not None
        assert route.oracle_class == rd.SCALAR_NMSE
        assert route.numerics == "ppl_contract"
        assert route.route.startswith("default:")


def test_roofline_covers_iq3_iq4nl_when_hand_listed_routes_are_removed(monkeypatch):
    monkeypatch.setattr(gates, "CPU_SOURCE_ROUTES", ())
    monkeypatch.setattr(gates, "CPU_MULTI_FILE_ROUTES", ())
    monkeypatch.setattr(gates, "CPU_ROUTE_PATHS_ALL", ())
    monkeypatch.setattr(rc, "AD_HOC_ADMITTED_PAIRS", frozenset())
    assert rc.route_is_admitted(IQ3, "DequantizerIQ3S") is True
    assert rc.route_is_admitted(IQ4NL, "IQ4_NL_UnpackerU") is True


def test_family_to_oracle_class_mapping():
    cases = {
        CPU + "iqk/iqk_gemm_1bit.cpp": rd.SCALAR_NMSE,
        CPU + "iqk/iqk_gemm_iqk_quants.cpp": rd.SCALAR_NMSE,
        CPU + "iqk/iqk_quantize.cpp": rd.SCALAR_NMSE,
        CPU + "arch/x86/quants.c": rd.SCALAR_NMSE,
        CPU + "llamafile/sgemm.cpp": rd.SCALAR_NMSE,
        CPU + "iqk/iqk_flash_attn.cpp": rd.FA_ANCHOR_BITS,
        CPU + "ggml-cpu.c": rd.MODEL_IDENTITY,
        CPU + "ops.cpp": rd.MODEL_IDENTITY,
        CPU + "vec.cpp": rd.MODEL_IDENTITY,
        CPU + "brand_new_kernel.cpp": rd.MODEL_IDENTITY,   # unclassified compiled file
        "src/llama-model-loader.cpp": rd.MODEL_IDENTITY,
        "src/models/qwen3next.cpp": rd.MODEL_IDENTITY,
    }
    for path, oracle in cases.items():
        route = rd.default_route(path)
        assert route is not None, path
        assert route.oracle_class == oracle, path
        assert oracle in rd.ORACLE_CLASSES


def test_files_with_no_possible_oracle_are_refused_with_a_reason():
    for path in (CPU + "CMakeLists.txt", "CMakeLists.txt", CPU + "cmake/x.cmake",
                 CPU + "common.h", CPU + "iqk/iqk_config.h", CPU + "iqk/iqk_stubs.cpp",
                 CPU + "arch/x86/cpu-feats.cpp", CPU + "kleidiai/kleidiai.cpp",
                 "tests/test-backend-ops.cpp", "README.md", "ggml/src/ggml-cuda/mmq.cu", ""):
        assert rd.default_route(path) is None, path
        assert rd.refusal_for(path), path


def test_path_normalisation_does_not_dodge_a_refusal():
    assert rd.default_route("./ggml/src/ggml-cpu/CMakeLists.txt") is None
    assert rd.default_route(CPU + "../ggml-cpu/CMakeLists.txt") is None


def test_every_hand_listed_route_path_is_classified_consistently():
    for route in gates.CPU_SOURCE_ROUTES:
        found = rd.classify_path(route.path)
        assert isinstance(found, rd.DefaultRoute), route.route
        if route.model_identity:
            assert found.oracle_class == rd.MODEL_IDENTITY, route.route
        if route.numerics == "ppl_contract":
            assert found.oracle_class == rd.SCALAR_NMSE, route.route


def test_default_never_overrides_hand_listed_verdicts():
    # repack.cpp is hand-listed and standing-refused (no op oracle): stays uncovered.
    repack = rc.ROUTE_FILE_SYMBOL[("MUL_MAT_ID", "q6_k")]
    assert rc.route_is_admitted(*repack) is False
    assert rd.default_route_uncovered_by_hand_list(repack[0], gates.CPU_ROUTE_PATHS_ALL) is None
    # a hand-listed file with an unlisted symbol keeps the hand-listed answer too
    assert rc.route_is_admitted(IQ3, "NoSuchSymbol") is False


def test_new_hot_file_is_in_scope_without_a_human():
    assert rc.route_is_admitted(CPU + "iqk/iqk_gemm_1bit.cpp", "anything") is True
    assert rc.route_is_admitted(CPU + "CMakeLists.txt", "anything") is False
