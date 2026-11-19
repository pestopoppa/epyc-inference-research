"""AKX-AUTO-9: live admission through generated default routes (gates.default_cpu_route)."""
from __future__ import annotations

import pytest

from . import cpu_route_witness, gates

IQ3 = "ggml/src/ggml-cpu/quants.c"
IQ4NL = "ggml/src/ggml-cpu/iqk/iqk_gemm_hypothetical.cpp"
GRAPHOPS = "ggml/src/ggml-cpu/binary-ops.cpp"
PRE = "int f(int a) {\n    return a + 1;\n}\n"
POST = "int f(int a) {\n    return a + 2;\n}\n"
PATCH = "@@ -2,1 +2,1 @@ int f(int a) {\n-    return a + 1;\n+    return a + 2;\n"


def _scope(path, post=POST, patch=PATCH, symbol="whatever"):
    return gates.affected_op_scope((path,), target_surface=path, target_symbol=symbol,
                                   source_text=post, pre_source_text=PRE, patch_text=patch)


def test_unnamed_quant_file_is_admitted_under_ppl_contract():
    for path in (IQ3, IQ4NL):
        route = gates.default_cpu_route(path)
        assert route.generated and route.numerics == "ppl_contract" and not route.model_identity
        assert _scope(path) == ("MUL_MAT", "MUL_MAT_ID")
        got, refusal = gates.admit_cpu_route(path, "x", POST, PRE, PATCH)
        assert refusal is None and got.route == "default:quant_kernel"


def test_unnamed_graph_file_is_admitted_under_model_identity():
    route = gates.default_cpu_route(GRAPHOPS)
    assert route.model_identity and route.numerics == "bit_exact" and route.ops == ()
    assert route.identity_targets == "peers" and route.identity_repeats == 3
    assert _scope(GRAPHOPS) == ()


def test_every_generated_route_has_a_witness_and_valid_numerics():
    for path in (IQ3, IQ4NL, GRAPHOPS, "ggml/src/ggml-cpu/newfile.cpp"):
        route = gates.default_cpu_route(path)
        assert route is not None, path
        assert route.numerics in ("bit_exact", "ppl_contract")
        assert not (route.numerics == "ppl_contract" and route.model_identity)
        w = cpu_route_witness.WITNESSES[route.route]
        assert w.reference == ("ppl_contract" if route.numerics == "ppl_contract"
                               else "model_identity")


@pytest.mark.parametrize("added,why", [
    ("    system(\"id\");", "process/filesystem"),
    ("#define X 1", "`#` outside"),
    ("    /* open", "block comment"),
    ("    int x = 1; \\", "continuation"),
])
def test_added_lines_keep_the_lexical_refusals(added, why):
    patch = f"@@ -2,1 +2,1 @@\n-    return a + 1;\n+{added}\n"
    out = _scope(IQ3, patch=patch)
    assert isinstance(out, gates.Verdict) and not out.passed and why in out.reason


def test_empty_patch_or_source_refuses():
    assert isinstance(_scope(IQ3, patch=""), gates.Verdict)
    assert isinstance(_scope(IQ3, post=""), gates.Verdict)


@pytest.mark.parametrize("path", [
    "ggml/src/ggml-cpu/CMakeLists.txt", "ggml/src/ggml-cpu/iqk/iqk_config.h",
    "ggml/src/ggml-cpu/iqk/iqk_stubs.cpp", "ggml/src/ggml-cuda/mmq.cu",
    "ggml/src/ggml-cpu/amx/amx.cpp", "tests/test-backend-ops.cpp",
    "ggml/src/ggml-cpu/iqk/iqk_flash_attn.cpp"])
def test_refused_and_flash_attention_files_get_no_generated_route(path):
    assert gates.default_cpu_route(path) is None


def test_hand_listed_paths_keep_their_hand_listed_verdict():
    # sgemm.cpp and iqk_mul_mat.cpp are hand-listed: an unlisted symbol stays refused.
    for path in ("ggml/src/ggml-cpu/llamafile/sgemm.cpp", "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp",
                 "ggml/src/ggml-cpu/repack.cpp", "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp",
                 "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"):
        assert gates.default_cpu_route(path) is None
        out = _scope(path, symbol="not_a_listed_symbol")
        assert isinstance(out, gates.Verdict) and not out.passed


def test_flag_off_restores_hand_listed_only_admission(monkeypatch):
    monkeypatch.setenv(gates.DEFAULT_ROUTE_ADMISSION_ENV, "0")
    assert gates.default_cpu_route(IQ3) is None
    out = _scope(IQ3)
    assert isinstance(out, gates.Verdict) and not out.passed
    monkeypatch.setenv(gates.DEFAULT_ROUTE_ADMISSION_ENV, "1")
    assert _scope(IQ3) == ("MUL_MAT", "MUL_MAT_ID")


def test_fold_boundary_treats_generated_quant_paths_as_ppl_contract_relevant(monkeypatch):
    monkeypatch.setenv(gates.DEFAULT_ROUTE_ADMISSION_ENV, "0")  # relevance is flag-independent
    assert gates.ppl_contract_relevant("ggml/src/ggml-cpu/quants.c")
    assert gates.ppl_contract_relevant("ggml/src/ggml-cpu/arch/x86/quants.c")
    assert not gates.ppl_contract_relevant("ggml/src/ggml-cpu/binary-ops.cpp")


def test_thresholds_are_untouched():
    # The widening adds scope, not tolerance: the contract constants are the pre-existing ones.
    assert gates.PPL_REL_BAR == 0.005
    assert gates.PPL_CONTRACT_AGREEMENT_FLOOR == 0.98
    assert gates.PPL_CONTRACT_CANARY_TOKENS == 1024
