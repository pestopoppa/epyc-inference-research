"""2026-10-03 scope widening: the loop can author Fable CPU seeds 1, 3, 4, 6 and 9.

Scope rules are exercised on the real DS41 anchor tree when it is present (the markers
and helpers must resolve against the code the loop actually edits); the references and
the whole-model gate are exercised on synthetic probe output / injected servers, so
nothing here builds, launches a server or touches a lock.
"""
from __future__ import annotations

import difflib
import json
import math
import re
import struct
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from . import (cpu_fusion_reference as fusion, cpu_norm_reference as norm,
               cpu_route_witness, gates, model_identity)

ANCHOR = Path("/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925")
needs_anchor = pytest.mark.skipif(not (ANCHOR / "ggml/src/ggml-cpu/ggml-cpu.c").is_file(),
                                  reason="DS41 anchor tree not present")


def _scope(path, symbol, head, candidate):
    patch = "".join(difflib.unified_diff(head.splitlines(True), candidate.splitlines(True), n=0))
    return (gates.affected_op_scope((path,), target_surface=path, target_symbol=symbol,
                                    source_text=candidate, pre_source_text=head,
                                    patch_text=patch),
            gates.admit_cpu_route(path, symbol, candidate, head, patch))


def _insert_before(text, marker, block):
    at = text.index(marker)
    return text[:at] + block + text[at:]


def _insert_after_line(text, marker, block):
    at = text.index("\n", text.index(marker)) + 1
    return text[:at] + block + text[at:]


# ------------------------------------------------------------------ scope

@needs_anchor
def test_seed3_fusion_helper_beside_try_fuse_ops_is_admitted():
    path = "ggml/src/ggml-cpu/ggml-cpu.c"
    head = (ANCHOR / path).read_text(encoding="utf-8")
    helper = ("// seed 3: fused rms_norm -> hc_mixes split-K\n"
              "static void ggml_cpu_fused_norm_mixes(const struct ggml_compute_params * params,\n"
              "        struct ggml_tensor * norm, struct ggml_tensor * mm) {\n"
              "    (void) params; (void) norm; (void) mm;\n"
              "}\n\n"
              "static const int ggml_cpu_fused_norm_mixes_min_k = 4096;\n\n")
    call = ("    if (node->op == GGML_OP_RMS_NORM && cgraph->nodes[node_n + 1]->op == "
            "GGML_OP_MUL_MAT) {\n        (void) ggml_cpu_fused_norm_mixes_min_k;\n    }\n")
    candidate = _insert_before(head, "static int ggml_cpu_try_fuse_ops(", helper)
    candidate = _insert_after_line(candidate, "    struct ggml_tensor * node = cgraph->nodes[node_n];", call)
    ops, (route, refusal) = _scope(path, "ggml_cpu_try_fuse_ops", head, candidate)
    assert refusal is None and route.route == "cpu_graph_sync"
    assert ops == route.ops and "RMS_NORM" in ops and "MUL_MAT" in ops
    assert cpu_route_witness.WITNESSES["cpu_graph_sync"].norm_mulmat


@needs_anchor
@pytest.mark.parametrize("block,why", [
    ("int ggml_cpu_global_counter = 0;\n", "must start with `static`"),
    ("#define ggml_barrier(tp) ((void) 0)\n", "only `#include <...>`"),
    ('#include "evil.h"\n', "only `#include <...>`"),
    ("static void helper(void) {\n", "brace-balanced"),
])
def test_file_scope_insertions_that_are_not_new_static_helpers_are_refused(block, why):
    path = "ggml/src/ggml-cpu/ggml-cpu.c"
    head = (ANCHOR / path).read_text(encoding="utf-8")
    candidate = _insert_before(head, "static int ggml_cpu_try_fuse_ops(", block)
    verdict, (_route, refusal) = _scope(path, "ggml_cpu_try_fuse_ops", head, candidate)
    assert not verdict.passed and why in refusal


@needs_anchor
def test_helper_inside_another_function_and_edits_elsewhere_stay_refused():
    path = "ggml/src/ggml-cpu/ggml-cpu.c"
    head = (ANCHOR / path).read_text(encoding="utf-8")
    inside = _insert_after_line(head, "void ggml_compute_forward_mul_mat(", "")
    # a "static" line inside a non-admitted function body (depth > 0) is not a helper
    body = head.index("static void ggml_compute_forward_mul_mat_id(")
    nested = head[:body] + head[body:].replace(
        "{\n", "{\n    static int counter = 0; (void) counter;\n", 1)
    verdict, (_r, refusal) = _scope(path, "ggml_cpu_try_fuse_ops", head, nested)
    assert not verdict.passed and "not at file scope" in refusal
    assert inside == head


@needs_anchor
def test_seed6_mul_mat_body_route():
    path = "ggml/src/ggml-cpu/ggml-cpu.c"
    head = (ANCHOR / path).read_text(encoding="utf-8")
    helper = ("static inline bool ggml_cpu_small_prelude(int64_t ne11) {\n"
              "    return ne11 > 0 && ne11 <= 4;\n}\n\n")
    candidate = _insert_before(head, "void ggml_compute_forward_mul_mat(", helper)
    candidate = _insert_after_line(candidate, "    const int nth = params->nth;\n    \n"
                                   if "    const int nth = params->nth;\n    \n" in candidate
                                   else "    GGML_TENSOR_BINARY_OP_LOCALS",
                                   "    (void) ggml_cpu_small_prelude(ne11);\n")
    for symbol in ("ggml_compute_forward_mul_mat", "ggml_compute_forward_mul_mat_one_chunk"):
        ops, (route, refusal) = _scope(path, symbol, head, candidate)
        assert (route.route, refusal, ops) == ("cpu_mul_mat_body", None, ("MUL_MAT",))
    witness = cpu_route_witness.WITNESSES["cpu_mul_mat_body"]
    assert witness.breakpoint == ("break", "ggml_compute_forward_mul_mat")
    assert set(witness.quants) == {"Q8_0", "Q4_K", "Q5_K", "F16"}
    # MUL_MAT_ID is not this route
    mmid = head.index("static void ggml_compute_forward_mul_mat_id(")
    bad = head[:mmid] + head[mmid:].replace("{\n", "{\n    int x = 0; (void) x;\n", 1)
    verdict, _ = _scope(path, "ggml_compute_forward_mul_mat", head, bad)
    assert not verdict.passed


@needs_anchor
def test_seed1_loader_placement_route_admits_includes_helpers_and_body_edits():
    path = "src/llama-model-loader.cpp"
    head = (ANCHOR / path).read_text(encoding="utf-8")
    candidate = _insert_after_line(head, "#include <unistd.h>",
                                   "#include <sys/syscall.h>\n#include <numaif.h>\n")
    helper = ("static void llama_place_rows_by_node(void * data, size_t nbytes, int64_t nrows) {\n"
              "    (void) data; (void) nbytes; (void) nrows;\n}\n\n")
    candidate = _insert_before(candidate, "bool llama_model_loader::load_all_data_parallel(", helper)
    start = candidate.index("bool llama_model_loader::load_all_data(")
    body = candidate.index("{\n", start) + 2
    candidate = candidate[:body] + "    llama_place_rows_by_node(nullptr, 0, 0);\n" + candidate[body:]
    ops, (route, refusal) = _scope(path, "llama_model_loader::load_all_data", head, candidate)
    assert (route.route, refusal, ops) == ("cpu_weight_placement", None, ())
    assert route.model_identity
    # every other loader function stays outside the route
    other = head.index("bool llama_model_loader::load_all_data_parallel(")
    before = head.rindex("\n}\n", 0, other)
    bad = head[:before] + "\n    int stray = 0; (void) stray;" + head[before:]
    verdict, _ = _scope(path, "load_all_data", head, bad)
    assert not verdict.passed


@needs_anchor
def test_seed4_vector_sumsq_lands_on_the_numerics_route_and_barrier_does_not():
    path = "ggml/src/ggml-cpu/ops.cpp"
    head = (ANCHOR / path).read_text(encoding="utf-8")
    start = head.index("static void ggml_compute_forward_rms_norm_f32(")
    # The anchor is the live accumulator: its sum-of-squares line moved from HEAD's
    # `ggml_float sum = 0.0;` to an eight-double-chain declaration (2026-10 keeps).
    found = re.compile(r"ggml_float sum = 0\.0;|double sum0 = 0\.0[^;\n]*;").search(head, start)
    line, decl = found.start(), found.group(0)
    vector = head[:line] + decl + " ggml_float acc[8] = {0};" + head[line + len(decl):]
    ops, (route, refusal) = _scope(path, "ggml_compute_forward_rms_norm_f32", head, vector)
    assert (route.route, refusal, ops) == ("cpu_norm_numerics", None,
                                           ("RMS_NORM", "RMS_NORM_MUL_ADD"))
    barrier = head[:line] + decl + " ggml_barrier(params->threadpool);" + \
        head[line + len(decl):]
    verdict, (_r, refusal) = _scope(path, "ggml_compute_forward_rms_norm_f32", head, barrier)
    assert not verdict.passed and "cpu_norm_rowsplit" in verdict.reason \
        and "cpu_norm_numerics" in verdict.reason


def test_route_reference_uses_the_admitting_route():
    from .iqk_witness import Result
    with mock.patch.object(cpu_route_witness, "check",
                           return_value=Result("pass", "ok", "")) as check:
        gates.check_cpu_route_reference(Path("/b"), Path("/s"), resolved_recipe=object(),
                                        path="ggml/src/ggml-cpu/ops.cpp",
                                        target_symbol="ggml_compute_forward_rms_norm_f32",
                                        route_name="cpu_norm_numerics")
    assert check.call_args.kwargs["route"] == "cpu_norm_numerics"
    refused = gates.check_cpu_route_reference(
        Path("/b"), Path("/s"), resolved_recipe=object(), path="ggml/src/ggml-cpu/ops.cpp",
        target_symbol="ggml_compute_forward_rms_norm_f32", route_name="dense_q8_tinyblas")
    assert (refused.gate, refused.passed) == ("oracle_unavailable", False)


def test_every_route_has_a_witness_and_run_wires_the_admitting_route():
    assert {r.route for r in gates.CPU_SOURCE_ROUTES} <= set(cpu_route_witness.WITNESSES)
    source = (Path(__file__).with_name("run.py")).read_text(encoding="utf-8")
    assert "route_name=admitted_route.route" in source
    assert "gates.check_model_output_identity(" in source


def test_model_identity_route_is_never_judged_by_a_route_witness(tmp_path):
    recipe = SimpleNamespace(backend="cpu", launch_env=(("GGML_IQK", "1"),),
                             validate_launch=lambda *a: None, template=None, port=1,
                             topology_prefix=())
    (tmp_path / "src").mkdir()
    (tmp_path / "src/llama-model-loader.cpp").write_text("")
    result = cpu_route_witness.check(tmp_path, resolved_recipe=recipe, source_root=tmp_path,
                                     route="cpu_weight_placement",
                                     source_path="src/llama-model-loader.cpp")
    assert result.status == "unavailable"


# ------------------------------------------------------------------ RMS_NORM tolerance

def _norm_output(case, rows_bytes, digests=None):
    _rows, _w, digest = norm._inputs(case)
    lines = [f"{norm.MARKER} {case.mode} {' '.join(map(str, case.ne))} "
             f"{norm._f32_bits(case.eps):08x} {norm.THREADS} {case.reps} {norm.SEED}",
             f"I {digest:016x}"]
    flat = b"".join(rows_bytes)
    for rep in range(case.reps):
        value = (digests or {}).get(rep, norm._fnv1a(flat))
        lines.append(f"D {rep} {value:016x}")
    lines += [f"O {r} {row.hex()}" for r, row in enumerate(rows_bytes)]
    return "\n".join(lines)


def test_norm_tolerance_mode_accepts_reordered_arithmetic_and_rejects_real_errors():
    case = norm.NormCase("t", "plain", (512, 3, 1, 1), 1e-6)
    rows, _w, _d = norm._inputs(case)
    exact = [norm.head_semantics(row, case.eps).tobytes() for row in rows]
    assert norm.compare(case, _norm_output(case, exact)).status == "pass"
    assert norm.compare(case, _norm_output(case, exact), bit_exact=False).status == "pass"
    # float64-rounded (a double-accumulator variant): not HEAD's bits, well inside 2^-16
    ref = [array("f", norm.float64_reference(row, case.eps)).tobytes() for row in rows]
    if ref != exact:
        assert norm.compare(case, _norm_output(case, ref)).status == "wrong"
    loose = norm.compare(case, _norm_output(case, ref), bit_exact=False)
    assert loose.status == "pass" and '"tolerance_mode": true' in loose.detail
    # a 2^-13 relative error (a lost lane, a wrong scale) fails tolerance mode
    skewed = [array("f", [v * (1 + 2.0 ** -13) for v in norm.float64_reference(row, case.eps)]).tobytes()
              for row in rows]
    assert norm.compare(case, _norm_output(case, skewed), bit_exact=False).status == "wrong"
    # a nondeterministic repetition fails tolerance mode too
    racy = _norm_output(case, ref, digests={1: 0x1234})
    assert norm.compare(case, racy, bit_exact=False).status == "wrong"


# ------------------------------------------------------------------ hc_mixes fusion fixture

SMALL = fusion.FusionCase("small", "positive", 256, 2, 3, threads=4, reps=2)


def _fusion_output(case, values, digests=None):
    _x, _w, digest = fusion._inputs(case)
    rows = [array("f", values[t]).tobytes() for t in range(case.nt)]
    lines = [f"{fusion.MARKER} {case.mode} {case.K} {case.nt} {case.M} "
             f"{fusion._f32_bits(fusion.EPS):08x} {case.threads} {case.reps} {fusion.SEED}",
             f"I {digest:016x}"]
    for rep in range(case.reps):
        lines.append(f"D {rep} {(digests or {}).get(rep, fusion._fnv1a(b''.join(rows))):016x}")
    lines += [f"O {t} {row.hex()}" for t, row in enumerate(rows)]
    return "\n".join(lines)


@pytest.mark.parametrize("mode", ["positive", "signed"])
def test_fusion_reference_accepts_float_results_and_rejects_structural_errors(mode):
    case = fusion.FusionCase("small", mode, 256, 2, 3, threads=4, reps=2)
    x, w, _digest = fusion._inputs(case)
    good = [[ref for ref, _s in row] for row in fusion.reference(case, x, w)]
    assert fusion.compare(case, _fusion_output(case, good)).status == "pass"
    # HEAD-like error: normed activations rounded to F16 before the dot
    rows = fusion.reference(case, x, w)
    assert all(scale > 0 for row in rows for _ref, scale in row)
    # drop the last 1/16 of K for column t=0 (a missing split-K slice)
    sub = fusion.FusionCase("sub", mode, 256, 2, 3)
    eps = struct.unpack("<f", struct.pack("<f", fusion.EPS))[0]
    col = x[:case.K]
    inv = 1.0 / math.sqrt(math.fsum(v * v for v in col) / case.K + eps)
    dropped = [math.fsum(a * b for a, b in zip(w[m * case.K:(m + 1) * case.K - 16], col)) * inv
               for m in range(case.M)]
    bad = [dropped, good[1]]
    if mode == "positive":
        assert fusion.compare(case, _fusion_output(case, bad)).status == "wrong"
    assert sub.K == case.K
    # missing normalisation
    unscaled = [[ref / inv for ref in good[0]], good[1]]
    assert fusion.compare(case, _fusion_output(case, unscaled)).status == "wrong"
    # a racy reduction
    assert fusion.compare(case, _fusion_output(case, good, digests={1: 7})).status == "wrong"


def test_fusion_reference_refuses_fixture_drift_as_unavailable():
    x, w, _d = fusion._inputs(SMALL)
    good = [[ref for ref, _s in row] for row in fusion.reference(SMALL, x, w)]
    text = _fusion_output(SMALL, good)
    hashline = next(line for line in text.splitlines() if line.startswith("I "))
    text = text.replace(hashline, "I " + "0" * 16, 1)
    with pytest.raises(ValueError):
        fusion.compare(SMALL, text)


def test_fusion_probe_mirrors_the_ds41_graph():
    source = fusion.PROBE.read_text(encoding="utf-8")
    for token in ("ggml_reshape_2d(ctx, x3, K, nt)", "ggml_rms_norm(ctx, flat, eps)",
                  "GGML_TYPE_F16, K, M", "ggml_mul_mat(ctx, wt, norm)"):
        assert token in source
    assert any(c.K == 20480 and c.M == 24 and c.nt == 3 for c in fusion.CASES)


# ------------------------------------------------------------------ whole-model identity

def _recipe(build, *, port=8090, env=None):
    return SimpleNamespace(
        backend="cpu", build_dir=build, port=port, template=None,
        command_argv=(f"{build}/bin/llama-server", "-m", "/m.gguf", "--port", str(port)),
        topology_prefix=("numactl", "--interleave=all"),
        launch_env=tuple(sorted((env or {"LD_LIBRARY_PATH": f"{build}/bin"}).items())))


REQUESTS = ((f"p{i}", json.dumps({"prompt": "x", "temperature": 0, "n_predict": 8}).encode())
            for i in range(3))


def _check(serves, *, requests=None):
    calls = []

    def serve_fn(recipe, selected):
        calls.append((recipe.build_dir, len(selected)))
        out = serves.pop(0)
        if isinstance(out, Exception):
            raise out
        return [(pid, digest, "") for (pid, _body), digest in zip(selected, out)]

    reqs = tuple(requests if requests is not None else
                 ((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode())
                  for i in range(3)))
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=reqs, serve_fn=serve_fn)
    return result, calls


def test_model_identity_passes_on_identical_completions_using_two_requests():
    result, calls = _check([["h1", "h2"], ["h1", "h2"]])
    assert result.status == "pass" and calls == [("/a", 2), ("/c", 2)]


def test_model_identity_wrong_only_when_the_anchor_reproduces_itself():
    result, calls = _check([["h1", "h2"], ["h1", "XX"], ["h1", "h2"]])
    assert result.status == "wrong" and len(calls) == 3
    result, _ = _check([["h1", "h2"], ["h1", "XX"], ["h1", "zz"]])
    assert result.status == "unavailable" and "anchor's own" in result.reason


def test_model_identity_failures_are_never_a_verdict():
    result, _ = _check([RuntimeError("boot")])
    assert result.status == "unavailable"
    nongreedy = ((f"p{i}", json.dumps({"prompt": "x", "temperature": 0.7}).encode())
                 for i in range(2))
    result, calls = _check([], requests=nongreedy)
    assert result.status == "unavailable" and calls == []
    shifted = model_identity.check(anchor_recipe=_recipe("/a"),
                                   candidate_recipe=_recipe("/c", port=9999),
                                   requests=(("p", b'{"temperature":0}'),),
                                   serve_fn=lambda *a: [])
    assert shifted.status == "unavailable"


def test_model_identity_gate_maps_to_verdicts():
    with mock.patch.object(model_identity, "check",
                           return_value=model_identity.IdentityResult("wrong", "r", "d")):
        verdict = gates.check_model_output_identity(anchor_recipe=None, candidate_recipe=None,
                                                    requests=())
    assert (verdict.gate, verdict.passed) == ("reference_comparison", False)
    with mock.patch.object(model_identity, "check",
                           return_value=model_identity.IdentityResult("unavailable", "r")):
        verdict = gates.check_model_output_identity(anchor_recipe=None, candidate_recipe=None,
                                                    requests=())
    assert (verdict.gate, verdict.passed) == ("oracle_unavailable", False)


@needs_anchor
def test_balanced_conditional_helpers_are_admitted_and_open_ones_refused():
    path = "ggml/src/ggml-cpu/ggml-cpu.c"
    head = (ANCHOR / path).read_text(encoding="utf-8")
    helper = ("#if defined(__AVX512F__)\n"
              "static inline void ggml_cpu_prefetch_rows(const char * p, int n) {\n"
              "#ifdef __GNUC__\n"
              "    for (int i = 0; i < n; ++i) { __builtin_prefetch(p + 64*i, 0, 1); }\n"
              "#endif\n"
              "}\n"
              "#else\n"
              "static inline void ggml_cpu_prefetch_rows(const char * p, int n) { (void) p; (void) n; }\n"
              "#endif\n\n")
    ok = _insert_before(head, "static thread_ret_t ggml_graph_compute_thread(", helper)
    ops, (route, refusal) = _scope(path, "ggml_graph_compute_thread", head, ok)
    assert (route.route, refusal) == ("cpu_graph_sync", None)
    open_if = _insert_before(head, "static thread_ret_t ggml_graph_compute_thread(",
                             "#if 0\nstatic int unused_flag = 0;\n")
    verdict, (_r, refusal) = _scope(path, "ggml_graph_compute_thread", head, open_if)
    assert not verdict.passed and "not closed" in refusal
