"""2026-10-04 structural routes: cpu_graph_sched (R1), cpu_model_fused_op (R2) and
cpu_graph_optimize (R3) -- the loop can author the structural CPU seeds.

Scope rules run on the real DS41 anchor tree when present (markers, bodies and case
boundaries must resolve against the code the loop edits); the identity gate runs on
injected servers, so nothing here builds, launches a server or touches a lock.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import cpu_route_witness, gates, model_identity

ANCHOR = Path("/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925")
needs_anchor = pytest.mark.skipif(not (ANCHOR / "src/models/qwen4exp.cpp").is_file(),
                                  reason="DS41 anchor tree (with qwen4exp) not present")

CPU_C = "ggml/src/ggml-cpu/ggml-cpu.c"


def _head(path):
    return (ANCHOR / path).read_text(encoding="utf-8")


def _patch(head, candidate):
    return "".join(difflib.unified_diff(head.splitlines(True), candidate.splitlines(True), n=0))


def _scope(edits, symbol, surface):
    """edits: {path: candidate text}. Returns (affected_op_scope result, file_texts)."""
    texts = {path: (text, _head(path), _patch(_head(path), text)) for path, text in edits.items()}
    single = texts[next(iter(texts))] if len(texts) == 1 else (None, None, None)
    return gates.affected_op_scope(tuple(texts), target_surface=surface, target_symbol=symbol,
                                   source_text=single[0], pre_source_text=single[1],
                                   patch_text=single[2], file_texts=texts), texts


def _before(text, marker, block):
    at = text.index(marker)
    return text[:at] + block + text[at:]


def _after_line(text, marker, block):
    at = text.index("\n", text.index(marker)) + 1
    return text[:at] + block + text[at:]


# ------------------------------------------------------------------ R1 cpu_graph_sched

def _cogroup_patch(*, use_ref=True, pragma_in="", field_at_tail=True):
    text = _head(CPU_C)
    field = "    atomic_int GGML_CACHE_ALIGN group_chunk[8]; // per-group chunk counter\n"
    if field_at_tail:
        text = _after_line(text, "    enum ggml_status ec;", field)
    else:
        text = _after_line(text, "    struct ggml_cplan  * cplan;", field)
    text = text.replace("    struct ggml_threadpool * threadpool;\n    int ith;\n};",
                        "    struct ggml_threadpool * threadpool;\n    int ith;\n"
                        "    int group; // co-group id\n};", 1)
    helper = ("static void ggml_group_barrier(struct ggml_threadpool * tp, int group) {\n"
              "    (void) tp; (void) group;\n}\n\n")
    text = _before(text, "enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph,", helper)
    guard = ("    const bool cogroup = !cplan->use_ref; (void) cogroup;\n" if use_ref else
             "    const bool cogroup = true; (void) cogroup;\n")
    text = _after_line(text, "    bool disposable_threadpool = false;", guard + pragma_in)
    return text


@needs_anchor
def test_r1_cogroup_scheduler_patch_lands_on_cpu_graph_sched():
    scope, texts = _scope({CPU_C: _cogroup_patch()}, "ggml_graph_compute", CPU_C)
    route, refusal = gates.admit_cpu_route(CPU_C, "ggml_graph_compute", *texts[CPU_C])
    assert refusal is None and route.route == "cpu_graph_sched"
    assert scope == route.ops and "TOPK_MOE" in scope
    assert route.model_identity and route.identity_targets == "peers"
    assert route.identity_repeats == 3
    assert cpu_route_witness.WITNESSES["cpu_graph_sched"].norm_mulmat


@needs_anchor
def test_r1_sync_targets_still_prefer_cpu_graph_sync_and_fall_through_to_sched():
    # a walk + struct-append + helper patch named by the walk symbol: sync refuses (struct
    # and ggml_graph_compute are not its bodies), sched admits.
    text = _cogroup_patch()
    patch = _patch(_head(CPU_C), text)
    route, refusal = gates.admit_cpu_route(CPU_C, "ggml_graph_compute_thread", text,
                                           _head(CPU_C), patch)
    assert refusal is None and route.route == "cpu_graph_sched"
    assert [r.route for r in gates.cpu_source_routes(CPU_C, "ggml_barrier")] == \
        ["cpu_graph_sync", "cpu_graph_sched"]


@needs_anchor
def test_r1_requires_use_ref_fallback():
    scope, _ = _scope({CPU_C: _cogroup_patch(use_ref=False)}, "ggml_graph_compute", CPU_C)
    assert not scope.passed and "use_ref" in scope.reason


@needs_anchor
def test_r1_struct_fields_are_append_only():
    scope, _ = _scope({CPU_C: _cogroup_patch(field_at_tail=False)}, "ggml_graph_compute", CPU_C)
    assert not scope.passed and "not an append at the tail" in scope.reason
    removed = _head(CPU_C).replace("    int32_t      prio;        // Scheduling priority\n", "", 1)
    scope, _ = _scope({CPU_C: removed}, "ggml_threadpool", CPU_C)
    assert not scope.passed and "pure-insertion append" in scope.reason


@needs_anchor
def test_r1_pragma_omp_only_inside_ggml_barrier():
    bad = _cogroup_patch(pragma_in="    #pragma omp barrier\n")
    scope, _ = _scope({CPU_C: bad}, "ggml_graph_compute", CPU_C)
    assert not scope.passed and ("forbidden pattern" in scope.reason
                                 or "preprocessor directive" in scope.reason)
    # Round-5 resolution C (2026-10-06 ppl_contract review): no admitted body of ANY
    # route may add a preprocessor directive, so the former `#pragma omp flush` inside
    # ggml_barrier is now refused too (over-refusal is the accepted side).
    flush = _after_line(_cogroup_patch(), "void ggml_barrier(struct ggml_threadpool * tp) {",
                        "    #pragma omp flush\n")
    scope, _ = _scope({CPU_C: flush}, "ggml_graph_compute", CPU_C)
    assert isinstance(scope, gates.Verdict) and not scope.passed, scope
    assert "preprocessor directive" in scope.reason


@needs_anchor
def test_r1_other_functions_stay_refused():
    text = _cogroup_patch()
    mmid = text.index("static void ggml_compute_forward_mul_mat_id(")
    text = text[:mmid] + text[mmid:].replace("{\n", "{\n    int x = 0; (void) x;\n", 1)
    scope, _ = _scope({CPU_C: text}, "ggml_graph_compute", CPU_C)
    assert not scope.passed and "outside every admitted body" in scope.reason


# ------------------------------------------------------------------ R2 cpu_model_fused_op

Q4E = "src/models/qwen4exp.cpp"
GGML_H = "ggml/include/ggml.h"
GGML_C = "ggml/src/ggml.c"
OPS_H = "ggml/src/ggml-cpu/ops.h"
OPS_CPP = "ggml/src/ggml-cpu/ops.cpp"
MODEL_H = "src/llama-model.h"


def _fused_op_edits(*, case_label="GGML_OP_Q4E_HC_MIX", enum_mid=False, ctor="ggml_q4e_hc_mix"):
    q4e = _head(Q4E)
    q4e = _before(q4e, "ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(",
                  "static bool q4e_use_fused_hc(void) {\n    return true;\n}\n\n")
    q4e = _after_line(q4e, "    const int64_t nt     = x->ne[2];",
                      "    if (q4e_use_fused_hc()) { (void) 0; }\n")
    h = _head(GGML_H)
    if enum_mid:
        h = _after_line(h, "        GGML_OP_DSV4_HC_POST,", "        GGML_OP_Q4E_HC_MIX,\n")
    else:
        h = _after_line(h, "        GGML_OP_GLU,", "        GGML_OP_Q4E_HC_MIX,\n")
    h = _before(h, "    GGML_API struct ggml_tensor * ggml_dsv4_hc_pre(",
                f"    GGML_API struct ggml_tensor * {ctor}(\n"
                "            struct ggml_context * ctx,\n"
                "            struct ggml_tensor  * x);\n\n")
    c = _head(GGML_C)
    c = c.replace('    "GLU",\n};', '    "GLU",\n    "Q4E_HC_MIX",\n};', 1)
    c = c.replace('    "glu(x)",\n};', '    "glu(x)",\n    "q4e_hc_mix(x)",\n};', 1)
    c = c.replace('static_assert(GGML_OP_COUNT == 104, "GGML_OP_COUNT != 104");',
                  'static_assert(GGML_OP_COUNT == 105, "GGML_OP_COUNT != 105");')
    c = _before(c, "// ggml_dsv4_hc_post",
                f"struct ggml_tensor * {ctor}(\n        struct ggml_context * ctx,\n"
                "        struct ggml_tensor  * x) {\n"
                "    struct ggml_tensor * result = ggml_dup_tensor(ctx, x);\n"
                "    result->op     = GGML_OP_Q4E_HC_MIX;\n    result->src[0] = x;\n"
                "    return result;\n}\n\n")
    oh = _after_line(_head(OPS_H),
                     "void ggml_compute_forward_dsv4_hc_post(const struct ggml_compute_params * params, "
                     "struct ggml_tensor * dst);",
                     "void ggml_compute_forward_q4e_hc_mix(const struct ggml_compute_params * params, "
                     "struct ggml_tensor * dst);\n")
    oc = _head(OPS_CPP) + ("\nstatic void q4e_hc_mix_f32(const ggml_compute_params * params, "
                           "ggml_tensor * dst) {\n    (void) params; (void) dst;\n}\n\n"
                           "void ggml_compute_forward_q4e_hc_mix(const ggml_compute_params * params, "
                           "ggml_tensor * dst) {\n    q4e_hc_mix_f32(params, dst);\n}\n")
    cpu = _head(CPU_C)
    cpu = _before(cpu, "        case GGML_OP_DSV4_HC_COMB:\n            {\n"
                       "                ggml_compute_forward_dsv4_hc_comb(params, tensor);",
                  f"        case {case_label}:\n            {{\n"
                  "                ggml_compute_forward_q4e_hc_mix(params, tensor);\n"
                  "            } break;\n")
    cpu = _before(cpu, "        case GGML_OP_COUNT_EQUAL:\n        case GGML_OP_SOLVE_TRI:",
                  f"        case {case_label}:\n            {{\n"
                  "                n_tasks = n_threads;\n            } break;\n")
    return {Q4E: q4e, GGML_H: h, GGML_C: c, OPS_H: oh, OPS_CPP: oc, CPU_C: cpu}


@needs_anchor
def test_r2_hc_fused_op_across_seven_files_is_admitted():
    scope, texts = _scope(_fused_op_edits(), "llama_model_qwen4exp::graph::build_hc_mix", Q4E)
    assert scope == (), scope
    route, refusal = gates.admit_cpu_multi_route(tuple(sorted(texts)),
                                                 "llama_model_qwen4exp::graph::build_hc_mix",
                                                 texts)
    assert refusal is None and route.route == "cpu_model_fused_op"
    assert route.identity_arch == ("qwen4exp",) and route.identity_targets == "shared_peers"
    assert cpu_route_witness.WITNESSES["cpu_model_fused_op"].reference == "model_identity"


@needs_anchor
def test_r2_dispatch_case_for_an_existing_op_is_refused():
    scope, _ = _scope(_fused_op_edits(case_label="GGML_OP_RMS_NORM"), "build_hc_mix", Q4E)
    assert not scope.passed and "not enumerators this patch appends" in scope.reason


@needs_anchor
def test_r2_case_inside_another_case_is_refused():
    edits = _fused_op_edits()
    cpu = _head(CPU_C)
    edits[CPU_C] = _after_line(cpu, "        case GGML_OP_DSV4_HC_PRE:",
                               "        case GGML_OP_Q4E_HC_MIX:\n            {\n"
                               "                n_tasks = n_threads;\n            } break;\n")
    scope, _ = _scope(edits, "build_hc_mix", Q4E)
    assert not scope.passed and "case/default label" in scope.reason


@needs_anchor
def test_r2_enumerator_must_be_appended_before_count():
    scope, _ = _scope(_fused_op_edits(enum_mid=True), "build_hc_mix", Q4E)
    assert not scope.passed and "not an append at the tail" in scope.reason


@needs_anchor
def test_r2_existing_names_shared_builders_and_ggml_only_patches_are_refused():
    scope, _ = _scope(_fused_op_edits(ctor="ggml_rms_norm"), "build_hc_mix", Q4E)
    assert not scope.passed and "already occurs in HEAD" in scope.reason
    edits = _fused_op_edits()
    graph = "src/llama-graph.cpp"
    if (ANCHOR / graph).is_file():
        edits[graph] = _head(graph) + "\n// shared builder edit\n"
        scope, _ = _scope(edits, "build_hc_mix", Q4E)
        assert not scope.passed and "build_moe_ffn" in scope.reason
    edits = _fused_op_edits()
    del edits[Q4E]
    scope, _ = _scope(edits, "build_hc_mix", GGML_H)
    assert not scope.passed and "must change src/models/qwen4exp.cpp" in scope.reason


@needs_anchor
def test_r2_seed6_load_time_packing_with_a_layer_field():
    q4e = _after_line(_head(Q4E), "void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {",
                      "    // seed 6: pack wqkv | wqkv_gate | beta | alpha per GDN layer\n")
    model_h = _head(MODEL_H).replace("    struct llama_layer_nextn nextn;\n};",
                                     "    struct llama_layer_nextn nextn;\n\n"
                                     "    struct ggml_tensor * ssm_in_packed = nullptr;\n};", 1)
    scope, _ = _scope({Q4E: q4e, MODEL_H: model_h}, "load_arch_tensors", Q4E)
    assert scope == (), scope
    bad = _head(MODEL_H).replace("struct llama_layer {\n",
                                 "struct llama_layer {\n    int first_field_moved = 0;\n", 1)
    scope, _ = _scope({Q4E: q4e, MODEL_H: bad}, "load_arch_tensors", Q4E)
    assert not scope.passed and "not an append at the tail" in scope.reason


# ------------------------------------------------------------------ R3 cpu_graph_optimize

@needs_anchor
def test_r3_graph_optimize_hook():
    cpp = "ggml/src/ggml-cpu/ggml-cpu.cpp"
    head = _head(cpp)
    hook = ("static void ggml_backend_cpu_graph_optimize(ggml_backend_t backend, "
            "struct ggml_cgraph * cgraph) {\n"
            "    ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) backend->context;\n"
            "    if (ctx->use_ref) { return; }\n    (void) cgraph;\n}\n\n")
    text = _before(head, "static const struct ggml_backend_i ggml_backend_cpu_i = {", hook)
    text = text.replace("    /* .graph_optimize          = */ NULL,",
                        "    /* .graph_optimize          = */ ggml_backend_cpu_graph_optimize,", 1)
    scope, texts = _scope({cpp: text, CPU_C: _cogroup_patch()}, "graph_optimize", cpp)
    assert not isinstance(scope, gates.Verdict), scope
    route, refusal = gates.admit_cpu_multi_route(tuple(sorted(texts)), "graph_optimize", texts)
    assert refusal is None and route.route == "cpu_graph_optimize"
    assert route.identity_repeats == 3 and route.identity_targets == "peers"
    scope, _ = _scope({cpp: text}, "graph_optimize", cpp)        # hook alone, use_ref named
    assert not isinstance(scope, gates.Verdict), scope
    scope, _ = _scope({CPU_C: _cogroup_patch()}, "graph_optimize", CPU_C)
    assert not scope.passed and "must change ggml/src/ggml-cpu/ggml-cpu.cpp" in scope.reason
    other = text.replace("    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,",
                         "    /* .graph_compute           = */ NULL,", 1)
    scope, _ = _scope({cpp: other}, "graph_optimize", cpp)
    assert not scope.passed


# ------------------------------------------------------------------ identity gate

def _recipe(build, port=8080):
    return SimpleNamespace(
        backend="cpu", build_dir=build,
        command_argv=(f"{build}/bin/llama-server", "-m", "/m.gguf", "--port", str(port)),
        topology_prefix=("numactl", "--interleave=all"), port=port,
        launch_env=(("LD_LIBRARY_PATH", f"{build}/bin"),))


def _serve(script, seen):
    def serve_fn(recipe, selected):
        seen.append((recipe.build_dir, [json.loads(body) for _pid, body in selected]))
        out = script.pop(0)
        return [(pid, digest, "") for (pid, _b), digest in zip(selected, out)]
    return serve_fn


REQS = tuple((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode()) for i in range(3))


def test_repetition_identity_passes_and_turns_the_prompt_cache_off():
    seen = []
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=REQS, repeats=3,
                                  serve_fn=_serve([["h1", "h2"], ["h1", "h2"] * 3], seen))
    assert result.status == "pass" and "3x" in result.reason
    assert [(b, len(r)) for b, r in seen] == [("/a", 2), ("/c", 6)]
    assert all(body["cache_prompt"] is False for _b, rows in seen for body in rows)


def test_repetition_identity_flags_a_race_only_against_a_reproducible_anchor():
    racy = ["h1", "h2", "h1", "XX", "h1", "h2"]
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=REQS, repeats=3,
                                  serve_fn=_serve([["h1", "h2"], racy, ["h1", "h2"] * 3], []))
    assert result.status == "wrong" and "race" in result.reason
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=REQS, repeats=3,
                                  serve_fn=_serve([["h1", "h2"], racy,
                                                   ["h1", "h2", "zz", "h2", "h1", "h2"]], []))
    assert result.status == "unavailable"


def test_identity_targets_need_the_model_arch_and_every_target_passes(monkeypatch):
    own = ("q38fn", _recipe("/a"), _recipe("/c"), REQS)
    peer = ("ds41", _recipe("/a", 8090), _recipe("/c", 8090), REQS)
    arch = {8080: "qwen4exp", 8090: "deepseek41"}
    read = lambda recipe: arch[recipe.port]  # noqa: E731
    verdict = gates.check_model_identity_targets([peer], required_arch=("qwen4exp",),
                                                 architecture=read)
    assert (verdict.gate, verdict.passed) == ("oracle_unavailable", False)
    outcomes = {8080: model_identity.IdentityResult("pass", "ok"),
                8090: model_identity.IdentityResult("wrong", "differs")}
    monkeypatch.setattr(model_identity, "check",
                        lambda **kw: outcomes[kw["anchor_recipe"].port])
    verdict = gates.check_model_identity_targets([own, peer], required_arch=("qwen4exp",),
                                                 architecture=read, repeats=3)
    assert (verdict.gate, verdict.passed) == ("reference_comparison", False)
    assert "ds41" in verdict.reason
    outcomes[8090] = model_identity.IdentityResult("pass", "ok")
    verdict = gates.check_model_identity_targets([own, peer], architecture=read, repeats=3)
    assert verdict.passed and "2 target(s)" in verdict.reason


def test_every_route_has_a_witness_and_run_wires_identity_targets():
    names = {r.route for r in (*gates.CPU_SOURCE_ROUTES, *gates.CPU_MULTI_FILE_ROUTES)}
    assert names <= set(cpu_route_witness.WITNESSES)
    assert {"cpu_graph_sched", "cpu_model_fused_op", "cpu_graph_optimize"} <= names
    source = Path(__file__).with_name("run.py").read_text(encoding="utf-8")
    assert "gates.check_model_identity_targets(" in source
    assert "lane_peer_serving[peer.entry.name]" in source
    assert "gates.admit_cpu_multi_route(" in source
    program = Path(__file__).with_name("program.md").read_text(encoding="utf-8")
    for name in ("cpu_graph_sched", "cpu_model_fused_op", "cpu_graph_optimize"):
        assert f"**`{name}`**" in program
