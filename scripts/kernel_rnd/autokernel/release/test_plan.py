#!/usr/bin/env python3
"""test_plan.py — the regression barrier for the release-plan compiler (§10.1, §3.2).

WHY THIS FILE EXISTS
--------------------
Every defect this compiler can have makes a freeze CHEAPER, which is exactly why it
would not be noticed. A backend that quietly falls out of scope, a role that quietly
falls out of the matrix, a cell that quietly merges with another, a "backend unchanged"
that quietly means "we did not look" — all four produce a smaller, faster, greener
release and none of them announces itself. So the assertions below are mostly about
things NOT shrinking:

  * a live role that cannot be planned stays on the record as unplannable, and a role
    served by an unrecognised binary is never silently attributed to CPU (the seed's
    `return "cpu"` fallthrough, re-derived here and asserted to differ);
  * `llama_gpu` cannot be dropped by leaving it out of the target — the only sanctioned
    narrowing is §3.2 with a receipt;
  * cells drop ONLY on stage 1 + stage 2 agreement, a reconciled surface, and named
    incumbent artifacts with hashes; each of those three is removed in turn and the
    drop must stop;
  * a stage disagreement is a FAIL filed against build identity, never a preference for
    the cheaper stage; and
  * a backend the trace observed executing cannot simultaneously be unchanged.

NO inference, NO benchmark, NO build, NO process, NO production path is touched. The
only writes are into a `tempfile.TemporaryDirectory` this suite creates and removes.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/release/test_plan.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/release/test_plan.py
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

# RELATIVE, not `sys.path.insert` + `from autokernel import …`. The flat idiom binds
# this suite to a SECOND copy of `schemas` and `surface`: under
# `unittest discover -t .` the package is already imported as
# `scripts.kernel_rnd.autokernel`, so `autokernel.evaluator.surface` becomes a
# different module object with different classes. Every `isinstance` guard across
# that boundary then fails silently — `compile_release_plan` refuses a genuine
# `surface.BackendUnchangedResult` because it is the other copy's, and
# `test_release_integration.py` cannot share a fixture with this suite at all.
# README, "Import convention": *"Both shortcuts load a second copy of schemas.py."*
from .. import schemas as S
from ..evaluator import surface as SU
from . import plan as P

NOW = "2026-08-03T12:00:00+00:00"
BASE_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
CAND_COMMIT = "aa11bb22cc33dd44ee55ff6677889900aabbccdd"

TREE = "/mnt/raid0/llm/llama.cpp"
CPU_BIN_DIR = f"{TREE}/build/bin"
GPU_BIN_DIR = f"{TREE}/build-hip/bin"
CPU_BIN = f"{CPU_BIN_DIR}/llama-server"
GPU_BIN = f"{GPU_BIN_DIR}/llama-server"
STABLE_CPU = "/mnt/raid0/llm/kernels/production/cpu"
STABLE_GPU = "/mnt/raid0/llm/kernels/production/gpu"
EXPERIMENTAL_BUILD = "/mnt/raid0/llm/llama.cpp-experimental/build"

MODEL_A = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
MODEL_B = "/mnt/raid0/llm/models/Qwen3.5-27B-Q4_K_M.gguf"
DIGEST_A = "a" * 63 + "1"
DIGEST_B = "b" * 63 + "2"
ARCHIVE = "/mnt/raid0/llm/kernels/archive/v8/gpu"
BIN_DIGEST = "c" * 63 + "3"
LIB_DIGEST = "d" * 63 + "4"


# =============================================================================
# Fixtures
# =============================================================================

def role_prior(*, binary, model_path, status="live_stack", quant="Q8_0",
               arch="qwen35moe", ctx=262144, kv=("q8_0", "q8_0"), ubatch=8192,
               slots=16, numa="2x48t_half_instances", entries=None, mem_gb=37.0,
               ld=(), spec=None, model_id=None):
    """One role record shaped like `orchestration/derived/stack_priors.yaml`."""
    spec = spec if spec is not None else {"enabled": False, "type": None,
                                          "draft_max": None}
    runtime = {
        "binary_path": binary,
        "ld_library_path": list(ld),
        "cache": {"context_tokens": ctx, "slots": slots, "ubatch": ubatch,
                  "kv_type_k": kv[0], "kv_type_v": kv[1]},
        "flags": {"flash_attn": True, "spec": spec},
    }
    record = {
        "deployment_status": status,
        "model_id": model_id,
        "serving": {
            "numa_policy": numa,
            "launch": {
                "entries": entries if entries is not None else [],
                "requirements": {"model_path": model_path},
                "runtime": runtime,
            },
        },
        "policy": {"model_mem_gb": mem_gb},
        "model": {"quant": quant, "arch": arch, "family": "qwen", "params_b": 35.0,
                  "active_b": 3.0, "mem_gb": mem_gb},
    }
    if binary is None:
        record["serving"]["launch"]["runtime"].pop("binary_path")
    return record


def entry(port, *, numa_instance=0, cpu_shape_class="full", slots=None):
    out = {"port": port, "numa_instance": numa_instance,
           "cpu_shape_class": cpu_shape_class}
    if slots is not None:
        out["slots"] = slots
    return out


def clean_priors():
    """A lineup with two CPU servers and one GPU server; every field present."""
    return {"roles": {
        # One CPU server on 8072 with three roles — the dedup case.
        "worker_general": role_prior(binary=CPU_BIN, model_path=MODEL_A,
                                     entries=[entry(8072)], ld=[CPU_BIN_DIR]),
        "worker_math": role_prior(binary=CPU_BIN, model_path=MODEL_A,
                                  entries=[entry(8072)], ld=[CPU_BIN_DIR]),
        "toolrunner": role_prior(binary=CPU_BIN, model_path=MODEL_A,
                                 entries=[entry(8072)], ld=[CPU_BIN_DIR]),
        # A second CPU server, different context -> its own cell.
        "frontdoor": role_prior(binary=CPU_BIN, model_path=MODEL_A, ctx=65536,
                                entries=[entry(8070)], ld=[CPU_BIN_DIR]),
        # One GPU server with two roles.
        "architect_general": role_prior(
            binary=GPU_BIN, model_path=MODEL_B, quant="Q4_K_M", arch="qwen35dense",
            ctx=65536, numa="single_96t", mem_gb=17.0,
            entries=[entry(8083, numa_instance=None, cpu_shape_class="gpu_host_lane")],
            ld=[GPU_BIN_DIR]),
        "coder_escalation": role_prior(
            binary=GPU_BIN, model_path=MODEL_B, quant="Q4_K_M", arch="qwen35dense",
            ctx=65536, numa="single_96t", mem_gb=17.0,
            entries=[entry(8083, numa_instance=None, cpu_shape_class="gpu_host_lane")],
            ld=[GPU_BIN_DIR]),
        # A real row that is NOT a protected cell.
        "qwen35_122b_q4km": role_prior(binary=None, model_path=None,
                                       status="benchmark_or_candidate"),
    }}


def phase_protocol(phase, protocol_id):
    return P.PhaseProtocol(
        phase=phase, protocol_id=protocol_id, metric="tokens_per_second",
        direction="higher_better",
        thresholds={"non_inferiority_ratio_pass": "supplied by the protocol",
                    "fail_below": "supplied by the protocol"},
        threshold_source=f"measurement/protocols/... ({protocol_id})")


def cpu_binding(**overrides):
    kwargs = dict(
        backend="llama_cpu",
        stable_production_path=STABLE_CPU,
        production_tree_path=TREE,
        binary_roots=(STABLE_CPU, CPU_BIN_DIR),
        phases=("decode", "prefill"),
        protocols={"decode": phase_protocol("decode", "P-BENCH-1"),
                   "prefill": phase_protocol("prefill", "P-BENCH-PREFILL-1")},
        linkage=P.LinkageRequirement(source_tree="llama.cpp", ggml_generation="0.19.0",
                                     required_ld_library_path=(CPU_BIN_DIR,)),
        ceiling=P.ComplexityCeiling(max_diff_lines=2000, max_files_touched=50),
        co_residency_required=True,
        host_capacity_budget_gb=900.0,
    )
    kwargs.update(overrides)
    return P.BackendBinding(**kwargs)


def gpu_binding(**overrides):
    kwargs = dict(
        backend="llama_gpu",
        stable_production_path=STABLE_GPU,
        production_tree_path=TREE,
        binary_roots=(STABLE_GPU, GPU_BIN_DIR),
        phases=("decode", "prefill"),
        protocols={"decode": phase_protocol("decode", "P-GPU-1"),
                   "prefill": phase_protocol("prefill", "P-GPU-1")},
        linkage=P.LinkageRequirement(source_tree="llama.cpp", ggml_generation="0.19.0",
                                     required_ld_library_path=(GPU_BIN_DIR,)),
        ceiling=P.ComplexityCeiling(max_diff_lines=2000, max_files_touched=50),
        host_capacity_budget_gb=64.0,
    )
    kwargs.update(overrides)
    return P.BackendBinding(**kwargs)


def bindings():
    return {"llama_cpu": cpu_binding(), "llama_gpu": gpu_binding()}


def target(**overrides):
    kwargs = dict(
        source_tree="llama.cpp",
        production_base_commit=BASE_COMMIT,
        candidate_commit=CAND_COMMIT,
        candidate_branch="llama.cpp-experimental/ak-avx512-repack",
        candidate_build_root=EXPERIMENTAL_BUILD,
        candidate_id="akc-0001",
        backends=("llama_cpu", "llama_gpu"),
        change_classes=("arithmetic",),
        diff_lines=120,
        files_touched=3,
        touches_shared_core=False,
    )
    kwargs.update(overrides)
    return P.ReleaseTarget(**kwargs)


def derived_surface(*, candidate_id="akc-0001", full_tree=False):
    """A derived manifest that over-approximates to BOTH llama backends."""
    return SU.AffectedSurface(
        candidate_id=candidate_id,
        backends=("llama_cpu", "llama_gpu"),
        link_targets=("libggml-cpu.so", "libggml-hip.so"),
        objects=("ggml-cpu.o",),
        touched_files=("ggml/src/ggml-cpu/ggml-cpu-quants.c",),
        symbols=("ggml_vec_dot_q8_0_q8_0", "ggml_hip_mul_mat"),
        op_registrations=(SU.OpRegistration("MUL_MAT", "llama_cpu", "pred_cpu"),
                          SU.OpRegistration("MUL_MAT", "llama_gpu", "pred_hip")),
        dispatch_predicates=("pred_cpu", "pred_hip"),
        over_approximations=(),
        axes_derived=SU.SURFACE_AXES,
        coverage=S.Check(S.PASS),
        full_tree=full_tree,
        inputs={"fixture": "test_plan"},
    )


def traced_surface(*, candidate_id="akc-0001", backends=("llama_cpu",)):
    events = []
    for backend in backends:
        events.append(SU.DispatchEvent(
            op_name="MUL_MAT", backend=backend,
            kernel_symbol=("ggml_vec_dot_q8_0_q8_0" if backend == "llama_cpu"
                           else "ggml_hip_mul_mat"),
            link_target=("libggml-cpu.so" if backend == "llama_cpu"
                         else "libggml-hip.so"),
            dispatch_predicate=("pred_cpu" if backend == "llama_cpu" else "pred_hip")))
    return SU.TracedSurface(
        candidate_id=candidate_id, trace_ref="trace://fixture", events=tuple(events),
        truncated=False, completeness=S.Check(S.PASS), no_fallback=S.Check(S.PASS))


def reconciliation(**kwargs):
    return SU.reconcile_surface(derived_surface(), traced_surface(**kwargs))


def unreconciled():
    """A derived manifest with no trace at all — COULD_NOT_CHECK on every axis."""
    return SU.reconcile_surface(derived_surface(), None)


def stage1(backend, *, changed):
    check = (S.Check(S.FAIL, ("1 diff path lies inside the closure",)) if changed
             else S.Check(S.PASS))
    return SU.SourceClosureIdentity(
        backend=backend, closure_size=120,
        changed_in_closure=(("ggml/src/ggml-cpu/ggml-cpu-quants.c",) if changed else ()),
        unmapped_diff_paths=(), toolchain_differences=(), check=check,
        base_commit=BASE_COMMIT, candidate_commit=CAND_COMMIT)


def stage2(backend, *, identical):
    differing = () if identical else ((".text", "aa", "bb"),)
    check = S.Check(S.PASS) if identical else S.Check(S.FAIL, ("normalized .text differs",))
    return SU.NormalizedBinaryIdentity(
        backend=backend, candidate_ref=f"{backend}:candidate",
        base_ref=f"{backend}:base-rebuild",
        differing=tuple(f"{d[0]}" for d in differing), rebuild_verified=True,
        check=check, base_commit=BASE_COMMIT)


def in_scope():
    return SU.EvidenceTransferScope(
        same_models=True, same_recipes=True, candidate_topology_hash="topo-1",
        incumbent_topology_hash="topo-1", era_boundary_crossed=False)


def gpu_unchanged(**kwargs):
    kwargs.setdefault("stage1", stage1("llama_gpu", changed=False))
    kwargs.setdefault("stage2", stage2("llama_gpu", identical=True))
    kwargs.setdefault("transfer_scope", in_scope())
    return SU.backend_unchanged(**kwargs)


def cpu_changed():
    return SU.backend_unchanged(stage1=stage1("llama_cpu", changed=True),
                                stage2=None, transfer_scope=in_scope())


def cpu_changed_gpu():
    """The GPU binary changed too — a shared-ggml-core champion."""
    return SU.backend_unchanged(stage1=stage1("llama_gpu", changed=True), stage2=None,
                                transfer_scope=in_scope())


def incumbent(backend="llama_gpu"):
    return P.IncumbentEvidence(
        backend=backend, era_id="E8",
        artifacts=((f"{ARCHIVE}/llama-server", BIN_DIGEST),
                   (f"{ARCHIVE}/libggml-hip.so", LIB_DIGEST)),
        protocol_ids=("P-GPU-1",),
        archive_path=ARCHIVE)


def receipts():
    return {
        "llama_cpu": P.StablePathReceipt(
            backend="llama_cpu", stable_path=STABLE_CPU, resolved_target=CPU_BIN_DIR,
            observed_at=NOW),
        "llama_gpu": P.StablePathReceipt(
            backend="llama_gpu", stable_path=STABLE_GPU, resolved_target=GPU_BIN_DIR,
            observed_at=NOW),
    }


def coverage():
    return P.OpShapeCoverage(
        covered={"llama_cpu": {"MUL_MAT": ("4096x4096x1", "8192x512x1")},
                 "llama_gpu": {"MUL_MAT": ("4096x4096x1",)}},
        source_ref="corpus://fixture")


def compile_ok(**overrides):
    """The happy path: a CPU-local champion, GPU unchanged and dropped with a receipt."""
    kwargs = dict(
        target=target(),
        bindings=bindings(),
        priors=clean_priors(),
        reconciliation=reconciliation(),
        compiled_at=NOW,
        unchanged_by_backend={"llama_cpu": cpu_changed(), "llama_gpu": gpu_unchanged()},
        incumbent_evidence={"llama_gpu": incumbent()},
        stable_path_receipts=receipts(),
        op_coverage=coverage(),
        model_digests={MODEL_A: DIGEST_A, MODEL_B: DIGEST_B},
    )
    kwargs.update(overrides)
    return P.compile_release_plan(**kwargs)


def codes(plan_obj):
    return sorted({f.code for f in plan_obj.findings}
                  | {f.code for b in plan_obj.backends for f in b.findings})


# =============================================================================
# Path handling and backend classification — the seed's `return "cpu"` defect
# =============================================================================

class TestPathClassification(unittest.TestCase):

    def test_component_boundary_not_substring(self):
        # THE defect: `/…/build` is a string prefix of `/…/build-hip/bin/llama-server`.
        self.assertFalse(P.path_is_under(f"{TREE}/build", GPU_BIN))
        self.assertTrue(P.path_is_under(f"{TREE}/build", CPU_BIN))
        self.assertTrue(P.path_is_under(GPU_BIN_DIR, GPU_BIN))
        self.assertTrue(P.path_is_under(GPU_BIN, GPU_BIN))

    def test_normalize_refuses_relative_and_empty(self):
        for bad in ("", "   ", "build/bin/llama-server", None, 7):
            with self.assertRaises(P.PlanInputError):
                P.normalize_path(bad, label="x")
        self.assertEqual(P.normalize_path("//a//b//", label="x"), "/a/b")

    def test_seed_substring_classifier_and_ours_disagree(self):
        # The seed's classifier, re-derived here so this test does not import from a
        # repo it may not write to. Its final `return "cpu"` is the fail-open branch.
        def seed_backend_of(path: str) -> str:
            if "build-hip" in path:
                return "gpu"
            if "whisper" in path:
                return "stt"
            if "qwentts" in path:
                return "tts"
            return "cpu"

        mystery = "/mnt/raid0/llm/other-engine/bin/server"
        self.assertEqual(seed_backend_of(mystery), "cpu")
        backend, claimants = P._classify(mystery, bindings())
        self.assertIsNone(backend)
        self.assertEqual(claimants, ())

    def test_nested_roots_resolve_to_the_longest(self):
        outer = cpu_binding(binary_roots=(f"{TREE}/build",))
        inner = gpu_binding(backend="llama_gpu", binary_roots=(f"{TREE}/build/bin",),
                            stable_production_path=STABLE_GPU)
        backend, claimants = P._classify(CPU_BIN, {"llama_cpu": outer,
                                                   "llama_gpu": inner})
        self.assertEqual(backend, "llama_gpu")
        self.assertEqual(claimants, ("llama_cpu", "llama_gpu"))

    def test_ambiguous_roots_are_a_finding_not_a_guess(self):
        # Two bindings whose roots are the SAME depth both claim the binary.
        a = cpu_binding(binary_roots=(CPU_BIN_DIR,))
        b = gpu_binding(binary_roots=(CPU_BIN_DIR,))
        backend, claimants = P._classify(CPU_BIN, {"llama_cpu": a, "llama_gpu": b})
        self.assertIsNone(backend)
        self.assertEqual(claimants, ("llama_cpu", "llama_gpu"))


# =============================================================================
# Role extraction — nothing leaves the matrix without a receipt
# =============================================================================

class TestRoleExtraction(unittest.TestCase):

    def test_live_roles_flatten_one_fact_per_entry(self):
        facts, unplannable, out_of_scope = P.extract_role_facts(clean_priors())
        self.assertEqual(unplannable, ())
        self.assertEqual([r.role for r in out_of_scope], ["qwen35_122b_q4km"])
        self.assertEqual(out_of_scope[0].deployment_status, "benchmark_or_candidate")
        self.assertEqual(len(facts), 6)

    def test_missing_binary_or_model_is_recorded_not_skipped(self):
        priors = clean_priors()
        priors["roles"]["no_binary"] = role_prior(binary=None, model_path=MODEL_A)
        priors["roles"]["no_model"] = role_prior(binary=CPU_BIN, model_path=None)
        _, unplannable, _ = P.extract_role_facts(priors)
        by_role = {u.role: u for u in unplannable}
        self.assertEqual(set(by_role), {"no_binary", "no_model"})
        for record in by_role.values():
            self.assertEqual(record.code, P.F_ROLE_RECIPE_INCOMPLETE)
            self.assertTrue(record.reason)

    def test_two_entries_become_two_facts(self):
        priors = {"roles": {"dual": role_prior(
            binary=CPU_BIN, model_path=MODEL_A,
            entries=[entry(8070, numa_instance=0), entry(8071, numa_instance=1)])}}
        facts, _, _ = P.extract_role_facts(priors)
        self.assertEqual(len(facts), 2)
        self.assertEqual({f.numa_instance for f in facts}, {0, 1})

    def test_entryless_live_role_still_yields_a_fact(self):
        priors = {"roles": {"solo": role_prior(binary=CPU_BIN, model_path=MODEL_A,
                                               entries=[])}}
        facts, unplannable, _ = P.extract_role_facts(priors)
        self.assertEqual(unplannable, ())
        self.assertEqual(len(facts), 1)
        self.assertIsNone(facts[0].port)

    def test_bool_is_not_an_int(self):
        priors = {"roles": {"weird": role_prior(binary=CPU_BIN, model_path=MODEL_A,
                                                slots=True, entries=[entry(8070)])}}
        facts, _, _ = P.extract_role_facts(priors)
        self.assertIsNone(facts[0].slots)  # `slots: true` must not become 1

    def test_priors_without_roles_refuses(self):
        with self.assertRaises(P.PlanInputError):
            P.extract_role_facts({"stack_priors_version": 3})


# =============================================================================
# Refusals — wiring and authority defects raise, they are not findings
# =============================================================================

class TestRefusals(unittest.TestCase):

    def test_serving_runtime_may_not_travel_the_kernel_freeze_path(self):
        with self.assertRaises(P.KernelFreezePathRefused):
            P.BackendBinding(backend="serving_runtime",
                             stable_production_path=STABLE_CPU,
                             production_tree_path=TREE, binary_roots=(CPU_BIN_DIR,),
                             phases=("decode",))
        with self.assertRaises(P.KernelFreezePathRefused):
            target(backends=("llama_cpu", "llama_gpu", "serving_runtime"))

    def test_production_branch_is_refused(self):
        with self.assertRaises(P.ProductionWriteRefused):
            target(candidate_branch="production-consolidated-v9")
        with self.assertRaises(P.ProductionWriteRefused):
            target(candidate_branch="production-speech-v2")

    def test_building_inside_a_production_tree_is_refused(self):
        with self.assertRaises(P.ProductionWriteRefused):
            compile_ok(target=target(candidate_build_root=f"{TREE}/build"))

    def test_freeze_scope_is_the_whole_tree(self):
        # §1.5: leaving llama_gpu out of the target is the cheapest scope exploit.
        with self.assertRaises(P.PlanInputError) as ctx:
            target(backends=("llama_cpu",))
        self.assertIn("union of backends served by the tree", str(ctx.exception))

    def test_empty_diff_is_refused(self):
        with self.assertRaises(P.PlanInputError):
            target(candidate_commit=BASE_COMMIT)

    def test_surface_for_another_candidate_is_refused(self):
        other = SU.reconcile_surface(derived_surface(candidate_id="akc-9999"),
                                     traced_surface(candidate_id="akc-9999"))
        with self.assertRaises(P.PlanInputError):
            compile_ok(reconciliation=other)

    def test_unchanged_result_over_another_base_is_refused(self):
        bad = SU.backend_unchanged(
            stage1=SU.SourceClosureIdentity(
                backend="llama_gpu", closure_size=1, changed_in_closure=(),
                unmapped_diff_paths=(), toolchain_differences=(), check=S.Check(S.PASS),
                base_commit="f" * 40, candidate_commit=CAND_COMMIT),
            stage2=None, transfer_scope=in_scope())
        with self.assertRaises(P.PlanInputError) as ctx:
            compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                             "llama_gpu": bad})
        self.assertIn("stage-1 diff", str(ctx.exception))

    def test_unchanged_result_for_an_unserved_backend_is_refused(self):
        with self.assertRaises(P.PlanInputError):
            compile_ok(unchanged_by_backend={
                "llama_cpu": cpu_changed(),
                "llama_gpu": gpu_unchanged(),
                "whisper_stt": SU.backend_unchanged(
                    stage1=stage1("whisper_stt", changed=False), stage2=None)})

    def test_missing_binding_is_refused(self):
        with self.assertRaises(P.PlanInputError):
            compile_ok(bindings={"llama_cpu": cpu_binding()})

    def test_unreconciled_manifest_type_is_refused(self):
        with self.assertRaises(P.PlanInputError):
            compile_ok(reconciliation=derived_surface())

    def test_binding_phase_set_must_match_the_objective(self):
        with self.assertRaises(P.PlanInputError) as ctx:
            cpu_binding(phases=("decode",))
        self.assertIn("half the objective", str(ctx.exception))

    def test_thresholds_without_a_source_are_refused(self):
        with self.assertRaises(P.PlanInputError):
            P.PhaseProtocol(phase="decode", protocol_id="P-BENCH-1",
                            metric="tokens_per_second", direction="higher_better",
                            thresholds={"pass": 0.98})

    def test_placeholder_digests_are_refused(self):
        with self.assertRaises(P.PlanInputError):
            P.IncumbentEvidence(backend="llama_gpu", era_id="E8",
                                artifacts=(("bin", "0" * 64),))  # placeholder digest
        with self.assertRaises(P.PlanInputError):
            P.ModelIdentity(model_path=MODEL_A, sha256="f" * 64)

    def test_incumbent_evidence_must_name_something(self):
        with self.assertRaises(P.PlanInputError):
            P.IncumbentEvidence(backend="llama_gpu", era_id="E8", artifacts=())


# =============================================================================
# Deduplication — by measurement identity, never by role or basename
# =============================================================================

class TestDeduplication(unittest.TestCase):

    def test_three_roles_on_one_server_are_one_cell_that_protects_three(self):
        plan_obj = compile_ok()
        cpu_cells = plan_obj.cells_for("llama_cpu")
        big = [c for c in cpu_cells
               if c.phase == "decode" and c.context_tokens == 262144
               and c.co_residency == P.CO_RESIDENCY_SINGLE]
        self.assertEqual(len(big), 1)
        self.assertEqual(big[0].protected_roles,
                         ("toolrunner", "worker_general", "worker_math"))
        self.assertEqual([p for _, p in big[0].protected_entries], [8072, 8072, 8072])

    def test_a_different_context_does_not_merge(self):
        plan_obj = compile_ok()
        contexts = {c.context_tokens for c in plan_obj.cells_for("llama_cpu")
                    if c.phase == "decode"}
        self.assertEqual(contexts, {262144, 65536})

    def test_same_basename_different_file_does_not_merge(self):
        # The seed keyed dedup on `model_path.split("/")[-1]`.
        priors = {"roles": {
            "a": role_prior(binary=CPU_BIN, model_path="/models/x/model.gguf",
                            entries=[entry(8070)], ld=[CPU_BIN_DIR]),
            "b": role_prior(binary=CPU_BIN, model_path="/models/y/model.gguf",
                            entries=[entry(8071)], ld=[CPU_BIN_DIR]),
        }}
        plan_obj = compile_ok(priors=priors, model_digests={})
        decode_single = [c for c in plan_obj.cells_for("llama_cpu")
                         if c.phase == "decode" and c.co_residency == "single"]
        self.assertEqual(len(decode_single), 2)

    def test_two_paths_one_digest_do_merge(self):
        priors = {"roles": {
            "a": role_prior(binary=CPU_BIN, model_path="/models/real/m.gguf",
                            entries=[entry(8070)], ld=[CPU_BIN_DIR]),
            "b": role_prior(binary=CPU_BIN, model_path="/lmstudio/models/m.gguf",
                            entries=[entry(8071)], ld=[CPU_BIN_DIR]),
        }}
        plan_obj = compile_ok(priors=priors,
                              model_digests={"/models/real/m.gguf": DIGEST_A,
                                             "/lmstudio/models/m.gguf": DIGEST_A})
        decode_single = [c for c in plan_obj.cells_for("llama_cpu")
                         if c.phase == "decode" and c.co_residency == "single"]
        self.assertEqual(len(decode_single), 1)
        self.assertEqual(decode_single[0].protected_roles, ("a", "b"))

    def test_merge_takes_the_worst_check_not_the_first_roles(self):
        # Three roles share one server. One launcher forgets the tree's LD_LIBRARY_PATH.
        # The merged cell protects that role too, so it may not report linkage PASS.
        priors = clean_priors()
        priors["roles"]["worker_math"]["serving"]["launch"]["runtime"][
            "ld_library_path"] = []
        plan_obj = compile_ok(priors=priors)
        cell = [c for c in plan_obj.cells_for("llama_cpu")
                if c.phase == "decode" and c.context_tokens == 262144
                and c.co_residency == P.CO_RESIDENCY_SINGLE][0]
        self.assertIn("worker_math", cell.protected_roles)
        self.assertEqual(cell.checks["linkage"].outcome, S.COULD_NOT_CHECK)
        self.assertEqual(cell.check.outcome, S.COULD_NOT_CHECK)

    def test_merge_applies_the_stricter_footprint_and_records_the_disagreement(self):
        priors = clean_priors()
        priors["roles"]["worker_math"]["policy"]["model_mem_gb"] = 41.0
        priors["roles"]["worker_math"]["model"]["mem_gb"] = 41.0
        plan_obj = compile_ok(priors=priors)
        cell = [c for c in plan_obj.cells_for("llama_cpu")
                if c.phase == "decode" and c.context_tokens == 262144
                and c.co_residency == P.CO_RESIDENCY_SINGLE][0]
        self.assertEqual(cell.capacity_floor["resident_gb_max"], 37.0)
        self.assertIn("footprint_disagreement", cell.capacity_floor)
        self.assertEqual(cell.checks["capacity_floor"].outcome, S.COULD_NOT_CHECK)

    def test_path_only_identity_is_recorded_as_advisory(self):
        plan_obj = compile_ok(model_digests={})
        self.assertIn(P.F_MODEL_IDENTITY_BY_PATH_ONLY, codes(plan_obj))
        advisory = [f for f in plan_obj.findings
                    if f.code == P.F_MODEL_IDENTITY_BY_PATH_ONLY][0]
        self.assertFalse(advisory.gating)
        self.assertEqual(plan_obj.check.outcome, S.PASS)

    def test_every_live_role_lands_in_some_cell(self):
        plan_obj = compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                                    "llama_gpu": cpu_changed_gpu()})
        protected = set(plan_obj.protected_roles())
        expected = {r for r, v in clean_priors()["roles"].items()
                    if v["deployment_status"] == "live_stack"}
        self.assertEqual(protected, expected)


# =============================================================================
# Co-residency
# =============================================================================

class TestCoResidency(unittest.TestCase):

    def test_label_matches_the_schema_vocabulary(self):
        plan_obj = compile_ok()
        labels = {c.co_residency for c in plan_obj.cells_for("llama_cpu")}
        self.assertIn(P.CO_RESIDENCY_SINGLE, labels)
        self.assertTrue(any(x.startswith(P.CO_RESIDENCY_PREFIX) for x in labels))
        for label in labels:
            self.assertRegex(label, S._CO_RESIDENCY_RE)

    def test_both_a_single_and_a_co_resident_cell_exist(self):
        plan_obj = compile_ok()
        by_mode = {}
        for cell in plan_obj.cells_for("llama_cpu"):
            if cell.phase == "decode" and cell.context_tokens == 262144:
                by_mode[cell.co_residency] = cell
        self.assertEqual(len(by_mode), 2)
        self.assertEqual({c.protected_roles for c in by_mode.values()},
                         {("toolrunner", "worker_general", "worker_math")})

    def test_single_server_lineup_files_the_missing_coresident_cell(self):
        priors = {"roles": {"solo": role_prior(binary=CPU_BIN, model_path=MODEL_A,
                                               entries=[entry(8070)], ld=[CPU_BIN_DIR]),
                            "gpu": role_prior(binary=GPU_BIN, model_path=MODEL_B,
                                              entries=[entry(8083)], ld=[GPU_BIN_DIR])}}
        plan_obj = compile_ok(priors=priors)
        self.assertIn(P.F_CORESIDENT_CELL_MISSING, codes(plan_obj))
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)

    def test_group_id_is_deterministic(self):
        facts, _, _ = P.extract_role_facts(clean_priors())
        cpu = tuple(f for f in facts if f.binary_path == CPU_BIN)
        a = P.derive_co_residency_group("llama_cpu", cpu)
        b = P.derive_co_residency_group("llama_cpu", tuple(reversed(cpu)))
        self.assertEqual(a.group_id, b.group_id)
        self.assertEqual(len(a.members), 2)  # ports 8070 and 8072


# =============================================================================
# §3.2 — the backend-unchanged join
# =============================================================================

class TestBackendUnchanged(unittest.TestCase):

    def test_happy_path_drops_gpu_with_a_receipt(self):
        plan_obj = compile_ok()
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertTrue(gpu.cells_dropped)
        self.assertEqual(gpu.cells, ())
        self.assertIsNotNone(gpu.transfer_receipt)
        # One GPU server, two roles, two phases, single residency -> two cells, and the
        # receipt NAMES them rather than reporting a count nobody can audit.
        self.assertEqual(gpu.transfer_receipt.dropped_cell_count, 2)
        self.assertEqual(len(gpu.transfer_receipt.dropped_cell_ids), 2)
        self.assertEqual([a["sha256"] for a in
                          gpu.transfer_receipt.to_dict()["incumbent"]["artifacts"]],
                         [BIN_DIGEST, LIB_DIGEST])
        self.assertEqual(plan_obj.check.outcome, S.PASS)

    def test_cpu_keeps_its_matrix(self):
        plan_obj = compile_ok()
        cpu = [b for b in plan_obj.backends if b.backend == "llama_cpu"][0]
        self.assertFalse(cpu.cells_dropped)
        self.assertTrue(cpu.cells)

    def test_no_drop_without_named_incumbent_artifacts(self):
        plan_obj = compile_ok(incumbent_evidence={})
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertFalse(gpu.cells_dropped)
        self.assertTrue(gpu.cells)
        self.assertIn(P.F_TRANSFER_RECEIPT_INCOMPLETE, codes(plan_obj))

    def test_no_drop_without_stage_two(self):
        only_gate = SU.backend_unchanged(stage1=stage1("llama_gpu", changed=False),
                                         stage2=None, transfer_scope=in_scope())
        plan_obj = compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                                    "llama_gpu": only_gate})
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertFalse(gpu.cells_dropped)
        self.assertTrue(gpu.cells)

    def test_no_drop_when_the_surface_is_not_reconciled(self):
        plan_obj = compile_ok(reconciliation=unreconciled())
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertFalse(gpu.cells_dropped)
        self.assertFalse(plan_obj.narrowing_permitted)
        self.assertIn(P.F_SURFACE_UNRECONCILED, codes(plan_obj))

    def test_no_drop_when_the_surface_escaped(self):
        escaped_derived = SU.AffectedSurface(
            candidate_id="akc-0001", backends=("llama_cpu",),
            link_targets=("libggml-cpu.so",), objects=("ggml-cpu.o",),
            touched_files=("a.c",), symbols=("ggml_vec_dot_q8_0_q8_0",),
            op_registrations=(SU.OpRegistration("MUL_MAT", "llama_cpu", "pred_cpu"),),
            dispatch_predicates=("pred_cpu",), over_approximations=(),
            axes_derived=SU.SURFACE_AXES, coverage=S.Check(S.PASS), full_tree=False,
            inputs={})
        recon = SU.reconcile_surface(
            escaped_derived, traced_surface(backends=("llama_cpu", "llama_gpu")))
        self.assertTrue(recon.hard_failure)
        plan_obj = compile_ok(reconciliation=recon)
        self.assertIn(P.F_SURFACE_ESCAPE, codes(plan_obj))
        self.assertEqual(plan_obj.check.outcome, S.FAIL)
        self.assertFalse(any(b.cells_dropped for b in plan_obj.backends))

    def test_stage_disagreement_is_a_hard_finding_not_the_cheaper_answer(self):
        disagree = SU.backend_unchanged(stage1=stage1("llama_gpu", changed=False),
                                        stage2=stage2("llama_gpu", identical=False),
                                        transfer_scope=in_scope())
        plan_obj = compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                                    "llama_gpu": disagree})
        self.assertIn(P.F_BUILD_IDENTITY_STAGE_DISAGREEMENT, codes(plan_obj))
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertFalse(gpu.cells_dropped)
        self.assertTrue(gpu.cells)
        filed = [f for f in gpu.findings
                 if f.code == P.F_BUILD_IDENTITY_STAGE_DISAGREEMENT][0]
        self.assertEqual(filed.filed_against, "build_identity")
        self.assertEqual(plan_obj.check.outcome, S.FAIL)

    def test_reverse_disagreement_also_fails(self):
        disagree = SU.backend_unchanged(stage1=stage1("llama_gpu", changed=True),
                                        stage2=stage2("llama_gpu", identical=True),
                                        transfer_scope=in_scope())
        plan_obj = compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                                    "llama_gpu": disagree})
        self.assertIn(P.F_BUILD_IDENTITY_STAGE_DISAGREEMENT, codes(plan_obj))
        self.assertEqual(plan_obj.check.outcome, S.FAIL)

    def test_a_traced_backend_cannot_be_declared_unchanged(self):
        recon = SU.reconcile_surface(derived_surface(),
                                     traced_surface(backends=("llama_cpu", "llama_gpu")))
        self.assertFalse(recon.hard_failure)
        plan_obj = compile_ok(reconciliation=recon)
        self.assertIn(P.F_TRACED_BACKEND_DECLARED_UNCHANGED, codes(plan_obj))
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertFalse(gpu.cells_dropped)
        self.assertEqual(plan_obj.check.outcome, S.FAIL)

    def test_out_of_scope_incumbent_evidence_blocks_the_drop(self):
        stale = gpu_unchanged(transfer_scope=SU.EvidenceTransferScope(
            same_models=True, same_recipes=True, candidate_topology_hash="topo-2",
            incumbent_topology_hash="topo-1", era_boundary_crossed=False))
        plan_obj = compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                                    "llama_gpu": stale})
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertFalse(gpu.cells_dropped)

    def test_missing_test_keeps_the_matrix_and_is_recorded(self):
        plan_obj = compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed()})
        self.assertIn(P.F_BACKEND_UNCHANGED_TEST_NOT_RUN, codes(plan_obj))
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertTrue(gpu.cells)
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)

    def test_drop_suppresses_but_retains_cell_findings(self):
        priors = clean_priors()
        priors["roles"]["architect_general"]["serving"]["launch"]["runtime"][
            "ld_library_path"] = []
        plan_obj = compile_ok(priors=priors)
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertTrue(gpu.cells_dropped)
        self.assertNotIn(P.F_LINKAGE_REQUIREMENT_UNPROVEN,
                         [f.code for f in gpu.findings])
        self.assertIn(P.F_LINKAGE_REQUIREMENT_UNPROVEN,
                      [f.code for f in gpu.suppressed_findings])
        self.assertEqual(plan_obj.check.outcome, S.PASS)


# =============================================================================
# Protocols, thresholds, capacity, linkage, coverage
# =============================================================================

class TestPerCellJoins(unittest.TestCase):

    def test_adapter_prerequisites_are_preserved_with_their_exact_outcomes(self):
        binding_map = bindings()
        binding_map["llama_cpu"] = cpu_binding(prerequisites={
            "ratified_protocol_registry": S.Check(S.COULD_NOT_CHECK, ("missing",)),
            "pinned_instrument": S.Check(S.FAIL, ("wrong instrument",)),
        })
        plan_obj = compile_ok(bindings=binding_map)
        backend = next(b for b in plan_obj.backends if b.backend == "llama_cpu")
        self.assertEqual(
            backend.checks["prerequisite.ratified_protocol_registry"].outcome,
            S.COULD_NOT_CHECK)
        self.assertEqual(backend.checks["prerequisite.pinned_instrument"].outcome,
                         S.FAIL)
        self.assertEqual(plan_obj.check.outcome, S.FAIL)

    def test_undeclared_protocol_is_could_not_check_not_invented(self):
        binding_map = bindings()
        binding_map["llama_cpu"] = cpu_binding(
            protocols={"decode": phase_protocol("decode", "P-BENCH-1")})
        plan_obj = compile_ok(bindings=binding_map)
        self.assertIn(P.F_RELEASE_PROTOCOL_UNDEFINED, codes(plan_obj))
        prefill = [c for c in plan_obj.cells_for("llama_cpu") if c.phase == "prefill"]
        self.assertTrue(prefill)
        self.assertTrue(all(c.protocol is None for c in prefill))
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)

    def test_named_protocol_without_bands_is_its_own_finding(self):
        # "We know which protocol governs" and "we know its bands" are different facts,
        # and a compiler that supplied the second would be inventing a threshold.
        binding_map = bindings()
        binding_map["llama_cpu"] = cpu_binding(
            protocols={"decode": P.PhaseProtocol(
                phase="decode", protocol_id="P-BENCH-1", metric="tokens_per_second",
                direction="higher_better"),
                "prefill": phase_protocol("prefill", "P-BENCH-PREFILL-1")})
        plan_obj = compile_ok(bindings=binding_map)
        self.assertIn(P.F_PHASE_THRESHOLDS_UNDECLARED, codes(plan_obj))
        decode = [c for c in plan_obj.cells_for("llama_cpu") if c.phase == "decode"][0]
        self.assertEqual(decode.protocol.protocol_id, "P-BENCH-1")
        self.assertEqual(decode.checks["protocol"].outcome, S.COULD_NOT_CHECK)
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)

    def test_missing_quant_or_arch_is_a_finding(self):
        priors = clean_priors()
        priors["roles"]["frontdoor"]["model"]["quant"] = None
        priors["roles"]["frontdoor"]["model"]["arch"] = None
        plan_obj = compile_ok(priors=priors)
        self.assertIn(P.F_ROLE_RECIPE_INCOMPLETE, codes(plan_obj))
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)

    def test_a_backend_with_no_live_role_does_not_pass_vacuously(self):
        priors = {"roles": {k: v for k, v in clean_priors()["roles"].items()
                            if k not in ("architect_general", "coder_escalation")}}
        plan_obj = compile_ok(priors=priors,
                              unchanged_by_backend={"llama_cpu": cpu_changed(),
                                                    "llama_gpu": cpu_changed_gpu()})
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertEqual(gpu.cells, ())
        self.assertIsNone(gpu.transfer_receipt)
        self.assertIn(P.F_NO_PROTECTED_CELLS, codes(plan_obj))

    def test_capacity_floor_is_fixed_from_the_incumbent(self):
        cell = compile_ok().cells_for("llama_cpu")[0]
        self.assertEqual(cell.capacity_floor["resident_gb_max"], 37.0)
        self.assertIn("context_tokens_min", cell.capacity_floor)
        self.assertIn("must not need more memory", cell.capacity_floor["direction"])

    def test_missing_footprint_is_a_finding_not_a_zero_floor(self):
        priors = clean_priors()
        priors["roles"]["frontdoor"]["policy"] = {}
        priors["roles"]["frontdoor"]["model"]["mem_gb"] = None
        plan_obj = compile_ok(priors=priors)
        self.assertIn(P.F_CAPACITY_FLOOR_INCOMPLETE, codes(plan_obj))
        cell = [c for c in plan_obj.cells_for("llama_cpu")
                if c.context_tokens == 65536][0]
        self.assertIsNone(cell.capacity_floor["resident_gb_max"])
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)

    def test_missing_ld_library_path_is_a_finding(self):
        priors = clean_priors()
        for role in ("worker_general", "worker_math", "toolrunner", "frontdoor"):
            priors["roles"][role]["serving"]["launch"]["runtime"][
                "ld_library_path"] = []
        plan_obj = compile_ok(priors=priors)
        self.assertIn(P.F_LINKAGE_REQUIREMENT_UNPROVEN, codes(plan_obj))
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)

    def test_linkage_verifier_is_located_in_the_research_repo(self):
        cell = compile_ok().cells_for("llama_cpu")[0]
        self.assertEqual(cell.linkage.verifier_repo, "epyc-inference-research")
        self.assertTrue(cell.linkage.verifier.endswith("verify_ggml_linkage.sh"))

    def test_uncovered_affected_op_blocks(self):
        plan_obj = compile_ok(op_coverage=P.OpShapeCoverage(covered={}))
        self.assertIn(P.F_UNCOVERED_AFFECTED_OP, codes(plan_obj))
        cpu = [b for b in plan_obj.backends if b.backend == "llama_cpu"][0]
        self.assertEqual(cpu.uncovered_ops, ("MUL_MAT",))
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)

    def test_dropped_backend_owes_no_op_coverage(self):
        plan_obj = compile_ok(op_coverage=P.OpShapeCoverage(
            covered={"llama_cpu": {"MUL_MAT": ("4096x4096x1",)}}))
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertTrue(gpu.cells_dropped)
        self.assertEqual(gpu.uncovered_ops, ())
        self.assertEqual(plan_obj.check.outcome, S.PASS)

    def test_quality_transfer_refusal_is_advisory(self):
        transfer = P.QualityTransfer(backend="llama_cpu", model_key=f"sha256:{DIGEST_A}",
                                     paired_parity_proven=False,
                                     deterministic_replay_valid=True,
                                     era_boundary_crossed=False,
                                     evidence_ref="ev://x")
        plan_obj = compile_ok(quality_transfer=[transfer])
        self.assertIn(P.F_QUALITY_TRANSFER_REFUSED, codes(plan_obj))
        self.assertEqual(plan_obj.check.outcome, S.PASS)
        cell = [c for c in plan_obj.cells_for("llama_cpu")
                if c.quality_transfer is not None][0]
        self.assertEqual(cell.quality_transfer.check().outcome, S.FAIL)

    def test_quality_transfer_defaults_to_unknown_not_permitted(self):
        self.assertEqual(
            P.QualityTransfer(backend="llama_cpu", model_key="k").check().outcome,
            S.COULD_NOT_CHECK)

    def test_stable_path_receipt_is_required_and_checked(self):
        plan_obj = compile_ok(stable_path_receipts={})
        self.assertIn(P.F_STABLE_PATH_RECEIPT_MISSING, codes(plan_obj))

        wrong = dict(receipts())
        wrong["llama_cpu"] = P.StablePathReceipt(
            backend="llama_cpu", stable_path=STABLE_CPU,
            resolved_target="/mnt/raid0/llm/llama.cpp-experimental/build/bin",
            observed_at=NOW)
        bad = compile_ok(stable_path_receipts=wrong)
        self.assertIn(P.F_STABLE_PATH_NOT_IN_PRODUCTION_TREE, codes(bad))
        self.assertEqual(bad.check.outcome, S.FAIL)


# =============================================================================
# §10.6 — diff-complexity ceiling
# =============================================================================

class TestComplexityCeiling(unittest.TestCase):

    def test_within_the_ceiling_needs_no_review(self):
        plan_obj = compile_ok()
        self.assertFalse(plan_obj.requires_human_code_review)
        self.assertIsNone(plan_obj.to_dict()["review_marker"])

    def test_oversized_diff_marks_the_package(self):
        plan_obj = compile_ok(target=target(diff_lines=9000))
        self.assertTrue(plan_obj.requires_human_code_review)
        self.assertEqual(plan_obj.to_dict()["review_marker"],
                         P.REQUIRES_HUMAN_CODE_REVIEW)
        self.assertIn(P.F_DIFF_COMPLEXITY_CEILING_EXCEEDED, codes(plan_obj))

    def test_core_header_forces_review_at_any_size(self):
        plan_obj = compile_ok(target=target(change_classes=("core_header",),
                                            diff_lines=4, files_touched=1))
        self.assertTrue(plan_obj.requires_human_code_review)

    def test_shared_core_forces_review(self):
        plan_obj = compile_ok(target=target(touches_shared_core=True))
        self.assertTrue(plan_obj.requires_human_code_review)

    def test_ceiling_review_does_not_gate_the_plan(self):
        plan_obj = compile_ok(target=target(diff_lines=9000))
        self.assertEqual(plan_obj.check.outcome, S.PASS)


# =============================================================================
# The artifact itself
# =============================================================================

class TestPlanArtifact(unittest.TestCase):

    def test_dict_is_canonicalizable_and_hash_is_stable(self):
        a, b = compile_ok(), compile_ok()
        S.canonical_json(a.to_dict())
        self.assertEqual(a.sha256(), b.sha256())

    def test_hash_changes_with_scope(self):
        a = compile_ok()
        b = compile_ok(op_coverage=P.OpShapeCoverage(covered={}))
        self.assertNotEqual(a.sha256(), b.sha256())

    def test_no_authority_flavoured_keys(self):
        self.assertEqual(S.find_authority_flavoured_keys(compile_ok().to_dict()), [])

    def test_the_artifact_says_a_human_executes_it(self):
        payload = compile_ok().to_dict()
        self.assertEqual(payload["executed_by"], "operator")
        self.assertIn("never executes a freeze", payload["notice"])

    def test_cells_carry_no_command_line(self):
        for cell in compile_ok().to_dict()["backends"][0]["cells"]:
            self.assertIsNone(cell["command_line"])
            self.assertTrue(cell["recipe_constructor_required"])

    def test_every_cell_is_at_the_production_optimal_recipe(self):
        for cell in compile_ok().cells:
            self.assertEqual(cell.recipe_class, P.RECIPE_CLASS)

    def test_nothing_leaves_the_matrix_without_a_receipt(self):
        priors = clean_priors()
        priors["roles"]["mystery"] = role_prior(
            binary="/mnt/raid0/llm/other-engine/bin/server", model_path=MODEL_A,
            entries=[entry(9000)])
        plan_obj = compile_ok(priors=priors)
        unplannable = {u.role: u.code for u in plan_obj.unplannable_roles}
        self.assertEqual(unplannable, {"mystery": P.F_ROLE_BINARY_UNCLASSIFIED})
        self.assertIn(P.F_ROLE_BINARY_UNCLASSIFIED, codes(plan_obj))
        self.assertEqual(plan_obj.check.outcome, S.COULD_NOT_CHECK)
        self.assertEqual([r.role for r in plan_obj.out_of_scope_roles],
                         ["qwen35_122b_q4km"])

    def test_a_role_on_another_tree_is_out_of_scope_not_unclassified(self):
        binding_map = bindings()
        binding_map["whisper_stt"] = P.BackendBinding(
            backend="whisper_stt",
            stable_production_path="/mnt/raid0/llm/kernels/production/stt",
            production_tree_path="/mnt/raid0/llm/whisper.cpp",
            binary_roots=("/mnt/raid0/llm/whisper.cpp/build/bin",),
            phases=("encode", "decode", "end_to_end"))
        priors = clean_priors()
        priors["roles"]["speech"] = role_prior(
            binary="/mnt/raid0/llm/whisper.cpp/build/bin/whisper-server",
            model_path="/mnt/raid0/llm/models/ggml-large-v3.bin",
            entries=[entry(8090)])
        plan_obj = compile_ok(priors=priors, bindings=binding_map)
        reasons = {r.role: r.reason for r in plan_obj.out_of_scope_roles}
        self.assertIn("speech", reasons)
        self.assertIn("not in this tree's freeze scope", reasons["speech"])
        self.assertEqual(plan_obj.unplannable_roles, ())

    def test_canary_roles_track_the_kept_cells(self):
        plan_obj = compile_ok()
        cpu = [b for b in plan_obj.backends if b.backend == "llama_cpu"][0]
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertEqual(cpu.canary_roles,
                         ("frontdoor", "toolrunner", "worker_general", "worker_math"))
        self.assertEqual(gpu.canary_roles, ())

    def test_plan_embeds_into_a_valid_release_package(self):
        plan_obj = compile_ok()
        package = {
            "schema": S.SCHEMA_RELEASE_PACKAGE,
            "package_id": "akr-0001",
            "campaign_id": "ak-0001",
            "source_tree": "llama.cpp",
            "sealed_candidate": {"candidate_id": "akc-0001", "seal_sha256": "1" * 64,
                                 "binary_sha256": "2" * 64, "linkage_sha256": "3" * 64,
                                 "build_receipt_sha256": "4" * 64},
            "t3_verdict": {"verdict": "PASS", "bundle_sha256": "5" * 64,
                           "phase_results": {"identity_preflight": "PASS"}},
            "active_waivers": [],
            "release_plan": plan_obj.to_dict(),
            "transaction_plan": {"next_version": "production-consolidated-v9"},
            "rollback_plan": {"incumbent_archive_path": "/mnt/raid0/llm/kernels/archive/v8",
                              "incumbent_binary_sha256": "6" * 64},
            "draft_era_registry_row": {"era_id": "E9-cpu-kernel"},
            "draft_autopilot_rebaseline_note": "fail-closed hold until operator reseed",
            "linkage_verification": {"status": S.PASS, "receipt": "lnk://fixture"},
            "operator_command_sequence": [
                {"command": "git tag production-consolidated-v9",
                 "validation_receipt": "val://1", "validated": True}],
            "change_classes": ["arithmetic"],
            "requires_human_code_review": plan_obj.requires_human_code_review,
            "diff_complexity": {"diff_size": 120, "files_touched": 3,
                                "touches_shared_core": False},
            "created_at": NOW,
        }
        self.assertEqual(S.validate_release_package(package), [])


# =============================================================================
# Self-audit and cross-module agreements
# =============================================================================

class TestSelfAudit(unittest.TestCase):

    def test_module_writes_nothing_and_signals_nothing(self):
        self.assertEqual(P.audit_plan_module_is_read_only().outcome, S.PASS)

    def test_audit_catches_a_write(self):
        bad = "from pathlib import Path\ndef f(p):\n    Path(p).write_text('x')\n"
        self.assertEqual(P.audit_plan_module_is_read_only(bad).outcome, S.FAIL)

    def test_production_branch_regex_agrees_with_schemas(self):
        self.assertEqual(P.PRODUCTION_BRANCH_RE.pattern,
                         S._PRODUCTION_BRANCH_RE.pattern)

    def test_worst_check_agrees_with_the_evaluator_reducer(self):
        cases = [S.Check(S.PASS), S.Check(S.COULD_NOT_CHECK, ("a",)),
                 S.Check(S.FAIL, ("b",))]
        for left in cases:
            for right in cases:
                self.assertEqual(P.worst_check(left, right).outcome,
                                 SU._combine_checks(left, right).outcome)
        self.assertEqual(P.worst_check().outcome, S.PASS)

    def test_source_tree_map_is_the_schema_one(self):
        served = {b for b, t in S.SOURCE_TREE_BY_BACKEND.items() if t == "llama.cpp"}
        self.assertEqual(served, {"llama_cpu", "llama_gpu"})
        self.assertNotIn(P.STACK_CHANGE_BACKEND, S.SOURCE_TREE_BY_BACKEND)

    def test_every_finding_code_has_a_spec(self):
        for code in P.FINDING_CODES:
            severity, outcome, meaning = P.FINDING_SPEC[code]
            self.assertIn(severity, P.SEVERITIES)
            self.assertIn(outcome, (S.PASS, S.FAIL, S.COULD_NOT_CHECK))
            self.assertTrue(meaning)
        # A blocking finding may never carry a PASS outcome: it would gate on nothing.
        for code, (severity, outcome, _) in P.FINDING_SPEC.items():
            if severity == P.SEVERITY_BLOCKING:
                self.assertNotEqual(outcome, S.PASS, code)

    def test_cell_refuses_an_off_recipe_class_or_bad_residency_label(self):
        cell = compile_ok().cells[0]
        fields = {f: getattr(cell, f) for f in
                  ("cell_id", "backend", "phase", "model", "quant", "architecture",
                   "context_tokens", "kv_type_k", "kv_type_v", "ubatch", "concurrency",
                   "speculation", "placement", "co_residency", "recipe_class",
                   "protocol", "capacity_floor", "linkage", "protected_roles",
                   "protected_entries", "checks", "quality_transfer")}
        with self.assertRaises(P.PlanInputError):
            P.ReleaseCell(**{**fields, "recipe_class": "baseline"})
        with self.assertRaises(P.PlanInputError):
            P.ReleaseCell(**{**fields, "co_residency": "co_resident"})

    def test_finding_severity_cannot_be_chosen(self):
        finding = P.PlanFinding(code=P.F_SURFACE_ESCAPE, detail="x")
        self.assertEqual(finding.severity, P.SEVERITY_BLOCKING)
        self.assertEqual(finding.outcome, S.FAIL)
        with self.assertRaises(P.PlanInputError):
            P.PlanFinding(code="INVENTED", detail="x")
        with self.assertRaises(P.PlanInputError):
            P.PlanFinding(code=P.F_SURFACE_ESCAPE, detail="   ")


# =============================================================================
# The loader
# =============================================================================

class TestLoader(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_json_round_trip(self):
        path = self.tmp / "priors.json"
        path.write_text(json.dumps(clean_priors()), encoding="utf-8")
        loaded = P.load_compiled_priors(path)
        facts, _, out = P.extract_role_facts(loaded)
        self.assertEqual(len(facts), 6)
        self.assertEqual(len(out), 1)

    def test_yaml_round_trip(self):
        yaml = self._yaml_or_skip()
        path = self.tmp / "priors.yaml"
        path.write_text(yaml.safe_dump(clean_priors()), encoding="utf-8")
        facts, _, _ = P.extract_role_facts(P.load_compiled_priors(path))
        self.assertEqual(len(facts), 6)

    def test_unknown_suffix_refuses(self):
        path = self.tmp / "priors.txt"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(P.PlanInputError):
            P.load_compiled_priors(path)

    def test_non_mapping_refuses(self):
        path = self.tmp / "priors.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(P.PlanInputError):
            P.load_compiled_priors(path)

    def test_missing_file_raises_rather_than_returning_empty(self):
        with self.assertRaises(OSError):
            P.load_compiled_priors(self.tmp / "absent.json")

    def _yaml_or_skip(self):
        try:
            import yaml
        except ImportError:  # pragma: no cover
            self.skipTest("PyYAML is not installed")
        return yaml


# =============================================================================
# Red-team regressions (2026-08-03)
#
# Each test below is a defect that WAS present and is now fixed. They are grouped
# here rather than folded into the sections above so that the barrier they form is
# visible as one: every one of them made a release CHEAPER or QUIETER, and every one
# of them passed the original 90 tests.
# =============================================================================

class TestPathTraversalContainment(unittest.TestCase):
    """`path_is_under` compares paths it cannot resolve, so a `..` broke it BOTH ways.

    The module performs no I/O by design, so `/a/b/../c` was compared verbatim. That
    made the production-tree containment test simultaneously over- and under-inclusive,
    and the under-inclusive direction cleared invariant 3's refusal.
    """

    def test_traversing_build_root_cannot_clear_the_production_write_refusal(self):
        # Resolves to /mnt/raid0/llm/llama.cpp/build-ak — INSIDE the frozen production
        # tree — but does not begin with it as a string, so the containment test said
        # "outside" and the plan compiled.
        evil = "/mnt/raid0/llm/llama.cpp-experimental/../llama.cpp/build-ak"
        with self.assertRaises(P.PlanInputError) as ctx:
            target(candidate_build_root=evil)
        self.assertIn("..", str(ctx.exception))
        with self.assertRaises(P.PlanInputError):
            P.normalize_path("/mnt/raid0/llm/./llama.cpp/build-ak", label="x")

    def test_traversing_receipt_target_cannot_clear_the_stable_path_check(self):
        # Resolves to /mnt/raid0/llm/elsewhere/bin — OUTSIDE the production tree — but
        # `startswith(TREE + "/")` said inside, so F_STABLE_PATH_NOT_IN_PRODUCTION_TREE
        # never fired for the one case it exists to catch.
        with self.assertRaises(P.PlanInputError):
            P.StablePathReceipt(backend="llama_cpu", stable_path=STABLE_CPU,
                                resolved_target=f"{TREE}/../elsewhere/bin",
                                observed_at=NOW)

    def test_a_traversing_role_binary_is_unplannable_not_misclassified(self):
        priors = clean_priors()
        priors["roles"]["frontdoor"]["serving"]["launch"]["runtime"]["binary_path"] = (
            f"{TREE}/build/bin/../../../whisper.cpp/build/bin/whisper-server")
        plan_obj = compile_ok(priors=priors)
        self.assertIn("frontdoor", [r.role for r in plan_obj.unplannable_roles])
        self.assertNotIn("frontdoor", plan_obj.protected_roles())

    def test_containment_still_holds_for_ordinary_paths(self):
        self.assertTrue(P.path_is_under(TREE, CPU_BIN))
        self.assertTrue(P.path_is_under(TREE, TREE))
        self.assertFalse(P.path_is_under(f"{TREE}/build", GPU_BIN))


class TestAnchoredPatternMatching(unittest.TestCase):
    """`re.match(r"^…$", x)` also accepts `x + "\\n"` — `$` matches before a final \\n.

    A digest piped out of `sha256sum` without `.strip()` therefore cleared both the
    format check and `is_placeholder_digest`, whose own regex has the same shape.
    """

    def test_placeholder_digest_with_a_trailing_newline_is_still_a_placeholder(self):
        smuggled = "0" * 64 + "\n"
        self.assertFalse(S.is_placeholder_digest(smuggled))  # schemas cannot see it
        with self.assertRaises(P.PlanInputError):
            P.IncumbentEvidence(backend="llama_gpu", era_id="E8",
                                artifacts=((f"{ARCHIVE}/llama-server", smuggled),))
        with self.assertRaises(P.PlanInputError):
            P.ModelIdentity(model_path=MODEL_A, sha256=smuggled)

    def test_two_models_cannot_share_one_smuggled_digest_and_merge(self):
        # The consequence of the above on the DEDUP key: one fabricated digest on two
        # different GGUFs collapsed two cells into one, and the second model stopped
        # being measured. Now the roles are retained as unplannable with a reason
        # instead — a role can leave the matrix, but not without a receipt.
        smuggled = "0" * 64 + "\n"
        self.assertEqual(P.ModelIdentity(model_path=MODEL_A, sha256=None).key,
                         f"path:{MODEL_A}")
        plan_obj = compile_ok(model_digests={MODEL_A: smuggled, MODEL_B: smuggled})
        self.assertEqual(plan_obj.cells, ())
        self.assertEqual({r.code for r in plan_obj.unplannable_roles},
                         {P.F_ROLE_RECIPE_INCOMPLETE})
        self.assertTrue(all("sha256" in r.reason for r in plan_obj.unplannable_roles))
        self.assertNotEqual(plan_obj.check.outcome, S.PASS)

    def test_commit_and_residency_labels_are_fully_anchored(self):
        with self.assertRaises(P.PlanInputError):
            target(candidate_commit=CAND_COMMIT + "\n")
        cell = compile_ok().cells[0]
        fields = {f: getattr(cell, f) for f in
                  ("cell_id", "backend", "phase", "model", "quant", "architecture",
                   "context_tokens", "kv_type_k", "kv_type_v", "ubatch", "concurrency",
                   "speculation", "placement", "co_residency", "recipe_class",
                   "protocol", "capacity_floor", "linkage", "protected_roles",
                   "protected_entries", "checks", "quality_transfer")}
        with self.assertRaises(P.PlanInputError):
            P.ReleaseCell(**{**fields, "co_residency": "single\n"})

    def test_a_frozen_branch_name_with_a_trailing_newline_is_still_refused(self):
        # The refusal predicate keeps its loose anchoring on purpose: erring towards
        # refusing is the safe direction for a production-branch test.
        with self.assertRaises(P.ProductionWriteRefused):
            target(candidate_branch="production-consolidated-v8\n")


class TestDropVerdictIsRederived(unittest.TestCase):
    """`may_drop_cells` is a plain FIELD on a constructible dataclass, not a property.

    `backend_unchanged()` derives it correctly, but the compiler read it rather than
    re-deriving it — so a result built by hand emptied an entire backend's matrix and
    the plan still read PASS.
    """

    @staticmethod
    def _forged(**overrides):
        kwargs = dict(
            backend="llama_gpu",
            stage1=stage1("llama_gpu", changed=True),
            stage2=stage2("llama_gpu", identical=False),
            transfer_scope=SU.EvidenceTransferScope(),
            agreement=S.Check(S.FAIL, ("the stages disagree",)),
            unchanged=S.Check(S.FAIL, ("the normalized binary differs",)),
            may_drop_cells=True, findings=(), blocking_reasons=())
        kwargs.update(overrides)
        return SU.BackendUnchangedResult(**kwargs)

    def test_a_verdict_contradicting_its_own_evidence_is_refused(self):
        with self.assertRaises(P.PlanInputError) as ctx:
            compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                             "llama_gpu": self._forged()})
        self.assertIn("may_drop_cells", str(ctx.exception))

    def test_each_precondition_is_checked_individually(self):
        clean = dict(stage1=stage1("llama_gpu", changed=False),
                     stage2=stage2("llama_gpu", identical=True),
                     transfer_scope=in_scope(),
                     agreement=S.Check(S.PASS), unchanged=S.Check(S.PASS))
        # The fully consistent hand-built result is accepted — the check is about
        # agreement with the evidence, not about provenance of the object.
        self.assertEqual(P.drop_verdict_contradictions(self._forged(**clean)), ())
        for name, broken in (
                ("stage1", dict(stage1=stage1("llama_gpu", changed=True))),
                ("stage2_absent", dict(stage2=None)),
                ("stage2", dict(stage2=stage2("llama_gpu", identical=False))),
                ("agreement", dict(agreement=S.Check(S.FAIL, ("x",)))),
                ("unchanged", dict(unchanged=S.Check(S.COULD_NOT_CHECK, ("x",)))),
                ("scope", dict(transfer_scope=SU.EvidenceTransferScope())),
                ("findings", dict(findings=(SU.BuildIdentityFinding(
                    code=SU.FINDING_STAGE2_NOT_RUN, severity="blocking", detail="x"),))),
                ("blocking", dict(blocking_reasons=("x",)))):
            with self.subTest(name):
                self.assertTrue(
                    P.drop_verdict_contradictions(self._forged(**{**clean, **broken})),
                    f"{name} did not contradict may_drop_cells=True")

    def test_a_genuine_backend_unchanged_result_still_drops(self):
        plan_obj = compile_ok()
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertTrue(gpu.cells_dropped)
        self.assertEqual(P.drop_verdict_contradictions(gpu_unchanged()), ())

    def test_single_backend_speech_noop_is_a_failure_not_a_transfer(self):
        backend = "whisper_stt"
        tree = "/mnt/raid0/llm/whisper.cpp"
        stable = "/mnt/raid0/llm/kernels/production/stt"
        build = "/mnt/raid0/llm/whisper.cpp-experimental/build/bin"
        binding = P.BackendBinding(
            backend=backend, stable_production_path=stable,
            production_tree_path=tree, binary_roots=(stable, f"{tree}/build/bin"),
            phases=("encode", "decode", "end_to_end"))
        target_obj = target(
            source_tree="whisper.cpp", backends=(backend,),
            candidate_branch="whisper.cpp-experimental/ak9-noop",
            candidate_build_root=build)
        derived = SU.AffectedSurface(
            candidate_id="akc-0001", backends=(backend,),
            link_targets=("whisper-cli",), objects=("whisper.o",),
            touched_files=("src/whisper.cpp",), symbols=("whisper_decode",),
            op_registrations=(), dispatch_predicates=(), over_approximations=(),
            axes_derived=SU.SURFACE_AXES, coverage=S.Check(S.PASS),
            full_tree=False, inputs={"fixture": "speech-noop"})
        unchanged = SU.backend_unchanged(
            stage1=stage1(backend, changed=False),
            stage2=stage2(backend, identical=True), transfer_scope=in_scope())
        plan_obj = P.compile_release_plan(
            target=target_obj, bindings={backend: binding}, priors={"roles": {}},
            reconciliation=SU.reconcile_surface(derived, None), compiled_at=NOW,
            unchanged_by_backend={backend: unchanged},
            incumbent_evidence={backend: incumbent(backend)},
            stable_path_receipts={backend: P.StablePathReceipt(
                backend=backend, stable_path=stable,
                resolved_target=f"{tree}/build/bin", observed_at=NOW)},
            op_coverage=P.OpShapeCoverage(covered={}), model_digests={})
        speech = plan_obj.backends[0]
        self.assertIsNone(speech.transfer_receipt)
        self.assertIn(P.F_SINGLE_BACKEND_NOOP_CANDIDATE,
                      [finding.code for finding in speech.findings])
        self.assertEqual(plan_obj.check.outcome, S.FAIL)

    def test_a_stage_pair_naming_no_commits_may_not_drop_cells(self):
        # `compile_release_plan` cross-checks the stage commits against the release only
        # `if base is not None`, so a result with null commits passed vacuously and took
        # the matrix with it. Absence of the cross-check is not a passed cross-check.
        unanchored = SU.backend_unchanged(
            stage1=SU.SourceClosureIdentity(
                backend="llama_gpu", closure_size=1, changed_in_closure=(),
                unmapped_diff_paths=(), toolchain_differences=(), check=S.Check(S.PASS)),
            stage2=SU.NormalizedBinaryIdentity(
                backend="llama_gpu", candidate_ref="c", base_ref="b", differing=(),
                rebuild_verified=True, check=S.Check(S.PASS)),
            transfer_scope=in_scope())
        self.assertTrue(unanchored.may_drop_cells)
        plan_obj = compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                                    "llama_gpu": unanchored})
        gpu = [b for b in plan_obj.backends if b.backend == "llama_gpu"][0]
        self.assertFalse(gpu.cells_dropped)
        self.assertTrue(gpu.cells)
        self.assertIn(P.F_BACKEND_UNCHANGED_RESULT_UNANCHORED, codes(plan_obj))
        self.assertNotEqual(plan_obj.check.outcome, S.PASS)

    def test_a_disagreement_is_filed_from_the_agreement_not_the_findings_tuple(self):
        # Deleting the thing the check inspects: `findings=()` while `agreement` is
        # still FAIL used to file nothing at all and leave the plan at PASS.
        stripped = self._forged(stage1=stage1("llama_gpu", changed=False),
                                transfer_scope=in_scope(), may_drop_cells=False,
                                agreement=S.Check(S.FAIL, ("source clean, binary differs",)))
        plan_obj = compile_ok(unchanged_by_backend={"llama_cpu": cpu_changed(),
                                                    "llama_gpu": stripped},
                              incumbent_evidence={})
        self.assertIn(P.F_BUILD_IDENTITY_STAGE_DISAGREEMENT, codes(plan_obj))
        self.assertEqual(plan_obj.check.outcome, S.FAIL)
        filed = [f for b in plan_obj.backends for f in b.findings
                 if f.code == P.F_BUILD_IDENTITY_STAGE_DISAGREEMENT]
        self.assertEqual({f.filed_against for f in filed}, {"build_identity"})


class TestUndeclaredCeilingIsNotAnInfiniteOne(unittest.TestCase):
    """§10.6 was skipped entirely when an adapter declared no `ComplexityCeiling`.

    The kind-based rules (`core_header`, shared ggml core) are not size bands at all
    (AK-D30), and they were being skipped along with the bands.
    """

    def test_a_core_header_diff_forces_review_without_a_declared_ceiling(self):
        plan_obj = compile_ok(
            target=target(diff_lines=250_000, files_touched=4_000,
                          touches_shared_core=True, change_classes=("core_header",)),
            bindings={"llama_cpu": cpu_binding(ceiling=None),
                      "llama_gpu": gpu_binding(ceiling=None)})
        self.assertTrue(plan_obj.requires_human_code_review)
        self.assertEqual(plan_obj.to_dict()["review_marker"],
                         P.REQUIRES_HUMAN_CODE_REVIEW)
        joined = " | ".join(plan_obj.review_reasons)
        self.assertIn("core_header", joined)
        self.assertIn("shared ggml core", joined)

    def test_an_undeclared_ceiling_is_itself_a_review_reason(self):
        plan_obj = compile_ok(bindings={"llama_cpu": cpu_binding(ceiling=None),
                                        "llama_gpu": gpu_binding()})
        self.assertTrue(plan_obj.requires_human_code_review)
        self.assertIn("llama_cpu: no complexity/blast-radius ceiling is declared, so "
                      "the diff's size was never evaluated against one (§10.6)",
                      plan_obj.review_reasons)
        self.assertNotIn("llama_gpu: no complexity", " | ".join(plan_obj.review_reasons))

    def test_a_declared_ceiling_that_is_met_still_needs_no_review(self):
        self.assertFalse(compile_ok().requires_human_code_review)


class TestMalformedPriorsAreRecordedNotFatal(unittest.TestCase):
    """A data defect in the priors must become a record, never an abort or a silence."""

    def test_an_unusable_ld_library_path_is_a_finding_not_a_crash(self):
        # One relative string in one launcher's environment aborted the whole compile,
        # which is the one outcome that produces no record at all.
        priors = clean_priors()
        (priors["roles"]["worker_general"]["serving"]["launch"]["runtime"]
         ["ld_library_path"]) = ["./lib", CPU_BIN_DIR]
        plan_obj = compile_ok(priors=priors)
        self.assertIn(P.F_LINKAGE_REQUIREMENT_UNPROVEN, codes(plan_obj))
        self.assertNotEqual(plan_obj.check.outcome, S.PASS)

    def test_a_malformed_launch_entry_leaves_a_receipt(self):
        # An entry is a SERVER. The filtering comprehension dropped one with no record —
        # the seed's silent `continue`, one level down.
        priors = clean_priors()
        priors["roles"]["worker_general"]["serving"]["launch"]["entries"] = [
            entry(8072), "garbage"]
        plan_obj = compile_ok(priors=priors)
        recorded = [r for r in plan_obj.unplannable_roles if r.role == "worker_general"]
        self.assertTrue(recorded)
        self.assertIn("entries[1]", recorded[0].reason)

    def test_all_entries_malformed_does_not_masquerade_as_one_entryless_server(self):
        priors = clean_priors()
        priors["roles"]["frontdoor"]["serving"]["launch"]["entries"] = ["a", 7]
        plan_obj = compile_ok(priors=priors)
        self.assertEqual(
            2, len([r for r in plan_obj.unplannable_roles if r.role == "frontdoor"]))

    def test_a_non_string_role_key_does_not_raise_typeerror(self):
        facts, unplannable, out_of_scope = P.extract_role_facts(
            {"roles": {8072: {"deployment_status": "retired"},
                       "frontdoor": {"deployment_status": "retired"}}})
        self.assertEqual((facts, unplannable), ((), ()))
        self.assertEqual({r.role for r in out_of_scope}, {"8072", "frontdoor"})


if __name__ == "__main__":
    unittest.main()
