"""The gates, and the one property that makes them gates: order."""
import ast
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from autokernel.loop import archive, gates, gdn_reference


def _function_node(source, name):
    """Return one complete function node without coupling tests to source spelling."""
    nodes = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == name]
    assert len(nodes) == 1, f"expected one {name}, found {len(nodes)}"
    return nodes[0]


def _function(source, name):
    return ast.unparse(_function_node(source, name))


def _calls(node, name):
    """Find semantic calls by final function name, independent of call formatting."""
    return [call for call in ast.walk(node) if isinstance(call, ast.Call)
            and ((isinstance(call.func, ast.Name) and call.func.id == name)
                 or (isinstance(call.func, ast.Attribute) and call.func.attr == name))]


def _champion_build(dest, targets=gates.DEFAULT_TARGETS):
    """Stand in for `gates.compiles`: produce the binary the anchor is measured with."""
    (Path(dest) / "bin").mkdir(parents=True, exist_ok=True)
    (Path(dest) / "bin" / "llama-bench").write_text("elf", encoding="utf-8")
    return gates.Verdict("compile", True)


def _broken_build(dest, targets=gates.DEFAULT_TARGETS):
    return gates.Verdict("compile", False, "build failed")


class TheShortCircuitMustBeReal(unittest.TestCase):
    """`run_all` documented a short-circuit it could not perform.

    It took `*verdicts: Verdict`, and Python evaluates every argument before the call,
    so `run_all(compiles(...), op_correctness(...))` ran the correctness suite even when
    the build had just failed -- against whatever stale binary sat in the candidate build
    directory. The reported verdicts stayed correct -- the first failure short-circuits the
    RETURN, so the extra verdict was discarded -- but every failed build in run 9 still paid
    for a full test-backend-ops run against a stale artifact. A gate that runs after the
    previous gate refused is not a gate, even when nobody reads its answer.
    """

    def test_a_later_check_is_never_called_after_a_refusal(self):
        ran = []

        def failing():
            ran.append("build")
            return gates.Verdict("compile", False, "build failed")

        def must_not_run():
            ran.append("correctness")
            return gates.Verdict("correctness", True)

        passed, verdicts = gates.run_all(failing, must_not_run)
        self.assertFalse(passed)
        self.assertEqual(ran, ["build"],
                         "the correctness suite ran against a binary the build never made")
        self.assertEqual(len(verdicts), 1)

    def test_all_checks_run_when_each_passes(self):
        ran = []

        def ok(name):
            def check():
                ran.append(name)
                return gates.Verdict(name, True)
            return check

        passed, verdicts = gates.run_all(ok("compile"), ok("correctness"))
        self.assertTrue(passed)
        self.assertEqual(ran, ["compile", "correctness"])
        self.assertEqual(len(verdicts), 2)


class HardenedCandidateGates(unittest.TestCase):
    def test_determinism_compares_output_hashes_not_return_codes(self):
        rows = iter((SimpleNamespace(autokernel_output_hashes="a/a"),
                     SimpleNamespace(autokernel_output_hashes="b/b")))
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(gates.subprocess, "run", return_value=mock.Mock(
                 returncode=0, stdout="[]", stderr="")), \
             mock.patch.object(gates.residency, "loader_env", return_value={}), \
             mock.patch.object(gates.bench, "hardened_row", side_effect=lambda *a, **k: next(rows)):
            verdict = gates.deterministic(Path("/build"), Path("/model"), runs=2)
        self.assertFalse(verdict.passed)
        self.assertIn("outputs changed", verdict.reason)

    def test_no_fallback_dispatch_rejects_cpu_assignment(self):
        row = {"state": gates.census.OBSERVED, "nodes_total": 2000,
               "op_backend": {"MUL_MAT": {"ROCm0": 20, "CPU": 1}}}
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(gates.census, "run_dispatch_probe", return_value=row), \
             mock.patch.object(gates.residency, "loader_env", return_value={}):
            verdict = gates.no_fallback_dispatch(
                Path("/build"), Path("/model"), pp=512, tg=0)
        self.assertFalse(verdict.passed)
        self.assertIn("fallback event", verdict.reason)

    def test_live_gate_chain_names_both_hardenings(self):
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        gate_node = next(node for node in ast.walk(ast.parse(source))
                         if isinstance(node, ast.FunctionDef) and node.name == "gate")
        body = ast.unparse(gate_node)
        self.assertIn("gates.deterministic", body)
        self.assertIn("gates.no_fallback_dispatch", body)

    def test_the_runner_passes_callables_not_evaluated_verdicts(self):
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        # Select every source-build chain, not the runtime-only oracles. Whole-source
        # target validation intentionally adds a second compile/correctness chain.
        chains = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
                  and ast.unparse(node.func) == "gates.run_all"
                  and any(isinstance(arg, ast.Lambda) and isinstance(arg.body, ast.Call)
                          and ast.unparse(arg.body.func) == "gates.compiles" for arg in node.args)]
        self.assertTrue(chains)
        for chain in chains:
            self.assertTrue(all(isinstance(arg, ast.Lambda) for arg in chain.args))
            self.assertEqual([ast.unparse(arg.body.func) for arg in chain.args],
                             ["gates.compiles", "gates.op_correctness"])


class ARefusedPatchMustSurviveTheReset(unittest.TestCase):
    """Run 9 lost all ten candidate patches.

    `reset_tree` returns the worktree to the champion before each iteration, so a
    refused patch exists nowhere afterwards. Seven of those ten died on `MUL_MAT failed
    on ROCm0` and not one can be reproduced, re-read or diagnosed. A negative written up
    without its diff is not evidence anyone can act on.
    """

    def test_the_runner_saves_the_diff_before_gating(self):
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        self.assertIn("def keep_the_diff(", source)
        # It must run BEFORE the gate, because a failed build still leaves a patch
        # worth reading and that is the last moment it exists on disk.
        gate_node = next(node for node in ast.walk(ast.parse(source))
                         if isinstance(node, ast.FunctionDef) and node.name == "gate")
        # `keep_the_diff` takes the LANE as well as the hypothesis since the gate
        # became per-worker: with concurrent lanes a bare `<mechanism>.patch` is two
        # lanes overwriting one file, which loses diffs the same way run 9 did. The
        # property under test is the ORDER, so match the call, not one spelling of
        # its argument list.
        calls = {ast.unparse(node.func): node.lineno for node in ast.walk(gate_node)
                 if isinstance(node, ast.Call)}
        self.assertLess(calls["keep_the_diff"], calls["gates.compiles"])

    def test_an_empty_diff_writes_nothing(self):
        """An actor that changed nothing must not leave an empty patch file that
        later reads as a real attempt."""
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        self.assertIn("archive.retain_patch", _function(source, "keep_the_diff"))
        body = inspect.getsource(archive.retain_patch)
        self.assertIn("if not patch:", body)
        self.assertIn("return None", body)


class AnOracleThatCannotRunIsNotAFailedPatch(unittest.TestCase):
    """The worst defect in this rebuild: the correctness gate never ran, and said the
    patch was wrong every time.

    `op_correctness` passed `--suite-seed <n>`, which test-backend-ops does not accept
    in this tree. It printed usage and exited 1, so EVERY candidate was refused with
    "MUL_MAT failed on ROCm0". Proven against the anchor, which passes 1139/1139: the
    exact gate command exits 1, and exits 0 with the flag removed. Seven of ten run-9
    iterations died on this and were written into durable memory as measured negatives.
    """

    def _fake_run(self, stdout, code):
        from unittest import mock
        return mock.patch.object(gates.subprocess, "run",
                                 return_value=mock.Mock(stdout=stdout, stderr="",
                                                        returncode=code))

    def test_usage_text_is_a_harness_fault_not_a_correctness_verdict(self):
        usage = "Usage: test-backend-ops [mode] [-o <op,..>]\n    valid modes:\n"
        with mock.patch.object(Path, "is_file", return_value=True), \
                self._fake_run(usage, 1):
            verdict = gates.op_correctness(Path("/nonexistent"))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gate, "oracle_unavailable",
                         "an argument error must never read as a failed patch")
        self.assertNotIn("MUL_MAT failed", verdict.reason)

    def test_a_real_failure_is_still_a_correctness_verdict(self):
        ran = ("Backend 1/2: ROCm0\n  1100/1139 tests passed\n"
               "  Backend ROCm0: FAIL\n1/2 backends passed\n")
        with mock.patch.object(Path, "is_file", return_value=True), \
                self._fake_run(ran, 1):
            verdict = gates.op_correctness(Path("/nonexistent"))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gate, "correctness")
        self.assertIn("MUL_MAT failed", verdict.reason)

    def test_a_pass_requires_proof_the_suite_executed(self):
        ran = ("Backend 1/2: ROCm0\n  1139/1139 tests passed\n"
               "  Backend ROCm0: \033[1;32mOK\033[0m\n2/2 backends passed\nOK\n")
        with mock.patch.object(Path, "is_file", return_value=True), \
                self._fake_run(ran, 0):
            self.assertTrue(gates.op_correctness(Path("/nonexistent")).passed)

    def test_a_silent_zero_exit_does_not_pass(self):
        """Exit 0 with no evidence the suite ran is not a pass."""
        with mock.patch.object(Path, "is_file", return_value=True), \
                self._fake_run("", 0):
            verdict = gates.op_correctness(Path("/nonexistent"))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gate, "oracle_unavailable")

    def test_other_backend_or_zero_target_cases_cannot_pass(self):
        for output in (
            "Backend 1/1: CPU\n  3/3 tests passed\n  Backend CPU: OK\n1/1 backends passed\n",
            "Backend 1/2: ROCm0\n  0/0 tests passed\n  Backend ROCm0: OK\n2/2 backends passed\n",
        ):
            with mock.patch.object(Path, "is_file", return_value=True), \
                    self._fake_run(output, 0):
                verdict = gates.op_correctness(Path("/nonexistent"), op="GATED_DELTA_NET")
            self.assertEqual(verdict.gate, "oracle_unavailable")


_DS41_ANCHOR = Path("/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925")


class TheOpSelectorIsWhatTestBackendOpsMatches(unittest.TestCase):
    """DS41-C96: `-o GLU` selected 0/0 CPU cases, on every cpu_graph_sync candidate.

    test-backend-ops matches `-o` against `ggml_op_desc()`, which names a UNARY or GLU
    node by its sub-op (SILU, SWIGLU, ...), never "UNARY" or "GLU". The route's ten
    earlier suites passed (MUL_MAT 1139/1139 ... CONCAT 210/210) and the eleventh, GLU,
    printed `0/0 tests passed`: 10 `oracle_unavailable` refusals and 5 scope_blocked
    hypotheses (experiments rows 108-213), and a planner that abstained 8 of 12 times.
    """

    _PASS = ("Backend 1/1: CPU\n  SWIGLU(type=f32,ne_a=[128,2,2,2],v=0,swapped=0): OK\n"
             "  48/48 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n")

    def _run(self, op, stdout=_PASS, code=0):
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(gates.residency, "loader_env", return_value={}), \
             mock.patch.object(gates.subprocess, "run", return_value=mock.Mock(
                 returncode=code, stdout=stdout, stderr="")) as invoke:
            verdict = gates.op_correctness(Path("/build"), op=op, backend="CPU")
        argv = invoke.call_args.args[0]
        return verdict, argv[argv.index("-o") + 1]

    def test_unary_and_glu_select_their_sub_ops_and_plain_ops_pass_through(self):
        for op, names in gates.BACKEND_OPS_SELECTORS.items():
            _verdict, selector = self._run(op)
            self.assertEqual(tuple(selector.split(",")), names)
            self.assertNotIn(op, selector.split(","))
        for op in ("CLAMP", "SUM_ROWS", "TOPK_MOE", "MUL_MAT"):
            self.assertEqual(self._run(op)[1], op)

    def test_the_glu_suite_that_refused_every_candidate_now_passes_on_evidence(self):
        verdict, _selector = self._run("GLU")
        self.assertEqual((verdict.gate, verdict.passed), ("correctness", True))

    def test_an_empty_or_failing_expanded_suite_still_refuses(self):
        empty = ("Backend 1/1: CPU\n  0/0 tests passed\n  Backend CPU: OK\n"
                 "1/1 backends passed\nOK\n")
        verdict, _selector = self._run("GLU", empty)
        self.assertEqual((verdict.gate, verdict.passed), ("oracle_unavailable", False))
        self.assertIn("for GLU", verdict.reason)
        failed = ("Backend 1/1: CPU\n  SWIGLU(type=f32,ne_a=[5,7,11,13],v=1,swapped=1): FAIL\n"
                  "  47/48 tests passed\n  Backend CPU: FAIL\n0/1 backends passed\nFAIL\n")
        verdict, _selector = self._run("GLU", failed, code=1)
        self.assertEqual((verdict.gate, verdict.passed), ("correctness", False))
        self.assertIn("GLU failed on CPU", verdict.reason)

    def test_anchor_failing_sub_ops_are_named_not_selected(self):
        for op, names in gates.UNSELECTABLE_ON_ANCHOR.items():
            self.assertFalse(set(names) & set(gates.BACKEND_OPS_SELECTORS[op]), op)
        self.assertEqual(gates.UNSELECTABLE_ON_ANCHOR, {"UNARY": ("EXP", "EXPM1")})

    def test_no_route_hands_the_oracle_a_name_it_never_matches(self):
        for route in gates.CPU_SOURCE_ROUTES:
            for op in route.ops:
                self.assertFalse(set(gates.backend_ops_selector(op).split(","))
                                 & set(gates.BACKEND_OPS_SELECTORS), (route.route, op))

    def test_graph_sync_scope_covers_every_solo_eligible_op_and_the_moe_chain(self):
        route = next(r for r in gates.CPU_SOURCE_ROUTES if r.route == "cpu_graph_sync")
        # ggml_cpu_node_is_solo's switch at the DS41 anchor (ggml-cpu.c, ad8979943).
        solo = {"ADD", "SUB", "MUL", "DIV", "SCALE", "CLAMP", "FILL", "SQR", "SQRT", "LOG",
                "SIN", "COS", "SUM_ROWS", "UNARY", "GLU", "CPY", "CONT", "DUP"}
        self.assertLessEqual(solo, set(route.ops))
        self.assertIn("TOPK_MOE", route.ops)
        self.assertEqual(len(route.ops), len(set(route.ops)))
        # the DS41 graph ops are all still there: coverage only grew
        self.assertLessEqual(set(gates._DS41_GRAPH_OPS), set(route.ops))

    @unittest.skipUnless((_DS41_ANCHOR / "ggml/src/ggml.c").is_file(),
                         "DS41 anchor tree not present")
    def test_selectors_and_solo_set_match_the_ds41_anchor_sources(self):
        import re
        ggml = (_DS41_ANCHOR / "ggml/src/ggml.c").read_text(encoding="utf-8")
        start = ggml.index("const char * ggml_op_desc(")
        desc = ggml[start:ggml.index("\n}\n", start)]
        self.assertEqual(set(re.findall(r"t->op == GGML_OP_(\w+)", desc)),
                         set(gates.BACKEND_OPS_SELECTORS))
        for table, op in (("GGML_UNARY_OP_NAME", "UNARY"), ("GGML_GLU_OP_NAME", "GLU")):
            start = ggml.index(f"static const char * {table}[")
            names = tuple(re.findall(r'"(\w+)"', ggml[start:ggml.index("};", start)]))
            excluded = gates.UNSELECTABLE_ON_ANCHOR.get(op, ())
            self.assertEqual(tuple(name for name in names if name not in excluded),
                             gates.BACKEND_OPS_SELECTORS[op], table)
            self.assertLessEqual(set(excluded), set(names), table)
        cpu = (_DS41_ANCHOR / "ggml/src/ggml-cpu/ggml-cpu.c").read_text(encoding="utf-8")
        start = cpu.index("static bool ggml_cpu_node_is_solo(")
        solo = set(re.findall(r"case GGML_OP_(\w+):", cpu[start:cpu.index("\n}\n", start)]))
        route = next(r for r in gates.CPU_SOURCE_ROUTES if r.route == "cpu_graph_sync")
        self.assertTrue(solo)
        self.assertLessEqual(solo, set(route.ops))
        tbo = (_DS41_ANCHOR / "tests/test-backend-ops.cpp").read_text(encoding="utf-8")
        self.assertIn('return "TOPK_MOE";', tbo)
        self.assertIn("GATING_FUNC_SQRT_SOFTPLUS", tbo)


class AffectedOpAndIndependentReference(unittest.TestCase):
    def test_cpu_gdn_route_has_independent_reference(self):
        source = "// ggml_compute_forward_gated_delta_net\nold\nnew\n// ggml_compute_forward_next\n"
        patch = "@@ -2 +2 @@\n-old\n+new\n"
        self.assertEqual(gates.affected_op_scope(
            ("ggml/src/ggml-cpu/ops.cpp",),
            target_surface="ggml/src/ggml-cpu/ops.cpp",
            target_symbol="ggml_compute_forward_gated_delta_net_f32",
            source_text=source, patch_text=patch),
            ("GATED_DELTA_NET",))
        self.assertFalse(gates.affected_op_scope(
            ("ggml/src/ggml-cpu/ops.cpp",),
            target_surface="ggml/src/ggml-cpu/ops.cpp",
            target_symbol="ggml_compute_forward_gated_delta_net_f32",
            source_text=source, patch_text="@@ -4 +4 @@\n-old\n+new\n").passed)

    def test_cuda_gdn_and_vecdot_routes(self):
        self.assertEqual(gates.affected_op_scope(
            ("ggml/src/ggml-cuda/gated_delta_net.cu",),
            target_surface="ggml/src/ggml-cuda/gated_delta_net.cu",
            target_symbol="gated_delta_net_cuda"), ("GATED_DELTA_NET",))
        self.assertEqual(gates.affected_op_scope(
            ("ggml/src/ggml-cuda/vecdotq.cuh",),
            target_surface="ggml/src/ggml-cuda/vecdotq.cuh",
            target_symbol="vec_dot_q5_0_q8_1_impl"), ("MUL_MAT", "MUL_MAT_ID"))
        for path, symbol in (("ggml/src/ggml-cuda/mmvq.cu", "vec_dot_q4_K_q8_1"),
                             ("ggml/src/ggml-cuda/mmq.cu", "ggml_cuda_should_use_mmq")):
            self.assertEqual(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol=symbol),
                ("MUL_MAT", "MUL_MAT_ID"))

    def test_unknown_and_shared_source_refuse(self):
        for paths in (("ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp",),
                      ("ggml/src/ggml-cpu/ops.cpp", "ggml/src/ggml-cpu/ggml-cpu.c")):
            verdict = gates.affected_op_scope(
                paths, target_surface=paths[0],
                target_symbol="ggml_compute_forward_gated_delta_net_f32")
            self.assertIsInstance(verdict, gates.Verdict)
            self.assertFalse(verdict.passed)

    def test_cpu_quant_routes_refuse_before_build_without_edited_case_reference(self):
        for path, symbol in (("ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp", "iqk_mul_mat"),
                             ("ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "iqk_gemm"),
                             ("ggml/src/ggml-cpu/arch/x86/quants.c",
                              "ggml_vec_dot_q8_0_q8_0")):
            verdict = gates.affected_op_scope(
                (path,), target_surface=path, target_symbol=symbol)
            self.assertIsInstance(verdict, gates.Verdict)
            self.assertFalse(verdict.passed)

    def test_cpu_iqk_moe_rows_route_requires_actual_function_body_hunks(self):
        source = ('extern "C" IQK_API bool iqk_mul_mat_moe_rows(long n) {\n'
                  '    changed();\n}\n'
                  'extern "C" IQK_API bool iqk_moe_fused_up_gate(long n) {\n'
                  '    sibling();\n}\n')
        path = "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp"
        scope = gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="iqk_mul_mat_moe_rows",
            source_text=source, patch_text="@@ -2 +2 @@\n-old\n+changed();\n")
        self.assertEqual(scope, ("MUL_MAT_ID",))
        for patch in ("@@ -4 +4 @@\n-old\n+sibling();\n",
                      "@@ -1 +1 @@\n-old\n+extern foo\n", ""):
            refused = gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="iqk_mul_mat_moe_rows",
                source_text=source, patch_text=patch)
            self.assertIsInstance(refused, gates.Verdict)
            self.assertFalse(refused.passed)

    def test_cpu_fused_iqk_route_is_body_confined_and_op_specific(self):
        source = ('extern "C" IQK_API bool iqk_mul_mat_moe_rows(long n) {\n'
                  '    sibling();\n}\n'
                  'extern "C" IQK_API bool iqk_moe_fused_up_gate(long n) {\n'
                  '    changed();\n}\n'
                  '#if defined __x86_64__\n')
        path = "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp"
        scope = gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="iqk_moe_fused_up_gate",
            source_text=source, patch_text="@@ -5 +5 @@\n-old\n+changed();\n")
        self.assertEqual(scope, ("MUL_MAT_ID",))
        for hunk in ("@@ -2 +2 @@\n-old\n+sibling();\n",
                     "@@ -7 +7 @@\n-old\n+#if defined __x86_64__\n"):
            refused = gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="iqk_moe_fused_up_gate",
                source_text=source, patch_text=hunk)
            self.assertIsInstance(refused, gates.Verdict)
            self.assertFalse(refused.passed)

    def test_q45_dot_route_requires_pre_and_post_body_confinement(self):
        path = "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"
        source = ("struct Q4Bits_AVX2 {\n    shared();\n};\n"
                  "struct DequantizerQ4K_AVX2 final : Base {\n    q4();\n};\n"
                  "struct DequantizerQ5K_AVX2 final : Base {\n    q5();\n};\n"
                  "inline __m128i unpack_q4_scales(const uint8_t * x) {\n    scale();\n}\n"
                  "inline __m256i unpack_q4_scales_2(const uint8_t * x) {\n    scale2();\n}\n"
                  "template <typename Dequantizer, int nrc_y>\n"
                  "static void mul_mat_qX_K_q8_2_X4_T() {\n    dot();\n}\n"
                  "struct DequantizerQ6K_AVX2 final : Base {\n    q6();\n};\n"
                  "case GGML_TYPE_Q4_K:\n"
                  "IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_X4_T, DequantizerQ4K_AVX2, kernels)\n"
                  "case GGML_TYPE_Q5_K:\n"
                  "IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_X4_T, DequantizerQ5K_AVX2, kernels)\n")
        def patch_at(token):
            line = source.splitlines().index(token) + 1
            return f"@@ -{line} +{line} @@\n-old\n+new\n"
        for token in ("    q4();", "    q5();", "    scale();",
                      "    scale2();", "    dot();"):
            self.assertEqual(gates.affected_op_scope(
                (path,), target_surface=path,
                target_symbol="mul_mat_qX_K_q8_2_X4_T",
                source_text=source, pre_source_text=source,
                patch_text=patch_at(token)), ("MUL_MAT", "MUL_MAT_ID"))
        for token in ("    shared();", "    q6();",
                      "case GGML_TYPE_Q4_K:",
                      "static void mul_mat_qX_K_q8_2_X4_T() {"):
            self.assertFalse(gates.affected_op_scope(
                (path,), target_surface=path,
                target_symbol="mul_mat_qX_K_q8_2_X4_T",
                source_text=source, pre_source_text=source,
                patch_text=patch_at(token)).passed)
        self.assertFalse(gates.affected_op_scope(
            (path,), target_surface=path,
            target_symbol="mul_mat_qX_K_q8_2_X4_T",
            source_text=source, patch_text=patch_at("    dot();")).passed)
        changed_signature = source.replace("static void mul_mat_qX_K_q8_2_X4_T() {",
                                           "static void mul_mat_qX_K_q8_2_X4_T(int x) {")
        self.assertFalse(gates.affected_op_scope(
            (path,), target_surface=path,
            target_symbol="mul_mat_qX_K_q8_2_X4_T",
            source_text=changed_signature, pre_source_text=source,
            patch_text=patch_at("    dot();")).passed)

    def test_cpu_iqk_reference_distinguishes_wrong_and_unavailable(self):
        from autokernel.loop import iqk_witness
        for status, gate in (("pass", "reference_comparison"),
                             ("wrong", "reference_comparison"),
                             ("unavailable", "oracle_unavailable")):
            with mock.patch.object(iqk_witness, "check",
                    return_value=iqk_witness.Result(status, "reason", "detail")):
                verdict = gates.check_cpu_iqk_reference(
                    Path("/build"), Path("/source"), resolved_recipe=object(),
                    target_symbol="iqk_mul_mat_moe_rows")
            self.assertEqual(verdict.gate, gate)
            self.assertEqual(verdict.passed, status == "pass")
        with mock.patch.object(iqk_witness, "check_fused",
                               return_value=iqk_witness.Result("pass", "fused")) as fused:
            verdict = gates.check_cpu_iqk_reference(
                Path("/build"), Path("/source"), resolved_recipe=object(),
                target_symbol="iqk_moe_fused_up_gate")
        self.assertTrue(verdict.passed)
        fused.assert_called_once()
        with mock.patch.object(iqk_witness, "check_q45_dot",
                               return_value=iqk_witness.Result("pass", "dot")) as dot:
            verdict = gates.check_cpu_iqk_reference(
                Path("/build"), Path("/source"), resolved_recipe=object(),
                target_symbol="mul_mat_qX_K_q8_2_X4_T")
        self.assertTrue(verdict.passed)
        dot.assert_called_once()

    # --- widened CPU source routes (DS41 scope, 2026-09-26) ---------------------------

    _SGEMM = ("class tinyBLAS {\n"
              "    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {\n"
              "        other_class();\n    }\n};\n"
              "template <typename TA, typename TB, typename TC>\n"
              "class tinyBLAS_Q0_AVX {\n  private:\n"
              "    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {\n"
              "        pack();\n    }\n"
              "    template <int RN>\n"
              "    NOINLINE void gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {\n"
              "        tile4xN(); // braces in comments { do not count\n    }\n"
              "    template <int RM>\n"
              "    NOINLINE void gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {\n"
              "        tileMx4();\n    }\n"
              "    template <int RM, int RN>\n"
              "    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {\n"
              "        tile();\n    }\n"
              "    inline __m256i load(const block_q8_0 *b) {\n"
              "        return q8();\n    }\n};\n")

    @staticmethod
    def _hunk(source, token, replacement="+new\n"):
        line = source.splitlines().index(token) + 1
        return f"@@ -{line} +{line} @@\n-old\n{replacement}"

    def test_dense_q8_tinyblas_route_is_class_confined(self):
        path = "ggml/src/ggml-cpu/llamafile/sgemm.cpp"
        for token in ("        pack();", "        tile4xN(); // braces in comments { do not count",
                      "        tileMx4();", "        tile();"):
            self.assertEqual(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="gemm4xN",
                source_text=self._SGEMM, pre_source_text=self._SGEMM,
                patch_text=self._hunk(self._SGEMM, token)), ("MUL_MAT",))
        for token in ("        other_class();", "        return q8();",
                      "class tinyBLAS_Q0_AVX {"):
            refused = gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="tinyBLAS_Q0_AVX",
                source_text=self._SGEMM, pre_source_text=self._SGEMM,
                patch_text=self._hunk(self._SGEMM, token))
            self.assertFalse(refused.passed)
            self.assertIn("dense_q8_tinyblas route refused", refused.reason)
        qs = gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="gemm4xN",
            source_text=self._SGEMM, pre_source_text=self._SGEMM,
            patch_text=self._hunk(self._SGEMM, "        tile4xN(); // braces in comments { do not count",
                                  "+        auto q = A[0].qs[3];\n"))
        self.assertFalse(qs.passed)
        self.assertIn("forbidden pattern", qs.reason)

    def test_planner_written_symbols_resolve_the_route_but_not_the_boundary(self):
        # DS41 run 10h: the planner named the symbol as it reads in source, and the
        # exact lookup refused an in-body patch as "affected native op ... unresolved".
        path = "ggml/src/ggml-cpu/llamafile/sgemm.cpp"
        written = "tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN (template body, RN=1..4)"
        self.assertEqual(gates.cpu_source_route(path, written).route, "dense_q8_tinyblas")
        self.assertEqual(gates.affected_op_scope(
            (path,), target_surface=path, target_symbol=written,
            source_text=self._SGEMM, pre_source_text=self._SGEMM,
            patch_text=self._hunk(self._SGEMM, "        tile4xN(); // braces in comments { do not count")),
            ("MUL_MAT",))
        refused = gates.affected_op_scope(
            (path,), target_surface=path, target_symbol=written,
            source_text=self._SGEMM, pre_source_text=self._SGEMM,
            patch_text=self._hunk(self._SGEMM, "        return q8();"))
        self.assertFalse(refused.passed)
        self.assertIn("dense_q8_tinyblas route refused", refused.reason)
        # Same path, two routes: the member name picks the right one.
        dispatch = "ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp"
        self.assertEqual(gates.cpu_source_route(dispatch, "ggml_iqk_try_mul_mat_id (N>1 path)").route,
                         "iqk_mmid_dispatch")
        self.assertEqual(gates.cpu_source_route(dispatch, "ggml_iqk_try_mul_mat").route,
                         "iqk_dense_dispatch")
        self.assertIsNone(gates.cpu_source_route(path, "tinyBLAS_Q4_0::gemv (other class)"))
        header = self._SGEMM.replace(
            "    NOINLINE void gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {",
            "    NOINLINE void gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n, int x) {")
        self.assertFalse(gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="gemm4xN",
            source_text=header, pre_source_text=self._SGEMM,
            patch_text=self._hunk(self._SGEMM, "        pack();")).passed)
        # an unadmitted symbol on the same file keeps the generic refusal
        self.assertFalse(gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="llamafile_sgemm",
            source_text=self._SGEMM, pre_source_text=self._SGEMM,
            patch_text=self._hunk(self._SGEMM, "        pack();")).passed)

    _SGEMM_FLOAT = ("template <int KN, typename D, typename V, typename TA, typename TB, typename TC>\n"
                    "class tinyBLAS {\n  public:\n"
                    "    bool matmul(int64_t m, int64_t n) {\n"
                    "        if (k % KN != 0)\n            return false;\n"
                    "        mnpack<4, 6, 2>(m, n, SIZE_N, 12); // { brace in a comment\n"
                    "        return true;\n    }\n"
                    "  private:\n"
                    "    template <int RM, int RN, int BM>\n"
                    "    inline void mnpack(int64_t m, int64_t n, int64_t SIZE_N, int64_t BN) {\n"
                    "        descend();\n    }\n"
                    "    template <int RM, int RN>\n"
                    "    inline void gemm_bloc(int64_t ii, int64_t jj) {\n"
                    "        accumulate();\n    }\n"
                    "    template <int RM, int RN, int BM>\n"
                    "    NOINLINE void gemm(int64_t m, int64_t n, int64_t BN) {\n"
                    "        barrier_and_jobs();\n    }\n};\n"
                    "template <typename TA, typename TB, typename TC>\n"
                    "class tinyBLAS_Q0_AVX {\n"
                    "    bool matmul(int64_t m, int64_t n) {\n"
                    "        q0_plan();\n    }\n};\n")

    def test_float_tinyblas_plan_route_is_matmul_body_only(self):
        path = "ggml/src/ggml-cpu/llamafile/sgemm.cpp"
        src = self._SGEMM_FLOAT
        plan = "        mnpack<4, 6, 2>(m, n, SIZE_N, 12); // { brace in a comment"
        for symbol in ("matmul", "tinyBLAS",
                       "tinyBLAS<16, float __vector(16), float __vector(16), unsigned short, "
                       "unsigned short, float>::matmul (small-M plan)"):
            self.assertEqual(gates.cpu_source_route(path, symbol).route, "float_tinyblas_plan")
            self.assertEqual(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol=symbol,
                source_text=src, pre_source_text=src,
                patch_text=self._hunk(src, plan, "+        mnpack<1, 6, 1>(m, n, 2, 1);\n")),
                ("MUL_MAT",))
        # numerics, the barrier-bearing gemm, the other class: all outside the boundary
        for token in ("        descend();", "        accumulate();",
                      "        barrier_and_jobs();", "        q0_plan();",
                      "class tinyBLAS {"):
            refused = gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="tinyBLAS::matmul",
                source_text=src, pre_source_text=src, patch_text=self._hunk(src, token))
            self.assertFalse(refused.passed, token)
            self.assertIn("float_tinyblas_plan route refused", refused.reason)
        for added in ("+        C[0] = 0;\n", "+        gemm_bloc<1, 3>(0, 0);\n",
                      "+        auto v = madd(a, b, c);\n", "+        _mm512_setzero_ps();\n"):
            refused = gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="matmul",
                source_text=src, pre_source_text=src, patch_text=self._hunk(src, plan, added))
            self.assertFalse(refused.passed, added)
            self.assertIn("forbidden pattern", refused.reason)
        header = src.replace("    bool matmul(int64_t m, int64_t n) {\n        if",
                             "    bool matmul(int64_t m, int64_t n, int t) {\n        if")
        self.assertFalse(gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="matmul",
            source_text=header, pre_source_text=src,
            patch_text=self._hunk(src, plan)).passed)

    def test_class_qualified_symbols_pick_their_own_tinyblas_route(self):
        path = "ggml/src/ggml-cpu/llamafile/sgemm.cpp"
        # the Q0 route keeps its names, qualified or bare
        for symbol in ("gemm", "gemm4xN", "tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm",
                       "tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::mnpack(long, long, long, long)"):
            self.assertEqual(gates.cpu_source_route(path, symbol).route, "dense_q8_tinyblas",
                             symbol)
        # a float-class member the plan route does not admit falls back by bare name and
        # is then refused by the Q0 class boundary (or resolves to nothing): fail-closed
        written = ("tinyBLAS<16, float __vector(16), float __vector(16), unsigned short, "
                   "unsigned short, float>::gemm<4, 3, 2>(long, long, long)")
        self.assertEqual(gates.cpu_source_route(path, written).route, "dense_q8_tinyblas")
        demangled = "void (anonymous namespace)::" + written + " [clone .constprop.0]"
        for symbol in (written, demangled):
            refused = gates.affected_op_scope(
                (path,), target_surface=path, target_symbol=symbol,
                source_text=self._SGEMM_FLOAT, pre_source_text=self._SGEMM_FLOAT,
                patch_text=self._hunk(self._SGEMM_FLOAT, "        barrier_and_jobs();"))
            self.assertFalse(refused.passed, symbol)

    _DISPATCH = ("namespace {\n"
                 "inline bool iqk_q8_0_enabled() {\n    return off();\n}\n"
                 "}\n"
                 'extern "C" bool ggml_iqk_try_mul_mat(const struct ggml_compute_params * params, '
                 "struct ggml_tensor * dst) {\n    dense();\n}\n"
                 'extern "C" bool ggml_iqk_try_mul_mat_id(const struct ggml_compute_params * params, '
                 "struct ggml_tensor * dst) {\n    moe();\n"
                 '    fprintf(stderr, "{ not a brace");\n}\n'
                 "#else  // iqk not implemented / disabled\n"
                 'extern "C" bool ggml_iqk_try_mul_mat_id(const struct ggml_compute_params * params, '
                 "struct ggml_tensor * dst) {\n    stub();\n}\n#endif\n")

    def test_iqk_dispatch_routes_split_by_symbol_and_skip_the_stub(self):
        path = "ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp"
        src = self._DISPATCH
        self.assertEqual(gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="ggml_iqk_try_mul_mat_id",
            source_text=src, pre_source_text=src,
            patch_text=self._hunk(src, "    moe();")), ("MUL_MAT_ID",))
        for token in ("    dense();", "    return off();"):
            self.assertEqual(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="ggml_iqk_try_mul_mat",
                source_text=src, pre_source_text=src,
                patch_text=self._hunk(src, token)), ("MUL_MAT", "MUL_MAT_ID"))
        for symbol, token in (("ggml_iqk_try_mul_mat_id", "    stub();"),
                              ("ggml_iqk_try_mul_mat_id", "    dense();"),
                              ("ggml_iqk_try_mul_mat", "    moe();")):
            self.assertFalse(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol=symbol,
                source_text=src, pre_source_text=src,
                patch_text=self._hunk(src, token)).passed)
        no_fence = src.replace("#else  // iqk not implemented / disabled\n", "")
        self.assertFalse(gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="ggml_iqk_try_mul_mat_id",
            source_text=no_fence, pre_source_text=no_fence,
            patch_text=self._hunk(no_fence, "    moe();")).passed)

    def test_cpu_graph_sync_route_scope_is_every_ds41_op(self):
        path = "ggml/src/ggml-cpu/ggml-cpu.c"
        src = ("void ggml_barrier(struct ggml_threadpool * tp) {\n    barrier();\n}\n"
               "static bool ggml_cpu_node_is_solo(const struct ggml_tensor * node) {\n"
               "    solo();\n}\n"
               "static int ggml_cpu_try_fuse_ops(\n        int i,\n"
               "        const struct ggml_cplan * cplan) {\n    fuse();\n}\n"
               "static thread_ret_t ggml_graph_compute_thread(void * data) {\n    walk();\n}\n"
               "static void ggml_compute_forward_add(void) {\n    kernel();\n}\n")
        for token in ("    barrier();", "    solo();", "    fuse();", "    walk();"):
            self.assertEqual(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="ggml_barrier",
                source_text=src, pre_source_text=src,
                patch_text=self._hunk(src, token)), gates._DS41_SYNC_OPS)
        for token in ("    kernel();", "        int i,"):
            self.assertFalse(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="ggml_barrier",
                source_text=src, pre_source_text=src,
                patch_text=self._hunk(src, token)).passed)

    _OPS = ("static void ggml_compute_forward_norm_f32(\n"
            "        const ggml_compute_params * params,\n        ggml_tensor * dst) {\n"
            "    float sum = 0.0;\n    layer_norm();\n}\n"
            "template <ggml_rms_norm_fuse_op FUSE_OP>\n"
            "static void ggml_compute_forward_rms_norm_f32(\n"
            "        const ggml_compute_params * params,\n"
            "        ggml_tensor * dst_rms_norm,\n"
            "        ggml_tensor * dst_fused = nullptr) {\n"
            "    const int nth = params->nth;\n"
            "    for (int64_t i01 = ith; i01 < ne01; i01 += nth) {\n"
            "        ggml_float sum = 0.0;\n"
            "        for (int64_t i00 = 0; i00 < ne00; i00++) {\n"
            "            sum += (ggml_float)(x[i00] * x[i00]);\n        }\n"
            "        const float mean  = sum/ne00;\n"
            "        const float scale = 1.0f/sqrtf(mean + eps);\n"
            "        if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {\n"
            "            for (int64_t i00 = 0; i00 < ne00; i00++) {\n"
            "                y[i00] = x[i00] * scale * w[i00];\n            }\n"
            "        } else {\n"
            "            memcpy(y, x, ne00 * sizeof(float));\n"
            "            ggml_vec_scale_f32(ne00, y, scale);\n        }\n    }\n}\n"
            "void ggml_compute_forward_rms_norm(\n"
            "        const ggml_compute_params * params,\n        ggml_tensor * dst) {\n"
            "    dispatch_plain();\n}\n"
            "void ggml_compute_forward_rms_norm_mul_fused(\n"
            "        const ggml_compute_params * params,\n"
            "        ggml_tensor * dst_rms_norm,\n        ggml_tensor * dst_mul) {\n"
            "    dispatch_fused();\n}\n"
            "static void ggml_compute_forward_rms_norm_back_f32(\n"
            "        const ggml_compute_params * params,\n        ggml_tensor * dst) {\n"
            "    backward();\n}\n"
            "static void ggml_compute_forward_group_norm_f32(\n"
            "        const ggml_compute_params * params,\n        ggml_tensor * dst) {\n"
            "    group();\n}\n")

    def test_cpu_norm_rowsplit_route_is_the_rms_norm_body_only(self):
        path = "ggml/src/ggml-cpu/ops.cpp"
        src = self._OPS
        inner = "        const float mean  = sum/ne00;"
        ops = ("RMS_NORM", "RMS_NORM_MUL_ADD")
        for symbol in ("ggml_compute_forward_rms_norm_f32",
                       "ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>",
                       "ggml_compute_forward_rms_norm", "ggml_compute_forward_rms_norm_mul_fused",
                       "ggml_compute_forward_rms_norm_f32 (narrow rows: split within the row)"):
            self.assertEqual(gates.cpu_source_route(path, symbol).route, "cpu_norm_rowsplit")
            self.assertEqual(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol=symbol,
                source_text=src, pre_source_text=src,
                patch_text=self._hunk(src, inner, "+" + inner + "\n")), ops)
        for symbol in ("ggml_compute_forward_norm_f32", "ggml_compute_forward_group_norm_f32",
                       "ggml_compute_forward_rms_norm_back_f32"):
            self.assertIsNone(gates.cpu_source_route(path, symbol), symbol)
        # the other norms, the dispatchers and the header stay outside the boundary
        for token in ("    layer_norm();", "    dispatch_plain();", "    dispatch_fused();",
                      "    backward();", "    group();",
                      "        ggml_tensor * dst_fused = nullptr) {",
                      "template <ggml_rms_norm_fuse_op FUSE_OP>"):
            refused = gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="ggml_compute_forward_rms_norm",
                source_text=src, pre_source_text=src, patch_text=self._hunk(src, token))
            self.assertFalse(refused.passed, token)
            self.assertIn("cpu_norm_rowsplit route refused", refused.reason)
        header = src.replace("        ggml_tensor * dst_fused = nullptr) {",
                             "        ggml_tensor * dst_fused = nullptr, int split = 0) {")
        self.assertFalse(gates.affected_op_scope(
            (path,), target_surface=path, target_symbol="ggml_compute_forward_rms_norm_f32",
            source_text=header, pre_source_text=src,
            patch_text=self._hunk(src, inner)).passed)

    def test_cpu_norm_rowsplit_holds_heads_numerics_textually(self):
        path = "ggml/src/ggml-cpu/ops.cpp"
        src = self._OPS
        inner = "        const float mean  = sum/ne00;"
        admitted = (
            # HEAD's numerics, moved/re-indented, any loop index
            "ggml_float sum = 0.0;", "sum += (ggml_float)(x[i00] * x[i00]);",
            "sum += (ggml_float)(x[j] * x[j]);", "const float mean  = sum/ne00;",
            "const float scale = 1.0f/sqrtf(mean + eps);", "assert(scale > 0.0f);",
            "y[i00] = x[i00] * scale * w[i00];", "y[c] = x[c] * scale * w[c]; // own segment",
            "ggml_vec_scale_f32(c1 - c0, y + c0, scale);", "float eps;",
            "memcpy(&eps, dst_rms_norm->op_params, sizeof(float));",
            "GGML_ASSERT(eps >= 0.0f);",
            # the split itself
            "const int64_t nr = ne01*ne02*ne03;",
            "if (nth > 1 && nr < nth && dst->data != src0->data && ggml_rowcol_split_enabled()) {",
            "const ggml_rowcol_split split = get_rowcol_split(params, nr, ne00, sizeof(float));",
            "split.unpack(t, ne00, ir, c0, c1);",
            "memcpy(y + c0, x + c0, (c1 - c0) * sizeof(float));",
            "ggml_vec_cpy_f32(c1 - c0, y + c0, x + c0);",
            "for (int64_t i00 = c0; i00 < c1; i00++) {",
            "// every thread recomputes the full row sum, then scales its segment",
            "/* the scale is HEAD's */")
        for line in admitted:
            self.assertEqual(gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="ggml_compute_forward_rms_norm_f32",
                source_text=src, pre_source_text=src,
                patch_text=self._hunk(src, inner, "+            " + line + "\n")),
                ("RMS_NORM", "RMS_NORM_MUL_ADD"), line)
        refused = (
            "float sum = 0.0f;", "double sum = 0.0;", "ggml_float sum = 0;",
            "sum += x[i00] * x[i00];", "sum += (ggml_float)x[i00] * (ggml_float)x[i00];",
            "acc += (ggml_float)(x[i00] * x[i00]);", "partial[ith] = sum;",
            "const float mean  = total/ne00;", "const float scale = 1.0f/sqrtf(mean) + eps;",
            "const float inv = 1.0f/sqrtf(m + e);", "y[i00] = x[i00] * (scale * w[i00]);",
            "y[i00] = x[i00] / scale;", "ggml_vec_scale_f32(n, y, scale * 2.0f);",
            "ggml_vec_dot_f32(ne00, &s, 0, x, 0, x, 0, 1);", "ggml_vec_norm_f32(ne00, &s, x);",
            "__m512 v = _mm512_loadu_ps(x + i00);", "GGML_F32_VEC v = GGML_F32_VEC_LOAD(x);",
            "#pragma omp simd reduction(+:acc)", "ggml_barrier(params->threadpool);",
            "const float r = rsqrtf(q);", "acc = fmaf(v, v, acc);",
            "auto s = std::accumulate(x, x + ne00, 0.0);",
            "/* sum */ acc = 1;", "q += x[i] * x[i];")
        rowsplit = gates.cpu_route_named("cpu_norm_rowsplit")
        for line in refused:
            patch = self._hunk(src, inner, "+            " + line + "\n")
            self.assertIn("forbidden pattern",
                          gates._cpu_route_scope_refusal(rowsplit, src, src, patch), line)
            # 2026-10-03: the bit-exact route still refuses every one of these; the
            # arithmetic-changing ones now fall through to cpu_norm_numerics (tolerance
            # gate). An in-op barrier or an OpenMP pragma is refused by both routes.
            route, refusal = gates.admit_cpu_route(
                path, "ggml_compute_forward_rms_norm_f32", src, src, patch)
            if "ggml_barrier" in line or "#pragma omp" in line:
                self.assertIsNotNone(refusal, line)
                verdict = gates.affected_op_scope(
                    (path,), target_surface=path,
                    target_symbol="ggml_compute_forward_rms_norm_f32",
                    source_text=src, pre_source_text=src, patch_text=patch)
                self.assertFalse(verdict.passed, line)
                self.assertIn("forbidden pattern", verdict.reason)
            else:
                self.assertEqual((route.route, refusal), ("cpu_norm_numerics", None), line)

    def test_ops_cpp_route_leaves_the_gated_delta_net_rule_alone(self):
        path = "ggml/src/ggml-cpu/ops.cpp"
        self.assertIsNone(gates.cpu_source_route(path, "ggml_compute_forward_gated_delta_net_f32"))
        # run.py runs the widened-route reference only for an edit a route governs:
        # ops.cpp now hosts a route, and a GDN edit there must not acquire (and fail) it.
        source = (Path(__file__).resolve().parent / "run.py").read_text(encoding="utf-8")
        self.assertIn("route_edit = len(changed) == 1 and changed[0] in route_paths and \\\n"
                      "                gates.cpu_source_route(changed[0], "
                      "hypothesis.target_symbol) is not None", source)
        self.assertEqual(source.count("if cpu_launch and route_edit:"), 2)
        self.assertNotIn("changed[0] in route_paths and not cpu_launch", source)
        self.assertNotIn("and len(changed) == 1 and changed[0] in route_paths:", source)

    @unittest.skipUnless(Path("/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925"
                              "/ggml/src/ggml-cpu/ops.cpp").is_file(),
                         "DS41 anchor tree not present")
    def test_cpu_norm_rowsplit_admits_the_split_on_the_ds41_anchor(self):
        import difflib
        path = "ggml/src/ggml-cpu/ops.cpp"
        head = (Path("/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925") /
                path).read_text(encoding="utf-8")
        start = head.index("    // TODO: optimize\n",
                           head.index("static void ggml_compute_forward_rms_norm_f32("))
        split = (
            "    const int64_t nr = ne01*ne02*ne03;\n"
            "    if (nth > 1 && nr < nth && dst->data != src0->data && ggml_rowcol_split_enabled()) {\n"
            "        const ggml_rowcol_split split = get_rowcol_split(params, nr, ne00, sizeof(float));\n"
            "        for (int64_t t = split.t0; t < split.t1; ++t) {\n"
            "            int64_t ir, c0, c1;\n"
            "            split.unpack(t, ne00, ir, c0, c1);\n"
            "            const int64_t i01 = ir % ne01;\n"
            "            const int64_t i02 = (ir / ne01) % ne02;\n"
            "            const int64_t i03 = ir / (ne01*ne02);\n"
            "            const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);\n"
            "            ggml_float sum = 0.0;\n"
            "            for (int64_t i00 = 0; i00 < ne00; i00++) {\n"
            "                sum += (ggml_float)(x[i00] * x[i00]);\n"
            "            }\n"
            "            const float mean  = sum/ne00;\n"
            "            const float scale = 1.0f/sqrtf(mean + eps);\n"
            "            float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);\n"
            "            if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {\n"
            "                const float * w = (float *) ((char *) src1->data + (i01 % ne11)*nb11 + (i02 % ne12)*nb12 + (i03 % ne13)*nb13);\n"
            "                for (int64_t i00 = c0; i00 < c1; i00++) {\n"
            "                    y[i00] = x[i00] * scale * w[i00];\n"
            "                }\n"
            "            } else {\n"
            "                memcpy(y + c0, x + c0, (c1 - c0) * sizeof(float));\n"
            "                ggml_vec_scale_f32(c1 - c0, y + c0, scale);\n"
            "            }\n"
            "        }\n"
            "        return;\n"
            "    }\n")

        def verdict(candidate):
            patch = "".join(difflib.unified_diff(head.splitlines(True),
                                                 candidate.splitlines(True), n=0))
            return gates.affected_op_scope(
                (path,), target_surface=path, target_symbol="ggml_compute_forward_rms_norm_f32",
                source_text=candidate, pre_source_text=head, patch_text=patch)

        good = head[:start] + split + head[start:]
        self.assertEqual(verdict(good), ("RMS_NORM", "RMS_NORM_MUL_ADD"))

        def admitted(candidate):
            patch = "".join(difflib.unified_diff(head.splitlines(True),
                                                 candidate.splitlines(True), n=0))
            return gates.admit_cpu_route(path, "ggml_compute_forward_rms_norm_f32",
                                         candidate, head, patch)

        self.assertEqual(admitted(good)[0].route, "cpu_norm_rowsplit")
        for old, new in (("            ggml_float sum = 0.0;\n", "            float sum = 0.0f;\n"),
                         ("y[i00] = x[i00] * scale * w[i00];",
                          "y[i00] = x[i00] * (scale * w[i00]);")):
            candidate = head[:start] + split.replace(old, new) + head[start:]
            rowsplit = gates._cpu_route_scope_refusal(
                gates.cpu_route_named("cpu_norm_rowsplit"), candidate, head,
                "".join(difflib.unified_diff(head.splitlines(True),
                                             candidate.splitlines(True), n=0)))
            self.assertIn("forbidden pattern", rowsplit, new)
            # 2026-10-03: an arithmetic change is no longer refused outright; it falls
            # through to cpu_norm_numerics and its tolerance gate.
            self.assertEqual(admitted(candidate), (gates.cpu_route_named("cpu_norm_numerics"),
                                                   None), new)
            self.assertEqual(verdict(candidate), ("RMS_NORM", "RMS_NORM_MUL_ADD"))
        # the same split placed in NORM's body is outside the route
        norm = head.index("static void ggml_compute_forward_norm_f32(")
        norm_body = head.index("    for (int64_t i03 = 0; i03 < ne03; i03++) {", norm)
        self.assertFalse(verdict(head[:norm_body] + "    const int64_t nr = ne01;\n" +
                                 head[norm_body:]).passed)

    @unittest.skipUnless(Path("/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925"
                              "/ggml/src/ggml-cpu/ggml-cpu.c").is_file(),
                         "DS41 anchor tree not present")
    def test_route_markers_resolve_once_on_the_ds41_anchor(self):
        root = Path("/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925")
        for route in gates.CPU_SOURCE_ROUTES:
            text = (root / route.path).read_text(encoding="utf-8")
            bounds = gates._cpu_route_bounds(text, "HEAD", route)
            self.assertNotIsInstance(bounds, str, f"{route.route}: {bounds}")
            self.assertEqual([label for label, _a, _b in bounds[0]],
                             [label for label, _prefix in route.bodies])

    def test_cpu_route_reference_dispatches_to_the_route_witness(self):
        from autokernel.loop import cpu_route_witness, iqk_witness
        for status, gate in (("pass", "reference_comparison"),
                             ("wrong", "reference_comparison"),
                             ("unavailable", "oracle_unavailable")):
            with mock.patch.object(cpu_route_witness, "check",
                    return_value=iqk_witness.Result(status, "reason", "detail")) as check:
                verdict = gates.check_cpu_route_reference(
                    Path("/build"), Path("/source"), resolved_recipe=object(),
                    path="ggml/src/ggml-cpu/llamafile/sgemm.cpp", target_symbol="gemm4xN")
            self.assertEqual((verdict.gate, verdict.passed), (gate, status == "pass"))
            self.assertEqual(check.call_args.kwargs["route"], "dense_q8_tinyblas")
            with mock.patch.object(cpu_route_witness, "check",
                    return_value=iqk_witness.Result(status, "reason", "detail")) as check:
                gates.check_cpu_route_reference(
                    Path("/build"), Path("/source"), resolved_recipe=object(),
                    path="ggml/src/ggml-cpu/llamafile/sgemm.cpp", target_symbol="tinyBLAS::matmul")
            self.assertEqual(check.call_args.kwargs["route"], "float_tinyblas_plan")
            with mock.patch.object(cpu_route_witness, "check",
                    return_value=iqk_witness.Result(status, "reason", "detail")) as check:
                gates.check_cpu_route_reference(
                    Path("/build"), Path("/source"), resolved_recipe=object(),
                    path="ggml/src/ggml-cpu/ops.cpp",
                    target_symbol="ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>")
            self.assertEqual(check.call_args.kwargs["route"], "cpu_norm_rowsplit")
        unknown = gates.check_cpu_route_reference(
            Path("/build"), Path("/source"), resolved_recipe=object(),
            path="ggml/src/ggml-cpu/ops.cpp", target_symbol="ggml_compute_forward_concat")
        self.assertEqual((unknown.gate, unknown.passed), ("oracle_unavailable", False))

    def test_route_witness_requires_reference_then_candidate_hit_in_candidate_dso(self):
        from autokernel.loop import cpu_route_witness
        witness = cpu_route_witness.WITNESSES["iqk_mmid_dispatch"]
        dso = Path("/build/bin/libggml-cpu.so.0")
        ok = [{"schema": "epyc.autokernel.cpu_route_hit.v1", "status": "hit",
               "role": "independent_reference", "symbol": "ggml_backend_cpu_set_use_ref",
               "dso": str(dso.resolve())},
              {"schema": "epyc.autokernel.cpu_route_hit.v1", "status": "hit",
               "role": "candidate_route", "symbol": "ggml_iqk_try_mul_mat_id",
               "dso": str(dso.resolve())}]
        output = (f"  MUL_MAT_ID({witness.case}): OK\n  1/1 tests passed\n"
                  "  Backend CPU: OK\n[iqk] ACTIVE: MoE mul_mat_id via ik kernels "
                  "(type=12 activation=41 n_as=4)\n[Inferior 1 (process 1) exited normally]\n")
        self.assertEqual(cpu_route_witness.assess_case(witness, ok, output, 0, dso).status,
                         "pass")
        self.assertEqual(cpu_route_witness.assess_case(witness, ok[::-1], output, 0,
                                                       dso).status, "unavailable")
        other_dso = [dict(ok[0]), dict(ok[1], dso="/prod/libggml-cpu.so.0")]
        self.assertEqual(cpu_route_witness.assess_case(witness, other_dso, output, 0,
                                                       dso).status, "unavailable")
        silent = output.replace("[iqk] ACTIVE: MoE", "[iqk] quiet: MoE")
        self.assertEqual(cpu_route_witness.assess_case(witness, ok, silent, 0, dso).status,
                         "unavailable")
        two_cases = output.replace("1/1 tests passed", "2/2 tests passed")
        self.assertEqual(cpu_route_witness.assess_case(witness, ok, two_cases, 0,
                                                       dso).status, "unavailable")

    def test_every_widened_route_has_a_reviewed_witness(self):
        from autokernel.loop import cpu_quant_reference, cpu_route_witness
        for route in gates.CPU_SOURCE_ROUTES:
            witness = cpu_route_witness.WITNESSES[route.route]
            self.assertTrue(set(witness.ops) >= set(route.ops) or
                            route.route == "cpu_graph_sync", route.route)
            self.assertTrue(set(witness.quants) <= set(cpu_quant_reference.QUANTS) |
                            set(cpu_quant_reference.FLOAT_TYPES), route.route)
        float_route = cpu_route_witness.WITNESSES["float_tinyblas_plan"]
        # every float type the tinyBLAS class is instantiated for on AVX512 is checked
        self.assertEqual(set(float_route.quants), {"F16", "BF16", "F32"})
        import re as _re
        hit = ("void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), "
               "unsigned short, unsigned short, float>::gemm<4, 6, 2>(long, long, long) "
               "[clone .constprop.0]")
        self.assertIsNotNone(_re.search(float_route.symbol_pattern, hit))
        self.assertIsNone(_re.search(float_route.symbol_pattern, hit.replace(
            "unsigned short, unsigned short", "float, float")))
        norm = cpu_route_witness.WITNESSES["cpu_norm_rowsplit"]
        self.assertEqual(norm.reference, "rms_norm")
        self.assertEqual([(w.op, w.breakpoint) for w in (norm, *norm.also)],
                         [("RMS_NORM", ("break", "ggml_compute_forward_rms_norm")),
                          ("RMS_NORM_MUL_ADD",
                           ("break", "ggml_compute_forward_rms_norm_mul_fused"))])
        self.assertIsNotNone(_re.search(norm.symbol_pattern, "ggml_compute_forward_rms_norm"))
        for other in ("ggml_compute_forward_rms_norm_mul_fused",
                      "ggml_compute_forward_rms_norm_back"):
            self.assertIsNone(_re.search(norm.symbol_pattern, other))
        self.assertIsNone(_re.search(norm.also[0].symbol_pattern,
                                     "ggml_compute_forward_rms_norm"))
        # the exact vars() strings of the anchor's registered cases (VARS_TO_STR5)
        self.assertEqual(norm.case, "type=f32,ne=[64,5,4,3],v=0,eps=0.000001,inplace=0")
        self.assertEqual(norm.also[0].case,
                         "type=f32,ne=[64,5,4,3],eps=0.000001,broadcast=0,multi_add=0")

    def test_rms_norm_witness_needs_both_entry_hits_then_the_norm_reference(self):
        from autokernel.loop import cpu_norm_reference, cpu_quant_reference, cpu_route_witness
        from autokernel.loop.iqk_witness import Result
        recipe = SimpleNamespace(backend="cpu", launch_env={"GGML_IQK": "1"},
                                 topology_prefix=("taskset", "-c", "96-103"),
                                 template=None, port=None, validate_launch=lambda *a: None)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "ggml/src/ggml-cpu").mkdir(parents=True)
            (Path(root) / "ggml/src/ggml-cpu/ops.cpp").write_text("", encoding="utf-8")

            def run(hits, norm_status="pass"):
                with mock.patch.object(cpu_route_witness, "_engagement",
                                       side_effect=hits) as engage, \
                     mock.patch.object(cpu_norm_reference, "check_rms_norm_suite",
                                       return_value=cpu_norm_reference.NormResult(
                                           norm_status, "norm reason", "norm detail")) as norm, \
                     mock.patch.object(cpu_quant_reference, "check_cpu_quant_suite") as quant:
                    result = cpu_route_witness.check(
                        Path("/build"), resolved_recipe=recipe, source_root=Path(root),
                        route="cpu_norm_rowsplit", source_path="ggml/src/ggml-cpu/ops.cpp")
                quant.assert_not_called()
                return result, engage, norm

            ok = Result("pass", "hit", "detail")
            result, engage, norm = run([ok, ok])
            self.assertEqual(result.status, "pass")
            self.assertEqual([call.args[1].op for call in engage.call_args_list],
                             ["RMS_NORM", "RMS_NORM_MUL_ADD"])
            self.assertEqual(norm.call_args.kwargs["topology_prefix"],
                             ("taskset", "-c", "96-103"))
            # the fused entry not hit: refused before any numerics
            result, _engage, norm = run([ok, Result("unavailable", "no fused hit")])
            self.assertEqual(result.status, "unavailable")
            norm.assert_not_called()
            result, _engage, _norm = run([ok, ok], norm_status="wrong")
            self.assertEqual((result.status, result.reason), ("wrong", "norm reason"))

    def test_every_widened_route_reaches_the_actors(self):
        program = (Path(__file__).resolve().parent / "program.md").read_text()
        for route in gates.CPU_SOURCE_ROUTES:
            self.assertIn(f"`{route.route}`", program)
            self.assertIn(route.path, program)

    def test_wrong_vs_unavailable_reference_are_distinct(self):
        for status, gate in (("pass", "reference_comparison"),
                             ("wrong", "reference_comparison"),
                             ("unavailable", "oracle_unavailable")):
            with mock.patch.object(gdn_reference, "check_cpu_gdn",
                    return_value=gdn_reference.GDNResult(status, "reason", "detail")):
                verdict = gates.check_cpu_gdn_reference(Path("/build"), Path("/source"))
            self.assertEqual(verdict.gate, gate)
            self.assertEqual(verdict.passed, status == "pass")

    def test_default_invocation_does_not_use_optional_metric_flags(self):
        """Legacy instrument compatibility remains the default route."""
        output = ("Testing 1 devices\n\nBackend 1/1: ROCm0\n"
                  "  MUL_MAT(type=f32): OK\n  1/1 tests passed\n"
                  "  Backend ROCm0: OK\n1/1 backends passed\nOK\n")
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(gates.residency, "loader_env", return_value={}), \
             mock.patch.object(gates.subprocess, "run", return_value=mock.Mock(
                 returncode=0, stdout=output, stderr="")) as invoke:
            self.assertTrue(gates.op_correctness(Path("/build")).passed)
        argv = invoke.call_args.args[0]
        self.assertNotIn("--suite-seed", argv)
        self.assertNotIn("--autokernel-properties", argv)

    def test_the_invocation_is_the_one_proven_to_work_on_the_anchor(self):
        """The original op-selection argv remains the default form."""
        import inspect
        body = inspect.getsource(gates.op_correctness).split('"""', 2)[-1]
        for token in ('"test"', '"-o", backend_ops_selector(op)', '"-b", backend', '"-j", "1"'):
            self.assertIn(token, body, token)


class TheAnchorMustAdvanceWithTheChampion(unittest.TestCase):
    """Run 13 kept four patches whose MARGINAL effects were +5.574%, -0.209%, -0.478%
    and -2.864%. Only the first improved anything.

    The anchor was a fixed binary while the candidate worktree accumulated every kept
    patch, so each reported effect was cumulative against original v9. A patch that
    made the champion WORSE still cleared the floor, because the accumulated total
    did. The champion ended at +1.846% having been +5.574% after one patch.
    """

    def _source(self):
        return (Path(__file__).resolve().parent / "run.py").read_text()

    def test_the_anchor_arm_is_not_the_immutable_cli_argument(self):
        source = self._source()
        measure_node = next(node for node in ast.walk(ast.parse(source))
                            if isinstance(node, ast.FunctionDef) and node.name == "measure_for")
        block = ast.unparse(measure_node)
        self.assertIn("bench.Arm('anchor', anchor_build[0]", block)
        self.assertNotIn("bench.Arm('anchor', args.anchor_build", block,
                         "a static anchor makes every effect cumulative, not marginal")

    def test_it_advances_only_after_the_commit_succeeds(self):
        """An anchor advanced for a patch that did not land would silently raise the
        bar for everything after it. Since the sequential path's deletion the one
        commit is `commit_pooled`: the champion ref moves, THEN the anchor builds."""
        source = self._source()
        block = _function(source, "commit_pooled")
        self.assertIn("advance_champion", block)
        self.assertIn("promote_anchor", block)
        self.assertLess(block.index("advance_champion"),
                        block.index("promote_anchor("),
                        "promotion must follow the commit, never precede it")

    def test_the_guard_runs_before_the_headline_is_published(self):
        """The guard (`verify_anchor`) must precede any headline publish.

        A headline refreshed BEFORE `verify_anchor` would publish a number measured
        against a slot nobody has yet proven holds the champion -- run 18's void
        number, on the panel the operator reads. Found as a live mutation hole at
        the R21-7 port: swapping the two calls survived every suite, because the
        end-to-end tests compose the modules themselves and cannot see this wiring.

        R23-44 relocated the headline: `promote_anchor` advances the ACCUMULATOR and
        runs the guard but NO LONGER publishes; `publish_headline` moved into
        `accumulate_after_keep`, fired only when a bundle's SERVING gate promotes the
        champion of record. The ordering guarantee is preserved structurally: in
        `commit_pooled`, `promote_anchor()` (which contains `verify_anchor()`) is
        called before `accumulate_after_keep()` (which contains `publish_headline()`),
        and inside accumulate_after_keep the publish sits on the PROMOTE branch after
        the champion-of-record snapshot. Pinned here the same way the order is."""
        source = self._source()
        # promote_anchor advances the accumulator + guards, and must NOT publish.
        promote_node = _function_node(source, "promote_anchor")
        promote = ast.unparse(promote_node)
        verify_calls = _calls(promote_node, "verify_anchor")
        self.assertEqual(len(verify_calls), 1)
        guard_keywords = {item.arg: ast.unparse(item.value)
                          for item in verify_calls[0].keywords}
        self.assertEqual(guard_keywords, {"guard_floor": "prior_floor"})
        # 2026-09-07: per-keep headline RESTORED in promote_anchor (guard first, then
        # headline) -- R23-44 had frozen the headline for the whole accumulation phase.
        headline_calls = _calls(promote_node, "publish_headline")
        self.assertEqual(len(headline_calls), 1)
        self.assertLess((verify_calls[0].lineno, verify_calls[0].col_offset),
                        (headline_calls[0].lineno, headline_calls[0].col_offset),
                        "the headline must never publish ahead of the guard")
        # the headline lives in accumulate_after_keep, after the cor snapshot.
        accum = source.split("def _accumulate_after_keep(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("publish_headline()", accum)
        # cor is re-POINTED at the verified gen (never copied) before the headline publishes
        self.assertLess(accum.index("cor_build[0] = anchor_build[0]"), accum.index("publish_headline()"))
        # commit_pooled calls promote_anchor (guard) BEFORE accumulate_after_keep (headline).
        commit = source.split("def commit_pooled(", 1)[1].split("return pool.drive(", 1)[0]
        self.assertLess(commit.index("promote_anchor()"),
                        commit.index("accumulate_after_keep("))

    def test_the_guard_is_wired_with_the_real_code_digest(self):
        """R22-3: the hash pre-check only exists if `run.py` actually injects it.

        `anchor.verify(digest=None)` is the A/A-only fallback, so dropping this one
        kwarg silently reverts the whole triad -- run 21's healthy run aborts again
        and run 18's mismatch costs 20 pairs again -- while every injected-double
        test stays green. Same wiring-only blind spot as the ordering test above."""
        source = self._source()
        verify_node = _function_node(source, "verify_anchor")
        calls = _calls(verify_node, "verify")
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: ast.unparse(item.value) for item in calls[0].keywords}
        self.assertEqual(keywords.get("digest"), "anchor_integrity.object_digest")

    def test_an_excursion_note_reaches_the_headline_refresh(self):
        """The excursion-flagged promotion still publishes -- the anchor is
        hash-proven -- but the bundle must say the session's A/A read above the
        floor. `publish_headline` is the only caller that can carry that note."""
        source = self._source()
        block = source.split("def publish_headline()", 1)[1].split("def ", 1)[0]
        self.assertIn("anchor_guard_seen", block)
        self.assertIn("excursion", block)
        self.assertIn("note=", block)

    def test_the_promoted_build_is_BUILT_in_the_anchor_slot(self):
        """EXECUTED, not grepped. Its predecessor asserted that the string
        "shutil.move" appeared in the source; it passed while `shutil` was never
        imported, so the first real keep raised NameError and the anchor silently
        never advanced.

        The contract itself changed on 2026-08-30: promotion BUILDS the champion into
        the new slot instead of renaming a build directory into it. A CMake build
        directory is not relocatable, and that is the leading root cause of run 18.
        """
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            promoted = pool.promote_anchor(store, build=_champion_build,
                                           champion_commit="5ad3e36d")
            self.assertTrue((promoted / "bin" / "llama-bench").is_file())
            self.assertEqual(promoted.parent, store)

    def test_generations_do_not_collide(self):
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            seen = [pool.promote_anchor(store, build=_champion_build,
                                        champion_commit="5ad3e36d")
                    for _ in range(3)]
            self.assertEqual(len(set(seen)), 3, "each keep needs its own anchor")

    def test_a_champion_that_will_not_build_is_refused(self):
        """A promotion that produced no binary would make every later comparison
        measure nothing at all."""
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                pool.promote_anchor(Path(tmp), build=_broken_build,
                                    champion_commit="5ad3e36d")


class ThePooledPathMustAdvanceTheAnchorToo(unittest.TestCase):
    """The sequential path promotes the anchor inside its own commit. The pooled path
    has a separate commit, and not doing it there would leave the anchor static across
    every lane -- reproducing run 13's defect (cumulative effects reported as marginal,
    a -2.864% regression committed as a keep) at seven times the rate."""

    def _pooled_block(self):
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        return _function(source, "commit_pooled")

    def test_it_advances_the_champion_then_the_anchor(self):
        block = self._pooled_block()
        self.assertIn("pool.advance_champion", block)
        self.assertIn("promote_anchor", block)
        self.assertLess(block.index("pool.advance_champion"),
                        block.index("promote_anchor("),
                        "the anchor must follow the commit, never precede it")

    def test_it_never_promotes_a_lane_build_DIRECTORY(self):
        """A lane's build directory is not relocatable, so it can no longer be handed
        to the promotion at all: the champion is rebuilt into the anchor slot."""
        block = self._pooled_block()
        self.assertIn("promote_anchor()", block)
        self.assertNotIn("promote_anchor(worker.build_dir)", block)
        self.assertNotIn("promote_anchor(args.candidate_build)", block)

    def test_the_pooled_commit_is_actually_wired_in(self):
        """A commit_pooled that nothing calls is the defect it was written to fix."""
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        drive = source.split("return pool.drive(", 1)[1][:400]
        self.assertIn("commit=commit_pooled", drive)


class PromoteAnchorMustACTUALLYRun(unittest.TestCase):
    """`test_the_promoted_build_is_moved_out_of_the_candidate_slot` asserted that the
    string "shutil.move" appears in run.py. It passed. `shutil` was never imported.

    Run 14 kept a real +6.723% patch, advanced the champion, and then raised
    NameError inside promote_anchor -- so the keep was recorded as a lane_error and
    the anchor never advanced. A test that greps source for a spelling proves the
    spelling. This one imports the module and executes the function."""

    def test_the_module_imports_everything_promote_anchor_uses(self):
        from autokernel.loop import pool
        self.assertTrue(hasattr(pool, "json"),
                        "promote_anchor writes provenance.json; pool must import json")

    def test_the_promotion_actually_executes(self):
        """Exercise the real function, not the source text."""
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            promoted = pool.promote_anchor(Path(tmp), build=_champion_build,
                                           champion_commit="5ad3e36d")
            self.assertTrue((promoted / "bin" / "llama-bench").is_file())
            self.assertTrue((promoted / "provenance.json").is_file())


class PromotionMustSurvivePruning(unittest.TestCase):
    """Run 17 lost 23 of its 30 champion advances to this, and the errors were the
    lesser harm: the anchor stopped advancing after the FIRST keep, so every later
    effect was cumulative against a stale champion -- the defect the advancing anchor
    exists to prevent, reintroduced by the pruning meant to save disk.

    Numbering by COUNT collides the moment pruning holds the population steady. With
    keep=1 the count is always 1, so every promotion targeted anchor-gen-002 forever.
    """

    def test_repeated_promote_and_prune_never_collides(self):
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            seen = []
            for _ in range(5):
                promoted = pool.promote_anchor(store, build=_champion_build,
                                               champion_commit="5ad3e36d")
                pool.prune_anchor_generations(store, current=promoted)
                seen.append(promoted.name)
            self.assertEqual(len(set(seen)), 5, f"generations collided: {seen}")

    def test_it_refuses_to_reuse_an_existing_anchor(self):
        """Configuring into a slot that already holds a generation would build over
        another champion's CMakeCache -- the relocation hazard by another route."""
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "anchor-gen-001").mkdir()
            (store / "anchor-gen-002").mkdir()
            # numbering must step past BOTH, not reuse 002
            promoted = pool.promote_anchor(store, build=_champion_build,
                                           champion_commit="5ad3e36d")
            self.assertEqual(promoted.name, "anchor-gen-003")
            self.assertTrue((promoted / "bin").is_dir())


class TwoTierChampionWiring(unittest.TestCase):
    """R23-44: the accumulator advances on bench keeps; the champion of record advances
    only when a bundle's serving gate promotes it. These pin the run.py wiring the unit
    tests on accumulate.py cannot see (same wiring-only blind spot as the guard-order test)."""

    def _source(self):
        return (Path(__file__).resolve().parent / "run.py").read_text()

    def test_keep_gate_is_the_bench_confirm_rung_not_a_per_keep_serving_gate(self):
        src = self._source()
        # the removed per-keep serving gate must be gone; the keep gate is confirm.gate
        self.assertNotIn("def serving_confirm", src)
        commit = src.split("def commit_pooled(", 1)[1].split("return pool.drive(", 1)[0]
        self.assertIn("confirm.gate(", commit)
        self.assertIn("accumulate_after_keep(hypothesis.mechanism_id)", commit)

    def test_serving_gate_compares_champion_of_record_against_accumulator(self):
        src = self._source()
        accum = src.split("def _accumulate_after_keep(", 1)[1].split("\n    def ", 1)[0]
        # the serving A-arm is the champion-of-record build, B-arm the accumulator anchor
        self.assertIn("measured_serving_compare(serving_recipe, cor_build[0], anchor_build[0]", accum)
        # it fires only on a named trigger (R23-54: threshold OR the keep cadence, 8 since 2026-09-28), and
        # the trigger is recorded — never left for a reader to infer from the compounded
        # number the 2026-09-08 divergence discredited.
        self.assertIn("accumulate.gate_trigger(bundle[0], serving_floor_pct, accum_policy)",
                      accum)
        self.assertIn("if trigger is None:", accum)
        self.assertIn('"trigger": trigger', accum)
        # the cadence counter resets on BOTH outcomes, and each reset is persisted
        self.assertEqual(2, accum.count("mark_serving_gate_fired()"),
                         "the cadence counter must reset on promote AND on divergence")
        # promote advances cor + snapshots + headline; divergence journals evidence
        self.assertIn("accumulate.Outcome.PROMOTE", accum)
        self.assertIn("planner_evidence", accum)
        self.assertIn("measured_divergence", accum)

    def test_cor_build_is_a_protected_pointer_never_a_copy(self):
        src = self._source()
        # 2026-09-06: cor_build POINTS at the real anchor gen. A copied CMake build carries an
        # absolute RUNPATH into the source gen, which prune deleted, killing every accumulate
        # step. So: no copy, and the gen is handed to prune as `protect`.
        self.assertNotIn("snapshot_cor(", src)
        self.assertNotIn("shutil.copytree", src)
        self.assertIn("cor_build = [args.cor_build or args.anchor_build]", src)
        self.assertIn("protect=[cor_build[0]]", src)
        # a serving PROMOTE re-points cor at the verified accumulator gen
        self.assertIn("cor_build[0] = anchor_build[0]", src)

    def test_accumulate_failure_is_loud_and_never_unrecords_the_keep(self):
        src = self._source()
        wrapper = src.split("def accumulate_after_keep(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("except Exception", wrapper)
        self.assertIn("accum     FAILED", wrapper)
        self.assertIn("accum-error-", wrapper)

    def test_experimental_serving_uses_the_durable_accumulator(self):
        src = self._source()
        setup = src.split("def _is_ancestor", 1)[1].split("last_gate = [None]", 1)[0]
        self.assertIn("accumulate.load_bundle(", setup)
        self.assertNotIn("if experimental:", setup)
        wrapper = src.split("def accumulate_after_keep(", 1)[1].split("\n    def ", 1)[0]
        self.assertNotIn("or experimental", wrapper)
        compounded = src.split("def _accumulate_after_keep(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("cpu_compare(cor_build[0], anchor_build[0], rebind_feedback=False)", compounded)
        drive = src.split("return pool.drive(", 1)[1].split("\n        )", 1)[0]
        self.assertIn(
            "accumulate_valid_positive=experimental", drive)
    def test_fire_multiple_arg_defaults_to_operator_range(self):
        src = self._source()
        arg = src.split('"--fire-multiple"', 1)[1][:120]
        self.assertIn("default=2.5", arg)

    def test_publish_exposes_accumulator_state_for_the_dashboard(self):
        # R23-44 observability: the loop status must carry the bundle so the dashboard
        # can render keeps accumulating toward the serving gate.
        src = self._source()
        self.assertIn("def accumulator_state(", src)
        pub = src.split("def publish(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("accumulator=accumulator_state()", pub)
        acc = src.split("def accumulator_state(", 1)[1].split("\n    def ", 1)[0]
        for field in ("compounded_bench_pct", "fire_threshold_pct", "n_keeps",
                      "progress_fraction", "champion_of_record"):
            self.assertIn(field, acc)
