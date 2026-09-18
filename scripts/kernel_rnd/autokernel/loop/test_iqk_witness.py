"""No-hardware tests for the trusted IQK case-witness reducer."""
import json
from pathlib import Path
from unittest import TestCase, mock

from . import iqk_witness as witness


DSO = Path("/build/bin/libggml-cpu.so.0")
CASE = "type_a=q4_K,type_b=f32,n_mats=4,n_used=1,b=0,m=512,n=1,k=256"


def _record(role, symbol, *, status="hit", dso=None):
    return json.dumps({"schema": "epyc.autokernel.iqk_case_hit.v1",
                       "status": status, "role": role, "symbol": symbol,
                       "dso": str(DSO if dso is None else dso)})


class IQKWitnessTests(TestCase):
    def setUp(self):
        self.records = "\n".join((
            _record("independent_reference", "ggml_backend_cpu_set_use_ref"),
            _record("candidate_helper", witness.SYMBOL)))
        self.output = (f"MUL_MAT_ID({CASE}): OK\n  1/1 tests passed\n"
                       "  Backend CPU: OK\n"
                       "[iqk] ACTIVE: MoE mul_mat_id via ik kernels "
                       "(type=12 activation=99 n_as=4)\nexited normally\n")

    def assess(self, records=None, output=None, returncode=0):
        with mock.patch.object(Path, "resolve", return_value=DSO):
            return witness._assess_case(self.records if records is None else records,
                                        self.output if output is None else output,
                                        "", returncode, DSO, CASE, 12, "Q4_K")

    def test_exact_two_debugger_hits_and_one_passing_case(self):
        self.assertEqual(self.assess().status, "pass")

    def test_candidate_print_cannot_replace_debugger_pipe(self):
        self.assertEqual(self.assess(records="").status, "unavailable")
        self.assertEqual(self.assess(records=_record(
            "candidate_helper", witness.SYMBOL)).status, "unavailable")

    def test_wrong_dso_or_symbol_or_quant_marker_refuses(self):
        bad = self.records.replace("/build/bin/libggml-cpu.so.0", "/other/libggml-cpu.so")
        self.assertEqual(self.assess(records=bad).status, "unavailable")
        self.assertEqual(self.assess(output=self.output.replace("type=12", "type=13")).status,
                         "unavailable")

    def test_case_failure_is_wrong_and_missing_case_is_unavailable(self):
        self.assertEqual(self.assess(output=self.output.replace(": OK", ": FAIL"),
                                     returncode=1).status, "wrong")
        self.assertEqual(self.assess(output=self.output.replace("  1/1 tests passed", "  0/0 tests passed"))
                         .status, "unavailable")

    def test_two_quant_witnesses_precede_scalar_comparison(self):
        from . import cpu_quant_reference
        with mock.patch.object(witness, "_check_one", side_effect=(
                witness.Result("pass", "q4"), witness.Result("pass", "q5"))) as hit, \
             mock.patch.object(cpu_quant_reference, "check_cpu_quant_suite",
                return_value=cpu_quant_reference.QuantResult(
                    "pass", "scalar", "AK_CPU_QUANT_METRIC_V1 {\"quant\":\"Q4_K\"}")) as scalar:
            recipe = mock.Mock(launch_env={"GGML_IQK": "1"}, topology_prefix=())
            result = witness.check(Path("/build"), resolved_recipe=recipe,
                                   source_root=Path("/source"))
        self.assertEqual(result.status, "pass")
        self.assertEqual(hit.call_count, 2)
        self.assertEqual(scalar.call_args.kwargs["quants"], ("Q4_K", "Q5_K"))
        self.assertEqual(scalar.call_args.kwargs["ops"], ("MUL_MAT_ID",))
        self.assertIn("AK_CPU_QUANT_METRIC_V1", result.detail)

    def test_first_quant_failure_skips_other_quant_and_scalar(self):
        from . import cpu_quant_reference
        with mock.patch.object(witness, "_check_one",
                return_value=witness.Result("unavailable", "missing symbol")) as hit, \
             mock.patch.object(cpu_quant_reference, "check_cpu_quant_suite") as scalar:
            result = witness.check(Path("/build"), resolved_recipe=mock.Mock(),
                                   source_root=Path("/source"))
        self.assertEqual(result.status, "unavailable")
        hit.assert_called_once()
        scalar.assert_not_called()

    def test_fused_route_requires_exact_two_quant_graph_and_helper_hit(self):
        from . import cpu_quant_reference
        recipe = mock.Mock(backend="cpu", launch_env={"GGML_IQK": "1"},
                           topology_prefix=(), template=object(), port=1)
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(cpu_quant_reference, "check_cpu_quant_suite",
                return_value=cpu_quant_reference.QuantResult(
                    "pass", "trusted fused helper hit proven", path_verified=True)) as scalar:
            result = witness.check_fused(Path("/build"), resolved_recipe=recipe,
                                         source_root=Path("/source"))
        self.assertEqual(result.status, "pass")
        self.assertEqual(scalar.call_args.kwargs["quants"], ("Q4_K", "Q5_K"))
        self.assertEqual(scalar.call_args.kwargs["ops"], ("FUSED_UP_GATE",))
        self.assertTrue(scalar.call_args.kwargs["require_fused_hit"])

    def test_fused_route_rejects_wrong_resolved_recipe(self):
        recipe = mock.Mock(backend="cpu", launch_env={"GGML_IQK": "0"})
        result = witness.check_fused(Path("/build"), resolved_recipe=recipe,
                                     source_root=Path("/source"))
        self.assertEqual(result.status, "unavailable")

    def test_dot_hit_requires_matching_quant_width_and_candidate_dso(self):
        records = [json.loads(_record("candidate_helper", "iqk_moe_fused_up_gate")),
                   json.loads(_record("candidate_dot", "mul_mat_qX_K_q8_2_X4_T"))]
        records[1].update(quant="Q4_K", width=3,
                          frame_symbol="mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 3>")
        with mock.patch.object(Path, "resolve", return_value=DSO):
            self.assertEqual(witness._assess_fused_hit_records(
                records, DSO, dot_quant="Q4_K", width=3).status, "pass")
            for key, wrong in (("quant", "Q5_K"), ("width", 2),
                               ("dso", "/other/libggml-cpu.so.0")):
                bad = [dict(row) for row in records]
                bad[1][key] = wrong
                self.assertEqual(witness._assess_fused_hit_records(
                    bad, DSO, dot_quant="Q4_K", width=3).status, "unavailable")
            self.assertEqual(witness._assess_fused_hit_records(
                records[:1], DSO, dot_quant="Q4_K", width=3).status,
                "unavailable")

    def test_q45_dot_reference_requires_all_widths_direct_then_fused(self):
        from . import cpu_quant_reference
        recipe = mock.Mock(backend="cpu", launch_env={"GGML_IQK": "1"},
                           topology_prefix=(), template=object(), port=1)
        passed = cpu_quant_reference.QuantResult("pass", "scalar", "metric")
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(cpu_quant_reference, "check_cpu_quant_suite",
                               return_value=passed) as scalar:
            result = witness.check_q45_dot(Path("/build"), resolved_recipe=recipe,
                                            source_root=Path("/source"))
        self.assertEqual(result.status, "pass")
        self.assertEqual(scalar.call_count, 2)
        direct, fused = scalar.call_args_list
        self.assertEqual(direct.kwargs["ops"], ("MUL_MAT", "MUL_MAT_ID"))
        self.assertEqual(fused.kwargs["ops"], ("FUSED_UP_GATE",))
        for call in (direct, fused):
            self.assertEqual(call.kwargs["quants"], ("Q4_K", "Q5_K"))
            self.assertEqual(call.kwargs["widths"], tuple(range(1, 9)))
            self.assertEqual(call.kwargs["expert_mode"], "single")
        self.assertTrue(fused.kwargs["require_dot_hit"])
        self.assertTrue(fused.kwargs["require_fused_hit"])

    def test_q45_dot_refuses_before_fused_when_direct_wrong(self):
        from . import cpu_quant_reference
        recipe = mock.Mock(backend="cpu", launch_env={"GGML_IQK": "1"},
                           topology_prefix=(), template=object(), port=1)
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(cpu_quant_reference, "check_cpu_quant_suite",
                return_value=cpu_quant_reference.QuantResult("wrong", "direct wrong")) as scalar:
            result = witness.check_q45_dot(Path("/build"), resolved_recipe=recipe,
                                            source_root=Path("/source"))
        self.assertEqual(result.status, "wrong")
        scalar.assert_called_once()
