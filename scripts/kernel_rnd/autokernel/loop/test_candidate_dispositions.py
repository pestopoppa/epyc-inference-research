"""Abandoned candidates are recorded, and the Q4/Q5 dot gate admits the DS41 anchor.

Origin: DS41 run 9c (2026-09-25). A hoist in `mul_mat_qX_K_q8_2_X4_T` was authored
twice, accepted by the critic twice and refused twice by `op_scope` before any
build: the gate's boundary markers named two helpers that exist only in the v10
keeps tree, so on the DS41 anchor it could never pass, and its reason was the
generic IQK text. The iteration then moved to a new hypothesis, was stopped, and
its only experiment row called the candidate "never attempted".
"""
import ast
from pathlib import Path
import unittest

from autokernel.loop import gates, loop, pipeline

KQUANTS = "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"
SYMBOL = "mul_mat_qX_K_q8_2_X4_T"

#: Shape of the DS41 anchor (5a60152ae) and production v10: NO unpack_q4_scales
#: helpers between the Q5_K dequantizer and the dot template.
ANCHOR = ("struct Q4Bits_AVX2 {\n    shared();\n};\n"
          "struct DequantizerQ4K_AVX2 final : Base {\n    q4();\n};\n"
          "struct DequantizerQ5K_AVX2 final : Base {\n    q5();\n};\n"
          "template <typename Dequantizer, int nrc_y>\n"
          "static void mul_mat_qX_K_q8_2_X4_T(int n) {\n"
          "    float d8[8*nrc_y];\n"
          "    for (int ix = 0; ix < n; ++ix) {\n"
          "        convert();\n"
          "    }\n"
          "}\n"
          "struct DequantizerQ6K_AVX2 final : Base {\n    q6();\n};\n"
          "case GGML_TYPE_Q4_K:\n"
          "IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_X4_T, DequantizerQ4K_AVX2, kernels)\n"
          "case GGML_TYPE_Q5_K:\n"
          "IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_X4_T, DequantizerQ5K_AVX2, kernels)\n")


def _line(text, token):
    return text.splitlines().index(token) + 1


def _scope(source, pre, patch):
    return gates.affected_op_scope((KQUANTS,), target_surface=KQUANTS,
                                   target_symbol=SYMBOL, source_text=source,
                                   pre_source_text=pre, patch_text=patch)


class Q45DotGateOnAnAnchorWithoutTheKeepsHelpers(unittest.TestCase):

    def test_the_run9c_hoist_shape_is_admitted(self):
        # Replace a body line and insert a pre-pass inside the dot body: the
        # run 9c candidate's shape, which the old marker list refused outright.
        candidate = ANCHOR.replace(
            "    float d8[8*nrc_y];\n",
            "    std::vector<float> ybuf(16*nrc_y);\n"
            "    for (int i = 0; i < n; ++i) {\n        prepass();\n    }\n")
        at = _line(ANCHOR, "    float d8[8*nrc_y];")
        patch = f"@@ -{at} +{at},4 @@\n-old\n+new\n"
        self.assertEqual(_scope(candidate, ANCHOR, patch), ("MUL_MAT", "MUL_MAT_ID"))
        for token in ("    q4();", "    q5();", "        convert();"):
            at = _line(ANCHOR, token)
            self.assertEqual(_scope(ANCHOR, ANCHOR, f"@@ -{at} +{at} @@\n-a\n+b\n"),
                             ("MUL_MAT", "MUL_MAT_ID"), token)

    def test_a_sibling_edit_is_refused_with_the_rule_not_the_generic_text(self):
        for token in ("    shared();", "    q6();", "case GGML_TYPE_Q4_K:"):
            at = _line(ANCHOR, token)
            verdict = _scope(ANCHOR, ANCHOR, f"@@ -{at} +{at} @@\n-a\n+b\n")
            self.assertIsInstance(verdict, gates.Verdict, token)
            self.assertFalse(verdict.passed)
            self.assertIn("lies outside every admitted body", verdict.reason)
            self.assertIn("mul_mat_qX_K_q8_2_X4_T", verdict.reason)
            self.assertNotIn("generic per-type [iqk] ACTIVE marker", verdict.reason)

    def test_helpers_must_agree_between_head_and_candidate(self):
        with_helper = ANCHOR.replace(
            "template <typename Dequantizer, int nrc_y>\n",
            "inline __m128i unpack_q4_scales(const uint8_t * x) {\n    scale();\n}\n"
            "template <typename Dequantizer, int nrc_y>\n")
        at = _line(ANCHOR, "        convert();")
        verdict = _scope(with_helper, ANCHOR, f"@@ -{at} +{at + 3} @@\n-a\n+b\n")
        self.assertFalse(verdict.passed)
        self.assertIn("helper set differs", verdict.reason)
        # Present on both sides, the helper body is itself admitted.
        at = _line(with_helper, "    scale();")
        self.assertEqual(_scope(with_helper, with_helper, f"@@ -{at} +{at} @@\n-a\n+b\n"),
                         ("MUL_MAT", "MUL_MAT_ID"))

    def test_a_missing_required_marker_names_itself(self):
        drifted = ANCHOR.replace("struct DequantizerQ6K_AVX2 final : Base",
                                 "struct DequantizerQ6K final : Base")
        at = _line(ANCHOR, "        convert();")
        verdict = _scope(drifted, drifted, f"@@ -{at} +{at} @@\n-a\n+b\n")
        self.assertFalse(verdict.passed)
        self.assertIn("struct DequantizerQ6K_AVX2 final :", verdict.reason)
        self.assertIn("occurs 0 times", verdict.reason)


def _hypothesis(mechanism="akm-q4k-x4-actscale-hoist"):
    return loop.Hypothesis(mechanism_id=mechanism, statement="hoist the scales",
                           falsifier="no effect above the floor",
                           target_surface=KQUANTS, target_symbol=SYMBOL)


class _Planner:
    def __init__(self, hypotheses):
        self.hypotheses = list(hypotheses)
        self.proposals = 0

    def propose(self, context):
        self.proposals += 1
        return self.hypotheses[min(self.proposals, len(self.hypotheses)) - 1]

    def author(self, hypothesis, context):
        return (KQUANTS,)


class _Critic:
    def __init__(self, hypothesis_verdicts=(), patch_verdicts=()):
        self.hypothesis_verdicts = list(hypothesis_verdicts)
        self.patch_verdicts = list(patch_verdicts)

    def review_hypothesis(self, hypothesis, context):
        return self.hypothesis_verdicts.pop(0) if self.hypothesis_verdicts \
            else loop.Review(True)

    def review_patch(self, hypothesis, paths, context):
        return self.patch_verdicts.pop(0) if self.patch_verdicts else loop.Review(True)


REFUSAL = "CPU IQK Q4_K/Q5_K dot route refused before build: hunk outside"


def _refusing_gate(hypothesis, paths):
    return False, [gates.Verdict("op_scope", False, REFUSAL)]


class AbandonedCandidatesGetTheirOwnRow(unittest.TestCase):

    def test_run9c_sequence_records_both_refusals_and_the_stop_does_not_deny_them(self):
        """author -> accept -> op_scope refuse, twice; then STOP on the next proposal."""
        recorded = []
        planner = _Planner([_hypothesis(), _hypothesis("akm-next")])
        outcome = loop.iterate(
            planner=planner, critic=_Critic(), context={},
            measure=lambda h, p: self.fail("a refused patch must not be measured"),
            gate=_refusing_gate, commit=lambda h, p, c: self.fail("no commit"),
            should_abandon=lambda: planner.proposals >= 1 and len(recorded) >= 2,
            record_abandoned=recorded.append)
        self.assertEqual([c.status for c in recorded], ["gate_refused", "gate_refused"])
        for patch_round, candidate in enumerate(recorded, start=1):
            self.assertEqual(candidate.refusal_gate, "op_scope")
            self.assertEqual(candidate.reasons, [REFUSAL])
            self.assertEqual(candidate.hypothesis.mechanism_id, "akm-q4k-x4-actscale-hoist")
            self.assertEqual((candidate.hypothesis_round, candidate.patch_round),
                             (1, patch_round))
            self.assertEqual(candidate.gate_verdicts[-1].gate, "op_scope")
            row = candidate.to_attempt()
            self.assertEqual(row["refusal_gate"], "op_scope")
            self.assertEqual(row["reason"], REFUSAL)
        self.assertEqual(outcome.status, "stopped_mid_formation")
        self.assertEqual(outcome.reasons[0], loop.STOPPED_AFTER_DISPOSALS)
        self.assertNotIn("never attempted", " ".join(outcome.reasons))
        self.assertIn("gate_refused by op_scope", outcome.reasons[-1])
        self.assertEqual(len(outcome.abandoned_candidates), 2)
        self.assertEqual(outcome.to_attempt()["abandoned_candidates"],
                         outcome.abandoned_candidates)

    def test_a_clean_stop_keeps_the_never_attempted_text(self):
        outcome = loop.iterate(
            planner=_Planner([_hypothesis()]), critic=_Critic(), context={},
            measure=lambda h, p: None, gate=_refusing_gate, commit=lambda h, p, c: None,
            should_abandon=lambda: True, record_abandoned=lambda c: self.fail("none"))
        self.assertEqual(outcome.reasons, [loop.STOPPED_MID_FORMATION])
        self.assertEqual(outcome.abandoned_candidates, [])
        self.assertNotIn("abandoned_candidates", outcome.to_attempt())

    def test_critic_rejections_are_recorded_even_when_a_later_round_is_measured(self):
        recorded = []
        from autokernel.loop import bench
        comparison = bench.Comparison(
            surface="tg128", anchor_samples=[100.0], candidate_samples=[100.0],
            effect=0.0, estimator="median_over_median", pairs=5, noise_floor_pct=1.0,
            residency={"invocations": 10, "resident": 10})
        outcome = loop.iterate(
            planner=_Planner([_hypothesis("akm-first"), _hypothesis("akm-second")]),
            critic=_Critic([loop.Review(False, "unsupported premise")],
                           [loop.Review(False, "patch edits Q6_K")]),
            context={}, measure=lambda h, p: comparison,
            gate=lambda h, p: (True, [gates.Verdict("compile", True)]),
            commit=lambda h, p, c: None, record_abandoned=recorded.append)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual([(c.status, c.refusal_gate, c.reasons[0]) for c in recorded],
                         [("hypothesis_rejected", "critic:hypothesis", "unsupported premise"),
                          ("patch_rejected", "critic:patch", "patch edits Q6_K")])
        self.assertEqual([c.hypothesis.mechanism_id for c in recorded],
                         ["akm-first", "akm-second"])
        self.assertEqual([row["status"] for row in outcome.abandoned_candidates],
                         ["hypothesis_rejected", "patch_rejected"])

    def test_a_recording_fault_is_carried_not_raised(self):
        def broken(candidate):
            raise OSError("store is read-only")
        outcome = loop.iterate(
            planner=_Planner([_hypothesis()]), critic=_Critic(), context={},
            measure=lambda h, p: None, gate=_refusing_gate, commit=lambda h, p, c: None,
            hypothesis_rounds=1, record_abandoned=broken)
        self.assertEqual(outcome.status, "refused_at_formation")
        self.assertEqual(len(outcome.abandoned_candidates), 2)
        self.assertTrue(all("store is read-only" in row["record_error"]
                            for row in outcome.abandoned_candidates))


class PoolAttachesLineageAndTheRefusedDiffsReservation(unittest.TestCase):

    def test_abandoned_rows_are_not_iterations_and_own_their_reservation(self):
        from autokernel.loop import dispatch_guard
        records, abandoned = [], []
        counter = iter(range(1, 100))

        def reserve(worker, hypothesis, paths):
            n = next(counter)
            return dispatch_guard.Reservation(f"identity-{n}", 1, f"diff-{n}")

        worker = pipeline.Worker("lane0", Path("/w/lane0"), Path("/b/lane0"))
        outcomes = pipeline.run_pool(
            workers=[worker], make_planner=lambda w: _Planner([_hypothesis()]),
            make_critic=lambda w: _Critic(), build_context=dict,
            make_gate=lambda w: _refusing_gate, make_measure=lambda w: None,
            commit=lambda w, h, p, c: None, champion_head=lambda: "c0",
            reset_to_champion=lambda w: "c0", record=records.append, iterations=1,
            reserve_candidate=reserve,
            record_abandoned=lambda w, c: abandoned.append((w.name, c)))
        self.assertEqual(len(outcomes), 1)          # the batch count is untouched
        self.assertEqual(records, outcomes)
        # 3 hypothesis rounds x 2 patch rounds, each refused at op_scope.
        self.assertEqual(len(abandoned), 6)
        self.assertEqual([c.attempt_identity for _w, c in abandoned],
                         [f"identity-{n}" for n in range(1, 7)])
        self.assertEqual({(w, c.branch_id, c.spawn_parent, c.depth) for w, c in abandoned},
                         {("lane0", "detached:lane0", "c0", 1)})
        final = outcomes[0]
        self.assertEqual(final.status, "refused_at_formation")
        self.assertIsNone(final.attempt_identity)  # never inherits a refused diff's
        self.assertEqual(len(final.abandoned_candidates), 6)


class RunWiresTheRecorder(unittest.TestCase):

    def test_run_py_records_retains_and_narrates_every_abandoned_candidate(self):
        source = (Path(__file__).with_name("run.py")).read_text(encoding="utf-8")
        tree = ast.parse(source)
        recorder = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "record_abandoned_pooled"]
        self.assertEqual(len(recorder), 1)
        called = {getattr(call.func, "attr", getattr(call.func, "id", None))
                  for call in ast.walk(recorder[0]) if isinstance(call, ast.Call)}
        self.assertTrue({"keep_the_diff", "record", "finish", "publish", "print"} <= called)
        drives = [call for call in ast.walk(tree) if isinstance(call, ast.Call)
                  and getattr(call.func, "attr", None) == "drive"]
        self.assertTrue(any(kw.arg == "record_abandoned"
                            and getattr(kw.value, "id", None) == "record_abandoned_pooled"
                            for call in drives for kw in call.keywords))


if __name__ == "__main__":
    unittest.main()
