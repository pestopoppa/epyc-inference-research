"""The profiler table reaches the planner and both critic passes.

`rocprofv3 --kernel-trace` has run on every attempt since the GPU lane opened, and
`gpu_source_evidence` sealed a full per-signature table -- durations, call counts,
both arms -- into `exact_duration_comparison`. The controller read exactly ONE field
out of it (`relative_improvement_fraction`) and discarded the rest, so the planner
chose which kernel to attack while blind to the profile of the tree it was editing.

These tests pin the reduction and the wiring. They do not measure anything new.
"""
import contextlib
import tempfile
import unittest
from pathlib import Path

from autokernel.controller import discovery_controller as dc
from autokernel.controller import gpu_source_evidence as ev


@contextlib.contextmanager
def _controller_fixture():
    """A real tracker and config; `_context` refuses stand-ins for either."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = dc.ControllerConfig(root, 1, dry_run=True)
        yield dc._tracker(dc.DurableState(root)), config


def _routes(**pairs):
    return {name: {"total_duration_ns": ns, "calls": calls}
            for name, (ns, calls) in pairs.items()}


COMPARISON = {
    "anchor_routes": _routes(
        mul_mat_vec_q_q5_0=(700_000, 13_803),
        rms_norm_f32=(200_000, 512),
        rope_f32=(100_000, 512),
    ),
    "candidate_routes": _routes(
        mul_mat_vec_q_q5_0=(630_000, 13_803),
        rms_norm_f32=(200_000, 512),
        rope_f32=(100_000, 512),
    ),
    "anchor_total_duration_ns": 1_000_000,
    "candidate_total_duration_ns": 930_000,
    "relative_improvement_fraction": 0.07,
}


class KernelHotspots(unittest.TestCase):

    def test_ranked_by_anchor_share_descending(self):
        rows = ev.kernel_hotspots(COMPARISON)
        self.assertEqual([row["signature"] for row in rows],
                         ["mul_mat_vec_q_q5_0", "rms_norm_f32", "rope_f32"])
        self.assertAlmostEqual(rows[0]["anchor_share_of_device_time"], 0.7)
        self.assertEqual(rows[0]["anchor_calls"], 13_803)

    def test_per_route_effect_is_reported(self):
        rows = ev.kernel_hotspots(COMPARISON)
        self.assertAlmostEqual(rows[0]["relative_improvement_fraction"], 0.1)
        self.assertEqual(rows[1]["relative_improvement_fraction"], 0.0)

    def test_a_route_missing_from_one_arm_is_null_not_zero(self):
        """'never dispatched' must never be read as 'took no time'."""
        comparison = dict(COMPARISON)
        comparison["candidate_routes"] = _routes(rms_norm_f32=(200_000, 512))
        rows = {row["signature"]: row for row in ev.kernel_hotspots(comparison)}
        self.assertIsNone(rows["mul_mat_vec_q_q5_0"]["candidate_total_duration_ns"])
        self.assertIsNone(rows["mul_mat_vec_q_q5_0"]["relative_improvement_fraction"])
        # It still ranks first: it is the anchor's hotspot whether or not the
        # candidate dispatched it.
        self.assertEqual(ev.kernel_hotspots(comparison)[0]["signature"],
                         "mul_mat_vec_q_q5_0")

    def test_limit_is_honoured(self):
        self.assertEqual(len(ev.kernel_hotspots(COMPARISON, limit=2)), 2)

    def test_malformed_comparison_yields_no_rows_rather_than_raising(self):
        for bad in ({}, {"anchor_routes": {}}, {"anchor_routes": {}, "candidate_routes": {}},
                    {"anchor_routes": {}, "candidate_routes": {},
                     "anchor_total_duration_ns": 0},
                    {"anchor_routes": {}, "candidate_routes": {},
                     "anchor_total_duration_ns": True}):
            self.assertEqual(ev.kernel_hotspots(bad), [], bad)


class HotspotsReachBothActors(unittest.TestCase):

    def _state(self, **row):
        base = {"result_sha256": "a" * 64, "status": "screened"}
        base.update(row)
        return {"iterations": [base]}

    def test_context_carries_the_most_recent_profile(self):
        rows = ev.kernel_hotspots(COMPARISON)
        state = self._state(hotspots=rows)
        with _controller_fixture() as (tracker, config):
            context = dc._context(state, tracker, 3, config, None)
        self.assertEqual(context["kernel_hotspots"], rows)
        self.assertEqual(context["kernel_hotspots_from_result_sha256"], "a" * 64)

    def test_the_newest_profile_wins_not_the_first(self):
        """Every accepted patch moves the distribution; a stale ranking aims the
        loop at the previous champion's hotspots."""
        old = ev.kernel_hotspots(COMPARISON)
        newer = dict(COMPARISON)
        newer["anchor_routes"] = _routes(rope_f32=(900_000, 512))
        newer["anchor_total_duration_ns"] = 900_000
        newer["candidate_routes"] = _routes(rope_f32=(900_000, 512))
        fresh = ev.kernel_hotspots(newer)
        state = {"iterations": [
            {"result_sha256": "a" * 64, "status": "screened", "hotspots": old},
            {"result_sha256": "b" * 64, "status": "screened", "hotspots": fresh},
        ]}
        with _controller_fixture() as (tracker, config):
            context = dc._context(state, tracker, 4, config, None)
        self.assertEqual(context["kernel_hotspots"][0]["signature"], "rope_f32")
        self.assertEqual(context["kernel_hotspots_from_result_sha256"], "b" * 64)

    def test_a_turn_with_no_profile_yet_reports_empty_not_missing(self):
        with _controller_fixture() as (tracker, config):
            context = dc._context({"iterations": []}, tracker, 1, config, None)
        self.assertEqual(context["kernel_hotspots"], [])
        self.assertIsNone(context["kernel_hotspots_from_result_sha256"])

    def test_rows_without_a_profile_are_skipped_not_treated_as_empty(self):
        state = {"iterations": [
            {"result_sha256": "a" * 64, "status": "screened",
             "hotspots": ev.kernel_hotspots(COMPARISON)},
            {"result_sha256": "b" * 64, "status": "authoring_refused"},   # no profile
        ]}
        with _controller_fixture() as (tracker, config):
            context = dc._context(state, tracker, 5, config, None)
        self.assertEqual(context["kernel_hotspots_from_result_sha256"], "a" * 64)

    def test_the_screen_row_persists_the_profile_for_later_turns(self):
        screen = dc.SealedScreen(
            receipt_path="/tmp/x.json", result_sha256="c" * 64, effect_fraction=0.07,
            classification="screened_out", baseline_sha256="d" * 64,
            source_proof_sha256="e" * 64, dispatch_proof_sha256="f" * 64,
            hotspots=tuple(ev.kernel_hotspots(COMPARISON)))
        fields = dc._screen_iteration_fields(screen, repetition=1)
        self.assertEqual(fields["hotspots"][0]["signature"], "mul_mat_vec_q_q5_0")


if __name__ == "__main__":
    unittest.main()
