#!/usr/bin/env python3
import unittest

from . import lanes as L


class LaneRegistryTest(unittest.TestCase):
    def test_default_registry_includes_quarters_and_historical_deeper_splits(self):
        registry = L.default_lane_registry()
        self.assertEqual(
            [key for key in registry if key.startswith("cpu-q")],
            ["cpu-q0", "cpu-q1", "cpu-q2", "cpu-q3"])
        for count, threads in ((4, 48), (8, 24), (16, 12), (32, 6), (48, 4)):
            self.assertEqual(
                len([key for key in registry if key.startswith(f"cpu-{count}x{threads}t-")]),
                count)
        self.assertTrue(L.may_gate_performance_claim(registry["cpu-full"]))
        self.assertTrue(L.may_gate_performance_claim(registry["gpu-full"]))
        self.assertFalse(L.may_gate_performance_claim(registry["cpu-q0"]))

    def test_partial_cpu_lane_requires_no_mmap_and_explicit_membind(self):
        with self.assertRaisesRegex(L.LaneError, "--no-mmap"):
            L.LaneSpec("bad", L.CPU_REGION, 1, 1, "cpu-full", L.QUARTER,
                       "0-23", (0,), True)
        with self.assertRaisesRegex(L.LaneError, "membind"):
            L.LaneSpec("bad", L.CPU_REGION, 1, 1, "cpu-full", L.QUARTER,
                       "0-23", (), False)

    def test_concurrent_cpu_lanes_refuse_core_overlap_but_allow_shared_memory_node(self):
        registry = L.default_lane_registry()
        L.validate_concurrent_cpu_lanes([registry["cpu-q0"], registry["cpu-q1"]])
        with self.assertRaisesRegex(L.LaneError, "overlap physical CPUs"):
            L.validate_concurrent_cpu_lanes([registry["cpu-full"], registry["cpu-q1"]])
        alias = L.LaneSpec("alias", L.CPU_REGION, 1, 1, "cpu-full", L.QUARTER,
                           "24-47", (0,), False)
        # Sharing a memory node is permitted and measured as a transfer effect;
        # the CPU footprints themselves remain disjoint.
        L.validate_concurrent_cpu_lanes([registry["cpu-q0"], alias])

    def test_historical_48_way_shape_is_cpu_disjoint_despite_shared_membind(self):
        lanes = list(L.historical_split_lane_registry().values())
        deepest = [lane for lane in lanes if lane.lane_id.startswith("cpu-48x4t-")]
        self.assertEqual(len(deepest), 48)
        L.validate_concurrent_cpu_lanes(deepest)
        self.assertEqual(deepest[0].cpu_list, "0-1,96-97")
        self.assertEqual(deepest[-1].cpu_list, "94-95,190-191")
        self.assertTrue(all(not lane.use_mmap and lane.membind_nodes for lane in deepest))

    def test_historical_lane_cost_is_physical_core_share_not_smt_thread_count(self):
        registry = L.default_lane_registry()
        self.assertEqual(registry["cpu-q0"].cost, 1.0)
        self.assertEqual(registry["cpu-4x48t-00"].cost, 1.0)
        self.assertLess(registry["cpu-48x4t-00"].cost,
                        registry["cpu-32x6t-00"].cost)


class TransferCalibrationTest(unittest.TestCase):
    def calibration(self, lane="cpu-q0", order=("a", "b", "c"), *, aa=True):
        full = ("a", "b", "c")
        return L.RankCalibration(
            "arithmetic", lane, "cpu-full", full, order, full,
            L.spearman_rank_fidelity(order, full), ("ake-screen", "ake-full"),
            aa, "prediction://ak-ln2/arithmetic")

    def test_cheapest_lane_requires_class_specific_measured_rank_fidelity(self):
        registry = L.default_lane_registry()
        selected = L.select_screening_lane(
            registry, "arithmetic", [self.calibration("cpu-q1")],
            minimum_rank_fidelity=0.8)
        self.assertEqual(selected.lane_id, "cpu-q1")
        # A calibration for another mechanism class is not transferable.
        selected = L.select_screening_lane(
            registry, "layout", [self.calibration("cpu-q1")],
            minimum_rank_fidelity=0.8)
        self.assertEqual(selected.lane_id, "cpu-full")

    def test_aa_is_necessary_but_does_not_create_transfer_evidence(self):
        registry = L.default_lane_registry()
        selected = L.select_screening_lane(
            registry, "arithmetic", [self.calibration(aa=False)],
            minimum_rank_fidelity=0.8)
        self.assertEqual(selected.lane_id, "cpu-full")
        selected = L.select_screening_lane(
            registry, "arithmetic", [], minimum_rank_fidelity=0.8)
        self.assertEqual(selected.lane_id, "cpu-full")

    def test_rank_fidelity_is_recomputed_and_rank_inversion_is_visible(self):
        self.assertEqual(L.spearman_rank_fidelity(("a", "b", "c"),
                                                  ("a", "b", "c")), 1.0)
        self.assertEqual(L.spearman_rank_fidelity(("a", "b", "c"),
                                                  ("c", "b", "a")), -1.0)
        with self.assertRaisesRegex(L.LaneError, "does not match recomputed"):
            L.RankCalibration(
                "arithmetic", "cpu-q0", "cpu-full", ("a", "b"),
                ("a", "b"), ("b", "a"), 1.0, ("ake-1",), True,
                "prediction://wrong")

    def test_op_screening_fans_out_only_on_profiled_bottleneck(self):
        registry = L.default_lane_registry()
        calibrations = [self.calibration("cpu-q0"), self.calibration("cpu-q1")]
        plan = L.plan_op_screening(
            [L.ProfiledOp("MUL_MAT", 0.6, "ake-profile"),
             L.ProfiledOp("RMS_NORM", 0.2, "ake-profile")],
            ("akc-1", "akc-2", "akc-3"), registry, calibrations,
            change_class="arithmetic", minimum_rank_fidelity=0.8)
        self.assertEqual([row.op for row in plan], ["MUL_MAT"] * 3)
        self.assertEqual([row.wave for row in plan], [0, 0, 1])
        self.assertTrue(all(row.full_verification_required for row in plan))


if __name__ == "__main__":
    unittest.main()
