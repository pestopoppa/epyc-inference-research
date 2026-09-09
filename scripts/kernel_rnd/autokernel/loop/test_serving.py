"""The serving-throughput measurement, tested WITHOUT launching a server.

What must hold before this gates a keep: the recipe is general over spec-decode type
(a model with no drafter emits no drafter flags), the champion recipe reproduces the
DF2-5-validated np4 command, and the paired A/B / floor arithmetic is right and
fail-closed when uncalibrated.
"""
import dataclasses
import json
from pathlib import Path
import unittest
from unittest import mock

from autokernel.loop import serving


RECIPE = serving.Recipe(
    name="t", model="/m/target.gguf",
    spec_decode={"type": "draft-dflash", "drafter": "/m/draft.gguf", "ngld": 99, "draft_n_max": 8},
    np=4, ctx=16384)


class RecipeArgv(unittest.TestCase):
    def test_the_dflash_recipe_emits_the_drafter_flags(self):
        argv = RECIPE.server_argv(Path("/B"), 18311)
        self.assertEqual(argv[argv.index("-np") + 1], "4")
        self.assertEqual(argv[argv.index("--spec-type") + 1], "draft-dflash")
        self.assertEqual(argv[argv.index("-md") + 1], "/m/draft.gguf")
        self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], "8")
        self.assertIn("--no-kv-unified", argv)

    def test_a_none_spec_recipe_emits_no_drafter_flags(self):
        plain = dataclasses.replace(RECIPE, spec_decode={"type": "none"})
        argv = plain.server_argv(Path("/B"), 1)
        self.assertNotIn("-md", argv)
        self.assertNotIn("--spec-type", argv)
        # still a valid server command
        self.assertEqual(argv[argv.index("-np") + 1], "4")

    def test_mtp_is_general_too(self):
        mtp = dataclasses.replace(RECIPE, spec_decode={"type": "draft-mtp", "drafter": "/m/mtp.gguf"})
        argv = mtp.server_argv(Path("/B"), 1)
        self.assertEqual(argv[argv.index("--spec-type") + 1], "draft-mtp")

    def test_kv_unified_flips_the_flag(self):
        self.assertIn("--kv-unified", dataclasses.replace(RECIPE, kv_unified=True).server_argv(Path("/B"), 1))

    def test_the_shipped_champion_recipe_matches_df2_5_np4(self):
        r = serving.Recipe.load(Path(__file__).resolve().parents[4]
                                / "artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json")
        argv = " ".join(r.server_argv(Path("/mnt/raid0/llm/tmp/champ2/build-hip"), 18099))
        for token in ("-np 4", "-c 16384", "--spec-type draft-dflash",
                      "--spec-draft-n-max 8", "--no-kv-unified", "-ngld 99"):
            self.assertIn(token, argv, token)


class Arithmetic(unittest.TestCase):
    def test_compare_is_fail_closed_without_a_floor(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 120.0]):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=None)
        self.assertAlmostEqual(out["effect_pct"], 20.0, places=3)
        self.assertIsNone(out["decisive"])  # uncalibrated -> never decisive

    def test_compare_is_decisive_above_the_floor(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 120.0]):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)
        self.assertTrue(out["decisive"])

    def test_a_within_floor_effect_is_not_decisive(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 100.4]):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)
        self.assertFalse(out["decisive"])

    def test_floor_is_p95_of_the_aa_spread(self):
        with mock.patch.object(serving, "_measure_once",
                               side_effect=[100.0, 101.0, 99.0, 100.5, 99.5]):
            out = serving.calibrate_floor(RECIPE, Path("/b"), samples=5)
        self.assertGreater(out["floor_pct"], 0.0)
        self.assertEqual(out["samples"], 5)


class PlannedObservationSeam(unittest.TestCase):
    def test_frozen_requests_drive_actual_launcher_seam_and_collect_slots(self):
        recipe = serving.Recipe(name="planned", model="/m", np=2, n_predict=4)
        requests = tuple((f"p{i}", json.dumps({"prompt": f"prompt-{i}"}).encode())
                         for i in range(2))

        class FakeProcess:
            pid = 4321
            returncode = None

            def poll(self):
                return None

            def terminate(self):
                return None

            def wait(self, timeout):
                return 0

            def kill(self):
                raise AssertionError("graceful fake teardown should not kill")

        class FakeSampler:
            proof = {"samples": 2, "vram_reads": 2, "resident": True,
                     "peak_vram_bytes": 2**30, "median_vram_bytes": 2**30,
                     "peak_kfd_processes": 1, "sclk_min_mhz": 1000,
                     "sclk_max_mhz": 1000, "clock_stable": True}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        class Response:
            def __init__(self, body):
                self.body = body

            def read(self):
                return self.body

        def urlopen(request, timeout):
            if isinstance(request, str):
                return Response(b"ok")
            return Response(json.dumps({"stop": True, "timings": {
                "predicted_n": 4, "predicted_per_second": 10.0}}).encode())

        observations = []
        with mock.patch.object(serving.subprocess, "Popen", return_value=FakeProcess()), \
                mock.patch.object(serving.residency, "Sampler", return_value=FakeSampler()), \
                mock.patch.object(serving.urllib.request, "urlopen", side_effect=urlopen), \
                mock.patch.object(serving, "verify_env_readback"):
            value = serving._measure_once(recipe, Path("/b"), 18000,
                                          frozen_requests=requests,
                                          observation=observations)
        self.assertEqual(value, 20.0)
        measured = [row for row in observations[0]["requests"]
                    if row["phase"] == "measurement"]
        self.assertEqual([row["prompt_id"] for row in measured],
                         ["p0", "p1"])
        self.assertTrue(all(row["terminal"] for row in observations[0]["requests"]))
        self.assertEqual(observations[0]["process_pid"], 4321)
        self.assertEqual(observations[0]["teardown"], "terminated")

    def test_partial_slot_failure_retains_warmup_and_measurement_records(self):
        recipe = serving.Recipe(name="planned", model="/m", np=2, n_predict=4)
        requests = tuple((f"p{i}", json.dumps({"prompt": f"prompt-{i}"}).encode())
                         for i in range(2))

        class FakeProcess:
            pid = 4321
            returncode = None

            def poll(self):
                return None

            def terminate(self):
                return None

            def wait(self, timeout):
                return 0

            def kill(self):
                return None

        class FakeSampler:
            proof = {"samples": 2, "vram_reads": 2, "resident": True,
                     "peak_vram_bytes": 2**30, "median_vram_bytes": 2**30,
                     "peak_kfd_processes": 1, "sclk_min_mhz": 1000,
                     "sclk_max_mhz": 1000, "clock_stable": True}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        class Response:
            def __init__(self, body):
                self.body = body

            def read(self):
                return self.body

        def urlopen(request, timeout):
            if isinstance(request, str):
                return Response(b"ok")
            if b"prompt-1" in request.data:
                raise OSError("slot failed")
            return Response(json.dumps({"stop": True, "timings": {
                "predicted_n": 4, "predicted_per_second": 10.0}}).encode())

        observations = []
        with mock.patch.object(serving.subprocess, "Popen", return_value=FakeProcess()), \
                mock.patch.object(serving.residency, "Sampler", return_value=FakeSampler()), \
                mock.patch.object(serving.urllib.request, "urlopen", side_effect=urlopen), \
                mock.patch.object(serving, "verify_env_readback"):
            with self.assertRaises(serving.ServerDied):
                serving._measure_once(recipe, Path("/b"), 18000,
                                      frozen_requests=requests,
                                      observation=observations)
        rows = observations[0]["requests"]
        self.assertEqual([(row["phase"], row["slot_index"]) for row in rows],
                         [("warmup", 0), ("warmup", 1),
                          ("measurement", 0), ("measurement", 1)])
        self.assertEqual(sum(row["error"] is not None for row in rows), 2)

        malformed = (
            {"stop": True},
            {"stop": True, "timings": {"predicted_n": True,
                                         "predicted_per_second": 10.0}},
            {"stop": True, "timings": {"predicted_n": 4,
                                         "predicted_per_second": float("inf")}},
        )
        for payload in malformed:
            with self.subTest(payload=payload):
                observations = []

                def malformed_urlopen(request, timeout):
                    return Response(b"ok" if isinstance(request, str)
                                    else json.dumps(payload).encode())

                with mock.patch.object(serving.subprocess, "Popen",
                                       return_value=FakeProcess()), \
                        mock.patch.object(serving.residency, "Sampler",
                                          return_value=FakeSampler()), \
                        mock.patch.object(serving.urllib.request, "urlopen",
                                          side_effect=malformed_urlopen), \
                        mock.patch.object(serving, "verify_env_readback"):
                    with self.assertRaises(serving.ServerDied):
                        serving._measure_once(recipe, Path("/b"), 18000,
                                              frozen_requests=requests,
                                              observation=observations)
                self.assertTrue(all(row["error"] for row
                                    in observations[0]["requests"]))
                self.assertTrue(all(row["predicted_n"] is None for row
                                    in observations[0]["requests"]))

class ServerAffinity(unittest.TestCase):
    """R23-49: the server's host threads must be PINNABLE, and unpinned must stay the
    default until the serving floor is re-calibrated under a pin.

    Kernel-verified 2026-09-07: every logical CPU on this host shares a physical core with
    0-95, the CPU campaign's bench region, so an unpinned llama-server lands on the cores
    another campaign is timing -- and its own numbers inherit that contention. The fix is
    real but it CHANGES THE MEASURED CONDITION, so it cannot be switched on silently: the
    default stays None, and a recipe that sets it declares a different measurement.
    """

    def test_unpinned_is_the_default_and_prepends_nothing(self):
        r = serving.Recipe(name="r", model="/m.gguf")
        self.assertIsNone(r.cpu_list)
        argv = r.server_argv(Path("/b"), 8080)
        self.assertTrue(argv[0].endswith("llama-server"), argv[0])
        self.assertNotIn("taskset", argv)

    def test_cpu_list_prepends_taskset_before_the_binary(self):
        r = serving.Recipe(name="r", model="/m.gguf", cpu_list="184-191")
        argv = r.server_argv(Path("/b"), 8080)
        self.assertEqual(argv[:3], ["taskset", "-c", "184-191"])
        self.assertTrue(argv[3].endswith("llama-server"), argv[3])

    def test_describe_states_the_pin_so_an_artifact_records_the_condition(self):
        self.assertIn("cpu=unpinned", serving.Recipe(name="r", model="/m").describe())
        self.assertIn("cpu=184-191",
                      serving.Recipe(name="r", model="/m", cpu_list="184-191").describe())

if __name__ == "__main__":
    unittest.main()
