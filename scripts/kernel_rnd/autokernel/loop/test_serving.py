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
        self.assertEqual(set(observations[0]), {
            "schema", "process_pid", "requests", "residency", "teardown", "failure"})

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


class LifecycleObservationHook(unittest.TestCase):
    class Sampler:
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

    class Observer:
        def __init__(self, events, fail=None, resolved=True):
            self.events, self.fail = events, fail
            self.shutdown_resolved = resolved

        def _call(self, name, *args):
            self.events.append((name, *args))
            if self.fail == name:
                raise RuntimeError(f"observer {name} failed")

        def start(self, phase):
            self._call("observer.start", phase)

        def attach_target(self, pid):
            self._call("observer.attach", pid)

        def phase(self, phase):
            self._call("observer.phase", phase)

        def checkpoint(self, label):
            self._call("observer.checkpoint", label)

        def finish(self):
            self._call("observer.finish")

    @staticmethod
    def _recipe():
        return serving.Recipe(name="hook", model="/m", np=1, n_predict=4)

    @staticmethod
    def _requests():
        return (("p0", json.dumps({"prompt": "fixture"}).encode()),)

    def _run(self, *, observer_fail=None, observer_resolved=True,
             popen_error=None, readback_error=None, request_error=None,
             sampler_error=None, residency_record_error=None):
        events = []
        self.last_events = events

        class Process:
            pid = 4321
            returncode = None

            def poll(self):
                return None

            def terminate(self):
                events.append(("process.terminate",))

            def wait(self, timeout):
                events.append(("process.wait", timeout))
                return 0

            def kill(self):
                events.append(("process.kill",))

        def popen(*_args, **_kwargs):
            events.append(("popen",))
            if popen_error:
                raise popen_error
            return Process()

        def urlopen(request, timeout):
            if isinstance(request, str):
                events.append(("health",))
                return self.Response(b"ok")
            phase = "warmup" if not any(row[0] == "request.warmup" for row in events) \
                else "measurement"
            events.append((f"request.{phase}",))
            if request_error:
                raise request_error
            return self.Response(json.dumps({"stop": True, "timings": {
                "predicted_n": 4, "predicted_per_second": 10.0}}).encode())

        def readback(*_args, **_kwargs):
            events.append(("readback",))
            if readback_error:
                raise readback_error
            return {}

        observer = self.Observer(events, observer_fail, observer_resolved)
        original_residency_record = serving._residency_record

        def residency_record(*args, **kwargs):
            if residency_record_error:
                raise residency_record_error
            return original_residency_record(*args, **kwargs)

        sampler_effect = sampler_error if sampler_error else lambda: self.Sampler()
        with mock.patch.object(serving.subprocess, "Popen", side_effect=popen), \
                mock.patch.object(serving.residency, "Sampler", side_effect=sampler_effect), \
                mock.patch.object(serving.urllib.request, "urlopen", side_effect=urlopen), \
                mock.patch.object(serving, "verify_env_readback", side_effect=readback), \
                mock.patch.object(serving, "_residency_record", side_effect=residency_record):
            result = serving._measure_once(
                self._recipe(), Path("/b"), 18000, frozen_requests=self._requests(),
                observation_session=observer)
        return result, events

    def test_hook_covers_setup_load_placement_health_requests_and_owned_teardown(self):
        value, events = self._run()
        self.assertEqual(value, 10.0)
        expected = [
            ("observer.start", "setup"), ("observer.phase", "load"), ("popen",),
            ("observer.attach", 4321), ("observer.phase", "placement"), ("health",),
            ("observer.phase", "health"), ("readback",),
            ("observer.phase", "warmup"), ("request.warmup",),
            ("observer.phase", "measurement"), ("request.measurement",),
            ("observer.checkpoint", "measurement_end"),
            ("observer.phase", "teardown"), ("process.terminate",),
            ("process.wait", 30), ("observer.finish",)]
        self.assertEqual(events, expected)

    def test_observer_failure_does_not_skip_cleanup_or_replace_success(self):
        value, events = self._run(observer_fail="observer.phase")
        self.assertEqual(value, 10.0)
        self.assertIn(("process.terminate",), events)
        self.assertIn(("process.wait", 30), events)
        self.assertEqual(events[-1], ("observer.finish",))

    def test_popen_failure_still_finishes_observer_and_preserves_original(self):
        with self.assertRaisesRegex(OSError, "launch failed"):
            self._run(popen_error=OSError("launch failed"),
                      residency_record_error=RuntimeError("secondary export"),
                      observer_fail="observer.finish")
        self.assertEqual(self.last_events[-1], ("observer.finish",))

    def test_readback_failure_tears_down_finishes_and_preserves_original(self):
        try:
            self._run(readback_error=serving.EnvReadbackFailed("readback-original"))
        except serving.EnvReadbackFailed as exc:
            self.assertEqual(str(exc), "readback-original")
        else:
            self.fail("readback failure was not preserved")
        self.assertLess(self.last_events.index(("observer.phase", "teardown")),
                        self.last_events.index(("process.terminate",)))
        self.assertEqual(self.last_events[-1], ("observer.finish",))

    def test_request_failure_tears_down_and_finishes_observer(self):
        with self.assertRaises(serving.ServerDied):
            self._run(request_error=OSError("request failed"))
        self.assertIn(("process.wait", 30), self.last_events)
        self.assertEqual(self.last_events[-1], ("observer.finish",))

    def test_unresolved_observer_refuses_successor_after_owned_cleanup(self):
        with self.assertRaises(serving.lifecycle_observation.ObserverShutdownUnresolved):
            self._run(observer_resolved=False)

    def test_sampler_constructor_failure_still_finishes_observer(self):
        with self.assertRaisesRegex(RuntimeError, "sampler constructor"):
            self._run(sampler_error=RuntimeError("sampler constructor"))
        self.assertEqual(self.last_events,
                         [("observer.start", "setup"), ("observer.finish",)])

    def test_residency_export_failure_follows_cleanup_and_still_finishes_observer(self):
        with self.assertRaisesRegex(RuntimeError, "residency export"):
            self._run(residency_record_error=RuntimeError("residency export"))
        self.assertIn(("process.wait", 30), self.last_events)
        self.assertEqual(self.last_events[-1], ("observer.finish",))

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
