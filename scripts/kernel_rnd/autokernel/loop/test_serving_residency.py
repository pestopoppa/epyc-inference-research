"""R23-60: the SERVING path must prove GPU residency, the way the bench path does.

WHY THIS FILE EXISTS. `bench.run_once` has sampled residency across every llama-bench
invocation since the rebuild; `serving._measure_once` sampled nothing. So every serving
number this campaign took -- the 4.581% floor and the gate reading that held the champion
included -- was un-PROVEN as GPU-resident. That is a MISSING PROOF, not a suspected
defect, and it is the one class of defect that cannot be repaired afterwards: a residency
tuple invented on read claims warrant the original run never captured.

Nothing here launches a server. The sampler is stubbed and every sample is synthetic.

THE CONTROL. `TheEvidenceChecksAreNotVacuous` runs the SAME assertion helper the positive
tests use against a row produced by the pre-R23-60 shape (a `_measure_once` that returns a
bare float and records nothing) and requires it to FAIL. A test that passes on missing
data is precisely the failure mode this task exists to prevent, so the helper's ability to
fail is itself under test.
"""
import dataclasses
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from autokernel.loop import residency, serving


RECIPE = serving.Recipe(name="rt", model="/m/t.gguf", np=2, n_predict=8)

#: A launch record's residency window, for the stubbed `_measure_once`. Enclosing, so a
#: stub cannot accidentally manufacture the covering property the real path must earn.
STUB_WINDOW = {"window_start": 1.0, "window_end": 10.0,
               "request_start": 3.0, "request_end": 8.0}


def _proof(*, peak: int = 8 << 30, median: int | None = None, samples: int = 40,
           vram_reads: int | None = None, kfd: int = 1) -> dict:
    """A synthetic `residency.Sampler.proof`, in the shape the real sampler emits."""
    median = peak if median is None else median
    vram_reads = samples if vram_reads is None else vram_reads
    return {"peak_vram_bytes": peak, "median_vram_bytes": median,
            "vram_reads": vram_reads, "peak_kfd_processes": kfd,
            "sclk_min_mhz": 1700, "sclk_max_mhz": 1700, "clock_stable": True,
            "samples": samples,
            "resident": peak >= residency.RESIDENT_FLOOR_BYTES}


def _sampler_class(proof: dict, made: list | None = None):
    """A stand-in for `residency.Sampler`: no thread, no sysfs, synthetic samples."""

    class _Stub:
        def __init__(self) -> None:
            self._proof = dict(proof)
            if made is not None:
                made.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        @property
        def proof(self) -> dict:
            return dict(self._proof)

    return _Stub


def _record(proof: dict) -> dict:
    """One per-launch residency record built from a synthetic proof."""
    return serving._residency_record(_sampler_class(proof)(), **STUB_WINDOW)


def _stub_measure(values, proof: dict | None = None):
    """A `_measure_once` that records evidence the way the real one does."""
    body = _proof() if proof is None else proof
    it = iter(values)

    def _measure(recipe, build_dir, port, boot_timeout_s=360, *, evidence=None):
        if evidence is not None:
            evidence.append(_record(body))
        return next(it)

    return _measure


def _launch(proof: dict, *, recipe: serving.Recipe = RECIPE,
            evidence: list | None = None, verify_error: Exception | None = None,
            made: list | None = None):
    """Drive the real `_measure_once` with a fake server and a stubbed sampler.

    Returns `(aggregate_tok_s, popen_wall_time)` -- the second value is what proves the
    residency window opened BEFORE the process that loads the model.
    """
    launched: dict = {}

    class _Proc:
        pid = 4321
        returncode = 0

        def poll(self):
            return None

        def terminate(self):
            launched["terminated"] = True

        def wait(self, _timeout=None):
            return 0

    def _popen(argv, **_kw):
        launched["at"] = time.time()
        return _Proc()

    class _Resp:
        def read(self):
            return json.dumps({"timings": {"predicted_n": recipe.n_predict,
                                           "predicted_per_second": 25.0}}).encode()

    def _urlopen(_req, timeout=None, **_kw):
        return _Resp()

    stack = [mock.patch.object(serving.subprocess, "Popen", _popen),
             mock.patch.object(serving.urllib.request, "urlopen", _urlopen),
             mock.patch.object(serving.residency, "Sampler",
                               _sampler_class(proof, made))]
    if verify_error is not None:
        stack.append(mock.patch.object(serving, "verify_env_readback",
                                       side_effect=verify_error))
    with stack[0], stack[1], stack[2]:
        if verify_error is not None:
            with stack[3]:
                value = serving._measure_once(recipe, Path("/B"), 18311,
                                              evidence=evidence)
        else:
            value = serving._measure_once(recipe, Path("/B"), 18311, evidence=evidence)
    return value, launched["at"]


def _assert_proven(case: unittest.TestCase, block) -> None:
    """THE evidence check. Used by every positive test AND by the control that requires
    it to fail on an absent block -- if this helper can pass on missing data, every test
    below is decorative."""
    case.assertEqual(block.get("status"), serving.RESIDENCY_PROVEN)
    case.assertTrue(block.get("covers_request_phase"))
    case.assertGreaterEqual(block.get("peak_vram_bytes") or 0,
                            residency.RESIDENT_FLOOR_BYTES)
    case.assertGreaterEqual(block.get("median_vram_bytes") or 0,
                            residency.RESIDENT_FLOOR_BYTES)
    case.assertGreaterEqual(block.get("peak_kfd_processes") or 0, 1)
    case.assertGreater(block.get("samples") or 0, 0)
    case.assertIsNotNone(block.get("window_start"))
    case.assertIsNotNone(block.get("window_end"))


class ThePerLaunchRecord(unittest.TestCase):
    def test_a_launch_records_what_the_bench_path_records(self):
        evidence: list = []
        value, _ = _launch(_proof(), evidence=evidence)
        self.assertEqual(value, 50.0)                      # np=2 x 25.0 t/s, unchanged
        self.assertEqual(len(evidence), 1)
        _assert_proven(self, evidence[0])
        self.assertTrue(evidence[0]["sampled"])
        self.assertEqual(evidence[0]["resident_floor_bytes"],
                         residency.RESIDENT_FLOOR_BYTES)

    def test_the_window_opens_before_the_server_process_and_encloses_the_requests(self):
        """The window must overlap the actual LOAD and the request phase, not the boot."""
        evidence: list = []
        _, popen_at = _launch(_proof(), evidence=evidence)
        rec = evidence[0]
        self.assertLessEqual(rec["window_start"], popen_at)
        self.assertLessEqual(rec["window_start"], rec["request_start"])
        self.assertLessEqual(rec["request_start"], rec["request_end"])
        self.assertLessEqual(rec["request_end"], rec["window_end"])
        self.assertTrue(serving.covers_request_phase(rec))

    def test_exactly_one_sampler_is_created_per_launch(self):
        """R23-60 makes serving.py THE sampler. A second one here would double-sample."""
        made: list = []
        _launch(_proof(), made=made)
        self.assertEqual(len(made), 1)

    def test_a_failed_launch_still_leaves_its_window_on_the_record(self):
        evidence: list = []
        with self.assertRaises(serving.EnvReadbackFailed):
            _launch(_proof(), evidence=evidence,
                    verify_error=serving.EnvReadbackFailed("readback"))
        self.assertEqual(len(evidence), 1)
        self.assertIsNone(evidence[0]["request_start"])
        self.assertEqual(evidence[0]["status"], serving.RESIDENCY_UNPROVEN)


class AWindowThatMissesThePhenomenon(unittest.TestCase):
    """A sample whose window does not overlap the phenomenon proves nothing about it --
    which is exactly why a post-hoc 0% VRAM reading is the NORMAL result on a finished
    llama-bench and is not evidence of a CPU run."""

    def test_non_coverage_is_detectable_from_the_recorded_timestamps_alone(self):
        self.assertTrue(serving.covers_request_phase(
            {"window_start": 100.0, "window_end": 140.0,
             "request_start": 120.0, "request_end": 130.0}))
        self.assertFalse(serving.covers_request_phase(       # window ends too early
            {"window_start": 100.0, "window_end": 110.0,
             "request_start": 120.0, "request_end": 130.0}))
        self.assertFalse(serving.covers_request_phase(       # partial overlap only
            {"window_start": 100.0, "window_end": 125.0,
             "request_start": 120.0, "request_end": 130.0}))
        self.assertFalse(serving.covers_request_phase(       # opens after the requests
            {"window_start": 121.0, "window_end": 140.0,
             "request_start": 120.0, "request_end": 130.0}))
        self.assertFalse(serving.covers_request_phase({}))   # unknown is not covering

    def test_a_non_covering_window_cannot_be_proven(self):
        rec = serving._residency_record(_sampler_class(_proof())(),
                                        window_start=100.0, window_end=110.0,
                                        request_start=120.0, request_end=130.0)
        self.assertEqual(rec["status"], serving.RESIDENCY_UNPROVEN)

    def test_an_empty_reading_over_a_non_covering_window_is_not_an_abort(self):
        """Absence measured outside the phenomenon is not evidence of absence, so it
        must NOT trigger the refusal -- it is `unproven`, like any other blind window."""
        rec = serving._residency_record(_sampler_class(_proof(peak=0))(),
                                        window_start=100.0, window_end=110.0,
                                        request_start=120.0, request_end=130.0)
        self.assertEqual(rec["status"], serving.RESIDENCY_UNPROVEN)
        serving._refuse_if_not_resident(RECIPE, rec)         # does not raise


class TheFailurePolicy(unittest.TestCase):
    def test_an_unsampleable_launch_records_unproven_and_does_not_claim_proof(self):
        """An unreadable sysfs node is an INSTRUMENT fault, not evidence about the run.
        It is recorded, never silently passed as proven, and never aborts the launch."""
        evidence: list = []
        value, _ = _launch(_proof(peak=0, median=0, vram_reads=0), evidence=evidence)
        self.assertEqual(value, 50.0)                        # the number still returns
        rec = evidence[0]
        self.assertEqual(rec["status"], serving.RESIDENCY_UNPROVEN)
        self.assertFalse(rec["sampled"])
        self.assertNotEqual(rec["status"], serving.RESIDENCY_PROVEN)
        with self.assertRaises(AssertionError):              # and cannot pass the check
            _assert_proven(self, rec)

    def test_a_launch_sampled_non_resident_aborts(self):
        """Zero VRAM throughout a window that DID cover the request phase is positive
        evidence the launch was not on the device. Unrecoverable after the fact, so it
        refuses rather than reports."""
        evidence: list = []
        with self.assertRaises(serving.ServingNotResident) as ctx:
            _launch(_proof(peak=0, median=0), evidence=evidence)
        self.assertIn("RESIDENCY REFUTED", str(ctx.exception))
        self.assertEqual(len(evidence), 1)                   # the window is still filed
        self.assertEqual(evidence[0]["status"], serving.RESIDENCY_UNPROVEN)

    def test_the_residency_refusal_is_distinguishable_from_the_env_readback_refusal(self):
        with self.assertRaises(serving.ServingNotResident) as res:
            _launch(_proof(peak=0, median=0))
        readback = dataclasses.replace(
            RECIPE, env={"GGML_NOHUGEPAGE_PROCESS": "1"},
            env_readback=({"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS",
                           "expect": {"1": "0", "unset": "1"}},))
        with self.assertRaises(serving.EnvReadbackFailed) as env:
            serving.verify_env_readback(readback, 1, status_text="THP_enabled:\t1")
        # By CLASS, in both directions...
        self.assertNotIsInstance(res.exception, serving.EnvReadbackFailed)
        self.assertNotIsInstance(env.exception, serving.ServingNotResident)
        self.assertFalse(issubclass(serving.ServingNotResident,
                                    serving.EnvReadbackFailed))
        # ...and by MESSAGE, for a human or a triage script reading a log.
        self.assertIn("RESIDENCY REFUTED", str(res.exception))
        self.assertNotIn("RESIDENCY REFUTED", str(env.exception))
        self.assertIn("env", str(env.exception))


class TheRowsCarryTheEvidence(unittest.TestCase):
    def test_calibrate_floor_carries_residency_and_leaves_the_floor_alone(self):
        runs = [100.0, 101.0, 99.0, 100.5, 99.5]
        with mock.patch.object(serving, "_measure_once", side_effect=_stub_measure(runs)):
            row = serving.calibrate_floor(RECIPE, Path("/b"), samples=5)
        _assert_proven(self, row["residency"])
        self.assertEqual(row["residency"]["invocations"], 5)
        self.assertEqual(row["residency"]["proven"], 5)
        self.assertEqual(len(row["launch_residency"]), 5)
        # The floor arithmetic is untouched: same value the spread reports, same runs.
        self.assertEqual(row["floor_pct"], serving._spread(runs)["p95_dev_pct"])
        self.assertEqual(row["runs"], runs)

    def test_compare_carries_residency_per_arm_and_leaves_the_verdict_alone(self):
        with mock.patch.object(serving, "_measure_once",
                               side_effect=_stub_measure([100.0, 120.0])):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)
        _assert_proven(self, out["residency"])
        self.assertEqual(out["residency"]["invocations"], 2)
        self.assertEqual(len(out["anchor_residency"]), 1)
        self.assertEqual(len(out["candidate_residency"]), 1)
        _assert_proven(self, out["anchor_residency"][0])
        _assert_proven(self, out["candidate_residency"][0])
        # Decision arithmetic unchanged.
        self.assertAlmostEqual(out["effect_pct"], 20.0, places=3)
        self.assertTrue(out["decisive"])
        self.assertEqual(out["schema"], "epyc.autokernel.serving_ab.v1")

    def test_one_unproven_launch_makes_the_whole_row_unproven(self):
        """A row is a claim about ALL of its launches."""
        proofs = [_record(_proof()), _record(_proof(vram_reads=0))]
        self.assertEqual(serving._residency_fold(proofs)["status"],
                         serving.RESIDENCY_UNPROVEN)
        self.assertEqual(serving._residency_fold(proofs)["proven"], 1)
        self.assertEqual(serving._residency_fold(proofs)["invocations"], 2)


class TheFloorFile(unittest.TestCase):
    def test_a_written_floor_records_that_its_launches_were_proven(self):
        with mock.patch.object(serving, "_measure_once",
                               side_effect=_stub_measure([100.0] * 5)):
            row = serving.calibrate_floor(RECIPE, Path("/b"), samples=5)
        with tempfile.TemporaryDirectory() as tmp:
            body = json.loads(serving.write_floor(tmp, RECIPE, row).read_text())
            reading = serving.load_floor(tmp, RECIPE)
        _assert_proven(self, body["residency"])
        self.assertEqual(body["residency"]["invocations"], 5)
        self.assertEqual(reading.residency_status, serving.RESIDENCY_PROVEN)
        self.assertEqual(reading.provenance, "verified")

    def test_a_row_with_no_evidence_is_stamped_unproven_rather_than_left_silent(self):
        row = {"schema": "epyc.autokernel.serving_floor.v1", "recipe": RECIPE.name,
               "recipe_hash": RECIPE.recipe_hash, "floor_pct": 4.581,
               "runs": [100.0], "median_tok_s": 100.0}
        with tempfile.TemporaryDirectory() as tmp:
            body = json.loads(serving.write_floor(tmp, RECIPE, row).read_text())
            reading = serving.load_floor(tmp, RECIPE)
        self.assertEqual(body["residency"]["status"], serving.RESIDENCY_UNPROVEN)
        self.assertEqual(body["residency"]["invocations"], 0)
        self.assertIn("cannot be added afterwards", body["residency"]["note"])
        self.assertEqual(reading.residency_status, serving.RESIDENCY_UNPROVEN)
        self.assertEqual(body["floor_pct"], 4.581)            # value untouched

    def test_a_legacy_floor_file_with_no_residency_block_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = serving.floor_path(tmp, RECIPE)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"schema": "epyc.autokernel.serving_floor.v1", "recipe": RECIPE.name,
                 "recipe_hash": RECIPE.recipe_hash, "floor_pct": 4.581,
                 "runs": [100.0, 101.0]}), encoding="utf-8")
            reading = serving.load_floor(tmp, RECIPE)
        self.assertEqual(reading.provenance, "verified")
        self.assertEqual(reading.floor_pct, 4.581)
        self.assertEqual(reading.residency_status, serving.RESIDENCY_UNPROVEN)

    def test_an_unstamped_legacy_file_is_still_grandfathered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = serving.floor_path(tmp, RECIPE)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"floor_pct": 4.581}), encoding="utf-8")
            reading = serving.load_floor(tmp, RECIPE)
        self.assertEqual(reading.provenance, "unverified")
        self.assertEqual(reading.floor_pct, 4.581)
        self.assertEqual(reading.residency_status, serving.RESIDENCY_UNPROVEN)


class TheEvidenceChecksAreNotVacuous(unittest.TestCase):
    """THE CONTROL. Every assertion above must FAIL when the evidence is absent.

    A test that passes on missing data is the exact failure this whole change exists to
    prevent -- it would report the serving path as proven while proving nothing.
    """

    def test_the_shared_checker_fails_on_an_empty_block(self):
        with self.assertRaises(AssertionError):
            _assert_proven(self, {})

    def test_the_shared_checker_fails_on_each_field_individually(self):
        good = _record(_proof())
        for field in ("status", "covers_request_phase", "peak_vram_bytes",
                      "median_vram_bytes", "peak_kfd_processes", "samples",
                      "window_start", "window_end"):
            with self.subTest(field=field):
                missing = {k: v for k, v in good.items() if k != field}
                with self.assertRaises(AssertionError):
                    _assert_proven(self, missing)

    def test_the_pre_r2360_shape_produces_a_row_the_checker_refuses(self):
        """A `_measure_once` that returns a bare float and records nothing -- the serving
        path exactly as it was before R23-60."""
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 120.0]):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)
        self.assertEqual(out["anchor_residency"], [])
        self.assertEqual(out["candidate_residency"], [])
        self.assertEqual(out["residency"]["invocations"], 0)
        with self.assertRaises(AssertionError):
            _assert_proven(self, out["residency"])

    def test_a_floor_from_the_pre_r2360_shape_is_refused_by_the_checker_too(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0] * 5):
            row = serving.calibrate_floor(RECIPE, Path("/b"), samples=5)
        with tempfile.TemporaryDirectory() as tmp:
            body = json.loads(serving.write_floor(tmp, RECIPE, row).read_text())
        with self.assertRaises(AssertionError):
            _assert_proven(self, body["residency"])
        self.assertEqual(body["residency"]["status"], serving.RESIDENCY_UNPROVEN)


if __name__ == "__main__":
    unittest.main()
