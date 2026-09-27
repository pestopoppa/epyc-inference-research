"""Hardware-free tests for the independent RMS_NORM oracle (cpu_norm_rowsplit route)."""
from array import array
import math
import re
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import cpu_norm_reference as fixture


SMALL = fixture.NormCase("small_narrow", "plain", (300, 3, 1, 1), 1e-6)
SMALL_FUSED = fixture.NormCase("small_fused", "fused", (200, 3, 1, 1), 1e-6)
SMALL_VIEW = fixture.NormCase("small_view", "view", (100, 2, 1, 1), 1e-6)


def probe_output(case, rows_out=None, digests=None, input_hash=None, header=None):
    """What an exact candidate probe prints, optionally with substituted rows."""
    rows, weights, digest = fixture._inputs(case)
    expected = [fixture.head_semantics(rows[r], case.eps,
                                       weights[r] if weights is not None else None)
                for r in range(case.rows)]
    rows_out = expected if rows_out is None else rows_out
    want = fixture._fnv1a(b"".join(row.tobytes() for row in rows_out))
    lines = [header or (f"{fixture.MARKER} {case.mode} {' '.join(map(str, case.ne))} "
                        f"{fixture._f32_bits(case.eps):08x} {fixture.THREADS} {case.reps} "
                        f"{fixture.SEED}"),
             f"I {digest if input_hash is None else input_hash:016x}"]
    for rep in range(case.reps):
        lines.append(f"D {rep} {(want if digests is None else digests[rep]):016x}")
    for r, row in enumerate(rows_out):
        lines.append(f"O {r} {row.tobytes().hex()}")
    return "\n".join(lines) + "\n", rows, weights, expected


def f32(values):
    return array("f", values)


def head_scale(row, eps):
    """HEAD's per-row float scale, as `head_semantics` forms it."""
    total = 0.0
    for square in f32([x * x for x in row]):
        total += square
    mean = fixture._f32(total / len(row))
    return fixture._f32(1.0 / fixture._f32(math.sqrt(fixture._f32(mean + fixture._f32(eps)))))


class ExactCandidatePasses(unittest.TestCase):
    def test_exact_output_passes_for_every_mode(self):
        for case in (SMALL, SMALL_FUSED, SMALL_VIEW,
                     fixture.NormCase("s_inplace", "inplace", (64, 2, 1, 1), 1e-6, reps=5),
                     fixture.NormCase("s_bcast", "fused_bcast", (64, 3, 2, 1), 0.5)):
            output, *_ = probe_output(case)
            result = fixture.compare(case, output)
            self.assertEqual(result.status, "pass", (case.name, result.reason))
            self.assertIn('"bit_identical": true', result.detail)

    def test_generator_is_exact_and_spans_the_intended_magnitudes(self):
        values = [fixture._value(fixture.SEED, i, 12, 25) for i in range(4000)]
        self.assertEqual(list(f32(values)), values)          # already float32
        exponents = {math.frexp(abs(v))[1] - 1 for v in values}
        self.assertEqual(min(exponents), -12)
        self.assertEqual(max(exponents), 12)
        self.assertTrue(any(v < 0 for v in values) and any(v > 0 for v in values))
        # every significand bit is live: the fixture is not a set of "nice" dyadics
        self.assertGreater(sum(math.frexp(v)[0] * 2 ** 24 % 2 == 1 for v in values), 1500)

    def test_head_semantics_stays_inside_the_float64_bound(self):
        for case in (SMALL, SMALL_FUSED, fixture.CASES[0]):
            rows, weights, _ = fixture._inputs(case)
            for r in range(case.rows):
                weight = weights[r] if weights is not None else None
                head = fixture.head_semantics(rows[r], case.eps, weight)
                ref = fixture.float64_reference(rows[r], case.eps, weight)
                worst = max(abs(a - b) / abs(b) for a, b in zip(head, ref))
                self.assertLess(worst, fixture.REL_BOUND / 2, case.name)

    def test_probe_constants_match_the_reference(self):
        probe = fixture.PROBE.read_text(encoding="utf-8")
        self.assertIn(f"VIEW_PAD = {fixture.VIEW_PAD};", probe)
        self.assertIn(f"VIEW_OFFSET = {fixture.VIEW_OFFSET};", probe)
        for constant in ("0x9E3779B97F4A7C15", "0xBF58476D1CE4E5B9", "0x94D049BB133111EB",
                         "0x100000001B3", "0xCBF29CE484222325"):
            self.assertIn(constant, probe)
        self.assertIn("value((uint64_t) seed, (uint64_t) i, 12, 25)", probe)
        self.assertIn("value((uint64_t) seed + 1, (uint64_t) i, 4, 9)", probe)


class WrongCandidatesAreWrong(unittest.TestCase):
    """Each mutation is a plausible split bug; each must be `wrong`, never a pass."""

    def _wrong(self, case, rows_out):
        output, *_ = probe_output(case, rows_out=rows_out)
        result = fixture.compare(case, output)
        self.assertEqual(result.status, "wrong")
        return result

    def test_float_accumulator(self):
        output, rows, _w, expected = probe_output(SMALL)
        bad = []
        for row in rows:
            total = f32([0.0])[0]
            for x in row:
                total = f32([total + f32([x * x])[0]])[0]
            scale = fixture._f32(1.0 / fixture._f32(math.sqrt(
                fixture._f32(fixture._f32(total / len(row)) + fixture._f32(SMALL.eps)))))
            bad.append(f32([x * scale for x in row]))
        self.assertNotEqual([r.tobytes() for r in bad], [r.tobytes() for r in expected])
        self.assertIn("not bit-identical", self._wrong(SMALL, bad).reason)

    def test_divide_instead_of_reciprocal_scale(self):
        _o, rows, _w, _e = probe_output(SMALL)
        bad = []
        for row in rows:
            total = 0.0
            for square in f32([x * x for x in row]):
                total += square
            denom = fixture._f32(math.sqrt(fixture._f32(fixture._f32(total / len(row)) +
                                                       fixture._f32(SMALL.eps))))
            bad.append(f32([x / denom for x in row]))
        self._wrong(SMALL, bad)

    def test_fused_product_reassociated(self):
        _o, rows, weights, _e = probe_output(SMALL_FUSED)
        bad = []
        for row, weight in zip(rows, weights):
            scale = head_scale(row, SMALL_FUSED.eps)
            bad.append(f32([x * fixture._f32(scale * w) for x, w in zip(row, weight)]))
        self._wrong(SMALL_FUSED, bad)

    def test_segment_skipped_duplicated_or_unscaled(self):
        _o, rows, _w, expected = probe_output(SMALL)
        skipped = [f32(r) for r in expected]
        skipped[1][100:150] = f32([0.0] * 50)
        duplicated = [f32(r) for r in expected]
        duplicated[2][150:200] = duplicated[2][100:150]
        stale = [f32(r) for r in expected]
        stale[0][250:300] = f32(rows[0][250:300])     # in-place race: unscaled input
        for bad in (skipped, duplicated, stale):
            self._wrong(SMALL, bad)

    def test_one_ulp_in_one_element(self):
        _o, _r, _w, expected = probe_output(SMALL)
        bad = [f32(r) for r in expected]
        bits = fixture._f32_bits(bad[2][299]) ^ 1
        bad[2][299] = array("f", array("I", [bits]).tobytes())[0]
        result = self._wrong(SMALL, bad)
        self.assertIn("row=2 column=299", result.reason)
        self.assertIn("1 of 300 columns differ", result.reason)

    def test_later_repetition_differs(self):
        case = fixture.NormCase("race", "inplace", (64, 2, 1, 1), 1e-6, reps=4)
        good, *_ = probe_output(case)
        want = int(re.search(r"^D 0 ([0-9a-f]+)$", good, re.M).group(1), 16)
        output, *_ = probe_output(case, digests=[want, want, want ^ 1, want])
        result = fixture.compare(case, output)
        self.assertEqual(result.status, "wrong")
        self.assertIn("repetitions [2] of 4", result.reason)


class UntrustedOutputIsUnavailable(unittest.TestCase):
    def test_identity_drift_and_truncation_raise(self):
        output, *_ = probe_output(SMALL)
        for broken in (output.replace(f"{fixture.MARKER} plain 300", f"{fixture.MARKER} plain 301"),
                       probe_output(SMALL, input_hash=1)[0],
                       "\n".join(output.splitlines()[:-1]) + "\n",
                       output + "O 0 00\n",
                       output + "X junk\n",
                       output.replace(fixture.MARKER, "NOPE")):
            with self.assertRaises(ValueError):
                fixture.compare(SMALL, broken)

    def test_suite_maps_infrastructure_to_unavailable(self):
        missing = fixture.check_rms_norm_suite(Path("/nonexistent/build"), Path("/nonexistent"))
        self.assertEqual(missing.status, "unavailable")
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(subprocess, "run", side_effect=[
                 subprocess.CompletedProcess([], 0, "", ""),
                 subprocess.CompletedProcess([], 0, "garbage\n", "")]):
            bad = fixture.check_rms_norm_suite(Path("/build"), Path("/source"), cases=(SMALL,))
        self.assertEqual((bad.status, bad.reason), ("unavailable",
                                                    "small_narrow probe output invalid"))

    def test_compile_uses_toolchain_env_and_run_uses_launch_env(self):
        output, *_ = probe_output(SMALL)
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.dict("os.environ", {"PATH": "/usr/bin:/bin"}), \
             mock.patch.object(subprocess, "run", side_effect=[
                 subprocess.CompletedProcess([], 0, "", ""),
                 subprocess.CompletedProcess([], 0, output, "")]) as run:
            result = fixture.check_rms_norm_suite(
                Path("/build"), Path("/source"), cases=(SMALL,),
                launch_env={"OMP_NUM_THREADS": "48"}, topology_prefix=("taskset", "-c", "0-7"))
        self.assertEqual(result.status, "pass", result.reason)
        compile_call, run_call = run.call_args_list
        self.assertEqual(compile_call.kwargs["env"].get("PATH"), "/usr/bin:/bin")
        self.assertNotIn("PATH", run_call.kwargs["env"])
        self.assertTrue(run_call.kwargs["env"]["LD_LIBRARY_PATH"].startswith("/build/bin"))
        argv = run_call.args[0]
        self.assertEqual(argv[:3], ["taskset", "-c", "0-7"])
        self.assertEqual(argv[4:], ["plain", "300", "3", "1", "1",
                                    f"{fixture._f32_bits(1e-6):08x}", "8", "3",
                                    str(fixture.SEED)])


class CaseCoverage(unittest.TestCase):
    def test_cases_exercise_split_fusion_inplace_and_row_paths(self):
        cases = fixture.CASES
        self.assertEqual(fixture.THREADS, 8)
        narrow = [c for c in cases if c.rows < fixture.THREADS]
        wide = [c for c in cases if c.rows >= fixture.THREADS]
        # the DS41 hc_mixes shapes, split across more threads than rows
        self.assertTrue({(20480, 2, 1, 1), (20480, 3, 1, 1)} <=
                        {c.ne for c in narrow if c.mode == "plain"})
        self.assertEqual({c.mode for c in narrow}, set(fixture.MODES))
        self.assertTrue(any(c.ne[2] > 1 for c in narrow))            # 3-D row walk
        self.assertTrue(any(c.ne[0] % 16 for c in narrow))           # ragged segments
        self.assertTrue(any(c.mode == "inplace" and c.reps >= 16 for c in narrow))
        self.assertTrue(any(c.eps >= 0.1 for c in cases))
        self.assertTrue(any(c.mode == "plain" and c.ne[3] > 1 for c in wide))
        self.assertTrue(any(c.mode.startswith("fused") for c in wide))
        self.assertTrue(any(c.rows == 1 and c.ne[0] <= 4096 for c in cases))  # tiny-solo
        self.assertEqual(len({c.name for c in cases}), len(cases))


if __name__ == "__main__":
    unittest.main()
