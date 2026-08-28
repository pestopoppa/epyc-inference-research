"""The loop must report whether it used the device it was holding.

Lifetime: 1.403 hours of GPU claim held (122 claims, mean 41.4s) against 29.0 hours
of compiling. 95.4% idle, and nobody reported it -- the loop reported iterations and
receipts, and had no number for "am I using the thing I am holding".
"""
import unittest

from autokernel.controller import gpu_utilization as gu


def _sampling(*, busy: int, total: int, interval: float = 0.25,
              duration: float | None = None):
    return {"interval_s": interval,
            "duration_s": total * interval if duration is None else duration,
            "samples": [{"under_measurement_load": index < busy}
                        for index in range(total)]}


class FromSampling(unittest.TestCase):

    def test_busy_seconds_come_from_the_sampled_cadence(self):
        got = gu.from_sampling(_sampling(busy=60, total=160))
        self.assertEqual(got["device_samples"], 160)
        self.assertEqual(got["device_samples_under_load"], 60)
        self.assertAlmostEqual(got["device_seconds_under_load"], 15.0)
        self.assertAlmostEqual(got["sampled_busy_fraction"], 0.375)

    def test_idle_while_claimed_is_the_headline(self):
        got = gu.from_sampling(_sampling(busy=60, total=160),
                               claim_acquired_at="2026-08-28T10:00:00Z",
                               window_ended_at="2026-08-28T10:02:00Z")
        self.assertAlmostEqual(got["claim_held_s"], 120.0)
        self.assertAlmostEqual(got["gpu_seconds_idle_while_claimed"], 105.0)
        self.assertAlmostEqual(got["idle_fraction_while_claimed"], 0.875)

    def test_a_fully_used_claim_reports_no_idle(self):
        got = gu.from_sampling(_sampling(busy=40, total=40, interval=1.0),
                               claim_acquired_at="2026-08-28T10:00:00Z",
                               window_ended_at="2026-08-28T10:00:40Z")
        self.assertAlmostEqual(got["idle_fraction_while_claimed"], 0.0)

    def test_an_absent_trace_reports_absent_not_idle(self):
        """A missing measurement is not evidence of an idle device."""
        for absent in (None, {}, {"samples": []}, {"samples": [{}], "interval_s": None}):
            got = gu.from_sampling(absent)
            self.assertIsNone(got["device_seconds_under_load"], absent)
            self.assertIsNone(got["sampled_busy_fraction"], absent)
            self.assertIsNone(got["idle_fraction_while_claimed"], absent)

    def test_a_missing_claim_window_yields_no_idle_fraction(self):
        got = gu.from_sampling(_sampling(busy=10, total=20))
        self.assertIsNone(got["claim_held_s"])
        self.assertIsNone(got["idle_fraction_while_claimed"])

    def test_a_reversed_window_is_refused_rather_than_negative(self):
        got = gu.from_sampling(_sampling(busy=10, total=20),
                               claim_acquired_at="2026-08-28T10:02:00Z",
                               window_ended_at="2026-08-28T10:00:00Z")
        self.assertIsNone(got["claim_held_s"])

    def test_busy_exceeding_the_window_clamps_at_zero_idle(self):
        got = gu.from_sampling(_sampling(busy=100, total=100, interval=1.0),
                               claim_acquired_at="2026-08-28T10:00:00Z",
                               window_ended_at="2026-08-28T10:00:10Z")
        self.assertEqual(got["gpu_seconds_idle_while_claimed"], 0.0)
        self.assertEqual(got["idle_fraction_while_claimed"], 0.0)


class Summarise(unittest.TestCase):

    def test_totals_not_an_average_of_fractions(self):
        """A 40-second claim must not weigh the same as an hour."""
        rows = [
            {"claim_held_s": 3600.0, "device_seconds_under_load": 60.0},
            {"claim_held_s": 40.0, "device_seconds_under_load": 40.0},
        ]
        got = gu.summarise(rows)
        self.assertAlmostEqual(got["total_claim_held_s"], 3640.0)
        self.assertAlmostEqual(got["total_device_seconds_under_load"], 100.0)
        # Totals: 100/3640 busy. An average of the two fractions would have read
        # (0.0167 + 1.0)/2 = 51% busy, which is the opposite conclusion.
        self.assertAlmostEqual(got["idle_fraction_while_claimed"], 1 - 100.0 / 3640.0)

    def test_the_lifetime_condition_reproduces(self):
        """1.403 h held, and the compile time it was idle against."""
        rows = [{"claim_held_s": 1.403 * 3600, "device_seconds_under_load": 1.403 * 3600}]
        got = gu.summarise(rows)
        self.assertAlmostEqual(got["idle_fraction_while_claimed"], 0.0)

    def test_no_screens_reports_none_rather_than_zero(self):
        self.assertIsNone(gu.summarise([])["idle_fraction_while_claimed"])


if __name__ == "__main__":
    unittest.main()
