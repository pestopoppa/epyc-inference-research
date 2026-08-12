#!/usr/bin/env python3
"""Tests for sampled-claim teardown ordering."""
from __future__ import annotations

import unittest

from scripts.benchmark.autokernel_claimed_sampling import stop_sampler_and_release


class FakeSampler:
    def __init__(self, error=None):
        self.error = error

    def stop(self):
        if self.error is not None:
            raise self.error
        return "sampled"


class FakeClaim:
    def __init__(self, error=None):
        self.error = error
        self.called = False

    def release(self):
        self.called = True
        if self.error is not None:
            raise self.error
        return "released"


class ClaimedSamplingTest(unittest.TestCase):
    def test_sampler_failure_cannot_skip_release(self):
        claim = FakeClaim()
        sampled, released, errors = stop_sampler_and_release(
            sampler=FakeSampler(RuntimeError("cadence")), claim=claim)
        self.assertIsNone(sampled)
        self.assertEqual(released, "released")
        self.assertTrue(claim.called)
        self.assertEqual(str(errors[0]), "cadence")

    def test_release_failure_is_retained_after_sampling(self):
        sampled, released, errors = stop_sampler_and_release(
            sampler=FakeSampler(), claim=FakeClaim(RuntimeError("release")))
        self.assertEqual(sampled, "sampled")
        self.assertIsNone(released)
        self.assertEqual(str(errors[0]), "release")


if __name__ == "__main__":
    unittest.main()
