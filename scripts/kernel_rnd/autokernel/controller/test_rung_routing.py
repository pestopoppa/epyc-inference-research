"""R23-25 mechanism-family rung routing (pure)."""
from __future__ import annotations

import pytest

from autokernel.controller import rung_routing as rr


@pytest.mark.parametrize("text,family", [
    ("raise nwarps for the mmvq kernel", "launch_geometry"),
    ("lower VGPR pressure to improve occupancy", "occupancy"),
    ("retile the MMQ with a larger tile size", "tiling"),
])
def test_size_dependent_families_skip_the_screen(text, family):
    h = {"mechanism_id": "m", "statement": text}
    assert rr.classify(h) == (family, rr.CONFIRM)
    assert rr.skips_screen(h)


@pytest.mark.parametrize("text,family", [
    ("hoist the loop-invariant scale out of the inner loop", "redundant_work"),
    ("algebraic simplification of the rope trig", "algebraic"),
    ("fuse the up and gate projections into a single launch", "dispatch_count"),
])
def test_size_independent_families_screen_first(text, family):
    h = {"statement": text}
    assert rr.classify(h)[0] == family and rr.rung_for(h) == rr.SCREEN


def test_size_dependent_wins_when_both_are_named():
    h = {"statement": "fuse the kernels and tile size retune"}
    assert rr.classify(h) == ("tiling", rr.CONFIRM)


def test_unclassified_keeps_the_cheap_rung():
    assert rr.classify({"statement": "something novel"}) == (rr.UNCLASSIFIED, rr.SCREEN)
    assert rr.classify({}) == (rr.UNCLASSIFIED, rr.SCREEN)
    assert "screen first" in rr.route_record({})["reason"]
    assert "size-dependent" in rr.route_record({"statement": "grid size"})["reason"]
