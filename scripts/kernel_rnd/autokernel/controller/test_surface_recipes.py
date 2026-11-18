"""U3-DEFAULTS-b per-surface recipe pins and the measurement refusal."""
from __future__ import annotations

import pytest

from autokernel.controller import surface_recipes as sr

CPU, GPU = "c" * 64, "a" * 64


def pins():
    return sr.SurfaceRecipes.from_mapping({"gpu": GPU, "cpu": CPU})


def test_matching_pair_accepted_and_cross_surface_refused():
    p = pins()
    p.require_match("cpu", CPU)
    p.require_match("gpu", GPU)
    with pytest.raises(sr.SurfaceRecipeRefused):
        p.require_match("gpu", CPU)       # CPU recipe measured on the GPU surface
    with pytest.raises(sr.SurfaceRecipeRefused):
        p.require_match("cpu", GPU)


def test_missing_surface_is_refused_not_defaulted():
    only_cpu = sr.SurfaceRecipes.from_mapping({"cpu": CPU})
    with pytest.raises(sr.SurfaceRecipeRefused, match="no recipe for surface"):
        only_cpu.require_match("gpu", GPU)


def test_roundtrip_and_digest_stable():
    p = pins()
    record = {"surface_recipes": p.to_dict()}
    q = sr.SurfaceRecipes.from_record(record)
    assert q == p and q.digest() == p.digest()
    other = sr.SurfaceRecipes.from_mapping({"cpu": CPU, "gpu": "b" * 64})
    assert other.digest() != p.digest()


@pytest.mark.parametrize("bad", [
    {}, {"cpu": "short"}, {"npu": CPU}, {"cpu": CPU.upper()}, {"cpu": 3},
])
def test_malformed_pins_refused(bad):
    with pytest.raises(sr.SurfaceRecipeRefused):
        sr.SurfaceRecipes.from_mapping(bad)


def test_record_without_block_or_with_extras_refused():
    with pytest.raises(sr.SurfaceRecipeRefused):
        sr.SurfaceRecipes.from_record({})
    block = pins().to_dict()
    block["extra"] = 1
    with pytest.raises(sr.SurfaceRecipeRefused):
        sr.SurfaceRecipes.from_record({"surface_recipes": block})
    with pytest.raises(sr.SurfaceRecipeRefused):
        sr.SurfaceRecipes(tuple([("gpu", GPU), ("cpu", CPU)]))   # unsorted
