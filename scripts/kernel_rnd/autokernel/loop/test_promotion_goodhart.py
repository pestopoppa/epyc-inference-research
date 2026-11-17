import pytest

from . import goodhart_count as gc
from . import promotion_evidence as pe

C, H = "a" * 40, "b" * 64


def L(i, e=0.02, c=C, h=H):
    return {"pair_id": i, "effect": e, "production_commit": c, "recipe_hash": h}


def test_compounds_disjoint_heldout_launches():
    r = pe.compound_from_heldout(launches=[L("h1", 0.10), L("h2", 0.10)], deciding_pair_ids=["d1"],
                                 production_commit=C, recipe_hash=H, min_launches=2)
    assert r["compounded_gain"] == pytest.approx(0.21) and r["deciding_pair_ids"] == ["d1"]


@pytest.mark.parametrize("kw,msg", [
    (dict(launches=[L("d1")], deciding_pair_ids=["d1"]), "reuses"),
    (dict(launches=[L("h1"), L("h1")], deciding_pair_ids=["d1"]), "duplicate"),
    (dict(launches=[L("h1")], deciding_pair_ids=[]), "deciding pair ids"),
    (dict(launches=[L("h1", c="c" * 40)], deciding_pair_ids=["d"]), "not pinned"),
    (dict(launches=[L("h1", h="c" * 64)], deciding_pair_ids=["d"]), "not pinned"),
    (dict(launches=[L("h1", e=float("nan"))], deciding_pair_ids=["d"]), "invalid effect"),
    (dict(launches=[L("h1")], deciding_pair_ids=["d"], min_launches=2), "required 2"),
])
def test_refusals(kw, msg):
    with pytest.raises(pe.PromotionEvidenceRefusal, match=msg):
        pe.compound_from_heldout(production_commit=C, recipe_hash=H, **kw)


def test_unpinned_production_refuses():
    with pytest.raises(pe.PromotionEvidenceRefusal):
        pe.compound_from_heldout(launches=[L("h")], deciding_pair_ids=["d"], production_commit="HEAD", recipe_hash=H)


FULL = {"serving": 100.0, "rebaselined": 90.0}


def A(k, e, **loo):
    return {"keep_id": k, "keep_time_effect": e, "loo": loo}


def test_counts_keeps_whose_loo_arm_is_at_least_as_fast_on_any_surface():
    r = gc.count_goodharted([A("k1", 0.03, serving=95.0, rebaselined=85.0),
                             A("k2", 0.02, serving=100.0, rebaselined=80.0),   # tie counts
                             A("k3", 0.01, serving=90.0, rebaselined=91.0)],   # re-baselined surface convicts
                            full_bundle=FULL)
    assert r["goodharted_count"] == 2 and r["goodharted_keep_ids"] == ["k2", "k3"]
    assert [a["keep_time_effect"] for a in r["arms"]] == [0.03, 0.02, 0.01]
    assert r["arms"][2]["surfaces"] == ["rebaselined"]


@pytest.mark.parametrize("arms", [[], [A("k", 0.1)], [{"keep_time_effect": 0.1, "loo": {"serving": 1}}],
                                  [A("k", 0.1, other=1.0)], [A("k", None, serving=1.0)],
                                  [A("k", 0.1, serving=float("nan"))]])
def test_malformed_input_refuses(arms):
    with pytest.raises(ValueError):
        gc.count_goodharted(arms, full_bundle=FULL)
