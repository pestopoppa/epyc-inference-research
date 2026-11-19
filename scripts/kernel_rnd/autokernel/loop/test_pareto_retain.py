from __future__ import annotations

from . import pareto_retain as pr
from . import test_cross_target as tct
from .test_cross_target import world, _hold, _gate  # noqa: F401  (fixtures)
from . import cross_target as ct


def _c(cid, wid, q, s, kept=False):
    return {"candidate_id": cid, "workload_id": wid, "quality_delta": q, "speed": s,
            "kept": kept, "patch_id": "p-" + cid}


def test_retain_keeps_only_nondominated_misses_tagged_by_workload():
    out = pr.retain([], [_c("a", "w1", -0.01, 1.30), _c("b", "w1", -0.02, 1.10),   # b dominated
                         _c("c", "w1", 0.00, 1.05), _c("k", "w1", 0.0, 2.0, kept=True),
                         _c("d", "w2", -0.5, 1.0), _c("bad", "w2", float("nan"), 1.0)],
                    campaign_id="c1", campaign_seq=1)
    assert sorted((r["workload_id"], r["candidate_id"]) for r in out) == \
        [("w1", "a"), ("w1", "c"), ("w2", "d")]
    assert all(r["campaign_id"] == "c1" for r in out)


def _history():
    recs = pr.retain([], [_c("old", "w1", -0.1, 1.9)], campaign_id="c1", campaign_seq=1)
    recs = pr.retain(recs, [_c("a", "w1", -0.01, 1.3), _c("e", "w2", 0.0, 1.2)],
                     campaign_id="c2", campaign_seq=2)
    recs = pr.retain(recs, [_c("f", "w1", 0.0, 1.1), _c("g", "w3", -0.2, 1.4)],
                     campaign_id="c3", campaign_seq=3)
    return recs


def test_census_query_lists_front_for_the_last_two_campaigns():
    recs = _history()
    got = pr.front(recs)                       # campaigns 2 and 3 only; c1 ('old') excluded
    assert {w: [r["candidate_id"] for r in rows] for w, rows in got.items()} == \
        {"w1": ["a", "f"], "w2": ["e"], "w3": ["g"]}
    assert {r["campaign_id"] for rows in got.values() for r in rows} <= {"c2", "c3"}
    assert list(pr.front(recs, workload_id="w2")) == ["w2"]


def test_screen_consumes_the_front_first_and_store_round_trips(tmp_path):
    recs = _history()
    fresh = [{"candidate_id": "n1"}, {"candidate_id": "g"}, {"candidate_id": "n2"},
             {"candidate_id": "a"}]
    ordered = pr.screen_order(fresh, recs, "w2")
    assert [c["candidate_id"] for c in ordered] == ["g", "a", "n1", "n2"]   # best speed first
    assert [c["pareto_retained"] for c in ordered] == [True, True, False, False]
    # a candidate retained on the SAME workload is not a cross-workload reuse
    assert [c["candidate_id"] for c in pr.screen_order(fresh, recs, "w3")][0] == "a"
    path = pr.save(tmp_path / "x" / pr.STORE_NAME, recs)
    assert pr.load(path) == recs


def test_cross_target_eligible_orders_retained_series_first(world):
    repo, q38 = world["repo"], world["q38"]
    keep, held_row = _hold(world)
    _gate(world, held_row["patch_id"])
    assert "pareto_retained" not in ct.eligible(q38, repo=repo)[0]    # default unchanged
    on = ct.eligible(q38, repo=repo, retained=[held_row["patch_id"]])
    off = ct.eligible(q38, repo=repo, retained=["nope"])
    assert on[0]["pareto_retained"] is True and off[0]["pareto_retained"] is False
