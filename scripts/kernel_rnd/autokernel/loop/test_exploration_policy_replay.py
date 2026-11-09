import sqlite3

import pytest

from . import exploration_policy_replay as ep


def _pool(effs, arms=None):
    return [{"id": str(i), "campaign": "c", "mechanism_id": f"m{i}", "arm": (arms or ["a"] * len(effs))[i],
             "status": "measured_null", "effect": e, "pos": i} for i, e in enumerate(effs)]


def test_auc_matches_process_metrics_definition():
    from . import process_metrics as pm
    effs = [-0.1, 0.02, 0.01, 0.05, 0.0]
    sc = ep.score(_pool(effs), 5)
    rows = [{"status": "measured_null", "effect_fraction": e} for e in effs]
    assert sc["auc"] == pytest.approx(pm.calculate(rows)["auc_best_so_far_effect_over_steps"])


def test_recorded_policy_reproduces_order():
    p = _pool([0.1, 0.2, 0.3])
    r = ep.replay_order(p, ep.p_recorded)
    assert [x["pos"] for x in r["picked"]] == [0, 1, 2]


def test_prefix_only_candidates_hide_outcomes():
    def cheat(prefix, cands):
        return int(cands[0]["effect"])
    with pytest.raises(KeyError):
        ep.replay_order(_pool([0.1, 0.2]), cheat)


def test_batch_sees_one_prefix_and_counts_waste():
    seen = []

    def spy(prefix, cands):
        seen.append(len(prefix))
        return 0
    r = ep.replay_order(_pool([0.1] * 4, ["a", "a", "b", "b"]), spy, width=2)
    assert seen == [0, 0, 2, 2] and r["wasted"] == 2


def test_ucb_beats_recorded_when_good_arm_is_late():
    arms = ["bad"] * 10 + ["good"] * 3 + ["x", "y"]
    effs = [-0.1] * 10 + [0.3, 0.3, 0.3, -0.1, -0.1]
    p = _pool(effs, arms)
    rec = ep.score(ep.replay_order(p, ep.p_recorded)["picked"], len(p))
    ucb = ep.score(ep.replay_order(p, ep.make_ucb())["picked"], len(p))
    assert ucb["auc"] > rec["auc"]


def test_stop_charges_endpoint_loss_and_saves_steps():
    p = _pool([0.01] + [0.0] * 9 + [0.5])
    r = ep.replay_pool(p, n_random=5, stops=(3,))
    s = r["stop_policies"]["plateau_k3"]
    assert s["steps_saved"] > 0 and s["endpoint_loss_vs_full_recorded"] == pytest.approx(0.49)
    assert s["auc_delta_vs_full_recorded"] < 0


def test_deterministic():
    p = _pool([0.1, -0.2, 0.05, 0.3, 0.0, 0.2], list("abcabc"))
    assert ep.replay_pool(p, n_random=20, seed=3) == ep.replay_pool(p, n_random=20, seed=3)


def test_load_pool_is_readonly_and_filters(tmp_path):
    db = tmp_path / "experiments.db"
    c = sqlite3.connect(db)
    c.execute("create table experiments(attempt_id,recorded_at,campaign_id,mechanism_id,target_symbol,"
              "target_surface,status,effect_fraction,payload)")
    c.executemany("insert into experiments values(?,?,?,?,?,?,?,?,?)",
                  [("1", "t1", "c", "m1", "s", "x", "measured_null", 0.1, "{}"),
                   ("2", "t2", "c", "m2", "s", "x", "planner_transient", None, "{}"),
                   ("3", "t3", "c", "m3", None, "x", "kept", 0.2, "{}")])
    c.commit(); c.close()
    before = db.read_bytes()
    pool = ep.load_pool(db)
    assert [r["id"] for r in pool] == ["1", "3"] and pool[1]["arm"] == "?"
    assert db.read_bytes() == before
