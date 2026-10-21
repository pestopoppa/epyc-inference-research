import json
import random
import sqlite3

from . import rex_arm_replay as rx


def test_h_and_rungs_from_gates():
    g = [{"gate": "compile", "passed": True}, {"gate": "correctness", "passed": True, "detail": "8/10 tests passed"},
         {"gate": "reference_comparison", "passed": False}]
    r = rx.rungs_from_gates(g)
    assert [x["name"] for x in r] == ["compile", "op_test", "nmse_reference_comparison"]
    assert abs(rx.ladder_h(r) - (1 + 0.8 + 0) / 3) < 1e-12


def test_deterministic_and_known_ordering():
    a = rx.Archive(rx.synthetic_fixture())
    r1 = rx.replay(a, 30, 40, "t")
    r2 = rx.replay(a, 30, 40, "t")
    assert r1 == r2
    # REx prefers refining high-h nodes (kept 0.9) over low-h (0.05): beats iid on AUC mean
    assert r1["rex_C50"]["auc_frontier"]["mean"] > r1["iid"]["auc_frontier"]["mean"]
    assert r1["rex_C50"]["coverage"] == 1.0 or r1["rex_C50"]["coverage"] > 0.5


def test_fixed_follows_chain_and_unobserved_counted():
    nodes = [{"proposal_id": "a", "parent_id": None, "rungs": [], "repair_attempts": 0, "kept": False},
             {"proposal_id": "a1", "parent_id": "a", "rungs": [], "repair_attempts": 1, "kept": True}]
    a = rx.Archive(nodes)
    tr = rx.run_fixed(a, 5, random.Random(0))
    assert tr.first_kept == 2 and tr.curve() == [0, 1, 1, 1, 1]
    tr = rx.run_split(rx.Archive(nodes[:1]), 5, random.Random(0), 2)
    assert tr.unobserved == 2 and tr.observed == 1
    tr = rx.run_rex(rx.Archive(nodes[:1]), 5, random.Random(0), 5)
    assert tr.unobserved == 1  # refine of childless node: one call, no outcome


def test_budget_never_exceeded_and_bad_parent():
    a = rx.Archive(rx.synthetic_fixture(6))
    for fn in rx.arms().values():
        assert fn(a, 7, random.Random(1)).spent <= 7
    try:
        rx.Archive([{"proposal_id": "x", "parent_id": "nope"}])
        assert False
    except ValueError:
        pass


def test_db_adapter_and_cli(tmp_path):
    db = tmp_path / "c" / "store" / "experiments.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("create table experiments(attempt_id,campaign_id,status,payload)")
    g = [{"gate": "compile", "passed": True}]
    con.execute("insert into experiments values('1','c','kept',?)", (json.dumps({"gates": g, "patch_round": 2}),))
    con.execute("insert into experiments values('2','c','abstained',?)", (json.dumps({}),))
    con.commit(); con.close()
    nodes, skipped = rx.adapt_experiments_db([str(db)])
    assert len(nodes) == 1 and skipped == 1 and nodes[0]["repair_attempts"] == 1 and nodes[0]["kept"]
    out = tmp_path / "r.json"
    assert rx.main(["--fixture", "--reps", "5", "--budget", "10", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert set(rep["arms"]) >= {"fixed", "iid", "rex_C5", "rex_C50", "split_R0", "split_R3"}
