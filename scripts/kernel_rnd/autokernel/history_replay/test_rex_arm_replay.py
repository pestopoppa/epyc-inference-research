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


def _store(tmp_path, name, rows):
    db = tmp_path / "camp" / name / "experiments.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("create table experiments(attempt_id,recorded_at,campaign_id,status,payload)")
    for i, (st, pr, hr, mech, extra) in enumerate(rows):
        payload = {"patch_round": pr, "hypothesis_round": hr, "mechanism_id": mech, "branch_id": "detached:lane0"}
        payload.update(extra)
        con.execute("insert into experiments values(?,?,?,?,?)",
                    (f"a{i}", f"2026-10-01T00:00:{i:02d}Z", "ak-loop", st, json.dumps(payload)))
    con.commit(); con.close()
    return str(db)


def test_chain_adapter_links_patch_rounds_and_excludes_summaries(tmp_path):
    g_ok = [{"gate": "compile", "passed": True}, {"gate": "correctness", "passed": True, "detail": "9/10 tests passed"},
            {"gate": "reference_comparison", "passed": True}]
    panel = [{"members": [{"calls": 2, "decoded_tokens": 100}, {"calls": 1, "decoded_tokens": 50}]}]
    db = _store(tmp_path, "store", [
        ("abstained", 0, 0, "m1", {}),                                   # not an attempt
        ("patch_rejected", 1, 1, "m1", {"author_panels": panel}),        # chain 1 root
        ("gate_refused", 2, 1, "m1", {"gates": [{"gate": "op_scope", "passed": False}]}),
        ("patch_rounds_exhausted", 2, 1, "m1", {}),                      # summary: excluded
        ("measured_null", 1, 1, "m2", {"gates": g_ok}),                  # chain 2, lone root
        ("patch_rejected", 1, 1, "m3", {}),                              # chain 3 root
        ("kept", 2, 1, "m3", {"gates": g_ok}),
        ("kept", 2, 2, "m3", {"gates": g_ok}),                           # different hypothesis_round: no parent
        ("measurement_invalid", 1, 1, "m4", {}),                         # instrument outcome: excluded
    ])
    nodes, st = rx.adapt_experiments_db_chains([db])
    by = {n["proposal_id"].split(":")[-1]: n for n in nodes}
    assert set(by) == {"a1", "a2", "a4", "a5", "a6", "a7"}
    pre = "camp/store:"
    assert by["a1"]["parent_id"] is None and by["a2"]["parent_id"] == pre + "a1"
    assert by["a4"]["parent_id"] is None and by["a6"]["parent_id"] == pre + "a5"
    assert by["a7"]["parent_id"] is None  # round 2 with no round 1 of its own hypothesis_round
    assert by["a1"]["call_cost"] == 3 and by["a1"]["author_decoded_tokens"] == 150 and by["a2"]["call_cost"] == 1
    assert by["a2"]["repair_attempts"] == 1 and by["a6"]["kept"] and not by["a5"]["kept"]
    assert by["a4"]["op_test_passed"] == 9 and by["a4"]["op_test_total"] == 10
    assert [r["name"] for r in by["a4"]["rungs"]] == ["compile", "op_test", "nmse_reference_comparison"]
    assert st["rows_total"] == 9 and st["rows_included"] == 6
    assert st["excluded_status_counts"] == {"abstained": 1, "patch_rounds_exhausted": 1, "measurement_invalid": 1}
    assert st["chains"] == 4 and st["chain_length_counts"] == {"2": 2, "1": 2}
    # unit cost basis ignores panels
    assert {n["call_cost"] for n in rx.adapt_experiments_db_chains([db], "unit")[0]} == {1}


def test_chain_archive_has_repair_tree_and_replays_end_to_end(tmp_path):
    rows = []
    for k in range(6):
        rows.append(("patch_rejected", 1, k + 1, f"m{k}", {}))
        rows.append(("kept" if k % 2 == 0 else "measured_null", 2, k + 1, f"m{k}",
                     {"gates": [{"gate": "compile", "passed": True}]}))
    db = _store(tmp_path, "s", rows)
    nodes, _ = rx.adapt_experiments_db_chains([db])
    a = rx.Archive(nodes)  # no ValueError: every parent_id resolves inside the archive
    assert a.has_tree() and len(a.roots) == 6
    out, dump = tmp_path / "r.json", tmp_path / "arch.jsonl"
    assert rx.main(["--experiments-db", db, "--repair-chains", "--reps", "5", "--budget", "8",
                    "--dump-archive", str(dump), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["archive"]["has_repair_tree"] and rep["repair_chain_stats"]["chains"] == 6
    assert rep["arms"]["rex_C10"]["coverage"] is not None
    assert len(dump.read_text().splitlines()) == 12
    assert rx.load_jsonl(dump).has_tree()


def test_chain_adapter_opens_store_read_only(tmp_path):
    db = _store(tmp_path, "ro", [("patch_rejected", 1, 1, "m", {})])
    before = open(db, "rb").read()
    rx.adapt_experiments_db_chains([db])
    assert open(db, "rb").read() == before
