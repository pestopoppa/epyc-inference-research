import sqlite3

from . import lineage_edit_diagnostics as ld


def _r(sym, text, status="measured_null", eff=0.0):
    return {"id": text[:5], "campaign": "c", "symbol": sym, "statement": text, "status": status, "effect": eff}


A = "Replace the lane-divergent scale decode in vec_dot with unconditional indexed loads and straight-line permute"
B = "Fuse the SwiGLU gate epilogue into mul_mat_vec_q single kernel"


def test_labels_cover_known_phrases():
    assert ld.classify("Fix an overflow in the index") == "bug_fix"
    assert ld.classify("Fuse the gate epilogue") == "composition"
    assert ld.classify("Remove the redundant barrier") == "pruning"
    assert ld.classify("Tune the block size") == "hyperparameter_tuning"
    assert ld.classify("zzz qqq") == "unlabeled" and ld.classify(None) == "unlabeled"
    assert set(ld.LABELS) == {"bug_fix", "external_dependency", "architectural_change", "composition",
                              "local_refinement", "pruning", "refactor", "efficiency", "hyperparameter_tuning"}


def test_cycle_requires_intervening_different_key():
    rows = [_r("f", A), _r("f", A)]           # immediate retry: not a cycle
    assert ld.detect_cycling(rows)["recycled"] == 0
    rows = [_r("f", A), _r("g", B), _r("f", A)]  # A..B..A: cycle
    c = ld.detect_cycling(rows)
    assert c["recycled"] == 1 and c["flags"] == [False, False, True] and c["matches"][2] == 0


def test_different_symbol_is_not_a_cycle():
    assert ld.detect_cycling([_r("f", A), _r("g", B), _r("h", A)])["recycled"] == 0


def test_yield_over_frequency_and_deterministic():
    rows = [_r("f", "Fuse x epilogue", "kept", 0.1), _r("g", "Remove dead branch"), _r("h", "Remove redundant load"),
            _r("i", "Remove unnecessary store")]
    t = ld.taxonomy_yield(rows)
    assert t["composition"]["yield_over_frequency"] == 4.0 and t["pruning"]["yield_over_frequency"] == 0.0
    assert ld.taxonomy_yield(rows) == t
    assert abs(sum(v["share"] for v in t.values()) - 1.0) < 1e-12


def test_readonly_analyze(tmp_path):
    db = tmp_path / "experiments.db"
    c = sqlite3.connect(db)
    c.execute("create table experiments(attempt_id,recorded_at,campaign_id,target_symbol,statement,status,effect_fraction)")
    c.executemany("insert into experiments values(?,?,?,?,?,?,?)",
                  [("1", "1", "c", "f", A, "kept", 0.1), ("2", "2", "c", "g", B, "measured_null", 0.0),
                   ("3", "3", "c", "f", A, "measured_null", 0.0)])
    c.commit(); c.close()
    b = db.read_bytes()
    r = ld.analyze(db)
    assert r["cycling"]["recycled"] == 1 and r["cycling"]["cycling_rate"] == 1 / 3
    assert db.read_bytes() == b
