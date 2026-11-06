import json

from . import ak_tier1_rank_replay as t


def _row(diff, eff, ts, key):
    return {"diff": diff, "effect": eff, "ts": ts, "key": key, "window": None, "status": "x"}


def test_strict_pairing_needs_same_diff_and_two_attempts():
    rows = [_row("a", 0.1, "2026-10-01T00:00:00Z", "1"), _row("a", 0.2, "2026-10-01T01:00:00Z", "2"),
            _row("b", 0.1, "2026-10-01T00:00:00Z", "3"), _row(None, 0.1, "t", "4"),
            _row("c", None, "t", "5"), _row("c", 0.3, "t", "6"),
            _row("d", 0.1, "2026-10-01T00:00:00Z", "7"), _row("d", 0.1, "2026-10-01T00:00:00Z", "7")]
    p = t.build_pairs(rows)
    assert [x["diff"] for x in p] == ["a"]
    assert p[0]["screen"] == 0.1 and p[0]["confirmed"] == 0.2 and p[0]["gap_s"] == 3600.0


def test_spearman_ties_and_degenerate():
    assert abs(t.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(t.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert t.spearman([1, 1, 1], [1, 2, 3]) is None
    assert t.spearman([1, 2], [1, 2]) is None


def test_verdict_underdetermined_on_small_n_and_deterministic():
    rows = []
    for i, (s, c) in enumerate([(0.1, -0.2), (0.2, 0.1), (-0.1, 0.3), (0.05, -0.01)]):
        rows += [_row(f"d{i}", s, "2026-10-01T00:00:00Z", f"s{i}"), _row(f"d{i}", c, "2026-10-01T00:20:00Z", f"c{i}")]
    r1, r2 = t.replay(rows, 300), t.replay(rows, 300)
    assert r1 == r2 and r1["n_pairs"] == 4
    assert r1["verdict"].startswith("UNDERDETERMINED") or "DOES NOT" in r1["verdict"]
    assert not json.dumps(r1).count("NaN")


def test_perfect_rank_agreement_excludes_nothing_below():
    rows = []
    for i in range(12):
        rows += [_row(f"d{i}", i / 100, "2026-10-01T00:00:00Z", f"s{i}"), _row(f"d{i}", i / 50 + 0.001, "2026-10-01T02:00:00Z", f"c{i}")]
    r = t.replay(rows, 300)
    assert r["spearman_rho"] > 0.99 and r["sign_agreement"]["rate"] >= 0.9
    assert r["flip_rate_by_duration_gap"]["3600-infs"]["rate"] == 0.0
