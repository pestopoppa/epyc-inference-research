import json

from . import campaign_status as rr

SNAP = {
    "now": 10000,
    "rows": [{"id": "r1", "state": "RUNNING", "started_at": 4000, "progress": 0.5, "elapsed_s": 6000,
              "heartbeat_at": 9900},
             {"id": "r2", "state": "RUNNING", "started_at": 9000, "eta_s": 60, "heartbeat_at": 1000},
             {"id": "r3", "state": "BACKLOG"}],
    "bins": [{"name": "gpu", "idle_since": 1000, "reason": "no ready row"},
             {"name": "q1", "idle_since": 9800}],
    "escalations": [{"id": "e1", "text": "should I proceed?"},
                    {"id": "e2", "text": "PPL tol", "decision": "Accept +0.17% PPL on 27B?"}],
    "judgment_calls": [{"topic": "PPL tolerance", "recommendation": "accept 0.2%",
                        "options": [{"name": "0.2%", "tradeoff": "keeps H15", "criteria": {"ppl": "PASS"}},
                                    {"name": "0.1%", "tradeoff": "drops H15", "criteria": {"ppl": "FAIL", "e2e": "PASS"}}]},
                       {"topic": "park track", "options": []}],
}


def test_row_eta_and_stale():
    l = rr.row_lines(SNAP)
    assert "ETA 1h40m" in l[0] and "STALE" not in l[0]
    assert "STALE-HEARTBEAT" in l[1] and len(l) == 2  # BACKLOG row has no line


def test_idle_bins_only_when_backlog_and_over_10min():
    b = rr.idle_bins(SNAP)
    assert [x["bin"] for x in b] == ["gpu"] and b[0]["reason"] == "no ready row"  # q1 only 200 s
    # negative control: nothing in BACKLOG -> no idle report
    s = dict(SNAP, rows=[r for r in SNAP["rows"] if r["state"] != "BACKLOG"])
    assert rr.idle_bins(s) == []


def test_escalation_without_named_decision_dropped():
    rep = rr.build_report(SNAP)
    assert [e["id"] for e in rep["escalations"]] == ["e2"]
    assert rep["escalations_dropped_no_decision"] == ["e1"]
    assert "should I proceed" not in rr.render(rep)
    assert "Accept +0.17% PPL on 27B?" in rr.render(rep)


def test_blank_decision_is_dropped():
    s = dict(SNAP, escalations=[{"id": "x", "decision": "   "}])
    assert rr.build_report(s)["escalations"] == []


def test_single_decision_package_recommendation_first_and_table():
    rep = rr.build_report(SNAP)
    pkg = rep["decision_package"]
    assert len(pkg["items"]) == 2 and pkg["decided_by"] == "operator"
    assert pkg["items"][0]["options"][0]["criteria"] == {"e2e": "n/a", "ppl": "PASS"}
    out = rr.render(rep)
    assert out.count("DECISION PACKAGE") == 1 and "RECOMMEND: accept 0.2%" in out
    assert out.index("DECISION PACKAGE") < out.index("ROWS")
    assert rr.build_report(dict(SNAP, judgment_calls=[]))["decision_package"] is None


def test_cli(tmp_path, capsys):
    f = tmp_path / "s.json"; f.write_text(json.dumps(SNAP))
    assert rr.main([str(f), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["idle_bins"][0]["bin"] == "gpu"


def test_publish_atomic_roundtrip_and_negative_row_id(tmp_path):
    import pytest
    p = rr.publish(tmp_path, "rowA", SNAP)
    d = json.loads(p.read_text())
    assert d["schema"] == rr.SCHEMA and d["row_id"] == "rowA"
    assert [e["id"] for e in d["escalations"]] == ["e2"]       # undecided escalation never published
    assert not list((tmp_path / "campaign-status").glob(".*tmp"))
    for bad in ("", "../x", ".h", "a/b"):
        with pytest.raises(ValueError):
            rr.publish(tmp_path, bad, SNAP)
