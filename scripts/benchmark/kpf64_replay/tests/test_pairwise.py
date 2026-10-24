import json

import analysis
import pairwise


def req(s, k, total, cache, ideal, pred=1, inferred=None, chars=None):
    return {"session": s, "k": k, "total_tok": total, "cache_n": cache, "prompt_n": total - cache, "ideal_tok": ideal,
            "predicted_n": pred, "prompt_chars": chars or total * 4, "reset": False,
            "inferred_evict": (k > 0 and ideal > 0 and cache < ideal - 32) if inferred is None else inferred}


def mk(tmp, arm, rows):
    d = tmp / arm / "replay"
    d.mkdir(parents=True)
    (d / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def sess_dir(tmp, ids, n=3):
    d = tmp / "sessions"
    d.mkdir()
    for i in ids:
        (d / f"{i}.json").write_text(json.dumps({"id": i, "store": "claude", "turns": [
            {"pre": f"<tool:Bash>\nx{k}", "ref": "r", "reset": False} for k in range(n)]}))
    return d


def test_pairwise_same_draws_and_pooled(tmp_path):
    ids = [f"s{i}" for i in range(6)]
    for p in ("pass1", "pass2"):
        for arm, c in (("A", 0), ("B", 90), ("C", 90), ("D", 90)):
            rows = []
            for s in ids:
                rows += [req(s, 0, 100, 0, 0), req(s, 1, 200, c, 100)]
            mk(tmp_path, f"{p}-{arm}", rows)
    sd = sess_dir(tmp_path, ids, 2)
    res = pairwise.run(tmp_path, sd, B=50, passes=("pass1", "pass2"))
    assert res["paired_sessions"] == 6
    assert res["contrasts"]["p1: C-B"]["gap_pp"]["mean"] == 0
    assert res["contrasts"]["p1: B-A"]["gap_pp"]["real_effect"]
    assert "pooled: D-A" in res["pooled_contrasts"] and "pooled: D-C" in res["pooled_contrasts"]
    assert res["excluded_sessions"] == []
    md = pairwise.to_markdown(res)
    assert "pooled: B-A" in md


def test_excluded_session_and_reason(tmp_path):
    rows = [req("s1", 0, 100, 0, 0), req("s1", 1, 200, 100, 100), req("s2", 0, 100, 0, 0), {"session": "s2", "k": 1, "error": "HTTPError: 400"}]
    mk(tmp_path, "pass1-A", rows)
    (tmp_path / "pass1-A" / "server.log").write_text(
        "94.1 E srv    send_error: task id = 5, error: request (66982 tokens) exceeds the available context size (65536 tokens), try\n")
    e = pairwise.excluded_record(tmp_path / "pass1-A")
    assert e[0]["session"] == "s2" and e[0]["server_reasons"][0]["tokens"] == 66982 and e[0]["last_ok_k"] == 0


def test_classify_classes(tmp_path):
    sd = sess_dir(tmp_path, ["s1"], 5)
    rows = [req("s1", 0, 1000, 0, 0, pred=10),
            req("s1", 1, 2000, 1005, 1500, pred=10),    # slot cap 1010 -> estimator (cache>=cap-32)
            req("s1", 2, 3000, 1900, 2500, pred=10),    # cap 2010, cache 1900 -> join (110 short)
            req("s1", 3, 4000, 1000, 3500, pred=10),    # cap 3010, cache 1000 -> partial
            req("s1", 4, 5000, 0, 4500, pred=10)]       # full
    r = pairwise.classify(rows, sd)
    assert [x["cls"] for x in r] == ["estimator", "join", "partial", "full"]
    assert r[0]["tag"] == "tool:Bash" and r[1]["deficit_vs_slot"] == 110
    s = pairwise.summarise(r, 5)
    assert s["real_misses"] == 3 and s["by_class"]["estimator"] == 1


def test_belief_emit_per_arm_windows_and_cli(tmp_path):
    import sys
    sys.path.insert(0, "/workspace/scripts/vidya")
    for arm, c in (("A", 0), ("B", 90)):
        mk(tmp_path, arm, [req("s1", 0, 100, 0, 0), req("s1", 1, 200, c, 100)])
        (tmp_path / arm / "replay" / "run_meta.json").write_text("{}")
    pin = {"role": "frontdoor", "model": "m", "kernel_ref": "k"}
    def w(a):
        return {"window_id": f"g-{a}", "region_lock_claim": "q0", "start_utc": "2026-10-22T00:00:00Z", "end_utc": "2026-10-22T01:00:00Z"}
    out = tmp_path / "res.json"
    argv = ["--arm", f"A={tmp_path / 'A' / 'replay'}", "--arm", f"B={tmp_path / 'B' / 'replay'}", "--baseline", "A",
            "--out-json", str(out), "--out-md", str(tmp_path / "res.md"), "--boot", "20"]
    import analysis as an
    assert an.main(argv) == 0
    bad = argv + ["--belief-out", str(tmp_path / "bel")]
    try:
        an.main(bad)
        raise AssertionError("needs pin/window")
    except SystemExit as e:
        assert e.code == 2
    (tmp_path / "win.json").write_text(json.dumps({"A": w("A"), "B": w("B")}))
    assert an.main(argv + ["--belief-out", str(tmp_path / "bel"), "--belief-pin", json.dumps(pin),
                           "--belief-window", "@" + str(tmp_path / "win.json"), "--belief-run-id", "t1"]) == 0
    from adapters import chunk_kv
    rows = chunk_kv.rows_for_run(tmp_path / "bel")
    assert len(rows) == 2 and {r["row"]["extra"]["window"]["window_id"] for r in rows} == {"g-A", "g-B"}
    res = json.loads(out.read_text())
    try:
        an.belief_measurements(res, pin, {"A": w("A")})
        raise AssertionError("missing arm window must raise")
    except ValueError:
        pass
