"""KPF-LB2 amendments: sub-agent linking, bursts, think-time gaps, opencode V1/V2, zero-session
failure, the two ideal bounds, provider-realised share, fan-out stats."""
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

import adapter
import analysis

HERE = Path(__file__).resolve().parents[1]
T0 = 1_790_000_000.0


def iso(sec):
    import datetime as dt
    return dt.datetime.fromtimestamp(T0 + sec, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def w(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def a_line(sec, text, mid, usage=None, tool=None):
    blocks = [{"type": "text", "text": text}]
    if tool:
        blocks.append({"type": "tool_use", "id": tool, "name": "Agent", "input": {"prompt": "go"}})
    return {"type": "assistant", "timestamp": iso(sec),
            "message": {"id": mid, "content": blocks, **({"usage": usage} if usage else {})}}


def u_line(sec, text="x" * 40, result_for=None, agent_id=None):
    if result_for:
        d = {"type": "user", "timestamp": iso(sec), "message": {"content": [
            {"type": "tool_result", "tool_use_id": result_for, "content": "done " * 20}]}}
        if agent_id:
            d["toolUseResult"] = {"agentId": agent_id, "isAsync": True, "status": "async_launched"}
        return d
    return {"type": "user", "timestamp": iso(sec), "message": {"content": text}}


def child_lines(base, n=4, tag="c"):
    out = []
    for i in range(n):
        out.append(u_line(base + 10 * i, f"{tag} task {i} " * 30))
        out.append(a_line(base + 10 * i + 1, f"{tag} answer {i}", f"m{tag}{i}"))
    return out


def mk_family_home(tmp, big_parent=False):
    """Claude parent: request 0 (no spawn), request 1 spawns TWO sub-agents (explicit links), request 2
    spawns a third that has no agentId link (time link)."""
    h = tmp / "home"
    proj = h / ".claude/projects/p"
    pad = "P" * (300_000 if big_parent else 60)
    parent = [u_line(0, "start " + pad),
              a_line(1, "A0", "pm0", {"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 0}),
              u_line(5, "next " * 30),
              a_line(6, "A1", "pm1", {"input_tokens": 5, "cache_read_input_tokens": 95}, tool="tu1"),
              a_line(6.1, "A1b", "pm1", {"input_tokens": 5, "cache_read_input_tokens": 95}, tool="tu2"),  # same message id: usage once
              u_line(6.5, result_for="tu1", agent_id="aaa"), u_line(6.6, result_for="tu2", agent_id="bbb"),
              a_line(20, "A2", "pm2", {"input_tokens": 20, "cache_read_input_tokens": 0}, tool="tu3"),
              u_line(21, result_for="tu3"),                       # no agentId -> time link
              a_line(40, "A3", "pm3")]
    w(proj / "sess1.jsonl", parent)
    w(proj / "sess1/subagents/agent-aaa.jsonl", child_lines(T0 * 0 + 7, tag="a"))
    w(proj / "sess1/subagents/agent-bbb.jsonl", child_lines(8, tag="b"))
    w(proj / "sess1/subagents/agent-ccc.jsonl", child_lines(22, tag="c"))
    return h


def test_turn_gaps_spawns_and_usage_dedupe(tmp_path):
    h = mk_family_home(tmp_path)
    t = adapter.load_turns(("claude", str(h / ".claude/projects/p/sess1.jsonl"), None))
    assert len(t) == 4
    assert t[0]["gap_s"] is None
    assert t[1]["gap_s"] == pytest.approx(5 - 1, abs=1e-3)        # A0 end (1s) -> next user (5s)
    assert t[1]["spawns"] == ["tu1", "tu2"]
    assert t[1]["usage"] == {"prompt": 100, "cache_read": 95, "cache_write": 0}   # pm1 counted once, not twice
    sh = adapter.provider_share(t)
    assert sh["share"] == pytest.approx((90 + 95 + 0) / (100 + 100 + 20))


def test_children_link_explicit_then_time_then_fanout(tmp_path):
    h = mk_family_home(tmp_path)
    cand = ("claude", str(h / ".claude/projects/p/sess1.jsonl"), None)
    pt = adapter.load_turns(cand)
    kids = {Path(k["cand"][1]).stem: k for k in adapter.children_of(cand, pt)}
    assert kids["agent-aaa"]["parent_turn"] == 1 and kids["agent-aaa"]["link"] == "explicit" and kids["agent-aaa"]["async"]
    assert kids["agent-bbb"]["parent_turn"] == 1
    assert kids["agent-ccc"]["link"] == "time" and kids["agent-ccc"]["parent_turn"] == 2   # first event 22s -> after request 2
    recs = [{"id": "p", "parent": None, "start": 0, "end": 50, "requests": 4}] + [
        {"id": f"k{i}", "parent": "p", "start": 7 + i, "end": 40, "requests": 4} for i in range(3)]
    f = adapter.fanout_stats(recs)
    assert f["child_request_share"] == pytest.approx(12 / 16) and f["peak_children_per_parent"] == 3
    assert f["peak_active_lineages"] == 4 and f["children_per_parent_max"] == 3


def test_flat_mode_keeps_subagents_as_candidates_but_families_does_not(tmp_path):
    h = mk_family_home(tmp_path)
    flat = adapter.candidates("claude", str(h), include_subagents=True)
    fam = adapter.candidates("claude", str(h), include_subagents=False)
    assert len(flat) == 4 and len(fam) == 1
    picked = adapter.sample_sessions(str(h), n=2, stores=("claude",), min_requests=3, mode="flat")
    assert all(p[4]["link"] is None or p[4]["link"]["parent_turn"] is not None for p in picked)
    assert any(p[4]["link"] for p in picked)                       # a sampled sub-agent knows its parent


def test_families_mode_non_replayable_parent_is_an_anchor(tmp_path):
    h = mk_family_home(tmp_path, big_parent=True)                  # parent outgrows the 200k-char cap
    picked = adapter.sample_sessions(str(h), n=1, stores=("claude",), min_requests=3, mode="families",
                                     min_children=2, min_child_requests=3)
    s, cand, turns, st, extra = picked[0]
    assert extra["replayable"] is False and turns == [] and len(extra["children"]) == 3
    assert sorted(k["parent_turn"] for k in extra["children"]) == [1, 1, 2]


def test_zero_sessions_fail_loudly(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(adapter.ZeroSessionsError):
        adapter.sample_sessions(str(empty), n=3, stores=("claude",))
    with pytest.raises(adapter.ZeroSessionsError):                 # candidates exist, none pass the filter
        h = mk_family_home(tmp_path)
        adapter.sample_sessions(str(h), n=1, stores=("claude",), min_requests=999)
    # a store explicitly allowed empty is redistributed, but an all-empty sample still raises
    with pytest.raises(adapter.ZeroSessionsError):
        adapter.sample_sessions(str(empty), n=3, stores=("claude", "codex"), allow_empty_store=True)
    r = subprocess.run([sys.executable, str(HERE / "adapter.py"), "--home", str(empty), "--out", str(tmp_path / "o"),
                        "--stores", "claude"], capture_output=True, text=True)
    assert r.returncode == 4 and "zero sessions" in r.stderr


MS = 1_790_000_000_000      # opencode stores epoch milliseconds


def mk_opencode(tmp, v1_sessions=(), v2_sessions=(), both=(), with_v2_table=True):
    db = tmp / "oc.db"
    con = sqlite3.connect(db)
    con.execute("create table session(id text, parent_id text, time_created int, time_updated int)")
    con.execute("create table message(id text, session_id text, time_created int, data text)")
    con.execute("create table part(id text, message_id text, session_id text, time_created int, data text)")
    if with_v2_table:
        con.execute("create table session_message(id text, session_id text, type text, seq int, time_created int, data text)")
    for sid in set(v1_sessions) | set(v2_sessions) | set(both):
        con.execute("insert into session values(?,null,1,2)", (sid,))
    for sid, tag in [(s, "V1") for s in set(v1_sessions) | set(both)]:
        con.execute("insert into message values(?,?,1,?)", (f"m-{sid}", sid, json.dumps({"role": "user"})))
        con.execute("insert into part values(?,?,?,1,?)", (f"p-{sid}", f"m-{sid}", sid, json.dumps({"type": "text", "text": f"{tag} text {sid}"})))
    for sid in set(v2_sessions) | set(both):
        con.execute("insert into session_message values(?,?,?,?,?,?)", (f"u-{sid}", sid, "user", 1, MS + 1000, json.dumps({"text": f"V2 user {sid}", "time": {"created": MS + 1000}})))
        con.execute("insert into session_message values(?,?,?,?,?,?)", (f"a-{sid}", sid, "assistant", 2, MS + 2000, json.dumps({
            "time": {"created": MS + 2000}, "tokens": {"input": 5, "cache": {"read": 15, "write": 0}},
            "content": [{"type": "text", "text": f"V2 answer {sid}"},
                        {"type": "tool", "id": "t", "name": "read", "time": {"ran": MS + 2100, "completed": MS + 2200},
                         "state": {"status": "completed", "input": {"p": 1}, "content": [{"type": "text", "text": "file body"}]}}]})))
    con.commit()
    return con


def test_opencode_layout_detection_and_never_both(tmp_path):
    con = mk_opencode(tmp_path, v1_sessions=("s1",), v2_sessions=("s2",), both=("s3",))
    assert adapter.opencode_layouts(con) == {"v1", "v2"}
    assert adapter.opencode_session_layout(con, "s1") == "v1"
    assert adapter.opencode_session_layout(con, "s2") == "v2"
    assert adapter.opencode_session_layout(con, "s3") == "v2"      # migrated session: V2 wins
    assert adapter.opencode_session_layout(con, "missing") is None
    s3 = "".join(i[1] for i in adapter.opencode_items("s3", con) if i[0] != "USAGE")
    assert "V2 answer" in s3 and "V1 text" not in s3               # V1 rows of s3 never read
    turns = adapter.items_to_turns(adapter.opencode_items("s2", con))
    assert len(turns) == 2 and "file body" in turns[1]["pre"] and "read" in turns[0]["ref"]
    assert turns[0]["usage"] == {"prompt": 20, "cache_read": 15, "cache_write": 0}
    assert turns[1]["gap_s"] == pytest.approx(0.1, abs=1e-3)             # tool ran at +100ms ... completed: emitted gap
    v1 = adapter.items_to_turns(adapter.opencode_items("s1", con))
    assert v1 == [] or "V1 text" in v1[0]["pre"]                   # user-only V1 session


def test_opencode_without_v2_table_and_unknown_schema(tmp_path):
    con = mk_opencode(tmp_path, v1_sessions=("s1",), with_v2_table=False)
    assert adapter.opencode_layouts(con) == {"v1"} and adapter.opencode_session_layout(con, "s1") == "v1"
    h = tmp_path / "home"
    (h / ".local/share/opencode").mkdir(parents=True)
    db = sqlite3.connect(h / ".local/share/opencode/opencode.db")
    db.execute("create table something_else(x int)")
    db.commit()
    db.close()
    with pytest.raises(adapter.ZeroSessionsError, match="neither opencode layout"):
        adapter.candidates("opencode", str(h))


def test_head_index_cross_session_bound():
    b = adapter.HEAD_BLOCK
    head = "H" * (3 * b)
    idx = adapter.HeadIndex()
    assert idx.match_and_add(head + "a" * 2000, "s1") == 0
    assert idx.cross_chars(head + "b" * 2000, "s1") == 0                # own session never counts
    assert idx.cross_chars(head + "b" * 2000, "s2") == 3 * b            # shared head, block lower bound
    assert idx.cross_chars("Z" * 5000, "s2") == 0
    idx.add(head + "b" * 2000, "s2")
    assert idx.cross_chars(head + "a" * 2000, "s1") == 3 * b            # now s2 also holds it


def test_manifest_reports_both_bounds_provider_share_and_links(tmp_path):
    h = mk_family_home(tmp_path)
    picked = adapter.sample_sessions(str(h), n=1, stores=("claude",), min_requests=3, mode="families",
                                     min_children=2, min_child_requests=3)
    m = adapter.build_manifest(picked, {})
    assert m["schema"] == "kpf64.sessions.v2"
    assert m["child_links"] == {"explicit": 2, "time": 1}
    assert m["fanout"]["child_sessions"] == 3 and m["fanout"]["peak_children_per_parent"] >= 2
    top = m["totals"]
    assert top["ideal_share_cross_head"] >= top["ideal_share"]
    assert m["totals"]["provider_cache_share"] == pytest.approx(185 / 220)
    out = tmp_path / "s"
    adapter.write_sessions(picked, out, m)
    par = json.loads((out / "claude-00.json").read_text())
    assert [c["parent_turn"] for c in par["children"]] == [1, 1, 2]
    assert json.loads((out / "claude-00.c00.json").read_text())["parent"] == "claude-00"


# ---- driver ---------------------------------------------------------------------------------------
def run_driver(args, window="1"):
    import os
    env = {k: v for k, v in os.environ.items() if k != "WINDOW_CONFIRMED"}
    env["WINDOW_CONFIRMED"] = window
    return subprocess.run([sys.executable, str(HERE / "replay_driver.py"), *args], capture_output=True, text=True, env=env)


def families_dir(tmp, big_parent):
    h = mk_family_home(tmp, big_parent)
    s = tmp / "s"
    r = subprocess.run([sys.executable, str(HERE / "adapter.py"), "--home", str(h), "--out", str(s), "--n", "1",
                        "--stores", "claude", "--mode", "families", "--min-requests", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return s


@pytest.mark.parametrize("big_parent", [False, True])
def test_burst_replay_pauses_parent_and_runs_children_concurrently(tmp_path, big_parent):
    s = families_dir(tmp_path, big_parent)
    o = tmp_path / "o"
    r = run_driver(["--sessions", str(s), "--out", str(o), "--dry-run", "--history", "live", "--children", "bursts",
                    "--snap-interval", "0.2"])
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in open(o / "requests.jsonl")]
    kids = [x for x in rows if x["is_child"]]
    assert {x["session"] for x in kids} == {"claude-00.c00", "claude-00.c01", "claude-00.c02"}
    assert len(kids) == 12
    meta = json.loads((o / "run_meta.json").read_text())
    assert meta["bursts"] == 2 and sorted(meta["burst_sizes"]) == [1, 2] and meta["children"] == "bursts"
    assert meta["peak_active_lineages"] >= 2 and meta["peak_inflight_requests"] >= 1
    if not big_parent:
        # parent paused: its request after the burst starts only once every child of the burst has ended
        p = {x["k"]: x for x in rows if x["session"] == "claude-00"}
        burst1 = [x for x in kids if x["session"] in ("claude-00.c00", "claude-00.c01")]
        assert p[2]["t_start"] >= max(x["t_end"] for x in burst1) - 1e-6
    else:
        assert not any(x["session"] == "claude-00" for x in rows)       # anchor only: no parent requests
    res = analysis.analyse({"F": o}, "F", B=20)
    f = res["arms"]["F"]["fanout"]
    assert f["child_request_share"] > 0.5 and f["peak_inflight_requests"] == meta["peak_inflight_requests"]


def test_children_off_skips_anchor_only_parent_and_zero_sessions_refuse(tmp_path):
    s = families_dir(tmp_path, big_parent=True)
    r = run_driver(["--sessions", str(s), "--out", str(tmp_path / "o"), "--dry-run"])
    assert r.returncode != 0 and "zero replayable sessions" in r.stderr
    (tmp_path / "e").mkdir()
    (tmp_path / "e/manifest.json").write_text(json.dumps({"sessions": []}))
    r = run_driver(["--sessions", str(tmp_path / "e"), "--out", str(tmp_path / "o2"), "--dry-run"])
    assert r.returncode != 0 and "zero sessions" in r.stderr


def test_think_gaps_are_emitted_with_scale_and_cap(tmp_path):
    h = tmp_path / "home"
    lines = []
    for i in range(3):
        lines += [u_line(i * 1.0, f"q{i} " * 40), a_line(i * 1.0 + 0.1, f"a{i}", f"m{i}")]
    w(h / ".claude/projects/p/s.jsonl", lines)
    s = tmp_path / "s"
    r = subprocess.run([sys.executable, str(HERE / "adapter.py"), "--home", str(h), "--out", str(s), "--n", "1",
                        "--stores", "claude", "--min-requests", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    gaps = [t["gap_s"] for t in json.loads((s / "claude-00.json").read_text())["turns"]]
    assert gaps[0] is None and gaps[1] == pytest.approx(0.9, abs=1e-3)
    o0 = tmp_path / "o0"
    t0 = time.time()
    run_driver(["--sessions", str(s), "--out", str(o0), "--dry-run"])
    quick = time.time() - t0
    o1 = tmp_path / "o1"
    run_driver(["--sessions", str(s), "--out", str(o1), "--dry-run", "--think-scale", "1.0", "--think-cap-s", "0.4"])
    w1 = [json.loads(x)["think_wait_s"] for x in open(o1 / "requests.jsonl")]
    assert w1 == [0.0, 0.4, 0.4]                                           # capped, k=0 never waits
    assert all(json.loads(x)["think_wait_s"] == 0.0 for x in open(o0 / "requests.jsonl")) and quick < 30


def test_oversubscribed_flag_and_cross_head_in_analysis(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    head = "SHARED-HEAD " * 400
    for i in range(3):
        (s / f"claude-0{i}.json").write_text(json.dumps({"id": f"claude-0{i}", "store": "claude", "replayable": True, "turns": [
            {"pre": head + f"u{i}" * 100, "ref": "r", "reset": False}, {"pre": "more " * 50, "ref": "", "reset": False}]}))
    (s / "manifest.json").write_text(json.dumps({"sessions": [{"id": f"claude-0{i}", "store": "claude", "parent": None}
                                                              for i in range(3)], "totals": {"sessions": 3, "ideal_share": 0.5,
                                                              "ideal_share_cross_head": 0.6, "provider_cache_share": 0.9, "by_store": {}}}))
    o = tmp_path / "o"
    r = run_driver(["--sessions", str(s), "--out", str(o), "--dry-run", "--active", "6", "--slots", "2"])
    assert r.returncode == 0, r.stderr
    meta = json.loads((o / "run_meta.json").read_text())
    assert meta["oversubscribed"] is True and meta["active"] == 6 and meta["slots"] == 2
    rows = [json.loads(x) for x in open(o / "requests.jsonl")]
    assert max(x["ideal_union_tok"] for x in rows) > 0 and all(x["ideal_union_tok"] >= x["ideal_tok"] for x in rows)
    res = analysis.analyse({"E": o}, "E", B=20, manifest=s / "manifest.json")
    a = res["arms"]["E"]
    assert a["ideal_cross_head_share"] >= a["ideal_share"] and res["reference"]["provider_cache_share"] == 0.9
    assert "cross-head" in analysis.to_markdown(res)


# ---- DS-LB1: per-session accumulated service time next to arrival order ----------------------------
def test_driver_logs_arrival_order_and_session_service_time(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    for i in range(3):
        (s / f"claude-0{i}.json").write_text(json.dumps({"id": f"claude-0{i}", "store": "claude", "replayable": True, "turns": [
            {"pre": f"q{i} " * 60, "ref": "r", "reset": False}, {"pre": "more " * 40, "ref": "", "reset": False},
            {"pre": "again " * 40, "ref": "", "reset": False}]}))
    (s / "manifest.json").write_text(json.dumps({"sessions": [{"id": f"claude-0{i}", "store": "claude", "parent": None}
                                                              for i in range(3)], "totals": {}}))
    o = tmp_path / "o"
    r = run_driver(["--sessions", str(s), "--out", str(o), "--dry-run", "--active", "3", "--history", "live"])
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in open(o / "requests.jsonl")]
    assert sorted(x["arrival_seq"] for x in rows) == list(range(9))            # one global arrival order
    by = {}
    for x in sorted(rows, key=lambda x: x["arrival_seq"]):
        by.setdefault(x["session"], []).append(x)
    for sid, rs in by.items():
        assert rs[0]["svc_before_ms"] == 0.0
        for a, b in zip(rs, rs[1:]):                                            # accumulation is exact and monotone
            assert b["svc_before_ms"] == pytest.approx(a["svc_after_ms"])
            assert a["svc_after_ms"] == pytest.approx(a["svc_before_ms"] + a["svc_ms"])
    res = analysis.analyse({"A": o}, "A", B=20)
    od = res["arms"]["A"]["ordering"]
    assert od["requests"] == 9 and od["service_ms_total"] > 0


def test_ordering_report_known_answer():
    mk = lambda sess, seq, ts, te, before, svc: {"session": sess, "arrival_seq": seq, "t_start": ts, "t_end": te,
                                                  "svc_before_ms": before, "svc_ms": svc, "svc_after_ms": before + svc}
    reqs = [mk("a", 0, 0, 10, 500, 5),       # arrived first but has consumed more service
            mk("b", 1, 1, 9, 100, 5),        # overlaps a, little service: LAS would admit b first
            mk("c", 2, 20, 30, 0, 5),        # starts after both finished: never queued with them
            mk("a", 3, 2, 8, 505, 5)]        # same session as a: not a cross-session pair
    od = analysis.ordering_report(reqs)
    assert od["overlapping_cross_session_pairs"] == 2          # a0-b1 and b1-a3 (a0-a3 same session)
    assert od["las_vs_fcfs_disagreements"] == 1 and od["las_disagreement_share"] == 0.5
    assert analysis.ordering_report([])["las_disagreement_share"] is None
