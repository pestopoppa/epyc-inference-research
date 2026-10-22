import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import adapter
import analysis

HERE = Path(__file__).resolve().parents[1]


def w(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def mk_home(tmp):
    h = tmp / "home"
    cl = []
    for i in range(3):  # 3 requests: user, asst, tool result, asst, user, asst
        pass
    cl = [{"type": "user", "message": {"content": "hello " * 50}},
          {"type": "assistant", "message": {"content": [{"type": "text", "text": "A1"},
                                                          {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x"}}]}},
          {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file body " * 40}]}},
          {"type": "assistant", "message": {"content": [{"type": "text", "text": "A2"}]}},
          {"type": "system", "subtype": "compact_boundary"},
          {"type": "user", "message": {"content": "after compaction " * 20}},
          {"type": "assistant", "message": {"content": [{"type": "text", "text": "A3"}]}}]
    w(h / ".claude/projects/p1/s1.jsonl", cl)
    cx = [{"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"text": "q " * 30}]}},
          {"type": "response_item", "payload": {"type": "function_call", "name": "sh", "arguments": "{}"}},
          {"type": "response_item", "payload": {"type": "function_call_output", "output": "out " * 30}},
          {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"text": "done"}]}}]
    w(h / ".codex/sessions/2026/10/01/rollout-x.jsonl", cx)
    db = h / ".local/share/opencode/opencode.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("create table session(id text, time_created int)")
    con.execute("create table message(id text, session_id text, time_created int, data text)")
    con.execute("create table part(id text, message_id text, session_id text, time_created int, data text)")
    con.execute("insert into session values('s1',1)")
    con.execute("insert into message values('m1','s1',1,?)", (json.dumps({"role": "user"}),))
    con.execute("insert into message values('m2','s1',2,?)", (json.dumps({"role": "assistant"}),))
    con.execute("insert into part values('p1','m1','s1',1,?)", (json.dumps({"type": "text", "text": "ask " * 30}),))
    con.execute("insert into part values('p2','m2','s1',2,?)", (json.dumps({"type": "tool", "tool": "read", "state": {"input": {"filePath": "/y"}, "output": "body " * 30}}),))
    con.execute("insert into part values('p3','m2','s1',3,?)", (json.dumps({"type": "text", "text": "fin"}),))
    con.commit()
    con.close()
    return h


def test_claude_turns_and_reset(tmp_path):
    h = mk_home(tmp_path)
    t = adapter.load_turns(("claude", str(h / ".claude/projects/p1/s1.jsonl"), None))
    assert [x["reset"] for x in t] == [False, False, True]
    assert t[0]["ref"].count("<asst>") == 2          # text + tool_use form one generated run
    assert "A2" in t[1]["ref"] and "A3" in t[2]["ref"]
    prof = adapter.ideal_profile(t)
    assert prof[0][1] == 0 and prof[2][1] == 0       # first request and post-compaction: nothing cacheable
    assert prof[1][1] == len(t[0]["pre"]) + len(t[0]["ref"])   # append-only: whole prior prompt+output


def test_codex_and_opencode(tmp_path):
    h = mk_home(tmp_path)
    c = adapter.load_turns(("codex", str(next((h / ".codex").rglob("*.jsonl"))), None))
    assert len(c) == 2 and c[1]["pre"].startswith("<tool:exec>")
    o = adapter.load_turns(("opencode", "s1", str(h / ".local/share/opencode/opencode.db")))
    assert len(o) == 2 and "fin" in o[1]["ref"]


def test_ideal_known_answer():
    turns = [{"pre": "a" * 100, "ref": "b" * 10, "reset": False}, {"pre": "c" * 40, "ref": "", "reset": False}]
    prof = adapter.ideal_profile(turns)
    assert prof == [(100, 0), (150, 110)]
    st = adapter.session_stats(turns)
    assert abs(st["ideal_share"] - 110 / 250) < 1e-12


def test_common_prefix_len():
    a = "x" * 10000 + "A" + "y" * 5
    b = "x" * 10000 + "B" + "y" * 5
    assert adapter.common_prefix_len(a, b) == 10000
    assert adapter.common_prefix_len("abc", "abcdef") == 3


def test_sampling_deterministic_and_filtered(tmp_path):
    h = mk_home(tmp_path)
    oc = [str(h / ".local/share/opencode/opencode.db")]        # a synthetic snapshot standing in for the ro DBs
    a = adapter.sample_sessions(str(h), n=3, seed=1, min_requests=2, opencode_dbs=oc)
    b = adapter.sample_sessions(str(h), n=3, seed=1, min_requests=2, opencode_dbs=oc)
    assert [x[1] for x in a] == [x[1] for x in b]
    assert {x[0] for x in a} == {"claude", "codex", "opencode"}
    assert adapter.sample_sessions(str(h), n=3, seed=1, min_requests=3, opencode_dbs=oc) != [] and \
        all(len(x[2]) >= 3 for x in adapter.sample_sessions(str(h), n=3, seed=1, min_requests=3, opencode_dbs=oc))
    t = adapter.truncate(a[0][2] * 1, 5, 99)
    assert t == []                                    # prompt cap drops over-long leading request


def test_adapter_cli_aggregates_only(tmp_path):
    h = mk_home(tmp_path)
    r = subprocess.run([sys.executable, str(HERE / "adapter.py"), "--home", str(h), "--opencode-db", str(h / ".local/share/opencode/opencode.db"), "--out", str(tmp_path / "o"),
                        "--n", "3", "--min-requests", "2"], capture_output=True, text=True, check=True)
    assert "hello" not in r.stdout and json.loads(r.stdout)["sessions"] == 3
    assert "hello" not in (tmp_path / "o/manifest.json").read_text()


def write_arm(d, rows, events=3):
    d.mkdir(parents=True)
    (d / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "run_meta.json").write_text(json.dumps({"events": {"observed_eviction_events": events, "log_present": True, "by_pattern": {}}}))


def req(s, k, total, cache, ideal, inf=False):
    return {"session": s, "k": k, "total_tok": total, "cache_n": cache, "prompt_n": total - cache,
            "ideal_tok": ideal, "inferred_evict": inf}


def test_analysis_known_answer(tmp_path):
    base = [req("s1", 0, 100, 0, 0), req("s1", 1, 200, 100, 100), req("s2", 0, 100, 0, 0), req("s2", 1, 200, 100, 100)]
    bad = [req("s1", 0, 100, 0, 0), req("s1", 1, 200, 0, 100, True), req("s2", 0, 100, 0, 0), req("s2", 1, 200, 100, 100)]
    write_arm(tmp_path / "A", bad, events=7)
    write_arm(tmp_path / "B", base, events=0)
    res = analysis.analyse({"A": tmp_path / "A", "B": tmp_path / "B"}, baseline="A", B=300)
    a, b = res["arms"]["A"], res["arms"]["B"]
    assert abs(a["realised_share"] - 100 / 600) < 1e-12 and abs(a["ideal_share"] - 200 / 600) < 1e-12
    assert abs(a["gap_pp"] - 100 * (100 / 600)) < 1e-9 and b["gap_pp"] == 0
    assert a["uncached_per_request"] == 500 / 4 and a["inferred_evictions"] == 1
    assert a["observed_eviction_events"] == 7
    c = res["contrasts_vs_baseline"]["B_minus_A"]["gap_pp"]
    assert c["mean"] < 0 and "gap pp" in analysis.to_markdown(res).lower()


def test_belief_hook_shape_and_write(tmp_path):
    write_arm(tmp_path / "A", [req("s1", 0, 100, 0, 0), req("s1", 1, 200, 150, 190)])
    write_arm(tmp_path / "B", [req("s1", 0, 100, 0, 0), req("s1", 1, 200, 190, 190)])
    res = analysis.analyse({"A": tmp_path / "A", "B": tmp_path / "B"}, "A", B=50)
    pin = {"role": "frontdoor", "model": "m", "kernel_ref": "v10"}
    win = {"window_id": "w", "region_lock_claim": "c", "start_utc": "2026-10-22T00:00:00Z", "end_utc": "2026-10-22T01:00:00Z"}
    ms = analysis.belief_measurements(res, pin, win)
    assert [m["kind"] for m in ms] == ["prefix_hit_share"] * 2 and ms[0]["is_control"] and not ms[1]["is_control"]
    assert (ms[0]["realised_tokens"], ms[0]["ideal_tokens"]) == (150, 190) and ms[0]["source_program"] == "KPF-64"

    class Fake:
        called = None

        @staticmethod
        def write_belief_measurements(out_dir, **kw):
            Fake.called = kw
            return Path(out_dir) / "chunk_kv.beliefs.jsonl"

    p = analysis.write_belief_measurements(tmp_path, run_id="r", producer="p", results_path="x", res=res,
                                           served_pin=pin, window=win, chunk_kv=Fake)
    assert p.name == "chunk_kv.beliefs.jsonl" and Fake.called["measurements"] == ms


def run(args, env_extra=None, window=None):
    env = {k: v for k, v in os.environ.items() if k != "WINDOW_CONFIRMED"}
    if window:
        env["WINDOW_CONFIRMED"] = window
    env.update(env_extra or {})
    return subprocess.run(args, capture_output=True, text=True, env=env)


def test_driver_refuses_without_window(tmp_path):
    r = run([sys.executable, str(HERE / "replay_driver.py"), "--sessions", str(tmp_path), "--out", str(tmp_path)])
    assert r.returncode == 3 and "WINDOW_CONFIRMED" in r.stderr


def test_driver_refuses_production_port(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"sessions": []}))
    for port in ("8080", "8070", "8099"):
        r = run([sys.executable, str(HERE / "replay_driver.py"), "--sessions", str(tmp_path), "--out", str(tmp_path / "o"),
                 "--port", port], window="1")
        assert r.returncode != 0 and "refusing port" in (r.stderr + r.stdout)


def test_launcher_refusals(tmp_path):
    L = str(HERE / "arm_launcher.sh")
    r = run(["bash", L, "--arm", "A", "--dry-run"])
    assert r.returncode == 3 and "WINDOW_CONFIRMED" in r.stderr
    r = run(["bash", L, "--arm", "A", "--dry-run"], {"KPF64_PORT": "8080"}, window="1")
    assert r.returncode == 3 and "refusing port 8080" in r.stderr
    r = run(["bash", L, "--arm", "A", "--dry-run"], {}, window="1")
    assert r.returncode == 2 and "KPF64_SERVER_BIN" in r.stderr
    r = run(["bash", L, "--arm", "A", "--dry-run"], {"KPF64_SERVER_BIN": "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"}, window="1")
    assert r.returncode == 3 and "frozen" in r.stderr


def test_launcher_dry_run_prints_command(tmp_path):
    fake = tmp_path / "cpu" / "llama-server"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    r = run(["bash", str(HERE / "arm_launcher.sh"), "--arm", "B", "--dry-run"], {"KPF64_SERVER_BIN": str(fake)}, window="1")
    assert r.returncode == 0, r.stderr
    assert "--ctx-checkpoints 32" in r.stdout and "--cache-ram 0" in r.stdout and "-np 4" in r.stdout and "nothing started" in r.stdout
    r = run(["bash", str(HERE / "arm_launcher.sh"), "--arm", "Z", "--dry-run"], {"KPF64_SERVER_BIN": str(fake)}, window="1")
    assert r.returncode == 2


def test_driver_stub_end_to_end(tmp_path):
    h = mk_home(tmp_path)
    s = tmp_path / "s"
    subprocess.run([sys.executable, str(HERE / "adapter.py"), "--home", str(h), "--opencode-db", str(h / ".local/share/opencode/opencode.db"), "--out", str(s), "--n", "3", "--min-requests", "2"],
                   check=True, capture_output=True)
    o = tmp_path / "o"
    r = run([sys.executable, str(HERE / "replay_driver.py"), "--sessions", str(s), "--out", str(o), "--dry-run",
             "--history", "live", "--snap-interval", "0.2"], window="1")
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in open(o / "requests.jsonl")]
    assert len(rows) == 7 and all(x["cache_n"] <= x["total_tok"] for x in rows)
    r2 = run([sys.executable, str(HERE / "replay_driver.py"), "--sessions", str(s), "--out", str(o), "--dry-run",
              "--history", "live"], window="1")                        # resumable: nothing re-sent
    assert r2.returncode == 0 and len(open(o / "requests.jsonl").readlines()) == 7
    res = analysis.analyse({"stub": o}, "stub", B=50)
    assert res["arms"]["stub"]["requests"] == 7


def test_error_session_excluded_from_all_arms(tmp_path):
    ok = [req("s1", 0, 100, 0, 0), req("s1", 1, 200, 100, 100), req("s2", 0, 100, 0, 0), req("s2", 1, 200, 100, 100)]
    write_arm(tmp_path / "A", ok)
    write_arm(tmp_path / "B", ok[:2] + [req("s2", 0, 100, 0, 0), {"session": "s2", "k": 1, "error": "timeout"}])
    res = analysis.analyse({"A": tmp_path / "A", "B": tmp_path / "B"}, "A", B=20)
    assert res["arms"]["A"]["sessions"] == 1 and res["arms"]["A"]["excluded_sessions"] == ["s2"]


def _driver(monkeypatch):
    import importlib
    monkeypatch.setenv("WINDOW_CONFIRMED", "1")  # import-time guard; build_body itself starts nothing
    return importlib.import_module("replay_driver")


def test_request_body_cap_and_raw_completion(monkeypatch):
    d = _driver(monkeypatch)
    assert d.cap_n_predict("x" * 30, 128) == 10 and d.cap_n_predict("x" * 4000, 128) == 128
    assert d.cap_n_predict("", 128) == 1 and d.cap_n_predict(None, 128) == 1
    b = d.build_body("PROMPT", "x" * 30, "live", 128)
    assert b["n_predict"] == 10 and b["cache_prompt"] is True and b["prompt"] == "PROMPT" and b["temperature"] == 0.0
    # raw /completion: no chat template, so no thinking knob and no messages/max_tokens keys
    assert not {"chat_template_kwargs", "messages", "max_tokens"} & set(b)
    assert d.build_body("P", "x" * 4000, "reference", 128)["n_predict"] == 1


def test_reference_bound_excludes_unheld_output():
    turns = [{"pre": "AAAA", "ref": "OUT1", "reset": False}, {"pre": "BB", "ref": "OUT2", "reset": False}]
    live = list(adapter.iter_prompts(turns, ["gen1", None]))
    ref = list(adapter.iter_prompts(turns, None, slot_holds_output=False))
    assert live[1][2] == "AAAAgen1" and live[1][1] == "AAAAgen1BB"
    assert ref[1][1] == "AAAAOUT1BB" and ref[1][2] == "AAAA"
    assert adapter.common_prefix_len(ref[1][1], ref[1][2]) == 4  # not 8: the slot never held OUT1
    assert list(adapter.iter_prompts(turns, None))[1][2] == "AAAAOUT1"  # default behaviour unchanged
