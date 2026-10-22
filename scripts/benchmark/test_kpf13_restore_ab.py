"""Server-free tests for scripts/benchmark/kpf13/kpf13_restore_ab.py (KPF-13.1 / 13.1a driver).

No llama-server, no model, no GPU: the cell protocol runs against an in-process fake that models the one server
behaviour the protocol depends on (a fork child takes the longest shared prefix among the OTHER slots' stored prompts),
so the A/B/A/B order, the source construction (A forks at T, B forks at T-N) and the judging are all exercised.
Run: python3 -m pytest scripts/benchmark/test_kpf13_restore_ab.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("kpf13_restore_ab", Path(__file__).parent / "kpf13" / "kpf13_restore_ab.py")
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)

N = m.N_GEN


# ---------------------------------------------------------------- helpers
def run(tokens=None, dn=30, da=20, fork=0, cache=0, gaps=None, err=None):
    return {"tokens": list(range(N)) if tokens is None else tokens, "draft_n": dn, "draft_n_accepted": da,
            "n_fork_tokens": fork, "cache_n": cache, "gaps": gaps or [], "error": err}


def twin(**kw):
    return run(**kw)


# ---------------------------------------------------------------- rules are declared and hashed
def test_rules_hash_is_stable_and_complete():
    h = m.rules_sha256()
    assert h == m.rules_sha256() and len(h) == 64
    for k in ("sampling", "twin", "order", "cell_invalid_if", "cell_pass", "arm_verdict", "n_selection", "decisions",
              "restore_does_not_carry", "first_draft"):
        assert k in m.RULES, k
    assert m.RULES["sampling"]["n_generated"] == 64 and m.RULES["sampling"]["seed"] == 42
    assert m.RULES["order"].startswith("W1 A1 B1 A2 B2 W2")
    txt = json.dumps(m.RULES)
    assert "pending_h" in txt and "selector RNG" in txt
    json.dumps(m.RULES)  # serialisable into rules.json


# ---------------------------------------------------------------- judging
def test_identical_restore_passes():
    v = m.judge_run([twin(), twin()], run(fork=2048), expect_fork=2048)
    assert v["verdict"] == "PASS" and v["tokens_ok"] and v["accept_ok"]


def test_token_divergence_fails_and_near_tie_is_labelled():
    toks = list(range(N))
    toks[10] = 999
    gaps = [0.5] * N
    gaps[10] = 1e-5
    v = m.judge_run([twin(), twin()], run(tokens=toks, fork=2048, gaps=gaps), 2048)
    assert v["verdict"] == "FAIL" and not v["tokens_ok"] and v["first_divergence_near_tie"] is True


def test_acceptance_drop_fails_and_is_flagged():
    v = m.judge_run([twin(), twin()], run(da=17, fork=2048), 2048)
    assert v["verdict"] == "FAIL" and v["acceptance_drop"] is True and not v["accept_ok"]


def test_acceptance_gain_fails_without_being_a_drop():
    v = m.judge_run([twin(), twin()], run(da=25, fork=2048), 2048)
    assert v["verdict"] == "FAIL" and v["acceptance_drop"] is False


def test_twin_spread_is_the_tolerance():
    t2 = list(range(N))
    t2[40] = 777
    tw = [twin(da=20), twin(tokens=t2, da=22, dn=31)]
    # the twins agree for 40 tokens, so X agreeing with a twin past position 40 and inside the count ranges passes
    assert m.judge_run(tw, run(da=21, dn=30, fork=1), 1)["verdict"] == "PASS"
    short = list(range(N))
    short[5] = 555
    assert m.judge_run(tw, run(tokens=short, fork=1), 1)["verdict"] == "FAIL"


def test_vacuous_twin_is_invalid_not_pass():
    v = m.judge_run([twin(dn=0), twin(dn=0)], run(dn=0, da=0, fork=1), 1)
    assert v["verdict"] == "INVALID"
    assert m.judge_run([twin(dn=None), twin()], run(fork=1), 1)["verdict"] == "INVALID"


def test_fork_not_taken_where_designed_is_invalid():
    assert m.judge_run([twin(), twin()], run(fork=0), expect_fork=2048)["verdict"] == "INVALID"
    assert m.judge_run([twin(), twin()], run(fork=1984), expect_fork=1984)["verdict"] == "PASS"


def test_non_fresh_twin_is_invalid():
    assert m.judge_run([twin(cache=100), twin()], run(fork=1), 1)["verdict"] == "INVALID"


def test_short_or_errored_run_is_invalid():
    assert m.judge_run([twin(), twin()], run(tokens=[1, 2, 3], fork=1), 1)["verdict"] == "INVALID"
    assert m.judge_run([twin(), twin()], run(err="HTTP 500", fork=1), 1)["verdict"] == "INVALID"


def test_arm_verdict_counts_invalid_as_not_pass():
    ok = {"verdict": "PASS"}
    bad = {"verdict": "FAIL"}
    inv = {"verdict": "INVALID"}
    assert m.arm_verdict([ok, ok])["verdict"] == "PASS"
    assert m.arm_verdict([ok, inv])["verdict"] == "INCONCLUSIVE"
    assert m.arm_verdict([ok, inv, bad])["verdict"] == "FAIL"
    assert m.arm_verdict([])["verdict"] == "INCONCLUSIVE"
    r = m.arm_verdict([ok, inv, bad])
    assert (r["attempts"], r["pass"], r["fail"], r["invalid"]) == (3, 1, 1, 1)


def test_same_run_compares_tokens_and_counts():
    assert m.same_run(run(), run())
    assert not m.same_run(run(), run(da=19))
    assert not m.same_run(run(), run(dn=31))
    t = list(range(N))
    t[63] = 9
    assert not m.same_run(run(), run(tokens=t))


def test_select_n_smallest_passing_else_largest_flagged():
    p, f = {"verdict": "PASS"}, {"verdict": "FAIL"}
    assert m.select_n({64: [f], 512: [p]}, [64, 512]) == {"N": 512, "n_selected_by_default": False}
    assert m.select_n({64: [p, p], 512: [p]}, [512, 64]) == {"N": 64, "n_selected_by_default": False}
    assert m.select_n({64: [f], 512: [f]}, [64, 512]) == {"N": 512, "n_selected_by_default": True}
    assert m.select_n({}, [64, 512])["n_selected_by_default"] is True


# ---------------------------------------------------------------- decisions
def av(v, drop=False):
    return {"verdict": v, "acceptance_drop": drop, "attempts": 3, "pass": 3 if v == "PASS" else 0, "fail": 0, "invalid": 0}


def test_decide_skip_yes_when_restore_and_skip_pass_and_equal():
    d = m.decide({"plain_drafter": True, "default": {"A": av("PASS"), "B": av("PASS")},
                  "skip": {"A": av("PASS"), "equal_to_default_A": True},
                  "full": {"A": av("PASS"), "equal_to_default_A": True}})
    assert d["restore_faithful"] and not d["trigger_port_30141"]
    assert d["kpf13_1a_skip_load_dft_plain"] == "YES" and d["dft_full_equals_restore"] == "YES"
    assert "no re-prefill" in d["kpf19_1_choice"]


def test_decide_skip_no_when_outputs_differ():
    d = m.decide({"plain_drafter": True, "default": {"A": av("PASS"), "B": None},
                  "skip": {"A": av("PASS"), "equal_to_default_A": False}})
    assert d["kpf13_1a_skip_load_dft_plain"] == "NO"
    assert d["dft_full_equals_restore"] == "NOT_MEASURED"


def test_decide_acceptance_drop_is_the_30141_trigger_token_only_is_not():
    d = m.decide({"plain_drafter": True, "default": {"A": av("FAIL", drop=True), "B": av("PASS")}})
    assert d["trigger_port_30141"] is True and d["investigate"] is False
    assert "re-prefill" in d["kpf19_1_choice"]
    d = m.decide({"plain_drafter": True, "default": {"A": av("FAIL", drop=False), "B": av("FAIL")}})
    assert d["trigger_port_30141"] is False and d["investigate"] is True
    assert d["kpf19_1_choice"].startswith("neither")
    assert m.decide({"plain_drafter": True, "default": {"A": av("FAIL"), "B": None},
                     "skip": {"A": av("PASS"), "equal_to_default_A": True}})["kpf13_1a_skip_load_dft_plain"].startswith(
        "NOT_DECIDABLE")


def test_decide_swa_drafter_never_decides_skip():
    d = m.decide({"plain_drafter": False, "default": {"A": av("PASS"), "B": None},
                  "skip": {"A": av("PASS"), "equal_to_default_A": True}})
    assert d["kpf13_1a_skip_load_dft_plain"] == "NOT_MEASURED"


# ---------------------------------------------------------------- fake server + prompt set for the protocol tests
class FakeSrv:
    def fork_lines_between(self, t0, t1):
        return []

    def lines_between(self, t0, t1):
        return []


class FakeClient:
    """Models: a fork child (slot_fork true, cache_prompt true) forks at the longest common prefix with another slot's
    stored prompt (>= fork_min). Every request stores its prompt in its slot (the real slot keeps its KV). Tokens depend
    only on the prompt; draft counts are fixed unless `penalty` hits a given fork length."""

    def __init__(self, fork_min=512, penalty=None, drift_on_fork_len=None):
        self.slots: dict[int, list[int]] = {}
        self.fork_min, self.penalty, self.drift = fork_min, penalty, drift_on_fork_len
        self.log: list[tuple] = []

    def erase(self, slot):
        self.slots.pop(slot, None)
        self.log.append(("erase", slot))

    def complete(self, body, timeout):
        ids, slot = body["prompt"], body["id_slot"]
        fork = body.get("slot_fork", True) and body["cache_prompt"]
        nfork = 0
        if fork:
            best = 0
            for s, t in self.slots.items():
                if s == slot:
                    continue
                k = 0
                for x, y in zip(ids, t):
                    if x != y:
                        break
                    k += 1
                best = max(best, k)
            if best >= self.fork_min:
                nfork = best
        self.slots[slot] = list(ids)
        npred = body["n_predict"]
        self.log.append(("req", slot, len(ids), nfork, npred))
        toks = [(sum(ids) + 7 * i) % 997 for i in range(npred)]
        if self.drift is not None and nfork == self.drift and npred > 20:
            toks[20] = (toks[20] + 1) % 997
        da = 20
        if self.penalty is not None and nfork == self.penalty:
            da = 17
        t = {"n_fork_tokens": nfork, "fork_src_kind": "end" if nfork else None, "cache_n": nfork,
             "prompt_n": len(ids) - nfork, "prompt_ms": 1.0, "predicted_ms": 1.0}
        if npred:
            t.update(draft_n=30, draft_n_accepted=da)
        return {"http": 200, "wall_s": 0.0, "t_send": 0.0, "t_end": 0.0, "tokens": toks, "probs": [], "timings": t}


def make_prompt_set(path: Path, groups=4, trunk_len=2100):
    lines = []
    for g in range(groups):
        lines.append({"kind": "trunk", "group": g, "ids": [10_000 * (g + 1) + i for i in range(trunk_len)]})
        for i, ln in enumerate((5, 140)):
            lines.append({"kind": "pair", "group": g, "suffix_index": i,
                          "suffix_ids": [900_000 + 1000 * g + 100 * i + j for j in range(ln)]})
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def make_ctx(tmp_path, client, **argv_extra):
    base = ["run", "--out", str(tmp_path / "out"), "--cpu-list", "72-95", "--fork-bin", "/nonexistent/llama-server",
            "--prompt-set", str(make_prompt_set(tmp_path / "ps.jsonl"))]
    args = m.parse_args(base + argv_extra.get("argv", []))
    args.window_id = "t"
    ps = m.PromptSet(Path(args.prompt_set), args.trunk_tokens, args.suffix_lens)
    ctx = m.Ctx(args, ps, ["-np", "4", "--kv-unified"], m.Deadline(None))
    ctx.client, ctx.results, ctx.start_utc = client, {}, m.now_iso()
    return ctx


def run_default(tmp_path, client, steps_filter=None):
    ctx = make_ctx(tmp_path, client)
    pl = m.Plan(ctx, "default", ctx.results, False)
    pl.srv = FakeSrv()
    for st in pl.build():
        if st.fn and ("sources" in st.desc or "cell g" in st.desc):
            st.fn()
    return ctx, pl


# ---------------------------------------------------------------- protocol: order, sources, forks
def test_cell_order_is_W1_A1_B1_A2_B2_W2_and_forks_land_where_designed(tmp_path):
    cl = FakeClient()
    ctx, pl = run_default(tmp_path, cl)
    cid = [k for k, c in ctx.results["default"].items() if c["group"] == 0 and c["suffix_len"] == 1][0]
    # read the stored request rows of that cell in order
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows" / "default.jsonl").read_text().splitlines()]
    cell_rows = [r for r in rows if r.get("cell") == cid]
    assert [r["arm"] for r in cell_rows] == ["W1", "A1", "B1", "A2", "B2", "W2"]
    byarm = {r["arm"]: r for r in cell_rows}
    N_ = ctx.results["default"][cid]["N"]
    T = 2048
    assert byarm["A1"]["n_fork_tokens"] == T and byarm["A2"]["n_fork_tokens"] == T
    assert byarm["B1"]["n_fork_tokens"] == T - N_ and byarm["B2"]["n_fork_tokens"] == T - N_
    assert byarm["W1"]["n_fork_tokens"] == 0 and byarm["W2"]["n_fork_tokens"] == 0 and byarm["W1"]["cache_n"] == 0
    # slot 3 (the restore source) is erased before each B child and rebuilt after it
    seq = [e for e in cl.log]
    assert ("erase", 3) in seq


def test_all_equal_server_gives_pass_verdicts_and_a_equals_b(tmp_path):
    ctx, pl = run_default(tmp_path, FakeClient())
    ana = m.analyze_results(ctx.results, {0}, [64, 512])
    assert ana["n_selection"] == {"N": 64, "n_selected_by_default": False}
    assert ana["configs"]["default"]["A"]["verdict"] == "PASS"
    assert ana["configs"]["default"]["B"]["verdict"] == "PASS"
    assert ana["decisions"]["restore_faithful"] is True
    assert all(c["A_equals_B"] for c in ana["cells"].values())
    # held-out cells use the selected N, dev cells sweep both
    dev_ns = sorted({c["N"] for c in ana["cells"].values() if c["dev"]})
    ev_ns = {c["N"] for c in ana["cells"].values() if not c["dev"]}
    assert dev_ns == [64, 512] and ev_ns == {64}


def test_restore_acceptance_drop_is_detected_and_triggers(tmp_path):
    ctx, pl = run_default(tmp_path, FakeClient(penalty=2048))
    ana = m.analyze_results(ctx.results, {0}, [64, 512])
    assert ana["configs"]["default"]["A"]["verdict"] == "FAIL"
    assert ana["configs"]["default"]["B"]["verdict"] == "PASS"
    d = ana["decisions"]
    assert d["trigger_port_30141"] is True and "re-prefill" in d["kpf19_1_choice"]


def test_recompute_divergence_is_isolated_to_arm_b(tmp_path):
    ctx, pl = run_default(tmp_path, FakeClient(drift_on_fork_len=2048 - 64))
    ana = m.analyze_results(ctx.results, {0}, [64, 512])
    # N=64 recompute (fork at 1984) drifts on DEV, so N is NOT 64; the A arm is untouched
    assert ana["n_selection"]["N"] == 512
    assert ana["configs"]["default"]["A"]["verdict"] == "PASS"


def test_twins_are_reused_across_the_n_sweep(tmp_path):
    ctx, pl = run_default(tmp_path, FakeClient())
    dev = [c for c in ctx.results["default"].values() if c["group"] == 0 and c["suffix_len"] == 1]
    assert len(dev) == 2
    have_twin = [c for c in dev if c["twins"].get("W1")]
    inherited = [c for c in dev if c.get("twins_from")]
    assert len(have_twin) == 1 and len(inherited) == 1


def test_non_default_launch_is_judged_against_default_twins(tmp_path):
    cl = FakeClient()
    ctx, pl = run_default(tmp_path, cl)
    sk = m.Plan(ctx, "skip", ctx.results, False)
    sk.srv = FakeSrv()
    steps = sk.build()
    assert [s.desc for s in steps if "cell g" in s.desc] and all("g0" not in s.desc for s in steps if "cell g" in s.desc)
    for st in steps:
        if st.fn and ("sources" in st.desc or "cell g" in st.desc):
            st.fn()
    ana = m.analyze_results(ctx.results, {0}, [64, 512])
    assert ana["configs"]["skip"]["A"]["verdict"] == "PASS"
    assert ana["decisions"]["kpf13_1a_skip_load_dft_plain"] == "YES"


def test_skip_launch_that_diverges_is_NO_and_a_drifting_own_twin_is_invalid(tmp_path):
    ctx, pl = run_default(tmp_path, FakeClient())
    sk = m.Plan(ctx, "skip", ctx.results, False)
    sk.srv = FakeSrv()
    sk.ctx.client = FakeClient(penalty=2048)  # the skip launch loses acceptance
    for st in sk.build():
        if st.fn and ("sources" in st.desc or "cell g" in st.desc):
            st.fn()
    ana = m.analyze_results(ctx.results, {0}, [64, 512])
    assert ana["decisions"]["kpf13_1a_skip_load_dft_plain"] == "NO"
    # launch variance: own W1 outside the default twins' range => the cell is INVALID, not a pass
    for c in ctx.results["skip"].values():
        c["twins"]["W1"]["tokens"][3] += 1
    ana = m.analyze_results(ctx.results, {0}, [64, 512])
    assert ana["configs"]["skip"]["A"]["verdict"] == "INCONCLUSIVE" or ana["configs"]["skip"]["A"]["invalid"] > 0
    assert ana["decisions"]["kpf13_1a_skip_load_dft_plain"] in ("NO", "INCONCLUSIVE")


# ---------------------------------------------------------------- deadline / port ownership / parse
def test_deadline_fits_and_expires():
    t = [1000.0]
    d = m.Deadline(1500.0, clock=lambda: t[0])
    assert d.remaining() == 500 and not d.expired()
    assert d.fits(100, reserve=150) and not d.fits(400, reserve=150)
    t[0] = 1500.1
    assert d.expired()
    assert m.Deadline(None).remaining() == float("inf") and not m.Deadline(None).expired()


def test_utc_epoch_is_timezone_independent():
    assert m.utc_epoch("1970-01-02T00:00:00Z") == 86400.0
    with pytest.raises(ValueError):
        m.utc_epoch("tomorrow")


def test_pid_owns_port_proves_the_listener_not_the_answerer():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert m.port_listening(port)
        assert m.pid_owns_port(os.getpid(), port) is True
        assert m.pid_owns_port(os.getppid() if os.getppid() != os.getpid() else 1, port) is False
        assert m.pid_owns_port(2 ** 22 - 1, port) is False  # no such pid
    finally:
        s.close()
    assert not m.port_listening(port)


def test_parse_first_draft_stops_at_next_block():
    lines = [" - seq_id 0, draft candidate   0, pos   0:  11 ( 0.9) 'a'", " - seq_id 0, draft candidate   1, pos   0:  12",
             " - seq_id 0, draft candidate   0, pos   1:  21 ( 0.8) 'b'", " - seq_id 0, draft candidate   0, pos   0:  31 ( 0.7) 'c'"]
    assert m.parse_first_draft(lines) == [11, 21]
    assert m.parse_first_draft(["nothing"]) is None


def test_top2_gaps():
    e = {"top_logprobs": [{"id": 1, "logprob": -0.1}, {"id": 2, "logprob": -2.0}]}
    g = m.top2_gaps([e, {"top_logprobs": [{"id": 1, "prob": 1.0}]}])
    assert g[0] is not None and g[0] > 0.7 and g[1] is None


def test_argv_problems():
    base = ["-np", "4", "--kv-unified", "--spec-type", "draft-mtp"]
    assert m.argv_problems(base, "plain-mtp") == []
    assert any("--port" in p for p in m.argv_problems(base + ["--port", "1"], "plain-mtp"))
    assert any("-t" in p for p in m.argv_problems(base + ["-t", "24"], "plain-mtp"))
    assert any("draft-mtp" in p for p in m.argv_problems(["-np", "4", "--kv-unified"], "plain-mtp"))
    assert any("DFlash2" in p for p in m.argv_problems(base, "dflash2"))
    assert m.argv_problems(["-np", "4", "--kv-unified", "--spec-type", "draft-dflash", "-md", "/m/x-DFlash2-Q8_0.gguf"],
                           "dflash2") == []
    assert any("-np" in p for p in m.argv_problems(["--kv-unified", "--spec-type", "draft-mtp"], "plain-mtp"))
    # the GPU argv carries its own -t (host threads); only the CPU device forbids it
    assert m.argv_problems(base + ["-t", "8"], "plain-mtp", device="gpu") == []


def test_shipped_argv_file_is_valid():
    argv, comments = m.load_argv_file(m.DEFAULT_ARGV)
    assert m.argv_problems(argv, "plain-mtp") == []
    assert "--no-cache-idle-slots" in argv and "--kv-unified" in argv
    assert comments


# ---------------------------------------------------------------- CLI: dry-run estimate, refusals, metadata first
def fake_bin(tmp_path):
    b = tmp_path / "bin" / "llama-server"
    b.parent.mkdir()
    b.write_bytes(b"x --slot-fork-min-tokens x --slot-fork-dft-full x LLAMA_KPF_MUTATE x dft_partial_noop x")
    return b


def test_dry_run_prints_plan_and_estimate_and_starts_nothing(tmp_path, capsys, monkeypatch):
    started = []
    monkeypatch.setattr(m.Server, "start", lambda self: started.append(1))
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "72-95", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(ps), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and not started
    assert "TOTAL est" in out and "launch default" in out and "launch skip" in out and "launch full" in out
    assert "W1 A1 B1 A2 B2 W2" in out and "rules_sha256=" in out
    assert not (tmp_path / "o").exists()


def test_execute_refuses_without_deadline_and_region_claim(tmp_path, capsys, monkeypatch):
    started = []
    monkeypatch.setattr(m.Server, "start", lambda self: started.append(1))
    monkeypatch.delenv("KPF13_REGION", raising=False)
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "72-95", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(ps)])
    out = capsys.readouterr().out
    assert rc == 2 and not started
    assert "--deadline-utc" in out and "region-lock claim" in out


def test_rules_are_written_before_the_first_launch(tmp_path, monkeypatch):
    seen = {}

    def fake_start(self):
        seen["rules"] = (self.ctx.out / "rules.json").exists()
        seen["meta"] = json.loads((self.ctx.out / "run_meta.json").read_text())
        raise m.Abort("test stop")

    monkeypatch.setattr(m.Server, "start", fake_start)
    monkeypatch.setenv("KPF13_REGION", "72-95")
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "72-95", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(ps), "--deadline-s", "3600"])
    assert rc == 3
    assert seen["rules"] is True
    assert seen["meta"]["rules_sha256"] == m.rules_sha256() and seen["meta"]["rules_declared_before_first_launch"]
    saved = json.loads((tmp_path / "o" / "rules.json").read_text())
    assert saved["rules_sha256"] == m.rules_sha256()


def test_preflight_catches_missing_knob_overlap_and_bad_ports(tmp_path):
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    b = tmp_path / "llama-server"
    b.write_bytes(b"nothing here")
    args = m.parse_args(["run", "--out", str(tmp_path / "o"), "--cpu-list", "72-95", "--fork-bin", str(b),
                         "--prompt-set", str(ps), "--port", "8083", "--dev-groups", "1"])
    pset = m.PromptSet(ps, args.trunk_tokens, args.suffix_lens)
    argv, _ = m.load_argv_file(m.DEFAULT_ARGV)
    probs, _ = m.preflight(args, pset, argv)
    j = "\n".join(probs)
    assert "8000-8999" in j and "knob" in j and "overlap" in j


def test_analyze_command_recomputes_from_stored_cells(tmp_path, capsys):
    ctx, pl = run_default(tmp_path, FakeClient())
    for cfg, cells in ctx.results.items():
        for cid, c in cells.items():
            m.write_json(tmp_path / "out" / "cells" / cfg / f"{cid}.json", c)
    args = m.parse_args(["analyze", "--out", str(tmp_path / "out")])
    assert m.cmd_analyze(args) == 0
    v = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert v["decisions"]["restore_faithful"] is True and v["rules_sha256"] == m.rules_sha256()


def test_reduced_preset_is_eight_heldout_cells_one_launch(tmp_path, capsys, monkeypatch):
    started = []
    monkeypatch.setattr(m.Server, "start", lambda self: started.append(1))
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    args = m.parse_args(["run", "--out", str(tmp_path / "o"), "--cpu-list", "24-47,120-143", "--reduced"])
    assert args.configs == ["default"] and args.dev_groups == [] and args.n_list == [64]
    assert args.eval_groups == [0, 1, 2, 3] and args.suffix_lens == [1, 128]
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "24-47,120-143", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(ps), "--reduced", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and not started
    assert out.count("cell g") == 8 and "launch default" in out and "launch skip" not in out and "TOTAL est" in out


def test_reduced_run_judges_all_eight_cells_as_heldout(tmp_path):
    args = m.parse_args(["run", "--out", str(tmp_path / "out"), "--cpu-list", "24-47", "--reduced",
                         "--prompt-set", str(make_prompt_set(tmp_path / "ps.jsonl"))])
    args.window_id = "t"
    ps = m.PromptSet(Path(args.prompt_set), args.trunk_tokens, args.suffix_lens)
    ctx = m.Ctx(args, ps, ["-np", "4", "--kv-unified"], m.Deadline(None))
    ctx.client, ctx.results, ctx.start_utc = FakeClient(), {}, m.now_iso()
    pl = m.Plan(ctx, "default", ctx.results, False)
    pl.srv = FakeSrv()
    for st in pl.build():
        if st.fn and ("sources" in st.desc or "cell g" in st.desc):
            st.fn()
    ana = m.analyze_results(ctx.results, set(args.dev_groups), args.n_list)
    assert ana["configs"]["default"]["A"]["attempts"] == 8 and ana["configs"]["default"]["A"]["verdict"] == "PASS"
    assert ana["configs"]["default"]["B"]["attempts"] == 8
    assert all(not c["dev"] for c in ana["cells"].values())
