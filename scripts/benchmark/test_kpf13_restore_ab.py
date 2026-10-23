"""Server-free tests for scripts/benchmark/kpf13/kpf13_restore_ab.py (KPF-13.1 / 13.1a driver).

No llama-server, no model, no GPU: the cell protocol runs against an in-process fake that models the one server
behaviour the protocol depends on (a fork child takes the longest shared prefix among the OTHER slots' stored prompts),
so the A/B/A/B order, the source construction (A forks at T, B forks at T-N), the no-fork cache-hit controls (rules v2:
CA = cache hit at T with S remaining, CB = cache hit at T-N with N+S remaining) and the judging are all exercised.
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


def ctl(cache=2048, prompt_n=1, **kw):
    r = run(cache=cache, **kw)
    r["prompt_n"] = prompt_n
    return r


CA_EXP = {"cache_n": 2048, "prompt_n": 1}


# ---------------------------------------------------------------- rules are declared and hashed
def test_rules_hash_is_stable_and_complete():
    h = m.rules_sha256()
    assert h == m.rules_sha256() and len(h) == 64
    for k in ("sampling", "twin", "control", "order", "suffix_cells", "cell_invalid_if", "cell_pass", "near_tie_band",
              "arm_verdict", "n_selection", "decisions", "restore_does_not_carry", "first_draft", "supersedes"):
        assert k in m.RULES, k
    assert m.RULES["schema"] == "epyc.kpf13.rules.v2"
    assert m.RULES["sampling"]["n_generated"] == 64 and m.RULES["sampling"]["seed"] == 42
    assert m.RULES["order"].startswith("W1 CA1 A1 CB1 B1 CA2 A2 CB2 B2 W2")
    assert "INFORMATIONAL" in m.RULES["twin"] and "fork_equals_cache_hit" in m.RULES["decisions"]
    assert "%g" not in m.RULES["near_tie_band"] and "0.95" in m.RULES["near_tie_band"]
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


def test_control_basis_checks_partition_and_no_fork():
    ok = m.judge_run([ctl(), ctl()], run(fork=2048), 2048, kind="control", ref_expect=CA_EXP)
    assert ok["verdict"] == "PASS" and ok["basis"] == "control"
    # a control that forked, or that did not reproduce the arm's partition, cannot judge anything
    assert m.judge_run([ctl(fork=2048), ctl()], run(fork=2048), 2048, kind="control",
                       ref_expect=CA_EXP)["verdict"] == "INVALID"
    assert m.judge_run([ctl(cache=1984, prompt_n=65), ctl()], run(fork=2048), 2048, kind="control",
                       ref_expect=CA_EXP)["verdict"] == "INVALID"
    assert m.judge_run([ctl(dn=0), ctl()], run(fork=2048), 2048, kind="control", ref_expect=CA_EXP)["verdict"] == "INVALID"
    # a cache-hit control is NOT a fresh twin: the twin rule would reject it, the control rule accepts it
    assert m.judge_run([ctl(), ctl()], run(fork=2048), 2048)["verdict"] == "INVALID"
    assert m.judge_run([], run(fork=2048), 2048, kind="control")["reasons"] == ["no control"]


def test_near_tie_label_uses_the_given_band():
    toks = list(range(N))
    toks[0] = 999
    gaps = [0.5] * N
    gaps[0] = 0.02  # the KPF-13.1 S=1 first-token gaps were 0.001-0.03
    x = run(tokens=toks, fork=2048, gaps=gaps)
    assert m.judge_run([twin(), twin()], x, 2048)["first_divergence_near_tie"] is False      # v1 floor 1e-3
    v = m.judge_run([twin(), twin()], x, 2048, band=0.05)
    assert v["first_divergence_near_tie"] is True and v["first_divergence_gap"] == 0.02 and v["verdict"] == "FAIL"


def _band_cell(w_gaps, c_gaps, w_toks=None, c_toks=None, S=1):
    w = run(tokens=w_toks, gaps=w_gaps)
    c = ctl(tokens=c_toks, gaps=c_gaps, prompt_n=S)
    return {"T": 2048, "N": 64, "suffix_len": S, "twins": {"W1": w}, "controls": {"CA1": c}}


def test_near_tie_band_is_the_measured_twin_control_spread():
    # no spread at all -> the floor
    b = m.near_tie_band([_band_cell([0.5] * N, [0.5] * N)])
    assert b["band"] == m.NEAR_TIE_FLOOR and b["n_positions"] == N and b["quantile_value"] == 0
    # a spread of 0.04 on every position -> band 0.04 (nearest-rank q95)
    b = m.near_tie_band([_band_cell([0.5] * N, [0.54] * N)])
    assert abs(b["band"] - 0.04) < 1e-9
    # positions at or after the first token divergence are not pooled
    ct = list(range(N))
    ct[10] = 999
    cg = [0.5] * 10 + [0.9] * (N - 10)
    b = m.near_tie_band([_band_cell([0.5] * N, cg, c_toks=ct)])
    assert b["n_positions"] == 10 and b["band"] == m.NEAR_TIE_FLOOR
    # an invalid control (wrong partition) is excluded; no pool -> floor
    bad = _band_cell([0.5] * N, [0.9] * N)
    bad["controls"]["CA1"]["cache_n"] = 0
    assert m.near_tie_band([bad]) == {"band": m.NEAR_TIE_FLOOR, "quantile_value": None, "n_positions": 0,
                                      "floor": m.NEAR_TIE_FLOOR, "q": m.NEAR_TIE_QUANTILE}


def test_control_expect():
    assert m.control_expect(2048, 64, 1, "A") == {"cache_n": 2048, "prompt_n": 1}
    assert m.control_expect(2048, 64, 1, "B") == {"cache_n": 1984, "prompt_n": 65}
    assert m.control_expect(2048, None, 2, "A") == {"cache_n": 2048, "prompt_n": 2}
    assert m.control_expect(2048, None, 2, "B") is None


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

    def __init__(self, fork_min=512, penalty=None, drift_on_fork_len=None, cold_shift=False):
        self.slots: dict[int, list[int]] = {}
        self.fork_min, self.penalty, self.drift = fork_min, penalty, drift_on_fork_len
        self.cold_shift = cold_shift  # a cold full prefill (cache_n 0) takes a different first token: batch shape
        self.log: list[tuple] = []

    def erase(self, slot):
        self.slots.pop(slot, None)
        self.log.append(("erase", slot))

    @staticmethod
    def _lcp(a, b):
        k = 0
        for x, y in zip(a, b):
            if x != y:
                break
            k += 1
        return k

    def complete(self, body, timeout):
        ids, slot = body["prompt"], body["id_slot"]
        fork = body.get("slot_fork", True) and body["cache_prompt"]
        own = self._lcp(ids, self.slots.get(slot, [])) if body["cache_prompt"] else 0
        own = min(own, len(ids) - 1)  # the server always evaluates at least one token
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
            if best >= self.fork_min and best > own:
                nfork = best
        self.slots[slot] = list(ids)
        npred = body["n_predict"]
        self.log.append(("req", slot, len(ids), nfork, npred))
        cache_n = nfork or own
        toks = [(sum(ids) + 7 * i) % 997 for i in range(npred)]
        if self.cold_shift and cache_n == 0 and npred:
            toks[0] = (toks[0] + 1) % 997
        if self.drift is not None and nfork == self.drift and npred > 20:
            toks[20] = (toks[20] + 1) % 997
        da = 20
        if self.penalty is not None and nfork == self.penalty:
            da = 17
        t = {"n_fork_tokens": nfork, "fork_src_kind": "end" if nfork else None, "cache_n": cache_n,
             "prompt_n": len(ids) - cache_n, "prompt_ms": 1.0, "predicted_ms": 1.0}
        if npred:
            t.update(draft_n=30, draft_n_accepted=da)
        return {"http": 200, "wall_s": 0.0, "t_send": 0.0, "t_end": 0.0, "tokens": toks, "probs": [], "timings": t}


def make_prompt_set(path: Path, groups=4, trunk_len=2100):
    lines = []
    for g in range(groups):
        lines.append({"kind": "trunk", "group": g, "ids": [10_000 * (g + 1) + i for i in range(trunk_len)]})
        for i, ln in enumerate((5, 140, 8)):
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
    assert [r["arm"] for r in cell_rows if "_seed" not in r["arm"]] == \
        ["W1", "CA1", "A1", "CB1", "B1", "CA2", "A2", "CB2", "B2", "W2"]
    # CA re-seeds trunk[:T-N] then trunk (the restore source's own construction), CB seeds trunk[:T-N]
    assert [r["arm"] for r in cell_rows if r["arm"].startswith("CA1")] == ["CA1_seed0", "CA1_seed1", "CA1"]
    assert [r["arm"] for r in cell_rows if r["arm"].startswith("CB1")] == ["CB1_seed0", "CB1"]
    byarm = {r["arm"]: r for r in cell_rows}
    N_ = ctx.results["default"][cid]["N"]
    T = 2048
    assert byarm["A1"]["n_fork_tokens"] == T and byarm["A2"]["n_fork_tokens"] == T
    assert byarm["B1"]["n_fork_tokens"] == T - N_ and byarm["B2"]["n_fork_tokens"] == T - N_
    assert byarm["W1"]["n_fork_tokens"] == 0 and byarm["W2"]["n_fork_tokens"] == 0 and byarm["W1"]["cache_n"] == 0
    # the controls never fork and land exactly on the arms' partitions
    for k in ("CA1", "CA2", "CB1", "CB2"):
        assert byarm[k]["n_fork_tokens"] == 0 and byarm[k]["slot"] == 2, k
    assert byarm["CA1"]["cache_n"] == T and byarm["CA1"]["prompt_n"] == 1
    assert byarm["CB1"]["cache_n"] == T - N_ and byarm["CB1"]["prompt_n"] == N_ + 1
    # slot 2 is erased right before and right after every control (nothing can fork from or into it)
    for i, e in enumerate(cl.log):
        if e[0] == "req" and e[1] == 2 and e[4] == N and e[3] == 0 and i + 1 < len(cl.log):
            assert cl.log[i + 1] == ("erase", 2)
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


def test_shape_only_twin_divergence_is_informational_not_a_failure(tmp_path):
    """The KPF-13.1 v1 situation: the cold twin's batch partition gives a different first token, while every arm
    equals its same-partition no-fork control. v2 PASSes the arms and reports the twin column as FAIL (info only)."""
    ctx, pl = run_default(tmp_path, FakeClient(cold_shift=True))
    ana = m.analyze_results(ctx.results, {0}, [64, 512])
    assert ana["configs"]["default"]["A"]["verdict"] == "PASS"
    assert ana["configs"]["default"]["B"]["verdict"] == "PASS"
    assert ana["configs"]["default"]["info_vs_twin"]["A"]["verdict"] == "FAIL"
    d = ana["decisions"]
    assert d["restore_faithful"] is True and d["trigger_port_30141"] is False and d["fork_equals_cache_hit"] is True
    assert all(c["A_vs_twin"]["verdict"] == "FAIL" and c["CA_equals_twin"] is False for c in ana["cells"].values())


def test_v1_cells_without_controls_are_invalid_not_fail(tmp_path):
    ctx, pl = run_default(tmp_path, FakeClient(cold_shift=True))
    for c in ctx.results["default"].values():
        c.pop("controls")
    ana = m.analyze_results(ctx.results, {0}, [64, 512])
    assert ana["configs"]["default"]["A"]["verdict"] == "INCONCLUSIVE"
    assert ana["configs"]["default"]["A"]["fail"] == 0 and ana["decisions"]["trigger_port_30141"] is False


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
    # the skip launch's own controls: a fresh trunk seed then P (cache hit at T, S remaining)
    for c in ctx.results["skip"].values():
        assert set(c["controls"]) == {"CA1", "CA2"}
        assert c["controls"]["CA1"]["cache_n"] == 2048 and c["controls"]["CA1"]["prompt_n"] == c["suffix_len"]


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
    assert "W1 CA1 A1 CB1 B1 CA2 A2 CB2 B2 W2" in out and "W1 CA1 A1 CA2 A2" in out and "rules_sha256=" in out
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


def test_reduced_preset_is_twelve_heldout_cells_one_launch(tmp_path, capsys, monkeypatch):
    started = []
    monkeypatch.setattr(m.Server, "start", lambda self: started.append(1))
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    args = m.parse_args(["run", "--out", str(tmp_path / "o"), "--cpu-list", "24-47,120-143", "--reduced"])
    assert args.configs == ["default"] and args.dev_groups == [] and args.n_list == [64]
    assert args.eval_groups == [0, 1, 2, 3] and args.suffix_lens == [1, 128, 2]
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "24-47,120-143", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(ps), "--reduced", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and not started
    assert out.count("cell g") == 12 and "launch default" in out and "launch skip" not in out and "TOTAL est" in out
    assert out.count("S=2 ") == 4


def test_reduced_run_judges_all_twelve_cells_as_heldout(tmp_path):
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
    assert ana["configs"]["default"]["A"]["attempts"] == 12 and ana["configs"]["default"]["A"]["verdict"] == "PASS"
    assert ana["configs"]["default"]["B"]["attempts"] == 12
    assert all(not c["dev"] for c in ana["cells"].values())


def test_threads_override_and_stop_file_flags_parse(tmp_path):
    a = m.parse_args(["run", "--out", str(tmp_path), "--cpu-list", "24-47,120-143", "--threads", "24",
                      "--stop-file", str(tmp_path / "STOP")])
    assert a.threads == 24 and a.stop_file.endswith("STOP")
    ctx = make_ctx(tmp_path, FakeClient(), argv=["--threads", "24"])
    srv = m.Server(ctx, "default", Path("/x/llama-server"))
    assert srv.argv[srv.argv.index("-t") + 1] == "24" and srv.argv[srv.argv.index("-tb") + 1] == "24"


# ---------------------------------------------------------------- rules v3 (2026-10-23): K >= 3 controls, measured spread
def test_rules_v3_is_declared_hashed_and_distinct_from_v2():
    assert m.RULES["schema"] == "epyc.kpf13.rules.v2" and m.rules_sha256() == m.rules_sha256("v2")
    assert m.RULES_V3["schema"] == "epyc.kpf13.rules.v3"
    assert m.rules_sha256("v3") != m.rules_sha256("v2") and len(m.rules_sha256("v3")) == 64
    for k in ("acceptance_spread", "min_control_repeats", "supersedes", "cell_pass", "decisions", "order"):
        assert k in m.RULES_V3, k
    assert "e0e6e528" in m.RULES_V3["supersedes"] and "decision_grade" in m.RULES_V3["decisions"]
    assert "CA3 CB3" in m.RULES_V3["order"] and "spread" in m.RULES_V3["cell_pass"]["acceptance"]
    json.dumps(m.RULES_V3)


def test_controls_of_returns_every_repeat_in_order():
    c = {"controls": {"CA2": ctl(dn=2), "CA10": ctl(dn=10), "CA1": ctl(dn=1), "CB1": ctl(dn=5), "CA3": ctl(dn=3)}}
    assert [r["draft_n"] for r in m.controls_of(c, "A")] == [1, 2, 3, 10]
    assert [r["draft_n"] for r in m.controls_of(c, "B")] == [5]


def test_spread_is_pooled_from_controls_only():
    cells = [{"cell": "g0", "T": 2048, "N": 64, "suffix_len": 1,
              "controls": {"CA1": ctl(dn=87, da=40), "CA2": ctl(dn=91, da=39), "CA3": ctl(dn=89, da=40),
                           "CB1": ctl(cache=1984, prompt_n=65, dn=81, da=42),
                           "CB2": ctl(cache=1984, prompt_n=65, dn=81, da=42),
                           "CB3": ctl(cache=1984, prompt_n=65, dn=81, da=42)},
              "runs": {"A1": run(dn=200, da=1)}},   # an arm run never widens the spread
             {"cell": "g1", "T": 2048, "N": 64, "suffix_len": 1,
              "controls": {"CA1": ctl(dn=77, da=43), "CA2": ctl(dn=77, da=43), "CA3": ctl(dn=78, da=41)}}]
    sp = m.acceptance_spread(cells)
    assert sp["spread_dn"] == 4 and sp["spread_da"] == 2 and sp["min_k"] == 3 and sp["n_pooled"] == 3


def test_v3_slack_turns_a_v2_zero_width_failure_into_a_pass_but_keeps_real_drops():
    refs = [ctl(dn=127, da=31), ctl(dn=127, da=31), ctl(dn=127, da=31)]
    x = run(dn=123, da=32, fork=2048)
    assert m.judge_run(refs, x, 2048, kind="control", ref_expect=CA_EXP)["verdict"] == "FAIL"         # v2: zero width
    j = m.judge_run(refs, x, 2048, kind="control", ref_expect=CA_EXP, slack=(4, 1))
    assert j["verdict"] == "PASS" and j["accept_interval"] == {"draft_n": [123, 131], "draft_n_accepted": [30, 32]}
    drop = m.judge_run(refs, run(dn=127, da=28, fork=2048), 2048, kind="control", ref_expect=CA_EXP, slack=(4, 1))
    assert drop["verdict"] == "FAIL" and drop["acceptance_drop"] is True


def test_token_rule_with_k_controls_uses_the_smallest_pairwise_agreement():
    base = list(range(N))
    c2 = base[:]
    c2[30] = 999
    refs = [ctl(tokens=base), ctl(tokens=base), ctl(tokens=c2)]
    x = base[:]
    x[40] = 999   # agrees with control 1 for 40 >= 30 positions
    assert m.judge_run(refs, run(tokens=x, fork=2048), 2048, kind="control", ref_expect=CA_EXP)["tokens_ok"] is True
    y = base[:]
    y[10] = 999
    assert m.judge_run(refs, run(tokens=y, fork=2048), 2048, kind="control", ref_expect=CA_EXP)["tokens_ok"] is False


def run_v3(tmp_path, client, k=3):
    ctx = make_ctx(tmp_path, client, argv=["--rules", "v3", "--control-repeats", str(k)])
    pl = m.Plan(ctx, "default", ctx.results, False)
    pl.srv = FakeSrv()
    for st in pl.build():
        if st.fn and ("sources" in st.desc or "cell g" in st.desc):
            st.fn()
    return ctx, pl


def test_v3_cell_order_runs_k_controls_per_arm(tmp_path):
    ctx, pl = run_v3(tmp_path, FakeClient(), k=4)
    cid = [k for k, c in ctx.results["default"].items() if c["group"] == 0 and c["suffix_len"] == 1][0]
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows" / "default.jsonl").read_text().splitlines()]
    arms = [r["arm"] for r in rows if r.get("cell") == cid and "_seed" not in r["arm"]]
    assert arms == ["W1", "CA1", "A1", "CB1", "B1", "CA2", "A2", "CB2", "B2", "CA3", "CB3", "CA4", "CB4", "W2"]
    c = ctx.results["default"][cid]
    assert c["control_repeats"] == 4 and c["rules_schema"] == "epyc.kpf13.rules.v3"
    for k in ("CA3", "CA4"):
        assert c["controls"][k]["cache_n"] == 2048 and c["controls"][k]["n_fork_tokens"] == 0
    for k in ("CB3", "CB4"):
        assert c["controls"][k]["cache_n"] == 2048 - c["N"]


def test_v3_analysis_is_decision_grade_with_three_controls(tmp_path):
    ctx, pl = run_v3(tmp_path, FakeClient(penalty=2048))
    ana = m.analyze_results(ctx.results, {0}, [64, 512], True, "v3")
    assert ana["rules"] == "v3" and ana["rules_sha256"] == m.rules_sha256("v3")
    assert ana["acceptance_spread"]["default"]["spread_dn"] == 0 and ana["min_controls_per_judged_arm"] == 3
    d = ana["decisions"]
    assert d["decision_grade"] is True and d["trigger_port_30141"] is True     # 20 -> 17 accepted: a real drop


def test_v3_rejudge_of_two_control_cells_is_not_decision_grade(tmp_path):
    ctx, pl = run_default(tmp_path, FakeClient())        # v2-shaped cells: 2 controls per arm
    ana = m.analyze_results(ctx.results, {0}, [64, 512], True, "v3")
    assert ana["configs"]["default"]["A"]["verdict"] == "PASS"
    d = ana["decisions"]
    assert d["decision_grade"] is False and d["restore_faithful"] == "NOT_DECISION_GRADE"
    assert d["restore_verdict"] == "PASS"
    v2 = m.analyze_results(ctx.results, {0}, [64, 512])
    assert "decision_grade" not in v2["decisions"] and v2["rules"] == "v2"


def test_v3_flags_and_refusals_parse(tmp_path):
    a = m.parse_args(["run", "--out", str(tmp_path), "--rules", "v3"])
    assert a.control_repeats == 3
    assert m.parse_args(["run", "--out", str(tmp_path)]).control_repeats == 2
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    bad = m.parse_args(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--rules", "v3",
                        "--control-repeats", "2", "--drafter", "dflash2", "--reduced", "--with-skip",
                        "--numa-membind", "3", "--prompt-set", str(ps), "--fork-bin", str(fake_bin(tmp_path))])
    probs, _ = m.preflight(bad, m.PromptSet(ps, bad.trunk_tokens, bad.suffix_lens), [])
    j = "\n".join(probs)
    assert "rules v3 need --control-repeats >= 3" in j and "--with-skip is plain-drafter only" in j
    assert "--numa-membind is the GPU launch prefix only" in j


def test_reduced_with_skip_adds_the_13_1a_launch(tmp_path):
    a = m.parse_args(["run", "--out", str(tmp_path), "--reduced", "--with-skip", "--rules", "v3"])
    assert a.configs == ["default", "skip"] and a.eval_groups == [0, 1, 2, 3]


def test_gpu_prefix_is_numactl_then_taskset(tmp_path):
    ctx = make_ctx(tmp_path, FakeClient(), argv=["--device", "gpu", "--numa-membind", "3", "--cpu-list", "184-191"])
    assert ctx.prefix == ["numactl", "--membind=3", "--", "taskset", "-c", "184-191"]
    srv = m.Server(ctx, "default", Path("/x/llama-server"))
    assert "-tb" not in srv.argv and srv.full_cmd()[:6] == ctx.prefix


def test_shipped_gpu_dflash2_argv_file_is_valid():
    argv, comments = m.load_argv_file(m.HERE / "argv.gpu-27b-dflash2.txt")
    assert m.argv_problems(argv, "dflash2", "gpu") == []
    assert m.flag_val(argv, ("--spec-type",)) == "draft-dflash" and m.flag_val(argv, ("--spec-draft-n-max",)) == "7"
    assert m.flag_val(argv, ("-md",)).endswith("Qwen3.8-27B-DFlash2-Q8_0.gguf")
    assert m.flag_val(argv, ("-m",)).endswith("Qwen3.8-27B-Q8_0.gguf") and "--no-cache-idle-slots" in argv
    assert m.flag_val(argv, ("--device",)) == "ROCm0" and "--slot-save-path" not in argv and comments


def test_dry_run_v3_gpu_dflash2_plan(tmp_path, capsys, monkeypatch):
    started = []
    monkeypatch.setattr(m.Server, "start", lambda self: started.append(1))
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(ps), "--reduced", "--rules", "v3", "--device", "gpu", "--drafter", "dflash2",
                 "--numa-membind", "3", "--argv-file", str(m.HERE / "argv.gpu-27b-dflash2.txt"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and not started
    assert "rules=v3" in out and "controls/arm=3" in out and out.count("cell g") == 12
    assert "CB2 B2 CA3 CB3 W2" in out and "numactl --membind=3 -- taskset -c 184-191" in out


# --- launch read-back thread-affinity judge (ROCm runtime helper thread exemption, GPU only) ---
_ONLINE = list(range(192))
_CLAIM = list(range(184, 192))


def _threads(n_full: int = 0, extra: dict | None = None) -> dict:
    t = {100 + i: [184] for i in range(5)}
    t.update({200 + i: [185 + i] for i in range(7)})
    t.update({300 + i: list(_ONLINE) for i in range(n_full)})
    t.update(extra or {})
    return t


def test_affinity_gpu_one_full_mask_thread_passes_and_is_recorded():
    j = m.judge_thread_affinity(_threads(1), _CLAIM, "gpu", _ONLINE)
    assert j["problem"] is None
    assert j["runtime_threads_unpinned"] == [300]
    assert set(j["allowed"]) <= set(_CLAIM)


def test_affinity_gpu_three_full_mask_threads_abort():
    j = m.judge_thread_affinity(_threads(3), _CLAIM, "gpu", _ONLINE)
    assert j["problem"] and "outside" not in j["problem"] and "3 threads" in j["problem"]


def test_affinity_gpu_partial_out_of_claim_thread_aborts():
    j = m.judge_thread_affinity(_threads(1, {999: [0, 1, 2]}), _CLAIM, "gpu", _ONLINE)
    assert j["problem"] and "outside the claimed" in j["problem"]


def test_affinity_cpu_full_mask_thread_aborts():
    j = m.judge_thread_affinity(_threads(1), _CLAIM, "cpu", _ONLINE)
    assert j["problem"] and "outside the claimed" in j["problem"]
    assert j["runtime_threads_unpinned"] == []


def test_affinity_per_thread_reader_and_union_agree(tmp_path, monkeypatch):
    per = m.read_cpus_allowed_per_thread(os.getpid())
    assert per and os.getpid() in per
    assert m.read_cpus_allowed(os.getpid()) == sorted({c for v in per.values() for c in v})


# ---------------------------------------------------------------- KPF-GL2 (2026-10-23): the affinity read-back reaches the record
def _own_process_server(tmp_path, monkeypatch, env, per_thread):
    """A Server whose 'server process' is this pytest process (real /proc exe + cmdline), with the per-thread affinity,
    online cpus, /proc environ and port ownership injected: read_back() runs its real code path."""
    import sys
    ctx = make_ctx(tmp_path, FakeClient(), argv=["--device", "gpu", "--cpu-list", "184-191", "--numa-membind", "3"])
    srv = m.Server(ctx, "default", Path(sys.executable))
    me = os.getpid()
    cmd = [x for x in Path(f"/proc/{me}/cmdline").read_bytes().decode().split("\0") if x]
    srv.argv = cmd[-1:]

    class _P:
        pid = me
    srv.proc = _P()
    srv.log.parent.mkdir(parents=True, exist_ok=True)
    srv.log.write_text("1.0 llama_context: kv_unified = true\n"
                       "1.0 cross-slot prefix fork enabled: mode = checkpoint, min tokens = 1024, cache_idle_slots = 0\n")
    monkeypatch.setattr(m, "pid_owns_port", lambda pid, port: True)
    monkeypatch.setattr(m, "read_cpus_allowed_per_thread", lambda pid: per_thread)
    monkeypatch.setattr(m, "read_online_cpus", lambda: list(_ONLINE))
    monkeypatch.setattr(m, "read_proc_env", lambda pid: dict(env))
    return ctx, srv


def test_gl2_gpu_readback_with_one_exempt_thread_round_trips_into_the_record(tmp_path, monkeypatch):
    ctx, srv = _own_process_server(tmp_path, monkeypatch,
                                   {"GGML_CUDA_DISABLE_GRAPHS": "1", "HSA_OVERRIDE_CPU_AFFINITY_DEBUG": "0"}, _threads(1))
    rb = srv.read_back()
    srv.readback = rb
    assert rb["problems"] == [] and rb["runtime_threads_unpinned"] == [300] and rb["runtime_threads_unpinned_count"] == 1
    pl = m.Plan(ctx, "default", ctx.results, False)
    pl.srv = srv
    pl.finish_record()
    rec = json.loads(next((ctx.out / "records").glob("kpf13-default-*.json")).read_text())
    ar = rec["argv_readback"]
    for k in m.READBACK_RECORD_KEYS:
        assert k in ar, k
    assert ar["runtime_threads_unpinned"] == [300] and ar["runtime_threads_unpinned_count"] == 1
    assert ar["cpus_allowed_non_exempt"] and set(ar["cpus_allowed_non_exempt"]) <= set(_CLAIM)
    assert ar["hip_graphs"] == "disabled" and ar["env_GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert ar["env_HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] == "0"
    # the belief-measurements line carries the same projection
    line = json.loads((ctx.out / "belief_measurements.jsonl").read_text().splitlines()[-1])
    assert line["argv_readback"]["runtime_threads_unpinned_count"] == 1


def test_gl2_knob_held_records_count_zero_and_graphs_on_is_visible(tmp_path, monkeypatch):
    ctx, srv = _own_process_server(tmp_path, monkeypatch, {}, _threads(0))
    rb = srv.read_back()
    assert rb["runtime_threads_unpinned_count"] == 0 and rb["runtime_threads_unpinned"] == []
    assert rb["hip_graphs"] == "enabled" and rb["env_GGML_CUDA_DISABLE_GRAPHS"] is None
    proj = m.readback_record(rb)
    assert proj["runtime_threads_unpinned_count"] == 0 and proj["hip_graphs"] == "enabled"
    assert m.readback_record(None) is None and m.readback_record({})["runtime_threads_unpinned_count"] is None


def test_gl2_hip_graphs_state_is_presence_of_the_variable():
    assert m.hip_graphs_state({"GGML_CUDA_DISABLE_GRAPHS": "1"}, "gpu") == "disabled"
    assert m.hip_graphs_state({"GGML_CUDA_DISABLE_GRAPHS": "0"}, "gpu") == "disabled"   # ggml: getenv != nullptr
    assert m.hip_graphs_state({}, "gpu") == "enabled"
    assert m.hip_graphs_state({"GGML_CUDA_DISABLE_GRAPHS": "1"}, "cpu") == "n/a (cpu)"


# ---------------------------------------------------------------- KPF-13.1b: rules v4 (spec-OFF discriminator, CB3 order probe, GPU clause)
V4_CELLS = "g0-S2,g1-S2,g3-S2,g2-S1,g2-S2,g3-S1"
V4_ARGV = ["--rules", "v4", "--reduced", "--only-cells", V4_CELLS]


class FakeClientSpecOff(FakeClient):
    """No drafter: no draft counts reported."""

    def complete(self, body, timeout):
        r = super().complete(body, timeout)
        r["timings"].pop("draft_n", None)
        r["timings"].pop("draft_n_accepted", None)
        return r


def run_v4(tmp_path, clients: dict, extra=()):
    ctx = make_ctx(tmp_path, None, argv=V4_ARGV + list(extra))
    for cfg in ctx.args.configs:
        ctx.client = clients[cfg]
        pl = m.Plan(ctx, cfg, ctx.results, False)
        pl.srv = FakeSrv()
        for st in pl.build():
            if st.fn and ("sources" in st.desc or "cell g" in st.desc):
                st.fn()
    return ctx


def test_v4_rules_are_new_and_v3_is_unchanged():
    assert m.rules_sha256("v3") == "7466b774bfb586056805078c72cd4575a21e12b09b34f5c3e177b70d4418e276"
    assert m.RULES_V4["schema"] == "epyc.kpf13.rules.v4" and m.rules_sha256("v4") not in (m.rules_sha256("v3"),
                                                                                           m.rules_sha256("v2"))
    for k in ("gpu_clause", "decisions_v4", "judged_controls", "spec_off_judge", "spec_on_judge", "cells", "order"):
        assert k in m.RULES_V4, k
    assert m.RULES_V4["cells"]["target"] == {"g0-S2": "A", "g1-S2": "A", "g3-S2": "A", "g2-S1": "B"}
    assert "CB_A_abs" in m.RULES_V4["order"] and "0.031567" in m.RULES_V4["spec_on_judge"]
    json.dumps(m.RULES_V4)


def test_v4_args_run_spec_off_first_on_six_cells(tmp_path):
    a = m.parse_args(["run", "--out", str(tmp_path)] + V4_ARGV)
    assert a.configs == ["default_specoff", "default"] and a.control_repeats == 3
    assert sorted(a.only_cells) == [(0, 2), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]


def test_v4_cell_order_has_the_probe_and_all_equal_is_draft_path(tmp_path):
    ctx = run_v4(tmp_path, {"default_specoff": FakeClientSpecOff(), "default": FakeClient()})
    assert sorted(ctx.results) == ["default", "default_specoff"]
    assert sorted(c["cell"] for c in ctx.results["default"].values()) == \
        sorted(["g0-S2-N64", "g1-S2-N64", "g3-S2-N64", "g2-S1-N64", "g2-S2-N64", "g3-S1-N64"])
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows" / "default_specoff.jsonl").read_text().splitlines()]
    arms = [r["arm"] for r in rows if r.get("cell") == "g0-S2-N64" and "_seed" not in r["arm"]]
    assert arms == ["W1", "CA1", "A1", "CB1", "B1", "CA2", "A2", "CB2", "B2", "CA3", "CB3",
                    "A3", "CB_A_abs", "A4", "CB_A_pres", "CA4", "CB_CA_pres", "CA5", "CB_CA_abs", "W2"]
    c = ctx.results["default_specoff"]["g0-S2-N64"]
    T, N_ = 2048, 64
    assert c["spec_off"] is True and c["order_probe"]["A3"]["n_fork_tokens"] == T
    assert c["order_probe"]["A4"]["n_fork_tokens"] == T
    for k in ("CB_A_abs", "CB_A_pres", "CB_CA_pres", "CB_CA_abs"):
        assert c["order_probe"][k]["cache_n"] == T - N_ and c["order_probe"][k]["n_fork_tokens"] == 0, k
    ana = m.analyze_results(ctx.results, set(), [64], False, "v4")
    assert ana["rules"] == "v4" and ana["configs"]["default_specoff"]["A"]["verdict"] == "PASS"
    assert ana["configs"]["default_specoff"]["B"]["verdict"] == "PASS" and ana["configs"]["default"]["A"]["verdict"] == "PASS"
    d = ana["decisions"]
    assert d["decision_grade"] is True and d["kpf13_1b_locus"] == "draft_path"
    assert d["spec_on_reproduces_v3"]["overall"] == "no" and d["cb_order_effect"] == {"default_specoff": "none",
                                                                                      "default": "none"}
    assert d["first_draft_A1_vs_CA1"]["unobservable"] == 6


def test_v4_spec_off_restore_divergence_is_a_target_restore_defect(tmp_path):
    ctx = run_v4(tmp_path, {"default_specoff": FakeClientSpecOff(drift_on_fork_len=2048), "default": FakeClient()})
    ana = m.analyze_results(ctx.results, set(), [64], False, "v4")
    assert ana["configs"]["default_specoff"]["A"]["verdict"] == "FAIL"
    d = ana["decisions"]
    assert d["kpf13_1b_locus"] == "target_restore_defect" and d["kpf13_1_dflash2_gpu"] == "RESTORE_DEFECT_TARGET_SIDE"
    assert "g0-S2:A" in d["kpf13_1b_locus_failing"] and d["kpf19_1_choice_dflash2"].startswith("neither")


def test_v4_spec_on_near_tie_with_clean_spec_off_is_the_draft_path(tmp_path):
    ctx = run_v4(tmp_path, {"default_specoff": FakeClientSpecOff(), "default": FakeClient(drift_on_fork_len=2048)})
    ana = m.analyze_results(ctx.results, set(), [64], False, "v4")
    d = ana["decisions"]
    assert d["kpf13_1b_locus"] == "draft_path" and ana["configs"]["default"]["A"]["verdict"] == "FAIL"
    # the fake carries no top-2 probabilities, so the divergence cannot be labelled a near-tie: not cleared
    assert d["kpf13_1_dflash2_gpu"] == "INVESTIGATE" and d["spec_on_reproduces_v3"]["per_cell"]["g0-S2"] == "moved"


def test_v4_decide_gpu_clause_outcomes():
    nt = {"verdict": "FAIL", "runs": [{"verdict": "FAIL", "accept_ok": True, "first_divergence_near_tie": True,
                                        "acceptance_drop": False, "first_divergence": 2}]}
    ok = {"verdict": "PASS", "runs": [{"verdict": "PASS"}]}
    runs = {("default_specoff", gs, a): ok for gs in list(m.V4_TARGET_CELLS) + m.V4_CHECK_CELLS for a in "AB"}
    runs.update({("default", gs, a): ok for gs in list(m.V4_TARGET_CELLS) + m.V4_CHECK_CELLS for a in "AB"})
    runs[("default", "g0-S2", "A")] = nt
    d = m.decide_v4({}, runs, True)
    assert d["kpf13_1_dflash2_gpu"] == "TARGET_RESTORE_FAITHFUL_DRAFT_PATH_NEAR_TIE"
    assert d["kpf19_1_choice_dflash2"].startswith("keep load_dft") and d["spec_on_reproduces_v3"]["per_cell"]["g0-S2"] == "yes"
    drop = {"verdict": "FAIL", "runs": [{"verdict": "FAIL", "accept_ok": False, "acceptance_drop": True}]}
    runs[("default", "g1-S2", "A")] = drop
    assert m.decide_v4({}, runs, True)["kpf13_1_dflash2_gpu"] == "DRAFT_PATH_ACCEPTANCE_DROP"
    assert m.decide_v4({}, runs, False)["kpf13_1b_locus"] == "NOT_DECISION_GRADE"
    del runs[("default_specoff", "g3-S1", "B")]
    assert m.decide_v4({}, runs, True)["kpf13_1b_locus"] == "INCONCLUSIVE"       # a missing arm is not a pass


def _cb_cell(sigs: dict) -> dict:
    """sigs: CB name -> first token; every run otherwise identical."""
    def r(t):
        return run(tokens=[t] + list(range(1, N)))
    ctl = {"CA1": r(5), "CB1": r(sigs["CB1"]), "CB2": r(sigs["CB2"]), "CB3": r(sigs["CB3"])}
    pr = {k: r(sigs[k]) for k in ("CB_A_abs", "CB_A_pres", "CB_CA_pres", "CB_CA_abs")}
    return {"controls": ctl, "order_probe": pr}


def test_v4_cb_order_effect_classes():
    base = dict(CB1=1, CB2=1, CB3=1, CB_A_abs=1, CB_A_pres=1, CB_CA_pres=1, CB_CA_abs=1)
    assert m.cb_order_class(_cb_cell(base), True)["label"] == "none"
    pred = dict(base, CB3=5, CB_CA_pres=5, CB_CA_abs=5)                     # every CB after a CA differs
    co = m.cb_order_class(_cb_cell(pred), True)
    assert co["label"] == "predecessor" and co["cb3_tokens_equal_ca1"] is True
    slot = dict(base, CB3=7, CB_CA_pres=7, CB_A_pres=7)                      # every CB with slot 3 present differs
    assert m.cb_order_class(_cb_cell(slot), True)["label"] == "slot3_presence"
    assert m.cb_order_class(_cb_cell(dict(base, CB3=9)), True)["label"] == "other"
    c = _cb_cell(base)
    del c["order_probe"]["CB_CA_abs"]
    assert m.cb_order_class(c, True)["label"] == "INVALID"


def test_v4_gpu_clause_needs_two_nonzero_vram_samples_and_graphs_disabled():
    assert m.gpu_clause_problems({"device": "cpu"}) == []
    good = {"device": "gpu", "residency": {"kfd_vram_bytes_samples": [5 << 30, 5 << 30], "hip_graphs": "disabled"}}
    assert m.gpu_clause_problems(good) == []
    assert m.gpu_clause_problems({"device": "gpu"})
    assert m.gpu_clause_problems({"device": "gpu", "residency": {"kfd_vram_bytes_samples": [5 << 30, 0],
                                                                 "hip_graphs": "disabled"}})
    assert m.gpu_clause_problems({"device": "gpu", "residency": {"kfd_vram_bytes_samples": [1, 1],
                                                                 "hip_graphs": "enabled"}})


def test_v4_gpu_cells_carry_own_pid_vram_samples_and_unproven_residency_is_invalid(tmp_path, monkeypatch):
    vram = {"v": 48 << 30}
    monkeypatch.setattr(m, "kfd_pids", lambda: {4242: vram["v"]})
    ctx = make_ctx(tmp_path, None, argv=V4_ARGV + ["--device", "gpu", "--cpu-list", "184-191"])

    class _Srv(FakeSrv):
        class proc:
            pid = 4242
        readback = {"hip_graphs": "disabled"}
    for cfg, cl in (("default_specoff", FakeClientSpecOff()), ("default", FakeClient())):
        ctx.client = cl
        if cfg == "default":
            vram["v"] = 0          # the drafter-on server holds no VRAM: residency not proven
        pl = m.Plan(ctx, cfg, ctx.results, False)
        pl.srv = _Srv()
        for st in pl.build():
            if st.fn and ("sources" in st.desc or "cell g" in st.desc):
                st.fn()
    c = ctx.results["default_specoff"]["g0-S2-N64"]
    assert c["residency"]["own_pid"] == 4242 and c["residency"]["kfd_vram_bytes_samples"] == [48 << 30] * 2
    ana = m.analyze_results(ctx.results, set(), [64], False, "v4")
    assert ana["configs"]["default_specoff"]["A"]["verdict"] == "PASS"
    assert ana["configs"]["default"]["A"]["verdict"] == "INCONCLUSIVE"
    assert ana["cells"]["default:g0-S2-N64"]["A"]["gpu_clause"]


def test_v4_dry_run_gpu_plan_and_refusals(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(m.Server, "start", lambda self: None)
    ps = make_prompt_set(tmp_path / "ps.jsonl")
    base = ["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)),
            "--prompt-set", str(ps), "--device", "gpu", "--drafter", "dflash2", "--numa-membind", "3",
            "--argv-file", str(m.HERE / "argv.gpu-27b-dflash2.txt"), "--dry-run"]
    assert m.main(base + V4_ARGV + ["--draft-debug"]) == 0
    out = capsys.readouterr().out
    assert "rules=v4" in out and "--- launch default_specoff" in out and "--- launch default:" in out
    assert out.count("cell g") == 12 and "speculation OFF: drafter flags removed" in out and "'-md'" in out
    assert "--draft-debug: -lv 4" in out and "CB_A_abs" in out and "PREFLIGHT: inputs PASS" in out
    assert out.index("--- launch default_specoff") < out.index("--- launch default:")
    m.main(base + ["--rules", "v4", "--reduced", "--only-cells", "g0-S2"])
    assert "rules v4 cells are predeclared" in capsys.readouterr().out
    m.main(base + ["--rules", "v3", "--reduced", "--only-cells", "g0-S2"])
    assert "--only-cells is a rules v4 option" in capsys.readouterr().out
    m.main(base + ["--rules", "v4", "--only-cells", V4_CELLS])
    assert "rules v4 (KPF-13.1b) runs on the --reduced preset" in capsys.readouterr().out


def test_spec_off_server_strips_the_drafter_and_debug_log(tmp_path):
    ctx = make_ctx(tmp_path, FakeClient(), argv=["--device", "gpu", "--cpu-list", "184-191", "--draft-debug"])
    ctx.gate_argv, _ = m.load_argv_file(m.HERE / "argv.gpu-27b-dflash2.txt")
    off = m.Server(ctx, "default_specoff", Path("/x/llama-server"))
    assert "-md" not in off.argv and "-lv" not in off.argv and m.spec_off_problems(off.argv) == []
    on = m.Server(ctx, "default", Path("/x/llama-server"))
    assert "-md" in on.argv and on.argv[-2:] == ["-lv", "4"]


def test_log_window_reads_only_appended_bytes_and_bisects_by_time(tmp_path):
    ctx = make_ctx(tmp_path, FakeClient())
    srv = m.Server(ctx, "default", Path("/x/llama-server"))
    srv.log.parent.mkdir(parents=True, exist_ok=True)
    fl = "slot fork: src = 3 (idle), kind = checkpoint, n_fork_tokens = 2048, task.n_tokens = 2050, t = 1.5 ms"
    srv.log.write_text(f"10.0 a\n20.0 {fl}\n30.0 c\n")
    assert srv.lines_between(19.8, 20.1) == [fl]
    assert srv.fork_lines_between(19.8, 20.1)[0]["n_fork_tokens"] == 2048
    with open(srv.log, "a") as fh:
        fh.write("40.0 seq_id 1, draft candidate   0, pos   0:  777 (0.9) 'x'\n41.0 partial-no-newline")
    assert srv.lines_between(39.9, 40.2) == ["seq_id 1, draft candidate   0, pos   0:  777 (0.9) 'x'"]
    assert m.parse_first_draft(srv.lines_between(39.9, 40.2)) == [777]
    assert len(srv.log_lines()) == 4 and srv.lines_between(100, 200) == []
