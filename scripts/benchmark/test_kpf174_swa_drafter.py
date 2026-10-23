"""Server-free tests for scripts/benchmark/kpf174/kpf174_swa_drafter.py (KPF-17.4 / 17.4a / 19.1a on 27B + DFlash2).

No llama-server, no model, no GPU. The cell protocol runs against an in-process fake that models what the protocol
relies on: a request with checkpoint_at [-1] pins a checkpoint at its prompt end; a fork child takes the largest pinned
checkpoint <= its common prefix with another slot; an emulated drafter-window defect lowers draft_n_accepted.
Run: python3 -m pytest scripts/benchmark/test_kpf174_swa_drafter.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("kpf174_swa_drafter",
                                               Path(__file__).parent / "kpf174" / "kpf174_swa_drafter.py")
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)
k13 = m.k13
N = m.N_GEN
GEO = m.junctions(2048, 4, 512)
MODEL = "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf"


def test_geometry_matches_the_drafter_cache_bound():
    assert GEO["swa_cells"] == 2048 * 4 + 512 == 8704
    assert GEO["c"] == [2148, 4096, 10240] and GEO["L"] == 14336
    assert GEO["pruned"] == {2148: True, 4096: True, 10240: False}
    assert GEO["c"][2] > GEO["swa_cells"]          # the 17.4a junction is above the SWA cache bound


def test_rules_are_declared_and_hashed():
    h = m.rules_sha256()
    assert h == m.rules_sha256() and len(h) == 64
    for k in ("cell", "cell_invalid_if", "cell_judge", "launch_verdict", "decisions", "recorded_range", "min_controls"):
        assert k in m.RULES, k
    assert set(m.RULES["decisions"]) == {"kpf17_4", "kpf19_1a", "kpf17_4a"}
    assert "MEASUREMENT" in m.RULES["decisions"]["kpf17_4a"]
    json.dumps(m.RULES)


# ---------------------------------------------------------------- pure judging
def r(dn=30, da=20, fork=0, cache=0, prompt_n=64, tokens=None, src=None):
    return {"tokens": list(range(N)) if tokens is None else tokens, "draft_n": dn, "draft_n_accepted": da,
            "n_fork_tokens": fork, "cache_n": cache, "prompt_n": prompt_n, "gaps": [], "fork_src_slot": src}


def cell(c=2148, arms=None, controls=None, pinned=True, pruned=True, g=1):
    return {"cell": f"g{g}-c{c}", "group": g, "c": c, "S": 64, "pruned": pruned, "ok": True,
            "source_checkpoints": [{"n_tokens": c, "pinned": pinned, "pos_min": c - 1, "pos_max": c - 1}],
            "arms": arms if arms is not None else [r(fork=c, cache=c, src=0), r(fork=c, cache=c, src=0)],
            "controls": controls if controls is not None else [r(cache=c) for _ in range(3)],
            "fresh": r(cache=0, prompt_n=c + 64)}


def test_honest_cell_passes_and_records_no_loss():
    j = m.judge_cell(cell(), (0, 0))
    assert j["verdict"] == "PASS" and j["n_controls"] == 3
    assert j["kpf17_4a"] == {"fork_taken": [True, True], "loss_tokens": [0, 0]}


def test_drafter_window_defect_is_an_acceptance_failure_beyond_the_spread():
    bad = cell(arms=[r(da=12, fork=2148, cache=2148, src=0), r(fork=2148, cache=2148, src=0)])
    j = m.judge_cell(bad, (2, 1))
    assert j["verdict"] == "FAIL" and j["acceptance_drop"] is True
    ok = cell(arms=[r(da=19, fork=2148, cache=2148, src=0), r(fork=2148, cache=2148, src=0)])
    assert m.judge_cell(ok, (2, 1))["verdict"] == "PASS"     # inside the measured spread


def test_fork_not_taken_or_from_the_wrong_slot_or_missing_pin_is_invalid():
    assert m.judge_cell(cell(arms=[r(fork=0, cache=0, src=None)] * 2), (0, 0))["verdict"] == "INVALID"
    assert m.judge_cell(cell(arms=[r(fork=2148, cache=2148, src=3)] * 2), (0, 0))["verdict"] == "INVALID"
    assert m.judge_cell(cell(pinned=False), (0, 0))["verdict"] == "INVALID"


def test_partition_mismatch_in_a_control_is_invalid_and_two_controls_are_not_enough():
    assert m.judge_cell(cell(controls=[r(cache=2000)] * 3), (0, 0))["verdict"] == "INVALID"
    assert m.judge_cell(cell(controls=[r(cache=2148)] * 2), (0, 0))["verdict"] == "INVALID"
    vac = cell(controls=[r(cache=2148, dn=0)] * 3)
    assert m.judge_cell(vac, (0, 0))["verdict"] == "INVALID"


def test_17_4a_loss_is_measured_not_judged():
    lossy = cell(c=10240, pruned=False, arms=[r(fork=10240, cache=10240, prompt_n=64 + 512, src=0)] * 2)
    j = m.judge_cell(lossy, (0, 0))
    assert j["kpf17_4a"]["loss_tokens"] == [512, 512]


def test_launch_verdicts_and_mutant_scope():
    P, F, I = {"verdict": "PASS"}, {"verdict": "FAIL"}, {"verdict": "INVALID"}
    assert m.launch_verdict("honest", [P, P])["verdict"] == "PASS"
    assert m.launch_verdict("honest", [P, F])["verdict"] == "FAIL"
    assert m.launch_verdict("honest", [P, I])["verdict"] == "INCONCLUSIVE"
    assert m.launch_verdict("mutant", [P, F])["verdict"] == "DETECTED"
    assert m.launch_verdict("mutant", [P, P])["verdict"] == "NOT_DETECTED"
    # partial_noop: a failure only at a live (unpruned) junction is out of scope
    v = m.launch_verdict("mutant", [P, P, F], [True, True, False])
    assert v["verdict"] == "NOT_DETECTED" and v["out_of_scope"] == 1


def test_decisions_need_the_mutant_detected():
    lv = lambda h, fo, mnt, mpn: {"honest": {"verdict": h}, "flag_on": {"verdict": fo},
                                  "mut_no_transfer": {"verdict": mnt}, "mut_partial_noop": {"verdict": mpn}}
    d = m.decide(lv("PASS", "PASS", "DETECTED", "DETECTED"))
    assert d["kpf17_4"] == "PASS" and d["kpf19_1a"] == "PASS" and d["pinned_junction_on_swa_drafter_allowed"]
    d = m.decide(lv("PASS", "PASS", "NOT_DETECTED", "NOT_DETECTED"))
    assert d["kpf17_4"] == "NOT_ESTABLISHED" and d["kpf19_1a"] == "NOT_ESTABLISHED"
    assert not d["pinned_junction_on_swa_drafter_allowed"]
    assert m.decide(lv("FAIL", "PASS", "DETECTED", "DETECTED"))["kpf17_4"] == "FAIL"
    assert m.decide(lv("PASS", "FAIL", "DETECTED", "DETECTED"))["kpf19_1a"] == "FAIL"
    d = m.decide({"honest": {"verdict": "PASS"}}, prior_191a={"verdict": "PASS", "record": "x.json"})
    assert d["kpf19_1a"].startswith("PRIOR: PASS") and d["kpf17_4"] == "INCONCLUSIVE"


def test_prior_191a_is_found_in_records(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"kpf_task": "KPF-18", "decisions": {"kpf19_1a": "PASS"}}))
    assert m.find_prior_191a([str(tmp_path / "*.json")]) is None
    (tmp_path / "b.json").write_text(json.dumps({"kpf_task": "KPF-17.4+17.4a+19.1a",
                                                 "decisions": {"kpf19_1a": "INCONCLUSIVE"}}))
    assert m.find_prior_191a([str(tmp_path / "*.json")]) is None
    (tmp_path / "c.json").write_text(json.dumps({"kpf_task": "KPF-19.1a", "decisions": {"kpf19_1a": "PASS"}}))
    assert m.find_prior_191a([str(tmp_path / "*.json")])["verdict"] == "PASS"


# ---------------------------------------------------------------- the protocol against a fake server
class FakeClient:
    """slot -> stored prompt; pinned checkpoints at prompt ends of requests with checkpoint_at [-1]. A fork child (slot_fork
    absent, cache_prompt true) forks from the largest pinned checkpoint p <= lcp (>= fork_min) on ANOTHER slot. `defect`
    lowers draft_n_accepted of forked children at the given junctions (an emulated drafter-window shortfall)."""

    def __init__(self, fork_min=1024, defect=(), loss=0):
        self.store: dict[int, list[int]] = {}
        self.pins: dict[int, list[int]] = {}
        self.fork_min, self.defect, self.loss = fork_min, set(defect), loss
        self.log: list[tuple] = []

    def erase(self, slot):
        self.store.pop(slot, None)
        self.pins.pop(slot, None)
        self.log.append(("erase", slot))

    def slots_(self):
        return [{"id": s, "checkpoints": [{"n_tokens": p, "pos_min": p - 1, "pos_max": p - 1, "pinned": True,
                                           "size_mib": 150.0} for p in self.pins.get(s, [])]} for s in range(4)]

    def complete(self, body, timeout):
        ids, slot = body["prompt"], body["id_slot"]
        fork = body.get("slot_fork", True) and body["cache_prompt"]
        own = 0
        if body["cache_prompt"]:
            for x, y in zip(ids, self.store.get(slot, [])):
                if x != y:
                    break
                own += 1
        own = min(own, len(ids) - 1)
        nfork, src = 0, None
        if fork:
            for s, t in self.store.items():
                if s == slot:
                    continue
                k = 0
                for x, y in zip(ids, t):
                    if x != y:
                        break
                    k += 1
                cands = [p for p in self.pins.get(s, []) if p <= k and p < len(ids) and p >= self.fork_min]
                if cands and max(cands) > max(nfork, own):
                    nfork, src = max(cands), s
        if nfork:
            self.pins[slot] = [p for p in self.pins.get(src, []) if p <= nfork]
        elif own < len(self.store.get(slot, [])):
            self.pins[slot] = [p for p in self.pins.get(slot, []) if p <= own]
        self.store[slot] = list(ids)
        if body.get("checkpoint_at") == [-1]:
            self.pins.setdefault(slot, []).append(len(ids))
        npred = body["n_predict"]
        self.log.append(("req", slot, len(ids), nfork, npred))
        cache_n = nfork or own
        t = {"n_fork_tokens": nfork, "fork_src_slot": src, "fork_src_kind": "checkpoint" if nfork else None,
             "cache_n": cache_n, "prompt_n": len(ids) - cache_n + (self.loss if nfork else 0), "prompt_ms": 1.0,
             "predicted_ms": 1.0}
        if npred:
            t.update(draft_n=30, draft_n_accepted=12 if (nfork and nfork in self.defect) else 20)
        toks = [(sum(ids) + 7 * i) % 997 for i in range(npred)]
        return {"http": 200, "wall_s": 0.0, "t_send": 0.0, "t_end": 0.0, "tokens": toks, "probs": [], "timings": t}


class FakeClientWithSlots(FakeClient):
    def slots(self):
        return self.slots_()


def make_prompts(path: Path, groups=(1, 2), n=15000, model=MODEL) -> Path:
    lines = [{"kind": "meta", "model": model}]
    for g in groups:
        lines.append({"kind": "trunk", "group": g, "ids": [10_000 * g + i for i in range(n)]})
        lines.append({"kind": "pair", "group": g, "suffix_index": 0, "suffix_ids": [900_000 + 1000 * g + j for j in range(80)]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def make_ctx(tmp_path, extra=()):
    a = m.parse_args(["run", "--out", str(tmp_path / "out"), "--cpu-list", "184-191", "--fork-bin", "/x/llama-server",
                      "--prompt-set", str(make_prompts(tmp_path / "ps.jsonl")), *extra])
    a.window_id = "t"
    ctx = k13.Ctx(a, None, ["-np", "4", "--kv-unified"], k13.Deadline(None))
    ctx.start_utc = k13.now_iso()
    return a, ctx


def run_launch(tmp_path, launch, client, results=None):
    a, ctx = make_ctx(tmp_path)
    ctx.client = client
    ps = m.Prompts(Path(a.prompt_set))
    results = {} if results is None else results
    rn = m.Runner(ctx, GEO, ps, launch, results)
    for g in a.groups:
        rn.group(g)
    return ctx, results, client


def test_protocol_sources_forks_controls_and_partition(tmp_path):
    ctx, res, cl = run_launch(tmp_path, "honest", FakeClientWithSlots())
    cells = res["honest"]
    assert sorted(cells) == ["g1-c10240", "g1-c2148", "g1-c4096", "g2-c10240", "g2-c2148", "g2-c4096"]
    for cid, c in cells.items():
        assert [x["n_fork_tokens"] for x in c["arms"]] == [c["c"], c["c"]] and all(x["fork_src_slot"] == 0 for x in c["arms"])
        assert len(c["controls"]) == 3
        assert all(x["n_fork_tokens"] == 0 and x["cache_n"] == c["c"] and x["prompt_n"] == 64 for x in c["controls"])
        assert c["fresh"]["cache_n"] == 0 and c["fresh"]["n_fork_tokens"] == 0
        assert {ck["n_tokens"] for ck in c["source_checkpoints"]} == {2148, 4096, 10240, 14336}
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows" / "honest.jsonl").read_text().splitlines()]
    order = [x["arm"] for x in rows if x.get("c") == 4096 and x.get("group") == 1 and "_seed" not in x["arm"]]
    assert order == ["X1", "C1", "X2", "C2", "C3", "F"]
    # the control replays the source's build sequence up to c (same partition), never beyond
    seeds = [x for x in rows if x.get("c") == 4096 and x.get("group") == 1 and x["arm"].startswith("C1_seed")]
    assert [x["arm"] for x in seeds] == ["C1_seed_2148", "C1_seed_4096"]
    src = [x["arm"] for x in rows if x.get("arm", "").startswith("source") and x["group"] == 1]
    assert src == ["source_2148", "source_4096", "source_10240", "source_14336"]
    ana = m.analyze(res, GEO)
    assert ana["launches"]["honest"]["verdict"] == "PASS"
    assert all(e["loss_tokens"] == [0, 0] and e["fork_taken"] == [True, True] for e in ana["kpf17_4a"])
    assert {e["c"]: e["above_swa_cells"] for e in ana["kpf17_4a"]}[10240] is True


def test_full_matrix_decisions_from_fakes(tmp_path):
    res: dict = {}
    run_launch(tmp_path / "h", "honest", FakeClientWithSlots(), res)
    run_launch(tmp_path / "f", "flag_on", FakeClientWithSlots(), res)
    run_launch(tmp_path / "n", "mut_no_transfer", FakeClientWithSlots(defect=GEO["c"]), res)
    run_launch(tmp_path / "p", "mut_partial_noop", FakeClientWithSlots(defect=[2148, 4096]), res)
    d = m.analyze(res, GEO)["decisions"]
    assert d["kpf17_4"] == "PASS" and d["kpf19_1a"] == "PASS"
    # partial_noop failing only at the LIVE junction does not count as detection
    res2 = {k: v for k, v in res.items() if k != "mut_partial_noop"}
    run_launch(tmp_path / "p2", "mut_partial_noop", FakeClientWithSlots(defect=[10240]), res2)
    assert m.analyze(res2, GEO)["decisions"]["kpf17_4"] == "NOT_ESTABLISHED"


def test_17_4a_measures_a_reprocessing_loss(tmp_path):
    ctx, res, cl = run_launch(tmp_path, "honest", FakeClientWithSlots(loss=300))
    ana = m.analyze(res, GEO)
    assert all(e["loss_tokens"] == [300, 300] for e in ana["kpf17_4a"])
    assert ana["launches"]["honest"]["verdict"] == "PASS"      # a loss is measured, it is not a verdict


def test_resume_skips_stored_cells(tmp_path):
    a, ctx = make_ctx(tmp_path)
    ps = m.Prompts(Path(a.prompt_set))
    stored = {"honest": {"g1-c2148": {"ok": True}, "g1-c4096": {"ok": True}, "g1-c10240": {"ok": True}}}
    rn = m.Runner(ctx, GEO, ps, "honest", stored)
    assert rn.pending([1, 2]) == [(2, 10240), (2, 4096), (2, 2148)]


# ---------------------------------------------------------------- CLI
def fake_bin(tmp_path, knobs=True):
    b = tmp_path / "bin" / "llama-server"
    b.parent.mkdir(parents=True)
    b.write_bytes(b"x --slot-fork-min-tokens x --slot-fork-dft-full x LLAMA_KPF_MUTATE x dft_no_transfer x "
                  b"dft_partial_noop x" if knobs else b"x --slot-fork-min-tokens x")
    return b


def test_dry_run_plan_and_knob_refusal(tmp_path, capsys):
    ps = make_prompts(tmp_path / "ps.jsonl")
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(ps), "--numa-membind", "3", "--prior-191a", "", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and not (tmp_path / "o").exists()
    assert "swa_cells 8704" in out and "junctions [2148, 4096, 10240]" in out and "TOTAL est" in out
    for ln in ("honest", "flag_on", "mut_no_transfer", "mut_partial_noop"):
        assert f"--- launch {ln} " in out
    assert "numactl --membind=3 -- taskset -c 184-191" in out
    rc = m.main(["run", "--out", str(tmp_path / "o2"), "--cpu-list", "184-191",
                 "--fork-bin", str(fake_bin(tmp_path / "b2", knobs=False)), "--prompt-set", str(ps), "--prior-191a", "",
                 "--dry-run"])
    out = capsys.readouterr().out
    assert "knob b'dft_no_transfer' not compiled" in out and "knob b'--slot-fork-dft-full' not compiled" in out


def test_execute_refuses_without_deadline_claim_or_binary(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("KPF13_REGION", raising=False)
    ps = make_prompts(tmp_path / "ps.jsonl")
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--prompt-set", str(ps), "--prior-191a", ""])
    out = capsys.readouterr().out
    assert rc == 2 and "--fork-bin is required" in out and "--deadline-utc" in out and "region-lock claim" in out


def test_prior_191a_skips_its_launches_in_the_plan(tmp_path, capsys):
    (tmp_path / "rec.json").write_text(json.dumps({"kpf_task": "KPF-19.1a", "decisions": {"kpf19_1a": "PASS"}}))
    ps = make_prompts(tmp_path / "ps.jsonl")
    m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)),
            "--prompt-set", str(ps), "--prior-191a", str(tmp_path / "rec.json"), "--dry-run"])
    out = capsys.readouterr().out
    assert "KPF-19.1a PRIOR found" in out and "--- launch flag_on" not in out and "--- launch mut_partial_noop" in out


def test_prompt_set_for_another_model_is_refused(tmp_path):
    ps = m.Prompts(make_prompts(tmp_path / "ps.jsonl", model="/m/other.gguf"))
    assert any("tokenized for" in p for p in ps.problems([1], 14336, 64, MODEL))
    assert any("trunk" in p for p in m.Prompts(make_prompts(tmp_path / "s.jsonl", n=1000)).problems([1], 14336, 64, MODEL))


def test_launch_configs_are_registered_for_the_kpf13_server():
    for k, v in m.LAUNCHES.items():
        assert k13.CONFIGS[f"kpf174_{k}"] is v
    assert k13.CONFIGS["kpf174_mut_no_transfer"]["mutate"] == "dft_no_transfer"
    assert k13.CONFIGS["kpf174_flag_on"]["extra"] == ["--slot-fork-dft-full"]


def test_honest_runs_two_groups_and_the_other_launches_one():
    a = m.parse_args(["run", "--out", "/x"])
    assert m.groups_of(a, "honest") == [1, 2] and m.groups_of(a, "mut_partial_noop") == [1]
    assert a.launches == ["honest", "mut_partial_noop", "flag_on", "mut_no_transfer"]


# ---------------------------------------------------------------- mock execution: main() end to end on a fake server
def _fake_server_factory(clients, seen):
    class FakeServer:
        def __init__(self, ctx, cfg_name, binary):
            self.ctx, self.cfg_name, self.label = ctx, cfg_name, f"{cfg_name}-fake"
            self.argv, self.readback, self.env_note = [str(binary), "--fake"], {}, {}
            self.linkage, self.stop_rec, self.proc = {"pass": True}, {}, None

        def start(self):
            # rules and run metadata are on disk BEFORE the first launch
            assert (self.ctx.out / "rules.json").exists() and (self.ctx.out / "run_meta.json").exists()
            seen.append(self.cfg_name)
            self.ctx.client = clients[self.cfg_name]

        def stop(self):
            self.stop_rec = {"verified_dead": True}
    return FakeServer


def test_main_executes_every_launch_and_resumes_without_a_load(tmp_path, monkeypatch, capsys):
    clients = {"kpf174_honest": FakeClientWithSlots(), "kpf174_mut_partial_noop": FakeClientWithSlots(defect=[2148, 4096]),
               "kpf174_flag_on": FakeClientWithSlots(), "kpf174_mut_no_transfer": FakeClientWithSlots(defect=GEO["c"])}
    seen: list[str] = []
    monkeypatch.setattr(k13, "Server", _fake_server_factory(clients, seen))
    monkeypatch.setattr(k13, "port_listening", lambda p: False)
    monkeypatch.setenv("KPF13_REGION", "184-191")
    ps = make_prompts(tmp_path / "ps.jsonl")
    argv = ["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)),
            "--prompt-set", str(ps), "--prior-191a", "", "--deadline-s", "3600", "--no-device-claim", "test",
            "--card", str(tmp_path / "nocard"), "--numa-membind", "3"]
    assert m.main(argv) == 0
    assert seen == ["kpf174_honest", "kpf174_mut_partial_noop", "kpf174_flag_on", "kpf174_mut_no_transfer"]
    v = json.loads((tmp_path / "o" / "verdict.json").read_text())
    assert v["decisions"]["kpf17_4"] == "PASS" and v["decisions"]["kpf19_1a"] == "PASS"
    recs = list((tmp_path / "o" / "records").glob("*.json"))
    assert len(recs) == 4 and len((tmp_path / "o" / "belief_measurements.jsonl").read_text().splitlines()) == 4
    rules = json.loads((tmp_path / "o" / "rules.json").read_text())
    assert rules["rules_sha256"] == m.rules_sha256() and rules["cell_judge_rules_sha256_kpf13_v3"] == k13.rules_sha256("v3")
    assert len(list((tmp_path / "o" / "cells" / "honest").glob("*.json"))) == 6
    # a second window with --resume loads nothing: every cell is stored
    seen.clear()
    assert m.main(argv + ["--resume"]) == 0 and seen == []
    out = capsys.readouterr().out
    assert "all cells stored, skipped (resume)" in out
    # this row's own record now serves as the 19.1a prior for a later run (e.g. the KPF-18 S1 sweep)
    recs_191a = [json.loads(p.read_text()) for p in recs]
    assert any(r["decisions"]["kpf19_1a"] == "PASS" and "19.1a" in r["kpf_task"] for r in recs_191a)


def test_main_drains_on_the_stop_file_before_a_launch(tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(k13, "Server", _fake_server_factory({"kpf174_honest": FakeClientWithSlots()}, seen))
    monkeypatch.setattr(k13, "port_listening", lambda p: False)
    monkeypatch.setenv("KPF13_REGION", "184-191")
    stop = tmp_path / "STOP"
    stop.write_text("")
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(make_prompts(tmp_path / "ps.jsonl")), "--prior-191a", "", "--deadline-s", "3600",
                 "--no-device-claim", "test", "--card", str(tmp_path / "nocard"), "--stop-file", str(stop)])
    assert rc == 5 and seen == []
