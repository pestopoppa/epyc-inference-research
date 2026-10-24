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


# ---------------------------------------------------------------- KPF-GL2 (2026-10-23): the kpf174 record keeps the affinity read-back
def test_gl2_kpf174_record_keeps_the_affinity_readback(tmp_path):
    a, ctx = make_ctx(tmp_path)
    ps = m.Prompts(Path(a.prompt_set))
    rn = m.Runner(ctx, GEO, ps, "honest", {})

    class _Srv:
        argv, env_note, linkage, stop_rec = ["/x/llama-server"], {}, {"pass": True}, {"verified_dead": True}
        readback = {"exe_matches_binary": True, "cpus_allowed": list(range(184, 192)),
                    "cpus_allowed_non_exempt": list(range(184, 192)), "runtime_threads_unpinned": [4242],
                    "runtime_threads_unpinned_count": 1, "env_GGML_CUDA_DISABLE_GRAPHS": "1",
                    "env_HSA_OVERRIDE_CPU_AFFINITY_DEBUG": "0", "hip_graphs": "disabled", "problems": []}
    rn.srv = _Srv()
    rec = rn.finish_record({"launches": {}, "decisions": {}})
    ar = rec["argv_readback"]
    assert set(ar) == set(k13.READBACK_RECORD_KEYS)
    assert ar["runtime_threads_unpinned"] == [4242] and ar["runtime_threads_unpinned_count"] == 1
    assert ar["cpus_allowed_non_exempt"] == list(range(184, 192)) and ar["hip_graphs"] == "disabled"
    on_disk = json.loads(next((ctx.out / "records").glob("kpf174-honest-*.json")).read_text())
    assert on_disk["argv_readback"]["runtime_threads_unpinned_count"] == 1


# ---------------------------------------------------------------- KPF-17.4b (rules v2): junction sweep + speculation OFF at c1
ARGV_FILE = k13.HERE / "argv.gpu-27b-dflash2.txt"
SWEEP = [2049, 2148, 2560, 3072, 4096]


def test_174b_rules_v2_is_new_and_v1_is_unchanged():
    assert m.rules_sha256() == m.rules_sha256("v1") == "26b04e195e4a12e3d4dd7f247cd83b161270c0a660d1021f7f31511ebf9cd97c"
    assert m.rules_sha256("v2") != m.rules_sha256("v1") and m.RULES_V2["schema"] == "epyc.kpf174.rules.v2"
    for k in ("launches_v2", "spec_off_judge", "decisions_v2"):
        assert k in m.RULES_V2, k
    assert set(m.RULES_V2["decisions_v2"]) == {"c1_locus", "junction_onset", "sweep_2148", "not_decided_here"}
    json.dumps(m.RULES_V2)


def test_174b_geometry_override_keeps_L_and_marks_build_points():
    g = m.junctions(2048, 4, 512, SWEEP)
    assert g["c"] == SWEEP and g["build"] == SWEEP and g["L"] == 14336 and g["n_ubatch"] == 512
    assert all(g["pruned"].values()) and m.geometry_problems(g) == []
    c1 = m.junctions(2048, 4, 512, [2148], [2148, 4096, 10240])
    assert c1["c"] == [2148] and c1["build"] == [2148, 4096, 10240] and m.geometry_problems(c1) == []
    assert m.geometry_problems(m.junctions(2048, 4, 512, [2148], [4096]))          # a cell that is not a build point
    assert "build" not in GEO and "n_ubatch" not in GEO                            # v1 geometry is untouched


def test_174b_strip_drafter_removes_every_drafter_flag_and_keeps_the_target():
    argv, _ = k13.load_argv_file(ARGV_FILE)
    off, removed = k13.strip_drafter(argv)
    assert "-md" in removed and "--spec-type" in removed and "-ngld" in removed and "--device-draft" in removed
    assert "--spec-draft-n-max" in removed and k13.spec_off_problems(off) == []
    assert k13.flag_val(off, ("-m",)).endswith("Qwen3.8-27B-Q8_0.gguf") and k13.flag_val(off, ("--device",)) == "ROCm0"
    assert "--kv-unified" in off and "--no-cache-idle-slots" in off and k13.flag_val(off, ("-np",)) == "4"
    assert len(off) + len(removed) == len(argv)
    assert k13.spec_off_problems(argv)                                            # the full argv is NOT spec-off


def test_174b_spec_off_server_argv_has_no_drafter_and_no_debug_log(tmp_path):
    a, ctx = make_ctx(tmp_path)
    argv, _ = k13.load_argv_file(ARGV_FILE)
    ctx.gate_argv = argv
    a.draft_debug = True
    srv = k13.Server(ctx, "kpf174_honest_specoff", Path("/x/llama-server"))
    assert "-md" not in srv.argv and "--spec-type" not in srv.argv and "-lv" not in srv.argv
    assert k13.spec_off_problems(srv.argv) == [] and "-md" in srv.drafter_removed
    on = k13.Server(ctx, "kpf174_sweep", Path("/x/llama-server"))
    assert "-md" in on.argv and "-lv" in on.argv


class FakeClientSpecOff(FakeClientWithSlots):
    """No drafter: no draft counts. `tok_defect`: a forked child at these junctions takes another token at position 36
    (the c1 divergence shape); `ctl_jitter`: the k-th control of every cell differs at token 50 (device nondeterminism)."""

    def __init__(self, tok_defect=(), ctl_jitter=False, **kw):
        super().__init__(**kw)
        self.tok_defect, self.ctl_jitter, self.n_ctl = set(tok_defect), ctl_jitter, 0

    def complete(self, body, timeout):
        r = super().complete(body, timeout)
        t = r["timings"]
        t.pop("draft_n", None)
        t.pop("draft_n_accepted", None)
        if body["n_predict"] and t["n_fork_tokens"] in self.tok_defect:
            r["tokens"][36] = (r["tokens"][36] + 1) % 997
        if self.ctl_jitter and body["n_predict"] and body.get("slot_fork") is False and body["id_slot"] == 2:
            self.n_ctl += 1
            if self.n_ctl % 3 == 0:
                r["tokens"][50] = (r["tokens"][50] + 1) % 997
        return r


def run_174b(tmp_path, launch, client, geo, extra=()):
    a, ctx = make_ctx(tmp_path, ["--rules", "v2", *extra])
    ctx.client = client
    ps = m.Prompts(Path(a.prompt_set))
    res: dict = {}
    rn = m.Runner(ctx, geo, ps, launch, res)
    for g in a.groups:
        rn.group(g)
    return ctx, res


def test_174b_sweep_builds_every_junction_and_controls_replay_up_to_c(tmp_path):
    geo = m.junctions(2048, 4, 512, SWEEP)
    ctx, res = run_174b(tmp_path, "sweep", FakeClientWithSlots(), geo)
    assert sorted(res["sweep"]) == sorted(f"g{g}-c{c}" for g in (1, 2) for c in SWEEP)
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows" / "sweep.jsonl").read_text().splitlines()]
    src = [x["arm"] for x in rows if x["arm"].startswith("source") and x["group"] == 1]
    assert src == [f"source_{c}" for c in SWEEP] + ["source_14336"]
    seeds = [x["arm"] for x in rows if x.get("c") == 2148 and x["group"] == 1 and x["arm"].startswith("C1_seed")]
    assert seeds == ["C1_seed_2049", "C1_seed_2148"]
    for c in res["sweep"].values():
        assert [x["n_fork_tokens"] for x in c["arms"]] == [c["c"]] * 2 and c["build"][-1] == 14336
    ana = m.analyze(res, geo, None, "v2")
    assert ana["rules_sha256"] == m.rules_sha256("v2") and ana["launches"]["sweep"]["verdict"] == "PASS"
    assert ana["decisions"]["junction_onset"]["pattern"] == "none_fail" and ana["decisions"]["sweep_2148"] == "PASS"
    assert ana["decisions"]["c1_locus"] == "NOT_RUN"


def test_174b_sweep_onset_patterns_are_mechanical():
    sw = lambda fails: [(c, g, "FAIL" if c in fails else "PASS") for g in (1, 2) for c in SWEEP]
    d = m.decide_174b({"sweep": sw({2049, 2148})}, 512, SWEEP)
    assert d["junction_onset"]["pattern"] == "ubatch_misaligned_only" and d["sweep_2148"] == "FAIL"
    assert d["junction_onset"]["first_failing_junction"] == 2049
    assert d["junction_onset"]["per_group"]["1"] == {"fail": [2049, 2148], "pass": [2560, 3072, 4096], "other": []}
    d = m.decide_174b({"sweep": sw({2049})}, 512, SWEEP)
    assert d["junction_onset"]["pattern"] == "first_build_point_only"     # 2148 is misaligned too and passes
    d = m.decide_174b({"sweep": sw({2560})}, 512, [2560, 3072])
    assert d["junction_onset"]["pattern"] == "first_build_point_only"
    d = m.decide_174b({"sweep": sw({2148, 3072})}, 512, SWEEP)
    assert d["junction_onset"]["pattern"] == "other" and d["sweep_2148"] == "FAIL"


def test_174b_spec_off_c1_clean_is_drafter_on_only(tmp_path):
    geo = m.junctions(2048, 4, 512, [2148], [2148, 4096, 10240])
    ctx, res = run_174b(tmp_path, "honest_specoff", FakeClientSpecOff(), geo)
    assert sorted(res["honest_specoff"]) == ["g1-c2148", "g2-c2148"]
    c = res["honest_specoff"]["g1-c2148"]
    assert c["spec_off"] is True and all(x["draft_n"] is None for x in c["arms"] + c["controls"])
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows" / "honest_specoff.jsonl").read_text().splitlines()]
    assert [x["arm"] for x in rows if x["arm"].startswith("source") and x["group"] == 1] == \
        ["source_2148", "source_4096", "source_10240", "source_14336"]                 # the ORIGINAL c1 build
    ana = m.analyze(res, geo, None, "v2")
    assert ana["launches"]["honest_specoff"]["verdict"] == "PASS"
    assert ana["decisions"]["c1_locus"] == "drafter_on_only"


def test_174b_spec_off_target_divergence_is_a_target_restore_defect(tmp_path):
    geo = m.junctions(2048, 4, 512, [2148], [2148, 4096, 10240])
    ctx, res = run_174b(tmp_path, "honest_specoff", FakeClientSpecOff(tok_defect=[2148]), geo)
    ana = m.analyze(res, geo, None, "v2")
    cell = ana["cells"]["honest_specoff:g1-c2148"]
    assert cell["verdict"] == "FAIL" and cell["first_divergence"] == 36
    assert ana["decisions"]["c1_locus"] == "target_restore_defect"


def test_174b_spec_off_nondeterministic_controls_or_live_drafter_are_invalid(tmp_path):
    geo = m.junctions(2048, 4, 512, [2148], [2148, 4096, 10240])
    ctx, res = run_174b(tmp_path, "honest_specoff", FakeClientSpecOff(ctl_jitter=True), geo)
    ana = m.analyze(res, geo, None, "v2")
    assert ana["cells"]["honest_specoff:g1-c2148"]["verdict"] == "INVALID"
    assert ana["decisions"]["c1_locus"] == "INCONCLUSIVE"
    live = cell(c=2148, controls=[r(cache=2148) for _ in range(3)])                  # draft_n 30: drafter still on
    assert m.judge_cell(live, (0, 0), spec_off=True)["verdict"] == "INVALID"


def test_174b_dry_run_plans_and_refusals(tmp_path, capsys):
    ps = make_prompts(tmp_path / "ps.jsonl")
    base = ["run", "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)), "--prompt-set", str(ps),
            "--argv-file", str(ARGV_FILE), "--numa-membind", "3", "--dry-run"]
    rc = m.main(base + ["--out", str(tmp_path / "s"), "--rules", "v2", "--launches", "sweep",
                        "--junctions", ",".join(map(str, SWEEP))])
    out = capsys.readouterr().out
    assert rc == 0 and "rules=v2" in out and "junctions [2049, 2148, 2560, 3072, 4096]" in out
    assert "--- launch sweep " in out and "10 pending cells" in out and "PREFLIGHT: inputs PASS" in out
    rc = m.main(base + ["--out", str(tmp_path / "o"), "--rules", "v2", "--launches", "honest_specoff",
                        "--junctions", "2148", "--build-junctions", "2148,4096,10240"])
    out = capsys.readouterr().out
    assert rc == 0 and "speculation OFF, drafter flags removed" in out and "'-md'" in out
    assert "build points [2148, 4096, 10240] + L" in out and "2 pending cells" in out and "PREFLIGHT: inputs PASS" in out
    m.main(base + ["--out", str(tmp_path / "x"), "--rules", "v2", "--launches", "honest_specoff", "--junctions", "4096"])
    assert "honest_specoff is the c1 cell only" in capsys.readouterr().out
    m.main(base + ["--out", str(tmp_path / "y"), "--junctions", "2148"])
    assert "KPF-17.4b options: --rules v2" in capsys.readouterr().out
    m.main(base + ["--out", str(tmp_path / "z"), "--rules", "v2", "--launches", "honest,sweep"])
    assert "rules v2 (KPF-17.4b) runs only" in capsys.readouterr().out


def test_174b_main_end_to_end_spec_off_writes_v2_rules_and_locus(tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(k13, "Server", _fake_server_factory({"kpf174_honest_specoff": FakeClientSpecOff()}, seen))
    monkeypatch.setattr(k13, "port_listening", lambda p: False)
    monkeypatch.setenv("KPF13_REGION", "184-191")
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(make_prompts(tmp_path / "ps.jsonl")), "--argv-file", str(ARGV_FILE),
                 "--deadline-s", "3600", "--no-device-claim", "test", "--card", str(tmp_path / "nocard"),
                 "--rules", "v2", "--launches", "honest_specoff", "--junctions", "2148",
                 "--build-junctions", "2148,4096,10240"])
    assert rc == 0 and seen == ["kpf174_honest_specoff"]
    rules = json.loads((tmp_path / "o" / "rules.json").read_text())
    assert rules["rules_version"] == "v2" and rules["rules_sha256"] == m.rules_sha256("v2")
    v = json.loads((tmp_path / "o" / "verdict.json").read_text())
    assert v["decisions"]["c1_locus"] == "drafter_on_only" and v["rules"] == "v2"
    rec = json.loads(next((tmp_path / "o" / "records").glob("*.json")).read_text())
    assert rec["spec_off"] is True and rec["rules"] == "v2"


# ---------------------------------------------------------------- VB-KVU-PF.5 (2026-10-23): record_sha256 re-derives on read
def _adapter_digest(row: dict) -> str:
    """The belief adapter's re-derivation (root scripts/vidya/adapters/kv_prefix_fork.py `_digest` on the body)."""
    import hashlib
    body = {k: v for k, v in row.items() if k != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_vbkvupf5_record_with_int_keyed_pruned_rederives_after_a_json_round_trip(tmp_path):
    a, ctx = make_ctx(tmp_path)
    rn = m.Runner(ctx, GEO, m.Prompts(Path(a.prompt_set)), "honest", {})
    assert all(isinstance(k, int) for k in GEO["pruned"])                         # the trigger is present
    rec = rn.finish_record({"launches": {}, "decisions": {}})
    rt = json.loads(json.dumps(rec))
    assert k13.sha256_json({k: v for k, v in rt.items() if k != "record_sha256"}) == rec["record_sha256"]
    assert _adapter_digest(rt) == rec["record_sha256"]
    line = json.loads((ctx.out / "belief_measurements.jsonl").read_text().splitlines()[-1])
    assert _adapter_digest(line) == line["record_sha256"]
    on_disk = json.loads(next((ctx.out / "records").glob("kpf174-honest-*.json")).read_text())
    assert _adapter_digest(on_disk) == on_disk["record_sha256"]


def test_vbkvupf5_old_style_hash_does_not_rederive_and_seal_fixes_it():
    rec = {"run_id": "x", "geometry": {"pruned": {2148: True, 4096: True, 10240: False}}}
    old = dict(rec, record_sha256=k13.sha256_json(rec))                          # the pre-fix write side
    assert _adapter_digest(json.loads(json.dumps(old))) != old["record_sha256"]
    sealed = k13.seal_record(old)
    assert _adapter_digest(sealed) == sealed["record_sha256"] and set(sealed["geometry"]["pruned"]) == {"2148", "4096", "10240"}


def test_vbkvupf5_reseal_script_writes_a_new_file_and_keeps_the_original(tmp_path):
    spec = importlib.util.spec_from_file_location("reseal", Path(__file__).parent / "kpf174" / "reseal_records.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    bad = {"run_id": "kpf174-honest-1", "geometry": {"pruned": {2148: True, 10240: False}}, "kpf_task": "KPF-17.4"}
    bad["record_sha256"] = k13.sha256_json(bad)
    good = k13.seal_record({"run_id": "kpf174-flag_on-1", "geometry": {"pruned": {"2148": True}}})
    src = tmp_path / "run" / "belief_measurements.jsonl"
    src.parent.mkdir()
    src.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n")
    before = src.read_bytes()
    assert rs.main([str(src)]) == 0
    assert src.read_bytes() == before                                              # the original is kept
    dst = src.with_name("belief_measurements.resealed-vbkvupf5.jsonl")
    rows = [json.loads(x) for x in dst.read_text().splitlines()]
    assert len(rows) == 2 and all(_adapter_digest(r) == r["record_sha256"] for r in rows)
    assert rows[1] == good
    man = json.loads(dst.with_name(dst.name + ".manifest.json").read_text())
    assert [x["action"] for x in man["rows"]] == ["resealed", "unchanged (already re-derives)"]
    assert man["rows"][0]["old_sha_explained_by_int_keyed_pruned"] is True
    assert rs.main([str(src)]) == 2                                                # never overwrites its own output
    tampered = dict(json.loads(json.dumps(bad)), kpf_task="edited")                 # unexplained mismatch: refused
    src2 = tmp_path / "t.jsonl"
    src2.write_text(json.dumps(tampered) + "\n")
    assert rs.main([str(src2)]) == 2 and not src2.with_name("t.resealed-vbkvupf5.jsonl").exists()


# ---------------------------------------------------------------- KPF-19.1a / 17.4 CPU leg (2026-10-24, workspace-f8)
CPU_ARGV_FILE = k13.HERE / "argv.cpu-27b-dflash2.txt"


def test_cpu_argv_is_the_gpu_argv_on_cpu():
    gpu, _ = k13.load_argv_file(ARGV_FILE)
    cpu, _ = k13.load_argv_file(CPU_ARGV_FILE)
    assert k13.argv_problems(cpu, "dflash2", "cpu") == []
    assert k13.argv_problems(gpu, "dflash2", "cpu")          # the GPU argv carries -t: refused on CPU
    exp = list(gpu)
    for f in ("-t", "-ngl", "-ngld"):
        i = exp.index(f)
        del exp[i:i + 2]
    exp = k13.set_flag(exp, ("--device",), "none")
    exp = k13.set_flag(exp, ("--device-draft",), "none")
    assert cpu == exp                                          # same flags, same order: only the listed changes
    for f in ("-m", "-md", "-np", "-c", "-b", "-ub", "-ctk", "-ctv", "--cache-ram", "--spec-type", "--spec-draft-n-max",
              "--chat-template-file"):
        assert k13.flag_val(cpu, (f,)) == k13.flag_val(gpu, (f,)), f
    assert "ROCm0" not in cpu and "-ngl" not in cpu and "-ngld" not in cpu


def test_cpu_server_argv_sets_threads_from_the_claim_and_spec_off_drops_the_draft_device(tmp_path):
    a, ctx = make_ctx(tmp_path, ["--device", "cpu", "--threads", "24"])
    ctx.gate_argv, _ = k13.load_argv_file(CPU_ARGV_FILE)
    assert ctx.prefix == ["taskset", "-c", "184-191"]          # no numactl on CPU unless --numa-membind is given
    on = k13.Server(ctx, "kpf174_honest", Path("/x/llama-server"))
    assert k13.flag_val(on.argv, ("-t",)) == "24" and k13.flag_val(on.argv, ("-tb",)) == "24"
    assert k13.flag_val(on.argv, ("--device",)) == "none" and "-md" in on.argv
    off = k13.Server(ctx, "kpf174_honest_specoff", Path("/x/llama-server"))
    assert "-md" not in off.argv and "--device-draft" not in off.argv and k13.spec_off_problems(off.argv) == []
    assert k13.flag_val(off.argv, ("--device",)) == "none"


def test_cpu_dry_run_plans_the_191a_leg_and_the_c1_spec_off_cell(tmp_path, capsys):
    ps = make_prompts(tmp_path / "ps.jsonl")
    base = ["run", "--device", "cpu", "--cpu-list", "48-71,144-167", "--threads", "24", "--fork-bin",
            str(fake_bin(tmp_path)), "--prompt-set", str(ps), "--argv-file", str(CPU_ARGV_FILE), "--prior-191a", "",
            "--prefill-tok-s", "27", "--decode-tok-s", "6", "--load-s", "120", "--dry-run"]
    rc = m.main(base + ["--out", str(tmp_path / "v1")])
    out = capsys.readouterr().out
    assert rc == 0 and not (tmp_path / "v1").exists()
    assert "device=cpu" in out and "PREFLIGHT: inputs PASS" in out and "rules=v1" in out
    assert "launch prefix: taskset -c 48-71,144-167" in out and "numactl" not in out
    for ln in ("honest", "flag_on", "mut_no_transfer", "mut_partial_noop"):
        assert f"--- launch {ln} " in out
    rc = m.main(base + ["--out", str(tmp_path / "v2"), "--rules", "v2", "--launches", "honest_specoff",
                        "--junctions", "2148", "--build-junctions", "2148,4096,10240"])
    out = capsys.readouterr().out
    assert rc == 0 and "PREFLIGHT: inputs PASS" in out and "'-md'" in out and "'--device-draft'" in out
    assert "2 pending cells" in out
