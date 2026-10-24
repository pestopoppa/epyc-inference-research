"""Server-free tests for scripts/benchmark/kpf174/kpf174c_bisect.py (KPF-17.4c target-restore bisect, speculation OFF).

No llama-server, no model, no GPU. The protocol runs against LayoutFake, an in-process stand-in that models what the
bisect's matched-layout argument relies on (82c16126f code reading):
  - the unified KV cell allocator of llama_kv_cache: contiguous find_slot from `head` (reset to 0 when head > used + 2n),
    seq_rm sets head to the lowest freed cell, seq_cp_attn adds the dst seq bit to the source's cells, n_kv = pad256(
    used_max_p1);
  - the server's prompt chunking (n_batch 512, breaks n - 512 and n - 4 before the end), checkpoint_at [-1] pinned
    prompt-end checkpoints, the fork (kind end / checkpoint, largest p), the own-slot checkpoint restore of pre_decode;
  - LAYOUT-SENSITIVE numerics: every computed chunk carries a hash of (n_kv, the visible cell indices, the visible cells'
    own numerics, the recurrent numerics). Tokens depend on content only; logprobs on content + numerics.
So "no defect" here means exactly the hypothesis the bisect tests: GPU results depend on layout but every state copy is
exact. Defects: `recurrent` (a state copy perturbs the recurrent numerics), `kv` (reading a shared cell perturbs),
`position` (the fork leaves the recurrent pos one short: the next chunk logs 'non-consecutive token position').
Run: /usr/bin/python3 -m pytest scripts/benchmark/test_kpf174c_bisect.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("kpf174c_bisect", Path(__file__).parent / "kpf174" / "kpf174c_bisect.py")
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)
k13 = m.k13
N = m.N_GEN
MODEL = "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf"
INF = 1 << 40
# small geometry for speed (the allocator logic is size-independent)
T_JUNCTIONS, T_BUILD, T_L, T_S, T_FORK_MIN = [600, 1024], [600, 1024, 1536], 2048, 16, 128
GPU_ARGV = Path(m.DEFAULT_ARGV)


def h(obj) -> int:
    return int(hashlib.sha256(repr(obj).encode()).hexdigest()[:12], 16)


class LayoutFake:
    PAD = 256

    def __init__(self, size=4096, fork_min=T_FORK_MIN, defect=(), layout_sensitive=True):
        self.cells: list[dict | None] = [None] * size
        self.head = 0
        self.slots_ = {s: {"tokens": [], "rec": None, "ckpts": []} for s in range(4)}
        self.fork_min, self.defect, self.layout_sensitive = fork_min, set(defect), layout_sensitive
        self.lines: list[tuple[float, str]] = []
        self.reqs: list[dict] = []

    # -- allocator (llama_kv_cache)
    def used(self) -> int:
        return sum(1 for c in self.cells if c is not None)

    def used_max_p1(self) -> int:
        for i in range(len(self.cells) - 1, -1, -1):
            if self.cells[i] is not None:
                return i + 1
        return 0

    def seq_rm(self, seq: int, p0: int, p1: int) -> None:
        new_head = None
        for i, c in enumerate(self.cells):
            if c is not None and seq in c["seqs"] and p0 <= c["pos"] < p1:
                c["seqs"].discard(seq)
                if not c["seqs"]:
                    self.cells[i] = None
                    new_head = i if new_head is None else new_head
        if new_head is not None and new_head < self.head:
            self.head = new_head

    def find_slot(self, n: int) -> int:
        head = self.head
        if head > self.used() + 2 * n:
            head = 0
        i = head
        while True:
            if i + n > len(self.cells):
                i = 0
                continue
            if all(self.cells[j] is None for j in range(i, i + n)):
                return i
            i += 1

    def compute(self, slot: int, toks: list[int], pos0: int) -> None:
        st = self.slots_[slot]
        if st["rec"] is not None and pos0 != st["rec"]["pos"] + 1:
            self.lines.append((time.time(), f"find_slot: non-consecutive token position {pos0 + len(toks) - 1} after "
                                            f"{st['rec']['pos']} for sequence {slot}"))
        idx = self.find_slot(len(toks))
        for j, t in enumerate(toks):
            self.cells[idx + j] = {"pos": pos0 + j, "seqs": {slot}, "tok": t, "num": None}
        self.head = idx + len(toks)
        nkv = -(-self.used_max_p1() // self.PAD) * self.PAD
        vis = [i for i in range(nkv) if self.cells[i] is not None and slot in self.cells[i]["seqs"]]
        shared = "kv" in self.defect and any(len(self.cells[i]["seqs"]) > 1 for i in vis)
        prev = tuple(self.cells[i]["num"] for i in vis if self.cells[i]["num"] is not None)
        rec_num = st["rec"]["num"] if st["rec"] else 0
        key = (nkv, tuple(vis)) if self.layout_sensitive else (len(vis),)
        num = h(("chunk", rec_num, key, prev, shared))
        for j in range(len(toks)):
            self.cells[idx + j]["num"] = num
        sem = h(("sem", st["rec"]["sem"] if st["rec"] else 0, tuple(toks)))
        st["rec"] = {"pos": pos0 + len(toks) - 1, "sem": sem, "num": num}

    def copy_state(self, rec: dict) -> dict:
        r = dict(rec)
        if "recurrent" in self.defect:
            r["num"] = h(("lossy", r["num"]))
        return r

    @staticmethod
    def chunks(start: int, end: int) -> list[tuple[int, int]]:
        out, cur = [], start
        while cur < end:
            stop = min(end, cur + 512)
            for b in (end - 512, end - 4):
                if cur < b < stop:
                    stop = b
            out.append((cur, stop))
            cur = stop
        return out

    @staticmethod
    def sample(rec: dict) -> tuple[int, dict]:
        s = rec["sem"]
        tok = s % 1000
        lp = -((s >> 10) % 997) / 1000.0 - 1e-7 * (rec["num"] % 1000)
        return tok, {"id": tok, "logprob": lp, "top_logprobs": [{"id": tok, "logprob": lp},
                                                                {"id": tok + 1, "logprob": lp - 0.5}]}

    # -- server API (k13.Client shape)
    def erase(self, slot: int) -> None:
        self.seq_rm(slot, 0, INF)
        self.slots_[slot] = {"tokens": [], "rec": None, "ckpts": []}

    def slots(self) -> list[dict]:
        return [{"id": s, "checkpoints": [{k: ck[k] for k in ("n_tokens", "pos_min", "pos_max", "pinned")} | {"size_mib": 1.0}
                                          for ck in o["ckpts"]]} for s, o in self.slots_.items()]

    def complete(self, body: dict, timeout: float) -> dict:
        t0 = time.time()
        ids, slot, npred = list(body["prompt"]), body["id_slot"], body["n_predict"]
        st = self.slots_[slot]
        lcp = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))  # noqa: E731
        own = lcp(st["tokens"], ids) if body["cache_prompt"] else 0
        nfork, src, kind = 0, None, None
        if body.get("slot_fork", True) and body["cache_prompt"] and len(ids) > self.fork_min:
            best = None
            for s, o in self.slots_.items():
                if s == slot or not o["tokens"]:
                    continue
                k = lcp(o["tokens"], ids)
                if k < self.fork_min:
                    continue
                cands = []
                if len(o["tokens"]) == k and k < len(ids) and o["rec"] and o["rec"]["pos"] == k - 1:
                    cands.append((k, "end", None))
                for ck in o["ckpts"]:
                    p = ck["n_tokens"]
                    if 0 < p <= k and p < len(ids) and p >= self.fork_min and ck["pos_max"] + 1 == p:
                        cands.append((p, "checkpoint", ck))
                for cd in cands:
                    if best is None or cd[0] > best[0]:
                        best = (*cd, s)
            if best and best[0] > own:
                p, kind, ck, s = best
                self.seq_rm(slot, 0, INF)
                for c in self.cells:
                    if c is not None and s in c["seqs"] and c["pos"] < p:
                        c["seqs"].add(slot)
                st["rec"] = self.copy_state(ck["state"] if ck else self.slots_[s]["rec"])
                if "position" in self.defect:
                    st["rec"]["pos"] -= 1
                st["tokens"] = ids[:p]
                st["ckpts"] = [dict(x) for x in self.slots_[s]["ckpts"] if x["n_tokens"] <= p]
                nfork, src = p, s
        n_past = lcp(st["tokens"], ids) if body["cache_prompt"] else 0
        if n_past == len(ids):
            n_past -= 1
        if n_past > 0 and st["rec"] is not None and st["rec"]["pos"] >= n_past:
            cks = [x for x in st["ckpts"] if x["pos_max"] <= n_past and x["pos_min"] < n_past]
            if cks:
                n_past = min(n_past, cks[-1]["n_tokens"])
                st["rec"] = self.copy_state(cks[-1]["state"])
            else:
                n_past = 0
        st["ckpts"] = [x for x in st["ckpts"] if x["pos_max"] <= n_past]
        self.seq_rm(slot, n_past, INF)
        st["tokens"] = ids[:n_past]
        if n_past == 0:
            st["rec"] = None
        for a, b in self.chunks(n_past, len(ids)):
            self.compute(slot, ids[a:b], a)
            st["tokens"] = ids[:b]
        if body.get("checkpoint_at") == [-1]:
            st["ckpts"].append({"n_tokens": len(ids), "pos_min": len(ids) - 1, "pos_max": len(ids) - 1, "pinned": True,
                                "state": dict(st["rec"])})
        toks, probs = [], []
        for i in range(npred):
            t, top = self.sample(st["rec"])
            toks.append(t)
            probs.append(top)
            if i < npred - 1:
                self.compute(slot, [t], len(st["tokens"]))
                st["tokens"].append(t)
        self.reqs.append({"slot": slot, "n": len(ids), "n_fork": nfork, "kind": kind, "cache_n": n_past})
        tim = {"cache_n": n_past, "prompt_n": len(ids) - n_past, "n_fork_tokens": nfork, "fork_src_slot": src,
               "fork_src_kind": kind, "prompt_ms": 1.0, "predicted_ms": 1.0}
        return {"http": 200, "wall_s": 0.0, "t_send": t0, "t_end": time.time(), "tokens": toks, "probs": probs,
                "timings": tim}


def make_prompts(path: Path, groups=(1, 2), n=T_L + 100, model=MODEL) -> Path:
    lines = [{"kind": "meta", "model": model}]
    for g in groups:
        lines.append({"kind": "trunk", "group": g, "ids": [10_000 * g + i for i in range(n)]})
        lines.append({"kind": "pair", "group": g, "suffix_index": 0, "suffix_ids": [900_000 + 1000 * g + j for j in range(80)]})
        lines.append({"kind": "filler", "group": g, "ids": [500_000 + 1000 * g + j for j in range(n)]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def small_args(tmp_path, extra=()):
    return ["run", "--out", str(tmp_path / "out"), "--cpu-list", "184-191", "--fork-bin", "/x/llama-server",
            "--prompt-set", str(make_prompts(tmp_path / "ps.jsonl")), "--junctions", ",".join(map(str, T_JUNCTIONS)),
            "--build-junctions", ",".join(map(str, T_BUILD)), "--L", str(T_L), "--suffix-tokens", str(T_S),
            "--fork-min", str(T_FORK_MIN), *extra]


def run_cells(tmp_path, client, monkeypatch):
    a = m.parse_args(small_args(tmp_path))
    a.window_id = "t"
    ctx = k13.Ctx(a, None, ["-np", "4", "--kv-unified"], k13.Deadline(None))
    ctx.start_utc = k13.now_iso()
    ctx.client = client
    geo = m.geometry(a.junctions, a.build_junctions, a.L)
    ps = m.Prompts(Path(a.prompt_set))
    results: dict = {}
    rn = m.Runner(ctx, geo, ps, results)
    monkeypatch.setattr(m.Runner, "vram_sample", lambda self: 1 << 30)
    rn.srv = type("S", (), {"lines_between": lambda self, t0, t1: [ln for t, ln in client.lines if t0 - 0.5 <= t <= t1 + 0.5]})()
    cells = m.planned_cells(a.groups, geo["c"])
    for g, c in cells:
        rn.cell(g, c)
    ana = m.analyze(results, [m.cell_id(g, c) for g, c in cells])
    return results, ana, rn


# ---------------------------------------------------------------- rules + geometry
def test_rules_are_declared_hashed_and_name_the_decision_words():
    assert m.RULES["schema"] == "epyc.kpf174c.rules.v1"
    assert m.rules_sha256() == k13.sha256_json(m.RULES) and len(m.rules_sha256()) == 64
    for w in ("position", "recurrent", "kv", "none_reproduced", "undetermined"):
        assert w in m.RULES["decisions"]["locus"]
    for w in ("control_layout_numerics", "not_reproduced", "not_explained"):
        assert w in m.RULES["decisions"]["explains_174b"]
    assert set(m.RULES["arms"]) == set(m.ARMS) and set(m.MATCHED) == {"Xc~Cmask", "Xe~C0", "Ks~C0"}
    assert m.RULES["repeats"] == {"Xc": 2, "Cf": 3, "Ks": 2, "Cmask": 3, "C0": 3, "Xe": 2}


def test_default_geometry_is_the_17_4b_geometry_and_cache_ram_safe():
    geo = m.geometry(list(m.DEFAULT_JUNCTIONS), list(m.DEFAULT_BUILD), m.DEFAULT_L)
    assert geo["c"] == [2148, 4096] and geo["build"] == [2148, 4096, 10240] and geo["L"] == 14336
    assert geo["ks_ext"] == {2148: 3222, 4096: 6144}
    assert m.geometry_problems(geo, 64, 1024) == []
    assert m.planned_cells([1, 2], geo["c"]) == [(1, 2148), (2, 2148), (1, 4096), (2, 4096)]
    bad = m.geometry([3000], [2148, 4096], 14336)
    assert any("not a build point" in x for x in m.geometry_problems(bad, 64, 1024))


def test_the_spec_off_config_is_registered_and_strips_the_drafter():
    cfg = k13.CONFIGS[m.CONFIG_NAME]
    assert cfg["spec_off"] is True and cfg["mutate"] is None and cfg["extra"] == []
    argv, _ = k13.load_argv_file(GPU_ARGV)
    off, removed = k13.strip_drafter(argv)
    assert "-md" in removed and not k13.spec_off_problems(off)


# ---------------------------------------------------------------- judging (pure)
def run_(tokens=None, lps=None, role_nf=0, cache_n=600, prompt_n=T_S, src=None, nonconsec=0):
    tokens = list(range(N)) if tokens is None else tokens
    lps = [-0.1] * N if lps is None else lps
    return {"http": 200, "tokens": tokens, "top2": [[[t, lp], [t + 1, lp - 0.5]] for t, lp in zip(tokens, lps)],
            "top2_kind": "logprob", "draft_n": None, "n_fork_tokens": role_nf, "fork_src_slot": src, "cache_n": cache_n,
            "prompt_n": prompt_n, "nonconsec_lines": nonconsec}


def full_cell(c=600, over=None):
    runs = {"Xc": [run_(role_nf=c, src=0, cache_n=c) for _ in range(2)], "Cf": [run_(cache_n=c) for _ in range(3)],
            "Ks": [run_(cache_n=c) for _ in range(2)], "Cmask": [run_(cache_n=c) for _ in range(3)],
            "C0": [run_(cache_n=c) for _ in range(3)], "Xe": [run_(role_nf=c, src=0, cache_n=c) for _ in range(2)]}
    runs.update(over or {})
    ck = [{"n_tokens": c, "pos_min": c - 1, "pos_max": c - 1, "pinned": True}]
    return {"cell": f"g1-c{c}", "group": 1, "c": c, "S": T_S, "device": "gpu", "runs": runs,
            "source_checkpoints": {p: ck for p in ("S", "K1", "K2", "E")}, "vram_own_pid_bytes": [5, 6]}


def test_compare_bit_identity_and_logprob_only_drift():
    a, b = run_(), run_()
    r = m.compare(a, b)
    assert r["identical"] and r["first_token_divergence"] is None and r["first_logprob_divergence"] is None
    lps = [-0.1] * N
    lps[3] = -0.1 + 1e-6
    r = m.compare(run_(lps=lps), a)
    assert not r["identical"] and r["first_token_divergence"] is None and r["first_logprob_divergence"] == 3
    assert r["max_abs_dlogprob_prefix"] == pytest.approx(1e-6) and r["abs_dlogprob_token0"] == 0.0
    toks = list(range(N))
    toks[10] = 999
    r = m.compare(run_(tokens=toks), a)
    assert r["first_token_divergence"] == 10 and r["ref_margin_at_divergence"] == pytest.approx(0.5)


def test_validity_partition_fork_not_taken_and_position_evidence():
    assert m.run_problems(run_(cache_n=599), "ref", 600, T_S)
    assert any("not taken" in x for x in m.run_problems(run_(role_nf=0), "fork", 600, T_S))
    assert not m.run_problems(run_(role_nf=512, src=0, cache_n=512), "fork", 600, T_S)
    assert m.position_evidence(run_(role_nf=512, src=0, cache_n=512), "fork", 600, T_S)
    assert m.position_evidence(run_(cache_n=600, nonconsec=1), "restore", 600, T_S)
    assert any("non-consecutive" in x for x in m.run_problems(run_(nonconsec=2), "ref", 600, T_S))
    assert m.run_problems({**run_(), "draft_n": 7}, "ref", 600, T_S)


def test_unstable_or_short_groups_are_never_judged():
    lps = [-0.1] * N
    lps[0] = -0.2
    j = m.judge_cell(full_cell(over={"C0": [run_(), run_(), run_(lps=lps)]}))
    assert not j["comparisons"]["Ks~C0"]["judged"] and "C0 unstable" in j["comparisons"]["Ks~C0"]["why"][0]
    j = m.judge_cell(full_cell(over={"Cmask": [run_(), run_()]}))
    assert not j["comparisons"]["Xc~Cmask"]["judged"] and not j["valid"]
    c = full_cell()
    c["vram_own_pid_bytes"] = [5]
    assert not m.judge_cell(c)["valid"] and not m.judge_cell(c)["comparisons"]["Ks~C0"]["judged"]
    c = full_cell()
    c["source_checkpoints"]["K2"] = []
    assert any("phase K2" in x for x in m.judge_cell(c)["cell_problems"])


def decide_for(over_by_cell: dict) -> dict:
    cells = {cid: m.judge_cell({**full_cell(over=ov), "cell": cid}) for cid, ov in over_by_cell.items()}
    return m.decide(cells, list(over_by_cell))


def test_decide_matrix():
    lps = [-0.1] * N
    lps[0] = -0.1 - 1e-5
    drift = [run_(lps=lps, cache_n=600) for _ in range(3)]
    fdrift = [run_(lps=lps, role_nf=600, src=0, cache_n=600) for _ in range(2)]
    # every matched pair identical, the 17.4b control differs -> the 17.4b divergence is the control's layout
    d = decide_for({"a": {"Cf": drift}, "b": {}})
    assert d["locus"] == "none_reproduced" and d["decision_grade"] and d["explains_174b"] == "control_layout_numerics"
    d = decide_for({"a": {}, "b": {}})
    assert d["locus"] == "none_reproduced" and d["explains_174b"] == "not_reproduced"
    # the restore alone differs at matched layout -> recurrent
    d = decide_for({"a": {"Ks": drift[:2]}, "b": {}})
    assert d["locus"] == "recurrent" and d["matched"]["Ks~C0"]["differs"] == ["a"]
    # restore clean, fork differs -> kv (attention-cell share)
    d = decide_for({"a": {"Xe": fdrift}, "b": {}})
    assert d["locus"] == "kv" and d["fork_differs_where_restore_identical"] == ["a"]
    d = decide_for({"a": {"Xc": fdrift, "Cf": drift}, "b": {}})
    assert d["locus"] == "kv" and d["explains_174b"] == "not_explained"
    # position evidence wins
    d = decide_for({"a": {"Xe": [run_(role_nf=600, src=0, cache_n=600, nonconsec=1)] * 2}, "b": {}})
    assert d["locus"] == "position"
    # a missing cell: no universal claim
    cells = {"a": m.judge_cell({**full_cell(), "cell": "a"})}
    d = m.decide(cells, ["a", "b"])
    assert d["locus"] == "undetermined" and not d["decision_grade"] and not d["complete"]
    # but an existence claim stands on a partial run
    cells = {"a": m.judge_cell({**full_cell(over={"Ks": drift[:2]}), "cell": "a"})}
    assert m.decide(cells, ["a", "b"])["locus"] == "recurrent"


# ---------------------------------------------------------------- the protocol against the allocator-faithful fake
def test_protocol_places_every_matched_pair_on_the_same_layout_and_finds_the_17_4b_divergence(tmp_path, monkeypatch):
    res, ana, rn = run_cells(tmp_path, LayoutFake(), monkeypatch)
    assert sorted(res) == ["g1-c1024", "g1-c600", "g2-c1024", "g2-c600"]
    d = ana["decisions"]
    assert d["locus"] == "none_reproduced" and d["decision_grade"] and d["explains_174b"] == "control_layout_numerics"
    for cid, j in ana["cells"].items():
        assert j["valid"], j["cell_problems"]
        assert all(g["stable"] for g in j["groups"].values())
        for k in m.MATCHED:
            assert j["comparisons"][k]["identical"], (cid, k)
        # the 17.4b control was built on another layout: differs in logprob bits from token 0, never in tokens here
        assert not j["comparisons"]["Xc~Cf"]["identical"] and j["comparisons"]["Xc~Cf"]["first_logprob_divergence"] == 0
        assert not j["comparisons"]["Cmask~C0"]["identical"]       # masked occupancy alone moves the numerics
    for c in res.values():
        assert {r["fork_src_kind"] for r in c["runs"]["Xc"]} == {"checkpoint"}
        assert {r["fork_src_kind"] for r in c["runs"]["Xe"]} == {"end"}
        assert all(r["cache_n"] == c["c"] and r["prompt_n"] == T_S for v in c["runs"].values() for r in v)


def test_protocol_detects_a_lossy_recurrent_copy(tmp_path, monkeypatch):
    _, ana, _ = run_cells(tmp_path, LayoutFake(defect={"recurrent"}), monkeypatch)
    d = ana["decisions"]
    assert d["locus"] == "recurrent" and len(d["matched"]["Ks~C0"]["differs"]) == 4
    assert d["explains_174b"] == "not_explained"


def test_protocol_detects_a_shared_cell_defect_as_kv(tmp_path, monkeypatch):
    _, ana, _ = run_cells(tmp_path, LayoutFake(defect={"kv"}), monkeypatch)
    d = ana["decisions"]
    assert d["locus"] == "kv" and not d["matched"]["Ks~C0"]["differs"]
    assert len(d["matched"]["Xe~C0"]["differs"]) == 4 and len(d["matched"]["Xc~Cmask"]["differs"]) == 4


def test_protocol_detects_position_bookkeeping_from_the_server_log(tmp_path, monkeypatch):
    _, ana, _ = run_cells(tmp_path, LayoutFake(defect={"position"}), monkeypatch)
    d = ana["decisions"]
    assert d["locus"] == "position" and d["position_evidence"]
    assert all("non-consecutive" in x for ev in d["position_evidence"].values() for x in ev)


def test_layout_blind_numerics_reproduce_nothing(tmp_path, monkeypatch):
    _, ana, _ = run_cells(tmp_path, LayoutFake(layout_sensitive=False), monkeypatch)
    d = ana["decisions"]
    assert d["locus"] == "none_reproduced" and d["explains_174b"] == "not_reproduced"


def test_cache_ram_guard_refuses_a_request_that_would_go_through_the_ram_cache(tmp_path, monkeypatch):
    a = m.parse_args(small_args(tmp_path))
    ctx = k13.Ctx(a, None, [], k13.Deadline(None))
    rn = m.Runner(ctx, m.geometry(a.junctions, a.build_junctions, a.L), None, {})
    rn.shadow[0] = list(range(2000))
    with pytest.raises(k13.Abort, match="f_keep"):
        rn.guard_cache_ram(0, list(range(600)) + [7] * 16)
    rn.guard_cache_ram(0, list(range(1200)) + [7] * 16)   # f_keep 0.6: fine
    rn.shadow[0] = []
    rn.guard_cache_ram(0, [1, 2, 3])


# ---------------------------------------------------------------- CLI: dry run, refusals, end-to-end with a fake server
def fake_bin(tmp_path, knobs=True):
    b = tmp_path / "bin" / "llama-server"
    b.parent.mkdir(parents=True)
    b.write_bytes(b"x --slot-fork-min-tokens x" if knobs else b"x")
    return b


def test_dry_run_plans_without_starting_anything(tmp_path, capsys):
    ps = make_prompts(tmp_path / "ps.jsonl", n=m.DEFAULT_L + 10)
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--fork-bin", str(fake_bin(tmp_path)),
                 "--prompt-set", str(ps), "--numa-membind", "3", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and not (tmp_path / "o").exists()
    assert "['g1-c2148', 'g2-c2148', 'g1-c4096', 'g2-c4096']" in out and "Ks extensions {2148: 3222, 4096: 6144}" in out
    assert "speculation OFF, drafter flags removed" in out and "TOTAL est" in out and "PREFLIGHT: inputs PASS" in out
    assert "numactl --membind=3 -- taskset -c 184-191" in out
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191",
                 "--fork-bin", str(fake_bin(tmp_path / "b2", knobs=False)), "--prompt-set", str(ps), "--dry-run"])
    assert "knob b'--slot-fork-min-tokens' not compiled" in capsys.readouterr().out


def test_execute_refuses_without_deadline_claim_or_binary(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("KPF13_REGION", raising=False)
    ps = make_prompts(tmp_path / "ps.jsonl", n=m.DEFAULT_L + 10)
    rc = m.main(["run", "--out", str(tmp_path / "o"), "--cpu-list", "184-191", "--prompt-set", str(ps)])
    out = capsys.readouterr().out
    assert rc == 2 and "--fork-bin is required" in out and "--deadline-utc" in out and "region-lock claim" in out


def test_short_filler_or_trunk_is_refused(tmp_path):
    ps = m.Prompts(make_prompts(tmp_path / "ps.jsonl", n=5000))
    assert any("trunk" in p for p in ps.problems_174c([1], 14336, 64, [2148], MODEL))
    assert any("filler" in p for p in ps.problems_174c([1], 14336, 64, [2148], MODEL))


def _fake_server_factory(client, seen):
    class FakeServer:
        def __init__(self, ctx, cfg_name, binary):
            self.ctx, self.cfg_name, self.label = ctx, cfg_name, f"{cfg_name}-fake"
            self.argv, self.readback, self.env_note = [str(binary), "--fake"], {}, {}
            self.linkage, self.stop_rec, self.proc, self.drafter_removed = {"pass": True}, {}, None, ["-md", "x"]

        def start(self):
            assert (self.ctx.out / "rules.json").exists() and (self.ctx.out / "run_meta.json").exists()
            seen.append(self.cfg_name)
            self.ctx.client = client

        def lines_between(self, t0, t1):
            return [ln for t, ln in client.lines if t0 - 0.5 <= t <= t1 + 0.5]

        def log_lines(self):
            return ["print_info: n_swa            = 0"]

        def stop(self):
            self.stop_rec = {"verified_dead": True}
    return FakeServer


def test_main_end_to_end_writes_rules_first_then_cells_verdict_and_one_record(tmp_path, monkeypatch, capsys):
    seen: list[str] = []
    monkeypatch.setattr(k13, "Server", _fake_server_factory(LayoutFake(), seen))
    monkeypatch.setattr(k13, "port_listening", lambda p: False)
    monkeypatch.setattr(m.Runner, "vram_sample", lambda self: 1 << 30)
    monkeypatch.setenv("KPF13_REGION", "184-191")
    argv = small_args(tmp_path, ["--fork-bin", str(fake_bin(tmp_path)), "--deadline-s", "3600", "--no-device-claim", "t",
                                 "--card", str(tmp_path / "nocard"), "--numa-membind", "3"])
    assert m.main(argv) == 0 and seen == [m.CONFIG_NAME]
    out = tmp_path / "out"
    v = json.loads((out / "verdict.json").read_text())
    assert v["decisions"]["locus"] == "none_reproduced" and v["rules_sha256"] == m.rules_sha256()
    assert json.loads((out / "rules.json").read_text())["rules_sha256"] == m.rules_sha256()
    recs = list((out / "records").glob("*.json"))
    assert len(recs) == 1 and len(list((out / "cells" / m.LAUNCH).glob("*.json"))) == 4
    rec = json.loads(recs[0].read_text())
    assert rec["kpf_task"] == "KPF-17.4c" and rec["spec_off"] and rec["decisions"]["locus"] == "none_reproduced"
    assert rec["target_n_swa_from_log"] == [0]
    body = {k: v for k, v in rec.items() if k != "record_sha256"}
    assert rec["record_sha256"] == k13.sha256_json(body)
    # resume: every cell stored -> no load
    seen.clear()
    assert m.main(argv + ["--resume"]) == 0 and seen == []
    assert "all cells stored, no load" in capsys.readouterr().out


def test_main_drains_on_the_stop_file_before_the_launch(tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(k13, "Server", _fake_server_factory(LayoutFake(), seen))
    monkeypatch.setattr(k13, "port_listening", lambda p: False)
    monkeypatch.setenv("KPF13_REGION", "184-191")
    stop = tmp_path / "STOP"
    stop.write_text("")
    argv = small_args(tmp_path, ["--fork-bin", str(fake_bin(tmp_path)), "--deadline-s", "3600", "--no-device-claim", "t",
                                 "--card", str(tmp_path / "nocard"), "--stop-file", str(stop)])
    assert m.main(argv) == 5 and seen == []
