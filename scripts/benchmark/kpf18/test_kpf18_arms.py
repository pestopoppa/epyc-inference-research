"""Server-free tests for the KPF-18.1/.2/.3 arms (pure functions, plan estimates, record building).

Nothing here starts a server, touches the GPU or reads a model: the gate is imported as a module and its dry-run planners
are driven with a spec-only prompt set (token ids are never read at plan time).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kpf18_arms as A  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("kpf18_gate_test_target", HERE / "kpf18_gate.py")
assert _SPEC is not None and _SPEC.loader is not None
GATE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = GATE
_SPEC.loader.exec_module(GATE)


# ---------------------------------------------------------------- helpers
def ref_hash(ids):
    h = A.FNV_BASIS
    for t in ids:
        t &= 0xFFFFFFFF
        for b in range(4):
            h ^= (t >> (8 * b)) & 0xFF
            h = (h * A.FNV_PRIME) & ((1 << 64) - 1)
    return f"{h:016x}"


def mk_slot(sid, ids, ckpts=(), processing=False):
    """A /slots row as the fork build reports it: n_prompt_tokens, prefix_hash of the whole prompt, per-checkpoint hashes."""
    marks = [c for c in ckpts] + [len(ids)]
    lad = A.prefix_hashes(ids, marks)
    return {"id": sid, "is_processing": processing, "n_prompt_tokens": len(ids), "prefix_hash": lad[len(ids)],
            "checkpoints": [{"n_tokens": c, "pos_min": 0, "pos_max": c - 1, "pinned": False, "prefix_hash": lad[c]}
                            for c in ckpts]}


def toks(n, base=1):
    return [base + (i * 7919) % 50000 for i in range(n)]


# ---------------------------------------------------------------- hash ladder
def test_prefix_hash_matches_a_plain_fnv_walk_at_every_mark():
    ids = toks(50)
    lad = A.prefix_hashes(ids, [0, 1, 7, 50, 51])
    assert lad[0] == f"{A.FNV_BASIS:016x}"
    for m in (1, 7, 50):
        assert lad[m] == ref_hash(ids[:m])
    assert 51 not in lad  # beyond the prompt: the server never reports it either


def test_lcp():
    assert A.lcp([1, 2, 3], [1, 2, 9, 3]) == 2
    assert A.lcp([], [1]) == 0


# ---------------------------------------------------------------- the KPF-11 predictor
def test_predictor_picks_the_largest_checkpoint_not_beyond_the_shared_prefix():
    src = toks(8000)
    slots = [mk_slot(0, src, ckpts=(2000, 5000, 7996)), mk_slot(1, [], ())]
    task = src[:5000] + [99999] + toks(600, base=3)           # shares exactly 5000 tokens
    p = A.predict_fork(slots, task, dst_id=1, n_min=1024)
    assert (p["p"], p["src"], p["kind"]) == (5000, 0, "checkpoint")
    assert [c["p"] for c in p["candidates"]] == [2000, 5000]  # 7996 is past the shared prefix


def test_predictor_cell_boundaries_c_minus_1_c_c_plus_1():
    src = toks(9000)
    slots = [mk_slot(0, src, ckpts=(4000, 8496, 8996))]
    for lcp_, want in ((3999, 0), (4000, 4000), (4001, 4000), (8495, 4000), (8496, 8496), (8497, 8496),
                       (8995, 8496), (8996, 8996)):
        task = src[:lcp_] + [99999] + toks(300, base=3)
        assert A.predict_fork(slots, task, 1, 1024)["p"] == want, lcp_


def test_predictor_checkpoint_must_be_below_task_length_and_above_min():
    src = toks(3000)
    slots = [mk_slot(0, src, ckpts=(500, 2996))]
    assert A.predict_fork(slots, src[:2996], 1, 1024)["p"] == 0          # p == n_task is excluded, 500 < n_min
    assert A.predict_fork(slots, src[:2997], 1, 1024)["p"] == 2996       # p < n_task
    assert A.predict_fork(slots, src[:2997], 1, 0)["p"] == 2996


def test_predictor_identical_resend_uses_the_n_minus_4_checkpoint():
    src = toks(6000)
    slots = [mk_slot(0, src, ckpts=(5484, 5996))]
    p = A.predict_fork(slots, src, 1, 1024)
    assert (p["p"], p["kind"]) == (5996, "checkpoint")


def test_predictor_end_kind_needs_an_idle_source_that_is_a_prefix_of_the_task():
    src = toks(4000)
    task = src + toks(100, base=5)
    idle = [mk_slot(0, src)]
    busy = [mk_slot(0, src, processing=True)]
    assert A.predict_fork(idle, task, 1, 1024)["kind"] == "end" and A.predict_fork(idle, task, 1, 1024)["p"] == 4000
    assert A.predict_fork(busy, task, 1, 1024)["p"] == 0
    assert A.predict_fork(idle, task, 1, 1024, n_swa=128)["p"] == 0     # SWA: no end snapshot
    assert A.predict_fork(idle, task, 1, 1024, enabled=False)["p"] == 0  # fork-off arm


def test_predictor_kv_mode_uses_the_exact_lcp_capped_below_the_task_length():
    src = toks(3000)
    slots = [mk_slot(0, src)]
    p = A.predict_fork(slots, src[:2000], 1, 1024, mode="kv", lcp_by_slot={0: 2000})
    assert (p["p"], p["kind"]) == (1999, "kv")
    assert A.predict_fork(slots, src[:2000], 1, 1024, mode="kv", lcp_by_slot={0: 500})["p"] == 0


def test_predictor_prefers_an_idle_source_on_a_tie_and_flags_it():
    src = toks(5000)
    slots = [mk_slot(0, src, ckpts=(4000,), processing=True), mk_slot(2, src, ckpts=(4000,))]
    p = A.predict_fork(slots, src[:4500], 1, 1024)
    assert p["src"] == 2 and p["tie"] is True


def test_predictor_ignores_the_destination_slots_own_checkpoints():
    src = toks(5000)
    slots = [mk_slot(1, src, ckpts=(4000,))]
    assert A.predict_fork(slots, src[:4500], 1, 1024)["p"] == 0


# ---------------------------------------------------------------- components and cells
def test_components_client_derived_then_server_overrides():
    pred = {"kind": "checkpoint", "p": 5000}
    c = A.components(pred, 5400, 6000, None)
    assert (c["tgt_kv_hit"], c["recurrent_ckpt_pos"], c["drafter_kv_hit"], c["drafter_exposed"]) == (5400, 5000, None, False)
    c2 = A.components(pred, 5400, 6000, {"drafter_kv_hit": 4096, "tgt_kv_hit": 5400})
    assert c2["drafter_kv_hit"] == 4096 and c2["source"]["drafter_kv_hit"] == "server" and c2["drafter_exposed"]
    assert A.components(pred, 7000, 6000, None)["tgt_kv_hit"] == 5999        # capped at n_task - 1


def test_server_components_reads_only_the_three_fork_fields():
    s = {"fork": {"n_fork_tokens": 10, "drafter_kv_hit": 7}}
    assert A.server_components(s) == {"drafter_kv_hit": 7}
    assert A.server_components({}) == {}


def test_cell_keeps_errors_and_never_invents_a_zero():
    ok = A.make_cell(arm="a", cell_id="c0", role="sib", slot=1, n_task=100, lcp_client=60, timings={"n_fork_tokens": 60},
                     n_fork_slot=60, pred={"p": 60, "kind": "checkpoint"}, ckpts_before=[], expected=60)
    bad = A.make_cell(arm="a", cell_id="c1", role="sib", slot=1, n_task=100, lcp_client=60, timings=None, n_fork_slot=None,
                      pred={"p": 60, "kind": "checkpoint"}, ckpts_before=[], expected=60, error="HTTP 500")
    assert ok["n_fork_tokens"] == 60 and bad["n_fork_tokens"] is None and bad["error"] == "HTTP 500"
    t = A.tally_expected([ok, bad])
    assert (t["n"], t["n_error"], t["n_equal_expected"], t["predeclared_rule_satisfied"]) == (2, 1, 1, False)


def test_tally_prediction_counts_zero_while_a_checkpoint_was_usable():
    def cell(i, got, pred):
        return A.make_cell(arm="b", cell_id=i, role="s", slot=1, n_task=10, lcp_client=5, timings={"n_fork_tokens": got},
                           n_fork_slot=got, pred={"p": pred, "kind": "checkpoint"}, ckpts_before=[])
    t = A.tally_prediction([cell("x", 0, 5000), cell("y", 5000, 5000), cell("z", 0, 0)])
    assert t["n_equal_predicted"] == 2 and t["n_zero_while_checkpoint_usable"] == 1
    assert t["zero_while_checkpoint_usable"] == ["x"]


def test_within_floor():
    vs = {"top1": True, "maxabs_union": 0.0005, "kld": 0.0002}
    assert A.within_floor(vs, 0.001, 0.001)["within"] is True
    assert A.within_floor(vs, 0.0001, 0.001)["within"] is False        # larger than the floor
    assert A.within_floor({**vs, "top1": False}, 1, 1)["within"] is False
    # a floor of exactly 0 is lifted to FLOOR_EPS, not left at 0
    assert A.within_floor({"top1": True, "maxabs_union": 5e-5, "kld": 0.0}, 0.0, 0.0)["within"] is True
    assert A.within_floor(None, 0.1, 0.1)["within"] is None


def test_floor_stats():
    f = A.floor_stats([{"maxabs_union": 0.1, "kld": 0.01, "prefix": 40}, {"maxabs_union": 0.3, "kld": 0.02, "prefix": 256}])
    assert f["max_maxabs"] == 0.3 and f["max_kld"] == 0.02 and f["prefix_min"] == 40


# ---------------------------------------------------------------- drafter configurations
GPU_ARGV = ("-m /m/Qwen3.8-27B-Q8_0.gguf -np 4 -ub 512 -md /m/Qwen3.8-27B-DFlash2-Q8_0.gguf -ngld 99 "
            "--spec-type draft-dflash --spec-draft-n-max 7 --device ROCm0 --device-draft ROCm0 -b 512 "
            "--no-cache-idle-slots").split()
CPU_ARGV = ("-m /m/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -np 4 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4 "
            "--reasoning off --device none --device-draft none --kv-unified").split()


def test_strip_speculative_removes_every_drafter_flag_and_keeps_the_rest():
    s = A.strip_speculative(GPU_ARGV)
    assert "-md" not in s and "--spec-type" not in s and "--device-draft" not in s and "-ngld" not in s
    assert "--device" in s and "ROCm0" in s and "-b" in s and "--no-cache-idle-slots" in s and "-ub" in s
    c = A.strip_speculative(CPU_ARGV)
    assert "--spec-type" not in c and "--spec-draft-n-max" not in c and "--device" in c and "none" in c
    assert "--reasoning" in c and "--kv-unified" in c


def test_config_names():
    assert A.config_name(GPU_ARGV) == "dflash2"
    assert A.config_name(CPU_ARGV) == "mtp"
    assert A.config_name(A.strip_speculative(GPU_ARGV)) == "nodraft"


def test_parse_configs_default_and_extra():
    cfgs = A.parse_configs("as-given,nodraft", [], GPU_ARGV, lambda p: [])
    assert [n for n, _ in cfgs] == ["dflash2", "nodraft"]
    cfgs = A.parse_configs("as-given,nodraft", ["mtp=/x/mtp.txt"], GPU_ARGV, lambda p: ["--spec-type", "draft-mtp"])
    assert [n for n, _ in cfgs] == ["dflash2", "nodraft", "mtp"]
    # an as-given argv that already has no drafter does not duplicate the nodraft arm
    assert [n for n, _ in A.parse_configs("as-given,nodraft", [], A.strip_speculative(GPU_ARGV), lambda p: [])] == ["nodraft"]
    with pytest.raises(ValueError):
        A.parse_configs("bogus", [], GPU_ARGV, lambda p: [])
    with pytest.raises(ValueError):
        A.parse_configs("as-given", ["nopath"], GPU_ARGV, lambda p: [])


def test_ub_from_argv():
    assert A.ub_from_argv(GPU_ARGV) == 512
    assert A.ub_from_argv(CPU_ARGV) == 2048
    assert A.ub_from_argv(["-m", "x"]) == 512


# ---------------------------------------------------------------- planning
def test_sweep_lengths_cover_exactly_one_ub_period_at_a_fixed_stride():
    L = A.sweep_lengths(8000, 512, 16)
    assert len(L) == 16 and L[0] == 7168 and L[1] - L[0] == 32 and L[-1] - L[0] == 15 * 32 < 512
    assert max(L) <= 8000
    assert len({x % 512 for x in L}) == 16                      # 16 distinct phases of the ubatch
    assert A.sweep_lengths(600, 512, 16) == []                  # trunk too short
    assert A.sweep_lengths(4096, 2048, 16)[0] == 2048


def test_d_positions_cells_and_holdout():
    pos = A.d_positions(8000, 16000, 512)
    assert pos == {"junction": 8000, "n_minus_4_minus_ub": 15484, "n_minus_4": 15996}
    cells = A.d_cells(pos)
    assert len(cells) == 9 and {c["delta"] for c in cells} == {-1, 0, 1}
    assert [c["lcp"] for c in cells if c["position_name"] == "junction"] == [7999, 8000, 8001]
    h = A.holdout_lcp(pos)
    assert all(abs(h - v) > 3 for v in pos.values()) and 8000 < h < 15484


def test_divergent_suffix_avoids_the_shared_next_token():
    got = A.divergent_suffix([[5, 1, 2], [6, 1, 2]], avoid_token=5, take=2)
    assert got == (1, [6, 1])
    assert A.divergent_suffix([[5]], 5, 1) is None


def test_dense_positions_are_bounded_by_the_checkpoint_at_limit():
    assert A.dense_positions(20000, 1024) == [1024 * k for k in range(1, 9)]
    assert A.dense_positions(3000, 1024) == [1024, 2048]
    assert all(p < 3000 for p in A.dense_positions(3000, 1024))


def test_classify_end_checkpoint_and_multi_turn_gap():
    assert A.classify_end_ckpt([1000], 1000, 64)["where"] == "prompt_end"
    assert A.classify_end_ckpt([1064], 1000, 64)["where"] == "generation_end"
    assert A.classify_end_ckpt([1063], 1000, 64)["where"] == "generation_end"
    assert A.classify_end_ckpt([], 1000, 64)["where"] == "absent"
    assert A.classify_end_ckpt([500], 1000, 64)["where"] == "other"
    g = A.multi_turn_gap(1000, 64, 1000)
    assert g["gap"] == 64 and g["covered"] is False
    assert A.multi_turn_gap(1000, 64, 1064)["covered"] is True
    assert A.multi_turn_gap(1000, 64, None)["covered"] is None


def test_mean_abs_dlogprob_uses_the_common_prefix_only():
    def run(lps, toks_):
        return {"tokens": toks_, "probs": [{"id": t, "logprob": lp} for t, lp in zip(toks_, lps)]}
    a = run([-0.1, -0.2, -0.3], [1, 2, 3])
    b = run([-0.1, -0.4, -9.0], [1, 2, 7])
    assert A.common_prefix_len([a, b]) == 2
    d = A.mean_abs_dlogprob(a, b, 2)
    assert d["n_positions"] == 2 and d["mean_abs_dlogprob"] == pytest.approx(0.1)
    # logprob falls back to the matching top_logprobs entry
    e = {"probs": [{"id": 4, "top_logprobs": [{"id": 9, "logprob": -5}, {"id": 4, "logprob": -0.5}]}]}
    assert A.logprob_of_chosen(e["probs"][0]) == -0.5


def test_rules_are_hashed_and_only_cover_arm_items():
    assert A.rules_for([1, 2, 3]) == {}
    assert A.rules_sha256([1]) is None


# ---------------------------------------------------------------- gate wiring
def test_gate_registers_the_arm_items_and_all_stays_1_to_7():
    for it in A.BUILDERS:
        assert it in GATE.ITEMS
    assert {1, 2, 3, 4, 5, 6, 7} <= set(GATE.ITEMS)
    assert set(A.ARM_CMDS) == {"kpf181", "kpf182", "kpf183", "kpf18arms", "kpf18s1"}


# ================================================================ a simulated server (plumbing tests)
class FakeClient:
    """Just enough of kpf18_gate.Client for the arms: /slots with checkpoints + prefix_hash ladder + fork block, /metrics
    fork counters, erase, and /completion with the fork behaviour of the KPF-11 rule (the predictor itself decides: these
    tests exercise the arm plumbing and record building, not the server). Generated tokens and logprobs are a pure
    function of the prompt, so a fork and a cold run of the same prompt are bit-identical."""

    def __init__(self, n_min=1024, ub=2048, enabled=True, hole=None, expose=False, drafter_hit=None, ignore_cache_flag=False,
                 chat_fails=False):
        self.n_min, self.ub, self.enabled = n_min, ub, enabled
        self.chat_fails = chat_fails
        self.expose, self.drafter_hit, self.ignore_cache_flag = expose, drafter_hit, ignore_cache_flag
        self.slots_state = {i: {"tokens": [], "ckpts": [], "last_fork": 0, "comps": {}} for i in range(4)}
        self.n_forks = self.n_fork_tokens = 0
        self.log = []
        self.hole = hole or (lambda ids, p: p)     # lets a test make the "server" deviate from the rule

    def slots(self):
        out = []
        for i, st in self.slots_state.items():
            toks_ = st["tokens"]
            marks = [c["n_tokens"] for c in st["ckpts"]] + [len(toks_)]
            lad = A.prefix_hashes(toks_, marks)
            out.append({"id": i, "is_processing": False, "n_prompt_tokens": len(toks_), "prefix_hash": lad[len(toks_)],
                        "checkpoints": [{"n_tokens": c["n_tokens"], "pos_min": 0, "pos_max": c["n_tokens"] - 1,
                                         "pinned": c["pinned"], "prefix_hash": lad[c["n_tokens"]]} for c in st["ckpts"]],
                        "fork": {"n_fork_tokens": st["last_fork"], **st.get("comps", {})}})
        return out

    def _req(self, method, path, body=None, timeout=0):
        """/detokenize and /v1/chat/completions, with a word-per-token tokenizer so a longer conversation extends the
        shorter one's token ids (what a chat template does to a growing message list)."""
        if path == "/detokenize":
            return 200, {"content": " ".join("w" for _ in body["tokens"])}
        if path == "/v1/chat/completions":
            if self.chat_fails:
                return 500, "chat template error"
            n_words = sum(len(m["content"].split()) for m in body["messages"])
            ids = [(i * 31 + 7) % 50000 for i in range(n_words)]
            r = self.complete({"prompt": ids, "n_predict": body["max_tokens"], "id_slot": body["id_slot"],
                               "cache_prompt": body.get("cache_prompt", True)})
            return 200, {"usage": {"prompt_tokens": len(ids)}, "choices": [{"message": {"content": "ok"}}],
                         "timings": r["timings"]}
        return 404, ""

    def metrics(self):
        return {"llamacpp:slot_forks_total": float(self.n_forks), "llamacpp:slot_fork_tokens_total": float(self.n_fork_tokens),
                "llamacpp:kv_cells_used": 0.0}

    def erase(self, slot):
        self.slots_state[slot] = {"tokens": [], "ckpts": [], "last_fork": 0, "comps": {}}

    def complete(self, body, timeout=0):
        ids, slot, n = body["prompt"], body["id_slot"], body["n_predict"]
        cache = True if self.ignore_cache_flag else body.get("cache_prompt", True)
        p, src = 0, None
        if self.enabled and cache and body.get("slot_fork", True):
            pr = A.predict_fork(self.slots(), ids, slot, self.n_min, mode="checkpoint")
            p, src = self.hole(ids, pr["p"]), pr["src"]
        seed = sum(ids[::97]) + len(ids)
        gen = [(seed + 31 * i) % 997 for i in range(n)]
        probs = [{"id": t, "token": "x", "logprob": -0.1 * (i % 5),
                  "top_logprobs": [{"id": t, "logprob": -0.1 * (i % 5)}, {"id": t + 1, "logprob": -3.0}]}
                 for i, t in enumerate(gen)] if body.get("n_probs") else []
        ck = []
        for pos in body.get("checkpoint_at", []):
            pos = len(ids) if pos == -1 else pos
            if 0 < pos <= len(ids):
                ck.append({"n_tokens": pos, "pinned": True})
        if len(ids) > self.ub + 8:
            ck += [{"n_tokens": len(ids) - 4 - self.ub, "pinned": False}, {"n_tokens": len(ids) - 4, "pinned": False}]
        comps = {}
        if self.expose and p:
            comps = {"tgt_kv_hit": p, "recurrent_ckpt_pos": p,
                     "drafter_kv_hit": self.drafter_hit(ids, p) if self.drafter_hit else p}
        self.slots_state[slot] = {"tokens": list(ids) + gen, "ckpts": ck, "last_fork": p, "comps": comps}
        t = {"prompt_n": len(ids) - p, "cache_n": p, "predicted_n": n, "predicted_ms": 1.0, "prompt_ms": 5.0,
             "prompt_per_second": 100.0, "predicted_per_second": 30.0}
        if p:
            t.update(n_fork_tokens=p, fork_src_kind="checkpoint", fork_src_slot=src)
            self.n_forks += 1
            self.n_fork_tokens += p
        return {"http": 200, "wall_s": 0.01, "t_send": 0.0, "cache_prompt": cache, "tokens": gen, "probs": probs,
                "timings": t, "timings_keys": sorted(t)}


@pytest.fixture()
def cpu_spec():
    names = ("TRUNKS", "SUFFIXES", "SRC_TAIL_LONG", "SPEED_GROUP", "CPU_MODE")
    saved = {k: getattr(GATE, k) for k in names}
    spec = dict(GATE.SPEC)
    GATE.set_cpu_spec()
    yield
    for k, v in saved.items():
        setattr(GATE, k, v)
    GATE.SPEC.clear()
    GATE.SPEC.update(spec)


def mk_args(tmp_path, **over):
    ap = argparse.ArgumentParser()
    A.add_arguments(ap)
    ns = ap.parse_args([])
    base = dict(out=str(tmp_path), card="/nonexistent", fork_bin="/x/bin/llama-server", fork_min=1024, n_predict=32,
                n_probs=5, port=18999, budget_min=1e9, window_id="w", gpu_window_id=None, trigger_tokens=32,
                item2_groups=None, item2_aa_repeats=1, item3_groups=None, item3_suffix=2)
    base.update(over)
    for k, v in base.items():
        setattr(ns, k, v)
    return ns


def filled_prompt_set():
    ps = GATE.PromptSet.from_spec()
    for g, grp in ps.groups.items():
        grp["trunk"] = toks(grp["T"], base=100 + g)
        grp["suffixes"] = [[60000 + 10 * g + i] + toks(L - 1, base=7 + i) for i, L in enumerate(grp["suffix_lens"])]
        grp["src_tail"] = [61000 + g] + toks(GATE.SRC_TAIL - 1, base=3)
        grp["src_tail_long"] = [62000 + g] + toks(GATE.SRC_TAIL_LONG - 1, base=5)
        grp["filler"] = toks(grp["T"], base=11)
    ps.spec_only = False
    ps.decoders = [toks(64), toks(64, base=2)]
    return ps


class FakeSrv(SimpleNamespace):
    pass


def mk_env(tmp_path, cpu_spec, *, cfg="mtp", fork_min=1024, argv=None, **argover):
    args = mk_args(tmp_path, fork_min=fork_min, **argover)
    gate_argv = argv or CPU_ARGV
    ctx = GATE.Ctx(args, filled_prompt_set(), gate_argv, ["taskset", "-c", "0"], {})
    ctx.client = FakeClient(n_min=fork_min, ub=A.ub_from_argv(gate_argv), enabled=bool(fork_min))
    srv = FakeSrv(fork_min=fork_min, argv=["/x/llama-server"] + gate_argv, readback={
        "slot_fork_from_launch_log": "cross-slot prefix fork enabled: mode = checkpoint, min tokens = 1024, cache_idle_slots = 0"
        if fork_min else None}, binary="/x/llama-server")
    return ctx, A.Env(ctx, GATE, srv, cfg)


# ---------------------------------------------------------------- KPF-18.1: evaluators
def _cell(role, got, *, expected=None, pred=None, **kw):
    return A.make_cell(arm="a", cell_id=kw.pop("cid", "c"), role=role, slot=1, n_task=1000, lcp_client=500,
                       timings={"n_fork_tokens": got} if got is not None else None, n_fork_slot=got,
                       pred={"p": pred if pred is not None else 0, "kind": "checkpoint"}, ckpts_before=[],
                       expected=expected, error=None if got is not None else "HTTP 500", **kw)


def test_eval_181_A_counts_and_never_drops_an_error():
    cells = [_cell("source", 0), _cell("sibling", 8000, expected=8000), _cell("sibling", 8000, expected=8000),
             _cell("sibling", None, expected=8000)]
    r = A.eval_181_A(cells)
    assert (r["n"], r["n_equal_expected"], r["n_error"]) == (3, 2, 1) and r["predeclared_rule_satisfied"] is False
    assert A.eval_181_A([_cell("sibling", 8000, expected=8000)])["predeclared_rule_satisfied"] is True
    tail = A.eval_181_A([_cell("sibling", 0), _cell("sibling", 0)])
    assert tail["predeclared_rule_satisfied"] is None and tail["n_fork_zero"] == 2 and tail["n_fork_max"] == 0


def test_eval_181_B_requires_prediction_floor_and_hits():
    good_vs = {"top1": True, "maxabs_union": 0.0, "kld": 0.0}
    floors = [{"maxabs_union": 0.0, "kld": 0.0, "prefix": 256}]
    ok = _cell("child", 5996, pred=5996, must_hit=True, cid="B1")
    ok["vs_fresh"] = good_vs
    r = A.eval_181_B([ok], floors)
    assert r["predeclared_rule_satisfied"] is True and r["n_within_floor"] == 1
    zero = _cell("child", 0, pred=5996, must_hit=True, cid="B2")
    zero["vs_fresh"] = good_vs
    r = A.eval_181_B([ok, zero], floors)
    assert r["predeclared_rule_satisfied"] is False
    assert r["n_zero_while_checkpoint_usable"] == 1 and r["n_must_hit_got_zero"] == 1
    bad = _cell("child", 5996, pred=5996, must_hit=True, cid="B4")
    bad["vs_fresh"] = {"top1": True, "maxabs_union": 0.2, "kld": 0.0}
    assert A.eval_181_B([bad], floors)["n_outside_floor"] == 1
    nov = _cell("child", 5996, pred=5996, cid="B5")                         # no cold control recorded: unmeasured, not passed
    assert A.eval_181_B([nov], floors)["predeclared_rule_satisfied"] is False
    off = A.eval_181_B([_cell("child", 0), _cell("child", 12)], [], fork_on=False)
    assert off["n_equal_expected"] == 1 and off["predeclared_rule_satisfied"] is False


def test_eval_181_C_all_lengths_must_fork_and_stay_within_the_floor():
    floors = [{"maxabs_union": 0.01, "kld": 0.001, "prefix": 100}, {"maxabs_union": 0.02, "kld": 0.002, "prefix": 256}]
    def child(L, got, m, top1=True):
        c = _cell("child", got, expected=L, cid=f"L{L}")
        c["vs_fresh"] = {"top1": top1, "maxabs_union": m, "kld": 0.0005, "prefix": 90}
        return c
    ok = [child(7168, 7168, 0.015), child(7200, 7200, 0.02)]
    r = A.eval_181_C(ok, floors)
    assert r["predeclared_rule_satisfied"] is True and r["greedy_prefix_vs_fresh_median"] == 90
    bad = ok + [child(7232, 7168, 0.0)]
    r = A.eval_181_C(bad, floors)
    assert r["n_not_forked_at_L"] == 1 and r["predeclared_rule_satisfied"] is False
    assert A.eval_181_C(ok + [child(7232, 7232, 0.5)], floors)["n_outside_floor"] == 1
    assert A.eval_181_C(ok + [child(7232, 7232, 0.0, top1=False)], floors)["predeclared_rule_satisfied"] is False


def test_eval_181_D_measures_the_drop_at_the_holdout_and_does_not_subtract_it():
    cells = [_cell("child", 8000, pred=8000, cid="j0"), _cell("child", 0, pred=0, cid="j-1")]
    h = _cell("child", 4096, pred=8000, cid="h", holdout=True, lcp_target=9000)
    r = A.eval_181_D(cells + [h])
    assert r["n_cells"] == 2 and r["predeclared_rule_satisfied"] is True
    assert r["holdout_drop"]["drop"] == 3904 and r["holdout_drop"]["kind"] == "partial"
    r2 = A.eval_181_D([_cell("child", 0, pred=8000, cid="x")])
    assert r2["predeclared_rule_satisfied"] is False and r2["n_zero_while_checkpoint_usable"] == 1
    h0 = _cell("child", 0, pred=8000, cid="h", holdout=True)
    assert A.eval_181_D(cells + [h0])["holdout_drop"]["kind"] == "full"


# ---------------------------------------------------------------- KPF-18.1: arms against the simulated server
def test_arm_A_fork_on_and_fork_off(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_181_A(env, [0], 3, with_ckpt=True, expect="T", tag="")
    rr = R.extra["rule_result"]
    assert rr["n_equal_expected"] == 3 and rr["predeclared_rule_satisfied"] is True
    assert all(c["n_fork_tokens"] == 4096 for c in R.extra["cells"] if c["role"] == "sibling")
    assert all(c["n_fork_tokens_slot"] == c["n_fork_tokens"] for c in R.extra["cells"] if c["role"] == "sibling")
    assert R.extra["counters_delta"]["slot_forks_total"] == 3
    assert all(v["ok"] == v["n"] for v in {k: {"ok": sum(1 for c in cs if c["ok"]), "n": len(cs)} for k, cs in R.checks.items()}.values())
    Rt = A.run_181_A(env, [0], 3, with_ckpt=False, expect=None, tag="")
    assert Rt.extra["rule_result"]["n_fork_max"] < 4096          # no junction: nothing reaches the trunk length
    ctx0, env0 = mk_env(tmp_path / "off", cpu_spec, fork_min=0)
    (tmp_path / "off").mkdir(exist_ok=True)
    R0 = A.run_181_A(env0, [0], 3, with_ckpt=True, expect="zero", tag="_forkoff")
    assert R0.extra["rule_result"]["predeclared_rule_satisfied"] is True and R0.extra["counters_delta"]["slot_forks_total"] == 0


def test_arm_A_detects_a_server_that_forks_short(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    ctx.client.hole = lambda ids, p: max(0, p - 3)               # the "server" loses 3 tokens
    R = A.run_181_A(env, [0], 3, with_ckpt=True, expect="T", tag="")
    assert R.extra["rule_result"]["predeclared_rule_satisfied"] is False
    assert R.extra["rule_result"]["n_not_equal_expected"] == 3


def test_arm_B_four_cases(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_181_B(env, 0, "")
    ch = {c["cell"]: c for c in R.extra["cells"] if c["role"] == "child"}
    assert set(ch) == {"B1_resend_no_ckpt", "B2_resend_ckpt_end", "B3_child_eq_trunk_junction", "B4_child_eq_trunk_trunk_source"}
    assert ch["B1_resend_no_ckpt"]["n_fork_tokens"] == ch["B1_resend_no_ckpt"]["n_task"] - 4
    b3 = ch["B3_child_eq_trunk_junction"]    # the junction at T is not usable (p < n_task); the n-4-ub checkpoint inside the trunk is
    assert b3["n_fork_tokens"] == b3["predicted_p"] == 4352 - 4 - 2048
    assert ch["B4_child_eq_trunk_trunk_source"]["n_fork_tokens"] == 4096 - 4
    assert R.extra["rule_result"]["predeclared_rule_satisfied"] is True
    assert len(R.eq["control_floor"]) == 2                       # P0 and trunk each controlled once, B2/B4 reuse


def test_arm_C_sweep_covers_sixteen_lengths_with_drafter_on_and_off(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    lengths = A.sweep_lengths(4096, 2048, 16)
    R = A.run_181_C(env, 0, lengths, True)
    rr = R.extra["rule_result"]
    assert rr["n_lengths"] == 16 and rr["n_forked_at_L"] == 16 and rr["predeclared_rule_satisfied"] is True
    Roff = A.run_181_C(env, 0, lengths, False)
    assert Roff.dflash2 is False and Roff.extra["drafter_requested"] is False
    assert all(c["dflash_requested"] is False for c in Roff.extra["cells"])
    assert Roff.extra["rule_result"]["n_lengths"] == 16


def test_arm_D_nine_cells_plus_holdout_match_the_prediction(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_181_D(env, 0, 2048, True)
    rr = R.extra["rule_result"]
    assert rr["n_cells"] == 9 and rr["predeclared_rule_satisfied"] is True
    assert rr["holdout_drop"]["drop"] == 0 and rr["holdout_drop"]["kind"] == "none"
    cells = {c["cell"]: c for c in R.extra["cells"] if c["role"] == "child" and not c.get("holdout")}
    assert cells["junction-1"]["n_fork_tokens"] == 0 and cells["junction+0"]["n_fork_tokens"] == 4096
    assert cells["n_minus_4+0"]["n_fork_tokens"] == 8192 - 4 and cells["n_minus_4-1"]["n_fork_tokens"] == 8192 - 4 - 2048
    assert R.extra["source_checkpoint_set"] == [4096, 8192 - 4 - 2048, 8192 - 4]


def test_arm_D_flags_a_drafter_that_disables_the_fork(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    ctx.client.hole = lambda ids, p: 0                           # a configuration that kills every fork
    R = A.run_181_D(env, 0, 2048, True)
    rr = R.extra["rule_result"]
    assert rr["predeclared_rule_satisfied"] is False and rr["n_zero_while_checkpoint_usable"] > 0
    assert rr["holdout_drop"]["kind"] == "full"


# ---------------------------------------------------------------- KPF-18.1: plan, dry-run estimates, preflight
def mk_plan_ctx(tmp_path, **over):
    args = mk_args(tmp_path, **over)
    ps = GATE.PromptSet.from_spec()
    return GATE.Ctx(args, ps, GPU_ARGV, ["numactl"], {}), args


def test_plan_181_gpu_shape_has_all_arms_and_a_fork_off_baseline(tmp_path):
    ctx, _ = mk_plan_ctx(tmp_path)
    steps = A.BUILDERS[181](ctx, GATE)
    descs = [s.desc for s in steps]
    assert sum("linkage + launch" in d for d in descs) == 3      # fork on (dflash2), fork-off baseline, nodraft D
    assert any("181 A [checkpoint_at=[T]]" in d for d in descs) and any("fork-OFF baseline" in d for d in descs)
    assert any(d.startswith("181 C") and "drafter off" in d for d in descs) and any(d.startswith("181 C") and "drafter on" in d for d in descs)
    assert any("181 D [nodraft]" in d for d in descs) and any("181 D [dflash2]" in d for d in descs)
    total = sum(s.est_s for s in steps)
    assert 600 < total < 4 * 3600                                # booking list: GPU ~1 h; an estimate, not a measurement
    assert all(s.est_s >= 0 for s in steps)


def test_plan_181_arm_selection_and_cpu_shape(tmp_path, cpu_spec):
    args = mk_args(tmp_path, kpf181_arms="D", kpf_configs="as-given")
    ctx = GATE.Ctx(args, GATE.PromptSet.from_spec(), CPU_ARGV, ["taskset"], {})
    steps = A.BUILDERS[181](ctx, GATE)
    assert sum("linkage + launch" in s.desc for s in steps) == 1 and any(s.desc.startswith("181 D [mtp]") for s in steps)


def test_preflight_181(tmp_path):
    ctx, args = mk_plan_ctx(tmp_path)
    probs, notes = A.PREFLIGHTS[181](ctx, GATE)
    assert probs == [] and any("181 C lengths 7168..7648 stride 32" in n for n in notes)
    args.kpf181_lengths = 8
    assert any(">= 16" in p for p in A.PREFLIGHTS[181](ctx, GATE)[0])
    args.kpf181_lengths = 16
    args.kpf181_groups = "9"
    assert any("group 9" in p for p in A.PREFLIGHTS[181](ctx, GATE)[0])
    args.kpf181_groups = None
    args.fork_min = 0
    assert any("--fork-min >= 1" in p for p in A.PREFLIGHTS[181](ctx, GATE)[0])


def test_rules_181_are_predeclared_and_hashed():
    assert set(A.RULES[181]) == {"common", "A", "B", "C", "D"}
    sha = A.rules_sha256([181])
    assert sha and sha == A.rules_sha256([181])
    assert A.rules_for([181])["181"]["task"] == "KPF-18.1"


# ================================================================ KPF-18.2
def test_eval_182_attr_is_tri_state_about_the_drafter_clause():
    def child(n_fork, comps, **kw):
        c = _cell("child", n_fork, pred=n_fork, cid=kw.pop("cid", "s"), **kw)
        c["components"], c["checkpoint_set_before"] = comps, [4096]
        return c
    full = {"tgt_kv_hit": 4096, "recurrent_ckpt_pos": 4096, "drafter_kv_hit": 4096}
    nodr = {"tgt_kv_hit": 4096, "recurrent_ckpt_pos": 4096, "drafter_kv_hit": None}
    assert A.eval_182_attr([child(4096, full)], drafter_on=True)["predeclared_rule_satisfied"] is True
    r = A.eval_182_attr([child(4096, nodr)], drafter_on=True)
    assert r["predeclared_rule_satisfied"] is None and r["n_drafter_not_measured"] == 1 and not r["drafter_clause_evaluated"]
    assert A.eval_182_attr([child(4096, nodr)], drafter_on=False)["predeclared_rule_satisfied"] is True
    low = {**full, "drafter_kv_hit": 2048}                       # drafter below n_fork: the minimum is 2048, n_fork 4096
    r = A.eval_182_attr([child(4096, low)], drafter_on=True)
    assert r["predeclared_rule_satisfied"] is False and r["n_drafter_below_nfork"] == 1 and r["n_nfork_equals_min"] == 0
    zero = {**full, "recurrent_ckpt_pos": 0}
    assert A.eval_182_attr([child(4096, zero)], drafter_on=False)["n_zero_component_with_fork"] == 1
    # an error is a failed sibling, not a missing one; and a run that never forked is vacuous
    assert A.eval_182_attr([child(None, full)], drafter_on=False)["predeclared_rule_satisfied"] is False
    v = A.eval_182_attr([child(0, {"tgt_kv_hit": 0, "recurrent_ckpt_pos": None, "drafter_kv_hit": None})], drafter_on=False)
    assert v["vacuous"] is True and v["predeclared_rule_satisfied"] is False


def test_eval_182_parity_lists_every_defect():
    vec = {"mtp:drafter_on": [("g0.s0", 4096), ("g0.s1", 4096)], "mtp:request_off": [("g0.s0", 4096), ("g0.s1", 4096)],
           "nodraft:launch": [("g0.s0", 4096), ("g0.s1", 0)]}
    r = A.eval_182_parity(vec)
    assert r["n_defects_total"] == 1 and r["predeclared_rule_satisfied"] is False
    assert r["comparisons"]["nodraft:launch"]["defects"] == [{"sibling": "g0.s1", "mtp:drafter_on": 4096, "nodraft:launch": 0}]
    assert r["comparisons"]["mtp:request_off"]["n_defects"] == 0
    vec["nodraft:launch"] = [("g0.s0", 4096), ("g0.s1", 4096)]
    assert A.eval_182_parity(vec)["predeclared_rule_satisfied"] is True
    assert A.eval_182_parity({"only": [("a", 1)]})["predeclared_rule_satisfied"] is False   # nothing to compare against
    never = {"a": [("x", 0)], "b": [("x", 0)]}                                               # nobody forked: vacuous, not a pass
    assert A.eval_182_parity(never)["predeclared_rule_satisfied"] is False
    assert A.eval_182_parity({})["predeclared_rule_satisfied"] is None


def test_eval_182_cold_and_policy():
    def cold(role, nf, cache_n=0):
        c = _cell(role, nf, cid=role)
        c["cache_n"] = cache_n
        return c
    ch = _cell("child", 4096, pred=4096, cid="c")
    ch["vs_fresh"] = {"top1": True, "maxabs_union": 0.0, "kld": 0.0}
    fl = [{"maxabs_union": 0.0, "kld": 0.0, "prefix": 16}]
    assert A.eval_182_cold([ch, cold("fresh_A", 0), cold("fresh_B", 0)], fl)["predeclared_rule_satisfied"] is True
    r = A.eval_182_cold([ch, cold("fresh_A", 0), cold("fresh_B", 100)], fl)
    assert r["predeclared_rule_satisfied"] is False and r["n_cold_with_fork_or_cache"] == 1
    assert A.eval_182_cold([ch], fl)["predeclared_rule_satisfied"] is None            # no cold run at all: not a pass
    j = _cell("child", 4348, pred=4348, cid="j", policy="junction_only")
    d = _cell("child", 4096, pred=4096, cid="d", policy="dense")
    p = A.eval_182_policy([j, d])
    assert p["dense_minus_junction_only"] == -252 and p["predeclared_rule_satisfied"] is True
    d2 = _cell("child", 3000, pred=4096, cid="d", policy="dense")
    assert A.eval_182_policy([j, d2])["predeclared_rule_satisfied"] is False


def test_protocol_drafter_clause_abstains_when_the_server_does_not_expose_it(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    shared: dict = {}
    R = A.run_182_protocol(env, [0], 3, "drafter_on", True, True, shared)
    rr = R.extra["rule_result"]
    assert rr["n_siblings"] == 3 and rr["n_nfork_equals_min"] == 3 and rr["n_drafter_not_measured"] == 3
    assert rr["predeclared_rule_satisfied"] is None and rr["vacuous"] is False
    assert R.extra["cold_control"]["predeclared_rule_satisfied"] is True and R.extra["cold_control"]["n_cold_runs"] == 6
    assert shared["vec"]["mtp:drafter_on"] == [("g0.s0", 4096), ("g0.s1", 4096), ("g0.s2", 4096)]
    ob = [o for o in R.kpf_obs if o["role"] == "child"][-1]
    assert ob["hit_components"]["tgt_kv_hit"] == 4096 and ob["hit_components"]["recurrent_ckpt_pos"] == 4096


def test_protocol_with_exposed_drafter_component(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    ctx.client.expose = True
    R = A.run_182_protocol(env, [0], 2, "drafter_on", True, False, {})
    assert R.extra["rule_result"]["predeclared_rule_satisfied"] is True
    assert R.extra["rule_result"]["n_drafter_not_measured"] == 0
    ctx.client.drafter_hit = lambda ids, p: p - 512                # the drafter only holds a shorter prefix
    R2 = A.run_182_protocol(env, [0], 2, "drafter_on", True, False, {})
    r2 = R2.extra["rule_result"]
    assert r2["predeclared_rule_satisfied"] is False and r2["n_drafter_below_nfork"] == 2 and r2["n_nfork_equals_min"] == 0


def test_parity_across_columns_and_launches(tmp_path, cpu_spec):
    shared: dict = {}
    ctx, env = mk_env(tmp_path, cpu_spec)
    A.run_182_protocol(env, [0], 3, "drafter_on", True, False, shared)
    A.run_182_protocol(env, [0], 3, "request_off", False, False, shared)
    nod_argv = A.strip_speculative(CPU_ARGV)
    ctx2, env2 = mk_env(tmp_path / "b", cpu_spec, cfg="nodraft", argv=nod_argv)
    A.run_182_protocol(env2, [0], 3, "launch", True, False, shared)
    assert list(shared["vec"]) == ["mtp:drafter_on", "mtp:request_off", "nodraft:launch"]
    R = A.run_182_parity_summary(env2, shared)
    assert R.extra["rule_result"]["predeclared_rule_satisfied"] is True and R.extra["rule_result"]["n_defects_total"] == 0
    # a nodraft launch whose forks differ is a counted defect
    ctx3, env3 = mk_env(tmp_path / "c", cpu_spec, cfg="dflash2")
    ctx3.client.hole = lambda ids, p: max(0, p - 1)
    shared2 = {"vec": dict(shared["vec"])}
    A.run_182_protocol(env3, [0], 3, "launch", True, False, shared2)
    r = A.run_182_parity_summary(env3, shared2).extra["rule_result"]
    assert r["predeclared_rule_satisfied"] is False and r["n_defects_total"] == 3


def test_cold_probe_passes_and_aborts_when_cache_prompt_false_is_not_honoured(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_182_cold_probe(env, 0)
    assert R.extra["rule_result"]["cold_control_expressible"] is True
    ctx2, env2 = mk_env(tmp_path / "x", cpu_spec)
    ctx2.client.ignore_cache_flag = True
    with pytest.raises(GATE.Abort):
        A.run_182_cold_probe(env2, 0)
    assert any(r.extra.get("rule_result", {}).get("cold_control_expressible") is False for r in ctx2.open_runs)


def test_policy_control_records_both_policies(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_182_policy(env, 0, 1024)
    rr = R.extra["rule_result"]
    assert rr["n_fork_junction_only"] == rr["n_fork_dense"] == 4352 - 4 and rr["predeclared_rule_satisfied"] is True
    src = {c["policy"]: c for c in R.extra["cells"] if c["role"] == "source"}
    assert src["dense"]["checkpoint_at_requested"] == [1024, 2048, 3072, 4096] and src["junction_only"]["checkpoint_at_requested"] == [4096]


def test_plan_and_preflight_182(tmp_path):
    ctx, args = mk_plan_ctx(tmp_path)
    steps = A.BUILDERS[182](ctx, GATE)
    descs = [s.desc for s in steps]
    assert sum("linkage + launch" in d for d in descs) == 2             # the dflash2 launch and the nodraft launch
    assert any("cold-control probe" in d for d in descs) and any("request_off" in d for d in descs)
    assert any("policy control" in d for d in descs) and any("parity summary" in d for d in descs)
    assert sum(1 for d in descs if d.startswith("182 protocol")) == 3   # dflash2/drafter_on, dflash2/request_off, nodraft/launch
    assert descs[-1].startswith("[") and "SIGTERM" in descs[-1]         # always ends with a verified-dead stop
    assert 300 < sum(s.est_s for s in steps) < 4 * 3600
    probs, notes = A.PREFLIGHTS[182](ctx, GATE)
    assert probs == [] and any("drafter_kv_hit" in n for n in notes)
    args.kpf182_parts = "1,9"
    assert any("subset of 1,2,3,4,5" in p for p in A.PREFLIGHTS[182](ctx, GATE)[0])
    args.kpf182_parts = "5"
    args.kpf182_dense_u = 100000
    assert any("dense-u" in p for p in A.PREFLIGHTS[182](ctx, GATE)[0])
    args.kpf182_dense_u = 1024
    args.kpf182_sibs = 9
    assert any("--kpf182-sibs" in p for p in A.PREFLIGHTS[182](ctx, GATE)[0])


def test_rules_182_cover_all_five_parts():
    assert set(A.RULES[182]) == {"common", "parity", "attribution", "vacuity", "cold", "policy"}


# ================================================================ KPF-18.3
def test_eval_183_cost_flags_a_split_cost_only_above_the_without_arm_spread():
    def rows(with_, without):
        return ([{"group": 0, "arm": "ckpt", "rep": i, "prompt_ms": v, "prompt_per_second": 1000 / v, "wall_s": v / 1000}
                 for i, v in enumerate(with_)]
                + [{"group": 0, "arm": "noCkpt", "rep": i, "prompt_ms": v, "prompt_per_second": 1000 / v, "wall_s": v / 1000}
                   for i, v in enumerate(without)])
    r = A.eval_183_cost(rows([105, 106, 104, 107, 105], [100, 101, 99, 100, 100]))
    g = r["groups"]["0"]
    assert g["delta"] == pytest.approx(0.05) and g["noise_floor_without"] == pytest.approx(0.02)
    assert g["trigger_intra_batch_capture_followup"] is True and r["n_groups_triggering"] == 1
    r = A.eval_183_cost(rows([100, 101, 99, 100, 100], [100, 101, 99, 100, 100]))
    assert r["groups"]["0"]["trigger_intra_batch_capture_followup"] is False
    r = A.eval_183_cost(rows([110, 110], [100, 100]))                        # 2 reps: no floor can be measured
    assert r["groups"]["0"]["trigger_intra_batch_capture_followup"] is None and r["n_groups_unmeasured"] == 1
    assert r["measurement_class"] == "timing_indicative"


def test_eval_183_long_floor_abstention_and_fork_requirement():
    def run(lp_shift=0.0, n=40):
        toks_ = list(range(1, n + 1))
        return {"tokens": toks_, "probs": [{"id": t, "logprob": -0.5 - lp_shift * (i % 3)} for i, t in enumerate(toks_)]}
    child, a_, b_ = run(0.00001), run(0.0), run(0.001)
    r = A.eval_183_long(child, a_, b_, n_fork=60000, T=60000, n_task=60512)
    assert r["abstain"] is False and r["within_floor"] is True and r["predeclared_rule_satisfied"] is True
    assert r["common_greedy_prefix"] == 40
    far = run(0.5)
    r = A.eval_183_long(far, a_, b_, n_fork=60000, T=60000, n_task=60512)
    assert r["within_floor"] is False and r["predeclared_rule_satisfied"] is False
    assert A.eval_183_long(child, a_, b_, n_fork=0, T=60000, n_task=60512)["predeclared_rule_satisfied"] is False   # never forked
    assert A.eval_183_long(child, a_, b_, n_fork=60000, T=60000, n_task=30000)["predeclared_rule_satisfied"] is False  # < 50k
    short = run(0.0, n=8)
    r = A.eval_183_long(short, short, short, n_fork=60000, T=60000, n_task=60512)
    assert r["abstain"] is True and r["predeclared_rule_satisfied"] is None


def test_classify_lane_checkpoints():
    c = A.classify_lane_checkpoints([900, 2000, 3001, 3005], 3005, 2048)
    assert c["near_end"] == [2000, 3001, 3005] and c["mid_prompt"] == [900]


def test_cost_arm_alternates_order_and_records_the_timing_rows(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_183_cost(env, [0, 1], 3)
    rows = R.extra["cost_rows"]
    assert len(rows) == 12 and {r["arm"] for r in rows} == {"ckpt", "noCkpt"}
    g0 = [r["arm"] for r in rows if r["group"] == 0]
    assert g0 == ["ckpt", "noCkpt", "noCkpt", "ckpt", "ckpt", "noCkpt"]        # A B / B A / A B
    assert R.extra["rule_result"]["groups"]["0"]["trigger_intra_batch_capture_followup"] is False
    assert R.extra["measurement_class"] == "timing_indicative"
    cells = [c for c in R.extra["cells"] if c["role"] == "cost_request"]
    assert [c["checkpoint_at"] for c in cells[:2]] == [[4096], None] and all(c["n_fork_tokens"] == 0 for c in cells)


def test_end_checkpoint_position_and_multi_turn(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_183_end(env, 0, 64)
    rr = R.extra["rule_result"]
    assert rr["end_checkpoint"]["where"] == "prompt_end" and rr["end_checkpoint"]["pinned"] == [4352]
    assert rr["turn2_fork"]["required"] == 4352 + 64 and rr["turn2_fork"]["gap"] == 0 and rr["predeclared_rule_satisfied"] is True
    # a server that only reaches the prompt end (not the reply) records the gap instead
    ctx2, env2 = mk_env(tmp_path / "b", cpu_spec)
    ctx2.client.hole = lambda ids, p: min(p, 4352)
    rr2 = A.run_183_end(env2, 0, 64).extra["rule_result"]
    assert rr2["turn2_fork"]["gap"] == 64 and rr2["turn2_fork"]["covered"] is False and rr2["predeclared_rule_satisfied"] is False


def test_lane_probe_tags_both_lanes_and_records_a_chat_failure_without_aborting(tmp_path, cpu_spec):
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_183_lane(env, 0, 3000)
    rr = R.extra["rule_result"]
    assert rr["lanes_recorded"] == ["chat", "completion_raw"] and rr["n_probe_errors"] == 0
    raw, chat = rr["probes"]
    assert raw["mid_prompt"] == [] and len(chat["near_end"]) == 2 and chat["turn2_n_fork_tokens"] == 3001
    assert set(R.lanes) == {"completion_raw", "chat"}
    ctx2, env2 = mk_env(tmp_path / "b", cpu_spec)
    ctx2.client.chat_fails = True
    rr2 = A.run_183_lane(env2, 0, 3000).extra["rule_result"]
    assert rr2["n_probe_errors"] == 1 and rr2["lanes_recorded"] == ["completion_raw"]


def test_long_prompt_arm_with_a_lowered_threshold(tmp_path, cpu_spec, monkeypatch):
    monkeypatch.setattr(A, "MIN_LONG_PROMPT", 4000)          # the CPU spec has no 50k trunk; the arm logic is the same
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_183_long(env, 0)
    rr = R.extra["rule_result"]
    assert rr["forked_at_T"] is True and rr["abstain"] is False and rr["predeclared_rule_satisfied"] is True
    assert rr["n_task"] == 4352 and rr["common_greedy_prefix"] == 32
    assert R.checks["prompt_ge_50k"][0]["ok"] is True


def test_plan_183_gpu_defaults_are_cost_and_long_prompt(tmp_path):
    ctx, args = mk_plan_ctx(tmp_path)
    steps = A.BUILDERS[183](ctx, GATE)
    descs = [s.desc for s in steps]
    assert len(steps) == 4 and "linkage + launch" in descs[0] and descs[1].startswith("183.1") and descs[2].startswith("183.4")
    assert "60512-token prompt (g5)" in descs[2]
    assert 300 < sum(s.est_s for s in steps) < 4 * 3600
    probs, notes = A.PREFLIGHTS[183](ctx, GATE)
    assert probs == [] and any("GPU default 1,4" in n for n in notes) and any("183.4 long trunk group 5" in n for n in notes)
    args.kpf183_reps = 2
    assert any("noise floor" in p for p in A.PREFLIGHTS[183](ctx, GATE)[0])
    args.kpf183_reps = 5
    args.kpf183_parts = "1,2,3,4"
    assert [s.desc[:5] for s in A.BUILDERS[183](ctx, GATE)][1:5] == ["183.1", "183.2", "183.3", "183.4"]
    args.kpf183_parts = "7"
    assert any("subset of 1,2,3,4" in p for p in A.PREFLIGHTS[183](ctx, GATE)[0])


def test_plan_183_cpu_defaults_are_end_checkpoint_and_lane_and_part_4_is_refused(tmp_path, cpu_spec):
    args = mk_args(tmp_path)
    ctx = GATE.Ctx(args, GATE.PromptSet.from_spec(), CPU_ARGV, ["taskset"], {})
    descs = [s.desc for s in A.BUILDERS[183](ctx, GATE)]
    assert [d[:5] for d in descs[1:3]] == ["183.2", "183.3"]
    probs, notes = A.PREFLIGHTS[183](ctx, GATE)
    assert probs == [] and any("--cpu default 2,3" in n for n in notes)
    args.kpf183_parts = "4"
    assert any("needs a trunk >= 50000" in p for p in A.PREFLIGHTS[183](ctx, GATE)[0])


def test_rules_183_and_the_three_commands_wire_to_three_items():
    assert set(A.RULES[183]) == {"common", "cost", "end", "lane", "long"}
    assert A.ARM_CMDS["kpf18arms"] == [181, 182, 183] and A.ARM_CMDS["kpf18s1"] == [184]
    assert set(A.BUILDERS) == {181, 182, 183, 184} and set(A.PREFLIGHTS) == {181, 182, 183, 184}
    assert A.rules_for([181, 182, 183]).keys() == {"181", "182", "183"}


# ================================================================ KPF-18.0e: item 2 judged at the A/A floor
def _cmp(top1=True, prefix=256, maxabs=0.0, kld=0.0, identical=False, tokens_identical=None):
    return {"top1": top1, "prefix": prefix, "maxabs_union": maxabs, "maxabs_intersection": maxabs, "kld": kld,
            "identical": identical, "tokens_identical": identical if tokens_identical is None else tokens_identical,
            "n_cmp": 256, "len_a": 256, "len_b": 256}


def test_judge_item2_without_an_aa_control_is_unjudgeable_never_a_pass():
    j = A.judge_item2([_cmp(identical=True)], [], 256)
    assert j["judgeable"] is False and j["predeclared_rule_satisfied"] is None and "not a pass" in j["reason"]
    assert A.judge_item2([], [_cmp(identical=True)], 256)["predeclared_rule_satisfied"] is None


def test_judge_item2_zero_floor_makes_bit_identity_owed():
    aa = [_cmp(identical=True) for _ in range(3)]
    ok = A.judge_item2([_cmp(identical=True), _cmp(identical=True)], aa, 256)
    assert ok["bit_identity_owed"] is True and ok["predeclared_rule_satisfied"] is True and ok["n_ok"] == 2
    assert ok["floor_thin"] is False
    # a fork pair that is only within eps but not bit-identical fails while the A/A floor is zero
    bad = A.judge_item2([_cmp(identical=True), _cmp(identical=False, maxabs=1e-6, kld=1e-9)], aa, 256)
    assert bad["predeclared_rule_satisfied"] is False and bad["not_ok_pairs"] == [1]
    assert bad["per_pair"][1]["identical_ok"] is False


def test_judge_item2_nonzero_floor_accepts_a_fork_difference_inside_it():
    # the A/A control itself differs (placement non-invariance): maxabs up to 0.04, KL up to 2e-3, shortest prefix 43
    aa = [_cmp(maxabs=0.04, kld=2e-3, prefix=43), _cmp(maxabs=0.01, kld=1e-4, prefix=256), _cmp(identical=True)]
    inside = _cmp(maxabs=0.03, kld=1e-3, prefix=60)
    j = A.judge_item2([inside], aa, 256)
    assert j["bit_identity_owed"] is False and j["aa_floor"]["max_maxabs"] == 0.04 and j["aa_floor"]["prefix_min"] == 43
    assert j["predeclared_rule_satisfied"] is True and j["n_within"] == 1 and j["n_prefix_ok"] == 1
    assert j["fork_bit_identical"] == 0


def test_judge_item2_fails_each_axis_beyond_the_floor_and_counts_it():
    aa = [_cmp(maxabs=0.04, kld=2e-3, prefix=43), _cmp(maxabs=0.02, kld=1e-3, prefix=100), _cmp(identical=True)]
    cases = [_cmp(maxabs=0.05, kld=1e-3, prefix=100),      # max-abs beyond the floor
             _cmp(maxabs=0.01, kld=5e-3, prefix=100),      # KL beyond the floor
             _cmp(maxabs=0.01, kld=1e-3, prefix=42),       # greedy prefix shorter than the shortest A/A prefix
             _cmp(top1=False, maxabs=0.0, kld=0.0),        # first token flips
             _cmp(maxabs=0.01, kld=1e-3, prefix=200)]      # fine
    j = A.judge_item2(cases, aa, 256)
    assert j["n_pairs"] == 5 and j["n_ok"] == 1 and j["n_not_ok"] == 4 and j["not_ok_pairs"] == [0, 1, 2, 3]
    assert j["predeclared_rule_satisfied"] is False
    assert [r["maxabs_ok"] for r in j["per_pair"][:2]] == [False, True]
    assert j["per_pair"][1]["kld_ok"] is False and j["per_pair"][2]["prefix_ok"] is False and j["per_pair"][3]["top1"] is False


def test_judge_item2_flags_a_thin_floor_and_none_handling():
    j = A.judge_item2([_cmp(identical=True)], [_cmp(identical=True)], 256)
    assert j["floor_thin"] is True and j["n_aa"] == 1
    # missing prob data (n_probs 0) cannot be within a floor: conservative failure, not a crash
    nop = _cmp(identical=False); nop["maxabs_union"] = None; nop["kld"] = None
    assert A.judge_item2([nop], [_cmp(maxabs=0.1, kld=0.1, prefix=10)], 256)["predeclared_rule_satisfied"] is False


class _PlacementNoiseClient(FakeClient):
    """FakeClient whose first-position probability depends on the request's call parity, the way a placement-sensitive
    kernel's does: an own continuation sits at 0 or `wobble_own` below the clean value (so own-vs-own replicates differ by
    up to `wobble_own`), a fork child sits a constant `wobble_fork` below it."""

    def __init__(self, *a, wobble_own=0.0, wobble_fork=0.0, **kw):
        super().__init__(*a, **kw)
        self.wobble_own, self.wobble_fork, self.calls = wobble_own, wobble_fork, 0

    def complete(self, body, timeout=0):
        r = super().complete(body, timeout)
        self.calls += 1
        is_child = bool((r.get("timings") or {}).get("n_fork_tokens"))
        if is_child:
            w = self.wobble_fork
        else:
            w = self.wobble_own * (self.calls % 2)
        if w and r["probs"]:
            for pr in r["probs"]:
                pr["top_logprobs"][0]["logprob"] -= w
        return r


def _item2_run(tmp_path, cpu_spec, client, aa_repeats, groups=(0,)):
    ctx, env = mk_env(tmp_path, cpu_spec)
    client.n_min, client.ub = ctx.client.n_min, ctx.client.ub
    ctx.client = client
    R = GATE.Run(ctx, 2, "fork_on", env.srv, True)
    n = ctx.args.n_predict
    for g in groups:
        T = GATE.G(ctx, g)["T"]
        if aa_repeats == 0:
            GATE.item2_seed_source(ctx, R, g, T)
        for i in range(2):
            GATE.item2_pair(ctx, R, g, i, T, n, ctx.args.n_probs, aa_repeats)
    return ctx, R, ARMS_JUDGE(R, n)


def ARMS_JUDGE(R, n):
    return A.judge_item2(R.eq.get("item2_fork_vs_own", []), R.eq.get("item2_aa_control", []), n)


def test_item2_pair_records_the_aa_control_beside_every_fork_vs_own_pair(tmp_path, cpu_spec):
    ctx, R, j = _item2_run(tmp_path, cpu_spec, FakeClient(), aa_repeats=2)
    assert len(R.eq["item2_fork_vs_own"]) == 2 and len(R.eq["item2_aa_control"]) == 4 and len(R.eq["item2_fork_repeat"]) == 4
    assert len(R.ident["bit_exact_ckpt"]) == 2                      # the legacy bit-exact count is still recorded
    assert j["judgeable"] is True and j["n_pairs"] == 2 and j["n_aa"] == 4
    # deterministic fake: fork == cold == own, A/A bit-identical -> floor zero -> bit-identity owed and met
    assert j["bit_identity_owed"] is True and j["predeclared_rule_satisfied"] is True


def test_item2_pair_with_aa_zero_is_the_legacy_flow_and_unjudgeable(tmp_path, cpu_spec):
    ctx, R, j = _item2_run(tmp_path, cpu_spec, FakeClient(), aa_repeats=0)
    assert len(R.eq["item2_fork_vs_own"]) == 2 and "item2_aa_control" not in R.eq
    assert j["judgeable"] is False and j["predeclared_rule_satisfied"] is None


def test_item2_a_fork_that_deviates_more_than_the_floor_fails_and_one_inside_it_passes(tmp_path, cpu_spec):
    # own-vs-own wobbles by 0.02 in probability space; a fork child wobbling by the same amount is inside the floor
    ctx, R, j = _item2_run(tmp_path, cpu_spec, _PlacementNoiseClient(wobble_own=0.02, wobble_fork=0.02), aa_repeats=2)
    assert j["bit_identity_owed"] is False and j["aa_floor"]["max_maxabs"] > 0
    assert j["predeclared_rule_satisfied"] is True, j
    # a fork child far outside that floor must fail every pair
    ctx2, R2, j2 = _item2_run(tmp_path / "far", cpu_spec, _PlacementNoiseClient(wobble_own=0.02, wobble_fork=0.9), aa_repeats=2)
    assert j2["predeclared_rule_satisfied"] is False and j2["n_not_ok"] == 2


def test_item2_plan_estimate_grows_with_aa_repeats_and_ends_with_the_judgement_step(tmp_path):
    ctx, args = mk_plan_ctx(tmp_path, item2_groups="0", item2_aa_repeats=0)
    legacy = GATE.item2(ctx)
    args.item2_aa_repeats = 2
    with_aa = GATE.item2(ctx)
    assert sum(s.est_s for s in with_aa) > sum(s.est_s for s in legacy)
    assert any("2 A/A repeat" in s.desc for s in with_aa) and any("NO A/A control" in s.desc for s in legacy)
    for steps in (legacy, with_aa):
        assert "judge item 2" in steps[-2].desc and steps[-1].desc.endswith("then finalize its records")
    assert "NOT owed" in A.ITEM2_RULE and "UNJUDGEABLE" in A.ITEM2_RULE


# ================================================================ KPF-18-S1 (one GPU sweep from one load)
def test_sweep_lengths_s1_are_offsets_around_the_ubatch_boundaries():
    L = A.sweep_lengths_s1(8000, 512)
    assert L == [7168, 7169, 7168 + 255, 7168 + 256, 7168 + 511, 7168 + 512] and max(L) <= 8000
    assert A.sweep_lengths_s1(600, 512) == []


def test_c_lengths_s1_cuts_arm_C_to_six_lengths_drafter_on_only(tmp_path):
    ctx, args = mk_plan_ctx(tmp_path, kpf181_arms="C", kpf181_c_lengths="s1")
    steps = A.BUILDERS[181](ctx, GATE)
    c = [s.desc for s in steps if s.desc.startswith("181 C")]
    assert len(c) == 1 and "drafter on (S1 cut): 6 trunk lengths" in c[0]
    probs, notes = A.PREFLIGHTS[181](ctx, GATE)
    assert probs == [] and any("181 C S1 cut" in n for n in notes)
    args.kpf181_c_lengths = "bogus"
    assert any("only 's1'" in p for p in A.PREFLIGHTS[181](ctx, GATE)[0])


def test_s1_composite_is_one_load_plus_the_aa_control(tmp_path):
    ctx, args = mk_plan_ctx(tmp_path)
    steps = A.BUILDERS[184](ctx, GATE)
    descs = [s.desc for s in steps]
    arm_launches = [d for d in descs if "linkage + launch" in d and "kpf18s1" in d]
    assert len(arm_launches) == 1                                   # ONE load for all the arm work
    assert not any("fork-OFF baseline" in d or "[nodraft]" in d for d in descs)
    for tag in ("181 A [checkpoint_at", "181 B", "181 C g0 drafter on (S1 cut)", "181 D [dflash2]", "182 cold-control",
                "182 protocol [dflash2 / drafter_on]", "182 protocol [dflash2 / request_off]", "182 policy",
                "182 parity summary", "183.1", "183.4"):
        assert any(tag in d for d in descs), tag
    assert not any("drafter off" in d for d in descs)
    first_arm = next(i for i, d in enumerate(descs) if "kpf18s1" in d)
    assert first_arm > 0                                            # the item-2 A/A launches come first
    # step 5 is ON by default: the localisation arm on the one load, then item 3 on group 3 on its own launch
    assert any("184 item-3 localisation g3" in d and "tokens 150..170" in d for d in descs)
    assert any(d.startswith("[item3-fork_on") and "linkage + launch" in d for d in descs)
    assert any(d.startswith("item3 g3 ") for d in descs) and not any(d.startswith("item3 g5") for d in descs)
    assert args.item3_groups == "3" and args.item2_groups == "0,3,5"      # A/A control trimmed to three trunk sizes
    assert not any(d.startswith("item2 g1.") or d.startswith("item2 g7.") for d in descs)
    args.s1_skip_aa = True
    assert "kpf18s1" in A.BUILDERS[184](ctx, GATE)[0].desc
    args.s1_no_item3 = True
    d2 = [s.desc for s in A.BUILDERS[184](ctx, GATE)]
    assert not any("localisation" in d or d.startswith("item3") or "[item3" in d for d in d2)
    args.s1_no_item3 = False
    args.n_predict = 256                                            # the fixture's 32 would never reach token 170
    probs, notes = A.PREFLIGHTS[184](ctx, GATE)
    assert probs == [] and any(n.startswith("S1:") for n in notes)


def test_s1_rules_predeclare_all_three_arm_families():
    r = A.rules_for([184])
    assert set(r) == {"181", "182", "183", "184"} and r["184"]["task"] == "KPF-18-S1"


def test_localisation_window_topk_and_margin():
    probs = [{"id": 7, "logprob": -0.1, "top_logprobs": [{"id": 7, "logprob": -0.1}, {"id": 9, "logprob": -2.0},
                                                          {"id": 3, "logprob": -5.0}]} for _ in range(10)]
    w = A.window_topk(probs, list(range(10)), 3, 6, k=2)
    assert [x["pos"] for x in w] == [3, 4, 5] and w[0]["top"] == [(7, -0.1), (9, -2.0)]
    assert w[0]["margin"] == pytest.approx(1.9) and w[0]["token"] == 3
    assert A.window_topk(probs, [], 8, 20)[-1]["pos"] == 9              # clipped to what was generated
    assert A.top2_margin({"top_logprobs": [{"id": 1, "logprob": -1}]}) is None


def _stream(tokens, margins=None, lp=-0.5):
    margins = margins or {}
    return {"tokens": tokens, "probs": [{"id": t, "logprob": lp, "top_logprobs": [
        {"id": t, "logprob": lp}, {"id": t + 1, "logprob": lp - margins.get(i, 3.0)}]} for i, t in enumerate(tokens)]}


def test_eval_184_localise_classifications():
    base = list(range(200))
    flip = base[:161] + [999] + base[162:]
    # only fork flips (A/A identical) with a margin above the A/A noise -> a fork-effect candidate, item 3 stays open
    r = A.eval_184_localise({"fork": _stream(flip), "nofork": _stream(base, {161: 0.5}), "nofork_repeat": _stream(base)}, 150, 171)
    assert r["pairs"]["fork_vs_nofork"]["first_divergence"] == 161 and r["pairs"]["fork_vs_nofork"]["in_window"] is True
    assert r["pairs"]["nofork_vs_repeat"]["first_divergence"] is None
    assert r["classification"] == "fork_effect_candidate" and r["keeps_item3_open"] is True
    assert r["margin_nofork_at_fork_divergence"] == pytest.approx(0.5) and r["aa_noise_max_abs_dlogprob"] == pytest.approx(0.0)
    # the no-fork arms also flip at 161 -> near-tie sensitive position
    r = A.eval_184_localise({"fork": _stream(flip), "nofork": _stream(base), "nofork_repeat": _stream(flip)}, 150, 171)
    assert r["classification"] == "near_tie_sensitive_position" and r["keeps_item3_open"] is False
    # the A/A flips earlier -> undetermined (stays open), no divergence at all -> none
    early = base[:100] + [998] + base[101:]
    assert A.eval_184_localise({"fork": _stream(flip), "nofork": _stream(base), "nofork_repeat": _stream(early)}, 150, 171)["classification"] == "aa_diverges_earlier"
    assert A.eval_184_localise({"fork": _stream(base), "nofork": _stream(base), "nofork_repeat": _stream(base)}, 150, 171)["classification"] == "no_fork_divergence"
    # a margin inside the A/A noise is NOT a fork-effect candidate
    noisy = _stream(base, lp=-0.5)
    noisy2 = _stream(base, lp=-0.9)                                       # A/A chosen-token logprobs differ by 0.4 everywhere
    r = A.eval_184_localise({"fork": _stream(flip), "nofork": noisy, "nofork_repeat": noisy2}, 150, 171)
    assert r["aa_noise_max_abs_dlogprob"] == pytest.approx(0.4) and r["classification"] == "fork_effect_candidate"   # margin 3.0 > 0.4
    r = A.eval_184_localise({"fork": _stream(flip), "nofork": _stream(base, {161: 0.2}), "nofork_repeat": noisy2}, 150, 171)
    assert r["classification"] == "within_aa_noise"


def test_localise_window_arg_and_preflight(tmp_path):
    ctx, args = mk_plan_ctx(tmp_path)
    assert A._loc_window(args) == (150, 171)
    args.s1_loc_window = "9"
    with pytest.raises(ValueError):
        A._loc_window(args)
    assert any("expected LO,HI" in p for p in A.PREFLIGHTS[184](ctx, GATE)[0])
    args.s1_loc_window = "150,171"
    args.n_predict = 100
    assert any("never generated" in p for p in A.PREFLIGHTS[184](ctx, GATE)[0])
    args.n_predict = 256
    args.s1_loc_group = 99
    assert any("--s1-loc-group 99" in p for p in A.PREFLIGHTS[184](ctx, GATE)[0])


# ================================================================ --resume
def test_step_key_ignores_launch_timestamps_but_not_content():
    a = GATE.step_key(184, 3, "[kpf18s1-dflash2-20261022T092027Z] linkage + launch")
    b = GATE.step_key(184, 3, "[kpf18s1-dflash2-20261023T011500Z] linkage + launch")
    assert a == b and a != GATE.step_key(184, 4, "[kpf18s1-dflash2-20261023T011500Z] linkage + launch")
    assert a != GATE.step_key(184, 3, "[kpf18s1-dflash2-20261023T011500Z] linkage + launch X")


DESCS = ["[item2-fork_on-1] linkage + launch + healthy", "open item-2 record", "item2 g0.0 S=512: fork child", "judge item 2",
         "[item2-fork_on-1] SIGTERM own pid, verify dead, then finalize its records",
         "[kpf18s1-d-1] linkage + launch + healthy", "181 A [checkpoint_at=[T]] groups [0]", "181 B g0: identical resend",
         "182 parity summary: x", "183.1 chunk-split cost", "[kpf18s1-d-1] SIGTERM own pid, verify dead, then finalize its records",
         "[item3-fork_on-1] linkage + launch + healthy", "item3 g3 T=32000", "[item3-fork_on-1] SIGTERM own pid, verify dead"]


def test_resume_plan_skips_finished_blocks_and_done_arm_steps_only():
    none = GATE.resume_plan(DESCS, lambda k: False)
    assert not any(none)
    # item-2 block finished (its stop step done): the whole block is skipped; the S1 block is unfinished:
    done = {4, 6, 7, 9}
    sk = GATE.resume_plan(DESCS, lambda k: k in done)
    assert sk[0:5] == [True] * 5
    assert sk[5] is False and sk[10] is False                          # the launch and the stop of an unfinished block re-run
    assert sk[6] and sk[7] and sk[9] and not sk[8]                     # done arm steps skip, the (not done) summary re-runs
    assert not any(sk[11:])
    # a done step that is NOT an arm step (item 2/3 keep closure state) re-runs inside an unfinished block
    sk2 = GATE.resume_plan(DESCS, lambda k: k in {2, 3})
    assert not any(sk2)
    # everything finished: every block skipped
    assert all(GATE.resume_plan(DESCS, lambda k: k in {4, 10, 13}))


def test_parity_vectors_survive_a_resume(tmp_path, cpu_spec):
    import json as _json
    ctx, env = mk_env(tmp_path, cpu_spec)
    rec = {"kpf_task": "KPF-18.2", "kpf18_arm": "parity_mtp_request_off", "column": "request_off", "kpf_config": "mtp",
           "siblings_vector": [["g0.s0", 4096], ["g0.s1", 4096]]}
    ref = dict(rec, kpf18_arm="parity_mtp_drafter_on", column="drafter_on")
    (tmp_path / "belief_measurements.jsonl").write_text(_json.dumps(rec) + "\n" + "not json\n" + _json.dumps(ref) + "\n")
    assert A.load_vectors_from_records(tmp_path)["mtp:request_off"] == [("g0.s0", 4096), ("g0.s1", 4096)]
    R = A.run_182_parity_summary(env, {})                              # nothing in memory: both columns come from the records
    rr = R.extra["rule_result"]
    assert rr["reference"] == "mtp:drafter_on" and rr["columns"] == ["mtp:drafter_on", "mtp:request_off"]
    assert rr["predeclared_rule_satisfied"] is True
    assert A.load_vectors_from_records(tmp_path / "nope") == {}


# ---------------------------------------------------------------- VB-KVU-PF.2: every record attests a rows file
def test_parity_summary_rows_file_holds_the_compared_columns(tmp_path, cpu_spec):
    import json as _json
    shared = {"vec": {"mtp:drafter_on": [("g0.s0", 4096)], "mtp:request_off": [("g0.s0", 4096)]}}
    ctx, env = mk_env(tmp_path, cpu_spec)
    R = A.run_182_parity_summary(env, shared)
    lines = [_json.loads(x) for x in R.rows.path.read_text().splitlines()]
    assert [x["column"] for x in lines] == ["mtp:drafter_on", "mtp:request_off"]
    assert lines[0]["siblings_vector"] == [["g0.s0", 4096]]


def test_finish_emits_rows_sha256_for_a_run_that_issued_no_request(tmp_path, cpu_spec):
    import json as _json
    ctx, env = mk_env(tmp_path, cpu_spec)
    srv = env.srv
    for k, v in dict(env_note={}, log=tmp_path / "srv.log", linkage=None, sampler=None, t_start=None, t_stop=None,
                     stop_rec=None, log_counts=lambda: {}).items():
        setattr(srv, k, v)
    ctx.bin_sha = lambda b: "0" * 64
    R = A.run_182_parity_summary(env, {})                  # no columns, no request: nothing written to rows
    assert not R.rows.path.exists()
    rec = R.finish()
    assert R.rows.path.is_file() and rec["rows_sha256"] == GATE.sha256_file(R.rows.path)
    assert isinstance(rec["rows_sha256"], str) and len(rec["rows_sha256"]) == 64
    side = [_json.loads(x) for x in (ctx.out / "belief_measurements.jsonl").read_text().splitlines()]
    assert side[-1]["arm"] == "parity_summary" and side[-1]["rows_sha256"] == rec["rows_sha256"]
