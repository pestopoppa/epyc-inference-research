"""Server-free tests for the KPF-18.1/.2/.3 arms (pure functions, plan estimates, record building).

Nothing here starts a server, touches the GPU or reads a model: the gate is imported as a module and its dry-run planners
are driven with a spec-only prompt set (token ids are never read at plan time).
"""
from __future__ import annotations

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
    assert set(A.ARM_CMDS) == {"kpf181", "kpf182", "kpf183", "kpf18arms"}
