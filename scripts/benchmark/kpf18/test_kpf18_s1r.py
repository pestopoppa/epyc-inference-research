"""Server-free tests for kpf18s1r (item 185, KPF-18-S1 remainder + oracle repair).

Nothing here starts a server, touches the GPU or reads a model. The step functions run against FakeServer, a small
simulation of the fork build's HTTP surface: /completion (KPF-11 fork, own-slot reuse with checkpoint rollback, the
--cache-ram prompt cache save-on-reslot / load-into-slot rule, logit_bias), streaming TTFT, /slots save|restore|erase.
It exercises plumbing and record building; the rules themselves are tested on hand-made inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_kpf18_arms as TA  # noqa: E402  (loads the gate as a module: TA.GATE)
import kpf18_arms as A  # noqa: E402
import kpf18_s1r as S  # noqa: E402

GATE = TA.GATE


# ================================================================ wiring
def test_registered_as_item_185_with_resume_regex():
    assert A.ARM_CMDS["kpf18s1r"] == [185] and 185 in A.ARM_ITEMS and A.TASK_OF[185] == "KPF-18-S1R"
    assert 185 in GATE.ITEMS and 185 in A.BUILDERS and 185 in A.PREFLIGHTS
    assert set(A.RULES[185]) == {"common", "tail", "item2", "long", "kpf191a", "hsf"}
    assert A.rules_sha256([185]) and A.rules_sha256([185]) != A.rules_sha256([184])
    assert GATE._ARM_STEP_RE.match("185 (4) A_tailonly tail512 g0") and GATE._ARM_STEP_RE.match("184 item-3")
    assert not GATE._ARM_STEP_RE.match("186 x")


def mk_args(tmp_path, **over):
    ap = argparse.ArgumentParser()
    A.add_arguments(ap)
    S.add_arguments(ap)
    ns = ap.parse_args([])
    base = dict(out=str(tmp_path), card="/nonexistent", fork_bin="/x/bin/llama-server", fork_min=1024, n_predict=24,
                n_probs=5, port=18999, budget_min=1e9, window_id="w", gpu_window_id=None, trigger_tokens=32,
                kpf_configs="as-given", s1r_predeclaration=None, s1r_forced_seq=None, s1r_shm_dir=str(tmp_path / "shm"))
    base.update(over)
    for k, v in base.items():
        setattr(ns, k, v)
    return ns


def test_plan_order_estimates_and_resume_keys(tmp_path):
    args = mk_args(tmp_path)
    ctx = GATE.Ctx(args, GATE.PromptSet.from_spec(), TA.GPU_ARGV, ["numactl"], {})
    steps = A.BUILDERS[185](ctx, GATE)
    d = [s.desc for s in steps]
    assert "linkage + launch" in d[0] and "SIGTERM own pid" in d[-1] and sum("linkage + launch" in x for x in d) == 1
    body = d[1:-1]
    assert len(body) == 2 + 6 + 1 + 2 + 3 and all(x.startswith("185 ") for x in body)
    assert body[0].startswith("185 (4) A_tailonly tail512") and "FIRST step on the fresh load" in body[0]
    order = [x.split(" ")[1] for x in body]
    assert order == ["(4)", "(4)"] + ["(2)"] * 6 + ["(3)"] + ["step6"] * 2 + ["step7"] * 3
    total = sum(s.est_s for s in steps)
    assert 15 * 60 < total < 60 * 60
    # resume: finished arm steps are skipped inside an unfinished server block; the launch always re-runs
    done = {2, 3, 9}
    skip = GATE.resume_plan(d, lambda k: k in done)
    assert skip[0] is False and skip[2] and skip[3] and skip[9] and not skip[1] and not skip[-1]
    # descs are stable across restarts (the launch label's timestamp is stripped from the key)
    steps2 = A.BUILDERS[185](ctx, GATE)
    assert [GATE.step_key(185, k, s.desc) for k, s in enumerate(steps)] == [GATE.step_key(185, k, s.desc) for k, s in enumerate(steps2)]


def test_step_subset_and_unknown_step_is_a_preflight_problem(tmp_path):
    args = mk_args(tmp_path, s1r_steps="tail,bogus")
    ctx = GATE.Ctx(args, GATE.PromptSet.from_spec(), TA.GPU_ARGV, ["numactl"], {})
    probs, notes = A.PREFLIGHTS[185](ctx, GATE)
    assert any("bogus" in p for p in probs) and any("predeclaration" in p for p in probs)
    assert any("--s1r-forced-seq" in p for p in probs)
    steps = A.BUILDERS[185](ctx, GATE)
    assert len(steps) == 1 + 2 + 1


# ================================================================ pure helpers
def test_load_forced_checks_both_shas(tmp_path):
    ids = [5, 6, 7]
    good = {"ids": ids, "ids_sha256": S.sha256_ids(ids)}
    p = tmp_path / "f.json"
    p.write_text(json.dumps(good))
    doc, probs = S.load_forced(p, S.sha256_ids(ids))
    assert doc["ids"] == ids and probs == []
    _, probs = S.load_forced(p, S.FORCED_SHA256)
    assert probs and "predeclared" in probs[0]
    p.write_text(json.dumps({"ids": ids, "ids_sha256": "0" * 64}))
    _, probs = S.load_forced(p, S.sha256_ids(ids))
    assert any("does not match" in x for x in probs)


def test_predeclared_forced_file_matches_when_present():
    f = Path("/mnt/raid0/llm/tmp/f8-launchers-20261023/kpf18-s1-remainder/forced_seq_183p4.json")
    if not f.exists():
        pytest.skip("launcher dir not on this host")
    doc, probs = S.load_forced(f, S.FORCED_SHA256)
    assert probs == [] and len(doc["ids"]) == 256 and doc["prompt"]["ids_sha256"] == S.LONG_PROMPT_SHA256


def test_choose_fillers_have_zero_lcp_with_every_prompt_and_distinct_heads():
    ps = TA.filled_prompt_set()
    fil = S.choose_fillers(ps)
    assert len(fil) >= S.N_FILLERS and all(len(f) == S.FILLER_LEN for f in fil)
    heads = [f[0] for f in fil]
    assert len(set(heads)) == len(heads)
    for g in ps.groups.values():
        for seq in [g["trunk"], g["src_tail"], g["src_tail_long"], g["filler"], *g["suffixes"]]:
            assert all(A.lcp(f, seq) == 0 for f in fil)


def test_stall_and_page_residency(tmp_path):
    assert S.stall_ms({"ttft_ms": 120.0, "timings": {"prompt_ms": 20.0}}) == 100.0
    assert S.stall_ms({"ttft_ms": None, "timings": {"prompt_ms": 20.0}}) is None
    f = tmp_path / "x.bin"
    f.write_bytes(os.urandom(1 << 20))
    r = S.page_residency(f)
    assert r is not None and 0.0 <= r <= 1.0
    assert S.fsync_path(f) >= 0.0
    S.drop_pages(f)                      # must not raise (tmpfs may keep the pages: residency is measured, not assumed)
    assert S.page_residency(tmp_path / "missing") is None
    (tmp_path / "empty").write_bytes(b"")
    assert S.page_residency(tmp_path / "empty") is None


# ---------------------------------------------------------------- (4) A_tailonly
def _sib(n, pred, err=None):
    return {"role": "sibling", "n_fork_tokens": None if err else n, "predicted_p": pred, "error": err}


def test_tail_verdicts():
    T = 8000
    ck_unpinned = [{"n_tokens": 8000, "pinned": False}, {"n_tokens": 8508, "pinned": False}]
    a = S.eval_tail([_sib(8000, 8000)] * 3, T, ck_unpinned)
    b = S.eval_tail([_sib(0, 0)] * 3, T, [{"n_tokens": 8488, "pinned": False}, {"n_tokens": 8996, "pinned": False}])
    assert a["T_checkpoint_unpinned"] and not a["T_checkpoint_pinned"] and a["n_eq_T"] == 3
    v = S.verdict_tail({"result": a, "fresh_load": True}, {"result": b})
    assert v["verdict"] == "TAILONLY_OWN_NEAR_END_CKPT"
    conf = S.eval_tail([_sib(7996, 7996)] * 3, T, [{"n_tokens": 7996, "pinned": False}])
    assert S.verdict_tail({"result": conf}, {"result": b})["verdict"] == "TAILONLY_ARM_ORDER_CONFOUND"
    pinned = S.eval_tail([_sib(8000, 8000)] * 3, T, [{"n_tokens": 8000, "pinned": True}])
    assert S.verdict_tail({"result": pinned}, {"result": b})["verdict"] == "TAILONLY_PREDICTION_MISMATCH"
    off = S.eval_tail([_sib(8000, 8000)] * 2 + [_sib(None, 8000, err="HTTP 500")], T, ck_unpinned)
    assert off["n_error"] == 1 and S.verdict_tail({"result": off}, {"result": b})["verdict"] == "TAILONLY_PREDICTION_MISMATCH"
    assert S.verdict_tail(None, {"result": b})["verdict"] == "INCOMPLETE"


# ---------------------------------------------------------------- (2) item 2
def _cmp(identical=False, top1=True, maxabs=0.0, kld=0.0, prefix=256):
    return {"identical": identical, "tokens_identical": identical, "top1": top1, "maxabs_union": maxabs, "kld": kld,
            "prefix": prefix}


def _cell(fo, qo, fq=None, valid=True, cell="c"):
    return {"cell": cell, "valid": valid, "invalid_reasons": [] if valid else ["Q_not_prompt_cache_load_at_T:0,8512,0"],
            "fork_vs_own": fo, "pcache_vs_own": qo, "fork_vs_pcache": fq or _cmp()}


def test_item2_cell_validity():
    T, S_, n = 8000, 512, 4
    F = {"http": 200, "tokens": [1] * n, "timings": {"n_fork_tokens": T, "fork_src_kind": "checkpoint", "fork_src_slot": 0}}
    O = {"http": 200, "tokens": [1] * n, "timings": {"cache_n": T}}
    Q = {"http": 200, "tokens": [1] * n, "timings": {"cache_n": T, "prompt_n": S_, "n_fork_tokens": 0}}
    assert S.item2_cell_validity(F, O, Q, T=T, S=S_, n=n) == []
    Qbad = {**Q, "timings": {"cache_n": 0, "prompt_n": T + S_}}
    assert any(x.startswith("Q_not_prompt_cache_load_at_T") for x in S.item2_cell_validity(F, O, Qbad, T=T, S=S_, n=n))
    Fbad = {**F, "timings": {"n_fork_tokens": T - 4, "fork_src_kind": "checkpoint", "fork_src_slot": 0}}
    assert any(x.startswith("F_not_forked_at_T") for x in S.item2_cell_validity(Fbad, O, Q, T=T, S=S_, n=n))


def test_item2_verdicts():
    floor = _cmp(maxabs=0.05, kld=0.02, prefix=30)
    ok = _cmp(maxabs=0.03, kld=0.01, prefix=40)
    out = _cmp(maxabs=0.2, kld=0.01, prefix=40)
    cells = [_cell(ok, floor) for _ in range(6)]
    v = S.verdict_item2(cells, 256)
    assert v["verdict"] == "ITEM2_WITHIN_PLACEMENT_FLOOR" and v["judge_item2_with_pcache_floor"]["n_ok"] == 6
    v = S.verdict_item2(cells[:5] + [_cell(out, floor)], 256)
    assert v["verdict"] == "ITEM2_OUTSIDE_PLACEMENT_FLOOR" and v["judge_item2_with_pcache_floor"]["n_not_ok"] == 1
    # a zero (bit-identical) placement floor makes bit-identity owed
    zero = _cmp(identical=True)
    v = S.verdict_item2([_cell(ok, zero) for _ in range(6)], 256)
    assert v["verdict"] == "ITEM2_OUTSIDE_PLACEMENT_FLOOR" and v["judge_item2_with_pcache_floor"]["bit_identity_owed"]
    v = S.verdict_item2([_cell(zero, zero) for _ in range(6)], 256)
    assert v["verdict"] == "ITEM2_WITHIN_PLACEMENT_FLOOR"
    bad = [_cell(ok, floor, valid=False) for _ in range(4)] + [_cell(ok, floor) for _ in range(2)]
    v = S.verdict_item2(bad, 256)
    assert v["verdict"] == "ITEM2_CONTROL_NOT_ESTABLISHED" and len(v["invalid_cells"]) == 4
    assert S.verdict_item2(cells[:5], 256)["verdict"] == "INCOMPLETE"


# ---------------------------------------------------------------- (3) teacher-forced 18.3 part 4
def test_long_tf_rule():
    K = 20
    base = [-0.5 - 0.01 * i for i in range(K)]
    first = {"n_fork_tokens": 60000, "fork_src_kind": "checkpoint"}
    lp = {"child": [x + 0.001 for x in base], "coldA": list(base), "coldB": [x + 0.002 for x in base]}
    r = S.eval_long_tf(lp, child_first=first, T=60000, n_task=60512)
    assert r["verdict"] == "SATISFIED" and r["n_common_valid"] == K and r["predeclared_rule_satisfied"] is True
    lp2 = {"child": [x + 0.01 for x in base], "coldA": list(base), "coldB": [x + 0.002 for x in base]}
    assert S.eval_long_tf(lp2, child_first=first, T=60000, n_task=60512)["verdict"] == "NOT_SATISFIED"
    assert S.eval_long_tf(lp, child_first={"n_fork_tokens": 0}, T=60000, n_task=60512)["verdict"] == "NOT_SATISFIED"
    holes = {k: [None if i % 2 else x for i, x in enumerate(v)] for k, v in lp.items()}
    r = S.eval_long_tf(holes, child_first=first, T=60000, n_task=60512)
    assert r["verdict"] == "ABSTAIN" and r["n_common_valid"] == 10
    # identical cold runs: the floor is lifted to FLOOR_EPS, not zero
    lp3 = {"child": [x + 5e-5 for x in base], "coldA": list(base), "coldB": list(base)}
    r = S.eval_long_tf(lp3, child_first=first, T=60000, n_task=60512)
    assert r["floor_used"] == A.FLOOR_EPS and r["verdict"] == "SATISFIED"


# ---------------------------------------------------------------- step 6
def _r(tokens, dn, da, **t):
    return {"http": 200, "tokens": tokens, "timings": {"draft_n": dn, "draft_n_accepted": da, **t}}


def test_191a_classes_and_verdict():
    T, n = 8000, 3
    ch = _r([1, 2, 3], 10, 5, n_fork_tokens=T)
    fr = _r([1, 2, 3], 10, 5, cache_n=0)
    ss = _r([1, 2, 3], 10, 5, cache_n=T)
    c = S.eval_191a_cell(ch, fr, fr, ss, T=T, n=n)
    assert c["valid"] and c["class"] == "equal_fresh"
    fr2 = _r([1, 2, 4], 10, 5, cache_n=0)
    c2 = S.eval_191a_cell(ch, fr2, fr2, ss, T=T, n=n)
    assert c2["class"] == "equal_sameshape"
    c3 = S.eval_191a_cell(ch, fr2, fr2, _r([1, 9, 9], 10, 4, cache_n=T), T=T, n=n)
    assert c3["class"] == "not_equal"
    nod = S.eval_191a_cell(ch, _r([1, 2, 3], 0, 0, cache_n=0), fr, ss, T=T, n=n)
    assert not nod["valid"] and any("drafter did not run" in x for x in nod["invalid_reasons"])
    assert S.verdict_191a([c, c])["verdict"] == "FLAG_OFF_EQUAL"
    assert S.verdict_191a([c, c2])["verdict"] == "FLAG_OFF_EQUAL_SAMESHAPE"
    assert S.verdict_191a([c, c3])["verdict"] == "FLAG_OFF_NOT_EQUAL"
    assert S.verdict_191a([c, nod])["verdict"] == "FLAG_OFF_INVALID"
    assert S.verdict_191a([c])["verdict"] == "INCOMPLETE"
    assert c["kpf174a_above_swa_bound"] is False and S.eval_191a_cell(ch, fr, fr, ss, T=16000, n=n)["kpf174a_above_swa_bound"]


# ---------------------------------------------------------------- step 7
def test_hsf_rates_and_verdict():
    rows = []
    for m, ms in (("tmpfs_save", 100.0), ("tmpfs_restore", 50.0), ("disk_save", 120.0), ("disk_restore_cold", 400.0)):
        for k in range(3):
            rows.append({"mode": m, "ms": ms, "bytes": 1e9, "valid": True, "fsync_ms": 200.0 if m == "disk_save" else None,
                         "residency_before": 0.0 if m == "disk_restore_cold" else None})
    for k in range(3):
        rows.append({"mode": "cache_baseline", "ms": 10.0, "filler_bytes": 1e8, "valid": True})
        rows.append({"mode": "cache_save", "ms": 110.0, "entry_bytes": 1.1e9, "valid": True})
        rows.append({"mode": "cache_load", "ms": 60.0, "bytes": 1.1e9, "valid": True})
    r = S.eval_hsf(rows, 8000)
    m = r["modes"]
    assert m["tmpfs_save"]["rate_GBps_median"] == pytest.approx(10.0)
    assert m["tmpfs_restore"]["rate_GBps_median"] == pytest.approx(20.0)
    assert m["disk_save"]["rate_incl_fsync_GBps_median"] == pytest.approx(1e9 / 0.32 / 1e9)
    assert r["derived_disk_only_read_GBps"] == pytest.approx(1e9 / 0.35 / 1e9)
    assert r["cache_ram"]["save_rate_GBps"] == pytest.approx(1e9 / 0.1 / 1e9)
    assert r["cache_ram"]["load_rate_GBps"] == pytest.approx(1.1e9 / 0.05 / 1e9)
    assert r["n_min_over_modes"] == 3 and "NOT COVERED" in r["gather_rate"]
    assert S.verdict_hsf([r, r, r])["verdict"] == "MEASURED"
    short = S.eval_hsf(rows[:-1], 8000)
    assert S.verdict_hsf([r, r, short])["verdict"] == "MEASURED_PARTIAL"
    assert S.verdict_hsf([r])["verdict"] == "INCOMPLETE"


# ================================================================ a simulated fork-build server
class FakeServer:
    """The HTTP surface the steps use. Fork: the KPF-11 rule (A.predict_fork). Own reuse: LCP, rolled back to the largest
    checkpoint <= LCP when the slot holds more than the LCP (recurrent memory). --cache-ram: a request whose slot holds a
    prompt with f_keep < 0.5 saves that slot (unless an entry already contains it) and then loads the best entry
    (f_keep >= 0.25) into the slot, as server_prompt_cache::alloc/load do. Generated ids are a pure function of the prompt;
    a +bias logit_bias entry forces the first token. Slot files are written through symlinks like fopen does."""

    def __init__(self, slot_dir: Path, n_min=1024, ub=512):
        self.slot_dir, self.n_min, self.ub = slot_dir, n_min, ub
        self.s = {i: {"tokens": [], "ckpts": [], "fork": 0} for i in range(4)}
        self.cache: list[dict] = []
        self.files: dict[str, list[int]] = {}
        self.n_forks = self.n_fork_tokens = 0
        self.calls = []

    # --- /slots, /metrics
    def slots(self):
        out = []
        for i, st in self.s.items():
            ids = st["tokens"]
            marks = [c["n_tokens"] for c in st["ckpts"]] + [len(ids)]
            lad = A.prefix_hashes(ids, marks)
            out.append({"id": i, "is_processing": False, "n_prompt_tokens": len(ids), "prefix_hash": lad[len(ids)],
                        "checkpoints": [{"n_tokens": c["n_tokens"], "pos_min": 0, "pos_max": c["n_tokens"] - 1,
                                         "pinned": c["pinned"], "size_mib": 150.0, "prefix_hash": lad[c["n_tokens"]]}
                                        for c in st["ckpts"]],
                        "fork": {"n_fork_tokens": st["fork"]}})
        return out

    def metrics(self):
        return {"llamacpp:slot_forks_total": float(self.n_forks), "llamacpp:slot_fork_tokens_total": float(self.n_fork_tokens),
                "llamacpp:kv_cells_used": 0.0}

    def erase(self, slot):
        self.s[slot] = {"tokens": [], "ckpts": [], "fork": 0}

    def _req(self, method, path, body=None, timeout=0):
        slot = int(path.split("/slots/")[1].split("?")[0])
        action = path.split("action=")[1]
        p = self.slot_dir / body["filename"]
        if action == "save":
            ids = list(self.s[slot]["tokens"])
            with open(p, "wb") as fh:           # follows a symlink, like fopen
                fh.write(b"\0" * (100 * len(ids) + 64))
            self.files[body["filename"]] = ids
            return 200, {"id_slot": slot, "filename": body["filename"], "n_saved": len(ids), "n_written": 100 * len(ids) + 64,
                         "timings": {"save_ms": 1.0 + len(ids) / 1e4}}
        if action == "restore":
            ids = self.files[body["filename"]]
            n = os.path.getsize(p)
            self.s[slot] = {"tokens": list(ids), "ckpts": [], "fork": 0}
            return 200, {"id_slot": slot, "filename": body["filename"], "n_restored": len(ids), "n_read": n,
                         "timings": {"restore_ms": 2.0 + len(ids) / 1e4}}
        return 404, ""

    # --- prompt cache
    def _cache_path(self, slot, ids):
        cur = self.s[slot]["tokens"]
        if not cur:
            return                                                 # an empty slot never touches the cache (f_keep NaN)
        lcp = A.lcp(cur, ids)
        sim = lcp / len(ids)
        f_keep = (sim * len(ids)) / len(cur) if sim > 0.1 else 0.0
        if f_keep >= 0.5:
            return
        if not any(A.lcp(e["tokens"], cur) == len(cur) for e in self.cache):
            self.cache = [e for e in self.cache if A.lcp(e["tokens"], cur) != len(e["tokens"])]
            self.cache.append({"tokens": list(cur), "ckpts": [dict(c) for c in self.s[slot]["ckpts"]]})
        best, fk_best, sim_best = None, lcp / len(cur), lcp / len(ids)
        for e in self.cache:
            lc = A.lcp(e["tokens"], ids)
            fk, sm = lc / len(e["tokens"]), lc / len(ids)
            if fk < 0.25:
                continue
            if fk_best < fk and sim_best < sm:
                best, fk_best, sim_best = e, fk, sm
        if best is not None:
            self.cache.remove(best)
            self.s[slot] = {"tokens": best["tokens"], "ckpts": best["ckpts"], "fork": 0}

    # --- /completion
    def complete(self, body, timeout=0):
        ids, slot, n = list(body["prompt"]), body["id_slot"], body["n_predict"]
        self.calls.append({k: v for k, v in body.items() if k != "prompt"} | {"n_prompt": len(ids)})
        cache = body.get("cache_prompt", True)
        fork_ok = body.get("slot_fork", True)
        self._cache_path(slot, ids)
        p, src = 0, None
        if fork_ok and cache and self.n_min and len(ids) > self.n_min:
            pr = A.predict_fork(self.slots(), ids, slot, self.n_min, mode="checkpoint")
            p, src = pr["p"], pr["src"]
        if p:
            cache_n = p
            ck = [dict(c) for c in self.s[src]["ckpts"] if c["n_tokens"] == p]
            self.n_forks += 1
            self.n_fork_tokens += p
        elif cache:
            cur = self.s[slot]["tokens"]
            lcp = A.lcp(cur, ids)
            if lcp < len(cur):
                ok = [c["n_tokens"] for c in self.s[slot]["ckpts"] if c["n_tokens"] <= lcp]
                lcp = max(ok) if ok else 0
            cache_n = min(lcp, len(ids) - 1) if lcp == len(ids) else lcp
            ck = [dict(c) for c in self.s[slot]["ckpts"] if c["n_tokens"] <= cache_n]
        else:
            cache_n, ck = 0, []
        for pos in body.get("checkpoint_at", []):
            pos = len(ids) if pos == -1 else pos
            if 0 < pos <= len(ids) and not any(c["n_tokens"] == pos for c in ck):
                ck.append({"n_tokens": pos, "pinned": True})
        if len(ids) - cache_n > 8 and len(ids) > self.ub + 8:
            ck = [c for c in ck if c["pinned"] or c["n_tokens"] < len(ids) - self.ub]
            ck += [{"n_tokens": len(ids) - self.ub, "pinned": False}, {"n_tokens": len(ids) - 4, "pinned": False}]
        seed = sum(ids[::97]) + len(ids)
        gen = [(seed + 31 * i) % 997 for i in range(n)]
        bias = {int(t): b for t, b in body.get("logit_bias", [])}
        if bias and n:
            gen[0] = max(bias, key=bias.get)
        probs = [{"id": t, "token": "x", "logprob": -0.1 - 0.001 * (sum(ids[-3:]) % 7),
                  "top_logprobs": [{"id": t, "logprob": -0.1}, {"id": t + 1, "logprob": -3.0}]}
                 for t in gen] if body.get("n_probs") else []
        self.s[slot] = {"tokens": ids + gen[:max(0, n - 1)], "ckpts": ck, "fork": p}
        dn = 0 if body.get("speculative.n_max") == 0 or n < 2 else n // 2
        t = {"prompt_n": len(ids) - cache_n, "cache_n": cache_n, "predicted_n": n, "predicted_ms": 1.0,
             "prompt_ms": 5.0, "prompt_per_second": 100.0, "predicted_per_second": 30.0, "draft_n": dn,
             "draft_n_accepted": dn // 2}
        if p:
            t.update(n_fork_tokens=p, fork_src_kind="checkpoint", fork_src_slot=src)
        return {"http": 200, "wall_s": 0.01, "t_send": 0.0, "cache_prompt": cache, "tokens": gen, "probs": probs,
                "timings": t, "timings_keys": sorted(t)}

    def stream(self, body, stop=None, on_token=None, timeout=0):
        r = self.complete(body)
        r.update(ttft_ms=r["timings"]["prompt_ms"] + 7.0, token_times=[], aborted=False)
        return r


class FakeSrv(SimpleNamespace):
    def log_counts(self):
        return {}


def mk_run_env(tmp_path, **over):
    args = mk_args(tmp_path, **over)
    ps = TA.filled_prompt_set()
    ctx = GATE.Ctx(args, ps, TA.GPU_ARGV, ["numactl"], {})
    label = "kpf18s1r-test"
    slot_dir = ctx.out / "slot-save" / label
    slot_dir.mkdir(parents=True, exist_ok=True)
    Path(args.s1r_shm_dir).mkdir(parents=True, exist_ok=True)
    ctx.client = FakeServer(slot_dir, ub=512)
    srv = FakeSrv(label=label, fork_min=1024, argv=["/x/llama-server"] + TA.GPU_ARGV, binary="/x/llama-server",
                  readback={"slot_fork_from_launch_log": "cross-slot prefix fork enabled: mode = checkpoint, min tokens = 1024"},
                  env_note={}, linkage={}, sampler=None, t_start=None, t_stop=None, stop_rec={}, log=tmp_path / "srv.log")
    env = A.Env(ctx, GATE, srv, "dflash2")
    pre = tmp_path / "PREDECLARATION.md"
    pre.write_text("rules\n")
    args.s1r_predeclaration = str(pre)
    g = args.s1r_long_group
    P = ps.groups[g]["trunk"] + ps.groups[g]["suffixes"][0]
    forced = [11 + (i * 13) % 900 for i in range(256)]
    fpath = tmp_path / "forced.json"
    fpath.write_text(json.dumps({"ids": forced, "ids_sha256": S.sha256_ids(forced),
                                 "prompt": {"ids_sha256": S.sha256_ids(P)}}))
    args.s1r_forced_seq, args.s1r_forced_sha256 = str(fpath), S.sha256_ids(forced)
    probs = S._prepare(ctx, GATE)
    assert probs == [], probs
    return ctx, env


def _records(ctx):
    return [json.loads(p.read_text()) for p in sorted((ctx.out / "records").glob("*.json"))]


def test_steps_end_to_end_on_the_fake_server_and_verdict(tmp_path):
    ctx, env = mk_run_env(tmp_path, s1r_tf_positions=20, s1r_hsf_reps=3, n_predict=12)
    S.run_tail(env, "tail512")
    S.run_tail(env, "tail1000")
    for g, i in S.ITEM2_CELLS:
        S.run_item2_cell(env, g, i)
    S.run_long_tf(env)
    for g in S.KPF191_GROUPS:
        S.run_191a_cell(env, g)
    for g in S.HSF_GROUPS:
        S.run_hsf(env, g)
    recs = _records(ctx)
    assert len(recs) == 2 + 6 + 1 + 2 + 3
    for r in recs:                                         # schema kept: manifest + binary sha (None here) + argv + rows sha
        assert r["schema"] == "epyc.kpf.run.v1" and r["kpf18_item"] == 185 and r["rows_sha256"]
        assert r["s1r"]["step_complete"] and r["s1r"]["predeclaration_sha256"] == S.sha256_path(ctx.args.s1r_predeclaration)
        assert "argv" in r and "prompt_set_sha256" in r
    first = [r for r in recs if r["s1r"]["step_id"] == "tail:tail512"][0]
    assert first["s1r"]["fresh_load"] is True
    assert [r for r in recs if r["s1r"]["step_id"] == "tail:tail1000"][0]["s1r"]["fresh_load"] is False
    # the fake server's prompt cache really moved the source into slot 2 for Q
    i2 = [r["s1r"]["result"] for r in recs if r["s1r"]["step"] == "item2"]
    assert all(c["valid"] for c in i2), [c["invalid_reasons"] for c in i2]
    lg = [r["s1r"]["result"] for r in recs if r["s1r"]["step"] == "long"][0]
    assert lg["n_common_valid"] == 20 and lg["verdict"] in ("SATISFIED", "NOT_SATISFIED")
    assert all(v is None for v in lg["failfast_at"].values())
    hs = [r["s1r"]["result"] for r in recs if r["s1r"]["step"] == "hsf"]
    assert all(h["n_min_over_modes"] >= 3 or h["modes"]["disk_restore_cold"]["n"] < 3 for h in hs)
    assert all(h["modes"]["tmpfs_save"]["n"] == 3 and h["modes"]["cache_load"]["n"] == 3 for h in hs)
    assert not list(Path(ctx.args.s1r_shm_dir).glob("kpf18s1r-*"))          # own scratch removed
    assert not list((ctx.out / "slot-save" / env.srv.label).glob("hsf_*"))
    v = S.verdict(ctx.out, S.sha256_path(ctx.args.s1r_predeclaration), n_cmp=12)
    iv = v["item_verdicts"]
    assert iv["item4_tailonly"] == "TAILONLY_OWN_NEAR_END_CKPT"
    assert iv["item2_pcache_floor"] in ("ITEM2_WITHIN_PLACEMENT_FLOOR", "ITEM2_OUTSIDE_PLACEMENT_FLOOR")
    assert iv["step6_kpf191a"] in ("FLAG_OFF_EQUAL", "FLAG_OFF_EQUAL_SAMESHAPE", "FLAG_OFF_NOT_EQUAL")
    assert iv["step7_hsf"] in ("MEASURED", "MEASURED_PARTIAL")
    assert v["verdict"] == "COMPLETE"
    # records made under another predeclaration are refused
    v2 = S.verdict(ctx.out, "0" * 64, n_cmp=12)
    assert v2["verdict"] == "INCOMPLETE" and len(v2["excluded_predeclaration_mismatch"]) == len(recs)


def test_teacher_forcing_fails_fast_when_the_slot_does_not_extend(tmp_path):
    ctx, env = mk_run_env(tmp_path, s1r_tf_positions=20)
    real = ctx.client.complete

    def no_reuse(body, timeout=0):                         # a server that re-prefills every position
        b = dict(body)
        if len(b["prompt"]) > 60000 and b.get("slot_fork") is False and b.get("cache_prompt", True):
            b["cache_prompt"] = False
        return real(b)
    ctx.client.complete = no_reuse
    S.run_long_tf(env)
    lg = [r["s1r"]["result"] for r in _records(ctx) if r["s1r"]["step"] == "long"][0]
    assert lg["failfast_at"]["child"] == 1 and lg["verdict"] == "ABSTAIN"


def test_budget_guard_refuses_a_step_that_does_not_fit(tmp_path):
    ctx, env = mk_run_env(tmp_path, budget_min=0.01)
    with pytest.raises(GATE.Abort):
        S._need(env, 600.0)


def test_stop_file_refuses_a_new_step(tmp_path, monkeypatch):
    ctx, env = mk_run_env(tmp_path)
    stop = tmp_path / "STOP"
    monkeypatch.setenv("KPF18_S1R_STOPFILE", str(stop))
    S._need(env, 1.0)
    stop.write_text("")
    with pytest.raises(GATE.Abort, match="STOP file"):
        S._need(env, 1.0)


def test_verdict_takes_the_latest_finished_record_and_ignores_unfinished(tmp_path):
    out = tmp_path / "o"
    (out / "records").mkdir(parents=True)
    sha = "a" * 64

    def w(name, step_id, step, result, complete=True, end="2026-10-23T10:00:00Z"):
        (out / "records" / name).write_text(json.dumps({"kpf18_item": 185, "end_utc": end, "s1r": {
            "step_id": step_id, "step": step, "step_complete": complete, "predeclaration_sha256": sha, "result": result}}))
    b = S.eval_tail([_sib(0, 0)] * 3, 8000, [])
    a_bad = S.eval_tail([_sib(8000, 7996)] * 3, 8000, [{"n_tokens": 8000, "pinned": False}])
    a_good = S.eval_tail([_sib(8000, 8000)] * 3, 8000, [{"n_tokens": 8000, "pinned": False}])
    w("1.json", "tail:tail512", "tail", a_bad, end="2026-10-23T09:00:00Z")
    w("2.json", "tail:tail512", "tail", a_good, end="2026-10-23T09:30:00Z")
    w("3.json", "tail:tail1000", "tail", b)
    w("4.json", "item2:g0s0", "item2", _cell(_cmp(), _cmp()), complete=False)
    v = S.verdict(out, sha)
    assert v["item_verdicts"]["item4_tailonly"] == "TAILONLY_OWN_NEAR_END_CKPT"
    assert v["item_verdicts"]["item2_pcache_floor"] == "INCOMPLETE" and v["records_step_complete"] == 3
    assert v["items"]["item2_pcache_floor"]["folded_item3_v2"]["verdict"] == "NOT_READ"


def test_fold_item3v2_reads_its_own_rule_module(tmp_path):
    d = tmp_path / "i3"
    (d / "out" / "records").mkdir(parents=True)
    (d / "PREDECLARATION.md").write_text("x\n")
    (d / "item3v2_rule.py").write_text(
        "def verdict(out, sha):\n    return {'verdict': 'INCOMPLETE', 'valid_loads': 0, 'needed_loads': 3, "
        "'class_counts': {}, 'predeclaration_sha256': sha}\n")
    f = S.fold_item3v2(d)
    assert f["verdict"] == "INCOMPLETE" and f["predeclaration_sha256"] == hashlib.sha256(b"x\n").hexdigest()
    assert S.fold_item3v2(tmp_path / "nope")["verdict"] == "NOT_READ"
