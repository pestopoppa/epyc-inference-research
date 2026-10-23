#!/usr/bin/env python3
"""KPF-18-S1 remainder + oracle repair: command `kpf18s1r` (item 185) of the KPF-18 gate (kpf18_gate.py).

handoffs/active/kv-prefix-fork-and-paged-attention.md, box "KPF-18-S1 remainder + oracle repair" (root 501f3a318):
  (4) A_tailonly on a fresh load (plus a tail-1000 sub-arm that separates "own near-end checkpoint" from "inherited junction")
  (2) item 2 re-judged against a placement-varied same-shape control: the source's own continuation loaded onto ANOTHER
      slot through the --cache-ram prompt cache (no fork), used as the floor of kpf18_arms.judge_item2
  (3) 18.3 part 4 with d() over the per-position logprob of a teacher-forced common sequence (one request per position,
      a +100 logit bias on the forced token; the reported logprob is the softmax of the RAW logits, so the bias does not
      move it)
  S1 step 6: KPF-19.1a on the GPU fork build, flag OFF (the flag and the mutation hook are not compiled into it: recorded)
  S1 step 7: HSF-4 / HSF-5 slot-file save/restore (tmpfs vs disk, cold restores after fadvise) and --cache-ram rates
The decision rules are the launcher's PREDECLARATION.md; its sha256 (--s1r-predeclaration) is stamped into every record
(`s1r.predeclaration_sha256`) and verdict() refuses records made under another sha. RULES[185] is the short machine copy
the gate writes into run_meta.json before any server starts.

One load (the argv file as given), steps in the predeclared order. Every step finishes ITS OWN record when it ends, so a
hard end loses at most the step in progress, and `--resume` re-runs only the steps without a done marker. A step whose
estimate (x --s1r-budget-factor) does not fit the remaining budget is not started (Abort -> the gate exits rc 3).

The pure evaluators and verdict() take no server and are what verdict.py in the launcher directory calls.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import mmap
import os
import statistics
import time
from pathlib import Path

import kpf18_arms as ARMS

ITEM = 185
CMD = "kpf18s1r"
TASK = "KPF-18-S1R"

# predeclared constants (PREDECLARATION.md)
FORCED_SHA256 = "ec9ad963c7482f6d703e6077ddced625d17c884a67f0fc5c80677c65c1296865"
LONG_PROMPT_SHA256 = "74ee91a9899677a30f9ec55bb2a0e5e6bc544d3e1deff0038f079f656f8148c8"
LOGIT_BIAS = 100.0
TF_MIN_COMMON = ARMS.MIN_COMMON_POSITIONS          # 16
TF_FAILFAST_PROMPT_N = 8
TF_POS_S = 0.15                                    # per teacher-forced position (1-token extension + HTTP), estimate
MIN_LONG = ARMS.MIN_LONG_PROMPT                    # 50000
ITEM2_CELLS = ((0, 0), (0, 1), (0, 2), (0, 3), (3, 0), (3, 2))
ITEM2_MIN_VALID = 3
TAIL_SUBARMS = ("tail512", "tail1000")
TAIL_EXTRA = 488                                   # tail1000 = src_tail (512) + src_tail_long[:488]
KPF191_GROUPS = (0, 1)
KPF191_N = 32
SWA_BOUND = 2048 * 4 + 512                         # DFlash2 n_swa x n_seq + ub (KPF-17.4a)
HSF_GROUPS = (0, 1, 3)
HSF_REPS = 3
HSF_EXTRA = 32
COLD_RESIDENCY_MAX = 0.05
FILLER_LEN = 64
N_FILLERS = 15                                     # item2 uses 0-1, HSF 2 (state size), 3-8 baseline, 9-14 cache reps
GB = 1e9
STEP_KEYS = ("tail", "item2", "long", "kpf191a", "hsf")

ARMS.ARM_ITEMS = tuple(sorted(set(ARMS.ARM_ITEMS) | {ITEM}))
ARMS.ARM_CMDS[CMD] = [ITEM]
ARMS.TASK_OF[ITEM] = TASK

ARMS.RULES[ITEM] = {
    "common": ("Rules: PREDECLARATION.md of the launcher (sha256 in every record as s1r.predeclaration_sha256; verdict() "
               "refuses another sha). One load of the argv file as given, steps in order tail, item2, long, kpf191a, hsf. "
               "Counts and rates are reported, never a pass word."),
    "tail": ("(4) A_tailonly as 18.1 rule A (no checkpoint_at, 3 siblings, RECORDED) as the FIRST step on a fresh load after "
             "an explicit erase, plus a tail1000 sub-arm (tail = src_tail + src_tail_long[:488]). TAILONLY_OWN_NEAR_END_CKPT "
             "iff every tail512 sibling n_fork_tokens == T == the KPF-11 prediction with T unpinned on slot 0 and every "
             "tail1000 sibling equals its prediction; TAILONLY_ARM_ORDER_CONFOUND iff every tail512 sibling < T; else "
             "TAILONLY_PREDICTION_MISMATCH."),
    "item2": ("(2) 6 cells (g0 s0-3, g3 s0,s2). F = fork child, O = own continuation (S1 item2_replicate); Q = trunk+tail "
              "seeded on slot 3, evicted into --cache-ram by a filler, loaded onto slot 2 (holding a filler) by trunk+"
              "suffix with slot_fork false. judge_item2(compare(F,O), floor = compare(Q,O)). Valid cell: F n_fork == T "
              "from slot 0, O cache_n == T, Q cache_n == T and prompt_n == S and no fork, 256 tokens each. "
              "ITEM2_WITHIN_PLACEMENT_FLOOR / ITEM2_OUTSIDE_PLACEMENT_FLOOR (>= 3 valid), ITEM2_CONTROL_NOT_ESTABLISHED "
              "(< 3 valid of 6 finished), INCOMPLETE."),
    "long": ("(3) g5 P = trunk 60000 + suffix0, child (fork at T) / coldA / coldB (cache_prompt false, checkpoint_at [T]); "
             "position i = one request P+forced[:i], n_predict 1, logit_bias +100 on forced[i], speculative.n_max 0; "
             "lp = reported raw-logit logprob of forced[i]; d = mean |dlp| over positions valid in all three; SATISFIED "
             "iff forked at T and len(P) >= 50000 and >= 16 common and max(d(child,A), d(child,B)) <= max(d(A,B), 1e-4); "
             "NOT_SATISFIED iff >= 16 common and not; ABSTAIN iff < 16 common. Forced sequence sha256 " + FORCED_SHA256),
    "kpf191a": ("Step 6, flag OFF only (the build lacks --slot-fork-dft-full and LLAMA_KPF_MUTATE): g0 and g1 suffix0, "
                "trunk checkpoint_at [-1] on slot 0, child fork on slot 1, fresh + fresh_B cold on slots 2/3, sameshape = "
                "own-slot cache hit with the same remainder on slot 3, 32 tokens. equal_fresh / equal_sameshape / "
                "not_equal on tokens + draft_n + draft_n_accepted. FLAG_OFF_EQUAL / FLAG_OFF_EQUAL_SAMESHAPE / "
                "FLAG_OFF_NOT_EQUAL / FLAG_OFF_INVALID."),
    "hsf": ("Step 7: L = 8000, 16000, 32000: tmpfs save/restore x3, disk save (+fsync) x3, cold disk restore x3 "
            "(fadvise DONTNEED, residency <= 5%), hot x1, --cache-ram save/load stall (TTFT - prompt_ms) x3 + filler "
            "baseline x3. Rates in GB/s with n; gather rate NOT COVERED. MEASURED / MEASURED_PARTIAL / INCOMPLETE."),
}


# ================================================================ CLI
def add_arguments(ap) -> None:
    g = ap.add_argument_group("KPF-18-S1 remainder (command kpf18s1r, item 185)")
    g.add_argument("--s1r-predeclaration", default=None, help="the launcher's PREDECLARATION.md (sha256 into every record)")
    g.add_argument("--s1r-forced-seq", default=None, help="forced_seq_183p4.json (teacher-forced sequence, sha-pinned)")
    g.add_argument("--s1r-forced-sha256", default=FORCED_SHA256)
    g.add_argument("--s1r-steps", default=",".join(STEP_KEYS), help="subset of " + ",".join(STEP_KEYS))
    g.add_argument("--s1r-tf-positions", type=int, default=256)
    g.add_argument("--s1r-long-group", type=int, default=5)
    g.add_argument("--s1r-hsf-reps", type=int, default=HSF_REPS)
    g.add_argument("--s1r-shm-dir", default="/dev/shm")
    g.add_argument("--s1r-budget-factor", type=float, default=1.25,
                   help="a step is started only if its estimate x this fits the remaining --budget-min")


# ================================================================ pure helpers (no server)
def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def sha256_path(p) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def load_forced(path, want_sha: str) -> tuple[dict | None, list[str]]:
    probs: list[str] = []
    if not path:
        return None, ["--s1r-forced-seq not given"]
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return None, [f"forced sequence {path}: {exc}"]
    ids = doc.get("ids") or []
    if sha256_ids(ids) != want_sha:
        probs.append(f"forced sequence ids sha256 {sha256_ids(ids)} != predeclared {want_sha}")
    if doc.get("ids_sha256") != sha256_ids(ids):
        probs.append("forced sequence file: ids_sha256 field does not match its ids")
    return doc, probs


def logprob_of_chosen(entry: dict):
    return ARMS.logprob_of_chosen(entry)


def choose_fillers(ps, n: int = 16, length: int = FILLER_LEN) -> list[list[int]]:
    """n fillers of `length` tokens with pairwise distinct first tokens that start no trunk, suffix or tail of the prompt
    set: their LCP with every prompt the steps send is 0, so a filler on a slot makes f_keep 0 (prompt-cache path) and two
    fillers are never 'already in the cache' copies of each other."""
    forbid = set()
    for g in ps.groups.values():
        for k in ("trunk", "src_tail", "src_tail_long", "filler"):
            if g.get(k):
                forbid.add(g[k][0])
        for s in g.get("suffixes") or []:
            if s:
                forbid.add(s[0])
    src = None
    for gi in sorted(ps.groups, reverse=True):
        sx = ps.groups[gi].get("suffixes") or []
        if sx and sx[-1] and len(sx[-1]) >= 2000:
            src = sx[-1]
            break
    if src is None:
        return []
    out, used = [], set()
    off = 1000
    while len(out) < n and off + length <= len(src):
        cand = src[off:off + length]
        if cand[0] not in forbid and cand[0] not in used:
            out.append(list(cand))
            used.add(cand[0])
        off += length + 7
    return out


def stall_ms(r: dict) -> float | None:
    """Pre-eval stall of a streamed request: client TTFT minus the server's prompt_ms."""
    t = r.get("timings") or {}
    if r.get("ttft_ms") is None or t.get("prompt_ms") is None:
        return None
    return float(r["ttft_ms"]) - float(t["prompt_ms"])


def page_residency(path) -> float | None:
    """Fraction of the file's pages resident in the page cache (mincore over a read-only mapping; faults nothing in)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size == 0:
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            return None
        try:
            n = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vec = (ctypes.c_ubyte * n)()
            if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                return None
            return sum(1 for v in bytes(vec) if v & 1) / n
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def fsync_path(path) -> float:
    """fsync the file (through a read-only fd, which Linux allows); returns the seconds it took."""
    t0 = time.time()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return time.time() - t0


def drop_pages(path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# ---------------------------------------------------------------- (4) A_tailonly
def eval_tail(cells: list[dict], T: int, slot0_ckpts: list[dict]) -> dict:
    sib = [c for c in cells if c.get("role") == "sibling"]
    ok = [c for c in sib if ARMS._ok(c)]
    vals = [c["n_fork_tokens"] for c in ok]
    t_unpinned = any(c.get("n_tokens") == T and not c.get("pinned") for c in slot0_ckpts)
    t_pinned = any(c.get("n_tokens") == T and c.get("pinned") for c in slot0_ckpts)
    return {"n_siblings": len(sib), "n_error": len(sib) - len(ok),
            "n_equal_prediction": sum(1 for c in ok if c["n_fork_tokens"] == c.get("predicted_p")),
            "n_eq_T": sum(1 for v in vals if v == T), "n_below_T": sum(1 for v in vals if v < T), "n_zero": sum(1 for v in vals if v == 0),
            "n_fork_min": min(vals) if vals else None, "n_fork_max": max(vals) if vals else None,
            "observed": [c.get("n_fork_tokens") for c in sib], "predicted": [c.get("predicted_p") for c in sib],
            "slot0_checkpoints": [{"n_tokens": c.get("n_tokens"), "pinned": c.get("pinned")} for c in slot0_ckpts],
            "T_checkpoint_unpinned": t_unpinned, "T_checkpoint_pinned": t_pinned, "T": T}


def verdict_tail(r512: dict | None, r1000: dict | None) -> dict:
    if not r512 or not r1000:
        return {"verdict": "INCOMPLETE", "have": [k for k, v in (("tail512", r512), ("tail1000", r1000)) if v]}
    a, b = r512["result"], r1000["result"]
    own = (a["n_error"] == 0 and a["n_siblings"] == 3 and a["n_eq_T"] == 3 and a["n_equal_prediction"] == 3
           and a["T_checkpoint_unpinned"] and not a["T_checkpoint_pinned"]
           and b["n_error"] == 0 and b["n_siblings"] == 3 and b["n_equal_prediction"] == 3)
    if own:
        v = "TAILONLY_OWN_NEAR_END_CKPT"
    elif a["n_error"] == 0 and a["n_siblings"] > 0 and a["n_below_T"] == a["n_siblings"]:
        v = "TAILONLY_ARM_ORDER_CONFOUND"
    else:
        v = "TAILONLY_PREDICTION_MISMATCH"
    return {"verdict": v, "tail512": a, "tail1000": b, "tail512_fresh_load": r512.get("fresh_load")}


# ---------------------------------------------------------------- (2) item 2 vs the prompt-cache control
def item2_cell_validity(F: dict, O: dict, Q: dict, *, T: int, S: int, n: int) -> list[str]:
    bad = []
    for nm, r in (("F", F), ("O", O), ("Q", Q)):
        if r.get("http") != 200 or r.get("error"):
            bad.append(f"{nm}_http")
        if len(r.get("tokens") or []) != n:
            bad.append(f"{nm}_tokens_{len(r.get('tokens') or [])}")
    tf, to, tq = F.get("timings") or {}, O.get("timings") or {}, Q.get("timings") or {}
    if not (tf.get("n_fork_tokens") == T and tf.get("fork_src_kind") == "checkpoint" and tf.get("fork_src_slot") == 0):
        bad.append(f"F_not_forked_at_T:{tf.get('n_fork_tokens')},{tf.get('fork_src_kind')},{tf.get('fork_src_slot')}")
    if to.get("cache_n") != T or (to.get("n_fork_tokens") or 0):
        bad.append(f"O_not_own_restore_at_T:{to.get('cache_n')},{to.get('n_fork_tokens')}")
    if tq.get("cache_n") != T or tq.get("prompt_n") != S or (tq.get("n_fork_tokens") or 0):
        bad.append(f"Q_not_prompt_cache_load_at_T:{tq.get('cache_n')},{tq.get('prompt_n')},{tq.get('n_fork_tokens')}")
    return bad


def verdict_item2(cells: list[dict], n_cmp: int, planned: int = len(ITEM2_CELLS)) -> dict:
    """cells: the `result` blocks of the finished item-2 records (one per cell)."""
    if len(cells) < planned:
        return {"verdict": "INCOMPLETE", "n_cells_finished": len(cells), "n_cells_planned": planned}
    valid = [c for c in cells if c.get("valid")]
    out = {"n_cells_finished": len(cells), "n_cells_valid": len(valid),
           "invalid_cells": [{"cell": c.get("cell"), "why": c.get("invalid_reasons")} for c in cells if not c.get("valid")],
           "fork_vs_pcache_bit_identical": sum(1 for c in valid if (c.get("fork_vs_pcache") or {}).get("identical")),
           "fork_vs_pcache_tokens_identical": sum(1 for c in valid if (c.get("fork_vs_pcache") or {}).get("tokens_identical")),
           "pcache_vs_own_bit_identical": sum(1 for c in valid if (c.get("pcache_vs_own") or {}).get("identical")),
           "fork_vs_own_bit_identical": sum(1 for c in valid if (c.get("fork_vs_own") or {}).get("identical"))}
    if len(valid) < ITEM2_MIN_VALID:
        out["verdict"] = "ITEM2_CONTROL_NOT_ESTABLISHED"
        return out
    j = ARMS.judge_item2([c["fork_vs_own"] for c in valid], [c["pcache_vs_own"] for c in valid], n_cmp)
    out["judge_item2_with_pcache_floor"] = j
    if j.get("predeclared_rule_satisfied") is None:
        out["verdict"] = "ITEM2_CONTROL_NOT_ESTABLISHED"
    else:
        out["verdict"] = "ITEM2_WITHIN_PLACEMENT_FLOOR" if j["predeclared_rule_satisfied"] else "ITEM2_OUTSIDE_PLACEMENT_FLOOR"
    return out


# ---------------------------------------------------------------- (3) 18.3 part 4, teacher-forced
def eval_long_tf(lp: dict[str, list], *, child_first: dict, T: int, n_task: int) -> dict:
    """lp[arm] = per-position logprob of forced[i] (None = position invalid in that arm), arms child/coldA/coldB."""
    names = ("child", "coldA", "coldB")
    K = min(len(lp.get(a) or []) for a in names) if all(lp.get(a) is not None for a in names) else 0
    common = [i for i in range(K) if all(lp[a][i] is not None for a in names)]
    forked = child_first.get("n_fork_tokens") == T and child_first.get("fork_src_kind") == "checkpoint"
    base = {"T": T, "n_task": n_task, "prompt_ge_50k": n_task >= MIN_LONG, "forked_at_T": forked,
            "child_first": child_first, "positions": K, "n_common_valid": len(common),
            "n_valid": {a: sum(1 for x in (lp.get(a) or []) if x is not None) for a in names}}
    if len(common) < TF_MIN_COMMON:
        return {**base, "verdict": "ABSTAIN", "reason": f"{len(common)} positions valid in all three < {TF_MIN_COMMON}",
                "predeclared_rule_satisfied": None}

    def d(x, y):
        ds = [abs(lp[x][i] - lp[y][i]) for i in common]
        return {"mean_abs_dlogprob": sum(ds) / len(ds), "max_abs_dlogprob": max(ds), "n_positions": len(ds),
                "pos0_abs_dlogprob": abs(lp[x][0] - lp[y][0]) if 0 in common else None}
    fa, fb, ab = d("child", "coldA"), d("child", "coldB"), d("coldA", "coldB")
    worst = max(fa["mean_abs_dlogprob"], fb["mean_abs_dlogprob"])
    floor = max(ab["mean_abs_dlogprob"], ARMS.FLOOR_EPS)
    sat = bool(forked and base["prompt_ge_50k"] and worst <= floor)
    return {**base, "d_child_vs_coldA": fa, "d_child_vs_coldB": fb, "d_coldA_vs_coldB": ab, "worst_child_vs_cold": worst,
            "floor_used": floor, "within_floor": worst <= floor, "predeclared_rule_satisfied": sat,
            "verdict": "SATISFIED" if sat else "NOT_SATISFIED"}


# ---------------------------------------------------------------- step 6, KPF-19.1a
def _dstats(r: dict) -> tuple:
    t = r.get("timings") or {}
    return (list(r.get("tokens") or []), t.get("draft_n"), t.get("draft_n_accepted"))


def eval_191a_cell(child: dict, fresh: dict, fresh_b: dict, same: dict, *, T: int, n: int) -> dict:
    bad = []
    for nm, r in (("child", child), ("fresh", fresh), ("fresh_B", fresh_b), ("sameshape", same)):
        if r.get("http") != 200 or r.get("error") or len(r.get("tokens") or []) != n:
            bad.append(f"{nm}_http_or_len")
    tc, tf, ts = child.get("timings") or {}, fresh.get("timings") or {}, same.get("timings") or {}
    if tc.get("n_fork_tokens") != T:
        bad.append(f"child_n_fork_{tc.get('n_fork_tokens')}_ne_T")
    if not (tf.get("draft_n") or 0) > 0:
        bad.append("fresh_draft_n_zero (drafter did not run)")
    if tf.get("cache_n") not in (0, None):
        bad.append(f"fresh_cache_n_{tf.get('cache_n')}")
    if ts.get("cache_n") != T or (ts.get("n_fork_tokens") or 0):
        bad.append(f"sameshape_not_own_hit_at_T:{ts.get('cache_n')},{ts.get('n_fork_tokens')}")
    c, f, fb, s = _dstats(child), _dstats(fresh), _dstats(fresh_b), _dstats(same)
    eq_fresh = c == f
    eq_same = c == s
    cls = "equal_fresh" if eq_fresh else ("equal_sameshape" if eq_same else "not_equal")
    return {"valid": not bad, "invalid_reasons": bad, "class": cls, "T": T,
            "tokens_eq_fresh": c[0] == f[0], "draft_eq_fresh": c[1:] == f[1:],
            "tokens_eq_sameshape": c[0] == s[0], "draft_eq_sameshape": c[1:] == s[1:],
            "fresh_aa_identical": f == fb, "fresh_aa_tokens_identical": f[0] == fb[0],
            "draft": {"child": c[1:], "fresh": f[1:], "fresh_B": fb[1:], "sameshape": s[1:]},
            "kpf174a_above_swa_bound": T > SWA_BOUND, "kpf174a_n_fork": tc.get("n_fork_tokens")}


def verdict_191a(cells: list[dict], planned: int = len(KPF191_GROUPS)) -> dict:
    if len(cells) < planned:
        return {"verdict": "INCOMPLETE", "n_cells_finished": len(cells)}
    valid = [c for c in cells if c.get("valid")]
    cls = [c["class"] for c in valid]
    if len(valid) < planned:
        v = "FLAG_OFF_INVALID"
    elif any(x == "not_equal" for x in cls):
        v = "FLAG_OFF_NOT_EQUAL"
    elif all(x == "equal_fresh" for x in cls):
        v = "FLAG_OFF_EQUAL"
    else:
        v = "FLAG_OFF_EQUAL_SAMESHAPE"
    legs = cells[0].get("legs") if cells else None
    return {"verdict": v, "n_cells_finished": len(cells), "n_cells_valid": len(valid),
            "classes": {k: cls.count(k) for k in ("equal_fresh", "equal_sameshape", "not_equal")},
            "invalid": [{"T": c.get("T"), "why": c.get("invalid_reasons")} for c in cells if not c.get("valid")],
            "kpf174a": [{"T": c.get("T"), "above_swa_bound": c.get("kpf174a_above_swa_bound"), "n_fork": c.get("kpf174a_n_fork")}
                        for c in cells],
            "legs": legs, "kpf174": "NOT_COVERED (the drafter cache's real pos_min is not exposed over HTTP)"}


# ---------------------------------------------------------------- step 7, HSF-4 / HSF-5
HSF_MODES = ("tmpfs_save", "tmpfs_restore", "disk_save", "disk_restore_cold", "cache_save", "cache_load", "cache_baseline")


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def eval_hsf(rows: list[dict], L: int) -> dict:
    """rows: one per measured operation {mode, ms, bytes, valid, ...}. Rates in GB/s (1e9 B/s) with n."""
    by = {m: [r for r in rows if r.get("mode") == m and r.get("valid")] for m in HSF_MODES + ("disk_restore_hot",)}
    out: dict = {"L": L, "unit_rate": "GB/s (1e9 bytes/s)", "unit_time": "ms", "modes": {}}
    for m, rs in by.items():
        ms = [r["ms"] for r in rs if r.get("ms") is not None]
        rates = [r["bytes"] / (r["ms"] / 1000.0) / GB for r in rs if r.get("bytes") and r.get("ms")]
        out["modes"][m] = {"n": len(rs), "ms_median": _med(ms), "ms_min": min(ms) if ms else None,
                           "ms_max": max(ms) if ms else None, "bytes_median": _med([r.get("bytes") for r in rs]),
                           "rate_GBps_median": _med(rates)}
    ds = by["disk_save"]
    out["modes"]["disk_save"]["rate_incl_fsync_GBps_median"] = _med(
        [r["bytes"] / ((r["ms"] + r["fsync_ms"]) / 1000.0) / GB for r in ds if r.get("bytes") and r.get("ms") is not None
         and r.get("fsync_ms") is not None])
    out["modes"]["disk_save"]["fsync_ms_median"] = _med([r.get("fsync_ms") for r in ds])
    out["modes"]["disk_restore_cold"]["residency_before_max"] = max(
        [r["residency_before"] for r in by["disk_restore_cold"] if r.get("residency_before") is not None], default=None)
    # derived: disk-only time = disk - tmpfs (labelled derived)
    m = out["modes"]

    def derived(a, b):
        if m[a]["ms_median"] is None or m[b]["ms_median"] is None or not m[a]["bytes_median"]:
            return None
        dt = m[a]["ms_median"] - m[b]["ms_median"]
        return (m[a]["bytes_median"] / (dt / 1000.0) / GB) if dt > 0 else None
    out["derived_disk_only_read_GBps"] = derived("disk_restore_cold", "tmpfs_restore")
    out["derived_note"] = "derived = bytes / (t_disk - t_tmpfs) from medians; D2H/H2D are in both terms"
    base = m["cache_baseline"]["ms_median"]
    b_l = _med([r.get("entry_bytes") for r in by["cache_save"]])
    b_f = _med([r.get("filler_bytes") for r in by["cache_baseline"]])
    sv, ld = m["cache_save"]["ms_median"], m["cache_load"]["ms_median"]
    out["cache_ram"] = {
        "save_stall_ms_median": sv, "load_stall_ms_median": ld, "baseline_stall_ms_median": base,
        "entry_bytes_median": b_l, "filler_bytes_median": b_f,
        "save_rate_GBps": ((b_l - b_f) / ((sv - base) / 1000.0) / GB) if None not in (b_l, b_f, sv, base) and sv > base else None,
        "load_rate_GBps": (b_l / ((ld - base) / 1000.0) / GB) if None not in (b_l, ld, base) and ld > base else None,
        "bytes_basis": "slot-file n_written at L (target KV + recurrent + token ids) + /slots checkpoint size_mib of the "
                       "evicted slot; the drafter state in the cache entry is NOT counted (rate = lower bound)"}
    out["n_min_over_modes"] = min(v["n"] for k, v in m.items() if k in HSF_MODES)
    out["gather_rate"] = "NOT COVERED (no mapped-host UMBP gather code exists)"
    return out


def verdict_hsf(results: list[dict], planned: int = len(HSF_GROUPS), reps: int = HSF_REPS) -> dict:
    if len(results) < planned:
        return {"verdict": "INCOMPLETE", "n_lengths_finished": len(results), "n_lengths_planned": planned}
    short = [{"L": r["L"], "mode": k, "n": v["n"]} for r in results for k, v in r["modes"].items()
             if k in HSF_MODES and v["n"] < reps]
    return {"verdict": "MEASURED" if not short else "MEASURED_PARTIAL", "short_cells": short,
            "per_length": results, "gather_rate": "NOT COVERED (no mapped-host UMBP gather code exists)"}


# ---------------------------------------------------------------- verdict over a records directory
def load_records(out_dir) -> list[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(str(out_dir), "records", "*.json"))):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        if r.get("kpf18_item") == ITEM and isinstance(r.get("s1r"), dict):
            recs.append({"record": p, **r})
    return recs


def _latest(recs: list[dict]) -> dict:
    """step id -> the latest finished record of that step (a step re-run after a crash between finish and marker)."""
    out: dict = {}
    for r in sorted(recs, key=lambda x: x.get("end_utc") or ""):
        out[r["s1r"]["step_id"]] = r
    return out


def fold_item3v2(item3v2_dir) -> dict:
    """KPF-18.0d(item 3, group 3) v2, read through its OWN rule module and its own predeclaration (never re-run here)."""
    if not item3v2_dir:
        return {"verdict": "NOT_READ", "why": "no --item3v2-dir"}
    d = Path(item3v2_dir)
    rule, pre = d / "item3v2_rule.py", d / "PREDECLARATION.md"
    if not rule.exists() or not pre.exists():
        return {"verdict": "NOT_READ", "why": f"missing {rule} or {pre}"}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("item3v2_rule_folded", rule)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        res = mod.verdict(str(d / "out"), sha256_path(pre))
    except Exception as exc:  # noqa: BLE001 - a fold-in failure is reported, never fatal
        return {"verdict": "NOT_READ", "why": f"{type(exc).__name__}: {exc}"[:300]}
    return {"verdict": res.get("verdict"), "valid_loads": res.get("valid_loads"), "needed_loads": res.get("needed_loads"),
            "class_counts": res.get("class_counts"), "predeclaration_sha256": res.get("predeclaration_sha256"),
            "dir": str(d)}


def verdict(out_dir, predecl_sha: str | None, *, item3v2_dir=None, n_cmp: int = 256) -> dict:
    allr = load_records(out_dir)
    done = [r for r in allr if r["s1r"].get("step_complete")]
    stale = [r["record"] for r in done if predecl_sha and r["s1r"].get("predeclaration_sha256") != predecl_sha]
    use = _latest([r for r in done if r["record"] not in stale])

    def res(step_id):
        r = use.get(step_id)
        return {**r["s1r"]["result"], "record": r["record"], "fresh_load": r["s1r"].get("fresh_load")} if r else None
    t512, t1000 = res("tail:tail512"), res("tail:tail1000")
    v_tail = verdict_tail({"result": t512, "fresh_load": (t512 or {}).get("fresh_load")} if t512 else None,
                          {"result": t1000} if t1000 else None)
    i2 = [res(f"item2:g{g}s{i}") for g, i in ITEM2_CELLS]
    v_item2 = verdict_item2([c for c in i2 if c], n_cmp)
    v_item2["folded_item3_v2"] = fold_item3v2(item3v2_dir)
    f3 = v_item2["folded_item3_v2"].get("verdict")
    v_item2["combined_reading"] = ("items 2 and 3 both read as placement/batch-shape numerics"
                                   if v_item2["verdict"] == "ITEM2_WITHIN_PLACEMENT_FLOOR" and f3 == "CLOSES_AS_NUMERICS"
                                   else f"item 2: {v_item2['verdict']}; item 3 v2: {f3}")
    lg = res("long:tf")
    v_long = lg if lg else {"verdict": "INCOMPLETE"}
    c191 = [res(f"kpf191a:g{g}") for g in KPF191_GROUPS]
    v_191 = verdict_191a([c for c in c191 if c])
    hs = [res(f"hsf:g{g}") for g in HSF_GROUPS]
    v_hsf = verdict_hsf([h for h in hs if h])
    items = {"item4_tailonly": v_tail, "item2_pcache_floor": v_item2, "item3_183p4_teacher_forced": v_long,
             "step6_kpf191a": v_191, "step7_hsf": v_hsf}
    complete = all(v.get("verdict") not in (None, "INCOMPLETE") for v in items.values())
    return {"rule": "PREDECLARATION.md (root 501f3a318, box 'KPF-18-S1 remainder + oracle repair')",
            "predeclaration_sha256": predecl_sha, "records_seen": len(allr), "records_step_complete": len(done),
            "excluded_predeclaration_mismatch": stale,
            "verdict": "COMPLETE" if complete else "INCOMPLETE",
            "item_verdicts": {k: v.get("verdict") for k, v in items.items()}, "items": items}


# ================================================================ steps (take the gate module as `gate`)
def _step_open(env, step_id: str, arm: str, key: str, **extra):
    ctx = env.ctx
    R = ARMS.open_run(env, ITEM, arm, rule=ARMS.RULES[ITEM][key], arm_key=key)
    R.extra["s1r"] = {"step_id": step_id, "step": key, "predeclaration": ctx.args.s1r_predeclaration,
                      "predeclaration_sha256": getattr(ctx, "_s1r_predecl_sha", None), "step_complete": False,
                      "fresh_load": getattr(env.srv, "_s1r_steps_done", 0) == 0, **extra}
    return R


def _step_close(env, R, result: dict) -> None:
    R.extra["s1r"].update(result=result, step_complete=True, finished_in_step=True)
    ARMS.close_counts(env, R)
    R.finish()
    env.srv._s1r_steps_done = getattr(env.srv, "_s1r_steps_done", 0) + 1


def _need(env, est_s: float) -> None:
    """Step boundary: the gate's own checks, the launcher STOP file (env KPF18_S1R_STOPFILE: a started step is finished,
    no new one starts), and the remaining budget against this step's estimate."""
    ctx = env.ctx
    ctx.check_boundary()
    sf = os.environ.get("KPF18_S1R_STOPFILE")
    if sf and os.path.exists(sf):
        raise env.gate.Abort(f"STOP file {sf} present: no new step")
    left = ctx.args.budget_min * 60 - (time.time() - ctx.t0)
    want = est_s * ctx.args.s1r_budget_factor
    if left < want:
        raise env.gate.Abort(f"step needs ~{want:.0f} s, {left:.0f} s of budget left: not started (a later grant resumes it)")


def _row(R, role: str, r: dict, **kw) -> dict:
    return R.row(role, r, **kw)


def run_tail(env, sub: str) -> None:
    gate, ctx = env.gate, env.ctx
    grp = gate.G(ctx, 0)
    T, trunk = grp["T"], grp["trunk"]
    tail = grp["src_tail"] if sub == "tail512" else grp["src_tail"] + grp["src_tail_long"][:TAIL_EXTRA]
    arm = f"A_tailonly_{sub}"
    R = _step_open(env, f"tail:{sub}", arm, "tail", sub=sub, tail_len=len(tail))
    env.erase_all()
    env.send(R, "source", trunk + tail, 0, n_predict=0, arm=arm, cell_id="g0.src", fork=False, ckpt=None, group=0)
    ck0 = ARMS.checkpoint_set(ARMS.slot_by_id(ctx.client.slots(), 0))
    for i in range(3):
        ctx.check_boundary()
        env.erase(1)
        env.send(R, "sibling", trunk + grp["suffixes"][i], 1, n_predict=1, arm=arm, cell_id=f"g0.s{i}", fork=True,
                 group=0, suffix=i)
    env.erase_all()
    _step_close(env, R, eval_tail(R.extra["cells"], T, ck0))


def run_item2_cell(env, g: int, i: int) -> None:
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    grp = gate.G(ctx, g)
    T, trunk, n, npb = grp["T"], grp["trunk"], a.n_predict, a.n_probs
    suf = grp["suffixes"][i]
    P = trunk + suf
    fil = ctx._s1r_fillers
    R = _step_open(env, f"item2:g{g}s{i}", f"item2_pcache_g{g}s{i}", "item2", group=g, suffix=i)
    cl = ctx.client
    # F and O: S1's item2_replicate, unchanged
    gate.item2_seed_source(ctx, R, g, T)
    cl.erase(1)
    F = _row(R, "fork_child", cl.complete(gate.body_for(P, n, 1, fork=True, n_probs=npb)), group=g, suffix=i, slot=1)
    O = _row(R, "source_own", cl.complete(gate.body_for(P, n, 0, fork=False, cache=True, n_probs=npb)), group=g, suffix=i, slot=0)
    # Q: the source's own continuation loaded onto ANOTHER slot through the --cache-ram prompt cache, no fork
    ctx.check_boundary()
    gate.erase_all(ctx)
    _row(R, "q_seed_source", cl.complete(gate.body_for(trunk + grp["src_tail"], 0, 3, fork=False, ckpt=[T], n_probs=0)),
         group=g, slot=3)
    ck3 = ARMS.checkpoint_set(ARMS.slot_by_id(cl.slots(), 3))
    ev = _row(R, "q_evict_filler", cl.stream(gate.body_for(fil[0], 1, 3, fork=False, n_probs=0)), slot=3)
    _row(R, "q_prime_filler", cl.complete(gate.body_for(fil[1], 1, 2, fork=False, n_probs=0)), slot=2)
    Q = _row(R, "pcache_own", cl.complete(gate.body_for(P, n, 2, fork=False, cache=True, n_probs=npb)), group=g, suffix=i, slot=2)
    gate.erase_all(ctx)
    bad = item2_cell_validity(F, O, Q, T=T, S=len(suf), n=n)
    fo, qo, fq = gate.compare(F, O, n), gate.compare(Q, O, n), gate.compare(F, Q, n)
    R.eq.setdefault("item2_fork_vs_own", []).append(fo)
    R.eq.setdefault("item2_pcache_vs_own", []).append(qo)
    R.eq.setdefault("item2_fork_vs_pcache", []).append(fq)
    tm = {k: {x: (r.get("timings") or {}).get(x) for x in ("prompt_n", "cache_n", "n_fork_tokens", "fork_src_kind",
                                                            "fork_src_slot", "draft_n", "draft_n_accepted")}
          for k, r in (("F", F), ("O", O), ("Q", Q))}
    _step_close(env, R, {"cell": f"g{g}.s{i}", "T": T, "S": len(suf), "valid": not bad, "invalid_reasons": bad,
                         "fork_vs_own": fo, "pcache_vs_own": qo, "fork_vs_pcache": fq, "timings": tm,
                         "q_source_checkpoints": ck3, "q_evict_stall_ms": stall_ms(ev)})


def tf_arm(env, R, name: str, slot: int, P: list[int], forced: list[int], first: dict) -> dict:
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    lps: list = []
    first_t: dict = {}
    stop_at = None
    for k in range(len(forced)):
        if k % 32 == 0:
            ctx.check_boundary()
        if k == 0:
            body = gate.body_for(P, 1, slot, fork=first["fork"], cache=first["cache"], dflash=False, ckpt=first.get("ckpt"),
                                 n_probs=a.n_probs)
        else:
            body = gate.body_for(P + forced[:k], 1, slot, fork=False, cache=True, dflash=False, n_probs=a.n_probs)
        body["logit_bias"] = [[int(forced[k]), LOGIT_BIAS]]
        r = R.row(f"tf_{name}", ctx.client.complete(body), slot=slot, pos=k)
        t = r.get("timings") or {}
        tok = (r.get("tokens") or [None])[0]
        lp = logprob_of_chosen(r["probs"][0]) if r.get("probs") else None
        ok = tok == forced[k] and lp is not None
        if k == 0:
            first_t = {x: t.get(x) for x in ("prompt_n", "cache_n", "n_fork_tokens", "fork_src_kind", "fork_src_slot",
                                             "prompt_ms")}
        else:
            ok = ok and t.get("prompt_n") == 1 and t.get("cache_n") == len(P) + k - 1 and not t.get("n_fork_tokens")
            if (t.get("prompt_n") or 0) > TF_FAILFAST_PROMPT_N:
                lps.append(None)
                stop_at = k
                break
        lps.append(lp if ok else None)
    lps += [None] * (len(forced) - len(lps))
    return {"lp": lps, "first": first_t, "failfast_at": stop_at}


def run_long_tf(env) -> None:
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    g = a.s1r_long_group
    grp = gate.G(ctx, g)
    T, trunk = grp["T"], grp["trunk"]
    P = trunk + grp["suffixes"][0]
    doc = ctx._s1r_forced
    want_p = (doc.get("prompt") or {}).get("ids_sha256")
    if sha256_ids(P) != want_p:
        raise gate.Abort(f"18.3.4 prompt sha256 {sha256_ids(P)} != the forced sequence's prompt {want_p}: refusing")
    forced = list(doc["ids"][:a.s1r_tf_positions])
    R = _step_open(env, "long:tf", "long_teacher_forced", "long", group=g, T=T, n_task=len(P),
                   forced_ids_sha256=doc.get("ids_sha256"), positions=len(forced))
    env.erase_all()
    env.send(R, "source", trunk + grp["src_tail"], 0, n_predict=0, arm="long_teacher_forced", cell_id="src", fork=False,
             ckpt=[T], group=g)
    per = {}
    for name, slot, first in (("child", 1, {"fork": True, "cache": True, "ckpt": None}),
                              ("coldA", 2, {"fork": False, "cache": False, "ckpt": [T]}),
                              ("coldB", 3, {"fork": False, "cache": False, "ckpt": [T]})):
        per[name] = tf_arm(env, R, name, slot, P, forced, first)
    env.erase_all()
    res = eval_long_tf({k: v["lp"] for k, v in per.items()}, child_first=per["child"]["first"], T=T, n_task=len(P))
    res["first"] = {k: v["first"] for k, v in per.items()}
    res["failfast_at"] = {k: v["failfast_at"] for k, v in per.items()}
    res["lp"] = {k: v["lp"] for k, v in per.items()}
    _step_close(env, R, res)


def build_legs(gate, fork_bin) -> dict:
    bd = Path(fork_bin).parent
    flag = gate.bytes_contain(bd / "libllama-common.so", b"--slot-fork-dft-full")
    mut = gate.bytes_contain(bd / "libllama-server-impl.so", b"LLAMA_KPF_MUTATE")
    return {"flag_off": "RUN (the build's only behaviour)",
            "flag_on": ("NOT_RUNNABLE_ON_THIS_BUILD (--slot-fork-dft-full absent from libllama-common.so)" if flag is False
                        else "NOT_RUN_THIS_LOAD (needs its own load with --slot-fork-dft-full)" if flag else "UNKNOWN (lib unreadable)"),
            "mutation_dft_no_transfer": ("NOT_RUNNABLE_ON_THIS_BUILD (LLAMA_KPF_MUTATE absent from libllama-server-impl.so)"
                                         if mut is False else "NOT_RUN_THIS_LOAD" if mut else "UNKNOWN (lib unreadable)"),
            "flag_compiled_in": flag, "mutation_hook_compiled_in": mut}


def run_191a_cell(env, g: int) -> None:
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    grp = gate.G(ctx, g)
    T, trunk = grp["T"], grp["trunk"]
    P = trunk + grp["suffixes"][0]
    arm = f"kpf191a_flagoff_g{g}"
    R = _step_open(env, f"kpf191a:g{g}", arm, "kpf191a", group=g, T=T)
    n = KPF191_N
    env.erase_all()
    env.send(R, "trunk", trunk, 0, n_predict=0, arm=arm, cell_id="trunk", fork=False, ckpt=[-1], group=g)
    ch, _ = env.send(R, "child", P, 1, n_predict=n, arm=arm, cell_id="child", fork=True, n_probs=a.n_probs, expected=T, group=g)
    fa, _ = env.send(R, "fresh", P, 2, n_predict=n, arm=arm, cell_id="fresh", fork=False, cache=False, n_probs=a.n_probs, group=g)
    fb, _ = env.send(R, "fresh_B", P, 3, n_predict=n, arm=arm, cell_id="fresh_B", fork=False, cache=False, n_probs=a.n_probs, group=g)
    env.erase(2, 3)
    env.send(R, "sameshape_trunk", trunk, 3, n_predict=0, arm=arm, cell_id="ss_trunk", fork=False, ckpt=[-1], group=g)
    ss, _ = env.send(R, "sameshape", P, 3, n_predict=n, arm=arm, cell_id="sameshape", fork=False, cache=True,
                     n_probs=a.n_probs, group=g)
    env.erase_all()
    res = eval_191a_cell(ch, fa, fb, ss, T=T, n=n)
    res["legs"] = ctx._s1r_legs
    _step_close(env, R, res)


def _slot_action(gate, ctx, slot: int, action: str, fn: str) -> dict:
    t0 = time.time()
    st, body = ctx.client._req("POST", f"/slots/{slot}?action={action}", {"filename": fn}, timeout=1800)
    wall = (time.time() - t0) * 1000.0
    if st != 200 or not isinstance(body, dict):
        raise gate.Abort(f"slot {action} {fn} on slot {slot}: HTTP {st} {str(body)[:200]}")
    return {**body, "wall_ms": round(wall, 3)}


def run_hsf(env, g: int) -> None:
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    grp = gate.G(ctx, g)
    L, trunk = grp["T"], grp["trunk"]
    reps = a.s1r_hsf_reps
    fil = ctx._s1r_fillers
    cl = ctx.client
    if len(fil) < 3 + 4 * reps:
        raise gate.Abort(f"HSF needs {3 + 4 * reps} fillers, have {len(fil)}")
    R = _step_open(env, f"hsf:g{g}", f"hsf_L{L}", "hsf", group=g, L=L)
    slot_dir = ctx.out / "slot-save" / env.srv.label
    slot_dir.mkdir(parents=True, exist_ok=True)
    shm = Path(a.s1r_shm_dir) / f"kpf18s1r-{os.getpid()}-{L}"
    rows: list[dict] = []
    made: list[Path] = []

    def add(mode, **kw):
        row = {"mode": mode, "L": L, **kw}
        rows.append(row)
        R.rows.add({"t": round(time.time(), 3), "run_id": R.run_id, "role": f"hsf_{mode}", **row})
        return row
    try:
        shm.mkdir(parents=True, exist_ok=True)
        made.append(shm)
        env.erase_all()
        _row(R, "hsf_prefill", cl.complete(gate.body_for(trunk, 0, 0, fork=False, n_probs=0)), slot=0)
        ck0 = (ARMS.slot_by_id(cl.slots(), 0) or {}).get("checkpoints") or []
        ck_bytes = int(sum(float(c.get("size_mib") or 0.0) for c in ck0) * 1024 * 1024)
        # slot file on tmpfs (symlink inside the server's slot-save dir; fs_validate_filename forbids a path)
        fn_t = f"hsf_tmpfs_{L}.bin"
        (slot_dir / fn_t).symlink_to(shm / fn_t)
        made.append(slot_dir / fn_t)
        n_written = None
        for k in range(reps):
            r = _slot_action(gate, ctx, 0, "save", fn_t)
            n_written = r.get("n_written")
            add("tmpfs_save", rep=k, ms=(r.get("timings") or {}).get("save_ms"), bytes=r.get("n_written"),
                n_tokens=r.get("n_saved"), wall_ms=r["wall_ms"], valid=r.get("n_saved") == L)
        for k in range(reps):
            cl.erase(1)
            r = _slot_action(gate, ctx, 1, "restore", fn_t)
            add("tmpfs_restore", rep=k, ms=(r.get("timings") or {}).get("restore_ms"), bytes=r.get("n_read"),
                n_tokens=r.get("n_restored"), wall_ms=r["wall_ms"], valid=r.get("n_restored") == L)
        v = cl.complete(gate.body_for(trunk + grp["suffixes"][0][:1], 1, 1, fork=False, cache=True, n_probs=0))
        _row(R, "hsf_restore_check", v, slot=1)
        restore_ok = (v.get("timings") or {}).get("cache_n") == L
        if not restore_ok:
            for row in rows:
                if row["mode"] == "tmpfs_restore":
                    row["valid"] = False
        add("restore_check", cache_n=(v.get("timings") or {}).get("cache_n"), valid=restore_ok)
        # slot file on /mnt/raid0 (the run's out dir): save + timed fsync, cold restore after fadvise DONTNEED
        fn_d = f"hsf_disk_{L}.bin"
        made.append(slot_dir / fn_d)
        for k in range(reps):
            r = _slot_action(gate, ctx, 0, "save", fn_d)
            fs = fsync_path(slot_dir / fn_d) * 1000.0
            add("disk_save", rep=k, ms=(r.get("timings") or {}).get("save_ms"), fsync_ms=round(fs, 3),
                bytes=r.get("n_written"), n_tokens=r.get("n_saved"), wall_ms=r["wall_ms"], valid=r.get("n_saved") == L)
        for k in range(reps):
            drop_pages(slot_dir / fn_d)
            res_b = page_residency(slot_dir / fn_d)
            cl.erase(1)
            r = _slot_action(gate, ctx, 1, "restore", fn_d)
            add("disk_restore_cold", rep=k, ms=(r.get("timings") or {}).get("restore_ms"), bytes=r.get("n_read"),
                n_tokens=r.get("n_restored"), residency_before=res_b, wall_ms=r["wall_ms"],
                valid=r.get("n_restored") == L and res_b is not None and res_b <= COLD_RESIDENCY_MAX and restore_ok)
        res_h = page_residency(slot_dir / fn_d)
        cl.erase(1)
        r = _slot_action(gate, ctx, 1, "restore", fn_d)
        add("disk_restore_hot", ms=(r.get("timings") or {}).get("restore_ms"), bytes=r.get("n_read"),
            residency_before=res_h, valid=r.get("n_restored") == L and restore_ok)
        # filler-sized state (the --cache-ram baseline's byte count)
        cl.erase(1)
        _row(R, "hsf_filler_state", cl.complete(gate.body_for(fil[2], 0, 1, fork=False, n_probs=0)), slot=1)
        fn_f = f"hsf_tmpfs_filler_{L}.bin"
        (slot_dir / fn_f).symlink_to(shm / fn_f)
        made.append(slot_dir / fn_f)
        filler_bytes = _slot_action(gate, ctx, 1, "save", fn_f).get("n_written")
        # --cache-ram: baseline = a filler evicting a filler-only slot (distinct fillers: never 'already cached')
        fi = 3
        for k in range(reps):
            cl.erase(1)
            _row(R, "hsf_base_fill", cl.complete(gate.body_for(fil[fi], 1, 1, fork=False, n_probs=0)), slot=1)
            b = _row(R, "hsf_base_evict", cl.stream(gate.body_for(fil[fi + 1], 1, 1, fork=False, n_probs=0)), slot=1)
            fi += 2
            add("cache_baseline", rep=k, ms=stall_ms(b), filler_bytes=filler_bytes, valid=stall_ms(b) is not None)
        cl.erase(1)
        # save then load: the entry moves slot 0 -> cache -> slot 2 -> cache -> slot 3 -> cache -> slot 1
        evict_slots, hit_slots = (0, 2, 3), (2, 3, 1)
        entry = list(trunk)
        for k in range(reps):
            es, hs = evict_slots[k % 3], hit_slots[k % 3]
            ckk = (ARMS.slot_by_id(cl.slots(), es) or {}).get("checkpoints") or []
            ckb = int(sum(float(c.get("size_mib") or 0.0) for c in ckk) * 1024 * 1024)
            ev = _row(R, "hsf_cache_evict", cl.stream(gate.body_for(fil[fi], 1, es, fork=False, n_probs=0)), slot=es)
            add("cache_save", rep=k, ms=stall_ms(ev), entry_bytes=(n_written or 0) + ckb, ckpt_bytes=ckb,
                valid=stall_ms(ev) is not None and n_written is not None)
            cl.erase(hs)
            _row(R, "hsf_cache_prime", cl.complete(gate.body_for(fil[fi + 1], 1, hs, fork=False, n_probs=0)), slot=hs)
            fi += 2
            extra = grp["suffixes"][0][HSF_EXTRA * k:HSF_EXTRA * (k + 1)]
            hit = _row(R, "hsf_cache_hit", cl.stream(gate.body_for(entry + extra, 1, hs, fork=False, cache=True, n_probs=0)),
                       slot=hs)
            th = hit.get("timings") or {}
            add("cache_load", rep=k, ms=stall_ms(hit), bytes=(n_written or 0) + ckb, cache_n=th.get("cache_n"),
                prompt_n=th.get("prompt_n"), entry_len=len(entry),
                valid=stall_ms(hit) is not None and th.get("cache_n") == len(entry) and th.get("prompt_n") == len(extra))
            entry = entry + extra
        env.erase_all()
    finally:
        for p in reversed(made):
            try:
                if p.is_symlink() or p.is_file():
                    tgt = p.resolve() if p.is_symlink() else None
                    p.unlink()
                    if tgt is not None and tgt.exists() and str(tgt).startswith(str(shm)):
                        tgt.unlink()
                elif p.is_dir():
                    for q in p.iterdir():
                        q.unlink()
                    p.rmdir()
            except OSError:
                pass
    res = eval_hsf(rows, L)
    res["checkpoints_slot0"] = [{"n_tokens": c.get("n_tokens"), "size_mib": c.get("size_mib"), "pinned": c.get("pinned")}
                                for c in ck0]
    res["checkpoint_bytes_slot0"] = ck_bytes
    res["rows"] = rows
    _step_close(env, R, res)


# ================================================================ plan
def _est_item2(gate, T: int, S: int, n: int) -> float:
    return (2 * gate.pf_s(T + gate.SRC_TAIL, 0) + 3 * gate.pf_s(S, 2 * T) + 3 * gate.dec_s(n, 2 * T, True)
            + 2 * gate.pf_s(FILLER_LEN, 0) + 10 * gate.REQ_OVERHEAD_S)


def body_185(ctx, gate, env) -> list:
    a = ctx.args
    Step = gate.Step
    want = [x.strip() for x in (a.s1r_steps or "").split(",") if x.strip()]
    steps: list = []

    def S_(g):
        return gate.G(ctx, g)
    if "tail" in want:
        T = S_(0)["T"]
        for sub in TAIL_SUBARMS:
            tl = 512 + (TAIL_EXTRA if sub == "tail1000" else 0)
            est = gate.pf_s(T + tl, 0) + sum(gate.pf_s(s + 1, T) for s in S_(0)["suffix_lens"][:3]) + 8 * gate.REQ_OVERHEAD_S

            def fn(sub=sub, est=est):
                _need(env, est)
                run_tail(env, sub)
            steps.append(Step(f"185 (4) A_tailonly {sub} g0 T={T}: erase all, source trunk+tail({tl}) no checkpoint_at, "
                              f"3 siblings on slot 1 (rule tail{'; FIRST step on the fresh load' if sub == 'tail512' else ''})",
                              est, fn))
    if "item2" in want:
        for g, i in ITEM2_CELLS:
            T, S = S_(g)["T"], S_(g)["suffix_lens"][i]
            est = _est_item2(gate, T, S, a.n_predict)

            def fn(g=g, i=i, est=est):
                _need(env, est)
                run_item2_cell(env, g, i)
            steps.append(Step(f"185 (2) item2 g{g}.s{i} T={T} S={S}: F fork / O own / Q own via --cache-ram onto another "
                              f"slot, {a.n_predict} tok each (rule item2)", est, fn))
    if "long" in want:
        g = a.s1r_long_group
        T, S0 = S_(g)["T"], S_(g)["suffix_lens"][0]
        K = a.s1r_tf_positions
        # cold A prefills beside the source + child (~T unique cells: the child shares the trunk), cold B beside ~2T
        est = (gate.pf_s(T + gate.SRC_TAIL, 0) + gate.pf_s(S0, T) + gate.pf_s(T + S0, T) + gate.pf_s(T + S0, 2 * T)
               + 3 * K * TF_POS_S + 8 * gate.REQ_OVERHEAD_S)

        def fn(est=est):
            _need(env, est)
            run_long_tf(env)
        steps.append(Step(f"185 (3) 18.3.4 teacher-forced g{g} P={T + S0}: child fork / coldA / coldB (checkpoint_at [T]), "
                          f"{K} positions x 3 arms, 1 request per position (rule long)", est, fn))
    if "kpf191a" in want:
        for g in KPF191_GROUPS:
            T, S0 = S_(g)["T"], S_(g)["suffix_lens"][0]
            est = (3 * gate.pf_s(T, 0) + 2 * gate.pf_s(T + S0, 2 * T) + 2 * gate.pf_s(S0, T)
                   + 4 * gate.dec_s(KPF191_N, T, True) + 8 * gate.REQ_OVERHEAD_S)

            def fn(g=g, est=est):
                _need(env, est)
                run_191a_cell(env, g)
            steps.append(Step(f"185 step6 KPF-19.1a flag off g{g} T={T}: child fork vs fresh vs sameshape, {KPF191_N} tok, "
                              f"draft_n/draft_n_accepted (rule kpf191a)", est, fn))
    if "hsf" in want:
        for g in HSF_GROUPS:
            L = S_(g)["T"]
            gb = L * 35e3 / GB + 0.16
            est = gate.pf_s(L, 0) + a.s1r_hsf_reps * (4 * gb / 2.0 + 4 * gb / 1.0) + 30 * gate.REQ_OVERHEAD_S

            def fn(g=g, est=est):
                _need(env, est)
                run_hsf(env, g)
            steps.append(Step(f"185 step7 HSF-4/5 L={L}: slot save/restore tmpfs + disk (fsync, cold fadvise) + --cache-ram "
                              f"save/load stall, x{a.s1r_hsf_reps} (rule hsf)", est, fn))
    return steps


def build_185(ctx, gate) -> list:
    a = ctx.args
    name, argv = ARMS.get_configs(ctx, gate)[0]
    srv = ARMS.server_cls(gate)(ctx, f"kpf18s1r-{name}-{gate.utc_tag()}", Path(a.fork_bin), a.fork_min, argv)
    env = ARMS.Env(ctx, gate, srv, name)
    _prepare(ctx, gate)
    return gate.start_steps(ctx, srv) + body_185(ctx, gate, env) + [gate.stop_step(srv)]


def _prepare(ctx, gate) -> list[str]:
    """Load the pinned inputs onto ctx (predeclaration sha, forced sequence, fillers, build legs); returns problems."""
    a = ctx.args
    probs = []
    if a.s1r_predeclaration and Path(a.s1r_predeclaration).exists():
        ctx._s1r_predecl_sha = sha256_path(a.s1r_predeclaration)
    else:
        ctx._s1r_predecl_sha = None
        probs.append(f"--s1r-predeclaration missing or unreadable: {a.s1r_predeclaration}")
    doc, fp = load_forced(a.s1r_forced_seq, a.s1r_forced_sha256)
    ctx._s1r_forced = doc or {}
    probs += fp
    ctx._s1r_fillers = choose_fillers(ctx.ps) if not ctx.ps.spec_only else []
    ctx._s1r_legs = build_legs(gate, a.fork_bin) if a.fork_bin else {}
    return probs


def preflight_185(ctx, gate) -> tuple[list[str], list[str]]:
    a = ctx.args
    probs = _prepare(ctx, gate)
    notes = [f"S1R predeclaration {a.s1r_predeclaration} sha256 {ctx._s1r_predecl_sha}",
             f"S1R forced sequence {a.s1r_forced_seq} ids sha256 {(ctx._s1r_forced or {}).get('ids_sha256')} "
             f"(predeclared {a.s1r_forced_sha256})",
             f"S1R steps {a.s1r_steps}; item2 cells {list(ITEM2_CELLS)}; 19.1a groups {list(KPF191_GROUPS)}; HSF groups "
             f"{list(HSF_GROUPS)} x{a.s1r_hsf_reps}; teacher-forced positions {a.s1r_tf_positions}",
             f"S1R KPF-19.1a legs on {a.fork_bin}: {ctx._s1r_legs}"]
    if not ctx.ps.spec_only:
        g = a.s1r_long_group
        P = ctx.ps.groups[g]["trunk"] + ctx.ps.groups[g]["suffixes"][0]
        want = (ctx._s1r_forced or {}).get("prompt", {}).get("ids_sha256")
        if sha256_ids(P) != want:
            probs.append(f"18.3.4 prompt (g{g} trunk+suffix0) sha256 {sha256_ids(P)} != forced sequence prompt {want}")
        else:
            notes.append(f"S1R 18.3.4 prompt g{g} = {len(P)} tokens, sha256 matches the forced sequence's prompt")
        if len(ctx._s1r_fillers) < N_FILLERS:
            probs.append(f"only {len(ctx._s1r_fillers)} fillers with a zero-LCP first token found (need {N_FILLERS})")
        if len(ctx._s1r_forced.get("ids") or []) < a.s1r_tf_positions:
            probs.append("forced sequence shorter than --s1r-tf-positions")
    else:
        notes.append("S1R: spec-only prompt set (plan only): prompt sha, fillers and forced length are checked at execute")
    if not Path(a.s1r_shm_dir).is_dir():
        probs.append(f"--s1r-shm-dir {a.s1r_shm_dir} is not a directory")
    bad = [x for x in (a.s1r_steps or "").split(",") if x.strip() and x.strip() not in STEP_KEYS]
    if bad:
        probs.append(f"--s1r-steps: unknown {bad}")
    return probs, notes


ARMS.BUILDERS[ITEM] = build_185
ARMS.PREFLIGHTS[ITEM] = preflight_185


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="verdict over a kpf18s1r --out directory (pure; starts nothing)")
    ap.add_argument("verdict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--predeclaration", required=True)
    ap.add_argument("--item3v2-dir", default=None)
    ap.add_argument("--write", default=None)
    x = ap.parse_args()
    v = verdict(x.out, sha256_path(x.predeclaration), item3v2_dir=x.item3v2_dir)
    s = json.dumps(v, indent=1, default=str)
    if x.write:
        tmp = x.write + ".tmp"
        Path(tmp).write_text(s + "\n")
        os.replace(tmp, x.write)
    print(json.dumps({"verdict": v["verdict"], "item_verdicts": v["item_verdicts"], "records_seen": v["records_seen"],
                      "records_step_complete": v["records_step_complete"],
                      "excluded_predeclaration_mismatch": len(v["excluded_predeclaration_mismatch"])}))
