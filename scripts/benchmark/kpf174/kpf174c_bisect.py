#!/usr/bin/env python3
"""KPF-17.4c driver: bisect what the GPU TARGET restore carries at the c1 junction (speculation OFF), one GPU load.

Handoff: handoffs/active/kv-prefix-fork-and-paged-attention.md, row KPF-17.4c (filed 2026-10-23 by the KPF-17.4b read-out:
"With the drafter off, g2 diverges at c = 2148 token 56 and g1 does not. Bisect what the target restore carries (KV vs
recurrent/SWA state vs position bookkeeping) on the same image gpu-20261023-82c16126f-kpf191, argv and shim, K=3,
predeclared before compute"). Written 2026-10-24 (workspace-f8). Class: correctness (bit identity), no timing claim.

WHAT THE TARGET RESTORE CARRIES (code reading, fork commit 82c16126f; Qwen3.8-27B is `qwen35`, hybrid GDN, n_swa 0)
  do_slot_fork (tools/server/server-context.cpp:1902), checkpoint kind:
    attention KV  [0, p): common_context_seq_cp_attn -> llama_kv_cache::seq_cp: the dst seq bit is ADDED to the source's
                  cells (zero copy: the same cells, the same bytes).
    recurrent     ckpt->load_tgt(PARTIAL_ONLY) -> llama_memory_hybrid::state_read skips the attention half and
                  llama_memory_recurrent::state_read writes the checkpoint's r/s rows into a NEW cell of dst.
    position      dst.prompt.tokens = task.tokens[0, p), checkpoints with n_tokens <= p cloned; pre_decode then takes
                  n_past = p (recurrent pos = p - 1 < pos_next = p: no second restore).
    SWA state     none on the target (n_swa 0: no SWA cache; KPF-17.4a), so "recurrent/SWA" is the recurrent half only.
  An own-slot checkpoint restore (pre_decode, server-context.cpp:4470-4510) uses the SAME load_tgt on the slot's own cells.

THE CONFOUND THIS DESIGN REMOVES (why 17.4b's X != C is not yet a restore defect)
  The 17.4b control C seeds trunk[:c] on slot 2 WHILE the source holds 14336 cells, so C's cells sit at index 14336+ and
  were computed at a larger n_kv than the source's cells [0, c) (written on an empty cache). The fork child X attends
  the source's cells at index 0..c-1. So X and C differ in KV/recurrent CONTENT bits and in query-time LAYOUT (cell
  indices, n_kv = pad256(used_max_p1)), not only in "fork vs no fork". In the 17.4b spec-off cells X differs from C in
  the logprob of EVERY generated token from token 0 (|d| ~1e-4 nats), like the cold run F does, while C1 == C2 == C3
  bit for bit: a numerics signature, not (yet) a state defect.

  This driver builds no-fork references on the SAME layout as each fork/restore arm, so with no defect every matched
  pair is bit-identical (same KV bytes, same recurrent bytes, same cell indices, same n_kv, same batch partition):
    Xc     the 17.4b fork child (source built trunk[:2148], [:4096], [:10240], [:14336], pinned checkpoint at c)
    Cmask  no fork: slot 2 seeds trunk[:c] on an EMPTY cache (cells 0..c-1, the source's bytes), then slot 3 seeds a
           filler of L - c tokens (cells c..L-1, masked for slot 2), then P: the suffix lands at L, as Xc's does
    Cf     the 17.4b control (seeded while the source is resident): links this run to the 17.4b result
    Ks     no fork, checkpoint restore: slot 0 seeds trunk[:c] then extends to c + min(2048, c // 2), then P rewinds onto
           its own pinned checkpoint at c (recurrent rows from load_tgt, own cells 0..c-1, suffix at c)
    C0     no fork, no restore: slot 2 seeds trunk[:c] on an empty cache, then P (suffix at c)
    Xe     fork from a source holding exactly trunk[:c] (kind "end" or "checkpoint": both carry the same bytes)
  Matched comparisons (the bisect): Xc~Cmask (fork, L-occupied layout), Xe~C0 (fork, c-only layout), Ks~C0 (restore, no
  fork). Layout comparisons (descriptive): Cmask~C0 (masked occupancy only), Cf~Cmask (the 17.4b control's layout and
  content). Reproduction: Xc~Cf (the 17.4b comparison itself).

RULES: RULES below (sha256 recorded in rules.json and run_meta.json BEFORE the launch). Every verdict is a pure function of
the stored cells (`analyze`). PROCESS DISCIPLINE is the KPF-13.1 driver's (own child by captured PID, port ownership by
socket inode, KFD residency, ggml linkage, /proc read-back, hard deadline + watchdog), imported through kpf174.

Exit codes: 0 ok, 2 refused (preflight), 3 abort, 4 error, 5 deadline/stop drain, 75 claim held. stdlib only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("kpf174_swa_drafter", HERE / "kpf174_swa_drafter.py")
k174 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(k174)
k13 = k174.k13

TASK = "KPF-17.4c"
SCHEMA = k13.SCHEMA
N_GEN = 64
SEED = k13.SEED
LAUNCH = "bisect_specoff"
CONFIG_NAME = "kpf174c_bisect_specoff"
k13.CONFIGS.setdefault(CONFIG_NAME, {"label": "kpf174c_bisect_spec_off", "mutate": None, "extra": [], "arms": ("A",),
                                     "knob": [b"--slot-fork-min-tokens"], "plain_only": False, "spec_off": True})
DEFAULT_ARGV = k174.DEFAULT_ARGV
DEFAULT_PROMPT_SET = k174.DEFAULT_PROMPT_SET
DEFAULT_JUNCTIONS = (2148, 4096)          # c1 = n_swa + 100 (4 x 512 + 100, misaligned) and the aligned v1 build point
DEFAULT_BUILD = (2148, 4096, 10240)       # the 17.4 v1 / 17.4b honest_specoff source build points; + L
DEFAULT_L = 14336
REF_REPEATS = 3                           # K = 3 same-shape references
ARM_REPEATS = 2
NONCONSEC = "non-consecutive token position"   # llama_memory_recurrent::find_slot LLAMA_LOG_WARN (default log level)
F_KEEP_MIN = 0.5                          # get_available_slot: f_keep < 0.5 on a non-empty slot -> --cache-ram save/load

ARMS = {
    "Xc": {"role": "fork", "phase": "S", "repeats": ARM_REPEATS,
           "desc": "fork child on slot 1 from the 17.4b source (slot 0 built trunk[:b] for b in build + [L], each "
                   "checkpoint_at [-1]); the 17.4b X"},
    "Cf": {"role": "ref", "phase": "S", "repeats": REF_REPEATS,
           "desc": "17.4b control: slot 2 replays the build points <= c while the source is resident, then P"},
    "Ks": {"role": "restore", "phase": "K", "repeats": ARM_REPEATS,
           "desc": "no fork: slot 0 seeds the build points <= c, extends to c + min(2048, c // 2), then P restores its "
                   "own pinned checkpoint at c (empty cache otherwise)"},
    "Cmask": {"role": "ref", "phase": "M", "repeats": REF_REPEATS,
              "desc": "no fork: slot 2 replays the build points <= c on an empty cache, slot 3 seeds a filler of L - c "
                      "tokens, then P on slot 2 (Xc's layout: relevant cells 0..c-1, masked c..L-1, suffix at L)"},
    "C0": {"role": "ref", "phase": "E", "repeats": REF_REPEATS,
           "desc": "no fork, no restore: slot 2 replays the build points <= c on an empty cache, then P"},
    "Xe": {"role": "fork", "phase": "E", "repeats": ARM_REPEATS,
           "desc": "fork child on slot 1 from a source (slot 0) that replayed exactly the build points <= c on an "
                   "empty cache (C0's layout)"},
}
COMPARISONS = {
    "Xc~Cmask": {"a": "Xc", "b": "Cmask", "kind": "matched",
                 "tests": "the fork (shared cells + recurrent rows from the source's stored pinned checkpoint) at the "
                          "17.4b layout"},
    "Xe~C0": {"a": "Xe", "b": "C0", "kind": "matched",
              "tests": "the fork from a source holding exactly [0, c) at the c-only layout"},
    "Ks~C0": {"a": "Ks", "b": "C0", "kind": "matched",
              "tests": "the checkpoint restore of the recurrent rows (load_tgt) with own cells, no fork"},
    "Cmask~C0": {"a": "Cmask", "b": "C0", "kind": "layout",
                 "tests": "masked occupancy and n_kv only (same bytes, same relevant-cell indices)"},
    "Cf~Cmask": {"a": "Cf", "b": "Cmask", "kind": "layout",
                 "tests": "the 17.4b control's layout and content (seeded while the source is resident)"},
    "Xc~Cf": {"a": "Xc", "b": "Cf", "kind": "reproduce", "tests": "the 17.4b comparison itself"},
}
MATCHED = tuple(k for k, v in COMPARISONS.items() if v["kind"] == "matched")

RULES = {
    "schema": "epyc.kpf174c.rules.v1",
    "task": ("KPF-17.4c: localize the GPU target-side restore defect at c1 (KPF-17.4b honest_specoff: g2-c2148 X != C "
             "from token 56, g1 token-identical; kpf174 rules v2 sha 2be6f9b0...). Bisect KV vs recurrent vs position "
             "bookkeeping. Speculation OFF, image gpu-20261023-82c16126f-kpf191, the 17.4b argv file and shim"),
    "declared": "before the launch; every verdict is a pure function of the stored cells (analyze)",
    "sampling": {"temperature": 0.0, "seed": SEED, "ignore_eos": True, "n_generated": N_GEN, "n_probs": 2},
    "geometry": {"groups": [1, 2], "junctions": list(DEFAULT_JUNCTIONS), "build_points": list(DEFAULT_BUILD),
                 "L": DEFAULT_L, "S": 64, "ks_extension": "c + min(2048, c // 2)",
                 "filler": "the prompt set's filler ids of the group, first L - c tokens"},
    "carried_by_the_restore": ("attention KV [0, p) shared by seq bit (seq_cp_attn); recurrent r/s rows from the "
                               "checkpoint (load_tgt PARTIAL_ONLY into a new dst cell); n_past = p and the cloned "
                               "checkpoints. No SWA state on qwen35 (n_swa 0); no drafter (speculation OFF)"),
    "arms": {k: v["desc"] for k, v in ARMS.items()},
    "repeats": {k: v["repeats"] for k, v in ARMS.items()},
    "cell_order": ("per cell (g, c): phase S (erase all; source build; Xc1 Cf1 Xc2 Cf2 Cf3; slots 1-3 erased around each), "
                   "phase K (twice: erase all; Ks source; Ks), phase M (three times: erase all; Cmask), phase E (three "
                   "times: erase all; C0; then erase all; Xe source; Xe1; Xe2). Cells: c = 2148 for g1, g2 first, then "
                   "c = 4096 for g1, g2"),
    "observable": ("per run: the 64 greedy token ids and the 64 x top-2 (id, logprob) pairs as returned; signature = "
                   "both, compared EXACTLY (bit identity). Descriptive: first token divergence, first logprob "
                   "divergence, max |d chosen-token logprob| over the agreeing prefix, |d| at token 0, the reference's "
                   "top-2 margin at the token divergence"),
    "validity": {
        "launch": ["speculation OFF: the /proc cmdline carries no drafter flag and every run reports draft_n 0 or none",
                   "HIP graphs DISABLED, read back from the server's /proc/<pid>/environ (GGML_CUDA_DISABLE_GRAPHS set), "
                   "as every 17.4 / 17.4b server", "HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 read back from the same environ",
                   "kv_unified true, fork mode checkpoint, cache_idle_slots 0, ggml linkage PASS (kpf13 read-back)",
                   "own-pid KFD VRAM > 0 after healthy (kpf13) and in both per-cell samples"],
        "run": ["HTTP 200 with 64 tokens and 64 top-2 entries", "draft_n 0 or none",
                "reference (Cf, Cmask, C0): n_fork_tokens 0, cache_n == c, prompt_n == S, no 'non-consecutive token "
                "position' warning in the server log during the run",
                "fork (Xc, Xe): n_fork_tokens != 0 (0 = the fork was not taken: INVALID) and fork_src_slot == 0",
                "restore (Ks): n_fork_tokens 0"],
        "cell": ["the source's /slots list holds a pinned checkpoint with n_tokens == c and pos_max == c - 1 before Xc "
                 "(phase S), before each Ks (phase K) and before Xe (phase E)",
                 "two own-pid KFD VRAM samples > 0 taken during the cell (after the first suffix request and after the "
                 "last)", "every arm has its repeats (references 3, arms 2)"],
        "stability": ("a group's repeats must be bit-identical among themselves; a comparison is judged only between "
                      "two valid, stable groups of one valid cell (else it is unjudged, never a pass)"),
        "cache_ram": ("the protocol never sends a request whose slot is non-empty with lcp / held < 0.5 (the server "
                      "would save to and could load from --cache-ram, which copies cells to new indices); the runner "
                      "aborts on such a request (protocol bug), so no arm is served from the RAM prompt cache"),
    },
    "position_evidence": ("for a fork or restore run: n_fork_tokens not in {0, c}, or cache_n != c, or prompt_n != S, or "
                          "a 'non-consecutive token position' warning in the server log between its send and its end"),
    "decisions": {
        "locus": ("first that applies: position iff position evidence in any fork/restore run; recurrent iff Ks~C0 "
                  "differs in at least one judged cell; kv iff Ks~C0 is identical in every judged cell and, in a cell "
                  "where Ks~C0 is judged identical, Xc~Cmask or Xe~C0 differs; none_reproduced iff decision_grade and "
                  "all three matched comparisons are identical in every planned cell; else undetermined. position, "
                  "recurrent and kv are existence claims (one cell suffices); none_reproduced is universal and needs "
                  "decision_grade"),
        "decision_grade": ("every planned cell attempted and valid, with all three matched comparisons judged"),
        "explains_174b": ("control_layout_numerics iff locus none_reproduced and Xc~Cf differs in at least one judged "
                          "cell (the fork child equals a no-fork control built on its own layout; the 17.4b control "
                          "differed because it was built on another); not_reproduced iff locus none_reproduced and "
                          "Xc~Cf is identical in every judged cell; not_explained iff Xc~Cmask differs in any judged "
                          "cell; else undetermined"),
        "layout_sensitivity": "descriptive: per-cell identical/differs of Cmask~C0 and Cf~Cmask",
        "alignment": ("descriptive: for each matched comparison, the junctions where it differs (2148 = 4 x 512 + 100, "
                      "4096 = 8 x 512)"),
        "not_decided_here": ("KPF-17.4, KPF-19.1a, pinned_junction_on_swa_drafter_allowed and the KPF-13.1b verdict keep "
                             "their verdicts; 17.4c only localizes"),
    },
    "all_attempts_counted": ("every completed cell attempt is judged and counted (INVALID included); a cell interrupted "
                             "by the deadline or the stop file is re-run from its start on resume and its partial rows "
                             "stay in rows/ (counted in the record, never judged)"),
}


def rules_sha256() -> str:
    return k13.sha256_json(RULES)


# ---------------------------------------------------------------- geometry (pure)
def ks_extension(c: int) -> int:
    return c + min(2048, c // 2)


def build_upto(build: list[int], c: int) -> list[int]:
    return [b for b in sorted(build) if b <= c]


def geometry(junctions: list[int], build: list[int], L: int) -> dict:
    return {"c": sorted(set(junctions)), "build": sorted(set(build)), "L": L,
            "ks_ext": {c: ks_extension(c) for c in sorted(set(junctions))}}


def geometry_problems(geo: dict, S: int, fork_min: int) -> list[str]:
    p = []
    for c in geo["c"]:
        if c not in geo["build"]:
            p.append(f"junction {c} is not a build point {geo['build']} (each junction must be a pinned prompt end)")
        ext = geo["ks_ext"][c]
        if c / ext < F_KEEP_MIN + 0.1:
            p.append(f"Ks extension {ext} for c {c}: f_keep {c / ext:.2f} too close to {F_KEEP_MIN} (--cache-ram path)")
        if c <= fork_min:
            p.append(f"junction {c} <= --fork-min {fork_min}")
    if geo["build"] and max(geo["build"]) >= geo["L"]:
        p.append(f"build point {max(geo['build'])} >= L {geo['L']}")
    if S < 8:
        p.append(f"S {S} < 8")
    return p


def planned_cells(groups: list[int], junctions: list[int]) -> list[tuple[int, int]]:
    """The cell order: the c1 junction (smallest) for every group first, then the next junction, and so on."""
    return [(g, c) for c in sorted(set(junctions)) for g in groups]


def cell_id(g: int, c: int) -> str:
    return f"g{g}-c{c}"


# ---------------------------------------------------------------- observables (pure)
def top2(entry: dict) -> list:
    top = entry.get("top_logprobs") or entry.get("top_probs") or []
    out = []
    for t in top[:2]:
        if "logprob" in t:
            out.append([int(t["id"]), float(t["logprob"])])
        elif "prob" in t:
            out.append([int(t["id"]), float(t["prob"])])
    return out


def top2_kind(probs: list[dict]) -> str | None:
    for e in probs or []:
        for t in (e.get("top_logprobs") or e.get("top_probs") or [])[:1]:
            return "logprob" if "logprob" in t else "prob" if "prob" in t else None
    return None


def as_logprob(v: float, kind: str | None) -> float:
    if kind == "prob":
        return math.log(v) if v > 0 else float("-inf")
    return v


def signature(run: dict, n: int = N_GEN) -> str:
    return json.dumps([list(run.get("tokens") or [])[:n], list(run.get("top2") or [])[:n]], separators=(",", ":"))


def first_diff(a: list, b: list, n: int) -> int | None:
    for i in range(n):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            return i
    return None


def compare(a: dict, b: dict, n: int = N_GEN) -> dict:
    """a, b: one representative run of each (stable) group."""
    ta, tb = list(a.get("tokens") or [])[:n], list(b.get("tokens") or [])[:n]
    pa, pb = list(a.get("top2") or [])[:n], list(b.get("top2") or [])[:n]
    ka, kb = a.get("top2_kind"), b.get("top2_kind")
    dt = first_diff(ta, tb, n)
    dl = first_diff(pa, pb, n)
    upto = dt if dt is not None else n
    d = [abs(as_logprob(pa[i][0][1], ka) - as_logprob(pb[i][0][1], kb)) for i in range(min(upto, len(pa), len(pb)))
         if pa[i] and pb[i]]
    margin = None
    if dt is not None and dt < len(pb) and len(pb[dt]) >= 2:
        margin = as_logprob(pb[dt][0][1], kb) - as_logprob(pb[dt][1][1], kb)
    return {"identical": signature(a, n) == signature(b, n), "first_token_divergence": dt, "first_logprob_divergence": dl,
            "max_abs_dlogprob_prefix": max(d) if d else None, "abs_dlogprob_token0": d[0] if d else None,
            "ref_margin_at_divergence": margin, "n_prefix_compared": len(d)}


def run_problems(run: dict | None, role: str, c: int, S: int, n: int = N_GEN) -> list[str]:
    if run is None:
        return ["missing"]
    p = []
    if run.get("http") != 200 or run.get("error"):
        p.append(f"HTTP {run.get('http')} {run.get('error')}")
    if len(run.get("tokens") or []) < n:
        p.append(f"{len(run.get('tokens') or [])} tokens < {n}")
    t2 = run.get("top2") or []
    if len(t2) < n or any(len(x) < 2 for x in t2[:n]):
        p.append("top-2 logprobs missing")
    if run.get("draft_n"):
        p.append(f"draft_n {run.get('draft_n')} in a speculation-OFF launch")
    nf = run.get("n_fork_tokens") or 0
    if role == "ref":
        if nf:
            p.append(f"n_fork_tokens {nf} on a no-fork reference")
        if run.get("cache_n") != c or run.get("prompt_n") != S:
            p.append(f"partition cache_n {run.get('cache_n')} prompt_n {run.get('prompt_n')} != ({c}, {S})")
        if run.get("nonconsec_lines"):
            p.append(f"{run.get('nonconsec_lines')} '{NONCONSEC}' warning(s) on a no-fork reference")
    elif role == "fork":
        if not nf:
            p.append("n_fork_tokens 0: the fork was not taken")
        elif run.get("fork_src_slot") != 0:
            p.append(f"fork_src_slot {run.get('fork_src_slot')} != 0")
    elif role == "restore":
        if nf:
            p.append(f"n_fork_tokens {nf} on the restore arm (a fork was taken)")
    return p


def position_evidence(run: dict | None, role: str, c: int, S: int) -> list[str]:
    if run is None or role not in ("fork", "restore"):
        return []
    e = []
    nf = run.get("n_fork_tokens") or 0
    if role == "fork" and nf not in (0, c):
        e.append(f"n_fork_tokens {nf} != {c}")
    if run.get("cache_n") != c:
        e.append(f"cache_n {run.get('cache_n')} != {c}")
    if run.get("prompt_n") != S:
        e.append(f"prompt_n {run.get('prompt_n')} != {S}")
    if run.get("nonconsec_lines"):
        e.append(f"{run.get('nonconsec_lines')} '{NONCONSEC}' warning(s) in the server log during the run")
    return e


def pinned_ok(ckpts: list[dict] | None, c: int) -> bool:
    return any(ck.get("pinned") and ck.get("n_tokens") == c and ck.get("pos_max") == c - 1 for ck in (ckpts or []))


def judge_cell(cell: dict, n: int = N_GEN) -> dict:
    c, S = cell["c"], cell["S"]
    groups, pos_ev = {}, []
    for arm, spec in ARMS.items():
        runs = (cell.get("runs") or {}).get(arm) or []
        probs = [f"{arm}{i + 1}: {x}" for i, r in enumerate(runs) for x in run_problems(r, spec["role"], c, S, n)]
        if len(runs) < spec["repeats"]:
            probs.append(f"{arm}: {len(runs)} runs < {spec['repeats']}")
        ev = [f"{arm}{i + 1}: {x}" for i, r in enumerate(runs) for x in position_evidence(r, spec["role"], c, S)]
        pos_ev += ev
        sigs = {signature(r, n) for r in runs if r}
        groups[arm] = {"valid": not probs, "problems": probs, "n_runs": len(runs), "n_signatures": len(sigs),
                       "stable": (len(sigs) == 1) if not probs else None, "position_evidence": ev}
    cell_probs = []
    for ph in ("S", "K1", "K2", "E"):
        if not pinned_ok((cell.get("source_checkpoints") or {}).get(ph), c):
            cell_probs.append(f"phase {ph}: the source holds no pinned checkpoint with n_tokens {c}, pos_max {c - 1}")
    if cell.get("device", "gpu") == "gpu":
        vs = cell.get("vram_own_pid_bytes") or []
        if len(vs) < 2 or any(not v or v <= 0 for v in vs):
            cell_probs.append(f"own-pid KFD VRAM samples {vs}: residency not proven during the cell")
    comps = {}
    for name, cp in COMPARISONS.items():
        ga, gb = groups[cp["a"]], groups[cp["b"]]
        why = []
        if cell_probs:
            why.append("cell invalid")
        for arm, g in ((cp["a"], ga), (cp["b"], gb)):
            if not g["valid"]:
                why.append(f"{arm} invalid")
            elif not g["stable"]:
                why.append(f"{arm} unstable ({g['n_signatures']} signatures over {g['n_runs']} repeats)")
        if why:
            comps[name] = {"judged": False, "why": why}
        else:
            comps[name] = {"judged": True, **compare(cell["runs"][cp["a"]][0], cell["runs"][cp["b"]][0], n)}
    valid = not cell_probs and all(g["valid"] for g in groups.values())
    return {"cell": cell["cell"], "group": cell["group"], "c": c, "valid": valid, "cell_problems": cell_probs,
            "groups": groups, "comparisons": comps, "position_evidence": pos_ev}


def decide(judged: dict[str, dict], planned: list[str]) -> dict:
    """RULES['decisions'] (pure). judged: cell id -> judge_cell(); planned: the cell ids the run plans."""
    def split(name: str) -> dict:
        out = {"identical": [], "differs": [], "unjudged": []}
        for cid in planned:
            j = judged.get(cid)
            cp = (j or {}).get("comparisons", {}).get(name) or {}
            if not cp.get("judged"):
                out["unjudged"].append(cid)
            else:
                out["identical" if cp["identical"] else "differs"].append(cid)
        return out
    m = {name: split(name) for name in COMPARISONS}
    attempted = [cid for cid in planned if cid in judged]
    grade = (len(attempted) == len(planned) and all(judged[c]["valid"] for c in attempted)
             and all(not m[k]["unjudged"] for k in MATCHED))
    pos = {cid: judged[cid]["position_evidence"] for cid in attempted if judged[cid]["position_evidence"]}
    ks_same = set(m["Ks~C0"]["identical"])
    fork_diff = sorted({cid for k in ("Xc~Cmask", "Xe~C0") for cid in m[k]["differs"] if cid in ks_same})
    if pos:
        locus = "position"
    elif m["Ks~C0"]["differs"]:
        locus = "recurrent"
    elif fork_diff:
        locus = "kv"
    elif grade and all(len(m[k]["identical"]) == len(planned) for k in MATCHED):
        locus = "none_reproduced"
    else:
        locus = "undetermined"
    if locus == "none_reproduced":
        if m["Xc~Cf"]["differs"]:
            expl = "control_layout_numerics"
        elif m["Xc~Cf"]["identical"] and not m["Xc~Cf"]["unjudged"]:
            expl = "not_reproduced"
        else:
            expl = "undetermined"
    elif m["Xc~Cmask"]["differs"]:
        expl = "not_explained"
    else:
        expl = "undetermined"
    junction = {cid: judged[cid]["c"] for cid in attempted}
    align = {k: sorted({junction[cid] for cid in m[k]["differs"]}) for k in MATCHED}
    return {"locus": locus, "decision_grade": grade, "explains_174b": expl,
            "position_evidence": pos, "fork_differs_where_restore_identical": fork_diff,
            "matched": {k: m[k] for k in MATCHED},
            "layout_sensitivity": {k: m[k] for k in ("Cmask~C0", "Cf~Cmask")},
            "reproduces_174b": m["Xc~Cf"], "alignment": align,
            "cells_planned": len(planned), "cells_attempted": len(attempted),
            "cells_valid": sum(1 for c in attempted if judged[c]["valid"]),
            "complete": len(attempted) == len(planned),
            "not_decided_here": RULES["decisions"]["not_decided_here"]}


def analyze(results: dict[str, dict], planned: list[str], n: int = N_GEN) -> dict:
    judged = {cid: judge_cell(c, n) for cid, c in sorted(results.items())}
    return {"rules_sha256": rules_sha256(), "rules": RULES["schema"], "cells": judged, "decisions": decide(judged, planned)}


# ---------------------------------------------------------------- prompt source (the KPF-18 27B prompt set + fillers)
class Prompts(k174.Prompts):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.fillers: dict[int, list[int]] = {}
        for ln in self.path.read_text().splitlines():
            r = json.loads(ln)
            if r.get("kind") == "filler":
                self.fillers[r["group"]] = r["ids"]

    def problems_174c(self, groups: list[int], L: int, S: int, cs: list[int], model: str | None) -> list[str]:
        p = self.problems(groups, L, S, model)
        for g in groups:
            if len(self.fillers.get(g, [])) < L - min(cs):
                p.append(f"group {g}: filler {len(self.fillers.get(g, []))} < L - c {L - min(cs)}")
        return p


# ---------------------------------------------------------------- the run
class Runner:
    def __init__(self, ctx, geo: dict, prompts: Prompts, results: dict) -> None:
        self.ctx, self.geo, self.ps, self.results = ctx, geo, prompts, results
        self.srv = None
        self._rows = None
        self.vram_file = None
        self._vstop = threading.Event()
        self.shadow: dict[int, list[int]] = {}

    @property
    def rows(self):
        if self._rows is None:
            self._rows = k13.Jsonl(self.ctx.out / "rows" / f"{LAUNCH}.jsonl")
        return self._rows

    def pending(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(g, c) for g, c in cells if cell_id(g, c) not in self.results]

    # -- server
    def start(self) -> None:
        self.srv = k13.Server(self.ctx, CONFIG_NAME, Path(self.ctx.args.fork_bin))
        self.srv.start()   # GPU: k13 waits <= --vram-drain-timeout-s for the card to drain, then the empty-card checks
        rb = getattr(self.srv, "readback", None) or {}
        if self.ctx.args.device == "gpu" and getattr(self.srv, "proc", None) is not None:
            if rb.get("hip_graphs") != "disabled":
                raise k13.Abort(f"HIP graphs {rb.get('hip_graphs')!r} on the server (/proc environ): the predeclared "
                                "condition is DISABLED (the gpu-leak-1 shim), as every 17.4 / 17.4b server")
            if rb.get("env_HSA_OVERRIDE_CPU_AFFINITY_DEBUG") != "0":
                raise k13.Abort(f"HSA_OVERRIDE_CPU_AFFINITY_DEBUG {rb.get('env_HSA_OVERRIDE_CPU_AFFINITY_DEBUG')!r} in "
                                "the server environ (want '0', the shim's GPS-PIN-1 knob)")
        self.vram_file = self.ctx.out / "vram" / f"{getattr(self.srv, 'label', LAUNCH)}.txt"
        self.vram_file.parent.mkdir(parents=True, exist_ok=True)
        if getattr(self.srv, "proc", None) is not None:   # 1 Hz own-pid KFD VRAM sampler
            threading.Thread(target=self._vram_loop, daemon=True).start()

    def _vram_loop(self) -> None:
        pid = self.srv.proc.pid
        with open(self.vram_file, "a") as fh:
            while not self._vstop.wait(1.0) and k13.pid_alive(pid):
                fh.write(f"{time.time():.1f} {k13.kfd_pids().get(pid, 0)} {k13.card_vram_used(Path(self.ctx.args.card))}\n")
                fh.flush()

    def vram_sample(self) -> int | None:
        proc = getattr(self.srv, "proc", None)
        return k13.kfd_pids().get(proc.pid, 0) if proc is not None else None

    def stop(self) -> None:
        self._vstop.set()
        if self.srv:
            self.srv.stop()

    def log_count(self, t0: float, t1: float, needle: str) -> int:
        fn = getattr(self.srv, "lines_between", None)
        if fn is None:
            return 0
        return sum(1 for ln in fn(t0, t1) if needle in ln)

    # -- requests
    def guard_cache_ram(self, slot: int, ids: list[int]) -> None:
        held = self.shadow.get(slot) or []
        if held:
            k = k13.prefix_len(held, ids, min(len(held), len(ids)))
            if k / len(held) < F_KEEP_MIN:
                raise k13.Abort(f"protocol bug: slot {slot} holds {len(held)} tokens, lcp {k}: f_keep "
                                f"{k / len(held):.3f} < {F_KEEP_MIN} would route the request through --cache-ram")

    def req(self, role: str, ids: list[int], slot: int, *, fork: bool, n_predict: int, ckpt=None,
            meta: dict | None = None) -> dict:
        ctx = self.ctx
        ctx.check_boundary()
        self.guard_cache_ram(slot, ids)
        body = k13.body_for(ids, n_predict, slot, fork=fork, cache=True, ckpt=ckpt)
        r = ctx.client.complete(body, timeout=min(7200.0, ctx.deadline.remaining() + k13.WATCHDOG_GRACE_S - 10))
        t = r.get("timings") or {}
        probs = r.get("probs") or []
        run = {"role": role, "slot": slot, "http": r.get("http"), "error": r.get("error"),
               "tokens": r.get("tokens") or [], "top2": [top2(e) for e in probs], "top2_kind": top2_kind(probs),
               "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
               "n_fork_tokens": int(t.get("n_fork_tokens") or 0), "fork_src_kind": t.get("fork_src_kind"),
               "fork_src_slot": t.get("fork_src_slot"), "cache_n": int(t.get("cache_n") or 0),
               "prompt_n": int(t.get("prompt_n") or 0), "prompt_ms": t.get("prompt_ms"),
               "predicted_ms": t.get("predicted_ms"), "wall_s": r.get("wall_s"),
               "t_send": r.get("t_send"), "t_end": r.get("t_end")}
        if n_predict > 0 and r.get("t_send") is not None:
            run["nonconsec_lines"] = self.log_count(r["t_send"], r.get("t_end") or time.time(), NONCONSEC)
        self.rows.add({"t": round(time.time(), 3), "launch": LAUNCH, **(meta or {}),
                       **{k: v for k, v in run.items() if k != "top2"}})
        if r.get("http") != 200 or r.get("error"):
            raise k13.Abort(f"{role}: HTTP {r.get('http')} {r.get('error')}")
        self.shadow[slot] = list(ids) + list(run["tokens"][:-1] if n_predict > 0 else [])
        return run

    def erase(self, *slots: int) -> None:
        for s in slots:
            self.ctx.client.erase(s)
            self.shadow[s] = []

    def erase_all(self) -> None:
        self.erase(0, 1, 2, 3)

    def seed(self, slot: int, trunk: list[int], points: list[int], tag: str, m: dict) -> None:
        for x in points:
            self.req(f"{tag}_{x}", trunk[:x], slot, fork=False, n_predict=0, ckpt=[-1], meta={**m, "arm": f"{tag}_{x}"})

    def slot_checkpoints(self, slot: int) -> list[dict]:
        s = [x for x in self.ctx.client.slots() if x.get("id") == slot]
        return [{k: ck.get(k) for k in ("n_tokens", "pos_min", "pos_max", "pinned", "size_mib")}
                for ck in ((s[0] if s else {}).get("checkpoints") or [])]

    def cell(self, g: int, c: int) -> dict:
        a, geo = self.ctx.args, self.geo
        L, S = geo["L"], a.suffix_tokens
        trunk = self.ps.trunks[g][:L]
        P = trunk[:c] + self.ps.suffixes[g][:S]
        filler = self.ps.fillers[g][:L - c]
        upto = build_upto(geo["build"], c)
        full = list(geo["build"]) + [L]
        ext = geo["ks_ext"][c]
        m = {"group": g, "c": c}
        cell = {"cell": cell_id(g, c), "group": g, "c": c, "S": S, "L": L, "build": full, "build_upto_c": upto,
                "ks_ext": ext, "filler_len": len(filler), "launch": LAUNCH, "device": a.device,
                "prompt_sha256": k13.sha256_json(P), "rules_sha256": rules_sha256(), "start_utc": k13.now_iso(),
                "runs": {k: [] for k in ARMS}, "source_checkpoints": {}, "vram_own_pid_bytes": []}
        runs = cell["runs"]

        def P_on(arm: str, slot: int, fork: bool, i: int) -> dict:
            r = self.req(arm, P, slot, fork=fork, n_predict=N_GEN, meta={**m, "arm": f"{arm}{i}"})
            if not cell["vram_own_pid_bytes"]:
                cell["vram_own_pid_bytes"].append(self.vram_sample())
            return r

        # phase S: the 17.4b geometry (source resident; X1 C1 X2 C2 C3)
        self.erase_all()
        self.seed(0, trunk, full, "source", m)
        cell["source_checkpoints"]["S"] = self.slot_checkpoints(0)

        def xc(i: int) -> None:
            self.erase(1)
            runs["Xc"].append(P_on("Xc", 1, True, i))
            self.erase(1)

        def cf(i: int) -> None:
            self.erase(2)
            self.seed(2, trunk, upto, f"Cf{i}_seed", m)
            runs["Cf"].append(P_on("Cf", 2, False, i))
            self.erase(2)
        xc(1)
        cf(1)
        xc(2)
        for i in range(2, REF_REPEATS + 1):
            cf(i)
        # phase K: own-slot checkpoint restore, no fork
        for i in range(1, ARM_REPEATS + 1):
            self.erase_all()
            self.seed(0, trunk, upto + [ext], f"Ks{i}_seed", m)
            cell["source_checkpoints"][f"K{i}"] = self.slot_checkpoints(0)
            runs["Ks"].append(P_on("Ks", 0, False, i))
        # phase M: Xc's layout without a fork
        for i in range(1, REF_REPEATS + 1):
            self.erase_all()
            self.seed(2, trunk, upto, f"Cmask{i}_seed", m)
            self.req(f"Cmask{i}_filler", filler, 3, fork=False, n_predict=0, meta={**m, "arm": f"Cmask{i}_filler"})
            runs["Cmask"].append(P_on("Cmask", 2, False, i))
        # phase E: the c-only layout
        for i in range(1, REF_REPEATS + 1):
            self.erase_all()
            self.seed(2, trunk, upto, f"C0{i}_seed", m)
            runs["C0"].append(P_on("C0", 2, False, i))
        self.erase_all()
        self.seed(0, trunk, upto, "Xe_source", m)
        cell["source_checkpoints"]["E"] = self.slot_checkpoints(0)
        for i in range(1, ARM_REPEATS + 1):
            self.erase(1)
            runs["Xe"].append(P_on("Xe", 1, True, i))
        cell["vram_own_pid_bytes"].append(self.vram_sample())
        self.erase_all()
        cell["end_utc"] = k13.now_iso()
        self.results[cell["cell"]] = cell
        k13.write_json(self.ctx.out / "cells" / LAUNCH / f"{cell['cell']}.json", cell)
        print(f"[kpf174c]   {cell['cell']} " + json.dumps(
            {k: [(r["n_fork_tokens"], r["cache_n"], r["prompt_n"], r.get("fork_src_kind")) for r in v]
             for k, v in runs.items()}), flush=True)
        return cell

    def finish_record(self, ana: dict) -> dict:
        ctx, srv, a = self.ctx, self.srv, self.ctx.args
        vmax = vn = 0
        if self.vram_file and self.vram_file.exists():
            for ln in self.vram_file.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    vmax, vn = max(vmax, int(parts[1])), vn + (int(parts[1]) > 0)
        rows = self.rows.path
        partial = 0
        if rows.exists():
            done = set(self.results)
            seen = {cell_id(r.get("group"), r.get("c")) for r in map(json.loads, rows.read_text().splitlines())
                    if r.get("group") is not None}
            partial = len(seen - done)
        rec = {
            "schema": SCHEMA, "kpf_task": TASK, "launch": LAUNCH, "launch_task": "17.4c",
            "run_id": f"kpf174c-{LAUNCH}-{k13.utc_tag()}", "arm": k13.CONFIGS[CONFIG_NAME]["label"],
            "binary": str(a.fork_bin), "binary_sha256": ctx.bin_sha(Path(a.fork_bin)),
            "argv": getattr(srv, "argv", None), "launch_prefix": list(ctx.prefix),
            "argv_readback": k13.readback_record(getattr(srv, "readback", None)) if srv else None,
            "env": getattr(srv, "env_note", None), "linkage": getattr(srv, "linkage", None),
            "stop": getattr(srv, "stop_rec", None),
            "drafter_flags_removed": getattr(srv, "drafter_removed", None),
            "vram_own_pid_peak_bytes": vmax, "vram_nonzero_samples": vn, "vram_file": str(self.vram_file),
            "prompt_set": str(self.ps.path), "prompt_set_sha256": self.ps.sha256, "geometry": self.geo,
            "suffix_tokens": a.suffix_tokens, "groups": a.groups, "n_predict": N_GEN, "seed": SEED,
            "rules": RULES["schema"], "rules_sha256": rules_sha256(), "spec_off": True,
            "decisions": ana.get("decisions"), "cells_interrupted": partial,
            "window_id": a.window_id, "device": a.device, "cpu_list": a.cpu_list,
            "start_utc": ctx.start_utc, "end_utc": k13.now_iso(), "host": ctx.host,
            "cells_dir": str(ctx.out / "cells" / LAUNCH), "rows": str(rows),
            "rows_sha256": k13.sha256_file(rows), "n_cells": len(self.results),
            "instrument_class": "serving", "category": "CANDIDATE",
            "grading_note": "deterministic conformance (bit identity at matched layout); counts, not a pass word",
        }
        rec = k13.seal_record(rec)
        k13.write_json(ctx.out / "records" / f"{rec['run_id']}.json", rec)
        k13.Jsonl(ctx.out / "belief_measurements.jsonl").add(rec)
        return rec


def load_results(out: Path) -> dict:
    res: dict = {}
    d = out / "cells" / LAUNCH
    if d.exists():
        for f in sorted(d.glob("*.json")):
            res[f.stem] = json.loads(f.read_text())
    return res


# ---------------------------------------------------------------- estimate (planning only)
def cell_cost(geo: dict, c: int, S: int) -> tuple[int, int, int]:
    """(prefill tokens, decode tokens, requests) of one cell."""
    L, upto = geo["L"], build_upto(geo["build"], c)
    nb = len(upto)
    prefill = (L + REF_REPEATS * c + ARM_REPEATS * geo["ks_ext"][c] + REF_REPEATS * L + REF_REPEATS * c + c
               + (3 * ARM_REPEATS + 3 * REF_REPEATS) * S)
    decode = (3 * ARM_REPEATS + 3 * REF_REPEATS) * N_GEN
    reqs = (len(geo["build"]) + 1 + ARM_REPEATS + REF_REPEATS * (nb + 1) + ARM_REPEATS * (nb + 2)
            + REF_REPEATS * (nb + 2) + REF_REPEATS * (nb + 1) + nb + ARM_REPEATS)
    return prefill, decode, reqs


def estimate(a, geo: dict, cells: list[tuple[int, int]]) -> float:
    if not cells:
        return 0.0
    tot = a.load_s + 30
    for _g, c in cells:
        pf, dec, nr = cell_cost(geo, c, a.suffix_tokens)
        tot += pf / a.prefill_tok_s + dec / a.decode_tok_s + 1.0 * nr
    return tot


# ---------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fork-bin", default=None, help="the GPU image (gpu-20261023-82c16126f-kpf191/bin/llama-server)")
    ap.add_argument("--argv-file", default=str(DEFAULT_ARGV))
    ap.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET)
    ap.add_argument("--groups", default="1,2")
    ap.add_argument("--junctions", default=",".join(map(str, DEFAULT_JUNCTIONS)))
    ap.add_argument("--build-junctions", default=",".join(map(str, DEFAULT_BUILD)))
    ap.add_argument("--L", dest="L", type=int, default=DEFAULT_L)
    ap.add_argument("--suffix-tokens", type=int, default=64)
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--cpu-list", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--numa-membind", type=int, default=None)
    ap.add_argument("--port", type=int, default=18989)
    ap.add_argument("--fork-min", type=int, default=1024)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--deadline-utc", default=None)
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--window-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--load-timeout-s", type=float, default=900.0)
    ap.add_argument("--card", default=str(k13.DEFAULT_CARD))
    ap.add_argument("--vram-free-max-gib", type=float, default=2.0)
    ap.add_argument("--vram-drain-timeout-s", type=float, default=k13.VRAM_DRAIN_TIMEOUT_S)
    ap.add_argument("--vram-drain-poll-s", type=float, default=k13.VRAM_DRAIN_POLL_S)
    ap.add_argument("--no-region-lock", default=None, metavar="REASON")
    ap.add_argument("--no-device-claim", default=None, metavar="REASON")
    ap.add_argument("--prefill-tok-s", type=float, default=730.0, help="estimate only (KPF-13.1 v3 GPU median 734)")
    ap.add_argument("--decode-tok-s", type=float, default=25.0, help="estimate only")
    ap.add_argument("--load-s", type=float, default=60.0, help="estimate only")
    a = ap.parse_args(argv)
    a.groups = [int(x) for x in a.groups.split(",") if x.strip()]
    a.junctions = [int(x) for x in a.junctions.split(",") if x.strip()]
    a.build_junctions = [int(x) for x in a.build_junctions.split(",") if x.strip()]
    a.draft_debug = False
    return a


def preflight(a, ps: Prompts | None, gate_argv: list[str], geo: dict) -> list[str]:
    p = []
    if not (1024 <= a.port <= 65535) or 8000 <= a.port <= 8999:
        p.append(f"--port {a.port} refused (never 8000-8999 or < 1024)")
    if k13.port_listening(a.port):
        p.append(f"--port {a.port} already listening")
    if a.device == "gpu" and k13.port_listening(8083):
        p.append("production :8083 is listening (its owner must stop it first)")
    p += k13.argv_problems(gate_argv, "dflash2", a.device)
    off, removed = k13.strip_drafter(gate_argv)
    p += k13.spec_off_problems(off)
    if "-md" not in removed and "--model-draft" not in removed:
        p.append("speculation OFF: the argv file carries no -md to remove (not the 17.4b DFlash2 argv?)")
    if not a.fork_bin:
        p.append("--fork-bin is required (gpu-20261023-82c16126f-kpf191)")
    elif not Path(a.fork_bin).exists():
        p.append(f"--fork-bin {a.fork_bin} does not exist")
    else:
        for kn in k13.CONFIGS[CONFIG_NAME]["knob"]:
            if not k13.bytes_contain_any_lib(Path(a.fork_bin), kn):
                p.append(f"knob {kn!r} not compiled into {a.fork_bin}")
    p += geometry_problems(geo, a.suffix_tokens, a.fork_min)
    if ps is None:
        p.append("no prompt set")
    else:
        p += ps.problems_174c(a.groups, geo["L"], a.suffix_tokens, geo["c"], k13.flag_val(gate_argv, ("-m", "--model")))
    if not a.cpu_list:
        p.append("--cpu-list is required (GPU: the claimed host threads)")
    return p


def main(argv=None) -> int:
    a = parse_args(argv)
    gate_argv, comments = k13.load_argv_file(Path(a.argv_file))
    geo = geometry(a.junctions, a.build_junctions, a.L)
    out = Path(a.out)
    cells = planned_cells(a.groups, geo["c"])
    planned = [cell_id(g, c) for g, c in cells]
    if a.cmd == "analyze":
        res = load_results(out)
        if not res:
            print(f"no cells under {out / 'cells' / LAUNCH}")
            return 2
        ana = analyze(res, planned)
        k13.write_json(out / "verdict.json", ana)
        print(json.dumps(ana["decisions"], indent=1))
        return 0
    a.window_id = a.window_id or f"kpf174c-{k13.utc_tag()}"
    try:
        ps = Prompts(Path(a.prompt_set))
    except (OSError, ValueError, KeyError) as exc:
        print(f"prompt set unreadable: {exc}")
        ps = None
    probs = preflight(a, ps, gate_argv, geo)
    deadline = (k13.Deadline(k13.utc_epoch(a.deadline_utc)) if a.deadline_utc else
                k13.Deadline(time.time() + a.deadline_s) if a.deadline_s is not None else k13.Deadline(None))
    if not a.dry_run:
        if deadline.epoch is None:
            probs.append("--deadline-utc (or --deadline-s) is required to execute")
        if not a.no_region_lock and os.environ.get("KPF13_REGION") != a.cpu_list:
            probs.append(f"not inside a region-lock claim for {a.cpu_list!r} (KPF13_REGION="
                         f"{os.environ.get('KPF13_REGION')!r}); pass --no-region-lock REASON if the caller holds it")
    a.fork_bin = a.fork_bin or "/nonexistent"
    ctx = k13.Ctx(a, None, gate_argv, deadline)
    ctx.start_utc = k13.now_iso()
    results = load_results(out) if a.resume else {}
    todo = Runner(ctx, geo, ps, results).pending(cells) if ps else cells
    est = estimate(a, geo, todo)
    print("=" * 100)
    print(f"KPF-17.4c target-restore bisect (speculation OFF)  device={a.device}  rules={RULES['schema']} "
          f"rules_sha256={rules_sha256()[:16]}  repeats: references {REF_REPEATS}, arms {ARM_REPEATS}")
    print(f"geometry: junctions {geo['c']}, build points {geo['build']} + L {geo['L']}, Ks extensions {geo['ks_ext']}, "
          f"S {a.suffix_tokens}, groups {a.groups}")
    print(f"launch {LAUNCH}: speculation OFF, drafter flags removed: {k13.strip_drafter(gate_argv)[1]}")
    print(f"cells (in order): {planned}; pending {[cell_id(g, c) for g, c in todo]}"
          + ("  [all stored: no load]" if not todo else ""))
    for g, c in todo:
        pf, dec, nr = cell_cost(geo, c, a.suffix_tokens)
        print(f"  {cell_id(g, c)}: phases S (Xc x{ARM_REPEATS}, Cf x{REF_REPEATS}) K (Ks x{ARM_REPEATS}) "
              f"M (Cmask x{REF_REPEATS}) E (C0 x{REF_REPEATS}, Xe x{ARM_REPEATS}); prefill {pf} tok, decode {dec} tok, "
              f"{nr} requests")
    if ctx.prefix:
        print(f"launch prefix: {' '.join(ctx.prefix)}")
    print(f"TOTAL est {k13.fmt_s(est)} wall (rates: prefill {a.prefill_tok_s} tok/s, decode {a.decode_tok_s} tok/s, "
          f"load {a.load_s:.0f}s; GPU rates are planning values)")
    if probs:
        print("PREFLIGHT PROBLEMS (dry-run exits 0 regardless; execute refuses):")
        for x in probs:
            print(f"  - {x}")
    else:
        print("PREFLIGHT: inputs PASS")
    if a.dry_run:
        print("DRY RUN complete: no process was started, nothing was sent.")
        return 0
    if probs:
        print("REFUSED (preflight).")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    k13.write_json(out / "rules.json", {"rules": RULES, "rules_sha256": rules_sha256()})
    k13.write_json(out / "run_meta.json", {
        "args": vars(a), "estimate_s": est, "started": k13.now_iso(), "rules_sha256": rules_sha256(),
        "rules_declared_before_first_launch": True, "geometry": geo, "planned_cells": planned,
        "argv_file": a.argv_file, "argv_file_sha256": k13.sha256_file(Path(a.argv_file)), "argv_comments": comments,
        "prompt_set_sha256": ps.sha256, "launch_prefix": list(ctx.prefix),
        "device_claim": a.no_device_claim or ("mi210_0" if a.device == "gpu" else "n/a"),
        "region_lock": {"KPF13_REGION": os.environ.get("KPF13_REGION"), "skipped": a.no_region_lock}})
    if not todo:
        ana = analyze(results, planned)
        k13.write_json(out / "verdict.json", ana)
        print("[kpf174c] all cells stored, no load (resume); verdict: " + json.dumps(ana["decisions"]["locus"]))
        return 0
    dog = k13.Watchdog(ctx)
    claim_cm = None
    runner = Runner(ctx, geo, ps, results)

    def on_sig(signum, _f):
        try:
            runner.stop()
        finally:
            raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    rc = 0
    try:
        if a.device == "gpu" and not a.no_device_claim:
            sys.path.insert(0, str(k13.CLAIM_MOD_DIR))
            from autokernel.resource import device_claim as dc  # noqa: E402
            try:
                claim_cm = dc.acquire_device_claim(
                    "mi210_0", purpose=f"KPF-17.4c target-restore bisect on :{a.port}", campaign_id="KPF-17.4c",
                    journal=dc.ClaimJournal(out / "claim" / "device-claim-journal.jsonl"), holder_label="kpf174c",
                    timeout_s=0, max_hold_s=(deadline.remaining() + 600))
            except dc.DeviceClaimError as exc:
                print(f"[kpf174c] REFUSED: mi210_0 device claim not acquired: {exc}", flush=True)
                return 75
            claim_cm.__enter__()
        dog.start()
        if a.stop_file and Path(a.stop_file).exists():
            raise k13.Abort(f"stop file {a.stop_file} present: draining before the launch")
        first = estimate(a, geo, todo[:1])
        if not deadline.fits(first):
            raise k13.Abort(f"deadline: the load + first cell (est {k13.fmt_s(first)}) does not fit; draining")
        print(f"[kpf174c {time.strftime('%H:%M:%SZ', time.gmtime())}] launch {LAUNCH} (est {k13.fmt_s(est)})", flush=True)
        try:
            runner.start()
            for g, c in todo:
                if a.stop_file and Path(a.stop_file).exists():
                    raise k13.Abort("stop file present: draining at a cell boundary")
                per = estimate(a, geo, [(g, c)]) - a.load_s - 30
                if not deadline.fits(per):
                    raise k13.Abort(f"deadline: cell {cell_id(g, c)} est {k13.fmt_s(per)} does not fit; draining")
                runner.cell(g, c)
        finally:
            runner.stop()
            try:
                runner.finish_record(analyze(load_results(out), planned))
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf174c] record failed: {exc}", flush=True)
    except k13.Abort as exc:
        print(f"[kpf174c] ABORT: {exc}", flush=True)
        rc = 5 if ("deadline" in str(exc) or "stop file" in str(exc)) else 3
    except Exception as exc:  # noqa: BLE001
        print(f"[kpf174c] ERROR: {type(exc).__name__}: {exc}", flush=True)
        rc = 4
    finally:
        try:
            runner.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[kpf174c] cleanup of own pid failed: {exc}", flush=True)
        dog.stop()
        if claim_cm is not None:
            try:
                claim_cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf174c] claim release failed: {exc}", flush=True)
        try:
            res = load_results(out)
            if res:
                ana = analyze(res, planned)
                k13.write_json(out / "verdict.json", ana)
                print("[kpf174c] verdict: " + json.dumps({k: ana["decisions"][k] for k in
                                                          ("locus", "decision_grade", "explains_174b")}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[kpf174c] analyze failed: {exc}", flush=True)
        proc = getattr(runner.srv, "proc", None) if runner.srv else None
        if proc is not None and k13.pid_alive(proc.pid):
            print(f"[kpf174c] WARNING: own child pid {proc.pid} still alive", flush=True)
            rc = rc or 4
    print(f"[kpf174c] exit rc={rc} (0 ok, 2 refused, 3 abort, 4 error, 5 deadline/stop drain, 75 claim held)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
