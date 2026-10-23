#!/usr/bin/env python3
"""KPF-17.4 / 17.4a / 19.1a driver: pinned fork junctions on the SWA DRAFTER of Qwen3.8-27B + DFlash2, one GPU window.

Handoff: handoffs/active/kv-prefix-fork-and-paged-attention.md, rows KPF-17.4, KPF-17.4a, KPF-19.1a (re-targeted
2026-10-21 by the backlog audit from the retired Gemma-class target to the 27B + DFlash2 pair, "TOGETHER ... in ONE server
load (same build, same argv)"). Written 2026-10-23 (workspace-f8). Class: correctness (token identity and draft counts)
plus one measurement (17.4a); no timing claim, no quiet host.

WHY A NEW HARNESS (the "SWA range fix" the FILLERS row names)
  The 2026-10-20 job-23 failures (recorded range != restored range, continuation max|d| 0.33) were ORACLE defects, fixed
  by f3a3bb4b1 (matched control, measured floor, byte-exact restore) and verified by queue job 24 (2026-10-21 00:01Z:
  swa-cpp-normal OK, use_recorded_range DETECTED). That fix covers an SWA TARGET (gemma-3-1b). On the re-target the SWA
  cache is the DRAFTER: Qwen3.8-27B is `qwen35` (hybrid GDN, n_swa 0) and Qwen3.8-27B-DFlash2 is `dflash` with
  sliding_window 2048 on all 5 layers. Neither tests/test-kpf-swa-checkpoint.cpp (loads one model, SKIPs on n_swa 0) nor
  test_slot_fork.py::test_kpf174 (target checkpoints only, a ~1k-token trunk inside the drafter window) can see the
  drafter's window. This driver judges the drafter's restored window behaviourally: the target verifies every draft, so
  a drafter state that misses part of its window shows up ONLY in draft_n / draft_n_accepted; tokens are the second check.

WHAT IT RUNS (one window, up to four launches of the SAME binary and argv; every launch is its own child process)
  honest      flag off. Per prompt group g: source slot 0 <- the BUILD SEQUENCE trunk[:c1], trunk[:c2], trunk[:c3],
              trunk[:L] (each request extends the last, checkpoint_at [-1]: every junction is a pinned prompt-end
              checkpoint). Per junction c (descending): fork children X1, X2 on slot 1 (P = trunk[:c] + suffix[:S], must
              fork from slot 0 at exactly c), K same-shape CONTROLS on slot 2 (no fork: replay the build sequence up to
              trunk[:c], then P = a prompt-cache hit at c with the S-token remainder: the source's and the arm's exact
              batch partition, the KPF-13 lesson), and one informational FRESH cold run on slot 3. Slots 1-3 erased.
  flag_on     the same cells with --slot-fork-dft-full (KPF-19.1a "flag on"; for an SWA drafter slot_fork_dft_plain() is
              false, so the code path is expected identical: the launch proves it).
  mut_no_transfer   LLAMA_KPF_MUTATE=dft_no_transfer: the child gets NO drafter state. 19.1a requires it DETECTED.
  mut_partial_noop  LLAMA_KPF_MUTATE=dft_partial_noop: the child's drafter window comes only from the attention-only
              share of the SOURCE's live drafter cache, never from the checkpoint. At junctions the source's SWA cache has
              already overwritten (c < L - swa_cells) that is the "recorded range instead of the real window" mutation
              of KPF-17.4; it must be DETECTED there. At a junction still inside the source's live window it is expected
              output-neutral (reported, not required).
  Junctions: c1 = n_swa + 100, c2 = 2 n_swa (as the C++ case), c3 = the 17.4a junction above the drafter SWA cache bound
  swa_cells = n_swa * n_parallel + n_ubatch (8704 at n_swa 2048, -np 4, -ub 512). L > swa_cells + c2 so c1/c2 are pruned
  from the source's live drafter cache when the children fork.

KPF-17.4b (2026-10-23, rules v2 = RULES_V2, `--rules v2`): localize the deterministic c1-junction divergence of the
  2026-10-23 GPU run (g1/g2 at c = n_swa + 100 = 2148; honest, flag_on and both mutants diverge identically, so the drafter
  window is not the cause). Two launches, each its own driver invocation and out dir:
    sweep           speculation ON, `--junctions 2049,2148,2560,3072,4096`: every junction is a cell AND a build point
                    (the source is built trunk[:2049], trunk[:2148], ..., trunk[:L]); where does the failure start?
    honest_specoff  speculation OFF (the argv file minus every drafter flag: no -md), the ORIGINAL geometry
                    (`--junctions 2148 --build-junctions 2148,4096,10240`): X != C there = a target-restore defect at c1,
                    independent of the drafter. Spec-off cells are tokens only and need token-identical controls.

RULES: RULES below (sha256 recorded in rules.json and run_meta.json BEFORE the first launch). Verdicts are a pure function
of the stored cells (`analyze`). The per-cell judgement is the KPF-13.1 rules-v3 judge (kpf13_restore_ab.judge_run with
the measured control spread). PROCESS DISCIPLINE is the KPF-13.1 driver's (own child by captured PID, port ownership by
socket inode, KFD residency, ggml linkage, /proc read-back, hard deadline + watchdog); this module imports it.

Exit codes: 0 ok, 2 refused (preflight), 3 abort, 4 error, 5 deadline/stop drain, 75 claim held. stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("kpf13_restore_ab", HERE.parent / "kpf13" / "kpf13_restore_ab.py")
k13 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(k13)

TASK = "KPF-17.4+17.4a+19.1a"
SCHEMA = k13.SCHEMA
N_GEN = 64
SEED = k13.SEED
DEFAULT_ARGV = HERE.parent / "kpf13" / "argv.gpu-27b-dflash2.txt"
DEFAULT_PROMPT_SET = "/mnt/raid0/llm/tmp/overnight-20261020/kpf18-run/prompt_set.jsonl"
DEFAULT_PRIOR_191A = "/mnt/raid0/llm/tmp/kpf18-s1/out/records/*.json"
MIN_CONTROLS = 3

# launches (kpf13 Server CONFIGS shape). `knob` strings must be compiled into the binary or a sibling .so (refusal).
LAUNCHES = {
    "honest": {"label": "kpf174_191a_flag_off", "mutate": None, "extra": [], "arms": ("A",),
               "knob": [b"--slot-fork-min-tokens"], "plain_only": False, "role": "honest", "task": "17.4+17.4a+19.1a"},
    "flag_on": {"label": "kpf191a_flag_on", "mutate": None, "extra": ["--slot-fork-dft-full"], "arms": ("A",),
                "knob": [b"--slot-fork-min-tokens", b"--slot-fork-dft-full"], "plain_only": False, "role": "honest",
                "task": "19.1a"},
    "mut_no_transfer": {"label": "kpf191a_mut_dft_no_transfer", "mutate": "dft_no_transfer", "extra": [],
                        "arms": ("A",), "knob": [b"--slot-fork-min-tokens", b"LLAMA_KPF_MUTATE", b"dft_no_transfer"],
                        "plain_only": False, "role": "mutant", "task": "19.1a"},
    "mut_partial_noop": {"label": "kpf174_mut_dft_partial_noop", "mutate": "dft_partial_noop", "extra": [],
                         "arms": ("A",), "knob": [b"--slot-fork-min-tokens", b"LLAMA_KPF_MUTATE", b"dft_partial_noop"],
                         "plain_only": False, "role": "mutant", "task": "17.4"},
    # KPF-17.4b (rules v2)
    "sweep": {"label": "kpf174b_junction_sweep", "mutate": None, "extra": [], "arms": ("A",),
              "knob": [b"--slot-fork-min-tokens"], "plain_only": False, "role": "honest", "task": "17.4b"},
    "honest_specoff": {"label": "kpf174b_c1_spec_off", "mutate": None, "extra": [], "arms": ("A",),
                       "knob": [b"--slot-fork-min-tokens"], "plain_only": False, "role": "honest", "task": "17.4b",
                       "spec_off": True},
}
LAUNCHES_174B = ("sweep", "honest_specoff")
for _k, _v in LAUNCHES.items():
    k13.CONFIGS.setdefault(f"kpf174_{_k}", _v)
LAUNCHES_191A = ("flag_on", "mut_no_transfer")

RULES = {
    "schema": "epyc.kpf174.rules.v1",
    "task": ("KPF-17.4 (SWA checkpoint range, re-targeted to the 27B + DFlash2 SWA drafter), KPF-17.4a (long-context SWA "
             "fork loss, measured), KPF-19.1a (DFlash2 SWA-drafter fork, flag off/on, dft_no_transfer must FAIL)"),
    "declared": "before the first launch; every verdict is a pure function of the stored cells (analyze)",
    "sampling": {"temperature": 0.0, "seed": SEED, "ignore_eos": True, "n_generated": N_GEN, "n_probs": 2},
    "cell": ("one (prompt group g, junction c): source slot 0 built once per group by the BUILD SEQUENCE trunk[:c1], "
             "trunk[:c2], trunk[:c3], trunk[:L] (each n_predict 0, cache_prompt on, slot_fork off, checkpoint_at [-1]: "
             "every junction is a pinned prompt-end checkpoint, each request extends the previous one); arm runs X1, X2 "
             "= fork children on slot 1 (P = trunk[:c] + suffix[:S], slot_fork on, cache_prompt on); controls C1..CK on "
             "slot 2 (no fork: replay the build sequence up to and including trunk[:c], then P, slot_fork off, "
             "cache_prompt on: a no-fork prompt-cache hit at c with the S-token remainder and the SAME batch partition "
             "as the source and the arm); one fresh cold run F on slot 3 (informational). Order per junction: X1 C1 X2 "
             "C2 C3 .. CK F; junctions descending; slots 1-3 erased before and after use"),
    "cell_invalid_if": [
        "an HTTP error or fewer than n_generated tokens (ignore_eos)",
        "a control has draft_n == 0 or None (speculation inactive: vacuous)",
        "a control reports n_fork_tokens != 0, or cache_n != c, or prompt_n != S (partition not reproduced)",
        "an arm run's n_fork_tokens != c or fork_src_slot != 0 (the fork was not taken where designed)",
        "the source's /slots checkpoint list lacks a pinned checkpoint with n_tokens == c",
    ],
    "cell_judge": ("kpf13_restore_ab.judge_run against the cell's K controls with rules-v3 slack: tokens: X agrees with "
                   "some control for at least the smallest pairwise control agreement over 64 tokens; acceptance: "
                   "draft_n and draft_n_accepted of X within [min - spread, max + spread] of the cell's controls, where "
                   "spread is the launch's pooled (max - min) over every valid cell's K controls (controls only). "
                   "A cell PASSES iff both arm runs PASS"),
    "min_controls": MIN_CONTROLS,
    "launch_verdict": {
        "honest": "PASS iff every attempted cell PASSES (all attempts counted, INVALID included); FAIL if any valid "
                  "cell FAILS; else INCONCLUSIVE",
        "mutant": ("DETECTED iff at least one valid cell FAILS (scope: `detect_scope`); NOT_DETECTED iff every cell "
                   "in scope is valid and PASSES; else INCONCLUSIVE"),
        "detect_scope": {"mut_no_transfer": "every junction",
                         "mut_partial_noop": "PRUNED junctions only (c < L - swa_cells); junctions still inside the "
                                             "source's live drafter window are reported, not required"},
    },
    "decisions": {
        "kpf17_4": ("PASS iff honest PASS and mut_partial_noop DETECTED (the oracle provably sees a drafter window "
                    "shortfall). honest PASS with NOT_DETECTED -> NOT_ESTABLISHED (oracle blind to the window: no claim). "
                    "honest FAIL -> FAIL. Pinned-junction use on SWA drafter caches is gated on PASS"),
        "kpf19_1a": ("PASS iff honest PASS and flag_on PASS and mut_no_transfer DETECTED; FAIL if honest or flag_on "
                     "FAIL; NOT_ESTABLISHED if both honest launches PASS and the mutant is NOT_DETECTED; else "
                     "INCONCLUSIVE. Skipped (prior) when a complete KPF-19.1a record already exists (resume-skip)"),
        "kpf17_4a": ("MEASUREMENT, no pass word: per junction, fork taken (n_fork_tokens == c) and loss_tokens = "
                     "prompt_n - S of each arm run (tokens re-processed beyond the suffix). Prediction from code reading "
                     "(find_fork_source's SWA gate reads the TARGET n_swa, 0 on qwen35): fork taken, loss 0 at every "
                     "junction including c3 > swa_cells"),
    },
    "recorded_range": ("for every pinned source checkpoint the /slots entry (n_tokens, pos_min, pos_max, size_mib) is "
                       "stored. These are the TARGET's recorded range (hybrid: recurrent pos_min ~ pos_max); the server "
                       "records no drafter pos range, so the drafter window is judged behaviourally (cell_judge + "
                       "mut_partial_noop), which is the 27B + DFlash2 form of the task's 'recorded vs real' question"),
    "info_only": ["the fresh cold run F (A_vs_fresh column; it takes a different batch partition)",
                  "top-2 logprob drift of the greedy token vs the first control over the agreeing prefix",
                  "wall times, prompt_ms"],
    "restore_does_not_carry": k13.RULES["restore_does_not_carry"],
}


RULES_V2 = json.loads(json.dumps(RULES))
RULES_V2.update({
    "schema": "epyc.kpf174.rules.v2",
    "task": ("KPF-17.4b: localize the deterministic c1-junction divergence of the 2026-10-23 GPU run (kpf174 rules v1 sha "
             "26b04e19..., /mnt/raid0/llm/tmp/f8-launchers-20261023/kpf17-4-19-1a-27b-dflash2/VERDICT.json: honest FAIL "
             "at g1-c2148 token 36 gap 0.164 and g2-c2148 token 39 gap 0.493; c 4096 and 10240 PASS). It decides where "
             "the failure lives; it re-judges nothing and decides neither KPF-17.4 nor KPF-19.1a"),
    "supersedes": "nothing: v1 and its verdicts stand; v2 adds two launches and their decisions",
    "launches_v2": {
        "sweep": ("speculation ON, honest (no mutation, flag off), --junctions 2049,2148,2560,3072,4096: every junction "
                  "is a cell and a build point (source = trunk[:2049], trunk[:2148], trunk[:2560], trunk[:3072], "
                  "trunk[:4096], trunk[:L]); same cell protocol and judge as v1 (spec-on, K controls, measured spread)"),
        "honest_specoff": ("speculation OFF: the SAME argv file with every drafter flag removed (-md, -ngld, --spec-type, "
                           "--spec-draft-n-max, --device-draft; the read-back refuses a /proc cmdline that still carries "
                           "one), the ORIGINAL v1 geometry (cells at c = 2148 only, build points 2148, 4096, 10240, L), "
                           "groups 1 and 2, the same cell protocol"),
    },
    "spec_off_judge": ("tokens only: X1, X2 must equal the controls on all 64 tokens. Every run must report draft_n 0 or "
                       "none (else INVALID: the drafter is still on). The K controls must be token-identical among "
                       "themselves (else INVALID: the target alone is not deterministic on this device, and no token "
                       "comparison can locate a restore defect). The cell_invalid_if partition/fork/pin checks apply"),
    "decisions_v2": {
        "c1_locus": ("from honest_specoff over its c = 2148 cells: target_restore_defect iff at least one valid cell "
                     "FAILS (a fork child differs from token-identical no-fork controls with no drafter loaded); "
                     "drafter_on_only iff every cell PASSES (the c1 divergence needs the drafter: verify-batch shape or "
                     "drafter state, not the target restore); INCONCLUSIVE otherwise"),
        "junction_onset": ("from sweep: per group the junctions that FAIL and PASS; failing_junctions = the union; "
                           "first_failing_junction = the smallest; pattern = none_fail | ubatch_misaligned_only (every "
                           "failing c has c mod n_ubatch != 0 and every passing c has c mod n_ubatch == 0) | "
                           "first_build_point_only (the only failing c is the first build point, 2049) | other. "
                           "Descriptive: a pattern is a lead for the next probe, not a cause"),
        "sweep_2148": ("the sweep's c = 2148 cells: FAIL means the c1 divergence does not need 2148 to be the source's "
                       "first build request (in the sweep 2049 is); PASS means it does"),
        "not_decided_here": "kpf17_4, kpf19_1a and pinned_junction_on_swa_drafter_allowed keep their v1 verdicts",
    },
})
RULES_BY_VERSION = {"v1": RULES, "v2": RULES_V2}


def rules_sha256(version: str = "v1") -> str:
    return k13.sha256_json(RULES_BY_VERSION[version])


# ---------------------------------------------------------------- geometry
def junctions(n_swa: int, n_parallel: int, n_ubatch: int, cells: list[int] | None = None,
              build: list[int] | None = None) -> dict:
    """Default (v1): cells = build points = [c1, c2, c3]. KPF-17.4b: `cells` overrides the junctions judged, `build` the
    source's build points (default: the cells; must contain every cell junction). L never moves."""
    swa_cells = n_swa * n_parallel + n_ubatch
    c1, c2 = n_swa + 100, 2 * n_swa
    c3 = ((swa_cells + 1536) // 512) * 512          # above the SWA cache bound, on a ubatch multiple (10240 at 2048/4/512)
    L = c3 + 4096                                    # 14336: c1, c2 < L - swa_cells (pruned), c3 >= L - swa_cells (live)
    cs = sorted(set(cells)) if cells else [c1, c2, c3]
    bp = sorted(set(build)) if build else list(cs)
    geo = {"c": cs, "L": L, "swa_cells": swa_cells, "n_swa": n_swa,
           "pruned": {c: c < L - swa_cells for c in cs}}
    if cells or build:
        geo.update(build=bp, n_ubatch=n_ubatch)
    return geo


def geometry_problems(geo: dict) -> list[str]:
    p = []
    bp = geo.get("build", geo["c"])
    if not set(geo["c"]) <= set(bp):
        p.append(f"cell junctions {geo['c']} are not all build points {bp} (each junction must be a pinned prompt end)")
    if max(bp) >= geo["L"]:
        p.append(f"build point {max(bp)} >= L {geo['L']}")
    return p


# ---------------------------------------------------------------- judging (pure)
def cell_as_k13(cell: dict) -> dict:
    """The kpf13 cell shape (T = junction, suffix_len = S, controls CA*, runs A*) so its judge and spread apply."""
    return {"cell": cell["cell"], "T": cell["c"], "N": None, "suffix_len": cell["S"],
            "controls": {f"CA{i + 1}": r for i, r in enumerate(cell.get("controls") or [])},
            "runs": {f"A{i + 1}": r for i, r in enumerate(cell.get("arms") or [])}}


def judge_cell(cell: dict, slack: tuple[int, int], spec_off: bool = False) -> dict:
    kc = cell_as_k13(cell)
    refs = k13.controls_of(kc, "A")
    exp = {"cache_n": cell["c"], "prompt_n": cell["S"]}
    js = []
    for x in cell.get("arms") or []:
        j = k13.judge_run(refs, x, cell["c"], kind="control", ref_expect=exp, slack=slack, spec_off=spec_off,
                          identical_refs=spec_off)
        if j["verdict"] != "INVALID" and x.get("fork_src_slot") not in (0, None):
            j = {**j, "verdict": "INVALID", "invalid": [f"fork_src_slot {x.get('fork_src_slot')} != 0"]}
        js.append(j)
    pinned = [ck for ck in (cell.get("source_checkpoints") or []) if ck.get("n_tokens") == cell["c"] and ck.get("pinned")]
    kinds = [j["verdict"] for j in js]
    if not pinned:
        v = "INVALID"
    elif len(refs) < MIN_CONTROLS and "FAIL" not in kinds:
        v = "INVALID"
    else:
        v = "FAIL" if "FAIL" in kinds else ("INVALID" if "INVALID" in kinds or not kinds else "PASS")
    fresh = cell.get("fresh")
    a_vs_fresh = (k13.cell_arm_verdict([fresh], cell.get("arms") or [], cell["c"], kind="twin",
                                       spec_off=spec_off)["verdict"] if fresh else None)
    loss = [(x.get("prompt_n") or 0) - cell["S"] for x in cell.get("arms") or [] if x]
    taken = [x.get("n_fork_tokens") == cell["c"] for x in cell.get("arms") or [] if x]
    return {"verdict": v, "runs": js, "n_controls": len(refs), "pinned_checkpoint_present": bool(pinned), "spec_off": spec_off,
            "first_divergence": min((j["first_divergence"] for j in js if j.get("first_divergence") is not None),
                                    default=None),
            "acceptance_drop": any(j.get("acceptance_drop") for j in js), "A_vs_fresh": a_vs_fresh,
            "kpf17_4a": {"fork_taken": taken, "loss_tokens": loss}}


def launch_verdict(role: str, judged: list[dict], scope: list[bool] | None = None) -> dict:
    """judged: per-cell judge dicts; scope: per cell, whether it counts for a mutant's detection."""
    if scope is None:
        scope = [True] * len(judged)
    inn = [j for j, s in zip(judged, scope) if s]
    n = len(inn)
    p = sum(1 for j in inn if j["verdict"] == "PASS")
    f = sum(1 for j in inn if j["verdict"] == "FAIL")
    inv = sum(1 for j in inn if j["verdict"] == "INVALID")
    if role == "honest":
        v = "PASS" if n and p == n else ("FAIL" if f else "INCONCLUSIVE")
    else:
        v = "DETECTED" if f else ("NOT_DETECTED" if n and p == n else "INCONCLUSIVE")
    return {"verdict": v, "attempts": n, "pass": p, "fail": f, "invalid": inv,
            "out_of_scope": len(judged) - n, "acceptance_drop": any(j.get("acceptance_drop") for j in inn)}


def decide(lv: dict, prior_191a: dict | None = None) -> dict:
    h = (lv.get("honest") or {}).get("verdict", "NOT_RUN")
    fo = (lv.get("flag_on") or {}).get("verdict", "NOT_RUN")
    mnt = (lv.get("mut_no_transfer") or {}).get("verdict", "NOT_RUN")
    mpn = (lv.get("mut_partial_noop") or {}).get("verdict", "NOT_RUN")
    out: dict = {"inputs": {"honest": h, "flag_on": fo, "mut_no_transfer": mnt, "mut_partial_noop": mpn}}
    if h == "FAIL":
        out["kpf17_4"] = "FAIL"
    elif h == "PASS" and mpn == "DETECTED":
        out["kpf17_4"] = "PASS"
    elif h == "PASS" and mpn == "NOT_DETECTED":
        out["kpf17_4"] = "NOT_ESTABLISHED"
    else:
        out["kpf17_4"] = "INCONCLUSIVE"
    out["pinned_junction_on_swa_drafter_allowed"] = out["kpf17_4"] == "PASS"
    if prior_191a:
        out["kpf19_1a"] = f"PRIOR: {prior_191a.get('verdict')} ({prior_191a.get('record')})"
    elif "FAIL" in (h, fo):
        out["kpf19_1a"] = "FAIL"
    elif h == "PASS" and fo == "PASS" and mnt == "DETECTED":
        out["kpf19_1a"] = "PASS"
    elif h == "PASS" and fo == "PASS" and mnt == "NOT_DETECTED":
        out["kpf19_1a"] = "NOT_ESTABLISHED"
    else:
        out["kpf19_1a"] = "INCONCLUSIVE"
    return out


def decide_174b(cells_by_launch: dict, n_ubatch: int, build: list[int] | None) -> dict:
    """RULES_V2['decisions_v2'] (pure). cells_by_launch: {launch: [(c, group, verdict), ...]}."""
    out: dict = {}
    so = cells_by_launch.get("honest_specoff")
    if so is None:
        out["c1_locus"] = "NOT_RUN"
    else:
        vs = [v for c, _g, v in so if c == 2148]
        if any(v == "FAIL" for v in vs):
            out["c1_locus"] = "target_restore_defect"
        elif vs and all(v == "PASS" for v in vs):
            out["c1_locus"] = "drafter_on_only"
        else:
            out["c1_locus"] = "INCONCLUSIVE"
    sw = cells_by_launch.get("sweep")
    if sw is None:
        out["junction_onset"] = "NOT_RUN"
        out["sweep_2148"] = "NOT_RUN"
        return out
    per_group: dict = {}
    for c, g, v in sw:
        per_group.setdefault(g, {"fail": [], "pass": [], "other": []})[
            "fail" if v == "FAIL" else "pass" if v == "PASS" else "other"].append(c)
    for d in per_group.values():
        for k in d:
            d[k] = sorted(d[k])
    failing = sorted({c for c, _g, v in sw if v == "FAIL"})
    passing = sorted({c for c, _g, v in sw if v == "PASS"} - set(failing))
    if not failing:
        pattern = "none_fail"
    elif all(c % n_ubatch for c in failing) and not any(c % n_ubatch for c in passing):
        pattern = "ubatch_misaligned_only"
    elif build and failing == [min(build)]:
        pattern = "first_build_point_only"
    else:
        pattern = "other"
    out["junction_onset"] = {"per_group": {str(g): d for g, d in sorted(per_group.items())},
                             "failing_junctions": failing, "passing_junctions": passing,
                             "first_failing_junction": failing[0] if failing else None, "pattern": pattern,
                             "n_ubatch": n_ubatch}
    v2148 = [v for c, _g, v in sw if c == 2148]
    out["sweep_2148"] = ("FAIL" if "FAIL" in v2148 else "PASS" if v2148 and all(v == "PASS" for v in v2148)
                         else "INCONCLUSIVE" if v2148 else "NOT_RUN")
    out["not_decided_here"] = RULES_V2["decisions_v2"]["not_decided_here"]
    return out


def analyze(results: dict, geo: dict, prior_191a: dict | None = None, rules: str = "v1") -> dict:
    """results: {launch: {cell_id: cell}} -> verdicts (pure)."""
    out: dict = {"launches": {}, "cells": {}, "spread": {}, "kpf17_4a": [], "rules_sha256": rules_sha256(rules),
                 "rules": rules}
    lv: dict = {}
    per174b: dict = {}
    for launch, cells in results.items():
        spec_off = bool(LAUNCHES.get(launch, {}).get("spec_off"))
        valid = [cell_as_k13(c) for c in cells.values() if c.get("ok")]
        sp = k13.acceptance_spread(valid, spec_off=spec_off)
        out["spread"][launch] = {k: sp[k] for k in ("spread_dn", "spread_da", "n_pooled", "min_k")}
        slack = (sp["spread_dn"], sp["spread_da"])
        judged, scope = [], []
        for cid, c in sorted(cells.items()):
            j = judge_cell(c, slack, spec_off=spec_off)
            if launch in LAUNCHES_174B:
                per174b.setdefault(launch, []).append((c["c"], c["group"], j["verdict"]))
            out["cells"][f"{launch}:{cid}"] = {"group": c["group"], "c": c["c"], "pruned": c.get("pruned"), **j}
            judged.append(j)
            scope.append(bool(c.get("pruned")) if launch == "mut_partial_noop" else True)
            if launch == "honest":
                out["kpf17_4a"].append({"group": c["group"], "c": c["c"], "above_swa_cells": c["c"] >= geo["swa_cells"],
                                        "pruned_in_source": c.get("pruned"), **j["kpf17_4a"]})
        role = LAUNCHES[launch]["role"]
        lv[launch] = launch_verdict(role, judged, scope)
    out["launches"] = lv
    if rules == "v2":
        out["decisions"] = decide_174b(per174b, int(geo.get("n_ubatch") or 512), geo.get("build"))
    else:
        out["decisions"] = decide(lv, prior_191a)
    return out


# ---------------------------------------------------------------- prior KPF-19.1a (resume-skip across rows)
def find_prior_191a(patterns: list[str]) -> dict | None:
    """A complete KPF-19.1a result written by this driver OR by the KPF-18 S1 runner (kpf18-s1-gpu-sweep lists 19.1a
    too): any record with kpf_task containing '19.1a' and decisions.kpf19_1a in (PASS, FAIL). First match wins."""
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                r = json.loads(Path(f).read_text())
            except (OSError, ValueError):
                continue
            if "19.1a" not in str(r.get("kpf_task", "")):
                continue
            v = ((r.get("decisions") or {}).get("kpf19_1a") or "")
            if v in ("PASS", "FAIL"):
                return {"verdict": v, "record": f}
    return None


# ---------------------------------------------------------------- prompt source (the KPF-18 27B prompt set)
class Prompts:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.sha256 = k13.sha256_file(self.path)
        side = self.path.with_name(self.path.name + ".sha256")
        self.sha_ok = (side.read_text().split()[0] == self.sha256) if side.exists() else None
        self.trunks: dict[int, list[int]] = {}
        self.suffixes: dict[int, list[int]] = {}
        self.meta: dict = {}
        for ln in self.path.read_text().splitlines():
            r = json.loads(ln)
            if r.get("kind") == "meta":
                self.meta = r
            elif r.get("kind") == "trunk":
                self.trunks[r["group"]] = r["ids"]
            elif r.get("kind") == "pair" and r.get("suffix_index") == 0:
                self.suffixes[r["group"]] = r["suffix_ids"]

    def problems(self, groups: list[int], L: int, S: int, model: str | None) -> list[str]:
        p = []
        if self.sha_ok is False:
            p.append(f"prompt set sha256 does not match its .sha256 file: {self.path}")
        if model and self.meta.get("model") and os.path.realpath(self.meta["model"]) != os.path.realpath(model):
            p.append(f"prompt set was tokenized for {self.meta['model']}, the argv serves {model}")
        for g in groups:
            if len(self.trunks.get(g, [])) < L:
                p.append(f"group {g}: trunk {len(self.trunks.get(g, []))} < L {L}")
            if len(self.suffixes.get(g, [])) < S:
                p.append(f"group {g}: suffix 0 shorter than S {S}")
        return p


# ---------------------------------------------------------------- the run
class Runner:
    def __init__(self, ctx, geo: dict, prompts: Prompts, launch: str, results: dict) -> None:
        self.ctx, self.geo, self.ps, self.launch = ctx, geo, prompts, launch
        self.results = results.setdefault(launch, {})
        self.srv = None
        self._rows = None
        self.vram_file = None
        self._vstop = threading.Event()

    @property
    def rows(self):
        if self._rows is None:
            self._rows = k13.Jsonl(self.ctx.out / "rows" / f"{self.launch}.jsonl")
        return self._rows

    def pending(self, groups: list[int]) -> list[tuple[int, int]]:
        return [(g, c) for g in groups for c in sorted(self.geo["c"], reverse=True)
                if not (self.results.get(cell_id(g, c)) or {}).get("ok")]

    # -- server
    def start(self) -> None:
        # GPU: the previous launch's own server was verified dead, but KFD returns its VRAM with a lag. k13.Server.start
        # waits (bounded, --vram-drain-timeout-s, logged with the drained value) for the card to drain before its
        # empty-card check; this replaced an unlogged 90 s / 3 s loop here (KPF-13.1b VRAM-drain fix, 2026-10-23).
        self.srv = k13.Server(self.ctx, f"kpf174_{self.launch}", Path(self.ctx.args.fork_bin))
        self.srv.start()
        self.vram_file = self.ctx.out / "vram" / f"{self.srv.label}.txt"
        self.vram_file.parent.mkdir(parents=True, exist_ok=True)
        if getattr(self.srv, "proc", None) is not None:   # 1 Hz own-pid KFD VRAM sampler (the record's residency proof)
            threading.Thread(target=self._vram_loop, daemon=True).start()

    def _vram_loop(self) -> None:
        pid = self.srv.proc.pid
        with open(self.vram_file, "a") as fh:
            while not self._vstop.wait(1.0) and k13.pid_alive(pid):
                v = k13.kfd_pids().get(pid, 0)
                fh.write(f"{time.time():.1f} {v} {k13.card_vram_used(Path(self.ctx.args.card))}\n")
                fh.flush()

    def stop(self) -> None:
        self._vstop.set()
        if self.srv:
            self.srv.stop()

    # -- requests
    def req(self, role: str, ids: list[int], slot: int, *, fork: bool, cache: bool, n_predict: int, ckpt=None,
            meta: dict | None = None) -> dict:
        ctx = self.ctx
        ctx.check_boundary()
        body = k13.body_for(ids, n_predict, slot, fork=fork, cache=cache, ckpt=ckpt)
        r = ctx.client.complete(body, timeout=min(7200.0, ctx.deadline.remaining() + k13.WATCHDOG_GRACE_S - 10))
        t = r.get("timings") or {}
        probs = r.get("probs") or []
        run = {"role": role, "slot": slot, "http": r.get("http"), "error": r.get("error"),
               "tokens": r.get("tokens") or [], "gaps": k13.top2_gaps(probs),
               "lp_greedy": [max(k13.top_probs(e).values()) if k13.top_probs(e) else None for e in probs],
               "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
               "n_fork_tokens": int(t.get("n_fork_tokens") or 0), "fork_src_kind": t.get("fork_src_kind"),
               "fork_src_slot": t.get("fork_src_slot"), "cache_n": int(t.get("cache_n") or 0),
               "prompt_n": int(t.get("prompt_n") or 0), "prompt_ms": t.get("prompt_ms"),
               "predicted_ms": t.get("predicted_ms"), "wall_s": r.get("wall_s")}
        self.rows.add({"t": round(time.time(), 3), "launch": self.launch, **(meta or {}),
                       **{k: v for k, v in run.items() if k not in ("gaps", "lp_greedy")}})
        if r.get("http") != 200 or r.get("error"):
            raise k13.Abort(f"{role}: HTTP {r.get('http')} {r.get('error')}")
        return run

    def erase(self, *slots: int) -> None:
        for s in slots:
            self.ctx.client.erase(s)

    def group(self, g: int) -> None:
        a, geo = self.ctx.args, self.geo
        L, S, cs = geo["L"], a.suffix_tokens, sorted(geo["c"])
        trunk = self.ps.trunks[g][:L]
        suf = self.ps.suffixes[g][:S]
        meta = {"group": g}
        build = list(geo.get("build", cs)) + [L]   # the build sequence: every build point is a pinned prompt END

        def seed(slot: int, upto: int, tag: str, m: dict) -> None:
            for x in build:
                if x > upto:
                    break
                self.req(f"{tag}_{x}", trunk[:x], slot, fork=False, cache=True, n_predict=0, ckpt=[-1],
                         meta={**m, "arm": f"{tag}_{x}"})

        self.erase(0, 1, 2, 3)
        seed(0, L, "source", meta)
        slot0 = [x for x in self.ctx.client.slots() if x.get("id") == 0]
        src_ckpts = [{k: ck.get(k) for k in ("n_tokens", "pos_min", "pos_max", "pinned", "size_mib")}
                     for ck in ((slot0[0] if slot0 else {}).get("checkpoints") or [])]
        for c in sorted(cs, reverse=True):
            if (self.results.get(cell_id(g, c)) or {}).get("ok"):
                continue
            self.ctx.check_boundary()
            P = trunk[:c] + suf
            m = {**meta, "c": c}
            cell = {"cell": cell_id(g, c), "group": g, "c": c, "S": S, "L": L, "pruned": geo["pruned"][c],
                    "launch": self.launch, "source_checkpoints": src_ckpts, "prompt_sha256": k13.sha256_json(P),
                    "arms": [], "controls": [], "fresh": None, "ok": False, "start_utc": k13.now_iso(),
                    "rules_sha256": rules_sha256(self.ctx.args.rules), "build": build,
                    "spec_off": bool(LAUNCHES[self.launch].get("spec_off"))}

            def control(i: int) -> dict:
                self.erase(2)
                seed(2, c, f"C{i}_seed", m)
                r = self.req(f"control{i}", P, 2, fork=False, cache=True, n_predict=N_GEN, meta={**m, "arm": f"C{i}"})
                self.erase(2)
                return r

            def child(i: int) -> dict:
                self.erase(1)
                r = self.req(f"arm{i}", P, 1, fork=True, cache=True, n_predict=N_GEN, meta={**m, "arm": f"X{i}"})
                self.erase(1)
                return r

            cell["arms"].append(child(1))
            cell["controls"].append(control(1))
            cell["arms"].append(child(2))
            for i in range(2, a.control_repeats + 1):
                cell["controls"].append(control(i))
            self.erase(3)
            cell["fresh"] = self.req("fresh", P, 3, fork=False, cache=False, n_predict=N_GEN, meta={**m, "arm": "F"})
            self.erase(3)
            cell["ok"], cell["end_utc"] = True, k13.now_iso()
            self.results[cell["cell"]] = cell
            k13.write_json(self.ctx.out / "cells" / self.launch / f"{cell['cell']}.json", cell)
            print(f"[kpf174]   {self.launch}:{cell['cell']} " + json.dumps(
                {k: [(r["n_fork_tokens"], r["cache_n"], r["prompt_n"], r["draft_n"], r["draft_n_accepted"])
                     for r in v] for k, v in (("X", cell["arms"]), ("C", cell["controls"]))}), flush=True)

    def finish_record(self, ana: dict) -> dict:
        ctx, srv, a = self.ctx, self.srv, self.ctx.args
        vmax = vn = 0
        if self.vram_file and self.vram_file.exists():
            for ln in self.vram_file.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    vmax, vn = max(vmax, int(parts[1])), vn + (int(parts[1]) > 0)
        rec = {
            "schema": SCHEMA, "kpf_task": TASK, "launch": self.launch, "launch_task": LAUNCHES[self.launch]["task"],
            "run_id": f"kpf174-{self.launch}-{k13.utc_tag()}", "arm": LAUNCHES[self.launch]["label"],
            "binary": str(a.fork_bin), "binary_sha256": ctx.bin_sha(Path(a.fork_bin)),
            "argv": srv.argv if srv else None, "launch_prefix": list(ctx.prefix),
            "argv_readback": k13.readback_record(srv.readback) if srv else None,
            "env": srv.env_note if srv else None, "linkage": srv.linkage if srv else None,
            "stop": srv.stop_rec if srv else None,
            "vram_own_pid_peak_bytes": vmax, "vram_nonzero_samples": vn, "vram_file": str(self.vram_file),
            "prompt_set": str(self.ps.path), "prompt_set_sha256": self.ps.sha256, "geometry": self.geo,
            "suffix_tokens": a.suffix_tokens, "groups": groups_of(a, self.launch), "control_repeats": a.control_repeats,
            "n_predict": N_GEN, "seed": SEED, "rules": a.rules, "rules_sha256": rules_sha256(a.rules),
            "spec_off": bool(LAUNCHES[self.launch].get("spec_off")),
            "drafter_flags_removed": getattr(srv, "drafter_removed", None) if srv else None,
            "launch_verdict": (ana.get("launches") or {}).get(self.launch), "spread": (ana.get("spread") or {}).get(self.launch),
            "decisions": ana.get("decisions"), "window_id": a.window_id, "device": a.device, "cpu_list": a.cpu_list,
            "start_utc": ctx.start_utc, "end_utc": k13.now_iso(), "host": ctx.host,
            "cells_dir": str(ctx.out / "cells" / self.launch), "rows": str(self.rows.path),
            "rows_sha256": k13.sha256_file(self.rows.path), "n_cells": len(self.results),
            "instrument_class": "serving", "category": "CANDIDATE",
            "grading_note": "deterministic conformance + mutation; counts, not a pass word; projects under VB-KVU-PF",
        }
        rec = k13.seal_record(rec)   # VB-KVU-PF.5: hash the read-back form (geometry.pruned has int keys)
        k13.write_json(ctx.out / "records" / f"{rec['run_id']}.json", rec)
        k13.Jsonl(ctx.out / "belief_measurements.jsonl").add(rec)
        return rec


def groups_of(a, launch: str) -> list[int]:
    return a.groups if launch in ("honest",) + LAUNCHES_174B else a.other_groups


def cell_id(g: int, c: int) -> str:
    return f"g{g}-c{c}"


def load_results(out: Path) -> dict:
    res: dict = {}
    if (out / "cells").exists():
        for d in sorted((out / "cells").glob("*")):
            for f in sorted(d.glob("*.json")):
                res.setdefault(d.name, {})[f.stem] = json.loads(f.read_text())
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fork-bin", default=None, help="the GPU image carrying the KPF-19.1 hooks (required to run)")
    ap.add_argument("--argv-file", default=str(DEFAULT_ARGV))
    ap.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET)
    ap.add_argument("--groups", default="1,2", help="prompt groups of the honest launch (trunks >= L tokens)")
    ap.add_argument("--other-groups", default="1", help="prompt groups of the flag_on and mutant launches")
    ap.add_argument("--launches", default="honest,mut_partial_noop,flag_on,mut_no_transfer",
                    help="in order; 17.4's pair (honest, mut_partial_noop) first so a short window still decides it")
    ap.add_argument("--control-repeats", type=int, default=MIN_CONTROLS)
    ap.add_argument("--suffix-tokens", type=int, default=64)
    ap.add_argument("--n-swa", type=int, default=2048, help="the drafter's sliding_window (DFlash2 GGUF: 2048)")
    ap.add_argument("--rules", choices=sorted(RULES_BY_VERSION), default="v1",
                    help="v1 = KPF-17.4/17.4a/19.1a (2026-10-23 run); v2 = KPF-17.4b localization (sweep, honest_specoff)")
    ap.add_argument("--junctions", default="",
                    help="KPF-17.4b: comma list of cell junctions (default: the v1 geometry c1,c2,c3); each is also a "
                         "build point unless --build-junctions says otherwise")
    ap.add_argument("--build-junctions", default="",
                    help="KPF-17.4b: comma list of the source's build points (default: --junctions); must contain every "
                         "cell junction. The original c1 geometry is --junctions 2148 --build-junctions 2148,4096,10240")
    ap.add_argument("--prior-191a", default=DEFAULT_PRIOR_191A,
                    help="comma list of record globs; a complete KPF-19.1a there skips flag_on + mut_no_transfer")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--cpu-list", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--numa-membind", type=int, default=None)
    ap.add_argument("--port", type=int, default=18974)
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
    ap.add_argument("--prefill-tok-s", type=float, default=500.0,
                    help="estimate only (27B GPU prefill; KPF-18 S1 rows 2026-10-22: median 649, p10 384 tok/s)")
    ap.add_argument("--decode-tok-s", type=float, default=25.0,
                    help="estimate only (27B GPU + DFlash2 decode; S1 rows: median 37.5, p10 23 tok/s)")
    ap.add_argument("--load-s", type=float, default=150.0, help="estimate only: server load")
    a = ap.parse_args(argv)
    a.groups = [int(x) for x in a.groups.split(",") if x.strip()]
    a.other_groups = [int(x) for x in a.other_groups.split(",") if x.strip()]
    a.launches = [x.strip() for x in a.launches.split(",") if x.strip()]
    a.prior_191a = [x for x in (a.prior_191a or "").split(",") if x.strip()]
    a.junctions = [int(x) for x in a.junctions.split(",") if x.strip()]
    a.build_junctions = [int(x) for x in a.build_junctions.split(",") if x.strip()]
    a.draft_debug = False
    return a


def argv_ub_np(argv: list[str]) -> tuple[int, int]:
    return int(k13.flag_val(argv, ("-ub", "--ubatch-size")) or 512), int(k13.flag_val(argv, ("-np", "--parallel")) or 1)


def preflight(a, ps: Prompts | None, gate_argv: list[str], geo: dict) -> list[str]:
    p = []
    if not (1024 <= a.port <= 65535) or 8000 <= a.port <= 8999:
        p.append(f"--port {a.port} refused (never 8000-8999 or < 1024)")
    if k13.port_listening(a.port):
        p.append(f"--port {a.port} already listening")
    if a.device == "gpu" and k13.port_listening(8083):
        p.append("production :8083 is listening (its owner must stop it first)")
    p += k13.argv_problems(gate_argv, "dflash2", a.device)
    if not a.fork_bin:
        p.append("--fork-bin is required (the GPU image with the KPF-19.1 hooks)")
    elif not Path(a.fork_bin).exists():
        p.append(f"--fork-bin {a.fork_bin} does not exist (build it: see the launcher's build_gpu_kpf191.sh)")
    else:
        for ln in a.launches:
            for kn in LAUNCHES[ln]["knob"]:
                if not k13.bytes_contain_any_lib(Path(a.fork_bin), kn):
                    p.append(f"launch {ln}: knob {kn!r} not compiled into {a.fork_bin}")
    for ln in a.launches:
        if ln not in LAUNCHES:
            p.append(f"unknown launch {ln!r}")
    v2 = [x for x in a.launches if x in LAUNCHES_174B]
    if a.rules == "v2":
        if not v2 or len(v2) != len(a.launches):
            p.append(f"rules v2 (KPF-17.4b) runs only the launches {list(LAUNCHES_174B)} (got {a.launches})")
        if "honest_specoff" in a.launches and geo["c"] != [2148]:
            p.append(f"honest_specoff is the c1 cell only: --junctions 2148 (got {geo['c']})")
    else:
        if v2:
            p.append(f"launches {v2} are KPF-17.4b: --rules v2")
        if "honest" not in a.launches:
            p.append("the honest launch is required (every decision reads it)")
        if a.junctions or a.build_junctions:
            p.append("--junctions / --build-junctions are KPF-17.4b options: --rules v2")
    p += geometry_problems(geo)
    if any(LAUNCHES.get(x, {}).get("spec_off") for x in a.launches):
        off, removed = k13.strip_drafter(gate_argv)
        p += k13.spec_off_problems(off)
        if "-md" not in removed and "--model-draft" not in removed:
            p.append("speculation OFF: the argv file carries no -md to remove (not the DFlash2 profile?)")
    if a.control_repeats < MIN_CONTROLS:
        p.append(f"--control-repeats {a.control_repeats} < {MIN_CONTROLS}")
    if a.fork_min >= min(geo["c"]):
        p.append(f"--fork-min {a.fork_min} >= the smallest junction {min(geo['c'])}")
    if ps is None:
        p.append("no prompt set")
    else:
        p += ps.problems(sorted(set(a.groups + a.other_groups)), geo["L"], a.suffix_tokens,
                         k13.flag_val(gate_argv, ("-m", "--model")))
    if not a.cpu_list:
        p.append("--cpu-list is required (GPU: the claimed host threads)")
    return p


def estimate(a, geo: dict, launches: list[str], pending: dict) -> float:
    pf, dec = a.prefill_tok_s, a.decode_tok_s
    tot = 0.0
    for ln in launches:
        cells = pending.get(ln, [])
        if not cells:
            continue
        tot += a.load_s + 30
        groups = sorted({g for g, _ in cells})
        tot += len(groups) * geo["L"] / pf
        for _g, c in cells:
            req = (N_GEN / dec + a.suffix_tokens / pf + 1.0)
            tot += 2 * req + a.control_repeats * (c / pf + req) + (c + a.suffix_tokens) / pf + N_GEN / dec + 2.0
    return tot


def main(argv=None) -> int:
    a = parse_args(argv)
    gate_argv, comments = k13.load_argv_file(Path(a.argv_file))
    ub, npar = argv_ub_np(gate_argv)
    geo = junctions(a.n_swa, npar, ub, a.junctions or None, a.build_junctions or None)
    out = Path(a.out)
    prior = find_prior_191a(a.prior_191a) if a.rules == "v1" else None
    if a.cmd == "analyze":
        res = load_results(out)
        if not res:
            print(f"no cells under {out / 'cells'}")
            return 2
        ana = analyze(res, geo, prior, a.rules)
        k13.write_json(out / "verdict.json", ana)
        print(json.dumps({k: ana[k] for k in ("launches", "decisions", "spread", "kpf17_4a")}, indent=1))
        return 0
    a.window_id = a.window_id or f"kpf174-{k13.utc_tag()}"
    try:
        ps = Prompts(Path(a.prompt_set))
    except (OSError, ValueError, KeyError) as exc:
        print(f"prompt set unreadable: {exc}")
        ps = None
    launches = [x for x in a.launches if not (prior and x in LAUNCHES_191A)]
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
    pending = {ln: Runner(ctx, geo, ps, ln, results).pending(groups_of(a, ln)) if ps else [] for ln in launches}
    est = estimate(a, geo, launches, pending)
    print("=" * 100)
    print(f"KPF-17.4/17.4a/19.1a SWA-drafter fork run  device={a.device}  rules={a.rules} "
          f"rules_sha256={rules_sha256(a.rules)[:16]}  controls/cell={a.control_repeats}")
    print(f"geometry: n_swa {geo['n_swa']}, -np {npar}, -ub {ub} -> swa_cells {geo['swa_cells']}; L {geo['L']}; "
          f"junctions {geo['c']} (pruned in source: {geo['pruned']}); build points {geo.get('build', geo['c'])} + L; "
          f"S {a.suffix_tokens}; groups {a.groups}")
    for ln in a.launches:
        if LAUNCHES.get(ln, {}).get("spec_off"):
            print(f"launch {ln}: speculation OFF, drafter flags removed: {k13.strip_drafter(gate_argv)[1]}")
    if prior:
        print(f"KPF-19.1a PRIOR found ({prior}): skipping {list(LAUNCHES_191A)}")
    for ln in launches:
        print(f"--- launch {ln} ({LAUNCHES[ln]['label']}: mutate={LAUNCHES[ln]['mutate']}, extra={LAUNCHES[ln]['extra']}): "
              f"{len(pending[ln])} pending cells {pending[ln]}"
              + ("  [already complete: no load]" if not pending[ln] else ""))
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
    k13.write_json(out / "rules.json", {"rules": RULES_BY_VERSION[a.rules], "rules_version": a.rules,
                                        "rules_sha256": rules_sha256(a.rules),
                                        "cell_judge_rules_sha256_kpf13_v3": k13.rules_sha256("v3")})
    k13.write_json(out / "run_meta.json", {
        "args": vars(a), "estimate_s": est, "started": k13.now_iso(), "rules": a.rules,
        "rules_sha256": rules_sha256(a.rules),
        "rules_declared_before_first_launch": True, "geometry": geo, "launches": launches, "prior_191a": prior,
        "argv_file": a.argv_file, "argv_file_sha256": k13.sha256_file(Path(a.argv_file)), "argv_comments": comments,
        "prompt_set_sha256": ps.sha256, "launch_prefix": list(ctx.prefix),
        "device_claim": a.no_device_claim or ("mi210_0" if a.device == "gpu" else "n/a"),
        "region_lock": {"KPF13_REGION": os.environ.get("KPF13_REGION"), "skipped": a.no_region_lock}})
    dog = k13.Watchdog(ctx)
    claim_cm = None
    runners: list[Runner] = []

    def cleanup(*_x):
        for r in runners:
            try:
                r.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf174] cleanup of own pid failed: {exc}", flush=True)

    def on_sig(signum, _f):
        cleanup()
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
                    "mi210_0", purpose=f"KPF-17.4/19.1a SWA-drafter fork on :{a.port}", campaign_id="KPF-17.4",
                    journal=dc.ClaimJournal(out / "claim" / "device-claim-journal.jsonl"), holder_label="kpf174",
                    timeout_s=0, max_hold_s=(deadline.remaining() + 600))
            except dc.DeviceClaimError as exc:
                print(f"[kpf174] REFUSED: mi210_0 device claim not acquired: {exc}", flush=True)
                return 75
            claim_cm.__enter__()
        dog.start()
        for ln in launches:
            r = Runner(ctx, geo, ps, ln, results)
            todo = r.pending(groups_of(a, ln))
            if not todo:
                print(f"[kpf174] launch {ln}: all cells stored, skipped (resume)", flush=True)
                continue
            per_launch = estimate(a, geo, [ln], {ln: todo})
            if a.stop_file and Path(a.stop_file).exists():
                raise k13.Abort(f"stop file {a.stop_file} present: draining before launch {ln}")
            if not deadline.fits(per_launch):
                raise k13.Abort(f"deadline: launch {ln} est {k13.fmt_s(per_launch)} does not fit; draining")
            runners.append(r)
            print(f"[kpf174 {time.strftime('%H:%M:%SZ', time.gmtime())}] launch {ln} (est {k13.fmt_s(per_launch)})",
                  flush=True)
            try:
                r.start()
                for g in sorted({g for g, _ in todo}):
                    if a.stop_file and Path(a.stop_file).exists():
                        raise k13.Abort("stop file present: draining at a group boundary")
                    r.group(g)
            finally:
                r.stop()
                try:
                    r.finish_record(analyze(load_results(out), geo, prior, a.rules))
                except Exception as exc:  # noqa: BLE001
                    print(f"[kpf174] record for {ln} failed: {exc}", flush=True)
    except k13.Abort as exc:
        print(f"[kpf174] ABORT: {exc}", flush=True)
        rc = 5 if ("deadline" in str(exc) or "stop file" in str(exc)) else 3
    except Exception as exc:  # noqa: BLE001
        print(f"[kpf174] ERROR: {type(exc).__name__}: {exc}", flush=True)
        rc = 4
    finally:
        cleanup()
        dog.stop()
        if claim_cm is not None:
            try:
                claim_cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf174] claim release failed: {exc}", flush=True)
        try:
            res = load_results(out)
            if res:
                ana = analyze(res, geo, prior, a.rules)
                k13.write_json(out / "verdict.json", ana)
                print("[kpf174] verdict: " + json.dumps(ana["decisions"]), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[kpf174] analyze failed: {exc}", flush=True)
        alive = [r.srv.proc.pid for r in runners if r.srv and r.srv.proc and k13.pid_alive(r.srv.proc.pid)]
        if alive:
            print(f"[kpf174] WARNING: own child pid(s) still alive: {alive}", flush=True)
            rc = rc or 4
    print(f"[kpf174] exit rc={rc} (0 ok, 2 refused, 3 abort, 4 error, 5 deadline/stop drain, 75 claim held)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
