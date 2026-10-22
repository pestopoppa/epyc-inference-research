#!/usr/bin/env python3
"""KVU-12.1 / KVU-12.3 hybrid-checkpoint policy arms for the CPU np x ctx study (workspace-f8, 2026-10-21).

What this adds to the KVU-12 driver (`kvu_cpu_matrix.py`, imported, NOT modified):
  KVU-12.1  four arms on one conversation workload: dense (today), tail_junction, fixed-interval (+cap), coarse grid.
  KVU-12.3  a decode-boundary arm (snapshot every N generated tokens) vs dense on a generation-then-resume workload.
Class of every number: counts / prompt_n / bytes = CORRECTNESS (build-role claim, any quiet-host state).  TTFT and
decode tok/s = INDICATIVE timing (operator ruling 2026-10-21: only a champion BASELINE needs a quiet host; this is not one).

Design points (all predeclared in PLAN.md section 11; do not edit them after the first cell starts):
  * The unit of work is a CELL = (model, arm, L, wave) [KVU-12.1] or (model, arm, G, wave) [KVU-12.3]. A cell is
    self-contained (its own cold build), so it can be skipped, resumed or abandoned at a STOP file without losing state.
  * Filler contract: `--stop-file` is honoured between cells (a started cell is finished), `--deadline-epoch` stops new cells
    that would not fit (est x calibrated factor + margin), per-cell resume reads `cells.jsonl` (result OK = done).
  * Every request is a per-request row through the shared VB-KVU-CPU emitter (`kvu_cpu_matrix.finalize_row`).
  * Checkpoint positions/bytes are the ones the SERVER materialized: replayed from its own log lines (`-lv 4`), never
    inferred from the policy. The policy simulator (`simulate_*`) only prices the plan and cross-checks the log.
  * Nothing here builds llama.cpp. The decode-boundary arm needs a server patch (PLAN.md section 11.5); until the capability
    probe passes the arm is recorded BLOCKED_ON_PATCH and everything else keeps running.
"""
import argparse, dataclasses, hashlib, json, math, os, re, statistics, subprocess, sys, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kvu_cpu_matrix as M                                     # noqa: E402  (driver reused, never edited here)

SCHEMA = "epyc.kvu121.ckpt_arms.v1"
CELLS_FILE, REQS_FILE = "cells.jsonl", "requests.jsonl"

# ----------------------------------------------------------------------------------------------- predeclared constants
# ---- Estimate model (REPLACES the single 1.72 factor; coordinator ruling 2026-10-22 after /mnt/raid0/llm/tmp/d4-why-20261022/REPORT.md).
# The shared driver's prefill_tps(L) = 300*(1-0.35*L/131072) (195 tok/s at 128k) was calibrated on 2.7k prompts; D4 measured ~125 tok/s at 64.6k and ~65 at 130k,
# and decode 0.08-0.12 tok/s/stream while sharing batches with prefill. One factor taken from the cheapest shape (np4 L64k, 1723/1000) hid actual/estimate
# ratios from 0.29 (np4 t8 anchors) to >=3.1 (np8 L128k, censored). So: cumulative cold-prefill time T(n) is interpolated log-log through MEASURED points
# (below), a resume of n tokens from checkpoint position p costs T(n)-T(p) (the depth tax is inside T), and the correction is per (model, kind, unit) key,
# learned from this run's own cells (Calibrator); an unseen key carries SAFETY_UNSEEN.
PREFILL_POINTS = (   # (tokens, cold-prefill seconds at 96 threads, one stream prefilling), provenance
    (2700, 2700 / 298.0, "frontdoor live log ~298 tok/s @2.7k (driver prefill_tps comment); older, single point"),
    (64595, 516.0, "D4 wave-0 TTFT increments 513/518/512 s for 64,595-token prompts (k4_A/k4_B/k5 waves.json: 125-126 tok/s; later streams 109-115)"),
    (130112, 2000.0, "D4 k5_s4 server.log: 130k prompts 1838-2145 s (65-71 tok/s), serial, REPORT.md section 2a"),
)
D4_RATIOS = {   # actual/driver-estimate per shape in the 2026-10-21 window (REPORT.md section 3): why a single factor cannot work
    "np4_L64k": (1.72, 1.81), "np8_L64k_unified": 2.19, "np8_L64k_split": 2.02, "np4_t8_anchor": 0.29, "bank_build": 1.45, "mtpoff": 1.89, "np8_L128k_s4": ">=3.1",
}
# measured aggregate decode tok/s per stream while OTHER slots prefill (shared batches). np=1 has no sharing; its 40 tok/s is ASSUMED (driver AGG_DECODE[1]).
DECODE_SHARED_MEASURED = {(4, 65536): 6.8, (8, 65536): 1.5, (8, 131072): 0.10}
DECODE_NP1_ASSUMED = 40.0
LOAD_WARM_S = {"q36_35b": 20.0, "q38_27b": 60.0}   # 35B: D4 step wall minus cell wall = 16-19 s (page cache warm); 27B ASSUMED cold
SAFETY_UNSEEN = 1.25           # applied to a (model, kind, unit) key this run has not measured yet; replaced by the measured actual/est ratio after its first cell
EST_MARGIN_S = 90              # filler protocol: stop starting new items 90 s before the hard end
DENSE_MIN_STEP, DENSE_CTX = 8192, 32          # v10/champion defaults (common/common.h:633-634 per UFH14-A7); passed EXPLICITLY so the arm is unambiguous
HUGE_STEP = 1 << 30            # --checkpoint-min-step that disables min-step spaced checkpoints (tail_junction)
COARSE_I = 16384               # coarse grid: stride 2x dense, so resume is at most I tokens earlier than dense
FIXED_CANDIDATES = ((4096, 8), (8192, 8), (16384, 8))    # (stride, cap) candidates evaluated at L<=32k; ONE is selected for holdout
TURN_USER_TOK, TURN_ASST_TOK = 1536, 512      # one turn pair ~ 2,048 tokens
PROBE_NPREDICT = 16            # resume probes are prefill-bound; 16 generated tokens give TTFT without decode noise
PROBE_BACK_TOK = (2048, 8192, 16384)          # edit depth measured back from the end of the conversation; plus "half" (L/2)
UBATCH = 2048
HEADROOM_TOK = 8192            # -c above the largest L of a launch (extend probe + edits + slack)
GEN_G = (1024, 4096)           # KVU-12.3 generated lengths
GEN_FRACS = (0.95, 0.5, 0.1)   # divergence points inside the generated text, probed deep -> shallow (A7: 13/10/12 of 35 diverged first-third/middle/last-10%)
GEN_EVERY, GEN_CAP = 128, 32   # decode-boundary arm: snapshot every 128 generated tokens, at most 32 per sequence
GEN_BASE_TOK = 4096            # KVU-12.3 turn-1 prompt length
GEN_SLACK_TOK = 4              # MTP verify rounds end up to draft_n_max (=4) tokens past a boundary
GEN_SUFFIX_TOK = 128           # fresh user text appended after the edited generation in turn 2
WAVES_DEFAULT = 3
WAVES_CAP_BY_L = {"128k": 2}      # a 128k cell is ~3 h at a quarter: two waves (the rule needs >= 2 for a spread)
LS_ARM = {"8k": 8192, "32k": 32768, "64k": 65536, "128k": 131072}
STAGES = ("select", "holdout", "gen")
# P1 fork build (checkpoint_at, pinned junctions). Resolve its mtime against the branch HEAD commit date before trusting it (CLAUDE.md build-artifact rule).
FORK_BIN = "/mnt/raid0/llm/worktrees/kv-prefix-fork-p1r2-20261020/build-cpu/bin/llama-server"

MODELS = {
    "q36_35b": dict(path=M.MODEL, mtp=True, prefill_scale=1.0, decode_tps=40.0, ram_gib=45, kv_b_tok=M.KV_BYTES_PER_TOKEN,
                    lens_select=("8k", "32k"), lens_holdout=("64k", "128k"), gen_G=GEN_G,
                    note="frontdoor Qwen3.6-35B-A3B Q8 MTP, production template; prefill/decode rates are the driver's calibrated constants"),
    "q38_27b": dict(path="/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf", mtp=False, prefill_scale=0.15, decode_tps=9.0, ram_gib=60, kv_b_tok=21.0e3,
                    lens_select=("8k",), lens_holdout=(), gen_G=(1024,),
                    note="ASSUMPTIONS: dense 27B prefill ~ active-param ratio 3.3/27 of the A3B rate (likely pessimistic by up to 2x; the first cell recalibrates); "
                         "decode ~ 446.8 GB/s / 28.6 GB x ~0.6. Only L=8k and G=1024 are planned for the 27B by default (a 32k cold build is ~1 h at a quarter); "
                         "add more with --lens-select / --lens-holdout / --gen-g"),
}


# ----------------------------------------------------------------------------------------------- arms
@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    kind: str                       # "ckpt" (KVU-12.1 conversation workload) | "gen" (KVU-12.3 generation-then-resume)
    min_step: int = DENSE_MIN_STEP
    ctx_ckpts: int = DENSE_CTX
    junction: bool = False          # request carries checkpoint_at [{"message": -1, "at": "start"}] (turn boundary) every request
    decode_every: int = 0
    decode_cap: int = 0
    needs: tuple = ()               # server capabilities beyond the champion: "checkpoint_at" (fork build), "decode_ckpt" (patch)
    what: str = ""

    def server_extra(self):
        # decode-boundary parameters travel in the REQUEST (request_extra), never as server flags: an unpatched binary must reach the
        # capability probe and be recorded BLOCKED_ON_PATCH instead of dying on an unknown flag.
        return ("--ctx-checkpoints", str(self.ctx_ckpts), "--checkpoint-min-step", str(self.min_step), "-lv", "4")

    def request_extra(self):
        x = {}
        if self.junction:
            x["checkpoint_at"] = [{"message": -1, "at": "start"}]
        if self.decode_every:
            x["checkpoint_decode_every"] = self.decode_every
            x["checkpoint_decode_cap"] = self.decode_cap
        return x


def fixed_arm(stride, cap):
    return Arm(f"fixed_s{stride // 1024}k_c{cap}", "ckpt", min_step=stride, ctx_ckpts=cap,
               what=f"fixed interval ~{stride} tokens (--checkpoint-min-step; snapped to the next user-message start), capped at {cap} per slot")


ARM_DENSE = Arm("dense", "ckpt", what="today: user-message-start checkpoints >= 8192 apart + last user message + n-4-n_ubatch and n-4 (ctx-checkpoints 32)")
ARM_TAIL = Arm("tail_junction", "ckpt", min_step=HUGE_STEP, ctx_ckpts=DENSE_CTX, junction=True, needs=("checkpoint_at",),
               what="tail (n-4, n-4-n_ubatch) + ONE pinned junction at the last turn boundary; min-step spacing disabled. Realised count is "
                    "recorded; exactly <=2 needs patch P-B (PLAN 11.5)")
ARM_COARSE = Arm("coarse", "ckpt", min_step=COARSE_I, ctx_ckpts=DENSE_CTX, what=f"coarser grid: stride {COARSE_I} (2x dense), resume at most {COARSE_I} tokens earlier than dense")
GEN_DENSE = Arm("gen_dense", "gen", what="generation then divergent turn 2; checkpoints at prompt time only (today)")
GEN_DECODE = Arm("gen_decode128", "gen", decode_every=GEN_EVERY, decode_cap=GEN_CAP, ctx_ckpts=DENSE_CTX + GEN_CAP, needs=("decode_ckpt",),
                 what=f"recurrent+drafter snapshot every {GEN_EVERY} generated tokens, cap {GEN_CAP} per sequence (needs server patch P-A)")


def arms_for_stage(stage, selection_for_model=None, fixed=FIXED_CANDIDATES):
    if stage == "select":
        return [ARM_DENSE, ARM_TAIL, ARM_COARSE] + [fixed_arm(s, c) for s, c in fixed]
    if stage == "holdout":
        out = [ARM_DENSE, ARM_TAIL, ARM_COARSE]
        if selection_for_model:
            s, c = selection_for_model
            out.append(fixed_arm(s, c))
        return out
    if stage == "gen":
        return [GEN_DENSE, GEN_DECODE]
    raise ValueError(stage)


def arm_by_name(name):
    for a in (ARM_DENSE, ARM_TAIL, ARM_COARSE, GEN_DENSE, GEN_DECODE):
        if a.name == name:
            return a
    m = re.fullmatch(r"fixed_s(\d+)k_c(\d+)", name)
    if m:
        return fixed_arm(int(m.group(1)) * 1024, int(m.group(2)))
    raise KeyError(name)


# ----------------------------------------------------------------------------------------------- server argv
def arm_argv(model, arm, c, port, threads, binary):
    """Production frontdoor argv with ONLY these bench deltas: -np 1 split slot of c tokens, --cache-ram 0 (isolates checkpoint resume from
    prompt-cache reload), --no-cache-idle-slots (driver default), model path, the arm's checkpoint flags, -lv 4 (the server logs checkpoint
    events at trace level). Binary = the fork build (checkpoint_at lives there)."""
    a = M.prod_argv(1, c, False, MODELS[model]["mtp"], port, threads, binary, extra=arm.server_extra())
    a[a.index("-m") + 1] = MODELS[model]["path"]
    a[a.index("--cache-ram") + 1] = "0"
    return a


def slot_ctx(lens):
    return max(LS_ARM[l] for l in lens) + HEADROOM_TOK


# ----------------------------------------------------------------------------------------------- server checkpoint events (replayed from the server log)
_PFX = r"(?:id\s+(?P<slot>\d+) \| task (?P<task>-?\d+) \| )?"
_F = r"\(pos_min = (?P<pmin>-?\d+), pos_max = (?P<pmax>-?\d+)(?:, n_tokens = (?P<n>\d+))?[^)]*?size = (?P<mib>[\d.]+) MiB(?P<flags>[^)]*)\)"
EV_PATTERNS = (
    ("created", re.compile(_PFX + r"created context checkpoint (?P<idx>\d+) of (?P<max>\d+) " + _F)),
    ("erased_close", re.compile(_PFX + r"erasing context checkpoint too close to an earlier one " + _F)),
    ("erased_old", re.compile(_PFX + r"erasing old context checkpoint " + _F)),
    ("erased_invalid", re.compile(_PFX + r"erased invalidated context checkpoint " + _F)),
    ("restored", re.compile(_PFX + r"restored context checkpoint \(pos_min = (?P<pmin>-?\d+), pos_max = (?P<pmax>-?\d+), n_tokens = (?P<n>\d+), n_past = (?P<npast>\d+), size = (?P<mib>[\d.]+) MiB\)")),
    ("full_reprocess", re.compile(_PFX + r"forcing full prompt re-processing")),
)


def parse_ckpt_events(text):
    """Server-log text -> ordered list of checkpoint events. n_tokens falls back to pos_max+1 when the build does not print it."""
    out = []
    for ln, line in enumerate(text.splitlines()):
        for kind, rx in EV_PATTERNS:
            m = rx.search(line)
            if not m:
                continue
            d = m.groupdict()
            ev = dict(kind=kind, line=ln, slot=int(d["slot"]) if d.get("slot") is not None else None)
            if kind == "full_reprocess":
                out.append(ev); break
            ev["mib"] = float(d["mib"])
            ev["pos_max"] = int(d["pmax"])
            ev["n_tokens"] = int(d["n"]) if d.get("n") else ev["pos_max"] + 1
            if kind == "created":
                fl = d.get("flags") or ""
                ev["pinned"] = "pinned" in fl
                ev["decode"] = "decode" in fl
            if kind == "restored":
                ev["n_past"] = int(d["npast"])
            out.append(ev); break
    return out


class CkptSet:
    """Retained-checkpoint set replayed from events (one sequence = one slot). Never trusts the policy, only what the server logged."""
    def __init__(self):
        self.live = {}                       # n_tokens -> dict(mib, pinned, decode)

    def apply(self, events):
        created = 0
        for ev in events:
            k = ev["kind"]
            if k == "created":
                self.live[ev["n_tokens"]] = dict(mib=ev["mib"], pinned=ev.get("pinned", False), decode=ev.get("decode", False)); created += 1
            elif k in ("erased_close", "erased_old", "erased_invalid"):
                self.live.pop(ev["n_tokens"], None)
        return created

    def positions(self):
        return sorted(self.live)

    def summary(self):
        ps = self.positions()
        mibs = [self.live[p]["mib"] for p in ps]
        return dict(count=len(ps), positions=ps, bytes_total_mib=round(sum(mibs), 3), mib_each=[round(x, 3) for x in mibs],
                    mib_per_ckpt_med=(round(statistics.median(mibs), 3) if mibs else None),
                    pinned=sum(1 for p in ps if self.live[p]["pinned"]), decode=sum(1 for p in ps if self.live[p]["decode"]),
                    gaps=[b - a for a, b in zip(ps, ps[1:])], max_gap=(max([b - a for a, b in zip(ps, ps[1:])]) if len(ps) > 1 else None))


def resume_summary(events):
    """What the server did at the start of a request: the restore (or full re-process) it chose."""
    r = [e for e in events if e["kind"] == "restored"]
    if r:
        return dict(restored_n_tokens=r[-1]["n_tokens"], restored_n_past=r[-1]["n_past"], full_reprocess=False)
    return dict(restored_n_tokens=None, restored_n_past=None, full_reprocess=any(e["kind"] == "full_reprocess" for e in events))


# ----------------------------------------------------------------------------------------------- policy simulator (prices the plan; cross-checks the log)
def simulate_prefill(starts, n_total, arm, held=(), ubatch=UBATCH, from_pos=0, id_task=1):
    """Checkpoints a request would create while prefilling tokens [from_pos, n_total), following server-context.cpp pre_decode:
    first user-start while the set is empty, user starts more than min_step past the newest, the last user message, n-4-n_ubatch and n-4,
    plus the pinned junction. `held` = surviving (n_tokens, pinned, id_task) from earlier requests. Returns the retained list after the
    min-step closeness and capacity eviction of create_checkpoint. APPROXIMATE (chunk boundaries are modelled at checkpoint positions)."""
    held = [tuple(h) for h in held if h[0] <= from_pos]
    last_user = max((s for s in starts if 0 < s < n_total), default=None)
    want = []
    back = max([h[0] for h in held], default=None)
    for s in sorted(x for x in starts if 0 < x < n_total and x >= from_pos):
        if back is None or s > back + arm.min_step or s == last_user:
            want.append((s, False)); back = s
    for off in (4 + ubatch, 4):
        p = n_total - off
        if p > from_pos:
            want.append((p, False))
    if arm.junction and last_user is not None:
        want.append((last_user, True))
    cks = list(held)
    for p, pinned in sorted(set(want)):
        if any(c[0] == p for c in cks):
            cks = [(c[0], c[1] or pinned, c[2]) if c[0] == p else c for c in cks]
            continue
        last, keep = None, []
        for c in sorted(cks):                                  # evict unpinned older checkpoints within min_step of an earlier one
            if (not c[1]) and c[2] != id_task and last is not None and c[0] <= last + arm.min_step:
                continue
            keep.append(c); last = c[0]
        cks = keep
        while len(cks) >= arm.ctx_ckpts:
            old = [c for c in sorted(cks) if not c[1]]
            if not old:
                break
            cks.remove(old[0])
        cks.append((p, pinned, id_task))
    return sorted(cks)


def resume_pos(ckpts, lcp):
    """Server restore rule: newest checkpoint with n_tokens <= LCP (else full re-process from 0)."""
    ok = [c[0] for c in ckpts if c[0] <= lcp]
    return max(ok) if ok else 0


def probe_plan(L_tok, n_total):
    """Resume probes in execution order (shallow -> deep): extend_1 is the CONTROL (exact extension, live state, no checkpoint needed), then
    edits at predeclared depths back from the end (these are the ones that need a checkpoint: a re-rendered history that diverges mid-prompt)."""
    plan = [("extend_1", None)]
    backs = sorted({b for b in PROBE_BACK_TOK if b < 0.9 * L_tok} | {L_tok // 2})
    for b in backs:
        plan.append((f"edit_back_{b}", b))
    return plan


def message_starts(n_msgs, tok_user=TURN_USER_TOK, tok_asst=TURN_ASST_TOK, overhead=8):
    """Approximate token start of each message for a conversation of alternating user/assistant messages (user first)."""
    out, pos = [], 0
    for i in range(n_msgs):
        out.append(pos)
        pos += (tok_user if i % 2 == 0 else tok_asst) + overhead
    return out


def cold_prefill_s(n, model="q36_35b", threads=96):
    """Cold prefill seconds for n tokens: log-log interpolation of t(n) through PREFILL_POINTS (linear below the first point, last segment's exponent
    above the last). Threads scale linearly and the 27B by MODELS[...]['prefill_scale'] (both ASSUMED; only the 35B at 96 threads is measured)."""
    if n <= 0:
        return 0.0
    pts = [(a, b) for a, b, _ in PREFILL_POINTS]
    if n <= pts[0][0]:
        t = pts[0][1] * n / pts[0][0]
    else:
        i = 0 if n <= pts[1][0] else 1
        (n0, t0), (n1, t1) = pts[i], pts[i + 1]
        t = t0 * math.exp(math.log(t1 / t0) / math.log(n1 / n0) * math.log(n / n0))
    return t / MODELS[model]["prefill_scale"] * 96.0 / threads


def prefill_increment_s(n_from, n_to, model="q36_35b", threads=96):
    """Seconds to prefill tokens [n_from, n_to) on top of an existing n_from-token state (resume from a checkpoint): T(n_to) - T(n_from)."""
    return max(0.0, cold_prefill_s(n_to, model, threads) - cold_prefill_s(n_from, model, threads))


def assumptions(model, threads, kind):
    """What the estimate rests on that D4 did NOT measure (printed with the plan)."""
    a = []
    if model != "q36_35b":
        a.append(f"{model}: prefill scale {MODELS[model]['prefill_scale']} x the 35B curve and load {LOAD_WARM_S[model]:.0f} s are ASSUMED")
    if threads != 96:
        a.append(f"{threads} threads: prefill time scaled by 96/{threads} (linear) is ASSUMED; D4 ran at 96")
    if kind == "gen":
        a.append(f"np1 decode {MODELS[model]['decode_tps']:.0f} tok/s ({model}) and the snapshot cost 0.4 s are ASSUMED (D4 measured only decode shared with prefill: {DECODE_SHARED_MEASURED})")
    a.append("np=1 serial prefill is the D4 wave-0 shape (measured); depth-resume cost = T(n)-T(checkpoint), the depth tax is read off the measured curve")
    return a


def est_ckpt_cell(model, arm, Lname, threads=96, safety=SAFETY_UNSEEN):
    """Seconds (raw model, planned = raw x safety for an unseen key) for one KVU-12.1 cell: cold build + the probe plan priced by the policy simulator."""
    L = LS_ARM[Lname]
    mod = MODELS[model]
    inc = lambda a, b: prefill_increment_s(a, b, model, threads)
    pair = TURN_USER_TOK + TURN_ASST_TOK + 16
    n_msgs = max(3, 2 * round(L / pair) + 1)
    starts = message_starts(n_msgs)
    n_total = starts[-1] + TURN_USER_TOK
    ck = simulate_prefill([s for i, s in enumerate(starts) if i % 2 == 0], n_total, arm)
    t = cold_prefill_s(n_total, model, threads) + 2.0
    prev_end = n_total
    n_new = n_total + pair
    ustarts = [s for i, s in enumerate(starts) if i % 2 == 0] + [n_total + TURN_ASST_TOK + 16]
    rp = prev_end                                   # extend_1 = CONTROL: the new prompt extends the cached tokens exactly, so the live state is used (A7: 179/230 turns)
    t += inc(rp, n_new) + 2.0
    ck = simulate_prefill(ustarts, n_new, arm, held=[c for c in ck if c[0] <= rp], from_pos=rp, id_task=2)
    for name, back in probe_plan(L, n_new)[1:]:
        lcp = max(1, n_new - back)
        rp = resume_pos(ck, lcp)
        t += inc(rp, n_new) + 2.0
        ck = simulate_prefill(ustarts, n_new, arm, held=[c for c in ck if c[0] <= rp], from_pos=rp, id_task=3)
    return t, t * safety


def est_gen_cell(model, arm, G, threads=96, safety=SAFETY_UNSEEN):
    """Seconds (raw, planned) for one KVU-12.3 cell: base prefill + G decoded tokens (+snapshot cost) + three divergent turn-2 probes (deep -> shallow).
    np=1, nothing shares the batch, so decode is the ASSUMED unshared rate (MODELS[...]['decode_tps'], 35B 40 tok/s)."""
    mod = MODELS[model]
    dec = mod["decode_tps"] * min(1.0, threads / 96.0 + 0.25)
    t = cold_prefill_s(GEN_BASE_TOK, model, threads) + G / dec
    if arm.decode_every:
        t += (G / arm.decode_every) * 0.4                      # ASSUMPTION: ~0.4 s per recurrent+drafter snapshot (memcpy of ~100 MiB), measured by the first cell
    n2 = GEN_BASE_TOK + G + GEN_SUFFIX_TOK
    for f in GEN_FRACS:
        g = int(G * f)
        if arm.decode_every:
            rp = GEN_BASE_TOK + (g // arm.decode_every) * arm.decode_every
        else:
            rp = GEN_BASE_TOK - 4
        t += prefill_increment_s(rp, n2, model, threads) + PROBE_NPREDICT / dec + 2.0
    t += 5.0
    return t, t * safety


# ----------------------------------------------------------------------------------------------- plan
@dataclasses.dataclass
class Cell:
    model: str
    arm: str
    kind: str
    unit: str            # "8k".. for ckpt, "G1024".. for gen
    wave: int
    est_raw: float
    est_cal: float
    bin_tag: str = ""

    @property
    def key(self):
        """Resume identity. Includes the binary tag, so a cell measured on one binary is never skipped for another (the 12.3 dense baseline
        must be re-run on the patched build)."""
        return f"{self.model}|{self.arm}|{self.kind}|{self.unit}|w{self.wave}|b{self.bin_tag}"


@dataclasses.dataclass
class Launch:
    id: str
    model: str
    arm: str
    stage: str
    c: int
    cells: list
    needs: tuple
    load_s: float
    binary: str = ""         # "" = no binary exists for this launch (arm blocked); never launched

    def est_cal(self):
        return self.load_s + sum(c.est_cal for c in self.cells)


def load_estimate(model):
    return LOAD_WARM_S[model]


def bin_tag(path):
    """Cheap, stable identity of a binary for resume keys (rows carry the real sha256)."""
    try:
        rp = os.path.realpath(path); st = os.stat(rp)
        return hashlib.sha1(f"{rp}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:8]
    except OSError:
        return "missing"


def build_plan(models, stages, waves=WAVES_DEFAULT, threads=96, selection=None, lens_override=None, only_arms=None, cal=SAFETY_UNSEEN, bins=None, gen_g=None):
    """Ordered launches. Order = stage (select, holdout, gen) -> model -> arm; cells inside a launch = L ascending, waves inner.
    `selection` = {model: (stride, cap)} from the select stage (holdout refuses to plan a fixed arm without it).
    `bins` = {"ckpt": fork build (KVU-12.1 arms and, unless "gen" is given, the gen dense baseline), "gen": patched build for KVU-12.3}.
    The decode-boundary arm has NO binary until "gen" is given (then it is BLOCKED_ON_PATCH without launching a server)."""
    bins = bins or {"ckpt": FORK_BIN, "gen": ""}
    out = []
    for st in stages:
        for model in models:
            lo = ((lens_override or {}).get(st) or {}).get(model)
            lens = tuple(lo) if lo is not None else (MODELS[model]["lens_select"] if st == "select" else MODELS[model]["lens_holdout"])
            if st in ("select", "holdout") and not lens:
                continue
            for arm in arms_for_stage(st, (selection or {}).get(model)):
                if only_arms and arm.name not in only_arms:
                    continue
                cells = []
                if st == "gen":
                    binary = bins.get("gen") or ("" if "decode_ckpt" in arm.needs else bins["ckpt"])
                    tag = bin_tag(binary) if binary else "none"
                    Gs = (gen_g or {}).get(model) or MODELS[model]["gen_G"]
                    for G in Gs:
                        raw, cal_s = est_gen_cell(model, arm, G, threads, cal)
                        for w in range(waves):
                            cells.append(Cell(model, arm.name, "gen", f"G{G}", w, raw, cal_s, tag))
                    c = GEN_BASE_TOK + max(Gs) + GEN_SUFFIX_TOK + HEADROOM_TOK
                else:
                    binary = bins["ckpt"]; tag = bin_tag(binary)
                    for l in lens:
                        raw, cal_s = est_ckpt_cell(model, arm, l, threads, cal)
                        for w in range(min(waves, WAVES_CAP_BY_L.get(l, waves))):
                            cells.append(Cell(model, arm.name, "ckpt", l, w, raw, cal_s, tag))
                    c = slot_ctx(lens)
                out.append(Launch(f"{st}__{model}__{arm.name}", model, arm.name, st, c, cells, arm.needs, load_estimate(model), binary))
    return out


def print_plan(launches, a):
    """Per launch: per-cell raw model estimate and planned (raw x safety for an unseen key), then totals; flags any cell that cannot fit --cap-s."""
    tot = tot_raw = 0.0
    cap = getattr(a, "cap_s", 0) or 0
    models = sorted({lc.model for lc in launches})
    for lc in launches:
        arm = arm_by_name(lc.arm)
        print(f"\n== launch {lc.id}  model={lc.model} arm={lc.arm} -c {lc.c} np=1 needs={list(lc.needs) or 'champion-only'} "
              f"binary={lc.binary or 'NONE -> BLOCKED_ON_PATCH (no server is started; pass --gen-binary)'}")
        print(f"   WHAT {arm.what}")
        print("   SERVER EXTRA " + " ".join(arm.server_extra()) + ("   REQUEST EXTRA " + json.dumps(arm.request_extra()) if arm.request_extra() else ""))
        seen = {}
        for c in lc.cells:
            seen.setdefault(c.unit, []).append(c)
        for u, cs in seen.items():
            flag = "" if not cap or cs[0].est_cal + lc.load_s <= cap else f"   REFUSED at start: exceeds --cap-s {cap}"
            print(f"   cell {u:>6} x{len(cs)} waves  model {cs[0].est_raw/60:6.1f} min  planned {cs[0].est_cal/60:6.1f} min each{flag}")
        tot += lc.est_cal(); tot_raw += lc.load_s + sum(c.est_raw for c in lc.cells)
        print(f"   LAUNCH model {(lc.load_s + sum(c.est_raw for c in lc.cells))/3600:.2f} h  planned (safety x{a.safety} on unseen keys) = {lc.est_cal()/3600:.2f} h")
    print(f"\nTOTAL launches={len(launches)} cells={sum(len(l.cells) for l in launches)} model={tot_raw/3600:.2f} h planned(safety x{a.safety})={tot/3600:.2f} h "
          f"(threads={a.threads}; D4-measured prefill curve, per-key recalibration from this run's own cells replaces the safety factor)")
    for m in models:
        for kind in sorted({c.kind for lc in launches if lc.model == m for c in lc.cells}):
            for x in assumptions(m, a.threads, kind):
                print(f"ASSUMPTION[{m}/{kind}] {x}")


# ----------------------------------------------------------------------------------------------- conversation building
def fit_conversation(measure, make, target_tok, lo, hi, tol=0.01, max_iter=24):
    """Find the integer size parameter k in [lo, hi] so that measure(make(k)) is within tol of target_tok (token count grows with k).
    `make(k)` -> messages, `measure(messages)` -> tokens. Pure bisection; the caller supplies both functions, so tests need no server.
    Returns (messages, tokens, k) of the closest probe seen."""
    best = None
    for _ in range(max_iter):
        k = (lo + hi) // 2
        msgs = make(k); n = measure(msgs)
        if best is None or abs(n - target_tok) < abs(best[1] - target_tok):
            best = (msgs, n, k)
        if abs(n - target_tok) <= tol * target_tok or lo >= hi - 1:
            break
        if n < target_tok: lo = k
        else: hi = k
    return best


def conv_messages(texts):
    """Alternating user/assistant messages, user first, ending on a user message (len(texts) must be odd)."""
    assert len(texts) % 2 == 1
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)]


def edit_message(messages, idx, tag):
    out = [dict(m) for m in messages]
    out[idx]["content"] = tag + out[idx]["content"]
    return out


def edit_index_for_back(starts, n_total, back_tok):
    """Message index whose start is nearest to (n_total - back_tok), never message 0 (an edit at 0 is a cold prompt, not a resume)."""
    target = n_total - back_tok
    best = min(range(1, len(starts)), key=lambda i: abs(starts[i] - target))
    return best


# ----------------------------------------------------------------------------------------------- requests
def _sse(resp):
    for raw in resp:
        line = raw.decode(errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            continue
        try:
            yield json.loads(body)
        except ValueError:
            continue


def stream_request(port, path, body, collect_ids=False, timeout=14400):
    """POST a streaming request; returns dict(status, error, ttft, text, ids, timings, t_start, t_end, n_chunks)."""
    rec = dict(status=None, error=None, ttft=None, text="", ids=[], timings={}, n_chunks=0)
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); rec["t_start"] = t0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["status"] = r.status
            for j in _sse(r):
                if "error" in j:
                    rec["error"] = json.dumps(j["error"])[:300]; continue
                ch = (j.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or j.get("content") or ""
                toks = j.get("tokens") or []
                if collect_ids: rec["ids"] += toks
                if (piece or toks) and rec["ttft"] is None:
                    rec["ttft"] = time.time() - t0
                rec["text"] += piece
                rec["n_chunks"] += 1
                if j.get("timings"):
                    rec["timings"] = j["timings"]
                if j.get("usage"):
                    rec["usage"] = j["usage"]
    except urllib.error.HTTPError as e:
        rec["status"] = e.code; rec["error"] = e.read().decode(errors="ignore")[:300]
    except Exception as e:                                       # noqa: BLE001 - recorded, not swallowed
        rec["error"] = repr(e)[:300]
    rec["t_end"] = time.time()
    tm = rec["timings"]
    us = rec.get("usage") or {}
    cached_u = (us.get("prompt_tokens_details") or {}).get("cached_tokens")
    rec.update(prompt_n=tm.get("prompt_n", (us["prompt_tokens"] - (cached_u or 0)) if us.get("prompt_tokens") is not None else None),
               cache_n=tm.get("cache_n", cached_u), prompt_ms=tm.get("prompt_ms"), predicted_n=tm.get("predicted_n"),
               decode_tps=tm.get("predicted_per_second"), n_decode_ckpts=tm.get("n_decode_ckpts"))
    return rec


def chat_request(port, messages, max_tokens, extra=None):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 42, "stream": True, "ignore_eos": True,
            "cache_prompt": True, "id_slot": 0, "stream_options": {"include_usage": True}, "timings_per_token": False}
    body.update(extra or {})
    return stream_request(port, "/v1/chat/completions", body)


def completion_request(port, prompt_ids, n_predict, extra=None):
    body = {"prompt": prompt_ids, "n_predict": n_predict, "ignore_eos": True, "temperature": 0, "seed": 42, "stream": True,
            "return_tokens": True, "cache_prompt": True, "id_slot": 0}
    body.update(extra or {})
    return stream_request(port, "/completion", body, collect_ids=True)


def render_tokens(port, messages):
    """Rendered-prompt token count through the server's own /apply-template + /tokenize (includes the generation prompt)."""
    r = M.http(port, "/apply-template", {"messages": messages})
    return len(M.http(port, "/tokenize", {"content": r["prompt"], "add_special": True})["tokens"])


# ----------------------------------------------------------------------------------------------- budgets (filler contract)
def effective_deadline(deadline_epoch, cap_s, now=time.time):
    """Hard end = the earlier of the filler grant's end and (now + --cap-s); 0 = none."""
    cands = [int(deadline_epoch)] if deadline_epoch else []
    if cap_s:
        cands.append(int(now()) + int(cap_s))
    return min(cands) if cands else 0


class Budget:
    """STOP file between cells; hard end with a margin; a cell that does not fit is not started."""
    def __init__(self, stop_file="", deadline_epoch=0, margin_s=EST_MARGIN_S, now=time.time):
        self.stop_file, self.deadline, self.margin, self.now = stop_file, int(deadline_epoch or 0), margin_s, now

    def stop_reason(self):
        if self.stop_file and os.path.exists(self.stop_file):
            return f"STOP file {self.stop_file}"
        return None

    def fits(self, est_s):
        return (not self.deadline) or (self.now() + est_s + self.margin <= self.deadline)

    def remaining(self):
        return (self.deadline - self.now()) if self.deadline else None


class Calibrator:
    """Per-(model, kind, unit) actual/estimate ratios learned from this run's own completed cells (and from an earlier run's cells.jsonl rows that
    carry est_raw_s). There is NO global factor: a key never measured gets `safety`; a measured key uses the WORST ratio seen for it (conservative)."""
    def __init__(self, safety=SAFETY_UNSEEN):
        self.safety, self.ratios = safety, {}

    @staticmethod
    def key(model, kind, unit):
        return (model, kind, unit)

    def observe(self, model, kind, unit, est_raw_s, wall_s):
        if est_raw_s and est_raw_s > 0 and wall_s and wall_s > 0:
            self.ratios.setdefault(self.key(model, kind, unit), []).append(wall_s / est_raw_s)

    def factor(self, model, kind, unit):
        r = self.ratios.get(self.key(model, kind, unit))
        return max(r) if r else self.safety

    def planned(self, cell):
        return cell.est_raw * self.factor(cell.model, cell.kind, cell.unit)

    @classmethod
    def from_rows(cls, rows, safety=SAFETY_UNSEEN):
        c = cls(safety)
        for r in rows:
            if r.get("result") == "OK" and r.get("est_raw_s") and r.get("wall_s"):
                c.observe(r.get("model"), r.get("kind"), r.get("unit"), r["est_raw_s"], r["wall_s"])
        return c


def read_rows(path):
    p = Path(path)
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def done_keys(cells_path):
    """Per-cell resume -> (done, parked, blocked). OK rows are done; a cell with >= 2 failed attempts is parked (never a hot retry loop
    under a filler); BLOCKED_ON_PATCH cells are skipped until the caller passes --retry-blocked (a new binary changes the key anyway)."""
    done, fails, blocked = set(), {}, set()
    p = Path(cells_path)
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            k = r.get("cell_key")
            if not k: continue
            if r.get("result") == "OK": done.add(k)
            elif r.get("result") == "BLOCKED_ON_PATCH": blocked.add(k)
            elif r.get("result") in ("ERRORS", "REFUSED", "EXCEPTION", "PARTIAL"): fails[k] = fails.get(k, 0) + 1
    parked = {k for k, n in fails.items() if n >= 2}
    return done, parked, blocked


# ----------------------------------------------------------------------------------------------- decision rules (predeclared; pure arithmetic over cells.jsonl)
def _med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def _group(rows):
    g = {}
    for r in rows:
        if r.get("result") != "OK" or r.get("kind") != "ckpt":
            continue
        g.setdefault((r["model"], r["unit"], r["arm"]), []).append(r)
    return g


def _bad_keys(rows):
    """(model, unit, arm) of every failed attempt (ERRORS / REFUSED / EXCEPTION): the denominator is every request, so one failed attempt fails the arm at that L."""
    bad = set()
    for r in rows:
        if r.get("result") in ("ERRORS", "REFUSED", "EXCEPTION") and r.get("cell_key"):
            parts = r["cell_key"].split("|")
            if len(parts) >= 4 and parts[2] == "ckpt":
                bad.add((parts[0], parts[3], parts[1]))
    return bad


def compare_arm_to_dense(arm_rows, dense_rows):
    """Rule input for ONE (model, L, arm): per probe, is the arm's resume prompt_n within the dense wave-to-wave spread, and are the
    retained-checkpoint bytes per sequence lower. Missing probe / error anywhere = fail (denominator = every request)."""
    probes = sorted({p for r in dense_rows for p in r["probe_prompt_n"]})
    det, ok_a = {}, True
    for p in probes:
        dv = [r["probe_prompt_n"].get(p) for r in dense_rows]
        av = [r["probe_prompt_n"].get(p) for r in arm_rows]
        if any(x is None for x in dv + av):
            det[p] = dict(pass_=False, why="missing value / errored request"); ok_a = False; continue
        spread = max(dv) - min(dv)
        gap = _med(av) - _med(dv)
        det[p] = dict(dense_med=_med(dv), arm_med=_med(av), dense_spread=spread, gap_tok=gap, pass_=abs(gap) <= spread)
        ok_a = ok_a and det[p]["pass_"]
    bd, ba = _med([r["ckpt_bytes_mib"] for r in dense_rows]), _med([r["ckpt_bytes_mib"] for r in arm_rows])
    bytes_ok = (bd is not None and ba is not None and ba < bd)
    junction_ok = all(r.get("junction_materialized") is not False for r in arm_rows)      # a junction arm whose junction never appeared measured nothing
    if not junction_ok:
        ok_a = False
    return dict(prompt_n_ok=ok_a, bytes_ok=bytes_ok, junction_materialized=junction_ok, bytes_arm_mib=ba, bytes_dense_mib=bd, probes=det,
                bytes_saved_pct=(round((bd - ba) / bd * 100, 2) if bd and ba is not None else None))


def decide_121(rows, dense="dense"):
    """KVU-12.1 predeclared rule: an alternate arm replaces dense only if, at EVERY L it has data for (and it must cover every L dense has),
    its resume prompt_n is within the dense arm's wave-to-wave spread at every probe AND its checkpoint bytes per sequence are lower;
    else dense stays. Also non-gating: resume regret in tokens and bytes saved."""
    g, bad = _group(rows), _bad_keys(rows)
    out, models = {}, sorted({k[0] for k in g})
    for model in models:
        arms = sorted(({k[2] for k in g if k[0] == model} | {k[2] for k in bad if k[0] == model}) - {dense})
        dense_L = sorted({k[1] for k in g if k[0] == model and k[2] == dense}, key=lambda l: LS_ARM.get(l, 0))
        for arm in arms:
            per_L, full = {}, True
            for L in dense_L:
                ar, dr = g.get((model, L, arm)), g.get((model, L, dense))
                if (model, L, arm) in bad or (model, L, dense) in bad:
                    per_L[L] = dict(status="FAIL", why="a failed attempt exists at this (model, L, arm); denominator = every request"); continue
                if not ar:
                    per_L[L] = dict(status="MISSING"); full = False; continue
                if len(ar) < 2 or len(dr) < 2:
                    per_L[L] = dict(status="TOO_FEW_WAVES", n_arm=len(ar), n_dense=len(dr)); full = False; continue
                c = compare_arm_to_dense(ar, dr); c["status"] = "PASS" if (c["prompt_n_ok"] and c["bytes_ok"]) else "FAIL"
                per_L[L] = c
            have = [L for L in dense_L if per_L[L]["status"] in ("PASS", "FAIL")]
            allpass = bool(have) and all(per_L[L]["status"] == "PASS" for L in have)
            if not full: verdict = "INCOMPLETE" if allpass or not have else "DENSE_STAYS"
            else: verdict = "REPLACES_DENSE" if allpass else "DENSE_STAYS"
            out[f"{model}|{arm}"] = dict(verdict=verdict, per_L=per_L)
    return out


def select_fixed(rows, dense="dense", select_lens=("8k", "32k")):
    """Holdout selection (predeclared): among fixed-interval candidates, use ONLY the select-stage L values; a candidate qualifies when it
    PASSES the 12.1 rule at every select L; pick the lowest bytes_arm at the largest select L; ties -> larger stride, then larger cap.
    Returns {model: (stride_tokens, cap) or None}. None = no candidate qualifies, so no fixed arm goes to holdout (dense stays by default)."""
    g = _group([r for r in rows if r.get("unit") in select_lens])
    bad = _bad_keys(rows)
    res = {}
    for model in sorted({k[0] for k in g}):
        cands = []
        for arm in sorted({k[2] for k in g if k[0] == model and k[2].startswith("fixed_")}):
            ok, byt = True, None
            for L in select_lens:
                ar, dr = g.get((model, L, arm)), g.get((model, L, dense))
                if (model, L, arm) in bad or (model, L, dense) in bad: ok = False; break
                if not ar or not dr or len(ar) < 2 or len(dr) < 2: ok = False; break
                c = compare_arm_to_dense(ar, dr)
                if not (c["prompt_n_ok"] and c["bytes_ok"]): ok = False; break
                byt = c["bytes_arm_mib"]
            if ok:
                m = re.fullmatch(r"fixed_s(\d+)k_c(\d+)", arm)
                cands.append((byt, -int(m.group(1)), -int(m.group(2)), (int(m.group(1)) * 1024, int(m.group(2)))))
        res[model] = sorted(cands)[0][3] if cands else None
    return res


def decide_123(rows, arm="gen_decode128", dense="gen_dense", every=GEN_EVERY, slack=GEN_SLACK_TOK):
    """KVU-12.3 predeclared rule per (model, G), over every divergence point and every wave:
    CONFIRMED only if, with the SAME binary sha as the dense arm, (1) turn-2 resume prompt_n <= (tokens past the divergence) + every + slack
    at every probe [resume lands at most one boundary before the divergence], (2) it saves >= 50% of dense's re-processed generated tokens
    at every probe whose divergence is >= 1024 tokens into the generation, (3) retained decode snapshots <= the cap and every snapshot
    was logged, (4) turn-1 decode tok/s within max(2x dense wave spread, 3%) of dense [INDICATIVE]. Any missing value / error = NOT_CONFIRMED."""
    out = {}
    byk = {}
    for r in rows:
        if r.get("kind") != "gen": continue
        byk.setdefault((r["model"], r["unit"], r["arm"]), []).append(r)
    for (model, unit) in sorted({(k[0], k[1]) for k in byk}):
        A, D = [r for r in byk.get((model, unit, arm), []) if r.get("result") == "OK"], [r for r in byk.get((model, unit, dense), []) if r.get("result") == "OK"]
        if len(A) < 2 or len(D) < 2:
            out[f"{model}|{unit}"] = dict(verdict="INCOMPLETE", n_arm=len(A), n_dense=len(D)); continue
        why, per = [], {}
        if len({r.get("binary_sha256") for r in A + D}) != 1: why.append("binary sha256 differs between arm and dense")
        for f in sorted({k for r in D for k in r["probe_prompt_n"]}):
            av, dv = [r["probe_prompt_n"].get(f) for r in A], [r["probe_prompt_n"].get(f) for r in D]
            tail = [r["probe_tail_tok"].get(f) for r in A]
            gen_in = [r["probe_g"].get(f) for r in A]
            if any(x is None for x in av + dv + tail + gen_in):
                why.append(f"probe {f}: missing value"); continue
            c1 = all(a <= t + every + slack for a, t in zip(av, tail))
            dense_gen = [d - t for d, t in zip(dv, tail)]            # tokens of the generated part dense re-processes
            c2 = True if min(gen_in) < 1024 else all((d - a) >= 0.5 * dg for d, a, dg in zip(dv, av, dense_gen))
            per[f] = dict(arm_med=_med(av), dense_med=_med(dv), c1_resume_within_boundary=c1, c2_saves_half=c2)
            if not (c1 and c2): why.append(f"probe {f}: c1={c1} c2={c2}")
        cap_ok = all(r.get("decode_ckpts_retained", 0) <= GEN_CAP and r.get("decode_ckpts_logged", 0) >= 1 for r in A)
        if not cap_ok: why.append("snapshot cap exceeded or none logged")
        dd = [r.get("turn1_decode_tps") for r in D]; ad = [r.get("turn1_decode_tps") for r in A]
        dec = None
        if None not in dd + ad and _med(dd):
            sp = (max(dd) - min(dd)) / _med(dd) * 100.0
            dec = dict(gap_pct=round((_med(ad) - _med(dd)) / _med(dd) * 100.0, 2), thr_pct=round(max(2 * sp, 3.0), 2))
            dec["ok"] = dec["gap_pct"] >= -dec["thr_pct"]
            if not dec["ok"]: why.append(f"turn-1 decode tok/s {dec['gap_pct']}% below dense beyond {dec['thr_pct']}% (indicative)")
        out[f"{model}|{unit}"] = dict(verdict="CONFIRMED" if not why else "NOT_CONFIRMED", reasons=why, probes=per, turn1_decode=dec)
    return out


# ----------------------------------------------------------------------------------------------- running
def _la(launch, arm, c):
    return dict(id=launch.id, np=1, c=c, unified=False, mtp=MODELS[launch.model]["mtp"], cells=[(cc.unit, cc.kind) for cc in launch.cells], kind="ckpt_arms")


def emit_request_row(row_fields, a, la, srv, facts, loadavg, wave):
    """Per-request row through the shared VB-KVU-CPU emitter (kvu_cpu_matrix.finalize_row)."""
    base = dict(row_fields, pmode="ckpt_arms", L=row_fields.get("L"), cell_key=row_fields.get("cell_key"))
    wr = dict(prompt_n=row_fields.get("prompt_n"), n_prompt_tokens_cache=row_fields.get("cache_n"), n_fork_tokens=None)
    row = M.finalize_row(base, a, la, srv, facts, loadavg, wave=wave, wave_rec=wr)
    row["validation_problems"] = M.validate_row(row)
    return row


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


class LogTail:
    """Reads the server log from an offset; events since the last read."""
    def __init__(self, path):
        self.path, self.off = path, 0
        self.off = os.path.getsize(path) if os.path.exists(path) else 0

    def new_text(self):
        with open(self.path, "rb") as f:
            f.seek(self.off); data = f.read()
        self.off += len(data)
        return data.decode(errors="ignore")

    def events(self):
        return parse_ckpt_events(self.new_text())


def run_ckpt_cell(ctx, cell):
    """KVU-12.1 cell: cold build of an L-token conversation, then the probe plan. Returns the cell summary row."""
    a, srv, port, corpus, arm, model = ctx["a"], ctx["srv"], ctx["port"], ctx["corpus"], ctx["arm"], cell.model
    L = LS_ARM[cell.unit]
    tail, cs = LogTail(srv.dir / "server.log"), CkptSet()
    # same text for every arm of a wave: the corpus cursor is a pure function of (model, L, wave)
    corpus.cursor = (int(hashlib.sha1(f"{model}|{cell.unit}|{cell.wave}".encode()).hexdigest()[:8], 16) * 7919) % max(1, len(corpus.text) // 2)
    ratio = ctx.setdefault("ratio", 4.0)
    pair = TURN_USER_TOK + TURN_ASST_TOK + 16
    n_pairs = max(1, (L - PROBE_NPREDICT - 64 - 64) // pair)       # the last user message absorbs the remainder (16 .. ~2 turns of tokens)
    cell_texts = {}

    def make(k):                                     # k = scale in tokens*10 of the final user message; keeps earlier turns fixed-size
        texts = []
        corpus.cursor = cell_texts.setdefault("c0", corpus.cursor)
        for i in range(2 * n_pairs + 1):
            ntok = TURN_USER_TOK if i % 2 == 0 else TURN_ASST_TOK
            if i == 2 * n_pairs: ntok = max(16, k // 10)
            texts.append(corpus.take(int(ntok * ratio)))
        return conv_messages(texts)

    measure = lambda msgs: render_tokens(port, msgs)
    msgs, n_total, _k = fit_conversation(measure, make, L - PROBE_NPREDICT - 64, lo=160, hi=(pair + TURN_USER_TOK) * 10, tol=0.01)
    texts_tok = [M.n_tokens(port, m["content"]) for m in msgs]
    starts = [0]
    over = max(0, (n_total - sum(texts_tok)) // max(1, len(msgs)))
    for t in texts_tok[:-1]:
        starts.append(starts[-1] + t + over)
    user_starts = [s for m, s in zip(msgs, starts) if m["role"] == "user"]
    reqs = ctx["reqs"] = []                          # visible to run_launch so a killed cell still leaves a PARTIAL row
    la = _la(ctx["launch"], arm, ctx["launch"].c)
    loadavg = [round(x, 2) for x in os.getloadavg()]

    def do(name, messages, lcp_expected, n_prev_total):
        t_pre = tail.events()                        # drain anything unrelated first
        rec = chat_request(port, messages, PROBE_NPREDICT, arm.request_extra())
        evs = tail.events()
        created = cs.apply(evs)
        res = resume_summary(evs)
        n_new = render_tokens(port, messages) if rec["error"] is None else None
        ideal = (n_new - lcp_expected) if (n_new is not None and lcp_expected is not None) else None
        row = dict(cell_key=cell.key, model=model, policy_arm=arm.name, unit=cell.unit, L=L, probe=name, wave=cell.wave, status=rec["status"], error=rec["error"],
                   ttft_s=rec["ttft"], prompt_n=rec["prompt_n"], cache_n=rec["cache_n"], n_prompt_total=n_new, ideal_prompt_n_approx=ideal,
                   regret_tok=(rec["prompt_n"] - ideal if (ideal is not None and rec["prompt_n"] is not None) else None),
                   created_this_request=created, retained=cs.summary(), **res,
                   result_class="counts=correctness; ttft_s=INDICATIVE timing")
        reqs.append(row)
        append_jsonl(ctx["req_path"], emit_request_row(row, a, la, srv, ctx["facts"], loadavg, cell.wave))
        return rec, row

    rec0, r0 = do("cold_build", msgs, 0, 0)
    build_summary = cs.summary()
    created_build = r0["created_this_request"]
    cur, cur_starts, n_cur = msgs, starts, r0["n_prompt_total"] or n_total
    for name, back in probe_plan(L, n_cur):
        if name == "extend_1":
            tx = [corpus.take(int(TURN_ASST_TOK * ratio)), corpus.take(int(TURN_USER_TOK * ratio))]
            nxt = cur[:] + [{"role": "assistant", "content": tx[0]}, {"role": "user", "content": tx[1]}]
            lcp = n_cur                              # control: exact extension, expected prompt_n ~ the appended turn pair for EVERY arm
        else:
            idx = edit_index_for_back(cur_starts, n_cur, back)
            nxt = edit_message(cur, idx, f"[edit w{cell.wave} {name}] ")
            lcp = cur_starts[idx]
        rec, row = do(name, nxt, lcp, n_cur)
        if rec["error"] is None and row["n_prompt_total"]:
            if name == "extend_1":
                cur_starts = cur_starts + [n_cur, n_cur + TURN_ASST_TOK + over]
            cur, n_cur = nxt, row["n_prompt_total"]
    ok = all(r["error"] is None and r["status"] == 200 for r in reqs)
    pn = {r["probe"]: r["prompt_n"] for r in reqs if r["probe"] != "cold_build"}
    cn = {r["probe"]: r["cache_n"] for r in reqs if r["probe"] != "cold_build"}
    pred = simulate_prefill([s for m, s in zip(msgs, starts) if m["role"] == "user"], n_total, arm)
    row = dict(schema=SCHEMA, cell_key=cell.key, result="OK" if ok else "ERRORS", kind="ckpt", model=model, arm=arm.name, unit=cell.unit, L=L, wave=cell.wave,
               n_requests=len(reqs), n_ok=sum(1 for r in reqs if r["error"] is None and r["status"] == 200),
               ckpt_count=build_summary["count"], ckpt_bytes_mib=build_summary["bytes_total_mib"], ckpt_mib_each=build_summary["mib_each"],
               ckpt_positions=build_summary["positions"], ckpt_gaps=build_summary["gaps"], ckpt_max_gap=build_summary["max_gap"],
               ckpt_pinned=build_summary["pinned"], states_created_cold=created_build,
               states_per_request_max=max((r["created_this_request"] for r in reqs), default=None),
               meets_le2_states=(max((r["created_this_request"] for r in reqs), default=99) <= 2) if arm.junction else None,
               junction_materialized=(build_summary["pinned"] >= 1) if arm.junction else None,
               probe_prompt_n=pn, probe_cache_n=cn, probe_ttft_s={r["probe"]: r["ttft_s"] for r in reqs},
               probe_regret_tok={r["probe"]: r["regret_tok"] for r in reqs},
               sim_positions_cold=[c[0] for c in pred], sim_matches_log=(sorted(c[0] for c in pred) == sorted(build_summary["positions"])),
               binary_sha256=M.file_sha256(os.path.realpath(srv.argv[0])), result_class="counts/prompt_n/bytes=correctness; ttft=INDICATIVE")
    return row


def run_gen_cell(ctx, cell):
    """KVU-12.3 cell: turn 1 = GEN_BASE_TOK prompt + G generated tokens; turn 2 = same history with ONE token substituted inside the generation
    at g (deep -> shallow), the rest of the generation re-sent, plus a fresh suffix. Measures how much of the generation turn 2 re-processes."""
    a, srv, port, corpus, arm, model = ctx["a"], ctx["srv"], ctx["port"], ctx["corpus"], ctx["arm"], cell.model
    G = int(cell.unit[1:])
    tail, cs = LogTail(srv.dir / "server.log"), CkptSet()
    corpus.cursor = (int(hashlib.sha1(f"{model}|{cell.unit}|{cell.wave}|gen".encode()).hexdigest()[:8], 16) * 7919) % max(1, len(corpus.text) // 2)
    doc, n_doc, _r = M.make_doc(corpus, port, GEN_BASE_TOK - 16, ctx.setdefault("ratio", 4.0))
    base_ids = M.http(port, "/tokenize", {"content": doc, "add_special": True})["tokens"]
    sfx_ids = M.http(port, "/tokenize", {"content": corpus.take(int(GEN_SUFFIX_TOK * 3.5)), "add_special": False})["tokens"][:GEN_SUFFIX_TOK]
    alt_a = M.http(port, "/tokenize", {"content": " .", "add_special": False})["tokens"][0]
    alt_b = M.http(port, "/tokenize", {"content": " ,", "add_special": False})["tokens"][0]
    la = _la(ctx["launch"], arm, ctx["launch"].c)
    loadavg = [round(x, 2) for x in os.getloadavg()]
    extra = {}
    if arm.decode_every:
        extra = {"checkpoint_decode_every": arm.decode_every, "checkpoint_decode_cap": arm.decode_cap}
    reqs = ctx["reqs"] = []
    tail.events()
    t1 = completion_request(port, base_ids, G, extra)
    ev1 = tail.events(); cs.apply(ev1)
    s1 = cs.summary()
    gen_ids = t1["ids"]
    t1_ok = t1["error"] is None and t1["status"] == 200 and len(gen_ids) >= G - GEN_SLACK_TOK
    row1 = dict(cell_key=cell.key, model=model, policy_arm=arm.name, unit=cell.unit, L=len(base_ids), probe="turn1", wave=cell.wave, status=t1["status"], error=t1["error"],
                ttft_s=t1["ttft"], prompt_n=t1["prompt_n"], cache_n=t1["cache_n"], retained=s1, decode_tps=t1["decode_tps"], result_class="counts=correctness; ttft/decode=INDICATIVE")
    append_jsonl(ctx["req_path"], emit_request_row(row1, a, la, srv, ctx["facts"], loadavg, cell.wave))
    pn, cn, pg, ptail, ttft, resumes = {}, {}, {}, {}, {}, {}
    reqs.append(row1)
    if t1_ok:
        for f in GEN_FRACS:                          # deep -> shallow: surviving snapshots <= g are never erased by a deeper probe
            g = max(1, min(len(gen_ids) - 2, int(len(gen_ids) * f)))
            alt = alt_a if gen_ids[g] != alt_a else alt_b
            prompt2 = base_ids + gen_ids[:g] + [alt] + gen_ids[g + 1:] + sfx_ids
            rec = completion_request(port, prompt2, PROBE_NPREDICT, extra)
            evs = tail.events(); cs.apply(evs); rs = resume_summary(evs)
            key = f"g{int(f * 100)}"
            pn[key], cn[key], pg[key], ttft[key] = rec["prompt_n"], rec["cache_n"], g, rec["ttft"]
            ptail[key] = len(prompt2) - (len(base_ids) + g)          # tokens at/after the divergence the server must process regardless
            resumes[key] = rs
            row = dict(cell_key=cell.key, model=model, policy_arm=arm.name, unit=cell.unit, L=len(prompt2), probe=key, wave=cell.wave, status=rec["status"], error=rec["error"],
                       ttft_s=rec["ttft"], prompt_n=rec["prompt_n"], cache_n=rec["cache_n"], n_prompt_total=len(prompt2), g_in_generation=g, retained=cs.summary(), **rs,
                       result_class="counts=correctness; ttft=INDICATIVE timing")
            append_jsonl(ctx["req_path"], emit_request_row(row, a, la, srv, ctx["facts"], loadavg, cell.wave))
            reqs.append(row)
    ok = t1_ok and all(r["error"] is None and r["status"] == 200 for r in reqs)
    ndec = sum(1 for e in ev1 if e["kind"] == "created" and e.get("decode"))
    return dict(schema=SCHEMA, cell_key=cell.key, result="OK" if ok else "ERRORS", kind="gen", model=model, arm=arm.name, unit=cell.unit, G=G, wave=cell.wave,
                n_requests=len(reqs), n_ok=sum(1 for r in reqs if r["error"] is None and r["status"] == 200), n_generated=len(gen_ids),
                turn1_decode_tps=t1["decode_tps"], turn1_ttft_s=t1["ttft"], probe_prompt_n=pn, probe_cache_n=cn, probe_g=pg, probe_tail_tok=ptail, probe_ttft_s=ttft,
                probe_resume=resumes, decode_ckpts_logged=ndec, decode_ckpts_retained=s1["decode"], ckpt_count_after_turn1=s1["count"],
                ckpt_bytes_after_turn1_mib=s1["bytes_total_mib"], ckpt_positions_after_turn1=s1["positions"],
                binary_sha256=M.file_sha256(os.path.realpath(srv.argv[0])), result_class="counts/prompt_n/bytes=correctness; ttft/decode=INDICATIVE")


def probe_capability(port, arm):
    """True only if the running server proves the capability the arm needs (never assumed from the binary name)."""
    if "decode_ckpt" in arm.needs:
        try:
            props = M.http(port, "/props", None, timeout=30)
        except Exception:                                        # noqa: BLE001
            return False
        return bool(props.get("checkpoint_decode"))
    return True


def partial_row(cell, reqs, wall_s):
    """What a cell that was killed (cap, STOP escalation, SIGTERM) had completed: one compact entry per finished request. The full per-request rows
    (validated capture rows with retained-checkpoint state) are already in requests.jsonl, written as each request completed."""
    done = [dict(probe=r.get("probe"), prompt_n=r.get("prompt_n"), cache_n=r.get("cache_n"), ttft_s=r.get("ttft_s"), error=r.get("error"),
                 retained_count=(r.get("retained") or {}).get("count"), retained_positions=(r.get("retained") or {}).get("positions")) for r in reqs]
    return dict(schema=SCHEMA, cell_key=cell.key, result="PARTIAL", model=cell.model, kind=cell.kind, arm=cell.arm, unit=cell.unit, wave=cell.wave,
                n_requests_done=len(done), requests_done=done, wall_s=round(wall_s, 1), est_raw_s=round(cell.est_raw, 1),
                ts=time.strftime("%FT%TZ", time.gmtime()), note="indicative only; not a decision row (no OK summary)")


def run_launch(a, launch, todo, budget, calib, ctx_common, out):
    arm = arm_by_name(launch.arm)
    ldir = out / launch.id; ldir.mkdir(parents=True, exist_ok=True)
    argv = arm_argv(launch.model, arm, launch.c, a.port, a.threads, launch.binary)
    env = dict(os.environ); env.update(M.PROD_ENV); env.pop("GGML_NUMA_REPACK_INTERLEAVE", None)
    prefix = ["numactl", "--interleave=all", "--", "taskset", "-c", a.cpu_list]
    srv = M.Server(argv, env, prefix, ldir, a.port); srv.bin_dir = os.path.dirname(os.path.realpath(argv[0]))
    say = lambda s: print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {s}", flush=True)
    say(f"LAUNCH {launch.id} cells_todo={len(todo)} argv={json.dumps(prefix + argv)}")
    srv.start()
    try:
        st = srv.healthy()
        say(f"health={st}")
        if st != "HEALTHY":
            append_jsonl(out / CELLS_FILE, dict(schema=SCHEMA, cell_key=f"launch|{launch.id}", result="SERVER_" + st, ts=time.strftime("%FT%TZ", time.gmtime()))); return "server"
        link = srv.linkage(); say(f"LINKAGE {json.dumps(link)}")
        if not link.get("all_in_kernel_store"):
            append_jsonl(out / CELLS_FILE, dict(schema=SCHEMA, cell_key=f"launch|{launch.id}", result="LINKAGE_MISMATCH", linkage=link)); return "linkage"
        if not probe_capability(a.port, arm):          # a binary was supplied but does not prove the capability
            for c in todo:
                append_jsonl(out / CELLS_FILE, dict(schema=SCHEMA, cell_key=c.key, result="BLOCKED_ON_PATCH", needs=list(arm.needs), model=c.model, arm=c.arm, unit=c.unit, wave=c.wave,
                                                    ts=time.strftime("%FT%TZ", time.gmtime())))
            say(f"BLOCKED_ON_PATCH {launch.id}: /props does not report {arm.needs}; see PLAN.md 11.5 patch spec")
            return "blocked"
        slog = (ldir / "server.log").read_text(errors="ignore")
        facts = {k: (re.findall(p, slog) or [None])[-1] for k, p in M.LOAD_PATS.items()}
        facts["rss_after_load_gib"] = srv.rss_gib()[0]
        M.one_request(a.port, "Warm up. " * 50, 16, "warmup")
        ctx = dict(ctx_common, a=a, srv=srv, port=a.port, arm=arm, launch=launch, facts=facts, req_path=out / REQS_FILE)
        for c in todo:
            why = budget.stop_reason()
            if why:
                say(f"STOP {why}: not starting {c.key}"); return "stop"
            est = calib.planned(c)                    # per-(model, kind, unit) corrected estimate, never a global factor
            if not budget.fits(est):
                say(f"REFUSED_START {c.key}: corrected est {est:.0f}s (+{budget.margin}s margin) exceeds remaining {budget.remaining():.0f}s; not started")
                append_jsonl(out / CELLS_FILE, dict(schema=SCHEMA, cell_key=c.key, result="SKIPPED_NO_FIT", est_planned_s=round(est, 1), remaining_s=round(budget.remaining() or 0, 1),
                                                    ts=time.strftime("%FT%TZ", time.gmtime())))
                return "nofit"
            t0 = time.time(); ctx["reqs"] = []; row = None
            try:
                row = run_ckpt_cell(ctx, c) if c.kind == "ckpt" else run_gen_cell(ctx, c)
            except M.CellRefused as e:
                row = dict(schema=SCHEMA, cell_key=c.key, result="REFUSED", reason=str(e))
            except Exception as e:                                # noqa: BLE001 - recorded as a failed attempt; parked after 2
                row = dict(schema=SCHEMA, cell_key=c.key, result="EXCEPTION", reason=repr(e)[:400])
            finally:
                if row is None:                               # SystemExit / SIGTERM at the cap: keep what completed (per-request rows are already on disk)
                    append_jsonl(out / CELLS_FILE, partial_row(c, ctx.get("reqs") or [], time.time() - t0))
            wall = time.time() - t0
            row.update(est_raw_s=round(c.est_raw, 1), est_planned_s=round(est, 1), wall_s=round(wall, 1), ts=time.strftime("%FT%TZ", time.gmtime()),
                       cpu_list=a.cpu_list, threads=a.threads, model=c.model, kind=c.kind, unit=c.unit, actual_over_model=round(wall / max(c.est_raw, 1.0), 3))
            append_jsonl(out / CELLS_FILE, row)
            if row["result"] == "OK":
                calib.observe(c.model, c.kind, c.unit, c.est_raw, wall)
            say(f"CELL {c.key} result={row['result']} wall={row['wall_s']}s model_est={row['est_raw_s']}s planned={row['est_planned_s']}s actual/model={row['actual_over_model']}")
        return "done"
    finally:
        dead = srv.stop(); say(f"STOP server dead={dead} pid={srv.proc.pid}")


def parse_kv_list(spec, cast=str):
    """'q36_35b=8k,32k;q38_27b=8k' -> {'q36_35b': ('8k','32k'), 'q38_27b': ('8k',)}; an empty value means none ('q38_27b=')."""
    out = {}
    for part in [x for x in (spec or "").split(";") if x.strip()]:
        k, _, v = part.partition("=")
        out[k.strip()] = tuple(cast(x) for x in v.split(",") if x.strip())
    return out


def plan_from_args(a):
    sel = load_selection(a)
    bins = {"ckpt": a.fork_binary, "gen": a.gen_binary}
    lens = {"select": parse_kv_list(a.lens_select), "holdout": parse_kv_list(a.lens_holdout)}
    return sel, build_plan(a.models.split(","), a.stages.split(","), a.waves, a.threads, sel, lens_override=lens,
                           only_arms=set(a.arms.split(",")) if a.arms else None, cal=a.safety, bins=bins, gen_g=parse_kv_list(a.gen_g, int))


def cmd_run(a):
    if a.dry_run:
        return cmd_plan(a)
    if a.port in range(8000, 8201):
        sys.exit("refusing production port range 8000-8200")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sel, launches = plan_from_args(a)
    if "holdout" in a.stages.split(",") and not sel:
        sys.exit("holdout needs --selection (run the select stage, then `decide --write-selection`); refusing to choose a stride after the fact")
    for b in sorted({lc.binary for lc in launches if lc.binary}):
        r = subprocess.run([M.LINKAGE, b], capture_output=True, text=True)
        print(f"verify_ggml_linkage {b} rc={r.returncode}", flush=True)
        if r.returncode != 0:
            sys.exit(f"verify_ggml_linkage failed for {b}; refusing to measure")
    corpus = M.Corpus(a.corpus_roots.split(","))
    print(f"corpus sha256={corpus.sha} files={corpus.nfiles}", flush=True)
    deadline = effective_deadline(a.deadline_epoch, a.cap_s)
    budget = Budget(a.stop_file, deadline)
    done, parked, blocked = done_keys(out / CELLS_FILE)
    calib = Calibrator.from_rows(read_rows(out / CELLS_FILE), a.safety)
    if a.retry_blocked:
        blocked = set()
    reason, n_blocked, n_nofit = "all cells done", 0, 0
    for lc in launches:
        todo = [c for c in lc.cells if c.key not in done and c.key not in parked and c.key not in blocked]
        if not todo:
            continue
        if not lc.binary:                                       # no binary can serve this arm: record once, start nothing
            for c in todo:
                append_jsonl(out / CELLS_FILE, dict(schema=SCHEMA, cell_key=c.key, result="BLOCKED_ON_PATCH", needs=list(lc.needs), model=c.model, arm=c.arm,
                                                    unit=c.unit, wave=c.wave, ts=time.strftime("%FT%TZ", time.gmtime())))
            print(f"BLOCKED_ON_PATCH {lc.id}: no binary (pass --gen-binary once patch P-A is built); see PLAN.md 11.5", flush=True)
            n_blocked += len(todo); continue
        why = budget.stop_reason()
        if why:
            reason = why; break
        if not budget.fits(lc.load_s + calib.planned(todo[0])):        # corrected estimate of the launch's first cell must fit the remaining cap/deadline
            print(f"REFUSED_START {lc.id}: load {lc.load_s:.0f}s + corrected first cell {calib.planned(todo[0]):.0f}s exceeds remaining "
                  f"{budget.remaining():.0f}s; trying the next launch", flush=True)
            n_nofit += 1; continue
        res = run_launch(a, lc, todo, budget, calib, dict(corpus=corpus), out)
        if res == "blocked": n_blocked += len(todo)
        if res == "nofit": n_nofit += 1                                # a later, cheaper launch may still fit
        if res == "stop":
            reason = "STOP file"; break
    print(f"RUN_END reason={reason} blocked_cells={n_blocked} refused_start_launches={n_nofit}", flush=True)
    return 0


def load_selection(a):
    if not a.selection:
        return None
    raw = json.loads(Path(a.selection).read_text())
    return {m: (tuple(v) if v else None) for m, v in raw["selection"].items()}


def cmd_plan(a):
    sel, launches = plan_from_args(a)
    if "holdout" in a.stages.split(",") and not sel:
        print("NOTE: the holdout fixed-interval arm is pending selection (--selection from `decide --write-selection`); dense/tail_junction/coarse are planned.")
    print_plan(launches, a)
    ram = max(MODELS[m]["ram_gib"] for m in a.models.split(","))
    print(f"RESOURCES: one server at a time, np=1, cpu-list {a.cpu_list} (-t {a.threads}); RAM <= ~{ram} GiB (mlock, --no-mmap) + checkpoints "
          f"(<= {DENSE_CTX + GEN_CAP} x ~0.1-0.3 GiB); no GPU; port {a.port}; fork binary {a.fork_binary}; gen binary {a.gen_binary or 'NONE (decode arm blocked)'}")
    return 0


def cmd_decide(a):
    rows = [json.loads(x) for x in (Path(a.out) / CELLS_FILE).read_text().splitlines() if x.strip()]
    out = dict(schema=SCHEMA, class_="correctness (counts/prompt_n/bytes); TTFT indicative", rules="PLAN.md section 11.4",
               kvu121=decide_121(rows), selection=select_fixed(rows), kvu123=decide_123(rows),
               blocked=sorted({r["cell_key"] for r in rows if r.get("result") == "BLOCKED_ON_PATCH"}))
    print(json.dumps(out, indent=1))
    if a.write_selection:
        Path(a.write_selection).write_text(json.dumps(dict(schema=SCHEMA, selection=out["selection"], select_lens=["8k", "32k"],
                                                           written=time.strftime("%FT%TZ", time.gmtime())), indent=1) + "\n")
    return 0



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(sp):
        sp.add_argument("--models", default="q36_35b,q38_27b")
        sp.add_argument("--stages", default="select,gen", help="select | holdout | gen (comma list, run in this order)")
        sp.add_argument("--arms", default="", help="comma list of arm names to keep (default: all of the stage)")
        sp.add_argument("--waves", type=int, default=WAVES_DEFAULT)
        sp.add_argument("--lens-select", default="", help="per-model L override for the select stage, e.g. 'q36_35b=8k;q38_27b=8k' (default: MODELS table)")
        sp.add_argument("--lens-holdout", default="", help="per-model L override for the holdout stage, e.g. 'q36_35b=64k,128k;q38_27b=64k'")
        sp.add_argument("--gen-g", default="", help="per-model generated lengths for the gen stage, e.g. 'q36_35b=1024,4096;q38_27b=1024'")
        sp.add_argument("--threads", type=int, default=24, help="one quarter by default (filler)")
        sp.add_argument("--cpu-list", default="72-95")
        sp.add_argument("--safety", type=float, default=SAFETY_UNSEEN, help="multiplier on the model estimate for a (model, kind, unit) key this run has not measured; "
                                                                            "measured keys use their own worst actual/model ratio (no global factor)")
        sp.add_argument("--cap-s", type=int, default=0, help="hard cap on this run's wall time: no cell whose corrected estimate exceeds the remaining cap is started (plan flags them)")
        sp.add_argument("--selection", default="", help="selection.json written by `decide --write-selection` (holdout)")
        sp.add_argument("--fork-binary", default=FORK_BIN, help="build serving the KVU-12.1 arms (needs checkpoint_at)")
        sp.add_argument("--gen-binary", default="", help="patched build (patch P-A) serving the KVU-12.3 arms; both gen arms use it so the baseline shares the binary. "
                                                         "Empty: gen_dense runs on --fork-binary, the decode arm is BLOCKED_ON_PATCH")
        sp.add_argument("--port", type=int, default=18971)
    pl = sub.add_parser("plan", help="dry-run: launches, cells, estimates, resources; starts nothing"); common(pl)
    ru = sub.add_parser("run", help="run as a filler: STOP file + deadline + per-cell resume"); common(ru)
    ru.add_argument("--out", required=True); ru.add_argument("--stop-file", default=""); ru.add_argument("--deadline-epoch", type=int, default=0)
    ru.add_argument("--dry-run", action="store_true"); ru.add_argument("--retry-blocked", action="store_true"); ru.add_argument("--corpus-roots", default="/workspace/handoffs,/workspace/docs")
    de = sub.add_parser("decide", help="predeclared rules over cells.jsonl (starts nothing)")
    de.add_argument("--out", required=True); de.add_argument("--write-selection", default="")
    a = p.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
