#!/usr/bin/env python3
"""KPF-18.1 / 18.2 / 18.3 arms for the KPF-18 gate runner (kpf18_gate.py).

handoffs/active/kv-prefix-fork-and-paged-attention.md, rows KPF-18.1 (hit-length rows A-D), KPF-18.2 (drafter
parity, per-component attribution, anti-vacuity, cold control, policy control) and KPF-18.3 (chunk-split cost,
end-checkpoint position, request lane, >=50k warm-vs-cold logprob).

This module holds everything that does not need a server:
  - the server's prefix-hash ladder and the KPF-11 fork-position PREDICTOR (what n_fork_tokens the rule says);
  - drafter-configuration derivation (`strip_speculative`), sweep/position planning, cell + component records;
  - the PREDECLARED decision rules (RULES, written into run_meta.json before any server starts) and the pure
    evaluators that turn per-sibling cells into counts;
  - argparse additions and the registry the gate wires into (`BUILDERS`, `PREFLIGHTS`).
The per-arm step builders (server launches, requests) are in the same file, below, and take the gate module as an
argument (`gate`) instead of importing it: `python3 kpf18_gate.py` runs as `__main__`, and importing it again by
name would create a second copy of its module state (CPU_MODE, TRUNKS, ...).

Record schema is unchanged: `epyc.kpf.run.v1`, one record per server launch x arm, with the per-sibling raw cells
under `cells` (n_fork_tokens as returned in the request's timings AND as read back from /slots, the predicted
value, the checkpoint set that existed before the request, errors). Decision-rule results are COUNTS; no pass word.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from typing import Callable

ARM_ITEMS = (181, 182, 183)
ARM_CMDS = {"kpf181": [181], "kpf182": [182], "kpf183": [183], "kpf18arms": [181, 182, 183]}
TASK_OF = {181: "KPF-18.1", 182: "KPF-18.2", 183: "KPF-18.3"}

BUILDERS: dict[int, Callable] = {}      # item -> fn(ctx, gate) -> list[Step]   (filled by the arm sections below)
PREFLIGHTS: dict[int, Callable] = {}    # item -> fn(ctx, gate) -> (problems, notes)
RULES: dict[int, dict] = {}             # item -> {arm key: predeclared rule text}  (filled by the arm sections)

# server/prompt constants the predictor and the planners share with the runner
FNV_BASIS = 0x14650FB0739D0383         # tools/server/server-context.cpp, the /slots prefix_hash ladder
FNV_PRIME = 0x100000001B3
_M64 = (1 << 64) - 1
N_CKPT_AT_MAX = 8                       # `checkpoint_at` entries a request may pin
FLOOR_EPS = 1e-4                        # probability / logprob resolution below which two floats are "equal"


# ---------------------------------------------------------------- server prefix-hash ladder
def prefix_hashes(ids: list[int], marks) -> dict[int, str]:
    """FNV-1a 64 over the int32 little-endian token ids of ids[:m], for each requested mark m <= len(ids).
    Same constants and walk as the server's /slots `prefix_hash` ladder (the server's offset basis has one digit
    fewer than the textbook FNV value: keep it identical)."""
    want = sorted({int(m) for m in marks if 0 <= int(m) <= len(ids)})
    out: dict[int, str] = {}
    h, iw = FNV_BASIS, 0
    for i in range(len(ids) + 1):
        while iw < len(want) and want[iw] == i:
            out[i] = f"{h:016x}"
            iw += 1
        if iw >= len(want) or i == len(ids):
            break
        t = int(ids[i]) & 0xFFFFFFFF
        for b in range(4):
            h ^= (t >> (8 * b)) & 0xFF
            h = (h * FNV_PRIME) & _M64
    return out


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------- the KPF-11 fork-position predictor
def slot_by_id(slots: list[dict], sid: int) -> dict | None:
    for s in slots or []:
        if s.get("id") == sid:
            return s
    return None


def checkpoint_set(slot: dict | None) -> list[dict]:
    """The checkpoints /slots reports for one slot, reduced to what the rule reads."""
    out = []
    for c in (slot or {}).get("checkpoints") or []:
        out.append({"n_tokens": int(c.get("n_tokens", 0)), "pos_max": c.get("pos_max"), "pinned": bool(c.get("pinned")),
                    "prefix_hash": c.get("prefix_hash") or ""})
    return out


def predict_fork(slots: list[dict], task_ids: list[int], dst_id: int, n_min: int, *, mode: str = "checkpoint",
                 n_swa: int = 0, lcp_by_slot: dict[int, int] | None = None, enabled: bool = True) -> dict:
    """What `find_fork_source` selects for a task sent to slot `dst_id`, from a /slots snapshot taken BEFORE the request.

    KPF-11 rule, checkpoint mode (recurrent/hybrid targets): the largest p over every other slot such that p is a
    checkpoint of that slot with  0 < p < n_task,  p >= n_min,  prefix_hash(task[:p]) == the checkpoint's hash (i.e.
    p <= lcp) and pos_max + 1 == p; plus kind "end": an IDLE source whose whole prompt is a prefix of the task
    (n_src == lcp < n_task, n_src >= n_min, no SWA). Ties: idle first, then the lowest slot id (the server prefers the
    most recently used; unknown from /slots, so a tie is flagged `tie`).
    KV mode (plain-attention targets): p = min(lcp, n_task - 1) for every source with lcp >= n_min; needs the exact
    lcp, supplied as `lcp_by_slot` (the runner keeps a shadow of what it sent to each slot).
    Not modelled: LoRA mismatch, WAIT_OTHER children, kv_altered slots, a source's posmax guard in KV mode.
    """
    n_task = len(task_ids)
    res = {"p": 0, "src": None, "kind": "none", "idle": None, "tie": False, "mode": mode, "n_min": n_min,
           "n_task": n_task, "candidates": [], "enabled": bool(enabled)}
    if not enabled or n_task <= 0:
        return res
    marks = set()
    for s in slots or []:
        if s.get("id") == dst_id:
            continue
        marks.add(int(s.get("n_prompt_tokens") or 0))
        for c in s.get("checkpoints") or []:
            marks.add(int(c.get("n_tokens", 0)))
    ladder = prefix_hashes(task_ids, [m for m in marks if 0 < m <= n_task])
    cands: list[dict] = []
    for s in sorted(slots or [], key=lambda x: x.get("id", 0)):
        sid = s.get("id")
        if sid == dst_id:
            continue
        n_src = int(s.get("n_prompt_tokens") or 0)
        if n_src <= 0:
            continue
        idle = not s.get("is_processing")
        if mode == "kv":
            l_ = (lcp_by_slot or {}).get(sid, 0)
            if l_ >= n_min and l_ > 0:
                cands.append({"src": sid, "p": min(l_, n_task - 1), "kind": "kv", "idle": idle})
            continue
        if idle and n_swa == 0 and 0 < n_src < n_task and n_src >= n_min and ladder.get(n_src) == s.get("prefix_hash"):
            cands.append({"src": sid, "p": n_src, "kind": "end", "idle": idle})
        for c in s.get("checkpoints") or []:
            p = int(c.get("n_tokens", 0))
            if p <= 0 or p >= n_task or p < n_min:
                continue
            if ladder.get(p) != c.get("prefix_hash"):
                continue                                    # p > lcp: this checkpoint is not a prefix of the task
            if c.get("pos_max") is not None and int(c["pos_max"]) + 1 != p:
                continue
            cands.append({"src": sid, "p": p, "kind": "checkpoint", "idle": idle})
    res["candidates"] = cands
    if not cands:
        return res
    best_p = max(c["p"] for c in cands)
    top = [c for c in cands if c["p"] == best_p]
    top.sort(key=lambda c: (not c["idle"], c["src"]))
    res.update(p=best_p, src=top[0]["src"], kind=top[0]["kind"], idle=top[0]["idle"],
               tie=len({(c["src"], c["kind"]) for c in top}) > 1)
    return res


def components(pred: dict, lcp_client: int | None, n_task: int, server_comps: dict | None) -> dict:
    """The per-component hit lengths KPF-18.2 item 2 wants (tgt_kv_hit, recurrent_ckpt_pos, drafter_kv_hit).
    The server does not report them today. tgt_kv_hit and recurrent_ckpt_pos are DERIVED client-side (lcp of the task
    with what the chosen source holds, capped at n_task - 1; the position of the checkpoint the rule picks);
    drafter_kv_hit is read from /slots[dst].fork if the build exposes it, else None = NOT MEASURED (never a guess).
    A server-reported value overrides the client-derived one and `source` says which. See SERVER_FORK_FIELDS."""
    sc = dict(server_comps or {})
    out = {"tgt_kv_hit": None, "recurrent_ckpt_pos": None, "drafter_kv_hit": None, "source": {}}
    if lcp_client is not None:
        out["tgt_kv_hit"] = min(int(lcp_client), max(0, n_task - 1))
        out["source"]["tgt_kv_hit"] = "client"
    if pred.get("kind") in ("checkpoint", "end"):
        out["recurrent_ckpt_pos"] = int(pred["p"])
        out["source"]["recurrent_ckpt_pos"] = "client-predicted"
    for k in ("tgt_kv_hit", "recurrent_ckpt_pos", "drafter_kv_hit"):
        if sc.get(k) is not None:
            out[k] = int(sc[k])
            out["source"][k] = "server"
    out["drafter_exposed"] = sc.get("drafter_kv_hit") is not None
    return out


SERVER_FORK_FIELDS = ("tgt_kv_hit", "recurrent_ckpt_pos", "drafter_kv_hit")


def server_components(slot: dict | None) -> dict:
    f = (slot or {}).get("fork") or {}
    return {k: f.get(k) for k in SERVER_FORK_FIELDS if f.get(k) is not None}


def make_cell(*, arm: str, cell_id: str, role: str, slot: int, n_task: int, lcp_client: int | None, timings: dict | None,
              n_fork_slot: int | None, pred: dict, ckpts_before: list[dict], expected: int | None = None,
              error: str | None = None, **extra) -> dict:
    """One sibling / cell, raw: nothing is dropped, errors and abstentions are cells too."""
    t = timings or {}
    n_fork = int(t.get("n_fork_tokens") or 0)
    c = {"arm": arm, "cell": cell_id, "role": role, "slot": slot, "n_task": n_task, "lcp_client": lcp_client,
         "n_fork_tokens": None if error else n_fork, "n_fork_tokens_slot": n_fork_slot,
         "fork_src_kind": t.get("fork_src_kind"), "fork_src_slot": t.get("fork_src_slot"),
         "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
         "predicted_p": pred.get("p"), "predicted_kind": pred.get("kind"), "predicted_src": pred.get("src"),
         "predicted_tie": pred.get("tie"), "n_candidates": len(pred.get("candidates") or []),
         "expected": expected, "checkpoint_set_before": [x["n_tokens"] for x in ckpts_before],
         "pinned_before": [x["n_tokens"] for x in ckpts_before if x["pinned"]], "error": error}
    c.update(extra)
    return c


# ---------------------------------------------------------------- predeclared rules (counts, not a pass word)
def _ok(c: dict) -> bool:
    return not c.get("error") and c.get("n_fork_tokens") is not None


def tally_expected(cells: list[dict]) -> dict:
    """Cells with an `expected` n_fork_tokens: how many equal it. Errors and abstentions are counted, never dropped."""
    use = [c for c in cells if c.get("expected") is not None]
    err = [c for c in use if not _ok(c)]
    eq = [c for c in use if _ok(c) and c["n_fork_tokens"] == c["expected"]]
    return {"n": len(use), "n_error": len(err), "n_equal_expected": len(eq),
            "n_not_equal_expected": len(use) - len(err) - len(eq),
            "observed": [c.get("n_fork_tokens") for c in use], "expected": [c["expected"] for c in use],
            "predeclared_rule_satisfied": (len(eq) == len(use)) if use else None}


def tally_prediction(cells: list[dict]) -> dict:
    """Observed n_fork_tokens vs the KPF-11 predictor, plus the 'zero while a usable checkpoint exists' defect count."""
    use = [c for c in cells if c.get("predicted_p") is not None]
    ok = [c for c in use if _ok(c)]
    eq = [c for c in ok if c["n_fork_tokens"] == c["predicted_p"]]
    zero_pos = [c["cell"] for c in ok if c["n_fork_tokens"] == 0 and c["predicted_p"] > 0]
    return {"n": len(use), "n_error": len(use) - len(ok), "n_equal_predicted": len(eq),
            "n_not_equal_predicted": len(ok) - len(eq), "n_zero_while_checkpoint_usable": len(zero_pos),
            "zero_while_checkpoint_usable": zero_pos,
            "n_ties": sum(1 for c in use if c.get("predicted_tie"))}


def within_floor(vs: dict | None, floor_max_abs: float | None, floor_max_kld: float | None) -> dict:
    """vs: gate.compare() dict child-vs-fresh. First-token top-1 equal AND max-abs prob and coarse KL no larger than the
    control floor (max over the run's fresh-vs-fresh pairs), both floors lifted to FLOOR_EPS."""
    if not vs:
        return {"top1": None, "maxabs_ok": None, "kld_ok": None, "within": None}
    fa = max(floor_max_abs or 0.0, FLOOR_EPS)
    fk = max(floor_max_kld or 0.0, FLOOR_EPS)
    m, k = vs.get("maxabs_union"), vs.get("kld")
    top1 = bool(vs.get("top1"))
    mo = (m is not None) and m <= fa
    ko = (k is not None) and k <= fk
    return {"top1": top1, "maxabs_ok": mo, "kld_ok": ko, "within": bool(top1 and mo and ko),
            "maxabs": m, "kld": k, "floor_maxabs_used": fa, "floor_kld_used": fk}


def floor_stats(floors: list[dict]) -> dict:
    mu = [f["maxabs_union"] for f in floors if f and f.get("maxabs_union") is not None]
    kl = [f["kld"] for f in floors if f and f.get("kld") is not None]
    pre = sorted(f["prefix"] for f in floors if f)
    return {"n": len(floors), "max_maxabs": max(mu) if mu else None, "max_kld": max(kl) if kl else None,
            "prefix_median": statistics.median(pre) if pre else None, "prefix_min": pre[0] if pre else None}


def median_or_none(xs: list[float]):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rules_for(items: list[int]) -> dict:
    out = {}
    for it in items:
        if it in RULES:
            out[str(it)] = {"task": TASK_OF[it], **RULES[it]}
    return out


def rules_sha256(items: list[int]) -> str | None:
    r = rules_for(items)
    if not r:
        return None
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- drafter configurations
_SPEC_EXACT = {"-md", "--model-draft", "-ngld", "--n-gpu-layers-draft", "--gpu-layers-draft", "--device-draft", "-devd",
               "-cd", "--ctx-size-draft", "-td", "--threads-draft", "-tbd", "--threads-batch-draft", "-ctkd",
               "--cache-type-k-draft", "-ctvd", "--cache-type-v-draft"}


def _is_spec_flag(t: str) -> bool:
    return t in _SPEC_EXACT or t.startswith("--spec-") or t.startswith("--draft")


def flag_value(argv: list[str], names: tuple[str, ...]) -> str | None:
    for i, t in enumerate(argv):
        if t in names and i + 1 < len(argv):
            return argv[i + 1]
        for n in names:
            if t.startswith(n + "="):
                return t.split("=", 1)[1]
    return None


def strip_speculative(argv: list[str]) -> list[str]:
    """The same argv with every speculative / draft-model flag (and its value) removed: the 'no drafter' launch."""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        if _is_spec_flag(t):
            if "=" not in t and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        out.append(t)
        i += 1
    return out


def config_name(argv: list[str]) -> str:
    st = flag_value(argv, ("--spec-type",))
    md = flag_value(argv, ("-md", "--model-draft"))
    if st == "draft-mtp":
        return "mtp"
    if st == "draft-dflash":
        return "dflash2"
    if st or md:
        return f"draft-{st or 'model'}"
    return "nodraft"


def parse_configs(spec: str, extra: list[str] | None, base_argv: list[str],
                  load_argv: Callable[[str], list[str]]) -> list[tuple[str, list[str]]]:
    """`--kpf-configs` (comma list of 'as-given' | 'nodraft') plus repeated `--kpf-config-argv NAME=ARGVFILE`.
    Returns [(name, argv)], the first entry being the launched shape; names are unique."""
    out: list[tuple[str, list[str]]] = []
    seen: set[str] = set()

    def add(name: str, argv: list[str]) -> None:
        if name in seen:
            return
        seen.add(name)
        out.append((name, list(argv)))
    for tok in [t.strip() for t in (spec or "as-given").split(",") if t.strip()]:
        if tok == "as-given":
            add(config_name(base_argv), base_argv)
        elif tok == "nodraft":
            add("nodraft", strip_speculative(base_argv))
        else:
            raise ValueError(f"--kpf-configs: unknown token {tok!r} (as-given | nodraft)")
    for e in extra or []:
        name, _, path = e.partition("=")
        if not name or not path:
            raise ValueError(f"--kpf-config-argv expects NAME=ARGVFILE, got {e!r}")
        add(name, load_argv(path))
    return out


def ub_from_argv(argv: list[str], default: int = 512) -> int:
    v = flag_value(argv, ("-ub", "--ubatch-size"))
    return int(v) if v and v.isdigit() else default


# ---------------------------------------------------------------- planning helpers (pure)
def sweep_lengths(t_avail: int, ub: int, n: int) -> list[int]:
    """>= n trunk lengths at a fixed stride across ONE -ub period, all <= t_avail: L0 + k*stride, k < n, stride = ub // n,
    L0 the largest multiple of ub with L0 + ub <= t_avail. [] if the trunk cannot hold them."""
    if n < 1 or ub < n or t_avail < 2 * ub:
        return []
    stride = ub // n
    l0 = ((t_avail - ub) // ub) * ub
    if l0 <= 0:
        return []
    return [l0 + k * stride for k in range(n)]


def d_positions(junction: int, n_src: int, ub: int) -> dict[str, int]:
    """The checkpoint positions in force on a source of n_src tokens whose request pinned `junction`: the junction and
    the two near-end ones the server makes ({4 + n_ubatch, 4} tokens before the prompt end)."""
    return {"junction": junction, "n_minus_4_minus_ub": n_src - 4 - ub, "n_minus_4": n_src - 4}


def d_cells(positions: dict[str, int], deltas=(-1, 0, 1)) -> list[dict]:
    return [{"name": f"{nm}{d:+d}", "position_name": nm, "c": c, "delta": d, "lcp": c + d}
            for nm, c in positions.items() for d in deltas]


def holdout_lcp(positions: dict[str, int]) -> int:
    """A shared-prefix length at least 3 tokens away from every c-1, c, c+1 cell, between the junction and the first
    near-end checkpoint, so the rule can be checked at a length that was not used to set it."""
    vals = sorted(positions.values())
    h = (vals[0] + vals[1]) // 2 + 13
    while any(abs(h - v) <= 3 for v in vals):
        h += 7
    return h


def divergent_suffix(ids_after: list[list[int]], avoid_token: int | None, take: int) -> tuple[int, list[int]] | None:
    """The first candidate suffix whose first token differs from `avoid_token` (so the shared prefix ends exactly where
    intended), cut to `take` tokens. Returns (index, ids) or None."""
    for i, s in enumerate(ids_after):
        if s and (avoid_token is None or s[0] != avoid_token):
            return i, list(s[:take])
    return None


def dense_positions(n_prompt: int, u: int) -> list[int]:
    """Dense policy: checkpoint_at every u tokens, at most N_CKPT_AT_MAX of them, all strictly inside the prompt."""
    return [u * k for k in range(1, N_CKPT_AT_MAX + 1) if u * k < n_prompt]


def classify_end_ckpt(pinned_n_tokens: list[int], n_prompt: int, n_generated: int) -> dict:
    """Where does the pinned `checkpoint_at [-1]` checkpoint sit: at the prompt end, or at the generation end?"""
    at_prompt = [n for n in pinned_n_tokens if n == n_prompt]
    at_gen = [n for n in pinned_n_tokens if n_generated > 0 and n in (n_prompt + n_generated, n_prompt + n_generated - 1)]
    other = [n for n in pinned_n_tokens if n not in at_prompt and n not in at_gen]
    where = "prompt_end" if at_prompt else ("generation_end" if at_gen else ("other" if other else "absent"))
    return {"where": where, "pinned": sorted(pinned_n_tokens), "n_prompt": n_prompt, "n_generated": n_generated,
            "at_prompt_end": bool(at_prompt), "at_generation_end": bool(at_gen), "other": other}


def multi_turn_gap(n_turn1_prompt: int, n_reply: int, n_fork_tokens: int | None) -> dict:
    need = n_turn1_prompt + n_reply
    if n_fork_tokens is None:
        return {"required": need, "n_fork_tokens": None, "gap": None, "covered": None}
    return {"required": need, "n_fork_tokens": n_fork_tokens, "gap": need - n_fork_tokens,
            "covered": n_fork_tokens >= need}


def logprob_of_chosen(entry: dict) -> float | None:
    """log-probability of the sampled token in one completion_probabilities entry."""
    if entry.get("logprob") is not None:
        return float(entry["logprob"])
    tid = entry.get("id")
    for t in entry.get("top_logprobs") or []:
        if t.get("id") == tid and t.get("logprob") is not None:
            return float(t["logprob"])
    return None


def mean_abs_dlogprob(a: dict, b: dict, upto: int) -> dict:
    """mean |logprob_a - logprob_b| of the sampled token over the first `upto` generated positions (callers pass the
    greedy prefix common to every run being compared, so both sides share the context at every position)."""
    pa, pb = a.get("probs") or [], b.get("probs") or []
    ds = []
    for i in range(min(upto, len(pa), len(pb))):
        x, y = logprob_of_chosen(pa[i]), logprob_of_chosen(pb[i])
        if x is not None and y is not None:
            ds.append(abs(x - y))
    return {"n_positions": len(ds), "mean_abs_dlogprob": (sum(ds) / len(ds)) if ds else None,
            "max_abs_dlogprob": max(ds) if ds else None}


def common_prefix_len(runs: list[dict]) -> int:
    toks = [r.get("tokens") or [] for r in runs]
    if not toks:
        return 0
    n = min(len(t) for t in toks)
    i = 0
    while i < n and all(t[i] == toks[0][i] for t in toks):
        i += 1
    return i


# ---------------------------------------------------------------- CLI
def add_arguments(ap) -> None:
    g = ap.add_argument_group("KPF-18.1/.2/.3 arms (commands kpf181 | kpf182 | kpf183 | kpf18arms)")
    g.add_argument("--kpf-configs", default="as-given,nodraft",
                   help="drafter launch configurations: as-given (the argv file) and/or nodraft (it with every "
                        "speculative flag stripped)")
    g.add_argument("--kpf-config-argv", action="append", default=[], metavar="NAME=ARGVFILE",
                   help="an extra drafter configuration from its own argv file (e.g. dflash2=..., mtp=...)")
    g.add_argument("--kpf-n-swa", type=int, default=0, help="sliding-window width of the target (0 = none); the "
                   "predictor's end-kind source needs 0")
    g.add_argument("--kpf181-arms", default="A,B,C,D")
    g.add_argument("--kpf181-groups", default=None, help="trunk groups for arms A/B/D (default 0)")
    g.add_argument("--kpf181-sibs", type=int, default=3)
    g.add_argument("--kpf181-lengths", type=int, default=16, help="arm C: lengths across one -ub period (>= 16)")
    g.add_argument("--kpf181-c-group", default=None, help="arm C trunk group (default 0)")
    g.add_argument("--kpf182-parts", default="1,2,3,4,5")
    g.add_argument("--kpf182-groups", default=None, help="trunk groups for the parity/attribution protocol (default 0,2)")
    g.add_argument("--kpf182-sibs", type=int, default=3)
    g.add_argument("--kpf182-dense-u", type=int, default=1024, help="policy control: dense checkpoint spacing U")
    g.add_argument("--kpf183-parts", default=None, help="1 cost, 2 end-ckpt position + multi-turn, 3 lane, 4 >=50k "
                   "warm-vs-cold (default: GPU 1,4 ; --cpu 2,3)")
    g.add_argument("--kpf183-groups", default=None, help="part 1 trunk groups (default 0,3)")
    g.add_argument("--kpf183-reps", type=int, default=5, help="part 1 repeats per arm (A B A B ...)")
    g.add_argument("--kpf183-reply", type=int, default=64, help="part 2 turn-1 reply length")
    g.add_argument("--kpf183-long-group", default=None, help="part 4 trunk group (default: the first with T >= 50000)")
    g.add_argument("--kpf183-chat-probe-tokens", type=int, default=1500)


def groups_arg(spec: str | None, default: list[int]) -> list[int]:
    if not spec:
        return list(default)
    return [int(x) for x in spec.split(",") if x.strip() != ""]


# ---------------------------------------------------------------- wiring into the gate
def register(items: dict, gate) -> None:
    """Add the arm builders to the gate's ITEMS table. `gate` is the gate module object (never re-imported by name)."""
    for it, fn in BUILDERS.items():
        items[it] = (lambda ctx, fn=fn: fn(ctx, gate))


def preflight_arms(ctx, items: list[int], gate) -> tuple[list[str], list[str]]:
    probs: list[str] = []
    notes: list[str] = []
    for it in items:
        if it in PREFLIGHTS:
            p, n = PREFLIGHTS[it](ctx, gate)
            probs += p
            notes += n
    return probs, notes


# ================================================================ step toolkit (takes the gate module as `gate`)
import re  # noqa: E402
from pathlib import Path  # noqa: E402

_FORK_MODE_RE = re.compile(r"mode = (\w+)")
_SERVER_CLS: dict[int, type] = {}
CHAMPION_TIP = "e33229348"


def server_cls(gate):
    """gate.Server with its argv taken from a per-configuration base instead of ctx.gate_argv (drafter configs)."""
    if id(gate) in _SERVER_CLS:
        return _SERVER_CLS[id(gate)]

    class ArmServer(gate.Server):
        def __init__(self, ctx, label, binary, fork_min, argv_base, **kw):
            self._argv_base = list(argv_base)
            super().__init__(ctx, label, binary, fork_min, **kw)

        def build_argv(self):
            a = gate.set_flag(list(self._argv_base), ("--slot-save-path",),
                              str(self.ctx.out / "slot-save" / self.label))
            if self.cache_idle is not None:
                a = gate.set_bool(a, "--cache-idle-slots", "--no-cache-idle-slots", self.cache_idle)
            a += ["--port", str(self.ctx.args.port), "--metrics"]
            if self.fork_min is not None:
                a += ["--slot-fork-min-tokens", str(self.fork_min)]
            a += self.extra_args
            return [str(self.binary)] + a

    _SERVER_CLS[id(gate)] = ArmServer
    return ArmServer


def get_configs(ctx, gate) -> list[tuple[str, list[str]]]:
    if not hasattr(ctx, "_kpf_cfgs"):
        a = ctx.args
        ctx._kpf_cfgs = parse_configs(a.kpf_configs, a.kpf_config_argv, ctx.gate_argv,
                                      lambda p: gate.parse_argv_file(Path(p))[0])
    return ctx._kpf_cfgs


class Env:
    """One launched server + the runner-side shadow of what each slot holds. Everything an arm does to the server goes
    through `send` so every request leaves a raw cell (prediction, n_fork_tokens from the timings AND from /slots,
    checkpoint sets, errors) and a VB-KVU-PF observation."""

    def __init__(self, ctx, gate, srv, cfg_name: str) -> None:
        self.ctx, self.gate, self.srv, self.cfg_name = ctx, gate, srv, cfg_name
        self.shadow: dict[int, list[int]] = {}

    # -- server facts
    @property
    def n_min(self) -> int:
        return int(self.srv.fork_min or 0)

    @property
    def enabled(self) -> bool:
        return self.n_min > 0

    def fork_mode(self) -> str:
        line = (self.srv.readback or {}).get("slot_fork_from_launch_log") or ""
        m = _FORK_MODE_RE.search(line)
        return m.group(1) if m else "checkpoint"

    def counters(self) -> dict:
        m = self.ctx.client.metrics()
        return {k: m.get(f"llamacpp:{k}") for k in ("slot_forks_total", "slot_fork_tokens_total")}

    # -- slot hygiene (also clears the shadow)
    def erase(self, *slots: int) -> None:
        for s in slots:
            self.ctx.client.erase(s)
            self.shadow.pop(s, None)

    def erase_all(self) -> None:
        self.erase(0, 1, 2, 3)

    # -- one request, one cell
    def send(self, R, role: str, ids: list[int], slot: int, *, n_predict: int, arm: str, cell_id: str,
             ckpt: list[int] | None = None, fork: bool | None = True, cache: bool = True, dflash: bool = True,
             n_probs: int = 0, expected: int | None = None, **meta) -> tuple[dict, dict]:
        gate, ctx = self.gate, self.ctx
        snap = ctx.client.slots()
        others = [s for s in snap if s.get("id") != slot]
        lcp_map = {s["id"]: lcp(self.shadow.get(s["id"], []), ids) for s in others}
        pred = predict_fork(snap, ids, slot, self.n_min, mode=self.fork_mode(), n_swa=ctx.args.kpf_n_swa,
                            lcp_by_slot=lcp_map, enabled=bool(self.enabled and fork and cache))
        ck_before = [{"n_tokens": c["n_tokens"], "pinned": c["pinned"], "slot": s["id"]}
                     for s in others for c in checkpoint_set(s)]
        sets = {s["id"]: [c["n_tokens"] for c in checkpoint_set(s)] for s in others}
        lcp_best = max(lcp_map.values()) if lcp_map else 0
        r = ctx.client.complete(gate.body_for(ids, n_predict, slot, fork=fork, cache=cache, dflash=dflash, ckpt=ckpt,
                                              n_probs=n_probs))
        common = dict(arm=arm, cell_id=cell_id, role=role, slot=slot, n_task=len(ids), lcp_client=lcp_best,
                      pred=pred, ckpts_before=ck_before, expected=expected, ckpt_sets_by_slot=sets,
                      checkpoint_at=ckpt, cache_prompt=cache, fork_requested=fork, dflash_requested=dflash,
                      n_predict=n_predict, **meta)
        try:
            R.row(role, r, slot=slot, cell=cell_id, **{k: v for k, v in meta.items() if k not in ("lane",)})
        except gate.Abort as exc:
            R.extra.setdefault("cells", []).append(make_cell(timings=r.get("timings"), n_fork_slot=None,
                                                             error=str(exc)[:400], **common))
            raise
        after = slot_by_id(ctx.client.slots(), slot) or {}
        n_fork_slot = (after.get("fork") or {}).get("n_fork_tokens")
        self.shadow[slot] = list(ids) + list(r.get("tokens") or [])
        cell = make_cell(timings=r.get("timings"), n_fork_slot=n_fork_slot, **common)
        sc = server_components(after)
        comp = components(pred, lcp_best, len(ids), sc)
        cell["components"] = comp
        n_fork = cell["n_fork_tokens"] or 0
        cell["tokens_lost_to_missing_checkpoint"] = max(0, min(lcp_best, len(ids) - 1) - n_fork) if self.enabled and fork and cache else 0
        R.extra.setdefault("cells", []).append(cell)
        flat = sorted({x["n_tokens"] for x in ck_before})
        R.obs(role, checkpoint_set=flat, tgt_kv_hit=comp["tgt_kv_hit"], recurrent_ckpt_pos=comp["recurrent_ckpt_pos"],
              drafter_kv_hit=comp["drafter_kv_hit"], n_fork_tokens=n_fork,
              tokens_lost_to_missing_checkpoint=cell["tokens_lost_to_missing_checkpoint"], lane=meta.get("lane") or "completion_raw",
              pinned_count=sum(1 for x in ck_before if x["pinned"]), cell=cell_id)
        return r, cell


def controls_block(env: Env, item: int) -> dict:
    return {"manifest": "prompt_set_sha256 + binary_sha256 + argv (read back from /proc and the launch log) are in this "
                        "record; per-sibling raw cells (n_fork_tokens from timings and from /slots, prediction, "
                        "checkpoint sets, errors) are under `cells`",
            "baseline": f"the fork-off arm (--slot-fork-min-tokens 0) of the same command on the same binary, and "
                        f"champion tip {CHAMPION_TIP} (item 6 / KPF-18.0b)",
            "held_out": "lengths or prompts used to SET a rule are never used to judge it (arm D holdout cell; arm C/D "
                        "positions come from the server's own checkpoint offsets, not from fitted numbers)",
            "counted": "every sibling is a cell, including errors, aborts and abstentions",
            "reported_as": "counts under rule_result; no pass word",
            "config": env.cfg_name, "kpf_task": TASK_OF[item]}


def open_run(env: Env, item: int, arm: str, *, dflash2: bool = True, rule: str | None = None, **extra):
    R = env.gate.Run(env.ctx, item, arm, env.srv, dflash2)
    R.extra.update(kpf_task=TASK_OF[item], kpf18_arm=arm, kpf_config=env.cfg_name, fork_min_tokens=env.n_min,
                   fork_mode=env.fork_mode(), predeclared_rule=rule, predeclared_rules_sha256=rules_sha256([item]),
                   cells=[], controls=controls_block(env, item), **extra)
    R.extra["counters_start"] = env.counters()
    return R


def close_counts(env: Env, R) -> None:
    """Anti-vacuity: the server's own fork counters over this run's window against the per-request sums."""
    c1 = env.counters()
    c0 = R.extra.get("counters_start") or {}
    d = {k: (None if c1.get(k) is None or c0.get(k) is None else c1[k] - c0[k]) for k in c1}
    cells = R.extra.get("cells") or []
    n_forks = sum(1 for c in cells if (c.get("n_fork_tokens") or 0) > 0)
    tok = sum(int(c.get("n_fork_tokens") or 0) for c in cells)
    R.extra["counters_end"] = c1
    R.extra["counters_delta"] = d
    R.extra["armed"] = {"n_requests": len(cells), "n_requests_with_fork": n_forks, "sum_n_fork_tokens": tok,
                        "checkpoint_sets_nonempty": sum(1 for c in cells if c.get("pinned_before") or c.get("checkpoint_set_before"))}
    if d.get("slot_forks_total") is not None:
        R.check("counter_slot_forks_equal_requests_with_fork", d["slot_forks_total"] == n_forks, [d["slot_forks_total"], n_forks])
    if d.get("slot_fork_tokens_total") is not None:
        R.check("counter_fork_tokens_equal_sum_n_fork", d["slot_fork_tokens_total"] == tok, [d["slot_fork_tokens_total"], tok])
    slot_mismatch = [c["cell"] for c in cells if c.get("n_fork_tokens_slot") is not None and c.get("n_fork_tokens") is not None
                     and c["n_fork_tokens_slot"] != c["n_fork_tokens"]]
    R.check("n_fork_tokens_timings_equal_slot", not slot_mismatch, slot_mismatch)


def role_cells(R, role: str) -> list[dict]:
    return [c for c in (R.extra.get("cells") or []) if c.get("role") == role]


def est_start(gate) -> float:
    return gate.LINK_S + gate.LOAD_S


# ================================================================ KPF-18.1  hit-length rows A-D
RULES[181] = {
    "common": ("Matched instrument: the same binary, --slot-fork-min-tokens off (0) vs on (N), the same prompt set and "
               "argv. Raw counts per sibling (n_fork_tokens from the request timings and from /slots, the KPF-11 "
               "prediction, the checkpoint set that existed) are kept; every sibling counts, including errors; "
               "results are COUNTS under rule_result, never a pass word. Held-out rule: the predictor is the handoff's "
               "KPF-11 rule (largest checkpoint n_tokens <= lcp with p < task size); nothing is fitted on these cells."),
    "A": ("A_ckpt: slot 0 prefills trunk(T)+private tail with checkpoint_at=[T]; each sibling = trunk + its own suffix "
          "(distinct first tokens) on slot 1, one at a time, fork on. SATISFIED iff every sibling n_fork_tokens == T. "
          "A_tailonly: same without checkpoint_at; RECORDED only (expected near 0: the source's near-end checkpoints lie "
          "past the shared prefix): counts of n_fork_tokens == 0 and the max. Fork-off arms (min tokens 0): SATISFIED "
          "iff every sibling n_fork_tokens == 0."),
    "B": ("B1 identical resend of trunk+suffix0 (source without checkpoint_at), B2 the same with checkpoint_at [-1], "
          "B3 child == trunk with the source = trunk+suffix0 pinned at T, B4 child == trunk with the source = the trunk "
          "alone. Each child is on slot 1, fork on, after the source's own n_predict-0 prefill. SATISFIED iff (a) every "
          "child n_fork_tokens equals the KPF-11 prediction, (b) no child has n_fork_tokens == 0 while a usable "
          "pre-end checkpoint existed in /slots (0 < n_tokens <= min(lcp, n_task-1), n_tokens >= min tokens, hash "
          "equal), (c) B1, B2, B4 have n_fork_tokens > 0, (d) every child's first-token top-1 equals the cold run's "
          "and its max-abs prob / coarse KL are within max(fresh-vs-fresh floor, 1e-4) of this run. Fork-off: every "
          "child n_fork_tokens == 0."),
    "C": ("Trunk-length sweep: >= 16 lengths L0 + k*stride (stride = ub // 16, one -ub period), one trunk group; for each "
          "length slot 0 prefills trunk[:L]+tail with checkpoint_at=[L]; the fork child (trunk[:L]+suffix0), then two "
          "cold runs (cache_prompt false) on slots 2 and 3 give the fresh-vs-fresh floor. Run with the drafter on and "
          "off (speculative.n_max 0 on every request). SATISFIED iff for EVERY length n_fork_tokens == L and the "
          "child is within the floor: top-1 equal, max-abs prob and coarse KL each <= max(the largest "
          "fresh-vs-fresh value over this run's lengths, 1e-4)."),
    "D": ("Hit-count prediction: one source (trunk T + long tail, checkpoint_at=[T]) holds the junction T and the two "
          "near-end checkpoints N-4-ub and N-4. For each c in {T, N-4-ub, N-4} and delta in {-1, 0, +1} a child "
          "shares exactly c+delta tokens with the source and then diverges (distinct next token). One launch per "
          "drafter configuration (as given, no drafter, others given). SATISFIED for a configuration iff every cell "
          "n_fork_tokens == the KPF-11 prediction and no cell has n_fork_tokens == 0 while the prediction is > 0 "
          "(an unidentified drafter configuration must not disable fork reuse). A drafter-dependent drop is "
          "MEASURED once, at a held-out shared length that is not any c-1/c/c+1: drop = predicted - observed there, "
          "plus a cold-run first-token comparison at that length; the drop is recorded, never subtracted from the "
          "main cells."),
}


def _sib(cells: list[dict], role: str = "sibling") -> list[dict]:
    return [c for c in cells if c.get("role") == role]


def eval_181_A(cells: list[dict]) -> dict:
    s = _sib(cells)
    t = tally_expected(s)
    vals = [c["n_fork_tokens"] for c in s if _ok(c)]
    t.update(n_siblings=len(s), n_fork_min=min(vals) if vals else None, n_fork_max=max(vals) if vals else None,
             n_fork_zero=sum(1 for v in vals if v == 0), n_fork_positive=sum(1 for v in vals if v > 0),
             n_error_cells=sum(1 for c in s if not _ok(c)))
    return t


def eval_181_B(cells: list[dict], floors: list[dict], *, fork_on: bool = True) -> dict:
    ch = _sib(cells, "child")
    if not fork_on:
        for c in ch:
            c["expected"] = 0
        t = tally_expected(ch)
        t["n_children"] = len(ch)
        return t
    pred = tally_prediction(ch)
    fs = floor_stats(floors)
    must = [c for c in ch if c.get("must_hit")]
    must_zero = [c["cell"] for c in must if _ok(c) and c["n_fork_tokens"] == 0]
    within = [within_floor(c.get("vs_fresh"), fs["max_maxabs"], fs["max_kld"]) for c in ch if _ok(c)]
    n_within = sum(1 for w in within if w["within"])
    n_not = sum(1 for w in within if w["within"] is False)
    n_unmeasured = sum(1 for w in within if w["within"] is None)
    sat = (pred["n_error"] == 0 and pred["n_not_equal_predicted"] == 0 and pred["n_zero_while_checkpoint_usable"] == 0
           and not must_zero and n_not == 0 and n_unmeasured == 0 and len(ch) > 0)
    return {"n_children": len(ch), **pred, "n_must_hit": len(must), "n_must_hit_got_zero": len(must_zero),
            "must_hit_got_zero": must_zero, "n_within_floor": n_within, "n_outside_floor": n_not,
            "n_floor_unmeasured": n_unmeasured, "floor": fs, "predeclared_rule_satisfied": sat}


def eval_181_C(cells: list[dict], floors: list[dict]) -> dict:
    ch = _sib(cells, "child")
    fs = floor_stats(floors)
    rows = []
    for c in ch:
        w = within_floor(c.get("vs_fresh"), fs["max_maxabs"], fs["max_kld"])
        rows.append({"cell": c["cell"], "L": c.get("expected"), "n_fork_tokens": c.get("n_fork_tokens"),
                     "forked_at_L": _ok(c) and c["n_fork_tokens"] == c.get("expected"), "top1": w.get("top1"),
                     "within": w.get("within"), "prefix_vs_fresh": (c.get("vs_fresh") or {}).get("prefix")})
    pre = [r["prefix_vs_fresh"] for r in rows if r["prefix_vs_fresh"] is not None]
    n_fork_ok = sum(1 for r in rows if r["forked_at_L"])
    n_within = sum(1 for r in rows if r["within"])
    return {"n_lengths": len(ch), "n_forked_at_L": n_fork_ok, "n_not_forked_at_L": len(ch) - n_fork_ok,
            "n_top1_equal": sum(1 for r in rows if r["top1"]), "n_within_floor": n_within,
            "n_outside_floor": sum(1 for r in rows if r["within"] is False), "floor": fs,
            "greedy_prefix_vs_fresh_median": median_or_none(pre), "greedy_prefix_floor_median": fs["prefix_median"],
            "lengths": rows,
            "predeclared_rule_satisfied": bool(ch) and n_fork_ok == len(ch) and n_within == len(ch)}


def eval_181_D(cells: list[dict]) -> dict:
    main = [c for c in _sib(cells, "child") if not c.get("holdout")]
    hold = [c for c in _sib(cells, "child") if c.get("holdout")]
    pred = tally_prediction(main)
    drop = None
    if hold and _ok(hold[0]) and hold[0].get("predicted_p") is not None:
        h = hold[0]
        d = h["predicted_p"] - h["n_fork_tokens"]
        drop = {"cell": h["cell"], "lcp": h.get("lcp_client"), "predicted": h["predicted_p"], "observed": h["n_fork_tokens"],
                "drop": d, "kind": "none" if d == 0 else ("full" if h["n_fork_tokens"] == 0 else "partial")}
    sat = (len(main) > 0 and pred["n_error"] == 0 and pred["n_not_equal_predicted"] == 0
           and pred["n_zero_while_checkpoint_usable"] == 0)
    return {"n_cells": len(main), **pred, "holdout_drop": drop, "predeclared_rule_satisfied": sat}


def _est_trunk_fan(gate, T: int, tail: int, S: int, n_sib: int) -> float:
    return gate.pf_s(T + tail, 0) + n_sib * (gate.pf_s(S + 1, T + tail) + 2 * gate.REQ_OVERHEAD_S) + gate.REQ_OVERHEAD_S


def _fresh_pair(env: Env, R, ids, child_r, *, n: int, arm: str, cell_id: str, key: str, n_probs: int, dflash: bool,
                cache_store: dict | None = None) -> tuple[dict, dict]:
    """Two cold runs (cache_prompt false) on slots 2 and 3 of `ids`; returns (floor, vs_fresh) as gate.compare dicts and
    appends them to R.eq. With cache_store, a prompt already controlled is not re-run."""
    gate = env.gate
    if cache_store is not None and key in cache_store:
        fa, fb = cache_store[key]
        floor = None
    else:
        fa, _ = env.send(R, "fresh_A", ids, 2, n_predict=n, arm=arm, cell_id=f"{cell_id}.freshA", fork=False, cache=False,
                         dflash=dflash, n_probs=n_probs)
        fb, _ = env.send(R, "fresh_B", ids, 3, n_predict=n, arm=arm, cell_id=f"{cell_id}.freshB", fork=False, cache=False,
                         dflash=dflash, n_probs=n_probs)
        floor = gate.compare(fa, fb, n)
        R.eq.setdefault("control_floor", []).append(floor)
        if cache_store is not None:
            cache_store[key] = (fa, fb)
    vs = gate.compare(child_r, fa, n)
    R.eq.setdefault("vs_fresh", []).append(vs)
    return floor, vs


def run_181_A(env: Env, groups: list[int], n_sib: int, *, with_ckpt: bool, expect: str | None, tag: str):
    gate, ctx = env.gate, env.ctx
    key = "A"
    arm = f"A_{'ckpt' if with_ckpt else 'tailonly'}{tag}"
    R = open_run(env, 181, arm, rule=RULES[181][key], arm_key=key, with_checkpoint_at=with_ckpt, expect=expect)
    for g in groups:
        grp = gate.G(ctx, g)
        T, trunk = grp["T"], grp["trunk"]
        env.erase_all()
        ctx.check_boundary()
        env.send(R, "source", trunk + grp["src_tail"], 0, n_predict=0, arm=arm, cell_id=f"g{g}.src", fork=False,
                 ckpt=[T] if with_ckpt else None, group=g)
        for i in range(n_sib):
            ctx.check_boundary()
            env.erase(1)
            exp = T if expect == "T" else (0 if expect == "zero" else None)
            env.send(R, "sibling", trunk + grp["suffixes"][i], 1, n_predict=1, arm=arm, cell_id=f"g{g}.s{i}", fork=True,
                     expected=exp, group=g, suffix=i)
        env.erase_all()
    close_counts(env, R)
    R.extra["rule_result"] = eval_181_A(R.extra["cells"])
    return R


def run_181_B(env: Env, g: int, tag: str):
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    n = a.n_predict
    arm = f"B_identical_resend_and_child_eq_trunk{tag}"
    R = open_run(env, 181, arm, rule=RULES[181]["B"], arm_key="B", fork_on=env.enabled)
    grp = gate.G(ctx, g)
    T, trunk = grp["T"], grp["trunk"]
    P0 = trunk + grp["suffixes"][0]
    cases = [("B1_resend_no_ckpt", P0, P0, None, "P0", True), ("B2_resend_ckpt_end", P0, P0, [-1], "P0", True),
             ("B3_child_eq_trunk_junction", P0, trunk, [T], "trunk", False),
             ("B4_child_eq_trunk_trunk_source", trunk, trunk, None, "trunk", True)]
    cache: dict = {}
    for name, src_ids, child_ids, ck, key, must_hit in cases:
        ctx.check_boundary()
        env.erase_all()
        env.send(R, "source", src_ids, 0, n_predict=0, arm=arm, cell_id=f"{name}.src", fork=False, ckpt=ck, case=name)
        r, cell = env.send(R, "child", child_ids, 1, n_predict=n, arm=arm, cell_id=name, fork=True, n_probs=a.n_probs,
                           expected=0 if not env.enabled else None, case=name, must_hit=must_hit and env.enabled)
        if env.enabled:  # cold controls only where a fork can happen; fork-off needs none
            _, vs = _fresh_pair(env, R, child_ids, r, n=n, arm=arm, cell_id=name, key=key, n_probs=a.n_probs, dflash=True,
                                cache_store=cache)
            cell["vs_fresh"] = vs
        env.erase_all()
    close_counts(env, R)
    R.extra["rule_result"] = eval_181_B(R.extra["cells"], R.eq.get("control_floor", []), fork_on=env.enabled)
    return R


def run_181_C(env: Env, g: int, lengths: list[int], dflash: bool):
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    n = a.n_predict
    arm = f"C_length_sweep_drafter_{'on' if dflash else 'off'}"
    R = open_run(env, 181, arm, dflash2=dflash, rule=RULES[181]["C"], arm_key="C", lengths=lengths, drafter_requested=dflash)
    grp = gate.G(ctx, g)
    trunk = grp["trunk"]
    for k, L in enumerate(lengths):
        ctx.check_boundary()
        env.erase_all()
        tk = trunk[:L]
        env.send(R, "source", tk + grp["src_tail"], 0, n_predict=0, arm=arm, cell_id=f"L{L}.src", fork=False, ckpt=[L],
                 dflash=dflash, length=L)
        r, cell = env.send(R, "child", tk + grp["suffixes"][0], 1, n_predict=n, arm=arm, cell_id=f"L{L}", fork=True,
                           n_probs=a.n_probs, dflash=dflash, expected=L, length=L)
        floor, vs = _fresh_pair(env, R, tk + grp["suffixes"][0], r, n=n, arm=arm, cell_id=f"L{L}", key=f"L{L}",
                                n_probs=a.n_probs, dflash=dflash)
        cell["vs_fresh"], cell["floor"] = vs, floor
        env.erase_all()
    close_counts(env, R)
    R.extra["rule_result"] = eval_181_C(R.extra["cells"], R.eq.get("control_floor", []))
    return R


def run_181_D(env: Env, g: int, ub: int, dflash2: bool):
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    arm = f"D_hit_count_prediction_{env.cfg_name}"
    R = open_run(env, 181, arm, dflash2=dflash2, rule=RULES[181]["D"], arm_key="D", ub=ub)
    grp = gate.G(ctx, g)
    T, trunk = grp["T"], grp["trunk"]
    src_ids = trunk + grp["src_tail_long"]
    pos = d_positions(T, len(src_ids), ub)
    R.extra["positions"] = pos
    env.erase_all()
    env.send(R, "source", src_ids, 0, n_predict=0, arm=arm, cell_id="src", fork=False, ckpt=[T])
    R.extra["source_checkpoint_set"] = [c["n_tokens"] for c in checkpoint_set(slot_by_id(ctx.client.slots(), 0))]
    take = 256

    def child_ids(lcp_: int) -> tuple[list[int], int]:
        got = divergent_suffix(grp["suffixes"], src_ids[lcp_], take)
        if got is None:
            raise gate.Abort(f"arm D: every suffix starts with the source's next token at lcp {lcp_}")
        return src_ids[:lcp_] + got[1], got[0]

    for cdef in d_cells(pos):
        ctx.check_boundary()
        ids, si = child_ids(cdef["lcp"])
        env.erase(1)
        env.send(R, "child", ids, 1, n_predict=1, arm=arm, cell_id=cdef["name"], fork=True, group=g, suffix=si,
                 position_name=cdef["position_name"], c=cdef["c"], delta=cdef["delta"], lcp_target=cdef["lcp"])
    h = holdout_lcp(pos)
    ctx.check_boundary()
    ids, si = child_ids(h)
    env.erase(1)
    r, cell = env.send(R, "child", ids, 1, n_predict=16, arm=arm, cell_id=f"holdout{h}", fork=True, n_probs=a.n_probs,
                       group=g, suffix=si, holdout=True, lcp_target=h)
    cold, _ = env.send(R, "cold", ids, 2, n_predict=16, arm=arm, cell_id=f"holdout{h}.cold", fork=False, cache=False,
                       n_probs=a.n_probs)
    cell["vs_cold"] = gate.compare(r, cold, 16)
    R.eq.setdefault("holdout_vs_cold", []).append(cell["vs_cold"])
    env.erase_all()
    close_counts(env, R)
    R.extra["rule_result"] = eval_181_D(R.extra["cells"])
    return R


def _ub_of(env_argv: list[str]) -> int:
    return ub_from_argv(env_argv)


def _c_group181(ctx) -> int:
    return int(ctx.args.kpf181_c_group) if ctx.args.kpf181_c_group is not None else 0


def build_181(ctx, gate) -> list:
    a = ctx.args
    Step, Srv = gate.Step, server_cls(gate)
    arms = [x.strip().upper() for x in a.kpf181_arms.split(",") if x.strip()]
    cfgs = get_configs(ctx, gate)
    prim_name, prim_argv = cfgs[0]
    groups = groups_arg(a.kpf181_groups, [0])
    n_sib = a.kpf181_sibs
    steps: list = []

    def S_(g):
        return gate.G(ctx, g)

    def servers(label, fork_min, name, argv):
        srv = Srv(ctx, f"kpf181-{label}-{name}-{gate.utc_tag()}", Path(a.fork_bin), fork_min, argv)
        return srv, Env(ctx, gate, srv, name)

    # -- server 1: fork on, the launched drafter shape: A, B, C, D
    srv, env = servers("on", a.fork_min, prim_name, prim_argv)
    steps += gate.start_steps(ctx, srv)
    has_drafter = config_name(prim_argv) != "nodraft"
    if "A" in arms:
        for with_ckpt in (True, False):
            est = sum(_est_trunk_fan(gate, S_(g)["T"], gate.SRC_TAIL, S_(g)["suffix_lens"][0], n_sib) for g in groups)
            steps.append(Step(f"181 A [{'checkpoint_at=[T]' if with_ckpt else 'tail only, no checkpoint_at'}] groups {groups}: "
                              f"source slot 0 trunk+tail, {n_sib} siblings on slot 1, fork on (rule A)", est,
                              lambda w=with_ckpt: run_181_A(env, groups, n_sib, with_ckpt=w, expect="T" if w else None, tag="")))
    if "B" in arms:
        g = groups[0]
        T, S0 = S_(g)["T"], S_(g)["suffix_lens"][0]
        est = 4 * (gate.pf_s(T + S0, 0) + gate.pf_s(S0, T) + gate.dec_s(a.n_predict, T, True)) + \
            2 * (2 * gate.pf_s(T + S0, 2 * T) + 2 * gate.dec_s(a.n_predict, 2 * T, True)) + 12 * gate.REQ_OVERHEAD_S
        steps.append(Step(f"181 B g{g}: identical resend (no ckpt / ckpt [-1]) and child==trunk (junction source / trunk "
                          f"source), cold A+B controls (rule B)", est, lambda: run_181_B(env, g, "")))
    if "C" in arms:
        gc = _c_group181(ctx)
        ub = _ub_of(prim_argv)
        lengths = sweep_lengths(S_(gc)["T"], ub, a.kpf181_lengths)
        for dflash in ([True, False] if has_drafter else [True]):
            est = sum(gate.pf_s(L + gate.SRC_TAIL, 0) + gate.pf_s(S_(gc)["suffix_lens"][0], L)
                      + 2 * gate.pf_s(L + S_(gc)["suffix_lens"][0], 2 * L)
                      + 3 * gate.dec_s(a.n_predict, L, dflash) + 8 * gate.REQ_OVERHEAD_S for L in lengths)
            steps.append(Step(f"181 C g{gc} drafter {'on' if dflash else 'off'}: {len(lengths)} trunk lengths "
                              f"{lengths[0] if lengths else '-'}..{lengths[-1] if lengths else '-'} (stride "
                              f"{(lengths[1] - lengths[0]) if len(lengths) > 1 else '-'}, ub {ub}), source+child+2 cold "
                              f"runs each (rule C)", est, lambda d=dflash, L=lengths: run_181_C(env, gc, L, d)))
    if "D" in arms:
        g = groups[0]
        T = S_(g)["T"]
        est = gate.pf_s(T + gate.SRC_TAIL_LONG, 0) + 10 * (gate.pf_s(256, T) + 2 * gate.REQ_OVERHEAD_S) + \
            gate.pf_s(T + 1000, 2 * T) + gate.dec_s(16, T, True) * 2
        steps.append(Step(f"181 D [{prim_name}] g{g}: source T={T}+long tail, 9 cells c-1/c/c+1 over "
                          f"{{junction, N-4-ub, N-4}} + 1 held-out length (rule D)", est,
                          lambda: run_181_D(env, g, _ub_of(prim_argv), has_drafter)))
    steps.append(gate.stop_step(srv))

    # -- server 2: the fork-off baseline on the same binary and argv: A and B must see zero forks
    if "A" in arms or "B" in arms:
        srv0, env0 = servers("off", 0, prim_name, prim_argv)
        steps += gate.start_steps(ctx, srv0)
        if "A" in arms:
            for with_ckpt in (True, False):
                est = sum(_est_trunk_fan(gate, S_(g)["T"], gate.SRC_TAIL, S_(g)["suffix_lens"][0], n_sib) for g in groups)
                steps.append(Step(f"181 A fork-OFF baseline [{'ckpt' if with_ckpt else 'tail only'}]: every sibling must "
                                  f"report n_fork_tokens 0", est,
                                  lambda w=with_ckpt: run_181_A(env0, groups, n_sib, with_ckpt=w, expect="zero", tag="_forkoff")))
        if "B" in arms:
            g = groups[0]
            T, S0 = S_(g)["T"], S_(g)["suffix_lens"][0]
            est = 4 * (gate.pf_s(T + S0, 0) + gate.pf_s(T + S0, T) + gate.dec_s(a.n_predict, T, True)) + 8 * gate.REQ_OVERHEAD_S
            steps.append(Step(f"181 B fork-OFF baseline g{g}: the four cases, every child n_fork_tokens 0", est,
                              lambda: run_181_B(env0, g, "_forkoff")))
        steps.append(gate.stop_step(srv0))

    # -- servers 3..: arm D once per further drafter configuration (no drafter, mtp, dflash2, ...)
    if "D" in arms:
        for name, argv in cfgs[1:]:
            srv_c, env_c = servers("on", a.fork_min, name, argv)
            g = groups[0]
            T = S_(g)["T"]
            est = gate.pf_s(T + gate.SRC_TAIL_LONG, 0) + 10 * (gate.pf_s(256, T) + 2 * gate.REQ_OVERHEAD_S) + \
                gate.pf_s(T + 1000, 2 * T) + gate.dec_s(16, T, True) * 2
            steps += gate.start_steps(ctx, srv_c)
            steps.append(Step(f"181 D [{name}] g{g}: same nine cells + held-out length on a {name} launch (rule D)", est,
                              lambda e=env_c, av=argv: run_181_D(e, g, _ub_of(av), config_name(av) != "nodraft")))
            steps.append(gate.stop_step(srv_c))
    return steps


def preflight_181(ctx, gate) -> tuple[list[str], list[str]]:
    a = ctx.args
    probs: list[str] = []
    notes: list[str] = []
    try:
        cfgs = get_configs(ctx, gate)
    except (ValueError, OSError) as exc:
        return [f"KPF-18.1 drafter configurations: {exc}"], notes
    arms = [x.strip().upper() for x in a.kpf181_arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ("A", "B", "C", "D")]
    if bad:
        probs.append(f"--kpf181-arms: unknown arms {bad}")
    if a.fork_min < 1:
        probs.append("KPF-18.1 needs --fork-min >= 1 (the fork-off baseline launch is added by the runner)")
    groups = groups_arg(a.kpf181_groups, [0]) + ([_c_group181(ctx)] if "C" in arms else [])
    for g in groups:
        if g not in ctx.ps.groups:
            probs.append(f"KPF-18.1 trunk group {g} not in the prompt set")
    if probs:
        return probs, notes
    prim = cfgs[0][1]
    ub = ub_from_argv(prim)
    if "C" in arms:
        T = ctx.ps.groups[_c_group181(ctx)]["T"]
        if a.kpf181_lengths < 16:
            probs.append("--kpf181-lengths must be >= 16 (KPF-18.1 arm C)")
        L = sweep_lengths(T, ub, max(1, a.kpf181_lengths))
        if not L:
            probs.append(f"arm C: trunk {T} cannot hold {a.kpf181_lengths} lengths across one -ub {ub} period")
        elif L[0] < a.fork_min:
            probs.append(f"arm C: shortest length {L[0]} < --fork-min {a.fork_min}")
        else:
            notes.append(f"181 C lengths {L[0]}..{L[-1]} stride {L[1] - L[0]} (ub {ub}) on group {_c_group181(ctx)}")
    if "D" in arms:
        g = groups_arg(a.kpf181_groups, [0])[0]
        T = ctx.ps.groups[g]["T"]
        if gate.SRC_TAIL_LONG <= ub + 8:
            probs.append(f"arm D: long source tail {gate.SRC_TAIL_LONG} <= ub {ub} + 8: the near-end checkpoints "
                         "would collide with the junction")
        if T < a.fork_min:
            probs.append(f"arm D: junction {T} < --fork-min {a.fork_min}")
    notes.append(f"181 drafter configurations: {[n for n, _ in cfgs]}; fork-off baseline launch added for arms A/B")
    return probs, notes


BUILDERS[181] = build_181
PREFLIGHTS[181] = preflight_181


# ================================================================ KPF-18.2  drafter parity, attribution, anti-vacuity
RULES[182] = {
    "common": ("Same controls as 18.1: manifest = prompt-set sha256 + binary sha256 + argv from the launch log; baseline = "
               "the fork-off arm on the same binary and champion tip e33229348; per-sibling raw n_fork_tokens, "
               "logits and errors retained; held-out lengths or prompts not used to set the rule; every sibling counted "
               "including failures, aborts and abstentions; counts, not a pass word. Protocol cell: slot 0 prefills "
               "trunk+tail with checkpoint_at=[T]; each sibling = trunk + its own suffix on slot 1, fork on, 16 tokens."),
    "parity": ("(1) The SAME prompts are run in every drafter column: drafter on (the launched shape: MTP on the CPU "
               "frontdoor, DFlash2 on the :8083 shape), the same server with speculative.n_max 0 on every request, a launch "
               "with no drafter, and any --kpf-config-argv configuration (e.g. a DFlash2 launch). SATISFIED iff every "
               "sibling's n_fork_tokens is identical in every column (the drafter state travels in the checkpoint); "
               "every difference is counted as a defect and listed."),
    "attribution": ("(2) For every sibling log tgt_kv_hit (client-derived: lcp with the source capped at n_task-1, or the "
                    "server's value), recurrent_ckpt_pos (the checkpoint the KPF-11 rule picks, or the server's) and "
                    "drafter_kv_hit (ONLY if the server exposes it in /slots[dst].fork; else NOT MEASURED), and the minimum "
                    "of the measured ones. SATISFIED iff n_fork_tokens == that minimum for every sibling, no sibling with a "
                    "drafter on has drafter_kv_hit < n_fork_tokens, and no component is zero while n_fork_tokens > 0. If the "
                    "server does not expose drafter_kv_hit the drafter clause is NOT EVALUATED and the tri-state result is "
                    "None (never True) unless another clause already failed."),
    "vacuity": ("(3) Per arm: the checkpoint set that existed (/slots), hit tokens, tokens lost to a missing checkpoint "
                "(max(0, min(lcp, n_task-1) - n_fork_tokens)), and the server's own fork counters (slot_forks_total, "
                "slot_fork_tokens_total) over the arm's window. An arm is VACUOUS (cannot be satisfied) if no sibling forked or "
                "no checkpoint ever existed; n_fork_tokens is read from the request's slot timings AND /slots, and the "
                "counters must equal the per-request sums."),
    "cold": ("(4) cache_prompt=false in the same process is the cold control: first a probe (a fork-eligible child with "
             "cache_prompt false must report n_fork_tokens 0 and cache_n 0, and the same child with cache_prompt true must "
             "fork: otherwise the arm aborts, the control is not expressible); then for each sibling two cold runs (A, B) "
             "on slots 2 and 3. SATISFIED iff every cold run has n_fork_tokens 0 and cache_n 0 and every sibling is within "
             "the cold-vs-cold floor (top-1 equal, max-abs prob and coarse KL <= max(floor, 1e-4))."),
    "policy": ("(5) Identical resend (trunk+suffix0, sent to slot 1) under the launched drafter: source pinned with "
               "checkpoint_at=[T] only vs source pinned with dense checkpoints every U tokens (<= 8). n_fork_tokens per "
               "policy and per component recorded. SATISFIED iff each policy's n_fork_tokens equals the KPF-11 prediction "
               "for its own checkpoint set; the difference dense - junction-only is reported as a count."),
}

PARITY_COLUMNS = ("drafter_on", "request_off")


def eval_182_attr(cells: list[dict], *, drafter_on: bool) -> dict:
    ch = _sib(cells, "child")
    rows, n_eq, n_below, n_abst, n_zero, lost = [], 0, 0, 0, 0, 0
    for c in ch:
        comp = c.get("components") or {}
        vals = {k: comp.get(k) for k in SERVER_FORK_FIELDS if comp.get(k) is not None}
        hmin = min(vals.values()) if vals else None
        ok = _ok(c) and hmin is not None and c["n_fork_tokens"] == hmin
        n_eq += 1 if ok else 0
        d = comp.get("drafter_kv_hit")
        if drafter_on and d is None:
            n_abst += 1
        if drafter_on and d is not None and _ok(c) and d < c["n_fork_tokens"]:
            n_below += 1
        if _ok(c) and c["n_fork_tokens"] > 0 and any(v == 0 for v in vals.values()):
            n_zero += 1
        lost += int(c.get("tokens_lost_to_missing_checkpoint") or 0)
        rows.append({"cell": c["cell"], "n_fork_tokens": c.get("n_fork_tokens"), "min": hmin, **{k: comp.get(k) for k in SERVER_FORK_FIELDS},
                     "checkpoint_set_before": c.get("checkpoint_set_before")})
    n_forked = sum(1 for c in ch if _ok(c) and c["n_fork_tokens"] > 0)
    n_ckpt = sum(1 for c in ch if c.get("checkpoint_set_before"))
    vacuous = (n_forked == 0) or (n_ckpt == 0)
    n_err = sum(1 for c in ch if not _ok(c))
    failed = bool(n_err or n_eq != len(ch) or n_below or n_zero or vacuous or not ch)
    evaluated = (not drafter_on) or n_abst == 0
    sat = False if failed else (True if evaluated else None)
    return {"n_siblings": len(ch), "n_error": n_err, "n_nfork_equals_min": n_eq, "n_drafter_below_nfork": n_below,
            "n_drafter_not_measured": n_abst, "drafter_clause_evaluated": evaluated, "n_zero_component_with_fork": n_zero,
            "n_forked": n_forked, "n_with_checkpoint_set": n_ckpt, "vacuous": vacuous,
            "tokens_lost_to_missing_checkpoint": lost, "siblings": rows, "predeclared_rule_satisfied": sat}


def eval_182_cold(cells: list[dict], floors: list[dict]) -> dict:
    cold = [c for c in cells if c.get("role") in ("fresh_A", "fresh_B")]
    bad = [c["cell"] for c in cold if not _ok(c) or c["n_fork_tokens"] != 0 or (c.get("cache_n") or 0) != 0]
    fs = floor_stats(floors)
    ch = [c for c in _sib(cells, "child") if c.get("vs_fresh") is not None]
    w = [within_floor(c["vs_fresh"], fs["max_maxabs"], fs["max_kld"]) for c in ch]
    n_in = sum(1 for x in w if x["within"])
    return {"n_cold_runs": len(cold), "n_cold_with_fork_or_cache": len(bad), "cold_with_fork_or_cache": bad,
            "n_siblings_compared": len(ch), "n_within_floor": n_in, "n_outside_floor": sum(1 for x in w if x["within"] is False),
            "floor": fs, "predeclared_rule_satisfied": (not bad and len(ch) > 0 and n_in == len(ch)) if cold else None}


def eval_182_parity(vectors: dict[str, list], ref: str | None = None) -> dict:
    """vectors: column -> [(sibling key, n_fork_tokens)]. Every column must report the same n_fork_tokens per sibling."""
    cols = list(vectors)
    ref = ref or (cols[0] if cols else None)
    if ref is None:
        return {"columns": [], "predeclared_rule_satisfied": None}
    rv = dict(vectors[ref])
    out = {"reference": ref, "columns": cols, "n_reference_siblings": len(rv),
           "n_reference_with_fork": sum(1 for v in rv.values() if v), "comparisons": {}}
    defects = 0
    for c in cols:
        if c == ref:
            continue
        ov = dict(vectors[c])
        keys = sorted(set(rv) | set(ov))
        mism = [{"sibling": k, ref: rv.get(k), c: ov.get(k)} for k in keys if rv.get(k) != ov.get(k)]
        defects += len(mism)
        out["comparisons"][c] = {"n_compared": len(keys), "n_equal": len(keys) - len(mism), "n_defects": len(mism),
                                 "defects": mism}
    out["n_defects_total"] = defects
    out["predeclared_rule_satisfied"] = (len(cols) > 1 and defects == 0 and out["n_reference_with_fork"] > 0
                                         and all(v is not None for v in rv.values())) if cols else None
    return out


def eval_182_policy(cells: list[dict]) -> dict:
    ch = _sib(cells, "child")
    by = {}
    for c in ch:
        by[c.get("policy")] = c
    pred = tally_prediction(ch)
    j, d = by.get("junction_only"), by.get("dense")
    delta = (d["n_fork_tokens"] - j["n_fork_tokens"]) if j and d and _ok(j) and _ok(d) else None
    return {**pred, "n_fork_junction_only": j.get("n_fork_tokens") if j else None,
            "n_fork_dense": d.get("n_fork_tokens") if d else None, "dense_minus_junction_only": delta,
            "components": {k: (v.get("components") or {}) for k, v in by.items()},
            "predeclared_rule_satisfied": (pred["n_error"] == 0 and pred["n_not_equal_predicted"] == 0 and len(by) == 2)}


def _drafter_on(env: Env, dflash: bool) -> bool:
    return bool(dflash and config_name(list(getattr(env.srv, "argv", None) or [])) != "nodraft")


def run_182_cold_probe(env: Env, g: int):
    """Is cache_prompt=false expressible? A fork-eligible child with it must not fork; the same child without must."""
    gate, ctx = env.gate, env.ctx
    arm = "cold_control_probe"
    R = open_run(env, 182, arm, rule=RULES[182]["cold"], arm_key="cold_probe")
    grp = gate.G(ctx, g)
    T, trunk = grp["T"], grp["trunk"]
    P = trunk + grp["suffixes"][0]
    env.erase_all()
    env.send(R, "source", trunk + grp["src_tail"], 0, n_predict=0, arm=arm, cell_id="src", fork=False, ckpt=[T])
    _, pos = env.send(R, "child", P, 1, n_predict=1, arm=arm, cell_id="positive_control_cache_prompt_true", fork=True, expected=T)
    _, neg = env.send(R, "cold", P, 2, n_predict=1, arm=arm, cell_id="cold_cache_prompt_false", fork=True, cache=False,
                      expected=0)
    ok_pos = pos["n_fork_tokens"] == T
    ok_neg = neg["n_fork_tokens"] == 0 and (neg.get("cache_n") or 0) == 0
    R.check("cold_control_expressible", ok_pos and ok_neg, {"positive_n_fork": pos["n_fork_tokens"], "expected_positive": T,
                                                             "cold_n_fork": neg["n_fork_tokens"], "cold_cache_n": neg.get("cache_n")})
    env.erase_all()
    close_counts(env, R)
    R.extra["rule_result"] = {"cold_control_expressible": ok_pos and ok_neg, "positive_n_fork": pos["n_fork_tokens"],
                              "cold_n_fork": neg["n_fork_tokens"], "cold_cache_n": neg.get("cache_n"),
                              "predeclared_rule_satisfied": ok_pos and ok_neg}
    if not (ok_pos and ok_neg):
        raise gate.Abort("cache_prompt=false is not an expressible cold control on this server "
                         f"(positive fork {pos['n_fork_tokens']} vs {T}, cold n_fork {neg['n_fork_tokens']}, cache_n {neg.get('cache_n')})")
    return R


def run_182_protocol(env: Env, groups: list[int], n_sib: int, column: str, dflash: bool, cold: bool, shared: dict):
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    arm = f"parity_{env.cfg_name}_{column}"
    drafter_on = _drafter_on(env, dflash)
    R = open_run(env, 182, arm, dflash2=dflash, rule=RULES[182]["parity"] + " || " + RULES[182]["attribution"],
                 arm_key="protocol", column=column, drafter_requested=dflash, drafter_on=drafter_on, cold_control=cold)
    n = 16
    vec = []
    for g in groups:
        grp = gate.G(ctx, g)
        T, trunk = grp["T"], grp["trunk"]
        ctx.check_boundary()
        env.erase_all()
        env.send(R, "source", trunk + grp["src_tail"], 0, n_predict=0, arm=arm, cell_id=f"g{g}.src", fork=False, ckpt=[T],
                 dflash=dflash, group=g)
        for i in range(n_sib):
            ctx.check_boundary()
            env.erase(1)
            P = trunk + grp["suffixes"][i]
            r, cell = env.send(R, "child", P, 1, n_predict=n, arm=arm, cell_id=f"g{g}.s{i}", fork=True, dflash=dflash,
                               n_probs=a.n_probs if cold else 0, group=g, suffix=i)
            vec.append((f"g{g}.s{i}", cell["n_fork_tokens"]))
            if cold:
                _, vs = _fresh_pair(env, R, P, r, n=n, arm=arm, cell_id=f"g{g}.s{i}", key=f"g{g}.s{i}", n_probs=a.n_probs,
                                    dflash=dflash)
                cell["vs_fresh"] = vs
                env.erase(2, 3)
    shared.setdefault("vec", {})[f"{env.cfg_name}:{column}"] = vec
    close_counts(env, R)
    R.extra["rule_result"] = eval_182_attr(R.extra["cells"], drafter_on=drafter_on)
    if cold:
        R.extra["cold_control"] = eval_182_cold(R.extra["cells"], R.eq.get("control_floor", []))
    R.extra["siblings_vector"] = vec
    return R


def run_182_policy(env: Env, g: int, dense_u: int):
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    arm = f"policy_control_{env.cfg_name}"
    R = open_run(env, 182, arm, rule=RULES[182]["policy"], arm_key="policy", dense_u=dense_u)
    grp = gate.G(ctx, g)
    T, trunk = grp["T"], grp["trunk"]
    P = trunk + grp["suffixes"][0]
    for policy, ck in (("junction_only", [T]), ("dense", dense_positions(len(P), dense_u))):
        ctx.check_boundary()
        env.erase_all()
        env.send(R, "source", P, 0, n_predict=0, arm=arm, cell_id=f"{policy}.src", fork=False, ckpt=ck, policy=policy,
                 checkpoint_at_requested=ck)
        env.send(R, "child", P, 1, n_predict=16, arm=arm, cell_id=policy, fork=True, policy=policy)
    env.erase_all()
    close_counts(env, R)
    R.extra["rule_result"] = eval_182_policy(R.extra["cells"])
    return R


def run_182_parity_summary(env: Env, shared: dict):
    R = open_run(env, 182, "parity_summary", rule=RULES[182]["parity"], arm_key="parity")
    R.extra["vectors"] = shared.get("vec", {})
    R.extra["note"] = ("compares the per-sibling n_fork_tokens recorded by the protocol runs of every drafter column; the "
                       "columns come from different launches (see each protocol record for its binary/argv)")
    R.extra["rule_result"] = eval_182_parity(shared.get("vec", {}))
    return R


def _est_protocol(gate, T: int, S: int, n_sib: int, cold: bool, n: int = 16) -> float:
    e = gate.pf_s(T + gate.SRC_TAIL, 0)
    for _ in range(n_sib):
        e += gate.pf_s(S, T) + gate.dec_s(n, T, True) + 3 * gate.REQ_OVERHEAD_S
        if cold:
            e += 2 * (gate.pf_s(T + S, 2 * T) + gate.dec_s(n, 2 * T, True)) + 2 * gate.REQ_OVERHEAD_S
    return e


def build_182(ctx, gate) -> list:
    a = ctx.args
    Step, Srv = gate.Step, server_cls(gate)
    parts = {x.strip() for x in a.kpf182_parts.split(",") if x.strip()}
    cfgs = get_configs(ctx, gate)
    groups = groups_arg(a.kpf182_groups, [0, 2])
    n_sib = a.kpf182_sibs
    shared: dict = {}
    blocks: list[tuple] = []        # (srv, body steps)

    def S_(g):
        return gate.G(ctx, g)

    def launch(name, argv):
        srv = Srv(ctx, f"kpf182-{name}-{gate.utc_tag()}", Path(a.fork_bin), a.fork_min, argv)
        return srv, Env(ctx, gate, srv, name)

    def proto_steps(env, column, dflash, cold):
        est = sum(_est_protocol(gate, S_(g)["T"], S_(g)["suffix_lens"][0], n_sib, cold) for g in groups)
        return [Step(f"182 protocol [{env.cfg_name} / {column}] groups {groups}: source + {n_sib} siblings per group, "
                     f"16 tok{', cold A+B per sibling' if cold else ''} (rules parity/attribution/vacuity"
                     f"{'/cold' if cold else ''})", est,
                     lambda e=env, c=column, d=dflash, k=cold: run_182_protocol(e, groups, n_sib, c, d, k, shared))]

    prim_name, prim_argv = cfgs[0]
    has_drafter = config_name(prim_argv) != "nodraft"
    srv, env = launch(prim_name, prim_argv)
    body: list = []
    if "4" in parts:
        g = groups[0]
        T = S_(g)["T"]
        body.append(Step(f"182 cold-control probe g{g}: cache_prompt=false child must not fork, cache_prompt=true child "
                         f"must (rule cold)", gate.pf_s(T + gate.SRC_TAIL, 0) + 4 * gate.REQ_OVERHEAD_S + 2 * gate.pf_s(512, T),
                         lambda: run_182_cold_probe(env, g)))
    if parts & {"1", "2", "3", "4"}:
        body += proto_steps(env, "drafter_on", True, "4" in parts)
    if "1" in parts and has_drafter:
        body += proto_steps(env, "request_off", False, False)
    if "5" in parts:
        g = groups[0]
        T, S0 = S_(g)["T"], S_(g)["suffix_lens"][0]
        body.append(Step(f"182 policy control [{prim_name}] g{g}: identical resend, checkpoint_at=[T] only vs dense every "
                         f"{a.kpf182_dense_u} (rule policy)", 2 * (gate.pf_s(T + S0, 0) + gate.pf_s(1, T + S0)
                         + gate.dec_s(16, T, True) + 4 * gate.REQ_OVERHEAD_S), lambda: run_182_policy(env, g, a.kpf182_dense_u)))
    blocks.append((srv, env, body))
    if "1" in parts:
        for name, argv in cfgs[1:]:
            srv_c, env_c = launch(name, argv)
            blocks.append((srv_c, env_c, proto_steps(env_c, "launch", True, False)))
    last_env = blocks[-1][1]
    if "1" in parts:
        blocks[-1][2].append(Step("182 parity summary: every column's n_fork_tokens per sibling, identical or counted as defects",
                                  0, lambda: run_182_parity_summary(last_env, shared)))
    steps: list = []
    for srv_b, _, body_b in blocks:
        steps += gate.start_steps(ctx, srv_b) + body_b + [gate.stop_step(srv_b)]
    return steps


def preflight_182(ctx, gate) -> tuple[list[str], list[str]]:
    a = ctx.args
    probs: list[str] = []
    notes: list[str] = []
    try:
        cfgs = get_configs(ctx, gate)
    except (ValueError, OSError) as exc:
        return [f"KPF-18.2 drafter configurations: {exc}"], notes
    parts = {x.strip() for x in a.kpf182_parts.split(",") if x.strip()}
    if not parts or not parts <= {"1", "2", "3", "4", "5"}:
        probs.append(f"--kpf182-parts {a.kpf182_parts!r}: use a subset of 1,2,3,4,5")
    if a.fork_min < 1:
        probs.append("KPF-18.2 needs --fork-min >= 1")
    groups = groups_arg(a.kpf182_groups, [0, 2])
    for g in groups:
        if g not in ctx.ps.groups:
            probs.append(f"KPF-18.2 trunk group {g} not in the prompt set")
    if not 1 <= a.kpf182_sibs <= len(GATE_SUFFIXES(gate)):
        probs.append(f"--kpf182-sibs {a.kpf182_sibs} outside 1..{len(GATE_SUFFIXES(gate))}")
    if "5" in parts and not dense_positions(1 << 30, max(1, a.kpf182_dense_u)):
        probs.append("--kpf182-dense-u yields no dense positions")
    if not probs and "5" in parts:
        n0 = ctx.ps.groups[groups[0]]["T"] + ctx.ps.groups[groups[0]]["suffix_lens"][0]
        if not dense_positions(n0, a.kpf182_dense_u):
            probs.append(f"--kpf182-dense-u {a.kpf182_dense_u} >= the policy prompt length {n0}")
    names = [n for n, _ in cfgs]
    notes.append(f"182 drafter columns: {names[0]} on/off + launches {names[1:]}; request-level off only if the launch has a drafter")
    notes.append("182 attribution: the server does not report drafter_kv_hit today (see RUNBOOK 'to enable'): the drafter "
                 "clause abstains, its result is None, never True")
    return probs, notes


def GATE_SUFFIXES(gate) -> list[int]:
    return list(gate.SUFFIXES)


BUILDERS[182] = build_182
PREFLIGHTS[182] = preflight_182


# ================================================================ KPF-18.3  chunk-split cost, end-ckpt, lane, >=50k warm-vs-cold
MIN_LONG_PROMPT = 50000
MIN_COMMON_POSITIONS = 16
MIN_COST_REPS = 3

RULES[183] = {
    "common": ("Same controls as 18.1/18.2 (manifest, fork-off baseline, raw cells, every request counted, counts not a "
               "pass word). Resource classes: part 1 = timing, INDICATIVE (may share the GPU with other work; the "
               "comparison is within one launch, alternating A B B A, against its own noise floor); parts 2, 3, 4 = "
               "correctness."),
    "cost": ("(1) Same trunk length T, cold slot, prefill only (n_predict 1, no fork): checkpoint_at=[T] on trunk+tail vs "
             "no checkpoint_at, alternating A B / B A for `reps` repeats per group (reps >= 3). Per group: delta = "
             "median(prompt_ms with) / median(prompt_ms without) - 1; noise floor = (max - min) / median of the WITHOUT arm. "
             "trigger_intra_batch_capture_followup = (delta > noise floor). The trigger is a flag for a measurement-only "
             "follow-up, not a verdict; prompt tokens/s and wall are recorded beside prompt_ms."),
    "end": ("(2) After a request with checkpoint_at=[-1] and a reply of `reply` tokens, read the pinned checkpoints of its "
            "slot: prompt_end (n_tokens == n_prompt), generation_end (n_prompt + reply, or one less) or other/absent. "
            "Then turn 2 = turn-1 prompt + the reply's token ids + a new suffix is sent to another slot (fork) and to "
            "the same slot (own reuse). SATISFIED iff the fork's n_fork_tokens >= n_prompt + reply; otherwise the gap "
            "(required - n_fork_tokens) is recorded, and the same for the own-slot reuse via cache_n."),
    "lane": ("(3) Each trunk request records its lane: completion_raw (/completion: near-end checkpoints only, "
             "pinned ones from checkpoint_at) or chat (/v1/chat/completions with message delimiters: user-start "
             "checkpoints as well). A raw trunk and a chat trunk are prefilled and their checkpoint positions are "
             "classified into near_end (within 4 + ub of the prompt end) and mid_prompt; a chat turn 2 on another slot "
             "records its n_fork_tokens. RECORDED ONLY (no verdict); an HTTP error on the chat probe is recorded as the "
             "probe's own result, not an abort."),
    "long": ("(4) On a trunk of >= 50000 tokens with a shared prefix: slot 0 prefills trunk+tail with checkpoint_at=[T]; the "
             "fork child (trunk+suffix0, slot 1) and two cold runs (cache_prompt false, slots 2 and 3) decode n_predict "
             "tokens at T=0 under the same (alone) batch composition. d(x, y) = mean |logprob_x - logprob_y| of the sampled "
             "token over the greedy prefix common to all three runs (>= 16 positions, else the arm abstains). "
             "SATISFIED iff the fork child forked at T and max(d(child, coldA), d(child, coldB)) <= max(d(coldA, coldB), "
             "1e-4). A needle-retrieval pass alone never closes the case; none is used."),
}


def classify_lane_checkpoints(ckpt_ns: list[int], n_prompt: int, ub: int) -> dict:
    near_end = sorted(n for n in ckpt_ns if n_prompt - (4 + ub) <= n <= n_prompt)
    mid = sorted(n for n in ckpt_ns if n < n_prompt - (4 + ub))
    return {"near_end": near_end, "mid_prompt": mid, "n_checkpoints": len(ckpt_ns), "n_prompt": n_prompt, "ub": ub}


def eval_183_cost(rows: list[dict]) -> dict:
    """rows: one per cold prefill: {group, arm: 'ckpt'|'noCkpt', rep, prompt_ms, prompt_per_second, wall_s}."""
    out = {}
    for g in sorted({r["group"] for r in rows}):
        a = [r["prompt_ms"] for r in rows if r["group"] == g and r["arm"] == "ckpt" and r.get("prompt_ms") is not None]
        b = [r["prompt_ms"] for r in rows if r["group"] == g and r["arm"] == "noCkpt" and r.get("prompt_ms") is not None]
        ma, mb = median_or_none(a), median_or_none(b)
        spread_b = ((max(b) - min(b)) / mb) if len(b) >= MIN_COST_REPS and mb else None
        spread_a = ((max(a) - min(a)) / ma) if len(a) >= MIN_COST_REPS and ma else None
        delta = (ma / mb - 1.0) if ma and mb else None
        trig = (delta > spread_b) if (delta is not None and spread_b is not None) else None
        out[str(g)] = {"n_with": len(a), "n_without": len(b), "median_prompt_ms_with": ma, "median_prompt_ms_without": mb,
                       "delta": delta, "noise_floor_without": spread_b, "spread_with": spread_a,
                       "median_prompt_per_second_with": median_or_none([r.get("prompt_per_second") for r in rows
                                                                          if r["group"] == g and r["arm"] == "ckpt"]),
                       "median_prompt_per_second_without": median_or_none([r.get("prompt_per_second") for r in rows
                                                                             if r["group"] == g and r["arm"] == "noCkpt"]),
                       "median_wall_s_with": median_or_none([r.get("wall_s") for r in rows if r["group"] == g and r["arm"] == "ckpt"]),
                       "median_wall_s_without": median_or_none([r.get("wall_s") for r in rows if r["group"] == g and r["arm"] == "noCkpt"]),
                       "trigger_intra_batch_capture_followup": trig}
    trig_all = [v["trigger_intra_batch_capture_followup"] for v in out.values()]
    return {"groups": out, "n_groups": len(out), "n_groups_triggering": sum(1 for t in trig_all if t),
            "n_groups_unmeasured": sum(1 for t in trig_all if t is None),
            "measurement_class": "timing_indicative"}


def eval_183_long(child: dict, cold_a: dict, cold_b: dict, *, n_fork: int | None, T: int, n_task: int) -> dict:
    pre = common_prefix_len([child, cold_a, cold_b])
    base = {"n_task": n_task, "T": T, "n_fork_tokens": n_fork, "forked_at_T": (n_fork == T) if n_fork is not None else None,
            "common_greedy_prefix": pre, "prompt_ge_50k": n_task >= MIN_LONG_PROMPT}
    if pre < MIN_COMMON_POSITIONS:
        return {**base, "abstain": True, "reason": f"common greedy prefix {pre} < {MIN_COMMON_POSITIONS}",
                "predeclared_rule_satisfied": None}
    fa, fb, ab = (mean_abs_dlogprob(child, cold_a, pre), mean_abs_dlogprob(child, cold_b, pre),
                  mean_abs_dlogprob(cold_a, cold_b, pre))
    if None in (fa["mean_abs_dlogprob"], fb["mean_abs_dlogprob"], ab["mean_abs_dlogprob"]):
        return {**base, "abstain": True, "reason": "a run carries no sampled-token logprobs",
                "predeclared_rule_satisfied": None}
    worst = max(fa["mean_abs_dlogprob"], fb["mean_abs_dlogprob"])
    floor = max(ab["mean_abs_dlogprob"], FLOOR_EPS)
    return {**base, "abstain": False, "d_fork_vs_coldA": fa, "d_fork_vs_coldB": fb, "d_coldA_vs_coldB": ab,
            "worst_fork_vs_cold": worst, "floor_used": floor, "within_floor": worst <= floor,
            "predeclared_rule_satisfied": bool(base["forked_at_T"] and base["prompt_ge_50k"] and worst <= floor)}


def eval_183_end(cls: dict, gap_fork: dict, gap_own: dict) -> dict:
    return {"end_checkpoint": cls, "turn2_fork": gap_fork, "turn2_own_slot": gap_own,
            "predeclared_rule_satisfied": gap_fork.get("covered")}


def run_183_cost(env: Env, groups: list[int], reps: int):
    gate, ctx = env.gate, env.ctx
    arm = "chunk_split_cost"
    R = open_run(env, 183, arm, rule=RULES[183]["cost"], arm_key="cost", reps=reps, measurement_class="timing_indicative",
                 grade_note="timing INDICATIVE: n=reps per cell inside one launch; the comparison is the with/without ratio "
                            "against the without-arm's own spread, not an absolute speed")
    rows = []
    for g in groups:
        grp = gate.G(ctx, g)
        T = grp["T"]
        ids = grp["trunk"] + grp["src_tail"]
        for rep in range(reps):
            order = [("ckpt", [T]), ("noCkpt", None)]
            if rep % 2:
                order.reverse()
            for name, ck in order:
                ctx.check_boundary()
                env.erase_all()
                r, cell = env.send(R, "cost_request", ids, 0, n_predict=1, arm=arm, cell_id=f"g{g}.r{rep}.{name}", fork=False,
                                   ckpt=ck, group=g, cost_arm=name, rep=rep)
                t = r.get("timings") or {}
                row = {"group": g, "arm": name, "rep": rep, "prompt_ms": t.get("prompt_ms"),
                       "prompt_per_second": t.get("prompt_per_second"), "wall_s": r.get("wall_s"), "prompt_n": t.get("prompt_n"),
                       "n_task": len(ids)}
                cell.update({k: v for k, v in row.items() if k not in cell})
                rows.append(row)
    env.erase_all()
    close_counts(env, R)
    R.extra["cost_rows"] = rows
    R.extra["rule_result"] = eval_183_cost(rows)
    return R


def run_183_end(env: Env, g: int, reply_n: int):
    gate, ctx = env.gate, env.ctx
    arm = "end_ckpt_position_and_multi_turn"
    R = open_run(env, 183, arm, rule=RULES[183]["end"], arm_key="end", reply_n=reply_n)
    grp = gate.G(ctx, g)
    P1 = grp["trunk"] + grp["suffixes"][0]
    env.erase_all()
    r1, c1 = env.send(R, "turn1", P1, 0, n_predict=reply_n, arm=arm, cell_id="turn1", fork=False, ckpt=[-1], group=g)
    reply = list(r1.get("tokens") or [])
    s0 = slot_by_id(ctx.client.slots(), 0)
    pinned = [c["n_tokens"] for c in checkpoint_set(s0) if c["pinned"]]
    cls = classify_end_ckpt(pinned, len(P1), len(reply))
    user2 = (grp["suffixes"][1] or [])[:256]
    P2 = P1 + reply + user2
    f, fc = env.send(R, "turn2_fork", P2, 1, n_predict=16, arm=arm, cell_id="turn2_fork", fork=True, group=g,
                     required=len(P1) + len(reply))
    gap_fork = multi_turn_gap(len(P1), len(reply), fc["n_fork_tokens"])
    o, oc = env.send(R, "turn2_own_slot", P2, 0, n_predict=16, arm=arm, cell_id="turn2_own", fork=False, group=g,
                     required=len(P1) + len(reply))
    gap_own = multi_turn_gap(len(P1), len(reply), oc.get("cache_n"))
    env.erase_all()
    close_counts(env, R)
    R.extra["rule_result"] = eval_183_end(cls, gap_fork, gap_own)
    R.extra["end_checkpoint_position"] = cls
    return R


def _chat_body(text_msgs: list[dict], slot: int, n: int = 16) -> dict:
    return {"model": "kpf18", "messages": text_msgs, "max_tokens": n, "temperature": 0.0, "id_slot": slot,
            "cache_prompt": True}


def run_183_lane(env: Env, g: int, probe_tokens: int):
    gate, ctx = env.gate, env.ctx
    arm = "request_lane"
    R = open_run(env, 183, arm, rule=RULES[183]["lane"], arm_key="lane", chat_probe_tokens=probe_tokens)
    grp = gate.G(ctx, g)
    ub = ub_from_argv(list(env.srv.argv))
    probes: list[dict] = []
    # -- raw /completion lane
    env.erase_all()
    P = grp["trunk"][:probe_tokens] + grp["suffixes"][0]
    env.send(R, "trunk_raw", P, 0, n_predict=1, arm=arm, cell_id="raw", fork=False, lane="completion_raw", group=g)
    s0 = slot_by_id(ctx.client.slots(), 0)
    ns = [c["n_tokens"] for c in checkpoint_set(s0)]
    probes.append({"lane": "completion_raw", "n_prompt": len(P), **classify_lane_checkpoints(ns, len(P), ub)})
    # -- chat lane (message delimiters present): a one-user-message prompt, then a second turn on another slot
    env.erase_all()
    chat: dict = {"lane": "chat"}
    try:
        st, body = ctx.client._req("POST", "/detokenize", {"tokens": grp["trunk"][:probe_tokens]}, timeout=600)
        if st != 200:
            raise RuntimeError(f"/detokenize HTTP {st}")
        text = body["content"] if isinstance(body, dict) else str(body)
        m1 = [{"role": "user", "content": text + "\n\nReply with one short sentence."}]
        st, res = ctx.client._req("POST", "/v1/chat/completions", _chat_body(m1, 0), timeout=3600)
        if st != 200 or not isinstance(res, dict):
            raise RuntimeError(f"chat turn 1 HTTP {st}: {str(res)[:200]}")
        n_prompt = int((res.get("usage") or {}).get("prompt_tokens") or 0)
        s0 = slot_by_id(ctx.client.slots(), 0)
        ns = [c["n_tokens"] for c in checkpoint_set(s0)]
        chat.update(n_prompt=n_prompt, **classify_lane_checkpoints(ns, n_prompt, ub),
                    pinned=[c["n_tokens"] for c in checkpoint_set(s0) if c["pinned"]])
        reply = (((res.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        m2 = m1 + [{"role": "assistant", "content": reply}, {"role": "user", "content": "And one more short sentence."}]
        st, res2 = ctx.client._req("POST", "/v1/chat/completions", _chat_body(m2, 1), timeout=3600)
        if st != 200 or not isinstance(res2, dict):
            raise RuntimeError(f"chat turn 2 HTTP {st}: {str(res2)[:200]}")
        t2 = res2.get("timings") or {}
        chat.update(turn2_n_fork_tokens=t2.get("n_fork_tokens"), turn2_fork_src_kind=t2.get("fork_src_kind"),
                    turn2_cache_n=t2.get("cache_n"), turn2_prompt_n=t2.get("prompt_n"))
        R.obs("trunk_chat", checkpoint_set=ns, n_fork_tokens=t2.get("n_fork_tokens"), lane="chat")
    except Exception as exc:  # noqa: BLE001 - the lane probe is an observation: its failure is its result
        chat["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    probes.append(chat)
    env.erase_all()
    close_counts(env, R)
    R.extra["lane_probes"] = probes
    R.extra["rule_result"] = {"lanes_recorded": sorted({p["lane"] for p in probes if not p.get("error")}),
                              "n_probe_errors": sum(1 for p in probes if p.get("error")), "probes": probes,
                              "predeclared_rule_satisfied": None}
    return R


def run_183_long(env: Env, g: int):
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    arm = "warm_vs_cold_long_prompt"
    n = a.n_predict
    R = open_run(env, 183, arm, rule=RULES[183]["long"], arm_key="long")
    grp = gate.G(ctx, g)
    T, trunk = grp["T"], grp["trunk"]
    P = trunk + grp["suffixes"][0]
    ctx.check_boundary()
    env.erase_all()
    env.send(R, "source", trunk + grp["src_tail"], 0, n_predict=0, arm=arm, cell_id="src", fork=False, ckpt=[T], group=g)
    r, cell = env.send(R, "child", P, 1, n_predict=n, arm=arm, cell_id="fork_child", fork=True, n_probs=a.n_probs, expected=T,
                       group=g)
    ca, _ = env.send(R, "fresh_A", P, 2, n_predict=n, arm=arm, cell_id="cold_A", fork=False, cache=False, n_probs=a.n_probs, group=g)
    cb, _ = env.send(R, "fresh_B", P, 3, n_predict=n, arm=arm, cell_id="cold_B", fork=False, cache=False, n_probs=a.n_probs, group=g)
    R.check("prompt_ge_50k", len(P) >= MIN_LONG_PROMPT, len(P))
    env.erase_all()
    close_counts(env, R)
    R.extra["rule_result"] = eval_183_long(r, ca, cb, n_fork=cell["n_fork_tokens"], T=T, n_task=len(P))
    return R


def _long_group183(ctx) -> int | None:
    a = ctx.args
    if a.kpf183_long_group is not None:
        return int(a.kpf183_long_group)
    for g in sorted(ctx.ps.groups):
        if ctx.ps.groups[g]["T"] >= MIN_LONG_PROMPT:
            return g
    return None


def _parts183(ctx, gate) -> list[str]:
    a = ctx.args
    if a.kpf183_parts:
        return [x.strip() for x in a.kpf183_parts.split(",") if x.strip()]
    return ["2", "3"] if gate.CPU_MODE else ["1", "4"]


def build_183(ctx, gate) -> list:
    a = ctx.args
    Step, Srv = gate.Step, server_cls(gate)
    parts = _parts183(ctx, gate)
    cfgs = get_configs(ctx, gate)
    name, argv = cfgs[0]
    srv = Srv(ctx, f"kpf183-{name}-{gate.utc_tag()}", Path(a.fork_bin), a.fork_min, argv)
    env = Env(ctx, gate, srv, name)
    steps = gate.start_steps(ctx, srv)

    def S_(g):
        return gate.G(ctx, g)

    if "1" in parts:
        groups = groups_arg(a.kpf183_groups, [0, 3])
        est = sum(2 * a.kpf183_reps * (gate.pf_s(S_(g)["T"] + gate.SRC_TAIL, 0) + gate.dec_s(1, S_(g)["T"], True)
                                       + 3 * gate.REQ_OVERHEAD_S) for g in groups)
        steps.append(Step(f"183.1 chunk-split cost groups {groups}: {a.kpf183_reps} x (checkpoint_at=[T] / none) alternating, "
                          f"cold prefill T+tail, timing INDICATIVE (rule cost)", est,
                          lambda: run_183_cost(env, groups, a.kpf183_reps)))
    if "2" in parts:
        g = 0
        T = S_(g)["T"]
        est = gate.pf_s(T + S_(g)["suffix_lens"][0], 0) + gate.dec_s(a.kpf183_reply, T, True) + \
            gate.pf_s(256 + a.kpf183_reply, T) + gate.pf_s(256 + a.kpf183_reply, T) + 10 * gate.REQ_OVERHEAD_S
        steps.append(Step(f"183.2 end-checkpoint position + multi-turn (turn-2 resends turn-1 prompt+reply) g{g}, reply "
                          f"{a.kpf183_reply} tok (rule end)", est, lambda: run_183_end(env, g, a.kpf183_reply)))
    if "3" in parts:
        g = 0
        n = a.kpf183_chat_probe_tokens
        est = 2 * gate.pf_s(n + 512, 0) + gate.pf_s(100, n) + 8 * gate.REQ_OVERHEAD_S
        steps.append(Step(f"183.3 request lane tag: raw /completion trunk vs /v1/chat/completions trunk ({n} tok), checkpoint "
                          f"classification, chat turn 2 on another slot (rule lane)", est,
                          lambda: run_183_lane(env, g, n)))
    if "4" in parts:
        g = _long_group183(ctx)
        if g is not None:
            T, S0 = S_(g)["T"], S_(g)["suffix_lens"][0]
            est = gate.pf_s(T + gate.SRC_TAIL, 0) + gate.pf_s(S0, T) + gate.dec_s(a.n_predict, T, True) + \
                2 * (gate.pf_s(T + S0, 2 * T) + gate.dec_s(a.n_predict, 2 * T, True)) + 6 * gate.REQ_OVERHEAD_S
            steps.append(Step(f"183.4 warm-vs-cold dlogprob on a {T + S0}-token prompt (g{g}): fork child vs 2 cold runs, "
                              f"{a.n_predict} tok at T=0 (rule long)", est, lambda: run_183_long(env, g)))
    steps.append(gate.stop_step(srv))
    return steps


def preflight_183(ctx, gate) -> tuple[list[str], list[str]]:
    a = ctx.args
    probs: list[str] = []
    notes: list[str] = []
    try:
        get_configs(ctx, gate)
    except (ValueError, OSError) as exc:
        return [f"KPF-18.3 drafter configurations: {exc}"], notes
    parts = _parts183(ctx, gate)
    if not parts or not set(parts) <= {"1", "2", "3", "4"}:
        probs.append(f"--kpf183-parts {parts!r}: use a subset of 1,2,3,4")
        return probs, notes
    if a.fork_min < 1:
        probs.append("KPF-18.3 needs --fork-min >= 1")
    notes.append(f"183 parts {parts} ({'--cpu default 2,3' if gate.CPU_MODE and not a.kpf183_parts else 'GPU default 1,4' if not a.kpf183_parts else 'explicit'}); "
                 "resource class: part 1 timing INDICATIVE, parts 2/3/4 correctness")
    if "1" in parts:
        groups = groups_arg(a.kpf183_groups, [0, 3])
        for g in groups:
            if g not in ctx.ps.groups:
                probs.append(f"KPF-18.3 part 1 trunk group {g} not in the prompt set")
        if a.kpf183_reps < MIN_COST_REPS:
            probs.append(f"--kpf183-reps {a.kpf183_reps} < {MIN_COST_REPS}: no noise floor can be measured")
    if "2" in parts:
        if a.kpf183_reply < 8:
            probs.append("--kpf183-reply must be >= 8")
        if 0 not in ctx.ps.groups:
            probs.append("KPF-18.3 part 2 needs trunk group 0")
    if "3" in parts and a.kpf183_chat_probe_tokens < 256:
        probs.append("--kpf183-chat-probe-tokens must be >= 256")
    if "4" in parts:
        g = _long_group183(ctx)
        if g is None or g not in ctx.ps.groups:
            probs.append(f"KPF-18.3 part 4 needs a trunk >= {MIN_LONG_PROMPT} tokens: none in this prompt set "
                         f"(spec {'CPU' if gate.CPU_MODE else 'GPU'}); run it in the GPU window")
        elif ctx.ps.groups[g]["T"] < MIN_LONG_PROMPT:
            probs.append(f"--kpf183-long-group {g}: T {ctx.ps.groups[g]['T']} < {MIN_LONG_PROMPT}")
        else:
            notes.append(f"183.4 long trunk group {g}: T {ctx.ps.groups[g]['T']}")
    return probs, notes


BUILDERS[183] = build_183
PREFLIGHTS[183] = preflight_183
