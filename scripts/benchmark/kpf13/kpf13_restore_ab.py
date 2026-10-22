#!/usr/bin/env python3
"""KPF-13.1 / KPF-13.1a driver: drafter state at a cross-slot prefix fork, restore vs recompute A/B.

Handoff: handoffs/active/kv-prefix-fork-and-paged-attention.md (RTG-58), rows KPF-13.1 and KPF-13.1a.
Written 2026-10-21 (code only, NOT yet run). Class: correctness (token identity and draft counts, not timing).

WHAT IT MEASURES
  For each (prompt group g, suffix length S) cell, on ONE llama-server (the P1 fork build, a plain-KV MTP drafter or
  DFlash2), greedy, seed 42, n_predict 64:
    W1, W2  the UNFORKED TWINS. slot_fork=false, cache_prompt=false on a fresh slot: the target AND the drafter are
            prefilled over the whole prompt. This is the baseline and, with W1 vs W2, the run-to-run spread.
    A       RESTORE. A child on a fresh slot forks from a source slot at the junction p = T (the end of the trunk):
            the drafter state comes through do_slot_fork exactly as KPF-13 ships it (attention-only share, then
            load_dft(PARTIAL_ONLY)).
    B       RECOMPUTE over the last N tokens. The source slot holds only trunk[:T-N], so the child forks at T-N and its
            own prefill recomputes the target AND drafter state over [T-N, T+S). (Server-side "re-prefill the drafter
            only" does not exist on any build; this is the request-level equivalent, and the only difference from the
            task text is that the target tail is recomputed too, which is bit-for-bit what an ordinary prefill does.)
  Order inside a cell is  W1 A1 B1 A2 B2 W2  (A/B/A/B, the twins bracketing). 13.1a adds two more server launches that
  run the A arm only, against the default launch's twins of the same cells:
    skip    LLAMA_KPF_MUTATE=dft_partial_noop: keep the attention-only share, skip load_dft (what 13.1a proposes;
            the hook is the test-only switch already on the P1 branch, recorded in the record).
    full    --slot-fork-dft-full: full drafter KV via seq_cp (KPF-19.1's explicit-restore variant).
  N is chosen on DEV prompts (group 0) from --n-list, then FIXED for the held-out groups (1..): "held-out prompts for N".

WHAT IT DOES NOT MEASURE (recorded in every record)
  * Raw first-draft TOKENS are not exposed by /completion. For MTP the proxy is (tokens, draft_n, draft_n_accepted);
    DFlash2 logs candidates at debug level, so --draft-debug adds -lv 4 and parses them.
  * Not carried by the restore (stated, not measured as absent): MTP pending_h (common_speculative_impl_draft_mtp has
    no get_state/set_state) and the DFlash2 per-seq selector RNG (seed pinned here).
  * A fork inside a prompt (S >= 1) rebuilds pending_h through the suffix prefill; the S=1 cell is the sharpest one.

PREDECLARED RULES: RULES below is written to <out>/rules.json and run_meta.json with its sha256 BEFORE the first
launch, and every verdict is a pure function of the stored cells (`analyze` recomputes it).

PROCESS DISCIPLINE (same as the KPF-18 runner): the driver starts llama-server as its own child and signals ONLY that
PID (SIGTERM, wait, SIGKILL, then `ps -p` must fail). It never signals by name. Port ownership is proved by matching
the listening socket inode against the child's /proc/<pid>/fd. A hard deadline (--deadline-utc, required to execute)
drains at a step boundary and a watchdog kills the own child past it. GPU runs take the mi210_0 device claim; CPU runs
must be inside a region-lock claim (run_kpf13.sh sets KPF13_REGION).

Exit codes: 0 ok, 2 refused (preflight), 3 abort, 4 error, 5 deadline/preempt drain, 75 claim held.
stdlib only (`requests` is not installed on this host). Harness pieces (Client, port checks, launch read-back, record
shape epyc.kpf.run.v1) are vendored minimally from scripts/benchmark/kpf18/kpf18_gate.py (branch f8/kpf18-arms).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1]  # .../scripts
SCHEMA = "epyc.kpf.run.v1"
TASK = "KPF-13.1"
LINKAGE = SCRIPTS / "utils" / "verify_ggml_linkage.sh"
CLAIM_MOD_DIR = SCRIPTS / "kernel_rnd"
DEFAULT_BIN = "/mnt/raid0/llm/worktrees/kv-prefix-fork-p1r2-20261020/build-cpu/bin/llama-server"
DEFAULT_PROMPT_SET = ("/mnt/raid0/llm/tmp/kv-prefix-fork/p1r-20261020/kpf18cpu/run-20261020T163128Z/prompt_set.jsonl")
DEFAULT_ARGV = HERE / "argv.cpu-35b-mtp.txt"
KFD_PROC = Path("/sys/class/kfd/kfd/proc")
DEFAULT_CARD = Path("/sys/class/drm/card2/device")
GIB = 1024 ** 3

PROD_ENV = {"HIP_PATH": "/opt/rocm", "ROCM_PATH": "/opt/rocm", "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores",
            "OMP_WAIT_POLICY": "active", "OMP_DYNAMIC": "false", "GGML_IQK": "1"}
LDP_TAIL = ["/opt/rocm/lib", "/usr/lib/llvm-20/lib", "/opt/AMD/aocc-compiler-5.0.0/lib"]
DROP_ENV_PREFIX = ("LLAMA_ARG_", "LLAMA_KPF_", "KPF_", "GGML_", "ROCBLAS_", "HIPBLASLT_", "AMD_LOG", "HIP_TRACE")
DROP_ENV_KEYS = ("LD_PRELOAD", "LLAMA_SERVER_SLOTS_DEBUG", "LD_LIBRARY_PATH")
FORBIDDEN_IN_ARGV = ["--port", "--slot-fork-min-tokens", "--metrics", "--slot-fork-dft-full", "--slot-save-path"]
FORBIDDEN_IN_ARGV_CPU = ["-t", "--threads", "-tb", "--threads-batch"]  # the driver sets them from --cpu-list on CPU

# Server launch configs. `mutate` is the LLAMA_KPF_MUTATE value (test-only hook on the P1 branch); `extra` are argv.
CONFIGS = {
    "default": {"label": "restore_default", "mutate": None, "extra": [], "arms": ("A", "B"),
                "knob": [b"--slot-fork-min-tokens"], "plain_only": False},
    "skip": {"label": "restore_skip_load_dft", "mutate": "dft_partial_noop", "extra": [], "arms": ("A",),
             "knob": [b"--slot-fork-min-tokens", b"LLAMA_KPF_MUTATE", b"dft_partial_noop"], "plain_only": True},
    "full": {"label": "restore_dft_full", "mutate": None, "extra": ["--slot-fork-dft-full"], "arms": ("A",),
             "knob": [b"--slot-fork-min-tokens", b"--slot-fork-dft-full"], "plain_only": True},
}

N_GEN = 64
SEED = 42
STOP_RESERVE_S = 150.0   # a step is not started unless est + this fits before the deadline
WATCHDOG_GRACE_S = 120.0  # past the deadline the watchdog kills the own child after this long

# ---------------------------------------------------------------- predeclared rules (hashed into the run metadata)
RULES = {
    "schema": "epyc.kpf13.rules.v1",
    "task": "KPF-13.1 (restore vs recompute A/B) and KPF-13.1a (skip load_dft for plain drafters)",
    "declared": "before the first launch; every verdict below is a pure function of the stored cells (analyze)",
    "sampling": {"temperature": 0.0, "seed": SEED, "ignore_eos": True, "n_generated": N_GEN, "n_probs": 2},
    "twin": ("unforked slot: slot_fork=false, cache_prompt=false (target and drafter fully prefilled). Two twins W1, W2 "
             "bracket each cell in the default launch; skip/full launches run W1 only and are compared against the "
             "default launch's twins of the same cell"),
    "order": "W1 A1 B1 A2 B2 W2 (A = restore, B = recompute over the last N tokens); skip/full: W1 A1 A2",
    "cell_invalid_if": [
        "an HTTP error or fewer than n_generated tokens (ignore_eos)",
        "a twin has draft_n == 0 or None (speculation inactive: the comparison would be vacuous)",
        "a twin reports cache_n != 0 or n_fork_tokens != 0 (it was not a fresh unforked run)",
        "a restore child's n_fork_tokens != T, or a recompute child's n_fork_tokens != T-N (the fork was not taken "
        "where designed)",
        "a skip/full launch's own twin W1 falls outside the default launch's twin range for that cell (launch variance)",
    ],
    "cell_pass": {
        "tokens": ("the first 64 generated token ids of X equal a twin's. If W1 and W2 themselves differ at position k, "
                   "X must agree with some twin for at least k positions (twin spread)"),
        "acceptance": ("draft_n and draft_n_accepted of X lie in [min, max] of the twins' values for the same request "
                       "(twins equal => exact equality; no extra tolerance is granted)"),
        "both_required": True,
    },
    "arm_verdict": ("over HELD-OUT (eval) cells only; dev cells are reported, not counted. PASS iff every attempted cell "
                    "passes (all attempts counted, invalid included); FAIL if any valid cell fails; else INCONCLUSIVE"),
    "n_selection": ("N is the smallest value in --n-list whose B arm PASSES every DEV cell; if none passes, the largest "
                    "N is used and `n_selected_by_default` is recorded true. N is never revised after the eval cells start"),
    "decisions": {
        "restore_faithful": "arm A verdict on the default launch == PASS",
        "trigger_port_30141": ("arm A verdict FAIL with at least one cell where draft_n_accepted of X is BELOW the twins' "
                               "minimum (an acceptance drop after fork restore). A token divergence without an "
                               "acceptance drop is reported as `investigate`, not as the trigger"),
        "kpf13_1a_skip_load_dft_plain": ("YES iff restore_faithful AND the skip launch's A verdict == PASS AND the skip "
                                         "launch's A runs equal the default launch's A runs cell by cell on tokens, "
                                         "draft_n and draft_n_accepted. NO if restore_faithful and skip is not equal. "
                                         "NOT_MEASURED if the skip launch did not run or the drafter is SWA/DFlash2"),
        "kpf19_1_choice": ("restore PASS: full drafter KV through the fork (share or --slot-fork-dft-full), no "
                           "re-prefill. restore FAIL and recompute PASS: drafter re-prefill over the last N. both FAIL: "
                           "neither; investigate"),
        "dft_full_equals_restore": "same test as skip, for the --slot-fork-dft-full launch",
    },
    "first_draft": ("identity of the first draft block is evaluated only when the server logged it for both runs "
                    "(--draft-debug, DFlash2); otherwise it is recorded `unobservable` and the rule above uses the "
                    "(tokens, draft_n, draft_n_accepted) proxy. It is never a pass condition"),
    "restore_does_not_carry": [
        "MTP pending_h: common_speculative_impl_draft_mtp has no get_state/set_state, so common_speculative_set_state is "
        "a no-op for it; a fork inside the prompt rebuilds it through the suffix prefill (the S=1 cell is the sharpest)",
        "DFlash2 per-seq selector RNG: re-seeded or continued after a restore, so draft sampling is not reproducible "
        "across a fork; the target verifies so output is unaffected; seed is pinned to 42 in every request",
    ],
    "info_only": ["TTFT and fork wall time are recorded and are NOT part of any verdict (correctness class, no quiet host)"],
}


def rules_sha256() -> str:
    return sha256_json(RULES)


# ---------------------------------------------------------------- small utils
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_file(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(path)


class Jsonl:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self._lock = path, threading.Lock()

    def add(self, rec: dict) -> None:
        with self._lock, open(self.path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


def flag_val(argv: list[str], names: tuple) -> str | None:
    for i, t in enumerate(argv[:-1]):
        if t in names:
            return argv[i + 1]
    return None


def set_flag(argv: list[str], names: tuple, value: str | None) -> list[str]:
    out, done, i = [], False, 0
    while i < len(argv):
        if argv[i] in names:
            if value is not None and not done:
                out += [names[0], value]
                done = True
            i += 2
            continue
        out.append(argv[i])
        i += 1
    if value is not None and not done:
        out += [names[0], value]
    return out


def parse_cpu_list(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def fmt_s(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def utc_epoch(s: str) -> float:
    """UTC string -> epoch seconds, independent of the host timezone."""
    import calendar
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ"):
        try:
            return float(calendar.timegm(time.strptime(s, fmt)))
        except ValueError:
            continue
    raise ValueError(f"deadline {s!r}: use YYYY-MM-DDTHH:MM:SSZ (UTC)")


# ---------------------------------------------------------------- process / port facts (read-only)
def listening_inodes(port: int) -> set[str]:
    """Socket inodes in LISTEN state on `port` (from /proc/net/tcp{,6}); no subprocess."""
    out: set[str] = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for ln in Path(f).read_text().splitlines()[1:]:
                parts = ln.split()
                if len(parts) > 9 and parts[3] == "0A" and int(parts[1].rsplit(":", 1)[1], 16) == port:
                    out.add(parts[9])
        except (OSError, ValueError, IndexError):
            continue
    return out


def port_listening(port: int) -> bool:
    return bool(listening_inodes(port))


def pid_owns_port(pid: int, port: int) -> bool:
    """True iff a LISTEN socket on `port` is held by /proc/<pid>/fd. A foreign server answering /health on the port
    (job 06, 2026-10-20) fails this."""
    inodes = listening_inodes(port)
    if not inodes:
        return False
    try:
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                tgt = os.readlink(fd)
            except OSError:
                continue
            if tgt.startswith("socket:[") and tgt[8:-1] in inodes:
                return True
    except OSError:
        return False
    return False


def pid_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0


def read_proc_env(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace")
    except OSError:
        return {}
    return dict(x.split("=", 1) for x in raw.split("\0") if "=" in x)


def read_cpus_allowed(pid: int) -> list[int] | None:
    try:
        for ln in Path(f"/proc/{pid}/status").read_text().splitlines():
            if ln.startswith("Cpus_allowed_list:"):
                return parse_cpu_list(ln.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


def bytes_contain_any_lib(binary: Path, needle: bytes) -> bool:
    """Knob-compiled-in check (the `strings | grep` rule): the binary or a CURRENT .so beside it contains `needle`."""
    # only the CURRENT libs: the unversioned *.so names (symlink chain) resolved, never the stale versioned files that a
    # long-lived build dir accumulates (build-cpu/bin holds .so.0.0.10351 .. .10362)
    cands = [binary] + sorted({Path(os.path.realpath(x)) for x in binary.parent.glob("*.so")})
    for p in cands:
        try:
            if needle in p.read_bytes():
                return True
        except OSError:
            continue
    return False


def kfd_pids() -> dict[int, int]:
    res: dict[int, int] = {}
    try:
        for d in KFD_PROC.iterdir():
            if d.name.isdigit():
                v = 0
                for f in d.glob("vram_*"):
                    try:
                        v += int(f.read_text())
                    except (OSError, ValueError):
                        pass
                res[int(d.name)] = v
    except OSError:
        pass
    return res


def card_vram_used(card: Path) -> int | None:
    try:
        return int((card / "mem_info_vram_used").read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- judging (pure; the unit tests exercise exactly these)
def top_probs(entry: dict) -> dict[int, float]:
    top = entry.get("top_logprobs") or entry.get("top_probs") or []
    out = {}
    for t in top:
        if "logprob" in t:
            out[int(t["id"])] = math.exp(float(t["logprob"]))
        elif "prob" in t:
            out[int(t["id"])] = float(t["prob"])
    return out


def top2_gaps(probs: list[dict]) -> list[float | None]:
    """Per generated position: p1 - p2 of the reported top-k (None if fewer than two entries). Used only to label a
    first divergence as a near-tie; never part of a verdict."""
    gaps = []
    for e in probs or []:
        v = sorted(top_probs(e).values(), reverse=True)
        gaps.append(round(v[0] - v[1], 6) if len(v) >= 2 else None)
    return gaps


def prefix_len(a: list[int], b: list[int], n: int) -> int:
    k = 0
    for x, y in zip(a[:n], b[:n]):
        if x != y:
            break
        k += 1
    return k


def judge_run(twins: list[dict], x: dict, expect_fork: int | None, n: int = N_GEN) -> dict:
    """One arm run against the twins of its cell. Returns {verdict: PASS|FAIL|INVALID, reasons, ...}."""
    reasons: list[str] = []
    invalid: list[str] = []
    if not twins:
        return {"verdict": "INVALID", "reasons": ["no twin"], "invalid": ["no twin"]}
    for i, w in enumerate(twins):
        if w.get("error") or len(w.get("tokens") or []) < n:
            invalid.append(f"twin{i + 1} short or failed ({len(w.get('tokens') or [])} tokens)")
        if not w.get("draft_n"):
            invalid.append(f"twin{i + 1} draft_n {w.get('draft_n')!r}: speculation inactive (vacuous)")
        if w.get("cache_n") not in (0, None) or w.get("n_fork_tokens") not in (0, None):
            invalid.append(f"twin{i + 1} not fresh (cache_n {w.get('cache_n')}, n_fork_tokens {w.get('n_fork_tokens')})")
    if x.get("error") or len(x.get("tokens") or []) < n:
        invalid.append(f"run short or failed ({len(x.get('tokens') or [])} tokens)")
    if expect_fork is not None and x.get("n_fork_tokens") != expect_fork:
        invalid.append(f"n_fork_tokens {x.get('n_fork_tokens')} != designed {expect_fork}")
    if invalid:
        return {"verdict": "INVALID", "reasons": invalid, "invalid": invalid}
    tw_tokens = [w["tokens"][:n] for w in twins]
    twin_agree = prefix_len(tw_tokens[0], tw_tokens[1], n) if len(tw_tokens) >= 2 else n
    best = max(prefix_len(x["tokens"], t, n) for t in tw_tokens)
    tokens_ok = best >= twin_agree
    if not tokens_ok:
        reasons.append(f"tokens: best twin prefix {best} < required {twin_agree}")
    dn = [w["draft_n"] for w in twins]
    da = [w.get("draft_n_accepted") or 0 for w in twins]
    x_dn, x_da = x.get("draft_n") or 0, x.get("draft_n_accepted") or 0
    acc_ok = min(dn) <= x_dn <= max(dn) and min(da) <= x_da <= max(da)
    acc_drop = x_da < min(da)
    if not acc_ok:
        reasons.append(f"acceptance: draft_n {x_dn} not in [{min(dn)},{max(dn)}] or accepted {x_da} not in "
                       f"[{min(da)},{max(da)}]")
    near_tie = None
    if not tokens_ok:
        gaps = x.get("gaps") or []
        near_tie = bool(best < len(gaps) and gaps[best] is not None and gaps[best] < 1e-3)
    return {"verdict": "PASS" if tokens_ok and acc_ok else "FAIL", "reasons": reasons, "tokens_ok": tokens_ok,
            "accept_ok": acc_ok, "acceptance_drop": acc_drop, "twin_agree": twin_agree, "prefix_best": best,
            "first_divergence_near_tie": near_tie}


def same_run(a: dict, b: dict, n: int = N_GEN) -> bool:
    return (a.get("tokens", [])[:n] == b.get("tokens", [])[:n] and a.get("draft_n") == b.get("draft_n")
            and a.get("draft_n_accepted") == b.get("draft_n_accepted"))


def cell_arm_verdict(twins: list[dict], runs: list[dict], expect_fork: int | None) -> dict:
    js = [judge_run(twins, r, expect_fork) for r in runs if r is not None]
    kinds = [j["verdict"] for j in js]
    v = "FAIL" if "FAIL" in kinds else ("INVALID" if "INVALID" in kinds or not kinds else "PASS")
    return {"verdict": v, "runs": js, "acceptance_drop": any(j.get("acceptance_drop") for j in js)}


def arm_verdict(cell_verdicts: list[dict]) -> dict:
    n = len(cell_verdicts)
    p = sum(1 for c in cell_verdicts if c["verdict"] == "PASS")
    f = sum(1 for c in cell_verdicts if c["verdict"] == "FAIL")
    inv = sum(1 for c in cell_verdicts if c["verdict"] == "INVALID")
    v = "PASS" if n > 0 and p == n else ("FAIL" if f > 0 else "INCONCLUSIVE")
    return {"verdict": v, "attempts": n, "pass": p, "fail": f, "invalid": inv,
            "acceptance_drop": any(c.get("acceptance_drop") for c in cell_verdicts)}


def select_n(dev_b_verdicts: dict[int, list[dict]], n_list: list[int]) -> dict:
    """dev_b_verdicts: N -> [cell verdict dicts of the B arm on DEV cells]."""
    for n in sorted(n_list):
        cv = dev_b_verdicts.get(n) or []
        if cv and all(c["verdict"] == "PASS" for c in cv):
            return {"N": n, "n_selected_by_default": False}
    return {"N": max(n_list), "n_selected_by_default": True}


def decide(summary: dict) -> dict:
    """summary: {"default": {"A": arm_verdict, "B": arm_verdict|None}, "skip": {"A":..,"equal_to_default_A": bool|None},
    "full": {...}, "plain_drafter": bool}. Pure function of arm verdicts (RULES['decisions'])."""
    d = summary.get("default", {})
    a, b = d.get("A"), d.get("B")
    restore = a["verdict"] if a else "NOT_MEASURED"
    out: dict = {"restore_faithful": restore == "PASS", "restore_verdict": restore,
                 "recompute_verdict": b["verdict"] if b else "NOT_MEASURED"}
    if restore == "FAIL":
        out["trigger_port_30141"] = bool(a.get("acceptance_drop"))
        out["investigate"] = not a.get("acceptance_drop")
    else:
        out["trigger_port_30141"] = False
        out["investigate"] = False
    if restore == "PASS":
        out["kpf19_1_choice"] = "full drafter KV through the fork (share or --slot-fork-dft-full), no re-prefill"
    elif restore == "FAIL" and b and b["verdict"] == "PASS":
        out["kpf19_1_choice"] = "drafter re-prefill over the last N tokens"
    elif restore == "FAIL":
        out["kpf19_1_choice"] = "neither passes: investigate"
    else:
        out["kpf19_1_choice"] = "NOT_MEASURED"
    for cfg, key in (("skip", "kpf13_1a_skip_load_dft_plain"), ("full", "dft_full_equals_restore")):
        s = summary.get(cfg)
        if not summary.get("plain_drafter", True) or s is None or s.get("A") is None:
            out[key] = "NOT_MEASURED"
        elif restore != "PASS":
            out[key] = "NOT_DECIDABLE (restore not faithful or not measured)"
        elif s["A"]["verdict"] == "FAIL" or s.get("equal_to_default_A") is False:
            out[key] = "NO"
        elif s["A"]["verdict"] == "PASS" and s.get("equal_to_default_A") is True:
            out[key] = "YES"
        else:
            out[key] = "INCONCLUSIVE"
    return out


# ---------------------------------------------------------------- analysis of stored cells (pure given the dicts)
def cell_id(g: int, S: int, N: int | None) -> str:
    return f"g{g}-S{S}" + (f"-N{N}" if N is not None else "")


def analyze_results(results: dict, dev_groups: set[int], n_list: list[int], plain_drafter: bool = True) -> dict:
    """results: {config: {cell_id: cell}} where cell = {"group","suffix_len","N","T","twins":{"W1","W2"},"runs":
    {"A1","B1","A2","B2"},"ok"}. A cell of the default launch may inherit its twins (`twins_from`). Non-default launches
    run held-out groups only and are judged against the DEFAULT launch's twins of the same (group, suffix_len)."""
    dflt = results.get("default", {})

    def twins_of(cell: dict, cells: dict) -> list[dict]:
        ref = cells.get(cell["twins_from"], {}) if cell.get("twins_from") else cell
        return [w for w in (ref.get("twins", {}).get("W1"), ref.get("twins", {}).get("W2")) if w]

    # N selection on DEV cells (B arm, default launch)
    dev_b: dict[int, list[dict]] = {}
    for c in dflt.values():
        if c.get("group") in dev_groups and c.get("N") is not None and c.get("ok", True):
            exp = (c["T"] - c["N"]) if c.get("T") else None
            dev_b.setdefault(c["N"], []).append(
                cell_arm_verdict(twins_of(c, dflt), [c["runs"].get("B1"), c["runs"].get("B2")], exp))
    sel = select_n(dev_b, n_list) if dev_b else {"N": None, "n_selected_by_default": None}

    def default_cell(g, S):
        cands = [c for c in dflt.values() if c.get("group") == g and c.get("suffix_len") == S and c.get("ok", True)]
        pick = [c for c in cands if c.get("N") == sel["N"]]
        return (pick or cands or [None])[0]

    out: dict = {"n_selection": sel, "configs": {}, "cells": {}}
    summary: dict = {"plain_drafter": plain_drafter}
    for cfg, cells in results.items():
        per_arm: dict[str, list[dict]] = {"A": [], "B": []}
        eq_default: list[bool] = []
        for cid, c in sorted(cells.items()):
            is_dev = c.get("group") in dev_groups
            expT = c.get("T")
            launch = None
            if cfg == "default":
                tw = twins_of(c, cells)
                dc = None
            else:
                dc = default_cell(c.get("group"), c.get("suffix_len"))
                tw = twins_of(dc, dflt) if dc else []
                w1 = c.get("twins", {}).get("W1")
                # launch-variance check: this launch's own twin must lie inside the default launch's twin range
                launch = (judge_run(tw, w1, None)["verdict"] if (tw and w1) else "INVALID")
            a_v = cell_arm_verdict(tw, [c["runs"].get("A1"), c["runs"].get("A2")], expT)
            if launch is not None and launch != "PASS":
                a_v = {**a_v, "verdict": "INVALID", "launch_check": launch}
            rec = {"group": c.get("group"), "suffix_len": c.get("suffix_len"), "N": c.get("N"), "dev": is_dev,
                   "A": a_v}
            if launch is not None:
                rec["launch_check"] = launch
            if "B1" in c.get("runs", {}):
                exp_b = (c["T"] - c["N"]) if c.get("T") and c.get("N") is not None else None
                rec["B"] = cell_arm_verdict(tw, [c["runs"].get("B1"), c["runs"].get("B2")], exp_b)
                rec["A_equals_B"] = all(same_run(c["runs"][f"A{i}"], c["runs"][f"B{i}"]) for i in (1, 2)
                                        if c["runs"].get(f"A{i}") and c["runs"].get(f"B{i}"))
            if dc is not None:
                rec["A_equals_default_A"] = bool(all(
                    same_run(c["runs"][f"A{i}"], dc["runs"][f"A{i}"]) for i in (1, 2)
                    if c["runs"].get(f"A{i}") and dc["runs"].get(f"A{i}"))) and all(
                    c["runs"].get(f"A{i}") and dc["runs"].get(f"A{i}") for i in (1, 2))
                if not is_dev:
                    eq_default.append(rec["A_equals_default_A"])
            out["cells"][f"{cfg}:{cid}"] = rec
            if not is_dev:
                per_arm["A"].append(a_v)
                if "B" in rec and (sel["N"] is None or c.get("N") == sel["N"]):
                    per_arm["B"].append(rec["B"])
        av = arm_verdict(per_arm["A"]) if per_arm["A"] else None
        bv = arm_verdict(per_arm["B"]) if per_arm["B"] else None
        out["configs"][cfg] = {"A": av, "B": bv}
        summary[cfg] = {"A": av, "B": bv,
                        "equal_to_default_A": None if cfg == "default" or not eq_default else all(eq_default)}
    out["decisions"] = decide(summary)
    out["rules_sha256"] = rules_sha256()
    return out


# ---------------------------------------------------------------- draft-debug parse (DFlash2 logs candidates at -lv 4)
_DRAFT_RE = re.compile(r"seq_id (\d+), draft candidate\s+0, pos\s+(\d+):\s+(\d+)")


def parse_first_draft(lines: list[str]) -> list[int] | None:
    """Candidate-0 token ids of the FIRST draft block in `lines` (stops when pos restarts at 0 after having started)."""
    out: list[int] = []
    for ln in lines:
        m = _DRAFT_RE.search(ln)
        if not m:
            continue
        pos, tok = int(m.group(2)), int(m.group(3))
        if out and pos == 0:
            break
        out.append(tok)
    return out or None


# ---------------------------------------------------------------- deadline / estimate / prompts
class Deadline:
    def __init__(self, epoch: float | None, clock=time.time) -> None:
        self.epoch, self.clock = epoch, clock

    def remaining(self) -> float:
        return float("inf") if self.epoch is None else self.epoch - self.clock()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def fits(self, est_s: float, reserve: float = STOP_RESERVE_S) -> bool:
        return self.remaining() >= est_s + reserve


class Rates:
    """Seconds model. Defaults are the MEASURED CPU-quarter rates of the 2026-10-20 KPF-18 CPU re-run on the same model
    (cpu-queue-f8 job 40, 35B-A3B MTP, -t 24, a quarter shared with timing blocks): prefill 16-36 tok/s (mid 30),
    decode with MTP 3.1-4.2 tok/s (3.5). The runner's own GPU-derived model ran ~3x low there, so these are used instead."""

    def __init__(self, prefill_tok_s=30.0, decode_tok_s=3.5, load_s=240.0, stop_s=30.0, req_s=1.0) -> None:
        self.pf, self.dec, self.load, self.stop, self.req = prefill_tok_s, decode_tok_s, load_s, stop_s, req_s

    def request(self, prefill: int, decode: int) -> float:
        return prefill / self.pf + decode / self.dec + self.req


def cpu_threads(cpu_list: str) -> int:
    return len(parse_cpu_list(cpu_list))


class PromptSet:
    """Reads a KPF-18 prompt set (kinds trunk / pair) and cuts trunks to `trunk_tokens` and suffixes to the requested
    lengths. A prefix of a trunk is a valid trunk; groups differ by corpus rotation so a group is held out from the
    others (no cross-group fork: the shared prefix is a few tokens, far below --slot-fork-min-tokens)."""

    def __init__(self, path: Path, trunk_tokens: int, suffix_lens: list[int]) -> None:
        self.path, self.trunk_tokens, self.suffix_lens = Path(path), trunk_tokens, list(suffix_lens)
        self.sha256 = sha256_file(self.path)
        side = self.path.with_name(self.path.name + ".sha256")
        self.sha_ok = None
        if side.exists():
            self.sha_ok = side.read_text().split()[0] == self.sha256
        self.trunks: dict[int, list[int]] = {}
        self.pairs: dict[int, dict[int, list[int]]] = {}
        for ln in self.path.read_text().splitlines():
            r = json.loads(ln)
            if r.get("kind") == "trunk":
                self.trunks[r["group"]] = r["ids"]
            elif r.get("kind") == "pair":
                self.pairs.setdefault(r["group"], {})[r["suffix_index"]] = r["suffix_ids"]

    def groups(self) -> list[int]:
        return sorted(g for g, t in self.trunks.items() if len(t) >= self.trunk_tokens)

    def trunk(self, g: int) -> list[int]:
        return self.trunks[g][: self.trunk_tokens]

    def suffix(self, g: int, S: int) -> list[int]:
        """Suffix of length S for group g. The suffix index is the position of S in --suffix-lens, so suffixes of
        different S differ at the first token (no accidental cross-hits)."""
        idx = self.suffix_lens.index(S)
        ids = self.pairs[g][idx]
        return ids[:S]

    def problems(self, groups: list[int]) -> list[str]:
        p = []
        if self.sha_ok is False:
            p.append(f"prompt set sha256 does not match its .sha256 file: {self.path}")
        for g in groups:
            if g not in self.trunks:
                p.append(f"group {g} not in the prompt set")
                continue
            if len(self.trunks[g]) < self.trunk_tokens:
                p.append(f"group {g} trunk has {len(self.trunks[g])} < {self.trunk_tokens} tokens")
            for i, S in enumerate(self.suffix_lens):
                have = len(self.pairs.get(g, {}).get(i, []))
                if have < S:
                    p.append(f"group {g} suffix {i} has {have} < {S} tokens")
        firsts = [self.pairs[groups[0]][i][0] for i in range(len(self.suffix_lens))
                  if groups and groups[0] in self.pairs and i in self.pairs[groups[0]]]
        if len(set(firsts)) != len(firsts):
            p.append(f"suffix first tokens not distinct: {firsts}")
        return p


# ---------------------------------------------------------------- HTTP client (vendored minimally from kpf18_gate.py)
class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 3600.0):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=max(1.0, timeout)) as r:
                raw, status = r.read(), r.status
        except urllib.error.HTTPError as e:
            return e.code, e.read()[:2000].decode(errors="replace")
        return status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode(errors="replace"))

    def health(self) -> bool:
        try:
            st, body = self._req("GET", "/health", timeout=5)
            return st == 200 and isinstance(body, dict) and body.get("status") == "ok"
        except Exception:  # noqa: BLE001
            return False

    def slots(self) -> list[dict]:
        st, body = self._req("GET", "/slots", timeout=60)
        if st != 200:
            raise RuntimeError(f"/slots HTTP {st}")
        return body

    def erase(self, slot: int) -> None:
        st, body = self._req("POST", f"/slots/{slot}?action=erase", {}, timeout=120)
        if st != 200:
            raise RuntimeError(f"erase slot {slot}: HTTP {st} {str(body)[:200]}")

    def complete(self, body: dict, timeout: float) -> dict:
        t0 = time.time()
        st, res = self._req("POST", "/completion", body, timeout=timeout)
        out = {"http": st, "wall_s": round(time.time() - t0, 3), "t_send": t0, "t_end": time.time()}
        if st != 200 or not isinstance(res, dict):
            out["error"] = str(res)[:800]
            return out
        out.update(tokens=res.get("tokens") or [], probs=res.get("completion_probabilities") or [],
                   timings=res.get("timings") or {})
        return out


def body_for(ids: list[int], n_predict: int, slot: int, *, fork: bool, cache: bool, ckpt: list[int] | None = None,
             n_probs: int = 2) -> dict:
    b = {"prompt": ids, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": cache, "return_tokens": True,
         "ignore_eos": True, "seed": SEED, "id_slot": slot}
    if n_probs and n_predict > 0:
        b["n_probs"] = n_probs
    if not fork:
        b["slot_fork"] = False
    if ckpt:
        b["checkpoint_at"] = ckpt
    return b


LOG_RES = {
    "kv_unified": re.compile(r"kv_unified\s*=\s*'?(true|false)'?"),
    "fork_enabled": re.compile(r"cross-slot prefix fork enabled: mode = (\w+), min tokens = (\d+), cache_idle_slots = (\d)"),
    "fork_disabled": re.compile(r"--slot-fork-min-tokens (\d+): cross-slot prefix fork disabled: (.*)"),
    "build": re.compile(r"build[ :=]+(\d+) \(([0-9a-f]+)\)"),
    "fork_line": re.compile(r"slot fork: src = (\d+) \((idle|busy)\), kind = (\w+), n_fork_tokens = (\d+), "
                            r"task\.n_tokens = (\d+), t = ([0-9.]+) ms"),
}


class Abort(RuntimeError):
    pass


# ---------------------------------------------------------------- run context + server lifecycle
class Ctx:
    def __init__(self, args, ps: "PromptSet | None", gate_argv: list[str], deadline: Deadline) -> None:
        self.args, self.ps, self.gate_argv, self.deadline = args, ps, gate_argv, deadline
        self.out = Path(args.out)
        self.client: Client | None = None
        self.own_children: list["Server"] = []
        self.host = socket.gethostname()
        self.t0 = time.time()
        self.rates = Rates(args.prefill_tok_s, args.decode_tok_s, args.load_s)
        self.prefix = ["taskset", "-c", args.cpu_list] if args.cpu_list else []
        self.bins_sha: dict[str, str | None] = {}

    def bin_sha(self, p: Path) -> str | None:
        k = str(p)
        if k not in self.bins_sha:
            self.bins_sha[k] = sha256_file(p)
        return self.bins_sha[k]

    def check_boundary(self) -> None:
        for s in self.own_children:
            if s.proc and s.proc.poll() is not None:
                raise Abort(f"server {s.label} died rc={s.proc.returncode}")
            if s.proc and not pid_owns_port(s.proc.pid, self.args.port):
                raise Abort(f"server {s.label}: pid {s.proc.pid} no longer owns port {self.args.port}")
        if self.deadline.expired():
            raise Abort("deadline reached")


class Server:
    def __init__(self, ctx: Ctx, cfg_name: str, binary: Path) -> None:
        self.ctx, self.cfg_name, self.cfg = ctx, cfg_name, CONFIGS[cfg_name]
        self.label = f"{cfg_name}-{utc_tag()}"
        self.binary, self.bin_dir = Path(binary), Path(binary).parent
        self.proc: subprocess.Popen | None = None
        self.log = ctx.out / "logs" / f"{self.label}.server.log"
        self.readback: dict = {}
        self.linkage: dict = {}
        self.stop_rec: dict = {}
        self.t_start = self.t_stop = None
        self.argv = self.build_argv()
        self.env, self.env_note = self.build_env()

    def build_argv(self) -> list[str]:
        a = list(self.ctx.gate_argv)
        a = set_flag(a, ("--slot-save-path",), str(self.ctx.out / "slot-save" / self.label))
        if self.ctx.args.device == "cpu":
            n = str(cpu_threads(self.ctx.args.cpu_list))
            a = set_flag(a, ("-t", "--threads"), n)
            a = set_flag(a, ("-tb", "--threads-batch"), n)
        a += ["--port", str(self.ctx.args.port), "--metrics", "--slot-fork-min-tokens", str(self.ctx.args.fork_min)]
        a += self.cfg["extra"]
        if self.ctx.args.draft_debug:
            a += ["-lv", "4"]
        return [str(self.binary)] + a

    def build_env(self) -> tuple[dict, dict]:
        drop = [k for k in os.environ if k.startswith(DROP_ENV_PREFIX) or k in DROP_ENV_KEYS]
        env = {k: v for k, v in os.environ.items() if k not in drop}
        env.update(PROD_ENV)
        env["LD_LIBRARY_PATH"] = ":".join([str(self.bin_dir)] + LDP_TAIL)
        if self.cfg["mutate"]:
            env["LLAMA_KPF_MUTATE"] = self.cfg["mutate"]
        return env, {"dropped_ambient": sorted(drop), "set": {**PROD_ENV, "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
                                                              **({"LLAMA_KPF_MUTATE": self.cfg["mutate"]}
                                                                 if self.cfg["mutate"] else {})}}

    def full_cmd(self) -> list[str]:
        return list(self.ctx.prefix) + self.argv

    def run_linkage(self) -> dict:
        r = subprocess.run(["bash", str(LINKAGE), str(self.binary), str(self.bin_dir)], capture_output=True, text=True,
                           env=dict(self.env), timeout=120)
        rep = self.ctx.out / "logs" / f"{self.label}.linkage.txt"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(r.stdout + r.stderr)
        ok = r.returncode == 0 and any(ln.startswith("PASS") for ln in r.stdout.splitlines())
        return {"rc": r.returncode, "pass": ok, "report": str(rep)}

    def start(self) -> None:
        ctx = self.ctx
        if port_listening(ctx.args.port):
            raise Abort(f"port {ctx.args.port} already listening (not ours: refusing to launch or to talk to it)")
        if ctx.args.device == "gpu":
            if port_listening(8083):
                raise Abort("production :8083 is listening: its owner must stop it via orchestrator_stack.py first")
            foreign = {p: v for p, v in kfd_pids().items() if v > 0}
            if foreign:
                raise Abort(f"foreign KFD processes hold VRAM: {foreign}")
            used = card_vram_used(Path(ctx.args.card))
            if used is not None and used > ctx.args.vram_free_max_gib * GIB:
                raise Abort(f"card VRAM used {used / GIB:.2f} GiB before launch (> {ctx.args.vram_free_max_gib})")
        for k in self.cfg["knob"]:
            if not bytes_contain_any_lib(self.binary, k):
                raise Abort(f"knob {k!r} is not compiled into {self.binary} (a null from an absent knob is not evidence)")
        self.linkage = self.run_linkage()
        if not self.linkage["pass"]:
            raise Abort(f"verify_ggml_linkage.sh FAILED for {self.binary} (see {self.linkage['report']})")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        (ctx.out / "slot-save" / self.label).mkdir(parents=True, exist_ok=True)
        print(f"[kpf13] launch {self.label}: {shlex.join(self.full_cmd())}", flush=True)
        self.t_start = time.time()
        self.proc = subprocess.Popen(self.full_cmd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env,
                                     start_new_session=True)
        ctx.own_children.append(self)
        threading.Thread(target=self._pump, daemon=True).start()
        client = Client(f"http://127.0.0.1:{ctx.args.port}")
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise Abort(f"server {self.label} exited rc={self.proc.returncode} before healthy (see {self.log})")
            if ctx.deadline.expired():
                raise Abort("deadline reached while the server was loading")
            if client.health() and pid_owns_port(self.proc.pid, ctx.args.port):
                break
            if time.time() - t0 > ctx.args.load_timeout_s:
                raise Abort(f"server {self.label} not healthy and port-owned after {ctx.args.load_timeout_s} s")
            time.sleep(2.0)
        self.readback = self.read_back()
        ctx.client = client
        if ctx.args.device == "gpu":
            time.sleep(2.5)
            if kfd_pids().get(self.proc.pid, 0) <= 0:
                raise Abort(f"own pid {self.proc.pid} holds no KFD VRAM after healthy (HIP residency not proven)")

    def _pump(self) -> None:
        with open(self.log, "ab") as fh:
            for line in iter(self.proc.stdout.readline, b""):
                fh.write(b"%.3f " % time.time() + line)
                fh.flush()

    def log_lines(self) -> list[str]:
        try:
            return self.log.read_text(errors="replace").splitlines()
        except OSError:
            return []

    def read_back(self) -> dict:
        txt = "\n".join(self.log_lines())
        pid = self.proc.pid
        rb: dict = {"pid": pid}
        try:
            rb["exe"] = os.path.realpath(f"/proc/{pid}/exe")
            cmd = [x for x in Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0") if x]
        except OSError as exc:
            raise Abort(f"cannot read /proc for own pid {pid}: {exc}")
        rb["exe_matches_binary"] = rb["exe"] == os.path.realpath(self.binary)
        rb["proc_cmdline_tail_matches_argv"] = cmd[-len(self.argv):] == self.argv
        rb["port_owned_by_pid"] = pid_owns_port(pid, self.ctx.args.port)
        env = read_proc_env(pid)
        rb["env_LLAMA_KPF_MUTATE"] = env.get("LLAMA_KPF_MUTATE")
        m = LOG_RES["kv_unified"].search(txt)
        rb["kv_mode_from_launch_log"] = m.group(0) if m else None
        m = LOG_RES["fork_enabled"].search(txt)
        rb["slot_fork_from_launch_log"] = m.group(0) if m else None
        rb["slot_fork_mode"] = m.group(1) if m else None
        m2 = LOG_RES["fork_disabled"].search(txt)
        rb["slot_fork_disabled_line"] = m2.group(0) if m2 else None
        m = LOG_RES["build"].search(txt)
        rb["build_from_launch_log"] = f"{m.group(1)} ({m.group(2)})" if m else None
        allowed = read_cpus_allowed(pid)
        rb["cpus_allowed"] = allowed
        probs = []
        if not rb["exe_matches_binary"]:
            probs.append(f"/proc/{pid}/exe = {rb['exe']}, not {self.binary}")
        if not rb["proc_cmdline_tail_matches_argv"]:
            probs.append("/proc cmdline differs from the launched argv")
        if not rb["port_owned_by_pid"]:
            probs.append(f"pid {pid} does not own the listening socket on {self.ctx.args.port}")
        if rb["kv_mode_from_launch_log"] is None or "true" not in rb["kv_mode_from_launch_log"]:
            probs.append(f"launch log kv_unified line {rb['kv_mode_from_launch_log']!r} (need true)")
        if not rb["slot_fork_from_launch_log"]:
            probs.append(f"no 'cross-slot prefix fork enabled' line ({rb['slot_fork_disabled_line']})")
        else:
            if LOG_RES["fork_enabled"].search(rb["slot_fork_from_launch_log"]).group(3) != "0":
                probs.append("cache_idle_slots != 0: idle sources would be cleared by the first launched task "
                             "(the protocol needs --no-cache-idle-slots)")
            if rb["slot_fork_mode"] != "checkpoint":
                probs.append(f"fork mode {rb['slot_fork_mode']!r}: the load_dft path under test exists only in "
                             "checkpoint mode")
        if self.cfg["mutate"] != rb["env_LLAMA_KPF_MUTATE"]:
            probs.append(f"LLAMA_KPF_MUTATE in /proc env is {rb['env_LLAMA_KPF_MUTATE']!r}, config wants {self.cfg['mutate']!r}")
        if self.ctx.args.cpu_list:
            want = parse_cpu_list(self.ctx.args.cpu_list)
            if allowed is None or sorted(allowed) != want:
                probs.append(f"Cpus_allowed_list {allowed} != claimed {want}")
        rb["problems"] = probs
        if probs:
            raise Abort(f"launch read-back failed for {self.label}: " + "; ".join(probs))
        return rb

    def fork_lines_between(self, t0: float, t1: float) -> list[dict]:
        out = []
        for ln in self.log_lines():
            ts, _, rest = ln.partition(" ")
            try:
                t = float(ts)
            except ValueError:
                continue
            if t0 - 0.5 <= t <= t1 + 0.5:
                m = LOG_RES["fork_line"].search(rest)
                if m:
                    out.append({"src": int(m.group(1)), "state": m.group(2), "kind": m.group(3),
                                "n_fork_tokens": int(m.group(4)), "task_tokens": int(m.group(5)), "t_ms": float(m.group(6))})
        return out

    def lines_between(self, t0: float, t1: float) -> list[str]:
        out = []
        for ln in self.log_lines():
            ts, _, rest = ln.partition(" ")
            try:
                t = float(ts)
            except ValueError:
                continue
            if t0 - 0.5 <= t <= t1 + 0.5:
                out.append(rest)
        return out

    def stop(self) -> None:
        if self.proc is None or self.stop_rec:
            return
        rec = {"pid": self.proc.pid, "sent": [], "at": now_iso()}
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            rec["sent"].append("TERM")
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.send_signal(signal.SIGKILL)
                rec["sent"].append("KILL")
                self.proc.wait(timeout=60)
        alive = pid_alive(self.proc.pid)
        rec.update(returncode=self.proc.returncode, verified_dead=not alive)
        self.t_stop = time.time()
        self.stop_rec = rec
        if self in self.ctx.own_children:
            self.ctx.own_children.remove(self)
        self.ctx.client = None
        print(f"[kpf13] stopped {self.label} pid {self.proc.pid}: {rec}", flush=True)
        if alive:
            raise RuntimeError(f"own child {self.proc.pid} still alive after TERM/KILL")


class Watchdog:
    """Past deadline + grace, kill the own children (captured PIDs only), verify dead, and exit 5."""

    def __init__(self, ctx: Ctx, grace_s: float = WATCHDOG_GRACE_S) -> None:
        self.ctx, self.grace = ctx, grace_s
        self._stop = threading.Event()

    def start(self) -> None:
        if self.ctx.deadline.epoch is not None:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.wait(5.0):
            if self.ctx.deadline.remaining() < -self.grace:
                print("[kpf13] WATCHDOG: past deadline + grace, killing own children", flush=True)
                for s in list(self.ctx.own_children):
                    try:
                        s.stop()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[kpf13] watchdog stop failed: {exc}", flush=True)
                os._exit(5)

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------- the cell protocol
class Step:
    def __init__(self, desc: str, est_s: float, fn=None) -> None:
        self.desc, self.est_s, self.fn = desc, est_s, fn


class Plan:
    """Builds the step list for one launch config. The same builder serves --dry-run (no fn is called) and execution."""

    def __init__(self, ctx: Ctx, cfg_name: str, results: dict, pending_only: bool) -> None:
        self.ctx, self.cfg_name, self.cfg = ctx, cfg_name, CONFIGS[cfg_name]
        self.results = results.setdefault(cfg_name, {})
        self.pending_only = pending_only
        self.srv: Server | None = None
        self.state: dict = {"N": None, "n_sel": None}
        self._rows: Jsonl | None = None

    @property
    def rows(self) -> Jsonl:
        """Created on first use so a --dry-run writes nothing."""
        if self._rows is None:
            self._rows = Jsonl(self.ctx.out / "rows" / f"{self.cfg_name}.jsonl")
        return self._rows

    # ---- request helper
    def req(self, role: str, ids: list[int], slot: int, *, fork: bool, cache: bool, n_predict: int, ckpt=None,
            meta: dict | None = None) -> dict:
        ctx = self.ctx
        ctx.check_boundary()
        body = body_for(ids, n_predict, slot, fork=fork, cache=cache, ckpt=ckpt)
        r = ctx.client.complete(body, timeout=min(7200.0, ctx.deadline.remaining() + WATCHDOG_GRACE_S - 10))
        t = r.get("timings") or {}
        forks = self.srv.fork_lines_between(r["t_send"], r["t_end"]) if self.srv else []
        run = {"role": role, "slot": slot, "http": r.get("http"), "error": r.get("error"),
               "tokens": r.get("tokens") or [], "gaps": top2_gaps(r.get("probs") or []),
               "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
               "n_fork_tokens": int(t.get("n_fork_tokens") or 0), "fork_src_kind": t.get("fork_src_kind"),
               "fork_src_slot": t.get("fork_src_slot"), "cache_n": int(t.get("cache_n") or 0),
               "prompt_n": int(t.get("prompt_n") or 0), "prompt_ms": t.get("prompt_ms"),
               "predicted_ms": t.get("predicted_ms"), "wall_s": r.get("wall_s"),
               "fork_op_ms": forks[-1]["t_ms"] if forks else None, "fork_log": forks[-1] if forks else None,
               "request_sha256": sha256_json({k: v for k, v in body.items() if k != "prompt"}) + ":" + sha256_json(ids)[:16]}
        if ctx.args.draft_debug and self.srv:
            run["first_draft"] = parse_first_draft(self.srv.lines_between(r["t_send"], r["t_end"]))
        self.rows.add({"t": round(time.time(), 3), **(meta or {}), **{k: v for k, v in run.items() if k != "gaps"}})
        if r.get("http") != 200 or r.get("error"):
            raise Abort(f"{role}: HTTP {r.get('http')} {r.get('error')}")
        return run

    def erase(self, slots) -> None:
        for s in slots:
            self.ctx.client.erase(s)

    # ---- estimate helpers (seconds)
    def est_req(self, prefill: int, decode: int = N_GEN) -> float:
        return self.ctx.rates.request(prefill, decode)

    # ---- plan
    def build(self) -> list[Step]:
        ctx, a = self.ctx, self.ctx.args
        steps: list[Step] = []
        T = a.trunk_tokens
        has_b = "B" in self.cfg["arms"]
        # default launch: dev groups sweep --n-list, eval groups use N* (resolved at run time; estimated at n_list[0]).
        # other launches (13.1a): held-out groups only, A arm only, no N.
        groups = (a.dev_groups + a.eval_groups) if has_b else list(a.eval_groups)
        todo: list[tuple[int, int, int | None]] = []
        for g in groups:
            ns = (a.n_list if g in a.dev_groups else [a.n_list[0]]) if has_b else [None]
            for N in ns:
                for S in a.suffix_lens:
                    done = [c for c in self.results.values() if c.get("ok") and c.get("group") == g
                            and c.get("suffix_len") == S and (g in a.dev_groups and has_b and c.get("N") == N
                                                              or not (g in a.dev_groups and has_b))]
                    if self.pending_only and done:
                        continue
                    todo.append((g, S, N))
        if not todo:
            return []
        steps.append(Step(f"[{self.cfg_name}] check knob, linkage, launch, healthy, port ownership, read-back "
                          f"({self.cfg['label']}: mutate={self.cfg['mutate']}, extra={self.cfg['extra']})",
                          ctx.rates.load + 5, self._start))
        by_gn: dict[tuple[int, int | None], list[int]] = {}
        for g, S, N in todo:
            by_gn.setdefault((g, N), []).append(S)
        twin_seen: set[tuple[int, int]] = set()
        for (g, N), Ss in by_gn.items():
            n_est = N or 0
            setup_est = (self.est_req(T - n_est, 0) + (self.est_req(n_est, 0) if has_b else 0) + 4 * ctx.rates.req)
            steps.append(Step(f"[{self.cfg_name}] g{g} N={N}: sources (slot 0 = trunk" + (f"[:T-N], slot 3 = trunk via "
                              f"one-hop fork" if has_b else "") + f"), T={T}", setup_est,
                              lambda g=g, N=N: self._setup(g, N)))
            for S in Ss:
                n_tw = 0 if (has_b and (g, S) in twin_seen) else (2 if has_b else 1)
                twin_seen.add((g, S))
                est = n_tw * self.est_req(T + S) + 2 * self.est_req(S)
                if has_b:
                    est += 2 * self.est_req(n_est + S) + 2 * self.est_req(n_est, 0)
                est += 8 * ctx.rates.req
                steps.append(Step(f"[{self.cfg_name}] cell g{g} S={S} N={N}: "
                                  + ("W1 A1 B1 A2 B2 W2" if has_b else "W1 A1 A2"), est,
                                  lambda g=g, S=S, N=N: self._cell(g, S, N)))
        steps.append(Step(f"[{self.cfg_name}] SIGTERM own pid, verify dead, write record", ctx.rates.stop, self._stop))
        return steps

    def _start(self) -> None:
        self.srv = Server(self.ctx, self.cfg_name, Path(self.ctx.args.fork_bin))
        self.srv.start()

    def _stop(self) -> None:
        try:
            if self.srv:
                self.srv.stop()
        finally:
            self.finish_record()

    def _resolve_n(self, g: int, N: int | None) -> int | None:
        """Eval groups use the N selected on DEV; dev groups use the N they are sweeping."""
        a = self.ctx.args
        if N is None or g in a.dev_groups:
            return N
        if self.state["N"] is None:
            dflt = self.ctx.results.get("default", {})
            devb: dict[int, list[dict]] = {}
            for cid, c in dflt.items():
                if c.get("group") in a.dev_groups and c.get("N") is not None and c.get("ok"):
                    tw = [w for w in (c["twins"].get("W1"), c["twins"].get("W2")) if w] if "twins" in c else []
                    if not tw and c.get("twins_from"):
                        ref = dflt.get(c["twins_from"], {})
                        tw = [w for w in (ref.get("twins", {}).get("W1"), ref.get("twins", {}).get("W2")) if w]
                    devb.setdefault(c["N"], []).append(
                        cell_arm_verdict(tw, [c["runs"].get("B1"), c["runs"].get("B2")], c["T"] - c["N"]))
            sel = select_n(devb, a.n_list)
            self.state["N"], self.state["n_sel"] = sel["N"], sel
            print(f"[kpf13] N selected on DEV: {sel}", flush=True)
        return self.state["N"]

    def _setup(self, g: int, N: int | None) -> None:
        ctx, a = self.ctx, self.ctx.args
        N = self._resolve_n(g, N)
        ctx.check_boundary()
        trunk = ctx.ps.trunk(g)
        T = len(trunk)
        self.erase((0, 1, 2, 3))
        meta = {"phase": "setup", "group": g, "N": N, "config": self.cfg_name}
        if "B" in self.cfg["arms"]:
            self.req("source0_trunk_minus_N", trunk[: T - N], 0, fork=False, cache=True, n_predict=0, ckpt=[-1], meta=meta)
            if a.source_build == "fork":
                s3 = self.req("source3_one_hop", trunk, 3, fork=True, cache=True, n_predict=0, ckpt=[-1], meta=meta)
                if s3["n_fork_tokens"] != T - N:
                    raise Abort(f"source 3 did not fork at T-N: n_fork_tokens {s3['n_fork_tokens']} != {T - N}")
            else:
                self.req("source3_fresh", trunk, 3, fork=False, cache=False, n_predict=0, ckpt=[-1], meta=meta)
        else:
            self.req("source0_trunk", trunk, 0, fork=False, cache=True, n_predict=0, ckpt=[-1], meta=meta)

    def _regen_source3(self, g: int, N: int) -> None:
        trunk = self.ctx.ps.trunk(g)
        meta = {"phase": "regen", "group": g, "N": N, "config": self.cfg_name}
        if self.ctx.args.source_build == "fork":
            s3 = self.req("source3_one_hop", trunk, 3, fork=True, cache=True, n_predict=0, ckpt=[-1], meta=meta)
            if s3["n_fork_tokens"] != len(trunk) - N:
                raise Abort(f"source 3 regen did not fork at T-N: {s3['n_fork_tokens']} != {len(trunk) - N}")
        else:
            self.req("source3_fresh", trunk, 3, fork=False, cache=False, n_predict=0, ckpt=[-1], meta=meta)

    def _cell(self, g: int, S: int, N: int | None) -> None:
        ctx = self.ctx
        has_b = "B" in self.cfg["arms"]
        N = self._resolve_n(g, N)
        cid = cell_id(g, S, N)
        trunk = ctx.ps.trunk(g)
        T = len(trunk)
        P = trunk + ctx.ps.suffix(g, S)
        meta = {"phase": "cell", "cell": cid, "group": g, "S": S, "N": N, "config": self.cfg_name}
        cell = {"cell": cid, "group": g, "suffix_len": S, "N": N, "T": T, "config": self.cfg_name,
                "prompt_sha256": sha256_json(P), "runs": {}, "twins": {}, "ok": False, "start_utc": now_iso()}
        # twins are N-independent: reuse the first N's twins for the same (g, S) in the default launch
        reuse = None
        if has_b:
            for oid, oc in self.results.items():
                if oc.get("group") == g and oc.get("suffix_len") == S and oc.get("twins", {}).get("W1") and oc.get("ok"):
                    reuse = oid
                    break
        n = N_GEN

        def twin(tag: str) -> dict:
            r = self.req(f"twin_{tag}", P, 2, fork=False, cache=False, n_predict=n, meta={**meta, "arm": tag})
            self.erase((2,))
            return r

        def child(tag: str) -> dict:
            r = self.req(f"arm_{tag}", P, 1, fork=True, cache=True, n_predict=n, meta={**meta, "arm": tag})
            self.erase((1,))
            return r

        if reuse:
            cell["twins_from"] = reuse
        else:
            cell["twins"]["W1"] = twin("W1")
        if has_b:
            cell["runs"]["A1"] = child("A1")                      # restore: source 3 (trunk, T) is the longest candidate
            self.erase((3,))                                      # B must fork from slot 0 (trunk[:T-N]) only
            cell["runs"]["B1"] = child("B1")
            self._regen_source3(g, N)
            cell["runs"]["A2"] = child("A2")
            self.erase((3,))
            cell["runs"]["B2"] = child("B2")
            self._regen_source3(g, N)
            if not reuse:
                cell["twins"]["W2"] = twin("W2")
        else:
            cell["runs"]["A1"] = child("A1")
            cell["runs"]["A2"] = child("A2")
        # fork designed positions (validated again by the judge; recorded here for the rows)
        cell["expect"] = {"A": T, "B": (T - N) if has_b else None}
        cell["end_utc"] = now_iso()
        cell["ok"] = True
        self.results[cid] = cell
        self.ctx.results.setdefault(self.cfg_name, {})[cid] = cell
        write_json(self.ctx.out / "cells" / self.cfg_name / f"{cid}.json", cell)
        print(f"[kpf13]   cell {self.cfg_name}:{cid} stored: "
              + json.dumps({k: [r["n_fork_tokens"], r["draft_n"], r["draft_n_accepted"]]
                            for k, r in {**cell["twins"], **cell["runs"]}.items()}), flush=True)

    # ---- record (epyc.kpf.run.v1 shape, KPF-13.1 extension)
    def finish_record(self) -> None:
        ctx, srv = self.ctx, self.srv
        a = ctx.args
        cells = dict(self.results)
        ana = analyze_results({self.cfg_name: cells, **({"default": ctx.results.get("default", {})}
                                                        if self.cfg_name != "default" else {})},
                              set(a.dev_groups), a.n_list)
        sums = [r for c in cells.values() for r in c.get("runs", {}).values()]
        rec = {
            "schema": SCHEMA, "kpf_task": TASK, "run_id": f"kpf13-{self.cfg_name}-{utc_tag()}",
            "arm": self.cfg["label"], "binary": str(a.fork_bin), "binary_sha256": ctx.bin_sha(Path(a.fork_bin)),
            "build_from_launch_log": srv.readback.get("build_from_launch_log") if srv else None,
            "argv": srv.argv if srv else None, "launch_prefix": list(ctx.prefix),
            "argv_readback": {k: srv.readback.get(k) for k in ("exe_matches_binary", "proc_cmdline_tail_matches_argv",
                                                              "port_owned_by_pid", "env_LLAMA_KPF_MUTATE",
                                                              "cpus_allowed", "slot_fork_mode")} if srv else None,
            "env": srv.env_note if srv else None,
            "kv_mode_from_launch_log": srv.readback.get("kv_mode_from_launch_log") if srv else None,
            "slot_fork_from_launch_log": srv.readback.get("slot_fork_from_launch_log") if srv else None,
            "prompt_set": str(ctx.ps.path), "prompt_set_sha256": ctx.ps.sha256,
            "drafter": a.drafter, "dflash2": a.drafter == "dflash2",
            "spec_type_from_argv": flag_val(ctx.gate_argv, ("--spec-type",)),
            "n_predict": N_GEN, "seed": SEED,
            "equivalence": {"counts_basis": "per cell: tokens identical to a twin, draft_n and draft_n_accepted inside "
                                            "the twins' range; see rules_sha256",
                            "configs": ana["configs"].get(self.cfg_name), "n_selection": ana["n_selection"],
                            "decisions": ana["decisions"]},
            "ttft_ms": [r["prompt_ms"] for r in sums if r and r.get("prompt_ms") is not None],
            "prompt_tokens_processed": sum(r.get("prompt_n", 0) for r in sums if r),
            "cache_n": sum(r.get("cache_n", 0) for r in sums if r),
            "n_fork_tokens": sum(r.get("n_fork_tokens", 0) for r in sums if r),
            "fork_op_ms": [r["fork_op_ms"] for r in sums if r and r.get("fork_op_ms") is not None],
            "window_id": a.window_id, "gpu_window_id": a.gpu_window_id,
            "start_utc": ctx.start_utc, "end_utc": now_iso(), "host": ctx.host,
            "rules_sha256": rules_sha256(), "restore_does_not_carry": RULES["restore_does_not_carry"],
            "first_draft_observable": a.draft_debug,
            "residency": {"linkage": srv.linkage if srv else None, "stop": srv.stop_rec if srv else None,
                          "device": a.device, "cpu_list": a.cpu_list},
            "cells_dir": str(ctx.out / "cells" / self.cfg_name), "rows": str(self.rows.path),
            "rows_sha256": sha256_file(self.rows.path), "n_cells": len(cells),
            "instrument_class": "serving", "category": "CANDIDATE",
            "grading_note": "n per cell as listed, no codified protocol cited: an OBSERVATION; projects under "
                            "VB-KVU-PF and grades Judged/Located through claim_tuple.grade(). Counts, not a pass word.",
        }
        rec["record_sha256"] = sha256_json(rec)
        write_json(ctx.out / "records" / f"{rec['run_id']}.json", rec)
        Jsonl(ctx.out / "belief_measurements.jsonl").add(rec)
        print(f"[kpf13] record {rec['run_id']} ({self.cfg_name}): " + json.dumps(ana["configs"].get(self.cfg_name)),
              flush=True)


# ---------------------------------------------------------------- preflight, plan printing, main
def load_argv_file(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text().splitlines()
    comments = [ln for ln in lines if ln.lstrip().startswith("#")]
    body = " ".join(ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#"))
    return shlex.split(body), comments


def argv_problems(argv: list[str], drafter: str, device: str = "cpu") -> list[str]:
    p = []
    if argv and argv[0].endswith("llama-server"):
        p.append("argv file must not contain the binary")
    for f in FORBIDDEN_IN_ARGV + (FORBIDDEN_IN_ARGV_CPU if device == "cpu" else []):
        if f in argv:
            p.append(f"argv file must not contain {f} (the driver sets it)")
    st = flag_val(argv, ("--spec-type",))
    if drafter == "plain-mtp" and st != "draft-mtp":
        p.append(f"--spec-type {st!r}: the plain-MTP profile needs draft-mtp")
    if drafter == "dflash2":
        md = flag_val(argv, ("-md", "--model-draft")) or ""
        if st != "draft-dflash" or "dflash2" not in md.lower():
            p.append(f"--spec-type {st!r} / -md {md!r}: the DFlash2 profile needs draft-dflash and a DFlash2 drafter")
    if "--kv-unified" not in argv:
        p.append("missing --kv-unified (the gate shape; the fork's cell accounting assumes it)")
    np_ = flag_val(argv, ("-np", "--parallel"))
    if np_ != "4":
        p.append(f"-np {np_!r}: the protocol uses slots 0..3 (need 4)")
    return p


def preflight(args, ps: PromptSet | None, gate_argv: list[str]) -> tuple[list[str], list[str]]:
    probs, notes = [], []
    if not (1024 <= args.port <= 65535) or 8000 <= args.port <= 8999:
        probs.append(f"--port {args.port} refused: never 8000-8999 (production/standard ports) or < 1024")
    if port_listening(args.port):
        probs.append(f"--port {args.port} is already listening")
    if args.device == "gpu" and port_listening(8083):
        probs.append("production :8083 is listening (its owner must stop it first)")
    probs += argv_problems(gate_argv, args.drafter, args.device)
    b = Path(args.fork_bin)
    if not b.exists():
        probs.append(f"--fork-bin {b} does not exist")
    else:
        notes.append(f"fork-bin {b} (mtime {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(b.stat().st_mtime))})")
        for cfg in args.configs:
            for k in CONFIGS[cfg]["knob"]:
                if not bytes_contain_any_lib(b, k):
                    probs.append(f"config {cfg}: knob {k!r} not compiled into {b} (needs the "
                                 "experimental/kpf-19-1-17-3-20261021 or later build)")
    if not LINKAGE.exists():
        probs.append(f"linkage script missing: {LINKAGE}")
    if args.cpu_list:
        try:
            cpus = parse_cpu_list(args.cpu_list)
            if not cpus:
                raise ValueError("empty")
            notes.append(f"cpu-list {args.cpu_list} = {len(cpus)} cpus; -t/-tb set to {len(cpus)}")
        except ValueError:
            probs.append(f"--cpu-list {args.cpu_list!r} unparsable")
    if args.drafter == "dflash2":
        bad = [c for c in args.configs if CONFIGS[c]["plain_only"]]
        if bad:
            probs.append(f"configs {bad} are plain-drafter only (an SWA drafter always needs load_dft): drop them")
    if ps is None:
        probs.append("no prompt set")
    else:
        probs += ps.problems(sorted(set(args.dev_groups + args.eval_groups)))
        for g in args.dev_groups + args.eval_groups:
            if g in ps.trunks and len(ps.trunks[g]) < args.trunk_tokens:
                probs.append(f"group {g} shorter than --trunk-tokens")
    if min(args.n_list) < 1 or max(args.n_list) >= args.trunk_tokens:
        probs.append("--n-list must satisfy 1 <= N < --trunk-tokens")
    if args.trunk_tokens < args.fork_min + 1:
        probs.append(f"--trunk-tokens {args.trunk_tokens} must exceed --fork-min {args.fork_min}")
    if args.trunk_tokens - max(args.n_list) < args.fork_min:
        probs.append(f"T-N = {args.trunk_tokens - max(args.n_list)} < --fork-min {args.fork_min}: the recompute fork would not be taken")
    if set(args.dev_groups) & set(args.eval_groups):
        probs.append("dev and eval groups overlap (N must be chosen on held-out prompts)")
    if not args.eval_groups:
        probs.append("no eval groups")
    return probs, notes


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fork-bin", default=DEFAULT_BIN)
    ap.add_argument("--argv-file", default=str(DEFAULT_ARGV))
    ap.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET)
    ap.add_argument("--drafter", choices=["plain-mtp", "dflash2"], default="plain-mtp")
    ap.add_argument("--configs", default="default,skip,full",
                    help="comma list of launches, in order: default (A restore + B recompute), skip, full (13.1a)")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--cpu-list", default=None, help="the CPU list this run holds (required: a CPU quarter, or the GPU "
                    "host threads); also the taskset list. The wrapper's region-lock claim must name the same list")
    ap.add_argument("--port", type=int, default=18973)
    ap.add_argument("--fork-min", type=int, default=1024)
    ap.add_argument("--trunk-tokens", type=int, default=2048)
    ap.add_argument("--suffix-lens", default="1,128", help="comma list; the prompt set must carry that many suffixes")
    ap.add_argument("--n-list", default="64,512", help="candidate N (tokens recomputed before the fork point)")
    ap.add_argument("--dev-groups", default="0")
    ap.add_argument("--eval-groups", default="1,2,3")
    ap.add_argument("--source-build", choices=["fork", "fresh"], default="fork",
                    help="how the restore source (slot 3) is built: a one-hop fork + N-token tail (cheap, the default) "
                         "or a fresh full prefill (N-independent, T tokens per rebuild)")
    ap.add_argument("--reduced", action="store_true",
                    help="audit-sized preset: acceptance over 64 tokens on 8 held-out cells (groups 0-3 x suffix 1,128), "
                         "ONE launch (default: A restore, B recompute, twins), N fixed at 64, no dev sweep, no skip/full. "
                         "Run once per drafter (--drafter plain-mtp, then --drafter dflash2 with its own --argv-file)")
    ap.add_argument("--draft-debug", action="store_true", help="add -lv 4 and parse DFlash2 first-draft candidates")
    ap.add_argument("--deadline-utc", default=None, help="HARD deadline, YYYY-MM-DDTHH:MM:SSZ (required to execute)")
    ap.add_argument("--deadline-s", type=float, default=None, help="relative deadline (tests, short runs)")
    ap.add_argument("--window-id", default=None)
    ap.add_argument("--gpu-window-id", default=os.environ.get("KPF13_GPU_WINDOW_ID"))
    ap.add_argument("--dry-run", action="store_true", help="validate, print the plan and the estimate; start nothing")
    ap.add_argument("--resume", action="store_true", help="skip cells already stored under <out>/cells")
    ap.add_argument("--load-timeout-s", type=float, default=900.0)
    ap.add_argument("--card", default=str(DEFAULT_CARD))
    ap.add_argument("--vram-free-max-gib", type=float, default=2.0)
    ap.add_argument("--no-region-lock", default=None, metavar="REASON", help="CPU: skip the region-lock proof (recorded)")
    ap.add_argument("--no-device-claim", default=None, metavar="REASON", help="GPU: skip the mi210_0 claim (recorded)")
    ap.add_argument("--prefill-tok-s", type=float, default=30.0, help="estimate only")
    ap.add_argument("--decode-tok-s", type=float, default=3.5, help="estimate only")
    ap.add_argument("--load-s", type=float, default=240.0, help="estimate only: server load")
    args = ap.parse_args(argv)
    args.configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    args.suffix_lens = [int(x) for x in args.suffix_lens.split(",") if x.strip()]
    args.n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    args.dev_groups = [int(x) for x in args.dev_groups.split(",") if x.strip() != ""]
    args.eval_groups = [int(x) for x in args.eval_groups.split(",") if x.strip() != ""]
    if args.reduced:
        # 8 held-out prompts: every group is held out because N is a single pre-declared value (nothing is tuned on them)
        args.configs, args.dev_groups, args.eval_groups, args.n_list = ["default"], [], [0, 1, 2, 3], [64]
        args.suffix_lens = [1, 128]
    return args


def build_deadline(args) -> Deadline:
    if args.deadline_utc:
        return Deadline(utc_epoch(args.deadline_utc))
    if args.deadline_s is not None:
        return Deadline(time.time() + args.deadline_s)
    return Deadline(None)


def cmd_analyze(args) -> int:
    out = Path(args.out)
    results: dict = {}
    for cfg_dir in sorted((out / "cells").glob("*")):
        for f in sorted(cfg_dir.glob("*.json")):
            results.setdefault(cfg_dir.name, {})[f.stem] = json.loads(f.read_text())
    if not results:
        print(f"no cells under {out / 'cells'}")
        return 2
    ana = analyze_results(results, set(args.dev_groups), args.n_list, args.drafter == "plain-mtp")
    write_json(out / "verdict.json", ana)
    print(json.dumps({"configs": ana["configs"], "n_selection": ana["n_selection"], "decisions": ana["decisions"],
                      "rules_sha256": ana["rules_sha256"]}, indent=1))
    return 0


def print_plan(ctx: Ctx, plans: list[tuple[str, list[Step]]], probs: list[str], notes: list[str]) -> float:
    total = 0.0
    print("=" * 100)
    print(f"KPF-13.1 restore-vs-recompute A/B   device={ctx.args.device}  drafter={ctx.args.drafter}  "
          f"rules_sha256={rules_sha256()[:16]}")
    print(f"rates (estimate only): prefill {ctx.rates.pf} tok/s, decode {ctx.rates.dec} tok/s, load {ctx.rates.load:.0f}s")
    for n in notes:
        print(f"  note: {n}")
    for cfg, steps in plans:
        s = sum(x.est_s for x in steps)
        total += s
        print(f"--- launch {cfg}: {len(steps)} steps, est {fmt_s(s)}")
        for st in steps:
            print(f"   [{fmt_s(st.est_s):>7}] {st.desc}")
    q = ctx.args.cpu_list and cpu_threads(ctx.args.cpu_list)
    print(f"TOTAL est {fmt_s(total)} wall ({total / 3600:.2f} h); hard-deadline reserve {STOP_RESERVE_S:.0f}s per step; "
          f"estimate is {'~3x-pessimistic-calibrated' if ctx.args.device == 'cpu' else 'GPU rates unverified'}"
          + (f"; one claimed region of {q} cpus => ~{total / 3600:.2f} Q-h" if q else ""))
    if probs:
        print("PREFLIGHT PROBLEMS (dry-run exits 0 regardless; execute refuses):")
        for p in probs:
            print(f"  - {p}")
    else:
        print("PREFLIGHT: inputs PASS (live-only checks run at execute: linkage, read-back, port ownership, claim)")
    return total


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    args.window_id = args.window_id or f"kpf13-{utc_tag()}"
    gate_argv, comments = load_argv_file(Path(args.argv_file))
    ps = None
    try:
        ps = PromptSet(Path(args.prompt_set), args.trunk_tokens, args.suffix_lens)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"prompt set unreadable: {exc}")
    deadline = build_deadline(args)
    probs, notes = preflight(args, ps, gate_argv)
    if not args.cpu_list:
        probs.append("--cpu-list is required (the claimed region or, on GPU, the claimed host threads; no default)")
        args.cpu_list = "0"  # placeholder for the plan only
    if not args.dry_run:
        if deadline.epoch is None:
            probs.append("--deadline-utc (or --deadline-s) is required to execute")
        if not args.no_region_lock and os.environ.get("KPF13_REGION") != args.cpu_list:
            probs.append(f"not inside a region-lock claim for {args.cpu_list!r} (KPF13_REGION="
                         f"{os.environ.get('KPF13_REGION')!r}): run through run_kpf13.sh, or pass --no-region-lock REASON")
    ctx = Ctx(args, ps, gate_argv, deadline)
    ctx.start_utc = now_iso()
    ctx.results = {}
    out = Path(args.out)
    if args.resume:
        for cfg_dir in sorted((out / "cells").glob("*")) if (out / "cells").exists() else []:
            for f in sorted(cfg_dir.glob("*.json")):
                ctx.results.setdefault(cfg_dir.name, {})[f.stem] = json.loads(f.read_text())
    plans: list[tuple[str, list[Step]]] = []
    plan_objs: list[Plan] = []
    for cfg in args.configs:
        if cfg not in CONFIGS:
            probs.append(f"unknown config {cfg!r}")
            continue
        if ps is None:
            break
        pl = Plan(ctx, cfg, ctx.results, args.resume)
        plan_objs.append(pl)
        plans.append((cfg, pl.build()))
    total = print_plan(ctx, plans, probs, notes)
    if args.dry_run:
        print("DRY RUN complete: no process was started, nothing was sent.")
        return 0
    if probs:
        print("REFUSED (preflight).")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "rules.json", {"rules": RULES, "rules_sha256": rules_sha256()})
    write_json(out / "run_meta.json", {
        "args": vars(args), "estimate_s": total, "started": now_iso(), "notes": notes,
        "rules_sha256": rules_sha256(), "rules_declared_before_first_launch": True,
        "deadline_utc": args.deadline_utc, "deadline_epoch": deadline.epoch,
        "region_lock": {"KPF13_REGION": os.environ.get("KPF13_REGION"), "skipped": args.no_region_lock},
        "device_claim": args.no_device_claim or ("mi210_0" if args.device == "gpu" else "n/a (cpu)"),
        "argv_file": str(args.argv_file), "argv_file_sha256": sha256_file(Path(args.argv_file)),
        "argv_comments": comments, "prompt_set_sha256": ps.sha256,
        "kernel_context": "P1 fork build; the test-only LLAMA_KPF_MUTATE hook is used ONLY for the `skip` launch"})

    dog = Watchdog(ctx)
    claim_cm = None

    def cleanup(*_a):
        for s in list(ctx.own_children):
            try:
                s.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf13] cleanup of own pid failed: {exc}", flush=True)

    def on_sig(signum, _f):
        cleanup()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    rc = 0
    try:
        if args.device == "gpu" and not args.no_device_claim:
            sys.path.insert(0, str(CLAIM_MOD_DIR))
            from autokernel.resource import device_claim as dc  # noqa: E402
            journal = dc.ClaimJournal(out / "claim" / "device-claim-journal.jsonl")
            try:
                claim_cm = dc.acquire_device_claim(
                    "mi210_0", purpose=f"KPF-13.1 restore-vs-recompute A/B on :{args.port}", campaign_id="KPF-13.1",
                    journal=journal, holder_label="kpf13", timeout_s=0,
                    max_hold_s=(deadline.remaining() + 600) if deadline.epoch else total + 1800)
            except dc.DeviceClaimError as exc:
                print(f"[kpf13] REFUSED: mi210_0 device claim not acquired: {exc}", flush=True)
                return 75
            claim = claim_cm.__enter__()
            write_json(out / "claim" / "device-claim-receipt.json", claim.receipt().to_dict())
        dog.start()
        for (cfg, steps), pl in zip(plans, plan_objs):
            for st in steps:
                stop_step = "SIGTERM own pid" in st.desc  # always runs: it is how the own child dies
                if not stop_step and not deadline.fits(st.est_s):
                    raise Abort(f"deadline: {fmt_s(max(0, deadline.remaining()))} left, next step est "
                                f"{fmt_s(st.est_s)} + reserve; draining at this boundary")
                t0 = time.time()
                print(f"[kpf13 {time.strftime('%H:%M:%SZ', time.gmtime())}] {st.desc} (est {fmt_s(st.est_s)})", flush=True)
                st.fn()
                print(f"[kpf13]   done in {fmt_s(time.time() - t0)}", flush=True)
    except Abort as exc:
        print(f"[kpf13] ABORT: {exc}", flush=True)
        rc = 5 if "deadline" in str(exc) else 3
    except Exception as exc:  # noqa: BLE001
        print(f"[kpf13] ERROR: {type(exc).__name__}: {exc}", flush=True)
        rc = 4
    finally:
        cleanup()
        dead = [s.stop_rec for pl in plan_objs if pl.srv for s in [pl.srv]]
        for pl in plan_objs:  # an aborted launch still leaves a (partial) record
            if pl.srv and not list((ctx.out / "records").glob(f"kpf13-{pl.cfg_name}-*")):
                try:
                    pl.finish_record()
                except Exception as exc:  # noqa: BLE001
                    print(f"[kpf13] could not write partial record for {pl.cfg_name}: {exc}", flush=True)
        try:
            cmd_analyze(args)
        except Exception as exc:  # noqa: BLE001
            print(f"[kpf13] analyze failed: {exc}", flush=True)
        dog.stop()
        if claim_cm is not None:
            try:
                claim_cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf13] claim release failed: {exc}", flush=True)
        alive = [r.get("pid") for r in dead if r and r.get("pid") and pid_alive(r["pid"])]
        if alive:
            print(f"[kpf13] WARNING: own child pid(s) still alive: {alive}", flush=True)
            rc = rc or 4
    print(f"[kpf13] exit rc={rc} (0 ok, 2 refused, 3 abort, 4 error, 5 deadline drain, 75 claim held)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
